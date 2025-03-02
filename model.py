# GodModeDariush: The ultimate transformer
# Copyright (c) 2025 hosein davod abadi farahani 

import jax
import jax.numpy as jnp
import haiku as hk
from jax import config as jax_config
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.lax import with_sharding_constraint as pjit_sharding_constraint
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset
from typing import Optional, List, Dict, Any, Tuple, Callable
from tqdm import tqdm
import functools
import logging
import optax
import numpy as np
from dataclasses import dataclass, field
import jax.tree_util as tree_util
import threading
import queue
import os
import time
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from jax.experimental import mesh_utils
import multiprocessing as mp
from collections import deque, OrderedDict
import hashlib
import shutil
import lru_cache as lru
from tensorboardX import SummaryWriter
import boto3
from google.cloud import storage

# تنظیمات JAX برای اجرای توزیع‌شده
jax_config.update("jax_spmd_mode", "allow_all")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 1. تنظیمات پیشرفته
@dataclass
class DariushConfig:
    # اندازه‌های اصلی مدل
    vocab_size: int = 262144
    emb_size: int = 16384
    num_q_heads: int = 256
    num_kv_heads: int = 32
    key_size: int = 256
    num_layers: int = 128
    num_experts: int = 128
    num_selected_experts: int = 16
    widening_factor: float = 5.0
    max_seq_len: int = 32768
    
    # تنظیمات بهینه‌سازی و آموزش
    init_scale: float = 0.005
    dropout_rate: float = 0.05
    sparse_factor: int = 8
    batch_size: int = 64
    num_micro_batches: int = 8
    learning_rate: float = 3e-5
    warmup_steps: int = 5000
    total_steps: int = 200000
    checkpoint_interval: int = 5000
    log_interval: int = 100
    
    # تنظیمات شاردینگ
    data_axis: str = "data"
    model_axis: str = "model"
    expert_axis: str = "expert"
    tensor_axis: str = "tensor"
    shard_activations: bool = True
    
    # ویژگی‌های پیشرفته
    use_swiglu: bool = True
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    
    # تنظیمات دیتالودر و توکنایزر
    cache_size: int = 20000
    num_workers: int = 16
    prefetch_size: int = 50
    
    # توکن‌های خاص
    special_tokens: Dict[str, int] = field(default_factory=lambda: {
        "[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "[CLS]": 4, "[SEP]": 5
    })

    def partition_rules(self) -> List[Tuple[Tuple[str, ...], P]]:
        """تعریف قوانین شاردینگ برای مدل"""
        return [
            (("embedding", "w"), P(None, "data", "model", "tensor")),
            (("multi_head_attention", "(query|key|value)", "w"), P("data", "model", "tensor")),
            (("multi_head_attention", "linear", "w"), P("model", "data", "tensor")),
            (("moe", "router", "w"), P("data", "expert")),
            (("moe", "expert", "w"), P("expert", "data", "model")),
            (("moe", "expert_out", "w"), P("expert", "model", "data")),
            (("rms_norm", "scale"), P(None)),
            (("output", "w"), P("model", "data", "tensor")),
            (("kv_cache", "k"), P("data", "model")),
            (("kv_cache", "v"), P("data", "model")),
        ]

    def get_mesh(self) -> jax.sharding.Mesh:
        """ایجاد مش برای شاردینگ"""
        devices = jax.devices()
        return jax.sharding.Mesh(devices, ("data", "model", "expert", "tensor"))

    def validate(self):
        """اعتبارسنجی تنظیمات"""
        assert self.num_q_heads % self.num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.batch_size % self.num_micro_batches == 0, "batch_size must be divisible by num_micro_batches"
        logger.info("Configuration validated successfully.")

config = DariushConfig()
config.validate()

# 2. توکنایزر پیشرفته
class DariushTokenizer:
    def __init__(self, languages: List[str] = ["fa", "en", "ar"]):
        """راه‌اندازی توکنایزر چندزبانه"""
        self.tokenizers: Dict[str, Tokenizer] = {lang: Tokenizer(models.BPE(unk_token="[UNK]")) for lang in languages}
        self.cache = lru.LRU(config.cache_size)
        self.languages = languages
        self.special_tokens = config.special_tokens
        self.stats = {"hits": 0, "misses": 0}

    def train(self, data_paths: Dict[str, str]):
        """آموزش توکنایزر برای هر زبان"""
        for lang in self.languages:
            logger.info(f"Training tokenizer for language: {lang}")
            if lang not in data_paths:
                raise ValueError(f"No data path provided for language: {lang}")
            dataset = load_dataset(data_paths[lang], split="train[:20%]")
            tokenizer = self.tokenizers[lang]
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=config.vocab_size,
                special_tokens=list(self.special_tokens.keys()),
                min_frequency=2,
                show_progress=True,
                continuing_subword_prefix="##"
            )
            tokenizer.train_from_iterator(dataset["text"], trainer=trainer)
            tokenizer.enable_padding(pad_id=self.special_tokens["[PAD]"], pad_token="[PAD]")
            tokenizer.save(f"dariush_tokenizer_{lang}.json")
            logger.info(f"Tokenizer for {lang} saved to dariush_tokenizer_{lang}.json")

    def encode(self, text: str, lang: str) -> List[int]:
        """رمزگذاری متن برای زبان مشخص"""
        if lang not in self.languages:
            raise ValueError(f"Unsupported language: {lang}")
        key = (lang, hashlib.sha256(text.encode()).hexdigest())
        if key in self.cache:
            self.stats["hits"] += 1
            return self.cache[key]
        tokens = self.tokenizers[lang].encode(text).ids
        self.cache[key] = tokens
        self.stats["misses"] += 1
        return tokens

    def decode(self, tokens: List[int], lang: str) -> str:
        """رمزگشایی توکن‌ها به متن"""
        if lang not in self.languages:
            raise ValueError(f"Unsupported language: {lang}")
        return self.tokenizers[lang].decode(tokens)

    def pad(self, sequences: List[List[int]], max_len: int = config.max_seq_len) -> jnp.ndarray:
        """پد کردن توالی‌ها به طول ثابت"""
        padded = []
        for seq in sequences:
            seq = seq[:max_len]
            padded_seq = seq + [self.special_tokens["[PAD]"]] * max(0, max_len - len(seq))
            padded.append(padded_seq)
        return jnp.array(padded)

    def batch_encode(self, texts: List[str], lang: str, max_len: int = config.max_seq_len) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """رمزگذاری دسته‌ای متون"""
        encoded = [self.encode(text, lang) for text in texts]
        input_ids = self.pad(encoded, max_len)
        mask = (input_ids != self.special_tokens["[PAD]"]).astype(jnp.float32)[:, None, None, :]
        return input_ids, mask

    def encode_parallel(self, texts: List[str], lang: str, num_threads: int = config.num_workers) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """رمزگذاری موازی متون"""
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            encoded = list(executor.map(lambda text: self.encode(text, lang), texts))
        return self.batch_encode([e for e in encoded], lang)

    def get_stats(self) -> Dict[str, int]:
        """دریافت آمار کش"""
        return self.stats

    def clear_cache(self):
        """پاک کردن کش"""
        self.cache.clear()
        self.stats = {"hits": 0, "misses": 0}
        logger.info("Tokenizer cache cleared.")

# 3. دیتالودر پیشرفته
class DariushDataLoader:
    def __init__(self, tokenizer: DariushTokenizer, batch_size: int, datasets: Dict[str, List[str]], 
                 num_workers: int = config.num_workers, prefetch_size: int = config.prefetch_size):
        """راه‌اندازی دیتالودر چندزبانه"""
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.datasets = datasets
        self.num_workers = num_workers
        self.prefetch_size = prefetch_size
        self.queue = mp.Queue(maxsize=prefetch_size)
        self.priority_queue = queue.PriorityQueue(maxsize=prefetch_size)
        self.total_samples = {lang: len(data) for lang, data in datasets.items()}
        self.cache = deque(maxlen=2000)
        self.cache_lock = threading.Lock()
        self.running = False
        self.languages = list(datasets.keys())

    def start(self):
        """شروع کارگرها"""
        self.running = True
        self.processes = []
        for i in range(self.num_workers):
            p = mp.Process(target=self._worker_fn, args=(i,))
            p.daemon = True
            p.start()
            self.processes.append(p)
        logger.info(f"Started {self.num_workers} data loader workers.")

    def stop(self):
        """توقف کارگرها"""
        self.running = False
        for p in self.processes:
            p.terminate()
            p.join()
        logger.info("Data loader workers stopped.")

    def _worker_fn(self, worker_id: int):
        """تابع کارگر برای بارگذاری داده‌ها"""
        while self.running:
            try:
                with self.cache_lock:
                    if self.cache and np.random.random() < 0.4:
                        batch = self.cache[np.random.randint(len(self.cache))]
                    else:
                        lang = np.random.choice(self.languages)
                        dataset = self.datasets[lang]
                        start_idx = np.random.randint(0, self.total_samples[lang] - self.batch_size)
                        batch_texts = dataset[start_idx:start_idx + self.batch_size]
                        input_ids, mask = self.tokenizer.batch_encode(batch_texts, lang)
                        batch = {
                            "input_ids": input_ids,
                            "labels": input_ids,
                            "mask": mask,
                            "lang": lang
                        }
                        self.cache.append(batch)
                priority = np.random.random()
                self.priority_queue.put((priority, batch))
                self.queue.put(batch, timeout=10)
            except queue.Full:
                if not self.running:
                    break
                time.sleep(1)
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    def __iter__(self):
        return self

    def __next__(self):
        if not self.running:
            raise StopIteration
        return self.queue.get()

    def prefetch(self):
        """پیش‌بارگذاری داده‌ها"""
        logger.info("Starting data prefetching...")
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._worker_fn, i) for i in range(self.num_workers)]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Prefetch error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """دریافت آمار دیتالودر"""
        return {
            "queue_size": self.queue.qsize(),
            "priority_queue_size": self.priority_queue.qsize(),
            "cache_size": len(self.cache),
            "total_samples": self.total_samples
        }

    def clear_cache(self):
        """پاک کردن کش دیتالودر"""
        with self.cache_lock:
            self.cache.clear()
        logger.info("Data loader cache cleared.")

# 4. نرمال‌سازی RMS پیشرفته
class DariushRMSNorm(hk.Module):
    def __init__(self, emb_size: int, eps: float = 1e-6, name: str = "rms_norm"):
        """راه‌اندازی نرمال‌سازی RMS"""
        super().__init__(name=name)
        self.emb_size = emb_size
        self.eps = eps

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """اعمال نرمال‌سازی RMS"""
        scale = hk.get_parameter("scale", [self.emb_size], init=jnp.ones)
        scale = pjit_sharding_constraint(scale, P(None))
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps) * scale
        return normed.astype(jnp.bfloat16)

    def reset(self):
        """بازنشانی پارامترها"""
        hk.set_parameter("scale", jnp.ones(self.emb_size))
        logger.info(f"RMSNorm {self.name} reset.")

# 5. تعبیه موقعیت چرخشی پیشرفته
class DariushRotaryEmbedding(hk.Module):
    def __init__(self, dim: int, base: int = 10000, max_seq_len: int = config.max_seq_len, name: str = "rotary_emb"):
        """راه‌اندازی تعبیه موقعیت چرخشی"""
        super().__init__(name=name)
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
        self.register_buffer("inv_freq", inv_freq)

    def __call__(self, x: jnp.ndarray, offset: int = 0) -> jnp.ndarray:
        """اعمال تعبیه موقعیت چرخشی"""
        seq_len = x.shape[1]
        pos = jnp.arange(seq_len, dtype=jnp.float32) + offset
        angles = pos[:, None] * self.inv_freq[None, :]
        sin_val = jnp.sin(angles)
        cos_val = jnp.cos(angles)
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        x_rot = jnp.concatenate([-x2, x1], axis=-1)
        return x * cos_val + x_rot * sin_val

# 6. SwiGLU پیشرفته
class DariushSwiGLU(hk.Module):
    def __init__(self, hidden_size: int, name: str = "swiglu"):
        """راه‌اندازی فعال‌سازی SwiGLU"""
        super().__init__(name=name)
        self.hidden_size = hidden_size

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """اعمال فعال‌سازی SwiGLU"""
        w1 = hk.Linear(self.hidden_size, name="w1", w_init=hk.initializers.TruncatedNormal(stddev=0.02))
        w2 = hk.Linear(self.hidden_size, name="w2", w_init=hk.initializers.TruncatedNormal(stddev=0.02))
        return jax.nn.silu(w1(x)) * w2(x)

# 7. Flash Attention پیشرفته
class DariushFlashAttention(hk.Module):
    def __init__(self, num_heads: int, key_size: int, block_size: int = 128, name: str = "flash_attention"):
        """راه‌اندازی توجه سریع"""
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.block_size = block_size

    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """اعمال توجه سریع"""
        batch, seq_len, _ = q.shape
        q = q.reshape(batch, seq_len, self.num_heads, self.key_size)
        k = k.reshape(batch, seq_len, self.num_heads, self.key_size)
        v = v.reshape(batch, seq_len, self.num_heads, self.key_size)

        def block_attention(q_block, k_block, v_block, mask_block):
            attn_logits = jnp.einsum("...hd,...kd->...hk", q_block, k_block) / jnp.sqrt(self.key_size)
            if mask_block is not None:
                attn_logits = jnp.where(mask_block, attn_logits, -1e30)
            attn_weights = jax.nn.softmax(attn_logits)
            return jnp.einsum("...hk,...kd->...hd", attn_weights, v_block)

        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        q_blocks = q.reshape(batch, num_blocks, self.block_size, self.num_heads, self.key_size)
        k_blocks = k.reshape(batch, num_blocks, self.block_size, self.num_heads, self.key_size)
        v_blocks = v.reshape(batch, num_blocks, self.block_size, self.num_heads, self.key_size)
        mask_blocks = mask.reshape(batch, 1, num_blocks, self.block_size) if mask is not None else None

        @functools.partial(shard_map, mesh=config.get_mesh(), 
                           in_specs=(P("data", None, "model", "tensor"), P("data", None, "model", "tensor"), 
                                     P("data", None, "model", "tensor"), P("data", None)),
                           out_specs=P("data", "model", "tensor"), check_rep=False)
        def sharded_block_attention(qb, kb, vb, mb):
            return block_attention(qb, kb, vb, mb)

        outputs = jax.vmap(sharded_block_attention)(q_blocks, k_blocks, v_blocks, mask_blocks)
        return outputs.reshape(batch, seq_len, self.num_heads * self.key_size)

# 8. توجه پراکنده پیشرفته
class DariushSparseAttention(hk.Module):
    def __init__(self, num_heads: int, key_size: int, sparse_factor: int = config.sparse_factor, name: str = "sparse_attention"):
        """راه‌اندازی توجه پراکنده"""
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.sparse_factor = sparse_factor

    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """اعمال توجه پراکنده"""
        batch, seq_len, _ = q.shape
        q = q.reshape(batch, seq_len, self.num_heads, self.key_size)
        k = k.reshape(batch, seq_len, self.num_heads, self.key_size)
        v = v.reshape(batch, seq_len, self.num_heads, self.key_size)

        sparse_seq_len = seq_len // self.sparse_factor
        q_sparse = q[:, ::self.sparse_factor, :, :]
        k_sparse = k[:, ::self.sparse_factor, :, :]
        v_sparse = v[:, ::self.sparse_factor, :, :]

        attn_logits = jnp.einsum("...qhd,...khd->...hqk", q_sparse, k_sparse) / jnp.sqrt(self.key_size)
        if mask is not None:
            mask_sparse = mask[:, :, ::self.sparse_factor]
            attn_logits = jnp.where(mask_sparse, attn_logits, -1e30)
        attn_weights = jax.nn.softmax(attn_logits)
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, v_sparse)
        return attn_output.reshape(batch, sparse_seq_len, self.num_heads * self.key_size)

# 9. Mixture of Experts پیشرفته
class DariushRouter(hk.Module):
    def __init__(self, num_experts: int, num_selected_experts: int, name: str = "router"):
        """راه‌اندازی روتر MoE"""
        super().__init__(name=name)
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts

    def __call__(self, inputs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """انتخاب کارشناسان با روتر"""
        w = hk.get_parameter("w", [inputs.shape[-1], self.num_experts], 
                            init=hk.initializers.TruncatedNormal(stddev=0.02))
        w = pjit_sharding_constraint(w, P("data", "expert"))
        logits = jnp.dot(inputs.astype(jnp.float32), w)
        noise = jax.random.gumbel(jax.random.PRNGKey(0), logits.shape) * 0.05
        probs = jax.nn.softmax(logits + noise)
        gates, indices = jax.lax.top_k(probs, self.num_selected_experts)
        return gates, indices

class DariushMoELayer(hk.Module):
    def __init__(self, config: DariushConfig, mesh: jax.sharding.Mesh, name: str = "moe"):
        """راه‌اندازی لایه MoE"""
        super().__init__(name=name)
        self.config = config
        self.mesh = mesh
        self.router = DariushRouter(config.num_experts, config.num_selected_experts)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """اعمال لایه MoE"""
        gates, indices = self.router(inputs)
        expert_outputs = []

        def expert_fn(x: jnp.ndarray) -> jnp.ndarray:
            """تابع کارشناس جداگانه"""
            w = hk.Linear(int(self.config.widening_factor * self.config.emb_size), name="expert",
                         w_init=hk.initializers.TruncatedNormal(stddev=0.02))
            w_out = hk.Linear(self.config.emb_size, name="expert_out",
                            w_init=hk.initializers.TruncatedNormal(stddev=0.02))
            if self.config.use_swiglu:
                return w_out(DariushSwiGLU(self.config.emb_size)(x))
            return w_out(jax.nn.gelu(w(x)))

        for _ in range(self.config.num_experts):
            expert_outputs.append(expert_fn(inputs))

        expert_outputs = jnp.stack(expert_outputs, axis=1)  # [batch, experts, seq, emb]

        @functools.partial(shard_map, mesh=self.mesh, 
                           in_specs=(P("data", None, "expert"), P("expert", "data", "model", "tensor")),
                           out_specs=P("data", "model", "tensor"), check_rep=False)
        def compute_expert_output(inputs, expert_outs):
            return jax.vmap(lambda x, idx: x[idx])(inputs, indices)

        selected_outputs = compute_expert_output(inputs, expert_outputs)
        return (selected_outputs * gates[..., None]).sum(axis=1)

# 10. توجه چندسر پیشرفته
class DariushMultiHeadAttention(hk.Module):
    def __init__(self, config: DariushConfig, name: str = "multi_head_attention"):
        """راه‌اندازی توجه چندسر"""
        super().__init__(name=name)
        self.config = config
        self.rotary = DariushRotaryEmbedding(config.key_size)

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, 
                 kv_cache: Optional[Dict] = None) -> Tuple[jnp.ndarray, Dict]:
        """اعمال توجه چندسر"""
        q_w = hk.Linear(self.config.num_q_heads * self.config.key_size, name="query",
                       w_init=hk.initializers.TruncatedNormal(stddev=0.02))
        k_w = hk.Linear(self.config.num_kv_heads * self.config.key_size, name="key",
                       w_init=hk.initializers.TruncatedNormal(stddev=0.02))
        v_w = hk.Linear(self.config.num_kv_heads * self.config.key_size, name="value",
                       w_init=hk.initializers.TruncatedNormal(stddev=0.02))
        out_w = hk.Linear(self.config.emb_size, name="linear",
                         w_init=hk.initializers.TruncatedNormal(stddev=0.02))

        q = q_w(x).reshape(*x.shape[:-1], self.config.num_q_heads, self.config.key_size)
        k = k_w(x).reshape(*x.shape[:-1], self.config.num_kv_heads, self.config.key_size)
        v = v_w(x).reshape(*x.shape[:-1], self.config.num_kv_heads, self.config.key_size)

        q = self.rotary(q)
        k = self.rotary(k)

        if kv_cache is not None:
            k = kv_cache["k"]
            v = kv_cache["v"]

        if self.config.use_flash_attention:
            flash_attn = DariushFlashAttention(self.config.num_q_heads, self.config.key_size)
            attn_output = flash_attn(q, k, v, mask)
        else:
            sparse_attn = DariushSparseAttention(self.config.num_q_heads, self.config.key_size, self.config.sparse_factor)
            attn_output = sparse_attn(q, k, v, mask)

        return out_w(attn_output), {"k": k, "v": v}

# 11. لایه پیشرفته
class DariushLayer(hk.Module):
    def __init__(self, config: DariushConfig, mesh: jax.sharding.Mesh, layer_idx: int, name: str = "dariush_layer"):
        """راه‌اندازی لایه ترانسفورمر"""
        super().__init__(name=f"{name}_{layer_idx}")
        self.config = config
        self.mesh = mesh
        self.layer_idx = layer_idx
        self.attn = DariushMultiHeadAttention(config)
        self.moe = DariushMoELayer(config, mesh)
        self.norm1 = DariushRMSNorm(config.emb_size)
        self.norm2 = DariushRMSNorm(config.emb_size)
        self.dropout = hk.dropout if config.dropout_rate > 0 else lambda x: x

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, 
                 kv_cache: Optional[Dict] = None) -> Tuple[jnp.ndarray, Dict]:
        """اعمال لایه ترانسفورمر"""
        if self.config.gradient_checkpointing:
            attn_out, new_cache = hk.checkpoint(lambda x: self.attn(self.norm1(x), mask, kv_cache))(x)
        else:
            attn_out, new_cache = self.attn(self.norm1(x), mask, kv_cache)
        x = x + self.dropout(attn_out, rate=self.config.dropout_rate, salt=jax.random.PRNGKey(self.layer_idx))
        moe_out = self.moe(self.norm2(x))
        x = x + self.dropout(moe_out, rate=self.config.dropout_rate, salt=jax.random.PRNGKey(self.layer_idx + 1))
        return x, new_cache

# 12. مدل اصلی
class GodModeDariush(hk.Module):
    def __init__(self, config: DariushConfig, mesh: jax.sharding.Mesh, name: str = "godmode_dariush"):
        """راه‌اندازی مدل اصلی"""
        super().__init__(name=name)
        self.config = config
        self.mesh = mesh
        self.embedding = hk.Embed(config.vocab_size, config.emb_size, name="embedding",
                                 w_init=hk.initializers.TruncatedNormal(stddev=config.init_scale))
        self.layers = [DariushLayer(config, mesh, i) for i in range(config.num_layers)]
        self.norm = DariushRMSNorm(config.emb_size)
        self.output = hk.Linear(config.vocab_size, name="output",
                               w_init=hk.initializers.TruncatedNormal(stddev=config.init_scale))

    def __call__(self, input_ids: jnp.ndarray, mask: Optional[jnp.ndarray] = None, 
                 kv_cache: Optional[List[Dict]] = None) -> Tuple[jnp.ndarray, List[Dict]]:
        """اعمال مدل اصلی"""
        x = self.embedding(input_ids)
        x = pjit_sharding_constraint(x, P(self.config.data_axis, None, self.config.model_axis, self.config.tensor_axis))
        new_kv_cache = [] if kv_cache is None else kv_cache

        for i, layer in enumerate(self.layers):
            x, layer_cache = layer(x, mask, new_kv_cache[i] if kv_cache else None)
            new_kv_cache.append(layer_cache)

        x = self.norm(x)
        logits = self.output(x)
        return logits, new_kv_cache

    def init_memory(self, batch_size: int, seq_len: int) -> List[Dict]:
        """راه‌اندازی حافظه KV"""
        return [{"k": jnp.zeros((batch_size, seq_len, self.config.num_kv_heads, self.config.key_size), dtype=jnp.bfloat16),
                 "v": jnp.zeros((batch_size, seq_len, self.config.num_kv_heads, self.config.key_size), dtype=jnp.bfloat16)}
                for _ in range(self.config.num_layers)]

    def generate(self, input_ids: jnp.ndarray, max_len: int = 200, temperature: float = 0.7, 
                 top_k: int = 40, top_p: float = 0.9, beam_width: int = 5, repetition_penalty: float = 1.2) -> jnp.ndarray:
        """تولید متن با Beam Search و Nucleus Sampling"""
        kv_cache = self.init_memory(input_ids.shape[0], input_ids.shape[1])
        beams = [(input_ids, 0.0, kv_cache)]  # (sequence, score, cache)
        seen_tokens = set()

        for step in range(max_len):
            new_beams = []
            for seq, score, cache in beams:
                logits, new_cache = self(seq, kv_cache=cache)
                next_logits = logits[:, -1, :] / temperature
                
                # اعمال جریمه تکرار
                for token in seen_tokens:
                    next_logits = jnp.where(next_logits == token, next_logits / repetition_penalty, next_logits)
                
                top_k_logits, top_k_tokens = jax.lax.top_k(next_logits, top_k)
                probs = jax.nn.softmax(top_k_logits)
                
                sorted_probs = jnp.sort(probs, axis=-1, descending=True)
                cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
                mask = cumulative_probs <= top_p
                filtered_probs = jnp.where(mask, probs, 0.0)
                filtered_probs /= jnp.sum(filtered_probs, axis=-1, keepdims=True)
                
                for i in range(top_k):
                    if filtered_probs[:, i] > 0:
                        new_token = top_k_tokens[:, i:i+1]
                        new_seq = jnp.concatenate([seq, new_token], axis=1)
                        new_score = score + jnp.log(filtered_probs[:, i])
                        new_beams.append((new_seq, new_score, new_cache))
                        seen_tokens.add(new_token.item())

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            if jnp.all(beams[0][0][:, -1] == self.config.special_tokens["[EOS]"]):
                break

        return beams[0][0]

    def evaluate(self, input_ids: jnp.ndarray, labels: jnp.ndarray) -> float:
        """ارزیابی مدل"""
        logits, _ = self(input_ids)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return jnp.mean(loss)

# 13. مدیریت چک‌پوینت پیشرفته
class DariushCheckpointManager:
    def __init__(self, save_dir: str = "dariush_checkpoints", cloud_storage: str = "s3", max_checkpoints: int = 10):
        """راه‌اندازی مدیریت چک‌پوینت"""
        self.save_dir = save_dir
        self.cloud_storage = cloud_storage
        self.max_checkpoints = max_checkpoints
        os.makedirs(save_dir, exist_ok=True)
        if cloud_storage == "s3":
            self.s3 = boto3.client('s3')
        elif cloud_storage == "gcs":
            self.gcs = storage.Client()
        else:
            raise ValueError(f"Unsupported cloud storage: {cloud_storage}")
        self.checkpoints = OrderedDict()
        self.lock = threading.Lock()

    def save(self, params: Any, step: int, metadata: Dict = None):
        """ذخیره چک‌پوینت به صورت محلی و ابری"""
        with self.lock:
            path = os.path.join(self.save_dir, f"checkpoint_step_{step}.pkl")
            flat_params, tree_def = jax.tree_util.tree_flatten(params)
            checkpoint_data = {
                "params": flat_params,
                "tree_def": tree_def,
                "metadata": metadata or {"step": step, "timestamp": time.time()}
            }
            with open(path, "wb") as f:
                pickle.dump(checkpoint_data, f)
            
            if self.cloud_storage == "s3":
                self.s3.upload_file(path, "dariush-bucket", f"checkpoints/checkpoint_step_{step}.pkl")
            elif self.cloud_storage == "gcs":
                bucket = self.gcs.bucket("dariush-bucket")
                blob = bucket.blob(f"checkpoints/checkpoint_step_{step}.pkl")
                blob.upload_from_filename(path)
            
            self.checkpoints[step] = path
            if len(self.checkpoints) > self.max_checkpoints:
                oldest_step = min(self.checkpoints.keys())
                os.remove(self.checkpoints.pop(oldest_step))
            logger.info(f"Checkpoint saved at step {step} to {path}")

    def load(self, step: int) -> Tuple[Any, Dict]:
        """بارگذاری چک‌پوینت از محلی یا ابری"""
        with self.lock:
            path = os.path.join(self.save_dir, f"checkpoint_step_{step}.pkl")
            if not os.path.exists(path):
                if self.cloud_storage == "s3":
                    self.s3.download_file("dariush-bucket", f"checkpoints/checkpoint_step_{step}.pkl", path)
                elif self.cloud_storage == "gcs":
                    bucket = self.gcs.bucket("dariush-bucket")
                    blob = bucket.blob(f"checkpoints/checkpoint_step_{step}.pkl")
                    blob.download_to_filename(path)
            with open(path, "rb") as f:
                checkpoint_data = pickle.load(f)
            params = jax.tree_util.tree_unflatten(checkpoint_data["tree_def"], checkpoint_data["params"])
            return params, checkpoint_data["metadata"]

    def get_latest_checkpoint(self) -> Optional[int]:
        """دریافت آخرین گام چک‌پوینت"""
        with self.lock:
            return max(self.checkpoints.keys()) if self.checkpoints else None

    def cleanup(self):
        """پاکسازی چک‌پوینت‌ها"""
        with self.lock:
            for path in self.checkpoints.values():
                if os.path.exists(path):
                    os.remove(path)
            self.checkpoints.clear()
            logger.info("All checkpoints cleaned up.")

# 14. مانیتورینگ پیشرفته
class DariushMonitor:
    def __init__(self, log_dir: str = "dariush_logs"):
        """راه‌اندازی مانیتورینگ با TensorBoard"""
        self.writer = SummaryWriter(log_dir)
        self.metrics = {
            "loss": [],
            "grad_norm": [],
            "learning_rate": [],
            "step": [],
            "time": []
        }
        self.start_time = time.time()
        self.lock = threading.Lock()

    def log(self, step: int, loss: float, grad_norm: float, learning_rate: float):
        """ثبت متریک‌ها"""
        with self.lock:
            elapsed = time.time() - self.start_time
            self.writer.add_scalar("Loss", loss, step)
            self.writer.add_scalar("Gradient Norm", grad_norm, step)
            self.writer.add_scalar("Learning Rate", learning_rate, step)
            self.writer.add_scalar("Time", elapsed, step)
            self.metrics["loss"].append(loss)
            self.metrics["grad_norm"].append(grad_norm)
            self.metrics["learning_rate"].append(learning_rate)
            self.metrics["step"].append(step)
            self.metrics["time"].append(elapsed)
            logger.info(f"Step {step} | Loss: {loss:.4f} | Grad Norm: {grad_norm:.4f} | LR: {learning_rate:.6f} | Time: {elapsed:.2f}s")

    def plot(self, metric: str):
        """رسم نمودار برای متریک مشخص"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics["step"], self.metrics[metric], label=metric.capitalize())
        plt.xlabel("Step")
        plt.ylabel(metric.capitalize())
        plt.title(f"{metric.capitalize()} Over Training Steps")
        plt.legend()
        plt.grid(True)
        plot_path = f"{metric}_plot.png"
        plt.savefig(plot_path)
        self.writer.add_image(f"{metric} Plot", plt.imread(plot_path), global_step=max(self.metrics["step"]))
        plt.close()

    def save_metrics(self, file_path: str = "training_metrics.json"):
        """ذخیره متریک‌ها در فایل"""
        with self.lock:
            with open(file_path, "w") as f:
                json.dump(self.metrics, f, indent=4)
            logger.info(f"Metrics saved to {file_path}")

    def summary(self):
        """خلاصه‌سازی متریک‌ها"""
        with self.lock:
            avg_loss = np.mean(self.metrics["loss"])
            avg_grad_norm = np.mean(self.metrics["grad_norm"])
            total_time = sum(self.metrics["time"])
            logger.info(f"Training Summary: Avg Loss = {avg_loss:.4f}, Avg Grad Norm = {avg_grad_norm:.4f}, Total Time = {total_time:.2f}s")
            for metric in ["loss", "grad_norm", "learning_rate"]:
                self.plot(metric)
            self.save_metrics()

# 15. بهینه‌ساز پیشرفته
class DariushOptimizer:
    def __init__(self, config: DariushConfig):
        """راه‌اندازی بهینه‌ساز پیشرفته"""
        self.config = config
        self.schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=config.warmup_steps,
            decay_steps=config.total_steps - config.warmup_steps,
            end_value=config.learning_rate * 0.1
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=self.schedule, b1=0.9, b2=0.95, weight_decay=0.01),
            optax.scale_by_schedule(lambda step: 1.0)
        )

    def init(self, params: Any) -> Any:
        """راه‌اندازی حالت بهینه‌ساز"""
        return self.optimizer.init(params)

    def update(self, grads: Any, opt_state: Any, params: Any) -> Tuple[Any, Any]:
        """به‌روزرسانی پارامترها"""
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    def get_learning_rate(self, step: int) -> float:
        """دریافت نرخ یادگیری برای گام مشخص"""
        return self.schedule(step)

# 16. آموزش پیشرفته
def train_dariush(model: GodModeDariush, tokenizer: DariushTokenizer, mesh: jax.sharding.Mesh, 
                         config: DariushConfig, datasets: Dict[str, List[str]]):
    """آموزش مدل"""
    dataloader = DariushDataLoader(tokenizer, config.batch_size, datasets)
    dataloader.start()
    
    optimizer = DariushOptimizer(config)
    
    @hk.transform
    def forward_fn(input_ids: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """تابع جلو برای محاسبه لاجیت‌ها"""
        logits, _ = model(input_ids, mask)
        return logits

    def loss_fn(params: Any, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """محاسبه تابع خسارت"""
        logits = forward_fn.apply(params, None, batch["input_ids"], batch["mask"])
        labels = batch["labels"]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return jnp.mean(loss)

    params = forward_fn.init(jax.random.PRNGKey(42), jnp.ones((1, config.max_seq_len), dtype=jnp.int32))
    opt_state = optimizer.init(params)

    @jax.jit
    def update_step(params: Any, opt_state: Any, batch: Dict[str, jnp.ndarray]) -> Tuple[Any, Any, jnp.ndarray, float]:
        """گام به‌روزرسانی با گرادیان‌ها"""
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        grad_norm = optax.global_norm(grads)
        new_params, new_opt_state = optimizer.update(grads, opt_state, params)
        return new_params, new_opt_state, loss, grad_norm

    checkpoint_mgr = DariushCheckpointManager()
    monitor = DariushMonitor()
    step = 0

    latest_step = checkpoint_mgr.get_latest_checkpoint()
    if latest_step is not None:
        params, metadata = checkpoint_mgr.load(latest_step)
        step = latest_step + 1
        logger.info(f"Resumed training from step {step} with metadata: {metadata}")

    for batch in tqdm(dataloader, total=config.total_steps, desc="Training Dariush"):
        if step >= config.total_steps:
            break

        # تقسیم به میکروبچ‌ها برای Gradient Accumulation
        micro_batches = [
            {
                k: v[i * config.batch_size // config.num_micro_batches:(i + 1) * config.batch_size // config.num_micro_batches]
                for k, v in batch.items()
            }
            for i in range(config.num_micro_batches)
        ]
        accumulated_grads = None
        total_loss = 0.0

        for micro_batch in micro_batches:
            params, opt_state, micro_loss, micro_grad_norm = update_step(params, opt_state, micro_batch)
            total_loss += micro_loss
            if accumulated_grads is None:
                accumulated_grads = micro_grad_norm
            else:
                accumulated_grads += micro_grad_norm

        avg_loss = total_loss / config.num_micro_batches
        avg_grad_norm = accumulated_grads / config.num_micro_batches
        lr = optimizer.get_learning_rate(step)

        if step % config.log_interval == 0:
            monitor.log(step, avg_loss, avg_grad_norm, lr)

        if step % config.checkpoint_interval == 0 and step > 0:
            checkpoint_mgr.save(params, step, {"loss": float(avg_loss), "grad_norm": float(avg_grad_norm)})

        step += 1

    dataloader.stop()
    monitor.summary()
    checkpoint_mgr.save(params, config.total_steps, {"final_step": step})
    return params

# 17. تست و اعتبارسنجی
def validate_dariush(model: GodModeDariush, tokenizer: DariushTokenizer, 
                            test_texts: List[str], lang: str) -> float:
    """اعتبارسنجی مدل"""
    input_ids, mask = tokenizer.batch_encode(test_texts, lang)
    labels = input_ids
    loss = model.evaluate(input_ids, labels)
    logger.info(f"Validation Loss for {lang}: {loss:.4f}")
    return loss

def generate_dariush_samples(model: GodModeDariush, tokenizer: DariushTokenizer, prompts: List[str], 
                     lang: str, num_samples: int = 5) -> List[str]:
    """تولید نمونه‌های متنی"""
    samples = []
    for prompt in prompts[:num_samples]:
        input_ids, _ = tokenizer.batch_encode([prompt], lang)
        generated = model.generate(input_ids)
        decoded = tokenizer.decode(generated[0], lang)
        samples.append(decoded)
        logger.info(f"Generated for {lang}: {decoded}")
    return samples

# 18. اجرا
if __name__ == "__main__":
    # تنظیمات اولیه
    config = DariushConfig()
    config.validate()
    mesh = config.get_mesh()

    # آماده‌سازی توکنایزر
    tokenizer = DariushTokenizer()
    data_paths = {
        "fa": "oscar_fa",
        "en": "oscar_en",
        "ar": "oscar_ar"
    }
    tokenizer.train(data_paths)

    # آماده‌سازی دیتاست‌ها
    datasets = {
        "fa": load_dataset("oscar", "unshuffled_deduplicated_fa", split="train[:20%]")["text"],
        "en": load_dataset("oscar", "unshuffled_deduplicated_en", split="train[:20%]")["text"],
        "ar": load_dataset("oscar", "unshuffled_deduplicated_ar", split="train[:20%]")["text"]
    }

    # راه‌اندازی و آموزش مدل
    with mesh:
        model = GodModeDariush(config, mesh)
        params = train_dariush(model, tokenizer, mesh, config, datasets)

        # تست و اعتبارسنجی
        test_texts = [
            "جهان از نگاه من یک راز بزرگ است",
            "زندگی پر از شگفتی است",
            "آینده در دستان ماست",
            "علم کلید پیشرفت است",
            "هنر زبان احساسات است"
        ]
        validate_dariush(model, tokenizer, test_texts, "fa")
        samples = generate_dariush_samples(model, tokenizer, test_texts, "fa")
        for i, sample in enumerate(samples):
            print(f"Sample {i+1}: {sample}")

# این کد از سورس‌های زیر الهام گرفته شده:
# - DariushGPT (Copyright (c) 2025 hosein davod abadi farahani)
# - xAI Transformer (Copyright 2024 X.AI Corp., Apache License 2.0)
# - الهام از LLaMA, Mixtral, GPT-4, Grok و تکنیک‌های پیشرفته 2025
