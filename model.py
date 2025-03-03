# GodModeDariush: The ultimate transformer, expanded to 1000 lines
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
import seaborn as sns
from jax.experimental import mesh_utils
import multiprocessing as mp
from collections import deque, OrderedDict
import hashlib
import shutil
import lru_cache as lru
from tensorboardX import SummaryWriter
import boto3
from google.cloud import storage
import bottle
from bottle import Bottle, request, response
import wandb

# تنظیمات JAX برای اجرای توزیع‌شده
jax_config.update("jax_spmd_mode", "allow_all")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 1. تنظیمات پیشرفته
@dataclass
class DariushConfig:
    """
    تنظیمات اصلی مدل GodModeDariush با قابلیت‌های پیشرفته و مقیاس‌پذیری پویا
    این کلاس همه پارامترهای لازم برای مدل رو تعریف می‌کنه
    """
    # اندازه‌های اصلی مدل
    vocab_size: int = 50000       # اندازه واژگان برای چندزبانگی
    emb_size: int = 8192          # اندازه تعبیه برای ظرفیت بالا
    num_q_heads: int = 128        # هدهای کوئری برای Grouped-Query Attention
    num_kv_heads: int = 16        # هدهای کلید/مقدار
    key_size: int = 128           # اندازه کلید برای دقت بالا
    num_layers: int = 128         # تعداد لایه‌ها برای عمق معماری
    num_experts: int = 256        # تعداد کارشناسان در Mixture of Experts
    num_selected_experts: int = 16 # تعداد کارشناسان انتخاب‌شده در MoE
    widening_factor: float = 4.0  # ضریب گسترش برای MoE
    max_seq_len: int = 8192       # حداکثر طول دنباله برای پردازش
    
    # تنظیمات بهینه‌سازی و آموزش
    init_scale: float = 0.02      # مقیاس اولیه برای پایداری وزن‌ها
    dropout_rate: float = 0.1     # نرخ Dropout برای تعمیم بهتر
    sparse_factor: int = 8        # فاکتور پراکندگی برای بهینه‌سازی توجه
    batch_size: int = 64          # اندازه دسته برای آموزش توزیع‌شده
    num_micro_batches: int = 8    # تعداد میکروبچ‌ها برای Gradient Accumulation
    learning_rate: float = 1e-5   # نرخ یادگیری برای آموزش پایدار
    warmup_steps: int = 1000      # گام‌های گرم کردن برای بهینه‌سازی
    total_steps: int = 200000     # کل گام‌های آموزش برای یادگیری کامل
    checkpoint_interval: int = 5000 # فاصله ذخیره‌سازی چک‌پوینت‌ها
    log_interval: int = 100       # فاصله ثبت متریک‌ها
    
    # تنظیمات شاردینگ
    data_axis: str = "data"       # محور شاردینگ داده
    model_axis: str = "model"     # محور شاردینگ مدل
    expert_axis: str = "expert"   # محور شاردینگ کارشناسان
    tensor_axis: str = "tensor"   # محور شاردینگ تنسورها
    shard_activations: bool = True # فعال‌سازی شاردینگ برای فعال‌سازی‌ها
    
    # ویژگی‌های پیشرفته
    use_swiglu: bool = True       # استفاده از SwiGLU برای فعال‌سازی
    use_flash_attention: bool = True # استفاده از Flash Attention برای کارایی
    gradient_checkpointing: bool = True # استفاده از Gradient Checkpointing برای کاهش حافظه
    use_speculative_decoding: bool = True # استفاده از Speculative Decoding برای تولید سریع‌تر
    use_dynamic_sparsity: bool = True # استفاده از توجه پراکنده پویا
    use_adversarial_training: bool = True # آموزش Adversarial برای پایداری
    use_knowledge_distillation: bool = True # Knowledge Distillation برای بهینه‌سازی
    use_vision: bool = True       # پشتیبانی از پردازش تصویر
    use_audio: bool = True        # پشتیبانی از پردازش صوت
    
    # تنظیمات دیتالودر و توکنایزر
    cache_size: int = 10000       # اندازه کش برای بهینه‌سازی
    num_workers: int = 16         # تعداد کارگرها برای پردازش موازی
    prefetch_size: int = 50       # اندازه پیش‌بارگذاری داده‌ها
    
    # توکن‌های خاص
    special_tokens: Dict[str, int] = field(default_factory=lambda: {
        "[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "[CLS]": 4, "[SEP]": 5
    })

    def partition_rules(self) -> List[Tuple[Tuple[str, ...], P]]:
        """تعریف قوانین شاردینگ برای اجزای مختلف مدل"""
        return [
            (("embedding", "w"), P(None, "data", "model", "tensor")),
            (("multi_head_attention", "(query|key|value)", "w"), P("data", "model", "tensor")),
            (("multi_head_attention", "linear", "w"), P("model", "data", "tensor")),
            (("moe", "router", "w"), P("data", "expert")),
            (("moe", "expert", "w"), P("expert", "data", "model", "tensor")),
            (("moe", "expert_out", "w"), P("expert", "model", "data", "tensor")),
            (("rms_norm", "scale"), P(None)),
            (("output", "w"), P("model", "data", "tensor")),
            (("kv_cache", "k"), P("data", "model")),
            (("kv_cache", "v"), P("data", "model")),
            (("vision_transformer", "patch_embedding", "w"), P("data", "model")),
            (("whisper_encoder", "conv", "w"), P("data", "model"))
        ]

    def get_mesh(self) -> jax.sharding.Mesh:
        """ایجاد مش شاردینگ با تطبیق دستگاه‌های موجود"""
        devices = jax.devices()
        num_devices = len(devices)
        if num_devices < 4:
            logger.warning(f"تعداد دستگاه‌ها ({num_devices}) کمتر از محورهاست (۴). مش تنظیم می‌شه.")
        return jax.sharding.Mesh(devices, ("data", "model", "expert", "tensor"))

    def validate(self):
        """اعتبارسنجی تنظیمات برای اطمینان از سازگاری"""
        assert self.num_q_heads % self.num_kv_heads == 0, "num_q_heads باید مضرب num_kv_heads باشه"
        assert self.max_seq_len > 0, "max_seq_len باید مثبت باشه"
        assert self.batch_size % self.num_micro_batches == 0, "batch_size باید مضرب num_micro_batches باشه"
        assert self.num_experts >= self.num_selected_experts, "num_experts باید بیشتر یا مساوی num_selected_experts باشه"
        logger.info("تنظیمات با موفقیت اعتبارسنجی شد.")

config = DariushConfig()
config.validate()

# 2. توکنایزر پیشرفته
class DariushTokenizer:
    """
    توکنایزر چندزبانه با کش LRU، پیش‌پردازش پیشرفته و مدیریت خطا
    این کلاس برای رمزگذاری و رمزگشایی متون در زبان‌های مختلف استفاده می‌شه
    """
    def __init__(self, languages: List[str] = ["fa", "en", "ar"]):
        self.tokenizers: Dict[str, Tokenizer] = {lang: Tokenizer(models.BPE(unk_token="[UNK]")) for lang in languages}
        self.cache = lru.LRU(config.cache_size)
        self.languages = languages
        self.special_tokens = config.special_tokens
        self.stats = {"hits": 0, "misses": 0, "augmentations": 0, "processed_texts": 0}

    def train(self, data_paths: Dict[str, str]):
        """آموزش توکنایزر برای هر زبان با استفاده از دیتاست مشخص"""
        for lang in self.languages:
            logger.info(f"شروع آموزش توکنایزر برای زبان: {lang}")
            if lang not in data_paths:
                raise ValueError(f"مسیر داده برای زبان {lang} پیدا نشد")
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
            tokenizer.enable_truncation(max_length=config.max_seq_len)
            tokenizer.save(f"dariush_tokenizer_{lang}.json")
            logger.info(f"توکنایزر برای {lang} با موفقیت ذخیره شد در dariush_tokenizer_{lang}.json")

    def preprocess_text(self, text: str, lang: str) -> str:
        """پیش‌پردازش متن با مدیریت خطا و تمیز کردن داده"""
        if text is None:
            raise ValueError("متن ورودی نمی‌تونه None باشه")
        text = text.strip().lower()
        if lang == "fa":
            text = text.replace("ي", "ی").replace("ك", "ک").replace("ۀ", "ه")
        elif lang == "ar":
            text = text.replace("أ", "ا").replace("إ", "ا").replace("ي", "ی")
        elif lang == "en":
            text = text.replace("’", "'").replace("“", '"')
        return text

    def encode(self, text: str, lang: str) -> List[int]:
        """رمزگذاری متن با کش و مدیریت خطا"""
        if lang not in self.languages:
            raise ValueError(f"زبان {lang} پشتیبانی نمی‌شه")
        text = self.preprocess_text(text, lang)
        key = (lang, hashlib.sha256(text.encode()).hexdigest())
        if key in self.cache:
            self.stats["hits"] += 1
            return self.cache[key]
        tokens = self.tokenizers[lang].encode(text).ids
        self.cache[key] = tokens
        self.stats["misses"] += 1
        self.stats["processed_texts"] += 1
        return tokens

    def decode(self, tokens: List[int], lang: str) -> str:
        """رمزگشایی توکن‌ها به متن با بررسی زبان"""
        if lang not in self.languages:
            raise ValueError(f"زبان {lang} پشتیبانی نمی‌شه")
        return self.tokenizers[lang].decode(tokens)

    def pad(self, sequences: List[List[int]], max_len: int = config.max_seq_len) -> jnp.ndarray:
        """پد کردن دنباله‌ها برای یکنواختی طول ورودی‌ها"""
        padded = []
        for seq in sequences:
            seq = seq[:max_len]
            padded_seq = seq + [self.special_tokens["[PAD]"]] * max(0, max_len - len(seq))
            padded.append(padded_seq)
        return jnp.array(padded)

    def augment_text(self, text: str, lang: str) -> str:
        """تقویت متن برای افزایش تنوع داده‌های آموزشی"""
        words = text.split()
        if np.random.random() < 0.3:
            idx = np.random.randint(len(words))
            words[idx] = self.tokenizers[lang].decode([self.special_tokens["[UNK]"]])
        elif np.random.random() < 0.2:
            idx = np.random.randint(len(words))
            words.pop(idx)
        elif np.random.random() < 0.1:
            idx = np.random.randint(len(words))
            words.insert(idx, words[idx])
        self.stats["augmentations"] += 1
        return " ".join(words)

    def batch_encode(self, texts: List[str], lang: str, max_len: int = config.max_seq_len, 
                     augment: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """رمزگذاری دسته‌ای متون با گزینه تقویت داده"""
        if augment:
            texts = [self.augment_text(text, lang) for text in texts]
        encoded = [self.encode(text, lang) for text in texts]
        input_ids = self.pad(encoded, max_len)
        mask = (input_ids != self.special_tokens["[PAD]"]).astype(jnp.float32)[:, None, None, :]
        return input_ids, mask

    def encode_parallel(self, texts: List[str], lang: str, num_threads: int = config.num_workers) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """رمزگذاری موازی متون برای سرعت بیشتر"""
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            encoded = list(executor.map(lambda text: self.encode(text, lang), texts))
        return self.batch_encode([e for e in encoded], lang)

    def get_stats(self) -> Dict[str, int]:
        """دریافت آمار عملکرد توکنایزر"""
        return self.stats

    def clear_cache(self):
        """پاک کردن کش برای آزادسازی حافظه"""
        self.cache.clear()
        self.stats["hits"] = 0
        self.stats["misses"] = 0
        logger.info("کش توکنایزر پاک شد.")

# 3. دیتالودر پیشرفته
class DariushDataLoader:
    """
    دیتالودر چندزبانه با شاردینگ پویا و مدیریت کارگرها
    این کلاس برای بارگذاری داده‌ها به صورت موازی و کارآمد استفاده می‌شه
    """
    def __init__(self, tokenizer: DariushTokenizer, batch_size: int, datasets: Dict[str, List[str]], 
                 num_workers: int = config.num_workers, prefetch_size: int = config.prefetch_size):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.datasets = datasets
        self.num_workers = min(num_workers, mp.cpu_count())  # تنظیم پویا تعداد کارگرها
        self.prefetch_size = prefetch_size
        self.queue = mp.Queue(maxsize=prefetch_size)
        self.priority_queue = queue.PriorityQueue(maxsize=prefetch_size)
        self.total_samples = {lang: len(data) for lang, data in datasets.items()}
        self.cache = deque(maxlen=2000)
        self.cache_lock = threading.Lock()
        self.running = False
        self.languages = list(datasets.keys())
        self.difficulty = {lang: 1.0 for lang in self.languages}
        self.shard_index = {lang: 0 for lang in self.languages}

    def start(self):
        """شروع کارگرها برای بارگذاری داده‌ها"""
        self.running = True
        self.processes = []
        for i in range(self.num_workers):
            p = mp.Process(target=self._worker_fn, args=(i,))
            p.daemon = True
            p.start()
            self.processes.append(p)
        logger.info(f"{self.num_workers} کارگر دیتالودر شروع به کار کردن.")

    def stop(self):
        """توقف کارگرها و آزادسازی منابع"""
        self.running = False
        for p in self.processes:
            p.terminate()
            p.join()
        logger.info("کارگرهای دیتالودر متوقف شدن.")

    def _worker_fn(self, worker_id: int):
        """تابع کارگر برای بارگذاری داده‌ها با شاردینگ و تقویت"""
        shard_size = self.batch_size * 10
        while self.running:
            try:
                with self.cache_lock:
                    if self.cache and np.random.random() < 0.4:
                        batch = self.cache[np.random.randint(len(self.cache))]
                    else:
                        lang = np.random.choice(self.languages, 
                                              p=[self.difficulty[l] / sum(self.difficulty.values()) for l in self.languages])
                        dataset = self.datasets[lang]
                        shard_start = (self.shard_index[lang] * shard_size) % self.total_samples[lang]
                        shard_end = min(shard_start + shard_size, self.total_samples[lang])
                        shard_texts = dataset[shard_start:shard_end]
                        batch_texts = np.random.choice(shard_texts, self.batch_size, replace=False)
                        input_ids, mask = self.tokenizer.batch_encode(batch_texts, lang, augment=True)
                        batch = {
                            "input_ids": input_ids,
                            "labels": input_ids,
                            "mask": mask,
                            "lang": lang,
                            "difficulty": self.difficulty[lang]
                        }
                        self.cache.append(batch)
                        self.shard_index[lang] += 1
                priority = self.difficulty[lang] + np.random.random()
                self.priority_queue.put((priority, batch))
                self.queue.put(batch, timeout=10)
            except KeyError:
                logger.error(f"زبان {lang} توی دیتاست پیدا نشد!")
                continue
            except queue.Full:
                if not self.running:
                    break
                time.sleep(1)
            except Exception as e:
                logger.error(f"کارگر {worker_id} با خطا مواجه شد: {e}")

    def __iter__(self):
        """ایجاد ایتراتور برای دیتالودر"""
        return self

    def __next__(self):
        """دریافت دسته بعدی از داده‌ها"""
        if not self.running:
            raise StopIteration
        return self.queue.get()

    def update_difficulty(self, lang: str, loss: float):
        """به‌روزرسانی سختی زبان برای Curriculum Learning"""
        with self.cache_lock:
            self.difficulty[lang] = max(0.1, min(10.0, self.difficulty[lang] + loss * 0.05))
            logger.debug(f"سختی زبان {lang} به‌روزرسانی شد: {self.difficulty[lang]}")

    def get_stats(self) -> Dict[str, Any]:
        """دریافت آمار عملکرد دیتالودر"""
        return {
            "queue_size": self.queue.qsize(),
            "priority_queue_size": self.priority_queue.qsize(),
            "cache_size": len(self.cache),
            "total_samples": self.total_samples,
            "shard_index": self.shard_index,
            "difficulty": self.difficulty
        }

    def clear_cache(self):
        """پاک کردن کش دیتالودر"""
        with self.cache_lock:
            self.cache.clear()
        logger.info("کش دیتالودر پاک شد.")
       

# 4. نرمال‌سازی RMS پیشرفته
class DariushRMSNorm(hk.Module):
    """
    نرمال‌سازی RMS برای تعبیه‌ها با بهینه‌سازی حافظه و شاردینگ
    این کلاس برای نرمال‌سازی تعبیه‌ها قبل از پردازش استفاده می‌شه
    """
    def __init__(self, emb_size: int, eps: float = 1e-6, name: str = "rms_norm"):
        super().__init__(name=name)
        self.emb_size = emb_size
        self.eps = eps
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        اعمال نرمال‌سازی RMS به ورودی‌ها
        Args:
            x: تنسور ورودی با شکل [batch, seq_len, emb_size]
        Returns:
            تنسور نرمال‌شده با همان شکل
        """
        scale = hk.get_parameter("scale", [self.emb_size], init=jnp.ones)
        scale = pjit_sharding_constraint(scale, P(None))
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps) * scale
        return normed.astype(jnp.bfloat16)
    
    def reset(self):
        """بازنشانی پارامترهای مقیاس به مقدار اولیه"""
        hk.set_parameter("scale", jnp.ones(self.emb_size))
        logger.info(f"DariushRMSNorm {self.name} بازنشانی شد.")

    def get_stats(self) -> Dict[str, float]:
        """دریافت آمار نرمال‌سازی برای تحلیل"""
        scale = hk.get_parameter("scale", [self.emb_size], init=jnp.ones)
        return {
            "mean_scale": float(jnp.mean(scale)),
            "std_scale": float(jnp.std(scale))
        }

# 5. تعبیه موقعیت چرخشی پیشرفته
class DariushRotaryEmbedding(hk.Module):
    """
    تعبیه موقعیت چرخشی برای افزودن اطلاعات موقعیتی به توالی‌ها
    این کلاس از روش Rotary Position Embedding برای مدل استفاده می‌کنه
    """
    def __init__(self, dim: int, base: int = 10000, max_seq_len: int = config.max_seq_len, 
                 name: str = "rotary_emb"):
        super().__init__(name=name)
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def __call__(self, x: jnp.ndarray, offset: int = 0) -> jnp.ndarray:
        """
        اعمال تعبیه موقعیت چرخشی به ورودی‌ها
        Args:
            x: تنسور ورودی با شکل [batch, seq_len, num_heads, key_size]
            offset: جابجایی موقعیت برای تولید متن
        Returns:
            تنسور با تعبیه‌های موقعیتی اعمال‌شده
        """
        seq_len = x.shape[1]
        pos = jnp.arange(seq_len, dtype=jnp.float32) + offset
        angles = pos[:, None] * self.inv_freq[None, :]
        sin_val = jnp.sin(angles)
        cos_val = jnp.cos(angles)
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        x_rot = jnp.concatenate([-x2, x1], axis=-1)
        return x * cos_val + x_rot * sin_val
    
    def get_angles(self, seq_len: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """محاسبه زوایا برای تحلیل یا دیباگ"""
        pos = jnp.arange(seq_len, dtype=jnp.float32)
        angles = pos[:, None] * self.inv_freq[None, :]
        return jnp.sin(angles), jnp.cos(angles)

    def validate_input(self, x: jnp.ndarray):
        """اعتبارسنجی شکل ورودی برای جلوگیری از خطا"""
        assert x.shape[-1] == self.dim, f"آخرین بُعد ورودی ({x.shape[-1]}) باید برابر با dim ({self.dim}) باشه"

# 6. SwiGLU پیشرفته
class DariushSwiGLU(hk.Module):
    """
    فعال‌سازی SwiGLU برای بهبود عملکرد شبکه‌های عصبی
    این کلاس یه جایگزین قدرتمند برای ReLU و GELU هست
    """
    def __init__(self, hidden_size: int, name: str = "swiglu"):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        اعمال فعال‌سازی SwiGLU به ورودی‌ها
        Args:
            x: تنسور ورودی با شکل [batch, seq_len, emb_size]
        Returns:
            تنسور فعال‌شده با همان شکل
        """
        w1 = hk.Linear(self.hidden_size, name="w1", 
                      w_init=hk.initializers.TruncatedNormal(stddev=0.02))
        w2 = hk.Linear(self.hidden_size, name="w2", 
                      w_init=hk.initializers.TruncatedNormal(stddev=0.02))
        return jax.nn.silu(w1(x)) * w2(x)
    
    def get_weights(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """دریافت وزن‌های w1 و w2 برای تحلیل یا دیباگ"""
        w1 = hk.get_parameter("w1", shape=[self.hidden_size, self.hidden_size], 
                             init=hk.initializers.TruncatedNormal(stddev=0.02))
        w2 = hk.get_parameter("w2", shape=[self.hidden_size, self.hidden_size], 
                             init=hk.initializers.TruncatedNormal(stddev=0.02))
        return w1, w2
    
    def reset(self):
        """بازنشانی وزن‌ها به مقادیر اولیه"""
        hk.set_parameter("w1", hk.initializers.TruncatedNormal(stddev=0.02)([self.hidden_size, self.hidden_size]))
        hk.set_parameter("w2", hk.initializers.TruncatedNormal(stddev=0.02)([self.hidden_size, self.hidden_size]))
        logger.info(f"DariushSwiGLU {self.name} بازنشانی شد.")

# 7. Flash Attention 2 پیشرفته
class DariushFlashAttention2(hk.Module):
    """
    Flash Attention 2 برای بهینه‌سازی توجه با سرعت بالا و مصرف حافظه کم
    این کلاس جایگزین توجه استاندارد برای کارایی بهتره
    """
    def __init__(self, num_heads: int, key_size: int, block_size: int = 128, 
                 name: str = "flash_attention2"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.block_size = block_size
    
    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, 
                 mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        اعمال Flash Attention 2 به کوئری، کلید و مقدار
        Args:
            q: تنسور کوئری با شکل [batch, seq_len, num_heads, key_size]
            k: تنسور کلید با شکل [batch, seq_len, num_heads, key_size]
            v: تنسور مقدار با شکل [batch, seq_len, num_heads, key_size]
            mask: ماسک اختیاری با شکل [batch, 1, seq_len]
        Returns:
            تنسور خروجی با شکل [batch, seq_len, num_heads * key_size]
        """
        batch, seq_len, _ = q.shape
        q = q.reshape(batch, seq_len, self.num_heads, self.key_size)
        k = k.reshape(batch, seq_len, self.num_heads, self.key_size)
        v = v.reshape(batch, seq_len, self.num_heads, self.key_size)

        def block_attention(q_block, k_block, v_block, mask_block):
            """محاسبه توجه در بلوک‌های کوچک برای بهینه‌سازی حافظه"""
            attn_logits = jnp.einsum("...hd,...kd->...hk", q_block, k_block) / jnp.sqrt(self.key_size)
            if mask_block is not None:
                attn_logits += mask_block * -1e30  # اعمال ماسک با بهینه‌سازی
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
    
    def get_block_stats(self, seq_len: int) -> Dict[str, int]:
        """دریافت آمار بلوک‌ها برای تحلیل عملکرد"""
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        return {
            "num_blocks": num_blocks,
            "block_size": self.block_size,
            "total_heads": self.num_heads
        }
    
    def validate_inputs(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray):
        """اعتبارسنجی ورودی‌ها برای جلوگیری از خطا"""
        assert q.shape == k.shape == v.shape, "شکل‌های q، k و v باید یکسان باشن"
        assert q.shape[-1] == self.key_size, f"آخرین بُعد باید {self.key_size} باشه"



# 8. توجه پراکنده پویا پیشرفته
class DariushDynamicSparseAttention(hk.Module):
    """
    توجه پراکنده پویا برای کاهش پیچیدگی محاسباتی با انتخاب هوشمند
    این کلاس توجه رو فقط به مهم‌ترین بخش‌های توالی محدود می‌کنه
    """
    def __init__(self, num_heads: int, key_size: int, sparse_factor: int = config.sparse_factor, 
                 name: str = "dynamic_sparse_attention"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.sparse_factor = sparse_factor
        
    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, 
                 mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        اعمال توجه پراکنده پویا به ورودی‌ها
        Args:
            q: تنسور کوئری با شکل [batch, seq_len, num_heads, key_size]
            k: تنسور کلید با شکل [batch, seq_len, num_heads, key_size]
            v: تنسور مقدار با شکل [batch, seq_len, num_heads, key_size]
            mask: ماسک اختیاری با شکل [batch, 1, seq_len]
        Returns:
            تنسور خروجی با شکل [batch, sparse_seq_len, num_heads * key_size]
        """
        batch, seq_len, _ = q.shape
        q = q.reshape(batch, seq_len, self.num_heads, self.key_size)
        k = k.reshape(batch, seq_len, self.num_heads, self.key_size)
        v = v.reshape(batch, seq_len, self.num_heads, self.key_size)

        # محاسبه اهمیت برای انتخاب پویا
        importance = jnp.mean(jnp.abs(q), axis=-1)
        sparse_indices = jax.lax.top_k(importance, seq_len // self.sparse_factor)[1]
        q_sparse = q[jnp.arange(batch)[:, None], sparse_indices]
        k_sparse = k[jnp.arange(batch)[:, None], sparse_indices]
        v_sparse = v[jnp.arange(batch)[:, None], sparse_indices]

        attn_logits = jnp.einsum("...qhd,...khd->...hqk", q_sparse, k_sparse) / jnp.sqrt(self.key_size)
        if mask is not None:
            mask_sparse = mask[:, :, sparse_indices]
            attn_logits = jnp.where(mask_sparse, attn_logits, -1e30)
        attn_weights = jax.nn.softmax(attn_logits)
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, v_sparse)
        return attn_output.reshape(batch, seq_len // self.sparse_factor, self.num_heads * self.key_size)
    
    def get_sparsity_stats(self, importance: jnp.ndarray) -> Dict[str, Any]:
        """دریافت آمار پراکندگی برای تحلیل"""
        sparse_len = importance.shape[1] // self.sparse_factor
        return {
            "sparse_length": sparse_len,
            "sparse_factor": self.sparse_factor,
            "mean_importance": float(jnp.mean(importance))
        }
    
    def validate_inputs(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray):
        """اعتبارسنجی ورودی‌ها برای اطمینان از سازگاری"""
        assert q.shape == k.shape == v.shape, "شکل‌های q، k و v باید یکسان باشن"
        assert q.shape[-1] == self.key_size, f"آخرین بُعد باید {self.key_size} باشه"

# 9. Mixture of Experts پیشرفته
class DariushRouter(hk.Module):
    """
    روتر Mixture of Experts برای انتخاب هوشمند کارشناسان
    این کلاس تصمیم می‌گیره کدوم کارشناسان برای هر ورودی فعال بشن
    """
    def __init__(self, num_experts: int, num_selected_experts: int, name: str = "router"):
        super().__init__(name=name)
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts
        
    def __call__(self, inputs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        انتخاب کارشناسان با استفاده از Softmax و نویز Gumbel
        Args:
            inputs: تنسور ورودی با شکل [batch, seq_len, emb_size]
        Returns:
            gates: وزن‌های انتخاب با شکل [batch, seq_len, num_selected_experts]
            indices: اندیس‌های کارشناسان انتخاب‌شده با شکل [batch, seq_len, num_selected_experts]
        """
        w = hk.get_parameter("w", [inputs.shape[-1], self.num_experts], 
                            init=hk.initializers.TruncatedNormal(stddev=0.02))
        w = pjit_sharding_constraint(w, P("data", "expert"))
        logits = jnp.dot(inputs.astype(jnp.float32), w)
        noise = jax.random.gumbel(jax.random.PRNGKey(time.time_ns()), logits.shape) * 0.05
        probs = jax.nn.softmax(logits + noise)
        gates, indices = jax.lax.top_k(probs, self.num_selected_experts)
        return gates, indices
    
    def get_router_stats(self, probs: jnp.ndarray) -> Dict[str, float]:
        """دریافت آمار روتر برای تحلیل انتخاب کارشناسان"""
        return {
            "mean_prob": float(jnp.mean(probs)),
            "max_prob": float(jnp.max(probs)),
            "selected_experts": self.num_selected_experts
        }
    
    def reset(self):
        """بازنشانی وزن‌های روتر"""
        hk.set_parameter("w", hk.initializers.TruncatedNormal(stddev=0.02)([self.num_experts, self.num_experts]))
        logger.info(f"DariushRouter {self.name} بازنشانی شد.")

class DariushMoELayer(hk.Module):
    """
    لایه Mixture of Experts برای پردازش موازی با کارشناسان متعدد
    این کلاس از روتر برای انتخاب کارشناسان و پردازش ورودی‌ها استفاده می‌کنه
    """
    def __init__(self, config: DariushConfig, mesh: jax.sharding.Mesh, name: str = "moe"):
        super().__init__(name=name)
        self.config = config
        self.mesh = mesh
        self.router = DariushRouter(config.num_experts, config.num_selected_experts)
        
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        اعمال لایه MoE به ورودی‌ها
        Args:
            inputs: تنسور ورودی با شکل [batch, seq_len, emb_size]
        Returns:
            تنسور خروجی با شکل [batch, seq_len, emb_size]
        """
        gates, indices = self.router(inputs)
        expert_outputs = []

        def expert_fn(x: jnp.ndarray) -> jnp.ndarray:
            """تابع کارشناس جداگانه با SwiGLU یا GELU"""
            w = hk.Linear(int(self.config.widening_factor * self.config.emb_size), name="expert",
                         w_init=hk.initializers.TruncatedNormal(stddev=0.02))
            w_out = hk.Linear(self.config.emb_size, name="expert_out",
                            w_init=hk.initializers.TruncatedNormal(stddev=0.02))
            if self.config.use_swiglu:
                return w_out(DariushSwiGLU(self.config.emb_size)(x))
            return w_out(jax.nn.gelu(w(x)))

        for _ in range(self.config.num_experts):
            expert_outputs.append(expert_fn(inputs))

        expert_outputs = jnp.stack(expert_outputs, axis=1)  # [batch, experts, seq_len, emb_size]

        @functools.partial(shard_map, mesh=self.mesh, 
                           in_specs=(P("data", None, "expert"), P("expert", "data", "model", "tensor")),
                           out_specs=P("data", "model", "tensor"), check_rep=False)
        def compute_expert_output(inputs, expert_outs):
            """محاسبه خروجی کارشناسان با شاردینگ"""
            return jax.vmap(lambda x, idx: x[idx])(inputs, indices)

        selected_outputs = compute_expert_output(inputs, expert_outputs)
        return (selected_outputs * gates[..., None]).sum(axis=1)
    
    def get_expert_usage(self, indices: jnp.ndarray) -> Dict[int, float]:
        """محاسبه درصد استفاده از هر کارشناس"""
        usage = {}
        for idx in range(self.config.num_experts):
            usage[idx] = float(jnp.mean(indices == idx))
        return usage
    
    def validate_experts(self, gates: jnp.ndarray, indices: jnp.ndarray):
        """اعتبارسنجی خروجی روتر"""
        assert gates.shape[-1] == self.config.num_selected_experts, "تعداد gates باید برابر num_selected_experts باشه"

# 10. توجه چندسر پیشرفته
class DariushMultiHeadAttention(hk.Module):
    """
    توجه چندسر با گزینه‌های Flash Attention 2 و پراکندگی پویا
    این کلاس هسته اصلی مکانیزم توجه مدل رو تشکیل می‌ده
    """
    def __init__(self, config: DariushConfig, name: str = "multi_head_attention"):
        super().__init__(name=name)
        self.config = config
        self.rotary = DariushRotaryEmbedding(config.key_size)
        
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, 
                 kv_cache: Optional[Dict] = None) -> Tuple[jnp.ndarray, Dict]:
        """
        اعمال توجه چندسر به ورودی‌ها
        Args:
            x: تنسور ورودی با شکل [batch, seq_len, emb_size]
            mask: ماسک اختیاری با شکل [batch, 1, seq_len]
            kv_cache: کش اختیاری برای تولید متن با شکل Dict["k", "v"]
        Returns:
            خروجی توجه با شکل [batch, seq_len, emb_size]
            کش جدید برای تولید متن
        """
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
            flash_attn = DariushFlashAttention2(self.config.num_q_heads, self.config.key_size)
            attn_output = flash_attn(q, k, v, mask)
        elif self.config.use_dynamic_sparsity:
            sparse_attn = DariushDynamicSparseAttention(self.config.num_q_heads, self.config.key_size)
            attn_output = sparse_attn(q, k, v, mask)
        else:
            attn_logits = jnp.einsum("...qhd,...khd->...hqk", q, k) / jnp.sqrt(self.config.key_size)
            if mask is not None:
                attn_logits = jnp.where(mask, attn_logits, -1e30)
            attn_weights = jax.nn.softmax(attn_logits)
            attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, v).reshape(*x.shape[:-1], -1)

        return out_w(attn_output), {"k": k, "v": v}
    
    def get_attention_weights(self, q: jnp.ndarray, k: jnp.ndarray, mask: Optional[jnp.ndarray]) -> jnp.ndarray:
        """دریافت وزن‌های توجه برای تحلیل یا بصری‌سازی"""
        attn_logits = jnp.einsum("...qhd,...khd->...hqk", q, k) / jnp.sqrt(self.config.key_size)
        if mask is not None:
            attn_logits = jnp.where(mask, attn_logits, -1e30)
        return jax.nn.softmax(attn_logits)
    
    def validate_cache(self, kv_cache: Optional[Dict]):
        """اعتبارسنجی کش برای تولید متن"""
        if kv_cache is not None:
            assert "k" in kv_cache and "v" in kv_cache, "کش باید شامل k و v باشه"

# 11. لایه Dariush پیشرفته
class DariushLayer(hk.Module):
    """
    لایه اصلی ترنسفورمر با توجه چندسر و MoE
    این کلاس واحد اصلی پردازش رو تشکیل می‌ده
    """
    def __init__(self, config: DariushConfig, mesh: jax.sharding.Mesh, layer_idx: int, 
                 name: str = "dariush_layer"):
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
        """
        اعمال لایه ترنسفورمر به ورودی‌ها
        Args:
            x: تنسور ورودی با شکل [batch, seq_len, emb_size]
            mask: ماسک اختیاری با شکل [batch, 1, seq_len]
            kv_cache: کش اختیاری برای تولید متن
        Returns:
            تنسور خروجی و کش جدید
        """
        if self.config.gradient_checkpointing:
            attn_out, new_cache = hk.checkpoint(lambda x: self.attn(self.norm1(x), mask, kv_cache))(x)
        else:
            attn_out, new_cache = self.attn(self.norm1(x), mask, kv_cache)
        x = x + self.dropout(attn_out, rate=self.config.dropout_rate, 
                            salt=jax.random.PRNGKey(self.layer_idx))
        moe_out = self.moe(self.norm2(x))
        x = x + self.dropout(moe_out, rate=self.config.dropout_rate, 
                            salt=jax.random.PRNGKey(self.layer_idx + 1))
        return x, new_cache
    
    def get_layer_stats(self, x: jnp.ndarray) -> Dict[str, float]:
        """دریافت آمار لایه برای تحلیل عملکرد"""
        return {
            "input_mean": float(jnp.mean(x)),
            "input_std": float(jnp.std(x))
        }
    
    def reset(self):
        """بازنشانی لایه برای شروع مجدد"""
        self.norm1.reset()
        self.norm2.reset()
        logger.info(f"DariushLayer {self.name} بازنشانی شد.")


# 12. Vision Transformer پیشرفته
class VisionTransformer(hk.Module):
    """
    Vision Transformer برای پردازش تصاویر و تبدیل به تعبیه‌های برداری
    این کلاس تصاویر رو به پچ‌های کوچک تقسیم می‌کنه و پردازش می‌کنه
    """
    def __init__(self, emb_size: int, num_patches: int = 64, num_heads: int = 8, num_layers: int = 12, 
                 name: str = "vision_transformer"):
        super().__init__(name=name)
        self.emb_size = emb_size
        self.num_patches = num_patches
        self.patch_embedding = hk.Linear(emb_size)
        self.positional_embedding = hk.Embed(num_patches, emb_size)
        self.transformer_layers = [hk.Linear(emb_size) for _ in range(num_layers)]  # ساده‌سازی برای نمونه
        self.mlp_head = hk.Linear(emb_size)
        
    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        """
        اعمال Vision Transformer به تصاویر
        Args:
            images: تنسور تصاویر با شکل [batch, height, width, channels]
        Returns:
            تعبیه‌های برداری با شکل [batch, num_patches, emb_size]
        """
        patches = self._extract_patches(images)
        patch_emb = self.patch_embedding(patches)
        pos_emb = self.positional_embedding(jnp.arange(self.num_patches))
        x = patch_emb + pos_emb
        for layer in self.transformer_layers:
            x = jax.nn.relu(layer(x))
        return self.mlp_head(x)
    
    def _extract_patches(self, images: jnp.ndarray) -> jnp.ndarray:
        """استخراج پچ‌ها از تصاویر با تقسیم‌بندی"""
        batch, height, width, channels = images.shape
        patch_size = height // int(self.num_patches ** 0.5)
        patches = images.reshape(batch, self.num_patches, -1)
        return patches
    
    def get_patch_stats(self, patches: jnp.ndarray) -> Dict[str, float]:
        """دریافت آمار پچ‌ها برای تحلیل"""
        return {
            "patch_count": self.num_patches,
            "mean_patch": float(jnp.mean(patches)),
            "std_patch": float(jnp.std(patches))
        }

# 13. Whisper Encoder پیشرفته
class WhisperEncoder(hk.Module):
    """
    Whisper-inspired Encoder برای پردازش سیگنال‌های صوتی
    این کلاس صوت رو به تعبیه‌های برداری تبدیل می‌کنه
    """
    def __init__(self, emb_size: int, name: str = "whisper_encoder"):
        super().__init__(name=name)
        self.emb_size = emb_size
        self.conv1 = hk.Conv1D(64, kernel_shape=3, stride=2, padding="VALID")
        self.conv2 = hk.Conv1D(128, kernel_shape=3, stride=2, padding="VALID")
        self.flatten = hk.Flatten()
        self.proj = hk.Linear(self.emb_size)
        
    def __call__(self, audio: jnp.ndarray) -> jnp.ndarray:
        """
        اعمال Whisper Encoder به سیگنال صوتی
        Args:
            audio: تنسور صوتی با شکل [batch, time_steps, channels]
        Returns:
            تعبیه‌های برداری با شکل [batch, emb_size]
        """
        x = self.conv1(audio)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)
        x = self.flatten(x)
        return self.proj(x)
    
    def get_audio_stats(self, audio: jnp.ndarray) -> Dict[str, float]:
        """دریافت آمار سیگنال صوتی برای تحلیل"""
        return {
            "mean_amplitude": float(jnp.mean(audio)),
            "std_amplitude": float(jnp.std(audio))
        }

# 14. مدل اصلی پیشرفته
class GodModeDariush(hk.Module):
    """
    مدل اصلی GodModeDariush با پشتیبانی چندرسانه‌ای و چندوظیفگی
    این کلاس همه اجزای مدل رو کنار هم میاره
    """
    def __init__(self, config: DariushConfig, mesh: jax.sharding.Mesh, name: str = "godmode_dariush"):
        super().__init__(name=name)
        self.config = config
        self.mesh = mesh
        self.embedding = hk.Embed(config.vocab_size, config.emb_size, name="embedding",
                                 w_init=hk.initializers.TruncatedNormal(stddev=config.init_scale))
        self.vision_transformer = VisionTransformer(config.emb_size) if config.use_vision else None
        self.whisper_encoder = WhisperEncoder(config.emb_size) if config.use_audio else None
        self.layers = [DariushLayer(config, mesh, i) for i in range(config.num_layers)]
        self.norm = DariushRMSNorm(config.emb_size)
        self.output = hk.Linear(config.vocab_size, name="output",
                               w_init=hk.initializers.TruncatedNormal(stddev=config.init_scale))
        self.translation_head = hk.Linear(config.emb_size, name="translation_head")
        self.summary_head = hk.Linear(config.emb_size, name="summary_head")
        self.qa_head = hk.Linear(config.emb_size, name="qa_head")
        
    def __call__(self, input_ids: Optional[jnp.ndarray] = None, images: Optional[jnp.ndarray] = None, 
                 audio: Optional[jnp.ndarray] = None, mask: Optional[jnp.ndarray] = None, 
                 kv_cache: Optional[List[Dict]] = None, task: str = "language_modeling") -> Tuple[jnp.ndarray, List[Dict]]:
        """
        اعمال مدل به ورودی‌های چندرسانه‌ای
        Args:
            input_ids: تنسور متنی با شکل [batch, seq_len]
            images: تنسور تصاویر با شکل [batch, height, width, channels]
            audio: تنسور صوتی با شکل [batch, time_steps, channels]
            mask: ماسک اختیاری
            kv_cache: کش برای تولید متن
            task: نوع وظیفه (language_modeling, translation, summary, qa)
        Returns:
            خروجی مدل و کش جدید
        """
        if input_ids is not None:
            x = self.embedding(input_ids)
        elif images is not None and self.config.use_vision:
            x = self.vision_transformer(images)
        elif audio is not None and self.config.use_audio:
            x = self.whisper_encoder(audio)
        else:
            raise ValueError("حداقل یه ورودی (متن، تصویر یا صوت) باید داده بشه")

        x = pjit_sharding_constraint(x, P(self.config.data_axis, None, self.config.model_axis, self.config.tensor_axis))
        new_kv_cache = [] if kv_cache is None else kv_cache

        for i, layer in enumerate(self.layers):
            x, layer_cache = layer(x, mask, new_kv_cache[i] if kv_cache else None)
            new_kv_cache.append(layer_cache)

        x = self.norm(x)
        if task == "translation":
            return self.translation_head(x), new_kv_cache
        elif task == "summary":
            return self.summary_head(x), new_kv_cache
        elif task == "qa":
            return self.qa_head(x), new_kv_cache
        else:
            logits = self.output(x)
            return logits, new_kv_cache
    
    def init_memory(self, batch_size: int, seq_len: int) -> List[Dict]:
        """راه‌اندازی حافظه KV برای تولید متن"""
        return [{"k": jnp.zeros((batch_size, seq_len, self.config.num_kv_heads, self.config.key_size), dtype=jnp.bfloat16),
                 "v": jnp.zeros((batch_size, seq_len, self.config.num_kv_heads, self.config.key_size), dtype=jnp.bfloat16)}
                for _ in range(self.config.num_layers)]
    
    def validate_inputs(self, input_ids, images, audio):
        """اعتبارسنجی ورودی‌ها برای جلوگیری از خطا"""
        assert any([input_ids is not None, images is not None, audio is not None]), "حداقل یه ورودی باید داده بشه"

# 15. مدیریت چک‌پوینت پیشرفته
class DariushCheckpointManager:
    """
    مدیر چک‌پوینت برای ذخیره و بازیابی مدل با پشتیبانی ابری
    این کلاس برای مدیریت حالت مدل در طول آموزش استفاده می‌شه
    """
    def __init__(self, save_dir: str = "dariush_checkpoints", cloud_storage: str = "s3", max_checkpoints: int = 10):
        self.save_dir = save_dir
        self.cloud_storage = cloud_storage
        self.max_checkpoints = max_checkpoints
        os.makedirs(save_dir, exist_ok=True)
        if cloud_storage == "s3":
            self.s3 = boto3.client('s3')
        elif cloud_storage == "gcs":
            self.gcs = storage.Client()
        else:
            raise ValueError(f"ذخیره‌سازی ابری {cloud_storage} پشتیبانی نمی‌شه")
        self.checkpoints = OrderedDict()
        self.lock = threading.Lock()
        
    def save(self, params: Any, step: int, metadata: Dict = None):
        """
        ذخیره چک‌پوینت در حافظه محلی و ابری
        Args:
            params: پارامترهای مدل
            step: گام فعلی آموزش
            metadata: اطلاعات اضافی برای ذخیره
        """
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
            logger.info(f"چک‌پوینت در گام {step} ذخیره شد در {path}")
    
    def load(self, step: int) -> Tuple[Any, Dict]:
        """
        بازیابی چک‌پوینت از حافظه محلی یا ابری
        Args:
            step: گام مورد نظر برای بازیابی
        Returns:
            پارامترهای مدل و متادیتا
        """
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
        """دریافت گام آخرین چک‌پوینت ذخیره‌شده"""
        with self.lock:
            return max(self.checkpoints.keys()) if self.checkpoints else None
    
    def cleanup(self):
        """پاکسازی همه چک‌پوینت‌ها از حافظه محلی"""
        with self.lock:
            for path in self.checkpoints.values():
                if os.path.exists(path):
                    os.remove(path)
            self.checkpoints.clear()
            logger.info("همه چک‌پوینت‌ها پاک شدن.")

# این کد از سورس‌های زیر الهام گرفته شده:
# - DariushGPT (Copyright (c) 2025 hosein davod abadi farahani)
# - xAI Transformer (Copyright 2024 X.AI Corp., Apache License 2.0)
# - الهام از LLaMA, Mixtral, GPT-4, Grok و تکنیک‌های پیشرفته 2025
