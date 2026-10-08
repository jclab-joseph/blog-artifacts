import base64
import hashlib
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image
from scipy.stats import spearmanr
from tqdm.auto import tqdm

HERE = Path(__file__).parent

# SearchQuery prompt from the google/embeddinggemma-2 model card. Gemini
# Embedding 2 documents the same prefix format in place of task_type.
GEMMA_QUERY_PREFIX = "task: search result | query: "
# Qwen3-VL-Embedding takes the instruction as a system prompt. The T2I one is
# the MMEB-V2 instruction for caption->image retrieval.
QWEN_T2I_INSTRUCTION = "Find me an everyday image that matches the given caption."
QWEN_DOC_INSTRUCTION = "Find a document page image that answers the user's question."

CONFIGS = [
    {
        "name": "google/embeddinggemma-2",
        "kind": "sbert",
        "variant": "original",
        "model_kwargs": {"torch_dtype": "bfloat16"},
        "query_prompt": {"image": GEMMA_QUERY_PREFIX, "doc": GEMMA_QUERY_PREFIX},
    },
    {
        "name": "Qwen/Qwen3-VL-Embedding-2B",
        "kind": "sbert",
        "variant": "original",
        "model_kwargs": {"torch_dtype": "bfloat16"},
        "query_prompt": {"image": QWEN_T2I_INSTRUCTION, "doc": QWEN_DOC_INSTRUCTION},
    },
    {
        # OpenRouter applies the model's query:/passage: prefix itself through
        # input_type (default query). Images reject input_type and match the
        # local model's document embedding without it (cos 0.99, measured).
        "name": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        "kind": "openrouter",
        "variant": "OpenRouter",
    },
    {
        "name": "gemini-embedding-2",
        "kind": "gemini",
        "variant": "API",
        "query_prompt": {"image": GEMMA_QUERY_PREFIX, "doc": GEMMA_QUERY_PREFIX},
    },
    {
        "name": "voyage-multimodal-3.5",
        "kind": "voyage",
        "variant": "API",
    },
]

IMPLEMENTATION_VERSIONS = {
    "sbert": 1,
    "openrouter": 1,
    "gemini": 1,
    "voyage": 1,
}

# Every model sees the same pixels: images are shrunk to this long side and
# re-encoded as JPEG. This also bounds Voyage's per-pixel billing.
MAX_IMAGE_SIDE = 1280
JPEG_QUALITY = 90
PREPROCESS = f"maxside{MAX_IMAGE_SIDE}-jpeg{JPEG_QUALITY}"

CACHE_PATH = Path(os.environ["BENCH_CACHE"]) if os.environ.get("BENCH_CACHE") else HERE / "benchmark_cache.json"
RESULT_PATH = Path(os.environ["BENCH_RESULT"]) if os.environ.get("BENCH_RESULT") else HERE / "benchmark_results.md"
# Per-batch embeddings, so an interrupted API run never pays twice and the
# XM3600 image corpus is encoded once for both caption languages.
EMB_DIR = Path(os.environ.get("BENCH_EMB_DIR", HERE / "emb_cache"))
# Hard stop on paid API spend, summed over everything in EMB_DIR.
BUDGET_USD = float(os.environ.get("BENCH_BUDGET_USD", "4.0"))

STS_BENCHMARKS = [
    {"name": "English(STS-B val)", "type": "sts", "loader": "load_english_stsb"},
    {"name": "Korean(KLUE-STS val)", "type": "sts", "loader": "load_korean_klue_sts"},
]

RETRIEVAL_BENCHMARKS = [
    {"name": "English(XM3600 T2I)", "type": "retrieval", "loader": "load_xm3600_en", "direction": "t2i", "task": "image"},
    {"name": "English(XM3600 I2T)", "type": "retrieval", "loader": "load_xm3600_en", "direction": "i2t", "task": "image"},
    {"name": "Korean(XM3600 T2I)", "type": "retrieval", "loader": "load_xm3600_ko", "direction": "t2i", "task": "image"},
    {"name": "Korean(XM3600 I2T)", "type": "retrieval", "loader": "load_xm3600_ko", "direction": "i2t", "task": "image"},
    {"name": "English(ViDoRe DocVQA test)", "type": "retrieval", "loader": "load_vidore_docvqa", "direction": "t2i", "task": "doc"},
    {"name": "Korean(KoViDoRe v2 economic)", "type": "retrieval", "loader": "load_kovidore_economic", "direction": "t2i", "task": "doc"},
]


def read_env_key(name):
    value = os.environ.get(name)
    if not value and (HERE / ".env").exists():
        for line in (HERE / ".env").read_text().splitlines():
            if line.startswith(f"{name}="):
                value = line.split("=", 1)[1].strip().strip('"\'')
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


def prepare_image(image):
    image = image.convert("RGB")
    if max(image.size) > MAX_IMAGE_SIDE:
        image = image.copy()
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def jpeg_size(jpeg):
    return Image.open(io.BytesIO(jpeg)).size


def data_url(jpeg):
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


class ApiBase:
    """Shared request pacing and retries for the API encoders."""

    REQUESTS_PER_MIN = 300
    workers = 4

    def __init__(self):
        self._lock = threading.Lock()
        self._next_free = 0.0

    def _throttle(self):
        with self._lock:
            now = time.monotonic()
            start = max(self._next_free, now)
            self._next_free = start + 60 / self.REQUESTS_PER_MIN
        time.sleep(start - now)

    def _post(self, requests, url, headers, payload):
        err = None
        for attempt in range(8):
            self._throttle()
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=300)
                if resp.status_code == 200:
                    return resp.json()
                err = f"{resp.status_code} {resp.text[:300]}"
                if 400 <= resp.status_code < 500 and resp.status_code not in (408, 429):
                    break
            except requests.RequestException as exc:
                err = repr(exc)
            time.sleep(min(2 ** attempt, 60))
        raise RuntimeError(f"{type(self).__name__} request failed: {err}")

    def close(self):
        pass


class SBertEncoder:
    """Local Sentence Transformers model; query prompts go through `prompt=`.

    For EmbeddingGemma 2 the prompt is prepended text; for Qwen3-VL-Embedding
    it replaces the default system instruction.
    """

    workers = 1
    text_batch_size = 64
    image_batch_size = 8

    def __init__(self, cfg):
        import torch
        from sentence_transformers import SentenceTransformer

        self.torch = torch
        model_kwargs = dict(cfg.get("model_kwargs") or {})
        if model_kwargs.get("torch_dtype") == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        self.model = SentenceTransformer(cfg["name"], model_kwargs=model_kwargs, trust_remote_code=True)
        self.query_prompt = cfg.get("query_prompt") or {}

    def encode_texts(self, texts, role, task=None):
        prompt = self.query_prompt.get(task) if role == "query" else None
        emb = self.model.encode(texts, prompt=prompt, batch_size=len(texts), convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32), 0.0

    def encode_images(self, jpegs):
        images = [Image.open(io.BytesIO(j)).convert("RGB") for j in jpegs]
        emb = self.model.encode(images, batch_size=len(images), convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32), 0.0

    def estimate_cost(self, texts=None, jpegs=None):
        return 0.0

    def close(self):
        import gc

        self.model = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class OpenRouterEncoder(ApiBase):
    """nvidia/llama-nemotron-embed-vl-1b-v2 through OpenRouter's embeddings API.

    Free models are capped per day (1000 requests with credits), so batches
    are large; 20 requests/min is the documented free-model rate.
    """

    URL = "https://openrouter.ai/api/v1/embeddings"
    REQUESTS_PER_MIN = 18
    workers = 2
    text_batch_size = 128
    image_batch_size = 8

    def __init__(self, cfg):
        import requests

        super().__init__()
        self.requests = requests
        self.model_name = cfg["name"]
        self.headers = {"Authorization": f"Bearer {read_env_key('OPENROUTER_API_KEY')}"}

    def _embed(self, inputs, input_type=None):
        payload = {"model": self.model_name, "input": inputs, "encoding_format": "float"}
        if input_type:
            payload["input_type"] = input_type
        body = self._post(self.requests, self.URL, self.headers, payload)
        data = sorted(body["data"], key=lambda x: x["index"])
        if len(data) != len(inputs):
            raise RuntimeError(f"OpenRouter returned {len(data)} embeddings for {len(inputs)} inputs")
        return np.asarray([d["embedding"] for d in data], dtype=np.float32), float(body.get("usage", {}).get("cost") or 0.0)

    def encode_texts(self, texts, role, task=None):
        # STS follows the local model's default prompt ("passage: ").
        return self._embed(texts, "query" if role == "query" else "passage")

    def encode_images(self, jpegs):
        return self._embed([{"content": [{"type": "image_url", "image_url": {"url": data_url(j)}}]} for j in jpegs])

    def estimate_cost(self, texts=None, jpegs=None):
        return 0.0


class GeminiEncoder(ApiBase):
    """gemini-embedding-2 via google-genai.

    Each input must be its own Content: a bare list is merged into a single
    embedding. At most 6 images and 100 contents per request. The API reports
    no usage, so cost is computed from the published prices.
    """

    TEXT_USD_PER_TOKEN = 0.20e-6
    # "$0.45 / 1M tokens (about $0.00012 per image)" on the pricing page.
    IMAGE_USD = 0.00012
    workers = 4
    text_batch_size = 100
    image_batch_size = 6

    def __init__(self, cfg):
        from google import genai
        from google.genai import types
        from transformers import AutoTokenizer

        super().__init__()
        self.types = types
        self.client = genai.Client(api_key=read_env_key("GEMINI_API_KEY"))
        self.model_name = cfg["name"]
        self.query_prompt = cfg.get("query_prompt") or {}
        # Gemini's tokenizer is not public; the Gemma one is the closest
        # stand-in and only feeds the (small) text cost estimate.
        self.tokenizer = AutoTokenizer.from_pretrained("google/embeddinggemma-2")

    def _embed(self, parts):
        contents = [self.types.Content(parts=[p]) for p in parts]
        err = None
        for attempt in range(8):
            self._throttle()
            try:
                result = self.client.models.embed_content(model=self.model_name, contents=contents)
                if len(result.embeddings) != len(parts):
                    raise RuntimeError(f"Gemini returned {len(result.embeddings)} embeddings for {len(parts)} inputs")
                return np.asarray([e.values for e in result.embeddings], dtype=np.float32)
            except Exception as exc:  # google-genai raises its own error types
                err = repr(exc)
                code = getattr(exc, "code", None)
                if isinstance(code, int) and 400 <= code < 500 and code not in (408, 429):
                    break
            time.sleep(min(2 ** attempt, 60))
        raise RuntimeError(f"Gemini request failed: {err}")

    def _text_cost(self, texts):
        return sum(len(self.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts) * self.TEXT_USD_PER_TOKEN

    def encode_texts(self, texts, role, task=None):
        prompt = self.query_prompt.get(task, "") if role == "query" else ""
        texts = [prompt + t for t in texts]
        return self._embed([self.types.Part(text=t) for t in texts]), self._text_cost(texts)

    def encode_images(self, jpegs):
        parts = [self.types.Part.from_bytes(data=j, mime_type="image/jpeg") for j in jpegs]
        return self._embed(parts), self.IMAGE_USD * len(jpegs)

    def estimate_cost(self, texts=None, jpegs=None):
        cost = self.IMAGE_USD * len(jpegs or [])
        if texts:
            # Upper bound that skips tokenizing: about one token per 2 chars
            # covers Korean; the prompt prefix adds ~8 tokens.
            cost += sum(len(t) / 2 + 10 for t in texts) * self.TEXT_USD_PER_TOKEN
        return cost


class VoyageEncoder(ApiBase):
    URL = "https://api.voyageai.com/v1/multimodalembeddings"
    TEXT_USD_PER_TOKEN = 0.12e-6
    USD_PER_PIXEL = 0.60e-9
    workers = 4
    text_batch_size = 128
    image_batch_size = 16

    def __init__(self, cfg):
        import requests

        super().__init__()
        self.requests = requests
        self.model_name = cfg["name"]
        self.headers = {"Authorization": f"Bearer {read_env_key('VOYAGE_API_KEY')}"}

    @classmethod
    def billed_pixels(cls, jpeg):
        # Billing clamps each image to [50K, 2M] pixels.
        w, h = jpeg_size(jpeg)
        return min(max(w * h, 50_000), 2_000_000)

    def _embed(self, inputs, input_type):
        body = self._post(self.requests, self.URL, self.headers, {"model": self.model_name, "inputs": inputs, "input_type": input_type})
        data = sorted(body["data"], key=lambda x: x["index"])
        return np.asarray([d["embedding"] for d in data], dtype=np.float32), body["usage"]

    def encode_texts(self, texts, role, task=None):
        # STS sends no input_type, i.e. no retrieval prompt.
        input_type = "query" if role == "query" else None
        emb, usage = self._embed([{"content": [{"type": "text", "text": t}]} for t in texts], input_type)
        return emb, usage["text_tokens"] * self.TEXT_USD_PER_TOKEN

    def encode_images(self, jpegs):
        inputs = [{"content": [{"type": "image_base64", "image_base64": data_url(j)}]} for j in jpegs]
        emb, _ = self._embed(inputs, "document")
        return emb, sum(self.billed_pixels(j) for j in jpegs) * self.USD_PER_PIXEL

    def estimate_cost(self, texts=None, jpegs=None):
        cost = sum(self.billed_pixels(j) for j in jpegs or []) * self.USD_PER_PIXEL
        cost += sum(len(t) / 2 + 2 for t in texts or []) * self.TEXT_USD_PER_TOKEN
        return cost


def build_encoder(cfg):
    return {
        "sbert": SBertEncoder,
        "openrouter": OpenRouterEncoder,
        "gemini": GeminiEncoder,
        "voyage": VoyageEncoder,
    }[cfg["kind"]](cfg)


_budget_lock = threading.Lock()
_spent = None


def spent_usd():
    """API spend recorded in EMB_DIR; scanned once, then kept up to date."""
    global _spent
    with _budget_lock:
        if _spent is None:
            _spent = 0.0
            for path in EMB_DIR.glob("*/*.npz"):
                with np.load(path) as f:
                    _spent += float(f["cost"])
        return _spent


def add_spent(cost):
    global _spent
    spent_usd()
    with _budget_lock:
        _spent += cost


def cached_encode(encoder, cfg, spec, n, make_batch, encode_batch, batch_size):
    """Encodes items [0, n) in fixed batches, persisting each batch.

    `spec` identifies the inputs (dataset, role, prompt task, item ids); with
    the config and batch size it names the cache directory. Returns the
    stacked embeddings, the summed per-batch encode seconds and the cost.
    """
    key = json.dumps(
        {
            "implementation_version": IMPLEMENTATION_VERSIONS[cfg["kind"]],
            "config": config_signature(cfg),
            "spec": spec,
            "batch_size": batch_size,
            "preprocess": PREPROCESS,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    directory = EMB_DIR / hashlib.sha256(key.encode()).hexdigest()[:24]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "key.json").write_text(key)

    starts = list(range(0, n, batch_size))
    todo = [s for s in starts if not (directory / f"{s:07d}.npz").exists()]

    def run(start):
        items = make_batch(start, min(start + batch_size, n))
        estimate = encoder.estimate_cost(**items)
        # Checked per batch, so parallel workers cannot overshoot by more
        # than one batch each.
        if estimate and spent_usd() + estimate > BUDGET_USD:
            raise RuntimeError(f"Budget ${BUDGET_USD:.2f} would be exceeded (spent ${spent_usd():.4f})")
        t0 = time.perf_counter()
        emb, cost = encode_batch(items)
        elapsed = time.perf_counter() - t0
        tmp = directory / f"{start:07d}.tmp.npz"
        np.savez(tmp, emb=emb, elapsed=elapsed, cost=cost)
        os.replace(tmp, directory / f"{start:07d}.npz")
        add_spent(cost)

    if todo:
        desc = f"{spec['dataset']}:{spec['role']}"
        if encoder.workers > 1:
            with ThreadPoolExecutor(encoder.workers) as pool:
                for _ in tqdm(pool.map(run, todo), total=len(todo), desc=desc, unit="batch", leave=False):
                    pass
        else:
            for start in tqdm(todo, desc=desc, unit="batch", leave=False):
                run(start)

    embs, elapsed, cost = [], 0.0, 0.0
    for start in starts:
        with np.load(directory / f"{start:07d}.npz") as f:
            embs.append(f["emb"])
            elapsed += float(f["elapsed"])
            cost += float(f["cost"])
    emb = np.vstack(embs)
    if emb.shape[0] != n:
        raise RuntimeError(f"{directory}: expected {n} embeddings, got {emb.shape[0]}")
    return emb, elapsed, cost


def encode_texts(encoder, cfg, dataset, role, texts, task=None):
    spec = {
        "dataset": dataset,
        "role": role,
        "task": task,
        "texts_sha": hashlib.sha256("\x00".join(texts).encode()).hexdigest(),
    }
    return cached_encode(
        encoder, cfg, spec, len(texts),
        lambda a, b: {"texts": texts[a:b]},
        lambda items: encoder.encode_texts(items["texts"], role, task),
        encoder.text_batch_size,
    )


def encode_images(encoder, cfg, dataset, ids, images):
    spec = {
        "dataset": dataset,
        "role": "image",
        "ids_sha": hashlib.sha256("\x00".join(map(str, ids)).encode()).hexdigest(),
    }
    return cached_encode(
        encoder, cfg, spec, len(ids),
        lambda a, b: {"jpegs": [prepare_image(img) for img in images[a:b]]},
        lambda items: encoder.encode_images(items["jpegs"]),
        encoder.image_batch_size,
    )


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def eval_sts(encoder, cfg, benchmark, s1, s2, y):
    e1, t1, c1 = encode_texts(encoder, cfg, benchmark["name"] + ":s1", "sts", s1)
    e2, t2, c2 = encode_texts(encoder, cfg, benchmark["name"] + ":s2", "sts", s2)
    rho = float(spearmanr(np.sum(normalize(e1) * normalize(e2), axis=1), y).correlation)
    elapsed = t1 + t2
    return {"samples": len(s1), "spearman": rho, "elapsed_s": elapsed, "sec_per_pair": elapsed / len(s1), "cost_usd": c1 + c2}


def ranking_metrics(scores, relevant):
    """`relevant[i]` maps corpus column -> gain for row i of `scores`."""
    k_max = 10
    top = np.argpartition(-scores, k_max, axis=1)[:, :k_max]
    order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
    top = np.take_along_axis(top, order, axis=1)

    recall = {1: [], 5: [], 10: []}
    mrr, ndcg = [], []
    discounts = 1 / np.log2(np.arange(2, 7))
    for row, gains in zip(top, relevant):
        hits = [gains.get(int(col), 0) for col in row]
        for k in recall:
            recall[k].append(any(hits[:k]))
        first = next((rank for rank, g in enumerate(hits, start=1) if g > 0), None)
        mrr.append(1 / first if first else 0.0)
        ideal = sorted(gains.values(), reverse=True)[:5]
        ndcg.append(float(np.dot(hits[:5], discounts)) / float(np.dot(ideal, discounts[:len(ideal)])))
    return {
        "recall@1": float(np.mean(recall[1])),
        "recall@5": float(np.mean(recall[5])),
        "recall@10": float(np.mean(recall[10])),
        "mrr@10": float(np.mean(mrr)),
        "ndcg@5": float(np.mean(ndcg)),
    }


def eval_retrieval(encoder, cfg, benchmark, data):
    q_texts = [q["text"] for q in data["queries"]]
    q_emb, q_time, q_cost = encode_texts(encoder, cfg, data["name"] + ":queries", "query", q_texts, benchmark["task"])
    c_emb, c_time, c_cost = encode_images(encoder, cfg, data["image_set"], data["corpus_ids"], data["corpus_images"])
    scores = normalize(q_emb) @ normalize(c_emb).T

    col = {cid: i for i, cid in enumerate(data["corpus_ids"])}
    if benchmark["direction"] == "t2i":
        relevant = [{col[c]: g for c, g in data["relevant"][q["id"]].items()} for q in data["queries"]]
        n_queries, n_corpus = len(q_texts), len(col)
    else:
        # Image -> caption: every caption of the image counts as relevant.
        by_image = [dict() for _ in col]
        for qi, q in enumerate(data["queries"]):
            for c, g in data["relevant"][q["id"]].items():
                by_image[col[c]][qi] = g
        scores, relevant = scores.T, by_image
        n_queries, n_corpus = len(col), len(q_texts)

    return {
        "queries": n_queries,
        "corpus": n_corpus,
        **ranking_metrics(scores, relevant),
        "query_encode_s": q_time,
        "corpus_encode_s": c_time,
        "sec_per_query_text": q_time / len(q_texts),
        "sec_per_image": c_time / len(col),
        # Kept apart because XM3600 en/ko share one image corpus encoding.
        "query_cost_usd": q_cost,
        "corpus_cost_usd": c_cost,
        "image_set": data["image_set"],
    }


def load_english_stsb():
    ds = load_dataset("mteb/stsbenchmark-sts", split="validation")
    return ds["sentence1"], ds["sentence2"], [float(v) / 5.0 for v in ds["score"]]


def load_korean_klue_sts():
    ds = load_dataset("klue/klue", "sts", split="validation")
    return ds["sentence1"], ds["sentence2"], [float(x["real-label"]) / 5.0 for x in ds["labels"]]


def load_xm3600(lang):
    # The image corpus is identical across languages (same ids and files), so
    # both languages read the English copy and share one image cache.
    corpus = load_dataset("mteb/XM3600T2IRetrieval", "en-corpus", split="test")
    queries = load_dataset("mteb/XM3600T2IRetrieval", f"{lang}-queries", split="test")
    qrels = load_dataset("mteb/XM3600T2IRetrieval", f"{lang}-qrels", split="test")
    relevant = {}
    for row in qrels:
        if row["score"] > 0:
            relevant.setdefault(row["query-id"], {})[row["corpus-id"]] = row["score"]
    return {
        "name": f"xm3600-{lang}",
        "image_set": "xm3600",
        "queries": [{"id": q["id"], "text": q["text"]} for q in queries if q["id"] in relevant],
        "corpus_ids": corpus["id"],
        "corpus_images": LazyImages(corpus, "image"),
        "relevant": relevant,
    }


def load_xm3600_en():
    return load_xm3600("en")


def load_xm3600_ko():
    return load_xm3600("ko")


def load_beir_images(name, repo, id_col, qid_col, query_col):
    corpus = load_dataset(repo, "corpus", split="test")
    queries = load_dataset(repo, "queries", split="test")
    qrels = load_dataset(repo, "qrels", split="test")
    qrel_q = "query-id" if "query-id" in qrels.column_names else "query_id"
    qrel_c = "corpus-id" if "corpus-id" in qrels.column_names else "corpus_id"
    relevant = {}
    for row in qrels:
        if row["score"] > 0:
            relevant.setdefault(row[qrel_q], {})[row[qrel_c]] = row["score"]
    return {
        "name": name,
        "image_set": name,
        "queries": [{"id": q[qid_col], "text": q[query_col]} for q in queries if q[qid_col] in relevant],
        "corpus_ids": corpus[id_col],
        "corpus_images": LazyImages(corpus, "image"),
        "relevant": relevant,
    }


def load_vidore_docvqa():
    return load_beir_images("vidore-docvqa", "vidore/docvqa_test_subsampled_beir", "corpus-id", "query-id", "query")


def load_kovidore_economic():
    return load_beir_images("kovidore-v2-economic", "whybe-choi/kovidore-v2-economic-beir", "corpus_id", "query_id", "query")


class LazyImages:
    """Decodes dataset images per slice; a full corpus would not fit in RAM."""

    def __init__(self, dataset, column):
        self.dataset = dataset
        self.column = column

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, s):
        return self.dataset[s][self.column]


BENCHMARK_LOADERS = {
    "load_english_stsb": load_english_stsb,
    "load_korean_klue_sts": load_korean_klue_sts,
    "load_xm3600_en": load_xm3600_en,
    "load_xm3600_ko": load_xm3600_ko,
    "load_vidore_docvqa": load_vidore_docvqa,
    "load_kovidore_economic": load_kovidore_economic,
}


def load_cache():
    if not CACHE_PATH.exists():
        return {}
    with CACHE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache):
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)


def config_signature(cfg):
    return {k: cfg.get(k) for k in ("kind", "name", "variant", "model_kwargs", "query_prompt")}


def cache_key(cfg, benchmark):
    payload = {
        "implementation_version": IMPLEMENTATION_VERSIONS[cfg["kind"]],
        "config": config_signature(cfg),
        "benchmark": benchmark,
        "preprocess": PREPROCESS,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def display_name(cfg):
    return f"{cfg['name']} ({cfg['variant']})" if "variant" in cfg else cfg["name"]


NOTES = """## 참고

- 로컬 모델(google/embeddinggemma-2, Qwen/Qwen3-VL-Embedding-2B)은 RTX 3090 에서 bfloat16, sentence-transformers 로 돌렸습니다.
- nvidia/llama-nemotron-embed-vl-1b-v2 는 OpenRouter 무료 엔드포인트(provider: Nvidia)로 돌렸습니다. 텍스트는 `input_type` (retrieval query=`query`, STS=`passage`)으로 모델의 `query: `/`passage: ` prefix 를 서버가 붙이고, 이미지는 `input_type` 없이 보냈습니다 (로컬 HF 모델의 document 임베딩과 cos 0.99 로 일치함을 확인).
- gemini-embedding-2 는 Gemini API(google-genai), voyage-multimodal-3.5 는 Voyage API 로 돌렸습니다. 둘 다 기본 출력 차원(3072 / 1024)입니다.
- 모든 이미지는 긴 변 1280px 이하로 줄이고 JPEG(q90)로 다시 인코딩해 모든 모델에 같은 픽셀을 넣었습니다 (XM3600 은 원본이 640px 이하라 리사이즈 없음). 그 뒤의 해상도/타일링은 각 모델의 기본값입니다 (EmbeddingGemma 2 는 이미지당 280 토큰).
- query prefix: EmbeddingGemma 2 와 Gemini 는 `task: search result | query: `, Qwen3-VL-Embedding 은 instruction 으로 XM3600 에 `Find me an everyday image that matches the given caption.`, 문서 검색에 `Find a document page image that answers the user's question.`, Voyage 는 `input_type=query`/`document` 를 썼습니다. 이미지에는 prefix 를 붙이지 않았습니다. STS 는 이전 글과 같이 prefix 없이 원문을 넣었습니다 (Nemotron 은 서버가 prefix 를 항상 붙이므로 `passage`).
- XM3600 은 이미지 3,600장을 영어/한국어 캡션과 짝지은 데이터셋이고, I2T 는 같은 임베딩으로 이미지→캡션 방향을 계산한 값입니다 (그 이미지의 캡션 중 하나라도 top-k 에 있으면 hit, 캡션은 query prompt 로 인코딩).
- Recall@k 는 정답이 top-k 안에 하나라도 있는 비율, nDCG@5 는 binary relevance 기준입니다.
- 인코딩 시간은 배치별 소요 시간의 합(재시도/대기 포함)이고, API 는 병렬 요청(OpenRouter 2, Gemini/Voyage 4)이었으므로 실제 경과 시간은 이보다 짧습니다. 로컬 모델은 같은 GPU 에서 다른 작업이 함께 돌던 상태라 참고용입니다.
- 비용(USD)은 Voyage 는 응답 usage 와 공시 단가, Gemini 는 usage 를 주지 않아 공시 단가(이미지당 $0.00012, 텍스트 $0.20/1M, 토큰 수는 Gemma 토크나이저로 추정)로 계산했습니다. Voyage 계정 무료 할당량이 적용되면 실제 청구액은 더 적습니다. 위 표 외에 스모크 테스트와 아래 진단으로 약 $0.07 를 더 썼습니다.
- Gemini 의 문서 검색 점수가 낮은 것이 1280px 리사이즈 때문인지 보려고 ViDoRe DocVQA 를 원본 해상도로 다시 돌려 봤는데, Recall@1 0.2195 → 0.2439, nDCG@5 0.2933 → 0.3157 로 차이가 작았습니다.
- STS 에 prefix 를 넣지 않았기 때문에, `task: sentence similarity | query: ` 같은 prefix 를 권장하는 EmbeddingGemma 2 / Gemini 는 STS 점수가 실제 권장 사용법보다 낮게 나왔을 수 있습니다.
"""


def write_results(sts_rows, retrieval_rows):
    with RESULT_PATH.open("w", encoding="utf-8") as f:
        f.write("# Multimodal Embedding Benchmark\n\n")
        f.write("## STS\n\n")
        f.write("| Model | Dataset | Samples | Spearman | Elapsed(s) | Avg sec/pair |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for r in sts_rows:
            f.write(
                f"| {r['model']} | {r['dataset']} | {r['samples']} | {r['spearman']:.4f} | "
                f"{r['elapsed_s']:.2f} | {r['sec_per_pair']:.5f} |\n"
            )

        f.write("\n## Retrieval\n\n")
        f.write("| Model | Dataset | Queries | Corpus | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | Avg sec/text | Avg sec/image |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in retrieval_rows:
            f.write(
                f"| {r['model']} | {r['dataset']} | {r['queries']} | {r['corpus']} | "
                f"{r['recall@1']:.4f} | {r['recall@5']:.4f} | {r['recall@10']:.4f} | {r['mrr@10']:.4f} | {r['ndcg@5']:.4f} | "
                f"{r['sec_per_query_text']:.5f} | {r['sec_per_image']:.5f} |\n"
            )

        # Each embedding is paid once: I2T reuses the T2I embeddings, and the
        # XM3600 image corpus is shared by both caption languages.
        costs, seen = {}, set()
        for r in sts_rows:
            costs[r["model"]] = costs.get(r["model"], 0.0) + r["cost_usd"]
        for r in retrieval_rows:
            if r["dataset"].endswith("I2T)"):
                continue
            cost = r["query_cost_usd"]
            if (r["model"], r["image_set"]) not in seen:
                seen.add((r["model"], r["image_set"]))
                cost += r["corpus_cost_usd"]
            costs[r["model"]] = costs.get(r["model"], 0.0) + cost
        f.write("\n## Cost (USD)\n\n| Model | Cost |\n|---|---:|\n")
        for model, cost in costs.items():
            f.write(f"| {model} | {cost:.4f} |\n")
        f.write(f"| **Total** | **{sum(costs.values()):.4f}** |\n")

        f.write("\n" + NOTES)


if __name__ == "__main__":
    loaded = {}
    cache = load_cache()
    sts_rows, retrieval_rows = [], []

    only = os.environ.get("BENCH_ONLY")
    for cfg in CONFIGS:
        name = display_name(cfg)
        if only and not any(o in name for o in only.split(",")):
            continue
        encoder = None
        try:
            for benchmark in STS_BENCHMARKS + RETRIEVAL_BENCHMARKS:
                key = cache_key(cfg, benchmark)
                if key in cache:
                    result = dict(cache[key])
                    print("Using cached result for", name, benchmark["name"], flush=True)
                else:
                    if encoder is None:
                        print("Loading", name, flush=True)
                        encoder = build_encoder(cfg)
                    if benchmark["loader"] not in loaded:
                        loaded[benchmark["loader"]] = BENCHMARK_LOADERS[benchmark["loader"]]()
                    data = loaded[benchmark["loader"]]
                    if benchmark["type"] == "sts":
                        result = eval_sts(encoder, cfg, benchmark, *data)
                    else:
                        result = eval_retrieval(encoder, cfg, benchmark, data)
                    cache[key] = result
                    save_cache(cache)

                result.update(model=name, dataset=benchmark["name"])
                (sts_rows if benchmark["type"] == "sts" else retrieval_rows).append(result)
                print(name, benchmark["name"], result, flush=True)
        finally:
            if encoder is not None:
                encoder.close()
            print(f"API spend so far: ${spent_usd():.4f}", flush=True)

    write_results(sts_rows, retrieval_rows)
