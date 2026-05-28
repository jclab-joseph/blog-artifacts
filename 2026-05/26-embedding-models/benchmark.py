import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from datasets import load_dataset
from scipy.stats import spearmanr
from huggingface_hub import hf_hub_download

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

import torch

CONFIGS = [
    {"name": "jhgan/ko-sbert-sts", "kind": "sbert"},
    {"name": "snunlp/KR-SBERT-V40K-klueNLI-augSTS", "kind": "sbert"},
    {"name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "kind": "sbert"},
    {
        "name": "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
        "kind": "onnx",
        "file_name": "onnx/model_fp16.onnx",
        "variant": "fp16",
        "tokenizer_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    },
    {
        "name": "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
        "kind": "onnx",
        "file_name": "onnx/model_int8.onnx",
        "variant": "int8",
        "tokenizer_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    },
    {"name": "Xenova/all-MiniLM-L12-v2", "kind": "onnx", "file_name": "onnx/model.onnx", "variant": "original"},
    {"name": "Xenova/all-MiniLM-L12-v2", "kind": "onnx", "file_name": "onnx/model_int8.onnx", "variant": "int8"},
    # {
    #     "name": "google/embeddinggemma-300m",
    #     "kind": "sbert",
    #     "variant": "original",
    #     "query_prefix": "query: ",
    #     "passage_prefix": "title: \"none\" | text: ",
    # },
    {
        "name": "unsloth/embeddinggemma-300m-GGUF",
        "kind": "gguf",
        "file_name": "embeddinggemma-300M-Q8_0.gguf",
        "variant": "Q8_0",
        "tokenizer_name": "google/embeddinggemma-300m",
        "query_prefix": "query: ",
        "passage_prefix": "title: \"none\" | text: ",
    },
    {
        "name": "unsloth/embeddinggemma-300m-GGUF",
        "kind": "gguf",
        "file_name": "embeddinggemma-300m-Q4_0.gguf",
        "variant": "Q4_0",
        "tokenizer_name": "google/embeddinggemma-300m",
        "query_prefix": "query: ",
        "passage_prefix": "title: \"none\" | text: ",
    },
    {"name": "sentence-transformers/distiluse-base-multilingual-cased-v2", "kind": "sbert", "variant": "original"},
    {"name": "Xenova/distiluse-base-multilingual-cased-v2", "kind": "onnx", "file_name": "onnx/model_int8.onnx", "variant": "int8"},
    # {
    #     "name": "exp-models/dragonkue-KoEn-E5-Tiny",
    #     "kind": "onnx",
    #     "file_name": "onnx/model.onnx",
    #     "variant": "original",
    #     "query_prefix": "query: ",
    #     "passage_prefix": "passage: ",
    # },
    # {
    #     "name": "exp-models/dragonkue-KoEn-E5-Tiny", "kind": "gguf", "file_name": "ggml-model-q8_0.gguf", "variant": "Q8_0",
    #     "query_prefix": "query: ",
    #     "passage_prefix": "passage: ",
    # }, # not working
    # {
    #     "name": "exp-models/dragonkue-KoEn-E5-Tiny-ONNX",
    #     "kind": "onnx",
    #     "file_name": "onnx/model_qint8_arm64.onnx",
    #     "variant": "int8",
    #     "query_prefix": "query: ",
    #     "passage_prefix": "passage: ",
    # }
    {
        "name": "jc-lab/multilingual-e5-small-ko-v2-gguf",
        "kind": "gguf",
        "file_name": "ggml-model-q8_0.gguf",
        "variant": "Q8_0",
        "tokenizer_name": "dragonkue/multilingual-e5-small-ko-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    {
        "name": "jc-lab/multilingual-e5-small-ko-v2-gguf",
        "kind": "gguf",
        "file_name": "ggml-model-q4_k_m.gguf",
        "variant": "Q4_K_M",
        "tokenizer_name": "dragonkue/multilingual-e5-small-ko-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
]

IMPLEMENTATION_VERSIONS = {
    "sbert": 1,
    "onnx": 1,
    "gguf": 2,
}

CACHE_PATH = Path(__file__).with_name("benchmark_cache.json")
RESULT_PATH = Path(__file__).with_name("benchmark_results.md")

STS_BENCHMARKS = [
    {"name": "English(STS-B val)", "type": "sts", "loader": "load_english_stsb"},
    {"name": "Korean(KLUE-STS val)", "type": "sts", "loader": "load_korean_klue_sts"},
]

RETRIEVAL_BENCHMARKS = [
    {
        "name": "English(MSMARCO Passage Ranking top250 test)",
        "type": "retrieval",
        "loader": "load_msmarco_passage_ranking",
    },
    {
        "name": "Korean(MIRACL-ko top250 train)",
        "type": "retrieval",
        "loader": "load_miracl_ko",
    },
]


def cosine_sim(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a * b, axis=1)

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask[..., None].astype(np.float32)
    summed = (last_hidden_state * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts

class SBertEncoder:
    def __init__(self, model_name, file_name=None, tokenizer_name=None):
        model_kwargs = {}
        st_name = tokenizer_name or model_name

        if file_name is not None:
            model_kwargs["gguf_file"] = hf_hub_download(repo_id=model_name, filename=file_name)

        self.model = SentenceTransformer(st_name, model_kwargs=model_kwargs)
        self.tokenizer = self.model.tokenizer

    def encode(self, texts, batch_size=64):
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)

    def close(self):
        pass


class ONNXEncoder:
    def __init__(self, model_name, file_name, tokenizer_name=None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        model_path = hf_hub_download(repo_id=model_name, filename=file_name)

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def encode(self, texts, batch_size=64):
        vecs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="np",
            )

            feeds = {k: enc[k] for k in self.input_names if k in enc}

            if "token_type_ids" in self.input_names and "token_type_ids" not in feeds:
                feeds["token_type_ids"] = np.zeros_like(enc["input_ids"], dtype=np.int64)

            outputs = self.session.run(None, feeds)

            selected = outputs[0]

            if selected.ndim == 3:
                pooled = mean_pool(selected, enc["attention_mask"])
            elif selected.ndim == 2:
                pooled = selected
            else:
                raise ValueError(f"Unexpected output shape: {selected.shape}")

            vecs.append(pooled)

        return np.vstack(vecs)

    def close(self):
        self.session = None


class GGUFEncoder:
    def __init__(self, model_name, file_name, tokenizer_name=None):
        try:
            import llama_cpp
        except ImportError as exc:
            raise ImportError("GGUF benchmarking requires llama-cpp-python to be installed.") from exc

        self.max_length = 256
        self.tokenizer = None
        model_path = hf_hub_download(repo_id=model_name, filename=file_name)
        self.model = llama_cpp.Llama(
            model_path=model_path,
            embedding=True,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_MEAN,
            n_ctx=512,
            n_batch=self.max_length,
            n_ubatch=self.max_length,
            verbose=False,
            n_gpu_layers=-1,
        )
        metadata = self.model.metadata or {}
        add_bos_value = metadata.get("tokenizer.ggml.add_bos_token", "true")
        self.add_bos_token = str(add_bos_value).lower() == "true"

    def _truncate_texts(self, texts):
        truncated = []

        for text in texts:
            input_ids = self.model.tokenize(
                text.encode("utf-8"),
                add_bos=self.add_bos_token,
                special=False,
            )[:self.max_length]

            if self.add_bos_token and input_ids:
                input_ids = input_ids[1:]

            truncated.append(
                self.model.detokenize(input_ids).decode("utf-8", errors="ignore")
            )

        return truncated

    def count_tokens(self, texts):
        total = 0
        for text in texts:
            total += len(
                self.model.tokenize(
                    text.encode("utf-8"),
                    add_bos=self.add_bos_token,
                    special=False,
                )[:self.max_length]
            )
        return total

    def encode(self, texts, batch_size=64):
        vecs = []

        for i in range(0, len(texts), batch_size):
            batch = self._truncate_texts(texts[i:i + batch_size])

            result = self.model.create_embedding(batch)
            data = sorted(result["data"], key=lambda x: x["index"])
            embeddings = [item["embedding"] for item in data]

            vecs.append(np.asarray(embeddings, dtype=np.float32))

        return np.vstack(vecs)

    def close(self):
        if self.model is not None:
            self.model.close()
            self.model = None


def total_tokens(tokenizer, s1, s2):
    c1 = sum(len(tokenizer(x, truncation=True, max_length=256)["input_ids"]) for x in s1)
    c2 = sum(len(tokenizer(x, truncation=True, max_length=256)["input_ids"]) for x in s2)
    return c1 + c2


def eval_dataset(encoder, s1, s2, y, batch_size=64):
    t0 = time.perf_counter()
    e1 = encoder.encode(s1, batch_size=batch_size)
    e2 = encoder.encode(s2, batch_size=batch_size)
    elapsed = time.perf_counter() - t0

    rho = float(spearmanr(cosine_sim(e1, e2), y).correlation)
    if hasattr(encoder, "count_tokens"):
        tokens = encoder.count_tokens(s1) + encoder.count_tokens(s2)
    else:
        tokens = total_tokens(encoder.tokenizer, s1, s2)
    return {"samples": len(s1), "spearman": rho, "elapsed_s": elapsed, "tokens": tokens, "tokens_per_s": tokens / elapsed, "sec_per_pair": elapsed / len(s1)}


def load_english_stsb():
    ds = load_dataset("mteb/stsbenchmark-sts", split="validation")
    return ds["sentence1"], ds["sentence2"], [float(v) / 5.0 for v in ds["score"]]


def load_korean_klue_sts():
    ds = load_dataset("klue/klue", "sts", split="validation")
    return ds["sentence1"], ds["sentence2"], [float(x["real-label"]) / 5.0 for x in ds["labels"]]


def load_retrieval_benchmark(dataset_name, qrels_config, queries_config, corpus_config, split):
    qrels_ds = load_dataset(dataset_name, qrels_config, split=split)
    query_ds = load_dataset(dataset_name, queries_config, split=split)
    corpus_ds = load_dataset(dataset_name, corpus_config, split=split)

    queries = {row["_id"]: row["text"] for row in query_ds}
    corpus = {row["_id"]: row for row in corpus_ds}
    relevant = {}

    for row in qrels_ds:
        if row["score"] <= 0:
            continue
        qid = row["query-id"]
        cid = row["corpus-id"]
        if qid not in queries or cid not in corpus:
            continue
        relevant.setdefault(qid, set()).add(cid)

    query_ids = [qid for qid in queries if qid in relevant]

    documents = []
    for cid, row in corpus.items():
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        documents.append({"id": cid, "text": f"{title}\n{text}".strip() if title else text})

    return {
        "queries": [{"id": qid, "text": queries[qid]} for qid in query_ids],
        "documents": documents,
        "relevant": {qid: sorted(relevant[qid]) for qid in query_ids},
    }


def load_msmarco_passage_ranking():
    return load_retrieval_benchmark(
        "mteb/MSMARCO_test_top_250_only_w_correct",
        "default",
        "queries",
        "corpus",
        "test",
    )


def load_miracl_ko():
    return load_retrieval_benchmark(
        "mteb/MIRACLRetrieval_ko_top_250_only_w_correct",
        "default",
        "queries",
        "corpus",
        "train",
    )


def load_cache():
    if not CACHE_PATH.exists():
        return {}
    with CACHE_PATH.open("r", encoding="utf-8") as f:
        raw_cache = json.load(f)

    normalized_cache = {}
    changed = False

    for raw_key, value in raw_cache.items():
        try:
            payload = json.loads(raw_key)
        except json.JSONDecodeError:
            normalized_cache[raw_key] = value
            continue

        config = payload.get("config", {})
        if "query_prefix" not in config:
            config["query_prefix"] = None
            changed = True
        if "passage_prefix" not in config:
            config["passage_prefix"] = None
            changed = True

        normalized_key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if normalized_key != raw_key:
            changed = True
        normalized_cache[normalized_key] = value

    if changed:
        save_cache(normalized_cache)

    return normalized_cache


def save_cache(cache):
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)


def config_signature(cfg):
    return {
        "kind": cfg["kind"],
        "name": cfg["name"],
        "variant": cfg.get("variant"),
        "file_name": cfg.get("file_name"),
        "tokenizer_name": cfg.get("tokenizer_name"),
        "query_prefix": cfg.get("query_prefix"),
        "passage_prefix": cfg.get("passage_prefix"),
    }


def cache_key(cfg, benchmark):
    payload = {
        "implementation_version": IMPLEMENTATION_VERSIONS[cfg["kind"]],
        "config": config_signature(cfg),
        "benchmark": benchmark,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def apply_prefix(texts, prefix):
    if not prefix:
        return texts
    return [f"{prefix}{text}" for text in texts]


def eval_retrieval_dataset(encoder, benchmark, cfg, batch_size=64):
    queries = benchmark["queries"]
    documents = benchmark["documents"]
    relevant = benchmark["relevant"]
    query_texts = apply_prefix([row["text"] for row in queries], cfg.get("query_prefix"))
    document_texts = apply_prefix([row["text"] for row in documents], cfg.get("passage_prefix"))

    t0 = time.perf_counter()
    query_embeddings = encoder.encode(query_texts, batch_size=batch_size)
    doc_embeddings = encoder.encode(document_texts, batch_size=batch_size)
    elapsed = time.perf_counter() - t0

    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    scores = query_embeddings @ doc_embeddings.T

    doc_ids = [row["id"] for row in documents]
    query_ids = [row["id"] for row in queries]
    recall_hits = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks = []

    for index, qid in enumerate(query_ids):
        ranking = np.argsort(-scores[index])[:5]
        top_doc_ids = [doc_ids[pos] for pos in ranking]
        gold = set(relevant[qid])

        for k in recall_hits:
            if any(doc_id in gold for doc_id in top_doc_ids[:k]):
                recall_hits[k] += 1

        rr = 0.0
        for rank, doc_id in enumerate(top_doc_ids, start=1):
            if doc_id in gold:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    total = len(query_ids)
    # tokens = total_tokens(
    #     encoder.tokenizer,
    #     query_texts,
    #     document_texts,
    # )
    return {
        "queries": total,
        "documents": len(documents),
        "recall@1": recall_hits[1] / total,
        "recall@3": recall_hits[3] / total,
        "recall@5": recall_hits[5] / total,
        "mrr@5": float(np.mean(reciprocal_ranks)),
        "elapsed_s": elapsed,
        "tokens": -1,
        "tokens_per_s": 0 / elapsed,
        "sec_per_query": elapsed / total,
    }


def build_encoder(cfg):
    if cfg["kind"] == "sbert":
        return SBertEncoder(
            cfg["name"],
            file_name=cfg.get("file_name", None),
            tokenizer_name=cfg.get("tokenizer_name", None),
        )
    if cfg["kind"] == "gguf":
        return GGUFEncoder(
            cfg["name"],
            cfg["file_name"],
            tokenizer_name=cfg.get("tokenizer_name", None),
        )

    return ONNXEncoder(
        cfg["name"],
        cfg["file_name"],
        tokenizer_name=cfg.get("tokenizer_name", None),
    )


def display_name(cfg):
    return f"{cfg['name']} ({cfg['variant']})" if "variant" in cfg else cfg["name"]


def write_results(sts_rows, retrieval_rows):
    with RESULT_PATH.open("w", encoding="utf-8") as f:
        f.write("# Embedding Benchmark\n\n")
        f.write("## STS\n\n")
        f.write("| Model | Dataset | Samples | Spearman | Elapsed(s) | Total tokens | Tokens/s | Avg sec/pair |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in sts_rows:
            f.write(
                f"| {r['model']} | {r['dataset']} | {r['samples']} | {r['spearman']:.4f} | "
                f"{r['elapsed_s']:.2f} | {r['tokens']} | {r['tokens_per_s']:.2f} | {r['sec_per_pair']:.5f} |\n"
            )

        f.write("\n## Retrieval\n\n")
        f.write("| Model | Dataset | Queries | Docs | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Elapsed(s) | Total tokens | Tokens/s | Avg sec/query |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in retrieval_rows:
            f.write(
                f"| {r['model']} | {r['dataset']} | {r['queries']} | {r['documents']} | "
                f"{r['recall@1']:.4f} | {r['recall@3']:.4f} | {r['recall@5']:.4f} | {r['mrr@5']:.4f} | "
                f"{r['elapsed_s']:.2f} | {r['tokens']} | {r['tokens_per_s']:.2f} | {r['sec_per_query']:.5f} |\n"
            )


if __name__ == "__main__":
    benchmark_loaders = {
        "load_english_stsb": load_english_stsb,
        "load_korean_klue_sts": load_korean_klue_sts,
        "load_msmarco_passage_ranking": load_msmarco_passage_ranking,
        "load_miracl_ko": load_miracl_ko,
    }
    loaded_benchmarks = {}
    cache = load_cache()
    sts_rows = []
    retrieval_rows = []

    for cfg in CONFIGS:
        name = display_name(cfg)
        print("Loading", name, flush=True)
        encoder = build_encoder(cfg)
        try:
            for benchmark in STS_BENCHMARKS + RETRIEVAL_BENCHMARKS:
                benchmark_name = benchmark["name"]
                key = cache_key(cfg, benchmark)

                if key in cache:
                    result = dict(cache[key])
                    print("Using cached result for", name, benchmark_name, flush=True)
                else:
                    loader_name = benchmark["loader"]
                    if loader_name not in loaded_benchmarks:
                        loaded_benchmarks[loader_name] = benchmark_loaders[loader_name]()
                    loaded = loaded_benchmarks[loader_name]

                    if benchmark["type"] == "sts":
                        result = eval_dataset(encoder, *loaded)
                    else:
                        result = eval_retrieval_dataset(encoder, loaded, cfg)

                    cache[key] = result
                    save_cache(cache)

                result.update(model=name, dataset=benchmark_name)
                if benchmark["type"] == "sts":
                    sts_rows.append(result)
                else:
                    retrieval_rows.append(result)
                print(name, benchmark_name, result, flush=True)
        finally:
            encoder.close()

    write_results(sts_rows, retrieval_rows)
