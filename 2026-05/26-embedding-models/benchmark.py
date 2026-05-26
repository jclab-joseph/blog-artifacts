import time
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
    # {
    #     "name": "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
    #     "kind": "onnx",
    #     "file_name": "onnx/model_fp16.onnx",
    #     "variant": "fp16",
    #     "tokenizer_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # },
    # {
    #     "name": "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
    #     "kind": "onnx",
    #     "file_name": "onnx/model_int8.onnx",
    #     "variant": "int8",
    #     "tokenizer_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # },
    # {"name": "Xenova/all-MiniLM-L12-v2", "kind": "onnx", "file_name": "onnx/model.onnx", "variant": "original"},
    # {"name": "Xenova/all-MiniLM-L12-v2", "kind": "onnx", "file_name": "onnx/model_fp16.onnx", "variant": "fp16"},
    # {"name": "Xenova/all-MiniLM-L12-v2", "kind": "onnx", "file_name": "onnx/model_int8.onnx", "variant": "int8"},
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
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer

    def encode(self, texts, batch_size=64):
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)


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

        print("inputs:", [(i.name, i.shape, i.type) for i in self.session.get_inputs()])
        print("outputs:", [(o.name, o.shape, o.type) for o in self.session.get_outputs()])

    def encode(self, texts, batch_size=64):
        vecs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )

            feeds = {k: enc[k] for k in self.input_names if k in enc}

            if "token_type_ids" in self.input_names and "token_type_ids" not in feeds:
                feeds["token_type_ids"] = np.zeros_like(enc["input_ids"], dtype=np.int64)

            outputs = self.session.run(None, feeds)

            for meta, value in zip(self.session.get_outputs(), outputs):
                print(meta.name, value.shape)

            selected = outputs[0]
            print("selected : ", selected.ndim)

            if selected.ndim == 3:
                pooled = mean_pool(selected, enc["attention_mask"])
            elif selected.ndim == 2:
                pooled = selected
            else:
                raise ValueError(f"Unexpected output shape: {selected.shape}")

            vecs.append(pooled)

        return np.vstack(vecs)


def total_tokens(tokenizer, s1, s2):
    c1 = sum(len(tokenizer(x, truncation=True, max_length=512)["input_ids"]) for x in s1)
    c2 = sum(len(tokenizer(x, truncation=True, max_length=512)["input_ids"]) for x in s2)
    return c1 + c2


def eval_dataset(encoder, s1, s2, y, batch_size=64):
    t0 = time.perf_counter()
    e1 = encoder.encode(s1, batch_size=batch_size)
    e2 = encoder.encode(s2, batch_size=batch_size)
    elapsed = time.perf_counter() - t0

    rho = float(spearmanr(cosine_sim(e1, e2), y).correlation)
    tokens = total_tokens(encoder.tokenizer, s1, s2)
    return {"samples": len(s1), "spearman": rho, "elapsed_s": elapsed, "tokens": tokens, "tokens_per_s": tokens / elapsed, "sec_per_pair": elapsed / len(s1)}


def load_english_stsb():
    ds = load_dataset("mteb/stsbenchmark-sts", split="validation")
    return ds["sentence1"], ds["sentence2"], [float(v) / 5.0 for v in ds["score"]]


def load_korean_klue_sts():
    ds = load_dataset("klue/klue", "sts", split="validation")
    return ds["sentence1"], ds["sentence2"], [float(x["real-label"]) / 5.0 for x in ds["labels"]]


def build_encoder(cfg):
    if cfg["kind"] == "sbert":
        return SBertEncoder(cfg["name"])

    return ONNXEncoder(
        cfg["name"],
        cfg["file_name"],
        tokenizer_name=cfg.get("tokenizer_name", None),
    )


def display_name(cfg):
    return f"{cfg['name']} ({cfg['variant']})" if cfg["kind"] == "onnx" else cfg["name"]


if __name__ == "__main__":
    en = load_english_stsb()
    ko = load_korean_klue_sts()
    rows = []

    for cfg in CONFIGS:
        name = display_name(cfg)
        print("Loading", name, flush=True)
        encoder = build_encoder(cfg)
        for dname, data in [("English(STS-B val)", en), ("Korean(KLUE-STS val)", ko)]:
            result = eval_dataset(encoder, *data)
            result.update(model=name, dataset=dname)
            rows.append(result)
            print(name, dname, result, flush=True)

    with open("benchmark_results.md", "w", encoding="utf-8") as f:
        f.write("# Embedding STS Benchmark\n\n")
        f.write("| Model | Dataset | Samples | Spearman | Elapsed(s) | Total tokens | Tokens/s | Avg sec/pair |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['model']} | {r['dataset']} | {r['samples']} | {r['spearman']:.4f} | {r['elapsed_s']:.2f} | {r['tokens']} | {r['tokens_per_s']:.2f} | {r['sec_per_pair']:.5f} |\n")
