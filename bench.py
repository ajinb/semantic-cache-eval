#!/usr/bin/env python3
"""How often does a semantic cache answer the wrong question? Writes results/.

Datasets: Quora Question Pairs (natural duplicates) and PAWS (adversarial non-duplicates that share
almost all their words). Models come through fastembed and run locally.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from fastembed import TextEmbedding

import guard

BASE = "https://huggingface.co/datasets/{}/resolve/refs%2Fconvert%2Fparquet/{}/{}/0000.parquet"
SOURCES = {
    "quora": ("sentence-transformers/quora-duplicates", "pair-class", "train"),
    "paws": ("google-research-datasets/paws", "labeled_final", "test"),
}
MODELS = ["BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"]
THRESHOLDS = [round(x, 2) for x in np.arange(0.70, 1.0, 0.01)]
BASE_RATES = (0.05, 0.2, 0.5)


def fetch(name: str):
    repo, config, split = SOURCES[name]
    path = Path("data") / f"{name}.parquet"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        with urllib.request.urlopen(BASE.format(repo, config, split), timeout=300) as response:  # noqa: S310
            path.write_bytes(response.read())
    table = pq.read_table(path).to_pylist()
    first, second = ("sentence1", "sentence2")
    return [(row[first], row[second], int(row["label"])) for row in table]


def similarity(model, a: list[str], b: list[str]) -> np.ndarray:
    x = np.array(list(model.passage_embed(a, batch_size=128)))
    y = np.array(list(model.passage_embed(b, batch_size=128)))
    return (x * y).sum(axis=1)


def sweep(sim, labels, allowed):
    positive, negative = labels == 1, labels == 0
    rows = []
    for t in THRESHOLDS:
        above = (sim >= t) & allowed
        tpr, fpr = float(above[positive].mean()), float(above[negative].mean())
        row = {"threshold": t, "tpr": round(tpr, 4), "fpr": round(fpr, 4)}
        for pi in BASE_RATES:
            denominator = tpr * pi + fpr * (1 - pi)
            row[f"precision_at_{int(pi * 100)}pct_repeats"] = (
                round(tpr * pi / denominator, 4) if denominator else None
            )
        rows.append(row)
    return rows


def auc(sim, labels) -> float:
    ranks = np.argsort(np.argsort(sim)) + 1
    n1, n0 = int(labels.sum()), int((labels == 0).sum())
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate_pairs(name, pairs, model_name, model):
    a, b, labels = zip(*pairs, strict=True)
    labels = np.array(labels)
    sim = similarity(model, list(a), list(b))
    allowed = np.array([not guard.blocks(x, y) for x, y in zip(a, b, strict=True)])
    return {
        "dataset": name,
        "model": model_name,
        "pairs": len(pairs),
        "positive_share": round(float(labels.mean()), 3),
        "auc": round(auc(sim, labels), 4),
        "no_guard": sweep(sim, labels, np.ones_like(allowed)),
        "with_guard": sweep(sim, labels, allowed),
    }


def cache_size_effect(questions, model_name, model, rng, queries=1000):
    """Fresh questions against a growing cache: how often does something clear the threshold?"""
    pool = sorted(set(questions))
    rng.shuffle(pool)
    probes, cached = pool[:queries], pool[queries : queries + 20000]
    q = np.array(list(model.passage_embed(probes, batch_size=128)), dtype=np.float32)
    c = np.array(list(model.passage_embed(cached, batch_size=128)), dtype=np.float32)
    rows, sample = [], []
    for n in (1000, 5000, 20000):
        best = np.zeros(queries, dtype=np.float32)
        where = np.zeros(queries, dtype=int)
        for start in range(0, queries, 256):
            scores = q[start : start + 256] @ c[:n].T
            best[start : start + 256], where[start : start + 256] = scores.max(1), scores.argmax(1)
        row = {"cache_size": n}
        for t in (0.85, 0.90, 0.95):
            row[f"hit_rate_at_{t}"] = round(float((best >= t).mean()), 4)
        rows.append(row)
        if n == 20000:
            hits = [i for i in range(queries) if 0.90 <= best[i] < 0.95]
            for i in rng.sample(hits, min(60, len(hits))):
                sample.append(
                    {
                        "query": probes[i],
                        "cached": cached[where[i]],
                        "score": round(float(best[i]), 3),
                    }
                )
    return {"model": model_name, "rows": rows, "sample_0.90_to_0.95": sample}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quora-pairs", type=int, default=5000)
    parser.add_argument("--paws-pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    quora, paws = fetch("quora"), fetch("paws")
    quora_sample = rng.sample(quora, args.quora_pairs)
    paws = rng.sample(paws, min(args.paws_pairs, len(paws)))
    questions = [q for row in quora for q in row[:2]]
    results, cache = [], []
    for name in MODELS:
        model = TextEmbedding(name)
        results.append(evaluate_pairs("quora", quora_sample, name, model))
        results.append(evaluate_pairs("paws", paws, name, model))
        if name == MODELS[0]:  # the cache-size effect is about scale, one model shows it
            cache.append(cache_size_effect(questions, name, model, rng))
        print("done", name, flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/thresholds.json").write_text(json.dumps(results, indent=1) + "\n")
    Path("results/cache_size.json").write_text(
        json.dumps(cache, indent=1, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
