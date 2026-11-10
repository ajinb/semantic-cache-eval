# semantic-cache-eval

How often does a semantic cache answer the wrong question? Threshold sweeps on 5,000 Quora Question Pairs and 2,000 PAWS pairs, precision adjusted for how often users actually repeat themselves, the effect of cache size, and a set of cheap guards.

Companion to [Your Semantic Cache Is Answering the Wrong Question](https://cloudandsre.com/blog/semantic-cache-false-positives-measured/).

## Method

Two embedding models through fastembed (`BAAI/bge-small-en-v1.5`, `all-MiniLM-L6-v2`), cosine thresholds from 0.70 to 0.99.

- **Quora** pairs are natural: duplicates and near-misses that real users wrote.
- **PAWS** pairs are adversarial: same words, different order, different meaning. This is the case a bag-of-meanings embedding is worst at.
- **Cache-size effect**: 1,000 fresh questions against caches of 1k, 5k and 20k other questions, counting how often anything clears the threshold.
- **Guards** (`guard.py`, standard library): refuse a hit when the numbers, negation, proper nouns or word order differ.

Precision is reported at three base rates (5%, 20%, 50% of traffic being a true repeat), because the share of wrong answers served depends on it as much as on the threshold.

## Results

| Dataset | Model | AUC | Threshold | TPR | FPR | Precision at 5% / 20% / 50% repeats |
|---|---|---:|---:|---:|---:|---|
| quora | bge-small-en-v1.5 | 0.88 | 0.85 | 81% | 23.1% | 16% / 47% / 78% |
| quora | bge-small-en-v1.5 | 0.88 | 0.90 | 58% | 10.9% | 22% / 57% / 84% |
| quora | bge-small-en-v1.5 | 0.88 | 0.95 | 28% | 2.5% | 37% / 74% / 92% |
| paws | bge-small-en-v1.5 | 0.60 | 0.85 | 99% | 99.6% | 5% / 20% / 50% |
| paws | bge-small-en-v1.5 | 0.60 | 0.90 | 98% | 98.3% | 5% / 20% / 50% |
| paws | bge-small-en-v1.5 | 0.60 | 0.95 | 92% | 88.7% | 5% / 21% / 51% |
| quora | all-MiniLM-L6-v2 | 0.87 | 0.85 | 60% | 13.1% | 20% / 54% / 82% |
| quora | all-MiniLM-L6-v2 | 0.87 | 0.90 | 42% | 6.8% | 25% / 61% / 86% |
| quora | all-MiniLM-L6-v2 | 0.87 | 0.95 | 22% | 2.0% | 36% / 73% / 91% |
| paws | all-MiniLM-L6-v2 | 0.62 | 0.85 | 98% | 96.4% | 5% / 20% / 50% |
| paws | all-MiniLM-L6-v2 | 0.62 | 0.90 | 95% | 91.4% | 5% / 21% / 51% |
| paws | all-MiniLM-L6-v2 | 0.62 | 0.95 | 86% | 75.2% | 6% / 22% / 53% |

With the guards in `guard.py` (numbers, negation, proper nouns, word order), bge-small on Quora at 0.90 moves from TPR 58% / FPR 10.9% to TPR 44% / FPR 6.6%; on PAWS at 0.90 the FPR falls from 98% to 51%.

| Cache size | Fresh questions clearing 0.85 | 0.90 | 0.95 |
|---:|---:|---:|---:|
| 1,000 | 0.7% | 0.4% | 0.2% |
| 5,000 | 2.1% | 1.1% | 0.4% |
| 20,000 | 10.1% | 3.4% | 0.9% |

Full sweeps from 0.70 to 0.99 in [`results/thresholds.json`](results/thresholds.json), and 60 sampled borderline pairs in [`results/cache_size.json`](results/cache_size.json).

## Run it

```bash
pip install -r requirements-ci.txt
pytest -q
python bench.py          # downloads both datasets as parquet from Hugging Face
```

## Limits

Question-pair datasets are a proxy for cache traffic. Thresholds are model-specific and don't transfer. The guards are English-only heuristics.

## License

Apache-2.0
