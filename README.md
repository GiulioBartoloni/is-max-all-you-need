# Is Max all you need? Generalizing SPLADE's pooling

> From-scratch re-implementation of **SPLADE** with a **learnable pooling family** (max · sum · p-norm · attention) — an expert-level study of whether max pooling is actually optimal for sparse neural retrieval.

Information Retrieval course project · University of Pisa · Giulio Bartoloni

---

## About

This repository re-implements the SPLADE sparse neural retriever from scratch and extends it by generalizing its **pooling function** — the step that collapses per-token vocabulary predictions into a single sparse document/query vector.

It is based on:

> T. Formal, C. Lassance, B. Piwowarski, S. Clinchant.
> *From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective.* SIGIR '22.
> arXiv:2205.04733

The project targets the **expert level** of the course: *re-implement the proposed solution, then improve it by introducing an alternative strategy.* Concretely, it:

1. **Re-implements** SPLADE (BERT + MLM head → log-saturation → pooling → sparse vector, with a MarginMSE + FLOPS training objective);
2. **Replicates** the released `splade_v2_max` baseline as a correctness check;
3. **Improves** it by replacing the fixed `max` pooling with a learnable family and studying the effect on effectiveness and efficiency.

> The official [`naver/splade`](https://github.com/naver/splade) repository is used **only as a reference** for validating this re-implementation — the model code here is written from scratch.

## Research question

SPLADE's original result showed that `max` pooling clearly beats `sum`. But `max` and `sum` are the two endpoints of a single family. This project asks:

- Is `max` actually **optimal**, or merely better than `sum`?
- Can a **learnable** aggregator (p-norm with a trainable exponent, or attention pooling) match or beat it?
- Does the best pooling **differ for queries vs documents** (short vs long inputs)?

Because pooling only shapes the representation — the output is still a sparse, inverted-index-compatible vector — this is a pure **effectiveness** lever with no added retrieval cost.

## Pooling variants

All variants live behind one interface in `src/pooling.py`:

| Variant | Params | Notes |
|---|---|---|
| `max` | 0 | Baseline — reproduces the paper. |
| `sum` | 0 | Sanity check — expected to be worse. |
| `p-norm` | 1 (learnable `p`) | Interpolates mean↔max; log-sum-exp form for numerical stability; init `p` high. |
| `attention` | many | Learned per-position weights. |
| `asymmetric` | — | Separate pooling for query vs document (`p_q`, `p_d`). |

## Repository structure

```
splade-pooling/
├── src/
│   ├── model.py        # SPLADE forward pass
│   ├── pooling.py      # swappable pooling: max / sum / p-norm / attention (+ asymmetric)
│   ├── loss.py         # MarginMSE (ranking) + FLOPS regularizer
│   ├── data.py         # DistilMSE triple loader
│   ├── train.py        # training loop + checkpointing
│   ├── index.py        # build sparse index over the collection
│   └── evaluate.py     # MRR@10 / nDCG@10 / FLOPS
├── configs/            # experiment configs (backbone, seq len, lambda, pooling)
├── notebooks/          # Kaggle training / evaluation notebooks
├── scripts/            # smoke test + helpers
├── results/            # logs, metrics, figures
├── data/               # (gitignored) MS MARCO, teacher scores
└── README.md
```

## Setup

```bash
git clone https://github.com/<you>/splade-pooling.git
cd splade-pooling
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Local development and the smoke test run on **CPU**. Training and full-collection indexing run on **Kaggle (P100, 16 GB)**.

## Data

Fixed training setting: **DistilMSE** (distillation from pre-computed cross-encoder teacher scores — no teacher is run locally).

- MS MARCO passage collection + `dev` / TREC DL 2019 query sets
- Pre-computed teacher scores / DistilMSE triples

<!-- TODO: add exact download links / Kaggle Dataset names and place under data/ (gitignored) -->

## Usage

```bash
# 0. Smoke test (CPU) — must pass before any Kaggle run
python scripts/smoke_test.py

# 1. Train  (choose pooling + regularization strength)
python -m src.train  --config configs/distilmse.yaml  --pooling max  --lambda_q <..> --lambda_d <..>

# 2. Index the collection
python -m src.index  --checkpoint results/<run>/model.pt

# 3. Evaluate
python -m src.evaluate  --index results/<run>/index  --queries dev
```

<!-- TODO: adjust to your actual CLI once implemented -->

## Results

**Replication check** — our `max` re-implementation vs the released baseline:

| Model | MRR@10 (dev) | R@1k |
|---|---|---|
| `splade_v2_max` (reference) | ~34.0 | — |
| ours (`max`) | _TODO_ | _TODO_ |

**Pooling comparison** (fixed DistilMSE setting, DistilBERT, best config at FLOPS ≤ 3):

| Pooling | MRR@10 (dev) | R@1k | nDCG@10 (DL19) | FLOPS |
|---|---|---|---|---|
| `max` | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| `sum` | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| `p-norm` | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| `attention` | _TODO_ | _TODO_ | _TODO_ | _TODO_ |

Figures (see `results/figures/`): effectiveness–FLOPS frontier · learned-`p` trajectory · query vs document pooling (`p_q` vs `p_d`) · effectiveness by document length.

## Reproducing the baseline

The `max` model must land within ~1 point of `splade_v2_max` before any variant result is trusted. Differences from the paper (backbone, batch size, truncated step schedule) are expected; what matters for the pooling study is the **relative** comparison under a fixed setting.

## Citation

```bibtex
@inproceedings{formal2022splade,
  title     = {From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective},
  author    = {Formal, Thibault and Lassance, Carlos and Piwowarski, Benjamin and Clinchant, St{\'e}phane},
  booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year      = {2022}
}
```

## Acknowledgements & license

The reference implementation [`naver/splade`](https://github.com/naver/splade) is released under **CC BY-NC-SA 4.0** (non-commercial, share-alike). This is an academic course project; check those terms before any redistribution or reuse of derived material.

<!-- TODO: choose a license for your own code, or state "coursework — not licensed for reuse". -->
