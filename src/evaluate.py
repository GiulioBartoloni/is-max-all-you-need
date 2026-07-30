"""
evaluate.py -- retrieve against a sparse index and report MRR@10.

Encodes the dev queries with the query pooling head, scores them against the
sharded index written by index.py, and computes MRR@10 / Recall@k plus an
estimate of the FLOPS efficiency metric (the x-axis of the main plot).

Usage:
    python evaluate.py --checkpoint /kaggle/working/ckpt_max_....pt \
                       --index_dir /kaggle/working/index_max \
                       --out_run   /kaggle/working/run_max.tsv
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from scipy import sparse
from transformers import AutoTokenizer

from model import Splade
from index import load_model


def load_queries(path, limit=0):
    qids, texts = [], []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                qids.append(parts[0])
                texts.append(parts[1])
            if limit and len(qids) >= limit:
                break
    return qids, texts


def load_qrels(path):
    """MS MARCO qrels: qid <tab> 0 <tab> pid <tab> rel."""
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, pid, rel = parts[0], parts[2], int(parts[3])
            if rel > 0:
                qrels.setdefault(qid, set()).add(pid)
    return qrels


def encode_queries(model, tokenizer, texts, device, batch_size, max_length):
    """Encode queries and return a sparse CSR matrix (n_queries, vocab)."""
    rows_idx, rows_val, indptr = [], [], [0]
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            vecs = model.encode(enc["input_ids"], enc["attention_mask"], "query")
        vecs = vecs.float().cpu()
        for row in vecs:
            nz = torch.nonzero(row, as_tuple=False).squeeze(-1)
            rows_idx.append(nz.numpy().astype(np.int32))
            rows_val.append(row[nz].numpy().astype(np.float32))
            indptr.append(indptr[-1] + nz.numel())

    vocab = model.backbone.config.vocab_size
    return sparse.csr_matrix(
        (np.concatenate(rows_val), np.concatenate(rows_idx),
         np.array(indptr, dtype=np.int64)),
        shape=(len(texts), vocab),
    )


def search(Q, index_dir, topk, chunk):
    """Score queries against every shard, keeping a running top-k."""
    n_q = Q.shape[0]
    best_scores = np.full((n_q, topk), -np.inf, dtype=np.float32)
    best_pids = np.empty((n_q, topk), dtype=object)

    shards = sorted(glob.glob(os.path.join(index_dir, "shard_*.npz")))
    for s_i, shard_path in enumerate(shards):
        mat = sparse.load_npz(shard_path)
        pids = np.load(shard_path.replace(".npz", "_pids.npy"),
                       allow_pickle=True)
        matT = mat.T.tocsc()
        print(f"  shard {s_i+1}/{len(shards)}: {mat.shape[0]:,} docs", flush=True)

        for start in range(0, n_q, chunk):
            stop = min(start + chunk, n_q)
            scores = (Q[start:stop] @ matT).toarray()      # (chunk, n_docs)

            k = min(topk, scores.shape[1])
            part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            part_scores = np.take_along_axis(scores, part, axis=1)

            cand_scores = np.concatenate([best_scores[start:stop], part_scores], axis=1)
            cand_pids = np.concatenate(
                [best_pids[start:stop], pids[part]], axis=1)

            order = np.argsort(-cand_scores, axis=1)[:, :topk]
            best_scores[start:stop] = np.take_along_axis(cand_scores, order, axis=1)
            best_pids[start:stop] = np.take_along_axis(cand_pids, order, axis=1)

    return best_scores, best_pids


def mrr_at_k(qids, best_pids, qrels, k=10):
    total, counted = 0.0, 0
    for i, qid in enumerate(qids):
        rel = qrels.get(qid)
        if not rel:
            continue
        counted += 1
        for rank, pid in enumerate(best_pids[i][:k], start=1):
            if pid in rel:
                total += 1.0 / rank
                break
    return (total / counted if counted else 0.0), counted


def recall_at_k(qids, best_pids, qrels, k):
    total, counted = 0.0, 0
    for i, qid in enumerate(qids):
        rel = qrels.get(qid)
        if not rel:
            continue
        counted += 1
        hits = sum(1 for pid in best_pids[i][:k] if pid in rel)
        total += hits / len(rel)
    return total / counted if counted else 0.0


def estimate_flops(Q, index_dir, sample_shards=1):
    """FLOPS metric: expected number of matching terms per query-doc pair.

    Sum_j p_j^query * p_j^doc, where p_j is the fraction of queries/docs in
    which term j is active. Estimated from the dev queries and a shard sample.
    """
    vocab = Q.shape[1]
    q_active = np.asarray((Q > 0).sum(axis=0)).ravel() / Q.shape[0]

    shards = sorted(glob.glob(os.path.join(index_dir, "shard_*.npz")))[:sample_shards]
    d_active = np.zeros(vocab, dtype=np.float64)
    n_docs = 0
    for shard_path in shards:
        mat = sparse.load_npz(shard_path)
        d_active += np.asarray((mat > 0).sum(axis=0)).ravel()
        n_docs += mat.shape[0]
    d_active /= max(n_docs, 1)

    return float((q_active * d_active).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--index_dir", required=True)
    p.add_argument("--out_run", default="")
    p.add_argument("--queries_tsv", default=(
        "/kaggle/input/datasets/giuliobartolonids/splade-data/irds/"
        "msmarco-passage/collectionandqueries/queries.dev.small.tsv"))
    p.add_argument("--qrels", default=(
        "/kaggle/input/datasets/giuliobartolonids/splade-data/irds/"
        "msmarco-passage/collectionandqueries/qrels.dev.small.tsv"))
    p.add_argument("--topk", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--query_chunk", type=int, default=32)
    p.add_argument("--limit_queries", type=int, default=0,
                   help="0 = all dev queries")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, backbone, variant = load_model(args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    qids, texts = load_queries(args.queries_tsv, args.limit_queries)
    qrels = load_qrels(args.qrels)
    print(f"{len(qids):,} queries | {len(qrels):,} with qrels")

    t0 = time.time()
    Q = encode_queries(model, tokenizer, texts, device,
                       args.batch_size, args.max_length)
    print(f"encoded queries in {time.time()-t0:.0f}s "
          f"(avg {Q.nnz/Q.shape[0]:.1f} terms/query)")

    t0 = time.time()
    scores, pids = search(Q, args.index_dir, args.topk, args.query_chunk)
    print(f"searched in {time.time()-t0:.0f}s")

    mrr, counted = mrr_at_k(qids, pids, qrels, k=10)
    r100 = recall_at_k(qids, pids, qrels, 100)
    r1000 = recall_at_k(qids, pids, qrels, min(1000, args.topk))
    flops = estimate_flops(Q, args.index_dir)

    print("=" * 55)
    print(f"variant     : {variant}")
    print(f"MRR@10      : {mrr:.4f}   ({counted:,} judged queries)")
    print(f"Recall@100  : {r100:.4f}")
    print(f"Recall@1000 : {r1000:.4f}")
    print(f"FLOPS (est) : {flops:.3f}")
    print("=" * 55)

    if args.out_run:
        with open(args.out_run, "w") as f:
            for i, qid in enumerate(qids):
                for rank, (pid, sc) in enumerate(
                        zip(pids[i], scores[i]), start=1):
                    if pid is None:
                        continue
                    f.write(f"{qid}\t{pid}\t{rank}\t{sc:.4f}\n")
        print(f"run written -> {args.out_run}")

        summary = args.out_run.rsplit(".", 1)[0] + "_metrics.json"
        with open(summary, "w") as f:
            json.dump({"variant": variant, "mrr@10": mrr,
                       "recall@100": r100, "recall@1000": r1000,
                       "flops": flops, "checkpoint": args.checkpoint}, f, indent=2)
        print(f"metrics written -> {summary}")


if __name__ == "__main__":
    main()