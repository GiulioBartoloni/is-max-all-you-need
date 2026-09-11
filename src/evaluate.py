"""
evaluate.py -- search a sparse index and report the retrieval metrics.

The script encodes the dev queries with the query pooling layer and scores them
against the shards that index.py wrote. It then prints MRR@10, Recall@100,
Recall@1000 and an estimate of the FLOPS metric. Those numbers are the columns
of results/metrics.csv, and FLOPS is the x-axis of the trade-off plot.

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

from index import load_model


def load_queries(path, limit=0):
    """Return the ids and the texts of a queries TSV file, in file order."""
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
    """Return {qid: set of relevant pids} from a qrels file.

    One line of the MS MARCO qrels holds: qid, 0, pid, relevance. The dev set
    marks about one relevant passage per query.
    """
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
    """Encode the queries into one sparse CSR matrix of shape (n_queries, vocab).

    The function builds the CSR form directly: the terms of a query go into the
    flat arrays, and indptr marks where each query starts.
    """
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
            # same threshold as index.py: below it the terms are noise
            nz = torch.nonzero(row > 1e-6, as_tuple=False).squeeze(-1)
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
    """Score every query against every shard and keep the best topk documents.

    The score of a query-document pair is their dot product. One sparse matrix
    product therefore gives the scores of a chunk of queries against a whole
    shard. Only one shard stays in memory at a time: the loop merges the
    running best of each query with the best of the new shard.

    Args:
        Q: the query matrix of encode_queries.
        chunk: number of queries per matrix product. A large chunk is faster
            but its dense score block is chunk * shard_size floats.

    Returns:
        The scores and the pids of the topk documents per query, both sorted by
        score.
    """
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

            # argpartition puts the k best of the shard in the first k columns
            # without a sort of the whole row, which is much cheaper.
            k = min(topk, scores.shape[1])
            part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            part_scores = np.take_along_axis(scores, part, axis=1)

            # merge: sort the running best together with the k new candidates
            # and keep the first topk of them.
            cand_scores = np.concatenate([best_scores[start:stop], part_scores], axis=1)
            cand_pids = np.concatenate(
                [best_pids[start:stop], pids[part]], axis=1)

            order = np.argsort(-cand_scores, axis=1)[:, :topk]
            best_scores[start:stop] = np.take_along_axis(cand_scores, order, axis=1)
            best_pids[start:stop] = np.take_along_axis(cand_pids, order, axis=1)

    return best_scores, best_pids


def mrr_at_k(qids, best_pids, qrels, k=10):
    """Return the mean reciprocal rank at k, and the number of judged queries.

    A query counts 1/rank for its first relevant document in the top k, and 0
    if it has none there. The loop skips the queries without a judgement:
    their result is unknown, not wrong.
    """
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
    """Return the mean recall at k over the judged queries.

    The recall of a query is the part of its relevant documents that the top k
    holds.
    """
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
    """Return the FLOPS metric: the mean number of terms that a pair shares.

    For each vocabulary term, take the part of the queries and the part of the
    documents in which the term is active. The product of the two parts is the
    chance that both sides hold the term. The sum over the terms is then the
    number of terms that a random query and a random document share. That
    number is the work that the retrieval does for the pair.

    The document side comes from a sample of the shards, because one shard
    already gives a stable estimate.
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
    p = argparse.ArgumentParser(
        description="Search a sparse index and report the metrics."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--index_dir", required=True)
    p.add_argument("--out_run", default="",
                   help="optional TSV with the ranking: qid, pid, rank, score")
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
        # The ranking of every query, plus the same metrics as a JSON file next
        # to it. A shard smaller than topk leaves empty places, which are None.
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
