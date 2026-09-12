"""
index.py -- encode a document collection into a sparse index.

The script loads a checkpoint of train.py and encodes the documents with the document pooling layer. 
It writes the vectors to disk as sparse matrices (scipy CSR), in shards. 
evaluate.py then searches those shards.

Usage:
    python index.py --checkpoint /kaggle/working/ckpt_max_....pt \
                    --out_dir /kaggle/working/index_max
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from scipy import sparse
from transformers import AutoTokenizer

from model import Splade


def load_model(checkpoint_path, device):
    """Rebuild the trained model from a checkpoint of train.py.

    The checkpoint holds the config of its run,.

    Returns:
        The model in eval mode, the name of its backbone, and its variant.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    variant = cfg.get("variant", "max")
    backbone = cfg.get("backbone", "distilbert-base-uncased")

    model = Splade(variant).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded checkpoint: variant={variant} step={ckpt.get('step')}")
    return model, backbone, variant


def iter_collection(path, allowed=None, max_docs=None):
    """Yield (pid, text) from collection.tsv, one line at a time.

    Args:
        allowed: set of the pids to keep, or None for all of them.
        max_docs: stop after this many documents, or None for all of them.
    """
    n = 0
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            pid, text = parts[0], parts[1]
            if allowed is not None and pid not in allowed:
                continue
            yield pid, text
            n += 1
            if max_docs and n >= max_docs:
                return


def encode_batch(model, tokenizer, texts, device, max_length, use_fp16):
    """Encode a batch of documents into a dense (batch, vocab) tensor on the CPU."""
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        if use_fp16 and device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                vecs = model.encode(enc["input_ids"], enc["attention_mask"], "doc")
        else:
            vecs = model.encode(enc["input_ids"], enc["attention_mask"], "doc")
    return vecs.float().cpu()


def rows_to_sparse(vecs, topk_terms):
    """Turn a dense (batch, vocab) tensor into one (indices, values) pair per row.

    Args:
        topk_terms: keep only the largest terms of a row, or 0 to keep all of
            them.
    """
    out = []
    for row in vecs:
        nz = torch.nonzero(row > 1e-6, as_tuple=False).squeeze(-1)
        vals = row[nz]
        if topk_terms and nz.numel() > topk_terms:
            keep = torch.topk(vals, topk_terms).indices
            nz, vals = nz[keep], vals[keep]
        # float16 halves the memory of the shard while it fills up.
        out.append((nz.numpy().astype(np.int32),
                    vals.numpy().astype(np.float16)))
    return out


def save_shard(out_dir, shard_id, rows, pids, vocab_size):
    """Write one shard as a CSR matrix, plus the pids of its rows.

    A CSR matrix keeps the values of all the rows in one flat array. 
    indptr holds the position where each row starts, so it grows by the number of terms of the row before it.

    The values go back to float32 here, because scipy cannot multiply float16 matrices at search time.
    """
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    for i, (idx, _) in enumerate(rows):
        indptr[i + 1] = indptr[i] + len(idx)
    indices = np.concatenate([r[0] for r in rows]) if rows else np.array([], np.int32)
    values = np.concatenate([r[1] for r in rows]) if rows else np.array([], np.float16)

    mat = sparse.csr_matrix(
        (values.astype(np.float32), indices, indptr),
        shape=(len(rows), vocab_size),
    )
    path = os.path.join(out_dir, f"shard_{shard_id:04d}.npz")
    sparse.save_npz(path, mat)
    np.save(os.path.join(out_dir, f"shard_{shard_id:04d}_pids.npy"),
            np.array(pids, dtype=object), allow_pickle=True)
    return path


def main():
    p = argparse.ArgumentParser(
        description="Encode a collection into a sparse index."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--collection_tsv", default=(
        "/kaggle/input/datasets/giuliobartolonids/splade-data/irds/"
        "msmarco-passage/collectionandqueries/collection.tsv"))
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--shard_size", type=int, default=500_000)
    p.add_argument("--max_docs", type=int, default=0,
                   help="0 = whole collection")
    p.add_argument("--doc_ids", default="",
                   help="optional file with one pid per line to restrict to")
    p.add_argument("--topk_terms", type=int, default=0,
                   help="cap nonzero terms per doc (0 = keep all)")
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True,
                   help="encode in half precision on a GPU")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model, backbone, variant = load_model(args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    vocab_size = model.backbone.config.vocab_size

    allowed = None
    if args.doc_ids:
        with open(args.doc_ids) as f:
            allowed = {line.strip() for line in f if line.strip()}
        print(f"restricting to {len(allowed):,} pids")

    shard_id, shard_rows, shard_pids = 0, [], []
    batch_texts, batch_pids = [], []
    total, t0 = 0, time.time()

    def flush_batch():
        """Encode what is in the batch buffer and move it into the shard."""
        nonlocal batch_texts, batch_pids, total
        if not batch_texts:
            return
        vecs = encode_batch(model, tokenizer, batch_texts, device,
                            args.max_length, args.fp16)
        shard_rows.extend(rows_to_sparse(vecs, args.topk_terms))
        shard_pids.extend(batch_pids)
        total += len(batch_texts)
        batch_texts, batch_pids = [], []

    for pid, text in iter_collection(args.collection_tsv, allowed,
                                     args.max_docs or None):
        batch_texts.append(text)
        batch_pids.append(pid)

        if len(batch_texts) >= args.batch_size:
            flush_batch()

            # about every 50k documents, because total moves by a full batch
            if total % 50_000 < args.batch_size:
                rate = total / max(time.time() - t0, 1e-9)
                print(f"  {total:,} docs | {rate:.0f} docs/s", flush=True)

            # A shard goes to disk as soon as it is full, so the run needs
            # memory for one shard only.
            if len(shard_rows) >= args.shard_size:
                path = save_shard(args.out_dir, shard_id, shard_rows,
                                  shard_pids, vocab_size)
                print(f"saved {path} ({len(shard_rows):,} docs)")
                shard_id += 1
                shard_rows, shard_pids = [], []

    # the last partial batch and the last partial shard
    flush_batch()
    if shard_rows:
        path = save_shard(args.out_dir, shard_id, shard_rows, shard_pids, vocab_size)
        print(f"saved {path} ({len(shard_rows):,} docs)")
        shard_id += 1

    meta = {
        "checkpoint": args.checkpoint,
        "variant": variant,
        "vocab_size": vocab_size,
        "num_docs": total,
        "num_shards": shard_id,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"indexed {total:,} docs in {elapsed/60:.1f} min -> {args.out_dir}")


if __name__ == "__main__":
    main()
