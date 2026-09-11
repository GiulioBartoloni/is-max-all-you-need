"""
train.py -- train SPLADE with one pooling variant.

The script trains the model on the DistilMSE triples and writes the checkpoint
that index.py and evaluate.py read. It also appends a CSV log with the parts of
the loss and, for the p-norm variant, the value of the learned exponent.

The study ran on Kaggle, where a session stops after about 9 hours. A stopped
session leaves a checkpoint behind, and a new run with the same settings
continues from it until max_steps.

The imports below have no package prefix, so src/ must be on sys.path. The
Kaggle setup cell does:
    sys.path.insert(0, "/kaggle/working/repo/src")

Usage (CLI):
    python train.py --variant max --lambda_q 3e-4 --lambda_d 3e-4 --seed 1

Usage (notebook):
    from train import Config, run_training
    run_training(Config(variant="max", lambda_q=3e-4, lambda_d=3e-4))
"""

import argparse
import csv
import os
import pickle
import random
import time
from dataclasses import dataclass, asdict, fields

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from model import Splade
from loss import SpladeLoss
from data import make_dataloader


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclass
class Config:
    """Every setting of one run. It also goes into the checkpoint, as a dict."""

    # --- experiment knobs (what changes across the grid) ---
    variant: str = "max"            # sum | max | p-norm | attention
    lambda_q: float = 3e-4
    lambda_d: float = 3e-4
    seed: int = 1

    # --- schedule ---
    max_steps: int = 35000
    warmup_steps: int = 10000       # steps of the ramp from 0 to the lambdas
    batch_size: int = 32
    accum_steps: int = 1            # effective batch = batch_size * accum_steps
    lr: float = 1e-4
    p_lr: float = 1e-3              # own learning rate for the p-norm exponent
    max_grad_norm: float = 1.0

    # --- tokenization / data volume ---
    max_length: int = 128
    max_triples: int = 2_000_000    # cap on the triples held in RAM

    # --- fixed model ---
    backbone: str = "distilbert-base-uncased"

    # --- logging / checkpointing ---
    log_every: int = 100
    checkpoint_every: int = 1000

    # --- paths (the Kaggle layout of the dataset) ---
    collection_tsv: str = (
        "/kaggle/input/datasets/giuliobartolonids/splade-data/irds/"
        "msmarco-passage/collectionandqueries/collection.tsv"
    )
    train_queries_tsv: str = (
        "/kaggle/input/datasets/giuliobartolonids/splade-data/irds/"
        "msmarco-passage/collectionandqueries/queries.train.tsv"
    )
    teacher_path: str = (
        "/kaggle/input/datasets/giuliobartolonids/splade-data/teacher.tsv"
    )
    checkpoint_path: str = ""       # empty -> derived from the run settings
    working_dir: str = "/kaggle/working"

    def resolved_checkpoint_path(self) -> str:
        """Return the checkpoint path, and build a name for it if it is empty.

        The name holds the variant, the two lambdas and the seed. Two runs of
        the grid therefore cannot overwrite each other, and a repeated run
        finds its own checkpoint again.
        """
        if self.checkpoint_path:
            return self.checkpoint_path
        name = (
            f"ckpt_{self.variant}"
            f"_lq{self.lambda_q:g}_ld{self.lambda_d:g}"
            f"_seed{self.seed}.pt"
        )
        return os.path.join(self.working_dir, name)


# --------------------------------------------------------------------------
# Data adapters -- direct TSV reads, no ir_datasets dependency
# --------------------------------------------------------------------------
class TsvDocstore:
    """Read a document of collection.tsv by its id, without the file in RAM.

    The collection has 8.8M lines of "pid TAB text", which is too much to hold
    next to the model. The class scans the file once for a {pid: byte offset}
    index, and then every lookup is a seek plus one line.

    The scan takes about 2 minutes. With an index_cache path, the index also
    goes to a pickle file, and the next session loads it instead.
    """

    def __init__(self, collection_path, index_cache=None):
        self._path = collection_path
        self._index_cache = index_cache
        self._index = {}
        self._built = False

    def build(self):
        """Load the offset index from the cache, or scan the file for it."""
        if self._index_cache and os.path.exists(self._index_cache):
            with open(self._index_cache, "rb") as f:
                self._index = pickle.load(f)
            self._built = True
            print(f"docstore index loaded from cache ({len(self._index):,} docs)")
            return

        print("building docstore offset index (~2 min)...")
        t0 = time.time()
        with open(self._path, "rb") as f:
            # A `for line in f` loop reads ahead into a buffer, which makes
            # tell() point past the line. readline() keeps the two in step.
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                pid = line.split(b"\t", 1)[0].decode().strip()
                self._index[pid] = offset
        self._built = True
        print(f"index built: {len(self._index):,} docs in {time.time()-t0:.0f}s")

        if self._index_cache:
            with open(self._index_cache, "wb") as f:
                pickle.dump(self._index, f)
            print(f"index cached -> {self._index_cache}")

    def __getitem__(self, pid):
        if not self._built:
            self.build()
        with open(self._path, "rb") as f:
            f.seek(self._index[str(pid)])
            _, text = f.readline().decode().split("\t", 1)
            return text.strip()


def load_query_lookup(tsv_path):
    """Return {qid: text} from a queries.*.tsv file."""
    lookup = {}
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                lookup[parts[0]] = parts[1]
    return lookup


def load_triples(path, max_triples=None):
    """Read the teacher triples of the DistilMSE file (Hofstatter et al.).

    One line holds:
        pos_score TAB neg_score TAB qid TAB pos_pid TAB neg_pid

    max_triples caps the lines that go into RAM. A pass of 35k steps at batch
    32 uses about 1.12M triples, so 2M gives enough diversity to the shuffle
    without the full 40M file.
    """
    triples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_triples and i >= max_triples:
                break
            parts = line.strip().split("\t")
            pos_score, neg_score = float(parts[0]), float(parts[1])
            qid, pos_pid, neg_pid = parts[2], parts[3], parts[4]
            triples.append((pos_score, neg_score, qid, pos_pid, neg_pid))
    return triples


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def set_seed(seed):
    """Seed every random source that a run uses, for a repeatable shuffle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    """Move the tensors of a batch to the device and keep the other values."""
    return {k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def save_checkpoint(path, step, model, optimizer, losses, cfg):
    """Write the state of a run to one file.

    The optimizer state is part of it, because the moments of AdamW must
    survive the end of a Kaggle session.
    """
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "losses": losses,
        "config": asdict(cfg),
    }, path)


def append_log_row(csv_path, row):
    """Append one row to the CSV log, and write the header if the file is new."""
    header = ["step", "total", "ranking", "flops_q", "flops_d", "lambda", "p_q", "p_d"]
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if not exists:
            w.writerow(header)
        w.writerow(row)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def run_training(cfg: Config):
    """Train one variant and return the path of its checkpoint.

    If a checkpoint with that path is already there, the run continues from its
    step instead of starting again.
    """
    print(
        f"=== SPLADE training: variant={cfg.variant} "
        f"lambda_q={cfg.lambda_q:g} lambda_d={cfg.lambda_d:g} "
        f"seed={cfg.seed} ==="
    )
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device,
          f"({torch.cuda.get_device_name(0)})" if device == "cuda" else "")

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)

    # --- docstore ---
    index_cache = os.path.join(cfg.working_dir, "docstore_index.pkl")
    doc_lookup = TsvDocstore(cfg.collection_tsv, index_cache=index_cache)
    doc_lookup.build()   # build it now, so its cost does not hide in step 1

    # --- query lookup ---
    t0 = time.time()
    print("loading train queries ...")
    query_lookup = load_query_lookup(cfg.train_queries_tsv)
    print(f"  {len(query_lookup):,} queries in {time.time()-t0:.0f}s")

    # --- teacher triples ---
    t0 = time.time()
    print("loading teacher triples ...")
    triples = load_triples(cfg.teacher_path, max_triples=cfg.max_triples)
    print(f"  {len(triples):,} triples in {time.time()-t0:.0f}s")

    loader = make_dataloader(
        triples, query_lookup, doc_lookup,
        tokenizer,
        batch_size=cfg.batch_size,
        shuffle=True,
        max_length=cfg.max_length,
    )

    # --- model / loss / optimizer ---
    model = Splade(cfg.variant).to(device)
    loss_fn = SpladeLoss(cfg.lambda_q, cfg.lambda_d)

    # The exponent p of the p-norm variant gets its own learning rate. It is a
    # single scalar with a small gradient, so at the rate of the backbone it
    # almost does not move. The other variants have no p, and then one group is
    # enough.
    p_params, base_params = [], []
    for name, param in model.named_parameters():
        if name in ("query_pool.p", "doc_pool.p"):
            p_params.append(param)
        else:
            base_params.append(param)

    if p_params:
        optimizer = AdamW([
            {"params": base_params, "lr": cfg.lr},
            {"params": p_params, "lr": cfg.p_lr},
        ])
        print(f"separate lr for p: {cfg.p_lr}")
    else:
        optimizer = AdamW(model.parameters(), lr=cfg.lr)

    # --- resume from a checkpoint if one is there ---
    ckpt_path = cfg.resolved_checkpoint_path()
    log_csv = ckpt_path.rsplit(".", 1)[0] + "_log.csv"
    start_step, losses = 0, []
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        # The optimizer state also holds the learning rates of the saved run.
        # Put the configured ones back, so a resumed run can change them.
        optimizer.param_groups[0]["lr"] = cfg.lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]["lr"] = cfg.p_lr
        print("lr after load:", [g["lr"] for g in optimizer.param_groups])
        start_step = ckpt["step"]
        losses = ckpt.get("losses", [])
        print(f"resumed from step {start_step}")

    if start_step >= cfg.max_steps:
        print("already at max_steps; nothing to do.")
        return ckpt_path

    # --- training loop ---
    model.train()
    step = start_step
    micro = 0
    optimizer.zero_grad()
    done = False

    # One pass over the loader is not enough for max_steps, so the outer loop
    # starts it again. Every pass reshuffles the triples.
    while not done:
        for batch in loader:
            batch = to_device(batch, device)

            pos_score, neg_score, q_vec, pos_vec, neg_vec = model(
                batch["query_input_ids"], batch["query_attention_mask"],
                batch["pos_input_ids"],   batch["pos_attention_mask"],
                batch["neg_input_ids"],   batch["neg_attention_mask"],
            )

            # The positive and the negative document are both documents of the
            # index, so the FLOPS penalty sees them as one group.
            doc_vecs = torch.cat([pos_vec, neg_vec], dim=0)

            # loss_fn also returns a total, but that total holds the target
            # lambdas. The lambdas here ramp up over warmup_steps instead, so
            # only the parts of the loss come from loss_fn. Without the ramp,
            # the dense vectors of step 0 give a loss of some millions, and the
            # model escapes it with vectors of only zeros.
            _, ranking, flops_q, flops_d = loss_fn(
                pos_score, neg_score,
                batch["teacher_pos"], batch["teacher_neg"],
                q_vec, doc_vecs,
            )

            warm = 1.0 if cfg.warmup_steps <= 0 else min(
                1.0, (step + 1) / cfg.warmup_steps
            )
            lq = cfg.lambda_q * warm
            ld = cfg.lambda_d * warm
            loss = ranking + lq * flops_q + ld * flops_d

            # The gradients of accum_steps micro-batches add up in the same
            # buffer, so each one contributes only its share of the loss.
            (loss / cfg.accum_steps).backward()

            micro += 1
            if micro % cfg.accum_steps != 0:
                continue

            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm
                )
            optimizer.step()

            # A step can push p below 0, where the log space power mean
            # breaks. The clamp holds it at 0.5, which is already past the
            # mean end of the pooling family.
            with torch.no_grad():
                for param in p_params:
                    param.clamp_(min=0.5)
            optimizer.zero_grad()
            step += 1

            if step % cfg.log_every == 0:
                # Only the p-norm variant has a p; the others log a NaN.
                p_q = model.query_pool.p.item() if hasattr(model.query_pool, "p") else float("nan")
                p_d = model.doc_pool.p.item() if hasattr(model.doc_pool, "p") else float("nan")
                row = (
                    step, loss.item(), ranking.item(),
                    flops_q.item(), flops_d.item(), lq, p_q, p_d
                )
                losses.append(row)
                append_log_row(log_csv, row)
                print(
                    f"step {step:>6} | total {row[1]:.4f} "
                    f"| rank {row[2]:.4f} "
                    f"| flops_q {row[3]:.2f} | flops_d {row[4]:.2f} "
                    f"| lambda {lq:.2e} | p_q {p_q:.3f} | p_d {p_d:.3f}"
                )

            if step % cfg.checkpoint_every == 0:
                save_checkpoint(ckpt_path, step, model, optimizer, losses, cfg)
                print(f"  checkpoint @ step {step} -> {ckpt_path}")

            if step >= cfg.max_steps:
                done = True
                break

    save_checkpoint(ckpt_path, step, model, optimizer, losses, cfg)
    print(f"training complete @ step {step} -> {ckpt_path}")
    return ckpt_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    """Give every field of Config a --flag, with its type and its default."""
    p = argparse.ArgumentParser(
        description="Train SPLADE for the pooling study."
    )
    defaults = Config()
    for f in fields(Config):
        p.add_argument(f"--{f.name}", type=f.type,
                       default=getattr(defaults, f.name))
    return p.parse_args()


def main():
    cfg = Config(**vars(parse_args()))
    run_training(cfg)


if __name__ == "__main__":
    main()
