"""
train.py — SPLADE training loop for the pooling study.

The real-data version of the loop already validated by smoke_test.py. Runs on
Kaggle (Tesla P100, ~9h sessions) and produces a checkpoint that index.py /
evaluate.py consume.

Placement: this file lives in src/ next to model.py, loss.py, data.py. The bare
imports below assume src/ is on sys.path (the Kaggle setup cell does
sys.path.insert(0, "/kaggle/working/repo/src")).

Usage (CLI):
    python train.py --variant max --lambda_q 3e-4 --lambda_d 3e-4 --seed 1

Usage (notebook):
    from train import Config, run_training
    run_training(Config(variant="max", lambda_q=3e-4, lambda_d=3e-4))
"""

import argparse
import csv
import os
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
    # --- experiment knobs (what changes across the grid) ---
    variant: str = "max"            # sum | max | p-norm | attention
    lambda_q: float = 3e-4          # query FLOPS weight (target, post-warmup)
    lambda_d: float = 3e-4          # doc   FLOPS weight (target, post-warmup)
    seed: int = 1

    # --- schedule ---
    max_steps: int = 35000          # optimizer steps (paper regime: 30-40k)
    warmup_steps: int = 10000       # linear lambda ramp 0 -> target
    batch_size: int = 32            # micro-batch that actually hits the GPU
    accum_steps: int = 1            # gradient accumulation; effective batch = batch_size * accum_steps
    lr: float = 1e-4
    max_grad_norm: float = 1.0      # grad clipping (set <= 0 to disable)

    # --- tokenization / data volume ---
    max_length: int = 128
    max_triples: int = 2_000_000    # teacher triples loaded into RAM (see note in load_triples)

    # --- fixed model ---
    backbone: str = "distilbert-base-uncased"

    # --- logging / checkpointing ---
    log_every: int = 100
    checkpoint_every: int = 1000

    # --- paths / datasets ---
    ir_datasets_home: str = "/kaggle/input/msmarco-ir-datasets"
    collection: str = "msmarco-passage"
    train_dataset: str = "msmarco-passage/train"
    teacher_path: str = "/kaggle/input/teacher-scores/bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv"
    checkpoint_path: str = ""       # derived from run identity if left empty
    working_dir: str = "/kaggle/working"

    def resolved_checkpoint_path(self) -> str:
        # A per-run default so grid runs never clobber each other's checkpoints.
        if self.checkpoint_path:
            return self.checkpoint_path
        name = f"ckpt_{self.variant}_lq{self.lambda_q:g}_ld{self.lambda_d:g}_seed{self.seed}.pt"
        return os.path.join(self.working_dir, name)


# --------------------------------------------------------------------------
# Call-site data adapters (belong here per the handoff, not in data.py)
# --------------------------------------------------------------------------
class DocstoreWrapper:
    """Make an ir_datasets docstore behave like doc_lookup[pid] -> text."""
    def __init__(self, docstore):
        self.ds = docstore

    def __getitem__(self, pid):
        return self.ds.get(pid).text


def load_triples(path, max_triples=None):
    """Parse the Hofstatter teacher TSV:
        pos_score \t neg_score \t qid \t pos_pid \t neg_pid

    max_triples caps how many lines are read. 35k steps * batch 32 ~= 1.12M
    triples per pass, so the 2M default gives shuffle diversity without loading
    all ~40M lines into RAM. Raise it for wider coverage at RAM/time cost.
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def save_checkpoint(path, step, model, optimizer, losses, cfg):
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "losses": losses,
        "config": asdict(cfg),   # lets index.py/evaluate.py rebuild Splade(variant)
    }, path)


def append_log_row(csv_path, row):
    header = ["step", "total", "ranking", "flops_q", "flops_d", "lambda"]
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
    print(f"=== SPLADE training: variant={cfg.variant} "
          f"lambda_q={cfg.lambda_q:g} lambda_d={cfg.lambda_d:g} seed={cfg.seed} ===")
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # ir_datasets resolves its home dir from this env var; set it before use.
    os.environ["IR_DATASETS_HOME"] = cfg.ir_datasets_home
    import ir_datasets  # local import so the env var above is honored

    # --- data sources ---
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)

    t0 = time.time()
    print("building query lookup ...")
    query_lookup = {q.query_id: q.text
                    for q in ir_datasets.load(cfg.train_dataset).queries_iter()}
    print(f"  {len(query_lookup)} queries in {time.time() - t0:.0f}s")

    docstore = ir_datasets.load(cfg.collection).docs_store()
    doc_lookup = DocstoreWrapper(docstore)

    t0 = time.time()
    print("loading teacher triples ...")
    triples = load_triples(cfg.teacher_path, max_triples=cfg.max_triples)
    print(f"  {len(triples)} triples in {time.time() - t0:.0f}s")

    loader = make_dataloader(triples, query_lookup, doc_lookup,
                             tokenizer, batch_size=cfg.batch_size, shuffle=True)

    # --- model / loss / optim ---
    model = Splade(cfg.variant).to(device)
    # loss_fn's own lambdas are unused: we take its three live component tensors
    # and rebuild `total` with the warmed-up lambdas each step (see loop below).
    loss_fn = SpladeLoss(cfg.lambda_q, cfg.lambda_d)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    # --- resume ---
    ckpt_path = cfg.resolved_checkpoint_path()
    log_csv = ckpt_path.rsplit(".", 1)[0] + "_log.csv"
    start_step, losses = 0, []
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt["step"]
        losses = ckpt.get("losses", [])
        print(f"resumed from step {start_step}")

    if start_step >= cfg.max_steps:
        print("checkpoint already at/past max_steps; nothing to do.")
        return ckpt_path

    # --- loop ---
    model.train()
    step = start_step
    micro = 0
    optimizer.zero_grad()
    done = False

    while not done:
        for batch in loader:
            batch = to_device(batch, device)

            # forward: query encoded once, pos/neg encoded; returns scores AND
            # sparse vectors (the FLOPS regularizer needs the vectors).
            pos_score, neg_score, q_vec, pos_vec, neg_vec = model(
                batch["query_input_ids"], batch["query_attention_mask"],
                batch["pos_input_ids"],   batch["pos_attention_mask"],
                batch["neg_input_ids"],   batch["neg_attention_mask"],
            )

            # pos and neg are both documents in the index -> both regularized.
            # Stack before the loss's per-term cross-doc mean (correct population).
            doc_vecs = torch.cat([pos_vec, neg_vec], dim=0)

            # Ignore loss_fn's internal `total`; rebuild it with warmed lambdas so
            # the FLOPS penalty ramps in instead of collapsing the loss at step 0.
            _total, ranking, flops_q, flops_d = loss_fn(
                pos_score, neg_score,
                batch["teacher_pos"], batch["teacher_neg"],
                q_vec, doc_vecs,
            )

            warm = 1.0 if cfg.warmup_steps <= 0 else min(1.0, (step + 1) / cfg.warmup_steps)
            lq, ld = cfg.lambda_q * warm, cfg.lambda_d * warm
            loss = ranking + lq * flops_q + ld * flops_d

            (loss / cfg.accum_steps).backward()
            micro += 1
            if micro % cfg.accum_steps != 0:
                continue  # keep accumulating gradient; don't step yet

            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % cfg.log_every == 0:
                row = (step, loss.item(), ranking.item(),
                       flops_q.item(), flops_d.item(), lq)
                losses.append(row)
                append_log_row(log_csv, row)
                print(f"step {step:>6} | total {row[1]:.4f} | rank {row[2]:.4f} "
                      f"| flops_q {row[3]:.2f} | flops_d {row[4]:.2f} | lambda {lq:.2e}")

            if step % cfg.checkpoint_every == 0:
                save_checkpoint(ckpt_path, step, model, optimizer, losses, cfg)
                print(f"  checkpoint @ step {step} -> {ckpt_path}")

            if step >= cfg.max_steps:
                done = True
                break

    save_checkpoint(ckpt_path, step, model, optimizer, losses, cfg)
    print(f"training complete @ step {step}; final checkpoint -> {ckpt_path}")
    return ckpt_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
_INT_FIELDS = {"max_steps", "warmup_steps", "batch_size", "accum_steps",
               "max_length", "max_triples", "seed", "log_every", "checkpoint_every"}
_FLOAT_FIELDS = {"lambda_q", "lambda_d", "lr", "max_grad_norm"}


def parse_args():
    p = argparse.ArgumentParser(description="Train SPLADE for the pooling study.")
    defaults = Config()
    for f in fields(Config):
        if f.name in _INT_FIELDS:
            t = int
        elif f.name in _FLOAT_FIELDS:
            t = float
        else:
            t = str
        p.add_argument(f"--{f.name}", type=t, default=getattr(defaults, f.name))
    return p.parse_args()


def main():
    cfg = Config(**vars(parse_args()))
    run_training(cfg)


if __name__ == "__main__":
    main()