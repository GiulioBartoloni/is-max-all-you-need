"""
GATE 1 -- pipeline smoke test.

Wires data -> model -> loss together and overfits a tiny fake fixture for a few
dozen steps. It proves the whole pipeline is correct in miniature:
  * the loss goes DOWN (the model is learning on toy data),
  * gradients reach the pooling parameter (p),
  * gradients reach the backbone.

Runs on CPU in ~1-2 minutes (the cost is DistilBERT forward/backward, plus a
one-time model download on first run). No real data needed.

Run from the folder containing model.py / loss.py / data.py / pooling.py:
    uv run python smoke_test.py
"""

import time
import torch
from transformers import AutoTokenizer

from model import Splade
from loss import SpladeLoss
from data import make_dataloader

torch.manual_seed(0)

# --- tiny fake fixture: 4 triples, teacher clearly prefers the positive ------
triples = [
    # (pos_score, neg_score, qid, pos_pid, neg_pid)
    (6.0, 1.0, "q1", "d1", "d2"),
    (5.0, 0.5, "q2", "d3", "d4"),
    (7.0, 2.0, "q3", "d2", "d5"),
    (4.0, 1.0, "q4", "d5", "d1"),
]
query_lookup = {
    "q1": "where do cats sleep",
    "q2": "how fast can dogs run",
    "q3": "what color is the sky",
    "q4": "how tall are giraffes",
}
doc_lookup = {
    "d1": "cats usually sleep on warm soft surfaces like beds and mats",
    "d2": "the sky appears blue because of light scattering",
    "d3": "dogs can run very fast over short distances",
    "d4": "bananas are a good source of potassium",
    "d5": "giraffes are the tallest living land animals",
}

STEPS = 50
LR = 1e-4
LAMBDA_Q = 1e-3
LAMBDA_D = 1e-3


def run_batch(model, batch):
    """One forward pass: encode the triple, produce scores + vectors for the loss."""
    q_vec   = model.encode(batch["query_input_ids"], batch["query_attention_mask"], "query")
    pos_vec = model.encode(batch["pos_input_ids"],   batch["pos_attention_mask"],   "doc")
    neg_vec = model.encode(batch["neg_input_ids"],   batch["neg_attention_mask"],   "doc")

    pos_score = model.score(q_vec, pos_vec)
    neg_score = model.score(q_vec, neg_vec)

    # FLOPS on documents is computed over BOTH the positive and negative docs
    doc_vecs = torch.cat([pos_vec, neg_vec], dim=0)
    return pos_score, neg_score, q_vec, doc_vecs


def main():
    print("=" * 60)
    print("building fixture, model (p-norm), loss ...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    loader = make_dataloader(triples, query_lookup, doc_lookup, tokenizer,
                             batch_size=4, shuffle=False)
    batch = next(iter(loader))          # one fixed batch -- we overfit it

    model = Splade("p-norm")
    model.train()
    loss_fn = SpladeLoss(LAMBDA_Q, LAMBDA_D)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"training {STEPS} steps on one batch (lr={LR}) ...")
    t0 = time.time()
    losses = []
    for step in range(STEPS):
        optimizer.zero_grad()
        pos_score, neg_score, q_vec, doc_vecs = run_batch(model, batch)
        total, ranking, q_flops, d_flops = loss_fn(
            pos_score, neg_score, batch["teacher_pos"], batch["teacher_neg"],
            q_vec, doc_vecs,
        )
        total.backward()
        optimizer.step()
        losses.append(total.item())
        # if step % 10 == 0 or step == STEPS - 1:
        print(f"  step {step:3d}  loss={total.item():8.4f}  "
                f"rank={ranking.item():7.3f}  q_flops={q_flops.item():7.3f}  d_flops={d_flops.item():7.3f}")
    print(f"trained in {time.time() - t0:.1f}s")
    print("-" * 60)

    # ---- the three checks -------------------------------------------------
    ok = True

    # 1) learning: loss went down
    if losses[-1] < losses[0]:
        print(f"[PASS] loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}")
    else:
        ok = False
        print(f"[FAIL] loss did NOT decrease: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # 2) gradients reached the pooling parameter (p-norm has a learnable p)
    if model.query_pool.p.grad is not None and torch.isfinite(model.query_pool.p.grad):
        print(f"[PASS] pooling p got a gradient ({model.query_pool.p.grad.item():+.4f})")
    else:
        ok = False
        print(f"[FAIL] pooling p gradient = {model.query_pool.p.grad}")

    # 3) gradients reached the backbone
    backbone_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.backbone.parameters()
    )
    if backbone_grad:
        print("[PASS] backbone got gradients")
    else:
        ok = False
        print("[FAIL] backbone got NO gradients")

    print("=" * 60)
    print("SMOKE TEST PASSED -- pipeline is wired correctly" if ok
          else "SMOKE TEST FAILED -- see above")


if __name__ == "__main__":
    main()