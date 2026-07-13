"""
Test harness for pooling.py — run it, read the PASS/FAIL report.

Run from the folder that contains pooling.py:
    uv run python test_pooling.py
If your pooling.py lives under src/, change the import below to:
    from src.pooling import SumPooling, MaxPooling, PNormPooling, AttentionPooling
"""

import torch
from pooling import SumPooling, MaxPooling, PNormPooling, AttentionPooling


# --- tiny test runner: collects results instead of stopping at first failure ---
_results = []

def check(name, fn):
    try:
        fn()
        _results.append((name, True, ""))
        print(f"[PASS] {name}")
    except AssertionError as e:
        _results.append((name, False, str(e)))
        print(f"[FAIL] {name}  ->  {e}")
    except Exception as e:
        _results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"[ERROR] {name}  ->  {type(e).__name__}: {e}")


# ======================================================================
# SUM / MAX / P-NORM  (unchanged)
# ======================================================================
def test_shapes():
    x = torch.rand(2, 5, 100)
    mask = torch.ones(2, 5, dtype=torch.long)
    for pool in (SumPooling(), MaxPooling(), PNormPooling()):
        out = pool(x, mask)
        assert out.shape == (2, 100), f"{type(pool).__name__} gave {tuple(out.shape)}, expected (2, 100)"

def test_sum_known_value():
    x = torch.tensor([[[0.2], [0.9], [0.5]]])
    mask = torch.ones(1, 3, dtype=torch.long)
    out = SumPooling()(x, mask)
    assert torch.allclose(out, torch.tensor([[1.6]]), atol=1e-5), f"sum={out.item():.4f}, expected 1.6"

def test_max_known_value():
    x = torch.tensor([[[0.2], [0.9], [0.5]]])
    mask = torch.ones(1, 3, dtype=torch.long)
    out = MaxPooling()(x, mask)
    assert torch.allclose(out, torch.tensor([[0.9]]), atol=1e-5), f"max={out.item():.4f}, expected 0.9"

def test_max_masking_negative():
    x = torch.tensor([[[-5.], [-2.], [-8.]]])
    mask = torch.tensor([[1, 1, 0]])
    out = MaxPooling()(x, mask)
    assert torch.allclose(out, torch.tensor([[-2.0]]), atol=1e-5), \
        f"max with padding={out.item():.4f}, expected -2.0"

def test_pnorm_approaches_max():
    x = torch.rand(2, 4, 6) * 3.0
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    pool = PNormPooling()
    pool.p.data = torch.tensor(1000.0)
    got = pool(x, mask)
    ref = MaxPooling()(x, mask)
    assert torch.allclose(got, ref, atol=1e-2), f"large-p should approach max; diff={(got-ref).abs().max():.4f}"

def test_pnorm_equals_mean_at_p1():
    x = torch.rand(2, 4, 6)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    pool = PNormPooling()
    pool.p.data = torch.tensor(1.0)
    got = pool(x, mask)
    m = mask.float().unsqueeze(-1)
    ref = (x * m).sum(dim=1) / m.sum(dim=1)
    assert torch.allclose(got, ref, atol=1e-4), f"p=1 should equal masked mean; diff={(got-ref).abs().max():.6f}"

def test_pnorm_gradient_flows():
    x = torch.rand(2, 4, 6)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    pool = PNormPooling()
    pool(x, mask).sum().backward()
    assert pool.p.grad is not None, "p.grad is None -> p not in graph"
    assert torch.isfinite(pool.p.grad), f"p.grad is {pool.p.grad} -> non-finite (check -inf vs -1e9 masking)"

def test_pnorm_stable_on_large_values():
    x = torch.rand(2, 4, 6) * 50.0
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    pool = PNormPooling()
    pool.p.data = torch.tensor(100.0)
    assert torch.isfinite(pool(x, mask)).all(), "output has inf/nan"


# ======================================================================
# ATTENTION  (new) -- constructor needs vocab_size, unlike the others
# ======================================================================
VOCAB = 6

def test_attn_shape():
    x = torch.rand(2, 4, VOCAB)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    out = AttentionPooling(vocab_size=VOCAB)(x, mask)
    assert out.shape == (2, VOCAB), f"got {tuple(out.shape)}, expected (2, {VOCAB})"

def test_attn_masking_no_leak():
    # Huge value ONLY at a padding position. If masking works, its ~0 weight
    # means it must not blow up the output.
    x = torch.rand(1, 3, VOCAB)
    x[0, 2, :] = 1e6                       # position 2 is padding (see mask)
    mask = torch.tensor([[1, 1, 0]])
    out = AttentionPooling(vocab_size=VOCAB)(x, mask)
    assert out.max().item() < 100.0, \
        f"padding leaked: output max={out.max().item():.1f} (huge padding value got weight)"

def test_attn_weights_sum_to_one():
    # Reconstruct the internal weights the same way forward does.
    x = torch.rand(2, 4, VOCAB)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    pool = AttentionPooling(vocab_size=VOCAB)
    scores = pool.scorer(x)
    masked = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
    weights = torch.softmax(masked, dim=1)         # (batch, seq, 1)
    sums = weights.sum(dim=1).squeeze(-1)          # (batch,)
    assert torch.allclose(sums, torch.ones(2), atol=1e-4), f"weights sum to {sums.tolist()}, expected ~1"
    assert weights[0, 3, 0] < 1e-4 and weights[1, 2, 0] < 1e-4, "padding positions got non-negligible weight"

def test_attn_gradient_flows():
    x = torch.rand(2, 4, VOCAB)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    pool = AttentionPooling(vocab_size=VOCAB)
    pool(x, mask).sum().backward()
    g = pool.scorer.weight.grad
    assert g is not None, "scorer.weight.grad is None -> layer not in graph"
    assert torch.isfinite(g).all(), "scorer gradient has inf/nan"
    assert g.abs().sum().item() > 0, "scorer gradient is all zero -> layer had no effect"


if __name__ == "__main__":
    print("=" * 60)
    check("shape: sum/max/pnorm -> (batch, vocab)", test_shapes)
    check("sum known value = 1.6", test_sum_known_value)
    check("max known value = 0.9", test_max_known_value)
    check("max masking (negatives, no leak) = -2.0", test_max_masking_negative)
    check("p-norm large p  ~= max", test_pnorm_approaches_max)
    check("p-norm p=1      ~= masked mean", test_pnorm_equals_mean_at_p1)
    check("p-norm gradient flows into p", test_pnorm_gradient_flows)
    check("p-norm stable on large values", test_pnorm_stable_on_large_values)
    print("-" * 60)
    check("attention shape -> (batch, vocab)", test_attn_shape)
    check("attention masking: padding does not leak", test_attn_masking_no_leak)
    check("attention weights sum to 1, padding ~0", test_attn_weights_sum_to_one)
    check("attention gradient flows into scorer", test_attn_gradient_flows)
    print("=" * 60)
    passed = sum(1 for _, ok, _ in _results if ok)
    print(f"{passed}/{len(_results)} passed")