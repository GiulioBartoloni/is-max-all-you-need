"""Masked pooling layers for aggregating token-level representations.

Each module reduces a padded batch of token embeddings of shape
``(batch, seq_len, vocab)`` into a single vector per sequence of shape
``(batch, vocab)``, using a binary ``mask`` of shape ``(batch, seq_len)`` in
which ``1`` marks a real token and ``0`` marks padding.
"""

import torch


class SumPooling(torch.nn.Module):
    """Sum the token embeddings over the sequence, ignoring padding."""

    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        """Aggregate token embeddings by summation.

        Args:
            x: Token embeddings of shape ``(batch, seq_len, vocab)``.
            mask: Binary mask of shape ``(batch, seq_len)`` where ``1`` marks
                a real token and ``0`` marks padding.

        Returns:
            Summed embeddings of shape ``(batch, vocab)``.
        """
        unsqueezed_mask = mask.unsqueeze(-1)

        masked = x * unsqueezed_mask
        final = masked.sum(dim=1)

        return final


class MaxPooling(torch.nn.Module):
    """Take the element-wise maximum over the sequence, ignoring padding."""

    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        """Aggregate token embeddings by element-wise maximum.

        Padding positions are set to ``-inf`` before the reduction so they can
        never be selected, even when all real values are negative.

        Args:
            x: Token embeddings of shape ``(batch, seq_len, vocab)``.
            mask: Binary mask of shape ``(batch, seq_len)`` where ``1`` marks
                a real token and ``0`` marks padding.

        Returns:
            Element-wise maxima of shape ``(batch, vocab)``.
        """
        unsqueezed_mask = mask.unsqueeze(-1)

        masked = x.masked_fill(unsqueezed_mask == 0, float('-inf'))
        final = masked.max(dim=1).values

        return final


class PNormPooling(torch.nn.Module):
    """Generalized-mean (power-mean) pooling with a learnable exponent ``p``.

    Computes ``(mean_i x_i**p)**(1/p)`` over the non-padded tokens. The
    exponent ``p`` is a trainable parameter that interpolates between pooling
    behaviours: ``p = 1`` recovers the masked mean, while large ``p``
    approaches max pooling. The computation is done in log space for numerical
    stability, which assumes non-negative inputs ``x``.

    Attributes:
        p: Learnable exponent, initialized to ``10.0``.
    """

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor(10.0))

    def forward(self, x, mask):
        """Aggregate token embeddings by the learnable generalized mean.

        Args:
            x: Non-negative token embeddings of shape
                ``(batch, seq_len, vocab)``.
            mask: Binary mask of shape ``(batch, seq_len)`` where ``1`` marks
                a real token and ``0`` marks padding.

        Returns:
            Generalized-mean pooled embeddings of shape ``(batch, vocab)``.
        """
        eps = 1e-10
        N = mask.float().sum(dim=1, keepdim=True)
        logs_x = torch.log(x + eps)

        unsqueezed_mask = mask.unsqueeze(-1)
        masked = logs_x.masked_fill(unsqueezed_mask == 0, -1e9)

        logsumexp_x = torch.logsumexp(masked * self.p, dim=1)
        averaged_x = logsumexp_x - torch.log(N)

        result = torch.exp(averaged_x / self.p)

        return result


class AttentionPooling(torch.nn.Module):
    """Weighted pooling with attention weights from a learnable scorer.

    A linear layer scores each token, padding positions are masked out, and the
    scores are normalized with a softmax over the sequence to produce attention
    weights that combine the token embeddings.

    Attributes:
        scorer: Linear layer mapping each token embedding to a scalar score.
    """

    def __init__(self, vocab_size):
        """Initialize the pooling layer.

        Args:
            vocab_size: Dimensionality ``dim`` of the token embeddings, i.e.
                the input feature size of the scoring layer.
        """
        super().__init__()
        self.scorer = torch.nn.Linear(vocab_size, 1)

    def forward(self, x, mask):
        """Aggregate token embeddings by learned attention weights.

        Args:
            x: Token embeddings of shape ``(batch, seq_len, vocab)``.
            mask: Binary mask of shape ``(batch, seq_len)`` where ``1`` marks
                a real token and ``0`` marks padding.

        Returns:
            Attention-weighted embeddings of shape ``(batch, vocab)``.
        """
        scores = self.scorer(x)

        unsqueezed_mask = mask.unsqueeze(-1)
        masked = scores.masked_fill(unsqueezed_mask == 0, -1e9)

        weights = torch.softmax(masked, dim=1)
        result = (x * weights).sum(dim=1)

        return result


def make_pooling(name, vocab_size=None):
    """Construct a pooling module by name.

    Provides a single entry point so callers can select a pooling strategy
    from a configuration string without importing the individual classes or
    knowing their differing constructor signatures.

    Args:
        name: Pooling strategy, one of ``"sum"``, ``"max"``, ``"p-norm"`` or
            ``"attention"``.
        vocab_size: Dimensionality ``dim`` of the token embeddings. Required
            for ``"attention"`` (the scoring layer's input size) and ignored by
            the other strategies.

    Returns:
        The corresponding pooling module.

    Raises:
        ValueError: If ``name`` is not a known strategy, or if ``"attention"``
            is requested without a ``vocab_size``.
    """
    if name == "sum":
        return SumPooling()
    if name == "max":
        return MaxPooling()
    if name == "p-norm":
        return PNormPooling()
    if name == "attention":
        if vocab_size is None:
            raise ValueError("'attention' pooling requires vocab_size")
        return AttentionPooling(vocab_size)

    raise ValueError(
        f"unknown pooling '{name}'; expected one of: "
        "sum, max, p-norm, attention"
    )