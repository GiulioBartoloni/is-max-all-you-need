"""
pooling.py -- masked pooling layers for the SPLADE encoder.

The backbone gives one vocabulary-sized prediction per input token. A pooling
layer collapses those predictions into a single vector for the whole query or
document. This is the step that this project generalizes: `sum` and `max` are
the two ends of one family, and the learnable variants below sit between them
or replace them.

All the layers share one interface:

    x     token predictions, shape (batch, seq_len, vocab)
    mask  attention mask, shape (batch, seq_len). A 1 marks a real token and a
          0 marks padding.
    out   one vector per sequence, shape (batch, vocab)

Each layer must ignore the padded positions. If it does not, the result of a
sequence changes with the longest sequence in its batch.
"""

import torch


class SumPooling(torch.nn.Module):
    """Add the token predictions together. No parameters.

    This is the weak end of the family: the paper reports that max pooling
    clearly beats it.
    """

    def forward(self, x, mask):
        # Padding adds 0 to a sum, so one multiplication is enough here.
        return (x * mask.unsqueeze(-1)).sum(dim=1)


class MaxPooling(torch.nn.Module):
    """Keep the largest prediction of each vocabulary term. No parameters.

    This is the baseline of the study, and the pooling of the paper.
    """

    def forward(self, x, mask):
        # -inf keeps the padded positions out of the maximum. A 0 fill would
        # not: a 0 wins as soon as every real value is negative.
        masked = x.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        return masked.max(dim=1).values


class PNormPooling(torch.nn.Module):
    """Power mean with a learnable exponent p.

    The layer computes ``(mean_i x_i ** p) ** (1 / p)`` over the real tokens.
    The exponent selects a point in the pooling family: p = 1 gives the masked
    mean, and a large p comes close to the maximum. Because p is a trainable
    parameter, the model learns that point, and the study does not fix it.

    The forward pass works in log space. A direct ``x ** p`` overflows for a
    large p, but the log space form only needs an exponential at the end. This
    form needs a non-negative x, which the log(1 + relu(logits)) saturation of
    model.py guarantees.

    Attributes:
        p: the learnable exponent. It starts at 10.0, which is already close to
            max pooling, so training starts near the known-good baseline.
    """

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor(10.0))

    def forward(self, x, mask):
        eps = 1e-10                                   # keeps log(0) finite
        n_tokens = mask.float().sum(dim=1, keepdim=True)
        log_x = torch.log(x + eps)

        # -1e9 is still very negative after the multiplication by p, so the
        # padded positions add about 0 to the sum. train.py keeps p above 0.5,
        # which also keeps the sign of the product.
        masked = log_x.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

        # Log space form of the power mean: the logsumexp is the log of
        # sum_i x_i ** p, and the subtraction turns that sum into a mean.
        log_sum = torch.logsumexp(masked * self.p, dim=1)
        log_mean = log_sum - torch.log(n_tokens)

        return torch.exp(log_mean / self.p)


class AttentionPooling(torch.nn.Module):
    """Weighted sum with the weights of a learned scorer.

    A linear layer gives one score per token. A softmax over the sequence turns
    the scores into weights that add up to 1, and the layer combines the token
    predictions with those weights.

    This variant has the most parameters, but one weight applies to a whole
    token. The layer therefore cannot keep a high value for one vocabulary term
    and drop the other terms of the same token. It also stays a weighted mean,
    so it cannot isolate a single peak like max pooling does.

    Attributes:
        scorer: linear layer from one token prediction to one score.
    """

    def __init__(self, vocab_size):
        """Build the scorer. vocab_size is the feature width of x."""
        super().__init__()
        self.scorer = torch.nn.Linear(vocab_size, 1)

    def forward(self, x, mask):
        scores = self.scorer(x)

        # The smallest value of the dtype makes the softmax give about 0 to the
        # padded positions.
        masked = scores.masked_fill(mask.unsqueeze(-1) == 0,
                                    torch.finfo(scores.dtype).min)
        weights = torch.softmax(masked, dim=1)

        return (x * weights).sum(dim=1)


def make_pooling(name, vocab_size=None):
    """Build a pooling layer by name.

    One entry point lets train.py select a variant from a configuration string,
    without knowledge of the different constructors.

    Args:
        name: "sum", "max", "p-norm" or "attention".
        vocab_size: feature width of x. Only "attention" needs it.

    Raises:
        ValueError: if the name is unknown, or if "attention" comes without a
            vocab_size.
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
