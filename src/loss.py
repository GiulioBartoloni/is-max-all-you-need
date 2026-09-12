"""
loss.py - the SPLADE training objective.

The objective has two parts: the model learns the ranking from a cross-encoder teacher, and a penalty keeps the vectors sparse.
"""

import torch


def margin_mse(pos_score, neg_score, teacher_pos_score, teacher_neg_score):
    """Return the mean squared error of the student margin against the teacher.

    The teacher scores come pre-computed from a cross-encoder. 
    The student only has to reproduce the difference between the positive and the negative score. 
    It does not have to reproduce the two absolute values, so the two models do not need the same scale.
    """
    student_margin = pos_score - neg_score
    teacher_margin = teacher_pos_score - teacher_neg_score

    return torch.nn.functional.mse_loss(student_margin, teacher_margin)


def flops(vectors):
    """Return the FLOPS estimate of a batch of sparse vectors.

    For each vocabulary term, take its mean value over the batch and square it.
    """
    return (vectors.mean(dim=0) ** 2).sum()


class SpladeLoss(torch.nn.Module):
    """The ranking loss plus a weighted FLOPS penalty on each side.

    Attributes:
        lambda_q: weight of the penalty on the query vectors.
        lambda_d: weight of the penalty on the document vectors.
    """

    def __init__(self, lambda_q, lambda_d):
        super().__init__()
        self.lambda_q = lambda_q
        self.lambda_d = lambda_d

    def forward(self, pos_score, neg_score, teacher_pos_score, teacher_neg_score,
                query_vectors, doc_vectors):
        """Return the total loss and its three parts."""
        ranking = margin_mse(pos_score, neg_score, teacher_pos_score, teacher_neg_score)
        query_flops = flops(query_vectors)
        doc_flops = flops(doc_vectors)

        total = ranking + (self.lambda_q * query_flops) + (self.lambda_d * doc_flops)

        return total, ranking, query_flops, doc_flops
