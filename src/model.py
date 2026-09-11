"""
model.py -- the SPLADE encoder.

The model turns a text into one sparse vector over the BERT vocabulary. It
then scores a query against a document with a dot product of two vectors.
"""

import torch
from transformers import AutoModelForMaskedLM

from pooling import make_pooling


class Splade(torch.nn.Module):
    """A masked language model with a sparse vocabulary vector as its output.

    A forward pass has three steps:

    1. The DistilBERT masked language model head gives one vocabulary-sized
       prediction per token.
    2. log(1 + relu(logits)) drops the negative predictions and damps the
       large ones. The relu makes the vector sparse: only the terms with a
       positive prediction stay in it.
    3. A pooling layer collapses the per-token predictions into one vector.

    Queries and documents get two separate pooling layers. The two sides have
    very different lengths, so the best pooling can also differ, which is one
    of the questions of the study. The layers without parameters behave the
    same on both sides.

    Attributes:
        backbone: the DistilBERT masked language model.
        query_pool: pooling layer for the queries.
        doc_pool: pooling layer for the documents.
    """

    def __init__(self, pooling_name):
        super().__init__()
        self.backbone = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
        vocab_size = self.backbone.config.vocab_size
        self.query_pool = make_pooling(pooling_name, vocab_size)
        self.doc_pool = make_pooling(pooling_name, vocab_size)

    def encode(self, input_ids, attention_mask, which):
        """Encode a batch of texts into sparse vectors of shape (batch, vocab).

        Args:
            which: "query" for the query pooling layer, "doc" for the document
                one.
        """
        logits = self.backbone(input_ids=input_ids,
                               attention_mask=attention_mask).logits
        saturated = torch.log1p(torch.relu(logits))

        pool = self.query_pool if which == "query" else self.doc_pool
        return pool(saturated, attention_mask)

    def score(self, encoded_query, encoded_doc):
        """Score each query against its document with a dot product."""
        return (encoded_query * encoded_doc).sum(dim=1)

    def forward(self, query_input_ids, query_attention_mask,
                pos_input_ids, pos_attention_mask,
                neg_input_ids, neg_attention_mask):
        """Score one batch of training triples.

        A triple is a query, a document that is relevant to it, and a document
        that is not.

        Returns:
            The positive score, the negative score, and the three vectors. The
            loss needs the vectors for its sparsity term.
        """
        encoded_query = self.encode(query_input_ids, query_attention_mask, "query")
        encoded_pos = self.encode(pos_input_ids, pos_attention_mask, "doc")
        encoded_neg = self.encode(neg_input_ids, neg_attention_mask, "doc")

        pos_score = self.score(encoded_query, encoded_pos)
        neg_score = self.score(encoded_query, encoded_neg)

        return pos_score, neg_score, encoded_query, encoded_pos, encoded_neg
