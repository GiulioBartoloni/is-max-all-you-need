import torch

class TripleDataset(torch.utils.data.Dataset):
    def __init__(self, triples, query_lookup, doc_lookup):
        self.triples = triples
        self.query_lookup = query_lookup
        self.doc_lookup = doc_lookup

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, i):
        pos_score, neg_score, qid, pos_pid, neg_pid = self.triples[i]
        return {
            "query":       self.query_lookup[qid],
            "positive":    self.doc_lookup[pos_pid],
            "negative":    self.doc_lookup[neg_pid],
            "teacher_pos": pos_score,
            "teacher_neg": neg_score,
        }
        
def collate_fn_factory(tokenizer, max_length=128):
    """Build a collate_fn that tokenizes a batch of triple dicts.

    Returns a function suitable for DataLoader's ``collate_fn`` argument. It
    takes a list of examples (each the dict produced by ``TripleDataset``) and
    tokenizes the query/positive/negative texts with padding, returning batched
    tensors plus the teacher scores.
    """
    def collate_fn(batch):
        queries   = [ex["query"] for ex in batch]
        positives = [ex["positive"] for ex in batch]
        negatives = [ex["negative"] for ex in batch]

        def tok(texts):
            return tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

        q = tok(queries)
        p = tok(positives)
        n = tok(negatives)

        teacher_pos = torch.tensor([ex["teacher_pos"] for ex in batch], dtype=torch.float)
        teacher_neg = torch.tensor([ex["teacher_neg"] for ex in batch], dtype=torch.float)

        return {
            "query_input_ids":       q["input_ids"],
            "query_attention_mask":  q["attention_mask"],
            "pos_input_ids":         p["input_ids"],
            "pos_attention_mask":    p["attention_mask"],
            "neg_input_ids":         n["input_ids"],
            "neg_attention_mask":    n["attention_mask"],
            "teacher_pos":           teacher_pos,
            "teacher_neg":           teacher_neg,
        }

    return collate_fn


def make_dataloader(triples, query_lookup, doc_lookup, tokenizer,
                    batch_size=8, shuffle=True, max_length=128):
    """Build a DataLoader yielding tokenized triple batches."""
    dataset = TripleDataset(triples, query_lookup, doc_lookup)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_factory(tokenizer, max_length),
    )