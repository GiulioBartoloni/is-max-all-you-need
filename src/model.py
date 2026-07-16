import torch
from transformers import AutoModelForMaskedLM
from pooling import make_pooling
    
class Splade(torch.nn.Module):
    def __init__(self, pooling_name):
        super().__init__()
        self.backbone = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
        vocab_size = self.backbone.config.vocab_size
        self.query_pool = make_pooling(pooling_name, vocab_size)
        self.doc_pool = make_pooling(pooling_name, vocab_size)
        
    def encode(self, input_ids, attention_mask, which):
        logits = self.backbone(input_ids=input_ids, attention_mask=attention_mask).logits
        
        log_saturated = torch.log1p(torch.relu(logits))
        
        if which=='query':
            return self.query_pool(log_saturated, attention_mask)
        
        return self.doc_pool(log_saturated, attention_mask)
    
    def score(self, encoded_query, encoded_doc):
        
        return (encoded_query * encoded_doc).sum(dim=1)
    
    def forward(self, query_input_ids, query_attention_mask, positive_doc_input_ids, positive_doc_attention_mask, negative_doc_input_ids, negative_doc_attention_mask):    
        encoded_query = self.encode(query_input_ids, query_attention_mask, 'query')
        encoded_positive_doc = self.encode(positive_doc_input_ids, positive_doc_attention_mask, 'doc')
        encoded_negative_doc = self.encode(negative_doc_input_ids, negative_doc_attention_mask, 'doc')
            
        positive_score = self.score(encoded_query, encoded_positive_doc)
        negative_score = self.score(encoded_query, encoded_negative_doc)
        
        return positive_score, negative_score, encoded_query, encoded_positive_doc, encoded_negative_doc