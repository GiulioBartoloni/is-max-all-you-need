import torch

def margin_mse(pos_score, neg_score, teacher_pos_score, teacher_neg_score):
    student_margin = pos_score - neg_score
    teacher_margin = teacher_pos_score - teacher_neg_score
    
    return torch.nn.functional.mse_loss(student_margin, teacher_margin)
     
def flops(vectors):
    return (vectors.mean(dim=0) ** 2).sum()

class SpladeLoss(torch.nn.Module):
    def __init__(self, lambda_q, lambda_d):
        super().__init__()
        self.lambda_q = lambda_q
        self.lambda_d = lambda_d
    
    def forward(self, pos_score, neg_score, teacher_pos_score, teacher_neg_score, query_vectors, doc_vectors):
        ranking = margin_mse(pos_score, neg_score, teacher_pos_score, teacher_neg_score)
        query_flops = flops(query_vectors)
        doc_flops = flops(doc_vectors)
        
        total = ranking + (self.lambda_q * query_flops) + (self.lambda_d * doc_flops)
        
        return total, ranking, query_flops, doc_flops