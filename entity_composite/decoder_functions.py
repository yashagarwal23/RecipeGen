import torch
import torch.nn.functional as F

def attention_combined(hidden_output, prev_output, ws, vs):
    W_S = prev_output.matmul(ws)
    addition = torch.tanh(W_S + hidden_output.unsqueeze(1))
    attention_score = F.softmax(addition.matmul(vs), dim=1)
    attention_output = torch.sum(prev_output*attention_score, dim=1)
    return torch.cat([hidden_output, attention_output], dim=1)
