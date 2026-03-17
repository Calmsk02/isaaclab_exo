import torch

def normalize(v, eps: float = 1e-8):
    return v / (torch.norm(v, dim=-1, keepdim=True) + eps)

def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))