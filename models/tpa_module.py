import torch
import torch.nn as nn
import torch.nn.functional as F

class ProductionTPA(nn.Module):
    """Temporal Prototype Attention (TPA) module"""
    def __init__(self, dim: int, num_prototypes: int = 24, heads: int = 4, dropout: float = 0.1,
                 temperature: float = 0.07, topk_ratio: float = 0.5):
        super().__init__()
        assert dim % heads == 0

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.topk_ratio = topk_ratio

        self.proto = nn.Parameter(torch.randn(num_prototypes, dim) * 0.02)
        self.pre_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.fuse = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        P = self.num_prototypes

        x_norm = self.pre_norm(x)

        K = self.k_proj(x_norm)
        V = self.v_proj(x_norm)
        Qp = self.q_proj(self.proto).unsqueeze(0).expand(B, -1, -1)

        def split_heads(t, length):
            return t.view(B, length, self.heads, self.head_dim).transpose(1, 2)

        Qh = split_heads(Qp, P)
        Kh = split_heads(K, T)
        Vh = split_heads(V, T)

        Qh = F.normalize(Qh, dim=-1)
        Kh = F.normalize(Kh, dim=-1)

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / self.temperature
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        proto_tokens = torch.matmul(attn, Vh)
        proto_tokens = proto_tokens.transpose(1, 2).contiguous().view(B, P, D)

        prototype_scores = torch.sum(proto_tokens**2, dim=2)

        topk = max(1, int(P * self.topk_ratio))
        _, topk_indices = torch.topk(prototype_scores, k=topk, dim=1)

        D = proto_tokens.size(2)
        topk_indices = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        topk_tokens = torch.gather(proto_tokens, 1, topk_indices)
        
        z_tpa = topk_tokens.mean(dim=1)
        
        z = self.fuse(z_tpa)

        return z