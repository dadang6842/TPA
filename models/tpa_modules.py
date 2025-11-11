import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class ProductionTPA(nn.Module):
    def __init__(self, dim, num_prototypes=16, heads=4, dropout=0.1,
                 temperature=0.07):
        super().__init__()
        assert dim % heads == 0

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.num_prototypes = num_prototypes
        self.temperature = temperature

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

    def forward(self, x):
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

        z_tpa = proto_tokens.mean(dim=1)

        z = self.fuse(z_tpa)

        return z

class GAPModel(nn.Module):
    def __init__(self, backbone: nn.Module, d_model=128, num_classes=6):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        pooled = features.mean(dim=1)
        logits = self.fc(pooled)
        return logits

class TPAModel(nn.Module):
    def __init__(self, backbone: nn.Module, d_model=128, num_classes=6, tpa_config: Dict = None):
        super().__init__()
        self.backbone = backbone
        
        tpa_config = tpa_config if tpa_config is not None else {}
        self.tpa = ProductionTPA(dim=d_model, **tpa_config)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        z = self.tpa(features)
        logits = self.classifier(z)
        return logits

class GatedTPAModel(nn.Module):
    def __init__(self, backbone: nn.Module, d_model=128, num_classes=6, tpa_config: Dict = None):
        super().__init__()
        self.backbone = backbone
        
        tpa_config = tpa_config if tpa_config is not None else {}
        self.tpa = ProductionTPA(dim=d_model, **tpa_config)

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        features = self.backbone(x)

        z_gap = features.mean(dim=1)

        z_tpa = self.tpa(features)

        gate_input = torch.cat([z_gap, z_tpa], dim=-1)
        g = self.gate(gate_input)

        z = g * z_gap + (1 - g) * z_tpa

        logits = self.classifier(z)
        return logits