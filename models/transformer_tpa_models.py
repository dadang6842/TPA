import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from tpa_module import ProductionTPA

class PositionalEncoding(nn.Module):
    """Positional Encoding layer for Transformer."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerBackbone(nn.Module):
    """Transformer Encoder Backbone for temporal feature extraction."""
    def __init__(self, in_ch: int, d_model: int,
                 num_layers: int = 2, n_heads: int = 4, ff_dim: int = 256,
                 dropout: float = 0.1, max_seq_len: int = 200):
        super().__init__()

        self.d_model = d_model

        self.input_projection = nn.Linear(in_ch, d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.norm(x)
        return x

class GAPModel(nn.Module):
    """Transformer + Global Average Pooling (GAP) Model."""
    def __init__(self, d_model: int, num_classes: int, in_ch: int):
        super().__init__()
        self.backbone = TransformerBackbone(in_ch=in_ch, d_model=d_model) 
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = features.mean(dim=1)
        logits = self.fc(pooled)
        return logits

class TPAModel(nn.Module):
    """Transformer + Temporal Prototype Attention (TPA) Model."""
    def __init__(self, d_model: int, num_classes: int, in_ch: int, tpa_config: Dict):
        super().__init__()
        self.backbone = TransformerBackbone(in_ch=in_ch, d_model=d_model)
        self.tpa = ProductionTPA(dim=d_model, **tpa_config)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        z = self.tpa(features)
        logits = self.classifier(z)
        return logits

class GatedTPAModel(nn.Module):
    """Transformer + Gated Fusion of GAP and TPA Model."""
    def __init__(self, d_model: int, num_classes: int, in_ch: int, tpa_config: Dict):
        super().__init__()
        self.backbone = TransformerBackbone(in_ch=in_ch, d_model=d_model)
        self.tpa = ProductionTPA(dim=d_model, **tpa_config)
        
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)

        z_gap = features.mean(dim=1)
        z_tpa = self.tpa(features)

        gate_input = torch.cat([z_gap, z_tpa], dim=-1)
        g = self.gate(gate_input)

        z = g * z_gap + (1 - g) * z_tpa

        logits = self.classifier(z)
        return logits

def create_model(model_name: str, in_ch: int, num_classes: int) -> nn.Module:
    """Creates a Transformer-based model instance (GAP, TPA, or Gated-TPA)."""
    
    d_model = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tpa_config = {
        'num_prototypes': 24, 
        'heads': 4,
        'dropout': 0.1,
        'temperature': 0.07,
        'topk_ratio': 0.5 
    }
    
    if model_name == "GAP":
        model = GAPModel(d_model=d_model, num_classes=num_classes, in_ch=in_ch)
    elif model_name == "TPA":
        model = TPAModel(d_model=d_model, num_classes=num_classes, in_ch=in_ch, tpa_config=tpa_config)
    elif model_name == "Gated-TPA":
        model = GatedTPAModel(d_model=d_model, num_classes=num_classes, in_ch=in_ch, tpa_config=tpa_config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    return model.to(device).float()