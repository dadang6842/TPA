import torch
import torch.nn as nn
from typing import Dict
from tpa_module import ProductionTPA

class BiLSTMBackbone(nn.Module):
    """LSTM Backbone: Bi-directional LSTM"""
    def __init__(self, in_ch: int, d_model: int, 
                 hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1): 
        super().__init__()
        self.d_model = d_model

        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0 
        )

        self.projection = nn.Linear(hidden_dim * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = self.projection(lstm_out)
        out = self.layer_norm(out)
        out = self.dropout_layer(out)
        
        return out

class GAPModel(nn.Module):
    """LSTM + Global Average Pooling (GAP) Model."""
    def __init__(self, d_model: int, num_classes: int, in_ch: int):
        super().__init__()
        self.backbone = BiLSTMBackbone(in_ch=in_ch, d_model=d_model) 
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = features.mean(dim=1)
        logits = self.fc(pooled)
        return logits

class TPAModel(nn.Module):
    """LSTM + Temporal Prototype Attention (TPA) Model."""
    def __init__(self, d_model: int, num_classes: int, in_ch: int, tpa_config: Dict):
        super().__init__()
        self.backbone = BiLSTMBackbone(in_ch=in_ch, d_model=d_model)
        self.tpa = ProductionTPA(dim=d_model, **tpa_config)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        z = self.tpa(features)
        logits = self.classifier(z)
        return logits

class GatedTPAModel(nn.Module):
    """LSTM + Gated Fusion of GAP and TPA Model."""
    def __init__(self, d_model: int, num_classes: int, in_ch: int, tpa_config: Dict):
        super().__init__()
        self.backbone = BiLSTMBackbone(in_ch=in_ch, d_model=d_model)
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
    """Creates an LSTM-based model instance (GAP, TPA, or Gated-TPA)."""

    d_model = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # TPA Config
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