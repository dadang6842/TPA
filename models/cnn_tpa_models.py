import torch
import torch.nn as nn
from typing import Dict, Tuple, Union
from tpa_module import ProductionTPA

class ConvBNAct(nn.Module):
    """Convolution, Batch Normalization, and Activation block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: Union[int, None] = None, groups: int = 1):
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        
        self.c = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, 
            padding=padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.c(x)))

class MultiPathCNN(nn.Module):
    """CNN Backbone: Multi-scale temporal feature extraction."""
    def __init__(self, in_ch: int, d_model: int = 128, branches: Tuple[int, ...] = (3,5,9,15), stride: int = 2):
        super().__init__()
        
        hidden_channels = d_model
        
        self.pre = ConvBNAct(in_ch, hidden_channels, kernel_size=1)
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(hidden_channels, hidden_channels, k, stride=stride, groups=hidden_channels), 
                ConvBNAct(hidden_channels, hidden_channels, kernel_size=1)
            )
            for k in branches
        ])
        
        self.post = ConvBNAct(len(branches) * hidden_channels, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        fmap = self.post(torch.cat([b(self.pre(x)) for b in self.branches], dim=1))
        return fmap.transpose(1, 2)

class GAPModel(nn.Module):
    """CNN + Global Average Pooling (GAP) Model."""
    def __init__(self, d_model: int, num_classes: int, in_ch: int):
        super().__init__()
        self.backbone = MultiPathCNN(in_ch=in_ch, d_model=d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = features.mean(dim=1)
        logits = self.fc(pooled)
        return logits

class TPAModel(nn.Module):
    """CNN + Temporal Prototype Attention (TPA) Model."""
    def __init__(self, d_model: int, num_classes: int, in_ch: int, tpa_config: Dict):
        super().__init__()
        self.backbone = MultiPathCNN(in_ch=in_ch, d_model=d_model)
        self.tpa = ProductionTPA(dim=d_model, **tpa_config)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        z = self.tpa(features)
        logits = self.classifier(z)
        return logits

class GatedTPAModel(nn.Module):
    """CNN + Gated Fusion of GAP and TPA Model."""
    def __init__(self, d_model: int, num_classes: int, in_ch: int, tpa_config: Dict):
        super().__init__()
        self.backbone = MultiPathCNN(in_ch=in_ch, d_model=d_model)
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
    """Creates a CNN-based model instance (GAP, TPA, or Gated-TPA)."""
    
    d_model: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    tpa_config = {
        'num_prototypes': 24, 
        'heads': 4,
        'dropout': 0.1,
        'temperature':  0.07,
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