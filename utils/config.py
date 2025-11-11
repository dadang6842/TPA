import random
import numpy as np
import torch
from dataclasses import dataclass

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

@dataclass
class Config:
    data_dir: str = ""
    save_dir: str = ""
    dataset_type: str = ""
    num_classes: int = 6
    
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    label_smoothing: float = 0.05

    patience: int = 20
    min_delta: float = 0.0001
    val_split: float = 0.2

    d_model: int = 128
    in_channels: int = 9
    max_seq_len: int = 128

    tpa_num_prototypes: int = 16
    tpa_heads: int = 4
    tpa_dropout: float = 0.1
    tpa_temperature: float = 0.07

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2