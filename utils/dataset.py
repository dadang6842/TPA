import os
import numpy as np
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

from .config import Config, SEED


class PreloadedDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__()
        self.X = torch.from_numpy(X).float()

        if y.min() >= 1:
            y = y - 1

        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_preprocess_dataset(
    base_dir: str, 
    dataset_name: str, 
    dataset_type: str, 
    cfg: Config
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    
    dataset_dir = os.path.join(base_dir, dataset_name)
    
    X_train_raw = np.load(os.path.join(dataset_dir, "X_train.npy"))
    y_train = np.load(os.path.join(dataset_dir, "y_train.npy"))
    X_test_raw = np.load(os.path.join(dataset_dir, "X_test.npy"))
    y_test = np.load(os.path.join(dataset_dir, "y_test.npy"))

    if dataset_type == 'UCI':
        X_train = np.transpose(X_train_raw, (0, 2, 1))
        X_test = np.transpose(X_test_raw, (0, 2, 1))
    else:
        X_train = X_train_raw
        X_test = X_test_raw
    
    train_dataset = PreloadedDataset(X_train, y_train)
    test_dataset = PreloadedDataset(X_test, y_test)
    
    n_total = len(train_dataset)
    indices = np.arange(n_total)
    y_labels = train_dataset.y.numpy()

    train_indices, val_indices = train_test_split(
        indices,
        test_size=cfg.val_split,
        random_state=SEED,
        stratify=y_labels
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    g = torch.Generator(device='cpu').manual_seed(SEED)
    train_loader = DataLoader(
        train_subset, 
        cfg.batch_size, 
        shuffle=True,
        num_workers=cfg.num_workers, 
        generator=g
    )
    val_loader = DataLoader(
        val_subset, 
        cfg.batch_size, 
        num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        cfg.batch_size, 
        num_workers=cfg.num_workers
    )
    
    in_channels = X_train.shape[-1]
    max_seq_len = X_train.shape[1]

    return train_loader, val_loader, test_loader, in_channels, max_seq_len