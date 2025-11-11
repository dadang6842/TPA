import copy
import random
import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from .config import Config, SEED


def train_one_epoch(model, loader, opt, cfg: Config):
    model.train()
    loss_sum = 0
    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)

        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            loss_sum += loss.item() * y.size(0)

    return {
        "loss": loss_sum / total if total > 0 else 0,
        "acc": correct / total if total > 0 else 0
    }


@torch.no_grad()
def evaluate(model, loader, cfg: Config):
    model.eval()
    ys, ps = [], []

    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)
        logits = model(x)
        ps.append(logits.argmax(dim=-1).cpu().numpy())
        ys.append(y.cpu().numpy())

    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    return acc, f1


def train_model(model, train_loader, val_loader, test_loader, cfg: Config, model_name: str):
    print(f"\n[Training {model_name}]")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_acc, best_wts = 0.0, None
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        stats = train_one_epoch(model, train_loader, opt, cfg)
        val_acc, val_f1 = evaluate(model, val_loader, cfg)

        if val_acc > best_acc + cfg.min_delta:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: Train Acc={stats['acc']:.4f}, Val Acc={val_acc:.4f}, F1={val_f1:.4f}")

        if patience_counter >= cfg.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_wts:
        model.load_state_dict(best_wts)

    test_acc, test_f1 = evaluate(model, test_loader, cfg)

    print(f"\n[{model_name} Results]")
    print(f"  Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    return {
        'Test_Accuracy': float(test_acc), 
        'Test_F1_Score': float(test_f1)
    }