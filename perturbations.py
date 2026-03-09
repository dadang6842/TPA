"""
Included perturbations
1) Temporal scaling (tempo shift)
2) Additive Gaussian noise
3) Additive bias drift
4) Multiplicative scale drift
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# =============================================================================
# 1) Temporal scaling
# =============================================================================
def temporal_scaling_one(x_ct: np.ndarray, speed: float) -> np.ndarray:
    """
    Apply temporal scaling to a single sample x_ct of shape (C, T).
    """
    if speed <= 0:
        raise ValueError(f"`speed` must be positive, got {speed}")

    if x_ct.ndim != 2:
        raise ValueError(f"`x_ct` must have shape (C, T), got {x_ct.shape}")

    c, t = x_ct.shape
    t_new = max(4, int(round(t / speed)))

    old_grid = np.linspace(0.0, 1.0, t, dtype=np.float32)
    new_grid = np.linspace(0.0, 1.0, t_new, dtype=np.float32)

    x_tmp = np.zeros((c, t_new), dtype=np.float32)
    for ch in range(c):
        x_tmp[ch] = np.interp(new_grid, old_grid, x_ct[ch]).astype(np.float32)

    restore_grid = np.linspace(0.0, 1.0, t, dtype=np.float32)
    x_out = np.zeros((c, t), dtype=np.float32)
    for ch in range(c):
        x_out[ch] = np.interp(restore_grid, new_grid, x_tmp[ch]).astype(np.float32)

    return x_out


def apply_temporal_scaling(X_nct: np.ndarray, speed: float) -> np.ndarray:
    """
    Apply temporal scaling to a batch X of shape (N, C, T).
    """
    if X_nct.ndim != 3:
        raise ValueError(f"`X_nct` must have shape (N, C, T), got {X_nct.shape}")

    out = np.empty_like(X_nct, dtype=np.float32)
    for i in range(X_nct.shape[0]):
        out[i] = temporal_scaling_one(X_nct[i], speed)
    return out


# =============================================================================
# 2) Additive Gaussian noise
# =============================================================================
def apply_additive_gaussian_noise(
    X_nct: np.ndarray,
    level: float,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Add channel-wise Gaussian noise to X of shape (N, C, T).
    """
    if X_nct.ndim != 3:
        raise ValueError(f"`X_nct` must have shape (N, C, T), got {X_nct.shape}")

    if rng is None:
        rng = np.random.default_rng()

    X = X_nct.astype(np.float32).copy()

    sigma_c = X.transpose(1, 0, 2).reshape(X.shape[1], -1).std(axis=1).astype(np.float32)
    sigma_c = sigma_c + 1e-8

    noise = rng.normal(loc=0.0, scale=1.0, size=X.shape).astype(np.float32)
    X_noisy = X + noise * (level * sigma_c[None, :, None])

    return X_noisy.astype(np.float32)


# =============================================================================
# 3) Additive bias drift
# =============================================================================
def apply_additive_bias_drift(X_nct: np.ndarray, level: float) -> np.ndarray:
    """
    Add constant channel-wise bias to X of shape (N, C, T).
    """
    if X_nct.ndim != 3:
        raise ValueError(f"`X_nct` must have shape (N, C, T), got {X_nct.shape}")

    X = X_nct.astype(np.float32).copy()

    sigma_c = X.transpose(1, 0, 2).reshape(X.shape[1], -1).std(axis=1).astype(np.float32)
    sigma_c = sigma_c + 1e-8

    bias = (level * sigma_c).astype(np.float32)
    X_bias = X + bias[None, :, None]

    return X_bias.astype(np.float32)


# =============================================================================
# 4) Multiplicative scale drift
# =============================================================================
def apply_multiplicative_scale_drift(X_nct: np.ndarray, level: float) -> np.ndarray:
    """
    Apply multiplicative scaling to X of shape (N, C, T):
    """
    if X_nct.ndim != 3:
        raise ValueError(f"`X_nct` must have shape (N, C, T), got {X_nct.shape}")

    X = X_nct.astype(np.float32)
    return (X * (1.0 + float(level))).astype(np.float32)


# =============================================================================
# Unified interface
# =============================================================================
def apply_perturbation(
    X_nct: np.ndarray,
    perturbation_type: str,
    level: float,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Unified wrapper for perturbations.
    """
    p = perturbation_type.lower()

    if p == "tempo":
        return apply_temporal_scaling(X_nct, speed=level)
    elif p == "gaussian":
        return apply_additive_gaussian_noise(X_nct, level=level, rng=rng)
    elif p == "bias":
        return apply_additive_bias_drift(X_nct, level=level)
    elif p == "scale":
        return apply_multiplicative_scale_drift(X_nct, level=level)
    else:
        raise ValueError(
            f"Unknown perturbation_type: {perturbation_type}. "
            f"Choose from ['tempo', 'gaussian', 'bias', 'scale']"
        )


# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Example dummy data: N=8 samples, C=9 channels, T=128 timesteps
    X = rng.normal(size=(8, 9, 128)).astype(np.float32)

    X_tempo = apply_perturbation(X, perturbation_type="tempo", level=0.90)
    X_gauss = apply_perturbation(X, perturbation_type="gaussian", level=0.10, rng=rng)
    X_bias  = apply_perturbation(X, perturbation_type="bias", level=0.10)
    X_scale = apply_perturbation(X, perturbation_type="scale", level=0.10)

    print("Original :", X.shape)
    print("Tempo    :", X_tempo.shape)
    print("Gaussian :", X_gauss.shape)
    print("Bias     :", X_bias.shape)
    print("Scale    :", X_scale.shape)