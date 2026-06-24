from __future__ import annotations
import torch
import numpy as np
import random


def set_seed(seed: int):
    """Set seed for reproducibility across torch, numpy, random."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_returns(rewards: list[float], gamma: float) -> list[float]:
    """Compute discounted returns G_t = sum_{t'>=t} gamma^{t'-t} * r_{t'}.

    Implemented in: Module 03, Assignment 1.
    Used in: Module 03 A1.
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def compute_gae(rewards: list[float], values: list[float], dones: list[bool],
                gamma: float = 0.99, lam: float = 0.95) -> torch.Tensor:
    """Generalized Advantage Estimation.

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t = sum_{l>=0} (gamma * lam)^l * delta_{t+l}

    Implemented in: Module 03, Assignment 2.
    Used in: Module 03 A2/A3; Module 06 A2.

    Args:
        rewards: list of length T
        values:  list of length T+1 (last entry is V(s_T) bootstrap)
        dones:   list of length T

    Returns:
        advantages: Tensor of shape (T,)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae
    return torch.FloatTensor(advantages)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of variance in y_true explained by y_pred.

    EV = 1 - Var(y_true - y_pred) / Var(y_true)
    EV=1 means perfect prediction; EV<0 means worse than mean prediction.
    """
    var_y = np.var(y_true)
    return float("nan") if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
