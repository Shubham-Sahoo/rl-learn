from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    """Configurable multi-layer perceptron.

    Implemented in: Module 02, Assignment 2.
    Used in: all deep RL modules.
    """

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: list[int] = (256, 256), activation=nn.ReLU):
        super().__init__()
        # TODO (Module 02, A2): build nn.Sequential from input_dim → hidden_dims → output_dim
        # Use activation() between layers; no activation on final layer
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DuelingNet(nn.Module):
    """Dueling DQN architecture: shared trunk → V(s) and A(s,a) heads.

    Q(s,a) = V(s) + A(s,a) - mean_a'[A(s,a')]

    Implemented in: Module 02, Assignment 3.
    Used in: Module 02 A3.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        # TODO (Module 02, A3): shared trunk MLP, then separate value and advantage heads
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values of shape (batch, n_actions)."""
        # TODO: combine V and A with mean-subtraction trick
        raise NotImplementedError


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic for discrete action spaces.

    Implemented in: Module 03, Assignment 2.
    Used in: Module 03 A2/A3.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        # TODO (Module 03, A2): shared trunk + actor head (logits) + critic head (scalar)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action_logits, state_value). Shapes: (B, n_actions), (B,)."""
        raise NotImplementedError


class GaussianPolicyHead(nn.Module):
    """Squashed Gaussian policy for continuous control (SAC).

    a = tanh(mu + sigma * eps),  eps ~ N(0, I)
    log_prob corrected for tanh squashing.

    Implemented in: Module 05, Assignment 2.
    Used in: Module 05 A2.
    """

    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # TODO (Module 05, A2): trunk MLP → mean head + log_std head
        raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_std) before squashing."""
        raise NotImplementedError

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action, log_prob) using reparameterization trick.

        action = tanh(mean + std * eps)
        log_prob -= sum(log(1 - action^2 + eps))   # squashing correction
        """
        raise NotImplementedError


class TwinQNetwork(nn.Module):
    """Twin Q-networks for SAC (reduces overestimation).

    Implemented in: Module 05, Assignment 2.
    Used in: Module 05 A2.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # TODO (Module 05, A2): two independent MLP Q-networks
        raise NotImplementedError

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (Q1, Q2) both of shape (batch,)."""
        raise NotImplementedError
