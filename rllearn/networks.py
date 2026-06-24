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
        layers = []
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingNet(nn.Module):
    """Dueling DQN architecture: shared trunk → V(s) and A(s,a) heads.

    Q(s,a) = V(s) + A(s,a) - mean_a'[A(s,a')]

    Implemented in: Module 02, Assignment 3.
    Used in: Module 02 A3.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self.advantage_head = nn.Linear(hidden_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values of shape (batch, n_actions)."""
        feat = self.trunk(x)
        V = self.value_head(feat)           # (B, 1)
        A = self.advantage_head(feat)       # (B, n_actions)
        Q = V + A - A.mean(dim=1, keepdim=True)
        return Q


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic for discrete action spaces.

    Implemented in: Module 03, Assignment 2.
    Used in: Module 03 A2/A3.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_dim, n_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action_logits, state_value). Shapes: (B, n_actions), (B,)."""
        feat = self.trunk(x)
        logits = self.actor_head(feat)
        value = self.critic_head(feat).squeeze(-1)
        return logits, value


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
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_std) before squashing."""
        feat = self.net(obs)
        mean = self.mean_head(feat)
        log_std = torch.clamp(self.log_std_head(feat), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action, log_prob) using reparameterization trick.

        action = tanh(mean + std * eps)
        log_prob -= sum(log(1 - action^2 + eps))   # squashing correction
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return y_t, log_prob


class TwinQNetwork(nn.Module):
    """Twin Q-networks for SAC (reduces overestimation).

    Implemented in: Module 05, Assignment 2.
    Used in: Module 05 A2.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (Q1, Q2) both of shape (batch,)."""
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa).squeeze(-1)
