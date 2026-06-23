from __future__ import annotations
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """Circular replay buffer for off-policy RL (DQN, SAC).

    Implemented in: Module 02, Assignment 2.
    Used in: Module 02 A2/A3; Module 05 A2.
    """

    def __init__(self, capacity: int):
        # TODO (Module 02, A2): initialize a deque with maxlen=capacity
        # Store transitions as (state, action, reward, next_state, done)
        raise NotImplementedError

    def push(self, state, action: int, reward: float, next_state, done: bool):
        # TODO: append transition to self._storage
        raise NotImplementedError

    def sample(self, batch_size: int) -> tuple:
        """Return (states, actions, rewards, next_states, dones) as numpy arrays."""
        # TODO: random.sample from self._storage; stack into arrays
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class PrioritizedReplayBuffer:
    """Proportional prioritized experience replay.

    Implemented in: Module 02, Assignment 3.
    Used in: Module 02 A3.
    """

    def __init__(self, capacity: int, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_frames: int = 100_000):
        # TODO (Module 02, A3): initialize segment tree or sorted storage
        # alpha: prioritization exponent; beta: IS correction exponent (annealed)
        raise NotImplementedError

    def push(self, state, action, reward, next_state, done, error: float):
        # TODO: store transition with priority = (|error| + eps)^alpha
        raise NotImplementedError

    def sample(self, batch_size: int) -> tuple:
        """Return (batch, importance_weights, indices)."""
        # TODO: sample proportional to priority; compute IS weights
        raise NotImplementedError

    def update_priorities(self, indices: list[int], errors: np.ndarray):
        # TODO: update stored priorities at given indices
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class RolloutBuffer:
    """On-policy rollout storage for PPO / A2C.

    Implemented in: Module 03, Assignment 3.
    Used in: Module 03 A3.
    """

    def __init__(self):
        self.obs: list = []
        self.actions: list = []
        self.rewards: list = []
        self.dones: list = []
        self.values: list = []
        self.log_probs: list = []

    def add(self, obs, action, reward: float, done: bool, value: float, log_prob: float):
        # TODO (Module 03, A3): append each quantity to its list
        raise NotImplementedError

    def get(self) -> dict:
        """Return dict of stacked tensors: obs, actions, rewards, dones, values, log_probs."""
        # TODO: convert lists to torch tensors and return as dict
        raise NotImplementedError

    def clear(self):
        self.__init__()
