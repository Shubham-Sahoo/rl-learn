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
        self.capacity = capacity
        self._storage: deque = deque(maxlen=capacity)

    def push(self, state, action: int, reward: float, next_state, done: bool):
        self._storage.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        """Return (states, actions, rewards, next_states, dones) as numpy arrays."""
        batch = random.sample(self._storage, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self) -> int:
        return len(self._storage)


class PrioritizedReplayBuffer:
    """Proportional prioritized experience replay.

    Implemented in: Module 02, Assignment 3.
    Used in: Module 02 A3.
    """

    def __init__(self, capacity: int, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_frames: int = 100_000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self._storage: list = []
        self._priorities = np.zeros(capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0
        self._frame = 0

    def push(self, state, action, reward, next_state, done, error: float):
        max_prio = self._priorities[:self._size].max() if self._size > 0 else 1.0
        if self._size < self.capacity:
            self._storage.append((state, action, reward, next_state, done))
            self._size += 1
        else:
            self._storage[self._pos] = (state, action, reward, next_state, done)
        prio = (abs(error) + 1e-6) ** self.alpha
        self._priorities[self._pos] = max(prio, max_prio)
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple:
        """Return (batch, importance_weights, indices)."""
        self._frame += 1
        beta = min(1.0, self.beta_start + self._frame * (1.0 - self.beta_start) / self.beta_frames)

        probs = self._priorities[:self._size] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self._size, batch_size, p=probs, replace=False)
        samples = [self._storage[i] for i in indices]

        weights = (self._size * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        batch = (np.array(states, dtype=np.float32),
                 np.array(actions, dtype=np.int64),
                 np.array(rewards, dtype=np.float32),
                 np.array(next_states, dtype=np.float32),
                 np.array(dones, dtype=np.float32))
        return batch, np.array(weights, dtype=np.float32), indices

    def update_priorities(self, indices: list[int], errors: np.ndarray):
        for idx, err in zip(indices, errors):
            self._priorities[idx] = (abs(err) + 1e-6) ** self.alpha

    def __len__(self) -> int:
        return self._size


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
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def get(self) -> dict:
        """Return dict of stacked tensors: obs, actions, rewards, dones, values, log_probs."""
        import torch
        return {
            "obs": torch.FloatTensor(np.array(self.obs)),
            "actions": torch.LongTensor(self.actions),
            "rewards": torch.FloatTensor(self.rewards),
            "dones": torch.FloatTensor(self.dones),
            "values": torch.FloatTensor(self.values),
            "log_probs": torch.FloatTensor(self.log_probs),
        }

    def clear(self):
        self.__init__()
