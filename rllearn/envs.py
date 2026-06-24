from __future__ import annotations
import gymnasium as gym
import numpy as np


def make_env(env_id: str, seed: int = 0, record_video: bool = False,
             video_folder: str = "videos/") -> gym.Env:
    """Create a seeded gymnasium environment with episode stats recording.

    Implemented in: Module 02, Assignment 2.
    Used in: all modules.
    """
    env = gym.make(env_id, render_mode="rgb_array" if record_video else None)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if record_video:
        env = gym.wrappers.RecordVideo(env, video_folder)
    env.reset(seed=seed)
    return env


class NormalizeObsWrapper(gym.ObservationWrapper):
    """Running mean/std normalization for observations.

    Implemented in: Module 03, Assignment 3 (for MuJoCo envs).
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        obs_shape = env.observation_space.shape
        self.obs_mean = np.zeros(obs_shape, dtype=np.float64)
        self.obs_var = np.ones(obs_shape, dtype=np.float64)
        self.count = 0

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # Update running stats (Welford's algorithm)
        self.count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.count
        delta2 = obs - self.obs_mean
        self.obs_var += (delta * delta2 - self.obs_var) / self.count
        # Normalize
        return (obs - self.obs_mean) / (np.sqrt(self.obs_var) + self.epsilon)
