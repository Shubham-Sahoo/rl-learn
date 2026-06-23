from __future__ import annotations
import gymnasium as gym
import numpy as np


def make_env(env_id: str, seed: int = 0, record_video: bool = False,
             video_folder: str = "videos/") -> gym.Env:
    """Create a seeded gymnasium environment with episode stats recording.

    Implemented in: Module 02, Assignment 2.
    Used in: all modules.
    """
    # TODO (Module 02, A2):
    # 1. gym.make(env_id, render_mode="rgb_array" if record_video else None)
    # 2. wrap with RecordEpisodeStatistics
    # 3. wrap with RecordVideo if record_video
    # 4. env.reset(seed=seed)
    raise NotImplementedError


class NormalizeObsWrapper(gym.ObservationWrapper):
    """Running mean/std normalization for observations.

    Implemented in: Module 03, Assignment 3 (for MuJoCo envs).
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        super().__init__(env)
        # TODO (Module 03, A3): maintain running mean and var (Welford's algorithm)
        raise NotImplementedError

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # TODO: normalize obs using running stats
        raise NotImplementedError
