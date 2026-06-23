# RL Curriculum Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 7-module deep RL curriculum repository with self-contained lecture notes (full LaTeX derivations), skeleton assignment notebooks, and a pre-scaffolded `rllearn/` internal package.

**Architecture:** Pre-scaffolded `rllearn/` package with `NotImplementedError` stubs that notebooks import from day one. Lecture notes are standalone Markdown chapters. Notebooks use Jupytext `.py` format (converted to `.ipynb`). TensorBoard for logging. Solutions live on a separate `solutions` branch.

**Tech Stack:** Python 3.11, uv, PyTorch ≥ 2.3, Gymnasium ≥ 0.29, TensorBoard, Jupytext, nbformat, trl, transformers

## Global Constraints

- Python >= 3.11; package manager: uv
- All equations in `$$...$$` display LaTeX (GitHub-renderable)
- Notebooks created as Jupytext `.py` files with `# %% [markdown]` / `# %%` markers, then converted to `.ipynb` via `jupytext --to notebook`
- Every implementation cell ends with `raise NotImplementedError` (removed only in solutions branch)
- Training loops are provided (not TODOs) — only algorithm internals are TODOs
- Success thresholds stated explicitly in each verification cell
- TensorBoard: `make_writer` from `rllearn.logging`; `%tensorboard --logdir runs/` in each training notebook
- `rllearn/` stub docstrings include "Implemented in: Module XX, Assignment Y" and "Used in: ..."
- Solutions branch: `solutions` — not linked from README

---

### Task 0: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `README.md`
- Create: all module directories and `projects/`, `rllearn/`

**Interfaces:**
- Produces: installable `rllearn` package skeleton; `uv sync` works

- [ ] **Step 1: Create `.python-version`**

```
3.11
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "rl-learn"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.3",
    "gymnasium[classic-control,box2d,mujoco]>=0.29",
    "numpy>=1.26",
    "tensorboard>=2.17",
    "jupyter>=1.0",
    "matplotlib>=3.9",
    "tqdm>=4.66",
    "trl>=0.9",
    "transformers>=4.43",
    "datasets>=2.20",
    "peft>=0.12",
    "jupytext>=1.16",
    "nbformat>=5.9",
]

[tool.uv]
dev-dependencies = ["pytest>=8.0", "ipykernel>=6.29"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["rllearn"]
```

- [ ] **Step 3: Create directory skeleton**

```bash
mkdir -p rllearn
mkdir -p module_01_foundations module_02_value_based module_03_policy_based
mkdir -p module_04_model_based module_05_advanced_policy
mkdir -p module_06_rl_for_llms module_07_alignment_frontier
mkdir -p projects
```

- [ ] **Step 4: Write `README.md`**

Copy the full curriculum markdown (the CS231n-style syllabus from the project brief) into `README.md`. This is the reference document subagents use for module content.

- [ ] **Step 5: Verify**

```bash
uv sync
uv run python -c "print('env ok')"
```
Expected: no errors, `.venv` created.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .python-version README.md
git commit -m "feat: project scaffolding — uv env, directory structure"
```

---

### Task 1: rllearn/ Package Stubs

**Files:**
- Create: `rllearn/__init__.py`
- Create: `rllearn/buffers.py`
- Create: `rllearn/networks.py`
- Create: `rllearn/envs.py`
- Create: `rllearn/utils.py`
- Create: `rllearn/logging.py`

**Interfaces:**
- Produces: importable stubs for all downstream notebooks
- All classes/functions raise `NotImplementedError` until implemented in the corresponding assignment

- [ ] **Step 1: Create `rllearn/__init__.py`**

```python
from rllearn import buffers, networks, envs, utils, logging
```

- [ ] **Step 2: Create `rllearn/buffers.py`**

```python
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
```

- [ ] **Step 3: Create `rllearn/networks.py`**

```python
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
```

- [ ] **Step 4: Create `rllearn/envs.py`**

```python
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
```

- [ ] **Step 5: Create `rllearn/utils.py`**

```python
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
    # TODO (Module 03, A1): work backwards; G_T = r_T, G_t = r_t + gamma * G_{t+1}
    raise NotImplementedError


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
    # TODO (Module 03, A2): iterate backwards; accumulate advantage
    # Common mistake: forgetting to zero-out next advantage at episode boundaries (dones)
    raise NotImplementedError


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of variance in y_true explained by y_pred.

    EV = 1 - Var(y_true - y_pred) / Var(y_true)
    EV=1 means perfect prediction; EV<0 means worse than mean prediction.
    """
    var_y = np.var(y_true)
    return float("nan") if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
```

- [ ] **Step 6: Create `rllearn/logging.py`**

```python
from __future__ import annotations
import time
from torch.utils.tensorboard import SummaryWriter


def make_writer(run_name: str, log_dir: str = "runs") -> SummaryWriter:
    """Create a TensorBoard SummaryWriter with timestamped run directory.

    Usage in notebooks:
        %load_ext tensorboard
        %tensorboard --logdir runs/
        writer = make_writer("dqn_cartpole")
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    full_path = f"{log_dir}/{run_name}_{timestamp}"
    return SummaryWriter(log_dir=full_path)
```

- [ ] **Step 7: Verify package imports**

```bash
uv run python -c "
from rllearn.buffers import ReplayBuffer, PrioritizedReplayBuffer, RolloutBuffer
from rllearn.networks import MLP, DuelingNet, ActorCritic, GaussianPolicyHead, TwinQNetwork
from rllearn.envs import make_env
from rllearn.utils import set_seed, compute_returns, compute_gae, explained_variance
from rllearn.logging import make_writer
print('all imports ok')
"
```
Expected: `all imports ok` (raises `NotImplementedError` only when called, not on import).

- [ ] **Step 8: Commit**

```bash
git add rllearn/
git commit -m "feat: rllearn/ package — pre-scaffolded stubs for all modules"
```

---

### Task 2: Module 01 — Foundations

**Files:**
- Create: `module_01_foundations/lecture_notes.md`
- Create: `module_01_foundations/assignment1_mdp_gridworld.py` → convert to `.ipynb`
- Create: `module_01_foundations/assignment2_dynamic_programming.py` → convert to `.ipynb`

**Interfaces:**
- Consumes: nothing (no rllearn imports — pure numpy)
- Produces: student intuition for MDPs, Bellman equations, DP

- [ ] **Step 1: Write `lecture_notes.md`**

Write a self-contained chapter covering ALL of the following, with every equation in `$$...$$` display LaTeX and a "(Why? Because...)" annotation for every non-obvious derivation step:

**Sections required:**
1. Intuition — credit assignment problem; RL vs supervised learning; Markov property intuition
2. Formal Setup — MDP tuple $(S, A, P, R, \gamma)$; define each component explicitly
3. Value Functions — derive $V^\pi(s)$ and $Q^\pi(s,a)$; show the relationship:
   $$V^\pi(s) = \sum_a \pi(a|s)\, Q^\pi(s,a)$$
4. Bellman Expectation Equations — full derivation:
   $$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma V^\pi(s')\bigr]$$
   $$Q^\pi(s,a) = \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s')\, Q^\pi(s',a')\bigr]$$
5. Bellman Optimality — derive $V^*$ and $Q^*$:
   $$V^*(s) = \max_a \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma V^*(s')\bigr]$$
   $$Q^*(s,a) = \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')\bigr]$$
6. Policy Iteration — pseudocode; convergence argument
7. Value Iteration — pseudocode; relate to Bellman optimality operator as a contraction
8. Failure Modes — effect of $\gamma$ on agent horizon; when DP is intractable (large state space)
9. Research Bridge — PRMs assign per-step rewards (Q(s,a) over reasoning steps); RLHF PPO uses $V^\pi(s)$ as critic baseline; $\gamma=1$ in RLHF and why
10. Appendix: Proof that Bellman optimality operator is a $\gamma$-contraction in $\ell^\infty$

- [ ] **Step 2: Write `assignment1_mdp_gridworld.py`**

Create a Jupytext Python file. It must contain these cells in order:

```python
# %% [markdown]
# # Assignment 1: GridWorld MDP from Scratch
# **Prerequisites:** Read `lecture_notes.md` §1–5 before starting.
# **Learning objectives:**
# - Implement a stochastic GridWorld as a formal MDP
# - Perform policy evaluation by solving Bellman equations iteratively
# - Visualize value functions as heatmaps
# - Observe how discount factor γ changes agent behavior

# %% [markdown]
# ## Part 1: Theory Recap
#
# The Bellman expectation equation for $V^\pi$:
# $$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma V^\pi(s')\bigr]$$
#
# Policy evaluation iterates this until $\max_s |V_{k+1}(s) - V_k(s)| < \theta$.

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional

# %% [markdown]
# ## Part 2: Implementation

# %%
class GridWorld:
    """
    Configurable GridWorld MDP.

    Grid symbols: ' ' = empty, 'W' = wall, 'G' = goal (+1), 'H' = hole (-1), 'S' = start.
    Actions: 0=up, 1=right, 2=down, 3=left.
    Stochastic: with prob (1-slip), take intended action; slip prob split to perpendiculars.
    """
    ACTIONS = {0: (-1,0), 1: (0,1), 2: (1,0), 3: (0,-1)}
    ACTION_NAMES = {0:'↑', 1:'→', 2:'↓', 3:'←'}

    def __init__(self, grid: List[List[str]], gamma: float = 0.99, slip: float = 0.0):
        self.grid = grid
        self.gamma = gamma
        self.slip = slip
        self.n_rows = len(grid)
        self.n_cols = len(grid[0])
        self.states = [(r,c) for r in range(self.n_rows) for c in range(self.n_cols)
                       if grid[r][c] != 'W']
        self.terminal = {(r,c) for r in range(self.n_rows) for c in range(self.n_cols)
                         if grid[r][c] in ('G','H')}

    def _reward(self, state: Tuple) -> float:
        r, c = state
        sym = self.grid[r][c]
        return 1.0 if sym == 'G' else (-1.0 if sym == 'H' else 0.0)

    def _next_state(self, state: Tuple, action: int) -> Tuple:
        dr, dc = self.ACTIONS[action]
        r2, c2 = state[0]+dr, state[1]+dc
        if 0 <= r2 < self.n_rows and 0 <= c2 < self.n_cols and self.grid[r2][c2] != 'W':
            return (r2, c2)
        return state

    def get_transitions(self, state: Tuple, action: int) -> List[Tuple[float, Tuple, float, bool]]:
        """
        Return list of (probability, next_state, reward, done).

        TODO: handle stochastic transitions (slip prob).
        With slip=0 this is deterministic.

        With slip > 0:
          - prob (1-slip): intended action
          - prob slip/2: perpendicular left
          - prob slip/2: perpendicular right
        """
        # TODO: compute perpendicular actions = [(action-1)%4, (action+1)%4]
        # TODO: build transitions list; merge probabilities for identical next_states
        # TODO: for terminal states, return [(1.0, state, 0.0, True)]
        raise NotImplementedError

    def policy_evaluation(self, policy: Dict[Tuple, int], theta: float = 1e-6) -> Dict[Tuple, float]:
        """
        Iterative policy evaluation. Solve:
          V(s) = sum_{s'} P(s'|s,pi(s)) [R(s,pi(s),s') + gamma * V(s')]

        Returns V: dict mapping state -> value.

        TODO: initialize V=0 everywhere; iterate Bellman until max|V_new - V_old| < theta.
        Common mistake: not zeroing V at terminal states after each update.
        """
        raise NotImplementedError

    def visualize_values(self, V: Dict[Tuple, float],
                         policy: Optional[Dict[Tuple, int]] = None,
                         title: str = "Value Function"):
        """Plot value function as heatmap; overlay policy arrows if provided."""
        # TODO: create numpy array of V values; plt.imshow with RdYlGn cmap
        # TODO: annotate each cell with its value; overlay ACTION_NAMES arrows for policy
        # TODO: mark walls as black, terminal states with G/H labels
        raise NotImplementedError


# %% [markdown]
# ## Part 3: Training & Verification

# %%
# Standard 4x4 grid (similar to FrozenLake)
GRID = [
    ['S', ' ', ' ', ' '],
    [' ', 'W', ' ', 'H'],
    [' ', ' ', ' ', 'H'],
    ['H', ' ', ' ', 'G'],
]

env = GridWorld(GRID, gamma=0.99, slip=0.1)
uniform_policy = {s: np.random.randint(4) for s in env.states}

V = env.policy_evaluation(uniform_policy)
env.visualize_values(V, uniform_policy, "Uniform Policy — Values")

# Verification: terminal states should have value 0.0
for s in env.terminal:
    assert abs(V.get(s, 0.0)) < 1e-3, f"Terminal state {s} should have V≈0, got {V.get(s)}"
print("✓ Terminal state values correct")

# %% [markdown]
# ## Part 4: Ablations
#
# **Ablation 1:** Change `gamma` from 0.99 → 0.1. Rerun policy_evaluation.
# What happens to the values of states far from the goal?

# %%
# Ablation 1: myopic agent
# TODO: create GridWorld with gamma=0.1, evaluate same policy, visualize, compare

# %% [markdown]
# **Observation (fill in):** With gamma=0.1, states far from the goal have values close to ___
# because ___.

# %%
# Ablation 2: slip=0.3 (very stochastic environment)
# TODO: evaluate same policy with slip=0.3; note how uncertainty degrades values near holes

# %% [markdown]
# ## Part 5: Reflection
#
# 1. Why is the Markov property essential for policy evaluation to work?
# 2. RLHF reward models score full responses (not per-token). How does this relate
#    to the credit assignment problem you just saw in GridWorld?
# 3. If gamma=1 and the environment has no terminal state, what goes wrong in policy_evaluation?

# %%
# Your answers here (markdown cell below)

# %% [markdown]
# **Answers:**
# 1.
# 2.
# 3.
```

- [ ] **Step 3: Write `assignment2_dynamic_programming.py`**

Must contain these skeleton functions (all raise NotImplementedError):

```python
# Key functions with these exact signatures:

def policy_improvement(env: GridWorld, V: Dict[Tuple, float]) -> Dict[Tuple, int]:
    """
    Greedy policy improvement: pi'(s) = argmax_a sum_{s'} P(s'|s,a)[R + gamma*V(s')]
    Returns new policy dict.
    Common mistake: not handling terminal states (their improvement is trivial).
    """
    raise NotImplementedError


def policy_iteration(env: GridWorld, theta: float = 1e-6) -> Tuple[Dict, Dict]:
    """
    Full policy iteration loop: evaluate → improve → repeat until stable.
    Returns (optimal_V, optimal_policy).
    Track iteration count and convergence delta for the ablation.
    """
    raise NotImplementedError


def value_iteration(env: GridWorld, theta: float = 1e-6) -> Tuple[Dict, Dict]:
    """
    Value iteration: V(s) <- max_a sum_{s'} P(s'|s,a)[R + gamma*V(s')]
    Returns (optimal_V, optimal_policy).
    Extract policy at end via one greedy pass.
    """
    raise NotImplementedError
```

Training cell: run both algorithms on the same GridWorld; assert they reach identical optimal values within 1e-4. Time each; plot convergence curves (delta vs iteration). Verify against `gymnasium.envs.toy_text.frozen_lake.FrozenLakeEnv` optimal policy (6 < n_iters < 200 for 4×4).

- [ ] **Step 4: Convert to notebooks**

```bash
cd module_01_foundations
uv run jupytext --to notebook assignment1_mdp_gridworld.py
uv run jupytext --to notebook assignment2_dynamic_programming.py
```

- [ ] **Step 5: Verify**

```bash
uv run jupyter nbconvert --to script module_01_foundations/assignment1_mdp_gridworld.ipynb --stdout > /dev/null
echo "notebook valid"
```

- [ ] **Step 6: Commit**

```bash
git add module_01_foundations/
git commit -m "feat: module 01 — MDP foundations lecture notes + assignments"
```

---

### Task 3: Module 02 — Value-Based Methods

**Files:**
- Create: `module_02_value_based/lecture_notes.md`
- Create: `module_02_value_based/assignment1_tabular_qlearning.py` → `.ipynb`
- Create: `module_02_value_based/assignment2_deep_qnetwork.py` → `.ipynb`
- Create: `module_02_value_based/assignment3_rainbow_components.py` → `.ipynb`

**Interfaces:**
- Consumes: `rllearn.buffers.ReplayBuffer` (stub — student implements in A2)
- Produces: working `ReplayBuffer`, `MLP`, `make_env` implementations in `rllearn/`

**Note:** Assignments 2 and 3 instruct the student to implement `ReplayBuffer` in `rllearn/buffers.py`, `MLP` in `rllearn/networks.py`, and `make_env` in `rllearn/envs.py` before completing the notebook TODOs.

- [ ] **Step 1: Write `lecture_notes.md`**

Self-contained chapter. Required sections + equations:

1. Intuition — TD learning as "driving estimate update"; MC vs TD tradeoff
2. MC vs TD — bias-variance table; n-step TD as interpolation
3. Q-Learning:
   $$Q(s,a) \leftarrow Q(s,a) + \alpha\bigl[R + \gamma \max_{a'} Q(s',a') - Q(s,a)\bigr]$$
   Explain why this is off-policy (target uses $\max$, not behavior policy).
4. SARSA (on-policy TD):
   $$Q(s,a) \leftarrow Q(s,a) + \alpha\bigl[R + \gamma Q(s',a') - Q(s,a)\bigr]$$
5. DQN — experience replay rationale; target network rationale:
   $$\mathcal{L}(\theta) = \mathbb{E}\bigl[(R + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta))^2\bigr]$$
6. Double DQN — overestimation bias proof sketch; fix:
   $$y = R + \gamma\, Q\!\left(s',\, \arg\max_{a'} Q(s',a';\theta);\; \theta^-\right)$$
7. Dueling DQN:
   $$Q(s,a;\theta) = V(s;\theta_V) + A(s,a;\theta_A) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a';\theta_A)$$
8. Prioritized Experience Replay:
   $$p_i = |\delta_i| + \varepsilon, \quad P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \quad w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$$
9. Failure modes — divergence without replay/target network (show loss exploding); exploration inadequacy
10. Research Bridge — reward hacking ↔ Q-overestimation; offline RL (CQL) avoids overestimation for LLM fine-tuning

- [ ] **Step 2: Write `assignment1_tabular_qlearning.py`**

Skeleton functions:

```python
class TabularQAgent:
    def __init__(self, n_states: int, n_actions: int, alpha: float,
                 gamma: float, epsilon: float, epsilon_min: float, epsilon_decay: float):
        # TODO: initialize Q-table as numpy zeros (n_states, n_actions)
        raise NotImplementedError

    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        # TODO: with prob epsilon random; else argmax Q[state]
        raise NotImplementedError

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> float:
        """Q-learning update. Returns TD error."""
        # TODO: td_target = reward + gamma * max(Q[next_state]) * (1 - done)
        # TODO: td_error = td_target - Q[state, action]
        # TODO: Q[state, action] += alpha * td_error
        raise NotImplementedError

    def decay_epsilon(self):
        # TODO: epsilon = max(epsilon_min, epsilon * epsilon_decay)
        raise NotImplementedError


def sarsa_update(Q: np.ndarray, state: int, action: int, reward: float,
                 next_state: int, next_action: int,
                 alpha: float, gamma: float, done: bool) -> float:
    """On-policy SARSA update. Returns TD error."""
    # TODO: td_target = reward + gamma * Q[next_state, next_action] * (1 - done)
    raise NotImplementedError
```

Verify: Q-learning solves Taxi-v3 (mean episode reward > 7 over last 100 episodes within 10k episodes).
Ablation: compare Q-learning vs SARSA near a cliff (CliffWalking-v0). Plot episode reward vs episode for both.

- [ ] **Step 3: Write `assignment2_deep_qnetwork.py`**

Preamble instructs: "Before running Part 2, implement `ReplayBuffer` in `rllearn/buffers.py` and `MLP` in `rllearn/networks.py` and `make_env` in `rllearn/envs.py`."

```python
class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, lr: float = 1e-3,
                 gamma: float = 0.99, buffer_capacity: int = 10_000,
                 batch_size: int = 64, target_update_freq: int = 100):
        # Provided (not a TODO) — uses rllearn stubs:
        from rllearn.buffers import ReplayBuffer
        from rllearn.networks import MLP
        self.online_net = MLP(obs_dim, n_actions)
        self.target_net = MLP(obs_dim, n_actions)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.buffer = ReplayBuffer(buffer_capacity)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.n_updates = 0

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy using online_net."""
        # TODO: epsilon-greedy; use torch.no_grad() for forward pass
        raise NotImplementedError

    def store_transition(self, obs, action, reward, next_obs, done):
        # Provided: self.buffer.push(obs, action, reward, next_obs, done)
        self.buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> float:
        """Sample from buffer, compute DQN loss, gradient step. Returns loss."""
        if len(self.buffer) < self.batch_size:
            return 0.0
        # TODO: sample batch from self.buffer
        # TODO: current_q = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # TODO: with torch.no_grad(): next_q = target_net(next_states).max(1).values
        # TODO: target_q = rewards + gamma * next_q * (1 - dones)
        # TODO: loss = F.mse_loss(current_q, target_q)
        # TODO: optimizer step; increment n_updates
        # TODO: if n_updates % target_update_freq == 0: sync target network
        raise NotImplementedError

    def sync_target(self):
        # TODO: hard copy online → target
        raise NotImplementedError
```

Training loop: provided. Logs `train/episode_reward`, `train/td_loss`, `train/q_value_mean`, `train/epsilon` via TensorBoard.
Verify: CartPole-v1 mean reward ≥ 450 in last 50 episodes within 300 episodes.
Ablation cells for: (a) no replay buffer (batch_size=1, no storage), (b) no target network (target_update_freq=1).

- [ ] **Step 4: Write `assignment3_rainbow_components.py`**

```python
def double_dqn_target(online_net, target_net, next_states: torch.Tensor,
                      rewards: torch.Tensor, dones: torch.Tensor,
                      gamma: float) -> torch.Tensor:
    """
    Double DQN target: y = r + gamma * Q(s', argmax_{a'} Q(s',a';theta); theta-)
    Fixes overestimation bias from vanilla DQN.
    Common mistake: using target_net for BOTH action selection and evaluation.
    """
    # TODO: action_selection = online_net(next_states).argmax(1)   # online selects
    # TODO: next_q = target_net(next_states).gather(1, action_selection.unsqueeze(1)).squeeze(1)
    # TODO: return rewards + gamma * next_q * (1 - dones)
    raise NotImplementedError
```

Preamble instructs: implement `DuelingNet` in `rllearn/networks.py` and `PrioritizedReplayBuffer` in `rllearn/buffers.py`.

Verify: LunarLander-v2 mean reward ≥ 200 in last 100 episodes. Compare convergence speed of vanilla DQN vs Double vs Dueling (3 TensorBoard runs on same plot).

- [ ] **Step 5: Convert + verify notebooks, commit**

```bash
cd module_02_value_based
uv run jupytext --to notebook assignment1_tabular_qlearning.py
uv run jupytext --to notebook assignment2_deep_qnetwork.py
uv run jupytext --to notebook assignment3_rainbow_components.py
cd ..
git add module_02_value_based/
git commit -m "feat: module 02 — value-based methods lecture notes + DQN assignments"
```

---

### Task 4: Module 03 — Policy-Based Methods & Actor-Critic

**Files:**
- Create: `module_03_policy_based/lecture_notes.md`
- Create: `module_03_policy_based/assignment1_reinforce.py` → `.ipynb`
- Create: `module_03_policy_based/assignment2_actor_critic.py` → `.ipynb`
- Create: `module_03_policy_based/assignment3_ppo_from_scratch.py` → `.ipynb`

**Interfaces:**
- Consumes: `rllearn.networks.ActorCritic` (stub); `rllearn.utils.compute_returns`, `compute_gae` (stubs)
- Produces: working `compute_returns`, `compute_gae`, `ActorCritic`, `RolloutBuffer`, `NormalizeObsWrapper`

- [ ] **Step 1: Write `lecture_notes.md`**

Required sections + equations:

1. Intuition — why value-based fails for continuous actions; why stochastic policies matter
2. Policy Gradient Theorem — full derivation using log-derivative trick:
   $$\nabla_\theta J(\theta) = \mathbb{E}_\pi\!\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q^\pi(s_t,a_t)\right]$$
   Derive the log-derivative trick: $\nabla_\theta p_\theta(x) = p_\theta(x)\nabla_\theta \log p_\theta(x)$
3. REINFORCE — Monte Carlo estimate; high variance explanation:
   $$\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$$
4. Baseline trick — prove baseline doesn't bias gradient; advantage function:
   $$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$
5. TD Advantage — one-step:
   $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
6. GAE — full derivation as exponentially weighted n-step advantages:
   $$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$$
   Show $\lambda=0$ recovers TD(0); $\lambda=1$ recovers Monte Carlo.
7. PPO-Clip — trust region motivation; probability ratio; clip objective:
   $$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$$
   $$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\bigl(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\bigr)\right]$$
   Full loss: $\mathcal{L} = \mathcal{L}^{CLIP} - c_1 \mathcal{L}^{VF} + c_2 S[\pi_\theta]$
8. Failure modes — no entropy bonus → premature convergence; no clipping → catastrophic collapse
9. Research Bridge — InstructGPT PPO pipeline; RLHF KL penalty as soft trust region; entropy bonus ↔ LLM temperature

- [ ] **Step 2: Write `assignment1_reinforce.py`**

```python
class PolicyNet(nn.Module):
    """Softmax policy network for discrete actions."""
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        # TODO: MLP with ReLU; final layer outputs raw logits (no softmax)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return action logits of shape (batch, n_actions)."""
        raise NotImplementedError


def select_action(policy_net: PolicyNet, obs: np.ndarray) -> Tuple[int, torch.Tensor]:
    """Sample action from policy; return (action, log_prob)."""
    # TODO: obs → tensor; forward; Categorical(logits=...); sample; log_prob
    raise NotImplementedError


def reinforce_update(optimizer, log_probs: List[torch.Tensor],
                     returns: List[float]) -> float:
    """
    REINFORCE gradient update.
    Loss = -sum_t log_pi(a_t|s_t) * G_t   (negative for gradient ascent)
    Returns scalar loss value.
    """
    # TODO: normalize returns (subtract mean, divide by std + 1e-8)
    # TODO: loss = -sum of log_prob * return for each timestep
    # TODO: optimizer.zero_grad(); loss.backward(); optimizer.step()
    raise NotImplementedError
```

Instruct student to implement `compute_returns` in `rllearn/utils.py` first.
Verify: CartPole-v1 ≥ 450 mean reward in last 50 episodes within 500 episodes.
Ablation: REINFORCE with vs without return normalization — plot gradient magnitude std over training.

- [ ] **Step 3: Write `assignment2_actor_critic.py`**

Instruct student to implement `ActorCritic` in `rllearn/networks.py` and `compute_gae` in `rllearn/utils.py` first.

```python
def actor_critic_update(ac_net: ActorCritic, optimizer,
                        obs: torch.Tensor, action: int,
                        reward: float, next_obs: torch.Tensor,
                        done: bool, gamma: float) -> Tuple[float, float]:
    """
    One-step A2C update.
    TD advantage: delta = r + gamma * V(s') * (1-done) - V(s)
    Actor loss: -log_pi(a|s) * delta.detach()
    Critic loss: delta^2
    Returns (actor_loss, critic_loss).
    Common mistake: using delta (with gradient) for actor loss — must detach.
    """
    raise NotImplementedError
```

GAE ablation: train on LunarLander-v2 with λ ∈ {0.0, 0.5, 0.95, 1.0}; plot episode reward vs step for all 4.
Verify: LunarLander-v2 mean reward ≥ 150 within 1500 episodes with λ=0.95.

- [ ] **Step 4: Write `assignment3_ppo_from_scratch.py`**

Instruct: implement `RolloutBuffer` in `rllearn/buffers.py` and `NormalizeObsWrapper` in `rllearn/envs.py` first.

```python
def compute_ppo_loss(log_probs_new: torch.Tensor, log_probs_old: torch.Tensor,
                     advantages: torch.Tensor, clip_eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PPO-Clip policy loss.
    r_t = exp(log_pi_new - log_pi_old)
    L_CLIP = mean(min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t))
    Returns (loss, clip_fraction) — clip_fraction = fraction of ratios outside [1-eps, 1+eps].
    """
    raise NotImplementedError


def ppo_update(actor_critic: ActorCritic, optimizer, rollout_buffer: RolloutBuffer,
               clip_eps: float = 0.2, c1: float = 0.5, c2: float = 0.01,
               n_epochs: int = 10, batch_size: int = 64) -> dict:
    """
    Full PPO update over n_epochs of minibatch gradient steps.
    Returns dict with keys: policy_loss, value_loss, entropy_loss, clip_fraction, explained_variance.
    """
    # TODO: get data from rollout_buffer; compute advantages (already GAE-computed in buffer)
    # TODO: normalize advantages: (A - mean(A)) / (std(A) + 1e-8)
    # TODO: for each epoch, shuffle and iterate minibatches:
    #   - recompute log_probs_new and values_new from actor_critic
    #   - compute_ppo_loss for policy
    #   - value loss: 0.5 * mean((values_new - returns)^2)
    #   - entropy: mean(Categorical(logits=logits).entropy())
    #   - total_loss = policy_loss - c2 * entropy + c1 * value_loss
    raise NotImplementedError
```

Training loop: collect 2048 steps per iteration (provided); call `ppo_update`; log all metrics.
Target: HalfCheetah-v4 mean reward ≥ 3000 within 3M steps (or Ant-v4 ≥ 2500).
Ablation cells: (a) no clipping (clip_eps=∞), (b) no entropy bonus (c2=0), (c) 1 epoch vs 10 epochs.

- [ ] **Step 5: Convert, verify, commit**

```bash
cd module_03_policy_based
uv run jupytext --to notebook assignment1_reinforce.py
uv run jupytext --to notebook assignment2_actor_critic.py
uv run jupytext --to notebook assignment3_ppo_from_scratch.py
cd ..
git add module_03_policy_based/
git commit -m "feat: module 03 — policy gradient + PPO lecture notes + assignments"
```

---

### Task 5: Module 04 — Model-Based RL

**Files:**
- Create: `module_04_model_based/lecture_notes.md`
- Create: `module_04_model_based/assignment1_dyna_q.py` → `.ipynb`
- Create: `module_04_model_based/assignment2_world_models.py` → `.ipynb`

- [ ] **Step 1: Write `lecture_notes.md`**

Required sections + equations:

1. Intuition — sample efficiency tradeoff; model error compounding
2. Dyna-Q — pseudocode with n planning steps; tabular model update
3. World Models — transition model $f_\phi$; reward model $g_\phi$; latent space planning:
   $$\mathcal{L}_{transition} = \|f_\phi(s_t, a_t) - s_{t+1}\|^2$$
4. RSSM — recurrent state space model; stochastic + deterministic latent; ELBO objective (high level)
5. Failure modes — model exploitation; compounding errors; when to trust model
6. Research Bridge — AlphaZero MCTS; constitutional AI self-play; chain-of-thought as world model simulation

- [ ] **Step 2: Write `assignment1_dyna_q.py`**

```python
class DynaQAgent:
    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 n_planning_steps: int = 10):
        # TODO: Q-table zeros; model = {} (dict mapping (s,a) -> (r, s'))
        raise NotImplementedError

    def select_action(self, state: int) -> int:
        # TODO: epsilon-greedy
        raise NotImplementedError

    def update_q(self, state: int, action: int, reward: float, next_state: int) -> float:
        """Direct RL update from real experience. Returns TD error."""
        raise NotImplementedError

    def update_model(self, state: int, action: int, reward: float, next_state: int):
        """Store observed transition in model."""
        # TODO: self.model[(state, action)] = (reward, next_state)
        raise NotImplementedError

    def plan(self):
        """Run n_planning_steps synthetic Q-updates from model."""
        # TODO: for _ in range(n_planning_steps):
        #   sample random (s,a) from self.model.keys()
        #   r, s' = self.model[(s,a)]
        #   Q-update using r, s'
        raise NotImplementedError

    def step(self, state: int, action: int, reward: float,
             next_state: int, done: bool):
        """Full Dyna-Q step: update_q + update_model + plan."""
        raise NotImplementedError
```

Verify: GridWorld from Module 01 (4×4). Compare cumulative reward vs real_steps for n_planning ∈ {0, 5, 20, 50}. With n=50, should achieve optimal policy in ≤ 50 real steps.

- [ ] **Step 3: Write `assignment2_world_models.py`**

```python
class TransitionModel(nn.Module):
    """Predict next observation from (obs, action)."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # TODO: MLP(obs_dim + action_dim, obs_dim)
        raise NotImplementedError

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return predicted next_obs of same shape as obs."""
        raise NotImplementedError


class RewardModel(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        raise NotImplementedError

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def train_world_model(transition_model: TransitionModel, reward_model: RewardModel,
                      optimizer, replay_buffer, batch_size: int = 256) -> Tuple[float, float]:
    """Train one step. Returns (transition_loss, reward_loss)."""
    raise NotImplementedError


def generate_model_rollout(transition_model, reward_model, start_obs: torch.Tensor,
                           policy, horizon: int = 5) -> List[Tuple]:
    """Generate synthetic (obs, action, reward, next_obs) transitions."""
    raise NotImplementedError
```

Verify: one-step prediction MSE on CartPole < 0.01 after 1000 gradient steps on 5k offline transitions.

- [ ] **Step 4: Convert, verify, commit**

```bash
cd module_04_model_based
uv run jupytext --to notebook assignment1_dyna_q.py
uv run jupytext --to notebook assignment2_world_models.py
cd ..
git add module_04_model_based/
git commit -m "feat: module 04 — model-based RL lecture notes + Dyna-Q/world model assignments"
```

---

### Task 6: Module 05 — Advanced Policy Optimization

**Files:**
- Create: `module_05_advanced_policy/lecture_notes.md`
- Create: `module_05_advanced_policy/assignment1_trpo_natural_gradient.py` → `.ipynb`
- Create: `module_05_advanced_policy/assignment2_soft_actor_critic.py` → `.ipynb`
- Create: `module_05_advanced_policy/assignment3_multi_agent_intro.py` → `.ipynb`

**Interfaces:**
- Produces: working `GaussianPolicyHead`, `TwinQNetwork`, `PrioritizedReplayBuffer`

- [ ] **Step 1: Write `lecture_notes.md`**

Required sections + equations:

1. Intuition — parameter space vs policy space; why gradient descent is misleading
2. Fisher Information Matrix:
   $$F_{ij}(\theta) = \mathbb{E}_\pi\!\left[\frac{\partial \log\pi_\theta}{\partial\theta_i}\frac{\partial\log\pi_\theta}{\partial\theta_j}\right]$$
3. Natural Gradient: $\tilde{\nabla}J = F^{-1}\nabla J$; KL-geometry interpretation
4. TRPO — surrogate objective; KL constraint:
   $$\max_\theta \hat{\mathbb{E}}_t\!\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}\hat{A}_t\right] \quad \text{s.t.}\quad \hat{\mathbb{E}}_t[\text{KL}(\pi_{\theta_\text{old}}\|\pi_\theta)] \leq \delta$$
   Conjugate gradient method for $F^{-1}g$ without materializing $F$.
5. Maximum Entropy RL:
   $$J(\pi) = \mathbb{E}_\tau\!\left[\sum_t\gamma^t\bigl(R(s_t,a_t) + \alpha H(\pi(\cdot|s_t))\bigr)\right]$$
6. SAC — twin Q-networks; reparameterization trick:
   $$a = \tanh(\mu_\phi(s) + \sigma_\phi(s)\odot\varepsilon),\quad\varepsilon\sim\mathcal{N}(0,I)$$
   Log-prob with squashing correction:
   $$\log\pi(a|s) = \log\mathcal{N}(\tilde{a};\mu,\sigma) - \sum_i \log(1-a_i^2+\varepsilon)$$
   Q-target: $y = r + \gamma(\min_{i=1,2}Q_{\theta_i^-}(s',\tilde{a}') - \alpha\log\pi(\tilde{a}'|s'))$
   Automatic temperature: $\mathcal{L}(\alpha) = \mathbb{E}[-\alpha\log\pi(a|s) - \alpha\bar{H}]$
7. MARL — cooperative vs competitive; partial observability; centralized training / decentralized execution
8. Research Bridge — temperature α ↔ LLM sampling temperature; SAC twin critics ↔ reward model ensemble

- [ ] **Step 2: Write `assignment1_trpo_natural_gradient.py`**

```python
def compute_kl_divergence(policy_net: nn.Module, obs_batch: torch.Tensor,
                          old_log_probs: torch.Tensor) -> torch.Tensor:
    """KL(pi_old || pi_new) estimated from obs_batch."""
    raise NotImplementedError


def conjugate_gradient(A_fn, b: torch.Tensor, n_steps: int = 10,
                       damping: float = 1e-2) -> torch.Tensor:
    """
    Solve Ax = b via conjugate gradient without materializing A.
    A_fn: callable that computes A @ v (Fisher-vector product).
    Used to compute F^{-1} * gradient.
    """
    raise NotImplementedError


def fisher_vector_product(policy_net: nn.Module, obs_batch: torch.Tensor,
                          vector: torch.Tensor, damping: float = 1e-2) -> torch.Tensor:
    """Compute (F + damping*I) @ vector efficiently via double backprop."""
    raise NotImplementedError


def natural_gradient_step(policy_net: nn.Module, obs_batch: torch.Tensor,
                           policy_loss: torch.Tensor, max_kl: float = 0.01) -> bool:
    """
    Compute natural gradient direction and do line search.
    Returns True if update was accepted (KL constraint satisfied).
    """
    raise NotImplementedError
```

Ablation: compare vanilla gradient vs natural gradient on CartPole — plot KL divergence between updates; show natural gradient produces consistent policy-space steps.

- [ ] **Step 3: Write `assignment2_soft_actor_critic.py`**

Instruct: implement `GaussianPolicyHead` and `TwinQNetwork` in `rllearn/networks.py` first.

```python
def sac_critic_loss(q_net: TwinQNetwork, q_target: TwinQNetwork,
                    policy: GaussianPolicyHead,
                    batch: tuple, gamma: float, log_alpha: torch.Tensor) -> torch.Tensor:
    """
    Twin critic loss.
    y = r + gamma * (min(Q1', Q2')(s', a'_tilde) - alpha * log_pi(a'_tilde|s'))
    L_Q = 0.5 * mean((Q1(s,a) - y)^2 + (Q2(s,a) - y)^2)
    Common mistake: using current q_net (not q_target) for next_state value.
    """
    raise NotImplementedError


def sac_actor_loss(policy: GaussianPolicyHead, q_net: TwinQNetwork,
                   obs: torch.Tensor, log_alpha: torch.Tensor) -> torch.Tensor:
    """
    Actor loss: maximize E[min(Q1,Q2)(s, a_tilde) - alpha * log_pi(a_tilde|s)]
    Loss = -mean(min(Q1,Q2) - alpha * log_pi)
    """
    raise NotImplementedError


def sac_alpha_loss(log_alpha: torch.Tensor, log_pi: torch.Tensor,
                   target_entropy: float) -> torch.Tensor:
    """
    Automatic temperature tuning.
    L(alpha) = -alpha * (log_pi + target_entropy).detach()
    target_entropy = -action_dim (heuristic)
    """
    raise NotImplementedError
```

Verify: HalfCheetah-v4 mean reward ≥ 5000 within 1M steps.
Ablation: fixed α=0.2 vs fixed α=0.01 vs learned α — show sample efficiency difference.

- [ ] **Step 4: Write `assignment3_multi_agent_intro.py`**

```python
class CoopGridWorld:
    """Two-agent cooperative gridworld. Both must reach their goals simultaneously."""
    def __init__(self, grid_size: int = 5, n_agents: int = 2):
        raise NotImplementedError

    def reset(self) -> List[np.ndarray]:
        """Return list of per-agent observations."""
        raise NotImplementedError

    def step(self, actions: List[int]) -> Tuple[List, List[float], List[bool], dict]:
        """Return (obs_list, reward_list, done_list, info)."""
        raise NotImplementedError


class IndependentQLearner:
    def __init__(self, agent_id: int, n_states: int, n_actions: int,
                 alpha: float, gamma: float, epsilon: float):
        raise NotImplementedError

    def select_action(self, state: int) -> int:
        raise NotImplementedError

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> float:
        raise NotImplementedError
```

Verify: 2-agent CoopGridWorld solved (joint success rate ≥ 0.8) within 5000 episodes.

- [ ] **Step 5: Convert, verify, commit**

```bash
cd module_05_advanced_policy
uv run jupytext --to notebook assignment1_trpo_natural_gradient.py
uv run jupytext --to notebook assignment2_soft_actor_critic.py
uv run jupytext --to notebook assignment3_multi_agent_intro.py
cd ..
git add module_05_advanced_policy/
git commit -m "feat: module 05 — TRPO + SAC + MARL lecture notes + assignments"
```

---

### Task 7: Module 06 — RL for Language Models

**Files:**
- Create: `module_06_rl_for_llms/lecture_notes.md`
- Create: `module_06_rl_for_llms/assignment1_reward_modeling.py` → `.ipynb`
- Create: `module_06_rl_for_llms/assignment2_rlhf_pipeline.py` → `.ipynb`
- Create: `module_06_rl_for_llms/assignment3_grpo_implementation.py` → `.ipynb`

- [ ] **Step 1: Write `lecture_notes.md`**

Required sections + equations:

1. Intuition — RL/LLM correspondence table (state=context, action=token, policy=LM, episode=generation)
2. Reward Modeling — Bradley-Terry model; preference loss:
   $$\mathcal{L}_{RM} = -\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma\!\left(R_\phi(x,y_w) - R_\phi(x,y_l)\right)\right]$$
3. RLHF-PPO pipeline — diagram: SFT → RM training → PPO fine-tuning; KL penalty:
   $$R_{total}(x,y) = R_{RM}(x,y) - \beta\,\text{KL}\!\left(\pi_\theta(y|x)\,\|\,\pi_{ref}(y|x)\right)$$
4. Token-level PPO — credit assignment over tokens; value network; GAE on token sequence
5. GRPO — group sampling; group-normalized advantage:
   $$\tilde{A}_i = \frac{r_i - \mu_r}{\sigma_r + \varepsilon}$$
   Loss identical to PPO-Clip but using $\tilde{A}_i$; why no value network needed
6. DPO — closed-form optimal policy derivation; loss:
   $$\mathcal{L}_{DPO} = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$
7. DPO limitations — offline; no process rewards; length bias
8. Research Bridge — InstructGPT pipeline; DeepSeek-R1 GRPO on math; KL-β schedule

- [ ] **Step 2: Write `assignment1_reward_modeling.py`**

```python
class SentimentRewardModel(nn.Module):
    """LSTM-based reward model trained on IMDB preferences."""
    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        # TODO: nn.Embedding + nn.LSTM + linear head → scalar output
        raise NotImplementedError

    def forward(self, input_ids: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """Return scalar reward of shape (batch,)."""
        raise NotImplementedError


def bradley_terry_loss(r_winner: torch.Tensor, r_loser: torch.Tensor) -> torch.Tensor:
    """
    L = -mean(log(sigma(r_winner - r_loser)))
    r_winner, r_loser: scalar rewards for preferred/dispreferred responses.
    """
    raise NotImplementedError


def evaluate_ranking_accuracy(model, eval_loader) -> float:
    """Fraction of pairs where model correctly ranks winner above loser."""
    raise NotImplementedError
```

Dataset: IMDB reviews. Construct comparison pairs: 5-star reviews as preferred, 1-star as dispreferred.
Verify: ranking accuracy ≥ 0.80 on held-out pairs after 5 epochs.

- [ ] **Step 3: Write `assignment2_rlhf_pipeline.py`**

Use `trl.PPOTrainer` as scaffolding. The TODOs are in the reward computation.

```python
def compute_kl_penalty(log_probs_policy: torch.Tensor,
                       log_probs_ref: torch.Tensor,
                       beta: float = 0.1) -> torch.Tensor:
    """
    Per-token KL penalty: beta * (log_pi_theta - log_pi_ref)
    Summed over token dimension.
    This is the soft trust region that prevents reward hacking.
    """
    raise NotImplementedError


def compute_rlhf_reward(response_ids: torch.Tensor, reward_model,
                        policy_model, ref_model, tokenizer,
                        beta: float = 0.1) -> torch.Tensor:
    """
    R_total = R_RM(x, y) - beta * KL(pi_theta || pi_ref)
    Returns per-sample scalar reward for PPOTrainer.
    """
    raise NotImplementedError
```

Log: `train/kl_from_ref`, `train/reward_mean`, `train/reward_std` across training steps.
Verify: reward increases while KL stays < 10 nats within 200 PPO steps on GPT-2 Small.

- [ ] **Step 4: Write `assignment3_grpo_implementation.py`**

```python
def grpo_group_sample(policy, tokenizer, prompt: str,
                      group_size: int = 8, max_new_tokens: int = 64) -> Tuple[List[str], torch.Tensor]:
    """
    Sample group_size responses from policy for the same prompt.
    Returns (responses, log_probs) where log_probs has shape (group_size,).
    """
    raise NotImplementedError


def normalize_group_rewards(rewards: List[float]) -> torch.Tensor:
    """
    Normalize within group: A_i = (r_i - mean(r)) / (std(r) + 1e-8)
    Returns tensor of shape (group_size,).
    Common mistake: not adding epsilon to std — division by zero when all rewards are equal.
    """
    raise NotImplementedError


def grpo_loss(log_probs_new: torch.Tensor, log_probs_old: torch.Tensor,
              advantages: torch.Tensor, clip_eps: float = 0.2) -> torch.Tensor:
    """
    PPO-Clip loss using GRPO normalized advantages (no value network needed).
    Identical formula to PPO-Clip but advantages are group-normalized rewards.
    """
    raise NotImplementedError
```

Task: math expression evaluation (generate Python expression → verify with `eval()`; reward +1 correct, 0 wrong).
Verify: solve rate improves from baseline by ≥ 10 percentage points within 500 GRPO steps.
Ablation: compare GRPO (group_size=8) vs PPO with learned value head — wall clock time per step.

- [ ] **Step 5: Convert, verify, commit**

```bash
cd module_06_rl_for_llms
uv run jupytext --to notebook assignment1_reward_modeling.py
uv run jupytext --to notebook assignment2_rlhf_pipeline.py
uv run jupytext --to notebook assignment3_grpo_implementation.py
cd ..
git add module_06_rl_for_llms/
git commit -m "feat: module 06 — RLHF + GRPO + DPO lecture notes + assignments"
```

---

### Task 8: Module 07 — Alignment Frontier

**Files:**
- Create: `module_07_alignment_frontier/lecture_notes.md`
- Create: `module_07_alignment_frontier/assignment1_dpo_vs_rlhf.py` → `.ipynb`
- Create: `module_07_alignment_frontier/assignment2_process_reward_models.py` → `.ipynb`
- Create: `module_07_alignment_frontier/assignment3_vlm_alignment.py` → `.ipynb`

- [ ] **Step 1: Write `lecture_notes.md`**

Required sections + equations:

1. PRMs vs ORMs — sparse vs dense reward; step-level credit assignment
2. PRM scoring: $V_{PRM}(s_t) = P(\text{correct} \mid \text{reasoning prefix}_t)$
3. Monte Carlo PRM labels: estimate $P(\text{correct} \mid \text{step}_t)$ via $N$ rollout completions
4. Best-of-N with PRM: product of step scores; comparison with ORM reranking
5. VLM alignment — hallucination taxonomy; CHAIR metric:
   $$\text{CHAIR}_s = \frac{|\{s : \exists\text{ hallucinated object mention}\}|}{|\text{sentences}|}$$
6. VLM-specific RLHF failures — visual grounding requires separate supervision; sycophancy on visual claims
7. RLHF at scale — 4x memory (policy + ref + critic + RM); LoRA mitigations; async PPO
8. Research Bridge — o1/o3 PRM; Qwen-VL/InternVL alignment; scalable oversight open problems

- [ ] **Step 2: Write `assignment1_dpo_vs_rlhf.py`**

```python
def dpo_loss(policy_model, ref_model, input_ids_w: torch.Tensor,
             input_ids_l: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """
    DPO loss:
    L = -E[log sigma(beta * log(pi_theta(y_w|x)/pi_ref(y_w|x))
                   - beta * log(pi_theta(y_l|x)/pi_ref(y_l|x)))]

    Compute log-probs as sum of token log-probs over response tokens only.
    Common mistake: including prompt tokens in the log-prob sum.
    """
    raise NotImplementedError


def compute_implicit_reward(policy_model, ref_model,
                            input_ids: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Extract DPO implicit reward: beta * log(pi_theta(y|x) / pi_ref(y|x))
    This is the reward DPO implicitly optimizes — useful for analysis.
    """
    raise NotImplementedError
```

Comparison experiment: fine-tune GPT-2 Small on Anthropic HH-RLHF dataset with both DPO and PPO-RLHF.
Log for both: win rate vs reference (using reward model as judge), output length, KL from reference.
Verify: DPO training is stable (loss monotonically decreasing over 3 epochs).

- [ ] **Step 3: Write `assignment2_process_reward_models.py`**

```python
class ProcessRewardModel(nn.Module):
    """Scores individual reasoning steps."""
    def __init__(self, input_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        # TODO: MLP(input_dim, 1) with sigmoid output
        raise NotImplementedError

    def forward(self, step_embedding: torch.Tensor) -> torch.Tensor:
        """Return step score in [0,1] of shape (batch,)."""
        raise NotImplementedError


def monte_carlo_step_label(solution_steps: List[str], step_idx: int,
                           oracle_fn, n_rollouts: int = 16) -> float:
    """
    Estimate P(correct final answer | correct through step_idx) by sampling
    n_rollouts continuations from step_idx and checking oracle correctness.
    Returns float in [0, 1].
    """
    raise NotImplementedError


def best_of_n_with_prm(policy, prm, tokenizer, prompt: str,
                       n: int = 16) -> str:
    """
    Generate n solutions; score each with PRM (product of step scores);
    return highest-scoring solution.
    """
    raise NotImplementedError


def prm_beam_search(policy, prm, tokenizer, prompt: str,
                    beam_width: int = 4, max_steps: int = 8) -> str:
    """
    Beam search over reasoning steps guided by PRM step scores.
    At each step, expand all beams, score, keep top beam_width.
    """
    raise NotImplementedError
```

Dataset: GSM8K (math word problems). ORM = final answer correct/wrong. PRM labels via monte_carlo_step_label.
Verify: PRM best-of-16 improves GSM8K accuracy by ≥ 5% over ORM best-of-16.

- [ ] **Step 4: Write `assignment3_vlm_alignment.py`**

```python
def compute_chair_score(generated_captions: List[str],
                        ground_truth_objects: List[List[str]]) -> dict:
    """
    CHAIR_s = fraction of sentences with ≥1 hallucinated object.
    CHAIR_i = fraction of mentioned objects that are hallucinated.
    Returns {'CHAIR_s': float, 'CHAIR_i': float}.
    """
    raise NotImplementedError


class HallucinationRewardModel(nn.Module):
    """Binary classifier: does caption hallucinate objects not in image?"""
    def __init__(self, vision_dim: int = 768, text_dim: int = 768,
                 hidden_dim: int = 256):
        super().__init__()
        # TODO: project vision + text → concat → MLP → scalar (sigmoid)
        raise NotImplementedError

    def forward(self, vision_features: torch.Tensor,
                text_features: torch.Tensor) -> torch.Tensor:
        """Return P(faithful) in [0,1] of shape (batch,)."""
        raise NotImplementedError


def dpo_vlm_loss(policy_vlm, ref_vlm, batch: dict, beta: float = 0.1) -> torch.Tensor:
    """
    DPO loss adapted for VLM: log-probs conditioned on both image + text prompt.
    batch keys: image_features, prompt_ids, winner_ids, loser_ids.
    Same formula as text DPO but forward pass includes image tokens.
    """
    raise NotImplementedError
```

Task: apply DPO (with LoRA via peft) to LLaVA-1.5-7B on MSCOCO caption preferences.
Verify: CHAIR_s reduces by ≥ 5 percentage points before/after DPO fine-tuning.
Note: if GPU memory is insufficient, provide fallback using LLaVA-1.5-7B with 4-bit quantization.

- [ ] **Step 5: Convert, verify, commit**

```bash
cd module_07_alignment_frontier
uv run jupytext --to notebook assignment1_dpo_vs_rlhf.py
uv run jupytext --to notebook assignment2_process_reward_models.py
uv run jupytext --to notebook assignment3_vlm_alignment.py
cd ..
git add module_07_alignment_frontier/
git commit -m "feat: module 07 — alignment frontier lecture notes + DPO/PRM/VLM assignments"
```

---

### Task 9: Projects Files + Final Touches

**Files:**
- Create: `projects/midterm_project.md`
- Create: `projects/final_project.md`

- [ ] **Step 1: Write `projects/midterm_project.md`**

Three options with full rubrics (copy from README + expand with submission format, evaluation criteria, expected deliverables):
- Option A: Custom Env + PPO with curriculum learning
- Option B: Offline RL (CQL/IQL on D4RL)
- Option C: Reward Hacking Study (empirical paper format)

Each option: Background (1 paragraph), Task specification, Success criteria, Deliverables (code + writeup), Suggested timeline.

- [ ] **Step 2: Write `projects/final_project.md`**

Three options:
- Option A: GRPO for Reasoning
- Option B: VLM Alignment Research (target workshop submission)
- Option C: RL for Diffusion/Flow Matching (DDPO)

Each option: same structure as midterm + Related Work pointers (3-4 specific papers), Baseline to beat, Workshop/venue suggestion.

- [ ] **Step 3: Verify full repo structure**

```bash
find . -name "*.md" -o -name "*.ipynb" -o -name "*.py" | grep -v ".git" | sort
```

Expected: 7 lecture_notes.md, 19 .ipynb files, 19 .py source files, 2 project files, pyproject.toml.

- [ ] **Step 4: Commit**

```bash
git add projects/
git commit -m "feat: midterm and final project briefs"
```

---

### Task 10: Solutions Branch

**Files:** All `.py` source files (the ones converted to notebooks)

- [ ] **Step 1: Create solutions branch**

```bash
git checkout -b solutions
```

- [ ] **Step 2: Fill in all NotImplementedError stubs**

For each `.py` source file across all modules, replace every `raise NotImplementedError` with correct implementation. Key implementations to verify:

- `rllearn/buffers.py`: `ReplayBuffer`, `PrioritizedReplayBuffer`, `RolloutBuffer`
- `rllearn/networks.py`: `MLP`, `DuelingNet`, `ActorCritic`, `GaussianPolicyHead`, `TwinQNetwork`
- `rllearn/envs.py`: `make_env`, `NormalizeObsWrapper`
- `rllearn/utils.py`: `compute_returns`, `compute_gae`
- All assignment skeleton functions

- [ ] **Step 3: Regenerate notebooks from solved sources**

```bash
for f in module_*/assignment*.py; do
  uv run jupytext --to notebook "$f"
done
```

- [ ] **Step 4: Commit solutions**

```bash
git add -A
git commit -m "solutions: complete implementations for all assignments"
```

- [ ] **Step 5: Return to master**

```bash
git checkout master
echo "solutions branch created; not linked from README"
```

---

## Self-Review Notes

- All 11 tasks are independently executable after Task 0 + Task 1 are complete
- Tasks 2–8 (modules) can run in parallel since lecture notes + notebooks don't depend on each other
- `rllearn/` stubs are importable from Task 1 onward; implementations happen within module tasks
- No TBD or placeholder content — every equation is specified, every function signature is concrete
- Success thresholds specified for every training verification step
- Solutions branch is last; depends on all module tasks completing first
