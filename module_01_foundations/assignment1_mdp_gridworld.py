# %% [markdown]
# # Assignment 1: GridWorld MDP from Scratch
# **Prerequisites:** Read `lecture_notes.md` §1–5 before starting.
#
# **Learning objectives:**
# - Implement a stochastic GridWorld as a formal MDP
# - Perform policy evaluation by solving Bellman equations iteratively
# - Visualize value functions as heatmaps
# - Observe how discount factor γ changes agent behavior

# %% [markdown]
# ## Part 1: Theory Recap
#
# The Bellman expectation equation for $V^\pi$:
#
# $$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma V^\pi(s')\bigr]$$
#
# Policy evaluation iterates this update until $\max_s |V_{k+1}(s) - V_k(s)| < \theta$.
#
# The state-value and action-value functions are related by:
#
# $$V^\pi(s) = \sum_a \pi(a|s)\, Q^\pi(s,a)$$
#
# A **stochastic** GridWorld adds a slip probability: with prob `slip/2` each, the agent moves
# in one of the two perpendicular directions instead of the intended one.

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional

# %% [markdown]
# ## Part 2: Implementation
#
# Implement the `GridWorld` class below. Each method has a docstring explaining what it should do.
# All unimplemented methods raise `NotImplementedError` — replace that with your implementation.
#
# **Hints:**
# - For `get_transitions`: terminal states should absorb (return immediately, no further transitions).
# - For `policy_evaluation`: terminal states always have value 0; do not update them.
# - For `visualize_values`: use `plt.imshow`, annotate each cell with its value, overlay arrows.

# %%
class GridWorld:
    """
    Configurable stochastic GridWorld MDP.

    Grid symbols:
        ' ' = empty cell
        'W' = wall (impassable; not a state)
        'G' = goal  (reward +1, terminal)
        'H' = hole  (reward -1, terminal)
        'S' = start (reward  0, non-terminal)

    Actions: 0=up, 1=right, 2=down, 3=left.

    Stochastic dynamics (slip > 0):
        With prob (1 - slip): take intended action.
        With prob slip/2:     move perpendicular left  (action - 1) % 4.
        With prob slip/2:     move perpendicular right (action + 1) % 4.

    If an action would move the agent into a wall or off the grid, the agent stays in place.
    """
    ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    ACTION_NAMES = {0: '↑', 1: '→', 2: '↓', 3: '←'}

    def __init__(self, grid: List[List[str]], gamma: float = 0.99, slip: float = 0.0):
        """
        Parameters
        ----------
        grid  : 2-D list of strings (symbols defined above).
        gamma : discount factor ∈ [0, 1).
        slip  : probability of a perpendicular move ∈ [0, 1).
        """
        raise NotImplementedError

    def _reward(self, state: Tuple) -> float:
        """
        Return the immediate reward for *being in* `state`.

        +1.0 for goal 'G', -1.0 for hole 'H', 0.0 otherwise.
        """
        raise NotImplementedError

    def _next_state(self, state: Tuple, action: int) -> Tuple:
        """
        Return the deterministic next state when taking `action` from `state`.

        If the resulting position is out-of-bounds or a wall, return `state` unchanged
        (the agent bounces back).
        """
        raise NotImplementedError

    def get_transitions(self, state: Tuple, action: int) -> List[Tuple[float, Tuple, float, bool]]:
        """
        Return the full transition distribution for (state, action).

        Returns a list of tuples: (probability, next_state, reward, done).

        Special cases:
        - If `state` is terminal, return [(1.0, state, 0.0, True)] — the episode is over,
          no reward is collected and the agent stays in the absorbing terminal state.
        - If slip == 0, the list has exactly one entry with probability 1.0.
        - If slip > 0, merge probabilities for identical next_states (so the list never has
          duplicate next_states).

        Steps to implement:
        1. Return early if state is terminal.
        2. Compute perpendicular actions: left = (action - 1) % 4, right = (action + 1) % 4.
        3. Build a raw list: [(1 - slip, intended), (slip/2, perp_left), (slip/2, perp_right)].
        4. For each (prob, act), compute next_state = self._next_state(state, act).
        5. Merge: accumulate probabilities for identical next_states into a dict, then convert
           back to a list of (prob, next_state, reward, done) tuples.
        6. reward = self._reward(next_state); done = next_state in self.terminal.
        """
        raise NotImplementedError

    def policy_evaluation(self, policy: Dict[Tuple, int], theta: float = 1e-6) -> Dict[Tuple, float]:
        """
        Iterative policy evaluation. Repeatedly apply:

            V(s) ← Σ_{s'} P(s'|s,π(s)) [R(s,π(s),s') + γ V(s')]

        until max_s |V_new(s) - V_old(s)| < theta.

        Returns V: dict mapping each state → float value.

        Common mistakes to avoid:
        - Do NOT update terminal states; their value is always 0.
        - Use the value from the *start* of the sweep (synchronous update), not in-place
          updates within a sweep — or equivalently, use in-place but track delta correctly.
          (Either synchronous or in-place/asynchronous converges; synchronous is cleaner.)

        Steps to implement:
        1. Initialize V = {s: 0.0 for s in self.states}.
        2. Outer loop: repeat until delta < theta.
        3. Inner loop: for each non-terminal state s, compute the Bellman update using
           self.get_transitions(s, policy[s]).
        4. Track delta = max over all states of |V_new(s) - V_old(s)|.
        5. Return V.
        """
        raise NotImplementedError

    def visualize_values(self,
                         V: Dict[Tuple, float],
                         policy: Optional[Dict[Tuple, int]] = None,
                         title: str = "Value Function"):
        """
        Visualize the value function as a heatmap. Optionally overlay policy arrows.

        Steps to implement:
        1. Create a 2-D numpy array `grid_vals` of shape (n_rows, n_cols) filled with np.nan.
        2. Fill in grid_vals[r][c] = V[(r,c)] for each non-wall, non-terminal state.
           Set terminal 'G' cells to +1.0 and 'H' cells to -1.0 for visual clarity.
        3. Use plt.imshow(grid_vals, cmap='RdYlGn', vmin=-1, vmax=1) for a red-yellow-green
           color scale (red = bad, green = good).
        4. Annotate each non-wall cell with its value (2 decimal places).
        5. If policy is provided, overlay ACTION_NAMES arrows on non-terminal states.
        6. Mark wall cells as black (set their alpha or fill with a black patch).
        7. Add title, colorbar, and plt.tight_layout(); call plt.show().
        """
        raise NotImplementedError


# %% [markdown]
# ## Part 3: Training & Verification
#
# Run policy evaluation on a random policy over the standard 4×4 GridWorld.
# The verification checks that terminal states have V ≈ 0 (required by convention).

# %%
# Standard 4x4 grid (similar to FrozenLake-v1)
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
# **Observation (fill in):** With slip=0.3, the value of states adjacent to holes ___
# compared to slip=0.0 because ___.

# %% [markdown]
# ## Part 5: Reflection
#
# Answer the questions below in the markdown cell provided.
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
