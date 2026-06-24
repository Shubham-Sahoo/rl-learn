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
        self.grid = grid
        self.gamma = gamma
        self.slip = slip
        self.n_rows = len(grid)
        self.n_cols = len(grid[0])

        # Build set of valid states (non-wall cells)
        self.states: List[Tuple] = []
        self.terminal: set = set()
        self.walls: set = set()

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                cell = grid[r][c]
                if cell == 'W':
                    self.walls.add((r, c))
                else:
                    self.states.append((r, c))
                    if cell in ('G', 'H'):
                        self.terminal.add((r, c))

    def _reward(self, state: Tuple) -> float:
        """
        Return the immediate reward for *being in* `state`.

        +1.0 for goal 'G', -1.0 for hole 'H', 0.0 otherwise.
        """
        r, c = state
        cell = self.grid[r][c]
        if cell == 'G':
            return 1.0
        elif cell == 'H':
            return -1.0
        return 0.0

    def _next_state(self, state: Tuple, action: int) -> Tuple:
        """
        Return the deterministic next state when taking `action` from `state`.

        If the resulting position is out-of-bounds or a wall, return `state` unchanged
        (the agent bounces back).
        """
        r, c = state
        dr, dc = self.ACTIONS[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols and (nr, nc) not in self.walls:
            return (nr, nc)
        return state

    def get_transitions(self, state: Tuple, action: int) -> List[Tuple[float, Tuple, float, bool]]:
        """
        Return the full transition distribution for (state, action).

        Returns a list of tuples: (probability, next_state, reward, done).
        """
        # Terminal states absorb
        if state in self.terminal:
            return [(1.0, state, 0.0, True)]

        # Build raw (prob, action) list
        perp_left = (action - 1) % 4
        perp_right = (action + 1) % 4
        raw = [
            (1.0 - self.slip, action),
            (self.slip / 2, perp_left),
            (self.slip / 2, perp_right),
        ]

        # Merge probabilities for identical next_states
        prob_dict: Dict[Tuple, float] = {}
        for prob, act in raw:
            if prob == 0.0:
                continue
            ns = self._next_state(state, act)
            prob_dict[ns] = prob_dict.get(ns, 0.0) + prob

        transitions = []
        for ns, prob in prob_dict.items():
            reward = self._reward(ns)
            done = ns in self.terminal
            transitions.append((prob, ns, reward, done))
        return transitions

    def policy_evaluation(self, policy: Dict[Tuple, int], theta: float = 1e-6) -> Dict[Tuple, float]:
        """
        Iterative policy evaluation.
        """
        V = {s: 0.0 for s in self.states}

        while True:
            delta = 0.0
            for s in self.states:
                if s in self.terminal:
                    continue
                a = policy[s]
                v_new = 0.0
                for prob, ns, reward, done in self.get_transitions(s, a):
                    v_new += prob * (reward + self.gamma * V.get(ns, 0.0))
                delta = max(delta, abs(v_new - V[s]))
                V[s] = v_new
            if delta < theta:
                break
        return V

    def visualize_values(self,
                         V: Dict[Tuple, float],
                         policy: Optional[Dict[Tuple, int]] = None,
                         title: str = "Value Function"):
        """
        Visualize the value function as a heatmap. Optionally overlay policy arrows.
        """
        grid_vals = np.full((self.n_rows, self.n_cols), np.nan)

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                state = (r, c)
                if state in self.walls:
                    continue
                cell = self.grid[r][c]
                if cell == 'G':
                    grid_vals[r][c] = 1.0
                elif cell == 'H':
                    grid_vals[r][c] = -1.0
                else:
                    grid_vals[r][c] = V.get(state, 0.0)

        fig, ax = plt.subplots(figsize=(self.n_cols * 1.2, self.n_rows * 1.2))
        im = ax.imshow(grid_vals, cmap='RdYlGn', vmin=-1, vmax=1, aspect='equal')

        # Mark walls black
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if (r, c) in self.walls:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='black'))

        # Annotate cells
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                state = (r, c)
                if state in self.walls:
                    continue
                val = grid_vals[r][c]
                if not np.isnan(val):
                    ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                            fontsize=8, color='black')

        # Overlay policy arrows
        if policy is not None:
            for s in self.states:
                if s in self.terminal:
                    continue
                r, c = s
                if s in policy:
                    ax.text(c, r - 0.3, self.ACTION_NAMES[policy[s]],
                            ha='center', va='center', fontsize=12, color='navy')

        ax.set_title(title)
        ax.set_xticks(range(self.n_cols))
        ax.set_yticks(range(self.n_rows))
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()


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
env_myopic = GridWorld(GRID, gamma=0.1, slip=0.1)
V_myopic = env_myopic.policy_evaluation(uniform_policy)
env_myopic.visualize_values(V_myopic, uniform_policy, "Myopic Agent (gamma=0.1)")

# %% [markdown]
# **Observation (fill in):** With gamma=0.1, states far from the goal have values close to 0
# because future rewards are discounted so heavily they contribute negligibly.

# %%
# Ablation 2: slip=0.3 (very stochastic environment)
env_stoch = GridWorld(GRID, gamma=0.99, slip=0.3)
V_stoch = env_stoch.policy_evaluation(uniform_policy)
env_stoch.visualize_values(V_stoch, uniform_policy, "Stochastic Environment (slip=0.3)")

# %% [markdown]
# **Observation (fill in):** With slip=0.3, the value of states adjacent to holes decreases
# compared to slip=0.0 because there is a higher probability of accidentally falling into a hole.

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
# 1. The Markov property ensures that V(s) depends only on the current state s, not the history.
#    Without it, we cannot define a consistent Bellman backup since the value of s would depend
#    on how we arrived at s.
# 2. With a single end-of-response reward, earlier tokens receive no direct credit signal —
#    analogous to states far from the goal in GridWorld. The agent must propagate reward backwards
#    through the trajectory, which is the credit assignment problem.
# 3. If gamma=1 and there are no terminal states, the Bellman updates never converge because
#    returns are infinite sums that do not contract to a fixed point.
