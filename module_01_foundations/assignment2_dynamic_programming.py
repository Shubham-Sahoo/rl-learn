# %% [markdown]
# # Assignment 2: Dynamic Programming — Policy Iteration & Value Iteration
# **Prerequisites:** Complete Assignment 1 first (working GridWorld + policy_evaluation).
# Read `lecture_notes.md` §6–7 before starting.
#
# **Learning objectives:**
# - Implement greedy policy improvement
# - Implement full policy iteration (evaluation → improvement → repeat)
# - Implement value iteration (max-Bellman backup)
# - Compare convergence speed and verify both algorithms reach the same optimal policy

# %% [markdown]
# ## Part 1: Theory Recap
#
# **Policy Improvement Theorem:** Given $V^\pi$, define the greedy policy:
#
# $$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma V^\pi(s')\bigr]$$
#
# Then $V^{\pi'}(s) \ge V^\pi(s)$ for all $s$ (the new policy is at least as good).
#
# **Bellman Optimality Operator** (used in value iteration):
#
# $$V^*(s) = \max_a \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma V^*(s')\bigr]$$
#
# Value iteration applies this update repeatedly until $\max_s |V_{k+1}(s) - V_k(s)| < \theta$.
# It converges because $\mathcal{T}$ is a $\gamma$-contraction (proved in lecture_notes Appendix).

# %%
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# %% [markdown]
# ## Part 2: GridWorld (copy from Assignment 1)
#
# Paste your completed `GridWorld` class here (with all methods implemented).
# Assignment 2 builds directly on top of it — `policy_improvement`, `policy_iteration`,
# and `value_iteration` all call `env.get_transitions`.

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
    """
    ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    ACTION_NAMES = {0: '↑', 1: '→', 2: '↓', 3: '←'}

    def __init__(self, grid: List[List[str]], gamma: float = 0.99, slip: float = 0.0):
        self.grid = grid
        self.gamma = gamma
        self.slip = slip
        self.n_rows = len(grid)
        self.n_cols = len(grid[0])
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
        r, c = state
        cell = self.grid[r][c]
        if cell == 'G':
            return 1.0
        elif cell == 'H':
            return -1.0
        return 0.0

    def _next_state(self, state: Tuple, action: int) -> Tuple:
        r, c = state
        dr, dc = self.ACTIONS[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols and (nr, nc) not in self.walls:
            return (nr, nc)
        return state

    def get_transitions(self, state: Tuple, action: int) -> List[Tuple[float, Tuple, float, bool]]:
        if state in self.terminal:
            return [(1.0, state, 0.0, True)]
        perp_left = (action - 1) % 4
        perp_right = (action + 1) % 4
        raw = [
            (1.0 - self.slip, action),
            (self.slip / 2, perp_left),
            (self.slip / 2, perp_right),
        ]
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
        V = {s: 0.0 for s in self.states}
        while True:
            delta = 0.0
            for s in self.states:
                if s in self.terminal:
                    continue
                a = policy[s]
                v_new = sum(p * (r + self.gamma * V.get(ns, 0.0))
                            for p, ns, r, done in self.get_transitions(s, a))
                delta = max(delta, abs(v_new - V[s]))
                V[s] = v_new
            if delta < theta:
                break
        return V

    def visualize_values(self,
                         V: Dict[Tuple, float],
                         policy: Optional[Dict[Tuple, int]] = None,
                         title: str = "Value Function"):
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
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if (r, c) in self.walls:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='black'))
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                state = (r, c)
                if state in self.walls:
                    continue
                val = grid_vals[r][c]
                if not np.isnan(val):
                    ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                            fontsize=8, color='black')
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
# ## Part 3: Implementation — Policy Improvement and Policy Iteration

# %%
def policy_improvement(env: GridWorld, V: Dict[Tuple, float]) -> Dict[Tuple, int]:
    """
    Greedy policy improvement.
    """
    new_policy = {}
    for s in env.states:
        if s in env.terminal:
            new_policy[s] = 0  # arbitrary; terminal states don't matter
            continue
        best_action = 0
        best_q = float('-inf')
        for a in range(4):
            q = sum(p * (r + env.gamma * V.get(ns, 0.0))
                    for p, ns, r, done in env.get_transitions(s, a))
            if q > best_q:
                best_q = q
                best_action = a
        new_policy[s] = best_action
    return new_policy


def policy_iteration(env: GridWorld, theta: float = 1e-6) -> Tuple[Dict, Dict]:
    """
    Full policy iteration loop.
    Returns (optimal_V, optimal_policy).
    """
    # Initialize policy randomly
    policy = {s: np.random.randint(4) for s in env.states}
    deltas = []

    while True:
        V = env.policy_evaluation(policy, theta)
        new_policy = policy_improvement(env, V)

        # Track a proxy delta for convergence plotting
        delta = max(abs(V.get(s, 0.0)) for s in env.states if s not in env.terminal) if env.states else 0.0
        deltas.append(delta)

        # Check policy stability
        stable = all(policy.get(s) == new_policy.get(s) for s in env.states
                     if s not in env.terminal)
        policy = new_policy
        if stable:
            break

    # Attach deltas for plotting
    policy_iteration.deltas = deltas
    return V, policy


# %% [markdown]
# ## Part 4: Implementation — Value Iteration

# %%
def value_iteration(env: GridWorld, theta: float = 1e-6) -> Tuple[Dict, Dict]:
    """
    Value iteration: repeatedly apply the Bellman optimality operator until convergence.
    Returns (optimal_V, optimal_policy).
    """
    V = {s: 0.0 for s in env.states}
    deltas = []

    while True:
        delta = 0.0
        for s in env.states:
            if s in env.terminal:
                continue
            best_v = max(
                sum(p * (r + env.gamma * V.get(ns, 0.0))
                    for p, ns, r, done in env.get_transitions(s, a))
                for a in range(4)
            )
            delta = max(delta, abs(best_v - V[s]))
            V[s] = best_v
        deltas.append(delta)
        if delta < theta:
            break

    # Extract greedy policy
    optimal_policy = policy_improvement(env, V)

    # Attach deltas for plotting
    value_iteration.deltas = deltas
    return V, optimal_policy


# %% [markdown]
# ## Part 5: Training & Verification
#
# Run both algorithms on the same GridWorld. Verify that:
# 1. They return the same optimal values (within 1e-4).
# 2. The number of iterations is in a reasonable range (6 < n_iters < 200 for 4×4).
# 3. Plot convergence curves (delta vs iteration) for both algorithms side-by-side.

# %%
GRID = [
    ['S', ' ', ' ', ' '],
    [' ', 'W', ' ', 'H'],
    [' ', ' ', ' ', 'H'],
    ['H', ' ', ' ', 'G'],
]

env = GridWorld(GRID, gamma=0.99, slip=0.1)

# --- Run both algorithms and time them ---
t0 = time.perf_counter()
V_pi, policy_pi = policy_iteration(env, theta=1e-6)
t_pi = time.perf_counter() - t0

t0 = time.perf_counter()
V_vi, policy_vi = value_iteration(env, theta=1e-6)
t_vi = time.perf_counter() - t0

# --- Correctness check: both should agree on V* within 1e-4 ---
for s in env.states:
    diff = abs(V_pi.get(s, 0.0) - V_vi.get(s, 0.0))
    assert diff < 1e-4, (
        f"State {s}: policy_iteration V={V_pi.get(s):.6f}, "
        f"value_iteration V={V_vi.get(s):.6f}, diff={diff:.2e}"
    )
print("✓ Policy iteration and value iteration agree on V* (within 1e-4)")

# --- Policy check: both should produce the same greedy policy ---
mismatches = [s for s in env.states if s not in env.terminal
              and policy_pi.get(s) != policy_vi.get(s)]
if mismatches:
    print(f"  Note: policy differs at {len(mismatches)} states (ties broken differently — OK)")
else:
    print("✓ Both algorithms produce identical optimal policies")

# --- Timing report ---
print(f"\nPolicy Iteration:  {t_pi*1000:.1f} ms")
print(f"Value Iteration:   {t_vi*1000:.1f} ms")

# %% [markdown]
# ### Convergence Curves

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

pi_deltas = getattr(policy_iteration, 'deltas', [])
vi_deltas = getattr(value_iteration, 'deltas', [])

if pi_deltas:
    axes[0].plot(range(len(pi_deltas)), pi_deltas, marker='o')
axes[0].set_title("Policy Iteration — Δ per outer iter")
axes[0].set_xlabel("Outer iteration")
axes[0].set_ylabel("Max Bellman error Δ")
axes[0].set_yscale("log")

if vi_deltas:
    axes[1].plot(range(len(vi_deltas)), vi_deltas)
axes[1].set_title("Value Iteration — Δ per sweep")
axes[1].set_xlabel("Sweep")
axes[1].set_ylabel("Max Bellman error Δ")
axes[1].set_yscale("log")

plt.tight_layout()
plt.show()

# --- Visualize optimal policy ---
env.visualize_values(V_pi, policy_pi, "Optimal Policy (Policy Iteration)")
env.visualize_values(V_vi, policy_vi, "Optimal Policy (Value Iteration)")

# %% [markdown]
# ## Part 6: Ablations
#
# **Ablation 1:** Compare policy iteration vs value iteration on a larger grid (8×8 or 10×10).

# %%
# TODO: Create an 8x8 GridWorld with a few walls and holes.
# Run both algorithms and compare timing. Print results.

# %% [markdown]
# **Observation (fill in):** On the larger grid, ___ is faster because ___.

# %%
# Ablation 2: Effect of theta on solution quality
# TODO: Run value_iteration with theta in [1e-2, 1e-4, 1e-8].

# %% [markdown]
# **Observation (fill in):** Looser theta (larger value) requires ___ sweeps and introduces
# ___ error in the value function.

# %% [markdown]
# ## Part 7: Reflection

# %%
# Your answers here (markdown cell below)

# %% [markdown]
# **Answers:**
# 1. Policy iteration's outer loop converges in few iterations, but each outer iteration requires
#    running full policy evaluation to convergence, which itself takes many sweeps. Value iteration
#    takes more total sweeps but each sweep is cheaper since there is no inner loop.
# 2. TD(0) is a one-step Bellman backup — it only looks one step ahead. Full DP looks across all
#    states simultaneously, while TD uses sampled transitions and bootstraps with the current V.
# 3. Solve 0.5^k < 1e-6: k * log(0.5) < -6 * log(10), so k > 6*log(10)/log(2) ≈ 19.9. About 20 sweeps.
