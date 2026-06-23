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
        raise NotImplementedError

    def _reward(self, state: Tuple) -> float:
        raise NotImplementedError

    def _next_state(self, state: Tuple, action: int) -> Tuple:
        raise NotImplementedError

    def get_transitions(self, state: Tuple, action: int) -> List[Tuple[float, Tuple, float, bool]]:
        raise NotImplementedError

    def policy_evaluation(self, policy: Dict[Tuple, int], theta: float = 1e-6) -> Dict[Tuple, float]:
        raise NotImplementedError

    def visualize_values(self,
                         V: Dict[Tuple, float],
                         policy: Optional[Dict[Tuple, int]] = None,
                         title: str = "Value Function"):
        raise NotImplementedError


# %% [markdown]
# ## Part 3: Implementation — Policy Improvement and Policy Iteration

# %%
def policy_improvement(env: GridWorld, V: Dict[Tuple, float]) -> Dict[Tuple, int]:
    """
    Greedy policy improvement.

    For each non-terminal state, compute the action-value:
        Q(s, a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]

    Then set:
        π'(s) = argmax_a Q(s, a)

    For terminal states, return any fixed action (e.g., 0) — it does not matter because
    the episode ends immediately.

    Parameters
    ----------
    env : GridWorld instance (provides get_transitions, states, terminal, gamma).
    V   : dict mapping state → float value (result of policy_evaluation).

    Returns
    -------
    new_policy : dict mapping state → int action.

    Common mistake: forgetting terminal states in the output dict — make sure every state
    in env.states has an entry in the returned policy.
    """
    raise NotImplementedError


def policy_iteration(env: GridWorld, theta: float = 1e-6) -> Tuple[Dict, Dict]:
    """
    Full policy iteration loop.

    Algorithm:
        1. Initialize policy π randomly (or all-zeros).
        2. Repeat:
            a. Evaluate: V ← env.policy_evaluation(π, theta)
            b. Improve:  π' ← policy_improvement(env, V)
            c. If π' == π (for all states), stop.
            d. Set π ← π'.
        3. Return (V, π).

    Track per-iteration convergence delta (max |V_new - V_old| at end of each eval)
    and store in a list for plotting.

    Parameters
    ----------
    env   : GridWorld instance.
    theta : convergence threshold for policy evaluation.

    Returns
    -------
    (optimal_V, optimal_policy) : tuple of (dict, dict).

    Hint: to detect policy stability, compare π'[s] == π[s] for all s.
    """
    raise NotImplementedError


# %% [markdown]
# ## Part 4: Implementation — Value Iteration

# %%
def value_iteration(env: GridWorld, theta: float = 1e-6) -> Tuple[Dict, Dict]:
    """
    Value iteration: repeatedly apply the Bellman optimality operator until convergence.

    Update rule (applied to every non-terminal state each sweep):
        V(s) ← max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]

    Stop when max_s |V_new(s) - V_old(s)| < theta.

    After convergence, extract the greedy policy with one pass (same as policy_improvement).

    Track delta per iteration for convergence plotting.

    Parameters
    ----------
    env   : GridWorld instance.
    theta : convergence threshold.

    Returns
    -------
    (optimal_V, optimal_policy) : tuple of (dict, dict).

    Hint: terminal states always have V(s) = 0; do not update them.
    Also track deltas in a list so you can plot convergence curves in the verification section.
    """
    raise NotImplementedError


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
#
# Plot the convergence delta (max Bellman error) vs iteration for both algorithms.
# Policy iteration should converge in very few outer iterations (each costing a full eval).
# Value iteration converges monotonically, taking more steps but each step is cheap.

# %%
# NOTE: To make this work, your policy_iteration and value_iteration functions must
# return convergence delta histories. Modify them to also return deltas, or store them
# as attributes on the return value, then plot here.
#
# Expected plot: value_iteration shows a smooth exponential decay.
# Policy iteration shows a small number of points (one per outer iteration).

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# TODO: plot deltas from policy_iteration on axes[0]
# TODO: plot deltas from value_iteration on axes[1]
axes[0].set_title("Policy Iteration — Δ per outer iter")
axes[0].set_xlabel("Outer iteration")
axes[0].set_ylabel("Max Bellman error Δ")
axes[0].set_yscale("log")

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
# Which is faster in wall-clock time? Does the gap match your expectations from the lecture notes?

# %%
# TODO: Create an 8x8 GridWorld with a few walls and holes.
# Run both algorithms and compare timing. Print results.

# %% [markdown]
# **Observation (fill in):** On the larger grid, ___ is faster because ___.

# %%
# Ablation 2: Effect of theta on solution quality
# TODO: Run value_iteration with theta in [1e-2, 1e-4, 1e-8].
# For each theta, measure: number of sweeps, max error vs the theta=1e-8 solution.
# Plot a table or bar chart.

# %% [markdown]
# **Observation (fill in):** Looser theta (larger value) requires ___ sweeps and introduces
# ___ error in the value function.

# %% [markdown]
# ## Part 7: Reflection
#
# Answer the questions below in the markdown cell provided.
#
# 1. Policy iteration converges in few outer iterations. Why does this not mean it is always
#    faster than value iteration? (Think about what each outer iteration costs.)
# 2. In RLHF PPO, the critic $V^\pi(s)$ is updated using TD(0): a one-step Bellman backup.
#    How does this relate to value iteration? What is "missing" compared to full DP?
# 3. The Bellman optimality operator is a $\gamma$-contraction. If $\gamma = 0.5$, how many
#    value iteration sweeps are needed to reduce the initial error by a factor of $10^6$?
#    (Hint: solve $0.5^k < 10^{-6}$.)

# %%
# Your answers here (markdown cell below)

# %% [markdown]
# **Answers:**
# 1.
# 2.
# 3.
