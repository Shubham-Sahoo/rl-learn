# %% [markdown]
# # Module 04, Assignment 1: Dyna-Q
#
# ## Prerequisites
# - Module 01 A1: GridWorld environment
# - Lecture notes sections 1–2
#
# ## Learning Objectives
# 1. Understand how model-based planning augments model-free RL
# 2. Implement the Dyna-Q tabular model
# 3. Observe the sample efficiency gains from planning steps
# 4. Understand model exploitation and when it matters

# %%
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Tuple

from rllearn.logging import make_writer

# %% [markdown]
# ## Part 1: Theory Recap
#
# ### Dyna-Q Core Equations
#
# **Direct RL update (from real experience):**
#
# $$Q(s, a) \leftarrow Q(s, a) + \alpha \bigl[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\bigr]$$
#
# **Model update:**
#
# $$\mathcal{M}(s, a) \leftarrow (r, s') \quad \text{(deterministic tabular model)}$$
#
# **Planning update (from model):**
#
# $$\tilde{s}, \tilde{a} \sim \text{Uniform}(\mathcal{M}\text{.keys()})$$
#
# $$\tilde{r}, \tilde{s}' = \mathcal{M}(\tilde{s}, \tilde{a})$$
#
# $$Q(\tilde{s}, \tilde{a}) \leftarrow Q(\tilde{s}, \tilde{a}) + \alpha \bigl[\tilde{r} + \gamma \max_{a'} Q(\tilde{s}', a') - Q(\tilde{s}, \tilde{a})\bigr]$$
#
# The full Dyna-Q step for each real transition is: **direct update → model update → n planning updates**.
#
# **Intuition:** Each real step costs one environment interaction. Each planning step costs a dictionary
# lookup. With `n_planning_steps=50`, the agent extracts 51× the information from each real step,
# propagating value updates through 50 previously-seen transitions.

# %% [markdown]
# ## Part 2: Implement DynaQAgent
#
# Fill in every `raise NotImplementedError` below. Do not change any method signatures.
#
# **Implementation notes:**
# - `self.Q` is a 2D numpy array of shape `(n_states, n_actions)`, initialized to zeros.
# - `self.model` is a plain Python dict mapping `(state, action) -> (reward, next_state)`.
# - `select_action` uses ε-greedy: with probability `epsilon` pick a random action,
#   otherwise pick `argmax Q(state, :)`.
# - `update_q` performs a single Bellman update and returns the TD error (scalar float).
# - `plan` runs `self.n_planning_steps` synthetic Q-updates; each iteration samples a random
#   `(s, a)` pair from `self.model.keys()` (use `random.choice(list(...))`).
# - `step` calls `update_q`, `update_model`, then `plan` in that order.

# %%
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


# %% [markdown]
# ## Part 3: Training Loop
#
# We use FrozenLake-v1 with `is_slippery=False` — a deterministic 4×4 gridworld where the agent
# must navigate from start (top-left) to goal (bottom-right) avoiding holes.
#
# The training loop below is provided. Read through it before running.

# %%
def make_frozen_lake():
    """Create deterministic FrozenLake-v1."""
    return gym.make("FrozenLake-v1", is_slippery=False)


def train_dyna_q(n_planning_steps: int = 10,
                 n_episodes: int = 300,
                 alpha: float = 0.5,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 seed: int = 42,
                 log_tensorboard: bool = False) -> Tuple[List[float], List[int]]:
    """Train Dyna-Q on FrozenLake-v1 (deterministic).

    Returns
    -------
    episode_rewards : list of episode total reward (0.0 or 1.0 on FrozenLake)
    real_steps      : cumulative real environment steps at end of each episode
    """
    np.random.seed(seed)
    random.seed(seed)

    env = make_frozen_lake()
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = DynaQAgent(
        n_states=n_states,
        n_actions=n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        n_planning_steps=n_planning_steps,
    )

    writer = None
    if log_tensorboard:
        writer = make_writer(f"dyna_q_n{n_planning_steps}")

    episode_rewards: List[float] = []
    real_steps: List[int] = []
    total_steps = 0

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.step(state, action, float(reward), next_state, done)
            state = next_state
            ep_reward += reward
            total_steps += 1

        episode_rewards.append(ep_reward)
        real_steps.append(total_steps)

        if writer is not None:
            cumulative_reward = float(np.sum(episode_rewards))
            writer.add_scalar("train/cumulative_reward", cumulative_reward, ep)
            writer.add_scalar("train/episode_reward", ep_reward, ep)

    if writer is not None:
        writer.close()

    env.close()
    return episode_rewards, real_steps


# %% [markdown]
# ### TensorBoard (optional — run the cell below before training)

# %%
# %load_ext tensorboard
# %tensorboard --logdir runs/

# %% [markdown]
# ### Verification
#
# With `n_planning_steps=50`, Dyna-Q should solve FrozenLake (reach the goal) within 50 real
# environment steps. A "solve" is defined as the first episode that earns reward 1.0.

# %%
print("Training Dyna-Q with n_planning_steps=50 ...")
rewards_50, steps_50 = train_dyna_q(n_planning_steps=50, n_episodes=300, seed=42)

# Find the first episode where the agent reached the goal
first_success = next((i for i, r in enumerate(rewards_50) if r > 0.0), None)
real_steps_to_first_success = steps_50[first_success] if first_success is not None else None

if first_success is not None:
    print(f"First success at episode {first_success}, real env steps used: {real_steps_to_first_success}")
else:
    print("Agent never reached the goal — check your implementation.")

assert first_success is not None, "Agent never reached the goal in 300 episodes."
assert real_steps_to_first_success <= 50, (
    f"Expected first success within 50 real steps, got {real_steps_to_first_success}. "
    "Increase n_planning_steps or fix the plan() implementation."
)
print(f"Verification passed: optimal policy found in {real_steps_to_first_success} real steps (<= 50).")

# %% [markdown]
# ## Part 4: Ablation — Planning Steps
#
# Compare cumulative reward vs. real environment steps for `n_planning_steps` ∈ {0, 5, 20, 50}.
# Plot all four curves on the same axes.

# %%
planning_configs = [0, 5, 20, 50]
results: Dict[int, Tuple[List[float], List[int]]] = {}

for n in planning_configs:
    print(f"Training n_planning_steps={n} ...")
    r, s = train_dyna_q(n_planning_steps=n, n_episodes=300, seed=42)
    results[n] = (r, s)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for n, (ep_rewards, real_steps_list) in results.items():
    cumulative = np.cumsum(ep_rewards)
    axes[0].plot(cumulative, label=f"n_plan={n}")
    axes[1].plot(real_steps_list, cumulative, label=f"n_plan={n}")

axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Cumulative Reward")
axes[0].set_title("Cumulative Reward vs. Episode")
axes[0].legend()

axes[1].set_xlabel("Real Environment Steps")
axes[1].set_ylabel("Cumulative Reward")
axes[1].set_title("Cumulative Reward vs. Real Steps (Sample Efficiency)")
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Observations
#
# Fill in your observations after running the ablation:
#
# 1. **Effect of planning steps on episode-based learning:** Higher `n_planning_steps` should reach
#    higher cumulative reward faster in terms of episodes. Why? Each episode's real transitions are
#    replayed many times synthetically, propagating value information across the Q-table.
#
# 2. **Effect on real-step efficiency (right plot):** Agents with more planning steps should achieve
#    the same cumulative reward with fewer real environment steps. This is the core Dyna-Q benefit.
#
# 3. **Diminishing returns:** Going from n=0 to n=5 is a large jump; n=20 to n=50 is smaller.
#    At some point, the model is perfectly learned and additional planning steps add no new information.
#
# *(Replace this with your own observations after running.)*

# %% [markdown]
# ## Part 5: Reflection
#
# Answer the questions below after completing Parts 2–4.
#
# **Q1 — Chain-of-thought and planning steps:**
# Dyna-Q's planning steps allow the agent to "think" about previously seen transitions before
# acting. How does this connect to chain-of-thought prompting in LLMs? In what sense is each
# CoT reasoning step analogous to a Dyna-Q planning step?
#
# **Q2 — Model exploitation:**
# What happens if the model is wrong (e.g., FrozenLake with `is_slippery=True` but the agent's
# model assumes deterministic transitions)? Sketch the failure mode: what will `self.model` store,
# and how will planning with it mislead the policy?
#
# **Q3 — Real-world deployment:**
# You are building an RL agent for a robot arm. Real environment steps cost $0.50 each (robot
# time, wear). Model training costs $0.001 per gradient step. At what point does adding more
# planning steps stop being economical? What factors determine the crossover?

# %% [markdown]
# **Answers:**
# 1.
# 2.
# 3.
