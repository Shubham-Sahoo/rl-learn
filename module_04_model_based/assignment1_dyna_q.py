# %% [markdown]
# # Module 04, Assignment 1: Dyna-Q

# %%
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Tuple

from rllearn.logging import make_writer

# %% [markdown]
# ## Part 2: Implement DynaQAgent

# %%
class DynaQAgent:
    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 n_planning_steps: int = 10):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning_steps = n_planning_steps
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)
        self.model: Dict[Tuple[int, int], Tuple[float, int]] = {}

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update_q(self, state: int, action: int, reward: float, next_state: int) -> float:
        """Direct RL update from real experience. Returns TD error."""
        td_target = reward + self.gamma * np.max(self.Q[next_state])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return float(td_error)

    def update_model(self, state: int, action: int, reward: float, next_state: int):
        """Store observed transition in model."""
        self.model[(state, action)] = (reward, next_state)

    def plan(self):
        """Run n_planning_steps synthetic Q-updates from model."""
        if not self.model:
            return
        keys = list(self.model.keys())
        for _ in range(self.n_planning_steps):
            s, a = random.choice(keys)
            r, ns = self.model[(s, a)]
            td_target = r + self.gamma * np.max(self.Q[ns])
            td_error = td_target - self.Q[s, a]
            self.Q[s, a] += self.alpha * td_error

    def step(self, state: int, action: int, reward: float,
             next_state: int, done: bool):
        """Full Dyna-Q step: update_q + update_model + plan."""
        self.update_q(state, action, reward, next_state)
        self.update_model(state, action, reward, next_state)
        self.plan()


# %% [markdown]
# ## Part 3: Training Loop

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
    """Train Dyna-Q on FrozenLake-v1 (deterministic)."""
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
# ### Verification

# %%
print("Training Dyna-Q with n_planning_steps=50 ...")
rewards_50, steps_50 = train_dyna_q(n_planning_steps=50, n_episodes=300, seed=42)

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
# ## Part 5: Reflection

# %% [markdown]
# **Answers:**
# 1. Each planning step is analogous to a CoT reasoning step: instead of taking a real action,
#    the agent "thinks" about a past experience, propagating value information.
# 2. With is_slippery=True, the model would store deterministic transitions that are wrong. Planning
#    would propagate incorrect Q-values leading to suboptimal or wrong policies.
# 3. The crossover is where cost_per_planning_step * n_planning >= cost_per_real_step. Beyond that
#    point (n >= 500 here), additional planning is not economical.
