# %% [markdown]
# # Assignment 1: Tabular Q-Learning and SARSA
# **Prerequisites:** Read `lecture_notes.md` §1–3 before starting.
#
# **Learning objectives:**
# - Implement tabular Q-learning and SARSA from scratch
# - Understand the off-policy vs on-policy distinction through an ablation
# - Verify convergence on Taxi-v3 and CliffWalking-v0
# - Observe how behavior near cliffs differs between Q-learning and SARSA

# %% [markdown]
# ## Part 1: Theory Recap
#
# **Q-learning update (off-policy):**
#
# $$Q(s,a) \leftarrow Q(s,a) + \alpha\bigl[R + \gamma \max_{a'} Q(s',a') - Q(s,a)\bigr]$$
#
# **SARSA update (on-policy):**
#
# $$Q(s,a) \leftarrow Q(s,a) + \alpha\bigl[R + \gamma Q(s',a') - Q(s,a)\bigr]$$
#
# The key difference: Q-learning targets the *greedy* next action ($\max_{a'}$); SARSA targets
# the *actual* next action $a'$ drawn from the $\epsilon$-greedy behavior policy.
#
# **Why does this matter?** In a dangerous environment (CliffWalking), SARSA accounts for
# exploratory actions that might fall off the cliff. Q-learning ignores them and learns the
# optimal (risky) cliff-edge path.

# %%
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from typing import Optional
from collections import defaultdict

# %% [markdown]
# ## Part 2: Implementation
#
# Implement `TabularQAgent` and `sarsa_update` below.
#
# **Hints:**
# - `Q` is a 2-D numpy array of shape `(n_states, n_actions)`, initialized to zeros.
# - `select_action`: with probability `epsilon` return `np.random.randint(n_actions)`;
#   otherwise return `np.argmax(self.Q[state])`.
# - `update`: compute TD error, update `self.Q[state, action]`, return TD error.
# - `decay_epsilon`: clamp at `epsilon_min`.
# - `sarsa_update`: same pattern but uses `Q[next_state, next_action]` (not max).

# %%
class TabularQAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    def __init__(self, n_states: int, n_actions: int, alpha: float,
                 gamma: float, epsilon: float, epsilon_min: float,
                 epsilon_decay: float):
        """
        Parameters
        ----------
        n_states      : total number of discrete states
        n_actions     : total number of discrete actions
        alpha         : learning rate
        gamma         : discount factor
        epsilon       : initial exploration rate
        epsilon_min   : floor for epsilon after decay
        epsilon_decay : multiplicative decay applied each episode
        """
        # TODO: initialize Q-table as numpy zeros (n_states, n_actions)
        raise NotImplementedError

    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection.

        With probability epsilon: random action.
        Otherwise: argmax Q[state].
        """
        # TODO: with prob epsilon random; else argmax Q[state]
        raise NotImplementedError

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> float:
        """Q-learning update. Returns TD error.

        td_target = reward + gamma * max(Q[next_state]) * (1 - done)
        td_error  = td_target - Q[state, action]
        Q[state, action] += alpha * td_error
        """
        # TODO: td_target = reward + gamma * max(Q[next_state]) * (1 - done)
        # TODO: td_error = td_target - Q[state, action]
        # TODO: Q[state, action] += alpha * td_error
        raise NotImplementedError

    def decay_epsilon(self):
        """Multiply epsilon by epsilon_decay; clamp at epsilon_min."""
        # TODO: epsilon = max(epsilon_min, epsilon * epsilon_decay)
        raise NotImplementedError


def sarsa_update(Q: np.ndarray, state: int, action: int, reward: float,
                 next_state: int, next_action: int,
                 alpha: float, gamma: float, done: bool) -> float:
    """On-policy SARSA update. Returns TD error.

    td_target = reward + gamma * Q[next_state, next_action] * (1 - done)
    td_error  = td_target - Q[state, action]
    Q[state, action] += alpha * td_error

    Note: updates Q in-place.
    """
    # TODO: td_target = reward + gamma * Q[next_state, next_action] * (1 - done)
    # TODO: td_error = td_target - Q[state, action]
    # TODO: Q[state, action] += alpha * td_error
    # TODO: return td_error
    raise NotImplementedError


# %% [markdown]
# ## Part 3: Training Loop Helpers
#
# The training loops below are provided. Read through them to understand the interaction pattern
# before running the verification cells.

# %%
def train_qlearning(env_id: str, n_episodes: int = 10_000,
                    alpha: float = 0.1, gamma: float = 0.99,
                    epsilon: float = 1.0, epsilon_min: float = 0.01,
                    epsilon_decay: float = 0.9995,
                    seed: int = 42) -> tuple[TabularQAgent, list[float]]:
    """Train a TabularQAgent with Q-learning. Returns (agent, episode_rewards)."""
    env = gym.make(env_id)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    agent = TabularQAgent(n_states, n_actions, alpha, gamma,
                          epsilon, epsilon_min, epsilon_decay)
    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
        agent.decay_epsilon()
        episode_rewards.append(total_reward)

    env.close()
    return agent, episode_rewards


def train_sarsa(env_id: str, n_episodes: int = 10_000,
                alpha: float = 0.1, gamma: float = 0.99,
                epsilon: float = 1.0, epsilon_min: float = 0.01,
                epsilon_decay: float = 0.9995,
                seed: int = 42) -> tuple[np.ndarray, list[float]]:
    """Train with SARSA. Returns (Q_table, episode_rewards)."""
    env = gym.make(env_id)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        # Choose first action
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = int(np.argmax(Q[obs]))
        total_reward = 0.0
        done = False
        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # Choose next action (on-policy)
            if np.random.random() < epsilon:
                next_action = np.random.randint(n_actions)
            else:
                next_action = int(np.argmax(Q[next_obs]))
            sarsa_update(Q, obs, action, reward, next_obs, next_action,
                         alpha, gamma, done)
            obs = next_obs
            action = next_action
            total_reward += reward
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

    env.close()
    return Q, episode_rewards


def smooth(rewards: list[float], window: int = 100) -> np.ndarray:
    """Running mean over `window` episodes."""
    return np.convolve(rewards, np.ones(window) / window, mode='valid')


# %% [markdown]
# ## Part 4: Verification — Taxi-v3
#
# Train Q-learning on Taxi-v3. The agent must achieve a mean episode reward > 7 over the last
# 100 episodes within 10,000 episodes.
#
# **Expected behavior:** reward starts around −400 (random policy) and climbs toward +8–9 as the
# agent learns to pick up and drop off passengers efficiently.

# %%
print("Training Q-learning on Taxi-v3 (10k episodes)...")
taxi_agent, taxi_rewards = train_qlearning(
    "Taxi-v3",
    n_episodes=10_000,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.9995,
)

last_100_mean = np.mean(taxi_rewards[-100:])
print(f"Mean reward (last 100 episodes): {last_100_mean:.2f}")

# Verification
assert last_100_mean > 7.0, (
    f"Q-learning on Taxi-v3 did not converge: mean={last_100_mean:.2f} (need > 7.0). "
    "Check your update() implementation."
)
print("✓ Taxi-v3: Q-learning converged (mean reward > 7)")

# %%
# Plot learning curve
plt.figure(figsize=(10, 4))
plt.plot(smooth(taxi_rewards, 100), label="Q-learning (smoothed, w=100)")
plt.axhline(y=7.0, color='red', linestyle='--', label="Target: 7.0")
plt.xlabel("Episode")
plt.ylabel("Episode Reward (smoothed)")
plt.title("Q-Learning on Taxi-v3")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 5: Ablation — Q-Learning vs SARSA on CliffWalking-v0
#
# CliffWalking-v0 has a narrow corridor with a cliff (large negative reward) along the bottom row.
# The optimal path hugs the cliff edge; a safer but longer path exists along the top.
#
# **Prediction (fill in before running):**
# - Q-learning should learn the ___ path because it targets the greedy policy.
# - SARSA should learn the ___ path because it accounts for exploratory actions.

# %%
print("Training Q-learning on CliffWalking-v0...")
cliff_ql_agent, cliff_ql_rewards = train_qlearning(
    "CliffWalking-v0",
    n_episodes=1_000,
    alpha=0.5,
    gamma=0.99,
    epsilon=0.1,
    epsilon_min=0.1,
    epsilon_decay=1.0,  # keep epsilon fixed for clean comparison
)

print("Training SARSA on CliffWalking-v0...")
cliff_sarsa_Q, cliff_sarsa_rewards = train_sarsa(
    "CliffWalking-v0",
    n_episodes=1_000,
    alpha=0.5,
    gamma=0.99,
    epsilon=0.1,
    epsilon_min=0.1,
    epsilon_decay=1.0,
)

# %%
# Plot comparison
plt.figure(figsize=(10, 4))
plt.plot(smooth(cliff_ql_rewards, 50), label="Q-learning (smoothed, w=50)", color="steelblue")
plt.plot(smooth(cliff_sarsa_rewards, 50), label="SARSA (smoothed, w=50)", color="darkorange")
plt.axhline(y=-13, color='green', linestyle='--', label="Optimal cliff-edge: −13")
plt.axhline(y=-17, color='purple', linestyle='--', label="Safe detour: ~−17")
plt.xlabel("Episode")
plt.ylabel("Episode Reward (smoothed)")
plt.title("Q-Learning vs SARSA on CliffWalking-v0")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 6: Observation Questions
#
# Answer the questions below in the markdown cells.
#
# **Q1:** During CliffWalking training with fixed $\epsilon = 0.1$, which algorithm achieves higher
# *average* reward during training? Why?

# %% [markdown]
# **Answer Q1:**
# (fill in)

# %% [markdown]
# **Q2:** If you set $\epsilon = 0$ (pure greedy) for both algorithms, would their learned policies
# differ? What would happen to each algorithm's training behavior?

# %% [markdown]
# **Answer Q2:**
# (fill in)

# %% [markdown]
# **Q3:** Q-learning is off-policy. What advantage does this give when replaying old experience
# (as in DQN)? Why can SARSA not use a replay buffer directly?

# %% [markdown]
# **Answer Q3:**
# (fill in)

# %% [markdown]
# ## Part 7: Reflection
#
# 1. In RLHF, the language model is fine-tuned with a reward model that scores entire responses.
#    Is this more like Monte Carlo or TD learning? What is the "state" and "action" in this context?
#
# 2. Q-overestimation is a known failure mode of tabular Q-learning. Can you construct a simple
#    2-state MDP where Q-learning consistently overestimates one action's value due to initialization?

# %% [markdown]
# **Answers:**
# 1.
# 2.
