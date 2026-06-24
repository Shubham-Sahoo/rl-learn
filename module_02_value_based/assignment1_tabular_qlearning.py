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

# %%
class TabularQAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    def __init__(self, n_states: int, n_actions: int, alpha: float,
                 gamma: float, epsilon: float, epsilon_min: float,
                 epsilon_decay: float):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> float:
        """Q-learning update. Returns TD error."""
        td_target = reward + self.gamma * np.max(self.Q[next_state]) * (1 - done)
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return float(td_error)

    def decay_epsilon(self):
        """Multiply epsilon by epsilon_decay; clamp at epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def sarsa_update(Q: np.ndarray, state: int, action: int, reward: float,
                 next_state: int, next_action: int,
                 alpha: float, gamma: float, done: bool) -> float:
    """On-policy SARSA update. Returns TD error."""
    td_target = reward + gamma * Q[next_state, next_action] * (1 - done)
    td_error = td_target - Q[state, action]
    Q[state, action] += alpha * td_error
    return float(td_error)


# %% [markdown]
# ## Part 3: Training Loop Helpers

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

# %% [markdown]
# **Answer Q1:**
# SARSA achieves higher average reward during training with fixed epsilon=0.1 because it accounts
# for the exploration policy's tendency to accidentally walk off the cliff, and learns the safer
# path that avoids the cliff edge.

# %% [markdown]
# **Answer Q2:**
# With epsilon=0, both algorithms would learn the same optimal (cliff-edge) policy. Training
# behavior would be identical since there is no exploration to distinguish them.

# %% [markdown]
# **Answer Q3:**
# Q-learning is off-policy: it learns the value of the greedy policy regardless of behavior.
# This means old transitions remain valid targets even if they were collected under a different
# behavior policy. SARSA is on-policy: it learns the value of the behavior policy it is currently
# following. Old transitions from a different policy would have the wrong targets.

# %% [markdown]
# ## Part 7: Reflection

# %% [markdown]
# **Answers:**
# 1. RLHF scoring of full responses resembles Monte Carlo learning — the return G_0 is the reward
#    model score for the entire response. The "state" is the (prompt, partial response) pair and the
#    "action" is generating the next token. High variance arises because many different continuations
#    can follow the same partial response.
# 2. Consider a 2-state MDP with states A and B. From A, two actions: action 0 transitions to B with
#    reward sampled from N(0, 10), action 1 stays at A with reward 0. Q-learning initializes Q(A,0)=0
#    and will consistently overestimate once it observes a positive sample, since max bootstrapping
#    never accounts for future negative samples.
