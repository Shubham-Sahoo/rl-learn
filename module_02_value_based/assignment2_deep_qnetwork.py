# %% [markdown]
# # Assignment 2: Deep Q-Network (DQN)
# **Prerequisites:** Read `lecture_notes.md` §4 and complete Assignment 1.
#
# **Learning objectives:**
# - Implement `ReplayBuffer`, `MLP`, and `make_env` in `rllearn/`
# - Implement DQN's core methods: epsilon-greedy, TD update, target sync
# - Understand why experience replay and target networks are critical
# - Train DQN to solve CartPole-v1 (≥ 450 mean reward in last 50 episodes)

# %% [markdown]
# ## Part 1: Implement `rllearn/` Stubs (Do This First!)

# %%
# Verify rllearn stubs are implemented
from rllearn.buffers import ReplayBuffer
from rllearn.networks import MLP
from rllearn.envs import make_env

buf = ReplayBuffer(100)
import numpy as np
buf.push(np.zeros(4), 0, 1.0, np.zeros(4), False)
assert len(buf) == 1, "ReplayBuffer.__len__ or push broken"

import torch
net = MLP(4, 2)
out = net(torch.zeros(1, 4))
assert out.shape == (1, 2), f"MLP output shape wrong: {out.shape}"

env = make_env("CartPole-v1", seed=0)
obs, _ = env.reset()
assert obs.shape == (4,), f"make_env obs shape wrong: {obs.shape}"
env.close()

print("✓ rllearn stubs are correctly implemented")

# %% [markdown]
# ## Part 2: Theory Recap
#
# **DQN Loss:**
#
# $$\mathcal{L}(\theta) = \mathbb{E}\bigl[(R + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta))^2\bigr]$$

# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from tqdm import trange
import random


# %% [markdown]
# ## Part 3: DQN Agent Implementation

# %%
class DQNAgent:
    """Deep Q-Network agent with experience replay and target network."""

    def __init__(self, obs_dim: int, n_actions: int, lr: float = 1e-3,
                 gamma: float = 0.99, buffer_capacity: int = 10_000,
                 batch_size: int = 64, target_update_freq: int = 100):
        # Provided — uses rllearn stubs (implement those first!)
        self.online_net = MLP(obs_dim, n_actions)
        self.target_net = MLP(obs_dim, n_actions)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.buffer = ReplayBuffer(buffer_capacity)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.n_updates = 0
        self.n_actions = n_actions

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection using online_net."""
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            return int(self.online_net(obs_t).argmax(1).item())

    def store_transition(self, obs, action, reward, next_obs, done):
        """Store a transition in the replay buffer."""
        self.buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> float:
        """Sample from buffer, compute DQN loss, gradient step. Returns loss value."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1).values
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.n_updates += 1
        if self.n_updates % self.target_update_freq == 0:
            self.sync_target()

        return loss.item()

    def sync_target(self):
        """Hard-copy online network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())


# %% [markdown]
# ## Part 4: Training Loop and TensorBoard Logging

# %%
def train_dqn(env_id: str = "CartPole-v1",
              n_episodes: int = 300,
              lr: float = 1e-3,
              gamma: float = 0.99,
              buffer_capacity: int = 10_000,
              batch_size: int = 64,
              target_update_freq: int = 100,
              epsilon_start: float = 1.0,
              epsilon_end: float = 0.01,
              epsilon_decay: float = 0.995,
              seed: int = 42,
              run_name: str = "dqn_cartpole") -> tuple[DQNAgent, list[float]]:
    """
    Train a DQNAgent. Returns (agent, episode_rewards).
    """
    env = make_env(env_id, seed=seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, lr=lr, gamma=gamma,
                     buffer_capacity=buffer_capacity, batch_size=batch_size,
                     target_update_freq=target_update_freq)

    writer = SummaryWriter(f"runs/{run_name}")
    epsilon = epsilon_start
    episode_rewards = []

    for ep in trange(n_episodes, desc=run_name):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        total_loss = 0.0
        steps = 0
        done = False

        while not done:
            action = agent.select_action(obs, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(obs, action, reward, next_obs, float(done))
            loss = agent.update()
            obs = next_obs
            total_reward += reward
            total_loss += loss
            steps += 1

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        # TensorBoard logging
        writer.add_scalar("train/episode_reward", total_reward, ep)
        writer.add_scalar("train/td_loss", total_loss / max(steps, 1), ep)
        writer.add_scalar("train/epsilon", epsilon, ep)

        # Log mean Q-value from a sample batch
        if len(agent.buffer) >= batch_size:
            states_b, _, _, _, _ = agent.buffer.sample(batch_size)
            with torch.no_grad():
                q_vals = agent.online_net(torch.tensor(states_b))
                writer.add_scalar("train/q_value_mean", q_vals.max(1).values.mean().item(), ep)

    writer.close()
    env.close()
    return agent, episode_rewards


# %%
print("Training DQN on CartPole-v1 (300 episodes)...")
dqn_agent, dqn_rewards = train_dqn(
    env_id="CartPole-v1",
    n_episodes=300,
    run_name="dqn_cartpole_baseline",
)

# %% [markdown]
# ## Part 5: Verification

# %%
def smooth(rewards, window=50):
    return np.convolve(rewards, np.ones(window) / window, mode='valid')

last_50_mean = np.mean(dqn_rewards[-50:])
print(f"Mean reward (last 50 episodes): {last_50_mean:.1f}")

assert last_50_mean >= 450, (
    f"DQN did not converge on CartPole-v1: {last_50_mean:.1f} < 450. "
    "Check your update() or sync_target() implementation."
)
print("✓ CartPole-v1: DQN converged (mean reward ≥ 450)")

plt.figure(figsize=(10, 4))
plt.plot(dqn_rewards, alpha=0.3, color="steelblue", label="Per-episode reward")
plt.plot(range(49, len(dqn_rewards)), smooth(dqn_rewards, 50), color="steelblue",
         label="Smoothed (w=50)")
plt.axhline(y=450, color='red', linestyle='--', label="Target: 450")
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("DQN on CartPole-v1")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 6: Ablations

# %%
print("Ablation A: No replay buffer (batch_size=1)...")
_, no_replay_rewards = train_dqn(
    env_id="CartPole-v1",
    n_episodes=300,
    batch_size=1,
    buffer_capacity=1,
    run_name="dqn_cartpole_no_replay",
)

plt.figure(figsize=(10, 4))
plt.plot(range(49, 300), smooth(dqn_rewards, 50), label="Baseline DQN", color="steelblue")
plt.plot(range(49, 300), smooth(no_replay_rewards, 50), label="No replay (batch=1)", color="red")
plt.axhline(y=450, color='gray', linestyle='--', alpha=0.5)
plt.xlabel("Episode")
plt.ylabel("Episode Reward (smoothed)")
plt.title("Ablation A: Effect of Experience Replay")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Observation A (fill in):**
# Without replay, the training is more noisy and may not converge as reliably because consecutive
# transitions are highly correlated, violating the i.i.d. assumption of SGD.

# %%
print("Ablation B: No target network (target_update_freq=1)...")
_, no_target_rewards = train_dqn(
    env_id="CartPole-v1",
    n_episodes=300,
    target_update_freq=1,
    run_name="dqn_cartpole_no_target",
)

plt.figure(figsize=(10, 4))
plt.plot(range(49, 300), smooth(dqn_rewards, 50), label="Baseline DQN", color="steelblue")
plt.plot(range(49, 300), smooth(no_target_rewards, 50),
         label="No target net (freq=1)", color="orange")
plt.axhline(y=450, color='gray', linestyle='--', alpha=0.5)
plt.xlabel("Episode")
plt.ylabel("Episode Reward (smoothed)")
plt.title("Ablation B: Effect of Target Network")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Observation B (fill in):**
# Without a stable target network, the targets shift every gradient step, creating an unstable
# loss landscape that can cause oscillation or divergence.

# %% [markdown]
# ## Part 7: Reflection

# %% [markdown]
# **Answers:**
# 1. Q-learning is off-policy so it learns Q*(s,a) regardless of the behavior policy. Old
#    transitions stored in the replay buffer are still valid targets since Q* doesn't depend on
#    the collection policy. SARSA would need transitions collected under the current policy.
# 2. Too small C: nearly equivalent to no target network — unstable. Too large C: target becomes
#    stale, slowing learning. The optimal C depends on the environment and architecture.
# 3. In offline RLHF, Q-overestimation causes the policy to assign high value to out-of-distribution
#    responses that weren't in the preference dataset. CQL adds a regularizer that penalizes Q-values
#    for unseen (out-of-distribution) actions.
