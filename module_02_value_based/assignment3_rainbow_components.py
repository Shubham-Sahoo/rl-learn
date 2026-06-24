# %% [markdown]
# # Assignment 3: Rainbow Components — Double DQN, Dueling DQN, PER
# **Prerequisites:** Read `lecture_notes.md` §5–7 and complete Assignment 2.

# %%
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from rllearn.networks import MLP, DuelingNet
from rllearn.buffers import ReplayBuffer, PrioritizedReplayBuffer
from rllearn.envs import make_env

# Verify DuelingNet
net = DuelingNet(obs_dim=8, n_actions=4)
out = net(torch.zeros(2, 8))
assert out.shape == (2, 4), f"DuelingNet output shape wrong: {out.shape}"
print(f"DuelingNet output (batch=2, n_actions=4): {out.shape} ✓")

# Verify PrioritizedReplayBuffer
per_buf = PrioritizedReplayBuffer(capacity=1000, alpha=0.6, beta_start=0.4)
for i in range(200):
    per_buf.push(np.zeros(8), 0, 1.0, np.zeros(8), False, error=float(i) * 0.01)
batch, weights, indices = per_buf.sample(32)
assert len(batch) == 5, "PER sample should return (states, actions, rewards, next_states, dones)"
assert len(weights) == 32, "IS weights shape mismatch"
per_buf.update_priorities(indices, np.ones(32) * 0.5)
print(f"PrioritizedReplayBuffer: sample shape OK, len={len(per_buf)} ✓")

# %% [markdown]
# ## Part 3: Double DQN Target

# %%
def double_dqn_target(online_net: torch.nn.Module,
                      target_net: torch.nn.Module,
                      next_states: torch.Tensor,
                      rewards: torch.Tensor,
                      dones: torch.Tensor,
                      gamma: float) -> torch.Tensor:
    """
    Compute Double DQN targets.

    y = rewards + gamma * Q(s', argmax_{a'} Q(s',a'; theta); theta_minus) * (1 - dones)
    """
    with torch.no_grad():
        # Online net selects action
        action_selection = online_net(next_states).argmax(1)
        # Target net evaluates it
        next_q = target_net(next_states).gather(1, action_selection.unsqueeze(1)).squeeze(1)
        return rewards + gamma * next_q * (1 - dones)


# Quick sanity check
with torch.no_grad():
    online = MLP(4, 2)
    target = MLP(4, 2)
    ns = torch.zeros(8, 4)
    r = torch.ones(8)
    d = torch.zeros(8)
    out = double_dqn_target(online, target, ns, r, d, gamma=0.99)
    assert out.shape == (8,), f"double_dqn_target shape wrong: {out.shape}"
print("double_dqn_target: shape OK ✓")

# %% [markdown]
# ## Part 4: Agent Variants

# %%
class DoubleDQNAgent:
    """DQN agent using Double DQN target (fixes overestimation bias)."""

    def __init__(self, obs_dim: int, n_actions: int, lr: float = 1e-3,
                 gamma: float = 0.99, buffer_capacity: int = 50_000,
                 batch_size: int = 64, target_update_freq: int = 200):
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
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            return int(self.online_net(t).argmax(1).item())

    def store_transition(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> float:
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
            targets = double_dqn_target(
                self.online_net, self.target_net, next_states, rewards, dones, self.gamma
            )
        loss = F.mse_loss(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.n_updates += 1
        if self.n_updates % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return loss.item()


class DuelingDQNAgent:
    """DQN agent using Dueling network architecture."""

    def __init__(self, obs_dim: int, n_actions: int, lr: float = 1e-3,
                 gamma: float = 0.99, buffer_capacity: int = 50_000,
                 batch_size: int = 64, target_update_freq: int = 200):
        self.online_net = DuelingNet(obs_dim, n_actions)
        self.target_net = DuelingNet(obs_dim, n_actions)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.buffer = ReplayBuffer(buffer_capacity)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.n_updates = 0
        self.n_actions = n_actions

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            return int(self.online_net(t).argmax(1).item())

    def store_transition(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> float:
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
            self.target_net.load_state_dict(self.online_net.state_dict())
        return loss.item()


# %% [markdown]
# ## Part 5: Training Loop

# %%
def train_agent(agent, env_id: str = "LunarLander-v2",
                n_episodes: int = 600,
                epsilon_start: float = 1.0,
                epsilon_end: float = 0.01,
                epsilon_decay: float = 0.997,
                seed: int = 42,
                run_name: str = "agent") -> list[float]:
    """Train any DQN-style agent. Returns episode_rewards list."""
    env = make_env(env_id, seed=seed)
    writer = SummaryWriter(f"runs/{run_name}")
    epsilon = epsilon_start
    episode_rewards = []

    for ep in trange(n_episodes, desc=run_name):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(obs, action, reward, next_obs, float(done))
            agent.update()
            obs = next_obs
            total_reward += reward

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
        writer.add_scalar("train/episode_reward", total_reward, ep)
        writer.add_scalar("train/epsilon", epsilon, ep)

    writer.close()
    env.close()
    return episode_rewards


# %% [markdown]
# ## Part 6: Run All Three Variants

# %%
env_id = "LunarLander-v2"
obs_dim = gym.make(env_id).observation_space.shape[0]
n_actions = gym.make(env_id).action_space.n


class VanillaDQNAgent:
    """Vanilla DQN (for comparison) — same as Assignment 2."""

    def __init__(self, obs_dim, n_actions, lr=1e-3, gamma=0.99,
                 buffer_capacity=50_000, batch_size=64, target_update_freq=200):
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

    def select_action(self, obs, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            return int(self.online_net(t).argmax(1).item())

    def store_transition(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    def update(self):
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
            self.target_net.load_state_dict(self.online_net.state_dict())
        return loss.item()


# %%
print("Running Vanilla DQN on LunarLander-v2...")
vanilla_agent = VanillaDQNAgent(obs_dim, n_actions)
vanilla_rewards = train_agent(vanilla_agent, run_name="lunar_vanilla_dqn")

print("Running Double DQN on LunarLander-v2...")
double_agent = DoubleDQNAgent(obs_dim, n_actions)
double_rewards = train_agent(double_agent, run_name="lunar_double_dqn")

print("Running Dueling DQN on LunarLander-v2...")
dueling_agent = DuelingDQNAgent(obs_dim, n_actions)
dueling_rewards = train_agent(dueling_agent, run_name="lunar_dueling_dqn")

# %% [markdown]
# ## Part 7: Verification

# %%
def smooth(rewards, window=100):
    return np.convolve(rewards, np.ones(window) / window, mode='valid')

last_100_vanilla = np.mean(vanilla_rewards[-100:])
last_100_double = np.mean(double_rewards[-100:])
last_100_dueling = np.mean(dueling_rewards[-100:])

print(f"Vanilla DQN  (last 100): {last_100_vanilla:.1f}")
print(f"Double DQN   (last 100): {last_100_double:.1f}")
print(f"Dueling DQN  (last 100): {last_100_dueling:.1f}")

best_score = max(last_100_vanilla, last_100_double, last_100_dueling)
assert best_score >= 200, (
    f"No variant reached ≥ 200 on LunarLander-v2. Best: {best_score:.1f}. "
    "Check your double_dqn_target or DuelingNet implementation."
)
print(f"✓ LunarLander-v2: at least one variant solved (best={best_score:.1f} ≥ 200)")

# %%
plt.figure(figsize=(12, 5))
n = len(vanilla_rewards)
w = 100
x = range(w - 1, n)
plt.plot(x, smooth(vanilla_rewards, w), label="Vanilla DQN", color="steelblue")
plt.plot(x, smooth(double_rewards, w), label="Double DQN", color="darkorange")
plt.plot(x, smooth(dueling_rewards, w), label="Dueling DQN", color="green")
plt.axhline(y=200, color='red', linestyle='--', label="Solved: 200")
plt.xlabel("Episode")
plt.ylabel("Episode Reward (smoothed, w=100)")
plt.title("Vanilla DQN vs Double DQN vs Dueling DQN — LunarLander-v2")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 8: Observations

# %% [markdown]
# **Answer Q1:**
# (fill in)

# %% [markdown]
# **Answer Q2:**
# (fill in)

# %% [markdown]
# **Answer Q3:**
# IS correction is necessary for convergence because prioritized sampling introduces bias —
# high-priority transitions are sampled more often than their true frequency. Without IS weights,
# the gradient update is biased toward high-error transitions. Omitting IS weights while keeping
# prioritized sampling can cause overemphasis of high-error (often noisy) transitions.

# %% [markdown]
# ## Part 9: Reflection

# %% [markdown]
# **Answers:**
# 1. Double DQN: overestimation bias. Dueling: inefficient state-value estimation.
#    PER: uniform sampling ignores transition importance. n-step returns: myopic 1-step targets.
#    Noisy Networks: epsilon-greedy exploration is undirected. Distributional RL: ignores value distribution.
# 2. Use two separate models: one to select the action (the policy model currently trained) and
#    another frozen "target" model to evaluate it. This mirrors Double DQN's separation of selection
#    and evaluation to reduce overestimation of out-of-distribution responses.
