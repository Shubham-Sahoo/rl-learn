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
#
# **Before running any code below, you must implement three stubs in `rllearn/`.**
# The DQN agent imports them directly; the notebook will crash if they raise `NotImplementedError`.
#
# ### 1a. `rllearn/buffers.py` → `ReplayBuffer`
#
# ```python
# def __init__(self, capacity: int):
#     self._storage = deque(maxlen=capacity)
#
# def push(self, state, action, reward, next_state, done):
#     self._storage.append((state, action, reward, next_state, done))
#
# def sample(self, batch_size: int) -> tuple:
#     batch = random.sample(self._storage, batch_size)
#     states, actions, rewards, next_states, dones = zip(*batch)
#     return (np.array(states, dtype=np.float32),
#             np.array(actions, dtype=np.int64),
#             np.array(rewards, dtype=np.float32),
#             np.array(next_states, dtype=np.float32),
#             np.array(dones, dtype=np.float32))
#
# def __len__(self): return len(self._storage)
# ```
#
# ### 1b. `rllearn/networks.py` → `MLP`
#
# ```python
# def __init__(self, input_dim, output_dim, hidden_dims=(256, 256), activation=nn.ReLU):
#     super().__init__()
#     layers = []
#     dims = [input_dim] + list(hidden_dims) + [output_dim]
#     for i in range(len(dims) - 1):
#         layers.append(nn.Linear(dims[i], dims[i+1]))
#         if i < len(dims) - 2:
#             layers.append(activation())
#     self.net = nn.Sequential(*layers)
#
# def forward(self, x): return self.net(x)
# ```
#
# ### 1c. `rllearn/envs.py` → `make_env`
#
# ```python
# def make_env(env_id, seed=0, record_video=False, video_folder="videos/"):
#     env = gym.make(env_id, render_mode="rgb_array" if record_video else None)
#     env = gym.wrappers.RecordEpisodeStatistics(env)
#     if record_video:
#         env = gym.wrappers.RecordVideo(env, video_folder)
#     env.reset(seed=seed)
#     return env
# ```
#
# Once you have implemented the stubs, run the cell below to verify they are importable.

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
#
# Key design choices:
# 1. **Experience replay** ($\mathcal{D}$): sample random mini-batches → i.i.d. data for SGD.
# 2. **Target network** ($\theta^-$): frozen copy updated every `target_update_freq` steps →
#    stable regression targets.
#
# Without (1): gradients are highly correlated → oscillation.
# Without (2): targets move every step → non-stationary loss surface → divergence.

# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from tqdm import trange

# %% [markdown]
# ## Part 3: DQN Agent Implementation
#
# The `__init__` method is **provided** (it uses your `rllearn` stubs).
# Implement the three methods marked `TODO`.
#
# **Hints:**
# - `select_action`: use `torch.no_grad()` for the forward pass; use `random.random()` for epsilon.
# - `update`:
#   - Sample from `self.buffer` — it returns numpy arrays; convert to tensors.
#   - `current_q = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)`
#   - `next_q = target_net(next_states).max(1).values` (inside `torch.no_grad()`)
#   - `target_q = rewards + gamma * next_q * (1 - dones)`
#   - `loss = F.mse_loss(current_q, target_q.detach())`
#   - `optimizer.zero_grad(); loss.backward(); optimizer.step()`
#   - Increment `self.n_updates`; call `sync_target()` when `n_updates % target_update_freq == 0`.
# - `sync_target`: `self.target_net.load_state_dict(self.online_net.state_dict())`

# %%
import random


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
        """Epsilon-greedy action selection using online_net.

        With prob epsilon: random action.
        Otherwise: argmax Q(obs) from online_net (use torch.no_grad()).
        """
        # TODO: epsilon-greedy; use torch.no_grad() for forward pass
        raise NotImplementedError

    def store_transition(self, obs, action, reward, next_obs, done):
        """Store a transition in the replay buffer."""
        # Provided
        self.buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> float:
        """Sample from buffer, compute DQN loss, gradient step. Returns loss value.

        Steps:
        1. Return 0.0 if buffer has fewer than batch_size transitions.
        2. Sample batch: states, actions, rewards, next_states, dones = self.buffer.sample(...)
        3. Convert to tensors (float32 for states/rewards/dones, long for actions).
        4. current_q = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        5. with torch.no_grad():
               next_q = target_net(next_states).max(1).values
               target_q = rewards + gamma * next_q * (1 - dones)
        6. loss = F.mse_loss(current_q, target_q)
        7. optimizer.zero_grad(); loss.backward(); optimizer.step()
        8. Increment n_updates; sync target if n_updates % target_update_freq == 0.
        9. Return loss.item()
        """
        if len(self.buffer) < self.batch_size:
            return 0.0
        # TODO: sample batch from self.buffer
        # TODO: current_q = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # TODO: with torch.no_grad(): next_q = target_net(next_states).max(1).values
        # TODO: target_q = rewards + gamma * next_q * (1 - dones)
        # TODO: loss = F.mse_loss(current_q, target_q)
        # TODO: optimizer step; increment n_updates
        # TODO: if n_updates % target_update_freq == 0: sync target network
        raise NotImplementedError

    def sync_target(self):
        """Hard-copy online network weights to target network."""
        # TODO: hard copy online → target
        raise NotImplementedError


# %% [markdown]
# ## Part 4: Training Loop and TensorBoard Logging
#
# The training loop below is provided. It logs four metrics to TensorBoard:
# - `train/episode_reward`: total reward per episode
# - `train/td_loss`: DQN loss per episode
# - `train/q_value_mean`: mean max-Q over last batch
# - `train/epsilon`: current epsilon
#
# To view logs: `tensorboard --logdir runs/` (in a terminal from the repo root)

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

    Logs to TensorBoard under runs/<run_name>.
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
#
# The DQN agent should achieve mean episode reward ≥ 450 over the last 50 episodes.
# CartPole-v1 max is 500 (episode terminates when the pole falls or after 500 steps).

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
#
# ### Ablation A: No Experience Replay (batch_size=1, online training)
#
# Train DQN with `batch_size=1` and `buffer_capacity=1`. This forces the network to train on
# each transition immediately (online TD), with no replay. Observe the training instability.
#
# **Prediction (fill in before running):** Training without replay will be ___ because ___.

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
# (describe what you see — does no-replay training converge? Is it more noisy?)

# %% [markdown]
# ### Ablation B: No Target Network (target_update_freq=1)
#
# Set `target_update_freq=1` so the target network is synced at every update step.
# This is equivalent to having no target network (both networks are always the same).
#
# **Prediction (fill in before running):** Without a target network, training will be ___ because ___.

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
# (describe what you see — does removing the target network cause instability?)

# %% [markdown]
# ## Part 7: Reflection
#
# 1. Q-learning is off-policy. Why is this necessary for experience replay to work correctly?
#    Would SARSA (on-policy) benefit from a replay buffer?
#
# 2. The target network is updated every `C` steps. What happens if `C` is too small? Too large?
#    Is there an optimal `C`?
#
# 3. In offline RL for LLMs (e.g., RLHF without online interaction), the "replay buffer" is the
#    fixed preference dataset. What problems does Q-overestimation cause in this setting, and how
#    does Conservative Q-Learning (CQL) address them?

# %% [markdown]
# **Answers:**
# 1.
# 2.
# 3.
