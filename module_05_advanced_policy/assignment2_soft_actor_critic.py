# %% [markdown]
# # Module 05, Assignment 2: Soft Actor-Critic (SAC)
#
# ## Prerequisites
# Before running Part 2, implement in rllearn/:
# - GaussianPolicyHead in rllearn/networks.py
# - TwinQNetwork in rllearn/networks.py
#
# ## Learning Objectives
# 1. Implement the max-entropy RL objective
# 2. Understand reparameterization trick for continuous actions
# 3. Implement twin Q-network targets
# 4. Implement automatic temperature tuning

# %% [markdown]
# ## Part 0: Theory Recap
#
# SAC optimizes the **maximum entropy objective**:
#
# $$J(\pi) = \mathbb{E}_\tau\!\left[\sum_t\gamma^t\bigl(R(s_t,a_t) + \alpha H(\pi(\cdot|s_t))\bigr)\right]$$
#
# Key components:
# - **Reparameterization:** $a = \tanh(\mu_\phi(s) + \sigma_\phi(s)\odot\varepsilon)$, $\varepsilon\sim\mathcal{N}(0,I)$
# - **Twin critics:** $y = r + \gamma(\min_{i=1,2}Q_{\theta_i^-}(s',\tilde{a}') - \alpha\log\pi(\tilde{a}'|s'))$
# - **Automatic temperature:** $\mathcal{L}(\alpha) = \mathbb{E}[-\alpha\log\pi(a|s) - \alpha\bar{H}]$
#
# ## Part 1: Implement rllearn Stubs
#
# **Before running Part 2, implement the following in `rllearn/networks.py`:**
#
# ### GaussianPolicyHead
# ```python
# class GaussianPolicyHead(nn.Module):
#     LOG_STD_MIN = -5
#     LOG_STD_MAX = 2
#
#     def __init__(self, obs_dim, action_dim, hidden_dim=256):
#         # trunk: Linear(obs_dim, hidden_dim) → ReLU → Linear(hidden_dim, hidden_dim) → ReLU
#         # mean_head: Linear(hidden_dim, action_dim)
#         # log_std_head: Linear(hidden_dim, action_dim)
#
#     def forward(self, obs) -> (mean, log_std):
#         # trunk → mean_head and log_std_head
#         # clamp log_std to [LOG_STD_MIN, LOG_STD_MAX]
#
#     def sample(self, obs) -> (action, log_prob):
#         # mean, log_std = forward(obs)
#         # std = exp(log_std)
#         # eps ~ N(0, I)
#         # x_t = mean + std * eps           (reparameterization)
#         # action = tanh(x_t)               (squashing)
#         # log_prob = N(x_t; mean, std).log_prob - sum(log(1 - action^2 + 1e-6))
# ```
#
# ### TwinQNetwork
# ```python
# class TwinQNetwork(nn.Module):
#     def __init__(self, obs_dim, action_dim, hidden_dim=256):
#         # q1: MLP(obs_dim + action_dim → hidden → hidden → 1)
#         # q2: MLP(obs_dim + action_dim → hidden → hidden → 1)
#
#     def forward(self, obs, action) -> (Q1, Q2):
#         # x = cat([obs, action], dim=-1)
#         # return q1(x).squeeze(-1), q2(x).squeeze(-1)
# ```
#
# Also implement `ReplayBuffer` in `rllearn/buffers.py`:
# ```python
# class ReplayBuffer:
#     def __init__(self, capacity):
#         self._storage = deque(maxlen=capacity)
#
#     def push(self, state, action, reward, next_state, done):
#         self._storage.append((state, action, reward, next_state, done))
#
#     def sample(self, batch_size) -> tuple:
#         batch = random.sample(self._storage, batch_size)
#         s, a, r, ns, d = zip(*batch)
#         return np.array(s), np.array(a), np.array(r, dtype=np.float32), np.array(ns), np.array(d)
#
#     def __len__(self):
#         return len(self._storage)
# ```

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import Tuple

from rllearn.networks import GaussianPolicyHead, TwinQNetwork
from rllearn.buffers import ReplayBuffer
from rllearn.logging import make_writer

# %% [markdown]
# ## Part 2: Implement SAC Loss Functions
#
# Implement the three loss functions below. Each has detailed docstrings explaining
# the expected computation. Read the docstring carefully before implementing.

# %%
def sac_critic_loss(q_net: TwinQNetwork, q_target: TwinQNetwork,
                    policy: GaussianPolicyHead,
                    batch: tuple, gamma: float, log_alpha: torch.Tensor) -> torch.Tensor:
    """
    Twin critic loss.

    y = r + gamma * (min(Q1', Q2')(s', a'_tilde) - alpha * log_pi(a'_tilde|s'))
    L_Q = 0.5 * mean((Q1(s,a) - y)^2 + (Q2(s,a) - y)^2)

    Args:
        q_net: current twin Q-network (being trained)
        q_target: target twin Q-network (slowly updated copy, used for y)
        policy: current actor policy
        batch: tuple of (obs, actions, rewards, next_obs, dones) as torch Tensors
        gamma: discount factor
        log_alpha: log of temperature parameter (learnable)

    Steps:
    1. Unpack batch: obs, actions, rewards, next_obs, dones.
    2. Compute Q1(s,a) and Q2(s,a) from q_net — these are the "predictions."
    3. With torch.no_grad():
       a. Sample a'_tilde, log_pi from policy.sample(next_obs).
       b. Compute Q1_target(s', a'), Q2_target(s', a') from q_target.
       c. alpha = log_alpha.exp().detach()
       d. y = rewards + gamma * (1 - dones) * (min(Q1_target, Q2_target) - alpha * log_pi)
    4. L_Q = 0.5 * mean((Q1 - y)^2 + (Q2 - y)^2)
    5. Return L_Q.

    Common mistake: using q_net (not q_target) for the next_state value →
    creates a feedback loop that causes training instability.
    *(Why target network? It provides a stable regression target. Without it,
    the target y changes with every gradient step, creating a "chasing your tail" problem.)*
    """
    raise NotImplementedError


def sac_actor_loss(policy: GaussianPolicyHead, q_net: TwinQNetwork,
                   obs: torch.Tensor, log_alpha: torch.Tensor) -> torch.Tensor:
    """
    Actor loss: maximize E[min(Q1,Q2)(s, a_tilde) - alpha * log_pi(a_tilde|s)]
    Loss = -mean(min(Q1,Q2) - alpha * log_pi)

    Args:
        policy: current actor policy
        q_net: current twin Q-network (NOT target — we want gradients through Q)
        obs: batch of observations
        log_alpha: log temperature (detached for actor loss)

    Steps:
    1. Sample a_tilde, log_pi from policy.sample(obs).
       *(Why reparameterization? We need gradients to flow through a_tilde into policy params.
       Direct sampling would block gradient flow. The reparameterization trick a = tanh(mu + sigma*eps)
       makes a a deterministic function of (mu, sigma, eps), allowing backprop through a.)*
    2. Compute Q1(s, a_tilde), Q2(s, a_tilde) from q_net (with gradient!).
    3. q_min = min(Q1, Q2).
    4. alpha = log_alpha.exp().detach()
    5. actor_loss = mean(alpha * log_pi - q_min)
    6. Return actor_loss.

    *(Why detach alpha here? We update alpha separately. The actor loss should treat alpha
    as a fixed coefficient, not a variable to optimize through.)*
    """
    raise NotImplementedError


def sac_alpha_loss(log_alpha: torch.Tensor, log_pi: torch.Tensor,
                   target_entropy: float) -> torch.Tensor:
    """
    Automatic temperature tuning.
    L(alpha) = -alpha * (log_pi + target_entropy).detach()
    target_entropy = -action_dim (heuristic)

    Args:
        log_alpha: log of temperature (learnable scalar tensor)
        log_pi: log probabilities of sampled actions, shape (batch,)
        target_entropy: desired entropy level (negative action_dim by convention)

    Steps:
    1. alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
    2. Return alpha_loss.

    *(Why detach log_pi? The temperature update should only adjust alpha to match the
    entropy constraint. We don't want gradients to flow back into the policy through this loss.)*

    *(Why target_entropy = -action_dim? This heuristic (Haarnoja et al. 2018) aims for
    one bit of uncertainty per action dimension, keeping the policy reasonably exploratory
    without being too random.)*
    """
    raise NotImplementedError


# %% [markdown]
# ## Part 3: Training Loop on Pendulum-v1
#
# The training loop below is provided. Pendulum-v1 has:
# - Continuous action space: torque in [-2, 2]
# - Observation: [cos(theta), sin(theta), theta_dot]
# - Reward: -(theta^2 + 0.1*theta_dot^2 + 0.001*torque^2)
# - Target: mean reward >= -200 within 100k environment steps.
#
# (Stretch goal: HalfCheetah-v4 >= 5000 within 1M steps.)
#
# **TensorBoard metrics:**
# - `train/episode_reward`
# - `train/critic_loss`
# - `train/actor_loss`
# - `train/alpha`

# %%
%load_ext tensorboard
%tensorboard --logdir runs/


# %%
def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.005):
    """Polyak averaging: target = tau * source + (1 - tau) * target."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


def train_sac(
    env_id: str = "Pendulum-v1",
    total_steps: int = 100_000,
    batch_size: int = 256,
    buffer_capacity: int = 100_000,
    gamma: float = 0.99,
    tau: float = 0.005,
    lr_actor: float = 3e-4,
    lr_critic: float = 3e-4,
    lr_alpha: float = 3e-4,
    hidden_dim: int = 256,
    learning_starts: int = 5_000,
    update_freq: int = 1,
    seed: int = 42,
):
    """Train SAC on a continuous control environment.

    Returns (episode_rewards, alpha_history).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    target_entropy = -action_dim  # heuristic

    # Networks
    policy = GaussianPolicyHead(obs_dim, action_dim, hidden_dim)
    q_net = TwinQNetwork(obs_dim, action_dim, hidden_dim)
    q_target = TwinQNetwork(obs_dim, action_dim, hidden_dim)
    q_target.load_state_dict(q_net.state_dict())
    for p in q_target.parameters():
        p.requires_grad = False

    # Temperature
    log_alpha = torch.zeros(1, requires_grad=True)

    # Optimizers
    actor_opt = optim.Adam(policy.parameters(), lr=lr_actor)
    critic_opt = optim.Adam(q_net.parameters(), lr=lr_critic)
    alpha_opt = optim.Adam([log_alpha], lr=lr_alpha)

    # Replay buffer
    replay = ReplayBuffer(buffer_capacity)

    writer = make_writer(f"sac_{env_id}")
    episode_rewards = []
    alpha_history = []

    obs, _ = env.reset(seed=seed)
    ep_reward = 0.0
    ep_step = 0

    for step in range(total_steps):
        # Action selection
        if step < learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action, _ = policy.sample(torch.FloatTensor(obs).unsqueeze(0))
                action = action.squeeze(0).numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        ep_step += 1

        replay.push(obs, action, reward, next_obs, float(terminated))

        if done or truncated:
            episode_rewards.append(ep_reward)
            writer.add_scalar("train/episode_reward", ep_reward, step)
            if len(episode_rewards) % 10 == 0:
                mean_10 = np.mean(episode_rewards[-10:])
                print(f"Step {step:7d} | Episodes: {len(episode_rewards):4d} | "
                      f"Mean reward (last 10): {mean_10:.1f} | "
                      f"alpha: {log_alpha.exp().item():.4f}")
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_step = 0
        else:
            obs = next_obs

        # Training updates
        if step >= learning_starts and len(replay) >= batch_size and step % update_freq == 0:
            s, a, r, ns, d = replay.sample(batch_size)
            obs_t = torch.FloatTensor(s)
            act_t = torch.FloatTensor(a)
            rew_t = torch.FloatTensor(r)
            next_obs_t = torch.FloatTensor(ns)
            done_t = torch.FloatTensor(d)

            batch = (obs_t, act_t, rew_t, next_obs_t, done_t)

            # Critic update
            critic_opt.zero_grad()
            c_loss = sac_critic_loss(q_net, q_target, policy, batch, gamma, log_alpha)
            c_loss.backward()
            critic_opt.step()

            # Actor update
            actor_opt.zero_grad()
            a_tilde, log_pi = policy.sample(obs_t)
            a_loss = sac_actor_loss(policy, q_net, obs_t, log_alpha)
            a_loss.backward()
            actor_opt.step()

            # Alpha update
            alpha_opt.zero_grad()
            al_loss = sac_alpha_loss(log_alpha, log_pi.detach(), target_entropy)
            al_loss.backward()
            alpha_opt.step()

            # Target network update
            soft_update(q_target, q_net, tau)

            alpha_val = log_alpha.exp().item()
            alpha_history.append(alpha_val)
            writer.add_scalar("train/critic_loss", c_loss.item(), step)
            writer.add_scalar("train/actor_loss", a_loss.item(), step)
            writer.add_scalar("train/alpha", alpha_val, step)

    env.close()
    writer.close()
    return episode_rewards, alpha_history


# %%
print("Training SAC on Pendulum-v1 (100k steps)...")
rewards, alphas = train_sac(env_id="Pendulum-v1", total_steps=100_000, seed=42)

# Verification
if len(rewards) >= 10:
    last_10_mean = float(np.mean(rewards[-10:]))
    print(f"\nMean reward (last 10 episodes): {last_10_mean:.1f}")
    if last_10_mean >= -200:
        print("✓ Pendulum-v1: SAC converged (mean reward >= -200)")
    else:
        print(f"✗ Pendulum-v1: not yet converged (need >= -200). Check implementations.")

# %% [markdown]
# ## Part 4: Ablation — Fixed vs Learned Temperature
#
# Compare three settings on Pendulum-v1:
# - **Fixed alpha=0.2** (high entropy, more exploration)
# - **Fixed alpha=0.01** (low entropy, more exploitation)
# - **Learned alpha** (automatic tuning)
#
# Expected: Learned alpha adapts to the task, showing competitive or better sample efficiency.

# %%
def train_sac_fixed_alpha(alpha_value: float, total_steps: int = 100_000, seed: int = 42):
    """Train SAC with a fixed alpha value (no temperature learning)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = GaussianPolicyHead(obs_dim, action_dim, hidden_dim=256)
    q_net = TwinQNetwork(obs_dim, action_dim, hidden_dim=256)
    q_target = TwinQNetwork(obs_dim, action_dim, hidden_dim=256)
    q_target.load_state_dict(q_net.state_dict())
    for p in q_target.parameters():
        p.requires_grad = False

    log_alpha = torch.log(torch.tensor(alpha_value))  # fixed, no grad
    actor_opt = optim.Adam(policy.parameters(), lr=3e-4)
    critic_opt = optim.Adam(q_net.parameters(), lr=3e-4)
    replay = ReplayBuffer(100_000)

    episode_rewards = []
    obs, _ = env.reset(seed=seed)
    ep_reward = 0.0

    for step in range(total_steps):
        if step < 5000:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action, _ = policy.sample(torch.FloatTensor(obs).unsqueeze(0))
                action = action.squeeze(0).numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        replay.push(obs, action, reward, next_obs, float(terminated))

        if done or truncated:
            episode_rewards.append(ep_reward)
            obs, _ = env.reset()
            ep_reward = 0.0
        else:
            obs = next_obs

        if step >= 5000 and len(replay) >= 256 and step % 1 == 0:
            s, a, r, ns, d = replay.sample(256)
            obs_t = torch.FloatTensor(s)
            act_t = torch.FloatTensor(a)
            rew_t = torch.FloatTensor(r)
            next_obs_t = torch.FloatTensor(ns)
            done_t = torch.FloatTensor(d)
            batch = (obs_t, act_t, rew_t, next_obs_t, done_t)

            critic_opt.zero_grad()
            c_loss = sac_critic_loss(q_net, q_target, policy, batch, 0.99, log_alpha)
            c_loss.backward()
            critic_opt.step()

            actor_opt.zero_grad()
            a_loss = sac_actor_loss(policy, q_net, obs_t, log_alpha)
            a_loss.backward()
            actor_opt.step()

            soft_update(q_target, q_net, tau=0.005)

    env.close()
    return episode_rewards


# %%
print("Running alpha ablation...")
print("Training with fixed alpha=0.2...")
rewards_02 = train_sac_fixed_alpha(alpha_value=0.2, total_steps=100_000, seed=42)

print("Training with fixed alpha=0.01...")
rewards_001 = train_sac_fixed_alpha(alpha_value=0.01, total_steps=100_000, seed=42)

print("Using learned alpha results from above...")
rewards_learned = rewards  # from train_sac above

# %%
def smooth(values, window=10):
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10, 4))
plt.plot(smooth(rewards_02, 10), label="Fixed alpha=0.2", color='darkorange')
plt.plot(smooth(rewards_001, 10), label="Fixed alpha=0.01", color='green')
plt.plot(smooth(rewards_learned, 10), label="Learned alpha", color='steelblue')
plt.axhline(y=-200, color='red', linestyle='--', label="Target: -200")
plt.xlabel("Episode")
plt.ylabel("Episode Reward (smoothed)")
plt.title("SAC Temperature Ablation on Pendulum-v1")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 5: Reflection
#
# **Q1:** How does the entropy term $\alpha H(\pi(\cdot|s))$ relate to LLM temperature sampling?
# In RLHF, a KL penalty term $\beta \text{KL}(\pi \| \pi_\text{ref})$ is often added to the
# objective. What is the relationship between $\alpha$, $\beta$, and the LLM sampling temperature?

# %% [markdown]
# **Answer Q1:**
# (fill in)

# %% [markdown]
# **Q2:** Why twin critics instead of one? What specific failure mode does a single critic
# exhibit, and why does taking the minimum of two critics reduce this failure mode?
# (Hint: think about what a greedy policy does when one Q-network overestimates a particular action.)

# %% [markdown]
# **Answer Q2:**
# (fill in)

# %% [markdown]
# **Q3:** SAC uses the reparameterization trick: $a = \tanh(\mu_\phi(s) + \sigma_\phi(s) \odot \varepsilon)$.
# Why can't we use `a ~ Categorical(logits=policy(s))` (a direct sample) and then call `.backward()`
# on the actor loss? What does reparameterization do differently that enables gradient flow?

# %% [markdown]
# **Answer Q3:**
# (fill in)
