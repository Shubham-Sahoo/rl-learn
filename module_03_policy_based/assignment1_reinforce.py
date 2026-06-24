# %% [markdown]
# # Assignment 1: REINFORCE — Monte Carlo Policy Gradient

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym
from typing import List, Tuple

from rllearn.utils import compute_returns

# %% [markdown]
# ## Part 2: Implement the Policy Network

# %%
class PolicyNet(nn.Module):
    """Softmax policy network for discrete actions."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return action logits of shape (batch, n_actions)."""
        return self.net(x)


# %% [markdown]
# ## Part 3: Implement Action Selection

# %%
def select_action(policy_net: PolicyNet, obs: np.ndarray) -> Tuple[int, torch.Tensor]:
    """Sample action from policy; return (action, log_prob)."""
    obs_t = torch.FloatTensor(obs)
    logits = policy_net(obs_t)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob


# %% [markdown]
# ## Part 4: Implement the REINFORCE Update

# %%
def reinforce_update(optimizer: optim.Optimizer,
                     log_probs: List[torch.Tensor],
                     returns: List[float]) -> float:
    """REINFORCE gradient update."""
    returns_t = torch.FloatTensor(returns)
    # Normalize returns
    returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
    log_probs_t = torch.stack(log_probs)
    loss = -torch.sum(log_probs_t * returns_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# %% [markdown]
# ## Part 5: Training Loop

# %%
def smooth(values: List[float], window: int = 50) -> np.ndarray:
    """Running mean over `window` episodes."""
    return np.convolve(values, np.ones(window) / window, mode='valid')


def train_reinforce(env_id: str = "CartPole-v1",
                    n_episodes: int = 500,
                    gamma: float = 0.99,
                    lr: float = 1e-3,
                    hidden_dim: int = 128,
                    seed: int = 42) -> Tuple[PolicyNet, List[float]]:
    """Train a REINFORCE agent. Returns (policy, episode_rewards)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PolicyNet(obs_dim, n_actions, hidden_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []
        done = False

        while not done:
            action, log_prob = select_action(policy, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            log_probs.append(log_prob)
            rewards.append(float(reward))

        returns = compute_returns(rewards, gamma)
        loss = reinforce_update(optimizer, log_probs, returns)
        episode_rewards.append(sum(rewards))

        if (ep + 1) % 50 == 0:
            mean_50 = np.mean(episode_rewards[-50:])
            print(f"Episode {ep+1:4d} | Mean (last 50): {mean_50:.1f} | Loss: {loss:.4f}")

    env.close()
    return policy, episode_rewards


# %% [markdown]
# ## Part 6: Verification — CartPole-v1

# %%
print("Training REINFORCE on CartPole-v1 (500 episodes)...")
policy, rewards = train_reinforce(
    env_id="CartPole-v1",
    n_episodes=500,
    gamma=0.99,
    lr=1e-3,
    hidden_dim=128,
    seed=42,
)

last_50_mean = float(np.mean(rewards[-50:]))
print(f"\nMean reward (last 50 episodes): {last_50_mean:.1f}")

assert last_50_mean >= 450, (
    f"REINFORCE on CartPole-v1 did not converge: mean={last_50_mean:.1f} (need >= 450). "
    "Check PolicyNet, select_action, reinforce_update, and compute_returns."
)
print("✓ CartPole-v1: REINFORCE converged (mean reward >= 450)")

# %%
plt.figure(figsize=(10, 4))
plt.plot(rewards, alpha=0.3, color='steelblue', label='Episode reward')
plt.plot(np.arange(49, len(rewards)), smooth(rewards, 50),
         color='steelblue', linewidth=2, label='Smoothed (w=50)')
plt.axhline(y=450, color='red', linestyle='--', label='Target: 450')
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("REINFORCE on CartPole-v1")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 7: Ablation — Return Normalization

# %%
def reinforce_update_no_norm(optimizer: optim.Optimizer,
                              log_probs: List[torch.Tensor],
                              returns: List[float]) -> Tuple[float, float]:
    """REINFORCE update WITHOUT return normalization."""
    returns_t = torch.FloatTensor(returns)
    log_probs_t = torch.stack(log_probs)
    loss = -torch.sum(log_probs_t * returns_t)

    optimizer.zero_grad()
    loss.backward()

    grad_norm = 0.0
    for p in optimizer.param_groups[0]['params']:
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5

    optimizer.step()
    return loss.item(), grad_norm


def reinforce_update_with_norm(optimizer: optim.Optimizer,
                                log_probs: List[torch.Tensor],
                                returns: List[float]) -> Tuple[float, float]:
    """REINFORCE update WITH return normalization. Returns (loss_value, grad_norm)."""
    returns_t = torch.FloatTensor(returns)
    returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
    log_probs_t = torch.stack(log_probs)
    loss = -torch.sum(log_probs_t * returns_t)

    optimizer.zero_grad()
    loss.backward()

    grad_norm = 0.0
    for p in optimizer.param_groups[0]['params']:
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5

    optimizer.step()
    return loss.item(), grad_norm


def train_ablation(use_norm: bool, n_episodes: int = 300, seed: int = 42):
    """Train REINFORCE with or without return normalization. Returns (rewards, grad_norms)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PolicyNet(obs_dim, n_actions, hidden_dim=128)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    update_fn = reinforce_update_with_norm if use_norm else reinforce_update_no_norm

    episode_rewards, grad_norms = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        log_probs_ep, rewards_ep = [], []
        done = False
        while not done:
            action, lp = select_action(policy, obs)
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            log_probs_ep.append(lp)
            rewards_ep.append(float(r))

        returns = compute_returns(rewards_ep, gamma=0.99)
        _, gn = update_fn(optimizer, log_probs_ep, returns)
        episode_rewards.append(sum(rewards_ep))
        grad_norms.append(gn)

    env.close()
    return episode_rewards, grad_norms


# %%
print("Training REINFORCE with return normalization...")
rewards_norm, gnorms_norm = train_ablation(use_norm=True, n_episodes=300)

print("Training REINFORCE without return normalization...")
rewards_no_norm, gnorms_no_norm = train_ablation(use_norm=False, n_episodes=300)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(smooth(rewards_norm, 20), label="With normalization", color='steelblue')
axes[0].plot(smooth(rewards_no_norm, 20), label="Without normalization", color='darkorange')
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Episode Reward (smoothed)")
axes[0].set_title("Return Normalization: Episode Rewards")
axes[0].legend()

axes[1].plot(smooth(gnorms_norm, 20), label="With normalization", color='steelblue')
axes[1].plot(smooth(gnorms_no_norm, 20), label="Without normalization", color='darkorange')
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Gradient Norm (smoothed)")
axes[1].set_title("Return Normalization: Gradient Norm Variance")
axes[1].legend()

plt.tight_layout()
plt.show()

norm_gn_std = float(np.std(gnorms_norm))
no_norm_gn_std = float(np.std(gnorms_no_norm))
print(f"Gradient norm std  (normalized): {norm_gn_std:.4f}")
print(f"Gradient norm std (unnormalized): {no_norm_gn_std:.4f}")
print(f"Ratio (unnorm / norm): {no_norm_gn_std / (norm_gn_std + 1e-8):.2f}x higher variance")

# %% [markdown]
# **Answer Q1:**
# Without normalization, returns can be large positive values in long successful episodes,
# causing large gradient magnitudes. Normalization removes the scale of returns, reducing
# the variance in gradient norms.

# %% [markdown]
# **Answer Q2:**
# Normalization is not a baseline in the theoretical sense. A true baseline b(s) is state-dependent
# and is subtracted before multiplication: E[log pi * (G - b)] = E[log pi * G] because
# E[log pi * b] = 0. Normalization changes the *scale* of each episode's gradients but is not
# statistically unbiased in the same sense.

# %% [markdown]
# **Answer Q3:**
# With very long episodes, REINFORCE must wait for the entire episode to complete before any
# update. This gives extremely delayed feedback and very high variance estimates of G_t for
# early timesteps. Actor-Critic addresses this by using a bootstrapped TD estimate V(s_{t+1})
# to provide a low-variance advantage estimate at every step.

# %% [markdown]
# **Answers:**
# 1. In RLHF, G_0 = reward model score for the full response. This is analogous to REINFORCE's
#    Monte Carlo estimate — a single scalar for the entire trajectory. High variance arises from
#    the diversity of possible continuations and reward model noise.
# 2. Causality: future actions don't affect past rewards. Using G_0 at all timesteps would include
#    past rewards that cannot be influenced by action a_t, adding irrelevant noise to the gradient.
