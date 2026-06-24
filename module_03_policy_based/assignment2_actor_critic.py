# %% [markdown]
# # Assignment 2: Actor-Critic with GAE

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym
from typing import List, Tuple

from rllearn.networks import ActorCritic
from rllearn.utils import compute_gae

# %% [markdown]
# ## Part 2: Implement the One-Step Actor-Critic Update

# %%
def actor_critic_update(ac_net: ActorCritic,
                         optimizer: optim.Optimizer,
                         obs: torch.Tensor,
                         action: int,
                         reward: float,
                         next_obs: torch.Tensor,
                         done: bool,
                         gamma: float) -> Tuple[float, float]:
    """
    One-step A2C update.
    """
    logits, value = ac_net(obs)
    _, next_value = ac_net(next_obs)
    value = value.squeeze()
    next_value = next_value.squeeze()

    delta = reward + gamma * next_value * (1 - int(done)) - value

    action_tensor = torch.tensor(action, dtype=torch.long)
    log_prob = torch.distributions.Categorical(logits=logits).log_prob(action_tensor)

    actor_loss = -log_prob * delta.detach()
    critic_loss = delta ** 2
    loss = actor_loss + critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return actor_loss.item(), critic_loss.item()


# %% [markdown]
# ## Part 3: Training Loop

# %%
def smooth(values: List[float], window: int = 50) -> np.ndarray:
    """Running mean over `window` episodes."""
    return np.convolve(values, np.ones(window) / window, mode='valid')


def train_actor_critic(env_id: str = "LunarLander-v2",
                       n_episodes: int = 1500,
                       gamma: float = 0.99,
                       lr: float = 3e-4,
                       hidden_dim: int = 256,
                       seed: int = 42) -> Tuple[ActorCritic, List[float]]:
    """Train a one-step Actor-Critic agent. Returns (ac_net, episode_rewards)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    ac_net = ActorCritic(obs_dim, n_actions, hidden_dim)
    optimizer = optim.Adam(ac_net.parameters(), lr=lr)

    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs_t = torch.FloatTensor(obs)
        total_reward = 0.0
        done = False

        while not done:
            with torch.no_grad():
                logits, _ = ac_net(obs_t.unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_obs_t = torch.FloatTensor(next_obs)

            actor_loss, critic_loss = actor_critic_update(
                ac_net, optimizer,
                obs_t.unsqueeze(0), action, reward,
                next_obs_t.unsqueeze(0), done, gamma
            )

            obs_t = next_obs_t
            total_reward += reward

        episode_rewards.append(total_reward)

        if (ep + 1) % 100 == 0:
            mean_100 = np.mean(episode_rewards[-100:])
            print(f"Episode {ep+1:4d} | Mean (last 100): {mean_100:.1f}")

    env.close()
    return ac_net, episode_rewards


# %% [markdown]
# ## Part 4: Verification — LunarLander-v2

# %%
print("Training one-step Actor-Critic on LunarLander-v2 (1500 episodes)...")
ac_net, rewards = train_actor_critic(
    env_id="LunarLander-v2",
    n_episodes=1500,
    gamma=0.99,
    lr=3e-4,
    hidden_dim=256,
    seed=42,
)

last_100_mean = float(np.mean(rewards[-100:]))
print(f"\nMean reward (last 100 episodes): {last_100_mean:.1f}")

assert last_100_mean >= 150, (
    f"Actor-Critic on LunarLander-v2 did not converge: mean={last_100_mean:.1f} (need >= 150). "
    "Check ActorCritic.forward, actor_critic_update (detach!), and compute_gae."
)
print("✓ LunarLander-v2: Actor-Critic converged (mean reward >= 150)")

# %%
plt.figure(figsize=(10, 4))
plt.plot(rewards, alpha=0.2, color='steelblue')
plt.plot(np.arange(49, len(rewards)), smooth(rewards, 50),
         color='steelblue', linewidth=2, label='Smoothed (w=50)')
plt.axhline(y=150, color='red', linestyle='--', label='Target: 150')
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("One-Step Actor-Critic on LunarLander-v2")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 5: Ablation — GAE Lambda

# %%
def train_gae_actor_critic(lam: float,
                            env_id: str = "LunarLander-v2",
                            n_episodes: int = 1500,
                            gamma: float = 0.99,
                            lr: float = 3e-4,
                            hidden_dim: int = 256,
                            seed: int = 42) -> List[float]:
    """Train an Actor-Critic with GAE. Returns episode reward list."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    ac_net = ActorCritic(obs_dim, n_actions, hidden_dim)
    optimizer = optim.Adam(ac_net.parameters(), lr=lr)

    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)

        obs_list, action_list, reward_list, done_list, value_list = [], [], [], [], []
        done = False

        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                logits, value = ac_net(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs_list.append(obs_t)
            action_list.append(action)
            reward_list.append(float(reward))
            done_list.append(done)
            value_list.append(value.squeeze().item())

            obs = next_obs

        # Bootstrap value for last state
        last_obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            _, last_value = ac_net(last_obs_t)
        value_list_with_bootstrap = value_list + [last_value.squeeze().item()]

        # Compute GAE advantages
        advantages = compute_gae(reward_list, value_list_with_bootstrap, done_list,
                                  gamma=gamma, lam=lam)
        returns_t = advantages + torch.FloatTensor(value_list)

        # Full-episode update
        obs_batch = torch.cat(obs_list, dim=0)
        action_batch = torch.LongTensor(action_list)

        logits_batch, values_batch = ac_net(obs_batch)
        values_batch = values_batch.squeeze()

        dist_batch = torch.distributions.Categorical(logits=logits_batch)
        log_probs_batch = dist_batch.log_prob(action_batch)

        actor_loss = -(log_probs_batch * advantages.detach()).mean()
        critic_loss = ((values_batch - returns_t.detach()) ** 2).mean()
        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(sum(reward_list))

    env.close()
    return episode_rewards


# %%
lambdas = [0.0, 0.5, 0.95, 1.0]
colors = ['steelblue', 'darkorange', 'green', 'purple']
all_rewards = {}

for lam in lambdas:
    print(f"Training with λ={lam}...")
    r = train_gae_actor_critic(lam=lam, n_episodes=1500, seed=42)
    all_rewards[lam] = r
    last_100 = np.mean(r[-100:])
    print(f"  λ={lam}: mean reward (last 100) = {last_100:.1f}")

# %%
plt.figure(figsize=(12, 5))
for lam, color in zip(lambdas, colors):
    r = all_rewards[lam]
    smoothed = smooth(r, 50)
    plt.plot(np.arange(49, len(r)), smoothed,
             label=f"λ={lam}", color=color, linewidth=2)
plt.axhline(y=150, color='red', linestyle='--', label='Target: 150', alpha=0.7)
plt.xlabel("Episode")
plt.ylabel("Episode Reward (smoothed, w=50)")
plt.title("GAE λ Ablation on LunarLander-v2")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Answer Q1:**
# (fill in based on your results)

# %% [markdown]
# **Answer Q2:**
# At λ=0, early training curves may be slow or erratic because the TD(0) advantage estimate has
# high bias when the critic V is far from the true value function.

# %% [markdown]
# **Answer Q3:**
# With λ=1 (Monte Carlo), a catastrophic crash at the end of an episode propagates large negative
# advantages all the way back to early actions. Actions taken early that were actually good receive
# a very negative gradient signal — this is the credit assignment problem with high variance.

# %% [markdown]
# **Answers:**
# 1. InstructGPT's single terminal reward resembles λ=1 (full Monte Carlo). Credit assignment is
#    hard because the sparse reward provides no per-token signal. The KL penalty acts like a
#    per-token regularizer that helps propagate signal.
# 2. Gradient interference is possible: the critic pushes features toward accurate value prediction,
#    while the actor pushes features toward better action selection. Using separate networks avoids
#    this but is less sample efficient.
