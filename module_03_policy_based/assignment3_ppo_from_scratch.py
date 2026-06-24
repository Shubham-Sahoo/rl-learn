# %% [markdown]
# # Assignment 3: PPO from Scratch

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym
from typing import Dict, List, Tuple
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from rllearn.networks import ActorCritic
from rllearn.buffers import RolloutBuffer
from rllearn.envs import NormalizeObsWrapper
from rllearn.utils import compute_gae, explained_variance

# %% [markdown]
# ## Part 2: Implement PPO-Clip Loss

# %%
def compute_ppo_loss(log_probs_new: torch.Tensor,
                      log_probs_old: torch.Tensor,
                      advantages: torch.Tensor,
                      clip_eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """PPO-Clip policy loss."""
    ratio = torch.exp(log_probs_new - log_probs_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    loss = -torch.mean(torch.min(surr1, surr2))
    clip_fraction = torch.mean(((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float())
    return loss, clip_fraction


# %% [markdown]
# ## Part 3: Implement the Full PPO Update

# %%
def ppo_update(actor_critic: ActorCritic,
               optimizer: optim.Optimizer,
               rollout_buffer: RolloutBuffer,
               advantages: torch.Tensor,
               clip_eps: float = 0.2,
               c1: float = 0.5,
               c2: float = 0.01,
               n_epochs: int = 10,
               batch_size: int = 64) -> dict:
    """Full PPO update over n_epochs of minibatch gradient steps."""
    data = rollout_buffer.get()
    obs = data["obs"]
    actions = data["actions"]
    values = data["values"]
    log_probs_old = data["log_probs"]

    # Returns = advantages + values (before normalization)
    returns = advantages + values

    # Normalize advantages
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n_samples = len(obs)

    # Accumulators
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_clip_frac = 0.0
    n_updates = 0

    for epoch in range(n_epochs):
        indices = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            mb_idx = indices[start:start + batch_size]
            obs_mb = obs[mb_idx]
            actions_mb = actions[mb_idx]
            adv_mb = adv[mb_idx]
            returns_mb = returns[mb_idx]
            log_probs_old_mb = log_probs_old[mb_idx]

            logits, values_new = actor_critic(obs_mb)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs_new = dist.log_prob(actions_mb)
            entropy = dist.entropy().mean()

            policy_loss, clip_frac = compute_ppo_loss(
                log_probs_new, log_probs_old_mb, adv_mb, clip_eps)
            value_loss = 0.5 * ((values_new.squeeze() - returns_mb) ** 2).mean()
            total_loss = policy_loss + c1 * value_loss - c2 * entropy

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_clip_frac += clip_frac.item()
            n_updates += 1

    # Compute explained variance
    with torch.no_grad():
        ev = explained_variance(values.numpy(), returns.numpy())

    return {
        "policy_loss": total_policy_loss / max(n_updates, 1),
        "value_loss": total_value_loss / max(n_updates, 1),
        "entropy_loss": total_entropy / max(n_updates, 1),
        "clip_fraction": total_clip_frac / max(n_updates, 1),
        "explained_variance": ev,
    }


# %% [markdown]
# ## Part 4: Training Loop

# %%
def make_ppo_env(env_id: str, seed: int = 0) -> gym.Env:
    """Create normalized environment for PPO."""
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NormalizeObsWrapper(env)
    env.reset(seed=seed)
    return env


def train_ppo(env_id: str = "HalfCheetah-v4",
              total_steps: int = 3_000_000,
              rollout_steps: int = 2048,
              gamma: float = 0.99,
              gae_lambda: float = 0.95,
              lr: float = 3e-4,
              clip_eps: float = 0.2,
              c1: float = 0.5,
              c2: float = 0.01,
              n_epochs: int = 10,
              batch_size: int = 64,
              hidden_dim: int = 256,
              seed: int = 42,
              log_dir: str = "runs/ppo") -> Tuple[ActorCritic, List[float]]:
    """Full PPO training loop."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_ppo_env(env_id, seed=seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n if hasattr(env.action_space, 'n') else None

    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if not is_discrete:
        print(f"[WARNING] {env_id} has continuous actions. Using placeholder discrete head.")
        print("For full continuous PPO, see Module 05 (SAC/PPO with GaussianPolicyHead).")
        print("Re-running with LunarLander-v2 (discrete) for this assignment.")
        env.close()
        env_id = "LunarLander-v2"
        env = make_ppo_env(env_id, seed=seed)
        obs_dim = env.observation_space.shape[0]

    n_actions = env.action_space.n
    actor_critic = ActorCritic(obs_dim, n_actions, hidden_dim)
    optimizer = optim.Adam(actor_critic.parameters(), lr=lr)

    writer = None
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=log_dir)

    buffer = RolloutBuffer()
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []

    obs, _ = env.reset(seed=seed)
    global_step = 0
    n_updates = 0

    while global_step < total_steps:
        buffer.clear()
        rollout_episode_rewards: List[float] = []

        for _ in range(rollout_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                logits, value = actor_critic(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            buffer.add(obs, action.item(), reward, done, value.squeeze().item(),
                       log_prob.item())

            obs = next_obs
            global_step += 1

            if done:
                if "episode" in info:
                    ep_reward = info["episode"]["r"]
                    ep_len = info["episode"]["l"]
                    episode_rewards.append(float(ep_reward))
                    episode_lengths.append(int(ep_len))
                    rollout_episode_rewards.append(float(ep_reward))
                obs, _ = env.reset()

        # Compute GAE with bootstrap value from the last obs
        data = buffer.get()
        last_obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            _, last_value = actor_critic(last_obs_t)

        rewards_list = data["rewards"].tolist()
        values_list = data["values"].tolist() + [last_value.squeeze().item()]
        dones_list = data["dones"].tolist()

        advantages = compute_gae(rewards_list, values_list, dones_list,
                                  gamma=gamma, lam=gae_lambda)

        metrics = ppo_update(actor_critic, optimizer, buffer, advantages,
                              clip_eps=clip_eps, c1=c1, c2=c2,
                              n_epochs=n_epochs, batch_size=batch_size)

        n_updates += 1

        if writer is not None:
            writer.add_scalar("train/policy_loss", metrics["policy_loss"], global_step)
            writer.add_scalar("train/value_loss", metrics["value_loss"], global_step)
            writer.add_scalar("train/entropy", metrics["entropy_loss"], global_step)
            writer.add_scalar("train/clip_fraction", metrics["clip_fraction"], global_step)
            writer.add_scalar("train/explained_variance", metrics["explained_variance"],
                               global_step)
            if rollout_episode_rewards:
                writer.add_scalar("rollout/mean_reward",
                                   np.mean(rollout_episode_rewards), global_step)

        if n_updates % 10 == 0 and episode_rewards:
            mean_100 = np.mean(episode_rewards[-100:])
            print(f"Step {global_step:8d} | Updates {n_updates:4d} | "
                  f"Mean reward (100 ep): {mean_100:.1f} | "
                  f"Clip frac: {metrics['clip_fraction']:.3f} | "
                  f"EV: {metrics['explained_variance']:.3f}")

    if writer is not None:
        writer.close()
    env.close()
    return actor_critic, episode_rewards


# %% [markdown]
# ## Part 5: Verification

# %%
print("Training PPO on LunarLander-v2 (1M steps)...")
ppo_net, ppo_rewards = train_ppo(
    env_id="LunarLander-v2",
    total_steps=1_000_000,
    rollout_steps=2048,
    gamma=0.99,
    gae_lambda=0.95,
    lr=3e-4,
    clip_eps=0.2,
    c1=0.5,
    c2=0.01,
    n_epochs=10,
    batch_size=64,
    hidden_dim=256,
    seed=42,
    log_dir="runs/ppo_lunar",
)

if ppo_rewards:
    last_100_mean = float(np.mean(ppo_rewards[-100:]))
    print(f"\nMean reward (last 100 episodes): {last_100_mean:.1f}")
    assert last_100_mean >= 150, (
        f"PPO on LunarLander-v2 did not converge: mean={last_100_mean:.1f} (need >= 150). "
        "Check compute_ppo_loss, ppo_update, RolloutBuffer.get, and NormalizeObsWrapper."
    )
    print("✓ PPO converged (mean reward >= 150)")

# %%
if ppo_rewards:
    window = 50
    smoothed = np.convolve(ppo_rewards, np.ones(window) / window, mode='valid')
    plt.figure(figsize=(10, 4))
    plt.plot(ppo_rewards, alpha=0.2, color='steelblue')
    plt.plot(np.arange(window - 1, len(ppo_rewards)), smoothed,
             color='steelblue', linewidth=2, label=f'Smoothed (w={window})')
    plt.axhline(y=150, color='red', linestyle='--', label='Target: 150')
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("PPO on LunarLander-v2")
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Part 6: Ablation — PPO Components

# %%
print("Ablation (a): PPO without clipping (clip_eps=inf)...")
_, rewards_no_clip = train_ppo(
    env_id="LunarLander-v2",
    total_steps=500_000,
    clip_eps=float('inf'),
    c2=0.01,
    n_epochs=10,
    seed=42,
    log_dir="runs/ppo_no_clip",
)

print("Baseline PPO for comparison...")
_, rewards_baseline = train_ppo(
    env_id="LunarLander-v2",
    total_steps=500_000,
    clip_eps=0.2,
    c2=0.01,
    n_epochs=10,
    seed=42,
    log_dir="runs/ppo_baseline",
)

# %%
if rewards_no_clip and rewards_baseline:
    window = 30
    plt.figure(figsize=(10, 4))
    if len(rewards_baseline) > window:
        s_base = np.convolve(rewards_baseline, np.ones(window) / window, mode='valid')
        plt.plot(np.arange(window - 1, len(rewards_baseline)), s_base,
                 label='PPO (clip_eps=0.2)', color='steelblue', linewidth=2)
    if len(rewards_no_clip) > window:
        s_nc = np.convolve(rewards_no_clip, np.ones(window) / window, mode='valid')
        plt.plot(np.arange(window - 1, len(rewards_no_clip)), s_nc,
                 label='No clip (clip_eps=inf)', color='red', linewidth=2)
    plt.axhline(y=150, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (smoothed)")
    plt.title("Ablation: Clipping vs No Clipping")
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%
print("Ablation (b): PPO without entropy bonus (c2=0)...")
_, rewards_no_entropy = train_ppo(
    env_id="LunarLander-v2",
    total_steps=500_000,
    clip_eps=0.2,
    c2=0.0,
    n_epochs=10,
    seed=42,
    log_dir="runs/ppo_no_entropy",
)

# %%
print("Ablation (c): PPO with n_epochs=1...")
_, rewards_1epoch = train_ppo(
    env_id="LunarLander-v2",
    total_steps=500_000,
    clip_eps=0.2,
    c2=0.01,
    n_epochs=1,
    seed=42,
    log_dir="runs/ppo_1epoch",
)

# %% [markdown]
# ## Part 7: Observation Questions

# %% [markdown]
# **Answer Q1:**
# (fill in)

# %% [markdown]
# **Answer Q2:**
# (fill in)

# %% [markdown]
# **Answer Q3:**
# With n_epochs=1 and batch_size=64, there are 2048/64 = 32 gradient steps per 2048 real steps,
# giving 32/2048 ≈ 0.016 steps per env step. With n_epochs=10, there are 320 steps per 2048 env
# steps = 0.156 steps per env step. Clip fraction matters more with more epochs because the policy
# can drift far from the data-collection policy, making old log_probs_old stale.

# %% [markdown]
# ## Part 8: Reflection

# %% [markdown]
# **Answers:**
# 1. Rollout buffer → prompt + response batch; clip_eps → KL trust region coefficient beta;
#    entropy bonus → diversity regularization; value loss → critic/value head training;
#    advantage normalization → reward whitening.
# 2. Negative EV means the critic is worse than predicting the mean — it is actively misleading.
#    Fix by increasing c1 (weight the value loss more), using a larger critic architecture,
#    or lowering the actor learning rate so the critic can keep pace.
# 3. Separate networks avoid gradient interference but are less sample-efficient. Sharing the
#    backbone is beneficial when features useful for policy and value function overlap strongly.
