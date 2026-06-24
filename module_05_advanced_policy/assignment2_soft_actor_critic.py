# %% [markdown]
# # Module 05, Assignment 2: Soft Actor-Critic (SAC)

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

# %%
def sac_critic_loss(q_net: TwinQNetwork, q_target: TwinQNetwork,
                    policy: GaussianPolicyHead,
                    batch: tuple, gamma: float, log_alpha: torch.Tensor) -> torch.Tensor:
    """Twin critic loss."""
    obs, actions, rewards, next_obs, dones = batch

    # Current Q estimates
    q1, q2 = q_net(obs, actions)

    with torch.no_grad():
        # Sample next actions from current policy
        next_actions, next_log_pi = policy.sample(next_obs)
        # Target Q values
        q1_target, q2_target = q_target(next_obs, next_actions)
        alpha = log_alpha.exp().detach()
        q_min_target = torch.min(q1_target, q2_target) - alpha * next_log_pi
        y = rewards + gamma * (1 - dones) * q_min_target

    loss_q = 0.5 * (F.mse_loss(q1, y) + F.mse_loss(q2, y))
    return loss_q


def sac_actor_loss(policy: GaussianPolicyHead, q_net: TwinQNetwork,
                   obs: torch.Tensor, log_alpha: torch.Tensor) -> torch.Tensor:
    """Actor loss: maximize E[min(Q1,Q2)(s, a_tilde) - alpha * log_pi(a_tilde|s)]"""
    a_tilde, log_pi = policy.sample(obs)
    q1, q2 = q_net(obs, a_tilde)
    q_min = torch.min(q1, q2)
    alpha = log_alpha.exp().detach()
    actor_loss = (alpha * log_pi - q_min).mean()
    return actor_loss


def sac_alpha_loss(log_alpha: torch.Tensor, log_pi: torch.Tensor,
                   target_entropy: float) -> torch.Tensor:
    """Automatic temperature tuning."""
    alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
    return alpha_loss


# %% [markdown]
# ## Part 3: Training Loop on Pendulum-v1

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
    """Train SAC on a continuous control environment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    target_entropy = -action_dim

    # Networks
    policy = GaussianPolicyHead(obs_dim, action_dim, hidden_dim)
    q_net = TwinQNetwork(obs_dim, action_dim, hidden_dim)
    q_target = TwinQNetwork(obs_dim, action_dim, hidden_dim)
    q_target.load_state_dict(q_net.state_dict())
    for p in q_target.parameters():
        p.requires_grad = False

    log_alpha = torch.zeros(1, requires_grad=True)

    actor_opt = optim.Adam(policy.parameters(), lr=lr_actor)
    critic_opt = optim.Adam(q_net.parameters(), lr=lr_critic)
    alpha_opt = optim.Adam([log_alpha], lr=lr_alpha)

    replay = ReplayBuffer(buffer_capacity)
    writer = make_writer(f"sac_{env_id}")
    episode_rewards = []
    alpha_history = []

    obs, _ = env.reset(seed=seed)
    ep_reward = 0.0

    for step in range(total_steps):
        if step < learning_starts:
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
            writer.add_scalar("train/episode_reward", ep_reward, step)
            if len(episode_rewards) % 10 == 0:
                mean_10 = np.mean(episode_rewards[-10:])
                print(f"Step {step:7d} | Episodes: {len(episode_rewards):4d} | "
                      f"Mean reward (last 10): {mean_10:.1f} | "
                      f"alpha: {log_alpha.exp().item():.4f}")
            obs, _ = env.reset()
            ep_reward = 0.0
        else:
            obs = next_obs

        if step >= learning_starts and len(replay) >= batch_size and step % update_freq == 0:
            s, a, r, ns, d = replay.sample(batch_size)
            obs_t = torch.FloatTensor(s)
            act_t = torch.FloatTensor(a)
            rew_t = torch.FloatTensor(r)
            next_obs_t = torch.FloatTensor(ns)
            done_t = torch.FloatTensor(d)

            batch = (obs_t, act_t, rew_t, next_obs_t, done_t)

            # Critic update
            critic_loss = sac_critic_loss(q_net, q_target, policy, batch, gamma, log_alpha)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            # Actor update
            actor_loss = sac_actor_loss(policy, q_net, obs_t, log_alpha)
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            # Alpha update
            with torch.no_grad():
                _, log_pi_sample = policy.sample(obs_t)
            alpha_loss = sac_alpha_loss(log_alpha, log_pi_sample, target_entropy)
            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()

            # Soft update target networks
            soft_update(q_target, q_net, tau=tau)

            writer.add_scalar("train/critic_loss", critic_loss.item(), step)
            writer.add_scalar("train/actor_loss", actor_loss.item(), step)
            writer.add_scalar("train/alpha", log_alpha.exp().item(), step)
            alpha_history.append(log_alpha.exp().item())

    writer.close()
    env.close()
    return episode_rewards, alpha_history


# %% [markdown]
# ## Part 4: Verification

# %%
print("Training SAC on Pendulum-v1 (100k steps)...")
sac_rewards, alpha_hist = train_sac(
    env_id="Pendulum-v1",
    total_steps=100_000,
    seed=42,
)

if sac_rewards:
    last_10_mean = np.mean(sac_rewards[-10:])
    print(f"Mean reward (last 10 episodes): {last_10_mean:.1f}")
    assert last_10_mean >= -200, (
        f"SAC on Pendulum-v1 did not converge: mean={last_10_mean:.1f} (need >= -200)."
    )
    print("✓ Pendulum-v1: SAC converged")

# %% [markdown]
# ## Part 5: Reflection

# %% [markdown]
# **Answers:**
# 1. The entropy term encourages exploration and prevents premature convergence to a deterministic
#    policy. It also acts as a regularizer, making the policy more robust.
# 2. SAC is more sample-efficient than PPO on continuous control because it uses off-policy data
#    (replay buffer) and the reparameterization trick allows direct gradient flow through actions.
# 3. Automatic temperature tuning removes the need to tune alpha manually. It adapts alpha so that
#    the policy maintains the target entropy, balancing exploration and exploitation automatically.
