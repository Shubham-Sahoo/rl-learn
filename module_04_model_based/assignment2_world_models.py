# %% [markdown]
# # Module 04, Assignment 2: Neural World Models

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import List, Tuple

from rllearn.logging import make_writer

# %% [markdown]
# ## Part 2: Implement TransitionModel and RewardModel

# %%
class TransitionModel(nn.Module):
    """Predict next observation from (obs, action)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return predicted next_obs of same shape as obs."""
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class RewardModel(nn.Module):
    """Predict scalar reward from (obs, action)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return predicted reward of shape (batch, 1)."""
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


def train_world_model(transition_model: TransitionModel,
                      reward_model: RewardModel,
                      optimizer: torch.optim.Optimizer,
                      replay_buffer,
                      batch_size: int = 256) -> Tuple[float, float]:
    """Train one gradient step on world model. Returns (transition_loss, reward_loss)."""
    obs_b, action_b, reward_b, next_obs_b = replay_buffer.sample(batch_size)

    # One-hot encode actions
    n_actions = transition_model.net[0].in_features - obs_b.shape[1]
    action_oh = F.one_hot(action_b, num_classes=n_actions).float()

    # Transition loss
    pred_next_obs = transition_model(obs_b, action_oh)
    transition_loss = F.mse_loss(pred_next_obs, next_obs_b)

    # Reward loss
    pred_reward = reward_model(obs_b, action_oh)
    reward_loss = F.mse_loss(pred_reward, reward_b)

    total_loss = transition_loss + reward_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return transition_loss.item(), reward_loss.item()


def generate_model_rollout(transition_model: TransitionModel,
                           reward_model: RewardModel,
                           start_obs: torch.Tensor,
                           policy,
                           horizon: int = 5) -> List[Tuple]:
    """Generate synthetic (obs, action, reward, next_obs) transitions using learned model."""
    n_actions = transition_model.net[0].in_features - start_obs.shape[0]
    rollout = []
    obs = start_obs.unsqueeze(0)  # (1, obs_dim)

    transition_model.eval()
    reward_model.eval()

    with torch.no_grad():
        for _ in range(horizon):
            action_int = policy(obs.squeeze(0))
            action_t = torch.LongTensor([action_int])
            action_oh = F.one_hot(action_t, num_classes=n_actions).float()

            next_obs = transition_model(obs, action_oh)
            reward = reward_model(obs, action_oh).squeeze().item()

            rollout.append((
                obs.squeeze(0).numpy(),
                action_int,
                reward,
                next_obs.squeeze(0).numpy(),
            ))
            obs = next_obs

    return rollout


# %% [markdown]
# ## Part 3: Collect Offline Data and Train

# %%
def collect_offline_data(env_id: str = "CartPole-v1",
                         n_transitions: int = 5000,
                         seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect transitions using a random policy."""
    np.random.seed(seed)
    env = gym.make(env_id)

    all_obs, all_actions, all_rewards, all_next_obs = [], [], [], []
    obs, _ = env.reset(seed=seed)
    collected = 0

    while collected < n_transitions:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        all_obs.append(obs.copy())
        all_actions.append(action)
        all_rewards.append(float(reward))
        all_next_obs.append(next_obs.copy())
        collected += 1
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

    env.close()
    return (np.array(all_obs, dtype=np.float32),
            np.array(all_actions, dtype=np.int64),
            np.array(all_rewards, dtype=np.float32),
            np.array(all_next_obs, dtype=np.float32))


class SimpleReplayBuffer:
    """Minimal replay buffer for world model training."""

    def __init__(self, obs: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, next_obs: np.ndarray):
        self.obs = torch.FloatTensor(obs)
        self.actions = torch.LongTensor(actions)
        self.rewards = torch.FloatTensor(rewards).unsqueeze(1)
        self.next_obs = torch.FloatTensor(next_obs)
        self.size = len(obs)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.obs[idx], self.actions[idx],
                self.rewards[idx], self.next_obs[idx])


print("Collecting 5000 offline transitions from CartPole-v1 ...")
obs_data, action_data, reward_data, next_obs_data = collect_offline_data(
    n_transitions=5000, seed=42)

split = int(0.8 * len(obs_data))
train_buf = SimpleReplayBuffer(obs_data[:split], action_data[:split],
                               reward_data[:split], next_obs_data[:split])
eval_obs = torch.FloatTensor(obs_data[split:])
eval_actions = torch.LongTensor(action_data[split:])
eval_next_obs = torch.FloatTensor(next_obs_data[split:])

print(f"Train transitions: {train_buf.size}  |  Eval transitions: {len(eval_obs)}")

# %%
env_tmp = gym.make("CartPole-v1")
OBS_DIM = env_tmp.observation_space.shape[0]
ACTION_DIM = env_tmp.action_space.n
env_tmp.close()

transition_model = TransitionModel(OBS_DIM, ACTION_DIM, hidden_dim=256)
reward_model = RewardModel(OBS_DIM, ACTION_DIM, hidden_dim=256)
optimizer = torch.optim.Adam(
    list(transition_model.parameters()) + list(reward_model.parameters()),
    lr=3e-4,
)

writer = make_writer("world_model_cartpole")

print("Training world model for 1000 gradient steps ...")
t_losses, r_losses = [], []

for step in range(1000):
    t_loss, r_loss = train_world_model(
        transition_model, reward_model, optimizer, train_buf, batch_size=256)
    t_losses.append(t_loss)
    r_losses.append(r_loss)

    writer.add_scalar("train/transition_loss", t_loss, step)
    writer.add_scalar("train/reward_loss", r_loss, step)

    if (step + 1) % 200 == 0:
        print(f"Step {step+1:4d} | t_loss={t_loss:.5f}  r_loss={r_loss:.5f}")

writer.close()
print("Training complete.")

# %% [markdown]
# ### Verification: One-Step Prediction MSE < 0.01

# %%
transition_model.eval()
with torch.no_grad():
    action_onehot = F.one_hot(eval_actions, num_classes=ACTION_DIM).float()
    pred_next_obs = transition_model(eval_obs, action_onehot)
    eval_mse = F.mse_loss(pred_next_obs, eval_next_obs).item()

print(f"One-step prediction MSE on held-out data: {eval_mse:.6f}")

writer_eval = make_writer("world_model_eval")
writer_eval.add_scalar("eval/mse", eval_mse, 0)
writer_eval.close()

assert eval_mse < 0.01, (
    f"One-step MSE too high: {eval_mse:.6f} (need < 0.01). "
    "Check TransitionModel architecture and train_world_model."
)
print("Verification passed: one-step MSE < 0.01")

# %% [markdown]
# ## Part 4: Ablation — Compounding Error over Multi-Step Horizons

# %%
def compute_rollout_mse(transition_model: TransitionModel,
                        env_id: str,
                        n_rollouts: int = 100,
                        max_horizon: int = 20,
                        seed: int = 42) -> dict:
    """Roll out both model and real env from the same start state; compare at each horizon."""
    env = gym.make(env_id)
    transition_model.eval()
    mse_by_horizon: dict = {h: [] for h in range(1, max_horizon + 1)}

    for i in range(n_rollouts):
        obs_real, _ = env.reset(seed=seed + i)
        obs_model = torch.FloatTensor(obs_real).unsqueeze(0)

        for h in range(1, max_horizon + 1):
            action = env.action_space.sample()
            obs_real_next, _, terminated, truncated, _ = env.step(action)

            action_t = torch.LongTensor([action])
            action_oh = F.one_hot(action_t, num_classes=ACTION_DIM).float()
            with torch.no_grad():
                obs_model_next = transition_model(obs_model, action_oh)

            mse = float(F.mse_loss(obs_model_next,
                                   torch.FloatTensor(obs_real_next).unsqueeze(0)).item())
            mse_by_horizon[h].append(mse)

            if terminated or truncated:
                break

            obs_model = obs_model_next
            obs_real = obs_real_next

    env.close()
    return {h: float(np.mean(v)) for h, v in mse_by_horizon.items() if len(v) > 0}


print("Computing multi-step rollout MSE ...")
horizon_mse = compute_rollout_mse(
    transition_model, "CartPole-v1", n_rollouts=100, max_horizon=20, seed=0)

# %%
horizons = sorted(horizon_mse.keys())
mse_values = [horizon_mse[h] for h in horizons]

plt.figure(figsize=(8, 4))
plt.plot(horizons, mse_values, marker='o', color='steelblue', linewidth=2)
plt.xlabel("Rollout Horizon h")
plt.ylabel("Mean MSE (model vs. real env)")
plt.title("Compounding Error in Multi-Step Model Rollouts (CartPole-v1)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

for h in [1, 5, 10, 20]:
    if h in horizon_mse:
        print(f"Horizon {h:2d}: MSE = {horizon_mse[h]:.5f}")

# %% [markdown]
# **Answers:**
# 1. Error compounds because each step uses predicted obs as input. RSSM uses stochastic latents
#    and a KL penalty to keep imagined rollouts near the training distribution.
# 2. Trust model when horizon is short (h <= 5), state is in-distribution, ensemble disagreement
#    is low. Use real experience for novel states or contact-rich dynamics.
# 3. Use short horizons (1-3 steps), train a model ensemble, collect more real data when ensemble
#    disagreement is high (MBPO-style), use data augmentation with model rollouts.
