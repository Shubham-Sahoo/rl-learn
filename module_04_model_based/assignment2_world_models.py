# %% [markdown]
# # Module 04, Assignment 2: Neural World Models
#
# ## Prerequisites
# - Module 02 A2: ReplayBuffer (implement rllearn/buffers.py first)
# - Module 02 A2: make_env (implement rllearn/envs.py first)
# - Lecture notes sections 3–4
#
# ## Learning Objectives
# 1. Build a learned transition model for CartPole
# 2. Train it on offline data and evaluate one-step prediction MSE
# 3. Generate synthetic rollouts to augment real experience
# 4. Understand compounding error over multi-step rollouts

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
# ## Part 1: Theory Recap
#
# ### Transition and Reward Models
#
# We learn two parameterized functions from replay data $\mathcal{D}$:
#
# $$f_\phi : (s_t, a_t) \mapsto \hat{s}_{t+1}$$
#
# $$g_\phi : (s_t, a_t) \mapsto \hat{r}_t$$
#
# **Transition loss:**
#
# $$\mathcal{L}_{\text{transition}}(\phi) = \mathbb{E}_{(s_t, a_t, s_{t+1}) \sim \mathcal{D}}\!\left[\|f_\phi(s_t, a_t) - s_{t+1}\|^2\right]$$
#
# **Reward loss:**
#
# $$\mathcal{L}_{\text{reward}}(\phi) = \mathbb{E}_{(s_t, a_t, r_t) \sim \mathcal{D}}\!\left[(g_\phi(s_t, a_t) - r_t)^2\right]$$
#
# ### Compounding Error
#
# At horizon $H$, error accumulates because each predicted $\hat{s}_{t+1}$ feeds back as input:
#
# $$\hat{s}_{t+H} = f_\phi(\ldots f_\phi(f_\phi(s_t, a_t), a_{t+1}) \ldots, a_{t+H-1})$$
#
# Even a small one-step MSE $\epsilon$ compounds: errors at step $t$ are inputs to step $t+1$, and
# because $f_\phi$ is nonlinear, small perturbations can amplify. In practice, MSE at horizon $H$
# grows super-linearly with $H$ for chaotic environments.

# %% [markdown]
# ## Part 2: Implement TransitionModel and RewardModel
#
# Fill in every `raise NotImplementedError`. Do not change method signatures.
#
# **TransitionModel:** MLP with architecture `obs_dim + action_dim → hidden_dim → ReLU → hidden_dim → ReLU → obs_dim`.
# The input is the concatenation of `obs` and a one-hot encoded action.
#
# **RewardModel:** Same architecture but output is a scalar (shape `(batch, 1)`).
#
# **train_world_model:** Sample a batch from `replay_buffer.sample(batch_size)`, compute both
# losses, sum them, and take one gradient step. Return `(transition_loss.item(), reward_loss.item())`.
#
# **generate_model_rollout:** Starting from `start_obs`, roll out the transition model for
# `horizon` steps. At each step call `policy(obs)` to get an action, then use the transition and
# reward models to predict `next_obs` and `reward`. Return a list of
# `(obs, action, reward, next_obs)` tuples.

# %%
class TransitionModel(nn.Module):
    """Predict next observation from (obs, action)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # TODO: MLP(obs_dim + action_dim, obs_dim)
        raise NotImplementedError

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return predicted next_obs of same shape as obs."""
        raise NotImplementedError


class RewardModel(nn.Module):
    """Predict scalar reward from (obs, action)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        raise NotImplementedError

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return predicted reward of shape (batch, 1)."""
        raise NotImplementedError


def train_world_model(transition_model: TransitionModel,
                      reward_model: RewardModel,
                      optimizer: torch.optim.Optimizer,
                      replay_buffer,
                      batch_size: int = 256) -> Tuple[float, float]:
    """Train one gradient step on world model. Returns (transition_loss, reward_loss)."""
    raise NotImplementedError


def generate_model_rollout(transition_model: TransitionModel,
                           reward_model: RewardModel,
                           start_obs: torch.Tensor,
                           policy,
                           horizon: int = 5) -> List[Tuple]:
    """Generate synthetic (obs, action, reward, next_obs) transitions using learned model.

    Parameters
    ----------
    transition_model : learned f_phi
    reward_model     : learned g_phi
    start_obs        : initial observation tensor of shape (obs_dim,)
    policy           : callable obs_tensor -> action_int
    horizon          : number of steps to roll out

    Returns
    -------
    List of (obs, action, reward, next_obs) tuples as numpy arrays / floats.
    """
    raise NotImplementedError


# %% [markdown]
# ## Part 3: Collect Offline Data and Train
#
# The data collection and training loop below are provided. Read through before running.

# %%
def collect_offline_data(env_id: str = "CartPole-v1",
                         n_transitions: int = 5000,
                         seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect transitions using a random policy.

    Returns
    -------
    obs, actions, rewards, next_obs — each a numpy array with n_transitions rows.
    """
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
    """Minimal replay buffer for world model training (no rllearn dependency)."""

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

# 80/20 train/eval split
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
# ### TensorBoard (optional — run before training)

# %%
# %load_ext tensorboard
# %tensorboard --logdir runs/

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
#
# Roll out the transition model from held-out starting observations for horizons h ∈ {1, 5, 10, 20}.
# Measure MSE between model predictions and ground-truth observations from real rollouts.

# %%
def compute_rollout_mse(transition_model: TransitionModel,
                        env_id: str,
                        n_rollouts: int = 100,
                        max_horizon: int = 20,
                        seed: int = 42) -> dict:
    """Roll out both model and real env from the same start state; compare at each horizon.

    Returns dict horizon -> mean MSE over n_rollouts.
    """
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
# ### Observations
#
# 1. **One-step MSE is low** (< 0.01 from verification). The model learned the CartPole dynamics well.
#
# 2. **MSE grows with horizon** due to compounding: the error at step $h$ feeds into the model's
#    input at step $h+1$, and the distribution shift from the training data accumulates.
#
# 3. **When is the model trustworthy?** Look at the elbow in the curve. Beyond the elbow, model
#    predictions are unreliable and planning with them will mislead the policy.
#
# *(Replace with your own observations after running.)*

# %% [markdown]
# ## Part 5: Reflection
#
# **Q1 — Why does error compound? How does RSSM mitigate this?**
# Each step feeds the model's prediction as its next input. Errors accumulate because $f_\phi$
# is not trained on its own predictions — only on real transitions. RSSM mitigates this with
# (a) a stochastic latent $z_t$ that can represent uncertainty over future states, and (b) a KL
# penalty that keeps the prior (used during imagination) close to the posterior (used with real obs),
# ensuring imagined rollouts remain in-distribution.
#
# **Q2 — When would you trust model rollouts vs. real experience?**
# Trust the model when: rollout horizon is short (h ≤ 5), the start state is within the training
# distribution, and ensemble disagreement (if available) is low. Use real experience when: the
# model is newly initialized, you are in a novel state (exploration), or the environment has
# contact-rich/discontinuous dynamics that are hard to fit.
#
# **Q3 — Design question:**
# Suppose you have a 5000-step real dataset (as above) and unlimited model compute. What would
# you change to maximize the policy's real-world performance? Consider: rollout horizon, model
# ensemble, data augmentation, and when to collect more real data.

# %% [markdown]
# **Answers:**
# 1.
# 2.
# 3.
