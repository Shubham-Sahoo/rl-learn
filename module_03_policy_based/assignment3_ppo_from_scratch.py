# %% [markdown]
# # Assignment 3: PPO from Scratch
# **Prerequisites:** Read `lecture_notes.md` §7–8 and complete Assignments 1–2.
#
# **Learning objectives:**
# - Implement the PPO-Clip policy loss
# - Implement the full PPO update with minibatch epochs
# - Implement `RolloutBuffer` and `NormalizeObsWrapper`
# - Verify on HalfCheetah-v4 (mean reward ≥ 3000) or Ant-v4 (mean reward ≥ 2500) within 3M steps
# - Ablate clipping, entropy bonus, and number of update epochs

# %% [markdown]
# ## Part 0: Implement rllearn Stubs First
#
# **Before writing any code in this notebook, implement the following:**
#
# ### 1. `RolloutBuffer` in `rllearn/buffers.py`
#
# ```python
# class RolloutBuffer:
#     def add(self, obs, action, reward, done, value, log_prob):
#         # Append each quantity to its respective list
#
#     def get(self) -> dict:
#         # Return dict with keys: obs, actions, rewards, dones, values, log_probs
#         # Convert each list to a torch.FloatTensor
#         # actions and dones should be LongTensor and FloatTensor respectively
# ```
#
# **Why a rollout buffer (not replay)?** PPO is on-policy — it uses data from the *current*
# policy only. After each update, the buffer is cleared and refilled with fresh rollouts.
# A replay buffer (off-policy) would be incorrect here: old transitions come from different
# policies, making the importance-weighted objective incorrect.
#
# ### 2. `NormalizeObsWrapper` in `rllearn/envs.py`
#
# ```python
# class NormalizeObsWrapper(gym.ObservationWrapper):
#     def __init__(self, env, epsilon=1e-8):
#         super().__init__(env)
#         # Initialize: self.mean = np.zeros(obs_shape)
#         #             self.var  = np.ones(obs_shape)
#         #             self.count = 0
#         #             self.epsilon = epsilon
#
#     def observation(self, obs):
#         # Welford update: count += 1; delta = obs - mean; mean += delta/count
#         #                 delta2 = obs - mean; var = var + (delta*delta2 - var)/count
#         # Normalize: return (obs - self.mean) / (np.sqrt(self.var) + self.epsilon)
# ```
#
# **Why normalize observations for MuJoCo?** MuJoCo joint angles and velocities live on very
# different scales. Unnormalized observations cause poor conditioning of the neural network's
# input layer, leading to slow convergence or divergence. Welford's algorithm is numerically
# stable for streaming updates without storing all past observations.

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
# ## Part 1: Theory Recap
#
# **PPO probability ratio:**
#
# $$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$$
#
# **PPO-Clip objective:**
#
# $$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\bigl(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\bigr)\right]$$
#
# **Full PPO loss (maximize):**
#
# $$\mathcal{L} = \mathcal{L}^{CLIP} - c_1 \mathcal{L}^{VF} + c_2 S[\pi_\theta]$$
#
# **Key design choices in PPO:**
# - **Multiple epochs:** Reuse the same rollout for $K$ gradient steps (more sample efficient
#   than vanilla policy gradient, which takes one step per rollout).
# - **Minibatches:** Shuffle and split the rollout into minibatches for each epoch.
# - **Advantage normalization:** Normalize $\hat{A}_t$ over the entire rollout *before* the
#   update loop (not per-minibatch).
# - **Clip fraction monitoring:** If > 0.5, the policy is changing too fast.

# %% [markdown]
# ## Part 2: Implement PPO-Clip Loss
#
# **Steps:**
# 1. Compute the ratio in log-space for numerical stability:
#    `ratio = torch.exp(log_probs_new - log_probs_old)`
# 2. Compute the unclipped surrogate: `surr1 = ratio * advantages`
# 3. Compute the clipped surrogate:
#    `surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages`
# 4. **Loss = -mean(min(surr1, surr2))**. Negative because we do gradient descent but want
#    to maximize the objective.
# 5. Clip fraction: `mean((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps))`
#
# **Edge case:** When `clip_eps = float('inf')`, `torch.clamp` should leave the ratio unchanged.
# This disables clipping entirely (ablation setting).

# %%
def compute_ppo_loss(log_probs_new: torch.Tensor,
                      log_probs_old: torch.Tensor,
                      advantages: torch.Tensor,
                      clip_eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PPO-Clip policy loss.

    r_t = exp(log_pi_new - log_pi_old)
    L_CLIP = mean(min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t))
    loss = -L_CLIP   (negative for gradient ascent via descent)

    Parameters
    ----------
    log_probs_new : log pi_theta(a_t | s_t) for current parameters theta
    log_probs_old : log pi_theta_old(a_t | s_t) from rollout collection
    advantages    : GAE advantages, already normalized
    clip_eps      : clipping epsilon (e.g. 0.2). float('inf') disables clipping.

    Returns
    -------
    (loss, clip_fraction)
        loss          : scalar tensor (to minimize)
        clip_fraction : scalar tensor, fraction of ratios outside [1-eps, 1+eps]
    """
    # TODO: ratio = torch.exp(log_probs_new - log_probs_old)
    # TODO: surr1 = ratio * advantages
    # TODO: surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    # TODO: loss = -torch.mean(torch.min(surr1, surr2))
    # TODO: clip_fraction = torch.mean(((ratio < 1-clip_eps) | (ratio > 1+clip_eps)).float())
    # TODO: return loss, clip_fraction
    raise NotImplementedError


# %% [markdown]
# ## Part 3: Implement the Full PPO Update
#
# **Algorithm (per iteration):**
# 1. Get data from rollout buffer.
# 2. Normalize advantages over the full rollout: `(A - mean(A)) / (std(A) + 1e-8)`.
# 3. For each of `n_epochs` epochs:
#    a. Shuffle indices.
#    b. For each minibatch of size `batch_size`:
#       - Re-evaluate `log_probs_new`, `values_new`, and `entropy` using current `actor_critic`.
#       - `policy_loss, clip_frac = compute_ppo_loss(log_probs_new, log_probs_old_mb, adv_mb, clip_eps)`
#       - `value_loss = 0.5 * mean((values_new - returns_mb) ** 2)`
#       - `entropy_loss = mean(Categorical(logits=logits).entropy())`
#       - `total_loss = policy_loss + c1 * value_loss - c2 * entropy_loss`
#       - Gradient step.
# 4. Return dict with logged statistics.
#
# **Implementation notes:**
# - `returns = advantages + values` (the *un-normalized* advantages + values from rollout).
#   Normalize advantages AFTER computing returns.
# - `explained_variance(values_pred, returns)` measures how well the critic fits the returns.
#   EV close to 1 = good critic; EV < 0 = critic worse than the mean prediction.

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
    """
    Full PPO update over n_epochs of minibatch gradient steps.

    Parameters
    ----------
    actor_critic    : shared actor-critic network
    optimizer       : optimizer (e.g. Adam)
    rollout_buffer  : filled RolloutBuffer from latest rollout
    advantages      : GAE advantages of shape (T,), computed outside this function
    clip_eps        : PPO clip epsilon (set to float('inf') to disable clipping)
    c1              : value loss coefficient
    c2              : entropy bonus coefficient (set to 0 to disable entropy)
    n_epochs        : number of full passes over the rollout buffer per update
    batch_size      : minibatch size

    Returns
    -------
    dict with keys:
        policy_loss       : float, mean policy loss across all minibatches
        value_loss        : float, mean value loss across all minibatches
        entropy_loss      : float, mean entropy (higher = more stochastic policy)
        clip_fraction     : float, fraction of ratios clipped
        explained_variance: float, EV of the value function
    """
    # TODO: data = rollout_buffer.get()
    # TODO: extract obs, actions, values, log_probs_old from data
    # TODO: returns = advantages + values  (before normalization — used for critic target)
    # TODO: normalize advantages: adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    # TODO: n_samples = len(obs)
    # TODO: initialize accumulators for metrics
    # TODO: for epoch in range(n_epochs):
    #         indices = torch.randperm(n_samples)
    #         for start in range(0, n_samples, batch_size):
    #             mb_idx = indices[start:start+batch_size]
    #             obs_mb, actions_mb, adv_mb, returns_mb, log_probs_old_mb = [get minibatch]
    #             logits, values_new = actor_critic(obs_mb)
    #             dist = Categorical(logits=logits)
    #             log_probs_new = dist.log_prob(actions_mb)
    #             entropy = dist.entropy().mean()
    #             policy_loss, clip_frac = compute_ppo_loss(log_probs_new, log_probs_old_mb, adv_mb, clip_eps)
    #             value_loss = 0.5 * ((values_new.squeeze() - returns_mb) ** 2).mean()
    #             total_loss = policy_loss + c1 * value_loss - c2 * entropy
    #             optimizer.zero_grad(); total_loss.backward(); optimizer.step()
    #             [accumulate metrics]
    # TODO: compute explained variance using rllearn.utils.explained_variance
    # TODO: return dict of averaged metrics
    raise NotImplementedError


# %% [markdown]
# ## Part 4: Training Loop
#
# The training loop below is provided. It:
# 1. Wraps the environment with `NormalizeObsWrapper`
# 2. Collects 2048 steps per iteration into `RolloutBuffer`
# 3. Calls `compute_gae` to fill advantages
# 4. Calls `ppo_update` and logs all returned metrics
# 5. Optionally logs to TensorBoard

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
    """
    Full PPO training loop.

    Returns (actor_critic, episode_rewards) where episode_rewards is a list of
    episode returns collected during training.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_ppo_env(env_id, seed=seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n if hasattr(env.action_space, 'n') else None

    # PPO on continuous action spaces uses GaussianPolicyHead (Module 05).
    # For discrete environments (CartPole, LunarLander) we use ActorCritic directly.
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
        # Collect rollout_steps transitions
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

            # Collect episode statistics from RecordEpisodeStatistics wrapper
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
#
# **Target environments:**
# - HalfCheetah-v4: mean reward ≥ 3000 within 3M steps (requires MuJoCo)
# - LunarLander-v2: mean reward ≥ 200 within 1M steps (discrete, always available)
#
# **Note:** The training loop above automatically falls back to LunarLander-v2 if the
# target environment uses continuous actions (full continuous PPO is covered in Module 05).

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
#
# We ablate three key PPO components to understand their individual contributions.
# Each ablation trains for fewer steps to save time; the relative differences are the focus.

# %% [markdown]
# ### Ablation (a): No Clipping (`clip_eps = float('inf')`)
#
# **Prediction:** Without clipping, large policy updates are possible. The policy can change
# dramatically in a single iteration, making the rollout data stale and leading to instability.

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
                 label='No clip (clip_eps=∞)', color='red', linewidth=2)
    plt.axhline(y=150, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (smoothed)")
    plt.title("Ablation: Clipping vs No Clipping")
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Ablation (b): No Entropy Bonus (`c2 = 0`)
#
# **Prediction:** Without the entropy term, the policy collapses to near-deterministic early in
# training. Once collapsed, the gradient signal becomes very small and learning stalls.

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
if rewards_no_entropy and rewards_baseline:
    window = 30
    plt.figure(figsize=(10, 4))
    if len(rewards_baseline) > window:
        s_base = np.convolve(rewards_baseline, np.ones(window) / window, mode='valid')
        plt.plot(np.arange(window - 1, len(rewards_baseline)), s_base,
                 label='PPO (c2=0.01)', color='steelblue', linewidth=2)
    if len(rewards_no_entropy) > window:
        s_ne = np.convolve(rewards_no_entropy, np.ones(window) / window, mode='valid')
        plt.plot(np.arange(window - 1, len(rewards_no_entropy)), s_ne,
                 label='No entropy (c2=0)', color='darkorange', linewidth=2)
    plt.axhline(y=150, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (smoothed)")
    plt.title("Ablation: Entropy Bonus vs No Entropy")
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Ablation (c): 1 Epoch vs 10 Epochs
#
# **Prediction:** Fewer epochs per rollout means the rollout data is used less efficiently,
# requiring more environment steps to converge. More epochs reuse data but risk overfitting to
# the rollout (which clipping helps prevent).

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

print("PPO with n_epochs=10 (baseline)...")
# Already trained above in rewards_baseline, reuse if available
if not rewards_baseline:
    _, rewards_baseline = train_ppo(
        env_id="LunarLander-v2",
        total_steps=500_000,
        clip_eps=0.2,
        c2=0.01,
        n_epochs=10,
        seed=42,
        log_dir="runs/ppo_10epoch",
    )

# %%
if rewards_1epoch and rewards_baseline:
    window = 30
    plt.figure(figsize=(10, 4))
    if len(rewards_baseline) > window:
        s_base = np.convolve(rewards_baseline, np.ones(window) / window, mode='valid')
        plt.plot(np.arange(window - 1, len(rewards_baseline)), s_base,
                 label='n_epochs=10', color='steelblue', linewidth=2)
    if len(rewards_1epoch) > window:
        s_1e = np.convolve(rewards_1epoch, np.ones(window) / window, mode='valid')
        plt.plot(np.arange(window - 1, len(rewards_1epoch)), s_1e,
                 label='n_epochs=1', color='green', linewidth=2)
    plt.axhline(y=150, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (smoothed)")
    plt.title("Ablation: 1 Epoch vs 10 Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Part 7: Observation Questions
#
# **Q1:** In the clipping ablation, did the unclipped PPO diverge, plateau, or converge more
# slowly? Connect your observation to the stale data problem: why does old rollout data become
# harmful when the policy changes a lot between collection and update?

# %% [markdown]
# **Answer Q1:**
# (fill in)

# %% [markdown]
# **Q2:** In the entropy ablation, at what point during training did the no-entropy policy
# begin to diverge from the baseline? What does this tell you about the role of entropy
# as an exploration mechanism vs. a regularizer?

# %% [markdown]
# **Answer Q2:**
# (fill in)

# %% [markdown]
# **Q3:** With n_epochs=1, each rollout of 2048 steps produces a single gradient step of
# batch_size=64. How many gradient steps per environment step does this give you? With n_epochs=10
# and batch_size=64, how many? Why does the clip fraction diagnostic matter more when using
# many epochs?

# %% [markdown]
# **Answer Q3:**
# (fill in)

# %% [markdown]
# ## Part 8: Reflection
#
# 1. **InstructGPT PPO mapping.** Map each PPO component to its RLHF analogue:
#    - Rollout buffer → ?
#    - Clip epsilon → ?
#    - Entropy bonus → ?
#    - Value loss → ?
#    - Advantage normalization → ?
#
# 2. **Explained variance as a training signal.** If EV is consistently negative, what does
#    this say about the critic? What would you do to fix it (architecture, learning rate,
#    coefficient $c_1$)?
#
# 3. **When would you use a separate actor and critic network?** What does sharing a backbone
#    gain and what does it risk? Think about gradient interference between the policy and value
#    losses.

# %% [markdown]
# **Answers:**
# 1.
# 2.
# 3.
