# %% [markdown]
# # Assignment 2: Actor-Critic with GAE
# **Prerequisites:** Read `lecture_notes.md` §5–6 and complete Assignment 1.
#
# **Learning objectives:**
# - Implement a shared-backbone actor-critic network
# - Implement the one-step TD Actor-Critic update (A2C)
# - Implement Generalized Advantage Estimation (GAE)
# - Verify convergence on LunarLander-v2 (mean reward ≥ 150 with λ=0.95)
# - Ablate λ ∈ {0.0, 0.5, 0.95, 1.0} to observe the bias-variance tradeoff

# %% [markdown]
# ## Part 0: Implement rllearn Stubs First
#
# **Before writing any code in this notebook, implement the following:**
#
# ### 1. `ActorCritic` in `rllearn/networks.py`
#
# ```python
# class ActorCritic(nn.Module):
#     def __init__(self, obs_dim, n_actions, hidden_dim=256):
#         super().__init__()
#         # Shared trunk: obs_dim → hidden_dim → ReLU → hidden_dim → ReLU
#         # Actor head: hidden_dim → n_actions  (raw logits)
#         # Critic head: hidden_dim → 1         (scalar value)
#
#     def forward(self, x):
#         # trunk_out = self.trunk(x)
#         # logits = self.actor_head(trunk_out)         shape: (B, n_actions)
#         # value  = self.critic_head(trunk_out).squeeze(-1)  shape: (B,)
#         # return logits, value
# ```
#
# **Why a shared trunk?** The trunk learns a state representation useful for both the policy and
# value function. This improves sample efficiency — the value function gradient helps improve the
# feature extractor.
#
# ### 2. `compute_gae` in `rllearn/utils.py`
#
# ```python
# def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
#     # values has length T+1: values[T] is the bootstrap value for the next state
#     # Iterate backwards from t=T-1 to 0:
#     #   delta_t = rewards[t] + gamma * values[t+1] * (1-dones[t]) - values[t]
#     #   gae_t   = delta_t + gamma * lam * (1-dones[t]) * gae_{t+1}
#     # Return torch.FloatTensor of shape (T,)
# ```
#
# **Critical bug to avoid:** not zeroing out `gae_{t+1}` at episode boundaries with `(1-dones[t])`.
# This mixes advantages across episode boundaries, giving wildly incorrect estimates at the end of
# each episode.

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
# ## Part 1: Theory Recap
#
# **TD advantage (one-step):**
#
# $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
#
# **Advantage function:**
#
# $$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$
#
# **Generalized Advantage Estimation:**
#
# $$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$$
#
# | $\lambda$ | Estimator | Bias | Variance |
# |---|---|---|---|
# | 0 | TD(0): $\delta_t$ alone | High | Low |
# | 1 | Monte Carlo: $G_t - V(s_t)$ | Zero | High |
# | 0.95 | Typical PPO setting | Low | Moderate |
#
# **Actor-Critic key insight:** The critic $V_\phi(s)$ provides a baseline for the actor update.
# By parameterizing $V_\phi$ and learning it alongside $\pi_\theta$, we get a low-variance
# advantage estimate without waiting for full episode returns.

# %% [markdown]
# ## Part 2: Implement the One-Step Actor-Critic Update
#
# **Algorithm:**
# 1. Forward pass: `logits, value = ac_net(obs)` and `_, next_value = ac_net(next_obs)`
# 2. Compute TD advantage: `delta = reward + gamma * next_value * (1 - done) - value`
# 3. **Actor loss:** `-log_pi(a|s) * delta.detach()`
# 4. **Critic loss:** `delta ** 2`
# 5. Total loss: `actor_loss + critic_loss`
# 6. Backward and step.
#
# **Why `delta.detach()` for the actor?**
# Without detach, the gradient of $\delta_t = r + \gamma V(s') - V(s)$ flows into the actor.
# The actor would then try to minimize $V(s)$ to maximize the apparent advantage — causing a
# corrupted objective that destabilizes training. Always detach the TD error before using it
# to weight the log-probability gradient.

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

    TD advantage: delta = r + gamma * V(s') * (1 - done) - V(s)
    Actor loss:   -log_pi(a|s) * delta.detach()
    Critic loss:  delta^2
    Total loss:   actor_loss + critic_loss

    Returns
    -------
    (actor_loss, critic_loss) : scalar float values for logging

    Common mistake: using delta (with gradient) for actor loss — must detach.
    """
    # TODO: forward on obs → logits, value; forward on next_obs → _, next_value
    # TODO: delta = reward + gamma * next_value * (1 - int(done)) - value
    #        (use .squeeze() to get scalar values from (1,)-shaped tensors)
    # TODO: actor_loss = -Categorical(logits=logits).log_prob(action_tensor) * delta.detach()
    # TODO: critic_loss = delta ** 2
    # TODO: loss = actor_loss + critic_loss
    # TODO: zero_grad → backward → step
    # TODO: return actor_loss.item(), critic_loss.item()
    raise NotImplementedError


# %% [markdown]
# ## Part 3: Training Loop
#
# The training loop below is provided. It uses one-step Actor-Critic (no GAE — that's used
# in the PPO assignment). Read through it to understand the step-by-step update structure.

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
#
# Train the one-step Actor-Critic on LunarLander-v2. The agent must achieve a
# **mean episode reward ≥ 150** over the last 100 episodes within 1500 episodes.
#
# **Expected behavior:** reward starts around −200 (random crashes) and climbs toward +200
# (successful landings) as both actor and critic improve jointly.
#
# **Note:** LunarLander-v2 is stochastic. Some seeds may converge faster or slower.
# If you are near but not reaching 150, try seed=0 or seed=7.

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
#
# **Hypothesis:** The choice of $\lambda$ controls the bias-variance tradeoff in advantage
# estimation, affecting learning speed and final performance.
#
# - $\lambda = 0.0$: pure TD(0) — low variance but high bias (if $V$ is imperfect, every step
#   uses a biased advantage).
# - $\lambda = 0.5$: mild lookahead — moderate bias-variance.
# - $\lambda = 0.95$: standard PPO setting — aggressive lookahead with some variance reduction.
# - $\lambda = 1.0$: full Monte Carlo — unbiased but high variance.
#
# We train a batch Actor-Critic (collects a full episode, then does a single GAE-based update)
# to isolate the effect of $\lambda$. The one-step A2C above cannot use GAE directly.

# %%
def train_gae_actor_critic(lam: float,
                            env_id: str = "LunarLander-v2",
                            n_episodes: int = 1500,
                            gamma: float = 0.99,
                            lr: float = 3e-4,
                            hidden_dim: int = 256,
                            seed: int = 42) -> List[float]:
    """
    Train an Actor-Critic with GAE. Returns episode reward list.

    Uses full-episode rollouts to properly apply compute_gae.
    """
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
# ## Part 6: Observation Questions
#
# **Q1:** Based on your plots, which λ value converged fastest? Which had the most stable
# (lowest variance) learning curve? Explain the result in terms of the bias-variance tradeoff.

# %% [markdown]
# **Answer Q1:**
# (fill in)

# %% [markdown]
# **Q2:** At λ=0 (TD(0) only), the advantage estimate has high bias when $V_\phi$ is far from
# the true value function. How does this manifest in the early training curve (first ~300 episodes)?

# %% [markdown]
# **Answer Q2:**
# (fill in)

# %% [markdown]
# **Q3:** At λ=1 (Monte Carlo), what happens in long episodes where one catastrophic action
# causes a crash at the end? How does this affect the gradient for actions taken early in the
# episode? Why is this problematic for training?

# %% [markdown]
# **Answer Q3:**
# (fill in)

# %% [markdown]
# ## Part 7: Reflection
#
# 1. The InstructGPT PPO pipeline uses a token-level MDP where the reward is only given at the
#    final token (end-of-sequence). In terms of λ: does this resemble λ=0 or λ=1? What problem
#    does this cause for credit assignment, and how does the KL penalty help?
#
# 2. In the Actor-Critic, the shared trunk receives gradients from both the actor loss and the
#    critic loss. Could this cause interference? When might it be beneficial to use separate
#    networks for actor and critic?

# %% [markdown]
# **Answers:**
# 1.
# 2.
