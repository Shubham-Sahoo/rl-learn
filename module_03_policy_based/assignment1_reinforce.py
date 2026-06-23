# %% [markdown]
# # Assignment 1: REINFORCE — Monte Carlo Policy Gradient
# **Prerequisites:** Read `lecture_notes.md` §1–4 before starting.
#
# **Learning objectives:**
# - Implement a softmax policy network for discrete action spaces
# - Implement the REINFORCE update (Monte Carlo policy gradient)
# - Verify convergence on CartPole-v1 (mean reward ≥ 450 in last 50 episodes)
# - Understand the effect of return normalization via ablation

# %% [markdown]
# ## Part 0: Implement rllearn Stubs First
#
# **Before writing any code in this notebook, implement the following in `rllearn/utils.py`:**
#
# ```python
# def compute_returns(rewards: list[float], gamma: float) -> list[float]:
#     """G_t = r_t + gamma * r_{t+1} + ... """
#     # Work backwards: G_T = r_T, G_t = r_t + gamma * G_{t+1}
# ```
#
# **Hint:** Use a backward loop. Start with `G = 0`, then for each timestep $t$ from $T-1$
# down to $0$: `G = rewards[t] + gamma * G`. Return the list in forward order.
#
# The function should handle an empty reward list (return `[]`) and a single-element list
# (return `[rewards[0]]`).

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym
from typing import List, Tuple

# Import our stub (you must implement it first)
from rllearn.utils import compute_returns

# %% [markdown]
# ## Part 1: Theory Recap
#
# **Policy objective:**
#
# $$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$
#
# **Policy Gradient Theorem (log-derivative trick applied):**
#
# $$\nabla_\theta J(\theta) = \mathbb{E}_\pi\!\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q^\pi(s_t,a_t)\right]$$
#
# **REINFORCE estimate** (Monte Carlo, $G_t$ as proxy for $Q^\pi$):
#
# $$\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$$
#
# **Why does REINFORCE have high variance?** $G_t$ is a sum of many stochastic future rewards.
# From the same $(s_t, a_t)$, different future trajectories can yield very different $G_t$ values
# due to noise in $r_{t+1}, r_{t+2}, \ldots$ — noise that has nothing to do with the action at $t$.
#
# **Variance reduction (baseline trick):** Subtract $V^\pi(s_t)$ from $G_t$. This does not bias
# the gradient (because $\mathbb{E}_{a \sim \pi}[\nabla_\theta \log \pi_\theta(a|s)] = 0$), but
# centers the updates around zero, reducing their variance.

# %% [markdown]
# ## Part 2: Implement the Policy Network
#
# Implement a two-hidden-layer MLP that maps observations to action logits.
#
# **Architecture:** `obs_dim → hidden_dim → hidden_dim → n_actions`
# - ReLU activations between layers
# - **No softmax** on the output — raw logits. The sampling step applies softmax implicitly
#   via `torch.distributions.Categorical(logits=...)`.
#
# **Why no softmax here?** `Categorical(logits=...)` applies log-softmax internally for numerical
# stability. Applying softmax twice would change the distribution.

# %%
class PolicyNet(nn.Module):
    """Softmax policy network for discrete actions."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        """
        Parameters
        ----------
        obs_dim    : dimension of the observation space
        n_actions  : number of discrete actions
        hidden_dim : width of each hidden layer
        """
        super().__init__()
        # TODO: build a 2-layer MLP with ReLU activations
        # Architecture: obs_dim → hidden_dim → ReLU → hidden_dim → ReLU → n_actions
        # Use nn.Sequential with nn.Linear and nn.ReLU layers
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return action logits of shape (batch, n_actions)."""
        raise NotImplementedError


# %% [markdown]
# ## Part 3: Implement Action Selection
#
# **Steps:**
# 1. Convert `obs` (numpy array) to a float tensor: `torch.FloatTensor(obs)`
# 2. Forward through the policy network to get logits
# 3. Create `torch.distributions.Categorical(logits=logits)`
# 4. Sample an action: `dist.sample()`
# 5. Compute `dist.log_prob(action)` — this is what we differentiate through
#
# **Return:** `(action.item(), log_prob)` where `log_prob` is a scalar tensor with gradient.

# %%
def select_action(policy_net: PolicyNet, obs: np.ndarray) -> Tuple[int, torch.Tensor]:
    """Sample action from policy; return (action, log_prob).

    The returned log_prob tensor must retain its gradient for backprop through REINFORCE.
    """
    # TODO: obs → FloatTensor → forward → Categorical(logits=...) → sample → log_prob
    raise NotImplementedError


# %% [markdown]
# ## Part 4: Implement the REINFORCE Update
#
# **Steps:**
# 1. Convert `returns` to a FloatTensor
# 2. **Normalize returns:** subtract mean, divide by (std + 1e-8). This is a simple but effective
#    variance reduction technique (equivalent to a running-mean baseline when averaged over many
#    episodes).
# 3. Compute the loss: `loss = -sum(log_prob * return for each timestep)`.
#    The negative sign turns gradient ascent into gradient descent (standard convention).
# 4. Call `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.
# 5. Return the scalar loss value: `loss.item()`.
#
# **Common mistakes:**
# - Forgetting the negative sign → gradient descent on reward (training diverges)
# - Not stacking log_probs into a tensor before multiplying → shape mismatch

# %%
def reinforce_update(optimizer: optim.Optimizer,
                     log_probs: List[torch.Tensor],
                     returns: List[float]) -> float:
    """
    REINFORCE gradient update.

    Loss = -sum_t log_pi(a_t|s_t) * G_t   (negative for gradient ascent via descent)

    Parameters
    ----------
    optimizer : the policy network optimizer
    log_probs : list of scalar tensors from select_action, one per timestep
    returns   : list of discounted returns from compute_returns, one per timestep

    Returns
    -------
    float : the scalar loss value (for logging)
    """
    # TODO: convert returns to FloatTensor
    # TODO: normalize returns: (returns - mean) / (std + 1e-8)
    # TODO: stack log_probs into a single tensor
    # TODO: loss = -torch.sum(log_probs_tensor * normalized_returns)
    # TODO: zero_grad → backward → step
    # TODO: return loss.item()
    raise NotImplementedError


# %% [markdown]
# ## Part 5: Training Loop
#
# The training loop below is provided. Read through it to understand the episode structure
# before running the verification cell.

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
#
# Train REINFORCE on CartPole-v1. The agent must achieve a **mean episode reward ≥ 450** over
# the last 50 episodes within 500 episodes.
#
# **Expected behavior:** reward starts around 20–50 (random policy) and climbs to 490–500
# (near-perfect balance) as the policy gradient updates improve the policy.

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
#
# **Hypothesis:** Normalizing the returns reduces gradient variance, leading to more stable
# training (lower std of gradient norms over time).
#
# We compare:
# - **Normalized:** the default `reinforce_update` (subtracts mean, divides by std)
# - **Unnormalized:** a variant that skips normalization

# %%
def reinforce_update_no_norm(optimizer: optim.Optimizer,
                              log_probs: List[torch.Tensor],
                              returns: List[float]) -> Tuple[float, float]:
    """REINFORCE update WITHOUT return normalization.

    Returns (loss_value, grad_norm) where grad_norm is the L2 norm of all gradients
    after the backward pass (before the optimizer step zeros them).
    """
    returns_t = torch.FloatTensor(returns)
    log_probs_t = torch.stack(log_probs)
    loss = -torch.sum(log_probs_t * returns_t)

    optimizer.zero_grad()
    loss.backward()

    # Capture gradient norm before stepping
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

# Episode rewards
axes[0].plot(smooth(rewards_norm, 20), label="With normalization", color='steelblue')
axes[0].plot(smooth(rewards_no_norm, 20), label="Without normalization", color='darkorange')
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Episode Reward (smoothed)")
axes[0].set_title("Return Normalization: Episode Rewards")
axes[0].legend()

# Gradient norms
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
# ## Part 8: Observation Questions
#
# Answer the questions in the markdown cells below.
#
# **Q1:** Why does return normalization reduce gradient variance? What statistical property of $G_t$
# is the culprit when normalization is absent?

# %% [markdown]
# **Answer Q1:**
# (fill in)

# %% [markdown]
# **Q2:** Normalizing returns changes the *effective* learning rate for each episode (long episodes
# with large $G_t$ values get their gradient scaled down). Does this constitute a baseline in the
# theoretical sense? What is the difference between normalization and subtracting $V^\pi(s_t)$?

# %% [markdown]
# **Answer Q2:**
# (fill in)

# %% [markdown]
# **Q3:** REINFORCE requires waiting until the end of an episode before updating. What does this
# mean for environments with very long episodes (e.g., MuJoCo locomotion for 1000 steps)? How
# does Actor-Critic address this?

# %% [markdown]
# **Answer Q3:**
# (fill in)

# %% [markdown]
# ## Part 9: Reflection
#
# 1. In RLHF, each "episode" is a single prompt-response pair. The "return" $G_0$ is the reward
#    model score for the entire response. How does this compare to REINFORCE's Monte Carlo estimate?
#    What is the analogue of high variance in this context?
#
# 2. The REINFORCE gradient at step $t$ uses $G_t = \sum_{t' \ge t} \gamma^{t'-t} r_{t'}$, not
#    $G_0$. Why does causality allow us to drop past rewards $r_0, \ldots, r_{t-1}$? What would
#    go wrong if we used $G_0$ for all timesteps?

# %% [markdown]
# **Answers:**
# 1.
# 2.
