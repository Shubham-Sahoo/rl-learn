# %% [markdown]
# # Module 05, Assignment 1: TRPO and the Natural Gradient
#
# ## Prerequisites
# - Module 03 A1: REINFORCE (PolicyNet and select_action)
# - Lecture notes sections 2–4
#
# ## Learning Objectives
# 1. Understand Fisher Information Matrix as a Riemannian metric
# 2. Implement conjugate gradient for F^{-1}g without materializing F
# 3. Implement Fisher-vector products via double backprop
# 4. Run a natural gradient update with line search

# %% [markdown]
# ## Part 0: Theory Recap
#
# Vanilla gradient descent in parameter space does **not** correspond to consistent steps
# in policy space. Two parameter vectors that are numerically close can produce wildly
# different distributions over actions.
#
# The **natural gradient** corrects for this by rescaling the gradient with the inverse
# Fisher Information Matrix:
#
# $$\tilde{\nabla}J = F^{-1}\nabla J$$
#
# **TRPO** makes this practical via:
# 1. A constrained surrogate objective with a KL trust region.
# 2. Conjugate gradient to compute $F^{-1}g$ without ever materializing $F$.
# 3. A backtracking line search to enforce the KL constraint.

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import Tuple, List

# %% [markdown]
# ## Part 1: Policy Network
#
# We reuse the same PolicyNet architecture from Module 03 A1.
# Architecture: `obs_dim → hidden_dim → ReLU → hidden_dim → ReLU → n_actions` (logits).

# %%
class PolicyNet(nn.Module):
    """Softmax policy network for discrete action spaces."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        # TODO: build a 2-layer MLP with ReLU activations (same as Module 03 A1)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return action logits of shape (batch, n_actions)."""
        raise NotImplementedError


def select_action(policy_net: PolicyNet, obs: np.ndarray) -> Tuple[int, torch.Tensor]:
    """Sample action from policy; return (action, log_prob).

    Identical to Module 03 A1: obs → FloatTensor → forward → Categorical(logits=...) → sample.
    """
    # TODO: mirror Module 03 A1 select_action
    raise NotImplementedError


# %% [markdown]
# ## Part 2: Core TRPO Components
#
# Implement the following four functions in order. Each one builds on the previous.
#
# **Order of implementation:**
# 1. `compute_kl_divergence` — needed by `natural_gradient_step` for line search.
# 2. `fisher_vector_product` — needed by `conjugate_gradient`.
# 3. `conjugate_gradient` — solves $Fx = b$ using FVP.
# 4. `natural_gradient_step` — orchestrates CG + line search.

# %%
def compute_kl_divergence(policy_net: nn.Module, obs_batch: torch.Tensor,
                          old_log_probs: torch.Tensor) -> torch.Tensor:
    """KL(pi_old || pi_new) estimated from obs_batch.

    Steps:
    1. Forward pass through policy_net to get current logits.
    2. Create Categorical distribution from current logits.
    3. old_log_probs are already computed; you need old probabilities = exp(old_log_probs).
    4. KL(p || q) = sum_a p(a) * (log p(a) - log q(a))
       But for a batch: mean over observations of sum_a p_old(a|s) * log(p_old(a|s)/p_new(a|s))

    Hint: Use torch.distributions.Categorical.log_prob to get log q(a) for all actions.
    Or compute via: kl = sum_a softmax(old_logits) * (log_softmax(old_logits) - log_softmax(new_logits))
    Since we don't have old_logits, use: kl = mean over obs of KL between old and new distributions.

    Common mistake: confusing KL(old || new) with KL(new || old). TRPO uses KL(old || new).
    """
    raise NotImplementedError


def conjugate_gradient(A_fn, b: torch.Tensor, n_steps: int = 10,
                       damping: float = 1e-2) -> torch.Tensor:
    """
    Solve Ax = b via conjugate gradient without materializing A.

    A_fn: callable that computes A @ v (Fisher-vector product).
    Used to compute F^{-1} * gradient.

    Algorithm:
        x = 0
        r = b - A(x) = b
        p = r
        for k in range(n_steps):
            Ap = A_fn(p)
            alpha = (r^T r) / (p^T Ap)
            x = x + alpha * p
            r_new = r - alpha * Ap
            beta = (r_new^T r_new) / (r^T r)
            p = r_new + beta * p
            r = r_new

    *(Why CG? It solves Ax=b in at most n iterations for an n-dimensional system,
    using only matrix-vector products. For large neural nets, n can be large but
    10-20 CG steps often give a good approximation.)*
    """
    raise NotImplementedError


def fisher_vector_product(policy_net: nn.Module, obs_batch: torch.Tensor,
                          vector: torch.Tensor, damping: float = 1e-2) -> torch.Tensor:
    """Compute (F + damping*I) @ vector efficiently via double backprop.

    Steps:
    1. Compute KL(pi_old || pi_new) where pi_old = pi_theta (current policy, treated as fixed).
       Since we want the FIM at the current params, compute:
       kl = mean over obs of KL(Categorical(logits_detached) || Categorical(logits))
       where logits_detached = policy_net(obs_batch).detach()
    2. Compute kl_grad = grad(kl, policy_net.parameters(), create_graph=True)
    3. Flatten kl_grad into a single vector (same shape as `vector`).
    4. Compute kl_grad_dot_v = (kl_grad_flat * vector).sum()
    5. Compute fvp = grad(kl_grad_dot_v, policy_net.parameters())
    6. Flatten fvp and add damping: fvp_flat + damping * vector

    *(Why double backprop? The FIM is E[(grad log pi)(grad log pi)^T], and
    Fv = E[(grad log pi)(grad log pi)^T v]. The gradient of (grad log pi)^T v
    with respect to theta gives exactly Fv without materializing F.
    Using KL is equivalent because the Hessian of KL at theta_old = theta equals the FIM.)*

    Note: use torch.autograd.grad with create_graph=True for the first backprop.
    """
    raise NotImplementedError


def natural_gradient_step(policy_net: nn.Module, obs_batch: torch.Tensor,
                           policy_loss: torch.Tensor, max_kl: float = 0.01) -> bool:
    """
    Compute natural gradient direction and do line search.
    Returns True if update was accepted (KL constraint satisfied).

    Steps:
    1. Compute vanilla gradient g = grad(policy_loss, params), flattened.
    2. Define A_fn = lambda v: fisher_vector_product(policy_net, obs_batch, v).
    3. natural_grad = conjugate_gradient(A_fn, g).
    4. Compute step_size: scale natural_grad so that 0.5 * ng^T F ng = max_kl.
       step_size = sqrt(2 * max_kl / (ng^T @ A_fn(ng) + 1e-8))
    5. Save old params (flat copy).
    6. Backtracking line search (up to 10 steps, each halving the step):
       - Set new_params = old_params + step_size * natural_grad
       - Load new_params into policy_net
       - Compute KL = compute_kl_divergence(policy_net, obs_batch, old_log_probs)
       - If KL <= max_kl: accept and return True
       - Else: halve step_size and try again
    7. If all line search steps fail: restore old params and return False.

    Helper: use _get_flat_params / _set_flat_params below.
    """
    raise NotImplementedError


# %% [markdown]
# ### Helper utilities for flat parameter manipulation

# %%
def _get_flat_params(model: nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a single 1D tensor."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def _set_flat_params(model: nn.Module, flat_params: torch.Tensor):
    """Load a flat parameter tensor back into model parameters (in-place)."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset + numel].view_as(p))
        offset += numel


def _get_flat_grad(loss: torch.Tensor, model: nn.Module,
                   retain_graph: bool = False) -> torch.Tensor:
    """Compute gradients of loss w.r.t. model params, return as flat vector."""
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=retain_graph)
    return torch.cat([g.view(-1) for g in grads])


# %% [markdown]
# ## Part 3: Training Loop — Natural Gradient vs Vanilla Gradient
#
# The training loop below is provided. Read through it to understand how natural_gradient_step
# fits into a policy gradient loop.
#
# **TensorBoard metrics:**
# - `train/episode_reward`
# - `train/kl_divergence`
# - `train/step_size`
#
# **Expected observation:** Natural gradient produces more consistent KL steps per update.

# %%
%load_ext tensorboard
%tensorboard --logdir runs/

# %%
from rllearn.logging import make_writer


def collect_rollout(policy_net: PolicyNet, env: gym.Env,
                    n_steps: int = 2048, gamma: float = 0.99):
    """Collect a rollout of n_steps transitions.

    Returns dict with: obs, actions, log_probs, returns, episode_rewards.
    """
    obs_list, action_list, log_prob_list, reward_list, done_list = [], [], [], [], []
    episode_rewards = []
    ep_reward = 0.0

    obs, _ = env.reset()
    for _ in range(n_steps):
        action, log_prob = select_action(policy_net, obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        obs_list.append(obs)
        action_list.append(action)
        log_prob_list.append(log_prob.detach())
        reward_list.append(float(reward))
        done_list.append(done)
        ep_reward += float(reward)

        if done:
            episode_rewards.append(ep_reward)
            ep_reward = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

    # Compute discounted returns (no bootstrapping — treat truncation as done)
    returns = []
    G = 0.0
    for r, d in zip(reversed(reward_list), reversed(done_list)):
        if d:
            G = 0.0
        G = r + gamma * G
        returns.insert(0, G)

    obs_tensor = torch.FloatTensor(np.array(obs_list))
    old_log_probs = torch.stack(log_prob_list)
    returns_tensor = torch.FloatTensor(returns)
    # Normalize returns for variance reduction
    returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

    return {
        "obs": obs_tensor,
        "actions": torch.LongTensor(action_list),
        "old_log_probs": old_log_probs,
        "returns": returns_tensor,
        "episode_rewards": episode_rewards,
    }


def compute_policy_loss(policy_net: PolicyNet, obs: torch.Tensor,
                        actions: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """REINFORCE loss: -mean(log_pi(a|s) * G)."""
    logits = policy_net(obs)
    dist = torch.distributions.Categorical(logits=logits)
    log_probs = dist.log_prob(actions)
    return -(log_probs * returns).mean()


def train_natural_gradient(
    env_id: str = "CartPole-v1",
    n_updates: int = 50,
    n_steps_per_update: int = 2048,
    gamma: float = 0.99,
    max_kl: float = 0.01,
    hidden_dim: int = 64,
    seed: int = 42,
    use_natural: bool = True,
    lr: float = 3e-4,
):
    """Train policy with natural gradient (TRPO-style) or vanilla gradient.

    Returns (episode_rewards_all, kl_list).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PolicyNet(obs_dim, n_actions, hidden_dim)
    label = "natural" if use_natural else "vanilla"
    writer = make_writer(f"trpo_{label}_{env_id}")
    optimizer = optim.Adam(policy.parameters(), lr=lr) if not use_natural else None

    all_episode_rewards = []
    kl_list = []
    global_step = 0

    for update in range(n_updates):
        # Collect rollout
        rollout = collect_rollout(policy, env, n_steps=n_steps_per_update, gamma=gamma)
        obs = rollout["obs"]
        actions = rollout["actions"]
        old_log_probs = rollout["old_log_probs"]
        returns = rollout["returns"]

        if rollout["episode_rewards"]:
            mean_reward = np.mean(rollout["episode_rewards"])
            all_episode_rewards.extend(rollout["episode_rewards"])
            writer.add_scalar("train/episode_reward", mean_reward, global_step)
            print(f"Update {update+1:3d} | Mean ep reward: {mean_reward:.1f}")

        # Compute loss
        loss = compute_policy_loss(policy, obs, actions, returns)

        if use_natural:
            # Natural gradient step
            accepted = natural_gradient_step(policy, obs, loss, max_kl=max_kl)
            if not accepted:
                print(f"  WARNING: line search failed at update {update+1}")
        else:
            # Vanilla gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Measure KL after update
        with torch.no_grad():
            kl = compute_kl_divergence(policy, obs, old_log_probs).item()
        kl_list.append(kl)
        writer.add_scalar("train/kl_divergence", kl, global_step)
        global_step += n_steps_per_update

    env.close()
    writer.close()
    return all_episode_rewards, kl_list


# %% [markdown]
# Run both training runs. This may take a few minutes.

# %%
print("Training with NATURAL gradient...")
rewards_nat, kl_nat = train_natural_gradient(use_natural=True, n_updates=50, seed=42)

print("\nTraining with VANILLA gradient...")
rewards_van, kl_van = train_natural_gradient(use_natural=False, n_updates=50, seed=42)


# %% [markdown]
# ## Part 4: Ablation — KL Divergence per Update
#
# Plot the KL divergence per update for both methods.
# **Expected:** Natural gradient maintains a nearly constant KL (~max_kl=0.01),
# while vanilla gradient takes inconsistent steps.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Episode rewards
def smooth(values, window=5):
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window)/window, mode='valid')

axes[0].plot(smooth(rewards_nat, 5), label="Natural gradient", color='steelblue')
axes[0].plot(smooth(rewards_van, 5), label="Vanilla gradient", color='darkorange')
axes[0].set_xlabel("Episode (smoothed)")
axes[0].set_ylabel("Episode Reward")
axes[0].set_title("Natural vs Vanilla Gradient: Rewards")
axes[0].legend()

# KL divergence per update
axes[1].plot(kl_nat, label="Natural gradient", color='steelblue', marker='o', markersize=3)
axes[1].plot(kl_van, label="Vanilla gradient", color='darkorange', marker='o', markersize=3)
axes[1].axhline(y=0.01, color='gray', linestyle='--', label="max_kl=0.01")
axes[1].set_xlabel("Update step")
axes[1].set_ylabel("KL Divergence")
axes[1].set_title("KL Divergence per Update")
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Natural gradient KL std: {np.std(kl_nat):.5f}")
print(f"Vanilla gradient KL std: {np.std(kl_van):.5f}")
print(f"Ratio (vanilla/natural): {np.std(kl_van)/(np.std(kl_nat)+1e-8):.2f}x")

# %% [markdown]
# ## Part 5: Reflection
#
# Answer the questions below.
#
# **Q1:** Why does vanilla gradient descent take inconsistent steps in policy space?
# Consider what happens when the policy is near-deterministic (one action gets nearly
# all probability) vs. when it is near-uniform.

# %% [markdown]
# **Answer Q1:**
# (fill in)

# %% [markdown]
# **Q2:** How does TRPO relate to PPO? TRPO uses a hard KL constraint solved via conjugate
# gradient. PPO uses a soft constraint via clipping. What are the tradeoffs of each approach
# in terms of computation, implementation complexity, and constraint enforcement?

# %% [markdown]
# **Answer Q2:**
# (fill in)

# %% [markdown]
# **Q3:** The Fisher-vector product requires double backpropagation. Describe what
# `create_graph=True` does in `torch.autograd.grad` and why it is necessary here.

# %% [markdown]
# **Answer Q3:**
# (fill in)
