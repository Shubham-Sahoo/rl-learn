# %% [markdown]
# # Module 05, Assignment 1: TRPO and the Natural Gradient

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

# %%
class PolicyNet(nn.Module):
    """Softmax policy network for discrete action spaces."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
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


def select_action(policy_net: PolicyNet, obs: np.ndarray) -> Tuple[int, torch.Tensor]:
    """Sample action from policy; return (action, log_prob)."""
    obs_t = torch.FloatTensor(obs)
    logits = policy_net(obs_t)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob


# %% [markdown]
# ## Part 2: Core TRPO Components

# %%
def compute_kl_divergence(policy_net: nn.Module, obs_batch: torch.Tensor,
                          old_log_probs: torch.Tensor) -> torch.Tensor:
    """KL(pi_old || pi_new) estimated from obs_batch."""
    logits_new = policy_net(obs_batch)
    # Build old distribution from old log_probs (we have per-action old log probs via distribution)
    # We re-compute the old distribution by treating old logits as detached current logits
    logits_old = policy_net(obs_batch).detach()
    p_old = torch.softmax(logits_old, dim=-1)
    log_p_old = torch.log_softmax(logits_old, dim=-1)
    log_p_new = torch.log_softmax(logits_new, dim=-1)
    # KL(p_old || p_new) = sum_a p_old(a) * (log_p_old(a) - log_p_new(a))
    kl = (p_old * (log_p_old - log_p_new)).sum(dim=-1).mean()
    return kl


def fisher_vector_product(policy_net: nn.Module, obs_batch: torch.Tensor,
                          vector: torch.Tensor, damping: float = 1e-2) -> torch.Tensor:
    """Compute (F + damping*I) @ vector efficiently via double backprop."""
    logits = policy_net(obs_batch)
    logits_old = logits.detach()
    p_old = torch.softmax(logits_old, dim=-1)
    log_p_old = torch.log_softmax(logits_old, dim=-1)
    log_p_new = torch.log_softmax(logits, dim=-1)
    kl = (p_old * (log_p_old - log_p_new)).sum(dim=-1).mean()

    # First backprop with create_graph=True
    kl_grads = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
    kl_grad_flat = torch.cat([g.view(-1) for g in kl_grads])

    # Second backprop: grad of (kl_grad^T @ vector)
    kl_grad_dot_v = (kl_grad_flat * vector).sum()
    fvp_grads = torch.autograd.grad(kl_grad_dot_v, policy_net.parameters())
    fvp_flat = torch.cat([g.contiguous().view(-1) for g in fvp_grads])

    return fvp_flat + damping * vector


def conjugate_gradient(A_fn, b: torch.Tensor, n_steps: int = 10,
                       damping: float = 1e-2) -> torch.Tensor:
    """Solve Ax = b via conjugate gradient without materializing A."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    r_dot_r = torch.dot(r, r)

    for _ in range(n_steps):
        Ap = A_fn(p)
        alpha = r_dot_r / (torch.dot(p, Ap) + 1e-8)
        x = x + alpha * p
        r_new = r - alpha * Ap
        r_dot_r_new = torch.dot(r_new, r_new)
        if r_dot_r_new < 1e-10:
            break
        beta = r_dot_r_new / (r_dot_r + 1e-8)
        p = r_new + beta * p
        r = r_new
        r_dot_r = r_dot_r_new

    return x


def natural_gradient_step(policy_net: nn.Module, obs_batch: torch.Tensor,
                           policy_loss: torch.Tensor, max_kl: float = 0.01) -> bool:
    """Compute natural gradient direction and do line search."""
    # Vanilla gradient
    g = _get_flat_grad(policy_loss, policy_net, retain_graph=True)

    # Natural gradient via CG
    def A_fn(v):
        return fisher_vector_product(policy_net, obs_batch, v)

    natural_grad = conjugate_gradient(A_fn, g, n_steps=10)

    # Compute step size
    ng_Fng = torch.dot(natural_grad, A_fn(natural_grad))
    step_size = torch.sqrt(2 * max_kl / (ng_Fng + 1e-8))

    # Save old params and old log probs
    old_params = _get_flat_params(policy_net).clone()
    with torch.no_grad():
        logits = policy_net(obs_batch)
        old_log_probs = torch.log_softmax(logits, dim=-1).detach()

    # Backtracking line search
    for i in range(10):
        new_params = old_params + step_size.item() * natural_grad
        _set_flat_params(policy_net, new_params)
        kl = compute_kl_divergence(policy_net, obs_batch, old_log_probs)
        if kl.item() <= max_kl:
            return True
        step_size = step_size * 0.5

    # Restore old params if line search failed
    _set_flat_params(policy_net, old_params)
    return False


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
# ## Part 3: Training Loop

# %%
from rllearn.logging import make_writer


def collect_rollout(policy_net: PolicyNet, env: gym.Env,
                    n_steps: int = 2048, gamma: float = 0.99):
    """Collect a rollout of n_steps transitions."""
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
    """Train with natural gradient or vanilla gradient for comparison."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = PolicyNet(obs_dim, n_actions, hidden_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    writer = make_writer(f"trpo_{'natural' if use_natural else 'vanilla'}_{env_id}")
    all_rewards = []

    for update in range(n_updates):
        rollout = collect_rollout(policy_net, env, n_steps_per_update, gamma)
        obs = rollout["obs"]
        actions = rollout["actions"]
        returns = rollout["returns"]

        loss = compute_policy_loss(policy_net, obs, actions, returns)

        if use_natural:
            accepted = natural_gradient_step(policy_net, obs, loss, max_kl=max_kl)
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accepted = True

        ep_rewards = rollout["episode_rewards"]
        if ep_rewards:
            mean_reward = np.mean(ep_rewards)
            all_rewards.extend(ep_rewards)
            writer.add_scalar("train/episode_reward", mean_reward, update)
            print(f"Update {update+1:3d} | Mean reward: {mean_reward:.1f} | "
                  f"N episodes: {len(ep_rewards)} | Accepted: {accepted}")

    writer.close()
    env.close()
    return all_rewards


# %% [markdown]
# ## Part 4: Verification

# %%
print("Training with Natural Gradient (TRPO) on CartPole-v1...")
ng_rewards = train_natural_gradient(
    env_id="CartPole-v1",
    n_updates=50,
    n_steps_per_update=2048,
    use_natural=True,
    seed=42,
)

if ng_rewards:
    last_mean = np.mean(ng_rewards[-20:]) if len(ng_rewards) >= 20 else np.mean(ng_rewards)
    print(f"Mean reward (last 20 episodes): {last_mean:.1f}")

# %% [markdown]
# ## Part 5: Reflection

# %% [markdown]
# **Answers:**
# 1. The Fisher matrix measures the sensitivity of the distribution to parameter changes.
#    Natural gradient preconditions the euclidean gradient to account for the geometry of the
#    distribution manifold, making each update consistent in KL terms.
# 2. CG solves Fx=g without materializing F (O(n^2) space), using only matrix-vector products
#    which cost O(n) via double backprop. For large networks this is critical.
# 3. The KL constraint ensures the policy does not change too drastically — each update stays
#    within a trust region defined by max_kl. PPO approximates this with a clip, which is cheaper.
