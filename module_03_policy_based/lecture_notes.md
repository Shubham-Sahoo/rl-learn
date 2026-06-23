# Module 03: Policy-Based Methods — REINFORCE, Actor-Critic, and PPO

> **How to read this chapter.** Plain-English intuition first, then formal derivation, then equations
> with inline annotations. Each section builds directly on the previous. The Research Bridge at the
> end connects every concept to modern LLM alignment (InstructGPT, RLHF, KL penalties).

---

## 1. Intuition: Why Not Just Learn Q-Values?

### 1.1 The Limits of Value-Based Methods

DQN and its variants learn a Q-function $Q(s, a)$ and derive a policy implicitly: act greedily. This
works well when the action space is small and discrete. But three failure modes appear quickly:

**Continuous actions.** If $a \in \mathbb{R}^d$, computing $\arg\max_a Q(s, a)$ requires solving an
optimization problem at every step. For a 30-DoF robot arm this is expensive at best, intractable
at worst.

**Stochastic policies are sometimes optimal.** In rock-paper-scissors, any deterministic policy loses
to an adversary who observes it. The optimal (Nash equilibrium) policy is $\text{Uniform}(\{R, P, S\})$.
Q-values cannot represent this: $\arg\max Q$ is always deterministic.

**Smooth policy improvement.** A small change to the Q-function can flip the greedy action
discontinuously ($\arg\max$ has zero gradient almost everywhere). Policy gradient methods instead
move the policy parameters smoothly in the direction of higher expected return.

### 1.2 The Policy Gradient Idea

Instead of learning $Q^*$ and deriving $\pi^*$ implicitly, **directly parameterize the policy**:

$$\pi_\theta(a \mid s) = \text{softmax}(f_\theta(s))_a$$

and optimize the **policy objective**:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

where $\tau = (s_0, a_0, r_0, s_1, \ldots)$ is a trajectory and $R(\tau) = \sum_t \gamma^t r_t$ is
the discounted return. We want $\nabla_\theta J(\theta)$ so we can do gradient ascent.

---

## 2. Policy Gradient Theorem

### 2.1 The Log-Derivative Trick

We need $\nabla_\theta \mathbb{E}_{x \sim p_\theta}[f(x)]$ where $p_\theta$ is a parameterized
distribution and $f$ is any function of $x$ (think: return of a trajectory).

$$\nabla_\theta \mathbb{E}_{x \sim p_\theta}[f(x)] = \mathbb{E}_{x \sim p_\theta}[f(x) \cdot \nabla_\theta \log p_\theta(x)]$$

**Derivation:**

$$\nabla_\theta \mathbb{E}_{x \sim p_\theta}[f(x)]
= \nabla_\theta \int f(x)\, p_\theta(x)\, dx
= \int f(x)\, \nabla_\theta p_\theta(x)\, dx$$

Now apply the identity $\nabla_\theta p_\theta(x) = p_\theta(x)\, \nabla_\theta \log p_\theta(x)$
(which follows from the chain rule: $\nabla \log p = \nabla p / p$):

$$= \int f(x)\, p_\theta(x)\, \nabla_\theta \log p_\theta(x)\, dx
= \mathbb{E}_{x \sim p_\theta}\!\left[f(x) \cdot \nabla_\theta \log p_\theta(x)\right]$$

This is remarkable: we converted a gradient of an expectation (hard: the distribution changes with
$\theta$) into an expectation of a gradient (easy: sample trajectories, compute $\nabla_\theta \log \pi_\theta$).

### 2.2 Applying to Policy Gradients

A trajectory $\tau$ has log-probability $\log p_\theta(\tau) = \sum_t \log \pi_\theta(a_t \mid s_t)$
(the environment dynamics cancel because they do not depend on $\theta$).

Applying the log-derivative trick to $J(\theta) = \mathbb{E}_\tau[R(\tau)]$:

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\!\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q^\pi(s_t,a_t)\right]$$

*(Why $Q^\pi$ appears:* The future return from $(s_t, a_t)$ under $\pi_\theta$ is exactly the
action-value $Q^\pi(s_t, a_t)$. Causality allows us to drop past rewards; only future rewards depend
on actions taken from $t$ onward. This is the **Policy Gradient Theorem**.)

The gradient says: **increase the log-probability of action $a_t$ proportional to how good that
action was** (measured by $Q^\pi$).

---

## 3. REINFORCE: Monte Carlo Policy Gradient

### 3.1 The Algorithm

REINFORCE uses the full episode return $G_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$ as an unbiased
estimator of $Q^\pi(s_t, a_t)$:

$$\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$$

**Algorithm:**
1. Roll out a full episode under $\pi_\theta$: $(s_0, a_0, r_0, \ldots, s_T)$.
2. Compute returns $G_t$ backwards: $G_T = r_T$, $G_t = r_t + \gamma G_{t+1}$.
3. Compute policy gradient and update $\theta$.

### 3.2 Why REINFORCE Has High Variance

$G_t$ is a sum of many stochastic rewards — its variance scales with trajectory length. Two
trajectories from the same $(s_t, a_t)$ can produce wildly different $G_t$ values due to future
randomness unrelated to the action taken at $t$.

**Consequence:** Large gradient variance → noisy updates → slow convergence → need many samples.

---

## 4. Baseline Trick: Variance Reduction

### 4.1 Adding a Baseline

We can subtract any function $b(s_t)$ from the return without biasing the gradient:

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\!\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))\right]$$

**Proof that $b$ does not bias the gradient:**

$$\mathbb{E}_\pi\!\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot b(s_t)\right]
= \mathbb{E}_{s_t}\!\left[b(s_t)\, \mathbb{E}_{a_t \sim \pi}[\nabla_\theta \log \pi_\theta(a_t|s_t)]\right]$$

By the log-derivative trick in reverse:
$\mathbb{E}_{a \sim \pi}[\nabla_\theta \log \pi_\theta(a|s)] = \nabla_\theta \sum_a \pi_\theta(a|s) = \nabla_\theta 1 = 0$.

So the baseline term has zero mean and subtracting it only reduces variance — it never biases the
gradient direction.

### 4.2 The Advantage Function

The optimal baseline (in the MSE sense) is $b(s_t) = V^\pi(s_t)$, giving the **advantage**:

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

Interpretation: $A^\pi(s, a)$ measures whether action $a$ is *better than average* in state $s$.
- $A^\pi > 0$: action is better than the mean → increase its probability.
- $A^\pi < 0$: action is worse than the mean → decrease its probability.
- $A^\pi = 0$: action is average → no update.

This centers the signal around zero, dramatically reducing gradient variance.

---

## 5. Actor-Critic: Online Advantage Estimation

### 5.1 TD Advantage (One-Step)

Actor-Critic methods learn a **value function** (the critic) online to estimate the advantage,
enabling updates after every step rather than waiting for an entire episode:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$\delta_t$ is the **TD error** and is an unbiased estimator of $A^\pi(s_t, a_t)$ under the current
policy if $V$ is exact. In practice $V$ is approximated, introducing some bias.

**Actor update:** $\nabla_\theta J \approx \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t$

**Critic update:** Minimize $\delta_t^2$ (or equivalently, fit $V$ to the Bellman target
$r_t + \gamma V(s_{t+1})$).

**Critical implementation note:** The actor loss must use `delta.detach()`. If the gradient of
$\delta_t$ flows into the actor update, the actor will try to minimize $V(s_t)$ (to increase the
apparent advantage) — a catastrophic bug that destabilizes training.

---

## 6. Generalized Advantage Estimation (GAE)

### 6.1 Bias-Variance Tradeoff in Advantage Estimation

| Estimator | Bias | Variance | Notes |
|---|---|---|---|
| TD(0): $\delta_t$ | High (if $V$ wrong) | Low | One-step, fast |
| Monte Carlo: $G_t - V(s_t)$ | Zero | High | Full episode, slow |
| n-step TD | Intermediate | Intermediate | Tradeoff controlled by $n$ |

We want a single estimator that smoothly interpolates the entire spectrum.

### 6.2 GAE Derivation

Define the $k$-step advantage: $\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l r_{t+l} + \gamma^k V(s_{t+k}) - V(s_t)$.

Notice $\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}$ where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.

GAE takes an exponentially weighted average over all $k$:

$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$$

**Boundary cases:**
- $\lambda = 0$: $\hat{A}_t = \delta_t$ — TD(0) advantage (high bias, low variance).
- $\lambda = 1$: $\hat{A}_t = \sum_{l \ge 0} \gamma^l \delta_{t+l} = G_t - V(s_t)$ — Monte Carlo advantage (zero bias, high variance).

**Implementation (backwards pass):**

```python
gae = 0
advantages = []
for t in reversed(range(T)):
    delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
    gae = delta + gamma * lam * (1 - dones[t]) * gae
    advantages.insert(0, gae)
```

The `(1 - dones[t])` term zeros out the bootstrap at episode boundaries — a common source of bugs
if forgotten.

**Typical setting:** $\lambda = 0.95$ provides a good bias-variance balance in practice.

---

## 7. PPO-Clip: Proximal Policy Optimization

### 7.1 The Problem with Large Policy Updates

Gradient ascent on $J(\theta)$ can take a step so large that the policy changes dramatically,
visiting previously unseen states, where the old data provides no reliable gradient signal.
This causes catastrophic performance collapse — after one bad update the policy may never recover.

TRPO (Trust Region Policy Optimization) constrains updates via a KL-divergence constraint:
$\mathbb{E}_{s}[D_\text{KL}(\pi_{\text{old}} \| \pi_\theta)] \le \delta$.
But TRPO requires second-order optimization (expensive).

### 7.2 Probability Ratio

PPO works with the **probability ratio** between new and old policies:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$$

- $r_t = 1$: policy unchanged at this $(s_t, a_t)$.
- $r_t > 1$: new policy assigns higher probability to action $a_t$.
- $r_t < 1$: new policy assigns lower probability to action $a_t$.

The standard policy gradient objective becomes $J(\theta) = \mathbb{E}_t[r_t(\theta) \hat{A}_t]$
(importance-weighted expected advantage).

### 7.3 PPO-Clip Objective

Instead of a hard KL constraint, PPO clips the ratio to prevent large policy changes:

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\bigl(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\bigr)\right]$$

**Intuition:**
- When $\hat{A}_t > 0$ (action was good): we want to increase $r_t$ (raise probability). But
  the clip prevents $r_t > 1 + \varepsilon$ — we can't get too enthusiastic.
- When $\hat{A}_t < 0$ (action was bad): we want to decrease $r_t$ (lower probability). But
  the clip prevents $r_t < 1 - \varepsilon$ — we can't over-punish from a single sample.

The $\min$ always takes the **pessimistic** bound: if the clipped objective is worse, we use it.
This ensures the surrogate is a lower bound on the true objective.

### 7.4 Full PPO Loss

PPO jointly trains actor and critic with an entropy bonus:

$$\mathcal{L} = \mathcal{L}^{CLIP} - c_1 \mathcal{L}^{VF} + c_2 S[\pi_\theta]$$

Where:
- $\mathcal{L}^{VF} = \mathbb{E}_t[(V_\theta(s_t) - V_t^\text{target})^2]$ — value function MSE loss.
- $S[\pi_\theta] = -\mathbb{E}_t[\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)]$ — policy entropy.
- $c_1 \approx 0.5$: value loss coefficient.
- $c_2 \approx 0.01$: entropy bonus coefficient.

The entropy bonus prevents premature convergence to a deterministic policy, maintaining exploration.

### 7.5 PPO Training Loop

1. **Collect rollout:** Run $\pi_{\theta_\text{old}}$ for $T$ steps, storing $(s_t, a_t, r_t, d_t, V_t, \log\pi_t)$.
2. **Compute advantages:** Use GAE with the collected values and rewards.
3. **Normalize advantages:** $\hat{A}_t \leftarrow (\hat{A}_t - \mu) / (\sigma + \varepsilon)$.
4. **Update for $K$ epochs:** Sample random minibatches from the rollout buffer; compute
   $\mathcal{L}$ with the current $\pi_\theta$; gradient step.
5. **Update $\pi_{\theta_\text{old}} \leftarrow \pi_\theta$** and repeat.

Multiple epochs over the same rollout is PPO's key efficiency advantage over vanilla policy gradients.

---

## 8. Failure Modes

### 8.1 No Entropy Bonus → Premature Convergence

Without the $c_2 S[\pi_\theta]$ term, the policy quickly becomes near-deterministic. Once a
suboptimal action dominates, the policy stops exploring, reinforcing the suboptimal behavior.
Symptoms: loss decreases rapidly in early training, then plateaus far from optimal.

### 8.2 No Clipping → Catastrophic Collapse

Without the clip, a single large gradient step can move the policy far from the region where the
advantage estimates are reliable. The new policy then collects data from a completely different
distribution, causing a feedback loop of increasingly bad updates. Catastrophic collapse: policy
performance drops to near-random and never recovers.

### 8.3 Detach Bug in Actor-Critic

If the TD error $\delta_t$ is not detached before computing the actor loss, gradients flow back
through $V(s_t)$ — the actor learns to minimize the state value (making actions look better by
depressing the baseline). Training appears to work initially but is actually maximizing a corrupted
objective. Always use `delta.detach()` for the actor loss.

### 8.4 GAE Episode Boundary Bug

Forgetting to zero out the advantage at episode boundaries (via `(1 - done)`) means GAE mixes
returns across episode boundaries. In environments with terminal rewards, this causes wildly
incorrect advantage estimates, especially for the last few timesteps.

### 8.5 Advantage Normalization Timing

Normalize advantages **across the entire rollout buffer** (not per-minibatch). Per-minibatch
normalization changes the relative scale of advantages between batches and destabilizes training.

---

## 9. Research Bridge

### 9.1 InstructGPT: PPO for Language Models

InstructGPT (Ouyang et al., 2022) fine-tunes GPT-3 with RLHF using the exact PPO algorithm
described above. The setup:

1. **Reward model (RM):** Trained from human pairwise preferences to score (prompt, response) pairs.
2. **Reference policy $\pi_{\text{ref}}$:** The SFT (supervised fine-tuned) model, kept frozen.
3. **PPO objective:** Maximize $\mathbb{E}[r_\phi(\text{prompt}, \text{response})] - \beta \cdot D_\text{KL}(\pi_\theta \| \pi_{\text{ref}})$.

The KL penalty $\beta \cdot D_\text{KL}$ plays the role of the PPO clip — it prevents the policy
from deviating too far from the reference model, limiting reward hacking.

**Token-level MDP:** The "state" at each step is the prompt + tokens generated so far; the "action"
is the next token; the "reward" is zero for intermediate tokens and the RM score at the end-of-sequence
token (with the KL penalty subtracted throughout).

### 9.2 KL Penalty as Soft Trust Region

The KL penalty $\beta \cdot D_\text{KL}(\pi \| \pi_{\text{ref}})$ is a **soft** trust region
constraint: the policy can deviate from $\pi_{\text{ref}}$ as much as the reward justifies, but pays
a cost proportional to the KL divergence. This is mathematically equivalent to the
Lagrangian relaxation of the TRPO hard constraint, with $\beta$ as the Lagrange multiplier.

Increasing $\beta$: the policy stays close to the reference model — safe but possibly undertrained.
Decreasing $\beta$: the policy optimizes the reward model aggressively — risk of reward hacking.

Adaptive KL controllers (used in InstructGPT) tune $\beta$ dynamically to target a desired KL.

### 9.3 Entropy Bonus ↔ LLM Temperature

The entropy bonus $c_2 S[\pi_\theta]$ in PPO and the temperature $T$ in LLM decoding are the same
mechanism viewed from different angles:

- **PPO entropy bonus:** Adds $c_2 H(\pi_\theta(a|s))$ to the objective during training, encouraging
  the learned policy to maintain randomness. Prevents premature determinism.
- **LLM temperature scaling:** Divides logits by $T$ before softmax. $T > 1$: flatter distribution
  (higher entropy, more diverse outputs). $T < 1$: sharper distribution (lower entropy, greedier).

In RLHF: if the entropy bonus is too weak ($c_2 \approx 0$), the policy collapses to always
producing its most-rewarded response regardless of the prompt — a form of mode collapse. The entropy
term acts as a regularizer that maintains the diversity of the SFT prior.

### 9.4 Clip Fraction as a Diagnostic

The clip fraction (fraction of $r_t(\theta)$ outside $[1-\varepsilon, 1+\varepsilon]$) is an
important training diagnostic:

- **Clip fraction ≈ 0:** Policy is not changing much — learning rate may be too small or the
  algorithm is converged.
- **Clip fraction ≈ 0.1–0.3:** Healthy update regime.
- **Clip fraction > 0.5:** Policy is changing too fast — rollout data is becoming stale. Reduce
  learning rate or increase rollout size.

In InstructGPT, the PPO clip fraction is monitored alongside the KL divergence to ensure stable
fine-tuning.

---

*Next: Assignment 1 — REINFORCE on CartPole-v1. Assignment 2 — Actor-Critic with GAE on
LunarLander-v2. Assignment 3 — PPO from scratch on HalfCheetah-v4.*
