# Module 02: Value-Based Methods — Q-Learning, DQN, and Rainbow

> **How to read this chapter.** Plain-English intuition first, then formal notation, then equations
> with inline annotations. Each section builds on the previous; the Research Bridge at the end ties
> everything to modern LLM alignment work.

---

## 1. Intuition: Temporal Difference Learning

### 1.1 Driving with an Estimate

Imagine driving to an unfamiliar destination. Before you leave you estimate the trip will take 2
hours. After 30 minutes of faster-than-expected traffic you update: "looks like 1.5 hours total."
You did not wait until you arrived — you updated your estimate *mid-journey* using new information.

This is **temporal difference (TD) learning**: update your estimate of the value of a state using
a *later* estimate, before the full return is known. The key equation:

$$\text{new estimate} \leftarrow \text{old estimate} + \alpha \underbrace{[\text{target} - \text{old estimate}]}_{\delta_t: \text{ TD error}}$$

The TD error $\delta_t$ measures how much your current estimate was wrong. A large positive error
means you undervalued the current state; a large negative error means you overvalued it.

### 1.2 MC vs TD: The Fundamental Tradeoff

Two extreme strategies for estimating $Q^\pi$:

**Monte Carlo (MC):** Wait until the episode ends, then use the actual return $G_t$ as the target.

- Unbiased: the target is the true return, no bootstrapping.
- High variance: $G_t$ depends on many stochastic transitions; different episodes give very
  different returns from the same state.
- Requires episodic tasks (episodes must terminate).
- Slow to propagate information: a good reward at step 100 takes 100 episodes to reach step 1.

**Temporal Difference (TD):** Use the *bootstrapped* target $R + \gamma V(s')$ after just one step.

- Low variance: target depends on only one transition, not a full trajectory.
- Biased: $V(s')$ is an approximation; errors in $V$ corrupt the target.
- Works in continuing (non-episodic) tasks.
- Fast propagation: information flows back one step per update.

| Dimension | Monte Carlo | TD(0) |
|---|---|---|
| Target | Actual return $G_t$ | $R + \gamma V(s')$ |
| Bias | None | Yes (bootstrap bias) |
| Variance | High | Low |
| Episodes needed | Full | Single step |
| Convergence | Slower | Faster in practice |

**n-step TD as interpolation.** Use $n$ steps of actual reward before bootstrapping:

$$G_t^{(n)} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})$$

As $n \to \infty$, this converges to the MC return. TD($\lambda$) elegantly mixes all $n$-step
returns using eligibility traces.

---

## 2. Q-Learning: Off-Policy TD Control

### 2.1 The Q-Update

Q-learning maintains a table (or function approximator) $Q(s, a)$ and updates after each step:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\bigl[R + \gamma \max_{a'} Q(s',a') - Q(s,a)\bigr]$$

*(Why $\max_{a'}$? The target uses the best possible action in $s'$, regardless of which action the
behavior policy actually took. This is the Bellman optimality equation evaluated greedily.)*

### 2.2 Why Q-Learning Is Off-Policy

**Off-policy** means the policy used to *collect data* (behavior policy) differs from the policy
being *improved* (target policy).

In Q-learning:
- **Behavior policy** (what the agent actually does): $\epsilon$-greedy — random with prob $\epsilon$,
  greedy otherwise.
- **Target policy** (what we are learning): fully greedy — $\pi(s) = \arg\max_a Q(s,a)$.

The $\max_{a'}$ in the TD target says: "the next-state value is what we could get if we acted
greedily, even though we may have actually acted randomly." This decoupling is what makes
Q-learning off-policy.

**Implication:** Q-learning can learn from data collected by any behavior policy — including a
human expert, a random agent, or a logged dataset. This is the foundation of *offline RL*.

**Convergence guarantee:** Tabular Q-learning converges to $Q^*$ given sufficient exploration and
diminishing learning rate, by the theory of stochastic approximation.

---

## 3. SARSA: On-Policy TD Control

### 3.1 The SARSA Update

SARSA (State, Action, Reward, next-State, next-Action) uses the action *actually taken* in $s'$:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\bigl[R + \gamma Q(s',a') - Q(s,a)\bigr]$$

where $a'$ is sampled from the *same* $\epsilon$-greedy policy.

*(Why is this on-policy? The target $Q(s', a')$ uses $a'$, which is drawn from the behavior policy.
The update evaluates the behavior policy, not a hypothetically greedy one.)*

### 3.2 Q-Learning vs SARSA: Cliff Walk

The difference becomes stark in dangerous environments (e.g., CliffWalking-v0):

- **Q-learning** learns the *optimal* cliff-edge path (more reward, higher risk). During training,
  the $\epsilon$-greedy behavior policy occasionally falls off the cliff — but the Q-update targets
  greedy actions regardless.
- **SARSA** learns a *safer* path that accounts for the exploratory $\epsilon$ actions. The target
  $Q(s', a')$ includes the possibility of random cliff-falls, so SARSA avoids the cliff edge.

| Algorithm | Policy type | Learns | Cliff behavior |
|---|---|---|---|
| Q-learning | Off-policy | Optimal (greedy) policy | Cliff-edge (risky) |
| SARSA | On-policy | Current $\epsilon$-greedy policy | Safe detour |

---

## 4. Deep Q-Network (DQN)

### 4.1 Scaling Beyond Tables

Tabular Q-learning is infeasible when $|S|$ is exponentially large (e.g., Atari: $\sim 10^{20000}$
pixel configurations). DQN replaces the Q-table with a neural network:

$$Q(s,a;\theta) \approx Q^*(s,a)$$

The network maps observations directly to Q-values for all actions.

### 4.2 DQN Loss Function

DQN minimizes the mean-squared TD error:

$$\mathcal{L}(\theta) = \mathbb{E}\bigl[(R + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta))^2\bigr]$$

*(Why $\theta^-$ instead of $\theta$? This is the **target network** — a frozen copy of the online
network that is updated periodically, not at every gradient step. See §4.4.)*

### 4.3 Experience Replay

**Problem:** Consecutive transitions $(s_t, a_t, r_t, s_{t+1})$ are highly correlated. Training on
them in sequence violates the i.i.d. assumption of SGD. The network might overfit to the current
trajectory, catastrophically forgetting earlier parts of the environment.

**Solution:** Maintain a **replay buffer** $\mathcal{D}$ of past transitions (capacity $\sim 10^4$–$10^6$).
At each training step, sample a random mini-batch from $\mathcal{D}$.

Benefits:
1. **Decorrelation:** Randomly sampled transitions are approximately i.i.d.
2. **Sample efficiency:** Each transition is used multiple times (not just once before being discarded).
3. **Stability:** Old transitions from diverse policies provide a smoothly varying learning signal.

**Why off-policy is required:** The replay buffer stores transitions from *past* behavior policies.
Q-learning (off-policy) can use them correctly; on-policy methods (SARSA, PPO) cannot.

### 4.4 Target Network

**Problem:** The TD target $R + \gamma \max_{a'} Q(s', a'; \theta)$ is computed using the same
parameters $\theta$ being updated. This creates a moving-target problem: every gradient step
changes both the predictions AND the targets, causing unstable oscillations or divergence.

**Solution:** Maintain a separate **target network** with parameters $\theta^-$, updated only every
$C$ steps (hard copy: $\theta^- \leftarrow \theta$).

Benefits:
1. **Stable targets:** The TD target is fixed for $C$ steps, giving a stable regression problem.
2. **Breaking correlations:** The online and target networks see slightly different data.

Without the target network, training on Atari typically diverges within thousands of steps.

---

## 5. Double DQN: Fixing Overestimation Bias

### 5.1 The Overestimation Problem

In vanilla DQN, the TD target uses:

$$y = R + \gamma \max_{a'} Q(s', a'; \theta^-)$$

The $\max$ over noisy Q-estimates systematically overestimates: if $Q(s', a'; \theta^-)$ has any
noise, then $\max_{a'} Q(s', a'; \theta^-)$ is biased upward by Jensen's inequality
($\mathbb{E}[\max X_i] \ge \max \mathbb{E}[X_i]$ when $X_i$ are noisy estimators of the same
underlying value).

**Consequence:** The agent overvalues states and actions, causing overconfident Q-values and
suboptimal policies.

### 5.2 The Double DQN Fix

Decouple action *selection* from action *evaluation*:

$$y = R + \gamma\, Q\!\left(s',\, \arg\max_{a'} Q(s',a';\theta);\; \theta^-\right)$$

- **Online network** ($\theta$): selects the best action $a^* = \arg\max_{a'} Q(s', a'; \theta)$.
- **Target network** ($\theta^-$): evaluates that action $Q(s', a^*; \theta^-)$.

*(Why does this reduce overestimation? The two networks are trained on different mini-batches.
Their errors are approximately independent. The action chosen by one network is unlikely to be
the action with the highest noise in the other. The product of two independent errors is smaller
than the square of one error.)*

---

## 6. Dueling DQN: Separating V and A

### 6.1 The Value–Advantage Decomposition

Not all actions matter equally. In many states, the choice of action has little effect on outcome
(e.g., hovering in free space in a racing game). A separate estimate of the state value $V(s)$ and
the action advantage $A(s,a) = Q(s,a) - V(s)$ can be learned more efficiently.

The Dueling architecture produces:

$$Q(s,a;\theta) = V(s;\theta_V) + A(s,a;\theta_A) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a';\theta_A)$$

*(Why subtract the mean advantage? Without this, $V$ and $A$ are not identifiable: adding a
constant to $V$ and subtracting it from all $A$s leaves $Q$ unchanged. Subtracting the mean
advantage forces $\sum_{a'} A(s,a';\theta_A) = 0$, making the decomposition unique.)*

### 6.2 Why It Helps

The shared trunk learns a state representation that is useful for both the value and advantage
heads. In states where all actions are equally good, the advantage head outputs near-zero and the
gradient flows primarily to the value head — a form of automatic curriculum. This speeds up
learning in environments with large "boring" regions of the state space.

---

## 7. Prioritized Experience Replay (PER)

### 7.1 Not All Transitions Are Equally Useful

Uniform sampling wastes compute on transitions the agent already understands well (small TD
error). Transitions with large TD errors represent surprising events — they carry the most
learning signal.

### 7.2 Priority and Sampling

Assign each transition $i$ a priority proportional to its TD error:

$$p_i = |\delta_i| + \varepsilon, \quad P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \quad w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$$

Where:
- $|\delta_i|$: absolute TD error from the last time transition $i$ was sampled.
- $\varepsilon > 0$: small constant ensuring $p_i > 0$ for all transitions (no transition gets
  probability zero).
- $\alpha \in [0,1]$: controls how much prioritization. $\alpha = 0$: uniform sampling.
  $\alpha = 1$: fully greedy prioritization.
- $\beta \in [0,1]$: importance-sampling (IS) exponent. Compensates for the bias introduced by
  non-uniform sampling. Annealed from $\beta_{\text{start}}$ to $1$ over training.
- $w_i$: IS weight used to correct the gradient: multiply each sample's loss by $w_i$.

*(Why IS weights? Non-uniform sampling introduces bias into the SGD gradient estimator. IS weights
correct for this, ensuring convergence to the same fixed point as uniform sampling.)*

---

## 8. Failure Modes

### 8.1 Divergence Without Replay or Target Network

The deadly triad: function approximation + bootstrapping + off-policy data can cause divergence.

| Missing component | Effect |
|---|---|
| No experience replay | Correlated transitions → non-i.i.d. gradients → oscillation |
| No target network | Moving targets → unstable regression → loss explodes |
| Both missing | Nearly always diverges on Atari-scale problems |

Even with both components, DQN can diverge on environments with:
- Sparse rewards (exploration problem)
- Very long episodes (gradients dominated by terminal rewards)
- Aliased observations (partial observability violates Markov assumption)

### 8.2 Exploration Inadequacy

Epsilon-greedy exploration is naive: it explores uniformly at random, ignoring which states are
most informative. In large state spaces the agent may never reach high-reward states during
training. Fixes: count-based exploration, intrinsic curiosity, noisy networks.

### 8.3 Q-Overestimation as a Silent Bug

Q-overestimation can appear to make training "work" — the Q-values rise, the policy improves
initially — then plateau or regress as the inflated Q-values cause the agent to act overconfidently
and stop exploring. This is especially insidious because the loss may appear low while the policy
is degrading.

---

## 9. Research Bridge

### 9.1 Reward Hacking and Q-Overestimation

Reward hacking in RL and Q-overestimation share the same root: the agent exploits quirks of the
reward signal (or its estimate) that are not aligned with the true objective.

In Q-overestimation: the agent believes certain state-action pairs are more valuable than they are,
leading it to pursue those states aggressively — even at the cost of true performance.

In RLHF reward hacking: a language model learns to produce responses that score highly on the
reward model (which is an approximation of human preference) but do not actually satisfy human
intent. The reward model, like $Q(\theta^-)$, is a frozen or slowly updated approximation — and
the policy exploits its blind spots.

**The connection:** Double DQN's fix (decoupling action selection from evaluation using separate
networks) has an analogue in RLHF: **KL penalties** prevent the policy from straying too far from
the reference model, limiting its ability to exploit the reward model.

### 9.2 Offline RL for LLMs: CQL and Conservative Q-Functions

In **offline RL**, the agent cannot interact with the environment — it must learn entirely from a
fixed logged dataset. This is exactly the setting for RLHF fine-tuning of LLMs: you have a dataset
of (prompt, response, reward) tuples collected with some behavior policy.

Vanilla Q-learning applied to offline data overestimates Q-values for out-of-distribution (OOD)
actions — actions that appear rarely or never in the dataset. The agent then acts on these inflated
values during deployment, leading to poor performance.

**Conservative Q-Learning (CQL)** adds a regularization term that penalizes high Q-values for OOD
actions:

$$\mathcal{L}_{\text{CQL}}(\theta) = \mathcal{L}_{\text{DQN}}(\theta) + \lambda \cdot \mathbb{E}_{s \sim \mathcal{D}}\left[\log \sum_a \exp Q(s,a;\theta) - \mathbb{E}_{a \sim \pi_\beta(a|s)}[Q(s,a;\theta)]\right]$$

This forces Q-values to be low for unseen actions, preventing overconfident exploitation. CQL
variants are directly applicable to offline LLM fine-tuning, where the "actions" are token
sequences and the "dataset" is a preference-labeled corpus.

---

*Next: Assignment 1 — tabular Q-learning on Taxi-v3. Assignment 2 — DQN on CartPole-v1.
Assignment 3 — Rainbow components on LunarLander-v2.*
