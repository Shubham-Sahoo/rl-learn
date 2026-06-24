# Module 4: Model-Based Reinforcement Learning

> **Alignment relevance:** Chain-of-thought reasoning is a form of world model simulation — the model "plans" through intermediate steps before committing to an answer, trading compute for better final-step accuracy in exactly the way Dyna-Q trades planning steps for fewer environment interactions.

---

## 1. Intuition

### 1.1 The Sample Efficiency Problem

Every real-environment step has a cost: robot wear, API fees, human time, safety risk. Model-free methods like DQN and PPO are *data hungry* — DQN needs millions of Atari frames to match human-level play, while a human child learns Breakout in minutes. The core issue is that model-free agents discard causal structure. Each $(s, a, r, s')$ tuple is treated as an opaque data point; the agent never builds an understanding of *why* $s'$ followed $a$ in $s$.

**Model-based RL** inserts a learned or known transition function into the loop. Instead of waiting for the real environment to supply $10^6$ transitions, the agent queries its internal model millions of times at negligible cost. If the model is good enough, policy improvement from synthetic data approximates policy improvement from real data.

### 1.2 The Tradeoff: Sample Efficiency vs. Model Error

A perfect model gives unlimited free training signal. Real models are imperfect — they are fit on limited data and generalize imperfectly to novel $(s, a)$ pairs. The two failure modes pull in opposite directions:

- **Too few real steps:** The model was trained on too little data and its predictions are wrong. Planning against a bad model yields a policy that exploits the model's errors (model exploitation / Dyna bug).
- **Too many model rollouts:** Even a small per-step error compounds exponentially over long rollouts. A 1% per-step error becomes a 10% error over 10 steps, 63% over 100 steps.

The practical sweet spot depends on the model quality and the horizon used for planning. Short rollouts with a decent model beat pure model-free in most continuous control benchmarks (MBPO, Dreamer).

### 1.3 Two Paradigms

**Dyna-style** (this module, §2): Use a tabular or shallow model for *background planning* — interleave real experience with synthetic Q-updates from randomly sampled model transitions.

**Latent world models** (§3–4): Learn a compressed latent state space and plan inside it. The model learns *abstract* dynamics; the policy never sees raw observations during planning.

---

## 2. Dyna-Q

### 2.1 Algorithm

Dyna-Q (Sutton 1990) augments tabular Q-learning with $n$ planning steps per real environment step. The model $\mathcal{M}$ is a lookup table: $\mathcal{M}(s, a) \mapsto (r, s')$.

**Pseudocode:**

```
Initialize Q(s, a) = 0 for all s, a
Initialize model M = {}

for each real step:
    observe s; choose a ← ε-greedy(Q, s)
    execute a; observe r, s'
    
    # Direct RL update (real experience)
    Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]
    
    # Model update
    M(s, a) ← (r, s')
    
    # Planning: n synthetic Q-updates
    for i in 1..n:
        s̃, ã ← random sample from M.keys()
        r̃, s̃' ← M(s̃, ã)
        Q(s̃, ã) ← Q(s̃, ã) + α [r̃ + γ max_{a'} Q(s̃', a') - Q(s̃, ã)]
```

### 2.2 Q-Update Equation

The direct RL update and each planning update share the same Bellman target:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \underbrace{\bigl[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\bigr]}_{\text{TD error } \delta}$$

*(Why? Because the Bellman optimality equation says $Q^*(s,a) = r + \gamma \max_{a'} Q^*(s', a')$. The TD error $\delta$ is the gap between the current estimate and the target; we take a step $\alpha$ in the direction that closes this gap.)*

### 2.3 Effect of Planning Steps

With $n = 0$, Dyna-Q reduces to standard Q-learning. With $n > 0$, each real step propagates information to $n$ additional $(s, a)$ pairs — value information "diffuses" through the Q-table much faster. Empirically, $n = 50$ planning steps on a simple maze reduces real steps to convergence by roughly 50×.

$$\text{Effective data} \approx T_{\text{real}} \times (1 + n)$$

*(Why only approximately? Because synthetic transitions are not i.i.d. — they are re-sampled from a fixed model that may be stale or incorrect. As the policy improves, previously visited $(s, a)$ may never be visited again, yet they still contribute to planning.)*

---

## 3. World Models

### 3.1 Learning the Dynamics

In continuous state spaces, a tabular model is impractical. We instead learn parameterized functions:

$$f_\phi : (s_t, a_t) \mapsto \hat{s}_{t+1}$$

$$g_\phi : (s_t, a_t) \mapsto \hat{r}_t$$

where $f_\phi$ is the **transition model** and $g_\phi$ is the **reward model**. Both are typically MLPs trained by supervised regression on replay buffer data.

### 3.2 Transition Loss

$$\mathcal{L}_{\text{transition}}(\phi) = \mathbb{E}_{(s_t, a_t, s_{t+1}) \sim \mathcal{D}}\!\left[\|f_\phi(s_t, a_t) - s_{t+1}\|^2\right]$$

*(Why MSE? Under a Gaussian noise model $s_{t+1} = f_\phi(s_t, a_t) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$, the maximum-likelihood estimator is exactly the MSE minimizer. For heteroscedastic noise, you would learn a variance head and use negative log-likelihood instead.)*

### 3.3 Reward Loss

$$\mathcal{L}_{\text{reward}}(\phi) = \mathbb{E}_{(s_t, a_t, r_t) \sim \mathcal{D}}\!\left[(g_\phi(s_t, a_t) - r_t)^2\right]$$

### 3.4 Latent Space Planning

Raw pixel observations are high-dimensional and redundant. **World models** (Ha & Schmidhuber 2018, Hafner et al. 2019) learn an encoder $e_\phi : s \mapsto z$ that maps to a compact latent state $z \in \mathbb{R}^d$. Dynamics are learned in latent space:

$$f_\phi : (z_t, a_t) \mapsto \hat{z}_{t+1}$$

The policy $\pi_\theta(z_t)$ operates on latent states. Planning rollouts are generated entirely in $\mathbb{R}^d$, avoiding costly environment steps. The decoder $d_\phi : z \mapsto \hat{s}$ is used only for visualization and auxiliary losses, not for planning.

---

## 4. RSSM (Recurrent State Space Model)

### 4.1 Deterministic + Stochastic Latent

The Recurrent State Space Model (Hafner et al., Dreamer 2019) decomposes the latent state into two components:

- **Deterministic state** $h_t \in \mathbb{R}^{d_h}$: updated by a GRU, carries long-range memory.
- **Stochastic state** $z_t \in \mathbb{R}^{d_z}$: sampled from a distribution conditioned on $h_t$, captures aleatoric uncertainty.

The recurrence:

$$h_t = \text{GRU}(h_{t-1}, z_{t-1}, a_{t-1})$$

$$z_t \sim q_\phi(z_t \mid h_t, s_t) \quad \text{(posterior, uses observation)}$$

$$\hat{z}_t \sim p_\phi(z_t \mid h_t) \quad \text{(prior, used during imagination)}$$

*(Why split into deterministic and stochastic? The deterministic $h_t$ allows gradients to flow back through many timesteps via GRU's gating (analogous to LSTM cells). The stochastic $z_t$ lets the model express genuine uncertainty about the future — important when planning over multi-step horizons.)*

### 4.2 ELBO Objective (High-Level Sketch)

The RSSM is trained by maximizing the Evidence Lower BOund (ELBO) on the log-likelihood of observations:

$$\mathcal{L}_{\text{RSSM}} = \mathbb{E}_q\!\left[\sum_t \underbrace{\log p(s_t \mid h_t, z_t)}_{\text{reconstruction}} + \underbrace{\log p(r_t \mid h_t, z_t)}_{\text{reward prediction}} - \underbrace{\beta \, D_{\text{KL}}(q_\phi(z_t \mid h_t, s_t) \| p_\phi(z_t \mid h_t))}_{\text{regularization toward prior}}\right]$$

*(Why the KL term? We want the posterior (which uses observations) to stay close to the prior (which does not). At test time / during imagination, we can only access the prior. If prior and posterior diverge too much, imagined rollouts will look nothing like real rollouts, and the policy trained on imagination will fail in reality.)*

The $\beta$ hyperparameter balances reconstruction fidelity vs. compression of the stochastic state.

---

## 5. Failure Modes

### 5.1 Model Exploitation

A policy trained against an imperfect model can find states where the model's predictions are systematically optimistic — regions of state space that were rarely visited during data collection, where the model extrapolates incorrectly. The policy "exploits" these model errors to collect spuriously high rewards.

**Symptoms:** Policy performs well in simulation (high imagined return) but fails in the real environment. Diverging model loss after policy update.

**Mitigations:**
- **Ensemble disagreement** as uncertainty signal: train $K$ models; avoid states where predictions diverge.
- **Short rollout horizons:** limit imagination to $H = 1$–$5$ steps where errors are still small.
- **Pessimism under uncertainty:** penalize predicted reward by uncertainty estimate.

### 5.2 Compounding Errors

Even an unbiased model has per-step MSE $\epsilon$. Over a horizon $H$, errors compound:

$$\mathbb{E}[\|\hat{s}_H - s_H\|] \approx H \cdot \epsilon \quad \text{(linear, optimistic)}$$

In practice, errors compound super-linearly in chaotic systems (Lyapunov exponent > 0). The practical rule of thumb: trust the model for $H \le 5$–$10$ steps; beyond that, use a value function to "bootstrap" the remaining return (as in MBPO).

### 5.3 When to Trust the Model

| Situation | Trust model? | Reason |
|---|---|---|
| Near training distribution | Yes | Low extrapolation error |
| Novel state (high ensemble disagreement) | No | High epistemic uncertainty |
| Short horizon ($H \le 5$) | Yes | Errors have not compounded |
| Long horizon ($H > 20$) | No | Compounding makes predictions unreliable |
| Dense reward, smooth dynamics | Yes | Easy to fit accurately |
| Sparse reward, contact-rich dynamics | No | Hard to model; small errors cause large value errors |

---

## 6. Research Bridge

### 6.1 AlphaZero and MCTS

AlphaZero (Silver et al. 2018) uses a *perfect* model (the game rules) for Monte Carlo Tree Search (MCTS). The value network $V_\theta(s)$ acts as a leaf evaluator, allowing the tree search to terminate at depth $d$ rather than rollout to terminal. This is the model-based analogue of value function bootstrapping: the model provides the dynamics, the value network provides the terminal estimate.

MCTS with $N$ simulations per move is equivalent to Dyna-Q with $N$ planning steps where the model is exact.

### 6.2 Constitutional AI and Self-Play

Constitutional AI (Anthropic 2022) has the model critique its own outputs using a set of principles. This is a form of world model: the model simulates how a critic would evaluate a response before committing to it. The "planning" is textual — the model generates a draft, then imagines the critic's feedback, then revises. This matches the Dyna pattern: real data (human feedback) is expensive; synthetic data (self-critique) is cheap.

### 6.3 Chain-of-Thought as World Model Simulation

Standard next-token prediction is model-free: given context, predict the next token. Chain-of-thought (Wei et al. 2022) prompts the model to externalize intermediate reasoning steps before answering. This is equivalent to running a world model forward: each reasoning step $z_1, z_2, \ldots, z_k$ is a latent state in a trajectory that terminates at the answer.

The key insight is that *test-time compute* (longer chains) substitutes for *model capacity*: a smaller model with 100-step CoT can match a larger model with direct prediction, just as Dyna-Q with more planning steps can match a model-free agent with more environment steps.

**Formal analogy:**

| RL concept | CoT analogue |
|---|---|
| Transition model $f_\phi(s_t, a_t)$ | Next-thought predictor |
| Latent state $z_t$ | Intermediate reasoning token |
| Planning horizon $H$ | Chain length $k$ |
| Policy $\pi(a \mid z_t)$ | Final answer decoder |
| Model exploitation | Hallucination (confident but wrong) |

---

*End of Module 4 lecture notes.*
