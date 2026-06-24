# Module 6: Reinforcement Learning for Language Models

> **One-sentence framing:** RL is the mechanism behind InstructGPT, GRPO (DeepSeek-R1), and modern LLM alignment.

> **How to read this chapter.** Plain-English intuition first, then formal derivation, then equations
> with inline annotations. Each section builds directly on the previous. The Research Bridge at the
> end connects every concept to real-world alignment systems like InstructGPT and DeepSeek-R1.

---

## 1. Intuition — The RL/LLM Correspondence

Language models and RL agents are structurally identical once you squint at them correctly.
A language model generating a response is an RL agent taking a sequence of actions (tokens)
in an environment (the context window) to maximize a reward (human preference).

| RL Concept | LLM Equivalent |
|---|---|
| State $s_t$ | Context (prompt + tokens so far) |
| Action $a_t$ | Next token |
| Policy $\pi_\theta$ | Language model |
| Reward $R$ | Human preference score / verifier |
| Episode | Full generation |
| State space $\mathcal{S}$ | All possible token sequences |
| Action space $\mathcal{A}$ | Vocabulary (e.g., 50,257 tokens for GPT-2) |

*(Why? Because) the Markov property holds at the token level: given the full context (prompt +
generated tokens so far), the next-token distribution depends only on the current state, not on
any additional history. The "episode" ends when the model emits an end-of-sequence token.*

**Key difference from classic RL:** The action space is enormous (~50K tokens) but sparse — only
a few tokens are grammatically/contextually plausible at each step. The policy must learn both
*what* to say and *how* to say it.

---

## 2. Reward Modeling

### 2.1 Why We Need a Reward Model

Human feedback is the ground truth but is expensive: a single annotator can rate only ~100
responses per hour. We train a **reward model** (RM) to approximate human preferences, then use
that RM to provide dense rewards for RL fine-tuning.

### 2.2 The Bradley-Terry Preference Model

Given a prompt $x$ and two responses $y_w$ (winner/preferred) and $y_l$ (loser/dispreferred),
the Bradley-Terry model defines the probability of preferring $y_w$:

$$P(y_w \succ y_l \mid x) = \sigma\!\left(R_\phi(x, y_w) - R_\phi(x, y_l)\right)$$

where $R_\phi$ is the learned reward function and $\sigma$ is the sigmoid. This says: prefer
$y_w$ iff its scalar reward is higher.

### 2.3 Preference Loss

We fit $R_\phi$ by maximum likelihood on a dataset of $(x, y_w, y_l)$ triples:

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma\!\left(R_\phi(x,y_w) - R_\phi(x,y_l)\right)\right]$$

*(Why? Because) this is simply binary cross-entropy with labels "preferred" and "dispreferred."
Minimizing it pushes $R_\phi(x, y_w) > R_\phi(x, y_l)$ for all training pairs.*

**Architecture:** Typically, the RM shares the backbone of the SFT model with a linear head that
maps the final hidden state to a scalar. This leverages the language model's understanding of text
quality.

**Evaluation metric:** Ranking accuracy (fraction of held-out pairs where $R_\phi(y_w) > R_\phi(y_l)$),
not MSE. *(Why? Because) we care about relative ordering, not absolute reward values. Two reward
functions that differ by a constant give the same policy.*

---

## 3. RLHF-PPO Pipeline

### 3.1 Three-Stage Pipeline

```
Stage 1: SFT (Supervised Fine-Tuning)
    Pretrained LM → fine-tune on high-quality demonstrations → SFT model

Stage 2: RM Training
    SFT model backbone → fit on preference comparisons → Reward Model R_φ

Stage 3: PPO Fine-Tuning
    SFT model → policy π_θ (initialized from SFT)
    Frozen SFT model → reference policy π_ref
    Use R_φ to score generations, update π_θ with PPO
```

### 3.2 KL Penalty

Naive RLHF would maximize $R_{RM}$ without constraint, quickly producing degenerate text that
"hacks" the reward model (e.g., repetitive token sequences that score high on sentiment but are
meaningless). The KL penalty prevents this:

$$R_{total}(x,y) = R_{RM}(x,y) - \beta\,\text{KL}\!\left(\pi_\theta(y|x)\,\|\,\pi_{ref}(y|x)\right)$$

$$= R_{RM}(x,y) - \beta \sum_t \log\frac{\pi_\theta(a_t|s_t)}{\pi_{ref}(a_t|s_t)}$$

*(Why? Because) the KL term penalizes divergence from the SFT model. If $\pi_\theta$ starts
generating text very different from $\pi_{ref}$, the KL penalty increases, pushing the policy
back. This is the soft trust region of RLHF.*

**Hyperparameter $\beta$:**
- Too low: reward hacking — model diverges to exploit RM weaknesses
- Too high: KL collapse — model barely moves from SFT, reward barely improves
- Typical range: $\beta \in [0.01, 0.5]$; often scheduled to start low and increase

---

## 4. Token-Level PPO

### 4.1 Credit Assignment Over Tokens

Classic PPO assigns one reward per episode. For LLMs, an episode is a full generation of $T$
tokens. If we only reward at the end, we have extremely sparse credit assignment — we don't know
which tokens contributed to the reward.

Token-level PPO addresses this by:
1. Decomposing the KL penalty per-token: $-\beta \log(\pi_\theta(a_t|s_t) / \pi_{ref}(a_t|s_t))$
2. Adding the RM reward only at the final token (or spreading via a process reward model)
3. Running GAE (Generalized Advantage Estimation) over the token sequence

### 4.2 Value Network

A value network $V_\psi(s_t)$ estimates expected future reward at each token position. This is
a separate head (or model) trained in parallel with the policy. The value network enables:

$$\hat{A}_t^{GAE} = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$ is the TD residual.

*(Why does token-level matter?) Without it, a model generating 200 tokens gets one signal and
must figure out which token choices caused a good/bad response. Token-level signals let the
model learn "token 50 started a tangent that lowered reward" rather than blaming the whole
generation equally.*

---

## 5. GRPO — Group Relative Policy Optimization

### 5.1 Core Idea

GRPO (introduced in DeepSeek-R1) eliminates the value network entirely by using **group sampling**
to estimate advantages. For each prompt, sample $G$ responses from the current policy:

$$\{y_1, y_2, \ldots, y_G\} \sim \pi_\theta(\cdot | x)$$

Score each with the reward model: $\{r_1, r_2, \ldots, r_G\}$.

### 5.2 Group-Normalized Advantage

Normalize rewards within the group:

$$\tilde{A}_i = \frac{r_i - \mu_r}{\sigma_r + \varepsilon}$$

where $\mu_r = \frac{1}{G}\sum_i r_i$, $\sigma_r = \text{std}(r_1, \ldots, r_G)$, and
$\varepsilon = 10^{-8}$ prevents division by zero.

*(Why? Because) $\mu_r$ is a Monte Carlo estimate of the value $V^\pi(x)$ for this prompt.
Subtracting it is exactly the baseline subtraction from vanilla policy gradient — it reduces
variance. The $\sigma_r$ normalization further stabilizes learning across prompts with
different reward scales.*

**Common mistake:** Forgetting $\varepsilon$. If all $G$ responses in a group receive identical
rewards (e.g., all correct or all wrong), $\sigma_r = 0$ and we'd divide by zero. Adding
$\varepsilon$ makes the advantage zero in this case, meaning no update — which is exactly right
(we can't tell which response was better).

### 5.3 GRPO Loss

The loss is identical to PPO-Clip but using $\tilde{A}_i$:

$$\mathcal{L}_{GRPO} = -\mathbb{E}_i\!\left[\min\!\left(\rho_i \tilde{A}_i,\; \text{clip}(\rho_i, 1-\varepsilon, 1+\varepsilon) \tilde{A}_i\right)\right]$$

where $\rho_i = \pi_\theta(y_i|x) / \pi_{\theta_{old}}(y_i|x)$ is the importance weight.

### 5.4 Why No Value Network?

The value network in PPO requires:
- A separate model (memory overhead)
- Training loss for value estimation (compute overhead)
- Careful tuning of value coefficient $c_1$

GRPO avoids all this by using group sampling as a Monte Carlo value estimate.
**Trade-off:** GRPO needs $G \geq 4$ samples per prompt, increasing inference cost.
For tasks with cheap verifiers (math, code), this is a good trade. For tasks with
expensive RM calls, PPO with value head may be preferable.

---

## 6. DPO — Direct Preference Optimization

### 6.1 The Key Insight

RLHF requires training a reward model *and then* running PPO — two expensive stages. DPO
observes that the optimal policy under the KL-constrained objective has a closed form:

$$\pi^*(y|x) = \frac{\pi_{ref}(y|x) \exp(R(x,y)/\beta)}{Z(x)}$$

Inverting this: $R(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$.

The partition function $Z(x)$ cancels in the Bradley-Terry comparison, giving:

$$P(y_w \succ y_l | x) = \sigma\!\left(\beta\log\frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

### 6.2 DPO Loss

Substituting $\pi_\theta$ for $\pi^*$ and fitting by MLE:

$$\mathcal{L}_{DPO} = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

*(Why is this simpler?) No reward model needed — we directly optimize the policy on preference
pairs. The reference model $\pi_{ref}$ is frozen (typically the SFT checkpoint). We just compute
log-probabilities under $\pi_\theta$ and $\pi_{ref}$ for $y_w$ and $y_l$.*

**Implementation:** For each batch, compute:
1. $\log \pi_\theta(y_w|x)$, $\log \pi_\theta(y_l|x)$ — forward pass through policy
2. $\log \pi_{ref}(y_w|x)$, $\log \pi_{ref}(y_l|x)$ — forward pass through frozen reference
3. Log-ratio differences, scaled by $\beta$
4. Binary cross-entropy loss

---

## 7. DPO Limitations

DPO is elegant but has several practical weaknesses:

**Offline learning:** DPO trains on a fixed dataset of preference pairs. Unlike PPO, it cannot
generate new responses to explore the policy's current behavior. Distribution shift accumulates
as $\pi_\theta$ diverges from the distribution that generated the preference data.

**No process rewards:** DPO assigns credit at the response level. It cannot learn from
step-by-step reasoning quality (process reward models) — only outcome quality.

**Length bias:** DPO tends to prefer shorter responses because longer sequences have lower
log-probability under any LM. The log-ratio $\log \pi_\theta(y_w) / \pi_{ref}(y_w)$ is a sum
over tokens, so longer $y_w$ systematically disadvantages preferred responses. Mitigations:
normalize by sequence length or use length-controlled DPO variants.

**Reward hacking via log-ratio manipulation:** The gradient of $\mathcal{L}_{DPO}$ with respect
to $\pi_\theta$ pushes up $\log \pi_\theta(y_w)$ and down $\log \pi_\theta(y_l)$. But the model
can also satisfy the objective by *decreasing* $\log \pi_{ref}(y_l)$ — which it cannot, since
$\pi_{ref}$ is frozen. However, it can make $\pi_\theta(y_w) / \pi_{ref}(y_w)$ very large by
concentrating probability mass on $y_w$ tokens, even if those tokens are out-of-distribution.

---

## 8. Failure Modes

### 8.1 Reward Hacking in RLHF

The reward model is an imperfect proxy. PPO will find ways to achieve high $R_{RM}$ that don't
correspond to actual human preference — this is Goodhart's Law ("when a measure becomes a target,
it ceases to be a good measure").

Common hacking patterns:
- Repetition: repeating high-reward phrases
- Verbosity: longer responses score higher in some RMs
- Sycophancy: agreeing with the user, even incorrectly
- Format exploitation: bullet points, headers that look authoritative

The KL penalty mitigates but does not eliminate hacking. Ongoing research: process reward models,
constitutional AI, iterative RLHF with refreshed preference data.

### 8.2 KL Collapse (Too High $\beta$)

If $\beta$ is too large, the KL penalty dominates the reward signal. The policy barely deviates
from $\pi_{ref}$, reward improves negligibly, and the model is essentially unchanged from SFT.

Symptom: KL divergence stays near zero; reward barely increases above SFT baseline.

### 8.3 Entropy Collapse (Too Low $\beta$)

If $\beta$ is too small, the policy optimizes the reward model aggressively. This leads to:
- Reward hacking (see above)
- **Entropy collapse:** The policy concentrates on a few high-reward modes, losing diversity.
  The model outputs nearly identical responses regardless of input.

Symptom: Response entropy (bits per token) drops sharply; KL divergence explodes.

**Diagnosis:** Track `train/entropy`, `train/kl_from_ref`, and `train/reward_mean` simultaneously
during training. A healthy run shows reward increasing, KL growing slowly, entropy stable.

---

## 9. Research Bridge

### 9.1 InstructGPT (OpenAI, 2022)

The paper that operationalized RLHF for large-scale LLM alignment:
- Stage 1: Fine-tune GPT-3 on ~13K demonstration prompts from contractors
- Stage 2: Train RM on ~33K preference comparisons (contractors rank 4–9 outputs)
- Stage 3: PPO with KL penalty ($\beta = 0.02$) for ~31K prompts

Key finding: A 1.3B InstructGPT model was preferred over 175B GPT-3 by human raters. Alignment
quality dominates raw model scale.

### 9.2 DeepSeek-R1 and GRPO on Math

DeepSeek-R1 uses GRPO with a verifiable reward signal (math answer correctness) instead of a
learned RM. This eliminates RM training and RM hacking simultaneously.

Pipeline:
1. Cold start: SFT on chain-of-thought demonstrations
2. GRPO with math oracle reward (+1 correct, 0 wrong)
3. Rejection sampling + SFT on best GRPO outputs
4. Final GRPO sweep

Result: Reasoning capability emerges from GRPO without any human preference labels — just
a binary verifier. *(Why does this work?) Math has ground truth. The oracle reward doesn't
overfit because it's perfect — there's nothing to hack.*

### 9.3 KL-β Schedule Dynamics

Static $\beta$ is suboptimal. During early training, the policy is close to $\pi_{ref}$ and the
KL penalty has little effect — a lower $\beta$ allows faster exploration. Later, when the policy
starts drifting, increasing $\beta$ prevents hacking.

Adaptive $\beta$ schedules (used in practice):
- **KL targeting:** Adjust $\beta$ to maintain a target KL (e.g., $\text{KL} \approx 1$ nat)
- **PPO's adaptive learning rate:** Treat the KL constraint as a Lagrangian multiplier

The analogy to TRPO is exact: PPO approximates TRPO's trust region with a clipping heuristic;
RLHF approximates TRPO's constraint with a KL penalty term in the reward.
