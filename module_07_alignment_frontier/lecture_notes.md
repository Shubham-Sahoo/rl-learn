# Module 7: The Alignment Frontier

> **One-sentence framing:** This module connects all prior RL concepts to open research problems in AI alignment.

> **How to read this chapter.** We begin with the process/outcome reward distinction, move through PRM
> scoring and Monte Carlo labeling, then tackle VLM alignment challenges, and close with a research
> bridge to current frontier systems. Each section builds on the previous. Inline *(Why? Because ...)*
> annotations explain every non-obvious design choice.

---

## 1. PRMs vs ORMs — Process vs Outcome Reward Models

### 1.1 The Supervision Signal Spectrum

When training a model to solve multi-step reasoning tasks, the most natural reward is **outcome-based**:
did the model get the right final answer?

$$r(x, y) = \mathbf{1}[\text{final answer is correct}]$$

This is called an **Outcome Reward Model (ORM)**. It is easy to label: you only need the ground-truth
answer, not any intermediate judgment. The downside is **sparse feedback** — a chain of 10 reasoning
steps gets a single bit of reward at the very end.

*(Why is sparsity a problem? Because the policy must figure out, from one end-of-episode signal, which
of the 10 steps caused success or failure. With long chains, the credit assignment problem becomes
nearly intractable.)*

**Process Reward Models (PRMs)** instead provide **step-level feedback**:

$$r(x, y_{1:t}) = \text{quality score of the reasoning prefix up to step } t$$

This creates a dense reward signal: every intermediate step is judged independently. The tradeoff is
labeling cost — a human (or oracle) must evaluate each step, not just the final answer.

### 1.2 Step-Level Credit Assignment

Consider a 4-step math solution where steps 1–3 are correct but step 4 makes an arithmetic error.

- **ORM:** Final answer wrong → reward = 0 for the entire trajectory. The policy receives no signal
  that steps 1–3 were good.
- **PRM:** Steps 1–3 get high scores; step 4 gets a low score. The policy can learn to preserve
  the good prefix and fix only the broken step.

*(Why does this matter? Because in long chains of thought (e.g., GPT-o1 style reasoning), errors
are often localized. ORM cannot distinguish "early mistake" from "late mistake"; PRM can.)*

### 1.3 Formal Definitions

Let $x$ be the problem, $y_{1:T}$ be a sequence of $T$ reasoning steps, and $y^*$ be the correct
final answer.

**ORM:**
$$r_\text{ORM}(x, y_{1:T}) = \mathbf{1}[y_T = y^*]$$

**PRM:**
$$r_\text{PRM}(x, y_{1:t}) = f_\phi(x, y_{1:t}) \in [0, 1] \quad \forall t \in \{1, \ldots, T\}$$

where $f_\phi$ is a learned scoring function. *(Why a neural scorer? Because "is this step correct?"
is not mechanically checkable for open-ended reasoning; it requires semantic understanding.)*

---

## 2. PRM Scoring

### 2.1 The Value Interpretation

A PRM step score can be interpreted as a **conditional correctness probability**:

$$V_\text{PRM}(s_t) = P(\text{correct final answer} \mid \text{reasoning prefix}_t)$$

*(Why a probability? Because it gives a calibrated signal — 0.9 means "probably fine," 0.1 means
"likely derailed." Compare this to a raw scalar reward that requires careful normalization.)*

This reframes PRM scoring as **value estimation** in a Markov chain where:
- States $s_t$ = reasoning prefix $y_{1:t}$
- Actions = next reasoning step $y_{t+1}$
- Terminal state = final answer
- "Value" = probability of reaching the correct terminal state

### 2.2 Combining Step Scores

Given step scores $p_1, p_2, \ldots, p_T \in [0, 1]$ for a complete solution, two common
aggregation strategies are used:

**Product (joint probability interpretation):**
$$\text{score}(y_{1:T}) = \prod_{t=1}^{T} p_t$$

This treats each step as an independent Bernoulli event. *(Why product? Because if every step
must be correct for the solution to succeed, the joint probability is the product of marginals
under independence.)*

**Minimum (weakest-link interpretation):**
$$\text{score}(y_{1:T}) = \min_{t \in \{1,\ldots,T\}} p_t$$

This treats the score as the reliability of the most suspect step. *(Why min? Because a chain
of reasoning is only as strong as its weakest link — one fatal error invalidates the whole.)*

**Empirical findings:** The product method tends to work better for Best-of-N selection (§4),
while the minimum is more robust to PRM miscalibration on intermediate steps.

### 2.3 PRM vs Value Function

In standard RL, the value function $V^\pi(s)$ is defined as expected return under policy $\pi$.
The PRM score is analogous but:
- Trained offline (supervised, not via TD updates)
- Evaluated at inference time (no policy interaction required)
- Binary label: correct/incorrect (not a scalar reward)

*(Why offline training? Because collecting PRM training data requires running a policy to generate
step sequences, then labeling each step — the labeling can be done after the fact, without an
online RL loop.)*

---

## 3. Monte Carlo PRM Labels

### 3.1 The Labeling Challenge

Manually labeling every step of every solution is expensive. A scalable alternative: **Monte Carlo
estimation of step correctness**.

The key insight: we don't need a human to judge whether step $t$ is correct. We can instead ask:
*"Given the solution so far, how often does continuing from here lead to a correct final answer?"*

$$P(\text{correct} \mid \text{step}_t) \approx \frac{1}{N} \sum_{j=1}^{N} \mathbf{1}[\text{rollout}_j \text{ from step } t \text{ reaches correct answer}]$$

*(Why does this work? Because if a step introduces an error, most continuations from that point
will fail to reach the correct answer. If the step is correct, random continuations can often
recover.)*

### 3.2 Algorithm

```
Input: solution prefix y_{1:t}, oracle(y) → {True, False}, n_rollouts N

1. For j = 1..N:
   a. Sample a completion y_{t+1:T}^j ~ policy(· | y_{1:t})
   b. label_j = oracle(y_T^j)  # is final answer correct?

2. Return (1/N) * sum(label_j)
```

This is essentially **Monte Carlo value estimation** applied to step-level supervision.

*(Why is this approximate? Because (a) the rollout policy may not be the final policy, and
(b) N is finite. In practice N=16–256 gives good signal, with larger N used for the earliest
steps where variance is highest.)*

### 3.3 Bias-Variance Tradeoff in MC Labels

| N | Variance | Labeling cost | Use case |
|---|----------|---------------|----------|
| 4 | High | Cheap | Rough filtering |
| 16 | Medium | Moderate | Training PRMs |
| 64 | Low | Expensive | Calibration |
| 256 | Very low | Very expensive | Gold labels |

*(Why does early-step variance matter most? Because more future steps remain, so completion
outcomes vary more. Later steps have fewer remaining decisions, so completions converge faster.)*

---

## 4. Best-of-N with PRM

### 4.1 Best-of-N Selection

The simplest way to use a PRM at inference time is **Best-of-N** (also called rejection sampling):

```
1. Generate N independent solutions y_{1:T}^1, ..., y_{1:T}^N
2. Score each: score_j = product(PRM(y_{1:t}^j) for t in 1..T)
3. Return argmax_j score_j
```

*(Why generate independently? Because we want diverse candidates — the N solutions should explore
different reasoning paths, not minor variations of the same path.)*

### 4.2 PRM vs ORM Reranking

**ORM reranking:**
- Score each candidate with the ORM
- Select the candidate with highest ORM score
- Problem: ORM only sees the final answer, so it can't distinguish a lucky correct answer from
  a reliably-derived correct answer

**PRM reranking:**
- Score each step of each candidate
- Aggregate step scores (product or min)
- Select the highest-scoring candidate
- Advantage: Can detect "correct answer, wrong reasoning" and penalize it

*(Why penalize "correct answer, wrong reasoning"? Because in test-time compute scaling, the goal
is robust reasoning that generalizes, not lucky answers that pass the specific test case.)*

### 4.3 Efficiency Analysis

Best-of-N has **quadratic cost** in the worst case: N solutions × T steps to score. But in
practice:

**Compute budget:** $N \times T_\text{avg}$ forward passes for generation, $N \times T_\text{avg}$
forward passes for PRM scoring. Total: $2NT_\text{avg}$.

**Comparison:** ORM Best-of-N costs $N(T_\text{avg} + 1)$ — similar, since ORM is one final score.

**Scaling:** Empirically, PRM Best-of-16 ≈ ORM Best-of-256 on math reasoning benchmarks. The
efficiency gain comes from PRM's ability to identify good reasoning paths rather than lucky endpoints.

*(Why does PRM scale better? Because it provides a more informative signal per candidate —
it's like evaluating 16 candidates at step-level depth vs. 256 candidates at output-only depth.)*

---

## 5. VLM Alignment — Visual Language Models

### 5.1 What Are VLMs?

**Visual Language Models (VLMs)** extend LLMs with vision encoders (e.g., CLIP, SigLIP) that
project image patches into the language model's token embedding space. Examples: LLaVA, Qwen-VL,
InternVL, GPT-4V, Gemini.

The core alignment challenge: LLMs trained on text learn a rich world model from text alone.
VLMs must **ground** language in visual percepts — and text-trained priors can conflict with
visual evidence.

### 5.2 Hallucination Taxonomy

VLMs are prone to **hallucination** — generating text that is not grounded in the input image:

| Type | Description | Example |
|------|-------------|---------|
| Object hallucination | Mentions object not present in image | "There is a red umbrella" (none in image) |
| Attribute hallucination | Wrong color/size/count for real object | "The dog is black" (dog is white) |
| Relationship hallucination | Wrong spatial/relational claim | "The cat is on the table" (cat is under it) |
| Action hallucination | Claims non-occurring action | "The man is running" (man is standing) |

*(Why are VLMs especially prone to object hallucination? Because language priors are strong: a
model trained to describe "a kitchen scene" associates it with common kitchen objects. If vision
features are weak or ambiguous, the language prior overrides visual evidence.)*

### 5.3 The CHAIR Metric

**CHAIR (Caption Hallucination Assessment with Image Relevance)** measures object-level
hallucination in image captions using a ground-truth object set.

**CHAIR$_s$** (sentence-level):

$$\text{CHAIR}_s = \frac{|\{s : \exists \text{ hallucinated object mention in } s\}|}{|\text{sentences}|}$$

This measures the fraction of sentences containing at least one hallucinated object mention.

*(Why sentence-level? Because one hallucinated object in a long sentence might be a minor issue,
but CHAIR$_s$ still flags the sentence — giving a conservative upper bound on hallucination rate.)*

**CHAIR$_i$** (instance-level):

$$\text{CHAIR}_i = \frac{|\text{hallucinated objects}|}{|\text{mentioned objects}|}$$

This measures the fraction of all object mentions that are hallucinations.

*(Why two metrics? CHAIR$_s$ penalizes hallucination presence/absence per sentence; CHAIR$_i$
penalizes hallucination density. A model that mentions 100 objects with 1 hallucination has
CHAIR$_s$ = 1.0 (if in one sentence) but CHAIR$_i$ = 0.01. Both views are informative.)*

**Computing CHAIR:**
1. Parse generated caption for noun phrases
2. Map nouns to MSCOCO object categories (or similar ontology)
3. Compare to ground-truth objects in the image annotation
4. Object is hallucinated if it appears in the caption but not the ground-truth set

---

## 6. VLM-Specific RLHF Failures

### 6.1 Visual Grounding Requires Separate Supervision

Standard LLM RLHF trains a reward model on text-only preferences. For VLMs:
- Human raters must evaluate caption/answer quality *conditioned on the image*
- Text-only reward models cannot detect visual hallucinations
- The reward model must be multimodal, processing both image and text

*(Why can't you reuse a text RM? Because "The cat is on the table" is a perfectly fine sentence
in isolation. Only when grounded in the image does it become right or wrong. Text reward models
have no access to the visual evidence.)*

### 6.2 Sycophancy on Visual Claims

LLMs are known to exhibit **sycophancy** — agreeing with the user's stated beliefs even when
incorrect. In VLMs, this manifests as:

- User: "Is this a cat?" (pointing at a dog)
- VLM: "Yes, this is a cat." *(sycophantic agreement)*

The model's language prior (agreeing with user questions) overrides visual evidence.

*(Why is this particularly dangerous in VLMs? Because users often trust visual claims more than
abstract claims — "what is in this image" feels objective, not subjective.)*

### 6.3 Distributional Shift Between Vision and Text Modalities

VLMs are typically trained in two phases:
1. **Pretraining:** Vision encoder and language model trained separately, then connected
2. **Fine-tuning:** RLHF/SFT on instruction-following data

The vision-language connector (often a linear projection or MLP) is trained on a specific
distribution of image-text pairs. When RLHF fine-tuning shifts the language model, the
visual representations can become misaligned.

*(Why is this a problem? Because the vision encoder is frozen during RLHF. If the language
model's internal representation changes significantly, the projection from vision space into
language space becomes stale.)*

**Mitigation strategies:**
- Use LoRA (§7) to limit language model drift during RLHF
- Periodically recalibrate the vision-language connector
- Include visual grounding tasks in the RLHF dataset

---

## 7. RLHF at Scale

### 7.1 Memory Overhead: The 4× Problem

Standard PPO-based RLHF requires four models in memory simultaneously:

| Model | Role | Memory |
|-------|------|--------|
| Policy ($\pi_\theta$) | Being trained | $M$ |
| Reference ($\pi_\text{ref}$) | KL penalty baseline | $M$ |
| Critic ($V_\phi$) | Advantage estimation | $M$ |
| Reward model ($r_\psi$) | Reward computation | $M$ |

**Total: ~4× the base model memory.** For a 70B model at 16-bit precision:
$4 \times 70 \times 10^9 \times 2 \text{ bytes} \approx 560 \text{ GB}$ — requires ≥ 7 H100s.

*(Why is this unavoidable? Because all four models must be accessible during the rollout phase:
the policy generates tokens, the reward model scores them, the reference model computes KL,
and the critic estimates baselines.)*

### 7.2 LoRA Mitigations

**Low-Rank Adaptation (LoRA)** freezes the base model and adds low-rank update matrices:

$$W' = W + \Delta W = W + BA$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with $r \ll \min(d, k)$.

Applied to RLHF:
- Policy = base model + LoRA adapters (only adapters are trained)
- Reference = same base model, no adapters (or frozen base with initial LoRA = 0)
- Critic = new LoRA head on base model
- Reward model = frozen (pre-trained separately)

**Memory saving:** Only $2dr$ extra parameters per layer. For $r=64$, $d=4096$: $2 \times 4096 \times 64 = 524$K vs. $4096^2 = 16$M base parameters — a 30× reduction in trainable params.

*(Why not just share model weights? Because the policy and reference must differ — that's the
point of RLHF. But they can share the frozen base, with only the LoRA adapters being distinct.)*

### 7.3 Async PPO

Standard PPO blocks generation: the training loop waits for all rollout tokens before updating.
**Async PPO** decouples generation from training:

```
Generator workers:  [gen batch 1] [gen batch 2] [gen batch 3] ...
Trainer:                 [train on batch 0] [train on batch 1] ...
```

This introduces a **staleness** issue: the policy generating batch $k+1$ is slightly older than
the policy trained on batch $k$. In practice, 1–2 batches of staleness is tolerable.

*(Why does staleness matter? Because PPO is on-policy — it assumes data was generated by the
current policy. Staleness is essentially importance-ratio drift, which the PPO clip handles up
to a point.)*

### 7.4 Memory-Efficient Training Patterns

**Gradient checkpointing:** Recompute activations during backward pass instead of storing them.
Trades compute for memory.

**Flash Attention:** Fused CUDA kernel for attention that avoids materializing the full
$n \times n$ attention matrix.

**ZeRO (Zero Redundancy Optimizer):** Shards optimizer state, gradients, and parameters across
GPUs (DeepSpeed ZeRO stages 1–3).

**Offloading:** Move inactive models (reference, reward model) to CPU between rollout phases.

*(Why not always use all these tricks? Because they add engineering complexity and some (gradient
checkpointing) slow down training by 20–30%. Use them when GPU memory is the bottleneck.)*

---

## 8. Research Bridge

### 8.1 o1/o3 and PRMs

OpenAI's o1 and o3 models are widely believed to use PRMs for training and inference-time compute
scaling. Key patterns:
- **Verification during training:** PRM provides step-level supervision for chain-of-thought traces
- **Best-of-N at inference:** Multiple "thoughts" generated, PRM selects the best
- **Process supervision dataset (PRM800K):** 800K step-level labels on grade school math

*(Why doesn't OpenAI confirm this? PRMs are a key differentiator. The architecture remains
proprietary, though the PRM800K dataset was open-sourced.)*

### 8.2 Qwen-VL and InternVL Alignment

Qwen-VL and InternVL represent the frontier of open-weight VLM alignment:
- **Qwen-VL-Chat:** RLHF with human preference data for visual instruction following
- **InternVL-Chat:** Constitutional AI-style self-critique to reduce hallucination
- Both use LoRA for memory-efficient RLHF at 7B–72B scale

*(Why are open-weight VLMs important for alignment research? Because closed VLMs (GPT-4V, Gemini)
cannot be studied mechanistically — we cannot measure CHAIR, check reward model internals, or
ablate training choices.)*

### 8.3 Scalable Oversight Open Problems

1. **Weak-to-strong generalization:** Can a weak supervisor's labels bootstrap a strong model?
   *(Why unsolved? Because the strong model can already exceed human ability in many domains,
   making human labels unreliable.)*

2. **PRM robustness:** PRMs trained on math may not transfer to code or science reasoning.
   *(Why? Because "step correctness" is domain-specific — a correct step in algebra looks
   different from a correct step in a proof.)*

3. **Process vs. outcome tradeoff at scale:** At sufficient scale, do PRMs still outperform ORMs,
   or does outcome supervision become sufficient? Open empirical question.

### 8.4 Constitutional AI

Constitutional AI (Anthropic, 2022) provides an alternative to human labeling:

1. **Critique:** Model critiques its own output against a constitution (list of principles)
2. **Revision:** Model revises its output based on the critique
3. **RL-CAI:** RLHF with an AI feedback model trained on critique-revision pairs

This connects to PRM in an interesting way: the critique step is a form of **process supervision**
— the model evaluates whether each aspect of its response satisfies the constitutional principle.

*(Why is Constitutional AI relevant to this module? Because it shows that process-level feedback
(the critique) can be automated, reducing the human labeling cost that makes PRMs expensive.)*

---

## Summary

| Concept | Key Idea | RL Connection |
|---------|----------|---------------|
| ORM | Sparse terminal reward | Monte Carlo returns |
| PRM | Dense step reward | Value function / TD targets |
| MC PRM Labels | Rollout-based label estimation | TD(0) vs. Monte Carlo value estimation |
| Best-of-N | Inference-time compute scaling | Policy evaluation, not improvement |
| CHAIR | Precision of visual grounding | Reward function design |
| LoRA RLHF | Low-rank policy updates | Trust region methods |
| Async PPO | Decoupled generation/training | Importance sampling staleness |

**The central theme of this module:** All the technical tools developed in Modules 1–6 (MDPs,
value estimation, policy gradients, model-based RL, PPO, reward modeling) appear again in the
alignment frontier. PRM training is value function regression. Best-of-N is policy evaluation.
RLHF is policy optimization with a learned reward. The alignment frontier is not a different
field — it is deep RL applied at scale to language and vision.
