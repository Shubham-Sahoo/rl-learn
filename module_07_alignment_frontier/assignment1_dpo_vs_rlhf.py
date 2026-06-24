# %% [markdown]
# # Module 07, Assignment 1: DPO vs RLHF
#
# ## Prerequisites
# - Module 06 A2: RLHF pipeline (compute_kl_penalty, compute_rlhf_reward)
# - Lecture notes (DPO section from Module 06, and this module's sections 1–4)
#
# ## Learning Objectives
# 1. Derive and implement the DPO loss from scratch
# 2. Understand the implicit reward DPO optimizes
# 3. Compare DPO vs PPO-RLHF on the same dataset
# 4. Understand DPO's limitations (offline, no process rewards)

# %% [markdown]
# ## Part 0: Theory Recap — DPO Derivation
#
# **The RLHF objective** (maximize expected reward subject to KL constraint):
#
# $$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)}\!\left[r(x,y)\right] - \beta\, D_\text{KL}\!\left[\pi_\theta(\cdot|x) \,\|\, \pi_\text{ref}(\cdot|x)\right]$$
#
# **The optimal policy** (closed-form solution):
#
# $$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_\text{ref}(y \mid x) \exp\!\left(\frac{r(x,y)}{\beta}\right)$$
#
# **Rearranging for the reward:**
#
# $$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_\text{ref}(y \mid x)} + \beta \log Z(x)$$
#
# **Bradley-Terry preference model** (probability that $y_w$ is preferred over $y_l$):
#
# $$P(y_w \succ y_l \mid x) = \sigma\!\left(r(x, y_w) - r(x, y_l)\right)$$
#
# **Substituting the rearranged reward** (the $\log Z(x)$ terms cancel):
#
# $$P(y_w \succ y_l \mid x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)$$
#
# **DPO loss** (negative log-likelihood of preferences under the parameterized policy):
#
# $$\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]$$
#
# *(Why does DPO avoid explicit reward training? Because it directly parameterizes the reward
# via the policy ratio — no separate reward model required. The policy IS the implicit reward model.)*

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("transformers not available; Parts 3–4 require it.")

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("datasets not available; Parts 3–4 require it.")

from rllearn.logging import make_writer

# %% [markdown]
# ## Part 1: Implement the DPO Loss
#
# The DPO loss requires computing **per-sequence log-probabilities** under both the policy and
# reference model. The log-probability of a sequence is the **sum of token-level log-probs
# over the response tokens only** (not the prompt).
#
# **Common mistake:** Including prompt tokens in the log-prob sum. Prompt tokens are conditioned
# on, not generated — their log-probs should not be included in the objective.
#
# **Implementation steps:**
# 1. Forward pass through `policy_model` with `input_ids_w` (winner) and `input_ids_l` (loser)
# 2. Compute token log-probs: `log_probs = F.log_softmax(logits, dim=-1)`
# 3. Gather the log-prob of the actual next token at each position
# 4. Sum over response tokens only (slice off prompt prefix)
# 5. Repeat for `ref_model` (no gradient)
# 6. Compute the DPO loss formula
#
# **Hint:** The logits at position $t$ predict token $t+1$. If your input is `[prompt | response]`,
# the response log-probs start at index `len(prompt)` in the logit sequence.

# %%
def get_sequence_log_probs(model: nn.Module, input_ids: torch.Tensor,
                           response_start_idx: int) -> torch.Tensor:
    """
    Compute the sum of log-probs over response tokens for a batch of sequences.

    Args:
        model: Causal language model
        input_ids: Token ids of shape (batch, seq_len)
        response_start_idx: Index where the response begins (prompt length)

    Returns:
        log_prob_sum: Shape (batch,) — sum of log P(token | context) for response tokens
    """
    # TODO: forward pass → gather log-probs of actual tokens → sum over response slice
    raise NotImplementedError


def dpo_loss(policy_model: nn.Module, ref_model: nn.Module,
             input_ids_w: torch.Tensor, input_ids_l: torch.Tensor,
             beta: float = 0.1,
             prompt_length: int = 0) -> torch.Tensor:
    """
    DPO loss:
    L = -E[log sigma(beta * log(pi_theta(y_w|x)/pi_ref(y_w|x))
                   - beta * log(pi_theta(y_l|x)/pi_ref(y_l|x)))]

    Compute log-probs as sum of token log-probs over response tokens only.
    Common mistake: including prompt tokens in the log-prob sum.

    Args:
        policy_model: The model being trained (gradients flow through this)
        ref_model: The frozen reference model (no gradient)
        input_ids_w: Token ids for preferred (winner) response, shape (batch, seq_len)
        input_ids_l: Token ids for rejected (loser) response, shape (batch, seq_len)
        beta: KL penalty coefficient (higher = stay closer to reference)
        prompt_length: Number of prompt tokens to exclude from log-prob sum

    Returns:
        loss: Scalar DPO loss
    """
    raise NotImplementedError


def compute_implicit_reward(policy_model: nn.Module, ref_model: nn.Module,
                            input_ids: torch.Tensor,
                            beta: float,
                            prompt_length: int = 0) -> torch.Tensor:
    """
    Extract DPO implicit reward: beta * log(pi_theta(y|x) / pi_ref(y|x))
    This is the reward DPO implicitly optimizes — useful for analysis.

    Args:
        policy_model: Trained policy
        ref_model: Frozen reference model
        input_ids: Token ids of shape (batch, seq_len)
        beta: KL penalty coefficient
        prompt_length: Number of prompt tokens to exclude

    Returns:
        implicit_reward: Shape (batch,)
    """
    raise NotImplementedError

# %% [markdown]
# ## Part 2: Verify Your DPO Loss
#
# Before training, verify the implementation with unit tests:
#
# **Test 1 — Loss sign:** At initialization (policy ≈ reference), the loss should be close to
# $\log 2 \approx 0.693$ because $\sigma(0) = 0.5$.
#
# **Test 2 — Gradient direction:** After one step, the policy's log-prob for the winner should
# increase relative to the loser's log-prob.
#
# **Test 3 — Beta scaling:** With $\beta = 0$, the KL penalty disappears and the loss becomes
# purely discriminative (no reference).

# %%
def test_dpo_loss_at_initialization() -> bool:
    """
    At initialization (policy = reference), DPO loss ≈ log(2) ≈ 0.693.
    Tests that your implementation has the correct sign and baseline value.
    """
    # TODO: create two tiny identical models, create dummy input_ids, assert loss ≈ 0.693
    raise NotImplementedError


def test_gradient_direction() -> bool:
    """
    After one DPO gradient step, the implicit reward for the winner should be
    strictly greater than for the loser (reward_winner > reward_loser).
    """
    # TODO: create tiny models, one gradient step, check implicit rewards
    raise NotImplementedError


# Run tests (uncomment after implementing)
# assert test_dpo_loss_at_initialization(), "Test 1 failed: loss at init ≠ log(2)"
# assert test_gradient_direction(), "Test 2 failed: gradient direction incorrect"
# print("All unit tests passed.")

# %% [markdown]
# ## Part 3: Training — DPO on Anthropic HH-RLHF
#
# We fine-tune GPT-2 Small on a small subset of the Anthropic Helpful-Harmless RLHF dataset.
#
# **Dataset format:** Each example has `chosen` (preferred response) and `rejected` (dispreferred)
# alongside a `prompt`. We tokenize and form `input_ids_w = [prompt | chosen]` and
# `input_ids_l = [prompt | rejected]`.
#
# **TensorBoard metrics:**
# - `train/dpo_loss` — should monotonically decrease
# - `train/reward_margin` — implicit reward winner minus loser; should increase
#
# **Verification:** DPO loss monotonically decreasing over 3 epochs.

# %%
def load_hh_rlhf_subset(tokenizer, max_examples: int = 500,
                        max_length: int = 256) -> List[Dict]:
    """
    Load a small subset of Anthropic HH-RLHF and tokenize.
    Returns list of dicts with keys: input_ids_w, input_ids_l, prompt_length.
    """
    if not DATASETS_AVAILABLE:
        raise RuntimeError("pip install datasets")

    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    dataset = dataset.select(range(min(max_examples, len(dataset))))

    examples = []
    for item in dataset:
        prompt = item["chosen"].rsplit("\n\nAssistant:", 1)[0] + "\n\nAssistant:"
        chosen = item["chosen"][len(prompt):]
        rejected = item["rejected"][len(prompt):]

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        chosen_ids = tokenizer.encode(chosen, add_special_tokens=False)
        rejected_ids = tokenizer.encode(rejected, add_special_tokens=False)

        # Truncate to max_length
        ids_w = (prompt_ids + chosen_ids)[:max_length]
        ids_l = (prompt_ids + rejected_ids)[:max_length]

        examples.append({
            "input_ids_w": torch.tensor(ids_w),
            "input_ids_l": torch.tensor(ids_l),
            "prompt_length": len(prompt_ids),
        })

    return examples


def collate_fn(batch: List[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Pad a batch of variable-length sequences."""
    max_len_w = max(item["input_ids_w"].size(0) for item in batch)
    max_len_l = max(item["input_ids_l"].size(0) for item in batch)
    prompt_lengths = [item["prompt_length"] for item in batch]

    padded_w = torch.full((len(batch), max_len_w), pad_token_id, dtype=torch.long)
    padded_l = torch.full((len(batch), max_len_l), pad_token_id, dtype=torch.long)

    for i, item in enumerate(batch):
        padded_w[i, :item["input_ids_w"].size(0)] = item["input_ids_w"]
        padded_l[i, :item["input_ids_l"].size(0)] = item["input_ids_l"]

    return {
        "input_ids_w": padded_w,
        "input_ids_l": padded_l,
        "prompt_length": min(prompt_lengths),  # conservative: shortest prompt
    }


def train_dpo(policy_model: nn.Module, ref_model: nn.Module, tokenizer,
              train_data: List[Dict], beta: float = 0.1,
              n_epochs: int = 3, batch_size: int = 4,
              lr: float = 1e-5, device: str = "cpu",
              run_name: str = "dpo_training") -> Dict[str, List[float]]:
    """
    DPO training loop on the HH-RLHF subset.

    PROVIDED — do not modify. Implement dpo_loss and compute_implicit_reward above.

    Logs to TensorBoard:
    - train/dpo_loss
    - train/reward_margin (implicit reward winner - loser)
    """
    writer = make_writer(run_name)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)
    ref_model.eval()
    ref_model.to(device)
    policy_model.to(device)

    history: Dict[str, List[float]] = {"dpo_loss": [], "reward_margin": []}
    global_step = 0

    for epoch in range(n_epochs):
        policy_model.train()
        indices = torch.randperm(len(train_data)).tolist()
        epoch_losses = []

        for batch_start in range(0, len(train_data), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch = [train_data[i] for i in batch_indices]
            batch_dict = collate_fn(batch, tokenizer.pad_token_id or 0)

            ids_w = batch_dict["input_ids_w"].to(device)
            ids_l = batch_dict["input_ids_l"].to(device)
            prompt_len = batch_dict["prompt_length"]

            optimizer.zero_grad()
            loss = dpo_loss(policy_model, ref_model, ids_w, ids_l,
                            beta=beta, prompt_length=prompt_len)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                r_w = compute_implicit_reward(policy_model, ref_model, ids_w, beta, prompt_len)
                r_l = compute_implicit_reward(policy_model, ref_model, ids_l, beta, prompt_len)
                reward_margin = (r_w - r_l).mean().item()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            history["dpo_loss"].append(loss_val)
            history["reward_margin"].append(reward_margin)

            writer.add_scalar("train/dpo_loss", loss_val, global_step)
            writer.add_scalar("train/reward_margin", reward_margin, global_step)
            global_step += 1

        print(f"Epoch {epoch+1}/{n_epochs} | mean loss: {np.mean(epoch_losses):.4f}")

    writer.close()
    return history


# %% [markdown]
# ### Run Training
#
# Requires GPU or patience (GPT-2 Small is ~500M params).
# Use `device = "cuda"` if available.

# %%
# %load_ext tensorboard
# %tensorboard --logdir runs/

# %%
# Uncomment to run training (requires transformers + datasets):
#
# if TRANSFORMERS_AVAILABLE and DATASETS_AVAILABLE:
#     MODEL_NAME = "gpt2"  # GPT-2 Small (~124M params)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     print("Loading tokenizer and models...")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     tokenizer.pad_token = tokenizer.eos_token
#
#     policy_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
#     ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
#     ref_model.requires_grad_(False)
#
#     print("Loading dataset...")
#     train_data = load_hh_rlhf_subset(tokenizer, max_examples=500)
#
#     print(f"Training DPO on {len(train_data)} examples, device={device}")
#     history = train_dpo(policy_model, ref_model, tokenizer, train_data,
#                         beta=0.1, n_epochs=3, device=device)
#
#     # Verification: loss should be monotonically decreasing (or at least trend down)
#     epoch_size = len(history["dpo_loss"]) // 3
#     epoch_means = [np.mean(history["dpo_loss"][i*epoch_size:(i+1)*epoch_size])
#                    for i in range(3)]
#     print(f"Epoch mean losses: {[f'{m:.4f}' for m in epoch_means]}")
#     assert epoch_means[0] > epoch_means[-1], "Loss did not decrease over training!"
#     print("Verification PASSED: DPO loss decreased over training.")
# else:
#     print("Skipping training: install transformers and datasets to run.")

# %% [markdown]
# ## Part 4: Ablation — Beta Sensitivity
#
# Run DPO with $\beta \in \{0.05, 0.1, 0.5\}$ and plot the **implicit reward gap** (winner minus
# loser) vs. training steps.
#
# **Expected result:**
# - Small $\beta$ (e.g., 0.05): Fast reward separation, but may stray far from reference
# - Large $\beta$ (e.g., 0.5): Slower separation, stays closer to reference distribution
# - Medium $\beta$ (0.1): Typical default, balances both
#
# *(Why does beta control this? Because larger beta penalizes KL divergence more heavily,
# meaning the policy cannot move as far from the reference to maximize reward.)*

# %%
def run_beta_ablation(policy_model_factory, ref_model_factory, tokenizer,
                      train_data: List[Dict],
                      betas: List[float] = [0.05, 0.1, 0.5],
                      n_epochs: int = 1,
                      device: str = "cpu") -> Dict[float, List[float]]:
    """
    Run DPO training for each beta value.
    Returns dict mapping beta → list of reward_margin values over training.

    PROVIDED — do not modify. Implement dpo_loss and compute_implicit_reward above.
    """
    results: Dict[float, List[float]] = {}

    for beta in betas:
        print(f"\nRunning beta={beta}...")
        policy = policy_model_factory()
        ref = ref_model_factory()
        ref.requires_grad_(False)

        history = train_dpo(policy, ref, tokenizer, train_data,
                            beta=beta, n_epochs=n_epochs, device=device,
                            run_name=f"dpo_beta_{beta}")
        results[beta] = history["reward_margin"]

    return results


def plot_beta_ablation(results: Dict[float, List[float]]) -> None:
    """Plot reward margin vs. training steps for each beta."""
    plt.figure(figsize=(10, 5))
    for beta, margins in results.items():
        plt.plot(margins, label=f"β={beta}")
    plt.xlabel("Training step")
    plt.ylabel("Implicit reward margin (winner − loser)")
    plt.title("DPO Beta Ablation: Reward Margin vs. Training Steps")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Uncomment to run ablation:
# if TRANSFORMERS_AVAILABLE and DATASETS_AVAILABLE:
#     MODEL_NAME = "gpt2"
#     factory = lambda: AutoModelForCausalLM.from_pretrained(MODEL_NAME)
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     tokenizer.pad_token = tokenizer.eos_token
#     train_data = load_hh_rlhf_subset(tokenizer, max_examples=200)
#
#     ablation_results = run_beta_ablation(factory, factory, tokenizer, train_data)
#     plot_beta_ablation(ablation_results)

# %% [markdown]
# ## Part 5: Reflection Questions
#
# Answer the following in the markdown cell below. There is no single correct answer;
# demonstrate conceptual understanding.
#
# **Q1: What reward does DPO implicitly optimize?**
# Consider: how does `compute_implicit_reward` relate to the RLHF reward objective?
# What happens to the KL term?
#
# **Q2: Why can't DPO incorporate process rewards?**
# DPO optimizes over *complete sequences* using offline preference pairs. How does this
# architectural choice prevent process-level (step-by-step) reward signals from being used?
# What would you need to change to support process rewards in a DPO-like framework?
#
# **Q3: When would you prefer DPO over PPO-RLHF?**
# Consider: dataset requirements, compute budget, stability, and reward hacking risks.

# %% [markdown]
# ### Your Answers
#
# **A1 (Implicit reward):**
# _Your answer here._
#
# **A2 (Process rewards):**
# _Your answer here._
#
# **A3 (DPO vs PPO-RLHF):**
# _Your answer here._
