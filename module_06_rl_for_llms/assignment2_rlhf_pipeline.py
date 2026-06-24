# %% [markdown]
# # Module 06, Assignment 2: RLHF PPO Pipeline
#
# **Prerequisites:**
# - Assignment 1: Trained reward model (or use the provided checkpoint)
# - Lecture notes sections 3–4
# - trl library (installed via pyproject.toml)
#
# **Learning Objectives:**
# 1. Understand the full RLHF pipeline: SFT → RM → PPO
# 2. Implement KL penalty for trust region enforcement
# 3. Observe reward/KL tradeoff during training
# 4. Understand reward hacking as KL grows

# %% [markdown]
# ## Part 1: Theory Recap
#
# The RLHF objective modifies the raw RM reward with a KL penalty:
#
# $$R_{total}(x,y) = R_{RM}(x,y) - \beta\,\text{KL}\!\left(\pi_\theta(y|x)\,\|\,\pi_{ref}(y|x)\right)$$
#
# Expanding the KL term per-token:
#
# $$\text{KL}(\pi_\theta \| \pi_{ref}) = \sum_{t=1}^{T} \log\frac{\pi_\theta(a_t | s_t)}{\pi_{ref}(a_t | s_t)}$$
#
# **Why is the per-token sum the right KL?**
# The probability of a sequence under an autoregressive LM factors as
# $\pi(y|x) = \prod_t \pi(a_t|s_t)$, so $\log \pi(y|x) = \sum_t \log \pi(a_t|s_t)$.
# The KL divergence between the *sequence* distributions therefore equals the sum of
# per-token KL terms.

# %%
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
from typing import List

# %% [markdown]
# ## Part 2: Implement KL Penalty and RLHF Reward
#
# ### KL Penalty
#
# The per-token KL penalty is:
#
# $$\text{KL-penalty}(x,y) = \beta \sum_{t=1}^{T} \left(\log \pi_\theta(a_t|s_t) - \log \pi_{ref}(a_t|s_t)\right)$$
#
# *(Why sum over tokens, not mean?)* The KL divergence between sequences is a sum, not a mean.
# Using the mean would underweight longer sequences and cause inconsistent trust regions
# across different generation lengths.

# %%
def compute_kl_penalty(log_probs_policy: torch.Tensor,
                       log_probs_ref: torch.Tensor,
                       beta: float = 0.1) -> torch.Tensor:
    """
    Per-token KL penalty: beta * sum(log_pi_theta - log_pi_ref) over token dimension.
    This is the soft trust region that prevents reward hacking.

    Args:
        log_probs_policy: (batch, seq_len) log probabilities under current policy π_θ
        log_probs_ref: (batch, seq_len) log probabilities under reference policy π_ref
        beta: KL coefficient (higher = stronger regularization toward π_ref)

    Returns:
        kl_penalties: (batch,) per-sample KL penalty (positive = π_θ diverged from π_ref)

    Hint: The KL divergence D_KL(p||q) = sum(p * (log p - log q)).
    Here we approximate it token-by-token: just sum (log_pi_theta - log_pi_ref) per sequence.
    This is the "unilateral" KL approximation used in InstructGPT.
    """
    raise NotImplementedError


def compute_rlhf_reward(response_ids: torch.Tensor, reward_model: nn.Module,
                        policy_model, ref_model,
                        tokenizer, beta: float = 0.1) -> torch.Tensor:
    """
    R_total = R_RM(x, y) - beta * KL(pi_theta || pi_ref)
    Returns per-sample scalar reward for PPOTrainer.

    Args:
        response_ids: (batch, seq_len) token IDs of generated responses
        reward_model: SentimentRewardModel or compatible scorer
        policy_model: current policy (AutoModelForCausalLMWithValueHead)
        ref_model: frozen reference model (AutoModelForCausalLM)
        tokenizer: shared tokenizer
        beta: KL coefficient

    Returns:
        rewards: (batch,) total reward per sample

    Steps:
    1. Get R_RM scores from reward_model (pass response_ids + lengths)
    2. Compute log_probs under policy_model for response_ids
    3. Compute log_probs under ref_model for response_ids (no grad)
    4. Compute KL penalty via compute_kl_penalty
    5. Return R_RM - KL_penalty
    """
    raise NotImplementedError


# %% [markdown]
# ## Part 3: RLHF Training with trl PPOTrainer
#
# The setup below is provided. We use GPT-2 Small fine-tuned on IMDB as the SFT model,
# and our trained reward model from Assignment 1 to score generations.
#
# **Pipeline:**
# 1. Encode a prompt (IMDB review prefix)
# 2. Policy generates a continuation (the "response")
# 3. Reward model scores the continuation
# 4. PPO updates the policy to maximize total reward

# %%
def build_rlhf_trainer(reward_model_path: str = "reward_model.pt",
                       model_name: str = "lvwerra/gpt2-imdb"):
    """Set up the trl PPOTrainer for RLHF fine-tuning.

    Args:
        reward_model_path: path to saved SentimentRewardModel state dict
        model_name: HuggingFace model ID (GPT-2 fine-tuned on IMDB — already SFT'd)

    Returns:
        (ppo_trainer, tokenizer, reward_model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1.41e-5,
        batch_size=16,
        mini_batch_size=4,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        log_with="tensorboard",
        project_kwargs={"logging_dir": "runs/rlhf_ppo"},
        kl_penalty="kl",
        init_kl_coef=0.2,   # Initial beta
        target=6,            # Target KL (adaptive controller)
        horizon=10000,
    )

    # Policy model (with value head for PPO)
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

    # Reference model (frozen SFT checkpoint)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

    # IMDB dataset for prompts
    dataset = load_dataset("imdb", split="train")
    dataset = dataset.rename_columns({"text": "review"})

    def tokenize_prompt(sample):
        # Use first 32 tokens as prompt
        encoding = tokenizer(sample["review"], truncation=True, max_length=32,
                             padding="max_length", return_tensors="pt")
        sample["input_ids"] = encoding["input_ids"][0]
        sample["query"] = tokenizer.decode(encoding["input_ids"][0])
        return sample

    dataset = dataset.map(tokenize_prompt)
    dataset.set_format("torch")

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=lambda data: dict(
            (key, torch.stack([d[key] for d in data]))
            for key in data[0] if isinstance(data[0][key], torch.Tensor)
        ),
    )

    return ppo_trainer, tokenizer, policy_model, ref_model


def run_rlhf_training(n_steps: int = 200, beta: float = 0.1):
    """Run RLHF PPO training loop.

    Logs to TensorBoard:
    - train/kl_from_ref: KL divergence from reference model (should stay < 10 nats)
    - train/reward_mean: mean total reward per step (should increase)
    - train/reward_std: standard deviation of rewards

    Verification: reward increases while KL stays < 10 nats within 200 PPO steps.
    """
    from rllearn.logging import make_writer
    import numpy as np

    writer = make_writer("rlhf_ppo")
    ppo_trainer, tokenizer, policy_model, ref_model = build_rlhf_trainer()

    # Generation settings
    gen_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 64,
    }

    for step, batch in enumerate(ppo_trainer.dataloader):
        if step >= n_steps:
            break

        query_tensors = batch["input_ids"]

        # Generate responses from current policy
        response_tensors = ppo_trainer.generate(query_tensors, **gen_kwargs)

        # Decode for reward model scoring
        responses = [
            tokenizer.decode(r.squeeze(), skip_special_tokens=True)
            for r in response_tensors
        ]

        # Compute rewards (using trl's built-in KL handling for simplicity here)
        # In a full custom implementation you would call compute_rlhf_reward
        rewards = [torch.tensor(float(len(r) > 0)) for r in responses]  # placeholder

        # PPO update
        stats = ppo_trainer.step(
            list(query_tensors),
            list(response_tensors),
            rewards
        )

        # Log metrics
        kl = stats.get("objective/kl", 0.0)
        reward_mean = np.mean([r.item() for r in rewards])
        reward_std = np.std([r.item() for r in rewards])

        writer.add_scalar("train/kl_from_ref", kl, step)
        writer.add_scalar("train/reward_mean", reward_mean, step)
        writer.add_scalar("train/reward_std", reward_std, step)

        if step % 20 == 0:
            print(f"Step {step:4d} | kl={kl:.3f} | reward={reward_mean:.3f} ± {reward_std:.3f}")

    writer.close()
    print("Training complete. Open TensorBoard to inspect reward/KL curves.")


# %% [markdown]
# ### Run Training

# %%
# %load_ext tensorboard
# %tensorboard --logdir runs/

# %%
# Uncomment to run (requires ~4GB RAM, ~10 min on CPU):
# run_rlhf_training(n_steps=200, beta=0.1)

# %% [markdown]
# **Verification:** After 200 PPO steps:
# - `train/reward_mean` should be higher than the initial reward (SFT baseline)
# - `train/kl_from_ref` should stay below 10 nats
#
# If KL diverges rapidly (> 10 nats in < 50 steps), increase `beta`.
# If reward barely improves, decrease `beta` or check that the reward model is loaded correctly.

# %% [markdown]
# ## Part 4: Beta Ablation — Reward/KL Tradeoff
#
# The $\beta$ hyperparameter controls the reward/KL tradeoff:
# - Small $\beta$: more reward, more KL (risk of hacking)
# - Large $\beta$: less KL, less reward improvement

# %%
def run_beta_ablation(betas: List[float] = [0.01, 0.1, 0.5], n_steps: int = 100):
    """Train RLHF with different beta values and compare reward/KL curves.

    Expected findings:
    - beta=0.01: high reward, but KL grows quickly (reward hacking risk)
    - beta=0.1: balanced — reward improves while KL stays bounded
    - beta=0.5: KL stays near zero, but reward barely improves (KL collapse)
    """
    import matplotlib.pyplot as plt
    results = {}

    for beta in betas:
        print(f"\n--- Beta = {beta} ---")
        from rllearn.logging import make_writer
        import numpy as np

        writer = make_writer(f"rlhf_beta_{beta}")
        ppo_trainer, tokenizer, policy_model, ref_model = build_rlhf_trainer()

        gen_kwargs = dict(
            min_length=-1, top_k=0, top_p=1.0,
            do_sample=True, pad_token_id=tokenizer.eos_token_id, max_new_tokens=32
        )

        step_rewards, step_kls = [], []

        for step, batch in enumerate(ppo_trainer.dataloader):
            if step >= n_steps:
                break

            query_tensors = batch["input_ids"]
            response_tensors = ppo_trainer.generate(query_tensors, **gen_kwargs)
            rewards = [torch.tensor(1.0) for _ in response_tensors]  # placeholder

            stats = ppo_trainer.step(list(query_tensors), list(response_tensors), rewards)
            kl = stats.get("objective/kl", 0.0)
            reward_mean = float(np.mean([r.item() for r in rewards]))

            step_rewards.append(reward_mean)
            step_kls.append(kl)
            writer.add_scalar("train/reward_mean", reward_mean, step)
            writer.add_scalar("train/kl_from_ref", kl, step)

        writer.close()
        results[beta] = {"rewards": step_rewards, "kls": step_kls}

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for beta, data in results.items():
        axes[0].plot(data["rewards"], label=f"β={beta}")
        axes[1].plot(data["kls"], label=f"β={beta}")

    axes[0].set_title("Reward vs Training Steps")
    axes[0].set_xlabel("PPO Step")
    axes[0].set_ylabel("Mean Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("KL from Reference vs Training Steps")
    axes[1].set_xlabel("PPO Step")
    axes[1].set_ylabel("KL Divergence (nats)")
    axes[1].axhline(10, color="red", linestyle="--", label="KL limit (10 nats)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("beta_ablation.png", dpi=150)
    plt.show()
    return results


# Uncomment to run:
# ablation_results = run_beta_ablation()

# %% [markdown]
# ## Part 5: Reflection Questions
#
# **Q1. What happens at β=0?**
# > With $\beta = 0$, there is no KL penalty. The policy is free to diverge arbitrarily from
# > $\pi_{ref}$. PPO will find whatever text maximizes $R_{RM}$ — which is likely not meaningful
# > text. The reward model was trained on human-generated text, and its extrapolation to
# > out-of-distribution text is unreliable. Common outcomes: degenerate repetitive sequences,
# > token sequences that "look" high-reward but are meaningless, rapid entropy collapse.
# > This is reward hacking in its most extreme form.
#
# **Q2. How does the KL penalty connect to the KL constraint in TRPO?**
# > TRPO (Trust Region Policy Optimization) constrains the policy update by requiring
# > $\text{KL}(\pi_{old} \| \pi_{new}) \leq \delta$ as a hard constraint, enforced via
# > conjugate gradient and line search. The RLHF KL penalty is a **Lagrangian relaxation**
# > of this constraint: instead of hard $\text{KL} \leq \delta$, we add $\beta \cdot \text{KL}$
# > to the objective. By the KKT conditions, at the optimum there exists $\beta^*$ such that
# > the soft and hard constrained solutions are equivalent. In practice, adaptive $\beta$ schedules
# > (adjust $\beta$ to maintain target KL) make this connection explicit.
#
# **Q3 (Bonus). Why does trl use a value head on the same model rather than a separate value network?**
# > Write your answer here.
