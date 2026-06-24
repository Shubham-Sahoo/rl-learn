# %% [markdown]
# # Module 06, Assignment 2: RLHF PPO Pipeline

# %%
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
from typing import List

# %% [markdown]
# ## Part 2: Implement KL Penalty and RLHF Reward

# %%
def compute_kl_penalty(log_probs_policy: torch.Tensor,
                       log_probs_ref: torch.Tensor,
                       beta: float = 0.1) -> torch.Tensor:
    """
    Per-token KL penalty: beta * sum(log_pi_theta - log_pi_ref) over token dimension.
    """
    kl = (log_probs_policy - log_probs_ref).sum(dim=-1)
    return beta * kl


def compute_rlhf_reward(response_ids: torch.Tensor, reward_model: nn.Module,
                        policy_model, ref_model,
                        tokenizer, beta: float = 0.1) -> torch.Tensor:
    """R_total = R_RM(x, y) - beta * KL(pi_theta || pi_ref)"""
    batch_size = response_ids.shape[0]

    # Get RM scores
    lengths = (response_ids != tokenizer.pad_token_id).sum(dim=-1).cpu()
    with torch.no_grad():
        rm_scores = reward_model(response_ids, lengths)

    # Get log probs under policy
    with torch.no_grad():
        policy_outputs = policy_model(response_ids, labels=response_ids)
        # log probs per token: shape (batch, seq)
        log_probs_policy = -policy_outputs.loss.unsqueeze(0).expand(batch_size)

    # Get log probs under ref
    with torch.no_grad():
        ref_outputs = ref_model(response_ids, labels=response_ids)
        log_probs_ref = -ref_outputs.loss.unsqueeze(0).expand(batch_size)

    kl_penalty = compute_kl_penalty(log_probs_policy, log_probs_ref, beta=beta)
    return rm_scores - kl_penalty


# %% [markdown]
# ## Part 3: RLHF Training with trl PPOTrainer

# %%
def build_rlhf_trainer(reward_model_path: str = "reward_model.pt",
                       model_name: str = "lvwerra/gpt2-imdb"):
    """Set up the trl PPOTrainer for RLHF fine-tuning."""
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
        init_kl_coef=0.2,
        target=6,
        horizon=10000,
    )

    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

    dataset = load_dataset("imdb", split="train")
    dataset = dataset.rename_columns({"text": "review"})

    def tokenize_prompt(sample):
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


# %% [markdown]
# ## Part 4: Reflection

# %% [markdown]
# **Answers:**
# 1. Without the KL penalty, the policy can diverge arbitrarily from the reference model,
#    leading to reward hacking: high RM scores for degenerate or incoherent text.
# 2. Beta too small: policy can exploit the RM (reward hacking). Beta too large: policy stays
#    too close to the reference and doesn't learn to maximize reward.
# 3. The KL divergence grows monotonically with training if not constrained. The adaptive
#    controller adjusts beta to keep KL near the target, preventing catastrophic forgetting.
