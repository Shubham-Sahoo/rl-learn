# %% [markdown]
# # Module 07, Assignment 3: VLM Alignment
#
# ## Prerequisites
# - Module 07 A1: DPO loss implementation
# - Lecture notes sections 5–7
# - peft library (for LoRA)
#
# ## Learning Objectives
# 1. Implement the CHAIR hallucination metric
# 2. Train a hallucination reward model on image-caption pairs
# 3. Apply DPO with LoRA to reduce VLM hallucinations
# 4. Understand VLM-specific alignment challenges

# %% [markdown]
# ## Part 0: Theory Recap — VLM Hallucination and CHAIR
#
# **CHAIR$_s$** (sentence-level hallucination rate):
#
# $$\text{CHAIR}_s = \frac{|\{s : \exists \text{ hallucinated object mention in sentence } s\}|}{|\text{sentences}|}$$
#
# **CHAIR$_i$** (instance-level hallucination rate):
#
# $$\text{CHAIR}_i = \frac{|\text{hallucinated object mentions}|}{|\text{total object mentions}|}$$
#
# **DPO loss for VLM** (same formula as text DPO, conditioned on image tokens):
#
# $$\mathcal{L}_\text{DPO}^{VLM} = -\mathbb{E}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|v,x)}{\pi_\text{ref}(y_w|v,x)} - \beta \log \frac{\pi_\theta(y_l|v,x)}{\pi_\text{ref}(y_l|v,x)}\right)\right]$$
#
# where $v$ is the visual input (image tokens) and $x$ is the text prompt.
#
# *(Why does VLM DPO need the image in the forward pass? Because log P(caption | image) ≠
# log P(caption) — the probability of a caption is conditioned on both the image and text context.
# A hallucinatory caption about a non-existent object has high text likelihood but low
# image-conditioned likelihood in a well-aligned model.)*

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import re
import random
from typing import List, Dict, Tuple, Optional, Set

from rllearn.logging import make_writer

# %% [markdown]
# ## Part 1: CHAIR Metric
#
# Implement both CHAIR variants.
#
# **Algorithm:**
# 1. For each generated caption, extract all mentioned object nouns
# 2. Compare to the ground-truth object set for that image
# 3. An object mention is "hallucinated" if the object does not appear in the ground-truth set
#
# **CHAIR$_s$:** For each caption, split into sentences. If any sentence contains a hallucinated
# object, that sentence is marked. CHAIR$_s$ = fraction of marked sentences.
#
# **CHAIR$_i$:** Count total mentioned objects and total hallucinated objects across all captions.
# CHAIR$_i$ = hallucinated / total.
#
# *(Why split into sentences for CHAIR$_s$? Because a long caption might have one hallucination
# in one sentence but be otherwise accurate. CHAIR$_s$ measures the proportion of sentences
# contaminated by hallucination, not the proportion of words.)*

# %%
def extract_object_mentions(text: str, object_vocabulary: Set[str]) -> List[str]:
    """
    Extract all object mentions from a text string that appear in the vocabulary.

    Args:
        text: Caption or sentence string
        object_vocabulary: Set of known object category names (lowercase)

    Returns:
        List of mentioned objects (may contain duplicates)
    """
    # TODO: Tokenize text, lowercase, return tokens that appear in object_vocabulary
    raise NotImplementedError


def compute_chair_score(generated_captions: List[str],
                        ground_truth_objects: List[List[str]],
                        object_vocabulary: Optional[Set[str]] = None) -> dict:
    """
    CHAIR_s = fraction of sentences with ≥1 hallucinated object.
    CHAIR_i = fraction of mentioned objects that are hallucinated.

    Args:
        generated_captions: List of generated caption strings, one per image
        ground_truth_objects: List of ground-truth object sets, one per image.
                              Each inner list contains the object names present in that image.
        object_vocabulary: Set of all known object names. If None, inferred from ground_truth_objects.

    Returns:
        dict with keys 'CHAIR_s' (float) and 'CHAIR_i' (float)
    """
    raise NotImplementedError

# %% [markdown]
# ### Verify CHAIR Implementation
#
# Before training, verify your CHAIR implementation on simple examples.
#
# **Test case:**
# - Caption 1: "A cat sitting on a table with a vase." (ground truth: cat, table)
#   - Hallucinated: vase → CHAIR_s sentence has hallucination
# - Caption 2: "A dog running in the park." (ground truth: dog, park)
#   - No hallucination
#
# Expected: CHAIR_s = 0.5 (1 out of 2 captions has hallucination); CHAIR_i = 1/4 = 0.25

# %%
def test_chair_basic() -> bool:
    """Verify CHAIR implementation on known examples."""
    captions = [
        "A cat sitting on a table with a vase.",
        "A dog running in the park.",
    ]
    gt_objects = [
        ["cat", "table"],
        ["dog", "park"],
    ]
    vocab = {"cat", "table", "vase", "dog", "park"}
    result = compute_chair_score(captions, gt_objects, vocab)

    # CHAIR_s: caption 1 has hallucination (vase not in gt), caption 2 does not → 0.5
    # CHAIR_i: total mentions = cat, table, vase, dog, park = 5; hallucinated = vase = 1 → 0.2
    print(f"CHAIR_s: {result['CHAIR_s']:.3f} (expected ~0.5)")
    print(f"CHAIR_i: {result['CHAIR_i']:.3f} (expected ~0.2)")
    return abs(result["CHAIR_s"] - 0.5) < 0.01 and abs(result["CHAIR_i"] - 0.2) < 0.01


# assert test_chair_basic(), "CHAIR implementation incorrect"
# print("CHAIR unit test passed.")

# %% [markdown]
# ## Part 2: Hallucination Reward Model
#
# Implement a binary classifier that predicts whether a caption is faithful to an image.
#
# **Architecture:**
# - Project vision features: `vision_dim → hidden_dim` (linear)
# - Project text features: `text_dim → hidden_dim` (linear)
# - Concatenate projected features: `2 * hidden_dim`
# - MLP: `2 * hidden_dim → hidden_dim → 1` with ReLU → sigmoid
#
# *(Why concatenate instead of cross-attention? For simplicity in this assignment. Real VLM
# reward models use cross-attention to allow vision features to modulate text scoring.)*

# %%
class HallucinationRewardModel(nn.Module):
    """Binary classifier: does caption hallucinate objects not in image?

    Returns P(faithful) — higher = more faithful, lower = more hallucinatory.
    """

    def __init__(self, vision_dim: int = 768, text_dim: int = 768,
                 hidden_dim: int = 256):
        super().__init__()
        # TODO: project vision + text → concat → MLP → scalar (sigmoid)
        raise NotImplementedError

    def forward(self, vision_features: torch.Tensor,
                text_features: torch.Tensor) -> torch.Tensor:
        """Return P(faithful) in [0,1] of shape (batch,).

        Args:
            vision_features: Shape (batch, vision_dim)
            text_features: Shape (batch, text_dim)

        Returns:
            faithfulness_score: Shape (batch,) with values in [0, 1]
        """
        raise NotImplementedError

# %% [markdown]
# ## Part 3: VLM DPO Loss
#
# Adapt the DPO loss for the VLM setting. The key difference: the log-probability of a response
# is conditioned on BOTH image tokens and text prompt tokens.
#
# **Batch format:**
# ```python
# batch = {
#     'image_features': Tensor (batch, vision_dim),   # pre-extracted image features
#     'prompt_ids': Tensor (batch, prompt_len),        # tokenized text prompt
#     'winner_ids': Tensor (batch, win_len),           # preferred caption tokens
#     'loser_ids': Tensor (batch, lose_len),           # rejected caption tokens
# }
# ```
#
# **Simplification for this assignment:** We use pre-extracted features (no actual VLM inference).
# The `policy_vlm` and `ref_vlm` are `HallucinationRewardModel` instances that score captions.
# The "log-prob" is approximated as `log(reward_model_score)`.
#
# *(For full LLaVA-1.5-7B fine-tuning with LoRA, see the bonus section at the end.)*

# %%
def dpo_vlm_loss(policy_vlm: HallucinationRewardModel,
                 ref_vlm: HallucinationRewardModel,
                 batch: dict,
                 beta: float = 0.1) -> torch.Tensor:
    """
    DPO loss adapted for VLM: log-probs conditioned on both image + text prompt.
    batch keys: image_features, prompt_ids, winner_ids, loser_ids.
    Same formula as text DPO but forward pass includes image tokens.

    Simplification: we use HallucinationRewardModel scores as proxies for
    conditional log-probs. score → log(score) approximates log P(caption | image).

    Args:
        policy_vlm: The model being trained (gradients flow through this)
        ref_vlm: The frozen reference model (no gradient)
        batch: Dict with keys image_features, winner_features, loser_features
               (pre-extracted text embeddings for winner/loser captions)
        beta: KL penalty coefficient

    Returns:
        loss: Scalar DPO loss
    """
    raise NotImplementedError

# %% [markdown]
# ## Part 4: Training — Hallucination Reduction
#
# We use a simplified setup with pre-extracted features (no actual VLM inference required).
#
# **Toy task:** MSCOCO-style captions with known object sets.
# - Faithful captions: mention only objects present in the image
# - Hallucinatory captions: mention 1–2 objects NOT in the image
#
# **Training pipeline:**
# 1. Pre-train HallucinationRewardModel as a binary classifier (faithful vs. hallucinatory)
# 2. Create preference pairs: faithful caption = winner, hallucinatory = loser
# 3. Apply DPO to push the policy toward faithful captions

# %%
COCO_OBJECTS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "ball", "kite", "baseball bat",
    "skateboard", "surfboard", "bottle", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "orange", "pizza", "chair", "couch",
    "plant", "bed", "desk", "toilet", "tv", "laptop", "phone",
    "microwave", "oven", "sink", "refrigerator", "book", "clock", "vase",
]


def generate_toy_vlm_dataset(n_images: int = 300,
                              vision_dim: int = 768,
                              text_dim: int = 768,
                              seed: int = 42) -> List[Dict]:
    """
    Generate a synthetic VLM dataset with pre-extracted features.

    Each example:
    - image_features: (vision_dim,) tensor — simulates CLIP image features
    - faithful_features: (text_dim,) tensor — simulates faithful caption features
    - hallucinatory_features: (text_dim,) tensor — simulates hallucinatory caption features
    - faithful_caption: str
    - hallucinatory_caption: str
    - gt_objects: List[str] — objects present in the image

    PROVIDED — do not modify.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    dataset = []

    for i in range(n_images):
        # Sample 3–5 ground-truth objects
        gt_objects = rng.sample(COCO_OBJECTS, rng.randint(3, 5))
        hallucination_objects = [o for o in COCO_OBJECTS if o not in gt_objects]
        hallucinated = rng.sample(hallucination_objects, 2)

        faithful_caption = f"An image with {', '.join(gt_objects)}."
        hallucinatory_caption = f"An image with {', '.join(gt_objects[:2] + hallucinated)}."

        # Image features: base vector for this "image"
        image_feat = torch.tensor(np_rng.randn(vision_dim).astype(np.float32))

        # Faithful text features: correlated with image
        faithful_feat = image_feat + torch.tensor(
            np_rng.randn(text_dim).astype(np.float32)) * 0.3

        # Hallucinatory features: less correlated with image
        halluc_feat = image_feat + torch.tensor(
            np_rng.randn(text_dim).astype(np.float32)) * 1.5

        dataset.append({
            "image_features": image_feat,
            "faithful_features": faithful_feat,
            "hallucinatory_features": halluc_feat,
            "faithful_caption": faithful_caption,
            "hallucinatory_caption": hallucinatory_caption,
            "gt_objects": gt_objects,
        })

    return dataset


def pretrain_hallucination_rm(rm: HallucinationRewardModel,
                               dataset: List[Dict],
                               n_epochs: int = 5,
                               lr: float = 1e-3,
                               device: str = "cpu",
                               run_name: str = "halluc_rm_pretrain") -> List[float]:
    """
    Pre-train the hallucination reward model as a binary classifier.
    Faithful caption → label 1.0; hallucinatory → label 0.0.

    PROVIDED — do not modify.
    """
    writer = make_writer(run_name)
    optimizer = torch.optim.Adam(rm.parameters(), lr=lr)
    rm.to(device).train()
    losses = []

    for epoch in range(n_epochs):
        random.shuffle(dataset)
        epoch_losses = []
        for item in dataset:
            img = item["image_features"].unsqueeze(0).to(device)
            faithful = item["faithful_features"].unsqueeze(0).to(device)
            halluc = item["hallucinatory_features"].unsqueeze(0).to(device)

            optimizer.zero_grad()
            score_faithful = rm(img, faithful)     # should → 1
            score_halluc = rm(img, halluc)         # should → 0

            loss = (F.binary_cross_entropy(score_faithful, torch.ones_like(score_faithful)) +
                    F.binary_cross_entropy(score_halluc, torch.zeros_like(score_halluc)))
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = np.mean(epoch_losses)
        losses.append(mean_loss)
        writer.add_scalar("train/rm_bce", mean_loss, epoch)
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | RM BCE: {mean_loss:.4f}")

    writer.close()
    return losses


def train_dpo_vlm(policy_rm: HallucinationRewardModel,
                  ref_rm: HallucinationRewardModel,
                  dataset: List[Dict],
                  n_epochs: int = 5,
                  beta: float = 0.1,
                  lr: float = 5e-4,
                  device: str = "cpu",
                  run_name: str = "dpo_vlm") -> List[float]:
    """
    Apply DPO to push the policy RM toward faithful captions.

    PROVIDED — do not modify. Implement dpo_vlm_loss above.
    """
    writer = make_writer(run_name)
    optimizer = torch.optim.Adam(policy_rm.parameters(), lr=lr)
    ref_rm.to(device).eval()
    ref_rm.requires_grad_(False)
    policy_rm.to(device).train()

    losses = []
    global_step = 0

    for epoch in range(n_epochs):
        random.shuffle(dataset)
        epoch_losses = []

        for item in dataset:
            img = item["image_features"].unsqueeze(0).to(device)
            winner = item["faithful_features"].unsqueeze(0).to(device)
            loser = item["hallucinatory_features"].unsqueeze(0).to(device)

            batch = {
                "image_features": img,
                "winner_features": winner,
                "loser_features": loser,
            }

            optimizer.zero_grad()
            loss = dpo_vlm_loss(policy_rm, ref_rm, batch, beta=beta)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            writer.add_scalar("train/dpo_vlm_loss", loss.item(), global_step)
            global_step += 1

        mean_loss = np.mean(epoch_losses)
        losses.append(mean_loss)
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | DPO loss: {mean_loss:.4f}")

    writer.close()
    return losses


def evaluate_chair(rm: HallucinationRewardModel, dataset: List[Dict],
                   threshold: float = 0.5,
                   device: str = "cpu") -> Dict[str, float]:
    """
    Compute CHAIR on a dataset using the RM to classify captions.
    The RM decides whether to "output" the faithful or hallucinatory caption.

    PROVIDED — do not modify.
    """
    rm.eval()
    generated_captions = []
    gt_objects_list = []

    with torch.no_grad():
        for item in dataset:
            img = item["image_features"].unsqueeze(0).to(device)
            faithful_feat = item["faithful_features"].unsqueeze(0).to(device)
            halluc_feat = item["hallucinatory_features"].unsqueeze(0).to(device)

            # Use the RM score to choose which caption to "generate"
            faithful_score = rm(img, faithful_feat).item()
            halluc_score = rm(img, halluc_feat).item()

            # If the RM prefers faithful, output faithful caption (no hallucination)
            if faithful_score >= halluc_score:
                generated_captions.append(item["faithful_caption"])
            else:
                generated_captions.append(item["hallucinatory_caption"])

            gt_objects_list.append(item["gt_objects"])

    vocab = set(COCO_OBJECTS)
    return compute_chair_score(generated_captions, gt_objects_list, vocab)

# %% [markdown]
# ### Run Training Pipeline

# %%
# %load_ext tensorboard
# %tensorboard --logdir runs/

# %%
# PROVIDED PIPELINE — do not modify. Implement the functions above.
# Uncomment to run:
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")
#
# print("Generating toy VLM dataset...")
# full_dataset = generate_toy_vlm_dataset(n_images=300)
# train_data = full_dataset[:240]
# val_data = full_dataset[240:]
#
# # Step 1: Pre-train Hallucination RM
# print("\nPre-training hallucination reward model...")
# policy_rm = HallucinationRewardModel(vision_dim=768, text_dim=768, hidden_dim=256)
# pretrain_losses = pretrain_hallucination_rm(policy_rm, train_data, n_epochs=5, device=device)
#
# # Baseline CHAIR before DPO
# chair_before = evaluate_chair(policy_rm, val_data, device=device)
# print(f"\nCHAIR before DPO: CHAIR_s={chair_before['CHAIR_s']:.3f}, CHAIR_i={chair_before['CHAIR_i']:.3f}")
#
# # Step 2: DPO fine-tuning
# print("\nApplying DPO to reduce hallucinations...")
# ref_rm = HallucinationRewardModel(vision_dim=768, text_dim=768, hidden_dim=256)
# ref_rm.load_state_dict(policy_rm.state_dict())  # ref = pre-trained RM snapshot
#
# dpo_losses = train_dpo_vlm(policy_rm, ref_rm, train_data, n_epochs=5, beta=0.1, device=device)
#
# # CHAIR after DPO
# chair_after = evaluate_chair(policy_rm, val_data, device=device)
# print(f"CHAIR after DPO:  CHAIR_s={chair_after['CHAIR_s']:.3f}, CHAIR_i={chair_after['CHAIR_i']:.3f}")
#
# improvement = chair_before['CHAIR_s'] - chair_after['CHAIR_s']
# print(f"\nCHAIR_s reduction: {improvement*100:.1f} percentage points")
# assert improvement >= 0.05, f"CHAIR_s did not reduce by ≥5pp (got {improvement*100:.1f}pp)"
# print("Verification PASSED: CHAIR_s reduced by ≥5 percentage points.")

# %% [markdown]
# ## Part 5: Ablation — DPO Beta vs. KL Tradeoff
#
# Run DPO with $\beta \in \{0.05, 0.1, 0.5\}$ and plot CHAIR$_s$ vs. KL divergence from
# the reference.
#
# **Expected result:**
# - Small $\beta$: Better CHAIR reduction but larger KL divergence (drifts from reference)
# - Large $\beta$: Less CHAIR reduction but stays close to reference
#
# *(Why does this tradeoff matter for VLMs? Because large KL divergence from the reference
# can cause the model to "forget" visual grounding learned during pre-training. The reference
# model was carefully pre-trained; we want to improve alignment without destroying capabilities.)*

# %%
def compute_kl_from_ref(policy_rm: HallucinationRewardModel,
                        ref_rm: HallucinationRewardModel,
                        dataset: List[Dict],
                        device: str = "cpu") -> float:
    """
    Approximate KL divergence between policy and reference RM scores on the dataset.
    KL ≈ E[log(policy_score / ref_score)] over faithful captions.

    PROVIDED — do not modify.
    """
    policy_rm.eval()
    ref_rm.eval()
    kl_values = []

    with torch.no_grad():
        for item in dataset:
            img = item["image_features"].unsqueeze(0).to(device)
            feat = item["faithful_features"].unsqueeze(0).to(device)

            p_score = policy_rm(img, feat).clamp(1e-6, 1 - 1e-6)
            r_score = ref_rm(img, feat).clamp(1e-6, 1 - 1e-6)
            kl = (torch.log(p_score) - torch.log(r_score)).item()
            kl_values.append(kl)

    return float(np.mean(kl_values))


def run_beta_chair_ablation(train_data: List[Dict], val_data: List[Dict],
                             betas: List[float] = [0.05, 0.1, 0.5],
                             device: str = "cpu") -> Dict:
    """
    Run DPO for each beta and record CHAIR_s and KL divergence.

    PROVIDED — do not modify. Implement HallucinationRewardModel and dpo_vlm_loss.
    """
    results = {"betas": betas, "chair_s": [], "kl": []}

    for beta in betas:
        print(f"\nRunning beta={beta}...")
        policy_rm = HallucinationRewardModel(vision_dim=768, text_dim=768, hidden_dim=256)
        ref_rm = HallucinationRewardModel(vision_dim=768, text_dim=768, hidden_dim=256)
        pretrain_hallucination_rm(policy_rm, train_data, n_epochs=3, device=device,
                                  run_name=f"rm_pretrain_beta_{beta}")
        ref_rm.load_state_dict(policy_rm.state_dict())

        train_dpo_vlm(policy_rm, ref_rm, train_data, n_epochs=3,
                      beta=beta, device=device, run_name=f"dpo_vlm_beta_{beta}")

        chair = evaluate_chair(policy_rm, val_data, device=device)
        kl = compute_kl_from_ref(policy_rm, ref_rm, val_data, device=device)

        results["chair_s"].append(chair["CHAIR_s"])
        results["kl"].append(kl)
        print(f"beta={beta} | CHAIR_s={chair['CHAIR_s']:.3f} | KL={kl:.4f}")

    return results


def plot_chair_kl_tradeoff(results: Dict) -> None:
    """Plot CHAIR_s vs. KL divergence for each beta."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar([str(b) for b in results["betas"]], results["chair_s"], color=["blue", "green", "red"])
    ax1.set_xlabel("Beta")
    ax1.set_ylabel("CHAIR_s (lower = better)")
    ax1.set_title("CHAIR_s vs. Beta")
    ax1.grid(axis="y", alpha=0.3)

    ax2.scatter(results["kl"], results["chair_s"], s=100,
                c=["blue", "green", "red"][:len(results["betas"])])
    for beta, kl, cs in zip(results["betas"], results["kl"], results["chair_s"]):
        ax2.annotate(f"β={beta}", (kl, cs), textcoords="offset points", xytext=(5, 5))
    ax2.set_xlabel("KL divergence from reference")
    ax2.set_ylabel("CHAIR_s")
    ax2.set_title("CHAIR_s vs. KL Divergence Tradeoff")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# Uncomment to run ablation:
# ablation_results = run_beta_chair_ablation(train_data, val_data, betas=[0.05, 0.1, 0.5])
# plot_chair_kl_tradeoff(ablation_results)

# %% [markdown]
# ## Part 6: Bonus — Full LLaVA Fine-tuning with LoRA
#
# For students with access to a GPU ≥ 24GB VRAM.
#
# The following sketch shows how to apply LoRA DPO to the actual LLaVA-1.5-7B model.
# **This is a code sketch only — it will not run without additional setup.**
#
# ```python
# from transformers import LlavaForConditionalGeneration, AutoProcessor
# from peft import LoraConfig, get_peft_model, TaskType
#
# # Load model
# model = LlavaForConditionalGeneration.from_pretrained(
#     "llava-hf/llava-1.5-7b-hf",
#     torch_dtype=torch.float16,
#     device_map="auto",
# )
#
# # Apply LoRA — target only the language model attention layers
# # Vision encoder is frozen; LoRA limits LM drift during DPO
# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     r=64,                              # LoRA rank
#     lora_alpha=128,                    # scaling factor
#     target_modules=["q_proj", "v_proj"],  # attention weights only
#     lora_dropout=0.05,
#     bias="none",
# )
# policy_vlm = get_peft_model(model, lora_config)
# policy_vlm.print_trainable_parameters()
# # trainable params: ~8M / 7B = ~0.1% of total
#
# # Reference model: same base, no LoRA adapters (or merged adapters with zero init)
# ref_vlm = LlavaForConditionalGeneration.from_pretrained(
#     "llava-hf/llava-1.5-7b-hf",
#     torch_dtype=torch.float16,
#     device_map="auto",
# )
# ref_vlm.requires_grad_(False)
#
# # DPO training loop (same as dpo_vlm_loss but with real VLM forward passes)
# # Dataset: LVIS-Instruct4V or RLHF-V preference pairs
# ```
#
# *(Why LoRA for VLM DPO? Because full fine-tuning a 7B model requires ~28GB VRAM for the
# policy alone, plus another 28GB for the reference = 56GB. With LoRA, both share the same
# frozen base (28GB) and only the adapters differ (~32MB each). Total ≈ 30GB.)*

# %% [markdown]
# ## Part 7: Course Capstone Reflection
#
# This is the final assignment of the RL-Learn curriculum. The questions below are open-ended
# and are designed to help you synthesize everything you've learned.
#
# **Q1: What unsolved problem in VLM alignment would you tackle for your final project?**
# Consider: hallucination reduction, visual grounding, sycophancy, distributional shift,
# reward hacking in visual domains, or multimodal process rewards.
#
# **Q2: Which module's technique was most surprising to you, and why?**
# Consider: the connection between TD learning and PRM labeling, RLHF as policy gradient,
# DPO's implicit reward, Monte Carlo tree search as PRM beam search, etc.
#
# **Q3: How would you design a PRM for a VLM reasoning task?**
# Hint: What is a "step" in visual reasoning? How would you collect step-level labels?
# Could Monte Carlo labeling work? What would the oracle be?

# %% [markdown]
# ### Your Answers
#
# **A1 (VLM alignment research direction):**
# _Your answer here._
#
# **A2 (Most surprising technique):**
# _Your answer here._
#
# **A3 (PRM for VLM reasoning):**
# _Your answer here._
