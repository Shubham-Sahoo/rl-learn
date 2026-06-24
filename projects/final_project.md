# Final Project Brief

**Curriculum context:** Modules 01–07 complete (MDP foundations, value-based methods, policy gradients & PPO, model-based RL, advanced policy methods, RL for LLMs, alignment frontier).

**Due:** End of Module 07.

This is the capstone project. The scope is intentionally more ambitious than the midterm — each option is designed so that a strong execution could plausibly result in a workshop paper submission. Choose **one** option. Six weeks of part-time work (10–14 hours/week) is the intended scope.

---

## Option A: GRPO for Reasoning — Reproducing DeepSeek-R1 on a Toy Task

### Background

In early 2025, DeepSeek-R1 demonstrated that frontier-level mathematical reasoning could be elicited from a language model using Group Relative Policy Optimization (GRPO) — without a separate reward model. The key insight: if you have a verifiable oracle (a Python math checker, a theorem prover, a unit test runner), you can compute binary rewards directly and use them to train the policy via policy gradient. No learned reward model, no preference data, no RLHF pipeline. GRPO differs from standard PPO by computing advantages *within a group* of completions sampled for the same prompt, rather than using a value baseline. This makes it well-suited to settings where many rollouts per prompt are cheap.

This project reproduces the core technique at tractable scale. You will not reproduce DeepSeek-R1's compute budget; you will reproduce its *algorithmic idea* on problems small enough to verify that the mechanism works.

### Task Specification

Fine-tune GPT-2 Medium (or a comparable open-weight LM of your choice) on GSM8K-style arithmetic word problems using GRPO. Use a Python `eval()` oracle as the reward function: if the model's final answer, extracted via regex, evaluates to the correct integer, reward = 1; otherwise reward = 0. Implement from scratch:

1. **Group sampling** — for each prompt, sample G completions (G = 8 is a reasonable default).
2. **Group-normalized advantages** — normalize rewards within the group: `A_i = (r_i - mean(r)) / (std(r) + epsilon)`. This is the GRPO advantage estimator; refer to Assignment 3 in Module 06.
3. **Policy gradient update** — PPO-clip objective over the group advantages, with a KL penalty against the reference (frozen) model.
4. **Optional (encouraged):** Add a *format reward* — a small bonus (0.1–0.2) for answers that show step-by-step reasoning before the final answer, even if the final answer is wrong.

Compare GRPO-trained model vs an SFT baseline fine-tuned on gold solutions.

**Constraints:**
- Training must be feasible on a single consumer GPU (use gradient checkpointing, bf16, and/or LoRA if needed).
- The oracle must be deterministic and not use the model's own generations to avoid reward hacking.
- Evaluate on a held-out split (do not train and evaluate on the same prompts).

### Related Work

| Paper | Relevance |
|---|---|
| DeepSeek-R1 (DeepSeek AI, 2025) | Introduced GRPO for math reasoning; the direct inspiration for this project |
| RLVR: Reinforcement Learning with Verifiable Rewards (2024) | Systematic study of when verifiable rewards work and when they fail |
| STaR: Self-Taught Reasoner (Zelikman et al., 2022) | Earlier RL-style approach to bootstrapping reasoning; useful for comparison |
| Let's Verify Step by Step (Lightman et al., 2023) | Process reward models vs outcome reward models; relevant to the format reward extension |

### Baseline to Beat

GPT-2 Medium fine-tuned with standard SFT (cross-entropy loss on gold GSM8K solutions), evaluated by **pass@1** on the held-out split.

### Success Criteria

| Criterion | Threshold |
|---|---|
| GRPO pass@1 | ≥ 5 percentage points improvement over SFT baseline pass@1 |
| GRPO pass@8 (best of 8) | ≥ 15 percentage points improvement over SFT pass@8 |
| Training stability | KL from reference policy remains bounded (does not diverge); log to TensorBoard |
| Group advantage sanity | Verify that group-normalized advantages have zero mean and unit std per batch; include a diagnostic plot |

### Deliverables

| File | Description |
|---|---|
| `grpo_trainer.py` | GRPO training loop with group sampling and normalized advantages |
| `sft_baseline.py` | SFT fine-tuning script for the baseline |
| `evaluate_gsm8k.py` | Evaluation script: pass@1 and pass@8 on held-out split |
| `report.md` | 6-page maximum, NeurIPS format (abstract, intro, method, experiments, conclusion) |

### Suggested Timeline

| Week | Goal |
|---|---|
| 1 | SFT baseline: data loading, fine-tuning, evaluation infrastructure, pass@1 on held-out split |
| 2 | GRPO core: group sampling, advantage normalization, basic policy gradient update |
| 3 | GRPO + KL penalty; verify training is stable; match or beat SFT on pass@1 |
| 4 | Scaling experiments: vary G (group size), KL coefficient, learning rate; format reward ablation |
| 5 | Analysis: advantage distribution plots, generation examples, failure mode analysis |
| 6 | Write report in NeurIPS format; code cleanup |

### Workshop / Venue Suggestions

- ICLR 2026 Workshop on RL for Language Models
- ACL 2026 Student Research Workshop
- NeurIPS 2026 Workshop on Instruction Tuning and Fine-Tuning

---

## Option B: VLM Alignment Research — Reducing Hallucinations via DPO

### Background

Vision-language models hallucinate at a surprisingly high rate: objects that are not present in an image appear in the generated caption, relationships between objects are fabricated, and attributes are assigned to the wrong referents. This is not merely an accuracy problem — in high-stakes applications (medical imaging, autonomous systems, accessibility tools), hallucinated descriptions can cause direct harm. Standard fine-tuning on correct captions improves fluency but does not specifically penalize hallucination. Direct Preference Optimization (DPO) offers a different lever: given pairs of (preferred output, dispreferred output) for the same input, DPO directly maximizes the likelihood ratio between them without needing a separate reward model. Applied to VLM hallucination, DPO can penalize hallucinated captions while rewarding grounded ones.

The key challenge in this project is **preference pair construction**: you must programmatically generate plausible-sounding but hallucinated captions as the dispreferred examples, using the MSCOCO annotation structure to know which objects are genuinely present.

### Task Specification

Fine-tune a small VLM — BLIP-2 (recommended for compute efficiency) or LLaVA-1.5-7B with LoRA + 4-bit quantization — on MSCOCO using DPO. Construct preference pairs programmatically:

- **Winner:** A ground-truth MSCOCO caption for the image.
- **Loser:** A paraphrased version of the winner with one or more hallucinated objects injected (objects from the COCO category list that do NOT appear in the image's ground-truth annotations).

Implement the DPO objective from scratch (or adapt the Module 07 assignment implementation). Measure CHAIR_s (sentence-level hallucination rate) and CHAIR_i (instance-level hallucination rate) before and after DPO fine-tuning on the MSCOCO validation split.

**Constraints:**
- The hallucinated objects injected into loser captions must be semantically plausible (same superclass as a present object, or visually similar objects — not random). Document your injection strategy.
- Evaluate captioning quality using CIDEr in addition to CHAIR; report both.
- At least one ablation: vary the DPO beta (temperature) parameter and report effect on CHAIR vs CIDEr tradeoff.

### Related Work

| Paper | Relevance |
|---|---|
| LLaVA-RLHF (Sun et al., 2023) | VLM alignment via RLHF; establishes the problem framing |
| Hallucination Augmented Contrastive Learning (Jiang et al., 2023) | HACL: contrastive approach to reducing VLM hallucination; a key baseline |
| RLHF-V (Yu et al., 2024) | Dense preference feedback for fine-grained VLM behavior correction |
| VLFeedback (Li et al., 2024) | Large-scale scalable preference data construction for VLMs |

### Baseline to Beat

Base VLM (BLIP-2 or LLaVA-1.5-7B) without any DPO fine-tuning, measured by CHAIR_s and CHAIR_i on MSCOCO validation.

### Success Criteria

| Criterion | Threshold |
|---|---|
| CHAIR_s reduction | ≥ 5 percentage points reduction after DPO fine-tuning |
| CHAIR_i reduction | ≥ 3 percentage points reduction after DPO fine-tuning |
| CIDEr preservation | CIDEr score does not degrade by more than 5 points (hallucination reduction should not destroy captioning quality) |
| Beta ablation | At least 3 values of DPO beta reported; CHAIR vs CIDEr tradeoff curve shown |

### Deliverables

| File | Description |
|---|---|
| `construct_preference_pairs.py` | Script that builds the DPO dataset from MSCOCO annotations |
| `dpo_vlm.py` | DPO fine-tuning script with LoRA and/or 4-bit quantization |
| `evaluate_chair.py` | CHAIR evaluation script on MSCOCO validation split |
| `report.md` | 6-page maximum, vision track format (abstract, intro, method, experiments, conclusion) |

### Suggested Timeline

| Week | Goal |
|---|---|
| 1 | Data pipeline: MSCOCO download, annotation parsing, preference pair construction, verify loser quality |
| 2 | DPO implementation: loss derivation, reference model freezing, LoRA setup; overfit on small batch to verify correctness |
| 3 | Full fine-tuning run; CHAIR and CIDEr evaluation before/after |
| 4 | Beta ablation; analyze failure cases (images where CHAIR improved vs degraded) |
| 5 | Additional ablations (loser injection strategy, dataset size); comparison to SFT baseline |
| 6 | Write report; code cleanup |

### Workshop / Venue Suggestions

- CVPR 2026 Workshop on Responsible Computer Vision
- ECCV 2026
- NAACL 2026 (multimodal track)

---

## Option C: RL for Diffusion / Flow Matching — DDPO

### Background

Denoising Diffusion Policy Optimization (DDPO) reframes the iterative denoising process of a diffusion model as a Markov decision process. Each denoising step is an action, the noise schedule defines the state transitions, and the reward is computed only at the final denoised image. This MDP formulation lets you apply policy gradient methods — specifically PPO — to fine-tune the diffusion model toward any differentiable or even black-box reward signal. The result is a general framework for aligning image generation models with human preferences, aesthetic scores, or task-specific objectives, without needing to backpropagate through the entire sampling chain. DDPO connects the RL curriculum directly to the generative modeling literature and opens alignment questions that are structurally similar to RLHF but in the continuous image domain.

At the same time, Diffusion-DPO provides a direct preference analogue: given pairs of (preferred image, dispreferred image) for the same prompt, DPO can fine-tune the diffusion model's score function. Comparing these two approaches reveals a tradeoff between online RL's flexibility and offline preference learning's stability.

### Task Specification

Fine-tune Stable Diffusion 1.5 (or SDXL-Turbo for faster iteration) using DDPO. Choose one of the following reward signals (or propose your own):

- **Aesthetic score:** Use the LAION aesthetic predictor (a pretrained classifier available on HuggingFace) as the reward. Higher aesthetic score = better reward.
- **Compressibility:** Use JPEG compression ratio as a reward proxy for visual simplicity. Images that compress more efficiently are simpler and more coherent.

Implement the full DDPO training loop: rollout the denoiser for a batch of prompts, collect (state, action, reward) trajectories, compute PPO advantages, and update the UNet weights with clipped policy gradient. Then implement **Diffusion-DPO** on the same reward signal: construct preference pairs by sampling two images per prompt and assigning preference based on reward, then apply the DPO objective to the diffusion model's score function. Compare both methods.

**Constraints:**
- Use gradient checkpointing and mixed precision; training must be feasible on a single A100 or equivalent (40GB VRAM).
- Report both a reward metric and a diversity metric; DDPO is known to collapse to reward-maximizing but low-diversity outputs.
- Evaluate on a fixed held-out prompt set (50–100 prompts); do not tune on the evaluation prompts.

### Related Work

| Paper | Relevance |
|---|---|
| DDPO: Training Diffusion Models with Reinforcement Learning (Black et al., 2023) | Introduced the MDP formulation for diffusion sampling; the core algorithm |
| DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models (Fan et al., 2023) | Online RL fine-tuning with KL penalty; close competitor to DDPO |
| Diffusion Model Alignment Using Direct Preference Optimization (Wallace et al., 2023) | Direct preference analogue for diffusion; the comparison target for this project |
| ReFL: Reward Feedback Learning (Xu et al., 2023) | Reward feedback via gradient through the reward model; alternative to policy gradient |

### Baseline to Beat

Base SD 1.5 without fine-tuning, evaluated by: (1) mean reward on held-out prompts, (2) human preference win rate vs DDPO-tuned model in a side-by-side comparison (use a small set, 20–30 comparisons is sufficient for a student project).

### Success Criteria

| Criterion | Threshold |
|---|---|
| DDPO reward improvement | DDPO fine-tuned model achieves ≥ 60% win rate vs base model on target reward metric |
| Diversity preservation | FID or mean pairwise LPIPS does not degrade by more than 15% relative to base model |
| DDPO vs Diffusion-DPO | Quantitative comparison on reward metric and diversity; at least one dimension where each method wins |
| Training stability | No mode collapse within the training budget; reward curve is monotonically improving or plateauing, not crashing |

### Deliverables

| File | Description |
|---|---|
| `ddpo_trainer.py` | DDPO training loop: rollout, trajectory collection, PPO update |
| `diffusion_dpo.py` | Diffusion-DPO implementation: preference pair construction, DPO loss for score function |
| `reward_fn.py` | Reward function(s): aesthetic score and/or compressibility |
| `evaluate_diversity.py` | Diversity evaluation: FID and/or pairwise LPIPS on held-out prompts |
| `report.md` | 6-page maximum (abstract, intro, method, experiments, conclusion) |

### Suggested Timeline

| Week | Goal |
|---|---|
| 1 | Reward function setup; verify the reward signal is sensible on a random sample of SD 1.5 outputs |
| 2 | DDPO training loop: implement rollout collection and PPO update; verify the loss decreases on a small batch |
| 3 | Full DDPO training run; measure reward and diversity metrics; tune KL penalty |
| 4 | Diffusion-DPO implementation; preference pair construction from reward comparisons; DPO loss for diffusion |
| 5 | Side-by-side comparison of DDPO vs Diffusion-DPO; failure mode analysis (mode collapse examples) |
| 6 | Write report; code cleanup |

### Workshop / Venue Suggestions

- ICLR 2026 (main track or workshop)
- NeurIPS 2026 Workshop on Generative Models
- ICML 2026

---

## General Guidance

### Compute Planning

Before committing to an option, estimate your compute budget. All three options can be completed on a single consumer GPU with careful choices:

| Option | Minimum GPU | Recommended | Estimated GPU-hours |
|---|---|---|---|
| A (GRPO) | RTX 3090 (24 GB) with LoRA | A100 40 GB | 20–40 hours |
| B (VLM DPO) | RTX 3090 with LoRA + 4-bit | A100 40 GB | 30–60 hours |
| C (DDPO) | A100 40 GB | A100 80 GB | 40–80 hours |

If you do not have access to a suitable GPU, use Google Colab Pro, Lambda Labs spot instances, or the HuggingFace Spaces GPU grant program.

### Report Standards

All reports should follow the conventions of an ML conference short paper:
- Abstract (150 words maximum)
- Clear problem statement and motivation
- Method section with equations (not just prose)
- Experiments with ablations (not just a single run)
- Honest discussion of limitations
- References (use BibTeX or equivalent)

Do not pad to the page limit. A tight 4-page report is better than a padded 6-page report.

### Code Standards

- All scripts must run from a clean environment with `pip install -r requirements.txt` (or `uv sync`).
- No Jupyter notebooks in the main submission (analysis notebooks are fine for exploratory work; final training and evaluation must be runnable scripts).
- Committed model checkpoints will not be graded. Commit only code and logs.

## Submission Checklist

Before submitting, verify:

- [ ] All scripts run end-to-end from scratch.
- [ ] TensorBoard logs are reproducible from the submitted training scripts.
- [ ] `report.md` is within the page limit and includes an abstract.
- [ ] Model weights are in `.gitignore`; only code and logs are committed.
- [ ] At least one baseline comparison is quantitative (numbers, not just qualitative description).
- [ ] Related work section cites the papers listed for your chosen option (and ideally a few more).
