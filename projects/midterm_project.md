# Midterm Project Brief

**Curriculum context:** Modules 01–04 complete (MDP foundations, value-based methods, policy gradients & PPO, model-based RL).

**Due:** End of Module 04.

Choose **one** of the three options below. Each option is scoped for approximately four weeks of part-time work (8–12 hours/week). Read all three before deciding — they differ substantially in what skills they exercise.

---

## Option A: Custom Environment + PPO with Curriculum Learning

### Background

Designing RL environments is as hard as solving them. A poorly specified observation space, a badly shaped reward, or a task that is simultaneously too easy and too hard can doom training before the algorithm ever runs. Curriculum learning — gradually increasing task difficulty based on the agent's recent performance — often makes the difference between convergence and failure on tasks with sparse or delayed rewards. This option puts you on both sides of the problem: first as an environment designer, then as a practitioner trying to make PPO learn efficiently on something you built.

### Task Specification

Design and implement a custom Gymnasium-compatible environment for a task of your choice (do not use any standard Gymnasium environment or its direct variants). The environment must have a non-trivial state space and a reward structure where curriculum learning plausibly helps. Train a PPO agent on the environment. Then implement curriculum learning: a mechanism that monitors the agent's success rate over a rolling window and adjusts some parameter of the environment (obstacle density, episode length, goal distance, etc.) to keep difficulty near a target success rate (e.g., 40–60%). Document all design decisions explicitly.

**Constraints:**
- Environment must pass `gymnasium.utils.passive_env_checker` with no errors or warnings.
- PPO implementation must be your own (no Stable-Baselines3 black box); you may use the implementation from Module 03 assignments as a starting point.
- Curriculum parameter adjustment must be automatic (not hand-scheduled) and driven by a rolling success-rate estimate.
- Log all runs to TensorBoard (episode return, success rate, current difficulty level).

### Success Criteria

| Criterion | Threshold |
|---|---|
| Policy quality | Agent achieves a success rate that would be non-trivial for a random agent (quantify what random baseline achieves) |
| Curriculum benefit | TensorBoard curves show curriculum learning converges faster or to a higher asymptote than constant-difficulty training |
| Environment correctness | `gymnasium.utils.passive_env_checker` passes with zero errors |
| Ablation | At least one ablation run (e.g., no curriculum, fixed difficulty) compared in TensorBoard |

### Deliverables

| File | Description |
|---|---|
| `env.py` | Custom Gymnasium environment, fully documented |
| `train.py` | PPO training loop with curriculum logic |
| `report.md` | Project report, 4 pages maximum |

**Report sections:**
1. **Environment design** — observation space, action space, reward function, rationale for curriculum parameter choice.
2. **PPO configuration** — hyperparameters, architecture, and why you chose them.
3. **Curriculum design** — rolling window size, target success rate, adjustment schedule.
4. **Results and ablations** — learning curves, what worked, what did not.

### Suggested Timeline

| Week | Goal |
|---|---|
| 1 | Environment design and implementation; pass `passive_env_checker`; verify a random agent can interact without crashing |
| 2 | PPO baseline (constant difficulty); establish that learning is at least possible |
| 3 | Curriculum learning implementation; ablation runs (with/without curriculum, different target success rates) |
| 4 | Write report; clean up code; record a short demonstration rollout video (optional but encouraged) |

---

## Option B: Offline RL — Behavioral Cloning vs CQL

### Background

Offline RL learns entirely from a fixed dataset of transitions collected by some behavior policy, without any additional environment interaction. This matters enormously in domains where online data collection is expensive, dangerous, or ethically unacceptable — think robotic manipulation, clinical treatment policy, or autonomous driving. The core challenge is distributional shift: a Q-function trained on logged data will be queried on out-of-distribution (OOD) actions during greedy policy extraction, and standard Bellman backups can catastrophically overestimate Q-values on those actions. Conservative Q-Learning (CQL) addresses this by adding an explicit penalty that suppresses Q-values on OOD actions, regularizing the learned policy toward the behavior distribution.

### Task Specification

Using the D4RL `hopper-medium-v2` dataset (preferred), or a CartPole offline dataset you generate yourself if MuJoCo is unavailable, implement the following from scratch:

1. **Behavioral Cloning (BC) baseline** — supervised learning on (state, action) pairs from the dataset, maximum likelihood for discrete actions or mean-squared error for continuous.
2. **Simplified CQL agent** — actor-critic with a conservative Q-penalty added to the standard Bellman loss. Use the closed-form CQL penalty: `alpha * (logsumexp(Q(s, a')) - Q(s, a_data))` summed over sampled random actions `a'`.

Evaluate both agents by rolling out the learned policy in the live environment for 10 episodes and reporting normalized score. Log Q-value statistics (mean, std, max) during training to TensorBoard.

**Constraints:**
- No offline RL libraries (no d3rlpy, no CORL); implement the algorithms yourself.
- You may use PyTorch for neural networks and D4RL/minari for dataset loading.
- CQL alpha must be treated as a hyperparameter; report at least two values.

### Success Criteria

| Criterion | Threshold |
|---|---|
| BC performance | Achieves ≥ 50% of expert normalized score on the chosen environment |
| CQL vs BC | CQL outperforms BC by ≥ 10 percentage points on normalized score |
| Q-value regularization | TensorBoard shows CQL suppresses Q-values on OOD actions relative to a no-penalty run |
| Stability | Neither agent crashes or produces NaN losses during training |

### Deliverables

| File | Description |
|---|---|
| `bc_agent.py` | Behavioral cloning implementation |
| `cql_agent.py` | CQL implementation |
| `evaluate.py` | Rollout evaluation script, reports normalized score |
| `report.md` | Project report, 4 pages maximum |

**Report sections:**
1. **Dataset analysis** — state/action statistics, what the behavior policy appears to be doing.
2. **Algorithm implementations** — BC loss, CQL loss derivation, network architectures.
3. **Results** — normalized scores, Q-value curves, sensitivity to CQL alpha.
4. **Discussion** — when would you prefer BC over CQL? What failure modes did you observe?

### Suggested Timeline

| Week | Goal |
|---|---|
| 1 | Dataset loading and inspection; implement and train BC baseline; reach ≥ 50% normalized score |
| 2 | CQL implementation; verify Q-value penalty is active by inspecting TensorBoard statistics |
| 3 | Hyperparameter sweep (alpha values, network sizes, learning rates); compare CQL vs BC |
| 4 | Write report; document failure modes; clean up code |

---

## Option C: Reward Hacking Study (Empirical Paper Format)

### Background

Reward hacking — the phenomenon where an agent finds unintended ways to maximize a proxy reward that diverge from the designer's true intent — is not an exotic edge case. It is a regular occurrence in RL deployments and a central challenge in AI alignment. From a robot that learns to exploit physics glitches to a language model that produces confident-sounding but incorrect answers to maximize a reward model score, the failure pattern is the same: the proxy reward is easier to hack than it is to satisfy in the intended way. Studying reward hacking empirically in small, interpretable environments builds the intuition needed to understand the RLHF reward hacking problem, where the consequences are harder to observe and harder to correct.

### Task Specification

Design three distinct reward misspecification scenarios in MiniGrid (`minigrid` package) or in a custom Gymnasium environment. For each scenario:

1. Specify the **intended behavior** in plain language and implement a reward function you expect will produce it.
2. Train a DQN or PPO agent to convergence.
3. Characterize the **actual behavior** the agent learns — if it hacks the reward, document exactly how.
4. Implement and evaluate at least one **mitigation** from the following list:
   - Potential-based reward shaping (add a shaping term that does not change the optimal policy theoretically)
   - KL penalty against a reference policy
   - Reward ensemble (train multiple reward heads, take the minimum)

The three scenarios must be meaningfully distinct — not three variants of the same hack. At least one scenario must involve a hack that is subtle enough that a naive reader would not immediately predict it from the reward specification alone.

**Constraints:**
- Each scenario must include a video or GIF of the hacked behavior (use `gymnasium.wrappers.RecordVideo`).
- Log all runs to TensorBoard; comparison runs (with/without mitigation) must be in the same TensorBoard experiment.
- The report must cite at least two papers from the reward hacking or RLHF alignment literature.

### Success Criteria

| Criterion | Threshold |
|---|---|
| Scenario diversity | Three scenarios that demonstrate qualitatively different reward hacking failure modes |
| Hack demonstration | Each scenario has a recorded rollout confirming the unintended behavior |
| Mitigation | At least one scenario shows a mitigation that measurably improves alignment with intended behavior (TensorBoard comparison) |
| Literature connection | Report cites ≥ 2 relevant papers and connects empirical observations to broader alignment concerns |

### Deliverables

```
scenarios/
  scenario_a/  (env definition, train script, recordings)
  scenario_b/
  scenario_c/
analysis.ipynb  (cross-scenario comparison, Q-value visualizations)
report.md       (5 pages maximum, empirical paper format)
```

**Report format (empirical paper):**
1. **Introduction** — motivation, connection to RLHF reward hacking.
2. **Methods** — environment descriptions, reward specifications, training details, mitigations.
3. **Results** — one subsection per scenario; hacking behavior + mitigation outcome.
4. **Discussion** — what generalizes across scenarios? What made mitigations succeed or fail? What would you do differently?

### Suggested Timeline

| Week | Goal |
|---|---|
| 1 | Scenario design (write down the three scenarios + expected hacking mode before implementing anything); get at least one scenario hacking reproducibly |
| 2 | All three scenarios producing hacking behavior; begin mitigation implementations |
| 3 | Mitigation experiments complete; analysis notebook drafted |
| 4 | Write report in empirical paper format; record clean rollout videos |

---

## Submission Checklist

Before submitting any option, verify:

- [ ] All scripts run from scratch with `python <script>.py` (no hidden state dependencies).
- [ ] TensorBoard logs are committed or clearly documented.
- [ ] `report.md` is within the page limit.
- [ ] No trained model weights in the repository (add `*.pt`, `*.pkl` to `.gitignore`).
- [ ] Code is commented at the level where a classmate could run and modify it.
