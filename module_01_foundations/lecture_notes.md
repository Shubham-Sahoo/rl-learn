# Module 01: MDP Foundations, Bellman Equations, and Dynamic Programming

> **How to read this chapter.** Every section follows the same structure: plain English first, then
> formal notation, then a derivation with inline annotations that explain *why* each step is written
> the way it is. If you find yourself staring at an equation, check the annotation below it.

---

## 1. Intuition

### 1.1 The Credit Assignment Problem

Imagine teaching a dog a new trick. You give it a treat when it finally sits — but the dog performed
*many* actions between your command and the correct sit: it sniffed the floor, wagged its tail,
looked away, then sat. Which action actually caused the reward? This is the **credit assignment
problem**: given a stream of actions and a delayed reward, how do you decide which past actions
deserve credit?

In supervised learning you sidestep this entirely. Each input is paired with a label and you get
a signal *immediately*. In RL the signal may come seconds, minutes, or thousands of steps after the
decision that caused it. A language model being trained with RLHF does not know which token in a
500-token response caused the human to prefer it.

### 1.2 RL vs Supervised Learning

| Dimension | Supervised Learning | Reinforcement Learning |
|---|---|---|
| Feedback | Immediate, per-example | Delayed, per-episode |
| Labels | Provided by oracle | Discovered by agent |
| Distribution | Fixed dataset | Changes as agent improves |
| Goal | Fit a function | Maximize cumulative reward |

The key shift: in RL, *your actions affect what data you see next*. A bad action can trap you in a
region of the state space where good data is unavailable. This feedback loop is what makes RL both
powerful and hard.

### 1.3 The Markov Property: Why History Does Not Matter

The Markov property says: **everything you need to predict the future is already in the current
state**. You do not need to remember how you arrived there.

Concretely: if you know where you are in a maze, you do not need to know the path you took to get
there — the optimal next action depends only on your current position. This is an assumption, not
a theorem. Real environments (e.g., a poker game where an opponent's previous bets reveal their
strategy) can violate it. For now, we assume it holds.

The payoff is enormous: it reduces the problem from "how do I optimize over all possible histories"
to "how do I optimize a mapping from current states to actions." That is what a policy is.

---

## 2. Formal Setup: The MDP Tuple

A **Markov Decision Process** is defined by a five-tuple:

$$\mathcal{M} = (S,\; A,\; P,\; R,\; \gamma)$$

Each component:

| Symbol | Name | Meaning |
|---|---|---|
| $S$ | State space | All possible situations the agent can be in |
| $A$ | Action space | All moves available to the agent |
| $P(s' \mid s,a)$ | Transition function | Probability of landing in $s'$ after taking $a$ in $s$ |
| $R(s,a,s')$ | Reward function | Scalar signal received when transitioning $s \xrightarrow{a} s'$ |
| $\gamma \in [0,1)$ | Discount factor | How much the agent cares about future vs immediate rewards |

### 2.1 State Space $S$

Can be discrete (positions in a gridworld, inventory counts) or continuous (joint angles, pixel
arrays). In this module we work with discrete, finite state spaces to keep the math tractable.

### 2.2 Action Space $A$

Similarly discrete or continuous. For the GridWorld in Assignment 1: four cardinal directions.

### 2.3 Transition Function $P$

Satisfies $\sum_{s'} P(s' \mid s, a) = 1$ for all $s, a$. The stochastic version (slip probability)
in Assignment 1 encodes that the agent sometimes moves in a perpendicular direction despite its
intention.

### 2.4 Reward Function $R$

Can depend on $(s, a)$ only, or on $(s, a, s')$. We use the full form. Convention: reward is
received *after* the transition completes.

### 2.5 Discount Factor $\gamma$

Intuitively: a reward of 1 received $t$ steps from now is worth $\gamma^t$ today. As $\gamma \to 0$
the agent becomes **myopic** — it only cares about immediate reward. As $\gamma \to 1$ the agent
values future rewards equally to present ones. We require $\gamma < 1$ to ensure the infinite sum
$\sum_{t=0}^{\infty} \gamma^t r_t$ converges (bounded by $r_{\max}/(1-\gamma)$).

### 2.6 A Trajectory

A sequence of interactions:

$$s_0 \xrightarrow{a_0} s_1 \xrightarrow{a_1} s_2 \xrightarrow{a_2} \cdots$$

The **return** $G_t$ from time $t$ is the discounted sum of future rewards:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

Note the recursive relationship: $G_t = r_t + \gamma G_{t+1}$. This is the seed of every Bellman
equation.

---

## 3. Value Functions

### 3.1 State-Value Function $V^\pi$

Under policy $\pi$, the **state-value function** is the expected return from state $s$:

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s\right]$$

*(Why? Because the agent does not control randomness in transitions or stochastic policies —
we average over all of it. The $\mathbb{E}_\pi$ subscript reminds us that actions are drawn from
$\pi$.)*

$V^\pi$ answers: *"if I follow policy $\pi$ forever starting from $s$, how much total reward do I
expect?"*

### 3.2 Action-Value Function $Q^\pi$

Sometimes we want to evaluate a specific action before committing to $\pi$ forever after:

$$Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0=s, a_0=a\right]$$

*(Why the same expectation? After the first action $a_0 = a$, we follow $\pi$ for all subsequent
steps. The expectation averages over the stochastic transitions from step 1 onward.)*

$Q^\pi$ answers: *"if I take action $a$ in state $s$ and then follow $\pi$, how much total reward
do I expect?"*

### 3.3 Relationship Between $V^\pi$ and $Q^\pi$

$$V^\pi(s) = \sum_a \pi(a|s)\, Q^\pi(s,a)$$

*(Why? $V^\pi(s)$ averages over all actions the policy might take. For each action $a$, the policy
chooses it with probability $\pi(a|s)$, and $Q^\pi(s,a)$ is the value of taking that action. So
$V^\pi$ is simply the policy-weighted average of $Q^\pi$ values.)*

This identity is used constantly: it connects the two value functions and is the key to both
policy evaluation and policy improvement.

---

## 4. Bellman Expectation Equations

The Bellman equations express value functions *recursively*: the value of a state is defined in
terms of the values of neighboring states. This is what makes iterative algorithms possible.

### 4.1 Bellman Equation for $V^\pi$

**Derivation:**

Start from the definition: $V^\pi(s) = \mathbb{E}_\pi[G_0 \mid s_0 = s]$.

Use the recursive return identity $G_0 = r_0 + \gamma G_1$:

$$V^\pi(s) = \mathbb{E}_\pi[r_0 + \gamma G_1 \mid s_0 = s]$$

*(Why can we expand like this? Because $G_0 = r_t + \gamma G_{t+1}$ is just arithmetic, not an
approximation. It holds exactly for any trajectory.)*

Now unroll the expectation over the first action and first transition:

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \bigl[R(s,a,s') + \gamma \mathbb{E}_\pi[G_1 \mid s_1 = s']\bigr]$$

*(Why the double sum? First we average over which action $\pi$ selects ($\sum_a \pi(a|s)$), then
over where the environment sends us ($\sum_{s'} P(s'|s,a)$). These are independent sources of
randomness.)*

Recognizing that $\mathbb{E}_\pi[G_1 \mid s_1 = s'] = V^\pi(s')$ by the Markov property and
stationarity:

$$\boxed{V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma V^\pi(s')\bigr]}$$

*(Why does $\mathbb{E}_\pi[G_1 \mid s_1 = s'] = V^\pi(s')$? Because the process is Markovian:
given $s_1 = s'$, the expected future return starting at time 1 is exactly $V^\pi(s')$ — it does
not depend on how we got to $s'$.)*

### 4.2 Bellman Equation for $Q^\pi$

By a parallel derivation (fix $a_0 = a$, then average over subsequent actions under $\pi$):

$$\boxed{Q^\pi(s,a) = \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s')\, Q^\pi(s',a')\bigr]}$$

*(Why the inner sum over $a'$? After transitioning to $s'$, the policy selects $a'$ with probability
$\pi(a'|s')$. The value from $s'$ onward is $\sum_{a'} \pi(a'|s') Q^\pi(s', a') = V^\pi(s')$.
So $Q^\pi(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$, which expands further into
the form above.)*

### 4.3 Key Insight: Linear System

For a fixed policy $\pi$ with $|S|$ states, the Bellman expectation equations are a *linear
system* in the unknowns $\{V^\pi(s)\}_{s \in S}$. In principle you could solve it directly via
matrix inversion: $V^\pi = (I - \gamma P^\pi)^{-1} R^\pi$ where $P^\pi$ is the $|S| \times |S|$
transition matrix under $\pi$. But matrix inversion is $O(|S|^3)$ — infeasible for large state
spaces. **Iterative policy evaluation** does it cheaper by repeatedly applying the Bellman update
until convergence.

---

## 5. Bellman Optimality Equations

So far we've analyzed a *fixed* policy $\pi$. Now we ask: what's the best possible behavior?

### 5.1 Optimal Value Functions

Define the **optimal state-value function**:

$$V^*(s) = \max_\pi V^\pi(s)$$

And the **optimal action-value function**:

$$Q^*(s,a) = \max_\pi Q^\pi(s,a)$$

A key theorem (proof in Appendix): there always exists an optimal *deterministic* policy $\pi^*$
such that $V^{\pi^*}(s) = V^*(s)$ for all $s$.

### 5.2 Bellman Optimality Equation for $V^*$

The optimal agent always picks the best action:

$$\boxed{V^*(s) = \max_a \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma V^*(s')\bigr]}$$

*(Why $\max_a$ instead of $\sum_a \pi(a|s)$? Because the optimal policy is greedy: it takes
whichever action maximizes the expected return. There is no probability distribution over actions —
the agent deterministically picks the best one.)*

### 5.3 Bellman Optimality Equation for $Q^*$

$$\boxed{Q^*(s,a) = \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')\bigr]}$$

*(Why $\max_{a'}$ on the right? After reaching $s'$, the optimal agent picks the best action from
there. So the future value from $s'$ is $\max_{a'} Q^*(s', a') = V^*(s')$.)*

### 5.4 The Connection: $V^*$ from $Q^*$

$$V^*(s) = \max_a Q^*(s,a)$$

And the optimal policy can be read off directly:

$$\pi^*(s) = \arg\max_a Q^*(s,a)$$

*(Why is this remarkable? If we know $Q^*$, we can act optimally without ever doing planning or
knowing the model $P$. This is the foundation of model-free Q-learning in Module 02.)*

---

## 6. Policy Iteration

Policy iteration alternates between two steps until convergence.

### 6.1 Algorithm Pseudocode

```
Initialize policy π arbitrarily (e.g., random)

repeat:
    ── Policy Evaluation ──────────────────────────────────────
    Initialize V(s) = 0 for all s
    repeat:
        Δ ← 0
        for each s in S:
            v ← V(s)
            V(s) ← Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]
            Δ ← max(Δ, |v - V(s)|)
    until Δ < θ  (convergence threshold, e.g. θ = 1e-6)

    ── Policy Improvement ─────────────────────────────────────
    policy_stable ← True
    for each s in S:
        old_action ← π(s)
        π(s) ← argmax_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]
        if old_action ≠ π(s):
            policy_stable ← False

until policy_stable
return V, π
```

### 6.2 Convergence Argument

**Why does this converge?** Policy improvement theorem: greedy improvement produces a policy $\pi'$
such that $V^{\pi'}(s) \ge V^\pi(s)$ for all $s$. Since there are finitely many deterministic
policies ($|A|^{|S|}$), and each step strictly improves (or keeps equal) the policy value, the
algorithm must terminate. At termination, the policy satisfies the Bellman optimality equations,
so it is optimal.

**Computational cost:** Each policy evaluation takes $O(|S|^2 |A|)$ per sweep, and we need enough
sweeps for convergence. Policy improvement takes $O(|S|^2 |A|)$ total. In practice, policy
iteration converges in very few outer iterations (often < 20) even for large grids.

---

## 7. Value Iteration

Value iteration collapses policy evaluation and improvement into a single update.

### 7.1 Algorithm Pseudocode

```
Initialize V(s) = 0 for all s

repeat:
    Δ ← 0
    for each s in S:
        v ← V(s)
        V(s) ← max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]
        Δ ← max(Δ, |v - V(s)|)
until Δ < θ

Extract policy:
    π(s) ← argmax_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]
return V, π
```

### 7.2 Bellman Optimality Operator as a Contraction

Define the **Bellman optimality operator** $\mathcal{T}$:

$$(\mathcal{T} V)(s) = \max_a \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma V(s')\bigr]$$

Value iteration is the iteration $V_{k+1} = \mathcal{T} V_k$.

**Claim:** $\mathcal{T}$ is a $\gamma$-contraction in the $\ell^\infty$ (sup-norm):

$$\|\mathcal{T} V - \mathcal{T} W\|_\infty \le \gamma \|V - W\|_\infty$$

*(Why is contraction useful? By Banach's fixed-point theorem, any contraction mapping on a complete
metric space has a unique fixed point, and the iteration converges to it geometrically. The fixed
point here is exactly $V^*$, the optimal value function.)*

Value iteration converges at a geometric rate: $\|V_k - V^*\|_\infty \le \gamma^k \|V_0 - V^*\|_\infty$.

### 7.3 Policy Iteration vs Value Iteration

| Dimension | Policy Iteration | Value Iteration |
|---|---|---|
| Outer loop | Until policy stable | Until $\Delta < \theta$ |
| Inner loop | Full policy evaluation | Single Bellman max backup |
| Convergence | Few outer iters, expensive inner | Many iters, cheap per iter |
| Preferred when | $|S|$ small, evaluation cheap | $|S|$ large, $\gamma$ close to 1 |

---

## 8. Failure Modes

### 8.1 The Effect of $\gamma$ on Agent Horizon

When $\gamma$ is small (e.g., $0.1$), the agent is highly myopic. A reward 10 steps away is worth
only $0.1^{10} \approx 10^{-10}$ in the present. In a maze with a distant goal, the agent behaves
as if the goal does not exist and may wander randomly. This can cause policy evaluation to converge
to near-zero values everywhere, making all states look equally bad.

Conversely, when $\gamma \to 1$, the discounted sum is no longer guaranteed to converge. In
environments with no terminal state, infinite loops collect infinite reward (positive or negative).
Value iteration may not converge.

**Practical rule:** Use $\gamma \in [0.95, 0.999]$ for episodic tasks. Use $\gamma = 1$ only in
episodic tasks (terminal states ensure finite returns).

### 8.2 When Dynamic Programming is Intractable

DP requires iterating over every state $s \in S$ and every action $a \in A$ at each step. This is
only feasible when:
1. $|S|$ and $|A|$ are small enough to enumerate,
2. The transition function $P(s'|s,a)$ is known exactly.

For real-world problems neither holds. A 2D Atari game has roughly $10^{20,000}$ possible pixel
configurations — enumeration is impossible. And in most RL problems (robotics, language models),
$P$ is unknown; the agent only observes samples.

**The solution:** Function approximation (neural networks) + sample-based methods (Q-learning,
policy gradients). These are covered in Modules 02–03.

### 8.3 Terminal State Convention

Terminal states (goal, hole, end-of-episode) have $V^*(s) = 0$ by convention. This is because the
episode ends there; no future reward is collected. A common bug in policy evaluation is applying
the Bellman update to terminal states, causing their values to drift from zero.

---

## 9. Research Bridge

### 9.1 Process Reward Models (PRMs) and $Q^\pi$

In reasoning LLMs (e.g., OpenAI's math solvers), a **process reward model** assigns a reward to
each reasoning *step*, not just the final answer. This is exactly $Q^\pi(s, a)$ over a chain-of-
thought trajectory: $s$ is the current reasoning state (prefix), $a$ is the next token/step, and
$Q^\pi(s,a)$ estimates the probability of reaching a correct final answer from this point. PRM
training is therefore an instance of learning $Q^*$ over a discrete action space (reasoning steps).

### 9.2 RLHF PPO and the Critic Baseline

In PPO-based RLHF (e.g., InstructGPT), the algorithm maintains a **critic** $V^\pi(s)$ as a
baseline to reduce variance in the policy gradient. Here $s$ is the token prefix and $V^\pi(s)$
estimates the expected reward (human preference score) for the full response starting from that
prefix. This is the direct analogue of the state-value function you compute in this module. The
critic is trained by minimizing $|V^\pi(s) - G_t|^2$ — exactly iterative policy evaluation.

### 9.3 Why RLHF Uses $\gamma = 1$

In RLHF, the reward is typically given once at the end of the full response. There is no temporal
discounting of individual tokens. Setting $\gamma = 1$ means every token's contribution to the
final reward is equally weighted. This is valid because the task is inherently episodic (a response
is finite), and we want the agent to optimize the entire response quality, not just the first few
tokens.

---

## 10. Appendix: Proof that the Bellman Optimality Operator is a $\gamma$-Contraction

**Theorem.** Let $\mathcal{T}$ be the Bellman optimality operator:
$$(\mathcal{T} V)(s) = \max_a \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma V(s')\bigr]$$

For any $V, W: S \to \mathbb{R}$:
$$\|\mathcal{T} V - \mathcal{T} W\|_\infty \le \gamma \|V - W\|_\infty$$

**Proof.**

Fix any state $s$. Let $a^* = \arg\max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V(s')]$ and
$b^* = \arg\max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma W(s')]$.

Then:

$$(\mathcal{T} V)(s) - (\mathcal{T} W)(s)$$
$$= \max_a \sum_{s'} P(s'|s,a)[R + \gamma V(s')] - \max_a \sum_{s'} P(s'|s,a)[R + \gamma W(s')]$$

Using the inequality $\max f - \max g \le \max(f - g)$:

$$\le \max_a \sum_{s'} P(s'|s,a) \gamma [V(s') - W(s')]$$

$$\le \gamma \max_a \sum_{s'} P(s'|s,a) |V(s') - W(s')|$$

*(Why can we move $\gamma$ outside? It is a positive constant. Why can we swap max and absolute
value? We bounded $V(s') - W(s') \le |V(s') - W(s')| \le \|V - W\|_\infty$.)*

$$\le \gamma \|V - W\|_\infty \max_a \sum_{s'} P(s'|s,a) = \gamma \|V - W\|_\infty$$

*(Why does the last sum equal 1? $P$ is a probability distribution over $s'$, so it sums to 1 for
any fixed $(s,a)$.)*

By symmetry, $(\mathcal{T} W)(s) - (\mathcal{T} V)(s) \le \gamma \|V - W\|_\infty$ as well.

Therefore:
$$|(\mathcal{T} V)(s) - (\mathcal{T} W)(s)| \le \gamma \|V - W\|_\infty$$

Taking the supremum over $s$:
$$\|\mathcal{T} V - \mathcal{T} W\|_\infty \le \gamma \|V - W\|_\infty \qquad \square$$

**Corollary.** Since $\gamma < 1$, $\mathcal{T}$ is a strict contraction on the complete metric
space $(\mathbb{R}^{|S|}, \|\cdot\|_\infty)$. By Banach's fixed-point theorem, there exists a
unique $V^*$ such that $\mathcal{T} V^* = V^*$, and the iteration $V_{k+1} = \mathcal{T} V_k$
converges to $V^*$ for any initialization $V_0$.

---

*Next: Assignment 1 — implement a stochastic GridWorld and solve it with policy evaluation.*
