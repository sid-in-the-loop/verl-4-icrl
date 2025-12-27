# Hierarchical In-Context Reinforcement Learning (ICML)

## Mathematical Formulation (GRPO Backbone)

---

## 1. Setup and Notation

Let

* ( \mathcal{D} = {(x, y^*)} ) be a reasoning dataset (e.g., **MATH500** for evaluation).
* ( x ): input problem
* ( y^* ): ground-truth final answer (verifiable)

We define a **two-level system**:

### Higher-Level (HL): Context Generator

[
\pi_\theta(c \mid x)
]
where

* ( c = (x', y') ) is a synthetic demonstration (can generalize to multiple demos).

### Lower-Level (LL): Task Solver

[
p_\phi(y \mid x, c)
]
which generates a reasoning trace + final answer ( y ).

### Context-Free Prior (Reference Policy)

[
q(y \mid x)
]
Typically:

* a **frozen copy** of the initial LL model ( p_{\phi_0}(y \mid x) ).

---

## 2. End-to-End Objective

The global objective is:
[
\max_{\theta,\phi}
\mathbb{E}*{x \sim \mathcal{D}}
\mathbb{E}*{c \sim \pi_\theta(\cdot \mid x)}
\mathbb{E}*{y \sim p*\phi(\cdot \mid x,c)}
\big[ R_{\text{task}}(x,y) \big]
]

Direct optimization is unstable → we decompose into **LL training**, **HL training**, and **alternation**.

---

## 3. Lower-Level Training (GRPO Backbone)

### 3.1 Standard GRPO (No Context)

For each input ( x ):

1. Sample a group of rollouts:
   [
   y_i \sim p_\phi(\cdot \mid x), \quad i=1,\dots,N
   ]

2. Compute task rewards:
   [
   r_i = R_{\text{task}}(x, y_i)
   ]

3. Compute group baseline:
   [
   b(x) = \frac{1}{N}\sum_{j=1}^N r_j
   ]

4. Advantages:
   [
   A_i = r_i - b(x)
   ]

5. GRPO objective:
   [
   \max_\phi
   \mathbb{E}\Big[\sum_i A_i \log p_\phi(y_i \mid x)\Big]

* \beta_{\text{LL}} , \mathbb{E}\big[ D_{\text{KL}}(p_\phi ,|, p_{\text{ref}}) \big]
  ]

---

### 3.2 Context-Conditioned GRPO (Experiments 3 & 5)

Now each rollout is conditioned on context:

[
c_i \sim \pi_\theta(\cdot \mid x), \quad
y_i \sim p_\phi(\cdot \mid x, c_i)
]

Rewards:
[
r_i = R_{\text{task}}(x, y_i)
]

Same baseline and advantage computation as above, but optimize:
[
\max_\phi
\mathbb{E}\Big[\sum_i A_i \log p_\phi(y_i \mid x,c_i)\Big]

* \beta_{\text{LL}} , \mathbb{E}\big[ D_{\text{KL}}(p_\phi(\cdot \mid x,c),|,p_{\text{ref}}(\cdot \mid x,c)) \big]
  ]

This is your **GRPO backbone with training-time context**.

---

## 4. Forcing Context Usage (KL-Maximization)

### 4.1 Motivation

Without constraints, the LL can **ignore context**.
We explicitly **reward divergence from a context-free prior**.

---

### 4.2 Context Utilization Term

Define:
[
U_\phi(x,c) = D_{\text{KL}}\big(p_\phi(\cdot \mid x,c),|,q(\cdot \mid x)\big)
]

Expanded:
[
U_\phi(x,c)
= \mathbb{E}*{y \sim p*\phi}
\big[\log p_\phi(y \mid x,c) - \log q(y \mid x)\big]
]

To prevent pathological divergence:
[
\tilde{U}*\phi(x,c) = \min(U*\phi(x,c), \tau)
]

---

### 4.3 Final LL Objective

[
\max_\phi
\mathbb{E}*{x,c,y}
\big[
R*{\text{task}}(x,y)

* \lambda \cdot \tilde{U}_\phi(x,c)
  \big]
  ]

---

### 4.4 GRPO Implementation Trick

Per-sample reward shaping:
[
r_i^{\text{total}} =
R_{\text{task}}(x,y_i)

* \lambda \cdot \text{clip}\big(
  \log p_\phi(y_i \mid x,c_i) - \log q(y_i \mid x),
  \tau'
  \big)
  ]

Then run **standard GRPO** on ( r_i^{\text{total}} ).

---

## 5. Higher-Level Training (Context Generator)

### 5.1 HL Reward

HL samples ( c \sim \pi_\theta(c \mid x) ).
LL produces ( y \sim p_\phi(y \mid x,c) ).

Define:
[
R_{\text{HL}}(x,c;\phi)
= R_{\text{task}}(x,y)

* \alpha \cdot H\big(p_\phi(\cdot \mid x,c)\big)
  ]

Equivalent confidence form:
[
R_{\text{HL}}(x,c;\phi)
= R_{\text{task}}(x,y)

* \alpha \cdot \mathbb{E}*{y \sim p*\phi}\big[\log p_\phi(y \mid x,c)\big]
  ]

---

### 5.2 KL-to-SFT Constraint

Let ( \pi_{\text{SFT}} ) be HL after supervised pretraining on DSFT.

Final HL objective:
[
\max_\theta
\mathbb{E}*{x,c \sim \pi*\theta}
\big[ R_{\text{HL}}(x,c;\phi) \big]

* \beta_{\text{HL}} , D_{\text{KL}}\big(
  \pi_\theta(\cdot \mid x),|,\pi_{\text{SFT}}(\cdot \mid x)
  \big)
  ]

---

## 6. Alternating Optimization

Initialize:

* ( \theta_0 \leftarrow ) SFT on DSFT
* ( \phi_0 \leftarrow ) GRPO baseline
* ( q(y|x) \leftarrow p_{\phi_0}(y|x) ) (frozen)

For ( t = 1,\dots,T ):

### Step 1: LL Update (Fix ( \theta ))

[
\phi_t \leftarrow
\arg\max_\phi
\mathbb{E}\big[
R_{\text{task}} + \lambda \tilde{U}_\phi
\big]
\quad \text{via GRPO}
]

### Step 2: HL Update (Fix ( \phi ))

[
\theta_t \leftarrow
\arg\max_\theta
\mathbb{E}\big[
R_{\text{HL}}(x,c;\phi_t)
\big]

* \beta_{\text{HL}} D_{\text{KL}}(\pi_\theta | \pi_{\text{SFT}})
  ]

---

## 7. Mapping to Your 5 Experiments

| Exp | HL         | LL Training | Context      |
| --: | ---------- | ----------- | ------------ |
|   1 | —          | GRPO        | None         |
|   2 | Pretrained | Frozen      | Test-time    |
|   3 | Pretrained | GRPO        | Train + Test |
|   4 | SFT        | Frozen      | Test-time    |
|   5 | SFT        | GRPO        | Train + Test |

---

## 8. Key Ablations (Mathematical Toggles)

### LL-Side

* **No context usage**: ( \lambda = 0 )
* **Different priors**: change ( q(y|x) )
* **No KL-to-ref**: ( \beta_{\text{LL}} = 0 )

### HL-Side

* **Task-only reward**: ( \alpha = 0 )
* **No SFT constraint**: ( \beta_{\text{HL}} = 0 )

### Training

* **Joint vs alternating**
* **Frozen LL (ICL-like)**

---

## 9. One-Line Interpretation

> The LL is trained with GRPO to solve tasks **and** maximize information gained from context, while the HL is trained to generate contexts that improve correctness and confidence, coordinated through alternating optimization.