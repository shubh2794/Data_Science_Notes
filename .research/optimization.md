# Research Brief — Optimization (mathematical optimization for ML)

## 1. Topic & scope

**In scope:** mathematical optimization as used to *train* ML models — i.e., choosing parameters θ that minimize a scalar loss/objective L(θ). Covers the optimization problem and its geometry (convex vs non-convex, local vs global minima, saddle points), first-order methods (gradient descent and its batch/stochastic/mini-batch variants), momentum and Nesterov acceleration, adaptive methods (AdaGrad, RMSProp, Adam, AdamW), learning-rate schedules, second-order ideas (Newton, Hessian, L-BFGS), convergence and conditioning (condition number κ), constrained optimization (Lagrange multipliers, KKT), and practical failure modes.

**Out of scope:** the mechanics of *computing* the gradient itself (backpropagation, autodiff — those belong to Calculus and Neural Network Training), full convex-optimization theory (duality gaps, interior-point/simplex solvers), combinatorial/integer optimization, and Bayesian/black-box hyperparameter optimization. Reinforcement-learning policy optimization is a separate node.

This brief **fills an existing stub** (the "Optimization" node already exists in data.js with no page).

## 2. Core concept

Optimization is the problem of finding parameter values that make a model best fit its objective. Training is posed as an **unconstrained minimization**:

  min_θ L(θ)

where θ ∈ ℝⁿ is the parameter vector (weights) and L is a differentiable scalar loss measuring prediction error over the data. Because L is almost never minimizable in closed form for modern models, it is minimized **iteratively**: start from an initial θ₀ and repeatedly take small steps that reduce L, using the gradient ∇L(θ) (the direction of steepest *increase*) to decide where to move. The overwhelming majority of deep-learning training is **first-order** (gradient-only) and **stochastic** (gradient estimated from a mini-batch).

## 3. Intuition

Picture L(θ) as a landscape — a surface whose height is the loss and whose horizontal coordinates are the parameters. Training is a ball rolling downhill. The gradient ∇L points uphill (steepest ascent), so we step in the **opposite** direction, −∇L, to descend. The step size is the **learning rate** η: too small and the ball crawls; too large and it overshoots, oscillates across valleys, or diverges.

Key terrain features:
- A **convex** bowl has a single global minimum; any downhill path reaches it. Linear/logistic regression and SVMs (with convex losses) live here.
- A **non-convex** landscape (every neural network) has many valleys (local minima), ridges, plateaus, and **saddle points** — places flat in some directions but sloped in others. In high dimensions, saddle points and flat plateaus, not bad local minima, are the dominant obstacle: a critical point is a local min only if it curves up in *all* n directions, which is exponentially unlikely, so most critical points are saddles.

Why it matters: optimization is the engine of all learning. The choice of optimizer, learning rate, and schedule often decides whether a model trains at all, how fast, and how well it generalizes — frequently mattering more than small architecture changes.

## 4. Key math (Unicode only)

**The problem and gradient.**
  min_θ L(θ),  θ ∈ ℝⁿ
  ∇L(θ) = [∂L/∂θ₁, ∂L/∂θ₂, …, ∂L/∂θₙ]ᵀ
  First-order optimality (interior): ∇L(θ*) = 0

**Convexity.** L is convex if for all x, y and t ∈ [0,1]:
  L(tx + (1−t)y) ≤ t·L(x) + (1−t)·L(y)
  Equivalent (twice-diff): Hessian H = ∇²L ⪰ 0 (positive semidefinite) everywhere.
  Consequence: for a convex L, every local minimum is a global minimum.

**Gradient descent (batch).**
  θ ← θ − η·∇L(θ)
  uses the full dataset per step — stable but expensive.

**Stochastic gradient descent (SGD), one example i.**
  θ ← θ − η·∇L(θ; xⁱ, yⁱ)
  noisy, cheap, high variance; the noise can help escape saddles/sharp minima.

**Mini-batch gradient descent (the default), batch of size b.**
  θ ← θ − η·(1/b)·Σ_(i∈B) ∇L(θ; xⁱ, yⁱ)
  trades variance against compute; b typically 32–512 (up to thousands at scale).

**Momentum** (γ ≈ 0.9): accumulate a velocity, damping oscillation across ravines.
  vₜ = γ·vₜ₋₁ + η·∇L(θ)
  θ ← θ − vₜ

**Nesterov accelerated gradient (NAG):** look ahead, then correct.
  vₜ = γ·vₜ₋₁ + η·∇L(θ − γ·vₜ₋₁)
  θ ← θ − vₜ

**AdaGrad** (per-parameter scaling; Gₜ = Σ of past squared gradients):
  θₜ₊₁,ᵢ = θₜ,ᵢ − (η / √(Gₜ,ᵢᵢ + ε))·gₜ,ᵢ
  great for sparse features; learning rate decays monotonically to ~0 (its flaw).

**RMSProp** (exponential moving average of squared gradients fixes AdaGrad's decay):
  E[g²]ₜ = 0.9·E[g²]ₜ₋₁ + 0.1·gₜ²
  θ ← θ − (η / √(E[g²]ₜ + ε))·gₜ      (η ≈ 0.001)

**Adam** (momentum + RMSProp + bias correction). With gₜ = ∇L(θₜ₋₁):
  mₜ = β₁·mₜ₋₁ + (1−β₁)·gₜ      (1st moment / mean)
  vₜ = β₂·vₜ₋₁ + (1−β₂)·gₜ²      (2nd moment / uncentered variance)
  m̂ₜ = mₜ / (1 − β₁ᵗ)          (bias correction)
  v̂ₜ = vₜ / (1 − β₂ᵗ)
  θₜ = θₜ₋₁ − η·m̂ₜ / (√v̂ₜ + ε)
  Defaults: β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸, η = 0.001.
  Bias correction matters because m and v start at 0, so early estimates are biased toward 0; dividing by (1 − βᵗ) (which → 1 as t grows) removes this. [verified against Ruder overview + Adam paper]

**AdamW** (decoupled weight decay; λ = decay coefficient): apply decay directly to weights instead of folding it into the gradient.
  θₜ = θₜ₋₁ − η·(m̂ₜ / (√v̂ₜ + ε) + λ·θₜ₋₁)
  For plain SGD, L2 penalty and weight decay are equivalent; for Adam they are **not**, because the adaptive denominator √v̂ rescales the L2 term unevenly. Decoupling restores true weight decay and improves generalization.

**Learning-rate schedules** (vary η over training step t; T = total steps):
  Step decay:  ηₜ = η₀·γ^⌊t/s⌋
  Cosine decay:  ηₜ = η_min + ½(η₀ − η_min)(1 + cos(π·t/T))
  Linear warmup (first T_w steps):  ηₜ = η₀·(t / T_w)
  Warmup-then-cosine is the modern default for large models.

**Newton's method (second-order).** Use curvature (Hessian H = ∇²L):
  θ ← θ − H⁻¹·∇L(θ)
  Quadratic local convergence, scale-invariant — but H is n×n, costing O(n²) memory and O(n³) per solve, infeasible when n is millions/billions.
  **L-BFGS** approximates H⁻¹ from recent gradient/step pairs with limited memory (O(n)); used for smaller/convex problems, rarely for deep nets with stochastic gradients.

**Conditioning.** For a quadratic with Hessian H, define the condition number:
  κ = λ_max(H) / λ_min(H)
  Gradient descent's error contracts roughly like ((κ − 1)/(κ + 1))² per step, so large κ (an elongated, ill-conditioned valley) means painfully slow zig-zag convergence. Momentum and adaptive methods reduce this sensitivity; Newton's method makes convergence independent of κ.

**Constrained optimization (Lagrange / KKT).**
  min_x f(x)  s.t.  gᵢ(x) ≤ 0,  hⱼ(x) = 0
  Lagrangian:  𝓛(x, λ, ν) = f(x) + Σᵢ λᵢ·gᵢ(x) + Σⱼ νⱼ·hⱼ(x)
  KKT conditions (necessary; sufficient when the problem is convex):
   1. Stationarity:  ∇f(x*) + Σᵢ λᵢ∇gᵢ(x*) + Σⱼ νⱼ∇hⱼ(x*) = 0
   2. Primal feasibility:  gᵢ(x*) ≤ 0,  hⱼ(x*) = 0
   3. Dual feasibility:  λᵢ ≥ 0
   4. Complementary slackness:  λᵢ·gᵢ(x*) = 0
  These underpin the SVM dual and any constrained training objective.

## 5. Practical takeaways & trade-offs

- **Mini-batch SGD is the workhorse.** Batch GD is too slow per step; pure SGD is too noisy. b = 32–512 is typical; larger batches need larger η and warmup.
- **Adam(W) is the default for deep nets / Transformers.** Robust to learning-rate choice, fast early progress, handles sparse and noisy gradients. AdamW (decoupled decay) is now standard for LLMs and ViTs because it generalizes better than Adam-with-L2.
- **Well-tuned SGD + momentum often generalizes slightly better than Adam** on some vision/CNN tasks and is still common there — at the cost of more learning-rate tuning. Worth noting sources disagree on which "wins"; it is task- and tuning-dependent. [reconciled across sources]
- **Learning rate is the single most important hyperparameter.** Too high → divergence/oscillation; too low → glacial training. Use warmup (stabilizes early large-gradient steps, especially with adaptive methods and large batches) then decay (cosine or step). Typical Adam η ≈ 1e-3 to 3e-4; LLM pretraining often peaks lower (e.g. ~1e-4 to 6e-4) with warmup over the first few thousand steps. [approximate, ranges vary]
- **Saddle points and plateaus, not local minima, are the real obstacle** in high dimensions. Stochastic noise and adaptive/momentum methods help escape them.
- **Ill-conditioning (large κ)** causes slow zig-zagging; normalization (BatchNorm/LayerNorm), good initialization, and adaptive methods mitigate it.
- **Vanishing gradients:** gradients shrink toward 0 in deep stacks (saturating activations, long chains) → early layers barely learn. Mitigations: ReLU-family activations, residual connections, normalization, careful init.
- **Exploding gradients:** gradients blow up (common in RNNs) → NaNs / divergence. Mitigation: **gradient clipping** (rescale g so ‖g‖ ≤ threshold).
- **Second-order methods rarely used at scale:** exact Hessian inversion is O(n³); for billion-parameter models this is hopeless. L-BFGS suits smaller smooth/convex problems but struggles with stochastic mini-batch noise.
- **Failure modes checklist:** diverging loss → lower η or add warmup/clipping; loss plateaus immediately → η too low or vanishing gradients; loss spikes to NaN → exploding gradients/clip; slow oscillation → ill-conditioning, try momentum/Adam or normalization.

## 6. Suggested interactive viz

Primary (recommended): an **interactive loss-surface descent explorer** on a 2-parameter contour plot.
- Render a non-trivial 2D loss surface as D3 contour lines — ideally an elongated/ill-conditioned bowl (large κ) so zig-zagging is visible, with an optional toggle to add a saddle.
- Animate the optimizer trajectory step by step from a draggable start point.
- Controls: optimizer selector (SGD / SGD+Momentum / RMSProp / Adam), a **learning-rate slider** η, and step/play. Show the trajectory diverging when η is too high and crawling when too low.
- This single viz makes learning rate, momentum's anti-oscillation effect, adaptive rescaling, and conditioning all tangible.

Secondary (optional second viz, matching the multi-viz style of the linear-algebra flagship page):
- A **learning-rate schedule plotter**: draw step, cosine, and warmup+cosine curves of ηₜ vs step with adjustable warmup length and total steps.
- Or a **1D convex-vs-non-convex** toy where a ball rolls down a user-switchable curve, illustrating local minima and saddles.

Match the flagship page conventions: `<svg>` at 640×360 in a `.viz` block, illustrative (not production) D3, neutral caption.

## 7. Candidate placement

- **Action: FILL THE STUB** (do not mint a new node).
- **Node id:** `Optimization` (already in data.js).
- **group:** `math`  |  **level:** 2  |  **domain:** Math & Statistics.
- **Proposed file / link:** `math/optimization.html` (add `link:"math/optimization.html"` to the existing node).
- **Existing desc (keep or lightly refine):** "Finding parameters that minimize loss: gradient descent, convexity, momentum, Adam."
- **Sibling depth to match:** `math/linear-algebra.html` (flagship: numbered `<h3>` sections, Unicode `.formula` blocks, several D3 SVG vizzes, neutral "compiled from various sources" footer).

**Cross-link candidates** (all node ids verified to exist in data.js):
- `Calculus` (math) — supplies ∇L; edge `["Calculus","Optimization"]` already exists.
- `Linear Algebra` (math) — Hessian, condition number κ, eigenvalues, quadratic forms.
- `Neural Network Training` (dl, → deep-learning/neural-network-training.html) — backprop produces the gradients these methods consume; edge `["Neural Network Training","Optimization"]` already exists.
- `Regularization` (ml) — AdamW decoupled weight decay vs L2; edge `["Regularization","Optimization"]` already exists.
- `Linear & Logistic Regression` (ml) — canonical convex training objective; edge `["Optimization","Linear & Logistic Regression"]` already exists.
- (Already-present `["Optimization","Feedforward / MLP"]` edge can stay.)

Note for ingestor: the four most relevant cross edges **already exist** in data.js — no new edges are strictly required, though `["Linear Algebra","Optimization"]` would be a sensible *addition* (currently absent) given the Hessian/conditioning content.

## 8. Sources consulted (audit trail only — do NOT surface on the page)

- https://www.ruder.io/optimizing-gradient-descent/ and https://arxiv.org/abs/1609.04747 — Ruder, "An overview of gradient descent optimization algorithms." Highly cited, authoritative survey; primary source for GD variants, momentum/NAG, AdaGrad/RMSProp/Adam update rules and defaults, saddle-point discussion.
- https://arxiv.org/abs/1412.6980 — Kingma & Ba, "Adam: A Method for Stochastic Optimization." The original Adam paper (peer-reviewed, ICLR 2015); source of the m̂/v̂ bias-corrected update and defaults β₁=0.9, β₂=0.999, ε=1e-8. Abstract page only fetched cleanly; update rule cross-verified against Ruder (consistent).
- https://arxiv.org/abs/1711.05101 / https://openreview.net/pdf?id=Bkg6RiCqY7 — Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW). Peer-reviewed (ICLR 2019); source for the L2-vs-weight-decay distinction in adaptive optimizers.
- https://davidrosenberg.github.io/mlcourse/Archive/2017/Notes/convex-optimization.pdf — abridgement of Boyd & Vandenberghe, *Convex Optimization*. Reputable NYU course notes distilling the standard textbook; for convexity, optimality, KKT. (PDF did not parse to text; concepts triangulated with the items below + standard knowledge.)
- https://ecal.studentorg.berkeley.edu/files/ce191/LEC13%20-%20KKT.pdf — UC Berkeley CE191 lecture notes on Lagrange multipliers & KKT; source for Lagrangian form and the four KKT conditions.

**Triangulation:** 5 distinct authoritative sources (2 primary papers, 1 widely-cited survey w/ paper+blog, 2 reputable university course notes derived from Boyd & Vandenberghe).

**[unverified] gaps:**
- Specific LLM-pretraining learning-rate numbers (peak η, warmup length) are given as approximate ranges; exact values are model-specific and were not pinned to a single primary source.
- The Boyd/Vandenberghe abridgement PDF and the Adam arXiv full text could not be extracted as clean text; their math was reconstructed from the search summaries + the Ruder source + standard knowledge and should be sanity-checked by the domain specialist (especially the GD convergence-rate expression ((κ−1)/(κ+1))², which is the standard exact-line-search quadratic result and is stated here as intuition).
