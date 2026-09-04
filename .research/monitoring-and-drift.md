# Research Brief — Monitoring & Drift

## 1. Topic & scope

**Topic:** Production ML/LLM model monitoring, and the detection of distribution shift ("drift") and model performance degradation over time.

**In scope:**
- Taxonomy of drift: data/covariate drift P(X), label/prior drift P(Y), concept drift P(Y|X); real vs virtual drift.
- Why deployed models degrade (ML and non-ML causes).
- Detection math (Unicode): PSI, KL, JS, KS test, Wasserstein, chi-square.
- Performance monitoring vs input monitoring; the ground-truth-lag problem and proxy metrics.
- Streaming/windowing detectors: ADWIN, DDM, Page-Hinkley.
- Alerting thresholds and the false-positive / alert-fatigue trade-off.
- LLM-specific drift: embedding drift, prompt/response drift, semantic monitoring.
- Remediation: retraining triggers, rollback, shadow/canary deployment.

**Out of scope:** General observability/logging infra plumbing; training-time data validation (touch only where it overlaps); detailed A/B test statistics (own node); model serving mechanics (own node).

---

## 2. Core concept

A model is trained on a frozen snapshot of the world; production is a moving target. **Drift** is any change in the live data or its relationship to the target that makes the deployed model less valid than it was at training time. **Monitoring** is the standing instrumentation that watches inputs, predictions, and (when available) outcomes to detect that degradation — ideally *before* it shows up as a business loss.

The clean framing: a model learns a joint distribution P(X, Y) = P(Y|X)·P(X) = P(X|Y)·P(Y). Drift is named by **which factor moves**:

- **Covariate / data drift** — P(X) changes, P(Y|X) stable. Inputs look different; the input→output rule still holds.
- **Label / prior drift** — P(Y) changes (equivalently P(X|Y) stable). The outcome mix shifts (e.g. fraud rate jumps).
- **Concept drift** — P(Y|X) changes. *Same input, different correct answer* — the rule itself moved.

A crucial distinction: **real drift** changes P(Y|X) and therefore degrades accuracy; **virtual drift** changes P(X) (or P(Y)) without changing the decision boundary, so the model may still be correct even though its inputs look unfamiliar. Input monitoring alone cannot tell these apart — which is the central reason monitoring is hard.

---

## 3. Intuition

- **The map vs the territory.** Training data is a finite map; the real world is effectively infinite and keeps changing. The map goes stale even if nothing in your code breaks.
- **Two clocks problem.** Input drift is observable *now* (you have the inputs). Accuracy drift is observable *later* (you need the true labels). Most of the difficulty is bridging that gap with proxies.
- **"Same question, new answer."** Concept drift is the spooky one: nothing about the inputs looks off, but the right answer has changed (COVID-era house prices, "good investment" after a rate shock). You usually catch it only via outcomes or business metrics, not input statistics.
- **Sudden vs gradual vs recurring.** Drift can be abrupt (a pipeline change, a new product launch), gradual (slow taste/behavior shift), incremental, or recurring/seasonal (weekday vs weekend). The temporal shape dictates the right detection window.
- **Most "drift" alerts are not model problems.** A large share of production failures are mundane: a broken upstream pipeline, a wrong model version deployed, a schema change, a units mismatch. Good monitoring catches these too — and they are often the majority of real incidents.

---

## 4. Key math (Unicode only)

**Joint decomposition (the taxonomy):**
P(X, Y) = P(Y|X) · P(X) = P(X|Y) · P(Y)
- covariate drift: P(X) ≠ P′(X), P(Y|X) = P′(Y|X)
- label/prior drift: P(Y) ≠ P′(Y), P(X|Y) = P′(X|Y)
- concept drift: P(Y|X) ≠ P′(Y|X), P(X) = P′(X)
- dataset shift: both P(X) and P(Y|X) move.

**Population Stability Index** (bin the feature; pᵢ = actual share, qᵢ = expected/reference share in bin i):
PSI = ∑ᵢ (pᵢ − qᵢ) · ln(pᵢ / qᵢ)
PSI is effectively a *symmetrized* KL — the "round-trip" relative entropy.
Rule-of-thumb thresholds: PSI < 0.1 → no meaningful shift; 0.1 ≤ PSI < 0.25 → moderate, watch; PSI ≥ 0.25 → major shift, investigate/act. (Some practitioners use 0.2 as the action line.)

**Kullback–Leibler divergence** (asymmetric, unbounded; 0 = identical):
D_KL(P ‖ Q) = ∑ᵢ pᵢ · ln(pᵢ / qᵢ)
Note D_KL(P‖Q) ≠ D_KL(Q‖P), and it blows up to ∞ if any qᵢ = 0 where pᵢ > 0.

**Jensen–Shannon divergence** (symmetric, smoothed, always finite), with M = ½(P + Q):
JSD(P ‖ Q) = ½ · D_KL(P ‖ M) + ½ · D_KL(Q ‖ M)
JSD ∈ [0, ln 2] (nats) or [0, 1] (bits / when using log₂); JS *distance* = √JSD. Common drift line ≈ 0.1.

**Kolmogorov–Smirnov two-sample statistic** (nonparametric, continuous 1-D; F = empirical CDFs):
D = supₓ |F₁(x) − F₂(x)|
Returns a p-value; reject "same distribution" when D large. One-dimensional only.

**Wasserstein-1 (earth-mover's) distance** (1-D, via inverse CDFs / quantile functions):
W₁(P, Q) = ∫₀¹ |F⁻¹_P(u) − F⁻¹_Q(u)| du   (equivalently ∫ |F_P(x) − F_Q(x)| dx)
Reports the *magnitude* of the shift in the feature's own units (often normalized by σ; drift line ≈ 0.1 σ).

**Chi-square test of homogeneity** (categorical; Oᵢ = observed count, Eᵢ = expected count per category):
χ² = ∑ᵢ (Oᵢ − Eᵢ)² / Eᵢ

**Maximum Mean Discrepancy** (kernel two-sample test, handles multivariate):
MMD²(P, Q) = E[k(x, x′)] + E[k(y, y′)] − 2·E[k(x, y)]   (x,x′∼P; y,y′∼Q)

**Page–Hinkley test** (CUSUM-style sequential detector on a stream; xₜ = observed value, x̄ₜ = running mean, δ = tolerance):
mₜ = ∑_{i≤t} (xᵢ − x̄ᵢ − δ),  Mₜ = min_{i≤t} mᵢ
alarm when (mₜ − Mₜ) > λ   (λ = sensitivity threshold)

**DDM (Drift Detection Method)** — monitors online error rate pₜ with std sₜ = √(pₜ(1−pₜ)/t); tracks the minimum point (p_min + s_min):
warning when pₜ + sₜ ≥ p_min + 2·s_min
drift   when pₜ + sₜ ≥ p_min + 3·s_min

**ADWIN (Adaptive Windowing)** — keep a window W; split into W₀ (older) and W₁ (recent). Declare drift and drop W₀ when:
|mean(W₀) − mean(W₁)| > ε_cut,  ε_cut derived from a Hoeffding bound at confidence δ.

---

## 5. Practical takeaways & trade-offs

**What to monitor (three layers, increasing latency, increasing trustworthiness):**
1. **Inputs / features** — cheapest, available instantly. Schema/range/null checks plus per-feature distribution distance (PSI/KS/Wasserstein). Catches pipeline breakage and covariate drift. *Problem:* feature distributions wobble constantly and most wobble is benign → alert fatigue.
2. **Predictions / output distribution** — low-dimensional, easy to two-sample-test; a shift here is a strong proxy for input shift (model weights are fixed). Often the best single signal you can get without labels.
3. **Accuracy / business outcome** — the only thing that *directly* answers "is the model still good?" but gated by ground-truth availability.

**The ground-truth-lag problem (the defining constraint):**
- **Feedback-loop length** sets how fast you can measure quality. Short loops (recommenders: click within seconds) vs long loops (fraud, credit: labels arrive weeks–months later via disputes; a dispute window of ~1–3 months is typical). If a fraud model breaks and labels lag months, damage compounds before detection.
- **Natural labels** (Maps ETA self-corrects) are a luxury; most tasks need **proxy metrics** — click-through rate, completion rate, dwell time, downstream conversion — which approximate quality in real time.
- **Beware degenerate feedback loops:** the model's own outputs shape the next inputs (popularity bias). Detect via output diversity / long-tail coverage; correct with randomized exploration and positional features.

**Statistical-test trade-offs (empirically established):**
- **KS test is sample-size-cursed.** With large n, statistical power explodes and the p-value flags trivial changes — e.g. it fires on a 0.5% shift once n > ~100,000. Good for small data (< ~1,000) or when tiny deviations genuinely matter; bad as a default in big production streams.
- **PSI / KL / JS are sample-size-stable** (distance metrics, not p-values) → predictable in production. PSI = "only big changes"; JS slightly more sensitive than KL.
- **Wasserstein** is the pragmatic middle ground: more sensitive than PSI, less trigger-happy than KS, and reports drift magnitude in interpretable units. Preferred for high-dimensional / embedding spaces where KS degrades.
- **Chi-square** for categoricals; **MMD** for multivariate research settings (rarely used in industry production as of the sources).
- **Always reduce dimensionality** before multivariate two-sample testing.

**Windowing & temporal detectors:**
- Reference window (baseline) vs sliding/current window; the window must be ≥ the seasonal cycle (a sub-week window can't see a weekly cycle). Sliding windows track recency; cumulative windows retain history.
- **Streaming detectors** watch the error stream directly: **DDM/EDDM** (error-rate based, good for abrupt drift), **Page-Hinkley** (CUSUM sequential, mean shift), **ADWIN** (adaptive variable window with formal guarantees), **KSWIN** (KS over a window).

**Alerting & false positives:**
- Tighter thresholds → catch drift early but drown in false alarms → engineers mute the alerts → real drift missed. The whole game is tuning the threshold (and requiring sustained breach, not a single spike) to balance miss-rate vs alert-fatigue. A monitoring-vendor observation: a large fraction (cited as ~80%) of detected "drifts" trace to human/pipeline errors, not genuine model decay — so alerts should route to root-cause triage, not auto-retrain.

**Why models degrade in production (a Google study of 96 ML-pipeline failures over 15 years found 60 were *non-ML* causes):**
- Non-ML: dependency breaks, wrong-version deploys, hardware/server failures, train–serve skew, schema/units changes.
- ML-specific: covariate/label/concept drift, edge cases, degenerate feedback loops, bad data collection.

**LLM-specific drift:**
- **Embedding / data drift:** distribution shift of input-prompt embeddings (new topics, new product launch shifts query mix). Measure distance between embedding distributions/centroids across time windows — **Wasserstein preferred over KS** because embeddings are high-dimensional.
- **Concept drift in LLMs:** prompts look statistically similar (low data drift) but the *desired* answer changed (e.g. "good investment" after market shift) → only catchable via downstream business metrics / user feedback, not input stats.
- **Prompt/response drift:** changes in prompt templates, retrieved context, or silent provider model-version updates can shift outputs even with stable user inputs.
- **Two-layer LLM monitoring pattern:** Layer 1 = cheap statistical embedding-distance test fires an alert; Layer 2 = **LLM-as-judge** samples the drifted prompts and *classifies the cause* ("new topic", "intent shift", "rising complexity", "language-style change") to make the alert actionable. Output-quality drift tracked via LLM-as-judge scores, toxicity/refusal rates, latency, and user thumbs.

**Remediation playbook (alert → action ladder):**
- **Triage first** — confirm it's real drift, not a broken pipeline/wrong version (most common). 
- **Rollback** to last-good model/prompt if quality dropped post-deploy.
- **Retraining triggers** — scheduled (cron), threshold-based (PSI/accuracy breach), or continuous; refresh data window and revalidate.
- **Shadow deployment** — run candidate model on live traffic without serving its outputs, compare to incumbent before promotion; **canary** = serve to a small % first.
- **For LLMs** — update prompts/system instructions, refresh RAG corpus/index, re-pin or re-evaluate the provider model version, adjust guardrails.

---

## 6. Suggested interactive viz

**Primary: "Drift detector under a sliding window."** Two distributions on a canvas — a fixed reference (training) histogram and a live "current window" histogram the user can drag/shift (slide the mean, fatten the variance, or rebalance class proportions). As the user drags:
- Live-update a readout of **PSI, KS-D (with p-value), JS distance, and Wasserstein** side by side.
- Color the panel green/amber/red against the standard thresholds (PSI 0.1 / 0.25 lines).
- A toggle for "sample size" (1k vs 100k) that visibly makes **KS flip to red on a tiny shift** while PSI stays calm — viscerally teaching the sample-size trap.

**Secondary idea (if a second viz fits):** a streaming error-rate timeline where an animated point crosses a Page-Hinkley / DDM warning then drift threshold, illustrating sequential detection and the warning→drift two-stage alarm.

These directly teach: (a) the four metrics agree/disagree differently, (b) thresholds, (c) the KS false-positive failure mode, (d) sequential detection.

---

## 7. Candidate placement

- **Action:** **Fill the stub.** Node already exists in `data.js` (line 132):
  `{id:"Monitoring & Drift", group:"mlops", level:2, desc:"Tracking live performance and detecting when data shifts away from training."}` — **no `link` field yet** (page-less stub).
- **Domain / group:** `mlops` ("MLOps & Deployment"). Confirmed correct; do not change.
- **Proposed page path:** `mlops/monitoring-and-drift.html` (matches sibling convention: `mlops/llmops.html`, `mlops/model-acceleration.html`, `mlops/speculative-decoding.html`).
- **Required `data.js` edit (for the ingestor, not me):** add `link:"mlops/monitoring-and-drift.html"` to the existing node object. Do **not** mint a new node.

**Cross-link candidates** (verified present in `data.js`; several edges already exist):
- **Model Deployment** (`mlops`) — drift is what you watch *after* deploy; shadow/canary/rollback live here. *(edge exists? add if missing)*
- **LLMOps** (`mlops`, has page) — embedding/semantic drift, LLM-as-judge monitoring. Edge `["LLMOps","Monitoring & Drift"]` already exists (line 341).
- **Experiment Tracking** (`mlops`) — retraining runs/metrics logged here; closes the retrain loop.
- **A/B Testing** (`mlops`) — edge `["A/B Testing","Monitoring & Drift"]` already exists (line 302); canary/shadow comparison.
- **Model Evaluation** — edge `["Model Evaluation","Monitoring & Drift"]` already exists (line 261); offline metrics = the baseline monitoring degrades from.
- **Data Quality & Governance** (`data-eng`) — edge `["Data Quality & Governance","Monitoring & Drift"]` already exists (line 339); schema/range checks overlap with input monitoring.
- **Dashboards** (`viz`) — edge `["Dashboards","Monitoring & Drift"]` already exists (line 263); how drift signals are surfaced.

Existing edges mean cross-linking is largely already wired in the graph; the page just needs in-body `<a>` links matching these.

---

## 8. Sources consulted (audit trail only — do NOT surface on the published page)

- https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html — Chip Huyen, *Designing ML Systems* companion essay. High credibility; primary practitioner reference for taxonomy, feedback-loop/ground-truth lag, monitoring layers, Google 60/96 failure stat.
- https://www.evidentlyai.com/blog/data-drift-detection-large-datasets — Evidently AI empirical comparison of KS/PSI/KL/JS/Wasserstein on large data. High credibility (tooling vendor, but data-backed); source for sample-size sensitivity and method-selection guidance.
- https://arize.com/blog-course/population-stability-index-psi/ and https://www.fiddler.ai/blog/measuring-data-drift-population-stability-index — PSI formula, thresholds, PSI↔KL relationship. Reputable monitoring vendors; corroborate each other.
- https://docs.aws.amazon.com/prescriptive-guidance/latest/gen-ai-lifecycle-operational-excellence/prod-monitoring-drift.html — AWS Prescriptive Guidance on LLM/GenAI drift. High credibility; source for two-layer embedding+LLM-judge pattern, Wasserstein-over-KS for embeddings, concept-vs-data drift in LLMs.
- https://arxiv.org/pdf/2203.11070 — "From Concept Drift to Model Degradation: ... Performance-Aware Drift Detectors" (arXiv survey). Peer-style; categorization of sequential (CUSUM/Page-Hinkley) vs window-based (ADWIN, DDM) detectors.
- https://link.springer.com/article/10.1007/s10462-025-11428-y — *Artificial Intelligence Review* survey on concept-drift detection. Peer-reviewed; real vs virtual drift, drift taxonomy.
- https://cran.r-project.org/web/packages/datadriftR/index.html — datadriftR docs enumerating DDM/EDDM/HDDM/KSWIN/ADWIN/PH detectors. Reputable; corroborates detector list.

**[unverified] gaps:**
- The "~80% of drifts are human/pipeline error" figure is a cited practitioner observation, not a controlled study — flag as anecdotal on the page if used.
- DDM 2σ/3σ warning/drift constants and exact ADWIN ε_cut Hoeffding form are stated from standard descriptions; the precise formulations vary by implementation — present as the canonical version, not the only one.
- JS divergence range bound (ln 2 nats vs 1 bit) depends on log base — stated both ways to avoid a downstream error.
