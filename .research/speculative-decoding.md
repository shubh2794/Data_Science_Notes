# Research brief — Speculative Decoding

> Scratch material for the ingestion pipeline. Not a published page. Source-neutral by design.
> Math is Unicode only (house rule). Notation here: **q = draft model distribution**, **p = target model distribution**.
> (Beware: the two seminal papers swap p/q — one uses q for draft, the other uses p for draft. This brief fixes q=draft, p=target throughout to avoid downstream confusion.)

## 1. Topic & scope

**Speculative decoding** (also called *speculative sampling* / *assisted generation*) is an inference-time acceleration technique for autoregressive language models. A small, cheap **draft model** proposes several future tokens; the large **target model** then verifies them all in a single parallel forward pass, accepting a run of them at once. Done with the right acceptance rule, it is **lossless** — the generated text is sampled from *exactly* the target model's distribution, so quality is provably unchanged.

In scope:
- The draft-then-verify loop and why it speeds up memory-bandwidth-bound decoding.
- The modified rejection-sampling acceptance criterion and the proof that the output distribution is preserved.
- The expected-tokens-per-step and walltime-speedup math (acceptance rate α, draft length γ, cost ratio c).
- Variants: independent draft model, self-speculative / layer-skipping, Medusa (extra heads), EAGLE (feature-level drafting), Lookahead (model-free / n-gram), prompt/retrieval lookup; tree attention verification.
- Trade-offs, failure modes, batch-size interactions, real numbers, tooling.

Out of scope (belongs to neighboring nodes): general quantization/pruning/distillation, KV-cache mechanics, MoE routing, parallel diffusion-style text decoding. Cross-link rather than duplicate.

## 2. Core concept

Autoregressive decoding emits one token per forward pass: to produce N tokens you run the full model N times, each pass dominated by streaming the model's weights from memory (it is **memory-bandwidth-bound**, not compute-bound — the GPU's arithmetic units sit mostly idle). Speculative decoding breaks this one-token-per-pass coupling:

1. **Draft.** A cheap model q autoregressively guesses the next γ tokens (e.g. γ = 4–7).
2. **Verify in parallel.** The target model p scores all γ+1 candidate positions in **one** forward pass (the prefix plus each drafted token). Because a single pass over γ+1 positions costs almost the same wall-clock time as a single pass over one position (still bandwidth-bound), this verification is nearly free relative to running p γ times.
3. **Accept / correct.** A modified rejection-sampling rule walks the drafted tokens left to right, accepting each with a probability tied to how well q matched p. On the first rejection it resamples one corrected token from an adjusted distribution and discards the rest.

Each cycle therefore emits between 1 and γ+1 tokens while invoking the expensive model only once. The output is distributed identically to plain sampling from p — the speedup is "free" in quality terms (for the exact variant).

## 3. Intuition

- **Easy tokens are common.** Much of generated text is predictable from context (closing brackets, common words, boilerplate continuations). A small model gets these right most of the time; the big model only needs to *check* them, which it can do in bulk.
- **Verification is cheaper than generation.** Checking "are these γ tokens what I would have produced?" runs in parallel; generating them runs serially. Speculative decoding converts serial big-model work into parallel big-model work plus cheap serial small-model work.
- **Mental model — autocomplete + proofreader.** The draft model is a fast typist filling in the obvious next words; the target model is a meticulous proofreader who reads the whole proposed phrase at once, keeps the correct prefix, and fixes the first mistake. Crucially, the proofreader's corrections make the final text indistinguishable from text the proofreader would have written alone.
- **Why it matters.** Inference, run billions of times, dominates the lifetime cost and latency of deployed LLMs. Speculative decoding cuts per-token latency ~2–3× with *zero* quality loss and no retraining of the target — a rare "free lunch" in serving.

## 4. Key math (Unicode only)

Setup: at a position, draft distribution q(x) and target distribution p(x) over the vocabulary. The draft samples token x ~ q.

**Acceptance rule (modified rejection sampling).**
- If p(x) ≥ q(x): **accept** x (the target likes it at least as much as the draft did).
- If p(x) < q(x): **accept** with probability p(x) / q(x); otherwise **reject**.
  Equivalently, accept with probability min(1, p(x) / q(x)).

**On rejection, resample** one token from the normalized residual distribution
  p′(x) = norm( max(0, p(x) − q(x)) ) = max(0, p(x) − q(x)) / ∑_x′ max(0, p(x′) − q(x′)) .

**Distribution-preservation proof (sketch).** The probability the loop outputs token x at this position is:
  P(accept x) + P(reject, then resample x)
  = min(p(x), q(x)) + ( p(x) − min(p(x), q(x)) )
  = p(x).
The accepted mass is min(p, q); the residual p − min(p, q) = max(0, p − q) is exactly recovered by the resample step. So **P(output = x) = p(x)** for any draft q — the draft only affects *speed*, never the output distribution. (This holds per position; chaining over the accepted run preserves the full sequence distribution.)

**Expected tokens per cycle.** Let α be the expected per-token acceptance probability (the *acceptance rate*), assuming tokens accept independently. Acceptance behaves like a capped geometric process (stop at first reject, cap at γ, plus one guaranteed corrected/bonus token):
  E[tokens] = (1 − α^(γ+1)) / (1 − α) .
(α → 1 gives γ+1; α → 0 gives 1.)

**Walltime speedup.** Let c be the cost ratio of one draft step to one target step (c = C_draft / C_target, with c ≪ 1). One cycle costs roughly γ draft steps + 1 target verification step ≈ (γc + 1) target-step-equivalents, and yields E[tokens]:
  Speedup = (1 − α^(γ+1)) / ( (1 − α)(γc + 1) ) .
- There is an **optimal γ** for each (α, c): larger γ raises potential tokens but linearly raises draft overhead γc and lowers the effective per-token accept chance.
- Net win requires roughly **α > c**: the draft must agree often enough to pay for its own cost.
- α is fundamentally bounded by the divergence between q and p — a draft too far from the target accepts rarely.

## 5. Practical takeaways & trade-offs

**When it wins**
- Memory-bandwidth-bound regimes: **low-to-moderate batch size / low concurrency**, latency-sensitive single-stream or few-stream serving. This is where the target model's parallel slack exists to absorb verification.
- Long outputs and large target models (the bigger the target, the more a cheap draft is worth).
- Tasks with predictable structure (code, summarization of grounded text) tend to show higher acceptance.

**When it does NOT help (failure modes)**
- **High concurrency / large batches.** Once the target model is already compute-saturated by many parallel requests, the "free" verification slack disappears; speedup shrinks and can plateau earlier than the non-speculative baseline (degradation reported around ~20–30+ concurrent requests on a single GPU). Multi-GPU tensor parallelism (TP > 1) mitigates this.
- **Poor draft–target alignment / low α.** Off-the-shelf drafts often underperform on domain-specific or distribution-shifted data; rejected drafts are wasted compute. Fine-tuning the draft on the target's domain (or on the target's own outputs) raises α.
- **γ set too high** under load can cause latency spikes (wasted draft work on tokens that get rejected).
- **Memory overhead:** an independent draft model means a second set of weights resident in GPU memory.

**Tuning knobs:** draft model size/quality (drives α and c), γ (draft length, has an optimum), acceptance scheme (exact vs. relaxed), and whether drafting is a tree vs. a single chain.

**Reported numbers (triangulated across sources):**
- Seminal encoder-decoder result: ~**2×–3×** on an 11B-class target with identical outputs (e.g. ~3.4× translation greedy, ~3.1× summarization greedy, ~2.6× translation with sampling).
- 70B-class decoder, distributed: ~**2×–2.5×** (e.g. ~1.92× summarization nucleus, ~2.01× greedy, ~2.46× code generation) with no quality change; a wide-shallow ~4B draft (few layers) was used to limit communication overhead.
- General rule of thumb from practitioner sources: at α ≥ ~0.6 and γ ≥ ~5, ~2–3× is typical for single-stream serving.
- Variant speedups (lossy/relaxed acceptance, so not distribution-identical): Medusa ~2×–2.5×; Lookahead ~1.5×–2.3×; EAGLE family among the strongest, with EAGLE reporting ~1.5×–2× over Medusa/Lookahead in head-to-head settings and ~2.4×+ overall on A100. **[unverified]** exact per-paper figures vary by hardware/task and should be treated as directional, not canonical.

**Variants — drafting strategy taxonomy:**
1. **Independent draft model** — a separate small LLM (often a smaller member of the same family). Classic, lossless with the exact acceptance rule. Cost: extra weights + alignment burden.
2. **Self-speculative / layer-skipping** (e.g. Draft & Verify) — the *target itself* drafts by skipping/early-exiting layers, then verifies with the full stack. No second model, no extra weights; draft and target are inherently aligned.
3. **Medusa** — adds a few lightweight extra decoding heads on top of the frozen target's last hidden state, each predicting a future position; candidates combined via tree attention. Uses a **typical-acceptance** threshold (lossy, not distribution-identical), and only the heads are trained.
4. **EAGLE (and EAGLE-2/3)** — drafts autoregressively at the **feature level** (the hidden state before the LM head) rather than the token level, reducing the uncertainty that hurts token-level drafting; later versions add dynamic draft **trees** and multi-layer feature fusion. Among the highest acceptance lengths reported.
5. **Lookahead decoding** — **model-free / training-free**: uses Jacobi-style parallel iteration to generate and verify n-grams against the target, maintaining an n-gram pool; no draft model and no datastore.
6. **Prompt / retrieval lookup (n-gram)** — draft candidates by copying n-grams from the prompt or a corpus; cheap and effective when output echoes input (RAG, editing, code).

**Verification structure:** **token-tree / tree attention** lets the system verify *many* candidate continuations (a branching tree of guesses) in one forward pass via a custom attention mask, raising the expected accepted length beyond a single linear draft chain.

**Lossless vs. lossy:** the exact modified-rejection rule (independent draft, self-speculative with full verification) preserves p exactly. Threshold/typical-acceptance schemes (e.g. Medusa) trade a small, usually-benign distribution change for more accepted tokens — flag this distinction on the page, since "speculative decoding is lossless" is only true for the exact variant.

## 6. Suggested interactive viz

**Primary (D3): "Draft, verify, accept" token-stream simulator.**
- Show a running sequence. Each cycle: the draft model emits γ ghosted candidate tokens; the target verifies; accepted tokens turn solid green, the first rejected token turns red and is replaced by a resampled (blue) "correction" token, and trailing drafts after the reject fade out.
- Sliders: acceptance rate α (0–1), draft length γ (1–10), cost ratio c (0–1). Live readouts compute and plot:
  - E[tokens] = (1 − α^(γ+1)) / (1 − α)
  - Speedup = (1 − α^(γ+1)) / ((1 − α)(γc + 1))
- A small curve panel: Speedup vs γ for the chosen (α, c), with the optimal γ marked — viewers *see* that more drafting is not always better and that α must exceed c to win.

**Optional secondary:** a side-by-side distribution view (draft q vs target p as bars) shading min(p, q) as "accept mass" and max(0, p − q) as the "resample residual," visually proving why the output equals p regardless of q.

## 7. Candidate placement

**Verdict: mint a new node** (do not just enrich). Speculative decoding currently appears only as a one-line sub-bullet inside the **Model Acceleration** page (`mlops/model-acceleration.html`) — "a small draft model proposes tokens a big model verifies." It is a substantial, math-rich topic (provable distribution preservation, a speedup calculus, a family of variants) that warrants its own page, consistent with how the site already breaks out comparably-sized inference/serving topics.

Proposed node:
- **id:** `Speculative Decoding`
- **group:** `mlops` (label "MLOps & Deployment") — it is an inference/serving acceleration technique; this matches its current home alongside Model Acceleration and LLMOps. (Secondary case could be made for `llm`, but the serving/efficiency framing fits `mlops` best and keeps it next to Model Acceleration.)
- **level:** 2
- **link / file path:** `mlops/speculative-decoding.html`
- **tree:** add `"Speculative Decoding"` to the `"MLOps & Deployment"` array in `tree`.

**Cross-link candidates (all exist in data.js; the ingestor should add the edges):**
- `Model Acceleration` — natural parent/sibling; reduce its sub-bullet to a pointer at this new page.
- `Large Language Models` — the models being accelerated; AR decoding is the bottleneck addressed.
- `Knowledge Distillation` — a common way to *train a well-aligned draft model* (higher α); already linked to Model Acceleration.
- `LLMOps` — production serving context (latency/throughput/cost), where the batch-size trade-offs live.
- (Optional) `Attention` / KV cache — speculative decoding composes with the KV cache and uses tree attention for verification; and `Mixture of Experts` is already cross-linked to Model Acceleration as a sibling efficiency idea.

Suggested cross edges to propose:
`["Speculative Decoding","Model Acceleration"]`, `["Speculative Decoding","Large Language Models"]`, `["Speculative Decoding","Knowledge Distillation"]`, `["Speculative Decoding","LLMOps"]`, and optionally `["Speculative Decoding","Attention"]`.

Suggested node desc (for data.js, neutral voice):
> "Speed up autoregressive decoding losslessly: a small draft model proposes several tokens that the large target model verifies in one parallel pass, with a modified rejection-sampling rule that provably preserves the target's output distribution. Covers the acceptance-rate/speedup math and variants (self-speculative, Medusa, EAGLE, lookahead)."

## 8. Sources consulted (audit trail only — must NOT appear on the published page)

- arxiv.org/abs/2211.17192 — Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding" (ICML 2023). Primary source: defines the algorithm, the acceptance rule, the distribution-preservation proof, and the E[tokens] / speedup formulas. Highly authoritative.
- arxiv.org/abs/2302.01318 — Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (DeepMind, 2023). Primary source: independent concurrent derivation, modified rejection sampling, Chinchilla-70B results, distributed wide-shallow draft. Highly authoritative. (Note: swaps p/q naming vs. the above.)
- aclanthology.org/2024.findings-acl.456.pdf — "Unlocking Efficiency in LLM Inference: A Comprehensive Survey of Speculative Decoding" (ACL Findings 2024). Peer-reviewed survey: taxonomy (independent draft / self-drafting / n-gram), token-tree verification, lossless vs lossy. Authoritative; one in-table label conflated Lookahead as layer-skipping — corrected against primary Lookahead source below.
- bentoml.com/llm/inference-optimization/speculative-decoding — practitioner guide. Useful for batch-size/concurrency failure modes, framework support (vLLM, SGLang), α/γ rules of thumb. Reputable engineering source; vendor blog, so treat numbers as directional.
- sites.google.com/view/medusa-llm — Medusa project page. Confirms extra-heads mechanism, tree attention, typical-acceptance (lossy), ~2–2.5× speedup. Primary author source for the variant.
- lmsys.org/blog/2023-11-21-lookahead-decoding/ + arxiv.org/abs/2402.02057 — Lookahead Decoding. Confirms model-free Jacobi-iteration + n-gram pool design and ~1.5×–2.3× speedup. Resolves the survey-table conflation.
- EAGLE: arxiv.org/abs/2401.15077 (EAGLE) and arxiv.org/abs/2406.16858 (EAGLE-2) — referenced for feature-level autoregression and dynamic draft trees. Figures cited as directional/**[unverified]** in exact magnitude.

### [unverified] gaps to flag downstream
- Exact head-to-head speedup magnitudes for EAGLE vs Medusa vs Lookahead vary by hardware (A100 etc.) and task; treat the variant numbers as directional, not canonical.
- "α ≥ 0.6, γ ≥ 5 → 2–3×" is a practitioner rule of thumb from a vendor source, not a peer-reviewed bound.
- The independence assumption behind E[tokens] = (1 − α^(γ+1))/(1 − α) is an approximation (real per-position acceptance is correlated); papers present it as an idealized model.
