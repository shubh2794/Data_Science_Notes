# Research brief — LLMOps

> Scratch material for the ingestion pipeline. Not a published page. Source-neutral by design.
> Math is Unicode only (house rule): use ∑, ×, ·, ≈, ≥, ², subscripts (p₉₅), never LaTeX.
> This is an **ENRICH** brief — a node and page already exist (`mlops/llmops.html`). See §7 for exactly what the current page covers vs. what is missing.

## 1. Topic & scope

**LLMOps** (Large Language Model Operations) is the discipline of building, deploying, serving, evaluating, monitoring, and governing LLM-powered applications in production. It is the LLM-specific extension of MLOps: it inherits the CI/CD, deployment, monitoring, and lifecycle backbone of MLOps but adds a new set of problems that arise because the "model" is frequently a third-party API, the artifact you actually own and version is the **prompt + context + tools**, outputs are **non-deterministic and open-ended**, there is usually **no ground-truth label**, and **per-call cost and latency** are first-class engineering concerns.

In scope:
- The LLMOps lifecycle and the concrete ways it diverges from classic MLOps.
- Prompt management & versioning (prompts as first-class artifacts).
- RAG / retrieval operations as a production surface.
- Evaluation: offline evals, LLM-as-judge, online / A-B / feedback signals.
- Observability & tracing: token usage, latency percentiles, cost per request, distributed traces.
- Guardrails & safety (input/output), governance & security (PII, prompt injection, OWASP LLM Top 10).
- Caching (exact prompt-prefix caching and semantic caching) and its cost/latency math.
- Inference serving & cost optimization (continuous batching, quantization, speculative decoding, KV cache, routing).
- Data / feedback loops and the fine-tuning-vs-prompting decision.

Out of scope (belongs to neighboring nodes — cross-link, don't duplicate): the *internals* of RAG retrieval and ANN indexing, the *mechanics* of speculative decoding's acceptance proof, the *details* of quantization/pruning, the LLM-as-a-judge bias taxonomy in full, drift detection theory. This page is the **operational hub** that points at those.

## 2. Core concept

LLMOps is the operational backbone of LLM systems: the tools, workflows, and practices that move an LLM prototype into a reliable, observable, cost-controlled, governed production service. Where MLOps is organized around *training data → model artifact → batch/online serving → drift monitoring*, LLMOps re-centers the lifecycle around the **prompt/context as the versioned artifact**, an **evaluation regime that works without labels**, **runtime safety guardrails**, and **continuous cost/latency/quality monitoring** of calls to a (often externally hosted) foundation model.

A useful framing: **LLMOps = MLOps + (prompts-as-artifacts) + (non-determinism) + (eval-without-ground-truth) + (runtime safety) + (token-cost economics).**

The canonical component set that recurs across authoritative sources:
1. **Prompt management** — version control, diffing, and environment aliases (dev/staging/prod) for prompts and prompt templates.
2. **Evaluation** — offline eval suites run in CI as regression tests; online evals on live traffic; LLM-as-judge plus deterministic checks plus human review.
3. **Tracing / observability** — record every step (prompt, completion, tool calls, retrieved context, token counts, latency, cost) as distributed traces.
4. **Serving & cost optimization** — batching, caching, quantization, speculative decoding, KV cache, model routing.
5. **Guardrails & governance** — input/output filtering, PII handling, prompt-injection defense, audit trails, policy enforcement, an AI gateway as a central control plane (rate limits, cost tracking, key management).
6. **Feedback loops** — capture user signals and production failures to improve prompts, eval sets, and (when justified) fine-tuning data.

## 3. Intuition

- **The artifact moved.** In classic ML you ship a trained weights file. In LLM apps the weights often belong to a provider; what *your* team changes day to day is the prompt template, the retrieval configuration, the tool definitions, and the model/version selection. So those become the things you version, diff, test, and roll back — exactly the role source code plays in normal software.
- **You can't unit-test a probability distribution with `assert ==`.** The same prompt yields different text each run, and "correct" is fuzzy. Evaluation therefore shifts from exact-match accuracy to graded, rubric-based, judge-assisted scoring plus human spot-checks — and because the live distribution drifts (new user behavior, provider model updates), you must keep evaluating *in production*, not only before launch.
- **Every call is a metered transaction.** A traditional model's marginal inference cost is near-zero CPU. An LLM call costs real money per token and can take hundreds of milliseconds to seconds. That turns latency percentiles, token budgets, caching, and routing from nice-to-haves into core SLOs.
- **The input channel is an attack surface.** Because instructions and data share one text channel, untrusted content (a user message, a retrieved web page, a tool result) can hijack the model. Safety becomes a runtime concern guarded on both the way in and the way out, not just a training-time alignment property.
- **Mental model — a busy restaurant kitchen.** The recipe (prompt) is versioned and tested; the line cook (model) is fast but inconsistent, so a quality checker (eval + guardrail) inspects plates before they leave; a host turns away repeat orders with pre-made dishes (cache); orders are routed to the right station by difficulty (model routing); and a manager watches tickets-per-minute, wait times, and food cost (observability) in real time.

## 4. Key math (Unicode only)

**Per-request cost.** With separate input/output pricing (the common provider model):
  cost = (n_in × price_in + n_out × price_out)
where n_in, n_out are prompt and completion token counts and prices are per-token (often quoted per 1K or 1M tokens, so price_per_1M / 1e6 per token). A simple single-rate approximation:
  cost ≈ tokens × price_per_token.

**Effective cost with caching.** With cache hit-rate h ∈ [0,1] and cached-call cost c_hit (≈ 0 for an exact cache; a small embedding-lookup cost for semantic):
  cost_eff = h · c_hit + (1 − h) · c_miss ≈ (1 − h) · c_miss   (when c_hit ≈ 0)
So a hit-rate of h gives roughly an (1 − h) cost multiplier — h = 0.6 ⇒ ~40% of the uncached spend. The same form gives effective latency:
  latency_eff = h · L_hit + (1 − h) · L_miss.

**Provider prefix/prompt caching.** When a long fixed prefix (system prompt, few-shot, retrieved context) is cached at the provider, the *cached portion* of input tokens is billed at a discount d (e.g. cached input ≈ 0.1× of normal, i.e. d ≈ 0.9 off). With fraction f of input tokens cached:
  input_cost ≈ n_in · price_in · (1 − f · d).

**Latency percentiles (the metric that matters).** Report the distribution, not the mean: p₅₀ (typical), p₉₅, p₉₉ (tail). The point: a mean of ≈ 2 s can hide a p₉₉ of ≈ 45 s. For streaming endpoints also split:
  total_latency = TTFT + (n_out − 1) × TPOT
where TTFT = time-to-first-token (dominated by the prefill pass over the prompt) and TPOT = time-per-output-token (the decode loop). Perceived responsiveness is mostly TTFT; total cost/throughput is mostly the decode side.

**Throughput / batching.** Continuous (in-flight) batching raises tokens/second by packing many requests' decode steps together; system throughput ≈ Σ over active requests of their per-step token emission, bounded by GPU memory (the KV cache) and compute. Higher batch size ⇒ higher throughput per GPU ⇒ lower cost-per-token, at the price of higher per-request latency — the central serving trade-off.

**LLM-as-judge agreement.** Treat the judge as a noisy classifier and calibrate against a small human-labeled set: report agreement / precision / recall vs. human labels before trusting it. Reported pairwise agreement of a strong judge with humans is ≈ 80%+ — useful, not perfect.

## 5. Practical takeaways & trade-offs

**Prompt management & versioning**
- Treat prompts (and templates, few-shot examples, tool/function schemas, and the model+version pin) as code: store in version control, diff changes, attach them to traces, and use environment aliases (dev / staging / prod) so a prompt can be promoted or rolled back independently of app code.
- Pin the *provider model version* explicitly — silent provider model updates are a real source of regressions; treat a provider model bump like a dependency upgrade and re-run evals.

**RAG / retrieval ops**
- In a RAG app the retrieval layer is part of the production surface: monitor retrieval quality (was the right context fetched?), context relevance, and **faithfulness/groundedness** (did the answer stick to the retrieved context?), not just final answer quality. RAG also adds an attack surface (OWASP LLM08: vector/embedding weaknesses — embedding poisoning, access-control leakage across tenants).
- Cross-link to RAG and Vector Databases (HNSW) rather than re-deriving retrieval internals here.

**Evaluation (the defining LLMOps problem)**
- **Offline evals**: a curated eval set scored by deterministic checks (format/schema/regex), rubric scorers, and LLM-judges; run in CI as regression tests so prompt/model changes can't silently degrade quality before merge.
- **LLM-as-judge**: scalable automated scoring. Pointwise (reference-free) scoring suits production monitoring; pairwise comparison suits A/B-style "which is better" decisions; reference-based suits correctness/faithfulness checks. Known systematic biases: position bias, verbosity bias, self-enhancement (self-preference) bias — mitigate by randomizing order, controlling for length, and calibrating against human labels. Cross-link to LLM-as-a-Judge.
- **Online evals + A/B**: run evals on live production traffic to catch edge cases, drift, abuse, and cost spikes that offline sets miss; combine with A/B tests, thumbs up/down, and implicit signals (edits, retries, abandonment).
- The mature pattern is **all three layers**: deterministic checks for objective constraints, LLM-judge for qualitative quality at scale, human annotation for calibration and edge cases. Offline catches regressions pre-merge; online surfaces drift/abuse/cost in real time.

**Observability & tracing**
- Distributed **tracing** is the LLM-specific addition: capture each step — prompt, retrieved context, tool calls, completion, token counts, latency, cost — so a bad output can be debugged end to end. Monitoring answers "is it working?"; tracing/observability answers "why isn't it?".
- Core metrics to dashboard: latency p₅₀/p₉₅/p₉₉ (and TTFT for streaming), error/timeout/rate-limit rates, prompt vs completion tokens, **cost per request / per 1K tokens / per model / per user-tenant-feature**, cache hit ratio, and quality/safety scores from online evals. Instrument with OpenTelemetry-style conventions so LLM metrics live in the same stack as the rest of the system.

**Caching (real numbers)**
- **Exact / prefix caching** (provider-side): caching a long fixed prefix yields large savings on the cached portion — reported provider numbers include ≈ 90% cost reduction and ≈ 85% latency reduction on long prompts for prefix caching, and ≈ 50% input-cost savings from automatic caching. Lossless for the cached tokens.
- **Semantic caching** (e.g. GPTCache / Redis vector cache): embed the query, return a stored response when a past query is within a similarity threshold. Reported cache hit rates ≈ 61–69% and API-call reductions up to ≈ 68.8% on suitable workloads; overall inference cost cuts of ≈ 30–70% on conversational/repetitive traffic, with latency dropping toward sub-100 ms on hits. **The single most important knob is the similarity threshold** — too loose returns wrong answers (false hits), too tight kills the hit rate. One cited figure: ≈ 31% of LLM queries are semantically similar to a prior request, i.e. large latent savings in uncached deployments. **[unverified]** — these figures come from vendor/engineering sources and a small set of papers; treat as directional, workload-dependent, not guarantees.

**Inference serving & cost optimization**
- **Continuous (in-flight) batching** (Orca-style, in vLLM/TGI/SGLang) is the foundation of modern serving: dynamically swap finished sequences out and new requests in mid-batch to keep the GPU busy.
- **Quantization**: FP16 → INT8/INT4 cuts memory ≈ 2–4× and inference cost ≈ 50% while typically retaining ≈ 95–99% of quality; BF16 → FP8/INT8 reported ≈ 30% better cost-per-token. Cross-link to Model Acceleration.
- **Speculative decoding**: ≈ 2–3× decode speedup with off-the-shelf draft models, lossless for the exact variant; cross-link to Speculative Decoding. Effectiveness drops at high batch concurrency (note the interaction).
- **Stacking** these (e.g. FP8 + FlashAttention-class kernels + continuous batching + speculative decoding) is reported at ≈ 5–8× better cost-efficiency than naive FP16 + static batching. **[unverified]** — hardware-specific (H100-class) vendor figures.
- **Model routing**: send easy queries to a small/cheap model and hard ones to a large model; combine with caching to attack cost from multiple sides.
- **Build vs buy**: self-hosting (vLLM/TGI/SGLang on your own GPUs) gives control and can be cheaper at scale but adds ops burden; provider APIs are simpler but meter per token and update models under you.

**Fine-tuning vs prompting decision**
- Default order of escalation: prompt engineering → few-shot / better context → RAG (inject knowledge at runtime) → fine-tuning (PEFT/LoRA) → full fine-tune / continued pretraining. Prompting and RAG are cheaper to iterate and easy to roll back; fine-tuning is justified when you need consistent format/style/behavior, latency from shorter prompts, or capability not reachable by context. RAG vs fine-tuning is often framed as "knowledge that changes often → RAG; behavior/format that must be reliable → fine-tune." Cross-link to PEFT and Context Engineering.

**Guardrails & safety**
- **Input guardrails**: detect/block prompt injection and jailbreaks, scan for and redact PII, and **fence untrusted content** (retrieved docs, tool outputs, user text) so it can't be read as instructions.
- **Output guardrails**: validate format/schema, filter unsafe/toxic content, check grounding/faithfulness, and never execute LLM output without validation (OWASP LLM05, improper output handling).
- Prefer established frameworks (NeMo Guardrails, Bedrock Guardrails, LLM Guard) over hand-rolled checks; treat guardrails as defense-in-depth, not a single gate.

**Governance & security — OWASP Top 10 for LLM Applications (2025)**
- LLM01 Prompt Injection (still #1) · LLM02 Sensitive Information Disclosure (PII / training-data leakage) · LLM03 Supply Chain (compromised models/datasets/plugins) · LLM04 Data & Model Poisoning · LLM05 Improper Output Handling · LLM06 Excessive Agency (over-broad tool/permission scope for agents) · LLM07 System Prompt Leakage · LLM08 Vector & Embedding Weaknesses (RAG) · LLM09 Misinformation · LLM10 Unbounded Consumption (DoS / "denial-of-wallet" / model extraction).
- Prompt injection has two flavors: **direct** (user manipulates the prompt) and **indirect** (malicious instructions arrive via retrieved/tool content). There is no complete fix today — manage it with defense-in-depth (input filtering, system-prompt hardening, output validation, least-privilege tool scope, behavioral monitoring).
- Governance layer: audit trails of prompts/outputs, policy enforcement, an **AI gateway** as a central control plane (key management, rate limiting, per-team cost tracking, budget caps to bound LLM10).

**Failure modes to call out**
- Treating offline eval as sufficient → drift and edge cases bite in production.
- No prompt versioning → unreproducible regressions; silent provider model updates → unexplained quality drops.
- Mean-only latency alerting → tail (p₉₉) pain is invisible.
- Over-loose semantic cache threshold → confidently wrong cached answers.
- Skipping output validation → injected/unsafe output executed downstream.

## 6. Suggested interactive viz

The existing page already has one viz (the request-flow cache-hit/miss diagram, `#viz-llmops`). For an enrich, **keep it** but add a second, more quantitative viz that makes the cost/latency economics tangible:

**Primary new viz (D3): "Cost & latency under caching + routing" calculator.**
- Sliders: requests/day, mean input tokens n_in, mean output tokens n_out, cache hit-rate h (0–1), and a "% routed to cheap model" slider; dropdown to pick model price tiers.
- Live readouts compute and plot:
  - cost_per_call = n_in × price_in + n_out × price_out
  - cost_eff = (1 − h) × cost_per_call (with optional routing blend)
  - monthly spend = cost_eff × requests/day × 30
  - latency_eff = h × L_hit + (1 − h) × L_miss
- A small curve: monthly spend vs. cache hit-rate h, so the viewer *sees* the (1 − h) slope and how a 60% hit-rate roughly cuts spend to ~40%.

**Optional secondary:** a "latency distribution" strip showing p₅₀/p₉₅/p₉₉ with a draggable tail, illustrating why mean-only alerting hides the p₉₉ problem (toggle a "long-tail outlier" to watch the mean barely move while p₉₉ explodes).

## 7. Candidate placement

**Verdict: ENRICH the existing node — do NOT mint.** The node already exists in `data.js` and has a page:
- node: `{id:"LLMOps", group:"mlops", level:2, link:"mlops/llmops.html", desc:"Operating LLM apps in production — prompt/version management, evaluation, guardrails, caching, and cost/latency monitoring."}`
- existing cross edges in `data.js`: `["LLMOps","Model Deployment"]`, `["LLMOps","Large Language Models"]`, `["LLMOps","LLM-as-a-Judge"]`, plus `["System Design","LLMOps"]` and `["Speculative Decoding","LLMOps"]`.

**What the current page (`mlops/llmops.html`) already covers (do not re-do, just deepen):**
- §01 How LLMOps differs from MLOps (prompt/context as artifact, non-determinism, cost/latency, injection/hallucination) — solid framing, thin.
- §02 The request path + a D3 cache hit/miss flow viz.
- §03 Evaluation — offline (judges/rubrics/exact checks in CI) vs online (A/B, thumbs, implicit), faithfulness/safety. Brief.
- §04 Cost & latency — caching (exact + semantic), batching/streaming/KV cache, model routing. Brief.
- §05 Guardrails & monitoring — input/output guardrails, fencing, monitor quality/cost/latency/drift. Brief.
- §06 Takeaways.

**Gaps this brief fills (the enrichment payload):**
- **Prompt management depth** — versioning/diffing, env aliases (dev/staging/prod), pinning provider model versions; currently only implied.
- **RAG/retrieval ops** as a production surface (retrieval quality, faithfulness, RAG attack surface) — absent.
- **Quantitative cost/latency math** — per-request cost, caching multiplier (1 − h), prefix-cache discount, TTFT/TPOT split, p₅₀/p₉₅/p₉₉ — page currently has only illustrative readouts, no formulas. The site is explicitly Unicode-math-friendly; add a formula block.
- **Real numbers** — semantic cache hit-rates (≈ 61–69%), prefix caching (≈ 90% cost / 85% latency), quantization (≈ 50% cost, 95–99% quality), speculative decoding (≈ 2–3×), stacked ≈ 5–8×. Page has none.
- **Inference serving detail** — continuous/in-flight batching, quantization, speculative decoding link, routing, build-vs-buy — only lightly touched.
- **Fine-tuning vs prompting/RAG decision ladder** — absent.
- **Governance/security as a structured topic** — OWASP LLM Top 10 (2025), direct vs indirect injection, AI gateway, audit trails, denial-of-wallet — page mentions injection/PII in one bullet only.
- **Observability depth** — tracing vs monitoring distinction, OpenTelemetry instrumentation, per-tenant/feature cost attribution.
- **Second viz** — the cost/latency calculator in §6.

**Cross-link candidates (all exist in `data.js`; the ingestor should ensure these edges exist).** Already present: `Model Deployment`, `Large Language Models`, `LLM-as-a-Judge`, `Speculative Decoding`. **Recommend adding:**
- `["LLMOps","Monitoring & Drift"]` — observability/drift is core to the enrichment.
- `["LLMOps","Model Acceleration"]` — quantization/serving-efficiency section.
- `["LLMOps","RAG"]` — RAG/retrieval-ops section.
- `["LLMOps","A/B Testing"]` — online evaluation section (page already links the ab-testing.html page inline).
- `["LLMOps","Context Engineering"]` — prompt/context-as-artifact (page already links inline).
- (Optional) `["LLMOps","Parameter-Efficient Fine-Tuning"]` — fine-tuning-vs-prompting ladder; `["LLMOps","Vector Databases (HNSW)"]` — RAG retrieval layer; `["LLMOps","Hallucination & Factuality"]` — faithfulness (page already links inline).

**No `data.js` node-desc change required**, though it could optionally be broadened to mention serving/cost-optimization and governance; current desc is still accurate. (Reminder: this agent does not edit `data.js` — recommendation only.)

## 8. Sources consulted (audit trail only — must NOT appear on the published page)

- mlflow.org/llmops — MLflow's LLMOps overview. Primary/official tooling source: definition, MLOps-vs-LLMOps differences, the component set (tracing, evaluation, prompt management with version control + env aliases, production monitoring, AI gateway, governance). Authoritative, vendor-neutral framing.
- owasp.org/www-project-top-10-for-large-language-model-applications (2025 PDF; list cross-read via aembit.io/blog/owasp-top-10-llm-risks-explained and promptfoo.dev/docs/red-team/owasp-llm-top-10) — the governance/security backbone (LLM01–LLM10, direct vs indirect prompt injection). Primary standard; PDF too large to fetch directly, list triangulated across two reputable summaries.
- evidentlyai.com/llm-guide/llm-as-a-judge — LLM-as-judge mechanics: pointwise/pairwise/reference-based, position/verbosity/self-enhancement biases, calibration vs human labels, ≈80%+ pairwise agreement figure, offline vs online use. Reputable engineering guide.
- arxiv.org/abs/2411.05276 (GPT Semantic Cache) + ResearchGate GPTCache writeup + spheron.network / introl.com caching guides — semantic & prefix caching numbers (hit-rates ≈ 61–69%, API-call reduction ≈ 68.8%, prefix caching ≈ 90% cost / 85% latency, ≈ 31% semantically-similar queries, similarity-threshold sensitivity). Mix of paper + reputable engineering blogs; numbers treated as directional.
- LLM inference optimization guides (morphllm.com, runpod.io, vLLM/quantization writeups) — continuous batching (Orca/vLLM), quantization ≈ 50% cost / 95–99% quality, speculative decoding ≈ 2–3× (stacked ≈ 5–8×), TTFT/TPOT, throughput-vs-latency batching trade-off, vLLM/TGI/SGLang throughput figures. Reputable engineering sources; H100-specific numbers flagged unverified.
- LLM observability guides (splunk.com LLM observability, traceloop.com, parseable.com vLLM OpenTelemetry, inference.net) — metric set (p₅₀/p₉₅/p₉₉, TTFT, error/rate-limit rates, per-token/per-tenant cost, cache hit ratio), tracing-vs-monitoring distinction, OpenTelemetry instrumentation, dashboard structure. Reputable engineering sources.
- LLM evaluation guides (arize.com, databricks.com best-practices, langchain/langsmith docs, milestone offline-vs-online) — offline vs online eval, the three-layer pattern (deterministic + judge + human), CI regression-test framing, drift/abuse/cost-spike detection online. Authoritative/reputable.

### [unverified] gaps to flag downstream
- All caching, quantization, and speculative-decoding magnitudes are workload- and hardware-dependent vendor/engineering figures, not peer-reviewed bounds — present as "reported ≈", directional.
- The ≈ 5–8× "stacked optimization" figure is a single H100-class vendor claim; do not present as canonical.
- LLM-as-judge "≈80%+ agreement" is for a strong judge (GPT-4-class) on pairwise comparison in cited research; agreement varies by task and judge model.
- OWASP list pulled from 2025 summaries (primary PDF exceeded fetch size); names verified consistent across two independent reputable summaries — high confidence but not read from the primary PDF directly.
