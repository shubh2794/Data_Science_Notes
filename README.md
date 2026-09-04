# Data Science — Interactive Notes

A growing, interactive knowledge base for data science. The entry point is a
**knowledge graph** that maps the whole field; each topic links through to a
notes page with live, hands-on D3.js visualizations.

> Open `index.html` in a browser to start. No build step — just static HTML + D3 from a CDN.

## Structure

Notes are filed into one folder per **domain**, mirroring the domains in the
knowledge graph. Each topic gets its **own HTML page** — one graph end-node,
one file — but pages share a single theme, sidebar generator, and data file
(see *Architecture* below), so they're no longer standalone copies.

A large topic can be split into **level-3 sub-topics** (declared in `subtree`
in `data.js`): the topic page becomes a hub/overview and each sub-topic gets its
own page and its own node hanging off the parent in the graph. Linear Algebra
is the first such split — `math/linear-algebra.html` is the hub, with six
`math/linear-algebra-*.html` sub-pages (they share `math/linear-algebra-viz.js`).

```
ml-notes/
├── index.html                              # 🌌  Knowledge Universe (toggle: ✦ Galaxies / ❋ Radial)
├── data.js                                 # 📊  single source of truth (nodes, edges, descriptions)
├── notes.css                               # 🎨  shared theme for every notes page
├── notes.js                                # ⚙️  shared sidebar generator + scrollspy
├── README.md
│
├── machine-learning/
│   ├── linear-logistic-regression.html     # ✅
│   ├── k-nearest-neighbors.html            # ✅
│   ├── clustering.html                     # ✅
│   ├── support-vector-machines.html        # ✅
│   ├── naive-bayes.html                    # ✅
│   ├── decision-trees.html                 # ✅
│   ├── regularization.html                 # ✅
│   ├── dimensionality-reduction-pca.html   # ✅
│   ├── model-evaluation.html               # ✅
│   └── reinforcement-learning.html         # ✅
│
├── deep-learning/
│   ├── neural-networks.html                # ✅
│   ├── convolutional-networks.html         # ✅
│   ├── attention.html                      # ✅
│   ├── transformers.html                   # ✅
│   ├── mixture-of-experts.html             # ✅
│   ├── generative-adversarial-networks.html# ✅
│   ├── diffusion-models.html               # ✅
│   └── graph-neural-networks.html          # ✅
│
└── nlp/                                    # 11/11 pages built
    ├── tokenization.html                   # ✅ STRONG
    ├── embeddings.html                     # ✅ STRONG (gold-standard)
    ├── language-models.html                # ✅ STRONG
    ├── retrieval-augmented-generation.html # ✅ STRONG (gold-standard)
    ├── text-preprocessing.html             # ✅ STRONG (TF-IDF formula + viz)
    ├── named-entity-recognition.html       # ✅
    ├── machine-translation.html            # ✅ STRONG (BLEU + attention formulas + BLEU viz)
    ├── textual-entailment.html             # ✅ STRONG (softmax/zero-shot formulas + zero-shot viz)
    ├── document-intelligence.html          # ✅
    ├── knowledge-graphs.html               # ✅ STRONG (TransE formulas + scoring viz)
    └── hidden-markov-models-crfs.html      # ✅
```

## Domain taxonomy

The knowledge graph organizes everything under twelve domains. **142 nodes ·
81 topic pages built · 246 connections.** Remaining unbuilt nodes are the
original foundational placeholders (math, programming, CV, and generic
data-eng/MLOps/viz stubs).

| Domain | Folder | Status |
|--------|--------|--------|
| Machine Learning | `machine-learning/` | ✅ Regression, kNN, k-Means, SVM, Naive Bayes, Trees, Regularization, PCA, Model Eval, RL, Bias-Variance, Learning Paradigms, ML Compared, Hyperparameter Tuning, ML Strategy (15/15) |
| Deep Learning | `deep-learning/` | ✅ 18/19 (Neural Nets, CNNs, Attention, Transformers, MoE, GANs, Diffusion, GNNs, NN Training, Fine-Tuning, Distillation, Distributed, Encoder/Decoder, SSM, World Models/JEPA, Arch Compared, Residual, End-to-End) — RNNs stub |
| NLP | `nlp/` | ✅ 11/11 (Tokenization, Embeddings, LMs, RAG, Preprocessing, NER, MT, Entailment, Doc Intelligence, KGs, HMMs/CRFs) |
| LLMs & Generative AI | `llm/` | ✅ 10/10 (LLMs, Context Eng, PEFT, Preference Opt, Reasoning, Hallucination, LLM-as-Judge, VLM, Diffusion LLMs, Context-Length Ext) |
| Agentic AI | `agentic/` | ✅ 5/5 (Agents, Agentic RL, Agent Skills, Design Patterns, Computer Control) |
| Data Engineering | `data-engineering/` | 🟡 Data Preprocessing, Sampling/Imbalance, Quality/Governance, Differential Privacy (originals ETL/Warehouse/Feature Store/Stream = stubs) |
| MLOps & Deployment | `mlops/` | 🟡 LLMOps, Model Acceleration, A/B Testing, Federated Learning (originals Deploy/Monitor/CI-CD/Tracking = stubs) |
| Speech & Audio | `speech/` | ✅ Speech Processing (1/1) |
| Model Architectures | `models/` | ✅ BERT, GPT, CLIP, LayoutLM (v1/v2/v3), Donut (each with notes + research papers + visuals) — add more models here |
| Math & Statistics | `math/` | 🟡 Linear Algebra (hub + 6 sub-topic pages: Vectors & Vector Spaces, Matrices & Rank, Systems/Determinants/Inverses, Projections & Least-Squares, Eigendecomposition & SVD, Quadratic Forms/Covariance/PCA), Optimization — Probability, Statistics, Calculus = stubs |
| Programming & Tools | `programming-tools/` | ⏳ Scaffold nodes only |
| Computer Vision | `computer-vision/` | ⏳ Scaffold nodes only |
| Data Viz & Communication | `data-viz/` | ⏳ Scaffold nodes only |

## Architecture

The project is data-driven and static (no build step, opens over `file://`):

- **`data.js`** — the single source of truth: every node, edge, description, and
  `link`. Loaded as a plain `<script>` (a global `GRAPH`-style set of objects), so
  it works without a server. Both the graph and every notes page read from it.
- **`notes.css`** — one shared theme; every notes page links `../notes.css`
  instead of inlining ~150 lines of CSS.
- **`notes.js`** — generates each page's sidebar (sibling topics + "On this page")
  **from `data.js` at runtime**, plus the scrollspy and the viz colour palette.
  A page declares nothing about its siblings — add a topic to `data.js` and every
  sidebar updates itself.
- **`index.html`** — the graph is a **collapsible** radial tree: click a domain to
  expand/collapse its topics, click the centre to toggle all. Cross-links are drawn
  only between currently-visible nodes, so the view stays readable as it grows.
  `index-force.html` keeps the original force-directed layout as a fallback.

A notes page is therefore just: header + `section.topic` content + its own inline
viz `<script>`, loading `../data.js → ../notes.js →` its viz.

## Conventions

- **One folder per domain**, kebab-case, named to match the graph.
- **One HTML file per topic** (per graph end-node), kebab-case after the topic name.
- Notes pages are **data-driven**: link `../notes.css`, an empty
  `<nav class="sidebar" id="sidebar">`, and load `../data.js` + `../notes.js`
  before the page's own inline viz script.
- In the graph, a node lights up (glowing ring) automatically once it has a `link:`
  in `data.js`; otherwise it shows "Notes coming soon".

## Adding a new topic

1. Add a `{id, group, level:2, desc, link:"domain/file.html"}` entry to `nodes`
   in **`data.js`**, add its `id` to the right domain array in `tree`, and
   (optionally) any `cross` edges.
2. Create `domain/file.html`: copy an existing page, swap in the new content and
   its viz. The sidebar and the graph node update themselves from `data.js`.
