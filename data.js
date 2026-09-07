/* ============================================================
   data.js — single source of truth for the knowledge graph.
   Loaded as a plain <script> (works over file://) so it exposes
   GROUPS, nodes, tree, cross, meta, rootData, ADJ as globals to
   both index.html (the graph) and notes.js (per-page sidebars).

   To add a topic:
     1. add a {id,group,level,desc,link?} entry to `nodes`
     2. add its id to the right domain array in `tree`
     3. (optional) add edges to `cross`  (3rd item "isa" = hierarchy bridge)
   To split a topic into sub-topics (level 3):
     1. add {id,group,level:3,desc,link?} entries to `nodes`
     2. list their ids under the parent topic in `subtree`
     (the parent keeps its own page as the hub/overview)
   Sidebars + the graph update themselves from this file.
   ============================================================ */

const GROUPS = {
  root:  {label:"Data Science", short:"Root", color:"#e6e9ef"},
  math:  {label:"Math & Statistics", short:"Math", color:"#5b9cff"},
  prog:  {label:"Programming & Tools", short:"Programming", color:"#4ade80"},
  dsa:   {label:"Data Structures & Algorithms", short:"DSA", color:"#fb7185"},
  dsys:  {label:"Data Systems", short:"Data Systems", color:"#818cf8"},
  deng:  {label:"Data Engineering", short:"Data Eng", color:"#22d3ee"},
  ml:    {label:"Machine Learning", short:"ML", color:"#ffb454"},
  dl:    {label:"Deep Learning", short:"DL", color:"#f87171"},
  cv:    {label:"Computer Vision", short:"CV", color:"#f472b6"},
  nlp:   {label:"NLP", short:"NLP", color:"#c084fc"},
  mlops: {label:"MLOps & Deployment", short:"MLOps", color:"#a3e635"},
  viz:   {label:"Data Viz & Communication", short:"Data Viz", color:"#fcd34d"},
  llm:   {label:"LLMs & Generative AI", short:"LLMs", color:"#2dd4bf"},
  agent: {label:"Agentic AI", short:"Agentic", color:"#fb923c"},
  speech:{label:"Speech & Audio", short:"Speech", color:"#38bdf8"},
  models:{label:"Model Architectures", short:"Models", color:"#e879f9"},
};

const nodes = [
  {id:"Data Science", group:"root", level:0, desc:"The umbrella field of extracting knowledge and insight from data — spanning the math, engineering, modeling, and communication needed to turn raw data into decisions."},

  // domains
  {id:"Math & Statistics", group:"math", level:1, desc:"The theoretical foundation: the language used to describe data, uncertainty, and learning algorithms."},
  {id:"Programming & Tools", group:"prog", level:1, desc:"The practical toolkit for manipulating data and implementing models at scale."},
  {id:"Data Engineering", group:"deng", level:1, desc:"Building the pipelines and storage that deliver clean, reliable data to models and analysts."},
  {id:"Machine Learning", group:"ml", level:1, desc:"Algorithms that learn patterns from data to make predictions or decisions without being explicitly programmed."},
  {id:"Deep Learning", group:"dl", level:1, desc:"Multi-layer neural networks that learn hierarchical representations directly from raw data."},
  {id:"Computer Vision", group:"cv", level:1, desc:"Recovering geometry, motion and meaning from images — the classical pipeline that runs from photons and lenses to features, correspondence and 3D structure. The learned view of vision lives in Deep Learning; this domain owns the geometry and signal processing it is built on."},
  {id:"Classical Vision", group:"cv", level:2, link:"vision/index.html", desc:"Hub page for the classical computer-vision series: the map of the subject, the pipeline that connects the parts, and a cheat sheet. Seven sub-topics below run from image formation to stereo and structure from motion."},
  {id:"Image Formation", group:"cv", level:3, link:"vision/image-formation.html", desc:"How a scene becomes an array of numbers: geometric primitives and 2D/3D transformations, the pinhole camera and projection, intrinsics and extrinsics, lens distortion, photometric formation and light transport, colour, and the sensor pipeline that ends in pixels."},
  {id:"Image Processing", group:"cv", level:3, desc:"Operations on pixels before any interpretation: point operators and histogram equalisation, linear filtering and convolution, separable and Gaussian kernels, non-linear and bilateral filtering, median and rank filters, mathematical morphology, pyramids and multi-resolution, and geometric warping."},
  {id:"Frequency Domain", group:"cv", level:3, desc:"Images as signals: the Fourier transform and the DFT, the sampling theorem and aliasing, filtering as multiplication in the frequency domain, the convolution theorem, windowing, and the wavelet and DCT transforms that underpin image compression."},
  {id:"Features & Matching", group:"cv", level:3, desc:"Finding what is repeatable: edges and the Canny detector, corners and the Harris response, blob and scale-space detection, invariant descriptors, matching strategies and ratio tests, and robust fitting with RANSAC."},
  {id:"Segmentation", group:"cv", level:3, desc:"Partitioning an image into regions: thresholding and Otsu's method, region growing and split-and-merge, watershed, active contours, graph cuts and normalised cuts, mean shift, and superpixels."},
  {id:"Alignment & Motion", group:"cv", level:3, desc:"Putting images into correspondence: parametric motion models, homography estimation, image warping and stitching, optical flow with Lucas-Kanade and Horn-Schunck, the aperture problem, and coarse-to-fine tracking."},
  {id:"Stereo & Structure from Motion", group:"cv", level:3, desc:"Recovering the third dimension: epipolar geometry and the fundamental matrix, rectification, stereo correspondence and disparity, triangulation, camera calibration, bundle adjustment, and the structure-from-motion pipeline."},
  {id:"NLP", group:"nlp", level:1, desc:"Natural Language Processing — teaching machines to understand and generate human language."},
  {id:"MLOps & Deployment", group:"mlops", level:1, desc:"The discipline of shipping, serving, and maintaining ML systems reliably in production."},
  {id:"Data Viz & Communication", group:"viz", level:1, desc:"Turning analysis into clear visuals and narratives that drive decisions."},
  {id:"LLMs & Generative AI", group:"llm", level:1, desc:"The modern generative stack built on large pretrained models: prompting and context, parameter-efficient adaptation, preference alignment, evaluation, and multimodal + diffusion variants."},
  {id:"Agentic AI", group:"agent", level:1, desc:"LLMs that plan, use tools, and act over multiple steps to accomplish goals — agents, the skills they wield, how they're trained, and the design patterns that structure them."},
  {id:"Model Architectures", group:"models", level:1, desc:"Landmark models, each with notes, key research papers, and visuals — BERT, GPT, and the families that followed."},

  // Math & Statistics
  {id:"Linear Algebra", group:"math", level:2, link:"math/linear-algebra.html", desc:"Vectors, matrices, eigen-decompositions — the substrate of nearly every ML computation. Hub page: the map of the subject, split into six sub-topics below."},
  // Linear Algebra sub-topics (level 3) — one page each, in book-chapter order
  {id:"Vectors & Vector Spaces", group:"math", level:3, link:"math/linear-algebra-vectors.html", desc:"Vectors as ordered lists and as arrows; the dot, outer, Hadamard and cross products; span, linear independence, basis, and subspaces — the vocabulary every later topic uses."},
  {id:"Matrices & Rank", group:"math", level:3, link:"math/linear-algebra-matrices.html", desc:"Matrices as data tables and as linear transformations; the matrix zoo, matrix multiplication and its four views, norms, rank, and the four fundamental subspaces (column, row, null, left-null)."},
  {id:"Systems, Determinants & Inverses", group:"math", level:3, link:"math/linear-algebra-systems.html", desc:"Solving Ax = b by row reduction (Gaussian / Gauss-Jordan elimination, RREF), the determinant as a volume-scaling singularity test, and the inverse, one-sided inverses, and the Moore-Penrose pseudoinverse."},
  {id:"Projections & Least-Squares", group:"math", level:3, link:"math/linear-algebra-projections.html", desc:"Projecting a vector onto a line or subspace, orthogonal matrices, Gram-Schmidt and the QR decomposition, and least-squares as the projection of the data onto the column space of the design matrix."},
  {id:"Eigendecomposition & SVD", group:"math", level:3, link:"math/linear-algebra-eigen-svd.html", desc:"Complex numbers, eigenvalues and eigenvectors, diagonalization, symmetric matrices, and the singular value decomposition — rank, subspaces, condition number, pseudoinverse, and low-rank approximation from one factorization."},
  {id:"Quadratic Forms, Covariance & PCA", group:"math", level:3, link:"math/linear-algebra-quadratic-pca.html", desc:"The quadratic form xᵀAx and matrix definiteness, covariance and correlation matrices as Gram matrices of centered data, and PCA as the eigendecomposition of the covariance matrix."},
  {id:"Probability", group:"math", level:2, desc:"Reasoning about uncertainty: distributions, Bayes' rule, expectation, conditional probability."},
  {id:"Statistics", group:"math", level:2, link:"math/statistics/index.html", desc:"Reasoning from a sample to the population it came from. Hub page: the map of the subject, split into nine sub-topics below — description, design, sampling distributions, estimation, testing, regression, resampling, categorical data, and ANOVA."},
  // Statistics sub-topics (level 3) — one page each, in course order
  {id:"Descriptive Statistics", group:"math", level:3, link:"math/statistics/descriptive.html", desc:"Summarising a batch of numbers before modelling it: histograms and shape, the mean/median/mode trio and when each lies, spread by SD, IQR and MAD, quartiles and boxplots, z-scores and standardisation, outliers, and the plots that reveal what a summary statistic hides."},
  {id:"Producing Data & Sampling", group:"math", level:3, link:"math/statistics/sampling-design.html", desc:"Where the numbers come from decides what they can prove: population vs sample, simple random / stratified / cluster designs, sampling and non-sampling bias, observational study vs randomised experiment, confounding, blinding, and why randomisation is what licenses a causal claim."},
  {id:"Sampling Distributions & CLT", group:"math", level:3, link:"math/statistics/sampling-distributions.html", desc:"The distribution of a statistic across hypothetical repeated samples: parameter vs statistic, the sampling distribution of the mean and of a proportion, the law of large numbers, the central limit theorem and its 1/√n standard error — the result every confidence interval and test rests on."},
  {id:"Estimation & Confidence Intervals", group:"math", level:3, link:"math/statistics/estimation.html", desc:"Turning a point estimate into an interval: standard error, the z and t intervals, what '95% confident' does and does not mean, margin of error and sample-size planning, intervals for proportions, and the bias-variance and coverage properties of an estimator."},
  {id:"Hypothesis Testing", group:"math", level:3, link:"math/statistics/hypothesis-testing.html", desc:"The logic of a significance test: null and alternative, the test statistic and its null distribution, p-values and what they are not, one- vs two-sided, type I and II error, power and sample size, the duality with confidence intervals, and the replication-crisis critique."},
  {id:"Regression & Correlation", group:"math", level:3, link:"math/statistics/regression.html", desc:"The inferential view of the fitted line: correlation and its traps, the regression effect and the fallacy named after it, inference on the slope, residual diagnostics, prediction vs confidence bands, extrapolation, and why r² is not a licence to claim causation."},
  {id:"Resampling (Bootstrap & Permutation)", group:"math", level:3, link:"math/statistics/resampling.html", desc:"Replacing distributional assumptions with computation: the bootstrap for standard errors and intervals, permutation tests for comparing groups, when resampling works and when it fails, and the connection to cross-validation."},
  {id:"Categorical Data & Chi-Square", group:"math", level:3, link:"math/statistics/categorical.html", desc:"Counts rather than measurements: two-way tables, conditional distributions, the chi-square tests of goodness-of-fit and independence, expected counts and residuals, odds ratios and relative risk, and Simpson's paradox."},
  {id:"ANOVA & Multiple Comparisons", group:"math", level:3, link:"math/statistics/anova.html", desc:"Comparing several groups at once: the F test as a ratio of between- to within-group variance, its assumptions, and then the multiplicity problem — data snooping, family-wise error, Bonferroni and Tukey, and false discovery rate control."},
  {id:"Calculus", group:"math", level:2, link:"math/calculus.html", desc:"Limits, derivatives, Taylor expansion, series, integrals, and matrix differentials — the engine behind optimization and backpropagation."},
  {id:"Optimization", group:"math", level:2, link:"math/optimization.html", desc:"Finding parameters that minimize loss: gradient descent, convexity, momentum, Adam."},

  // Programming & Tools
  {id:"Python", group:"prog", level:2, link:"programming/python.html", desc:"The lingua franca of data science — rich ecosystem of ML and data libraries."},
  {id:"SQL", group:"prog", level:2, link:"programming/sql.html", desc:"Querying and aggregating structured data from relational databases."},
  {id:"Pandas & NumPy", group:"prog", level:2, link:"programming/pandas-numpy.html", desc:"Vectorized arrays and dataframes for in-memory data manipulation. Hub page: why whole-array thinking wins and the shared performance ladder; the two libraries are sub-topics."},
  {id:"NumPy", group:"prog", level:3, link:"programming/numpy.html", desc:"The ndarray memory model — shape, dtype, strides, views vs copies, broadcasting, and axis reductions — the mental model every array library since has copied."},
  {id:"Pandas", group:"prog", level:3, link:"programming/pandas.html", desc:"Labelled, aligned tables on top of NumPy: Series/DataFrame and the Index, dtypes and Copy-on-Write, groupby (split-apply-combine), merge/join/concat, wide↔long reshaping, and missing data."},
  {id:"Git & Version Control", group:"prog", level:2, desc:"Tracking the history of a project as a directed acyclic graph of immutable snapshots: the content-addressed object model, HEAD/branch/index/working-tree, branching and three-way merge vs rebase, bisect as binary search over history, and the data-science caveats — notebooks that diff badly, data too large to commit, and what reproducibility actually requires."},
  {id:"R", group:"prog", level:2, desc:"A statistics-first language strong in inference and visualization."},
  {id:"Apache Spark", group:"prog", level:2, desc:"Distributed computation for datasets too large for one machine."},
  {id:"Data Structures & Algorithms", group:"dsa", level:1, link:"dsa/data-structures-algorithms.html", desc:"The CS core: arrays, trees, graphs, sorting/searching, dynamic programming, and Big-O complexity analysis."},
  {id:"Data Systems", group:"dsys", level:1, desc:"The infrastructure data lives in: relational and NoSQL stores, indexing and transactions, distributed storage and compute, streaming, lakes and warehouses, and the scaling patterns (caching, queues, load balancing, CAP) that hold it together at volume."},
  {id:"Arrays & Strings", group:"dsa", level:2, link:"dsa/arrays-strings.html", desc:"Contiguous indexed storage and string manipulation — the most common building-block structures."},
  {id:"Trees & Graphs", group:"dsa", level:2, link:"dsa/trees-graphs.html", desc:"Hierarchical and networked structures; traversal (BFS/DFS), balancing, and shortest paths."},
  {id:"Sorting & Searching", group:"dsa", level:2, link:"dsa/sorting-searching.html", desc:"Ordering data and locating elements efficiently — quicksort, mergesort, binary search."},
  {id:"Dynamic Programming", group:"dsa", level:2, link:"dsa/dynamic-programming.html", desc:"Solve problems by caching overlapping subproblems — the classic optimization technique."},
  {id:"Big-O Complexity", group:"dsa", level:2, link:"dsa/big-o-complexity.html", desc:"Reasoning about time and space cost as input grows — the language of algorithm efficiency."},
  {id:"Relational & SQL", group:"dsys", level:2, desc:"Tables, keys, joins, and the SQL query language over normalized relational schemas."},
  {id:"NoSQL", group:"dsys", level:2, desc:"Non-relational stores — key-value, document, columnar, graph — for scale and flexibility."},
  {id:"Indexing", group:"dsys", level:2, desc:"Data structures (B-trees, hashes) that make lookups fast at some write/space cost."},
  {id:"Transactions (ACID)", group:"dsys", level:2, desc:"Atomicity, consistency, isolation, durability — guarantees for correct concurrent writes."},
  {id:"Sharding & Replication", group:"dsys", level:2, desc:"Partitioning data across machines and copying it for scale and availability."},
  {id:"Hadoop & MapReduce", group:"dsys", level:2, desc:"The batch paradigm that started big data: distribute work as map and reduce steps over a cluster."},
  {id:"Streaming (Kafka)", group:"dsys", level:2, desc:"Processing unbounded event streams in near-real-time; Kafka as the durable log."},
  {id:"Data Lakes", group:"dsys", level:2, desc:"Store raw data at scale in open formats, schema-on-read, for flexible downstream use."},
  {id:"Distributed Storage", group:"dsys", level:2, desc:"Spreading data across nodes (HDFS, object stores) for capacity and fault tolerance."},
  {id:"Warehouses & Lakehouses", group:"dsys", level:2, desc:"Structured analytical stores and the lakehouse blend of lake flexibility + warehouse speed."},
  {id:"Scalability", group:"dsys", level:2, desc:"Handling growth by scaling up (bigger) or out (more machines); statelessness and partitioning."},
  {id:"Load Balancing", group:"dsys", level:2, desc:"Distributing requests across servers for throughput and resilience."},
  {id:"Caching", group:"dsys", level:2, desc:"Storing hot results closer to compute to cut latency and load (CDN, Redis, app cache)."},
  {id:"Message Queues", group:"dsys", level:2, desc:"Decoupling producers and consumers with async queues (Kafka, RabbitMQ, SQS)."},
  {id:"CAP Theorem", group:"dsys", level:2, desc:"Under partition, choose consistency or availability — the core distributed-systems trade-off."},

  // Data Engineering
  {id:"ETL Pipelines", group:"deng", level:2, desc:"Extract-Transform-Load workflows that move and reshape data between systems."},
  {id:"Data Warehousing", group:"deng", level:2, desc:"Central analytical stores (Snowflake, BigQuery) optimized for queries over large history."},
  {id:"Feature Stores", group:"deng", level:2, desc:"Managed repositories of curated features, shared and consistent across training and serving."},
  {id:"Stream Processing", group:"deng", level:2, desc:"Processing data in real time as it arrives (Kafka, Flink)."},

  // Machine Learning — specific algorithms (one notes page each)
  {id:"Linear & Logistic Regression", group:"ml", level:2, link:"machine-learning/linear-logistic-regression.html", desc:"Fit a line for continuous targets, or a sigmoid-squashed line for class probabilities. Hub page: what the two share, how they differ, scaling, regularization, and when to use them; each model is a sub-topic."},
  {id:"Linear Regression", group:"ml", level:3, link:"machine-learning/linear-regression.html", desc:"Least-squares line fitting: the assumptions, why squared error (Gaussian MLE, convexity), the normal equation vs gradient descent, R²/adjusted R² and residual diagnostics, with an interactive fit."},
  {id:"Logistic Regression", group:"ml", level:3, link:"machine-learning/logistic-regression.html", desc:"Classification with a sigmoid-squashed linear score: the sigmoid derived from log-odds, cross-entropy vs MSE and convexity, decision threshold and calibration, perfect separation, softmax and one-vs-rest for many classes."},
  {id:"k-Nearest Neighbors", group:"ml", level:2, link:"machine-learning/k-nearest-neighbors.html", desc:"Classify by majority vote among the k closest stored examples — lazy and non-parametric."},
  {id:"Clustering (k-Means)", group:"ml", level:2, link:"machine-learning/clustering.html", desc:"The flagship UNSUPERVISED algorithm here (no labels): partition data by alternating assign-to-centroid and move-centroid steps."},
  {id:"Dimensionality Reduction (PCA)", group:"ml", level:2, link:"machine-learning/dimensionality-reduction-pca.html", desc:"The other major unsupervised method: project data onto the directions of greatest variance (eigenvectors of the covariance matrix) to compress while preserving structure."},
  {id:"Support Vector Machines", group:"ml", level:2, link:"machine-learning/support-vector-machines.html", desc:"Find the maximum-margin separating hyperplane; the kernel trick handles non-linear boundaries."},
  {id:"Naive Bayes", group:"ml", level:2, link:"machine-learning/naive-bayes.html", desc:"Bayes' rule with a conditional-independence assumption — fast and strong for text."},
  {id:"Decision Trees & Ensembles", group:"ml", level:2, link:"machine-learning/decision-trees.html", desc:"Recursive impurity-reducing splits, combined via bagging (forests) and boosting."},
  {id:"Regularization", group:"ml", level:2, link:"machine-learning/regularization.html", desc:"Prevent overfitting by penalizing complex models: L1 (Lasso) drives weights to zero for sparsity, while L2 (Ridge) shrinks weights uniformly. Includes deep learning dropout and label smoothing."},
  {id:"Reinforcement Learning", group:"ml", level:2, link:"machine-learning/reinforcement-learning.html", desc:"The third ML paradigm (beyond supervised/unsupervised): an agent learns a policy by trial-and-error to maximize reward from environment feedback."},
  {id:"Model Evaluation", group:"ml", level:2, link:"machine-learning/model-evaluation.html", desc:"A CROSS-CUTTING methodology (not an algorithm): how to score ANY model honestly — train/test splits, cross-validation, precision/recall, ROC-AUC."},

  // Deep Learning  (most link to the notes page)
  {id:"Feedforward / MLP", group:"dl", level:2, link:"deep-learning/neural-networks.html", desc:"The base neural network: stacked layers of weighted sums + non-linearities, trained by backprop. CNNs, Transformers, etc. are specialized variants."},
  {id:"CNNs", group:"dl", level:2, link:"deep-learning/convolutional-networks.html", desc:"Convolutional networks that slide learnable filters across grids — the backbone of vision."},
  {id:"RNNs & LSTMs", group:"dl", level:2, link:"deep-learning/rnns-lstms.html", desc:"Recurrent networks that process sequences by carrying hidden state through time."},
  {id:"Attention", group:"dl", level:2, link:"deep-learning/attention.html", desc:"Learned content-based lookup: each element decides which others to read from and how much. From the 2015 seq2seq fix to MQA/GQA/MLA in modern LLMs."},
  {id:"Transformers", group:"dl", level:2, link:"deep-learning/transformers.html", desc:"Attention-based architecture that processes whole sequences in parallel — the basis of modern LLMs."},
  {id:"Mixture of Experts", group:"dl", level:2, link:"deep-learning/mixture-of-experts.html", desc:"Sparse scaling: replace the dense FFN with many expert FFNs and route each token to only the top-k — huge parameter count, small per-token compute (Switch, Mixtral, DeepSeek)."},
  {id:"GANs", group:"dl", level:2, link:"deep-learning/generative-adversarial-networks.html", desc:"A generator and discriminator competing in a minimax game to synthesize realistic data."},
  {id:"Diffusion Models", group:"dl", level:2, link:"deep-learning/diffusion-models.html", desc:"Generate data by learning to reverse a gradual noising process — SOTA for image synthesis."},
  {id:"Graph Neural Networks", group:"dl", level:2, link:"deep-learning/graph-neural-networks.html", desc:"Deep learning on graphs via neighbor message passing."},

  // NLP
  {id:"Tokenization", group:"nlp", level:2, link:"nlp/tokenization.html", desc:"Splitting text into sub-word units that models can consume."},
  {id:"Embeddings", group:"nlp", level:2, link:"nlp/embeddings.html", desc:"Dense vector representations where semantic similarity becomes geometric proximity."},
  {id:"Language Models", group:"nlp", level:2, link:"nlp/language-models.html", desc:"Models that predict the next token; scaled up, they become LLMs."},
  {id:"RAG", group:"nlp", level:2, link:"nlp/retrieval-augmented-generation.html", desc:"Retrieval-Augmented Generation — grounding LLM answers in retrieved documents."},
  {id:"BERT", group:"models", level:2, link:"models/bert.html", desc:"Bidirectional encoder Transformer pretrained with masked language modeling — the pretrain-then-finetune workhorse for NLP understanding tasks."},

  // Computer Vision
  {id:"Image Classification", group:"cv", level:2, desc:"Assigning a label to a whole image — the canonical vision task."},
  {id:"Object Detection", group:"cv", level:2, desc:"Locating and classifying multiple objects within an image (YOLO, Faster R-CNN)."},

  // MLOps
  {id:"Model Deployment", group:"mlops", level:2, desc:"Serving models behind APIs, batch jobs, or on-device."},
  {id:"Monitoring & Drift", group:"mlops", level:2, link:"mlops/monitoring-and-drift.html", desc:"Tracking live performance and detecting when data shifts away from training."},
  {id:"CI/CD", group:"mlops", level:2, desc:"Automated testing and deployment pipelines for code and models."},
  {id:"Experiment Tracking", group:"mlops", level:2, desc:"Logging runs, params, and metrics for reproducibility (MLflow, W&B)."},

  // Viz
  {id:"Dashboards", group:"viz", level:2, desc:"Interactive views that let stakeholders monitor metrics over time."},
  {id:"Data Storytelling", group:"viz", level:2, desc:"Framing analysis as a clear narrative aimed at a decision."},

  // LLMs & Generative AI
  {id:"Large Language Models", group:"llm", level:2, link:"llm/large-language-models.html", desc:"Transformer decoders scaled up on massive text via next-token prediction — yielding emergent in-context learning, instruction following, and reasoning."},
  {id:"Context Engineering", group:"llm", level:2, link:"llm/context-engineering.html", desc:"Designing what goes into the context window: prompting, few-shot examples, retrieved context, tools, and memory — feeding the model the right information at inference."},
  {id:"Parameter-Efficient Fine-Tuning", group:"llm", level:2, link:"llm/parameter-efficient-fine-tuning.html", desc:"Adapt a frozen pretrained model by training a tiny set of extra weights — LoRA, adapters, prefix/prompt tuning — instead of all parameters."},
  {id:"Policy / Preference Optimization", group:"llm", level:2, link:"llm/policy-preference-optimization.html", desc:"Align models to human preferences: RLHF (PPO) and direct methods (DPO, GRPO) that optimize a policy against a reward or preference signal."},
  {id:"LLM-as-a-Judge", group:"llm", level:2, link:"llm/llm-as-a-judge.html", desc:"Use a strong LLM to score or compare model outputs — scalable automated evaluation (autoraters), with care for bias, position effects, and calibration."},
  {id:"Diffusion LLMs", group:"llm", level:2, link:"llm/diffusion-llms.html", desc:"Generate text by iteratively denoising over discrete tokens rather than left-to-right — discrete/diffusion language models that can decode in parallel."},
  {id:"Vision-Language Models", group:"llm", level:2, link:"llm/vision-language-models.html", desc:"Multimodal models that jointly process images and text — a visual encoder bolted to an LLM for captioning, VQA, and grounded reasoning."},

  // Agentic AI
  {id:"Agents", group:"agent", level:2, link:"agentic/agents.html", desc:"LLMs that plan, call tools, and act in loops to accomplish goals — the move from single-shot answers to multi-step autonomy."},
  {id:"Agentic RL", group:"agent", level:2, link:"agentic/agentic-rl.html", desc:"Training agents with reinforcement learning over multi-step tool use and environment feedback — optimizing reward over trajectories, not single responses."},
  {id:"Agent Skills", group:"agent", level:2, link:"agentic/agent-skills.html", desc:"Reusable, composable capabilities an agent can invoke — packaged tools and procedures that extend what an agent can do."},
  {id:"Agentic Design Patterns", group:"agent", level:2, link:"agentic/agentic-design-patterns.html", desc:"Recurring architectures for agent systems — reflection, planning, tool use, and multi-agent collaboration."},

  // ===== curated expansion (substantial topics; micro-concepts live as sections of these) =====
  // Machine Learning
  {id:"Bias-Variance Tradeoff", group:"ml", level:2, link:"machine-learning/bias-variance-tradeoff.html", desc:"Why models under- or over-fit: test error splits into bias (too simple) and variance (too sensitive) — the classic U-curve and the modern double-descent twist."},
  {id:"Learning Paradigms", group:"ml", level:2, link:"machine-learning/learning-paradigms.html", desc:"The families of learning: supervised, unsupervised, self-supervised, semi-supervised, and reinforcement — what signal each learns from."},
  {id:"ML Algorithms Compared", group:"ml", level:2, link:"machine-learning/ml-algorithms-compared.html", desc:"A side-by-side of the classical algorithms — assumptions, decision boundaries, strengths, and when to reach for each."},
  {id:"ML Strategy", group:"ml", level:2, link:"machine-learning/ml-strategy.html", desc:"Andrew Ng's applied playbook (Machine Learning Yearning) for deciding what to work on next — dev/test set design, single-number metrics, error analysis, the avoidable-bias/variance/data-mismatch decomposition, learning curves, human-level performance, and the optimization verification test."},
  // Deep Learning
  {id:"Neural Network Training", group:"dl", level:2, link:"deep-learning/neural-network-training.html", desc:"How nets actually learn: backpropagation, loss functions, activation functions, weight initialization (Xavier), normalization (BatchNorm), dropout, and the batching/padding tricks that make it work."},
  {id:"Fine-Tuning & Transfer Learning", group:"dl", level:2, link:"deep-learning/fine-tuning-transfer-learning.html", desc:"Adapt a pretrained model to a new task — full fine-tuning vs feature extraction, catastrophic forgetting, and worked examples like BERT."},
  {id:"Knowledge Distillation", group:"dl", level:2, link:"deep-learning/knowledge-distillation.html", desc:"Train a small student to mimic a large teacher — compressing a model's behaviour into far fewer parameters."},
  {id:"Distributed Training", group:"dl", level:2, link:"deep-learning/distributed-training.html", desc:"Train models too big for one device: data, tensor, and pipeline parallelism, ZeRO sharding, plus gradient accumulation and checkpointing."},
  {id:"Encoder vs Decoder Models", group:"dl", level:2, link:"deep-learning/encoder-decoder-models.html", desc:"Three Transformer wirings — encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5) — and which task each suits."},
  {id:"State Space Models", group:"dl", level:2, link:"deep-learning/state-space-models.html", desc:"Sequence models (S4, Mamba) that scale linearly with length via a learned recurrence — a sub-quadratic alternative to attention."},
  {id:"World Models & JEPA", group:"dl", level:2, link:"deep-learning/world-models-jepa.html", desc:"Models that learn predictive representations of how the world evolves — including JEPA, predicting in latent space rather than pixels."},
  {id:"DL Architectures Compared", group:"dl", level:2, link:"deep-learning/dl-architectures-compared.html", desc:"A map of the architecture zoo — MLPs, CNNs, RNNs, Transformers, SSMs — by inductive bias, cost, and the data each fits."},
  {id:"End-to-End Deep Learning", group:"dl", level:2, link:"deep-learning/end-to-end-learning.html", desc:"Replace a hand-engineered pipeline with a single network mapping raw input to output — when abundant (input→output) data makes it win, when a multi-stage pipeline is better, and how to choose pipeline components by data availability and task simplicity."},
  // NLP
  {id:"Text Preprocessing", group:"nlp", level:2, link:"nlp/text-preprocessing.html", desc:"Getting raw text model-ready: cleaning, normalization, casing, stop-words, stemming/lemmatization, and where modern sub-word pipelines differ."},
  {id:"Named Entity Recognition", group:"nlp", level:2, link:"nlp/named-entity-recognition.html", desc:"Tagging spans of text as entities (people, places, orgs) — sequence labeling with BIO schemes."},
  {id:"Machine Translation", group:"nlp", level:2, link:"nlp/machine-translation.html", desc:"Mapping text between languages — from alignment-based statistical MT to attention-based neural MT and multilingual LLMs."},
  {id:"Textual Entailment", group:"nlp", level:2, link:"nlp/textual-entailment.html", desc:"Natural Language Inference: does a premise entail, contradict, or stay neutral toward a hypothesis?"},
  {id:"Document Intelligence", group:"nlp", level:2, link:"nlp/document-intelligence.html", desc:"Extracting structure and answers from documents — layout, tables, and figures in PDFs and scans, often multimodal."},
  {id:"Knowledge Graphs", group:"nlp", level:2, link:"nlp/knowledge-graphs.html", desc:"Representing facts as entities and typed relations — construction, embeddings, and use as structured grounding for retrieval and reasoning."},
  // Data Engineering
  {id:"Data Preprocessing", group:"deng", level:2, link:"data-engineering/data-preprocessing.html", desc:"Turning raw features model-ready: standardization vs normalization, scaling, encoding categoricals, and handling missing values."},
  {id:"Data Sampling & Imbalance", group:"deng", level:2, link:"data-engineering/data-sampling-imbalance.html", desc:"Building representative, balanced training sets — sampling strategies, class imbalance, over/under-sampling and SMOTE."},
  {id:"Data Quality & Governance", group:"deng", level:2, link:"data-engineering/data-quality-governance.html", desc:"Trustworthy data: filtering and deduplication, label quality and inter-annotator agreement, and PII handling/privacy."},
  {id:"Vector Databases (HNSW)", group:"dsys", level:2, link:"databases/vector-databases.html", desc:"The retrieval layer for semantic search and RAG — approximate nearest-neighbor (ANN) search over embedding vectors. HNSW (Hierarchical Navigable Small World) layered proximity graphs, the M/efConstruction/efSearch knobs, recall↔speed↔memory trade-offs, and how it compares to flat and IVF indexes."},
  // MLOps
  {id:"LLMOps", group:"mlops", level:2, link:"mlops/llmops.html", desc:"Operating LLM apps in production — prompt/version management, evaluation, guardrails, caching, and cost/latency monitoring."},
  {id:"Model Acceleration", group:"mlops", level:2, link:"mlops/model-acceleration.html", desc:"Serve models faster and cheaper — quantization, pruning, compilation, and speculative decoding."},
  {id:"Speculative Decoding", group:"mlops", level:2, link:"mlops/speculative-decoding.html", desc:"Speed up autoregressive decoding losslessly: a small draft model proposes several tokens that the large target model verifies in one parallel pass, with a modified rejection-sampling rule that provably preserves the target's output distribution. Covers the acceptance-rate/speedup math and variants (self-speculative, Medusa, EAGLE, lookahead)."},
  // LLMs & Generative AI
  {id:"Reasoning in LLMs", group:"llm", level:2, link:"llm/reasoning-in-llms.html", desc:"Eliciting multi-step thinking — chain-of-thought, self-consistency, tree/graph-of-thought, and reasoning-trained models."},
  {id:"Hallucination & Factuality", group:"llm", level:2, link:"llm/hallucination-factuality.html", desc:"Why LLMs fabricate and how to catch it — measuring factuality, detecting hallucinations, and grounding/abstention mitigations."},
  {id:"Context-Length Extension", group:"llm", level:2, link:"llm/context-length-extension.html", desc:"Pushing the context window longer — positional interpolation, RoPE scaling, and efficient long-context attention."},
  {id:"GPT", group:"models", level:2, link:"models/gpt.html", desc:"Generative Pre-trained Transformer — the decoder-only autoregressive lineage (GPT-1→4) behind modern LLMs; the generative counterpart to BERT."},
  {id:"CLIP", group:"models", level:2, link:"models/clip.html", desc:"Contrastive Language-Image Pre-training — dual image/text encoders aligned in a shared embedding space; zero-shot vision and the backbone of text-to-image models."},
  {id:"LayoutLM", group:"models", level:2, link:"models/layoutlm.html", desc:"The Document-AI family (v1→v2→v3) that fuses text, 2D layout (OCR bounding boxes), and image in one Transformer for visually-rich document understanding — forms, receipts, invoices, doc classification, and document VQA."},
  {id:"Donut", group:"models", level:2, link:"models/donut.html", desc:"OCR-free Document Understanding Transformer — a Swin vision encoder + BART-style decoder that reads document pixels and generates structured output (JSON) directly, no OCR. Pretrained with SynthDoG; DONUT-hole distills/prunes it for the edge."},

  // ===== aman-coverage additions (curated gaps) =====
  {id:"Speech & Audio", group:"speech", level:1, desc:"Processing and generating audio and speech — features, recognition, and synthesis across the audio modality."},
  {id:"Speech Processing", group:"speech", level:2, link:"speech/speech-processing.html", desc:"Turning audio into text and back — spectrogram features, automatic speech recognition (ASR), text-to-speech (TTS), and audio representation learning."},
  {id:"Hyperparameter Tuning", group:"ml", level:2, link:"machine-learning/hyperparameter-tuning.html", desc:"Searching for good settings (learning rate, depth, regularization) — grid/random search, Bayesian optimization, and early-stopping schemes like Hyperband."},
  {id:"Residual / Skip Connections", group:"dl", level:2, link:"deep-learning/residual-skip-connections.html", desc:"Add a layer's input to its output (x + F(x)) so gradients flow through very deep nets — the ResNet idea that unlocked depth."},
  {id:"Hidden Markov Models & CRFs", group:"nlp", level:2, link:"nlp/hidden-markov-models-crfs.html", desc:"Probabilistic sequence models for labeling — HMMs, MEMMs, and Conditional Random Fields that score whole label sequences (classic NER/tagging)."},
  {id:"Text Classification", group:"nlp", level:2, link:"nlp/text-classification.html", desc:"Assigning labels to text — binary/multi-class/multi-label schemes, BoW/TF-IDF and fine-tuned-encoder pipelines, sentiment & aspect-based sentiment, class imbalance, and precision/recall/F1 evaluation."},
  {id:"Text Summarization", group:"nlp", level:2, link:"nlp/text-summarization.html", desc:"Compressing documents — extractive (TextRank/centroid sentence ranking) vs abstractive (seq2seq/LLM) summaries, faithfulness & hallucination, and ROUGE evaluation."},
  {id:"Question Answering", group:"nlp", level:2, link:"nlp/question-answering.html", desc:"Answering questions over text — extractive span prediction (SQuAD), open-domain retriever-reader, generative/closed-book QA, answerability/abstention, and EM/F1 evaluation."},
  {id:"NLP Evaluation Metrics", group:"nlp", level:2, link:"nlp/evaluation-metrics.html", desc:"The canonical hub for scoring generated text — n-gram overlap (BLEU, ROUGE, METEOR, chrF), embedding-based (BERTScore, MoverScore), perplexity, and LLM-as-a-Judge with its failure modes."},
  {id:"Computer Control", group:"agent", level:2, link:"agentic/computer-control.html", desc:"Agents that operate a computer — clicking, typing, and navigating GUIs and browsers from screenshots to complete real-world tasks."},
  {id:"A/B Testing", group:"mlops", level:2, link:"mlops/ab-testing.html", desc:"Online evaluation by randomized experiment — split traffic, compare metrics, and test for statistically significant lift."},
  {id:"Federated Learning", group:"mlops", level:2, link:"mlops/federated-learning.html", desc:"Train across many devices or silos without centralizing data — local updates aggregated into a shared model, for privacy and edge settings."},
  {id:"Differential Privacy", group:"deng", level:2, link:"data-engineering/differential-privacy.html", desc:"A formal privacy guarantee — add calibrated noise so outputs barely change when any single record is added or removed, bounding information leakage."},
  {id:"Contrastive Learning", group:"dl", level:2, link:"deep-learning/contrastive-learning.html", desc:"Learn representations by pulling positive pairs together and pushing negatives — especially hard negatives — apart. The InfoNCE/NT-Xent objective behind SimCLR, MoCo, and CLIP-style multimodal embeddings."},
  {id:"Vision Transformers", group:"cv", level:2, link:"vision/vision-transformers.html", desc:"Apply the Transformer directly to images: split a picture into fixed-size patches, linearly embed them as a token sequence (plus a [CLS] token and positional embeddings), and run a standard encoder. At scale ViT matches or beats CNNs — the backbone behind CLIP, modern detectors/segmenters, and most vision-language models."},
];

const tree = {
  "Math & Statistics":["Linear Algebra","Probability","Statistics","Calculus","Optimization"],
  // Foundations — folded into the main tree so they render like any other domain
  "Programming & Tools":["Python","SQL","Pandas & NumPy","Git & Version Control","R","Apache Spark"],
  "Data Structures & Algorithms":["Arrays & Strings","Trees & Graphs","Sorting & Searching","Dynamic Programming","Big-O Complexity"],
  "Data Systems":["Relational & SQL","NoSQL","Indexing","Transactions (ACID)","Sharding & Replication","Vector Databases (HNSW)","Hadoop & MapReduce","Streaming (Kafka)","Data Lakes","Distributed Storage","Warehouses & Lakehouses","Scalability","Load Balancing","Caching","Message Queues","CAP Theorem"],
  "Data Engineering":["ETL Pipelines","Data Warehousing","Feature Stores","Stream Processing","Data Preprocessing","Data Sampling & Imbalance","Data Quality & Governance","Differential Privacy"],
  "Machine Learning":["Linear & Logistic Regression","k-Nearest Neighbors","Clustering (k-Means)","Dimensionality Reduction (PCA)","Support Vector Machines","Naive Bayes","Decision Trees & Ensembles","Regularization","Reinforcement Learning","Model Evaluation","Bias-Variance Tradeoff","Learning Paradigms","ML Algorithms Compared","Hyperparameter Tuning","ML Strategy"],
  "Deep Learning":["Feedforward / MLP","CNNs","RNNs & LSTMs","Attention","Transformers","Mixture of Experts","GANs","Diffusion Models","Graph Neural Networks","Neural Network Training","Fine-Tuning & Transfer Learning","Knowledge Distillation","Distributed Training","Encoder vs Decoder Models","State Space Models","World Models & JEPA","DL Architectures Compared","Residual / Skip Connections","Contrastive Learning","End-to-End Deep Learning"],
  "Computer Vision":["Classical Vision","Vision Transformers","Image Classification","Object Detection"],
  "NLP":["Tokenization","Embeddings","Language Models","RAG","Text Preprocessing","Named Entity Recognition","Machine Translation","Textual Entailment","Document Intelligence","Knowledge Graphs","Hidden Markov Models & CRFs","Text Classification","Text Summarization","Question Answering","NLP Evaluation Metrics"],
  "MLOps & Deployment":["Model Deployment","Monitoring & Drift","CI/CD","Experiment Tracking","LLMOps","Model Acceleration","Speculative Decoding","A/B Testing","Federated Learning"],
  "Data Viz & Communication":["Dashboards","Data Storytelling"],
  "LLMs & Generative AI":["Large Language Models","Context Engineering","Parameter-Efficient Fine-Tuning","Policy / Preference Optimization","LLM-as-a-Judge","Diffusion LLMs","Vision-Language Models","Reasoning in LLMs","Hallucination & Factuality","Context-Length Extension"],
  "Agentic AI":["Agents","Agentic RL","Agent Skills","Agentic Design Patterns","Computer Control"],
  "Speech & Audio":["Speech Processing"],
  "Model Architectures":["BERT","GPT","CLIP","LayoutLM","Donut"],
};

/* level-3 sub-topics: parent topic id → ordered child ids. The parent keeps its
   page as the hub; children are rendered as leaves hanging off the parent in the
   graph and as a "Sub-topics" / sibling list in the sidebars. */
const subtree = {
  "Linear & Logistic Regression":["Linear Regression","Logistic Regression"],
  "Pandas & NumPy":["NumPy","Pandas"],
  "Linear Algebra":["Vectors & Vector Spaces","Matrices & Rank","Systems, Determinants & Inverses","Projections & Least-Squares","Eigendecomposition & SVD","Quadratic Forms, Covariance & PCA"],
  "Classical Vision":["Image Formation","Image Processing","Frequency Domain","Features & Matching","Segmentation","Alignment & Motion","Stereo & Structure from Motion"],
  "Statistics":["Descriptive Statistics","Producing Data & Sampling","Sampling Distributions & CLT","Estimation & Confidence Intervals","Hypothesis Testing","Regression & Correlation","Resampling (Bootstrap & Permutation)","Categorical Data & Chi-Square","ANOVA & Multiple Comparisons"],
};

/* cross edges. 3rd element "isa" = hierarchy bridge (is-a / builds-on),
   oriented child → parent; rendered as a dotted arrow in the graph. */
const cross = [
  ["Classical Vision","CNNs"],["Classical Vision","Vision Transformers"],["Classical Vision","Image Classification"],
  ["Image Formation","Matrices & Rank"],["Image Formation","Systems, Determinants & Inverses"],
  ["Image Processing","CNNs"],["Image Processing","Image Classification"],
  ["Frequency Domain","Calculus"],["Frequency Domain","Linear Algebra"],
  ["Features & Matching","Object Detection"],["Features & Matching","CNNs"],
  ["Segmentation","Clustering (k-Means)"],["Segmentation","Object Detection"],
  ["Alignment & Motion","Optimization"],
  ["Stereo & Structure from Motion","Eigendecomposition & SVD"],["Stereo & Structure from Motion","Optimization"],
  /* Foundations ↔ Data Science commonalities (separate clusters, linked where they overlap) */
  ["Programming & Tools","Machine Learning"],["Programming & Tools","Data Engineering"],
  ["Pandas & NumPy","Linear Algebra"],["Pandas & NumPy","SQL"],["Pandas & NumPy","Apache Spark"],
  ["SQL","Data Engineering"],["Data Structures & Algorithms","Machine Learning"],["Data Systems","Data Engineering"],
  ["Apache Spark","Data Systems"],["Data Systems","Distributed Training"],
  ["Data Systems","MLOps & Deployment"],["Data Systems","LLMOps"],
  ["Regularization","Linear & Logistic Regression"],["Regularization","Feedforward / MLP"],
  ["Regularization","Optimization"],
  ["Linear Algebra","Feedforward / MLP"],["Optimization","Feedforward / MLP"],
  ["Calculus","Optimization"],["Calculus","Linear Algebra"],["Calculus","Neural Network Training"],["Calculus","Probability"],
  ["Linear Algebra","Optimization"],["Probability","Naive Bayes"],
  ["Statistics","Model Evaluation"],["Probability","Statistics"],
  ["BERT","GPT"],["BERT","LayoutLM"],["LayoutLM","Donut"],
  /* Statistics sub-topics ↔ the rest of the graph */
  ["Descriptive Statistics","Data Preprocessing"],
  ["Producing Data & Sampling","Data Sampling & Imbalance"],["Producing Data & Sampling","A/B Testing"],
  ["Sampling Distributions & CLT","Probability"],["Sampling Distributions & CLT","Model Evaluation"],["Sampling Distributions & CLT","A/B Testing"],["Sampling Distributions & CLT","Bias-Variance Tradeoff"],
  ["Estimation & Confidence Intervals","Model Evaluation"],["Estimation & Confidence Intervals","Regularization"],["Estimation & Confidence Intervals","Bias-Variance Tradeoff"],
  ["Hypothesis Testing","A/B Testing"],["Hypothesis Testing","Model Evaluation"],
  ["Regression & Correlation","Linear Regression"],["Regression & Correlation","Projections & Least-Squares"],
  ["Resampling (Bootstrap & Permutation)","Model Evaluation"],["Resampling (Bootstrap & Permutation)","Decision Trees & Ensembles"],
  ["Categorical Data & Chi-Square","Naive Bayes"],
  ["Categorical Data & Chi-Square","Model Evaluation"],["Categorical Data & Chi-Square","Logistic Regression"],
  ["Categorical Data & Chi-Square","A/B Testing"],["Categorical Data & Chi-Square","Resampling (Bootstrap & Permutation)"],
  ["Categorical Data & Chi-Square","Producing Data & Sampling"],["Categorical Data & Chi-Square","Hypothesis Testing"],
  ["ANOVA & Multiple Comparisons","Hyperparameter Tuning"],
  ["ANOVA & Multiple Comparisons","Model Evaluation"],["ANOVA & Multiple Comparisons","A/B Testing"],
  ["ANOVA & Multiple Comparisons","Regression & Correlation"],["ANOVA & Multiple Comparisons","Resampling (Bootstrap & Permutation)"],
  ["ANOVA & Multiple Comparisons","Categorical Data & Chi-Square"],["ANOVA & Multiple Comparisons","Hypothesis Testing"],
  ["Embeddings","Transformers"],["Transformers","Language Models"],
  ["Attention","Transformers"],["Attention","RNNs & LSTMs"],["Attention","Language Models"],
  ["Mixture of Experts","Transformers"],["Mixture of Experts","Feedforward / MLP"],["Mixture of Experts","Language Models"],
  ["Tokenization","Language Models"],["RAG","Embeddings"],
  ["CNNs","Image Classification"],["Vision Transformers","Transformers"],
  ["Vision Transformers","CLIP"],["Vision Transformers","Contrastive Learning"],["Vision Transformers","Image Classification"],
  ["Optimization","Linear & Logistic Regression"],
  ["k-Nearest Neighbors","Clustering (k-Means)"],
  ["Linear Algebra","Support Vector Machines"],["Linear Algebra","Linear & Logistic Regression"],
  ["Feature Stores","Model Deployment"],["Python","Pandas & NumPy"],
  ["Git & Version Control","Python"],["Git & Version Control","Trees & Graphs"],
  ["Git & Version Control","Sorting & Searching"],["Git & Version Control","CI/CD"],
  ["Git & Version Control","Experiment Tracking"],["Git & Version Control","Data Quality & Governance"],
  ["Git & Version Control","Model Deployment"],
  ["Python","Data Structures & Algorithms"],["Python","Big-O Complexity"],
  ["SQL","Data Warehousing"],["SQL","Descriptive Statistics"],["SQL","Apache Spark"],["SQL","ETL Pipelines"],["Apache Spark","Stream Processing"],
  ["Deep Learning","Machine Learning","isa"],["Language Models","RAG"],
  ["Object Detection","CNNs"],["Model Evaluation","Monitoring & Drift"],
  ["Model Evaluation","Linear & Logistic Regression"],["Model Evaluation","Decision Trees & Ensembles"],
  ["Descriptive Statistics","Statistics"],["Descriptive Statistics","Pandas & NumPy"],["Dashboards","Monitoring & Drift"],
  ["Linear Algebra","Embeddings"],["CNNs","Vision Transformers"],
  ["NLP","Deep Learning","isa"],
  ["Linear Algebra","Dimensionality Reduction (PCA)"],["Clustering (k-Means)","Dimensionality Reduction (PCA)"],
  ["Dimensionality Reduction (PCA)","Logistic Regression"],["Dimensionality Reduction (PCA)","Naive Bayes"],
  ["Reinforcement Learning","Feedforward / MLP"],["Reinforcement Learning","Optimization"],

  // Data Structures & Algorithms ties
  ["Data Structures & Algorithms","Big-O Complexity"],["Data Structures & Algorithms","Dynamic Programming"],
  ["Big-O Complexity","Arrays & Strings"],["Big-O Complexity","Trees & Graphs"],
  ["Big-O Complexity","Sorting & Searching"],["Big-O Complexity","Dynamic Programming"],
  ["Big-O Complexity","Model Acceleration"],["Big-O Complexity","Optimization"],
  ["Trees & Graphs","Graph Neural Networks"],
  ["Dynamic Programming","Reinforcement Learning"],
  ["Sorting & Searching","Vector Databases (HNSW)"],["Sorting & Searching","Indexing"],
  ["Arrays & Strings","Pandas & NumPy"],

  // LLMs & Generative AI ties
  ["Large Language Models","Transformers"],["Large Language Models","Language Models"],["Large Language Models","Mixture of Experts"],
  ["Context Engineering","RAG"],["Context Engineering","Large Language Models"],
  ["Parameter-Efficient Fine-Tuning","Large Language Models"],["Parameter-Efficient Fine-Tuning","Regularization"],
  ["Policy / Preference Optimization","Reinforcement Learning"],["Policy / Preference Optimization","Large Language Models"],
  ["LLM-as-a-Judge","Model Evaluation"],["LLM-as-a-Judge","Large Language Models"],
  ["Diffusion LLMs","Diffusion Models"],["Diffusion LLMs","Large Language Models"],
  ["Vision-Language Models","Vision Transformers"],["Vision-Language Models","Large Language Models"],["Vision-Language Models","Transformers"],

  // Agentic AI ties
  ["Agents","Large Language Models"],["Agents","RAG"],["Agents","Context Engineering"],
  ["Agentic RL","Reinforcement Learning"],["Agentic RL","Agents"],["Agentic RL","Policy / Preference Optimization"],
  ["Agent Skills","Agents"],["Agent Skills","Context Engineering"],["Agent Skills","Computer Control"],
  ["Agentic Design Patterns","Agents"],["Agentic Design Patterns","Reasoning in LLMs"],["Agentic Design Patterns","Agentic RL"],

  // domain bridges (builds-on)
  ["LLMs & Generative AI","Deep Learning","isa"],["Agentic AI","LLMs & Generative AI","isa"],
  ["Speech & Audio","Deep Learning","isa"],
  // aman-coverage ties
  ["Hyperparameter Tuning","Model Evaluation"],["Hyperparameter Tuning","Neural Network Training"],["Hyperparameter Tuning","Bias-Variance Tradeoff"],
  ["Residual / Skip Connections","Neural Network Training"],["Residual / Skip Connections","CNNs"],["Residual / Skip Connections","Transformers"],
  ["Hidden Markov Models & CRFs","Named Entity Recognition"],["Hidden Markov Models & CRFs","Naive Bayes"],
  ["Computer Control","Agents"],["Computer Control","Vision-Language Models"],
  ["A/B Testing","Model Evaluation"],["A/B Testing","Monitoring & Drift"],
  ["Monitoring & Drift","Model Deployment"],["Monitoring & Drift","Experiment Tracking"],
  ["Federated Learning","Distributed Training"],["Federated Learning","Differential Privacy"],
  ["Differential Privacy","Data Quality & Governance"],
  ["Speech Processing","Transformers"],["Speech Processing","Embeddings"],["Speech Processing","Vision-Language Models"],
  ["BERT","Encoder vs Decoder Models"],["BERT","Transformers"],["BERT","Fine-Tuning & Transfer Learning"],["BERT","Language Models"],["BERT","Named Entity Recognition"],["BERT","Textual Entailment"],
  ["GPT","Large Language Models"],["GPT","Transformers"],["GPT","Encoder vs Decoder Models"],["GPT","Policy / Preference Optimization"],["GPT","Attention"],
  ["CLIP","Vision-Language Models"],["CLIP","Embeddings"],["CLIP","Diffusion Models"],["CLIP","Transformers"],
  ["LayoutLM","Document Intelligence"],["LayoutLM","Transformers"],["LayoutLM","Vision Transformers"],
  ["Donut","Document Intelligence"],["Donut","Encoder vs Decoder Models"],["Donut","End-to-End Deep Learning"],["Donut","Knowledge Distillation"],["Donut","Attention"],
  ["Vector Databases (HNSW)","Embeddings"],["Vector Databases (HNSW)","RAG"],["Vector Databases (HNSW)","k-Nearest Neighbors"],["Vector Databases (HNSW)","Indexing"],["Vector Databases (HNSW)","NoSQL"],
  // wire in previously-isolated built DL pages
  ["GANs","Diffusion Models"],["GANs","Neural Network Training"],["GANs","CLIP"],
  ["Graph Neural Networks","Embeddings"],["Graph Neural Networks","Knowledge Graphs"],["Graph Neural Networks","Neural Network Training"],

  // curated-expansion ties
  ["Bias-Variance Tradeoff","Model Evaluation"],["Bias-Variance Tradeoff","Regularization"],["Bias-Variance Tradeoff","Decision Trees & Ensembles"],
  ["Bias-Variance Tradeoff","Linear & Logistic Regression"],["Data Preprocessing","Linear & Logistic Regression"],
  ["ML Strategy","Bias-Variance Tradeoff"],["ML Strategy","Model Evaluation"],["ML Strategy","Hyperparameter Tuning"],["ML Strategy","Reinforcement Learning"],["ML Strategy","End-to-End Deep Learning"],
  ["End-to-End Deep Learning","Machine Translation"],["End-to-End Deep Learning","Speech Processing"],["End-to-End Deep Learning","Transformers"],
  ["Learning Paradigms","Reinforcement Learning"],["Learning Paradigms","Clustering (k-Means)"],
  ["ML Algorithms Compared","Decision Trees & Ensembles"],["ML Algorithms Compared","Support Vector Machines"],["ML Algorithms Compared","Linear & Logistic Regression"],
  ["Neural Network Training","Feedforward / MLP"],["Neural Network Training","Optimization"],["Neural Network Training","Regularization"],
  ["Fine-Tuning & Transfer Learning","Large Language Models"],["Fine-Tuning & Transfer Learning","Parameter-Efficient Fine-Tuning"],
  ["Knowledge Distillation","Model Acceleration"],["Knowledge Distillation","Neural Network Training"],
  ["Distributed Training","Neural Network Training"],["Distributed Training","Large Language Models"],
  ["Encoder vs Decoder Models","Transformers"],["Encoder vs Decoder Models","Language Models"],
  ["State Space Models","Transformers"],["State Space Models","RNNs & LSTMs"],
  ["RNNs & LSTMs","Machine Translation"],
  ["World Models & JEPA","Reinforcement Learning"],["World Models & JEPA","Feedforward / MLP"],
  ["DL Architectures Compared","Transformers"],["DL Architectures Compared","CNNs"],["DL Architectures Compared","State Space Models"],
  ["Text Preprocessing","Tokenization"],["Text Preprocessing","Embeddings"],["Text Preprocessing","Named Entity Recognition"],["Text Preprocessing","Language Models"],
  ["Named Entity Recognition","Language Models"],["Named Entity Recognition","Knowledge Graphs"],
  ["Machine Translation","Transformers"],["Machine Translation","Language Models"],
  ["Textual Entailment","Language Models"],["Textual Entailment","Large Language Models"],["Textual Entailment","Hallucination & Factuality"],["Textual Entailment","LLM-as-a-Judge"],
  ["Document Intelligence","Vision-Language Models"],
  ["Knowledge Graphs","Embeddings"],["Knowledge Graphs","RAG"],
  ["Data Preprocessing","Pandas & NumPy"],["Data Preprocessing","Feature Stores"],
  ["Data Sampling & Imbalance","Model Evaluation"],["Data Sampling & Imbalance","Data Preprocessing"],["Data Sampling & Imbalance","Bias-Variance Tradeoff"],
  ["Data Quality & Governance","ETL Pipelines"],["Data Quality & Governance","Monitoring & Drift"],
  ["LLMOps","Model Deployment"],["LLMOps","Large Language Models"],["LLMOps","LLM-as-a-Judge"],
  ["LLMOps","Monitoring & Drift"],["LLMOps","Model Acceleration"],["LLMOps","RAG"],["LLMOps","A/B Testing"],["LLMOps","Context Engineering"],
  ["Model Acceleration","Model Deployment"],["Model Acceleration","Mixture of Experts"],
  ["Speculative Decoding","Model Acceleration"],["Speculative Decoding","Large Language Models"],["Speculative Decoding","Knowledge Distillation"],["Speculative Decoding","LLMOps"],["Speculative Decoding","Attention"],
  ["Reasoning in LLMs","Large Language Models"],["Reasoning in LLMs","Policy / Preference Optimization"],
  ["Hallucination & Factuality","Large Language Models"],["Hallucination & Factuality","RAG"],["Hallucination & Factuality","LLM-as-a-Judge"],
  ["Context-Length Extension","Large Language Models"],["Context-Length Extension","Attention"],

  // Linear Algebra sub-topic ties (the hub keeps its domain-level edges above)
  ["Vectors & Vector Spaces","Embeddings"],["Vectors & Vector Spaces","Pandas & NumPy"],["Vectors & Vector Spaces","k-Nearest Neighbors"],
  ["Matrices & Rank","Pandas & NumPy"],["Matrices & Rank","Feedforward / MLP"],
  ["Systems, Determinants & Inverses","Linear & Logistic Regression"],["Systems, Determinants & Inverses","Pandas & NumPy"],["Systems, Determinants & Inverses","Regularization"],
  ["Projections & Least-Squares","Linear & Logistic Regression"],["Projections & Least-Squares","Regularization"],
  ["Eigendecomposition & SVD","Dimensionality Reduction (PCA)"],["Eigendecomposition & SVD","Optimization"],["Eigendecomposition & SVD","Embeddings"],
  ["Quadratic Forms, Covariance & PCA","Dimensionality Reduction (PCA)"],["Quadratic Forms, Covariance & PCA","Optimization"],["Quadratic Forms, Covariance & PCA","Support Vector Machines"],

  // Regression + Pandas/NumPy sub-topic ties (hubs keep their existing edges)
  ["Linear Regression","Projections & Least-Squares"],["Linear Regression","Regularization"],["Linear Regression","Model Evaluation"],["Linear Regression","Bias-Variance Tradeoff"],
  ["Logistic Regression","Model Evaluation"],["Logistic Regression","Optimization"],["Logistic Regression","Naive Bayes"],["Logistic Regression","Feedforward / MLP"],["Logistic Regression","Text Classification"],
  ["NumPy","Matrices & Rank"],["NumPy","Vectors & Vector Spaces"],["NumPy","Arrays & Strings"],["NumPy","Python"],
  ["Pandas","SQL"],["Pandas","Data Preprocessing"],["Pandas","Descriptive Statistics"],["Pandas","Apache Spark"],

  // embeddings-primer ties
  ["Tokenization","Embeddings"],["BERT","Embeddings"],["Embeddings","Dimensionality Reduction (PCA)"],
  ["Contrastive Learning","Embeddings"],["Contrastive Learning","CLIP"],["Contrastive Learning","Vision-Language Models"],
  ["Contrastive Learning","Fine-Tuning & Transfer Learning"],["Contrastive Learning","Learning Paradigms"],
  ["Contrastive Learning","CNNs"],["Contrastive Learning","Neural Network Training"],

  // NLP core-batch ties (text classification, summarization, QA, evaluation hub)
  ["Text Classification","Naive Bayes"],["Text Classification","Embeddings"],["Text Classification","Text Preprocessing"],
  ["Text Classification","BERT"],["Text Classification","Model Evaluation"],["Text Classification","Data Sampling & Imbalance"],
  ["Text Summarization","Encoder vs Decoder Models"],["Text Summarization","Machine Translation"],["Text Summarization","NLP Evaluation Metrics"],
  ["Text Summarization","Hallucination & Factuality"],["Text Summarization","Knowledge Graphs"],["Text Summarization","Embeddings"],
  ["Question Answering","RAG"],["Question Answering","BERT"],["Question Answering","Embeddings"],
  ["Question Answering","Vector Databases (HNSW)"],["Question Answering","Textual Entailment"],["Question Answering","NLP Evaluation Metrics"],
  ["NLP Evaluation Metrics","Machine Translation"],["NLP Evaluation Metrics","Language Models"],
  ["NLP Evaluation Metrics","LLM-as-a-Judge"],["NLP Evaluation Metrics","Model Evaluation"],
];

/* ---- derived (shared by graph + sidebars) ---- */
const meta = {}; nodes.forEach(n => meta[n.id] = n);

/* PARENT: every non-root id → its hierarchy parent (domain → root, topic → domain, sub-topic → topic) */
const PARENT = {};
Object.entries(tree).forEach(([dom, kids]) => { PARENT[dom] = "Data Science"; kids.forEach(k => PARENT[k] = dom); });
Object.entries(subtree).forEach(([top, kids]) => kids.forEach(k => PARENT[k] = top));

const rootData = {
  ...meta["Data Science"],
  children: Object.keys(tree).map(dom => ({
    ...meta[dom],
    children: tree[dom].map(t => ({
      ...meta[t],
      ...(subtree[t] ? { children: subtree[t].map(c => ({...meta[c]})) } : {})
    }))
  }))
};

/* full-graph adjacency (tree + subtree + cross), independent of any collapse state */
const ADJ = {}; nodes.forEach(n => ADJ[n.id] = new Set());
Object.entries(tree).forEach(([dom, kids]) => {
  ADJ["Data Science"].add(dom); ADJ[dom].add("Data Science");
  kids.forEach(k => { ADJ[dom].add(k); ADJ[k].add(dom); });
});
Object.entries(subtree).forEach(([top, kids]) => kids.forEach(k => { if (ADJ[top] && ADJ[k]) { ADJ[top].add(k); ADJ[k].add(top); } }));
cross.forEach(([s, t]) => { if (ADJ[s] && ADJ[t]) { ADJ[s].add(t); ADJ[t].add(s); } });

/* find the node whose link matches the current page path (for sidebars) */
function nodeForPath(path){ return nodes.find(n => n.link && path.endsWith(n.link)); }
