# Research Brief — Pandas & NumPy (in-memory numerical & tabular computing in Python)

## 1. Topic & scope

**In scope:** the two foundational Python libraries for single-machine, in-memory numerical and tabular work, treated together because pandas is built on top of NumPy.
- **NumPy core:** the ndarray as a contiguous typed memory buffer; dtypes; shape/strides; C vs Fortran memory order; views vs copies (and the aliasing gotcha); broadcasting; axis-wise reductions; basic (slicing) vs fancy/advanced indexing; *why* vectorized operations are 10–100× faster than Python loops (compiled C loops, no per-element boxing, SIMD/BLAS).
- **pandas core:** Series & DataFrame on top of NumPy; the Index; dtypes including object/categorical/nullable; selection (loc/iloc/boolean masks); vectorized string & datetime accessors; split-apply-combine groupby; merge/join semantics and concat; reshaping (pivot/melt/stack/unstack); missing data (NaN/NA).
- **Performance & memory intuition:** vectorized O(n)-in-C vs Python-loop overhead, groupby ≈ hashing, merge ≈ hash-join, dtype downcasting and categorical for memory, and the common anti-patterns (iterrows / row-wise apply, chained-indexing SettingWithCopyWarning, the copy-vs-view trap).

**Out of scope:** distributed/out-of-core engines (Spark, Dask, Ray) beyond a one-line "when to graduate" pointer; pure plotting/visualization (belongs to Exploratory Data Analysis); the broader SciPy/scikit-learn stack; GPU array libraries (CuPy); Arrow/Polars internals beyond a comparison note; the Python language itself.

This brief **fills an existing page-less stub** — the `Pandas & NumPy` node already exists in `data.js` (line 62, `group:"prog"`) with no `link:`. It is the **first** page to be authored in the Programming & Tools (`prog`) domain.

## 2. Core concept

NumPy provides the **ndarray**: a fixed-size, multidimensional array of elements that all share **one dtype**, laid out in a single **contiguous block of memory**. All numerical work is expressed as **vectorized** operations on whole arrays, which dispatch to precompiled C (and BLAS) loops instead of Python-level iteration.

pandas builds two labeled structures on that foundation:
- **Series** — a 1-D array of values plus an **Index** (the row labels).
- **DataFrame** — a 2-D table: an ordered set of columns, each a Series, all sharing one Index. Conceptually a dict of equal-length, individually-typed columns.

The Index is the defining pandas feature: it gives **label-based alignment**, so operations between objects line up by label rather than position, and it powers selection, joins, grouping, and reshaping. Together the two libraries cover the "load → clean → reshape → aggregate → join" loop on data that fits in RAM.

## 3. Intuition

**Why an ndarray beats a Python list.** A Python list of numbers is an array of *pointers* to separate, individually-boxed Python objects scattered across the heap; each element carries object overhead and its own type. An ndarray is a flat, typed buffer — e.g. one million float64s sit as 8,000,000 contiguous bytes with no per-element object header. Two consequences follow and they are the whole reason the library exists:
1. **No per-element boxing.** The dtype is known once for the whole buffer, so a loop over it needs no per-item type checks or unboxing.
2. **Cache-friendly, compiled iteration.** Because the data is contiguous, a single compiled C loop (often auto-vectorized to SIMD, or handed to BLAS for linear algebra) sweeps the buffer near-optimally, using the CPU's fast cache. The Python interpreter is touched **once** per array operation, not once per element.

That is why `a + b` on arrays is typically **10–100×** faster than the equivalent Python `for` loop: you pay the interpreter's overhead a single time and let C do the n iterations.

**Strides as the trick behind views.** An array is "data pointer + dtype + shape + **strides**" where strides are the byte steps to move one element along each axis. Reshaping, slicing, and transposing usually just rewrite this *metadata* and point at the **same buffer** — a **view**, created with zero data copying. This is powerful and dangerous: a view aliases the original, so writing through it mutates the original (the aliasing gotcha).

**Broadcasting as virtual stretching.** When shapes differ, NumPy *pretends* the smaller array is repeated to fit — without ever copying it — by treating size-1 (or missing) axes as stride-0. Adding a length-3 row vector to a 4×3 matrix "stretches" the row down all 4 rows for free.

**pandas = NumPy + labels.** Think of a DataFrame as columnar NumPy arrays wearing name tags (the Index and column labels). The labels let you say "join these tables on `user_id`" or "average `revenue` within each `region`" declaratively — the split-apply-combine and relational-join vocabulary of SQL, expressed in Python and executed over vectorized columns.

## 4. Key math & rules (Unicode only)

**Memory layout.** For an ndarray of dtype with itemsize s bytes:
  total bytes = s · ∏ shapeᵢ
  C order (row-major): last axis varies fastest. For shape (m, n), strides = (n·s, s).
  Fortran order (column-major): first axis varies fastest. strides = (s, m·s).

A 4×3 array of 8-byte float64 in C order has strides (24, 8): step 24 bytes to advance one row, 8 bytes to advance one column.

**Element address (the stride formula).** For indices (i₀, i₁, …, i_{d−1}):
  offset = base + Σₖ iₖ · strideₖ
A **view** keeps base/buffer and only changes shape/strides; a **copy** allocates a new buffer.

**Broadcasting rules.** Compare shapes **right to left** (trailing dims first). Two dims are compatible iff:
  dimₐ = dim_b   OR   one of them = 1   (a missing leading dim is treated as 1)
The result dim is max(dimₐ, dim_b); a size-1 dim is virtually stretched (stride 0, no copy).
  Example: (8 × 1 × 6 × 1) with (7 × 1 × 5) → (8 × 7 × 6 × 5)
  Example: (256 × 256 × 3) with (3) → (256 × 256 × 3)
  Incompatible: (3) with (4) → ValueError (trailing dims 3 ≠ 4, neither is 1)

**Axis-wise reduction.** For A of shape (m, n):
  A.sum(axis=0) → shape (n,)   (collapse rows; sum down each column)
  A.sum(axis=1) → shape (m,)   (collapse columns; sum across each row)
Rule of thumb: the named axis is the one that disappears.

**Complexity intuition (neutral, order-of-magnitude).**
  Vectorized elementwise op over n elements: Θ(n) work in compiled C — same asymptotics as a Python loop but a large constant-factor win (≈10–100×) from no boxing + cache + SIMD.
  groupby on a key: ≈ Θ(n) average via **hashing** the keys into buckets.
  merge / join on a key: ≈ Θ(n + m) average via a **hash join** (build a hash table on one side, probe with the other); near Θ(n·m) blowup when keys are highly duplicated (many-to-many Cartesian product).
  Boolean-mask selection: Θ(n) scan producing a boolean array, then a gather.

**NaN semantics.** Missing numeric data is float NaN, which is **not equal to itself**:
  NaN ≠ NaN  ⇒  detect with isna()/notna(), never with `== NaN`.
Most reductions skip NaN by default (skipna=True), e.g. Series([1, NaN, 3]).mean() = 2.

## 5. Practical takeaways & trade-offs

**Vectorize; do not loop rows.** Per-row Python iteration defeats the entire point of the libraries. Reported benchmark on a 1-million-row DataFrame: `iterrows()` ≈ 47 s vs the vectorized column op ≈ 0.03 s — roughly a **1000×** gap on that workload. [number from a single performance write-up; treat as illustrative order-of-magnitude, not a guaranteed constant.] General ladder, fastest first: pure vectorized NumPy/pandas column ops → built-in vectorized methods (.str, .dt, .where, .clip) → `groupby().agg/transform` → `itertuples` → `apply(axis=1)` → `iterrows` (slowest). `apply` with a Python function is essentially a disguised loop and is not vectorized.

**The copy-vs-view trap (NumPy).** Basic slicing returns a **view** (same buffer); fancy/advanced indexing (integer-array or boolean-array indexing) returns a **copy**. Detect with the `.base` attribute: a view's `.base` points at the original, a copy's `.base` is None. Writing into a view silently mutates the parent — call `.copy()` when you need isolation.

**SettingWithCopyWarning (pandas).** Caused by **chained indexing**: `df[mask]["col"] = 0` first selects (which may return a copy), then assigns into that temporary — so the original may be left unchanged, and pandas warns. Fix: a **single** indexing op with `.loc`:
  `df.loc[mask, "col"] = 0`
When you genuinely want a detached subframe, take an explicit `.copy()` first. (In pandas 3.0, Copy-on-Write becomes the default and removes most of this ambiguity — assignments behave predictably and the warning largely goes away.) [pandas 3.0 default behavior — verify against the installed version before asserting on the page.]

**Selection cheat-sheet.**
  `.loc[row_labels, col_labels]` — label-based (endpoint inclusive).
  `.iloc[row_pos, col_pos]` — integer-position-based (endpoint exclusive, like Python slices).
  Boolean mask: `df[df.col > 0]` or `df.loc[df.col > 0, "other"]`.

**Memory levers.**
  - **dtype downcasting:** float64→float32, int64→int32/int16 where range allows — halves or quarters column memory.
  - **categorical dtype:** stores low-cardinality strings (e.g. "region", "status") as small integer codes + a category lookup, often a large memory win and faster groupby/merge on those keys.
  - **object dtype** (the default for arbitrary Python strings) is the memory/perf trap: it is an array of pointers to Python objects — no vectorization benefit. Prefer categorical, the dedicated nullable `string` dtype, or (where available) Arrow-backed dtypes.
  - **nullable dtypes** (Int64, Float64, boolean, string) carry a real missing-value mask, so integers can hold NA without being silently promoted to float.

**When to use / when not to.** Use Pandas & NumPy for data that fits comfortably in RAM (rule of thumb: working set up to a few GB, leaving headroom because intermediates and joins can multiply memory). Graduate to a chunked/columnar/distributed engine (Polars, DuckDB, Dask, or Spark) when the data exceeds memory or a single core becomes the bottleneck — this is the natural bridge to the Apache Spark node.

**Other common anti-patterns.** Repeatedly `concat`/`append` inside a loop (quadratic — build a list, concat once); merging without checking key uniqueness (use `validate=`); forgetting that many-to-many merges explode rows; relying on `==` for NaN; mutating while iterating.

## 6. Suggested interactive viz

The site requires ≥1 live D3 viz; the flagship pages use drag-driven SVG with a live readout (classes `.viz`, `.vhead`, `.vtitle`, `.vhint`, `.readout`). Recommended primary viz, plus optional secondary:

**Primary — "Broadcasting & strides explorer."** Show a small grid (e.g. a 4×3 matrix `A`) and a second operand the user can reshape via buttons/sliders among shapes like `(3,)`, `(4,1)`, `(1,3)`, `(2,)`. The viz:
- Aligns the two shapes **right-to-left** and color-codes each trailing dim pair green (compatible: equal or one is 1) or red (ValueError), printing the resulting broadcast shape live.
- Animates size-1 axes "stretching" (ghosted, stride-0 copies) over `A`, and renders `A + b` cell values in the readout — making "virtual repeat, no data copied" visible.

**Secondary — "View vs copy / stride playground."** Render a 1-D buffer as a row of memory cells. Let the user slice (`x[1:7:2]`) and watch a **view** highlight the *same* cells (with its stride arrow) vs a fancy index (`x[[0,3,5]]`) spawning a *separate* copied buffer. A toggle "write 99 into the selection" shows the parent changing for a view and staying unchanged for a copy — the aliasing gotcha, dramatized.

(A third candidate, if a tabular viz is wanted: a small split-apply-combine animation that visibly partitions colored rows by a key, applies a sum, and recombines — mirroring groupby.)

## 7. Candidate placement

- **Action:** **fill the existing stub** (do not mint a new node). Node already present in `data.js` line 62:
  `{id:"Pandas & NumPy", group:"prog", level:2, desc:"Vectorized arrays and dataframes for in-memory data manipulation."}`
- **group:** `prog` (Programming & Tools) — confirmed; this is the **first authored page** in that domain.
- **Proposed file path:** `programming/pandas-numpy.html` (new `programming/` folder — descriptive-kebab convention matching `machine-learning/`, `deep-learning/`, `data-engineering/`). The ingestor should add `link:"programming/pandas-numpy.html"` to the node.
- **Asset paths:** the page lives **one folder deep**, so reference shared assets with `../` exactly like siblings: `../vendor/d3.min.js`, `../notes.css`, (and `../notes.js` if used). Mirror `math/linear-algebra.html` structure: `<section class="topic" data-nav="…">` blocks, `.lede`, `.notes`, `.formula` (Unicode math), `.callout`, `.cmp` comparison tables, and `.viz` D3 blocks.

**Cross-link candidates** (all real `data.js` ids). Several edges already exist — the ingestor should keep these and add the missing ones rather than duplicating:
- **Python** (`prog`) — already linked (`["Python","Pandas & NumPy"]`); the host language. Keep.
- **Data Preprocessing** (data-engineering) — already linked both directions; primary consumer of pandas. Keep.
- **Exploratory Data Analysis** (data viz) — already linked (`["Exploratory Data Analysis","Pandas & NumPy"]`); EDA is done in pandas. Keep.
- **Arrays & Strings** (`dsa`) — already linked (`["Arrays & Strings","Pandas & NumPy"]`); contiguous indexed storage is the CS analogue of the ndarray. Keep.
- **Linear Algebra** (`math`) — **add**: NumPy is the matrix/vector engine; BLAS-backed `@`/dot directly realizes the linear-algebra page's operations. Strong, currently-missing edge.
- **SQL** (`prog`) — **add**: merge/join and groupby mirror SQL JOIN and GROUP BY; natural sibling within the same domain.
- **Apache Spark** (`prog`) — **add**: the "graduate beyond one machine / out-of-memory" successor; the Spark DataFrame API deliberately echoes pandas.

## 8. Sources consulted (audit trail only — do NOT surface on the published page)

- https://numpy.org/doc/stable/user/basics.broadcasting.html — official NumPy manual; authoritative for the right-to-left broadcasting rules and "no copies of stretched data."
- https://numpy.org/doc/stable/user/basics.copies.html — official NumPy manual; authoritative for views vs copies, strides/dtype metadata, `.base`, basic-slicing-view vs fancy-indexing-copy.
- https://www.nature.com/articles/s41586-020-2649-2 — "Array programming with NumPy," Nature 585:357–362 (2020), DOI 10.1038/s41586-020-2649-2; peer-reviewed primary source on the ndarray memory model, vectorization, strides example, and ecosystem role. (Nature page redirects to an auth wall; content read via the open-access Cambridge institutional-repository mirror below.)
- https://api.repository.cam.ac.uk/server/api/core/bitstreams/d5f3cd8a-00c0-4e1d-85c9-5604bd8d7716/content — Cambridge open-access mirror of the same Nature paper; high credibility (verbatim accepted manuscript). Source of the (24, 8)-strides example and BLAS/cache claims.
- https://pandas.pydata.org/docs/user_guide/groupby.html — official pandas user guide; authoritative for split-apply-combine (split/apply/combine; aggregation/transformation/filtration) and groupby defaults (sort, dropna, as_index, observed).
- https://pandas.pydata.org/docs/user_guide/merging.html — official pandas user guide; authoritative for merge `how=` ↔ SQL join mapping, concat vs merge vs join, key alignment, and `validate=`/`indicator=`.
- https://www.aidancooper.co.uk/pandas-anti-patterns/ — reputable practitioner write-up; source of the iterrows ≈47 s vs vectorized ≈0.03 s (1M rows) figure and the `.loc[mask, "col"]=` fix. Single-source benchmark → flagged illustrative.
- https://realpython.com/pandas-settingwithcopywarning/ — well-regarded explainer; corroborates the chained-indexing cause/fix and the underlying view-vs-copy mechanism (triangulates the anti-pattern with the NumPy copies doc).

**[unverified] gaps to confirm before publishing:**
- The 47 s vs 0.03 s (≈1000×) iterrows figure rests on **one** practitioner benchmark; present as illustrative, not a fixed constant.
- The general "10–100×" vectorization speedup is the widely-cited order of magnitude but is workload/dtype-dependent — keep it as a range, not a precise claim.
- pandas **3.0 Copy-on-Write becoming the default** (and largely retiring SettingWithCopyWarning) — verify against the version installed in the project's conda env (`py314_venv`) before stating it as current default behavior.
