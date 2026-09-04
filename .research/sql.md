# Research Brief — SQL (the query language, for data-science / analytics work)

## 1. Topic & scope

**In scope:** SQL framed as the **declarative query language for getting analysis-ready data out of structured tables** — extracting, filtering, aggregating, reshaping, and feature-engineering over relational data for analytics and ML.
- The **relational model & declarative paradigm** — you state WHAT you want; the optimizer chooses HOW (the plan).
- **Query anatomy and the LOGICAL execution order** (FROM/JOIN → WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → LIMIT) and the confusions it explains (no SELECT aliases in WHERE; WHERE vs HAVING).
- **JOINs** (inner / left / right / full / cross; the semi-join & anti-join idea via EXISTS / NOT EXISTS) and the **key-cardinality fan-out** gotcha.
- **Aggregation & GROUP BY**; **WINDOW functions** (OVER / PARTITION BY / ORDER BY, ROW_NUMBER / RANK / LAG / LEAD, running totals) as the analytics power tool, contrasted with GROUP BY.
- **Subqueries vs CTEs (WITH)** and **recursive CTEs**; **set operations** (UNION / INTERSECT / EXCEPT).
- **NULL three-valued logic** and its gotchas (NULL comparison, NOT IN).
- The **SQL ↔ pandas** mapping (GROUP BY ↔ groupby, JOIN ↔ merge).
- A **light, accurate note on indexes & EXPLAIN** — how they change query cost, how to read a plan — deferring real index internals to the db-domain Indexing node.
- **SQL dialects** at a glance (ANSI core vs Postgres / MySQL / BigQuery / SQLite).
- **SQL's role in the DS / analytics + feature-engineering workflow** (warehouses, OLAP, analytical SQL).

**Out of scope (hard scope guard):** database **internals** belong to the separate Databases (`db`) domain — storage engines, B-tree internals, ACID / transaction mechanics, NoSQL data models, sharding / replication. Those are covered by the db-domain nodes (Relational & SQL, NoSQL, Indexing, Transactions (ACID), Sharding & Replication), which are currently **page-less stubs** → reference them **in prose**, do not link and do not re-teach. Likewise pandas already has a page — draw the analogy and **link** to `programming/pandas-numpy.html` rather than re-teaching dataframes. DDL/admin (schema design, GRANT, indexes-as-create-statements), stored procedures, and full transaction control are only touched lightly as context.

This brief **fills an existing page-less stub** — the `SQL` node already exists in `data.js` (line 61, `group:"prog"`, `level:2`, no `link:`).

## 2. Core concept

SQL (Structured Query Language) is the **declarative** language for querying and manipulating data held in **relations** (tables). A table is a set of rows (tuples) over a fixed set of typed columns (attributes); rows are identified by keys and tables are connected by matching key values. You write a query that describes the **result set you want** — which columns, which filters, which groupings — and the database's **query optimizer/planner** translates that declaration into a concrete physical execution plan (which scans, which join algorithm, which index), choosing the lowest estimated-cost plan. You say WHAT, not HOW.

For data work, the core loop is: **select** the columns you need, **filter** rows (WHERE), **join** related tables on keys, **aggregate** into summaries (GROUP BY) or compute per-row analytics that keep every row (WINDOW functions), **reshape**, and **order/limit** the output — producing a clean, analysis-ready table that feeds EDA, dashboards, or an ML feature pipeline.

## 3. Intuition

**Declarative = describe the destination, not the route.** Unlike imperative code (pandas, a for-loop) where you spell out each step, SQL lets you describe the answer and delegates the procedure. The same query can run many different ways with wildly different speed; the optimizer uses table statistics (row counts, value distributions, available indexes) to pick a plan. This is freeing (you rarely hand-tune algorithms) and occasionally surprising (a tiny query rewrite or a stale statistic can flip the plan).

**Set-based, not row-by-row.** SQL thinks in whole tables (sets/multisets) at once, the same mindset as vectorization in NumPy/pandas: express the transformation over the entire column rather than looping individual rows. "Looping in SQL" (cursors, row-at-a-time) is the anti-pattern, just as `iterrows` is in pandas.

**The written order is a lie; the logical order explains everything.** You write SELECT first, but the engine evaluates FROM/JOIN first and SELECT near the end. Once you internalize the logical order, the classic beginner confusions dissolve — they are all "I referenced something that didn't exist yet at that step."

**GROUP BY collapses; WINDOW keeps.** A GROUP BY aggregate squashes each group into one summary row. A window function computes the same kind of aggregate but **attaches it to every original row** — so "each employee's salary alongside their department average" or "running total to date" is one window expression, not a self-join. This is the single biggest leverage point in analytical SQL.

**NULL means "unknown," not "zero" or "empty."** Because a comparison with an unknown is itself unknown, SQL uses three-valued logic (TRUE / FALSE / UNKNOWN), and a row only survives a filter if the predicate is TRUE. Most NULL surprises follow mechanically from that one rule.

**SQL ≈ pandas with a planner.** GROUP BY ↔ `groupby`, JOIN ↔ `merge`, WHERE ↔ boolean mask, ORDER BY ↔ `sort_values`, UNION ALL ↔ `concat`. The difference: SQL is declarative and optimizer-driven over (often) larger, on-disk / warehouse data; pandas is imperative and in-memory. (See `programming/pandas-numpy.html`.)

## 4. Key math & rules (Unicode only)

**Relational model (set notation).** A relation R is a set of tuples over attributes A₁ … Aₙ; the schema is R(A₁, … , Aₙ). A row r ∈ R. SQL tables are technically **multisets** (bags) — duplicate rows are allowed unless DISTINCT or a key forbids them. Core relational operators map to SQL:
  σ_predicate(R)        selection      → WHERE
  π_cols(R)             projection     → SELECT (with DISTINCT for true set projection)
  R ⋈ S                 inner join     → JOIN … ON
  R ⟕ S, R ⟖ S, R ⟗ S   left / right / full outer join
  R × S                 cross product  → CROSS JOIN
  R ∪ S, R ∩ S, R − S   union / intersect / except (set ops)

**Logical query processing order** (the order the engine evaluates clauses, regardless of how you write them):
  FROM / JOIN  →  WHERE  →  GROUP BY  →  HAVING  →  SELECT (incl. window fns)  →  DISTINCT  →  ORDER BY  →  LIMIT / OFFSET
Consequences that follow directly:
  • SELECT aliases are NOT visible in WHERE / GROUP BY / HAVING (SELECT runs later) but ARE visible in ORDER BY (runs after SELECT).
  • WHERE filters **rows before grouping**; HAVING filters **groups after aggregation** — so aggregate predicates (e.g. count(*) > 5) belong in HAVING, not WHERE.
  • Window functions run **after** GROUP BY/HAVING and **cannot** appear in WHERE/GROUP BY/HAVING — to filter on a window result (e.g. ROW_NUMBER = 1) wrap it in a subquery/CTE and filter the outer query.

**Aggregation.** GROUP BY g₁ … gₖ partitions rows by the distinct key combinations and collapses each partition to one row; the SELECT list may contain only the grouping keys and aggregate functions (sum, count, avg, min, max, …). Result cardinality = number of distinct key combinations.

**Window functions.** f(args) OVER (PARTITION BY p ORDER BY o [frame]) — computed per row, no collapse.
  ROW_NUMBER()  → 1,2,3,4 within partition, no ties
  RANK()        → 1,2,2,4 (ties share rank, gaps after)
  DENSE_RANK()  → 1,2,2,3 (ties share rank, no gaps)
  LAG(x, k), LEAD(x, k) → value k rows back / forward in the ordered partition
  Running total: sum(x) OVER (ORDER BY o ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
  Default frame **with** ORDER BY = RANGE UNBOUNDED PRECEDING … CURRENT ROW (cumulative); **without** ORDER BY = the whole partition.

**JOIN cardinality / fan-out.** If each left row matches mᵢ right rows, output rows = Σᵢ mᵢ. With a unique key on the right it is **1:1 / many:1** (no blow-up); with duplicate keys on both sides it is **many:many** and approaches a per-key Cartesian product. The classic bug: an aggregate **after** a fan-out join double-counts. Output size of CROSS JOIN of R and S = |R| · |S|.

**NULL three-valued logic.** Comparisons involving NULL yield UNKNOWN, not TRUE/FALSE:
  NULL = NULL   → UNKNOWN        (use IS NULL / IS NOT NULL, never = NULL)
  NULL = 5      → UNKNOWN
  TRUE  AND UNKNOWN → UNKNOWN     TRUE  OR UNKNOWN → TRUE
  FALSE AND UNKNOWN → FALSE       FALSE OR UNKNOWN → UNKNOWN
  NOT UNKNOWN → UNKNOWN
A WHERE predicate keeps a row only when it evaluates to TRUE (UNKNOWN is dropped, like FALSE).
  **NOT IN gotcha:** `x NOT IN (a, b, NULL)` ≡ `x≠a AND x≠b AND x≠NULL`; the last term is UNKNOWN, so the whole AND is never TRUE → **zero rows**. Prefer NOT EXISTS, or exclude NULLs from the subquery.

**Set operations** (operate on whole result sets, columns must align by position/type):
  UNION (distinct) / UNION ALL (keeps duplicates, cheaper), INTERSECT, EXCEPT (MINUS in some dialects).

**Cost intuition / EXPLAIN.** The planner estimates cost from statistics and picks the minimum. A **sequential scan** reads the whole table — Θ(N) — best when most rows qualify or the table is small. An **index scan** jumps to matching rows — roughly Θ(log N + k) for k results — a big win only when **selectivity is high** (k ≪ N). EXPLAIN prints the chosen plan with estimated cost; EXPLAIN ANALYZE also runs it and prints actual rows/time, exposing bad row-count estimates. (Index data-structure internals — B-trees, hash indexes — belong to the Indexing db node.)

## 5. Practical takeaways & trade-offs

**Learn the logical execution order first.** It is the single highest-leverage mental model — it predicts which clause can see which names, where aggregate vs row filters go (WHERE vs HAVING), and why window results need a wrapping subquery to filter.

**Filter early, project narrow.** Push selective WHERE predicates and only the needed columns; on columnar warehouses, selecting fewer columns is a real I/O win. SELECT * is convenient but costs bandwidth and breaks downstream when schemas change.

**Joins: guard against fan-out.** Before a join feeding an aggregate, know the key cardinality. If the "right" side is not unique on the join key, you will multiply rows and silently corrupt SUM/COUNT/AVG. Fixes (cleanest first): pre-aggregate the detail table to one row per entity before joining; use a window/rank to pick one row per key; or join on the true key. Use **LEFT JOIN** to preserve unmatched left rows (they get NULLs); use **anti-join** (`WHERE NOT EXISTS (…)` or `LEFT JOIN … WHERE right.key IS NULL`) for "rows with no match"; use **semi-join** (`WHERE EXISTS (…)` / `IN`) for "rows that have a match" without duplicating.

**Reach for window functions for analytics.** Top-N per group, running totals, period-over-period change (LAG), deduplication (ROW_NUMBER() = 1), moving averages, percentiles — all are window expressions and far cleaner/faster than correlated subqueries or self-joins. Remember they can't live in WHERE/HAVING — wrap and filter.

**Prefer CTEs (WITH) for readability; know the materialization caveat.** CTEs flatten nested subqueries into named, top-to-bottom steps and can be referenced multiple times. Optimization note: a single-reference, side-effect-free CTE is often inlined (folded into the parent for joint optimization); a multiply-referenced one is typically materialized once. Some engines treat a CTE as an **optimization fence** (computed independently) — historically true in older PostgreSQL; PostgreSQL 12+ inlines by default and exposes MATERIALIZED / NOT MATERIALIZED to force it. [Exact default differs by engine/version — verify per target dialect.] Recursive CTEs (WITH RECURSIVE: base term UNION [ALL] recursive term) handle hierarchies/graphs (org charts, BOM explosions, path-finding) that plain SQL can't.

**NULL discipline.** Always IS NULL / IS NOT NULL; never = NULL. Avoid NOT IN against a subquery that can produce NULL (use NOT EXISTS). Remember COUNT(col) ignores NULLs while COUNT(*) counts rows; AVG ignores NULLs (so it is not SUM/COUNT(*)). Outer-join unmatched columns are NULL — use COALESCE to default them.

**Indexes & EXPLAIN, lightly.** An index helps **selective** lookups, joins, and ORDER BY/GROUP BY on indexed columns; it does not help when the query touches most rows (the planner will rightly choose a seq scan). Indexes cost write throughput and storage. Read EXPLAIN to see scan types and join algorithms; use EXPLAIN ANALYZE when estimated rows diverge from actual (often the root cause of a slow plan). Defer the data-structure details to the Indexing db node.

**Dialects — at a glance** (write ANSI core where possible, expect divergence at the edges):
  • Row limiting: LIMIT n (Postgres / MySQL / SQLite / BigQuery) vs TOP n (SQL Server) vs ANSI FETCH FIRST n ROWS ONLY.
  • String concat: || (ANSI / Postgres / SQLite / Oracle) vs CONCAT() (MySQL default; MySQL || means OR) .
  • Quoting identifiers: double quotes "col" (ANSI / Postgres) vs backticks `col` (MySQL) vs [col] (SQL Server).
  • Types & functions: date/time, JSON, casting, and regex functions vary widely; BigQuery is standards-ish but strict and array/struct-rich; SQLite has dynamic typing and omits some features (e.g. RIGHT/FULL JOIN only added in 3.39).
  • Window functions are ANSI:2003 standard and now broadly supported (MySQL since 8.0, SQLite since 3.25).

**SQL ↔ pandas mapping** (link to pandas-numpy.html; do not re-teach):
  SELECT cols          ↔ df[cols]
  WHERE pred           ↔ df[mask]
  GROUP BY k AGG       ↔ df.groupby("k").agg(...)
  JOIN … ON            ↔ df.merge(other, on=…, how=…)
  ORDER BY             ↔ df.sort_values(...)
  UNION ALL            ↔ pd.concat([...])
  window OVER PARTITION ↔ df.groupby("k")["v"].transform(...) / rank / shift
  DISTINCT             ↔ df.drop_duplicates()
Trade-off: pandas is imperative + in-memory; SQL is declarative + optimizer-driven and scales to warehouse-sized data without loading it into RAM.

**Role in the DS / ML workflow.** Most production features and training sets are assembled in SQL against a **data warehouse / lakehouse** (Snowflake, BigQuery, Redshift, Databricks SQL, DuckDB). These are **OLAP, columnar** systems tuned for wide aggregations over huge row ranges (vs **OLTP** row stores tuned for high-QPS point reads/writes — that distinction is db-domain context). The pattern: warehouse SQL does the heavy filter/join/aggregate to a compact, analysis-ready table; pandas/Spark/ML then take over. SQL is the front of the feature-engineering pipeline (window functions for time-based features, GROUP BY rollups, joins to enrich) feeding ETL / feature stores. SQL-on-big-data engines (Spark SQL, Trino/Presto, Hive) apply the same declarative surface over distributed data.

## 6. Suggested interactive viz

Match the sibling pages' style: drag/click-driven SVG with a live readout (classes `.viz`, `.vhead`, `.vtitle`, `.vhint`, `.controls`, `.btn`, `.readout`).

**Primary — "Logical execution order pipeline."** Show a small input table flowing through the clause stages FROM/JOIN → WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → LIMIT as horizontal stations. The user toggles each clause on/off (and edits a simple predicate / group key), and the viz animates rows being filtered, collapsed (GROUP BY visibly merging rows into summary rows), re-expanded, sorted, and truncated — with a live row-count badge at each stage. A highlight callout fires when the user tries to reference a SELECT alias in WHERE, dramatizing "that name doesn't exist yet." This single viz teaches the page's core mental model.

**Secondary — "GROUP BY vs WINDOW."** Same input rows shown twice side by side: left applies GROUP BY (rows collapse to one summary per group); right applies the same aggregate as a window OVER (PARTITION BY …) (every row survives and gains the value as a new column). A toggle adds ORDER BY within the window to switch the right panel from "partition average" to "running total," visibly accumulating down the partition. Makes "collapse vs keep" tangible.

**Tertiary (optional) — "Join fan-out explorer."** Two small tables with adjustable key uniqueness; the viz draws match lines and shows the output row count exploding from many:1 to many:many, with a SUM badge double-counting once fan-out occurs — the aggregation-after-join gotcha, dramatized.

## 7. Candidate placement

- **Action:** **fill the existing stub** (do NOT mint a new node). Node present in `data.js` line 61:
  `{id:"SQL", group:"prog", level:2, desc:"Querying and aggregating structured data from relational databases."}`
- **group:** `prog` (Programming & Tools) — confirmed. Domain folder is `programming/` (already holds `python.html`, `pandas-numpy.html`).
- **Proposed file path:** `programming/sql.html`. The ingestor should add `link:"programming/sql.html"` to the node.
- **Asset paths (page is one folder deep):** `../vendor/d3.min.js`, `../notes.css`, `../notes.js`; cross-links to siblings as `../folder/page.html`. Mirror the sibling structure exactly: `<section class="topic" id=… data-nav=…>` with `<h3><span class="idx">NN</span> …</h3>`, `.lede`, `.notes`, `.formula` (Unicode math), `.callout`, `.cmp` tables, and `.viz` D3 blocks; close with `<script src="../notes.js"></script>`.

**Cross-link candidates** (all real `data.js` ids; some edges already exist — keep, don't duplicate):
- **Pandas & NumPy** (`prog`, `programming/pandas-numpy.html`) — already linked (`["Pandas & NumPy","SQL"]`). Primary analogy: GROUP BY ↔ groupby, JOIN ↔ merge. **Link in prose.** Keep.
- **Relational & SQL** (`db`) — the database-internals counterpart (tables/keys/normalization, storage). Currently a **page-less stub** → **in-prose reference only**, not a link. The natural "for storage/engine internals, see …" pointer.
- **Indexing** (`db`) — how indexes change query cost; this page covers the query-side view only. **Page-less stub** → **in-prose reference**, not a link.
- **ETL Pipelines** (`deng`) — already linked via the SQL↔Data Engineering edge; SQL is the transform layer of ELT/ETL. Keep / reinforce. (`["SQL","Data Engineering"]` exists; an explicit `["SQL","ETL Pipelines"]` would be apt — ingestor's call.)
- **Data Warehousing** (`deng`) — already linked (`["SQL","Data Warehousing"]`); the OLAP/columnar home of analytical SQL. Keep.
- **Exploratory Data Analysis** (`viz`) — **add**: SQL is the first-pass slice/aggregate tool before/alongside EDA. Natural, currently-missing edge.
- **Apache Spark** (`prog`) — **add**: Spark SQL applies the same declarative surface over distributed data; the "graduate beyond one warehouse/engine" successor. Sibling in the same domain.

## 8. Sources consulted (audit trail only — do NOT surface on the published page)

- https://www.postgresql.org/docs/current/tutorial-window.html — official PostgreSQL docs; authoritative for OVER/PARTITION BY/ORDER BY, windows not collapsing rows, default frames, running totals, and that windows are logically applied after GROUP BY/HAVING and cannot appear in WHERE/HAVING (wrap in subquery to filter).
- https://www.postgresql.org/docs/current/queries-with.html — official PostgreSQL docs; authoritative for CTE syntax, CTE-vs-subquery, recursive CTE algorithm (non-recursive base UNION recursive term), and MATERIALIZED / NOT MATERIALIZED inlining behavior.
- https://www.postgresql.org/docs/current/tutorial-join.html — official PostgreSQL docs; authoritative for inner/left/right/full outer/cross joins, NULL substitution for unmatched outer-join rows, and self-joins.
- https://www.postgresql.org/docs/current/using-explain.html — official PostgreSQL docs (referenced via search); authoritative for EXPLAIN cost output and seq-scan vs index-scan plan choice.
- https://builtin.com/data-science/sql-order-of-execution — reputable explainer; corroborates the logical execution order and the alias-in-WHERE / WHERE-vs-HAVING consequences. Triangulated with the same claim across DataCamp, Baeldung, and SQLBolt search results.
- https://modern-sql.com/concept/three-valued-logic and https://learn.microsoft.com/en-us/sql/t-sql/language-elements/null-and-unknown-transact-sql — standards-focused + vendor docs; corroborate 3VL truth tables, NULL=NULL → UNKNOWN, IS NULL requirement, and the NOT IN-with-NULL zero-rows gotcha.
- https://sqlpad.io/tutorial/sql-joins-duplicate-rows-how-to-fix-them/ and https://docs.cloud.google.com/looker/docs/2604/sql-experts-joins — practitioner + vendor docs; corroborate join fan-out / row-multiplication on non-unique keys and the pre-aggregate / rank-to-one-row fixes.
- https://scalegrid.io/blog/postgres-explain-cost/ — reputable write-up; corroborates EXPLAIN start-up/total cost and index-scan-helps-only-when-selective intuition. (Triangulated with the PostgreSQL using-explain doc.)
- https://clickhouse.com/resources/engineering/oltp-vs-olap and https://aws.amazon.com/what-is/olap/ — vendor engineering docs; corroborate OLAP/columnar vs OLTP row-store distinction and SQL's role over warehouse data (ROLAP, analytical SQL). Distilled to neutral context.
- SQL dialect differences (LIMIT vs TOP, || vs CONCAT, quoting, SQLite RIGHT/FULL JOIN since 3.39, window-fn support since MySQL 8.0 / SQLite 3.25): triangulated across multiple dialect-comparison write-ups (learnsql.com, codefinity, tutorialreference) plus the relevant vendor docs.

**[unverified] gaps to flag before publishing:**
- **CTE inlining vs materialization default** varies by engine and version (older PostgreSQL = optimization fence; PG 12+ inlines single-reference CTEs by default; other engines differ). State as "depends on the engine/version," not a universal rule.
- **Exact dialect version cutoffs** (SQLite RIGHT/FULL JOIN 3.39, window functions MySQL 8.0 / SQLite 3.25) come from secondary write-ups — verify against current official release notes before stating versions on the page; safest to phrase as "modern versions of … support …".
- **Index-scan complexity Θ(log N + k)** is the standard B-tree intuition, accurate as an order-of-magnitude mental model but engine/index-type dependent; present as intuition, and defer precise internals to the Indexing db node.
