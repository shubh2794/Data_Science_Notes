# tools/ — knowledge-base auditing

## hierarchy_audit.py

A read-only structural & connectivity auditor for the ml-notes knowledge base.
Run from anywhere:

```
python3 tools/hierarchy_audit.py        # JSON report to stdout
```

### What it reads (data sources)

1. **`data.js`** — parsed for `nodes`, `tree`, `subtree` (level-3 sub-topics), and `cross`. This is the single
   source of truth for hierarchy and graph wiring.
2. **`*.html` on disk** in each domain folder — the ground truth for which pages
   actually exist, and (for in-body links) what each page points at.

Note on backlinks: the site's sidebar **Connections** list, the **Local graph**,
and graph↔note switching are all derived at runtime from `ADJ = tree + cross`
(see `notes.js`). Cross edges are bidirectional by construction, so there is no
such thing as a hand-authored backlink that can rot — the meaningful failures
live in the **data layer** and the **data↔disk seam**, which is what this script
targets. It additionally scans hand-authored in-body `<a>`/`<img>` links, which
are NOT data-driven and therefore can (and do) break.

### What it checks (findings keys)

| Key | Meaning | Severity |
|-----|---------|----------|
| `broken_links` | a node's `link:` points to a file not on disk | blocker |
| `orphan_files` | an `.html` on disk that no node links to (unreachable) | blocker |
| `in_tree_not_in_nodes` | id listed in `tree` but missing from `nodes[]` | blocker |
| `leaf_nodes_not_in_tree` | level-2 node not placed in any `tree` domain | blocker |
| `subtree_parent_not_topic` | a `subtree` parent that is not a level-2 topic in `tree` | blocker |
| `subtree_child_not_level3` | a `subtree` child that is missing from `nodes[]` or not `level:3` | blocker |
| `level3_not_in_subtree` | level-3 node not placed under any parent in `subtree` | blocker |
| `cross_unknown_ids` | a `cross` edge references an unknown id | blocker |
| `broken_inbody_links` | a hand-authored `href`/`src` to a missing `.html` | high |
| `folder_domain_mismatch` | page's folder ≠ the node's domain in `tree` (misfiled) | high |
| `near_duplicate_ids` | two nodes share a normalized stem (merge candidates) | high |
| `duplicate_node_ids` | exact duplicate id in `nodes[]` | high |
| `weakly_connected_pages` | a built page with ≤1 cross edge (under-linked) | medium |
| `cross_isolated_leaves` | a leaf node with 0 cross edges | medium (low if stub) |
| `stub_create_candidates` | linkless node referenced by ≥1 cross edge — a page-to-build gap | medium |

### Output

A single JSON object: `{ "summary": {...}, "findings": {...} }`. Empty arrays
mean that check passed. Designed to be eyeballed or piped to `python3 -m json.tool`
/ `jq`.

### Known false positive to be aware of

A raw grep for LaTeX (`\frac`, `\sum`, `$$`) can match viz `<script>` code
(e.g. a JS template literal `$${value}`). When checking for LaTeX leakage,
restrict to prose / `.formula` blocks, not `<script>` bodies.

## How this capability should live

This is a **reusable script**, not an agent that re-derives the rules each run.
The notes-quality-reviewer agent remains the per-page conventions gate; this
script is the **whole-graph structural gate**. Recommended cadence: run after any
`data.js` change or new page, and treat non-empty `broken_*`, `*_not_in_*`, and
`folder_domain_mismatch` as merge-blockers.
