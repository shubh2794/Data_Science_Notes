#!/usr/bin/env python3
"""
hierarchy_audit.py — structural / connectivity audit for the ml-notes knowledge base.

Single source of truth is data.js (nodes, tree, cross). Ground truth for pages is
the set of *.html files on disk. The site's sidebar (Connections, backlinks, local
graph) is auto-derived from ADJ = tree + cross, so backlinks are bidirectional by
construction — meaning the meaningful failure modes live in the DATA layer and the
DATA<->DISK seam, not in hand-authored <a> tags.

This script reads, never writes. Run:  python3 tools/hierarchy_audit.py
"""
import os, re, sys, json
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data.js")

DOMAIN_FOLDERS = {
    "machine-learning": "Machine Learning",
    "math": "Math & Statistics",
    "programming": "Programming & Tools",
    "dsa": "Data Structures & Algorithms",
    "deep-learning": "Deep Learning",
    "models": "Model Architectures",
    "nlp": "NLP",
    "llm": "LLMs & Generative AI",
    "agentic": "Agentic AI",
    "data-engineering": "Data Engineering",
    "databases": "Data Systems",
    "mlops": "MLOps & Deployment",
    "speech": "Speech & Audio",
}

# ---------------------------------------------------------------- parse data.js
def load_data():
    src = open(DATA, encoding="utf-8").read()

    # nodes: each {id:"...", group:"...", level:N, link:"..."?, desc:"..."}
    nodes = []
    node_block = re.search(r"const nodes\s*=\s*\[(.*?)\n\];", src, re.S).group(1)
    for m in re.finditer(r"\{id:\"((?:[^\"\\]|\\.)*)\"\s*,\s*group:\"([^\"]+)\"\s*,\s*level:(\d+)(?:\s*,\s*link:\"([^\"]+)\")?", node_block):
        nodes.append({"id": m.group(1), "group": m.group(2),
                      "level": int(m.group(3)), "link": m.group(4)})

    # tree: "Domain":["a","b",...]
    tree = {}
    tree_block = re.search(r"const tree\s*=\s*\{(.*?)\n\};", src, re.S).group(1)
    for m in re.finditer(r"\"([^\"]+)\"\s*:\s*\[([^\]]*)\]", tree_block):
        kids = re.findall(r"\"((?:[^\"\\]|\\.)*)\"", m.group(2))
        tree[m.group(1)] = kids

    # subtree: "Parent topic":["child","child",...]  (level-3 sub-topics; optional block)
    subtree = {}
    sm = re.search(r"const subtree\s*=\s*\{(.*?)\n\};", src, re.S)
    if sm:
        for m in re.finditer(r"\"([^\"]+)\"\s*:\s*\[([^\]]*)\]", sm.group(1)):
            subtree[m.group(1)] = re.findall(r"\"((?:[^\"\\]|\\.)*)\"", m.group(2))

    # cross: ["a","b"] or ["a","b","isa"]
    cross = []
    cross_block = re.search(r"const cross\s*=\s*\[(.*?)\n\];", src, re.S).group(1)
    for m in re.finditer(r"\[\s*\"((?:[^\"\\]|\\.)*)\"\s*,\s*\"((?:[^\"\\]|\\.)*)\"(?:\s*,\s*\"([^\"]*)\")?\s*\]", cross_block):
        cross.append((m.group(1), m.group(2), m.group(3)))

    return nodes, tree, cross, subtree


def html_files(folder):
    """Every .html under a domain folder, at ANY depth, as a ROOT-relative path.
    Sub-topic series live in their own sub-folder (e.g. math/statistics/), so this
    must recurse — a flat listdir would miss them and report them as missing."""
    d = os.path.join(ROOT, folder)
    if not os.path.isdir(d):
        return
    for dirpath, _dirnames, filenames in os.walk(d):
        for f in sorted(filenames):
            if f.endswith(".html"):
                yield os.path.relpath(os.path.join(dirpath, f), ROOT).replace(os.sep, "/")


def disk_pages():
    pages = set()
    for folder in DOMAIN_FOLDERS:
        pages.update(html_files(folder))
    return pages


# ---------------------------------------------------------------- checks
def audit():
    nodes, tree, cross, subtree = load_data()
    by_id = {n["id"]: n for n in nodes}
    node_ids = set(by_id)
    pages = disk_pages()
    linked = {n["link"] for n in nodes if n["link"]}

    id_to_domain = {}
    for dom, kids in tree.items():
        for k in kids:
            id_to_domain[k] = dom
    # level-3 sub-topics inherit their parent topic's domain
    for top, kids in subtree.items():
        for k in kids:
            id_to_domain[k] = id_to_domain.get(top)
    sub_ids = set(k for kids in subtree.values() for k in kids)

    cross_deg = defaultdict(int)
    for s, t, _ in cross:
        if s in by_id: cross_deg[s] += 1
        if t in by_id: cross_deg[t] += 1

    R = {}

    # 1. data.js link -> missing file on disk
    R["broken_links"] = [{"id": n["id"], "link": n["link"]}
                         for n in nodes if n["link"] and n["link"] not in pages]

    # 2. html on disk with no node pointing at it
    R["orphan_files"] = sorted(p for p in pages if p not in linked)

    # 3. tree/nodes consistency
    tree_ids = set(k for kids in tree.values() for k in kids)
    R["in_tree_not_in_nodes"] = sorted(tree_ids - node_ids)
    R["leaf_nodes_not_in_tree"] = sorted(n["id"] for n in nodes
                                         if n["level"] == 2 and n["id"] not in tree_ids)

    # 3b. subtree consistency: parent must be a level-2 topic in the tree, children must be level-3 nodes,
    #     and every level-3 node must be placed under exactly one parent
    R["subtree_parent_not_topic"] = sorted(top for top in subtree if top not in tree_ids)
    R["subtree_child_not_level3"] = sorted(k for k in sub_ids if k not in by_id or by_id[k]["level"] != 3)
    R["level3_not_in_subtree"] = sorted(n["id"] for n in nodes if n["level"] == 3 and n["id"] not in sub_ids)

    # 4. cross edges referencing unknown ids
    R["cross_unknown_ids"] = [f"{s} -> {t}" for s, t, _ in cross
                              if s not in by_id or t not in by_id]

    # 5. folder vs tree-domain mismatch (misfiled nodes)
    mism = []
    for n in nodes:
        if not n["link"]:
            continue
        folder = n["link"].split("/")[0]
        expected = DOMAIN_FOLDERS.get(folder)
        actual = id_to_domain.get(n["id"])
        if expected and actual and expected != actual:
            mism.append({"id": n["id"], "link": n["link"],
                         "file_folder": folder, "tree_domain": actual})
    R["folder_domain_mismatch"] = mism

    # 6. fully isolated leaf nodes (no cross edge at all)
    R["cross_isolated_leaves"] = [{"id": n["id"], "has_page": bool(n["link"])}
                                  for n in nodes
                                  if n["level"] >= 2 and cross_deg[n["id"]] == 0]

    # 7. built pages that are weakly connected (<=1 cross edge)
    R["weakly_connected_pages"] = [{"id": n["id"], "link": n["link"], "cross_degree": cross_deg[n["id"]]}
                                   for n in nodes
                                   if n["link"] and cross_deg[n["id"]] <= 1]

    # 8. duplicate node ids
    seen, dups = set(), []
    for n in nodes:
        if n["id"] in seen:
            dups.append(n["id"])
        seen.add(n["id"])
    R["duplicate_node_ids"] = dups

    # 9. stub nodes (no link) that are heavily referenced -> create-a-page candidates
    ref = defaultdict(int)
    for s, t, _ in cross:
        ref[s] += 1; ref[t] += 1
    R["stub_create_candidates"] = sorted(
        ({"id": n["id"], "cross_refs": ref[n["id"]]}
         for n in nodes if n["level"] >= 2 and not n["link"] and ref[n["id"]] >= 1),
        key=lambda x: -x["cross_refs"])

    # 10. in-body href/src links to .html that resolve to a missing file on disk
    broken_inbody = []
    href_re = re.compile(r'(?:href|src)="([^"]+\.html[^"]*)"')
    for folder in DOMAIN_FOLDERS:
        for rel_page in html_files(folder):
            page = os.path.join(ROOT, rel_page)
            txt = open(page, encoding="utf-8").read()
            for m in href_re.finditer(txt):
                tgt = m.group(1).split("#")[0].split("?")[0]
                if tgt.startswith("http") or not tgt:
                    continue
                resolved = os.path.normpath(os.path.join(os.path.dirname(page), tgt))
                if not os.path.exists(resolved):
                    broken_inbody.append({"page": rel_page, "href": m.group(1)})
    R["broken_inbody_links"] = broken_inbody

    # 11. near-duplicate node ids (same normalized stem) — possible merge candidates
    def norm(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())
    stem = defaultdict(list)
    for n in nodes:
        base = re.sub(r"\s*\([^)]*\)", "", n["id"])  # drop parenthetical qualifier
        stem[norm(base)].append(n["id"])
    R["near_duplicate_ids"] = [v for v in stem.values() if len(v) > 1]

    # 12. duplicate cross edges — same unordered pair listed more than once.
    # ADJ = tree + cross is bidirectional, so ["a","b"] and ["b","a"] are the
    # same connection; either repetition is redundant data (and self-loops are
    # never meaningful). Report each offending pair with its occurrence count.
    pair_count = defaultdict(int)
    for s, t, _ in cross:
        pair_count[frozenset((s, t))] += 1
    R["duplicate_cross_edges"] = sorted(
        ({"pair": sorted(p), "count": c, "self_loop": len(p) == 1}
         for p, c in pair_count.items() if c > 1 or len(p) == 1),
        key=lambda x: (-x["count"], x["pair"]))

    summary = {
        "nodes_total": len(nodes),
        "leaf_nodes": sum(1 for n in nodes if n["level"] == 2),
        "subtopic_nodes": sum(1 for n in nodes if n["level"] == 3),
        "pages_linked_in_data": len(linked),
        "html_files_on_disk": len(pages),
        "cross_edges": len(cross),
    }
    return summary, R


if __name__ == "__main__":
    summary, R = audit()
    out = {"summary": summary, "findings": R}
    print(json.dumps(out, indent=1, ensure_ascii=False))
