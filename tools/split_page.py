#!/usr/bin/env python3
"""split_page.py — split one oversized notes page into a hub + child pages, one
level deeper, from a JSON manifest. The mechanical half of a split: a human (or
domain-specialist agent) decides the logical grouping and writes the manifest;
this script moves the markup, wires the links, and registers the new nodes in
data.js so nothing is copied by hand.

Usage:
    python3 tools/split_page.py manifest.json          # do it
    python3 tools/split_page.py manifest.json --dry    # report only, write nothing

Manifest shape (paths are ml-notes-root-relative):
{
  "parent_html": "dsa/algorithms/np-completeness.html",
  "parent_id":   "NP-Completeness & Approximation",     // node id in data.js
  "child_dir":   "dsa/algorithms/np-completeness",      // new folder for the children
  "keep_on_hub": ["what-np-completeness-is", "cheatsheet"],  // section ids that stay
  "map_intro":   "One optional sentence introducing the parts grid on the hub.",
  "children": [
    {
      "slug":  "complexity-classes",                    // -> child_dir/complexity-classes.html
      "id":    "P, NP & Reductions",                    // new node id (must be globally unique)
      "title": "P, NP &amp; Reductions",                // h1 (HTML-escaped)
      "part":  "the classes",                           // short kicker for the hub card
      "chs":   "P · NP · certificates · ≤ₚ",            // one-line chapter strip for the hub card
      "desc":  "Plain-text description for data.js and the child lede.",
      "sections": ["encodings", "decision-search-optimisation", "..."]
    }, ...
  ]
}

What it does, in order:
  1. parses the parent page into head / header / sections / tail;
  2. writes each child page (same head, +1 path depth, its own header & pager);
  3. rewrites the parent into a hub: kept sections + a .lamap parts grid;
  4. re-targets every anchor: #id links inside the split pages, and every
     "parent.html#id" link site-wide whose id moved to a child;
  5. inserts the child nodes + the subtree entry into data.js.

It refuses to run if the child dir already exists, a child id collides with an
existing node id, or any manifest section id is missing from the parent.
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMAIN_FOLDERS = ["math", "programming", "dsa", "databases", "data-engineering",
                  "machine-learning", "deep-learning", "vision", "nlp", "mlops",
                  "llm", "agentic", "speech", "models", "tools"]

LAMAP_CSS = """  .lamap{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin:18px 0 8px}
  .lamap a.card,.lamap .card{display:block;text-decoration:none;color:inherit;background:var(--panel-2);border:1px solid var(--line);border-radius:10px;padding:14px 16px;transition:border-color .15s}
  .lamap a.card:hover{border-color:var(--accent)}
  .lamap .part{font-size:11px;color:var(--muted);letter-spacing:.4px;text-transform:uppercase}
  .lamap .name{font-size:16px;font-weight:600;margin:4px 0 6px;color:var(--ink)}
  .lamap .chs{font-size:12px;color:var(--accent);margin-bottom:6px}
  .lamap .gist{font-size:13px;color:var(--muted);line-height:1.45}
"""

def esc(s):  # minimal text -> HTML
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def read(p):  return open(p, encoding="utf-8").read()
def write(p, s):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write(s)

# ---------------------------------------------------------------- parse parent
SECTION_OPEN = re.compile(r'<section class="topic" id="([^"]+)"[^>]*>')

def parse_sections(body):
    """Return (list of {id, start, end, html}) covering every section.topic.
    start includes an immediately preceding banner comment line if present."""
    secs = []
    for m in SECTION_OPEN.finditer(body):
        sid = m.group(1)
        # find matching </section> (sections are not nested in this corpus, but count anyway)
        i, depth = m.end(), 1
        while depth:
            nxt_open = body.find("<section", i)
            nxt_close = body.find("</section>", i)
            if nxt_close == -1:
                sys.exit(f"unbalanced <section> while scanning #{sid}")
            if nxt_open != -1 and nxt_open < nxt_close:
                depth += 1; i = nxt_open + 8
            else:
                depth -= 1; i = nxt_close + len("</section>")
        start = m.start()
        # pull in a banner comment directly above (e.g. <!-- ═══ 01 ═══ -->)
        prefix = body[:start]
        bm = re.search(r'(\n[ \t]*<!--[^\n]*-->[ \t]*\n[ \t]*)$', prefix)
        if bm:
            start = bm.start() + 1
        secs.append({"id": sid, "start": start, "end": i, "html": body[start:i]})
    return secs

def collect_ids(html):
    return set(re.findall(r'\bid="([^"]+)"', html))

# ------------------------------------------------- path depth bump for children
URL_ATTR = re.compile(r'\b(href|src)="([^"]*)"')
def bump_depth(html):
    """Child pages live one folder deeper than the parent: every relative URL
    (not #anchor, not absolute, not a scheme) gains one ../ prefix."""
    def fix(m):
        attr, url = m.group(1), m.group(2)
        if url.startswith(("#", "http://", "https://", "mailto:", "/", "data:")):
            return m.group(0)
        return f'{attr}="../{url}"'
    return URL_ATTR.sub(fix, html)

# ---------------------------------------------------------------- main
def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    dry = "--dry" in sys.argv
    mf = json.load(open(sys.argv[1], encoding="utf-8"))
    parent_rel = mf["parent_html"]
    parent_abs = os.path.join(ROOT, parent_rel)
    parent_dirrel = os.path.dirname(parent_rel)
    parent_file = os.path.basename(parent_rel)
    child_dirrel = mf["child_dir"]
    child_dirabs = os.path.join(ROOT, child_dirrel)
    children = mf["children"]
    keep = mf.get("keep_on_hub", [])
    warnings = []

    if os.path.isdir(child_dirabs) and os.listdir(child_dirabs):
        sys.exit(f"refusing: {child_dirrel}/ already exists and is not empty")
    if os.path.relpath(child_dirabs, os.path.dirname(parent_abs)).startswith(".."):
        sys.exit("refusing: child_dir must be a subfolder of the parent's folder")

    src = read(parent_abs)

    # --- data.js: parent node + uniqueness checks
    data_abs = os.path.join(ROOT, "data.js")
    data = read(data_abs)
    pid = mf["parent_id"]
    pat = re.compile(r'^([ \t]*)\{id:"' + re.escape(pid) + r'",\s*group:"([^"]+)",\s*level:(\d+),.*\},?\s*$', re.M)
    pm = pat.search(data)
    if not pm:
        sys.exit(f"parent node id not found in data.js: {pid}")
    indent, group, plevel = pm.group(1), pm.group(2), int(pm.group(3))
    clevel = plevel + 1
    existing_ids = set(re.findall(r'\{id:"((?:[^"\\]|\\.)*)"', data))
    for c in children:
        if c["id"] in existing_ids:
            sys.exit(f"child id collides with an existing node id: {c['id']}")
    if pid in re.findall(r'^\s*"((?:[^"\\]|\\.)*)":\s*\[', data[data.find("const subtree"):], re.M):
        sys.exit(f"refusing: {pid} already has a subtree entry")

    # --- parse the parent page
    head_end = src.find("</head>")
    body_start = src.find("<body>")
    sections = parse_sections(src)
    by_sid = {s["id"]: s for s in sections}
    manifest_ids = [sid for c in children for sid in c["sections"]]
    missing = [sid for sid in manifest_ids + keep if sid not in by_sid]
    if missing:
        sys.exit(f"manifest section ids not found in parent: {missing}")
    unassigned = [s["id"] for s in sections if s["id"] not in set(manifest_ids) | set(keep)]
    if unassigned:
        sys.exit(f"parent sections not assigned to any child or keep_on_hub: {unassigned}")
    dup = [sid for sid in manifest_ids if manifest_ids.count(sid) > 1]
    if dup:
        sys.exit(f"section ids assigned twice: {sorted(set(dup))}")

    # chrome pieces
    head = src[:head_end]            # up to but excluding </head>
    title_m = re.search(r"<title>([^<]*)</title>", head)
    parent_title = title_m.group(1).split("·")[-1].strip() if title_m else pid
    header_m = re.search(r'<header class="page">.*?</header>', src, re.S)
    if not header_m:
        sys.exit("no <header class=\"page\"> found in parent")
    first_sec_start = min(s["start"] for s in sections)
    last_sec_end = max(s["end"] for s in sections)
    chrome_top = src[body_start:first_sec_start]   # <body> .. header .. (up to first section)
    tail = src[last_sec_end:]                      # pager + script includes + </body>

    scripts_tail = "\n".join(re.findall(r'<script src="[^"]+"></script>', tail))

    # --- where does every element id live after the split?
    id_home = {}          # element id -> output file (root-relative)
    hub_ids = collect_ids(chrome_top) | {sid for sid in keep for _ in [0]}
    for sid in keep:
        for eid in collect_ids(by_sid[sid]["html"]):
            id_home[eid] = parent_rel
        id_home[sid] = parent_rel
    child_of = {}
    for c in children:
        crel = f"{child_dirrel}/{c['slug']}.html"
        for sid in c["sections"]:
            child_of[sid] = c
            for eid in collect_ids(by_sid[sid]["html"]):
                id_home[eid] = crel
            id_home[sid] = crel

    moved_section_ids = set(manifest_ids)

    # --- build child pages
    def child_head(c):
        h = head
        h = re.sub(r"<title>[^<]*</title>",
                   f"<title>Interactive Notes · {c['title']}</title>", h, count=1)
        return bump_depth(h)

    def retarget(html, self_rel):
        """Re-point #id links to wherever the id now lives."""
        page_ids = collect_ids(html)
        def fix(m):
            tid = m.group(1)
            if tid in page_ids:
                return m.group(0)
            home = id_home.get(tid)
            if home is None or home == self_rel:
                if home is None:
                    warnings.append(f"{self_rel}: unresolved anchor #{tid} (left as-is)")
                return m.group(0)
            relpath = os.path.relpath(os.path.join(ROOT, home),
                                      os.path.join(ROOT, os.path.dirname(self_rel))).replace(os.sep, "/")
            return f'href="{relpath}#{tid}"'
        return re.sub(r'href="#([^"]+)"', fix, html)

    n = len(children)
    out_pages = {}
    for i, c in enumerate(children):
        crel = f"{child_dirrel}/{c['slug']}.html"
        secs_html = "\n\n".join(bump_depth(by_sid[sid]["html"]) for sid in c["sections"])
        prev_a = (f'<a href="{children[i-1]["slug"]}.html">← {children[i-1]["title"]}</a>'
                  if i > 0 else f'<a href="../{parent_file}">← {esc(pid)} · overview</a>')
        next_a = (f'<a href="{children[i+1]["slug"]}.html">{children[i+1]["title"]} →</a>'
                  if i < n - 1 else f'<a href="../{parent_file}">{esc(pid)} · overview →</a>')
        page = f"""{child_head(c)}</head>
<body>
<div class="layout">
  <nav class="sidebar" id="sidebar"></nav>

  <main>
    <header class="page">
      <p style="margin:0 0 6px;font-size:13px;color:var(--muted)"><a href="../{parent_file}">{esc(pid)}</a> · part {i+1} of {n}</p>
      <h1>{c['title']}</h1>
      <p>{c.get('lede', esc(c['desc']))}</p>
      <div style="margin-top:10px">
        <span class="pill">{esc(pid)}</span>
        <span class="pill">{c.get('chs', f'Part {i+1} of {n}')}</span>
      </div>
    </header>

{secs_html}

    <nav class="pager" aria-label="{esc(pid)} series" style="display:flex;justify-content:space-between;gap:16px;margin:40px 0 24px;font-size:14px">
      {prev_a}
      <a href="../{parent_file}" style="color:var(--muted)">{esc(pid)} · overview</a>
      {next_a}
    </nav>
  </main>
</div>

{bump_depth(scripts_tail)}
</body>
</html>
"""
        out_pages[crel] = retarget(page, crel)

    # --- build the hub (rewritten parent)
    basename = os.path.basename(child_dirrel)
    cards = []
    for i, c in enumerate(children):
        cards.append(f"""        <a class="card" href="{basename}/{c['slug']}.html">
          <div class="part">Part {i+1}{(' · ' + esc(c['part'])) if c.get('part') else ''}</div>
          <div class="name">{c['title']}</div>
          <div class="chs">{c.get('chs','')}</div>
          <div class="gist">{esc(c['desc'])}</div>
        </a>""")
    intro = mf.get("map_intro", "")
    parts_section = f"""    <section class="topic" id="the-parts" data-nav="The parts">
      <h3>The notes, in {n} parts</h3>
      {f'<p class="lede">{intro}</p>' if intro else ''}
      <div class="lamap">
{chr(10).join(cards)}
      </div>
    </section>"""

    kept_in_order = [s for s in sections if s["id"] in set(keep)]
    hub_head = head if ".lamap{" in head else head.replace("</style>", LAMAP_CSS + "</style>")
    if "</style>" not in hub_head and ".lamap{" not in hub_head:
        hub_head = hub_head.replace("</head>", f"<style>\n{LAMAP_CSS}</style>\n")
    kept_html = "\n\n".join(s["html"] for s in kept_in_order)
    # the parts grid goes after the first kept intro section if one exists before the first child
    # section in document order, else directly after the header
    if kept_in_order and kept_in_order[0]["start"] == first_sec_start:
        body_mid = kept_in_order[0]["html"] + "\n\n" + parts_section
        rest = "\n\n".join(s["html"] for s in kept_in_order[1:])
        if rest: body_mid += "\n\n" + rest
    else:
        body_mid = parts_section + ("\n\n" + kept_html if kept_html else "")
    hub = f"""{hub_head}</head>
{chrome_top}{body_mid}
{tail}"""
    out_pages[parent_rel] = retarget(hub, parent_rel)

    # --- site-wide: re-point parent.html#movedid links from every other page
    sitewide = []
    link_pat = re.compile(r'(href=")([^"#]*)' + re.escape(parent_file) + r'#([^"]+)(")')
    for folder in DOMAIN_FOLDERS:
        d = os.path.join(ROOT, folder)
        if not os.path.isdir(d): continue
        for dirpath, _dn, fns in os.walk(d):
            for fn in fns:
                if not fn.endswith(".html") or fn.startswith("._"): continue
                frel = os.path.relpath(os.path.join(dirpath, fn), ROOT).replace(os.sep, "/")
                if frel == parent_rel or frel in out_pages: continue
                txt = read(os.path.join(ROOT, frel))
                def fix(m):
                    pre, anchor = m.group(2), m.group(3)
                    # only when it resolves to THIS parent (path suffix match)
                    target = os.path.normpath(os.path.join(os.path.dirname(frel), pre + parent_file)).replace(os.sep, "/")
                    if target != parent_rel or anchor not in id_home:
                        return m.group(0)
                    home = id_home[anchor]
                    if home == parent_rel:
                        return m.group(0)
                    rp = os.path.relpath(os.path.join(ROOT, home),
                                         os.path.join(ROOT, os.path.dirname(frel))).replace(os.sep, "/")
                    return f'{m.group(1)}{rp}#{anchor}{m.group(4)}'
                new = link_pat.sub(fix, txt)
                if new != txt:
                    sitewide.append(frel)
                    if not dry: write(os.path.join(ROOT, frel), new)

    # --- data.js: insert child nodes + subtree entry
    node_lines = "".join(
        f'{indent}{{id:"{c["id"]}", group:"{group}", level:{clevel}, link:"{child_dirrel}/{c["slug"]}.html", desc:"{c["desc"].replace(chr(92), chr(92)*2).replace(chr(34), chr(92) + chr(34))}"}},\n'
        for c in children)
    data_new = data[:pm.end()] + "\n" + node_lines.rstrip("\n") + data[pm.end():]
    sub_entry = f'  "{pid}":[' + ",".join(f'"{c["id"]}"' for c in children) + "],\n"
    sm = re.search(r"(const subtree\s*=\s*\{\n)(.*?)(\n\};)", data_new, re.S)
    if not sm:
        sys.exit("could not find subtree block in data.js")
    data_new = data_new[:sm.end(2)] + "\n" + sub_entry.rstrip("\n") + data_new[sm.end(2):]

    # --- write
    if not dry:
        for rel, content in out_pages.items():
            write(os.path.join(ROOT, rel), content)
        write(data_abs, data_new)

    print(f"{'DRY RUN — ' if dry else ''}split {parent_rel}")
    print(f"  {n} children -> {child_dirrel}/  (level {clevel}, group {group})")
    print(f"  hub keeps: {keep or '(none — header + parts grid only)'}")
    print(f"  sitewide anchor fixes in {len(sitewide)} page(s): {sitewide}")
    for w in sorted(set(warnings)):
        print(f"  WARN {w}")
    print(f"  data.js: +{n} nodes, +1 subtree entry under \"{pid}\"")

if __name__ == "__main__":
    main()
