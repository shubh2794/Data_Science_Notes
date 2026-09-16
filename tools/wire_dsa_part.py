"""Wire one finished part of a DSA series into the site.

The DSA domain is built as TWO seven-part level-3 series, each with its own hub:

    dsa/data-structures/index.html   — "Data Structures"
    dsa/algorithms/index.html        — "Algorithm Design & Analysis"

Page-building agents are told NOT to touch data.js, the hub, or their
neighbours' pagers, so that shared state is only ever edited from one place.
This script is that place — the DSA twin of wire_stats_part.py.

usage:
  wire_dsa_part.py <series> <partNo> "<node id>" <file.html> "<this part's title>" [<prev file.html>]

  <series> is the folder name: data-structures | algorithms

example:
  python3 tools/wire_dsa_part.py algorithms 2 "Recursion, Divide & Conquer" \
      divide-conquer.html "Recursion, Divide &amp; Conquer" analysis.html

Does the four things an agent is told not to do:
  1. set link: on the level-3 node in data.js
  2. turn the hub's pending card into a real link
  3. link any hub cheat-sheet "Part N" cells
  4. add the forward arrow to the previous part's pager

Idempotent: every step checks before it edits. It does NOT add cross edges —
run tools/hierarchy_audit.py afterwards and expect to add some by hand.
"""
import re, sys, os

ROOT = "/Users/shubhamborikar/Projects/Personal/Study/ml-notes"
SERIES = {"data-structures": "Data Structures",
          "algorithms": "Algorithm Design & Analysis"}

os.chdir(ROOT)
if len(sys.argv) < 6:
    sys.exit(__doc__)
series = sys.argv[1]
if series not in SERIES:
    sys.exit(f"FAIL: series must be one of {sorted(SERIES)}, got {series!r}")
part = int(sys.argv[2]); nid = sys.argv[3]; fname = sys.argv[4]; title = sys.argv[5]
prev = sys.argv[6] if len(sys.argv) > 6 else None
link = f"dsa/{series}/{fname}"
hub = f"dsa/{series}/index.html"
done = []

if not os.path.exists(link):
    sys.exit(f"FAIL: {link} does not exist — build the page first")

# 1 ── data.js
s = open("data.js").read()
anchor = f'  {{id:"{nid}", group:"dsa", level:3, '
if s.count(anchor) != 1:
    sys.exit(f"FAIL: node anchor for {nid!r} found {s.count(anchor)} times")
i = s.index(anchor)
seg = s[i:s.index("\n", i)]
if 'link:"' in seg:
    done.append("data.js already linked")
else:
    j = s.index('desc:"', i)
    s = s[:j] + f'link:"{link}", ' + s[j:]
    open("data.js", "w").write(s)
    done.append("data.js linked")

# 2 ── hub card
h = open(hub).read()
marker = f'<div class="part">Part {part} · '
if f'href="{fname}"' in h:
    done.append("hub card already linked")
elif marker not in h:
    done.append(f"WARN: no hub card marked 'Part {part} · '")
else:
    k = h.index(marker)
    open_tag = h.rindex('<div class="card pending">', 0, k)
    close = h.index("</div>", h.index('class="gist"', k))   # end of the gist div
    close = h.index("</div>", close + 6)                     # end of the card div
    h = (h[:open_tag] + f'<a class="card" href="{fname}">'
         + h[open_tag + len('<div class="card pending">'):close]
         + "</a>" + h[close + len("</div>"):])
    open(hub, "w").write(h)
    done.append("hub card linked")

# 3 ── hub cheat-sheet cells
h = open(hub).read()
cell = f"<td>Part {part}</td>"
n = h.count(cell)
if n:
    h = h.replace(cell, f'<td><a href="{fname}">Part {part}</a></td>')
    open(hub, "w").write(h)
done.append(f"cheat-sheet cells linked: {n}")

# 4 ── previous part's pager
if prev:
    pp = f"dsa/{series}/{prev}"
    if not os.path.exists(pp):
        done.append(f"WARN: prev page {pp} missing")
    else:
        t = open(pp).read()
        m = re.search(r'<nav class="pager".*?</nav>', t, re.S)
        if not m:
            done.append("WARN: prev page has no pager")
        else:
            nav = m.group(0)
            if f'href="{fname}"' in nav:
                done.append("prev pager already linked")
            elif "<span></span>" not in nav:
                done.append("WARN: prev pager has no empty span to fill")
            else:
                kk = nav.rfind("<span></span>")
                nav2 = nav[:kk] + f'<a href="{fname}">{title} →</a>' + nav[kk + len("<span></span>"):]
                open(pp, "w").write(t.replace(nav, nav2))
                done.append("prev pager linked")

# 5 ── remind about the hub's "still the pre-series drafts" callout
h = open(hub).read()
if "Interim page" in open(link).read() or "pre-series drafts" in h:
    done.append("NOTE: check the hub's draft callout and the page's own interim notice")

print(" | ".join(done))
