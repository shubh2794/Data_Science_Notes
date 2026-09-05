"""Wire one finished part of the nine-part Statistics series into the site.

The page-building agents are told NOT to touch data.js, the hub, or their
neighbours' pagers, so that shared state is only ever edited from one place.
This script is that place.

usage: wire_part.py <partNo> "<node id>" <file.html> "<this part's title>" [<prev file.html>]

Does the four things an agent is told not to do:
  1. set link: on the level-3 node in data.js
  2. turn the hub's pending card into a real link
  3. link any hub cheat-sheet "Part N" cells
  4. add the forward arrow to the previous part's pager
Idempotent: every step checks before it edits.
"""
import re, sys, os

ROOT = "/Users/shubhamborikar/Projects/Personal/Study/ml-notes"
os.chdir(ROOT)
part = int(sys.argv[1]); nid = sys.argv[2]; fname = sys.argv[3]; title = sys.argv[4]
prev = sys.argv[5] if len(sys.argv) > 5 else None
link = f"math/statistics/{fname}"
done = []

# 1 ── data.js
s = open("data.js").read()
anchor = f'  {{id:"{nid}", group:"math", level:3, '
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
p = "math/statistics/index.html"; h = open(p).read()
marker = f'<div class="part">Part {part} · '
if f'href="{fname}"' in h:
    done.append("hub card already linked")
else:
    k = h.index(marker)
    open_tag = h.rindex('<div class="card pending">', 0, k)
    close = h.index("</div>", h.index('class="gist"', k))  # end of the gist div
    close = h.index("</div>", close + 6)                    # end of the card div
    h = (h[:open_tag] + f'<a class="card" href="{fname}">'
         + h[open_tag + len('<div class="card pending">'):close]
         + "</a>" + h[close + len("</div>"):])
    open(p, "w").write(h)
    done.append("hub card linked")

# 3 ── hub cheat-sheet cells
h = open(p).read()
cell = f"<td>Part {part}</td>"
n = h.count(cell)
if n:
    h = h.replace(cell, f'<td><a href="{fname}">Part {part}</a></td>')
    open(p, "w").write(h)
done.append(f"cheat-sheet cells linked: {n}")

# 4 ── previous part's pager
if prev:
    pp = f"math/statistics/{prev}"
    t = open(pp).read()
    m = re.search(r'<nav class="pager".*?</nav>', t, re.S)
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

print(" | ".join(done))
