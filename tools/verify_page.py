"""Structural check for one notes page. Usage: verify_page.py <path/to/page.html> [...]

Checks the house rules that are cheap to get wrong and expensive to notice:
inline scripts, LaTeX leakage, SVG accessibility attributes, and lowercase Greek
inside a .lbl (which the uppercase style would silently turn into a different
symbol). Exits non-zero if anything fails.
"""
import re, sys, os

FAIL = 0
for path in sys.argv[1:]:
    s = open(path, encoding="utf-8").read()
    name = os.path.basename(path)
    problems = []

    if not s.rstrip().endswith("</html>"):
        problems.append("file does not end with </html> (truncated?)")
    for tag in ("section", "div", "svg", "script"):
        if s.count("<" + tag) != s.count("</" + tag + ">"):
            problems.append(f"unbalanced <{tag}>: {s.count('<'+tag)} open vs {s.count('</'+tag+'>')} close")

    inline = [m for m in re.finditer(r"<script(?![^>]*src=)[^>]*>(.*?)</script>", s, re.S)]
    big = [m for m in inline if len(m.group(1).strip()) > 1500]
    if big:
        problems.append(f"{len(big)} large inline <script> block(s) — page code belongs in a sibling .viz.js")

    prose = re.sub(r"<script.*?</script>", "", s, flags=re.S)   # avoid the $${} false positive
    if re.search(r"\\frac|\\begin\{|\$\$|MathJax|katex", prose):
        problems.append("LaTeX or a math typesetting library in the prose — equations must be Unicode")

    svgs = re.findall(r"<svg\b[^>]*>", s)
    for tag in svgs:
        for attr in ("viewBox", "role=", "aria-label"):
            if attr not in tag:
                problems.append(f"an <svg> is missing {attr.rstrip('=')}")
                break

    for m in re.finditer(r'<span class="lbl">(.*?)</span>', s, re.S):
        if re.search(r"[α-ωϊ-ώ]", m.group(1)) and "keep" not in m.group(1):
            problems.append(f'lowercase Greek unwrapped in a label: "{m.group(1)[:48]}" — wrap it in <span class="keep">')

    status = "FAIL" if problems else "ok"
    print(f"{name:34s} {status}   sections={s.count('<section class=')} svg={len(svgs)} size={len(s)//1024}KB")
    for p in dict.fromkeys(problems):
        print("    - " + p)
    if problems:
        FAIL = 1
sys.exit(FAIL)
