/* arrays-strings.viz.js — figures for dsa/data-structures/arrays-strings.html
   (part 1 of the Data Structures series).

   Loaded after ../../data.js, ../../notes.js and ../dsa-viz.js, so the globals
   AC (palette) and AL (rng / frame / axisB / axisL / gridY / row / bars /
   counter / stepper / …) are available.

   House rule obeyed throughout: every cost this page DISPLAYS — element moves,
   character comparisons, cache-line misses, hash evaluations, pointer steps —
   is produced by running the real routine under AL.counter() and reading the
   counter back afterwards. Nothing below is a number typed into a caption.
   Wherever a closed form exists it is evaluated independently and printed next
   to the measurement, so each figure is its own cross-check: if the two
   disagree, one of them is wrong and the figure says so out loud.

   Figures, in page order:
     01 #addr-svg    address arithmetic on an indexed row; index vs pointer walk
     02 #shift-svg   insert / delete shift cost at every position, measured
     03 #grow-svg    growth factor g: measured cost per append and space slack
     04 #rm-svg      row-major layout, traversal order, and measured cache misses
     05 #cat-svg     string building: repeated concatenation vs collect-and-join
     06 #tp-svg      two pointers, with the invariant displayed each step
     07 #win-svg     the sliding window, variable size, measured pointer moves
     08 #pre-svg     prefix sums: build once, answer any range in O(1)
     09 #dnf-svg     Dutch national flag: three regions, one pass, measured
     10 #naive-svg   the naive matcher, stepped, on its worst case
     11 #rk-svg      Rabin-Karp: the rolling hash, hits and spurious hits
     12 #pi-svg      the KMP prefix function, built cell by cell
     13 #race-svg    matcher race: measured comparisons, same text and pattern  */

/* ── shared little helpers ─────────────────────────────────────────────── */
const AS = {
  /* a <table class="cmp"> built from a header row + body rows, injected into a
     host element — used wherever a figure reports measured numbers as a table */
  table: function (sel, head, rows) {
    const h = d3.select(sel);
    if (h.empty()) return null;
    h.selectAll("*").remove();
    const t = h.append("table").attr("class", "cmp").style("margin", "10px 0 0");
    const hr = t.append("tr");
    head.forEach(c => hr.append("th").html(c));
    rows.forEach(r => {
      const tr = t.append("tr");
      r.forEach(c => tr.append("td").html(c));
    });
    return t;
  },
  int: d3.format(","),
  sig: function (x, d) { return (+x).toFixed(d === undefined ? 2 : d); },
  /* remove any stepper control bars a previous build left behind, so rebuilding
     a figure on a control change does not stack duplicate button rows */
  clearControls: function (svgNode) {
    svgNode.parentNode.querySelectorAll('div[role="group"]').forEach(el => el.remove());
  },
  /* an agree / disagree verdict, so a cross-check is visible rather than implied */
  verdict: function (ok) {
    return ok ? '<b style="color:' + AC.good + '">agree</b>'
              : '<b style="color:' + AC.bad + '">DISAGREE</b>';
  }
};

/* ══ FIGURE 01 ═══════════════════════════════════════════════════════════
   Address arithmetic. The point of the figure is that the address of A[i] is
   ONE multiply and ONE add away from the base, whatever i is — and that the
   same element reached through a chain of links costs i dereferences. Both
   counts come from AL.counter() wrapped around the two real routines.       */
(function () {
  const svg = d3.select("#addr-svg"); if (svg.empty()) return;
  const W = 690, H = 250, N = 12;

  const els = {
    base: d3.select("#addr-base"), size: d3.select("#addr-size"),
    i: d3.select("#addr-i"), iOut: d3.select("#addr-i-out"),
    out: d3.select("#addr-readout")
  };
  const VALUES = [17, 4, 23, 9, 41, 6, 38, 12, 55, 2, 30, 19];

  /* the two routines being compared, instrumented */
  function addressOf(base, itemsize, i, c) {
    c.add("mul"); c.add("add");                 // one multiply, one add — that is all
    c.add("randomAccessOps", 2);
    return base + i * itemsize;
  }
  function walkTo(head, i, c) {                 // what a linked chain costs instead
    let node = head;
    for (let k = 0; k < i; k++) { c.add("deref"); node = node.next; }
    return node;
  }
  /* a chain over the same values, so the comparison is like for like */
  function chain(values) {
    let head = null;
    for (let k = values.length - 1; k >= 0; k--) head = { v: values[k], next: head };
    return head;
  }

  function draw() {
    const base = +els.base.property("value");
    const itemsize = +els.size.property("value");
    const i = +els.i.property("value");
    els.iOut.text(i);

    const c = AL.counter();
    const addr = addressOf(base, itemsize, i, c);
    walkTo(chain(VALUES), i, c);

    /* independent cross-check: accumulate the cell widths one at a time rather
       than multiplying, which is the definition the multiply is a shortcut for */
    let acc = base;
    for (let k = 0; k < i; k++) acc += itemsize;
    const agrees = (acc === addr);

    const f = AL.frame(svg, W, H, { l: 18, r: 18, t: 16, b: 10 });
    const g = f.g;

    g.append("text").attr("x", 0).attr("y", 4).attr("font-size", 11).attr("fill", AC.muted)
      .text("the block of memory, drawn as equal-size cells");

    const r = AL.row(g, VALUES, {
      x: 0, y: 18, w: 50, h: 34, gap: 2, index: true,
      mark: k => k === i ? AC.accent : (k < i ? AC.panel : null),
      text: AC.ink
    });

    /* byte offsets under the index row */
    g.append("text").attr("x", 0).attr("y", 84).attr("font-size", 10).attr("fill", AC.muted)
      .text("byte offset from base");
    VALUES.forEach((v, k) => {
      g.append("text").attr("x", r.cellX(k)).attr("y", 100).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", k === i ? AC.a2 : AC.muted)
        .text(k * itemsize);
    });

    /* the address line */
    g.append("text").attr("x", 0).attr("y", 122).attr("font-size", 10).attr("fill", AC.muted)
      .text("address");
    VALUES.forEach((v, k) => {
      g.append("text").attr("x", r.cellX(k)).attr("y", 138).attr("text-anchor", "middle")
        .attr("font-size", 9.5).attr("fill", k === i ? AC.a2 : AC.line === null ? AC.muted : "#6d7686")
        .text(base + k * itemsize);
    });

    AL.arrow(g, r.cellX(i), 168, r.cellX(i), 146, { color: AC.a2, w: 1.6 });
    g.append("text").attr("x", r.cellX(i)).attr("y", 184).attr("text-anchor", "middle")
      .attr("font-size", 11).attr("fill", AC.a2).text("A[" + i + "] = " + VALUES[i]);

    /* the two access costs, as bars, measured */
    const y0 = 200, barMax = 300;
    const bars = [
      { lbl: "index A[" + i + "]", n: c.get("randomAccessOps"), col: AC.good },
      { lbl: "walk a chain to the " + (i + 1) + (i === 0 ? "st" : i === 1 ? "nd" : i === 2 ? "rd" : "th") + " node", n: c.get("deref"), col: AC.bad }
    ];
    const sc = d3.scaleLinear().domain([0, Math.max(2, N)]).range([0, barMax]);
    bars.forEach((b, k) => {
      const yy = y0 + k * 20;
      g.append("rect").attr("x", 210).attr("y", yy - 10).attr("width", Math.max(2, sc(b.n)))
        .attr("height", 13).attr("rx", 3).attr("fill", b.col).attr("opacity", 0.85);
      g.append("text").attr("x", 204).attr("y", yy).attr("text-anchor", "end")
        .attr("font-size", 11).attr("fill", AC.muted).text(b.lbl);
      g.append("text").attr("x", 214 + Math.max(2, sc(b.n))).attr("y", yy)
        .attr("font-size", 11).attr("fill", AC.ink)
        .text(b.n + (b.n === 1 ? " step" : " steps"));
    });

    els.out.html(
      "<b>addr(A[" + i + "]) = base + i × itemsize = " + base + " + " + i + " × " + itemsize
      + " = " + addr + "</b> &nbsp;·&nbsp; "
      + "measured: <b>" + c.get("mul") + "</b> multiply + <b>" + c.get("add") + "</b> add = <b>"
      + c.get("randomAccessOps") + " operations</b>, and that count does not change with i. "
      + "The same element reached by following links costs a measured <b>" + c.get("deref")
      + "</b> dereference" + (c.get("deref") === 1 ? "" : "s") + ". "
      + "Cross-check — adding itemsize " + i + " time" + (i === 1 ? "" : "s") + " instead of multiplying gives "
      + acc + ", which " + AS.verdict(agrees) + " with the multiplication."
    );
  }

  [els.base, els.size, els.i].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══ FIGURE 02 ═══════════════════════════════════════════════════════════
   What an insertion or a deletion costs, at every position. The bar chart is
   not a plot of the formula n − k; it is a plot of what AL.counter() recorded
   when the real shift loop was run once for each k. The formula is printed
   beside it so the two can be compared.                                     */
(function () {
  const svg = d3.select("#shift-svg"); if (svg.empty()) return;
  const W = 690, H = 320, N = 12;
  const BASE = [17, 4, 23, 9, 41, 6, 38, 12, 55, 2, 30, 19];

  const els = {
    op: d3.select("#shift-op"), k: d3.select("#shift-k"),
    kOut: d3.select("#shift-k-out"), out: d3.select("#shift-readout")
  };

  /* the real routines. `n` is the live element count; the block has room for
     one more so the insert has somewhere to shift into. */
  function insertAt(A, n, k, v, c) {
    for (let j = n; j > k; j--) { c.add("move"); A[j] = A[j - 1]; }
    c.add("write"); A[k] = v;
    return n + 1;
  }
  function deleteAt(A, n, k, c) {
    for (let j = k; j < n - 1; j++) { c.add("move"); A[j] = A[j + 1]; }
    c.add("blank"); A[n - 1] = null;
    return n - 1;
  }

  /* run every position once and keep the measured counts */
  function sweep(op) {
    const moves = [], top = op === "insert" ? N : N - 1;
    for (let k = 0; k <= top; k++) {
      const c = AL.counter();
      const A = [...BASE, null];
      if (op === "insert") insertAt(A, N, k, 99, c); else deleteAt(A, N, k, c);
      moves.push(c.get("move"));
    }
    return moves;
  }

  function draw() {
    const op = els.op.property("value");
    const top = op === "insert" ? N : N - 1;
    let k = AL.clamp(+els.k.property("value"), 0, top);
    els.k.attr("max", top);
    els.kOut.text(k);

    const moves = sweep(op);
    const c = AL.counter();
    const A = [...BASE, null];
    if (op === "insert") insertAt(A, N, k, 99, c); else deleteAt(A, N, k, c);
    const measured = c.get("move");
    const closed = op === "insert" ? (N - k) : (N - 1 - k);
    const total = moves.reduce((a, b) => a + b, 0);
    /* independent cross-check of the whole sweep: sum of (N − k) over the same
       range, computed from the closed form rather than from the counter */
    let closedTotal = 0;
    for (let j = 0; j <= top; j++) closedTotal += (op === "insert" ? N - j : N - 1 - j);
    const mean = total / (top + 1);

    const f = AL.frame(svg, W, H, { l: 20, r: 16, t: 16, b: 10 });
    const g = f.g;

    /* ── the row, before and after ───────────────────────────────────── */
    g.append("text").attr("x", 0).attr("y", 2).attr("font-size", 11).attr("fill", AC.muted)
      .text(op === "insert" ? "before — inserting 99 at index " + k
                            : "before — deleting the value at index " + k);
    const before = [...BASE, null];
    const r1 = AL.row(g, before, {
      x: 34, y: 12, w: 46, h: 28, gap: 2, index: true, fontSize: 12,
      mark: idx => idx === k ? AC.a2 : (idx > k && idx < N ? AC.panel : (idx === N ? AC.bg : null))
    });

    g.append("text").attr("x", 0).attr("y", 86).attr("font-size", 11).attr("fill", AC.muted)
      .text("after");
    const r2 = AL.row(g, A, {
      x: 34, y: 96, w: 46, h: 28, gap: 2, index: true, fontSize: 12,
      mark: idx => (op === "insert" && idx === k) ? AC.good
             : (op === "insert" && idx > k && idx <= N) ? AC.panel
             : (op === "delete" && idx >= k && idx < N - 1) ? AC.panel
             : (op === "delete" && idx >= N - 1) ? AC.bg : null
    });

    /* the shifted span, marked */
    const lo = op === "insert" ? k + 1 : k, hi = op === "insert" ? N : N - 2;
    if (hi >= lo) {
      const x1 = 34 + r1.cellX(lo), x2 = 34 + r1.cellX(hi);
      g.append("line").attr("x1", x1).attr("x2", x2).attr("y1", 74).attr("y2", 74)
        .attr("stroke", AC.accent).attr("stroke-width", 1.4).attr("stroke-dasharray", "3 3");
      AL.arrow(g, (x1 + x2) / 2, 74, op === "insert" ? x2 + 10 : x1 - 10, 74, { color: AC.accent, w: 1.4 });
      g.append("text").attr("x", (x1 + x2) / 2).attr("y", 68).attr("text-anchor", "middle")
        .attr("font-size", 10.5).attr("fill", AC.accent)
        .text(measured + " element" + (measured === 1 ? "" : "s") + " move "
              + (op === "insert" ? "right" : "left"));
    } else {
      g.append("text").attr("x", 34 + r1.cellX(Math.max(0, lo - 1))).attr("y", 70)
        .attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", AC.good)
        .text("nothing moves");
    }

    /* ── the measured cost at every position ─────────────────────────── */
    const pw = 626, ph = 128, py = 160;
    const sub = g.append("g").attr("transform", "translate(20," + py + ")");
    const x = d3.scaleLinear().domain([-0.6, top + 0.6]).range([0, pw - 40]);
    const y = d3.scaleLinear().domain([0, N]).range([ph, 0]);
    AL.gridY(sub, y, pw - 40, 4);
    AL.axisB(sub, x, ph, Math.min(top + 1, 13), "position k", d3.format("d"));
    AL.axisL(sub, y, 4, "element moves, measured");

    const bw = Math.max(6, (pw - 40) / (top + 2) - 4);
    moves.forEach((m, j) => {
      sub.append("rect").attr("x", x(j) - bw / 2).attr("y", y(m))
        .attr("width", bw).attr("height", ph - y(m)).attr("rx", 2)
        .attr("fill", j === k ? AC.a2 : AC.accent).attr("opacity", j === k ? 1 : 0.6);
    });
    /* the closed form, drawn from the formula, not from the counter */
    const line = d3.line().x(d => x(d)).y(d => y(op === "insert" ? N - d : N - 1 - d));
    sub.append("path").datum(d3.range(0, top + 1)).attr("fill", "none")
      .attr("stroke", AC.good).attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3")
      .attr("d", line);
    sub.append("line").attr("x1", 0).attr("x2", pw - 40).attr("y1", y(mean)).attr("y2", y(mean))
      .attr("stroke", AC.violet).attr("stroke-width", 1.3).attr("stroke-dasharray", "2 4");
    AL.legend(sub, [
      { label: "measured moves", color: AC.accent },
      { label: op === "insert" ? "closed form n − k" : "closed form n − 1 − k", color: AC.good, dash: "4 3" },
      { label: "mean over all k = " + AS.sig(mean, 2), color: AC.violet, dash: "2 4" }
    ], pw - 200, 14);

    els.out.html(
      "At <b>k = " + k + "</b> the counter recorded <b>" + measured + "</b> element move"
      + (measured === 1 ? "" : "s") + "; the closed form "
      + (op === "insert" ? "n − k = " + N + " − " + k : "n − 1 − k = " + (N - 1) + " − " + k)
      + " = " + closed + " — " + AS.verdict(measured === closed) + ". "
      + "Summed over all " + (top + 1) + " positions the counter recorded <b>" + AS.int(total)
      + "</b> moves against a closed-form total of " + AS.int(closedTotal) + " — " + AS.verdict(total === closedTotal)
      + " — so the mean cost at a uniformly random position is <b>" + AS.sig(mean, 2)
      + "</b> moves, which is n/2 to within the end effect. Both the worst case and the average are <b><span class='keep'>Θ</span>(n)</b>."
    );
  }

  [els.op, els.k].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══ FIGURE 03 ═══════════════════════════════════════════════════════════
   The growth factor g. Both panels are simulations of the real append loop
   with AL.counter() threaded through it: panel A counts element copies, panel
   B samples the live slack after every append. The dashed curve in panel A is
   the closed form g/(g−1), evaluated independently of the simulation.       */
(function () {
  const svg = d3.select("#grow-svg"); if (svg.empty()) return;
  const W = 690, H = 300;

  const els = {
    g: d3.select("#grow-g"), gOut: d3.select("#grow-g-out"),
    n: d3.select("#grow-n"), nOut: d3.select("#grow-n-out"),
    out: d3.select("#grow-readout")
  };

  /* the real append loop. A capacity rule stated as a FACTOR cannot grow a
     1-slot table (floor(1 × 1.5) = 1), so — like every real implementation —
     it falls back on at least one extra slot. */
  function runAppends(n, gf, c) {
    let cap = 1, num = 0, slack = 0;
    for (let i = 0; i < n; i++) {
      if (num === cap) {
        const nc = Math.max(cap + 1, Math.floor(cap * gf));
        c.add("copy", num);                 // every live element is moved
        c.add("resize");
        cap = nc;
      }
      c.add("place");                       // the element is written into its slot
      num++;
      slack += (cap - num) / cap;
    }
    return { cap, num, meanSlack: slack / n };
  }
  /* independent recomputation of the copy count from the capacity sequence
     alone — no counter involved — used as the figure's cross-check */
  function copiesFromCapacities(n, gf) {
    let cap = 1, num = 0, copies = 0;
    while (num < n) {
      if (num === cap) { copies += num; cap = Math.max(cap + 1, Math.floor(cap * gf)); }
      num++;
    }
    return copies;
  }

  function draw() {
    const gsel = +els.g.property("value"), n = +els.n.property("value");
    els.gOut.text(AS.sig(gsel, 2)); els.nOut.text(AS.int(n));

    const gs = AL.linspace(1.1, 4, 59);
    const pts = gs.map(gf => {
      const c = AL.counter();
      const res = runAppends(n, gf, c);
      return { g: gf, cost: (c.get("place") + c.get("copy")) / n,
               slack: res.meanSlack, resizes: c.get("resize") };
    });
    const cSel = AL.counter();
    const sel = runAppends(n, gsel, cSel);
    const measuredCost = (cSel.get("place") + cSel.get("copy")) / n;
    const predicted = gsel / (gsel - 1);
    const xcheck = copiesFromCapacities(n, gsel);

    const f = AL.frame(svg, W, H, { l: 14, r: 14, t: 14, b: 12 });
    const G = f.g;
    const pw = 286, ph = 208, ox = 56, oy = 26;

    function panel(dx, title, yacc, ydom, ylab, extra) {
      const s = G.append("g").attr("transform", "translate(" + (dx + ox) + "," + oy + ")");
      G.append("text").attr("x", dx + ox).attr("y", oy - 10).attr("font-size", 11)
        .attr("fill", AC.ink).text(title);
      const x = d3.scaleLinear().domain([1.1, 4]).range([0, pw]);
      const y = d3.scaleLinear().domain(ydom).range([ph, 0]);
      AL.gridY(s, y, pw, 5);
      AL.axisB(s, x, ph, 5, "growth factor g");
      AL.axisL(s, y, 5, ylab);
      s.append("path").datum(pts).attr("fill", "none").attr("stroke", AC.accent)
        .attr("stroke-width", 2)
        .attr("d", d3.line().x(d => x(d.g)).y(d => y(AL.clamp(yacc(d), ydom[0], ydom[1]))));
      if (extra) extra(s, x, y);
      s.append("circle").attr("cx", x(gsel)).attr("cy", y(AL.clamp(yacc({ g: gsel, cost: measuredCost, slack: sel.meanSlack }), ydom[0], ydom[1])))
        .attr("r", 4.5).attr("fill", AC.a2);
      return { s, x, y };
    }

    panel(0, "measured cost per append (places + copies) ÷ n", d => d.cost, [0, 12],
      "cost per append", (s, x, y) => {
        s.append("path").datum(gs).attr("fill", "none").attr("stroke", AC.good)
          .attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3")
          .attr("d", d3.line().x(d => x(d)).y(d => y(AL.clamp(d / (d - 1), 0, 12))));
        AL.legend(s, [{ label: "measured", color: AC.accent },
                      { label: "closed form g/(g−1)", color: AC.good, dash: "4 3" }], pw - 150, 12);
      });

    panel(330, "measured mean space slack (capacity − live) ÷ capacity", d => d.slack, [0, 0.8],
      "wasted fraction", (s, x, y) => {
        s.append("path").datum(gs).attr("fill", "none").attr("stroke", AC.rose)
          .attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3")
          .attr("d", d3.line().x(d => x(d)).y(d => y(AL.clamp(1 - 1 / d, 0, 0.8))));
        AL.legend(s, [{ label: "measured mean", color: AC.accent },
                      { label: "worst-case slack 1 − 1/g", color: AC.rose, dash: "4 3" }], pw - 168, 12);
      });

    els.out.html(
      "At <b>g = " + AS.sig(gsel, 2) + "</b>, <b>n = " + AS.int(n) + "</b>: the counter recorded <b>"
      + AS.int(cSel.get("copy")) + "</b> element copies over <b>" + AS.int(cSel.get("resize"))
      + "</b> resizes, so the measured cost is <b>" + AS.sig(measuredCost, 3)
      + "</b> elementary writes per append against a predicted g/(g−1) = " + AS.sig(predicted, 3)
      + ". Cross-check — recounting the copies from the capacity sequence alone gives "
      + AS.int(xcheck) + "; that and the counter " + AS.verdict(xcheck === cSel.get("copy"))
      + ". Final capacity " + AS.int(sel.cap) + " for " + AS.int(sel.num)
      + " live elements; measured mean slack <b>" + AS.sig(100 * sel.meanSlack, 1)
      + "%</b> against a worst-case 1 − 1/g = " + AS.sig(100 * (1 - 1 / gsel), 1) + "%."
    );
  }

  [els.g, els.n].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══ FIGURE 04 ═══════════════════════════════════════════════════════════
   Row-major layout and what traversal order does to the cache. The cache is
   simulated explicitly — fully associative, LRU, C lines of L elements — and
   every miss count is read off AL.counter(). The row-major count is checked
   against ceil(n²/L), which it must equal because a row-major sweep touches
   each line once, in order, and never returns to it.                        */
(function () {
  const svg = d3.select("#rm-svg"); if (svg.empty()) return;
  const W = 690, H = 348;

  const els = {
    n: d3.select("#rm-n"), nOut: d3.select("#rm-n-out"),
    L: d3.select("#rm-l"), C: d3.select("#rm-c"), cOut: d3.select("#rm-c-out"),
    order: d3.select("#rm-order"), out: d3.select("#rm-readout")
  };

  /* fully associative LRU over `C` lines of `L` elements; returns per-access
     hit/miss decisions as well as the total, both measured on the same run */
  function traverse(n, L, C, order, c) {
    const lru = [];                         // most-recently-used at the end
    const tape = [];
    const touch = (idx) => {
      const line = Math.floor(idx / L);
      const at = lru.indexOf(line);
      if (at >= 0) { lru.splice(at, 1); lru.push(line); c.add("hit"); tape.push(1); }
      else {
        lru.push(line); if (lru.length > C) lru.shift();
        c.add("miss"); tape.push(0);
      }
    };
    if (order === "row") {
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) touch(i * n + j);
    } else {
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) touch(i * n + j);
    }
    return tape;
  }

  function draw() {
    const n = +els.n.property("value"), L = +els.L.property("value");
    const C = +els.C.property("value"), order = els.order.property("value");
    els.nOut.text(n); els.cOut.text(C);

    const cRow = AL.counter(), cCol = AL.counter();
    const tapeRow = traverse(n, L, C, "row", cRow);
    const tapeCol = traverse(n, L, C, "col", cCol);
    const tape = order === "row" ? tapeRow : tapeCol;
    const missRow = cRow.get("miss"), missCol = cCol.get("miss");
    const closedRow = Math.ceil(n * n / L);      // the independent prediction

    const f = AL.frame(svg, W, H, { l: 16, r: 16, t: 14, b: 12 });
    const g = f.g;

    /* ── the matrix, coloured by which cache line each cell lives in ──── */
    g.append("text").attr("x", 0).attr("y", 4).attr("font-size", 11).attr("fill", AC.ink)
      .text("A[i][j] stored row-major — colour = cache line, number = linear offset i·n + j");
    const gw = 250, cw = gw / n, gy = 18;
    const hue = d3.scaleSequential(d3.interpolateCool).domain([0, Math.ceil(n * n / L)]);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const idx = i * n + j, line = Math.floor(idx / L);
      g.append("rect").attr("x", 14 + j * cw).attr("y", gy + i * cw)
        .attr("width", cw - 1).attr("height", cw - 1).attr("rx", 1.5)
        .attr("fill", hue(line)).attr("opacity", 0.8);
      if (n <= 10) {
        g.append("text").attr("x", 14 + j * cw + cw / 2).attr("y", gy + i * cw + cw / 2 + 3.5)
          .attr("text-anchor", "middle").attr("font-size", Math.min(10, cw * 0.42))
          .attr("fill", "#0f1117").text(idx);
      }
    }
    /* the traversal arrow */
    const gx1 = 14, gy1 = gy;
    if (order === "row") {
      AL.arrow(g, gx1 + 2, gy1 - 6, gx1 + gw - 4, gy1 - 6, { color: AC.a2, w: 1.6 });
      g.append("text").attr("x", gx1 + gw / 2).attr("y", gy1 + n * cw + 16).attr("text-anchor", "middle")
        .attr("font-size", 10.5).attr("fill", AC.a2).text("traversed along rows — with the storage");
    } else {
      AL.arrow(g, gx1 - 6, gy1 + 2, gx1 - 6, gy1 + n * cw - 4, { color: AC.bad, w: 1.6 });
      g.append("text").attr("x", gx1 + gw / 2).attr("y", gy1 + n * cw + 16).attr("text-anchor", "middle")
        .attr("font-size", 10.5).attr("fill", AC.bad).text("traversed down columns — across the storage");
    }

    /* ── the hit/miss tape ────────────────────────────────────────────── */
    const tx = 310, ty = 18, shown = Math.min(tape.length, 64), per = 16, sq = 13;
    g.append("text").attr("x", tx).attr("y", 4).attr("font-size", 11).attr("fill", AC.ink)
      .text("first " + shown + " accesses: miss or hit");
    for (let k = 0; k < shown; k++) {
      g.append("rect").attr("x", tx + (k % per) * (sq + 2)).attr("y", ty + Math.floor(k / per) * (sq + 2))
        .attr("width", sq).attr("height", sq).attr("rx", 2)
        .attr("fill", tape[k] ? AC.good : AC.bad).attr("opacity", tape[k] ? 0.55 : 0.9);
    }
    const tapeH = Math.ceil(shown / per) * (sq + 2);
    AL.legend(g, [{ label: "miss — a line is fetched", color: AC.bad },
                  { label: "hit — the line is already resident", color: AC.good }], tx, ty + tapeH + 16);

    /* ── the two totals, measured ─────────────────────────────────────── */
    const by = ty + tapeH + 62, bw = 300;
    const sc = d3.scaleLinear().domain([0, Math.max(missRow, missCol)]).range([0, bw - 90]);
    [{ lbl: "row-major sweep", v: missRow, col: AC.good },
     { lbl: "column-major sweep", v: missCol, col: AC.bad }].forEach((b, k) => {
      const yy = by + k * 24;
      g.append("text").attr("x", tx).attr("y", yy).attr("font-size", 11).attr("fill", AC.muted).text(b.lbl);
      g.append("rect").attr("x", tx).attr("y", yy + 5).attr("width", Math.max(2, sc(b.v)))
        .attr("height", 11).attr("rx", 2).attr("fill", b.col).attr("opacity", 0.85);
      g.append("text").attr("x", tx + Math.max(2, sc(b.v)) + 8).attr("y", yy + 15)
        .attr("font-size", 11).attr("fill", AC.ink).text(AS.int(b.v) + " misses");
    });

    els.out.html(
      "<b>n = " + n + "</b> (" + AS.int(n * n) + " elements), line = <b>" + L
      + "</b> elements, cache = <b>" + C + "</b> lines, LRU. Measured: the row-major sweep took <b>"
      + AS.int(missRow) + "</b> misses and " + AS.int(cRow.get("hit")) + " hits; the column-major sweep took <b>"
      + AS.int(missCol) + "</b> misses and " + AS.int(cCol.get("hit")) + " hits — a factor of <b>"
      + AS.sig(missCol / missRow, 2) + "×</b> more memory traffic for the identical <span class='keep'>Θ</span>(n²) work. "
      + "Cross-check — a row-major sweep must touch each of the ceil(n²/L) = " + AS.int(closedRow)
      + " lines exactly once, and the counter says " + AS.int(missRow) + ": " + AS.verdict(missRow === closedRow) + "."
    );
  }

  [els.n, els.L, els.C, els.order].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══ FIGURE 05 ═══════════════════════════════════════════════════════════
   Building a string of length n, two ways, with every character copy counted.
   Repeated concatenation on an immutable string rebuilds the whole prefix
   each time; collecting into a growable array and joining once does not.     */
(function () {
  const svg = d3.select("#cat-svg"); if (svg.empty()) return;
  const W = 690, H = 300;

  const els = {
    n: d3.select("#cat-n"), nOut: d3.select("#cat-n-out"),
    scale: d3.select("#cat-scale"), out: d3.select("#cat-readout")
  };

  /* strategy A: s = s + ch, n times, on an IMMUTABLE string. Each step
     allocates a fresh string and copies every character of the old one. */
  function buildByConcat(n, c) {
    let len = 0;
    for (let i = 0; i < n; i++) {
      c.add("alloc");
      c.add("copy", len);      // the existing characters are copied into the new buffer
      c.add("copy", 1);        // then the new character is written
      len += 1;
    }
    return len;
  }
  /* strategy B: append into a growable array, then join once. */
  function buildByJoin(n, c) {
    let cap = 1, num = 0;
    for (let i = 0; i < n; i++) {
      if (num === cap) { c.add("alloc"); c.add("copy", num); cap *= 2; }
      c.add("copy", 1); num++;
    }
    c.add("alloc");
    c.add("copy", n);          // join walks the pieces once into one buffer
    return n;
  }

  function draw() {
    const nSel = +els.n.property("value");
    const logy = els.scale.property("value") === "log";
    els.nOut.text(nSel);

    const ns = d3.range(4, 405, 4);
    const series = ns.map(n => {
      const a = AL.counter(), b = AL.counter();
      buildByConcat(n, a); buildByJoin(n, b);
      return { n, cat: a.get("copy"), join: b.get("copy") };
    });
    const aSel = AL.counter(), bSel = AL.counter();
    buildByConcat(nSel, aSel); buildByJoin(nSel, bSel);
    const catSel = aSel.get("copy"), joinSel = bSel.get("copy");
    const closed = nSel * (nSel + 1) / 2;         // independent closed form for A

    const f = AL.frame(svg, W, H, { l: 64, r: 18, t: 20, b: 40 });
    const g = f.g, iw = f.iw, ih = f.ih;
    const x = d3.scaleLinear().domain([0, 404]).range([0, iw]);
    const maxY = d3.max(series, d => d.cat);
    const y = logy ? d3.scaleLog().domain([1, maxY]).range([ih, 0])
                   : d3.scaleLinear().domain([0, maxY]).range([ih, 0]);
    AL.gridY(g, y, iw, 5);
    AL.axisB(g, x, ih, 6, "final string length n");
    AL.axisL(g, y, 5, "character copies, measured", d3.format("~s"));

    const ln = k => d3.line().x(d => x(d.n)).y(d => y(Math.max(logy ? 1 : 0, d[k])));
    g.append("path").datum(series).attr("fill", "none").attr("stroke", AC.bad)
      .attr("stroke-width", 2.2).attr("d", ln("cat"));
    g.append("path").datum(series).attr("fill", "none").attr("stroke", AC.good)
      .attr("stroke-width", 2.2).attr("d", ln("join"));
    /* the closed form for strategy A, drawn from the formula rather than the run */
    g.append("path").datum(ns).attr("fill", "none").attr("stroke", AC.a2)
      .attr("stroke-width", 1.4).attr("stroke-dasharray", "5 4")
      .attr("d", d3.line().x(d => x(d)).y(d => y(Math.max(logy ? 1 : 0, d * (d + 1) / 2))));

    g.append("circle").attr("cx", x(nSel)).attr("cy", y(Math.max(logy ? 1 : 0, catSel)))
      .attr("r", 4).attr("fill", AC.bad);
    g.append("circle").attr("cx", x(nSel)).attr("cy", y(Math.max(logy ? 1 : 0, joinSel)))
      .attr("r", 4).attr("fill", AC.good);

    AL.legend(g, [
      { label: "s = s + ch, repeated — measured", color: AC.bad },
      { label: "closed form n(n+1)/2", color: AC.a2, dash: "5 4" },
      { label: "append to an array, then join once — measured", color: AC.good }
    ], 14, 16);

    els.out.html(
      "At <b>n = " + nSel + "</b>: repeated concatenation copied <b>" + AS.int(catSel)
      + "</b> characters across " + AS.int(aSel.get("alloc")) + " allocations; collect-and-join copied <b>"
      + AS.int(joinSel) + "</b> across " + AS.int(bSel.get("alloc")) + " — a ratio of <b>"
      + AS.sig(catSel / joinSel, 1) + "×</b>, and the ratio grows linearly in n. "
      + "Cross-check — the closed form n(n+1)/2 = " + AS.int(closed) + " and the counter "
      + AS.verdict(closed === catSel) + ". "
      + "Per character the quadratic method costs " + AS.sig(catSel / nSel, 1)
      + " copies against " + AS.sig(joinSel / nSel, 2) + " for the linear one."
    );
  }

  [els.n, els.scale].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══ FIGURE 06 ═══════════════════════════════════════════════════════════
   Two pointers. The frames come from running the real routine and snapshotting
   it at every decision; the counters come from the same run, so the readout is
   a measurement of exactly the trace on screen. The invariant is printed at
   every step because the invariant is the reason the routine is correct.     */
(function () {
  const svg = d3.select("#tp-svg"); if (svg.empty()) return;
  const W = 690, H = 210;

  const sel = d3.select("#tp-mode"), out = d3.select("#tp-readout");
  let st = null;

  function reverse(input, c, snap) {
    const A = [...input];
    let lo = 0, hi = A.length - 1;
    snap({ a: [...A], lo, hi, note: "start: nothing is in its final place yet" });
    while (lo < hi) {
      c.add("cmp");
      const t = A[lo]; A[lo] = A[hi]; A[hi] = t;
      c.add("swap"); c.add("write", 2);
      snap({ a: [...A], lo, hi, done: true,
             note: "swap A[" + lo + "] and A[" + hi + "] — both are now final" });
      lo++; hi--;
      c.add("move", 2);
      snap({ a: [...A], lo, hi, note: "close in: lo → " + lo + ", hi → " + hi });
    }
    c.add("cmp");
    snap({ a: [...A], lo, hi, fin: true, note: lo === hi
           ? "lo == hi: the middle element is already where it belongs — done"
           : "lo > hi: the pointers have crossed — done" });
    return A;
  }

  function twoSum(input, target, c, snap) {
    const A = [...input];
    let lo = 0, hi = A.length - 1;
    while (lo < hi) {
      c.add("cmp");
      const s = A[lo] + A[hi];
      c.add("add");
      if (s === target) {
        snap({ a: [...A], lo, hi, hit: true, sum: s,
               note: "A[" + lo + "] + A[" + hi + "] = " + A[lo] + " + " + A[hi] + " = " + s + " — found" });
        return [lo, hi];
      }
      if (s < target) {
        snap({ a: [...A], lo, hi, sum: s,
               note: "sum " + s + " &lt; " + target + " — the largest partner of A[" + lo + "] is already too small, so A[" + lo + "] is in no solution: lo++" });
        lo++; c.add("move");
      } else {
        snap({ a: [...A], lo, hi, sum: s,
               note: "sum " + s + " &gt; " + target + " — the smallest partner of A[" + hi + "] is already too big, so A[" + hi + "] is in no solution: hi−−" });
        hi--; c.add("move");
      }
    }
    c.add("cmp");
    snap({ a: [...A], lo, hi, fin: true, note: "the pointers crossed with no pair found" });
    return null;
  }

  const MODES = {
    reverse: {
      data: [3, 1, 4, 1, 5, 9, 2, 6],
      title: "reverse in place",
      inv: "A[0 … lo−1] and A[hi+1 … n−1] already hold their final, mirrored values; A[lo … hi] is untouched.",
      run: (c, snap) => reverse(MODES.reverse.data, c, snap)
    },
    twosum: {
      data: [1, 3, 4, 6, 8, 10, 13, 15], target: 17,
      title: "find a pair summing to 17 in a sorted array",
      inv: "If any pair summing to the target exists at all, both of its elements lie inside A[lo … hi].",
      run: (c, snap) => twoSum(MODES.twosum.data, MODES.twosum.target, c, snap)
    }
  };

  function build() {
    const key = sel.property("value"), M = MODES[key];
    const frames = [], c = AL.counter(), running = [];
    M.run(c, f => { frames.push(f); running.push(c.all()); });
    frames.forEach((f, i) => { f.counts = running[i]; });
    const tot = c.all();

    if (st) st.pause();
    svg.selectAll("*").remove();
    AS.clearControls(svg.node());

    st = AL.stepper(svg, {
      frames, delay: 750, label: "step",
      render: (f) => render(f, M, key)
    });

    const n = M.data.length;
    out.html(
      "<b>" + M.title + "</b> — measured over the whole run: <b>" + (tot.cmp || 0)
      + "</b> pointer-order tests, <b>" + (tot.add || 0) + "</b> sum evaluations, <b>"
      + (tot.swap || 0) + "</b> swaps, <b>" + (tot.move || 0) + "</b> pointer advances on an array of "
      + n + ". Each advance moves lo right or hi left by one and they never move back, so the two "
      + "pointers together take at most n − 1 steps before crossing: the loop is <b><span class='keep'>Θ</span>(n)</b> "
      + "with <b>O(1)</b> extra space. "
      + (key === "reverse"
         ? "Cross-check — the two pointers advanced " + (tot.move || 0)
           + " times in total, which should be 2·floor(n/2) = " + (2 * Math.floor(n / 2)) + ": "
           + AS.verdict((tot.move || 0) === 2 * Math.floor(n / 2)) + "."
         : "Cross-check — the scan evaluated " + (tot.add || 0) + " sums out of the n(n−1)/2 = "
           + (n * (n - 1) / 2) + " pairs a double loop would have tried, and " + (tot.add || 0)
           + " ≤ n − 1 = " + (n - 1) + ": " + AS.verdict((tot.add || 0) <= n - 1) + ".")
    );
  }

  function render(f, M, key) {
    const fr = AL.frame(svg, W, H, { l: 18, r: 18, t: 16, b: 10 });
    const g = fr.g;
    const n = f.a.length;

    g.append("text").attr("x", 0).attr("y", 4).attr("font-size", 11).attr("fill", AC.muted)
      .text(M.title);

    const r = AL.row(g, f.a, {
      x: 28, y: 26, w: 56, h: 34, gap: 3, index: true, fontSize: 14,
      mark: i => {
        if (i === f.lo && i === f.hi) return AC.violet;
        if (i === f.lo) return AC.accent;
        if (i === f.hi) return AC.a2;
        if (key === "reverse" && (i < f.lo || i > f.hi)) return "#1f3a2b";
        if (key === "twosum" && (i < f.lo || i > f.hi)) return AC.panel;
        return null;
      }
    });

    const yArrow = 84;
    if (f.lo >= 0 && f.lo < n) {
      AL.arrow(g, 28 + r.cellX(f.lo), yArrow + 22, 28 + r.cellX(f.lo), yArrow, { color: AC.accent, w: 1.6 });
      g.append("text").attr("x", 28 + r.cellX(f.lo)).attr("y", yArrow + 36).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("fill", AC.accent).text("lo");
    }
    if (f.hi >= 0 && f.hi < n) {
      AL.arrow(g, 28 + r.cellX(f.hi), yArrow + 22, 28 + r.cellX(f.hi), yArrow, { color: AC.a2, w: 1.6 });
      g.append("text").attr("x", 28 + r.cellX(f.hi)).attr("y", yArrow + 50).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("fill", AC.a2).text("hi");
    }

    g.append("text").attr("x", 0).attr("y", 152).attr("font-size", 11.5)
      .attr("fill", f.hit ? AC.good : AC.ink).html(f.note);
    g.append("text").attr("x", 0).attr("y", 176).attr("font-size", 11).attr("fill", AC.muted)
      .text("invariant:");
    g.append("text").attr("x", 62).attr("y", 176).attr("font-size", 11).attr("fill", AC.violet)
      .text(M.inv);
  }

  sel.on("change", build);
  build();
})();

/* ══ FIGURE 07 ═══════════════════════════════════════════════════════════
   The sliding window. Same instrumentation discipline: the frames and the
   counts are one run. The point the readout makes is the amortized one —
   the inner loop looks nested but each index is admitted once and evicted at
   most once, so the total pointer movement is bounded by 2n.                 */
(function () {
  const svg = d3.select("#win-svg"); if (svg.empty()) return;
  const W = 690, H = 250;

  const sel = d3.select("#win-mode"), out = d3.select("#win-readout");
  let st = null;

  /* longest window with no repeated element */
  function longestDistinct(seq, c, snap) {
    const seen = new Map();
    let l = 0, best = 0, bl = 0, br = -1;
    for (let r = 0; r < seq.length; r++) {
      c.add("radv");
      const ch = seq[r];
      while (seen.has(ch) && seen.get(ch) >= l) {
        c.add("ladv");
        seen.delete(seq[l]);
        l++;
        snap({ seq, l, r, best, bl, br, drop: true,
               note: "'" + ch + "' is already inside the window — evict from the left, l → " + l });
      }
      seen.set(ch, r);
      if (r - l + 1 > best) { best = r - l + 1; bl = l; br = r; }
      snap({ seq, l, r, best, bl, br,
             note: "admit '" + ch + "' at " + r + "; window [" + l + ", " + r + "] has length "
                   + (r - l + 1) + ", best so far " + best });
    }
    return { best, bl, br };
  }

  /* shortest window whose sum is at least the target */
  function shortestAtLeast(seq, target, c, snap) {
    let l = 0, sum = 0, best = Infinity, bl = 0, br = -1;
    for (let r = 0; r < seq.length; r++) {
      c.add("radv"); sum += seq[r]; c.add("add");
      snap({ seq, l, r, best: best === Infinity ? 0 : best, bl, br, sum,
             note: "admit " + seq[r] + " at " + r + "; window sum is " + sum });
      while (sum >= target) {
        if (r - l + 1 < best) { best = r - l + 1; bl = l; br = r; }
        snap({ seq, l, r, best, bl, br, sum, hit: true,
               note: "sum " + sum + " ≥ " + target + " — record length " + (r - l + 1)
                     + ", then shrink from the left" });
        sum -= seq[l]; c.add("add"); l++; c.add("ladv");
      }
    }
    return { best: best === Infinity ? 0 : best, bl, br };
  }

  const MODES = {
    distinct: {
      seq: "abcabcbb".split(""), title: "longest window with no repeated character — abcabcbb",
      inv: "The window seq[l … r] contains no repeated element.",
      run: (c, snap) => longestDistinct(MODES.distinct.seq, c, snap)
    },
    distinct2: {
      seq: "pwwkew".split(""), title: "longest window with no repeated character — pwwkew",
      inv: "The window seq[l … r] contains no repeated element.",
      run: (c, snap) => longestDistinct(MODES.distinct2.seq, c, snap)
    },
    atleast: {
      seq: [2, 3, 1, 2, 4, 3], target: 7, title: "shortest window with sum ≥ 7 — [2, 3, 1, 2, 4, 3]",
      inv: "l is the smallest left edge for which seq[l … r] still has sum ≥ target, once the shrink loop stops.",
      run: (c, snap) => shortestAtLeast(MODES.atleast.seq, MODES.atleast.target, c, snap)
    }
  };

  function build() {
    const key = sel.property("value"), M = MODES[key];
    const frames = [], c = AL.counter();
    const res = M.run(c, f => frames.push(f));

    if (st) st.pause();
    svg.selectAll("*").remove();
    AS.clearControls(svg.node());
    st = AL.stepper(svg, { frames, delay: 780, label: "step", render: f => render(f, M) });

    const n = M.seq.length, tot = c.all();
    const moves = (tot.radv || 0) + (tot.ladv || 0);
    out.html(
      "<b>" + M.title + "</b> — answer <b>" + (key === "atleast" ? res.best + " elements" : res.best + " characters")
      + "</b>, at [" + res.bl + ", " + res.br + "]. Measured on this run: the right edge advanced <b>"
      + (tot.radv || 0) + "</b> times and the left edge <b>" + (tot.ladv || 0)
      + "</b>, for <b>" + moves + "</b> pointer moves on a sequence of length " + n + ". "
      + "The inner <code>while</code> makes the code look quadratic; it is not, because <b>every index is "
      + "admitted exactly once and evicted at most once</b>, so the two edges together move at most 2n = "
      + (2 * n) + " times. Cross-check — measured " + moves + " ≤ " + (2 * n) + ": "
      + AS.verdict(moves <= 2 * n) + ". A double loop over all "
      + (n * (n + 1) / 2) + " sub-windows would be <span class='keep'>Θ</span>(n²)."
    );
  }

  function render(f, M) {
    const fr = AL.frame(svg, W, H, { l: 18, r: 18, t: 16, b: 10 });
    const g = fr.g;
    g.append("text").attr("x", 0).attr("y", 4).attr("font-size", 11).attr("fill", AC.muted).text(M.title);

    const n = f.seq.length;
    const r = AL.row(g, f.seq, {
      x: 30, y: 24, w: Math.min(56, 600 / n), h: 34, gap: 3, index: true, fontSize: 14,
      mark: i => (i >= f.l && i <= f.r) ? (f.hit ? AC.good : AC.accent) : (i < f.l ? AC.panel : null)
    });

    /* the window bracket */
    if (f.r >= f.l) {
      const x1 = 30 + r.cellX(f.l) - r.cellW / 2, x2 = 30 + r.cellX(f.r) + r.cellW / 2;
      g.append("path").attr("d", "M" + x1 + ",78 L" + x1 + ",86 L" + x2 + ",86 L" + x2 + ",78")
        .attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 1.6);
      g.append("text").attr("x", (x1 + x2) / 2).attr("y", 100).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("fill", AC.a2)
        .text("window [" + f.l + ", " + f.r + "], length " + (f.r - f.l + 1)
              + (f.sum !== undefined ? ", sum " + f.sum : ""));
    }
    /* the best window so far */
    if (f.br >= f.bl && f.br >= 0) {
      const x1 = 30 + r.cellX(f.bl) - r.cellW / 2, x2 = 30 + r.cellX(f.br) + r.cellW / 2;
      g.append("line").attr("x1", x1).attr("x2", x2).attr("y1", 118).attr("y2", 118)
        .attr("stroke", AC.good).attr("stroke-width", 3).attr("opacity", 0.8);
      g.append("text").attr("x", x2 + 8).attr("y", 122).attr("font-size", 10.5).attr("fill", AC.good)
        .text("best so far: " + f.best);
    }

    g.append("text").attr("x", 0).attr("y", 158).attr("font-size", 11.5)
      .attr("fill", f.drop ? AC.bad : AC.ink).html(f.note);
    g.append("text").attr("x", 0).attr("y", 184).attr("font-size", 11).attr("fill", AC.muted).text("invariant:");
    g.append("text").attr("x", 62).attr("y", 184).attr("font-size", 11).attr("fill", AC.violet).text(M.inv);
  }

  sel.on("change", build);
  build();
})();

/* ══ FIGURE 08 ═══════════════════════════════════════════════════════════
   Prefix sums: one linear pass buys constant-time range queries forever. The
   figure answers the selected query both ways — by the subtraction and by an
   honest loop over the range — and prints both answers and both measured
   addition counts, so the identity is checked rather than asserted.          */
(function () {
  const svg = d3.select("#pre-svg"); if (svg.empty()) return;
  const W = 690, H = 286;
  const A = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3], N = A.length;

  const els = {
    l: d3.select("#pre-l"), lOut: d3.select("#pre-l-out"),
    r: d3.select("#pre-r"), rOut: d3.select("#pre-r-out"),
    m: d3.select("#pre-m"), mOut: d3.select("#pre-m-out"),
    out: d3.select("#pre-readout")
  };

  function buildPrefix(a, c) {
    const P = new Array(a.length + 1); P[0] = 0;
    for (let i = 0; i < a.length; i++) { c.add("add"); P[i + 1] = P[i] + a[i]; }
    return P;
  }
  function queryPrefix(P, l, r, c) { c.add("sub"); return P[r + 1] - P[l]; }
  function queryDirect(a, l, r, c) {
    let s = 0;
    for (let i = l; i <= r; i++) { c.add("add"); s += a[i]; }
    return s;
  }

  function draw() {
    let l = +els.l.property("value"), r = +els.r.property("value");
    if (r < l) { r = l; els.r.property("value", r); }
    const m = +els.m.property("value");
    els.lOut.text(l); els.rOut.text(r); els.mOut.text(m);

    const cB = AL.counter();
    const P = buildPrefix(A, cB);
    const cQ = AL.counter(), cD = AL.counter();
    const viaPrefix = queryPrefix(P, l, r, cQ);
    const direct = queryDirect(A, l, r, cD);

    /* the same m random queries answered both ways, with a fixed seed */
    const rng = AL.rng(7), cMP = AL.counter(), cMD = AL.counter();
    const cMB = AL.counter();
    const P2 = buildPrefix(A, cMB);
    let mismatch = 0;
    for (let k = 0; k < m; k++) {
      const a1 = AL.randInt(rng, 0, N - 1), b1 = AL.randInt(rng, 0, N - 1);
      const lo = Math.min(a1, b1), hi = Math.max(a1, b1);
      const x = queryPrefix(P2, lo, hi, cMP), y = queryDirect(A, lo, hi, cMD);
      if (x !== y) mismatch++;
    }
    const prefixTotal = cMB.get("add") + cMP.get("sub");
    const directTotal = cMD.get("add");

    const f = AL.frame(svg, W, H, { l: 18, r: 18, t: 16, b: 10 });
    const g = f.g;
    const cw = 54, gap = 3, x0 = 56;

    g.append("text").attr("x", 0).attr("y", 16).attr("text-anchor", "start")
      .attr("font-size", 11).attr("fill", AC.muted).text("A");
    const rA = AL.row(g, A, {
      x: x0 + (cw + gap) / 2, y: 4, w: cw, h: 32, gap, index: true, fontSize: 13,
      mark: i => (i >= l && i <= r) ? AC.a2 : null
    });

    g.append("text").attr("x", 0).attr("y", 90).attr("font-size", 11).attr("fill", AC.muted).text("P");
    const rP = AL.row(g, P, {
      x: x0, y: 78, w: cw, h: 32, gap, index: true, fontSize: 13,
      mark: i => (i === l) ? AC.bad : (i === r + 1 ? AC.good : (i > l && i < r + 1 ? AC.panel : null))
    });

    /* the recurrence, drawn as a link from P[i] to P[i+1] */
    g.append("text").attr("x", x0).attr("y", 132).attr("font-size", 10.5).attr("fill", AC.muted)
      .text("P[0] = 0,  P[i+1] = P[i] + A[i]  — one pass, " + cB.get("add") + " additions, measured");

    /* the query, spelled out */
    const xl = x0 + rP.cellX(l), xr = x0 + rP.cellX(r + 1);
    g.append("path").attr("d", "M" + xl + ",154 L" + xl + ",164 L" + xr + ",164 L" + xr + ",154")
      .attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 1.5);
    g.append("text").attr("x", (xl + xr) / 2).attr("y", 182).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", AC.ink)
      .text("sum A[" + l + " … " + r + "] = P[" + (r + 1) + "] − P[" + l + "] = "
            + P[r + 1] + " − " + P[l] + " = " + viaPrefix);

    /* the two costs, as bars, over m queries */
    const by = 212, bw = 420;
    const sc = d3.scaleLinear().domain([0, Math.max(prefixTotal, directTotal)]).range([0, bw]);
    [{ lbl: "prefix: build " + cMB.get("add") + " + " + cMP.get("sub") + " subtractions", v: prefixTotal, col: AC.good },
     { lbl: "direct: re-add every range", v: directTotal, col: AC.bad }].forEach((b, k) => {
      const yy = by + k * 26;
      g.append("text").attr("x", 0).attr("y", yy).attr("font-size", 10.5).attr("fill", AC.muted).text(b.lbl);
      g.append("rect").attr("x", 0).attr("y", yy + 4).attr("width", Math.max(2, sc(b.v)))
        .attr("height", 11).attr("rx", 2).attr("fill", b.col).attr("opacity", 0.85);
      g.append("text").attr("x", Math.max(2, sc(b.v)) + 8).attr("y", yy + 14)
        .attr("font-size", 11).attr("fill", AC.ink).text(AS.int(b.v) + " arithmetic operations");
    });

    els.out.html(
      "Query A[" + l + " … " + r + "]: the subtraction gives <b>" + viaPrefix
      + "</b> using a measured <b>" + cQ.get("sub") + "</b> operation; summing the range honestly gives <b>"
      + direct + "</b> using a measured <b>" + cD.get("add") + "</b> additions. The two answers "
      + AS.verdict(viaPrefix === direct) + ". "
      + "Over <b>" + m + "</b> random queries on n = " + N + ": prefix sums cost <b>" + AS.int(prefixTotal)
      + "</b> operations (" + cMB.get("add") + " to build, one per query thereafter) against <b>"
      + AS.int(directTotal) + "</b> for direct summation, and all " + m + " answers matched ("
      + mismatch + " mismatches). The crossover is immediate: <span class='keep'>Θ</span>(n + q) beats "
      + "<span class='keep'>Θ</span>(q·n) as soon as q is more than a small constant."
    );
  }

  [els.l, els.r, els.m].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══ FIGURE 09 ═══════════════════════════════════════════════════════════
   The Dutch national flag partition: four regions, three pointers, one pass.
   Frames and counts are one run; the readout re-sorts the input independently
   and compares, so "the output is correct" is a check and not a claim.       */
(function () {
  const svg = d3.select("#dnf-svg"); if (svg.empty()) return;
  const W = 690, H = 252;

  const sel = d3.select("#dnf-input"), out = d3.select("#dnf-readout");
  const PRESETS = {
    mixed: [1, 0, 2, 1, 2, 0, 0, 1, 2],
    reverse: [2, 2, 2, 1, 1, 1, 0, 0, 0],
    sorted: [0, 0, 0, 1, 1, 1, 2, 2, 2],
    allmid: [1, 1, 1, 1, 1, 1, 1, 1, 1]
  };
  let st = null;

  function dnf(input, c, snap) {
    const A = [...input];
    let lo = 0, i = 0, hi = A.length - 1;
    const swap = (p, q) => { const t = A[p]; A[p] = A[q]; A[q] = t; c.add("swap"); };
    snap({ a: [...A], lo, i, hi, note: "start: everything is unexamined" });
    while (i <= hi) {
      c.add("cmp");
      if (A[i] === 0) {
        swap(lo, i);
        snap({ a: [...A], lo, i, hi, act: "lo",
               note: "A[i] = 0 → swap it into the 0-region, lo++ and i++" });
        lo++; i++;
      } else if (A[i] === 1) {
        snap({ a: [...A], lo, i, hi, act: "mid",
               note: "A[i] = 1 → it already belongs where it is, i++" });
        i++;
      } else {
        swap(i, hi);
        snap({ a: [...A], lo, i, hi, act: "hi",
               note: "A[i] = 2 → swap it to the 2-region, hi−− and DO NOT advance i — the value swapped in is unexamined" });
        hi--;
      }
    }
    c.add("cmp");
    snap({ a: [...A], lo, i, hi, fin: true, note: "i &gt; hi: the unexamined region is empty — done" });
    return A;
  }

  function build() {
    const key = sel.property("value"), input = PRESETS[key];
    const frames = [], c = AL.counter();
    const res = dnf(input, c, f => frames.push(f));

    if (st) st.pause();
    svg.selectAll("*").remove();
    AS.clearControls(svg.node());
    st = AL.stepper(svg, { frames, delay: 720, label: "step", render: f => render(f) });

    /* independent checks: the output is sorted, and it is a permutation */
    const sortedOK = res.every((v, i) => i === 0 || res[i - 1] <= v);
    const tally = a => a.reduce((m, v) => (m[v] = (m[v] || 0) + 1, m), {});
    const t1 = tally(input), t2 = tally(res);
    const sameMultiset = [0, 1, 2].every(k => (t1[k] || 0) === (t2[k] || 0));
    const n = input.length;

    out.html(
      "Input [" + input.join(", ") + "] → output [" + res.join(", ") + "]. Measured: <b>"
      + c.get("cmp") + "</b> value tests and <b>" + c.get("swap") + "</b> swaps on n = " + n
      + " elements, in <b>one pass</b> with O(1) extra space. Cross-check — the output is non-decreasing: "
      + AS.verdict(sortedOK) + "; and it holds the same multiset as the input ("
      + (t1[0] || 0) + " zeros, " + (t1[1] || 0) + " ones, " + (t1[2] || 0) + " twos): "
      + AS.verdict(sameMultiset) + ". Every test either advances i or retreats hi, and i + (n − 1 − hi) "
      + "increases by one each time, so the loop runs at most n + 1 = " + (n + 1)
      + " times: measured " + c.get("cmp") + " ≤ " + (n + 1) + ", " + AS.verdict(c.get("cmp") <= n + 1) + "."
    );
  }

  const COL = { 0: AC.bad, 1: AC.muted, 2: AC.accent };

  function render(f) {
    const fr = AL.frame(svg, W, H, { l: 18, r: 18, t: 16, b: 10 });
    const g = fr.g, n = f.a.length;

    const r = AL.row(g, f.a, {
      x: 34, y: 34, w: 52, h: 34, gap: 3, index: true, fontSize: 14,
      mark: i => {
        if (i < f.lo) return COL[0];
        if (i > f.hi) return COL[2];
        if (i < f.i) return "#3a4150";
        return AC.panel;
      }
    });

    /* region labels */
    const span = (a, b, label, col, y) => {
      if (b < a) return;
      const x1 = 34 + r.cellX(a) - r.cellW / 2, x2 = 34 + r.cellX(b) + r.cellW / 2;
      g.append("line").attr("x1", x1).attr("x2", x2).attr("y1", y).attr("y2", y)
        .attr("stroke", col).attr("stroke-width", 2.5).attr("opacity", 0.9);
      g.append("text").attr("x", (x1 + x2) / 2).attr("y", y - 6).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", col).text(label);
    };
    span(0, f.lo - 1, "all 0", COL[0], 24);
    span(f.lo, f.i - 1, "all 1", AC.ink, 24);
    span(f.i, f.hi, "unexamined", AC.a2, 24);
    span(f.hi + 1, n - 1, "all 2", COL[2], 24);

    const ptr = (idx, name, col, dy) => {
      if (idx < 0 || idx >= n) return;
      AL.arrow(g, 34 + r.cellX(idx), 92 + dy, 34 + r.cellX(idx), 74, { color: col, w: 1.5 });
      g.append("text").attr("x", 34 + r.cellX(idx)).attr("y", 106 + dy).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("fill", col).text(name);
    };
    ptr(f.lo, "lo", COL[0], 0);
    ptr(f.i, "i", AC.a2, 16);
    ptr(f.hi, "hi", COL[2], 32);

    g.append("text").attr("x", 0).attr("y", 178).attr("font-size", 11.5)
      .attr("fill", f.fin ? AC.good : AC.ink).html(f.note);
    g.append("text").attr("x", 0).attr("y", 204).attr("font-size", 11).attr("fill", AC.muted).text("invariant:");
    g.append("text").attr("x", 62).attr("y", 204).attr("font-size", 11).attr("fill", AC.violet)
      .text("A[0 … lo−1] are all 0 · A[lo … i−1] are all 1 · A[hi+1 … n−1] are all 2 · A[i … hi] not yet seen");
  }

  sel.on("change", build);
  build();
})();

/* ── shared string-matcher routines, all instrumented with AL.counter() ───
   Used by figures 10, 11, 12 and 13. Each counts "cmp" once per CHARACTER
   comparison actually performed, so the four are measured on the same unit
   and the race in figure 13 is a fair one.                                  */
const MATCH = {
  naive: function (T, P, c, snap) {
    const n = T.length, m = P.length, found = [];
    for (let s = 0; s + m <= n; s++) {
      let j = 0;
      for (; j < m; j++) {
        c.add("cmp");
        const ok = T[s + j] === P[j];
        if (snap) snap({ s, j, ok, n, m });
        if (!ok) break;
      }
      if (j === m) { found.push(s); if (snap) snap({ s, j: m, ok: true, hit: true, n, m }); }
    }
    return found;
  },

  /* the prefix function, 0-indexed: pi[q] is the length of the longest proper
     prefix of P that is also a suffix of P[0 .. q] */
  prefixFunction: function (P, c, snap) {
    const m = P.length, pi = new Array(m).fill(0);
    let k = 0;
    if (snap) snap({ q: 0, k: 0, pi: [...pi], set: 0, note: "pi[0] = 0 always — a one-character prefix has no PROPER prefix that is also a suffix" });
    for (let q = 1; q < m; q++) {
      while (k > 0 && P[k] !== P[q]) {
        c.add("cmp");
        if (snap) snap({ q, k, pi: [...pi], fail: true,
                         note: "P[" + k + "] = '" + P[k] + "' ≠ P[" + q + "] = '" + P[q]
                               + "' — fall back: k → pi[" + (k - 1) + "] = " + pi[k - 1] });
        k = pi[k - 1];
      }
      c.add("cmp");
      const eq = P[k] === P[q];
      if (eq) k++;
      pi[q] = k;
      if (snap) snap({ q, k, pi: [...pi], set: q, match: eq,
                       note: eq ? "P[" + (k - 1) + "] = '" + P[q] + "' = P[" + q + "] — extend: k → " + k + ", so pi[" + q + "] = " + k
                                : "no extension possible with k = 0, so pi[" + q + "] = 0" });
    }
    return pi;
  },

  /* brute-force prefix function, used only as an independent cross-check */
  prefixBrute: function (P) {
    const m = P.length, pi = new Array(m).fill(0);
    for (let q = 0; q < m; q++) {
      for (let k = q; k >= 1; k--) {
        let ok = true;
        for (let t = 0; t < k; t++) if (P[t] !== P[q - k + 1 + t]) { ok = false; break; }
        if (ok) { pi[q] = k; break; }
      }
    }
    return pi;
  },

  kmp: function (T, P, c, snap, cPre) {
    const n = T.length, m = P.length, found = [];
    const pi = MATCH.prefixFunction(P, cPre || c, null);
    let q = 0;
    for (let i = 0; i < n; i++) {
      while (q > 0 && P[q] !== T[i]) { c.add("cmp"); q = pi[q - 1]; }
      c.add("cmp");
      if (P[q] === T[i]) q++;
      if (snap) snap({ i, q, n, m, pi });
      if (q === m) { found.push(i - m + 1); q = pi[q - 1]; }
    }
    return { found, pi };
  },

  rabinKarp: function (T, P, d, qm, c, snap, valFn) {
    const val = valFn || (ch => +ch);
    const n = T.length, m = P.length, found = [], spurious = [], hashes = [];
    let h = 1;
    for (let i = 0; i < m - 1; i++) h = (h * d) % qm;
    let p = 0, t = 0;
    for (let i = 0; i < m; i++) {
      p = (d * p + val(P[i])) % qm;
      t = (d * t + val(T[i])) % qm;
      c.add("hashop", 2);
    }
    for (let s = 0; s + m <= n; s++) {
      hashes.push(t);
      let hit = false, ok = false;
      if (p === t) {
        hit = true; ok = true;
        for (let j = 0; j < m; j++) { c.add("cmp"); if (T[s + j] !== P[j]) { ok = false; break; } }
        if (ok) found.push(s); else spurious.push(s);
      }
      if (snap) snap({ s, t, p, hit, ok, n, m, h });
      if (s + m < n) {
        t = (d * (t - val(T[s]) * h) + val(T[s + m])) % qm;
        if (t < 0) t += qm;
        c.add("hashop", 4);
      }
    }
    return { found, spurious, hashes };
  },

  /* simplified Boyer-Moore: looking-glass scan plus the last-occurrence
     (bad character) jump. Counts the same unit as the others. */
  boyerMoore: function (T, P, c, snap) {
    const n = T.length, m = P.length, found = [];
    if (m === 0) return found;
    const last = new Map();
    for (let k = 0; k < m; k++) last.set(P[k], k);
    let i = m - 1, k = m - 1;
    while (i < n) {
      c.add("cmp");
      if (T[i] === P[k]) {
        if (snap) snap({ i, k, n, m, ok: true });
        if (k === 0) { found.push(i); i += m; k = m - 1; }
        else { i--; k--; }
      } else {
        if (snap) snap({ i, k, n, m, ok: false });
        const j = last.has(T[i]) ? last.get(T[i]) : -1;
        i += m - Math.min(k, j + 1);
        k = m - 1;
      }
    }
    return found;
  }
};

/* ══ FIGURE 10 ═══════════════════════════════════════════════════════════
   The naive matcher, stepped. Presets include the adversarial pair whose cost
   is exactly (n − m + 1)·m, which the readout checks against the counter.    */
(function () {
  const svg = d3.select("#naive-svg"); if (svg.empty()) return;
  const W = 690, H = 210;

  const sel = d3.select("#naive-input"), out = d3.select("#naive-readout");
  const PRESETS = {
    canon: { T: "abcabaabcabac", P: "abaa", label: "one occurrence, mixed alphabet" },
    small: { T: "acaabc", P: "aab", label: "four alignments, one match" },
    worst: { T: "aaaaaaaaaa", P: "aaab", label: "the adversarial pair — every alignment matches m−1 characters before failing" }
  };
  let st = null;

  function build() {
    const key = sel.property("value"), M = PRESETS[key];
    const frames = [], c = AL.counter();
    const found = MATCH.naive(M.T, M.P, c, f => frames.push(f));

    if (st) st.pause();
    svg.selectAll("*").remove();
    AS.clearControls(svg.node());
    st = AL.stepper(svg, { frames, delay: 420, label: "comparison", render: f => render(f, M) });

    const n = M.T.length, m = M.P.length, shifts = n - m + 1;
    const worstCase = shifts * m;
    out.html(
      "T = <code>" + M.T + "</code> (n = " + n + "), P = <code>" + M.P + "</code> (m = " + m + "). "
      + "Measured: <b>" + c.get("cmp") + "</b> character comparisons over " + shifts
      + " alignments, finding " + found.length + " occurrence" + (found.length === 1 ? "" : "s")
      + (found.length ? " at shift" + (found.length === 1 ? " " : "s ") + found.join(", ") : "")
      + ". The worst possible for these lengths is (n − m + 1)·m = " + shifts + " × " + m + " = "
      + worstCase + ", and this run used <b>" + AS.sig(100 * c.get("cmp") / worstCase, 0)
      + "%</b> of it"
      + (key === "worst" ? " — " + AS.verdict(c.get("cmp") === worstCase)
           + " with the claim that this pair attains the worst case exactly." : ".")
      + " No preprocessing, no memory: what the matcher learns at one alignment it throws away before the next."
    );
  }

  function render(f, M) {
    const fr = AL.frame(svg, W, H, { l: 18, r: 18, t: 16, b: 10 });
    const g = fr.g;
    const T = M.T.split(""), P = M.P.split("");
    const cw = Math.min(40, 620 / T.length);

    g.append("text").attr("x", 0).attr("y", 16).attr("font-size", 11).attr("fill", AC.muted).text("T");
    const rT = AL.row(g, T, {
      x: 26, y: 4, w: cw, h: 30, gap: 2, index: true, fontSize: 13,
      mark: i => {
        if (i === f.s + f.j) return f.ok ? AC.good : AC.bad;
        if (i >= f.s && i < f.s + f.j) return "#1f3a2b";
        if (i >= f.s && i < f.s + f.m) return AC.panel;
        return null;
      }
    });

    g.append("text").attr("x", 0).attr("y", 90).attr("font-size", 11).attr("fill", AC.muted).text("P");
    AL.row(g, P, {
      x: 26 + f.s * (cw + 2), y: 78, w: cw, h: 30, gap: 2, index: false, fontSize: 13,
      mark: i => {
        if (i === f.j) return f.ok ? AC.good : AC.bad;
        if (i < f.j) return "#1f3a2b";
        return null;
      }
    });

    g.append("text").attr("x", 0).attr("y", 140).attr("font-size", 11.5).attr("fill", AC.ink)
      .text("shift s = " + f.s + ", comparing P[" + Math.min(f.j, f.m - 1) + "] against T["
            + (f.s + Math.min(f.j, f.m - 1)) + "]");
    g.append("text").attr("x", 0).attr("y", 162).attr("font-size", 11.5)
      .attr("fill", f.hit ? AC.good : (f.ok ? AC.good : AC.bad))
      .text(f.hit ? "all m characters matched — an occurrence at shift " + f.s
                  : (f.ok ? "match — extend this alignment by one character"
                          : "mismatch — abandon this alignment, slide the pattern one place right"));
    g.append("text").attr("x", 0).attr("y", 186).attr("font-size", 11).attr("fill", AC.muted)
      .text("the naive matcher forgets everything it learned here before trying shift " + (f.s + 1));
  }

  sel.on("change", build);
  build();
})();

/* ══ FIGURE 11 ═══════════════════════════════════════════════════════════
   Rabin-Karp. The hash of each window is produced by the rolling recurrence;
   the figure independently recomputes every window's hash from scratch and
   reports whether the two agree, which is the check that the recurrence is
   the identity it claims to be.                                             */
(function () {
  const svg = d3.select("#rk-svg"); if (svg.empty()) return;
  const W = 690, H = 300;
  const T = "2359023141526739921", P = "31415", D = 10;

  const sel = d3.select("#rk-q"), out = d3.select("#rk-readout");
  let st = null;

  /* the honest recomputation: Horner over the window, no rolling */
  function hashFromScratch(str, s, m, d, q) {
    let v = 0;
    for (let i = 0; i < m; i++) v = (d * v + (+str[s + i])) % q;
    return v;
  }

  function build() {
    const Q = +sel.property("value");
    const frames = [], c = AL.counter();
    const res = MATCH.rabinKarp(T, P, D, Q, c, f => frames.push(f));

    /* cross-check every rolled hash against a fresh Horner evaluation */
    let disagree = 0;
    res.hashes.forEach((v, s) => { if (v !== hashFromScratch(T, s, P.length, D, Q)) disagree++; });

    if (st) st.pause();
    svg.selectAll("*").remove();
    AS.clearControls(svg.node());
    st = AL.stepper(svg, { frames, delay: 620, label: "window", render: f => render(f, Q, res) });

    const n = T.length, m = P.length, shifts = n - m + 1;
    out.html(
      "P = <code>" + P + "</code>, so p = " + P + " mod " + Q + " = <b>"
      + hashFromScratch(P, 0, m, D, Q) + "</b>. Over " + shifts + " windows the matcher recorded <b>"
      + (res.found.length + res.spurious.length) + "</b> hash hits: <b>" + res.found.length
      + "</b> genuine (shift" + (res.found.length === 1 ? " " : "s ") + res.found.join(", ") + ") and <b>"
      + res.spurious.length + "</b> spurious"
      + (res.spurious.length ? " (shift" + (res.spurious.length === 1 ? " " : "s ") + res.spurious.join(", ") + ")" : "")
      + ". Measured: <b>" + c.get("hashop") + "</b> arithmetic operations on hashes and only <b>"
      + c.get("cmp") + "</b> character comparisons — all of them spent verifying a hit, never on a window "
      + "the hash rejected. Cross-check — every one of the " + shifts + " rolled hashes was recomputed "
      + "from scratch by Horner's rule, and the two sets of values " + AS.verdict(disagree === 0)
      + (disagree ? " (" + disagree + " differ)" : " (0 differ)")
      + ". A larger modulus buys fewer spurious hits; try the options above."
    );
  }

  function render(f, Q, res) {
    const fr = AL.frame(svg, W, H, { l: 18, r: 18, t: 16, b: 10 });
    const g = fr.g;
    const chars = T.split("");
    const cw = 32, gap = 2;

    g.append("text").attr("x", 0).attr("y", 14).attr("font-size", 11).attr("fill", AC.muted).text("T");
    const rT = AL.row(g, chars, {
      x: 22, y: 2, w: cw, h: 28, gap, index: true, fontSize: 13,
      mark: i => (i >= f.s && i < f.s + f.m)
        ? (f.hit ? (f.ok ? AC.good : AC.a2) : AC.accent) : null
    });

    /* every window's hash, on one line, with the current one marked */
    g.append("text").attr("x", 0).attr("y", 76).attr("font-size", 10.5).attr("fill", AC.muted)
      .text("t_s");
    res.hashes.forEach((v, s) => {
      const isHit = v === f.p;
      g.append("text").attr("x", 22 + rT.cellX(s)).attr("y", 76).attr("text-anchor", "middle")
        .attr("font-size", 11)
        .attr("fill", s === f.s ? AC.ink : (isHit ? AC.a2 : "#6d7686"))
        .attr("font-weight", s === f.s ? "700" : "400")
        .text(v);
    });
    res.found.forEach(s => {
      g.append("text").attr("x", 22 + rT.cellX(s)).attr("y", 92).attr("text-anchor", "middle")
        .attr("font-size", 9).attr("fill", AC.good).text("match");
    });
    res.spurious.forEach(s => {
      g.append("text").attr("x", 22 + rT.cellX(s)).attr("y", 92).attr("text-anchor", "middle")
        .attr("font-size", 9).attr("fill", AC.a2).text("spurious");
    });

    /* the rolling step, with the actual numbers of this transition */
    const y0 = 128;
    g.append("text").attr("x", 0).attr("y", y0).attr("font-size", 11.5).attr("fill", AC.ink)
      .text("window at shift " + f.s + ": T[" + f.s + " … " + (f.s + f.m - 1) + "] = "
            + T.slice(f.s, f.s + f.m) + ",  t = " + f.t + ",  p = " + f.p);
    g.append("text").attr("x", 0).attr("y", y0 + 22).attr("font-size", 11.5)
      .attr("fill", f.hit ? (f.ok ? AC.good : AC.a2) : AC.muted)
      .text(f.hit ? (f.ok ? "t == p and the m characters really match — an occurrence at shift " + f.s
                          : "t == p but the characters differ — a SPURIOUS hit, caught by the verification")
                  : "t ≠ p, so this shift cannot be a match: skipped without looking at a single character");

    if (f.s + f.m < f.n) {
      g.append("text").attr("x", 0).attr("y", y0 + 52).attr("font-size", 11).attr("fill", AC.muted)
        .text("roll to the next window — drop the high digit, shift left, bring in the low digit:");
      const drop = +T[f.s], bring = +T[f.s + f.m];
      let nx = (D * (f.t - drop * f.h) + bring) % Q; if (nx < 0) nx += Q;
      g.append("text").attr("x", 0).attr("y", y0 + 74).attr("font-size", 12).attr("fill", AC.accent)
        .text("t' = (" + D + "·(t − T[" + f.s + "]·h) + T[" + (f.s + f.m) + "]) mod " + Q
              + " = (" + D + "·(" + f.t + " − " + drop + "·" + f.h + ") + " + bring + ") mod " + Q + " = " + nx);
      g.append("text").attr("x", 0).attr("y", y0 + 96).attr("font-size", 11).attr("fill", AC.muted)
        .text("h = d^(m−1) mod q = 10⁴ mod " + Q + " = " + f.h
              + " — a constant, so the roll is O(1) however long the pattern is");
    } else {
      g.append("text").attr("x", 0).attr("y", y0 + 74).attr("font-size", 11.5).attr("fill", AC.muted)
        .text("last window — nothing left to roll into");
    }
  }

  sel.on("change", build);
  build();
})();

/* ══ FIGURE 12 ═══════════════════════════════════════════════════════════
   The KMP prefix function, built cell by cell. The readout recomputes pi by
   brute force — for every q, the longest proper prefix of P that is also a
   suffix of P[0..q] — and compares, so the linear-time construction is
   checked against its own definition.                                       */
(function () {
  const svg = d3.select("#pi-svg"); if (svg.empty()) return;
  const W = 690, H = 268;

  const sel = d3.select("#pi-pattern"), out = d3.select("#pi-readout");
  const PRESETS = ["ababaca", "aabaabaaa", "abababca", "aaaaa"];
  let st = null;

  function build() {
    const P = sel.property("value");
    const frames = [], c = AL.counter();
    const pi = MATCH.prefixFunction(P, c, f => frames.push(f));
    const brute = MATCH.prefixBrute(P);
    const same = pi.every((v, i) => v === brute[i]);

    if (st) st.pause();
    svg.selectAll("*").remove();
    AS.clearControls(svg.node());
    st = AL.stepper(svg, { frames, delay: 720, label: "step", render: f => render(f, P) });

    const m = P.length;
    out.html(
      "P = <code>" + P + "</code> (m = " + m + ") gives pi = [" + pi.join(", ") + "]. "
      + "Measured: <b>" + c.get("cmp") + "</b> character comparisons to build the whole table. "
      + "The construction is <span class='keep'>Θ</span>(m) by an aggregate argument — k rises by at most one per "
      + "outer step, so its total increase is at most m − 1 = " + (m - 1)
      + ", and since every fall-back strictly decreases k and k never goes below zero, the fall-backs "
      + "total at most m − 1 as well; measured " + c.get("cmp") + " ≤ 2(m − 1) + 1 = " + (2 * (m - 1) + 1)
      + ": " + AS.verdict(c.get("cmp") <= 2 * (m - 1) + 1) + ". "
      + "Cross-check — recomputing pi directly from its definition, by testing every proper prefix "
      + "against every suffix, gives [" + brute.join(", ") + "]: " + AS.verdict(same) + "."
    );
  }

  function render(f, P) {
    const fr = AL.frame(svg, W, H, { l: 18, r: 18, t: 16, b: 10 });
    const g = fr.g;
    const chars = P.split(""), m = chars.length;
    const cw = Math.min(52, 520 / m), x0 = 40;

    g.append("text").attr("x", 0).attr("y", 22).attr("font-size", 11).attr("fill", AC.muted).text("P");
    const rP = AL.row(g, chars, {
      x: x0, y: 8, w: cw, h: 32, gap: 3, index: true, fontSize: 14,
      mark: i => {
        if (i === f.q) return f.fail ? AC.bad : AC.a2;
        if (i === f.k && f.q !== undefined && i < f.q) return AC.accent;
        if (i < f.q) return AC.panel;
        return null;
      }
    });

    g.append("text").attr("x", 0).attr("y", 92).attr("font-size", 11).attr("fill", AC.muted).text("pi");
    AL.row(g, f.pi.map((v, i) => (i <= (f.set === undefined ? f.q - 1 : f.set)) ? v : ""), {
      x: x0, y: 78, w: cw, h: 30, gap: 3, index: false, fontSize: 13,
      mark: i => (i === f.set ? AC.good : (i < (f.set === undefined ? f.q : f.set) ? AC.panel : null))
    });

    /* the prefix that is currently claimed to match the suffix ending at q */
    const kk = f.set !== undefined ? f.pi[f.set] : f.k;
    if (kk > 0 && f.set !== undefined) {
      const y = 126;
      g.append("text").attr("x", 0).attr("y", y + 20).attr("font-size", 10.5).attr("fill", AC.violet)
        .text("shift");
      AL.row(g, chars, {
        x: x0 + (f.set + 1 - kk) * (cw + 3), y, w: cw, h: 28, gap: 3, index: false, fontSize: 13,
        fill: "#1b2130",
        mark: i => i < kk ? AC.violet : null, text: "#9aa3b2"
      });
      g.append("text").attr("x", x0).attr("y", y + 48).attr("font-size", 10.5).attr("fill", AC.violet)
        .text("the " + kk + "-character prefix of P is also the suffix of P[0 … " + f.set + "]");
    }

    g.append("text").attr("x", 0).attr("y", 206).attr("font-size", 11.5)
      .attr("fill", f.fail ? AC.bad : AC.ink).text("q = " + f.q + ", k = " + f.k);
    g.append("text").attr("x", 0).attr("y", 228).attr("font-size", 11.5)
      .attr("fill", f.fail ? AC.bad : AC.ink).text(f.note);
    g.append("text").attr("x", 0).attr("y", 250).attr("font-size", 11).attr("fill", AC.muted)
      .text("pi[q] = the length of the longest PROPER prefix of P that is also a suffix of P[0 … q]");
  }

  sel.on("change", build);
  build();
})();

/* ══ FIGURE 13 ═══════════════════════════════════════════════════════════
   The race. Four matchers, one text and pattern, one unit of measurement —
   a character comparison actually performed — all read off AL.counter(). The
   table beside the bars reports the work each method does that ISN'T a
   character comparison, because that is where Rabin-Karp hides its cost. All
   four are also checked to have found the same occurrences.                 */
(function () {
  const svg = d3.select("#race-svg"); if (svg.empty()) return;
  const W = 690, H = 268;

  const sel = d3.select("#race-input"), out = d3.select("#race-readout");

  function randStr(alpha, n, seed) {
    const r = AL.rng(seed); let s = "";
    for (let i = 0; i < n; i++) s += alpha[Math.floor(r() * alpha.length)];
    return s;
  }
  /* a longer English paragraph, used to measure Boyer-Moore's comparisons per
     text character on prose rather than repeating a figure from elsewhere */
  const PROSE = ("the quick brown fox jumps over the lazy dog while the rain in spain falls "
    + "mainly on the plain and the plain truth is that pattern matching on ordinary english "
    + "prose is a very different problem from pattern matching on a binary string because the "
    + "alphabet is wide and most characters of the text are never examined at all by a matcher "
    + "that scans each window from its right end backwards toward its left end which is the "
    + "whole point of the method and the reason it is what production search tools reach for "
    + "when the pattern is long ").repeat(3);

  const PRESETS = {
    english: { T: "the rain in spain falls mainly on the plain", P: "plain",
               note: "natural-language text, 27-symbol alphabet, a rare pattern" },
    prose: { T: PROSE, P: "plain",
               note: "1 590 characters of English prose — long enough for the per-character rate to mean something" },
    prose8: { T: PROSE, P: "alphabet",
               note: "the same prose, an eight-character pattern — the maximum jump is m, so longer patterns skip more" },
    naiveworst: { T: "a".repeat(40), P: "aaab",
               note: "the naive worst case: every alignment matches m−1 characters, then fails" },
    bmworst: { T: "a".repeat(40), P: "baaa",
               note: "the Boyer-Moore worst case: the scan matches the tail every time and fails at the head" },
    binary: { T: randStr("01", 80, 17), P: "0101",
               note: "a two-symbol alphabet — few distinct characters, so jumps are short" },
    bmdemo: { T: "abacaabadcabacabaabb", P: "abacab",
               note: "a three-symbol alphabet where the right-to-left scan pays off" },
    dna: { T: randStr("ACGT", 80, 23), P: "ACGT",
               note: "a four-symbol alphabet, the usual sequence-search setting" }
  };

  function draw() {
    const key = sel.property("value"), M = PRESETS[key];
    const T = M.T, P = M.P, n = T.length, m = P.length;

    const cN = AL.counter(), cRK = AL.counter(), cK = AL.counter(), cKpre = AL.counter(), cBM = AL.counter();
    const fN = MATCH.naive(T, P, cN, null);
    const rRK = MATCH.rabinKarp(T, P, 256, 101, cRK, null, ch => ch.charCodeAt(0));
    const rK = MATCH.kmp(T, P, cK, null, cKpre);
    const fBM = MATCH.boyerMoore(T, P, cBM, null);

    const same = [rRK.found, rK.found, fBM].every(f => f.length === fN.length
      && f.every((v, i) => v === fN[i]));

    const rows = [
      { name: "naive", cmp: cN.get("cmp"), other: 0, extra: "no preprocessing", col: AC.bad },
      { name: "Rabin-Karp", cmp: cRK.get("cmp"), other: cRK.get("hashop"), col: AC.a2,
        extra: cRK.get("hashop") + " modular arithmetic ops (NOT character comparisons) · "
               + rRK.spurious.length + " spurious hits" },
      { name: "KMP", cmp: cK.get("cmp"), other: 0, col: AC.good,
        extra: cKpre.get("cmp") + " comparisons to build pi (m = " + m + ")" },
      { name: "Boyer-Moore", cmp: cBM.get("cmp"), other: 0, col: AC.violet,
        extra: "last-occurrence table over " + new Set(P.split("")).size + " distinct characters" }
    ];

    const f = AL.frame(svg, W, H, { l: 18, r: 18, t: 16, b: 10 });
    const g = f.g;
    g.append("text").attr("x", 0).attr("y", 6).attr("font-size", 11).attr("fill", AC.muted)
      .text("SOLID BARS = character comparisons only — n = " + n + ", m = " + m + " · " + M.note);

    const bx = 128, bw = 420;
    /* the scale has to admit Rabin-Karp's OTHER work, or the figure quietly
       argues that six comparisons is six units of work */
    const mx = d3.max(rows, d => d.cmp + d.other);
    const sc = d3.scaleLinear().domain([0, Math.max(1, mx)]).range([0, bw]);
    rows.forEach((r, k) => {
      const yy = 34 + k * 46;
      g.append("text").attr("x", bx - 10).attr("y", yy + 12).attr("text-anchor", "end")
        .attr("font-size", 12).attr("fill", AC.ink).text(r.name);
      g.append("rect").attr("x", bx).attr("y", yy).attr("width", Math.max(2, sc(r.cmp)))
        .attr("height", 16).attr("rx", 3).attr("fill", r.col).attr("opacity", 0.85);
      /* a hatched continuation for work that is NOT a character comparison, so
         the eye cannot read the solid bar as the whole cost. Drawn open, in a
         different unit, and labelled as such. */
      if (r.other > 0) {
        g.append("rect").attr("x", bx + sc(r.cmp)).attr("y", yy)
          .attr("width", Math.max(2, sc(r.other))).attr("height", 16).attr("rx", 3)
          .attr("fill", "none").attr("stroke", r.col).attr("stroke-width", 1.2)
          .attr("stroke-dasharray", "3 2");
        g.append("text").attr("x", bx + sc(r.cmp) + sc(r.other) + 8).attr("y", yy + 13)
          .attr("font-size", 11).attr("fill", r.col)
          .text(AS.int(r.cmp) + " cmp + " + AS.int(r.other) + " modular ops — two different units");
      } else {
        g.append("text").attr("x", bx + Math.max(2, sc(r.cmp)) + 8).attr("y", yy + 13)
          .attr("font-size", 12).attr("fill", AC.ink)
          .text(AS.int(r.cmp) + "   (" + AS.sig(r.cmp / n, 3) + " per text character)");
      }
      g.append("text").attr("x", bx).attr("y", yy + 31).attr("font-size", 10).attr("fill", AC.muted)
        .text(r.extra);
    });

    /* the reference lines: n, and the naive worst case */
    [{ v: n, lbl: "n = " + n + " (one look per text character)", col: AC.accent },
     { v: (n - m + 1) * m, lbl: "(n−m+1)·m = " + AS.int((n - m + 1) * m) + " (naive worst case)", col: AC.line }
    ].forEach(ref => {
      if (ref.v > mx) return;
      g.append("line").attr("x1", bx + sc(ref.v)).attr("x2", bx + sc(ref.v))
        .attr("y1", 28).attr("y2", 34 + rows.length * 46 - 16)
        .attr("stroke", ref.col).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 3");
      g.append("text").attr("x", bx + sc(ref.v) + 4).attr("y", 236).attr("font-size", 10)
        .attr("fill", ref.col).text(ref.lbl);
    });

    out.html(
      "T = <code>" + (T.length > 56 ? T.slice(0, 56) + "…" : T) + "</code>, P = <code>" + P + "</code>. "
      + "Measured character comparisons — naive <b>" + AS.int(cN.get("cmp")) + "</b>, Rabin-Karp <b>"
      + AS.int(cRK.get("cmp")) + "</b> (plus " + AS.int(cRK.get("hashop"))
      + " modular operations, which is where its work really is), KMP <b>" + AS.int(cK.get("cmp"))
      + "</b> matching plus " + cKpre.get("cmp") + " preprocessing, Boyer-Moore <b>"
      + AS.int(cBM.get("cmp")) + "</b> (<b>" + AS.sig(cBM.get("cmp") / n, 3)
      + "</b> per text character, against naive's " + AS.sig(cN.get("cmp") / n, 3) + "). "
      + "<b>Rabin-Karp's bar is not comparable to the others</b> — its "
      + AS.int(cRK.get("cmp")) + " comparisons are only the "
      + "verification of windows the fingerprint already accepted, and its real work is the "
      + AS.int(cRK.get("hashop")) + " modular operations drawn as the dashed continuation, in a "
      + "different unit. "
      + "Occurrences found: " + fN.length + (fN.length ? " at shift" + (fN.length === 1 ? " " : "s ") + fN.join(", ") : "")
      + ", and all four matchers returned the identical list: " + AS.verdict(same) + ". "
      + "KMP's matching comparisons are bounded by 2n = " + (2 * n) + " (each text character is compared once "
      + "on the way forward, and the fall-backs total at most n): measured " + cK.get("cmp") + " ≤ " + (2 * n)
      + ", " + AS.verdict(cK.get("cmp") <= 2 * n) + "."
    );
  }

  sel.on("change", draw);
  draw();
})();

/* ══ FIGURE 14 ═══════════════════════════════════════════════════════════
   What one string comparison actually costs. A hand-written mergesort keeps
   the number of string comparisons deterministic; the comparator counts the
   CHARACTER comparisons it performs, so the figure separates "how many
   comparisons" from "what a comparison costs" — two numbers that the phrase
   "n log n comparisons" silently merges.                                   */
(function () {
  const svg = d3.select("#strcmp-svg"); if (svg.empty()) return;
  const W = 690, H = 268;

  const els = {
    k: d3.select("#strcmp-k"), kOut: d3.select("#strcmp-k-out"),
    out: d3.select("#strcmp-readout")
  };
  const LEN = 32, ALPHA = "abcdefghijklmnopqrstuvwxyz";

  function make(k, shared, seed) {
    const r = AL.rng(seed);
    let pre = "";
    for (let i = 0; i < shared; i++) pre += ALPHA[Math.floor(r() * 26)];
    const out = [];
    for (let i = 0; i < k; i++) {
      let s = pre;
      for (let j = shared; j < LEN; j++) s += ALPHA[Math.floor(r() * 26)];
      out.push(s);
    }
    return out;
  }

  /* lexicographic comparison, counting every character it inspects */
  function cmpStr(a, b, c) {
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
      c.add("charcmp");
      if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1;
    }
    c.add("charcmp");
    return a.length - b.length;
  }
  function mergesort(A, c) {
    if (A.length <= 1) return A;
    const mid = Math.floor(A.length / 2);
    const L = mergesort(A.slice(0, mid), c), R = mergesort(A.slice(mid), c);
    const out = [];
    let i = 0, j = 0;
    while (i < L.length && j < R.length) {
      c.add("strcmp");
      if (cmpStr(L[i], R[j], c) <= 0) out.push(L[i++]); else out.push(R[j++]);
    }
    while (i < L.length) out.push(L[i++]);
    while (j < R.length) out.push(R[j++]);
    return out;
  }

  function draw() {
    const k = +els.k.property("value");
    els.kOut.text(k);

    const cases = [
      { name: "random strings", shared: 0, col: AC.good },
      { name: "8 shared leading characters", shared: 8, col: AC.a2 },
      { name: "24 shared leading characters", shared: 24, col: AC.bad }
    ].map(cs => {
      const data = make(k, cs.shared, 31 + cs.shared);
      const c = AL.counter();
      const sorted = mergesort(data, c);
      const ok = sorted.every((v, i) => i === 0 || sorted[i - 1] <= v);
      return Object.assign({}, cs, {
        strcmp: c.get("strcmp"), charcmp: c.get("charcmp"),
        per: c.get("charcmp") / c.get("strcmp"), ok
      });
    });
    const allSorted = cases.every(c => c.ok);
    /* the worst-case comparison count of top-down mergesort, evaluated from the
       closed form rather than from the run — every row must sit at or under it */
    const lg = Math.ceil(AL.log2(k));
    const bound = k * lg - Math.pow(2, lg) + 1;
    const underBound = cases.every(c => c.strcmp <= bound);
    const spread = d3.max(cases, c => c.strcmp) / d3.min(cases, c => c.strcmp);
    const priceSpread = d3.max(cases, c => c.charcmp) / d3.min(cases, c => c.charcmp);

    const f = AL.frame(svg, W, H, { l: 18, r: 18, t: 16, b: 10 });
    const g = f.g;
    g.append("text").attr("x", 0).attr("y", 6).attr("font-size", 11).attr("fill", AC.muted)
      .text("sorting " + k + " strings of " + LEN + " characters — measured character comparisons");

    const bx = 190, bw = 360;
    const mx = d3.max(cases, d => d.charcmp);
    const sc = d3.scaleLinear().domain([0, mx]).range([0, bw]);
    cases.forEach((cs, i) => {
      const yy = 38 + i * 58;
      g.append("text").attr("x", bx - 10).attr("y", yy + 12).attr("text-anchor", "end")
        .attr("font-size", 12).attr("fill", AC.ink).text(cs.name);
      g.append("rect").attr("x", bx).attr("y", yy).attr("width", Math.max(2, sc(cs.charcmp)))
        .attr("height", 17).attr("rx", 3).attr("fill", cs.col).attr("opacity", 0.85);
      g.append("text").attr("x", bx + Math.max(2, sc(cs.charcmp)) + 8).attr("y", yy + 14)
        .attr("font-size", 12).attr("fill", AC.ink).text(AS.int(cs.charcmp) + " characters");
      g.append("text").attr("x", bx).attr("y", yy + 34).attr("font-size", 10.5).attr("fill", AC.muted)
        .text(cs.strcmp + " string comparisons · " + AS.sig(cs.per, 1) + " characters per comparison");
    });

    g.append("text").attr("x", 0).attr("y", 232).attr("font-size", 11).attr("fill", AC.violet)
      .text("the number of COMPARISONS barely moves between rows (" + AS.sig(spread, 2)
            + "× spread) — their price moves by " + AS.sig(priceSpread, 0) + "×");

    els.out.html(
      "Sorting <b>" + k + "</b> strings of " + LEN + " characters, the mergesort performed <b>"
      + cases[0].strcmp + "</b>, <b>" + cases[1].strcmp + "</b> and <b>" + cases[2].strcmp
      + "</b> string comparisons — a spread of only " + AS.sig(spread, 2)
      + "×, and every one of them at or below mergesort's worst case of k·ceil(log₂ k) − 2^ceil(log₂ k) + 1 = "
      + bound + ": " + AS.verdict(underBound) + ". "
      + "The measured <i>character</i> comparisons were <b>" + AS.int(cases[0].charcmp) + "</b>, <b>"
      + AS.int(cases[1].charcmp) + "</b> and <b>" + AS.int(cases[2].charcmp)
      + "</b> — a spread of <b>" + AS.sig(priceSpread, 0) + "×</b> — or <b>"
      + AS.sig(cases[0].per, 1) + "</b>, <b>" + AS.sig(cases[1].per, 1)
      + "</b> and <b>" + AS.sig(cases[2].per, 1) + "</b> characters per comparison. "
      + "All three outputs were independently verified to be in non-decreasing order: " + AS.verdict(allSorted)
      + ". So a bound of <span class='keep'>Θ</span>(k log k) comparisons hides a factor equal to the mean "
      + "shared prefix length plus one, and the true cost is <span class='keep'>Θ</span>(k · log k · (p + 1)) "
      + "— the +1 is what the random-string row measures, where p is about zero and each comparison "
      + "still costs " + AS.sig(cases[0].per, 1) + " characters."
    );
  }

  els.k.on("input change", draw);
  draw();
})();
