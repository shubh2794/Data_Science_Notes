/* divide-conquer.viz.js — figures for dsa/algorithms/divide-conquer.html
   (part 2 of the Algorithm Design & Analysis series).

   Loaded after ../../data.js, ../../notes.js and ../dsa-viz.js, so the globals
   AC (palette) and AL (rng/frame/axisB/axisL/gridY/row/bars/binTree/counter/
   stepper/…) are available.

   House rule obeyed throughout: every count this page DISPLAYS — invocations,
   comparisons, digit multiplications, scalar multiplications, distance
   evaluations, search-tree nodes explored and pruned — is produced by running
   the real routine under AL.counter() and reading the counter back. Nothing
   below is a constant typed into a caption. Wherever a closed form exists it
   is evaluated independently and printed next to the measurement with an
   agree / DISAGREE flag, so each figure is its own cross-check.

   Figures, in page order:
     01 #ft-svg      naive fib call tree: measured nodes, height, repeats
     02 #dp-svg      measured recursion depth and call count for four shapes
     03 #hn-svg      Towers of Hanoi stepped: measured moves vs 2^n − 1
     04 #es-svg      a recursion re-run on an explicit stack, stepped
     05 #mm-svg      naive vs memoized fib: measured invocations
     06 #mg-svg      mergesort stepped, comparisons and moves measured
     07 #ms-svg      maximum subarray stepped, the crossing case highlighted
     08 #kr-svg      Karatsuba vs schoolbook: measured digit multiplications
     09 #st-svg      Strassen vs naive block: measured mults and add/subs
     10 #cp-svg      closest pair: measured distance evaluations vs brute force
     11 #co-svg      mergesort with an insertion-sort cutoff: measured total work
     12 #nq-svg      n-queens backtracking: nodes explored vs nodes pruned
     13 #en-svg      subsets / permutations enumeration trees, measured sizes
     14 #rq-svg      randomized quicksort: measured comparisons vs 2(n+1)Hₙ − 4n */

/* ── shared little helpers ─────────────────────────────────────────────── */
const DC = {
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
  /* Fibonacci by iteration — deliberately a DIFFERENT routine from the one the
     figures instrument, so "closed form" and "measurement" are independent. */
  fib: function (k) {
    let a = 0, b = 1;
    for (let i = 0; i < k; i++) { const t = a + b; a = b; b = t; }
    return a;                                   // fib(0)=0, fib(1)=1
  },
  flag: function (ok) {
    return ok ? '<span style="color:' + AC.good + '">✓ agree</span>'
              : '<span style="color:' + AC.bad + '">✗ DISAGREE</span>';
  },
  /* strip any stepper control groups a previous build left behind */
  clearControls: function (svgNode) {
    svgNode.parentNode.querySelectorAll('div[role="group"]').forEach(el => el.remove());
  }
};

/* ══ FIGURE 01 ═══════════════════════════════════════════════════════════
   The call tree of the naive Fibonacci recursion. The tree is built BY
   running the routine — every node in the drawing is an invocation that
   actually happened — and the node count is read off AL.counter(). The
   closed forms 2·F(n+1) − 1 (total calls) and F(n−k+1) (calls at argument k)
   are evaluated from an independent iterative Fibonacci and compared.      */
(function () {
  const svg = d3.select("#ft-svg"); if (svg.empty()) return;
  const W = 680, H = 330;

  /* the real routine, instrumented: c counts invocations, seen[] counts them
     per argument, and the returned object IS the call tree. */
  function fibTree(n, c, seen) {
    c.add("calls");
    seen[n] = (seen[n] || 0) + 1;
    if (n <= 1) return { arg: n, children: [] };
    const a = fibTree(n - 1, c, seen), b = fibTree(n - 2, c, seen);
    return { arg: n, children: [a, b] };
  }

  let n = 5, mode = "repeat";

  function build() {
    const c = AL.counter(), seen = {};
    const root = fibTree(n, c, seen);
    const calls = c.get("calls");

    /* mark first-vs-repeat occurrences in pre-order, and record each node's
       depth, by one walk over the tree that was actually produced */
    const firstSeen = Object.create(null);
    let height = 0, leaves = 0;
    (function walk(t, d) {
      t.depth = d;
      height = Math.max(height, d + 1);
      if (firstSeen[t.arg] === undefined) { firstSeen[t.arg] = true; t.repeat = false; }
      else t.repeat = true;
      if (t.children.length === 0) leaves++;
      t.children.forEach(k => walk(k, d + 1));
    })(root, 0);

    const distinct = Object.keys(seen).length;
    const predCalls = 2 * DC.fib(n + 1) - 1;
    const predLeaves = DC.fib(n + 1);

    draw(root, height);

    d3.select("#ft-readout").html(
      "n = <b>" + n + "</b> &nbsp;·&nbsp; invocations measured <b>" + DC.int(calls) + "</b>"
      + " &nbsp;·&nbsp; closed form 2·F(" + (n + 1) + ") − 1 = <b>" + DC.int(predCalls) + "</b> "
      + DC.flag(calls === predCalls)
      + " &nbsp;·&nbsp; height measured <b>" + height + "</b> (= n, because the spine stops at"
      + " the base case fib(1), not at fib(0)) " + DC.flag(height === n)
      + " &nbsp;·&nbsp; distinct arguments <b>" + distinct + "</b>"
      + " &nbsp;·&nbsp; leaves <b>" + DC.int(leaves) + "</b> = F(" + (n + 1) + ") "
      + DC.flag(leaves === predLeaves));

    /* per-argument table: measured invocations vs F(n−k+1)  (and F(n−1) at k = 0) */
    const rows = [];
    for (let k = n; k >= 0; k--) {
      const got = seen[k] || 0;
      let pred = null;
      if (n >= 2) pred = (k === 0) ? DC.fib(n - 1) : DC.fib(n - k + 1);
      rows.push([
        "<code>fib(" + k + ")</code>",
        "<b>" + DC.int(got) + "</b>",
        pred === null ? "—" : DC.int(pred),
        pred === null ? "—" : DC.flag(got === pred)
      ]);
    }
    DC.table("#ft-table",
      ["Subproblem", "Times invoked (measured)", "F(n−k+1), k ≥ 1 · F(n−1) at k = 0", "check"],
      rows);
  }

  function draw(root, height) {
    const fr = AL.frame(svg, W, H, { l: 18, r: 18, t: 16, b: 22 });
    const g = fr.g;
    const h = d3.hierarchy(root, d => d.children);
    const layout = d3.tree().size([fr.iw, fr.ih - 10]);
    layout(h);
    const nodes = h.descendants(), links = h.links();
    const r = AL.clamp(190 / Math.max(8, nodes.length), 5, 15);

    g.selectAll("path.lk").data(links).join("path").attr("class", "lk")
      .attr("d", d => "M" + d.source.x + "," + d.source.y + " L" + d.target.x + "," + d.target.y)
      .attr("fill", "none").attr("stroke", AC.line).attr("stroke-width", 1.2);

    const depthCol = d3.scaleLinear().domain([0, Math.max(1, height - 1)])
      .range([AC.accent, AC.violet]);

    const nd = g.selectAll("g.nd").data(nodes).join("g").attr("class", "nd")
      .attr("transform", d => "translate(" + d.x + "," + d.y + ")");
    nd.append("circle").attr("r", r)
      .attr("fill", d => mode === "repeat" ? (d.data.repeat ? AC.a2 : AC.accent)
                       : mode === "depth" ? depthCol(d.data.depth) : AC.panel2)
      .attr("stroke", AC.line);
    if (r >= 7) {
      nd.append("text").attr("y", 3.5).attr("text-anchor", "middle")
        .attr("font-size", Math.min(12, r * 1.1)).attr("fill", "#0f1117")
        .attr("font-weight", 600)
        .text(d => mode === "depth" ? d.data.depth : d.data.arg);
    }

    const key = mode === "repeat"
      ? [{ label: "first time this argument is seen", color: AC.accent },
         { label: "a recomputation — seen before", color: AC.a2 }]
      : mode === "depth"
      ? [{ label: "depth 0 (the root frame)", color: AC.accent },
         { label: "the deepest frames", color: AC.violet }]
      : [{ label: "one invocation", color: AC.panel2 }];
    AL.legend(g, key, 4, 12);
  }

  d3.select("#ft-n").on("input", function () {
    n = +this.value; d3.select("#ft-n-out").text(n); build();
  });
  d3.select("#ft-mode").on("change", function () { mode = this.value; build(); });
  d3.select("#ft-n-out").text(n);
  build();
})();

/* ══ FIGURE 02 ═══════════════════════════════════════════════════════════
   Measured recursion depth (left) and measured invocation count (right) for
   four recursion shapes. Every plotted point comes from running the real
   routine with a live-depth tracker and an AL.counter(); the table compares
   each measurement against a closed form derived independently.

   Naive Fibonacci is measured only up to n = FIBCAP: at n = 30 the routine
   makes 2·F(31) − 1 = 2,692,537 calls, and measuring a whole curve of those
   inside a page would be gratuitous. The curve stops where the measurement
   stops, rather than being extrapolated.                                   */
(function () {
  const svg = d3.select("#dp-svg"); if (svg.empty()) return;
  const W = 680, H = 300, FIBCAP = 22;

  /* integer logs — Math.log2 near a power of two is exactly the trap AL.log2's
     comment warns about, and these feed an equality check */
  const ilog2 = n => 31 - Math.clz32(n);                       // floor(log2 n), n ≥ 1
  const clog2 = n => (n & (n - 1)) === 0 ? ilog2(n) : ilog2(n) + 1;   // ceil(log2 n)

  /* ── the four instrumented routines ─────────────────────────────────── */
  function run(fn) {                       // returns {calls, depth}
    const c = AL.counter();
    const st = { d: 0, max: 0 };
    const enter = () => { c.add("calls"); st.d++; if (st.d > st.max) st.max = st.d; };
    const leave = () => { st.d--; };
    fn(enter, leave);
    return { calls: c.get("calls"), depth: st.max };
  }

  function linearSum(A) {
    return run((enter, leave) => {
      (function f(k) {                      // sum of A[0..k−1]
        enter();
        if (k === 0) { leave(); return 0; }
        const v = A[k - 1] + f(k - 1);
        leave(); return v;
      })(A.length);
    });
  }
  function binarySum(A) {
    return run((enter, leave) => {
      (function f(lo, hi) {                 // sum of A[lo..hi−1]
        enter();
        if (hi - lo === 0) { leave(); return 0; }
        if (hi - lo === 1) { leave(); return A[lo]; }
        const mid = lo + Math.floor((hi - lo) / 2);
        const v = f(lo, mid) + f(mid, hi);
        leave(); return v;
      })(0, A.length);
    });
  }
  /* binary search, worst case over every target: each even value is present,
     each odd value is absent, so the probe set covers hits and every gap */
  function binarySearchWorst(A) {
    let best = { calls: 0, depth: 0 };
    for (let t = -1; t <= 2 * A.length + 1; t++) {
      const r = run((enter, leave) => {
        (function f(lo, hi) {               // search A[lo..hi−1]
          enter();
          if (hi - lo <= 0) { leave(); return -1; }
          const mid = lo + Math.floor((hi - lo) / 2);
          let v;
          if (A[mid] === t) v = mid;
          else if (t < A[mid]) v = f(lo, mid);
          else v = f(mid + 1, hi);
          leave(); return v;
        })(0, A.length);
      });
      if (r.depth > best.depth) best = r;
    }
    return best;
  }
  function naiveFib(n) {
    return run((enter, leave) => {
      (function f(k) {
        enter();
        if (k <= 1) { leave(); return k; }
        const v = f(k - 1) + f(k - 2);
        leave(); return v;
      })(n);
    });
  }

  const SERIES = [
    { key: "lin",  label: "linear sum — recurse on n − 1",       color: AC.a2 },
    { key: "bin",  label: "binary sum — recurse on both halves", color: AC.accent },
    { key: "bs",   label: "binary search — recurse on one half", color: AC.teal },
    { key: "fib",  label: "naive fib — recurse on n−1 and n−2",  color: AC.rose }
  ];

  let nmax = 32, scale = "log";

  function measure(n) {
    const A = Array.from({ length: n }, (_, i) => 2 * (i + 1));   // even ⇒ odd probes miss
    const out = { n: n };
    out.lin = linearSum(A);
    out.bin = binarySum(A);
    out.bs  = binarySearchWorst(A);
    out.fib = n <= FIBCAP ? naiveFib(n) : null;
    return out;
  }

  function build() {
    const data = [];
    for (let n = 1; n <= nmax; n++) data.push(measure(n));
    draw(data);

    const last = data[data.length - 1], n = last.n;
    const pred = {
      lin: { d: n + 1,            c: n + 1 },
      bin: { d: clog2(n) + 1,     c: 2 * n - 1 },
      bs:  { d: ilog2(n) + 2,     c: ilog2(n) + 2 },
      fib: last.fib ? { d: n,     c: 2 * DC.fib(n + 1) - 1 } : null
    };
    const form = {
      lin: ["n + 1", "n + 1"],
      bin: ["ceil(log₂ n) + 1", "2n − 1"],
      bs:  ["floor(log₂ n) + 2", "floor(log₂ n) + 2"],
      fib: ["n  (spine ends at the base case)", "2·F(n+1) − 1"]
    };
    const rows = SERIES.map(s => {
      const m = last[s.key], p = pred[s.key];
      if (!m || !p) return ["<b style='color:" + s.color + "'>" + s.label + "</b>",
                            "not measured beyond n = " + FIBCAP, "—", "—", "—"];
      return [
        "<b style='color:" + s.color + "'>" + s.label + "</b>",
        "<b>" + DC.int(m.depth) + "</b>",
        "<code>" + form[s.key][0] + "</code> = " + DC.int(p.d) + " " + DC.flag(m.depth === p.d),
        "<b>" + DC.int(m.calls) + "</b>",
        "<code>" + form[s.key][1] + "</code> = " + DC.int(p.c) + " " + DC.flag(m.calls === p.c)
      ];
    });
    DC.table("#dp-table",
      ["Recursion, at n = " + n, "Peak depth (measured)", "predicted", "Invocations (measured)", "predicted"],
      rows);

    d3.select("#dp-readout").html(
      "at n = <b>" + n + "</b> &nbsp;·&nbsp; peak frames — linear sum <b>" + DC.int(last.lin.depth)
      + "</b>, binary sum <b>" + DC.int(last.bin.depth) + "</b>, binary search <b>"
      + DC.int(last.bs.depth) + "</b>"
      + (last.fib ? ", naive fib <b>" + DC.int(last.fib.depth) + "</b>" : "")
      + " &nbsp;·&nbsp; invocations — linear sum <b>" + DC.int(last.lin.calls)
      + "</b>, binary sum <b>" + DC.int(last.bin.calls) + "</b>, binary search <b>"
      + DC.int(last.bs.calls) + "</b>"
      + (last.fib ? ", naive fib <b>" + DC.int(last.fib.calls) + "</b>"
                  : ", naive fib not measured past n = " + FIBCAP));
  }

  function draw(data) {
    const fr = AL.frame(svg, W, H, { l: 8, r: 8, t: 10, b: 8 });
    const g = fr.g, pw = (fr.iw - 26) / 2, ph = fr.ih - 66;
    const mkPanel = (ox, title) => {
      const p = g.append("g").attr("transform", "translate(" + ox + ",22)");
      g.append("text").attr("x", ox + 42).attr("y", 14).attr("font-size", 12)
        .attr("fill", AC.ink).attr("font-weight", 600).text(title);
      return p.append("g").attr("transform", "translate(42,0)");
    };

    /* left panel — depth */
    const gl = mkPanel(0, "peak recursion depth (frames)");
    const iw = pw - 52, ih = ph - 26;
    const x = d3.scaleLinear().domain([1, nmax]).range([0, iw]);
    const dmax = d3.max(data, d => d.lin.depth);
    const y = d3.scaleLinear().domain([0, dmax * 1.08]).range([ih, 0]).nice();
    AL.gridY(gl, y, iw, 5); AL.axisB(gl, x, ih, 5, "n"); AL.axisL(gl, y, 5, null);
    SERIES.forEach(s => {
      const pts = data.filter(d => d[s.key]).map(d => [d.n, d[s.key].depth]);
      gl.append("path").datum(pts).attr("fill", "none").attr("stroke", s.color)
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", s.key === "fib" ? "4 3" : null)
        .attr("d", d3.line().x(p => x(p[0])).y(p => y(p[1])));
    });

    /* right panel — calls */
    const gr = mkPanel(pw + 26, "invocations" + (scale === "log" ? " (log axis)" : ""));
    const x2 = d3.scaleLinear().domain([1, nmax]).range([0, iw]);
    const cmax = d3.max(data, d => Math.max(d.lin.calls, d.bin.calls, d.fib ? d.fib.calls : 0));
    const y2 = scale === "log"
      ? d3.scaleLog().domain([1, cmax * 1.4]).range([ih, 0])
      : d3.scaleLinear().domain([0, cmax * 1.08]).range([ih, 0]).nice();
    AL.gridY(gr, y2, iw, 5); AL.axisB(gr, x2, ih, 5, "n");
    AL.axisL(gr, y2, 5, null, scale === "log" ? d3.format("~s") : null);
    SERIES.forEach(s => {
      const pts = data.filter(d => d[s.key] && (scale !== "log" || d[s.key].calls > 0))
                      .map(d => [d.n, d[s.key].calls]);
      gr.append("path").datum(pts).attr("fill", "none").attr("stroke", s.color)
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", s.key === "fib" ? "4 3" : null)
        .attr("d", d3.line().x(p => x2(p[0])).y(p => y2(p[1])));
    });

    AL.legend(g, SERIES.map(s => ({ label: s.label, color: s.color })), 46, ph + 42, { gap: 15 });
  }

  d3.select("#dp-nmax").on("input", function () {
    nmax = +this.value; d3.select("#dp-nmax-out").text(nmax); build();
  });
  d3.select("#dp-scale").on("change", function () { scale = this.value; build(); });
  d3.select("#dp-nmax-out").text(nmax);
  build();
})();

/* ── the Hanoi recursion, instrumented, shared by figures 03 and 04 ─────── */
function hanoiRecursive(n) {
  const c = AL.counter(), moves = [];
  let d = 0, maxd = 0;
  (function h(k, from, to, via) {
    c.add("calls"); d++; if (d > maxd) maxd = d;
    if (k === 0) { d--; return; }
    h(k - 1, from, via, to);
    c.add("moves");
    moves.push({ disk: k, from: from, to: to });
    h(k - 1, via, to, from);
    d--;
  })(n, 0, 2, 1);
  return { moves: moves, calls: c.get("calls"), nmoves: c.get("moves"), depth: maxd };
}

/* ══ FIGURE 03 ═══════════════════════════════════════════════════════════
   Towers of Hanoi, stepped. The move list is produced by the recursion above,
   so the animation IS the recursion's output; moves, invocations and peak
   depth all come from AL.counter() / the live-depth tracker and are printed
   next to the closed forms 2ⁿ − 1, 2ⁿ⁺¹ − 1 and n + 1.                    */
(function () {
  const svg = d3.select("#hn-svg"); if (svg.empty()) return;
  const W = 680, H = 260;
  let n = 4, st = null;

  function build() {
    const r = hanoiRecursive(n);

    /* replay the move list into a sequence of peg states */
    const pegs = [[], [], []];
    for (let k = n; k >= 1; k--) pegs[0].push(k);          // bottom → top
    const frames = [{ pegs: pegs.map(p => [...p]), i: -1, mv: null }];
    const cur = pegs.map(p => [...p]);
    r.moves.forEach((m, i) => {
      cur[m.from].pop();
      cur[m.to].push(m.disk);
      frames.push({ pegs: cur.map(p => [...p]), i: i, mv: m });
    });

    if (st) st.pause();
    DC.clearControls(svg.node());
    st = AL.stepper(svg, {
      frames: frames, delay: 520, label: "move",
      render: (f, i) => draw(f, i, frames.length, r)
    });

    const pm = Math.pow(2, n) - 1, pc = Math.pow(2, n + 1) - 1, pd = n + 1;
    DC.table("#hn-table",
      ["Quantity", "Measured", "Closed form", "check"],
      [["disk moves", "<b>" + DC.int(r.nmoves) + "</b>", "<code>2ⁿ − 1</code> = " + DC.int(pm), DC.flag(r.nmoves === pm)],
       ["invocations", "<b>" + DC.int(r.calls) + "</b>", "<code>2ⁿ⁺¹ − 1</code> = " + DC.int(pc), DC.flag(r.calls === pc)],
       ["peak recursion depth", "<b>" + DC.int(r.depth) + "</b>", "<code>n + 1</code> = " + pd, DC.flag(r.depth === pd)],
       ["invocations that move nothing", "<b>" + DC.int(r.calls - r.nmoves) + "</b>",
        "<code>2ⁿ</code> = " + DC.int(Math.pow(2, n)), DC.flag(r.calls - r.nmoves === Math.pow(2, n))]]);
  }

  function draw(f, idx, nf, r) {
    const fr = AL.frame(svg, W, H, { l: 18, r: 18, t: 14, b: 16 });
    const g = fr.g, pw = fr.iw / 3, base = fr.ih - 34, maxW = pw - 26;
    const names = ["A (source)", "B (spare)", "C (target)"];
    const dh = Math.min(20, (base - 26) / Math.max(1, n));

    for (let p = 0; p < 3; p++) {
      const cx = p * pw + pw / 2;
      g.append("rect").attr("x", cx - 3).attr("y", base - n * dh - 12).attr("width", 6)
        .attr("height", n * dh + 12).attr("rx", 3).attr("fill", AC.panel2).attr("stroke", AC.line);
      g.append("rect").attr("x", p * pw + 14).attr("y", base).attr("width", pw - 28)
        .attr("height", 6).attr("rx", 3).attr("fill", AC.line);
      g.append("text").attr("x", cx).attr("y", base + 24).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("fill", AC.muted).text(names[p]);
      f.pegs[p].forEach((disk, k) => {
        const w = 26 + (maxW - 26) * (disk - 1) / Math.max(1, n - 1);
        const moving = f.mv && f.mv.disk === disk;
        g.append("rect").attr("x", cx - w / 2).attr("y", base - (k + 1) * dh + 1)
          .attr("width", w).attr("height", dh - 2).attr("rx", 4)
          .attr("fill", moving ? AC.a2 : AC.accent).attr("opacity", moving ? 1 : 0.45)
          .attr("stroke", AC.line);
        g.append("text").attr("x", cx).attr("y", base - (k + 1) * dh + dh / 2 + 3)
          .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", "#0f1117")
          .attr("font-weight", 600).text(disk);
      });
    }

    const done = f.i + 1;
    g.append("text").attr("x", 0).attr("y", 12).attr("font-size", 12).attr("fill", AC.ink)
      .text(f.mv ? ("move " + done + " of " + r.nmoves + ":  disk " + f.mv.disk
                    + "  " + names[f.mv.from][0] + " → " + names[f.mv.to][0])
                 : "start — all " + n + " disks on peg A, largest at the bottom");
    d3.select("#hn-readout").html(
      "n = <b>" + n + "</b> &nbsp;·&nbsp; moves made so far <b>" + done + "</b> of <b>"
      + DC.int(r.nmoves) + "</b> measured (2ⁿ − 1 = " + DC.int(Math.pow(2, n) - 1) + ") "
      + DC.flag(r.nmoves === Math.pow(2, n) - 1)
      + " &nbsp;·&nbsp; invocations <b>" + DC.int(r.calls) + "</b> &nbsp;·&nbsp; peak depth <b>"
      + DC.int(r.depth) + "</b>");
  }

  d3.select("#hn-n").on("input", function () {
    n = +this.value; d3.select("#hn-n-out").text(n); build();
  });
  d3.select("#hn-n-out").text(n);
  build();
})();

/* ══ FIGURE 04 ═══════════════════════════════════════════════════════════
   The same recursion driven by an explicit stack of (n, from, to, via, stage)
   records. Every loop iteration is a frame. The peak stack size is measured
   and compared with the recursive version's measured peak depth, and the two
   emitted move sequences are compared element by element — which is the real
   claim being made, that the conversion changes storage and nothing else.  */
(function () {
  const svg = d3.select("#es-svg"); if (svg.empty()) return;
  const W = 680, H = 300;
  const PEG = ["A", "B", "C"];
  let n = 3, st = null;

  function hanoiExplicit(n) {
    const c = AL.counter(), moves = [], frames = [];
    const stack = [{ n: n, from: 0, to: 2, via: 1, stage: 0 }];
    let peak = 1;
    c.add("pushes");
    frames.push({ stack: stack.map(o => Object.assign({}, o)), moves: [...moves],
                  note: "push the top-level record (n = " + n + ", A → C via B) at stage 0" });
    while (stack.length) {
      const top = stack[stack.length - 1];
      let note;
      if (top.n === 0 || top.stage === 2) {
        note = top.n === 0 ? "top record has n = 0 — base case, pop"
                           : "stage 2: both recursive calls have returned — pop";
        stack.pop();
      } else if (top.stage === 0) {
        top.stage = 1;
        stack.push({ n: top.n - 1, from: top.from, to: top.via, via: top.to, stage: 0 });
        c.add("pushes");
        note = "stage 0 → 1: push the FIRST recursive call (n = " + (top.n - 1)
             + ", " + PEG[top.from] + " → " + PEG[top.via] + ")";
      } else {
        top.stage = 2;
        c.add("moves");
        moves.push({ disk: top.n, from: top.from, to: top.to });
        stack.push({ n: top.n - 1, from: top.via, to: top.to, via: top.from, stage: 0 });
        c.add("pushes");
        note = "stage 1 → 2: emit move disk " + top.n + " " + PEG[top.from] + " → " + PEG[top.to]
             + ", then push the SECOND recursive call";
      }
      if (stack.length > peak) peak = stack.length;
      frames.push({ stack: stack.map(o => Object.assign({}, o)), moves: [...moves], note: note });
    }
    return { moves: moves, pushes: c.get("pushes"), nmoves: c.get("moves"),
             peak: peak, frames: frames };
  }

  function build() {
    const rec = hanoiRecursive(n), exp = hanoiExplicit(n);
    const same = rec.moves.length === exp.moves.length && rec.moves.every((m, i) =>
      m.disk === exp.moves[i].disk && m.from === exp.moves[i].from && m.to === exp.moves[i].to);

    if (st) st.pause();
    DC.clearControls(svg.node());
    st = AL.stepper(svg, {
      frames: exp.frames, delay: 480, label: "iteration",
      render: (f, i) => draw(f, i, exp.frames.length, exp, rec, same)
    });

    DC.table("#es-table",
      ["Quantity", "Recursive version (measured)", "Explicit-stack version (measured)", "check"],
      [["records / frames created", "<b>" + DC.int(rec.calls) + "</b> invocations",
        "<b>" + DC.int(exp.pushes) + "</b> pushes", DC.flag(rec.calls === exp.pushes)],
       ["peak live records", "<b>" + DC.int(rec.depth) + "</b> stack frames",
        "<b>" + DC.int(exp.peak) + "</b> records", DC.flag(rec.depth === exp.peak)],
       ["moves emitted", "<b>" + DC.int(rec.nmoves) + "</b>", "<b>" + DC.int(exp.nmoves) + "</b>",
        DC.flag(rec.nmoves === exp.nmoves)],
       ["move sequence, position by position", "the reference",
        same ? "identical at every position" : "<b>differs</b>", DC.flag(same)]]);
  }

  function draw(f, idx, nf, exp, rec, same) {
    const fr = AL.frame(svg, W, H, { l: 16, r: 16, t: 14, b: 14 });
    const g = fr.g;
    const bw = 300, bh = 26, x0 = 4;

    g.append("text").attr("x", x0).attr("y", 10).attr("font-size", 11).attr("fill", AC.muted)
      .text("explicit stack — top of stack at the top");
    if (f.stack.length === 0) {
      g.append("text").attr("x", x0).attr("y", 34).attr("font-size", 12).attr("fill", AC.good)
        .text("stack empty — the loop ends, exactly as the recursion returns to its caller");
    }
    f.stack.slice().reverse().forEach((rc, k) => {
      const y = 20 + k * (bh + 4);
      if (y + bh > fr.ih - 40) return;
      g.append("rect").attr("x", x0).attr("y", y).attr("width", bw).attr("height", bh).attr("rx", 5)
        .attr("fill", k === 0 ? AC.panel2 : AC.panel).attr("stroke", k === 0 ? AC.accent : AC.line)
        .attr("stroke-width", k === 0 ? 2 : 1);
      g.append("text").attr("x", x0 + 10).attr("y", y + 17).attr("font-size", 12).attr("fill", AC.ink)
        .text("n = " + rc.n + "   " + PEG[rc.from] + " → " + PEG[rc.to]
              + "   via " + PEG[rc.via] + "   · stage " + rc.stage);
      if (k === 0) g.append("text").attr("x", x0 + bw + 8).attr("y", y + 17)
        .attr("font-size", 11).attr("fill", AC.accent).text("← top");
    });

    const mx = 360;
    g.append("text").attr("x", mx).attr("y", 10).attr("font-size", 11).attr("fill", AC.muted)
      .text("moves emitted so far (" + f.moves.length + " of " + exp.nmoves + ")");
    f.moves.forEach((m, k) => {
      const col = k % 2, row = Math.floor(k / 2);
      const y = 22 + row * 17;
      if (y > fr.ih - 42) return;
      g.append("text").attr("x", mx + col * 140).attr("y", y).attr("font-size", 11)
        .attr("fill", k === f.moves.length - 1 ? AC.a2 : AC.muted)
        .text((k + 1) + ".  disk " + m.disk + "   " + PEG[m.from] + " → " + PEG[m.to]);
    });

    g.append("text").attr("x", x0).attr("y", fr.ih - 8).attr("font-size", 12).attr("fill", AC.ink)
      .text(f.note);

    d3.select("#es-readout").html(
      "iteration <b>" + (idx + 1) + "</b> / " + nf
      + " &nbsp;·&nbsp; live records now <b>" + f.stack.length + "</b>"
      + " &nbsp;·&nbsp; peak over the whole run <b>" + exp.peak + "</b>"
      + " vs the recursion's measured peak depth <b>" + rec.depth + "</b> " + DC.flag(exp.peak === rec.depth)
      + " &nbsp;·&nbsp; emitted move sequence vs the recursive one: " + DC.flag(same));
  }

  d3.select("#es-n").on("input", function () {
    n = +this.value; d3.select("#es-n-out").text(n); build();
  });
  d3.select("#es-n-out").text(n);
  build();
})();

/* ══ FIGURE 05 ═══════════════════════════════════════════════════════════
   Naive versus memoized recursion, both instrumented. The memoized count is
   compared against 2n − 1 (valid for n ≥ 1) and the naive against
   2·F(n+1) − 1, each evaluated from the independent iterative Fibonacci.   */
(function () {
  const svg = d3.select("#mm-svg"); if (svg.empty()) return;
  const W = 680, H = 290;
  let nmax = 20;

  function naive(n) {
    const c = AL.counter();
    (function f(k) { c.add("calls"); return k <= 1 ? k : f(k - 1) + f(k - 2); })(n);
    return c.get("calls");
  }
  function memoized(n) {
    const c = AL.counter(), memo = Object.create(null);
    (function f(k) {
      c.add("calls");
      if (memo[k] !== undefined) return memo[k];
      const v = k <= 1 ? k : f(k - 1) + f(k - 2);
      memo[k] = v; return v;
    })(n);
    return c.get("calls");
  }

  function build() {
    const data = [];
    for (let n = 1; n <= nmax; n++) data.push({ n: n, naive: naive(n), memo: memoized(n) });
    draw(data);

    const rows = [];
    [Math.max(1, Math.round(nmax / 2)), nmax].forEach(n => {
      const d = data[n - 1];
      const pn = 2 * DC.fib(n + 1) - 1, pm = 2 * n - 1;
      rows.push(["<code>n = " + n + "</code>",
                 "<b>" + DC.int(d.naive) + "</b>", "<code>2·F(n+1) − 1</code> = " + DC.int(pn)
                 + " " + DC.flag(d.naive === pn),
                 "<b>" + DC.int(d.memo) + "</b>", "<code>2n − 1</code> = " + DC.int(pm)
                 + " " + DC.flag(d.memo === pm),
                 "<b>" + DC.int(Math.round(d.naive / d.memo)) + "×</b>"]);
    });
    DC.table("#mm-table",
      ["", "Naive invocations", "predicted", "Memoized invocations", "predicted", "ratio"], rows);

    const last = data[data.length - 1];
    d3.select("#mm-readout").html(
      "at n = <b>" + nmax + "</b> &nbsp;·&nbsp; naive <b>" + DC.int(last.naive)
      + "</b> invocations &nbsp;·&nbsp; memoized <b>" + DC.int(last.memo)
      + "</b> &nbsp;·&nbsp; ratio <b>" + DC.int(Math.round(last.naive / last.memo))
      + "×</b> &nbsp;·&nbsp; distinct subproblems <b>" + (nmax + 1) + "</b>"
      + " &nbsp;·&nbsp; both counts are measurements, each checked against its closed form");
  }

  function draw(data) {
    const fr = AL.frame(svg, W, H, { l: 62, r: 16, t: 18, b: 40 });
    const g = fr.g;
    const x = d3.scaleLinear().domain([1, nmax]).range([0, fr.iw]);
    const y = d3.scaleLog().domain([1, d3.max(data, d => d.naive) * 1.6]).range([fr.ih, 0]);
    AL.gridY(g, y, fr.iw, 6);
    AL.axisB(g, x, fr.ih, 6, "n");
    AL.axisL(g, y, 6, "invocations (log scale)", d3.format("~s"));
    const ln = (key, col) => g.append("path").datum(data).attr("fill", "none")
      .attr("stroke", col).attr("stroke-width", 2.2)
      .attr("d", d3.line().x(d => x(d.n)).y(d => y(Math.max(1, d[key]))));
    ln("naive", AC.rose); ln("memo", AC.good);
    g.selectAll("circle.a").data(data).join("circle").attr("class", "a")
      .attr("cx", d => x(d.n)).attr("cy", d => y(Math.max(1, d.naive))).attr("r", 2).attr("fill", AC.rose);
    g.selectAll("circle.b").data(data).join("circle").attr("class", "b")
      .attr("cx", d => x(d.n)).attr("cy", d => y(Math.max(1, d.memo))).attr("r", 2).attr("fill", AC.good);
    AL.legend(g, [
      { label: "naive recursion — measured, ≈ <span>φ</span>ⁿ", color: AC.rose },
      { label: "memoized recursion — measured, 2n − 1", color: AC.good }
    ], 8, 18);
  }

  d3.select("#mm-n").on("input", function () {
    nmax = +this.value; d3.select("#mm-n-out").text(nmax); build();
  });
  d3.select("#mm-n-out").text(nmax);
  build();
})();

/* ══ FIGURE 06 ═══════════════════════════════════════════════════════════
   Mergesort, stepped one completed merge at a time. The frames come from
   running the real routine; comparisons and element moves come from
   AL.counter(). Each merge's measured comparison count is checked against
   the independent identity  comparisons = m − t, where t is the length of
   the trailing output run that came from a single side, and the total is
   checked against the bounds ∑min(p,q) ≤ total ≤ ∑(m−1).                   */
(function () {
  const svg = d3.select("#mg-svg"); if (svg.empty()) return;
  const W = 680, H = 240;

  /* an input constructed so every merge interleaves — the true worst case */
  function adversarial(vals) {
    const m = vals.length;
    if (m <= 1) return vals.slice();
    const p = Math.floor(m / 2), q = m - p, L = [], R = [];
    let toR = true;
    for (let i = m - 1; i >= 0; i--) {
      if (toR && R.length < q) R.unshift(vals[i]);
      else if (L.length < p) L.unshift(vals[i]);
      else R.unshift(vals[i]);
      toR = !toR;
    }
    return adversarial(L).concat(adversarial(R));
  }

  const PRESETS = {
    worked:      [38, 27, 43, 3, 9, 82, 10, 1],
    sorted:      [1, 3, 9, 10, 27, 38, 43, 82],
    reversed:    [82, 43, 38, 27, 10, 9, 3, 1],
    adversarial: adversarial([1, 2, 3, 4, 5, 6, 7, 8]),
    worked16:    [38, 27, 43, 3, 9, 82, 10, 1, 55, 14, 76, 22, 61, 5, 90, 31]
  };

  function mergesort(input, c) {
    const a = [...input], events = [];
    (function rec(lo, hi) {
      if (hi - lo <= 1) return;
      const mid = lo + Math.floor((hi - lo) / 2);
      rec(lo, mid); rec(mid, hi);
      const L = a.slice(lo, mid), R = a.slice(mid, hi);
      const before = a.slice(lo, hi);
      let i = 0, j = 0, k = lo, cmp = 0;
      const side = [];                                  // which run each output came from
      while (i < L.length && j < R.length) {
        c.add("cmp"); cmp++;
        if (L[i] <= R[j]) { a[k++] = L[i++]; side.push(0); }
        else { a[k++] = R[j++]; side.push(1); }
        c.add("move");
      }
      while (i < L.length) { a[k++] = L[i++]; side.push(0); c.add("move"); }
      while (j < R.length) { a[k++] = R[j++]; side.push(1); c.add("move"); }
      /* t = length of the trailing output run from one single side */
      let t = 1;
      for (let s = side.length - 2; s >= 0 && side[s] === side[side.length - 1]; s--) t++;
      events.push({ lo, mid, hi, cmp, t, p: L.length, q: R.length,
                    before: before, after: a.slice(lo, hi), snapshot: [...a] });
    })(0, a.length);
    return { out: a, events: events };
  }

  let key = "worked", st = null;

  function build() {
    const input = PRESETS[key];
    const c = AL.counter();
    const r = mergesort(input, c);
    const total = c.all();

    const frames = [{ a: [...input], ev: null, cum: { cmp: 0, move: 0 } }];
    let cc = 0, cm = 0;
    r.events.forEach(e => {
      cc += e.cmp; cm += (e.hi - e.lo);
      frames.push({ a: [...e.snapshot], ev: e, cum: { cmp: cc, move: cm } });
    });

    if (st) st.pause();
    DC.clearControls(svg.node());
    st = AL.stepper(svg, {
      frames: frames, delay: 720, label: "merge",
      render: (f, i) => draw(f, i, frames.length, total, r)
    });

    const lower = r.events.reduce((s, e) => s + Math.min(e.p, e.q), 0);
    const upper = r.events.reduce((s, e) => s + (e.hi - e.lo - 1), 0);
    const movesIndep = r.events.reduce((s, e) => s + (e.hi - e.lo), 0);
    const n = input.length;
    const cl = Math.ceil(AL.log2(n) - 1e-9);
    const worstFormula = n * cl - Math.pow(2, cl) + 1;
    const sortedOK = r.out.every((v, i) => i === 0 || r.out[i - 1] <= v);

    const rows = r.events.map(e => [
      "<code>A[" + e.lo + ".." + (e.mid - 1) + "]</code> + <code>A[" + e.mid + ".." + (e.hi - 1) + "]</code>",
      e.p + " + " + e.q + " = " + (e.hi - e.lo),
      "<b>" + e.cmp + "</b>",
      "m − t = " + (e.hi - e.lo) + " − " + e.t + " = " + ((e.hi - e.lo) - e.t) + " "
        + DC.flag(e.cmp === (e.hi - e.lo) - e.t)
    ]);
    rows.push([
      "<b>total over " + r.events.length + " merges</b>",
      "moves <b>" + DC.int(total.move || 0) + "</b> = ∑ m = " + DC.int(movesIndep) + " "
        + DC.flag((total.move || 0) === movesIndep),
      "<b>" + DC.int(total.cmp || 0) + "</b>",
      "bounds ∑min(p,q) = " + lower + " ≤ " + (total.cmp || 0) + " ≤ ∑(m−1) = " + upper + " "
        + DC.flag((total.cmp || 0) >= lower && (total.cmp || 0) <= upper)
        + " &nbsp;· worst case n·ceil(log₂n) − 2^ceil(log₂n) + 1 = " + worstFormula
        + " " + DC.flag(upper === worstFormula)
        + " &nbsp;· output sorted " + DC.flag(sortedOK)
    ]);
    DC.table("#mg-table", ["Merge", "sizes p + q = m", "Comparisons (measured)", "cross-check"], rows);
  }

  function draw(f, idx, nf, total, r) {
    const fr = AL.frame(svg, W, H, { l: 20, r: 20, t: 16, b: 14 });
    const g = fr.g, n = f.a.length;
    const cw = Math.min(46, (fr.iw - 10) / n - 4);
    const e = f.ev;

    const row = AL.row(g, f.a, {
      x: 4, y: 40, w: cw, h: 34, gap: 4, fontSize: cw < 34 ? 11 : 13,
      mark: (i) => {
        if (!e) return AC.panel2;
        if (i >= e.lo && i < e.mid) return "#22304a";
        if (i >= e.mid && i < e.hi) return "#3a2f1e";
        return AC.panel2;
      }
    });
    const step = cw + 4;

    g.append("text").attr("x", 4).attr("y", 16).attr("font-size", 12).attr("fill", AC.ink)
      .text(e ? ("merge " + idx + " of " + (nf - 1) + ":  A[" + e.lo + ".." + (e.mid - 1)
                 + "] (" + e.p + ") with A[" + e.mid + ".." + (e.hi - 1) + "] (" + e.q + ")"
                 + "  →  " + e.cmp + " comparisons, " + (e.hi - e.lo) + " element moves")
                : "the input, before any merge — every single element is a sorted run of length 1");
    if (e) {
      g.append("text").attr("x", 4).attr("y", 32).attr("font-size", 11).attr("fill", AC.muted)
        .text("left run shaded blue · right run shaded amber · the bracket below marks what is now sorted");
      const x1 = 4 + e.lo * step, x2 = 4 + e.hi * step - 4;
      g.append("path").attr("d", "M" + x1 + ",96 L" + x1 + ",104 L" + x2 + ",104 L" + x2 + ",96")
        .attr("fill", "none").attr("stroke", AC.good).attr("stroke-width", 2);
      g.append("text").attr("x", (x1 + x2) / 2).attr("y", 118).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("fill", AC.good)
        .text("A[" + e.lo + ".." + (e.hi - 1) + "] sorted");
      g.append("text").attr("x", 4).attr("y", 142).attr("font-size", 11).attr("fill", AC.muted)
        .text("this merge emitted a trailing run of " + e.t + " element"
              + (e.t === 1 ? "" : "s") + " from one side without comparing, so comparisons = "
              + (e.hi - e.lo) + " − " + e.t + " = " + e.cmp);
    }

    g.append("text").attr("x", 4).attr("y", fr.ih - 6).attr("font-size", 12).attr("fill", AC.muted)
      .text("cumulative — comparisons " + f.cum.cmp + " of " + (total.cmp || 0)
            + "   ·   element moves " + f.cum.move + " of " + (total.move || 0));

    d3.select("#mg-readout").html(
      "step <b>" + (idx + 1) + "</b> / " + nf + " &nbsp;·&nbsp; comparisons so far <b>" + f.cum.cmp
      + "</b> &nbsp;·&nbsp; element moves so far <b>" + f.cum.move
      + "</b> &nbsp;·&nbsp; whole run: <b>" + DC.int(total.cmp || 0) + "</b> comparisons, <b>"
      + DC.int(total.move || 0) + "</b> moves over <b>" + r.events.length + "</b> merges");
  }

  d3.select("#mg-input").on("change", function () { key = this.value; build(); });
  build();
})();

/* ══ FIGURE 07 ═══════════════════════════════════════════════════════════
   Maximum subarray. The stepper walks the top-level crossing scan — left
   from mid, then right from mid+1 — showing the running sum and the best so
   far in each direction. The measured costs come from AL.counter() wrapped
   round the real recursion; the answer is computed three independent ways
   (divide and conquer, exhaustive search, the linear scan) and the three
   are compared on both the sum and the index range.                        */
(function () {
  const svg = d3.select("#ms-svg"); if (svg.empty()) return;
  const W = 680, H = 300;

  const PRESETS = {
    worked:   [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7],
    allneg:   [-4, -2, -7, -1, -5, -3, -9, -6],
    allpos:   [3, 1, 4, 1, 5, 9, 2, 6],
    leftonly: [2, 9, 8, 7, -30, -1, -2, -3, -4, -5, -6, -7]
  };

  /* (1) divide and conquer, instrumented */
  function crossing(A, lo, mid, hi, c) {
    let s = 0, bl = -Infinity, il = mid;
    for (let i = mid; i >= lo; i--) { c.add("look"); s += A[i]; if (s > bl) { bl = s; il = i; } }
    s = 0; let br = -Infinity, ir = mid + 1;
    for (let j = mid + 1; j <= hi; j++) { c.add("look"); s += A[j]; if (s > br) { br = s; ir = j; } }
    return { lo: il, hi: ir, sum: bl + br };
  }
  function dcMax(A, lo, hi, c) {
    c.add("calls");
    if (lo === hi) return { lo: lo, hi: hi, sum: A[lo] };
    const mid = Math.floor((lo + hi) / 2);
    const L = dcMax(A, lo, mid, c), R = dcMax(A, mid + 1, hi, c);
    const X = crossing(A, lo, mid, hi, c);
    if (L.sum >= R.sum && L.sum >= X.sum) return L;
    if (R.sum >= L.sum && R.sum >= X.sum) return R;
    return X;
  }
  /* (2) exhaustive, instrumented */
  function bruteMax(A, c) {
    let best = { lo: 0, hi: 0, sum: -Infinity };
    for (let i = 0; i < A.length; i++) {
      let s = 0;
      for (let j = i; j < A.length; j++) { c.add("sums"); s += A[j]; if (s > best.sum) best = { lo: i, hi: j, sum: s }; }
    }
    return best;
  }
  /* (3) the linear scan, instrumented */
  function scanMax(A, c) {
    let cur = A[0], start = 0, best = A[0], bl = 0, bh = 0;
    c.add("steps");
    for (let j = 1; j < A.length; j++) {
      c.add("steps");
      if (cur + A[j] >= A[j]) cur = cur + A[j]; else { cur = A[j]; start = j; }
      if (cur > best) { best = cur; bl = start; bh = j; }
    }
    return { lo: bl, hi: bh, sum: best };
  }

  let key = "worked", st = null;

  function build() {
    const A = PRESETS[key], n = A.length;
    const cDC = AL.counter(), cBF = AL.counter(), cSC = AL.counter();
    const dc = dcMax(A, 0, n - 1, cDC);
    const bf = bruteMax(A, cBF);
    const sc = scanMax(A, cSC);
    const agreeSum = (dc.sum === bf.sum) && (bf.sum === sc.sum);
    const agreeIdx = (dc.lo === bf.lo && dc.hi === bf.hi && bf.lo === sc.lo && bf.hi === sc.hi);

    /* the top-level crossing scan, frame by frame */
    const mid = Math.floor((n - 1) / 2);
    const frames = [{ phase: "split", note: "split at mid = " + mid + ": the left half is A[0.."
                      + mid + "], the right half is A[" + (mid + 1) + ".." + (n - 1) + "]" }];
    let s = 0, bl = -Infinity, il = mid;
    for (let i = mid; i >= 0; i--) {
      s += A[i]; if (s > bl) { bl = s; il = i; }
      frames.push({ phase: "left", i: i, run: s, best: bl, bestI: il,
                    note: "walking LEFT from mid: add A[" + i + "] = " + A[i]
                        + " → running sum " + s + ", best left-piece so far " + bl
                        + " starting at index " + il });
    }
    let s2 = 0, br = -Infinity, ir = mid + 1;
    for (let j = mid + 1; j < n; j++) {
      s2 += A[j]; if (s2 > br) { br = s2; ir = j; }
      frames.push({ phase: "right", j: j, run: s2, best: br, bestJ: ir, bl: bl, bestI: il,
                    note: "walking RIGHT from mid+1: add A[" + j + "] = " + A[j]
                        + " → running sum " + s2 + ", best right-piece so far " + br
                        + " ending at index " + ir });
    }
    frames.push({ phase: "done", bl: bl, bestI: il, br: br, bestJ: ir,
                  note: "best crossing subarray = A[" + il + ".." + ir + "], sum " + bl
                      + " + " + br + " = " + (bl + br) });

    if (st) st.pause();
    DC.clearControls(svg.node());
    st = AL.stepper(svg, {
      frames: frames, delay: 560, label: "step",
      render: (f, i) => draw(A, mid, f, i, frames.length, dc)
    });

    const nSub = n * (n + 1) / 2;
    DC.table("#ms-table",
      ["Method", "Answer found", "Sum", "Measured cost"],
      [["divide and conquer, <code>T(n) = 2T(n/2) + <span class='keep'>Θ</span>(n)</code>",
        "<code>A[" + dc.lo + ".." + dc.hi + "]</code>", "<b>" + dc.sum + "</b>",
        "<b>" + DC.int(cDC.get("calls")) + "</b> invocations, <b>" + DC.int(cDC.get("look"))
        + "</b> element inspections in the crossing scans"],
       ["exhaustive over all subarrays", "<code>A[" + bf.lo + ".." + bf.hi + "]</code>",
        "<b>" + bf.sum + "</b>",
        "<b>" + DC.int(cBF.get("sums")) + "</b> subarray sums · <code>n(n+1)/2</code> = "
        + DC.int(nSub) + " " + DC.flag(cBF.get("sums") === nSub)],
       ["the linear scan of §17", "<code>A[" + sc.lo + ".." + sc.hi + "]</code>",
        "<b>" + sc.sum + "</b>",
        "<b>" + DC.int(cSC.get("steps")) + "</b> steps · <code>n</code> = " + n + " "
        + DC.flag(cSC.get("steps") === n)],
       ["<b>three-way agreement</b>", agreeIdx ? "same index range" : "<b>index ranges differ</b>",
        DC.flag(agreeSum), agreeIdx ? DC.flag(true)
        : "ranges can differ only when two subarrays tie on the sum"]]);
  }

  function draw(A, mid, f, idx, nf, dc) {
    const fr = AL.frame(svg, W, H, { l: 18, r: 18, t: 16, b: 14 });
    const g = fr.g, n = A.length;
    const cw = Math.min(40, (fr.iw - 6) / n - 3), step = cw + 3;

    const row = AL.row(g, A, {
      x: 2, y: 36, w: cw, h: 32, gap: 3, fontSize: cw < 30 ? 10 : 12,
      mark: (i) => {
        if (f.phase === "left" && i === f.i) return AC.violet;
        if (f.phase === "right" && i === f.j) return AC.violet;
        if (i <= mid) return "#22304a";
        return "#3a2f1e";
      }
    });
    g.append("text").attr("x", 2).attr("y", 14).attr("font-size", 12).attr("fill", AC.ink)
      .text("A[0.." + (n - 1) + "] — left half blue, right half amber, current cell violet");
    const xm = 2 + (mid + 1) * step - 1.5;
    g.append("line").attr("x1", xm).attr("x2", xm).attr("y1", 28).attr("y2", 84)
      .attr("stroke", AC.rose).attr("stroke-width", 2).attr("stroke-dasharray", "3 3");
    g.append("text").attr("x", xm).attr("y", 24).attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", AC.rose).text("split");

    /* the best piece found so far, bracketed */
    const bracket = (a, b, col, lab, y) => {
      const x1 = 2 + a * step, x2 = 2 + b * step + cw;
      g.append("path").attr("d", "M" + x1 + "," + (y - 6) + " L" + x1 + "," + y
                                + " L" + x2 + "," + y + " L" + x2 + "," + (y - 6))
        .attr("fill", "none").attr("stroke", col).attr("stroke-width", 2);
      g.append("text").attr("x", (x1 + x2) / 2).attr("y", y + 13).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("fill", col).text(lab);
    };
    if (f.phase === "left") bracket(f.bestI, mid, AC.accent, "best left piece = " + f.best, 96);
    if (f.phase === "right") {
      bracket(f.bestI, mid, AC.accent, "best left = " + f.bl, 96);
      bracket(mid + 1, f.bestJ, AC.a2, "best right = " + f.best, 96);
    }
    if (f.phase === "done") bracket(f.bestI, f.bestJ, AC.good,
      "best CROSSING subarray, sum " + (f.bl + f.br), 96);

    g.append("text").attr("x", 2).attr("y", 140).attr("font-size", 12).attr("fill", AC.ink)
      .text(f.note);
    g.append("text").attr("x", 2).attr("y", 164).attr("font-size", 11).attr("fill", AC.muted)
      .text("the recursion's own answer over the whole array: A[" + dc.lo + ".." + dc.hi
            + "], sum " + dc.sum + "  — the crossing case above is only one of its three candidates");

    d3.select("#ms-readout").html(
      "step <b>" + (idx + 1) + "</b> / " + nf + " &nbsp;·&nbsp; " + f.note
      + " &nbsp;·&nbsp; overall answer <b>A[" + dc.lo + ".." + dc.hi + "]</b> with sum <b>"
      + dc.sum + "</b>");
  }

  d3.select("#ms-input").on("change", function () { key = this.value; build(); });
  build();
})();

/* ══ FIGURE 08 ═══════════════════════════════════════════════════════════
   Karatsuba against the schoolbook split. Both are real recursive routines
   over BigInt operands; the counter is bumped once per SINGLE-DIGIT
   multiplication (the base case), so the counts are the algorithms' own.
   Every product is compared with exact BigInt arithmetic, and the exponent
   is measured from consecutive counts rather than quoted from the recurrence. */
(function () {
  const svg = d3.select("#kr-svg"); if (svg.empty()) return;
  const W = 680, H = 320;
  const P10 = m => 10n ** BigInt(m);

  function school(x, y, n, c) {
    if (n === 1) { c.add("mul"); return x * y; }
    const m = n / 2, P = P10(m);
    const xh = x / P, xl = x % P, yh = y / P, yl = y % P;
    const a = school(xh, yh, m, c), b = school(xh, yl, m, c);
    const d = school(xl, yh, m, c), e = school(xl, yl, m, c);
    c.add("add", 3);
    return a * P * P + (b + d) * P + e;
  }
  function karatsuba(x, y, n, c) {
    if (n === 1) { c.add("mul"); return x * y; }
    const m = n / 2, P = P10(m);
    const xh = x / P, xl = x % P, yh = y / P, yl = y % P;
    const p = karatsuba(xh, yh, m, c);              // P = x₁y₁
    const q = karatsuba(xl, yl, m, c);              // Q = x₀y₀
    const sx = xh + xl, sy = yh + yl;               // may be one digit longer
    const cx = sx / P, rx = sx % P, cy = sy / P, ry = sy % P;
    const rr = karatsuba(rx, ry, m, c);             // the only half-size call left
    const r = cx * cy * P * P + (cx * ry + cy * rx) * P + rr;     // = sx·sy
    c.add("add", 8);
    return p * P * P + (r - p - q) * P + q;
  }

  let mode = "worked";

  function build() {
    if (mode === "worked") drawWorked(); else drawCounts();
  }

  function drawWorked() {
    const fr = AL.frame(svg, W, H, { l: 18, r: 18, t: 16, b: 14 });
    const g = fr.g;
    const x = 1234n, y = 5678n, B = 100n;
    const x1 = x / B, x0 = x % B, y1 = y / B, y0 = y % B;
    /* the four schoolbook subproducts and the three Karatsuba ones, computed here */
    const s11 = x1 * y1, s10 = x1 * y0, s01 = x0 * y1, s00 = x0 * y0;
    const sMid = s10 + s01;
    const kP = x1 * y1, kQ = x0 * y0, kR = (x1 + x0) * (y1 + y0), kMid = kR - kP - kQ;
    const assembled = kP * 10000n + kMid * 100n + kQ;
    const exact = x * y;

    g.append("text").attr("x", 0).attr("y", 12).attr("font-size", 12).attr("fill", AC.ink)
      .text("x = 1234 = " + x1 + "·10² + " + x0 + "      y = 5678 = " + y1 + "·10² + " + y0);

    const col = (ox, title, colr, lines, foot) => {
      g.append("rect").attr("x", ox).attr("y", 26).attr("width", 316).attr("height", 200)
        .attr("rx", 8).attr("fill", AC.panel2).attr("stroke", colr).attr("stroke-width", 1.5);
      g.append("text").attr("x", ox + 12).attr("y", 46).attr("font-size", 12.5)
        .attr("font-weight", 600).attr("fill", colr).text(title);
      lines.forEach((t, i) => g.append("text").attr("x", ox + 12).attr("y", 70 + i * 21)
        .attr("font-size", 12).attr("fill", AC.ink).text(t));
      g.append("text").attr("x", ox + 12).attr("y", 214).attr("font-size", 11.5)
        .attr("fill", AC.muted).text(foot);
    };
    col(0, "the schoolbook split — FOUR products", AC.rose, [
      "x₁·y₁ = " + x1 + " × " + y1 + " = " + s11,
      "x₁·y₀ = " + x1 + " × " + y0 + " = " + s10,
      "x₀·y₁ = " + x0 + " × " + y1 + " = " + s01,
      "x₀·y₀ = " + x0 + " × " + y0 + " = " + s00,
      "middle = " + s10 + " + " + s01 + " = " + sMid,
      "x·y = " + s11 + "·10⁴ + " + sMid + "·10² + " + s00
    ], "T(n) = 4·T(n/2) + Θ(n)  ⇒  Θ(n²)");
    col(336, "Karatsuba — THREE products", AC.good, [
      "P = x₁·y₁ = " + x1 + " × " + y1 + " = " + kP,
      "Q = x₀·y₀ = " + x0 + " × " + y0 + " = " + kQ,
      "R = (x₁+x₀)(y₁+y₀) = " + (x1 + x0) + " × " + (y1 + y0) + " = " + kR,
      "middle = R − P − Q = " + kR + " − " + kP + " − " + kQ + " = " + kMid,
      "  ← the SAME middle coefficient, " + (kMid === sMid ? "confirmed" : "MISMATCH"),
      "x·y = " + kP + "·10⁴ + " + kMid + "·10² + " + kQ
    ], "T(n) = 3·T(n/2) + Θ(n)  ⇒  Θ(n^log₂3) = Θ(n^1.585)");

    g.append("text").attr("x", 0).attr("y", 250).attr("font-size", 13).attr("fill", AC.a2)
      .text("assembled: " + kP + "·10⁴ + " + kMid + "·10² + " + kQ + " = " + assembled
            + "      exact 1234 × 5678 = " + exact);

    /* now RUN both routines on this very input and report their counters */
    const c1 = AL.counter(), c2 = AL.counter();
    const r1 = school(x, y, 4, c1), r2 = karatsuba(x, y, 4, c2);
    g.append("text").attr("x", 0).attr("y", 274).attr("font-size", 12).attr("fill", AC.muted)
      .text("running the two real recursions on n = 4 digits: schoolbook makes "
            + c1.get("mul") + " single-digit multiplications, Karatsuba makes " + c2.get("mul"));

    d3.select("#kr-readout").html(
      "1234 × 5678 &nbsp;·&nbsp; middle coefficient by addition <b>" + sMid
      + "</b> and by subtraction <b>" + kMid + "</b> " + DC.flag(sMid === kMid)
      + " &nbsp;·&nbsp; assembled <b>" + assembled + "</b> vs exact <b>" + exact + "</b> "
      + DC.flag(assembled === exact)
      + " &nbsp;·&nbsp; measured single-digit multiplications: schoolbook <b>" + c1.get("mul")
      + "</b>, Karatsuba <b>" + c2.get("mul") + "</b> "
      + DC.flag(r1 === exact && r2 === exact));

    DC.table("#kr-table",
      ["At n = 4 digits", "Schoolbook split", "Karatsuba", "check"],
      [["recursive products at this level", "4", "3", "—"],
       ["single-digit multiplications, measured", "<b>" + c1.get("mul") + "</b>",
        "<b>" + c2.get("mul") + "</b>", "n² = 16 and 3^log₂n = 9 " + DC.flag(c1.get("mul") === 16 && c2.get("mul") === 9)],
       ["linear-time additions/subtractions, measured", "<b>" + c1.get("add") + "</b>",
        "<b>" + c2.get("add") + "</b>", "Karatsuba trades multiplications for additions"],
       ["product returned", "<b>" + r1 + "</b>", "<b>" + r2 + "</b>",
        "exact = " + exact + " " + DC.flag(r1 === exact && r2 === exact)]]);
  }

  function drawCounts() {
    const sizes = [2, 4, 8, 16, 32, 64, 128];
    const rows = [], data = [];
    let allOK = true;
    const rnd = AL.rng(20250916);
    sizes.forEach(n => {
      let x = 0n, y = 0n;
      for (let i = 0; i < n; i++) {
        x = x * 10n + BigInt(1 + Math.floor(rnd() * 9));
        y = y * 10n + BigInt(1 + Math.floor(rnd() * 9));
      }
      const c1 = AL.counter(), c2 = AL.counter();
      const r1 = school(x, y, n, c1), r2 = karatsuba(x, y, n, c2);
      const ok = (r1 === x * y) && (r2 === x * y);
      if (!ok) allOK = false;
      data.push({ n: n, school: c1.get("mul"), kara: c2.get("mul"), ok: ok });
    });
    for (let i = 1; i < data.length; i++) {
      data[i].slopeS = AL.log2(data[i].school / data[i - 1].school);
      data[i].slopeK = AL.log2(data[i].kara / data[i - 1].kara);
    }

    const fr = AL.frame(svg, W, H, { l: 64, r: 150, t: 18, b: 42 });
    const g = fr.g;
    const x = d3.scaleLog().domain([2, 128]).range([0, fr.iw]);
    const y = d3.scaleLog().domain([2, d3.max(data, d => d.school) * 1.5]).range([fr.ih, 0]);
    AL.gridY(g, y, fr.iw, 6);
    AL.axisB(g, x, fr.ih, 4, "n — number of digits in each operand", d3.format("d"));
    AL.axisL(g, y, 6, "single-digit multiplications (measured)", d3.format("~s"));
    const ln = (k, col) => g.append("path").datum(data).attr("fill", "none").attr("stroke", col)
      .attr("stroke-width", 2.4).attr("d", d3.line().x(d => x(d.n)).y(d => y(d[k])));
    ln("school", AC.rose); ln("kara", AC.good);
    ["school", "kara"].forEach((k, idx) => {
      g.selectAll("circle.c" + idx).data(data).join("circle").attr("class", "c" + idx)
        .attr("cx", d => x(d.n)).attr("cy", d => y(d[k])).attr("r", 3)
        .attr("fill", idx ? AC.good : AC.rose);
    });
    AL.legend(g, [
      { label: "schoolbook split, a = 4", color: AC.rose },
      { label: "Karatsuba, a = 3", color: AC.good }
    ], 8, 20);

    const last = data[data.length - 1];
    g.append("text").attr("x", fr.iw + 8).attr("y", y(last.school) + 4).attr("font-size", 11)
      .attr("fill", AC.rose).text("slope " + last.slopeS.toFixed(3));
    g.append("text").attr("x", fr.iw + 8).attr("y", y(last.kara) + 4).attr("font-size", 11)
      .attr("fill", AC.good).text("slope " + last.slopeK.toFixed(3));

    data.forEach(d => rows.push([
      "<code>n = " + d.n + "</code>",
      "<b>" + DC.int(d.school) + "</b>", "n² = " + DC.int(d.n * d.n) + " "
        + DC.flag(d.school === d.n * d.n),
      "<b>" + DC.int(d.kara) + "</b>", "3^log₂n = " + DC.int(Math.round(Math.pow(3, AL.log2(d.n))))
        + " " + DC.flag(d.kara === Math.round(Math.pow(3, AL.log2(d.n)))),
      d.slopeK === undefined ? "—" : d.slopeS.toFixed(3) + " / " + d.slopeK.toFixed(3)
    ]));
    DC.table("#kr-table",
      ["Operand size", "Schoolbook (measured)", "predicted", "Karatsuba (measured)", "predicted",
       "measured slope log₂(cₙ/cₙ₋₁): school / Karatsuba"], rows);

    d3.select("#kr-readout").html(
      "at n = <b>128</b> digits &nbsp;·&nbsp; schoolbook <b>" + DC.int(last.school)
      + "</b> digit multiplications, Karatsuba <b>" + DC.int(last.kara)
      + "</b> &nbsp;·&nbsp; ratio <b>" + (last.school / last.kara).toFixed(1)
      + "×</b> &nbsp;·&nbsp; measured exponents <b>" + last.slopeS.toFixed(3) + "</b> and <b>"
      + last.slopeK.toFixed(3) + "</b> (log₂3 = " + AL.log2(3).toFixed(3) + ")"
      + " &nbsp;·&nbsp; every product checked against exact arithmetic " + DC.flag(allOK));
  }

  d3.select("#kr-mode").on("change", function () { mode = this.value; build(); });
  build();
})();

/* ══ FIGURE 09 ═══════════════════════════════════════════════════════════
   Strassen against the naive block recursion. Both are real routines that
   produce real matrices; the two results are compared entry by entry at
   every size, so a miscopied identity would show up as a failed check
   rather than as a plausible plot. Scalar multiplications and scalar
   additions/subtractions are counted separately, because the whole point is
   that the trade between them is what decides the crossover.               */
(function () {
  const svg = d3.select("#st-svg"); if (svg.empty()) return;
  const W = 680, H = 300;

  const zeros = n => Array.from({ length: n }, () => new Array(n).fill(0));
  function addM(A, B, c, sg) {
    const n = A.length, C = zeros(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { c.add("add"); C[i][j] = A[i][j] + sg * B[i][j]; }
    return C;
  }
  function blk(M, i0, j0, n) {
    const S = zeros(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) S[i][j] = M[i0 + i][j0 + j];
    return S;
  }
  function join4(A, B, C, D) {
    const m = A.length, n = 2 * m, M = zeros(n);
    for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) {
      M[i][j] = A[i][j]; M[i][j + m] = B[i][j]; M[i + m][j] = C[i][j]; M[i + m][j + m] = D[i][j];
    }
    return M;
  }
  function direct(A, B, c) {
    const n = A.length, C = zeros(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) { c.add("mul"); s += A[i][k] * B[k][j]; if (k > 0) c.add("add"); }
      C[i][j] = s;
    }
    return C;
  }
  function naive(A, B, c, cut) {
    const n = A.length;
    if (n <= cut) return direct(A, B, c);
    const m = n / 2;
    const A11 = blk(A, 0, 0, m), A12 = blk(A, 0, m, m), A21 = blk(A, m, 0, m), A22 = blk(A, m, m, m);
    const B11 = blk(B, 0, 0, m), B12 = blk(B, 0, m, m), B21 = blk(B, m, 0, m), B22 = blk(B, m, m, m);
    return join4(addM(naive(A11, B11, c, cut), naive(A12, B21, c, cut), c, 1),
                 addM(naive(A11, B12, c, cut), naive(A12, B22, c, cut), c, 1),
                 addM(naive(A21, B11, c, cut), naive(A22, B21, c, cut), c, 1),
                 addM(naive(A21, B12, c, cut), naive(A22, B22, c, cut), c, 1));
  }
  function strassen(A, B, c, cut) {
    const n = A.length;
    if (n <= cut) return direct(A, B, c);
    const m = n / 2;
    const A11 = blk(A, 0, 0, m), A12 = blk(A, 0, m, m), A21 = blk(A, m, 0, m), A22 = blk(A, m, m, m);
    const B11 = blk(B, 0, 0, m), B12 = blk(B, 0, m, m), B21 = blk(B, m, 0, m), B22 = blk(B, m, m, m);
    const M1 = strassen(addM(A11, A22, c, 1), addM(B11, B22, c, 1), c, cut);
    const M2 = strassen(addM(A21, A22, c, 1), B11, c, cut);
    const M3 = strassen(A11, addM(B12, B22, c, -1), c, cut);
    const M4 = strassen(A22, addM(B21, B11, c, -1), c, cut);
    const M5 = strassen(addM(A11, A12, c, 1), B22, c, cut);
    const M6 = strassen(addM(A21, A11, c, -1), addM(B11, B12, c, 1), c, cut);
    const M7 = strassen(addM(A12, A22, c, -1), addM(B21, B22, c, 1), c, cut);
    const C11 = addM(addM(M1, M4, c, 1), addM(M5, M7, c, -1), c, -1);   // M1 + M4 − M5 + M7
    const C12 = addM(M3, M5, c, 1);
    const C21 = addM(M2, M4, c, 1);
    const C22 = addM(addM(M1, M3, c, 1), addM(M2, M6, c, -1), c, -1);   // M1 − M2 + M3 + M6
    return join4(C11, C12, C21, C22);
  }

  const SIZES = [2, 4, 8, 16, 32, 64];
  let cut = 8, metric = "total";

  function build() {
    const rnd = AL.rng(4242), data = [];
    let allEq = true;
    SIZES.forEach(n => {
      if (n < cut) { data.push(null); return; }
      const A = zeros(n), B = zeros(n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        A[i][j] = AL.randInt(rnd, -4, 5); B[i][j] = AL.randInt(rnd, -4, 5);
      }
      const c1 = AL.counter(), c2 = AL.counter();
      const R1 = naive(A, B, c1, cut), R2 = strassen(A, B, c2, cut);
      let eq = true;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (R1[i][j] !== R2[i][j]) eq = false;
      if (!eq) allEq = false;
      data.push({ n: n, eq: eq,
        nm: c1.get("mul"), na: c1.get("add"), nt: c1.get("mul") + c1.get("add"),
        sm: c2.get("mul"), sa: c2.get("add"), st: c2.get("mul") + c2.get("add") });
    });
    const pts = data.filter(Boolean);
    draw(pts);

    /* the first size at which Strassen's chosen metric is strictly smaller */
    const keyN = metric === "total" ? "nt" : "nm", keyS = metric === "total" ? "st" : "sm";
    const cross = pts.find(d => d[keyS] < d[keyN]);
    const last = pts[pts.length - 1];

    DC.table("#st-table",
      ["n", "Naive: scalar × / ± / total", "Strassen: scalar × / ± / total",
       "multiplications ratio", "outputs identical"],
      pts.map(d => [
        "<code>" + d.n + "</code>",
        DC.int(d.nm) + " / " + DC.int(d.na) + " / <b>" + DC.int(d.nt) + "</b>",
        DC.int(d.sm) + " / " + DC.int(d.sa) + " / <b>" + DC.int(d.st) + "</b>"
          + (d.st < d.nt ? " <span style='color:" + AC.good + "'>◀ fewer</span>" : ""),
        (d.nm / d.sm).toFixed(3) + "×",
        DC.flag(d.eq)
      ]));

    d3.select("#st-readout").html(
      "cutoff <b>" + cut + "</b> &nbsp;·&nbsp; plotting <b>"
      + (metric === "total" ? "total scalar arithmetic operations" : "scalar multiplications")
      + "</b> &nbsp;·&nbsp; "
      + (cross ? "Strassen first uses fewer at n = <b>" + cross.n + "</b> ("
                 + DC.int(cross[keyS]) + " against " + DC.int(cross[keyN]) + ")"
               : "Strassen uses <b>more</b> at every size measured here, up to n = " + last.n)
      + " &nbsp;·&nbsp; at n = " + last.n + ": <b>" + DC.int(last[keyS]) + "</b> vs <b>"
      + DC.int(last[keyN]) + "</b>"
      + " &nbsp;·&nbsp; both routines produce identical matrices at every size " + DC.flag(allEq));
  }

  function draw(pts) {
    const fr = AL.frame(svg, W, H, { l: 70, r: 24, t: 18, b: 44 });
    const g = fr.g;
    const keyN = metric === "total" ? "nt" : "nm", keyS = metric === "total" ? "st" : "sm";
    const x = d3.scaleLog().domain([pts[0].n, pts[pts.length - 1].n]).range([0, fr.iw]);
    const lo = d3.min(pts, d => Math.min(d[keyN], d[keyS]));
    const hi = d3.max(pts, d => Math.max(d[keyN], d[keyS]));
    const y = d3.scaleLog().domain([lo / 1.6, hi * 1.6]).range([fr.ih, 0]);
    AL.gridY(g, y, fr.iw, 6);
    AL.axisB(g, x, fr.ih, 4, "n — matrix dimension", d3.format("d"));
    AL.axisL(g, y, 6, metric === "total" ? "scalar operations (measured)"
                                         : "scalar multiplications (measured)", d3.format("~s"));
    const ln = (k, col) => g.append("path").datum(pts).attr("fill", "none").attr("stroke", col)
      .attr("stroke-width", 2.4).attr("d", d3.line().x(d => x(d.n)).y(d => y(d[k])));
    ln(keyN, AC.rose); ln(keyS, AC.good);
    [[keyN, AC.rose, 0], [keyS, AC.good, 1]].forEach(([k, col, i]) => {
      g.selectAll("circle.s" + i).data(pts).join("circle").attr("class", "s" + i)
        .attr("cx", d => x(d.n)).attr("cy", d => y(d[k])).attr("r", 3).attr("fill", col);
    });
    AL.legend(g, [
      { label: "naive block recursion, 8 products per level", color: AC.rose },
      { label: "Strassen, 7 products and 18 add/subs per level", color: AC.good }
    ], 8, 20);
    const cross = pts.find(d => d[keyS] < d[keyN]);
    if (cross) {
      g.append("line").attr("x1", x(cross.n)).attr("x2", x(cross.n)).attr("y1", 0).attr("y2", fr.ih)
        .attr("stroke", AC.a2).attr("stroke-dasharray", "3 3").attr("stroke-width", 1.5);
      g.append("text").attr("x", x(cross.n) + 5).attr("y", fr.ih - 6).attr("font-size", 11)
        .attr("fill", AC.a2).text("Strassen ahead from n = " + cross.n);
    } else {
      g.append("text").attr("x", 8).attr("y", fr.ih - 6).attr("font-size", 11).attr("fill", AC.a2)
        .text("no crossover within the measured range at this cutoff");
    }
  }

  d3.select("#st-cut").on("input", function () {
    cut = Math.pow(2, +this.value); d3.select("#st-cut-out").text(cut); build();
  });
  d3.select("#st-metric").on("change", function () { metric = this.value; build(); });
  d3.select("#st-cut-out").text(cut);
  build();
})();

/* ══ FIGURE 10 ═══════════════════════════════════════════════════════════
   Closest pair of points. The recursive algorithm (presorted by y, y-order
   maintained by merging) and the exhaustive O(n²) search are both run; the
   distances they return are compared, and both distance-evaluation counts
   come from AL.counter(). The maximum number of strip partners examined for
   any one point is also measured, which is the constant the packing
   argument bounds.                                                         */
(function () {
  const svg = d3.select("#cp-svg"); if (svg.empty()) return;
  const W = 680, H = 330;
  let n = 48, seed = 9;

  const dist = (p, q) => Math.hypot(p.x - q.x, p.y - q.y);

  function brute(pts, c) {
    let best = Infinity, pair = null;
    for (let i = 0; i < pts.length; i++)
      for (let j = i + 1; j < pts.length; j++) {
        c.add("dist");
        const d = dist(pts[i], pts[j]);
        if (d < best) { best = d; pair = [pts[i], pts[j]]; }
      }
    return { d: best, pair: pair };
  }

  /* Px sorted by x, Py the same points sorted by y. Returns {d, pair, ys}
     where ys is this subproblem's points in y order, so the parent merges
     rather than re-sorts. */
  function rec(Px, Py, c, stats, top) {
    const m = Px.length;
    if (m <= 3) {
      const b = brute(Px, c);
      const ys = [...Px].sort((a, b2) => a.y - b2.y);
      return { d: b.d, pair: b.pair, ys: ys };
    }
    const half = Math.floor(m / 2);
    const Lx = Px.slice(0, half), Rx = Px.slice(half);
    const midX = Px[half].x;
    const inL = new Set(Lx.map(p => p.id));
    const Ly = Py.filter(p => inL.has(p.id)), Ry = Py.filter(p => !inL.has(p.id));
    stats.lines.push({ x: midX, m: m });
    const L = rec(Lx, Ly, c, stats, false), R = rec(Rx, Ry, c, stats, false);
    let best = L.d <= R.d ? L : R;
    let d = best.d, pair = best.pair;

    /* merge the two y-ordered lists — mergesort's combine, reused */
    const ys = [];
    let i = 0, j = 0;
    while (i < L.ys.length && j < R.ys.length) ys.push(L.ys[i].y <= R.ys[j].y ? L.ys[i++] : R.ys[j++]);
    while (i < L.ys.length) ys.push(L.ys[i++]);
    while (j < R.ys.length) ys.push(R.ys[j++]);

    const strip = ys.filter(p => Math.abs(p.x - midX) < d);
    for (let a = 0; a < strip.length; a++) {
      let looked = 0;
      for (let b = a + 1; b < strip.length && (strip[b].y - strip[a].y) < d; b++) {
        c.add("dist"); c.add("strip"); looked++;
        const dd = dist(strip[a], strip[b]);
        if (dd < d) { d = dd; pair = [strip[a], strip[b]]; }
      }
      if (looked > stats.maxLooked) stats.maxLooked = looked;
    }
    if (top) { stats.midX = midX; stats.delta = best.d; stats.strip = strip; }
    return { d: d, pair: pair, ys: ys };
  }

  function build() {
    const r = AL.rng(seed * 7919 + 11);
    const pts = Array.from({ length: n }, (_, i) => ({
      id: i, x: 20 + r() * 600, y: 20 + r() * 220
    }));
    const Px = [...pts].sort((a, b) => a.x - b.x || a.id - b.id);
    const Py = [...pts].sort((a, b) => a.y - b.y || a.id - b.id);
    const cDC = AL.counter(), cBF = AL.counter();
    const stats = { maxLooked: 0, midX: null, delta: null, strip: [], lines: [] };
    const dc = rec(Px, Py, cDC, stats, true);
    const bf = brute(pts, cBF);
    const agree = Math.abs(dc.d - bf.d) < 1e-9;

    draw(pts, dc, stats);

    const nPairs = n * (n - 1) / 2;
    DC.table("#cp-table",
      ["Method", "Closest distance found", "Distance evaluations (measured)", "check"],
      [["divide and conquer", dc.d.toFixed(4), "<b>" + DC.int(cDC.get("dist")) + "</b>",
        "<b>" + (cBF.get("dist") / cDC.get("dist")).toFixed(2) + "×</b> fewer than brute force"],
       ["exhaustive over all pairs", bf.d.toFixed(4), "<b>" + DC.int(cBF.get("dist")) + "</b>",
        "<code>n(n−1)/2</code> = " + DC.int(nPairs) + " " + DC.flag(cBF.get("dist") === nPairs)],
       ["the two answers", agree ? "identical to 1e−9" : "<b>DIFFERENT</b>", "—", DC.flag(agree)],
       ["of which, strip (boundary-case) evaluations", "—",
        "<b>" + DC.int(cDC.get("strip")) + "</b> of " + DC.int(cDC.get("dist")),
        "the rest are the size-≤3 base cases — the strip work is the minority"],
       ["most strip partners examined for one point", "—",
        "<b>" + stats.maxLooked + "</b>",
        "the packing argument bounds this by a constant (at most 7 in the box argument above), "
        + "independent of n " + DC.flag(stats.maxLooked <= 7)]]);
  }

  function draw(pts, dc, stats) {
    const fr = AL.frame(svg, W, H, { l: 14, r: 14, t: 16, b: 70 });
    const g = fr.g;
    const xs = d3.scaleLinear().domain([0, 640]).range([0, fr.iw]);
    const ys = d3.scaleLinear().domain([0, 260]).range([0, fr.ih]);
    const mx = xs(stats.midX), dpx = xs(stats.delta) - xs(0);

    stats.lines.forEach(L => {
      if (L.m === pts.length) return;
      g.append("line").attr("x1", xs(L.x)).attr("x2", xs(L.x)).attr("y1", 0).attr("y2", fr.ih)
        .attr("stroke", AC.line).attr("stroke-width", 1).attr("opacity", 0.8);
    });
    g.append("rect").attr("x", mx - dpx).attr("y", 0).attr("width", 2 * dpx).attr("height", fr.ih)
      .attr("fill", AC.a2).attr("opacity", 0.10);
    g.append("line").attr("x1", mx).attr("x2", mx).attr("y1", 0).attr("y2", fr.ih)
      .attr("stroke", AC.rose).attr("stroke-width", 2).attr("stroke-dasharray", "4 3");
    g.append("text").attr("x", mx + 5).attr("y", 12).attr("font-size", 11).attr("fill", AC.rose)
      .text("split at the median x");
    g.append("text").attr("x", mx - dpx + 4).attr("y", fr.ih - 6).attr("font-size", 11)
      .attr("fill", AC.a2).text("strip, width 2δ = " + (2 * stats.delta).toFixed(2));

    const inStrip = new Set(stats.strip.map(p => p.id));
    const inPair = new Set(dc.pair ? dc.pair.map(p => p.id) : []);
    g.selectAll("circle.p").data(pts).join("circle").attr("class", "p")
      .attr("cx", p => xs(p.x)).attr("cy", p => ys(p.y))
      .attr("r", p => inPair.has(p.id) ? 5 : 3)
      .attr("fill", p => inPair.has(p.id) ? AC.good : (inStrip.has(p.id) ? AC.a2 : AC.accent))
      .attr("opacity", p => inPair.has(p.id) ? 1 : (inStrip.has(p.id) ? 0.95 : 0.55));
    if (dc.pair) {
      g.append("line").attr("x1", xs(dc.pair[0].x)).attr("y1", ys(dc.pair[0].y))
        .attr("x2", xs(dc.pair[1].x)).attr("y2", ys(dc.pair[1].y))
        .attr("stroke", AC.good).attr("stroke-width", 2.5);
    }
    AL.legend(g, [
      { label: "point outside the strip", color: AC.accent },
      { label: "point inside the strip — a boundary candidate", color: AC.a2 },
      { label: "the closest pair found", color: AC.good },
      { label: "split lines of the deeper recursive calls", color: AC.line }
    ], 6, fr.ih + 20, { gap: 15 });

    d3.select("#cp-readout").html(
      "n = <b>" + pts.length + "</b> &nbsp;·&nbsp; <span class='keep'>δ</span> from the two recursive"
      + " calls <b>" + stats.delta.toFixed(3) + "</b> &nbsp;·&nbsp; final closest distance <b>"
      + dc.d.toFixed(4) + "</b>" + (dc.d < stats.delta - 1e-12
        ? " — improved inside the strip, so the answer is a crossing pair"
        : " — no strip pair beat it, so the answer lies wholly in one half")
      + " &nbsp;·&nbsp; points in the strip <b>" + stats.strip.length + "</b>"
      + " &nbsp;·&nbsp; most partners examined for one strip point <b>" + stats.maxLooked + "</b>");
  }

  d3.select("#cp-n").on("input", function () {
    n = +this.value; d3.select("#cp-n-out").text(n); build();
  });
  d3.select("#cp-seed").on("input", function () {
    seed = +this.value; d3.select("#cp-seed-out").text(seed); build();
  });
  d3.select("#cp-n-out").text(n); d3.select("#cp-seed-out").text(seed);
  build();
})();

/* ══ FIGURE 11 ═══════════════════════════════════════════════════════════
   Mergesort with an insertion-sort cutoff. Comparisons, element moves and
   invocations are all counted by AL.counter() inside the real routine, and
   every run's output is checked to be sorted. The weighted total adds w
   comparison-units per recursive call; w is a slider, i.e. a declared model
   parameter, and the readout says so — the three underlying counts are
   measurements, the weighting is a choice.                                 */
(function () {
  const svg = d3.select("#co-svg"); if (svg.empty()) return;
  const W = 680, H = 300;
  const CUTS = [1, 2, 4, 8, 16, 32, 64, 128];
  let lgN = 10, w = 8;

  function sortWithCutoff(A, cut, c) {
    const a = [...A];
    function insertion(lo, hi) {
      for (let i = lo + 1; i < hi; i++) {
        const key = a[i]; let j = i - 1;
        while (j >= lo) { c.add("cmp"); if (a[j] > key) { a[j + 1] = a[j]; c.add("mov"); j--; } else break; }
        a[j + 1] = key; c.add("mov");
      }
    }
    (function rec(lo, hi) {
      c.add("calls");
      if (hi - lo <= cut) { insertion(lo, hi); return; }
      const mid = lo + Math.floor((hi - lo) / 2);
      rec(lo, mid); rec(mid, hi);
      const L = a.slice(lo, mid), R = a.slice(mid, hi);
      let i = 0, j = 0, k = lo;
      while (i < L.length && j < R.length) {
        c.add("cmp"); c.add("mov");
        if (L[i] <= R[j]) a[k++] = L[i++]; else a[k++] = R[j++];
      }
      while (i < L.length) { a[k++] = L[i++]; c.add("mov"); }
      while (j < R.length) { a[k++] = R[j++]; c.add("mov"); }
    })(0, a.length);
    return a;
  }

  function build() {
    const n = Math.pow(2, lgN), TRIALS = 5;
    const data = [], bad = [];
    CUTS.forEach(cut => {
      let cmp = 0, mov = 0, calls = 0;
      for (let t = 0; t < TRIALS; t++) {
        const A = AL.perm(n, AL.rng(1000 + t));
        const c = AL.counter();
        const out = sortWithCutoff(A, cut, c);
        if (!out.every((v, i) => i === 0 || out[i - 1] <= v)) bad.push(cut);
        cmp += c.get("cmp"); mov += c.get("mov"); calls += c.get("calls");
      }
      data.push({ cut: cut, cmp: cmp / TRIALS, mov: mov / TRIALS, calls: calls / TRIALS });
    });
    data.forEach(d => { d.tot = d.cmp + d.mov + w * d.calls; });
    draw(data, n);

    const best = k => data.reduce((a, b) => (b[k] < a[k] ? b : a));
    DC.table("#co-table",
      ["Cutoff", "Comparisons (mean of 5 runs)", "Element moves", "Invocations",
       "Weighted total, w = " + w],
      data.map(d => [
        "<code>" + d.cut + "</code>",
        DC.int(Math.round(d.cmp)), DC.int(Math.round(d.mov)), DC.int(Math.round(d.calls)),
        "<b>" + DC.int(Math.round(d.tot)) + "</b>"
          + (d === best("tot") ? " <span style='color:" + AC.good + "'>◀ smallest</span>" : "")
      ]));

    d3.select("#co-readout").html(
      "n = <b>" + DC.int(n) + "</b>, five seeded permutations per cutoff &nbsp;·&nbsp; "
      + "fewest <b>comparisons</b> at cutoff <b>" + best("cmp").cut + "</b>"
      + " &nbsp;·&nbsp; fewest <b>element moves</b> at cutoff <b>" + best("mov").cut + "</b>"
      + " &nbsp;·&nbsp; fewest <b>invocations</b> at cutoff <b>" + best("calls").cut + "</b> (always the largest)"
      + " &nbsp;·&nbsp; smallest <b>weighted total</b> at cutoff <b>" + best("tot").cut
      + "</b> with w = " + w + " — w is a model parameter, not a measurement"
      + " &nbsp;·&nbsp; every run verified sorted " + DC.flag(bad.length === 0));
  }

  function draw(data, n) {
    const fr = AL.frame(svg, W, H, { l: 66, r: 20, t: 18, b: 56 });
    const g = fr.g;
    const x = d3.scaleLog().domain([1, 128]).range([0, fr.iw]);
    const hi = d3.max(data, d => Math.max(d.cmp, d.mov, d.calls, d.tot));
    const lo = d3.min(data, d => Math.min(d.cmp, d.mov, d.calls));
    const y = d3.scaleLog().domain([Math.max(1, lo / 2), hi * 1.5]).range([fr.ih, 0]);
    AL.gridY(g, y, fr.iw, 6);
    AL.axisB(g, x, fr.ih, 8, "cutoff — subarrays this size or smaller go to insertion sort", d3.format("d"));
    AL.axisL(g, y, 6, "measured count, mean over 5 runs", d3.format("~s"));
    const series = [
      { k: "cmp",   label: "comparisons (measured)",       col: AC.accent, dash: null },
      { k: "mov",   label: "element moves (measured)",     col: AC.teal,   dash: null },
      { k: "calls", label: "invocations (measured)",       col: AC.rose,   dash: null },
      { k: "tot",   label: "weighted total (model)",       col: AC.a2,     dash: "5 3" }
    ];
    series.forEach(s => {
      const p = g.append("path").datum(data).attr("fill", "none").attr("stroke", s.col)
        .attr("stroke-width", 2.2).attr("d", d3.line().x(d => x(d.cut)).y(d => y(Math.max(1, d[s.k]))));
      if (s.dash) p.attr("stroke-dasharray", s.dash);
      g.selectAll("circle." + s.k).data(data).join("circle").attr("class", s.k)
        .attr("cx", d => x(d.cut)).attr("cy", d => y(Math.max(1, d[s.k]))).attr("r", 2.6)
        .attr("fill", s.col);
    });
    const best = data.reduce((a, b) => (b.tot < a.tot ? b : a));
    g.append("line").attr("x1", x(best.cut)).attr("x2", x(best.cut)).attr("y1", 0).attr("y2", fr.ih)
      .attr("stroke", AC.a2).attr("stroke-dasharray", "2 3").attr("opacity", 0.7);
    AL.legend(g, series.map(s => ({ label: s.label, color: s.col, dash: s.dash })),
              6, fr.ih + 26, { gap: 14 });
    g.append("text").attr("x", fr.iw).attr("y", 12).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", AC.muted).text("n = " + DC.int(n));
  }

  d3.select("#co-n").on("input", function () {
    lgN = +this.value; d3.select("#co-n-out").text(Math.pow(2, lgN)); build();
  });
  d3.select("#co-w").on("input", function () {
    w = +this.value; d3.select("#co-w-out").text(w); build();
  });
  d3.select("#co-n-out").text(Math.pow(2, lgN)); d3.select("#co-w-out").text(w);
  build();
})();

/* ══ FIGURE 12 ═══════════════════════════════════════════════════════════
   n-queens by backtracking. Every event the real search performs — a legal
   placement, a rejected square, a backtrack, a completed solution — becomes
   a frame, and the counters are bumped inside the search itself. The node
   count is compared with the sizes of the unpruned search spaces (nⁿ and
   n!), both computed independently.                                        */
(function () {
  const svg = d3.select("#nq-svg"); if (svg.empty()) return;
  const W = 680, H = 330;
  let n = 4, st = null;

  function search(n) {
    const c = AL.counter(), frames = [], col = new Array(n).fill(-1);
    const sols = [];
    const safe = (cc, r) => {
      for (let k = 0; k < cc; k++)
        if (col[k] === r || Math.abs(col[k] - r) === cc - k) return false;
      return true;
    };
    (function place(cc) {
      c.add("nodes");
      if (cc === n) {
        c.add("sols"); sols.push([...col]);
        frames.push({ col: [...col], c: cc, r: -1, kind: "solution",
                      note: "all " + n + " columns filled — solution #" + c.get("sols")
                          + ": rows [" + col.join(", ") + "]" });
        return;
      }
      for (let r = 0; r < n; r++) {
        if (safe(cc, r)) {
          col[cc] = r; c.add("placed");
          frames.push({ col: [...col], c: cc, r: r, kind: "place",
                        note: "column " + cc + ", row " + r + " is safe — place and descend" });
          place(cc + 1);
          col[cc] = -1;
          frames.push({ col: [...col], c: cc, r: r, kind: "undo",
                        note: "backtrack out of column " + cc + " row " + r
                            + " — its subtree is finished, undo and try the next row" });
        } else {
          c.add("pruned");
          frames.push({ col: [...col], c: cc, r: r, kind: "prune",
                        note: "column " + cc + ", row " + r
                            + " is attacked — prune: the whole subtree below it is skipped" });
        }
      }
    })(0);
    return { frames: frames, counts: c.all(), sols: sols };
  }

  function build() {
    const r = search(n);
    if (st) st.pause();
    DC.clearControls(svg.node());
    st = AL.stepper(svg, {
      frames: r.frames, delay: 340, label: "event",
      render: (f, i) => draw(f, i, r.frames.length, r)
    });

    const nn = Math.pow(n, n);
    let fact = 1; for (let k = 2; k <= n; k++) fact *= k;
    DC.table("#nq-table",
      ["Quantity", "Value", "Note"],
      [["solutions found (measured)", "<b>" + DC.int(r.counts.sols || 0) + "</b>",
        n === 8 ? "the eight-queens problem has 92 solutions — <b>agrees</b>" :
        n === 4 ? "the 4×4 board has 2 solutions — <b>agrees</b>" : "all placements, symmetries included"],
       ["search-tree nodes explored (measured)", "<b>" + DC.int(r.counts.nodes || 0) + "</b>",
        "one per recursive call, including the leaves that record a solution"],
       ["squares rejected by the safety test (measured)", "<b>" + DC.int(r.counts.pruned || 0) + "</b>",
        "each one discards an entire subtree without exploring it"],
       ["legal placements made (measured)", "<b>" + DC.int(r.counts.placed || 0) + "</b>",
        "every one of these is later undone by a backtrack"],
       ["one queen per column, any row", DC.int(nn),
        "<code>nⁿ</code> — what generate-and-test would enumerate"],
       ["one queen per column, distinct rows", DC.int(fact),
        "<code>n!</code> — what the row constraint alone leaves"],
       ["nodes explored as a fraction of <code>nⁿ</code>", 
        "1 in " + DC.int(Math.round(nn / (r.counts.nodes || 1))),
        "this ratio grows with n — pruning buys a growing factor, not a smaller complexity class"]]);
  }

  function draw(f, idx, nf, res) {
    const fr = AL.frame(svg, W, H, { l: 16, r: 16, t: 14, b: 12 });
    const g = fr.g;
    const cell = Math.min(34, (fr.ih - 84) / n);
    const bx = 8, by = 26;

    g.append("text").attr("x", bx).attr("y", 14).attr("font-size", 12).attr("fill", AC.ink)
      .text("columns left to right, rows top to bottom — a queen sits in every fixed column");

    for (let cc = 0; cc < n; cc++) for (let rr = 0; rr < n; rr++) {
      const x = bx + cc * cell, y = by + rr * cell;
      const light = (cc + rr) % 2 === 0;
      g.append("rect").attr("x", x).attr("y", y).attr("width", cell).attr("height", cell)
        .attr("fill", light ? AC.panel2 : AC.panel).attr("stroke", AC.line).attr("stroke-width", 0.5);
    }
    /* queens placed so far */
    for (let cc = 0; cc < n; cc++) {
      if (f.col[cc] < 0) continue;
      const x = bx + cc * cell + cell / 2, y = by + f.col[cc] * cell + cell / 2;
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", cell * 0.32)
        .attr("fill", AC.accent).attr("stroke", AC.ink).attr("stroke-width", 1);
    }
    /* the current event */
    if (f.c < n && f.r >= 0) {
      const x = bx + f.c * cell, y = by + f.r * cell;
      const col = f.kind === "prune" ? AC.bad : (f.kind === "undo" ? AC.muted : AC.good);
      g.append("rect").attr("x", x + 1).attr("y", y + 1).attr("width", cell - 2).attr("height", cell - 2)
        .attr("fill", "none").attr("stroke", col).attr("stroke-width", 2.5);
      if (f.kind === "prune") {
        g.append("path").attr("d", "M" + (x + 6) + "," + (y + 6) + " L" + (x + cell - 6) + "," + (y + cell - 6)
                                 + " M" + (x + cell - 6) + "," + (y + 6) + " L" + (x + 6) + "," + (y + cell - 6))
          .attr("stroke", AC.bad).attr("stroke-width", 2);
      }
    }
    if (f.kind === "solution") {
      g.append("rect").attr("x", bx - 2).attr("y", by - 2).attr("width", n * cell + 4)
        .attr("height", n * cell + 4).attr("fill", "none").attr("stroke", AC.good).attr("stroke-width", 2.5);
    }

    const tx = bx + n * cell + 26;
    const lines = [
      "event " + (idx + 1) + " of " + DC.int(nf),
      "",
      "kind:  " + f.kind,
      "queens fixed:  " + f.col.filter(v => v >= 0).length + " of " + n,
      "rows so far:  [" + f.col.map(v => v < 0 ? "·" : v).join(" ") + "]",
      "",
      "whole search, measured:",
      "  nodes explored      " + DC.int(res.counts.nodes || 0),
      "  squares pruned      " + DC.int(res.counts.pruned || 0),
      "  placements made     " + DC.int(res.counts.placed || 0),
      "  solutions found     " + DC.int(res.counts.sols || 0)
    ];
    lines.forEach((t, i) => g.append("text").attr("x", tx).attr("y", by + 12 + i * 17)
      .attr("font-size", 11.5).attr("fill", i >= 7 ? AC.muted : AC.ink)
      .attr("font-family", '"SF Mono",Menlo,monospace').text(t));

    g.append("text").attr("x", bx).attr("y", fr.ih - 6).attr("font-size", 12).attr("fill", AC.ink)
      .text(f.note);

    d3.select("#nq-readout").html(
      "n = <b>" + n + "</b> &nbsp;·&nbsp; event <b>" + (idx + 1) + "</b> / " + DC.int(nf)
      + " &nbsp;·&nbsp; " + f.note
      + " &nbsp;·&nbsp; whole search: <b>" + DC.int(res.counts.nodes || 0) + "</b> nodes, <b>"
      + DC.int(res.counts.pruned || 0) + "</b> squares pruned, <b>"
      + DC.int(res.counts.sols || 0) + "</b> solutions");
  }

  d3.select("#nq-n").on("input", function () {
    n = +this.value; d3.select("#nq-n-out").text(n); build();
  });
  d3.select("#nq-n-out").text(n);
  build();
})();

/* ══ FIGURE 13 ═══════════════════════════════════════════════════════════
   The subset and permutation enumeration trees. The tree drawn is the one
   the real generator produced; node and leaf counts come from AL.counter()
   and are compared against 2^(n+1) − 1 / 2ⁿ and ∑ n!/j! / n!. Above eighty
   nodes the drawing degrades to a per-level node count, because a tree that
   does not fit is not a figure.                                            */
(function () {
  const svg = d3.select("#en-svg"); if (svg.empty()) return;
  const W = 680, H = 320;
  let mode = "subsets", n = 4;

  const fact = k => { let f = 1; for (let i = 2; i <= k; i++) f *= i; return f; };

  function subsetTree(n, c) {
    return (function go(i, chosen) {
      c.add("nodes");
      const node = { label: chosen.length ? "{" + chosen.join(",") + "}" : "{ }", children: [] };
      if (i === n) { c.add("leaves"); return node; }
      node.children.push(go(i + 1, chosen));
      node.children.push(go(i + 1, [...chosen, i + 1]));
      return node;
    })(0, []);
  }
  function permTree(n, c) {
    const all = Array.from({ length: n }, (_, i) => i + 1);
    return (function go(prefix, remaining) {
      c.add("nodes");
      const node = { label: prefix.length ? prefix.join("") : "·", children: [] };
      if (remaining.length === 0) { c.add("leaves"); return node; }
      remaining.forEach(e => node.children.push(go([...prefix, e], remaining.filter(x => x !== e))));
      return node;
    })([], all);
  }

  function build() {
    const c = AL.counter();
    const root = mode === "subsets" ? subsetTree(n, c) : permTree(n, c);
    const nodes = c.get("nodes"), leaves = c.get("leaves");
    const predLeaves = mode === "subsets" ? Math.pow(2, n) : fact(n);
    let predNodes;
    if (mode === "subsets") predNodes = Math.pow(2, n + 1) - 1;
    else { predNodes = 0; for (let j = 0; j <= n; j++) predNodes += fact(n) / fact(j); }

    /* per-level node counts, measured by walking the tree that was built */
    const perLevel = [];
    (function walk(t, d) { perLevel[d] = (perLevel[d] || 0) + 1; t.children.forEach(k => walk(k, d + 1)); })(root, 0);

    draw(root, perLevel, nodes);

    const formNodes = mode === "subsets" ? "2ⁿ⁺¹ − 1" : "∑ (j = 0…n) n!/j!";
    const formLeaves = mode === "subsets" ? "2ⁿ" : "n!";
    const rows = perLevel.map((v, d) => [
      "level " + d, "<b>" + DC.int(v) + "</b>",
      mode === "subsets" ? "2^" + d + " = " + DC.int(Math.pow(2, d))
                         : "n!/(n−k)! = " + DC.int(fact(n) / fact(n - d)),
      DC.flag(v === (mode === "subsets" ? Math.pow(2, d) : fact(n) / fact(n - d)))
    ]);
    rows.push(["<b>total nodes</b>", "<b>" + DC.int(nodes) + "</b>",
               "<code>" + formNodes + "</code> = " + DC.int(predNodes), DC.flag(nodes === predNodes)]);
    rows.push(["<b>leaves = results</b>", "<b>" + DC.int(leaves) + "</b>",
               "<code>" + formLeaves + "</code> = " + DC.int(predLeaves), DC.flag(leaves === predLeaves)]);
    rows.push(["<b>nodes per result</b>", "<b>" + (nodes / leaves).toFixed(3) + "</b>",
               mode === "subsets" ? "tends to 2 from below: 2 − 2^(−n)" : "tends to e = " + Math.E.toFixed(3) + " from below",
               "the constant overhead of enumerating, which does not grow with n"]);
    DC.table("#en-table", ["", "Measured", "Closed form", "check"], rows);

    d3.select("#en-readout").html(
      (mode === "subsets" ? "subsets" : "permutations") + " of a set of <b>" + n
      + "</b> elements &nbsp;·&nbsp; results (leaves) <b>" + DC.int(leaves) + "</b> = "
      + formLeaves + " " + DC.flag(leaves === predLeaves)
      + " &nbsp;·&nbsp; invocations (nodes) <b>" + DC.int(nodes) + "</b> = " + formNodes + " "
      + DC.flag(nodes === predNodes)
      + " &nbsp;·&nbsp; nodes per result <b>" + (nodes / leaves).toFixed(3) + "</b>"
      + " &nbsp;·&nbsp; depth <b>" + perLevel.length + "</b>");
  }

  function draw(root, perLevel, nodes) {
    const fr = AL.frame(svg, W, H, { l: 20, r: 20, t: 18, b: 24 });
    const g = fr.g;
    if (nodes <= 80) {
      const h = d3.hierarchy(root, d => d.children);
      d3.tree().size([fr.iw, fr.ih - 24])(h);
      g.selectAll("path.l").data(h.links()).join("path").attr("class", "l")
        .attr("d", d => "M" + d.source.x + "," + d.source.y + " L" + d.target.x + "," + d.target.y)
        .attr("fill", "none").attr("stroke", AC.line).attr("stroke-width", 1.1);
      const nd = g.selectAll("g.n").data(h.descendants()).join("g").attr("class", "n")
        .attr("transform", d => "translate(" + d.x + "," + d.y + ")");
      const isLeaf = d => !d.children || d.children.length === 0;
      nd.append("circle").attr("r", nodes > 40 ? 5 : 8)
        .attr("fill", d => isLeaf(d) ? AC.good : AC.panel2).attr("stroke", AC.line);
      nd.append("text").attr("y", d => isLeaf(d) ? 20 : -11).attr("text-anchor", "middle")
        .attr("font-size", nodes > 40 ? 9 : 10.5)
        .attr("fill", d => isLeaf(d) ? AC.good : AC.muted).text(d => d.data.label);
      AL.legend(g, [{ label: "an invocation", color: AC.panel2 },
                    { label: "a leaf — one result emitted", color: AC.good }], 2, 10);
    } else {
      const x = d3.scaleBand().domain(perLevel.map((_, i) => i)).range([0, fr.iw]).padding(0.25);
      const y = d3.scaleLinear().domain([0, d3.max(perLevel) * 1.1]).range([fr.ih - 10, 0]).nice();
      AL.gridY(g, y, fr.iw, 5);
      g.append("g").attr("class", "axis").attr("transform", "translate(0," + (fr.ih - 10) + ")")
        .call(d3.axisBottom(x));
      AL.axisL(g, y, 5, "invocations at this level of the tree", d3.format("~s"));
      g.selectAll("rect.b").data(perLevel).join("rect").attr("class", "b")
        .attr("x", (_, i) => x(i)).attr("y", d => y(d)).attr("width", x.bandwidth())
        .attr("height", d => fr.ih - 10 - y(d)).attr("rx", 3)
        .attr("fill", (_, i) => i === perLevel.length - 1 ? AC.good : AC.accent);
      g.selectAll("text.v").data(perLevel).join("text").attr("class", "v")
        .attr("x", (_, i) => x(i) + x.bandwidth() / 2).attr("y", d => y(d) - 5)
        .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.muted)
        .text(d => DC.int(d));
      g.append("text").attr("x", 0).attr("y", 10).attr("font-size", 11).attr("fill", AC.muted)
        .text(DC.int(nodes) + " nodes — too many to draw, so here is the node count per level"
              + " (the last level, in green, is the results)");
    }
  }

  d3.select("#en-mode").on("change", function () { mode = this.value; build(); });
  d3.select("#en-n").on("input", function () {
    n = +this.value; d3.select("#en-n-out").text(n); build();
  });
  d3.select("#en-n-out").text(n);
  build();
})();

/* ══ FIGURE 14 ═══════════════════════════════════════════════════════════
   Randomized quicksort and quickselect. Every point is the mean measured
   comparison count over `trials` seeded runs of the real routine, and every
   run's result is verified (fully sorted, or the correct order statistic).
   The overlaid curve is the exact expectation 2(n+1)Hₙ − 4n, evaluated
   independently of the code being measured.                                */
(function () {
  const svg = d3.select("#rq-svg"); if (svg.empty()) return;
  const W = 680, H = 300;
  const SIZES = [10, 25, 50, 100, 250, 500, 1000];
  let mode = "sort", trials = 40;

  function harmonic(n) { let s = 0; for (let i = 1; i <= n; i++) s += 1 / i; return s; }

  /* Lomuto partition with a uniformly random pivot, comparisons counted */
  function partition(a, lo, hi, r, c) {
    const p = AL.randInt(r, lo, hi - 1);
    let t = a[p]; a[p] = a[hi - 1]; a[hi - 1] = t;
    const pv = a[hi - 1];
    let i = lo;
    for (let j = lo; j < hi - 1; j++) {
      c.add("cmp");
      if (a[j] <= pv) { t = a[i]; a[i] = a[j]; a[j] = t; i++; }
    }
    t = a[i]; a[i] = a[hi - 1]; a[hi - 1] = t;
    return i;
  }
  function quicksort(input, r, c) {
    const a = [...input];
    (function rec(lo, hi) {
      if (hi - lo <= 1) return;
      const p = partition(a, lo, hi, r, c);
      rec(lo, p); rec(p + 1, hi);
    })(0, a.length);
    return a;
  }
  function quickselect(input, k, r, c) {
    const a = [...input];
    let lo = 0, hi = a.length;
    while (hi - lo > 1) {
      const p = partition(a, lo, hi, r, c);
      if (k === p) return a[p];
      if (k < p) hi = p; else lo = p + 1;
    }
    return a[lo];
  }

  function build() {
    const data = [];
    let bad = 0;
    SIZES.forEach(n => {
      let tot = 0;
      for (let t = 0; t < trials; t++) {
        const r = AL.rng(n * 7919 + t * 104729 + 13);
        const A = AL.perm(n, AL.rng(n * 31 + t * 17 + 5));       // a random permutation of 1..n
        const c = AL.counter();
        if (mode === "sort") {
          const out = quicksort(A, r, c);
          if (!out.every((v, i) => v === i + 1)) bad++;
        } else {
          const k = Math.floor((n - 1) / 2);
          const v = quickselect(A, k, r, c);
          if (v !== k + 1) bad++;
        }
        tot += c.get("cmp");
      }
      const mean = tot / trials;
      data.push({ n: n, mean: mean, pred: 2 * (n + 1) * harmonic(n) - 4 * n, perN: mean / n });
    });
    draw(data, bad);

    DC.table("#rq-table",
      mode === "sort"
        ? ["n", "Mean comparisons (measured)", "2(n+1)Hₙ − 4n", "measured / expected",
           "mean / (n log₂ n)"]
        : ["n", "Mean comparisons (measured)", "mean ÷ n", "the good-pivot bound 4n",
           "inside the bound"],
      data.map(d => mode === "sort"
        ? ["<code>" + d.n + "</code>", "<b>" + d.mean.toFixed(1) + "</b>", d.pred.toFixed(1),
           "<b>" + (d.mean / d.pred).toFixed(4) + "</b>",
           (d.mean / (d.n * AL.log2(d.n))).toFixed(3)]
        : ["<code>" + d.n + "</code>", "<b>" + d.mean.toFixed(1) + "</b>",
           "<b>" + d.perN.toFixed(3) + "</b>", DC.int(4 * d.n),
           DC.flag(d.mean < 4 * d.n)]));

    const last = data[data.length - 1];
    d3.select("#rq-readout").html(
      (mode === "sort" ? "randomized quicksort" : "randomized quickselect, median")
      + " &nbsp;·&nbsp; " + trials + " runs per size &nbsp;·&nbsp; at n = <b>" + last.n
      + "</b> measured mean <b>" + last.mean.toFixed(1) + "</b> comparisons"
      + (mode === "sort"
          ? " against the exact expectation <b>" + last.pred.toFixed(1) + "</b> — ratio <b>"
            + (last.mean / last.pred).toFixed(4) + "</b>; as a multiple of n·log₂n that is <b>"
            + (last.mean / (last.n * AL.log2(last.n))).toFixed(3)
            + "</b>, still climbing towards the asymptotic 2·ln2 = "
            + (2 * Math.LN2).toFixed(3) + " because the −4n term fades only slowly"
          : " = <b>" + last.perN.toFixed(3) + "·n</b>, inside the 4n bound "
            + DC.flag(last.mean < 4 * last.n))
      + " &nbsp;·&nbsp; every run's output verified " + DC.flag(bad === 0));
  }

  function draw(data, bad) {
    const fr = AL.frame(svg, W, H, { l: 70, r: 22, t: 18, b: 54 });
    const g = fr.g;
    const x = d3.scaleLog().domain([SIZES[0], SIZES[SIZES.length - 1]]).range([0, fr.iw]);
    const hi = d3.max(data, d => Math.max(d.mean, mode === "sort" ? d.pred : 4 * d.n));
    const lo = d3.min(data, d => d.mean);
    const y = d3.scaleLog().domain([lo / 1.6, hi * 1.4]).range([fr.ih, 0]);
    AL.gridY(g, y, fr.iw, 6);
    AL.axisB(g, x, fr.ih, 5, "n", d3.format("d"));
    AL.axisL(g, y, 6, "comparisons", d3.format("~s"));

    if (mode === "sort") {
      g.append("path").datum(data).attr("fill", "none").attr("stroke", AC.a2)
        .attr("stroke-width", 3).attr("opacity", 0.55)
        .attr("d", d3.line().x(d => x(d.n)).y(d => y(d.pred)));
    } else {
      g.append("path").datum(data).attr("fill", "none").attr("stroke", AC.a2)
        .attr("stroke-width", 2).attr("stroke-dasharray", "5 3")
        .attr("d", d3.line().x(d => x(d.n)).y(d => y(4 * d.n)));
    }
    g.append("path").datum(data).attr("fill", "none").attr("stroke", AC.accent)
      .attr("stroke-width", 2.2)
      .attr("d", d3.line().x(d => x(d.n)).y(d => y(d.mean)));
    g.selectAll("circle.m").data(data).join("circle").attr("class", "m")
      .attr("cx", d => x(d.n)).attr("cy", d => y(d.mean)).attr("r", 3.4).attr("fill", AC.accent);

    AL.legend(g, mode === "sort"
      ? [{ label: "measured mean comparisons over " + trials + " randomized runs", color: AC.accent },
         { label: "the exact expectation 2(n+1)Hₙ − 4n", color: AC.a2 }]
      : [{ label: "measured mean comparisons to find the median", color: AC.accent },
         { label: "the 4n bound from the good-pivot argument", color: AC.a2, dash: "5 3" }],
      6, fr.ih + 26, { gap: 15 });
    g.append("text").attr("x", fr.iw).attr("y", 12).attr("text-anchor", "end").attr("font-size", 11)
      .attr("fill", bad ? AC.bad : AC.muted)
      .text(bad ? bad + " runs returned a WRONG result" : "every run verified correct");
  }

  d3.select("#rq-mode").on("change", function () { mode = this.value; build(); });
  d3.select("#rq-trials").on("input", function () {
    trials = +this.value; d3.select("#rq-trials-out").text(trials); build();
  });
  d3.select("#rq-trials-out").text(trials);
  build();
})();
