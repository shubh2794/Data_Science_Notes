/* analysis.viz.js — figures for dsa/algorithms/analysis.html (part 1 of the
   Algorithm Design & Analysis series).

   Loaded after ../../data.js, ../../notes.js and ../dsa-viz.js, so the globals
   AC (palette) and AL (helpers: rng/frame/axisB/axisL/gridY/row/bars/counter/
   stepper/…) are available.

   House rule for this domain, obeyed throughout: every cost this page DISPLAYS
   — comparisons, shifts, bit flips, element copies, operation counts — is
   produced by running the real routine under AL.counter() and reading the
   counter back. No count below is a constant typed into a caption. Where a
   closed form is also shown, it is drawn from the formula independently, so
   the figure is its own cross-check: if the two lines separate, one of them is
   wrong.

   Figures, in page order:
     01 #is-svg        instrumented insertion sort, stepped, + measured line table
     02 #avg-svg       measured mean comparisons vs the exact closed form
     03 #theta-svg     the Theta sandwich: c1, c2 and the smallest n0 that works
     04 #growth-svg    growth-class curves, linear/log y-axis
     05 #cross-svg     crossover: a·n·log2 n vs b·n²
     06 #loops-svg     prefix-average 1/2/3, measured operation counts
     07 #rtree-svg     recursion tree for T(n) = a·T(n/b) + c·n^k, per-level work
     08 #master-svg    Master-theorem case classifier
     09 #ctr-svg       binary counter: bits flipped per INCREMENT, measured
     10 #dyn-svg       dynamic array: per-append cost and running amortized mean
     11 #pot-svg       the same table's potential, actual and amortized cost
     12 #dbl-svg       the doubling test, with the exponent fitted from ratios   */

/* ── shared little helpers ─────────────────────────────────────────────── */
const AN = {
  /* a <table class="cmp"> built from a header row + body rows, injected into a
     host element. Used wherever a figure reports measured numbers as a table. */
  table: function (sel, head, rows) {
    const h = d3.select(sel);
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
  sig: function (x, d) { return (+x).toFixed(d === undefined ? 2 : d); }
};

/* ══ FIGURE 01 ═══════════════════════════════════════════════════════════
   Insertion sort, instrumented. The frames come from running the real
   procedure and snapshotting it; the counters come from AL.counter(), so the
   readout and the per-line table are measurements of this exact run.        */
(function () {
  const svg = d3.select("#is-svg"); if (svg.empty()) return;
  const W = 680, H = 214;

  const PRESETS = {
    worked:  [5, 2, 4, 6, 1, 3],
    sorted:  [1, 2, 3, 4, 5, 6],
    reverse: [6, 5, 4, 3, 2, 1],
    random:  AL.perm(6, AL.rng(11))
  };

  /* The real routine, written exactly as the pseudocode is, with the counter
     threaded through. `snap` is called at every decision point so the stepper
     can replay it. The pseudocode line numbers of §04 are kept so the table
     below lines up:
       1 for j = 2 to n      2 key = A[j]      4 i = j-1
       5 while i > 0 and A[i] > key            6 A[i+1] = A[i]     7 i = i-1
       8 A[i+1] = key                                                        */
  function insertionSort(input, c, snap) {
    const A = [...input], n = A.length;
    for (let j = 1; j < n; j++) {                 // line 1 (0-based j)
      c.add("line1"); c.add("line2"); c.add("line4");
      const key = A[j];
      let i = j - 1;
      if (snap) snap({ a: [...A], j, i, key, phase: "pick",
                       note: "take key = A[" + j + "] = " + key + "; the prefix A[0.." + (j - 1) + "] is already sorted" });
      for (;;) {
        c.add("line5");                           // the while TEST itself
        let go = false;
        if (i >= 0) {
          c.add("cmp");                           // an element comparison happened
          go = A[i] > key;
          if (snap) snap({ a: [...A], j, i, key, phase: go ? "cmpT" : "cmpF",
                           note: "compare A[" + i + "] = " + A[i] + " > " + key + " → " + (go ? "yes, shift right" : "no, stop") });
        } else if (snap) {
          snap({ a: [...A], j, i, key, phase: "left",
                 note: "i < 0 — the test short-circuits, no element comparison" });
        }
        if (!go) break;
        c.add("line6"); c.add("line7"); c.add("shift");
        A[i + 1] = A[i];
        i--;
      }
      c.add("line8");
      A[i + 1] = key;
      if (snap) snap({ a: [...A], j, i: i + 1, key, phase: "place",
                       note: "place key at A[" + (i + 1) + "]; A[0.." + j + "] is now sorted" });
    }
    c.add("line1");                               // the final, failing loop test
    if (snap) snap({ a: [...A], j: n, i: -1, key: null, phase: "done",
                     note: "the loop test fails at j = n, so A[0..n−1] is sorted" });
    return A;
  }

  let key = "worked", st = null;

  function build() {
    const input = PRESETS[key];
    const frames = [], c = AL.counter();
    /* One run produces both: the frame list AND the counts. The counts shown
       are read off this same run — nothing is recomputed by hand. */
    const snapCounts = [];
    insertionSort(input, c, f => { frames.push(f); snapCounts.push(c.all()); });
    frames.forEach((f, i) => { f.counts = snapCounts[i]; });
    const total = c.all();
    const q = k => (total[k] || 0);            // a key never bumped is 0, not undefined
    const n = input.length;

    if (st) st.pause();
    d3.select("#is-svg").selectAll("*").remove();
    d3.select("#is-svg").node().parentNode.querySelectorAll('div[role="group"]')
      .forEach(el => el.remove());

    st = AL.stepper(svg, {
      frames: frames, delay: 620, label: "step",
      render: (f, idx) => draw(f, idx, frames.length)
    });

    /* measured per-line execution counts for the WHOLE run */
    const tj = q("line5"), sh = q("shift");
    AN.table("#is-lines",
      ["Line", "Cost", "Times executed", "Measured on this input"],
      [
        ["1 &nbsp;<code>for j = 2 to n</code>", "<code>c₁</code>", "<code>n</code>", "<b>" + q("line1") + "</b>"],
        ["2 &nbsp;<code>key = A[j]</code>", "<code>c₂</code>", "<code>n − 1</code>", "<b>" + q("line2") + "</b>"],
        ["4 &nbsp;<code>i = j − 1</code>", "<code>c₄</code>", "<code>n − 1</code>", "<b>" + q("line4") + "</b>"],
        ["5 &nbsp;<code>while i &gt; 0 and A[i] &gt; key</code>", "<code>c₅</code>", "<code>∑ⱼ₌₂ⁿ tⱼ</code>", "<b>" + tj + "</b>"],
        ["6 &nbsp;<code>A[i+1] = A[i]</code>", "<code>c₆</code>", "<code>∑ⱼ₌₂ⁿ (tⱼ − 1)</code>", "<b>" + q("line6") + "</b>"],
        ["7 &nbsp;<code>i = i − 1</code>", "<code>c₇</code>", "<code>∑ⱼ₌₂ⁿ (tⱼ − 1)</code>", "<b>" + q("line7") + "</b>"],
        ["8 &nbsp;<code>A[i+1] = key</code>", "<code>c₈</code>", "<code>n − 1</code>", "<b>" + q("line8") + "</b>"]
      ]);

    const inv = (function () {                    // an independent second count
      let k = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) if (input[p] > input[q]) k++;
      return k;
    })();
    d3.select("#is-check").html(
      "whole run · <b>" + q("cmp") + "</b> element comparisons · <b>" + sh + "</b> shifts · "
      + "<b>" + tj + "</b> while-tests (= ∑tⱼ) · shifts counted a second way, as inversions of the input: <b>"
      + inv + "</b> " + (inv === sh ? "✓ agree" : "✗ DISAGREE"));
  }

  function draw(f, idx, nf) {
    const fr = AL.frame(svg, W, H, { l: 24, r: 24, t: 16, b: 10 });
    const g = fr.g, n = f.a.length;
    const sortedEnd = f.phase === "done" ? n : f.j;   // A[0..j-1] is the sorted prefix

    const r = AL.row(g, f.a, {
      x: 40, y: 34, w: 46, h: 38, gap: 6, fontSize: 15,
      mark: (i) => {
        if (f.phase !== "done" && i === f.j && (f.phase === "pick")) return AC.a2;
        if (f.phase === "place" && i === f.i) return AC.good;
        if ((f.phase === "cmpT" || f.phase === "cmpF") && i === f.i) return AC.violet;
        if (i < sortedEnd) return "#22304a";
        return AC.panel2;
      }
    });

    g.append("text").attr("x", 40).attr("y", 20).attr("font-size", 11).attr("fill", AC.muted)
      .text("sorted prefix A[0.." + (sortedEnd - 1) + "]" + (f.key === null ? "" : "   ·   key = " + f.key));
    g.append("line").attr("x1", 38).attr("x2", 38 + sortedEnd * 52 - 6).attr("y1", 28).attr("y2", 28)
      .attr("stroke", AC.accent).attr("stroke-width", 2);

    if (f.i >= 0 && f.phase !== "done")
      AL.arrow(g, 40 + r.cellX(f.i), 110, 40 + r.cellX(f.i), 93, { color: AC.violet });
    if (f.phase !== "done" && f.j < n)
      AL.arrow(g, 40 + r.cellX(f.j), 110, 40 + r.cellX(f.j), 93, { color: AC.a2 });
    g.append("text").attr("x", 40).attr("y", 128).attr("font-size", 12).attr("fill", AC.ink).text(f.note);

    const c = f.counts;
    g.append("text").attr("x", 40).attr("y", 152).attr("font-size", 12).attr("fill", AC.muted)
      .text("so far —  element comparisons: " + (c.cmp || 0)
            + "    shifts: " + (c.shift || 0)
            + "    while-tests ∑tⱼ: " + (c.line5 || 0));
    d3.select("#is-readout").html(
      "step <b>" + (idx + 1) + "</b> / " + nf + " &nbsp;·&nbsp; comparisons <b>" + (c.cmp || 0)
      + "</b> &nbsp;·&nbsp; shifts <b>" + (c.shift || 0) + "</b> &nbsp;·&nbsp; while-tests <b>"
      + (c.line5 || 0) + "</b>");
  }

  d3.select("#is-input").on("change", function () { key = this.value; build(); });
  build();
})();

/* ══ FIGURE 02 ═══════════════════════════════════════════════════════════
   The average case, measured. For each n we sort `trials` seeded random
   permutations with the real routine and average the counter's comparison
   tally. The curve drawn on top is the exact closed form
       E[comparisons] = n(n−1)/4 + n − Hₙ,
   derived independently of the code. Two lines that never separate are two
   correct derivations; a gap would mean one of them is wrong.              */
(function () {
  const svg = d3.select("#avg-svg"); if (svg.empty()) return;
  const W = 680, H = 360;

  /* the same routine as figure 01, trimmed of the snapshotting */
  function cmpCount(input) {
    const A = [...input], n = A.length; let cmp = 0;
    for (let j = 1; j < n; j++) {
      const key = A[j]; let i = j - 1;
      while (i >= 0) { cmp++; if (A[i] > key) { A[i + 1] = A[i]; i--; } else break; }
      A[i + 1] = key;
    }
    return cmp;
  }
  function worstCount(n) { return n * (n - 1) / 2; }          // reversed input
  function harm(n) { let s = 0; for (let k = 1; k <= n; k++) s += 1 / k; return s; }
  function exact(n) { return n * (n - 1) / 4 + n - harm(n); }

  const NMAX = 40;
  let trials = 60;

  function render() {
    const rnd = AL.rng(20260916);
    const meas = [];
    for (let n = 2; n <= NMAX; n++) {
      let tot = 0;
      for (let t = 0; t < trials; t++) tot += cmpCount(AL.perm(n, rnd));
      meas.push({ n: n, v: tot / trials });
    }
    const fr = AL.frame(svg, W, H, { l: 56, r: 150, t: 22, b: 40 });
    const g = fr.g;
    const x = d3.scaleLinear().domain([2, NMAX]).range([0, fr.iw]);
    const y = d3.scaleLinear().domain([0, worstCount(NMAX) * 1.04]).range([fr.ih, 0]);
    AL.gridY(g, y, fr.iw, 6);
    AL.axisB(g, x, fr.ih, 8, "n");
    AL.axisL(g, y, 6, "element comparisons");

    const ln = d3.line().x(d => x(d.n)).y(d => y(d.v));
    const ns = d3.range(2, NMAX + 1);
    const series = [
      { k: "worst", name: "worst case  n(n−1)/2", color: AC.bad, dash: null, pts: ns.map(n => ({ n, v: worstCount(n) })) },
      { k: "loose", name: "loose guide  n²/4", color: AC.muted, dash: "3,3", pts: ns.map(n => ({ n, v: n * n / 4 })) },
      { k: "exact", name: "exact mean  n(n−1)/4 + n − Hₙ", color: AC.a2, dash: "6,4", pts: ns.map(n => ({ n, v: exact(n) })) },
      { k: "best", name: "best case  n − 1", color: AC.good, dash: null, pts: ns.map(n => ({ n, v: n - 1 })) }
    ];
    series.forEach(s => {
      const p = g.append("path").attr("fill", "none").attr("stroke", s.color)
        .attr("stroke-width", 2).attr("d", ln(s.pts));
      if (s.dash) p.attr("stroke-dasharray", s.dash);
    });
    g.selectAll("circle.m").data(meas).join("circle").attr("class", "m")
      .attr("cx", d => x(d.n)).attr("cy", d => y(d.v)).attr("r", 2.6).attr("fill", AC.accent);

    AL.legend(g, [
      { label: "measured mean over " + trials + " random inputs", color: AC.accent },
      { label: "exact mean  n(n−1)/4 + n − Hₙ", color: AC.a2, dash: "6,4" },
      { label: "worst case  n(n−1)/2", color: AC.bad },
      { label: "loose guide  n²/4", color: AC.muted, dash: "3,3" },
      { label: "best case  n − 1", color: AC.good }
    ], fr.iw + 10, 16);

    const last = meas[meas.length - 1], ex = exact(NMAX);
    d3.select("#avg-readout").html(
      "at n = " + NMAX + " &nbsp;·&nbsp; measured mean <b>" + AN.sig(last.v, 2)
      + "</b> comparisons &nbsp;·&nbsp; exact closed form <b>" + AN.sig(ex, 2)
      + "</b> &nbsp;·&nbsp; gap <b>" + AN.sig(100 * Math.abs(last.v - ex) / ex, 2)
      + "%</b> &nbsp;·&nbsp; worst case <b>" + AN.int(worstCount(NMAX))
      + "</b> &nbsp;·&nbsp; ratio worst / average <b>" + AN.sig(worstCount(NMAX) / ex, 3) + "</b>");
  }

  d3.select("#avg-trials").on("input", function () {
    trials = +this.value;
    d3.select("#avg-trials-out").text(trials);
    render();
  });
  d3.select("#avg-trials-out").text(trials);
  render();
})();

/* ══ FIGURE 03 ═══════════════════════════════════════════════════════════
   The Theta definition made operational. Pick c1 and c2; the figure SEARCHES
   for the smallest n0 such that 0 ≤ c1·g(n) ≤ f(n) ≤ c2·g(n) holds for every
   n in [n0, NMAX], and reports it. Nothing about n0 is asserted — it is found
   by scanning, which is also how you would find it by hand.                 */
(function () {
  const svg = d3.select("#theta-svg"); if (svg.empty()) return;
  const W = 680, H = 360, NMAX = 400;

  const FS = {
    half:  { name: "f(n) = n²/2 − 3n", f: n => 0.5 * n * n - 3 * n, g: n => n * n, gname: "n²" },
    quad:  { name: "f(n) = 3n² + 10n + 50", f: n => 3 * n * n + 10 * n + 50, g: n => n * n, gname: "n²" },
    cube:  { name: "f(n) = 6n³", f: n => 6 * n * n * n, g: n => n * n, gname: "n²" },
    nlogn: { name: "f(n) = n·log₂n + 4n", f: n => n * AL.log2(Math.max(n, 2)) + 4 * n, g: n => n * AL.log2(Math.max(n, 2)), gname: "n log n" }
  };
  /* each preset ships with a witness pair that actually works, so the figure
     opens on a meaningful n0 rather than on "no threshold" — except `cube`,
     where "no threshold" is the point */
  const WITNESS = { half: [0.07, 0.5], quad: [1, 4], cube: [0.07, 4], nlogn: [1, 2] };
  let key = "half", c1 = 0.07, c2 = 0.5;

  /* smallest n0 for which the sandwich holds on the whole of [n0, NMAX] */
  function findN0(F) {
    let n0 = null;
    for (let n = NMAX; n >= 1; n--) {
      const f = F.f(n), g = F.g(n);
      if (0 <= c1 * g && c1 * g <= f && f <= c2 * g) n0 = n; else break;
    }
    return n0;
  }

  function render() {
    const F = FS[key], n0 = findN0(F);
    const view = n0 === null ? 40 : Math.max(20, Math.min(NMAX, n0 * 4));
    const fr = AL.frame(svg, W, H, { l: 60, r: 150, t: 22, b: 40 });
    const g = fr.g;
    const ns = AL.linspace(1, view, 220);
    const ymax = Math.max(d3.max(ns, n => F.f(n)), d3.max(ns, n => c2 * F.g(n))) * 1.06;
    const x = d3.scaleLinear().domain([1, view]).range([0, fr.iw]);
    const y = d3.scaleLinear().domain([Math.min(0, d3.min(ns, n => F.f(n))), ymax]).range([fr.ih, 0]);
    AL.gridY(g, y, fr.iw, 6);
    AL.axisB(g, x, fr.ih, 7, "n");
    AL.axisL(g, y, 6, "value", d3.format("~s"));

    if (n0 !== null) {
      g.append("rect").attr("x", x(n0)).attr("y", 0).attr("width", fr.iw - x(n0)).attr("height", fr.ih)
        .attr("fill", AC.accent).attr("opacity", 0.07);
      g.append("line").attr("x1", x(n0)).attr("x2", x(n0)).attr("y1", 0).attr("y2", fr.ih)
        .attr("stroke", AC.accent).attr("stroke-dasharray", "4,3");
      g.append("text").attr("x", x(n0) + 5).attr("y", 13).attr("font-size", 11).attr("fill", AC.accent)
        .text("n₀ = " + n0);
    }
    const ln = d3.line().x(d => x(d)).y(d => y(F.f(d)));
    const l1 = d3.line().x(d => x(d)).y(d => y(c1 * F.g(d)));
    const l2 = d3.line().x(d => x(d)).y(d => y(c2 * F.g(d)));
    g.append("path").attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 2)
      .attr("stroke-dasharray", "5,4").attr("d", l2(ns));
    g.append("path").attr("fill", "none").attr("stroke", AC.good).attr("stroke-width", 2)
      .attr("stroke-dasharray", "5,4").attr("d", l1(ns));
    g.append("path").attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 2.6).attr("d", ln(ns));

    AL.legend(g, [
      { label: "f(n)", color: AC.accent },
      { label: "c₂·" + F.gname + "  (upper)", color: AC.a2, dash: "5,4" },
      { label: "c₁·" + F.gname + "  (lower)", color: AC.good, dash: "5,4" }
    ], fr.iw + 10, 16);

    d3.select("#theta-readout").html(
      n0 === null
        ? "with c₁ = " + AN.sig(c1, 3) + ", c₂ = " + AN.sig(c2, 2)
          + " there is <b>no n₀</b> — the sandwich fails somewhere in [1, " + NMAX
          + "] no matter how far right you look, so this f is <b>not</b> \u0398(" + F.gname + ") with these constants."
        : "with c₁ = " + AN.sig(c1, 3) + ", c₂ = " + AN.sig(c2, 2)
          + " the smallest working threshold, found by scanning, is <b>n₀ = " + n0 + "</b>"
          + " &nbsp;·&nbsp; check at n = n₀: &nbsp; c₁·g = <b>" + AN.sig(c1 * FS[key].g(n0), 2)
          + "</b> ≤ f = <b>" + AN.sig(FS[key].f(n0), 2)
          + "</b> ≤ c₂·g = <b>" + AN.sig(c2 * FS[key].g(n0), 2) + "</b>");
  }

  d3.select("#theta-f").on("change", function () {
    key = this.value;
    const w = WITNESS[key];
    if (w) {
      c1 = w[0]; c2 = w[1];
      d3.select("#theta-c1").property("value", c1); d3.select("#theta-c1-out").text(AN.sig(c1, 3));
      d3.select("#theta-c2").property("value", c2); d3.select("#theta-c2-out").text(AN.sig(c2, 2));
    }
    render();
  });
  d3.select("#theta-c1").on("input", function () { c1 = +this.value; d3.select("#theta-c1-out").text(AN.sig(c1, 3)); render(); });
  d3.select("#theta-c2").on("input", function () { c2 = +this.value; d3.select("#theta-c2-out").text(AN.sig(c2, 2)); render(); });
  d3.select("#theta-c1-out").text(AN.sig(c1, 3));
  d3.select("#theta-c2-out").text(AN.sig(c2, 2));
  render();
})();

/* ══ FIGURE 04 ═══════════════════════════════════════════════════════════
   The growth hierarchy. Curves plus a wall-clock table; every entry of the
   table is evaluated from the same function objects the curves are drawn
   from, so the table cannot drift away from the picture.                    */
(function () {
  const svg = d3.select("#growth-svg"); if (svg.empty()) return;
  const W = 680, H = 370;
  const FN = [
    { name: "1", color: AC.muted, f: () => 1 },
    { name: "log₂ n", color: AC.teal, f: n => AL.log2(Math.max(n, 1)) },
    { name: "√n", color: AC.good, f: n => Math.sqrt(n) },
    { name: "n", color: AC.accent, f: n => n },
    { name: "n log₂ n", color: AC.violet, f: n => n * AL.log2(Math.max(n, 1)) },
    { name: "n²", color: AC.a2, f: n => n * n },
    { name: "n³", color: AC.rose, f: n => n * n * n },
    { name: "2ⁿ", color: AC.bad, f: n => Math.pow(2, n) },
    { name: "n!", color: "#f472b6", f: n => { let p = 1; for (let k = 2; k <= n; k++) p *= k; return p; } }
  ];
  let N = 32, logY = true;

  function render() {
    const fr = AL.frame(svg, W, H, { l: 62, r: 110, t: 22, b: 40 });
    const g = fr.g;
    const x = d3.scaleLinear().domain([1, N]).range([0, fr.iw]);
    const ymax = d3.max(FN, s => s.f(N));
    const y = logY
      ? d3.scaleLog().domain([1, Math.max(10, ymax)]).range([fr.ih, 0]).clamp(true)
      : d3.scaleLinear().domain([0, ymax]).range([fr.ih, 0]);
    AL.gridY(g, y, fr.iw, 6);
    AL.axisB(g, x, fr.ih, 8, "n");
    AL.axisL(g, y, 6, logY ? "operations (log scale)" : "operations", d3.format("~s"));
    const xs = AL.linspace(1, N, 200);
    FN.forEach(s => {
      const ln = d3.line().x(d => x(d)).y(d => y(Math.max(logY ? 1 : 0, s.f(d))));
      g.append("path").attr("fill", "none").attr("stroke", s.color).attr("stroke-width", 2).attr("d", ln(xs));
      g.append("text").attr("x", fr.iw + 6).attr("y", AL.clamp(y(Math.max(logY ? 1 : 0, s.f(N))), 6, fr.ih))
        .attr("font-size", 11).attr("fill", s.color).attr("dominant-baseline", "middle").text(s.name);
    });
    d3.select("#growth-readout").html(
      "at n = <b>" + N + "</b> &nbsp;·&nbsp; "
      + FN.map(s => {
          const v = s.f(N);
          return s.name + " = " + (v >= 1e15 ? d3.format(".3~e")(v)
                                 : v >= 1e6 ? d3.format(".4~s")(v)
                                 : AN.int(Math.round(v)));
        }).join(" &nbsp;·&nbsp; "));
  }

  /* the wall-clock table, evaluated from the same function list */
  function table() {
    const RATE = 1e9;                                     // one billion steps per second
    const secs = s => {
      if (!isFinite(s)) return "∞";
      if (s < 1e-6) return d3.format(".2~s")(s * 1e9) + " ns";
      if (s < 1e-3) return d3.format(".3~r")(s * 1e6) + " µs";
      if (s < 1) return d3.format(".3~r")(s * 1e3) + " ms";
      if (s < 90) return d3.format(".3~r")(s) + " s";
      if (s < 5400) return d3.format(".3~r")(s / 60) + " min";
      if (s < 1.3e5) return d3.format(".3~r")(s / 3600) + " hours";
      if (s < 3.2e7) return d3.format(".3~r")(s / 86400) + " days";
      if (s < 3.2e12) return d3.format(".3~r")(s / 3.156e7) + " years";
      return d3.format(".2~e")(s / 3.156e7) + " years";
    };
    const sizes = [10, 100, 1000, 1e6];
    const pick = ["log₂ n", "n", "n log₂ n", "n²", "n³", "2ⁿ", "n!"];
    const rows = pick.map(nm => {
      const s = FN.find(q => q.name === nm);
      return ["<code>" + nm + "</code>"].concat(sizes.map(n => {
        if (n > 1000 && (nm === "2ⁿ" || nm === "n!")) return "—";
        if (n > 100 && nm === "n!") return "—";
        const ops = s.f(n);
        return secs(ops / RATE);
      }));
    });
    AN.table("#growth-table",
      ["cost function", "n = 10", "n = 100", "n = 1 000", "n = 1 000 000"], rows);
  }

  d3.select("#growth-n").on("input", function () { N = +this.value; d3.select("#growth-n-out").text(N); render(); });
  d3.select("#growth-scale").on("click", function () {
    logY = !logY; d3.select(this).text(logY ? "y-axis: log" : "y-axis: linear"); render();
  });
  d3.select("#growth-n-out").text(N);
  d3.select("#growth-scale").text("y-axis: log");
  render(); table();
})();

/* ══ FIGURE 05 ═══════════════════════════════════════════════════════════
   The crossover. An O(n log n) algorithm with a big constant against an O(n²)
   algorithm with a small one. The crossing point n* is FOUND by scanning, not
   solved for — it has no closed form in elementary functions.               */
(function () {
  const svg = d3.select("#cross-svg"); if (svg.empty()) return;
  const W = 680, H = 350, NLIM = 400000;
  let a = 100, b = 1;
  const F1 = n => a * n * AL.log2(Math.max(n, 2));       // the "good" class, big constant
  const F2 = n => b * n * n;                             // the "bad" class, small constant

  function crossover() {
    for (let n = 2; n <= NLIM; n++) if (F1(n) <= F2(n)) return n;
    return null;
  }

  function render() {
    const ns = crossover();
    const view = ns === null ? NLIM : Math.max(20, Math.round(ns * 1.8));
    const fr = AL.frame(svg, W, H, { l: 66, r: 140, t: 22, b: 40 });
    const g = fr.g;
    const xs = AL.linspace(2, view, 240);
    const x = d3.scaleLinear().domain([2, view]).range([0, fr.iw]);
    const y = d3.scaleLinear().domain([0, Math.max(F1(view), F2(view))]).range([fr.ih, 0]);
    AL.gridY(g, y, fr.iw, 6);
    AL.axisB(g, x, fr.ih, 7, "n", d3.format("~s"));
    AL.axisL(g, y, 6, "operations", d3.format("~s"));
    if (ns !== null) {
      g.append("rect").attr("x", 0).attr("y", 0).attr("width", x(ns)).attr("height", fr.ih)
        .attr("fill", AC.a2).attr("opacity", 0.07);
      g.append("line").attr("x1", x(ns)).attr("x2", x(ns)).attr("y1", 0).attr("y2", fr.ih)
        .attr("stroke", AC.ink).attr("stroke-dasharray", "4,3");
      g.append("text").attr("x", x(ns) + 6).attr("y", 14).attr("font-size", 11).attr("fill", AC.ink)
        .text("n* = " + AN.int(ns));
      g.append("text").attr("x", 6).attr("y", 14).attr("font-size", 11).attr("fill", AC.a2)
        .text("here the n² algorithm wins");
    }
    const l1 = d3.line().x(d => x(d)).y(d => y(F1(d)));
    const l2 = d3.line().x(d => x(d)).y(d => y(F2(d)));
    g.append("path").attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 2.4).attr("d", l1(xs));
    g.append("path").attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 2.4).attr("d", l2(xs));
    AL.legend(g, [
      { label: a + "·n·log₂ n", color: AC.accent },
      { label: b + "·n²", color: AC.a2 }
    ], fr.iw + 10, 18);

    const probe = [10, 100, 1000, 10000];
    AN.table("#cross-table",
      ["n"].concat(probe.map(n => AN.int(n))),
      [["<code>" + a + "·n·log₂ n</code>"].concat(probe.map(n => AN.int(Math.round(F1(n))))),
       ["<code>" + b + "·n²</code>"].concat(probe.map(n => AN.int(Math.round(F2(n))))),
       ["<b>cheaper</b>"].concat(probe.map(n => F1(n) < F2(n) ? "<b>n log n</b>" : "<b>n²</b>"))]);

    d3.select("#cross-readout").html(
      ns === null
        ? "the n log n algorithm never catches up below n = " + AN.int(NLIM)
        : "crossover found by scanning: <b>n* = " + AN.int(ns) + "</b> &nbsp;·&nbsp; at n* the costs are <b>"
          + AN.int(Math.round(F1(ns))) + "</b> and <b>" + AN.int(Math.round(F2(ns)))
          + "</b> &nbsp;·&nbsp; below n* the asymptotically <i>worse</i> algorithm is the faster one");
  }

  d3.select("#cross-a").on("input", function () { a = +this.value; d3.select("#cross-a-out").text(a); render(); });
  d3.select("#cross-b").on("input", function () { b = +this.value; d3.select("#cross-b-out").text(b); render(); });
  d3.select("#cross-a-out").text(a);
  d3.select("#cross-b-out").text(b);
  render();
})();

/* ══ FIGURE 06 ═══════════════════════════════════════════════════════════
   Five real routines, each run under AL.counter(), plotted log-log so the
   slope of a line IS its exponent. The point of the selection: two of them
   look like they should be in a different class than they are.             */
(function () {
  const svg = d3.select("#loops-svg"); if (svg.empty()) return;
  const W = 680, H = 380;

  /* ---- the routines, written as they would be written, then instrumented -- */
  function prefixAverage1(S, c) {              // nested loops
    const n = S.length, A = new Array(n).fill(0);
    for (let j = 0; j < n; j++) {
      let total = 0;
      for (let i = 0; i <= j; i++) { c.add("op"); total += S[i]; }
      A[j] = total / (j + 1);
    }
    return A;
  }
  /* the library calls that look like one operation. Each is charged for the
     work it really does: a slice copies its elements, a sum adds them. */
  function slice(S, lo, hi, c) { const o = []; for (let i = lo; i < hi; i++) { c.add("op"); o.push(S[i]); } return o; }
  function sum(S, c) { let t = 0; for (let i = 0; i < S.length; i++) { c.add("op"); t += S[i]; } return t; }
  function prefixAverage2(S, c) {              // ONE visible loop — still quadratic
    const n = S.length, A = new Array(n).fill(0);
    for (let j = 0; j < n; j++) {
      /* the single line  A[j] = sum(S[0 : j+1]) / (j+1)  — the slice and the
         sum are each linear in j, which the counter is what makes visible */
      A[j] = sum(slice(S, 0, j + 1, c), c) / (j + 1);
    }
    return A;
  }
  function prefixAverage3(S, c) {              // one loop, running total
    const n = S.length, A = new Array(n).fill(0);
    let total = 0;
    for (let j = 0; j < n; j++) { c.add("op"); total += S[j]; A[j] = total / (j + 1); }
    return A;
  }
  function disjoint1(A, B, C, c) {             // triple loop, cubic
    for (const a of A) for (const b of B) for (const d of C) { c.add("op"); if (a === b && b === d) return false; }
    return true;
  }
  function disjoint2(A, B, C, c) {             // triple loop, quadratic
    for (const a of A) for (const b of B) {
      c.add("op");
      if (a === b) for (const d of C) { c.add("op"); if (a === d) return false; }
    }
    return true;
  }

  const ROUTINES = [
    { key: "pa1", name: "prefix-average 1 · nested loops", color: AC.a2,
      closed: n => n * (n + 1) / 2, cf: "n(n+1)/2", cls: "Θ(n²)",
      run: (n, c) => prefixAverage1(d3.range(n), c) },
    { key: "pa2", name: "prefix-average 2 · one loop, hidden sum", color: AC.rose,
      closed: n => n * (n + 1), cf: "n(n+1)", cls: "Θ(n²)",
      run: (n, c) => prefixAverage2(d3.range(n), c) },
    { key: "pa3", name: "prefix-average 3 · one loop, running total", color: AC.good,
      closed: n => n, cf: "n", cls: "Θ(n)",
      run: (n, c) => prefixAverage3(d3.range(n), c) },
    { key: "dj1", name: "3-way disjoint 1 · triple loop", color: AC.bad,
      closed: n => n * n * n, cf: "n³", cls: "Θ(n³)",
      run: (n, c) => disjoint1(d3.range(n), d3.range(n), d3.range(n, 2 * n), c) },
    { key: "dj2", name: "3-way disjoint 2 · triple loop, guarded", color: AC.accent,
      closed: n => 2 * n * n, cf: "2n²", cls: "Θ(n²)",
      run: (n, c) => disjoint2(d3.range(n), d3.range(n), d3.range(n, 2 * n), c) }
  ];

  const NS = d3.range(4, 97, 4);
  const DATA = ROUTINES.map(r => ({
    r: r,
    pts: NS.map(n => { const c = AL.counter(); r.run(n, c); return { n: n, v: c.get("op") }; })
  }));

  function render() {
    const fr = AL.frame(svg, W, H, { l: 64, r: 210, t: 20, b: 42 });
    const g = fr.g;
    const x = d3.scaleLog().domain([4, 96]).range([0, fr.iw]);
    const ymax = d3.max(DATA, s => d3.max(s.pts, p => p.v));
    const y = d3.scaleLog().domain([1, ymax * 1.2]).range([fr.ih, 0]);
    AL.gridY(g, y, fr.iw, 6);
    AL.axisB(g, x, fr.ih, 5, "n (log scale)", d3.format("~s"));
    AL.axisL(g, y, 6, "measured operations (log scale)", d3.format("~s"));
    const ln = d3.line().x(d => x(d.n)).y(d => y(Math.max(1, d.v)));
    DATA.forEach(s => {
      g.append("path").attr("fill", "none").attr("stroke", s.r.color).attr("stroke-width", 2).attr("d", ln(s.pts));
      g.selectAll("circle." + s.r.key).data(s.pts).join("circle").attr("class", s.r.key)
        .attr("cx", d => x(d.n)).attr("cy", d => y(Math.max(1, d.v))).attr("r", 2.2).attr("fill", s.r.color);
    });
    AL.legend(g, DATA.map(s => ({ label: s.r.name, color: s.r.color })), fr.iw + 10, 18, { gap: 17 });

    /* the fitted slope on a log-log plot IS the exponent — measured, not assumed */
    function slope(pts) {
      const a = pts[pts.length - 6], b = pts[pts.length - 1];
      return (Math.log(b.v) - Math.log(a.v)) / (Math.log(b.n) - Math.log(a.n));
    }
    const N0 = 64, i0 = NS.indexOf(N0);
    AN.table("#loops-table",
      ["routine", "measured ops at n = " + N0, "closed form", "value", "fitted log-log slope", "class"],
      DATA.map(s => [
        "<span style='color:" + s.r.color + "'>■</span> " + s.r.name,
        "<b>" + AN.int(s.pts[i0].v) + "</b>",
        "<code>" + s.r.cf + "</code>",
        AN.int(s.r.closed(N0)),
        "<b>" + AN.sig(slope(s.pts), 2) + "</b>",
        "<code>" + s.r.cls + "</code>"
      ]));
    const bad = DATA.filter(s => s.pts[i0].v !== s.r.closed(N0));
    d3.select("#loops-readout").html(
      "at n = " + N0 + ", every measured count equals its closed form"
      + (bad.length ? " — except <b>" + bad.map(s => s.r.key).join(", ") + "</b> ✗"
                    : " ✓ (5 of 5 agree)"));
  }
  render();
})();

/* ══ FIGURE 07 ═══════════════════════════════════════════════════════════
   The recursion tree for T(n) = a·T(n/b) + c·n^k, drawn level by level with
   the per-level work summed. Every number in the level table is computed by
   walking the tree, and the total is the sum of the column — so the geometric
   argument in the prose can be checked against arithmetic.                  */
(function () {
  const svg = d3.select("#rtree-svg"); if (svg.empty()) return;
  const W = 680, H = 330;
  let a = 3, b = 4, k = 2, L = 4;

  function levels() {
    const n = Math.pow(b, L), out = [];
    for (let i = 0; i <= L; i++) {
      const nodes = Math.pow(a, i), size = n / Math.pow(b, i);
      const work = i === L ? nodes : nodes * Math.pow(size, k);   // leaves cost T(1) = 1
      out.push({ i: i, nodes: nodes, size: size, work: work });
    }
    return out;
  }

  function render() {
    const rows = levels(), n = Math.pow(b, L);
    const total = d3.sum(rows, r => r.work);
    const leafWork = rows[L].work, rootWork = rows[0].work;
    const fr = AL.frame(svg, W, H, { l: 78, r: 24, t: 26, b: 32 });
    const g = fr.g;
    const y = d3.scaleBand().domain(rows.map(r => r.i)).range([0, fr.ih]).padding(0.28);
    const x = d3.scaleLinear().domain([0, d3.max(rows, r => r.work) * 1.02]).range([0, fr.iw - 120]);
    g.append("text").attr("x", -70).attr("y", -10).attr("font-size", 11).attr("fill", AC.muted).text("level");
    g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", AC.muted).text("work at this level  (bar length ∝ work)");
    rows.forEach(r => {
      g.append("text").attr("x", -70).attr("y", y(r.i) + y.bandwidth() / 2 + 4).attr("font-size", 11)
        .attr("fill", AC.ink).text(r.i === L ? r.i + " (leaves)" : "" + r.i);
      g.append("text").attr("x", -70).attr("y", y(r.i) + y.bandwidth() / 2 + 15).attr("font-size", 9.5)
        .attr("fill", AC.muted).text(AN.int(r.nodes) + " × size " + (r.size >= 1 ? AN.int(Math.round(r.size)) : AN.sig(r.size, 2)));
      g.append("rect").attr("x", 0).attr("y", y(r.i)).attr("width", Math.max(1, x(r.work)))
        .attr("height", y.bandwidth()).attr("rx", 3)
        .attr("fill", r.i === L ? AC.a2 : AC.accent).attr("opacity", r.i === L ? 0.95 : 0.8);
      g.append("text").attr("x", Math.max(1, x(r.work)) + 8).attr("y", y(r.i) + y.bandwidth() / 2 + 4)
        .attr("font-size", 11).attr("fill", AC.muted)
        .text(AN.int(Math.round(r.work)) + "   (" + AN.sig(100 * r.work / total, 1) + "% of total)");
    });

    const logba = Math.log(a) / Math.log(b);
    AN.table("#rtree-table",
      ["level i", "nodes aⁱ", "size n/bⁱ", "work at level", "running total"],
      (function () {
        let run = 0;
        return rows.map(r => {
          run += r.work;
          return ["<b>" + r.i + (r.i === L ? " (leaves)" : "") + "</b>", AN.int(r.nodes),
                  r.size >= 1 ? AN.int(Math.round(r.size)) : AN.sig(r.size, 3),
                  "<b>" + AN.int(Math.round(r.work)) + "</b>", AN.int(Math.round(run))];
        });
      })());

    d3.select("#rtree-readout").html(
      "T(n) = " + a + "·T(n/" + b + ") + n^" + k + " &nbsp;with n = " + b + "^" + L + " = <b>" + AN.int(n)
      + "</b> &nbsp;·&nbsp; levels <b>" + (L + 1) + "</b> (depth log_" + b + " n = " + L + ")"
      + " &nbsp;·&nbsp; total work summed over levels <b>" + AN.int(Math.round(total)) + "</b>"
      + " &nbsp;·&nbsp; root <b>" + AN.sig(100 * rootWork / total, 1) + "%</b>"
      + " &nbsp;·&nbsp; leaves <b>" + AN.sig(100 * leafWork / total, 1) + "%</b>"
      + " &nbsp;·&nbsp; leaf count a^L = <b>" + AN.int(Math.pow(a, L)) + "</b> = n^(log_" + b + " " + a + ") = n^"
      + AN.sig(logba, 4)
      + " &nbsp;·&nbsp; per-level ratio a/b^k = <b>" + AN.sig(a / Math.pow(b, k), 4) + "</b>");
  }

  function bind(id, set) {
    d3.select(id).on("input", function () { set(+this.value); d3.select(id + "-out").text(this.value); render(); });
    d3.select(id + "-out").text(d3.select(id).property("value"));
  }
  bind("#rtree-a", v => a = v);
  bind("#rtree-b", v => b = v);
  bind("#rtree-k", v => k = v);
  bind("#rtree-l", v => L = v);
  render();
})();

/* ══ FIGURE 08 ═══════════════════════════════════════════════════════════
   The Master-theorem case classifier. It compares f(n) = n^k·log^p n with
   n^(log_b a) and reports which case applies — including "none", which is the
   answer for the two gap examples. The classic-recurrence table underneath is
   produced by the SAME classify() the sliders drive, so the table cannot say
   something the interactive part would contradict.                          */
(function () {
  const svg = d3.select("#master-svg"); if (svg.empty()) return;
  const W = 680, H = 330;

  function sup(e) {                              // a Unicode exponent, where possible
    const map = { "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", ".": "·", "-": "⁻" };
    const s = String(e);
    return [...s].every(ch => map[ch]) ? [...s].map(ch => map[ch]).join("") : "^(" + s + ")";
  }
  function npow(e) {                             // "n", "n²", "n^0.7925"
    const r = Math.abs(e - Math.round(e)) < 1e-9 ? Math.round(e) : +e.toFixed(4);
    if (r === 0) return "1";
    if (r === 1) return "n";
    return Number.isInteger(r) ? "n" + sup(r) : "n^" + r;
  }
  function lpow(e) {
    if (e === 0) return "";
    if (e === 1) return " log n";
    return " log" + sup(e) + " n";
  }
  /* n^kk · log^pp n, written the way a person would write it */
  function term(kk, pp) {
    const A = npow(kk), B = lpow(pp);
    if (A === "1") return B === "" ? "1" : B.trim();
    return A + B;
  }

  function classify(a, b, k, p) {
    const L = Math.log(a) / Math.log(b);
    const eps = 1e-9;
    if (k < L - eps) {
      return { c: "1", ok: true, L: L,
               why: "f(n) is polynomially smaller than " + term(L, 0) + " — the leaves dominate",
               res: "Θ(" + term(L, 0) + ")" };
    }
    if (Math.abs(k - L) < eps) {
      if (p > -1 + eps) {
        return { c: p === 0 ? "2" : "2 (extended — NOT the basic theorem)", ok: true, L: L,
                 why: p === 0 ? "f(n) = Θ(" + term(L, 0) + ") — every level costs the same"
                              : "f(n) = Θ(" + term(L, p) + ") with p ≠ 0, so the THREE BASIC CASES DO NOT APPLY"
                                + " — this is the standard p > −1 extension of case 2, and the basic theorem is silent here",
                 res: "Θ(" + term(L, p + 1) + ")" };
      }
      return { c: "none", ok: false, L: L,
               why: "k = log_b a but p ≤ −1 — outside case 2 and outside its standard extension",
               res: p === -1 ? "Θ(" + term(L, 0) + " log log n)   (known, but not from this theorem)"
                             : "Θ(" + term(L, 0) + ")   (known, but not from this theorem)" };
    }
    /* k > L: polynomially larger, and regularity holds with c = a/b^k < 1 */
    const creg = a / Math.pow(b, k);
    /* regularity: for f = n^k·log^p n with p ≥ 0, c = a/b^k works exactly. For
       p < 0 the log factor moves the wrong way and a/b^k can fail at every
       finite n, but a·f(n/b)/f(n) → a/b^k, so any constant strictly between
       a/b^k and 1 serves for large n. Say which situation we are in.          */
    return { c: "3", ok: true, L: L, creg: creg,
             why: "f(n) is polynomially larger than " + term(L, 0) + " and regularity holds"
                  + (p < 0
                      ? " with any constant c strictly between a/b" + sup(k) + " = " + creg.toFixed(4)
                        + " and 1 (p < 0, so a/b" + sup(k) + " itself is not enough at finite n)"
                      : " with c = a/b" + sup(k) + " = " + creg.toFixed(4) + " < 1")
                  + " — the root dominates",
             res: "Θ(" + term(k, p) + ")" };
  }

  let a = 2, b = 2, k = 1, p = 0;

  function render() {
    const r = classify(a, b, k, p);
    const fr = AL.frame(svg, W, H, { l: 66, r: 170, t: 22, b: 42 });
    const g = fr.g;
    const NHI = 1e6;
    const xs = d3.range(0, 121).map(i => Math.pow(10, 1 + 5 * i / 120));   // 10 … 1e6, log spaced
    const fF = n => Math.pow(n, k) * Math.pow(AL.log2(n), p);
    const gF = n => Math.pow(n, r.L);
    const x = d3.scaleLog().domain([10, NHI]).range([0, fr.iw]);
    const lo = Math.max(1e-3, d3.min(xs, n => Math.min(fF(n), gF(n))));
    const hi = d3.max(xs, n => Math.max(fF(n), gF(n)));
    const y = d3.scaleLog().domain([lo, hi * 1.3]).range([fr.ih, 0]);
    AL.gridY(g, y, fr.iw, 6);
    AL.axisB(g, x, fr.ih, 5, "n (log scale)", d3.format("~s"));
    AL.axisL(g, y, 6, "cost (log scale)", d3.format("~s"));
    const ln = h => d3.line().x(d => x(d)).y(d => y(AL.clamp(h(d), lo, hi * 1.3)))(xs);
    g.append("path").attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 2.4).attr("d", ln(gF));
    g.append("path").attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 2.4).attr("d", ln(fF));
    AL.legend(g, [
      { label: "n^(log_b a) = " + term(r.L, 0) + "   (the leaves)", color: AC.a2 },
      { label: "f(n) = " + term(k, p) + "   (the root)", color: AC.accent }
    ], fr.iw + 10, 18);
    g.append("text").attr("x", 4).attr("y", 12).attr("font-size", 12)
      .attr("fill", r.ok ? AC.good : AC.bad)
      .text(r.ok ? "case " + r.c : "no case applies");

    d3.select("#master-readout").html(
      "T(n) = " + a + "·T(n/" + b + ") + " + term(k, p)
      + " &nbsp;·&nbsp; log_" + b + " " + a + " = <b>" + r.L.toFixed(4) + "</b>"
      + " &nbsp;·&nbsp; k = <b>" + k + "</b>"
      + " &nbsp;·&nbsp; " + (Math.abs(k - r.L) < 1e-9 ? "k = log_b a" : (k < r.L ? "k &lt; log_b a" : "k &gt; log_b a"))
      + " &nbsp;·&nbsp; <b>" + (r.ok ? "case " + r.c : "NO CASE APPLIES") + "</b>"
      + " &nbsp;·&nbsp; " + r.why
      + " &nbsp;⇒&nbsp; T(n) = <b>" + r.res + "</b>");

    const CLASSIC = [
      ["mergesort", 2, 2, 1, 0, "T(n) = 2T(n/2) + n"],
      ["binary search", 1, 2, 0, 0, "T(n) = T(n/2) + 1"],
      ["tree traversal", 2, 2, 0, 0, "T(n) = 2T(n/2) + 1"],
      ["Karatsuba", 3, 2, 1, 0, "T(n) = 3T(n/2) + n"],
      ["Strassen", 7, 2, 2, 0, "T(n) = 7T(n/2) + n²"],
      ["naive matrix multiply", 8, 2, 2, 0, "T(n) = 8T(n/2) + n²"],
      ["textbook case 1 — leaves win", 9, 3, 1, 0, "T(n) = 9T(n/3) + n"],
      ["textbook case 2 — flat tree", 1, 1.5, 0, 0, "T(n) = T(2n/3) + 1"],
      ["textbook case 3 — root wins", 3, 4, 1, 1, "T(n) = 3T(n/4) + n log n"],
      ["gap of the basic theorem", 2, 2, 1, 1, "T(n) = 2T(n/2) + n log n"],
      ["gap of the extension too", 2, 2, 1, -1, "T(n) = 2T(n/2) + n / log n"]
    ];
    AN.table("#master-table",
      ["recurrence", "a", "b", "n^(log_b a)", "f(n)", "case", "T(n)"],
      CLASSIC.map(row => {
        const q = classify(row[1], row[2], row[3], row[4]);
        return ["<code>" + row[5] + "</code>", "" + row[1], "" + row[2],
                "<code>" + term(q.L, 0) + "</code>",
                "<code>" + term(row[3], row[4]) + "</code>",
                q.ok ? "<b>" + q.c + "</b>" : "<b style='color:" + AC.bad + "'>none</b>",
                "<code>" + q.res + "</code>"];
      }));
  }

  d3.select("#master-preset").on("change", function () {
    const v = this.value.split(",").map(Number);
    a = v[0]; b = v[1]; k = v[2]; p = v[3];
    d3.select("#master-a").property("value", a); d3.select("#master-a-out").text(a);
    d3.select("#master-b").property("value", b); d3.select("#master-b-out").text(b);
    d3.select("#master-k").property("value", k); d3.select("#master-k-out").text(k);
    d3.select("#master-p").property("value", p); d3.select("#master-p-out").text(p);
    render();
  });
  function bind(id, set) {
    d3.select(id).on("input", function () { set(+this.value); d3.select(id + "-out").text(this.value); render(); });
    d3.select(id + "-out").text(d3.select(id).property("value"));
  }
  bind("#master-a", v => a = v);
  bind("#master-b", v => b = v);
  bind("#master-k", v => k = v);
  bind("#master-p", v => p = v);
  render();
})();

/* ══ FIGURE 09 ═══════════════════════════════════════════════════════════
   A real k-bit binary counter, incremented n times, with every bit flip
   counted. The bars are the measured per-operation cost; the line is the
   running total divided by the operation index — the amortized cost as it
   actually develops, not as it is claimed to.                              */
(function () {
  const svg = d3.select("#ctr-svg"); if (svg.empty()) return;
  const W = 680, H = 330;
  let N = 64, K = 8;

  function increment(A, c) {                    // exactly the textbook procedure
    let i = 0;
    while (i < A.length && A[i] === 1) { A[i] = 0; c.add("flip"); i++; }
    if (i < A.length) { A[i] = 1; c.add("flip"); }
  }

  function run(n, k) {
    const A = new Array(k).fill(0), c = AL.counter(), per = [], cum = [];
    let prev = 0;
    for (let i = 1; i <= n; i++) {
      increment(A, c);
      const tot = c.get("flip");
      per.push({ i: i, v: tot - prev, tot: tot, avg: tot / i });
      prev = tot;
    }
    return { per: per, total: c.get("flip") };
  }

  function render() {
    const r = run(N, K);
    const fr = AL.frame(svg, W, H, { l: 56, r: 140, t: 22, b: 40 });
    const g = fr.g;
    const x = d3.scaleLinear().domain([0, N + 1]).range([0, fr.iw]);
    const y = d3.scaleLinear().domain([0, Math.max(3, d3.max(r.per, d => d.v)) + 0.5]).range([fr.ih, 0]);
    AL.gridY(g, y, fr.iw, 5);
    AL.axisB(g, x, fr.ih, 8, "INCREMENT number i");
    AL.axisL(g, y, 5, "bits flipped");
    const bw = Math.max(1.2, fr.iw / (N + 2) - 1.2);
    g.selectAll("rect.op").data(r.per).join("rect").attr("class", "op")
      .attr("x", d => x(d.i) - bw / 2).attr("y", d => y(d.v))
      .attr("width", bw).attr("height", d => fr.ih - y(d.v))
      .attr("fill", d => d.v >= 4 ? AC.bad : (d.v >= 2 ? AC.a2 : AC.accent));
    g.append("path").attr("fill", "none").attr("stroke", AC.good).attr("stroke-width", 2.2)
      .attr("d", d3.line().x(d => x(d.i)).y(d => y(d.avg))(r.per));
    g.append("line").attr("x1", 0).attr("x2", fr.iw).attr("y1", y(2)).attr("y2", y(2))
      .attr("stroke", AC.violet).attr("stroke-dasharray", "5,4");
    AL.legend(g, [
      { label: "bits flipped by this INCREMENT", color: AC.accent },
      { label: "running average (total / i)", color: AC.good },
      { label: "the amortized bound, 2", color: AC.violet, dash: "5,4" }
    ], fr.iw + 8, 18);

    /* the aggregate argument, evaluated independently: bit i flips floor(n/2^i) times */
    let agg = 0;
    for (let i = 0; i < K; i++) agg += Math.floor(N / Math.pow(2, i));
    const last = r.per[r.per.length - 1];
    d3.select("#ctr-readout").html(
      "k = " + K + " bits, n = " + N + " INCREMENTs &nbsp;·&nbsp; measured total flips <b>" + AN.int(r.total)
      + "</b> &nbsp;·&nbsp; aggregate argument ∑ᵢ floor(n/2ⁱ) = <b>" + AN.int(agg) + "</b> "
      + (agg === r.total ? "✓ agree" : "✗ DISAGREE")
      + " &nbsp;·&nbsp; amortized cost total/n = <b>" + AN.sig(last.avg, 4)
      + "</b> &nbsp;·&nbsp; bound 2n = <b>" + AN.int(2 * N)
      + "</b> &nbsp;·&nbsp; worst single operation <b>" + d3.max(r.per, d => d.v) + "</b> flips");
  }

  d3.select("#ctr-n").on("input", function () { N = +this.value; d3.select("#ctr-n-out").text(N); render(); });
  d3.select("#ctr-k").on("input", function () { K = +this.value; d3.select("#ctr-k-out").text(K); render(); });
  d3.select("#ctr-n-out").text(N); d3.select("#ctr-k-out").text(K);
  render();
})();

/* ══ FIGURES 10 & 11 ═════════════════════════════════════════════════════
   One real dynamic table, two views of the same run. TABLE-INSERT is the
   textbook procedure; cost is counted in elementary insertions (1 to place
   the item, plus one per item copied on an expansion). Figure 10 shows the
   per-operation cost with the running amortized mean; figure 11 shows the
   potential Phi = 2·num − size, the actual cost and the amortized cost
   c-hat = c + dPhi, which the theory says is 3.                            */
(function () {
  const svg10 = d3.select("#dyn-svg"), svg11 = d3.select("#pot-svg");
  if (svg10.empty() && svg11.empty()) return;
  const W = 680, H = 320;
  let N = 64, growth = 2;

  function run(n, g) {
    const c = AL.counter(), rows = [];
    let num = 0, size = 0, phiPrev = 0;
    for (let i = 1; i <= n; i++) {
      let cost = 0;
      if (size === 0) { size = 1; }
      else if (num === size) {
        const ns = Math.max(size + 1, Math.floor(size * g));
        cost += num; c.add("copy", num);        // copy every existing item
        size = ns;
      }
      cost += 1; c.add("store");                 // the elementary insertion
      num += 1;
      const phi = 2 * num - size;
      rows.push({ i: i, cost: cost, num: num, size: size, phi: phi,
                  hat: cost + phi - phiPrev, tot: c.get("store") + c.get("copy") });
      phiPrev = phi;
    }
    rows.forEach(r => { r.avg = r.tot / r.i; });
    return { rows: rows, total: c.get("store") + c.get("copy"),
             stores: c.get("store"), copies: c.get("copy") };
  }

  function renderCost() {
    if (svg10.empty()) return;
    const r = run(N, growth);
    const fr = AL.frame(svg10, W, H, { l: 56, r: 150, t: 22, b: 40 });
    const g = fr.g;
    const x = d3.scaleLinear().domain([0, N + 1]).range([0, fr.iw]);
    const y = d3.scaleLinear().domain([0, d3.max(r.rows, d => d.cost) * 1.08]).range([fr.ih, 0]);
    AL.gridY(g, y, fr.iw, 5);
    AL.axisB(g, x, fr.ih, 8, "append number i");
    AL.axisL(g, y, 5, "elementary insertions");
    const bw = Math.max(1.2, fr.iw / (N + 2) - 1.2);
    g.selectAll("rect.op").data(r.rows).join("rect").attr("class", "op")
      .attr("x", d => x(d.i) - bw / 2).attr("y", d => y(d.cost))
      .attr("width", bw).attr("height", d => fr.ih - y(d.cost))
      .attr("fill", d => d.cost > 1 ? AC.bad : AC.accent);
    g.append("path").attr("fill", "none").attr("stroke", AC.good).attr("stroke-width", 2.4)
      .attr("d", d3.line().x(d => x(d.i)).y(d => y(d.avg))(r.rows));
    AL.legend(g, [
      { label: "actual cost of append i", color: AC.accent },
      { label: "append that triggered a resize", color: AC.bad },
      { label: "running amortized mean (total / i)", color: AC.good }
    ], fr.iw + 8, 18);

    const spikes = r.rows.filter(d => d.cost > 1);
    d3.select("#dyn-readout").html(
      "growth factor <b>" + growth + "</b>, n = " + N
      + " &nbsp;·&nbsp; measured total <b>" + AN.int(r.total) + "</b> = "
      + AN.int(r.stores) + " stores + " + AN.int(r.copies) + " copies"
      + " &nbsp;·&nbsp; amortized <b>" + AN.sig(r.total / N, 4) + "</b> per append"
      + " &nbsp;·&nbsp; resizes: <b>" + spikes.length + "</b>"
      + " &nbsp;·&nbsp; most expensive single append: <b>" + d3.max(r.rows, d => d.cost)
      + "</b> insertions, at i = " + (spikes.length ? spikes[spikes.length - 1].i : "—")
      + " &nbsp;·&nbsp; asymptotic prediction g/(g−1) = <b>"
      + AN.sig(growth / (growth - 1), 3) + "</b> per append"
      + (growth === 2
          ? " &nbsp;·&nbsp; the 3n bound for doubling: " + AN.int(r.total) + " ≤ " + AN.int(3 * N)
            + " " + (r.total <= 3 * N ? "✓" : "✗")
          : ""));
  }

  function renderPot() {
    if (svg11.empty()) return;
    const r = run(N, 2);                        // the potential 2·num − size assumes doubling
    const fr = AL.frame(svg11, W, H, { l: 56, r: 150, t: 22, b: 40 });
    const g = fr.g;
    const x = d3.scaleLinear().domain([0, N + 1]).range([0, fr.iw]);
    const y = d3.scaleLinear().domain([0, d3.max(r.rows, d => Math.max(d.cost, d.phi, d.hat)) * 1.08])
      .range([fr.ih, 0]);
    AL.gridY(g, y, fr.iw, 5);
    AL.axisB(g, x, fr.ih, 8, "append number i");
    AL.axisL(g, y, 5, "cost / potential");
    const mk = (key, color, w, dash) => {
      const p = g.append("path").attr("fill", "none").attr("stroke", color).attr("stroke-width", w)
        .attr("d", d3.line().x(d => x(d.i)).y(d => y(d[key]))(r.rows));
      if (dash) p.attr("stroke-dasharray", dash);
    };
    mk("cost", AC.accent, 1.6);
    mk("phi", AC.a2, 2);
    mk("hat", AC.good, 2.6);
    AL.legend(g, [
      { label: "actual cost  cᵢ", color: AC.accent },
      { label: "potential  Φᵢ = 2·numᵢ − sizeᵢ", color: AC.a2 },
      { label: "amortized  ĉᵢ = cᵢ + Φᵢ − Φᵢ₋₁", color: AC.good }
    ], fr.iw + 8, 18);

    const hats = r.rows.map(d => d.hat);
    const three = hats.filter(h => h === 3).length;
    AN.table("#pot-table",
      ["i", "cᵢ actual", "numᵢ", "sizeᵢ", "Φᵢ = 2·num − size", "ΔΦ", "ĉᵢ = cᵢ + ΔΦ"],
      r.rows.slice(0, 10).map((d, j) => {
        const prev = j === 0 ? 0 : r.rows[j - 1].phi;
        return ["<b>" + d.i + "</b>", "" + d.cost, "" + d.num, "" + d.size, "" + d.phi,
                (d.phi - prev >= 0 ? "+" : "") + (d.phi - prev), "<b>" + d.hat + "</b>"];
      }));
    d3.select("#pot-readout").html(
      "n = " + N + " &nbsp;·&nbsp; measured ĉᵢ: min <b>" + d3.min(hats) + "</b>, max <b>" + d3.max(hats)
      + "</b> &nbsp;·&nbsp; <b>" + three + "</b> of " + N + " operations have ĉᵢ exactly 3"
      + " &nbsp;·&nbsp; ∑ĉᵢ = <b>" + AN.int(d3.sum(hats)) + "</b> ≥ ∑cᵢ = <b>" + AN.int(r.total)
      + "</b> " + (d3.sum(hats) >= r.total ? "✓" : "✗")
      + " &nbsp;·&nbsp; the gap is Φₙ − Φ₀ = <b>" + (r.rows[N - 1].phi - 0) + "</b>");
  }

  function all() { renderCost(); renderPot(); }
  d3.select("#dyn-n").on("input", function () { N = +this.value; d3.select("#dyn-n-out").text(N); all(); });
  d3.select("#dyn-g").on("input", function () { growth = +this.value; d3.select("#dyn-g-out").text(growth); renderCost(); });
  d3.select("#dyn-n-out").text(N); d3.select("#dyn-g-out").text(growth);
  all();
})();

/* ══ FIGURE 12 ═══════════════════════════════════════════════════════════
   The doubling test. Each routine is run at n, 2n, 4n, … under AL.counter();
   the ratio of consecutive measured costs, log base 2, is the fitted exponent
   b in T(n) ≈ c·n^b. Counting operations rather than reading a clock keeps
   the demonstration deterministic — with a stopwatch the procedure is
   identical and the numbers are noisier.                                    */
(function () {
  const svg = d3.select("#dbl-svg"); if (svg.empty()) return;
  const W = 680, H = 340;

  function linearScan(n, c) { let s = 0; for (let i = 0; i < n; i++) { c.add("op"); s += i; } return s; }
  function binarySearch(n, c) {
    let lo = 0, hi = n - 1, target = -1;
    while (lo <= hi) { c.add("op"); const mid = (lo + hi) >> 1; if (mid === target) return mid; if (mid < target) lo = mid + 1; else hi = mid - 1; }
    return -1;
  }
  function insertionWorst(n, c) {               // reversed input: the quadratic case
    const A = []; for (let i = n; i >= 1; i--) A.push(i);
    for (let j = 1; j < n; j++) {
      const key = A[j]; let i = j - 1;
      while (i >= 0) { c.add("op"); if (A[i] > key) { A[i + 1] = A[i]; i--; } else break; }
      A[i + 1] = key;
    }
  }
  function mergesort(n, c) {
    const rnd = AL.rng(7), A = AL.perm(n, rnd);
    (function ms(a) {
      if (a.length <= 1) return a;
      const mid = a.length >> 1, L = ms(a.slice(0, mid)), R = ms(a.slice(mid)), out = [];
      let i = 0, j = 0;
      while (i < L.length && j < R.length) { c.add("op"); if (L[i] <= R[j]) out.push(L[i++]); else out.push(R[j++]); }
      while (i < L.length) out.push(L[i++]);
      while (j < R.length) out.push(R[j++]);
      return out;
    })(A);
  }
  function tripleLoop(n, c) {
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) c.add("op");
  }

  const R = {
    linear: { name: "linear scan", fn: linearScan, exp: "1", cls: "Θ(n)", max: 8192 },
    binary: { name: "binary search", fn: binarySearch, exp: "0", cls: "Θ(log n)", max: 8192 },
    merge:  { name: "mergesort", fn: mergesort, exp: "1 + 1/log₂ n", cls: "Θ(n log n)", max: 8192 },
    insert: { name: "insertion sort, reversed input", fn: insertionWorst, exp: "2", cls: "Θ(n²)", max: 2048 },
    cube:   { name: "triple nested loop", fn: tripleLoop, exp: "3", cls: "Θ(n³)", max: 128 }
  };
  let key = "insert", n0 = 64;

  function render() {
    const r = R[key], pts = [];
    /* always at least five doublings: if the routine is too slow to reach n0,
       start lower rather than showing two points and a meaningless ratio */
    const start = Math.min(n0, Math.max(4, r.max / 16));
    for (let n = start; n <= r.max; n *= 2) {
      const c = AL.counter(); r.fn(n, c);
      pts.push({ n: n, v: Math.max(1, c.get("op")) });
    }
    pts.forEach((p, i) => {
      p.ratio = i === 0 ? null : p.v / pts[i - 1].v;
      p.b = i === 0 ? null : AL.log2(p.ratio);
    });
    const fr = AL.frame(svg, W, H, { l: 64, r: 130, t: 22, b: 42 });
    const g = fr.g;
    const x = d3.scaleLog().domain([pts[0].n, pts[pts.length - 1].n]).range([0, fr.iw]);
    const y = d3.scaleLog().domain([Math.max(1, d3.min(pts, p => p.v) / 2), d3.max(pts, p => p.v) * 2])
      .range([fr.ih, 0]);
    AL.gridY(g, y, fr.iw, 5);
    AL.axisB(g, x, fr.ih, 5, "n (log scale)", d3.format("~s"));
    AL.axisL(g, y, 5, "measured operations (log scale)", d3.format("~s"));
    g.append("path").attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 2.4)
      .attr("d", d3.line().x(p => x(p.n)).y(p => y(p.v))(pts));
    g.selectAll("circle.p").data(pts).join("circle").attr("class", "p")
      .attr("cx", p => x(p.n)).attr("cy", p => y(p.v)).attr("r", 4).attr("fill", AC.a2);
    pts.forEach(p => {
      if (p.b === null) return;
      g.append("text").attr("x", x(p.n)).attr("y", y(p.v) - 10).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", AC.muted).text("×" + AN.sig(p.ratio, 2));
    });

    AN.table("#dbl-table",
      ["n", "measured operations", "T(2n)/T(n)", "fitted exponent  log₂ ratio"],
      pts.map(p => ["<b>" + AN.int(p.n) + "</b>", AN.int(p.v),
                    p.ratio === null ? "—" : AN.sig(p.ratio, 3),
                    p.b === null ? "—" : "<b>" + AN.sig(p.b, 3) + "</b>"]));

    const lastB = pts[pts.length - 1].b;
    d3.select("#dbl-readout").html(
      r.name + " &nbsp;·&nbsp; fitted exponent at the largest doubling: <b>" + AN.sig(lastB, 3)
      + "</b> &nbsp;·&nbsp; predicted exponent <code>" + r.exp + "</code>"
      + " &nbsp;·&nbsp; class <code>" + r.cls + "</code>"
      + " &nbsp;·&nbsp; cost model T(n) ≈ c·n^" + AN.sig(lastB, 2)
      + " with c = <b>" + AN.sig(pts[pts.length - 1].v / Math.pow(pts[pts.length - 1].n, lastB), 3) + "</b>");
  }

  d3.select("#dbl-routine").on("change", function () { key = this.value; render(); });
  d3.select("#dbl-n0").on("input", function () { n0 = +this.value; d3.select("#dbl-n0-out").text(n0); render(); });
  d3.select("#dbl-n0-out").text(n0);
  render();
})();
