/* heaps-priority-queues.viz.js — figures for
   dsa/data-structures/heaps-priority-queues.html (part 4 of the Data Structures series).

   Loaded after ../../data.js, ../../notes.js and ../dsa-viz.js, so the globals
   AC (palette) and AL (rng / frame / axisB / axisL / gridY / row / binTree /
   counter / stepper / …) are available.

   House rule obeyed throughout: every cost this page DISPLAYS — comparisons,
   swaps, sift steps, links, cuts, potential — comes from running the real
   routine under AL.counter() and reading the counter back. Nothing below is a
   number typed into a caption. Wherever a closed form exists (floor(log2 n),
   2·floor(log2 n), the Σ ceil(n/2^(h+1))·2h build bound, 2n, n·log2 n, the
   4 − c potential bound, floor(log_φ n)) it is evaluated a SECOND, independent
   way and printed next to the measurement, so each figure cross-checks itself
   and says "agree" / "within bound" or "DISAGREE" / "EXCEEDS" in colour.

   The heap routines in HP below are the routines the page describes, 0-based,
   max-heap unless stated; the min-heap figures pass a reversed comparator.
   Every routine takes an AL.counter() and bumps "cmp" once per key comparison
   and "swap" once per exchange, so the same code both draws and measures.

   Figures, in FILE order (which is also page order). 15 figures.
     01 #twin-svg      the same level-order array as boxes and as a tree
     02 #height-svg    height = floor(log2 n), leaves = ceil(n/2), measured by walking
     03 #up-svg        sift-up stepped: insert 15 into the ten-element max-heap
     04 #down-svg      sift-down stepped: both children compared at every level
     05 #build-svg     bottom-up build-heap stepped, internal nodes in reverse
     06 #bsweep-svg    build cost vs n: bottom-up vs repeated insertion, measured
     07 #hs-svg        heapsort stepped, with the heap / sorted boundary
     08 #hsweep-svg    heapsort comparisons vs n on four input shapes + bottom-up variant
     09 #handle-svg    decreaseKey / delete through a handle (index heap), stepped
     10 #dary-svg      d-ary heaps: insert vs extract cost as d grows, two workloads
     11 #cons-svg      Fibonacci heap: consolidate after extractMin, stepped, with Φ
     12 #cut-svg       Fibonacci heap: cascading cut, stepped, with Φ and 4 − c
     13 #topk-svg      k smallest from a stream with a size-k max-heap
     14 #median-svg    running median with two heaps
     15 #local-svg     which indices a sift-down touches, and the gaps between them */

/* ── shared helpers ────────────────────────────────────────────────────── */
const HQ = {
  int: d3.format(","),
  sig: function (x, d) { return (+x).toFixed(d === undefined ? 2 : d); },
  verdict: function (ok, yes, no) {
    return ok ? '<b style="color:' + AC.good + '">' + (yes || "agree") + '</b>'
              : '<b style="color:' + AC.bad + '">' + (no || "DISAGREE") + '</b>';
  },
  clearControls: function (svgNode) {
    svgNode.parentNode.querySelectorAll('div[role="group"]').forEach(el => el.remove());
  },
  floorLog2: function (n) { let h = 0; while ((1 << (h + 1)) <= n) h++; return h; },
  nlogn: function (n) { return n <= 1 ? 0 : n * AL.log2(n); }
};

/* ── HP: the instrumented heap routines the page describes ─────────────── */
const HP = (function () {
  const parent = i => Math.floor((i - 1) / 2);
  const left = i => 2 * i + 1;
  const right = i => 2 * i + 2;

  /* `above(a, b)` is true when a belongs nearer the root than b: for a max-heap
     a > b, for a min-heap a < b. Every comparison goes through here. */
  const MAX = (a, b) => a > b;
  const MIN = (a, b) => a < b;

  function swap(a, i, j, c) { const t = a[i]; a[i] = a[j]; a[j] = t; c.add("swap"); }

  /* sift-up from index i. Returns the final index. `rec` (optional) receives a
     frame after every comparison so a stepper can replay the run. */
  function siftUp(a, i, c, above, rec) {
    above = above || MAX;
    while (i > 0) {
      const p = parent(i);
      c.add("cmp");
      if (above(a[i], a[p])) {
        if (rec) rec({ arr: a.slice(), focus: i, cmp: [i, p], swap: [i, p],
                       note: "a[" + i + "] = " + a[i] + " belongs above a[" + p + "] = " + a[p] + " → swap" });
        swap(a, i, p, c);
        i = p;
      } else {
        if (rec) rec({ arr: a.slice(), focus: i, cmp: [i, p], swap: null,
                       note: "a[" + i + "] = " + a[i] + " does not beat its parent a[" + p + "] = " + a[p] + " → stop" });
        return i;
      }
    }
    if (rec) rec({ arr: a.slice(), focus: 0, cmp: null, swap: null, note: "reached the root → stop" });
    return i;
  }

  /* sift-down from index i within a[0 … n−1]. At each node: compare left with
     the current best (1 comparison), then right with the best (1 comparison),
     then swap if the best is a child. That is the "compare both children"
     discipline: at most 2 comparisons per level, fewer at a node with one child. */
  function siftDown(a, i, n, c, above, rec) {
    above = above || MAX;
    for (;;) {
      const l = left(i), r = right(i);
      let best = i;
      let cmps = [];
      if (l < n) { c.add("cmp"); cmps.push([l, best]); if (above(a[l], a[best])) best = l; }
      if (r < n) { c.add("cmp"); cmps.push([r, best]); if (above(a[r], a[best])) best = r; }
      if (best === i) {
        if (rec) rec({ arr: a.slice(), focus: i, cmp: cmps, swap: null, size: n,
                       note: l < n ? "a[" + i + "] = " + a[i] + " already beats its child" + (r < n ? "ren" : "") + " → stop"
                                   : "a[" + i + "] is a leaf → stop" });
        return i;
      }
      if (rec) rec({ arr: a.slice(), focus: i, cmp: cmps, swap: [i, best], size: n,
                     note: "larger child a[" + best + "] = " + a[best] + " beats a[" + i + "] = " + a[i] + " → swap, continue at " + best });
      swap(a, i, best, c);
      i = best;
    }
  }

  /* bottom-up construction: sift-down every internal node, last one first */
  function build(a, c, above, rec) {
    const n = a.length;
    for (let i = Math.floor(n / 2) - 1; i >= 0; i--) {
      if (rec) rec({ arr: a.slice(), focus: i, cmp: null, swap: null, size: n, node: i,
                     note: "process internal node " + i + " (value " + a[i] + ")" });
      siftDown(a, i, n, c, above, rec);
    }
    return a;
  }

  /* construction by n repeated insertions */
  function buildByInsert(src, c, above) {
    const a = [];
    for (let k = 0; k < src.length; k++) { a.push(src[k]); siftUp(a, a.length - 1, c, above); }
    return a;
  }

  /* classic in-place heapsort (ascending, so a MAX-heap) */
  function heapsort(a, c, rec) {
    build(a, c, MAX, rec ? (f) => { f.phase = "build"; rec(f); } : null);
    for (let end = a.length - 1; end >= 1; end--) {
      if (rec) rec({ arr: a.slice(), focus: 0, cmp: null, swap: [0, end], size: end + 1, phase: "sort",
                     note: "swap the root " + a[0] + " into slot " + end + "; heap shrinks to " + end });
      swap(a, 0, end, c);
      siftDown(a, 0, end, c, MAX, rec ? (f) => { f.phase = "sort"; rec(f); } : null);
    }
    return a;
  }

  /* bottom-up heapsort's reheap: walk the max-child path to a leaf with ONE
     comparison per level, then climb back up until the sifted value fits, then
     shift the path down by one and drop the value in. Same result as siftDown,
     fewer comparisons when the value belongs deep (as heapsort's roots do). */
  function siftDownBottomUp(a, i, n, c) {
    const v = a[i];
    const path = [i];
    let j = i;
    while (left(j) < n) {                     // 1 comparison per level: which child is larger
      const l = left(j), r = right(j);
      if (r < n) { c.add("cmp"); j = a[r] > a[l] ? r : l; } else j = l;
      path.push(j);
    }
    let t = path.length - 1;                  // climb from the leaf while v beats the path node
    while (t > 0) { c.add("cmp"); if (v > a[path[t]]) t--; else break; }
    for (let s = 0; s < t; s++) { a[path[s]] = a[path[s + 1]]; c.add("move"); }   // shift the path up by one
    a[path[t]] = v;                           // and drop v where the climb stopped
  }
  function heapsortBottomUp(a, c) {
    const n = a.length;
    for (let i = Math.floor(n / 2) - 1; i >= 0; i--) siftDownBottomUp(a, i, n, c);
    for (let end = n - 1; end >= 1; end--) { swap(a, 0, end, c); siftDownBottomUp(a, 0, end, c); }
    return a;
  }

  /* ── d-ary heap (max), same comparison discipline generalised ── */
  function dary(d) {
    const par = i => Math.floor((i - 1) / d);
    const child = (i, k) => d * i + k + 1;
    function up(a, i, c) {
      while (i > 0) { const p = par(i); c.add("cmp"); if (a[i] > a[p]) { swap(a, i, p, c); i = p; } else return i; }
      return i;
    }
    function down(a, i, n, c) {
      for (;;) {
        let best = i;
        for (let k = 0; k < d; k++) { const ch = child(i, k); if (ch >= n) break; c.add("cmp"); if (a[ch] > a[best]) best = ch; }
        if (best === i) return i;
        swap(a, i, best, c); i = best;
      }
    }
    return { par, child, up, down,
      insert: (a, v, c) => { a.push(v); return up(a, a.length - 1, c); },
      extract: (a, c) => { const top = a[0]; const last = a.pop(); if (a.length) { a[0] = last; down(a, 0, a.length, c); } return top; },
      increaseKey: (a, i, v, c) => { a[i] = v; return up(a, i, c); } };
  }

  return { parent, left, right, MAX, MIN, swap, siftUp, siftDown, build, buildByInsert,
           heapsort, siftDownBottomUp, heapsortBottomUp, dary };
})();

/* ── small drawing helpers shared by the stepped figures ──────────────── */
const HD = {
  /* draw an array row and the same array as a tree, with a colouring function */
  twin: function (g, arr, opt) {
    const o = Object.assign({ x: 0, y: 0, w: 34, treeW: 420, treeX: 0, treeY: 60, levelH: 52, r: 15,
                              size: arr.length, mark: null, rowLabel: null }, opt || {});
    const row = AL.row(g, arr, { x: o.x, y: o.y, w: o.w, gap: 3, mark: o.mark, label: o.rowLabel });
    const inHeap = arr.map((v, i) => i < o.size ? v : null);
    const tree = AL.binTree(g, inHeap, { x: o.treeX, y: o.treeY, w: o.treeW, levelH: o.levelH, r: o.r,
                                         mark: o.mark });
    return { row, tree };
  },
  /* a little "counter panel" text block */
  counts: function (g, x, y, lines, opt) {
    const o = Object.assign({ size: 11.5, gap: 16, color: AC.muted }, opt || {});
    lines.forEach((t, i) => {
      g.append("text").attr("x", x).attr("y", y + i * o.gap).attr("font-size", o.size)
        .attr("fill", typeof t === "object" ? (t.color || o.color) : o.color)
        .attr("font-weight", typeof t === "object" && t.bold ? 700 : 400)
        .text(typeof t === "object" ? t.text : t);
    });
  }
};

/* ═══════════════════════════════════════════════════════════════════════
   FIGURE 01 — #twin-svg  the same array as boxes and as a tree
   ═══════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#twin-svg"); if (svg.empty()) return;
  const W = 760, H = 330;
  const A = [16, 14, 10, 8, 7, 9, 3, 2, 4, 1];
  const els = { i: d3.select("#twin-i"), iOut: d3.select("#twin-i-out"), base: d3.select("#twin-base"),
                out: d3.select("#twin-readout") };

  function draw() {
    const i = +els.i.property("value");
    const base = els.base.empty() ? "0" : els.base.property("value");
    els.iOut.text(i);
    const n = A.length;
    const p = i > 0 ? HP.parent(i) : null, l = HP.left(i), r = HP.right(i);
    const lIn = l < n, rIn = r < n;
    // an independent second computation of the same relations, from the 1-based formulas
    const j = i + 1;
    const p1 = j > 1 ? Math.floor(j / 2) - 1 : null, l1 = 2 * j - 1, r1 = 2 * j;
    const agree = (p === p1) && (l === l1) && (r === r1);

    const mark = (k) => k === i ? AC.accent : (k === p ? AC.a2 : ((k === l || k === r) && k < n ? AC.teal : null));
    const f = AL.frame(svg, W, H, { l: 14, r: 14, t: 24, b: 8 });
    const g = f.g;
    g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted)
      .text("the array, indices " + (base === "1" ? "1 … n (1-based labels)" : "0 … n−1 (0-based labels)"));
    const rowG = g.append("g");
    const row = AL.row(rowG, A, { x: 0, y: 4, w: 34, gap: 3, mark: mark, index: false });
    A.forEach((_, k) => {
      rowG.append("text").attr("x", row.cellX(k)).attr("y", 4 + 30 + 13).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", k === i ? AC.accent : AC.muted).text(base === "1" ? k + 1 : k);
    });
    // arrows from the selected cell to its parent and children in the row
    const ay = 4 + 30 + 20;
    if (p !== null) AL.arrow(rowG, row.cellX(i), ay, row.cellX(p), ay, { color: AC.a2, w: 1.4, head: 5 });
    if (lIn) AL.arrow(rowG, row.cellX(i), ay + 10, row.cellX(l), ay + 10, { color: AC.teal, w: 1.4, head: 5 });
    if (rIn) AL.arrow(rowG, row.cellX(i), ay + 18, row.cellX(r), ay + 18, { color: AC.teal, w: 1.4, head: 5 });

    g.append("text").attr("x", 0).attr("y", 96).attr("font-size", 11).attr("fill", AC.muted)
      .text("the same array read as a complete binary tree — level order, left to right");
    const tree = AL.binTree(g, A, { x: 0, y: 104, w: 430, levelH: 50, r: 15, mark: mark });
    // index labels beside the nodes
    A.forEach((_, k) => {
      const q = tree.pos(k);
      g.append("text").attr("x", q.x + 18).attr("y", q.y - 9).attr("font-size", 9.5).attr("fill", AC.muted)
        .text(base === "1" ? k + 1 : k);
    });
    AL.legend(g, [{ label: "selected index i", color: AC.accent }, { label: "parent", color: AC.a2 },
                  { label: "children", color: AC.teal }], 470, 118);
    const lab = k => k === null ? "none" : (base === "1" ? (k + 1) : k);
    HD.counts(g, 470, 180, [
      { text: "0-based:  parent = floor((i−1)/2), left = 2i+1, right = 2i+2", color: AC.muted },
      { text: "1-based:  parent = floor(i/2),     left = 2i,   right = 2i+1", color: AC.muted },
      { text: "selected " + (base === "1" ? "i = " + (i + 1) : "i = " + i) + "  value " + A[i], color: AC.ink, bold: true },
      { text: "parent → " + lab(p) + (p !== null ? " (value " + A[p] + ")" : "  — the root"), color: AC.a2 },
      { text: "left   → " + (lIn ? lab(l) + " (value " + A[l] + ")" : lab(l) + " ≥ n: absent"), color: AC.teal },
      { text: "right  → " + (rIn ? lab(r) + " (value " + A[r] + ")" : lab(r) + " ≥ n: absent"), color: AC.teal },
      { text: "leaf? " + (lIn ? "no" : "yes") + "  (i ≥ floor(n/2) = " + Math.floor(n / 2) + " ⇔ leaf)", color: AC.muted }
    ]);

    els.out.html("<b>n = " + n + ", i = " + (base === "1" ? (i + 1) + " (1-based)" : i + " (0-based)") + ".</b> "
      + "Relations computed with the 0-based formulas and, independently, with the 1-based formulas on i+1 then shifted back: "
      + HQ.verdict(agree) + ". "
      + "Heap property at i: " + (p !== null ? "a[parent] = " + A[p] + " ≥ a[i] = " + A[i] + " ✓" : "the root has no parent")
      + (lIn ? "; a[i] = " + A[i] + " ≥ a[left] = " + A[l] + (rIn ? " and ≥ a[right] = " + A[r] : "") + " ✓" : "; i is a leaf, nothing below to check")
      + ". Notice the children sit <b>side by side</b> in the row while the parent sits about <b>i/2</b> cells away: siblings are adjacent, generations are not.");
  }
  [els.i, els.base].forEach(s => { if (!s.empty()) s.on("input change", draw); });
  draw();
})();

/* ═══════════════════════════════════════════════════════════════════════
   FIGURE 02 — #height-svg  height, leaves, last internal node — measured
   ═══════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#height-svg"); if (svg.empty()) return;
  const W = 760, H = 330;
  const els = { n: d3.select("#height-n"), nOut: d3.select("#height-n-out"), out: d3.select("#height-readout") };

  function draw() {
    const n = +els.n.property("value");
    els.nOut.text(n);
    const arr = Array.from({ length: n }, (_, i) => i);          // values = indices, so the picture reads as a map
    // MEASURED: walk from the last node to the root counting edges; count nodes with no left child;
    // find the largest index that has a child.
    const c = AL.counter();
    let k = n - 1, edges = 0;
    while (k > 0) { k = HP.parent(k); edges++; c.add("walk"); }
    let leaves = 0, lastInternal = -1;
    for (let i = 0; i < n; i++) { c.add("visit"); if (HP.left(i) >= n) leaves++; else lastInternal = i; }
    // maximum height over ALL nodes (longest root-to-leaf walk), a second way to get the height
    let maxDepth = 0;
    for (let i = 0; i < n; i++) { let d = 0, j = i; while (j > 0) { j = HP.parent(j); d++; } if (d > maxDepth) maxDepth = d; }
    // CLOSED FORMS
    const hF = HQ.floorLog2(n), leavesF = Math.ceil(n / 2), lastF = Math.floor(n / 2) - 1;
    const ok = edges === hF && leaves === leavesF && lastInternal === lastF && maxDepth === hF;
    // nodes per height h vs the bound ceil(n / 2^(h+1))
    const heightOf = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      const l = HP.left(i), r = HP.right(i);
      heightOf[i] = Math.max(l < n ? heightOf[l] + 1 : 0, r < n ? heightOf[r] + 1 : 0);
    }
    const perH = [];
    for (let h = 0; h <= hF; h++) perH.push({ h, count: heightOf.filter(x => x === h).length, bound: Math.ceil(n / Math.pow(2, h + 1)) });
    const boundOk = perH.every(d => d.count <= d.bound);

    const f = AL.frame(svg, W, H, { l: 14, r: 14, t: 16, b: 8 });
    const g = f.g;
    const r = n > 16 ? 11 : 14;
    const levelH = n > 16 ? 48 : 56;
    const tree = AL.binTree(g, arr, { x: 0, y: 0, w: 470, levelH: levelH, r: r,
      mark: (i) => i === n - 1 ? AC.accent : (i === lastInternal ? AC.a2 : (HP.left(i) >= n ? "#1e3a2f" : null)) });
    AL.legend(g, [{ label: "last node, n−1 — its depth IS the height", color: AC.accent },
                  { label: "last internal node, floor(n/2)−1", color: AC.a2 },
                  { label: "leaves, ceil(n/2) of them", color: "#4ade80" }], 500, 14);
    HD.counts(g, 500, 76, [
      { text: "n = " + n, color: AC.ink, bold: true },
      { text: "height: walked " + edges + " edges up from n−1;  floor(log₂ n) = " + hF, color: edges === hF ? AC.good : AC.bad },
      { text: "deepest node over all n: " + maxDepth, color: maxDepth === hF ? AC.good : AC.bad },
      { text: "leaves: counted " + leaves + ";  ceil(n/2) = " + leavesF, color: leaves === leavesF ? AC.good : AC.bad },
      { text: "last internal: found " + lastInternal + ";  floor(n/2)−1 = " + lastF, color: lastInternal === lastF ? AC.good : AC.bad },
      { text: "nodes of height h ≤ ceil(n/2^(h+1)):", color: AC.muted }
    ].concat(perH.map(d => ({ text: "   h = " + d.h + ":  " + d.count + " ≤ " + d.bound + (d.count === d.bound ? "  (tight)" : ""),
                              color: d.count <= d.bound ? AC.good : AC.bad }))));

    els.out.html("<b>n = " + n + ".</b> Every quantity on the right was <b>measured by walking the index arithmetic</b> ("
      + c.get("walk") + " parent steps, " + c.get("visit") + " node visits) and then recomputed from the closed forms: "
      + HQ.verdict(ok) + ". The per-height counts against the ceil(n/2^(h+1)) bound: "
      + HQ.verdict(boundOk, "all within bound", "BOUND EXCEEDED") + "."
      + (n === 1 ? " At n = 1 the root is the only node, height 0, one leaf, and the last internal index is −1: there is none." : "")
      + (n === 2 ? " At n = 2 the height is 1, there is exactly one leaf, and node 0 is the last (and only) internal node." : "")
      + (n === 3 ? " At n = 3 the height is still 1 — the second level is now full — and there are two leaves." : "")
      + ((n & (n + 1)) === 0 && n > 3 ? " n = " + n + " is 2^(h+1)−1: every level is full, the perfect case." : "")
      + ((n & (n - 1)) === 0 && n > 2 ? " n = " + n + " is a power of two: a new level has just opened with a single node on it, and the height stepped up by one." : ""));
  }
  els.n.on("input change", draw);
  draw();
})();

/* ── a shared renderer for the stepped array+tree figures ─────────────── */
HD.stepFrame = function (svg, W, H, frame, opt) {
  const o = Object.assign({ treeW: 430, levelH: 50, r: 15, rowY: 4, treeY: 96, panelX: 470, panelY: 110,
                            lines: [], title: null, size: null, sorted: null }, opt || {});
  const f = AL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 8 });
  const g = f.g;
  const arr = frame.arr;
  const n = o.size === null ? arr.length : o.size;
  const cmpSet = new Set();
  (frame.cmp || []).forEach(p => { if (Array.isArray(p)) p.forEach(i => cmpSet.add(i)); else cmpSet.add(p); });
  if (frame.cmp && frame.cmp.length && !Array.isArray(frame.cmp[0])) frame.cmp.forEach(i => cmpSet.add(i));
  const swapSet = new Set(frame.swap || []);
  const mark = (i) => {
    if (o.sorted !== null && i >= o.sorted) return "#1e3a2f";
    if (swapSet.has(i)) return AC.rose;
    if (i === frame.focus) return AC.accent;
    if (cmpSet.has(i)) return AC.a2;
    if (i >= n) return "#2a2a2a";
    return null;
  };
  if (o.title) g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text(o.title);
  const w = arr.length > 12 ? 28 : 34;
  AL.row(g, arr, { x: 0, y: o.rowY, w: w, gap: 3, mark: mark });
  const inHeap = arr.map((v, i) => i < n ? v : null);
  AL.binTree(g, inHeap, { x: 0, y: o.treeY, w: o.treeW, levelH: o.levelH, r: o.r, mark: mark });
  if (frame.note) {
    g.append("text").attr("x", 0).attr("y", o.rowY + 62).attr("font-size", 11.5).attr("fill", AC.ink).text(frame.note);
  }
  HD.counts(g, o.panelX, o.panelY, o.lines);
  return g;
};

/* ═══════════════════════════════════════════════════════════════════════
   FIGURE 03 — #up-svg  sift-up stepped: insert 15 into the ten-element heap
   ═══════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#up-svg"); if (svg.empty()) return;
  const W = 760, H = 330;
  const BASE = [16, 14, 10, 8, 7, 9, 3, 2, 4, 1];
  const els = { key: d3.select("#up-key"), out: d3.select("#up-readout") };
  let st = null;

  function build() {
    const key = +els.key.property("value");
    const a = BASE.slice(); a.push(key);
    const c = AL.counter();
    const frames = [{ arr: a.slice(), focus: a.length - 1, cmp: null, swap: null,
                      note: "append " + key + " at index " + (a.length - 1) + " — the only slot that keeps the shape complete",
                      cmpN: 0, swapN: 0 }];
    HP.siftUp(a, a.length - 1, c, HP.MAX, (fr) => { fr.cmpN = c.get("cmp"); fr.swapN = c.get("swap"); frames.push(fr); });
    frames.push({ arr: a.slice(), focus: -1, cmp: null, swap: null, note: "done — heap property holds everywhere", cmpN: c.get("cmp"), swapN: c.get("swap") });
    // independent checks
    const n = a.length;
    const bound = HQ.floorLog2(n);                       // depth of the last index = floor(log2 n)
    let depth = 0, k = n - 1; while (k > 0) { k = HP.parent(k); depth++; }
    let valid = true; for (let i = 1; i < n; i++) if (a[HP.parent(i)] < a[i]) valid = false;
    const cmp = c.get("cmp"), sw = c.get("swap");
    // where did the key end up?  a second way to count swaps: its final depth vs its start depth
    const finalIdx = a.indexOf(key);
    let fd = 0; k = finalIdx; while (k > 0) { k = HP.parent(k); fd++; }
    const swapsByDepth = depth - fd;
    HQ.clearControls(svg.node());
    st = AL.stepper(svg, { frames, delay: 900, label: "step", render: (fr, i) => {
      HD.stepFrame(svg, W, H, fr, { lines: [
        { text: "insert " + key + " into a heap of " + (n - 1) + "  → n = " + n, color: AC.ink, bold: true },
        { text: "comparisons so far: " + fr.cmpN, color: AC.a2 },
        { text: "swaps so far: " + fr.swapN, color: AC.rose },
        { text: "bound: floor(log₂ " + n + ") = " + bound + " comparisons, " + bound + " swaps", color: AC.muted },
        { text: "depth of index " + (n - 1) + " (walked): " + depth, color: depth === bound ? AC.good : AC.bad },
        { text: "path: index " + (n - 1) + " → " + HP.parent(n - 1) + " → " + HP.parent(HP.parent(n - 1)) + " → 0", color: AC.muted }
      ] });
    } });
    els.out.html("<b>Inserted " + key + ".</b> Measured <b>" + cmp + "</b> comparisons and <b>" + sw + "</b> swaps; the bound floor(log₂ " + n + ") = " + bound
      + " for both: " + HQ.verdict(cmp <= bound && sw <= bound, "within bound", "EXCEEDS BOUND") + ". "
      + "Swaps recounted a second way — start depth " + depth + " minus final depth " + fd + " = " + swapsByDepth + " — " + HQ.verdict(swapsByDepth === sw) + ". "
      + "Heap property re-checked at every index after the run: " + HQ.verdict(valid, "holds", "VIOLATED") + ". "
      + (cmp === sw ? "The key climbed all the way to the root, so every comparison was followed by a swap and there was no final failing comparison." :
         "Comparisons exceed swaps by exactly one: the last comparison is the one that <i>fails</i> and stops the climb.")
      + " The key landed at index " + finalIdx + ".");
  }
  els.key.on("input change", build);
  build();
})();

/* ═══════════════════════════════════════════════════════════════════════
   FIGURE 04 — #down-svg  sift-down stepped, both children compared
   ═══════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#down-svg"); if (svg.empty()) return;
  const W = 760, H = 330;
  const els = { mode: d3.select("#down-mode"), out: d3.select("#down-readout") };

  function build() {
    const mode = els.mode.empty() ? "fix1" : els.mode.property("value");
    let a, start, title;
    if (mode === "fix1") { a = [16, 4, 10, 14, 7, 9, 3, 2, 8, 1]; start = 1; title = "restore the heap at index 1, whose value 4 is out of place (both subtrees are heaps)"; }
    else if (mode === "extract") { a = [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]; start = 0; title = "extractMax: move the last key (1) to the root, shrink, sift down"; }
    else { a = [1, 14, 10, 7, 8, 9, 3, 2, 4, 6]; start = 0; title = "a root that falls the full height, through a node with only one child"; }
    const c = AL.counter();
    const frames = [];
    let n = a.length;
    if (mode === "extract") {
      frames.push({ arr: a.slice(), focus: 0, cmp: null, swap: [0, n - 1], size: n, note: "take out the root 16; move a[9] = 1 to the root; heap size 10 → 9", cmpN: 0, swapN: 0 });
      a[0] = a[n - 1]; a.pop(); n = a.length;
    }
    frames.push({ arr: a.slice(), focus: start, cmp: null, swap: null, size: n, note: title, cmpN: 0, swapN: 0 });
    HP.siftDown(a, start, n, c, HP.MAX, (fr) => { fr.cmpN = c.get("cmp"); fr.swapN = c.get("swap"); frames.push(fr); });
    frames.push({ arr: a.slice(), focus: -1, cmp: null, swap: null, size: n, note: "done — the subtree at " + start + " is a heap again", cmpN: c.get("cmp"), swapN: c.get("swap") });
    // independent checks
    const h = HQ.floorLog2(n);
    let hs = 0; { // height of the start node, measured: longest downward walk
      const hh = new Array(n).fill(0);
      for (let i = n - 1; i >= 0; i--) { const l = HP.left(i), r = HP.right(i); hh[i] = Math.max(l < n ? hh[l] + 1 : 0, r < n ? hh[r] + 1 : 0); }
      hs = hh[start];
    }
    let valid = true; for (let i = 1; i < n; i++) if (a[HP.parent(i)] < a[i]) valid = false;
    const cmp = c.get("cmp"), sw = c.get("swap");
    HQ.clearControls(svg.node());
    AL.stepper(svg, { frames, delay: 900, label: "step", render: (fr) => {
      HD.stepFrame(svg, W, H, fr, { size: fr.size, lines: [
        { text: "sift-down from index " + start + ", heap size " + n, color: AC.ink, bold: true },
        { text: "comparisons so far: " + fr.cmpN, color: AC.a2 },
        { text: "swaps so far: " + fr.swapN, color: AC.rose },
        { text: "bound: 2 × height(" + start + ") = 2 × " + hs + " = " + (2 * hs), color: AC.muted },
        { text: "root bound: 2·floor(log₂ " + n + ") = " + (2 * h), color: AC.muted },
        { text: "at each level: left vs current, then right vs best", color: AC.muted }
      ] });
    } });
    els.out.html("<b>Sift-down from index " + start + " in a heap of " + n + ".</b> Measured <b>" + cmp + "</b> comparisons and <b>" + sw + "</b> swaps. "
      + "Height of the start node, measured by the longest downward walk: <b>" + hs + "</b>, so the bound is 2 × " + hs + " = " + (2 * hs) + " comparisons and " + hs + " swaps: "
      + HQ.verdict(cmp <= 2 * hs && sw <= hs, "within bound", "EXCEEDS BOUND") + ". "
      + "The root-level bound 2·floor(log₂ " + n + ") = " + (2 * h) + " also holds: " + HQ.verdict(cmp <= 2 * h, "yes", "NO") + ". "
      + "Heap property re-checked everywhere afterwards: " + HQ.verdict(valid, "holds", "VIOLATED") + ". "
      + (sw < hs ? "The value stopped early — " + sw + " swaps against a possible " + hs + " — and the comparisons at the stopping node were the ones that cost without moving anything." :
         "The value fell the full height: " + sw + " swaps in " + hs + " levels, and " + (cmp < 2 * sw ? "one of the levels had a single child, costing one comparison instead of two." : "every level visited had two children."))
      + " Note the comparison count is <i>not</i> exactly twice the swap count: a node with one child costs one comparison, and the stopping node costs comparisons but no swap.");
  }
  if (!els.mode.empty()) els.mode.on("input change", build);
  build();
})();

/* ═══════════════════════════════════════════════════════════════════════
   FIGURE 05 — #build-svg  bottom-up build-heap, stepped
   ═══════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#build-svg"); if (svg.empty()) return;
  const W = 760, H = 340;
  const els = { input: d3.select("#build-input"), out: d3.select("#build-readout") };

  /* the Σ over heights bound, computed from the tree itself (a second, independent route) */
  function sumBound(n) {
    const hF = HQ.floorLog2(n);
    let s = 0;
    for (let h = 1; h <= hF; h++) s += Math.ceil(n / Math.pow(2, h + 1)) * 2 * h;
    return s;
  }
  function build() {
    const which = els.input.empty() ? "classic" : els.input.property("value");
    let a;
    if (which === "classic") a = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7];
    else if (which === "sorted") a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    else if (which === "reverse") a = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    else { const r = AL.rng(2024); a = AL.perm(15, r); }
    const n = a.length;
    const c = AL.counter();
    const frames = [{ arr: a.slice(), focus: -1, cmp: null, swap: null, size: n, cmpN: 0, swapN: 0,
                      note: "start: leaves " + Math.floor(n / 2) + " … " + (n - 1) + " are already one-node heaps; first internal node is " + (Math.floor(n / 2) - 1) }];
    HP.build(a, c, HP.MAX, (fr) => { fr.cmpN = c.get("cmp"); fr.swapN = c.get("swap"); frames.push(fr); });
    frames.push({ arr: a.slice(), focus: -1, cmp: null, swap: null, size: n, cmpN: c.get("cmp"), swapN: c.get("swap"), note: "done — a max-heap" });
    const cmp = c.get("cmp"), sw = c.get("swap");
    const bound = sumBound(n), twoN = 2 * n, nlg = HQ.nlogn(n);
    let valid = true; for (let i = 1; i < n; i++) if (a[HP.parent(i)] < a[i]) valid = false;
    // independent re-measure: run build again on a fresh copy of the same input with a fresh counter
    const src = which === "classic" ? [4, 1, 3, 2, 16, 9, 10, 14, 8, 7] : (which === "sorted" ? Array.from({ length: 15 }, (_, i) => i + 1)
              : (which === "reverse" ? Array.from({ length: 15 }, (_, i) => 15 - i) : AL.perm(15, AL.rng(2024))));
    const c2 = AL.counter(); HP.build(src.slice(), c2);
    HQ.clearControls(svg.node());
    AL.stepper(svg, { frames, delay: 800, label: "step", render: (fr) => {
      const inH = fr.cmpN;
      HD.stepFrame(svg, W, H, fr, { size: n, treeY: 100, panelX: 470, panelY: 112, treeW: 440, levelH: n > 10 ? 46 : 50, r: n > 10 ? 13 : 15, lines: [
        { text: "bottom-up build, n = " + n, color: AC.ink, bold: true },
        { text: "internal nodes: " + (Math.floor(n / 2) - 1) + " down to 0", color: AC.muted },
        { text: "comparisons so far: " + inH, color: AC.a2 },
        { text: "swaps so far: " + fr.swapN, color: AC.rose },
        { text: "Σ ceil(n/2^(h+1))·2h bound: " + bound, color: AC.muted },
        { text: "2n = " + twoN + "     n·log₂n = " + HQ.sig(nlg, 1), color: AC.muted },
        { text: "final total: " + cmp + " comparisons, " + sw + " swaps", color: cmp <= bound ? AC.good : AC.bad }
      ] });
    } });
    els.out.html("<b>Input: " + which + " (n = " + n + ").</b> Measured <b>" + cmp + "</b> comparisons and <b>" + sw + "</b> swaps for the whole construction; a second run on a fresh copy with a fresh counter gives "
      + c2.get("cmp") + " / " + c2.get("swap") + " — " + HQ.verdict(c2.get("cmp") === cmp && c2.get("swap") === sw) + ". "
      + "The per-height bound Σ ceil(n/2^(h+1))·2h = <b>" + bound + "</b>: " + HQ.verdict(cmp <= bound, "within bound", "EXCEEDED") + "; "
      + "2n = " + twoN + ": " + HQ.verdict(cmp <= twoN + 2, "within", "EXCEEDED") + "; n·log₂n = " + HQ.sig(nlg, 1) + ", which the measured count sits " + (cmp < nlg ? "well below" : "at or above") + ". "
      + "Heap property re-checked at every index: " + HQ.verdict(valid, "holds", "VIOLATED") + ". "
      + (which === "reverse" ? "Descending input is already a max-heap: every internal node beats its children on the first look, so no swaps at all — but the comparisons are still paid." :
         which === "sorted" ? "Ascending input is the worst shape for a max-heap build: every internal node is smaller than everything beneath it and falls the full height of its subtree, so the swap count equals the sum of subtree heights." : ""));
  }
  if (!els.input.empty()) els.input.on("input change", build);
  build();
})();

/* ═══════════════════════════════════════════════════════════════════════
   FIGURE 06 — #bsweep-svg  build cost vs n: bottom-up vs repeated insertion
   ═══════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#bsweep-svg"); if (svg.empty()) return;
  const W = 760, H = 380;
  const els = { what: d3.select("#bsweep-what"), trials: d3.select("#bsweep-trials"), tOut: d3.select("#bsweep-trials-out"),
                out: d3.select("#bsweep-readout") };
  const NS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];

  function measure(trials) {
    const r = AL.rng(99);
    const rows = [];
    for (const n of NS) {
      let bu = 0, ins = 0, buS = 0, insS = 0;
      for (let t = 0; t < trials; t++) {
        const src = AL.perm(n, r);
        const c1 = AL.counter(); HP.build(src.slice(), c1); bu += c1.get("cmp"); buS += c1.get("swap");
        const c2 = AL.counter(); HP.buildByInsert(src, c2); ins += c2.get("cmp"); insS += c2.get("swap");
      }
      // worst shapes for repeated insertion into a MAX-heap: ascending input (every key climbs to the root)
      const asc = Array.from({ length: n }, (_, i) => i + 1);
      const c3 = AL.counter(); HP.buildByInsert(asc, c3);
      const c4 = AL.counter(); HP.build(asc.slice(), c4);           // ascending is also bottom-up's worst shape
      // exact worst case for repeated insertion: Σ depth(i) = Σ floor(log2(i+1)) for i = 0 … n−1
      let sumDepth = 0; for (let i = 0; i < n; i++) sumDepth += HQ.floorLog2(i + 1);
      rows.push({ n, bu: bu / trials, buS: buS / trials, ins: ins / trials, insS: insS / trials,
                  insWorst: c3.get("cmp"), insWorstSwap: c3.get("swap"), buWorst: c4.get("cmp"), sumDepth });
    }
    return rows;
  }

  function draw() {
    const what = els.what.empty() ? "cmp" : els.what.property("value");
    const trials = +els.trials.property("value");
    els.tOut.text(trials);
    const rows = measure(trials);
    const perN = what === "cmp";
    const f = AL.frame(svg, W, H, { l: 56, r: 16, t: 18, b: 40 });
    const g = f.g;
    const x = d3.scaleLog().domain([NS[0], NS[NS.length - 1]]).range([0, f.iw]);
    const series = perN
      ? [ { key: "insWorst", label: "repeated insertion, ascending input (worst)", color: AC.bad, val: d => d.insWorst / d.n },
          { key: "ins",      label: "repeated insertion, random input (mean of trials)", color: AC.a2, val: d => d.ins / d.n },
          { key: "buWorst",  label: "bottom-up, ascending input (a worst shape for swaps)", color: AC.violet, val: d => d.buWorst / d.n },
          { key: "bu",       label: "bottom-up, random input (mean of trials)", color: AC.accent, val: d => d.bu / d.n } ]
      : [ { key: "insWorstSwap", label: "repeated insertion swaps, ascending", color: AC.bad, val: d => d.insWorstSwap / d.n },
          { key: "insS", label: "repeated insertion swaps, random", color: AC.a2, val: d => d.insS / d.n },
          { key: "buS",  label: "bottom-up swaps, random", color: AC.accent, val: d => d.buS / d.n } ];
    const ymax = Math.max(...series.map(s => Math.max(...rows.map(s.val)))) * 1.08;
    const y = d3.scaleLinear().domain([0, ymax]).range([f.ih, 0]);
    AL.gridY(g, y, f.iw, 6);
    AL.axisB(g, x, 10, "n (log scale)", d3.format("~s"));
    AL.axisL(g, y, 6, perN ? "comparisons per element (total ÷ n)" : "swaps per element");
    // reference lines
    const refs = perN
      ? [ { label: "log₂ n − 1  (≈ the ascending worst, per element)", fn: n => AL.log2(n) - 1, color: AC.bad, dash: "3,4" },
          { label: "2  (the 2n bound, per element)", fn: () => 2, color: AC.accent, dash: "3,4" },
          { label: "1.88  (known average for bottom-up on random input)", fn: () => 1.88, color: AC.teal, dash: "2,3" },
          { label: "2.28  (known average for repeated insertion on random input)", fn: () => 2.28, color: AC.a2, dash: "2,3" } ]
      : [ { label: "1.28  (known average swaps for repeated insertion)", fn: () => 1.28, color: AC.a2, dash: "2,3" } ];
    refs.forEach(rf => {
      const line = d3.line().x(d => x(d)).y(d => y(Math.min(ymax, rf.fn(d))));
      g.append("path").datum(AL.linspace(NS[0], NS[NS.length - 1], 200)).attr("d", line)
        .attr("fill", "none").attr("stroke", rf.color).attr("stroke-width", 1).attr("stroke-dasharray", rf.dash).attr("opacity", 0.7);
    });
    series.forEach(s => {
      const line = d3.line().x(d => x(d.n)).y(d => y(s.val(d)));
      g.append("path").datum(rows).attr("d", line).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", 2.2);
      g.selectAll(null).data(rows).join("circle").attr("cx", d => x(d.n)).attr("cy", d => y(s.val(d))).attr("r", 3).attr("fill", s.color);
    });
    AL.legend(g, series.map(s => ({ label: s.label, color: s.color })).concat(refs.map(r => ({ label: r.label, color: r.color, dash: r.dash }))), 12, 14, { gap: 15 });

    const last = rows[rows.length - 1];
    const worstOk = last.insWorst === last.sumDepth;   // exact closed form for the ascending worst case comparisons
    const L = HQ.floorLog2(last.n);
    const closed = (last.n + 1) * L - Math.pow(2, L + 1) + 2;   // (n+1)·floor(log2 n) − 2^(floor(log2 n)+1) + 2, a third route
    els.out.html("<b>" + (perN ? "Comparisons" : "Swaps") + " per element, measured by running both constructions on the same inputs, " + trials + " random permutation" + (trials > 1 ? "s" : "") + " per n.</b> "
      + "At n = " + HQ.int(last.n) + ": bottom-up <b>" + HQ.sig(last.bu / last.n, 3) + "</b> comparisons/element (2n bound: " + HQ.verdict(last.bu <= 2 * last.n, "within", "EXCEEDED") + "; known average 1.88), "
      + "repeated insertion on random input <b>" + HQ.sig(last.ins / last.n, 3) + "</b> (known average ≈ 2.28, swaps " + HQ.sig(last.insS / last.n, 3) + " vs ≈ 1.28), "
      + "repeated insertion on ascending input <b>" + HQ.sig(last.insWorst / last.n, 2) + "</b> = log₂ n − O(1). "
      + "The ascending worst case has an exact closed form Σ floor(log₂(i+1)) = " + HQ.int(last.sumDepth) + " = (n+1)·floor(log₂n) − 2^(floor(log₂n)+1) + 2 = " + HQ.int(closed) + " comparisons; measured " + HQ.int(last.insWorst) + " — " + HQ.verdict(worstOk && closed === last.insWorst) + ". "
      + "Bottom-up on the same ascending input: " + HQ.sig(last.buWorst / last.n, 3) + " per element — still under 2. "
      + "<b>Read the shapes, not just the heights:</b> three curves are flat, one grows like log n. Repeated insertion is linear <i>on average</i> for random input and <span class='keep'>Θ</span>(n log n) on sorted input; bottom-up is linear in every case.");
  }
  [els.what, els.trials].forEach(s => { if (!s.empty()) s.on("input change", draw); });
  draw();
})();

/* ═══════════════════════════════════════════════════════════════════════
   FIGURE 07 — #hs-svg  heapsort stepped, heap / sorted boundary
   ═══════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#hs-svg"); if (svg.empty()) return;
  const W = 760, H = 360;
  const els = { input: d3.select("#hs-input"), out: d3.select("#hs-readout") };

  function build() {
    const which = els.input.empty() ? "classic" : els.input.property("value");
    let a;
    if (which === "classic") a = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7];
    else if (which === "sorted") a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    else if (which === "equal") a = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5];
    else a = AL.perm(12, AL.rng(31));
    const n = a.length;
    const c = AL.counter();
    const frames = [{ arr: a.slice(), focus: -1, cmp: null, swap: null, size: n, phase: "build", cmpN: 0, swapN: 0, note: "phase 1: build a max-heap bottom-up" }];
    let buildCmp = 0, buildSwap = 0, seenSort = false;
    HP.heapsort(a, c, (fr) => {
      if (fr.phase === "sort" && !seenSort) { seenSort = true; buildCmp = c.get("cmp"); buildSwap = c.get("swap"); }
      fr.cmpN = c.get("cmp"); fr.swapN = c.get("swap"); frames.push(fr);
    });
    if (!seenSort) { buildCmp = c.get("cmp"); buildSwap = c.get("swap"); }
    frames.push({ arr: a.slice(), focus: -1, cmp: null, swap: null, size: 0, phase: "done", cmpN: c.get("cmp"), swapN: c.get("swap"), note: "done — sorted ascending, in place" });
    const cmp = c.get("cmp"), sw = c.get("swap"), sortCmp = cmp - buildCmp;
    const sortedOk = a.every((v, i) => i === 0 || a[i - 1] <= v);
    const nlg = HQ.nlogn(n);
    // an independent bound on the sort phase: Σ over k = n−1 … 1 of 2·floor(log2 k)
    let sortBound = 0; for (let k = n - 1; k >= 1; k--) sortBound += 2 * HQ.floorLog2(k);
    HQ.clearControls(svg.node());
    AL.stepper(svg, { frames, delay: 700, label: "step", render: (fr) => {
      const sortedFrom = fr.phase === "build" ? n : (fr.phase === "done" ? 0 : fr.size);
      HD.stepFrame(svg, W, H, fr, { size: fr.phase === "done" ? 0 : fr.size, sorted: fr.phase === "build" ? null : sortedFrom, treeY: 100, panelX: 470, panelY: 112,
        treeW: 440, levelH: 48, r: n > 10 ? 13 : 15, lines: [
        { text: "heapsort, n = " + n + "  — phase: " + (fr.phase === "build" ? "build" : (fr.phase === "sort" ? "sort (heap size " + fr.size + ")" : "done")), color: AC.ink, bold: true },
        { text: "comparisons so far: " + fr.cmpN, color: AC.a2 },
        { text: "swaps so far: " + fr.swapN, color: AC.rose },
        { text: "build phase cost: " + buildCmp + " cmp, " + buildSwap + " swaps  (≤ 2n = " + (2 * n) + ")", color: AC.muted },
        { text: "sort phase bound: Σ 2·floor(log₂ k) = " + sortBound, color: AC.muted },
        { text: "n·log₂n = " + HQ.sig(nlg, 1) + "     2·n·log₂n = " + HQ.sig(2 * nlg, 1), color: AC.muted },
        { text: "green = sorted suffix, fixed forever", color: AC.good }
      ] });
    } });
    els.out.html("<b>Input: " + which + " (n = " + n + ").</b> Total <b>" + cmp + "</b> comparisons and <b>" + sw + "</b> swaps: build phase " + buildCmp + " / " + buildSwap + ", sort phase " + sortCmp + " / " + (sw - buildSwap) + ". "
      + "Build ≤ 2n = " + (2 * n) + ": " + HQ.verdict(buildCmp <= 2 * n, "within", "EXCEEDED") + "; sort phase ≤ Σ 2·floor(log₂ k) = " + sortBound + ": " + HQ.verdict(sortCmp <= sortBound, "within", "EXCEEDED") + ". "
      + "Against n·log₂n = " + HQ.sig(nlg, 1) + " the total is <b>" + HQ.sig(cmp / nlg, 2) + "×</b>. Output checked ascending: " + HQ.verdict(sortedOk, "sorted", "NOT SORTED") + ". "
      + (which === "sorted" ? "Already-sorted input is <i>not</i> cheap: the build must invert the whole array into a max-heap and the sort phase then pays its full log per extraction. Heapsort is not adaptive." :
         which === "equal" ? "All keys equal: every sift-down stops at the first comparison because the strict test never fires, so the sort phase costs 2 comparisons per extraction and the whole run is linear — the one input shape that is cheap." : ""));
  }
  if (!els.input.empty()) els.input.on("input change", build);
  build();
})();

/* ═══════════════════════════════════════════════════════════════════════
   FIGURE 08 — #hsweep-svg  heapsort comparisons vs n, four shapes + bottom-up variant
   ═══════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#hsweep-svg"); if (svg.empty()) return;
  const W = 760, H = 380;
  const els = { show: d3.select("#hsweep-show"), out: d3.select("#hsweep-readout") };
  const NS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];

  function measure() {
    const r = AL.rng(4242);
    return NS.map(n => {
      const rand = AL.perm(n, r);
      const asc = Array.from({ length: n }, (_, i) => i + 1);
      const desc = asc.slice().reverse();
      const eq = new Array(n).fill(7);
      const run = (arr, fn) => { const c = AL.counter(); const out = fn(arr.slice(), c); return { cmp: c.get("cmp"), ok: out.every((v, i) => i === 0 || out[i - 1] <= v) }; };
      const o = {
        n,
        rand: run(rand, HP.heapsort), asc: run(asc, HP.heapsort), desc: run(desc, HP.heapsort), eq: run(eq, HP.heapsort),
        buRand: run(rand, HP.heapsortBottomUp), buAsc: run(asc, HP.heapsortBottomUp)
      };
      // a second measurement of the random case on a fresh permutation, to show the spread is small
      o.rand2 = run(AL.perm(n, r), HP.heapsort);
      return o;
    });
  }
  const rows = measure();

  function draw() {
    const show = els.show.empty() ? "classic" : els.show.property("value");
    const f = AL.frame(svg, W, H, { l: 56, r: 16, t: 18, b: 40 });
    const g = f.g;
    const x = d3.scaleLog().domain([NS[0], NS[NS.length - 1]]).range([0, f.iw]);
    const per = (d, k) => d[k].cmp / HQ.nlogn(d.n);
    const series = show === "classic"
      ? [ { k: "rand", label: "classic heapsort, random permutation", color: AC.accent },
          { k: "asc",  label: "classic, already sorted ascending", color: AC.a2 },
          { k: "desc", label: "classic, sorted descending", color: AC.violet },
          { k: "eq",   label: "classic, all keys equal", color: AC.teal } ]
      : [ { k: "rand",   label: "classic heapsort, random", color: AC.accent },
          { k: "buRand", label: "bottom-up variant, random", color: AC.good },
          { k: "buAsc",  label: "bottom-up variant, ascending", color: AC.rose } ];
    const ymax = Math.max(2.3, ...series.map(s => Math.max(...rows.map(d => per(d, s.k))))) * 1.06;
    const y = d3.scaleLinear().domain([0, ymax]).range([f.ih, 0]);
    AL.gridY(g, y, f.iw, 6);
    AL.axisB(g, x, 10, "n (log scale)", d3.format("~s"));
    AL.axisL(g, y, 6, "comparisons ÷ (n·log₂n)");
    [{ v: 2, label: "2  — i.e. 2·n·log₂n", color: AC.muted }, { v: 1, label: "1  — i.e. n·log₂n, the information-theoretic scale", color: AC.muted }].forEach(rf => {
      g.append("line").attr("x1", 0).attr("x2", f.iw).attr("y1", y(rf.v)).attr("y2", y(rf.v)).attr("stroke", rf.color).attr("stroke-dasharray", "3,4");
      g.append("text").attr("x", f.iw - 4).attr("y", y(rf.v) - 4).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.muted).text(rf.label);
    });
    series.forEach(s => {
      const line = d3.line().x(d => x(d.n)).y(d => y(per(d, s.k)));
      g.append("path").datum(rows).attr("d", line).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", 2.2);
      g.selectAll(null).data(rows).join("circle").attr("cx", d => x(d.n)).attr("cy", d => y(per(d, s.k))).attr("r", 3).attr("fill", s.color);
    });
    AL.legend(g, series.map(s => ({ label: s.label, color: s.color })), 12, 14, { gap: 15 });
    const last = rows[rows.length - 1];
    const allSorted = rows.every(d => d.rand.ok && d.asc.ok && d.desc.ok && d.eq.ok && d.buRand.ok && d.buAsc.ok);
    els.out.html("<b>Comparisons divided by n·log₂n, measured at each n; every output re-checked as sorted: " + HQ.verdict(allSorted, "all sorted", "A RUN FAILED") + ".</b> "
      + "At n = " + HQ.int(last.n) + ", classic heapsort: random <b>" + HQ.sig(per(last, "rand"), 3) + "</b> (a second random permutation: " + HQ.sig(per(last, "rand2"), 3) + "), "
      + "ascending <b>" + HQ.sig(per(last, "asc"), 3) + "</b>, descending <b>" + HQ.sig(per(last, "desc"), 3) + "</b>, all-equal <b>" + HQ.sig(per(last, "eq"), 3) + "</b> "
      + "(that last one is " + HQ.int(last.eq.cmp) + " comparisons total, about " + HQ.sig(last.eq.cmp / last.n, 2) + " per element — linear). "
      + "Bottom-up variant: random <b>" + HQ.sig(per(last, "buRand"), 3) + "</b>, ascending <b>" + HQ.sig(per(last, "buAsc"), 3) + "</b>. "
      + "Sorted input costs classic heapsort essentially the same as random input — <b>" + HQ.sig(last.asc.cmp / last.rand.cmp, 2) + "×</b> — which is what non-adaptive means; "
      + "the bottom-up variant cuts the random-input count by <b>" + HQ.sig(100 * (1 - last.buRand.cmp / last.rand.cmp), 0) + "%</b>, from about 2 to about 1 per element per level, at the price of extra data movement the comparison count does not show.");
  }
  if (!els.show.empty()) els.show.on("input change", draw);
  draw();
})();

/* ═══════════════════════════════════════════════════════════════════════
   FIGURE 09 — #handle-svg  decreaseKey / delete through a handle (index heap)
   ═══════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#handle-svg"); if (svg.empty()) return;
  const W = 760, H = 360;
  const els = { op: d3.select("#handle-op"), out: d3.select("#handle-readout") };

  /* an indexed MIN-heap: key[], item[] (the handle names) and pos[item] → index.
     Every move updates pos, which is the whole point of the figure. */
  function makeHeap(keys, names) {
    const h = { key: keys.slice(), item: names.slice(), pos: {} };
    h.item.forEach((it, i) => { h.pos[it] = i; });
    return h;
  }
  function swapH(h, i, j, c) {
    [h.key[i], h.key[j]] = [h.key[j], h.key[i]];
    [h.item[i], h.item[j]] = [h.item[j], h.item[i]];
    h.pos[h.item[i]] = i; h.pos[h.item[j]] = j; c.add("swap"); c.add("posWrites", 2);
  }
  function up(h, i, c, rec) {
    while (i > 0) {
      const p = HP.parent(i); c.add("cmp");
      if (h.key[i] < h.key[p]) { rec({ focus: i, cmp: [i, p], swap: [i, p], note: "key " + h.key[i] + " at " + i + " is below its parent " + h.key[p] + " → swap, update pos" }); swapH(h, i, p, c); i = p; }
      else { rec({ focus: i, cmp: [i, p], swap: null, note: "key " + h.key[i] + " ≥ parent " + h.key[p] + " → stop" }); return; }
    }
    rec({ focus: 0, cmp: null, swap: null, note: "at the root → stop" });
  }
  function down(h, i, n, c, rec) {
    for (;;) {
      const l = HP.left(i), r = HP.right(i); let best = i; const cm = [];
      if (l < n) { c.add("cmp"); cm.push([l, best]); if (h.key[l] < h.key[best]) best = l; }
      if (r < n) { c.add("cmp"); cm.push([r, best]); if (h.key[r] < h.key[best]) best = r; }
      if (best === i) { rec({ focus: i, cmp: cm, swap: null, note: l < n ? "key " + h.key[i] + " already below its children → stop" : "leaf → stop" }); return; }
      rec({ focus: i, cmp: cm, swap: [i, best], note: "smaller child " + h.key[best] + " beats " + h.key[i] + " → swap, update pos" });
      swapH(h, i, best, c); i = best;
    }
  }

  function build() {
    const op = els.op.empty() ? "dec12" : els.op.property("value");
    let h, frames = [], c = AL.counter(), n, desc, target, newKey;
    const snap = (extra) => Object.assign({ key: h.key.slice(0, n), item: h.item.slice(0, n), pos: Object.assign({}, h.pos), n: n,
                                             cmpN: c.get("cmp"), swapN: c.get("swap"), posW: c.get("posWrites") }, extra);
    if (op === "dec12" || op === "dec8") {
      h = makeHeap([2, 5, 3, 9, 6, 7, 4, 10, 12, 8], ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]); n = 10;
      target = op === "dec12" ? "I" : "H"; newKey = op === "dec12" ? 1 : 6;
      desc = "decreaseKey(handle " + target + ", " + newKey + ")";
      frames.push(snap({ focus: -1, cmp: null, swap: null, note: "min-heap with handles A … J; pos[" + target + "] = " + h.pos[target] }));
      const i = h.pos[target]; c.add("lookup");
      frames.push(snap({ focus: i, cmp: null, swap: null, note: "pos[" + target + "] → index " + i + " in O(1); overwrite key " + h.key[i] + " → " + newKey }));
      h.key[i] = newKey;
      frames.push(snap({ focus: i, cmp: null, swap: null, note: "the only possible violation is the edge above " + i + " → sift-up" }));
      up(h, i, c, f => frames.push(snap(f)));
    } else {
      h = makeHeap([1, 20, 2, 25, 30, 3, 4], ["A", "B", "C", "D", "E", "F", "G"]); n = 7;
      target = "D"; desc = "delete(handle D)  — the key 25 at index 3";
      frames.push(snap({ focus: -1, cmp: null, swap: null, note: "min-heap with handles A … G; pos[D] = " + h.pos.D }));
      const i = h.pos.D; c.add("lookup");
      frames.push(snap({ focus: i, cmp: null, swap: [i, n - 1], note: "pos[D] → index " + i + "; move the last entry (key " + h.key[n - 1] + ", handle " + h.item[n - 1] + ") into it, shrink" }));
      h.key[i] = h.key[n - 1]; h.item[i] = h.item[n - 1]; h.pos[h.item[i]] = i; delete h.pos.D; n -= 1; c.add("posWrites");
      frames.push(snap({ focus: i, cmp: null, swap: null, note: "key " + h.key[i] + " at index " + i + ": compare with parent " + h.key[HP.parent(i)] + " — it is smaller, so this delete needs SIFT-UP, not sift-down" }));
      c.add("cmp");
      if (h.key[i] < h.key[HP.parent(i)]) up(h, i, c, f => frames.push(snap(f))); else down(h, i, n, c, f => frames.push(snap(f)));
    }
    frames.push(snap({ focus: -1, cmp: null, swap: null, note: "done" }));
    // independent checks: heap property, pos consistency (pos[item[i]] === i for all i, and every live handle present)
    let valid = true; for (let i = 1; i < n; i++) if (h.key[HP.parent(i)] > h.key[i]) valid = false;
    let posOk = true; for (let i = 0; i < n; i++) if (h.pos[h.item[i]] !== i) posOk = false;
    if (Object.keys(h.pos).length !== n) posOk = false;
    const cmp = c.get("cmp"), sw = c.get("swap");
    HQ.clearControls(svg.node());
    AL.stepper(svg, { frames, delay: 900, label: "step", render: (fr) => {
      const f = AL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 8 });
      const g = f.g;
      const cmpSet = new Set(); (fr.cmp || []).forEach(p => p.forEach(i => cmpSet.add(i)));
      const swapSet = new Set(fr.swap || []);
      const mark = i => swapSet.has(i) ? AC.rose : (i === fr.focus ? AC.accent : (cmpSet.has(i) ? AC.a2 : null));
      g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text("heap array: key (handle)");
      AL.row(g, fr.key.map((k, i) => k + " " + fr.item[i]), { x: 0, y: 2, w: 42, gap: 3, mark: mark, fontSize: 11 });
      g.append("text").attr("x", 0).attr("y", 66).attr("font-size", 11).attr("fill", AC.muted).text("pos: handle → index (the inverse map every move must update)");
      const names = Object.keys(fr.pos).sort();
      const pr = AL.row(g, names.map(nm => fr.pos[nm]), { x: 0, y: 72, w: 30, gap: 3, index: false,
        mark: (k) => { const nm = names[k]; const idx = fr.pos[nm]; return swapSet.has(idx) ? AC.rose : (idx === fr.focus ? AC.accent : null); } });
      names.forEach((nm, k) => g.append("text").attr("x", pr.cellX(k)).attr("y", 72 + 30 + 12).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.muted).text(nm));
      const tree = AL.binTree(g, fr.key, { x: 0, y: 130, w: 430, levelH: 48, r: 15, mark: mark });
      fr.key.forEach((_, i) => { const q = tree.pos(i); g.append("text").attr("x", q.x + 17).attr("y", q.y - 9).attr("font-size", 9.5).attr("fill", AC.muted).text(fr.item[i]); });
      g.append("text").attr("x", 0).attr("y", 130 + tree.height + 14).attr("font-size", 11.5).attr("fill", AC.ink).text(fr.note);
      HD.counts(g, 470, 140, [
        { text: desc, color: AC.ink, bold: true },
        { text: "handle lookups: " + c.get("lookup") + "  (O(1) each)", color: AC.muted },
        { text: "comparisons so far: " + fr.cmpN, color: AC.a2 },
        { text: "swaps so far: " + fr.swapN, color: AC.rose },
        { text: "pos[] writes so far: " + fr.posW, color: AC.teal },
        { text: "bound: floor(log₂ " + fr.n + ") = " + HQ.floorLog2(fr.n) + " swaps", color: AC.muted }
      ]);
    } });
    els.out.html("<b>" + desc + ".</b> Handle lookup was <b>O(1)</b> (one map read); the repair then cost <b>" + cmp + "</b> comparisons and <b>" + sw + "</b> swaps, each swap writing two entries of pos[] (" + c.get("posWrites") + " pos writes in all). "
      + "Bound floor(log₂ " + n + ") = " + HQ.floorLog2(n) + " swaps: " + HQ.verdict(sw <= HQ.floorLog2(n), "within", "EXCEEDED") + ". "
      + "Afterwards, heap property at every index: " + HQ.verdict(valid, "holds", "VIOLATED") + "; pos[] consistent (pos[item[i]] = i for all i, and exactly " + n + " live handles): " + HQ.verdict(posOk, "consistent", "STALE") + ". "
      + (op === "dec12" ? "The key fell from 12 to 1 and climbed three levels to the root — the maximum possible for index 8 — and handle I now names index 0. Without pos[], finding the entry for I would have been a linear scan." :
         op === "dec8" ? "The decrease stops after one swap: 6 is below its old parent 9 (index 3) but not below its new parent 5 (index 1), and the rest of the path is untouched — the common case for a small decrease." :
         "The moved-in key 4 was <i>smaller</i> than the parent 20 of the slot it landed in, so the correct repair was sift-up; an implementation that always sifts down after a delete would have left 4 sitting under 20 and reported success."));
  }
  if (!els.op.empty()) els.op.on("input change", build);
  build();
})();

/* ═══════════════════════════════════════════════════════════════════════
   FIGURE 10 — #dary-svg  d-ary heaps: per-operation cost as d grows
   ═══════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#dary-svg"); if (svg.empty()) return;
  const W = 760, H = 360;
  const els = { ratio: d3.select("#dary-ratio"), rOut: d3.select("#dary-ratio-out"), work: d3.select("#dary-work"), out: d3.select("#dary-readout") };
  const DS = [2, 3, 4, 6, 8, 12, 16];
  const N = 4096;

  /* Two workloads. "random": random insertion order, key changes by random amounts on
     random entries — the typical case, where a climb is short whatever d is.
     "climb": ascending insertion order (every insert climbs to the root) and every
     key change raises the key above everything (climbs to the root) — the shape the
     worst-case bounds log_d n and d·log_d n describe. */
  function measure(ratio, work) {
    const r = AL.rng(777);
    const keys = work === "climb" ? Array.from({ length: N }, (_, i) => i + 1) : AL.perm(N, r);
    const picks = Array.from({ length: N * ratio }, () => Math.floor(r() * N));
    const bumps = picks.map(() => Math.floor(r() * N));
    return DS.map(d => {
      const Hd = HP.dary(d);
      const a = [];
      const cI = AL.counter(); keys.forEach(k => Hd.insert(a, k, cI));
      const cK = AL.counter();
      let top = N;
      for (let t = 0; t < picks.length; t++) {
        const i = picks[t] % a.length;
        if (work === "climb") { top += 1; Hd.increaseKey(a, i, top, cK); }      // above everything: climbs to the root
        else Hd.increaseKey(a, i, a[i] + bumps[t], cK);
      }
      const cX = AL.counter(); const out = []; while (a.length) out.push(Hd.extract(a, cX));
      const sorted = out.every((v, i) => i === 0 || out[i - 1] >= v);
      return { d, ins: cI.get("cmp") / N, key: picks.length ? cK.get("cmp") / picks.length : 0, ext: cX.get("cmp") / N,
               total: cI.get("cmp") + cK.get("cmp") + cX.get("cmp"), sorted };
    });
  }

  function draw() {
    const ratio = +els.ratio.property("value");
    const work = els.work.empty() ? "random" : els.work.property("value");
    els.rOut.text(ratio);
    const rows = measure(ratio, work);
    const f = AL.frame(svg, W, H, { l: 56, r: 16, t: 18, b: 40 });
    const g = f.g;
    const x = d3.scalePoint().domain(DS).range([0, f.iw]).padding(0.5);
    const series = [ { k: "ins", label: "insert — comparisons per operation", color: AC.accent },
                     { k: "key", label: "increaseKey (the decreaseKey analogue) — per operation", color: AC.teal },
                     { k: "ext", label: "extract — comparisons per operation", color: AC.rose } ];
    const ymax = Math.max(48, ...rows.map(d => Math.max(d.ins, d.key, d.ext))) * 1.1;
    const y = d3.scaleLinear().domain([0, ymax]).range([f.ih, 0]);
    AL.gridY(g, y, f.iw, 6);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${f.ih})`).call(d3.axisBottom(x));
    g.append("text").attr("x", f.iw).attr("y", f.ih + 31).attr("text-anchor", "end").attr("font-size", 11).attr("fill", AC.muted).text("d — children per node (n = " + HQ.int(N) + ")");
    AL.axisL(g, y, 6, "comparisons per operation, measured");
    const refs = [ { label: "log_d n  (worst-case insert / decreaseKey)", fn: d => Math.log(N) / Math.log(d), color: AC.accent, dash: "3,4" },
                   { label: "d · log_d n  (worst-case extract)", fn: d => d * Math.log(N) / Math.log(d), color: AC.rose, dash: "3,4" } ];
    refs.forEach(rf => {
      const line = d3.line().x(d => x(d)).y(d => y(Math.min(ymax, rf.fn(d))));
      g.append("path").datum(DS).attr("d", line).attr("fill", "none").attr("stroke", rf.color).attr("stroke-width", 1).attr("stroke-dasharray", rf.dash).attr("opacity", 0.7);
    });
    series.forEach(s => {
      const line = d3.line().x(d => x(d.d)).y(d => y(d[s.k]));
      g.append("path").datum(rows).attr("d", line).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", 2.2);
      g.selectAll(null).data(rows).join("circle").attr("cx", d => x(d.d)).attr("cy", d => y(d[s.k])).attr("r", 3.5).attr("fill", s.color);
    });
    AL.legend(g, series.map(s => ({ label: s.label, color: s.color })).concat(refs.map(r => ({ label: r.label, color: r.color, dash: r.dash }))), 14, 14, { gap: 15 });
    const best = rows.reduce((a, b) => b.total < a.total ? b : a);
    const d2 = rows[0], d4 = rows.find(r => r.d === 4), d16 = rows[rows.length - 1];
    const allSorted = rows.every(r => r.sorted);
    // total-per-element for the readout, d = 2 vs best
    els.out.html("<b>" + (work === "climb" ? "Worst-shape workload" : "Random workload") + ": n = " + HQ.int(N) + " inserts, then " + HQ.int(N * ratio) + " increaseKeys (" + ratio + " per element), then " + HQ.int(N) + " extracts; every d's extraction order re-checked as descending: " + HQ.verdict(allSorted, "all correct", "A RUN FAILED") + ".</b> "
      + "Per operation at d = 2 / 4 / 16 — insert <b>" + HQ.sig(d2.ins, 2) + " / " + HQ.sig(d4.ins, 2) + " / " + HQ.sig(d16.ins, 2) + "</b>, "
      + "increaseKey <b>" + HQ.sig(d2.key, 2) + " / " + HQ.sig(d4.key, 2) + " / " + HQ.sig(d16.key, 2) + "</b>, "
      + "extract <b>" + HQ.sig(d2.ext, 2) + " / " + HQ.sig(d4.ext, 2) + " / " + HQ.sig(d16.ext, 2) + "</b>. "
      + "Total comparisons lowest at <b>d = " + best.d + "</b> (d = 2 costs " + HQ.sig(d2.total / best.total, 2) + "× that). "
      + (work === "climb"
          ? "Here every insert and every key change climbs the full height, so they track log_d n minus a small constant (measured " + HQ.sig(d2.ins, 1) + " at d = 2 against log₂ 4096 = 12; " + HQ.sig(d16.ins, 1) + " at d = 16 against 3) and a wider heap pays as soon as the key changes outnumber the extracts; slide the ratio up and the minimum moves right — this is the E/V knob of Dijkstra's algorithm (§23), and it is a <i>worst-case</i> knob."
          : "Here a random key rarely climbs more than a level or two whatever d is, so insert and increaseKey sit far below log_d n and the extract cost — which really does grow like d·log_d n — decides everything: d = 3 wins at every ratio and the minimum does <i>not</i> move right. Switch to the worst-shape workload to see the textbook trade appear."));
  }
  [els.ratio, els.work].forEach(s => { if (!s.empty()) s.on("input change", draw); });
  draw();
})();

/* ── a small instrumented Fibonacci heap (min), enough for the two figures ── */
const FIB = (function () {
  let idSeq = 0;
  function node(key) { return { id: ++idSeq, key, children: [], parent: null, mark: false }; }
  function make() { return { roots: [], min: null, n: 0 }; }
  function insert(H, key, c) {
    const x = node(key); H.roots.push(x); H.n++; c.add("rootAppend");
    if (!H.min || x.key < H.min.key) H.min = x;
    return x;
  }
  function link(H, y, x, c) {          // make y a child of x (x.key ≤ y.key), remove y from the root list
    H.roots.splice(H.roots.indexOf(y), 1);
    y.parent = x; y.mark = false; x.children.push(y); c.add("link");
  }
  function consolidate(H, c, rec) {
    const A = [];
    const order = H.roots.slice();       // process the roots as they were
    for (const w of order) {
      let x = w; let d = x.children.length;
      c.add("rootVisit");
      if (rec) rec({ note: "visit root " + x.key + " (degree " + d + ")", A: A.slice(), focus: x.id });
      while (A[d]) {
        let y = A[d]; c.add("cmp");
        if (x.key > y.key) { const t = x; x = y; y = t; }
        if (rec) rec({ note: "two roots of degree " + d + ": link " + y.key + " under " + x.key, A: A.slice(), focus: x.id, other: y.id });
        link(H, y, x, c);
        A[d] = null; d++;
      }
      A[d] = x;
    }
    H.min = null;
    for (const r of H.roots) { c.add("cmp"); if (!H.min || r.key < H.min.key) H.min = r; }
    return A;
  }
  function extractMin(H, c, rec) {
    const z = H.min; if (!z) return null;
    for (const ch of z.children) { ch.parent = null; H.roots.push(ch); c.add("rootAppend"); }
    H.roots.splice(H.roots.indexOf(z), 1);
    H.n--;
    if (rec) rec({ note: "remove min " + z.key + "; its " + z.children.length + " children join the root list", A: [], focus: null });
    z.children = [];
    if (H.roots.length) consolidate(H, c, rec); else H.min = null;
    return z;
  }
  function cut(H, x, y, c) {           // remove x from y's child list, make it a root
    y.children.splice(y.children.indexOf(x), 1);
    H.roots.push(x); x.parent = null; x.mark = false; c.add("cut");
  }
  function cascadingCut(H, y, c, rec) {
    const z = y.parent; c.add("cascade");
    if (!z) { if (rec) rec({ note: "cascading-cut(" + y.key + "): it is a root → stop", focus: y.id }); return; }
    if (!y.mark) { y.mark = true; c.add("mark"); if (rec) rec({ note: "cascading-cut(" + y.key + "): unmarked → MARK it (first child lost), stop", focus: y.id }); }
    else { if (rec) rec({ note: "cascading-cut(" + y.key + "): already marked (second child lost) → cut it too, and recurse on " + z.key, focus: y.id }); cut(H, y, z, c); cascadingCut(H, z, c, rec); }
  }
  function decreaseKey(H, x, k, c, rec) {
    x.key = k; const y = x.parent;
    if (y && x.key < y.key) {
      if (rec) rec({ note: "key " + k + " is now below its parent " + y.key + " → CUT it to the root list", focus: x.id });
      cut(H, x, y, c);
      cascadingCut(H, y, c, rec);
    } else if (rec) rec({ note: y ? "still ≥ parent " + y.key + " — nothing to do" : "x is a root — just update min", focus: x.id });
    if (x.key < H.min.key) H.min = x;
  }
  function countTrees(H) { return H.roots.length; }
  function countMarks(H) { let m = 0; const walk = v => { if (v.mark) m++; v.children.forEach(walk); }; H.roots.forEach(walk); return m; }
  function potential(H) { return countTrees(H) + 2 * countMarks(H); }
  function maxDegree(H) { let d = 0; const walk = v => { d = Math.max(d, v.children.length); v.children.forEach(walk); }; H.roots.forEach(walk); return d; }
  function size(v) { return 1 + v.children.reduce((s, ch) => s + size(ch), 0); }
  function clone(H) {                  // deep copy for frames
    const map = new Map();
    const cp = v => { const w = { id: v.id, key: v.key, mark: v.mark, children: [], parent: null }; map.set(v.id, w); v.children.forEach(ch => { const cw = cp(ch); cw.parent = w; w.children.push(cw); }); return w; };
    return { roots: H.roots.map(cp), min: H.min ? map.get(H.min.id) : null, n: H.n };
  }
  return { make, insert, extractMin, consolidate, decreaseKey, cut, cascadingCut, potential, countTrees, countMarks, maxDegree, size, clone, node };
})();

/* forest renderer: each root's subtree laid out by leaf count, roots side by side */
HD.forest = function (g, roots, opt) {
  const o = Object.assign({ x: 0, y: 0, r: 13, levelH: 44, unit: 30, gap: 18, mark: null, minId: null, showDeg: true }, opt || {});
  const width = v => v.children.length ? Math.max(1, v.children.reduce((s, ch) => s + width(ch), 0)) : 1;
  const pos = new Map();
  let cx = 0;
  const place = (v, left, depth) => {
    const w = width(v) * o.unit;
    const x = left + w / 2, y = depth * o.levelH;
    pos.set(v.id, { x, y, v });
    let l = left;
    v.children.forEach(ch => { place(ch, l, depth + 1); l += width(ch) * o.unit; });
  };
  roots.forEach(rt => { place(rt, cx, 0); cx += width(rt) * o.unit + o.gap; });
  const gg = g.append("g").attr("transform", `translate(${o.x},${o.y})`);
  // root list dashed line
  if (roots.length > 1) {
    const first = pos.get(roots[0].id), last = pos.get(roots[roots.length - 1].id);
    gg.append("line").attr("x1", first.x).attr("x2", last.x).attr("y1", first.y).attr("y2", last.y).attr("stroke", AC.muted).attr("stroke-dasharray", "3,4");
  }
  pos.forEach(p => p.v.children.forEach(ch => { const q = pos.get(ch.id);
    gg.append("line").attr("x1", p.x).attr("y1", p.y).attr("x2", q.x).attr("y2", q.y).attr("stroke", AC.line).attr("stroke-width", 1.5); }));
  pos.forEach(p => {
    const fill = o.mark ? (o.mark(p.v) || (p.v.mark ? "#3a3230" : AC.panel2)) : (p.v.mark ? "#3a3230" : AC.panel2);
    gg.append("circle").attr("cx", p.x).attr("cy", p.y).attr("r", o.r).attr("fill", fill)
      .attr("stroke", p.v.mark ? AC.a2 : (o.minId === p.v.id ? AC.good : AC.line)).attr("stroke-width", p.v.mark || o.minId === p.v.id ? 2 : 1);
    gg.append("text").attr("x", p.x).attr("y", p.y + 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.ink).text(p.v.key);
    if (o.showDeg && !p.v.parent) gg.append("text").attr("x", p.x).attr("y", p.y - o.r - 4).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", AC.muted).text("deg " + p.v.children.length);
  });
  return { g: gg, width: cx - o.gap, pos };
};

/* ═══════════════════════════════════════════════════════════════════════
   FIGURE 11 — #cons-svg  Fibonacci heap: extractMin + consolidate, stepped
   ═══════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#cons-svg"); if (svg.empty()) return;
  const W = 760, H = 380;
  const els = { out: d3.select("#cons-readout") };
  const KEYS = [10, 3, 7, 1, 8, 5, 12];
  const c = AL.counter();
  const H0 = FIB.make();
  KEYS.forEach(k => FIB.insert(H0, k, c));
  const frames = [];
  const snap = (extra) => Object.assign({ heap: FIB.clone(H0), t: FIB.countTrees(H0), m: FIB.countMarks(H0), phi: FIB.potential(H0),
                                           links: c.get("link"), visits: c.get("rootVisit"), cmp: c.get("cmp") }, extra);
  frames.push(snap({ note: KEYS.length + " lazy inserts: each is O(1), just appended to the root list; Φ rose by 1 each time", A: [], focus: null }));
  const phiBefore = FIB.potential(H0);
  const cmp0 = c.get("cmp");
  const z = FIB.extractMin(H0, c, (f) => frames.push(snap(f)));
  frames.push(snap({ note: "consolidated: every root now has a distinct degree; new min is " + H0.min.key, A: [], focus: null, done: true }));
  const phiAfter = FIB.potential(H0);
  const actual = c.get("rootVisit") + c.get("link");
  const Dn = Math.floor(Math.log(H0.n) / Math.log((1 + Math.sqrt(5)) / 2));
  // independent checks: degrees distinct among roots; each tree is heap-ordered; sizes are powers of two (no decreaseKey yet ⇒ binomial trees)
  const degs = H0.roots.map(r => r.children.length);
  const distinct = new Set(degs).size === degs.length;
  let ordered = true; const walk = v => { v.children.forEach(ch => { if (ch.key < v.key) ordered = false; walk(ch); }); }; H0.roots.forEach(walk);
  const pow2 = H0.roots.every(r => { const s = FIB.size(r); return (s & (s - 1)) === 0 && s === (1 << r.children.length); });
  const binary = H0.n.toString(2);

  HQ.clearControls(svg.node());
  AL.stepper(svg, { frames, delay: 1000, label: "step", render: (fr) => {
    const f = AL.frame(svg, W, H, { l: 14, r: 14, t: 26, b: 8 });
    const g = f.g;
    g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", AC.muted).text("root list (dashed), min in green, focus in blue, marked nodes amber-ringed");
    HD.forest(g, fr.heap.roots, { x: 8, y: 30, r: 13, levelH: 44, unit: 30, gap: 22, minId: fr.heap.min ? fr.heap.min.id : null,
      mark: v => v.id === fr.focus ? AC.accent : (v.id === fr.other ? AC.rose : null) });
    g.append("text").attr("x", 0).attr("y", 200).attr("font-size", 11.5).attr("fill", AC.ink).text(fr.note);
    // degree table A
    g.append("text").attr("x", 0).attr("y", 236).attr("font-size", 11).attr("fill", AC.muted).text("degree table A[0 … D]: the unique root of each degree seen so far");
    AL.row(g, [0, 1, 2, 3].map(d => fr.A && fr.A[d] ? fr.A[d].key : "·"), { x: 0, y: 244, w: 34, gap: 3 });
    HD.counts(g, 470, 236, [
      { text: "t(H) = " + fr.t + " trees,  m(H) = " + fr.m + " marked", color: AC.ink, bold: true },
      { text: "Φ = t + 2m = " + fr.phi, color: AC.a2 },
      { text: "root visits so far: " + fr.visits, color: AC.muted },
      { text: "links so far: " + fr.links, color: AC.rose },
      { text: "key comparisons so far: " + fr.cmp, color: AC.muted }
    ]);
  } });
  els.out.html("<b>Seven lazy inserts, then one extractMin.</b> Before the extract: t = " + KEYS.length + ", m = 0, <b>Φ = " + phiBefore + "</b>. "
    + "The extract removed " + z.key + " and consolidate then made <b>" + c.get("rootVisit") + "</b> root visits and <b>" + c.get("link") + "</b> links (" + (c.get("cmp") - cmp0) + " key comparisons), "
    + "leaving t = " + FIB.countTrees(H0) + ", <b>Φ = " + phiAfter + "</b>. "
    + "Actual work " + actual + " units; ΔΦ = " + (phiAfter - phiBefore) + "; amortized = actual + ΔΦ = <b>" + (actual + phiAfter - phiBefore) + "</b>, against the O(D(n)) promise with D(" + H0.n + ") ≤ floor(log_φ " + H0.n + ") = " + Dn + " — "
    + HQ.verdict(actual + phiAfter - phiBefore <= 2 * (Dn + 1), "within a small multiple of D(n)", "LARGER THAN EXPECTED") + ". "
    + "Checks: root degrees " + JSON.stringify(degs) + " all distinct — " + HQ.verdict(distinct) + "; every tree min-heap-ordered — " + HQ.verdict(ordered) + "; "
    + "with no decreaseKey yet, every tree of degree k has exactly 2ᵏ nodes (they are binomial trees) — " + HQ.verdict(pow2) + "; and n = " + H0.n + " = " + binary + "₂ has " + degs.length + " one-bits, one per tree — " + HQ.verdict(binary.split("1").length - 1 === degs.length) + ". "
    + "The " + KEYS.length + " inserts each paid 1 unit of potential in advance; the extract spent it.");
})();

/* ═══════════════════════════════════════════════════════════════════════
   FIGURE 12 — #cut-svg  Fibonacci heap: decreaseKey with cascading cuts
   ═══════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#cut-svg"); if (svg.empty()) return;
  const W = 760, H = 380;
  const els = { out: d3.select("#cut-readout") };
  /* build the drawn example by hand:  roots 2 and 20.
       2 → children 3, 40;   3 → children 5*, 9;   5* → children 8*, 15;   8* → child 30;   20 → child 25.   (* = marked) */
  const c = AL.counter();
  const Hh = FIB.make();
  const N = k => FIB.node(k);
  const n2 = N(2), n3 = N(3), n40 = N(40), n5 = N(5), n9 = N(9), n8 = N(8), n15 = N(15), n30 = N(30), n20 = N(20), n25 = N(25);
  const adopt = (p, ...kids) => kids.forEach(k => { k.parent = p; p.children.push(k); });
  adopt(n2, n3, n40); adopt(n3, n5, n9); adopt(n5, n8, n15); adopt(n8, n30); adopt(n20, n25);
  n5.mark = true; n8.mark = true;
  Hh.roots = [n2, n20]; Hh.min = n2; Hh.n = 10;
  const frames = [];
  const snap = (extra) => Object.assign({ heap: FIB.clone(Hh), t: FIB.countTrees(Hh), m: FIB.countMarks(Hh), phi: FIB.potential(Hh),
                                           cuts: c.get("cut"), cascades: c.get("cascade"), marks: c.get("mark") }, extra);
  frames.push(snap({ note: "before: 5 and 8 are marked (each has already lost one child). decreaseKey(30 → 1) is about to run", focus: n30.id }));
  const phiBefore = FIB.potential(Hh);
  FIB.decreaseKey(Hh, n30, 1, c, (f) => frames.push(snap(f)));
  frames.push(snap({ note: "done: 1 is the new min; three cuts made three new roots; 3 is now marked", focus: n30.id, done: true }));
  const phiAfter = FIB.potential(Hh);
  const cc = c.get("cascade"), cuts = c.get("cut");
  const bound = 4 - cc;
  const dPhi = phiAfter - phiBefore;
  let ordered = true; const walk = v => { v.children.forEach(ch => { if (ch.key < v.key) ordered = false; walk(ch); }); }; Hh.roots.forEach(walk);
  const rootsUnmarked = Hh.roots.every(r => !r.mark || r === n3);   // a cut node arrives unmarked

  HQ.clearControls(svg.node());
  AL.stepper(svg, { frames, delay: 1100, label: "step", render: (fr) => {
    const f = AL.frame(svg, W, H, { l: 14, r: 14, t: 26, b: 8 });
    const g = f.g;
    g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", AC.muted).text("marked nodes have an amber ring; the node in focus is blue; min is green-ringed");
    HD.forest(g, fr.heap.roots, { x: 8, y: 30, r: 13, levelH: 42, unit: 30, gap: 22, minId: fr.heap.min ? fr.heap.min.id : null,
      mark: v => v.id === fr.focus ? AC.accent : null });
    g.append("text").attr("x", 0).attr("y", 236).attr("font-size", 11.5).attr("fill", AC.ink).text(fr.note);
    HD.counts(g, 470, 250, [
      { text: "t(H) = " + fr.t + " trees,  m(H) = " + fr.m + " marked", color: AC.ink, bold: true },
      { text: "Φ = t + 2m = " + fr.phi + "   (started at " + phiBefore + ")", color: AC.a2 },
      { text: "cuts so far: " + fr.cuts, color: AC.rose },
      { text: "cascading-cut calls so far: " + fr.cascades, color: AC.muted },
      { text: "marks set so far: " + fr.marks, color: AC.muted }
    ]);
  } });
  els.out.html("<b>decreaseKey(30 → 1) on a ten-node heap with two marked nodes.</b> Measured: <b>" + cuts + "</b> cuts, <b>" + cc + "</b> cascading-cut calls (c = " + cc + "), " + c.get("mark") + " mark set. "
    + "Φ went from " + phiBefore + " (t = 2, m = 2) to " + phiAfter + " (t = " + FIB.countTrees(Hh) + ", m = " + FIB.countMarks(Hh) + "): <b>ΔΦ = " + (dPhi >= 0 ? "+" : "") + dPhi + "</b>. "
    + "The theorem's bound is ΔΦ ≤ 4 − c = " + bound + ": " + HQ.verdict(dPhi <= bound, dPhi === bound ? "holds with equality" : "holds", "VIOLATED") + ". "
    + "Actual work is O(c) = " + cc + " units plus the first cut; amortized cost = O(c) + (4 − c) = O(1) once a unit of potential is scaled to pay for one cut. "
    + "Checks afterwards: every tree still min-heap-ordered — " + HQ.verdict(ordered) + "; every node that was cut arrived in the root list unmarked — " + HQ.verdict(rootsUnmarked) + ". "
    + "Read the ΔΦ: two marked nodes were unmarked by their cuts (−2·2 = −4), " + (FIB.countTrees(Hh) - 2) + " new roots appeared (+" + (FIB.countTrees(Hh) - 2) + "), and one fresh mark on 3 (+2): −4 + " + (FIB.countTrees(Hh) - 2) + " + 2 = " + (dPhi >= 0 ? "+" : "") + dPhi + ".");
})();

/* ═══════════════════════════════════════════════════════════════════════
   FIGURE 13 — #topk-svg  k smallest from a stream with a size-k MAX-heap
   ═══════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#topk-svg"); if (svg.empty()) return;
  const W = 760, H = 330;
  const els = { k: d3.select("#topk-k"), kOut: d3.select("#topk-k-out"), out: d3.select("#topk-readout") };
  const STREAM = [7, 2, 9, 4, 1, 8, 3, 6, 5, 10, 0, 11];

  function build() {
    const k = +els.k.property("value");
    els.kOut.text(k);
    const c = AL.counter();
    const heap = [];                                  // MAX-heap of the k smallest seen so far
    const frames = [];
    const snap = (i, note, extra) => frames.push(Object.assign({ heap: heap.slice(), i, note, cmp: c.get("cmp"), swap: c.get("swap"), sub: c.get("sub") }, extra || {}));
    snap(-1, "stream arrives left to right; the heap keeps the k = " + k + " smallest so far, largest of them at the root");
    STREAM.forEach((v, i) => {
      if (heap.length < k) {
        heap.push(v); HP.siftUp(heap, heap.length - 1, c, HP.MAX);
        snap(i, "heap not full → insert " + v + " (sift-up)");
      } else {
        c.add("cmp"); c.add("sub");
        if (v < heap[0]) {
          const old = heap[0];
          heap[0] = v; HP.siftDown(heap, 0, heap.length, c, HP.MAX);
          snap(i, v + " < root " + old + " → replace the root, sift-down (" + old + " can never be among the k smallest)");
        } else {
          snap(i, v + " ≥ root " + heap[0] + " → discard in one comparison");
        }
      }
    });
    const result = heap.slice().sort((a, b) => a - b);
    const truth = STREAM.slice().sort((a, b) => a - b).slice(0, k);
    const correct = result.join() === truth.join();
    // the alternative: heapify all n as a MIN-heap, pop k — measured on the same data
    const c2 = AL.counter(); const all = STREAM.slice(); HP.build(all, c2, HP.MIN);
    const popped = []; for (let j = 0; j < k; j++) { popped.push(all[0]); const last = all.pop(); if (all.length) { all[0] = last; HP.siftDown(all, 0, all.length, c2, HP.MIN); } }
    const correct2 = popped.join() === truth.join();
    const c3 = AL.counter(); HP.heapsort(heap.slice(), c3);       // what it costs to hand the k results out in order
    const n = STREAM.length;
    const bound1 = n * Math.max(1, 2 * HQ.floorLog2(k) + 1);   // per element: 1 compare + a sift-down of ≤ 2·floor(log2 k)
    HQ.clearControls(svg.node());
    AL.stepper(svg, { frames, delay: 800, label: "step", render: (fr) => {
      const f = AL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 8 });
      const g = f.g;
      g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text("the stream (n = " + n + "); processed elements dimmed, current in blue");
      AL.row(g, STREAM, { x: 0, y: 2, w: 30, gap: 3, mark: (i) => i === fr.i ? AC.accent : (i < fr.i ? "#22262f" : null) });
      g.append("text").attr("x", 0).attr("y", 78).attr("font-size", 11).attr("fill", AC.muted).text("the max-heap of the k smallest so far (root = the largest of them, the current threshold)");
      AL.row(g, fr.heap, { x: 0, y: 86, w: 30, gap: 3, mark: (i) => i === 0 ? AC.a2 : null });
      AL.binTree(g, fr.heap, { x: 0, y: 150, w: 300, levelH: 44, r: 13, mark: (i) => i === 0 ? AC.a2 : null });
      g.append("text").attr("x", 0).attr("y", 300).attr("font-size", 11.5).attr("fill", AC.ink).text(fr.note);
      HD.counts(g, 470, 100, [
        { text: "k = " + k + ", n = " + n, color: AC.ink, bold: true },
        { text: "heap comparisons so far: " + fr.cmp, color: AC.a2 },
        { text: "heap swaps so far: " + fr.swap, color: AC.rose },
        { text: "of which threshold tests: " + fr.sub, color: AC.muted },
        { text: "bound per element: 1 + 2·floor(log₂ k) = " + (1 + 2 * HQ.floorLog2(k)), color: AC.muted },
        { text: "memory: k slots, never n", color: AC.good }
      ]);
    } });
    els.out.html("<b>k = " + k + " smallest of a stream of " + n + ".</b> Size-k max-heap: <b>" + c.get("cmp") + "</b> comparisons, " + c.get("swap") + " swaps, and only k cells of memory; result " + JSON.stringify(result) + " — "
      + HQ.verdict(correct, "correct", "WRONG") + ". Bound n·(1 + 2·floor(log₂ k)) = " + bound1 + ": " + HQ.verdict(c.get("cmp") <= bound1, "within", "EXCEEDED") + ". "
      + "The other algorithm — min-heapify all " + n + " in place (≤ 2n = " + (2 * n) + " comparisons) then pop " + k + " times — measured <b>" + c2.get("cmp") + "</b> comparisons, result " + HQ.verdict(correct2, "correct", "WRONG") + ", and it hands the k results out <i>in order</i> but needs all n in memory at once. "
      + "The stream version's results are unsorted; sorting them costs another " + c3.get("cmp") + " comparisons here (O(k log k)). "
      + "O(n log k) versus O(n + k log n): for small k the stream version does about one comparison per element once the threshold settles — most arrivals are discarded on the spot — and its decisive advantage is memory, k cells instead of n. At k = n it is simply repeated insertion into a heap that is never emptied, which is why its count stays low there; the fair comparison at large k includes the sort.");
  }
  els.k.on("input change", build);
  build();
})();

/* ═══════════════════════════════════════════════════════════════════════
   FIGURE 14 — #median-svg  running median with two heaps
   ═══════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#median-svg"); if (svg.empty()) return;
  const W = 760, H = 330;
  const els = { out: d3.select("#median-readout") };
  const STREAM = [5, 15, 1, 3, 8, 7, 9, 10, 20, 2];
  const c = AL.counter();
  const low = [], high = [];            // low: MAX-heap of the smaller half;  high: MIN-heap of the larger half
  const frames = [];
  const median = () => low.length === high.length ? (low[0] + high[0]) / 2 : low[0];
  const snap = (i, note) => frames.push({ low: low.slice(), high: high.slice(), i, note, cmp: c.get("cmp"), moves: c.get("move"), med: low.length ? median() : null });
  const pushMax = (a, v) => { a.push(v); HP.siftUp(a, a.length - 1, c, HP.MAX); };
  const pushMin = (a, v) => { a.push(v); HP.siftUp(a, a.length - 1, c, HP.MIN); };
  const popTop = (a, above) => { const t = a[0]; const last = a.pop(); if (a.length) { a[0] = last; HP.siftDown(a, 0, a.length, c, above); } return t; };
  const truths = [];
  snap(-1, "low = max-heap of the smaller half, high = min-heap of the larger half; |low| = |high| or |low| = |high| + 1");
  STREAM.forEach((v, i) => {
    let note;
    if (!low.length || (c.add("cmp"), v <= low[0])) { pushMax(low, v); note = v + (low.length === 1 ? " starts low" : " ≤ max(low) → into low"); }
    else { pushMin(high, v); note = v + " > max(low) = " + low[0] + " → into high"; }
    if (low.length > high.length + 1) { const t = popTop(low, HP.MAX); pushMin(high, t); c.add("move"); note += "; low too big → move " + t + " to high"; }
    else if (high.length > low.length) { const t = popTop(high, HP.MIN); pushMax(low, t); c.add("move"); note += "; high too big → move " + t + " to low"; }
    const sorted = STREAM.slice(0, i + 1).sort((a, b) => a - b);
    const truth = sorted.length % 2 ? sorted[(sorted.length - 1) / 2] : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
    truths.push(truth);
    snap(i, note + " → median " + median() + " (sorted check: " + truth + ")");
  });
  const allOk = truths.every((t, i) => t === frames[i + 1].med);
  HQ.clearControls(svg.node());
  AL.stepper(svg, { frames, delay: 900, label: "step", render: (fr) => {
    const f = AL.frame(svg, W, H, { l: 14, r: 14, t: 22, b: 8 });
    const g = f.g;
    g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text("the stream; current element in blue");
    AL.row(g, STREAM, { x: 0, y: 2, w: 30, gap: 3, mark: (i) => i === fr.i ? AC.accent : (i < fr.i ? "#22262f" : null) });
    g.append("text").attr("x", 0).attr("y", 78).attr("font-size", 11).attr("fill", AC.a2).text("low — max-heap, root = largest of the smaller half");
    AL.binTree(g, fr.low, { x: 0, y: 90, w: 300, levelH: 40, r: 13, mark: (i) => i === 0 ? AC.a2 : null });
    g.append("text").attr("x", 380).attr("y", 78).attr("font-size", 11).attr("fill", AC.teal).text("high — min-heap, root = smallest of the larger half");
    AL.binTree(g, fr.high, { x: 380, y: 90, w: 300, levelH: 40, r: 13, mark: (i) => i === 0 ? AC.teal : null });
    g.append("text").attr("x", 0).attr("y", 262).attr("font-size", 11.5).attr("fill", AC.ink).text(fr.note);
    HD.counts(g, 0, 284, [
      { text: (fr.med === null ? "median: —" : "median so far: " + fr.med) + "    |low| = " + fr.low.length + ", |high| = " + fr.high.length + "    comparisons so far: " + fr.cmp + ", rebalancing moves: " + fr.moves, color: AC.muted }
    ]);
  } });
  els.out.html("<b>Running median over " + STREAM.length + " arrivals.</b> Each arrival costs one comparison against max(low), one heap insert, and at most one rebalancing move (a pop and a push) — O(log n) per element, O(1) to read the median. "
    + "Measured <b>" + c.get("cmp") + "</b> comparisons and <b>" + c.get("move") + "</b> rebalancing moves in total; every intermediate median was recomputed independently by sorting the prefix: " + HQ.verdict(allOk, "all " + STREAM.length + " agree", "MISMATCH") + ". "
    + "Final median " + frames[frames.length - 1].med + " with |low| = " + low.length + ", |high| = " + high.length + ". The invariant that makes it work is two-fold: every key in low ≤ every key in high (the halves), and the sizes differ by at most one (the middle) — so the median is at one root or the average of both.");
})();

/* ═══════════════════════════════════════════════════════════════════════
   FIGURE 15 — #local-svg  which indices a sift-down touches, and the gaps
   ═══════════════════════════════════════════════════════════════════════ */
(function () {
  const svg = d3.select("#local-svg"); if (svg.empty()) return;
  const W = 760, H = 340;
  const els = { n: d3.select("#local-n"), nOut: d3.select("#local-n-out"), line: d3.select("#local-line"), out: d3.select("#local-readout") };

  function draw() {
    const nExp = +els.n.property("value");
    const n = 1 << nExp;
    const perLine = +(els.line.empty() ? 16 : els.line.property("value"));   // keys per cache line
    els.nOut.text(HQ.int(n) + " (2^" + nExp + ")");
    /* a random max-heap, then a key of −1 at the root so the sift falls the full height along a
       data-dependent path; every index READ during the sift is recorded in order */
    const r = AL.rng(1191 + nExp);
    const a = AL.perm(n, r);
    const cb = AL.counter(); HP.build(a, cb);
    a[0] = -1;
    const touched = [], path = [0];
    const c = AL.counter();
    (function sift(i) {
      for (;;) {
        const l = 2 * i + 1, rr = 2 * i + 2; let best = i; touched.push(i);
        if (l < n) { touched.push(l); c.add("cmp"); if (a[l] > a[best]) best = l; }
        if (rr < n) { touched.push(rr); c.add("cmp"); if (a[rr] > a[best]) best = rr; }
        if (best === i) return;
        const t = a[i]; a[i] = a[best]; a[best] = t; c.add("swap"); i = best; path.push(i);
      }
    })(0);
    const h = HQ.floorLog2(n);
    const lines = new Set(touched.map(i => Math.floor(i / perLine)));
    const pathGaps = path.slice(1).map((v, k) => v - path[k]);
    // closed-form bracket: a parent at depth d−1 has index in [2^(d−1) − 1, 2^d − 2], so the gap to its child (p+1 or p+2) is in [2^(d−1), 2^d]
    const gapsOk = pathGaps.every((gp, k) => gp >= Math.pow(2, k) && gp <= Math.pow(2, k + 1));
    const fullHeight = path.length - 1 === h;
    const linesOnPath = path.filter((_, k) => k > 0 && Math.floor(path[k] / perLine) !== Math.floor(path[k - 1] / perLine)).length;
    // sibling pairs read: (2i+1, 2i+2) — share a line unless 2i+2 is a multiple of perLine
    let sibPairs = 0, sibShare = 0, sibStraddlePredicted = 0;
    path.forEach(i => { const l = 2 * i + 1, rr = 2 * i + 2; if (rr < n) { sibPairs++; if (Math.floor(l / perLine) === Math.floor(rr / perLine)) sibShare++; if (rr % perLine === 0) sibStraddlePredicted++; } });
    const straddleOk = (sibPairs - sibShare) === sibStraddlePredicted;

    const f = AL.frame(svg, W, H, { l: 56, r: 16, t: 18, b: 40 });
    const g = f.g;
    const x = d3.scaleLinear().domain([0, Math.max(1, path.length - 1)]).range([0, f.iw]);
    const y = d3.scaleLog().domain([1, Math.max(2, n)]).range([f.ih, 0]);
    AL.gridY(g, y, f.iw, 5);
    AL.axisB(g, x, Math.min(path.length, 12), "depth of the sift-down step", d => d);
    AL.axisL(g, y, 5, "cells between successive touches on the path (log scale)", d3.format("~s"));
    g.append("line").attr("x1", 0).attr("x2", f.iw).attr("y1", y(perLine)).attr("y2", y(perLine)).attr("stroke", AC.a2).attr("stroke-dasharray", "3,4");
    g.append("text").attr("x", f.iw - 4).attr("y", y(perLine) - 4).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.a2).text("one cache line = " + perLine + " keys; a gap above this is a new line every time");
    // the bracket 2^(d−1) … 2^d as a band
    const band = d3.area().x((d, k) => x(k + 1)).y0(d => y(Math.pow(2, d))).y1(d => y(Math.pow(2, d + 1)));
    g.append("path").datum(pathGaps.map((_, k) => k)).attr("d", band).attr("fill", AC.accent).attr("opacity", 0.12);
    const line = d3.line().x((d, k) => x(k + 1)).y(d => y(d));
    g.append("path").datum(pathGaps).attr("d", line).attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 2.2);
    g.selectAll(null).data(pathGaps).join("circle").attr("cx", (d, k) => x(k + 1)).attr("cy", d => y(d)).attr("r", 3.5).attr("fill", AC.accent);
    g.append("line").attr("x1", x(1)).attr("x2", x(Math.max(1, path.length - 1))).attr("y1", y(1)).attr("y2", y(1)).attr("stroke", AC.teal).attr("stroke-width", 2.2);
    AL.legend(g, [{ label: "parent → child gap on this sift's path (band: the closed-form bracket 2^(d−1) … 2^d)", color: AC.accent },
                  { label: "left child → right child: always 1 cell (siblings are adjacent)", color: AC.teal }], 12, 14);

    els.out.html("<b>n = " + HQ.int(n) + ", " + perLine + " keys per cache line; one sift-down from the root of a random max-heap, falling " + (path.length - 1) + " levels" + (fullHeight ? " (the full height)" : "") + ".</b> "
      + "Measured: " + touched.length + " index reads (" + c.get("cmp") + " comparisons), touching <b>" + lines.size + "</b> distinct cache lines. "
      + "Parent-to-child gaps along the path: " + pathGaps.slice(0, 7).join(", ") + (pathGaps.length > 7 ? ", …" : "") + " — each within its closed-form bracket [2^(d−1), 2^d]: " + HQ.verdict(gapsOk) + ". "
      + "Of the " + (path.length - 1) + " parent→child steps, <b>" + linesOnPath + "</b> crossed into a new cache line. "
      + "Of the " + sibPairs + " sibling pairs read, <b>" + sibShare + "</b> shared a line and " + (sibPairs - sibShare) + " straddled a boundary — predicted exactly by \"2i+2 is a multiple of " + perLine + "\": " + HQ.verdict(straddleOk) + ". "
      + "The top " + HQ.floorLog2(perLine) + " levels (indices 0 … " + (perLine - 2) + ", plus the first node of the next level) fit in one line; below that, every level is a fresh line at a gap the hardware cannot predict. "
      + "So \"the array heap has good locality\" is worth exactly this: siblings, nearly always; generations, only at the top.");
  }
  [els.n, els.line].forEach(s => { if (!s.empty()) s.on("input change", draw); });
  draw();
})();
