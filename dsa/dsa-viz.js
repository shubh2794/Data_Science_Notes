/* dsa-viz.js — shared D3 + algorithm helpers for BOTH DSA series
   (dsa/data-structures/* and dsa/algorithms/*).

   Loaded by every part built for the series. One page carried over from the
   pre-series layout (algorithms/dynamic-programming) is still self-contained
   and does not use it — wire it in when that part is rebuilt.

   Loaded after ../../notes.js and before each part's own "<part>.viz.js":
       <script src="../dsa-viz.js"></script>
       <script src="hash-tables.viz.js"></script>

   Nothing here draws on its own; it is a toolbox the part pages call. Every
   consuming page supplies its own <svg width height viewBox role aria-label>;
   these helpers only ever append into an <svg> that already exists.

   Provides two globals:

   AC — palette, matching the custom properties in notes.css.

   AL — namespace:
     · random    AL.rng(seed) → mulberry32; AL.shuffle(a, r) (copy); AL.randInt(r, lo, hi);
                 AL.sample(a, k, r); AL.perm(n, r)
     · drawing   AL.frame(sel, W, H, m), AL.axisB, AL.axisL, AL.gridY, AL.gridX,
                 AL.legend(g, items, x, y), AL.cells(W, H, cols, rows, m, gap)
     · dsa       AL.row(g, values, opt)      — an array as a row of indexed boxes
                 AL.bars(g, values, opt)     — an array as bar heights (sorting figures)
                 AL.binTree(g, nodes, opt)   — a binary tree from a level-order array
                 AL.arrow(g, x1, y1, x2, y2, opt)
     · measure   AL.counter()  — an instrumented op counter, so a figure can DISPLAY a
                 measured count rather than assert one. Always prefer this over a
                 hand-written number in a caption.
     · control   AL.stepper(svg, opt) — prev/play/next/reset control bound to a frame list.
                 Every stepped figure in either series uses this, so the interaction is
                 identical everywhere and keyboard support is written once.
     · misc      AL.linspace(a, b, k), AL.clamp(x, lo, hi), AL.fmt(x, d), AL.log2(x)

   House rule for this domain: any cost claim a figure displays (comparisons, swaps,
   probes, node visits) must come from AL.counter() wrapped around the real routine —
   never from a constant typed into the caption. */

const AC = {
  accent: "#5b9cff", a2: "#ffb454", good: "#4ade80", bad: "#f87171",
  ink: "#e6e9ef", muted: "#9aa3b2", line: "#2a2f3a",
  grid: "#1b2130", panel: "#171a23", panel2: "#1e222d", bg: "#0f1117",
  violet: "#c084fc", teal: "#2dd4bf", rose: "#fb7185"
};

const AL = (function () {

  /* ── pseudo-random, seeded so every reader sees the same picture ───────── */
  function rng(seed) {                        // mulberry32
    let a = (seed >>> 0) || 1;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randInt(r, lo, hi) { return lo + Math.floor(r() * (hi - lo + 1)); }
  function shuffle(a, r) {                    // Fisher–Yates on a COPY
    const s = [...a];
    for (let i = s.length - 1; i > 0; i--) {
      const j = Math.floor(r() * (i + 1));
      const t = s[i]; s[i] = s[j]; s[j] = t;
    }
    return s;
  }
  function sample(a, k, r) {
    const out = [];
    for (let i = 0; i < k; i++) out.push(a[Math.floor(r() * a.length)]);
    return out;
  }
  function perm(n, r) { return shuffle(Array.from({ length: n }, (_, i) => i + 1), r); }

  /* ── misc ─────────────────────────────────────────────────────────────── */
  function linspace(a, b, k) {
    if (k <= 1) return [a];
    const out = [];
    for (let i = 0; i < k; i++) out.push(a + (b - a) * i / (k - 1));
    return out;
  }
  function clamp(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }
  function fmt(x, d) { return (d === undefined ? x : (+x).toFixed(d)).toString(); }
  /* Math.log2 rather than Math.log(x)/Math.LN2 — binTree's depth() floors this,
     so a power of two landing at 2.9999999999999996 would misplace a whole level. */
  function log2(x) { return Math.log2 ? Math.log2(x) : Math.log(x) / Math.LN2; }

  /* ── drawing scaffolding ──────────────────────────────────────────────── */
  function frame(sel, W, H, m) {
    const mm = Object.assign({ l: 46, r: 16, t: 14, b: 34 }, m || {});
    const svg = (typeof sel === "string") ? d3.select(sel) : sel;
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${mm.l},${mm.t})`);
    return { svg: svg, g: g, m: mm, iw: W - mm.l - mm.r, ih: H - mm.t - mm.b };
  }
  function axisB(g, x, ih, ticks, label, fmtFn) {
    const ax = g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`)
      .call(fmtFn ? d3.axisBottom(x).ticks(ticks || 6).tickFormat(fmtFn)
                  : d3.axisBottom(x).ticks(ticks || 6));
    if (label) g.append("text").attr("x", x.range()[1]).attr("y", ih + 31).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", AC.muted).text(label);
    return ax;
  }
  function axisL(g, y, ticks, label, fmtFn) {
    const ax = g.append("g").attr("class", "axis")
      .call(fmtFn ? d3.axisLeft(y).ticks(ticks || 5).tickFormat(fmtFn)
                  : d3.axisLeft(y).ticks(ticks || 5));
    if (label) g.append("text").attr("x", 0).attr("y", -4).attr("text-anchor", "start")
      .attr("font-size", 11).attr("fill", AC.muted).text(label);
    return ax;
  }
  function gridY(g, y, iw, ticks) {
    g.append("g").attr("class", "gridlines").selectAll("line").data(y.ticks(ticks || 5)).join("line")
      .attr("x1", 0).attr("x2", iw).attr("y1", d => y(d)).attr("y2", d => y(d)).attr("stroke", AC.grid);
  }
  function gridX(g, x, ih, ticks) {
    g.append("g").attr("class", "gridlines").selectAll("line").data(x.ticks(ticks || 5)).join("line")
      .attr("y1", 0).attr("y2", ih).attr("x1", d => x(d)).attr("x2", d => x(d)).attr("stroke", AC.grid);
  }
  function legend(g, items, x, y, opts) {     // items: [{label, color, dash}]
    const o = Object.assign({ gap: 16, size: 10 }, opts || {});
    const gg = g.append("g").attr("transform", `translate(${x},${y})`);
    items.forEach((it, i) => {
      const row = gg.append("g").attr("transform", `translate(0,${i * o.gap})`);
      const ln = row.append("line").attr("x1", 0).attr("x2", o.size + 4).attr("y1", -4).attr("y2", -4)
        .attr("stroke", it.color).attr("stroke-width", 2.5);
      if (it.dash) ln.attr("stroke-dasharray", it.dash);
      row.append("text").attr("x", o.size + 10).attr("y", 0).attr("font-size", 11)
        .attr("fill", AC.muted).text(it.label);
    });
    return gg;
  }
  function cells(W, H, cols, rows, m, gap) {  // small multiples
    const mm = Object.assign({ l: 40, r: 12, t: 24, b: 30 }, m || {});
    const gp = Object.assign({ x: 18, y: 26 }, gap || {});
    const cw = (W - (cols - 1) * gp.x) / cols, ch = (H - (rows - 1) * gp.y) / rows;
    const out = [];
    for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
      out.push({ i, j, k: j * cols + i, x: i * (cw + gp.x), y: j * (ch + gp.y),
                 w: cw, h: ch, m: mm, iw: cw - mm.l - mm.r, ih: ch - mm.t - mm.b });
    }
    return out;
  }
  function arrow(g, x1, y1, x2, y2, opt) {
    const o = Object.assign({ color: AC.muted, w: 1.5, dash: null, head: 5 }, opt || {});
    const gg = g.append("g");
    const el = gg.append("line").attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2)
      .attr("stroke", o.color).attr("stroke-width", o.w);
    if (o.dash) el.attr("stroke-dasharray", o.dash);
    const a = Math.atan2(y2 - y1, x2 - x1), h = o.head;
    gg.append("path")
      .attr("d", `M${x2},${y2} L${x2 - h * Math.cos(a - 0.4)},${y2 - h * Math.sin(a - 0.4)}`
               + ` L${x2 - h * Math.cos(a + 0.4)},${y2 - h * Math.sin(a + 0.4)} Z`)
      .attr("fill", o.color);
    return gg;
  }

  /* ── DSA-specific primitives ──────────────────────────────────────────── */

  /* An array drawn as a row of indexed boxes — the workhorse figure of both
     series. `opt.mark` maps an index to a fill colour (null = default), so a
     caller highlights the pivot, the window, the probe, whatever it is showing.
     Returns {cellX(i), cellW, g} so the caller can hang arrows off a cell. */
  function row(g, values, opt) {
    const o = Object.assign({
      x: 0, y: 0, w: 34, h: 30, gap: 3, index: true, mark: null,
      label: null, fill: AC.panel2, stroke: AC.line, text: AC.ink, fontSize: 13
    }, opt || {});
    const gg = g.append("g").attr("transform", `translate(${o.x},${o.y})`);
    const step = o.w + o.gap;
    values.forEach((v, i) => {
      const f = o.mark ? (o.mark(i, v) || o.fill) : o.fill;
      const c = gg.append("g").attr("transform", `translate(${i * step},0)`);
      c.append("rect").attr("width", o.w).attr("height", o.h).attr("rx", 4)
        .attr("fill", f).attr("stroke", o.stroke);
      c.append("text").attr("x", o.w / 2).attr("y", o.h / 2 + 4).attr("text-anchor", "middle")
        .attr("font-size", o.fontSize).attr("fill", o.text)
        .text(v === null || v === undefined ? "" : v);
      if (o.index) {
        c.append("text").attr("x", o.w / 2).attr("y", o.h + 13).attr("text-anchor", "middle")
          .attr("font-size", 10).attr("fill", AC.muted).text(i);
      }
    });
    if (o.label) {
      gg.append("text").attr("x", -8).attr("y", o.h / 2 + 4).attr("text-anchor", "end")
        .attr("font-size", 11).attr("fill", AC.muted).text(o.label);
    }
    return { g: gg, cellW: o.w, step: step, cellX: i => i * step + o.w / 2, height: o.h };
  }

  /* An array drawn as bars — for sorting figures, where relative magnitude is
     the point and the value text would be noise. */
  function bars(g, values, opt) {
    const o = Object.assign({ x: 0, y: 0, w: 260, h: 110, gap: 2, mark: null,
                              fill: AC.accent, max: null }, opt || {});
    const gg = g.append("g").attr("transform", `translate(${o.x},${o.y})`);
    const n = values.length, bw = (o.w - (n - 1) * o.gap) / n;
    const mx = o.max === null ? Math.max(...values) : o.max;
    values.forEach((v, i) => {
      const bh = mx > 0 ? (v / mx) * o.h : 0;
      gg.append("rect").attr("x", i * (bw + o.gap)).attr("y", o.h - bh)
        .attr("width", bw).attr("height", bh).attr("rx", 2)
        .attr("fill", o.mark ? (o.mark(i, v) || o.fill) : o.fill);
    });
    return { g: gg, barW: bw, barX: i => i * (bw + o.gap) + bw / 2 };
  }

  /* A binary tree given as a level-order array (null = absent), which is exactly
     how a heap is stored and a convenient way to hand-write a small BST.
     Draws edges first so nodes sit on top. `opt.mark(i, v)` colours a node. */
  function binTree(g, level, opt) {
    const o = Object.assign({ x: 0, y: 0, w: 420, levelH: 56, r: 15, mark: null,
                              fill: AC.panel2, stroke: AC.line, edge: AC.line,
                              label: null }, opt || {});
    const gg = g.append("g").attr("transform", `translate(${o.x},${o.y})`);
    const depth = i => Math.floor(log2(i + 1));
    const nLevels = level.length ? depth(level.length - 1) + 1 : 0;
    const pos = i => {
      const d = depth(i), first = Math.pow(2, d) - 1, k = i - first, span = Math.pow(2, d);
      return { x: o.w * (k + 0.5) / span, y: d * o.levelH + o.r };
    };
    level.forEach((v, i) => {
      if (v === null || v === undefined || i === 0) return;
      const p = Math.floor((i - 1) / 2);
      if (level[p] === null || level[p] === undefined) return;
      const a = pos(p), b = pos(i);
      gg.append("line").attr("x1", a.x).attr("y1", a.y).attr("x2", b.x).attr("y2", b.y)
        .attr("stroke", o.edge).attr("stroke-width", 1.5);
    });
    level.forEach((v, i) => {
      if (v === null || v === undefined) return;
      const p = pos(i);
      gg.append("circle").attr("cx", p.x).attr("cy", p.y).attr("r", o.r)
        .attr("fill", o.mark ? (o.mark(i, v) || o.fill) : o.fill).attr("stroke", o.stroke);
      gg.append("text").attr("x", p.x).attr("y", p.y + 4).attr("text-anchor", "middle")
        .attr("font-size", 12).attr("fill", AC.ink).text(v);
    });
    return { g: gg, pos: pos, levels: nLevels, height: nLevels * o.levelH };
  }

  /* ── measurement ──────────────────────────────────────────────────────── */

  /* An instrumented counter. Wrap the real routine, run it, and DISPLAY what it
     returns — so a caption that says "17 comparisons" is reporting a measurement
     rather than repeating an assumption.
         const c = AL.counter();
         quicksort(a, c.bump("cmp"), c.bump("swap"));
         c.get("cmp")  →  17            c.all()  →  {cmp: 17, swap: 9}       */
  function counter() {
    const n = Object.create(null);
    return {
      bump: k => (by) => { n[k] = (n[k] || 0) + (by === undefined ? 1 : by); },
      add: (k, by) => { n[k] = (n[k] || 0) + (by === undefined ? 1 : by); },
      get: k => n[k] || 0,
      all: () => Object.assign({}, n),
      reset: () => { for (const k in n) delete n[k]; }
    };
  }

  /* ── stepped-figure control ───────────────────────────────────────────── */

  /* Binds prev / play / next / reset to a list of frames and calls render(frame,
     index) for each. The caller owns the frames and the drawing; this owns the
     buttons, the timer, the keyboard, and the "frame i of n" readout.

         const st = AL.stepper(d3.select("#fig"), {
           frames: frames, render: (f, i) => draw(f, i), delay: 700
         });

     Accessibility: the control is a <div role="group"> of real <button>s placed
     after the <svg>; arrow keys step when any of them has focus. Returns
     {go(i), play(), pause(), index()} so a page can drive it from elsewhere. */
  function stepper(svgSel, opt) {
    const o = Object.assign({ frames: [], render: null, delay: 700, loop: false,
                              label: "step", autoplay: false }, opt || {});
    const svg = (typeof svgSel === "string") ? d3.select(svgSel) : svgSel;
    const host = d3.select(svg.node().parentNode);
    const n = o.frames.length;
    let i = 0, timer = null;

    const bar = host.append("div")
      .attr("role", "group").attr("aria-label", "Animation controls")
      .style("display", "flex").style("gap", "8px").style("align-items", "center")
      .style("margin", "8px 0 0").style("font-size", "12px");

    const mk = (txt, aria, fn) => bar.append("button")
      .attr("type", "button").attr("aria-label", aria)
      .style("background", "var(--panel-2)").style("color", "var(--ink)")
      .style("border", "1px solid var(--line)").style("border-radius", "6px")
      .style("padding", "3px 9px").style("cursor", "pointer").style("font-size", "12px")
      .text(txt).on("click", fn);

    mk("⟲", "Reset to the first step", () => { pause(); go(0); });
    mk("‹", "Previous step", () => { pause(); go(i - 1); });
    const playBtn = mk("▶", "Play", () => (timer ? pause() : play()));
    mk("›", "Next step", () => { pause(); go(i + 1); });
    const readout = bar.append("span").style("color", "var(--muted)");

    bar.on("keydown", (ev) => {
      if (ev.key === "ArrowRight") { pause(); go(i + 1); ev.preventDefault(); }
      else if (ev.key === "ArrowLeft") { pause(); go(i - 1); ev.preventDefault(); }
    });

    function go(k) {
      if (n === 0) return;
      i = o.loop ? ((k % n) + n) % n : clamp(k, 0, n - 1);
      if (o.render) o.render(o.frames[i], i);
      readout.text(`${o.label} ${i + 1} / ${n}`);
      if (!o.loop && i === n - 1) pause();
    }
    function play() {
      if (n === 0 || timer) return;
      if (!o.loop && i === n - 1) go(0);
      playBtn.text("❚❚").attr("aria-label", "Pause");
      timer = setInterval(() => go(i + 1), o.delay);
    }
    function pause() {
      if (timer) { clearInterval(timer); timer = null; }
      playBtn.text("▶").attr("aria-label", "Play");
    }

    go(0);
    if (o.autoplay) play();
    return { go, play, pause, index: () => i, frames: o.frames };
  }

  return {
    rng, randInt, shuffle, sample, perm,
    linspace, clamp, fmt, log2,
    frame, axisB, axisL, gridY, gridX, legend, cells, arrow,
    row, bars, binTree,
    counter, stepper
  };
})();
