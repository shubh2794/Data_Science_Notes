/* parallel.viz.js — figures for dsa/algorithms/parallel.html
   (part 9 of the Algorithm Design & Analysis series).

   Loaded after ../../data.js, ../../notes.js and ../dsa-viz.js, so the globals
   AC (palette) and AL (rng/frame/axisB/axisL/row/counter/stepper/…) exist.

   House rule obeyed throughout: every work, span, step, addition or
   interleaving count this page DISPLAYS is produced by running the real
   routine under AL.counter() and reading the counter back. The central device
   is a TRACER: each parallel algorithm below is written once against a tiny
   fork-join interface
       X.work(k)          k unit-cost instructions, in series
       X.par(f, g, …)     spawn f, g, … ; sync — they may run in parallel
   and there are two interchangeable tracers behind it:
       PA.algebra(c)      keeps (work, span) as numbers: seq adds both,
                          par adds work and takes the MAX of span. Fast.
       PA.dag(c)          records every unit instruction as a DAG node, so a
                          figure can draw the DAG, find its critical path, and
                          hand it to a scheduler.
   Both count work into c.add("work"); the load-time audit checks the two agree.

   Layout:  PA — routines (pure, DOM-free, loadable in node)
            figures — one block per <svg>, in page order. */

const PA = (function () {
  const nop = { add: function () {}, get: function () { return 0; } };

  /* ── tracers ─────────────────────────────────────────────────────────── */
  function algebra(c) {
    c = c || nop;
    let s = 0, w = 0;
    return {
      kind: "algebra",
      work(k) { k = (k === undefined) ? 1 : k; if (k <= 0) return; w += k; s += k; c.add("work", k); },
      par() {
        const fs = arguments, s0 = s; let m = s0;
        c.add("spawn", fs.length - 1);
        for (let i = 0; i < fs.length; i++) { s = s0; fs[i](); if (s > m) m = s; }
        s = m;
      },
      span() { return s; },
      total() { return w; }
    };
  }

  function dag(c, cap) {
    c = c || nop; cap = cap || 40000;
    const nodes = []; let cur = [];
    return {
      kind: "dag", nodes: nodes,
      work(k, label, kind) {
        k = (k === undefined) ? 1 : k;
        for (let i = 0; i < k; i++) {
          if (nodes.length >= cap) throw new Error("DAG larger than " + cap + " nodes");
          const id = nodes.length;
          nodes.push({ id: id, preds: cur.slice(), label: label || "", kind: kind || "work" });
          c.add("work"); cur = [id];
        }
      },
      par() {
        const fs = arguments, s0 = cur, ends = new Set();
        c.add("spawn", fs.length - 1);
        for (let i = 0; i < fs.length; i++) { cur = s0; fs[i](); cur.forEach(e => ends.add(e)); }
        cur = Array.from(ends);
      },
      span() { return analyse(nodes).span; },
      total() { return nodes.length; }
    };
  }

  /* depth (longest path from a source, counted in nodes), height (longest path to
     a sink), the span, and one critical path. Creation order is topological. */
  function analyse(nodes) {
    const n = nodes.length, depth = new Array(n), height = new Array(n).fill(1);
    const succ = nodes.map(() => []);
    let span = 0, end = -1;
    for (let i = 0; i < n; i++) {
      let d = 0;
      for (const p of nodes[i].preds) { if (depth[p] > d) d = depth[p]; succ[p].push(i); }
      depth[i] = d + 1;
      if (depth[i] > span) { span = depth[i]; end = i; }
    }
    for (let i = n - 1; i >= 0; i--) for (const s of succ[i]) if (height[s] + 1 > height[i]) height[i] = height[s] + 1;
    const path = new Set();
    let v = end;
    while (v >= 0) {
      path.add(v);
      let nx = -1;
      for (const p of nodes[v].preds) if (depth[p] === depth[v] - 1) { nx = p; break; }
      v = nx;
    }
    return { depth: depth, height: height, succ: succ, span: span, path: path };
  }

  /* ── fork-join building blocks ───────────────────────────────────────── */
  /* parallel for by recursive halving: Θ(log n) span just to spawn n iterations */
  function pfor(X, lo, hi, body) {
    if (hi - lo <= 0) return;
    if (hi - lo === 1) { body(lo); return; }
    const mid = (lo + hi) >> 1;
    X.work(1, "for " + lo + ".." + (hi - 1), "spawn");
    X.par(() => pfor(X, lo, mid, body), () => pfor(X, mid, hi, body));
  }

  /* ── the algorithms (each computes its real answer) ──────────────────── */
  function pfib(X, n) {
    X.work(1, "fib(" + n + ")", n < 2 ? "work" : "spawn");
    if (n < 2) return n;
    let a = 0, b = 0;
    X.par(() => { a = pfib(X, n - 1); }, () => { b = pfib(X, n - 2); });
    X.work(1, "sum fib(" + n + ")", "join");
    return a + b;
  }

  function psum(X, A, lo, hi) {
    if (hi - lo === 1) { X.work(1, "a[" + lo + "]", "work"); return A[lo]; }
    const mid = (lo + hi) >> 1; let l = 0, r = 0;
    X.work(1, "split " + lo + ".." + (hi - 1), "spawn");
    X.par(() => { l = psum(X, A, lo, mid); }, () => { r = psum(X, A, mid, hi); });
    X.work(1, "add " + lo + ".." + (hi - 1), "join");
    return l + r;
  }

  function serialSum(X, A) { let s = 0; for (let i = 0; i < A.length; i++) { X.work(1, "s += a[" + i + "]"); s += A[i]; } return s; }

  function loopBody(X, n, b) { const out = new Array(n); pfor(X, 0, n, i => { X.work(b, "body " + i, "work"); out[i] = i * i; }); return out; }

  /* merge sort, recursive calls in parallel, SERIAL merge — span Θ(n) */
  function msortNaive(X, A) {
    if (A.length <= 1) { X.work(1, "leaf", "work"); return A.slice(); }
    const mid = A.length >> 1; let L, R;
    X.work(1, "split n=" + A.length, "spawn");
    X.par(() => { L = msortNaive(X, A.slice(0, mid)); }, () => { R = msortNaive(X, A.slice(mid)); });
    const out = []; let i = 0, j = 0;
    while (out.length < A.length) {
      X.work(1, "merge n=" + A.length, "join");
      if (j >= R.length || (i < L.length && L[i] <= R[j])) out.push(L[i++]); else out.push(R[j++]);
    }
    return out;
  }

  /* binary search: first index q in [p, r+1] with x ≤ T[q] */
  function bsearch(X, x, T, p, r) {
    let lo = p, hi = Math.max(p, r + 1);
    X.work(1, "search", "work");
    while (lo < hi) { X.work(1, "search", "work"); const mid = (lo + hi) >> 1; if (x <= T[mid]) hi = mid; else lo = mid + 1; }
    return hi;
  }
  /* parallel merge of T[p1..r1] and T[p2..r2] into A[p3..] (inclusive bounds) */
  function pmerge(X, T, p1, r1, p2, r2, A, p3) {
    let n1 = r1 - p1 + 1, n2 = r2 - p2 + 1;
    if (n1 < n2) { let t = p1; p1 = p2; p2 = t; t = r1; r1 = r2; r2 = t; t = n1; n1 = n2; n2 = t; }
    if (n1 === 0) { X.work(1, "empty", "work"); return; }
    const q1 = (p1 + r1) >> 1;
    const q2 = bsearch(X, T[q1], T, p2, r2);
    const q3 = p3 + (q1 - p1) + (q2 - p2);
    A[q3] = T[q1];
    X.work(1, "place " + T[q1], "spawn");
    X.par(() => pmerge(X, T, p1, q1 - 1, p2, q2 - 1, A, p3),
          () => pmerge(X, T, q1 + 1, r1, q2, r2, A, q3 + 1));
  }
  function pmsort(X, A, p, r, B, s) {
    const n = r - p + 1;
    if (n === 1) { B[s] = A[p]; X.work(1, "leaf", "work"); return; }
    const T = new Array(n);                     // allocation assumed O(1) — see §17
    const q = (p + r) >> 1, qq = q - p + 1;
    X.work(1, "split n=" + n, "spawn");
    X.par(() => pmsort(X, A, p, q, T, 0), () => pmsort(X, A, q + 1, r, T, qq));
    pmerge(X, T, 0, qq - 1, qq, n - 1, B, s);
  }
  function pMergeSort(X, A) { const B = new Array(A.length); if (A.length) pmsort(X, A, 0, A.length - 1, B, 0); return B; }

  /* matrix multiply: two parallel loops, serial inner product — span Θ(n) */
  function matLoops(X, A, B) {
    const n = A.length, C = A.map(r => r.map(() => 0));
    pfor(X, 0, n, i => pfor(X, 0, n, j => {
      for (let k = 0; k < n; k++) { X.work(1, "c" + i + j + " += a" + i + k + "·b" + k + j, "work"); C[i][j] += A[i][k] * B[k][j]; }
    }));
    return C;
  }
  /* divide and conquer with a temporary: 8 products in parallel, then a parallel add */
  function matDC(X, A, B) {
    const n = A.length, C = A.map(r => r.map(() => 0));
    function rec(Cv, Av, Bv, m) {
      if (m === 1) { X.work(1, "mul", "work"); Cv.M[Cv.r][Cv.c] = Av.M[Av.r][Av.c] * Bv.M[Bv.r][Bv.c]; return; }
      const Tm = []; for (let i = 0; i < m; i++) Tm.push(new Array(m).fill(0));   // allocation O(1), assumed
      const Tv = { M: Tm, r: 0, c: 0 }, h = m >> 1;
      const q = (V, i, j) => ({ M: V.M, r: V.r + i * h, c: V.c + j * h });
      X.work(1, "partition m=" + m, "spawn");
      X.par(
        () => rec(q(Cv, 0, 0), q(Av, 0, 0), q(Bv, 0, 0), h), () => rec(q(Cv, 0, 1), q(Av, 0, 0), q(Bv, 0, 1), h),
        () => rec(q(Cv, 1, 0), q(Av, 1, 0), q(Bv, 0, 0), h), () => rec(q(Cv, 1, 1), q(Av, 1, 0), q(Bv, 0, 1), h),
        () => rec(q(Tv, 0, 0), q(Av, 0, 1), q(Bv, 1, 0), h), () => rec(q(Tv, 0, 1), q(Av, 0, 1), q(Bv, 1, 1), h),
        () => rec(q(Tv, 1, 0), q(Av, 1, 1), q(Bv, 1, 0), h), () => rec(q(Tv, 1, 1), q(Av, 1, 1), q(Bv, 1, 1), h));
      pfor(X, 0, m, i => pfor(X, 0, m, j => { X.work(1, "add", "join"); Cv.M[Cv.r + i][Cv.c + j] += Tm[i][j]; }));
    }
    rec({ M: C, r: 0, c: 0 }, { M: A, r: 0, c: 0 }, { M: B, r: 0, c: 0 }, n);
    return C;
  }
  function matSerial(A, B) {
    const n = A.length; return A.map((r, i) => r.map((_, j) => { let s = 0; for (let k = 0; k < n; k++) s += A[i][k] * B[k][j]; return s; }));
  }

  /* scan, naive (log n rounds, each a parallel loop over all n) — inclusive */
  function scanNaive(X, A, c) {
    let a = A.slice(); const n = a.length;
    for (let off = 1; off < n; off *= 2) {
      const b = a.slice(), src = a;
      pfor(X, 0, n, i => { X.work(1, "x" + i, "work"); if (i >= off) { b[i] = src[i - off] + src[i]; if (c) c.add("add"); } });
      a = b;
    }
    return a;
  }
  /* scan, work-efficient, recursive fork-join: up-sweep builds the sum tree,
     down-sweep pushes prefixes down it — exclusive */
  function scanEfficient(X, A, c) {
    const out = new Array(A.length);
    function up(lo, hi) {
      if (hi - lo === 1) { X.work(1, "leaf " + lo, "work"); return { lo: lo, hi: hi, sum: A[lo] }; }
      const mid = (lo + hi) >> 1; let l, r;
      X.work(1, "up split", "spawn");
      X.par(() => { l = up(lo, mid); }, () => { r = up(mid, hi); });
      X.work(1, "up add", "join"); if (c) c.add("add");
      return { lo: lo, hi: hi, sum: l.sum + r.sum, l: l, r: r };
    }
    function down(t, pre) {
      if (!t.l) { X.work(1, "out " + t.lo, "work"); out[t.lo] = pre; return; }
      X.work(1, "down", "spawn"); if (c) c.add("add");
      const rp = pre + t.l.sum;
      X.par(() => down(t.l, pre), () => down(t.r, rp));
    }
    if (A.length) down(up(0, A.length), 0);
    return out;
  }
  function serialScan(A, inclusive) {
    const o = []; let s = 0;
    for (const v of A) { if (inclusive) { s += v; o.push(s); } else { o.push(s); s += v; } }
    return o;
  }

  /* ── greedy scheduler on a recorded DAG ──────────────────────────────── */
  /* prio: "fifo" (oldest ready first), "long" (largest height first),
     "short" (smallest height first — still greedy, deliberately naive) */
  function greedy(nodes, p, prio, c, info) {
    c = c || nop;
    const A = info || analyse(nodes), n = nodes.length;
    const indeg = nodes.map(v => v.preds.length);
    let ready = []; for (let i = 0; i < n; i++) if (!indeg[i]) ready.push(i);
    const steps = []; let done = 0, complete = 0, incomplete = 0;
    const cmp = prio === "long" ? (a, b) => (A.height[b] - A.height[a]) || (a - b)
              : prio === "short" ? (a, b) => (A.height[a] - A.height[b]) || (a - b)
              : (a, b) => a - b;
    while (done < n) {
      ready.sort(cmp);
      const run = ready.slice(0, p), rest = ready.slice(p), next = [];
      for (const v of run) { for (const s of A.succ[v]) if (--indeg[s] === 0) next.push(s); }
      done += run.length; c.add("step"); c.add("executed", run.length);
      if (run.length === p) { complete++; c.add("complete"); } else { incomplete++; c.add("incomplete"); }
      steps.push({ run: run, ready: ready.length });
      ready = rest.concat(next);
      if (!run.length) throw new Error("scheduler stalled");
    }
    return { Tp: steps.length, steps: steps, complete: complete, incomplete: incomplete };
  }

  /* the Amdahl workload: s serial unit steps, then N independent unit tasks,
     run greedily on p processors. Returns the measured number of steps. */
  function runWorkload(s, N, p, c) {
    c = c || nop; let steps = 0;
    for (let i = 0; i < s; i++) { steps++; c.add("step"); c.add("work"); }
    let rem = N;
    while (rem > 0) { const t = Math.min(p, rem); rem -= t; steps++; c.add("step"); c.add("work", t); }
    return steps;
  }

  /* ── race enumeration ────────────────────────────────────────────────── */
  function program(mode, inc) {
    const one = mode === "atomic" ? ["faa"] : mode === "lock" ? ["acquire", "load", "add", "store", "release"] : ["load", "add", "store"];
    let out = []; for (let i = 0; i < inc; i++) out = out.concat(one); return out;
  }
  /* every feasible interleaving of `k` strands each running `prog` on a shared x */
  function interleavings(k, prog, c, keep) {
    c = c || nop; keep = keep || 5000;
    const hist = {}, kept = [];
    const pc = new Array(k).fill(0), reg = new Array(k).fill(0), trace = [];
    let x = 0, lock = -1;
    function dfs() {
      let moved = false;
      for (let t = 0; t < k; t++) {
        if (pc[t] >= prog.length) continue;
        const op = prog[pc[t]];
        if (op === "acquire" && lock !== -1) continue;          // blocked
        const save = { x: x, lock: lock, r: reg[t] };
        if (op === "load") reg[t] = x;
        else if (op === "add") reg[t] = reg[t] + 1;
        else if (op === "store") x = reg[t];
        else if (op === "faa") x = x + 1;
        else if (op === "acquire") lock = t;
        else if (op === "release") lock = -1;
        c.add("instr"); pc[t]++; trace.push({ t: t, op: op, x: x, r: reg[t] }); moved = true;
        dfs();
        trace.pop(); pc[t]--; x = save.x; lock = save.lock; reg[t] = save.r;
      }
      if (!moved) {
        const all = pc.every(v => v >= prog.length);
        if (!all) { c.add("deadlock"); return; }
        c.add("interleaving"); hist[x] = (hist[x] || 0) + 1;
        if (kept.length < keep) kept.push({ trace: trace.slice(), x: x });
      }
    }
    dfs();
    return { hist: hist, kept: kept };
  }

  /* ── scan frames for the animation (array, level-synchronous) ────────── */
  function scanFrames(A, which, c) {
    c = c || nop;
    const n = A.length, frames = [];
    let a = A.slice();
    frames.push({ before: a.slice(), after: a.slice(), arrows: [], title: "input", phase: "in" });
    if (which === "naive") {
      for (let off = 1; off < n; off *= 2) {
        const b = a.slice(), arrows = [];
        for (let i = off; i < n; i++) { b[i] = a[i - off] + a[i]; c.add("add"); arrows.push([i - off, i, "add"]); }
        c.add("round");
        frames.push({ before: a.slice(), after: b.slice(), arrows: arrows, title: "round: x[i] ← x[i − " + off + "] + x[i]", phase: "naive", adds: c.get("add"), rounds: c.get("round") });
        a = b;
      }
    } else {
      for (let d = 1; d < n; d *= 2) {
        const b = a.slice(), arrows = [];
        for (let k = 0; k < n; k += 2 * d) { const i = k + d - 1, j = k + 2 * d - 1; b[j] = a[i] + a[j]; c.add("add"); arrows.push([i, j, "add"]); }
        c.add("round");
        frames.push({ before: a.slice(), after: b.slice(), arrows: arrows, title: "up-sweep, stride " + (2 * d), phase: "up", adds: c.get("add"), rounds: c.get("round") });
        a = b;
      }
      const b0 = a.slice(); b0[n - 1] = 0;
      frames.push({ before: a.slice(), after: b0.slice(), arrows: [], title: "clear the root: x[n − 1] ← 0 (the total is saved)", phase: "clear", total: a[n - 1], adds: c.get("add"), rounds: c.get("round"), clear: n - 1 });
      a = b0;
      for (let d = n >> 1; d >= 1; d >>= 1) {
        const b = a.slice(), arrows = [];
        for (let k = 0; k < n; k += 2 * d) {
          const i = k + d - 1, j = k + 2 * d - 1;
          b[i] = a[j]; b[j] = a[i] + a[j]; c.add("add");
          arrows.push([j, i, "copy"]); arrows.push([i, j, "add"]);
        }
        c.add("round");
        frames.push({ before: a.slice(), after: b.slice(), arrows: arrows, title: "down-sweep, stride " + (2 * d), phase: "down", adds: c.get("add"), rounds: c.get("round") });
        a = b;
      }
    }
    return { frames: frames, result: a };
  }

  /* ── registries ─────────────────────────────────────────────────────── */
  function arr(n, seed) { const r = AL.rng(seed); return Array.from({ length: n }, () => 1 + Math.floor(r() * 9)); }
  function perm(n, seed) { return AL.perm(n, AL.rng(seed)); }
  function mat(n, seed) { const r = AL.rng(seed); return Array.from({ length: n }, () => Array.from({ length: n }, () => Math.floor(r() * 5) - 2)); }
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

  /* programs small enough to draw as a DAG / schedule strand by strand */
  const PROGS = {
    fib:   { name: "P-FIB(n)", lo: 2, hi: 7, hiS: 9, def: 5,
             run: (X, n) => { const v = pfib(X, n); return "fib(" + n + ") = " + v; } },
    sum:   { name: "parallel sum (reduction) of n numbers", lo: 2, hi: 16, hiS: 64, def: 8,
             run: (X, n) => { const A = arr(n, 7), v = psum(X, A, 0, n); return "sum = " + v + (v === A.reduce((a, b) => a + b, 0) ? " ✓" : " ✗"); } },
    loop:  { name: "parallel for, n iterations, body = 3 steps", lo: 2, hi: 16, hiS: 64, def: 8,
             run: (X, n) => { loopBody(X, n, 3); return "n = " + n + " iterations"; } },
    serial:{ name: "serial loop: sum of n numbers", lo: 2, hi: 16, hiS: 64, def: 8,
             run: (X, n) => { const A = arr(n, 7); return "sum = " + serialSum(X, A); } },
    amdahl:{ name: "4 serial steps, parallel for (body 2), 4 serial steps", lo: 2, hi: 16, hiS: 64, def: 8,
             run: (X, n) => { X.work(4, "serial setup", "join"); loopBody(X, n, 2); X.work(4, "serial wrap-up", "join"); return "serial part fixed at 8 steps"; } },
    msn:   { name: "merge sort, serial merge", lo: 2, hi: 16, hiS: 32, def: 8,
             run: (X, n) => { const A = perm(n, 11), B = msortNaive(X, A); return "sorted " + (same(B, A.slice().sort((a, b) => a - b)) ? "✓" : "✗"); } },
    msp:   { name: "merge sort, parallel merge", lo: 2, hi: 16, hiS: 32, def: 8,
             run: (X, n) => { const A = perm(n, 11), B = pMergeSort(X, A); return "sorted " + (same(B, A.slice().sort((a, b) => a - b)) ? "✓" : "✗"); } },
    mat:   { name: "matrix multiply, two parallel loops (n × n)", lo: 2, hi: 4, hiS: 4, def: 2, pow2: false,
             run: (X, n) => { const A = mat(n, 3), B = mat(n, 4); return "C = AB " + (same(matLoops(X, A, B), matSerial(A, B)) ? "✓" : "✗"); } }
  };

  /* routines measured across n for the growth figure; predicted shapes for the ratio */
  const lg = n => Math.log2(n);
  const GROW = {
    sum:  { name: "parallel sum (reduction)", ns: [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], W: "n", S: "log n", wf: n => n, sf: n => lg(n),
            run: (X, n) => { const A = arr(n, n); return psum(X, A, 0, n) === A.reduce((a, b) => a + b, 0); } },
    loop: { name: "parallel for, body = 1 step", ns: [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], W: "n", S: "log n", wf: n => n, sf: n => lg(n),
            run: (X, n) => { loopBody(X, n, 1); return true; } },
    msn:  { name: "merge sort, serial merge", ns: [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], W: "n log n", S: "n", wf: n => n * lg(n), sf: n => n,
            run: (X, n) => { const A = perm(n, n); return same(msortNaive(X, A), A.slice().sort((a, b) => a - b)); } },
    msp:  { name: "merge sort, parallel merge", ns: [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], W: "n log n", S: "log³ n", wf: n => n * lg(n), sf: n => Math.pow(lg(n), 3), S2: "log² n", sf2: n => lg(n) * lg(n),
            run: (X, n) => { const A = perm(n, n); return same(pMergeSort(X, A), A.slice().sort((a, b) => a - b)); } },
    scn:  { name: "scan, naive (log n rounds of a parallel loop)", ns: [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], W: "n log n", S: "log² n", wf: n => n * lg(n), sf: n => lg(n) * lg(n),
            run: (X, n) => { const A = arr(n, n); return same(scanNaive(X, A), serialScan(A, true)); } },
    sce:  { name: "scan, work-efficient (recursive up/down-sweep)", ns: [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], W: "n", S: "log n", wf: n => n, sf: n => lg(n),
            run: (X, n) => { const A = arr(n, n); return same(scanEfficient(X, A), serialScan(A, false)); } },
    mml:  { name: "matrix multiply, parallel loops (n × n)", ns: [2, 4, 8, 16, 32, 64], W: "n³", S: "n", wf: n => n * n * n, sf: n => n,
            run: (X, n) => { const A = mat(n, n), B = mat(n, n + 1); return same(matLoops(X, A, B), matSerial(A, B)); } },
    mmd:  { name: "matrix multiply, divide & conquer (n × n)", ns: [2, 4, 8, 16, 32, 64], W: "n³", S: "log² n", wf: n => n * n * n, sf: n => lg(n) * lg(n),
            run: (X, n) => { const A = mat(n, n), B = mat(n, n + 1); return same(matDC(X, A, B), matSerial(A, B)); } }
  };

  function measure(routine, n) {
    const c = AL.counter(), X = algebra(c);
    const ok = routine.run(X, n);
    return { n: n, work: c.get("work"), span: X.span(), spawn: c.get("spawn"), ok: ok };
  }
  function record(prog, n) {
    const c = AL.counter(), X = dag(c);
    const msg = prog.run(X, n);
    return { nodes: X.nodes, info: analyse(X.nodes), work: c.get("work"), spawn: c.get("spawn"), msg: msg };
  }

  return { algebra, dag, analyse, pfor, pfib, psum, msortNaive, pMergeSort, matLoops, matDC, matSerial,
           scanNaive, scanEfficient, serialScan, greedy, runWorkload, program, interleavings, scanFrames,
           PROGS, GROW, measure, record };
})();

if (typeof module !== "undefined" && module.exports) module.exports = PA;

/* shared DOM helpers */
const PX = {
  val(id) { const e = document.getElementById(id); return e ? e.value : null; },
  setText(id, t) { const e = document.getElementById(id); if (e) e.textContent = t; },
  fmt(x) { return (+x).toLocaleString("en-US").replace(/,/g, " "); },
  f2(x) { return (+x).toFixed(2); },
  setRange(id, lo, hi, v) {
    const e = document.getElementById(id); if (!e) return v;
    e.min = lo; e.max = hi; const c = AL.clamp(+v, lo, hi); e.value = c; return c;
  },
  kindColor(k) { return k === "spawn" ? AC.violet : k === "join" ? AC.teal : AC.accent; }
};

/* ═══════════════════════════════════════════════════════════════════════════
   §05 — the computation DAG explorer
   ═══════════════════════════════════════════════════════════════════════════ */
(function () {
  const svgEl = document.getElementById("dag-svg"); if (!svgEl) return;
  const W = 700, H = 380;
  const sel = document.getElementById("dag-prog"), rng = document.getElementById("dag-n"), crit = document.getElementById("dag-crit");

  function draw() {
    const P = PA.PROGS[sel.value];
    const n = PX.setRange("dag-n", P.lo, P.hi, rng.value);
    PX.setText("dag-n-val", n);
    const R = PA.record(P, n), I = R.info, nodes = R.nodes;
    const F = AL.frame(d3.select(svgEl), W, H, { l: 16, r: 16, t: 22, b: 30 });
    const byDepth = {};
    nodes.forEach(v => { const d = I.depth[v.id]; (byDepth[d] = byDepth[d] || []).push(v.id); });
    const maxW = Math.max(...Object.values(byDepth).map(a => a.length));
    const dx = F.iw / Math.max(1, I.span - 1 || 1), dy = F.ih / Math.max(1, maxW);
    const pos = {};
    Object.keys(byDepth).forEach(d => {
      const ids = byDepth[d], k = ids.length;
      ids.forEach((id, j) => { pos[id] = { x: I.span === 1 ? F.iw / 2 : (d - 1) * dx, y: F.ih / 2 + (j - (k - 1) / 2) * dy }; });
    });
    const r = Math.max(2.5, Math.min(8, dx / 3, dy / 2.6));
    const showCrit = crit.checked;
    nodes.forEach(v => v.preds.forEach(p => {
      const on = showCrit && I.path.has(v.id) && I.path.has(p) && I.depth[p] === I.depth[v.id] - 1;
      F.g.append("line").attr("x1", pos[p].x).attr("y1", pos[p].y).attr("x2", pos[v.id].x).attr("y2", pos[v.id].y)
        .attr("stroke", on ? AC.a2 : AC.line).attr("stroke-width", on ? 2.4 : 1);
    }));
    nodes.forEach(v => {
      const on = showCrit && I.path.has(v.id);
      F.g.append("circle").attr("cx", pos[v.id].x).attr("cy", pos[v.id].y).attr("r", r)
        .attr("fill", PX.kindColor(v.kind)).attr("stroke", on ? AC.a2 : AC.bg).attr("stroke-width", on ? 2.2 : 1)
        .append("title").text(v.label + "  (depth " + I.depth[v.id] + ")");
    });
    F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted)
      .text("time flows left → right; x = the earliest step a node could run with unlimited processors");
    AL.legend(F.g, [{ label: "spawn / split strand", color: AC.violet }, { label: "leaf / body work", color: AC.accent },
                    { label: "combine / sync strand", color: AC.teal }, { label: "critical path", color: AC.a2 }], 0, F.ih + 22, { gap: 0 })
      .selectAll("g").attr("transform", (d, i) => "translate(" + (i * 165) + ",0)");
    const T1 = R.work, Tinf = I.span;
    d3.select("#dag-readout").html(
      P.name + ", n = " + n + " — " + R.msg + "<br>" +
      "work T₁ = <b>" + T1 + "</b> strands (counted) · span T∞ = <b>" + Tinf + "</b> (longest path, counted in strands) · " +
      "parallelism T₁/T∞ = <b>" + PX.f2(T1 / Tinf) + "</b> · spawns = " + R.spawn + " · widest level = " + maxW);
  }
  sel.addEventListener("change", draw); rng.addEventListener("input", draw); crit.addEventListener("change", draw);
  draw();
})();

/* ═══════════════════════════════════════════════════════════════════════════
   §07 — greedy scheduling: T_p against the lower bounds and Brent's bound
   ═══════════════════════════════════════════════════════════════════════════ */
(function () {
  const svgEl = document.getElementById("sched-svg"); if (!svgEl) return;
  const W = 700, H = 420, PMAX = 16;
  const sel = document.getElementById("sched-prog"), nr = document.getElementById("sched-n"),
        pr = document.getElementById("sched-p"), prio = document.getElementById("sched-prio");
  let cache = { key: null };

  function draw() {
    const P = PA.PROGS[sel.value];
    const n = PX.setRange("sched-n", P.lo, P.hiS, nr.value), p = +pr.value;
    PX.setText("sched-n-val", n); PX.setText("sched-p-val", p);
    const key = sel.value + ":" + n + ":" + prio.value;
    if (cache.key !== key) {
      const R = PA.record(P, n), res = [];
      for (let q = 1; q <= PMAX; q++) { const c = AL.counter(); const g = PA.greedy(R.nodes, q, prio.value, c, R.info); res.push({ p: q, Tp: c.get("step"), g: g, c: c }); }
      cache = { key: key, R: R, res: res };
    }
    const R = cache.R, T1 = R.work, Tinf = R.info.span, cur = cache.res[p - 1];

    const svg = d3.select(svgEl); svg.selectAll("*").remove();
    /* top: T_p against p */
    const m = { l: 48, r: 150, t: 18, b: 26 }, iw = W - m.l - m.r, ih = 190 - m.t - m.b;
    const g = svg.append("g").attr("transform", "translate(" + m.l + "," + m.t + ")");
    const x = d3.scaleLinear().domain([1, PMAX]).range([0, iw]);
    const ymax = T1 + Tinf;
    const y = d3.scaleLog().domain([Math.max(1, Tinf * 0.7), ymax * 1.1]).range([ih, 0]);
    AL.gridY(g, y, iw, 4); AL.axisB(g, x, ih, 8, "processors p"); AL.axisL(g, y, 4, "steps", d3.format("~s"));
    const ps = d3.range(1, PMAX + 1);
    const line = f => d3.line().x(q => x(q)).y(q => y(Math.max(y.domain()[0], f(q))));
    g.append("path").attr("d", line(q => T1 / q)(ps)).attr("fill", "none").attr("stroke", AC.muted).attr("stroke-dasharray", "4 3");
    g.append("path").attr("d", line(() => Tinf)(ps)).attr("fill", "none").attr("stroke", AC.teal).attr("stroke-dasharray", "4 3");
    g.append("path").attr("d", line(q => T1 / q + Tinf)(ps)).attr("fill", "none").attr("stroke", AC.rose).attr("stroke-width", 1.6);
    g.append("path").attr("d", line(q => cache.res[q - 1].Tp)(ps)).attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 2);
    g.selectAll("circle.tp").data(cache.res).join("circle").attr("cx", d => x(d.p)).attr("cy", d => y(d.Tp))
      .attr("r", d => d.p === p ? 5 : 2.5).attr("fill", d => d.p === p ? AC.a2 : AC.accent);
    AL.legend(g, [{ label: "measured T_p (greedy)", color: AC.accent }, { label: "Brent: T₁/p + T∞", color: AC.rose },
                  { label: "work law: T₁/p", color: AC.muted, dash: "4 3" }, { label: "span law: T∞", color: AC.teal, dash: "4 3" }], iw + 14, 10);

    /* bottom: the schedule for the chosen p */
    const gy0 = 200, gh = H - gy0 - 30, gg = svg.append("g").attr("transform", "translate(" + m.l + "," + gy0 + ")");
    const Tp = cur.Tp, cw = (W - m.l - 16) / Tp, rh = Math.min(14, gh / p);
    gg.append("text").attr("x", 0).attr("y", -4).attr("font-size", 11).attr("fill", AC.muted)
      .text("the schedule at p = " + p + ": one column per step, one row per processor — amber columns are incomplete steps (some processor idle)");
    cur.g.steps.forEach((st, t) => {
      const full = st.run.length === p;
      if (!full) gg.append("rect").attr("x", t * cw).attr("y", 2).attr("width", Math.max(1, cw)).attr("height", p * rh)
        .attr("fill", AC.a2).attr("opacity", 0.14);
      st.run.forEach((v, k) => {
        gg.append("rect").attr("x", t * cw + (cw > 3 ? 0.5 : 0)).attr("y", 2 + k * rh + 0.5)
          .attr("width", Math.max(0.8, cw - (cw > 3 ? 1 : 0))).attr("height", rh - 1)
          .attr("fill", R.info.path.has(v) ? AC.a2 : PX.kindColor(R.nodes[v].kind)).attr("opacity", 0.9);
      });
    });
    gg.append("text").attr("x", 0).attr("y", p * rh + 16).attr("font-size", 10.5).attr("fill", AC.muted)
      .text("cells on the critical path are amber; a greedy scheduler never leaves a processor idle while a strand is ready");

    const c = cur.c, lower = Math.max(Math.ceil(T1 / p), Tinf), brent = T1 / p + Tinf;
    d3.select("#sched-readout").html(
      P.name + ", n = " + n + " · T₁ = <b>" + T1 + "</b>, T∞ = <b>" + Tinf + "</b>, parallelism = " + PX.f2(T1 / Tinf) + "<br>" +
      "p = " + p + ": measured T_p = <b>" + c.get("step") + "</b> steps · lower bound max(⌈T₁/p⌉, T∞) = " + lower +
      " · Brent's bound T₁/p + T∞ = " + PX.f2(brent) + " · " + (c.get("step") <= brent + 1e-9 && c.get("step") >= lower ? "inside both ✓" : "OUTSIDE ✗") + "<br>" +
      "complete steps = <b>" + c.get("complete") + "</b> ≤ ⌊T₁/p⌋ = " + Math.floor(T1 / p) + (c.get("complete") <= Math.floor(T1 / p) ? " ✓" : " ✗") +
      " · incomplete steps = <b>" + c.get("incomplete") + "</b> ≤ T∞ = " + Tinf + (c.get("incomplete") <= Tinf ? " ✓" : " ✗") +
      " · speedup T₁/T_p = " + PX.f2(T1 / c.get("step")) + " (" + PX.f2(100 * T1 / c.get("step") / p) + "% efficiency)");
  }
  [sel, prio].forEach(e => e.addEventListener("change", draw));
  [nr, pr].forEach(e => e.addEventListener("input", draw));
  draw();
})();

/* ═══════════════════════════════════════════════════════════════════════════
   §09 — Amdahl vs Gustafson: measured speedups against both laws
   ═══════════════════════════════════════════════════════════════════════════ */
(function () {
  const svgEl = document.getElementById("amdahl-svg"); if (!svgEl) return;
  const W = 700, H = 360;
  const fr = document.getElementById("amdahl-f"), pm = document.getElementById("amdahl-pmax"),
        mode = document.getElementById("amdahl-mode"), ys = document.getElementById("amdahl-y");
  const WORK = 10000, BASE = 1000;

  function strong(f, p, c) { const s = Math.round(f * WORK), N = WORK - s, Tp = PA.runWorkload(s, N, p, c); return { T1: WORK, Tp: Tp, S: WORK / Tp }; }
  function weak(f, p, c) {
    const s = Math.round(f * BASE), N0 = BASE - s, N = p * N0;
    const T1 = PA.runWorkload(s, N, 1), Tp = PA.runWorkload(s, N, p, c);
    return { T1: T1, Tp: Tp, S: T1 / Tp };
  }

  function draw() {
    const f = +fr.value, pmax = +pm.value, md = mode.value;
    PX.setText("amdahl-f-val", f.toFixed(2));
    const ps = []; for (let q = 1; q <= pmax; q *= 2) ps.push(q);
    const F = AL.frame(d3.select(svgEl), W, H, { l: 54, r: 190, t: 16, b: 36 });
    const x = d3.scaleLog().base(2).domain([1, pmax]).range([0, F.iw]);
    const y = ys.value === "log" ? d3.scaleLog().base(2).domain([1, pmax]).range([F.ih, 0])
                                 : d3.scaleLinear().domain([0, pmax]).range([F.ih, 0]);
    AL.gridY(F.g, y, F.iw, 5); AL.axisB(F.g, x, F.ih, Math.min(12, ps.length), "processors p (log scale)", d3.format("d"));
    AL.axisL(F.g, y, 5, "speedup", d3.format("~s"));
    const fine = d3.range(0, Math.log2(pmax) + 0.001, 0.05).map(e => Math.pow(2, e));
    const ln = fn => d3.line().x(q => x(q)).y(q => y(AL.clamp(fn(q), y.domain()[0], pmax)));
    F.g.append("path").attr("d", ln(q => q)(fine)).attr("fill", "none").attr("stroke", AC.muted).attr("stroke-dasharray", "3 3");
    const items = [{ label: "ideal: S = p", color: AC.muted, dash: "3 3" }];
    const cS = AL.counter(), cW = AL.counter();
    let lastS = null, lastW = null;
    if (md !== "weak") {
      F.g.append("path").attr("d", ln(q => 1 / (f + (1 - f) / q))(fine)).attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 1.8);
      const pts = ps.map(q => { const c = AL.counter(); const r = strong(f, q, c); if (q === pmax) { lastS = r; cS.add("step", c.get("step")); } return { p: q, S: r.S }; });
      F.g.selectAll("circle.s").data(pts).join("circle").attr("cx", d => x(d.p)).attr("cy", d => y(Math.max(y.domain()[0], d.S))).attr("r", 3.5).attr("fill", AC.accent);
      if (f > 0 && 1 / f <= pmax) F.g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(1 / f)).attr("y2", y(1 / f)).attr("stroke", AC.accent).attr("stroke-dasharray", "1 4");
      items.push({ label: "Amdahl 1/(f + (1−f)/p)", color: AC.accent }, { label: "measured, fixed size", color: AC.accent, dash: "1 5" });
    }
    if (md !== "strong") {
      F.g.append("path").attr("d", ln(q => q - f * (q - 1))(fine)).attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 1.8);
      const pts = ps.map(q => { const c = AL.counter(); const r = weak(f, q, c); if (q === pmax) { lastW = r; cW.add("step", c.get("step")); } return { p: q, S: r.S }; });
      F.g.selectAll("rect.w").data(pts).join("rect").attr("x", d => x(d.p) - 3.5).attr("y", d => y(Math.max(y.domain()[0], d.S)) - 3.5).attr("width", 7).attr("height", 7).attr("fill", AC.a2);
      items.push({ label: "Gustafson p − f(p − 1)", color: AC.a2 }, { label: "measured, size ∝ p", color: AC.a2, dash: "1 5" });
    }
    AL.legend(F.g, items, F.iw + 16, 12);
    let html = "serial fraction f = <b>" + f.toFixed(2) + "</b> · p = " + pmax + "<br>";
    if (lastS) html += "fixed size (T₁ = " + PX.fmt(lastS.T1) + " unit steps): measured T_p = <b>" + PX.fmt(cS.get("step")) + "</b> steps → speedup <b>" + PX.f2(lastS.S) +
      "</b>; Amdahl predicts " + PX.f2(1 / (f + (1 - f) / pmax)) + "; ceiling 1/f = " + (f > 0 ? PX.f2(1 / f) : "∞") + "<br>";
    if (lastW) html += "scaled size (T₁ = " + PX.fmt(lastW.T1) + " unit steps): measured T_p = <b>" + PX.fmt(cW.get("step")) + "</b> steps → scaled speedup <b>" + PX.f2(lastW.S) +
      "</b>; Gustafson predicts " + PX.f2(pmax - f * (pmax - 1));
    d3.select("#amdahl-readout").html(html);
  }
  fr.addEventListener("input", draw); [pm, mode, ys].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══════════════════════════════════════════════════════════════════════════
   §11 — measured work and span against n, for every routine on the page
   ═══════════════════════════════════════════════════════════════════════════ */
(function () {
  const svgEl = document.getElementById("growth-svg"); if (!svgEl) return;
  const W = 700, H = 360;
  const sel = document.getElementById("growth-alg"), nr = document.getElementById("growth-n");
  const memo = {};
  function series(key) {
    if (!memo[key]) memo[key] = PA.GROW[key].ns.map(n => PA.measure(PA.GROW[key], n));
    return memo[key];
  }
  function draw() {
    const G = PA.GROW[sel.value], S = series(sel.value);
    const k = PX.setRange("growth-n", 0, S.length - 1, nr.value), pt = S[k];
    PX.setText("growth-n-val", "n = " + pt.n);
    const F = AL.frame(d3.select(svgEl), W, H, { l: 58, r: 170, t: 16, b: 36 });
    const x = d3.scaleLog().base(2).domain([S[0].n, S[S.length - 1].n]).range([0, F.iw]);
    const ymax = d3.max(S, d => d.work), y = d3.scaleLog().domain([1, ymax * 1.5]).range([F.ih, 0]);
    AL.gridY(F.g, y, F.iw, 5); AL.axisB(F.g, x, F.ih, S.length, "n (log scale)", d3.format("d")); AL.axisL(F.g, y, 5, "unit steps (log scale)", d3.format("~s"));
    const lines = [{ k: "work", c: AC.accent, lab: "work T₁   (Θ(" + G.W + "))" }, { k: "span", c: AC.teal, lab: "span T∞   (Θ(" + G.S + "))" },
                   { k: "par", c: AC.a2, lab: "parallelism T₁/T∞" }];
    const v = (d, key) => key === "par" ? d.work / d.span : d[key];
    lines.forEach(L => {
      F.g.append("path").attr("d", d3.line().x(d => x(d.n)).y(d => y(Math.max(1, v(d, L.k))))(S)).attr("fill", "none").attr("stroke", L.c).attr("stroke-width", 2);
      F.g.selectAll(null).data(S).join("circle").attr("cx", d => x(d.n)).attr("cy", d => y(Math.max(1, v(d, L.k)))).attr("r", d => d === pt ? 5 : 2.5).attr("fill", L.c);
    });
    F.g.append("line").attr("x1", x(pt.n)).attr("x2", x(pt.n)).attr("y1", 0).attr("y2", F.ih).attr("stroke", AC.muted).attr("stroke-dasharray", "2 3");
    AL.legend(F.g, lines.map(L => ({ label: L.lab, color: L.c })), F.iw + 14, 12);
    const ratio = d => d.span / G.sf(d.n), wratio = d => d.work / G.wf(d.n);
    const prev = k > 0 ? S[k - 1] : null;
    d3.select("#growth-readout").html(
      G.name + ", n = " + pt.n + " — output checked against the serial routine: " + (pt.ok ? "<b>correct ✓</b>" : "<b>WRONG ✗</b>") + "<br>" +
      "work T₁ = <b>" + PX.fmt(pt.work) + "</b> · span T∞ = <b>" + PX.fmt(pt.span) + "</b> · parallelism = <b>" + PX.fmt(Math.round(pt.work / pt.span)) + "</b> · spawns = " + PX.fmt(pt.spawn) + "<br>" +
      "T₁ / " + G.W + " = " + PX.f2(wratio(pt)) + (prev ? " (was " + PX.f2(wratio(prev)) + " at n = " + prev.n + ")" : "") +
      " · T∞ / " + G.S + " = " + PX.f2(ratio(pt)) + (prev ? " (was " + PX.f2(ratio(prev)) + ")" : "") +
      (G.S2 ? " · T∞ / " + G.S2 + " = " + PX.f2(pt.span / G.sf2(pt.n)) + (prev ? " (was " + PX.f2(prev.span / G.sf2(prev.n)) + ")" : "") +
        " — at these sizes the lower-order log² n terms are still as large as the leading one (about log³ n / 6 when splits are even), so the first ratio is still falling toward its constant and the second still rising"
        : " — a ratio that levels off toward a constant is the Θ-bound showing through"));
  }
  sel.addEventListener("change", draw); nr.addEventListener("input", draw);
  draw();
})();

/* ═══════════════════════════════════════════════════════════════════════════
   §12 — determinacy races: every interleaving, enumerated
   ═══════════════════════════════════════════════════════════════════════════ */
(function () {
  const svgEl = document.getElementById("race-svg"); if (!svgEl) return;
  const W = 700, H = 330;
  const md = document.getElementById("race-mode"), cf = document.getElementById("race-cfg"), ir = document.getElementById("race-i");
  let cache = { key: null };
  function draw() {
    const key = md.value + ":" + cf.value;
    if (cache.key !== key) {
      const [k, inc] = cf.value.split("x").map(Number), c = AL.counter();
      const res = PA.interleavings(k, PA.program(md.value, inc), c);
      cache = { key: key, k: k, inc: inc, res: res, c: c };
    }
    const { k, inc, res, c } = cache;
    const idx = PX.setRange("race-i", 0, res.kept.length - 1, ir.value);
    PX.setText("race-i-val", "#" + (idx + 1));
    const svg = d3.select(svgEl); svg.selectAll("*").remove();
    /* left: histogram of final x */
    const m = { l: 44, t: 24, b: 34 }, hw = 190, hh = H - m.t - m.b;
    const g = svg.append("g").attr("transform", "translate(" + m.l + "," + m.t + ")");
    const vals = d3.range(1, k * inc + 1), total = c.get("interleaving");
    const x = d3.scaleBand().domain(vals).range([0, hw]).padding(0.25);
    const y = d3.scaleLinear().domain([0, d3.max(vals, v => res.hist[v] || 0) || 1]).nice().range([hh, 0]);
    AL.axisB(g, x, hh, vals.length, "final x"); AL.axisL(g, y, 4, "interleavings", d3.format("~s"));
    g.selectAll("rect").data(vals).join("rect").attr("x", v => x(v)).attr("width", x.bandwidth())
      .attr("y", v => y(res.hist[v] || 0)).attr("height", v => hh - y(res.hist[v] || 0))
      .attr("fill", v => v === k * inc ? AC.good : AC.bad);
    /* right: the chosen interleaving as a timeline */
    const it = res.kept[idx], tx = 300, tw = W - tx - 16, steps = it.trace.length, sw = tw / steps;
    const tg = svg.append("g").attr("transform", "translate(" + tx + "," + m.t + ")");
    tg.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text("interleaving #" + (idx + 1) + " of " + PX.fmt(total) + " — time runs left → right");
    const lh = 34;
    for (let t = 0; t < k; t++) tg.append("text").attr("x", -6).attr("y", 18 + t * lh + 12).attr("text-anchor", "end").attr("font-size", 11).attr("fill", AC.muted).text("");
    it.trace.forEach((e, s) => {
      const bx = s * sw, by = 18 + e.t * lh;
      const col = e.op === "store" || e.op === "faa" ? AC.a2 : e.op === "load" ? AC.accent : e.op === "add" ? AC.violet : AC.teal;
      tg.append("rect").attr("x", bx + 1).attr("y", by).attr("width", Math.max(2, sw - 2)).attr("height", 24).attr("rx", 3).attr("fill", col).attr("opacity", 0.85)
        .append("title").text("strand " + (e.t + 1) + ": " + e.op + " → x = " + e.x);
      if (sw > 22) tg.append("text").attr("x", bx + sw / 2).attr("y", by + 16).attr("text-anchor", "middle").attr("font-size", sw > 40 ? 10 : 8.5).attr("fill", AC.bg)
        .text(e.op === "acquire" ? "lock" : e.op === "release" ? "unlk" : e.op === "faa" ? "x+=1" : e.op === "store" ? "st" : e.op === "load" ? "ld" : "+1");
    });
    for (let t = 0; t < k; t++) tg.append("text").attr("x", 0).attr("y", 18 + t * lh - 2).attr("font-size", 9.5).attr("fill", AC.muted).text("strand " + (t + 1));
    const yX = 18 + k * lh + 18;
    tg.append("text").attr("x", 0).attr("y", yX - 4).attr("font-size", 10).attr("fill", AC.muted).text("shared x after each instruction");
    it.trace.forEach((e, s) => tg.append("text").attr("x", s * sw + sw / 2).attr("y", yX + 12).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.ink).text(e.x));
    tg.append("text").attr("x", 0).attr("y", yX + 44).attr("font-size", 12).attr("fill", it.x === k * inc ? AC.good : AC.bad)
      .text("final x = " + it.x + (it.x === k * inc ? " — the serial answer" : " — an update was lost"));
    AL.legend(tg, [{ label: "load", color: AC.accent }, { label: "add in register", color: AC.violet }, { label: "store / atomic add", color: AC.a2 }, { label: "lock / unlock", color: AC.teal }], 0, yX + 72, { gap: 0 })
      .selectAll("g").attr("transform", (d, i) => "translate(" + (i * 100) + ",0)");
    const right = res.hist[k * inc] || 0;
    d3.select("#race-readout").html(
      k + " strands × " + inc + " increment" + (inc > 1 ? "s" : "") + ", " + md.options[md.selectedIndex].text + ": enumerated <b>" + PX.fmt(total) + "</b> interleavings (" +
      PX.fmt(c.get("instr")) + " instructions simulated) · correct final x = " + (k * inc) + " in <b>" + PX.fmt(right) + "</b> of them (" + PX.f2(100 * right / total) + "%) · " +
      "distinct outcomes: " + Object.keys(res.hist).sort().join(", ") + (c.get("deadlock") ? " · deadlocked paths: " + c.get("deadlock") : ""));
  }
  [md, cf].forEach(e => e.addEventListener("change", draw)); ir.addEventListener("input", draw);
  draw();
})();

/* ═══════════════════════════════════════════════════════════════════════════
   §15 — the scan, animated: up-sweep / down-sweep against the naive rounds
   ═══════════════════════════════════════════════════════════════════════════ */
(function () {
  const svgEl = document.getElementById("scan-svg"); if (!svgEl) return;
  const W = 700, H = 300;
  const alg = document.getElementById("scan-alg"), nsel = document.getElementById("scan-n");
  function build() {
    const n = +nsel.value, A = Array.from({ length: n }, (_, i) => [3, 1, 7, 0, 4, 1, 6, 3, 2, 5, 1, 4, 8, 2, 3, 1][i]);
    const c = AL.counter(), R = PA.scanFrames(A, alg.value, c);
    const want = PA.serialScan(A, alg.value === "naive");
    const ok = JSON.stringify(R.result) === JSON.stringify(want);
    d3.select(svgEl.parentNode).selectAll("div[role=group]").remove();
    const cellW = Math.min(36, (W - 60) / n - 3), step = cellW + 3, x0 = (W - n * step) / 2;
    function render(f, i) {
      const svg = d3.select(svgEl); svg.selectAll("*").remove();
      const g = svg.append("g");
      g.append("text").attr("x", 16).attr("y", 20).attr("font-size", 12.5).attr("fill", AC.ink).text(f.title);
      const targets = new Set(f.arrows.map(a => a[1]));
      const srcs = new Set(f.arrows.map(a => a[0]));
      AL.row(g, f.before, { x: x0, y: 50, w: cellW, h: 30, gap: 3, index: false, label: "", mark: j => srcs.has(j) ? "rgba(91,156,255,.28)" : null });
      AL.row(g, f.after, { x: x0, y: 190, w: cellW, h: 30, gap: 3, index: true,
        mark: j => targets.has(j) ? "rgba(255,180,84,.35)" : (f.clear === j ? "rgba(248,113,113,.35)" : null) });
      g.append("text").attr("x", x0 - 6).attr("y", 69).attr("text-anchor", "end").attr("font-size", 10.5).attr("fill", AC.muted).text("before");
      g.append("text").attr("x", x0 - 6).attr("y", 209).attr("text-anchor", "end").attr("font-size", 10.5).attr("fill", AC.muted).text("after");
      f.arrows.forEach(a => {
        const xa = x0 + a[0] * step + cellW / 2, xb = x0 + a[1] * step + cellW / 2;
        AL.arrow(g, xa, 82, xb, 186, { color: a[2] === "copy" ? AC.teal : AC.a2, w: 1.4, head: 5 });
      });
      const last = i === R.frames.length - 1;
      d3.select("#scan-readout").html(
        (alg.value === "naive" ? "naive (inclusive) scan" : "work-efficient (exclusive) scan") + ", n = " + n +
        " · this frame: " + f.arrows.filter(a => a[2] === "add").length + " additions in parallel" +
        " · so far: <b>" + (f.adds || 0) + "</b> additions in <b>" + (f.rounds || 0) + "</b> parallel rounds" + (f.total !== undefined ? " · saved total = " + f.total : "") + "<br>" +
        "whole run (counted): <b>" + c.get("add") + "</b> additions, <b>" + c.get("round") + "</b> rounds — against n − 1 = " + (n - 1) + " additions serially" +
        (last ? " · result matches the serial " + (alg.value === "naive" ? "inclusive" : "exclusive") + " scan: " + (ok ? "<b>✓</b>" : "<b>✗</b>") : ""));
    }
    AL.stepper(d3.select(svgEl), { frames: R.frames, render: render, delay: 900, label: "frame" });
  }
  [alg, nsel].forEach(e => e.addEventListener("change", build));
  build();
})();

/* ═══════════════════════════════════════════════════════════════════════════
   load-time audit — the two tracers agree, and greedy obeys both bounds
   ═══════════════════════════════════════════════════════════════════════════ */
(function () {
  const out = document.getElementById("audit-readout"); if (!out) return;
  const c = AL.counter();
  Object.keys(PA.PROGS).forEach(k => {
    const P = PA.PROGS[k];
    for (let n = P.lo; n <= P.hi; n++) {
      const ca = AL.counter(), Xa = PA.algebra(ca); P.run(Xa, n);
      const R = PA.record(P, n);
      c.add("pairs");
      if (ca.get("work") !== R.work || Xa.span() !== R.info.span) c.add("mismatch");
      for (const p of [1, 2, 3, 5, 8]) {
        const g = PA.greedy(R.nodes, p, "short", null, R.info);
        c.add("schedules");
        if (g.Tp > R.work / p + R.info.span + 1e-9 || g.Tp < Math.max(Math.ceil(R.work / p), R.info.span)) c.add("violations");
      }
    }
  });
  out.innerHTML = "audit: " + c.get("pairs") + " (program, n) pairs traced both ways — work and span disagree in <b>" + c.get("mismatch") + "</b>; " +
    c.get("schedules") + " greedy schedules (the deliberately naive shortest-path-first rule) — outside [max(⌈T₁/p⌉, T∞), T₁/p + T∞] in <b>" + c.get("violations") + "</b>";
  const cell = document.getElementById("audit-cell");
  if (cell) cell.textContent = c.get("mismatch") === 0 && c.get("violations") === 0 ? "passed: " + c.get("pairs") + " traced pairs, " + c.get("schedules") + " schedules" : "FAILED — see readout";
})();
