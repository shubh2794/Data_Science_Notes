/* greedy.viz.js — figures for dsa/algorithms/greedy.html
   (part 4 of the Algorithm Design & Analysis series).

   Loaded after ../../data.js, ../../notes.js and ../dsa-viz.js, so the globals
   AC (palette) and AL (rng/frame/axisB/axisL/row/binTree/counter/stepper/…) are
   available.

   House rule obeyed throughout: every count this page DISPLAYS — comparisons,
   intervals examined, priority-queue operations, exchange steps, coins used,
   sets picked — is produced by running the real routine under AL.counter() and
   reading the counter back. Every greedy ANSWER is cross-checked against an
   independent computation in the same figure — brute force over all subsets /
   all orders for small n, a dynamic program, an exhaustive tree enumeration,
   the entropy bound — and printed with an agree / DISAGREE flag. A load-time
   audit (GA, at the bottom) repeats those cross-checks over hundreds of random
   instances and reports the counts in the cheat sheet.

   Layout of this file:
     GR — the instrumented routines, pure and DOM-free.
     SX — small DOM helpers shared by the figures.
     figures — one block per <svg>, in page order:
       01 #as-svg     activity selection stepped on a timeline, four rules, brute-force optimum
       02 #rules-svg  failure rate of the four rules on random instances
       03 #ip-svg     interval partitioning stepped, depth marked
       04 #late-svg   minimise maximum lateness: inversions exchanged one at a time
       05 #coin-svg   coin change greedy vs DP over amounts 1–100, two systems
       06 #knap-svg   fractional vs 0/1 knapsack on the same items, greedy vs brute force
       07 #huff-svg   Huffman merges stepped, the tree, the codes, L vs H
       08 #ent-svg    Huffman vs fixed-length vs entropy as a distribution's skew varies
       09 #mat-svg    a graphic matroid: independent sets, the exchange axiom, greedy vs brute force
       10 #task-svg   unit-task scheduling with deadlines as a matroid
       11 #cut-svg    the cut property on a small weighted graph
       12 #fuel-svg   fuel stops along a road: greedy stays ahead
       13 #sc-svg     set cover greedy stepped vs optimal, ratio measured, the tight family
       14 #lb-svg     list scheduling on identical machines vs brute-force optimum
       15 #tsp-svg    nearest-neighbour tour vs optimal, metric and non-metric
       16 #flow-svg   the greedy-vs-DP decision flow
     GA — the load-time audit. */

/* ═══════════════════════════════════════════════════════════════════════════
   GR — the instrumented routines
   ═══════════════════════════════════════════════════════════════════════════ */
const GR = (function () {
  const nop = { add: function () {}, get: function () { return 0; } };

  /* ── activity selection ─────────────────────────────────────────────── */
  /* activities: [{id, s, f}], half-open [s, f). compatible ⇔ s ≥ f' or s' ≥ f */
  function compat(a, b) { return a.s >= b.f || b.s >= a.f; }

  /* Earliest-finish greedy as the one-sort-one-scan routine. c.add("cmp") per
     "s ≥ last" test. rec(frame) after every decision. */
  function selectEF(acts, c, rec) {
    c = c || nop;
    const sorted = acts.slice().sort((a, b) => a.f - b.f || a.id - b.id);
    const chosen = []; let last = -Infinity;
    sorted.forEach((a, k) => {
      let keep;
      if (k === 0) keep = true;                       // first always kept, no comparison
      else { c.add("cmp"); keep = a.s >= last; }
      if (keep) { chosen.push(a); last = a.f; }
      if (rec) rec({ cand: a, keep, chosen: chosen.slice(), last });
    });
    return chosen;
  }

  /* The generic "pick by rule, discard conflicts, repeat" greedy, for the
     three wrong rules as well as the right one. c.add("exam") per candidate
     examined when choosing; c.add("cmp") per compatibility test. */
  function selectByRule(acts, rule, c, rec) {
    c = c || nop;
    let rem = acts.slice(); const chosen = [];
    while (rem.length) {
      let pick = null, key = Infinity;
      rem.forEach(a => {
        c.add("exam");
        let k;
        if (rule === "ef") k = a.f;
        else if (rule === "es") k = a.s;
        else if (rule === "sd") k = a.f - a.s;
        else { k = 0; rem.forEach(b => { if (b !== a) { c.add("cmp"); if (!compat(a, b)) k++; } }); }
        if (k < key || (k === key && a.id < pick.id)) { key = k; pick = a; }
      });
      chosen.push(pick);
      const skipped = [];
      rem = rem.filter(b => { if (b === pick) return false; c.add("cmp"); const ok = compat(pick, b); if (!ok) skipped.push(b); return ok; });
      if (rec) rec({ cand: pick, keep: true, chosen: chosen.slice(), skipped, key });
    }
    return chosen;
  }

  /* Brute force: largest pairwise-compatible subset, all 2^n subsets. */
  function bruteSelect(acts) {
    const n = acts.length; let best = [];
    for (let mask = 0; mask < (1 << n); mask++) {
      const sub = []; for (let i = 0; i < n; i++) if (mask & (1 << i)) sub.push(acts[i]);
      if (sub.length <= best.length) continue;
      let ok = true;
      for (let i = 0; i < sub.length && ok; i++) for (let j = i + 1; j < sub.length; j++) if (!compat(sub[i], sub[j])) { ok = false; break; }
      if (ok) best = sub;
    }
    return best;
  }

  /* ── interval partitioning ──────────────────────────────────────────── */
  /* Sort by start; assign each interval to a room that is free (its last
     interval finished at or before s), else open a new room. Returns the
     assignment and the number of rooms. c.add("cmp") per room checked. */
  function partition(acts, c, rec) {
    c = c || nop;
    const sorted = acts.slice().sort((a, b) => a.s - b.s || a.f - b.f || a.id - b.id);
    const rooms = [];                                   // finish time of the last interval in each room
    const assign = new Map();
    sorted.forEach(a => {
      let r = -1;
      for (let k = 0; k < rooms.length; k++) { c.add("cmp"); if (rooms[k] <= a.s) { r = k; break; } }
      if (r < 0) { rooms.push(a.f); r = rooms.length - 1; c.add("open"); } else rooms[r] = a.f;
      assign.set(a.id, r);
      if (rec) rec({ cand: a, room: r, opened: rooms.length, assign: new Map(assign) });
    });
    return { rooms: rooms.length, assign };
  }
  /* depth = max number of intervals containing a common point (half-open) */
  function depth(acts) {
    const pts = acts.map(a => a.s); let d = 0;
    pts.forEach(p => { let k = 0; acts.forEach(a => { if (a.s <= p && p < a.f) k++; }); d = Math.max(d, k); });
    return d;
  }

  /* ── minimise maximum lateness ──────────────────────────────────────── */
  /* jobs: [{id, t, d}] processing time t, deadline d. A schedule is an order. */
  function lateness(order) {
    let time = 0, maxL = 0, who = null; const rows = [];
    order.forEach(j => { time += j.t; const L = Math.max(0, time - j.d); rows.push({ id: j.id, f: time, L }); if (L > maxL) { maxL = L; who = j.id; } });
    return { maxL, who, rows, makespan: time };
  }
  function inversions(order) {
    const out = [];
    for (let i = 0; i < order.length; i++) for (let j = i + 1; j < order.length; j++) if (order[i].d > order[j].d) out.push([i, j]);
    return out;
  }
  /* Remove inversions by adjacent swaps, recording every swap and the max
     lateness after it. c.add("swap") per exchange. */
  function uninvert(order, c, rec) {
    c = c || nop;
    const o = order.slice();
    if (rec) rec({ order: o.slice(), swapped: null, L: lateness(o), inv: inversions(o).length });
    let changed = true;
    while (changed) {
      changed = false;
      for (let i = 0; i + 1 < o.length; i++) {
        if (o[i].d > o[i + 1].d) {
          const before = lateness(o).maxL;
          const t = o[i]; o[i] = o[i + 1]; o[i + 1] = t; c.add("swap");
          const after = lateness(o);
          if (rec) rec({ order: o.slice(), swapped: [i, i + 1], L: after, inv: inversions(o).length, before });
          changed = true;
        }
      }
    }
    return o;
  }
  function edf(jobs) { return jobs.slice().sort((a, b) => a.d - b.d || a.id.localeCompare(b.id)); }
  function permutations(arr) {
    const out = [];
    (function go(a, k) { if (k === a.length) { out.push(a.slice()); return; } for (let i = k; i < a.length; i++) { [a[k], a[i]] = [a[i], a[k]]; go(a, k + 1); [a[k], a[i]] = [a[i], a[k]]; } })(arr.slice(), 0);
    return out;
  }
  function bruteLateness(jobs) {
    let best = Infinity, arg = null;
    permutations(jobs).forEach(p => { const L = lateness(p).maxL; if (L < best) { best = L; arg = p; } });
    return { maxL: best, order: arg };
  }

  /* ── coin change ────────────────────────────────────────────────────── */
  function greedyCoins(coins, x, c) {
    c = c || nop;
    const sorted = coins.slice().sort((a, b) => b - a); const out = [];
    for (const d of sorted) { const k = Math.floor(x / d); c.add("cmp"); for (let i = 0; i < k; i++) { out.push(d); c.add("coin"); } x -= k * d; }
    return x === 0 ? out : null;
  }
  function dpCoins(coins, x, c) {
    c = c || nop;
    const INF = 1e9, best = new Array(x + 1).fill(INF), pick = new Array(x + 1).fill(0); best[0] = 0;
    for (let a = 1; a <= x; a++) for (const d of coins) { c.add("cell"); if (d <= a && best[a - d] + 1 < best[a]) { best[a] = best[a - d] + 1; pick[a] = d; } }
    if (best[x] >= INF) return null;
    const out = []; let a = x; while (a) { out.push(pick[a]); a -= pick[a]; }
    return out;
  }

  /* ── knapsack ───────────────────────────────────────────────────────── */
  /* items: [{id, w, v}] */
  function fractionalKnapsack(items, W, c) {
    c = c || nop;
    const sorted = items.slice().sort((a, b) => b.v / b.w - a.v / a.w || a.id.localeCompare(b.id));
    let cap = W, val = 0; const take = [];
    for (const it of sorted) {
      if (cap <= 0) break;                          // full: nothing further is examined
      c.add("item");
      if (it.w <= cap) { cap -= it.w; val += it.v; take.push({ id: it.id, frac: 1 }); }
      else { const fr = cap / it.w; val += it.v * fr; take.push({ id: it.id, frac: fr }); cap = 0; }
    }
    return { val, take, order: sorted };
  }
  function greedy01(items, W, c) {
    c = c || nop;
    const sorted = items.slice().sort((a, b) => b.v / b.w - a.v / a.w || a.id.localeCompare(b.id));
    let cap = W, val = 0; const take = [];
    for (const it of sorted) { c.add("item"); if (it.w <= cap) { cap -= it.w; val += it.v; take.push(it.id); } }
    return { val, take };
  }
  function brute01(items, W) {
    const n = items.length; let best = 0, arg = [];
    for (let mask = 0; mask < (1 << n); mask++) {
      let w = 0, v = 0; const ids = [];
      for (let i = 0; i < n; i++) if (mask & (1 << i)) { w += items[i].w; v += items[i].v; ids.push(items[i].id); }
      if (w <= W && v > best) { best = v; arg = ids; }
    }
    return { val: best, take: arg };
  }
  function bestSingle(items, W) { let b = 0; items.forEach(it => { if (it.w <= W && it.v > b) b = it.v; }); return b; }

  /* ── Huffman ────────────────────────────────────────────────────────── */
  /* A minimal binary min-heap keyed by (weight, birth order) so ties are
     deterministic. Counts every insert / extractMin as a PQ operation. */
  function huffman(freq, c, rec) {                    // freq: [{sym, w}]
    c = c || nop;
    const heap = []; let birth = 0;
    const less = (a, b) => a.w < b.w || (a.w === b.w && a.birth < b.birth);
    const push = nd => { c.add("pq"); heap.push(nd); let i = heap.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (less(heap[i], heap[p])) { [heap[i], heap[p]] = [heap[p], heap[i]]; i = p; } else break; } };
    const pop = () => { c.add("pq"); const top = heap[0]; const last = heap.pop(); if (heap.length) { heap[0] = last; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < heap.length && less(heap[l], heap[m])) m = l; if (r < heap.length && less(heap[r], heap[m])) m = r; if (m === i) break; [heap[i], heap[m]] = [heap[m], heap[i]]; i = m; } } return top; };
    freq.forEach(f => push({ sym: f.sym, w: f.w, birth: birth++, leaf: true }));
    if (heap.length === 1) return heap[0];
    const merges = [];
    while (heap.length > 1) {
      const x = pop(), y = pop();
      const z = { w: x.w + y.w, birth: birth++, left: x, right: y, leaf: false };
      merges.push({ x, y, z });
      push(z);
      if (rec) rec({ x, y, z, pending: heap.slice().sort((a, b) => a.w - b.w || a.birth - b.birth) });
    }
    return heap[0];
  }
  function codes(tree) {
    const out = {};
    /* a lone leaf gets the empty codeword (length 0) */
    (function walk(t, p) { if (t.leaf) { out[t.sym] = p; return; } walk(t.left, p + "0"); walk(t.right, p + "1"); })(tree, "");
    return out;
  }
  function costOf(tree) {                             // ∑ w · depth
    let tot = 0; (function walk(t, d) { if (t.leaf) { tot += t.w * d; return; } walk(t.left, d + 1); walk(t.right, d + 1); })(tree, 0); return tot;
  }
  function entropy(freq) {
    const tot = freq.reduce((s, f) => s + f.w, 0); let H = 0;
    freq.forEach(f => { if (f.w > 0) { const p = f.w / tot; H -= p * Math.log2(p); } });
    return H;
  }
  /* All full binary trees with the given labelled leaves: (2n − 3)!! of them.
     Built by inserting each leaf on every edge of every tree of the rest. */
  function allTrees(leaves) {
    if (leaves.length === 1) return [{ leaf: true, sym: leaves[0].sym, w: leaves[0].w }];
    const first = leaves[0], rest = allTrees(leaves.slice(1)), out = [];
    const ins = (t, x) => { const res = [{ leaf: false, left: x, right: t }]; if (!t.leaf) { ins(t.left, x).forEach(l => res.push({ leaf: false, left: l, right: t.right })); ins(t.right, x).forEach(r => res.push({ leaf: false, left: t.left, right: r })); } return res; };
    rest.forEach(t => ins(t, { leaf: true, sym: first.sym, w: first.w }).forEach(u => out.push(u)));
    return out;
  }
  function bruteHuffmanCost(freq) { let best = Infinity; allTrees(freq).forEach(t => { const cst = costOf(t); if (cst < best) best = cst; }); return best; }

  /* ── matroids ───────────────────────────────────────────────────────── */
  /* graphic matroid: edges [{id,u,v,w}]; independent ⇔ acyclic (union-find) */
  function acyclic(edges, c) {
    c = c || nop;
    const par = {}; const find = x => { while (par[x] !== undefined && par[x] !== x) x = par[x]; return x; };
    for (const e of edges) {
      c.add("find");
      const a = find(e.u), b = find(e.v);
      if (a === b) return false;
      par[a] = a; par[b] = b; par[a] = b;
    }
    return true;
  }
  function matroidGreedy(elems, indep, c, rec) {     // max-weight independent set, weights ≥ 0
    c = c || nop;
    const sorted = elems.slice().sort((a, b) => b.w - a.w || a.id.localeCompare(b.id)); const A = [];
    sorted.forEach(x => { c.add("test"); const ok = indep(A.concat([x]), c); if (ok) A.push(x); if (rec) rec({ x, ok, A: A.slice() }); });
    return A;
  }
  function bruteMatroid(elems, indep) {
    const n = elems.length; let best = -1, arg = [];
    for (let mask = 0; mask < (1 << n); mask++) {
      const sub = []; for (let i = 0; i < n; i++) if (mask & (1 << i)) sub.push(elems[i]);
      const w = sub.reduce((s, e) => s + e.w, 0);
      if (w > best && indep(sub)) { best = w; arg = sub; }
    }
    return { w: best, set: arg };
  }
  /* unit-task scheduling: tasks [{id, d, w}]; A independent ⇔ N_t(A) ≤ t ∀t */
  function tasksIndep(A, c) {
    c = c || nop;
    const n = Math.max(1, ...A.map(a => a.d));
    for (let t = 1; t <= n; t++) { c.add("count"); let k = 0; A.forEach(a => { if (a.d <= t) k++; }); if (k > t) return false; }
    return true;
  }
  function penaltyOf(order) { let p = 0; order.forEach((t, i) => { if (i + 1 > t.d) p += t.w; }); return p; }

  /* ── set cover ──────────────────────────────────────────────────────── */
  function greedyCover(U, sets, c, rec) {             // sets: [{id, els:Set}]
    c = c || nop;
    const unc = new Set(U); const picked = [];
    while (unc.size) {
      let best = null, gain = -1;
      sets.forEach(S => { c.add("scan"); let g = 0; S.els.forEach(e => { if (unc.has(e)) g++; }); if (g > gain || (g === gain && S.id < best.id)) { gain = g; best = S; } });
      if (gain === 0) break;                          // uncoverable — cannot happen when ∪ = U
      picked.push({ id: best.id, gain });
      best.els.forEach(e => unc.delete(e));
      if (rec) rec({ pick: best, gain, unc: new Set(unc), picked: picked.slice() });
    }
    return picked;
  }
  function bruteCover(U, sets) {
    const n = sets.length;
    for (let r = 1; r <= n; r++) {
      for (let mask = 0; mask < (1 << n); mask++) {
        let bits = 0; for (let i = 0; i < n; i++) if (mask & (1 << i)) bits++;
        if (bits !== r) continue;
        const cov = new Set(); for (let i = 0; i < n; i++) if (mask & (1 << i)) sets[i].els.forEach(e => cov.add(e));
        if (U.every(e => cov.has(e))) { const ids = []; for (let i = 0; i < n; i++) if (mask & (1 << i)) ids.push(sets[i].id); return ids; }
      }
    }
    return null;
  }
  function harmonic(n) { let h = 0; for (let i = 1; i <= n; i++) h += 1 / i; return h; }

  /* ── load balancing ─────────────────────────────────────────────────── */
  function listSchedule(jobs, m, c, rec) {
    c = c || nop;
    const load = new Array(m).fill(0), asg = [];
    jobs.forEach((p, i) => { let k = 0; for (let j = 1; j < m; j++) { c.add("cmp"); if (load[j] < load[k]) k = j; } load[k] += p; asg.push(k); if (rec) rec({ i, k, load: load.slice() }); });
    return { makespan: Math.max(...load), load, asg };
  }
  function bruteMakespan(jobs, m) {
    const n = jobs.length; let best = Infinity, arg = null;
    const total = Math.pow(m, n);
    for (let code = 0; code < total; code++) {
      const load = new Array(m).fill(0); let x = code; const a = [];
      for (let i = 0; i < n; i++) { const k = x % m; x = Math.floor(x / m); load[k] += jobs[i]; a.push(k); }
      const mk = Math.max(...load); if (mk < best) { best = mk; arg = a; }
    }
    return { makespan: best, asg: arg };
  }

  /* ── fuel stops ─────────────────────────────────────────────────────── */
  function greedyFuel(stations, R, dest, c, rec) {   // stations: sorted positions strictly between 0 and dest
    c = c || nop;
    let pos = 0; const stops = [];
    while (pos + R < dest) {
      let far = -1; stations.forEach(s => { c.add("cmp"); if (s > pos && s <= pos + R && s > far) far = s; });
      if (far < 0) return null;                       // stranded
      stops.push(far); pos = far;
      if (rec) rec({ pos, stops: stops.slice() });
    }
    return stops;
  }
  function bruteFuel(stations, R, dest) {
    const n = stations.length;
    for (let r = 0; r <= n; r++) for (let mask = 0; mask < (1 << n); mask++) {
      let bits = 0; for (let i = 0; i < n; i++) if (mask & (1 << i)) bits++;
      if (bits !== r) continue;
      const pts = [0]; for (let i = 0; i < n; i++) if (mask & (1 << i)) pts.push(stations[i]); pts.push(dest);
      let ok = true; for (let i = 0; i + 1 < pts.length; i++) if (pts[i + 1] - pts[i] > R) { ok = false; break; }
      if (ok) return pts.slice(1, -1);
    }
    return null;
  }

  /* ── TSP ────────────────────────────────────────────────────────────── */
  function nearestNeighbour(n, dist, start, c) {
    c = c || nop;
    const left = new Set(); for (let i = 0; i < n; i++) if (i !== start) left.add(i);
    const tour = [start]; let cur = start;
    while (left.size) { let nx = -1, best = Infinity; left.forEach(j => { c.add("cmp"); const d = dist(cur, j); if (d < best || (d === best && j < nx)) { best = d; nx = j; } }); tour.push(nx); left.delete(nx); cur = nx; }
    return tour;
  }
  function tourLength(tour, dist) { let L = 0; for (let i = 0; i < tour.length; i++) L += dist(tour[i], tour[(i + 1) % tour.length]); return L; }
  function bruteTSP(n, dist) {
    const rest = []; for (let i = 1; i < n; i++) rest.push(i);
    let best = Infinity, arg = null;
    permutations(rest).forEach(p => { const t = [0].concat(p); const L = tourLength(t, dist); if (L < best) { best = L; arg = t; } });
    return { len: best, tour: arg };
  }

  /* ── Horn formulas ──────────────────────────────────────────────────── */
  /* impl: [{body:[vars], head:var}], neg: [[vars]] meaning ¬v₁ ∨ ¬v₂ ∨ … */
  function horn(vars, impl, neg, c, rec) {
    c = c || nop;
    const T = new Set(); let changed = true;
    while (changed) {
      changed = false;
      for (const cl of impl) { c.add("check"); if (cl.body.every(v => T.has(v)) && !T.has(cl.head)) { T.add(cl.head); changed = true; if (rec) rec({ set: cl.head, T: new Set(T), by: cl }); } }
    }
    const ok = neg.every(cl => cl.some(v => !T.has(v)));
    return { sat: ok, T };
  }
  function bruteHorn(vars, impl, neg) {
    const n = vars.length;
    for (let mask = 0; mask < (1 << n); mask++) {
      const T = new Set(); vars.forEach((v, i) => { if (mask & (1 << i)) T.add(v); });
      if (impl.every(cl => !cl.body.every(v => T.has(v)) || T.has(cl.head)) && neg.every(cl => cl.some(v => !T.has(v)))) return T;
    }
    return null;
  }

  return {
    compat, selectEF, selectByRule, bruteSelect, partition, depth,
    lateness, inversions, uninvert, edf, permutations, bruteLateness,
    greedyCoins, dpCoins,
    fractionalKnapsack, greedy01, brute01, bestSingle,
    huffman, codes, costOf, entropy, allTrees, bruteHuffmanCost,
    acyclic, matroidGreedy, bruteMatroid, tasksIndep, penaltyOf,
    greedyCover, bruteCover, harmonic,
    listSchedule, bruteMakespan, greedyFuel, bruteFuel,
    nearestNeighbour, tourLength, bruteTSP, horn, bruteHorn
  };
})();

(function () {
if (typeof d3 === "undefined") return;

/* ═══════════════════════════════════════════════════════════════════════════
   SX — DOM helpers
   ═══════════════════════════════════════════════════════════════════════════ */
const SX = {
  int: d3.format(","),
  f1: d3.format(".1f"), f2: d3.format(".2f"), f3: d3.format(".3f"), f4: d3.format(".4f"),
  flag: ok => ok ? '<span style="color:' + AC.good + '">✓ agree</span>' : '<span style="color:' + AC.bad + '">✗ DISAGREE</span>',
  clearControls: function (svgNode) { d3.select(svgNode.parentNode).selectAll('div[role="group"]').remove(); },
  has: id => !d3.select("#" + id).empty(),
  on: (id, ev, fn) => { const s = d3.select("#" + id); if (!s.empty()) s.on(ev, fn); },
  val: (id, dflt) => { const s = d3.select("#" + id); return s.empty() ? dflt : s.property("value"); },
  checked: (id, dflt) => { const s = d3.select("#" + id); return s.empty() ? dflt : s.property("checked"); },
  setText: (id, t) => { const s = d3.select("#" + id); if (!s.empty()) s.text(t); },
  setHtml: (id, t) => { const s = d3.select("#" + id); if (!s.empty()) s.html(t); },
  ids: arr => arr.map(a => a.id).join(", ")
};

/* the shared instances, so prose and figures agree by construction */
const INST = {
  book: [[1,4],[3,5],[0,6],[5,7],[3,9],[5,9],[6,10],[8,11],[8,12],[2,14],[12,16]].map((p, i) => ({ id: i + 1, s: p[0], f: p[1] })),
  ces: [[0,10],[1,2],[3,4]].map((p, i) => ({ id: i + 1, s: p[0], f: p[1] })),
  csd: [[0,4],[3,5],[4,8]].map((p, i) => ({ id: i + 1, s: p[0], f: p[1] })),
  cfc: [[0,2],[3,5],[6,8],[9,11],[4,7],[1,4],[1,4],[1,4],[7,10],[7,10],[7,10]].map((p, i) => ({ id: i + 1, s: p[0], f: p[1] })),
  randomActs: (n, window, seed) => { const r = AL.rng(seed); return Array.from({ length: n }, (_, i) => { const s = Math.floor(r() * window); const d = 1 + Math.floor(r() * 11); return { id: i + 1, s, f: s + d }; }); }
};

/* ── 01  #as-svg  activity selection stepped on a timeline ────────────── */
(function () {
  if (!SX.has("as-svg")) return;
  const W = 680, H = 330;
  function build() {
    const rule = SX.val("as-rule", "ef"), which = SX.val("as-inst", "book");
    const acts = which === "rand" ? INST.randomActs(12, 40, 23) : INST[which];
    const c = AL.counter(); const frames = [];
    const chosen = rule === "ef" ? GR.selectEF(acts, c, f => frames.push(f)) : GR.selectByRule(acts, rule, c, f => frames.push(f));
    const opt = GR.bruteSelect(acts);
    const tmax = Math.max(...acts.map(a => a.f));
    const svg = d3.select("#as-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 40, r: 16, t: 10, b: 60 });
    const x = d3.scaleLinear().domain([0, tmax]).range([0, F.iw]);
    const rowH = Math.min(18, (F.ih - 30) / (acts.length + 1));
    const order = rule === "ef" ? acts.slice().sort((a, b) => a.f - b.f || a.id - b.id) : acts.slice();
    const yOf = a => order.indexOf(a) * rowH;
    const names = ["earliest finish", "earliest start", "shortest duration", "fewest conflicts"][["ef", "es", "sd", "fc"].indexOf(rule)];
    function render(f, i) {
      F.g.selectAll("*").remove();
      AL.gridX(F.g, x, F.ih - 12, 8);
      AL.axisB(F.g, x, F.ih - 12, 8, "time");
      const status = new Map();                      // id → kept/skipped
      const seen = frames.slice(0, i + 1);
      if (rule === "ef") {
        seen.forEach(fr => status.set(fr.cand.id, fr.keep ? "keep" : "skip"));
      } else {
        seen.forEach(fr => { status.set(fr.cand.id, "keep"); fr.skipped.forEach(s => status.set(s.id, "skip")); });
      }
      acts.forEach(a => {
        const st = status.get(a.id); const cur = f && f.cand.id === a.id;
        const col = cur ? AC.a2 : st === "keep" ? AC.good : st === "skip" ? AC.bad : AC.panel2;
        F.g.append("rect").attr("x", x(a.s)).attr("y", yOf(a) + 2).attr("width", Math.max(2, x(a.f) - x(a.s))).attr("height", rowH - 4).attr("rx", 3)
          .attr("fill", col).attr("stroke", AC.line).attr("opacity", st === "skip" && !cur ? 0.55 : 1);
        F.g.append("text").attr("x", x(a.s) + 4).attr("y", yOf(a) + rowH / 2 + 4).attr("font-size", 10).attr("fill", cur || st ? AC.bg : AC.ink).text("a" + a.id);
      });
      /* the brute-force optimum, as a dashed strip */
      const yb = acts.length * rowH + 8;
      F.g.append("text").attr("x", 0).attr("y", yb + 10).attr("font-size", 10).attr("fill", AC.muted).text("a brute-force optimum:");
      opt.forEach(a => F.g.append("rect").attr("x", x(a.s)).attr("y", yb + 14).attr("width", Math.max(2, x(a.f) - x(a.s))).attr("height", 10).attr("rx", 2).attr("fill", "none").attr("stroke", AC.violet).attr("stroke-dasharray", "4 2"));
      /* the "last" line for EF */
      if (rule === "ef" && f) F.g.append("line").attr("x1", x(f.last)).attr("x2", x(f.last)).attr("y1", 0).attr("y2", acts.length * rowH).attr("stroke", AC.teal).attr("stroke-dasharray", "3 3");
      const kept = f ? f.chosen.length : 0;
      F.g.append("text").attr("x", F.iw).attr("y", -1).attr("text-anchor", "end").attr("font-size", 11).attr("fill", AC.muted)
        .text(`${names}: ${f ? (rule === "ef" ? (f.keep ? "keep a" + f.cand.id : "skip a" + f.cand.id + " (starts before " + f.last + ")") : "take a" + f.cand.id + (f.skipped.length ? ", discard " + f.skipped.map(s => "a" + s.id).join(" ") : "")) : ""} · kept so far ${kept}`);
    }
    AL.stepper(svg, { frames, render, delay: 800, label: "decision" });
    const ok = chosen.length === opt.length;
    const cnt = c.all();
    SX.setHtml("as-readout", `<b>${names}</b> on ${acts.length} activities → <b>${chosen.length}</b> chosen {${chosen.map(a => "a" + a.id).join(", ")}} · brute force over all 2^${acts.length} = ${SX.int(1 << acts.length)} subsets: maximum compatible set has <b>${opt.length}</b> members ${ok ? SX.flag(true) : '<span style="color:' + AC.bad + '">✗ greedy is NOT optimal here</span>'} · measured: ${rule === "ef" ? cnt.cmp + " comparisons sₖ ≥ last (= n − 1 = " + (acts.length - 1) + " " + SX.flag(cnt.cmp === acts.length - 1) + ")" : (cnt.exam || 0) + " candidates examined, " + (cnt.cmp || 0) + " compatibility tests"}${which === "book" && rule !== "ef" ? " · note: this rule happens to " + (ok ? "succeed" : "fail") + " on this instance; switch the instance to its counterexample" : ""}`);
  }
  SX.on("as-rule", "change", build); SX.on("as-inst", "change", build);
  build();
})();

/* ── 02  #rules-svg  failure rate of the four rules ───────────────────── */
(function () {
  if (!SX.has("rules-svg")) return;
  const W = 680, H = 240, T = 300;
  function build() {
    const n = +SX.val("rules-n", 8); SX.setText("rules-n-out", n);
    const dens = SX.val("rules-dens", "mid"); const win = dens === "sparse" ? 60 : dens === "dense" ? 20 : 40;
    const rules = ["ef", "es", "sd", "fc"], names = ["earliest finish", "earliest start", "shortest duration", "fewest conflicts"];
    const fails = [0, 0, 0, 0], worst = [0, 0, 0, 0]; let optSum = 0; const c = AL.counter();
    for (let t = 0; t < T; t++) {
      const acts = INST.randomActs(n, win, 1000 + t);
      const opt = GR.bruteSelect(acts).length; optSum += opt;
      rules.forEach((r, k) => { const g = (r === "ef" ? GR.selectEF(acts, c) : GR.selectByRule(acts, r, c)).length; if (g < opt) { fails[k]++; worst[k] = Math.max(worst[k], opt - g); } if (g > opt) throw new Error("greedy exceeded brute force — impossible"); });
    }
    const F = AL.frame("#rules-svg", W, H, { l: 46, r: 16, t: 14, b: 40 });
    const x = d3.scaleBand().domain(names).range([0, F.iw]).padding(0.3);
    const y = d3.scaleLinear().domain([0, Math.max(0.05, 1.3 * Math.max(...fails) / T)]).range([F.ih, 0]).nice();
    AL.gridY(F.g, y, F.iw, 5); AL.axisB(F.g, x, F.ih, 4); AL.axisL(F.g, y, 5, "fraction of instances where the rule is not optimal", d3.format(".0%"));
    names.forEach((nm, k) => {
      F.g.append("rect").attr("x", x(nm)).attr("y", y(fails[k] / T)).attr("width", x.bandwidth()).attr("height", F.ih - y(fails[k] / T)).attr("fill", k === 0 ? AC.good : AC.bad).attr("rx", 3);
      F.g.append("text").attr("x", x(nm) + x.bandwidth() / 2).attr("y", y(fails[k] / T) - 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.ink).text(`${fails[k]} / ${T}`);
    });
    SX.setHtml("rules-readout", `${T} random instances, n = ${n}, window ${win}, brute force over 2^${n} subsets each · failures: earliest finish <b>${fails[0]}</b> ${SX.flag(fails[0] === 0)} · earliest start <b>${fails[1]}</b> (worst shortfall ${worst[1]}) · shortest duration <b>${fails[2]}</b> (worst ${worst[2]}) · fewest conflicts <b>${fails[3]}</b> (worst ${worst[3]}) · mean optimum ${SX.f2(optSum / T)} activities · measured work over the run: ${SX.int(c.get("cmp"))} compatibility tests, ${SX.int(c.get("exam"))} candidate examinations · no rule ever exceeded the optimum (asserted)`);
  }
  SX.on("rules-n", "input", build); SX.on("rules-dens", "change", build);
  build();
})();

/* ── 03  #ip-svg  interval partitioning stepped ───────────────────────── */
(function () {
  if (!SX.has("ip-svg")) return;
  const W = 680, H = 260;
  const LECT = [["a",9,10.5],["b",9,12.5],["c",9,10.5],["d",11,12.5],["e",11,14],["f",13,14.5],["g",13,14.5],["h",13,16.5],["i",15,16.5],["j",15,16.5]].map(p => ({ id: p[0], s: p[1], f: p[2] }));
  function partitionBy(acts, order, c, rec) {
    if (order === "start") return GR.partition(acts, c, rec);
    /* finish order: same room rule, different processing order */
    const sorted = acts.slice().sort((a, b) => a.f - b.f || a.s - b.s || String(a.id).localeCompare(String(b.id)));
    const rooms = []; const assign = new Map();
    sorted.forEach(a => {
      let r = -1;
      for (let k = 0; k < rooms.length; k++) { c.add("cmp"); if (rooms[k] <= a.s) { r = k; break; } }
      if (r < 0) { rooms.push(a.f); r = rooms.length - 1; c.add("open"); } else rooms[r] = Math.max(rooms[r], a.f);
      assign.set(a.id, r);
      if (rec) rec({ cand: a, room: r, opened: rooms.length, assign: new Map(assign) });
    });
    return { rooms: rooms.length, assign };
  }
  function build() {
    const order = SX.val("ip-order", "start"), which = SX.val("ip-inst", "book");
    const acts = which === "book" ? LECT : INST.randomActs(12, 30, 51).map(a => ({ id: String(a.id), s: a.s, f: a.f }));
    const c = AL.counter(); const frames = [];
    const res = partitionBy(acts, order, c, f => frames.push(f));
    const d = GR.depth(acts);
    /* where is the depth attained? first start time with d intervals alive */
    let where = null; acts.map(a => a.s).sort((p, q) => p - q).forEach(p => { if (where !== null) return; let k = 0; acts.forEach(a => { if (a.s <= p && p < a.f) k++; }); if (k === d) where = p; });
    const t0 = Math.min(...acts.map(a => a.s)), t1 = Math.max(...acts.map(a => a.f));
    const svg = d3.select("#ip-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 60, r: 16, t: 12, b: 40 });
    const x = d3.scaleLinear().domain([t0, t1]).range([0, F.iw]);
    const rowH = Math.min(30, (F.ih - 10) / Math.max(res.rooms, 4));
    /* a finish-order placement can put an interval in a room whose previous
       interval is compatible but which greedy-by-start would not have chosen;
       validity is checked here: no two intervals in one room overlap */
    let valid = true;
    for (let i = 0; i < acts.length; i++) for (let j = i + 1; j < acts.length; j++) if (res.assign.get(acts[i].id) === res.assign.get(acts[j].id) && !GR.compat(acts[i], acts[j])) valid = false;
    function render(f, i) {
      F.g.selectAll("*").remove();
      AL.gridX(F.g, x, F.ih, 8); AL.axisB(F.g, x, F.ih, 8, "time");
      const opened = f ? f.opened : 0;
      for (let r = 0; r < opened; r++) {
        F.g.append("text").attr("x", -8).attr("y", r * rowH + rowH / 2 + 4).attr("text-anchor", "end").attr("font-size", 11).attr("fill", AC.muted).text("room " + (r + 1));
        F.g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", r * rowH + rowH).attr("y2", r * rowH + rowH).attr("stroke", AC.grid);
      }
      if (f) f.assign.forEach((r, id) => {
        const a = acts.find(q => q.id === id); const cur = f.cand.id === id;
        F.g.append("rect").attr("x", x(a.s)).attr("y", r * rowH + 3).attr("width", Math.max(2, x(a.f) - x(a.s))).attr("height", rowH - 6).attr("rx", 3)
          .attr("fill", cur ? AC.a2 : (r === f.opened - 1 && cur) ? AC.rose : AC.accent).attr("stroke", AC.line);
        F.g.append("text").attr("x", x(a.s) + 4).attr("y", r * rowH + rowH / 2 + 4).attr("font-size", 11).attr("fill", AC.bg).text(id);
      });
      if (where !== null) { F.g.append("line").attr("x1", x(where)).attr("x2", x(where)).attr("y1", 0).attr("y2", F.ih).attr("stroke", AC.violet).attr("stroke-dasharray", "2 3");
        F.g.append("text").attr("x", x(where) + 3).attr("y", F.ih - 4).attr("font-size", 10).attr("fill", AC.violet).text(`depth ${d} at t = ${where}`); }
      F.g.append("text").attr("x", F.iw).attr("y", -2).attr("text-anchor", "end").attr("font-size", 11).attr("fill", AC.muted)
        .text(f ? `${f.cand.id} [${f.cand.s}, ${f.cand.f}) → room ${f.room + 1}${f.room === f.opened - 1 && f.assign.size && [...f.assign.values()].filter(v => v === f.room).length === 1 ? " (opened)" : ""} · rooms ${f.opened}` : "");
    }
    const st = AL.stepper(svg, { frames, render, delay: 800, label: "interval" });
    st.go(frames.length - 1);                          // default: the finished partition, rooms = depth; ⟲ rewinds
    SX.setHtml("ip-readout", `${acts.length} intervals in <b>${order}-time order</b> → <b>${res.rooms} rooms</b> · depth, computed independently by testing every start time: <b>${d}</b> ${res.rooms === d ? SX.flag(true) : '<span style="color:' + AC.bad + '">✗ more rooms than the depth — the guarantee holds only for start order</span>'} · partition valid (no overlap within a room): ${SX.flag(valid)} · measured: ${c.get("cmp") || 0} "is this room free?" tests, ${c.get("open")} rooms opened`);
  }
  SX.on("ip-order", "change", build); SX.on("ip-inst", "change", build);
  build();
})();

/* ── 04  #late-svg  minimise maximum lateness, inversions exchanged ────── */
(function () {
  if (!SX.has("late-svg")) return;
  const W = 680, H = 230;
  const JOBS = [["A",3,6],["B",2,8],["C",1,9],["D",4,9],["E",3,14],["F",2,15]].map(p => ({ id: p[0], t: p[1], d: p[2] }));
  function build() {
    const which = SX.val("late-start", "reverse");
    const byId = id => JOBS.find(j => j.id === id);
    let start;
    if (which === "reverse") start = JOBS.slice().reverse();
    else if (which === "one") start = ["A","B","C","E","D","F"].map(byId);
    else start = AL.shuffle(JOBS, AL.rng(31));
    const c = AL.counter(); const frames = [];
    const final = GR.uninvert(start, c, f => frames.push(f));
    const brute = GR.bruteLateness(JOBS);
    const edfL = GR.lateness(GR.edf(JOBS)).maxL;
    const finalL = GR.lateness(final).maxL;
    let monotone = true; for (let i = 1; i < frames.length; i++) if (frames[i].L.maxL > frames[i - 1].L.maxL) monotone = false;
    const total = JOBS.reduce((s, j) => s + j.t, 0);
    const svg = d3.select("#late-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 30, r: 16, t: 30, b: 50 });
    const x = d3.scaleLinear().domain([0, Math.max(total, ...JOBS.map(j => j.d)) + 1]).range([0, F.iw]);
    function render(f, i) {
      F.g.selectAll("*").remove();
      AL.axisB(F.g, x, F.ih, 8, "time");
      const rows = f.L.rows; let tcur = 0;
      f.order.forEach((j, k) => {
        const r = rows[k]; const sw = f.swapped && (k === f.swapped[0] || k === f.swapped[1]);
        F.g.append("rect").attr("x", x(tcur)).attr("y", 20).attr("width", x(tcur + j.t) - x(tcur)).attr("height", 44).attr("rx", 4)
          .attr("fill", r.L > 0 ? AC.bad : AC.accent).attr("stroke", sw ? AC.a2 : AC.line).attr("stroke-width", sw ? 3 : 1);
        F.g.append("text").attr("x", x(tcur + j.t / 2)).attr("y", 40).attr("text-anchor", "middle").attr("font-size", 13).attr("fill", AC.bg).text(j.id);
        F.g.append("text").attr("x", x(tcur + j.t / 2)).attr("y", 56).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.bg).text("L=" + r.L);
        /* deadline tick */
        F.g.append("line").attr("x1", x(j.d)).attr("x2", x(j.d)).attr("y1", 70 + k * 9).attr("y2", 78 + k * 9).attr("stroke", r.L > 0 ? AC.bad : AC.muted).attr("stroke-width", 2);
        F.g.append("text").attr("x", x(j.d) + 3).attr("y", 78 + k * 9).attr("font-size", 9).attr("fill", AC.muted).text("d" + j.id + "=" + j.d);
        tcur += j.t;
      });
      F.g.append("text").attr("x", 0).attr("y", -12).attr("font-size", 12).attr("fill", AC.ink)
        .text(`${f.swapped ? "swapped positions " + (f.swapped[0] + 1) + "," + (f.swapped[1] + 1) + " (" + f.order[f.swapped[1]].id + " ↔ " + f.order[f.swapped[0]].id + "): max lateness " + f.before + " → " + f.L.maxL : "start: max lateness " + f.L.maxL} · inversions left ${f.inv}`);
    }
    AL.stepper(svg, { frames, render, delay: 900, label: "swap" });
    SX.setHtml("late-readout", `start order ${start.map(j => j.id).join(" ")}: max lateness <b>${frames[0].L.maxL}</b> (job ${frames[0].L.who || "—"}), ${frames[0].inv} inversions · <b>${c.get("swap")} adjacent swaps</b> measured, each removing exactly one inversion ${SX.flag(c.get("swap") === frames[0].inv)} · max lateness never rose across the swaps ${SX.flag(monotone)} · final order ${final.map(j => j.id).join(" ")}: max lateness <b>${finalL}</b> = earliest-deadline schedule's ${edfL} ${SX.flag(finalL === edfL)} · brute force over all 6! = 720 orders: minimum max lateness <b>${brute.maxL}</b> ${SX.flag(brute.maxL === edfL)}`);
  }
  SX.on("late-start", "change", build);
  build();
})();

/* ── 05  #coin-svg  coin change greedy vs DP over amounts 1–100 ─────────── */
(function () {
  if (!SX.has("coin-svg")) return;
  const W = 680, H = 240, XMAX = 100;
  function build() {
    const coins = SX.val("coin-sys", "1,3,4").split(",").map(Number);
    const cg = AL.counter(), cd = AL.counter();
    const rows = [];
    for (let x = 1; x <= XMAX; x++) { const g = GR.greedyCoins(coins, x, cg), d = GR.dpCoins(coins, x, cd); rows.push({ x, g: g.length, d: d.length, gc: g, dc: d }); }
    const fails = rows.filter(r => r.g > r.d);
    if (rows.some(r => r.g < r.d)) throw new Error("greedy beat the DP — impossible");
    const F = AL.frame("#coin-svg", W, H, { l: 40, r: 16, t: 14, b: 36 });
    const x = d3.scaleBand().domain(rows.map(r => r.x)).range([0, F.iw]).padding(0.15);
    const y = d3.scaleLinear().domain([0, Math.max(...rows.map(r => r.g))]).range([F.ih, 0]).nice();
    AL.gridY(F.g, y, F.iw, 5);
    AL.axisB(F.g, d3.scaleLinear().domain([1, XMAX]).range([x(1), x(XMAX)]), F.ih, 10, "amount");
    AL.axisL(F.g, y, 5, "coins used");
    rows.forEach(r => {
      F.g.append("rect").attr("x", x(r.x)).attr("y", y(r.g)).attr("width", x.bandwidth()).attr("height", F.ih - y(r.g)).attr("fill", r.g > r.d ? AC.bad : AC.accent);
      if (r.g > r.d) F.g.append("rect").attr("x", x(r.x)).attr("y", y(r.d)).attr("width", x.bandwidth()).attr("height", 2).attr("fill", AC.ink);
    });
    AL.legend(F.g, [{ label: "greedy coin count (blue = optimal, red = beaten by the DP; the white tick is the DP's count)", color: AC.accent }], 8, 12);
    const ex = fails.length ? fails[0] : null;
    const kz = coins.length >= 3 ? `smallest counterexample must lie in (${coins[2] + 1}, ${coins[coins.length - 2] + coins[coins.length - 1]})` : "two coins: always canonical";
    SX.setHtml("coin-readout", `{${coins.join(", ")}} on amounts 1 … ${XMAX}: greedy uses more coins than the DP on <b>${fails.length} / ${XMAX}</b> amounts${ex ? ` — first at <b>${ex.x}</b>: greedy ${ex.gc.join(" + ")} (${ex.g} coins), DP ${ex.dc.join(" + ")} (${ex.d}) — ${kz}${ex.x > coins[2] + 1 && ex.x < coins[coins.length - 2] + coins[coins.length - 1] ? " " + SX.flag(true) : ""}` : " — canonical on this range" + (coins.length >= 3 ? "; " + kz + ", and no failure was found below that bound either" : "")} · greedy never beat the DP (asserted) · measured: ${cg.get("coin")} coins issued by greedy over the run, ${SX.int(cd.get("cell"))} DP cell updates`);
  }
  SX.on("coin-sys", "change", build);
  build();
})();

/* ── 06  #knap-svg  fractional vs 0/1 knapsack ────────────────────────── */
(function () {
  if (!SX.has("knap-svg")) return;
  const W = 680, H = 260;
  const K = {
    four: { W: 50, items: [["A",10,60],["B",20,100],["C",30,120],["D",25,90]] },
    three: { W: 50, items: [["1",10,60],["2",20,100],["3",30,120]] },
    tight: { W: 40, items: [["s",1,2],["h₁",20,20],["h₂",20,20]] },
    bad: { W: 100, items: [["small",1,2],["big",100,100]] }
  };
  function build() {
    const which = SX.val("knap-inst", "four");
    let W0, items;
    if (which === "rand") { const r = AL.rng(19); W0 = 60; items = Array.from({ length: 8 }, (_, i) => ({ id: String.fromCharCode(65 + i), w: 5 + Math.floor(r() * 30), v: 10 + Math.floor(r() * 90) })); }
    else { W0 = K[which].W; items = K[which].items.map(p => ({ id: p[0], w: p[1], v: p[2] })); }
    const fit = items.filter(it => it.w <= W0);
    const cf = AL.counter(), c01 = AL.counter();
    const frac = GR.fractionalKnapsack(fit, W0, cf), g01 = GR.greedy01(fit, W0, c01), opt = GR.brute01(fit, W0), M = GR.bestSingle(fit, W0);
    const fixed = Math.max(g01.val, M);
    const F = AL.frame("#knap-svg", W, H, { l: 16, r: 16, t: 22, b: 44 });
    /* left: the capacity bar packed by density order — fractional (top) and 0/1 (bottom) */
    const lw = F.iw * 0.55, x = d3.scaleLinear().domain([0, W0]).range([0, lw]);
    const dmax = Math.max(...fit.map(it => it.v / it.w));
    const hOf = it => 18 + 40 * (it.v / it.w) / dmax;
    const drawPack = (y0, label, take) => {
      F.g.append("text").attr("x", 0).attr("y", y0 - 4).attr("font-size", 11).attr("fill", AC.muted).text(label);
      F.g.append("rect").attr("x", 0).attr("y", y0).attr("width", lw).attr("height", 60).attr("fill", "none").attr("stroke", AC.line);
      let cur = 0;
      take.forEach(t => {
        const it = fit.find(q => q.id === t.id); const wpart = it.w * t.frac; if (wpart <= 0) return;
        const h = hOf(it);
        F.g.append("rect").attr("x", x(cur)).attr("y", y0 + 60 - h).attr("width", Math.max(1, x(cur + wpart) - x(cur))).attr("height", h).attr("fill", t.frac < 1 ? AC.a2 : AC.accent).attr("stroke", AC.bg);
        if (x(wpart) > 14) F.g.append("text").attr("x", x(cur + wpart / 2)).attr("y", y0 + 60 - h / 2 + 4).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.bg).text(it.id + (t.frac < 1 ? " ×" + SX.f2(t.frac) : ""));
        cur += wpart;
      });
      F.g.append("text").attr("x", lw).attr("y", y0 + 72).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.muted).text(`W = ${W0}; box height ∝ value density`);
    };
    drawPack(14, `fractional greedy — value ${SX.f1(frac.val)}`, frac.take);
    drawPack(114, `0/1 greedy by density — value ${g01.val} (items ${g01.take.join(", ") || "none"})`, g01.take.map(id => ({ id, frac: 1 })));
    /* right: bars */
    const bx = lw + 40, bw = F.iw - bx;
    const vals = [{ k: "fractional", v: frac.val, col: AC.a2 }, { k: "0/1 greedy", v: g01.val, col: AC.accent }, { k: "max(greedy, single)", v: fixed, col: AC.teal }, { k: "0/1 optimum (brute)", v: opt.val, col: AC.good }];
    const xb = d3.scaleBand().domain(vals.map(v => v.k)).range([0, bw]).padding(0.25);
    const yb = d3.scaleLinear().domain([0, Math.max(...vals.map(v => v.v)) * 1.15]).range([F.ih, 0]);
    const gb = F.g.append("g").attr("transform", `translate(${bx},0)`);
    vals.forEach(v => { gb.append("rect").attr("x", xb(v.k)).attr("y", yb(v.v)).attr("width", xb.bandwidth()).attr("height", F.ih - yb(v.v)).attr("fill", v.col).attr("rx", 3);
      gb.append("text").attr("x", xb(v.k) + xb.bandwidth() / 2).attr("y", yb(v.v) - 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.ink).text(SX.f1(v.v).replace(/\.0$/, ""));
      const words = v.k.split(" "); words.forEach((wd, i) => gb.append("text").attr("x", xb(v.k) + xb.bandwidth() / 2).attr("y", F.ih + 12 + i * 11).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", AC.muted).text(wd)); });
    gb.append("line").attr("x1", 0).attr("x2", bw).attr("y1", yb(opt.val / 2)).attr("y2", yb(opt.val / 2)).attr("stroke", AC.rose).attr("stroke-dasharray", "3 3");
    gb.append("text").attr("x", bw).attr("y", yb(opt.val / 2) - 3).attr("text-anchor", "end").attr("font-size", 9).attr("fill", AC.rose).text("OPT / 2");
    SX.setHtml("knap-readout", `${fit.length} items, W = ${W0} · fractional greedy <b>${SX.f1(frac.val)}</b> (${cf.get("item")} items examined) · 0/1 greedy by density <b>${g01.val}</b> {${g01.take.join(", ")}} · best single item ${M} · max(greedy, single) = <b>${fixed}</b> · brute force over 2^${fit.length} = ${1 << fit.length} subsets: 0/1 optimum <b>${opt.val}</b> {${opt.take.join(", ")}} · checks: fractional ≥ 0/1 optimum ${SX.flag(frac.val >= opt.val - 1e-9)} · fixed rule ≥ OPT/2 ${SX.flag(fixed >= opt.val / 2)} (ratio ${SX.f3(fixed / opt.val)}) · plain 0/1 greedy ${g01.val === opt.val ? "happens to be optimal here" : "is <b>not</b> optimal: ratio " + SX.f3(g01.val / opt.val)}`);
  }
  SX.on("knap-inst", "change", build);
  build();
})();

/* ── 07  #huff-svg  Huffman stepped: forest, queue, tree, codes ─────────── */
(function () {
  if (!SX.has("huff-svg")) return;
  const W = 680, H = 300;
  const ALPH = {
    six: [["a",45],["b",13],["c",12],["d",16],["e",9],["f",5]],
    dyadic: [["a",4],["b",2],["c",1],["d",1]],
    equal4: [["a",1],["b",1],["c",1],["d",1]],
    equal5: [["a",1],["b",1],["c",1],["d",1],["e",1]],
    two: [["a",99],["b",1]],
    one: [["a",7]],
    tie: [["a",1],["b",1],["c",2],["d",2]]
  };
  /* level-order array of the tree, with the merge step at which each internal node was created */
  function levelOrder(root, stepOf) {
    const arr = [], steps = [];
    (function put(t, i) { while (arr.length <= i) { arr.push(null); steps.push(null); } arr[i] = t; steps[i] = t.leaf ? 0 : stepOf.get(t); if (!t.leaf) { put(t.left, 2 * i + 1); put(t.right, 2 * i + 2); } })(root, 0);
    return { arr, steps };
  }
  function build() {
    const which = SX.val("huff-inst", "six");
    const freq = ALPH[which].map(p => ({ sym: p[0], w: p[1] }));
    const c = AL.counter(); const frames = [];
    const root = GR.huffman(freq, c, f => frames.push(f));
    const stepOf = new Map(); frames.forEach((f, i) => stepOf.set(f.z, i + 1));
    const lo = levelOrder(root, stepOf);
    const codes = GR.codes(root), B = GR.costOf(root), tot = freq.reduce((s, f) => s + f.w, 0);
    const L = freq.length === 1 ? 0 : B / tot, Hh = GR.entropy(freq);
    const brute = freq.length <= 7 ? GR.bruteHuffmanCost(freq) : null;
    const nTrees = freq.length <= 7 ? GR.allTrees(freq).length : null;
    const fixed = Math.ceil(Math.log2(Math.max(2, freq.length)));
    const allFrames = [{ init: true, pending: freq.map((f, i) => ({ sym: f.sym, w: f.w, birth: i, leaf: true })).sort((a, b) => a.w - b.w || a.birth - b.birth) }].concat(frames);
    const svg = d3.select("#huff-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const label = t => t.leaf ? `${t.sym}:${t.w}` : String(t.w);
    function render(f, i) {
      F.g.selectAll("*").remove();
      const step = i;                                   // number of merges done
      const arr = lo.arr.map((t, k) => t === null ? null : (t.leaf || lo.steps[k] <= step) ? label(t) : null);
      const levels = Math.max(1, Math.floor(Math.log2(lo.arr.length)) + 1);
      const tw = 430, levelH = Math.min(56, (H - 90) / Math.max(1, levels));
      AL.binTree(F.g, arr, { x: 0, y: 6, w: tw, levelH, r: 15, mark: (k, v) => { const t = lo.arr[k]; if (!t) return null; if (!t.leaf && lo.steps[k] === step && step > 0) return AC.a2; if (t.leaf) return (f.x === t || f.y === t) && !f.init ? AC.a2 : AC.panel2; return AC.panel2; } });
      /* queue on the right */
      const qx = tw + 20;
      F.g.append("text").attr("x", qx).attr("y", 14).attr("font-size", 11).attr("fill", AC.muted).text(f.init ? "queue (by weight):" : `after merge ${step}: queue`);
      const pend = f.init ? f.pending : f.pending;
      pend.forEach((t, k) => F.g.append("text").attr("x", qx).attr("y", 32 + k * 15).attr("font-size", 11).attr("fill", AC.ink).text(label(t) + (t.leaf ? "" : " (merged)")));
      if (!f.init) F.g.append("text").attr("x", qx).attr("y", 36 + pend.length * 15).attr("font-size", 11).attr("fill", AC.a2).text(`merged ${label(f.x)} + ${label(f.y)} = ${f.z.w}`);
      if (step === frames.length) {
        const cs = Object.keys(codes).map(s => `${s} ${codes[s]}`).join("   ");
        F.g.append("text").attr("x", 0).attr("y", H - 26).attr("font-size", 11).attr("fill", AC.good).text("codewords: " + (freq.length === 1 ? "a (length 0 — a single leaf)" : cs));
      }
    }
    const st = AL.stepper(svg, { frames: allFrames, render, delay: 900, label: "merge" });
    st.go(allFrames.length - 1);                       // default: the finished tree and its codewords; ⟲ rewinds
    const n = freq.length;
    SX.setHtml("huff-readout", `${n} symbol${n > 1 ? "s" : ""}, total frequency ${tot} · <b>${n - 1} merges</b>, merge weights ${frames.map(f => f.z.w).join(" + ") || "—"} = ${frames.reduce((s, f) => s + f.z.w, 0)} · tree cost B(T) = ∑ freq·depth = <b>${B}</b> ${SX.flag(B === frames.reduce((s, f) => s + f.z.w, 0))} · measured priority-queue operations <b>${c.get("pq")}</b>${n > 1 ? " = |C| inserts + 3(|C| − 1) = " + (4 * n - 3) + " " + SX.flag(c.get("pq") === 4 * n - 3) : ""} · L = <b>${SX.f4(L)}</b> bits/symbol, entropy H = ${SX.f4(Hh)}, fixed-length ${fixed} · H ≤ L &lt; H + 1: ${SX.flag(Hh - 1e-9 <= L && L < Hh + 1)}${Math.abs(L - Hh) < 1e-9 ? " (equality: dyadic)" : ""}${brute !== null ? ` · exhaustive search over all ${nTrees} full trees on ${n} labelled leaves: minimum cost ${brute} ${SX.flag(brute === B)}` : ""}${which === "one" ? " · one symbol: the algorithm returns a lone leaf of depth 0, so L = 0 = H; a real coder would assign a 1-bit codeword to stay decodable" : ""}${which === "tie" ? " · lengths here " + Object.values(codes).map(s => s.length).join(",") + "; the other tie-break (pairing the merged node with a leaf) gives lengths " + (Object.values(codes).every(s => s.length === 2) ? "3,3,2,1" : "2,2,2,2") + " with the same cost 12" : ""}`);
  }
  SX.on("huff-inst", "change", build);
  build();
})();

/* ── 08  #ent-svg  Huffman vs entropy vs fixed length as skew varies ────── */
(function () {
  if (!SX.has("ent-svg")) return;
  const W = 680, H = 250;
  function dist(k, s) { const w = Array.from({ length: k }, (_, i) => Math.pow(i + 1, -s)); const t = w.reduce((a, b) => a + b, 0); return w.map((x, i) => ({ sym: String.fromCharCode(97 + i), w: x / t })); }
  function build() {
    const k = +SX.val("ent-k", 8), s0 = +SX.val("ent-skew", 1.5);
    SX.setText("ent-k-out", k); SX.setText("ent-skew-out", SX.f1(s0));
    const skews = AL.linspace(0, 3, 31); const c = AL.counter();
    const pts = skews.map(s => { const f = dist(k, s); const root = GR.huffman(f, c); const L = GR.costOf(root); const Hh = GR.entropy(f); return { s, L, H: Hh, ok: Hh - 1e-9 <= L && L < Hh + 1 }; });
    const fixed = Math.ceil(Math.log2(k));
    const cur = (() => { const f = dist(k, s0); const root = GR.huffman(f); return { L: GR.costOf(root), H: GR.entropy(f), brute: k <= 6 ? GR.bruteHuffmanCost(f) : null }; })();
    const F = AL.frame("#ent-svg", W, H, { l: 46, r: 16, t: 14, b: 36 });
    const x = d3.scaleLinear().domain([0, 3]).range([0, F.iw]);
    const y = d3.scaleLinear().domain([0, Math.max(fixed, ...pts.map(p => p.H + 1)) * 1.05]).range([F.ih, 0]);
    AL.gridY(F.g, y, F.iw, 5); AL.axisB(F.g, x, F.ih, 6, "skew s — probabilities ∝ (rank)^(−s)"); AL.axisL(F.g, y, 5, "bits per symbol");
    const line = key => d3.line().x(p => x(p.s)).y(p => y(p[key]));
    F.g.append("path").datum(pts.map(p => ({ s: p.s, v: p.H + 1 }))).attr("fill", "none").attr("stroke", AC.muted).attr("stroke-dasharray", "4 3").attr("d", d3.line().x(p => x(p.s)).y(p => y(p.v)));
    F.g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(fixed)).attr("y2", y(fixed)).attr("stroke", AC.rose).attr("stroke-dasharray", "2 3");
    F.g.append("path").datum(pts).attr("fill", "none").attr("stroke", AC.teal).attr("stroke-width", 2).attr("d", line("H"));
    F.g.append("path").datum(pts).attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 2.5).attr("d", line("L"));
    F.g.append("line").attr("x1", x(s0)).attr("x2", x(s0)).attr("y1", 0).attr("y2", F.ih).attr("stroke", AC.accent).attr("stroke-dasharray", "3 3");
    F.g.append("circle").attr("cx", x(s0)).attr("cy", y(cur.L)).attr("r", 4).attr("fill", AC.a2);
    AL.legend(F.g, [{ label: "Huffman L (measured)", color: AC.a2 }, { label: "entropy H (the floor)", color: AC.teal }, { label: "H + 1 (the ceiling for Huffman)", color: AC.muted, dash: "4 3" }, { label: `fixed length ceil(log₂ ${k}) = ${fixed}`, color: AC.rose, dash: "2 3" }], 10, F.ih - 56);
    SX.setHtml("ent-readout", `|C| = ${k}, at skew s = ${SX.f1(s0)}: H = <b>${SX.f4(cur.H)}</b>, Huffman L = <b>${SX.f4(cur.L)}</b> (L − H = ${SX.f4(cur.L - cur.H)}), fixed-length ${fixed} · H ≤ L &lt; H + 1 held at all 31 skews ${SX.flag(pts.every(p => p.ok))} · at s = 0 (uniform) L = ${SX.f4(pts[0].L)} vs H = log₂${k} = ${SX.f4(Math.log2(k))}; at s = 3, L = ${SX.f4(pts[30].L)} vs H = ${SX.f4(pts[30].H)} — the gap grows as the distribution concentrates on one symbol, which cannot get fewer than 1 bit${cur.brute !== null ? ` · exhaustive check at this skew over all ${GR.allTrees(dist(k, s0)).length} trees: min ${SX.f4(cur.brute)} ${SX.flag(Math.abs(cur.brute - cur.L) < 1e-9)}` : " · (exhaustive tree search shown for |C| ≤ 6)"} · ${c.get("pq")} priority-queue operations across the 31 codes`);
  }
  SX.on("ent-k", "input", build); SX.on("ent-skew", "input", build);
  build();
})();

/* ── 09  #mat-svg  a graphic matroid: exchange axiom and greedy max-weight forest ── */
(function () {
  if (!SX.has("mat-svg")) return;
  const W = 680, H = 280;
  const V = { 1: [60, 60], 2: [200, 40], 3: [60, 200], 4: [200, 220], 5: [320, 130] };
  const E = [["12",1,2,7],["13",1,3,3],["23",2,3,5],["24",2,4,8],["34",3,4,6],["35",3,5,2],["45",4,5,4]].map(p => ({ id: p[0], u: p[1], v: p[2], w: p[3] }));
  const indep = (set, c) => GR.acyclic(set, c);
  function build() {
    const mode = SX.val("mat-mode", "exchange");
    const svg = d3.select("#mat-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const drawGraph = (g, colorOf, widthOf, labelOf) => {
      E.forEach(e => { const a = V[e.u], b = V[e.v]; g.append("line").attr("x1", a[0]).attr("y1", a[1]).attr("x2", b[0]).attr("y2", b[1]).attr("stroke", colorOf(e)).attr("stroke-width", widthOf(e));
        g.append("text").attr("x", (a[0] + b[0]) / 2 + 6).attr("y", (a[1] + b[1]) / 2 - 4).attr("font-size", 10).attr("fill", AC.muted).text(labelOf(e)); });
      Object.keys(V).forEach(k => { g.append("circle").attr("cx", V[k][0]).attr("cy", V[k][1]).attr("r", 13).attr("fill", AC.panel2).attr("stroke", AC.line); g.append("text").attr("x", V[k][0]).attr("y", V[k][1] + 4).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", AC.ink).text(k); });
    };
    /* brute force over all subsets: every maximal forest has 4 edges; max-weight forest */
    const brute = GR.bruteMatroid(E, s => indep(s));
    let maximalSizes = new Set(), nIndep = 0;
    for (let mask = 0; mask < (1 << E.length); mask++) {
      const sub = E.filter((_, i) => mask & (1 << i)); if (!indep(sub)) continue; nIndep++;
      const maximal = E.every(e => sub.includes(e) || !indep(sub.concat([e]))); if (maximal) maximalSizes.add(sub.length);
    }
    if (mode === "exchange") {
      const A = E.filter(e => ["12", "34"].includes(e.id)), B = E.filter(e => ["13", "24", "45"].includes(e.id));
      const c = AL.counter();
      const ext = B.filter(e => !A.includes(e)).map(e => ({ e, ok: indep(A.concat([e]), c) }));
      drawGraph(F.g, e => A.includes(e) ? AC.accent : B.includes(e) ? AC.a2 : AC.line, e => A.includes(e) || B.includes(e) ? 4 : 1.5, e => e.id + (A.includes(e) ? " ∈ A" : B.includes(e) ? " ∈ B" : ""));
      const tx = 380;
      F.g.append("text").attr("x", tx).attr("y", 24).attr("font-size", 12).attr("fill", AC.accent).text(`A = {${A.map(e => e.id).join(", ")}}  forest, ${5 - A.length} trees`);
      F.g.append("text").attr("x", tx).attr("y", 44).attr("font-size", 12).attr("fill", AC.a2).text(`B = {${B.map(e => e.id).join(", ")}}  forest, ${5 - B.length} trees`);
      F.g.append("text").attr("x", tx).attr("y", 72).attr("font-size", 11).attr("fill", AC.muted).text("(M2): |A| < |B| ⇒ some x ∈ B − A extends A");
      ext.forEach((r, i) => F.g.append("text").attr("x", tx).attr("y", 94 + i * 18).attr("font-size", 12).attr("fill", r.ok ? AC.good : AC.bad).text(`A ∪ {${r.e.id}} acyclic? ${r.ok ? "yes — extends A" : "no (closes a cycle)"}`));
      F.g.append("text").attr("x", tx).attr("y", 170).attr("font-size", 11).attr("fill", AC.muted).text(`all 2^7 = 128 edge subsets enumerated: ${nIndep} forests;`);
      F.g.append("text").attr("x", tx).attr("y", 186).attr("font-size", 11).attr("fill", AC.muted).text(`every maximal forest has ${[...maximalSizes].join(" or ")} edges`);
      SX.setHtml("mat-readout", `A = {${A.map(e => e.id).join(", ")}} (${A.length} edges, ${5 - A.length} trees), B = {${B.map(e => e.id).join(", ")}} (${B.length} edges, ${5 - B.length} trees) · elements of B − A that extend A: <b>${ext.filter(r => r.ok).map(r => r.e.id).join(", ") || "none"}</b> — the axiom needs at least one ${SX.flag(ext.some(r => r.ok))} · ${c.get("find")} union-find steps in the tests · exhaustive: ${nIndep} of 128 subsets are forests; maximal forests all have ${[...maximalSizes].join("/")} = |V| − 1 edges ${SX.flag(maximalSizes.size === 1 && maximalSizes.has(4))} (all bases the same size — the rank)`);
    } else {
      const c = AL.counter(); const frames = [];
      const A = GR.matroidGreedy(E, indep, c, f => frames.push(f));
      const wA = A.reduce((s, e) => s + e.w, 0);
      function render(f, i) {
        F.g.selectAll("*").remove();
        const seen = frames.slice(0, i + 1);
        const st = new Map(); seen.forEach(fr => st.set(fr.x.id, fr.ok ? "in" : "out"));
        drawGraph(F.g, e => f.x.id === e.id ? AC.a2 : st.get(e.id) === "in" ? AC.good : st.get(e.id) === "out" ? AC.bad : AC.line, e => st.has(e.id) || f.x.id === e.id ? 4 : 1.5, e => `${e.id} w=${e.w}`);
        const tx = 380;
        F.g.append("text").attr("x", tx).attr("y", 24).attr("font-size", 12).attr("fill", AC.ink).text("edges in decreasing weight:");
        frames.forEach((fr, k) => F.g.append("text").attr("x", tx).attr("y", 46 + k * 18).attr("font-size", 12).attr("fill", k > i ? AC.muted : fr.ok ? AC.good : AC.bad).attr("opacity", k > i ? 0.5 : 1).text(`${fr.x.id} (w=${fr.x.w})  ${k > i ? "" : fr.ok ? "accept — still a forest" : "reject — closes a cycle"}`));
        F.g.append("text").attr("x", tx).attr("y", 46 + frames.length * 18 + 10).attr("font-size", 12).attr("fill", AC.ink).text(`A = {${f.A.map(e => e.id).join(", ")}}  weight ${f.A.reduce((s, e) => s + e.w, 0)}`);
      }
      AL.stepper(svg, { frames, render, delay: 800, label: "edge" });
      SX.setHtml("mat-readout", `greedy (decreasing weight, keep if acyclic) → forest {${A.map(e => e.id).join(", ")}}, weight <b>${wA}</b>, ${A.length} edges · measured: ${c.get("test")} independence tests, ${c.get("find")} union-find steps · brute force over all 128 subsets: maximum-weight forest weight <b>${brute.w}</b> {${brute.set.map(e => e.id).join(", ")}} ${SX.flag(brute.w === wA)} · this is the maximum spanning tree; sorting by increasing weight instead gives Kruskal's minimum spanning tree`);
    }
  }
  SX.on("mat-mode", "change", build);
  build();
})();

/* ── 10  #task-svg  unit-task scheduling as a matroid ─────────────────── */
(function () {
  if (!SX.has("task-svg")) return;
  const W = 680, H = 250;
  const BOOK = [[1,4,70],[2,2,60],[3,4,50],[4,3,40],[5,1,30],[6,4,20],[7,6,10]].map(p => ({ id: String(p[0]), d: p[1], w: p[2] }));
  function build() {
    const which = SX.val("task-inst", "book");
    let tasks;
    if (which === "book") tasks = BOOK;
    else { const r = AL.rng(41); tasks = Array.from({ length: 7 }, (_, i) => ({ id: String(i + 1), d: 1 + Math.floor(r() * 7), w: 10 + Math.floor(r() * 80) })); }
    const n = tasks.length;
    const c = AL.counter(); const frames = [];
    const A = GR.matroidGreedy(tasks, (s, cc) => GR.tasksIndep(s, cc), c, f => frames.push(f));
    const onTime = A.slice().sort((a, b) => a.d - b.d || a.id.localeCompare(b.id));
    const late = tasks.filter(t => !A.includes(t));
    const sched = onTime.concat(late);
    const pen = GR.penaltyOf(sched);
    /* verify the canonical schedule really has A on time */
    let allOnTime = true; onTime.forEach((t, i) => { if (i + 1 > t.d) allOnTime = false; });
    let best = Infinity; GR.permutations(tasks).forEach(p => { const q = GR.penaltyOf(p); if (q < best) best = q; });
    const svg = d3.select("#task-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 10, r: 10, t: 10, b: 10 });
    const order = tasks.slice().sort((a, b) => b.w - a.w || a.id.localeCompare(b.id));
    function render(f, i) {
      F.g.selectAll("*").remove();
      F.g.append("text").attr("x", 0).attr("y", 12).attr("font-size", 11).attr("fill", AC.muted).text("tasks in decreasing penalty — accept if every deadline count Nₜ stays ≤ t");
      const st = new Map(); frames.slice(0, i + 1).forEach(fr => st.set(fr.x.id, fr.ok));
      AL.row(F.g, order.map(t => `${t.id}`), { x: 0, y: 20, w: 60, h: 34, gap: 6, index: false, mark: (k) => { const t = order[k]; if (f.x.id === t.id) return AC.a2; return st.has(t.id) ? (st.get(t.id) ? AC.good : AC.bad) : AC.panel2; } });
      order.forEach((t, k) => F.g.append("text").attr("x", k * 66 + 30).attr("y", 68).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.muted).text(`d=${t.d} w=${t.w}`));
      /* N_t counts for the current candidate set */
      const cand = frames.slice(0, i).filter(fr => fr.ok).map(fr => fr.x).concat([f.x]);
      const counts = []; for (let t = 1; t <= n; t++) counts.push(cand.filter(a => a.d <= t).length);
      F.g.append("text").attr("x", 0).attr("y", 96).attr("font-size", 11).attr("fill", AC.ink).text(`try task ${f.x.id}: Nₜ for A ∪ {${f.x.id}} = [${counts.join(", ")}] against t = [${Array.from({ length: n }, (_, k) => k + 1).join(", ")}] → ${f.ok ? "accept" : "REJECT (some Nₜ > t)"}`);
      /* the canonical schedule of the accepted set so far */
      const acc = f.A.slice().sort((a, b) => a.d - b.d || a.id.localeCompare(b.id));
      F.g.append("text").attr("x", 0).attr("y", 128).attr("font-size", 11).attr("fill", AC.muted).text("canonical schedule of the accepted set (slots 1 … n; a task is on time if slot ≤ deadline):");
      const rest = tasks.filter(t => !f.A.includes(t) && st.has(t.id) && !st.get(t.id));
      const slots = acc.concat(rest);
      for (let k = 0; k < n; k++) { const t = slots[k]; F.g.append("rect").attr("x", k * 66).attr("y", 136).attr("width", 60).attr("height", 34).attr("rx", 4).attr("fill", t ? (k + 1 <= t.d ? AC.accent : AC.bad) : AC.panel).attr("stroke", AC.line);
        F.g.append("text").attr("x", k * 66 + 30).attr("y", 158).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", t ? AC.bg : AC.muted).text(t ? `${t.id} (d${t.d})` : "slot " + (k + 1)); }
      F.g.append("text").attr("x", 0).attr("y", 196).attr("font-size", 11).attr("fill", AC.ink).text(`on-time set so far {${f.A.map(t => t.id).join(", ")}} · penalty of rejected tasks so far ${rest.reduce((s, t) => s + t.w, 0)}`);
    }
    const st = AL.stepper(svg, { frames, render, delay: 900, label: "task" });
    st.go(frames.length - 1);                          // default: every task decided, the full schedule; ⟲ rewinds
    SX.setHtml("task-readout", `${n} tasks · greedy on-time set <b>{${A.map(t => t.id).join(", ")}}</b>, rejected {${late.map(t => t.id).join(", ") || "—"}}, canonical schedule ⟨${sched.map(t => t.id).join(", ")}⟩ · every accepted task really on time in that schedule ${SX.flag(allOnTime)} · total penalty <b>${pen}</b> · measured: ${c.get("test")} independence tests, ${c.get("count")} deadline counts · brute force over all ${n}! = ${SX.int(GR.permutations(tasks).length)} orders: minimum penalty <b>${best}</b> ${SX.flag(best === pen)}`);
  }
  SX.on("task-inst", "change", build);
  build();
})();

/* ── 11  #cut-svg  the cut property ───────────────────────────────────── */
(function () {
  if (!SX.has("cut-svg")) return;
  const W = 680, H = 280;
  const V = { A: [50, 50], B: [190, 30], C: [330, 50], D: [50, 210], E: [190, 230], F: [330, 210] };
  const E = [["AB",4],["AD",2],["BC",6],["BD",5],["BE",3],["CE",7],["CF",1],["DE",8],["EF",5]].map(p => ({ id: p[0], u: p[0][0], v: p[0][1], w: p[1] }));
  const names = Object.keys(V);
  const connected = edges => { const par = {}; names.forEach(n => par[n] = n); const find = x => { while (par[x] !== x) x = par[x]; return x; }; edges.forEach(e => { par[find(e.u)] = find(e.v); }); return names.every(n => find(n) === find(names[0])); };
  /* all spanning trees: 5-edge subsets that are acyclic (⇔ connected with 5 edges) */
  const trees = [];
  for (let mask = 0; mask < (1 << E.length); mask++) { const sub = E.filter((_, i) => mask & (1 << i)); if (sub.length === 5 && GR.acyclic(sub)) trees.push(sub); }
  const wOf = t => t.reduce((s, e) => s + e.w, 0);
  const minW = Math.min(...trees.map(wOf));
  const msts = trees.filter(t => wOf(t) === minW);
  function build() {
    const S = new Set(SX.val("cut-s", "A,B,D").split(","));
    const cross = E.filter(e => S.has(e.u) !== S.has(e.v));
    const light = cross.reduce((b, e) => (b === null || e.w < b.w) ? e : b, null);
    const mstWithLight = msts.find(t => t.includes(light));
    /* the exchange: the lightest spanning tree that AVOIDS e, then T ∪ {e} − {e'} */
    const avoid = trees.filter(t => !t.includes(light)).sort((a, b) => wOf(a) - wOf(b))[0];
    let ex = null;
    if (avoid) {
      /* the cycle in avoid ∪ {e}: the tree path between e's endpoints */
      const adj = {}; names.forEach(n => adj[n] = []); avoid.forEach(e => { adj[e.u].push({ to: e.v, e }); adj[e.v].push({ to: e.u, e }); });
      const path = []; const seen = new Set();
      (function dfs(x, target, acc) { if (x === target) { path.push(...acc); return true; } seen.add(x); for (const nb of adj[x]) { if (!seen.has(nb.to)) { if (dfs(nb.to, target, acc.concat([nb.e]))) return true; } } return false; })(light.u, light.v, []);
      const ePrime = path.find(e => S.has(e.u) !== S.has(e.v));
      const T2 = avoid.filter(e => e !== ePrime).concat([light]);
      ex = { T: avoid, path, ePrime, T2, w1: wOf(avoid), w2: wOf(T2), tree: GR.acyclic(T2) && T2.length === 5 };
    }
    const F = AL.frame("#cut-svg", W, H, { l: 10, r: 10, t: 10, b: 10 });
    const draw = (g, ox, colorOf, widthOf, dashOf, label) => {
      const gg = g.append("g").attr("transform", `translate(${ox},0)`);
      E.forEach(e => { const a = V[e.u], b = V[e.v]; const col = colorOf(e); if (!col) return; gg.append("line").attr("x1", a[0]).attr("y1", a[1]).attr("x2", b[0]).attr("y2", b[1]).attr("stroke", col).attr("stroke-width", widthOf(e)).attr("stroke-dasharray", dashOf(e));
        gg.append("text").attr("x", (a[0] + b[0]) / 2 + 5).attr("y", (a[1] + b[1]) / 2 - 3).attr("font-size", 10).attr("fill", AC.muted).text(e.w); });
      names.forEach(n => { gg.append("circle").attr("cx", V[n][0]).attr("cy", V[n][1]).attr("r", 12).attr("fill", S.has(n) ? AC.accent : AC.panel2).attr("stroke", AC.line); gg.append("text").attr("x", V[n][0]).attr("y", V[n][1] + 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", S.has(n) ? AC.bg : AC.ink).text(n); });
      gg.append("text").attr("x", 40).attr("y", 262).attr("font-size", 11).attr("fill", AC.muted).text(label);
      return gg;
    };
    draw(F.g, 0, e => e === light ? AC.a2 : cross.includes(e) ? AC.rose : AC.line, e => e === light ? 4 : cross.includes(e) ? 2 : 1.5, e => cross.includes(e) && e !== light ? "5 3" : null, `cut S = {${[...S].join(", ")}} · light edge ${light.id} (w = ${light.w})`);
    const mstT = mstWithLight || msts[0];
    draw(F.g, 330, e => mstT.includes(e) ? (e === light ? AC.a2 : AC.good) : AC.grid, e => mstT.includes(e) ? 3.5 : 1, () => null, `an MST (weight ${minW})${mstWithLight ? " containing " + light.id : " — none contains " + light.id + " (unexpected)"}`);
    SX.setHtml("cut-readout", `cut S = {${[...S].join(", ")}} · crossing edges ${cross.map(e => e.id + "(" + e.w + ")").join(", ")} · light edge <b>${light.id}</b>, weight ${light.w} · all ${trees.length} spanning trees enumerated (5-edge acyclic subsets of 9 edges): minimum weight <b>${minW}</b>, ${msts.length} minimum tree${msts.length > 1 ? "s" : ""} · some minimum tree contains ${light.id}: ${SX.flag(!!mstWithLight)}${ex ? ` · the exchange on the lightest tree that avoids ${light.id} (weight ${ex.w1}${ex.w1 === minW ? ", itself an MST" : ", not an MST"}): adding ${light.id} closes the cycle ${ex.path.map(e => e.id).join("–")}+${light.id}, whose other crossing edge is ${ex.ePrime.id} (w = ${ex.ePrime.w}); T ∪ {${light.id}} − {${ex.ePrime.id}} is a spanning tree ${SX.flag(ex.tree)} of weight ${ex.w2} ≤ ${ex.w1} ${SX.flag(ex.w2 <= ex.w1)}` : " · every spanning tree contains the light edge (it is a bridge)"}`);
  }
  SX.on("cut-s", "change", build);
  build();
})();

/* ── 12  #fuel-svg  fuel stops ────────────────────────────────────────── */
(function () {
  if (!SX.has("fuel-svg")) return;
  const W = 680, H = 150;
  function build() {
    const R = +SX.val("fuel-r", 6); SX.setText("fuel-r-out", R);
    const which = SX.val("fuel-inst", "book");
    let st, D;
    if (which === "book") { st = [3, 5, 9, 11, 14, 17]; D = 20; }
    else { const r = AL.rng(9); D = 40; st = Array.from({ length: 9 }, () => 1 + Math.floor(r() * 38)).sort((a, b) => a - b).filter((v, i, a) => a.indexOf(v) === i); }
    const c = AL.counter(); const frames = [{ pos: 0, stops: [] }];
    const g = GR.greedyFuel(st, R, D, c, f => frames.push(f));
    const b = GR.bruteFuel(st, R, D);
    const svg = d3.select("#fuel-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 30, r: 30, t: 30, b: 30 });
    const x = d3.scaleLinear().domain([0, D]).range([0, F.iw]);
    function render(f, i) {
      F.g.selectAll("*").remove();
      F.g.append("rect").attr("x", 0).attr("y", 30).attr("width", F.iw).attr("height", 14).attr("fill", AC.panel2).attr("stroke", AC.line);
      AL.axisB(F.g, x, 44, 10, "position");
      /* range arc from the current position */
      F.g.append("rect").attr("x", x(f.pos)).attr("y", 30).attr("width", x(Math.min(D, f.pos + R)) - x(f.pos)).attr("height", 14).attr("fill", AC.a2).attr("opacity", 0.35);
      st.forEach(s => { const on = f.stops.includes(s); F.g.append("line").attr("x1", x(s)).attr("x2", x(s)).attr("y1", 22).attr("y2", 52).attr("stroke", on ? AC.good : AC.muted).attr("stroke-width", on ? 3 : 1.5); F.g.append("text").attr("x", x(s)).attr("y", 16).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", on ? AC.good : AC.muted).text(s); });
      F.g.append("circle").attr("cx", x(f.pos)).attr("cy", 37).attr("r", 6).attr("fill", AC.accent);
      F.g.append("text").attr("x", x(D)).attr("y", 16).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.ink).text("D = " + D);
      F.g.append("text").attr("x", 0).attr("y", -12).attr("font-size", 11).attr("fill", AC.ink).text(i === 0 ? `at 0, range ${R}: reachable stations {${st.filter(s => s <= R).join(", ")}}` : `stop ${i} at ${f.pos}: the farthest station within ${R} of the previous position · ${f.pos + R >= D ? "destination reachable — done" : "reachable next {" + st.filter(s => s > f.pos && s <= f.pos + R).join(", ") + "}"}`);
    }
    AL.stepper(svg, { frames, render, delay: 900, label: "stop" });
    SX.setHtml("fuel-readout", `stations ${st.join(", ")}, road ${D}, range ${R} · greedy: ${g === null ? "<b>impossible</b> — a gap wider than R" : "<b>" + g.length + " stops</b> at " + g.join(", ")} · brute force over all 2^${st.length} = ${1 << st.length} station subsets: ${b === null ? "no feasible set of stops" : "minimum <b>" + b.length + "</b> (e.g. " + (b.join(", ") || "none needed") + ")"} ${SX.flag((g === null && b === null) || (g !== null && b !== null && g.length === b.length))} · measured: ${c.get("cmp")} station-in-range tests`);
  }
  SX.on("fuel-r", "input", build); SX.on("fuel-inst", "change", build);
  build();
})();

/* ── 13  #sc-svg  set cover stepped ───────────────────────────────────── */
(function () {
  if (!SX.has("sc-svg")) return;
  const W = 680, H = 260;
  function tight(k) {
    const half = (1 << k) - 1; const U = []; for (let i = 1; i <= 2 * half; i++) U.push(i);
    const row1 = U.slice(0, half), row2 = U.slice(half);
    const sets = [{ id: "R1", els: new Set(row1) }, { id: "R2", els: new Set(row2) }];
    let p1 = 0, p2 = 0;
    for (let j = k; j >= 1; j--) { const h = 1 << (j - 1); const els = row1.slice(p1, p1 + h).concat(row2.slice(p2, p2 + h)); p1 += h; p2 += h; sets.push({ id: "T" + j, els: new Set(els) }); }
    return { U, sets };
  }
  function build() {
    const which = SX.val("sc-inst", "book");
    let U, sets;
    if (which === "book") { U = [1,2,3,4,5,6,7,8]; sets = [["A",[1,2,3,4]],["B",[5,6,7,8]],["C",[1,2,3,5,6]],["D",[4,7]],["E",[8]],["F",[4,8]]].map(p => ({ id: p[0], els: new Set(p[1]) })); }
    else if (which.startsWith("tight")) ({ U, sets } = tight(+which.slice(5)));
    else { const r = AL.rng(5); U = Array.from({ length: 12 }, (_, i) => i + 1); sets = Array.from({ length: 7 }, (_, i) => ({ id: String.fromCharCode(65 + i), els: new Set(AL.shuffle(U, r).slice(0, 2 + Math.floor(r() * 5))) })); const cov = new Set(); sets.forEach(S => S.els.forEach(e => cov.add(e))); U.forEach(e => { if (!cov.has(e)) sets[e % 7].els.add(e); }); }
    const c = AL.counter(); const frames = [{ picked: [], unc: new Set(U), gains: null }];
    const picked = GR.greedyCover(U, sets, c, f => frames.push(f));
    const opt = which.startsWith("tight") ? ["R1", "R2"] : GR.bruteCover(U, sets);
    const n = U.length, Hn = GR.harmonic(n), k = +which.slice(5);
    const svg = d3.select("#sc-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 40, r: 10, t: 16, b: 10 });
    const cw = Math.min(40, F.iw / (n + 1)), rh = Math.min(30, (F.ih - 20) / (sets.length + 1));
    function render(f, i) {
      F.g.selectAll("*").remove();
      U.forEach((e, j) => F.g.append("text").attr("x", j * cw + cw / 2).attr("y", 0).attr("text-anchor", "middle").attr("font-size", n > 30 ? 7 : 10).attr("fill", f.unc.has(e) ? AC.ink : AC.muted).text(n > 30 ? (j % 5 === 0 ? e : "") : e));
      sets.forEach((S, r) => {
        const isP = f.picked.some(p => p.id === S.id), cur = f.pick && f.pick.id === S.id;
        F.g.append("text").attr("x", -6).attr("y", 8 + r * rh + rh / 2 + 4).attr("text-anchor", "end").attr("font-size", 10).attr("fill", cur ? AC.a2 : isP ? AC.good : AC.muted).text(S.id);
        U.forEach((e, j) => { if (!S.els.has(e)) return; F.g.append("rect").attr("x", j * cw + 1).attr("y", 8 + r * rh + 1).attr("width", cw - 2).attr("height", rh - 2).attr("rx", 2).attr("fill", cur ? AC.a2 : isP ? AC.good : AC.accent).attr("opacity", f.unc.has(e) ? 0.95 : 0.25); });
      });
      /* gains of this round */
      const gains = sets.map(S => { let g = 0; S.els.forEach(e => { if ((i === 0 ? new Set(U) : frames[i - 1].unc).has(e)) g++; }); return S.id + ":" + g; });
      F.g.append("text").attr("x", 0).attr("y", 8 + sets.length * rh + 14).attr("font-size", 10).attr("fill", AC.muted).text(i === 0 ? `gains before round 1 — ${gains.join("  ")}` : `round ${i}: picked ${f.pick.id} with gain ${f.gain} · uncovered left ${f.unc.size}`);
    }
    AL.stepper(svg, { frames, render, delay: 900, label: "round" });
    const ratio = opt ? picked.length / opt.length : NaN;
    SX.setHtml("sc-readout", `n = ${n} elements, ${sets.length} sets · greedy picks <b>${picked.length}</b>: ${picked.map(p => p.id + " (+" + p.gain + ")").join(", ")} · ${which.startsWith("tight") ? "optimum " + opt.length + " (the two rows; k = " + k + " = log₂(n/2 + 1))" : "brute force over all 2^" + sets.length + " subfamilies: optimum <b>" + (opt ? opt.length : "—") + "</b> {" + (opt || []).join(", ") + "}"} · ratio <b>${SX.f2(ratio)}</b> ≤ H(${n}) = ${SX.f2(Hn)} ${SX.flag(ratio <= Hn + 1e-9)} · measured: ${c.get("scan")} set scans · every element covered by the greedy family: ${SX.flag(U.every(e => picked.some(p => sets.find(S => S.id === p.id).els.has(e))))}`);
  }
  SX.on("sc-inst", "change", build);
  build();
})();

/* ── 14  #lb-svg  list scheduling ─────────────────────────────────────── */
(function () {
  if (!SX.has("lb-svg")) return;
  const W = 680, H = 220;
  function build() {
    const which = SX.val("lb-inst", "book");
    let jobs, m = 3;
    if (which === "book") jobs = [2, 3, 4, 6, 2, 2];
    else if (which === "lpt") jobs = [6, 4, 3, 2, 2, 2];
    else if (which === "tight") jobs = [1, 1, 1, 1, 1, 1, 3];
    else { const r = AL.rng(13); jobs = Array.from({ length: 8 }, () => 1 + Math.floor(r() * 9)); }
    const c = AL.counter(); const frames = [{ i: -1, load: new Array(m).fill(0), k: -1 }];
    const res = GR.listSchedule(jobs, m, c, f => frames.push(f));
    const opt = GR.bruteMakespan(jobs, m);
    const total = jobs.reduce((a, b) => a + b, 0), lb1 = Math.max(...jobs), lb2 = total / m, bound = (2 - 1 / m) * opt.makespan;
    const svg = d3.select("#lb-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 40, r: 16, t: 50, b: 36 });
    const x = d3.scaleLinear().domain([0, Math.max(bound, res.makespan) + 1]).range([0, F.iw]);
    const rh = Math.min(40, (F.ih - 10) / m);
    function render(f, i) {
      F.g.selectAll("*").remove();
      AL.axisB(F.g, x, F.ih, 8, "time");
      const load = new Array(m).fill(0);
      for (let r = 0; r < m; r++) F.g.append("text").attr("x", -6).attr("y", r * rh + rh / 2 + 4).attr("text-anchor", "end").attr("font-size", 11).attr("fill", AC.muted).text("M" + (r + 1));
      for (let j = 0; j <= f.i; j++) { const k = res.asg[j]; F.g.append("rect").attr("x", x(load[k])).attr("y", k * rh + 4).attr("width", x(load[k] + jobs[j]) - x(load[k])).attr("height", rh - 8).attr("rx", 3).attr("fill", j === f.i ? AC.a2 : AC.accent).attr("stroke", AC.bg);
        F.g.append("text").attr("x", x(load[k] + jobs[j] / 2)).attr("y", k * rh + rh / 2 + 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.bg).text(jobs[j]); load[k] += jobs[j]; }
      [[lb1, "LB1 = max pⱼ", AC.teal], [lb2, "LB2 = ∑pⱼ/m", AC.violet], [opt.makespan, "optimum", AC.good], [bound, "(2 − 1/m)·OPT", AC.rose]].forEach(([v, lab, col], q) => { F.g.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", 0).attr("y2", F.ih).attr("stroke", col).attr("stroke-dasharray", "3 3"); F.g.append("text").attr("x", x(v) + 2).attr("y", -40 + q * 11).attr("font-size", 9).attr("fill", col).text(lab); });
      F.g.append("text").attr("x", 0).attr("y", F.ih + 32).attr("font-size", 11).attr("fill", AC.ink).text(f.i < 0 ? "no jobs assigned yet" : `job ${f.i + 1} (p = ${jobs[f.i]}) → M${f.k + 1}, the least loaded · loads [${f.load.join(", ")}]`);
    }
    const st = AL.stepper(svg, { frames, render, delay: 800, label: "job" });
    st.go(frames.length - 1);                          // default: the finished schedule against the bounds; ⟲ rewinds
    SX.setHtml("lb-readout", `jobs ${jobs.join(", ")} on m = ${m} · list scheduling: loads [${res.load.join(", ")}], makespan <b>${res.makespan}</b> · lower bounds: max pⱼ = ${lb1}, ∑pⱼ/m = ${SX.f2(lb2)} · brute force over all ${m}^${jobs.length} = ${SX.int(Math.pow(m, jobs.length))} assignments: optimum <b>${opt.makespan}</b> · ratio ${SX.f3(res.makespan / opt.makespan)} ≤ 2 − 1/m = ${SX.f3(2 - 1 / m)} ${SX.flag(res.makespan <= bound + 1e-9)}${which === "tight" ? " — the tight family: ratio exactly 2 − 1/m " + SX.flag(Math.abs(res.makespan / opt.makespan - (2 - 1 / m)) < 1e-9) : ""} · measured: ${c.get("cmp")} load comparisons`);
  }
  SX.on("lb-inst", "change", build);
  build();
})();

/* ── 15  #tsp-svg  nearest-neighbour tour vs optimal ──────────────────── */
(function () {
  if (!SX.has("tsp-svg")) return;
  const W = 680, H = 280;
  function build() {
    const which = SX.val("tsp-inst", "rand"), M = +SX.val("tsp-m", 10); SX.setText("tsp-m-out", M);
    let pts, dist, names;
    if (which === "four") {
      names = ["A", "B", "C", "D"]; pts = [[160, 60], [420, 60], [420, 220], [160, 220]];
      const D = { AB: 1, BC: 1, CD: 1, AC: 2, BD: 2, DA: M };
      dist = (i, j) => { const k = names[i] + names[j], k2 = names[j] + names[i]; return D[k] !== undefined ? D[k] : D[k2]; };
    } else if (which === "line") {
      const xs = [0, -1, 2, -4, 8, -16, 32]; names = xs.map(String);
      const sx = d3.scaleLinear().domain([-16, 32]).range([40, 620]);
      pts = xs.map((v, i) => [sx(v), 60 + (i % 2) * 120 + i * 10]);
      dist = (i, j) => Math.abs(xs[i] - xs[j]);
    } else {
      const r = AL.rng(29); pts = Array.from({ length: 8 }, () => [40 + r() * 560, 30 + r() * 220]); names = pts.map((_, i) => String(i));
      dist = (i, j) => Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
    }
    const n = pts.length; const c = AL.counter();
    const nn = GR.nearestNeighbour(n, dist, 0, c), nnL = GR.tourLength(nn, dist);
    const opt = GR.bruteTSP(n, dist);
    /* triangle inequality check */
    let metric = true; for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) for (let d = 0; d < n; d++) if (dist(a, d) > dist(a, b) + dist(b, d) + 1e-9) metric = false;
    const F = AL.frame("#tsp-svg", W, H, { l: 0, r: 0, t: 0, b: 0 });
    const path = (tour, col, w, dash, off) => { const g = F.g.append("g"); for (let i = 0; i < n; i++) { const a = pts[tour[i]], b = pts[tour[(i + 1) % n]]; g.append("line").attr("x1", a[0] + off).attr("y1", a[1] + off).attr("x2", b[0] + off).attr("y2", b[1] + off).attr("stroke", col).attr("stroke-width", w).attr("stroke-dasharray", dash); } };
    path(opt.tour, AC.good, 5, null, 0); path(nn, AC.a2, 2.5, "6 3", 0);
    if (which === "four") { const pairs = [[0,1],[1,2],[2,3],[0,2],[1,3],[3,0]]; pairs.forEach(([a, b]) => { const p = pts[a], q = pts[b]; F.g.append("line").attr("x1", p[0]).attr("y1", p[1]).attr("x2", q[0]).attr("y2", q[1]).attr("stroke", AC.grid).attr("stroke-width", 1); F.g.append("text").attr("x", (p[0] + q[0]) / 2 + 6).attr("y", (p[1] + q[1]) / 2 - 4).attr("font-size", 11).attr("fill", AC.muted).text(dist(a, b)); }); }
    pts.forEach((p, i) => { F.g.append("circle").attr("cx", p[0]).attr("cy", p[1]).attr("r", 9).attr("fill", i === 0 ? AC.accent : AC.panel2).attr("stroke", AC.line); F.g.append("text").attr("x", p[0]).attr("y", p[1] + 4).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.ink).text(names[i]); const k = nn.indexOf(i); F.g.append("text").attr("x", p[0] + 12).attr("y", p[1] - 8).attr("font-size", 9).attr("fill", AC.a2).text("#" + (k + 1)); });
    AL.legend(F.g, [{ label: "nearest-neighbour tour (numbers = visiting order)", color: AC.a2, dash: "6 3" }, { label: "optimal tour (brute force)", color: AC.good }], 10, H - 26);
    SX.setHtml("tsp-readout", `${n} cities, start ${names[0]} · nearest neighbour: ${nn.map(i => names[i]).join(" → ")} → ${names[0]}, length <b>${SX.f2(nnL)}</b> (${c.get("cmp")} distance comparisons) · brute force over all ${n - 1}!${n > 4 ? " = " + SX.int(GR.permutations(Array.from({ length: n - 1 }, (_, i) => i)).length) : ""} tours from ${names[0]}: optimum <b>${SX.f2(opt.len)}</b> (${opt.tour.map(i => names[i]).join(" → ")} → ${names[0]}) · ratio <b>${SX.f3(nnL / opt.len)}</b> · triangle inequality holds: ${metric ? "yes (metric)" : "<b>no</b> — the ratio (M + 3)/6 grows without bound in M"}${which === "line" ? " · on a line the optimum is twice the span; nearest neighbour zigzags across it" : ""}${which === "rand" && metric ? " · metric bound (ceil(log₂ n) + 1)/2 = " + SX.f2((Math.ceil(Math.log2(n)) + 1) / 2) + " " + SX.flag(nnL / opt.len <= (Math.ceil(Math.log2(n)) + 1) / 2 + 1e-9) : ""}`);
  }
  SX.on("tsp-inst", "change", build); SX.on("tsp-m", "input", build);
  build();
})();

/* ── 16  #flow-svg  the decision flow ─────────────────────────────────── */
(function () {
  if (!SX.has("flow-svg")) return;
  const W = 680, H = 300;
  const nodes = [
    { id: "q1", x: 400, y: 30, w: 270, t: "optimal substructure? (write the recurrence)", sec: "§02 — cut-and-paste: an optimal solution contains optimal solutions of its subproblems", href: "#two-properties" },
    { id: "srch", x: 115, y: 30, w: 210, t: "no recurrence → search / heuristics", sec: "§26 — nearest-neighbour tours and the other rules with no proof", href: "#heuristics" },
    { id: "q2", x: 400, y: 105, w: 340, t: "greedy-choice property? (prove it, or hunt a counterexample)", sec: "§03 — stays ahead, exchange, or matroid; §05, §09, §11 — how a counterexample is found", href: "#proof-templates" },
    { id: "gr", x: 115, y: 180, w: 200, t: "GREEDY — one sort, one scan", sec: "§04 activity selection · §07 partitioning · §08 lateness · §09 canonical coins · §10 fractional knapsack · §13 Huffman · §18 matroids · §19 unit tasks · §20 MST · §21 fuel stops · §24 Horn", href: "#activity-selection" },
    { id: "q3", x: 400, y: 180, w: 250, t: "polynomially many subproblems?", sec: "§27 — count the distinct states of the recurrence", href: "#when-greedy-fails" },
    { id: "dp", x: 250, y: 260, w: 250, t: "DYNAMIC PROGRAMMING — fill the table", sec: "weighted intervals (§06) · 0/1 knapsack (§11) · non-canonical coins (§09) — the Dynamic Programming page", href: "dynamic-programming.html" },
    { id: "ap", x: 530, y: 260, w: 280, t: "NP-hard: APPROXIMATE or BRANCH & BOUND", sec: "set cover H(n) (§22) · load balancing 2 − 1/m (§23) · knapsack 1/2 (§11) · B&B with the greedy relaxation as bound (§27)", href: "#set-cover" }
  ];
  const edges = [["q1", "srch", "no"], ["q1", "q2", "yes"], ["q2", "gr", "proved"], ["q2", "q3", "counterexample"], ["q3", "dp", "yes"], ["q3", "ap", "no (exponential)"]];
  const byId = id => nodes.find(nd => nd.id === id);
  const F = AL.frame("#flow-svg", W, H, { l: 0, r: 0, t: 0, b: 0 });
  edges.forEach(([a, b, lab]) => { const p = byId(a), q = byId(b);
    if (p.y === q.y) { AL.arrow(F.g, p.x - p.w / 2, p.y, q.x + q.w / 2, q.y, { color: AC.muted }); F.g.append("text").attr("x", (p.x - p.w / 2 + q.x + q.w / 2) / 2).attr("y", p.y - 6).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.a2).text(lab); }
    else { AL.arrow(F.g, p.x, p.y + 16, q.x, q.y - 16, { color: AC.muted }); F.g.append("text").attr("x", (p.x + q.x) / 2 + 6).attr("y", (p.y + q.y) / 2 + 4).attr("font-size", 10).attr("fill", AC.a2).text(lab); } });
  const gs = F.g.selectAll("g.node").data(nodes).join("g").attr("class", "node").attr("transform", d => `translate(${d.x - d.w / 2},${d.y - 16})`).attr("tabindex", 0).style("cursor", "pointer");
  gs.append("rect").attr("width", d => d.w).attr("height", 32).attr("rx", 6).attr("fill", d => d.id === "gr" ? AC.good : d.id === "dp" ? AC.accent : d.id === "ap" ? AC.rose : AC.panel2).attr("stroke", AC.line);
  gs.append("text").attr("x", d => d.w / 2).attr("y", 20).attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", d => ["gr", "dp", "ap"].includes(d.id) ? AC.bg : AC.ink).text(d => d.t);
  const show = d => SX.setHtml("flow-readout", `<b>${d.t}</b> — ${d.sec} · <a href="${d.href}">go there</a>`);
  gs.on("mouseenter", (ev, d) => show(d)).on("focus", (ev, d) => show(d)).on("click", (ev, d) => { show(d); });
  show(byId("q2"));
  SX.setHtml("flow-readout", `hover or focus a node · default: <b>greedy-choice property?</b> — §03 gives the three proof templates; §05, §09, §11 show how a three-element counterexample is found · on this page ${nodes.find(n => n.id === "gr").sec.split("·").length} problems ended at GREEDY, 3 at DYNAMIC PROGRAMMING, 3 at APPROXIMATE`);
})();

/* ── GA  the load-time audit: every greedy answer vs an independent computation over random instances ── */
(function () {
  if (!SX.has("audit-readout")) return;
  const out = [];
  const c = AL.counter();
  /* 1. activity selection: earliest finish vs brute force; the other rules' failure counts */
  { const T = 300; let ok = 0; const fails = { es: 0, sd: 0, fc: 0 };
    for (let t = 0; t < T; t++) { const acts = INST.randomActs(8, 40, 5000 + t); const opt = GR.bruteSelect(acts).length; if (GR.selectEF(acts, c).length === opt) ok++; ["es", "sd", "fc"].forEach(r => { if (GR.selectByRule(acts, r, c).length < opt) fails[r]++; }); }
    out.push(`activity selection: earliest finish = brute force on <b>${ok}/${T}</b> random 8-activity instances ${SX.flag(ok === T)} (earliest start failed ${fails.es}, shortest duration ${fails.sd}, fewest conflicts ${fails.fc})`); }
  /* 2. interval partitioning: rooms = depth; finish order exceeds depth how often */
  { const T = 500; let ok = 0, over = 0;
    for (let t = 0; t < T; t++) { const acts = INST.randomActs(12, 30, t + 50); const d = GR.depth(acts); if (GR.partition(acts, c).rooms === d) ok++;
      const sorted = acts.slice().sort((a, b) => a.f - b.f || a.s - b.s); const rooms = []; sorted.forEach(a => { let r = -1; for (let k = 0; k < rooms.length; k++) if (rooms[k] <= a.s) { r = k; break; } if (r < 0) rooms.push(a.f); else rooms[r] = Math.max(rooms[r], a.f); }); if (rooms.length > d) over++; }
    out.push(`interval partitioning: start-order rooms = depth on <b>${ok}/${T}</b> ${SX.flag(ok === T)}; finish-order exceeded the depth on <b>${over}/${T}</b>`); }
  /* 3. minimum lateness: EDF = brute force over all 6! orders; total-lateness counterexample */
  { const T = 150; let ok = 0; const r = AL.rng(77);
    for (let t = 0; t < T; t++) { const jobs = Array.from({ length: 6 }, (_, i) => ({ id: String.fromCharCode(65 + i), t: 1 + Math.floor(r() * 5), d: 2 + Math.floor(r() * 14) })); if (GR.lateness(GR.edf(jobs)).maxL === GR.bruteLateness(jobs).maxL) ok++; }
    const js = [["A",1,9],["B",3,5],["C",5,3],["D",3,2]].map(p => ({ id: p[0], t: p[1], d: p[2] }));
    const tot = o => { let time = 0, s = 0; o.forEach(j => { time += j.t; s += Math.max(0, time - j.d); }); return s; };
    const edfTot = tot(GR.edf(js)); let best = Infinity, bestO = null; GR.permutations(js).forEach(p => { const v = tot(p); if (v < best) { best = v; bestO = p; } });
    out.push(`minimum lateness: earliest deadline = brute force (720 orders) on <b>${ok}/${T}</b> ${SX.flag(ok === T)}; total-lateness instance: deadline order ${edfTot}, best ${best} (${bestO.map(j => j.id).join("")}) ${SX.flag(edfTot === 15 && best === 11)}`); }
  /* 4. coin change: canonical vs non-canonical, and the three-coin characterisation */
  { const sys = { "1,5,10,25": 0, "1,3,4": 0, "1,5,10,20,25": 0, "1,10,25": 0 };
    Object.keys(sys).forEach(k => { const coins = k.split(",").map(Number); for (let x = 1; x <= 100; x++) if (GR.greedyCoins(coins, x, c).length > GR.dpCoins(coins, x, c).length) sys[k]++; });
    let checked = 0, mism = 0;
    for (let a = 2; a <= 79; a++) for (let b = a + 1; b <= 80; b++) { const coins = [1, a, b]; let canon = true; for (let x = 1; x <= a + b; x++) if (GR.greedyCoins(coins, x).length > GR.dpCoins(coins, x).length) { canon = false; break; } const q = Math.floor(b / a), rr = b % a; const pred = !(rr > 0 && rr < a - q); checked++; if (pred !== canon) mism++; }
    out.push(`coin change, amounts 1–100: greedy worse than DP on {1,5,10,25} <b>${sys["1,5,10,25"]}</b>, {1,3,4} <b>${sys["1,3,4"]}</b>, {1,5,10,20,25} <b>${sys["1,5,10,20,25"]}</b>, {1,10,25} <b>${sys["1,10,25"]}</b> ${SX.flag(sys["1,5,10,25"] === 0 && sys["1,3,4"] === 24 && sys["1,5,10,20,25"] === 15 && sys["1,10,25"] === 30)}; three-coin rule "canonical iff r = 0 or r ≥ a − q" vs greedy-vs-DP on <b>${checked}</b> systems {1, a, b}: ${mism} mismatches ${SX.flag(mism === 0)}`); }
  /* 5. knapsack: fractional ≥ 0/1 brute; fixed greedy ≥ OPT/2; how often plain greedy is optimal */
  { const T = 300; let okF = 0, okH = 0, plain = 0; const r = AL.rng(3);
    for (let t = 0; t < T; t++) { const items = Array.from({ length: 8 }, (_, i) => ({ id: String(i), w: 1 + Math.floor(r() * 20), v: 1 + Math.floor(r() * 50) })); const W0 = 30; const fit = items.filter(it => it.w <= W0); const fr = GR.fractionalKnapsack(fit, W0).val, g = GR.greedy01(fit, W0).val, o = GR.brute01(fit, W0).val, M = GR.bestSingle(fit, W0); if (fr >= o - 1e-9) okF++; if (Math.max(g, M) >= o / 2) okH++; if (g === o) plain++; }
    out.push(`knapsack, 300 random 8-item instances: fractional ≥ 0/1 optimum <b>${okF}/${T}</b> ${SX.flag(okF === T)}; max(greedy, single) ≥ OPT/2 <b>${okH}/${T}</b> ${SX.flag(okH === T)}; plain density greedy happened to be optimal on ${plain}/${T}`); }
  /* 6. Huffman: cost = exhaustive minimum over all trees; H ≤ L < H + 1 */
  { const T = 200; let okB = 0, okH = 0; const r = AL.rng(8);
    for (let t = 0; t < T; t++) { const f = Array.from({ length: 5 }, (_, i) => ({ sym: "abcde"[i], w: 1 + Math.floor(r() * 40) })); const root = GR.huffman(f, c); const B = GR.costOf(root); if (B === GR.bruteHuffmanCost(f)) okB++; const tot = f.reduce((s, x) => s + x.w, 0), L = B / tot, Hh = GR.entropy(f); if (Hh - 1e-9 <= L && L < Hh + 1) okH++; }
    out.push(`Huffman, 200 random 5-symbol tables: cost = exhaustive minimum over all 105 trees <b>${okB}/${T}</b> ${SX.flag(okB === T)}; H ≤ L &lt; H + 1 <b>${okH}/${T}</b> ${SX.flag(okH === T)}`); }
  /* 7. matroid greedy on random weighted graphic matroids vs brute force */
  { const T = 150; let ok = 0; const r = AL.rng(21);
    for (let t = 0; t < T; t++) { const edges = []; const seen = new Set(); while (edges.length < 7) { const u = 1 + Math.floor(r() * 5), v = 1 + Math.floor(r() * 5); if (u === v) continue; const k = Math.min(u, v) + "-" + Math.max(u, v); if (seen.has(k)) continue; seen.add(k); edges.push({ id: k, u, v, w: 1 + Math.floor(r() * 20) }); }
      const g = GR.matroidGreedy(edges, s => GR.acyclic(s), c).reduce((s, e) => s + e.w, 0); if (g === GR.bruteMatroid(edges, s => GR.acyclic(s)).w) ok++; }
    out.push(`matroid greedy, 150 random weighted graphs (5 vertices, 7 edges): max-weight forest = brute force over 128 subsets <b>${ok}/${T}</b> ${SX.flag(ok === T)}`); }
  /* 8. unit-task scheduling vs all 6! orders */
  { const T = 100; let ok = 0; const r = AL.rng(33);
    for (let t = 0; t < T; t++) { const tasks = Array.from({ length: 6 }, (_, i) => ({ id: String(i + 1), d: 1 + Math.floor(r() * 6), w: 1 + Math.floor(r() * 50) })); const A = GR.matroidGreedy(tasks, s => GR.tasksIndep(s), c); const pen = tasks.filter(x => !A.includes(x)).reduce((s, x) => s + x.w, 0); let best = Infinity; GR.permutations(tasks).forEach(p => { const q = GR.penaltyOf(p); if (q < best) best = q; }); if (pen === best) ok++; }
    out.push(`unit-task scheduling, 100 random 6-task instances: greedy penalty = brute force over 720 orders <b>${ok}/${T}</b> ${SX.flag(ok === T)}`); }
  /* 9. set cover: greedy ≤ H(n)·OPT; how often optimal */
  { const T = 200; let okH = 0, exact = 0; const r = AL.rng(44);
    for (let t = 0; t < T; t++) { const U = Array.from({ length: 10 }, (_, i) => i + 1); const sets = Array.from({ length: 6 }, (_, i) => ({ id: String.fromCharCode(65 + i), els: new Set(AL.shuffle(U, r).slice(0, 2 + Math.floor(r() * 4))) })); const cov = new Set(); sets.forEach(S => S.els.forEach(e => cov.add(e))); U.forEach(e => { if (!cov.has(e)) sets[e % 6].els.add(e); });
      const g = GR.greedyCover(U, sets, c).length, o = GR.bruteCover(U, sets).length; if (g <= GR.harmonic(10) * o + 1e-9) okH++; if (g === o) exact++; }
    out.push(`set cover, 200 random instances (10 elements, 6 sets): greedy ≤ H(10)·OPT <b>${okH}/${T}</b> ${SX.flag(okH === T)}; greedy exactly optimal on ${exact}/${T}`); }
  /* 10. load balancing: LS ≤ (2 − 1/m)·OPT */
  { const T = 200; let ok = 0, exact = 0; const r = AL.rng(55);
    for (let t = 0; t < T; t++) { const jobs = Array.from({ length: 7 }, () => 1 + Math.floor(r() * 9)); const m = 3; const ls = GR.listSchedule(jobs, m, c).makespan, o = GR.bruteMakespan(jobs, m).makespan; if (ls <= (2 - 1 / m) * o + 1e-9) ok++; if (ls === o) exact++; }
    out.push(`load balancing, 200 random instances (7 jobs, 3 machines): list scheduling ≤ (2 − 1/3)·OPT <b>${ok}/${T}</b> ${SX.flag(ok === T)}; exactly optimal on ${exact}/${T}`); }
  /* 11. fuel stops vs brute force */
  { const T = 200; let ok = 0; const r = AL.rng(66);
    for (let t = 0; t < T; t++) { const st = Array.from({ length: 8 }, () => 1 + Math.floor(r() * 38)).sort((a, b) => a - b).filter((v, i, a) => a.indexOf(v) === i); const R = 5 + Math.floor(r() * 8); const g = GR.greedyFuel(st, R, 40, c), b = GR.bruteFuel(st, R, 40); if ((g === null && b === null) || (g && b && g.length === b.length)) ok++; }
    out.push(`fuel stops, 200 random roads: greedy stop count = brute force over all station subsets <b>${ok}/${T}</b> ${SX.flag(ok === T)}`); }
  /* 12. Horn formulas vs brute force; the invariant */
  { const T = 200; let ok = 0, inv = 0, sat = 0; const r = AL.rng(88); const vars = ["u", "v", "w", "x", "y", "z"];
    for (let t = 0; t < T; t++) { const impl = Array.from({ length: 3 + Math.floor(r() * 5) }, () => ({ body: AL.shuffle(vars, r).slice(0, Math.floor(r() * 3)), head: vars[Math.floor(r() * 6)] })); const neg = Array.from({ length: 1 + Math.floor(r() * 3) }, () => AL.shuffle(vars, r).slice(0, 1 + Math.floor(r() * 3)));
      const h = GR.horn(vars, impl, neg, c), b = GR.bruteHorn(vars, impl, neg); if (h.sat === (b !== null)) ok++; if (b !== null) { sat++; let all = true; h.T.forEach(v => { if (!b.has(v)) all = false; }); if (all) inv++; } }
    out.push(`Horn formulas, 200 random (6 variables): stingy verdict = brute force over 64 assignments <b>${ok}/${T}</b> ${SX.flag(ok === T)}; on the ${sat} satisfiable ones every forced variable is true in the brute-force model ${inv}/${sat} ${SX.flag(inv === sat)}`); }
  const allOk = !out.some(s => s.includes("DISAGREE"));
  SX.setHtml("audit-readout", `<b>Load-time audit</b> — ${out.length} families of random instances, every greedy answer recomputed independently:<br>` + out.map(s => "· " + s).join("<br>") + `<br>· overall: ${allOk ? '<span style="color:' + AC.good + '">all audits agree</span>' : '<span style="color:' + AC.bad + '">SOME AUDIT DISAGREES</span>'} · total instrumented work in the audit: ${SX.int(Object.values(c.all()).reduce((a, b) => a + b, 0))} counted operations`);
  SX.setHtml("audit-cell", allOk ? "all " + out.length + " families agree — details in the readout below" : "a family DISAGREES — see below");
})();

/* @@FIGURES-END@@ */
})();
