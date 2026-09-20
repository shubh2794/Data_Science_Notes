/* shortest-paths-mst-flow.viz.js — figures for dsa/algorithms/shortest-paths-mst-flow.html
   (part 6 of the Algorithm Design & Analysis series).

   Loaded after ../../data.js, ../../notes.js and ../dsa-viz.js, so the globals
   AC (palette) and AL (rng/frame/axisB/axisL/row/binTree/counter/stepper/…) are
   available.

   House rule obeyed throughout: every count this page DISPLAYS — relaxations,
   heap operations, finds, unions, augmentations, pushes, relabels — is produced
   by running the real routine under AL.counter() and reading the counter back.
   Every ANSWER is cross-checked in the same figure against an independent
   computation — brute force over all simple paths / spanning trees / cuts /
   matchings for small graphs, or a second algorithm (Bellman-Ford against
   Dijkstra, Prim against Kruskal against Borůvka, Edmonds-Karp against
   push-relabel against the minimum cut) — and printed with an agree / DISAGREE
   flag. A load-time audit (SA, at the bottom) repeats those checks over
   hundreds of random instances and reports the counts in the cheat sheet.

   Layout of this file:
     SR — the instrumented routines, pure and DOM-free (also loadable in node).
     SI — the shared instances (graphs) so prose and figures agree by construction.
     SX — small DOM helpers shared by the figures.
     figures — one block per <svg>, in page order (see the page's outline).
     SA — the load-time audit. */

/* ═══════════════════════════════════════════════════════════════════════════
   SR — the instrumented routines (each chunk appends with Object.assign)
   ═══════════════════════════════════════════════════════════════════════════ */
const SR = {};

/* the shared instances, so prose and figures agree by construction */
const SI = {};

/* ── chunk A routines ── */
/* ── chunk A routines: relaxation, brute force over simple paths, Dijkstra
      (index heap / array scan / lazy deletion), Bellman-Ford with the negative
      cycle reported, random weighted graphs ─────────────────────────────── */
Object.assign(SR, (function () {
  const nop = { add: function () {}, get: function () { return 0; } };
  const INF = Infinity;

  /* ── adjacency from a shared instance {n, edges:[{u,v,w}], directed} ────
     undirected graphs become both arcs; each arc remembers its edge index. */
  function spAdj(G) {
    const adj = Array.from({ length: G.n }, () => []);
    G.edges.forEach((e, i) => { adj[e.u].push({ u: e.u, v: e.v, w: e.w, i }); if (G.directed === false) adj[e.v].push({ u: e.v, v: e.u, w: e.w, i }); });
    return adj;
  }
  /* every arc in EDGE-LIST order (an undirected edge contributes both arcs, adjacent) */
  function spArcs(G) { const out = []; G.edges.forEach((e, i) => { out.push({ u: e.u, v: e.v, w: e.w, i }); if (G.directed === false) out.push({ u: e.v, v: e.u, w: e.w, i }); }); return out; }
  function pathWeight(G, path) { const adj = spAdj(G); let s = 0; for (let i = 1; i < path.length; i++) { const a = adj[path[i - 1]].find(x => x.v === path[i]); if (!a) return INF; s += a.w; } return s; }

  /* ── brute force: every SIMPLE path from s, by DFS with a visited set ──
     returns d[v] = min weight over simple paths (∞ if unreachable), the path that
     attains it, the number of simple paths enumerated (the empty path at s counts),
     and the number of paths ending at each vertex. Exponential; small graphs only. */
  function bruteAllPaths(G, s) {
    const n = G.n, adj = spAdj(G); const d = new Array(n).fill(INF), best = new Array(n).fill(null), cnt = new Array(n).fill(0);
    const seen = new Array(n).fill(false); const path = []; let paths = 0;
    (function go(u, len) { seen[u] = true; path.push(u); paths++; cnt[u]++; if (len < d[u]) { d[u] = len; best[u] = path.slice(); }
      adj[u].forEach(a => { if (!seen[a.v]) go(a.v, len + a.w); }); path.pop(); seen[u] = false; })(s, 0);
    return { d, best, paths, cnt };
  }
  /* ── brute force: every simple cycle reachable from s; the lightest one ── */
  function bruteCycles(G, s) {
    const n = G.n, adj = spAdj(G); const reach = spReach(G, [s]);
    let count = 0, minW = INF, minCycle = null; const seen = new Array(n).fill(false); const path = [];
    for (let r = 0; r < n; r++) { if (!reach[r]) continue;
      (function go(u, len) { seen[u] = true; path.push(u);
        adj[u].forEach(a => { if (a.v === r) { count++; if (len + a.w < minW) { minW = len + a.w; minCycle = path.slice(); } } else if (a.v > r && !seen[a.v]) go(a.v, len + a.w); });
        path.pop(); seen[u] = false; })(r, 0); }
    return { count, minW, minCycle, negative: minW < 0 };
  }
  /* forward reachability from a set of vertices */
  function spReach(G, from) { const adj = spAdj(G); const seen = new Array(G.n).fill(false); const st = from.slice(); st.forEach(v => { seen[v] = true; }); while (st.length) { const u = st.pop(); adj[u].forEach(a => { if (!seen[a.v]) { seen[a.v] = true; st.push(a.v); } }); } return seen; }
  /* the set of vertices with δ = −∞: everything reachable from a negative cycle reachable from s (brute force) */
  function bruteNegInf(G, s) {
    const n = G.n, adj = spAdj(G); const reach = spReach(G, [s]); const onNeg = new Array(n).fill(false);
    const seen = new Array(n).fill(false); const path = [];
    for (let r = 0; r < n; r++) { if (!reach[r]) continue;
      (function go(u, len) { seen[u] = true; path.push(u);
        adj[u].forEach(a => { if (a.v === r) { if (len + a.w < 0) path.forEach(x => { onNeg[x] = true; }); } else if (a.v > r && !seen[a.v]) go(a.v, len + a.w); });
        path.pop(); seen[u] = false; })(r, 0); }
    const from = []; for (let v = 0; v < n; v++) if (onNeg[v]) from.push(v);
    return spReach(G, from);
  }

  /* ── RELAX in a fixed edge order, pass after pass, until nothing changes ──
     order: an array of arc objects {u,v,w,i}. c.add("relax") per test, c.add("improve")
     per successful lowering. rec(frame) after every relaxation with d, pi, the arc, the
     pass number and whether it improved. Stops after a pass with no improvement or n passes. */
  function relaxSequence(G, s, order, c, rec, maxPasses) {
    c = c || nop; const n = G.n; const d = new Array(n).fill(INF), pi = new Array(n).fill(-1); d[s] = 0;
    const passes = []; const lim = maxPasses || n;
    for (let p = 1; p <= lim; p++) { let changed = 0;
      order.forEach(a => { c.add("relax"); let imp = false; if (d[a.u] < INF && d[a.u] + a.w < d[a.v]) { d[a.v] = d[a.u] + a.w; pi[a.v] = a.u; imp = true; changed++; c.add("improve"); }
        if (rec) rec({ pass: p, arc: a, improved: imp, d: d.slice(), pi: pi.slice() }); });
      passes.push({ pass: p, changed, d: d.slice() }); if (!changed) break; }
    return { d, pi, passes };
  }
  /* the edges of the shortest s ⇝ t path (by brute force), as arcs in path order */
  function spPathArcs(G, s, t) { const b = bruteAllPaths(G, s); const p = b.best[t]; if (!p) return []; const adj = spAdj(G); const out = []; for (let i = 1; i < p.length; i++) { let bestA = null; adj[p[i - 1]].forEach(a => { if (a.v === p[i] && (!bestA || a.w < bestA.w)) bestA = a; }); out.push(bestA); } return out; }
  /* an arc order that puts a given arc list first (or last), the rest in edge-list order */
  function spOrderWith(G, first, atEnd) { const all = spArcs(G); const idx = new Set(first.map(a => a.i)); const rest = all.filter(a => !idx.has(a.i)); return atEnd ? rest.concat(first) : first.concat(rest); }

  /* ── an index min-heap with decrease-key; c.add("sift") per level moved ── */
  function spHeap(c) {
    c = c || nop; const key = [], item = [], pos = {};
    function swap(i, j) { const k = key[i]; key[i] = key[j]; key[j] = k; const t = item[i]; item[i] = item[j]; item[j] = t; pos[item[i]] = i; pos[item[j]] = j; }
    function up(i) { while (i > 0) { const p = (i - 1) >> 1; c.add("sift"); if (key[p] <= key[i]) break; swap(i, p); i = p; } }
    function down(i) { const n = key.length; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < n && key[l] < key[m]) m = l; if (r < n && key[r] < key[m]) m = r; c.add("sift"); if (m === i) break; swap(i, m); i = m; } }
    return {
      size: () => key.length, has: h => pos[h] !== undefined, keyOf: h => key[pos[h]],
      push: (h, k) => { key.push(k); item.push(h); pos[h] = key.length - 1; up(key.length - 1); },
      decrease: (h, k) => { const i = pos[h]; key[i] = k; up(i); },
      pop: () => { const h = item[0], k = key[0]; const last = key.length - 1; swap(0, last); key.pop(); item.pop(); delete pos[h]; if (key.length) down(0); return { item: h, key: k }; },
      entries: () => item.map((h, i) => ({ item: h, key: key[i] }))
    };
  }

  /* ── Dijkstra with an index heap (decrease-key), every vertex in the queue from the start ──
     S = finalised set; when u is extracted its out-arcs are relaxed; an arc into S is skipped
     (counted as "intoS") — the finalised value is never touched again.  Counts: "extract",
     "relax" (arcs examined), "decrease" (successful relaxations = decrease-keys), "sift" (heap
     levels moved), "intoS".  Ties in the queue break on the smaller key then the smaller vertex
     id.  opt.relaxIntoS = true relaxes edges into S too (the relax-anyway variant); the default skips them.
     rec(frame) once per extraction. */
  function dijkstra(G, s, c, rec, opt) {
    c = c || nop; opt = opt || {}; const n = G.n, adj = spAdj(G); const d = new Array(n).fill(INF), pi = new Array(n).fill(-1), inS = new Array(n).fill(false); d[s] = 0;
    const H = spHeap(c); for (let v = 0; v < n; v++) H.push(v, v === s ? 0 : INF);
    const order = []; let ties = 0;
    while (H.size()) {
      const ent = H.entries(); const mn = Math.min(...ent.map(e => e.key)); const tied = ent.filter(e => e.key === mn).map(e => e.item).sort((a, b) => a - b);
      const u = tied[0]; if (tied.length > 1 && mn < INF) ties++;
      /* extract u: pop by decreasing its key below everything (tie-break by id made explicit) */
      H.decrease(u, -INF); const got = H.pop().item; c.add("extract"); if (got !== u) throw new Error("heap tie-break");
      inS[u] = true; order.push(u); const events = [];
      if (d[u] < INF) adj[u].forEach(a => { if (inS[a.v]) { c.add("intoS");
          /* opt.relaxIntoS: the relax-anyway variant, which relaxes the edge regardless — the value changes, but v is never re-extracted, so nothing downstream sees it */
          if (opt.relaxIntoS && d[u] + a.w < d[a.v]) { const old = d[a.v]; d[a.v] = d[u] + a.w; pi[a.v] = u; c.add("lateFix"); events.push({ arc: a, kind: "intoS-changed", old, now: d[a.v] }); } else events.push({ arc: a, kind: "intoS" }); return; } c.add("relax");
        if (d[u] + a.w < d[a.v]) { const old = d[a.v]; d[a.v] = d[u] + a.w; pi[a.v] = u; H.decrease(a.v, d[a.v]); c.add("decrease"); events.push({ arc: a, kind: "improve", old, now: d[a.v] }); }
        else events.push({ arc: a, kind: "no" }); });
      if (rec) rec({ u, d: d.slice(), pi: pi.slice(), inS: inS.slice(), events, frontier: H.entries().filter(e => e.key < INF).map(e => ({ v: e.item, key: e.key })), tied: tied.slice() });
    }
    return { d, pi, order, ties };
  }
  /* ── Dijkstra with an unsorted array: extract-min is a scan; c.add("scan") per entry looked at ── */
  function dijkstraArray(G, s, c) {
    c = c || nop; const n = G.n, adj = spAdj(G); const d = new Array(n).fill(INF), pi = new Array(n).fill(-1), inS = new Array(n).fill(false); d[s] = 0;
    for (let k = 0; k < n; k++) { let u = -1; for (let v = 0; v < n; v++) { c.add("scan"); if (!inS[v] && (u < 0 || d[v] < d[u])) u = v; } if (u < 0 || d[u] === INF) { c.add("extract"); inS[u >= 0 ? u : 0] = true; continue; }
      c.add("extract"); inS[u] = true;
      adj[u].forEach(a => { if (inS[a.v]) { c.add("intoS"); return; } c.add("relax"); if (d[u] + a.w < d[a.v]) { d[a.v] = d[u] + a.w; pi[a.v] = u; c.add("decrease"); } }); }
    return { d, pi };
  }
  /* ── Dijkstra with lazy deletion: a plain heap of (key, vertex) pairs, duplicates pushed,
     stale pairs discarded at pop time.  Counts "push", "pop", "stale", "relax", "sift". ── */
  function dijkstraLazy(G, s, c) {
    c = c || nop; const n = G.n, adj = spAdj(G); const d = new Array(n).fill(INF), pi = new Array(n).fill(-1), done = new Array(n).fill(false); d[s] = 0;
    const H = spHeap(c); let id = 0; H.push("e" + (id++), 0); const who = { e0: s };
    let maxSize = 1;
    while (H.size()) { const e = H.pop(); c.add("pop"); const u = who[e.item], k = e.key; if (done[u] || k > d[u]) { c.add("stale"); continue; } done[u] = true;
      adj[u].forEach(a => { if (done[a.v]) { c.add("intoS"); return; } c.add("relax"); if (d[u] + a.w < d[a.v]) { d[a.v] = d[u] + a.w; pi[a.v] = u; const h = "e" + (id++); who[h] = a.v; H.push(h, d[a.v]); c.add("push"); if (H.size() > maxSize) maxSize = H.size(); } }); }
    return { d, pi, maxSize };
  }

  /* ── Bellman-Ford, in place: rounds of relaxing every arc in edge-list order; early exit on a
     round with no change; a V-th round that still improves = a negative cycle reachable from s.
     Counts "relax" per arc test, "improve" per lowering.  Reports the cycle by walking π back
     n times from a vertex improved in round V and then around the π-cycle.  negInf = vertices
     improved in the extra round plus everything reachable from them (δ = −∞).
     rec(frame) once per round: {round, d, pi, changed, improved:[v…]}. */
  function bellmanFord(G, s, c, rec) {
    c = c || nop; const n = G.n, arcs = spArcs(G); const d = new Array(n).fill(INF), pi = new Array(n).fill(-1); d[s] = 0;
    let rounds = 0, stableAfter = null, cycle = null, cycleW = 0, negInf = null, witness = -1;
    if (rec) rec({ round: 0, d: d.slice(), pi: pi.slice(), changed: 0, improved: [] });
    for (let r = 1; r <= n; r++) { rounds = r; let changed = 0; const improved = []; const improvedSet = new Array(n).fill(false);
      arcs.forEach(a => { c.add("relax"); if (d[a.u] < INF && d[a.u] + a.w < d[a.v]) { d[a.v] = d[a.u] + a.w; pi[a.v] = a.u; changed++; c.add("improve"); if (!improvedSet[a.v]) { improvedSet[a.v] = true; improved.push(a.v); } } });
      if (rec) rec({ round: r, d: d.slice(), pi: pi.slice(), changed, improved: improved.slice(), check: r === n });
      if (!changed) { stableAfter = r - 1; break; }
      if (r === n) { /* the V-th round changed something: negative cycle */
        witness = improved[0]; let y = witness; for (let k = 0; k < n; k++) y = pi[y];
        cycle = [y]; let z = pi[y]; while (z !== y) { cycle.push(z); z = pi[z]; } cycle.reverse();
        cycleW = 0; const adj = spAdj(G); for (let k = 0; k < cycle.length; k++) { const a = cycle[k], b = cycle[(k + 1) % cycle.length]; let best = INF; adj[a].forEach(x => { if (x.v === b && x.w < best) best = x.w; }); cycleW += best; }
        negInf = spReach(G, improved); } }
    return { d, pi, rounds, stableAfter, negative: cycle !== null, cycle, cycleW, negInf, witness };
  }
  /* the two-array (k-indexed) Bellman-Ford of the DP view: dₖ(v) from d_{k−1} only */
  function bellmanFordTwoArray(G, s, c) { c = c || nop; const n = G.n, arcs = spArcs(G); let prev = new Array(n).fill(INF); prev[s] = 0; const rows = [prev.slice()]; for (let k = 1; k <= n - 1; k++) { const cur = prev.slice(); arcs.forEach(a => { c.add("relax"); if (prev[a.u] < INF && prev[a.u] + a.w < cur[a.v]) cur[a.v] = prev[a.u] + a.w; }); rows.push(cur.slice()); prev = cur; } return { d: prev, rows }; }

  /* ── random weighted graphs, reproducible: a random spanning structure from 0 so every
     vertex is reachable, then extra distinct arcs up to m in total; weights in [wlo, whi]. ── */
  function randGraph(n, m, seed, wlo, whi, directed) {
    const r = AL.rng(seed); const edges = []; const have = new Set();
    const add = (u, v) => { if (u === v) return false; const k = directed === false ? Math.min(u, v) + ":" + Math.max(u, v) : u + ":" + v; if (have.has(k)) return false; have.add(k); edges.push({ u, v, w: AL.randInt(r, wlo, whi) }); return true; };
    for (let v = 1; v < n; v++) add(AL.randInt(r, 0, v - 1), v);
    const cap = directed === false ? n * (n - 1) / 2 : n * (n - 1); let guard = 0;
    while (edges.length < Math.min(m, cap) && guard++ < 50 * m + 1000) add(AL.randInt(r, 0, n - 1), AL.randInt(r, 0, n - 1));
    return { n, edges, directed: directed !== false };
  }

  /* ── §06: the three Dijkstras on one graph, each under its own counter; d[] must agree ── */
  function dijkstraCosts(G, s) {
    const ca = AL.counter(), ch = AL.counter(), cl = AL.counter();
    const A = dijkstraArray(G, s, ca), H = dijkstra(G, s, ch), L = dijkstraLazy(G, s, cl);
    const agree = A.d.every((x, v) => x === H.d[v] && x === L.d[v]);
    return { array: ca.all(), heap: ch.all(), lazy: cl.all(), agree, d: H.d, maxLazy: L.maxSize };
  }
  /* sweep V for one density rule m(V); returns one row per V with the counts and the closed forms */
  function dijkstraSweep(sizes, mOf, seed) {
    return sizes.map((n, k) => { const m = Math.min(mOf(n), n * (n - 1)); const G = randGraph(n, m, seed + 17 * k, 1, 20, true); const E = G.edges.length; const R = dijkstraCosts(G, 0);
      return { V: n, E, array: R.array.scan, arrayAll: R.array.scan + R.array.relax, heap: R.heap.extract + R.heap.decrease, heapAll: R.heap.sift + R.heap.relax, lazy: R.lazy.push + R.lazy.pop, lazyAll: R.lazy.sift + R.lazy.relax, maxLazy: R.maxLazy,
               refArray: n * n, refHeap: (n + E) * Math.log2(n), refLazy: Math.max(1, E) * Math.log2(Math.max(2, E)), agree: R.agree }; });
  }
  /* the density (E / V²) at which the array scan first beats the heap, at a fixed V, under two
     accountings: "charged" = every heap operation billed its worst case ⌈log₂V⌉ (the bound's own
     accounting), "measured" = the sift steps the heap actually performed. */
  function dijkstraCrossover(n, seed) {
    const ps = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1]; const rows = []; const lg = Math.ceil(Math.log2(n));
    for (const p of ps) { const m = Math.max(n - 1, Math.round(p * n * n)); const G = randGraph(n, m, seed + Math.round(p * 1000), 1, 20, true); const R = dijkstraCosts(G, 0);
      rows.push({ p, E: G.edges.length, array: R.array.scan + R.array.relax, heapOps: R.heap.extract + R.heap.decrease, decrease: R.heap.decrease, heapCharged: (R.heap.extract + R.heap.decrease) * lg + R.heap.relax, heap: R.heap.sift + R.heap.relax, lazy: R.lazy.sift + R.lazy.relax, agree: R.agree }); }
    rows.forEach(r => { r.arrayBound = n * n + r.E; r.heapBound = (n + r.E) * lg + r.E; });
    const fc = rows.find(r => r.array <= r.heapCharged), fm = rows.find(r => r.array <= r.heap), fb = rows.find(r => r.arrayBound <= r.heapBound);
    return { rows, lg, charged: fc ? fc.p : null, chargedE: fc ? fc.E : null, measured: fm ? fm.p : null, measuredE: fm ? fm.E : null, bound: fb ? fb.p : null, boundE: fb ? fb.E : null, last: rows[rows.length - 1] };
  }

  return { spAdj, spArcs, pathWeight, bruteAllPaths, bruteCycles, spReach, bruteNegInf, dijkstraCosts, dijkstraSweep, dijkstraCrossover,
           relaxSequence, spPathArcs, spOrderWith, spHeap,
           dijkstra, dijkstraArray, dijkstraLazy, bellmanFord, bellmanFordTwoArray, randGraph };
})());

/* the shared instances of chunk A — hand-placed layouts (x, y) so every figure is reproducible */
Object.assign(SI, {
  /* §02 — five vertices s a b c t; the single shortest s ⇝ t path s→b→a→c→t carries every δ */
  gRelax: { n: 5, names: ["s", "a", "b", "c", "t"], directed: true,
    edges: [{ u: 0, v: 1, w: 6 }, { u: 0, v: 2, w: 2 }, { u: 2, v: 1, w: 3 }, { u: 1, v: 3, w: 1 }, { u: 2, v: 3, w: 7 }, { u: 3, v: 4, w: 2 }, { u: 1, v: 4, w: 5, b: -34 }, { u: 2, v: 4, w: 9, b: 34 }],
    pos: [[60, 130], [230, 60], [230, 200], [400, 130], [570, 130]] },
  /* §04 — six vertices; d[3] is lowered twice (12 → 11 → 8), and 4 and 5 tie at key 9 */
  gDij: { n: 6, names: ["s", "1", "2", "3", "4", "5"], directed: true,
    edges: [{ u: 0, v: 1, w: 7 }, { u: 0, v: 2, w: 3 }, { u: 0, v: 3, w: 12 }, { u: 2, v: 1, w: 3 }, { u: 2, v: 3, w: 8 }, { u: 1, v: 3, w: 2 }, { u: 1, v: 4, w: 5 }, { u: 3, v: 4, w: 1 }, { u: 3, v: 5, w: 1 }, { u: 4, v: 5, w: 2 }],
    pos: [[50, 150], [200, 60], [200, 240], [350, 150], [500, 60], [500, 240]] },
  /* §05 — the counterexample: s→a 2, s→b 1, a→b x (x = −3 by default), b→c 1 */
  gNeg: { n: 4, names: ["s", "a", "b", "c"], directed: true,
    edges: [{ u: 0, v: 1, w: 2 }, { u: 0, v: 2, w: 1 }, { u: 1, v: 2, w: -3 }, { u: 2, v: 3, w: 1 }],
    pos: [[70, 150], [260, 60], [260, 240], [470, 150]] },
  /* §07 — five vertices with a negative edge (1→2, −3), no negative cycle; the arc list is
     in the order that makes in-place Bellman-Ford take all V − 1 = 4 rounds */
  gBF: { n: 5, names: ["s", "1", "2", "3", "4"], directed: true,
    edges: [{ u: 4, v: 1, w: 3 }, { u: 3, v: 4, w: -2 }, { u: 2, v: 4, w: 6 }, { u: 2, v: 3, w: 4 }, { u: 1, v: 3, w: 6 }, { u: 1, v: 2, w: -3 }, { u: 0, v: 2, w: 5 }, { u: 0, v: 1, w: 4 }],
    pos: [[50, 150], [200, 60], [200, 240], [380, 60], [380, 240]] },
  /* §07 — the same graph with 4→1 = −6: the cycle 1→2→3→4→1 has weight −7 */
  gBFneg: { n: 5, names: ["s", "1", "2", "3", "4"], directed: true,
    edges: [{ u: 4, v: 1, w: -6 }, { u: 3, v: 4, w: -2 }, { u: 2, v: 4, w: 6 }, { u: 2, v: 3, w: 4 }, { u: 1, v: 3, w: 6 }, { u: 1, v: 2, w: -3 }, { u: 0, v: 2, w: 5 }, { u: 0, v: 1, w: 4 }],
    pos: [[50, 150], [200, 60], [200, 240], [380, 60], [380, 240]] }
});

/* ── chunk B routines ── */
/* ── chunk B routines: DAG shortest / longest paths and critical paths,
   difference constraints, 0-1 BFS and Dial's buckets, all-pairs by
   (min,+) squaring, Floyd-Warshall, Johnson, A* on a grid. Every routine
   is pure and DOM-free, takes an optional counter c (c.add(key)) and an
   optional recorder rec(frame); all names carry the b_ prefix so they never
   collide with the routines the other chunks define. ───────────────────── */
Object.assign(SR, (function () {
  const nop = { add: function () {}, get: function () { return 0; } };
  const INF = Infinity;

  /* adjacency from an SI graph {n, edges:[{u,v,w}], directed}; undirected = both arcs */
  function b_adj(G) {
    const adj = Array.from({ length: G.n }, () => []);
    G.edges.forEach((e, i) => { adj[e.u].push({ u: e.u, v: e.v, w: e.w, id: i }); if (G.directed === false) adj[e.v].push({ u: e.v, v: e.u, w: e.w, id: i }); });
    return adj;
  }
  function b_arcs(G) { const out = []; G.edges.forEach((e, i) => { out.push({ u: e.u, v: e.v, w: e.w, id: i }); if (G.directed === false) out.push({ u: e.v, v: e.u, w: e.w, id: i }); }); return out; }

  /* ── §09 DAG: topological order (Kahn), one relaxation pass, brute force ── */
  function b_topo(G) {
    const n = G.n, adj = b_adj(G), indeg = new Array(n).fill(0);
    adj.forEach(l => l.forEach(e => indeg[e.v]++));
    const q = []; for (let i = 0; i < n; i++) if (!indeg[i]) q.push(i);
    const order = []; while (q.length) { const u = q.shift(); order.push(u); adj[u].forEach(e => { if (--indeg[e.v] === 0) q.push(e.v); }); }
    return { order, ok: order.length === n, adj };
  }
  /* mode "short" | "long"; c.add("relax") per edge; rec({u, v, w, d, changed}) per relaxation, rec({vertex:u, d}) when a vertex is taken */
  function b_dagSP(G, src, mode, c, rec) {
    c = c || nop; const { order, adj, ok } = b_topo(G); const n = G.n; const long = mode === "long";
    const d = new Array(n).fill(long ? -INF : INF), pred = new Array(n).fill(-1); d[src] = 0;
    order.forEach(u => {
      if (rec) rec({ vertex: u, d: d.slice() });
      adj[u].forEach(e => { c.add("relax"); let changed = false;
        if (d[u] !== INF && d[u] !== -INF) { const cand = d[u] + e.w; if (long ? cand > d[e.v] : cand < d[e.v]) { d[e.v] = cand; pred[e.v] = u; changed = true; } }
        if (rec) rec({ u, v: e.v, w: e.w, d: d.slice(), changed }); });
    });
    return { d, pred, order, ok };
  }
  /* every simple path from src (DFS with a visited set — works on graphs with cycles too); returns per-vertex shortest, longest and count, and the number of paths enumerated */
  function b_bruteAllPaths(G, src, limit) {
    const n = G.n, adj = b_adj(G); const sh = new Array(n).fill(INF), lo = new Array(n).fill(-INF), cnt = new Array(n).fill(0);
    const seen = new Array(n).fill(false); let paths = 0; const cap = limit || 2e6;
    (function go(u, len) { if (paths >= cap) return; paths++; cnt[u]++; if (len < sh[u]) sh[u] = len; if (len > lo[u]) lo[u] = len;
      seen[u] = true; adj[u].forEach(e => { if (!seen[e.v]) go(e.v, len + e.w); }); seen[u] = false; })(src, 0);
    return { short: sh, long: lo, count: cnt, paths };
  }
  function b_pathOf(pred, v) { const p = []; while (v !== -1) { p.push(v); v = pred[v]; } return p.reverse(); }

  /* ── §09 critical path (activity-on-node): tasks [{id, dur, pre:[ids]}] ──
     Builds the edge-weighted DAG: source S = n, sink T = n + 1; edge (u, v) of weight dur(u) for
     each precedence u before v; S → v weight 0 for every task without predecessors; v → T weight
     dur(v) for every task without successors. Longest S → T path = the project length. */
  function b_taskGraph(tasks) {
    const n = tasks.length, S = n, T = n + 1, edges = [], hasPre = new Array(n).fill(false), hasSucc = new Array(n).fill(false);
    tasks.forEach((t, v) => t.pre.forEach(u => { edges.push({ u, v, w: tasks[u].dur }); hasPre[v] = true; hasSucc[u] = true; }));
    for (let v = 0; v < n; v++) { if (!hasPre[v]) edges.push({ u: S, v, w: 0 }); if (!hasSucc[v]) edges.push({ u: v, v: T, w: tasks[v].dur }); }
    return { n: n + 2, edges, directed: true, S, T };
  }
  /* earliest start ES (longest path from S), latest start LS (project length − longest path to T), slack, critical tasks; c.add("relax") per edge in each pass */
  function b_criticalPath(tasks, c) {
    c = c || nop; const G = b_taskGraph(tasks); const n = tasks.length;
    const fwd = b_dagSP(G, G.S, "long", c); const L = fwd.d[G.T];
    /* backward: longest path from each vertex to T = longest path from T in the reversed graph */
    const R = { n: G.n, edges: G.edges.map(e => ({ u: e.v, v: e.u, w: e.w })), directed: true };
    const bwd = b_dagSP(R, G.T, "long", c);
    const ES = [], EF = [], LS = [], LF = [], slack = [];
    for (let v = 0; v < n; v++) { ES[v] = fwd.d[v]; EF[v] = ES[v] + tasks[v].dur; LF[v] = L - (bwd.d[v] - tasks[v].dur); LS[v] = LF[v] - tasks[v].dur; slack[v] = LS[v] - ES[v]; }
    const critical = []; for (let v = 0; v < n; v++) if (slack[v] === 0) critical.push(v);
    const path = b_pathOf(fwd.pred, G.T).filter(v => v < n);
    return { G, length: L, ES, EF, LS, LF, slack, critical, path, fwd, bwd };
  }

  /* ── §10 Bellman-Ford with early exit and the negative cycle recovered ──
     rounds of relaxing every arc; c.add("relax") per arc examined; rec({round, d, changed}) after each round.
     Returns {d, pred, rounds, negCycle: null | [vertices]} */
  function b_bellmanFord(G, src, c, rec) {
    c = c || nop; const n = G.n, arcs = b_arcs(G); const d = new Array(n).fill(INF), pred = new Array(n).fill(-1); d[src] = 0;
    let rounds = 0, negCycle = null;
    for (let k = 1; k <= n; k++) {
      let changed = false; rounds = k;
      arcs.forEach(e => { c.add("relax"); if (d[e.u] !== INF && d[e.u] + e.w < d[e.v]) { d[e.v] = d[e.u] + e.w; pred[e.v] = e.u; changed = true; if (k === n && !negCycle) negCycle = e.v; } });
      if (rec) rec({ round: k, d: d.slice(), changed });
      if (!changed) break;
    }
    if (negCycle !== null) { /* walk back n steps to land on the cycle, then collect it */
      let v = negCycle; for (let i = 0; i < n; i++) v = pred[v]; const cyc = [v]; let u = pred[v]; while (u !== v) { cyc.push(u); u = pred[u]; } negCycle = cyc.reverse();
    }
    return { d, pred, rounds, negCycle };
  }
  /* constraints [{j, i, b}] meaning x_j − x_i ≤ b, variables 1..n; vertex 0 is v0 */
  function b_diffconGraph(n, cons) { const edges = []; for (let v = 1; v <= n; v++) edges.push({ u: 0, v, w: 0 }); cons.forEach(k => edges.push({ u: k.i, v: k.j, w: k.b })); return { n: n + 1, edges, directed: true }; }
  function b_diffconCheck(x, cons) { return cons.map(k => ({ j: k.j, i: k.i, b: k.b, lhs: x[k.j] - x[k.i], ok: x[k.j] - x[k.i] <= k.b })); }

  /* ── §11 Dijkstra with a binary heap and lazy deletion (the reference for §11 and §14) ──
     c.add("push"), c.add("pop"), c.add("relax"), c.add("stale") */
  function b_dijkstra(G, src, c) {
    c = c || nop; const n = G.n, adj = b_adj(G); const d = new Array(n).fill(INF), pred = new Array(n).fill(-1), done = new Array(n).fill(false); d[src] = 0;
    const heap = []; const push = (k, v) => { c.add("push"); heap.push([k, v]); let i = heap.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (heap[p][0] <= heap[i][0]) break; [heap[p], heap[i]] = [heap[i], heap[p]]; i = p; } };
    const pop = () => { c.add("pop"); const top = heap[0]; const last = heap.pop(); if (heap.length) { heap[0] = last; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < heap.length && heap[l][0] < heap[m][0]) m = l; if (r < heap.length && heap[r][0] < heap[m][0]) m = r; if (m === i) break; [heap[m], heap[i]] = [heap[i], heap[m]]; i = m; } } return top; };
    push(0, src); const order = [];
    while (heap.length) { const [k, u] = pop(); if (done[u]) { c.add("stale"); continue; } if (k > d[u]) { c.add("stale"); continue; } done[u] = true; order.push(u);
      adj[u].forEach(e => { c.add("relax"); if (d[u] + e.w < d[e.v]) { d[e.v] = d[u] + e.w; pred[e.v] = u; push(d[e.v], e.v); } }); }
    return { d, pred, order };
  }
  /* 0-1 BFS: weights in {0, 1}; the deque holds (vertex, label-at-insertion) pairs; an entry whose label is
     stale (larger than the vertex's current d) is skipped. c.add("front"), c.add("back"), c.add("pop"), c.add("stale"), c.add("relax");
     rec({u, deque:[{v, label}], d, labels}) after each expansion. Also reports the largest deque and the most distinct labels it ever held. */
  function b_zeroOneBFS(G, src, c, rec) {
    c = c || nop; const n = G.n, adj = b_adj(G); const d = new Array(n).fill(INF), pred = new Array(n).fill(-1), done = new Array(n).fill(false); d[src] = 0;
    const dq = [{ v: src, label: 0 }]; let maxLen = 1, distinct = 0, monotone = true; const order = [];
    while (dq.length) { const { v: u, label } = dq.shift(); c.add("pop"); if (label > d[u] || done[u]) { c.add("stale"); continue; } done[u] = true; order.push(u);
      adj[u].forEach(e => { c.add("relax"); if (d[u] + e.w < d[e.v]) { d[e.v] = d[u] + e.w; pred[e.v] = u; if (e.w === 0) { dq.unshift({ v: e.v, label: d[e.v] }); c.add("front"); } else { dq.push({ v: e.v, label: d[e.v] }); c.add("back"); } } });
      const labels = dq.map(x => x.label); const vals = new Set(labels); if (vals.size > distinct) distinct = vals.size; if (dq.length > maxLen) maxLen = dq.length;
      for (let i = 1; i < labels.length; i++) if (labels[i] < labels[i - 1]) monotone = false;
      if (rec) rec({ u, deque: dq.slice(), d: d.slice(), labels }); }
    return { d, pred, order, maxLen, maxDistinct: distinct, monotone };
  }
  /* Dial's algorithm: integer weights 0..C; circular array of C + 1 buckets; c.add("insert"), c.add("scan") per bucket looked at, c.add("pop"), c.add("relax"), c.add("stale") */
  function b_dial(G, src, C, c, rec) {
    c = c || nop; const n = G.n, adj = b_adj(G); const d = new Array(n).fill(INF), pred = new Array(n).fill(-1), done = new Array(n).fill(false); d[src] = 0;
    const B = Array.from({ length: C + 1 }, () => []); B[0].push(src); c.add("insert"); let cur = 0, left = 1; const order = []; let maxDist = 0;
    while (left > 0) {
      let idx = cur % (C + 1); c.add("scan");
      while (!B[idx].length) { cur++; idx = cur % (C + 1); c.add("scan"); }
      const u = B[idx].shift(); left--; c.add("pop"); if (done[u] || d[u] !== cur) { c.add("stale"); continue; } done[u] = true; order.push(u); if (cur > maxDist) maxDist = cur;
      adj[u].forEach(e => { c.add("relax"); if (d[u] + e.w < d[e.v]) { d[e.v] = d[u] + e.w; pred[e.v] = e.u; B[d[e.v] % (C + 1)].push(e.v); left++; c.add("insert"); } });
      if (rec) rec({ u, cur, buckets: B.map(b => b.slice()), d: d.slice() });
    }
    return { d, pred, order, maxDist, lastCur: cur };
  }
  /* grid graphs for §11: rows × cols, 4-neighbour undirected edges with seeded weights in [lo, hi] */
  function b_gridGraph(rows, cols, lo, hi, seed) {
    const r = AL.rng(seed); const edges = []; const id = (i, j) => i * cols + j;
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) { if (j + 1 < cols) edges.push({ u: id(i, j), v: id(i, j + 1), w: AL.randInt(r, lo, hi) }); if (i + 1 < rows) edges.push({ u: id(i, j), v: id(i + 1, j), w: AL.randInt(r, lo, hi) }); }
    return { n: rows * cols, edges, directed: false, rows, cols };
  }

  /* ── §12 (min,+) matrix product and repeated squaring ── */
  function b_weightMatrix(G) { const n = G.n; const W = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 0 : INF)); b_arcs(G).forEach(e => { if (e.w < W[e.u][e.v]) W[e.u][e.v] = e.w; }); return W; }
  function b_minPlus(A, B, c) { c = c || nop; const n = A.length; const C = Array.from({ length: n }, () => new Array(n).fill(INF)); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) { c.add("mult"); const v = A[i][k] + B[k][j]; if (v < C[i][j]) C[i][j] = v; } return C; }
  /* L^(1) = W, then L^(2m) = L^(m) ⊗ L^(m) until m ≥ n − 1; rec({m, L}) per matrix; c.add("mult") per (i,j,k); c.add("product") per product */
  function b_apspSquaring(W, c, rec) {
    c = c || nop; const n = W.length; let L = W.map(r => r.slice()), m = 1; const seq = [{ m: 1, L: L.map(r => r.slice()) }]; if (rec) rec(seq[0]);
    while (m < n - 1) { L = b_minPlus(L, L, c); c.add("product"); m *= 2; const f = { m, L: L.map(r => r.slice()) }; seq.push(f); if (rec) rec(f); }
    return { D: L, seq, products: seq.length - 1 };
  }
  /* the slow version: L^(m) = L^(m−1) ⊗ W for m = 2..n−1 (paths of at most m edges) */
  function b_apspSlow(W, c, rec) { c = c || nop; const n = W.length; let L = W.map(r => r.slice()); const seq = [{ m: 1, L: L.map(r => r.slice()) }]; for (let m = 2; m <= n - 1; m++) { L = b_minPlus(L, W, c); c.add("product"); const f = { m, L: L.map(r => r.slice()) }; seq.push(f); if (rec) rec(f); } return { D: L, seq, products: seq.length - 1 }; }

  /* ── §13 Floyd-Warshall with a "next" matrix, per-k frames, negative-cycle test, transitive closure ── */
  /* c.add("triple") per (k, i, j); rec({k, D, updated:[[i,j],…]}) after each k (and once for k = −1, the initial matrix) */
  function b_floydWarshall(W, c, rec) {
    c = c || nop; const n = W.length; const D = W.map(r => r.slice()); const next = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i !== j && W[i][j] < INF) ? j : -1));
    if (rec) rec({ k: -1, D: D.map(r => r.slice()), updated: [] });
    for (let k = 0; k < n; k++) { const upd = [];
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { c.add("triple"); if (D[i][k] + D[k][j] < D[i][j]) { D[i][j] = D[i][k] + D[k][j]; next[i][j] = next[i][k]; upd.push([i, j]); } }
      if (rec) rec({ k, D: D.map(r => r.slice()), updated: upd }); }
    let negCycle = false; for (let i = 0; i < n; i++) if (D[i][i] < 0) negCycle = true;
    return { D, next, negCycle };
  }
  function b_fwPath(next, i, j) { if (next[i][j] === -1 && i !== j) return null; const p = [i]; while (i !== j) { i = next[i][j]; if (i === -1) return null; p.push(i); } return p; }
  function b_pathWeight(W, p) { let s = 0; for (let i = 0; i + 1 < p.length; i++) s += W[p[i]][p[i + 1]]; return s; }
  /* boolean Floyd-Warshall: T[i][j] = 1 iff a path i ⇝ j (i = j counts); c.add("triple") */
  function b_transitiveClosure(G, c) { c = c || nop; const n = G.n; const T = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j)); b_arcs(G).forEach(e => { T[e.u][e.v] = true; }); for (let k = 0; k < n; k++) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { c.add("triple"); if (T[i][k] && T[k][j]) T[i][j] = true; } return T; }
  /* all pairs by brute force over simple paths (small graphs only) */
  function b_bruteAllPairs(G) { const n = G.n; const D = []; let paths = 0; for (let s = 0; s < n; s++) { const b = b_bruteAllPaths(G, s); D.push(b.short); paths += b.paths; } return { D, paths }; }
  /* all pairs by V runs of Bellman-Ford */
  function b_bfAllPairs(G, c) { const n = G.n; const D = []; for (let s = 0; s < n; s++) D.push(b_bellmanFord(G, s, c).d); return D; }

  /* ── §14 Johnson ── rec per phase; c counts "relax" (Bellman-Ford), "push"/"pop"/"relax2" (the Dijkstra runs) */
  function b_johnson(G, c, rec) {
    c = c || nop; const n = G.n;
    const Gp = { n: n + 1, edges: G.edges.slice().concat(Array.from({ length: n }, (_, v) => ({ u: n, v, w: 0 }))), directed: true };
    if (G.directed === false) Gp.edges = b_arcs(G).map(e => ({ u: e.u, v: e.v, w: e.w })).concat(Array.from({ length: n }, (_, v) => ({ u: n, v, w: 0 })));
    const bf = b_bellmanFord(Gp, n, c); if (bf.negCycle) return { negCycle: bf.negCycle };
    const h = bf.d.slice(0, n); if (rec) rec({ phase: "h", h: h.slice() });
    const arcs = b_arcs(G); const wh = arcs.map(e => ({ u: e.u, v: e.v, w: e.w, wh: e.w + h[e.u] - h[e.v] }));
    const allNonNeg = wh.every(e => e.wh >= 0); if (rec) rec({ phase: "reweight", h: h.slice(), wh: wh.slice() });
    const Gh = { n, edges: wh.map(e => ({ u: e.u, v: e.v, w: e.wh })), directed: true };
    const dc = { add: k => c.add(k === "relax" ? "relax2" : k) };
    const D = [], Dh = [];
    for (let s = 0; s < n; s++) { const r = b_dijkstra(Gh, s, dc); Dh.push(r.d.slice()); D.push(r.d.map((x, v) => x === INF ? INF : x - h[s] + h[v])); if (rec) rec({ phase: "dijkstra", s, h: h.slice(), wh: wh.slice(), dh: r.d.slice(), d: D[s].slice(), order: r.order }); }
    if (rec) rec({ phase: "done", h: h.slice(), wh: wh.slice(), D: D.map(r => r.slice()) });
    return { h, wh, allNonNeg, D, Dh, bfRounds: bf.rounds };
  }

  /* ── §15 A* on a grid ── grid {rows, cols, walls:Set("r,c"), start:[r,c], goal:[r,c]}, 4-neighbour unit moves.
     heuristic: "zero" | "manhattan" | "euclid" | "manhattan2". Ties on f broken by larger g (deeper first), then insertion order.
     c.add("expand") per node popped and expanded, c.add("push"), c.add("relax"); rec({expanded, open, g, f}) per expansion */
  function b_gridDist(a, b, kind) { const dr = Math.abs(a[0] - b[0]), dc = Math.abs(a[1] - b[1]); if (kind === "zero") return 0; if (kind === "euclid") return Math.sqrt(dr * dr + dc * dc); if (kind === "manhattan2") return 2 * (dr + dc); return dr + dc; }
  function b_astar(grid, kind, c, rec) {
    c = c || nop; const R = grid.rows, C = grid.cols; const id = (r, cc) => r * C + cc; const rc = k => [Math.floor(k / C), k % C];
    const s = id(grid.start[0], grid.start[1]), t = id(grid.goal[0], grid.goal[1]); const H = k => b_gridDist(rc(k), grid.goal, kind);
    const g = new Array(R * C).fill(INF), pred = new Array(R * C).fill(-1), closed = new Array(R * C).fill(false); g[s] = 0;
    let open = [{ f: H(s), g: 0, v: s, seq: 0 }]; let seq = 1; const expandedList = []; let reopened = 0;
    const neigh = k => { const [r, cc] = rc(k); const out = []; [[r - 1, cc], [r + 1, cc], [r, cc - 1], [r, cc + 1]].forEach(([a, b]) => { if (a >= 0 && a < R && b >= 0 && b < C && !grid.walls.has(a + "," + b)) out.push(id(a, b)); }); return out; };
    while (open.length) {
      let bi = 0; for (let i = 1; i < open.length; i++) { const a = open[i], b = open[bi]; if (a.f < b.f - 1e-9 || (Math.abs(a.f - b.f) < 1e-9 && (a.g > b.g || (a.g === b.g && a.seq < b.seq)))) bi = i; }
      const cur = open[bi]; open.splice(bi, 1); c.add("pop"); if (cur.g > g[cur.v]) { c.add("stale"); continue; } if (closed[cur.v]) { c.add("stale"); continue; }
      closed[cur.v] = true; c.add("expand"); expandedList.push(cur.v); if (rec) rec({ expanded: cur.v, g: g.slice(), openSet: open.map(o => o.v), closedList: expandedList.slice(), f: cur.f });
      if (cur.v === t) break;
      neigh(cur.v).forEach(v => { c.add("relax"); const ng = g[cur.v] + 1; if (ng < g[v]) { if (closed[v]) { closed[v] = false; reopened++; c.add("reopen"); } g[v] = ng; pred[v] = cur.v; open.push({ f: ng + H(v), g: ng, v, seq: seq++ }); c.add("push"); } });
    }
    const path = g[t] === INF ? null : b_pathOf(pred, t);
    return { dist: g[t], path, expanded: expandedList, g, pred, reopened, s, t, rc };
  }
  function b_gridBFS(grid) { const R = grid.rows, C = grid.cols; const id = (r, cc) => r * C + cc; const d = new Array(R * C).fill(INF); const s = id(grid.start[0], grid.start[1]); d[s] = 0; const q = [s]; let expanded = 0; while (q.length) { const u = q.shift(); expanded++; const r = Math.floor(u / C), cc = u % C; [[r - 1, cc], [r + 1, cc], [r, cc - 1], [r, cc + 1]].forEach(([a, b]) => { if (a >= 0 && a < R && b >= 0 && b < C && !grid.walls.has(a + "," + b)) { const v = id(a, b); if (d[v] === INF) { d[v] = d[u] + 1; q.push(v); } } }); } return { d, dist: d[id(grid.goal[0], grid.goal[1])], expanded }; }
  /* consistency check of a heuristic on the grid: h(u) ≤ w(u,v) + h(v) for every arc */
  function b_heuristicConsistent(grid, kind) { const R = grid.rows, C = grid.cols; let ok = true, admissible = true; const bfs = b_gridBFS({ rows: R, cols: C, walls: grid.walls, start: grid.goal, goal: grid.start }); for (let r = 0; r < R; r++) for (let cc = 0; cc < C; cc++) { if (grid.walls.has(r + "," + cc)) continue; const hu = b_gridDist([r, cc], grid.goal, kind); if (bfs.d[r * C + cc] < INF && hu > bfs.d[r * C + cc] + 1e-9) admissible = false; [[r - 1, cc], [r + 1, cc], [r, cc - 1], [r, cc + 1]].forEach(([a, b]) => { if (a >= 0 && a < R && b >= 0 && b < C && !grid.walls.has(a + "," + b)) { if (hu > 1 + b_gridDist([a, b], grid.goal, kind) + 1e-9) ok = false; } }); } return { consistent: ok, admissible }; }

  return { b_adj, b_arcs, b_topo, b_dagSP, b_bruteAllPaths, b_pathOf, b_taskGraph, b_criticalPath,
           b_bellmanFord, b_diffconGraph, b_diffconCheck,
           b_dijkstra, b_zeroOneBFS, b_dial, b_gridGraph,
           b_weightMatrix, b_minPlus, b_apspSquaring, b_apspSlow,
           b_floydWarshall, b_fwPath, b_pathWeight, b_transitiveClosure, b_bruteAllPairs, b_bfAllPairs,
           b_johnson, b_gridDist, b_astar, b_gridBFS, b_heuristicConsistent };
})());

/* ── chunk B instances ─────────────────────────────────────────────────── */
Object.assign(SI, {
  /* §09: seven tasks with durations and precedences (activity-on-node); the figure builds the edge-weighted DAG with a source and a sink */
  b_tasks: [
    { id: "A", dur: 3, pre: [] }, { id: "B", dur: 2, pre: [] }, { id: "C", dur: 4, pre: [0] }, { id: "D", dur: 5, pre: [0, 1] },
    { id: "E", dur: 2, pre: [2, 3] }, { id: "F", dur: 3, pre: [3] }, { id: "G", dur: 1, pre: [4, 5] }
  ],
  b_taskPos: [[110, 60], [110, 200], [230, 60], [230, 200], [350, 60], [350, 200], [470, 130], [20, 130], [560, 130]],
  /* §10: five unknowns, eight constraints x_j − x_i ≤ b; the flipped one (index 7) makes the system infeasible */
  b_diffcon: [
    { j: 1, i: 5, b: 5 }, { j: 5, i: 4, b: 2 }, { j: 4, i: 3, b: -2 }, { j: 3, i: 5, b: 1 },
    { j: 3, i: 2, b: -1 }, { j: 5, i: 2, b: 3 }, { j: 1, i: 3, b: 4 }, { j: 2, i: 1, b: -2 }
  ],
  b_diffconFlip: { j: 3, i: 5, b: -3 },
  b_diffconFlipIndex: 3,
  b_diffconPos: [[40, 150], [200, 60], [370, 60], [480, 150], [200, 240], [370, 240]],
  /* §11: the two hand-traced instances — six vertices with 0/1 weights (a back entry goes stale), and six vertices with weights 1..3 for Dial's buckets */
  b_g01small: { n: 6, edges: [{ u: 0, v: 1, w: 1 }, { u: 0, v: 2, w: 0 }, { u: 2, v: 3, w: 1 }, { u: 2, v: 4, w: 0 }, { u: 4, v: 3, w: 0 }, { u: 1, v: 5, w: 1 }, { u: 3, v: 5, w: 1 }, { u: 4, v: 5, w: 1 }], directed: true, pos: [[40, 130], [170, 40], [170, 220], [300, 130], [300, 230], [430, 130]] },
  b_gDialSmall: { n: 6, edges: [{ u: 0, v: 1, w: 2 }, { u: 0, v: 2, w: 3 }, { u: 1, v: 2, w: 1 }, { u: 1, v: 3, w: 3 }, { u: 2, v: 3, w: 1 }, { u: 2, v: 4, w: 2 }, { u: 3, v: 5, w: 2 }, { u: 4, v: 5, w: 3 }], directed: true, pos: [[40, 130], [170, 50], [170, 210], [300, 90], [300, 220], [430, 150]] },
  /* §12: four vertices, five arcs, one negative */
  b_g4: { n: 4, edges: [{ u: 0, v: 1, w: 5 }, { u: 0, v: 3, w: 9 }, { u: 1, v: 2, w: -2 }, { u: 2, v: 3, w: 4 }, { u: 3, v: 0, w: -1 }], directed: true, pos: [[60, 60], [200, 60], [200, 200], [60, 200]] },
  /* §13–§14: five vertices, eight arcs, two negative, no negative cycle */
  b_g5: { n: 5, edges: [{ u: 0, v: 1, w: 6 }, { u: 0, v: 3, w: 2 }, { u: 1, v: 2, w: -3 }, { u: 2, v: 4, w: 2 }, { u: 3, v: 1, w: -1 }, { u: 3, v: 4, w: 7 }, { u: 4, v: 0, w: 3 }, { u: 2, v: 3, w: 5 }], directed: true, pos: [[40, 120], [170, 40], [300, 120], [170, 200], [300, 220]] },
  /* §15: two 9 × 14 grids, 4-neighbour unit moves, start (4,1), goal (4,12). The "scattered" grid is the worked one:
     weighted A* (2 × Manhattan) returns a 25-step path where the shortest is 21. On the "two walls" grid it happens to stay optimal. */
  b_astarGrid: { rows: 9, cols: 14, start: [4, 1], goal: [4, 12], walls: new Set(["0,6", "0,7", "0,8", "0,11", "1,5", "2,0", "2,2", "2,3", "2,4", "2,5", "2,6", "2,7", "2,13", "3,5", "3,9", "4,4", "4,7", "4,11", "5,2", "5,3", "5,8", "5,9", "6,2", "7,1", "7,12"]) },
  b_astarGrid2: { rows: 9, cols: 14, start: [4, 1], goal: [4, 12], walls: new Set(["1,6", "2,6", "3,6", "4,6", "5,6", "6,6", "2,9", "3,9", "4,9", "5,9", "6,9", "7,9", "7,3", "6,3", "5,3", "1,10", "1,11", "1,12"]) }
});

/* ── chunk C routines ── */
/* ═══ chunk C — §16–§20: minimum spanning trees (cut/cycle, Kruskal, union-find, Prim, Borůvka & variants) ═══ */
Object.assign(SR, (function () {
  const nop = { add: () => {} };
  const INF = Infinity;

  /* ── graph helpers (undirected view of any SI graph: each edge is usable both ways) ── */
  function mstAdj(G) { const adj = Array.from({ length: G.n }, () => []); G.edges.forEach((e, i) => { adj[e.u].push({ to: e.v, w: e.w, i }); adj[e.v].push({ to: e.u, w: e.w, i }); }); return adj; }
  function mstEdgeName(G, i) { const e = G.edges[i]; const nm = G.names || null; return nm ? nm[e.u] + nm[e.v] : e.u + "–" + e.v; }
  function mstWeightOf(G, idx) { return idx.reduce((s, i) => s + G.edges[i].w, 0); }
  /* is the edge-index set a forest? (union-find without heuristics — the count is not measured here) */
  function mstAcyclic(G, idx) { const p = Array.from({ length: G.n }, (_, i) => i); const f = x => { while (p[x] !== x) x = p[x]; return x; }; for (const i of idx) { const a = f(G.edges[i].u), b = f(G.edges[i].v); if (a === b) return false; p[a] = b; } return true; }
  /* the unique path between a and b inside a tree given as edge indices → list of edge indices (empty if a = b, null if disconnected) */
  function mstTreePath(G, treeIdx, a, b) { const adj = Array.from({ length: G.n }, () => []); treeIdx.forEach(i => { const e = G.edges[i]; adj[e.u].push({ to: e.v, i }); adj[e.v].push({ to: e.u, i }); }); const seen = new Array(G.n).fill(false); let out = null; (function dfs(x, acc) { if (out) return; if (x === b) { out = acc; return; } seen[x] = true; for (const nb of adj[x]) if (!seen[nb.to]) dfs(nb.to, acc.concat([nb.i])); })(a, []); return out; }
  /* distances from root along a tree (edge indices) → {dist[], parentEdge[]} */
  function mstTreeDist(G, treeIdx, root) { const adj = Array.from({ length: G.n }, () => []); treeIdx.forEach(i => { const e = G.edges[i]; adj[e.u].push({ to: e.v, w: e.w, i }); adj[e.v].push({ to: e.u, w: e.w, i }); }); const dist = new Array(G.n).fill(INF), pe = new Array(G.n).fill(-1); dist[root] = 0; const q = [root]; while (q.length) { const u = q.shift(); adj[u].forEach(nb => { if (dist[nb.to] === INF) { dist[nb.to] = dist[u] + nb.w; pe[nb.to] = nb.i; q.push(nb.to); } }); } return { dist, parentEdge: pe }; }

  /* ── brute force: every spanning tree = every (n−1)-subset of edges that is acyclic ── */
  function bruteSpanningTrees(G) {
    const m = G.edges.length, k = G.n - 1, trees = [];
    let subsets = 0;
    (function go(start, acc) { if (acc.length === k) { subsets++; if (mstAcyclic(G, acc)) trees.push(acc.slice()); return; } for (let i = start; i < m; i++) { acc.push(i); go(i + 1, acc); acc.pop(); } })(0, []);
    const weights = trees.map(t => mstWeightOf(G, t));
    const minW = weights.length ? Math.min(...weights) : INF;
    const msts = trees.filter((t, i) => weights[i] === minW);
    const distinctW = [...new Set(weights)].sort((a, b) => a - b);
    const secondW = distinctW.length > 1 ? distinctW[1] : null;                          /* the second-smallest weight that any tree has */
    /* second-best MST in the strict sense: min weight over trees other than a fixed MST (if the MST is unique this is secondW; with ties it equals minW) */
    const secondBestW = trees.length > 1 ? Math.min(...trees.filter(t => t !== msts[0]).map(t => mstWeightOf(G, t))) : null;
    const secondBest = trees.filter(t => t !== msts[0] && mstWeightOf(G, t) === secondBestW);
    /* bottleneck: the heaviest edge of each tree */
    const bottlenecks = trees.map(t => Math.max(...t.map(i => G.edges[i].w)));
    const minBottleneck = bottlenecks.length ? Math.min(...bottlenecks) : INF;
    const mbsts = trees.filter((t, i) => bottlenecks[i] === minBottleneck);
    return { trees, weights, subsets, minW, msts, unique: msts.length === 1, secondW, secondBestW, secondBest, bottlenecks, minBottleneck, mbsts };
  }

  /* ── the cut property, checked: crossing edges of S, the light edge, and the exchange on a tree that avoids it ── */
  function mstCut(G, S, brute) {
    brute = brute || bruteSpanningTrees(G);
    const inS = v => S.indexOf(v) >= 0;
    const cross = G.edges.map((e, i) => i).filter(i => inS(G.edges[i].u) !== inS(G.edges[i].v));
    if (!cross.length) return { cross, light: -1 };
    let light = cross[0]; cross.forEach(i => { if (G.edges[i].w < G.edges[light].w) light = i; });
    const mstsWithLight = brute.msts.filter(t => t.indexOf(light) >= 0);
    /* the exchange: take the lightest spanning tree that AVOIDS the light edge; add it; on the cycle pick the HEAVIEST other crossing edge; swap */
    const avoiding = brute.trees.filter(t => t.indexOf(light) < 0).sort((a, b) => mstWeightOf(G, a) - mstWeightOf(G, b));
    let ex = null;
    if (avoiding.length) {
      const T = avoiding[0]; const e = G.edges[light];
      const path = mstTreePath(G, T, e.u, e.v);
      const crossingOnCycle = path.filter(i => cross.indexOf(i) >= 0);
      let ePrime = crossingOnCycle[0]; crossingOnCycle.forEach(i => { if (G.edges[i].w > G.edges[ePrime].w) ePrime = i; });
      const T2 = T.filter(i => i !== ePrime).concat([light]);
      ex = { T, path, crossingOnCycle, ePrime, T2, w1: mstWeightOf(G, T), w2: mstWeightOf(G, T2), isTree: T2.length === G.n - 1 && mstAcyclic(G, T2) };
    }
    return { cross, light, mstsWithLight, inSomeMst: mstsWithLight.length > 0, exchange: ex, brute };
  }
  /* the cycle property, checked on every cycle closed by a non-MST edge: the heaviest edge of the cycle is not in the MST (distinct weights) or some MST omits it */
  function mstCycleCheck(G, brute) {
    brute = brute || bruteSpanningTrees(G); const T = brute.msts[0]; const out = [];
    G.edges.forEach((e, i) => { if (T.indexOf(i) >= 0) return; const path = mstTreePath(G, T, e.u, e.v); const cyc = path.concat([i]); let heavy = cyc[0]; cyc.forEach(j => { if (G.edges[j].w > G.edges[heavy].w) heavy = j; }); out.push({ closing: i, cycle: cyc, heaviest: heavy, heaviestInMst: T.indexOf(heavy) >= 0, someMstOmits: brute.msts.some(t => t.indexOf(heavy) < 0) }); });
    return out;
  }

  /* ── union-find, instrumented. uf = {p[], rank[], n}; opts = {rank: bool, compress: "none" | "full" | "halving" | "splitting"}.
        c.add("find") per find, c.add("union") per union call, c.add("link") per link that actually merges, c.add("step") per parent-pointer hop. ── */
  function ufMake(n) { return { n, p: Array.from({ length: n }, (_, i) => i), rank: new Array(n).fill(0) }; }
  function ufFind(uf, x, c, opts) {
    c = c || nop; opts = opts || {}; const mode = opts.compress || "none"; c.add("find");
    if (mode === "full") { /* two passes: climb, then re-point everything on the path at the root */
      let r = x; while (uf.p[r] !== r) { c.add("step"); r = uf.p[r]; }
      let y = x; while (uf.p[y] !== r && y !== r) { const nx = uf.p[y]; uf.p[y] = r; y = nx; }
      return r;
    }
    if (mode === "halving") { while (uf.p[x] !== x) { c.add("step"); uf.p[x] = uf.p[uf.p[x]]; x = uf.p[x]; } return x; }
    if (mode === "splitting") { while (uf.p[x] !== x) { c.add("step"); const nx = uf.p[x]; uf.p[x] = uf.p[nx]; x = nx; } return x; }
    while (uf.p[x] !== x) { c.add("step"); x = uf.p[x]; } return x;
  }
  function ufLink(uf, rx, ry, c, opts) {
    c = c || nop; opts = opts || {}; if (rx === ry) return false; c.add("link");
    if (opts.rank) { if (uf.rank[rx] > uf.rank[ry]) uf.p[ry] = rx; else { uf.p[rx] = ry; if (uf.rank[rx] === uf.rank[ry]) uf.rank[ry]++; } }
    else uf.p[rx] = ry;                                                                   /* naive: x's root under y's root, whatever their sizes */
    return true;
  }
  function ufUnion(uf, x, y, c, opts) { c = c || nop; c.add("union"); const rx = ufFind(uf, x, c, opts), ry = ufFind(uf, y, c, opts); return ufLink(uf, rx, ry, c, opts); }
  function ufDepth(uf, x) { let d = 0; while (uf.p[x] !== x) { x = uf.p[x]; d++; } return d; }
  function ufHeight(uf) { let h = 0; for (let x = 0; x < uf.n; x++) h = Math.max(h, ufDepth(uf, x)); return h; }
  /* run an op sequence [{op:"u", a, b} | {op:"f", a}] and record a frame after each op: {op, steps (this op), p (copy), height} */
  function ufRun(n, ops, opts, c) {
    c = c || AL.counter(); const uf = ufMake(n); const frames = [{ op: null, steps: 0, p: uf.p.slice(), rank: uf.rank.slice(), height: 0, result: null }];
    ops.forEach(o => { const before = c.get("step"); let res = null; if (o.op === "u") res = ufUnion(uf, o.a, o.b, c, opts); else res = ufFind(uf, o.a, c, opts); frames.push({ op: o, steps: c.get("step") - before, p: uf.p.slice(), rank: uf.rank.slice(), height: ufHeight(uf), result: res }); });
    return { uf, frames, counter: c, steps: c.get("step"), finds: c.get("find"), unions: c.get("union"), links: c.get("link"), height: ufHeight(uf) };
  }
  const UF_COMBOS = [{ key: "naive", label: "no heuristic", opts: { rank: false, compress: "none" } }, { key: "rank", label: "union by rank only", opts: { rank: true, compress: "none" } }, { key: "pc", label: "path compression only", opts: { rank: false, compress: "full" } }, { key: "both", label: "both", opts: { rank: true, compress: "full" } }];
  /* the same sequence under all four heuristic combinations */
  function ufCompare(n, ops) { return UF_COMBOS.map(cb => { const R = ufRun(n, ops, cb.opts); return { key: cb.key, label: cb.label, steps: R.steps, finds: R.finds, unions: R.unions, height: R.height, maxFindSteps: Math.max(0, ...R.frames.filter(f => f.op && f.op.op === "f").map(f => f.steps)), perOp: R.steps / ops.length }; }); }
  /* a random sequence: n elements, m ops, each a union of a random pair or a find of a random element (seeded) */
  function ufRandomOps(n, m, seed, pUnion) { const r = AL.rng(seed); const ops = []; for (let i = 0; i < m; i++) { if (r() < (pUnion === undefined ? 0.5 : pUnion)) ops.push({ op: "u", a: AL.randInt(r, 0, n - 1), b: AL.randInt(r, 0, n - 1) }); else ops.push({ op: "f", a: AL.randInt(r, 0, n - 1) }); } return ops; }
  /* the adversary for naive linking: a chain of n − 1 unions, then finds of the deepest element */
  function ufChainOps(n, finds) { const ops = []; for (let i = 0; i + 1 < n; i++) ops.push({ op: "u", a: i, b: i + 1 }); for (let i = 0; i < finds; i++) ops.push({ op: "f", a: 0 }); return ops; }
  /* Ackermann-style A_k(j) of the analysis (exact for small arguments; returns Infinity past 2^53), α(n), log* n */
  function ackA(k, j) { if (k === 0) return j + 1; if (k === 1) return 2 * j + 1; if (k === 2) { const v = Math.pow(2, j + 1) * (j + 1) - 1; return isFinite(v) && v <= 9e15 ? v : INF; } let x = j; for (let i = 0; i <= j; i++) { x = ackA(k - 1, x); if (!isFinite(x) || x > 9e15) return INF; } return x; }
  function ackAlpha(n) { for (let k = 0; k < 6; k++) { const a = ackA(k, 1); if (a >= n) return k; } return 5; }
  function logStar(n) { let k = 0; while (n > 1) { n = Math.log2(n); k++; } return k; }

  /* ── Kruskal. Sort by a merge sort under the counter (c.add("cmp")), ties broken by input index → stable; then union-find with both heuristics.
        rec(frame) per edge examined: {i, accepted, reason, forest (indices so far), comps (representative per vertex)} ── */
  function kruskalSort(G, c) { c = c || nop; const idx = G.edges.map((_, i) => i); const less = (a, b) => { c.add("cmp"); return G.edges[a].w < G.edges[b].w || (G.edges[a].w === G.edges[b].w && a < b); }; const ms = arr => { if (arr.length < 2) return arr; const h = arr.length >> 1; const L = ms(arr.slice(0, h)), R = ms(arr.slice(h)); const out = []; let i = 0, j = 0; while (i < L.length && j < R.length) out.push(less(R[j], L[i]) ? R[j++] : L[i++]); return out.concat(L.slice(i), R.slice(j)); }; return ms(idx); }
  function kruskal(G, c, rec, ufOpts) {
    c = c || nop; ufOpts = ufOpts || { rank: true, compress: "full" };
    const order = kruskalSort(G, c); const uf = ufMake(G.n); const tree = []; let w = 0; const frames = [];
    order.forEach(i => { const e = G.edges[i]; const ru = ufFind(uf, e.u, c, ufOpts), rv = ufFind(uf, e.v, c, ufOpts); const ok = ru !== rv; if (ok) { ufLink(uf, ru, rv, c, ufOpts); c.add("union"); tree.push(i); w += e.w; }
      const rep = Array.from({ length: G.n }, (_, v) => { let x = v; while (uf.p[x] !== x) x = uf.p[x]; return x; });
      const f = { i, accepted: ok, reason: ok ? "endpoints in different trees" : "endpoints already connected — closes a cycle", forest: tree.slice(), rep, weight: w }; frames.push(f); if (rec) rec(f); });
    return { tree, weight: w, order, frames, complete: tree.length === G.n - 1 };
  }

  /* ── Prim with an indexed binary min-heap (decrease-key by position). key[v] = lightest edge from v into the tree.
        c.add("extract") per extract-min, c.add("decrease") per key improvement (from ∞ counted separately as "decreaseFromInf"), c.add("scan") per adjacency entry looked at, c.add("heapswap") per sift swap.
        rec(frame) per extraction: {v, edge (index into the tree or −1 for the root), key (copy), pi (copy), inTree (copy), tree (indices so far)} ── */
  function prim(G, root, c, rec) {
    c = c || nop; const n = G.n, adj = mstAdj(G);
    const key = new Array(n).fill(INF), pi = new Array(n).fill(-1), piEdge = new Array(n).fill(-1), inTree = new Array(n).fill(false);
    /* indexed heap */
    const heap = [], pos = new Array(n).fill(-1);
    const swap = (a, b) => { c.add("heapswap"); const t = heap[a]; heap[a] = heap[b]; heap[b] = t; pos[heap[a]] = a; pos[heap[b]] = b; };
    const up = i => { while (i > 0) { const p = (i - 1) >> 1; if (key[heap[i]] < key[heap[p]]) { swap(i, p); i = p; } else break; } };
    const down = i => { for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < heap.length && key[heap[l]] < key[heap[m]]) m = l; if (r < heap.length && key[heap[r]] < key[heap[m]]) m = r; if (m === i) break; swap(i, m); i = m; } };
    key[root] = 0; for (let v = 0; v < n; v++) { heap.push(v); pos[v] = heap.length - 1; } up(pos[root]);
    const tree = []; let w = 0; const frames = [];
    while (heap.length) {
      c.add("extract"); const u = heap[0]; const last = heap.pop(); pos[u] = -1; if (heap.length) { heap[0] = last; pos[last] = 0; down(0); }
      if (key[u] === INF) break;                                                            /* disconnected remainder */
      inTree[u] = true; if (piEdge[u] >= 0) { tree.push(piEdge[u]); w += key[u]; }
      adj[u].forEach(nb => { c.add("scan"); const v = nb.to; if (!inTree[v] && nb.w < key[v]) { if (key[v] === INF) c.add("decreaseFromInf"); else c.add("decreaseReal"); c.add("decrease"); key[v] = nb.w; pi[v] = u; piEdge[v] = nb.i; up(pos[v]); } });
      const f = { v: u, edge: piEdge[u], key: key.slice(), pi: pi.slice(), inTree: inTree.slice(), tree: tree.slice(), weight: w }; frames.push(f); if (rec) rec(f);
    }
    return { tree, weight: w, frames, key, pi, complete: tree.length === n - 1 };
  }
  /* Prim with the array scan instead of a heap: Θ(V²) — c.add("scan") per candidate examined in the min-search */
  function primArray(G, root, c) { c = c || nop; const n = G.n, adj = mstAdj(G); const key = new Array(n).fill(INF), piEdge = new Array(n).fill(-1), inTree = new Array(n).fill(false); key[root] = 0; const tree = []; let w = 0; for (let it = 0; it < n; it++) { let u = -1; for (let v = 0; v < n; v++) { if (inTree[v]) continue; c.add("scan"); if (u < 0 || key[v] < key[u]) u = v; } if (u < 0 || key[u] === INF) break; inTree[u] = true; if (piEdge[u] >= 0) { tree.push(piEdge[u]); w += key[u]; } adj[u].forEach(nb => { if (!inTree[nb.to] && nb.w < key[nb.to]) { key[nb.to] = nb.w; piEdge[nb.to] = nb.i; } }); } return { tree, weight: w }; }

  /* ── Borůvka. Each round: every component picks its lightest outgoing edge (ties by edge index, which is a consistent tie-break); all are added; components merge.
        c.add("round"), c.add("scan") per edge examined per round, c.add("union"). rec(frame) per round: {round, chosen (per component rep → edge index), added (indices), comps (rep per vertex, BEFORE the round)} ── */
  function boruvka(G, c, rec) {
    c = c || nop; const n = G.n; const uf = ufMake(n); const opts = { rank: true, compress: "full" }; const tree = []; let w = 0; let comps = n; const frames = [];
    const less = (a, b) => G.edges[a].w < G.edges[b].w || (G.edges[a].w === G.edges[b].w && a < b);
    while (comps > 1) {
      c.add("round"); const before = Array.from({ length: n }, (_, v) => ufFind(uf, v, null, opts));
      const best = {}; let any = false;
      G.edges.forEach((e, i) => { c.add("scan"); const a = before[e.u], b = before[e.v]; if (a === b) return; any = true; if (best[a] === undefined || less(i, best[a])) best[a] = i; if (best[b] === undefined || less(i, best[b])) best[b] = i; });
      if (!any) break;                                                                       /* disconnected: no outgoing edge anywhere */
      const added = []; Object.keys(best).forEach(k => { const i = best[k]; const e = G.edges[i]; c.add("union"); if (ufUnion(uf, e.u, e.v, null, opts)) { added.push(i); tree.push(i); w += e.w; comps--; } });
      const f = { round: frames.length + 1, chosen: Object.assign({}, best), added, comps: before, weight: w, tree: tree.slice() }; frames.push(f); if (rec) rec(f);
    }
    return { tree, weight: w, rounds: frames.length, frames, complete: tree.length === n - 1 };
  }

  /* ── the shortest-path tree from a root (Dijkstra with an array scan — small graphs only; weights ≥ 0) for the MST-vs-SPT contrast ── */
  function c_spt(G, root) { const n = G.n, adj = mstAdj(G); const d = new Array(n).fill(INF), pe = new Array(n).fill(-1), done = new Array(n).fill(false); d[root] = 0; for (let it = 0; it < n; it++) { let u = -1; for (let v = 0; v < n; v++) if (!done[v] && (u < 0 || d[v] < d[u])) u = v; if (u < 0 || d[u] === INF) break; done[u] = true; adj[u].forEach(nb => { if (d[u] + nb.w < d[nb.to]) { d[nb.to] = d[u] + nb.w; pe[nb.to] = nb.i; } }); } const tree = pe.filter(i => i >= 0); return { dist: d, parentEdge: pe, tree, weight: mstWeightOf(G, tree) }; }
  /* MST paths vs shortest paths from a root: per vertex the MST-path length, the true distance, the stretch */
  function mstVsSpt(G, root, mstTree) { const spt = c_spt(G, root); const td = mstTreeDist(G, mstTree, root); const rows = []; for (let v = 0; v < G.n; v++) if (v !== root) rows.push({ v, mstPath: td.dist[v], dist: spt.dist[v], extra: td.dist[v] - spt.dist[v], stretch: spt.dist[v] > 0 ? td.dist[v] / spt.dist[v] : 1 }); let worst = rows[0]; rows.forEach(r => { if (r.extra > worst.extra) worst = r; }); return { spt, rows, worst, mstWeight: mstWeightOf(G, mstTree), sptWeight: spt.weight, sameEdgeSet: spt.tree.length === mstTree.length && spt.tree.every(i => mstTree.indexOf(i) >= 0) }; }

  /* ── second-best MST by one swap: for every non-tree edge e, remove the heaviest tree edge on the path between its endpoints ── */
  function secondBestBySwap(G, mstTree) { const cands = []; G.edges.forEach((e, i) => { if (mstTree.indexOf(i) >= 0) return; const path = mstTreePath(G, mstTree, e.u, e.v); let heavy = path[0]; path.forEach(j => { if (G.edges[j].w > G.edges[heavy].w) heavy = j; }); const T2 = mstTree.filter(j => j !== heavy).concat([i]); cands.push({ add: i, remove: heavy, path, weight: mstWeightOf(G, T2), tree: T2 }); }); cands.sort((a, b) => a.weight - b.weight || a.add - b.add); return { candidates: cands, best: cands[0] || null }; }
  /* single-linkage clustering: cut the k − 1 heaviest MST edges */
  function mstClusters(G, mstTree, k) { const sorted = mstTree.slice().sort((a, b) => G.edges[b].w - G.edges[a].w || a - b); const cut = sorted.slice(0, Math.max(0, Math.min(k - 1, sorted.length))); const keep = mstTree.filter(i => cut.indexOf(i) < 0); const uf = ufMake(G.n); keep.forEach(i => ufUnion(uf, G.edges[i].u, G.edges[i].v)); const label = Array.from({ length: G.n }, (_, v) => ufFind(uf, v)); const groups = {}; label.forEach((r, v) => { (groups[r] = groups[r] || []).push(v); }); return { cut, keep, label, clusters: Object.values(groups) }; }
  /* a random connected weighted graph for the audit (distinct weights when `distinct`) */
  function mstRandomGraph(n, m, seed, distinct) { const r = AL.rng(seed); const edges = []; const seen = new Set(); for (let v = 1; v < n; v++) { const u = AL.randInt(r, 0, v - 1); edges.push({ u, v, w: 0 }); seen.add(u + "," + v); } let guard = 0; while (edges.length < m && guard++ < 1000) { const u = AL.randInt(r, 0, n - 1), v = AL.randInt(r, 0, n - 1); if (u === v) continue; const k = Math.min(u, v) + "," + Math.max(u, v); if (seen.has(k)) continue; seen.add(k); edges.push({ u: Math.min(u, v), v: Math.max(u, v), w: 0 }); } const ws = distinct ? AL.shuffle(Array.from({ length: edges.length }, (_, i) => i + 1), r) : edges.map(() => AL.randInt(r, 1, 6)); edges.forEach((e, i) => { e.w = ws[i]; }); return { n, edges, directed: false }; }

  return { mstAdj, mstEdgeName, mstWeightOf, mstAcyclic, mstTreePath, mstTreeDist, bruteSpanningTrees, mstCut, mstCycleCheck,
           ufMake, ufFind, ufLink, ufUnion, ufDepth, ufHeight, ufRun, ufCompare, ufRandomOps, ufChainOps, UF_COMBOS, ackA, ackAlpha, logStar,
           kruskalSort, kruskal, prim, primArray, boruvka, c_spt, mstVsSpt, secondBestBySwap, mstClusters, mstRandomGraph };
})());

/* the shared MST instance — the SAME six-vertex, nine-edge graph the Greedy Algorithms page states the cut property on (55 spanning trees, one of minimum weight 15) */
Object.assign(SI, {
  gMst: { n: 6, names: ["A", "B", "C", "D", "E", "F"], directed: false,
          edges: [{ u: 0, v: 1, w: 4 }, { u: 0, v: 3, w: 2 }, { u: 1, v: 2, w: 6 }, { u: 1, v: 3, w: 5 }, { u: 1, v: 4, w: 3 }, { u: 2, v: 4, w: 7 }, { u: 2, v: 5, w: 1 }, { u: 3, v: 4, w: 8 }, { u: 4, v: 5, w: 5 }],
          pos: [[40, 42], [170, 22], [300, 42], [40, 200], [170, 222], [300, 200]] },
  /* the union-find figure's preset sequences on n = 12 elements */
  ufN: 12,
  ufSeqs: {
    chain: { label: "chain: union(0,1), union(1,2), …, union(10,11), then find(0) ×2, find(5)", ops: [].concat(Array.from({ length: 11 }, (_, i) => ({ op: "u", a: i, b: i + 1 })), [{ op: "f", a: 0 }, { op: "f", a: 0 }, { op: "f", a: 5 }]) },
    pairs: { label: "pairs: (0,1) (2,3) … (10,11), then (0,2) (4,6) (8,10), (0,4) (0,8), then find(0) ×2, find(2), find(4)", ops: [{ op: "u", a: 0, b: 1 }, { op: "u", a: 2, b: 3 }, { op: "u", a: 4, b: 5 }, { op: "u", a: 6, b: 7 }, { op: "u", a: 8, b: 9 }, { op: "u", a: 10, b: 11 }, { op: "u", a: 0, b: 2 }, { op: "u", a: 4, b: 6 }, { op: "u", a: 8, b: 10 }, { op: "u", a: 0, b: 4 }, { op: "u", a: 0, b: 8 }, { op: "f", a: 0 }, { op: "f", a: 0 }, { op: "f", a: 2 }, { op: "f", a: 4 }] }
  }
});

/* ── chunk D routines ── */
/* ── chunk D routines: flow networks, residual graphs and cuts (§21), Ford-Fulkerson
   (§22), Edmonds-Karp and Dinic (§23), push-relabel (§24), bipartite matching with
   König and Hall (§25), the reductions — disjoint paths, project selection, image
   segmentation (§26). Flow graphs are {n, edges:[{u,v,cap}], pos, s, t}. ────────── */
Object.assign(SR, (function () {
  const nop = { add: function () {}, get: function () { return 0; } };
  const INF = Infinity;

  /* ── §21 the residual representation ──────────────────────────────────
     Every input edge i becomes a forward arc 2i (cap = c, flow = 0) and a reverse
     arc 2i+1 (cap = 0). The residual capacity of an arc is cap − flow and the two
     arcs of a pair carry opposite flows (flow[2i+1] = −flow[2i]), so pushing on the
     reverse arc is exactly the cancellation c_f(v,u) = f(u,v). Antiparallel input
     edges need no special treatment here: each has its own pair. */
  function flowBuild(G) {
    const n = G.n, adj = Array.from({ length: n }, () => []), arcs = [];
    G.edges.forEach((e, i) => {
      arcs.push({ id: 2 * i, from: e.u, to: e.v, cap: e.cap, flow: 0, edge: i, fwd: true });
      arcs.push({ id: 2 * i + 1, from: e.v, to: e.u, cap: 0, flow: 0, edge: i, fwd: false });
      adj[e.u].push(2 * i); adj[e.v].push(2 * i + 1);
    });
    return { n, s: G.s, t: G.t, arcs, adj, edges: G.edges };
  }
  const res = a => a.cap - a.flow;
  function flowPush(F, arcId, amt) { F.arcs[arcId].flow += amt; F.arcs[arcId ^ 1].flow -= amt; }
  /* flow value per input edge from a residual structure */
  function flowOfEdges(F) { return F.edges.map((e, i) => F.arcs[2 * i].flow); }
  /* residual arcs {u, v, c, fwd, edge} with c > 0, from an input graph and a per-edge flow vector */
  function flowResidual(G, f) {
    const out = [];
    G.edges.forEach((e, i) => { const fi = f[i] || 0; if (e.cap - fi > 0) out.push({ u: e.u, v: e.v, c: e.cap - fi, fwd: true, edge: i }); if (fi > 0) out.push({ u: e.v, v: e.u, c: fi, fwd: false, edge: i }); });
    return out;
  }
  /* check a per-edge flow vector against the two constraints; returns {value, capOk, consOk, excess[], inflow[], outflow[]} */
  function flowCheck(G, f) {
    const n = G.n, inflow = new Array(n).fill(0), outflow = new Array(n).fill(0); let capOk = true;
    G.edges.forEach((e, i) => { const fi = f[i] || 0; if (fi < 0 || fi > e.cap) capOk = false; outflow[e.u] += fi; inflow[e.v] += fi; });
    const excess = inflow.map((x, v) => x - outflow[v]); let consOk = true;
    for (let v = 0; v < n; v++) if (v !== G.s && v !== G.t && excess[v] !== 0) consOk = false;
    return { value: outflow[G.s] - inflow[G.s], capOk, consOk, excess, inflow, outflow };
  }
  /* a cut given as a set S (array of booleans, S[s] = true, S[t] = false): capacity c(S,T) and net flow f(S,T) */
  function flowCut(G, f, S) {
    let cap = 0, net = 0;
    G.edges.forEach((e, i) => { const fi = f ? (f[i] || 0) : 0; if (S[e.u] && !S[e.v]) { cap += e.cap; net += fi; } else if (!S[e.u] && S[e.v]) net -= fi; });
    return { cap, net };
  }
  /* brute force over all 2^(V−2) s-t cuts; c.add("cut") per cut examined. Returns the minimum capacity, one minimising S, the number of minimisers and the count. */
  function bruteMinCut(G, c) {
    c = c || nop; const n = G.n, others = []; for (let v = 0; v < n; v++) if (v !== G.s && v !== G.t) others.push(v);
    let best = INF, bestS = null, ties = 0, count = 0;
    for (let mask = 0; mask < (1 << others.length); mask++) {
      const S = new Array(n).fill(false); S[G.s] = true; others.forEach((v, k) => { if (mask >> k & 1) S[v] = true; });
      c.add("cut"); count++; const cap = flowCut(G, null, S).cap;
      if (cap < best) { best = cap; bestS = S; ties = 1; } else if (cap === best) ties++;
    }
    return { cap: best, S: bestS, ties, count };
  }
  /* the source side of the residual graph: vertices reachable from s in G_f (the min cut FF certifies) */
  function flowReachable(F) { const seen = new Array(F.n).fill(false); seen[F.s] = true; const q = [F.s]; while (q.length) { const u = q.shift(); F.adj[u].forEach(a => { const A = F.arcs[a]; if (res(A) > 0 && !seen[A.to]) { seen[A.to] = true; q.push(A.to); } }); } return seen; }

  /* ── §22 Ford-Fulkerson with a selectable path rule ───────────────────
     rule "dfs": depth-first in adjacency order; "bfs": shortest path (Edmonds-Karp);
     "worst": prefer any s→t path through the designated edge G.mid (default: the
     edge of smallest capacity) — the choice that makes the 4-vertex instance take
     2·M augmentations. c.add("aug") per augmentation, c.add("scan") per arc examined. */
  function ffFindDFS(F, c) {
    const par = new Array(F.n).fill(-1), seen = new Array(F.n).fill(false); seen[F.s] = true;
    let found = false;
    (function go(u) { if (u === F.t) { found = true; return; } for (const a of F.adj[u]) { if (found) return; c.add("scan"); const A = F.arcs[a]; if (res(A) > 0 && !seen[A.to]) { seen[A.to] = true; par[A.to] = a; go(A.to); } } })(F.s);
    return found ? ffTrace(F, par) : null;
  }
  function ffFindBFS(F, c, avoid, src, dst) {
    src = src === undefined ? F.s : src; dst = dst === undefined ? F.t : dst;
    const par = new Array(F.n).fill(-1), dist = new Array(F.n).fill(-1); dist[src] = 0; const q = [src]; let visits = 0;
    while (q.length) { const u = q.shift(); visits++; c.add("visit"); for (const a of F.adj[u]) { c.add("scan"); const A = F.arcs[a]; if (res(A) > 0 && dist[A.to] < 0 && !(avoid && avoid[A.to])) { dist[A.to] = dist[u] + 1; par[A.to] = a; q.push(A.to); } } }
    if (dist[dst] < 0) return null; const p = ffTrace(F, par, src, dst); p.dist = dist; p.visits = visits; return p;
  }
  function ffTrace(F, par, src, dst) {
    src = src === undefined ? F.s : src; dst = dst === undefined ? F.t : dst;
    const path = []; let v = dst; while (v !== src) { const a = par[v]; path.push(a); v = F.arcs[a].from; } path.reverse();
    let b = INF; path.forEach(a => { b = Math.min(b, res(F.arcs[a])); }); return { arcs: path, bottleneck: b };
  }
  function ffFindWorst(F, c, mid) {
    /* try s ⇝ mid.from, the mid arc, mid.to ⇝ t (vertex-disjoint), in either direction; else DFS */
    for (const a of [2 * mid, 2 * mid + 1]) { const A = F.arcs[a]; if (res(A) <= 0) continue;
      const avoid = new Array(F.n).fill(false); avoid[A.to] = true;
      const p1 = A.from === F.s ? { arcs: [], bottleneck: INF } : ffFindBFS(F, c, avoid, F.s, A.from); if (!p1) continue;
      const av2 = new Array(F.n).fill(false); av2[F.s] = true; p1.arcs.forEach(x => { av2[F.arcs[x].from] = true; av2[F.arcs[x].to] = true; });
      const p2 = A.to === F.t ? { arcs: [], bottleneck: INF } : ffFindBFS(F, c, av2, A.to, F.t); if (!p2) continue;
      const arcs = p1.arcs.concat([a], p2.arcs); let b = INF; arcs.forEach(x => { b = Math.min(b, res(F.arcs[x])); }); return { arcs, bottleneck: b };
    }
    return ffFindDFS(F, c);
  }
  function ffMaxflow(G, rule, c, rec, limit) {
    c = c || nop; rule = rule || "dfs"; limit = limit || 100000; const F = flowBuild(G); let value = 0, augs = 0;
    let mid = G.mid; if (mid === undefined) { mid = 0; G.edges.forEach((e, i) => { if (e.cap < G.edges[mid].cap) mid = i; }); }
    while (augs < limit) {
      const p = rule === "bfs" ? ffFindBFS(F, c) : rule === "worst" ? ffFindWorst(F, c, mid) : ffFindDFS(F, c);
      if (!p) break;
      p.arcs.forEach(a => flowPush(F, a, p.bottleneck)); value += p.bottleneck; augs++; c.add("aug");
      if (rec) rec({ k: augs, path: p.arcs.map(a => F.arcs[a].from).concat([F.t]), arcs: p.arcs.slice(), bottleneck: p.bottleneck, value, f: flowOfEdges(F), dist: p.dist || null });
    }
    return { value, f: flowOfEdges(F), augs, S: flowReachable(F), F };
  }

  /* ── §23 Edmonds-Karp, recording BFS layers and critical edges ─────────
     c.add("aug"), c.add("bfs") per BFS, c.add("visit") per vertex dequeued, c.add("scan") per arc.
     rec gets {k, path, arcs, bottleneck, value, dist, critical:[arc ids], f}. critCount[edge index][dir] counts critical events. */
  function ekMaxflow(G, c, rec) {
    c = c || nop; const F = flowBuild(G); let value = 0, augs = 0; const critCount = G.edges.map(() => [0, 0]); let maxDist = 0; const lens = [];
    for (;;) {
      c.add("bfs"); const p = ffFindBFS(F, c); if (!p) break;
      const critical = p.arcs.filter(a => res(F.arcs[a]) === p.bottleneck);
      critical.forEach(a => { critCount[F.arcs[a].edge][F.arcs[a].fwd ? 0 : 1]++; });
      p.arcs.forEach(a => flowPush(F, a, p.bottleneck)); value += p.bottleneck; augs++; c.add("aug"); lens.push(p.arcs.length); maxDist = Math.max(maxDist, p.arcs.length);
      if (rec) rec({ k: augs, path: p.arcs.map(a => F.arcs[a].from).concat([F.t]), arcs: p.arcs.slice(), bottleneck: p.bottleneck, value, dist: p.dist.slice(), critical, f: flowOfEdges(F) });
    }
    return { value, f: flowOfEdges(F), augs, critCount, lens, S: flowReachable(F), F };
  }
  /* Dinic: BFS level graph, then blocking flow by DFS with a current-arc pointer. c.add("phase"), c.add("aug") per path found, c.add("scan"). */
  function dinicMaxflow(G, c) {
    c = c || nop; const F = flowBuild(G); let value = 0, phases = 0, augs = 0;
    for (;;) {
      const level = new Array(F.n).fill(-1); level[F.s] = 0; const q = [F.s];
      while (q.length) { const u = q.shift(); F.adj[u].forEach(a => { c.add("scan"); const A = F.arcs[a]; if (res(A) > 0 && level[A.to] < 0) { level[A.to] = level[u] + 1; q.push(A.to); } }); }
      if (level[F.t] < 0) break; phases++; c.add("phase");
      const it = new Array(F.n).fill(0);
      const dfs = (u, lim) => { if (u === F.t) return lim; for (; it[u] < F.adj[u].length; it[u]++) { c.add("scan"); const A = F.arcs[F.adj[u][it[u]]]; if (res(A) > 0 && level[A.to] === level[u] + 1) { const got = dfs(A.to, Math.min(lim, res(A))); if (got > 0) { flowPush(F, A.id, got); return got; } } } return 0; };
      let got; while ((got = dfs(F.s, INF)) > 0) { value += got; augs++; c.add("aug"); }
    }
    return { value, f: flowOfEdges(F), phases, augs, S: flowReachable(F) };
  }

  /* a seeded random flow network on n vertices with m edges: s = 0, t = n − 1, every vertex reachable, capacities 1..9, positions on a 4-column layout */
  function flowRandom(seed, n, m, rng) {
    const r = rng(seed); const edges = [], set = new Set();
    const put = (u, v, cap) => { if (u === v || set.has(u + "," + v) || v === 0 || u === n - 1) return false; set.add(u + "," + v); edges.push({ u, v, cap }); return true; };
    for (let v = 1; v < n; v++) put(Math.floor(r() * v), v, 1 + Math.floor(r() * 9));
    let guard = 0; while (edges.length < m && guard++ < 1000) { const u = Math.floor(r() * (n - 1)), v = 1 + Math.floor(r() * (n - 1)); if (r() < 0.75 ? u < v : true) put(u, v, 1 + Math.floor(r() * 9)); }
    const cols = 4, pos = []; for (let v = 0; v < n; v++) { const col = v === 0 ? 0 : v === n - 1 ? cols - 1 : 1 + ((v - 1) % (cols - 2)); const rowsIn = v === 0 || v === n - 1 ? 1 : Math.ceil((n - 2) / (cols - 2)); const row = v === 0 || v === n - 1 ? 0 : Math.floor((v - 1) / (cols - 2)); pos.push([40 + col * 520 / (cols - 1), rowsIn === 1 ? 130 : 30 + row * 200 / Math.max(1, rowsIn - 1)]); }
    return { n, edges, s: 0, t: n - 1, pos, names: Array.from({ length: n }, (_, v) => v === 0 ? "s" : v === n - 1 ? "t" : String(v)) };
  }

  /* ── §24 push-relabel (generic, active vertices discharged FIFO) ──────
     c.add("pushSat"), c.add("pushNon"), c.add("relabel"). rec per operation: {op, u, v, amt, h, e, f}. */
  function prMaxflow(G, c, rec) {
    c = c || nop; const F = flowBuild(G), n = F.n; const h = new Array(n).fill(0), e = new Array(n).fill(0); h[F.s] = n;
    const active = [], inQ = new Array(n).fill(false);
    const activate = v => { if (v !== F.s && v !== F.t && !inQ[v] && e[v] > 0) { inQ[v] = true; active.push(v); } };
    const snap = (op, u, v, amt) => { if (rec) rec({ op, u, v, amt, h: h.slice(), e: e.slice(), f: flowOfEdges(F) }); };
    snap("init", -1, -1, 0);
    F.adj[F.s].forEach(a => { const A = F.arcs[a]; if (A.cap > 0) { flowPush(F, a, A.cap); e[A.to] += A.cap; e[F.s] -= A.cap; c.add("pushSat"); snap("push·sat", F.s, A.to, A.cap); activate(A.to); } });
    let ops = 0;
    while (active.length) {
      const u = active.shift(); inQ[u] = false;
      while (e[u] > 0) {                      /* discharge u */
        let pushed = false;
        for (const a of F.adj[u]) { const A = F.arcs[a]; if (res(A) > 0 && h[u] === h[A.to] + 1) {
          const amt = Math.min(e[u], res(A)); const sat = amt === res(A); flowPush(F, a, amt); e[u] -= amt; e[A.to] += amt; ops++;
          c.add(sat ? "pushSat" : "pushNon"); snap(sat ? "push·sat" : "push·non", u, A.to, amt); activate(A.to); pushed = true; if (e[u] === 0) break; } }
        if (!pushed) { let m = INF; F.adj[u].forEach(a => { const A = F.arcs[a]; if (res(A) > 0) m = Math.min(m, h[A.to]); }); h[u] = m + 1; ops++; c.add("relabel"); snap("relabel", u, -1, h[u]); }
        if (ops > 200000) break;
      }
    }
    return { value: e[F.t], f: flowOfEdges(F), h, e, S: flowReachable(F), F };
  }

  /* ── §25 bipartite matching ──────────────────────────────────────────
     B = {nL, nR, adj:[[r,…] per l]}. Reduction to flow: s → every l (cap 1), l → r (cap 1), r → t (cap 1). */
  function bipToFlow(B) { const s = B.nL + B.nR, t = s + 1, edges = []; for (let l = 0; l < B.nL; l++) edges.push({ u: s, v: l, cap: 1 }); for (let l = 0; l < B.nL; l++) B.adj[l].forEach(r => edges.push({ u: l, v: B.nL + r, cap: 1 })); for (let r = 0; r < B.nR; r++) edges.push({ u: B.nL + r, v: t, cap: 1 }); return { n: t + 1, edges, s, t }; }
  /* augmenting-path matching directly on the bipartite graph (the flow specialised): c.add("aug"), c.add("scan"). rec per augmentation {k, path:[l,r,l,r,…], matchL, matchR}. */
  function bipartiteMatch(B, c, rec) {
    c = c || nop; const matchL = new Array(B.nL).fill(-1), matchR = new Array(B.nR).fill(-1); let size = 0, k = 0;
    for (let l0 = 0; l0 < B.nL; l0++) {
      const seen = new Array(B.nR).fill(false), parR = new Array(B.nR).fill(-1);
      const q = [l0], parL = {}; parL[l0] = null; let found = -1;
      while (q.length && found < 0) { const l = q.shift(); for (const r of B.adj[l]) { c.add("scan"); if (seen[r]) continue; seen[r] = true; parR[r] = l; if (matchR[r] < 0) { found = r; break; } parL[matchR[r]] = r; q.push(matchR[r]); } }
      if (found < 0) continue;
      const path = []; let r = found; while (r >= 0) { const l = parR[r]; path.unshift(r); path.unshift(l); const nr = parL[l]; matchL[l] = r; matchR[r] = l; r = nr === null || nr === undefined ? -1 : nr; }
      size++; k++; c.add("aug"); if (rec) rec({ k, path, matchL: matchL.slice(), matchR: matchR.slice(), from: l0 });
    }
    return { size, matchL, matchR };
  }
  /* Hopcroft-Karp: phases of BFS layering + DFS for a maximal set of vertex-disjoint shortest augmenting paths. c.add("phase"), c.add("aug"). */
  function hopcroftKarp(B, c) {
    c = c || nop; const matchL = new Array(B.nL).fill(-1), matchR = new Array(B.nR).fill(-1); let size = 0, phases = 0;
    for (;;) {
      const dist = new Array(B.nL).fill(-1), q = []; for (let l = 0; l < B.nL; l++) if (matchL[l] < 0) { dist[l] = 0; q.push(l); }
      let reach = false; while (q.length) { const l = q.shift(); for (const r of B.adj[l]) { c.add("scan"); const l2 = matchR[r]; if (l2 < 0) reach = true; else if (dist[l2] < 0) { dist[l2] = dist[l] + 1; q.push(l2); } } }
      if (!reach) break; phases++; c.add("phase");
      const dfs = l => { for (const r of B.adj[l]) { c.add("scan"); const l2 = matchR[r]; if (l2 < 0 || (dist[l2] === dist[l] + 1 && dfs(l2))) { matchL[l] = r; matchR[r] = l; return true; } } dist[l] = -1; return false; };
      let got = 0; for (let l = 0; l < B.nL; l++) if (matchL[l] < 0 && dfs(l)) got++;
      size += got; c.add("aug", got);
    }
    return { size, matchL, matchR, phases };
  }
  /* brute force over all matchings (subsets of edges with no shared endpoint): max size and how many matchings were examined. */
  function bruteMatching(B) {
    const E = []; for (let l = 0; l < B.nL; l++) B.adj[l].forEach(r => E.push([l, r])); let best = 0, count = 0, bestM = [];
    const usedL = new Array(B.nL).fill(false), usedR = new Array(B.nR).fill(false);
    (function go(i, chosen) { if (i === E.length) { count++; if (chosen.length > best) { best = chosen.length; bestM = chosen.slice(); } return; } go(i + 1, chosen); const [l, r] = E[i]; if (!usedL[l] && !usedR[r]) { usedL[l] = usedR[r] = true; chosen.push(E[i]); go(i + 1, chosen); chosen.pop(); usedL[l] = usedR[r] = false; } })(0, []);
    return { size: best, matching: bestM, count };
  }
  /* König: Z = vertices reachable from unmatched L by alternating paths; cover = (L ∖ Z) ∪ (R ∩ Z). Returns {coverL, coverR, Z}. */
  function konigCover(B, M) {
    const zL = new Array(B.nL).fill(false), zR = new Array(B.nR).fill(false), q = [];
    for (let l = 0; l < B.nL; l++) if (M.matchL[l] < 0) { zL[l] = true; q.push(l); }
    while (q.length) { const l = q.shift(); for (const r of B.adj[l]) { if (M.matchL[l] === r || zR[r]) continue; zR[r] = true; const l2 = M.matchR[r]; if (l2 >= 0 && !zL[l2]) { zL[l2] = true; q.push(l2); } } }
    const coverL = [], coverR = []; for (let l = 0; l < B.nL; l++) if (!zL[l]) coverL.push(l); for (let r = 0; r < B.nR; r++) if (zR[r]) coverR.push(r);
    return { coverL, coverR, zL, zR, size: coverL.length + coverR.length };
  }
  function isVertexCover(B, inL, inR) { for (let l = 0; l < B.nL; l++) for (const r of B.adj[l]) if (!inL[l] && !inR[r]) return false; return true; }
  /* brute force over all 2^(nL+nR) vertex subsets */
  function bruteVertexCover(B) { const n = B.nL + B.nR; let best = INF, bestSet = null, count = 0; for (let m = 0; m < (1 << n); m++) { count++; const inL = [], inR = []; let sz = 0; for (let i = 0; i < n; i++) { const b = !!(m >> i & 1); if (i < B.nL) inL.push(b); else inR.push(b); if (b) sz++; } if (sz < best && isVertexCover(B, inL, inR)) { best = sz; bestSet = { inL, inR }; } } return { size: best, set: bestSet, count }; }
  /* Hall: the subset S ⊆ L with the largest deficiency |S| − |N(S)|; a violator exists iff deficiency > 0 */
  function hallViolator(B) { let worst = 0, S = null, count = 0; for (let m = 1; m < (1 << B.nL); m++) { count++; const N = new Set(); let sz = 0; for (let l = 0; l < B.nL; l++) if (m >> l & 1) { sz++; B.adj[l].forEach(r => N.add(r)); } if (sz - N.size > worst) { worst = sz - N.size; S = []; for (let l = 0; l < B.nL; l++) if (m >> l & 1) S.push(l); } } return { deficiency: worst, S, count }; }

  /* ── §26 reductions ──────────────────────────────────────────────────── */
  /* edge-disjoint s-t paths: unit capacities on every edge; vertex-disjoint: split each internal vertex v into v_in → v_out (cap 1) */
  function unitCapacity(G) { return { n: G.n, edges: G.edges.map(e => ({ u: e.u, v: e.v, cap: 1 })), s: G.s, t: G.t }; }
  function vertexSplit(G) { const n = G.n; const inn = v => (v === G.s || v === G.t) ? v : v, out = v => (v === G.s || v === G.t) ? v : v + n; const edges = []; for (let v = 0; v < n; v++) if (v !== G.s && v !== G.t) edges.push({ u: v, v: v + n, cap: 1 }); G.edges.forEach(e => edges.push({ u: out(e.u), v: inn(e.v), cap: 1 })); return { n: 2 * n, edges, s: G.s, t: G.t }; }
  /* brute force: max number of edge-disjoint s-t simple paths (tiny graphs only) — enumerate all simple paths, then the largest pairwise edge-disjoint family */
  function bruteDisjointPaths(G, vertexDisjoint) {
    const adj = Array.from({ length: G.n }, () => []); G.edges.forEach((e, i) => adj[e.u].push({ v: e.v, i })); const paths = [];
    (function go(u, seenV, edgesUsed) { if (u === G.t) { paths.push({ edges: edgesUsed.slice(), verts: seenV.slice() }); return; } adj[u].forEach(x => { if (!seenV.includes(x.v)) { seenV.push(x.v); edgesUsed.push(x.i); go(x.v, seenV, edgesUsed); edgesUsed.pop(); seenV.pop(); } }); })(G.s, [G.s], []);
    let best = 0; const m = paths.length;
    const compatible = (a, b) => vertexDisjoint ? a.verts.every(v => v === G.s || v === G.t || !b.verts.includes(v)) : a.edges.every(e => !b.edges.includes(e));
    for (let mask = 0; mask < (1 << m); mask++) { const chosen = []; for (let i = 0; i < m; i++) if (mask >> i & 1) chosen.push(paths[i]); let ok = true; for (let i = 0; i < chosen.length && ok; i++) for (let j = i + 1; j < chosen.length; j++) if (!compatible(chosen[i], chosen[j])) { ok = false; break; } if (ok && chosen.length > best) best = chosen.length; }
    return { best, paths: m };
  }
  /* project selection: profit[i] (negative = cost), prereq [[i, j]] meaning i requires j. Network: s → i (profit) for profit > 0, i → t (−profit) for profit < 0, i → j (∞) for each prerequisite. Optimum = Σ positive profits − min cut; the chosen set is the source side. */
  function projectSelection(profit, prereq, c) {
    const n = profit.length, s = n, t = n + 1, edges = []; let P = 0; const BIG = 1e9;
    profit.forEach((p, i) => { if (p > 0) { edges.push({ u: s, v: i, cap: p }); P += p; } else if (p < 0) edges.push({ u: i, v: t, cap: -p }); });
    prereq.forEach(([i, j]) => edges.push({ u: i, v: j, cap: BIG }));
    const G = { n: n + 2, edges, s, t }; const R = ekMaxflow(G, c); const chosen = []; for (let i = 0; i < n; i++) if (R.S[i]) chosen.push(i);
    return { optimum: P - R.value, minCut: R.value, positiveSum: P, chosen, G };
  }
  function bruteProjects(profit, prereq) { const n = profit.length; let best = -INF, bestSet = null, closures = 0; for (let m = 0; m < (1 << n); m++) { let ok = true; prereq.forEach(([i, j]) => { if ((m >> i & 1) && !(m >> j & 1)) ok = false; }); if (!ok) continue; closures++; let v = 0; for (let i = 0; i < n; i++) if (m >> i & 1) v += profit[i]; if (v > best) { best = v; bestSet = []; for (let i = 0; i < n; i++) if (m >> i & 1) bestSet.push(i); } } return { best, set: bestSet, closures }; }
  /* image segmentation on an R×C grid: I = intensities, muF/muB the two region means, λ the smoothness weight.
     D_p(1) = |I_p − muF|, D_p(0) = |I_p − muB|;  s → p with cap D_p(0)  (cut ⇒ p on the sink side ⇒ label 0),
     p → t with cap D_p(1)  (cut ⇒ label 1);  p ↔ q with cap λ for 4-neighbours. Label 1 = source side. */
  function segBuild(I, muF, muB, lambda) {
    const R = I.length, C = I[0].length, N = R * C, s = N, t = N + 1, edges = [], D0 = [], D1 = [];
    for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) { const p = i * C + j; D1[p] = Math.abs(I[i][j] - muF); D0[p] = Math.abs(I[i][j] - muB); if (D0[p] > 0) edges.push({ u: s, v: p, cap: D0[p] }); if (D1[p] > 0) edges.push({ u: p, v: t, cap: D1[p] }); }
    for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) { const p = i * C + j; if (lambda > 0) { if (j + 1 < C) { edges.push({ u: p, v: p + 1, cap: lambda }); edges.push({ u: p + 1, v: p, cap: lambda }); } if (i + 1 < R) { edges.push({ u: p, v: p + C, cap: lambda }); edges.push({ u: p + C, v: p, cap: lambda }); } } }
    return { n: N + 2, edges, s, t, R, C, D0, D1, lambda };
  }
  function segEnergy(G, label) { let E = 0; const { R, C } = G; for (let p = 0; p < R * C; p++) E += label[p] ? G.D1[p] : G.D0[p]; for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) { const p = i * C + j; if (j + 1 < C && label[p] !== label[p + 1]) E += G.lambda; if (i + 1 < R && label[p] !== label[p + C]) E += G.lambda; } return E; }
  function segMinCut(G, c) { const Rk = ekMaxflow(G, c); const label = []; for (let p = 0; p < G.R * G.C; p++) label.push(Rk.S[p] ? 1 : 0); return { value: Rk.value, label, energy: segEnergy(G, label), augs: Rk.augs }; }
  function segBrute(G) { const N = G.R * G.C; let best = INF, bestLabel = null, ties = 0; for (let m = 0; m < (1 << N); m++) { const lab = []; for (let p = 0; p < N; p++) lab.push(m >> p & 1); const E = segEnergy(G, lab); if (E < best) { best = E; bestLabel = lab; ties = 1; } else if (E === best) ties++; } return { energy: best, label: bestLabel, ties, count: 1 << N }; }

  return { flowBuild, flowResidual, flowCheck, flowCut, bruteMinCut, flowReachable, flowOfEdges,
           ffMaxflow, ekMaxflow, dinicMaxflow, flowRandom, prMaxflow,
           bipToFlow, bipartiteMatch, hopcroftKarp, bruteMatching, konigCover, isVertexCover, bruteVertexCover, hallViolator,
           unitCapacity, vertexSplit, bruteDisjointPaths, projectSelection, bruteProjects, segBuild, segEnergy, segMinCut, segBrute };
})());
Object.assign(SI, {
  /* the six-vertex flow network: s = 0, v1..v4 = 1..4, t = 5; maximum flow 23 (recomputed in every figure) */
  gFlow: { n: 6, s: 0, t: 5, names: ["s", "v₁", "v₂", "v₃", "v₄", "t"],
    edges: [{ u: 0, v: 1, cap: 16 }, { u: 0, v: 2, cap: 13 }, { u: 2, v: 1, cap: 4 }, { u: 1, v: 3, cap: 12 }, { u: 3, v: 2, cap: 9 }, { u: 2, v: 4, cap: 14 }, { u: 4, v: 3, cap: 7 }, { u: 3, v: 5, cap: 20 }, { u: 4, v: 5, cap: 4 }],
    pos: [[40, 130], [190, 40], [190, 220], [400, 40], [400, 220], [560, 130]] },
  /* the four-vertex pathology: s = 0, u = 1, v = 2, t = 3, M = 100; the middle edge u → v has capacity 1 */
  gFlowBad: { n: 4, s: 0, t: 3, names: ["s", "u", "v", "t"], mid: 2,
    edges: [{ u: 0, v: 1, cap: 100 }, { u: 0, v: 2, cap: 100 }, { u: 1, v: 2, cap: 1 }, { u: 1, v: 3, cap: 100 }, { u: 2, v: 3, cap: 100 }],
    pos: [[60, 130], [300, 30], [300, 230], [540, 130]] },
  /* a five-vertex network for the push-relabel hand-work: s = 0, a = 1, b = 2, c = 3, t = 4 */
  gFlowPR: { n: 5, s: 0, t: 4, names: ["s", "a", "b", "c", "t"],
    edges: [{ u: 0, v: 1, cap: 5 }, { u: 0, v: 2, cap: 3 }, { u: 1, v: 2, cap: 2 }, { u: 1, v: 3, cap: 2 }, { u: 2, v: 4, cap: 4 }, { u: 3, v: 4, cap: 3 }],
    pos: [[60, 130], [230, 40], [230, 220], [420, 40], [580, 130]] },
  /* the 4 × 4 bipartite instance (perfect matching exists) and its Hall-violating variant */
  gBip: { nL: 4, nR: 4, adj: [[0, 1], [0], [1, 2, 3], [2]] },
  gBipHall: { nL: 4, nR: 4, adj: [[0, 1], [0], [1], [2]] },
  /* project selection: profits (negative = a cost) and prerequisites [i requires j] */
  projProfit: [9, 6, -4, -5], projPrereq: [[0, 2], [1, 2], [1, 3]],
  /* the 3 × 3 pixel grid for segmentation: intensities on 0..10, region means 8 (foreground) and 2 (background) */
  segI: [[7, 8, 3], [8, 4, 7], [2, 3, 1]], segMuF: 8, segMuB: 2
});

/* ── chunk E routines ── */
/* ── chunk E: no routines of its own — the load-time audit (SA, in the figures part) calls the
   routines of chunks A–D exactly as the figures do, and its instance generators live inside the
   audit IIFE so nothing is added to the top-level scope. ─────────────────────────────────── */

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
  inf: v => (v === Infinity || v === -Infinity) ? "∞" : v,
  /* a 2-D table of cells: cells[i][j] → text; fill(i,j) → colour or null; returns {cx(j), cy(i), w, h} */
  grid: function (g, rows, cols, opt) {
    const o = Object.assign({ x: 0, y: 0, w: 30, h: 22, gap: 2, text: () => "", fill: () => null, rowLabel: null, colLabel: null, fontSize: 11, stroke: AC.line }, opt || {});
    const gg = g.append("g").attr("transform", `translate(${o.x},${o.y})`);
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const f = o.fill(i, j);
      gg.append("rect").attr("x", j * (o.w + o.gap)).attr("y", i * (o.h + o.gap)).attr("width", o.w).attr("height", o.h).attr("rx", 3)
        .attr("fill", f || AC.panel2).attr("stroke", o.stroke);
      const t = o.text(i, j);
      if (t !== "" && t !== null && t !== undefined) gg.append("text").attr("x", j * (o.w + o.gap) + o.w / 2).attr("y", i * (o.h + o.gap) + o.h / 2 + 4)
        .attr("text-anchor", "middle").attr("font-size", o.fontSize).attr("fill", f ? AC.bg : AC.ink).text(t);
    }
    if (o.rowLabel) for (let i = 0; i < rows; i++) gg.append("text").attr("x", -6).attr("y", i * (o.h + o.gap) + o.h / 2 + 4).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.muted).text(o.rowLabel(i));
    if (o.colLabel) for (let j = 0; j < cols; j++) gg.append("text").attr("x", j * (o.w + o.gap) + o.w / 2).attr("y", -5).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.muted).text(o.colLabel(j));
    return { g: gg, cx: j => j * (o.w + o.gap) + o.w / 2, cy: i => i * (o.h + o.gap) + o.h / 2, w: o.w, h: o.h, step: o.w + o.gap, stepY: o.h + o.gap };
  }
};

/* ── chunk A figures ── */
/* ── chunk A: a shared graph renderer, attached once to SX (no top-level const) ──
   SX.spDraw(g, G, opt): draws the arcs of a shared instance {n, edges, pos, names, directed}
   as gently curved paths with arrowheads and weight labels, then the vertices.
     opt.arc(e, i)   → {color, width, dash}  per edge (null = default)
     opt.vertex(v)   → {fill, stroke, ink}   per vertex
     opt.label(v)    → text drawn under (or above) the vertex, e.g. its d[]
     opt.labelColor(v), opt.bend (px, default 16), opt.r (radius, default 16)
   Returns {pos} so a caller can hang extra marks off the layout. */
if (!SX.spDraw) SX.spDraw = function (g, G, opt) {
  const o = Object.assign({ arc: () => null, vertex: () => null, label: () => "", labelColor: () => AC.good, bend: 16, r: 16, fontSize: 11 }, opt || {});
  const pos = G.pos.map(p => ({ x: p[0], y: p[1] }));
  G.edges.forEach((e, i) => {
    const a = pos[e.u], b = pos[e.v]; const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1; const nx = -dy / len, ny = dx / len;
    const bend = e.b !== undefined ? e.b : o.bend; const mx = (a.x + b.x) / 2 + nx * bend, my = (a.y + b.y) / 2 + ny * bend;
    const st = o.arc(e, i) || {}; const color = st.color || AC.line, width = st.width || 1.6;
    /* trim the curve so the head stops at the target circle */
    const t = 1 - (o.r + 2) / len; const ex = (1 - t) * (1 - t) * a.x + 2 * (1 - t) * t * mx + t * t * b.x, ey = (1 - t) * (1 - t) * a.y + 2 * (1 - t) * t * my + t * t * b.y;
    const p = g.append("path").attr("d", `M${a.x},${a.y} Q${mx},${my} ${ex},${ey}`).attr("fill", "none").attr("stroke", color).attr("stroke-width", width);
    if (st.dash) p.attr("stroke-dasharray", st.dash);
    if (G.directed !== false) { const ang = Math.atan2(ey - my, ex - mx); const h = 7;
      g.append("path").attr("d", `M${ex},${ey} L${ex - h * Math.cos(ang - 0.45)},${ey - h * Math.sin(ang - 0.45)} L${ex - h * Math.cos(ang + 0.45)},${ey - h * Math.sin(ang + 0.45)} Z`).attr("fill", color === AC.line ? AC.muted : color); }
    const lx = 0.25 * a.x + 0.5 * mx + 0.25 * b.x + nx * 9, ly = 0.25 * a.y + 0.5 * my + 0.25 * b.y + ny * 9;
    g.append("text").attr("x", lx).attr("y", ly + 4).attr("text-anchor", "middle").attr("font-size", o.fontSize).attr("fill", st.color && st.color !== AC.line ? st.color : AC.muted).text(st.text !== undefined ? st.text : e.w);
  });
  for (let v = 0; v < G.n; v++) { const p = pos[v]; const st = o.vertex(v) || {};
    g.append("circle").attr("cx", p.x).attr("cy", p.y).attr("r", o.r).attr("fill", st.fill || AC.panel2).attr("stroke", st.stroke || AC.line).attr("stroke-width", st.stroke ? 2 : 1);
    g.append("text").attr("x", p.x).attr("y", p.y + 4).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", st.ink || AC.ink).text(G.names ? G.names[v] : v);
    const lab = o.label(v); if (lab !== "" && lab !== null && lab !== undefined) { const below = G.labelBelow ? G.labelBelow[v] : (p.y >= 150); g.append("text").attr("x", p.x).attr("y", p.y + (below ? o.r + 14 : -o.r - 6)).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", o.labelColor(v)).text(lab); } }
  return { pos };
};

/* ── 01  #relax-svg  relaxation in four schedules on the five-vertex graph ── */
(function () {
  if (!SX.has("relax-svg")) return;
  const W = 680, H = 300; const G = SI.gRelax, N = G.names, n = G.n;
  const fmt = x => x === Infinity ? "∞" : x;
  function build() {
    const mode = SX.val("relax-order", "list");
    const sp = SR.spPathArcs(G, 0, 4);
    const order = mode === "spfirst" ? SR.spOrderWith(G, sp, false) : mode === "reverse" ? SR.spOrderWith(G, sp.slice().reverse(), false) : mode === "random" ? AL.shuffle(SR.spArcs(G), AL.rng(5)) : SR.spArcs(G);
    const brute = SR.bruteAllPaths(G, 0);
    const c = AL.counter(); const frames = []; const R = SR.relaxSequence(G, 0, order, c, f => frames.push(f));
    const relax = c.get("relax"), improve = c.get("improve");
    const atDelta = R.passes.map(p => p.d.filter((x, v) => x === brute.d[v]).length);
    const agree = R.d.every((x, v) => x === brute.d[v]);
    const svg = d3.select("#relax-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 0, r: 0, t: 6, b: 30 });
    const init = { pass: 0, arc: null, improved: false, d: (function () { const d = new Array(n).fill(Infinity); d[0] = 0; return d; })(), pi: new Array(n).fill(-1) };
    const all = [init].concat(frames);
    function render(f, k) {
      F.g.selectAll("*").remove();
      SX.spDraw(F.g, G, {
        arc: (e, i) => { if (f.arc && f.arc.i === i) return { color: f.improved ? AC.good : AC.a2, width: 3 }; if (f.pi[e.v] === e.u) return { color: AC.accent, width: 2 }; return null; },
        vertex: v => (f.arc && f.arc.v === v) ? { fill: f.improved ? AC.good : AC.a2, ink: AC.bg } : (f.arc && f.arc.u === v) ? { stroke: AC.a2 } : null,
        label: v => "d = " + fmt(f.d[v]), labelColor: v => f.d[v] === brute.d[v] ? AC.good : (f.d[v] === Infinity ? AC.muted : AC.a2) });
      const tight = f.d.filter((x, v) => x === brute.d[v]).length;
      F.g.append("text").attr("x", 8).attr("y", F.ih + 24).attr("font-size", 11).attr("fill", AC.ink)
        .text(f.arc ? `pass ${f.pass} · relax ${N[f.arc.u]}→${N[f.arc.v]} (w = ${f.arc.w}): d[${N[f.arc.u]}] + w = ${fmt(f.d[f.arc.u] === Infinity ? Infinity : f.d[f.arc.u] + f.arc.w)} ${f.improved ? "< old d → lowered to " + f.d[f.arc.v] : "≥ d[" + N[f.arc.v] + "] = " + fmt(f.d[f.arc.v]) + " → no change"} · ${tight} of ${n} vertices at δ · relaxations so far ${k}` : `initialised: d[s] = 0, every other d = ∞ · schedule: ${order.map(a => N[a.u] + "→" + N[a.v]).join(" ")}`);
    }
    AL.stepper(svg, { frames: all, render, delay: 550, label: "relaxation" });
    SX.setHtml("relax-readout", `schedule: ${order.map(a => N[a.u] + "→" + N[a.v]).join(", ")} · passes until a pass changes nothing: <b>${R.passes.length}</b> · measured relaxations <b>${relax}</b> = ${R.passes.length} × ${order.length} ${SX.flag(relax === R.passes.length * order.length)}, of which <b>${improve}</b> lowered an estimate · vertices at δ after each pass: ⟨${atDelta.join(", ")}⟩ · final d = ⟨${R.d.map(fmt).join(", ")}⟩ vs brute force over all ${brute.paths - 1} simple paths from s (${brute.paths} with the empty path): ⟨${brute.d.map(fmt).join(", ")}⟩ ${SX.flag(agree)} · the π-path to t: ${(function () { const p = []; let v = 4; while (v >= 0) { p.push(N[v]); v = R.pi[v]; } return p.reverse().join("→"); })()} of weight ${SR.pathWeight(G, (function () { const p = []; let v = 4; while (v >= 0) { p.push(v); v = R.pi[v]; } return p.reverse(); })())} ${SX.flag(SR.pathWeight(G, (function () { const p = []; let v = 4; while (v >= 0) { p.push(v); v = R.pi[v]; } return p.reverse(); })()) === brute.d[4])}`);
  }
  SX.on("relax-order", "change", build);
  build();
})();

/* ── 02  #dijkstra-svg  Dijkstra stepped per extraction, with the d[] / π[] table ── */
(function () {
  if (!SX.has("dijkstra-svg")) return;
  const W = 680, H = 300;
  function randInst() { const G = SR.randGraph(7, 13, 23, 1, 9, true); G.names = ["s", "1", "2", "3", "4", "5", "6"]; G.pos = [[50, 150], [170, 60], [170, 240], [300, 150], [420, 60], [420, 240], [520, 150]]; return G; }
  const fmt = x => x === Infinity ? "∞" : x;
  function build() {
    const G = SX.val("dij-inst", "six") === "six" ? SI.gDij : randInst(); const n = G.n, N = G.names;
    const c = AL.counter(); const frames = []; const R = SR.dijkstra(G, 0, c, f => frames.push(f));
    const brute = SR.bruteAllPaths(G, 0);
    const agree = R.d.every((x, v) => x === brute.d[v]);
    const outdeg = G.edges.length;
    const piOk = R.d.every((x, v) => { if (x === Infinity) return R.pi[v] < 0; const p = []; let y = v; while (y >= 0) { p.push(y); y = R.pi[y]; } return SR.pathWeight(G, p.reverse()) === x; });
    const svg = d3.select("#dijkstra-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 0, r: 0, t: 6, b: 30 });
    const init = { u: -1, d: (function () { const d = new Array(n).fill(Infinity); d[0] = 0; return d; })(), pi: new Array(n).fill(-1), inS: new Array(n).fill(false), events: [], frontier: [{ v: 0, key: 0 }], tied: [] };
    const all = [init].concat(frames); const last = all.length - 1;
    function render(f, k) {
      F.g.selectAll("*").remove();
      const ev = i => f.events.find(e => e.arc.i === i);
      SX.spDraw(F.g, G, {
        arc: (e, i) => { const x = ev(i); if (x) return x.kind === "improve" ? { color: AC.good, width: 3 } : x.kind === "no" ? { color: AC.a2, width: 2, dash: "4,3" } : { color: AC.muted, width: 1, dash: "2,3" }; if (f.pi[e.v] === e.u) return { color: AC.accent, width: k === last ? 3 : 2 }; return null; },
        vertex: v => v === f.u ? { fill: AC.a2, ink: AC.bg } : f.inS[v] ? { fill: AC.accent, ink: AC.bg } : null,
        label: v => f.inS[v] || v === f.u ? "δ = " + fmt(f.d[v]) : (f.d[v] === Infinity ? "" : "key " + f.d[v]),
        labelColor: v => f.inS[v] || v === f.u ? AC.good : AC.a2 });
      /* d / π table on the right */
      const tbl = SX.grid(F.g, n, 2, { x: 592, y: 22, w: 34, h: 20, gap: 2, fontSize: 11, rowLabel: i => N[i], colLabel: j => ["d", "π"][j],
        text: (i, j) => j === 0 ? fmt(f.d[i]) : (f.pi[i] < 0 ? "–" : N[f.pi[i]]), fill: (i, j) => i === f.u ? AC.a2 : f.inS[i] ? AC.accent : null });
      F.g.append("text").attr("x", 8).attr("y", F.ih + 24).attr("font-size", 11).attr("fill", AC.ink)
        .text(f.u < 0 ? `initialised: d[s] = 0, others ∞ · queue holds every vertex` : `extract ${N[f.u]} (key ${fmt(f.d[f.u])})${f.tied.length > 1 ? " — tie among " + f.tied.map(v => N[v]).join(", ") + ", smaller id first" : ""} · ${f.events.length ? f.events.map(e => N[e.arc.u] + "→" + N[e.arc.v] + (e.kind === "improve" ? ": " + fmt(e.old) + "→" + e.now : e.kind === "no" ? ": no change" : ": into S, skipped")).join(" · ") : "no out-edges"} · |S| = ${f.inS.filter(x => x).length}`);
    }
    AL.stepper(svg, { frames: all, render, delay: 800, label: "extraction" });
    SX.setHtml("dijkstra-readout", `V = ${n}, E = ${G.edges.length} · extraction order ${R.order.map(v => N[v]).join(" ")} · measured: extractions <b>${c.get("extract")}</b> = V ${SX.flag(c.get("extract") === n)}, edges examined <b>${c.get("relax") + c.get("intoS")}</b> = sum of out-degrees = ${outdeg} ${SX.flag(c.get("relax") + c.get("intoS") === outdeg)} (${c.get("relax")} relaxed${c.get("intoS") ? ", " + c.get("intoS") + " led into S and were skipped" : ""}), decrease-keys <b>${c.get("decrease")}</b> (${R.d.filter((x, v) => v !== 0 && x < Infinity).length} first discoveries + ${c.get("decrease") - R.d.filter((x, v) => v !== 0 && x < Infinity).length} genuine lowerings), heap sift steps <b>${c.get("sift")}</b>, ties at extraction <b>${R.ties}</b> · d = ⟨${R.d.map(fmt).join(", ")}⟩ vs brute force over all ${brute.paths - 1} simple paths from s: ⟨${brute.d.map(fmt).join(", ")}⟩ ${SX.flag(agree)} · every π-path re-sums to its d ${SX.flag(piOk)}`);
  }
  SX.on("dij-inst", "change", build);
  build();
})();

/* ── 03  #dijkstra-neg-svg  the negative-edge counterexample with a weight slider ── */
(function () {
  if (!SX.has("dijkstra-neg-svg")) return;
  const W = 680, H = 300;
  const fmt = x => x === Infinity ? "∞" : x;
  function inst(x) { const G = JSON.parse(JSON.stringify(SI.gNeg)); G.edges[2].w = x; return G; }
  function build() {
    const x = +SX.val("neg-w", -3); const variant = SX.val("neg-variant", "skip"); SX.setText("neg-w-val", (x < 0 ? "−" : "") + Math.abs(x));
    const G = inst(x), n = G.n, N = G.names;
    const c = AL.counter(); const frames = []; const R = SR.dijkstra(G, 0, c, f => frames.push(f), { relaxIntoS: variant === "relax" });
    const brute = SR.bruteAllPaths(G, 0);
    const wrong = R.d.map((v, i) => v !== brute.d[i]); const nWrong = wrong.filter(Boolean).length;
    /* sweep the weight from +2 down to −5 for both variants */
    const sweep = v => { for (let w = 2; w >= -5; w--) { const g = inst(w); const b = SR.bruteAllPaths(g, 0); const r = SR.dijkstra(g, 0, null, null, { relaxIntoS: v === "relax" }); if (!r.d.every((z, i) => z === b.d[i])) return { w, wrong: r.d.map((z, i) => z !== b.d[i] ? N[i] : "").filter(Boolean) }; } return null; };
    const fs = sweep("skip"), fr = sweep("relax");
    /* §05's two-Dijkstra patch for one negative edge (p, q) = (a, b): δ0 without the edge, from s and from b; δ = min(δ0(s, v), δ0(s, a) + w + δ0(b, v)) */
    const patch = (function () { const g0 = inst(x); const e = g0.edges.splice(2, 1)[0]; const A = SR.dijkstra(g0, 0).d, B = SR.dijkstra(g0, e.v).d; return { A, B, d: A.map((z, i) => Math.min(z, A[e.u] + x + B[i])) }; })();
    const patchOk = patch.d.every((z, i) => z === brute.d[i]);
    const svg = d3.select("#dijkstra-neg-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 0, r: 0, t: 6, b: 30 });
    const init = { u: -1, d: (function () { const d = new Array(n).fill(Infinity); d[0] = 0; return d; })(), pi: new Array(n).fill(-1), inS: new Array(n).fill(false), events: [], frontier: [], tied: [] };
    const all = [init].concat(frames); const last = all.length - 1;
    function render(f, k) {
      F.g.selectAll("*").remove();
      const ev = i => f.events.find(e => e.arc.i === i);
      SX.spDraw(F.g, G, {
        arc: (e, i) => { const z = ev(i); if (z) return z.kind === "improve" ? { color: AC.good, width: 3 } : z.kind === "no" ? { color: AC.a2, width: 2, dash: "4,3" } : z.kind === "intoS-changed" ? { color: AC.bad, width: 3 } : { color: AC.bad, width: 2, dash: "3,3" }; if (f.pi[e.v] === e.u) return { color: AC.accent, width: 2 }; if (i === 2) return { color: x < 0 ? AC.rose : AC.line, width: 1.6 }; return null; },
        vertex: v => v === f.u ? { fill: AC.a2, ink: AC.bg } : f.inS[v] ? (k === last && wrong[v] ? { fill: AC.accent, stroke: AC.bad, ink: AC.bg } : { fill: AC.accent, ink: AC.bg }) : null,
        label: v => (f.inS[v] || v === f.u) ? "d = " + fmt(f.d[v]) + (k === last && wrong[v] ? "  ≠ δ = " + fmt(brute.d[v]) : "") : (f.d[v] === Infinity ? "" : "key " + f.d[v]),
        labelColor: v => (k === last && wrong[v]) ? AC.bad : (f.inS[v] || v === f.u) ? AC.good : AC.a2 });
      F.g.append("text").attr("x", 8).attr("y", F.ih + 24).attr("font-size", 11).attr("fill", AC.ink)
        .text(f.u < 0 ? `initialised · w(a→b) = ${x}` : `extract ${N[f.u]} (key ${fmt(f.d[f.u])})${f.tied.length > 1 ? " — tie among " + f.tied.map(v => N[v]).join(", ") : ""} · ${f.events.length ? f.events.map(e => N[e.arc.u] + "→" + N[e.arc.v] + (e.kind === "improve" ? ": " + fmt(e.old) + "→" + e.now : e.kind === "no" ? ": no change" : e.kind === "intoS-changed" ? ": into S, d changed " + fmt(e.old) + "→" + e.now + " but " + N[e.arc.v] + " is never re-extracted" : ": into S, skipped")).join(" · ") : "no out-edges"}${k === last ? (nWrong ? ` · WRONG at ${wrong.map((w, v) => w ? N[v] : "").filter(Boolean).join(", ")}` : " · all correct") : ""}`);
    }
    const st = AL.stepper(svg, { frames: all, render, delay: 800, label: "extraction" }); st.go(last);
    SX.setHtml("dijkstra-neg-readout", `w(a→b) = ${x}, edges into S ${variant === "skip" ? "skipped" : "relaxed but never re-extracted"} · extraction order ${R.order.map(v => N[v]).join(" ")} · Dijkstra d = ⟨${R.d.map(fmt).join(", ")}⟩ vs brute force over all ${brute.paths - 1} simple paths: δ = ⟨${brute.d.map(fmt).join(", ")}⟩ ${nWrong === 0 ? '<span style="color:' + AC.good + '">Dijkstra correct here</span>' : '<span style="color:' + AC.bad + '">Dijkstra WRONG here — the expected failure</span>'}${nWrong ? " — wrong at " + wrong.map((w, v) => w ? N[v] : "").filter(Boolean).join(", ") + " (too HIGH, as the upper-bound property predicts: " + R.d.map((v, i) => wrong[i] ? v + " > " + brute.d[i] : "").filter(Boolean).join(", ") + ")" : ""} · sweep from +2 down to −5: first weight at which Dijkstra disagrees with brute force — skipping variant <b>${fs ? fs.w : "none"}</b>${fs ? " (wrong at " + fs.wrong.join(", ") + ")" : ""}, relax-anyway variant <b>${fr ? fr.w : "none"}</b>${fr ? " (wrong at " + fr.wrong.join(", ") + ")" : ""} · prediction: the route s→a→b beats the direct edge iff 2 + w &lt; 1 iff w ≤ −2, so the first failing integer weight should be −2 in both variants ${SX.flag(!!fs && fs.w === -2 && !!fr && fr.w === -2)} · at −1 the negative route ties the direct edge (2 + (−1) = 1) and the answer survives · two-Dijkstra patch for the one negative edge: δ₀ from s without a→b ⟨${patch.A.map(fmt).join(", ")}⟩, from b ⟨${patch.B.map(fmt).join(", ")}⟩, min(δ₀(s, ·), δ₀(s, a) + w + δ₀(b, ·)) = ⟨${patch.d.map(fmt).join(", ")}⟩ = brute force ${SX.flag(patchOk)}`);
  }
  SX.on("neg-w", "input", build); SX.on("neg-variant", "change", build);
  build();
})();

/* ── 04  #heapcost-svg  three Dijkstras across V at two densities, against the bounds ── */
(function () {
  if (!SX.has("heapcost-svg")) return;
  const W = 680, H = 320; const sizes = [8, 16, 32, 64, 128, 256];
  function build() {
    const fam = SX.val("hc-density", "sparse");
    const rows = fam === "sparse" ? SR.dijkstraSweep(sizes, n => 3 * n, 101) : SR.dijkstraSweep(sizes, n => Math.round(n * n / 4), 202);
    /* cross-check every size against Bellman-Ford (a second algorithm) */
    const bfOk = rows.every((r, k) => { const G = SR.randGraph(r.V, fam === "sparse" ? 3 * r.V : Math.round(r.V * r.V / 4), (fam === "sparse" ? 101 : 202) + 17 * k, 1, 20, true); const bf = SR.bellmanFord(G, 0); const dj = SR.dijkstra(G, 0); return !bf.negative && bf.d.every((x, v) => x === dj.d[v]); });
    const X = SR.dijkstraCrossover(128, 303);
    const svg = d3.select("#heapcost-svg");
    const F = AL.frame(svg, W, H, { l: 60, r: 16, t: 14, b: 40 });
    const lx = v => Math.log2(v), ly = v => Math.log10(Math.max(1, v));
    const allY = rows.flatMap(r => [r.array, r.heap, r.lazy, r.refArray, r.refHeap, r.refLazy]);
    const x = d3.scaleLinear().domain([lx(sizes[0]), lx(sizes[sizes.length - 1])]).range([0, F.iw]);
    const y = d3.scaleLinear().domain([0, Math.ceil(ly(Math.max(...allY)))]).range([F.ih, 0]);
    /* axes drawn by hand: powers of 2 on x, powers of 10 on y */
    sizes.forEach(v => { F.g.append("line").attr("x1", x(lx(v))).attr("x2", x(lx(v))).attr("y1", 0).attr("y2", F.ih).attr("stroke", AC.grid); F.g.append("text").attr("x", x(lx(v))).attr("y", F.ih + 16).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.muted).text(v); });
    for (let p = 0; p <= y.domain()[1]; p++) { F.g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(p)).attr("y2", y(p)).attr("stroke", AC.grid); F.g.append("text").attr("x", -6).attr("y", y(p) + 4).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.muted).text(p === 0 ? "1" : "10" + "⁰¹²³⁴⁵⁶⁷⁸⁹"[p]); }
    F.g.append("text").attr("x", F.iw).attr("y", F.ih + 32).attr("text-anchor", "end").attr("font-size", 11).attr("fill", AC.muted).text("V (log scale)");
    F.g.append("text").attr("x", 0).attr("y", -3).attr("font-size", 11).attr("fill", AC.muted).text("operations charged (log scale)");
    const path = (key, color, dash) => { const d = rows.map((r, i) => (i ? "L" : "M") + x(lx(r.V)) + "," + y(ly(r[key]))).join(" "); const p = F.g.append("path").attr("d", d).attr("fill", "none").attr("stroke", color).attr("stroke-width", dash ? 1.5 : 2.5); if (dash) p.attr("stroke-dasharray", dash); if (!dash) rows.forEach(r => F.g.append("circle").attr("cx", x(lx(r.V))).attr("cy", y(ly(r[key]))).attr("r", 3.5).attr("fill", color)); };
    path("refArray", AC.rose, "5,4"); path("refHeap", AC.accent, "5,4"); path("refLazy", AC.teal, "5,4");
    path("array", AC.rose); path("heap", AC.accent); path("lazy", AC.teal);
    AL.legend(F.g, [{ label: "array: scans (= V²)", color: AC.rose }, { label: "index-map heap: extractions + decrease-keys", color: AC.accent }, { label: "lazy heap: pushes + pops", color: AC.teal }, { label: "bounds V², (V + E) log₂V, E log₂E", color: AC.muted, dash: "5,4" }], 8, 14);
    const L = rows[rows.length - 1]; const f = SX.int;
    SX.setHtml("heapcost-readout", `${fam === "sparse" ? "sparse family, E ≈ 3V" : "dense family, E ≈ V²/4"} · at V = ${L.V}, E = ${f(L.E)}: array scans <b>${f(L.array)}</b> (= V² ${SX.flag(L.array === L.V * L.V)}), index-map heap <b>${f(L.heap)}</b> operations = ${L.V} extractions + ${f(L.heap - L.V)} decrease-keys (${(100 * (L.heap - L.V) / L.E).toFixed(1)}% of E), lazy heap <b>${f(L.lazy)}</b> pushes + pops (at most ${f(L.maxLazy)} pairs held at once) · reference values: (V + E) log₂V = ${f(Math.round(L.refHeap))}, E log₂E = ${f(Math.round(L.refLazy))} · every measured count below its bound at every V ${SX.flag(rows.every(r => r.array <= r.refArray && r.heap <= r.refHeap && r.lazy <= r.refLazy))} · the three d[] agree at every V ${SX.flag(rows.every(r => r.agree))}, and agree with Bellman-Ford ${SX.flag(bfOk)} · crossover at V = 128, sweeping E/V² from 0.02 to 1: on the bound's accounting (every relaxation a decrease-key) the array wins from density <b>${X.bound === null ? "never" : X.bound}</b>${X.bound !== null ? " (E = " + f(X.boundE) + ")" : ""}; charging the heap's measured operations ⌈log₂128⌉ = ${X.lg} levels each: <b>${X.charged === null ? "never, even at E = " + f(X.last.E) : "from " + X.charged}</b>; counting the heap's actual sift steps: <b>${X.measured === null ? "never" : "from " + X.measured}</b> — at full density the heap took ${f(X.last.heap)} elementary steps to the array's ${f(X.last.array)}, with only ${f(X.last.decrease)} decrease-keys in ${f(X.last.E)} edges ${SX.flag(X.rows.every(r => r.agree))}`);
  }
  SX.on("hc-density", "change", build);
  build();
})();

/* ── 05  #bf-svg  Bellman-Ford one round per step, with the edge list and the negative cycle ── */
(function () {
  if (!SX.has("bf-svg")) return;
  const W = 680, H = 300;
  const fmt = x => x === Infinity ? "∞" : x === -Infinity ? "−∞" : x;
  function build() {
    const base = SX.val("bf-inst", "plain") === "neg" ? SI.gBFneg : SI.gBF; const rev = SX.val("bf-order", "list") === "rev";
    const G = { n: base.n, names: base.names, directed: true, pos: base.pos, edges: rev ? base.edges.slice().reverse() : base.edges.slice() };
    const n = G.n, N = G.names, E = G.edges.length;
    const c = AL.counter(); const frames = []; const R = SR.bellmanFord(G, 0, c, f => frames.push(f));
    const brute = SR.bruteAllPaths(G, 0); const cyc = SR.bruteCycles(G, 0); const negInfB = SR.bruteNegInf(G, 0);
    const cycleSet = R.cycle ? new Set(R.cycle.map((v, k) => v + ">" + R.cycle[(k + 1) % R.cycle.length])) : null;
    const svg = d3.select("#bf-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 0, r: 0, t: 6, b: 30 });
    const last = frames.length - 1;
    function render(f, k) {
      F.g.selectAll("*").remove(); const imp = new Set(f.improved); const showCycle = R.cycle && k === last;
      SX.spDraw(F.g, G, {
        arc: (e, i) => { if (showCycle && cycleSet.has(e.u + ">" + e.v)) return { color: AC.bad, width: 3.5 }; if (f.pi[e.v] === e.u) return { color: AC.accent, width: 2 }; if (e.w < 0) return { color: AC.rose, width: 1.6 }; return null; },
        vertex: v => showCycle && R.negInf[v] ? { fill: AC.bad, ink: AC.bg } : imp.has(v) ? { fill: AC.good, ink: AC.bg } : v === 0 ? { fill: AC.accent, ink: AC.bg } : null,
        label: v => showCycle && R.negInf[v] ? "δ = −∞" : "d = " + fmt(f.d[v]),
        labelColor: v => showCycle && R.negInf[v] ? AC.bad : (f.d[v] === brute.d[v] && !R.negative ? AC.good : AC.a2) });
      /* the edge list, in relaxation order */
      const improvedArcs = new Set(); if (k > 0) { /* recompute which arcs improved in this round by replaying */ const prev = frames[k - 1]; const d = prev.d.slice(); G.edges.forEach((e, i) => { if (d[e.u] < Infinity && d[e.u] + e.w < d[e.v]) { d[e.v] = d[e.u] + e.w; improvedArcs.add(i); } }); }
      SX.grid(F.g, E, 1, { x: 560, y: 18, w: 96, h: 22, gap: 2, fontSize: 11, colLabel: () => "edge order", text: i => `${N[G.edges[i].u]}→${N[G.edges[i].v]}  (${G.edges[i].w})`, fill: i => improvedArcs.has(i) ? AC.good : (showCycle && cycleSet.has(G.edges[i].u + ">" + G.edges[i].v) ? AC.bad : null) });
      F.g.append("text").attr("x", 8).attr("y", F.ih + 24).attr("font-size", 11).attr("fill", AC.ink)
        .text(f.round === 0 ? `initialised: d[s] = 0, others ∞ · V − 1 = ${n - 1} rounds, then the test round` : f.check ? (f.changed ? `round ${f.round} (the test): ${f.changed} estimate${f.changed > 1 ? "s" : ""} still fell → NEGATIVE CYCLE ${R.cycle.map(v => N[v]).join("→")}→${N[R.cycle[0]]}, weight ${R.cycleW}` : `round ${f.round} (the test): nothing changed → no negative cycle reachable from s`) : `round ${f.round}: ${f.changed} relaxation${f.changed === 1 ? "" : "s"} improved (${f.improved.map(v => N[v]).join(", ") || "none"})${f.changed === 0 ? " → early exit" : ""} · ${f.d.filter((x, v) => x === brute.d[v]).length} of ${n} vertices at the brute-force δ`);
    }
    const st = AL.stepper(svg, { frames, render, delay: 900, label: "round" }); st.go(last);
    const agree = !R.negative && R.d.every((x, v) => x === brute.d[v]);
    const negInfOk = R.negative && R.negInf.every((b, v) => b === negInfB[v]);
    SX.setHtml("bf-readout", `V = ${n}, E = ${E}, edges relaxed ${rev ? "in reversed list order" : "in list order"} · rounds executed <b>${R.rounds}</b>${R.stableAfter !== null ? " (stable after round " + R.stableAfter + ", early exit in round " + R.rounds + ")" : " (the test round still changed something)"} · measured relaxations <b>${c.get("relax")}</b> = rounds × E = ${R.rounds} × ${E} ${SX.flag(c.get("relax") === R.rounds * E)}, of which <b>${c.get("improve")}</b> lowered an estimate · ` + (R.negative
      ? `NEGATIVE CYCLE reported: ${R.cycle.map(v => N[v]).join("→")}→${N[R.cycle[0]]} of weight <b>${R.cycleW}</b> (found from witness ${N[R.witness]}, whose estimate fell in the test round, by walking π back ${n} times) · brute force over the ${cyc.count} simple cycles reachable from s: lightest weighs ${cyc.minW} ${SX.flag(R.cycleW < 0 && cyc.negative)} · δ = −∞ for {${R.negInf.map((b, v) => b ? N[v] : "").filter(Boolean).join(", ")}} vs brute force (every vertex reachable from a negative cycle): {${negInfB.map((b, v) => b ? N[v] : "").filter(Boolean).join(", ")}} ${SX.flag(negInfOk)} · the finite d[] left for those vertices are not distances`
      : `d = ⟨${R.d.map(fmt).join(", ")}⟩ vs brute force over all ${brute.paths - 1} simple paths from s: ⟨${brute.d.map(fmt).join(", ")}⟩ ${SX.flag(agree)} · lightest simple cycle reachable from s weighs ${cyc.minW} ≥ 0 ${SX.flag(!cyc.negative)}`));
  }
  SX.on("bf-inst", "change", build); SX.on("bf-order", "change", build);
  build();
})();

/* ── chunk B figures ── */
/* ── shared drawing helper for chunk B's small directed graphs: straight arcs with arrowheads, weight labels, vertex discs ── */
SX.bDrawGraph = function (g, G, pos, opt) {
  const o = Object.assign({ r: 15, name: v => v, edgeColor: () => AC.line, edgeWidth: () => 1.5, edgeLabel: e => e.w, vertexFill: () => AC.panel2, vertexText: () => AC.ink, below: null, above: null, labelColor: () => AC.muted, curve: 0 }, opt || {});
  const undirected = G.directed === false; const arcs = undirected ? G.edges.map((e, i) => ({ u: e.u, v: e.v, w: e.w, id: i })) : SR.b_arcs(G);
  arcs.forEach(e => {
    const a = pos[e.u], b = pos[e.v]; const dx = b[0] - a[0], dy = b[1] - a[1]; const L = Math.hypot(dx, dy) || 1; const ux = dx / L, uy = dy / L;
    /* offset a reverse arc so both directions are visible */
    const rev = !undirected && arcs.some(f => f.u === e.v && f.v === e.u); const off = rev ? 6 : 0; const nx = -uy * off, ny = ux * off;
    const x1 = a[0] + ux * o.r + nx, y1 = a[1] + uy * o.r + ny, x2 = b[0] - ux * (o.r + (undirected ? 0 : 2)) + nx, y2 = b[1] - uy * (o.r + (undirected ? 0 : 2)) + ny;
    if (undirected) g.append("line").attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2).attr("stroke", o.edgeColor(e)).attr("stroke-width", o.edgeWidth(e));
    else AL.arrow(g, x1, y1, x2, y2, { color: o.edgeColor(e), w: o.edgeWidth(e), head: 7 });
    const lbl = o.edgeLabel(e); if (lbl !== null && lbl !== undefined) {
      const mx = (x1 + x2) / 2 - uy * 9 + nx, my = (y1 + y2) / 2 + ux * 9 + ny;
      g.append("text").attr("x", mx).attr("y", my + 4).attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", o.labelColor(e)).text(lbl); }
  });
  for (let v = 0; v < G.n; v++) { const p = pos[v]; if (!p) continue;
    g.append("circle").attr("cx", p[0]).attr("cy", p[1]).attr("r", o.r).attr("fill", o.vertexFill(v)).attr("stroke", AC.line);
    g.append("text").attr("x", p[0]).attr("y", p[1] + 4).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", o.vertexText(v)).text(o.name(v));
    if (o.below) { const t = o.below(v); if (t !== null && t !== undefined) g.append("text").attr("x", p[0]).attr("y", p[1] + o.r + 13).attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", AC.good).text(t); }
    if (o.above) { const t = o.above(v); if (t !== null && t !== undefined) g.append("text").attr("x", p[0]).attr("y", p[1] - o.r - 5).attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", AC.muted).text(t); }
  }
};
SX.bInf = x => x === Infinity ? "∞" : x === -Infinity ? "−∞" : x;

/* ── 06  #dag-svg  one pass over the task DAG: shortest / longest, the critical path lit ── */
(function () {
  if (!SX.has("dag-svg")) return;
  const W = 680, H = 300;
  function build() {
    const mode = SX.val("dag-mode", "long"); const tasks = SI.b_tasks; const n = tasks.length;
    const cp = SR.b_criticalPath(tasks, AL.counter()); const G = cp.G; const pos = SI.b_taskPos;
    const c = AL.counter(); const raw = []; const R = SR.b_dagSP(G, G.S, mode, c, f => raw.push(f));
    const relax = c.get("relax"); const brute = SR.b_bruteAllPaths(G, G.S); const key = mode === "long" ? "long" : "short";
    const agree = R.d.every((x, v) => x === brute[key][v]);
    const Gn = { n: G.n, edges: G.edges.map(e => ({ u: e.u, v: e.v, w: -e.w })), directed: true }; const neg = SR.b_dagSP(Gn, G.S, "short"); const negAgree = neg.d.every((x, v) => -x === cp.fwd.d[v]);
    const cc = AL.counter(); SR.b_criticalPath(tasks, cc);
    /* frames: initial, then one per vertex processed (state after its out-edges are relaxed), then a final frame with the path lit */
    const frames = [{ label: "initialised: d[S] = 0, every other d = " + (mode === "long" ? "−∞" : "∞") + " · topological order " + R.order.map(v => name(v)).join(" "), d: raw[0].d, u: -1, done: [] }];
    const done = [];
    for (let i = 0; i < raw.length; i++) { if (raw[i].vertex === undefined) continue; const u = raw[i].vertex; let j = i + 1; let last = raw[i]; const relaxed = []; while (j < raw.length && raw[j].vertex === undefined) { last = raw[j]; relaxed.push(raw[j]); j++; } done.push(u);
      frames.push({ label: "take " + name(u) + " (d = " + SX.bInf(last.d[u]) + "): relax " + (relaxed.length ? relaxed.map(r => name(u) + "→" + name(r.v) + (r.changed ? " ⇒ d[" + name(r.v) + "] = " + SX.bInf(r.d[r.v]) : " (no change)")).join(", ") : "nothing — no out-edges"), d: last.d, u, done: done.slice(), edges: relaxed.map(r => r.v) }); }
    const finalPath = SR.b_pathOf(R.pred, G.T);
    frames.push({ label: (mode === "long" ? "longest S → T = " + R.d[G.T] + " = project length; critical path " : "shortest S → T = " + R.d[G.T] + " along ") + finalPath.map(name).join(" → "), d: R.d, u: -1, done: done.slice(), path: finalPath, final: true });
    function name(v) { return v < n ? tasks[v].id : v === G.S ? "S" : "T"; }
    const svg = d3.select("#dag-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 30, r: 20, t: 24, b: 40 });
    function render(f) {
      F.g.selectAll("*").remove();
      const onPath = e => f.path && f.path.some((v, i) => i + 1 < f.path.length && v === e.u && f.path[i + 1] === e.v);
      SX.bDrawGraph(F.g, G, pos, { name, edgeColor: e => onPath(e) ? AC.good : (f.u === e.u ? AC.a2 : f.done.indexOf(e.u) >= 0 ? AC.accent : AC.line), edgeWidth: e => onPath(e) ? 3.5 : f.u === e.u ? 3 : 1.5,
        labelColor: e => f.u === e.u ? AC.a2 : AC.muted,
        vertexFill: v => f.path && f.path.indexOf(v) >= 0 ? AC.good : v === f.u ? AC.a2 : f.done.indexOf(v) >= 0 ? AC.accent : AC.panel2, vertexText: v => (v === f.u || f.done.indexOf(v) >= 0 || (f.path && f.path.indexOf(v) >= 0)) ? AC.bg : AC.ink,
        below: v => "d = " + SX.bInf(f.d[v]), above: v => v < n ? "dur " + tasks[v].dur : null });
      F.g.append("text").attr("x", 0).attr("y", F.ih + 30).attr("font-size", 11).attr("fill", AC.ink).text(f.label);
    }
    const st = AL.stepper(svg, { frames, render, delay: 800, label: "step" }); st.go(frames.length - 1);
    const slackTxt = tasks.map((t, v) => t.id + ":" + cp.slack[v]).join(" ");
    SX.setHtml("dag-readout", `V = ${G.n}, E = ${G.edges.length}, topological order ${R.order.map(name).join(" ")} · ${mode === "long" ? "longest" : "shortest"} from S: ⟨${R.d.map(SX.bInf).join(", ")}⟩ (order A B C D E F G S T) · measured relaxations <b>${relax}</b> = E ${SX.flag(relax === G.edges.length)} · brute force over all ${brute.paths - 1} paths leaving S (${brute.paths} counting the empty path): ${SX.flag(agree)} · longest by negating the weights and running the shortest sweep: ${SX.flag(negAgree)} · project length <b>${cp.length}</b>, critical path ${cp.path.map(v => tasks[v].id).join(" → ")}, slack ${slackTxt}, two sweeps = <b>${cc.get("relax")}</b> relaxations = 2E ${SX.flag(cc.get("relax") === 2 * G.edges.length)}`);
  }
  SX.on("dag-mode", "change", build);
  build();
})();

/* ── 07  #diffcon-svg  difference constraints: the constraint graph, Bellman-Ford by rounds, x[] and the checks; the flipped system's negative cycle ── */
(function () {
  if (!SX.has("diffcon-svg")) return;
  const W = 680, H = 300;
  function build() {
    const flip = SX.checked("diffcon-flip", false);
    const cons = SI.b_diffcon.map((k, i) => (flip && i === SI.b_diffconFlipIndex) ? SI.b_diffconFlip : k);
    const n = 5; const G = SR.b_diffconGraph(n, cons); const pos = SI.b_diffconPos;
    const c = AL.counter(); const rounds = []; const bf = SR.b_bellmanFord(G, 0, c, f => rounds.push(f));
    const x = bf.d; const checks = SR.b_diffconCheck(x, cons); const allOk = checks.every(k => k.ok);
    const shifted = x.map(v => v + 4); const shiftOk = SR.b_diffconCheck(shifted, cons).every(k => k.ok);
    const brute = bf.negCycle ? null : SR.b_bruteAllPaths(G, 0);
    const cycW = bf.negCycle ? bf.negCycle.reduce((s, u, i) => { const v = bf.negCycle[(i + 1) % bf.negCycle.length]; const e = G.edges.find(e => e.u === u && e.v === v); return s + e.w; }, 0) : null;
    const init = new Array(G.n).fill(Infinity); init[0] = 0;
    const frames = [{ label: "initialised: d[v₀] = 0, every other d = ∞", d: init, changed: [] }];
    let prev = init; rounds.forEach(r => { const ch = r.d.map((v, i) => v !== prev[i] ? i : -1).filter(i => i >= 0); frames.push({ label: `round ${r.round}: ${r.changed ? "changed d[" + ch.map(i => "v" + i).join("], d[") + "]" : "nothing changed — the estimates are final"}` + (r.round === G.n && r.changed ? " — the V-th round still improved something: NEGATIVE CYCLE" : ""), d: r.d, changed: ch, last: r.round === rounds[rounds.length - 1].round }); prev = r.d; });
    if (bf.negCycle) frames[frames.length - 1].cycle = bf.negCycle;
    const svg = d3.select("#diffcon-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 30, r: 20, t: 20, b: 40 });
    const name = v => v === 0 ? "v₀" : "v" + "₀₁₂₃₄₅"[v];
    function render(f) {
      F.g.selectAll("*").remove();
      const onCyc = e => f.cycle && f.cycle.some((u, i) => u === e.u && f.cycle[(i + 1) % f.cycle.length] === e.v);
      SX.bDrawGraph(F.g, G, pos, { name, edgeColor: e => onCyc(e) ? AC.bad : e.u === 0 ? AC.grid : (flip && e.id === n + SI.b_diffconFlipIndex) ? AC.a2 : AC.muted, edgeWidth: e => onCyc(e) ? 3.5 : e.u === 0 ? 1 : 1.8,
        labelColor: e => onCyc(e) ? AC.bad : e.u === 0 ? AC.line : AC.ink,
        vertexFill: v => f.cycle && f.cycle.indexOf(v) >= 0 ? AC.bad : f.changed.indexOf(v) >= 0 ? AC.a2 : v === 0 ? AC.accent : AC.panel2, vertexText: v => (f.changed.indexOf(v) >= 0 || v === 0 || (f.cycle && f.cycle.indexOf(v) >= 0)) ? AC.bg : AC.ink,
        below: v => "d = " + SX.bInf(f.d[v]) });
      F.g.append("text").attr("x", 0).attr("y", F.ih + 30).attr("font-size", 11).attr("fill", f.cycle ? AC.bad : AC.ink).text(f.label + (f.cycle ? " " + f.cycle.map(name).join(" → ") + " → " + name(f.cycle[0]) + " of weight " + cycW : ""));
    }
    const st = AL.stepper(svg, { frames, render, delay: 900, label: "round" }); st.go(frames.length - 1);
    const conTxt = checks.map((k, i) => `x${"₀₁₂₃₄₅"[k.j]} − x${"₀₁₂₃₄₅"[k.i]} = ${k.lhs} ≤ ${k.b} ${k.ok ? "✓" : "✗"}`).join("; ");
    if (bf.negCycle) SX.setHtml("diffcon-readout", `V = ${G.n}, E = ${G.edges.length} · Bellman-Ford ran <b>${bf.rounds}</b> rounds = V, <b>${c.get("relax")}</b> relaxations = ${bf.rounds} × ${G.edges.length} ${SX.flag(c.get("relax") === bf.rounds * G.edges.length)} · the V-th round still improved an estimate ⇒ <span style="color:${AC.bad}">negative cycle</span> ${bf.negCycle.map(name).join(" → ")} → ${name(bf.negCycle[0])}, weight ${cycW} &lt; 0 ${SX.flag(cycW < 0)} · summing its ${bf.negCycle.length} constraints gives 0 ≤ ${cycW}: the system is <b>infeasible</b> · the estimates after the last round, ⟨${x.map(SX.bInf).join(", ")}⟩, are not distances — constraints checked against them anyway: ${conTxt}`);
    else SX.setHtml("diffcon-readout", `V = ${G.n}, E = ${G.edges.length} · Bellman-Ford from v₀ stopped after <b>${bf.rounds}</b> rounds (the last changed nothing), <b>${c.get("relax")}</b> relaxations = ${bf.rounds} × ${G.edges.length} ${SX.flag(c.get("relax") === bf.rounds * G.edges.length)} · x = (${x.slice(1).join(", ")}) · constraints: ${conTxt} — all satisfied ${SX.flag(allOk)} · shifted by +4: (${shifted.slice(1).join(", ")}) also satisfies all ${SX.flag(shiftOk)} · brute force over all ${brute.paths - 1} paths from v₀ gives the same distances ${SX.flag(brute.short.every((v, i) => v === x[i]))}`);
  }
  SX.on("diffcon-flip", "change", build);
  build();
})();

/* ── 08  #dial-svg  0-1 BFS with the deque drawn / Dial's buckets, stepped per expansion; checked against a binary-heap Dijkstra ── */
(function () {
  if (!SX.has("dial-svg")) return;
  const W = 680, H = 360;
  function build() {
    const alg = SX.val("dial-alg", "01"), inst = SX.val("dial-inst", "grid"); const C = inst === "small" ? 3 : 4;
    let G, pos, r = 11;
    if (inst === "grid") { G = alg === "01" ? SR.b_gridGraph(5, 8, 0, 1, 7) : SR.b_gridGraph(5, 8, 1, C, 7); pos = []; for (let i = 0; i < 5; i++) for (let j = 0; j < 8; j++) pos.push([40 + j * 80, 24 + i * 46]); }
    else { G = alg === "01" ? SI.b_g01small : SI.b_gDialSmall; pos = G.pos.map(p => [p[0] + 60, p[1] * 0.85]); r = 14; }
    const c = AL.counter(); const frames = []; const R = alg === "01" ? SR.b_zeroOneBFS(G, 0, c, f => frames.push(f)) : SR.b_dial(G, 0, C, c, f => frames.push(f));
    const cd = AL.counter(); const dj = SR.b_dijkstra(G, 0, cd); const agree = R.d.every((x, i) => x === dj.d[i]);
    /* the invariant check over random 0/1 grids, repeated at load */
    let inv = 0, tot = 0; if (alg === "01") for (let seed = 1; seed <= 200; seed++) { const Gs = SR.b_gridGraph(4, 6, 0, 1, seed); const cs = AL.counter(); const rs = SR.b_zeroOneBFS(Gs, 0, cs); const ds = SR.b_dijkstra(Gs, 0); tot++; if (rs.maxDistinct <= 2 && rs.monotone && cs.get("pop") - cs.get("stale") === Gs.n && rs.d.every((x, i) => x === ds.d[i])) inv++; }
    const init = new Array(G.n).fill(Infinity); init[0] = 0;
    const all = [{ init: true, d: init, u: -1, deque: alg === "01" ? [{ v: 0, label: 0 }] : null, buckets: alg === "dial" ? Array.from({ length: C + 1 }, (_, i) => i === 0 ? [0] : []) : null, cur: 0, done: [] }];
    const done = []; frames.forEach(f => { done.push(f.u); all.push(Object.assign({}, f, { done: done.slice() })); });
    const svg = d3.select("#dial-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 20, r: 20, t: 12, b: 20 });
    function render(f) {
      F.g.selectAll("*").remove();
      SX.bDrawGraph(F.g, { n: G.n, edges: G.edges, directed: G.directed }, pos, { r, edgeColor: e => (f.u === e.u || (G.directed === false && f.u === e.v)) ? AC.a2 : e.w === 0 ? AC.teal : AC.line, edgeWidth: e => (f.u === e.u || (G.directed === false && f.u === e.v)) ? 2.5 : 1.5, edgeLabel: e => e.w, labelColor: e => e.w === 0 ? AC.teal : AC.muted,
        vertexFill: v => v === f.u ? AC.a2 : f.done.indexOf(v) >= 0 ? AC.accent : AC.panel2, vertexText: v => (v === f.u || f.done.indexOf(v) >= 0) ? AC.bg : AC.ink, name: v => SX.bInf(f.d[v]) });
      const y0 = F.ih - 78;
      F.g.append("text").attr("x", 0).attr("y", y0).attr("font-size", 11).attr("fill", AC.muted).text(alg === "01" ? "deque, front → back, as vertex(label):" : `buckets B[0 … ${C}] (label mod ${C + 1}), cur = ${f.cur}:`);
      if (alg === "01") { const items = f.deque.map(x => `${x.v}(${x.label})`); if (!items.length) items.push("∅"); AL.row(F.g, items, { x: 0, y: y0 + 8, w: 44, h: 24, gap: 3, index: false, fontSize: 11, mark: (i) => f.deque[i] && f.deque[i].label > f.d[f.deque[i].v] ? AC.line : null }); }
      else { f.buckets.forEach((b, i) => { const x = i * 130; const live = i === f.cur % (C + 1); F.g.append("rect").attr("x", x).attr("y", y0 + 8).attr("width", 122).attr("height", 40).attr("rx", 4).attr("fill", live ? AC.grid : AC.panel2).attr("stroke", live ? AC.a2 : AC.line); F.g.append("text").attr("x", x + 4).attr("y", y0 + 20).attr("font-size", 10).attr("fill", AC.muted).text(`B[${i}]`); F.g.append("text").attr("x", x + 6).attr("y", y0 + 40).attr("font-size", 11).attr("fill", AC.ink).text(b.length ? b.map(v => `${v}(${f.d[v]})`).join(" ") : "—"); }); }
      F.g.append("text").attr("x", 0).attr("y", F.ih + 12).attr("font-size", 11).attr("fill", AC.ink).text(f.init ? "initialised: d[0] = 0, others ∞" : `expanded ${f.u} (d = ${f.d[f.u]}); vertices labelled with their current d[]`);
    }
    const st = AL.stepper(svg, { frames: all, render, delay: 700, label: "expansion" }); st.go(all.length - 1);
    const zeros = G.edges.filter(e => e.w === 0).length;
    if (alg === "01") SX.setHtml("dial-readout", `V = ${G.n}, E = ${G.edges.length} (${zeros} zero-weight) · 0-1 BFS: <b>${c.get("pop")}</b> pops (${c.get("stale")} stale), <b>${c.get("front")}</b> front pushes, <b>${c.get("back")}</b> back pushes, <b>${c.get("relax")}</b> relaxations = ${G.directed === false ? "2E" : "E"} ${SX.flag(c.get("relax") === (G.directed === false ? 2 : 1) * G.edges.length)} · vertices expanded ${c.get("pop") - c.get("stale")} = V ${SX.flag(c.get("pop") - c.get("stale") === G.n)} · the deque never held more than <b>${R.maxDistinct}</b> distinct labels (≤ 2 ${SX.flag(R.maxDistinct <= 2)}) and its labels stayed non-decreasing ${SX.flag(R.monotone)}; longest deque ${R.maxLen} · distances ⟨${R.d.map(SX.bInf).join(", ")}⟩ agree with binary-heap Dijkstra (${cd.get("push")} pushes, ${cd.get("pop")} pops, ${cd.get("stale")} stale) ${SX.flag(agree)} · invariant re-checked on ${tot} random 4 × 6 grids: holds on <b>${inv}</b> ${SX.flag(inv === tot)}`);
    else SX.setHtml("dial-readout", `V = ${G.n}, E = ${G.edges.length}, weights 1 … ${C}, C = ${C}, ${C + 1} buckets · Dial: <b>${c.get("insert")}</b> inserts, <b>${c.get("pop")}</b> pops (${c.get("stale")} stale), <b>${c.get("scan")}</b> bucket inspections = pops + label of the last pop (${c.get("pop")} + ${R.lastCur}) ${SX.flag(c.get("scan") === c.get("pop") + R.lastCur)}${R.lastCur === R.maxDist ? ", here equal to pops + largest distance " + R.maxDist : ", the last pop being stale, " + (R.lastCur - R.maxDist) + " above the largest distance " + R.maxDist}; inspections − pops within [largest distance, largest distance + C] ${SX.flag(c.get("scan") - c.get("pop") >= R.maxDist && c.get("scan") - c.get("pop") <= R.maxDist + C)}, <b>${c.get("relax")}</b> relaxations = ${G.directed === false ? "2E" : "E"} ${SX.flag(c.get("relax") === (G.directed === false ? 2 : 1) * G.edges.length)} · every live label within C of the pointer, by construction · distances ⟨${R.d.map(SX.bInf).join(", ")}⟩ agree with binary-heap Dijkstra (${cd.get("push")} pushes, ${cd.get("pop")} pops, ${cd.get("stale")} stale) ${SX.flag(agree)}`);
  }
  SX.on("dial-alg", "change", build); SX.on("dial-inst", "change", build);
  build();
})();

/* ── 09  #apsp-svg  all pairs by (min, +) squaring: the matrices at each product, stepped ── */
(function () {
  if (!SX.has("apsp-svg")) return;
  const W = 680, H = 280;
  function build() {
    const mode = SX.val("apsp-mode", "fast"); const G = SI.b_g4; const n = G.n; const Wm = SR.b_weightMatrix(G);
    const c = AL.counter(); const R = mode === "fast" ? SR.b_apspSquaring(Wm, c) : SR.b_apspSlow(Wm, c);
    const cf = AL.counter(); const FW = SR.b_floydWarshall(Wm, cf); const br = SR.b_bruteAllPairs(G);
    const same = (A, B) => A.every((r, i) => r.every((x, j) => x === B[i][j]));
    const again = SR.b_minPlus(R.D, R.D); const idem = same(again, R.D);
    const expected = mode === "fast" ? Math.ceil(Math.log2(n - 1)) : n - 2;
    const frames = R.seq.map((s, i) => ({ m: s.m, L: s.L, prev: i ? R.seq[i - 1].L : null, i }));
    const svg = d3.select("#apsp-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 20, r: 16, t: 20, b: 30 });
    const pos = G.pos.map(p => [p[0] + 10, p[1] + 10]);
    function render(f) {
      F.g.selectAll("*").remove();
      SX.bDrawGraph(F.g, G, pos, { r: 14, edgeColor: e => e.w < 0 ? AC.rose : AC.muted, labelColor: e => e.w < 0 ? AC.rose : AC.ink });
      F.g.append("text").attr("x", 10).attr("y", F.ih + 22).attr("font-size", 11).attr("fill", AC.muted).text("negative edges in rose; no negative cycle");
      /* the previous matrix (small, grey) and the current one (large) */
      const x0 = 300;
      if (f.prev) { F.g.append("text").attr("x", x0).attr("y", 4).attr("font-size", 11).attr("fill", AC.muted).text(`L⁽${frames[f.i - 1].m}⁾`); SX.grid(F.g, n, n, { x: x0, y: 12, w: 26, h: 20, gap: 2, text: (i, j) => SX.bInf(f.prev[i][j]), fill: () => null, fontSize: 10, rowLabel: i => i, colLabel: j => j }); }
      const x1 = f.prev ? x0 + 150 : x0;
      F.g.append("text").attr("x", x1).attr("y", 4).attr("font-size", 12).attr("fill", AC.ink).text(f.i === 0 ? "L⁽¹⁾ = W  (paths of ≤ 1 edge)" : mode === "fast" ? `L⁽${f.m}⁾ = L⁽${frames[f.i - 1].m}⁾ ⊗ L⁽${frames[f.i - 1].m}⁾  (≤ ${f.m} edges)` : `L⁽${f.m}⁾ = L⁽${f.m - 1}⁾ ⊗ W  (≤ ${f.m} edges)`);
      const g = SX.grid(F.g, n, n, { x: x1, y: 12, w: 40, h: 30, gap: 3, text: (i, j) => SX.bInf(f.L[i][j]), fill: (i, j) => f.prev && f.L[i][j] < f.prev[i][j] ? AC.a2 : null, fontSize: 12, rowLabel: i => i, colLabel: j => j });
      const improved = f.prev ? f.L.flatMap((r, i) => r.map((x, j) => x < f.prev[i][j] ? `(${i},${j}) ${SX.bInf(f.prev[i][j])} → ${x}` : null).filter(Boolean)) : [];
      F.g.append("text").attr("x", x1).attr("y", 12 + n * 33 + 16).attr("font-size", 10.5).attr("fill", AC.ink).text(f.prev ? (improved.length ? "improved: " + improved.join(", ") : "no entry improved — the matrix is the distance matrix") : "wᵢᵢ = 0, ∞ where no edge");
      if (f.i === frames.length - 1) F.g.append("text").attr("x", x1).attr("y", 12 + n * 33 + 32).attr("font-size", 10.5).attr("fill", AC.good).text(`m = ${f.m} ≥ V − 1 = ${n - 1}: this is D`);
    }
    const st = AL.stepper(svg, { frames, render, delay: 1000, label: "matrix" }); st.go(frames.length - 1);
    SX.setHtml("apsp-readout", `V = ${n}, E = ${G.edges.length} · ${mode === "fast" ? "squarings" : "products"} <b>${R.products}</b> = ${mode === "fast" ? "⌈log₂(V − 1)⌉" : "V − 2"} = ${expected} ${SX.flag(R.products === expected)} · (min, +) steps measured <b>${c.get("mult")}</b> = ${R.products} × V³ = ${R.products * n ** 3} ${SX.flag(c.get("mult") === R.products * n ** 3)} · final matrix rows ${R.D.map(r => "⟨" + r.map(SX.bInf).join(", ") + "⟩").join(" ")} · Floyd-Warshall (${cf.get("triple")} triples = V³) ${SX.flag(same(FW.D, R.D))} · brute force over all ${br.paths - n} simple paths ${SX.flag(same(br.D, R.D))} · one more product leaves it unchanged ${SX.flag(idem)} · negative cycle? ${FW.negCycle ? "yes" : "no (every dᵢᵢ = 0)"} ${SX.flag(!FW.negCycle)}`);
  }
  SX.on("apsp-mode", "change", build);
  build();
})();

/* ── 10  #fw-svg  Floyd-Warshall stepped by k, row/column k tinted, improved cells flashed, one path reconstructed ── */
(function () {
  if (!SX.has("fw-svg")) return;
  const W = 680, H = 300;
  function build() {
    const pair = SX.val("fw-pair", "0,4").split(",").map(Number); const neg = SX.checked("fw-neg", false);
    const G = neg ? { n: 5, edges: SI.b_g5.edges.concat([{ u: 4, v: 1, w: -2 }]), directed: true, pos: SI.b_g5.pos } : SI.b_g5; const n = G.n; const Wm = SR.b_weightMatrix(G);
    const c = AL.counter(); const frames = []; const FW = SR.b_floydWarshall(Wm, c, f => frames.push(f));
    const cb = AL.counter(); const BF = neg ? null : SR.b_bfAllPairs(G, cb); const br = neg ? null : SR.b_bruteAllPairs(G);
    const same = (A, B) => A.every((r, i) => r.every((x, j) => x === B[i][j]));
    const path = neg ? null : SR.b_fwPath(FW.next, pair[0], pair[1]); const pw = path ? SR.b_pathWeight(Wm, path) : null;
    const svg = d3.select("#fw-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 20, r: 16, t: 20, b: 30 });
    const pos = G.pos.map(p => [p[0] + 10, p[1] + 10]);
    function render(f, idx) {
      F.g.selectAll("*").remove();
      const last = idx === frames.length - 1; const onPath = e => last && path && path.some((v, i) => i + 1 < path.length && v === e.u && path[i + 1] === e.v);
      SX.bDrawGraph(F.g, G, pos, { r: 14, edgeColor: e => onPath(e) ? AC.good : (neg && e.u === 4 && e.v === 1) ? AC.bad : e.w < 0 ? AC.rose : AC.muted, edgeWidth: e => onPath(e) ? 3.5 : 1.5, labelColor: e => e.w < 0 ? AC.rose : AC.ink,
        vertexFill: v => f.k === v ? AC.a2 : (last && path && path.indexOf(v) >= 0) ? AC.good : AC.panel2, vertexText: v => (f.k === v || (last && path && path.indexOf(v) >= 0)) ? AC.bg : AC.ink });
      const x1 = 360;
      F.g.append("text").attr("x", x1).attr("y", 4).attr("font-size", 12).attr("fill", AC.ink).text(f.k < 0 ? "D⁽⁰⁾ = W — no intermediates allowed" : `D⁽${f.k + 1}⁾ — intermediates ⊆ {0 … ${f.k}}  (k = ${f.k})`);
      const upd = new Set(f.updated.map(p => p.join(",")));
      SX.grid(F.g, n, n, { x: x1, y: 14, w: 40, h: 28, gap: 3, text: (i, j) => SX.bInf(f.D[i][j]), fill: (i, j) => upd.has(i + "," + j) ? AC.a2 : (f.k >= 0 && (i === f.k || j === f.k)) ? AC.grid : (i === j && f.D[i][j] < 0) ? AC.bad : null, fontSize: 12, rowLabel: i => i, colLabel: j => j });
      F.g.append("text").attr("x", x1).attr("y", 14 + n * 31 + 14).attr("font-size", 10.5).attr("fill", AC.ink).text(f.k < 0 ? "wᵢᵢ = 0; ∞ where there is no edge" : f.updated.length ? `improved ${f.updated.length} cell${f.updated.length > 1 ? "s" : ""}: ${f.updated.map(p => "(" + p.join(",") + ")").join(" ")}` : "no cell improved in this stage");
      if (last) F.g.append("text").attr("x", x1).attr("y", 14 + n * 31 + 30).attr("font-size", 10.5).attr("fill", FW.negCycle ? AC.bad : AC.good).text(FW.negCycle ? `diagonal ⟨${FW.D.map((r, i) => r[i]).join(", ")}⟩ has a negative entry: NEGATIVE CYCLE` : `diagonal all 0: no negative cycle; path ${pair[0]} → ${pair[1]}: ${path ? path.join(" → ") : "none"}`);
    }
    const st = AL.stepper(svg, { frames, render, delay: 900, label: "stage" }); st.go(frames.length - 1);
    const total = frames.reduce((s, f) => s + f.updated.length, 0);
    if (FW.negCycle) SX.setHtml("fw-readout", `V = ${n}, E = ${G.edges.length} (with 4 → 1 of weight −2) · triples measured <b>${c.get("triple")}</b> = V³ = ${n ** 3} ${SX.flag(c.get("triple") === n ** 3)} · ${total} cells improved · final diagonal ⟨${FW.D.map((r, i) => r[i]).join(", ")}⟩ — a negative entry means a negative cycle through that vertex ${SX.flag(FW.negCycle)}; the cycle 1 → 2 → 4 → 1 weighs (−3) + 2 + (−2) = ${SR.b_pathWeight(Wm, [1, 2, 4, 1])} · the off-diagonal entries are not shortest simple-path weights and the next matrix may loop, so no path is reconstructed`);
    else SX.setHtml("fw-readout", `V = ${n}, E = ${G.edges.length} · triples measured <b>${c.get("triple")}</b> = V³ = ${n ** 3} ${SX.flag(c.get("triple") === n ** 3)} · <b>${total}</b> cells improved over the ${n} stages · D rows ${FW.D.map(r => "⟨" + r.map(SX.bInf).join(", ") + "⟩").join(" ")} · V runs of Bellman-Ford (${cb.get("relax")} relaxations in total, early exit) ${SX.flag(same(BF, FW.D))} · brute force over all ${br.paths - n} non-empty simple paths ${SX.flag(same(br.D, FW.D))} · path ${pair[0]} → ${pair[1]} by following next: <b>${path.join(" → ")}</b>, edges re-summed ${pw} = D[${pair[0]}][${pair[1]}] = ${FW.D[pair[0]][pair[1]]} ${SX.flag(pw === FW.D[pair[0]][pair[1]])} · diagonal all 0: no negative cycle ${SX.flag(!FW.negCycle)}`);
  }
  SX.on("fw-pair", "change", build); SX.on("fw-neg", "change", build);
  build();
})();

/* ── 11  #johnson-svg  Johnson's algorithm stepped through its phases on the §13 graph ── */
(function () {
  if (!SX.has("johnson-svg")) return;
  const W = 680, H = 300;
  function build() {
    const G = SI.b_g5; const n = G.n; const Wm = SR.b_weightMatrix(G);
    const c = AL.counter(); const phases = []; const J = SR.b_johnson(G, c, f => phases.push(f));
    const FW = SR.b_floydWarshall(Wm); const br = SR.b_bruteAllPairs(G);
    const same = (A, B) => A.every((r, i) => r.every((x, j) => x === B[i][j]));
    const frames = [{ kind: "orig", label: "the graph G with its weights w — two negative edges, no negative cycle" }]
      .concat(phases.map(p => p.phase === "h" ? { kind: "h", h: p.h, label: `G′ = G + s* with zero edges; Bellman-Ford from s* gives h(v) = δ′(s*, v) — written above each vertex` }
        : p.phase === "reweight" ? { kind: "rw", h: p.h, wh: p.wh, label: "every edge relabelled ŵ(u, v) = w(u, v) + h(u) − h(v) — all non-negative, zero on the shortest-path tree from s*" }
        : p.phase === "dijkstra" ? { kind: "dj", h: p.h, wh: p.wh, s: p.s, dh: p.dh, d: p.d, order: p.order, label: `Dijkstra from ${p.s} on ŵ — settling order ${p.order.join(" ")}; d̂ then d = d̂ − h(${p.s}) + h(v) under each vertex` }
        : { kind: "done", h: p.h, wh: p.wh, D: p.D, label: "all five rows recovered — the distance matrix D under the original weights" }));
    const svg = d3.select("#johnson-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 20, r: 16, t: 24, b: 30 });
    const pos = G.pos.map(p => [p[0] + 50, p[1] + 14]); const sPos = [10, 130];
    function render(f) {
      F.g.selectAll("*").remove();
      const rw = f.kind === "rw" || f.kind === "dj" || f.kind === "done"; const whOf = e => f.wh.find(x => x.u === e.u && x.v === e.v).wh;
      if (f.kind === "h") { for (let v = 0; v < n; v++) { const p = pos[v]; AL.arrow(F.g, sPos[0] + 14, sPos[1], p[0] - 14, p[1], { color: AC.grid, w: 1, dash: "3,3" }); } F.g.append("circle").attr("cx", sPos[0]).attr("cy", sPos[1]).attr("r", 14).attr("fill", AC.violet).attr("stroke", AC.line); F.g.append("text").attr("x", sPos[0]).attr("y", sPos[1] + 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.bg).text("s*"); }
      SX.bDrawGraph(F.g, G, pos, { r: 14, edgeLabel: e => rw ? whOf(e) : e.w, edgeColor: e => rw ? (whOf(e) === 0 ? AC.teal : AC.muted) : (e.w < 0 ? AC.rose : AC.muted), labelColor: e => rw ? (whOf(e) === 0 ? AC.teal : AC.ink) : (e.w < 0 ? AC.rose : AC.ink),
        vertexFill: v => f.kind === "dj" && v === f.s ? AC.a2 : AC.panel2, vertexText: v => f.kind === "dj" && v === f.s ? AC.bg : AC.ink,
        above: v => f.h ? "h = " + f.h[v] : null, below: v => f.kind === "dj" ? `d̂ ${f.dh[v]} → d ${f.d[v]}` : null });
      if (f.kind === "done") { F.g.append("text").attr("x", 420).attr("y", 4).attr("font-size", 11).attr("fill", AC.muted).text("D = δ(i, j)"); SX.grid(F.g, n, n, { x: 420, y: 12, w: 36, h: 26, gap: 3, text: (i, j) => SX.bInf(f.D[i][j]), fill: () => null, fontSize: 11, rowLabel: i => i, colLabel: j => j }); }
      if (f.kind === "dj") { F.g.append("text").attr("x", 420).attr("y", 4).attr("font-size", 11).attr("fill", AC.muted).text(`row ${f.s} of D so far`); SX.grid(F.g, 1, n, { x: 420, y: 12, w: 36, h: 26, gap: 3, text: (i, j) => SX.bInf(f.d[j]), fill: () => null, fontSize: 11, colLabel: j => j }); }
      F.g.append("text").attr("x", 0).attr("y", F.ih + 24).attr("font-size", 11).attr("fill", AC.ink).text(f.label);
    }
    const st = AL.stepper(svg, { frames, render, delay: 1100, label: "phase" }); st.go(frames.length - 1);
    const minWh = Math.min(...J.wh.map(e => e.wh)); const zeros = J.wh.filter(e => e.wh === 0).length;
    const Wh = J.wh.reduce((M, e) => { M[e.u][e.v] = e.wh; return M; }, Array.from({ length: n }, () => new Array(n).fill(Infinity)));
    const pA = [0, 3, 1, 2, 4], pB = [0, 1, 2, 4]; const wA = SR.b_pathWeight(Wm, pA), whA = SR.b_pathWeight(Wh, pA), wB = SR.b_pathWeight(Wm, pB), whB = SR.b_pathWeight(Wh, pB); const shift = J.h[0] - J.h[4]; const shiftOk = whA - wA === shift && whB - wB === shift;
    SX.setHtml("johnson-readout", `V = ${n}, E = ${G.edges.length} · Bellman-Ford on G′ (${n + 1} vertices, ${G.edges.length + n} arcs): ${J.bfRounds} rounds, <b>${c.get("relax")}</b> relaxations · h = (${J.h.join(", ")}) · reweighted ŵ: ${J.wh.map(e => `${e.u}→${e.v} ${e.wh}`).join(", ")} — minimum ${minWh} ≥ 0 ${SX.flag(J.allNonNeg)}, ${zeros} zero edges · ${n} Dijkstra runs: ${c.get("push")} pushes, ${c.get("pop")} pops (${c.get("stale")} stale), <b>${c.get("relax2")}</b> relaxations = V·E = ${n * G.edges.length} ${SX.flag(c.get("relax2") === n * G.edges.length)} · D rows ${J.D.map(r => "⟨" + r.map(SX.bInf).join(", ") + "⟩").join(" ")} · agrees with Floyd-Warshall ${SX.flag(same(J.D, FW.D))} and with brute force over all ${br.paths - n} non-empty simple paths ${SX.flag(same(J.D, br.D))} · every path's weight shifted by h(u) − h(v): checked on 0→3→1→2→4 (w ${wA}, ŵ ${whA}) and 0→1→2→4 (w ${wB}, ŵ ${whB}), both by h(0) − h(4) = ${shift} ${SX.flag(shiftOk)}`);
  }
  build();
})();

/* ── 12  #astar-svg  A* on a grid with obstacles: expansions stepped, heuristics compared, path checked against BFS ── */
(function () {
  if (!SX.has("astar-svg")) return;
  const W = 680, H = 330;
  const NAMES = { zero: "zero (Dijkstra)", euclid: "Euclidean", manhattan: "Manhattan", manhattan2: "2 × Manhattan" };
  function build() {
    const kind = SX.val("astar-h", "manhattan"), inst = SX.val("astar-inst", "scatter");
    const grid = inst === "scatter" ? SI.b_astarGrid : SI.b_astarGrid2; const Rn = grid.rows, Cn = grid.cols;
    const bfs = SR.b_gridBFS(grid);
    const c = AL.counter(); const frames = []; const R = SR.b_astar(grid, kind, c, f => frames.push(f));
    const all = ["zero", "euclid", "manhattan", "manhattan2"].map(k => { const ck = AL.counter(); const r = SR.b_astar(grid, k, ck); const hc = SR.b_heuristicConsistent(grid, k); return { k, expanded: ck.get("expand"), dist: r.dist, reopened: r.reopened, admissible: hc.admissible, consistent: hc.consistent }; });
    const svg = d3.select("#astar-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 14, r: 10, t: 12, b: 26 });
    const cs = 30, gx = 0, gy = 0; const id = (r, cc) => r * Cn + cc; const rc = k => [Math.floor(k / Cn), k % Cn];
    const s = id(grid.start[0], grid.start[1]), t = id(grid.goal[0], grid.goal[1]);
    const allFrames = [{ init: true, g: [], openSet: [s], closedList: [], expanded: -1 }].concat(frames);
    function render(f, idx) {
      F.g.selectAll("*").remove();
      const last = idx === allFrames.length - 1; const closed = new Set(f.closedList), open = new Set(f.openSet); const onPath = new Set(last && R.path ? R.path : []);
      for (let r = 0; r < Rn; r++) for (let cc = 0; cc < Cn; cc++) { const k = id(r, cc); const wall = grid.walls.has(r + "," + cc);
        const fill = wall ? AC.line : onPath.has(k) ? AC.good : k === f.expanded ? AC.a2 : closed.has(k) ? AC.accent : AC.panel2;
        F.g.append("rect").attr("x", gx + cc * cs).attr("y", gy + r * cs).attr("width", cs - 2).attr("height", cs - 2).attr("rx", 3).attr("fill", fill).attr("stroke", open.has(k) && !closed.has(k) ? AC.a2 : AC.grid).attr("stroke-width", open.has(k) && !closed.has(k) ? 2 : 1);
        if (!wall && f.g && f.g[k] !== undefined && f.g[k] !== Infinity) F.g.append("text").attr("x", gx + cc * cs + (cs - 2) / 2).attr("y", gy + r * cs + cs / 2 + 3).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", (closed.has(k) || onPath.has(k) || k === f.expanded) ? AC.bg : AC.muted).text(f.g[k]);
        if (k === s || k === t) F.g.append("text").attr("x", gx + cc * cs + (cs - 2) / 2).attr("y", gy + r * cs + cs / 2 + 3).attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", "bold").attr("fill", onPath.has(k) || closed.has(k) ? AC.bg : AC.ink).text(k === s ? "S" : "G"); }
      /* side panel */
      const px = gx + Cn * cs + 14;
      F.g.append("text").attr("x", px).attr("y", 12).attr("font-size", 11).attr("fill", AC.ink).text("expansions, all four:");
      all.forEach((a, i) => { const cur = a.k === kind; F.g.append("text").attr("x", px).attr("y", 30 + i * 16).attr("font-size", 10.5).attr("fill", cur ? AC.a2 : AC.muted).text(`${NAMES[a.k]}: ${a.expanded}, path ${a.dist}${a.dist === bfs.dist ? "" : " ✗"}`); });
      F.g.append("text").attr("x", px).attr("y", 30 + 4 * 16 + 6).attr("font-size", 10.5).attr("fill", AC.muted).text(`true shortest (BFS): ${bfs.dist}`);
      F.g.append("text").attr("x", px).attr("y", 30 + 5 * 16 + 6).attr("font-size", 10.5).attr("fill", AC.muted).text(`free cells: ${Rn * Cn - grid.walls.size}`);
      F.g.append("text").attr("x", 0).attr("y", F.ih + 20).attr("font-size", 11).attr("fill", AC.ink).text(f.init ? `start: open = {S}, f(S) = h(S) = ${AL.fmt(SR.b_gridDist(grid.start, grid.goal, kind), 2)}` : last ? (R.path ? `goal popped: path of ${R.dist} steps after ${frames.length} expansions${R.dist === bfs.dist ? " — optimal" : " — NOT optimal, BFS finds " + bfs.dist}` : "goal unreachable") : `expanded (${rc(f.expanded).join(",")}) with f = ${AL.fmt(f.f, 2)}, g = ${f.g[f.expanded]}; ${f.openSet.length} in open, ${f.closedList.length} closed`);
    }
    const st = AL.stepper(svg, { frames: allFrames, render, delay: 250, label: "expansion" }); st.go(allFrames.length - 1);
    const hc = SR.b_heuristicConsistent(grid, kind);
    SX.setHtml("astar-readout", `${Rn} × ${Cn} grid, ${grid.walls.size} walls, ${Rn * Cn - grid.walls.size} free cells; true shortest path by BFS <b>${bfs.dist}</b> steps (BFS reaches ${bfs.expanded} cells) · ${NAMES[kind]}: <b>${c.get("expand")}</b> expansions, ${c.get("push")} pushes, ${R.reopened} reopened, path <b>${R.dist}</b> ${R.dist === bfs.dist ? "= optimal" : "≠ optimal — the heuristic overestimates and the search commits to a detour"} ${SX.flag(R.dist === bfs.dist || !hc.admissible)} · heuristic admissible on this grid ${hc.admissible ? "✓" : "✗"}, consistent ${hc.consistent ? "✓" : "✗"} · all four: ${all.map(a => `${NAMES[a.k]} ${a.expanded} expansions, path ${a.dist}${a.dist === bfs.dist ? "" : " <span style=\"color:" + AC.bad + "\">✗ longer than optimal</span>"}`).join("; ")} · the admissible heuristics all return ${bfs.dist} ${SX.flag(all.filter(a => a.admissible).every(a => a.dist === bfs.dist))} · dominance: zero ≥ Euclidean ≥ Manhattan in expansions ${SX.flag(all[0].expanded >= all[1].expanded && all[1].expanded >= all[2].expanded)} · weighted A* bound: ${all[3].dist} ≤ 2 × ${bfs.dist} = ${2 * bfs.dist} ${SX.flag(all[3].dist <= 2 * bfs.dist)}`);
  }
  SX.on("astar-h", "change", build); SX.on("astar-inst", "change", build);
  build();
})();

/* ── chunk C figures ── */
/* ── chunk C shared drawing helper: a small undirected weighted graph at a fixed hand-placed layout (SI.*.pos) ── */
(function () {
  SX.c_graph = function (g, G, opt) {
    const o = Object.assign({ ox: 0, oy: 0, r: 13, edgeColor: () => AC.line, edgeWidth: () => 1.5, edgeDash: () => null, hideEdge: () => false,
      vertexFill: () => AC.panel2, vertexStroke: () => AC.line, vertexInk: null, above: () => null, below: () => null, weightColor: () => AC.muted, weightFor: null }, opt || {});
    const gg = g.append("g").attr("transform", `translate(${o.ox},${o.oy})`);
    G.edges.forEach((e, i) => { if (o.hideEdge(i)) return; const a = G.pos[e.u], b = G.pos[e.v];
      const ln = gg.append("line").attr("x1", a[0]).attr("y1", a[1]).attr("x2", b[0]).attr("y2", b[1]).attr("stroke", o.edgeColor(i)).attr("stroke-width", o.edgeWidth(i)); const d = o.edgeDash(i); if (d) ln.attr("stroke-dasharray", d);
      /* weight label, nudged perpendicular to the edge so it does not sit on the line */
      const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.sqrt(dx * dx + dy * dy) || 1; const nx = -dy / L * 9, ny = dx / L * 9;
      gg.append("text").attr("x", (a[0] + b[0]) / 2 + nx).attr("y", (a[1] + b[1]) / 2 + ny + 4).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", o.weightColor(i)).text(o.weightFor ? o.weightFor(i) : e.w); });
    for (let v = 0; v < G.n; v++) { const p = G.pos[v]; const f = o.vertexFill(v);
      gg.append("circle").attr("cx", p[0]).attr("cy", p[1]).attr("r", o.r).attr("fill", f).attr("stroke", o.vertexStroke(v)).attr("stroke-width", 1.5);
      gg.append("text").attr("x", p[0]).attr("y", p[1] + 4).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", o.vertexInk ? o.vertexInk(v) : (f === AC.panel2 || f === AC.panel ? AC.ink : AC.bg)).text(G.names ? G.names[v] : v);
      const ab = o.above(v); if (ab) gg.append("text").attr("x", p[0]).attr("y", p[1] - o.r - 5).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", ab.color || AC.a2).text(ab.text !== undefined ? ab.text : ab);
      const be = o.below(v); if (be) gg.append("text").attr("x", p[0]).attr("y", p[1] + o.r + 13).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", be.color || AC.a2).text(be.text !== undefined ? be.text : be); }
    return gg;
  };
  SX.c_names = (G, idx) => idx.map(i => SR.mstEdgeName(G, i)).join(", ");
  /* a component palette for colouring vertices by representative */
  SX.c_compColors = [AC.accent, AC.a2, AC.teal, AC.violet, AC.rose, AC.good];
})();

/* ── 13  #cut-svg  the cut property on the shared instance: crossing edges, the light edge, brute force over all spanning trees, the exchange ── */
(function () {
  if (!SX.has("cut-svg")) return;
  const W = 680, H = 300; const G = SI.gMst; const nm = i => SR.mstEdgeName(G, i);
  const B = SR.bruteSpanningTrees(G);
  function build() {
    const S = SX.val("cut-s", "0,1,3").split(",").map(Number);
    const C = SR.mstCut(G, S, B); const inS = v => S.indexOf(v) >= 0;
    const svg = d3.select("#cut-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 12, r: 12, t: 10, b: 10 });
    const isCross = i => C.cross.indexOf(i) >= 0;
    SX.c_graph(F.g, G, { ox: 0, oy: 8, edgeColor: i => i === C.light ? AC.a2 : isCross(i) ? AC.rose : AC.line, edgeWidth: i => i === C.light ? 4 : isCross(i) ? 2 : 1.5, edgeDash: i => isCross(i) && i !== C.light ? "5 3" : null,
      vertexFill: v => inS(v) ? AC.accent : AC.panel2, weightColor: i => i === C.light ? AC.a2 : isCross(i) ? AC.rose : AC.muted });
    F.g.append("text").attr("x", 0).attr("y", 272).attr("font-size", 11).attr("fill", AC.muted).text(`cut S = {${S.map(v => G.names[v]).join(", ")}} · crossing edges dashed · light edge ${nm(C.light)} (w = ${G.edges[C.light].w})`);
    const ex = C.exchange;
    if (ex) {
      const inT = i => ex.T.indexOf(i) >= 0;
      SX.c_graph(F.g, G, { ox: 340, oy: 8, edgeColor: i => i === C.light ? AC.a2 : i === ex.ePrime ? AC.rose : inT(i) ? AC.good : AC.grid, edgeWidth: i => i === C.light || i === ex.ePrime ? 4 : inT(i) ? 3 : 1, edgeDash: i => i === ex.ePrime ? "6 3" : null,
        vertexFill: v => inS(v) ? AC.accent : AC.panel2, weightColor: i => i === C.light ? AC.a2 : i === ex.ePrime ? AC.rose : inT(i) ? AC.good : AC.grid });
      F.g.append("text").attr("x", 340).attr("y", 272).attr("font-size", 11).attr("fill", AC.muted).text(`a tree T avoiding ${nm(C.light)} (w = ${ex.w1}): add ${nm(C.light)}, drop ${nm(ex.ePrime)} → w = ${ex.w2}`);
    } else {
      F.g.append("text").attr("x", 340).attr("y", 140).attr("font-size", 11).attr("fill", AC.muted).text(`every spanning tree contains ${nm(C.light)} — it is a bridge`);
    }
    const mst = B.msts[0];
    SX.setHtml("cut-readout", `cut S = {${S.map(v => G.names[v]).join(", ")}} · crossing edges ${C.cross.map(i => nm(i) + " (" + G.edges[i].w + ")").join(", ")} · light edge <b>${nm(C.light)}</b>, w = ${G.edges[C.light].w} · brute force: ${B.subsets} five-edge subsets of the 9 edges, <b>${B.trees.length}</b> of them spanning trees, minimum weight <b>${B.minW}</b>, ${B.msts.length} tree${B.msts.length === 1 ? "" : "s"} of that weight — the MST {${SX.c_names(G, mst)}} is ${B.unique ? "unique" : "not unique"} ${SX.flag(B.unique)} · the light edge is in ${C.mstsWithLight.length === B.msts.length ? "every" : "some"} MST ${SX.flag(C.inSomeMst)}${ex ? ` · the exchange on the lightest tree that avoids ${nm(C.light)}: T = {${SX.c_names(G, ex.T)}} (w = ${ex.w1}); T ∪ {${nm(C.light)}} closes the cycle ${ex.path.map(nm).join("–")}–${nm(C.light)}; its other crossing edge${ex.crossingOnCycle.length > 1 ? "s are " + ex.crossingOnCycle.map(nm).join(", ") + " and the heaviest, " + nm(ex.ePrime) + " (w = " + G.edges[ex.ePrime].w + "), is dropped" : " is " + nm(ex.ePrime) + " (w = " + G.edges[ex.ePrime].w + "), which is dropped"}: T′ = T ∪ {${nm(C.light)}} − {${nm(ex.ePrime)}} is a spanning tree ${SX.flag(ex.isTree)} of weight ${ex.w1} + ${G.edges[C.light].w} − ${G.edges[ex.ePrime].w} = <b>${ex.w2}</b> ≤ ${ex.w1} ${SX.flag(ex.w2 <= ex.w1)}${ex.w2 === B.minW ? " — and it is the MST itself" : ""}` : ` · every spanning tree contains ${nm(C.light)} (a bridge), so there is nothing to exchange`} · cycle property, every non-MST edge: ${SR.mstCycleCheck(G, B).map(r => nm(r.closing) + " closes " + r.cycle.map(nm).join("–") + ", heaviest " + nm(r.heaviest) + (r.heaviestInMst ? " IS in the MST ✗" : " ∉ MST ✓")).join("; ")} · minimum bottleneck ${B.minBottleneck}: the MST's heaviest edge is ${Math.max(...mst.map(i => G.edges[i].w))} ${SX.flag(Math.max(...mst.map(i => G.edges[i].w)) === B.minBottleneck)}, and ${B.mbsts.length} of the ${B.trees.length} trees share that bottleneck (${B.mbsts.length - B.msts.length} of them are not minimum: ${B.mbsts.filter(t => t !== mst).map(t => "{" + SX.c_names(G, t) + "} w = " + SR.mstWeightOf(G, t)).join(", ")})`);
  }
  SX.on("cut-s", "change", build);
  build();
})();

/* ── 14  #kruskal-svg  Kruskal stepped: edges in sorted order, accepted lit, rejected struck, the forest growing ── */
(function () {
  if (!SX.has("kruskal-svg")) return;
  const W = 680, H = 320; const G = SI.gMst; const nm = i => SR.mstEdgeName(G, i);
  const B = SR.bruteSpanningTrees(G);
  function build() {
    const naive = SX.val("kruskal-uf", "both") === "naive";
    const c = AL.counter(); const K = SR.kruskal(G, c, null, naive ? { rank: false, compress: "none" } : { rank: true, compress: "full" });
    const frames = [{ init: true, forest: [], rep: Array.from({ length: G.n }, (_, v) => v), weight: 0 }].concat(K.frames);
    const svg = d3.select("#kruskal-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 12, r: 12, t: 10, b: 10 });
    function render(f, k) {
      F.g.selectAll("*").remove();
      const status = {}; frames.slice(1, k + 1).forEach(fr => { status[fr.i] = fr.accepted ? "acc" : "rej"; });
      const cur = f.init ? -1 : f.i;
      /* colour vertices by component representative */
      const reps = [...new Set(f.rep)]; const colOf = v => reps.length === G.n ? AC.panel2 : (reps.length === 1 ? AC.good : SX.c_compColors[reps.indexOf(f.rep[v]) % SX.c_compColors.length]);
      SX.c_graph(F.g, G, { ox: 0, oy: 10, edgeColor: i => i === cur ? AC.a2 : status[i] === "acc" ? AC.good : status[i] === "rej" ? AC.bad : AC.line, edgeWidth: i => i === cur ? 4 : status[i] === "acc" ? 3.5 : status[i] === "rej" ? 1.5 : 1.5, edgeDash: i => status[i] === "rej" ? "4 3" : null,
        vertexFill: v => colOf(v), weightColor: i => i === cur ? AC.a2 : status[i] === "acc" ? AC.good : status[i] === "rej" ? AC.bad : AC.muted });
      F.g.append("text").attr("x", 0).attr("y", 285).attr("font-size", 11).attr("fill", AC.ink).text(f.init ? "start: 6 singleton trees, forest A = ∅" : `${nm(f.i)} (w = ${G.edges[f.i].w}): ${f.accepted ? "ACCEPT — " + f.reason : "reject — " + f.reason}`);
      F.g.append("text").attr("x", 0).attr("y", 300).attr("font-size", 11).attr("fill", AC.muted).text(`forest {${SX.c_names(G, f.forest)}} · weight so far ${f.weight} · components ${[...new Set(f.rep)].length}`);
      /* the sorted edge list */
      const gx = 350; F.g.append("text").attr("x", gx).attr("y", 14).attr("font-size", 11).attr("fill", AC.muted).text("edges in sorted order (ties by input order — stable)");
      K.order.forEach((i, r) => { const y = 24 + r * 27; const st = i === cur ? "cur" : status[i] || "todo"; const fill = st === "cur" ? AC.a2 : st === "acc" ? AC.good : st === "rej" ? AC.bad : AC.panel2;
        F.g.append("rect").attr("x", gx).attr("y", y).attr("width", 300).attr("height", 23).attr("rx", 4).attr("fill", st === "todo" ? AC.panel2 : fill).attr("stroke", AC.line).attr("opacity", st === "todo" ? 1 : 0.9);
        F.g.append("text").attr("x", gx + 8).attr("y", y + 16).attr("font-size", 11).attr("fill", st === "todo" ? AC.ink : AC.bg).text(`${r + 1}. ${nm(i)}  w = ${G.edges[i].w}`);
        F.g.append("text").attr("x", gx + 292).attr("y", y + 16).attr("text-anchor", "end").attr("font-size", 10).attr("fill", st === "todo" ? AC.muted : AC.bg).text(st === "acc" ? "accepted — joins two trees" : st === "rej" ? "rejected — closes a cycle" : st === "cur" ? "examining: find(u) = find(v)?" : "");
        if (st === "rej") F.g.append("line").attr("x1", gx + 6).attr("x2", gx + 120).attr("y1", y + 12).attr("y2", y + 12).attr("stroke", AC.bg).attr("stroke-width", 1.5); });
    }
    const st = AL.stepper(svg, { frames, render, delay: 800, label: "edge" }); st.go(frames.length - 1);
    const E = G.edges.length, V = G.n;
    const lg = Math.ceil(Math.log2(E)); const msWorst = E * lg - Math.pow(2, lg) + 1;   /* merge sort's worst-case comparison count on E keys */
    SX.setHtml("kruskal-readout", `V = ${V}, E = ${E} · sorted order ${K.order.map(i => nm(i) + "(" + G.edges[i].w + ")").join(" ")} · accepted {${SX.c_names(G, K.tree)}}, weight <b>${K.weight}</b>, ${K.tree.length} = V − 1 edges ${SX.flag(K.tree.length === V - 1)} · measured: <b>${c.get("cmp")}</b> sort comparisons (merge sort; its worst case on E = ${E} keys is E⌈log₂ E⌉ − 2^⌈log₂ E⌉ + 1 = ${msWorst} ${SX.flag(c.get("cmp") <= msWorst)}), <b>${c.get("find")}</b> finds = 2E ${SX.flag(c.get("find") === 2 * E)}, <b>${c.get("union")}</b> unions = V − 1 ${SX.flag(c.get("union") === V - 1)}, ${c.get("step")} parent-pointer hops inside the finds (${naive ? "no heuristics" : "union by rank + path compression"}) · brute force over all ${B.trees.length} spanning trees: minimum weight ${B.minW} ${SX.flag(B.minW === K.weight)}, unique ${SX.flag(B.unique)}, and Kruskal's edge set is that tree ${SX.flag(B.msts[0].slice().sort().join() === K.tree.slice().sort().join())}`);
  }
  SX.on("kruskal-uf", "change", build);
  build();
})();

/* ── 15  #uf-svg  union-find built: a forest of 12 elements under a preset op sequence, with the two heuristics as toggles; measured steps ── */
(function () {
  if (!SX.has("uf-svg")) return;
  const W = 680, H = 330; const n = SI.ufN;
  /* the random and adversarial workloads are computed once — they do not depend on the toggles */
  const randOps = SR.ufRandomOps(1000, 5000, 7); const randCmp = SR.ufCompare(1000, randOps); const nU = randOps.filter(o => o.op === "u").length;
  const chainOps = SR.ufChainOps(1000, 4001); const chainCmp = SR.ufCompare(1000, chainOps);
  const halving = SR.ufRun(1000, randOps, { rank: true, compress: "halving" }), splitting = SR.ufRun(1000, randOps, { rank: true, compress: "splitting" });
  function layoutForest(p) {
    /* children lists; each root's subtree laid out with leaves 1 unit apart; roots side by side */
    const ch = Array.from({ length: n }, () => []); const roots = []; for (let v = 0; v < n; v++) { if (p[v] === v) roots.push(v); else ch[p[v]].push(v); }
    const width = new Array(n).fill(1); const depth = new Array(n).fill(0);
    roots.forEach(r => (function measure(v, d) { depth[v] = d; if (!ch[v].length) { width[v] = 1; return; } let w = 0; ch[v].forEach(u => { measure(u, d + 1); w += width[u]; }); width[v] = w; })(r, 0));
    const x = new Array(n).fill(0); let cursor = 0;
    roots.forEach(r => { (function place(v, left) { x[v] = left + width[v] / 2; let l = left; ch[v].forEach(u => { place(u, l); l += width[u]; }); })(r, cursor); cursor += width[r] + 0.6; });
    return { x, depth, total: cursor - 0.6, maxDepth: Math.max(...depth) };
  }
  function build() {
    const seqKey = SX.val("uf-seq", "chain"); const seq = SI.ufSeqs[seqKey] || SI.ufSeqs.chain;
    const opts = { rank: SX.checked("uf-rank", false), compress: SX.val("uf-compress", "none") };
    const R = SR.ufRun(n, seq.ops, opts); const frames = R.frames;
    const cmp = SR.ufCompare(n, seq.ops);
    const svg = d3.select("#uf-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 12, r: 12, t: 10, b: 10 });
    function render(f, k) {
      F.g.selectAll("*").remove();
      const L = layoutForest(f.p); const unit = Math.min(52, (F.iw - 20) / Math.max(1, L.total)); const levelH = Math.min(46, (F.ih - 70) / Math.max(1, L.maxDepth + 1));
      const px = v => 10 + L.x[v] * unit, py = v => 30 + L.depth[v] * levelH; const rr = Math.min(11, Math.floor(levelH / 2) - 1);
      const touched = new Set(); if (f.op) { touched.add(f.op.a); if (f.op.op === "u") touched.add(f.op.b); }
      for (let v = 0; v < n; v++) if (f.p[v] !== v) F.g.append("line").attr("x1", px(v)).attr("y1", py(v)).attr("x2", px(f.p[v])).attr("y2", py(f.p[v])).attr("stroke", AC.line).attr("stroke-width", 1.5);
      for (let v = 0; v < n; v++) { const root = f.p[v] === v; F.g.append("circle").attr("cx", px(v)).attr("cy", py(v)).attr("r", rr).attr("fill", touched.has(v) ? AC.a2 : root ? AC.accent : AC.panel2).attr("stroke", AC.line);
        F.g.append("text").attr("x", px(v)).attr("y", py(v) + 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", touched.has(v) || root ? AC.bg : AC.ink).text(v);
        if (opts.rank && root) F.g.append("text").attr("x", px(v) + 14).attr("y", py(v) - 6).attr("font-size", 9).attr("fill", AC.muted).text("r" + f.rank[v]); }
      const desc = !f.op ? `start: ${n} singletons, parent[i] = i` : f.op.op === "u" ? `union(${f.op.a}, ${f.op.b}) = link(find(${f.op.a}), find(${f.op.b}))${f.result ? "" : " — already one set, no link"}: ${f.steps} parent-pointer hop${f.steps === 1 ? "" : "s"}` : `find(${f.op.a}) → root ${f.result}: ${f.steps} hop${f.steps === 1 ? "" : "s"}${opts.compress !== "none" && f.steps > 1 ? " — the path is " + (opts.compress === "full" ? "compressed to the root" : opts.compress === "halving" ? "halved" : "split") : ""}`;
      F.g.append("text").attr("x", 0).attr("y", F.ih - 16).attr("font-size", 11).attr("fill", AC.ink).text(desc);
      F.g.append("text").attr("x", 0).attr("y", F.ih - 1).attr("font-size", 11).attr("fill", AC.muted).text(`height ${f.height} · hops so far ${frames.slice(0, k + 1).reduce((s, g) => s + g.steps, 0)} · ${opts.rank ? "union by rank" : "naive linking (x's root under y's root)"} + ${opts.compress === "none" ? "no compression" : "path " + (opts.compress === "full" ? "compression" : opts.compress)}${opts.rank ? " · roots show their rank" : ""}`);
    }
    const st = AL.stepper(svg, { frames, render, delay: 700, label: "op" }); st.go(frames.length - 1);
    const findRows = R.frames.filter(f => f.op && f.op.op === "f").map(f => `find(${f.op.a}) = ${f.steps}`).join(", ");
    const table = `<table class="cmp" style="margin:6px 0"><tr><th>heuristics</th><th>total hops (${seq.ops.length} ops)</th><th>hops per find (the ${cmp[0].finds} finds incl. those inside unions)</th><th>worst single find</th><th>final height</th></tr>${cmp.map(r => `<tr><td>${r.label}</td><td>${r.steps}</td><td>${SX.f2(r.steps / r.finds)}</td><td>${r.maxFindSteps}</td><td>${r.height}</td></tr>`).join("")}</table>`;
    SX.setHtml("uf-readout", `sequence “${seqKey}” on n = ${n}: ${R.unions} union calls (${R.links} links), ${R.finds} finds in all · with the current toggles: <b>${R.steps}</b> parent-pointer hops, explicit finds cost ${findRows}, final height <b>${R.height}</b> · the same sequence under all four combinations (re-run at load):${table}random workload, n = 1,000 elements, m = 5,000 operations (${nU} unions of random pairs, ${5000 - nU} finds of random elements; seeded): hops per operation — no heuristic <b>${SX.f2(randCmp[0].perOp)}</b> (final height ${randCmp[0].height}), rank only <b>${SX.f2(randCmp[1].perOp)}</b> (height ${randCmp[1].height} ≤ ⌊log₂ 1000⌋ = 9 ${SX.flag(randCmp[1].height <= 9)}), compression only <b>${SX.f2(randCmp[2].perOp)}</b>, both <b>${SX.f2(randCmp[3].perOp)}</b> (height ${randCmp[3].height}) — ratio no-heuristic / both = ${SX.f1(randCmp[0].perOp / randCmp[3].perOp)}× · one-pass variants on the same workload with rank: path halving ${SX.f2(halving.steps / randOps.length)}, path splitting ${SX.f2(splitting.steps / randOps.length)} hops per op · the adversary for naive linking (n = 1,000: union(0,1), union(1,2), …, then 4,001 finds of element 0): no heuristic <b>${SX.f1(chainCmp[0].perOp)}</b> hops per op (a chain of height ${chainCmp[0].height}), rank only ${SX.f2(chainCmp[1].perOp)}, compression only ${SX.f2(chainCmp[2].perOp)} (one find of ${chainCmp[2].maxFindSteps} hops, then 1 each), both ${SX.f2(chainCmp[3].perOp)} · α(n) for every n here is ${SR.ackAlpha(1000)} (A₃(1) = ${SR.ackA(3, 1)}), log* 1000 = ${SR.logStar(1000)}`);
  }
  SX.on("uf-seq", "change", build); SX.on("uf-rank", "change", build); SX.on("uf-compress", "change", build);
  build();
})();

/* ── 16  #prim-svg  Prim stepped from a chosen root: keys on the frontier, extract-min, the tree growing; key/π table per extraction ── */
(function () {
  if (!SX.has("prim-svg")) return;
  const W = 680, H = 320; const G = SI.gMst; const nm = i => SR.mstEdgeName(G, i); const N = G.names;
  const B = SR.bruteSpanningTrees(G); const K = SR.kruskal(G);
  function build() {
    const root = +SX.val("prim-root", 0);
    const c = AL.counter(); const P = SR.prim(G, root, c);
    const frames = [{ init: true, key: Array.from({ length: G.n }, (_, v) => v === root ? 0 : Infinity), pi: new Array(G.n).fill(-1), inTree: new Array(G.n).fill(false), tree: [], weight: 0, v: -1, edge: -1 }].concat(P.frames);
    const svg = d3.select("#prim-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 12, r: 12, t: 10, b: 10 });
    const fmtK = k => k === Infinity ? "∞" : k;
    function render(f, k) {
      F.g.selectAll("*").remove();
      const inT = i => f.tree.indexOf(i) >= 0; const cur = f.edge;
      const frontierEdge = i => { const e = G.edges[i]; return !inT(i) && ((f.inTree[e.u] && !f.inTree[e.v] && f.pi[e.v] === e.u && G.edges[i].w === f.key[e.v]) || (f.inTree[e.v] && !f.inTree[e.u] && f.pi[e.u] === e.v && G.edges[i].w === f.key[e.u])); };
      SX.c_graph(F.g, G, { ox: 0, oy: 10, edgeColor: i => i === cur ? AC.a2 : inT(i) ? AC.good : frontierEdge(i) ? AC.accent : AC.line, edgeWidth: i => i === cur ? 4 : inT(i) ? 3.5 : frontierEdge(i) ? 2 : 1.5, edgeDash: i => frontierEdge(i) ? "4 3" : null,
        vertexFill: v => v === f.v ? AC.a2 : f.inTree[v] ? AC.accent : AC.panel2, above: v => (!f.inTree[v] && f.key[v] < Infinity) ? { text: "key " + f.key[v], color: AC.a2 } : null, weightColor: i => i === cur ? AC.a2 : inT(i) ? AC.good : AC.muted });
      F.g.append("text").attr("x", 0).attr("y", 285).attr("font-size", 11).attr("fill", AC.ink).text(f.init ? `start: key[${N[root]}] = 0, every other key ∞; all ${G.n} vertices in the min-heap` : `extract-min → ${N[f.v]} (key ${f.key[f.v]})${f.edge >= 0 ? ", add edge " + nm(f.edge) : " (the root)"}; relax its neighbours' keys`);
      F.g.append("text").attr("x", 0).attr("y", 300).attr("font-size", 11).attr("fill", AC.muted).text(`tree {${SX.c_names(G, f.tree)}} · weight so far ${f.weight} · dashed = the lightest known edge to each frontier vertex`);
      /* key / π table */
      const gx = 380; F.g.append("text").attr("x", gx).attr("y", 14).attr("font-size", 11).attr("fill", AC.muted).text("after this extraction");
      SX.grid(F.g, G.n, 3, { x: gx + 20, y: 34, w: 80, h: 26, gap: 3, rowLabel: i => N[i], colLabel: j => ["key[v]", "π[v]", "status"][j],
        text: (i, j) => j === 0 ? fmtK(f.key[i]) : j === 1 ? (f.pi[i] < 0 ? "—" : N[f.pi[i]]) : (f.inTree[i] ? "in tree" : f.key[i] < Infinity ? "frontier" : "unseen"),
        fill: (i, j) => i === f.v ? AC.a2 : (f.inTree[i] ? AC.accent : null) });
    }
    const st = AL.stepper(svg, { frames, render, delay: 800, label: "extraction" }); st.go(frames.length - 1);
    const ca = AL.counter(); const PA = SR.primArray(G, root, ca);
    const sameSet = P.tree.slice().sort().join() === K.tree.slice().sort().join();
    SX.setHtml("prim-readout", `root ${N[root]} · extraction order ${P.frames.map(f => N[f.v]).join(" ")} · tree {${SX.c_names(G, P.tree)}}, weight <b>${P.weight}</b> · measured: <b>${c.get("extract")}</b> extract-mins = V ${SX.flag(c.get("extract") === G.n)}, <b>${c.get("decrease")}</b> decrease-keys (${c.get("decreaseFromInf")} of them the first key a vertex ever gets, ${c.get("decreaseReal")} genuine improvements), ${c.get("scan")} adjacency entries scanned = 2E ${SX.flag(c.get("scan") === 2 * G.edges.length)}, ${c.get("heapswap")} heap swaps · the array version (no heap) on the same root: weight ${PA.weight} ${SX.flag(PA.weight === P.weight)}, ${ca.get("scan")} candidates scanned in its V min-searches (V(V + 1)/2 = ${G.n * (G.n + 1) / 2}) · Kruskal's weight ${K.weight} ${SX.flag(K.weight === P.weight)}, brute force over all ${B.trees.length} trees ${B.minW} ${SX.flag(B.minW === P.weight)} · Prim's edge SET equals Kruskal's ${SX.flag(sameSet)}${sameSet ? " (the MST is unique, so it must — the tie at weight 5 between BD and EF cannot matter, because BD is the heaviest edge on the cycle A–B–D and so lies in no MST)" : " — they differ on a tie; both are minimum"}`);
  }
  SX.on("prim-root", "change", build);
  build();
})();

/* ── 17  #boruvka-svg  Borůvka by round on the shared instance; toggle to the MST-versus-shortest-path-tree overlay from a root ── */
(function () {
  if (!SX.has("boruvka-svg")) return;
  const W = 680, H = 320; const G = SI.gMst; const nm = i => SR.mstEdgeName(G, i); const N = G.names;
  const B = SR.bruteSpanningTrees(G); const mst = B.msts[0];
  const S2 = SR.secondBestBySwap(G, mst);
  function build() {
    const mode = SX.val("bor-mode", "boruvka"); const root = +SX.val("bor-root", 0);
    const c = AL.counter(); const Bo = SR.boruvka(G, c);
    const M = SR.mstVsSpt(G, root, mst);
    const svg = d3.select("#boruvka-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 12, r: 12, t: 10, b: 10 });
    if (mode === "boruvka") {
      const frames = [{ init: true, comps: Array.from({ length: G.n }, (_, v) => v), chosen: {}, added: [], tree: [], weight: 0, round: 0 }].concat(Bo.frames);
      function render(f, k) {
        F.g.selectAll("*").remove();
        const reps = [...new Set(f.comps)]; const colOf = v => reps.length === 1 ? AC.good : SX.c_compColors[reps.indexOf(f.comps[v]) % SX.c_compColors.length];
        const chosen = Object.values(f.chosen); const prev = f.tree.filter(i => f.added.indexOf(i) < 0);
        SX.c_graph(F.g, G, { ox: 0, oy: 10, edgeColor: i => f.added.indexOf(i) >= 0 ? AC.a2 : prev.indexOf(i) >= 0 ? AC.good : AC.line, edgeWidth: i => f.added.indexOf(i) >= 0 ? 4 : prev.indexOf(i) >= 0 ? 3.5 : 1.5, vertexFill: v => reps.length === G.n ? AC.panel2 : colOf(v), weightColor: i => f.added.indexOf(i) >= 0 ? AC.a2 : prev.indexOf(i) >= 0 ? AC.good : AC.muted });
        F.g.append("text").attr("x", 0).attr("y", 285).attr("font-size", 11).attr("fill", AC.ink).text(f.init ? `start: ${G.n} components, each vertex alone` : `round ${f.round}: ${reps.length} components, each picks its lightest outgoing edge; ${f.added.length} distinct edge${f.added.length === 1 ? "" : "s"} added`);
        F.g.append("text").attr("x", 0).attr("y", 300).attr("font-size", 11).attr("fill", AC.muted).text(`tree {${SX.c_names(G, f.tree)}} · weight so far ${f.weight}`);
        const gx = 360; F.g.append("text").attr("x", gx).attr("y", 14).attr("font-size", 11).attr("fill", AC.muted).text(f.init ? "components before round 1" : `each component's choice in round ${f.round}`);
        const groups = {}; f.comps.forEach((r, v) => { (groups[r] = groups[r] || []).push(v); });
        Object.keys(groups).forEach((r, k2) => { const y = 26 + k2 * 40; const members = groups[r].map(v => N[v]).join(""); const ch = f.chosen[r];
          F.g.append("rect").attr("x", gx).attr("y", y).attr("width", 296).attr("height", 34).attr("rx", 4).attr("fill", AC.panel2).attr("stroke", reps.length === G.n && f.init ? AC.line : colOf(+r));
          F.g.append("text").attr("x", gx + 8).attr("y", y + 14).attr("font-size", 11).attr("fill", AC.ink).text(`{${members}}`);
          F.g.append("text").attr("x", gx + 8).attr("y", y + 28).attr("font-size", 10).attr("fill", ch === undefined ? AC.muted : AC.a2).text(ch === undefined ? (f.init ? "—" : "no outgoing edge") : `lightest outgoing: ${nm(ch)} (w = ${G.edges[ch].w})`); });
      }
      const st = AL.stepper(svg, { frames, render, delay: 1000, label: "round" }); st.go(frames.length - 1);
    } else {
      const inM = i => mst.indexOf(i) >= 0, inS = i => M.spt.tree.indexOf(i) >= 0; const td = SR.mstTreeDist(G, mst, root);
      SX.c_graph(F.g, G, { ox: 0, oy: 10, edgeColor: i => inM(i) ? (inS(i) ? AC.good : AC.a2) : AC.grid, edgeWidth: i => inM(i) ? 3.5 : 1, vertexFill: v => v === root ? AC.accent : AC.panel2, above: v => v === root ? null : { text: "path " + td.dist[v], color: td.dist[v] > M.spt.dist[v] ? AC.rose : AC.good }, weightColor: i => inM(i) ? AC.good : AC.grid });
      F.g.append("text").attr("x", 0).attr("y", 285).attr("font-size", 11).attr("fill", AC.ink).text(`the MST (weight ${M.mstWeight}); labels = distance from ${N[root]} along the tree`);
      SX.c_graph(F.g, G, { ox: 340, oy: 10, edgeColor: i => inS(i) ? (inM(i) ? AC.good : AC.a2) : AC.grid, edgeWidth: i => inS(i) ? 3.5 : 1, vertexFill: v => v === root ? AC.accent : AC.panel2, above: v => v === root ? null : { text: "δ = " + M.spt.dist[v], color: AC.accent }, weightColor: i => inS(i) ? AC.good : AC.grid });
      F.g.append("text").attr("x", 340).attr("y", 285).attr("font-size", 11).attr("fill", AC.ink).text(`the shortest-path tree from ${N[root]} (weight ${M.sptWeight}); labels = δ(${N[root]}, v)`);
      F.g.append("text").attr("x", 0).attr("y", 300).attr("font-size", 11).attr("fill", AC.muted).text(`amber = in one tree but not the other · red labels = MST path longer than the shortest path`);
    }
    const rounds = Bo.rounds, bound = Math.ceil(Math.log2(G.n));
    SX.setHtml("boruvka-readout", `Borůvka: <b>${rounds}</b> rounds ≤ ⌈log₂ V⌉ = ${bound} ${SX.flag(rounds <= bound)} (${Bo.frames.map(f => "round " + f.round + ": +" + f.added.map(nm).join(",")).join("; ")}), ${c.get("scan")} edge scans = rounds × E ${SX.flag(c.get("scan") === rounds * G.edges.length)}, tree {${SX.c_names(G, Bo.tree)}}, weight <b>${Bo.weight}</b> = brute force ${B.minW} ${SX.flag(Bo.weight === B.minW)} · second-best MST: brute force over all ${B.trees.length} trees gives <b>${B.secondBestW}</b> (${B.secondBest.length} tree${B.secondBest.length === 1 ? "" : "s"}: ${B.secondBest.map(t => "{" + SX.c_names(G, t) + "}").join(", ")}); the one-swap method over the ${S2.candidates.length} non-tree edges gives ${S2.candidates.map(cnd => "+" + nm(cnd.add) + " −" + nm(cnd.remove) + " = " + cnd.weight).join(", ")} → best <b>${S2.best.weight}</b> ${SX.flag(S2.best.weight === B.secondBestW)} · MST versus shortest-path tree from ${N[root]}: SPT {${SX.c_names(G, M.spt.tree)}} weighs <b>${M.sptWeight}</b>, the MST <b>${M.mstWeight}</b> ${M.sptWeight >= M.mstWeight ? "(SPT ≥ MST, as any spanning tree must) " + SX.flag(true) : SX.flag(false)}; same edge set: ${M.sameEdgeSet ? "yes" : "no"} · per vertex (MST-path / δ): ${M.rows.map(r => N[r.v] + " " + r.mstPath + "/" + r.dist).join(", ")} — largest excess at <b>${N[M.worst.v]}</b>: ${M.worst.mstPath} along the MST against δ = ${M.worst.dist} (stretch ${SX.f2(M.worst.stretch)})`);
  }
  SX.on("bor-mode", "change", build); SX.on("bor-root", "change", build);
  build();
})();

/* ── chunk D figures ── */
/* ── chunk D drawing helper: a small directed network with hand-placed positions.
   Not a figure block; it only defines SX.dFlowNet, used by figures 18–23. ──────── */
(function () {
  /* SX.dFlowNet(g, G, opt) draws vertices at G.pos (scaled into opt's box) and the arcs opt.arcs
     (default G.edges) as straight arrows; a pair of antiparallel arcs is bent apart. opt.label(a, i),
     opt.color(a, i), opt.width(a, i), opt.nodeFill(v), opt.nodeStroke(v), opt.nodeText(v), opt.ypos(v)
     (override the y of a vertex — the push-relabel figure uses height as y). Returns {P: [[x,y]…], r}. */
  SX.dFlowNet = function (g, G, opt) {
    const o = Object.assign({ x: 0, y: 0, w: 300, h: 240, r: 13, arcs: G.edges, label: () => "", color: () => AC.line, width: () => 1.5, nodeFill: () => AC.panel2, nodeStroke: () => AC.line, nodeText: v => (G.names ? G.names[v] : String(v)), ypos: null, fontSize: 10, textColor: () => AC.muted, dash: () => null }, opt || {});
    const xs = G.pos.map(p => p[0]), ys = G.pos.map(p => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const sx = v => o.x + o.r + 4 + (x1 === x0 ? 0 : (G.pos[v][0] - x0) / (x1 - x0)) * (o.w - 2 * o.r - 8);
    const sy = v => o.ypos ? o.ypos(v) : o.y + o.r + 4 + (y1 === y0 ? 0 : (G.pos[v][1] - y0) / (y1 - y0)) * (o.h - 2 * o.r - 8);
    const P = []; for (let v = 0; v < G.n; v++) P.push([sx(v), sy(v)]);
    const has = new Set(o.arcs.map(a => a.u + "," + a.v));
    const gg = g.append("g");
    o.arcs.forEach((a, i) => {
      const [ax, ay] = P[a.u], [bx, by] = P[a.v]; const dx = bx - ax, dy = by - ay, L = Math.hypot(dx, dy) || 1; const ux = dx / L, uy = dy / L;
      const bend = has.has(a.v + "," + a.u) ? 14 : 0; const nx = -uy * bend, ny = ux * bend;
      const x1p = ax + ux * o.r + nx * 0.5, y1p = ay + uy * o.r + ny * 0.5, x2p = bx - ux * (o.r + 2) + nx * 0.5, y2p = by - uy * (o.r + 2) + ny * 0.5;
      const mx = (ax + bx) / 2 + nx, my = (ay + by) / 2 + ny; const col = o.color(a, i), wd = o.width(a, i);
      const path = gg.append("path").attr("d", bend ? `M${x1p},${y1p} Q${mx},${my} ${x2p},${y2p}` : `M${x1p},${y1p} L${x2p},${y2p}`).attr("fill", "none").attr("stroke", col).attr("stroke-width", wd);
      const ds = o.dash(a, i); if (ds) path.attr("stroke-dasharray", ds);
      const ang = bend ? Math.atan2(y2p - my, x2p - mx) : Math.atan2(dy, dx); const h = 6;
      gg.append("path").attr("d", `M${x2p},${y2p} L${x2p - h * Math.cos(ang - 0.45)},${y2p - h * Math.sin(ang - 0.45)} L${x2p - h * Math.cos(ang + 0.45)},${y2p - h * Math.sin(ang + 0.45)} Z`).attr("fill", col);
      const t = o.label(a, i); if (t !== "" && t !== null && t !== undefined) { const lx = bend ? (ax + bx) / 2 + nx * 1.3 : (ax + bx) / 2 - uy * 9, ly = bend ? (ay + by) / 2 + ny * 1.3 : (ay + by) / 2 + ux * 9; gg.append("text").attr("x", lx).attr("y", ly + 3.5).attr("text-anchor", "middle").attr("font-size", o.fontSize).attr("fill", o.textColor(a, i)).text(t); }
    });
    for (let v = 0; v < G.n; v++) { gg.append("circle").attr("cx", P[v][0]).attr("cy", P[v][1]).attr("r", o.r).attr("fill", o.nodeFill(v)).attr("stroke", o.nodeStroke(v)).attr("stroke-width", 1.5);
      gg.append("text").attr("x", P[v][0]).attr("y", P[v][1] + 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.ink).text(o.nodeText(v)); }
    return { P, r: o.r, g: gg };
  };
})();

/* ── 18  #residual-svg  the six-vertex network, a flow, its residual graph, and a cut ── */
(function () {
  if (!SX.has("residual-svg")) return;
  const W = 680, H = 300; const G = SI.gFlow; const n = G.n;
  const cutSets = { s: [0], s12: [0, 1, 2], s124: [0, 1, 2, 4], s1234: [0, 1, 2, 3, 4] };
  function build() {
    const which = SX.val("res-flow", "partial"), cutKey = SX.val("res-cut", "s124");
    const ek = SR.ekMaxflow(G);
    const f = which === "zero" ? G.edges.map(() => 0) : which === "max" ? ek.f : [11, 8, 1, 12, 4, 11, 7, 15, 4];
    const chk = SR.flowCheck(G, f); const S = new Array(n).fill(false); cutSets[cutKey].forEach(v => { S[v] = true; });
    const cut = SR.flowCut(G, f, S); const cb = AL.counter(); const B = SR.bruteMinCut(G, cb); const resid = SR.flowResidual(G, f);
    const svg = d3.select("#residual-svg"); const F = AL.frame(svg, W, H, { l: 10, r: 10, t: 22, b: 30 });
    F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text(`G with f/c  ·  |f| = ${chk.value}  ·  cut S = {${cutSets[cutKey].map(v => G.names[v]).join(", ")}}, c(S,T) = ${cut.cap}`);
    F.g.append("text").attr("x", 350).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text(`residual graph G_f  (${resid.length} arcs with c_f > 0)`);
    const inS = v => S[v];
    const crosses = a => inS(a.u) && !inS(a.v);
    SX.dFlowNet(F.g, G, { x: 0, y: 0, w: 320, h: F.ih, label: (e, i) => `${f[i]}/${e.cap}`, color: (e, i) => crosses(e) ? AC.rose : f[i] === e.cap ? AC.accent : AC.line, width: (e, i) => crosses(e) ? 2.5 : f[i] > 0 ? 2 : 1.2, textColor: (e, i) => crosses(e) ? AC.rose : f[i] > 0 ? AC.ink : AC.muted, nodeFill: v => inS(v) ? AC.panel2 : AC.grid, nodeStroke: v => inS(v) ? AC.a2 : AC.teal });
    const Gr = Object.assign({}, G, { edges: resid });
    SX.dFlowNet(F.g, Gr, { x: 350, y: 0, w: 320, h: F.ih, arcs: resid, label: a => String(a.c), color: a => a.fwd ? AC.accent : AC.violet, width: () => 1.5, textColor: a => a.fwd ? AC.accent : AC.violet, nodeFill: v => inS(v) ? AC.panel2 : AC.grid, nodeStroke: v => inS(v) ? AC.a2 : AC.teal });
    AL.legend(F.g, [{ label: "edge crossing the cut S → T", color: AC.rose }], 0, F.ih + 16);
    AL.legend(F.g, [{ label: "forward arc: c − f", color: AC.accent }], 350, F.ih + 16);
    AL.legend(F.g, [{ label: "reverse arc: cancels f", color: AC.violet }], 500, F.ih + 16);
    const cons = []; for (let v = 0; v < n; v++) if (v !== G.s && v !== G.t) cons.push(`${G.names[v]}: in ${chk.inflow[v]} = out ${chk.outflow[v]}`);
    SX.setHtml("residual-readout", `flow ${which === "zero" ? "f = 0" : which === "max" ? "f* from Edmonds-Karp" : "the partial flow"}: capacity constraint ${SX.flag(chk.capOk)}, conservation at every internal vertex — ${cons.join(", ")} ${SX.flag(chk.consOk)} · |f| = out(s) − in(s) = <b>${chk.value}</b> · cut S = {${cutSets[cutKey].map(v => G.names[v]).join(", ")}}: net flow f(S,T) = ${cut.net} = |f| ${SX.flag(cut.net === chk.value)}, capacity c(S,T) = ${cut.cap}, |f| ≤ c(S,T) ${SX.flag(chk.value <= cut.cap)} · brute force over all ${cb.get("cut")} = 2^(V−2) s-t cuts: minimum capacity <b>${B.cap}</b> at S = {${B.S.map((x, v) => x ? G.names[v] : null).filter(x => x).join(", ")}}${B.ties > 1 ? " (" + B.ties + " minimisers)" : " (unique)"} · residual arcs ${resid.length} ≤ 2E = ${2 * G.edges.length} ${SX.flag(resid.length <= 2 * G.edges.length)} · max flow by Edmonds-Karp ${ek.value} = min cut ${SX.flag(ek.value === B.cap)}`);
  }
  SX.on("res-flow", "change", build); SX.on("res-cut", "change", build);
  build();
})();

/* ── 19  #ff-svg  Ford-Fulkerson stepped: path rule, residual graph, the min cut lit ── */
(function () {
  if (!SX.has("ff-svg")) return;
  const W = 680, H = 300;
  function build() {
    const inst = SX.val("ff-inst", "six"), rule = SX.val("ff-rule", "dfs");
    const G = inst === "six" ? SI.gFlow : SI.gFlowBad; const n = G.n;
    const frames = [{ k: 0, path: [], arcs: [], bottleneck: 0, value: 0, f: G.edges.map(() => 0) }];
    const c = AL.counter(); const R = SR.ffMaxflow(G, rule, c, r => frames.push(r));
    const all = {}; ["dfs", "worst", "bfs"].forEach(rl => { const cc = AL.counter(); const rr = SR.ffMaxflow(G, rl, cc); all[rl] = { augs: rr.augs, scans: cc.get("scan"), value: rr.value }; });
    const cb = AL.counter(); const B = SR.bruteMinCut(G, cb);
    frames.push({ k: -1, path: [], arcs: [], bottleneck: 0, value: R.value, f: R.f, final: true });
    const svg = d3.select("#ff-svg"); SX.clearControls(svg.node()); const F = AL.frame(svg, W, H, { l: 10, r: 10, t: 22, b: 30 });
    function render(fr, i) {
      F.g.selectAll("*").remove(); const f = fr.f; const onPath = new Set(fr.arcs.map(a => Math.floor(a / 2)));
      const inS = v => fr.final && R.S[v];
      F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text(fr.final ? `|f*| = ${fr.value} · min cut S = {${R.S.map((x, v) => x ? G.names[v] : null).filter(x => x).join(", ")}} lit (reachable in G_f)` : fr.k === 0 ? "f = 0 (every edge 0/c)" : `augmentation ${fr.k}: path ${fr.path.map(v => G.names[v]).join(" → ")}, residual capacity ${fr.bottleneck}, |f| = ${fr.value}`);
      F.g.append("text").attr("x", 350).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text(fr.final ? "residual graph G_f*: no s → t path" : "residual graph G_f after this augmentation");
      SX.dFlowNet(F.g, G, { x: 0, y: 0, w: 320, h: F.ih, label: (e, i) => `${f[i]}/${e.cap}`, color: (e, i) => onPath.has(i) ? AC.a2 : (fr.final && R.S[e.u] && !R.S[e.v]) ? AC.rose : f[i] === e.cap ? AC.accent : AC.line, width: (e, i) => onPath.has(i) || (fr.final && R.S[e.u] && !R.S[e.v]) ? 2.5 : f[i] > 0 ? 2 : 1.2, textColor: (e, i) => onPath.has(i) ? AC.a2 : f[i] > 0 ? AC.ink : AC.muted, nodeFill: v => fr.path.includes(v) ? AC.a2 : inS(v) ? AC.panel2 : fr.final ? AC.grid : AC.panel2, nodeStroke: v => fr.final ? (R.S[v] ? AC.a2 : AC.teal) : AC.line });
      const resid = SR.flowResidual(G, f);
      SX.dFlowNet(F.g, Object.assign({}, G, { edges: resid }), { x: 350, y: 0, w: 320, h: F.ih, arcs: resid, label: a => String(a.c), color: a => a.fwd ? AC.accent : AC.violet, textColor: a => a.fwd ? AC.accent : AC.violet, nodeFill: () => AC.panel2, nodeStroke: v => fr.final ? (R.S[v] ? AC.a2 : AC.teal) : AC.line });
      F.g.append("text").attr("x", 0).attr("y", F.ih + 22).attr("font-size", 11).attr("fill", AC.ink).text(`rule: ${rule === "dfs" ? "depth-first, adjacency order" : rule === "bfs" ? "shortest path (BFS)" : "worst: prefer the path through the smallest-capacity edge"} · augmentations so far ${fr.final ? R.augs : fr.k}`);
    }
    const st = AL.stepper(svg, { frames, render, delay: 650, label: "step" }); st.go(frames.length - 1);
    SX.setHtml("ff-readout", `V = ${n}, E = ${G.edges.length}${inst === "bad" ? ", M = 100" : ""} · this run (${rule}): <b>${R.augs}</b> augmentations, ${c.get("scan")} arc scans, |f*| = <b>${R.value}</b> · all three rules measured — depth-first: ${all.dfs.augs} augmentations / ${all.dfs.scans} scans, worst: ${all.worst.augs} / ${all.worst.scans}, BFS: ${all.bfs.augs} / ${all.bfs.scans}; all reach |f| = ${all.dfs.value} ${SX.flag(all.dfs.value === all.bfs.value && all.worst.value === all.bfs.value)} · integer bound: ≤ |f*| = ${R.value} augmentations, each O(E)${inst === "bad" ? "; the worst rule attains it — 2M = " + 2 * 100 + " augmentations " + SX.flag(all.worst.augs === 200) : ""} · brute force over ${cb.get("cut")} s-t cuts: min cut ${B.cap}${B.ties > 1 ? " (" + B.ties + " minimisers)" : ""} = |f*| ${SX.flag(B.cap === R.value)}; the lit S = {${R.S.map((x, v) => x ? G.names[v] : null).filter(x => x).join(", ")}} has capacity ${SR.flowCut(G, null, R.S).cap} ${SX.flag(SR.flowCut(G, null, R.S).cap === R.value)}`);
  }
  SX.on("ff-inst", "change", build); SX.on("ff-rule", "change", build);
  build();
})();

/* ── 20  #ek-svg  Edmonds-Karp stepped: shortest path, BFS layers, critical arc, the bounds ── */
(function () {
  if (!SX.has("ek-svg")) return;
  const W = 680, H = 320;
  function build() {
    const inst = SX.val("ek-inst", "six");
    const G = inst === "six" ? SI.gFlow : SR.flowRandom(+inst.slice(1), 8, 16, AL.rng); const n = G.n, E = G.edges.length;
    const frames = []; const c = AL.counter(); const R = SR.ekMaxflow(G, c, r => frames.push(r));
    /* the final BFS (no path) — its layers for the last frame, and the reachable set */
    const finalDist = (function () { const F = R.F, d = new Array(n).fill(-1); d[G.s] = 0; const q = [G.s]; while (q.length) { const u = q.shift(); F.adj[u].forEach(a => { const A = F.arcs[a]; if (A.cap - A.flow > 0 && d[A.to] < 0) { d[A.to] = d[u] + 1; q.push(A.to); } }); } return d; })();
    const flows = [G.edges.map(() => 0)].concat(frames.map(fr => fr.f));
    const all = [{ k: 0, init: true, dist: frames.length ? frames[0].dist : finalDist, f: flows[0], path: [], arcs: [], critical: [], value: 0 }];
    frames.forEach((fr, i) => all.push(Object.assign({}, fr, { fBefore: flows[i] })));
    all.push({ k: frames.length + 1, final: true, dist: finalDist, f: R.f, path: [], arcs: [], critical: [], value: R.value });
    const cb = AL.counter(); const B = SR.bruteMinCut(G, cb); const cd = AL.counter(); const D = SR.dinicMaxflow(G, cd);
    const maxCrit = Math.max(0, ...R.critCount.map(x => Math.max(x[0], x[1])));
    const svg = d3.select("#ek-svg"); SX.clearControls(svg.node()); const F = AL.frame(svg, W, H, { l: 10, r: 10, t: 22, b: 30 });
    function render(fr) {
      F.g.selectAll("*").remove(); const f = fr.f; const onPath = new Set(fr.arcs.map(a => a >> 1)); const crit = new Set(fr.critical.map(a => a >> 1));
      F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text(fr.init ? "f = 0; BFS layers of G_f = G" : fr.final ? `no augmenting path: t unreachable · |f*| = ${fr.value} · S = {${finalDist.map((d, v) => d >= 0 ? G.names[v] : null).filter(x => x).join(", ")}}` : `augmentation ${fr.k}: ${fr.path.map(v => G.names[v]).join(" → ")} (length ${fr.arcs.length}), bottleneck ${fr.bottleneck}, |f| = ${fr.value}`);
      SX.dFlowNet(F.g, G, { x: 0, y: 0, w: 330, h: F.ih, label: (e, i) => `${f[i]}/${e.cap}`, color: (e, i) => crit.has(i) ? AC.rose : onPath.has(i) ? AC.a2 : f[i] === e.cap ? AC.accent : AC.line, width: (e, i) => onPath.has(i) ? 2.5 : f[i] > 0 ? 2 : 1.2, textColor: (e, i) => crit.has(i) ? AC.rose : onPath.has(i) ? AC.a2 : f[i] > 0 ? AC.ink : AC.muted, nodeFill: v => fr.path.includes(v) ? AC.a2 : AC.panel2, nodeStroke: v => fr.final ? (finalDist[v] >= 0 ? AC.a2 : AC.teal) : AC.line, fontSize: n > 6 ? 9 : 10 });
      /* layered view: columns by δ_f(s, ·) BEFORE this augmentation */
      const dist = fr.dist; const maxD = Math.max(1, ...dist.filter(d => d >= 0)); const cols = maxD + 2; const cw = 320 / cols; const byCol = {}; dist.forEach((d, v) => { const k = d < 0 ? maxD + 1 : d; (byCol[k] = byCol[k] || []).push(v); });
      const P = {}; Object.keys(byCol).forEach(k => { const vs = byCol[k]; vs.forEach((v, i) => { P[v] = [360 + cw * (+k + 0.5), 20 + (F.ih - 40) * (vs.length === 1 ? 0.5 : i / (vs.length - 1))]; }); });
      for (let k = 0; k <= maxD + 1; k++) F.g.append("text").attr("x", 360 + cw * (k + 0.5)).attr("y", F.ih + 12).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.muted).text(k <= maxD ? "δ = " + k : "∞");
      const resid = SR.flowResidual(G, fr.init || fr.final ? f : fr.fBefore);
      const rkeys = new Set(resid.map(a => a.u + "," + a.v));
      resid.forEach(a => { if (dist[a.u] < 0 || dist[a.v] < 0) return; const lvl = dist[a.v] === dist[a.u] + 1; const [ax, ay] = P[a.u], [bx, by] = P[a.v]; const on = fr.arcs.some(x => { const A = R.F.arcs[x]; return A.from === a.u && A.to === a.v; }); const L = Math.hypot(bx - ax, by - ay) || 1, ux = (bx - ax) / L, uy = (by - ay) / L; const off = rkeys.has(a.v + "," + a.u) ? 5 : 0; const ox = -uy * off, oy = ux * off; AL.arrow(F.g, ax + ux * 13 + ox, ay + uy * 13 + oy, bx - ux * 15 + ox, by - uy * 15 + oy, { color: on ? AC.a2 : lvl ? AC.accent : AC.grid, w: on ? 2.5 : lvl ? 1.2 : 0.8, dash: lvl ? null : "3,3" }); });
      for (let v = 0; v < n; v++) { const [x, y] = P[v]; F.g.append("circle").attr("cx", x).attr("cy", y).attr("r", 12).attr("fill", dist[v] < 0 ? AC.grid : fr.path.includes(v) ? AC.a2 : AC.panel2).attr("stroke", dist[v] < 0 ? AC.line : AC.accent); F.g.append("text").attr("x", x).attr("y", y + 4).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", dist[v] < 0 ? AC.muted : AC.ink).text(G.names[v]); }
      F.g.append("text").attr("x", 360).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text("BFS layers of G_f before this step (blue = level-graph arcs)");
    }
    const st = AL.stepper(svg, { frames: all, render, delay: 700, label: "step" }); st.go(all.length - 1);
    const critList = R.critCount.map((x, i) => { const e = G.edges[i]; const parts = []; if (x[0]) parts.push(`${G.names[e.u]}→${G.names[e.v]} ×${x[0]}`); if (x[1]) parts.push(`${G.names[e.v]}→${G.names[e.u]} (reverse) ×${x[1]}`); return parts.join(", "); }).filter(x => x);
    SX.setHtml("ek-readout", `V = ${n}, E = ${E} · augmentations <b>${R.augs}</b> ≤ V·E = ${n * E} ${SX.flag(R.augs <= n * E)}, path lengths ${R.lens.join(", ")} (non-decreasing ${SX.flag(R.lens.every((l, i) => i === 0 || l >= R.lens[i - 1]))}) · BFS runs ${c.get("bfs")}, vertex dequeues ${c.get("visit")}, arc scans ${c.get("scan")} · critical arcs: ${critList.join("; ") || "none"}; max per arc ${maxCrit} ≤ V/2 = ${n / 2} ${SX.flag(maxCrit <= n / 2)} · |f*| = <b>${R.value}</b> = Dinic ${D.value} (${D.phases} phases ≤ V − 1 = ${n - 1} ${SX.flag(D.phases <= n - 1)}) ${SX.flag(D.value === R.value)} = brute-force min cut over ${cb.get("cut")} cuts: ${B.cap}${B.ties > 1 ? " (" + B.ties + " minimisers)" : ""} ${SX.flag(B.cap === R.value)}`);
  }
  SX.on("ek-inst", "change", build);
  build();
})();

/* ── 21  #pr-svg  push-relabel stepped: heights as y, excess as fill, pushes and relabels ── */
(function () {
  if (!SX.has("pr-svg")) return;
  const W = 680, H = 360;
  function build() {
    const inst = SX.val("pr-inst", "five"); const G0 = inst === "five" ? SI.gFlowPR : SI.gFlow; const n = G0.n, E = G0.edges.length;
    const G = Object.assign({}, G0, { pos: G0.pos.map((p, v) => [v * 100, p[1]]) });   /* x by vertex index so no two vertices share a column; y is the height */
    const frames = []; const c = AL.counter(); const R = SR.prMaxflow(G, c, r => frames.push(r));
    const ek = SR.ekMaxflow(G); const cb = AL.counter(); const B = SR.bruteMinCut(G, cb);
    const maxH = Math.max(2 * n - 1, ...R.h); const maxE = Math.max(1, ...frames.map(fr => Math.max(...fr.e.filter((x, v) => v !== G.s))));
    const svg = d3.select("#pr-svg"); SX.clearControls(svg.node()); const F = AL.frame(svg, W, H, { l: 46, r: 16, t: 22, b: 34 });
    const yOf = h => F.ih - 10 - (F.ih - 30) * h / maxH;
    function render(fr, i) {
      F.g.selectAll("*").remove();
      for (let h = 0; h <= maxH; h += (maxH > 12 ? 2 : 1)) { F.g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", yOf(h)).attr("y2", yOf(h)).attr("stroke", AC.grid); F.g.append("text").attr("x", -6).attr("y", yOf(h) + 3).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.muted).text("h = " + h); }
      F.g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", yOf(n)).attr("y2", yOf(n)).attr("stroke", AC.a2).attr("stroke-dasharray", "4,4").attr("opacity", 0.6);
      const pushArc = fr.op.startsWith("push") ? fr : null;
      const isOn = (e, i) => pushArc && ((e.u === fr.u && e.v === fr.v) || (e.u === fr.v && e.v === fr.u));
      const shade = v => { if (v === G.s || v === G.t) return AC.panel2; const x = fr.e[v] / maxE; return x <= 0 ? AC.panel2 : d3.interpolateRgb(AC.panel2, AC.accent)(0.25 + 0.75 * x); };
      const net = SX.dFlowNet(F.g, G, { x: 0, y: 0, w: F.iw, h: F.ih, ypos: v => yOf(fr.h[v]), label: (e, i) => `${fr.f[i]}/${e.cap}`, color: (e, i) => isOn(e, i) ? AC.a2 : fr.f[i] === e.cap ? AC.accent : AC.line, width: (e, i) => isOn(e, i) ? 2.5 : fr.f[i] > 0 ? 2 : 1.2, textColor: (e, i) => isOn(e, i) ? AC.a2 : fr.f[i] > 0 ? AC.ink : AC.muted, nodeFill: shade, nodeStroke: v => (fr.op === "relabel" && v === fr.u) ? AC.a2 : AC.line, r: 14 });
      /* excess labels beside the vertices that hold excess */
      for (let v = 0; v < n; v++) if (v !== G.s && fr.e[v] > 0) F.g.append("text").attr("x", net.P[v][0]).attr("y", net.P[v][1] - 20).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", v === G.t ? AC.good : AC.a2).text("e = " + fr.e[v]);
      const nm = v => G.names[v];
      F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.ink).text(fr.op === "init" ? `initialise: h(s) = V = ${n}, every other height 0` : fr.op === "relabel" ? `#${i}: relabel ${nm(fr.u)} → ${fr.amt}  (1 + lowest residual neighbour)` : `#${i}: ${fr.op === "push·sat" ? "saturating" : "non-saturating"} push ${nm(fr.u)} → ${nm(fr.v)} of ${fr.amt}  (h(${nm(fr.u)}) = ${fr.h[fr.u]} = h(${nm(fr.v)}) + 1)`);
      F.g.append("text").attr("x", F.iw).attr("y", F.ih + 26).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.muted).text(`excess: ${G.names.map((s, v) => v === G.s ? null : s + " " + fr.e[v]).filter(x => x).join(" · ")}`);
    }
    const st = AL.stepper(svg, { frames, render, delay: 700, label: "operation" }); st.go(frames.length - 1);
    const relB = (2 * n - 1) * (n - 2), satB = 2 * n * E, nonB = 4 * n * n * (n + E);
    /* the cut the final residual graph defines: source side = vertices that cannot reach t */
    const canReachT = (function () { const F2 = R.F, seen = new Array(n).fill(false); seen[G.t] = true; const q = [G.t]; while (q.length) { const v = q.shift(); F2.adj[v].forEach(a => { const A = F2.arcs[a ^ 1]; if (A.cap - A.flow > 0 && !seen[A.from]) { seen[A.from] = true; q.push(A.from); } }); } return seen; })();
    const Sstar = canReachT.map(x => !x); const cutStar = SR.flowCut(G, null, Sstar).cap;
    SX.setHtml("pr-readout", `V = ${n}, E = ${E} · pushes: saturating <b>${c.get("pushSat")}</b> &lt; 2VE = ${satB} ${SX.flag(c.get("pushSat") < satB)}, non-saturating <b>${c.get("pushNon")}</b> &lt; 4V²(V + E) = ${SX.int(nonB)} ${SX.flag(c.get("pushNon") < nonB)} · relabels <b>${c.get("relabel")}</b> ≤ (2V − 1)(V − 2) = ${relB} ${SX.flag(c.get("relabel") <= relB)} · max height reached ${Math.max(...R.h)} ≤ 2V − 1 = ${2 * n - 1} ${SX.flag(Math.max(...R.h) <= 2 * n - 1)}; final h = ⟨${R.h.join(", ")}⟩ · |f| = e(t) = <b>${R.value}</b> = Edmonds-Karp ${ek.value} ${SX.flag(ek.value === R.value)} = brute-force min cut over ${cb.get("cut")} cuts: ${B.cap} ${SX.flag(B.cap === R.value)} (${B.ties === 1 ? "unique minimiser, so the smallest and largest minimum cuts coincide" : B.ties + " minimisers"}) · the cut read from the final residual graph (vertices that cannot reach t): S = {${Sstar.map((x, v) => x ? G.names[v] : null).filter(x => x).join(", ")}}, capacity ${cutStar} ${SX.flag(cutStar === R.value)}; vertices at height ≥ V: {${R.h.map((h, v) => h >= n ? G.names[v] : null).filter(x => x).join(", ")}}`);
  }
  SX.on("pr-inst", "change", build);
  build();
})();

/* ── 22  #match-svg  bipartite matching stepped: alternating paths, the matching, König's cover, Hall's violator ── */
(function () {
  if (!SX.has("match-svg")) return;
  const W = 680, H = 300;
  function build() {
    const inst = SX.val("match-inst", "perfect"); const B = inst === "perfect" ? SI.gBip : SI.gBipHall;
    const E = []; for (let l = 0; l < B.nL; l++) B.adj[l].forEach(r => E.push([l, r]));
    const frames = [{ k: 0, path: [], matchL: new Array(B.nL).fill(-1), matchR: new Array(B.nR).fill(-1) }];
    const c = AL.counter(); const M = SR.bipartiteMatch(B, c, r => frames.push(r));
    const K = SR.konigCover(B, M); const bm = SR.bruteMatching(B); const bc = SR.bruteVertexCover(B); const Hl = SR.hallViolator(B);
    const ch = AL.counter(); const HK = SR.hopcroftKarp(B, ch); const Gf = SR.bipToFlow(B); const cf = AL.counter(); const Fl = SR.ekMaxflow(Gf, cf);
    const NS = new Set(); if (Hl.S) Hl.S.forEach(l => B.adj[l].forEach(r => NS.add(r)));
    frames.push({ k: frames.length, final: true, path: [], matchL: M.matchL, matchR: M.matchR });
    const svg = d3.select("#match-svg"); SX.clearControls(svg.node()); const F = AL.frame(svg, W, H, { l: 14, r: 10, t: 22, b: 26 });
    const xL = 40, xR = 240, yOf = i => 24 + i * (F.ih - 48) / Math.max(1, B.nL - 1);
    function render(fr) {
      F.g.selectAll("*").remove();
      const inPath = (l, r) => { for (let i = 0; i + 1 < fr.path.length; i++) { const a = fr.path[i], b = fr.path[i + 1]; if (i % 2 === 0 ? (a === l && b === r) : (a === r && b === l)) return i % 2 === 0 ? "add" : "undo"; } return null; };
      F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text(fr.final ? `maximum matching |M| = ${M.size}; König cover C = {${K.coverL.map(l => "l" + l).concat(K.coverR.map(r => "r" + r)).join(", ")}}${Hl.S ? "; Hall violator S = {" + Hl.S.map(l => "l" + l).join(", ") + "}, N(S) = {" + [...NS].sort().map(r => "r" + r).join(", ") + "}" : ""}` : fr.k === 0 ? "M = ∅" : `augmentation ${fr.k} from l${fr.from}: ${fr.path.map((v, i) => (i % 2 ? "r" : "l") + v).join(" – ")} (length ${fr.path.length - 1}), |M| = ${fr.k}`);
      E.forEach(([l, r]) => { const st = inPath(l, r); const matched = fr.matchL[l] === r; const col = st === "add" ? AC.a2 : st === "undo" ? AC.a2 : matched ? AC.accent : AC.line; const ln = F.g.append("line").attr("x1", xL + 14).attr("y1", yOf(l)).attr("x2", xR - 14).attr("y2", yOf(r)).attr("stroke", col).attr("stroke-width", st || matched ? 3 : 1.2); if (st === "undo") ln.attr("stroke-dasharray", "5,4"); });
      for (let l = 0; l < B.nL; l++) { const cov = fr.final && K.coverL.includes(l), hall = fr.final && Hl.S && Hl.S.includes(l); F.g.append("circle").attr("cx", xL).attr("cy", yOf(l)).attr("r", 13).attr("fill", hall ? AC.a2 : fr.path[0] === l && !fr.final ? AC.a2 : AC.panel2).attr("stroke", cov ? AC.rose : AC.line).attr("stroke-width", cov ? 3 : 1.5); F.g.append("text").attr("x", xL).attr("y", yOf(l) + 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", hall ? AC.bg : AC.ink).text("l" + l); }
      for (let r = 0; r < B.nR; r++) { const cov = fr.final && K.coverR.includes(r), hall = fr.final && Hl.S && NS.has(r); F.g.append("circle").attr("cx", xR).attr("cy", yOf(r)).attr("r", 13).attr("fill", hall ? AC.a2 : AC.panel2).attr("stroke", cov ? AC.rose : AC.line).attr("stroke-width", cov ? 3 : 1.5); F.g.append("text").attr("x", xR).attr("y", yOf(r) + 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", hall ? AC.bg : AC.ink).text("r" + r); }
      /* right: the flow network G′ with the flow of the current matching */
      const Gn = Object.assign({}, Gf, { names: [].concat(Array.from({ length: B.nL }, (_, l) => "l" + l), Array.from({ length: B.nR }, (_, r) => "r" + r), ["s", "t"]), pos: [].concat(Array.from({ length: B.nL }, (_, l) => [100, 20 + l * 60]), Array.from({ length: B.nR }, (_, r) => [220, 20 + r * 60]), [[0, 110], [320, 110]]) });
      const fOf = e => e.u === Gf.s ? (fr.matchL[e.v] >= 0 ? 1 : 0) : e.v === Gf.t ? (fr.matchR[e.u - B.nL] >= 0 ? 1 : 0) : (fr.matchL[e.u] === e.v - B.nL ? 1 : 0);
      F.g.append("text").attr("x", 330).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text(`G′: unit capacities, |f| = ${fr.matchL.filter(x => x >= 0).length}`);
      SX.dFlowNet(F.g, Gn, { x: 330, y: 0, w: 330, h: F.ih, r: 11, label: e => fOf(e) ? "1" : "", color: e => fOf(e) ? AC.accent : AC.line, width: e => fOf(e) ? 2.2 : 1, textColor: () => AC.accent, nodeFill: () => AC.panel2, nodeStroke: v => fr.final && ((v < B.nL && K.coverL.includes(v)) || (v >= B.nL && v < B.nL + B.nR && K.coverR.includes(v - B.nL))) ? AC.rose : AC.line, fontSize: 9 });
    }
    const st = AL.stepper(svg, { frames, render, delay: 800, label: "step" }); st.go(frames.length - 1);
    const coverOk = SR.isVertexCover(B, Array.from({ length: B.nL }, (_, l) => K.coverL.includes(l)), Array.from({ length: B.nR }, (_, r) => K.coverR.includes(r)));
    SX.setHtml("match-readout", `|L| = |R| = 4, ${E.length} edges · augmenting-path routine: |M| = <b>${M.size}</b> in ${c.get("aug")} augmentations (${c.get("scan")} edge scans), M = {${M.matchL.map((r, l) => r >= 0 ? "l" + l + "r" + r : null).filter(x => x).join(", ")}} · flow on G′ (V′ = ${Gf.n}, E′ = ${Gf.edges.length}): |f| = ${Fl.value} in ${Fl.augs} augmentations ${SX.flag(Fl.value === M.size)} · brute force over all ${bm.count} matchings: max ${bm.size} ${SX.flag(bm.size === M.size)} · Hopcroft-Karp: ${HK.size} in ${HK.phases} phase${HK.phases === 1 ? "" : "s"} ${SX.flag(HK.size === M.size)} · König cover {${K.coverL.map(l => "l" + l).concat(K.coverR.map(r => "r" + r)).join(", ")}} of size ${K.size}: covers every edge ${SX.flag(coverOk)}, = |M| ${SX.flag(K.size === M.size)}, = brute-force minimum over ${bc.count} vertex subsets: ${bc.size} ${SX.flag(bc.size === K.size)} · Hall: largest deficiency |S| − |N(S)| over ${Hl.count} subsets = ${Hl.deficiency}${Hl.S ? " at S = {" + Hl.S.map(l => "l" + l).join(", ") + "}, N(S) = {" + [...NS].sort().map(r => "r" + r).join(", ") + "}" : " — condition holds"}; |L| − deficiency = ${B.nL - Hl.deficiency} = |M| ${SX.flag(B.nL - Hl.deficiency === M.size)}`);
  }
  SX.on("match-inst", "change", build);
  build();
})();

/* ── 23  #seg-svg  min-cut segmentation of the 3 × 3 grid across λ, checked against all 512 labellings ── */
(function () {
  if (!SX.has("seg-svg")) return;
  const W = 680, H = 320; const I = SI.segI, Rn = I.length, Cn = I[0].length, N = Rn * Cn;
  /* the sweep over λ = 0..8 (done once; each point is a full min cut + brute force) */
  const sweep = []; let prevLab = null;
  for (let lam = 0; lam <= 8; lam++) { const g = SR.segBuild(I, SI.segMuF, SI.segMuB, lam); const c = AL.counter(); const M = SR.segMinCut(g, c); const Bf = SR.segBrute(g); const flips = prevLab ? M.label.filter((x, p) => x !== prevLab[p]).length : 0; sweep.push({ lam, g, M, Bf, flips, augs: c.get("aug"), scans: c.get("scan") }); prevLab = M.label; }
  /* the other reductions of §26, re-derived once */
  const G6 = SI.gFlow; const ed = SR.ekMaxflow(SR.unitCapacity(G6)); const edB = SR.bruteDisjointPaths(G6, false); const vd = SR.ekMaxflow(SR.vertexSplit(G6)); const vdB = SR.bruteDisjointPaths(G6, true);
  const cp = AL.counter(); const PJ = SR.projectSelection(SI.projProfit, SI.projPrereq, cp); const PJB = SR.bruteProjects(SI.projProfit, SI.projPrereq); const pn = ["A", "B", "C", "D"];
  function build() {
    const lam = +SX.val("seg-lambda", 3); const S = sweep[lam]; const g = S.g, lab = S.M.label;
    const svg = d3.select("#seg-svg"); const F = AL.frame(svg, W, H, { l: 10, r: 14, t: 22, b: 30 });
    /* left: the pixel grid with the cut drawn between differently labelled neighbours */
    const cell = 52, gx = 6, gy = 30;
    F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text(`λ = ${lam}: energy ${S.M.energy}; blue = foreground`);
    for (let i = 0; i < Rn; i++) for (let j = 0; j < Cn; j++) { const p = i * Cn + j; F.g.append("rect").attr("x", gx + j * cell).attr("y", gy + i * cell).attr("width", cell - 3).attr("height", cell - 3).attr("rx", 4).attr("fill", lab[p] ? AC.accent : AC.panel2).attr("stroke", AC.line); F.g.append("text").attr("x", gx + j * cell + (cell - 3) / 2).attr("y", gy + i * cell + (cell - 3) / 2 + 5).attr("text-anchor", "middle").attr("font-size", 14).attr("fill", lab[p] ? AC.bg : AC.ink).text(I[i][j]); F.g.append("text").attr("x", gx + j * cell + 4).attr("y", gy + i * cell + 11).attr("font-size", 8).attr("fill", lab[p] ? AC.bg : AC.muted).text("p" + p); }
    for (let i = 0; i < Rn; i++) for (let j = 0; j < Cn; j++) { const p = i * Cn + j; if (j + 1 < Cn && lab[p] !== lab[p + 1]) F.g.append("rect").attr("x", gx + (j + 1) * cell - 4.5).attr("y", gy + i * cell).attr("width", 3).attr("height", cell - 3).attr("fill", AC.rose); if (i + 1 < Rn && lab[p] !== lab[p + Cn]) F.g.append("rect").attr("x", gx + j * cell).attr("y", gy + (i + 1) * cell - 4.5).attr("width", cell - 3).attr("height", 3).attr("fill", AC.rose); }
    F.g.append("text").attr("x", gx).attr("y", gy + Rn * cell + 14).attr("font-size", 10).attr("fill", AC.muted).text("rose bars: the cut (boundary pairs, λ each)");
    /* middle: the network, s above and t below the pixel nodes */
    const mx = 190, mw = 250, ny = 40, nh = F.ih - 60; const px = (p) => mx + 40 + (p % Cn) * (mw - 80) / (Cn - 1), py = (p) => ny + 30 + Math.floor(p / Cn) * (nh - 60) / (Rn - 1);
    F.g.append("text").attr("x", mx).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text(`network: V = ${g.n}, E = ${g.edges.length}; nodes show D₀/D₁`);
    const sx = mx + mw / 2, sy = ny - 12, tx = mx + mw / 2, ty = ny + nh + 8;
    /* terminal edges as stubs: from s into the top of each pixel (cap D₀, severed when the pixel is background), from the bottom of each pixel towards t (cap D₁, severed when foreground) */
    g.edges.forEach(e => { if (e.u === g.s) { const p = e.v; const cut = lab[p] === 0; AL.arrow(F.g, px(p) - 9, py(p) - 32, px(p) - 9, py(p) - 10, { color: cut ? AC.rose : AC.line, w: cut ? 2.2 : 1 }); } else if (e.v === g.t) { const p = e.u; const cut = lab[p] === 1; AL.arrow(F.g, px(p) + 9, py(p) + 10, px(p) + 9, py(p) + 32, { color: cut ? AC.rose : AC.line, w: cut ? 2.2 : 1 }); } });
    F.g.append("text").attr("x", sx + 16).attr("y", sy + 4).attr("font-size", 9).attr("fill", AC.muted).text("→ every pixel, cap D₀");
    F.g.append("text").attr("x", tx + 16).attr("y", ty + 4).attr("font-size", 9).attr("fill", AC.muted).text("← every pixel, cap D₁");
    for (let i = 0; i < Rn; i++) for (let j = 0; j < Cn; j++) { const p = i * Cn + j; if (j + 1 < Cn) F.g.append("line").attr("x1", px(p) + 12).attr("y1", py(p)).attr("x2", px(p + 1) - 12).attr("y2", py(p + 1)).attr("stroke", lab[p] !== lab[p + 1] ? AC.rose : lam ? AC.line : AC.grid).attr("stroke-width", lab[p] !== lab[p + 1] ? 2.2 : 1); if (i + 1 < Rn) F.g.append("line").attr("x1", px(p)).attr("y1", py(p) + 12).attr("x2", px(p + Cn)).attr("y2", py(p + Cn) - 12).attr("stroke", lab[p] !== lab[p + Cn] ? AC.rose : lam ? AC.line : AC.grid).attr("stroke-width", lab[p] !== lab[p + Cn] ? 2.2 : 1); }
    for (let p = 0; p < N; p++) { F.g.append("circle").attr("cx", px(p)).attr("cy", py(p)).attr("r", 12).attr("fill", lab[p] ? AC.accent : AC.panel2).attr("stroke", AC.line); F.g.append("text").attr("x", px(p)).attr("y", py(p) + 4).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", lab[p] ? AC.bg : AC.ink).text(`${g.D0[p]}/${g.D1[p]}`); }
    [[sx, sy, "s"], [tx, ty, "t"]].forEach(([x, y, t]) => { F.g.append("circle").attr("cx", x).attr("cy", y).attr("r", 11).attr("fill", AC.panel2).attr("stroke", AC.a2); F.g.append("text").attr("x", x).attr("y", y + 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.ink).text(t); });
    F.g.append("text").attr("x", mx).attr("y", F.ih + 24).attr("font-size", 10).attr("fill", AC.muted).text(`min cut ${S.M.value} in ${S.augs} augmentations (Edmonds-Karp)`);
    /* right: energy across λ */
    const rx = 470, rw = F.iw - rx, rh = F.ih - 40; const xs = d3.scaleLinear().domain([0, 8]).range([rx, rx + rw]); const ys = d3.scaleLinear().domain([0, Math.max(...sweep.map(s => s.M.energy)) * 1.1]).range([rh, 20]);
    F.g.append("text").attr("x", rx).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text("min energy vs λ (rose: labelling changes)");
    F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${rh})`).call(d3.axisBottom(xs).ticks(8)); F.g.append("g").attr("class", "axis").attr("transform", `translate(${rx},0)`).call(d3.axisLeft(ys).ticks(4));
    F.g.append("path").attr("d", d3.line().x(s => xs(s.lam)).y(s => ys(s.M.energy))(sweep)).attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 2);
    sweep.forEach(s => { if (s.flips) F.g.append("line").attr("x1", xs(s.lam)).attr("x2", xs(s.lam)).attr("y1", ys(s.M.energy) - 8).attr("y2", ys(s.M.energy) + 8).attr("stroke", AC.rose).attr("stroke-width", 2); F.g.append("circle").attr("cx", xs(s.lam)).attr("cy", ys(s.M.energy)).attr("r", s.lam === lam ? 5 : 2.5).attr("fill", s.lam === lam ? AC.a2 : AC.accent); });
    F.g.append("text").attr("x", rx + rw).attr("y", rh + 26).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.muted).text("λ");
    const allOk = sweep.every(s => s.M.energy === s.Bf.energy && s.M.value === s.M.energy);
    const flipText = sweep.filter(s => s.flips).map(s => `λ = ${s.lam}: ${s.flips} flip${s.flips > 1 ? "s" : ""} (${s.M.label.map((x, p) => x !== sweep[s.lam - 1].M.label[p] ? "p" + p : null).filter(x => x).join(", ")})`).join("; ");
    SX.setHtml("seg-readout", `λ = ${lam}: min cut <b>${S.M.value}</b> = energy of the labelling ${S.M.energy} ${SX.flag(S.M.value === S.M.energy)} = brute force over ${S.Bf.count} labellings: ${S.Bf.energy} ${SX.flag(S.Bf.energy === S.M.energy)}${S.Bf.ties > 1 ? " (" + S.Bf.ties + " optimal labellings; drawn: the source side of the residual reachability)" : " (unique)"} · labelling ${lab.slice(0, 3).join("")} ${lab.slice(3, 6).join("")} ${lab.slice(6, 9).join("")}, boundary pairs ${(S.M.energy - lab.reduce((a, x, p) => a + (x ? g.D1[p] : g.D0[p]), 0)) / Math.max(1, lam) | 0}${lam === 0 ? " (not charged)" : ""} · V = ${g.n}, E = ${g.edges.length}, ${S.augs} augmentations · the sweep λ = 0…8 agrees with brute force at every λ ${SX.flag(allOk)}; changes: ${flipText || "none"}; total flips ${sweep.reduce((a, s) => a + s.flips, 0)} · also re-derived — six-vertex network: edge-disjoint paths ${ed.value} = brute force ${edB.best} over ${edB.paths} simple paths ${SX.flag(ed.value === edB.best)}, vertex-disjoint ${vd.value} = ${vdB.best} ${SX.flag(vd.value === vdB.best)} · project selection: min cut ${PJ.minCut}, best profit ${PJ.positiveSum} − ${PJ.minCut} = <b>${PJ.optimum}</b> choosing {${PJ.chosen.map(i => pn[i]).join(", ")}} = brute force over ${PJB.closures} closed sets: ${PJB.best} ${SX.flag(PJ.optimum === PJB.best)}, ${cp.get("aug")} augmentations`);
  }
  SX.on("seg-lambda", "input", build);
  build();
})();

/* ── chunk E figures ── */
/* ── SA  the load-time audit: every family of routine on the page, on random small instances, against its brute-force twin ── */
(function () {
  if (!SX.has("audit-readout")) return;
  const out = []; const c = AL.counter(); const t0 = (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();
  const INF = Infinity;
  const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
  const sameM = (A, B) => A.length === B.length && A.every((r, i) => same(r, B[i]));
  const piPathOk = (G, R) => R.d.every((x, v) => { if (x === INF) return R.pi[v] < 0; const p = []; let y = v; while (y >= 0) { p.push(y); y = R.pi[y]; } return SR.pathWeight(G, p.reverse()) === x; });
  /* 1. Dijkstra, all four implementations, on non-negative weights (zeros included) = brute force over all simple paths; π-paths re-sum; Bellman-Ford agrees */
  { const T = 150; let ok = 0, okPi = 0, okBF = 0, okCnt = 0; const r = AL.rng(201);
    for (let t = 0; t < T; t++) { const n = AL.randInt(r, 2, 7), m = AL.randInt(r, n - 1, 12); const G = SR.randGraph(n, m, 1000 + t, 0, 9, true); const b = SR.bruteAllPaths(G, 0);
      const ch = AL.counter(); const H = SR.dijkstra(G, 0, ch); const A = SR.dijkstraArray(G, 0, c), L = SR.dijkstraLazy(G, 0, c), Bd = SR.b_dijkstra(G, 0, c); c.add("extract", ch.get("extract")); c.add("relax", ch.get("relax") + ch.get("intoS"));
      if (same(H.d, b.d) && same(A.d, b.d) && same(L.d, b.d) && same(Bd.d, b.d)) ok++; if (piPathOk(G, H) && piPathOk(G, L)) okPi++;
      const bf = SR.bellmanFord(G, 0, c); if (!bf.negative && same(bf.d, b.d)) okBF++;
      if (ch.get("extract") === n && ch.get("relax") + ch.get("intoS") === G.edges.length && ch.get("decrease") <= G.edges.length) okCnt++; }
    out.push(`Dijkstra, ${T} random directed graphs (V ≤ 7, E ≤ 12, weights 0 … 9): index-map heap = array scan = lazy heap = the §11 binary heap = brute force over all simple paths <b>${ok}/${T}</b> ${SX.flag(ok === T)}; every π-path re-sums to its d <b>${okPi}/${T}</b> ${SX.flag(okPi === T)}; Bellman-Ford agrees <b>${okBF}/${T}</b> ${SX.flag(okBF === T)}; extractions = V, edges examined = E, decrease-keys ≤ E <b>${okCnt}/${T}</b> ${SX.flag(okCnt === T)}`); }
  /* 2. Bellman-Ford with negative weights: distances where δ is finite, the negative cycle where it is not, the −∞ set, the two-array recurrence */
  { const T = 200; let nNeg = 0, okD = 0, okTwo = 0, okB = 0, okDet = 0, okCyc = 0, okInf = 0, okRounds = 0; const r = AL.rng(202);
    for (let t = 0; t < T; t++) { const n = AL.randInt(r, 2, 6), m = AL.randInt(r, n - 1, 10); const G = SR.randGraph(n, m, 2000 + t, -3, 9, true); const cyc = SR.bruteCycles(G, 0); const R = SR.bellmanFord(G, 0, c); const R2 = SR.b_bellmanFord(G, 0, c);
      if (R.negative === cyc.negative && (R2.negCycle !== null) === cyc.negative) okDet++;
      if (cyc.negative) { nNeg++; if (R.negative) { let w = 0, real = true; for (let k = 0; k < R.cycle.length; k++) { const a = R.cycle[k], bb = R.cycle[(k + 1) % R.cycle.length]; const arc = SR.spAdj(G)[a].find(x => x.v === bb); if (!arc) real = false; else w += arc.w; } if (real && w < 0 && w === R.cycleW) okCyc++; if (same(R.negInf, SR.bruteNegInf(G, 0))) okInf++; } }
      else { const b = SR.bruteAllPaths(G, 0); if (!R.negative && same(R.d, b.d)) okD++; if (same(SR.bellmanFordTwoArray(G, 0, c).d, b.d)) okTwo++; if (same(R2.d, b.d)) okB++; if (R.stableAfter !== null && R.stableAfter <= n - 1) okRounds++; } }
    const nPos = T - nNeg;
    out.push(`Bellman-Ford, ${T} random directed graphs (V ≤ 6, E ≤ 10, weights −3 … 9), ${nNeg} of them with a negative cycle reachable from s: the test round's verdict = brute force over all simple cycles <b>${okDet}/${T}</b> ${SX.flag(okDet === T)}; on the ${nPos} without one, d = brute force over all simple paths <b>${okD}/${nPos}</b> ${SX.flag(okD === nPos)}, the two-array recurrence agrees <b>${okTwo}/${nPos}</b> ${SX.flag(okTwo === nPos)}, the §10 routine agrees <b>${okB}/${nPos}</b> ${SX.flag(okB === nPos)}, stable within V − 1 rounds <b>${okRounds}/${nPos}</b> ${SX.flag(okRounds === nPos)}; on the ${nNeg} with one, the reported cycle is a real cycle of negative weight <b>${okCyc}/${nNeg}</b> ${SX.flag(okCyc === nNeg)} and the −∞ set = every vertex reachable from a negative cycle <b>${okInf}/${nNeg}</b> ${SX.flag(okInf === nNeg)}`); }
  /* 3. DAG one pass: shortest and longest = enumeration; longest by negation; critical paths on random task sets */
  { const T = 150; let ok = 0, okNeg = 0, okCP = 0; const r = AL.rng(203);
    for (let t = 0; t < T; t++) { const n = AL.randInt(r, 2, 7); const edges = []; for (let v = 1; v < n; v++) { const must = AL.randInt(r, 0, v - 1); for (let u = 0; u < v; u++) if (u === must || r() < 0.4) edges.push({ u, v, w: AL.randInt(r, -4, 9) }); } const G = { n, edges, directed: true };
      const b = SR.b_bruteAllPaths(G, 0); const s = SR.b_dagSP(G, 0, "short", c), l = SR.b_dagSP(G, 0, "long", c); if (s.ok && same(s.d, b.short) && same(l.d, b.long)) ok++;
      const Gn = { n, edges: edges.map(e => ({ u: e.u, v: e.v, w: -e.w })), directed: true }; const ns = SR.b_dagSP(Gn, 0, "short", c); if (same(ns.d.map(x => x === INF ? -INF : -x), l.d)) okNeg++;
      const k = AL.randInt(r, 1, 6); const tasks = Array.from({ length: k }, (_, i) => ({ id: String(i), dur: AL.randInt(r, 1, 6), pre: [] })); for (let i = 1; i < k; i++) for (let j = 0; j < i; j++) if (r() < 0.4) tasks[i].pre.push(j);
      const cp = SR.b_criticalPath(tasks, c); const bt = SR.b_bruteAllPaths(cp.G, cp.G.S); const pathLen = cp.path.reduce((a, v) => a + tasks[v].dur, 0);
      if (cp.length === bt.long[cp.G.T] && cp.slack.every(x => x >= 0) && cp.critical.length >= 1 && pathLen === cp.length && cp.path.every(v => cp.slack[v] === 0)) okCP++; }
    out.push(`DAG one pass, ${T} random DAGs (V ≤ 7, weights −4 … 9): shortest and longest = enumeration of every path from the source <b>${ok}/${T}</b> ${SX.flag(ok === T)}; longest = shortest on the negated weights <b>${okNeg}/${T}</b> ${SX.flag(okNeg === T)}; critical path on ${T} random task sets (≤ 6 tasks): project length = the longest chain by enumeration, every slack ≥ 0, the critical path re-sums to the length and has slack 0 throughout <b>${okCP}/${T}</b> ${SX.flag(okCP === T)}`); }
  /* 4. difference constraints: feasible ⇔ no negative cycle; the δ solution satisfies every constraint (and so does a shifted copy); an infeasible system's reported cycle is a contradiction */
  { const T = 150; let nInf = 0, okIff = 0, okSat = 0, okShift = 0, okCert = 0; const r = AL.rng(204);
    for (let t = 0; t < T; t++) { const n = AL.randInt(r, 2, 5), m = AL.randInt(r, 1, 8); const cons = Array.from({ length: m }, () => { let i = AL.randInt(r, 1, n), j = AL.randInt(r, 1, n); while (j === i) j = AL.randInt(r, 1, n); return { j, i, b: AL.randInt(r, -3, 5) }; });
      const G = SR.b_diffconGraph(n, cons); const bf = SR.b_bellmanFord(G, 0, c); const neg = SR.bruteCycles(G, 0).negative; if ((bf.negCycle !== null) === neg) okIff++;
      if (bf.negCycle) { nInf++; let w = 0, onlyCons = true; for (let k = 0; k < bf.negCycle.length; k++) { const a = bf.negCycle[k], bb = bf.negCycle[(k + 1) % bf.negCycle.length]; if (a === 0 || bb === 0) onlyCons = false; const es = G.edges.filter(e => e.u === a && e.v === bb); w += es.length ? Math.min(...es.map(e => e.w)) : INF; } if (onlyCons && w < 0) okCert++; }
      else { if (SR.b_diffconCheck(bf.d, cons).every(k => k.ok)) okSat++; const sh = AL.randInt(r, -5, 5); if (SR.b_diffconCheck(bf.d.map(x => x + sh), cons).every(k => k.ok)) okShift++; } }
    const nF = T - nInf;
    out.push(`difference constraints, ${T} random systems (≤ 5 unknowns, ≤ 8 constraints, bounds −3 … 5), ${nInf} infeasible: Bellman-Ford's verdict = "the constraint graph has a negative cycle", by enumerating every cycle <b>${okIff}/${T}</b> ${SX.flag(okIff === T)}; on the ${nF} feasible, x = δ(v₀, ·) satisfies every constraint <b>${okSat}/${nF}</b> ${SX.flag(okSat === nF)} and so does x + c for a random c <b>${okShift}/${nF}</b> ${SX.flag(okShift === nF)}; on the ${nInf} infeasible, the reported cycle uses only constraint edges and its bounds (the tightest per pair) sum below 0 <b>${okCert}/${nInf}</b> ${SX.flag(okCert === nInf)}`); }
  /* 5. 0-1 BFS and Dial's buckets = a binary-heap Dijkstra; the deque invariant; Dial's bucket-inspection count */
  { const T = 120; let ok01 = 0, okInv = 0, okDial = 0, okScan = 0; const r = AL.rng(205);
    for (let t = 0; t < T; t++) { const Gg = SR.b_gridGraph(AL.randInt(r, 2, 4), AL.randInt(r, 2, 5), 0, 1, 5000 + t); const c1 = AL.counter(); const z = SR.b_zeroOneBFS(Gg, 0, c1); c.add("relax", c1.get("relax")); const dj = SR.b_dijkstra(Gg, 0, c); if (same(z.d, dj.d)) ok01++; if (z.maxDistinct <= 2 && z.monotone && c1.get("pop") - c1.get("stale") === Gg.n) okInv++;
      const C = AL.randInt(r, 1, 4); const n = AL.randInt(r, 2, 7); const Gd = SR.randGraph(n, AL.randInt(r, n - 1, 12), 6000 + t, 0, C, true); const c2 = AL.counter(); const dl = SR.b_dial(Gd, 0, C, c2); c.add("relax", c2.get("relax")); const dj2 = SR.b_dijkstra(Gd, 0, c); if (same(dl.d, dj2.d)) okDial++; const sc = c2.get("scan") - c2.get("pop"); if (sc >= dl.maxDist && sc <= dl.maxDist + C) okScan++; }
    out.push(`small integer weights, ${T} random 0/1 grids (≤ 4 × 5) and ${T} random directed graphs with weights 0 … C (C ≤ 4, V ≤ 7): 0-1 BFS = binary-heap Dijkstra <b>${ok01}/${T}</b> ${SX.flag(ok01 === T)}, the deque held ≤ 2 distinct labels, non-decreasing, and expanded each vertex once <b>${okInv}/${T}</b> ${SX.flag(okInv === T)}; Dial = Dijkstra <b>${okDial}/${T}</b> ${SX.flag(okDial === T)}, bucket inspections − pops (the pointer's advances) between the largest finalised distance and that + C <b>${okScan}/${T}</b> ${SX.flag(okScan === T)}`); }
  /* 6. all pairs: squaring (fast and slow) = Floyd-Warshall = V × Bellman-Ford = brute force; the transitive closure; negative cycles seen by every method */
  { const T = 120; let nNeg = 0, okDet = 0, okAll = 0, okTC = 0, okJ = 0, okJneg = 0; const r = AL.rng(206);
    for (let t = 0; t < T; t++) { const n = AL.randInt(r, 2, 6); const G = SR.randGraph(n, AL.randInt(r, n - 1, 10), 7000 + t, -2, 9, true); let neg = false; for (let v = 0; v < n && !neg; v++) if (SR.bruteCycles(G, v).negative) neg = true;
      const W = SR.b_weightMatrix(G); const FW = SR.b_floydWarshall(W, c); const J = SR.b_johnson(G, c); if (FW.negCycle === neg && (!!J.negCycle) === neg) okDet++;
      if (neg) { nNeg++; if (J.negCycle && FW.negCycle) okJneg++; }
      else { const fast = SR.b_apspSquaring(W, c).D, slow = SR.b_apspSlow(W, c).D, bfa = SR.b_bfAllPairs(G, c), br = SR.b_bruteAllPairs(G).D; if (sameM(fast, br) && sameM(slow, br) && sameM(FW.D, br) && sameM(bfa, br)) okAll++;
        const Tc = SR.b_transitiveClosure(G, c); if (Tc.every((row, i) => row.every((x, j) => x === (FW.D[i][j] < INF)))) okTC++;
        if (J.allNonNeg && sameM(J.D, FW.D)) okJ++; } }
    const nPos = T - nNeg;
    out.push(`all pairs, ${T} random directed graphs (V ≤ 6, E ≤ 10, weights −2 … 9), ${nNeg} with a negative cycle: Floyd-Warshall's diagonal test and Johnson's Bellman-Ford both report it exactly when cycle enumeration finds one <b>${okDet}/${T}</b> ${SX.flag(okDet === T)}; on the ${nPos} without one, repeated squaring (fast and slow) = Floyd-Warshall = V runs of Bellman-Ford = brute force over all simple paths from every source <b>${okAll}/${nPos}</b> ${SX.flag(okAll === nPos)}, the boolean closure = "D &lt; ∞" <b>${okTC}/${nPos}</b> ${SX.flag(okTC === nPos)}, Johnson's ŵ all ≥ 0 and its matrix = Floyd-Warshall's <b>${okJ}/${nPos}</b> ${SX.flag(okJ === nPos)}`); }
  /* 7. A* on random grids: the three admissible heuristics find the BFS distance, never reopen, and expand exactly the necessary set below C*; weighted A* stays within its factor */
  { const T = 100; let okDist = 0, okReopen = 0, okNec = 0, okW = 0, nReach = 0, wLonger = 0; const r = AL.rng(207);
    for (let t = 0; t < T; t++) { const rows = AL.randInt(r, 3, 6), cols = AL.randInt(r, 3, 8); const walls = new Set(); for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) if (r() < 0.25) walls.add(i + "," + j); const free = []; for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) if (!walls.has(i + "," + j)) free.push([i, j]); if (free.length < 2) { t--; continue; }
      const a = free[AL.randInt(r, 0, free.length - 1)]; let bcell = free[AL.randInt(r, 0, free.length - 1)]; while (bcell[0] === a[0] && bcell[1] === a[1]) bcell = free[AL.randInt(r, 0, free.length - 1)]; const grid = { rows, cols, walls, start: a, goal: bcell };
      const bfs = SR.b_gridBFS(grid); const runs = ["zero", "euclid", "manhattan"].map(k => { const ck = AL.counter(); const R = SR.b_astar(grid, k, ck); c.add("expand", ck.get("expand")); return { k, R }; });
      if (runs.every(x => x.R.dist === bfs.dist)) okDist++; if (runs.every(x => x.R.reopened === 0)) okReopen++;
      if (bfs.dist < INF) { nReach++; let nec = true; runs.forEach(x => { const ex = new Set(x.R.expanded); for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) { const v = i * cols + j; if (walls.has(i + "," + j) || bfs.d[v] === INF) continue; const f = bfs.d[v] + SR.b_gridDist([i, j], bcell, x.k); if (f < bfs.dist - 1e-9 && !ex.has(v)) nec = false; if (f > bfs.dist + 1e-9 && ex.has(v)) nec = false; } }); if (nec) okNec++; }
      const w2 = SR.b_astar(grid, "manhattan2", c); if (w2.dist === bfs.dist || (bfs.dist < INF && w2.dist <= 2 * bfs.dist)) okW++; if (bfs.dist < INF && w2.dist > bfs.dist) wLonger++; }
    out.push(`A*, ${T} random grids (≤ 6 × 8, 25% walls, random start and goal, ${nReach} with the goal reachable): zero, Euclidean and Manhattan heuristics all return the BFS distance <b>${okDist}/${T}</b> ${SX.flag(okDist === T)}, never reopen a closed cell (consistency) <b>${okReopen}/${T}</b> ${SX.flag(okReopen === T)}, and expand every cell with g + h &lt; C* and none with g + h &gt; C* <b>${okNec}/${nReach}</b> ${SX.flag(okNec === nReach)}; 2 × Manhattan stays within twice the optimum <b>${okW}/${T}</b> ${SX.flag(okW === T)} and returned a longer path on ${wLonger} — inadmissible, as it should be`); }
  /* 8. minimum spanning trees: Kruskal = Prim (random root) = Borůvka = brute force over all spanning trees; distinct weights ⇒ one tree, the same edge set from all three; the cut, cycle, bottleneck and second-best facts */
  { const T = 120; let okW = 0, okSet = 0, nDist = 0, okCut = 0, okCyc = 0, okMB = 0, okSB = 0, okCl = 0; const r = AL.rng(208);
    for (let t = 0; t < T; t++) { const n = AL.randInt(r, 3, 6), m = AL.randInt(r, n - 1, Math.min(9, n * (n - 1) / 2)); const distinct = r() < 0.5; const G = SR.mstRandomGraph(n, m, 8000 + t, distinct); const B = SR.bruteSpanningTrees(G);
      const K = SR.kruskal(G, c), P = SR.prim(G, AL.randInt(r, 0, n - 1), c), Bo = SR.boruvka(G, c); if (K.weight === B.minW && P.weight === B.minW && Bo.weight === B.minW && K.complete && P.complete && Bo.complete) okW++;
      const key = tr => tr.slice().sort((a, b) => a - b).join(); if (distinct) { nDist++; if (B.unique && key(K.tree) === key(B.msts[0]) && key(P.tree) === key(B.msts[0]) && key(Bo.tree) === key(B.msts[0])) okSet++; }
      const S = []; for (let v = 0; v < n; v++) if (r() < 0.5) S.push(v); if (!S.length) S.push(0); if (S.length === n) S.pop(); const cut = SR.mstCut(G, S, B); if (cut.light < 0 || cut.inSomeMst) okCut++;
      if (SR.mstCycleCheck(G, B).every(x => x.someMstOmits)) okCyc++;
      if (Math.max(...K.tree.map(i => G.edges[i].w)) === B.minBottleneck) okMB++;
      const sb = SR.secondBestBySwap(G, K.tree); if (B.trees.length === 1 ? sb.best === null : (sb.best && sb.best.weight === B.secondBestW)) okSB++;
      const k = AL.randInt(r, 1, n); const cl = SR.mstClusters(G, K.tree, k); const spacing = cl.clusters.length > 1 ? Math.min(...G.edges.filter(e => cl.label[e.u] !== cl.label[e.v]).map(e => e.w)) : INF; const cutMin = cl.cut.length ? Math.min(...cl.cut.map(i => G.edges[i].w)) : INF; if (cl.clusters.length === k && spacing === cutMin) okCl++; }
    out.push(`minimum spanning trees, ${T} random connected graphs (V ≤ 6, E ≤ 9; ${nDist} with distinct weights): Kruskal = Prim from a random root = Borůvka = brute force over all spanning trees <b>${okW}/${T}</b> ${SX.flag(okW === T)}; with distinct weights the MST is unique and all three return that edge set <b>${okSet}/${nDist}</b> ${SX.flag(okSet === nDist)}; the light edge across a random cut lies in some MST <b>${okCut}/${T}</b> ${SX.flag(okCut === T)}; the heaviest edge on the cycle each non-tree edge closes is omitted by some MST <b>${okCyc}/${T}</b> ${SX.flag(okCyc === T)}; the MST's heaviest edge = the minimum bottleneck over all trees <b>${okMB}/${T}</b> ${SX.flag(okMB === T)}; the one-swap second-best = brute force <b>${okSB}/${T}</b> ${SX.flag(okSB === T)}; cutting the k − 1 heaviest MST edges gives k clusters whose spacing is the lightest cut edge <b>${okCl}/${T}</b> ${SX.flag(okCl === T)}`); }
  /* 9. union-find: the partition it maintains = connected components, under all four heuristic combinations; rank keeps every height ≤ ⌊log₂ n⌋; the chain adversary; the α table */
  { const T = 150; let okPart = 0, okH = 0, okChain = 0; const r = AL.rng(209);
    for (let t = 0; t < T; t++) { const n = AL.randInt(r, 2, 24), m = AL.randInt(r, 1, 40); const ops = SR.ufRandomOps(n, m, 9000 + t, 0.6);
      /* components of the graph whose edges are the union pairs, by BFS */
      const adj = Array.from({ length: n }, () => []); ops.forEach(o => { if (o.op === "u") { adj[o.a].push(o.b); adj[o.b].push(o.a); } }); const comp = new Array(n).fill(-1); let k = 0; for (let s = 0; s < n; s++) if (comp[s] < 0) { comp[s] = k; const q = [s]; while (q.length) { const u = q.shift(); adj[u].forEach(v => { if (comp[v] < 0) { comp[v] = k; q.push(v); } }); } k++; }
      let allOk = true, hOk = true; SR.UF_COMBOS.forEach(cb => { const R = SR.ufRun(n, ops, cb.opts, c); const rep = Array.from({ length: n }, (_, v) => SR.ufFind(R.uf, v)); for (let u = 0; u < n; u++) for (let v = u + 1; v < n; v++) if ((rep[u] === rep[v]) !== (comp[u] === comp[v])) allOk = false; if (cb.opts.rank && R.height > Math.floor(Math.log2(n))) hOk = false; }); if (allOk) okPart++; if (hOk) okH++;
      const ch = SR.ufChainOps(n, 1); const naive = SR.ufRun(n, ch, { rank: false, compress: "none" }, c), rk = SR.ufRun(n, ch, { rank: true, compress: "none" }, c); if (naive.height === n - 1 && rk.height <= 1) okChain++; }
    const alphaOk = [[2, 0], [3, 1], [4, 2], [7, 2], [8, 3], [2047, 3], [2048, 4], [1e6, 4]].every(([n, a]) => SR.ackAlpha(n) === a);
    out.push(`union-find, ${T} random operation sequences (n ≤ 24, ≤ 40 ops): "same set?" = connectivity by BFS over the union pairs, under all four heuristic combinations <b>${okPart}/${T}</b> ${SX.flag(okPart === T)}; with union by rank every final height ≤ ⌊log₂ n⌋ <b>${okH}/${T}</b> ${SX.flag(okH === T)}; the chain of n − 1 sequential unions has height n − 1 under naive linking and ≤ 1 under rank <b>${okChain}/${T}</b> ${SX.flag(okChain === T)}; α(n) = 0, 1, 2, 2, 3, 3, 4, 4 at n = 2, 3, 4, 7, 8, 2047, 2048, 10⁶ ${SX.flag(alphaOk)}`); }
  /* 10. maximum flow: Ford-Fulkerson (three path rules) = Edmonds-Karp = Dinic = push-relabel = brute-force minimum cut; the certificates and the counted bounds */
  { const T = 100; let okVal = 0, okCert = 0, okFlow = 0, okEK = 0, okPR = 0; const r = AL.rng(210);
    for (let t = 0; t < T; t++) { const n = AL.randInt(r, 4, 7), m = AL.randInt(r, n, 12); const G = SR.flowRandom(10000 + t, n, m, AL.rng); const E = G.edges.length;
      const cb = AL.counter(); const B = SR.bruteMinCut(G, cb); c.add("cut", cb.get("cut")); const ce = AL.counter(); const ek = SR.ekMaxflow(G, ce); c.add("aug", ce.get("aug")); const dn = SR.dinicMaxflow(G, c); const cp = AL.counter(); const pr = SR.prMaxflow(G, cp); c.add("push", cp.get("pushSat") + cp.get("pushNon")); c.add("relabel", cp.get("relabel"));
      const ffd = SR.ffMaxflow(G, "dfs", c), ffw = SR.ffMaxflow(G, "worst", c), ffb = SR.ffMaxflow(G, "bfs", c);
      if ([ek.value, dn.value, pr.value, ffd.value, ffw.value, ffb.value].every(v => v === B.cap)) okVal++;
      const chk = SR.flowCheck(G, ek.f); const chkP = SR.flowCheck(G, pr.f); if (chk.capOk && chk.consOk && chk.value === ek.value && chkP.capOk && chkP.consOk && chkP.value === pr.value) okFlow++;
      if (SR.flowCut(G, null, ek.S).cap === ek.value && SR.flowCut(G, ek.f, ek.S).net === ek.value && SR.flowCut(G, null, ffd.S).cap === ffd.value) okCert++;
      const maxCrit = Math.max(0, ...ek.critCount.map(x => Math.max(x[0], x[1]))); if (ek.augs <= n * E && maxCrit <= n / 2 && ek.lens.every((l, i) => i === 0 || l >= ek.lens[i - 1]) && dn.phases <= n - 1) okEK++;
      if (cp.get("pushSat") < 2 * n * E && cp.get("pushNon") < 4 * n * n * (n + E) && cp.get("relabel") <= (2 * n - 1) * (n - 2) && Math.max(...pr.h) <= 2 * n - 1) okPR++; }
    out.push(`maximum flow, ${T} random networks (V ≤ 7, E ≤ 12, capacities 1 … 9): Ford-Fulkerson under all three path rules = Edmonds-Karp = Dinic = push-relabel = brute force over all 2^(V−2) cuts <b>${okVal}/${T}</b> ${SX.flag(okVal === T)}; the returned flows satisfy capacity and conservation with the stated value <b>${okFlow}/${T}</b> ${SX.flag(okFlow === T)}; the residual-reachable set is a cut of capacity |f*| with net flow |f*| across it <b>${okCert}/${T}</b> ${SX.flag(okCert === T)}; Edmonds-Karp: augmentations ≤ VE, no arc critical more than V/2 times, path lengths non-decreasing, Dinic phases ≤ V − 1 <b>${okEK}/${T}</b> ${SX.flag(okEK === T)}; push-relabel: pushes and relabels within the generic bounds, heights ≤ 2V − 1 <b>${okPR}/${T}</b> ${SX.flag(okPR === T)}`); }
  /* 11. bipartite matching: augmenting paths = Hopcroft-Karp = the explicit flow = brute force over all matchings; König's cover; Hall's deficiency */
  { const T = 120; let okM = 0, okK = 0, okH = 0; const r = AL.rng(211);
    for (let t = 0; t < T; t++) { const nL = AL.randInt(r, 1, 4), nR = AL.randInt(r, 1, 4); const adj = Array.from({ length: nL }, () => { const row = []; for (let j = 0; j < nR; j++) if (r() < 0.5) row.push(j); return row; }); const B = { nL, nR, adj };
      const M = SR.bipartiteMatch(B, c), HK = SR.hopcroftKarp(B, c), bm = SR.bruteMatching(B), fl = SR.ekMaxflow(SR.bipToFlow(B), c); if (M.size === bm.size && HK.size === bm.size && fl.value === bm.size && M.matchL.every((rr, l) => rr < 0 || M.matchR[rr] === l)) okM++;
      const K = SR.konigCover(B, M); const bc = SR.bruteVertexCover(B); const isC = SR.isVertexCover(B, Array.from({ length: nL }, (_, l) => K.coverL.includes(l)), Array.from({ length: nR }, (_, j) => K.coverR.includes(j))); if (isC && K.size === M.size && bc.size === M.size) okK++;
      const H = SR.hallViolator(B); if (nL - H.deficiency === M.size && (H.deficiency > 0) === (M.size < nL)) okH++; }
    out.push(`bipartite matching, ${T} random bipartite graphs (|L|, |R| ≤ 4, edge probability ½): augmenting paths = Hopcroft-Karp = the unit-capacity flow = brute force over all matchings, and the matching is consistent both ways <b>${okM}/${T}</b> ${SX.flag(okM === T)}; König's cover covers every edge and has size |M| = the brute-force minimum cover <b>${okK}/${T}</b> ${SX.flag(okK === T)}; Hall: |L| − max deficiency = |M|, and a violator exists exactly when L is not fully matched <b>${okH}/${T}</b> ${SX.flag(okH === T)}`); }
  /* 12. the reductions: segmentation min cut = brute force over all labellings; project selection = all closures; disjoint paths = all path families */
  { const T = 60; let okSeg = 0, okPJ = 0, okDP = 0; const r = AL.rng(212);
    for (let t = 0; t < T; t++) { const R = AL.randInt(r, 2, 3), C = AL.randInt(r, 2, 3); const I = Array.from({ length: R }, () => Array.from({ length: C }, () => AL.randInt(r, 0, 10))); const g = SR.segBuild(I, AL.randInt(r, 6, 10), AL.randInt(r, 0, 4), AL.randInt(r, 0, 4)); const M = SR.segMinCut(g, c); const Bf = SR.segBrute(g); if (M.value === M.energy && M.energy === Bf.energy) okSeg++;
      const k = AL.randInt(r, 1, 5); const profit = Array.from({ length: k }, () => AL.randInt(r, -6, 9)); const pre = []; for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) if (i !== j && r() < 0.25) pre.push([i, j]); const PJ = SR.projectSelection(profit, pre, c); const PB = SR.bruteProjects(profit, pre); const closed = pre.every(([i, j]) => !PJ.chosen.includes(i) || PJ.chosen.includes(j)); if (PJ.optimum === PB.best && closed && PJ.chosen.reduce((a, i) => a + profit[i], 0) === PB.best) okPJ++;
      const n = AL.randInt(r, 3, 5); const G = SR.flowRandom(12000 + t, n, AL.randInt(r, n - 1, 7), AL.rng); const ed = SR.ekMaxflow(SR.unitCapacity(G), c).value, vd = SR.ekMaxflow(SR.vertexSplit(G), c).value; if (ed === SR.bruteDisjointPaths(G, false).best && vd === SR.bruteDisjointPaths(G, true).best) okDP++; }
    out.push(`the reductions, ${T} random instances each: segmentation of a random ≤ 3 × 3 image (λ ≤ 4) — min cut = the energy of its labelling = brute force over all 2ᴺ labellings <b>${okSeg}/${T}</b> ${SX.flag(okSeg === T)}; project selection (≤ 5 projects, random prerequisites) — P − min cut = the best closed set by enumeration, and the chosen set is closed and re-sums <b>${okPJ}/${T}</b> ${SX.flag(okPJ === T)}; edge- and vertex-disjoint s ⇝ t paths (V ≤ 5, E ≤ 7) = the largest pairwise-disjoint family of enumerated paths <b>${okDP}/${T}</b> ${SX.flag(okDP === T)}`); }
  const ms = ((typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now()) - t0;
  const allOk = !out.some(s => s.includes("DISAGREE"));
  SX.setHtml("audit-readout", `<b>Load-time audit</b> — ${out.length} families of random instances, every answer on this page recomputed independently (brute force over all simple paths, simple cycles, spanning trees, s-t cuts, matchings, vertex subsets, labellings, closures and path families; a second algorithm where enumeration is not the natural check):<br>` + out.map(s => "· " + s).join("<br>") + `<br>· overall: ${allOk ? '<span style="color:' + AC.good + '">all audits agree</span>' : '<span style="color:' + AC.bad + '">SOME AUDIT DISAGREES</span>'} · total instrumented work in the audit: ${SX.int(Object.values(c.all()).reduce((a, b) => a + b, 0))} counted operations in ${ms.toFixed(0)} ms`);
  SX.setHtml("audit-cell", allOk ? "all " + out.length + " families agree — details in the readout below" : "a family DISAGREES — see below");
})();

/* @@FIGURES-END@@ */
})();
