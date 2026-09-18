/* graphs.viz.js — figures for dsa/data-structures/graphs.html
   (Data Structures · part 7: Graph Representations & Traversal).

   Loaded after ../../data.js, ../../notes.js and ../dsa-viz.js, which supplies the
   shared toolbox (AC palette, AL.frame / AL.row / AL.stepper / AL.counter / AL.rng …).

   Layout of this file
     GR   — graphs and the algorithms on them, pure (no DOM): construction (edge list
            → adjacency lists / adjacency matrix / CSR), generators (random simple
            graph, path, star, grid, complete, cycle, random DAG), BFS with a queue
            trace, DFS (recursive AND explicit-stack, producing identical d/f times)
            with edge classification, the parenthesis check, cycle detection for both
            kinds of graph, connected components + union-find, bipartite 2-colouring
            with the odd cycle it returns on failure, topological sort by both
            methods + a linear-extension counter, strongly connected components by
            the two-pass (reverse-graph) and the single-pass (low-link) algorithms,
            articulation points and bridges, and BRUTE-FORCE cross-checks for every
            one of them (boolean matrix powers for distances, transitive closure for
            reachability / SCCs, vertex- and edge-deletion for cut vertices and
            bridges). Every routine takes an AL.counter() so that every number the
            page displays is a MEASUREMENT.
     GD   — drawing: a graph drawer (fixed or force positions, undirected or directed
            with arrowheads and curved reciprocal edges, per-node and per-edge colour
            hooks), an interval/bracket strip, a queue/stack strip, a matrix grid.
     figures — one IIFE per <svg id>, guarded by the element's existence.

   Conventions: vertices are 0 … n−1 and drawn with letters a, b, c …; an undirected
   edge {u, v} appears in BOTH adjacency lists; edge scans count every entry of every
   adjacency list examined (so 2E for undirected, E for directed); a "visit" is a
   dequeue (BFS) or a discovery (DFS); times d/f start at 1. */

const GR = (function () {

  const L = i => String.fromCharCode(97 + i);           // 0 → a, 1 → b, …
  const lbl = i => i < 26 ? L(i) : String(i);

  /* ── construction ───────────────────────────────────────────────────────── */
  /* graph(n, edges, directed): adjacency lists in insertion order, PLUS the edge list
     and per-entry edge ids so an undirected traversal can exclude the edge it came
     in on (not merely the parent vertex — that distinction matters for multi-edges). */
  function graph(n, edges, directed) {
    const adj = Array.from({ length: n }, () => []);      // adj[u] = [{v, id}]
    edges.forEach(([u, v], id) => {
      adj[u].push({ v, id });
      if (!directed && u !== v) adj[v].push({ v: u, id });
      else if (!directed && u === v) adj[u].push({ v: u, id });   // an undirected self-loop appears twice, so Σdeg = 2E still holds
    });
    return { n, m: edges.length, directed: !!directed, edges: edges.map(e => [e[0], e[1]]), adj };
  }
  function sortAdj(g) { g.adj.forEach(a => a.sort((p, q) => p.v - q.v || p.id - q.id)); return g; }
  function matrix(g, c) {
    const A = Array.from({ length: g.n }, () => new Array(g.n).fill(0));
    for (const [u, v] of g.edges) { A[u][v] += 1; if (c) c.add("cells"); if (!g.directed && u !== v) { A[v][u] += 1; if (c) c.add("cells"); } }
    return A;
  }
  function csr(g) {                                    // offsets[u] … offsets[u+1] index into targets
    const off = new Array(g.n + 1).fill(0);
    g.adj.forEach((a, u) => { off[u + 1] = off[u] + a.length; });
    const tg = new Array(off[g.n]);
    g.adj.forEach((a, u) => a.forEach((e, k) => { tg[off[u] + k] = e.v; }));
    return { offsets: off, targets: tg };
  }
  function reverse(g) { return graph(g.n, g.edges.map(([u, v]) => [v, u]), true); }
  function degrees(g) {
    const out = new Array(g.n).fill(0), inn = new Array(g.n).fill(0);
    for (const [u, v] of g.edges) { out[u]++; inn[v]++; }
    if (!g.directed) return { deg: g.adj.map(a => a.length), out, inn };
    return { deg: g.adj.map(a => a.length), out, inn };
  }

  /* ── generators (all seeded through the caller's rng) ───────────────────── */
  function random(n, m, directed, r, opt) {
    const o = Object.assign({ simple: true }, opt || {});
    const maxM = directed ? n * (n - 1) : n * (n - 1) / 2;
    if (o.simple) m = Math.min(m, maxM);
    const seen = new Set(), edges = [];
    let guard = 0;
    while (edges.length < m && guard++ < 200000) {
      const u = Math.floor(r() * n), v = Math.floor(r() * n);
      if (u === v) continue;
      const key = directed ? u * n + v : Math.min(u, v) * n + Math.max(u, v);
      if (o.simple && seen.has(key)) continue;
      seen.add(key); edges.push([u, v]);
    }
    return graph(n, edges, directed);
  }
  function randomDAG(n, m, r) {                      // edges only from lower to higher rank in a random permutation
    const rank = AL.shuffle(Array.from({ length: n }, (_, i) => i), r);
    const seen = new Set(), edges = [];
    let guard = 0;
    const maxM = n * (n - 1) / 2; m = Math.min(m, maxM);
    while (edges.length < m && guard++ < 200000) {
      let u = Math.floor(r() * n), v = Math.floor(r() * n);
      if (u === v) continue;
      if (rank[u] > rank[v]) { const t = u; u = v; v = t; }
      const key = u * n + v; if (seen.has(key)) continue;
      seen.add(key); edges.push([u, v]);
    }
    return graph(n, edges, true);
  }
  function path(n) { const e = []; for (let i = 0; i + 1 < n; i++) e.push([i, i + 1]); return graph(n, e, false); }
  function cycle(n) { const e = []; for (let i = 0; i < n; i++) e.push([i, (i + 1) % n]); return graph(n, e, false); }
  function star(n) { const e = []; for (let i = 1; i < n; i++) e.push([0, i]); return graph(n, e, false); }
  function complete(n, directed) { const e = []; for (let i = 0; i < n; i++) for (let j = directed ? 0 : i + 1; j < n; j++) if (i !== j) e.push([i, j]); return graph(n, e, directed); }
  function grid(w, h, walls) {                       // 4-neighbour grid; walls = Set of blocked cell ids
    const e = [], W = walls || new Set(), id = (x, y) => y * w + x;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (W.has(id(x, y))) continue;
      if (x + 1 < w && !W.has(id(x + 1, y))) e.push([id(x, y), id(x + 1, y)]);
      if (y + 1 < h && !W.has(id(x, y + 1))) e.push([id(x, y), id(x, y + 1)]);
    }
    return graph(w * h, e, false);
  }
  function binaryTree(n) { const e = []; for (let i = 1; i < n; i++) e.push([Math.floor((i - 1) / 2), i]); return graph(n, e, false); }

  /* ── BFS ────────────────────────────────────────────────────────────────── */
  /* bfs(g, sources, c, tr): sources may be one vertex or an array (multi-source).
     Counter keys: enq, deq, scans, maxQ. Trace frames (if tr) are taken after every
     dequeue and after every enqueue so the queue can be drawn at each moment. */
  function bfs(g, sources, c, tr) {
    const S = Array.isArray(sources) ? sources : [sources];
    const dist = new Array(g.n).fill(-1), parent = new Array(g.n).fill(-1), order = [];
    const q = []; let head = 0, maxQ = 0;
    const colour = new Array(g.n).fill(0);            // 0 white, 1 grey (in queue), 2 black (done)
    for (const s of S) { dist[s] = 0; colour[s] = 1; q.push(s); if (c) c.add("enq"); }
    maxQ = q.length;
    const snap = (ev, u, v) => { if (tr) tr({ ev, u, v, queue: q.slice(head), dist: dist.slice(), parent: parent.slice(), colour: colour.slice(), order: order.slice() }); };
    snap("start", -1, -1);
    while (head < q.length) {
      const u = q[head++]; if (c) c.add("deq"); order.push(u);
      snap("dequeue", u, -1);
      for (const e of g.adj[u]) {
        if (c) c.add("scans");
        if (colour[e.v] === 0) {
          colour[e.v] = 1; dist[e.v] = dist[u] + 1; parent[e.v] = u; q.push(e.v); if (c) c.add("enq");
          if (q.length - head > maxQ) maxQ = q.length - head;
          snap("enqueue", u, e.v);
        } else snap("skip", u, e.v);
      }
      colour[u] = 2;
    }
    if (c) c.add("maxQ", maxQ);
    return { dist, parent, order, maxQ };
  }
  function bfsPath(res, v) { const p = []; while (v !== -1) { p.push(v); v = res.parent[v]; } return p.reverse(); }
  function layers(dist) { const L = []; dist.forEach((d, v) => { if (d >= 0) (L[d] = L[d] || []).push(v); }); return L; }

  /* brute-force distances, two independent ways:
     (1) boolean matrix powers: dist(s,v) = least k with (A^k)[s][v] ≠ 0
     (2) unit-weight edge relaxation (Bellman-Ford), n−1 rounds over the edge list */
  function distByMatrixPower(g, s) {
    const n = g.n, A = matrix(g).map(r => r.map(x => x > 0 ? 1 : 0));
    const dist = new Array(n).fill(-1); dist[s] = 0;
    let P = A; let k = 1, found = 1;
    const mult = (X, Y) => X.map(row => Y[0].map((_, j) => row.some((x, t) => x && Y[t][j]) ? 1 : 0));
    while (found < n && k < n) {                      // (A^k)[s][v] ≠ 0 ⇔ a walk of length k; the least such k is the distance
      for (let v = 0; v < n; v++) if (dist[v] < 0 && P[s][v]) { dist[v] = k; found++; }
      P = mult(P, A); k++;
    }
    return dist;
  }
  function distByRelaxation(g, s) {
    const dist = new Array(g.n).fill(Infinity); dist[s] = 0;
    for (let round = 0; round < g.n - 1; round++) {
      let changed = false;
      for (const [u, v] of g.edges) {
        if (dist[u] + 1 < dist[v]) { dist[v] = dist[u] + 1; changed = true; }
        if (!g.directed && dist[v] + 1 < dist[u]) { dist[u] = dist[v] + 1; changed = true; }
      }
      if (!changed) break;
    }
    return dist.map(d => d === Infinity ? -1 : d);
  }
  function sameArr(a, b) { return a.length === b.length && a.every((x, i) => x === b[i]); }

  /* ── bipartite 2-colouring by BFS, returning an odd cycle on failure ───────── */
  function bipartite(g, c) {
    const col = new Array(g.n).fill(-1), parent = new Array(g.n).fill(-1), dist = new Array(g.n).fill(-1);
    for (let s = 0; s < g.n; s++) {
      if (col[s] >= 0) continue;
      col[s] = 0; dist[s] = 0; const q = [s]; let head = 0;
      while (head < q.length) {
        const u = q[head++]; if (c) c.add("deq");
        for (const e of g.adj[u]) {
          if (c) c.add("scans");
          if (col[e.v] < 0) { col[e.v] = 1 - col[u]; dist[e.v] = dist[u] + 1; parent[e.v] = u; q.push(e.v); }
          else if (col[e.v] === col[u]) {
            // odd cycle: u → … → lca ← … ← v, plus the edge (v, u)
            const pu = [], pv = []; let a = u, b = e.v;
            const anc = new Set(); for (let x = u; x !== -1; x = parent[x]) anc.add(x);
            let lca = e.v; while (!anc.has(lca)) lca = parent[lca];
            for (let x = u; x !== lca; x = parent[x]) pu.push(x); pu.push(lca);
            for (let x = e.v; x !== lca; x = parent[x]) pv.push(x);
            const cyc = pu.concat(pv.reverse());
            return { ok: false, colour: col, oddCycle: cyc, badEdge: [u, e.v] };
          }
        }
      }
    }
    return { ok: true, colour: col, oddCycle: null };
  }
  function checkColouring(g, col) { return g.edges.every(([u, v]) => col[u] !== col[v] && col[u] >= 0 && col[v] >= 0); }
  function isCycle(g, cyc) {                          // consecutive vertices adjacent and last→first adjacent, all distinct
    if (cyc.length < 3) return false;
    if (new Set(cyc).size !== cyc.length) return false;
    const has = (u, v) => g.adj[u].some(e => e.v === v);
    for (let i = 0; i < cyc.length; i++) if (!has(cyc[i], cyc[(i + 1) % cyc.length])) return false;
    return true;
  }

  /* ── DFS ────────────────────────────────────────────────────────────────── */
  /* dfs(g, c, tr, opt): full forest over vertices in index order (or opt.order).
     Records d/f, parent, the classification of EVERY adjacency entry (tree/back/
     forward/cross for directed; tree/back for undirected, classified on the first
     encounter and skipped on the second), and — with opt.iterative — runs the
     explicit-stack version that keeps a per-vertex iterator, which reproduces the
     recursion's d/f exactly. Counter keys: discover, scans, maxStack. */
  function dfs(g, c, tr, opt) {
    const o = Object.assign({ order: null, iterative: false, roots: null }, opt || {});
    const n = g.n, d = new Array(n).fill(0), f = new Array(n).fill(0), parent = new Array(n).fill(-1);
    const colour = new Array(n).fill(0), cls = new Array(g.m).fill(null), root = new Array(n).fill(-1);
    const seenEdge = new Array(g.m).fill(false);
    let time = 0, maxStack = 0, depth = 0;
    const finishOrder = [], stack = [];
    const snap = (ev, u, v, id) => { if (tr) tr({ ev, u, v, id, d: d.slice(), f: f.slice(), colour: colour.slice(), stack: stack.slice(), cls: cls.slice(), parent: parent.slice(), time }); };
    const classify = (u, e) => {
      if (c) c.add("scans");                           // every adjacency entry examined: 2E undirected, E directed
      if (!g.directed) {
        if (seenEdge[e.id]) return null;               // second sight of an undirected edge: skip
        seenEdge[e.id] = true;
      }
      if (c) c.add("classified");                      // each edge classified exactly once: E
      const v = e.v;
      if (colour[v] === 0) { cls[e.id] = "tree"; return "tree"; }
      if (colour[v] === 1) { cls[e.id] = "back"; snap("back", u, v, e.id); return "back"; }
      cls[e.id] = d[u] < d[v] ? "forward" : "cross"; snap(cls[e.id], u, v, e.id); return cls[e.id];
    };
    function visit(u, r) {
      colour[u] = 1; d[u] = ++time; root[u] = r; stack.push(u); if (stack.length > maxStack) maxStack = stack.length;
      if (c) c.add("discover"); snap("discover", u, -1);
      for (const e of g.adj[u]) {
        const k = classify(u, e);
        if (k === "tree") { parent[e.v] = u; visit(e.v, r); }
      }
      colour[u] = 2; f[u] = ++time; stack.pop(); finishOrder.push(u); snap("finish", u, -1);
    }
    function visitIter(s, r) {
      const it = new Array(n).fill(0);
      colour[s] = 1; d[s] = ++time; root[s] = r; stack.push(s); if (c) c.add("discover"); snap("discover", s, -1);
      while (stack.length) {
        if (stack.length > maxStack) maxStack = stack.length;
        const u = stack[stack.length - 1];
        if (it[u] < g.adj[u].length) {
          const e = g.adj[u][it[u]++];
          const k = classify(u, e);
          if (k === "tree") { parent[e.v] = u; colour[e.v] = 1; d[e.v] = ++time; root[e.v] = r; stack.push(e.v); if (c) c.add("discover"); snap("discover", e.v, -1); }
        } else { colour[u] = 2; f[u] = ++time; stack.pop(); finishOrder.push(u); snap("finish", u, -1); }
      }
    }
    const order = o.order || Array.from({ length: n }, (_, i) => i);
    const roots = [];
    for (const s of order) if (colour[s] === 0) { roots.push(s); if (o.iterative) visitIter(s, s); else visit(s, s); }
    if (c) c.add("maxStack", maxStack);
    return { d, f, parent, cls, root, roots, finishOrder, maxStack, directed: g.directed };
  }
  /* the parenthesis theorem, checked on every pair: nested ⇔ ancestor, else disjoint */
  function isAncestor(res, a, v) { while (v !== -1) { if (v === a) return true; v = res.parent[v]; } return false; }
  function checkParenthesis(res) {
    const n = res.d.length; let nested = 0, disjoint = 0, bad = 0;
    for (let u = 0; u < n; u++) for (let v = u + 1; v < n; v++) {
      const inUV = res.d[u] < res.d[v] && res.f[v] < res.f[u];  // v inside u
      const inVU = res.d[v] < res.d[u] && res.f[u] < res.f[v];
      const dis = res.f[u] < res.d[v] || res.f[v] < res.d[u];
      const anc = isAncestor(res, u, v), anc2 = isAncestor(res, v, u);
      if (inUV && anc && !anc2) nested++;
      else if (inVU && anc2 && !anc) nested++;
      else if (dis && !anc && !anc2) disjoint++;
      else bad++;
    }
    return { ok: bad === 0, nested, disjoint, bad };
  }
  function classCounts(res) { const k = { tree: 0, back: 0, forward: 0, cross: 0 }; res.cls.forEach(x => { if (x) k[x]++; }); return k; }

  /* ── cycles ─────────────────────────────────────────────────────────────── */
  function hasCycleDirected(g, c) { const r = dfs(g, c); return { cyclic: r.cls.some(x => x === "back"), res: r }; }
  /* undirected: exclude the edge we arrived by (its id), NOT the parent vertex */
  function hasCycleUndirected(g, c, byVertex) {
    const colour = new Array(g.n).fill(0), parent = new Array(g.n).fill(-1), inEdge = new Array(g.n).fill(-1);
    let found = null;
    function visit(u) {
      colour[u] = 1;
      for (const e of g.adj[u]) {
        if (c) c.add("scans");
        const skip = byVertex ? (e.v === parent[u]) : (e.id === inEdge[u]);
        if (skip) continue;
        if (colour[e.v] === 1) { if (!found) found = [u, e.v]; continue; }
        if (colour[e.v] === 0) { parent[e.v] = u; inEdge[e.v] = e.id; visit(e.v); }
      }
      colour[u] = 2;
    }
    for (let s = 0; s < g.n; s++) if (colour[s] === 0) visit(s);
    return { cyclic: found !== null, witness: found };
  }
  function cycleByCounting(g) {                       // undirected: acyclic ⇔ m = n − (number of components)
    return g.m !== g.n - components(g).count;
  }

  /* ── connectivity ───────────────────────────────────────────────────────── */
  function components(g, c) {
    const comp = new Array(g.n).fill(-1); let k = 0;
    for (let s = 0; s < g.n; s++) {
      if (comp[s] >= 0) continue;
      const st = [s]; comp[s] = k;
      while (st.length) { const u = st.pop(); if (c) c.add("visits"); for (const e of g.adj[u]) { if (c) c.add("scans"); if (comp[e.v] < 0) { comp[e.v] = k; st.push(e.v); } } }
      k++;
    }
    return { comp, count: k };
  }
  function unionFind(n) {
    const p = Array.from({ length: n }, (_, i) => i), sz = new Array(n).fill(1);
    const find = x => { while (p[x] !== x) { p[x] = p[p[x]]; x = p[x]; } return x; };
    const union = (a, b) => { a = find(a); b = find(b); if (a === b) return false; if (sz[a] < sz[b]) { const t = a; a = b; b = t; } p[b] = a; sz[a] += sz[b]; return true; };
    return { find, union, sets: () => { let k = 0; for (let i = 0; i < n; i++) if (p[i] === i) k++; return k; } };
  }
  function componentsByUnionFind(g) { const uf = unionFind(g.n); for (const [u, v] of g.edges) uf.union(u, v); return uf.sets(); }
  function sameComp(a, b) { const n = a.length; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if ((a[i] === a[j]) !== (b[i] === b[j])) return false; return true; }

  /* ── topological order ──────────────────────────────────────────────────── */
  function topoDFS(g, c) {
    const r = dfs(g, c);
    if (r.cls.some(x => x === "back")) return { ok: false, order: null, res: r };
    return { ok: true, order: r.finishOrder.slice().reverse(), res: r };
  }
  function topoKahn(g, c, tr) {
    const indeg = new Array(g.n).fill(0); for (const [, v] of g.edges) indeg[v]++;
    const q = []; for (let v = 0; v < g.n; v++) if (indeg[v] === 0) q.push(v);
    const order = []; let head = 0;
    const snap = (ev, u) => { if (tr) tr({ ev, u, queue: q.slice(head), indeg: indeg.slice(), order: order.slice() }); };
    snap("start", -1);
    while (head < q.length) {
      const u = q[head++]; order.push(u); if (c) c.add("deq");
      for (const e of g.adj[u]) { if (c) c.add("scans"); if (--indeg[e.v] === 0) q.push(e.v); }
      snap("emit", u);
    }
    return { ok: order.length === g.n, order: order.length === g.n ? order : null, partial: order };
  }
  function checkTopo(g, order) {
    if (!order || order.length !== g.n || new Set(order).size !== g.n) return false;
    const pos = new Array(g.n); order.forEach((v, i) => { pos[v] = i; });
    return g.edges.every(([u, v]) => pos[u] < pos[v]);
  }
  /* number of topological orders, by DP over subsets (n ≤ 16) */
  function countTopo(g) {
    const n = g.n; if (n > 18) return null;
    const pred = new Array(n).fill(0); for (const [u, v] of g.edges) pred[v] |= (1 << u);
    const dp = new Float64Array(1 << n); dp[0] = 1;
    for (let S = 0; S < (1 << n); S++) {
      if (!dp[S]) continue;
      for (let v = 0; v < n; v++) if (!(S & (1 << v)) && (pred[v] & S) === pred[v]) dp[S | (1 << v)] += dp[S];
    }
    return dp[(1 << n) - 1];
  }
  function hamiltonianPathInOrder(g, order) {         // unique order ⇔ consecutive vertices joined by an edge
    for (let i = 0; i + 1 < order.length; i++) if (!g.adj[order[i]].some(e => e.v === order[i + 1])) return false;
    return true;
  }

  /* ── strongly connected components ──────────────────────────────────────── */
  function sccKosaraju(g, c, tr) {
    const first = dfs(g, c);                          // pass 1: finish times on G
    const gr = reverse(g);
    const order = first.finishOrder.slice().reverse();  // decreasing finish time
    const comp = new Array(g.n).fill(-1); let k = 0; const roots = [];
    const colour = new Array(g.n).fill(0);
    for (const s of order) {
      if (comp[s] >= 0) continue;
      roots.push(s);
      const st = [s]; comp[s] = k; colour[s] = 1;
      while (st.length) {
        const u = st.pop(); if (c) c.add("discover2");
        for (const e of gr.adj[u]) { if (c) c.add("scans2"); if (comp[e.v] < 0) { comp[e.v] = k; colour[e.v] = 1; st.push(e.v); } }
      }
      if (tr) tr({ ev: "component", root: s, k, comp: comp.slice() });
      k++;
    }
    return { comp, count: k, first, order, roots, reversed: gr };
  }
  function sccTarjan(g, c) {
    const idx = new Array(g.n).fill(-1), low = new Array(g.n).fill(0), on = new Array(g.n).fill(false);
    const st = []; let t = 0, k = 0; const comp = new Array(g.n).fill(-1); let maxStack = 0;
    function visit(u) {
      idx[u] = low[u] = t++; st.push(u); on[u] = true; if (st.length > maxStack) maxStack = st.length; if (c) c.add("discover");
      for (const e of g.adj[u]) {
        if (c) c.add("scans");
        if (idx[e.v] < 0) { visit(e.v); low[u] = Math.min(low[u], low[e.v]); }
        else if (on[e.v]) low[u] = Math.min(low[u], idx[e.v]);
      }
      if (low[u] === idx[u]) { let w; do { w = st.pop(); on[w] = false; comp[w] = k; } while (w !== u); k++; }
    }
    for (let s = 0; s < g.n; s++) if (idx[s] < 0) visit(s);
    if (c) c.add("maxStack", maxStack);
    return { comp, count: k, low, idx };
  }
  function transitiveClosure(g, c) {                  // Floyd–Warshall-style boolean closure, Θ(n³)
    const n = g.n, R = matrix(g).map(r => r.map(x => x > 0));
    for (let i = 0; i < n; i++) R[i][i] = true;
    for (let k = 0; k < n; k++) for (let i = 0; i < n; i++) if (R[i][k]) for (let j = 0; j < n; j++) { if (c) c.add("cells"); if (R[k][j]) R[i][j] = true; }
    return R;
  }
  function sccBrute(g) {
    const R = transitiveClosure(g), comp = new Array(g.n).fill(-1); let k = 0;
    for (let u = 0; u < g.n; u++) { if (comp[u] >= 0) continue; comp[u] = k; for (let v = u + 1; v < g.n; v++) if (R[u][v] && R[v][u]) comp[v] = k; k++; }
    return { comp, count: k };
  }
  function condensation(g, comp, count) {
    const seen = new Set(), edges = [];
    for (const [u, v] of g.edges) { if (comp[u] !== comp[v]) { const key = comp[u] * count + comp[v]; if (!seen.has(key)) { seen.add(key); edges.push([comp[u], comp[v]]); } } }
    return graph(count, edges, true);
  }

  /* ── articulation points and bridges (low-link on an undirected DFS) ─────── */
  function cutVerticesAndBridges(g, c) {
    const n = g.n, d = new Array(n).fill(0), low = new Array(n).fill(0), parent = new Array(n).fill(-1), inEdge = new Array(n).fill(-1);
    const isCut = new Array(n).fill(false), bridges = []; let t = 0;
    function visit(u) {
      d[u] = low[u] = ++t; let children = 0; if (c) c.add("discover");
      for (const e of g.adj[u]) {
        if (c) c.add("scans");
        if (e.id === inEdge[u]) continue;
        if (d[e.v] === 0) {
          parent[e.v] = u; inEdge[e.v] = e.id; children++; visit(e.v);
          low[u] = Math.min(low[u], low[e.v]);
          if (parent[u] !== -1 && low[e.v] >= d[u]) isCut[u] = true;
          if (low[e.v] > d[u]) bridges.push([u, e.v, e.id]);
        } else low[u] = Math.min(low[u], d[e.v]);
      }
      if (parent[u] === -1 && children >= 2) isCut[u] = true;
    }
    for (let s = 0; s < n; s++) if (d[s] === 0) visit(s);
    return { isCut, bridges, d, low, parent };
  }
  function cutVerticesBrute(g) {                      // delete each vertex; does the component count (among the rest) rise?
    const base = components(g).count, out = new Array(g.n).fill(false);
    for (let x = 0; x < g.n; x++) {
      const keep = g.edges.filter(([u, v]) => u !== x && v !== x);
      const h = graph(g.n, keep, false);
      const cnt = components(h).count - 1;            // x itself is now isolated: subtract it
      out[x] = cnt > base;
    }
    return out;
  }
  function bridgesBrute(g) {
    const base = components(g).count, out = [];
    g.edges.forEach(([u, v], id) => { const h = graph(g.n, g.edges.filter((_, j) => j !== id), false); if (components(h).count > base) out.push(id); });
    return out;
  }

  /* ── representation space model (bytes; stated assumptions) ─────────────── */
  /* matrix: one entry per ordered pair — bits/8 for a bit matrix, 1 byte per cell for a byte matrix.
     adjacency list (dynamic arrays of ints): per vertex a 24-byte header + 4 bytes per entry (2E entries undirected).
     CSR: (n+1) offsets × 4 + entries × 4.  edge list: 8 bytes per edge. */
  function spaceModel(n, m, directed, opt) {
    const o = Object.assign({ cellBytes: 1, wordBytes: 4, listHeader: 24 }, opt || {});
    const entries = directed ? m : 2 * m;
    return {
      matrixBits: n * n / 8,
      matrix: n * n * o.cellBytes,
      list: n * o.listHeader + entries * o.wordBytes,
      csr: (n + 1) * o.wordBytes + entries * o.wordBytes,
      edgeList: m * 2 * o.wordBytes
    };
  }

  /* ── traversal cost on the matrix (for the Θ(V²) comparison) ─────────────── */
  function bfsMatrix(A, s, c) {
    const n = A.length, dist = new Array(n).fill(-1); dist[s] = 0; const q = [s]; let head = 0;
    while (head < q.length) { const u = q[head++]; if (c) c.add("deq"); for (let v = 0; v < n; v++) { if (c) c.add("cells"); if (A[u][v] && dist[v] < 0) { dist[v] = dist[u] + 1; q.push(v); } } }
    return dist;
  }

  return {
    L, lbl, graph, sortAdj, matrix, csr, reverse, degrees,
    random, randomDAG, path, cycle, star, complete, grid, binaryTree,
    bfs, bfsPath, layers, distByMatrixPower, distByRelaxation, sameArr,
    bipartite, checkColouring, isCycle,
    dfs, isAncestor, checkParenthesis, classCounts,
    hasCycleDirected, hasCycleUndirected, cycleByCounting,
    components, unionFind, componentsByUnionFind, sameComp,
    topoDFS, topoKahn, checkTopo, countTopo, hamiltonianPathInOrder,
    sccKosaraju, sccTarjan, transitiveClosure, sccBrute, condensation,
    cutVerticesAndBridges, cutVerticesBrute, bridgesBrute,
    spaceModel, bfsMatrix
  };
})();

if (typeof module !== "undefined") module.exports = { GR };
/* END-GR */

/* ═══════════════════════════════════════════════════════════════════════════
   GD — drawing
   ═══════════════════════════════════════════════════════════════════════════ */
const GD = (function () {
  if (typeof d3 === "undefined") return null;
  const L = GR.lbl;

  /* a small deterministic force layout (seeded init, fixed iteration count) so a
     random graph looks the same on every load and in the headless harness */
  function forceLayout(g, W, H, seed, iters) {
    const r = AL.rng(seed || 1), n = g.n, pos = [];
    for (let i = 0; i < n; i++) pos.push({ x: W * (0.15 + 0.7 * r()), y: H * (0.15 + 0.7 * r()) });
    const k = Math.sqrt(W * H / Math.max(1, n)) * 0.9;
    let temp = W / 8;
    for (let it = 0; it < (iters || 250); it++) {
      const disp = pos.map(() => ({ x: 0, y: 0 }));
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        let dx = pos[i].x - pos[j].x, dy = pos[i].y - pos[j].y, d = Math.sqrt(dx * dx + dy * dy) || 0.01;
        const f = k * k / d; dx /= d; dy /= d;
        disp[i].x += dx * f; disp[i].y += dy * f; disp[j].x -= dx * f; disp[j].y -= dy * f;
      }
      for (const [u, v] of g.edges) {
        if (u === v) continue;
        let dx = pos[u].x - pos[v].x, dy = pos[u].y - pos[v].y, d = Math.sqrt(dx * dx + dy * dy) || 0.01;
        const f = d * d / k; dx /= d; dy /= d;
        disp[u].x -= dx * f; disp[u].y -= dy * f; disp[v].x += dx * f; disp[v].y += dy * f;
      }
      for (let i = 0; i < n; i++) {
        // gravity to the centre keeps disconnected pieces on the canvas
        disp[i].x += (W / 2 - pos[i].x) * 0.05; disp[i].y += (H / 2 - pos[i].y) * 0.05;
        const d = Math.sqrt(disp[i].x * disp[i].x + disp[i].y * disp[i].y) || 0.01, s = Math.min(d, temp) / d;
        pos[i].x = AL.clamp(pos[i].x + disp[i].x * s, 18, W - 18); pos[i].y = AL.clamp(pos[i].y + disp[i].y * s, 18, H - 18);
      }
      temp *= 0.97;
    }
    return pos;
  }
  function circleLayout(n, cx, cy, R) { return Array.from({ length: n }, (_, i) => ({ x: cx + R * Math.cos(-Math.PI / 2 + 2 * Math.PI * i / n), y: cy + R * Math.sin(-Math.PI / 2 + 2 * Math.PI * i / n) })); }
  function gridLayout(w, h, x0, y0, cell) { const p = []; for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) p.push({ x: x0 + x * cell, y: y0 + y * cell }); return p; }

  /* drawGraph(sel, g, pos, opt): edges first, nodes on top. Hooks receive (id, u, v)
     for edges and v for nodes. A directed pair u→v, v→u is drawn as two curves. */
  function drawGraph(sel, g, pos, opt) {
    const o = Object.assign({ x: 0, y: 0, r: 14, fontSize: 12, fill: null, stroke: null, sw: null, text: null, textFill: null, label: null, labelAbove: null,
      edgeColor: null, edgeW: null, edgeDash: null, edgeLabel: null, edgeOpacity: null, hideEdge: null, arrow: g.directed }, opt || {});
    const gg = sel.append("g").attr("transform", `translate(${o.x},${o.y})`);
    const has = (u, v) => g.edges.some(e => e[0] === u && e[1] === v);
    g.edges.forEach(([u, v], id) => {
      if (o.hideEdge && o.hideEdge(id, u, v)) return;
      const col = o.edgeColor ? (o.edgeColor(id, u, v) || AC.line) : AC.line;
      const w = o.edgeW ? (o.edgeW(id, u, v) || 1.6) : 1.6;
      const dash = o.edgeDash ? o.edgeDash(id, u, v) : null;
      const op = o.edgeOpacity ? o.edgeOpacity(id, u, v) : 1;
      const P = pos[u], Q = pos[v];
      if (u === v) {                                  // self-loop: a small loop above the node
        const path = gg.append("path").attr("d", `M${P.x - 6},${P.y - o.r + 2} C${P.x - 26},${P.y - o.r - 30} ${P.x + 26},${P.y - o.r - 30} ${P.x + 6},${P.y - o.r + 2}`)
          .attr("fill", "none").attr("stroke", col).attr("stroke-width", w).attr("opacity", op);
        if (dash) path.attr("stroke-dasharray", dash);
        if (o.arrow) head(gg, P.x + 6, P.y - o.r + 2, Math.atan2(30, -20), col, op);
        return;
      }
      const dx = Q.x - P.x, dy = Q.y - P.y, d = Math.sqrt(dx * dx + dy * dy) || 1, ux = dx / d, uy = dy / d;
      const recip = g.directed && has(v, u);
      const x1 = P.x + ux * o.r, y1 = P.y + uy * o.r, x2 = Q.x - ux * (o.r + (o.arrow ? 2 : 0)), y2 = Q.y - uy * (o.r + (o.arrow ? 2 : 0));
      let path;
      if (recip) {                                    // bow each direction to its own side
        const mx = (x1 + x2) / 2 - uy * 14, my = (y1 + y2) / 2 + ux * 14;
        path = gg.append("path").attr("d", `M${x1},${y1} Q${mx},${my} ${x2},${y2}`).attr("fill", "none");
        if (o.arrow) head(gg, x2, y2, Math.atan2(y2 - my, x2 - mx), col, op);
        if (o.edgeLabel) { const t = o.edgeLabel(id, u, v); if (t) gg.append("text").attr("x", mx).attr("y", my + 4).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", col).text(t); }
      } else {
        path = gg.append("line").attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2);
        if (o.arrow) head(gg, x2, y2, Math.atan2(dy, dx), col, op);
        if (o.edgeLabel) { const t = o.edgeLabel(id, u, v); if (t) gg.append("text").attr("x", (x1 + x2) / 2 - uy * 8).attr("y", (y1 + y2) / 2 + ux * 8 + 3).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", col).text(t); }
      }
      path.attr("stroke", col).attr("stroke-width", w).attr("opacity", op);
      if (dash) path.attr("stroke-dasharray", dash);
    });
    for (let v = 0; v < g.n; v++) {
      const P = pos[v];
      gg.append("circle").attr("cx", P.x).attr("cy", P.y).attr("r", o.r)
        .attr("fill", o.fill ? (o.fill(v) || AC.panel2) : AC.panel2)
        .attr("stroke", o.stroke ? (o.stroke(v) || AC.line) : AC.line).attr("stroke-width", o.sw ? (o.sw(v) || 1.5) : 1.5);
      gg.append("text").attr("x", P.x).attr("y", P.y + 4).attr("text-anchor", "middle").attr("font-size", o.fontSize)
        .attr("fill", o.textFill ? (o.textFill(v) || AC.ink) : AC.ink).text(o.text ? o.text(v) : L(v));
      if (o.label) { const t = o.label(v); if (t !== null && t !== undefined && t !== "") gg.append("text").attr("x", P.x).attr("y", P.y + o.r + 11).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", AC.muted).text(t); }
      if (o.labelAbove) { const t = o.labelAbove(v); if (t) gg.append("text").attr("x", P.x).attr("y", P.y - o.r - 4).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", AC.a2).text(t); }
    }
    return { g: gg };
  }
  function head(gg, x, y, a, col, op) {
    const h = 7;
    gg.append("path").attr("d", `M${x},${y} L${x - h * Math.cos(a - 0.42)},${y - h * Math.sin(a - 0.42)} L${x - h * Math.cos(a + 0.42)},${y - h * Math.sin(a + 0.42)} Z`).attr("fill", col).attr("opacity", op === undefined ? 1 : op);
  }

  /* an n × n matrix as a grid of cells with row/column letters */
  function matrixGrid(g, A, opt) {
    const o = Object.assign({ x: 0, y: 0, cell: 22, mark: null, text: true }, opt || {});
    const gg = g.append("g").attr("transform", `translate(${o.x},${o.y})`), n = A.length;
    for (let j = 0; j < n; j++) { gg.append("text").attr("x", (j + 0.5) * o.cell).attr("y", -5).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.muted).text(L(j)); gg.append("text").attr("x", -6).attr("y", (j + 0.5) * o.cell + 4).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.muted).text(L(j)); }
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const f = o.mark ? o.mark(i, j, A[i][j]) : null;
      gg.append("rect").attr("x", j * o.cell).attr("y", i * o.cell).attr("width", o.cell - 1).attr("height", o.cell - 1).attr("rx", 2)
        .attr("fill", f || (A[i][j] ? AC.panel2 : AC.bg)).attr("stroke", AC.line);
      if (o.text) gg.append("text").attr("x", (j + 0.5) * o.cell).attr("y", (i + 0.5) * o.cell + 4).attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", A[i][j] ? AC.ink : AC.line).text(A[i][j]);
    }
    return gg;
  }
  /* adjacency lists as rows of boxes */
  function adjRows(g, graph, opt) {
    const o = Object.assign({ x: 0, y: 0, w: 22, h: 20, gap: 2, rowH: 24, mark: null }, opt || {});
    const gg = g.append("g").attr("transform", `translate(${o.x},${o.y})`);
    graph.adj.forEach((a, u) => {
      gg.append("text").attr("x", -8).attr("y", u * o.rowH + o.h / 2 + 4).attr("text-anchor", "end").attr("font-size", 11).attr("fill", AC.muted).text(L(u) + ":");
      a.forEach((e, k) => {
        const f = o.mark ? o.mark(u, e, k) : null;
        gg.append("rect").attr("x", k * (o.w + o.gap)).attr("y", u * o.rowH).attr("width", o.w).attr("height", o.h).attr("rx", 3).attr("fill", f || AC.panel2).attr("stroke", AC.line);
        gg.append("text").attr("x", k * (o.w + o.gap) + o.w / 2).attr("y", u * o.rowH + o.h / 2 + 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.ink).text(L(e.v));
        if (k < a.length - 1) gg.append("text").attr("x", (k + 1) * (o.w + o.gap) - o.gap / 2 - 1).attr("y", u * o.rowH + o.h / 2 + 4).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", AC.muted).text("");
      });
      if (!a.length) gg.append("text").attr("x", 2).attr("y", u * o.rowH + o.h / 2 + 4).attr("font-size", 10).attr("fill", AC.muted).text("∅");
    });
    return gg;
  }
  /* nested intervals [d, f] as brackets on a 1 … 2n timeline */
  function intervals(g, res, opt) {
    const o = Object.assign({ x: 0, y: 0, w: 600, rowH: 16, mark: null }, opt || {});
    const gg = g.append("g").attr("transform", `translate(${o.x},${o.y})`), n = res.d.length, T = 2 * n;
    const x = t => (t - 0.5) / T * o.w;
    // depth by ancestry gives the row
    const depth = v => { let k = 0; for (let p = res.parent[v]; p !== -1; p = res.parent[p]) k++; return k; };
    for (let t = 1; t <= T; t++) gg.append("text").attr("x", x(t)).attr("y", -4).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", AC.muted).text(t);
    for (let v = 0; v < n; v++) {
      const dpt = depth(v), y = dpt * o.rowH, col = o.mark ? (o.mark(v) || AC.accent) : AC.accent;
      gg.append("rect").attr("x", x(res.d[v]) - 4).attr("y", y).attr("width", x(res.f[v]) - x(res.d[v]) + 8).attr("height", o.rowH - 3).attr("rx", 3).attr("fill", col).attr("opacity", 0.22).attr("stroke", col);
      gg.append("text").attr("x", x(res.d[v]) + 2).attr("y", y + o.rowH - 6).attr("font-size", 10).attr("fill", AC.ink).text(`${L(v)} ${res.d[v]}/${res.f[v]}`);
    }
    return gg;
  }
  function note(g, x, y, txt, color, size) { return g.append("text").attr("x", x).attr("y", y).attr("font-size", size || 11.5).attr("fill", color || AC.ink).text(txt); }
  function lines(g, x, y, arr, opt) {
    const o = Object.assign({ lh: 17, size: 11, hi: [] }, opt || {});
    arr.forEach((t, i) => g.append("text").attr("x", x).attr("y", y + i * o.lh).attr("font-size", o.size).attr("fill", (o.hi.indexOf(i) >= 0 || (o.hiFn && o.hiFn(t, i))) ? AC.ink : AC.muted).text(t));
  }
  return { forceLayout, circleLayout, gridLayout, drawGraph, matrixGrid, adjRows, intervals, note, lines };
})();

/* ═══════════════════════════════════════════════════════════════════════════
   the hand examples (shared by several figures; every number in the page's
   prose was computed from these same objects by script before being written)
   ═══════════════════════════════════════════════════════════════════════════ */
const GX = (function () {
  const id = s => s.charCodeAt(0) - 97;
  const E = str => str.split(" ").map(t => [id(t[0]), id(t[1])]);
  const G1 = GR.graph(8, E("ab ac bd cd ce df ef eh fg gh"), false);
  const G1pos = [{ x: 50, y: 150 }, { x: 160, y: 70 }, { x: 160, y: 230 }, { x: 280, y: 110 }, { x: 280, y: 190 }, { x: 400, y: 150 }, { x: 510, y: 90 }, { x: 510, y: 210 }];
  const G2 = GR.graph(8, E("ab bc ca ad de ed be fg gh hf gd"), true);
  const G2pos = [{ x: 70, y: 80 }, { x: 190, y: 80 }, { x: 130, y: 190 }, { x: 320, y: 190 }, { x: 320, y: 80 }, { x: 470, y: 80 }, { x: 590, y: 80 }, { x: 530, y: 190 }];
  const D = GR.graph(8, E("ab ac bd cd de cf fe eg"), true);
  const Dpos = [{ x: 50, y: 150 }, { x: 160, y: 80 }, { x: 160, y: 220 }, { x: 280, y: 120 }, { x: 400, y: 170 }, { x: 280, y: 260 }, { x: 520, y: 170 }, { x: 560, y: 60 }];
  const H = GR.graph(10, E("ab bc ca cd de ef fg gd gh hi hj ij"), false);
  const Hpos = [{ x: 50, y: 100 }, { x: 50, y: 200 }, { x: 130, y: 150 }, { x: 220, y: 150 }, { x: 280, y: 80 }, { x: 370, y: 80 }, { x: 330, y: 220 }, { x: 440, y: 220 }, { x: 520, y: 150 }, { x: 520, y: 290 }];
  const C6 = str => GR.graph(6, E("ab bc cd de ef fa" + (str ? " " + str : "")), false);
  return { E, G1, G1pos, G2, G2pos, D, Dpos, H, Hpos, C6 };
})();
/* END-GX */

/* ═══════════════════════════════════════════════════════════════════════════
   figures
   ═══════════════════════════════════════════════════════════════════════════ */
(function () {
  if (typeof document === "undefined" || typeof d3 === "undefined") return;
  const $ = id => document.getElementById(id);
  const fig = (id, fn) => { if ($(id)) fn(); };
  const txt = (id, s) => { const el = $(id); if (el) el.textContent = s; };
  const on = (id, ev, fn) => { const el = $(id); if (el) el.addEventListener(ev, fn); };
  const val = id => { const el = $(id); return el ? el.value : null; };
  const chk = id => { const el = $(id); return el ? el.checked : false; };
  const f1 = x => (+x).toFixed(1), f2 = x => (+x).toFixed(2);
  const ok = b => b ? "✓" : "✗";
  const L = GR.lbl, LL = a => a.map(L).join(" ");
  const CLS = { tree: AC.accent, back: AC.bad, forward: AC.a2, cross: AC.violet };
  const LAYER = [AC.a2, AC.accent, AC.good, AC.violet, AC.teal, AC.rose, "#f0abfc", "#fde68a", "#93c5fd", "#86efac"];
  const bytes = b => b >= 1e9 ? f2(b / 1e9) + " GB" : b >= 1e6 ? f1(b / 1e6) + " MB" : b >= 1e3 ? f1(b / 1e3) + " kB" : b + " B";

  /* ── F01 degrees and the handshake lemma, measured ────────────────────── */
  fig("deg-svg", () => {
    function draw() {
      const kind = val("deg-kind") || "undirected", directed = kind === "directed";
      const n = 12, m = 18, g = GR.random(n, m, directed, AL.rng(5));
      const dg = GR.degrees(g), c = AL.counter();
      // count the adjacency entries by walking every list — that IS Σ deg
      let entries = 0; g.adj.forEach(a => { entries += a.length; c.add("entries", a.length); });
      const F = AL.frame("#deg-svg", 760, 300, { l: 20, r: 250, t: 14, b: 30 });
      const pos = GD.forceLayout(g, 300, 260, 5);
      GD.drawGraph(F.g, g, pos, { r: 12, label: v => directed ? `${dg.inn[v]}↓ ${dg.out[v]}↑` : `deg ${dg.deg[v]}` });
      // degree bars
      const bx = 330, by = 20, bw = 150, bh = 220;
      const vals = directed ? dg.out : dg.deg, mx = Math.max(...vals, 1);
      const y = d3.scaleBand().domain(d3.range(n)).range([by, by + bh]).padding(0.2), x = d3.scaleLinear().domain([0, mx]).range([0, bw]);
      vals.forEach((d, v) => { F.g.append("rect").attr("x", bx).attr("y", y(v)).attr("width", x(d)).attr("height", y.bandwidth()).attr("fill", AC.accent); F.g.append("text").attr("x", bx - 5).attr("y", y(v) + y.bandwidth() / 2 + 4).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.muted).text(L(v)); F.g.append("text").attr("x", bx + x(d) + 4).attr("y", y(v) + y.bandwidth() / 2 + 4).attr("font-size", 10).attr("fill", AC.ink).text(d); });
      F.g.append("text").attr("x", bx).attr("y", by - 6).attr("font-size", 10.5).attr("fill", AC.muted).text(directed ? "out-degree of each vertex" : "degree of each vertex");
      const sumIn = dg.inn.reduce((a, b) => a + b, 0), sumOut = dg.out.reduce((a, b) => a + b, 0), sumDeg = dg.deg.reduce((a, b) => a + b, 0);
      const lines = directed
        ? [`V = ${n}, E = ${m} (directed, simple)`, ``, `Σ out-degree = ${sumOut}`, `Σ in-degree  = ${sumIn}`, `E            = ${m}   ${ok(sumIn === m && sumOut === m)}`, ``, `adjacency-list entries walked: ${c.get("entries")}`, `= E   ${ok(entries === m)}`, ``, `max possible E = V(V−1) = ${n * (n - 1)}`, `density E / V(V−1) = ${f2(m / (n * (n - 1)))}`]
        : [`V = ${n}, E = ${m} (undirected, simple)`, ``, `Σ deg(v) = ${sumDeg}`, `2·E      = ${2 * m}   ${ok(sumDeg === 2 * m)}`, ``, `adjacency-list entries walked: ${c.get("entries")}`, `= 2E   ${ok(entries === 2 * m)}`, ``, `odd-degree vertices: ${dg.deg.filter(d => d % 2).length} (always even)`, `max possible E = V(V−1)/2 = ${n * (n - 1) / 2}`, `density E / (V(V−1)/2) = ${f2(m / (n * (n - 1) / 2))}`];
      GD.lines(F.g, 520, 20, lines, { hi: [2, 3, 4, 6, 7], size: 10.5, lh: 16 });
      txt("deg-readout", directed ? `Directed graph, V = ${n}, E = ${m}: Σ in-degree = ${sumIn}, Σ out-degree = ${sumOut}, both equal to E: ${ok(sumIn === m && sumOut === m)}. Walking every adjacency list touched ${entries} entries = E.` : `Undirected graph, V = ${n}, E = ${m}: Σ deg(v) = ${sumDeg} = 2E: ${ok(sumDeg === 2 * m)}. Walking every adjacency list touched ${entries} entries = 2E; ${dg.deg.filter(d => d % 2).length} vertices have odd degree (an even number, as the lemma forces).`);
    }
    on("deg-kind", "change", draw); draw();
  });

  /* ── F02 one graph, four representations, one edge lit in each ────────── */
  fig("rep-svg", () => {
    function draw() {
      const which = val("rep-graph") || "G2", g = which === "G1" ? GX.G1 : GX.G2, pos = which === "G1" ? GX.G1pos : GX.G2pos;
      const rep = val("rep-kind") || "list";
      const sel = $("rep-edge"); if (sel && sel.dataset.for !== which) { sel.innerHTML = ""; g.edges.forEach(([u, v], id) => { const o = document.createElement("option"); o.value = id; o.textContent = L(u) + (g.directed ? "→" : "–") + L(v); sel.appendChild(o); }); sel.dataset.for = which; sel.value = which === "G1" ? "4" : "3"; }
      const eid = +(val("rep-edge") || 0), [eu, ev] = g.edges[eid];
      const c = AL.counter();
      const F = AL.frame("#rep-svg", 760, 320, { l: 10, r: 10, t: 14, b: 10 });
      GD.drawGraph(F.g, g, pos.map(p => ({ x: p.x * 0.5, y: p.y * 0.95 + 20 })), { r: 12, edgeColor: (id) => id === eid ? AC.a2 : null, edgeW: id => id === eid ? 3 : null, fill: v => (v === eu || v === ev) ? "#3b2f14" : null, stroke: v => (v === eu || v === ev) ? AC.a2 : null });
      GD.note(F.g, 10, 12, `${which === "G1" ? "undirected" : "directed"} · V = ${g.n}, E = ${g.m}`, AC.muted, 10.5);
      const X = 330;
      const hit = (u, v) => (u === eu && v === ev) || (!g.directed && u === ev && v === eu);
      if (rep === "matrix") {
        const A = GR.matrix(g, c);
        GD.matrixGrid(F.g, A, { x: X + 20, y: 30, cell: 24, mark: (i, j, a) => hit(i, j) ? AC.a2 : null });
        let ones = 0; A.forEach(r => r.forEach(x => { ones += x; }));
        GD.lines(F.g, X + 225, 40, [`adjacency matrix`, `V² = ${g.n * g.n} cells`, `non-zero cells: ${ones}`, `= ${g.directed ? "E" : "2E"} = ${g.directed ? g.m : 2 * g.m}   ${ok(ones === (g.directed ? g.m : 2 * g.m))}`, ``, `edge ${L(eu)}${g.directed ? "→" : "–"}${L(ev)}: cell [${L(eu)}][${L(ev)}]${g.directed ? "" : ", [" + L(ev) + "][" + L(eu) + "]"}`, `edge test: 1 cell read`, `neighbours of ${L(eu)}: row ${L(eu)},`, `  V = ${g.n} cells scanned`, `cells written to build: ${c.get("cells")}`], { hi: [0, 5, 6, 7, 8], size: 10.5, lh: 16 });
      } else if (rep === "list") {
        GD.adjRows(F.g, g, { x: X + 30, y: 30, rowH: 28, mark: (u, e) => (u === eu && e.v === ev) || (!g.directed && u === ev && e.v === eu) ? AC.a2 : null });
        let entries = 0; g.adj.forEach(a => { entries += a.length; });
        GD.lines(F.g, X + 200, 40, [`adjacency lists`, `V = ${g.n} headers`, `entries: ${entries} = ${g.directed ? "E" : "2E"}   ${ok(entries === (g.directed ? g.m : 2 * g.m))}`, ``, `edge ${L(eu)}${g.directed ? "→" : "–"}${L(ev)}: in list ${L(eu)}${g.directed ? "" : " and list " + L(ev)}`, `edge test: scan list ${L(eu)}, ≤ ${g.adj[eu].length} entries`, `neighbours of ${L(eu)}: ${g.adj[eu].length} entries = deg`, `insertion order kept`], { hi: [0, 4, 5, 6], size: 10.5, lh: 16 });
      } else if (rep === "edgelist") {
        const rows = g.edges.map(([u, v]) => L(u) + (g.directed ? "→" : "–") + L(v));
        rows.forEach((t, i) => { const col = i % 2, row = Math.floor(i / 2); F.g.append("rect").attr("x", X + 30 + col * 70).attr("y", 30 + row * 26).attr("width", 60).attr("height", 22).attr("rx", 3).attr("fill", i === eid ? AC.a2 : AC.panel2).attr("stroke", AC.line); F.g.append("text").attr("x", X + 60 + col * 70).attr("y", 45 + row * 26).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.ink).text(t); F.g.append("text").attr("x", X + 22 + col * 70).attr("y", 45 + row * 26).attr("text-anchor", "end").attr("font-size", 9).attr("fill", AC.muted).text(i); });
        GD.lines(F.g, X + 200, 40, [`edge list`, `E = ${g.m} pairs, nothing per vertex`, ``, `edge ${L(eu)}${g.directed ? "→" : "–"}${L(ev)}: entry ${eid}`, `edge test: scan all E = ${g.m}`, `neighbours of ${L(eu)}: scan all E = ${g.m}`, `cheapest to build and to store;`, `a sort by (u, v) turns it into CSR`], { hi: [0, 3, 4, 5], size: 10.5, lh: 16 });
      } else {
        const C = GR.csr(g);
        AL.row(F.g, C.offsets, { x: X + 30, y: 40, w: 26, h: 22, gap: 2, label: "offsets", fontSize: 11 });
        const start = C.offsets[eu], end = C.offsets[eu + 1];
        let k = -1; for (let i = start; i < end; i++) if (C.targets[i] === ev) { k = i; break; }
        AL.row(F.g, C.targets.map(L), { x: X + 30, y: 110, w: 18, h: 22, gap: 1, label: "targets", fontSize: 10, mark: i => i === k ? AC.a2 : (i >= start && i < end ? "#243044" : null) });
        GD.lines(F.g, X + 30, 175, [`compressed sparse row (CSR)`, `offsets: V + 1 = ${g.n + 1} ints;  targets: ${C.targets.length} ints = ${g.directed ? "E" : "2E"}   ${ok(C.targets.length === (g.directed ? g.m : 2 * g.m))}`, `neighbours of ${L(eu)} = targets[offsets[${L(eu)}] … offsets[${L(eu)}+1])`, `  = targets[${start} … ${end}): ${end - start} contiguous ints`, `edge ${L(eu)}${g.directed ? "→" : "–"}${L(ev)} is targets[${k}]; edge test: scan that slice`, `two flat int arrays, no pointers; static once built`], { hi: [0, 2, 3, 4], size: 10.5, lh: 16 });
      }
      txt("rep-readout", `${which} (${g.directed ? "directed" : "undirected"}, V = ${g.n}, E = ${g.m}) as ${rep === "matrix" ? "an adjacency matrix" : rep === "list" ? "adjacency lists" : rep === "edgelist" ? "an edge list" : "CSR"}, with edge ${L(eu)}${g.directed ? "→" : "–"}${L(ev)} highlighted. Matrix: ${g.n * g.n} cells of which ${g.directed ? g.m : 2 * g.m} are non-zero; lists/CSR: ${g.directed ? g.m : 2 * g.m} entries (${g.directed ? "E" : "2E"}); edge list: ${g.m} pairs. Counts measured on the built structure.`);
    }
    ["rep-graph", "rep-kind", "rep-edge"].forEach(id => on(id, "change", draw)); draw();
  });

  /* ── F03 space against density, in the stated byte model ──────────────── */
  fig("space-svg", () => {
    function draw() {
      const n = +(val("space-n") || 1000), maxM = n * (n - 1) / 2;
      const ms = AL.linspace(1, Math.log10(maxM), 60).map(x => Math.round(Math.pow(10, x)));
      const series = [
        { k: "matrix", label: "adjacency matrix, 1 byte per cell", color: AC.bad },
        { k: "matrixBits", label: "adjacency matrix, 1 bit per cell", color: AC.rose },
        { k: "list", label: "adjacency lists (24 B header + 4 B per entry)", color: AC.accent },
        { k: "csr", label: "CSR (4 B per offset + 4 B per entry)", color: AC.good },
        { k: "edgeList", label: "edge list (8 B per edge)", color: AC.a2 }
      ];
      const rows = ms.map(m => Object.assign({ m }, GR.spaceModel(n, m, false)));
      const F = AL.frame("#space-svg", 760, 320, { l: 60, r: 250, t: 16, b: 40 });
      const x = d3.scaleLog().domain([1, maxM]).range([0, F.iw]), y = d3.scaleLog().domain([100, Math.max(n * n, 8 * maxM) * 1.5]).range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 5); AL.axisB(F.g, x, F.ih, 5, "E (edges, log)", d3.format("~s")); AL.axisL(F.g, y, 5, "bytes (log)", d3.format("~s"));
      series.forEach(s => { F.g.append("path").datum(rows).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", 2).attr("d", d3.line().x(d => x(d.m)).y(d => y(Math.max(100, d[s.k])))); });
      // crossovers, by scanning the same model
      let mByte = null, mBit = null; for (let m = 0; m <= maxM; m += Math.max(1, Math.floor(maxM / 4000))) { const s = GR.spaceModel(n, m, false); if (mByte === null && s.matrix < s.list) mByte = m; if (mBit === null && s.matrixBits < s.list) mBit = m; }
      [[mByte, AC.bad], [mBit, AC.rose]].forEach(([m, col]) => { if (m) F.g.append("line").attr("x1", x(m)).attr("x2", x(m)).attr("y1", 0).attr("y2", F.ih).attr("stroke", col).attr("stroke-dasharray", "3,3"); });
      AL.legend(F.g, series, 8, 14, { gap: 15 });
      const big = GR.spaceModel(1e6, 1e7, false), bigD = GR.spaceModel(1e6, 1e7, true);
      GD.lines(F.g, F.iw + 12, 8, [`V = ${d3.format(",")(n)}, undirected,`, `E from 1 to V(V−1)/2 = ${d3.format(",")(maxM)}`, `byte matrix beats lists once`, `  E ≥ ${mByte === null ? "never" : d3.format(",")(mByte)}  (density ${mByte === null ? "—" : (mByte / maxM).toFixed(3)})`, `bit matrix beats lists once`, `  E ≥ ${mBit === null ? "never" : d3.format(",")(mBit)}  (density ${mBit === null ? "—" : (mBit / maxM).toFixed(3)})`, ``, `V = 10⁶, E = 10·V, undirected:`, `  byte matrix  ${bytes(big.matrix)}`, `  bit matrix   ${bytes(big.matrixBits)}`, `  adj. lists   ${bytes(big.list)}`, `  CSR          ${bytes(big.csr)}`, `  edge list    ${bytes(big.edgeList)}`, `  (directed: lists ${bytes(bigD.list)}, CSR ${bytes(bigD.csr)})`], { hi: [2, 3, 4, 5, 8, 9, 10, 11, 12], size: 10.5, lh: 15.5 });
      txt("space-readout", `Byte model: matrix V² cells (1 byte or 1 bit each); lists 24-byte header per vertex plus 4 bytes per adjacency entry (2E entries, undirected); CSR 4 bytes per offset and per entry; edge list 8 bytes per edge. For V = ${d3.format(",")(n)} the byte matrix is smaller than the lists from E = ${mByte === null ? "never" : d3.format(",")(mByte)} (density ${mByte === null ? "—" : (mByte / maxM).toFixed(3)}) and the bit matrix from E = ${mBit === null ? "never" : d3.format(",")(mBit)} (density ${mBit === null ? "—" : (mBit / maxM).toFixed(3)}). At V = 10⁶ and E = 10⁷: byte matrix ${bytes(big.matrix)}, bit matrix ${bytes(big.matrixBits)}, lists ${bytes(big.list)}, CSR ${bytes(big.csr)}, edge list ${bytes(big.edgeList)}.`);
    }
    on("space-n", "change", draw); draw();
  });

  /* ── F04 traversal cost: adjacency lists against the matrix, measured ──── */
  fig("trav-svg", () => {
    const n = 64, maxM = n * (n - 1) / 2, r = AL.rng(3);
    const ms = [64, 128, 256, 512, 1024, 1536, 2016].map(m => Math.min(m, maxM));
    const rows = ms.map(m => { const g = GR.random(n, m, false, r), A = GR.matrix(g), c1 = AL.counter(), c2 = AL.counter(); const d1 = GR.bfs(g, 0, c1).dist, d2 = GR.bfsMatrix(A, 0, c2); const reached = d1.filter(x => x >= 0).length; return { m, list: c1.get("scans") + c1.get("deq"), scans: c1.get("scans"), cells: c2.get("cells"), reached, same: GR.sameArr(d1, d2), sumDeg: d1.map((d, v) => d >= 0 ? g.adj[v].length : 0).reduce((a, b) => a + b, 0) }; });
    const F = AL.frame("#trav-svg", 760, 300, { l: 60, r: 290, t: 16, b: 40 });
    const x = d3.scaleLinear().domain([0, maxM]).range([0, F.iw]), y = d3.scaleLinear().domain([0, Math.max(...rows.map(d => Math.max(d.cells, d.list))) * 1.1]).range([F.ih, 0]);
    AL.gridY(F.g, y, F.iw, 5); AL.axisB(F.g, x, F.ih, 6, "E (edges), V = 64 fixed"); AL.axisL(F.g, y, 5, "operations in one BFS from a");
    F.g.append("path").datum(rows).attr("fill", "none").attr("stroke", AC.bad).attr("stroke-width", 2).attr("d", d3.line().x(d => x(d.m)).y(d => y(d.cells)));
    F.g.append("path").datum(rows).attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 2).attr("d", d3.line().x(d => x(d.m)).y(d => y(d.list)));
    rows.forEach(d => { F.g.append("circle").attr("cx", x(d.m)).attr("cy", y(d.cells)).attr("r", 3).attr("fill", AC.bad); F.g.append("circle").attr("cx", x(d.m)).attr("cy", y(d.list)).attr("r", 3).attr("fill", AC.accent); });
    AL.legend(F.g, [{ label: "matrix: cells read = V × vertices reached", color: AC.bad }, { label: "lists: dequeues + adjacency entries scanned", color: AC.accent }], 8, 14);
    GD.lines(F.g, F.iw + 12, 8, [`V = 64, random simple graphs`].concat(rows.map(d => `E = ${d.m}: lists ${d.list}, matrix ${d.cells}${d.same ? "" : " ✗"}`)).concat([``, `lists: scans = Σ deg over reached = ${ok(rows.every(d => d.scans === d.sumDeg))}`, `matrix: cells = 64 × reached = ${ok(rows.every(d => d.cells === 64 * d.reached))}`, `same distances both ways: ${ok(rows.every(d => d.same))}`]), { hi: [1, 2, 3, 4, 5, 6, 7], size: 10.5, lh: 16 });
    txt("trav-readout", `BFS from a on random graphs with V = 64: with adjacency lists the work is dequeues + entries scanned = ` + rows.map(d => `${d.list} at E = ${d.m}`).join(", ") + `; with the matrix it is V × (vertices reached) = ` + rows.map(d => `${d.cells}`).join(", ") + ` cells read, regardless of E. Both computed the same distances on every graph: ${ok(rows.every(d => d.same))}.`);
  });

  /* ── F05 an implicit graph: a grid with walls, BFS flood fill by layer ──── */
  fig("grid-svg", () => {
    const W = 22, Hh = 11, r = AL.rng(17);
    const walls = new Set(); for (let i = 0; i < W * Hh; i++) if (r() < 0.24) walls.add(i);
    const s = 0; walls.delete(s);
    const g = GR.grid(W, Hh, walls);
    const c = AL.counter(), frames = [];
    const res = GR.bfs(g, s, c, f => { if (f.ev === "dequeue") frames.push(f); });
    const relax = GR.distByRelaxation(g, s), agree = GR.sameArr(res.dist, relax);
    const lay = GR.layers(res.dist), reached = res.dist.filter(d => d >= 0).length, open = W * Hh - walls.size;
    const cell = 30, pos = GD.gridLayout(W, Hh, 20, 20, cell);
    const F = AL.frame("#grid-svg", 760, 360, { l: 0, r: 0, t: 0, b: 0 });
    function render(f, i) {
      F.g.selectAll("*").remove();
      for (let v = 0; v < W * Hh; v++) {
        const p = pos[v], d = f.dist[v], col = walls.has(v) ? "#0b0d12" : d < 0 ? AC.panel : (f.colour[v] === 1 ? AC.a2 : LAYER[d % LAYER.length]);
        F.g.append("rect").attr("x", p.x).attr("y", p.y).attr("width", cell - 2).attr("height", cell - 2).attr("rx", 3).attr("fill", col).attr("opacity", walls.has(v) ? 1 : d < 0 ? 1 : 0.85).attr("stroke", v === f.u ? AC.ink : AC.line).attr("stroke-width", v === f.u ? 2 : 1);
        if (d >= 0) F.g.append("text").attr("x", p.x + cell / 2 - 1).attr("y", p.y + cell / 2 + 3).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", walls.has(v) ? AC.muted : "#0f1117").text(d);
      }
      const reachedNow = f.dist.filter(d => d >= 0).length;
      txt("grid-readout", `Step ${i + 1} of ${frames.length}: dequeued cell ${f.u} at distance ${f.dist[f.u]}; ${reachedNow} of ${open} open cells labelled so far, queue length ${f.queue.length}. Final: ${reached} cells reached in ${lay.length} layers (farthest ${lay.length - 1} steps), ${c.get("deq")} dequeues, ${c.get("scans")} edge scans = Σ deg of reached cells; edges generated on the fly from coordinates (${g.m} in all). Distances re-derived by unit-weight relaxation over the edge list: ${ok(agree)}.`);
    }
    const st = AL.stepper(d3.select("#grid-svg"), { frames, render, delay: 90 });
    st.go(frames.length - 1);                         // default view = the finished fill; ⟲ replays it
  });

})();
/* FIGS-1 */
(function () {
  if (typeof document === "undefined" || typeof d3 === "undefined") return;
  const $ = id => document.getElementById(id);
  const fig = (id, fn) => { if ($(id)) fn(); };
  const txt = (id, s) => { const el = $(id); if (el) el.textContent = s; };
  const on = (id, ev, fn) => { const el = $(id); if (el) el.addEventListener(ev, fn); };
  const val = id => { const el = $(id); return el ? el.value : null; };
  const f1 = x => (+x).toFixed(1), f2 = x => (+x).toFixed(2);
  const ok = b => b ? "✓" : "✗";
  const L = GR.lbl, LL = a => a.map(L).join(" ");
  const CLS = { tree: AC.accent, back: AC.bad, forward: AC.a2, cross: AC.violet };
  const LAYER = [AC.a2, AC.accent, AC.good, AC.violet, AC.teal, AC.rose, "#f0abfc", "#fde68a", "#93c5fd", "#86efac"];

  /* ── F06 BFS stepped on G1: the queue, the layers, the distances ───────── */
  fig("bfs-svg", () => {
    const g = GX.G1, pos = GX.G1pos, s = 0, c = AL.counter(), frames = [];
    const res = GR.bfs(g, s, c, f => { if (f.ev !== "skip") frames.push(f); });
    const brute = GR.distByMatrixPower(g, s), relax = GR.distByRelaxation(g, s);
    const F = AL.frame("#bfs-svg", 760, 330, { l: 10, r: 10, t: 10, b: 10 });
    function render(f, i) {
      F.g.selectAll("*").remove();
      const isTree = (id, u, v) => (f.parent[v] === u && f.dist[v] === f.dist[u] + 1) || (f.parent[u] === v && f.dist[u] === f.dist[v] + 1);
      GD.drawGraph(F.g, g, pos.map(p => ({ x: p.x * 0.78 + 10, y: p.y * 0.78 + 10 })), {
        r: 14, fill: v => f.colour[v] === 1 ? "#3b2f14" : f.colour[v] === 2 ? LAYER[f.dist[v] % LAYER.length] : null,
        textFill: v => f.colour[v] === 2 ? "#0f1117" : null, stroke: v => v === f.u ? AC.ink : (f.colour[v] === 1 ? AC.a2 : null), sw: v => v === f.u ? 2.5 : null,
        label: v => f.dist[v] >= 0 ? `d = ${f.dist[v]}` : "∞",
        edgeColor: (id, u, v) => isTree(id, u, v) ? AC.accent : (f.ev === "enqueue" && ((u === f.u && v === f.v) || (v === f.u && u === f.v)) ? AC.a2 : null),
        edgeW: (id, u, v) => isTree(id, u, v) ? 2.5 : null
      });
      // the queue
      GD.note(F.g, 470, 24, "queue (head at left)", AC.muted, 10.5);
      AL.row(F.g, f.queue.length ? f.queue.map(L) : [""], { x: 470, y: 32, w: 28, h: 26, gap: 3, index: false, mark: () => f.queue.length ? "#3b2f14" : AC.panel });
      // the distance table
      GD.note(F.g, 470, 90, "dist[v]  (parent)", AC.muted, 10.5);
      AL.row(F.g, f.dist.map(d => d < 0 ? "∞" : d), { x: 470, y: 98, w: 28, h: 24, gap: 3, index: false, mark: i => f.dist[i] >= 0 ? LAYER[f.dist[i] % LAYER.length] : null, text: "#0f1117" });
      f.dist.forEach((d, v) => F.g.append("text").attr("x", 470 + v * 31 + 14).attr("y", 137).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.muted).text(L(v)));
      f.parent.forEach((p, v) => F.g.append("text").attr("x", 470 + v * 31 + 14).attr("y", 151).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", AC.muted).text(p < 0 ? "·" : L(p)));
      const ev = f.ev === "start" ? `start: enqueue source ${L(s)}, dist 0` : f.ev === "dequeue" ? `dequeue ${L(f.u)} (dist ${f.dist[f.u]}); scan its ${g.adj[f.u].length} neighbours` : `edge ${L(f.u)}–${L(f.v)}: ${L(f.v)} was white → dist ${f.dist[f.v]}, parent ${L(f.u)}, enqueue`;
      GD.lines(F.g, 470, 180, [ev, ``, `visited so far: ${f.order.length} of ${g.n}`, `layers: ${GR.layers(f.dist).map(l => "{" + l.map(L).join("") + "}").join(" ")}`, ``, `final (measured): ${c.get("deq")} dequeues, ${c.get("enq")} enqueues,`, `${c.get("scans")} edge scans = 2E = ${2 * g.m} ${ok(c.get("scans") === 2 * g.m)};  max queue ${res.maxQ}`, `distances = matrix powers ${ok(GR.sameArr(res.dist, brute))} = relaxation ${ok(GR.sameArr(res.dist, relax))}`], { hi: [0, 3, 5, 6, 7], size: 10.5, lh: 16 });
      txt("bfs-readout", `Step ${i + 1} of ${frames.length}: ${ev}. Queue now [${LL(f.queue)}]. Final distances from a: ` + res.dist.map((d, v) => `${L(v)}=${d}`).join(", ") + `; visit order ${LL(res.order)}; layers ${GR.layers(res.dist).map(l => "{" + l.map(L).join(",") + "}").join(" ")}. Measured: ${c.get("deq")} dequeues, ${c.get("enq")} enqueues, ${c.get("scans")} edge scans (= 2E = ${2 * g.m}), queue never longer than ${res.maxQ}. Cross-checked against boolean matrix powers ${ok(GR.sameArr(res.dist, brute))} and unit-weight relaxation ${ok(GR.sameArr(res.dist, relax))}.`);
    }
    const st = AL.stepper(d3.select("#bfs-svg"), { frames, render, delay: 800 });
    st.go(frames.length - 1);
  });

  /* ── F07 BFS distances checked two other ways on a random graph ────────── */
  fig("bfscheck-svg", () => {
    // batch check at load: hundreds of random graphs, both kinds
    const r0 = AL.rng(99); let trials = 0, agree = 0, scansOk = 0;
    for (let t = 0; t < 300; t++) { const n = AL.randInt(r0, 2, 22), directed = t % 2 === 1, maxM = directed ? n * (n - 1) : n * (n - 1) / 2, m = Math.floor(r0() * (maxM + 1)); const g = GR.random(n, m, directed, r0); const s = AL.randInt(r0, 0, n - 1), c = AL.counter(); const b = GR.bfs(g, s, c); trials++; if (GR.sameArr(b.dist, GR.distByMatrixPower(g, s)) && GR.sameArr(b.dist, GR.distByRelaxation(g, s))) agree++; const sd = b.dist.map((d, v) => d >= 0 ? g.adj[v].length : 0).reduce((a, b) => a + b, 0); if (c.get("scans") === sd) scansOk++; }
    function draw() {
      const n = +(val("bfsc-n") || 18), seed = +(val("bfsc-seed") || 1), directed = (val("bfsc-kind") || "undirected") === "directed";
      const r = AL.rng(1000 + seed), m = Math.round(n * 1.6), g = GR.random(n, m, directed, r);
      const c = AL.counter(), b = GR.bfs(g, 0, c), p = GR.distByMatrixPower(g, 0), q = GR.distByRelaxation(g, 0);
      const F = AL.frame("#bfscheck-svg", 760, 320, { l: 10, r: 250, t: 10, b: 10 });
      const pos = GD.forceLayout(g, 480, 300, seed + 7);
      GD.drawGraph(F.g, g, pos, { r: 12, fill: v => b.dist[v] >= 0 ? LAYER[b.dist[v] % LAYER.length] : AC.panel, textFill: v => b.dist[v] >= 0 ? "#0f1117" : AC.muted, label: v => b.dist[v] >= 0 ? b.dist[v] : "unreachable", stroke: v => v === 0 ? AC.ink : null, sw: v => v === 0 ? 2.5 : null,
        edgeColor: (id, u, v) => (b.parent[v] === u || (!directed && b.parent[u] === v)) ? AC.accent : null, edgeW: (id, u, v) => (b.parent[v] === u || (!directed && b.parent[u] === v)) ? 2.5 : null });
      const lay = GR.layers(b.dist), reached = b.dist.filter(d => d >= 0).length;
      GD.lines(F.g, F.iw + 12, 8, [`${directed ? "directed" : "undirected"}, V = ${n}, E = ${m}, source a`, `reached ${reached} of ${n}; ${lay.length} layers`, `layer sizes: ${lay.map(l => l.length).join(", ")}`, ``, `BFS = matrix powers: ${ok(GR.sameArr(b.dist, p))}`, `BFS = relaxation:    ${ok(GR.sameArr(b.dist, q))}`, `scans ${c.get("scans")} = Σ deg(reached) ${ok(c.get("scans") === b.dist.map((d, v) => d >= 0 ? g.adj[v].length : 0).reduce((a, b) => a + b, 0))}`, `dequeues ${c.get("deq")} = reached ${ok(c.get("deq") === reached)}`, `max queue length ${b.maxQ}`, ``, `at load, ${trials} random graphs (both kinds,`, `V ≤ 22, all densities):`, `  distances agree 3 ways: ${agree}/${trials}`, `  scans = Σ deg(reached): ${scansOk}/${trials}`], { hi: [4, 5, 6, 7, 12, 13], size: 10.5, lh: 16 });
      txt("bfscheck-readout", `${directed ? "Directed" : "Undirected"} random graph, V = ${n}, E = ${m}, source a: ${reached} vertices reached in ${lay.length} layers of sizes ${lay.map(l => l.length).join(", ")}; tree edges in blue. Distances agree with boolean matrix powers ${ok(GR.sameArr(b.dist, p))} and with unit-weight relaxation ${ok(GR.sameArr(b.dist, q))}; ${c.get("scans")} edge scans = Σ deg over reached vertices, ${c.get("deq")} dequeues = vertices reached, longest queue ${b.maxQ}. At load: ${agree} of ${trials} random graphs agreed three ways and ${scansOk} of ${trials} had the exact scan count.`);
    }
    ["bfsc-n", "bfsc-seed", "bfsc-kind"].forEach(id => { on(id, "input", draw); on(id, "change", draw); }); draw();
  });

  /* ── F08 bipartite 2-colouring, and the chord that breaks it ───────────── */
  fig("bip-svg", () => {
    function draw() {
      const chord = val("bip-chord") || "none", g = GX.C6(chord === "none" ? "" : chord);
      const c = AL.counter(), B = GR.bipartite(g, c);
      const brute = (() => { for (let mask = 0; mask < 64; mask++) { const col = Array.from({ length: 6 }, (_, i) => (mask >> i) & 1); if (GR.checkColouring(g, col)) return true; } return false; })();
      const pos = GD.circleLayout(6, 180, 150, 110);
      const F = AL.frame("#bip-svg", 760, 300, { l: 10, r: 250, t: 10, b: 10 });
      const inCyc = (u, v) => { if (!B.oddCycle) return false; const k = B.oddCycle.length; for (let i = 0; i < k; i++) { const a = B.oddCycle[i], b = B.oddCycle[(i + 1) % k]; if ((a === u && b === v) || (a === v && b === u)) return true; } return false; };
      GD.drawGraph(F.g, g, pos, { r: 15, fill: v => B.colour[v] === 0 ? AC.accent : AC.a2, textFill: () => "#0f1117", label: v => `colour ${B.colour[v]}`,
        edgeColor: (id, u, v) => inCyc(u, v) ? AC.bad : (B.colour[u] === B.colour[v] ? AC.bad : null), edgeW: (id, u, v) => inCyc(u, v) ? 3 : null });
      // the two sides
      const side0 = B.colour.map((c, v) => c === 0 ? v : -1).filter(v => v >= 0), side1 = B.colour.map((c, v) => c === 1 ? v : -1).filter(v => v >= 0);
      GD.lines(F.g, 380, 30, [`C₆ ${chord === "none" ? "alone" : "+ chord " + chord[0] + "–" + chord[1]}: V = 6, E = ${g.m}`, ``, `BFS colouring: {${side0.map(L).join(",")}} / {${side1.map(L).join(",")}}`, B.ok ? `every edge joins the two sides ${ok(GR.checkColouring(g, B.colour))}` : `edge ${L(B.badEdge[0])}–${L(B.badEdge[1])} joins two vertices of one colour`, B.ok ? `bipartite ✓` : `odd cycle returned: ${LL(B.oddCycle)} (length ${B.oddCycle.length}) ${ok(GR.isCycle(g, B.oddCycle) && B.oddCycle.length % 2 === 1)}`, ``, `brute force over all 2⁶ = 64 colourings:`, `  a proper 2-colouring exists: ${ok(brute)} — agrees ${ok(brute === B.ok)}`, ``, `measured: ${c.get("deq")} dequeues, ${c.get("scans")} edge scans${B.ok ? " = 2E = " + 2 * g.m : " (stopped early)"}`], { hi: [2, 3, 4, 7], size: 10.5, lh: 16 });
      txt("bip-readout", `${chord === "none" ? "The 6-cycle alone" : "The 6-cycle plus chord " + chord[0] + "–" + chord[1]}: BFS colouring puts {${side0.map(L).join(", ")}} on one side and {${side1.map(L).join(", ")}} on the other. ` + (B.ok ? `Every edge crosses: bipartite ✓, confirmed by trying all 64 colourings ${ok(brute)}.` : `Edge ${L(B.badEdge[0])}–${L(B.badEdge[1])} joins two vertices of the same colour, and the odd cycle returned is ${LL(B.oddCycle)} (length ${B.oddCycle.length}); trying all 64 colourings confirms none is proper ${ok(!brute)}.`) + ` ${c.get("deq")} dequeues and ${c.get("scans")} edge scans, measured.`);
    }
    on("bip-chord", "change", draw); draw();
  });

  /* ── F09 DFS stepped: d/f times, the stack, edges classified live ──────── */
  fig("dfs-svg", () => {
    function build() {
      const which = val("dfs-graph") || "G2", g = which === "G1" ? GX.G1 : GX.G2, pos = which === "G1" ? GX.G1pos : GX.G2pos;
      const c = AL.counter(), frames = [];
      const res = GR.dfs(g, c, f => frames.push(f));
      const iter = GR.dfs(g, null, null, { iterative: true });
      const same = GR.sameArr(res.d, iter.d) && GR.sameArr(res.f, iter.f);
      const paren = GR.checkParenthesis(res), kc = GR.classCounts(res);
      return { which, g, pos, c, frames, res, same, paren, kc };
    }
    let S = build(), st = null;
    const F = AL.frame("#dfs-svg", 760, 340, { l: 10, r: 10, t: 10, b: 10 });
    function render(f, i) {
      const { g, pos, res, c, kc, paren, same } = S;
      F.g.selectAll("*").remove();
      GD.drawGraph(F.g, g, pos.map(p => ({ x: p.x * 0.72 + 10, y: p.y * 0.85 + 20 })), {
        r: 14, fill: v => f.colour[v] === 1 ? "#3b2f14" : f.colour[v] === 2 ? "#1e2a44" : null, stroke: v => v === f.u ? AC.ink : (f.colour[v] === 1 ? AC.a2 : f.colour[v] === 2 ? AC.accent : null), sw: v => v === f.u ? 2.5 : null,
        label: v => f.d[v] ? `${f.d[v]}/${f.f[v] || "·"}` : "",
        edgeColor: (id) => f.cls[id] ? CLS[f.cls[id]] : null, edgeW: id => f.cls[id] === "tree" ? 2.5 : f.cls[id] ? 2 : null, edgeDash: id => f.cls[id] && f.cls[id] !== "tree" ? "5,3" : null,
        edgeOpacity: id => f.cls[id] ? 1 : 0.45, edgeLabel: id => f.cls[id] ? f.cls[id][0].toUpperCase() : null
      });
      GD.note(F.g, 470, 24, "stack (bottom at left)", AC.muted, 10.5);
      AL.row(F.g, f.stack.length ? f.stack.map(L) : [""], { x: 470, y: 32, w: 28, h: 26, gap: 3, index: false, mark: () => f.stack.length ? "#3b2f14" : AC.panel });
      const ev = f.ev === "discover" ? `t = ${f.time}: discover ${L(f.u)} (d = ${f.time}), push` : f.ev === "finish" ? `t = ${f.time}: finish ${L(f.u)} (f = ${f.time}), pop` : `edge ${L(f.u)}→${L(f.v)}: ${L(f.v)} is ${f.ev === "back" ? "grey (on the stack) → BACK" : f.ev === "forward" ? "black, d[u] < d[v] → FORWARD" : "black, d[u] > d[v] → CROSS"}`;
      GD.lines(F.g, 470, 90, [ev, ``, `d/f so far: ` + f.d.slice(0, 4).map((d, v) => d ? `${L(v)} ${d}/${f.f[v] || "·"}` : "").filter(Boolean).join("  "), `            ` + f.d.slice(4).map((d, v) => d ? `${L(v + 4)} ${d}/${f.f[v + 4] || "·"}` : "").filter(Boolean).join("  "), `classes so far: ${["tree", "back", "forward", "cross"].map(k => k[0].toUpperCase() + " " + f.cls.filter(x => x === k).length).join(", ")}`, ``, `final (measured): ${c.get("discover")} discoveries,`, `  ${c.get("scans")} entries scanned = ${g.directed ? "E" : "2E"} = ${g.directed ? g.m : 2 * g.m} ${ok(c.get("scans") === (g.directed ? g.m : 2 * g.m))};`, `  ${c.get("classified")} edges classified = E`, `  tree ${kc.tree}, back ${kc.back}, forward ${kc.forward}, cross ${kc.cross}; max stack ${c.get("maxStack")}`, `  DFS trees: ${res.roots.length} (roots ${LL(res.roots)})`, `  parenthesis theorem on all ${paren.nested + paren.disjoint} pairs: ${ok(paren.ok)}`, `  explicit-stack version gives the same d/f: ${ok(same)}`], { hi: [0, 2, 4, 6, 7, 8], size: 10, lh: 15.5 });
      txt("dfs-readout", `Step ${i + 1} of ${S.frames.length}: ${ev}. Stack [${LL(f.stack)}]. Final d/f: ` + res.d.map((d, v) => `${L(v)} ${d}/${res.f[v]}`).join(", ") + `; edges: ${g.edges.map((e, id) => L(e[0]) + (g.directed ? "→" : "–") + L(e[1]) + " " + res.cls[id]).join(", ")}. Measured: ${c.get("discover")} discoveries, ${c.get("scans")} adjacency entries scanned (${g.directed ? "E" : "2E"}), ${c.get("classified")} edges classified, max stack depth ${c.get("maxStack")}, ${res.roots.length} DFS tree(s). Parenthesis theorem verified on all vertex pairs ${ok(paren.ok)}; explicit-stack DFS reproduces the same times ${ok(same)}.`);
    }
    function rebuild() {
      S = build();
      const host = d3.select("#dfs-svg").node().parentNode; host.querySelectorAll("[role=group]").forEach(e => e.remove());
      st = AL.stepper(d3.select("#dfs-svg"), { frames: S.frames, render, delay: 700 }); st.go(S.frames.length - 1);
    }
    on("dfs-graph", "change", rebuild); rebuild();
  });

  /* ── F10 the parenthesis theorem as nested intervals ───────────────────── */
  fig("paren-svg", () => {
    function draw() {
      const which = val("paren-graph") || "G2", g = which === "G1" ? GX.G1 : GX.G2;
      const res = GR.dfs(g), P = GR.checkParenthesis(res), n = g.n;
      const F = AL.frame("#paren-svg", 760, 260, { l: 20, r: 240, t: 24, b: 10 });
      GD.intervals(F.g, res, { x: 0, y: 0, w: 480, rowH: 22, mark: v => LAYER[res.root === undefined ? 0 : (res.roots.indexOf(res.root[v]) % LAYER.length)] });
      // the parenthesis string
      const events = []; for (let t = 1; t <= 2 * n; t++) { const v = res.d.indexOf(t); if (v >= 0) events.push("(" + L(v)); else events.push(L(res.f.indexOf(t)) + ")"); }
      GD.note(F.g, 0, 8 * 22 + 40, events.join(" "), AC.ink, 11);
      GD.lines(F.g, 500, 6, [`${which}: ${g.directed ? "directed" : "undirected"}, V = ${n}`, `times 1 … 2V = ${2 * n}`, ``, `all ${n * (n - 1) / 2} vertex pairs checked:`, `  nested (ancestor/descendant): ${P.nested}`, `  disjoint (neither): ${P.disjoint}`, `  overlapping without nesting: ${P.bad}`, `theorem holds: ${ok(P.ok)}`, ``, `DFS trees: ${res.roots.length} (roots ${LL(res.roots)})`, `each tree's intervals sit inside`, `its root's [d, f]`], { hi: [4, 5, 6, 7], size: 10.5, lh: 16 });
      txt("paren-readout", `DFS of ${which} (${g.directed ? "directed" : "undirected"}, V = ${n}): every interval [d, f] drawn as a bar, indented by tree depth; the parenthesis string is ${events.join(" ")}. Of the ${n * (n - 1) / 2} vertex pairs, ${P.nested} are nested (and in each the inner vertex is a descendant of the outer), ${P.disjoint} are disjoint (and neither is an ancestor of the other), ${P.bad} overlap without nesting — theorem verified ${ok(P.ok)}.`);
    }
    on("paren-graph", "change", draw); draw();
  });

})();
/* FIGS-2 */
(function () {
  if (typeof document === "undefined" || typeof d3 === "undefined") return;
  const $ = id => document.getElementById(id);
  const fig = (id, fn) => { if ($(id)) fn(); };
  const txt = (id, s) => { const el = $(id); if (el) el.textContent = s; };
  const on = (id, ev, fn) => { const el = $(id); if (el) el.addEventListener(ev, fn); };
  const val = id => { const el = $(id); return el ? el.value : null; };
  const f1 = x => (+x).toFixed(1), f2 = x => (+x).toFixed(2);
  const ok = b => b ? "✓" : "✗";
  const L = GR.lbl, LL = a => a.map(L).join(" ");
  const CLS = { tree: AC.accent, back: AC.bad, forward: AC.a2, cross: AC.violet };
  const LAYER = [AC.a2, AC.accent, AC.good, AC.violet, AC.teal, AC.rose, "#f0abfc", "#fde68a", "#93c5fd", "#86efac"];

  /* ── F11 cycle detection: directed against undirected, and the multi-edge trap ── */
  fig("cyc-svg", () => {
    const cases = {
      G2: { g: GX.G2, pos: GX.G2pos, name: "directed G2" },
      G1: { g: GX.G1, pos: GX.G1pos, name: "undirected G1" },
      tree: { g: GR.graph(7, GX.E("ab ac bd be cf cg"), false), pos: [{ x: 300, y: 50 }, { x: 180, y: 140 }, { x: 420, y: 140 }, { x: 120, y: 230 }, { x: 240, y: 230 }, { x: 360, y: 230 }, { x: 480, y: 230 }], name: "undirected tree" },
      multi: { g: GR.graph(3, [[0, 1], [0, 1], [1, 2]], false), pos: [{ x: 150, y: 140 }, { x: 300, y: 140 }, { x: 450, y: 140 }], name: "undirected, a–b doubled" },
      loop: { g: GR.graph(3, [[0, 1], [1, 1], [1, 2]], true), pos: [{ x: 150, y: 160 }, { x: 300, y: 160 }, { x: 450, y: 160 }], name: "directed with a self-loop" }
    };
    function draw() {
      const k = val("cyc-case") || "G2", C = cases[k], g = C.g;
      const F = AL.frame("#cyc-svg", 760, 300, { l: 10, r: 250, t: 10, b: 10 });
      let lines, witness = null, cls = null;
      if (g.directed) {
        const c = AL.counter(), r = GR.hasCycleDirected(g, c); cls = r.res.cls;
        const R = GR.transitiveClosure(g); let brute = false; for (const [u, v] of g.edges) if (R[v][u]) brute = true;
        const backs = g.edges.map((e, id) => cls[id] === "back" ? id : -1).filter(x => x >= 0);
        witness = backs.length ? g.edges[backs[0]] : null;
        lines = [`${C.name}: V = ${g.n}, E = ${g.m}`, ``, `DFS back edges: ${backs.map(id => L(g.edges[id][0]) + "→" + L(g.edges[id][1])).join(", ") || "none"}`, `cyclic: ${r.cyclic ? "yes" : "no"}`, ``, `brute force (transitive closure):`, `  some edge u→v with v ⇝ u: ${brute ? "yes" : "no"}  ${ok(brute === r.cyclic)}`, ``, `${c.get("scans")} entries scanned = E ${ok(c.get("scans") === g.m)}`, `(directed: all four classes possible;`, ` only BACK means a cycle)`];
      } else {
        const c = AL.counter(), r = GR.hasCycleUndirected(g, c), rv = GR.hasCycleUndirected(g, null, true), cnt = GR.cycleByCounting(g), comps = GR.components(g).count;
        const res = GR.dfs(g); cls = res.cls; witness = r.witness;
        lines = [`${C.name}: V = ${g.n}, E = ${g.m}`, ``, `DFS, skipping the ARRIVAL EDGE (by id):`, `  cyclic: ${r.cyclic ? "yes — grey neighbour " + L(r.witness[0]) + "–" + L(r.witness[1]) : "no"}`, `DFS, skipping the PARENT VERTEX:`, `  cyclic: ${rv.cyclic ? "yes" : "no"}${rv.cyclic !== r.cyclic ? "   ← misses the doubled edge" : ""}`, ``, `counting check: E = V − components?`, `  ${g.m} vs ${g.n} − ${comps} = ${g.n - comps} → ${cnt ? "cycle" : "acyclic (a forest)"} ${ok(cnt === r.cyclic)}`, ``, `${c.get("scans")} entries scanned = 2E ${ok(c.get("scans") === 2 * g.m)}`, `(undirected: only TREE and BACK edges)`];
      }
      GD.drawGraph(F.g, g, C.pos.map(p => ({ x: p.x * 0.8, y: p.y * 0.9 + 10 })), { r: 13, edgeColor: id => cls && cls[id] ? CLS[cls[id]] : null, edgeW: id => cls && cls[id] === "back" ? 3 : null, edgeDash: id => cls && cls[id] && cls[id] !== "tree" ? "5,3" : null, edgeLabel: id => cls && cls[id] ? cls[id][0].toUpperCase() : null });
      GD.lines(F.g, F.iw + 12, 8, lines, { hi: [2, 3, 5, 8], size: 10.5, lh: 16 });
      txt("cyc-readout", lines.filter(Boolean).join(" · "));
    }
    on("cyc-case", "change", draw); draw();
  });

  /* ── F12 connected components, DFS count against union-find ───────────── */
  fig("comp-svg", () => {
    // batch: hundreds of random graphs at load
    const r0 = AL.rng(31); let trials = 0, agree = 0, forestOk = 0, parenOk = 0, classOk = 0;
    for (let t = 0; t < 300; t++) { const n = AL.randInt(r0, 1, 30), m = Math.floor(r0() * n * 1.2); const g = GR.random(n, m, false, r0); const a = GR.components(g).count, b = GR.componentsByUnionFind(g); trials++; if (a === b) agree++; if ((g.m === g.n - a) === !GR.hasCycleUndirected(g).cyclic) forestOk++;
      const dr = GR.dfs(g), pk = GR.checkParenthesis(dr), kc = GR.classCounts(dr); if (pk.ok) parenOk++; if (kc.forward === 0 && kc.cross === 0 && kc.tree === g.n - dr.roots.length) classOk++; }
    function draw() {
      const n = 24, m = +(val("comp-m") || 14), g = GR.random(n, m, false, AL.rng(8)), c = AL.counter();
      const C = GR.components(g, c), uf = GR.componentsByUnionFind(g);
      const F = AL.frame("#comp-svg", 760, 320, { l: 10, r: 250, t: 10, b: 10 });
      const pos = GD.forceLayout(g, 480, 300, 8);
      GD.drawGraph(F.g, g, pos, { r: 11, fill: v => LAYER[C.comp[v] % LAYER.length], textFill: () => "#0f1117", fontSize: 10 });
      const sizes = []; C.comp.forEach(k => { sizes[k] = (sizes[k] || 0) + 1; }); sizes.sort((a, b) => b - a);
      const isolated = g.adj.filter(a => a.length === 0).length;
      GD.lines(F.g, F.iw + 12, 8, [`V = ${n}, E = ${m}`, ``, `components by DFS: ${C.count}`, `components by union-find: ${uf}  ${ok(C.count === uf)}`, `sizes: ${sizes.length > 10 ? sizes.slice(0, 10).join(", ") + ", …" : sizes.join(", ")}`, `isolated vertices: ${isolated}`, ``, `E − (V − components) = ${m - (n - C.count)}`, `  = independent cycles (${m - (n - C.count) === 0 ? "a forest" : "cyclic"})`, ``, `measured: ${c.get("visits")} visits, ${c.get("scans")} scans = 2E ${ok(c.get("scans") === 2 * m)}`, ``, `at load, ${trials} random graphs:`, `  DFS count = union-find: ${agree}/${trials}`, `  E = V − c ⇔ acyclic: ${forestOk}/${trials}`], { hi: [2, 3, 7, 8, 13, 14], size: 10.5, lh: 16 });
      txt("comp-readout", `Random undirected graph, V = ${n}, E = ${m}: ${C.count} connected components by the DFS outer loop, ${uf} by union-find ${ok(C.count === uf)}; component sizes ${sizes.join(", ")}; ${isolated} isolated vertices. E − (V − components) = ${m - (n - C.count)} independent cycles. ${c.get("visits")} vertex visits and ${c.get("scans")} edge scans, measured. At load: ${agree} of ${trials} random graphs agreed between DFS and union-find, and E = V − c matched acyclicity on ${forestOk} of ${trials}; the parenthesis theorem held on ${parenOk} of ${trials}, and the DFS had no forward or cross edges and exactly V − trees tree edges on ${classOk} of ${trials}.`);
    }
    on("comp-m", "input", draw); on("comp-m", "change", draw); draw();
  });

  /* ── F13 topological sort by both methods on one DAG ──────────────────── */
  fig("topo-svg", () => {
    const g = GX.D, pos = GX.Dpos, c1 = AL.counter(), c2 = AL.counter(), frames = [];
    const T1 = GR.topoDFS(g, c1), T2 = GR.topoKahn(g, c2, f => frames.push(f));
    const count = GR.countTopo(g), ham = GR.hamiltonianPathInOrder(g, T1.order);
    const dg = GR.degrees(g);
    const F = AL.frame("#topo-svg", 760, 330, { l: 10, r: 10, t: 10, b: 10 });
    function render(f, i) {
      F.g.selectAll("*").remove();
      const emitted = new Set(f.order);
      GD.drawGraph(F.g, g, pos.map(p => ({ x: p.x * 0.75 + 10, y: p.y * 0.8 + 20 })), { r: 13, fill: v => emitted.has(v) ? "#1e2a44" : (f.queue.indexOf(v) >= 0 ? "#3b2f14" : null), stroke: v => v === f.u ? AC.ink : (f.queue.indexOf(v) >= 0 ? AC.a2 : emitted.has(v) ? AC.accent : null), sw: v => v === f.u ? 2.5 : null, label: v => `in ${f.indeg[v]}`, edgeOpacity: (id, u) => emitted.has(u) ? 0.25 : 1 });
      GD.note(F.g, 470, 22, "Kahn: in-degree-0 queue", AC.muted, 10.5);
      AL.row(F.g, f.queue.length ? f.queue.map(L) : [""], { x: 470, y: 30, w: 26, h: 24, gap: 3, index: false, mark: () => f.queue.length ? "#3b2f14" : AC.panel });
      GD.note(F.g, 470, 82, "Kahn order so far", AC.muted, 10.5);
      AL.row(F.g, f.order.length ? f.order.map(L) : [""], { x: 470, y: 90, w: 26, h: 24, gap: 3, index: false, mark: () => f.order.length ? "#1e2a44" : AC.panel });
      GD.note(F.g, 470, 142, "DFS: reverse finishing order", AC.muted, 10.5);
      AL.row(F.g, T1.order.map(L), { x: 470, y: 150, w: 26, h: 24, gap: 3, index: false, mark: () => "#1e2a44" });
      T1.order.forEach((v, k) => F.g.append("text").attr("x", 470 + k * 29 + 13).attr("y", 188).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", AC.muted).text("f=" + T1.res.f[v]));
      GD.lines(F.g, 470, 210, [f.ev === "start" ? `start: in-degree-0 vertices ${LL(f.queue)}` : `emit ${L(f.u)}; decrement its ${g.adj[f.u].length} successors`, ``, `Kahn order valid on all ${g.m} edges: ${ok(GR.checkTopo(g, T2.order))}`, `DFS order valid on all ${g.m} edges: ${ok(GR.checkTopo(g, T1.order))}`, `orders differ: ${T1.order.join() !== T2.order.join() ? "yes" : "no"};  valid orders in all: ${count}`, `unique ⇔ Hamiltonian path: ${ham ? "yes" : "no"} ${ok((count === 1) === ham)}`, `Kahn: ${c2.get("deq")} dequeues, ${c2.get("scans")} scans = E ${ok(c2.get("scans") === g.m)}`, `DFS: ${c1.get("discover")} discoveries, ${c1.get("scans")} scans = E ${ok(c1.get("scans") === g.m)}`], { hi: [0, 2, 3, 4], size: 10, lh: 15.5 });
      txt("topo-readout", `Step ${i + 1} of ${frames.length}: ${f.ev === "start" ? "in-degree-0 vertices " + LL(f.queue) + " seeded" : "emitted " + L(f.u)}; queue [${LL(f.queue)}], in-degrees ${f.indeg.map((d, v) => L(v) + ":" + d).join(" ")}. Kahn's final order ${LL(T2.order)}; DFS reverse-finish order ${LL(T1.order)} (finish times ${T1.order.map(v => L(v) + "=" + T1.res.f[v]).join(", ")}); both validated against every edge ${ok(GR.checkTopo(g, T2.order) && GR.checkTopo(g, T1.order))}; ${count} valid orders exist in total (h is isolated: in-degree ${dg.inn[7]}, out-degree ${dg.out[7]}). Measured: Kahn ${c2.get("deq")} dequeues and ${c2.get("scans")} scans; DFS ${c1.get("discover")} discoveries and ${c1.get("scans")} scans.`);
    }
    const st = AL.stepper(d3.select("#topo-svg"), { frames, render, delay: 800 }); st.go(frames.length - 1);
  });

  /* ── F14 strongly connected components: the two passes ─────────────────── */
  fig("scc-svg", () => {
    const g = GX.G2, pos = GX.G2pos, c = AL.counter(), comps = [];
    const K = GR.sccKosaraju(g, c, f => comps.push(f));
    const T = GR.sccTarjan(g), B = GR.sccBrute(g);
    const okAll = GR.sameComp(K.comp, B.comp) && GR.sameComp(T.comp, B.comp) && K.count === B.count && T.count === B.count;
    const cond = GR.condensation(g, K.comp, K.count), condAcyclic = !GR.hasCycleDirected(cond).cyclic;
    const fC = {}; K.comp.forEach((k, v) => { fC[k] = Math.max(fC[k] || 0, K.first.f[v]); });
    // batch at load
    const r0 = AL.rng(77); let trials = 0, agree = 0, acyc = 0;
    for (let t = 0; t < 300; t++) { const n = AL.randInt(r0, 1, 20), m = Math.floor(r0() * n * 2); const h = GR.random(n, m, true, r0); const a = GR.sccKosaraju(h), b = GR.sccTarjan(h), z = GR.sccBrute(h); trials++; if (GR.sameComp(a.comp, z.comp) && GR.sameComp(b.comp, z.comp)) agree++; if (!GR.hasCycleDirected(GR.condensation(h, a.comp, a.count)).cyclic) acyc++; }
    // frames: pass 1 (finish order), reverse graph, then one frame per component found
    const frames = [{ ev: "pass1" }, { ev: "reverse" }].concat(comps.map(f => Object.assign({ ev: "comp" }, f)));
    const F = AL.frame("#scc-svg", 760, 340, { l: 10, r: 10, t: 10, b: 10 });
    const P = pos.map(p => ({ x: p.x * 0.72 + 10, y: p.y * 0.85 + 20 }));
    function render(f, i) {
      F.g.selectAll("*").remove();
      const gg = f.ev === "pass1" ? g : K.reversed;
      const done = f.ev === "comp" ? f.comp : new Array(g.n).fill(-1);
      GD.drawGraph(F.g, gg, P, { r: 14, fill: v => done[v] >= 0 ? LAYER[done[v] % LAYER.length] : null, textFill: v => done[v] >= 0 ? "#0f1117" : null, stroke: v => f.ev === "comp" && v === f.root ? AC.ink : null, sw: v => f.ev === "comp" && v === f.root ? 2.5 : null,
        label: v => f.ev === "pass1" ? `${K.first.d[v]}/${K.first.f[v]}` : `f = ${K.first.f[v]}`, edgeColor: (id, u, v) => f.ev === "comp" && done[u] >= 0 && done[u] === done[v] ? LAYER[done[u] % LAYER.length] : null, edgeW: (id, u, v) => f.ev === "comp" && done[u] >= 0 && done[u] === done[v] ? 2.5 : null });
      GD.note(F.g, 10, 12, f.ev === "pass1" ? "pass 1: DFS on G, discovery/finish times" : f.ev === "reverse" ? "Gᵀ: every edge reversed" : `pass 2 on Gᵀ, roots in decreasing f: component ${f.k + 1} from ${L(f.root)}`, AC.muted, 10.5);
      const txtL = [`pass-1 finish order: ${LL(K.first.finishOrder)}`, `pass-2 root order (decreasing f): ${LL(K.order)}`, ``];
      if (f.ev === "comp") txtL.push(`root ${L(f.root)} collects {${g.adj.map((_, v) => v).filter(v => f.comp[v] === f.k).map(L).join(",")}}`, ``);
      txtL.push(`components: ${K.count} — ` + Array.from({ length: K.count }, (_, k) => "{" + K.comp.map((x, v) => x === k ? L(v) : "").join("") + "}").join(" "), `f(C) per component: ${Array.from({ length: K.count }, (_, k) => fC[k]).join(" > ")}`, `condensation edges: ${cond.edges.map(e => e.map(k => "{" + K.comp.map((x, v) => x === k ? L(v) : "").join("") + "}").join("→")).join(", ")}`, `condensation acyclic: ${ok(condAcyclic)}`, ``, `single-pass low-link algorithm: same partition ${ok(GR.sameComp(T.comp, K.comp))}`, `brute force via transitive closure: ${ok(okAll)}`, `measured: pass 1 ${c.get("discover")} discoveries + ${c.get("scans")} scans;`, `  pass 2 ${c.get("discover2")} + ${c.get("scans2")} (each = E = ${g.m}) ${ok(c.get("scans") === g.m && c.get("scans2") === g.m)}`, ``, `at load, ${trials} random digraphs:`, `  all three agree ${agree}/${trials}; condensation acyclic ${acyc}/${trials}`);
      GD.lines(F.g, 440, 30, txtL, { hi: [0, 1, 3, 5, 6, 7, 8], size: 10, lh: 15.5 });
      txt("scc-readout", `Step ${i + 1} of ${frames.length}: ${f.ev === "pass1" ? "pass 1 DFS on G gives finish order " + LL(K.first.finishOrder) : f.ev === "reverse" ? "the transpose graph, same components" : "pass 2 from " + L(f.root) + " collects component " + (f.k + 1)}. Components: ${Array.from({ length: K.count }, (_, k) => "{" + K.comp.map((x, v) => x === k ? L(v) : "").filter(Boolean).join(",") + "}").join(", ")}, emitted in decreasing f(C) = ${Array.from({ length: K.count }, (_, k) => fC[k]).join(", ")}, which is a topological order of the condensation (edges ${cond.edges.map(e => e.map(k => "{" + K.comp.map((x, v) => x === k ? L(v) : "").join("") + "}").join("→")).join(", ")}; acyclic ${ok(condAcyclic)}). Low-link single pass and brute-force closure give the same partition ${ok(okAll)}. Measured: ${c.get("scans")} + ${c.get("scans2")} edge scans over the two passes. At load ${agree} of ${trials} random digraphs agreed across all three methods.`);
    }
    const st = AL.stepper(d3.select("#scc-svg"), { frames, render, delay: 900 }); st.go(frames.length - 1);
  });

  /* ── F15 articulation points and bridges by low-link, checked by deletion ── */
  fig("cut-svg", () => {
    const r0 = AL.rng(41); let trials = 0, agree = 0;
    for (let t = 0; t < 300; t++) { const n = AL.randInt(r0, 1, 18), m = Math.floor(r0() * n * 1.5); const h = GR.random(n, m, false, r0); const A = GR.cutVerticesAndBridges(h); trials++; if (GR.sameArr(A.isCut, GR.cutVerticesBrute(h)) && GR.sameArr(A.bridges.map(b => b[2]).sort((x, y) => x - y), GR.bridgesBrute(h).sort((x, y) => x - y))) agree++; }
    function draw() {
      const which = val("cut-graph") || "H";
      const g = which === "H" ? GX.H : which === "G1" ? GX.G1 : GR.path(6), pos = which === "H" ? GX.Hpos : which === "G1" ? GX.G1pos : GD.circleLayout(6, 250, 150, 0).map((_, i) => ({ x: 60 + i * 90, y: 150 }));
      const c = AL.counter(), A = GR.cutVerticesAndBridges(g, c), Ab = GR.cutVerticesBrute(g), Bb = GR.bridgesBrute(g);
      const bridgeIds = new Set(A.bridges.map(b => b[2]));
      const F = AL.frame("#cut-svg", 760, 320, { l: 10, r: 262, t: 10, b: 10 });
      GD.drawGraph(F.g, g, pos.map(p => ({ x: p.x * 0.85, y: p.y * 0.9 + 10 })), { r: 13, fill: v => A.isCut[v] ? "#3b1a1a" : null, stroke: v => A.isCut[v] ? AC.bad : null, sw: v => A.isCut[v] ? 2.5 : null, label: v => `${A.d[v]}/${A.low[v]}`, edgeColor: id => bridgeIds.has(id) ? AC.bad : (A.parent[g.edges[id][1]] === g.edges[id][0] || A.parent[g.edges[id][0]] === g.edges[id][1]) ? AC.accent : null, edgeW: id => bridgeIds.has(id) ? 3.5 : null, edgeDash: id => (A.parent[g.edges[id][1]] === g.edges[id][0] || A.parent[g.edges[id][0]] === g.edges[id][1]) ? null : "5,3" });
      const cuts = A.isCut.map((b, v) => b ? L(v) : "").filter(Boolean), cutsB = Ab.map((b, v) => b ? L(v) : "").filter(Boolean);
      GD.lines(F.g, F.iw + 12, 8, [`V = ${g.n}, E = ${g.m}; labels d/low`, ``, `cut vertices (low-link): ${cuts.join(", ") || "none"}`, `  by deleting each vertex: ${cutsB.join(", ") || "none"}  ${ok(GR.sameArr(A.isCut, Ab))}`, `bridges (low-link): ${A.bridges.map(b => L(b[0]) + "–" + L(b[1])).join(", ") || "none"}`, `  by deleting each edge:`, `  ${Bb.map(i => L(g.edges[i][0]) + "–" + L(g.edges[i][1])).join(", ") || "none"}  ${ok(GR.sameArr(A.bridges.map(b => b[2]).sort((x, y) => x - y), Bb.sort((x, y) => x - y)))}`, ``, `rule: non-root u is a cut vertex iff`, `  a tree child w has low[w] ≥ d[u];`, `  root iff ≥ 2 tree children;`, `  edge (u, w) bridge iff low[w] > d[u]`, ``, `measured: ${c.get("discover")} discoveries,`, `  ${c.get("scans")} scans = 2E ${ok(c.get("scans") === 2 * g.m)}`, `at load, ${trials} random graphs:`, `  ${agree}/${trials} agree with deletion`], { hi: [2, 3, 4, 5, 6, 13, 14], size: 10.5, lh: 15.5 });
      txt("cut-readout", `${which === "H" ? "The ten-vertex example" : which === "G1" ? "G1" : "A six-vertex path"}: articulation points ${cuts.join(", ") || "none"} (by deletion: ${cutsB.join(", ") || "none"}), bridges ${A.bridges.map(b => L(b[0]) + "–" + L(b[1])).join(", ") || "none"} (by deletion: ${Bb.map(i => L(g.edges[i][0]) + "–" + L(g.edges[i][1])).join(", ") || "none"}). Discovery/low values ${A.d.map((d, v) => L(v) + " " + d + "/" + A.low[v]).join(", ")}. ${c.get("discover")} discoveries and ${c.get("scans")} edge scans, measured. At load ${agree} of ${trials} random graphs agreed with brute-force deletion.`);
    }
    on("cut-graph", "change", draw); draw();
  });

  /* ── F16 BFS frontier against DFS stack across graph shapes, measured ──── */
  fig("mem-svg", () => {
    function draw() {
      const n = +(val("mem-n") || 1024), r = AL.rng(11), side = Math.round(Math.sqrt(n));
      const shapes = [["path", GR.path(n)], ["star", GR.star(n)], ["grid " + side + "×" + side, GR.grid(side, side)], ["binary tree", GR.binaryTree(n)], ["random E = 3V", GR.random(n, 3 * n, false, r)]];
      const rows = shapes.map(([name, g]) => { const cb = AL.counter(), cd = AL.counter(); GR.bfs(g, 0, cb); GR.dfs(g, cd); return { name, q: cb.get("maxQ"), s: cd.get("maxStack"), sb: cb.get("scans"), sd: cd.get("scans"), m: g.m, n: g.n }; });
      const F = AL.frame("#mem-svg", 760, 300, { l: 60, r: 250, t: 16, b: 46 });
      const x = d3.scaleBand().domain(rows.map(d => d.name)).range([0, F.iw]).padding(0.25), x1 = d3.scaleBand().domain([0, 1]).range([0, x.bandwidth()]).padding(0.1);
      const y = d3.scaleLinear().domain([0, Math.max(...rows.map(d => Math.max(d.q, d.s))) * 1.15]).range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 4); F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x)); AL.axisL(F.g, y, 4, "vertices held at once (max)");
      rows.forEach(d => { [[d.q, AC.accent, 0], [d.s, AC.a2, 1]].forEach(([v, col, k]) => { F.g.append("rect").attr("x", x(d.name) + x1(k)).attr("y", y(v)).attr("width", x1.bandwidth()).attr("height", F.ih - y(v)).attr("fill", col); F.g.append("text").attr("x", x(d.name) + x1(k) + x1.bandwidth() / 2).attr("y", y(v) - 4).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", AC.ink).text(v); }); });
      AL.legend(F.g, [{ label: "BFS: longest queue", color: AC.accent }, { label: "DFS: deepest stack", color: AC.a2 }], 8, 12);
      GD.lines(F.g, F.iw + 12, 8, [`V = ${n} for every shape, source a`].concat(rows.map(d => `${d.name}: queue ${d.q}, stack ${d.s}`)).concat([``, `edge scans = 2E on every shape:`, `  ${ok(rows.every(d => d.sb === 2 * d.m && d.sd === 2 * d.m))}`, `worst for BFS: ${rows.reduce((a, b) => b.q > a.q ? b : a).name}`, `worst for DFS: ${rows.reduce((a, b) => b.s > a.s ? b : a).name}`]), { hi: [1, 2, 3, 4, 5, 9, 10], size: 10.5, lh: 16 });
      txt("mem-readout", `V = ${n}: ` + rows.map(d => `${d.name} — BFS queue peaks at ${d.q}, DFS stack at ${d.s}`).join("; ") + `. Every traversal scanned exactly 2E adjacency entries ${ok(rows.every(d => d.sb === 2 * d.m && d.sd === 2 * d.m))}. Wide shapes (star, binary tree's last level, random) hurt BFS; deep shapes (path, grid's snake, random) hurt DFS.`);
    }
    on("mem-n", "change", draw); draw();
  });

})();
/* END-FIGURES */
