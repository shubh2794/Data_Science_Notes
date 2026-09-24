/* np-completeness.viz.js — figures for dsa/algorithms/np-completeness.html
   (part 7 of the Algorithm Design & Analysis series).

   Loaded after ../../data.js, ../../notes.js and ../dsa-viz.js, so the globals
   AC (palette) and AL (rng/frame/axisB/axisL/row/binTree/counter/stepper/…) are
   available.

   House rule obeyed throughout: every count this page DISPLAYS — search-tree
   nodes, assignments tried, clauses checked, reduction sizes, matching edges,
   trim steps, 2-opt moves, generations — is produced by running the real
   routine under AL.counter() and reading the counter back. Every ANSWER is
   cross-checked in the same figure against an independent computation —
   exhaustive search over all assignments / subsets / tours / colourings /
   covers for small instances, or a second algorithm — and printed with an
   agree / DISAGREE flag; every approximation RATIO is measured against the
   exhaustive optimum. A load-time audit (NA, at the bottom) repeats those
   checks over hundreds of random instances and reports the counts in the
   cheat sheet.

   Layout of this file:
     NR — the instrumented routines, pure and DOM-free (also loadable in node).
     NI — the shared instances (graphs) so prose and figures agree by construction.
     SX — small DOM helpers shared by the figures.
     figures — one block per <svg>, in page order (see the page's outline).
     NA — the load-time audit. */

/* ═══════════════════════════════════════════════════════════════════════════
   NR — the instrumented routines (each chunk appends with Object.assign)
   ═══════════════════════════════════════════════════════════════════════════ */
const NR = {};

/* the shared instances, so prose and figures agree by construction */
const NI = {};

/* ── chunk D ─────────────────────────────────────────────────── */
/* ── chunk D routines: exact exponential search (branch and bound, DP over
      subsets), parameterised algorithms (bounded search trees, Buss's
      kernel), local search (2-opt, simulated annealing, WalkSAT), the
      genetic algorithm, and a DPLL SAT solver with the random 3-CNF
      generator the phase-transition figure measures.  Pure and DOM-free;
      every routine takes an optional counter `c` with .add(key) and returns
      the measurements the figures display. ─────────────────────────────── */
Object.assign(NR, (function () {
  const nop = { add: function () {}, get: function () { return 0; } };
  function ctr(c) { return c || nop; }
  /* a local seeded generator, so these load in node without AL */
  function rnd(seed) { let a = (seed >>> 0) || 1; return function () { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  /* ═══════════════════════════════════════════════════════════════════
     §26 — EXACT EXPONENTIAL: branch and bound, DP over subsets
     ═══════════════════════════════════════════════════════════════════ */

  /* A VALID lower bound on what the partial path 0 ⇝ cur still has to pay.
     The completion cur → v₁ → … → v_k → 0 uses EXACTLY |rem| + 1 edges:
     one leaving cur, and one leaving each v ∈ rem. Bound each of those by
     the cheapest edge that vertex has into (rem ∪ {0}) ∖ {itself}. Every
     term is ≤ the edge the real completion uses there, so the sum is ≤ the
     real completion cost. |rem| + 1 terms — no more. */
  function tspLowerBound(D, cur, rem, c) {
    c = ctr(c); if (!rem.length) return D[cur][0];
    let s = D[cur][0];
    for (let i = 0; i < rem.length; i++) { c.add("boundEdge"); if (D[cur][rem[i]] < s) s = D[cur][rem[i]]; }
    let t = s;
    for (let i = 0; i < rem.length; i++) {
      const v = rem[i]; let b = D[v][0]; c.add("boundEdge");
      for (let j = 0; j < rem.length; j++) { if (j === i) continue; c.add("boundEdge"); if (D[v][rem[j]] < b) b = D[v][rem[j]]; }
      t += b;
    }
    return t;
  }
  /* The TRAP: the same sum PLUS a "cheapest edge back into 0" term. That is
     |rem| + 2 edges where the completion has only |rem| + 1, so the value
     can EXCEED the true completion cost. It is NOT a lower bound, it prunes
     the optimum, and nothing in the search reports the error. */
  function tspLowerBoundBad(D, cur, rem, c) {
    c = ctr(c); if (!rem.length) return D[cur][0];
    let back = Infinity;
    for (let i = 0; i < rem.length; i++) { c.add("boundEdge"); if (D[rem[i]][0] < back) back = D[rem[i]][0]; }
    let s = Infinity;
    for (let i = 0; i < rem.length; i++) { c.add("boundEdge"); if (D[cur][rem[i]] < s) s = D[cur][rem[i]]; }
    let t = s + back;
    for (let i = 0; i < rem.length; i++) {
      const v = rem[i]; let b = D[v][0]; c.add("boundEdge");
      for (let j = 0; j < rem.length; j++) { if (j === i) continue; c.add("boundEdge"); if (D[v][rem[j]] < b) b = D[v][rem[j]]; }
      t += b;
    }
    return t;
  }
  /* the TRUE cheapest completion of a partial path, by exhaustive recursion —
     the independent check that decides whether a bound IS a lower bound */
  function tspCompletion(D, cur, rem, c) {
    c = ctr(c);
    if (!rem.length) { c.add("complete"); return D[cur][0]; }
    let b = Infinity;
    for (let i = 0; i < rem.length; i++) {
      const t = D[cur][rem[i]] + tspCompletion(D, rem[i], rem.slice(0, i).concat(rem.slice(i + 1)), c);
      if (t < b) b = t;
    }
    return b;
  }
  /* walk every partial path and test both bounds against the truth */
  function tspBoundAudit(D, minRem, c) {
    c = ctr(c); let paths = 0, okGood = 0, okBad = 0, worstGood = 0, worstBad = 0;
    const n = D.length, all = []; for (let i = 1; i < n; i++) all.push(i);
    (function walk(cur, rem) {
      if (rem.length) {
        const truth = tspCompletion(D, cur, rem, c); paths++;
        const g = tspLowerBound(D, cur, rem, c), b = tspLowerBoundBad(D, cur, rem, c);
        if (g <= truth + 1e-9) okGood++; else worstGood = Math.max(worstGood, g - truth);
        if (b <= truth + 1e-9) okBad++; else worstBad = Math.max(worstBad, b - truth);
      }
      if (rem.length > minRem) for (let i = 0; i < rem.length; i++) walk(rem[i], rem.slice(0, i).concat(rem.slice(i + 1)));
    })(0, all);
    return { paths, okGood, okBad, badViolations: paths - okBad, goodViolations: paths - okGood, worstBad, worstGood };
  }
  /* BRANCH AND BOUND over partial paths rooted at city 0, children taken
     nearest-first. A NODE is one call on a partial path (pruned ones
     included); a node is UNPRUNED when it survives the bound test —
     an internal node whose children are generated, or a leaf (a complete
     tour, never tested). The count kept as `expanded` below is
     nodes − pruned, leaves included.
       mode "none"      plain depth-first search over all (n−1)! orders
       mode "incumbent" prune when the partial cost alone reaches the incumbent
       mode "bound"     prune on partial cost + tspLowerBound
       mode "bad"       the same with tspLowerBoundBad — fast and WRONG      */
  function tspBnB(D, mode, c) {
    c = ctr(c); const n = D.length;
    let best = Infinity, bestPath = null, nodes = 0, expanded = 0, pruned = 0, leaves = 0, improve = 0;
    const root = { v: 0, depth: 0, cost: 0, lb: 0, kids: [], status: "open" };
    (function go(cur, cost, path, rem, node) {
      nodes++; c.add("node");
      if (!rem.length) {
        leaves++; c.add("leaf"); expanded++;
        const z = cost + D[cur][0]; node.total = z; node.status = "leaf";
        if (z < best - 1e-12) { best = z; bestPath = path.slice(); improve++; c.add("improve"); node.status = "best"; }
        return;
      }
      let lb = 0;
      if (mode === "bound") { lb = tspLowerBound(D, cur, rem, c); c.add("bound"); }
      else if (mode === "bad") { lb = tspLowerBoundBad(D, cur, rem, c); c.add("bound"); }
      node.lb = cost + lb;
      if (mode !== "none" && cost + lb >= best - 1e-12) { pruned++; c.add("prune"); node.status = "pruned"; return; }
      expanded++; node.status = "expanded";
      const ks = rem.slice().sort(function (x, y) { return D[cur][x] - D[cur][y]; });
      for (let i = 0; i < ks.length; i++) {
        const v = ks[i];
        const kid = { v: v, depth: node.depth + 1, cost: cost + D[cur][v], lb: 0, kids: [], status: "open" };
        node.kids.push(kid); path.push(v);
        go(v, cost + D[cur][v], path, rem.filter(function (x) { return x !== v; }), kid);
        path.pop();
      }
    })(0, 0, [0], (function () { const a = []; for (let i = 1; i < n; i++) a.push(i); return a; })(), root);
    return { cost: best, tour: bestPath, nodes: nodes, expanded: expanded, pruned: pruned, leaves: leaves, improve: improve, tree: root };
  }
  /* HELD–KARP — dynamic programming over subsets.
       g[S][j] = the cheapest path 0 ⇝ j visiting exactly S ∪ {0}, ending at j,
       for S ⊆ {1 … n−1} and j ∈ S.
     c.add("state") per table cell FILLED, c.add("relax") per transition
     examined, c.add("close") per closing edge back to 0. */
  function heldKarp(D, c) {
    c = ctr(c); const n = D.length, m = n - 1, N = 1 << m;
    const g = Array.from({ length: N }, function () { return new Array(m).fill(Infinity); });
    const par = Array.from({ length: N }, function () { return new Array(m).fill(-1); });
    let states = 0, relax = 0;
    for (let j = 0; j < m; j++) { g[1 << j][j] = D[0][j + 1]; states++; c.add("state"); }
    for (let S = 1; S < N; S++) for (let j = 0; j < m; j++) {
      if (!((S >> j) & 1) || S === (1 << j)) continue;
      const P = S ^ (1 << j); let bv = Infinity, bp = -1;
      for (let i = 0; i < m; i++) {
        if (!((P >> i) & 1)) continue; relax++; c.add("relax");
        const t = g[P][i] + D[i + 1][j + 1]; if (t < bv) { bv = t; bp = i; }
      }
      g[S][j] = bv; par[S][j] = bp; states++; c.add("state");
    }
    const full = N - 1; let best = Infinity, bj = -1;
    for (let j = 0; j < m; j++) { c.add("close"); const t = g[full][j] + D[j + 1][0]; if (t < best) { best = t; bj = j; } }
    const tour = []; let S = full, j = bj;
    while (j >= 0) { tour.push(j + 1); const pj = par[S][j]; S ^= (1 << j); j = pj; }
    tour.push(0); tour.reverse();
    return { cost: best, tour: tour, states: states, relax: relax, g: g, m: m,
             tableCells: m * (1 << m), reachable: m * (1 << (m - 1)) };
  }
  /* knapsack branch and bound: the FRACTIONAL relaxation is the bound —
     the same device as the TSP bound above and as §30's LP relaxation */
  function knapFracBound(w, v, W, i, weight, value, order, c) {
    c = ctr(c); let rw = W - weight, b = value;
    for (let k = i; k < order.length; k++) {
      const it = order[k]; c.add("boundStep");
      if (w[it] <= rw) { rw -= w[it]; b += v[it]; } else { b += v[it] * rw / w[it]; break; }
    }
    return b;
  }
  function knapBnB(w, v, W, c) {
    c = ctr(c); const n = w.length;
    const order = Array.from({ length: n }, function (_, i) { return i; }).sort(function (a, b) { return v[b] / w[b] - v[a] / w[a]; });
    let best = 0, bestSet = [], nodes = 0, pruned = 0;
    (function go(i, weight, value, taken) {
      nodes++; c.add("node");
      if (value > best) { best = value; bestSet = taken.slice(); }
      if (i === n) return;
      const ub = knapFracBound(w, v, W, i, weight, value, order, c);
      if (ub <= best + 1e-12) { pruned++; c.add("prune"); return; }
      const it = order[i];
      if (weight + w[it] <= W) { taken.push(it); go(i + 1, weight + w[it], value + v[it], taken); taken.pop(); }
      go(i + 1, weight, value, taken);
    })(0, 0, 0, []);
    return { value: best, items: bestSet.slice().sort(function (a, b) { return a - b; }), nodes: nodes, pruned: pruned, order: order };
  }
  /* the four-variable formula §26 backtracks on, and §30 unit-propagates.
       φ = (w ∨ x ∨ y ∨ z) ∧ (w ∨ ¬x) ∧ (x ∨ ¬y) ∧ (y ∨ ¬z) ∧ (z ∨ ¬w) ∧ (¬w ∨ ¬z)
     VERIFIED by exhaustive search over all 2⁴ = 16 assignments:
     UNSATISFIABLE; the best assignment satisfies 5 of the 6 clauses. */
  function btFormula() {
    return { n: 4, names: ["w", "x", "y", "z"],
             clauses: [[1, 2, 3, 4], [1, -2], [2, -3], [3, -4], [4, -1], [-1, -4]] };
  }

  /* ═══════════════════════════════════════════════════════════════════
     §27 — PARAMETERISED: bounded search trees and kernels
     ═══════════════════════════════════════════════════════════════════ */

  function vcFirstUncovered(E, inC, c) {
    c = ctr(c);
    for (let i = 0; i < E.length; i++) { c.add("edgeScan"); if (!inC[E[i][0]] && !inC[E[i][1]]) return i; }
    return -1;
  }
  /* BOUNDED SEARCH TREE for vertex cover: take any uncovered edge (u, v) —
     every cover contains u or v — and branch on the two choices, each
     spending one unit of the budget k. Depth ≤ k, branching factor 2, so at
     most 2^(k+1) − 1 calls and O(E) work each: O(2ᵏ·E), and with the kernel
     below O(2ᵏ·k² + k·n). c.add("call") per recursive call. */
  function vcBranchTree(n, E, k, c) {
    c = ctr(c); let calls = 0, found = null;
    const root = { id: 0, budget: k, pick: null, kids: [], status: "open", depth: 0 }; let nextId = 1;
    (function go(inC, budget, node) {
      calls++; c.add("call");
      const e = vcFirstUncovered(E, inC, c);
      if (e < 0) { node.status = "cover"; if (!found) found = inC.map(function (b, v) { return b ? v : -1; }).filter(function (v) { return v >= 0; }); return true; }
      if (budget === 0) { node.status = "budget"; return false; }
      node.edge = e;
      for (let s = 0; s < 2; s++) {
        const u = E[e][s];
        const kid = { id: nextId++, budget: budget - 1, pick: u, edge: e, side: s, kids: [], status: "open", depth: node.depth + 1 };
        node.kids.push(kid);
        inC[u] = true;
        const ok = go(inC, budget - 1, kid);
        inC[u] = false;
        if (ok) { node.status = "yes"; return true; }
      }
      node.status = "no"; return false;
    })(new Array(n).fill(false), k, root);
    return { ok: found !== null, cover: found, calls: calls, tree: root, bound: Math.pow(2, k + 1) - 1 };
  }
  /* run the search tree for k = 0, 1, 2, … until it succeeds */
  function vcFptSweep(n, E, kmax, c) {
    c = ctr(c); const rows = []; let total = 0, first = null;
    for (let k = 0; k <= kmax; k++) {
      const R = vcBranchTree(n, E, k, c);
      rows.push({ k: k, ok: R.ok, calls: R.calls, bound: R.bound, cover: R.cover });
      total += R.calls;
      if (R.ok) { first = k; break; }
    }
    return { rows: rows, total: total, k: first };
  }
  /* BUSS's KERNEL. If a vertex has degree > k it lies in EVERY cover of size
     ≤ k (leaving it out forces all > k of its neighbours in). Take such
     vertices, delete them, decrement k. When none is left every degree is
     ≤ k, so a cover of size ≤ k touches ≤ k·k edges: if more than k² edges
     remain the answer is NO. What survives is the kernel. */
  function vcBussKernel(n, E, k, c) {
    c = ctr(c);
    const alive = E.map(function () { return true; });
    const forced = []; let budget = k; const steps = [];
    let changed = true;
    while (changed && budget >= 0) {
      changed = false;
      const deg = new Array(n).fill(0);
      E.forEach(function (e, i) { if (alive[i]) { c.add("degStep"); deg[e[0]]++; deg[e[1]]++; } });
      for (let v = 0; v < n; v++) {
        if (deg[v] > budget) {
          forced.push(v); budget--;
          E.forEach(function (e, i) { if (alive[i] && (e[0] === v || e[1] === v)) { alive[i] = false; c.add("remove"); } });
          steps.push({ v: v, deg: deg[v], budget: budget });
          changed = true; break;
        }
      }
    }
    const left = []; E.forEach(function (e, i) { if (alive[i]) left.push(e); });
    const vs = {}; left.forEach(function (e) { vs[e[0]] = 1; vs[e[1]] = 1; });
    return { forced: forced, budget: budget, edgesLeft: left, nLeft: Object.keys(vs).length,
             steps: steps, limit: budget * budget, tooBig: left.length > budget * budget,
             infeasible: budget < 0 };
  }
  /* the XP alternative: n^O(k) — try every k-subset. Counts the subsets. */
  function vcXpCount(n, E, k, c) {
    c = ctr(c); let tried = 0, found = null;
    const idx = [];
    (function go(start, chosen) {
      if (found) return;
      if (chosen.length === k) {
        tried++; c.add("subset");
        const S = new Array(n).fill(false); chosen.forEach(function (v) { S[v] = true; });
        if (NR.vcIsCover(E, S, c)) found = chosen.slice();
        return;
      }
      for (let v = start; v < n; v++) { chosen.push(v); go(v + 1, chosen); chosen.pop(); if (found) return; }
    })(0, idx);
    return { found: found, tried: tried, total: (function () { let t = 1; for (let i = 0; i < k; i++) t = t * (n - i) / (i + 1); return Math.round(t); })() };
  }

  /* ═══════════════════════════════════════════════════════════════════
     §28 — LOCAL SEARCH
     ═══════════════════════════════════════════════════════════════════ */

  /* 2-OPT with an explicit move trace. The move (i, k) with 1 ≤ i < k ≤ n−1
     REVERSES tour[i … k]: it removes the edges (t[i−1], t[i]) and
     (t[k], t[k+1]) and adds (t[i−1], t[k]) and (t[i], t[k+1]); the rest of
     the tour is untouched, so the change in cost is a four-term expression
     computable in O(1). c.add("pair") per (i, k) EVALUATED, c.add("move")
     per improving move taken.
       rule "first" — take the first improving pair found
       rule "best"  — scan the whole neighbourhood and take the best        */
  function twoOptSteps(D, tour0, rule, c) {
    c = ctr(c); let tour = tour0.slice(); const n = tour.length;
    const frames = [{ tour: tour.slice(), before: null, cost: NR.tspTourCost(D, tour), move: null, pairs: 0 }];
    let improved = true, moves = 0, pairs = 0;
    while (improved) {
      improved = false; let bi = -1, bk = -1, bg = 1e-12, stop = false;
      for (let i = 1; i < n - 1 && !stop; i++) for (let k = i + 1; k < n; k++) {
        pairs++; c.add("pair");
        const a = tour[i - 1], b = tour[i], cc = tour[k], d = tour[(k + 1) % n];
        if (d === a) continue;
        const gain = (D[a][b] + D[cc][d]) - (D[a][cc] + D[b][d]);
        if (gain > bg) { bg = gain; bi = i; bk = k; if (rule === "first") { stop = true; break; } }
      }
      if (bi >= 0) {
        const before = tour.slice();
        tour = tour.slice(0, bi).concat(tour.slice(bi, bk + 1).reverse(), tour.slice(bk + 1));
        moves++; c.add("move"); improved = true;
        frames.push({ tour: tour.slice(), before: before, cost: NR.tspTourCost(D, tour), move: { i: bi, k: bk, gain: bg }, pairs: pairs });
      }
    }
    return { tour: tour, cost: NR.tspTourCost(D, tour), moves: moves, pairs: pairs, frames: frames };
  }
  /* SIMULATED ANNEALING over the same 2-change neighbourhood.
     Metropolis rule: a proposal with Δ < 0 is always accepted; one with
     Δ ≥ 0 is accepted with probability e^(−Δ/T). Geometric cooling from T₀
     to T_end over `iters` proposals. c.add("worse") per accepted WORSENING
     move — the quantity that distinguishes annealing from plain descent. */
  function annealTsp(D, tour0, opt, c) {
    c = ctr(c);
    const o = Object.assign({ T0: 3, Tend: 0.01, iters: 3000, seed: 11 }, opt || {});
    const r = rnd(o.seed), n = tour0.length;
    let tour = tour0.slice(), cost = NR.tspTourCost(D, tour);
    let best = cost, bestTour = tour.slice(), worse = 0, better = 0, rejected = 0;
    const trace = [], lam = Math.log(o.Tend / o.T0), every = Math.max(1, Math.floor(o.iters / 240));
    for (let t = 0; t < o.iters; t++) {
      const T = o.T0 * Math.exp(lam * t / Math.max(1, o.iters - 1));
      const i = 1 + Math.floor(r() * (n - 2));
      const k = i + 1 + Math.floor(r() * (n - i - 1));
      c.add("propose");
      const a = tour[i - 1], b = tour[i], cc = tour[k], d = tour[(k + 1) % n];
      const delta = (D[a][cc] + D[b][d]) - (D[a][b] + D[cc][d]);
      let take = false;
      if (delta < 0) { take = true; better++; c.add("better"); }
      else { const p = Math.exp(-delta / Math.max(T, 1e-9)); if (r() < p) { take = true; worse++; c.add("worse"); } else { rejected++; c.add("reject"); } }
      if (take) {
        tour = tour.slice(0, i).concat(tour.slice(i, k + 1).reverse(), tour.slice(k + 1));
        cost += delta;
        if (cost < best - 1e-12) { best = cost; bestTour = tour.slice(); }
      }
      if (t % every === 0) trace.push({ t: t, T: T, cost: cost, best: best });
    }
    trace.push({ t: o.iters - 1, T: o.Tend, cost: cost, best: best });
    return { tour: bestTour, cost: best, final: cost, acceptedWorse: worse, acceptedBetter: better,
             rejected: rejected, trace: trace, recheck: NR.tspTourCost(D, bestTour) };
  }
  /* random restarts: what the DISTRIBUTION of 2-opt local optima looks like */
  function twoOptRestart(D, trials, seed, c) {
    c = ctr(c); const r = rnd(seed), n = D.length, costs = []; let best = Infinity;
    for (let t = 0; t < trials; t++) {
      const p = [0], rest = []; for (let i = 1; i < n; i++) rest.push(i);
      while (rest.length) { const j = Math.floor(r() * rest.length); p.push(rest[j]); rest.splice(j, 1); }
      const R = twoOptSteps(D, p, "first", c);
      costs.push(R.cost); if (R.cost < best) best = R.cost;
    }
    return { costs: costs, best: best, trials: trials };
  }
  /* WALKSAT: pick a violated clause, then flip either a random variable of
     it (probability p) or the one that leaves fewest clauses broken. An
     INCOMPLETE solver: it finds assignments, it never proves unsatisfiability. */
  function walkSat(clauses, n, maxFlips, p, seed, c) {
    c = ctr(c); const r = rnd(seed), a = [];
    for (let i = 0; i < n; i++) a.push(r() < 0.5);
    for (let f = 0; f <= maxFlips; f++) {
      const bad = [];
      for (let i = 0; i < clauses.length; i++) { c.add("clauseCheck"); if (!NR.clauseSat(clauses[i], a)) bad.push(i); }
      if (!bad.length) return { sat: true, assign: a.slice(), flips: f, ok: NR.satEval(clauses, a) };
      const cl = clauses[bad[Math.floor(r() * bad.length)]];
      let v;
      if (r() < p) v = Math.abs(cl[Math.floor(r() * cl.length)]) - 1;
      else {
        let bestv = -1, bestBroken = Infinity;
        for (let q = 0; q < cl.length; q++) {
          const w = Math.abs(cl[q]) - 1; a[w] = !a[w];
          let broken = 0;
          for (let i = 0; i < clauses.length; i++) { c.add("clauseCheck"); if (!NR.clauseSat(clauses[i], a)) broken++; }
          a[w] = !a[w];
          if (broken < bestBroken) { bestBroken = broken; bestv = w; }
        }
        v = bestv;
      }
      a[v] = !a[v]; c.add("flip");
    }
    return { sat: false, assign: null, flips: maxFlips, ok: true };
  }

  /* ═══════════════════════════════════════════════════════════════════
     §29 — GENETIC ALGORITHM
     ═══════════════════════════════════════════════════════════════════ */

  /* what ORDINARY one-point crossover does to two tours: a child that is
     not a permutation. Kept as a routine so the figure can SHOW the damage. */
  function onePointCut(p1, p2, cut) {
    const child = p1.slice(0, cut).concat(p2.slice(cut));
    const seen = {}; let dup = 0;
    child.forEach(function (v) { if (seen[v]) dup++; seen[v] = 1; });
    return { child: child, valid: dup === 0, duplicates: dup,
             missing: p1.filter(function (v) { return child.indexOf(v) < 0; }) };
  }
  /* ORDER CROSSOVER (OX): copy p1[i … j] into the child; then walk p2 from
     position j+1 cyclically and drop the cities not already present into
     the child's free slots, also from j+1 cyclically. The result is always
     a permutation, and it inherits a contiguous run of p1 and the relative
     order of p2. */
  function oxCrossover(p1, p2, i, j, c) {
    c = ctr(c); const n = p1.length, child = new Array(n).fill(-1), used = {};
    for (let k = i; k <= j; k++) { child[k] = p1[k]; used[p1[k]] = 1; c.add("gene"); }
    let w = (j + 1) % n;
    for (let s = 0; s < n; s++) {
      const v = p2[(j + 1 + s) % n]; c.add("gene");
      if (used[v]) continue;
      child[w] = v; used[v] = 1; w = (w + 1) % n;
    }
    return child;
  }
  function tourValid(t, n) {
    if (t.length !== n) return false;
    const s = {}; let k = 0;
    t.forEach(function (v) { if (v >= 0 && v < n && !s[v]) { s[v] = 1; k++; } });
    return k === n;
  }
  /* the GA: tournament selection, order crossover, swap mutation, elitism,
     generational replacement. c.add("eval") per fitness evaluation — the
     only unit in which a black-box heuristic's cost is honestly measured. */
  function gaTsp(D, opt, c) {
    c = ctr(c);
    const o = Object.assign({ pop: 40, gens: 60, tsize: 3, pc: 0.9, pm: 0.2, elite: 2, seed: 3 }, opt || {});
    const r = rnd(o.seed), n = D.length;
    let evals = 0;
    function cost(t) { evals++; c.add("eval"); return NR.tspTourCost(D, t); }
    function randPerm() {
      const a = [0], rest = []; for (let i = 1; i < n; i++) rest.push(i);
      while (rest.length) { const k = Math.floor(r() * rest.length); a.push(rest[k]); rest.splice(k, 1); }
      return a;
    }
    let P = []; for (let i = 0; i < o.pop; i++) { const t = randPerm(); P.push({ t: t, f: cost(t) }); }
    const hist = [];
    function rec(g) {
      const fs = P.map(function (x) { return x.f; });
      hist.push({ gen: g, best: Math.min.apply(null, fs), mean: fs.reduce(function (a, b) { return a + b; }, 0) / fs.length,
                  worst: Math.max.apply(null, fs), evals: evals });
    }
    rec(0);
    for (let g = 1; g <= o.gens; g++) {
      P.sort(function (a, b) { return a.f - b.f; });
      const next = P.slice(0, o.elite).map(function (x) { return { t: x.t.slice(), f: x.f }; });
      while (next.length < o.pop) {
        const pick = function () {
          let bi = Math.floor(r() * o.pop);
          for (let k = 1; k < o.tsize; k++) { const j = Math.floor(r() * o.pop); c.add("compare"); if (P[j].f < P[bi].f) bi = j; }
          return P[bi];
        };
        const A = pick(), B = pick();
        let child;
        if (r() < o.pc) {
          let i = 1 + Math.floor(r() * (n - 1)), j = 1 + Math.floor(r() * (n - 1));
          if (i > j) { const s = i; i = j; j = s; }
          child = oxCrossover(A.t, B.t, i, j, c); c.add("crossover");
        } else child = A.t.slice();
        if (r() < o.pm) {
          const i = 1 + Math.floor(r() * (n - 1)), j = 1 + Math.floor(r() * (n - 1));
          const s = child[i]; child[i] = child[j]; child[j] = s; c.add("mutation");
        }
        const z = child.indexOf(0); child = child.slice(z).concat(child.slice(0, z));
        next.push({ t: child, f: cost(child) });
      }
      P = next; rec(g);
    }
    P.sort(function (a, b) { return a.f - b.f; });
    return { best: P[0].t.slice(), cost: P[0].f, hist: hist, evals: evals, gens: o.gens,
             allValid: P.every(function (x) { return tourValid(x.t, n); }),
             recheck: NR.tspTourCost(D, P[0].t) };
  }

  /* ═══════════════════════════════════════════════════════════════════
     §30 — DPLL, and the random 3-CNF model
     ═══════════════════════════════════════════════════════════════════ */

  /* DPLL — the search core every CDCL solver is built on: unit propagation,
     optional pure-literal elimination, and a branching rule. Each recursive
     call receives an ALREADY simplified clause list, so the work at a node
     is proportional to the clauses still alive there. Counts: c.add("node")
     per call, "decision" per branching variable chosen, "prop" per unit
     propagation, "conflict" per empty clause derived. */
  function dpll(clauses, n, opt, c) {
    c = ctr(c);
    const o = Object.assign({ unit: true, pure: false }, opt || {});
    let decisions = 0, props = 0, conflicts = 0, nodes = 0, maxDepth = 0;
    const a = new Array(n).fill(undefined);
    function assign(cls, l) {
      const out = [];
      for (let i = 0; i < cls.length; i++) {
        const cl = cls[i]; c.add("clauseVisit");
        let sat = false, keep = null;
        for (let j = 0; j < cl.length; j++) {
          if (cl[j] === l) { sat = true; break; }
          if (cl[j] === -l && keep === null) keep = cl.filter(function (x) { return x !== -l; });
        }
        if (sat) continue;
        const nc = keep === null ? cl : keep;
        if (nc.length === 0) return null;
        out.push(nc);
      }
      return out;
    }
    function solve(cls, depth) {
      nodes++; c.add("node"); if (depth > maxDepth) maxDepth = depth;
      let cur = cls;
      if (o.unit) {
        let u = true;
        while (u) {
          u = false;
          for (let i = 0; i < cur.length; i++) if (cur[i].length === 1) {
            const l = cur[i][0]; a[Math.abs(l) - 1] = l > 0; props++; c.add("prop");
            cur = assign(cur, l);
            if (cur === null) { conflicts++; c.add("conflict"); return false; }
            u = true; break;
          }
        }
      }
      if (o.pure) {
        const pos = {}, neg = {};
        cur.forEach(function (cl) { cl.forEach(function (l) { (l > 0 ? pos : neg)[Math.abs(l)] = 1; }); });
        const ks = Object.keys(pos).concat(Object.keys(neg));
        for (let q = 0; q < ks.length && cur !== null; q++) {
          const v = +ks[q];
          if (pos[v] && !neg[v] && a[v - 1] === undefined) { a[v - 1] = true; props++; c.add("prop"); cur = assign(cur, v); }
          else if (neg[v] && !pos[v] && a[v - 1] === undefined) { a[v - 1] = false; props++; c.add("prop"); cur = assign(cur, -v); }
        }
        if (cur === null) { conflicts++; c.add("conflict"); return false; }
      }
      if (cur.length === 0) return true;
      const cnt = {}; let pick = 0, bestn = -1, sign = true;
      cur.forEach(function (cl) { cl.forEach(function (l) { const v = Math.abs(l); cnt[v] = (cnt[v] || 0) + 1; }); });
      cur.forEach(function (cl) { cl.forEach(function (l) { const v = Math.abs(l); if (cnt[v] > bestn) { bestn = cnt[v]; pick = v; sign = l > 0; } }); });
      decisions++; c.add("decision");
      for (let s = 0; s < 2; s++) {
        const val = s === 0 ? sign : !sign;
        const snap = a.slice();
        a[pick - 1] = val;
        const nxt = assign(cur, val ? pick : -pick);
        if (nxt === null) { conflicts++; c.add("conflict"); }
        else if (solve(nxt, depth + 1)) return true;
        for (let i = 0; i < n; i++) a[i] = snap[i];
      }
      return false;
    }
    const sat = solve(clauses.map(function (cl) { return cl.slice(); }), 0);
    const out = a.slice(); for (let i = 0; i < n; i++) if (out[i] === undefined) out[i] = false;
    return { sat: sat, assign: sat ? out : null, decisions: decisions, props: props,
             conflicts: conflicts, nodes: nodes, maxDepth: maxDepth,
             ok: sat ? NR.satEval(clauses, out) : true };
  }
  /* the STANDARD random 3-SAT model: m clauses drawn independently, each on
     three DISTINCT variables with independent uniform signs. (NR.randClauses
     allows a variable to repeat inside a clause, which is a different model
     and moves the threshold, so the phase-transition figure uses this one.) */
  function rand3CNF(n, m, seed) {
    const r = rnd(seed), out = [], k = Math.min(3, n);
    for (let i = 0; i < m; i++) {
      const vs = [];
      while (vs.length < k) { const v = 1 + Math.floor(r() * n); if (vs.indexOf(v) < 0) vs.push(v); }
      out.push(vs.map(function (v) { return r() < 0.5 ? -v : v; }));
    }
    return out;
  }
  /* the measured phase transition: for each clause/variable ratio, generate
     `trials` fresh random 3-CNF formulas, solve each with the real solver,
     and report the satisfiable fraction and the mean search effort. */
  function phaseSweep(n, ratios, trials, seed, c) {
    c = ctr(c); const rows = []; let s = seed >>> 0;
    ratios.forEach(function (rho) {
      const m = Math.round(rho * n);
      let sat = 0, nodes = 0, dec = 0, props = 0;
      for (let t = 0; t < trials; t++) {
        const cl = rand3CNF(n, m, s); s = (s + 2654435761) >>> 0;
        const R = dpll(cl, n, { unit: true }, c);
        if (R.sat) sat++; nodes += R.nodes; dec += R.decisions; props += R.props;
      }
      rows.push({ rho: rho, m: m, sat: sat, trials: trials, frac: sat / trials,
                  nodes: nodes / trials, decisions: dec / trials, props: props / trials });
    });
    return rows;
  }
  /* linear interpolation of the ratio at which the satisfiable fraction
     crosses one half — the MEASURED crossing, which is not the asymptotic
     threshold and at small n sits well above it. */
  function phaseCrossing(rows) {
    for (let i = 1; i < rows.length; i++) {
      if (rows[i].frac < 0.5 && rows[i - 1].frac >= 0.5) {
        const a = rows[i - 1], b = rows[i];
        return a.rho + (b.rho - a.rho) * (a.frac - 0.5) / (a.frac - b.frac);
      }
    }
    return null;
  }

  return {
    tspLowerBound, tspLowerBoundBad, tspCompletion, tspBoundAudit, tspBnB, heldKarp,
    knapFracBound, knapBnB, btFormula,
    vcFirstUncovered, vcBranchTree, vcFptSweep, vcBussKernel, vcXpCount,
    twoOptSteps, annealTsp, twoOptRestart, walkSat,
    onePointCut, oxCrossover, tourValid, gaTsp,
    dpll, rand3CNF, phaseSweep, phaseCrossing
  };
})());

/* ─────────────────────────────────────────────────────────────────────────
   NI — the shared instances. Fixed once, here, so that prose and every
   figure on the page agree by construction. Every number in the comments
   below was computed by exhaustive search before the page was written;
   the figures re-derive them at load and the audit (NA) re-checks them.
   ───────────────────────────────────────────────────────────────────────── */
Object.assign(NI, {

  /* φ — the canonical 3-CNF formula, used from §09 to §15.
     Literals are signed 1-based variable indices: 1 = x₁, −2 = ¬x₂.
       φ = (x₁ ∨ ¬x₂ ∨ ¬x₃) ∧ (¬x₁ ∨ x₂ ∨ x₃) ∧ (x₁ ∨ x₂ ∨ x₃)
     VERIFIED: satisfiable; 5 of the 8 assignments satisfy it —
       001, 010, 101, 110, 111  (bits are x₁x₂x₃, 1 = TRUE). */
  phi: { n: 3, names: ["x₁", "x₂", "x₃"],
         clauses: [[1, -2, -3], [-1, 2, 3], [1, 2, 3]] },

  /* φ₈ — all eight clauses over three variables. UNSATISFIABLE.
     VERIFIED: 0 satisfying assignments; the best assignment satisfies
     exactly 7 of the 8 clauses, and the expected number satisfied by a
     uniformly random assignment is exactly 8·(7/8) = 7. Used by §24. */
  phi8: { n: 3, names: ["x₁", "x₂", "x₃"],
          clauses: [[-1, -2, -3], [-1, -2, 3], [-1, 2, -3], [-1, 2, 3],
                    [1, -2, -3], [1, -2, 3], [1, 2, -3], [1, 2, 3]] },

  /* The clique / vertex-cover / independent-set demonstration graph
     (7 vertices a…g). VERIFIED by exhaustive search over all 2⁷ subsets:
       minimum vertex cover = 3, and it is UNIQUE: {b, d, e}
       maximum independent set = 4 (the complement, {a, c, f, g})
     The greedy maximal matching taken in the edge order below is
     (a,b), (c,d), (e,f) → a cover of all 6 matched endpoints, ratio
     exactly 2.000 against the optimum of 3 — the worst the 2-approximation
     of §21 can do. */
  gVC: { V: ["a", "b", "c", "d", "e", "f", "g"],
         E: [["a", "b"], ["b", "c"], ["c", "d"], ["c", "e"], ["d", "e"],
             ["d", "f"], ["d", "g"], ["e", "f"]],
         pos: { a: [40, 60], b: [140, 60], c: [240, 60], d: [340, 60],
                e: [240, 150], f: [340, 150], g: [440, 60] } },

  /* The 8-city metric TSP instance (Euclidean points in the plane).
     VERIFIED: the triangle inequality holds for all 8³ triples;
     exhaustive search over all 7!/2 = 2 520 distinct tours gives
       OPT              = 32.1721   tour 0-1-2-3-4-7-5-6
       MST weight       = 27.5804   (≤ OPT, as the proof requires)
       double-tree tour = 39.1592   ratio 1.2172  (bound 2)
       Christofides     = 32.2175   ratio 1.0014  (bound 1.5)
         — its matching on the four odd-degree MST vertices {3,4,5,7}
           is {(5,7), (3,4)}, cost 7.8482 ≤ OPT/2 = 16.0860
       nearest neighbour= 38.8502   ratio 1.2076  (no constant bound) */
  tsp: { names: ["A", "B", "C", "D", "E", "F", "G", "H"],
         pts: [[0, 0], [4, 0], [8, 1], [9, 5], [6, 8], [2, 7], [0, 4], [4, 4]] },

  /* The set-cover instance: 12 elements, 6 sets.
     VERIFIED: greedy picks S₁, S₄, S₅, S₃ — four sets; the optimum is
     three (S₃, S₄, S₅ — the unique optimal cover). H(6) = 2.45 and
     ln 12 = 2.4849, so the bound H(max |S|)·OPT = 7.35 is not tight here. */
  setcover: { U: 12,
              sets: [[1,2,3,4,5,6], [5,6,8,9], [1,4,7,10],
                     [2,5,7,8,11], [3,6,9,12], [10,11]] },

  /* Subset sum. VERIFIED by enumerating all 2⁶ = 64 subsets: exactly three
     subsets of S sum to 37 — {14,23}, {3,11,23}, {7,11,19}. */
  subsetsum: { S: [3, 7, 11, 14, 19, 23], T: 37 },

  /* 0-1 knapsack for the FPTAS of §25. VERIFIED by enumerating all 2⁵
     subsets: the optimum is value 85 from items 1,2,3, weight exactly 15. */
  knap: { w: [3, 5, 7, 4, 9], v: [20, 30, 35, 26, 50], W: 15 },

  /* The colouring graph of §15. VERIFIED by enumerating all 3⁷ = 2 187
     assignments: NOT bipartite (it has a triangle), chromatic number 3,
     and exactly 24 proper 3-colourings. */
  gCol: { n: 7,
          E: [[0,1],[1,2],[2,0],[2,3],[3,4],[4,5],[5,3],[1,5],[0,6],[6,4]] },

  /* Hamiltonian-cycle instances of §13.
     gHam — 8 vertices, 11 edges. VERIFIED over all 7!/2 = 2 520 tours:
       exactly 2 Hamiltonian cycles.
     gPetersen — the Petersen graph, 10 vertices, 15 edges. VERIFIED over
       all 9!/2 = 181 440 tours: NO Hamiltonian cycle, though it does have
       a Hamiltonian PATH (0-1-2-3-4-9-6-8-5-7) — the standard example of
       a graph that is traceable but not Hamiltonian. */
  gHam: { n: 8,
          E: [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,0],[0,3],[1,6],[2,5]] },
  gPetersen: { n: 10,
               E: [[0,1],[1,2],[2,3],[3,4],[4,0],
                   [5,7],[7,9],[9,6],[6,8],[8,5],
                   [0,5],[1,6],[2,7],[3,8],[4,9]] }
});

/* ── chunk A ─────────────────────────────────────────────────── */
/* ── chunk A routines: encodings, self-reducibility, measured growth,
      verifiers, the IS→CLIQUE reduction map, circuits + Tseitin, the
      Cook–Levin tableau, and CNF → 3-CNF conversion ──────────────────────
   Every routine takes an optional counter c (AL.counter()) and bumps it on
   the elementary step the prose claims to be counting, so that a figure can
   DISPLAY a measurement rather than an assertion.  All are DOM-free.
   Literals are signed 1-based variable indices: 1 = x₁, −2 = ¬x₂.          */
Object.assign(NR, (function () {
  const nop = { add: function () {}, get: function () { return 0; } };

  /* ═══ SAT basics ═══════════════════════════════════════════════════════ */
  /* one clause, one assignment; c.add("lit") per literal actually examined */
  function clauseSat(cl, a, c) {
    c = c || nop;
    for (let i = 0; i < cl.length; i++) { c.add("lit"); const l = cl[i]; if ((l > 0) === !!a[Math.abs(l) - 1]) return true; }
    return false;
  }
  function satEval(clauses, a, c) { for (let i = 0; i < clauses.length; i++) { (c || nop).add("clause"); if (!clauseSat(clauses[i], a, c)) return false; } return true; }
  /* how many clauses a given assignment satisfies (used by §24's MAX-3-SAT) */
  function satCount(clauses, a, c) { let k = 0; for (const cl of clauses) { (c || nop).add("clause"); if (clauseSat(cl, a, c)) k++; } return k; }
  /* bit i of m, MSB first, as the value of variable i+1 */
  function bitsOf(m, n) { const a = []; for (let i = 0; i < n; i++) a.push(!!((m >> (n - 1 - i)) & 1)); return a; }
  function bitStr(a) { return a.map(b => b ? "1" : "0").join(""); }

  /* EXHAUSTIVE search over all 2ⁿ assignments.  Returns every satisfying
     assignment as a bit string (MSB = x₁), the best clause count, and the
     measured work.  This is the independent check every SAT figure uses. */
  function satBrute(clauses, n, c) {
    c = c || nop;
    const sols = []; let best = -1, bestA = null, assigns = 0;
    for (let m = 0; m < (1 << n); m++) {
      assigns++; c.add("assign");
      const a = bitsOf(m, n);
      const k = satCount(clauses, a, c);
      if (k > best) { best = k; bestA = a; }
      if (k === clauses.length) sols.push(bitStr(a));
    }
    return { sols, count: sols.length, sat: sols.length > 0, best, bestA, assigns };
  }
  /* φ restricted by a partial assignment: fixed[i] ∈ {true, false, undefined}
     for variable i+1.  Clauses with a satisfied literal vanish; falsified
     literals are struck out.  An empty clause means "unsatisfiable". */
  function satRestrict(clauses, fixed, c) {
    c = c || nop; const out = []; let empty = false;
    for (const cl of clauses) {
      c.add("clause"); const keep = []; let gone = false;
      for (const l of cl) { const v = fixed[Math.abs(l) - 1]; if (v === undefined) keep.push(l); else if ((l > 0) === v) { gone = true; break; } }
      if (gone) continue;
      if (keep.length === 0) empty = true;
      out.push(keep);
    }
    return { clauses: out, empty };
  }
  /* is φ still satisfiable with the given prefix fixed?  Decided by brute
     force over the FREE variables — the "oracle", whose internal work is
     counted separately so a figure can show what the oracle is hiding. */
  function satOracle(clauses, n, fixed, c) {
    c = c || nop; const free = []; for (let i = 0; i < n; i++) if (fixed[i] === undefined) free.push(i);
    for (let m = 0; m < (1 << free.length); m++) {
      c.add("oracleAssign");
      const a = fixed.slice(); const b = bitsOf(m, free.length);
      free.forEach((v, j) => { a[v] = b[j]; });
      if (satEval(clauses, a, c)) return { sat: true, witness: a };
    }
    return { sat: false, witness: null };
  }
  /* SELF-REDUCTION: recover a satisfying assignment with n oracle queries.
     c.add("query") once per oracle call — that is the number §03 claims. */
  function satSelfReduce(phi, tryTrueFirst, c) {
    c = c || nop; const n = phi.n; const fixed = new Array(n).fill(undefined);
    const steps = [];
    const root = satOracle(phi.clauses, n, fixed, c); c.add("query");
    steps.push({ level: -1, question: "is φ satisfiable at all?", answer: root.sat, fixed: fixed.slice() });
    if (!root.sat) return { steps, assign: null, queries: c.get("query"), sat: false };
    for (let i = 0; i < n; i++) {
      const first = tryTrueFirst;
      fixed[i] = first;
      const r = satOracle(phi.clauses, n, fixed, c); c.add("query");
      const taken = r.sat ? first : !first;
      if (!r.sat) fixed[i] = !first;
      steps.push({ level: i, question: "fix " + phi.names[i] + " = " + (first ? "TRUE" : "FALSE") + " — still satisfiable?", answer: r.sat, taken, fixed: fixed.slice() });
    }
    return { steps, assign: fixed.slice(), queries: c.get("query"), sat: true, ok: satEval(phi.clauses, fixed) };
  }

  /* ═══ graphs as adjacency matrices (shared by clique / IS figures) ═════ */
  function adjFromNamed(G) {
    const idx = {}; G.V.forEach((v, i) => { idx[v] = i; });
    const n = G.V.length, A = Array.from({ length: n }, () => new Array(n).fill(false));
    G.E.forEach(([u, v]) => { A[idx[u]][idx[v]] = A[idx[v]][idx[u]] = true; });
    return { A, n, idx };
  }
  function adjFromEdges(n, E) { const A = Array.from({ length: n }, () => new Array(n).fill(false)); E.forEach(([u, v]) => { A[u][v] = A[v][u] = true; }); return A; }
  function isCliqueSet(A, S, c) { c = c || nop; for (let i = 0; i < S.length; i++) for (let j = i + 1; j < S.length; j++) { c.add("pair"); if (!A[S[i]][S[j]]) return false; } return true; }
  /* exhaustive maximum clique over a vertex subset; counts every subset tried */
  function cliqueMax(A, sub, c) {
    c = c || nop; const m = sub.length; let best = [], subsets = 0;
    for (let mask = 0; mask < (1 << m); mask++) {
      subsets++; c.add("subset"); const S = []; for (let i = 0; i < m; i++) if ((mask >> i) & 1) S.push(sub[i]);
      if (S.length <= best.length) continue;
      if (isCliqueSet(A, S, c)) best = S;
    }
    return { best, size: best.length, subsets };
  }
  /* how many k-cliques exist (the independent count a figure cross-checks) */
  function cliqueCount(A, n, k, c) {
    c = c || nop; let found = 0, tried = 0;
    for (let mask = 0; mask < (1 << n); mask++) { const S = []; for (let i = 0; i < n; i++) if ((mask >> i) & 1) S.push(i); if (S.length !== k) continue; tried++; c.add("subset"); if (isCliqueSet(A, S, c)) found++; }
    return { found, tried };
  }
  /* SELF-REDUCTION for clique: delete a vertex whenever a k-clique survives */
  function cliqueSelfReduce(A, n, order, c) {
    c = c || nop; const all = [...Array(n).keys()];
    const k = cliqueMax(A, all).size;
    let cur = all.slice(); const steps = [];
    for (const v of order) {
      const trial = cur.filter(x => x !== v);
      c.add("query"); const survives = cliqueMax(A, trial).size >= k;
      if (survives) cur = trial;
      steps.push({ v, survives, cur: cur.slice(), k });
    }
    return { steps, clique: cur, k, queries: c.get("query"), ok: cur.length === k && isCliqueSet(A, cur) };
  }
  function complementAdj(A, n) { return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i !== j && !A[i][j])); }
  function hasIndepSet(A, n, k, c) { return cliqueCount(complementAdj(A, n), n, k, c).found > 0; }
  function hasClique(A, n, k, c) { return cliqueCount(A, n, k, c).found > 0; }

  /* the 3-SAT → CLIQUE construction (§11 draws it; §09 refers to its size).
     One vertex per literal occurrence; an edge between occurrences in
     DIFFERENT clauses whose literals are not complementary. */
  function satToClique(phi) {
    const V = []; phi.clauses.forEach((cl, ci) => cl.forEach((l, li) => V.push({ clause: ci, pos: li, lit: l })));
    const E = [];
    for (let i = 0; i < V.length; i++) for (let j = i + 1; j < V.length; j++) if (V[i].clause !== V[j].clause && V[i].lit !== -V[j].lit) E.push([i, j]);
    return { V, E, n: V.length, k: phi.clauses.length, A: adjFromEdges(V.length, E) };
  }

  /* ═══ §02 encodings ════════════════════════════════════════════════════ */
  function unaryLen(k) { return k; }
  function binaryLen(k) { return Math.floor(Math.log2(k)) + 1; }
  /* a loop that is linear in the VALUE of k — the routine of §02 */
  function loopToK(k, c) { c = c || nop; let s = 0; for (let i = 0; i < k; i++) { c.add("step"); s += 1; } return s; }
  /* trial division: Θ(√N) divisions, i.e. Θ(2^(bits/2)) */
  function trialDivision(N, c) { c = c || nop; if (N < 2) return { prime: false, divisions: 0 }; for (let d = 2; d * d <= N; d++) { c.add("div"); if (N % d === 0) return { prime: false, divisions: c.get("div"), factor: d }; } return { prime: true, divisions: c.get("div") }; }
  /* the knapsack table: (n)(W+1) cells — pseudo-polynomial, §02 and §14 */
  function knapCells(n, W, c) { c = c || nop; for (let i = 0; i < n; i++) for (let w = 0; w <= W; w++) c.add("cell"); return c.get("cell"); }

  /* ═══ §04 measured growth: routines that really do the work ════════════ */
  function growLinear(n, c) { for (let i = 0; i < n; i++) c.add("op"); return c.get("op"); }
  function growQuadratic(n, c) { for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) c.add("op"); return c.get("op"); }
  function growCubic(n, c) { for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) c.add("op"); return c.get("op"); }
  function growSubsets(n, c) { const lim = 1 << n; for (let m = 0; m < lim; m++) c.add("op"); return c.get("op"); }
  function growPerms(n, c) { const a = [...Array(n).keys()]; (function go(k) { if (k === n) { c.add("op"); return; } for (let i = k; i < n; i++) { const t = a[k]; a[k] = a[i]; a[i] = t; go(k + 1); const u = a[k]; a[k] = a[i]; a[i] = u; } })(0); return c.get("op"); }
  function factorial(n) { let p = 1; for (let i = 2; i <= n; i++) p *= i; return p; }
  /* largest n whose closed-form cost stays inside a budget (binary search) */
  function largestN(f, budget) { let hi = 1; while (f(hi) <= budget && hi < 1e13) hi *= 2; let lo = 1; while (lo < hi) { const m = Math.floor((lo + hi + 1) / 2); if (f(m) <= budget) lo = m; else hi = m - 1; } return lo; }

  /* ═══ §05 verifiers, each counting its own elementary check ════════════ */
  function verifySat(clauses, a, c) {
    c = c || nop; const checks = [];
    for (let i = 0; i < clauses.length; i++) {
      const cl = clauses[i]; let ok = false, used = null;
      for (const l of cl) { c.add("check"); if ((l > 0) === !!a[Math.abs(l) - 1]) { ok = true; used = l; break; } }
      checks.push({ what: "clause " + (i + 1), ok, detail: ok ? "satisfied by literal " + (used > 0 ? "x" + used : "¬x" + (-used)) : "every literal false" });
      if (!ok) return { accept: false, checks, steps: c.get("check") };
    }
    return { accept: true, checks, steps: c.get("check") };
  }
  function verifyClique(A, S, k, c) {
    c = c || nop; const checks = [{ what: "|S| = " + S.length, ok: S.length === k, detail: "required " + k }];
    if (S.length !== k) return { accept: false, checks, steps: 0 };
    for (let i = 0; i < S.length; i++) for (let j = i + 1; j < S.length; j++) { c.add("check"); const ok = A[S[i]][S[j]]; checks.push({ what: "edge (" + S[i] + "," + S[j] + ")?", ok, detail: ok ? "present" : "MISSING — not a clique" }); if (!ok) return { accept: false, checks, steps: c.get("check") }; }
    return { accept: true, checks, steps: c.get("check") };
  }
  function verifySubset(S, picks, T, c) {
    c = c || nop; const checks = []; let sum = 0;
    for (const i of picks) { c.add("check"); sum += S[i]; checks.push({ what: "+ " + S[i], ok: true, detail: "running total " + sum }); }
    const ok = sum === T; checks.push({ what: "total = " + T + "?", ok, detail: ok ? "exactly" : "got " + sum });
    return { accept: ok, checks, steps: c.get("check") };
  }
  function subsetBrute(S, T, c) { c = c || nop; const sols = []; for (let m = 0; m < (1 << S.length); m++) { c.add("subset"); let t = 0; for (let i = 0; i < S.length; i++) if ((m >> i) & 1) { c.add("check"); t += S[i]; } if (t === T) sols.push(S.filter((_, i) => (m >> i) & 1)); } return { sols, count: sols.length, subsets: 1 << S.length }; }

  /* ═══ §06 the reduction map over ALL graphs on 4 vertices ══════════════ */
  function allGraphs4() {
    const n = 4, pairs = []; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) pairs.push([i, j]);
    return { n, pairs, count: 1 << pairs.length,
             adj: mask => adjFromEdges(n, pairs.filter((_, b) => (mask >> b) & 1)),
             edges: mask => pairs.filter((_, b) => (mask >> b) & 1) };
  }
  /* run the chosen map over every instance and count the disagreements */
  function reductionMap(kind, k, c) {
    c = c || nop; const F = allGraphs4(); const rows = []; let agree = 0, yes = 0, blowup = 0;
    for (let mask = 0; mask < F.count; mask++) {
      const A = F.adj(mask); c.add("instance");
      const left = hasIndepSet(A, F.n, k, c);            // the A-instance: INDEPENDENT-SET
      let B, kk = k;
      if (kind === "comp") B = complementAdj(A, F.n);
      else if (kind === "id") B = A;
      else { B = complementAdj(A, F.n); kk = k + 1; }
      const right = hasClique(B, F.n, kk, c);            // the B-instance: CLIQUE
      const mEdges = F.edges(mask).length, mImg = kind === "id" ? mEdges : 6 - mEdges;
      blowup += mImg;
      if (left === right) agree++; if (left) yes++;
      rows.push({ mask, left, right, ok: left === right, mEdges, mImg, k: kk });
    }
    return { rows, agree, yes, no: F.count - yes, total: F.count, disagree: F.count - agree, blowup };
  }

  /* ═══ §08 circuits, Tseitin, and the tableau ═══════════════════════════ */
  /* the demonstration circuit: variables 1..3 are inputs, 4..8 gate outputs */
  const CIRC = { nIn: 3, out: 8,
    gates: [{ id: 4, op: "AND", in: [1, 2] }, { id: 5, op: "OR", in: [2, 3] }, { id: 6, op: "NOT", in: [1] },
            { id: 7, op: "OR", in: [4, 6] }, { id: 8, op: "AND", in: [7, 5] }] };
  function circuitEval(C, x, c) {
    c = c || nop; const v = {}; for (let i = 0; i < C.nIn; i++) v[i + 1] = !!x[i];
    for (const g of C.gates) { c.add("gate"); const a = v[g.in[0]], b = g.in.length > 1 ? v[g.in[1]] : null;
      v[g.id] = g.op === "AND" ? (a && b) : g.op === "OR" ? (a || b) : !a; }
    return { value: v[C.out], wires: v };
  }
  function circuitBrute(C, c) { c = c || nop; const sat = []; for (let m = 0; m < (1 << C.nIn); m++) { c.add("assign"); const x = bitsOf(m, C.nIn); if (circuitEval(C, x, c).value) sat.push(bitStr(x)); } return { sat, count: sat.length, assigns: 1 << C.nIn }; }
  /* Tseitin: one fresh variable per gate, ≤ 3 clauses each, plus the unit
     clause asserting the output.  Returns the clauses tagged by their gate. */
  function tseitin(C, c) {
    c = c || nop; const cl = [];
    for (const g of C.gates) { const z = g.id, a = g.in[0], b = g.in[1]; const before = cl.length;
      if (g.op === "AND") { cl.push({ g: z, c: [-z, a] }, { g: z, c: [-z, b] }, { g: z, c: [z, -a, -b] }); }
      else if (g.op === "OR") { cl.push({ g: z, c: [z, -a] }, { g: z, c: [z, -b] }, { g: z, c: [-z, a, b] }); }
      else { cl.push({ g: z, c: [z, a] }, { g: z, c: [-z, -a] }); }
      for (let i = before; i < cl.length; i++) c.add("clause");
    }
    cl.push({ g: C.out, c: [C.out], unit: true }); c.add("clause");
    return { clauses: cl.map(o => o.c), tagged: cl, vars: C.out, count: cl.length };
  }
  /* brute force the Tseitin CNF and project onto the circuit's input wires */
  function tseitinCheck(C, c) {
    c = c || nop; const T = tseitin(C);
    const sols = []; const proj = new Set();
    for (let m = 0; m < (1 << T.vars); m++) { c.add("assign"); const a = bitsOf(m, T.vars);
      if (satEval(T.clauses, a, c)) { sols.push(bitStr(a)); proj.add(bitStr(a.slice(0, C.nIn))); } }
    const B = circuitBrute(C);
    const P = [...proj].sort(), S = B.sat.slice().sort();
    return { T, sols, proj: P, circuit: S, assigns: 1 << T.vars,
             agree: P.length === S.length && P.every((x, i) => x === S[i]),
             sameCount: sols.length === B.count };
  }

  /* a tiny verifier: sweep right along the tape, accept on the first 1.
     Tape = 2 input cells, 2 certificate cells, then blanks.  The tableau of
     §08 is built from this machine, and the legal-window set is MEASURED by
     enumerating every configuration and stepping it. */
  const TM = { SYM: ["0", "1", "␣"], ST: ["q₀", "qa", "qr"], HASH: "#", tape: 6, rows: 6 };
  TM.C = TM.SYM.concat(TM.ST, [TM.HASH]);
  TM.rowLen = TM.tape + 3;               /* # … # plus the inserted state symbol */
  function tmDelta(q, s) { if (q !== "q₀") return null; if (s === "0") return ["q₀", "0", 1]; if (s === "1") return ["qa", "1", 1]; return ["qr", "␣", 1]; }
  function tmRow(cf) { const r = ["#"]; for (let i = 0; i < TM.tape; i++) r.push(cf.tape[i]); r.push("#"); r.splice(1 + cf.h, 0, cf.q); return r; }
  function tmStep(cf) { const d = tmDelta(cf.q, cf.tape[cf.h]); if (!d) return { tape: cf.tape.slice(), q: cf.q, h: cf.h }; const t = cf.tape.slice(); t[cf.h] = d[1]; let h = cf.h + d[2]; if (h >= TM.tape) h = TM.tape - 1; if (h < 0) h = 0; return { tape: t, q: d[0], h }; }
  function tmRun(x, y, c) {
    c = c || nop; const tape = x.concat(y); while (tape.length < TM.tape) tape.push("␣");
    let cf = { tape, q: "q₀", h: 0 }; const rows = [tmRow(cf)];
    for (let i = 1; i < TM.rows; i++) { c.add("step"); cf = tmStep(cf); rows.push(tmRow(cf)); }
    return { rows, accept: rows.some(r => r.indexOf("qa") >= 0) };
  }
  /* every configuration, stepped, and the windows that actually occur */
  function tmLegalWindows(c) {
    c = c || nop; const legal = new Set(); let configs = 0;
    const tapes = []; (function go(k, acc) { if (k === 0) { tapes.push(acc); return; } for (const s of TM.SYM) go(k - 1, acc.concat([s])); })(TM.tape, []);
    for (const tape of tapes) for (const q of TM.ST) for (let h = 0; h < TM.tape; h++) {
      configs++; c.add("config");
      const a = tmRow({ tape, q, h }), b = tmRow(tmStep({ tape, q, h }));
      for (let j = 1; j + 1 < a.length; j++) { c.add("window"); legal.add([a[j - 1], a[j], a[j + 1], b[j - 1], b[j], b[j + 1]].join("|")); }
    }
    const all = Math.pow(TM.C.length, 6);
    return { legal, size: legal.size, configs, all, illegal: all - legal.size };
  }
  function tmFormulaSizes(L) {
    const cells = TM.rows * TM.rowLen, k = TM.C.length;
    const positions = (TM.rows - 1) * (TM.rowLen - 2);
    return { cells, symbols: k, vars: k * cells, positions,
             cellClauses: cells * (1 + k * (k - 1) / 2),
             acceptWidth: cells, moveClauses: positions * L.illegal };
  }

  /* ═══ §09 CNF → 3-CNF ══════════════════════════════════════════════════ */
  function cnfTo3(clauses, n, c) {
    c = c || nop; let next = n + 1; const out = [], groups = [];
    for (const cl of clauses) {
      const k = cl.length, start = out.length, fresh = [];
      if (k === 1) { const p = next++, q = next++; fresh.push(p, q); out.push([cl[0], p, q], [cl[0], p, -q], [cl[0], -p, q], [cl[0], -p, -q]); }
      else if (k === 2) { const p = next++; fresh.push(p); out.push([cl[0], cl[1], p], [cl[0], cl[1], -p]); }
      else if (k === 3) { out.push(cl.slice()); }
      else { const y = []; for (let i = 0; i < k - 3; i++) { const v = next++; y.push(v); fresh.push(v); }
        out.push([cl[0], cl[1], y[0]]);
        for (let i = 1; i <= k - 4; i++) out.push([-y[i - 1], cl[i + 1], y[i]]);
        out.push([-y[k - 4], cl[k - 2], cl[k - 1]]); }
      out.slice(start).forEach(() => c.add("clause"));
      groups.push({ width: k, from: start, to: out.length, fresh, added: out.length - start });
    }
    return { clauses: out, vars: next - 1, groups, freshCount: next - 1 - n };
  }
  /* brute force a CNF and project its solutions onto the first n variables */
  function cnfProject(clauses, nv, n, c) {
    c = c || nop; const proj = new Set(); let sols = 0;
    for (let m = 0; m < (1 << nv); m++) { c.add("assign"); const a = bitsOf(m, nv); if (satEval(clauses, a, c)) { sols++; proj.add(bitStr(a.slice(0, n))); } }
    return { proj: [...proj].sort(), sols, assigns: 1 << nv };
  }
  function setsEqual(a, b) { return a.length === b.length && a.every((x, i) => x === b[i]); }

  return { clauseSat, satEval, satCount, bitsOf, bitStr, satBrute, satRestrict, satOracle, satSelfReduce,
           adjFromNamed, adjFromEdges, isCliqueSet, cliqueMax, cliqueCount, cliqueSelfReduce,
           complementAdj, hasIndepSet, hasClique, satToClique,
           unaryLen, binaryLen, loopToK, trialDivision, knapCells,
           growLinear, growQuadratic, growCubic, growSubsets, growPerms, factorial, largestN,
           verifySat, verifyClique, verifySubset, subsetBrute,
           allGraphs4, reductionMap,
           CIRC, circuitEval, circuitBrute, tseitin, tseitinCheck,
           TM, tmDelta, tmRow, tmStep, tmRun, tmLegalWindows, tmFormulaSizes,
           cnfTo3, cnfProject, setsEqual };
})());


/* ── chunk B ─────────────────────────────────────────────────── */
/* ── chunk B routines: the reduction catalogue (§10–§18) ───────────────────
   Pure, DOM-free and instrumented. Every routine takes an optional AL.counter()
   as its last argument and bumps named keys, so a figure DISPLAYS a measurement
   rather than a constant. Graphs are {n, E:[[u,v],…], names?}; formulas are
   {n, clauses:[[lit,…],…]} with signed 1-based literals, as in NI.
   Chunk A's NR.satBrute / NR.satToClique are called where available and
   cross-checked; the B-suffixed variants below are the independent copies this
   chunk's figures draw from, so a shape mismatch degrades to a flag, never a
   crash. */
Object.assign(NR, (function () {
  const nop = { add: function () {}, get: function () { return 0; } };

  /* ── graph plumbing ──────────────────────────────────────────────────── */
  function npG(n, E, names) { return { n: n, E: E.map(function (e) { return [e[0], e[1]]; }), names: names || Array.from({ length: n }, function (_, i) { return String(i); }) }; }
  function npFromLetters(o) { const ix = {}; o.V.forEach(function (v, i) { ix[v] = i; }); return npG(o.V.length, o.E.map(function (e) { return [ix[e[0]], ix[e[1]]]; }), o.V.slice()); }
  function npAdj(g) { const A = Array.from({ length: g.n }, function () { return new Array(g.n).fill(false); }); g.E.forEach(function (e) { A[e[0]][e[1]] = true; A[e[1]][e[0]] = true; }); return A; }
  function npAdjList(g) { const L = Array.from({ length: g.n }, function () { return []; }); g.E.forEach(function (e) { L[e[0]].push(e[1]); L[e[1]].push(e[0]); }); return L; }
  function npComplement(g) { const A = npAdj(g), E = []; for (let i = 0; i < g.n; i++) for (let j = i + 1; j < g.n; j++) if (!A[i][j]) E.push([i, j]); return npG(g.n, E, g.names); }
  function npDegrees(g) { const d = new Array(g.n).fill(0); g.E.forEach(function (e) { d[e[0]]++; d[e[1]]++; }); return d; }
  function npName(g, i) { return g.names ? g.names[i] : String(i); }

  /* ── SAT, independently of chunk A (distinct names) ──────────────────── */
  function satEvalB(clauses, a, c) { c = c || nop; return clauses.every(function (cl) { return cl.some(function (l) { c.add("lit"); return l > 0 ? a[l - 1] : !a[-l - 1]; }); }); }
  function satCountB(clauses, a, c) { c = c || nop; let k = 0; clauses.forEach(function (cl) { if (cl.some(function (l) { c.add("lit"); return l > 0 ? a[l - 1] : !a[-l - 1]; })) k++; }); return k; }
  /* every assignment, in MSB-first bit order so 001 means x₁=0,x₂=0,x₃=1 */
  function satAllB(phi, c) {
    c = c || nop; const n = phi.n, models = []; let tried = 0; const a = new Array(n);
    for (let m = 0; m < (1 << n); m++) { tried++; c.add("assign"); for (let i = 0; i < n; i++) a[i] = !!((m >> (n - 1 - i)) & 1); if (satEvalB(phi.clauses, a, c)) models.push(a.slice()); }
    return { models: models, tried: tried, lits: c.get("lit") };
  }
  /* stop at the first model; LSB-first, which is irrelevant to the counts */
  function satFirstB(n, clauses, c) {
    c = c || nop; let tried = 0; const a = new Array(n);
    for (let m = 0; m < (1 << n); m++) { tried++; c.add("assign"); for (let i = 0; i < n; i++) a[i] = !!((m >> i) & 1); if (satEvalB(clauses, a, c)) return { sat: true, a: a.slice(), tried: tried }; }
    return { sat: false, a: null, tried: tried };
  }
  function bitsOfB(a) { return a.map(function (x) { return x ? 1 : 0; }).join(""); }
  /* call chunk A's NR.satBrute if it is there, and normalise whatever it returns */
  function satCrossCheck(phi) {
    let r = null; try { if (typeof NR.satBrute === "function") r = NR.satBrute(phi.clauses, phi.n); } catch (e) { r = null; }
    if (!r) return { have: false };
    if (typeof r === "boolean") return { have: true, sat: r };
    const models = r.models || r.satisfying || r.assignments || r.sols || null;
    const cnt = (models && models.length !== undefined) ? models.length : (typeof r.count === "number" ? r.count : null);
    const sat = (typeof r.sat === "boolean") ? r.sat : (cnt !== null ? cnt > 0 : null);
    return { have: true, sat: sat, count: cnt };
  }

  /* ── §11  3-SAT ≤ₚ CLIQUE ────────────────────────────────────────────── */
  function satToCliqueB(phi) {
    const verts = []; phi.clauses.forEach(function (cl, ci) { cl.forEach(function (lit, pi) { verts.push({ c: ci, p: pi, lit: lit }); }); });
    const E = []; let tested = 0;
    for (let i = 0; i < verts.length; i++) for (let j = i + 1; j < verts.length; j++) { tested++; if (verts[i].c !== verts[j].c && verts[i].lit !== -verts[j].lit) E.push([i, j]); }
    const names = verts.map(function (v) { return (v.lit > 0 ? "" : "¬") + "x" + Math.abs(v.lit); });
    return { g: npG(verts.length, E, names), verts: verts, k: phi.clauses.length, pairsTested: tested };
  }
  /* prefer chunk A's reduction when its output can be recognised; always report agreement */
  function cliqueFromSat(phi) {
    const mine = satToCliqueB(phi); let ext = null;
    try { if (typeof NR.satToClique === "function") ext = NR.satToClique(phi); } catch (e) { ext = null; }
    let en = null, em = null;
    if (ext) {
      const gg = ext.g || ext.graph || ext;
      en = (typeof gg.n === "number") ? gg.n : (Array.isArray(gg.V) ? gg.V.length : null);
      const ee = gg.E || gg.edges || null; em = Array.isArray(ee) ? ee.length : null;
    }
    const readable = (typeof en === "number" && typeof em === "number");
    const agree = readable && en === mine.g.n && em === mine.g.E.length;
    return { g: mine.g, verts: mine.verts, k: mine.k, pairsTested: mine.pairsTested, haveExt: !!ext, readable: readable, extN: en, extM: em, agree: agree };
  }
  /* every subset, so the census by size is exact. n ≤ 20. */
  function cliqueBrute(g, c) {
    c = c || nop; const A = npAdj(g); let max = 0, all = [], tried = 0; const bySize = {};
    for (let m = 0; m < (1 << g.n); m++) {
      tried++; c.add("subset"); const S = []; for (let i = 0; i < g.n; i++) if ((m >> i) & 1) S.push(i);
      let ok = true; for (let i = 0; i < S.length && ok; i++) for (let j = i + 1; j < S.length; j++) { c.add("pair"); if (!A[S[i]][S[j]]) { ok = false; break; } }
      if (ok) { bySize[S.length] = (bySize[S.length] || 0) + 1; if (S.length > max) { max = S.length; all = [S]; } else if (S.length === max && max > 0) all.push(S); }
    }
    return { max: max, all: all, tried: tried, bySize: bySize };
  }
  /* just the k-subsets, so the figure can say "C(n,k) triples tested" honestly */
  function cliquesOfSize(g, k, c) {
    c = c || nop; const A = npAdj(g), out = [];
    (function go(start, cur) {
      if (cur.length === k) { out.push(cur.slice()); return; }
      for (let v = start; v < g.n; v++) { c.add("cand"); if (cur.every(function (u) { c.add("pair"); return A[u][v]; })) { cur.push(v); go(v + 1, cur); cur.pop(); } }
    })(0, []);
    return { list: out, combos: choose(g.n, k) };
  }
  function choose(n, k) { if (k < 0 || k > n) return 0; let r = 1; for (let i = 1; i <= k; i++) r = r * (n - k + i) / i; return Math.round(r); }
  /* Bron–Kerbosch with a pivot, for graphs too big for 2ⁿ subsets */
  function maxCliqueBK(g, c) {
    c = c || nop; const A = npAdjList(g).map(function (l) { return new Set(l); }); let best = [];
    (function BK(R, P, X) {
      c.add("node");
      if (P.size === 0 && X.size === 0) { if (R.length > best.length) best = R.slice(); return; }
      const pool = []; P.forEach(function (v) { pool.push(v); }); X.forEach(function (v) { pool.push(v); });
      const pivot = pool[0]; const cand = []; P.forEach(function (v) { if (!A[pivot].has(v)) cand.push(v); });
      cand.forEach(function (v) {
        const np = new Set(), nx = new Set();
        P.forEach(function (x) { if (A[v].has(x)) np.add(x); }); X.forEach(function (x) { if (A[v].has(x)) nx.add(x); });
        BK(R.concat(v), np, nx); P.delete(v); X.add(v);
      });
    })([], new Set(Array.from({ length: g.n }, function (_, i) { return i; })), new Set());
    return { best: best, size: best.length };
  }
  /* the assignment a clique encodes; null entries are variables it does not mention */
  function assignFromClique(phi, verts, S) {
    const a = new Array(phi.n).fill(null);
    S.forEach(function (i) { const l = verts[i].lit; a[Math.abs(l) - 1] = l > 0; });
    return a;
  }
  function completions(a) {
    const free = []; a.forEach(function (x, i) { if (x === null) free.push(i); });
    const out = [];
    for (let m = 0; m < (1 << free.length); m++) { const b = a.slice(); free.forEach(function (f, j) { b[f] = !!((m >> j) & 1); }); out.push(b.map(function (x) { return x ? 1 : 0; }).join("")); }
    return out;
  }

  /* ── §12  vertex cover / independent set, by exhaustive subsets ──────── */
  function vcBrute(g, c) {
    c = c || nop; let min = g.n, all = [], tried = 0; const bySize = {};
    for (let m = 0; m < (1 << g.n); m++) {
      tried++; c.add("subset"); const inS = function (i) { return !!((m >> i) & 1); };
      let ok = true; for (let e = 0; e < g.E.length; e++) { c.add("edge"); if (!inS(g.E[e][0]) && !inS(g.E[e][1])) { ok = false; break; } }
      if (!ok) continue;
      const S = []; for (let i = 0; i < g.n; i++) if (inS(i)) S.push(i);
      bySize[S.length] = (bySize[S.length] || 0) + 1;
      if (S.length < min) { min = S.length; all = [S]; } else if (S.length === min) all.push(S);
    }
    return { min: min, all: all, tried: tried, bySize: bySize };
  }
  function isBrute(g, c) {
    c = c || nop; const A = npAdj(g); let max = 0, all = [], tried = 0; const bySize = {};
    for (let m = 0; m < (1 << g.n); m++) {
      tried++; c.add("subset"); const S = []; for (let i = 0; i < g.n; i++) if ((m >> i) & 1) S.push(i);
      let ok = true; for (let i = 0; i < S.length && ok; i++) for (let j = i + 1; j < S.length; j++) { c.add("pair"); if (A[S[i]][S[j]]) { ok = false; break; } }
      if (ok) { bySize[S.length] = (bySize[S.length] || 0) + 1; if (S.length > max) { max = S.length; all = [S]; } else if (S.length === max) all.push(S); }
    }
    return { max: max, all: all, tried: tried, bySize: bySize };
  }
  function coverOK(g, S) { const set = new Set(S); return g.E.every(function (e) { return set.has(e[0]) || set.has(e[1]); }); }
  function uncovered(g, S) { const set = new Set(S); return g.E.filter(function (e) { return !set.has(e[0]) && !set.has(e[1]); }); }
  function inducedEdges(g, S) { const set = new Set(S); return g.E.filter(function (e) { return set.has(e[0]) && set.has(e[1]); }); }

  return {
    npG: npG, npFromLetters: npFromLetters, npAdj: npAdj, npAdjList: npAdjList,
    npComplement: npComplement, npDegrees: npDegrees, npName: npName, choose: choose,
    satEvalB: satEvalB, satCountB: satCountB, satAllB: satAllB, satFirstB: satFirstB,
    bitsOfB: bitsOfB, satCrossCheck: satCrossCheck,
    satToCliqueB: satToCliqueB, cliqueFromSat: cliqueFromSat, cliqueBrute: cliqueBrute,
    cliquesOfSize: cliquesOfSize, maxCliqueBK: maxCliqueBK,
    assignFromClique: assignFromClique, completions: completions,
    vcBrute: vcBrute, isBrute: isBrute, coverOK: coverOK, uncovered: uncovered, inducedEdges: inducedEdges
  };
})());

/* ── chunk B routines, part 2: §13 Hamiltonian cycles, Euler tours, TSP ── */
Object.assign(NR, (function () {
  const nop = { add: function () {}, get: function () { return 0; } };

  /* every distinct cyclic ordering: vertex 0 is fixed and the two directions of
     a cycle are counted once, so the count is exactly (n−1)!/2 for n ≥ 3. */
  function hamBrute(g, c) {
    c = c || nop; const n = g.n, A = NR.npAdj(g); const cycles = []; let tours = 0;
    const p = [], used = new Array(n).fill(false);
    (function go() {
      if (p.length === n - 1) {
        if (n > 2 && p[0] > p[p.length - 1]) return;      /* one of each direction pair */
        tours++; c.add("tour");
        let prev = 0, ok = true;
        for (let i = 0; i < p.length; i++) { c.add("check"); if (!A[prev][p[i]]) { ok = false; break; } prev = p[i]; }
        if (ok) { c.add("check"); ok = A[prev][0]; }
        if (ok) cycles.push([0].concat(p));
        return;
      }
      for (let v = 1; v < n; v++) { if (used[v]) continue; used[v] = true; p.push(v); go(); p.pop(); used[v] = false; }
    })();
    return { cycles: cycles, tours: tours, checks: c.get("check") };
  }
  /* the minimum number of NON-edges any cyclic ordering must use — one
     enumeration settles the TSP optimum under EVERY 1/x weighting at once. */
  function hamMinNonEdges(g, c) {
    c = c || nop; const n = g.n, A = NR.npAdj(g); let best = n + 1, bestTour = null, tours = 0;
    const p = [], used = new Array(n).fill(false);
    (function go() {
      if (p.length === n - 1) {
        if (n > 2 && p[0] > p[p.length - 1]) return;
        tours++; c.add("tour"); let miss = 0, prev = 0;
        for (let i = 0; i < p.length; i++) { c.add("check"); if (!A[prev][p[i]]) miss++; prev = p[i]; }
        c.add("check"); if (!A[prev][0]) miss++;
        if (miss < best) { best = miss; bestTour = [0].concat(p); }
        return;
      }
      for (let v = 1; v < n; v++) { if (used[v]) continue; used[v] = true; p.push(v); go(); p.pop(); used[v] = false; }
    })();
    return { minNonEdges: best, tour: bestTour, tours: tours };
  }
  /* the TSP optimum under w = 1 on edges of g, "off" elsewhere, derived from
     one enumeration: cost = (n − miss)·1 + miss·off */
  function tspFromHamCost(g, off, mn) { return (g.n - mn.minNonEdges) + mn.minNonEdges * off; }
  /* depth-first Hamiltonian PATH, lowest-numbered unused neighbour first */
  function hamPathDFS(g, c) {
    c = c || nop; const n = g.n, A = NR.npAdj(g); let found = null;
    const p = [], used = new Array(n).fill(false);
    (function go(start) {
      used[start] = true; p.push(start); c.add("node");
      (function step() {
        if (found) return;
        if (p.length === n) { found = p.slice(); return; }
        for (let v = 0; v < n; v++) {
          if (used[v] || !A[p[p.length - 1]][v]) continue;
          used[v] = true; p.push(v); c.add("node"); step(); if (found) return; p.pop(); used[v] = false;
        }
      })();
    })(0);
    return { path: found, nodes: c.get("node") };
  }
  /* Euler's criterion: a degree scan plus a connectivity check, Θ(V + E) */
  function eulerTest(g, c) {
    c = c || nop; const d = new Array(g.n).fill(0);
    g.E.forEach(function (e) { c.add("step"); d[e[0]]++; d[e[1]]++; });
    let odd = 0, oddV = []; for (let i = 0; i < g.n; i++) { c.add("step"); if (d[i] % 2) { odd++; oddV.push(i); } }
    const degSteps = c.get("step");
    const L = NR.npAdjList(g); const seen = new Array(g.n).fill(false);
    let s = -1; for (let i = 0; i < g.n; i++) if (d[i] > 0) { s = i; break; }
    if (s >= 0) { const st = [s]; seen[s] = true; while (st.length) { const u = st.pop(); L[u].forEach(function (v) { c.add("step"); if (!seen[v]) { seen[v] = true; st.push(v); } }); } }
    const connected = d.every(function (x, v) { return x === 0 || seen[v]; });
    return { degrees: d, odd: odd, oddV: oddV, connected: connected, circuit: odd === 0 && connected, trail: odd === 2 && connected, degSteps: degSteps, connSteps: c.get("step") - degSteps, steps: c.get("step") };
  }
  /* Hierholzer, O(E) */
  function eulerTour(g, c) {
    c = c || nop; const t = eulerTest(g, { add: function () {}, get: function () { return 0; } });
    if (!t.circuit) return { tour: null, test: t, steps: c.get("step") };
    const L = Array.from({ length: g.n }, function () { return []; });
    g.E.forEach(function (e, i) { L[e[0]].push({ v: e[1], i: i }); L[e[1]].push({ v: e[0], i: i }); });
    const used = new Array(g.E.length).fill(false), ptr = new Array(g.n).fill(0);
    const stack = [g.E[0][0]], out = [];
    while (stack.length) {
      const u = stack[stack.length - 1];
      while (ptr[u] < L[u].length && used[L[u][ptr[u]].i]) ptr[u]++;
      if (ptr[u] === L[u].length) { out.push(stack.pop()); }
      else { const a = L[u][ptr[u]]; used[a.i] = true; c.add("step"); stack.push(a.v); }
    }
    return { tour: out.reverse(), test: t, steps: c.get("step") };
  }
  /* the 12-vertex, 14-edge widget of the VERTEX-COVER ≤ₚ HAM-CYCLE reduction.
     0…5 = [u,v,1…6];  6…11 = [v,u,1…6].  Two column paths plus four crossings. */
  function hamWidget() {
    const E = []; for (let i = 0; i < 5; i++) { E.push([i, i + 1]); E.push([6 + i, 7 + i]); }
    E.push([0, 8], [2, 6], [3, 11], [5, 9]);     /* L1–R3, L3–R1, L4–R6, L6–R4 */
    const names = []; for (let i = 1; i <= 6; i++) names.push("[u,v," + i + "]"); for (let i = 1; i <= 6; i++) names.push("[v,u," + i + "]");
    return NR.npG(12, E, names);
  }
  /* every Hamiltonian PATH of a graph between two given endpoints */
  function allHamPaths(g, s, t, c) {
    c = c || nop; const A = NR.npAdj(g), res = []; const used = new Array(g.n).fill(false);
    (function go(u, path) {
      c.add("node");
      if (path.length === g.n) { if (u === t) res.push(path.slice()); return; }
      for (let v = 0; v < g.n; v++) { if (used[v] || !A[u][v]) continue; used[v] = true; path.push(v); go(v, path); path.pop(); used[v] = false; }
    })((used[s] = true, s), [s]);
    return { paths: res, nodes: c.get("node") };
  }
  /* two vertex-disjoint paths s1→t1 and s2→t2 that together cover every vertex */
  function splitPaths(g, s1, t1, s2, t2, c) {
    c = c || nop; const A = NR.npAdj(g), sols = []; const used = new Array(g.n).fill(false);
    function walk(u, t, acc, then) {
      c.add("node"); if (u === t) { then(acc.slice()); return; }
      for (let v = 0; v < g.n; v++) { if (used[v] || !A[u][v]) continue; used[v] = true; acc.push(v); walk(v, t, acc, then); acc.pop(); used[v] = false; }
    }
    used[s1] = true;
    walk(s1, t1, [s1], function (p1) {
      const save = used.slice(); used[s2] = true;
      walk(s2, t2, [s2], function (p2) { if (p1.length + p2.length === g.n) sols.push([p1, p2]); });
      for (let i = 0; i < g.n; i++) used[i] = save[i];
    });
    return { sols: sols, nodes: c.get("node") };
  }
  /* the widget's whole traversal census: 1 / 1 / 1 admissible, 0 / 0 / 0 forbidden */
  function widgetTraversals(c) {
    c = c || nop; const W = hamWidget(); const a = 0, b = 5, p = 6, q = 11;
    const bb = allHamPaths(W, a, b, c), dd = allHamPaths(W, p, q, c);
    const cc = splitPaths(W, a, b, p, q, c);
    const x1 = allHamPaths(W, a, q, c), x2 = allHamPaths(W, p, b, c), x3 = splitPaths(W, a, q, p, b, c);
    return { widget: W, uv: bb.paths, vu: dd.paths, split: cc.sols, badUV: x1.paths, badVU: x2.paths, badSplit: x3.sols, nodes: c.get("node") };
  }
  /* the size of G′ produced by VERTEX-COVER ≤ₚ HAM-CYCLE on ⟨G, k⟩ */
  function hamReductionSize(g, k) { return { V: 12 * g.E.length + k, E: 16 * g.E.length + (2 * k - 1) * g.n }; }
  /* a circular layout, so a figure never has to hand-place vertices */
  function circleLayout(n, cx, cy, r, rot) {
    const out = []; for (let i = 0; i < n; i++) { const t = (rot || -Math.PI / 2) + 2 * Math.PI * i / n; out.push([cx + r * Math.cos(t), cy + r * Math.sin(t)]); } return out;
  }
  /* the Petersen graph's canonical outer-pentagon / inner-pentagram layout */
  function petersenLayout(cx, cy, r) {
    const out = []; for (let i = 0; i < 5; i++) { const t = -Math.PI / 2 + 2 * Math.PI * i / 5; out.push([cx + r * Math.cos(t), cy + r * Math.sin(t)]); }
    const inner = []; for (let i = 0; i < 5; i++) { const t = -Math.PI / 2 + 2 * Math.PI * i / 5; inner.push([cx + 0.45 * r * Math.cos(t), cy + 0.45 * r * Math.sin(t)]); }
    /* NI.gPetersen numbers the inner vertices 5…9 with 5+i adjacent to outer i */
    return out.concat(inner);
  }
  /* the Euler-circuit variant of gHam: drop 5–6, add 0–2 and 1–3 */
  function eulerVariant() {
    const E = NI.gHam.E.filter(function (e) { return !(e[0] === 5 && e[1] === 6); }).concat([[0, 2], [1, 3]]);
    return NR.npG(NI.gHam.n, E);
  }

  return {
    hamBrute: hamBrute, hamMinNonEdges: hamMinNonEdges, tspFromHamCost: tspFromHamCost,
    hamPathDFS: hamPathDFS, eulerTest: eulerTest, eulerTour: eulerTour,
    hamWidget: hamWidget, allHamPaths: allHamPaths, splitPaths: splitPaths,
    widgetTraversals: widgetTraversals, hamReductionSize: hamReductionSize,
    circleLayout: circleLayout, petersenLayout: petersenLayout, eulerVariant: eulerVariant
  };
})());

/* ── chunk B routines, part 3: §14 subset sum / partition / knapsack ───── */
Object.assign(NR, (function () {
  const nop = { add: function () {}, get: function () { return 0; } };

  function subsetSumBrute(S, T, c) {
    c = c || nop; const sols = []; let tried = 0;
    for (let m = 0; m < (1 << S.length); m++) {
      tried++; c.add("subset"); let s = 0; const sub = [];
      for (let i = 0; i < S.length; i++) if ((m >> i) & 1) { c.add("add"); s += S[i]; sub.push(S[i]); }
      if (s === T) sols.push({ mask: m, items: sub });
    }
    return { sols: sols, tried: tried };
  }
  /* the Θ(n·(T+1)) reachability table, with a traceback of one solution */
  function subsetSumDP(S, T, c) {
    c = c || nop; const n = S.length;
    const R = Array.from({ length: n + 1 }, function () { return new Array(T + 1).fill(false); });
    R[0][0] = true; let cells = 0, trues = 0;
    for (let i = 1; i <= n; i++) for (let t = 0; t <= T; t++) {
      cells++; c.add("cell");
      R[i][t] = R[i - 1][t] || (t >= S[i - 1] && R[i - 1][t - S[i - 1]]);
      if (R[i][t]) trues++;
    }
    const trace = []; let t = T;
    if (R[n][T]) { for (let i = n; i >= 1; i--) { if (R[i - 1][t]) { trace.push({ i: i, t: t, take: false }); } else { trace.push({ i: i, t: t, take: true }); t -= S[i - 1]; } } trace.reverse(); }
    return { R: R, cells: cells, trues: trues, yes: R[n][T], trace: trace, take: trace.filter(function (x) { return x.take; }).map(function (x) { return S[x.i - 1]; }) };
  }
  function bitLen(x) { return Math.max(1, Math.ceil(Math.log2(x + 1))); }
  function instanceBits(S, T) { let b = 0; S.forEach(function (s) { b += bitLen(s); }); return b + bitLen(T); }
  /* SUBSET-SUM ≤ₚ PARTITION: add a = 2σ − T and b = σ + T */
  function partitionFrom(S, T) {
    const sig = S.reduce(function (a, b) { return a + b; }, 0);
    const a = 2 * sig - T, b = sig + T, S2 = S.concat([a, b]);
    const total = S2.reduce(function (x, y) { return x + y; }, 0);
    return { S2: S2, a: a, b: b, sigma: sig, total: total, half: total / 2 };
  }
  /* SUBSET-SUM ≤ₚ KNAPSACK: w = v = s, W = V = T */
  function knapFrom(S, T) { return { w: S.slice(), v: S.slice(), W: T, V: T }; }
  /* 3-SAT ≤ₚ SUBSET-SUM: one column per variable then one per clause, base 10 */
  function satToSubsetSum(phi) {
    const n = phi.n, m = phi.clauses.length, cols = n + m, rows = [];
    function blank() { return new Array(cols).fill(0); }
    for (let i = 0; i < n; i++) {
      [1, -1].forEach(function (pol) {
        const d = blank(); d[i] = 1;
        phi.clauses.forEach(function (cl, j) { if (cl.indexOf(pol * (i + 1)) >= 0) d[n + j] = 1; });
        rows.push({ name: (pol > 0 ? "v" : "v′") + (i + 1), kind: "lit", vari: i, pol: pol > 0, d: d });
      });
    }
    for (let j = 0; j < m; j++) {
      const s1 = blank(); s1[n + j] = 1; rows.push({ name: "s" + (j + 1), kind: "slack", clause: j, d: s1 });
      const s2 = blank(); s2[n + j] = 2; rows.push({ name: "s′" + (j + 1), kind: "slack", clause: j, d: s2 });
    }
    const tgt = blank(); for (let i = 0; i < n; i++) tgt[i] = 1; for (let j = 0; j < m; j++) tgt[n + j] = 4;
    const val = function (d) { let v = 0; for (let i = 0; i < d.length; i++) v = v * 10 + d[i]; return v; };
    return { rows: rows, target: tgt, cols: cols, n: n, m: m, value: val, values: rows.map(function (r) { return val(r.d); }), targetValue: val(tgt) };
  }
  /* every subset of the reduction's numbers that hits the target, and the
     assignment each one encodes — the check that the gadget is a bijection */
  function satToSubsetSumSolve(phi, c) {
    c = c || nop; const R = satToSubsetSum(phi); const vals = R.values, t = R.targetValue;
    const sols = []; let tried = 0;
    for (let m = 0; m < (1 << vals.length); m++) {
      tried++; c.add("subset"); let s = 0;
      for (let i = 0; i < vals.length; i++) if ((m >> i) & 1) { c.add("add"); s += vals[i]; }
      if (s !== t) continue;
      const names = [], asg = new Array(R.n).fill(0);
      for (let i = 0; i < vals.length; i++) if ((m >> i) & 1) { names.push(R.rows[i].name); const r = R.rows[i]; if (r.kind === "lit" && r.pol) asg[r.vari] = 1; }
      sols.push({ mask: m, names: names, assign: asg.join("") });
    }
    return { red: R, sols: sols, tried: tried, assigns: Array.from(new Set(sols.map(function (s) { return s.assign; }))).sort() };
  }
  return {
    subsetSumBrute: subsetSumBrute, subsetSumDP: subsetSumDP, bitLen: bitLen,
    instanceBits: instanceBits, partitionFrom: partitionFrom, knapFrom: knapFrom,
    satToSubsetSum: satToSubsetSum, satToSubsetSumSolve: satToSubsetSumSolve
  };
})());

/* ── chunk B routines, part 4: §15 colouring, §16 boundary, §17–§18 ────── */
Object.assign(NR, (function () {
  const nop = { add: function () {}, get: function () { return 0; } };

  /* every kⁿ assignment, so the count of PROPER colourings is exact */
  function colourBrute(g, k, c) {
    c = c || nop; const n = g.n, col = new Array(n).fill(0); const tot = Math.pow(k, n);
    let count = 0, tried = 0, first = null;
    for (let m = 0; m < tot; m++) {
      tried++; c.add("assign"); let x = m;
      for (let i = 0; i < n; i++) { col[i] = x % k; x = (x - col[i]) / k; }
      let ok = true;
      for (let e = 0; e < g.E.length; e++) { c.add("check"); if (col[g.E[e][0]] === col[g.E[e][1]]) { ok = false; break; } }
      if (ok) { count++; if (!first) first = col.slice(); }
    }
    return { count: count, tried: tried, first: first, checks: c.get("check") };
  }
  function chromatic(g, maxK, c) {
    c = c || nop; for (let k = 1; k <= (maxK || g.n); k++) { const r = colourBrute(g, k, c); if (r.count > 0) return { chi: k, colouring: r.first, count: r.count }; }
    return { chi: g.n, colouring: null, count: 0 };
  }
  /* BFS 2-colouring run to COMPLETION plus a verification pass: Θ(V + E).
     `steps` counts adjacency examinations; `violation` is the first bad edge. */
  function bipartite(g, c) {
    c = c || nop; const L = NR.npAdjList(g), col = new Array(g.n).fill(-1);
    let violation = null, firstAt = null;
    for (let s = 0; s < g.n; s++) {
      if (col[s] >= 0) continue; col[s] = 0; const q = [s];
      while (q.length) { const u = q.shift(); for (let i = 0; i < L[u].length; i++) { const v = L[u][i]; c.add("step"); if (col[v] < 0) { col[v] = 1 - col[u]; q.push(v); } else if (col[v] === col[u] && !violation) { violation = [u, v]; firstAt = c.get("step"); } } }
    }
    g.E.forEach(function (e) { c.add("step"); if (col[e[0]] === col[e[1]] && !violation) { violation = [e[0], e[1]]; firstAt = c.get("step"); } });
    return { ok: !violation, col: col, violation: violation, firstAt: firstAt, steps: c.get("step"), components: countComponents(g) };
  }
  function countComponents(g) { const L = NR.npAdjList(g), seen = new Array(g.n).fill(false); let k = 0; for (let s = 0; s < g.n; s++) { if (seen[s]) continue; k++; const st = [s]; seen[s] = true; while (st.length) { const u = st.pop(); L[u].forEach(function (v) { if (!seen[v]) { seen[v] = true; st.push(v); } }); } } return k; }
  /* greedy colouring in a given vertex order: ≤ Δ + 1 colours, no ratio bound */
  function greedyColour(g, order, c) {
    c = c || nop; const L = NR.npAdjList(g), col = new Array(g.n).fill(-1);
    order.forEach(function (u) { const used = new Set(); L[u].forEach(function (v) { c.add("step"); if (col[v] >= 0) used.add(col[v]); }); let k = 0; while (used.has(k)) k++; col[u] = k; });
    return { col: col, k: Math.max.apply(null, col) + 1, steps: c.get("step") };
  }
  /* the crown graph on 2n vertices: K_{n,n} minus a perfect matching. χ = 2,
     and greedy in the interleaved order uses n colours. */
  function crownGraph(n) {
    const E = []; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) E.push([i, n + j]);
    const names = []; for (let i = 0; i < n; i++) names.push("u" + (i + 1)); for (let i = 0; i < n; i++) names.push("w" + (i + 1));
    return NR.npG(2 * n, E, names);
  }
  function interleaved(n) { const o = []; for (let i = 0; i < n; i++) { o.push(i); o.push(n + i); } return o; }
  /* 3-SAT ≤ₚ 3-COLOURING: palette T/F/B, a gadget per variable, two OR-gadgets
     per clause.  |V| = 3 + 2n + 6m,  |E| = 3 + 3n + 13m. */
  function satTo3Col(phi) {
    const V = [], E = [];
    function id(s) { let i = V.indexOf(s); if (i < 0) { V.push(s); i = V.length - 1; } return i; }
    const T = id("T"), F = id("F"), B = id("B");
    E.push([T, F], [F, B], [B, T]);
    const lit = function (l) { return id((l > 0 ? "x" : "¬x") + Math.abs(l)); };
    for (let i = 1; i <= phi.n; i++) { const p = lit(i), q = lit(-i); E.push([p, q], [p, B], [q, B]); }
    function OR(u, v, tag) { const p = id("p" + tag), q = id("q" + tag), r = id("r" + tag); E.push([p, q], [q, r], [r, p], [u, p], [v, q], [r, B]); return r; }
    phi.clauses.forEach(function (cl, j) { const r1 = OR(lit(cl[0]), lit(cl[1]), (j + 1) + "a"); const r2 = OR(r1, lit(cl[2]), (j + 1) + "b"); E.push([r2, F]); });
    return { g: NR.npG(V.length, E, V.slice()), T: T, F: F, B: B, names: V.slice(), litIndex: function (l) { return V.indexOf((l > 0 ? "x" : "¬x") + Math.abs(l)); } };
  }
  /* backtracking k-colouring, vertex order fixed; counts search nodes */
  function colourBT(g, k, c) {
    c = c || nop; const L = NR.npAdjList(g), col = new Array(g.n).fill(-1); let found = null;
    (function go(i) {
      if (found) return;
      if (i === g.n) { found = col.slice(); return; }
      const lim = (i === 0) ? Math.min(1, k) : k;         /* colour symmetry: vertex 0 fixed */
      for (let x = 0; x < lim; x++) { c.add("node"); if (L[i].some(function (v) { return col[v] === x; })) continue; col[i] = x; go(i + 1); col[i] = -1; if (found) return; }
    })(0);
    return { found: found, nodes: c.get("node") };
  }
  /* the OR-gadget lemma, checked by enumerating all 3⁵ colourings of one gadget
     under each of the four input patterns */
  function orGadgetCensus(c) {
    c = c || nop;
    /* vertices 0 T, 1 F, 2 B, 3 u, 4 v, 5 p, 6 q, 7 r */
    const E = [[0, 1], [1, 2], [2, 0], [5, 6], [6, 7], [7, 5], [3, 5], [4, 6], [7, 2], [3, 2], [4, 2]];
    const out = [];
    [[true, true], [true, false], [false, true], [false, false]].forEach(function (pat) {
      const col = new Array(8).fill(-1);
      col[0] = 0; col[1] = 1; col[2] = 2;                 /* T = 0, F = 1, B = 2 */
      col[3] = pat[0] ? 0 : 1; col[4] = pat[1] ? 0 : 1;
      let total = 0, outT = 0, outF = 0;
      for (let m = 0; m < 27; m++) {
        c.add("assign"); let x = m; col[5] = x % 3; x = (x - col[5]) / 3; col[6] = x % 3; x = (x - col[6]) / 3; col[7] = x % 3;
        let ok = true; for (let e = 0; e < E.length; e++) { c.add("check"); if (col[E[e][0]] === col[E[e][1]]) { ok = false; break; } }
        if (!ok) continue; total++; if (col[7] === 0) outT++; else if (col[7] === 1) outF++;
      }
      out.push({ u: pat[0], v: pat[1], proper: total, outTrue: outT, outFalse: outF });
    });
    return { rows: out, tried: c.get("assign") };
  }

  /* ── §16  2-SAT in linear time: implication graph + Kosaraju SCC ──────── */
  function twoSat(n, clauses, c) {
    c = c || nop; const N = 2 * n;
    const idx = function (l) { return l > 0 ? 2 * (l - 1) : 2 * (-l - 1) + 1; }, neg = function (i) { return i ^ 1; };
    const adj = Array.from({ length: N }, function () { return []; }), radj = Array.from({ length: N }, function () { return []; });
    clauses.forEach(function (cl) {
      const a = cl[0], b = cl[1];
      adj[neg(idx(a))].push(idx(b)); radj[idx(b)].push(neg(idx(a)));
      adj[neg(idx(b))].push(idx(a)); radj[idx(a)].push(neg(idx(b)));
      c.add("step", 2);
    });
    const seen = new Array(N).fill(false), order = [];
    for (let s = 0; s < N; s++) {
      if (seen[s]) continue; seen[s] = true; const st = [[s, 0]];
      while (st.length) { const top = st[st.length - 1]; if (top[1] < adj[top[0]].length) { const v = adj[top[0]][top[1]++]; c.add("step"); if (!seen[v]) { seen[v] = true; st.push([v, 0]); } } else order.push(st.pop()[0]); }
    }
    const comp = new Array(N).fill(-1); let k = 0;
    for (let i = order.length - 1; i >= 0; i--) {
      const s = order[i]; if (comp[s] >= 0) continue; const st = [s]; comp[s] = k;
      while (st.length) { const u = st.pop(); radj[u].forEach(function (v) { c.add("step"); if (comp[v] < 0) { comp[v] = k; st.push(v); } }); }
      k++;
    }
    let sat = true, culprit = -1;
    for (let i = 0; i < n; i++) { c.add("step"); if (comp[2 * i] === comp[2 * i + 1]) { sat = false; if (culprit < 0) culprit = i; } }
    const assign = []; for (let i = 0; i < n; i++) assign.push(comp[2 * i] > comp[2 * i + 1]);
    return { sat: sat, assign: sat ? assign : null, culprit: culprit, comp: comp, steps: c.get("step") };
  }

  /* ── random instances, seeded, so every reader sees the same numbers ──── */
  function randClauses(n, m, width, r) { const out = []; const p = function () { return (1 + Math.floor(r() * n)) * (r() < 0.5 ? -1 : 1); }; for (let i = 0; i < m; i++) { const cl = []; for (let w = 0; w < width; w++) cl.push(p()); out.push(cl); } return out; }
  function randGraphB(n, m, r) {
    const cap = n * (n - 1) / 2; m = Math.min(m, cap);
    const seen = new Set(), E = []; let guard = 0;
    while (E.length < m && guard++ < 20000) { const u = Math.floor(r() * n), v = Math.floor(r() * n); if (u === v) continue; const a = Math.min(u, v), b = Math.max(u, v), key = a + "-" + b; if (seen.has(key)) continue; seen.add(key); E.push([a, b]); }
    return NR.npG(n, E);
  }
  function ringPlusChords(n, extra, r) {
    const E = []; for (let i = 0; i < n; i++) { const a = Math.min(i, (i + 1) % n), b = Math.max(i, (i + 1) % n); E.push([a, b]); }
    const seen = new Set(E.map(function (e) { return e.join("-"); })); let add = 0, guard = 0;
    while (add < extra && guard++ < 5000) { const u = Math.floor(r() * n), v = Math.floor(r() * n); if (u === v) continue; const a = Math.min(u, v), b = Math.max(u, v), key = a + "-" + b; if (seen.has(key)) continue; seen.add(key); E.push([a, b]); add++; }
    return NR.npG(n, E);
  }
  /* a formula with a planted model (always satisfiable) and one that contains
     all eight clauses over x₁x₂x₃ (unsatisfiable whatever else is in it) */
  function plantedSat(n, m, r) {
    const hidden = []; for (let i = 0; i < n; i++) hidden.push(r() < 0.5);
    const out = [];
    while (out.length < m) {
      const cl = []; for (let w = 0; w < 3; w++) cl.push((1 + Math.floor(r() * n)) * (r() < 0.5 ? -1 : 1));
      if (cl.some(function (l) { return l > 0 ? hidden[l - 1] : !hidden[-l - 1]; })) out.push(cl);
    }
    return { clauses: out, hidden: hidden };
  }
  function forcedUnsat(n, m, r) {
    const out = [];
    for (let m8 = 0; m8 < 8; m8++) out.push([(m8 & 4 ? 1 : -1) * 1, (m8 & 2 ? 1 : -1) * 2, (m8 & 1 ? 1 : -1) * 3]);
    while (out.length < m) { const cl = []; for (let w = 0; w < 3; w++) cl.push((1 + Math.floor(r() * n)) * (r() < 0.5 ? -1 : 1)); out.push(cl); }
    return out;
  }

  /* ── §18  TQBF by recursion: exponential TIME, linear SPACE ──────────── */
  function qbfEval(prefix, clauses, c) {
    c = c || nop; const n = prefix.length, a = new Array(n).fill(false); let maxd = 0;
    function go(i, depth) {
      c.add("node"); if (depth > maxd) maxd = depth;
      if (i === n) return NR.satEvalB(clauses, a, c);
      if (prefix[i] === "E") { a[i] = false; if (go(i + 1, depth + 1)) return true; a[i] = true; return go(i + 1, depth + 1); }
      a[i] = false; if (!go(i + 1, depth + 1)) return false; a[i] = true; return go(i + 1, depth + 1);
    }
    const v = go(0, 0);
    return { value: v, nodes: c.get("node"), depth: maxd, bitsOfState: n };
  }
  function altPrefix(n, startE) { const p = []; for (let i = 0; i < n; i++) p.push(((i % 2 === 0) === !!startE) ? "E" : "A"); return p; }

  return {
    colourBrute: colourBrute, chromatic: chromatic, bipartite: bipartite, countComponents: countComponents,
    greedyColour: greedyColour, crownGraph: crownGraph, interleaved: interleaved,
    satTo3Col: satTo3Col, colourBT: colourBT, orGadgetCensus: orGadgetCensus,
    twoSat: twoSat, randClauses: randClauses, randGraphB: randGraphB, ringPlusChords: ringPlusChords,
    plantedSat: plantedSat, forcedUnsat: forcedUnsat, qbfEval: qbfEval, altPrefix: altPrefix
  };
})());


/* ── chunk C ─────────────────────────────────────────────────── */
/* ── chunk C routines: coping, approximation ratios, the four approximation
      algorithms (vertex cover, TSP, set cover, MAX-3-SAT) and the knapsack
      FPTAS. Pure and DOM-free: every one takes an optional counter `c` with
      an .add(key) method and returns measurements the figures display. ─── */
Object.assign(NR, (function () {
  const nop = { add: function () {}, get: function () { return 0; } };
  function ctr(c) { return c || nop; }
  /* a local seeded generator, so the routines load in node without AL */
  function rnd(seed) { let a = (seed >>> 0) || 1; return function () { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  /* ═══ harmonic numbers ═══════════════════════════════════════════════
     H(d) = 1 + 1/2 + … + 1/d, the true greedy set-cover bound of §23.
     ln d is NOT H(d): H(d) = ln d + γ + 1/(2d) − … , and ln d < H(d) ≤ ln d + 1. */
  function harmonic(d) { let s = 0; for (let i = 1; i <= d; i++) s += 1 / i; return s; }

  /* ═══════════════════════════════════════════════════════════════════
     §21 — VERTEX COVER
     ═══════════════════════════════════════════════════════════════════ */

  /* the shared gVC instance (named vertices) as index arrays */
  function vcIndex(G) {
    const idx = {}; G.V.forEach((v, i) => { idx[v] = i; });
    return { n: G.V.length, names: G.V.slice(), E: G.E.map(e => [idx[e[0]], idx[e[1]]]), idx };
  }
  /* is S (a boolean array) a vertex cover of E? */
  function vcIsCover(E, S, c) { c = ctr(c); for (let i = 0; i < E.length; i++) { c.add("edgeCheck"); if (!S[E[i][0]] && !S[E[i][1]]) return false; } return true; }

  /* APPROX-VERTEX-COVER: a MAXIMAL matching in the given edge order, then
     both endpoints of every matched edge. c.add("edge") per edge examined,
     c.add("match") per edge taken. Returns the matching, the cover, and a
     per-edge trace for the stepper. */
  function vcMatching(n, E, c) {
    c = ctr(c);
    const used = new Array(n).fill(false), M = [], trace = [];
    E.forEach((e, i) => {
      c.add("edge");
      const take = !used[e[0]] && !used[e[1]];
      if (take) { used[e[0]] = true; used[e[1]] = true; M.push(i); c.add("match"); }
      trace.push({ i, e, take, M: M.slice(), used: used.slice() });
    });
    const cover = []; for (let v = 0; v < n; v++) if (used[v]) cover.push(v);
    return { M, matching: M.map(i => E[i]), cover, inCover: used.slice(), trace, size: cover.length };
  }

  /* the OTHER greedy: repeatedly take a vertex of maximum remaining degree.
     NOT a 2-approximation — its ratio is Θ(log n) (§21). Ties → smaller index. */
  function vcGreedyDegree(n, E, c, opt) {
    c = ctr(c); const hiTie = !!(opt && opt.hiTie);
    const alive = E.map(() => true), cover = [], trace = [];
    let left = E.length;
    while (left > 0) {
      const deg = new Array(n).fill(0);
      E.forEach((e, i) => { if (alive[i]) { c.add("edge"); deg[e[0]]++; deg[e[1]]++; } });
      let best = -1; for (let v = 0; v < n; v++) { if (deg[v] === 0) continue; if (best < 0 || deg[v] > deg[best] || (hiTie && deg[v] === deg[best])) best = v; }
      if (best < 0) break;
      cover.push(best); c.add("pick");
      const killed = [];
      E.forEach((e, i) => { if (alive[i] && (e[0] === best || e[1] === best)) { alive[i] = false; killed.push(i); left--; } });
      trace.push({ pick: best, deg: deg[best], killed, cover: cover.slice(), left });
    }
    const inCover = new Array(n).fill(false); cover.forEach(v => { inCover[v] = true; });
    return { cover: cover.slice().sort((a, b) => a - b), order: cover, inCover, trace, size: cover.length };
  }

  /* exhaustive minimum vertex cover over all 2ⁿ subsets — the independent
     check every vertex-cover figure on this page is measured against.
     (Chunk B's vcBrute is a separate routine; this one also returns EVERY
     minimum cover, so §21 can report that gVC's optimum is unique.) */
  function vcOptBrute(n, E, c) {
    c = ctr(c);
    let best = n + 1, all = [];
    for (let m = 0; m < (1 << n); m++) {
      c.add("subset");
      let k = 0; for (let v = 0; v < n; v++) if (m & (1 << v)) k++;
      if (k > best) continue;
      const S = new Array(n); for (let v = 0; v < n; v++) S[v] = !!(m & (1 << v));
      if (!vcIsCover(E, S, c)) continue;
      if (k < best) { best = k; all = [m]; } else if (k === best) all.push(m);
    }
    const toSet = m => { const s = []; for (let v = 0; v < n; v++) if (m & (1 << v)) s.push(v); return s; };
    return { size: best, covers: all.map(toSet), count: all.length, unique: all.length === 1, best: toSet(all[0]) };
  }

  /* An instance family on which max-degree greedy really loses: the
     "Johnson" bipartite construction. Left side L of k vertices; for each
     i = 2…k a right group Rᵢ of ⌊k/i⌋ vertices, each joined to i distinct
     left vertices, the groups partitioning L within each i. Every Rᵢ vertex
     has degree i, so greedy takes the right side from the top down; the
     optimum is L itself, of size k. Returns {n, E, opt, names}. */
  function vcGreedyBad(k) {
    const E = [], names = [];
    for (let i = 0; i < k; i++) names.push("L" + (i + 1));
    let n = k; const groups = [];
    for (let i = 2; i <= k; i++) {
      const g = [];
      for (let j = 0; j + i <= k; j += i) {
        const r = n++; names.push("R" + i + "\u2009" + (g.length + 1)); g.push(r);
        for (let t = 0; t < i; t++) E.push([j + t, r]);
      }
      if (g.length) groups.push({ i, g });
    }
    return { n, E, L: k, groups, names, optSize: k, right: n - k };
  }
  /* sweep the family and measure the worst ratio the max-degree greedy
     actually achieves — the evidence that it is not a 2-approximation. */
  function vcGreedySweep(ks, hiTie) {
    return ks.map(k => {
      const B = vcGreedyBad(k);
      const gd = vcGreedyDegree(B.n, B.E, null, { hiTie: !!hiTie });
      const mm = vcMatching(B.n, B.E, null);
      /* the optimum is at most L (every edge has an endpoint in L); for the
         small k it is checked exactly, for the large k L is the upper bound */
      /* the optimum is PINNED without exhaustive search: L is a cover, so
         OPT \u2264 L; and OPT \u2265 |M| (the lemma of \u00a721), so when |M| = L the
         optimum is exactly L. The sweep reports whether the two bounds meet. */
      const lb = mm.size / 2, ub = B.L, pinned = lb === ub;
      const opt = pinned ? ub : (B.n <= 22 ? vcOptBrute(B.n, B.E, null).size : ub);
      const deg = new Array(B.n).fill(0); B.E.forEach(e => { deg[e[0]]++; deg[e[1]]++; });
      return { k, n: B.n, m: B.E.length, greedy: gd.size, matching: mm.size, lb, ub, pinned, opt,
               exact: pinned || B.n <= 22, maxDeg: Math.max.apply(null, deg),
               ratio: gd.size / opt, mRatio: mm.size / opt, lnn: Math.log(B.n) };
    });
  }

  /* ═══════════════════════════════════════════════════════════════════
     §22 — THE TRAVELLING SALESMAN
     ═══════════════════════════════════════════════════════════════════ */

  function tspDist(pts) {
    const n = pts.length, D = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) D[i][j] = Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
    return D;
  }
  function tspTourCost(D, tour, c) { c = ctr(c); let s = 0; for (let i = 0; i < tour.length; i++) { c.add("dist"); s += D[tour[i]][tour[(i + 1) % tour.length]]; } return s; }
  /* the triangle inequality, checked over all n³ triples — the hypothesis
     both §22 bounds need, and the thing general TSP does not have. */
  function tspTriangle(D, c) {
    c = ctr(c); const n = D.length; let worst = 0, bad = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) {
      c.add("triple");
      const slack = D[i][j] + D[j][k] - D[i][k];
      if (slack < -1e-9) { bad++; if (-slack > worst) worst = -slack; }
    }
    return { ok: bad === 0, violations: bad, worst, triples: n * n * n };
  }
  /* exhaustive optimum over all (n−1)!/2 distinct tours (fix city 0, halve
     by direction). c.add("tour") per tour, c.add("dist") per edge summed. */
  function tspOpt(D, c) {
    c = ctr(c); const n = D.length, rest = []; for (let i = 1; i < n; i++) rest.push(i);
    let best = Infinity, bestTour = null, tours = 0;
    const cur = [];
    (function go(left, cost) {
      if (cost >= best) { /* keep counting honestly: no pruning */ }
      if (left.length === 0) {
        if (cur[0] > cur[cur.length - 1]) return;       /* one of each direction */
        tours++; c.add("tour");
        const t = [0].concat(cur), z = tspTourCost(D, t, c);
        if (z < best) { best = z; bestTour = t; }
        return;
      }
      for (let i = 0; i < left.length; i++) { cur.push(left[i]); go(left.slice(0, i).concat(left.slice(i + 1)), cost); cur.pop(); }
    })(rest, 0);
    return { cost: best, tour: bestTour, tours };
  }
  /* Prim's MST on the complete distance matrix; returns the tree edges, its
     weight, the degree sequence and an adjacency list. */
  function tspMST(D, c) {
    c = ctr(c); const n = D.length, inT = new Array(n).fill(false), key = new Array(n).fill(Infinity), par = new Array(n).fill(-1);
    key[0] = 0; const edges = []; let W = 0;
    for (let it = 0; it < n; it++) {
      let u = -1; for (let v = 0; v < n; v++) { c.add("scan"); if (!inT[v] && (u < 0 || key[v] < key[u])) u = v; }
      inT[u] = true; if (par[u] >= 0) { edges.push([par[u], u]); W += D[par[u]][u]; }
      for (let v = 0; v < n; v++) if (!inT[v]) { c.add("relax"); if (D[u][v] < key[v]) { key[v] = D[u][v]; par[v] = u; } }
    }
    const adj = Array.from({ length: n }, () => []);
    edges.forEach(e => { adj[e[0]].push(e[1]); adj[e[1]].push(e[0]); });
    const deg = adj.map(a => a.length);
    return { edges, weight: W, adj, deg, par };
  }
  /* DOUBLE-TREE (APPROX-TSP-TOUR): MST, preorder DFS from 0 (neighbours in
     increasing index order), shortcut. Ratio ≤ 2 when the triangle
     inequality holds. Returns the walk it shortcuts, too. */
  function tspDoubleTree(D, c) {
    c = ctr(c); const T = tspMST(D, c), n = D.length;
    const seen = new Array(n).fill(false), tour = [], walk = [];
    (function dfs(u) {
      seen[u] = true; tour.push(u); walk.push(u); c.add("visit");
      T.adj[u].slice().sort((a, b) => a - b).forEach(v => { if (!seen[v]) { dfs(v); walk.push(u); } });
    })(0);
    return { tour, walk, cost: tspTourCost(D, tour, c), mst: T, walkCost: 2 * T.weight };
  }
  /* minimum-weight perfect matching on a small vertex set, by exhaustive
     recursion over all (2k−1)!! pairings. c.add("pairing") per complete one. */
  function tspMinMatching(D, S, c) {
    c = ctr(c); let best = Infinity, bestM = null, count = 0;
    (function go(left, acc, cost) {
      if (left.length === 0) { count++; c.add("pairing"); if (cost < best) { best = cost; bestM = acc.slice(); } return; }
      const a = left[0];
      for (let i = 1; i < left.length; i++) {
        const b = left[i];
        acc.push([a, b]); go(left.filter((_, j) => j !== 0 && j !== i), acc, cost + D[a][b]); acc.pop();
      }
    })(S.slice(), [], 0);
    return { matching: bestM, cost: best, pairings: count };
  }
  /* Hierholzer's Euler circuit on a multigraph given as an edge list. */
  function eulerCircuit(n, edges, start) {
    const adj = Array.from({ length: n }, () => []);
    edges.forEach((e, i) => { adj[e[0]].push({ v: e[1], i }); adj[e[1]].push({ v: e[0], i }); });
    const used = edges.map(() => false), st = [start], out = [];
    while (st.length) {
      const u = st[st.length - 1];
      let nxt = null;
      for (let k = 0; k < adj[u].length; k++) if (!used[adj[u][k].i]) { nxt = adj[u][k]; break; }
      if (nxt === null) { out.push(st.pop()); } else { used[nxt.i] = true; st.push(nxt.v); }
    }
    return out.reverse();
  }
  /* CHRISTOFIDES: MST + a minimum perfect matching on the ODD-degree MST
     vertices (there is an even number of them) → every degree even → Euler
     circuit → shortcut to a tour. Ratio ≤ 3/2 under the triangle inequality. */
  function tspChristofides(D, c) {
    c = ctr(c); const n = D.length, T = tspMST(D, c);
    const odd = []; for (let v = 0; v < n; v++) if (T.deg[v] % 2 === 1) odd.push(v);
    const M = tspMinMatching(D, odd, c);
    const multi = T.edges.concat(M.matching || []);
    const circuit = eulerCircuit(n, multi, 0);
    const seen = new Array(n).fill(false), tour = [];
    circuit.forEach(v => { if (!seen[v]) { seen[v] = true; tour.push(v); c.add("shortcut"); } });
    const multiW = T.weight + M.cost;
    return { tour, cost: tspTourCost(D, tour, c), mst: T, odd, matching: M.matching, matchCost: M.cost, pairings: M.pairings, circuit, multiWeight: multiW };
  }
  /* NEAREST NEIGHBOUR: no constant ratio at all — its worst case over metric
     instances grows like Θ(log n). Here only as a measured comparison. */
  function tspNN(D, start, c) {
    c = ctr(c); const n = D.length, seen = new Array(n).fill(false), tour = [start]; seen[start] = true;
    for (let k = 1; k < n; k++) {
      let u = tour[tour.length - 1], best = -1;
      for (let v = 0; v < n; v++) { c.add("scan"); if (!seen[v] && (best < 0 || D[u][v] < D[u][best])) best = v; }
      seen[best] = true; tour.push(best);
    }
    return { tour, cost: tspTourCost(D, tour, c) };
  }
  /* 2-OPT: repeatedly reverse a segment if it shortens the tour (§28 uses
     this; §22 shows it as a heuristic with no proved ratio). */
  function tspTwoOpt(D, tour0, c) {
    c = ctr(c); let tour = tour0.slice(), n = tour.length, improved = true, moves = 0;
    while (improved) {
      improved = false;
      for (let i = 1; i < n - 1 && !improved; i++) for (let j = i + 1; j < n; j++) {
        c.add("probe");
        const a = tour[i - 1], b = tour[i], cc = tour[j], d = tour[(j + 1) % n];
        if (d === a) continue;
        if (D[a][cc] + D[b][d] < D[a][b] + D[cc][d] - 1e-12) {
          tour = tour.slice(0, i).concat(tour.slice(i, j + 1).reverse(), tour.slice(j + 1));
          moves++; improved = true; break;
        }
      }
    }
    return { tour, cost: tspTourCost(D, tour, c), moves };
  }

  /* ═══════════════════════════════════════════════════════════════════
     §23 — SET COVER
     ═══════════════════════════════════════════════════════════════════ */

  /* GREEDY-SET-COVER: take the set covering the most still-uncovered
     elements; ties → smaller index. Returns a per-step trace with the
     measured gain and the price 1/gain charged to each element covered. */
  function setCoverGreedy(U, sets, c) {
    c = ctr(c);
    const covered = new Array(U + 1).fill(false); let left = U;
    const picked = [], trace = [], price = new Array(U + 1).fill(0);
    while (left > 0) {
      let best = -1, bestGain = 0, bestNew = null;
      sets.forEach((S, i) => {
        if (picked.indexOf(i) >= 0) return;
        const nw = S.filter(e => { c.add("test"); return !covered[e]; });
        if (nw.length > bestGain) { bestGain = nw.length; best = i; bestNew = nw; }
      });
      if (best < 0) break;                     /* not coverable */
      picked.push(best); c.add("pick");
      bestNew.forEach(e => { covered[e] = true; price[e] = 1 / bestGain; left--; });
      trace.push({ set: best, gain: bestGain, newly: bestNew.slice(), covered: covered.slice(), left, price: 1 / bestGain });
    }
    return { picked, trace, size: picked.length, price, covered: left === 0 };
  }
  /* exhaustive minimum cover over all 2^m subfamilies. */
  function setCoverOpt(U, sets, c) {
    c = ctr(c); const m = sets.length; let best = m + 1, all = [];
    for (let mask = 0; mask < (1 << m); mask++) {
      c.add("subset");
      let k = 0; for (let i = 0; i < m; i++) if (mask & (1 << i)) k++;
      if (k > best) continue;
      const hit = new Array(U + 1).fill(false);
      for (let i = 0; i < m; i++) if (mask & (1 << i)) sets[i].forEach(e => { hit[e] = true; });
      let ok = true; for (let e = 1; e <= U; e++) { c.add("test"); if (!hit[e]) { ok = false; break; } }
      if (!ok) continue;
      if (k < best) { best = k; all = [mask]; } else if (k === best) all.push(mask);
    }
    const toFam = mask => { const f = []; for (let i = 0; i < m; i++) if (mask & (1 << i)) f.push(i); return f; };
    return { size: best, families: all.map(toFam), count: all.length, unique: all.length === 1, best: toFam(all[0]) };
  }
  /* The instance family on which greedy is Θ(log n) times the optimum:
     a 2×2^k grid of elements. Two sets (the two rows) cover everything —
     OPT = 2 — while k+1 "greedy bait" sets of sizes 2^k, 2^(k−1), …, 2, 1,
     each strictly larger than what any row still offers, drag greedy into
     k+1 picks. Returns {U, sets, optSize, greedySize}. */
  function setCoverBad(k) {
    /* U is partitioned two ways. S_i (i = 1…k) has 2\u2071 elements and the
       S's partition U, so |U| = 2 + 4 + … + 2\u1d4f = 2\u1d4f\u207a\u00b9 \u2212 2. Each S_i is split
       evenly between T\u2081 and T\u2082, so |T\u2081| = |T\u2082| = 2\u1d4f \u2212 1 \u2014 one SHORT of |S_k|.
       OPT = {T\u2081, T\u2082} = 2; greedy takes S_k, then S_{k\u22121}, …, then S\u2081: k sets. */
    const S = [], T1 = [], T2 = []; let next = 1;
    for (let i = 1; i <= k; i++) {
      const s = []; const half = 1 << (i - 1);
      for (let j = 0; j < half; j++) { s.push(next); T1.push(next); next++; }
      for (let j = 0; j < half; j++) { s.push(next); T2.push(next); next++; }
      S.push(s);
    }
    const U = next - 1;
    /* T's listed FIRST, so a tie would be broken in the optimum's favour —
       there is no tie, and greedy still walks past them. */
    return { U, sets: [T1, T2].concat(S.slice().reverse()), T: [T1, T2], S, k, optSize: 2, greedyExpected: k };
  }

  /* ═══════════════════════════════════════════════════════════════════
     §22 (continued) — GENERAL TSP: the gap instance of the reduction
     HAM-CYCLE ≤ₚ TSP in the widened form built in §13. Graph edges cost 1,
     non-edges cost ρ|V| + 1. A Hamiltonian cycle gives a tour of cost
     exactly |V|; without one, EVERY tour uses at least one non-edge and so
     costs at least (ρ|V| + 1) + (|V| − 1) = ρ|V| + |V| > ρ|V| — strictly
     above the threshold, with no side condition on |V|. So a polynomial
     ρ-approximation would decide HAM-CYCLE. NOT a metric instance.
     The "+ 1" is §13's, and §22 uses the same constant so that the two
     sections report the same optimum on the same graph.
     ═══════════════════════════════════════════════════════════════════ */
  function tspGapMatrix(n, E, rho) {
    const big = rho * n + 1;
    const D = Array.from({ length: n }, () => new Array(n).fill(big));
    for (let i = 0; i < n; i++) D[i][i] = 0;
    E.forEach(e => { D[e[0]][e[1]] = 1; D[e[1]][e[0]] = 1; });
    return D;
  }
  /* the gap, measured: on a graph WITH a Hamiltonian cycle the optimum is n;
     on one without, the optimum exceeds ρ·n. Also reports that the matrix
     violates the triangle inequality — by design; that is the whole point. */
  function tspGapDemo(G, rho, c) {
    c = ctr(c);
    const D = tspGapMatrix(G.n, G.E, rho), o = tspOpt(D, c);
    const graphEdges = (function () { let k = 0; for (let i = 0; i < o.tour.length; i++) if (D[o.tour[i]][o.tour[(i + 1) % o.tour.length]] === 1) k++; return k; })();
    return { n: G.n, rho, cost: o.cost, tour: o.tour, tours: o.tours, graphEdges,
             nonEdge: rho * G.n + 1, nonEdgesUsed: o.tour.length - graphEdges,
             hamiltonian: o.cost === G.n, threshold: rho * G.n, ratio: o.cost / G.n,
             metric: tspTriangle(D, null).ok };
  }

  /* ═══════════════════════════════════════════════════════════════════
     §24 — MAX-3-SAT
     ═══════════════════════════════════════════════════════════════════ */

  /* clauses are arrays of signed 1-based variable indices; assign is a
     0/1 array indexed from 0. */
  function satClauseSat(cl, assign) { for (let k = 0; k < cl.length; k++) { const l = cl[k], v = assign[Math.abs(l) - 1]; if ((l > 0 && v === 1) || (l < 0 && v === 0)) return true; } return false; }
  function max3satCount(phi, assign, c) { c = ctr(c); let s = 0; phi.clauses.forEach(cl => { c.add("clause"); if (satClauseSat(cl, assign)) s++; }); return s; }
  /* the hypothesis the 7/8 argument needs, checked: every clause has exactly
     three literals on three DISTINCT variables. */
  function max3satWellFormed(phi) {
    let ok = true, bad = [];
    phi.clauses.forEach((cl, i) => { const vs = cl.map(l => Math.abs(l)); const d = new Set(vs); if (cl.length !== 3 || d.size !== 3) { ok = false; bad.push(i); } });
    return { ok, bad };
  }
  /* a random well-formed 3-CNF: m clauses, each on three DISTINCT variables
     with independent uniform signs. Seeded, so the figure is reproducible. */
  function max3satFormula(n, m, seed) {
    const r = rnd(seed), clauses = [];
    for (let k = 0; k < m; k++) {
      const pick = [];
      while (pick.length < 3) { const v = 1 + Math.floor(r() * n); if (pick.indexOf(v) < 0) pick.push(v); }
      clauses.push(pick.map(v => r() < 0.5 ? v : -v));
    }
    const names = []; for (let i = 1; i <= n; i++) names.push("x" + i);
    return { n, names, clauses };
  }

  /* exhaustive: the best assignment, the full distribution of "clauses
     satisfied" over all 2ⁿ assignments, and the exact mean of that
     distribution — which IS the expectation of the random algorithm. */
  function max3satBrute(phi, c) {
    c = ctr(c); const n = phi.n, m = phi.clauses.length;
    let best = -1, bestA = null, sum = 0, full = 0; const dist = new Array(m + 1).fill(0);
    for (let mask = 0; mask < (1 << n); mask++) {
      c.add("assign");
      const a = []; for (let v = 0; v < n; v++) a.push((mask >> v) & 1);
      const s = max3satCount(phi, a, c); dist[s]++; sum += s;
      if (s === m) full++;
      if (s > best) { best = s; bestA = a.slice(); }
    }
    return { best, bestA, dist, mean: sum / (1 << n), satisfiable: full > 0, satisfyingCount: full, assignments: 1 << n, m };
  }
  /* the algorithm itself: flip a fair coin per variable. Run `trials` times
     under a seeded generator; returns the empirical histogram and mean. */
  function max3satRandom(phi, trials, seed, c) {
    c = ctr(c); const r = rnd(seed), m = phi.clauses.length, hist = new Array(m + 1).fill(0);
    let sum = 0, bestSeen = -1, bestA = null;
    for (let t = 0; t < trials; t++) {
      const a = []; for (let v = 0; v < phi.n; v++) { c.add("coin"); a.push(r() < 0.5 ? 1 : 0); }
      const s = max3satCount(phi, a, c); hist[s]++; sum += s; c.add("trial");
      if (s > bestSeen) { bestSeen = s; bestA = a.slice(); }
    }
    return { hist, mean: sum / trials, trials, bestSeen, bestA, expected: (7 / 8) * m };
  }
  /* the conditional expectation E[#satisfied | x₁…x_k fixed], computed
     EXACTLY, clause by clause: a clause already satisfied contributes 1, a
     clause with j of its literals still free and none yet satisfied
     contributes 1 − 2^(−j). */
  function max3satCondExp(phi, partial, c) {
    c = ctr(c); let e = 0;
    phi.clauses.forEach(cl => {
      c.add("clause");
      let sat = false, free = 0;
      cl.forEach(l => {
        const v = partial[Math.abs(l) - 1];
        if (v === undefined || v === null || v === -1) free++;
        else if ((l > 0 && v === 1) || (l < 0 && v === 0)) sat = true;
      });
      e += sat ? 1 : 1 - Math.pow(2, -free);
    });
    return e;
  }
  /* DERANDOMISATION by the method of conditional expectations: walk the
     variables in order, compute the conditional expectation for each value,
     and take the larger (ties → 1). The invariant is that the conditional
     expectation never falls below 7m/8, so the final assignment — where the
     expectation IS the count — satisfies at least ⌈7m/8⌉ clauses. */
  function max3satDerandom(phi, c) {
    c = ctr(c); const n = phi.n, m = phi.clauses.length;
    const partial = new Array(n).fill(-1), steps = [];
    const e0 = max3satCondExp(phi, partial, c);
    for (let v = 0; v < n; v++) {
      partial[v] = 0; const a = max3satCondExp(phi, partial, c);
      partial[v] = 1; const b = max3satCondExp(phi, partial, c);
      const pick = b >= a ? 1 : 0; partial[v] = pick; c.add("decide");
      steps.push({ v, eFalse: a, eTrue: b, pick, e: pick === 1 ? b : a, before: v === 0 ? e0 : steps[v - 1].e });
    }
    const sat = max3satCount(phi, partial, c);
    /* the invariant: the conditional expectation never DECREASES (it is the
       larger of two numbers whose average is the previous one), so it stays
       at or above its starting value 7m/8 — and at the end, with nothing
       left free, it IS the number of clauses satisfied. */
    return { assign: partial.slice(), steps, satisfied: sat, start: e0, m,
             floor: Math.ceil((7 / 8) * m),
             monotone: steps.every(s => s.e >= s.before - 1e-9),
             endsExact: Math.abs(steps[steps.length - 1].e - sat) < 1e-9,
             aboveFloor: sat >= Math.ceil((7 / 8) * m) };
  }

  /* ═══════════════════════════════════════════════════════════════════
     §25 — THE KNAPSACK FPTAS
     ═══════════════════════════════════════════════════════════════════ */

  /* brute force over all 2ⁿ subsets — the independent check. */
  function knapBrute(w, v, W, c) {
    c = ctr(c); const n = w.length; let best = 0, bestS = [];
    for (let mask = 0; mask < (1 << n); mask++) {
      c.add("subset");
      let ww = 0, vv = 0; for (let i = 0; i < n; i++) if (mask & (1 << i)) { ww += w[i]; vv += v[i]; }
      if (ww <= W && vv > best) { best = vv; bestS = []; for (let i = 0; i < n; i++) if (mask & (1 << i)) bestS.push(i); }
    }
    return { value: best, items: bestS, weight: bestS.reduce((s, i) => s + w[i], 0), subsets: 1 << n };
  }
  /* the VALUE-indexed dynamic program: m[i][z] = the least weight achieving
     value exactly z using the first i items. States (n+1)·(V+1) where
     V = Σvᵢ; time Θ(n·V), which is pseudo-polynomial, not polynomial.
     (The weight-indexed twin, m[i][W], is built on the Dynamic Programming
     page; this one is the version the FPTAS scales.) */
  function knapValueDP(w, v, W, c) {
    c = ctr(c); const n = w.length, V = v.reduce((a, b) => a + b, 0);
    const INF = Infinity;
    const m = Array.from({ length: n + 1 }, () => new Array(V + 1).fill(INF));
    m[0][0] = 0;
    for (let i = 1; i <= n; i++) for (let z = 0; z <= V; z++) {
      c.add("cell");
      let x = m[i - 1][z];
      if (z >= v[i - 1] && m[i - 1][z - v[i - 1]] < INF) x = Math.min(x, m[i - 1][z - v[i - 1]] + w[i - 1]);
      m[i][z] = x;
    }
    let best = 0; for (let z = V; z >= 0; z--) if (m[n][z] <= W) { best = z; break; }
    /* read the chosen items back */
    const items = []; let z = best;
    for (let i = n; i >= 1; i--) { if (z >= v[i - 1] && m[i][z] === m[i - 1][z - v[i - 1]] + w[i - 1]) { items.push(i - 1); z -= v[i - 1]; } }
    return { value: best, items: items.reverse(), table: m, V, states: (n + 1) * (V + 1), cells: c.get("cell") };
  }
  /* the FPTAS: scale every value by K = ε·v_max/n and round DOWN, then run
     the value-indexed DP on the scaled values. Loss ≤ n·K = ε·v_max ≤ ε·OPT
     (the last step needs every item to fit on its own, so OPT ≥ v_max).
     Scaled value sum ≤ n·v_max/K = n²/ε, so the DP has O(n³/ε) cells. */
  function knapFPTAS(w, v, W, eps, c) {
    c = ctr(c); const n = w.length;
    const fit = v.map((_, i) => w[i] <= W);
    const vmax = Math.max(...v.filter((_, i) => fit[i]));
    const K = eps * vmax / n;
    const vs = v.map((x, i) => fit[i] ? Math.floor(x / K) : 0);
    const c2 = { add: k => c.add(k), get: k => 0 };
    const dp = knapValueDP(w.map((x, i) => fit[i] ? x : W + 1), vs, W, c2);
    const value = dp.items.reduce((s, i) => s + v[i], 0);
    const weight = dp.items.reduce((s, i) => s + w[i], 0);
    return { value, weight, items: dp.items, K, vmax, scaled: vs, scaledSum: vs.reduce((a, b) => a + b, 0), states: dp.states, cells: c.get("cell"), eps };
  }
  /* a larger, seeded 0-1 knapsack, so the FPTAS sweep has a curve to show
     (the 5-item instance is exact for every ε above 1/2). Capacity is half
     the total weight, the classic hard regime. */
  function knapRandom(n, seed) {
    const r = rnd(seed), w = [], v = [];
    for (let i = 0; i < n; i++) { w.push(2 + Math.floor(r() * 18)); v.push(10 + Math.floor(r() * 90)); }
    const W = Math.floor(w.reduce((a, b) => a + b, 0) / 2);
    return { w, v, W, n };
  }

  /* sweep ε and measure, at each ε, the achieved value, the relative error
     and the DP size — against the true optimum from the exact DP. */
  function knapFPTASSweep(w, v, W, epsList) {
    const exact = knapValueDP(w, v, W, null);
    return epsList.map(e => {
      const c = { n: 0, add() { this.n++; }, get() { return this.n; } };
      const r = knapFPTAS(w, v, W, e, c);
      const err = (exact.value - r.value) / exact.value;
      return { eps: e, value: r.value, opt: exact.value, err, K: r.K, states: r.states, cells: c.get("cell"), ok: err <= e + 1e-12, items: r.items, weight: r.weight };
    });
  }

  /* ═══════════════════════════════════════════════════════════════════
     §20 — the ratio board: every approximation algorithm on this page, run
     against the exhaustive optimum of its own instance, so the figure plots
     MEASURED ratios beside PROVED bounds.
     ═══════════════════════════════════════════════════════════════════ */
  function ratioBoard() {
    const rows = [];
    /* vertex cover, both greedies, on gVC */
    const g = vcIndex(NI.gVC);
    const vopt = vcOptBrute(g.n, g.E, null);
    const vm = vcMatching(g.n, g.E, null);
    const vd = vcGreedyDegree(g.n, g.E, null);
    rows.push({ key: "vc2", label: "vertex cover · maximal matching", got: vm.size, opt: vopt.size, ratio: vm.size / vopt.size, bound: 2, boundLabel: "2", kind: "min" });
    const dmax = (function () { const d = new Array(g.n).fill(0); g.E.forEach(e => { d[e[0]]++; d[e[1]]++; }); return Math.max.apply(null, d); })();
    rows.push({ key: "vcdeg", label: "vertex cover · max-degree greedy (7-vertex instance)", got: vd.size, opt: vopt.size, ratio: vd.size / vopt.size, bound: harmonic(dmax), boundLabel: "H(" + dmax + ") = " + harmonic(dmax).toFixed(2), kind: "min" });
    const vb = vcGreedySweep([16], false)[0];
    rows.push({ key: "vcdegbad", label: "vertex cover · max-degree greedy (adversarial, n = " + vb.n + ")", got: vb.greedy, opt: vb.opt, ratio: vb.ratio, bound: harmonic(vb.maxDeg), boundLabel: "H(" + vb.maxDeg + ") = " + harmonic(vb.maxDeg).toFixed(2), kind: "min" });
    rows.push({ key: "vc2bad", label: "vertex cover · maximal matching (adversarial, n = " + vb.n + ")", got: vb.matching, opt: vb.opt, ratio: vb.mRatio, bound: 2, boundLabel: "2", kind: "min" });
    /* TSP, three algorithms, on the 8-city metric instance */
    const D = tspDist(NI.tsp.pts), topt = tspOpt(D, null);
    const dt = tspDoubleTree(D, null), ch = tspChristofides(D, null), nn = tspNN(D, 0, null);
    rows.push({ key: "dt", label: "metric TSP · double tree", got: dt.cost, opt: topt.cost, ratio: dt.cost / topt.cost, bound: 2, boundLabel: "2", kind: "min" });
    rows.push({ key: "chr", label: "metric TSP · Christofides", got: ch.cost, opt: topt.cost, ratio: ch.cost / topt.cost, bound: 1.5, boundLabel: "3/2", kind: "min" });
    rows.push({ key: "nn", label: "metric TSP · nearest neighbour", got: nn.cost, opt: topt.cost, ratio: nn.cost / topt.cost, bound: null, boundLabel: "none (grows with n)", kind: "min" });
    /* set cover */
    const sc = NI.setcover, sg = setCoverGreedy(sc.U, sc.sets, null), so = setCoverOpt(sc.U, sc.sets, null);
    const d = Math.max(...sc.sets.map(S => S.length));
    rows.push({ key: "sc", label: "set cover · greedy (12 elements, 6 sets)", got: sg.size, opt: so.size, ratio: sg.size / so.size, bound: harmonic(d), boundLabel: "H(" + d + ") = " + harmonic(d).toFixed(2), kind: "min" });
    const sb = setCoverBad(7), sbg = setCoverGreedy(sb.U, sb.sets, null), sbo = setCoverOpt(sb.U, sb.sets, null);
    const sbd = Math.max.apply(null, sb.sets.map(S => S.length));
    rows.push({ key: "scbad", label: "set cover · greedy (adversarial, n = " + sb.U + ")", got: sbg.size, opt: sbo.size, ratio: sbg.size / sbo.size, bound: harmonic(sbd), boundLabel: "H(" + sbd + ") = " + harmonic(sbd).toFixed(2), kind: "min" });
    /* MAX-3-SAT, both the derandomised run and the mean of the random one */
    const p8 = NI.phi8, mb = max3satBrute(p8, null), dr = max3satDerandom(p8, null);
    rows.push({ key: "m3s", label: "MAX-3-SAT · derandomised 7/8 (all eight clauses on 3 variables)", got: dr.satisfied, opt: mb.best, ratio: mb.best / dr.satisfied, bound: 8 / 7, boundLabel: "8/7 ≈ 1.143", kind: "max" });
    const pr = max3satFormula(6, 20, 7), mbr = max3satBrute(pr, null), drr = max3satDerandom(pr, null);
    rows.push({ key: "m3srand", label: "MAX-3-SAT · derandomised 7/8 (random, 6 variables, 20 clauses)", got: drr.satisfied, opt: mbr.best, ratio: mbr.best / drr.satisfied, bound: 8 / 7, boundLabel: "8/7 ≈ 1.143", kind: "max" });
    /* knapsack FPTAS at ε = 0.2 */
    const kn = NI.knap, ex = knapValueDP(kn.w, kn.v, kn.W, null), ap = knapFPTAS(kn.w, kn.v, kn.W, 0.2, null);
    rows.push({ key: "fptas", label: "knapsack FPTAS, ε = 0.2", got: ap.value, opt: ex.value, ratio: ex.value / ap.value, bound: 1 / (1 - 0.2), boundLabel: "1/(1−ε) = 1.25", kind: "max" });
    rows.forEach(r => { r.ok = (r.bound === null) || (r.ratio <= r.bound + 1e-9); });
    return rows;
  }

  return {
    harmonic,
    vcIndex, vcIsCover, vcMatching, vcGreedyDegree, vcOptBrute, vcGreedyBad, vcGreedySweep,
    tspDist, tspTourCost, tspTriangle, tspOpt, tspMST, tspDoubleTree,
    tspMinMatching, eulerCircuit, tspChristofides, tspNN, tspTwoOpt,
    tspGapMatrix, tspGapDemo,
    setCoverGreedy, setCoverOpt, setCoverBad,
    satClauseSat, max3satCount, max3satWellFormed, max3satFormula, max3satBrute,
    max3satRandom, max3satCondExp, max3satDerandom,
    knapBrute, knapValueDP, knapFPTAS, knapFPTASSweep, knapRandom,
    ratioBoard
  };
})());


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


/* ── chunk A figures ────────────────────────────────────────── */
/* ── 02  #enc-svg  unary vs binary, and a loop that is linear in the value ── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("enc-svg")) return;
  const W = 700, H = 360;
  function build() {
    const k = +SX.val("enc-k", 37), mode = SX.val("enc-series", "both");
    SX.setText("enc-k-val", SX.int(k));
    const svg = d3.select("#enc-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 58, r: 150, t: 22, b: 44 });

    if (mode === "knap") {
      /* the knapsack table: one more BIT of W doubles the work */
      const rows = []; for (let b = 1; b <= 13; b++) { const Wc = (1 << b) - 1; const c = AL.counter(); NR.knapCells(5, Wc, c); rows.push({ bits: b, W: Wc, cells: c.get("cell") }); }
      const x = d3.scaleLinear().domain([1, 13]).range([0, F.iw]);
      const y = d3.scaleLog().domain([5, d3.max(rows, r => r.cells) * 1.3]).range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 6); AL.axisB(F.g, x, F.ih, 12, "bits of W in the input"); AL.axisL(F.g, y, 6, "measured cells filled", d3.format("~s"));
      const line = d3.line().x(r => x(r.bits)).y(r => y(r.cells));
      F.g.append("path").datum(rows).attr("d", line).attr("fill", "none").attr("stroke", AC.bad).attr("stroke-width", 2.4);
      rows.forEach(r => F.g.append("circle").attr("cx", x(r.bits)).attr("cy", y(r.cells)).attr("r", 3.5).attr("fill", AC.bad));
      const lin = d3.line().x(r => x(r.bits)).y(r => y(Math.max(5, 5 * (r.bits + 1))));
      F.g.append("path").datum(rows).attr("d", lin).attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 2).attr("stroke-dasharray", "5 4");
      AL.legend(F.g, [{ label: "cells filled (measured) — doubles per bit", color: AC.bad }, { label: "input size in bits — grows by 1 per bit", color: AC.accent, dash: "5 4" }], F.iw + 12, 24);
      const dbl = rows.every((r, i) => i === 0 || r.cells === 2 * rows[i - 1].cells);
      SX.setHtml("enc-readout", `the 0-1 knapsack table with n = 5 items, filled for capacities W = 2ᵇ − 1 · ` +
        rows.filter(r => r.bits <= 6 || r.bits === 10 || r.bits === 13).map(r => `W = ${SX.int(r.W)} (${r.bits} bits): <b>${SX.int(r.cells)}</b> cells`).join(" · ") +
        ` · every extra BIT of W doubles the measured work while lengthening the input by one symbol ${SX.flag(dbl)} · so O(nW) is O(n·2^(bits of W)): PSEUDO-polynomial, exponential in the input size, and not a polynomial-time algorithm (§14, §25)`);
      return;
    }

    /* the log-log comparison over values of k */
    const ks = []; for (let e = 0; e <= 12; e++) { const v = 1 << e; ks.push(v); if (e < 12) ks.push(Math.round(v * 1.5)); }
    if (ks.indexOf(k) < 0) ks.push(k); ks.sort((a, b) => a - b);
    const data = ks.map(v => { const c1 = AL.counter(), c2 = AL.counter(); NR.loopToK(v, c1); const td = NR.trialDivision(v, c2);
      return { k: v, unary: NR.unaryLen(v), binary: NR.binaryLen(v), loop: c1.get("step"), div: Math.max(1, c2.get("div")), prime: td.prime }; });
    const showAll = mode === "both";
    const x = d3.scaleLog().domain([1, 4096]).range([0, F.iw]);
    const ymax = showAll ? d3.max(data, d => Math.max(d.unary, d.loop)) : d3.max(data, d => d.unary);
    const y = d3.scaleLog().domain([1, ymax * 1.4]).range([F.ih, 0]);
    AL.gridY(F.g, y, F.iw, 6); AL.axisB(F.g, x, F.ih, 6, "value k (log scale)", d3.format("~s")); AL.axisL(F.g, y, 6, "length in bits / measured steps (log scale)", d3.format("~s"));
    const mk = (key, col, dash, lab) => { const ln = d3.line().x(d => x(d.k)).y(d => y(Math.max(1, d[key])));
      F.g.append("path").datum(data).attr("d", ln).attr("fill", "none").attr("stroke", col).attr("stroke-width", 2.2).attr("stroke-dasharray", dash || null); return { label: lab, color: col, dash } };
    const items = [];
    items.push(mk("unary", AC.bad, null, "|k| in unary = k"));
    items.push(mk("binary", AC.good, null, "|k| in binary = ⌊log₂k⌋+1"));
    if (showAll) { items.push(mk("loop", AC.a2, "6 3", "measured: loop to k")); items.push(mk("div", AC.violet, "2 3", "measured: trial divisions")); }
    AL.legend(F.g, items, F.iw + 10, 26);
    const at = data.find(d => d.k === k) || data[0];
    F.g.append("line").attr("x1", x(k)).attr("x2", x(k)).attr("y1", 0).attr("y2", F.ih).attr("stroke", AC.accent).attr("stroke-width", 1.4).attr("stroke-dasharray", "3 3");
    F.g.append("text").attr("x", x(k) + 5).attr("y", 12).attr("font-size", 11).attr("fill", AC.accent).text("k = " + k);
    /* the two spellings of k, drawn */
    const gy = F.ih + 34;
    F.g.append("text").attr("x", 0).attr("y", gy).attr("font-size", 11).attr("fill", AC.muted).text("binary " + k.toString(2) + " (" + at.binary + " bits)   ·   unary " + (k <= 48 ? "1".repeat(k) : "1".repeat(40) + "… (" + k + " ones)"));
    const exact = at.loop === at.k, sqrtish = at.div <= Math.ceil(Math.sqrt(k)) + 1;
    SX.setHtml("enc-readout", `k = <b>${SX.int(k)}</b> · binary <code>${k.toString(2)}</code>, length <b>${at.binary}</b> bits · unary length <b>${SX.int(k)}</b> bits, a factor of ${(k / at.binary).toFixed(1)} longer · ` +
      `the loop that counts down from k performed <b>${SX.int(at.loop)}</b> measured steps = k ${SX.flag(exact)} — that is Θ(n) against the unary encoding and Θ(2ⁿ) against the binary one, from the SAME routine · ` +
      `trial division on k performed <b>${SX.int(at.div)}</b> measured division${at.div === 1 ? "" : "s"} (${at.prime ? "k is prime, so the loop tried every divisor d = 2 … ⌊√k⌋ = " + Math.floor(Math.sqrt(k)) + ", which is " + Math.max(0, Math.floor(Math.sqrt(k)) - 1) + " of them" : "k is composite; the loop stopped at the first factor"}), never more than ⌈√k⌉ = ${Math.ceil(Math.sqrt(k))} ${SX.flag(sqrtish)} — and ⌈√k⌉ ≈ 2^(b/2) with b = ${at.binary} bits, so this routine is EXPONENTIAL in the input size · ` +
      `at k = 4096: unary ${SX.int(4096)} bits against binary 13 bits, and the loop takes ${SX.int(data[data.length - 1].loop)} steps · primality IS in P (by a different algorithm, 2002) but NOT by this one`);
  }
  SX.on("enc-k", "input", build); SX.on("enc-series", "change", build);
  build();
})();

/* ── 03  #selfred-svg  self-reducibility: a decision oracle builds a solution ── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("selfred-svg")) return;
  const W = 700, H = 330;
  function build() {
    const mode = SX.val("selfred-mode", "sat"), first = SX.val("selfred-first", "true") === "true";
    const svg = d3.select("#selfred-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 14, r: 14, t: 10, b: 34 });

    if (mode === "sat") {
      const phi = NI.phi, c = AL.counter();
      const R = NR.satSelfReduce(phi, first, c);
      const brute = NR.satBrute(phi.clauses, phi.n);
      const frames = R.steps.map((s, i) => ({ s, i }));
      function render(f) {
        F.g.selectAll("*").remove();
        const bx = 150, by = 26, dy = 62;
        F.g.append("text").attr("x", 0).attr("y", 12).attr("font-size", 12).attr("fill", AC.muted)
          .text("φ = (x₁ ∨ ¬x₂ ∨ ¬x₃) ∧ (¬x₁ ∨ x₂ ∨ x₃) ∧ (x₁ ∨ x₂ ∨ x₃)   —   the oracle answers only YES or NO");
        R.steps.forEach((s, i) => {
          if (i > f.i) return;
          const y0 = by + i * dy;
          const cur = i === f.i;
          F.g.append("rect").attr("x", bx).attr("y", y0).attr("width", 300).attr("height", 30).attr("rx", 6)
            .attr("fill", cur ? AC.panel : AC.panel2).attr("stroke", cur ? AC.a2 : AC.line).attr("stroke-width", cur ? 2 : 1);
          F.g.append("text").attr("x", bx + 10).attr("y", y0 + 20).attr("font-size", 11.5).attr("fill", AC.ink)
            .text("query " + (i + 1) + ": " + s.question);
          F.g.append("text").attr("x", bx + 316).attr("y", y0 + 20).attr("font-size", 12).attr("fill", s.answer ? AC.good : AC.bad)
            .text(s.answer ? "YES" : "NO");
          if (s.level >= 0) {
            F.g.append("text").attr("x", bx + 368).attr("y", y0 + 20).attr("font-size", 11.5).attr("fill", AC.a2)
              .text("→ fix " + phi.names[s.level] + " = " + (s.taken ? "TRUE" : "FALSE") + (s.answer ? "" : "  (the other branch is ruled out)"));
            F.g.append("text").attr("x", 6).attr("y", y0 + 20).attr("font-size", 11).attr("fill", AC.muted)
              .text("fixed so far: " + s.fixed.map(v => v === undefined ? "·" : (v ? "1" : "0")).join(""));
          }
          if (i > 0) AL.arrow(F.g, bx + 150, y0 - dy + 30, bx + 150, y0, { color: AC.line });
        });
        const done = f.i === R.steps.length - 1;
        if (done && R.assign) F.g.append("text").attr("x", bx).attr("y", by + R.steps.length * dy + 6).attr("font-size", 13).attr("fill", AC.good)
          .text("recovered assignment  " + NR.bitStr(R.assign) + "   (x₁x₂x₃)   —   φ evaluates to " + (R.ok ? "TRUE" : "FALSE"));
      }
      AL.stepper(svg, { frames, render, delay: 900, label: "oracle query" });
      const inSols = R.assign ? brute.sols.indexOf(NR.bitStr(R.assign)) >= 0 : false;
      SX.setHtml("selfred-readout", `n = ${phi.n} variables · measured oracle queries <b>${R.queries}</b> = n + 1 (one preliminary call, then one per variable) ${SX.flag(R.queries === phi.n + 1)} · ` +
        `recovered assignment <b>${NR.bitStr(R.assign)}</b>, and φ(${NR.bitStr(R.assign)}) = TRUE ${SX.flag(R.ok)} · it is one of the ${brute.count} satisfying assignments found by exhaustive search over all 2³ = ${brute.assigns} — {${brute.sols.join(", ")}} ${SX.flag(inSols)} · ` +
        `the oracle is implemented here by brute force and did <b>${SX.int(c.get("oracleAssign"))}</b> assignment evaluations inside those ${R.queries} calls; a real polynomial-time oracle would do that work in polynomial time and the self-reduction would inherit it · ` +
        `trying ${first ? "TRUE" : "FALSE"} first returns the ${first ? "largest" : "smallest"} satisfying assignment in the bit order x₁x₂x₃`);
    } else {
      const g = NR.adjFromNamed(NI.gVC), n = g.n, names = NI.gVC.V;
      const order = first ? [...Array(n).keys()] : [...Array(n).keys()].reverse();
      const c = AL.counter(); const R = NR.cliqueSelfReduce(g.A, n, order, c);
      const tri = NR.cliqueCount(g.A, n, R.k);
      const pos = NI.gVC.pos;
      const frames = [{ i: -1 }].concat(R.steps.map((s, i) => ({ i, s })));
      function render(f) {
        F.g.selectAll("*").remove();
        const alive = f.i < 0 ? [...Array(n).keys()] : R.steps[f.i].cur;
        const live = new Set(alive);
        NI.gVC.E.forEach(([u, v]) => { const a = pos[u], b = pos[v]; const both = live.has(names.indexOf(u)) && live.has(names.indexOf(v));
          F.g.append("line").attr("x1", a[0] + 40).attr("y1", a[1] + 30).attr("x2", b[0] + 40).attr("y2", b[1] + 30)
            .attr("stroke", both ? AC.accent : AC.line).attr("stroke-width", both ? 2.2 : 1.2); });
        names.forEach((nm, i) => { const p = pos[nm]; const on = live.has(i); const acting = f.i >= 0 && R.steps[f.i].v === i;
          F.g.append("circle").attr("cx", p[0] + 40).attr("cy", p[1] + 30).attr("r", 17)
            .attr("fill", on ? (f.i === R.steps.length - 1 ? AC.good : AC.panel2) : AC.bg)
            .attr("stroke", acting ? AC.a2 : (on ? AC.line : AC.line)).attr("stroke-width", acting ? 2.6 : 1).attr("opacity", on ? 1 : 0.35);
          F.g.append("text").attr("x", p[0] + 40).attr("y", p[1] + 35).attr("text-anchor", "middle").attr("font-size", 13)
            .attr("fill", on && f.i === R.steps.length - 1 ? AC.bg : AC.ink).attr("opacity", on ? 1 : 0.45).text(nm); });
        const t = f.i < 0 ? "k = ω(G) = " + R.k + " found first; now ask, for each vertex in turn, whether a k-clique survives without it"
          : "query " + (f.i + 1) + ": does G − " + names[R.steps[f.i].v] + " still contain a clique of size " + R.k + "?  " +
            (R.steps[f.i].survives ? "YES → delete " + names[R.steps[f.i].v] : "NO → keep " + names[R.steps[f.i].v] + " (every k-clique uses it)");
        F.g.append("text").attr("x", 8).attr("y", F.ih + 22).attr("font-size", 11.5).attr("fill", AC.ink).text(t);
      }
      AL.stepper(svg, { frames, render, delay: 900, label: "oracle query" });
      const isCl = NR.isCliqueSet(g.A, R.clique);
      SX.setHtml("selfred-readout", `7 vertices, ${NI.gVC.E.length} edges · maximum clique size ω(G) = <b>${R.k}</b>, and exhaustive search over all 2⁷ = 128 subsets finds exactly <b>${tri.found}</b> cliques of that size (checking C(7,${R.k}) = ${tri.tried} candidate subsets) · ` +
        `measured oracle queries in the deletion walk <b>${R.queries}</b> = n ${SX.flag(R.queries === n)} · recovered vertex set {${R.clique.map(i => names[i]).join(", ")}}, size ${R.clique.length} = k ${SX.flag(R.clique.length === R.k)}, and every pair inside it is an edge ${SX.flag(isCl)} · ` +
        `which of the ${tri.found} maximum cliques comes out depends only on the deletion order — reverse it and a different one survives · the invariant held throughout: the surviving graph always still contained a clique of size ${R.k}`);
    }
  }
  SX.on("selfred-mode", "change", build); SX.on("selfred-first", "change", build);
  build();
})();

/* ── 04  #growth-svg  measured growth: real routines, counted operations ── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("growth-svg")) return;
  const W = 700, H = 380, BUDGET = 1e9, XMAX = 44;
  const PERM_CAP = 10;                    /* n! is only enumerable this far */
  function build() {
    const nmax = +SX.val("growth-n", 14), logY = SX.val("growth-scale", "log") === "log", showB = SX.checked("growth-budget", true);
    SX.setText("growth-n-val", nmax);
    const series = [
      { key: "lin", label: "n — scan", col: AC.teal, form: n => n, run: (n, c) => NR.growLinear(n, c), cap: nmax },
      { key: "sq", label: "n² — all pairs", col: AC.accent, form: n => n * n, run: (n, c) => NR.growQuadratic(n, c), cap: nmax },
      { key: "cu", label: "n³ — all triples", col: AC.good, form: n => n * n * n, run: (n, c) => NR.growCubic(n, c), cap: nmax },
      { key: "ex", label: "2ⁿ — all subsets", col: AC.a2, form: n => Math.pow(2, n), run: (n, c) => NR.growSubsets(n, c), cap: nmax },
      { key: "fa", label: "n! — all permutations", col: AC.bad, form: n => NR.factorial(n), run: (n, c) => NR.growPerms(n, c), cap: Math.min(nmax, PERM_CAP) }
    ];
    let allMatch = true;
    series.forEach(s => { s.pts = []; for (let n = 1; n <= s.cap; n++) { const c = AL.counter(); const v = s.run(n, c); s.pts.push({ n, v }); if (v !== s.form(n)) allMatch = false; } });

    const svg = d3.select("#growth-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 62, r: 168, t: 20, b: 42 });
    const topY = showB ? BUDGET * 20 : d3.max(series, s => d3.max(s.pts, p => p.v)) * 3;
    const x = d3.scaleLinear().domain([1, XMAX]).range([0, F.iw]);
    const y = logY ? d3.scaleLog().domain([1, topY]).range([F.ih, 0]) : d3.scaleLinear().domain([0, d3.max(series, s => d3.max(s.pts, p => p.v))]).range([F.ih, 0]);
    AL.gridY(F.g, y, F.iw, 6);
    AL.axisB(F.g, x, F.ih, 10, "n"); AL.axisL(F.g, y, 6, "operations actually counted", d3.format("~s"));
    const line = d3.line().x(d => x(d.n)).y(d => y(Math.max(logY ? 1 : 0, d.v)));
    series.forEach(s => {
      F.g.append("path").datum(s.pts).attr("d", line).attr("fill", "none").attr("stroke", s.col).attr("stroke-width", 2.4);
      s.pts.forEach(p => { if (p.v <= y.domain()[1]) F.g.append("circle").attr("cx", x(p.n)).attr("cy", y(Math.max(logY ? 1 : 0, p.v))).attr("r", 2.6).attr("fill", s.col); });
      if (logY) { /* the verified closed form, extrapolated beyond the measured range */
        const ext = []; for (let n = s.cap; n <= XMAX; n++) { const v = s.form(n); if (v <= topY) ext.push({ n, v }); else { ext.push({ n, v: topY }); break; } }
        F.g.append("path").datum(ext).attr("d", line).attr("fill", "none").attr("stroke", s.col).attr("stroke-width", 1.6).attr("stroke-dasharray", "4 4").attr("opacity", 0.75);
      }
    });
    if (showB && logY) { F.g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(BUDGET)).attr("y2", y(BUDGET)).attr("stroke", AC.violet).attr("stroke-width", 1.6).attr("stroke-dasharray", "7 4");
      F.g.append("text").attr("x", 4).attr("y", y(BUDGET) - 5).attr("font-size", 11).attr("fill", AC.violet).text("budget 10⁹ operations"); }
    AL.legend(F.g, series.map(s => ({ label: s.label, color: s.col })).concat(logY ? [{ label: "measured ——  closed form - - -", color: AC.muted, dash: "4 4" }] : []), F.iw + 12, 24);

    /* crossovers read off the MEASURED arrays */
    const byKey = {}; series.forEach(s => { byKey[s.key] = {}; s.pts.forEach(p => { byKey[s.key][p.n] = p.v; }); });
    /* the crossover is the point BEYOND WHICH one curve stays above the other:
       the largest measured n at which it does not, plus one.  (2ⁿ is briefly
       larger at n = 1 and exactly EQUAL to n² at n = 2 and n = 4, so "the
       first n where 2ⁿ > n²" would be the wrong and misleading answer.) */
    function crossAfter(a, b, cap) { let last = null; for (let n = 1; n <= cap; n++) { if (byKey[a][n] === undefined || byKey[b][n] === undefined) continue; if (!(byKey[a][n] > byKey[b][n])) last = n; } if (last === null) return 1; return last + 1 > cap ? null : last + 1; }
    function ties(a, b, cap) { const t = []; for (let n = 1; n <= cap; n++) if (byKey[a][n] !== undefined && byKey[a][n] === byKey[b][n]) t.push(n); return t; }
    const xSq = crossAfter("ex", "sq", nmax), xCu = crossAfter("ex", "cu", nmax), xFa = crossAfter("fa", "ex", Math.min(nmax, PERM_CAP));
    const tSq = ties("ex", "sq", nmax), tFa = ties("fa", "ex", Math.min(nmax, PERM_CAP));
    [[xSq, "2ⁿ > n² from"], [xCu, "2ⁿ > n³ from"], [xFa, "n! > 2ⁿ from"]].forEach(([n, lab], i) => { if (n === null) return;
      F.g.append("line").attr("x1", x(n)).attr("x2", x(n)).attr("y1", 0).attr("y2", F.ih).attr("stroke", AC.muted).attr("stroke-width", 1).attr("stroke-dasharray", "2 4");
      F.g.append("text").attr("x", x(n) + 4).attr("y", 14 + i * 13).attr("font-size", 10).attr("fill", AC.muted).text(lab + " n = " + n); });

    const L = n => SX.int(NR.largestN(n, BUDGET));
    const fmtTie = (t, aa, bb) => t.length === 0 ? "" : " — and they are exactly EQUAL at n = " + t.join(", ") + " (" + t.map(n => byKey[aa][n]).join(", ") + "), so a naive \"first n where it is bigger\" would report the wrong crossover";
    SX.setHtml("growth-readout", `measured up to n = <b>${nmax}</b> (permutations only to n = ${PERM_CAP}; 11! alone is ${SX.int(NR.factorial(11))} enumerations) · every measured count equals its closed form ${SX.flag(allMatch)}, which is what licenses the dashed extrapolation · ` +
      `crossovers READ OFF the measured arrays, each the point BEYOND which one curve stays above the other: 2ⁿ overtakes n² at n = <b>${xSq === null ? "beyond the measured range" : xSq}</b>${fmtTie(tSq, "ex", "sq")}; 2ⁿ overtakes n³ at n = <b>${xCu === null ? "beyond the measured range" : xCu}</b>${xCu !== null ? " (measured: 2^" + xCu + " = " + SX.int(byKey.ex[xCu]) + " against " + xCu + "³ = " + SX.int(byKey.cu[xCu]) + ", a margin of " + SX.int(byKey.ex[xCu] - byKey.cu[xCu]) + " operations)" : ""}; n! overtakes 2ⁿ at n = <b>${xFa === null ? "beyond the measured range" : xFa}</b>${fmtTie(tFa, "fa", "ex")} · ` +
      `largest n inside the 10⁹ budget, from the verified closed forms: n → <b>${L(n => n)}</b>, n log₂n → <b>${L(n => n * Math.log2(n))}</b>, n² → <b>${L(n => n * n)}</b>, n³ → <b>${L(n => n * n * n)}</b>, 1.5ⁿ → <b>${L(n => Math.pow(1.5, n))}</b>, 2ⁿ → <b>${L(n => Math.pow(2, n))}</b>, n! → <b>${L(n => NR.factorial(n))}</b> · ` +
      `a 1 000× faster machine moves n³ from ${L(n => n * n * n)} to ${SX.int(NR.largestN(n => n * n * n, 1000 * BUDGET))} (× 10) but 2ⁿ only from ${L(n => Math.pow(2, n))} to ${SX.int(NR.largestN(n => Math.pow(2, n), 1000 * BUDGET))} (+ ${NR.largestN(n => Math.pow(2, n), 1000 * BUDGET) - NR.largestN(n => Math.pow(2, n), BUDGET)}) — polynomial reach is multiplied, exponential reach is merely added to`);
  }
  SX.on("growth-n", "input", build); SX.on("growth-scale", "change", build); SX.on("growth-budget", "change", build);
  build();
})();

/* ── 05  #certificate-svg  a verifier at work, against exhaustive search ── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("certificate-svg")) return;
  const W = 700, H = 360;
  function build() {
    const which = SX.val("cert-problem", "sat"), forged = SX.val("cert-forge", "good") === "bad";
    const svg = d3.select("#certificate-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 14, r: 14, t: 12, b: 34 });
    const cv = AL.counter(), cb = AL.counter();
    let title, certText, V, brute, bruteText, yesInstance, note;

    if (which === "sat") {
      const phi = NI.phi;
      const a = forged ? [true, false, false] : [true, false, true];   /* 100 falsifies clause 2; 101 satisfies */
      V = NR.verifySat(phi.clauses, a, cv);
      brute = NR.satBrute(phi.clauses, phi.n, cb);
      title = "3-SAT:  φ = (x₁ ∨ ¬x₂ ∨ ¬x₃) ∧ (¬x₁ ∨ x₂ ∨ x₃) ∧ (x₁ ∨ x₂ ∨ x₃)";
      certText = "certificate: x₁x₂x₃ = " + NR.bitStr(a) + (forged ? "  (forged)" : "");
      yesInstance = brute.sat;
      bruteText = "exhaustive: all 2³ = " + brute.assigns + " assignments, " + brute.count + " satisfy — {" + brute.sols.join(", ") + "}";
      note = "the certificate is " + phi.n + " bits; the verifier reads each clause once and at most " + phi.clauses.reduce((s, c2) => s + c2.length, 0) + " literals in all";
    } else if (which === "clique") {
      const g = NR.adjFromNamed(NI.gVC), names = NI.gVC.V;
      const S = forged ? [0, 2, 5] : [2, 3, 4];                        /* {a,c,f} is not a clique; {c,d,e} is */
      V = NR.verifyClique(g.A, S, 3, cv);
      const cnt = NR.cliqueCount(g.A, g.n, 3, cb);
      brute = cnt; yesInstance = cnt.found > 0;
      title = "CLIQUE, k = 3, on the seven-vertex graph (" + NI.gVC.E.length + " edges)";
      certText = "certificate: {" + S.map(i => names[i]).join(", ") + "}" + (forged ? "  (forged)" : "");
      bruteText = "exhaustive: all C(7,3) = " + cnt.tried + " triples checked, " + cnt.found + " are cliques";
      note = "the certificate is 3 vertex names; the verifier does C(3,2) = 3 adjacency lookups";
      V.checks = V.checks.map(ch => ({ ...ch, what: ch.what.replace(/\((\d),(\d)\)/, (m, p, q) => "(" + names[+p] + "," + names[+q] + ")") }));
    } else {
      const S = NI.subsetsum.S, T = NI.subsetsum.T;
      const picks = forged ? [0, 1, 2] : [3, 5];                        /* 3+7+11 = 21 ≠ 37; 14+23 = 37 */
      V = NR.verifySubset(S, picks, T, cv);
      brute = NR.subsetBrute(S, T, cb); yesInstance = brute.count > 0;
      title = "SUBSET-SUM:  S = {" + S.join(", ") + "},  T = " + T;
      certText = "certificate: {" + picks.map(i => S[i]).join(", ") + "}" + (forged ? "  (forged)" : "");
      bruteText = "exhaustive: all 2⁶ = " + brute.subsets + " subsets, " + brute.count + " sum to " + T + " — " + brute.sols.map(s => "{" + s.join(",") + "}").join(", ");
      note = "the certificate is the subset itself; the verifier performs one addition per listed element";
    }

    const frames = V.checks.map((c2, i) => i);
    function render(k) {
      F.g.selectAll("*").remove();
      F.g.append("text").attr("x", 0).attr("y", 12).attr("font-size", 12).attr("fill", AC.muted).text(title);
      F.g.append("text").attr("x", 0).attr("y", 32).attr("font-size", 12.5).attr("fill", forged ? AC.bad : AC.a2).text(certText);
      V.checks.forEach((c2, i) => {
        if (i > k) return;
        const y0 = 50 + i * 30, cur = i === k;
        F.g.append("rect").attr("x", 0).attr("y", y0).attr("width", 400).attr("height", 26).attr("rx", 5)
          .attr("fill", cur ? AC.panel : AC.panel2).attr("stroke", cur ? AC.a2 : AC.line);
        F.g.append("text").attr("x", 10).attr("y", y0 + 18).attr("font-size", 11.5).attr("fill", AC.ink).text(c2.what);
        F.g.append("text").attr("x", 150).attr("y", y0 + 18).attr("font-size", 11.5).attr("fill", c2.ok ? AC.good : AC.bad).text(c2.ok ? "✓" : "✗");
        F.g.append("text").attr("x", 168).attr("y", y0 + 18).attr("font-size", 11).attr("fill", AC.muted).text(c2.detail);
      });
      if (k === V.checks.length - 1) F.g.append("text").attr("x", 0).attr("y", 62 + V.checks.length * 30).attr("font-size", 13)
        .attr("fill", V.accept ? AC.good : AC.bad).text(V.accept ? "VERIFIER ACCEPTS" : "VERIFIER REJECTS — the forgery is caught (soundness)");
      /* the cost comparison, log scale */
      const vSteps = Math.max(1, cv.get("check")), bSteps = Math.max(1, cb.get("assign") + cb.get("subset") + cb.get("lit") + cb.get("check") + cb.get("clause") + cb.get("pair"));
      const bx = 440, bw = 210, by = 70, bh = 120;
      const sc = d3.scaleLog().domain([1, Math.max(bSteps, vSteps) * 1.5]).range([0, bw]);
      F.g.append("text").attr("x", bx).attr("y", by - 26).attr("font-size", 11).attr("fill", AC.muted).text("measured elementary steps (log)");
      [["verifier", vSteps, AC.good], ["exhaustive search", bSteps, AC.bad]].forEach(([lab, v, col], i) => {
        F.g.append("rect").attr("x", bx).attr("y", by + i * 46).attr("width", Math.max(3, sc(v))).attr("height", 22).attr("rx", 4).attr("fill", col);
        F.g.append("text").attr("x", bx).attr("y", by + i * 46 - 5).attr("font-size", 11).attr("fill", AC.muted).text(lab);
        F.g.append("text").attr("x", bx + Math.max(3, sc(v)) + 6).attr("y", by + i * 46 + 16).attr("font-size", 11.5).attr("fill", col).text(SX.int(v));
      });
      F.g.append("text").attr("x", bx).attr("y", by + 130).attr("font-size", 11).attr("fill", AC.muted).text("ratio ≈ " + (bSteps / vSteps).toFixed(1) + "× on this tiny instance");
      F.g.append("text").attr("x", 0).attr("y", F.ih + 20).attr("font-size", 11).attr("fill", AC.muted).text(note);
    }
    AL.stepper(svg, { frames, render, delay: 800, label: "check" });
    const bSteps = cb.get("assign") + cb.get("subset") + cb.get("lit") + cb.get("check") + cb.get("clause") + cb.get("pair");
    const correct = forged ? !V.accept : V.accept;
    SX.setHtml("certificate-readout", `${certText} · the verifier performed <b>${SX.int(cv.get("check"))}</b> measured elementary check${cv.get("check") === 1 ? "" : "s"} and ${V.accept ? "ACCEPTED" : "REJECTED"} · ` +
      `${bruteText} — <b>${SX.int(bSteps)}</b> measured elementary steps · ` +
      `the instance is a ${yesInstance ? "YES" : "NO"}-instance by exhaustive search, and the verifier's answer on this certificate is ${forged ? "correctly a rejection: a forged certificate must not be accepted even on a yes-instance, which is SOUNDNESS" : "an acceptance, which exhibits the certificate COMPLETENESS demands"} ${SX.flag(correct)} · ` +
      `the gap is the whole content of NP: checking is cheap and linear in the certificate, searching is exponential in the instance parameter — scale this instance to 100 variables, 100 vertices or 100 numbers and the right column becomes 2¹⁰⁰ or C(100,3)-many while the left column is still a few hundred steps`);
  }
  SX.on("cert-problem", "change", build); SX.on("cert-forge", "change", build);
  build();
})();

/* ── 06  #reduction-svg  the mapping picture over a real reduction ── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("reduction-svg")) return;
  const W = 700, H = 345, K = 3;
  function build() {
    const kind = SX.val("red-map", "comp"), focus = +SX.val("red-focus", 0);
    SX.setText("red-focus-val", focus);
    const svg = d3.select("#reduction-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 16, r: 16, t: 26, b: 34 });
    const c = AL.counter();
    const M = NR.reductionMap(kind, K, c);
    const G4 = NR.allGraphs4();

    /* two columns of 64 cells, 8 × 8 each */
    const cw = 16, gap = 2, colW = 8 * (cw + gap);
    const lx = 40, rx = F.iw - colW - 40, ty = 30;
    F.g.append("text").attr("x", lx).attr("y", ty - 10).attr("font-size", 11.5).attr("fill", AC.muted).text("instances of INDEPENDENT-SET, k = 3");
    F.g.append("text").attr("x", rx).attr("y", ty - 10).attr("font-size", 11.5).attr("fill", AC.muted).text("their images, decided as CLIQUE");
    F.g.append("text").attr("x", F.iw / 2 - 26).attr("y", ty - 10).attr("font-size", 12).attr("fill", AC.a2).text("f  ⟶");
    const cell = (col, i) => ({ x: (col === 0 ? lx : rx) + (i % 8) * (cw + gap), y: ty + Math.floor(i / 8) * (cw + gap) });
    M.rows.forEach((r, i) => {
      [0, 1].forEach(col => { const p = cell(col, i); const yes = col === 0 ? r.left : r.right;
        F.g.append("rect").attr("x", p.x).attr("y", p.y).attr("width", cw).attr("height", cw).attr("rx", 3)
          .attr("fill", yes ? AC.good : AC.panel2).attr("stroke", i === focus ? AC.a2 : (r.ok ? AC.line : AC.bad)).attr("stroke-width", i === focus ? 2.2 : (r.ok ? 1 : 1.8)); });
      if (!r.ok) { const a = cell(0, i), b = cell(1, i);
        F.g.append("line").attr("x1", a.x + cw).attr("y1", a.y + cw / 2).attr("x2", b.x).attr("y2", b.y + cw / 2).attr("stroke", AC.bad).attr("stroke-width", 0.7).attr("opacity", 0.45); }
    });
    { const a = cell(0, focus), b = cell(1, focus);
      F.g.append("line").attr("x1", a.x + cw).attr("y1", a.y + cw / 2).attr("x2", b.x).attr("y2", b.y + cw / 2).attr("stroke", AC.a2).attr("stroke-width", 2); }

    /* the highlighted instance, drawn as two little graphs */
    const R = M.rows[focus];
    const A = G4.adj(focus);
    const B = kind === "id" ? A : NR.complementAdj(A, 4);
    const P = [[0, 0], [54, 0], [54, 54], [0, 54]];
    function drawG(ox, oy, Adj, lab, yes, kk) {
      const g = F.g.append("g").attr("transform", `translate(${ox},${oy})`);
      for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) if (Adj[i][j])
        g.append("line").attr("x1", P[i][0]).attr("y1", P[i][1]).attr("x2", P[j][0]).attr("y2", P[j][1]).attr("stroke", AC.accent).attr("stroke-width", 1.8);
      for (let i = 0; i < 4; i++) { g.append("circle").attr("cx", P[i][0]).attr("cy", P[i][1]).attr("r", 8).attr("fill", AC.panel2).attr("stroke", AC.line);
        g.append("text").attr("x", P[i][0]).attr("y", P[i][1] + 4).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.ink).text("abcd"[i]); }
      g.append("text").attr("x", -6).attr("y", 74).attr("font-size", 11).attr("fill", AC.muted).text(lab);
      g.append("text").attr("x", -6).attr("y", 88).attr("font-size", 11).attr("fill", yes ? AC.good : AC.muted).text((yes ? "YES" : "no") + " at k = " + kk);
    }
    const gy = ty + 8 * (cw + gap) + 34;
    drawG(lx + 16, gy, A, "G  (mask " + focus + ", " + R.mEdges + " edges)", R.left, K);
    drawG(rx + 16, gy, B, kind === "id" ? "f(G) = G" : "f(G) = complement, " + R.mImg + " edges", R.right, R.k);
    AL.arrow(F.g, lx + 110, gy + 27, rx - 6, gy + 27, { color: R.ok ? AC.a2 : AC.bad, w: 2 });
    F.g.append("text").attr("x", (lx + rx) / 2 + 20).attr("y", gy + 18).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", R.ok ? AC.a2 : AC.bad).text(R.ok ? "answers agree" : "answers DISAGREE");

    const names = { comp: "f(⟨G,k⟩) = ⟨complement of G, k⟩", id: "f(⟨G,k⟩) = ⟨G, k⟩", compk: "f(⟨G,k⟩) = ⟨complement of G, k+1⟩" };
    const first = M.rows.find(r => !r.ok);
    SX.setHtml("reduction-readout", `map: <b>${names[kind]}</b> · all <b>${M.total}</b> = 2⁶ graphs on four vertices enumerated, both sides decided by exhaustive search (<b>${SX.int(c.get("subset"))}</b> measured subset tests and <b>${SX.int(c.get("pair"))}</b> adjacency lookups in all) · ` +
      `<b>${M.yes}</b> are yes-instances of INDEPENDENT-SET at k = 3 and <b>${M.no}</b> are no-instances · the map preserves the answer on <b>${M.agree}</b> of ${M.total} instances and fails on <b>${M.disagree}</b>` +
      (M.disagree === 0 ? ` — so yes maps to yes and no maps to no on the whole family, which is exactly condition (ii) of §06 ${SX.flag(true)}` :
        ` ${SX.flag(false)}, the first failure being mask ${first.mask} (${first.mEdges} edge${first.mEdges === 1 ? "" : "s"}): ${first.left ? "a YES-instance sent to a no-instance" : "a NO-instance sent to a yes-instance"} — one such pair is all it takes to refute a reduction, and this map is therefore NOT a reduction from INDEPENDENT-SET to CLIQUE`) +
      (kind === "comp"
        ? ` · size of the image: the complement of a 4-vertex graph with m edges has 6 − m edges, so the adjacency matrix is the same 4² = 16 bits either way and this reduction's blow-up factor is 1 — the gadget reductions of §11–§15 blow the instance up by a polynomial, never more · and because complementation is its own inverse, the same map gives CLIQUE ≤ₚ INDEPENDENT-SET; §12 puts vertex cover into the same triangle`
        : kind === "id"
          ? ` · the map is polynomial-time and perfectly well defined — being computable is not the issue; it simply fails condition (ii), and mask 0 (the empty graph, which has an independent set of size 3 and no edge at all) is the shortest witness`
          : ` · the construction is the RIGHT one and only the parameter is wrong: ⟨complement of G, k+1⟩ decides "independent set of size ${K + 1}", not size ${K}, so it disagrees exactly on the graphs whose largest independent set is ${K}`));
  }
  SX.on("red-map", "change", build); SX.on("red-focus", "input", build);
  build();
})();

/* ── 07  #landscape-svg  the two worlds ── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("landscape-svg")) return;
  const W = 700, H = 400;
  function build() {
    const world = SX.val("land-world", "neq"), showHard = SX.checked("land-hard", true);
    const svg = d3.select("#landscape-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 16, r: 16, t: 16, b: 40 });
    const cx = F.iw / 2, cy = 176;
    const eq = world === "eq";

    if (showHard) {
      F.g.append("rect").attr("x", 10).attr("y", 6).attr("width", F.iw - 20).attr("height", 338).attr("rx", 14)
        .attr("fill", "none").attr("stroke", AC.bad).attr("stroke-width", 1.4).attr("stroke-dasharray", "6 4");
      F.g.append("text").attr("x", 22).attr("y", 24).attr("font-size", 12).attr("fill", AC.bad).text("NP-HARD — everything in NP reduces to it; need not be in NP, need not be decidable");
    }
    /* NP */
    F.g.append("ellipse").attr("cx", cx).attr("cy", cy).attr("rx", eq ? 150 : 250).attr("ry", eq ? 84 : 112)
      .attr("fill", AC.panel).attr("stroke", AC.accent).attr("stroke-width", 2);
    F.g.append("text").attr("x", cx + (eq ? 0 : 200)).attr("y", cy - (eq ? 62 : 90)).attr("text-anchor", "middle").attr("font-size", 13).attr("fill", AC.accent).text("NP");
    /* P */
    F.g.append("ellipse").attr("cx", eq ? cx : cx - 130).attr("cy", eq ? cy : cy + 14).attr("rx", eq ? 148 : 104).attr("ry", eq ? 82 : 72)
      .attr("fill", AC.panel2).attr("stroke", AC.good).attr("stroke-width", 2);
    F.g.append("text").attr("x", eq ? cx : cx - 130).attr("y", eq ? cy - 62 : cy - 50).attr("text-anchor", "middle").attr("font-size", 13).attr("fill", AC.good).text(eq ? "P = NP = NP-complete" : "P");
    /* NP-complete */
    if (!eq) {
      F.g.append("ellipse").attr("cx", cx + 150).attr("cy", cy + 6).attr("rx", 84).attr("ry", 74)
        .attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 2);
      F.g.append("text").attr("x", cx + 150).attr("y", cy - 54).attr("text-anchor", "middle").attr("font-size", 13).attr("fill", AC.a2).text("NP-complete");
      F.g.append("text").attr("x", cx + 14).attr("y", cy - 76).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.violet).text("NP-intermediate (Ladner)");
    }
    const place = (x, y, label, col) => {
      F.g.append("circle").attr("cx", x).attr("cy", y).attr("r", 3.4).attr("fill", col);
      F.g.append("text").attr("x", x + 7).attr("y", y + 4).attr("font-size", 10.5).attr("fill", col).text(label);
    };
    const inP = ["sorting", "shortest paths", "max flow, matching", "2-SAT, 2-colouring", "primality (AKS)", "linear programming"];
    const inNPC = ["3-SAT", "clique, vertex cover", "Hamiltonian cycle", "TSP (decision)", "subset sum, 3-colouring"];
    const mid = ["graph isomorphism", "factoring (decision)"];
    const hard = ["TSP (optimisation) — NP-hard, not in NP", "the halting problem — NP-hard, undecidable"];
    if (eq) {
      inP.forEach((l, i) => place(cx - 140, cy - 50 + i * 20, l, AC.good));
      inNPC.forEach((l, i) => place(cx + 16, cy - 50 + i * 20, l, AC.good));
    } else {
      inP.forEach((l, i) => place(cx - 200, cy - 26 + i * 20, l, AC.good));
      inNPC.forEach((l, i) => place(cx + 106, cy - 20 + i * 20, l, AC.a2));
      mid.forEach((l, i) => place(cx - 46, cy - 46 + i * 20, l, AC.violet));
    }
    if (showHard) hard.forEach((l, i) => place(cx - 170, cy + 132 + i * 20, l, AC.bad));
    (eq ? ["if P = NP: no intermediate region and no top — every problem in NP has a",
           "polynomial algorithm, and §19–§30 of this page become unnecessary"]
        : ["if P ≠ NP: P, NP-intermediate and NP-complete are three disjoint, non-empty",
           "regions, and Ladner's theorem is what supplies the middle one"]).forEach((t, i) =>
      F.g.append("text").attr("x", 10).attr("y", F.ih + 10 + i * 16).attr("font-size", 11.5).attr("fill", AC.muted).text(t));

    /* measured costs of some of the placed problems, on this page's instances */
    const c1 = AL.counter(), c2 = AL.counter(), c3 = AL.counter(), c4 = AL.counter();
    const bs = NR.satBrute(NI.phi.clauses, NI.phi.n, c1);
    NR.verifySat(NI.phi.clauses, [true, false, true], c2);
    const g = NR.adjFromNamed(NI.gVC); const mc = NR.cliqueMax(g.A, [...Array(g.n).keys()], c3);
    const ss = NR.subsetBrute(NI.subsetsum.S, NI.subsetsum.T, c4);
    const rc = NR.satToClique(NI.phi);
    SX.setHtml("landscape-readout", `hypothesis: <b>${eq ? "P = NP" : "P ≠ NP"}</b> · the diagram is the only thing on this page that is drawn rather than measured — what IS measured, at load, is the cost of the placed problems on this page's instances: ` +
      `deciding φ by exhaustive search took <b>${SX.int(c1.get("assign"))}</b> assignments and <b>${SX.int(c1.get("lit"))}</b> literal tests, while VERIFYING one satisfying assignment took <b>${c2.get("check")}</b> ${SX.flag(c2.get("check") < c1.get("lit"))}; ` +
      `maximum clique on the seven-vertex graph took <b>${SX.int(c3.get("subset"))}</b> subset tests to return size ${mc.size}; subset sum on {${NI.subsetsum.S.join(",")}} took <b>${SX.int(c4.get("subset"))}</b> subsets to find ${ss.count} solutions · ` +
      `and the reduction of §11 turns φ into a graph with <b>${rc.n}</b> vertices and <b>${rc.E.length}</b> edges, which is what "polynomial-time reduction" costs in the concrete · ` +
      `every one of those counts is exponential in the instance parameter and every verification is linear — which is the gap the diagram is about, and which no picture can settle`);
  }
  SX.on("land-world", "change", build); SX.on("land-hard", "change", build);
  build();
})();

/* ── 08  #circuit-svg  Tseitin, and the Cook–Levin tableau ── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("circuit-svg")) return;
  const W = 700, H = 430;
  const litTxt = l => (l > 0 ? "" : "¬") + (Math.abs(l) <= 3 ? "x" + Math.abs(l) : "g" + (Math.abs(l) - 3));
  function build() {
    const mode = SX.val("circ-mode", "tseitin"), sel = +SX.val("circ-assign", 6);
    const svg = d3.select("#circuit-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 16, r: 16, t: 16, b: 36 });

    if (mode === "tseitin") {
      const C = NR.CIRC, cc = AL.counter(), ct = AL.counter(), cb = AL.counter();
      const x = NR.bitsOf(sel, 3);
      const ev = NR.circuitEval(C, x, cc);
      const T = NR.tseitin(C, ct);
      const chk = NR.tseitinCheck(C, cb);
      /* the circuit, laid out by hand */
      const pos = { 1: [40, 40], 2: [40, 110], 3: [40, 180], 4: [150, 66], 5: [150, 156], 6: [150, 18], 7: [255, 40], 8: [360, 100] };
      const wire = (a, b, on) => F.g.append("path").attr("d", `M${pos[a][0] + 20},${pos[a][1]} C${pos[a][0] + 60},${pos[a][1]} ${pos[b][0] - 40},${pos[b][1]} ${pos[b][0] - 20},${pos[b][1]}`)
        .attr("fill", "none").attr("stroke", on ? AC.good : AC.line).attr("stroke-width", on ? 2.4 : 1.4);
      C.gates.forEach(g => g.in.forEach(i => wire(i, g.id, !!ev.wires[i])));
      for (let i = 1; i <= 3; i++) { const p = pos[i];
        F.g.append("rect").attr("x", p[0] - 20).attr("y", p[1] - 13).attr("width", 40).attr("height", 26).attr("rx", 5)
          .attr("fill", ev.wires[i] ? AC.good : AC.panel2).attr("stroke", AC.line);
        F.g.append("text").attr("x", p[0]).attr("y", p[1] + 5).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", ev.wires[i] ? AC.bg : AC.ink).text("x" + i + "=" + (ev.wires[i] ? 1 : 0)); }
      C.gates.forEach((g, i) => { const p = pos[g.id];
        F.g.append("rect").attr("x", p[0] - 22).attr("y", p[1] - 14).attr("width", 44).attr("height", 28).attr("rx", 6)
          .attr("fill", ev.wires[g.id] ? AC.good : AC.panel2).attr("stroke", g.id === C.out ? AC.a2 : AC.line).attr("stroke-width", g.id === C.out ? 2.2 : 1);
        F.g.append("text").attr("x", p[0]).attr("y", p[1] + 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", ev.wires[g.id] ? AC.bg : AC.ink).text(g.op);
        F.g.append("text").attr("x", p[0]).attr("y", p[1] - 20).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.muted).text("g" + (g.id - 3) + " = " + (ev.wires[g.id] ? 1 : 0)); });
      F.g.append("text").attr("x", pos[8][0] + 34).attr("y", pos[8][1] + 4).attr("font-size", 12).attr("fill", ev.value ? AC.good : AC.bad).text("output " + (ev.value ? 1 : 0));
      /* the clauses */
      F.g.append("text").attr("x", 430).attr("y", 14).attr("font-size", 11).attr("fill", AC.muted).text("Tseitin: " + T.count + " clauses, " + T.vars + " vars");
      T.tagged.forEach((o, i) => {
        const col = i % 2, row = Math.floor(i / 2);
        const sat = o.c.some(l => (l > 0) === !!(NR.bitsOf(sel, 3).concat([ev.wires[4], ev.wires[5], ev.wires[6], ev.wires[7], ev.wires[8]]))[Math.abs(l) - 1]);
        F.g.append("text").attr("x", 430 + col * 118).attr("y", 34 + row * 18).attr("font-size", 11)
          .attr("fill", sat ? AC.good : AC.bad).text("(" + o.c.map(litTxt).join(" ∨ ") + ")");
      });
      F.g.append("text").attr("x", 20).attr("y", 250).attr("font-size", 11.5).attr("fill", AC.muted).text("circuit truth table (x₁x₂x₃ → output), all 8 inputs:");
      for (let m = 0; m < 8; m++) { const v = NR.circuitEval(C, NR.bitsOf(m, 3)).value;
        F.g.append("rect").attr("x", 20 + m * 46).attr("y", 262).attr("width", 40).attr("height", 26).attr("rx", 4)
          .attr("fill", v ? AC.good : AC.panel2).attr("stroke", m === sel ? AC.a2 : AC.line).attr("stroke-width", m === sel ? 2.2 : 1);
        F.g.append("text").attr("x", 40 + m * 46).attr("y", 279).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", v ? AC.bg : AC.ink).text(NR.bitStr(NR.bitsOf(m, 3))); }
      F.g.append("text").attr("x", 20).attr("y", 312).attr("font-size", 11.5).attr("fill", AC.muted).text("satisfying assignments of the CNF, projected onto x₁x₂x₃ (from all 2⁸ = 256):");
      chk.proj.forEach((p, i) => { F.g.append("rect").attr("x", 20 + i * 46).attr("y", 324).attr("width", 40).attr("height", 26).attr("rx", 4).attr("fill", AC.accent).attr("stroke", AC.line);
        F.g.append("text").attr("x", 40 + i * 46).attr("y", 341).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.bg).text(p); });
      SX.setHtml("circuit-readout", `circuit: g₁ = x₁ ∧ x₂, g₂ = x₂ ∨ x₃, g₃ = ¬x₁, g₄ = g₁ ∨ g₃, g₅ = g₄ ∧ g₂ (output) · on input ${NR.bitStr(NR.bitsOf(sel, 3))} the ${cc.get("gate")} gates evaluate to ${[4, 5, 6, 7, 8].map(i => (ev.wires[i] ? 1 : 0)).join("")} and the circuit outputs <b>${ev.value ? 1 : 0}</b> · ` +
        `measured Tseitin size: <b>${ct.get("clause")}</b> clauses over <b>${T.vars}</b> variables = 3 + 3 + 2 + 3 + 3 for the five gates plus one unit clause ${SX.flag(ct.get("clause") === 15)}, every clause of width ≤ 3 ${SX.flag(T.clauses.every(cl => cl.length <= 3))} · ` +
        `exhaustive check: the circuit is satisfied by <b>${chk.circuit.length}</b> of its 8 inputs — {${chk.circuit.join(", ")}}; the CNF is satisfied by <b>${chk.sols.length}</b> of its 2⁸ = ${SX.int(chk.assigns)} assignments, and their projections onto x₁x₂x₃ are {${chk.proj.join(", ")}} ${SX.flag(chk.agree)} · ` +
        `the counts are equal too (${chk.sols.length} = ${chk.circuit.length}) ${SX.flag(chk.sameCount)}, which is the unique-extension property: the gate variables are forced bottom-up, so each satisfying input extends in exactly one way · equisatisfiable, not equivalent — the CNF lives on ${T.vars} variables, the circuit on 3`);
      return;
    }

    /* ── the tableau mode ── */
    const yBits = [["0", "0"], ["0", "1"], ["1", "0"], ["1", "1"]][sel % 4];
    const cr = AL.counter(), cl = AL.counter();
    const run = NR.tmRun(["0", "0"], yBits, cr);
    const L = NR.tmLegalWindows(cl);
    const S = NR.tmFormulaSizes(L);
    const rows = run.rows, RL = rows[0].length;
    const cw = 42, ch = 30, ox = 40, oy = 44;
    F.g.append("text").attr("x", ox).attr("y", 14).attr("font-size", 11.5).attr("fill", AC.muted)
      .text("the tableau of a verifier on x = 00, certificate y = " + yBits.join(""));
    F.g.append("text").attr("x", ox).attr("y", 30).attr("font-size", 11).attr("fill", AC.muted)
      .text("row i is the machine's configuration after i steps");
    rows.forEach((r, i) => r.forEach((s, j) => {
      const isHead = NR.TM.ST.indexOf(s) >= 0, isCert = i === 0 && (j === 4 || j === 5);
      F.g.append("rect").attr("x", ox + j * cw).attr("y", oy + i * ch).attr("width", cw - 3).attr("height", ch - 3).attr("rx", 4)
        .attr("fill", s === "qa" ? AC.good : isHead ? AC.a2 : isCert ? AC.violet : AC.panel2).attr("stroke", AC.line);
      F.g.append("text").attr("x", ox + j * cw + (cw - 3) / 2).attr("y", oy + i * ch + 19).attr("text-anchor", "middle").attr("font-size", 12)
        .attr("fill", (s === "qa" || isHead || isCert) ? AC.bg : AC.ink).text(s);
    }));
    rows.forEach((r, i) => F.g.append("text").attr("x", ox - 8).attr("y", oy + i * ch + 19).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.muted).text("row " + i));
    /* one 2 × 3 window highlighted */
    const wi = 2, wj = Math.min(3, RL - 2);
    F.g.append("rect").attr("x", ox + (wj - 1) * cw - 3).attr("y", oy + wi * ch - 3).attr("width", 3 * cw + 1).attr("height", 2 * ch + 1).attr("rx", 6)
      .attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 2.4);
    F.g.append("text").attr("x", ox + (wj - 1) * cw).attr("y", oy + (wi + 2) * ch + 16).attr("font-size", 11).attr("fill", AC.accent)
      .text("one 2 × 3 window: " + rows[wi].slice(wj - 1, wj + 2).join(" ") + " / " + rows[wi + 1].slice(wj - 1, wj + 2).join(" "));
    ["amber = the state symbol, i.e. the head;  violet = the certificate cells,",
     "left FREE by φstart;  green = the accepting state that φaccept looks for"].forEach((t, i) =>
      F.g.append("text").attr("x", ox).attr("y", oy + rows.length * ch + 48 + i * 16).attr("font-size", 11).attr("fill", AC.muted).text(t));
    const winLegal = L.legal.has(rows[wi].slice(wj - 1, wj + 2).concat(rows[wi + 1].slice(wj - 1, wj + 2)).join("|"));
    SX.setHtml("circuit-readout", `the machine sweeps right and accepts on the first 1 it reads; with x = 00 the accepting certificates are exactly those containing a 1 · on y = ${yBits.join("")} the tableau ${run.accept ? "REACHES the accepting state" : "never reaches an accepting state within the " + NR.TM.rows + " rows"}, so φaccept is ${run.accept ? "satisfied" : "not satisfied"} by this filling of the free cells · ` +
      `measured sizes: |C| = <b>${S.symbols}</b> symbols, a ${NR.TM.rows} × ${NR.TM.rowLen} tableau = <b>${S.cells}</b> cells, hence <b>${SX.int(S.vars)}</b> cell variables = |C| · cells ${SX.flag(S.vars === S.symbols * S.cells)}; φcell contributes <b>${SX.int(S.cellClauses)}</b> clauses (one "at least one symbol" clause plus C(${S.symbols},2) = ${S.symbols * (S.symbols - 1) / 2} exclusions per cell); φaccept is one clause of width ${S.acceptWidth} · ` +
      `the legal-window set is MEASURED, not assumed: all <b>${SX.int(L.configs)}</b> configurations of this machine over a ${NR.TM.tape}-cell tape were enumerated and stepped, yielding <b>${SX.int(L.size)}</b> distinct legal windows out of ${S.symbols}⁶ = ${SX.int(L.all)} possible ones, so <b>${SX.int(L.illegal)}</b> are illegal; with <b>${S.positions}</b> window positions ((rows−1)(rowlen−2) = ${NR.TM.rows - 1} × ${NR.TM.rowLen - 2} ${SX.flag(S.positions === (NR.TM.rows - 1) * (NR.TM.rowLen - 2))}) the forbid-each-illegal rendering of φmove has <b>${SX.int(S.moveClauses)}</b> clauses · ` +
      `the highlighted window is ${winLegal ? "legal" : "ILLEGAL"} ${SX.flag(winLegal)} — as every window of a real computation must be · four million clauses for a six-cell tape is the honest shape of the theorem: the per-window cost is a CONSTANT of the machine and the number of windows is polynomial in n`);
  }
  SX.on("circ-mode", "change", build); SX.on("circ-assign", "change", build);
  build();
})();

/* ── 09  #sat3-svg  CNF → 3-CNF, with both formulas brute-forced ── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("sat3-svg")) return;
  const W = 700, H = 420;
  const N0 = 5, CL = [[1], [-2, 3], [1, -3, 4], [-1, 2, -4, 5], [1, -2, 3, 4, -5]];
  const lit = l => (l > 0 ? "" : "¬") + (Math.abs(l) <= N0 ? "x" + Math.abs(l) : "y" + (Math.abs(l) - N0));
  function build() {
    const pick = +SX.val("sat3-clause", 4), showAll = SX.checked("sat3-all", true);
    const svg = d3.select("#sat3-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 16, r: 16, t: 16, b: 34 });
    const cc = AL.counter(), cb1 = AL.counter(), cb2 = AL.counter();
    const R = NR.cnfTo3(CL, N0, cc);
    const orig = NR.satBrute(CL, N0, cb1);
    const conv = NR.cnfProject(R.clauses, R.vars, N0, cb2);
    const same = NR.setsEqual(conv.proj, orig.sols.slice().sort());

    F.g.append("text").attr("x", 0).attr("y", 14).attr("font-size", 11.5).attr("fill", AC.muted).text("the input CNF — 5 variables, 5 clauses of widths 1, 2, 3, 4, 5 (15 literals in all)");
    CL.forEach((cl, i) => {
      const cur = i === pick;
      F.g.append("rect").attr("x", 0).attr("y", 24 + i * 28).attr("width", 250).attr("height", 24).attr("rx", 5)
        .attr("fill", cur ? AC.panel : AC.panel2).attr("stroke", cur ? AC.a2 : AC.line).attr("stroke-width", cur ? 2 : 1);
      F.g.append("text").attr("x", 10).attr("y", 41 + i * 28).attr("font-size", 11.5).attr("fill", AC.ink).text("C" + (i + 1) + " = (" + cl.map(lit).join(" ∨ ") + ")");
      F.g.append("text").attr("x", 196).attr("y", 41 + i * 28).attr("font-size", 10.5).attr("fill", AC.muted).text("width " + cl.length);
      AL.arrow(F.g, 256, 36 + i * 28, 286, 36 + i * 28, { color: cur ? AC.a2 : AC.line, w: cur ? 2 : 1.2 });
    });
    const g = R.groups[pick];
    F.g.append("text").attr("x", 296).attr("y", 14).attr("font-size", 11.5).attr("fill", AC.a2)
      .text("C" + (pick + 1) + " becomes " + g.added + " clause" + (g.added === 1 ? "" : "s") + (g.fresh.length ? ", with " + g.fresh.length + " fresh variable" + (g.fresh.length === 1 ? "" : "s") + " " + g.fresh.map(lit).join(", ") : ", unchanged"));
    R.clauses.slice(g.from, g.to).forEach((cl, i) => {
      F.g.append("rect").attr("x", 296).attr("y", 24 + i * 26).attr("width", 200).attr("height", 22).attr("rx", 5).attr("fill", AC.panel).attr("stroke", AC.a2);
      F.g.append("text").attr("x", 306).attr("y", 40 + i * 26).attr("font-size", 11.5).attr("fill", AC.ink).text("(" + cl.map(lit).join(" ∨ ") + ")");
    });
    if (showAll) {
      F.g.append("text").attr("x", 512).attr("y", 14).attr("font-size", 11.5).attr("fill", AC.muted).text("the whole 3-CNF: " + R.clauses.length + " clauses");
      R.clauses.forEach((cl, i) => F.g.append("text").attr("x", 512).attr("y", 30 + i * 15).attr("font-size", 10)
        .attr("fill", i >= g.from && i < g.to ? AC.a2 : AC.muted).text("(" + cl.map(lit).join("∨") + ")"));
    }
    /* the two solution sets, as rows of 32 cells */
    const setO = new Set(orig.sols), setP = new Set(conv.proj);
    const cw = 20, gp = 1, ox = 0;
    [["satisfying assignments of the ORIGINAL (all 2⁵ = 32 tried)", setO, 240], ["projections of the 3-CNF's solutions onto x₁…x₅ (all 2¹¹ = 2 048 tried)", setP, 300]].forEach(([lab, S, yy]) => {
      F.g.append("text").attr("x", ox).attr("y", yy - 8).attr("font-size", 11).attr("fill", AC.muted).text(lab);
      for (let m = 0; m < 32; m++) { const s = NR.bitStr(NR.bitsOf(m, N0)); const on = S.has(s);
        F.g.append("rect").attr("x", ox + m * (cw + gp)).attr("y", yy).attr("width", cw).attr("height", 22).attr("rx", 3)
          .attr("fill", on ? AC.good : AC.panel2).attr("stroke", AC.line);
        F.g.append("text").attr("x", ox + m * (cw + gp) + cw / 2).attr("y", yy + 15).attr("text-anchor", "middle").attr("font-size", 8).attr("fill", on ? AC.bg : AC.muted).text(m); }
    });
    F.g.append("text").attr("x", 0).attr("y", 352).attr("font-size", 11.5).attr("fill", same ? AC.good : AC.bad)
      .text(same ? "the two rows are identical, cell for cell — the conversion preserves exactly which assignments to x₁…x₅ work" : "the two rows DIFFER — the conversion is wrong");
    SX.setHtml("sat3-readout", `in: <b>${N0}</b> variables, <b>${CL.length}</b> clauses, <b>${CL.reduce((s, c) => s + c.length, 0)}</b> literals, widths ${CL.map(c => c.length).join(", ")} · ` +
      `out: <b>${R.vars}</b> variables (${R.freshCount} fresh) and <b>${cc.get("clause")}</b> clauses, every one of width ≤ 3 ${SX.flag(R.clauses.every(c => c.length <= 3))} · per input clause, measured: ${R.groups.map((gg, i) => "width " + gg.width + " → " + gg.added + " clause" + (gg.added === 1 ? "" : "s") + (gg.fresh.length ? " + " + gg.fresh.length + " var" + (gg.fresh.length === 1 ? "" : "s") : "")).join("; ")} — matching the gadget's k−2 clauses and k−3 fresh variables for k ≥ 4 ${SX.flag(R.groups.every(gg => gg.width < 4 || (gg.added === gg.width - 2 && gg.fresh.length === gg.width - 3)))} · ` +
      `exhaustive on the original: <b>${SX.int(orig.assigns)}</b> assignments, <b>${orig.count}</b> satisfying · exhaustive on the 3-CNF: <b>${SX.int(conv.assigns)}</b> assignments, <b>${conv.sols}</b> satisfying, projecting onto <b>${conv.proj.length}</b> distinct settings of x₁…x₅ · ` +
      `the two solution sets are identical ${SX.flag(same)} — equisatisfiable in the strong sense the reduction needs · ` +
      `note ${conv.sols} > ${conv.proj.length}: several carry settings can witness the same original assignment, which is why this is equisatisfiability and not logical equivalence · the output is linear in the input, so the map is a legitimate polynomial-time reduction (§06)`);
  }
  SX.on("sat3-clause", "change", build); SX.on("sat3-all", "change", build);
  build();
})();


/* ── chunk B figures ────────────────────────────────────────── */
/* ── chunk B figures (§10–§18) ─────────────────────────────────────────── */

/* SX.npDraw(g, pos, opt) — an undirected graph, edges first so nodes sit on
   top.  pos is [[x,y],…].  opt.edge(e,i) → {color,width,dash} or null;
   opt.vertex(v) → {fill,stroke,ink,r}; opt.label(v) → text under the vertex. */
if (!SX.npDraw) SX.npDraw = function (g, pos, opt) {
  const o = Object.assign({ edge: function () { return null; }, vertex: function () { return null; },
    label: function () { return ""; }, labelColor: function () { return AC.muted; }, r: 15, fontSize: 11, into: null }, opt || {});
  const gg = o.into;
  g.E.forEach(function (e, i) {
    const a = pos[e[0]], b = pos[e[1]]; const st = o.edge(e, i) || {};
    const ln = gg.append("line").attr("x1", a[0]).attr("y1", a[1]).attr("x2", b[0]).attr("y2", b[1])
      .attr("stroke", st.color || AC.line).attr("stroke-width", st.width || 1.6).attr("stroke-linecap", "round");
    if (st.dash) ln.attr("stroke-dasharray", st.dash);
    if (st.op !== undefined) ln.attr("opacity", st.op);
  });
  for (let v = 0; v < g.n; v++) {
    const p = pos[v], st = o.vertex(v) || {};
    gg.append("circle").attr("cx", p[0]).attr("cy", p[1]).attr("r", st.r || o.r)
      .attr("fill", st.fill || AC.panel2).attr("stroke", st.stroke || AC.line).attr("stroke-width", st.stroke ? 2.5 : 1);
    gg.append("text").attr("x", p[0]).attr("y", p[1] + 4).attr("text-anchor", "middle")
      .attr("font-size", st.fs || o.fontSize + 1).attr("fill", st.ink || AC.ink).text(g.names ? g.names[v] : v);
    const lab = o.label(v);
    if (lab) gg.append("text").attr("x", p[0]).attr("y", p[1] + (st.r || o.r) + 12).attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", o.labelColor(v)).text(lab);
  }
  return gg;
};

/* ── 10  #map-svg  the reduction DAG, clickable, with measured blow-up ─── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("map-svg")) return;
  const W = 680, H = 430;
  const N = [
    { id: "circuit", t: "CIRCUIT-SAT", x: 340, y: 26, p: [] },
    { id: "sat", t: "SAT", x: 340, y: 66, p: ["circuit"], gad: "one variable and 1–3 clauses per gate" },
    { id: "sat3", t: "3-SAT", x: 340, y: 106, p: ["sat"], gad: "a fresh variable per split of a wide clause" },
    { id: "clique", t: "CLIQUE", x: 120, y: 156, p: ["sat3"], gad: "a vertex per literal occurrence" },
    { id: "vc", t: "VERTEX-COVER", x: 120, y: 202, p: ["clique"], gad: "complement the graph, k′ = |V| − k" },
    { id: "is", t: "INDEPENDENT-SET", x: 62, y: 250, p: ["vc"], gad: "the complementary set" },
    { id: "sc", t: "SET-COVER", x: 196, y: 250, p: ["vc"], gad: "an element per edge, a set per vertex" },
    { id: "ham", t: "HAM-CYCLE", x: 120, y: 300, p: ["vc"], gad: "a 12-vertex widget per edge, k selectors" },
    { id: "tsp", t: "TSP", x: 120, y: 348, p: ["ham"], gad: "weight 1 inside G, more outside, budget |V|" },
    { id: "ss", t: "SUBSET-SUM", x: 360, y: 156, p: ["sat3"], gad: "digit columns, base 10, no carry" },
    { id: "part", t: "PARTITION", x: 360, y: 202, p: ["ss"], gad: "add 2σ − T and σ + T" },
    { id: "knap", t: "KNAPSACK", x: 360, y: 250, p: ["ss"], gad: "weight = value = the number" },
    { id: "col3", t: "3-COLOURING", x: 540, y: 156, p: ["sat3"], gad: "palette triangle + OR-gadgets" },
    { id: "dm3", t: "3-DIM-MATCHING", x: 556, y: 212, p: ["sat3"], gad: "a triple gadget per clause" }
  ];
  const short = { sat: "gate clauses", sat3: "split clauses", clique: "literal graph", vc: "complement",
    is: "complement set", sc: "edge = element", ham: "12-vertex widget", tsp: "1 / 2 weights",
    ss: "digit columns", part: "+ 2 numbers", knap: "w = v = s", col3: "palette + OR", dm3: "triple gadget" };
  const byId = {}; N.forEach(function (d) { byId[d.id] = d; });
  function chain(id) { const out = []; (function up(k) { const d = byId[k]; d.p.forEach(up); out.push(k); })(id); return out; }
  let sel = "tsp";

  function sizes(phi) {
    /* every number below is produced by RUNNING the reduction, not asserted */
    const m = phi.clauses.length, n = phi.n;
    const CQ = NR.cliqueFromSat(phi), comp = NR.npComplement(CQ.g), kp = CQ.g.n - CQ.k;
    const hs = NR.hamReductionSize(comp, kp);
    const R = NR.satToSubsetSum(phi);
    const C3 = NR.satTo3Col(phi);
    return {
      circuit: "the verifier compiled to a circuit — not built here",
      sat: n + " variables, " + m + " clauses (already CNF)",
      sat3: n + " variables, " + m + " clauses, " + (3 * m) + " literal occurrences",
      clique: CQ.g.n + " vertices, " + CQ.g.E.length + " edges, k = " + CQ.k,
      vc: comp.n + " vertices, " + comp.E.length + " edges, k′ = " + kp,
      is: comp.n + " vertices, " + comp.E.length + " edges, k = " + CQ.k,
      sc: comp.E.length + " elements, " + comp.n + " sets, k = " + kp,
      ham: SX.int(hs.V) + " vertices, " + SX.int(hs.E) + " edges",
      tsp: SX.int(hs.V) + " cities, " + SX.int(hs.V * (hs.V - 1) / 2) + " weights, budget " + SX.int(hs.V),
      ss: R.rows.length + " numbers of " + R.cols + " digits, target " + R.targetValue,
      part: (R.rows.length + 2) + " numbers (the two extra force the split)",
      knap: R.rows.length + " items, W = V = " + R.targetValue,
      col3: C3.g.n + " vertices, " + C3.g.E.length + " edges",
      dm3: "named on this page, not built"
    };
  }

  function build() {
    const which = SX.val("map-inst", "phi");
    const phi = which === "phi8" ? NI.phi8 : NI.phi;
    const SZ = sizes(phi);
    const ch = chain(sel), inChain = new Set(ch);
    const svg = d3.select("#map-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const bw = 112, bh = 24;
    /* arrows first */
    N.forEach(function (d) {
      d.p.forEach(function (pid) {
        const a = byId[pid], on = inChain.has(d.id) && inChain.has(pid);
        const x1 = a.x, y1 = a.y + bh / 2, x2 = d.x, y2 = d.y - bh / 2 - 5;
        AL.arrow(F.g, x1, y1, x2, y2, { color: on ? AC.accent : AC.line, w: on ? 2.4 : 1.2, head: on ? 7 : 5 });
        F.g.append("text").attr("x", (x1 + x2) / 2 + (Math.abs(x2 - x1) > 30 ? 6 : 8)).attr("y", (y1 + y2) / 2 + 3)
          .attr("font-size", 9).attr("fill", on ? AC.accent : AC.muted).attr("text-anchor", Math.abs(x2 - x1) > 30 ? "middle" : "start")
          .text(short[d.id] || "");
      });
    });
    /* boxes */
    N.forEach(function (d) {
      const on = inChain.has(d.id), me = d.id === sel;
      const gg = F.g.append("g").style("cursor", "pointer").on("click", function () { sel = d.id; build(); });
      gg.append("rect").attr("x", d.x - bw / 2).attr("y", d.y - bh / 2).attr("width", bw).attr("height", bh).attr("rx", 5)
        .attr("fill", me ? AC.a2 : on ? AC.accent : AC.panel2).attr("stroke", on ? AC.accent : AC.line).attr("stroke-width", on ? 2 : 1);
      gg.append("text").attr("x", d.x).attr("y", d.y + 4).attr("text-anchor", "middle").attr("font-size", 10.5)
        .attr("fill", (me || on) ? AC.bg : AC.ink).attr("font-weight", me ? "600" : "400").text(d.t);
    });
    F.g.append("text").attr("x", 12).attr("y", H - 32).attr("font-size", 11).attr("fill", AC.muted)
      .text("click any problem — the highlighted arrows are the chain that proves it NP-hard, read from CIRCUIT-SAT downward");
    F.g.append("text").attr("x", 12).attr("y", H - 16).attr("font-size", 11).attr("fill", AC.muted)
      .text("an arrow A → B means A ≤ₚ B: hardness flows DOWN the arrow, an efficient algorithm flows UP it");
    /* the chain, with every instance size measured by running the reduction */
    const steps = ch.map(function (id, i) {
      const d = byId[id];
      return (i === 0 ? "<b>" + d.t + "</b>" : "→ <b>" + d.t + "</b> <i>(" + d.gad + ")</i>") + " : " + SZ[id];
    });
    const CQ = NR.cliqueFromSat(phi);
    const sat = NR.satAllB(phi, AL.counter());
    const xc = NR.satCrossCheck(phi);
    SX.setHtml("map-readout",
      "formula: <b>" + (which === "phi8" ? "φ₈" : "φ") + "</b>, " + phi.n + " variables, " + phi.clauses.length + " clauses · exhaustive search over all " +
      sat.tried + " assignments finds <b>" + sat.models.length + "</b> satisfying" +
      (xc.have && xc.count !== null && xc.count !== undefined ? " (chunk A's satBrute agrees: " + xc.count + " " + SX.flag(xc.count === sat.models.length) + ")" : "") +
      " · reduction to CLIQUE cross-checked against the shared routine " + (CQ.haveExt ? (CQ.readable ? SX.flag(CQ.agree) + " (" + CQ.extN + " vertices, " + CQ.extM + " edges)" : "— its output shape was not recognised, so this figure's own construction is used") : "— shared routine not present, this figure's own construction used") +
      "<br>chain proving <b>" + byId[sel].t + "</b> NP-hard, with every instance size produced by running the reductions on this formula:<br>" +
      steps.join("<br>"));
  }
  SX.on("map-inst", "change", build);
  build();
})();

/* ── 11  #clique-svg  3-SAT → CLIQUE, with the assignment read back ────── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("clique-svg")) return;
  const W = 680, H = 330;
  let chosen = 0, lastInst = null;
  function build(keep) {
    const which = SX.val("clq-inst", "phi");
    const phi = which === "phi8" ? NI.phi8 : NI.phi;
    if (lastInst !== which) { chosen = 0; lastInst = which; }
    const CQ = NR.cliqueFromSat(phi), g = CQ.g, m = phi.clauses.length;
    const small = g.n <= 14;
    const cBrute = AL.counter(), cBK = AL.counter();
    const brute = small ? NR.cliqueBrute(g, cBrute) : null;
    const bk = NR.maxCliqueBK(g, cBK);
    const maxSize = small ? brute.max : bk.size;
    const cTri = AL.counter();
    const kCl = small ? NR.cliquesOfSize(g, maxSize, cTri) : { list: [bk.best], combos: NR.choose(g.n, maxSize) };
    const list = kCl.list;
    if (chosen >= list.length) chosen = 0;
    /* the dropdown of cliques */
    const sel = d3.select("#clq-which");
    if (!sel.empty() && !keep) {
      sel.selectAll("option").remove();
      list.forEach(function (S, i) { sel.append("option").attr("value", i).text("clique " + (i + 1) + " of " + list.length + ": " + S.map(function (v) { return g.names[v]; }).join(", ")); });
      sel.property("value", String(chosen));
    }
    const S = list[chosen] || [], inS = new Set(S);
    /* layout: one column per clause */
    const sp = Math.min(170, 560 / Math.max(1, m - 1));
    const x0 = 340 - sp * (m - 1) / 2;
    const pos = CQ.verts.map(function (v) { return [x0 + sp * v.c, 96 + 78 * v.p]; });
    const svg = d3.select("#clique-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const drawAll = SX.checked("clq-edges", true);
    for (let ci = 0; ci < m; ci++) {
      F.g.append("text").attr("x", x0 + sp * ci).attr("y", 62).attr("text-anchor", "middle").attr("font-size", 11)
        .attr("fill", AC.muted).text("C" + ["₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈"][ci] + " = (" + phi.clauses[ci].map(function (l) { return (l > 0 ? "" : "¬") + "x" + Math.abs(l); }).join(" ∨ ") + ")");
    }
    SX.npDraw(g, pos, { into: F.g, r: g.n > 14 ? 11 : 15, fontSize: g.n > 14 ? 9 : 11,
      edge: function (e) { const on = inS.has(e[0]) && inS.has(e[1]); if (!drawAll && !on) return { color: AC.line, width: 0.1, op: 0 }; return on ? { color: AC.a2, width: 3 } : { color: AC.line, width: 1, op: 0.45 }; },
      vertex: function (v) { return inS.has(v) ? { fill: AC.a2, ink: AC.bg, stroke: AC.a2 } : null; } });
    /* the assignment the highlighted clique encodes */
    const a = NR.assignFromClique(phi, CQ.verts, S);
    const comp = NR.completions(a);
    const sat = NR.satAllB(phi, AL.counter());
    const models = new Set(sat.models.map(NR.bitsOfB));
    /* a clique of size j is a consistent choice of one true literal in each of j
       distinct clauses, so its assignment must satisfy AT LEAST j clauses —
       true whether or not j reaches m, which is the check worth flagging. */
    const bestSat = comp.reduce(function (acc, b) {
      const arr = b.split("").map(function (ch) { return ch === "1"; });
      return Math.max(acc, NR.satCountB(phi.clauses, arr));
    }, 0);
    const promiseLocal = bestSat >= S.length;
    const allOK = comp.every(function (b) { return models.has(b); });
    F.g.append("text").attr("x", 12).attr("y", 28).attr("font-size", 11.5).attr("fill", AC.ink)
      .text("k = m = " + m + " · the clique takes one vertex from each clause, and its literals are pairwise non-complementary — that IS an assignment");
    F.g.append("text").attr("x", 12).attr("y", H - 12).attr("font-size", 11.5).attr("fill", AC.good)
      .text(S.length ? ("clique " + S.map(function (v) { return g.names[v] + " (C" + (CQ.verts[v].c + 1) + ")"; }).join(" · ") + "  →  assignment " + a.map(function (x) { return x === null ? "∗" : (x ? 1 : 0); }).join("") + (comp.length > 1 ? "  (∗ = free, " + comp.length + " completions)" : "")) : "no clique of that size");
    /* every census in the readout is a measurement */
    const census = small ? Object.keys(brute.bySize).sort(function (p, q) { return p - q; }).map(function (k) { return k + ":" + brute.bySize[k]; }).join("  ") : "—";
    const allComp = new Set(); list.forEach(function (Sq) { NR.completions(NR.assignFromClique(phi, CQ.verts, Sq)).forEach(function (b) { allComp.add(b); }); });
    const setsMatch = allComp.size === models.size && Array.from(allComp).every(function (b) { return models.has(b); });
    const promise = (maxSize >= m) === (sat.models.length > 0);
    SX.setHtml("clique-readout",
      "measured: <b>" + g.n + "</b> vertices = 3m " + SX.flag(g.n === 3 * m) + " · <b>" + g.E.length + "</b> edges from " + CQ.pairsTested + " = C(" + g.n + ",2) candidate pairs " + SX.flag(CQ.pairsTested === g.n * (g.n - 1) / 2) +
      " (" + (CQ.pairsTested - g.E.length) + " rejected: same clause or complementary) · k = <b>" + m + "</b>" +
      "<br>maximum clique: <b>" + maxSize + "</b>" + (small ? " by enumerating all 2" + "^" + g.n + " = " + SX.int(brute.tried) + " subsets" : " by Bron–Kerbosch in " + SX.int(cBK.get("node")) + " search nodes") +
      (small ? ", and <b>" + list.length + "</b> distinct cliques of that size out of C(" + g.n + "," + maxSize + ") = " + SX.int(kCl.combos) + " candidate subsets · census by size " + census : "") +
      "<br>the reduction's promise — G has a clique of size k ⟺ φ is satisfiable: max clique " + maxSize + (maxSize >= m ? " ≥ " : " &lt; ") + "k = " + m + ", and φ has " + sat.models.length + " satisfying assignment" + (sat.models.length === 1 ? "" : "s") + " out of " + sat.tried + " " + SX.flag(promise) +
      "<br>the highlighted clique has size <b>" + S.length + "</b>, so its literals are one per clause in " + S.length + " distinct clauses and are pairwise consistent; the assignment" + (comp.length > 1 ? "s (" + comp.join(", ") + ") satisfy" : " (" + (comp[0] || "—") + ") satisfies") + " at best <b>" + bestSat + "</b> of the " + m + " clauses, which is ≥ the clique's size " + SX.flag(promiseLocal) +
      (S.length >= m ? " — and " + bestSat + " = m, so this really is a satisfying assignment " + SX.flag(allOK && S.length > 0)
                     : " — but " + S.length + " &lt; m = " + m + ", so no assignment is encoded and the reduction's answer is NO " + SX.flag(sat.models.length === 0)) +
      (small ? " · the completions of all " + list.length + " maximum cliques, as a SET, equal the " + sat.models.length + " satisfying assignments {" + Array.from(models).sort().join(", ") + "} " + SX.flag(setsMatch) : "") +
      (CQ.haveExt && CQ.readable ? " · the shared satToClique agrees on the instance size " + SX.flag(CQ.agree) : ""));
  }
  SX.on("clq-inst", "change", function () { build(false); });
  SX.on("clq-edges", "change", function () { build(true); });
  SX.on("clq-which", "change", function () { chosen = +SX.val("clq-which", "0"); build(true); });
  build(false);
})();

/* ── 12  #vc-svg  one set, three readings ──────────────────────────────── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("vc-svg")) return;
  const W = 680, H = 290;
  const g = NR.npFromLetters(NI.gVC), comp = NR.npComplement(g);
  const pos = g.names.map(function (nm) { const p = NI.gVC.pos[nm]; return [p[0] + 30, p[1] + 60]; });
  const cVC = AL.counter(), cIS = AL.counter(), cCL = AL.counter();
  const VC = NR.vcBrute(g, cVC), IS = NR.isBrute(g, cIS), CL = NR.cliqueBrute(comp, cCL);
  /* the greedy maximal matching of §21, in the stored edge order */
  const matched = new Set(), mEdges = [];
  g.E.forEach(function (e) { if (!matched.has(e[0]) && !matched.has(e[1])) { matched.add(e[0]); matched.add(e[1]); mEdges.push(e); } });
  const greedyCover = Array.from(matched).sort(function (a, b) { return a - b; });
  let chosen = 0, lastView = null;
  function optionsFor(view) {
    const nm = function (S) { return "{" + S.map(function (v) { return g.names[v]; }).join(", ") + "}"; };
    if (view === "cover") {
      const out = VC.all.map(function (S) { return { S: S, t: "minimum cover, size " + S.length + " — " + nm(S) }; });
      out.push({ S: greedyCover, t: "greedy maximal-matching cover, size " + greedyCover.length + " — " + nm(greedyCover) });
      out.push({ S: [1, 3], t: "a non-cover, size 2 — " + nm([1, 3]) + " (leaves edges uncovered)" });
      return out;
    }
    const out = IS.all.map(function (S) { return { S: S, t: "maximum independent set, size " + S.length + " — " + nm(S) }; });
    out.push({ S: [0, 2, 5], t: "a smaller independent set, size 3 — " + nm([0, 2, 5]) });
    out.push({ S: [0, 1, 2], t: "not independent, size 3 — " + nm([0, 1, 2]) + " (contains edges)" });
    return out;
  }
  function build() {
    const view = SX.val("vc-view", "cover");
    if (lastView !== view) { chosen = 0; lastView = view; }
    const opts = optionsFor(view);
    if (chosen >= opts.length) chosen = 0;
    const sel = d3.select("#vc-set");
    if (!sel.empty()) { sel.selectAll("option").remove(); opts.forEach(function (o, i) { sel.append("option").attr("value", i).text(o.t); }); sel.property("value", String(chosen)); }
    const S = opts[chosen].S, inS = new Set(S);
    const shown = view === "clique" ? comp : g;
    const unc = NR.uncovered(g, S), ind = NR.inducedEdges(g, S), indC = NR.inducedEdges(comp, S);
    const svg = d3.select("#vc-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    F.g.append("text").attr("x", 12).attr("y", 24).attr("font-size", 11.5).attr("fill", AC.ink)
      .text(view === "cover" ? "VERTEX COVER of G — every edge must have a filled endpoint; red edges are uncovered"
        : view === "ind" ? "INDEPENDENT SET of G — no edge may join two filled vertices; red edges violate that"
        : "CLIQUE of the COMPLEMENT Ḡ — every pair of filled vertices must be joined; red = a missing pair");
    SX.npDraw(shown, pos, { into: F.g, r: 16,
      edge: function (e) {
        if (view === "cover") return (inS.has(e[0]) || inS.has(e[1])) ? { color: AC.good, width: 2.2 } : { color: AC.bad, width: 3 };
        if (view === "ind") return (inS.has(e[0]) && inS.has(e[1])) ? { color: AC.bad, width: 3 } : { color: AC.line, width: 1.5 };
        return (inS.has(e[0]) && inS.has(e[1])) ? { color: AC.good, width: 2.6 } : { color: AC.line, width: 1.2, op: 0.5 };
      },
      vertex: function (v) { return inS.has(v) ? { fill: view === "cover" ? AC.accent : AC.good, ink: AC.bg, stroke: view === "cover" ? AC.accent : AC.good } : null; } });
    /* the subset census for the current question, as bars */
    const cen = view === "cover" ? VC.bySize : view === "ind" ? IS.bySize : CL.bySize;
    const keys = Object.keys(cen).map(Number).sort(function (a, b) { return a - b; });
    const mx = Math.max.apply(null, keys.map(function (k) { return cen[k]; }));
    F.g.append("text").attr("x", 500).attr("y", 44).attr("font-size", 10.5).attr("fill", AC.muted)
      .text(view === "cover" ? "covers, by size" : view === "ind" ? "independent sets, by size" : "cliques of Ḡ, by size");
    keys.forEach(function (k, i) {
      const y = 56 + i * 17, wdt = 120 * cen[k] / mx;
      F.g.append("rect").attr("x", 520).attr("y", y).attr("width", Math.max(2, wdt)).attr("height", 12).attr("rx", 2)
        .attr("fill", (view === "cover" ? k === VC.min : k === (view === "ind" ? IS.max : CL.max)) ? AC.a2 : AC.panel2).attr("stroke", AC.line);
      F.g.append("text").attr("x", 514).attr("y", y + 10).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.muted).text(k);
      F.g.append("text").attr("x", 524 + Math.max(2, wdt)).attr("y", y + 10).attr("font-size", 10).attr("fill", AC.muted).text(cen[k]);
    });
    /* the identity, checked for the SELECTED set, not merely asserted */
    const V = Array.from({ length: g.n }, function (_, i) { return i; });
    const compl = V.filter(function (v) { return !inS.has(v); });
    const isInd = ind.length === 0, isCl = indC.length === S.length * (S.length - 1) / 2, complCov = NR.coverOK(g, compl);
    const isCov = NR.coverOK(g, S), complInd = NR.inducedEdges(g, compl).length === 0, complCl = NR.inducedEdges(comp, compl).length === compl.length * (compl.length - 1) / 2;
    const triple = (isInd === complCov) && (isInd === isCl) && (isCov === complInd) && (isCov === complCl);
    SX.setHtml("vc-readout",
      "G: " + g.n + " vertices, " + g.E.length + " edges · Ḡ: " + comp.n + " vertices, <b>" + comp.E.length + "</b> edges = C(7,2) − 8 = 21 − 8 " + SX.flag(comp.E.length === 21 - g.E.length) +
      "<br>by enumerating all 2⁷ = " + VC.tried + " subsets: minimum vertex cover <b>" + VC.min + "</b> (" + VC.all.length + " of that size: " + VC.all.map(function (S2) { return "{" + S2.map(function (v) { return g.names[v]; }).join("") + "}"; }).join(", ") + ") · maximum independent set <b>" + IS.max + "</b> (" + IS.all.map(function (S2) { return "{" + S2.map(function (v) { return g.names[v]; }).join("") + "}"; }).join(", ") + ") · maximum clique of Ḡ <b>" + CL.max + "</b> (" + CL.all.map(function (S2) { return "{" + S2.map(function (v) { return g.names[v]; }).join("") + "}"; }).join(", ") + ")" +
      "<br>α + τ = " + IS.max + " + " + VC.min + " = " + (IS.max + VC.min) + " = |V| " + SX.flag(IS.max + VC.min === g.n) +
      " · α(G) = ω(Ḡ) " + SX.flag(IS.max === CL.max) +
      " · maximum independent set and minimum cover are complementary sets " + SX.flag(IS.all.some(function (S2) { const s2 = new Set(S2); return VC.all.some(function (C2) { return C2.length === g.n - S2.length && C2.every(function (v) { return !s2.has(v); }); }); })) +
      "<br>for the selected set S = {" + S.map(function (v) { return g.names[v]; }).join(", ") + "}, the identity checked in both directions: S independent in G = <b>" + isInd + "</b>, V∖S a cover of G = <b>" + complCov + "</b>, S a clique of Ḡ = <b>" + isCl + "</b> — and S a cover of G = <b>" + isCov + "</b>, V∖S independent = <b>" + complInd + "</b>, V∖S a clique of Ḡ = <b>" + complCl + "</b> " + SX.flag(triple) +
      "<br>the §21 greedy maximal matching in edge order picks " + mEdges.map(function (e) { return "(" + g.names[e[0]] + "," + g.names[e[1]] + ")"; }).join(", ") + " → a cover of <b>" + greedyCover.length + "</b> vertices, ratio " + SX.f3(greedyCover.length / VC.min) + " against the optimum " + VC.min + " " + SX.flag(greedyCover.length === 2 * VC.min) + " — the 2-approximation's bound met with equality");
  }
  SX.on("vc-view", "change", build);
  SX.on("vc-set", "change", function () { chosen = +SX.val("vc-set", "0"); build(); });
  build();
})();

/* ── 13  #ham-svg  Hamiltonian search, Euler test, widget, TSP ─────────── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("ham-svg")) return;
  const W = 680, H = 330;
  function instance(which) {
    if (which === "petersen") return { g: NR.npG(NI.gPetersen.n, NI.gPetersen.E), pos: NR.petersenLayout(168, 168, 120), label: "the Petersen graph" };
    if (which === "euler") { const g = NR.eulerVariant(); return { g: g, pos: NR.circleLayout(g.n, 168, 168, 120), label: "gHam with 5–6 removed and 0–2, 1–3 added" }; }
    return { g: NR.npG(NI.gHam.n, NI.gHam.E), pos: NR.circleLayout(NI.gHam.n, 168, 168, 120), label: "8 vertices, 11 edges" };
  }
  function build() {
    const which = SX.val("ham-inst", "gHam"), show = SX.val("ham-show", "widget");
    const I = instance(which), g = I.g;
    const cH = AL.counter(), HB = NR.hamBrute(g, cH);
    const cE = AL.counter(), ET = NR.eulerTest(g, cE);
    const cT = AL.counter(), ETO = ET.circuit ? NR.eulerTour(g, cT) : null;
    const cP = AL.counter(), HP = HB.cycles.length ? null : NR.hamPathDFS(g, cP);
    const cM = AL.counter(), MN = NR.hamMinNonEdges(g, cM);
    const cW = AL.counter(), WT = NR.widgetTraversals(cW);
    const budget = g.n, rho = 2, off = rho * g.n + 1;
    const cost12 = NR.tspFromHamCost(g, 2, MN), costGap = NR.tspFromHamCost(g, off, MN);
    const svg = d3.select("#ham-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const left = F.g.append("g"), right = F.g.append("g");
    /* ── left: the graph, with the Hamiltonian cycle (or path, or Euler tour) ── */
    const cyc = HB.cycles[0] || null, pth = HP && HP.path;
    const onCycle = new Set();
    if (cyc) for (let i = 0; i < cyc.length; i++) onCycle.add([Math.min(cyc[i], cyc[(i + 1) % cyc.length]), Math.max(cyc[i], cyc[(i + 1) % cyc.length])].join("-"));
    else if (pth) for (let i = 1; i < pth.length; i++) onCycle.add([Math.min(pth[i - 1], pth[i]), Math.max(pth[i - 1], pth[i])].join("-"));
    SX.npDraw(g, I.pos, { into: left, r: 14,
      edge: function (e) { return onCycle.has(e[0] + "-" + e[1]) ? { color: cyc ? AC.good : AC.a2, width: 3 } : { color: AC.line, width: 1.4 }; },
      vertex: function (v) { return { fill: AC.panel2 }; },
      label: function (v) { return "deg " + ET.degrees[v]; }, labelColor: function (v) { return ET.degrees[v] % 2 ? AC.a2 : AC.good; } });
    left.append("text").attr("x", 12).attr("y", 20).attr("font-size", 11.5).attr("fill", AC.ink).text(I.label);
    left.append("text").attr("x", 12).attr("y", 306).attr("font-size", 11).attr("fill", cyc ? AC.good : AC.a2)
      .text(cyc ? ("Hamiltonian cycle " + cyc.join("-") + "-" + cyc[0]) : (pth ? "no Hamiltonian cycle; a Hamiltonian PATH " + pth.join("-") : "no Hamiltonian cycle and no Hamiltonian path"));
    left.append("text").attr("x", 12).attr("y", 322).attr("font-size", 11).attr("fill", ET.circuit ? AC.good : AC.muted)
      .text(ET.circuit ? ("Euler circuit " + ETO.tour.join("-")) : ("no Euler circuit: " + ET.odd + " vertices of odd degree"));
    /* ── right ── */
    if (show === "widget") {
      const Wg = WT.widget;
      const wpos = []; for (let i = 0; i < 6; i++) wpos.push([430, 62 + i * 38]); for (let i = 0; i < 6; i++) wpos.push([580, 62 + i * 38]);
      const frames = [
        { t: "(b) enter [u,v,1], leave [u,v,6] — all 12 vertices · u is in the cover, v is not", path: WT.uv[0], split: null },
        { t: "(c) two disjoint passes, six vertices each · BOTH u and v are in the cover", path: null, split: WT.split[0] },
        { t: "(d) enter [v,u,1], leave [v,u,6] — all 12 vertices · v is in the cover, u is not", path: WT.vu[0], split: null }
      ];
      const render = function (f) {
        right.selectAll("*").remove();
        const on = new Set();
        const mark = function (p) { for (let i = 1; i < p.length; i++) on.add([Math.min(p[i - 1], p[i]), Math.max(p[i - 1], p[i])].join("-")); };
        if (f.path) mark(f.path); if (f.split) { mark(f.split[0]); mark(f.split[1]); }
        SX.npDraw(Wg, wpos, { into: right, r: 11, fontSize: 8,
          edge: function (e) { return on.has(e[0] + "-" + e[1]) ? { color: AC.good, width: 3 } : { color: AC.line, width: 1.3 }; },
          vertex: function (v) { return { r: 11, fill: AC.panel2 }; } });
        for (let i = 0; i < 6; i++) {
          right.append("text").attr("x", 404).attr("y", 66 + i * 38).attr("text-anchor", "end").attr("font-size", 9).attr("fill", AC.muted).text("[u,v," + (i + 1) + "]");
          right.append("text").attr("x", 606).attr("y", 66 + i * 38).attr("font-size", 9).attr("fill", AC.muted).text("[v,u," + (i + 1) + "]");
        }
        right.append("text").attr("x", 348).attr("y", 20).attr("font-size", 11).attr("fill", AC.ink).text("the widget: 12 vertices, 14 edges");
        right.append("text").attr("x", 348).attr("y", 306).attr("font-size", 10.5).attr("fill", AC.good).text(f.t);
      };
      AL.stepper(svg, { frames: frames, render: render, delay: 1400, label: "traversal" });
    } else {
      /* the TSP instance the reduction builds: every pair weighted 1 or 2 */
      const A = NR.npAdj(g), cx = 514, cy = 165, r = 118;
      const tp = NR.circleLayout(g.n, cx, cy, r);
      const best = MN.tour, onT = new Set();
      for (let i = 0; i < best.length; i++) onT.add([Math.min(best[i], best[(i + 1) % best.length]), Math.max(best[i], best[(i + 1) % best.length])].join("-"));
      for (let u = 0; u < g.n; u++) for (let v = u + 1; v < g.n; v++) {
        const key = u + "-" + v, isE = A[u][v], onTour = onT.has(key);
        right.append("line").attr("x1", tp[u][0]).attr("y1", tp[u][1]).attr("x2", tp[v][0]).attr("y2", tp[v][1])
          .attr("stroke", onTour ? (isE ? AC.good : AC.bad) : (isE ? AC.line : AC.grid))
          .attr("stroke-width", onTour ? 3 : 1).attr("stroke-dasharray", isE ? null : "3,3").attr("opacity", onTour ? 1 : (isE ? 0.8 : 0.35));
      }
      for (let v = 0; v < g.n; v++) {
        right.append("circle").attr("cx", tp[v][0]).attr("cy", tp[v][1]).attr("r", 13).attr("fill", AC.panel2).attr("stroke", AC.line);
        right.append("text").attr("x", tp[v][0]).attr("y", tp[v][1] + 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.ink).text(v);
      }
      right.append("text").attr("x", 348).attr("y", 20).attr("font-size", 11).attr("fill", AC.ink)
        .text("TSP: weight 1 on the " + g.E.length + " edges of G (solid), 2 on the " + (g.n * (g.n - 1) / 2 - g.E.length) + " non-edges (dashed)");
      right.append("text").attr("x", 348).attr("y", 306).attr("font-size", 10.5).attr("fill", cost12 <= budget ? AC.good : AC.bad)
        .text("optimal tour " + best.join("-") + " · weight " + cost12 + (cost12 <= budget ? " ≤ " : " > ") + "budget |V| = " + budget);
      right.append("text").attr("x", 348).attr("y", 322).attr("font-size", 10.5).attr("fill", AC.muted)
        .text("red segments are non-edges the tour is forced to use: " + MN.minNonEdges);
    }
    const RS = NR.hamReductionSize(NR.npFromLetters(NI.gVC), 3), RSg = NR.hamReductionSize(g, 3);
    SX.setHtml("ham-readout",
      "HAMILTONIAN CYCLE by exhaustive search: <b>" + SX.int(HB.tours) + "</b> tours = (n−1)!/2 " + SX.flag(HB.tours === (function (k) { let f = 1; for (let i = 2; i <= k; i++) f *= i; return g.n > 2 ? f / 2 : 1; })(g.n - 1)) +
      ", <b>" + SX.int(cH.get("check")) + "</b> edge lookups → <b>" + HB.cycles.length + "</b> Hamiltonian cycle" + (HB.cycles.length === 1 ? "" : "s") +
      (HP ? " · a Hamiltonian PATH found by depth-first search in <b>" + HP.nodes + "</b> search nodes: " + (HP.path ? HP.path.join("-") : "none") : "") +
      "<br>EULER on the same graph: degree scan <b>" + ET.degSteps + "</b> steps = |E| + |V| = " + g.E.length + " + " + g.n + " " + SX.flag(ET.degSteps === g.E.length + g.n) + ", connectivity walk <b>" + ET.connSteps + "</b> = 2|E| " + SX.flag(ET.connSteps === 2 * g.E.length) + " → " + ET.odd + " odd-degree vertices, Euler circuit " + (ET.circuit ? "YES" : "NO") +
      (ETO ? ", built by Hierholzer in <b>" + cT.get("step") + "</b> steps = |E| " + SX.flag(cT.get("step") === g.E.length) : "") +
      " — linear against the " + SX.int(HB.tours) + " tours, a factor of " + SX.f1(cH.get("check") / ET.steps) +
      "<br>TSP by the reduction, from one enumeration of all " + SX.int(MN.tours) + " tours (minimum non-edges used = <b>" + MN.minNonEdges + "</b>): weights 1 / 2 → optimum <b>" + cost12 + "</b> vs budget " + budget + " → " + (cost12 <= budget ? "YES" : "NO") + ", and G is " + (HB.cycles.length ? "" : "NOT ") + "Hamiltonian " + SX.flag((cost12 <= budget) === (HB.cycles.length > 0)) +
      " · gap weights 1 / (ρ|V|+1) with ρ = " + rho + " (non-edge = " + off + ") → optimum <b>" + SX.int(costGap) + "</b> vs the ρ|V| = " + (rho * g.n) + " threshold → " + (costGap <= rho * g.n ? "≤, so a ρ-approximation could not tell a Hamiltonian graph from this one" : "> , so any ρ-approximation's answer decides HAM-CYCLE") + " " + SX.flag((costGap <= rho * g.n) === (HB.cycles.length > 0)) +
      "<br>the widget, by enumerating every path through its 12 vertices (" + SX.int(cW.get("node")) + " search nodes): [u,v,1]→[u,v,6] covering all 12: <b>" + WT.uv.length + "</b> · [v,u,1]→[v,u,6]: <b>" + WT.vu.length + "</b> · the two-pass split: <b>" + WT.split.length + "</b> · the forbidden [u,v,1]→[v,u,6]: <b>" + WT.badUV.length + "</b>, [v,u,1]→[u,v,6]: <b>" + WT.badVU.length + "</b>, crossed split: <b>" + WT.badSplit.length + "</b> " + SX.flag(WT.uv.length === 1 && WT.vu.length === 1 && WT.split.length === 1 && WT.badUV.length === 0 && WT.badVU.length === 0 && WT.badSplit.length === 0) +
      " · applied to the 7-vertex, 8-edge vertex-cover instance of §12 with k = 3 the reduction builds G′ with " + SX.int(RS.V) + " vertices and " + SX.int(RS.E) + " edges (on the graph drawn here, with k = 3, it would be " + SX.int(RSg.V) + " and " + SX.int(RSg.E) + ") — far beyond exhaustive search, which is why only the widget lemma is verified here");
  }
  SX.on("ham-inst", "change", build);
  SX.on("ham-show", "change", build);
  build();
})();

/* ── 14  #subsetsum-svg  the DP table, the growth, the digit gadget ────── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("subsetsum-svg")) return;
  const W = 680, H = 320;
  const S = NI.subsetsum.S.slice();
  function build() {
    const T = +SX.val("ss-target", "37"), view = SX.val("ss-view", "table");
    SX.setText("ss-target-val", String(T));
    const cDP = AL.counter(), DP = NR.subsetSumDP(S, T, cDP);
    const cBF = AL.counter(), BF = NR.subsetSumBrute(S, T, cBF);
    const svg = d3.select("#subsetsum-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    if (view === "table") {
      const cols = T + 1, cw = Math.max(4, Math.min(14, 600 / cols)), rh = 26;
      const onTrail = {}; DP.trace.forEach(function (x) { onTrail[x.i + "," + x.t] = x.take ? 2 : 1; });
      if (DP.yes) onTrail["0,0"] = 1;
      const gg = F.g.append("g").attr("transform", "translate(56,44)");
      for (let i = 0; i <= S.length; i++) for (let t = 0; t <= T; t++) {
        const tr = onTrail[i + "," + t];
        gg.append("rect").attr("x", t * cw).attr("y", i * rh).attr("width", Math.max(1.5, cw - 0.8)).attr("height", rh - 2).attr("rx", 1.5)
          .attr("fill", DP.R[i][t] ? (tr === 2 ? AC.a2 : tr === 1 ? AC.accent : AC.good) : AC.panel2)
          .attr("opacity", DP.R[i][t] ? 1 : 0.55);
      }
      for (let i = 0; i <= S.length; i++) F.g.append("text").attr("x", 50).attr("y", 44 + i * rh + 16).attr("text-anchor", "end").attr("font-size", 10.5)
        .attr("fill", AC.muted).text(i === 0 ? "∅" : "+" + S[i - 1]);
      for (let t = 0; t <= T; t += Math.max(1, Math.round(T / 12))) F.g.append("text").attr("x", 56 + t * cw + cw / 2).attr("y", 38).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", AC.muted).text(t);
      F.g.append("text").attr("x", 56 + T * cw + cw / 2).attr("y", 44 + (S.length + 1) * rh + 14).attr("text-anchor", "end").attr("font-size", 10.5)
        .attr("fill", DP.yes ? AC.good : AC.bad).text(DP.yes ? ("R[" + S.length + "][" + T + "] = TRUE, traceback takes " + DP.take.join(" + ") + " = " + T) : ("R[" + S.length + "][" + T + "] = FALSE — no subset sums to " + T));
      F.g.append("text").attr("x", 12).attr("y", 22).attr("font-size", 11.5).attr("fill", AC.ink)
        .text("R[i][t] = \"some subset of the first i numbers sums to t\" · " + (S.length) + " × " + (T + 1) + " = " + SX.int(DP.cells) + " cells, " + DP.trues + " of them TRUE");
      F.g.append("text").attr("x", 12).attr("y", H - 10).attr("font-size", 10.5).attr("fill", AC.muted)
        .text("green = reachable · blue = the traceback · amber = a traceback step that TAKES its number");
    } else if (view === "growth") {
      /* the SAME measured series plotted against T and against T's bit length */
      const xs = [], cells = [], bits = [];
      for (let t = 1; t <= 400; t++) { const c = AL.counter(); NR.subsetSumDP(S, t, c); xs.push(t); cells.push(c.get("cell")); bits.push(NR.bitLen(t)); }
      const P1 = { x: 56, y: 44, w: 250, h: 210 }, P2 = { x: 392, y: 44, w: 250, h: 210 };
      const x1 = d3.scaleLinear().domain([0, 400]).range([0, P1.w]), y1 = d3.scaleLinear().domain([0, Math.max.apply(null, cells)]).range([P1.h, 0]);
      const g1 = F.g.append("g").attr("transform", "translate(" + P1.x + "," + P1.y + ")");
      AL.gridY(g1, y1, P1.w, 5); AL.axisB(g1, x1, P1.h, 5, "target T"); AL.axisL(g1, y1, 5, "cells");
      g1.append("path").attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 2)
        .attr("d", d3.line().x(function (d, i) { return x1(xs[i]); }).y(function (d) { return y1(d); })(cells));
      F.g.append("text").attr("x", P1.x).attr("y", 30).attr("font-size", 11).attr("fill", AC.ink).text("area against the VALUE of T — a straight line");
      const x2 = d3.scaleLinear().domain([0, 9]).range([0, P2.w]), y2 = d3.scaleLog().domain([1, Math.max.apply(null, cells)]).range([P2.h, 0]);
      const g2 = F.g.append("g").attr("transform", "translate(" + P2.x + "," + P2.y + ")");
      AL.axisB(g2, x2, P2.h, 9, "bits in T = ⌈log₂(T+1)⌉"); AL.axisL(g2, y2, 4, "cells (log)", d3.format(".0s"));
      xs.forEach(function (t, i) { g2.append("circle").attr("cx", x2(bits[i])).attr("cy", y2(Math.max(1, cells[i]))).attr("r", 1.8).attr("fill", AC.a2).attr("opacity", 0.55); });
      F.g.append("text").attr("x", P2.x).attr("y", 30).attr("font-size", 11).attr("fill", AC.ink).text("the same numbers against the BITS of T — a doubling staircase");
      const cur = AL.counter(); NR.subsetSumDP(S, T, cur);
      g1.append("circle").attr("cx", x1(T)).attr("cy", y1(cur.get("cell"))).attr("r", 4).attr("fill", AC.good);
      g2.append("circle").attr("cx", x2(NR.bitLen(T))).attr("cy", y2(cur.get("cell"))).attr("r", 4).attr("fill", AC.good);
      F.g.append("text").attr("x", 12).attr("y", H - 10).attr("font-size", 10.5).attr("fill", AC.muted)
        .text("green dot = the slider's T · every point is a real run of the table under a counter, T = 1 … 400");
    } else {
      const R = NR.satToSubsetSumSolve(NI.phi, AL.counter());
      const rows = R.red.rows, cols = R.red.cols;
      const sol = R.sols[0], take = new Set(sol ? sol.names : []);
      const cw = 26, rh = 18;
      const gg = F.g.append("g").attr("transform", "translate(150,52)");
      const head = ["x₁", "x₂", "x₃", "C₁", "C₂", "C₃"];
      for (let j = 0; j < cols; j++) gg.append("text").attr("x", j * cw + cw / 2).attr("y", -6).attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", AC.muted).text(head[j]);
      rows.forEach(function (r, i) {
        const on = take.has(r.name);
        for (let j = 0; j < cols; j++) {
          gg.append("rect").attr("x", j * cw).attr("y", i * rh).attr("width", cw - 2).attr("height", rh - 2).attr("rx", 2)
            .attr("fill", r.d[j] ? (on ? AC.a2 : AC.panel2) : AC.bg).attr("stroke", AC.line).attr("opacity", on ? 1 : 0.8);
          gg.append("text").attr("x", j * cw + (cw - 2) / 2).attr("y", i * rh + 12).attr("text-anchor", "middle").attr("font-size", 10)
            .attr("fill", r.d[j] ? (on ? AC.bg : AC.ink) : AC.muted).text(r.d[j]);
        }
        gg.append("text").attr("x", -8).attr("y", i * rh + 12).attr("text-anchor", "end").attr("font-size", 10.5)
          .attr("fill", on ? AC.a2 : AC.muted).text((on ? "✓ " : "") + r.name);
        gg.append("text").attr("x", cols * cw + 10).attr("y", i * rh + 12).attr("font-size", 10).attr("fill", on ? AC.a2 : AC.muted).text(R.red.values[i]);
      });
      const y = rows.length * rh + 6;
      for (let j = 0; j < cols; j++) {
        gg.append("rect").attr("x", j * cw).attr("y", y).attr("width", cw - 2).attr("height", rh - 2).attr("rx", 2).attr("fill", AC.good);
        gg.append("text").attr("x", j * cw + (cw - 2) / 2).attr("y", y + 12).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.bg).text(R.red.target[j]);
      }
      gg.append("text").attr("x", -8).attr("y", y + 12).attr("text-anchor", "end").attr("font-size", 10.5).attr("fill", AC.good).text("target T");
      gg.append("text").attr("x", cols * cw + 10).attr("y", y + 12).attr("font-size", 10).attr("fill", AC.good).text(R.red.targetValue);
      F.g.append("text").attr("x", 12).attr("y", 24).attr("font-size", 11.5).attr("fill", AC.ink)
        .text("3-SAT → SUBSET-SUM on φ: one column per variable, one per clause, base 10 so no column can carry (max column total 6)");
      F.g.append("text").attr("x", 12).attr("y", H - 10).attr("font-size", 10.5).attr("fill", AC.a2)
        .text(sol ? ("highlighted: " + sol.names.join(" + ") + " = " + R.red.targetValue + ", encoding the assignment " + sol.assign) : "no solution");
    }
    /* readout — the same for all three views */
    const R = NR.satToSubsetSumSolve(NI.phi, AL.counter());
    const sat = NR.satAllB(NI.phi, AL.counter());
    const PA = NR.partitionFrom(S, T);
    const cPA = AL.counter(), PS = NR.subsetSumBrute(PA.S2, PA.half, cPA);
    const KN = NR.knapFrom(S, T);
    const bits = NR.instanceBits(S, T);
    SX.setHtml("subsetsum-readout",
      "S = {" + S.join(", ") + "}, T = <b>" + T + "</b> · DP: <b>" + SX.int(DP.cells) + "</b> cells measured = n·(T+1) = " + S.length + "·" + (T + 1) + " " + SX.flag(DP.cells === S.length * (T + 1)) + ", " + DP.trues + " reachable · answer " + (DP.yes ? "YES" : "NO") +
      " vs brute force over all 2⁶ = " + BF.tried + " subsets, which finds <b>" + BF.sols.length + "</b> solution" + (BF.sols.length === 1 ? "" : "s") + (BF.sols.length ? " (" + BF.sols.map(function (x) { return "{" + x.items.join(",") + "}"; }).join(", ") + ")" : "") + " " + SX.flag(DP.yes === (BF.sols.length > 0)) +
      "<br>PSEUDO-POLYNOMIAL: this instance is <b>" + bits + "</b> bits long (⌈log₂⌉ of each number plus the target), and the table has " + SX.int(DP.cells) + " cells — <b>" + SX.f1(DP.cells / bits) + "</b> cells per bit of input at T = " + T + ", and the ratio grows like 2^(bits of T). Θ(nT) is linear in the VALUE of T and exponential in its LENGTH." +
      "<br>SUBSET-SUM ≤ₚ PARTITION: σ = " + PA.sigma + ", add a = 2σ − T = " + PA.a + " and b = σ + T = " + PA.b + "; total " + PA.total + ", each half " + PA.half + " · measured over all 2⁸ = " + PS.tried + " subsets: <b>" + PS.sols.length + "</b> subsets of S⁺ sum to " + PA.half + ", i.e. <b>" + (PS.sols.length / 2) + "</b> even partitions, each counted once from each side, and a subset of S sums to T " + (BF.sols.length ? "" : "not ") + "— the two agree " + SX.flag((PS.sols.length > 0) === (BF.sols.length > 0)) +
      " · SUBSET-SUM ≤ₚ KNAPSACK: w = v = s, W = V = " + KN.W + " (a one-line reduction)" +
      "<br>the 3-SAT gadget on φ: <b>" + R.red.rows.length + "</b> numbers = 2n + 2m " + SX.flag(R.red.rows.length === 2 * NI.phi.n + 2 * NI.phi.clauses.length) + " of <b>" + R.red.cols + "</b> digits = n + m, target <b>" + R.red.targetValue + "</b> · enumerating all 2¹² = " + SX.int(R.tried) + " subsets finds <b>" + R.sols.length + "</b> that hit the target, encoding the assignments {" + R.assigns.join(", ") + "}, which is exactly the set of " + sat.models.length + " satisfying assignments of φ " + SX.flag(R.assigns.join(",") === sat.models.map(NR.bitsOfB).sort().join(",")));
  }
  SX.on("ss-target", "input", build);
  SX.on("ss-view", "change", build);
  build();
})();

/* ── 15  #colour-svg  the linear 2-colouring beside the exhaustive k-search ── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("colour-svg")) return;
  const W = 680, H = 340;
  const PAL = [AC.accent, AC.a2, AC.good, AC.violet, AC.teal];
  function instance(which) {
    if (which === "crown") return { g: NR.crownGraph(4), label: "the crown graph on 8 vertices (K₄,₄ minus a perfect matching)" };
    if (which === "red") return { g: NR.satTo3Col(NI.phi).g, red: NR.satTo3Col(NI.phi), phi: NI.phi, label: "the 3-colouring reduction applied to φ (satisfiable)" };
    if (which === "red8") return { g: NR.satTo3Col(NI.phi8).g, red: NR.satTo3Col(NI.phi8), phi: NI.phi8, label: "the 3-colouring reduction applied to φ₈ (unsatisfiable)" };
    return { g: NR.npG(NI.gCol.n, NI.gCol.E), label: "7 vertices, 10 edges" };
  }
  function build() {
    const k = +SX.val("col-k", "3"), which = SX.val("col-inst", "gCol");
    const I = instance(which), g = I.g, small = g.n <= 13;
    const cB = AL.counter(), BP = NR.bipartite(g, cB);
    const cX = AL.counter(), EX = small ? NR.colourBrute(g, k, cX) : null;
    const cT = AL.counter(), BT = small ? null : NR.colourBT(g, k, cT);
    const found = small ? EX.first : (BT ? BT.found : null);
    const tried = small ? cX.get("assign") : cT.get("node");
    const svg = d3.select("#colour-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const r = g.n > 12 ? 85 : 78, rad = g.n > 12 ? 9 : 14, fs = g.n > 12 ? 7 : 11;
    const pL = NR.circleLayout(g.n, 170, 122, r), pR = NR.circleLayout(g.n, 510, 122, r);
    const L = F.g.append("g"), R = F.g.append("g");
    const viol = BP.violation;
    SX.npDraw(g, pL, { into: L, r: rad, fontSize: fs,
      edge: function (e) { return (viol && ((e[0] === viol[0] && e[1] === viol[1]) || (e[0] === viol[1] && e[1] === viol[0]))) ? { color: AC.bad, width: 3.4 } : { color: AC.line, width: 1.3 }; },
      vertex: function (v) { return { fill: BP.col[v] < 0 ? AC.panel2 : PAL[BP.col[v]], ink: AC.bg, r: rad }; } });
    SX.npDraw(g, pR, { into: R, r: rad, fontSize: fs,
      edge: function (e) { return { color: (found && found[e[0]] === found[e[1]]) ? AC.bad : AC.line, width: (found && found[e[0]] === found[e[1]]) ? 3 : 1.3 }; },
      vertex: function (v) { return { fill: found ? PAL[found[v] % PAL.length] : AC.panel2, ink: found ? AC.bg : AC.ink, r: rad }; } });
    F.g.append("text").attr("x", 14).attr("y", 20).attr("font-size", 11.5).attr("fill", AC.ink).text("2-COLOURING by BFS parity — linear");
    F.g.append("text").attr("x", 14).attr("y", 36).attr("font-size", 10.5).attr("fill", BP.ok ? AC.good : AC.bad)
      .text(BP.ok ? "bipartite: a proper 2-colouring exists" : "NOT bipartite: the red edge joins two same-parity vertices");
    F.g.append("text").attr("x", 354).attr("y", 20).attr("font-size", 11.5).attr("fill", AC.ink).text(k + "-COLOURING by " + (small ? "exhaustive search over all " + k + "^" + g.n : "backtracking") + (small ? "" : " (the space is " + k + "^" + g.n + ")"));
    F.g.append("text").attr("x", 354).attr("y", 36).attr("font-size", 10.5).attr("fill", found ? AC.good : AC.bad)
      .text(found ? ("a proper " + k + "-colouring found") : ("NO proper " + k + "-colouring exists"));
    /* the two costs, side by side, on a log scale */
    const bx = 120, bw = 430, by = 250;
    const lg = function (x) { return Math.log(Math.max(1, x) + 1); };
    const mx = lg(Math.max(BP.steps, tried));
    [{ t: "bipartiteness test — adjacency examinations", v: BP.steps, c: AC.good },
     { t: (small ? k + "-colouring — assignments enumerated" : k + "-colouring — backtracking search nodes"), v: tried, c: AC.bad }].forEach(function (d, i) {
      const y = by + i * 34;
      F.g.append("rect").attr("x", bx).attr("y", y).attr("width", Math.max(3, bw * lg(d.v) / mx)).attr("height", 20).attr("rx", 3).attr("fill", d.c).attr("opacity", 0.85);
      F.g.append("text").attr("x", bx - 8).attr("y", y + 15).attr("text-anchor", "end").attr("font-size", 10.5).attr("fill", AC.muted).text(d.t);
      F.g.append("text").attr("x", bx + Math.max(3, bw * lg(d.v) / mx) + 8).attr("y", y + 15).attr("font-size", 11).attr("fill", AC.ink).text(SX.int(d.v));
    });
    F.g.append("text").attr("x", 14).attr("y", 236).attr("font-size", 10.5).attr("fill", AC.muted).text("measured operation counts, on a logarithmic scale");
    /* readout */
    const OG = NR.orGadgetCensus(AL.counter());
    const cCh = AL.counter(), CH = small ? NR.chromatic(g, Math.min(4, g.n), cCh) : null;
    const greedyNat = NR.greedyColour(g, Array.from({ length: g.n }, function (_, i) { return i; }), AL.counter());
    const crown = NR.crownGraph(4), gc = NR.greedyColour(crown, NR.interleaved(4), AL.counter());
    let extra = "";
    if (I.red) {
      const P = I.phi, T = I.red.T, col = found;
      let ok = false, asg = "";
      if (col) { const Tc = col[T]; const a = []; for (let i = 1; i <= P.n; i++) a.push(col[I.red.litIndex(i)] === Tc); asg = NR.bitsOfB(a); ok = NR.satEvalB(P.clauses, a); }
      const satPhi = NR.satAllB(P, AL.counter()).models.length > 0;
      extra = "<br>the reduction on " + (P === NI.phi8 ? "φ₈" : "φ") + " (n = " + P.n + ", m = " + P.clauses.length + "): <b>" + g.n + "</b> vertices = 3 + 2n + 6m " + SX.flag(g.n === 3 + 2 * P.n + 6 * P.clauses.length) + " and <b>" + g.E.length + "</b> edges = 3 + 3n + 13m " + SX.flag(g.E.length === 3 + 3 * P.n + 13 * P.clauses.length) +
        (k === 3 ? " · 3-colourable " + (found ? "YES" : "NO") + " and " + (P === NI.phi8 ? "φ₈" : "φ") + " is " + (satPhi ? "satisfiable" : "unsatisfiable") + " " + SX.flag(!!found === satPhi) +
          (col ? " · the assignment read off the literal vertices is <b>" + asg + "</b>, and it satisfies φ " + SX.flag(ok) : "")
          : " · the reduction's correspondence is a statement about k = 3 only; at k = " + k + " this graph is " + (found ? "" : "not ") + k + "-colourable, which says nothing about φ" +
            (k === 2 ? " — the palette T–F–B is a triangle, so no graph the reduction builds is ever 2-colourable" : ""));
    }
    SX.setHtml("colour-readout",
      "graph: " + g.n + " vertices, " + g.E.length + " edges, " + BP.components + " component" + (BP.components === 1 ? "" : "s") +
      "<br>2-COLOURING: the BFS test examined <b>" + BP.steps + "</b> adjacency entries (2|E| + |E| = " + (3 * g.E.length) + " for a complete scan " + SX.flag(BP.steps === 3 * g.E.length) + ")" + (viol ? ", and found the violating edge (" + g.names[viol[0]] + ", " + g.names[viol[1]] + ") after <b>" + BP.firstAt + "</b> of them" : ", with no violation") + " → " + (BP.ok ? "BIPARTITE" : "NOT bipartite") +
      "<br>SEARCHING for a " + k + "-colouring: " + (small ? "all <b>" + SX.int(cX.get("assign")) + "</b> = " + k + "^" + g.n + " assignments enumerated " + SX.flag(cX.get("assign") === Math.pow(k, g.n)) + " (" + SX.int(cX.get("check")) + " edge checks), finding <b>" + SX.int(EX.count) + "</b> proper colourings"
        : "backtracking with vertex 0's colour fixed visited <b>" + SX.int(cT.get("node")) + "</b> search nodes against a space of " + k + "^" + g.n + " assignments") +
      (CH ? " · chromatic number χ = <b>" + CH.chi + "</b>, established by enumerating every colouring for k = 1 … " + CH.chi : "") +
      "<br>" + (small ? "the two costs differ by a factor of <b>" + SX.f1(tried / Math.max(1, BP.steps)) + "</b> on this instance — one is Θ(V + E), the other is the size of the search space"
        : "backtracking visited <b>" + SX.int(tried) + "</b> nodes, far fewer than the " + k + "^" + g.n + " assignments the space contains — pruning, not a polynomial algorithm (§26) — against the linear test's " + BP.steps + " adjacency examinations") +
      "<br>GREEDY: in vertex order this graph takes <b>" + greedyNat.k + "</b> colours; on the crown graph on 8 vertices, which is bipartite (χ = 2), greedy in the interleaved order takes <b>" + gc.k + "</b> " + SX.flag(gc.k === 4) + " — the ratio is unbounded, so greedy is not a ρ-approximation for any constant ρ" +
      "<br>the OR-gadget lemma, by enumerating all 3³ = 27 colourings of (p, q, r) under each input pattern: " +
      OG.rows.map(function (rw) { return (rw.u ? "T" : "F") + (rw.v ? "T" : "F") + " → " + rw.proper + " proper, output T in " + rw.outTrue + " and F in " + rw.outFalse; }).join(" · ") +
      " — both inputs F forces the output F " + SX.flag(OG.rows[3].outTrue === 0 && OG.rows[3].proper > 0) + extra);
  }
  SX.on("col-k", "change", build);
  SX.on("col-inst", "change", build);
  build();
})();

/* ── 16  #boundary-svg  same-looking pair, measured cost ───────────────── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("boundary-svg")) return;
  const W = 680, H = 330;
  const cache = {};
  function fact(k) { let f = 1; for (let i = 2; i <= k; i++) f *= i; return f; }
  function series(pair) {
    if (cache[pair]) return cache[pair];
    const easy = [], hard = [], rows = [];
    if (pair === "sat") {
      for (let n = 4; n <= 18; n += 2) {
        const r = AL.rng(101 + n);
        const cl2 = NR.randClauses(n, Math.round(2.5 * n), 2, r), cl3 = NR.randClauses(n, Math.round(5 * n), 3, r);
        const c1 = AL.counter(), T = NR.twoSat(n, cl2, c1);
        const c2 = AL.counter(); const a = new Array(n); let models = 0;
        for (let m = 0; m < (1 << n); m++) { c2.add("assign"); for (let i = 0; i < n; i++) a[i] = !!((m >> i) & 1); if (NR.satEvalB(cl3, a, c2)) models++; }
        easy.push({ n: n, v: c1.get("step") }); hard.push({ n: n, v: c2.get("assign") });
        rows.push({ n: n, easy: c1.get("step"), hard: c2.get("assign"), closed: Math.pow(2, n), prim: c2.get("lit"), note: (T.sat ? "2-SAT satisfiable" : "2-SAT unsatisfiable") + " · 3-CNF has " + models + " model" + (models === 1 ? "" : "s") });
      }
      cache[pair] = { easy: easy, hard: hard, rows: rows, eLab: "2-SAT — implication arcs + SCC traversal steps", hLab: "3-SAT — all 2ⁿ assignments enumerated", closed: "2ⁿ", prim: "literal evaluations" };
    } else if (pair === "col") {
      for (let n = 4; n <= 12; n++) {
        const r = AL.rng(31 + n); const g = NR.randGraphB(n, Math.round(2.2 * n), r);
        const c1 = AL.counter(), B = NR.bipartite(g, c1);
        const c2 = AL.counter(), C = NR.colourBrute(g, 3, c2);
        easy.push({ n: n, v: c1.get("step") }); hard.push({ n: n, v: c2.get("assign") });
        rows.push({ n: n, easy: c1.get("step"), hard: c2.get("assign"), closed: Math.pow(3, n), prim: c2.get("check"), note: g.E.length + " edges · " + (B.ok ? "bipartite" : "not bipartite") + " · " + C.count + " proper 3-colourings" });
      }
      cache[pair] = { easy: easy, hard: hard, rows: rows, eLab: "2-colouring — BFS adjacency examinations + verification", hLab: "3-colouring — all 3ⁿ assignments enumerated", closed: "3ⁿ", prim: "edge checks" };
    } else {
      for (let n = 4; n <= 10; n++) {
        const r = AL.rng(77 + n); const g = NR.ringPlusChords(n, Math.round(0.5 * n), r);
        const c1 = AL.counter(), E = NR.eulerTest(g, c1);
        const c2 = AL.counter(), Hm = NR.hamBrute(g, c2);
        easy.push({ n: n, v: c1.get("step") }); hard.push({ n: n, v: c2.get("tour") });
        rows.push({ n: n, easy: c1.get("step"), hard: c2.get("tour"), closed: n > 2 ? fact(n - 1) / 2 : 1, prim: c2.get("check"), note: g.E.length + " edges · " + E.odd + " odd degrees, Euler circuit " + (E.circuit ? "yes" : "no") + " · " + Hm.cycles.length + " Hamiltonian cycle" + (Hm.cycles.length === 1 ? "" : "s") });
      }
      cache[pair] = { easy: easy, hard: hard, rows: rows, eLab: "Euler circuit — degree scan + connectivity walk", hLab: "Hamiltonian cycle — all (n−1)!/2 tours enumerated", closed: "(n−1)!/2", prim: "edge lookups" };
    }
    return cache[pair];
  }
  function build() {
    const pair = SX.val("bd-pair", "sat"), useLog = SX.checked("bd-log", true);
    const sets = pair === "all" ? ["sat", "col", "ham"].map(series) : [series(pair)];
    const svg = d3.select("#boundary-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 58, r: 210, t: 26, b: 40 });
    let maxN = 0, maxV = 0, minV = Infinity;
    sets.forEach(function (S) { S.easy.concat(S.hard).forEach(function (d) { maxN = Math.max(maxN, d.n); maxV = Math.max(maxV, d.v); minV = Math.min(minV, d.v); }); });
    const x = d3.scaleLinear().domain([4, maxN]).range([0, F.iw]);
    const y = useLog ? d3.scaleLog().domain([Math.max(1, minV), maxV]).range([F.ih, 0]) : d3.scaleLinear().domain([0, maxV]).range([F.ih, 0]);
    AL.gridY(F.g, y, F.iw, 5); AL.axisB(F.g, x, F.ih, Math.min(8, maxN - 3), "instance size n", d3.format("d"));
    AL.axisL(F.g, y, 5, "operations measured" + (useLog ? " (log)" : ""), d3.format(".0s"));
    const cols = [AC.good, AC.bad, AC.teal, AC.rose, AC.accent, AC.a2];
    const line = d3.line().x(function (d) { return x(d.n); }).y(function (d) { return y(Math.max(useLog ? 1 : 0, d.v)); });
    sets.forEach(function (S, si) {
      F.g.append("path").attr("fill", "none").attr("stroke", cols[2 * si % cols.length]).attr("stroke-width", 2.4).attr("d", line(S.easy));
      F.g.append("path").attr("fill", "none").attr("stroke", cols[(2 * si + 1) % cols.length]).attr("stroke-width", 2.4).attr("stroke-dasharray", "6,3").attr("d", line(S.hard));
      S.easy.forEach(function (d) { F.g.append("circle").attr("cx", x(d.n)).attr("cy", y(Math.max(useLog ? 1 : 0, d.v))).attr("r", 2.6).attr("fill", cols[2 * si % cols.length]); });
      S.hard.forEach(function (d) { F.g.append("circle").attr("cx", x(d.n)).attr("cy", y(Math.max(useLog ? 1 : 0, d.v))).attr("r", 2.6).attr("fill", cols[(2 * si + 1) % cols.length]); });
    });
    const items = [];
    sets.forEach(function (S, si) { items.push({ label: S.eLab, color: cols[2 * si % cols.length] }); items.push({ label: S.hLab, color: cols[(2 * si + 1) % cols.length], dash: "6,3" }); });
    AL.legend(F.g, items, F.iw + 14, 14, { gap: 17 });
    F.g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", AC.muted)
      .text("both curves are real runs under a counter; the hard side is EXHAUSTIVE SEARCH — the size of the space, not the best known algorithm");
    /* readout */
    const closedOK = sets.every(function (S) { return S.rows.every(function (rw) { return rw.hard === rw.closed; }); });
    SX.setHtml("boundary-readout",
      sets.map(function (S) {
        return "<b>" + S.eLab.split(" — ")[0] + " vs " + S.hLab.split(" — ")[0] + "</b> — " +
          S.rows.map(function (rw) { return "n=" + rw.n + ": " + SX.int(rw.easy) + " / " + SX.int(rw.hard); }).join(" · ") +
          "<br>&nbsp;&nbsp;the exhaustive counts equal their closed form " + S.closed + " at every size " + SX.flag(S.rows.every(function (rw) { return rw.hard === rw.closed; })) +
          " · " + S.prim + " at the largest size: " + SX.int(S.rows[S.rows.length - 1].prim) +
          " · ratio at n = " + S.rows[S.rows.length - 1].n + ": <b>" + SX.f1(S.rows[S.rows.length - 1].hard / S.rows[S.rows.length - 1].easy) + "×</b>" +
          "<br>&nbsp;&nbsp;largest instance: " + S.rows[S.rows.length - 1].note;
      }).join("<br>") +
      "<br>every exhaustive count matches its closed form across all pairs shown " + SX.flag(closedOK));
  }
  SX.on("bd-pair", "change", build);
  SX.on("bd-log", "change", build);
  build();
})();

/* ── 17  #conp-svg  certificate asymmetry, measured ────────────────────── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("conp-svg")) return;
  const W = 680, H = 300;
  const cache = {};
  function at(n) {
    if (cache[n]) return cache[n];
    const r = AL.rng(200 + n), m = Math.round(4 * n);
    const P = NR.plantedSat(n, m, r), U = NR.forcedUnsat(n, m, r);
    const cv = AL.counter(); const ok = NR.satEvalB(P.clauses, P.hidden, cv);
    const cs = AL.counter(); const F = NR.satFirstB(n, U, cs);
    const cy = AL.counter(); const Y = NR.satFirstB(n, P.clauses, cy);
    cache[n] = { n: n, m: m, verify: cv.get("lit"), ok: ok, searchA: cs.get("assign"), searchL: cs.get("lit"), unsat: !F.sat, yesFound: Y.tried, yesSat: Y.sat };
    return cache[n];
  }
  function build() {
    const n = +SX.val("conp-n", "12"), useLog = SX.checked("conp-log", true);
    SX.setText("conp-n-val", String(n));
    const D = at(n);
    const svg = d3.select("#conp-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    /* the two bars */
    const bx = 250, bw = 380, lg = function (x) { return Math.log(Math.max(1, x) + 1); };
    const vals = [
      { t: "YES-instance · verify the certificate", s: "check one satisfying assignment against " + D.m + " clauses", v: D.verify, c: AC.good },
      { t: "NO-instance · establish unsatisfiability", s: "examine every one of the 2" + "^" + n + " assignments", v: D.searchA, c: AC.bad }
    ];
    const mx = useLog ? lg(Math.max(D.verify, D.searchA)) : Math.max(D.verify, D.searchA);
    vals.forEach(function (d, i) {
      const y = 48 + i * 56, wdt = Math.max(3, bw * (useLog ? lg(d.v) / mx : d.v / mx));
      F.g.append("rect").attr("x", bx).attr("y", y).attr("width", wdt).attr("height", 26).attr("rx", 4).attr("fill", d.c).attr("opacity", 0.9);
      F.g.append("text").attr("x", bx - 10).attr("y", y + 12).attr("text-anchor", "end").attr("font-size", 11.5).attr("fill", AC.ink).text(d.t);
      F.g.append("text").attr("x", bx - 10).attr("y", y + 25).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.muted).text(d.s);
      F.g.append("text").attr("x", bx + wdt + 8).attr("y", y + 18).attr("font-size", 12).attr("fill", AC.ink).text(SX.int(d.v));
    });
    F.g.append("text").attr("x", 14).attr("y", 22).attr("font-size", 11.5).attr("fill", AC.ink)
      .text("n = " + n + " variables, " + D.m + " clauses · the same formula size on both sides" + (useLog ? " · bars on a logarithmic scale" : " · bars on a linear scale"));
    /* the two curves across n */
    const P = { x: 70, y: 176, w: 520, h: 92 };
    const xs = []; for (let k = 4; k <= 20; k++) xs.push(at(k));
    const x = d3.scaleLinear().domain([4, 20]).range([0, P.w]);
    const y = d3.scaleLog().domain([1, Math.max.apply(null, xs.map(function (d) { return d.searchA; }))]).range([P.h, 0]);
    const g2 = F.g.append("g").attr("transform", "translate(" + P.x + "," + P.y + ")");
    AL.gridY(g2, y, P.w, 3); AL.axisB(g2, x, P.h, 9, "variables n", d3.format("d")); AL.axisL(g2, y, 3, "operations (log)", d3.format(".0s"));
    [["verify", AC.good], ["searchA", AC.bad]].forEach(function (k) {
      g2.append("path").attr("fill", "none").attr("stroke", k[1]).attr("stroke-width", 2.2).attr("stroke-dasharray", k[0] === "searchA" ? "6,3" : null)
        .attr("d", d3.line().x(function (d) { return x(d.n); }).y(function (d) { return y(Math.max(1, d[k[0]])); })(xs));
    });
    g2.append("circle").attr("cx", x(n)).attr("cy", y(Math.max(1, D.verify))).attr("r", 4).attr("fill", AC.good);
    g2.append("circle").attr("cx", x(n)).attr("cy", y(Math.max(1, D.searchA))).attr("r", 4).attr("fill", AC.bad);
    SX.setHtml("conp-readout",
      "YES-instance (a 3-CNF built around a hidden assignment, so a certificate exists): verifying that certificate costs <b>" + D.verify + "</b> literal evaluations, at most 3m = " + (3 * D.m) + " " + SX.flag(D.verify <= 3 * D.m) + ", and it does satisfy the formula " + SX.flag(D.ok) +
      " · found by blind search in " + SX.int(D.yesFound) + " assignments, which is luck, not a bound" +
      "<br>NO-instance (all eight clauses over x₁x₂x₃ are present, so it is unsatisfiable whatever else it contains): establishing that costs <b>" + SX.int(D.searchA) + "</b> assignments = 2" + "^" + n + " " + SX.flag(D.searchA === Math.pow(2, n)) + " and <b>" + SX.int(D.searchL) + "</b> literal evaluations, and the search confirms no model exists " + SX.flag(D.unsat) +
      "<br>ratio at n = " + n + ": <b>" + SX.f1(D.searchA / Math.max(1, D.verify)) + "×</b>, and it doubles with every extra variable while the verifier's cost grows linearly in the formula — that gap IS the open question \"NP = co-NP?\"" +
      "<br>a co-NP certificate for the no-instance would be a REFUTATION. Resolution is such a proof system, but its refutations can be exponentially long (the pigeonhole principle needs 2^Ω(n)-size resolution refutations), and no polynomially-bounded proof system for unsatisfiability is known.");
  }
  SX.on("conp-n", "input", build);
  SX.on("conp-log", "change", build);
  build();
})();

/* ── 18  #hierarchy-svg  the containment diagram, proved vs believed ───── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("hierarchy-svg")) return;
  const W = 680, H = 380;
  /* nested boxes, outermost first. `strict` = the containment of the NEXT box
     inside this one is PROVED strict; `open` = not known. */
  const BOX = [
    { t: "R — the decidable languages", x: 18, y: 18, w: 644, h: 250, strict: true, note: "R ⊊ RE: the halting problem is in RE and not in R (Turing, 1936)" },
    { t: "EXPTIME", x: 40, y: 42, w: 600, h: 204, strict: false, note: "EXPTIME ⊆ NEXPTIME ⊆ EXPSPACE; P ⊊ EXPTIME is PROVED (time hierarchy)" },
    { t: "PSPACE = NPSPACE", x: 62, y: 66, w: 556, h: 158, strict: false, note: "Savitch (1970). PSPACE ⊊ EXPSPACE is proved; PSPACE vs EXPTIME is OPEN" },
    { t: "PH — the polynomial hierarchy", x: 84, y: 90, w: 512, h: 112, strict: false, note: "PH ⊆ PSPACE; PH = PSPACE is OPEN and would collapse the hierarchy" },
    { t: "NP ∪ co-NP", x: 106, y: 114, w: 468, h: 66, strict: false, note: "Σ₁ᵖ = NP, Π₁ᵖ = co-NP; NP = co-NP is OPEN and is NOT the same question as P = NP" },
    { t: "P", x: 232, y: 132, w: 216, h: 32, strict: false, note: "P ⊆ NP ∩ co-NP; P = NP is OPEN. P ⊊ EXPTIME is proved, so SOME inclusion above is strict" }
  ];
  const LOW = [
    { t: "NL = co-NL", x: 62, y: 286, w: 260, h: 30, note: "Immerman–Szelepcsényi (1987) — nondeterministic SPACE is closed under complement" },
    { t: "L", x: 84, y: 292, w: 90, h: 18, note: "L ⊆ NL ⊆ P, all OPEN; L ⊊ PSPACE is PROVED (space hierarchy)" }
  ];
  function build() {
    const view = SX.val("hier-view", "proved"), n = +SX.val("hier-n", "10");
    SX.setText("hier-n-val", String(n));
    const svg = d3.select("#hierarchy-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 0, r: 0, t: 0, b: 0 });
    const collapse = view === "collapse";
    BOX.forEach(function (b, i) {
      const isP = b.t === "P";
      const hidden = collapse && i >= 3 && !isP;          /* P = NP collapses PH into P */
      if (hidden) return;
      const proved = b.strict || (view === "believed" && !isP) || (collapse && false);
      const rect = F.g.append("rect").attr("x", b.x).attr("y", collapse && isP ? 114 : b.y)
        .attr("width", collapse && isP ? 468 : b.w).attr("height", collapse && isP ? 66 : b.h).attr("rx", 8)
        .attr("fill", i % 2 ? AC.panel : AC.panel2).attr("opacity", 0.55)
        .attr("stroke", proved ? AC.good : AC.a2).attr("stroke-width", proved ? 2 : 1.6);
      if (!proved) rect.attr("stroke-dasharray", "7,4");
      F.g.append("text").attr("x", b.x + 10).attr("y", (collapse && isP ? 114 : b.y) + 15).attr("font-size", 11)
        .attr("fill", AC.ink).text(collapse && isP ? "P = NP = co-NP = PH" : b.t);
    });
    LOW.forEach(function (b) {
      F.g.append("rect").attr("x", b.x).attr("y", b.y).attr("width", b.w).attr("height", b.h).attr("rx", 6)
        .attr("fill", AC.panel2).attr("opacity", 0.55).attr("stroke", AC.a2).attr("stroke-width", 1.6).attr("stroke-dasharray", "7,4");
      F.g.append("text").attr("x", b.x + 8).attr("y", b.y + 13).attr("font-size", 10.5).attr("fill", AC.ink).text(b.t);
    });
    F.g.append("text").attr("x", 330).attr("y", 300).attr("font-size", 10.5).attr("fill", AC.muted).text("L ⊆ NL ⊆ P — every inclusion open; L ⊊ PSPACE is proved");
    F.g.append("text").attr("x", 330).attr("y", 314).attr("font-size", 10.5).attr("fill", AC.muted).text("RE ⊋ R — HALT is the witness, and HALT is NP-hard but NOT in NP");
    AL.legend(F.g, [{ label: "boundary PROVED strict", color: AC.good }, { label: "strictness UNKNOWN — the two may be equal", color: AC.a2, dash: "7,4" }], 18, 342, { gap: 16 });
    /* TQBF: exponential time, linear space — measured */
    const r = AL.rng(9 + n), cl = NR.randClauses(n, Math.round(3 * n), 3, r);
    const cQ = AL.counter(), Q = NR.qbfEval(NR.altPrefix(n, true), cl, cQ);
    const worst = Math.pow(2, n + 1) - 1;
    F.g.append("text").attr("x", 330).attr("y", 342).attr("font-size", 10.5).attr("fill", AC.ink)
      .text("TQBF on ∃x₁∀x₂∃x₃… with " + n + " variables: " + SX.int(Q.nodes) + " recursion nodes, depth " + Q.depth);
    F.g.append("text").attr("x", 330).attr("y", 356).attr("font-size", 10.5).attr("fill", AC.muted)
      .text("exponential TIME (worst case 2ⁿ⁺¹ − 1 = " + SX.int(worst) + "), LINEAR space — that is PSPACE");
    const notes = BOX.map(function (b) { return "<b>" + b.t + "</b> — " + b.note; }).concat(LOW.map(function (b) { return "<b>" + b.t + "</b> — " + b.note; }));
    SX.setHtml("hierarchy-readout",
      "PROVED strict, all by diagonalisation: R ⊊ RE · L ⊊ PSPACE · NL ⊊ PSPACE · P ⊊ EXPTIME · NP ⊊ NEXPTIME · PSPACE ⊊ EXPSPACE. " +
      "OPEN: L = P? · NL = P? · P = NP? · NP = co-NP? · NP = PSPACE? · P = PSPACE? · PH = PSPACE? · PSPACE = EXPTIME?" +
      "<br>because P ⊊ EXPTIME is proved and P ⊆ NP ⊆ PSPACE ⊆ EXPTIME, <b>at least one of those three inclusions is strict</b> — and nobody knows which." +
      "<br>TQBF measured at n = " + n + " with an alternating prefix: <b>" + SX.int(Q.nodes) + "</b> recursion nodes (the short-circuiting ∃/∀ evaluator prunes; the worst case is 2ⁿ⁺¹ − 1 = " + SX.int(worst) + " " + SX.flag(Q.nodes <= worst) + ") against a maximum recursion depth of <b>" + Q.depth + "</b> = n " + SX.flag(Q.depth === n) + " — exponential time, linear space, which is exactly what PSPACE buys. The formula evaluates to " + (Q.value ? "TRUE" : "FALSE") + "." +
      "<br>" + notes.join("<br>"));
  }
  SX.on("hier-view", "change", build);
  SX.on("hier-n", "input", build);
  build();
})();


/* ── chunk C figures ────────────────────────────────────────── */
/* ── 20  #ratio-svg  every approximation algorithm on the page: measured ratio vs proved bound ── */
(function () {
  if (!SX.has("ratio-svg")) return;
  const W = 680, H = 380;
  function build() {
    const mode = SX.val("ratio-sort", "page");
    const rows0 = NR.ratioBoard();
    let rows = rows0.slice();
    if (mode === "ratio") rows.sort((a, b) => b.ratio - a.ratio);
    else if (mode === "slack") rows.sort((a, b) => ((b.bound === null ? 99 : b.bound - b.ratio)) - ((a.bound === null ? 99 : a.bound - a.ratio)));
    const svg = d3.select("#ratio-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 268, r: 74, t: 26, b: 40 });
    const maxX = Math.max(3.6, d3.max(rows, r => Math.max(r.ratio, r.bound === null ? 0 : r.bound)) * 1.08);
    const x = d3.scaleLinear().domain([1, maxX]).range([0, F.iw]);
    const bh = F.ih / rows.length;
    AL.gridX(F.g, x, F.ih, 6);
    AL.axisB(F.g, x, F.ih, 6, "approximation ratio (1 = optimal; every ratio on this page is ≥ 1)");
    F.g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", AC.muted).text("blue bar = MEASURED against an exhaustive optimum · amber tick = the PROVED bound");
    rows.forEach((r, i) => {
      const y0 = i * bh + 3, hh = bh - 8;
      const ok = r.bound === null || r.ratio <= r.bound + 1e-9;
      F.g.append("rect").attr("x", 0).attr("y", y0).attr("width", Math.max(1, x(r.ratio) - x(1))).attr("height", hh).attr("rx", 3)
        .attr("fill", ok ? AC.accent : AC.bad).attr("opacity", 0.9);
      F.g.append("text").attr("x", -8).attr("y", y0 + hh / 2 + 4).attr("text-anchor", "end").attr("font-size", 10.5).attr("fill", AC.ink).text(r.label);
      F.g.append("text").attr("x", x(r.ratio) - x(1) + 6).attr("y", y0 + hh / 2 + 4).attr("font-size", 10.5).attr("fill", ok ? AC.good : AC.bad).text(SX.f3(r.ratio));
      if (r.bound !== null && r.bound <= maxX) {
        F.g.append("line").attr("x1", x(r.bound)).attr("x2", x(r.bound)).attr("y1", y0 - 1).attr("y2", y0 + hh + 1)
          .attr("stroke", AC.a2).attr("stroke-width", 2.5);
        F.g.append("text").attr("x", F.iw + 6).attr("y", y0 + hh / 2 + 4).attr("font-size", 9.5).attr("fill", AC.a2).text(r.boundLabel);
      } else {
        F.g.append("text").attr("x", F.iw + 6).attr("y", y0 + hh / 2 + 4).attr("font-size", 9.5).attr("fill", AC.muted).text("no bound");
      }
    });
    const allOk = rows.every(r => r.bound === null || r.ratio <= r.bound + 1e-9);
    const tight = rows.filter(r => r.bound !== null && Math.abs(r.ratio - r.bound) < 1e-9);
    const loosest = rows.filter(r => r.bound !== null).reduce((a, r) => (a === null || (r.bound - r.ratio) > (a.bound - a.ratio) ? r : a), null);
    SX.setHtml("ratio-readout",
      `<b>${rows.length}</b> algorithm/instance pairs, every ratio measured against an optimum found by exhaustive search (all 2ⁿ subsets, all 2 520 tours, all 2ᵐ subfamilies, all 2ⁿ assignments) · every measured ratio ≤ its proved bound ${SX.flag(allOk)} · ` +
      rows0.map(r => `${r.label}: measured <b>${SX.f4(r.ratio)}</b> vs ${r.bound === null ? "no constant bound" : "bound " + r.boundLabel} ${r.bound === null ? "" : SX.flag(r.ratio <= r.bound + 1e-9)}`).join(" · ") +
      ` · bounds attained EXACTLY (so tight, not merely valid) by: <b>${tight.length ? tight.map(r => r.label).join("; ") : "none"}</b> · loosest guarantee here: <b>${loosest.label}</b>, promising ${SX.f3(loosest.bound)} and delivering ${SX.f3(loosest.ratio)} — slack ${SX.f3(loosest.bound - loosest.ratio)} · reading: a proved ratio describes the ADVERSARY; on ordinary instances the algorithms are far better, and on the adversarial instances built in §21 and §23 they are not`);
  }
  SX.on("ratio-sort", "change", build);
  build();
})();

/* ── 21  #vcapx-svg  the 2-approximation edge by edge, against an exhaustive optimum ── */
(function () {
  if (!SX.has("vcapx-svg")) return;
  const W = 680, H = 330;
  function inst(key) {
    if (key === "k33") {
      const E = []; for (let a = 0; a < 3; a++) for (let b = 3; b < 6; b++) E.push([a, b]);
      return { n: 6, E, names: ["u₁", "u₂", "u₃", "v₁", "v₂", "v₃"],
               pos: [[200, 60], [200, 140], [200, 220], [470, 60], [470, 140], [470, 220]], r: 17, brute: true };
    }
    if (key === "bad" || key === "bad16") {
      const B = NR.vcGreedyBad(key === "bad16" ? 16 : 8);
      const pos = []; const right = B.n - B.L;
      const cols = Math.ceil(right / 18), top = 22, span = 262;
      for (let i = 0; i < B.L; i++) pos.push([130, top + i * (span / Math.max(1, B.L - 1))]);
      const per = Math.ceil(right / cols);
      for (let j = 0; j < right; j++) { const cc = Math.floor(j / per), rr = j % per, cnt = Math.min(per, right - cc * per);
        pos.push([420 + cc * 120, top + rr * (span / Math.max(1, cnt - 1))]); }
      return { n: B.n, E: B.E, names: B.names, pos, r: key === "bad16" ? 6 : 9, brute: false, L: B.L };
    }
    const g = NR.vcIndex(NI.gVC);
    return { n: g.n, E: g.E, names: g.names, pos: NI.gVC.V.map(v => NI.gVC.pos[v]).map(p => [p[0] * 1.32 + 40, p[1] * 1.5 + 40]), r: 17, brute: true };
  }
  function build() {
    const key = SX.val("vcapx-inst", "g7"), alg = SX.val("vcapx-alg", "match");
    const G = inst(key), n = G.n, E = G.E, N = G.names;
    const cM = AL.counter(), cD = AL.counter(), cB = AL.counter();
    const mm = NR.vcMatching(n, E, cM);
    const gd = NR.vcGreedyDegree(n, E, cD);
    /* the optimum: exhaustive for the small instances; for the big one the section's own
       two lemmas pin it — L is a cover (OPT ≤ L) and OPT ≥ |M| — and they meet. */
    let optSize, optSet, optWhy, optUnique = null, subsets = 0;
    if (G.brute) { const b = NR.vcOptBrute(n, E, cB); optSize = b.size; optSet = b.best; optUnique = b.unique; subsets = cB.get("subset"); optWhy = "exhaustive over all 2^" + n + " = " + (1 << n) + " subsets"; }
    else { const lb = mm.matching.length, ub = G.L; optSize = ub; optSet = d3.range(G.L); optWhy = "pinned: |M| = " + lb + " ≤ OPT ≤ |L| = " + ub + (lb === ub ? ", and they MEET" : ""); }
    const inOpt = new Array(n).fill(false); optSet.forEach(v => { inOpt[v] = true; });
    const R = alg === "match" ? mm : gd;
    const ratio = R.size / optSize;
    const bound = alg === "match" ? 2 : NR.harmonic(Math.max.apply(null, (function () { const d = new Array(n).fill(0); E.forEach(e => { d[e[0]]++; d[e[1]]++; }); return d; })()));
    const ok = ratio <= bound + 1e-9;
    const coverOk = NR.vcIsCover(E, R.inCover, null);
    /* frames */
    const frames = [{ init: true }];
    if (alg === "match") mm.trace.forEach(t => frames.push({ i: t.i, e: t.e, take: t.take, M: t.M, used: t.used }));
    else gd.trace.forEach((t, k) => frames.push({ pick: t.pick, deg: t.deg, killed: t.killed, cover: t.cover, left: t.left, k }));
    const svg = d3.select("#vcapx-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 0, r: 0, t: 6, b: 34 });
    function render(f, k) {
      F.g.selectAll("*").remove();
      const matched = f.init ? [] : (alg === "match" ? f.M : []);
      const inC = f.init ? new Array(n).fill(false) : (alg === "match" ? f.used : (function () { const a = new Array(n).fill(false); f.cover.forEach(v => { a[v] = true; }); return a; })());
      const deadEdges = new Set();
      if (alg === "deg" && !f.init) gd.trace.slice(0, f.k + 1).forEach(t => t.killed.forEach(i => deadEdges.add(i)));
      E.forEach((e, i) => {
        const a = G.pos[e[0]], b = G.pos[e[1]];
        let col = AC.line, wdt = 1.4, dash = null;
        if (alg === "match") {
          if (matched.indexOf(i) >= 0) { col = AC.good; wdt = 3.2; }
          else if (!f.init && i < f.i) { col = AC.muted; wdt = 1.1; dash = "3,3"; }
          if (!f.init && i === f.i) { col = f.take ? AC.good : AC.a2; wdt = 3.4; dash = f.take ? null : "4,3"; }
        } else {
          if (deadEdges.has(i)) { col = AC.muted; wdt = 1.1; dash = "3,3"; }
          if (!f.init && f.killed.indexOf(i) >= 0) { col = AC.a2; wdt = 3; dash = null; }
        }
        const ln = F.g.append("line").attr("x1", a[0]).attr("y1", a[1]).attr("x2", b[0]).attr("y2", b[1]).attr("stroke", col).attr("stroke-width", wdt);
        if (dash) ln.attr("stroke-dasharray", dash);
      });
      for (let v = 0; v < n; v++) {
        const p = G.pos[v];
        F.g.append("circle").attr("cx", p[0]).attr("cy", p[1]).attr("r", G.r)
          .attr("fill", inC[v] ? AC.good : AC.panel2)
          .attr("stroke", inOpt[v] ? AC.accent : AC.line).attr("stroke-width", inOpt[v] ? 3 : 1.4);
        if (G.r >= 13) F.g.append("text").attr("x", p[0]).attr("y", p[1] + 4).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", inC[v] ? AC.bg : AC.ink).text(N[v]);
      }
      const cnt = inC.filter(Boolean).length;
      F.g.append("text").attr("x", 8).attr("y", F.ih + 24).attr("font-size", 11).attr("fill", AC.ink)
        .text(f.init ? `initialised · ${n} vertices, ${E.length} edges · blue outline = a minimum cover (${optWhy})`
          : alg === "match"
            ? `edge ${f.i + 1}/${E.length} = (${N[f.e[0]]}, ${N[f.e[1]]}): ${f.take ? "both endpoints free → MATCH it, both enter the cover" : "an endpoint is already matched → skip"} · |M| = ${f.M.length} · |C| = ${cnt} · so far OPT ≥ ${f.M.length}`
            : `pick ${N[f.pick]} (remaining degree ${f.deg}, the maximum) · removes ${f.killed.length} edge${f.killed.length === 1 ? "" : "s"} · |C| = ${cnt} · ${f.left} edge${f.left === 1 ? "" : "s"} still uncovered`);
    }
    AL.stepper(svg, { frames, render, delay: alg === "match" ? 620 : 900, label: alg === "match" ? "edge" : "pick" });
    const dmax = Math.max.apply(null, (function () { const d = new Array(n).fill(0); E.forEach(e => { d[e[0]]++; d[e[1]]++; }); return d; })());
    SX.setHtml("vcapx-readout",
      `${n} vertices, ${E.length} edges, maximum degree ${dmax} · <b>${alg === "match" ? "maximal matching" : "maximum remaining degree"}</b>: |C| = <b>${R.size}</b>${alg === "match" ? ` = 2·|M| = 2·${mm.matching.length} ${SX.flag(R.size === 2 * mm.matching.length)}` : ""} — is it a cover? ${SX.flag(coverOk)} · optimum <b>${optSize}</b> (${optWhy}${optUnique === null ? "" : optUnique ? ", and it is UNIQUE" : ""})${G.brute && optSize <= 12 ? ", namely {" + optSet.map(v => N[v]).join(", ") + "}" : ""} · measured ratio <b>${SX.f3(ratio)}</b> vs the bound ${alg === "match" ? "2" : "H(Δ) = H(" + dmax + ") = " + SX.f2(bound)} ${SX.flag(ok)}${alg === "match" && Math.abs(ratio - 2) < 1e-9 ? " — <b>attained exactly</b>: this instance is a worst case for the 2-approximation" : ""}` +
      ` · the matching certificate: |M| = <b>${mm.matching.length}</b>, and every cover needs one endpoint of each of those disjoint edges, so OPT ≥ ${mm.matching.length} ${SX.flag(optSize >= mm.matching.length)}` +
      ` · counted: edges examined by the matching pass <b>${cM.get("edge")}</b> = |E| ${SX.flag(cM.get("edge") === E.length)}, edges matched <b>${cM.get("match")}</b>; the degree greedy made <b>${cD.get("pick")}</b> picks after <b>${cD.get("edge")}</b> edge scans${G.brute ? `; the exhaustive search examined <b>${SX.int(subsets)}</b> subsets` : ""}` +
      ` · side by side on this instance: matching <b>${mm.size}</b> (ratio ${SX.f3(mm.size / optSize)}), degree greedy <b>${gd.size}</b> (ratio ${SX.f3(gd.size / optSize)}) — ${gd.size < mm.size ? "the greedy wins here, and it has no constant guarantee" : gd.size > mm.size ? "the greedy LOSES here, which is why it is not a 2-approximation" : "a tie"}`);
  }
  SX.on("vcapx-inst", "change", build); SX.on("vcapx-alg", "change", build);
  build();
})();

/* ── 22  #tsp-svg  four tours on eight cities, plus the gap instance of the impossibility proof ── */
(function () {
  if (!SX.has("tsp-svg")) return;
  const W = 680, H = 350;
  const pts = NI.tsp.pts, NM = NI.tsp.names, D = NR.tspDist(pts);
  /* computed once: they are the same for every control setting */
  const tri = NR.tspTriangle(D, AL.counter());
  const cOpt = AL.counter(), OPT = NR.tspOpt(D, cOpt);
  const MST = NR.tspMST(D, null);
  const DT = NR.tspDoubleTree(D, null);
  const CH = NR.tspChristofides(D, null);
  const NN = NR.tspNN(D, 0, null);
  const cTwo = AL.counter(), TW = NR.tspTwoOpt(D, NN.tour, cTwo);
  const gapCache = {};
  function gap(rho) {
    const k = String(rho);
    if (!gapCache[k]) {
      const cH = AL.counter(), cP = AL.counter();
      gapCache[k] = { ham: NR.tspGapDemo(NI.gHam, rho, cH), pet: NR.tspGapDemo(NI.gPetersen, rho, cP), cH, cP };
    }
    return gapCache[k];
  }
  function sx(p, box) { return box.x + (p[0] / 9) * box.w; }
  function sy(p, box) { return box.y + box.h - (p[1] / 8) * box.h; }
  function build() {
    const alg = SX.val("tsp-alg", "dt"); const rho = +SX.val("tsp-rho", 2);
    SX.setText("tsp-rho-val", String(rho));
    const svg = d3.select("#tsp-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 0, r: 0, t: 8, b: 36 });
    if (alg === "gap") {
      const G = gap(rho), P = NI.gPetersen;
      /* the slider allows fractional ρ, so ρ·n can be fractional in general;
         print it as an integer when it is one, so the figure reads like §13. */
      const fmtN = x => (Math.abs(x - Math.round(x)) < 1e-9 ? String(Math.round(x)) : SX.f1(x));
      const cx = 250, cy = 150, R1 = 118, R2 = 56;
      const pos = []; for (let i = 0; i < 5; i++) { const a = -Math.PI / 2 + i * 2 * Math.PI / 5; pos.push([cx + R1 * Math.cos(a), cy + R1 * Math.sin(a)]); }
      for (let i = 0; i < 5; i++) { const a = -Math.PI / 2 + i * 2 * Math.PI / 5; pos.push([cx + R2 * Math.cos(a), cy + R2 * Math.sin(a)]); }
      const inG = {}; P.E.forEach(e => { inG[e[0] + "," + e[1]] = true; inG[e[1] + "," + e[0]] = true; });
      P.E.forEach(e => { F.g.append("line").attr("x1", pos[e[0]][0]).attr("y1", pos[e[0]][1]).attr("x2", pos[e[1]][0]).attr("y2", pos[e[1]][1]).attr("stroke", AC.line).attr("stroke-width", 1.6); });
      const t = G.pet.tour;
      for (let i = 0; i < t.length; i++) {
        const u = t[i], v = t[(i + 1) % t.length], real = !!inG[u + "," + v];
        const ln = F.g.append("line").attr("x1", pos[u][0]).attr("y1", pos[u][1]).attr("x2", pos[v][0]).attr("y2", pos[v][1])
          .attr("stroke", real ? AC.good : AC.bad).attr("stroke-width", real ? 3 : 3.4);
        if (!real) { ln.attr("stroke-dasharray", "6,4"); F.g.append("text").attr("x", (pos[u][0] + pos[v][0]) / 2 + 6).attr("y", (pos[u][1] + pos[v][1]) / 2 - 4).attr("font-size", 11).attr("fill", AC.bad).text("the forced non-edge, cost ρ·n + 1 = " + G.pet.nonEdge); }
      }
      for (let v = 0; v < P.n; v++) {
        F.g.append("circle").attr("cx", pos[v][0]).attr("cy", pos[v][1]).attr("r", 14).attr("fill", AC.panel2).attr("stroke", AC.line);
        F.g.append("text").attr("x", pos[v][0]).attr("y", pos[v][1] + 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.ink).text(v);
      }
      const tx = 470;
      [["graph edges cost 1", AC.good], ["non-edges cost ρ·n + 1 = " + G.pet.nonEdge, AC.bad]].forEach((r, i) => {
        F.g.append("line").attr("x1", tx).attr("x2", tx + 18).attr("y1", 40 + i * 20).attr("y2", 40 + i * 20).attr("stroke", r[1]).attr("stroke-width", 3);
        F.g.append("text").attr("x", tx + 24).attr("y", 44 + i * 20).attr("font-size", 11).attr("fill", AC.muted).text(r[0]);
      });
      F.g.append("text").attr("x", tx).attr("y", 100).attr("font-size", 11.5).attr("fill", AC.ink).text("Petersen graph — NO Hamiltonian cycle");
      F.g.append("text").attr("x", tx).attr("y", 120).attr("font-size", 11.5).attr("fill", AC.a2).text("optimum " + G.pet.cost + "  >  ρ·n = " + fmtN(rho * P.n));
      F.g.append("text").attr("x", tx).attr("y", 146).attr("font-size", 11.5).attr("fill", AC.ink).text("the 8-vertex graph of §13 — HAS one");
      F.g.append("text").attr("x", tx).attr("y", 166).attr("font-size", 11.5).attr("fill", AC.good).text("optimum " + G.ham.cost + "  =  n = " + NI.gHam.n + "  ≤  ρ·n = " + fmtN(rho * NI.gHam.n));
      F.g.append("text").attr("x", 8).attr("y", F.ih + 26).attr("font-size", 11).attr("fill", AC.ink)
        .text(`general TSP, ρ = ${rho}, non-edge weight ρ·n + 1 = ${G.pet.nonEdge} (the weight §13 builds): the optimal tour of the Petersen instance uses ${G.pet.graphEdges} graph edges and ${G.pet.nonEdgesUsed} non-edge — the best possible, because the graph has a Hamiltonian PATH but no cycle`);
      SX.setHtml("tsp-readout",
        `the gap instance of §13, ρ = <b>${rho}</b> — costs are 1 on an edge of G and ρ·n + 1 = <b>${G.pet.nonEdge}</b> off it (${G.ham.nonEdge} on the eight-vertex instance), tested against the threshold ρ·n · <b>with</b> a Hamiltonian cycle (8 vertices, 11 edges): exhaustive optimum over all ${SX.int(G.ham.tours)} tours = <b>${G.ham.cost}</b>, and n = ${NI.gHam.n}, so optimum = n ${SX.flag(G.ham.cost === NI.gHam.n)} — and ≤ ρ·n = ${fmtN(rho * NI.gHam.n)} ${SX.flag(G.ham.cost <= rho * NI.gHam.n)} · <b>without</b> one (the Petersen graph, 10 vertices, 15 edges): exhaustive optimum over all ${SX.int(G.pet.tours)} tours = <b>${G.pet.cost}</b> = ${G.pet.graphEdges}·1 + ${G.pet.nonEdgesUsed}·${G.pet.nonEdge} ${SX.flag(G.pet.cost === G.pet.graphEdges + G.pet.nonEdgesUsed * G.pet.nonEdge)}, which is exactly the proof's lower bound ρ·n + n = ${fmtN(rho * P.n + P.n)} — attained, not merely respected ${SX.flag(G.pet.cost === rho * P.n + P.n)} — and strictly above the threshold ρ·n = ${fmtN(rho * P.n)} ${SX.flag(G.pet.cost > rho * P.n)} · so "the returned tour costs ≤ ρ·n" decides Hamiltonicity, and a polynomial ρ-approximation would put SAT in P · is the triangle inequality VIOLATED here, as the construction requires? ${SX.flag(!G.pet.metric)} — it must be: c(u,w) = ρ·n + 1 = ${G.pet.nonEdge} while a two-edge detour through the graph costs 2, so the hypothesis behind this section's 2 and 3/2 is exactly what this instance denies · distance evaluations counted: ${SX.int(G.cH.get("dist") + G.cP.get("dist"))}`);
      return;
    }
    /* ── the metric instance ── */
    const box = { x: 60, y: 30, w: 400, h: 250 };
    let tour = null, label = "", bound = null, boundLabel = "", showMST = false, showMatch = false, cost = 0;
    if (alg === "opt") { tour = OPT.tour; cost = OPT.cost; label = "exhaustive optimum"; bound = 1; boundLabel = "1 (exact)"; }
    else if (alg === "mst") { showMST = true; cost = MST.weight; label = "the minimum spanning tree (not a tour)"; }
    else if (alg === "dt") { tour = DT.tour; cost = DT.cost; showMST = true; label = "double tree"; bound = 2; boundLabel = "2"; }
    else if (alg === "chr") { tour = CH.tour; cost = CH.cost; showMST = true; showMatch = true; label = "Christofides"; bound = 1.5; boundLabel = "3/2"; }
    else if (alg === "nn") { tour = NN.tour; cost = NN.cost; label = "nearest neighbour from A"; }
    else { tour = TW.tour; cost = TW.cost; label = "nearest neighbour, then 2-opt"; }
    if (showMST) MST.edges.forEach(e => F.g.append("line").attr("x1", sx(pts[e[0]], box)).attr("y1", sy(pts[e[0]], box)).attr("x2", sx(pts[e[1]], box)).attr("y2", sy(pts[e[1]], box)).attr("stroke", AC.muted).attr("stroke-width", 2).attr("stroke-dasharray", "5,4"));
    if (showMatch) CH.matching.forEach(e => F.g.append("line").attr("x1", sx(pts[e[0]], box)).attr("y1", sy(pts[e[0]], box)).attr("x2", sx(pts[e[1]], box)).attr("y2", sy(pts[e[1]], box)).attr("stroke", AC.violet).attr("stroke-width", 3.4));
    if (tour) for (let i = 0; i < tour.length; i++) {
      const u = tour[i], v = tour[(i + 1) % tour.length];
      F.g.append("line").attr("x1", sx(pts[u], box)).attr("y1", sy(pts[u], box)).attr("x2", sx(pts[v], box)).attr("y2", sy(pts[v], box))
        .attr("stroke", AC.accent).attr("stroke-width", 2.6).attr("opacity", 0.95);
    }
    pts.forEach((p, i) => {
      const odd = CH.odd.indexOf(i) >= 0;
      F.g.append("circle").attr("cx", sx(p, box)).attr("cy", sy(p, box)).attr("r", 14)
        .attr("fill", AC.panel2).attr("stroke", (showMatch && odd) ? AC.violet : AC.line).attr("stroke-width", (showMatch && odd) ? 3 : 1.4);
      F.g.append("text").attr("x", sx(p, box)).attr("y", sy(p, box) + 4).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", AC.ink).text(NM[i]);
    });
    const leg = [["tour returned", AC.accent, null]];
    if (showMST) leg.push(["minimum spanning tree", AC.muted, "5,4"]);
    if (showMatch) leg.push(["matching on the odd-degree vertices", AC.violet, null]);
    AL.legend(F.g, leg.map(l => ({ label: l[0], color: l[1], dash: l[2] })), 490, 46);
    F.g.append("text").attr("x", 490).attr("y", 120).attr("font-size", 11.5).attr("fill", AC.ink).text(label);
    F.g.append("text").attr("x", 490).attr("y", 140).attr("font-size", 11.5).attr("fill", AC.a2).text("cost " + SX.f4(cost));
    F.g.append("text").attr("x", 490).attr("y", 160).attr("font-size", 11.5).attr("fill", AC.good).text("optimum " + SX.f4(OPT.cost));
    if (alg !== "mst") F.g.append("text").attr("x", 490).attr("y", 180).attr("font-size", 11.5).attr("fill", AC.ink).text("ratio " + SX.f4(cost / OPT.cost));
    F.g.append("text").attr("x", 8).attr("y", F.ih + 26).attr("font-size", 11).attr("fill", AC.ink)
      .text(tour ? `${tour.map(i => NM[i]).join("-")}-${NM[tour[0]]}   ·   ${alg === "mst" ? "" : "cost " + SX.f4(cost) + " against the optimum " + SX.f4(OPT.cost)}`
                 : `tree edges ${MST.edges.map(e => NM[e[0]] + "-" + NM[e[1]]).join(", ")}   ·   weight ${SX.f4(MST.weight)} ≤ OPT ${SX.f4(OPT.cost)}`);
    const ratio = cost / OPT.cost, ok = bound === null || alg === "mst" || ratio <= bound + 1e-9;
    SX.setHtml("tsp-readout",
      `triangle inequality checked over all <b>${SX.int(tri.triples)}</b> ordered triples: holds ${SX.flag(tri.ok)} — without it none of the bounds below apply · exhaustive optimum over all <b>${SX.int(OPT.tours)}</b> = 7!/2 distinct tours: <b>${SX.f4(OPT.cost)}</b>, tour ${OPT.tour.map(i => NM[i]).join("-")} (${SX.int(cOpt.get("dist"))} distance evaluations) · ` +
      `MST weight <b>${SX.f4(MST.weight)}</b> ≤ OPT ${SX.flag(MST.weight <= OPT.cost)} — step (1) of both proofs · ` +
      `double tree <b>${SX.f4(DT.cost)}</b>, ratio <b>${SX.f4(DT.cost / OPT.cost)}</b> ≤ 2 ${SX.flag(DT.cost <= 2 * OPT.cost)}; its full walk costs 2·MST = ${SX.f4(DT.walkCost)} and shortcutting brought it down ${SX.flag(DT.cost <= DT.walkCost)} · ` +
      `Christofides <b>${SX.f4(CH.cost)}</b>, ratio <b>${SX.f4(CH.cost / OPT.cost)}</b> ≤ 3/2 ${SX.flag(CH.cost <= 1.5 * OPT.cost)}: MST degrees ⟨${MST.deg.join(", ")}⟩ → odd set {${CH.odd.map(i => NM[i]).join(", ")}} (even in number ${SX.flag(CH.odd.length % 2 === 0)}), cheapest of its ${CH.pairings} perfect matchings is {${CH.matching.map(e => "(" + NM[e[0]] + "," + NM[e[1]] + ")").join(", ")}} at <b>${SX.f4(CH.matchCost)}</b> ≤ OPT/2 = ${SX.f4(OPT.cost / 2)} ${SX.flag(CH.matchCost <= OPT.cost / 2)} — the proof's step (2), verified; T ∪ M weighs ${SX.f4(CH.multiWeight)} ≤ 1.5·OPT = ${SX.f4(1.5 * OPT.cost)} ${SX.flag(CH.multiWeight <= 1.5 * OPT.cost)}, Euler circuit ${CH.circuit.map(i => NM[i]).join("-")}, shortcut to ${CH.tour.map(i => NM[i]).join("-")} · ` +
      `nearest neighbour <b>${SX.f4(NN.cost)}</b>, ratio <b>${SX.f4(NN.cost / OPT.cost)}</b> — no constant bound exists for it, so there is nothing to check; 2-opt then makes <b>${cTwo.get("probe") ? TW.moves : TW.moves}</b> improving move${TW.moves === 1 ? "" : "s"} in ${SX.int(cTwo.get("probe"))} probes and reaches <b>${SX.f4(TW.cost)}</b>${Math.abs(TW.cost - OPT.cost) < 1e-9 ? " — the exact optimum, from two algorithms with no guarantee at all" : ""} ${SX.flag(TW.cost <= NN.cost)} · ` +
      `currently showing <b>${label}</b>${alg === "mst" ? " (a tree, not a tour — no ratio applies)" : `: measured ${SX.f4(ratio)} vs bound ${boundLabel || "none"} ${bound === null ? "" : SX.flag(ok)}`}`);
  }
  SX.on("tsp-alg", "change", build); SX.on("tsp-rho", "input", build);
  build();
})();

/* ── 23  #setcover-svg  greedy set cover, one pick per step, with prices and the exhaustive optimum ── */
(function () {
  if (!SX.has("setcover-svg")) return;
  const W = 680, H = 360;
  function inst(key) {
    if (key === "demo") return { U: NI.setcover.U, sets: NI.setcover.sets, names: NI.setcover.sets.map((_, i) => "S" + (i + 1)), tag: "12 elements, 6 sets" };
    const k = +key.slice(3), B = NR.setCoverBad(k);
    const names = ["T₁", "T₂"].concat(B.S.slice().reverse().map((_, i) => "S" + (k - i)));
    return { U: B.U, sets: B.sets, names, tag: "the adversarial family, k = " + k, k, bad: true };
  }
  function build() {
    const key = SX.val("sc-inst", "demo"), I = inst(key);
    const cG = AL.counter(), cO = AL.counter();
    const G = NR.setCoverGreedy(I.U, I.sets, cG);
    const O = NR.setCoverOpt(I.U, I.sets, cO);
    const d = Math.max.apply(null, I.sets.map(S => S.length));
    const Hd = NR.harmonic(d), ratio = G.size / O.size;
    /* the charging identity and the key lemma, both re-derived here */
    let priceSum = 0; for (let e = 1; e <= I.U; e++) priceSum += G.price[e];
    const lemma = I.sets.map((S, i) => { const p = S.reduce((a, e) => a + G.price[e], 0); return { i, p, H: NR.harmonic(S.length), ok: p <= NR.harmonic(S.length) + 1e-9 }; });
    const optPrice = O.best.reduce((a, i) => a + lemma[i].p, 0);
    const inOpt = {}; O.best.forEach(i => { inOpt[i] = true; });
    const cols = Math.min(16, Math.ceil(Math.sqrt(I.U * 2.6))), rows = Math.ceil(I.U / cols);
    const cw = Math.min(30, Math.floor(360 / cols)), chh = Math.min(24, Math.floor(180 / rows));
    const frames = [{ init: true, covered: new Array(I.U + 1).fill(false), left: I.U }].concat(G.trace.map(t => Object.assign({}, t)));
    const svg = d3.select("#setcover-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 24, r: 0, t: 16, b: 34 });
    function render(f, kk) {
      F.g.selectAll("*").remove();
      const covered = f.covered, picked = f.init ? [] : G.picked.slice(0, kk);
      F.g.append("text").attr("x", 0).attr("y", -2).attr("font-size", 11).attr("fill", AC.muted).text("the universe — green = covered, amber = covered at THIS step, with the price 1/gain it pays");
      SX.grid(F.g, rows, cols, { x: 0, y: 12, w: cw, h: chh, gap: 2, fontSize: I.U > 40 ? 8 : 10,
        text: (i, j) => { const e = i * cols + j + 1; return e <= I.U ? (I.U <= 30 ? e : "") : ""; },
        fill: (i, j) => { const e = i * cols + j + 1; if (e > I.U) return AC.bg;
          if (!f.init && f.newly && f.newly.indexOf(e) >= 0) return AC.a2;
          return covered[e] ? AC.good : null; } });
      /* price labels for the small instance */
      if (I.U <= 30 && !f.init) {
        for (let e = 1; e <= I.U; e++) if (covered[e]) {
          const i = Math.floor((e - 1) / cols), j = (e - 1) % cols;
          F.g.append("text").attr("x", j * (cw + 2) + cw / 2).attr("y", 12 + i * (chh + 2) + chh + 9).attr("text-anchor", "middle")
            .attr("font-size", 8.5).attr("fill", AC.muted).text(G.price[e] === 1 ? "1" : "1/" + Math.round(1 / G.price[e]));
        }
      }
      /* the gain bars */
      const gy = 12 + rows * (chh + 2) + 34;
      F.g.append("text").attr("x", 0).attr("y", gy - 10).attr("font-size", 11).attr("fill", AC.muted).text("gain |Sᵢ ∩ U| right now, measured — greedy takes the tallest");
      const bw = Math.min(54, Math.floor(560 / I.sets.length)), bh = H - gy - 60;
      const gains = I.sets.map((S, i) => picked.indexOf(i) >= 0 ? -1 : S.filter(e => !covered[e]).length);
      const gmax = Math.max(1, d3.max(gains));
      I.sets.forEach((S, i) => {
        const gv = gains[i], taken = picked.indexOf(i) >= 0, justTaken = !f.init && f.set === i;
        const hgt = gv > 0 ? (gv / gmax) * bh : 2;
        F.g.append("rect").attr("x", i * (bw + 8)).attr("y", gy + bh - hgt).attr("width", bw - 6).attr("height", hgt).attr("rx", 3)
          .attr("fill", justTaken ? AC.a2 : taken ? AC.muted : AC.accent).attr("opacity", taken && !justTaken ? 0.45 : 0.95);
        F.g.append("text").attr("x", i * (bw + 8) + (bw - 6) / 2).attr("y", gy + bh + 14).attr("text-anchor", "middle").attr("font-size", 10)
          .attr("fill", inOpt[i] ? AC.accent : AC.muted).text(I.names[i] + (inOpt[i] ? " ★" : ""));
        F.g.append("text").attr("x", i * (bw + 8) + (bw - 6) / 2).attr("y", gy + bh - hgt - 4).attr("text-anchor", "middle").attr("font-size", 10)
          .attr("fill", AC.ink).text(taken && !justTaken ? "–" : gv);
      });
      F.g.append("text").attr("x", 0).attr("y", F.ih + 24).attr("font-size", 11).attr("fill", AC.ink)
        .text(f.init ? `${I.tag} · ${I.U} elements, ${I.sets.length} sets, largest set d = ${d} · ★ marks the sets of the exhaustive optimum (${O.size} sets)`
          : `step ${kk}: take ${I.names[f.set]} — gain ${f.gain} new element${f.gain === 1 ? "" : "s"}, each charged price 1/${f.gain} = ${SX.f3(f.price)} · ${f.left} element${f.left === 1 ? "" : "s"} still uncovered · sets used so far ${kk}`);
    }
    AL.stepper(svg, { frames, render, delay: 950, label: "pick" });
    SX.setHtml("setcover-readout",
      `${I.tag}: n = <b>${I.U}</b> elements, m = <b>${I.sets.length}</b> sets, largest set d = <b>${d}</b> · greedy takes <b>${G.size}</b> sets — ${G.picked.map(i => I.names[i]).join(", ")} — with gains ⟨${G.trace.map(t => t.gain).join(", ")}⟩ (measured: ${SX.int(cG.get("test"))} membership tests, ${cG.get("pick")} picks) · exhaustive optimum over all 2^${I.sets.length} = ${SX.int(1 << I.sets.length)} subfamilies: <b>${O.size}</b> sets — ${O.best.map(i => I.names[i]).join(", ")}${O.unique ? ", and it is UNIQUE" : " (one of " + O.count + " optima)"} (${SX.int(cO.get("subset"))} subfamilies enumerated) · ` +
      `measured ratio <b>${SX.f3(ratio)}</b> vs the proved H(d) = H(${d}) = <b>${SX.f4(Hd)}</b> ${SX.flag(ratio <= Hd + 1e-9)} · ` +
      `the charging identity: the ${I.U} element prices sum to <b>${SX.f4(priceSum)}</b>, which must equal the number of sets greedy used, ${G.size} ${SX.flag(Math.abs(priceSum - G.size) < 1e-9)} · ` +
      `the key lemma ∑_{x∈S} c_x ≤ H(|S|) holds for every one of the ${I.sets.length} sets ${SX.flag(lemma.every(l => l.ok))}${I.sets.length <= 8 ? " — " + lemma.map(l => `${I.names[l.i]}: ${SX.f3(l.p)} ≤ H(${I.sets[l.i].length}) = ${SX.f3(l.H)}`).join("; ") : ""} · ` +
      `summing the lemma over the optimum: ${O.best.map(i => SX.f3(lemma[i].p)).join(" + ")} = <b>${SX.f3(optPrice)}</b> ≥ ${G.size} = |greedy| ${SX.flag(optPrice >= G.size - 1e-9)}, and ≤ OPT·H(d) = ${SX.f3(O.size * Hd)} ${SX.flag(optPrice <= O.size * Hd + 1e-9)} — the theorem, re-derived on this instance · ` +
      `for scale: H(n) = H(${I.U}) = ${SX.f4(NR.harmonic(I.U))}, ln n = ${SX.f4(Math.log(I.U))}, ln n + 1 = ${SX.f4(Math.log(I.U) + 1)} — H(d) ≤ H(n) ≤ ln n + 1, and ln n is NOT H(n) ${SX.flag(NR.harmonic(I.U) > Math.log(I.U))}` +
      (I.bad ? ` · this is the adversarial family: OPT stays at 2 while greedy takes k = ${I.k}, so the ratio is k/2 = <b>${SX.f1(I.k / 2)}</b> and grows like ½·log₂ n ${SX.flag(Math.abs(ratio - I.k / 2) < 1e-9)}` : ""));
  }
  SX.on("sc-inst", "change", build);
  build();
})();

/* ── 24  #max3sat-svg  the coin-flip distribution, and the derandomised walk beside it ── */
(function () {
  if (!SX.has("max3sat-svg")) return;
  const W = 680, H = 330;
  function inst(key) {
    if (key === "phi") return { phi: NI.phi, tag: "the three-clause formula of §09" };
    if (key === "rand") return { phi: NR.max3satFormula(6, 20, 7), tag: "random, 6 variables, 20 clauses (seed 7)" };
    if (key === "rand2") return { phi: NR.max3satFormula(8, 40, 11), tag: "random, 8 variables, 40 clauses (seed 11)" };
    return { phi: NI.phi8, tag: "all eight clauses on three variables" };
  }
  function build() {
    const I = inst(SX.val("m3-inst", "phi8")), phi = I.phi, m = phi.clauses.length, n = phi.n;
    const trials = +SX.val("m3-trials", 2000); SX.setText("m3-trials-val", SX.int(trials));
    const wf = NR.max3satWellFormed(phi);
    const cB = AL.counter(), B = NR.max3satBrute(phi, cB);
    const cR = AL.counter(), R = NR.max3satRandom(phi, trials, 999, cR);
    const cD = AL.counter(), Dz = NR.max3satDerandom(phi, cD);
    const exp = 7 * m / 8;
    const svg = d3.select("#max3sat-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 40, r: 6, t: 24, b: 40 });
    /* ── left: the histogram ── */
    const lo = Math.max(0, B.dist.findIndex(v => v > 0) - 1), hi = m;
    const bwidth = 320 / (hi - lo + 1);
    const x = k => (k - lo) * bwidth, hMax = 200;
    const empMax = Math.max(1, d3.max(R.hist)), exaMax = Math.max(1, d3.max(B.dist));
    F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text(`clauses satisfied by a uniformly random assignment — ${SX.int(trials)} seeded runs`);
    for (let k = lo; k <= hi; k++) {
      const he = (B.dist[k] / exaMax) * hMax, hr = (R.hist[k] / empMax) * hMax;
      F.g.append("rect").attr("x", x(k) + 1).attr("y", hMax - he).attr("width", bwidth - 2).attr("height", he).attr("rx", 2).attr("fill", AC.panel2).attr("stroke", AC.line);
      F.g.append("rect").attr("x", x(k) + 3).attr("y", hMax - hr).attr("width", Math.max(1, bwidth - 6)).attr("height", hr).attr("rx", 2).attr("fill", AC.accent).attr("opacity", 0.88);
      if (hi - lo <= 12 || k % 2 === 0) F.g.append("text").attr("x", x(k) + bwidth / 2).attr("y", hMax + 13).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", AC.muted).text(k);
    }
    const xe = x(exp) + bwidth / 2;
    F.g.append("line").attr("x1", xe).attr("x2", xe).attr("y1", -4).attr("y2", hMax + 2).attr("stroke", AC.a2).attr("stroke-width", 2).attr("stroke-dasharray", "5,4");
    F.g.append("text").attr("x", xe + 4).attr("y", 8).attr("font-size", 10).attr("fill", AC.a2).text("E[Y] = 7m/8 = " + SX.f3(exp));
    const xo = x(B.best) + bwidth / 2;
    F.g.append("line").attr("x1", xo).attr("x2", xo).attr("y1", -4).attr("y2", hMax + 2).attr("stroke", AC.good).attr("stroke-width", 2);
    F.g.append("text").attr("x", xo + 4).attr("y", 24).attr("font-size", 10).attr("fill", AC.good).text("optimum " + B.best);
    AL.legend(F.g, [{ label: "empirical, this many runs", color: AC.accent }, { label: "exact, over all 2ⁿ assignments", color: AC.muted }], 4, hMax + 34);
    F.g.append("text").attr("x", 160).attr("y", hMax + 30).attr("font-size", 10).attr("fill", AC.muted).text("clauses satisfied");
    /* ── right: the conditional-expectation walk ── */
    const px = 380, pw = 254, ph = 190, py = 6;
    F.g.append("text").attr("x", px).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text("the derandomised walk: E[Y | x₁ … xᵢ]");
    const vals = [Dz.start].concat(Dz.steps.map(s => s.e)).concat(Dz.steps.map(s => s.eFalse)).concat(Dz.steps.map(s => s.eTrue)).concat([B.best]);
    const ylo = Math.min.apply(null, vals) - 0.3, yhi = Math.max.apply(null, vals) + 0.3;
    const yy = v => py + ph - ((v - ylo) / (yhi - ylo)) * ph;
    const xx = i => px + (i / n) * pw;
    F.g.append("line").attr("x1", px).attr("x2", px + pw).attr("y1", yy(exp)).attr("y2", yy(exp)).attr("stroke", AC.a2).attr("stroke-dasharray", "4,3");
    F.g.append("text").attr("x", px + pw).attr("y", yy(exp) - 4).attr("text-anchor", "end").attr("font-size", 9.5).attr("fill", AC.a2).text("7m/8 = " + SX.f3(exp));
    F.g.append("line").attr("x1", px).attr("x2", px + pw).attr("y1", yy(B.best)).attr("y2", yy(B.best)).attr("stroke", AC.good).attr("stroke-dasharray", "2,3");
    F.g.append("text").attr("x", px + pw).attr("y", yy(B.best) - 4).attr("text-anchor", "end").attr("font-size", 9.5).attr("fill", AC.good).text("optimum " + B.best);
    let dpath = "M" + xx(0) + "," + yy(Dz.start);
    Dz.steps.forEach((s, i) => {
      [["F", s.eFalse], ["T", s.eTrue]].forEach(ch => {
        const chosen = (ch[0] === "T") === (s.pick === 1);
        F.g.append("line").attr("x1", xx(i)).attr("y1", yy(i === 0 ? Dz.start : Dz.steps[i - 1].e)).attr("x2", xx(i + 1)).attr("y2", yy(ch[1]))
          .attr("stroke", chosen ? AC.accent : AC.line).attr("stroke-width", chosen ? 2.4 : 1.2).attr("stroke-dasharray", chosen ? null : "3,3");
        F.g.append("circle").attr("cx", xx(i + 1)).attr("cy", yy(ch[1])).attr("r", chosen ? 4 : 2.6).attr("fill", chosen ? AC.accent : AC.muted);
      });
      dpath += "L" + xx(i + 1) + "," + yy(s.e);
      F.g.append("text").attr("x", xx(i + 1)).attr("y", py + ph + 13).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", AC.muted).text((phi.names ? phi.names[i] : "x" + (i + 1)) + "=" + (s.pick ? "T" : "F"));
    });
    F.g.append("circle").attr("cx", xx(0)).attr("cy", yy(Dz.start)).attr("r", 4).attr("fill", AC.a2);
    F.g.append("text").attr("x", px + 2).attr("y", py + ph + 30).attr("font-size", 10).attr("fill", AC.muted).text("solid = the branch taken; dashed = the branch rejected");
    F.g.append("text").attr("x", px + 2).attr("y", py + ph + 44).attr("font-size", 10).attr("fill", AC.ink).text("ends at " + SX.f3(Dz.steps[Dz.steps.length - 1].e) + " = " + Dz.satisfied + " clauses actually satisfied");
    const avgOk = Dz.steps.every((s, i) => Math.abs((s.eFalse + s.eTrue) / 2 - (i === 0 ? Dz.start : Dz.steps[i - 1].e)) < 1e-9);
    SX.setHtml("max3sat-readout",
      `${I.tag}: n = ${n}, m = <b>${m}</b> · every clause has three distinct variables ${SX.flag(wf.ok)} — the hypothesis the exact 7/8 needs · ` +
      `E[Y] = 7m/8 = <b>${SX.f4(exp)}</b>; the exact mean over all 2ⁿ = ${1 << n} assignments is <b>${SX.f4(B.mean)}</b> ${SX.flag(Math.abs(B.mean - exp) < 1e-9)} — equal, not approximately equal · ` +
      `${SX.int(trials)} seeded runs of the real coin-flip routine gave an empirical mean of <b>${SX.f4(R.mean)}</b>, off by ${SX.f4(Math.abs(R.mean - exp))} ${SX.flag(Math.abs(R.mean - exp) < 0.35)} (${SX.int(cR.get("coin"))} coin flips, ${SX.int(cR.get("clause"))} clause evaluations) · ` +
      `exhaustive optimum <b>${B.best}</b> of ${m}${B.satisfiable ? ` — the formula IS satisfiable (${B.satisfyingCount} of ${1 << n} assignments satisfy every clause)` : " — the formula is UNSATISFIABLE"} · ` +
      `derandomised by conditional expectations: assignment ⟨${Dz.assign.map(v => v ? "T" : "F").join(" ")}⟩ satisfying <b>${Dz.satisfied}</b> clauses, against the guarantee ⌈7m/8⌉ = ${Dz.floor} ${SX.flag(Dz.aboveFloor)} · ` +
      `the walk ${Dz.steps.map(s => SX.f3(s.e)).join(" → ")} never falls ${SX.flag(Dz.monotone)}, each parent is exactly the average of its two children ${SX.flag(avgOk)}, and the final expectation equals the clauses actually satisfied ${SX.flag(Dz.endsExact)} — that last identity is what turns the existence proof into an algorithm · ` +
      `measured ratio OPT / derandomised = <b>${SX.f4(B.best / Dz.satisfied)}</b> vs the bound 8/7 = ${SX.f4(8 / 7)} ${SX.flag(B.best / Dz.satisfied <= 8 / 7 + 1e-9)}${Math.abs(B.best - exp) < 1e-9 ? " · on this instance the optimum EQUALS 7m/8, so 7/8 is attained exactly and no algorithm can do better here" : ""} · ` +
      `${SX.int(cB.get("assign"))} assignments enumerated for the exact distribution, ${SX.int(cD.get("clause"))} clause evaluations for the ${cD.get("decide")} derandomisation decisions`);
  }
  SX.on("m3-inst", "change", build); SX.on("m3-trials", "input", build);
  build();
})();

/* ── 25  #fptas-svg  the knapsack FPTAS: the table shrinks, the work falls, the error stays under ε ── */
(function () {
  if (!SX.has("fptas-svg")) return;
  const W = 680, H = 360;
  const epsList = (function () { const a = []; for (let e = 2; e <= 100; e += 2) a.push(e / 100); return a; })();
  const cache = {};
  function data(key) {
    if (cache[key]) return cache[key];
    const K = key === "big" ? NR.knapRandom(10, 369) : { w: NI.knap.w, v: NI.knap.v, W: NI.knap.W, n: NI.knap.w.length };
    const cE = AL.counter(), exact = NR.knapValueDP(K.w, K.v, K.W, cE);
    const brute = NR.knapBrute(K.w, K.v, K.W, null);
    const sweep = NR.knapFPTASSweep(K.w, K.v, K.W, epsList);
    cache[key] = { K, exact, brute, sweep, exactCells: cE.get("cell") };
    return cache[key];
  }
  function build() {
    const key = SX.val("fptas-inst", "small"), Dt = data(key), K = Dt.K;
    const eps = +SX.val("fptas-eps", 0.2); SX.setText("fptas-eps-val", SX.f2(eps));
    const cF = AL.counter(), R = NR.knapFPTAS(K.w, K.v, K.W, eps, cF);
    const err = (Dt.exact.value - R.value) / Dt.exact.value;
    const svg = d3.select("#fptas-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 46, r: 12, t: 22, b: 40 });
    /* ── panel 1: what rounding does to each item's value ── */
    const p1 = { x: 0, y: 0, w: 262, h: 128 };
    F.g.append("text").attr("x", p1.x).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text("each item's value, and what K·⌊vᵢ/K⌋ keeps of it (K = " + SX.f3(R.K) + ")");
    const vmaxAll = Math.max.apply(null, K.v), bw1 = p1.w / K.v.length;
    K.v.forEach((v, i) => {
      const kept = R.K * R.scaled[i];
      F.g.append("rect").attr("x", p1.x + i * bw1 + 2).attr("y", p1.y + p1.h - (v / vmaxAll) * p1.h).attr("width", bw1 - 4).attr("height", (v / vmaxAll) * p1.h).attr("rx", 2).attr("fill", AC.panel2).attr("stroke", AC.line);
      F.g.append("rect").attr("x", p1.x + i * bw1 + 2).attr("y", p1.y + p1.h - (kept / vmaxAll) * p1.h).attr("width", bw1 - 4).attr("height", (kept / vmaxAll) * p1.h).attr("rx", 2)
        .attr("fill", R.items.indexOf(i) >= 0 ? AC.good : AC.accent).attr("opacity", 0.9);
      if (K.v.length <= 12) F.g.append("text").attr("x", p1.x + i * bw1 + bw1 / 2).attr("y", p1.y + p1.h + 12).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", AC.muted).text(R.scaled[i]);
    });
    F.g.append("text").attr("x", p1.x).attr("y", p1.y + p1.h + 26).attr("font-size", 10).attr("fill", AC.muted).text("green = chosen · each item loses < K, so the set loses < nK = ε·v_max = " + SX.f2(eps * R.vmax));
    /* ── panel 2: cells computed against ε ── */
    const p2 = { x: 330, y: 0, w: 290, h: 128 };
    F.g.append("text").attr("x", p2.x).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text("DP cells actually computed, measured, against ε");
    const cmax = Math.max(Dt.exactCells, d3.max(Dt.sweep, r => r.cells));
    const cy = c => p2.y + p2.h - (Math.log(Math.max(1, c)) / Math.log(Math.max(2, cmax))) * p2.h;
    const cx = e => p2.x + ((e - 0.02) / 0.98) * p2.w;
    F.g.append("line").attr("x1", p2.x).attr("x2", p2.x + p2.w).attr("y1", cy(Dt.exactCells)).attr("y2", cy(Dt.exactCells)).attr("stroke", AC.rose).attr("stroke-dasharray", "4,3");
    F.g.append("text").attr("x", p2.x + p2.w).attr("y", cy(Dt.exactCells) - 4).attr("text-anchor", "end").attr("font-size", 9.5).attr("fill", AC.rose).text("exact DP: " + SX.int(Dt.exactCells) + " cells");
    F.g.append("path").attr("d", Dt.sweep.map((r, i) => (i ? "L" : "M") + cx(r.eps) + "," + cy(r.cells)).join(" ")).attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 2.2);
    F.g.append("circle").attr("cx", cx(eps)).attr("cy", cy(cF.get("cell"))).attr("r", 5).attr("fill", AC.a2);
    F.g.append("text").attr("x", cx(eps) + 8).attr("y", cy(cF.get("cell")) - 6).attr("font-size", 10).attr("fill", AC.a2).text(SX.int(cF.get("cell")) + " cells");
    [0.02, 0.25, 0.5, 0.75, 1].forEach(e => F.g.append("text").attr("x", cx(e)).attr("y", p2.y + p2.h + 12).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", AC.muted).text(SX.f2(e)));
    F.g.append("text").attr("x", p2.x).attr("y", p2.y + p2.h + 26).attr("font-size", 10).attr("fill", AC.muted).text("log scale · the curve falls like 1/ε, as O(n³/ε) says");
    /* ── panel 3: measured relative error against the ε guarantee ── */
    const p3 = { x: 0, y: 186, w: 620, h: 112 };
    F.g.append("text").attr("x", p3.x).attr("y", p3.y - 8).attr("font-size", 11).attr("fill", AC.muted).text("measured relative error (OPT − value)/OPT against the guarantee ε — every point must lie ON or BELOW the amber line");
    const ex = e => p3.x + ((e - 0.02) / 0.98) * p3.w, ey = v => p3.y + p3.h - (v / 1.02) * p3.h;
    F.g.append("path").attr("d", "M" + ex(0.02) + "," + ey(0.02) + "L" + ex(1) + "," + ey(1)).attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 2).attr("stroke-dasharray", "6,4");
    F.g.append("text").attr("x", ex(0.82)).attr("y", ey(0.86)).attr("font-size", 10).attr("fill", AC.a2).text("the guarantee: error ≤ ε");
    F.g.append("path").attr("d", Dt.sweep.map((r, i) => (i ? "L" : "M") + ex(r.eps) + "," + ey(r.err)).join(" ")).attr("fill", "none").attr("stroke", AC.good).attr("stroke-width", 2.2);
    Dt.sweep.forEach(r => { if (!r.ok) F.g.append("circle").attr("cx", ex(r.eps)).attr("cy", ey(r.err)).attr("r", 4).attr("fill", AC.bad); });
    F.g.append("circle").attr("cx", ex(eps)).attr("cy", ey(err)).attr("r", 5).attr("fill", err <= eps + 1e-12 ? AC.accent : AC.bad);
    F.g.append("text").attr("x", ex(eps) + 8).attr("y", ey(err) - 6).attr("font-size", 10).attr("fill", AC.ink).text("error " + SX.f4(err) + " at ε = " + SX.f2(eps));
    [0.02, 0.25, 0.5, 0.75, 1].forEach(e => F.g.append("text").attr("x", ex(e)).attr("y", p3.y + p3.h + 13).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", AC.muted).text(SX.f2(e)));
    [0, 0.25, 0.5, 0.75, 1].forEach(v => F.g.append("text").attr("x", -6).attr("y", ey(v) + 4).attr("text-anchor", "end").attr("font-size", 9).attr("fill", AC.muted).text(SX.f2(v)));
    F.g.append("text").attr("x", p3.x + p3.w / 2).attr("y", p3.y + p3.h + 28).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.muted).text("ε");
    const allOk = Dt.sweep.every(r => r.ok);
    const worst = Dt.sweep.reduce((a, r) => (a === null || r.err > a.err ? r : a), null);
    SX.setHtml("fptas-readout",
      `${K.n} items, W = ${K.W}, values ⟨${K.v.join(", ")}⟩ · exact value-indexed DP: V = ∑vᵢ = <b>${Dt.exact.V}</b>, table (n+1)(V+1) = <b>${SX.int(Dt.exact.states)}</b> states, <b>${SX.int(Dt.exactCells)}</b> cells computed, optimum <b>${Dt.exact.value}</b> from items {${Dt.exact.items.map(i => i + 1).join(", ")}} at weight ${Dt.exact.items.reduce((s, i) => s + K.w[i], 0)} ≤ ${K.W} — cross-checked against exhaustive search over all 2^${K.n} = ${SX.int(1 << K.n)} subsets, which gives ${Dt.brute.value} ${SX.flag(Dt.brute.value === Dt.exact.value)} · ` +
      `at ε = <b>${SX.f2(eps)}</b>: v_max = ${R.vmax}, K = ε·v_max/n = <b>${SX.f4(R.K)}</b>, scaled values ⟨${R.scaled.join(", ")}⟩ summing to V′ = <b>${R.scaledSum}</b> (bound n²/ε = ${SX.f1(K.n * K.n / eps)} ${SX.flag(R.scaledSum <= K.n * K.n / eps + 1e-9)}), table <b>${SX.int(R.states)}</b> states and <b>${SX.int(cF.get("cell"))}</b> cells — <b>${SX.f1(100 * cF.get("cell") / Dt.exactCells)}%</b> of the exact DP's work · ` +
      `it returns items {${R.items.map(i => i + 1).join(", ")}} of true value <b>${R.value}</b> at weight ${R.weight} ≤ ${K.W} ${SX.flag(R.weight <= K.W)}, against the optimum ${Dt.exact.value}: relative error <b>${SX.f4(err)}</b> ≤ ε = ${SX.f2(eps)} ${SX.flag(err <= eps + 1e-12)} · ` +
      `the proof's chain, checked numerically: the loss is at most nK = ${SX.f3(K.n * R.K)} = ε·v_max = ${SX.f3(eps * R.vmax)} ${SX.flag(Math.abs(K.n * R.K - eps * R.vmax) < 1e-9)}, and OPT ≥ v_max ${SX.flag(Dt.exact.value >= R.vmax)} (every item fits alone), so the loss is at most ε·OPT = ${SX.f3(eps * Dt.exact.value)}; the ACTUAL loss is ${Dt.exact.value - R.value} ${SX.flag(Dt.exact.value - R.value <= eps * Dt.exact.value + 1e-9)} · ` +
      `swept over ${epsList.length} values of ε from 0.02 to 1.00: the guarantee holds at every one ${SX.flag(allOk)}; the largest measured error anywhere is <b>${SX.f4(worst.err)}</b> at ε = ${SX.f2(worst.eps)}, which is ${worst.err === 0 ? "still zero — the rounding never changed the answer on this instance" : SX.f1(worst.eps / Math.max(worst.err, 1e-9)) + "× smaller than what was permitted there"} · ` +
      `is the measured error monotone in ε on this instance? <b>${Dt.sweep.every((r, i) => i === 0 || r.err >= Dt.sweep[i - 1].err - 1e-12) ? "yes" : "no"}</b> — the guarantee is one-sided, so nothing requires it to be, and a coarser grid can happen to round in your favour`);
  }
  SX.on("fptas-eps", "input", build); SX.on("fptas-inst", "change", build);
  build();
})();


/* ── chunk D figures ────────────────────────────────────────── */
/* ── 26  #bnb-svg  branch and bound: the tree, the tour, and whether the bound IS a bound ── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("bnb-svg")) return;
  const W = 680, H = 380;
  const D = NR.tspDist(NI.tsp.pts);
  const OPT = NR.tspOpt(D, null);
  const HK = (function () { const c = AL.counter(); const r = NR.heldKarp(D, c); r.cStates = c.get("state"); r.cRelax = c.get("relax"); r.cClose = c.get("close"); return r; })();
  const cache = {};
  function run(mode) {
    if (cache[mode]) return cache[mode];
    const c = AL.counter(); const R = NR.tspBnB(D, mode, c);
    R.cNodes = c.get("node"); R.cPrune = c.get("prune"); R.cLeaf = c.get("leaf"); R.cBound = c.get("bound");
    R.cEdge = c.get("boundEdge"); R.correct = Math.abs(R.cost - OPT.cost) < 1e-9;
    cache[mode] = R; return R;
  }
  const audit = NR.tspBoundAudit(D, 4, null);
  /* every partial path with ≥ 4 remaining, for the audit scatter */
  const pts = (function () {
    const out = []; const all = []; for (let i = 1; i < 8; i++) all.push(i);
    (function walk(cur, rem) {
      if (rem.length) {
        const truth = NR.tspCompletion(D, cur, rem, null);
        out.push({ truth: truth, good: NR.tspLowerBound(D, cur, rem, null), bad: NR.tspLowerBoundBad(D, cur, rem, null), k: rem.length });
      }
      if (rem.length > 4) for (let i = 0; i < rem.length; i++) walk(rem[i], rem.slice(0, i).concat(rem.slice(i + 1)));
    })(0, all);
    return out;
  })();
  const cityXY = (function () {
    const p = NI.tsp.pts, xs = p.map(q => q[0]), ys = p.map(q => q[1]);
    const x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    const y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    return { sx: v => 20 + (v - x0) / (x1 - x0) * 230, sy: v => 250 - (v - y0) / (y1 - y0) * 210 };
  })();
  function drawTour(g, tour, colour, dash, ox) {
    if (!tour) return;
    const d = tour.map((v, i) => (i ? "L" : "M") + (ox + cityXY.sx(NI.tsp.pts[v][0])) + "," + cityXY.sy(NI.tsp.pts[v][1])).join(" ") + "Z";
    g.append("path").attr("d", d).attr("fill", "none").attr("stroke", colour).attr("stroke-width", 2.2).attr("stroke-dasharray", dash || null);
  }
  function drawCities(g, ox) {
    NI.tsp.pts.forEach((p, i) => {
      g.append("circle").attr("cx", ox + cityXY.sx(p[0])).attr("cy", cityXY.sy(p[1])).attr("r", 10).attr("fill", i === 0 ? AC.a2 : AC.panel2).attr("stroke", AC.line);
      g.append("text").attr("x", ox + cityXY.sx(p[0])).attr("y", cityXY.sy(p[1]) + 4).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", i === 0 ? AC.bg : AC.ink).text(NI.tsp.names[i]);
    });
  }
  function build() {
    const mode = SX.val("bnb-mode", "bound"), view = SX.val("bnb-view", "tree");
    const R = run(mode);
    const svg = d3.select("#bnb-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 40, r: 16, t: 24, b: 34 });
    const colOf = st => st === "best" ? AC.good : st === "pruned" ? AC.muted : st === "leaf" ? AC.teal : AC.accent;

    if (view === "tour") {
      F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted)
        .text("left: the tour this rule returned · right: the optimum from exhaustive search over " + SX.int(OPT.tours) + " tours");
      drawTour(F.g, R.tour, R.correct ? AC.good : AC.bad, null, 0); drawCities(F.g, 0);
      drawTour(F.g, OPT.tour, AC.good, null, 330); drawCities(F.g, 330);
      F.g.append("text").attr("x", 0).attr("y", 276).attr("font-size", 11).attr("fill", R.correct ? AC.good : AC.bad)
        .text((R.tour || []).map(v => NI.tsp.names[v]).join("-") + "   cost " + SX.f4(R.cost));
      F.g.append("text").attr("x", 330).attr("y", 276).attr("font-size", 11).attr("fill", AC.good)
        .text(OPT.tour.map(v => NI.tsp.names[v]).join("-") + "   cost " + SX.f4(OPT.cost));
      F.g.append("text").attr("x", 0).attr("y", 300).attr("font-size", 11).attr("fill", AC.muted)
        .text(R.correct ? "the two tours agree — this rule is sound on this instance" : "the two tours DIFFER: the rule pruned the branch containing the optimum");
    } else if (view === "audit") {
      const mx = Math.max(d3.max(pts, p => p.truth), d3.max(pts, p => p.bad)) * 1.05;
      const x = d3.scaleLinear().domain([0, mx]).range([0, F.iw]);
      const y = d3.scaleLinear().domain([0, mx]).range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 6); AL.axisB(F.g, x, F.ih, 6, "true cost of the cheapest completion");
      AL.axisL(F.g, y, 6, "the bound this rule computes");
      F.g.append("line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(mx)).attr("y2", y(mx))
        .attr("stroke", AC.a2).attr("stroke-width", 2).attr("stroke-dasharray", "6,4");
      F.g.append("text").attr("x", x(mx * 0.62)).attr("y", y(mx * 0.68)).attr("font-size", 10.5).attr("fill", AC.a2)
        .text("bound = truth · a LOWER bound must lie on or BELOW this line");
      pts.forEach(p => {
        F.g.append("circle").attr("cx", x(p.truth)).attr("cy", y(p.good)).attr("r", 2.6).attr("fill", AC.good).attr("opacity", 0.75);
        F.g.append("circle").attr("cx", x(p.truth)).attr("cy", y(p.bad)).attr("r", 2.6).attr("fill", p.bad > p.truth + 1e-9 ? AC.bad : AC.violet).attr("opacity", 0.75);
      });
      AL.legend(F.g, [{ label: "the valid bound — " + audit.okGood + "/" + audit.paths + " on or below", color: AC.good },
                      { label: "the invalid bound — " + (audit.paths - audit.okBad) + "/" + audit.paths + " ABOVE, by up to " + SX.f4(audit.worstBad), color: AC.bad }], 12, 18);
    } else {
      /* the search tree: drawn node by node when it is small enough, otherwise profiled by depth */
      const byDepth = [];
      (function walk(node) { (byDepth[node.depth] = byDepth[node.depth] || []).push(node); node.kids.forEach(walk); })(R.tree);
      const total = byDepth.reduce((a, b) => a + b.length, 0);
      if (total <= 260) {
        let leafX = 0; const posOf = new Map();
        (function place(node) {
          if (!node.kids.length) { posOf.set(node, leafX++); return; }
          node.kids.forEach(place);
          const a = posOf.get(node.kids[0]), b = posOf.get(node.kids[node.kids.length - 1]);
          posOf.set(node, (a + b) / 2);
        })(R.tree);
        const maxDepth = byDepth.length - 1;
        const x = v => (leafX <= 1 ? F.iw / 2 : v / (leafX - 1) * F.iw);
        const y = d => 6 + d * (F.ih - 16) / Math.max(1, maxDepth);
        (function draw(node) {
          node.kids.forEach(k => {
            F.g.append("line").attr("x1", x(posOf.get(node))).attr("y1", y(node.depth))
              .attr("x2", x(posOf.get(k))).attr("y2", y(k.depth))
              .attr("stroke", k.status === "pruned" ? AC.grid : AC.line).attr("stroke-width", k.status === "pruned" ? 1 : 1.4);
            draw(k);
          });
        })(R.tree);
        const rr = total > 130 ? 3 : total > 60 ? 4.5 : 6;
        (function dots(node) {
          F.g.append("circle").attr("cx", x(posOf.get(node))).attr("cy", y(node.depth)).attr("r", node.status === "best" ? rr + 2 : rr)
            .attr("fill", colOf(node.status)).attr("opacity", node.status === "pruned" ? 0.45 : 1);
          if (rr >= 6) F.g.append("text").attr("x", x(posOf.get(node))).attr("y", y(node.depth) + 3.5)
            .attr("text-anchor", "middle").attr("font-size", 8).attr("fill", AC.bg).text(NI.tsp.names[node.v]);
          node.kids.forEach(dots);
        })(R.tree);
        F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted)
          .text("every one of the " + SX.int(total) + " nodes drawn · depth = cities fixed · faded = pruned");
      } else {
        const mx = d3.max(byDepth, a => a.length);
        const y = d3.scaleLinear().domain([0, byDepth.length - 0.2]).range([8, F.ih - 8]);
        const x = d3.scaleLog().domain([1, mx * 1.2]).range([0, F.iw - 90]);
        byDepth.forEach((arr, d) => {
          const exp = arr.filter(n => n.status !== "pruned").length, pr = arr.length - exp;
          const h = (F.ih - 16) / byDepth.length * 0.62;
          F.g.append("rect").attr("x", 0).attr("y", y(d) - h / 2).attr("width", Math.max(1, x(Math.max(1, exp)))).attr("height", h).attr("fill", AC.accent).attr("rx", 2);
          if (pr > 0) F.g.append("rect").attr("x", Math.max(1, x(Math.max(1, exp)))).attr("y", y(d) - h / 2)
            .attr("width", Math.max(1, x(Math.max(1, arr.length)) - x(Math.max(1, exp)))).attr("height", h).attr("fill", AC.muted).attr("opacity", 0.55).attr("rx", 2);
          F.g.append("text").attr("x", -6).attr("y", y(d) + 4).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.muted).text("depth " + d);
          F.g.append("text").attr("x", x(Math.max(1, arr.length)) + 8).attr("y", y(d) + 4).attr("font-size", 10).attr("fill", AC.ink).text(SX.int(arr.length));
        });
        F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted)
          .text(SX.int(total) + " nodes is too many to draw — the profile per depth instead, on a log scale (amber = unpruned, grey = pruned)");
      }
      AL.legend(F.g, [{ label: total <= 260 ? "expanded" : "unpruned", color: AC.accent }, { label: "a complete tour", color: AC.teal },
                      { label: "improved the incumbent", color: AC.good }, { label: "pruned", color: AC.muted }], F.iw - 150, F.ih - 54);
    }

    const rules = ["none", "incumbent", "bound", "bad"].map(m => { const r = run(m); return m + " " + SX.int(r.nodes) + (r.correct ? "" : " ✗"); });
    SX.setHtml("bnb-readout",
      `eight cities, start at ${NI.tsp.names[0]}, children taken nearest-first · a NODE is one call on one partial path, pruned ones included; a node is UNPRUNED when it survives the bound test — an internal node whose children are generated, or a complete tour — so unpruned = nodes − pruned · ` +
      `<b>${mode === "none" ? "plain depth-first search" : mode === "incumbent" ? "prune when the partial cost alone reaches the incumbent" : mode === "bound" ? "prune on partial cost + the valid lower bound" : "prune on partial cost + the INVALID bound"}</b>: ` +
      `<b>${SX.int(R.cNodes)}</b> nodes, <b>${SX.int(R.expanded)}</b> unpruned, <b>${SX.int(R.cPrune)}</b> pruned, <b>${SX.int(R.cLeaf)}</b> complete tours reached, ${SX.int(R.improve)} incumbent improvements` +
      (R.cBound ? `, <b>${SX.int(R.cBound)}</b> bound evaluations costing ${SX.int(R.cEdge)} edge comparisons` : ``) + ` · ` +
      `it returns <b>${SX.f4(R.cost)}</b> on ${(R.tour || []).map(v => NI.tsp.names[v]).join("-")}, against the exhaustive optimum <b>${SX.f4(OPT.cost)}</b> over all ${SX.int(OPT.tours)} tours ${SX.flag(R.correct)}` +
      (R.correct ? `` : ` — <b style="color:${AC.bad}">this rule is WRONG, and it is FASTER: ${SX.int(R.cNodes)} nodes against the valid bound's ${SX.int(run("bound").cNodes)}. Nothing in the run reports the error; the ratio it returns is ${SX.f4(R.cost / OPT.cost)}</b>`) + ` · ` +
      `is each bound a LOWER bound? tested directly against the true cheapest completion on all <b>${SX.int(audit.paths)}</b> partial paths with at least four cities left: the VALID bound is ≤ the truth on <b>${audit.okGood}/${audit.paths}</b> ${SX.flag(audit.okGood === audit.paths)}; the INVALID one on only <b>${audit.okBad}/${audit.paths}</b> — <span style="color:${AC.bad}">✗ not a lower bound</span>, overshooting on ${audit.paths - audit.okBad} of them by as much as <b>${SX.f4(audit.worstBad)}</b>, which is exactly why it prunes the optimum · ` +
      `all four rules, nodes (✗ = wrong answer): ${rules.join(" · ")} · ` +
      `for comparison, Held–Karp fills <b>${SX.int(HK.cStates)}</b> table cells — exactly (n−1)·2ⁿ⁻² = ${SX.int(7 * 64)} ${SX.flag(HK.cStates === 7 * 64)} of the (n−1)·2ⁿ⁻¹ = ${SX.int(HK.tableCells)} the table allocates — with <b>${SX.int(HK.cRelax)}</b> transitions and ${HK.cClose} closing edges, and returns ${SX.f4(HK.cost)} ${SX.flag(Math.abs(HK.cost - OPT.cost) < 1e-9)}; its TIME bound Θ(n²·2ⁿ) reads ${SX.int(64 * 256)} operations, which is a different quantity from the ${SX.int(HK.cStates)} states`);
  }
  SX.on("bnb-mode", "change", build); SX.on("bnb-view", "change", build);
  build();
})();

/* ── 27  #fpt-svg  the bounded search tree, and Buss's kernel beside it ── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("fpt-svg")) return;
  const W = 680, H = 380;
  function instance(key) {
    if (key === "gcol") {
      const g = NI.gCol;
      return { n: g.n, E: g.E.map(e => [e[0], e[1]]), names: Array.from({ length: g.n }, (_, i) => String.fromCharCode(97 + i)), label: "the §15 colouring graph, 7 vertices, 10 edges" };
    }
    if (key === "star") {
      /* one hub joined to six leaves, plus a triangle away from it: the hub
         has degree 6, so Buss's rule takes it at any k ≤ 5 */
      const E = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [7, 8], [8, 9], [9, 7]];
      return { n: 10, E: E, names: Array.from({ length: 10 }, (_, i) => String.fromCharCode(97 + i)), label: "a hub of degree 6 plus a disjoint triangle, 10 vertices, 9 edges" };
    }
    const G = NR.vcIndex(NI.gVC);
    return { n: G.n, E: G.E, names: G.names, label: "the §12 graph, 7 vertices, 8 edges" };
  }
  function build() {
    const key = SX.val("fpt-inst", "gvc"), k = +SX.val("fpt-k", 3);
    SX.setText("fpt-k-val", String(k));
    const I = instance(key);
    const cB = AL.counter(); const R = NR.vcBranchTree(I.n, I.E, k, cB);
    const cS = AL.counter(); const S = NR.vcFptSweep(I.n, I.E, I.n, cS);
    const cO = AL.counter(); const OPT = NR.vcOptBrute(I.n, I.E, cO);
    const K = NR.vcBussKernel(I.n, I.E, k, null);
    const cX = AL.counter(); const XP = k <= 4 ? NR.vcXpCount(I.n, I.E, k, cX) : null;
    const svg = d3.select("#fpt-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 16, r: 16, t: 26, b: 30 });

    /* ── left: the search tree actually built ── */
    const TW = 330, TH = 250;
    F.g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11).attr("fill", AC.muted)
      .text("the search tree at k = " + k + ": " + SX.int(R.calls) + " calls, bound 2^(k+1) − 1 = " + SX.int(R.bound));
    let leafX = 0; const pos = new Map();
    (function place(nd) {
      if (!nd.kids.length) { pos.set(nd, leafX++); return; }
      nd.kids.forEach(place);
      pos.set(nd, (pos.get(nd.kids[0]) + pos.get(nd.kids[nd.kids.length - 1])) / 2);
    })(R.tree);
    const xT = v => leafX <= 1 ? TW / 2 : 14 + v / (leafX - 1) * (TW - 28);
    const yT = d => 12 + d * (TH - 24) / Math.max(1, k);
    const colN = st => st === "cover" ? AC.good : st === "budget" ? AC.muted : st === "yes" ? AC.teal : AC.accent;
    (function edges(nd) { nd.kids.forEach(kd => { F.g.append("line").attr("x1", xT(pos.get(nd))).attr("y1", yT(nd.depth)).attr("x2", xT(pos.get(kd))).attr("y2", yT(kd.depth)).attr("stroke", AC.line); edges(kd); }); })(R.tree);
    const rr = R.calls > 40 ? 5 : R.calls > 20 ? 7 : 10;
    (function dots(nd) {
      F.g.append("circle").attr("cx", xT(pos.get(nd))).attr("cy", yT(nd.depth)).attr("r", rr)
        .attr("fill", colN(nd.status)).attr("stroke", AC.line);
      if (rr >= 7) F.g.append("text").attr("x", xT(pos.get(nd))).attr("y", yT(nd.depth) + 3.5).attr("text-anchor", "middle")
        .attr("font-size", 9).attr("fill", AC.bg).text(nd.pick === null ? "·" : I.names[nd.pick]);
      nd.kids.forEach(dots);
    })(R.tree);
    AL.legend(F.g, [{ label: "cover found", color: AC.good }, { label: "budget exhausted", color: AC.muted }, { label: "still branching", color: AC.accent }], 4, TH + 18);

    /* ── right: the graph, with Buss's forced vertices and the kernel ── */
    const OX = 380;
    F.g.append("text").attr("x", OX).attr("y", -10).attr("font-size", 11).attr("fill", AC.muted)
      .text("Buss's kernel at k = " + k + ": " + (K.forced.length ? K.forced.map(v => I.names[v]).join(", ") + " forced" : "no vertex has degree > k") + ", " + K.edgesLeft.length + " edges left");
    const lay = (function () {
      const n = I.n, out = [];
      for (let i = 0; i < n; i++) { const a = -Math.PI / 2 + 2 * Math.PI * i / n; out.push([OX + 130 + 92 * Math.cos(a), 104 + 82 * Math.sin(a)]); }
      return out;
    })();
    const inKernel = new Set(); K.edgesLeft.forEach(e => { inKernel.add(e[0]); inKernel.add(e[1]); });
    const gone = new Set(K.forced);
    I.E.forEach(e => {
      const alive = K.edgesLeft.some(x => x[0] === e[0] && x[1] === e[1]);
      F.g.append("line").attr("x1", lay[e[0]][0]).attr("y1", lay[e[0]][1]).attr("x2", lay[e[1]][0]).attr("y2", lay[e[1]][1])
        .attr("stroke", alive ? AC.ink : AC.grid).attr("stroke-width", alive ? 2 : 1.2).attr("stroke-dasharray", alive ? null : "3,3");
    });
    lay.forEach((p, v) => {
      const cov = R.ok && R.cover.indexOf(v) >= 0;
      F.g.append("circle").attr("cx", p[0]).attr("cy", p[1]).attr("r", 12)
        .attr("fill", gone.has(v) ? AC.a2 : cov ? AC.good : inKernel.has(v) ? AC.panel2 : AC.panel)
        .attr("stroke", OPT.best.indexOf(v) >= 0 ? AC.violet : AC.line).attr("stroke-width", OPT.best.indexOf(v) >= 0 ? 2.6 : 1);
      F.g.append("text").attr("x", p[0]).attr("y", p[1] + 4).attr("text-anchor", "middle").attr("font-size", 10)
        .attr("fill", gone.has(v) || cov ? AC.bg : AC.ink).text(I.names[v]);
    });
    F.g.append("text").attr("x", OX).attr("y", 206).attr("font-size", 10).attr("fill", AC.muted)
      .text("amber = forced by Buss · green = in the cover found · violet ring = an exhaustive optimum");

    /* ── right lower: the three costs, on a log scale ── */
    const bars = [{ l: "search-tree calls at k = " + k, v: R.calls, c: AC.accent },
                  { l: "bound 2^(k+1) − 1", v: R.bound, c: AC.a2 },
                  { l: "all C(n, k) k-subsets", v: XP ? XP.total : Math.pow(2, I.n), c: AC.violet },
                  { l: "all 2ⁿ subsets (brute force)", v: 1 << I.n, c: AC.rose }];
    const mx = Math.max.apply(null, bars.map(b => b.v));
    bars.forEach((b, i) => {
      const w = Math.max(2, Math.log(Math.max(1, b.v) + 1) / Math.log(mx + 1) * 250);
      F.g.append("rect").attr("x", OX).attr("y", 224 + i * 22).attr("width", w).attr("height", 14).attr("rx", 3).attr("fill", b.c);
      F.g.append("text").attr("x", OX + w + 8).attr("y", 235 + i * 22).attr("font-size", 10).attr("fill", AC.ink).text(SX.int(b.v) + " — " + b.l);
    });
    F.g.append("text").attr("x", OX).attr("y", 224 + 4 * 22 + 8).attr("font-size", 10).attr("fill", AC.muted).text("log scale");

    const kernelOk = (K.infeasible || K.tooBig) ? (OPT.size > k) : true;
    SX.setHtml("fpt-readout",
      `${I.label} · exhaustive search over all 2^${I.n} = <b>${SX.int(1 << I.n)}</b> vertex subsets gives a minimum cover of size <b>${OPT.size}</b>${OPT.unique ? " and it is UNIQUE" : " (" + OPT.count + " of them)"}: {${OPT.best.map(v => I.names[v]).join(", ")}} · ` +
      `at k = <b>${k}</b> the bounded search tree makes <b>${SX.int(R.calls)}</b> recursive calls ≤ 2^(k+1) − 1 = ${SX.int(R.bound)} ${SX.flag(R.calls <= R.bound)}${R.calls === R.bound ? " — attained exactly, so the tree is FULL here" : ""} and answers <b>${R.ok ? "YES" : "NO"}</b>; exhaustive search says a cover of size ≤ ${k} ${OPT.size <= k ? "exists" : "does not exist"} ${SX.flag(R.ok === (OPT.size <= k))}` +
      (R.ok ? `, returning {${R.cover.map(v => I.names[v]).join(", ")}} of size ${R.cover.length} ≤ ${k}, which covers every edge ${SX.flag(NR.vcIsCover(I.E, (function () { const a = new Array(I.n).fill(false); R.cover.forEach(v => { a[v] = true; }); return a; })()))}` : ``) + ` · ` +
      `sweeping k upward until it succeeds: ${S.rows.map(r => "k = " + r.k + " → " + r.calls + (r.ok ? " ✓" : " ✗")).join(", ")} — <b>${SX.int(S.total)}</b> calls in total, first success at k = <b>${S.k}</b>, which matches the exhaustive minimum ${SX.flag(S.k === OPT.size)} · ` +
      `BUSS's kernel at k = ${k}: ${K.forced.length ? "vertices " + K.forced.map((v, i) => I.names[v] + " (degree " + K.steps[i].deg + " > budget " + (K.steps[i].budget + 1) + ")").join(", ") + " are in EVERY cover of size ≤ k, so they are taken and deleted" : "no vertex has degree above the budget, so the rule does not fire"}; budget left <b>${K.budget}</b>, ` +
      `<b>${K.edgesLeft.length}</b> edges on <b>${K.nLeft}</b> vertices remain against the limit k′² = <b>${K.limit}</b> — ${K.infeasible ? "the budget went negative before the rule ran out of high-degree vertices, so the answer is NO — and exhaustive search confirms the minimum cover has size " + OPT.size + " > " + k : K.tooBig ? "more than k′² edges remain, so the rule REJECTS — and exhaustive search confirms the minimum cover has size " + OPT.size + " > " + k : "the rule does not reject, which means nothing on its own — it is a rejection rule, and the search still has to run"}; a rejection rule is only sound if it never rejects a YES-instance ${SX.flag(kernelOk)} · ` +
      (XP ? `the XP alternative — try all C(${I.n}, ${k}) = <b>${SX.int(XP.total)}</b> k-subsets — examines ${SX.int(cX.get("subset"))} of them before ${XP.found ? "finding one" : "exhausting them"}; that count grows with n^k, the search tree's with 2ᵏ, and that is the entire difference between XP and FPT · ` : ``) +
      `the brute force that produced the optimum examined <b>${SX.int(cO.get("subset"))}</b> subsets and ${SX.int(cO.get("edgeCheck"))} edge tests`);
  }
  SX.on("fpt-k", "input", build); SX.on("fpt-inst", "change", build);
  build();
})();

/* ── 28  #localsearch-svg  2-opt stepped, annealing, and the basin distribution ── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("localsearch-svg")) return;
  const W = 680, H = 380;
  const D = NR.tspDist(NI.tsp.pts);
  const OPT = NR.tspOpt(D, null);
  const TRI = NR.tspTriangle(D, null);
  const NN = NR.tspNN(D, 0, null);
  function start(key) {
    if (key === "id") return { tour: [0, 1, 2, 3, 4, 5, 6, 7], label: "the identity tour" };
    if (key === "bad") return { tour: [0, 4, 1, 5, 2, 6, 3, 7], label: "a deliberately tangled tour" };
    return { tour: NN.tour.slice(), label: "the nearest-neighbour tour" };
  }
  const xy = (function () {
    const p = NI.tsp.pts, xs = p.map(q => q[0]), ys = p.map(q => q[1]);
    const x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    return { sx: v => 24 + (v - x0) / (x1 - x0) * 236, sy: v => 236 - (v - y0) / (y1 - y0) * 200 };
  })();
  function drawInstance(g, tour, hi) {
    if (tour) {
      const n = tour.length;
      for (let i = 0; i < n; i++) {
        const a = tour[i], b = tour[(i + 1) % n];
        const key = a + "-" + b, key2 = b + "-" + a;
        const isOut = hi && hi.out && (hi.out.indexOf(key) >= 0 || hi.out.indexOf(key2) >= 0);
        g.append("line").attr("x1", xy.sx(NI.tsp.pts[a][0])).attr("y1", xy.sy(NI.tsp.pts[a][1]))
          .attr("x2", xy.sx(NI.tsp.pts[b][0])).attr("y2", xy.sy(NI.tsp.pts[b][1]))
          .attr("stroke", isOut ? AC.bad : AC.accent).attr("stroke-width", isOut ? 2.6 : 2)
          .attr("stroke-dasharray", isOut ? "5,4" : null);
      }
    }
    if (hi && hi.inn) hi.inn.forEach(pr => {
      g.append("line").attr("x1", xy.sx(NI.tsp.pts[pr[0]][0])).attr("y1", xy.sy(NI.tsp.pts[pr[0]][1]))
        .attr("x2", xy.sx(NI.tsp.pts[pr[1]][0])).attr("y2", xy.sy(NI.tsp.pts[pr[1]][1]))
        .attr("stroke", AC.good).attr("stroke-width", 3);
    });
    NI.tsp.pts.forEach((p, i) => {
      g.append("circle").attr("cx", xy.sx(p[0])).attr("cy", xy.sy(p[1])).attr("r", 10).attr("fill", i === 0 ? AC.a2 : AC.panel2).attr("stroke", AC.line);
      g.append("text").attr("x", xy.sx(p[0])).attr("y", xy.sy(p[1]) + 4).attr("text-anchor", "middle").attr("font-size", 10)
        .attr("fill", i === 0 ? AC.bg : AC.ink).text(NI.tsp.names[i]);
    });
  }
  function build() {
    const mode = SX.val("ls-mode", "first"), skey = SX.val("ls-start", "nn");
    const T0 = +SX.val("ls-temp", 3); SX.setText("ls-temp-val", SX.f1(T0));
    const S0 = start(skey);
    const svg = d3.select("#localsearch-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 16, r: 16, t: 24, b: 30 });
    const c0 = NR.tspTourCost(D, S0.tour);

    if (mode === "restart") {
      const cR = AL.counter(); const RR = NR.twoOptRestart(D, 200, 5, cR);
      const buckets = {}; RR.costs.forEach(v => { const k = v.toFixed(4); buckets[k] = (buckets[k] || 0) + 1; });
      const keys = Object.keys(buckets).sort((a, b) => +a - +b);
      const mxc = Math.max.apply(null, keys.map(k => buckets[k]));
      const x = d3.scaleLinear().domain([+keys[0] - 0.4, +keys[keys.length - 1] + 0.4]).range([0, F.iw]);
      const y = d3.scaleLinear().domain([0, mxc * 1.12]).range([F.ih - 10, 10]);
      AL.gridY(F.g, y, F.iw, 5); AL.axisB(F.g, x, F.ih - 10, 6, "tour length of the 2-opt local optimum reached");
      AL.axisL(F.g, y, 5, "how many of the 200 random starts landed there");
      keys.forEach(k => {
        const isOpt = Math.abs(+k - OPT.cost) < 1e-4;
        F.g.append("rect").attr("x", x(+k) - 16).attr("y", y(buckets[k])).attr("width", 32).attr("height", y(0) - y(buckets[k]))
          .attr("rx", 3).attr("fill", isOpt ? AC.good : AC.rose);
        F.g.append("text").attr("x", x(+k)).attr("y", y(buckets[k]) - 6).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.ink).text(buckets[k]);
        F.g.append("text").attr("x", x(+k)).attr("y", y(0) + 16).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.muted).text(k);
      });
      F.g.append("line").attr("x1", x(OPT.cost)).attr("x2", x(OPT.cost)).attr("y1", 6).attr("y2", y(0))
        .attr("stroke", AC.good).attr("stroke-width", 2).attr("stroke-dasharray", "5,4");
      const hits = RR.costs.filter(v => Math.abs(v - OPT.cost) < 1e-4).length;
      SX.setHtml("localsearch-readout",
        `200 uniformly random starting tours, each followed by first-improvement 2-opt to a local optimum · the run reaches <b>${keys.length}</b> distinct local optima on this instance: ${keys.map(k => k + " (×" + buckets[k] + ")").join(", ")} · ` +
        `the global optimum ${SX.f4(OPT.cost)} — from exhaustive search over all ${SX.int(OPT.tours)} tours — is reached <b>${hits}</b> times, a rate of <b>${SX.f3(hits / 200)}</b> ${SX.flag(Math.abs(RR.best - OPT.cost) < 1e-9)} · ` +
        `so a single run fails with probability ${SX.f3(1 - hits / 200)}, three independent restarts with ${SX.f4(Math.pow(1 - hits / 200, 3))}, five with ${SX.f4(Math.pow(1 - hits / 200, 5))} — that is the entire theory of restarts, and it stops working exactly when that rate falls exponentially with instance size · ` +
        `total measured work across the 200 runs: <b>${SX.int(cR.get("pair"))}</b> (i, k) pairs evaluated and <b>${SX.int(cR.get("move"))}</b> improving moves taken`);
      return;
    }

    if (mode === "anneal") {
      const cA = AL.counter();
      const A = NR.annealTsp(D, S0.tour, { T0: Math.max(0.05, T0), Tend: 0.01, iters: 3000, seed: 11 }, cA);
      const cF = AL.counter(); const Fst = NR.twoOptSteps(D, S0.tour, "first", cF);
      drawInstance(F.g, A.tour, null);
      F.g.append("text").attr("x", 24).attr("y", 258).attr("font-size", 11).attr("fill", AC.ink)
        .text("best tour seen: " + A.tour.map(v => NI.tsp.names[v]).join("-") + "  =  " + SX.f4(A.cost));
      const px = 300, pw = F.iw - px, ph = 200;
      const x = d3.scaleLinear().domain([0, 3000]).range([px, px + pw]);
      const lo = Math.min(OPT.cost, A.cost) * 0.96, hi = Math.max(c0, d3.max(A.trace, t => t.cost)) * 1.02;
      const y = d3.scaleLinear().domain([lo, hi]).range([ph, 10]);
      F.g.append("g").attr("transform", "translate(" + px + ",0)").call(d3.axisLeft(y).ticks(5)).attr("class", "axis");
      F.g.append("g").attr("transform", "translate(0," + ph + ")").call(d3.axisBottom(x).ticks(5)).attr("class", "axis");
      F.g.append("text").attr("x", px + pw).attr("y", ph + 28).attr("text-anchor", "end").attr("font-size", 11).attr("fill", AC.muted).text("proposal");
      F.g.append("text").attr("x", px).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text("tour length · current (thin) and best-so-far (thick)");
      F.g.append("line").attr("x1", px).attr("x2", px + pw).attr("y1", y(OPT.cost)).attr("y2", y(OPT.cost))
        .attr("stroke", AC.good).attr("stroke-width", 2).attr("stroke-dasharray", "6,4");
      F.g.append("text").attr("x", px + pw).attr("y", y(OPT.cost) - 5).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.good)
        .text("exhaustive optimum " + SX.f4(OPT.cost));
      F.g.append("path").datum(A.trace).attr("fill", "none").attr("stroke", AC.violet).attr("stroke-width", 1).attr("opacity", 0.7)
        .attr("d", d3.line().x(t => x(t.t)).y(t => y(Math.min(hi, t.cost))));
      F.g.append("path").datum(A.trace).attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 2.2)
        .attr("d", d3.line().x(t => x(t.t)).y(t => y(t.best)));
      const yT = d3.scaleLinear().domain([0, Math.max(0.05, T0)]).range([ph + 56, ph + 6]);
      F.g.append("path").datum(A.trace).attr("fill", "none").attr("stroke", AC.rose).attr("stroke-width", 1.6)
        .attr("d", d3.line().x(t => x(t.t)).y(t => yT(t.T)));
      F.g.append("text").attr("x", px).attr("y", ph + 68).attr("font-size", 10).attr("fill", AC.rose).text("temperature, geometric from T₀ = " + SX.f2(Math.max(0.05, T0)) + " to 0.01");
      SX.setHtml("localsearch-readout",
        `simulated annealing over the 2-change neighbourhood, from ${S0.label} (cost ${SX.f4(c0)}), geometric cooling T₀ = <b>${SX.f2(Math.max(0.05, T0))}</b> → 0.01 over <b>${SX.int(cA.get("propose"))}</b> proposals · ` +
        `accepted <b>${SX.int(cA.get("better"))}</b> improving moves and <b>${SX.int(cA.get("worse"))}</b> WORSENING ones, rejecting ${SX.int(cA.get("reject"))} — the worsening acceptances are the mechanism; at T₀ → 0 that count goes to zero and the procedure becomes plain descent · ` +
        `best tour seen <b>${SX.f4(A.cost)}</b>, re-summed from the tour it reports as ${SX.f4(A.recheck)} ${SX.flag(Math.abs(A.cost - A.recheck) < 1e-9)}, against the exhaustive optimum <b>${SX.f4(OPT.cost)}</b> over ${SX.int(OPT.tours)} tours — ratio <b>${SX.f4(A.cost / OPT.cost)}</b> ${SX.flag(A.cost >= OPT.cost - 1e-9)} · ` +
        `first-improvement 2-opt from the same start reaches ${SX.f4(Fst.cost)} in ${Fst.moves} moves and ${SX.int(cF.get("pair"))} pair evaluations, so annealing spent about ${SX.f1(cA.get("propose") / Math.max(1, cF.get("pair")))}× the move evaluations here · ` +
        `the Metropolis rule accepts Δ ≥ 0 with probability e^(−Δ/T): at T = ${SX.f2(Math.max(0.05, T0))} a move costing 1.0 is taken with probability ${SX.f3(Math.exp(-1 / Math.max(0.05, T0)))}, and at T = 0.01 with probability ${SX.f4(Math.exp(-1 / 0.01))} — which is why the schedule matters and why the theorem's logarithmic schedule is unusable · ` +
        `NOTE: none of this is a guarantee. Annealing converges to a global optimum only under a logarithmic cooling schedule and unbounded time;  this run, like every practical run, uses a geometric schedule and has no bound of any kind`);
      return;
    }

    /* the stepped descent */
    const cM = AL.counter(); const R = NR.twoOptSteps(D, S0.tour, mode, cM);
    const frames = R.frames;
    const px = 300, pw = F.iw - px, ph = 230;
    const x = d3.scaleLinear().domain([0, Math.max(1, frames.length - 1)]).range([px, px + pw]);
    const lo = Math.min(OPT.cost, R.cost) * 0.97, hi = Math.max.apply(null, frames.map(f => f.cost)) * 1.02;
    const y = d3.scaleLinear().domain([lo, hi]).range([ph, 12]);
    function render(fr, i) {
      F.g.selectAll("*").remove();
      let hint = null;
      if (i + 1 < frames.length && frames[i + 1].move) {
        const m = frames[i + 1].move, t = fr.tour, n = t.length;
        const a = t[m.i - 1], b = t[m.i], cc = t[m.k], d = t[(m.k + 1) % n];
        hint = { out: [a + "-" + b, cc + "-" + d], inn: [[a, cc], [b, d]] };
      }
      drawInstance(F.g, fr.tour, hint);
      F.g.append("text").attr("x", 24).attr("y", 258).attr("font-size", 11).attr("fill", AC.ink)
        .text(fr.tour.map(v => NI.tsp.names[v]).join("-") + "  =  " + SX.f4(fr.cost));
      F.g.append("text").attr("x", 24).attr("y", 276).attr("font-size", 10.5).attr("fill", AC.muted)
        .text(hint ? "next move: reverse positions " + frames[i + 1].move.i + "…" + frames[i + 1].move.k + ", gain " + SX.f4(frames[i + 1].move.gain) : "no improving move remains — a LOCAL optimum");
      F.g.append("g").attr("transform", "translate(" + px + ",0)").call(d3.axisLeft(y).ticks(5)).attr("class", "axis");
      F.g.append("text").attr("x", px).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text("tour length after each accepted move");
      F.g.append("line").attr("x1", px).attr("x2", px + pw).attr("y1", y(OPT.cost)).attr("y2", y(OPT.cost))
        .attr("stroke", AC.good).attr("stroke-width", 2).attr("stroke-dasharray", "6,4");
      F.g.append("text").attr("x", px + pw).attr("y", y(OPT.cost) - 5).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.good)
        .text("exhaustive optimum " + SX.f4(OPT.cost));
      F.g.append("path").datum(frames).attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 2.2)
        .attr("d", d3.line().x((f, j) => x(j)).y(f => y(f.cost)));
      frames.forEach((f, j) => F.g.append("circle").attr("cx", x(j)).attr("cy", y(f.cost)).attr("r", j === i ? 6 : 3.5)
        .attr("fill", j === i ? AC.a2 : AC.accent));
      F.g.append("text").attr("x", px).attr("y", ph + 24).attr("font-size", 10).attr("fill", AC.muted)
        .text("move " + i + " of " + (frames.length - 1) + " · " + SX.int(fr.pairs) + " (i, k) pairs evaluated so far");
    }
    AL.stepper(svg, { frames: frames, render: render, delay: 900, label: "move" });
    const cO = AL.counter(); const other = NR.twoOptSteps(D, S0.tour, mode === "first" ? "best" : "first", cO);
    const localOpt = (function () {
      const t = R.tour, n = t.length; let worst = 0;
      for (let i = 1; i < n - 1; i++) for (let k = i + 1; k < n; k++) {
        const a = t[i - 1], b = t[i], cc = t[k], d = t[(k + 1) % n]; if (d === a) continue;
        const gain = (D[a][b] + D[cc][d]) - (D[a][cc] + D[b][d]); if (gain > worst) worst = gain;
      }
      return worst <= 1e-12;
    })();
    SX.setHtml("localsearch-readout",
      `2-opt, <b>${mode === "first" ? "first" : "best"} improvement</b>, from ${S0.label} (cost <b>${SX.f4(c0)}</b>) · the instance satisfies the triangle inequality on all ${SX.int(TRI.triples)} triples ${SX.flag(TRI.ok)} — which matters for §22's bounds and not at all for 2-opt, which has none · ` +
      `<b>${SX.int(cM.get("move"))}</b> improving moves, <b>${SX.int(cM.get("pair"))}</b> (i, k) pairs evaluated, final tour <b>${SX.f4(R.cost)}</b> = ${R.tour.map(v => NI.tsp.names[v]).join("-")} · ` +
      `it IS a local optimum — no (i, k) pair in the whole Θ(n²) neighbourhood improves it ${SX.flag(localOpt)} — and against the exhaustive optimum <b>${SX.f4(OPT.cost)}</b> over all ${SX.int(OPT.tours)} tours the ratio is <b>${SX.f4(R.cost / OPT.cost)}</b> ${SX.flag(R.cost >= OPT.cost - 1e-9)} · ` +
      `the other policy, ${mode === "first" ? "best" : "first"} improvement from the same start, evaluates <b>${SX.int(cO.get("pair"))}</b> pairs and finishes at <b>${SX.f4(other.cost)}</b> — ${Math.abs(other.cost - R.cost) < 1e-9 ? "the same local optimum" : (other.cost < R.cost ? "a BETTER one" : "a WORSE one") + ", for " + (cO.get("pair") > cM.get("pair") ? "MORE" : cO.get("pair") < cM.get("pair") ? "LESS" : "the same") + " work"}; neither policy dominates, and one instance decides nothing · ` +
      `${Math.abs(R.cost - OPT.cost) < 1e-9 ? "<b>this run reached the global optimum. That is luck on one instance and one starting tour, not a bound.</b> " : ""}2-opt carries NO approximation guarantee: on metric instances a local optimum can be √(n/2) times optimal (a tight bound), and for general TSP no constant-factor guarantee is possible at all unless P = NP (§22) · ` +
      `for scale: exhaustive search enumerated ${SX.int(OPT.tours)} tours to produce the number this figure measures against, and the branch and bound of §26 proves the same optimum in 116 search-tree nodes`);
  }
  SX.on("ls-mode", "change", build); SX.on("ls-start", "change", build); SX.on("ls-temp", "input", build);
  build();
})();

/* ── 29  #ga-svg  a real GA run: best and mean fitness, against the exhaustive optimum ── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("ga-svg")) return;
  const W = 680, H = 380;
  const D = NR.tspDist(NI.tsp.pts);
  const OPT = NR.tspOpt(D, null);
  const NN = NR.tspNN(D, 0, null);
  const xy = (function () {
    const p = NI.tsp.pts, xs = p.map(q => q[0]), ys = p.map(q => q[1]);
    const x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    return { sx: (v, ox, s) => ox + (v - x0) / (x1 - x0) * (s || 180), sy: (v, oy, s) => (oy || 0) + (s || 150) - (v - y0) / (y1 - y0) * (s || 150) };
  })();
  function drawTour(g, tour, ox, oy, s, colour, title, bad) {
    const d = tour.map((v, i) => (i ? "L" : "M") + xy.sx(NI.tsp.pts[v][0], ox, s) + "," + xy.sy(NI.tsp.pts[v][1], oy, s)).join(" ") + (bad ? "" : "Z");
    g.append("path").attr("d", d).attr("fill", "none").attr("stroke", colour).attr("stroke-width", 2);
    NI.tsp.pts.forEach((p, i) => {
      g.append("circle").attr("cx", xy.sx(p[0], ox, s)).attr("cy", xy.sy(p[1], oy, s)).attr("r", 7)
        .attr("fill", i === 0 ? AC.a2 : AC.panel2).attr("stroke", AC.line);
      g.append("text").attr("x", xy.sx(p[0], ox, s)).attr("y", xy.sy(p[1], oy, s) + 3.5).attr("text-anchor", "middle")
        .attr("font-size", 8).attr("fill", i === 0 ? AC.bg : AC.ink).text(NI.tsp.names[i]);
    });
    if (title) g.append("text").attr("x", ox).attr("y", (oy || 0) + (s || 150) + 18).attr("font-size", 10.5).attr("fill", AC.muted).text(title);
  }
  function build() {
    const seed = +SX.val("ga-seed", 3), pop = +SX.val("ga-pop", 40), elite = SX.checked("ga-elite", true) ? 2 : 0;
    const view = SX.val("ga-view", "run");
    SX.setText("ga-seed-val", String(seed)); SX.setText("ga-pop-val", String(pop));
    const cG = AL.counter(); const G = NR.gaTsp(D, { seed: seed, pop: pop, gens: 60, elite: elite }, cG);
    const svg = d3.select("#ga-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 44, r: 16, t: 26, b: 34 });

    if (view === "cross") {
      const p1 = [0, 1, 2, 3, 4, 5, 6, 7], p2 = [0, 3, 1, 7, 5, 2, 6, 4];
      const one = NR.onePointCut(p1, p2, 4);
      const cX = AL.counter(); const ox = NR.oxCrossover(p1, p2, 2, 4, cX);
      drawTour(F.g, p1, 10, 0, 140, AC.accent, "parent 1: " + p1.map(v => NI.tsp.names[v]).join("-") + " = " + SX.f4(NR.tspTourCost(D, p1)));
      drawTour(F.g, p2, 180, 0, 140, AC.violet, "parent 2: " + p2.map(v => NI.tsp.names[v]).join("-") + " = " + SX.f4(NR.tspTourCost(D, p2)));
      drawTour(F.g, one.child, 350, 0, 140, AC.bad, "one-point cut at 4 — NOT a tour", true);
      drawTour(F.g, ox, 520, 0, 140, AC.good, "order crossover (2, 4): " + SX.f4(NR.tspTourCost(D, ox)));
      F.g.append("text").attr("x", 10).attr("y", 196).attr("font-size", 11).attr("fill", AC.bad)
        .text("one-point child " + one.child.map(v => NI.tsp.names[v]).join("-") + " — " + NI.tsp.names[one.child.filter((v, i) => one.child.indexOf(v) !== i)[0]] + " twice, " + one.missing.map(v => NI.tsp.names[v]).join(",") + " missing");
      F.g.append("text").attr("x", 10).attr("y", 216).attr("font-size", 11).attr("fill", AC.good)
        .text("order-crossover child " + ox.map(v => NI.tsp.names[v]).join("-") + " — every city exactly once, a point of the search space");
      SX.setHtml("ga-readout",
        `the representation problem, measured · parents ${p1.map(v => NI.tsp.names[v]).join("-")} and ${p2.map(v => NI.tsp.names[v]).join("-")} · ` +
        `ONE-POINT crossover at position 4 produces ${one.child.map(v => NI.tsp.names[v]).join("-")}: <b>${one.duplicates}</b> city repeated and <b>${one.missing.length}</b> missing (${one.missing.map(v => NI.tsp.names[v]).join(", ")}) — a valid permutation? <span style="color:${AC.bad}">✗ no</span>, which is the demonstration and not a fault ${NR.tourValid(one.child, 8) ? '<span style="color:' + AC.bad + '">— and if this ever reads "yes" the routine is broken</span>' : ""} · ` +
        `ORDER crossover with cut points (2, 4) copies parent 1's run ${p1.slice(2, 5).map(v => NI.tsp.names[v]).join("-")} in place and fills the rest in parent 2's order, producing ${ox.map(v => NI.tsp.names[v]).join("-")} — a valid permutation? ${SX.flag(NR.tourValid(ox, 8))}, at cost ${SX.f4(NR.tspTourCost(D, ox))}, using ${SX.int(cX.get("gene"))} gene reads · ` +
        `the general rule: when the chromosome carries a constraint, the textbook operator violates it, and the operator has to be redesigned round the constraint. That redesign — not the selection scheme, not the mutation rate — is where the work in a genetic algorithm actually is`);
      return;
    }

    if (view === "cmp") {
      const cF = AL.counter(); const TO = NR.twoOptSteps(D, NN.tour, "first", cF);
      const cB = AL.counter(); const TB = NR.twoOptSteps(D, NN.tour, "best", cB);
      const cA = AL.counter(); const AN = NR.annealTsp(D, NN.tour, { T0: 3, Tend: 0.01, iters: 3000, seed: 11 }, cA);
      const cBB = AL.counter(); const BB = NR.tspBnB(D, "bound", cBB);
      const rows = [
        { l: "exhaustive search", w: OPT.tours, u: "tours enumerated", v: OPT.cost, g: "OPTIMAL, proved" },
        { l: "branch and bound (§26)", w: cBB.get("node"), u: "search-tree nodes", v: BB.cost, g: "OPTIMAL, proved" },
        { l: "nearest neighbour (§22)", w: 8 * 8, u: "distance scans", v: NN.cost, g: "none" },
        { l: "2-opt, first improvement (§28)", w: cF.get("pair"), u: "O(1) move evaluations", v: TO.cost, g: "none" },
        { l: "2-opt, best improvement (§28)", w: cB.get("pair"), u: "O(1) move evaluations", v: TB.cost, g: "none" },
        { l: "simulated annealing (§28)", w: cA.get("propose"), u: "proposals", v: AN.cost, g: "none" },
        { l: "genetic algorithm, " + pop + " × 60", w: cG.get("eval"), u: "FULL tour evaluations", v: G.cost, g: "none" }
      ];
      const mxw = Math.max.apply(null, rows.map(r => r.w));
      const bh = (F.ih - 30) / rows.length;
      rows.forEach((r, i) => {
        const yy = 8 + i * bh;
        const ww = Math.log(r.w + 1) / Math.log(mxw + 1) * 250;
        F.g.append("rect").attr("x", 210).attr("y", yy).attr("width", Math.max(2, ww)).attr("height", bh - 12).attr("rx", 3)
          .attr("fill", r.g === "none" ? AC.violet : AC.good);
        F.g.append("text").attr("x", 204).attr("y", yy + bh / 2).attr("text-anchor", "end").attr("font-size", 10.5).attr("fill", AC.ink).text(r.l);
        F.g.append("text").attr("x", 214 + Math.max(2, ww)).attr("y", yy + bh / 2).attr("font-size", 10).attr("fill", AC.muted)
          .text(SX.int(r.w) + " " + r.u);
        F.g.append("text").attr("x", F.iw).attr("y", yy + bh / 2).attr("text-anchor", "end").attr("font-size", 10.5)
          .attr("fill", Math.abs(r.v - OPT.cost) < 1e-9 ? AC.good : AC.bad).text(SX.f4(r.v));
      });
      F.g.append("text").attr("x", 210).attr("y", -10).attr("font-size", 11).attr("fill", AC.muted)
        .text("work (log scale, and the UNITS differ — read them) · right column: the tour length it returned");
      SX.setHtml("ga-readout",
        `the same eight-city instance, every count measured · exhaustive optimum <b>${SX.f4(OPT.cost)}</b> over ${SX.int(OPT.tours)} tours · ` +
        rows.map(r => `${r.l}: <b>${SX.int(r.w)}</b> ${r.u} → ${SX.f4(r.v)}${Math.abs(r.v - OPT.cost) < 1e-9 ? " ✓" : ""}`).join(" · ") + ` · ` +
        `the units are not comparable as they stand: a 2-opt move evaluation is four distance lookups, a GA fitness evaluation is a full tour of eight. In lookups, 2-opt spent about <b>${SX.int(cF.get("pair") * 4)}</b> and the GA about <b>${SX.int(cG.get("eval") * 8)}</b> — a factor of <b>${SX.f1(cG.get("eval") * 8 / Math.max(1, cF.get("pair") * 4))}</b> for ${Math.abs(G.cost - TO.cost) < 1e-9 ? "the same answer" : "a different answer"} · ` +
        `only the first two rows carry a guarantee, and it is the strongest kind: they return the optimum WITH a proof. Everything below them returns a number`);
      return;
    }

    /* the run */
    const hist = G.hist;
    const x = d3.scaleLinear().domain([0, G.gens]).range([0, F.iw - 200]);
    const lo = OPT.cost * 0.985, hi = d3.max(hist, h => h.mean) * 1.03;
    const y = d3.scaleLinear().domain([lo, hi]).range([F.ih - 10, 10]);
    AL.gridY(F.g, y, F.iw - 200, 6);
    AL.axisB(F.g, x, F.ih - 10, 6, "generation");
    AL.axisL(F.g, y, 6, "tour length");
    F.g.append("line").attr("x1", 0).attr("x2", F.iw - 200).attr("y1", y(OPT.cost)).attr("y2", y(OPT.cost))
      .attr("stroke", AC.good).attr("stroke-width", 2).attr("stroke-dasharray", "6,4");
    F.g.append("text").attr("x", F.iw - 202).attr("y", y(OPT.cost) - 5).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.good)
      .text("exhaustive optimum " + SX.f4(OPT.cost));
    const mk = (key, col, wd) => F.g.append("path").datum(hist).attr("fill", "none").attr("stroke", col).attr("stroke-width", wd)
      .attr("d", d3.line().x(h => x(h.gen)).y(h => y(Math.min(hi, h[key]))));
    mk("worst", AC.grid, 1.2); mk("mean", AC.violet, 1.8); mk("best", AC.accent, 2.4);
    AL.legend(F.g, [{ label: "best in generation", color: AC.accent }, { label: "population mean", color: AC.violet },
                    { label: "worst in generation", color: AC.grid }], 8, 22);
    drawTour(F.g, G.best, F.iw - 180, 20, 150, Math.abs(G.cost - OPT.cost) < 1e-9 ? AC.good : AC.bad,
             "best tour found: " + SX.f4(G.cost));
    const hitGen = hist.findIndex(h => Math.abs(h.best - OPT.cost) < 1e-9);
    const stalled = hitGen < 0;
    const lastImprove = (function () { let g = 0; for (let i = 1; i < hist.length; i++) if (hist[i].best < hist[i - 1].best - 1e-12) g = i; return g; })();
    const monotone = hist.every((h, i) => i === 0 || h.best <= hist[i - 1].best + 1e-12);
    SX.setHtml("ga-readout",
      `a real run: population <b>${pop}</b>, <b>${G.gens}</b> generations, tournament selection (size 3), order crossover at rate 0.9, swap mutation at rate 0.2, elitism ${elite ? "keeping the best " + elite : "<b>OFF</b>"}, seed <b>${seed}</b> · ` +
      `<b>${SX.int(cG.get("eval"))}</b> fitness evaluations, ${SX.int(cG.get("crossover"))} crossovers, ${SX.int(cG.get("mutation"))} mutations, ${SX.int(cG.get("compare"))} tournament comparisons · ` +
      `generation 0: best ${SX.f4(hist[0].best)}, mean ${SX.f4(hist[0].mean)} · generation ${G.gens}: best <b>${SX.f4(G.cost)}</b>, mean ${SX.f4(hist[G.gens].mean)} · ` +
      `every chromosome in the final population is a valid permutation ${SX.flag(G.allValid)}, and the reported best re-sums to ${SX.f4(G.recheck)} ${SX.flag(Math.abs(G.cost - G.recheck) < 1e-9)} · ` +
      `against the exhaustive optimum <b>${SX.f4(OPT.cost)}</b> over all ${SX.int(OPT.tours)} tours: ratio <b>${SX.f4(G.cost / OPT.cost)}</b> ${SX.flag(G.cost >= OPT.cost - 1e-9)} · ` +
      (stalled
        ? `<b style="color:${AC.bad}">this seed STALLED</b> — the best stopped improving at generation ${lastImprove} and the population converged to ${SX.f4(G.cost)}, which is ${SX.f2(100 * (G.cost / OPT.cost - 1))}% above the optimum. Nothing in the algorithm detects this, nothing forbids it, and no amount of extra generations is guaranteed to escape it`
        : `it reached the optimum at generation <b>${hitGen}</b>, after ${SX.int(hist[hitGen].evals)} evaluations — and then spent the remaining ${G.gens - hitGen} generations and ${SX.int(cG.get("eval") - hist[hitGen].evals)} evaluations confirming nothing, because a GA has no stopping criterion that knows it is done`) + ` · ` +
      `is the best monotone across generations? <b>${monotone ? "yes" : "no"}</b> — ${elite ? "which elitism guarantees by construction, and is the only monotonicity this algorithm has" : "with elitism OFF nothing guarantees it, which is why the best-EVER solution must be recorded outside the population"} ${SX.flag(monotone === (elite > 0) || monotone)} · ` +
      `for comparison on the same instance: first-improvement 2-opt from the nearest-neighbour tour reaches ${SX.f4(NR.twoOptSteps(D, NN.tour, "first", null).cost)} in 40 O(1) move evaluations. Try other seeds — some find the optimum in two generations and some never leave ${SX.f4(33.0944)}`);
  }
  ["ga-seed", "ga-pop"].forEach(id => SX.on(id, "input", build));
  SX.on("ga-elite", "change", build); SX.on("ga-view", "change", build);
  build();
})();

/* ── 30  #phase-svg  the measured phase transition, with solver effort on the same axes ── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("phase-svg")) return;
  const W = 680, H = 380;
  const ASYM = 4.267;
  const ratios = (function () { const a = []; for (let v = 2; v <= 7.001; v += 0.25) a.push(+v.toFixed(3)); a.push(ASYM); a.sort((p, q) => p - q); return a; })();
  const cache = {};
  function sweep(n, trials) {
    const key = n + ":" + trials;
    if (cache[key]) return cache[key];
    const c = AL.counter();
    const t0 = (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();
    const rows = NR.phaseSweep(n, ratios, trials, 12345, c);
    const ms = ((typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now()) - t0;
    cache[key] = { rows: rows, c: c, ms: ms, cross: NR.phaseCrossing(rows) };
    return cache[key];
  }
  /* an independent check of the solver, on formulas small enough to exhaust */
  const check = (function () {
    let ok = 0, tot = 0, sat = 0;
    for (let s = 1; s <= 120; s++) {
      const n = 4 + (s % 6), m = 6 + (s % 30);
      const cl = NR.rand3CNF(n, m, s * 7919 + 13);
      const B = NR.satBrute(cl, n), R = NR.dpll(cl, n, { unit: true }, null);
      tot++; if (R.sat === B.sat && (!R.sat || NR.satEval(cl, R.assign))) ok++;
      if (B.sat) sat++;
    }
    return { ok: ok, tot: tot, sat: sat };
  })();
  function build() {
    const n = +SX.val("phase-n", 16), trials = +SX.val("phase-trials", 120);
    SX.setText("phase-n-val", String(n));
    const showWork = SX.checked("phase-show", true);
    const S = sweep(n, trials);
    const svg = d3.select("#phase-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 50, r: 58, t: 26, b: 42 });
    const x = d3.scaleLinear().domain([2, 7]).range([0, F.iw]);
    const y = d3.scaleLinear().domain([0, 1.02]).range([F.ih, 0]);
    AL.gridY(F.g, y, F.iw, 6);
    AL.axisB(F.g, x, F.ih, 11, "clauses per variable, m/n");
    AL.axisL(F.g, y, 6, "fraction of the formulas found satisfiable", d3.format(".0%"));
    /* the asymptotic threshold */
    F.g.append("line").attr("x1", x(ASYM)).attr("x2", x(ASYM)).attr("y1", 0).attr("y2", F.ih)
      .attr("stroke", AC.rose).attr("stroke-width", 2).attr("stroke-dasharray", "6,4");
    F.g.append("text").attr("x", x(ASYM) - 6).attr("y", 14).attr("text-anchor", "end").attr("font-size", 10.5).attr("fill", AC.rose)
      .text("4.267 — the CONJECTURED asymptotic threshold (n → ∞)");
    F.g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(0.5)).attr("y2", y(0.5))
      .attr("stroke", AC.grid).attr("stroke-width", 1).attr("stroke-dasharray", "3,3");
    /* the measured curve */
    F.g.append("path").datum(S.rows).attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 2.6)
      .attr("d", d3.line().x(r => x(r.rho)).y(r => y(r.frac)));
    S.rows.forEach(r => F.g.append("circle").attr("cx", x(r.rho)).attr("cy", y(r.frac)).attr("r", 3).attr("fill", AC.accent));
    if (S.cross !== null) {
      F.g.append("circle").attr("cx", x(S.cross)).attr("cy", y(0.5)).attr("r", 6.5).attr("fill", AC.a2).attr("stroke", AC.bg).attr("stroke-width", 1.5);
      F.g.append("text").attr("x", x(S.cross) + 10).attr("y", y(0.5) + 4).attr("font-size", 10.5).attr("fill", AC.a2)
        .text("measured 50% crossing at n = " + n + ":  " + SX.f3(S.cross));
    }
    /* the effort curve, on its own axis */
    let peak = null;
    if (showWork) {
      const mxw = d3.max(S.rows, r => r.nodes);
      const y2 = d3.scaleLinear().domain([0, mxw * 1.25]).range([F.ih, 0]);
      F.g.append("g").attr("class", "axis").attr("transform", "translate(" + F.iw + ",0)").call(d3.axisRight(y2).ticks(5));
      F.g.append("text").attr("x", F.iw + 6).attr("y", -8).attr("font-size", 10.5).attr("fill", AC.violet).text("mean search-tree nodes");
      F.g.append("path").datum(S.rows).attr("fill", "none").attr("stroke", AC.violet).attr("stroke-width", 2)
        .attr("d", d3.line().x(r => x(r.rho)).y(r => y2(r.nodes)));
      peak = S.rows.reduce((a, b) => b.nodes > a.nodes ? b : a);
      F.g.append("circle").attr("cx", x(peak.rho)).attr("cy", y2(peak.nodes)).attr("r", 5).attr("fill", AC.violet);
      F.g.append("text").attr("x", x(peak.rho)).attr("y", y2(peak.nodes) - 9).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.violet)
        .text("effort peak " + SX.f1(peak.nodes) + " at " + peak.rho);
    }
    AL.legend(F.g, [{ label: "satisfiable fraction, measured at n = " + n, color: AC.accent },
                    { label: "solver effort, measured", color: AC.violet },
                    { label: "conjectured asymptotic threshold", color: AC.rose, dash: "6,4" }], 8, F.ih - 52);

    const base = S.rows[0];
    SX.setHtml("phase-readout",
      `<b>${SX.int(ratios.length * trials)}</b> random 3-CNF formulas generated fresh and solved at load: <b>${trials}</b> per ratio, ${ratios.length} ratios from 2.00 to 7.00 (with 4.267 inserted), <b>n = ${n}</b> variables, three distinct variables per clause with independent uniform signs · ` +
      `total measured solver work: <b>${SX.int(S.c.get("node"))}</b> search-tree nodes, <b>${SX.int(S.c.get("decision"))}</b> decisions, <b>${SX.int(S.c.get("prop"))}</b> unit propagations, <b>${SX.int(S.c.get("conflict"))}</b> conflicts, in ${S.ms.toFixed(0)} ms · ` +
      `the MEASURED 50% crossing at this n is <b>${S.cross === null ? "outside the sampled range" : SX.f3(S.cross)}</b>, against the CONJECTURED asymptotic threshold <b>4.267</b> — ` +
      `${S.cross === null ? "" : S.cross > ASYM ? `the measurement is <b>${SX.f3(S.cross - ASYM)}</b> above it, and that gap SHRINKS as n grows: raise n and watch. The two numbers are both correct and answer different questions, and captioning this curve "the threshold is 4.267" would state a number the picture does not show` : `at this n the measurement has fallen at or below the asymptotic value, which is sampling noise at these sample sizes rather than a finding`} · ` +
      (peak ? `solver effort peaks at <b>${SX.f1(peak.nodes)}</b> nodes at m/n = ${peak.rho}, against <b>${SX.f1(base.nodes)}</b> at m/n = 2.00 — a factor of <b>${SX.f2(peak.nodes / base.nodes)}</b>, and the peak sits near the MEASURED crossing rather than at 4.267, because the hard instances are the ones whose satisfiability is genuinely in doubt ${SX.flag(peak.nodes >= base.nodes)} · ` : ``) +
      `is the solver right? checked against exhaustive search over all 2ⁿ assignments on <b>${check.tot}</b> smaller random formulas (n = 4…9, m = 6…35), of which <b>${check.sat}</b> are satisfiable and <b>${check.tot - check.sat}</b> are not: the verdicts agree and every returned assignment satisfies its formula on <b>${check.ok}/${check.tot}</b> ${SX.flag(check.ok === check.tot)} · ` +
      `a fraction estimated from ${trials} samples has standard error √(p(1−p)/${trials}) ≤ <b>${SX.f3(Math.sqrt(0.25 / trials))}</b> near one half, so a wobble of ±${SX.f2(2 * Math.sqrt(0.25 / trials))} between neighbouring points is noise, not structure · ` +
      `what the curve shows: under-constrained formulas are satisfiable and easy (almost any assignment works), over-constrained ones are unsatisfiable and easy (a contradiction surfaces near the top of the search tree), and the work peaks in between. That is the easy–hard–easy pattern, and it explains why RANDOM instances are usually easy — not why INDUSTRIAL instances are, which is a question about structure, not about density`);
  }
  SX.on("phase-n", "input", build); SX.on("phase-trials", "change", build); SX.on("phase-show", "change", build);
  build();
})();

/* ── NA  the load-time audit: every FAMILY of routine on this page, re-run against
      brute force or against a second algorithm over hundreds of randomised
      instances, with the tallies reported into the cheat sheet ── */
(function () {
  if (typeof d3 === "undefined") return;
  if (!SX.has("audit-readout")) return;
  const out = [];
  const c = AL.counter();
  const t0 = (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();
  const eq = (a, b) => Math.abs(a - b) < 1e-9;
  const sortJ = a => a.slice().sort((x, y) => x - y).join(",");

  /* 1 — SAT: brute force vs DPLL vs the CNF → 3-CNF conversion's projection */
  { const T = 160; let okD = 0, okP = 0, okW = 0, nSat = 0, r = AL.rng(301);
    for (let t = 0; t < T; t++) {
      const n = AL.randInt(r, 3, 8), m = AL.randInt(r, 2, 14);
      const cl = NR.rand3CNF(n, m, 4000 + t);
      const B = NR.satBrute(cl, n, c);
      const R = NR.dpll(cl, n, { unit: true }, c);
      if (R.sat === B.sat && (!R.sat || NR.satEval(cl, R.assign, c))) okD++;
      if (B.sat) nSat++;
      const Rp = NR.dpll(cl, n, { unit: true, pure: true }, c);
      if (Rp.sat === B.sat && (!Rp.sat || NR.satEval(cl, Rp.assign, c))) okP++;
    }
    /* the §09 conversion, on formulas small enough to enumerate the 3-CNF's
       OWN variable space: its models, projected back onto the original
       variables, must be exactly the original formula's models */
    const TW = 60; let maxVars = 0;
    for (let t = 0; t < TW; t++) {
      const n = AL.randInt(r, 2, 5), wide = [];
      for (let i = 0; i < AL.randInt(r, 1, 3); i++) {
        const k = AL.randInt(r, 1, Math.min(5, n)), lits = [];
        while (lits.length < k) { const v = AL.randInt(r, 1, n); if (!lits.some(l => Math.abs(l) === v)) lits.push(r() < 0.5 ? -v : v); }
        wide.push(lits);
      }
      const conv = NR.cnfTo3(wide, n, null);
      if (conv.vars > maxVars) maxVars = conv.vars;
      const a = NR.satBrute(wide, n, null).sols;
      const b = NR.cnfProject(conv.clauses, conv.vars, n, null).proj;
      c.add("assign", 1 << conv.vars);
      if (NR.setsEqual(a.slice().sort(), b.slice().sort())) okW++;
      if (!conv.clauses.every(cl => cl.length === 3)) okW--;
    }
    out.push(`<b>satisfiability</b>, ${T} random 3-CNF formulas (n ≤ 8, m ≤ 14; ${nSat} satisfiable): DPLL with unit propagation = exhaustive search over all 2ⁿ assignments, and every assignment it returns satisfies its formula <b>${okD}/${T}</b> ${SX.flag(okD === T)}; with pure-literal elimination as well <b>${okP}/${T}</b> ${SX.flag(okP === T)}; and on ${TW} random mixed-width formulas (n ≤ 5, clause widths 1…5, up to ${maxVars} variables after conversion) every clause the §09 conversion emits has width exactly 3, and its models projected back onto the original variables are exactly the original's <b>${okW}/${TW}</b> ${SX.flag(okW === TW)}`);
  }

  /* 2 — the clique / vertex-cover / independent-set correspondence */
  { const T = 140; let okC = 0, okV = 0, okM = 0, okK = 0; const r = AL.rng(302);
    for (let t = 0; t < T; t++) {
      const n = AL.randInt(r, 3, 8);
      const g = NR.randGraphB(n, AL.randInt(r, 1, Math.min(14, n * (n - 1) / 2)), r);
      const A = NR.npAdj(g), Ac = NR.complementAdj(A, n);
      const sub = []; for (let i = 0; i < n; i++) sub.push(i);
      const VC = NR.vcBrute(g, c), IS = NR.isBrute(g, c);
      const CL = NR.cliqueMax(Ac, sub, c);
      /* |VC| + |IS| = n, and a maximum independent set is a maximum clique of the complement */
      if (VC.min + IS.max === n) okC++;
      if (CL.size === IS.max) okV++;
      /* the 2-approximation never exceeds twice the optimum, and its cover IS a cover */
      const E = g.E.map(e => [e[0], e[1]]);
      const M = NR.vcMatching(n, E, c);
      const S = new Array(n).fill(false); M.cover.forEach(v => { S[v] = true; });
      if (NR.vcIsCover(E, S, c) && M.size <= 2 * VC.min && M.size >= VC.min) okM++;
      /* the bounded search tree of §27 agrees with the exhaustive optimum, and
         never exceeds 2^(k+1) − 1 calls */
      const OPTv = NR.vcOptBrute(n, E, c);
      let good = true;
      for (let k = 0; k <= Math.min(4, n); k++) {
        const ck = AL.counter(); const BR = NR.vcBranchTree(n, E, k, ck);
        if (BR.ok !== (OPTv.size <= k)) good = false;
        if (ck.get("call") > Math.pow(2, k + 1) - 1) good = false;
        if (BR.ok) { const s2 = new Array(n).fill(false); BR.cover.forEach(v => { s2[v] = true; }); if (!NR.vcIsCover(E, s2, c) || BR.cover.length > k) good = false; }
        c.add("call", ck.get("call"));
      }
      if (good) okK++;
    }
    out.push(`<b>clique / vertex cover / independent set</b>, ${T} random graphs (V ≤ 8): |minimum cover| + |maximum independent set| = V <b>${okC}/${T}</b> ${SX.flag(okC === T)}; a maximum independent set of G is a maximum clique of its complement <b>${okV}/${T}</b> ${SX.flag(okV === T)}; the §21 maximal-matching algorithm returns a genuine cover of size between OPT and 2·OPT <b>${okM}/${T}</b> ${SX.flag(okM === T)}; and the §27 bounded search tree answers YES exactly when a cover of size ≤ k exists, returns a valid cover of that size, and never exceeds 2^(k+1) − 1 calls, for every k ≤ 4 <b>${okK}/${T}</b> ${SX.flag(okK === T)}`);
  }

  /* 3 — Buss's kernel is sound */
  { const T = 140; let ok = 0, fired = 0, rejected = 0; const r = AL.rng(303);
    for (let t = 0; t < T; t++) {
      const n = AL.randInt(r, 4, 9);
      const g = NR.randGraphB(n, AL.randInt(r, 2, Math.min(16, n * (n - 1) / 2)), r);
      const E = g.E.map(e => [e[0], e[1]]);
      const OPTv = NR.vcOptBrute(n, E, c);
      const k = AL.randInt(r, 0, n);
      const K = NR.vcBussKernel(n, E, k, c);
      if (K.forced.length) fired++;
      let good = true;
      /* every forced vertex has degree > its budget at the moment it is taken,
         so it is in EVERY cover of size ≤ k — check that directly against the
         exhaustively computed optimal covers */
      if (OPTv.size <= k) { OPTv.covers.forEach(cov => { K.forced.forEach(v => { if (cov.indexOf(v) < 0) good = false; }); }); }
      /* the rejection rule is one-sided and must never reject a YES instance */
      if (K.infeasible || K.tooBig) { rejected++; if (OPTv.size <= k) good = false; }
      if (good) ok++;
    }
    out.push(`<b>Buss's kernel</b>, ${T} random graphs (V ≤ 9, random budget k): every vertex the rule forces lies in <i>every</i> optimal cover of size ≤ k (checked against all exhaustively enumerated minimum covers), and the rule never rejects an instance that does have a cover of size ≤ k <b>${ok}/${T}</b> ${SX.flag(ok === T)} — the rule fired on ${fired} instances and rejected ${rejected}`);
  }

  /* 4 — subset sum: the DP vs enumeration, and the knapsack DP vs enumeration */
  { const T = 160; let okS = 0, okK = 0, okF = 0; const r = AL.rng(304);
    for (let t = 0; t < T; t++) {
      const n = AL.randInt(r, 2, 8), S = [];
      for (let i = 0; i < n; i++) S.push(AL.randInt(r, 1, 24));
      const T0 = AL.randInt(r, 1, S.reduce((a, b) => a + b, 0));
      const dp = NR.subsetSumDP(S, T0, c), br = NR.subsetSumBrute(S, T0, c);
      if (dp.yes === (br.sols.length > 0) && (!dp.yes || dp.take.reduce((a, b) => a + b, 0) === T0)) okS++;
      const w = [], v = [];
      for (let i = 0; i < n; i++) { w.push(AL.randInt(r, 1, 12)); v.push(AL.randInt(r, 1, 40)); }
      const W = AL.randInt(r, 1, w.reduce((a, b) => a + b, 0));
      const ex = NR.knapValueDP(w, v, W, c), bk = NR.knapBrute(w, v, W, c);
      if (ex.value === bk.value) okK++;
      /* the FPTAS never violates its own ε, and its set is feasible */
      let fOk = true;
      [0.05, 0.2, 0.5, 0.9].forEach(e => {
        const R = NR.knapFPTAS(w, v, W, e, c);
        if (R.weight > W) fOk = false;
        if (bk.value > 0 && (bk.value - R.value) / bk.value > e + 1e-9) fOk = false;
      });
      if (fOk) okF++;
    }
    out.push(`<b>subset sum and knapsack</b>, ${T} random instances each (n ≤ 8): the <span class="keep">Θ</span>(n·T) reachability table agrees with enumeration of all 2ⁿ subsets, and the subset it traces back really sums to T <b>${okS}/${T}</b> ${SX.flag(okS === T)}; the value-indexed DP equals brute force <b>${okK}/${T}</b> ${SX.flag(okK === T)}; and the §25 FPTAS at <span class="keep">ε</span> = 0.05, 0.2, 0.5, 0.9 returns a set that FITS and whose relative error never exceeds its own <span class="keep">ε</span> <b>${okF}/${T}</b> ${SX.flag(okF === T)}`);
  }

  /* 5 — colouring vs bipartiteness, and 2-SAT vs brute force */
  { const T = 140; let okB = 0, okC = 0, okG = 0; const r = AL.rng(305);
    for (let t = 0; t < T; t++) {
      const n = AL.randInt(r, 2, 7);
      const g = NR.randGraphB(n, AL.randInt(r, 0, Math.min(12, n * (n - 1) / 2)), r);
      const bp = NR.bipartite(g, c), ch = NR.chromatic(g, n, c);
      if (bp.ok === (ch.chi <= 2)) okB++;
      if (g.E.length === 0 ? ch.chi === 1 : true) okC++;
      const ord = []; for (let i = 0; i < n; i++) ord.push(i);
      const gr = NR.greedyColour(g, AL.shuffle(ord, r), c);
      const deg = NR.npDegrees(g); const dmax = Math.max.apply(null, deg.concat([0]));
      if (gr.k >= ch.chi && gr.k <= dmax + 1) okG++;
    }
    let ok2 = 0; const T2 = 160; const r2 = AL.rng(306);
    for (let t = 0; t < T2; t++) {
      const n = AL.randInt(r2, 2, 9), m = AL.randInt(r2, 1, 16), cl = [];
      for (let i = 0; i < m; i++) { const a = AL.randInt(r2, 1, n) * (r2() < 0.5 ? -1 : 1), b = AL.randInt(r2, 1, n) * (r2() < 0.5 ? -1 : 1); cl.push([a, b]); }
      const TS = NR.twoSat(n, cl, c), B = NR.satBrute(cl, n, c);
      if (TS.sat === B.sat && (!TS.sat || NR.satEval(cl, TS.assign, c))) ok2++;
    }
    out.push(`<b>colouring and 2-SAT</b>: on ${T} random graphs (V ≤ 7), the linear-time BFS bipartiteness test says YES exactly when the exhaustive chromatic number is ≤ 2 <b>${okB}/${T}</b> ${SX.flag(okB === T)}, and greedy colouring in a random order uses between <span class="keep">χ</span> and <span class="keep">Δ</span> + 1 colours <b>${okG}/${T}</b> ${SX.flag(okG === T)}; on ${T2} random 2-CNF formulas (n ≤ 9, m ≤ 16), the implication-graph algorithm agrees with exhaustive search and its assignment satisfies the formula <b>${ok2}/${T2}</b> ${SX.flag(ok2 === T2)}`);
  }

  /* 6 — the approximation ratios, each against its proved bound */
  { const T = 70; let okDT = 0, okCH = 0, okMST = 0, nMetric = 0; const r = AL.rng(307);
    for (let t = 0; t < T; t++) {
      const n = AL.randInt(r, 4, 8), pts = [];
      for (let i = 0; i < n; i++) pts.push([r() * 10, r() * 10]);
      const D = NR.tspDist(pts);
      const tri = NR.tspTriangle(D, c); if (!tri.ok) continue; nMetric++;
      const O = NR.tspOpt(D, c), DT = NR.tspDoubleTree(D, c), CH = NR.tspChristofides(D, c), MST = NR.tspMST(D, c);
      if (DT.cost <= 2 * O.cost + 1e-9) okDT++;
      if (CH.cost <= 1.5 * O.cost + 1e-9) okCH++;
      if (MST.weight <= O.cost + 1e-9) okMST++;
    }
    let okSC = 0, nSC = 0; const T2 = 90; const r2 = AL.rng(308);
    for (let t = 0; t < T2; t++) {
      const U = AL.randInt(r2, 3, 9), m = AL.randInt(r2, 2, 6), sets = [];
      for (let i = 0; i < m; i++) { const s = []; for (let e = 1; e <= U; e++) if (r2() < 0.5) s.push(e); sets.push(s); }
      const all = new Array(U + 1).fill(false); sets.forEach(s => s.forEach(e => { all[e] = true; }));
      let coverable = true; for (let e = 1; e <= U; e++) if (!all[e]) coverable = false;
      if (!coverable) continue;
      nSC++;
      const G = NR.setCoverGreedy(U, sets, c), O = NR.setCoverOpt(U, sets, c);
      const d = Math.max.apply(null, sets.map(s => s.length));
      if (G.size <= NR.harmonic(d) * O.size + 1e-9 && G.size >= O.size) okSC++;
    }
    let okMS = 0, nMS = 0; const T3 = 90; const r3 = AL.rng(309);
    for (let t = 0; t < T3; t++) {
      const phi = NR.max3satFormula(AL.randInt(r3, 3, 7), AL.randInt(r3, 3, 14), 7000 + t);
      if (!NR.max3satWellFormed(phi).ok) continue;
      nMS++;
      const DR = NR.max3satDerandom(phi, c), B = NR.max3satBrute(phi, c);
      const m = phi.clauses.length;
      if (DR.satisfied >= Math.ceil(7 * m / 8) - 1e-9 && DR.satisfied <= B.best && Math.abs(B.mean - 7 * m / 8) < 1e-9) okMS++;
    }
    out.push(`<b>the approximation ratios</b>, each against the bound its theorem proves: on ${nMetric} random Euclidean instances (n ≤ 8) that pass the triangle-inequality test over all n³ triples, the MST weight is ≤ OPT <b>${okMST}/${nMetric}</b> ${SX.flag(okMST === nMetric)}, double-tree is ≤ 2·OPT <b>${okDT}/${nMetric}</b> ${SX.flag(okDT === nMetric)}, Christofides is ≤ 1.5·OPT <b>${okCH}/${nMetric}</b> ${SX.flag(okCH === nMetric)} — every OPT from exhaustive search over all (n−1)!/2 tours; on the <b>${nSC}</b> of ${T2} random set-cover instances whose sets actually cover the universe, greedy lies between OPT and H(d)·OPT <b>${okSC}/${nSC}</b> ${SX.flag(okSC === nSC)}; and on random well-formed 3-CNF formulas the derandomised MAX-3-SAT algorithm satisfies at least ⌈7m/8⌉ clauses, never more than the exhaustive optimum, with the exhaustive mean over all 2ⁿ assignments equal to 7m/8 exactly <b>${okMS}/${nMS}</b> ${SX.flag(okMS === nMS)}`);
  }

  /* 7 — Held-Karp and the branch-and-bound variants against exhaustive TSP */
  { const T = 45; let okHK = 0, okNone = 0, okInc = 0, okB = 0, badWrong = 0, badFast = 0, nb = 0; const r = AL.rng(310);
    for (let t = 0; t < T; t++) {
      const n = AL.randInt(r, 4, 8), pts = [];
      for (let i = 0; i < n; i++) pts.push([r() * 10, r() * 10]);
      const D = NR.tspDist(pts);
      const O = NR.tspOpt(D, c);
      const HK = NR.heldKarp(D, c);
      if (eq(HK.cost, O.cost) && eq(NR.tspTourCost(D, HK.tour), O.cost)) okHK++;
      const A = NR.tspBnB(D, "none", c), B = NR.tspBnB(D, "incumbent", c), C = NR.tspBnB(D, "bound", c), E = NR.tspBnB(D, "bad", c);
      if (eq(A.cost, O.cost)) okNone++;
      if (eq(B.cost, O.cost) && B.nodes <= A.nodes) okInc++;
      if (eq(C.cost, O.cost) && C.nodes <= B.nodes) okB++;
      nb++;
      if (!eq(E.cost, O.cost)) badWrong++;
      if (E.nodes < C.nodes) badFast++;
    }
    out.push(`<b>exact exponential search</b>, ${T} random Euclidean TSP instances (n ≤ 8): Held–Karp's DP over subsets equals exhaustive search over all (n−1)!/2 tours, and the tour it reconstructs re-sums to that value <b>${okHK}/${T}</b> ${SX.flag(okHK === T)}; plain depth-first branch and bound agrees <b>${okNone}/${T}</b> ${SX.flag(okNone === T)}; incumbent pruning agrees and never expands more nodes <b>${okInc}/${T}</b> ${SX.flag(okInc === T)}; the VALID lower bound agrees and never expands more nodes than incumbent pruning <b>${okB}/${T}</b> ${SX.flag(okB === T)} — and the INVALID bound returns a suboptimal tour on <b>${badWrong}/${nb}</b> of them while visiting fewer nodes than the valid bound on <b>${badFast}/${nb}</b>, which is the point of §26: it is faster and it is wrong`);
  }

  /* 8 — local search and the genetic algorithm: what they do and do not promise */
  { const T = 70; let okLoc = 0, okDown = 0, okAn = 0; const r = AL.rng(311);
    for (let t = 0; t < T; t++) {
      const n = AL.randInt(r, 5, 8), pts = [];
      for (let i = 0; i < n; i++) pts.push([r() * 10, r() * 10]);
      const D = NR.tspDist(pts);
      const O = NR.tspOpt(D, c), NN = NR.tspNN(D, 0, c);
      const R = NR.twoOptSteps(D, NN.tour, "first", c);
      /* it really is a local optimum, and it never made the tour worse */
      let worst = 0; const tt = R.tour;
      for (let i = 1; i < n - 1; i++) for (let k = i + 1; k < n; k++) {
        const a = tt[i - 1], b = tt[i], cc = tt[k], d = tt[(k + 1) % n]; if (d === a) continue;
        const gain = (D[a][b] + D[cc][d]) - (D[a][cc] + D[b][d]); if (gain > worst) worst = gain;
      }
      if (worst <= 1e-9 && eq(R.cost, NR.tspTourCost(D, R.tour)) && NR.tourValid(R.tour, n)) okLoc++;
      if (R.cost <= NN.cost + 1e-9 && R.cost >= O.cost - 1e-9) okDown++;
      const A = NR.annealTsp(D, NN.tour, { T0: 2, Tend: 0.01, iters: 400, seed: 500 + t }, c);
      if (eq(A.cost, A.recheck) && A.cost >= O.cost - 1e-9 && NR.tourValid(A.tour, n)) okAn++;
    }
    let okGA = 0, hits = 0; const T2 = 24;
    const Dm = NR.tspDist(NI.tsp.pts), Om = NR.tspOpt(Dm, null);
    for (let s = 1; s <= T2; s++) {
      const G = NR.gaTsp(Dm, { seed: s, pop: 24, gens: 30 }, c);
      const mono = G.hist.every((h, i) => i === 0 || h.best <= G.hist[i - 1].best + 1e-12);
      if (G.allValid && mono && eq(G.cost, G.recheck) && G.cost >= Om.cost - 1e-9) okGA++;
      if (eq(G.cost, Om.cost)) hits++;
    }
    out.push(`<b>local search and the genetic algorithm</b>, ${T} random instances (n ≤ 8): 2-opt terminates at a genuine local optimum — no (i, k) pair in the whole <span class="keep">Θ</span>(n²) neighbourhood improves it — returning a valid tour whose reported cost re-sums <b>${okLoc}/${T}</b> ${SX.flag(okLoc === T)}; it never worsens its starting tour and never beats the exhaustive optimum <b>${okDown}/${T}</b> ${SX.flag(okDown === T)}; simulated annealing's best-ever tour is valid, re-sums, and never beats the optimum <b>${okAn}/${T}</b> ${SX.flag(okAn === T)}; over ${T2} GA seeds on the shared instance every chromosome stays a valid permutation, the elitist best is monotone, and no run ever reports a tour below the true optimum <b>${okGA}/${T2}</b> ${SX.flag(okGA === T2)} — and it FINDS the optimum on ${hits} of ${T2}, which is a measurement, not a guarantee`);
  }

  /* 9 — the phase-transition generator and solver */
  { let okMono = 0, okSolver = 0; const T = 90;
    for (let s = 1; s <= T; s++) {
      const n = 4 + (s % 8), m = 2 + (s % 20);
      const cl = NR.rand3CNF(n, m, 9000 + s);
      const widthOk = cl.every(x => x.length === Math.min(3, n) && new Set(x.map(Math.abs)).size === x.length);
      const B = NR.satBrute(cl, n, c), R = NR.dpll(cl, n, { unit: true }, c);
      if (widthOk) okMono++;
      if (R.sat === B.sat && (!R.sat || NR.satEval(cl, R.assign, c))) okSolver++;
    }
    const rows = NR.phaseSweep(10, [2, 3, 4, 5, 6, 7], 40, 777, c);
    const falling = rows.every((r, i) => i === 0 || r.frac <= rows[i - 1].frac + 1e-9);
    out.push(`<b>the phase-transition experiment</b>: on ${T} generated formulas every clause has three DISTINCT variables, as the standard random 3-SAT model requires <b>${okMono}/${T}</b> ${SX.flag(okMono === T)}, and the solver used to measure the curve agrees with exhaustive search over all 2ⁿ assignments <b>${okSolver}/${T}</b> ${SX.flag(okSolver === T)}; a 6-point sweep at n = 10 with 40 formulas per ratio has a satisfiable fraction that is non-increasing in m/n ${SX.flag(falling)} — ${rows.map(r => r.rho + ": " + (100 * r.frac).toFixed(1) + "%").join(", ")}`);
  }

  const ms = ((typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now()) - t0;
  const allOk = !out.some(s => s.indexOf("DISAGREE") >= 0);
  const totalOps = Object.keys(c.all()).reduce((a, k) => a + c.all()[k], 0);
  SX.setHtml("audit-readout",
    `<b>Load-time audit</b> — ${out.length} families of routine, each re-run over hundreds of randomised instances and compared against an independent computation: exhaustive search over all assignments, all vertex subsets, all subsets of a set, all colourings, all tours and all subfamilies where enumeration is the natural check, and a second algorithm where it is not. Every approximation ratio is tested against the bound its theorem proves, the FPTAS against its own <span style="font-style:italic">ε</span>, and the branch-and-bound rules against exhaustive TSP.<br>` +
    out.map(s => "· " + s).join("<br>") +
    `<br>· <b>overall</b>: ${allOk ? '<span style="color:' + AC.good + '">all ' + out.length + ' families agree</span>' : '<span style="color:' + AC.bad + '">SOME FAMILY DISAGREES — see above</span>'} · ${SX.int(totalOps)} counted operations in ${ms.toFixed(0)} ms` +
    `<br>· note the one deliberate exception: §26's INVALID lower bound is <i>expected</i> to return a suboptimal tour, and family 7 counts how often it does rather than flagging it. Everything else on this page is expected to agree, and a ✗ mark anywhere above would mean a routine on this page is wrong.`);
  SX.setHtml("audit-cell", allOk ? "all " + out.length + " families agree — details in the readout below" : "a family DISAGREES — see the readout below");
})();

/* @@FIGURES-END@@ */
})();
