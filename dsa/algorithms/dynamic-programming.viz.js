/* dynamic-programming.viz.js — figures for dsa/algorithms/dynamic-programming.html
   (part 5 of the Algorithm Design & Analysis series).

   Loaded after ../../data.js, ../../notes.js and ../dsa-viz.js, so the globals
   AC (palette) and AL (rng/frame/axisB/axisL/row/binTree/counter/stepper/…) are
   available.

   House rule obeyed throughout: every count this page DISPLAYS — recursive
   calls, memo hits, table cells filled, comparisons, relaxations, reconstruction
   steps — is produced by running the real routine under AL.counter() and reading
   the counter back. Every DP ANSWER is cross-checked in the same figure against
   an independent computation — brute force over all cuttings / subsets /
   subsequences / parenthesizations / orders / paths for small n, or a second
   algorithm (the O(n²) LIS against the O(n log n) one, Hirschberg against the
   full table, the DP against a closed form) — and printed with an agree /
   DISAGREE flag. A load-time audit (DA, at the bottom) repeats those checks over
   hundreds of random instances and reports the counts in the cheat sheet.

   Layout of this file:
     DR — the instrumented routines, pure and DOM-free (also loadable in node).
     SX — small DOM helpers shared by the figures.
     figures — one block per <svg>, in page order:
       01 #rod-tree-svg   rod cutting's call tree, naive vs memoized, grown call by call
       02 #rod-table-svg  rod cutting's r[] and s[] filled cell by cell, the cut reconstructed
       03 #subgraph-svg   the subproblem graph and its topological order
       04 #mt-svg         memoization vs tabulation measured across n; the sparse-state case
       05 #lin-svg        1-D DP strips: Fibonacci, stairs, house robber, maximum subarray
       06 #coin-svg       coin change: combinations vs compositions vs minimum coins, loop order
       07 #knap-svg       0/1 knapsack table with the reconstruction path lit
       08 #knap1d-svg     the 1-D knapsack: loop direction switchable, wrong direction's answer shown
       09 #wis-svg        weighted interval scheduling with p(j) drawn
       10 #lcs-svg        LCS table stepped, traceback lit
       11 #ed-svg         edit distance table with weighted operations and the alignment
       12 #hirsch-svg     Hirschberg's split
       13 #lis-svg        LIS: the Θ(n²) DP and the O(n log n) tails method, stepped
       14 #mc-svg         matrix chain m/s tables and the parenthesization tree
       15 #obst-svg       optimal BST: e/root tables and the tree, vs all trees
       16 #ivl-svg        interval DP: longest palindromic subsequence by length
       17 #tree-svg       tree DP: max independent set, (take, skip) per node
       18 #dag-svg        DP on a DAG: shortest / longest / count in topological order
       19 #tsp-svg        Held–Karp on 4–5 cities
       20 #digit-svg      digit DP state lattice
       21 #mq-svg         monotone-queue sliding-window minimum, measured
       22 #vi-svg         value iteration on a gridworld converging
       23 #vit-svg        Viterbi trellis
     DA — the load-time audit. */

/* ═══════════════════════════════════════════════════════════════════════════
   DR — the instrumented routines
   ═══════════════════════════════════════════════════════════════════════════ */
const DR = (function () {
  const nop = { add: function () {}, get: function () { return 0; } };
  const INF = Infinity;

  /* ── rod cutting ────────────────────────────────────────────────────── */
  /* prices p[1..n] (p[0] unused). Naive recursion: c.add("call") per invocation.
     rec(event) receives {type:"call"|"ret", n, parent, id} so a figure can grow the tree. */
  function rodNaive(p, n, c, rec) {
    c = c || nop; let id = 0;
    function cut(n, parent) {
      const me = id++; c.add("call");
      if (rec) rec({ type: "call", n, parent, id: me, hit: false });
      if (n === 0) { if (rec) rec({ type: "ret", id: me, v: 0 }); return 0; }
      let q = -INF;
      for (let i = 1; i <= n; i++) q = Math.max(q, p[i] + cut(n - i, me));
      if (rec) rec({ type: "ret", id: me, v: q });
      return q;
    }
    return cut(n, -1);
  }
  function rodMemo(p, n, c, rec) {
    c = c || nop; let id = 0; const memo = new Map();
    function cut(n, parent) {
      const me = id++; c.add("call");
      if (memo.has(n)) { c.add("hit"); if (rec) rec({ type: "call", n, parent, id: me, hit: true }); if (rec) rec({ type: "ret", id: me, v: memo.get(n) }); return memo.get(n); }
      c.add("compute");
      if (rec) rec({ type: "call", n, parent, id: me, hit: false });
      let q = 0;
      if (n > 0) { q = -INF; for (let i = 1; i <= n; i++) q = Math.max(q, p[i] + cut(n - i, me)); }
      memo.set(n, q);
      if (rec) rec({ type: "ret", id: me, v: q });
      return q;
    }
    return cut(n, -1);
  }
  /* Bottom-up with the first-piece table s[]. c.add("cell") per inner iteration
     (one candidate first piece examined); rec after each r[j] is fixed. */
  function rodTable(p, n, c, rec) {
    c = c || nop; const r = [0], s = [0];
    for (let j = 1; j <= n; j++) {
      let q = -INF, best = 0;
      for (let i = 1; i <= j; i++) { c.add("cell"); if (p[i] + r[j - i] > q) { q = p[i] + r[j - i]; best = i; } }
      r[j] = q; s[j] = best;
      if (rec) rec({ j, r: r.slice(), s: s.slice() });
    }
    return { r, s };
  }
  function rodCuts(s, n, c) { c = c || nop; const out = []; while (n > 0) { c.add("recon"); out.push(s[n]); n -= s[n]; } return out; }
  /* brute force over all 2^(n-1) cuttings */
  function rodBrute(p, n) {
    if (n === 0) return { best: 0, count: 1 };
    let best = -INF, count = 0;
    for (let mask = 0; mask < (1 << (n - 1)); mask++) {
      count++; let last = 0, v = 0;
      for (let i = 1; i < n; i++) if (mask >> (i - 1) & 1) { v += p[i - last]; last = i; }
      v += p[n - last]; if (v > best) best = v;
    }
    return { best, count };
  }

  /* ── Fibonacci ──────────────────────────────────────────────────────── */
  function fibNaive(n, c) { c = c || nop; function f(n) { c.add("call"); return n < 2 ? n : f(n - 1) + f(n - 2); } return f(n); }
  function fibMemo(n, c) { c = c || nop; const m = new Map(); function f(n) { c.add("call"); if (m.has(n)) { c.add("hit"); return m.get(n); } const v = n < 2 ? n : f(n - 1) + f(n - 2); m.set(n, v); return v; } return f(n); }
  function fibTable(n, c) { c = c || nop; const t = [0, 1]; for (let i = 2; i <= n; i++) { c.add("cell"); t[i] = t[i - 1] + t[i - 2]; } return n === 0 ? 0 : t[n]; }
  function fibIter(n) { let a = 0, b = 1; for (let i = 0; i < n; i++) { const t = a + b; a = b; b = t; } return a; }

  return { INF, rodNaive, rodMemo, rodTable, rodCuts, rodBrute, fibNaive, fibMemo, fibTable, fibIter };
})();

/* the shared instances, so prose and figures agree by construction */
const DI = {
  rodPrices: [0, 1, 5, 8, 9, 10, 17, 17, 20, 24, 30]
};

/* ── chunk A routines ── */
/* ── chunk A routines: rod cutting (split form), subproblem graphs, memo-vs-table
      measurement, the sparse recurrence, 1-D strips, coin change ─────────── */
Object.assign(DR, (function () {
  const nop = { add: function () {}, get: function () { return 0; } };
  const INF = Infinity;

  /* ── rod cutting, the "one cut, two halves" formulation ─────────────────
     r[j] = max( p[j],  max over i = 1..j−1 of  r[i] + r[j−i] ).
     With sym = true only i ≤ ⌊j/2⌋ is examined (the split is symmetric).
     c.add("cell") per candidate examined; s2[j] = 0 means "no cut". */
  function rodTableSplit(p, n, c, rec, sym) {
    c = c || nop; const r = [0], s = [0];
    for (let j = 1; j <= n; j++) {
      c.add("cell"); let q = p[j], best = 0;
      const top = sym ? Math.floor(j / 2) : j - 1;
      for (let i = 1; i <= top; i++) { c.add("cell"); if (r[i] + r[j - i] > q) { q = r[i] + r[j - i]; best = i; } }
      r[j] = q; s[j] = best;
      if (rec) rec({ j, r: r.slice(), s: s.slice() });
    }
    return { r, s };
  }
  /* reconstruction for the split form: s[j] = 0 → the piece j is sold whole */
  function rodCutsSplit(s, n, c) {
    c = c || nop; const out = [];
    (function go(j) { c.add("recon"); if (s[j] === 0) { out.push(j); return; } go(s[j]); go(j - s[j]); })(n);
    return out;
  }
  /* the number of cuttings (of the 2^(n−1)) that attain the optimum */
  function rodOptCount(p, n) {
    if (n === 0) return { best: 0, opt: 1, count: 1 };
    let best = -INF, opt = 0, count = 0;
    for (let mask = 0; mask < (1 << (n - 1)); mask++) {
      count++; let last = 0, v = 0;
      for (let i = 1; i < n; i++) if (mask >> (i - 1) & 1) { v += p[i - last]; last = i; }
      v += p[n - last];
      if (v > best) { best = v; opt = 1; } else if (v === best) opt++;
    }
    return { best, opt, count };
  }

  /* ── subproblem graphs ──────────────────────────────────────────────────
     G = { V: [{id, label}], E: [[from, to]], goal } — an edge x → y means
     "the recurrence for x consults y". Built by enumerating the recurrence. */
  function sgRod(n) {
    const V = [], E = [];
    for (let j = 0; j <= n; j++) V.push({ id: j, label: String(j) });
    for (let j = 1; j <= n; j++) for (let i = 1; i <= j; i++) E.push([j, j - i]);
    return { V, E, goal: n, kind: "rod", n };
  }
  function sgFib(n) {
    const V = [], E = [];
    for (let j = 0; j <= n; j++) V.push({ id: j, label: String(j) });
    for (let j = 2; j <= n; j++) { E.push([j, j - 1]); E.push([j, j - 2]); }
    return { V, E, goal: n, kind: "fib", n };
  }
  /* LCS on x, y: state (i, j) = prefixes x[0..i), y[0..j); id = i·(n+1) + j */
  function sgLcs(x, y) {
    const m = x.length, n = y.length, V = [], E = [];
    const id = (i, j) => i * (n + 1) + j;
    for (let i = 0; i <= m; i++) for (let j = 0; j <= n; j++) V.push({ id: id(i, j), label: `${i},${j}`, i, j });
    for (let i = 1; i <= m; i++) for (let j = 1; j <= n; j++) {
      if (x[i - 1] === y[j - 1]) E.push([id(i, j), id(i - 1, j - 1)]);
      else { E.push([id(i, j), id(i - 1, j)]); E.push([id(i, j), id(i, j - 1)]); }
    }
    return { V, E, goal: id(m, n), kind: "lcs", m, n, x, y };
  }
  /* memoized evaluation = DFS from the goal; returns the postorder (the order
     in which answers are FIXED) and the set of states reached. c.add("visit")
     per state computed, c.add("edge") per edge followed (a hit or a descent). */
  function sgDfs(G, c) {
    c = c || nop; const adj = new Map(); G.V.forEach(v => adj.set(v.id, []));
    G.E.forEach(([u, v]) => adj.get(u).push(v));
    const seen = new Set(), post = [], events = [];
    (function dfs(u) {
      seen.add(u); c.add("visit"); events.push({ type: "enter", id: u });
      for (const v of adj.get(u)) { c.add("edge"); if (!seen.has(v)) dfs(v); else events.push({ type: "hit", id: v, from: u }); }
      post.push(u); events.push({ type: "fix", id: u });
    })(G.goal);
    return { post, reached: seen, events };
  }
  /* tabulation order = a topological order of the REVERSED graph (every state
     after all it consults), by Kahn's algorithm on out-degree; c.add("pop") per state. */
  function sgTopo(G, c) {
    c = c || nop; const outdeg = new Map(), users = new Map();
    G.V.forEach(v => { outdeg.set(v.id, 0); users.set(v.id, []); });
    G.E.forEach(([u, v]) => { outdeg.set(u, outdeg.get(u) + 1); users.get(v).push(u); });
    const q = G.V.filter(v => outdeg.get(v.id) === 0).map(v => v.id), order = [];
    while (q.length) { const v = q.shift(); c.add("pop"); order.push(v);
      for (const u of users.get(v)) { outdeg.set(u, outdeg.get(u) - 1); if (outdeg.get(u) === 0) q.push(u); } }
    return order;
  }

  /* a memoized LCS used ONLY to count calls against the DFS of the subproblem graph
     (the LCS table itself is built later on the page); c.add("call") / c.add("hit") */
  function sgLcsMemoCalls(x, y, c) {
    c = c || nop; const m = new Map();
    function L(i, j) { c.add("call"); const k = i + "," + j; if (m.has(k)) { c.add("hit"); return m.get(k); }
      const v = (i === 0 || j === 0) ? 0 : (x[i - 1] === y[j - 1] ? 1 + L(i - 1, j - 1) : Math.max(L(i - 1, j), L(i, j - 1)));
      m.set(k, v); return v; }
    return { value: L(x.length, y.length), states: m.size };
  }

  /* ── the sparse recurrence: f(n) = max(n, f(⌊n/2⌋) + f(⌊n/3⌋) + f(⌊n/4⌋)), f(0) = 0.
     A top-down memo touches only the states reachable from n; a table needs n + 1. */
  function sparseMemo(n, c) {
    c = c || nop; const m = new Map();
    function f(n) { c.add("call"); if (m.has(n)) { c.add("hit"); return m.get(n); } c.add("state");
      const v = n === 0 ? 0 : Math.max(n, f(Math.floor(n / 2)) + f(Math.floor(n / 3)) + f(Math.floor(n / 4)));
      m.set(n, v); return v; }
    return { value: f(n), states: m.size };
  }
  function sparseTable(n, c) {
    c = c || nop; const t = [0];
    for (let i = 1; i <= n; i++) { c.add("cell"); t[i] = Math.max(i, t[Math.floor(i / 2)] + t[Math.floor(i / 3)] + t[Math.floor(i / 4)]); }
    return t[n];
  }

  /* ── one-dimensional strips ─────────────────────────────────────────────
     each returns the full dp array; rec(i, dp) after dp[i] is fixed; c.add("cell") per cell. */
  function stairs(n, c, rec) {           /* ways to climb n steps by 1 or 2: dp[0] = 1, dp[1] = 1 */
    c = c || nop; const d = [1, 1]; if (rec) { rec(0, d.slice(0, 1)); if (n >= 1) rec(1, d.slice(0, 2)); }
    for (let i = 2; i <= n; i++) { c.add("cell"); d[i] = d[i - 1] + d[i - 2]; if (rec) rec(i, d.slice()); }
    return d.slice(0, n + 1);
  }
  function stairsBrute(n) {              /* enumerate every 1/2 sequence */
    let count = 0; (function go(k) { if (k === n) { count++; return; } go(k + 1); if (k + 1 < n) go(k + 2); })(0); return count;
  }
  function robber(v, c, rec) {           /* best[i] = max(best[i−1], best[i−2] + v[i]); best[−1] = 0 */
    c = c || nop; const n = v.length, d = [];
    for (let i = 0; i < n; i++) { c.add("cell"); const skip = i >= 1 ? d[i - 1] : 0, take = (i >= 2 ? d[i - 2] : 0) + v[i]; d[i] = Math.max(skip, take); if (rec) rec(i, d.slice()); }
    return d;
  }
  function robberBrute(v) {              /* all 2^n subsets, keep those with no two adjacent */
    const n = v.length; let best = 0, legal = 0;
    for (let m = 0; m < (1 << n); m++) { if (m & (m << 1)) continue; legal++; let s = 0; for (let i = 0; i < n; i++) if (m >> i & 1) s += v[i]; if (s > best) best = s; }
    return { best, legal, total: 1 << n };
  }
  /* houses on a RING: house 0 and house n−1 are adjacent. State = (i, first house taken?);
     equivalently two linear passes, one forbidding house 0 and one forbidding house n−1. */
  function robberRing(v, c) {
    c = c || nop; const n = v.length; if (n === 1) return v[0];
    const a = robber(v.slice(1), c), b = robber(v.slice(0, n - 1), c);
    return Math.max(a[a.length - 1], b[b.length - 1]);
  }
  function robberRingBrute(v) {
    const n = v.length; let best = 0;
    for (let m = 0; m < (1 << n); m++) { if (m & (m << 1)) continue; if (n > 1 && (m & 1) && (m >> (n - 1) & 1)) continue; let s = 0; for (let i = 0; i < n; i++) if (m >> i & 1) s += v[i]; if (s > best) best = s; }
    return best;
  }
  /* the WRONG one-number state for the linear robber: "best over the prefix, greedily extended" —
     dp[i] = dp[i−1] + v[i] if house i−1 was not the last one taken, else dp[i−1]. It forgets which. */
  function robberNaive(v) {
    let best = 0, lastTaken = -2;
    for (let i = 0; i < v.length; i++) if (lastTaken !== i - 1) { best += v[i]; lastTaken = i; }
    return best;
  }
  function kadane(a, c, rec) {           /* end[i] = max(a[i], end[i−1] + a[i]); best = max end */
    c = c || nop; const n = a.length, end = [], bestArr = []; let best = -INF;
    for (let i = 0; i < n; i++) { c.add("cell"); end[i] = i === 0 ? a[0] : Math.max(a[i], end[i - 1] + a[i]); best = Math.max(best, end[i]); bestArr[i] = best; if (rec) rec(i, end.slice(), bestArr.slice()); }
    return { end, best, bestArr };
  }
  function kadaneBrute(a) {              /* all n(n+1)/2 non-empty subarrays */
    let best = -INF, count = 0; for (let i = 0; i < a.length; i++) { let s = 0; for (let j = i; j < a.length; j++) { s += a[j]; count++; if (s > best) best = s; } }
    return { best, count };
  }
  function minCostStairs(cost, c) {      /* pay cost[i] to step from i; start at 0 or 1; reach the top (index n) */
    c = c || nop; const n = cost.length, d = [cost[0], cost[1]];
    for (let i = 2; i < n; i++) { c.add("cell"); d[i] = cost[i] + Math.min(d[i - 1], d[i - 2]); }
    return { d, best: Math.min(d[n - 1], d[n - 2]) };
  }
  function minCostStairsBrute(cost) {
    const n = cost.length; let best = INF;
    (function go(i, paid) { if (i >= n) { if (paid < best) best = paid; return; } go(i + 1, paid + cost[i]); go(i + 2, paid + cost[i]); })(0, 0);
    (function go(i, paid) { if (i >= n) { if (paid < best) best = paid; return; } go(i + 1, paid + cost[i]); go(i + 2, paid + cost[i]); })(1, 0);
    return best;
  }
  function decodeWays(s, c) {            /* d[i] = ways to decode s[0..i): one digit 1–9, two digits 10–26 */
    c = c || nop; const n = s.length, d = [1]; d[1] = s[0] === "0" ? 0 : 1;
    for (let i = 2; i <= n; i++) { c.add("cell"); d[i] = 0; if (s[i - 1] !== "0") d[i] += d[i - 1]; const two = +s.slice(i - 2, i); if (s[i - 2] !== "0" && two >= 10 && two <= 26) d[i] += d[i - 2]; }
    return d;
  }
  function decodeBrute(s) {
    let count = 0; (function go(i) { if (i === s.length) { count++; return; } if (s[i] !== "0") go(i + 1); if (i + 1 < s.length && s[i] !== "0" && +s.slice(i, i + 2) <= 26) go(i + 2); })(0); return count;
  }
  function fibClosed(n) {                /* Binet, rounded; exact for n ≤ 70 in doubles */
    const phi = (1 + Math.sqrt(5)) / 2; return Math.round(Math.pow(phi, n) / Math.sqrt(5));
  }
  function fibMatrix(n, c) {             /* [[1,1],[1,0]]^n by repeated squaring; c.add("mult") per 2×2 product */
    c = c || nop; function mul(A, B) { c.add("mult"); return [[A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]], [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]]; }
    let R = [[1, 0], [0, 1]], M = [[1, 1], [1, 0]], k = n;
    while (k > 0) { if (k & 1) R = mul(R, M); M = mul(M, M); k >>= 1; }
    return R[0][1];
  }

  /* ── coin change: three problems, three loop nests ──────────────────────
     coins = sorted distinct positive denominations; rec(frame) after each inner update
     frame = { outer, inner, dp: copy, note }. c.add("cell") per inner-loop update. */
  function coinComb(coins, A, c, rec) {  /* combinations (multisets): coins OUTER, amount inner */
    c = c || nop; const d = new Array(A + 1).fill(0); d[0] = 1;
    for (const k of coins) for (let a = k; a <= A; a++) { c.add("cell"); d[a] += d[a - k]; if (rec) rec({ outer: k, inner: a, dp: d.slice() }); }
    return d;
  }
  function coinComp(coins, A, c, rec) {  /* compositions (ordered): amount OUTER, coins inner */
    c = c || nop; const d = new Array(A + 1).fill(0); d[0] = 1;
    for (let a = 1; a <= A; a++) for (const k of coins) { if (k > a) continue; c.add("cell"); d[a] += d[a - k]; if (rec) rec({ outer: a, inner: k, dp: d.slice() }); }
    return d;
  }
  function coinMin(coins, A, c, rec) {   /* fewest coins: d[a] = 1 + min d[a − k]; ∞ if unreachable; either loop order works */
    c = c || nop; const d = new Array(A + 1).fill(INF), from = new Array(A + 1).fill(-1); d[0] = 0;
    for (let a = 1; a <= A; a++) for (const k of coins) { if (k > a) continue; c.add("cell"); if (d[a - k] + 1 < d[a]) { d[a] = d[a - k] + 1; from[a] = k; } if (rec) rec({ outer: a, inner: k, dp: d.slice() }); }
    return { d, from };
  }
  function coinMinRecon(from, A) { const out = []; let a = A; while (a > 0 && from[a] > 0) { out.push(from[a]); a -= from[a]; } return a === 0 ? out : null; }
  function coinGreedy(coins, A) { const out = []; let a = A; for (let i = coins.length - 1; i >= 0; i--) while (a >= coins[i]) { out.push(coins[i]); a -= coins[i]; } return a === 0 ? out : null; }
  /* brute force: every multiset (non-increasing sequence) of coins summing to A */
  function coinCombBrute(coins, A) {
    let count = 0, fewest = INF;
    (function go(rem, maxIdx, used) { if (rem === 0) { count++; if (used < fewest) fewest = used; return; } for (let i = maxIdx; i >= 0; i--) if (coins[i] <= rem) go(rem - coins[i], i, used + 1); })(A, coins.length - 1, 0);
    return { count, fewest };
  }
  /* brute force: every ordered sequence of coins summing to A */
  function coinCompBrute(coins, A) {
    let count = 0; (function go(rem) { if (rem === 0) { count++; return; } for (const k of coins) if (k <= rem) go(rem - k); })(A); return count;
  }

  return { rodTableSplit, rodCutsSplit, rodOptCount, sgRod, sgFib, sgLcs, sgDfs, sgTopo, sgLcsMemoCalls, sparseMemo, sparseTable,
           stairs, stairsBrute, robber, robberBrute, robberRing, robberRingBrute, robberNaive, kadane, kadaneBrute, minCostStairs, minCostStairsBrute, decodeWays, decodeBrute, fibClosed, fibMatrix,
           coinComb, coinComp, coinMin, coinMinRecon, coinGreedy, coinCombBrute, coinCompBrute };
})());

Object.assign(DI, {
  robberVals: [2, 7, 9, 3, 1, 8, 4],
  kadaneVals: [-2, 1, -3, 4, -1, 2, 1, -5, 4],
  stairCosts: [1, 100, 1, 1, 1, 100, 1, 1, 100, 1],
  lcsSmall: ["ABCB", "BDCAB"],
  coinSets: { "1,2,5": [1, 2, 5], "1,3,4": [1, 3, 4], "1,5,10,25": [1, 5, 10, 25], "2,5": [2, 5] }
});

/* ── chunk B routines ── */
/* ── chunk B routines: 0/1 knapsack and its variants, weighted interval
   scheduling, LCS, edit distance, Hirschberg, LIS ─────────────────────── */
Object.assign(DR, (function () {
  const nop = { add: function () {}, get: function () { return 0; } };
  const INF = Infinity;

  /* ── 0/1 knapsack ───────────────────────────────────────────────────── */
  /* items: [{id, w, v}], capacity W (integer). dp[i][w] = best value using items 1..i
     at capacity w. c.add("cell") per cell of rows 1..n (n·(W+1) cells); c.add("recon")
     per walk-back step. rec({i, row}) after each row is complete. */
  function knap01(items, W, c, rec) {
    c = c || nop; const n = items.length;
    const dp = [Array(W + 1).fill(0)]; const keep = [Array(W + 1).fill(false)];
    for (let i = 1; i <= n; i++) {
      const it = items[i - 1]; dp[i] = Array(W + 1).fill(0); keep[i] = Array(W + 1).fill(false);
      for (let w = 0; w <= W; w++) {
        c.add("cell");
        let best = dp[i - 1][w], k = false;
        if (it.w <= w && dp[i - 1][w - it.w] + it.v > best) { best = dp[i - 1][w - it.w] + it.v; k = true; }
        dp[i][w] = best; keep[i][w] = k;
      }
      if (rec) rec({ i, row: dp[i].slice() });
    }
    /* reconstruction: walk from (n, W) upward */
    const take = []; const path = []; let w = W;
    for (let i = n; i >= 1; i--) { c.add("recon"); path.push([i, w]); if (keep[i][w]) { take.push(items[i - 1].id); w -= items[i - 1].w; } }
    path.push([0, w]);
    return { dp, keep, val: dp[n][W], take: take.reverse(), path, cells: n * (W + 1) };
  }
  /* brute force over all 2^n subsets */
  function knapBrute(items, W) {
    const n = items.length; let best = 0, arg = [], count = 0;
    for (let mask = 0; mask < (1 << n); mask++) {
      count++; let w = 0, v = 0; const ids = [];
      for (let i = 0; i < n; i++) if (mask & (1 << i)) { w += items[i].w; v += items[i].v; ids.push(items[i].id); }
      if (w <= W && v > best) { best = v; arg = ids; }
    }
    return { val: best, take: arg, count };
  }
  /* the rolling 1-D array. dir = "down" (capacity loop W..w_i: 0/1) or "up" (w_i..W: each
     item may be reused — the unbounded answer). c.add("cell") per (i, w) update examined.
     rec({i, w, arr, changed}) per inner step. */
  function knap1d(items, W, dir, c, rec) {
    c = c || nop; const f = Array(W + 1).fill(0);
    for (let i = 0; i < items.length; i++) {
      const it = items[i];
      if (dir === "down") { for (let w = W; w >= it.w; w--) { c.add("cell"); const old = f[w], cand = f[w - it.w] + it.v; const ch = cand > old; if (ch) f[w] = cand; if (rec) rec({ i, w, arr: f.slice(), changed: ch, old }); } }
      else { for (let w = it.w; w <= W; w++) { c.add("cell"); const old = f[w], cand = f[w - it.w] + it.v; const ch = cand > old; if (ch) f[w] = cand; if (rec) rec({ i, w, arr: f.slice(), changed: ch, old }); } }
    }
    return { f, val: f[W] };
  }
  /* unbounded knapsack over capacity: K(w) = max over items with w_i ≤ w of K(w − w_i) + v_i.
     c.add("cell") per (w, item) pair examined. Returns the multiset chosen by walking back. */
  function knapUnbounded(items, W, c) {
    c = c || nop; const K = Array(W + 1).fill(0), pick = Array(W + 1).fill(-1);
    for (let w = 1; w <= W; w++) for (let i = 0; i < items.length; i++) { if (items[i].w > w) continue; c.add("cell"); const cand = K[w - items[i].w] + items[i].v; if (cand > K[w]) { K[w] = cand; pick[w] = i; } }
    /* walk back from the best capacity ≤ W (K is monotone here since K(w) ≥ K(w−1) is not enforced; take argmax) */
    let wb = W; for (let w = 0; w <= W; w++) if (K[w] > K[wb]) wb = w;
    const take = []; let w = wb; while (w > 0 && pick[w] >= 0) { take.push(items[pick[w]].id); w -= items[pick[w]].w; }
    return { K, val: K[wb], take: take.reverse() };
  }
  /* brute force for unbounded: enumerate every multiset of items with total weight ≤ W */
  function knapUnboundedBrute(items, W) {
    let best = 0, arg = [], count = 0;
    function go(i, cap, v, taken) {
      if (i === items.length) { count++; if (v > best) { best = v; arg = taken.slice(); } return; }
      for (let k = 0; k * items[i].w <= cap; k++) { for (let t = 0; t < k; t++) taken.push(items[i].id); go(i + 1, cap - k * items[i].w, v + k * items[i].v, taken); for (let t = 0; t < k; t++) taken.pop(); }
    }
    go(0, W, 0, []);
    return { val: best, take: arg, count };
  }
  /* bounded knapsack by binary splitting: an item available k times becomes pieces of
     multiplicity 1, 2, 4, …, 2^(t−1) and a remainder k − (2^t − 1), where 2^t − 1 ≤ k.
     Number of pieces = ⌊log₂ k⌋ + 1 (= ⌈log₂(k + 1)⌉). Every count 0..k is a sum of a sub-multiset. */
  function knapBinarySplit(items) {
    const out = [];
    items.forEach(it => { let k = it.k, m = 1; while (k > 0) { const take = Math.min(m, k); out.push({ id: it.id + "×" + take, w: it.w * take, v: it.v * take, of: it.id, mult: take }); k -= take; m *= 2; } });
    return out;
  }
  function knapBoundedBrute(items, W) {
    let best = 0, arg = [], count = 0;
    function go(i, cap, v, taken) {
      if (i === items.length) { count++; if (v > best) { best = v; arg = taken.slice(); } return; }
      for (let k = 0; k <= items[i].k && k * items[i].w <= cap; k++) { for (let t = 0; t < k; t++) taken.push(items[i].id); go(i + 1, cap - k * items[i].w, v + k * items[i].v, taken); for (let t = 0; t < k; t++) taken.pop(); }
    }
    go(0, W, 0, []);
    return { val: best, take: arg, count };
  }
  /* subset sum as a boolean 1-D knapsack; c.add("cell") per update; returns reach[] */
  function subsetSum(a, T, c) {
    c = c || nop; const r = Array(T + 1).fill(false); r[0] = true;
    for (const x of a) for (let s = T; s >= x; s--) { c.add("cell"); if (r[s - x]) r[s] = true; }
    return r;
  }
  function subsetSumBrute(a, T) { let ok = false, count = 0; for (let m = 0; m < (1 << a.length); m++) { count++; let s = 0; for (let i = 0; i < a.length; i++) if (m & (1 << i)) s += a[i]; if (s === T) ok = true; } return { ok, count }; }

  /* ── weighted interval scheduling ──────────────────────────────────── */
  /* ivs: [{id, s, f, v}]. Sorts by finish (c.add("sortcmp") per comparator call),
     computes p(j) by binary search over finish times (c.add("probe") per probe),
     fills OPT with c.add("cell") per j, reconstructs with c.add("recon").
     rec({j, opt}) after each OPT(j). Indices are 1-based in the returned arrays. */
  function wisSolve(ivs, c, rec) {
    c = c || nop;
    const order = ivs.slice().sort((a, b) => { c.add("sortcmp"); return a.f - b.f || a.s - b.s || String(a.id).localeCompare(String(b.id)); });
    const n = order.length; const p = [0];
    for (let j = 1; j <= n; j++) {
      /* largest i < j with f_i ≤ s_j: finish times are sorted, so binary search for the
         right boundary of { i : f_i ≤ s_j } among indices 1..j−1 */
      let lo = 1, hi = j; const sj = order[j - 1].s;      // half-open [lo, hi): answer is lo − 1
      while (lo < hi) { const mid = (lo + hi) >> 1; c.add("probe"); if (order[mid - 1].f <= sj) lo = mid + 1; else hi = mid; }
      p[j] = lo - 1;
    }
    const opt = [0], choice = [false];
    for (let j = 1; j <= n; j++) { c.add("cell"); const withJ = order[j - 1].v + opt[p[j]], without = opt[j - 1]; choice[j] = withJ > without; opt[j] = Math.max(withJ, without); if (rec) rec({ j, opt: opt.slice() }); }
    const take = []; let j = n; while (j > 0) { c.add("recon"); if (choice[j]) { take.push(order[j - 1].id); j = p[j]; } else j--; }
    return { order, p, opt, choice, val: opt[n], take: take.reverse() };
  }
  function wisBrute(ivs) {
    const n = ivs.length; let best = 0, arg = [], count = 0;
    for (let m = 0; m < (1 << n); m++) {
      count++; const sel = []; for (let i = 0; i < n; i++) if (m & (1 << i)) sel.push(ivs[i]);
      let ok = true; for (let a = 0; a < sel.length && ok; a++) for (let b = a + 1; b < sel.length; b++) if (!(sel[a].f <= sel[b].s || sel[b].f <= sel[a].s)) { ok = false; break; }
      if (!ok) continue; const v = sel.reduce((s, x) => s + x.v, 0); if (v > best) { best = v; arg = sel.map(x => x.id); }
    }
    return { val: best, take: arg, count };
  }
  /* the unweighted earliest-finish greedy applied to weighted intervals (sum of the weights it picks) */
  function wisGreedyEF(ivs) { const o = ivs.slice().sort((a, b) => a.f - b.f || a.s - b.s); let last = -INF, v = 0; const take = []; for (const a of o) if (a.s >= last) { take.push(a.id); v += a.v; last = a.f; } return { val: v, take }; }

  /* ── longest common subsequence ────────────────────────────────────── */
  /* L[i][j] = LCS length of X[0..i) and Y[0..j). c.add("cell") per (i ≥ 1, j ≥ 1) cell.
     dirs[i][j] ∈ {"↖","↑","←"} records the case that won (ties → ↑). rec({i, j, L}) per cell. */
  function lcsTable(X, Y, c, rec) {
    c = c || nop; const m = X.length, n = Y.length;
    const L = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0)); const dirs = Array.from({ length: m + 1 }, () => Array(n + 1).fill(""));
    for (let i = 1; i <= m; i++) for (let j = 1; j <= n; j++) {
      c.add("cell");
      if (X[i - 1] === Y[j - 1]) { L[i][j] = L[i - 1][j - 1] + 1; dirs[i][j] = "↖"; }
      else if (L[i - 1][j] >= L[i][j - 1]) { L[i][j] = L[i - 1][j]; dirs[i][j] = "↑"; }
      else { L[i][j] = L[i][j - 1]; dirs[i][j] = "←"; }
      if (rec) rec({ i, j, v: L[i][j], d: dirs[i][j] });
    }
    return { L, dirs, len: L[m][n], cells: m * n };
  }
  /* traceback from (m, n) using the L values; pref = "up" (default) or "left" decides ties, so two
     different LCSs can be read off the same table. c.add("trace") per step; returns the string and the cells visited */
  function lcsTraceback(X, Y, t, c, pref) {
    c = c || nop; let i = X.length, j = Y.length; const out = [], path = []; const L = t.L;
    while (i > 0 && j > 0) {
      c.add("trace"); path.push([i, j]);
      if (X[i - 1] === Y[j - 1]) { out.push(X[i - 1]); i--; j--; }
      else if (pref === "left" ? L[i][j - 1] >= L[i - 1][j] : L[i - 1][j] >= L[i][j - 1]) { if (pref === "left") j--; else i--; }
      else { if (pref === "left") i--; else j--; }
    }
    return { s: out.reverse().join(""), path };
  }
  /* the naive recursion, no table — c.add("call") per call */
  function lcsNaive(X, Y, c) { c = c || nop; function f(i, j) { c.add("call"); if (i === 0 || j === 0) return 0; if (X[i - 1] === Y[j - 1]) return 1 + f(i - 1, j - 1); return Math.max(f(i - 1, j), f(i, j - 1)); } return f(X.length, Y.length); }
  /* every distinct LCS string, by exploring all optimal predecessors (small instances only) */
  function lcsAll(X, Y, t) {
    const L = t.L; const memo = new Map();
    function go(i, j) {
      if (i === 0 || j === 0) return new Set([""]);
      const key = i + "," + j; if (memo.has(key)) return memo.get(key);
      let res = new Set();
      if (X[i - 1] === Y[j - 1]) { go(i - 1, j - 1).forEach(s => res.add(s + X[i - 1])); }
      else { if (L[i - 1][j] === L[i][j]) go(i - 1, j).forEach(s => res.add(s)); if (L[i][j - 1] === L[i][j]) go(i, j - 1).forEach(s => res.add(s)); }
      memo.set(key, res); return res;
    }
    return [...go(X.length, Y.length)].sort();
  }
  function isSubsequence(s, t) { let k = 0; for (let i = 0; i < t.length && k < s.length; i++) if (t[i] === s[k]) k++; return k === s.length; }
  /* brute force: every subsequence of the shorter string, tested as a subsequence of the other */
  function lcsBrute(X, Y) {
    const S = X.length <= Y.length ? X : Y, T = S === X ? Y : X; let best = "", count = 0;
    for (let m = 0; m < (1 << S.length); m++) { count++; let s = ""; for (let i = 0; i < S.length; i++) if (m & (1 << i)) s += S[i]; if (s.length > best.length && isSubsequence(s, T)) best = s; }
    return { len: best.length, s: best, count };
  }
  /* the length only, in O(min(m, n)) space: two rows over the shorter string */
  function lcsLenSpace(X, Y, c) {
    c = c || nop; if (X.length < Y.length) { const t = X; X = Y; Y = t; }
    const n = Y.length; let prev = Array(n + 1).fill(0), cur = Array(n + 1).fill(0);
    for (let i = 1; i <= X.length; i++) { for (let j = 1; j <= n; j++) { c.add("cell"); cur[j] = X[i - 1] === Y[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]); } const t = prev; prev = cur; cur = t; }
    return { len: prev[n], words: 2 * (n + 1) };
  }

  /* ── edit distance ─────────────────────────────────────────────────── */
  /* D[i][j] = cost to turn a[0..i) into b[0..j). costs {ins, del, sub}; a match costs 0.
     c.add("cell") per (i ≥ 1, j ≥ 1) cell. op[i][j] records the winner (ties: match/sub, then del, then ins). */
  function edTable(a, b, costs, c, rec) {
    c = c || nop; const K = Object.assign({ ins: 1, del: 1, sub: 1 }, costs || {});
    const m = a.length, n = b.length;
    const D = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0)); const op = Array.from({ length: m + 1 }, () => Array(n + 1).fill(""));
    for (let i = 1; i <= m; i++) { D[i][0] = D[i - 1][0] + K.del; op[i][0] = "del"; }
    for (let j = 1; j <= n; j++) { D[0][j] = D[0][j - 1] + K.ins; op[0][j] = "ins"; }
    for (let i = 1; i <= m; i++) for (let j = 1; j <= n; j++) {
      c.add("cell");
      const same = a[i - 1] === b[j - 1];
      const diag = D[i - 1][j - 1] + (same ? 0 : K.sub), del = D[i - 1][j] + K.del, ins = D[i][j - 1] + K.ins;
      let best = diag, o = same ? "match" : "sub";
      if (del < best) { best = del; o = "del"; }
      if (ins < best) { best = ins; o = "ins"; }
      D[i][j] = best; op[i][j] = o;
      if (rec) rec({ i, j, v: best, o });
    }
    return { D, op, dist: D[m][n], cells: m * n, costs: K };
  }
  /* read the alignment off the op table: a list of {op, x, y} from (0,0) to (m,n) */
  function edAlign(a, b, t) {
    let i = a.length, j = b.length; const steps = [];
    while (i > 0 || j > 0) { const o = t.op[i][j]; if (o === "match" || o === "sub") { steps.push({ op: o, x: a[i - 1], y: b[j - 1], i, j }); i--; j--; } else if (o === "del") { steps.push({ op: "del", x: a[i - 1], y: "-", i, j }); i--; } else { steps.push({ op: "ins", x: "-", y: b[j - 1], i, j }); j--; } }
    return steps.reverse();
  }
  function edAlignCost(steps, costs) { const K = Object.assign({ ins: 1, del: 1, sub: 1 }, costs || {}); return steps.reduce((s, st) => s + (st.op === "match" ? 0 : K[st.op]), 0); }
  /* the naive recursion, no table — an independent check; c.add("call") per call */
  function edNaive(a, b, costs, c) {
    c = c || nop; const K = Object.assign({ ins: 1, del: 1, sub: 1 }, costs || {});
    function d(i, j) { c.add("call"); if (i === 0) return j * K.ins; if (j === 0) return i * K.del; const s = a[i - 1] === b[j - 1] ? 0 : K.sub; return Math.min(d(i - 1, j - 1) + s, d(i - 1, j) + K.del, d(i, j - 1) + K.ins); }
    return d(a.length, b.length);
  }
  /* unit-cost BFS over strings: the fewest single-character edits from a to b (small strings only) */
  function edBFS(a, b, limit) {
    const alpha = [...new Set((a + b).split(""))]; let frontier = new Set([a]); const seen = new Set([a]); let d = 0, expanded = 0;
    if (a === b) return { dist: 0, expanded };
    while (d < limit) {
      const next = new Set();
      for (const s of frontier) {
        expanded++;
        const cands = [];
        for (let i = 0; i < s.length; i++) cands.push(s.slice(0, i) + s.slice(i + 1));
        for (let i = 0; i <= s.length; i++) for (const ch of alpha) cands.push(s.slice(0, i) + ch + s.slice(i));
        for (let i = 0; i < s.length; i++) for (const ch of alpha) if (ch !== s[i]) cands.push(s.slice(0, i) + ch + s.slice(i + 1));
        for (const t of cands) { if (t === b) return { dist: d + 1, expanded }; if (Math.abs(t.length - b.length) > limit - d - 1) continue; if (!seen.has(t)) { seen.add(t); next.add(t); } }
      }
      frontier = next; d++;
    }
    return { dist: INF, expanded };
  }

  /* Dijkstra on the edit graph — an independent algorithm on the DAG view (simple O(V²) version).
     c.add("relax") per edge relaxation. */
  function edDijkstra(a, b, costs, c) {
    c = c || nop; const K = Object.assign({ ins: 1, del: 1, sub: 1 }, costs || {});
    const m = a.length, n = b.length, V = (m + 1) * (n + 1); const dist = Array(V).fill(INF), done = Array(V).fill(false);
    const id = (i, j) => i * (n + 1) + j; dist[0] = 0;
    for (let round = 0; round < V; round++) {
      let u = -1; for (let v = 0; v < V; v++) if (!done[v] && (u < 0 || dist[v] < dist[u])) u = v;
      if (u < 0 || dist[u] === INF) break; done[u] = true;
      const i = Math.floor(u / (n + 1)), j = u % (n + 1);
      const relax = (v, w) => { c.add("relax"); if (dist[u] + w < dist[v]) dist[v] = dist[u] + w; };
      if (i < m) relax(id(i + 1, j), K.del);
      if (j < n) relax(id(i, j + 1), K.ins);
      if (i < m && j < n) relax(id(i + 1, j + 1), a[i] === b[j] ? 0 : K.sub);
    }
    return dist[id(m, n)];
  }

  /* ── Hirschberg ────────────────────────────────────────────────────── */
  /* the last row of the edit-distance table for a against b, in O(n) space; c.add("cell") per interior cell */
  function edLastRow(a, b, K, c) {
    c = c || nop; const n = b.length; let prev = Array(n + 1).fill(0), cur = Array(n + 1).fill(0);
    for (let j = 1; j <= n; j++) prev[j] = prev[j - 1] + K.ins;
    for (let i = 1; i <= a.length; i++) {
      cur[0] = prev[0] + K.del;
      for (let j = 1; j <= n; j++) { c.add("cell"); const s = a[i - 1] === b[j - 1] ? 0 : K.sub; cur[j] = Math.min(prev[j - 1] + s, prev[j] + K.del, cur[j - 1] + K.ins); }
      const t = prev; prev = cur; cur = t;
    }
    return prev;
  }
  function rev(s) { return s.split("").reverse().join(""); }
  /* Hirschberg: the full alignment in O(m + n) space. Returns {steps, dist}; c.add("cell")
     counts every interior cell touched (forward and backward half-scores, and the base cases);
     rec({a, b, i0, j0, mid, split, fwd, bwd}) at every split. */
  function hirschberg(a, b, costs, c, rec) {
    c = c || nop; const K = Object.assign({ ins: 1, del: 1, sub: 1 }, costs || {});
    function align(a, b, i0, j0) {
      if (a.length === 0) return b.split("").map(ch => ({ op: "ins", x: "-", y: ch }));
      if (b.length === 0) return a.split("").map(ch => ({ op: "del", x: ch, y: "-" }));
      if (a.length === 1) { const t = edTable(a, b, K, c); return edAlign(a, b, t); }
      const mid = a.length >> 1;
      const fwd = edLastRow(a.slice(0, mid), b, K, c);
      const bwd = edLastRow(rev(a.slice(mid)), rev(b), K, c);
      let split = 0, best = INF;
      for (let j = 0; j <= b.length; j++) { const s = fwd[j] + bwd[b.length - j]; if (s < best) { best = s; split = j; } }
      if (rec) rec({ a, b, i0, j0, mid, split, fwd: fwd.slice(), bwd: bwd.slice(), cost: best });
      return align(a.slice(0, mid), b.slice(0, split), i0, j0).concat(align(a.slice(mid), b.slice(split), i0 + mid, j0 + split));
    }
    const steps = align(a, b, 0, 0);
    return { steps, dist: edAlignCost(steps, K) };
  }

  /* ── longest increasing subsequence ────────────────────────────────── */
  /* Θ(n²): dp[i] = 1 + max{dp[j] : j < i, a[j] < a[i]}; c.add("cmp") per (j, i) pair; rec after each i */
  function lisQuad(a, c, rec) {
    c = c || nop; const n = a.length, dp = Array(n).fill(1), prev = Array(n).fill(-1);
    for (let i = 0; i < n; i++) { for (let j = 0; j < i; j++) { c.add("cmp"); if (a[j] < a[i] && dp[j] + 1 > dp[i]) { dp[i] = dp[j] + 1; prev[i] = j; } } if (rec) rec({ i, dp: dp.slice(), prev: prev.slice() }); }
    let end = 0; for (let i = 1; i < n; i++) if (dp[i] > dp[end]) end = i;
    const idx = []; for (let k = end; k >= 0; k = prev[k]) idx.push(k);
    idx.reverse();
    return { dp, prev, len: n ? dp[end] : 0, idx, seq: idx.map(i => a[i]) };
  }
  /* O(n log n): tails[k] = the smallest tail of any increasing subsequence of length k + 1
     seen so far (tails is sorted); tailIdx[k] = the index holding it; pred[i] links each
     element to its predecessor at placement time. c.add("probe") per binary-search probe; rec after each i. */
  function lisTails(a, c, rec) {
    c = c || nop; const n = a.length, tails = [], tailIdx = [], pred = Array(n).fill(-1);
    for (let i = 0; i < n; i++) {
      let lo = 0, hi = tails.length;                                // first k with tails[k] ≥ a[i]  (strictly increasing LIS)
      while (lo < hi) { const mid = (lo + hi) >> 1; c.add("probe"); if (tails[mid] < a[i]) lo = mid + 1; else hi = mid; }
      tails[lo] = a[i]; tailIdx[lo] = i; pred[i] = lo > 0 ? tailIdx[lo - 1] : -1;
      if (rec) rec({ i, k: lo, tails: tails.slice(), tailIdx: tailIdx.slice(), pred: pred.slice() });
    }
    const idx = []; for (let k = tails.length ? tailIdx[tails.length - 1] : -1; k >= 0; k = pred[k]) idx.push(k);
    idx.reverse();
    return { tails, tailIdx, pred, len: tails.length, idx, seq: idx.map(i => a[i]) };
  }
  function lisBrute(a) {
    const n = a.length; let best = 0, arg = [], count = 0;
    for (let m = 0; m < (1 << n); m++) { count++; let ok = true, last = -INF, len = 0; const s = []; for (let i = 0; i < n && ok; i++) if (m & (1 << i)) { if (a[i] <= last) ok = false; else { last = a[i]; len++; s.push(a[i]); } } if (ok && len > best) { best = len; arg = s; } }
    return { len: best, seq: arg, count };
  }
  /* LIS as LCS(a, sorted(a)) — valid when the values are distinct */
  function lisViaLcs(a, c) { const s = a.slice().sort((x, y) => x - y); const t = lcsTable(a, s, c); return { len: t.len, seq: lcsTraceback(a, s, t).s }; }

  return {
    knap01, knapBrute, knap1d, knapUnbounded, knapUnboundedBrute, knapBinarySplit, knapBoundedBrute, subsetSum, subsetSumBrute,
    wisSolve, wisBrute, wisGreedyEF,
    lcsTable, lcsTraceback, lcsNaive, lcsAll, lcsBrute, lcsLenSpace, isSubsequence,
    edTable, edAlign, edAlignCost, edNaive, edBFS, edDijkstra, edLastRow, hirschberg,
    lisQuad, lisTails, lisBrute, lisViaLcs
  };
})());
Object.assign(DI, {
  /* the four items of the Greedy page's §10–§11 (W = 50): density greedy 160, optimum 220 {B, C} */
  knapGreedyItems: [{ id: "A", w: 10, v: 60 }, { id: "B", w: 20, v: 100 }, { id: "C", w: 30, v: 120 }, { id: "D", w: 25, v: 90 }],
  knapGreedyW: 50,
  /* the small instance worked in §10–§11: W = 10; 0/1 optimum 46 {1, 3}, unbounded optimum 48 {1, 4, 4} */
  knapSmall: [{ id: "1", w: 6, v: 30 }, { id: "2", w: 3, v: 14 }, { id: "3", w: 4, v: 16 }, { id: "4", w: 2, v: 9 }],
  knapSmallW: 10,
  /* weighted intervals worked in §12 (id, s, f, v) — unsorted on purpose */
  wisIntervals: [[1, 1, 4, 3], [2, 3, 5, 4], [3, 0, 6, 10], [4, 4, 7, 5], [5, 3, 9, 8], [6, 5, 9, 2], [7, 6, 10, 4], [8, 8, 11, 3]].map(p => ({ id: p[0], s: p[1], f: p[2], v: p[3] })),
  lcsX: "ABCBDAB", lcsY: "BDCABA",
  edA: "kitten", edB: "sitting",
  lisA: [10, 9, 2, 5, 3, 7, 101, 18]
});

/* ── chunk C routines ── */
Object.assign(DR, (function () {
  const nop = { add: function () {}, get: function () { return 0; } };
  const INF = Infinity;

  /* ── §17 matrix chain ───────────────────────────────────────────────── */
  /* dims p[0..n]; matrix i is p[i-1] × p[i]. m/s are 1-indexed [1..n][1..n].
     c.add("iter") per inner (i,j,k) iteration; rec after each m[i][j] is fixed. */
  function mcTable(p, c, rec) {
    c = c || nop; const n = p.length - 1;
    const m = [], s = [];
    for (let i = 0; i <= n; i++) { m.push(new Array(n + 1).fill(0)); s.push(new Array(n + 1).fill(0)); }
    for (let l = 2; l <= n; l++) {
      for (let i = 1; i + l - 1 <= n; i++) {
        const j = i + l - 1; m[i][j] = INF;
        for (let k = i; k < j; k++) {
          c.add("iter");
          const q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j];
          if (q < m[i][j]) { m[i][j] = q; s[i][j] = k; }
        }
        if (rec) rec({ l, i, j, m: m.map(r => r.slice()), s: s.map(r => r.slice()) });
      }
    }
    return { m, s, n };
  }
  function mcParens(s, i, j) { return i === j ? "A" + i : "(" + mcParens(s, i, s[i][j]) + mcParens(s, s[i][j] + 1, j) + ")"; }
  /* parenthesization tree as nested {i,j,k,left,right} */
  function mcTree(s, i, j) { if (i === j) return { i, j, leaf: true }; const k = s[i][j]; return { i, j, k, left: mcTree(s, i, k), right: mcTree(s, k + 1, j) }; }
  /* brute force: enumerate every full parenthesization of A_i..A_j, return {best, count} */
  function mcBrute(p, i, j) {
    if (i === undefined) { i = 1; j = p.length - 1; }
    if (i === j) return { best: 0, count: 1 };
    let best = INF, count = 0;
    for (let k = i; k < j; k++) {
      const L = mcBrute(p, i, k), R = mcBrute(p, k + 1, j);
      count += L.count * R.count;
      const v = L.best + R.best + p[i - 1] * p[k] * p[j];
      if (v < best) best = v;
    }
    return { best, count };
  }
  function catalan(k) { let c = 1; for (let i = 0; i < k; i++) c = c * 2 * (2 * i + 1) / (i + 2); return Math.round(c); }
  /* cost of a given parenthesization tree (used to score brute-force enumerations if needed) */
  function mcCostOfTree(t, p) { return t.leaf ? 0 : mcCostOfTree(t.left, p) + mcCostOfTree(t.right, p) + p[t.i - 1] * p[t.k] * p[t.j]; }

  /* ── §18 optimal BST ────────────────────────────────────────────────── */
  /* keys k1..kn with p[1..n], dummies d0..dn with q[0..n]. e[i][j] for 1 ≤ i ≤ n+1, i-1 ≤ j ≤ n.
     c.add("iter") per candidate root examined; rec after each e[i][j] fixed.
     opt.knuth: restrict r to [root[i][j-1], root[i+1][j]]. */
  function obstTable(p, q, c, rec, opt) {
    c = c || nop; opt = opt || {}; const n = p.length - 1;
    const e = [], w = [], root = [];
    for (let i = 0; i <= n + 2; i++) { e.push(new Array(n + 2).fill(0)); w.push(new Array(n + 2).fill(0)); root.push(new Array(n + 2).fill(0)); }
    for (let i = 1; i <= n + 1; i++) { e[i][i - 1] = q[i - 1]; w[i][i - 1] = q[i - 1]; }
    for (let l = 1; l <= n; l++) {
      for (let i = 1; i + l - 1 <= n; i++) {
        const j = i + l - 1; e[i][j] = INF; w[i][j] = w[i][j - 1] + p[j] + q[j];
        const lo = (opt.knuth && l > 1) ? root[i][j - 1] : i;
        const hi = (opt.knuth && l > 1) ? root[i + 1][j] : j;
        for (let r = lo; r <= hi; r++) {
          c.add("iter");
          const t = e[i][r - 1] + e[r + 1][j] + w[i][j];
          if (t < e[i][j] - 1e-12) { e[i][j] = t; root[i][j] = r; }
        }
        if (rec) rec({ l, i, j, e: e.map(r => r.slice()), root: root.map(r => r.slice()), w: w.map(r => r.slice()) });
      }
    }
    return { e, w, root, n };
  }
  /* the tree from root[][] as nested {key, left, right}; dummies as {dummy:i} */
  function obstTree(root, i, j) { if (i > j) return { dummy: j }; const r = root[i][j]; return { key: r, left: obstTree(root, i, r - 1), right: obstTree(root, r + 1, j) }; }
  /* expected cost of a given tree: ∑ p_i (depth_i + 1) + ∑ q_i depth_dummy */
  function obstCostOf(t, p, q) { let s = 0; (function walk(t, d) { if (t.dummy !== undefined) { s += q[t.dummy] * (d + 1); return; } s += p[t.key] * (d + 1); walk(t.left, d + 1); walk(t.right, d + 1); })(t, 0); return s; }
  /* brute force over every BST shape on keys i..j: returns {best, count} of expected costs. w(i,j) added at each root. */
  function obstBrute(p, q, i, j) {
    const n = p.length - 1; if (i === undefined) { i = 1; j = n; }
    const wsum = (a, b) => { let s = q[a - 1]; for (let t = a; t <= b; t++) s += p[t] + q[t]; return s; };
    function rec(i, j) {
      if (i > j) return { best: q[j], count: 1 };
      let best = INF, count = 0;
      for (let r = i; r <= j; r++) {
        const L = rec(i, r - 1), R = rec(r + 1, j);
        count += L.count * R.count;
        const v = L.best + R.best + wsum(i, j);
        if (v < best) best = v;
      }
      return { best, count };
    }
    return rec(i, j);
  }
  /* Knuth's root-monotonicity check on a finished table: root[i][j-1] ≤ root[i][j] ≤ root[i+1][j] for all i<j */
  function obstRootMonotone(root, n) { let ok = true, checked = 0; for (let i = 1; i <= n; i++) for (let j = i + 1; j <= n; j++) { checked++; if (!(root[i][j - 1] <= root[i][j] && root[i][j] <= root[i + 1][j])) ok = false; } return { ok, checked }; }

  /* ── §19 interval DP ────────────────────────────────────────────────── */
  /* longest palindromic subsequence: L[i][j] over s[i..j], 0-indexed. c.add("cell") per cell fixed. */
  function lpsTable(s, c, rec) {
    c = c || nop; const n = s.length; const L = []; for (let i = 0; i < n; i++) L.push(new Array(n).fill(0));
    for (let i = 0; i < n; i++) { L[i][i] = 1; c.add("cell"); }
    if (rec) rec({ l: 1, L: L.map(r => r.slice()) });
    for (let l = 2; l <= n; l++) for (let i = 0; i + l - 1 < n; i++) {
      const j = i + l - 1; c.add("cell");
      if (s[i] === s[j]) L[i][j] = (l === 2 ? 0 : L[i + 1][j - 1]) + 2; else L[i][j] = Math.max(L[i + 1][j], L[i][j - 1]);
      if (rec) rec({ l, i, j, L: L.map(r => r.slice()) });
    }
    return L;
  }
  function lpsRecon(s, L, i, j) { if (i > j) return ""; if (i === j) return s[i]; if (s[i] === s[j]) return s[i] + lpsRecon(s, L, i + 1, j - 1) + s[j]; return L[i + 1][j] >= L[i][j - 1] ? lpsRecon(s, L, i + 1, j) : lpsRecon(s, L, i, j - 1); }
  function isPal(t) { for (let a = 0, b = t.length - 1; a < b; a++, b--) if (t[a] !== t[b]) return false; return true; }
  /* brute force over all 2^n subsequences */
  function lpsBrute(s) { const n = s.length; let best = 0, count = 0; for (let mask = 0; mask < (1 << n); mask++) { count++; let t = ""; for (let i = 0; i < n; i++) if (mask >> i & 1) t += s[i]; if (isPal(t) && t.length > best) best = t.length; } return { best, count }; }
  /* longest palindromic SUBSTRING, expand-around-centre, returns {len, start} */
  function lpSubstring(s) { let best = 0, start = 0; const n = s.length; for (let cIdx = 0; cIdx < 2 * n - 1; cIdx++) { let a = Math.floor(cIdx / 2), b = a + (cIdx % 2); while (a >= 0 && b < n && s[a] === s[b]) { if (b - a + 1 > best) { best = b - a + 1; start = a; } a--; b++; } } return { len: best, start }; }
  /* burst balloons: values v[0..n-1]; padded with 1s; dp[i][j] over open interval (i, j) of the padded array. */
  function burstTable(v, c, rec) {
    c = c || nop; const a = [1].concat(v, [1]); const N = a.length; const dp = [], last = [];
    for (let i = 0; i < N; i++) { dp.push(new Array(N).fill(0)); last.push(new Array(N).fill(-1)); }
    for (let len = 2; len < N; len++) for (let i = 0; i + len < N; i++) {
      const j = i + len; c.add("cell");
      for (let k = i + 1; k < j; k++) { c.add("iter"); const val = dp[i][k] + dp[k][j] + a[i] * a[k] * a[j]; if (val > dp[i][j]) { dp[i][j] = val; last[i][j] = k; } }
      if (rec) rec({ len, i, j, dp: dp.map(r => r.slice()) });
    }
    return { dp, last, best: dp[0][N - 1], a };
  }
  /* burst order from the last[][] table: everything inside (i,k), then inside (k,j), then k */
  function burstOrder(last, i, j) { const k = last[i][j]; if (k < 0) return []; return burstOrder(last, i, k).concat(burstOrder(last, k, j), [k]); }
  /* brute force over all n! burst orders */
  function burstBrute(v) { let best = 0, count = 0; function go(arr, acc) { if (arr.length === 0) { count++; if (acc > best) best = acc; return; } for (let k = 0; k < arr.length; k++) { const l = k === 0 ? 1 : arr[k - 1], r = k === arr.length - 1 ? 1 : arr[k + 1]; go(arr.slice(0, k).concat(arr.slice(k + 1)), acc + l * arr[k] * r); } } go(v.slice(), 0); return { best, count }; }
  /* the "first burst" recurrence — WRONG on purpose, to show why: burst k first, then solve the two sides independently (they are not independent) */
  function burstFirstWrong(v) { const memo = new Map(); function f(arr) { if (!arr.length) return 0; const key = arr.join(","); if (memo.has(key)) return memo.get(key); let best = 0; for (let k = 0; k < arr.length; k++) { const l = k === 0 ? 1 : arr[k - 1], r = k === arr.length - 1 ? 1 : arr[k + 1]; const val = l * arr[k] * r + f(arr.slice(0, k)) + f(arr.slice(k + 1)); if (val > best) best = val; } memo.set(key, best); return best; } return f(v); }

  /* ── §20 tree DP ────────────────────────────────────────────────────── */
  /* tree: {n, parent[], w[], children[][]} rooted at 0. Maximum-weight independent set by (take, skip). c.add("visit") per node. */
  function treeMwis(T, c, rec) {
    c = c || nop; const n = T.n; const take = new Array(n).fill(0), skip = new Array(n).fill(0); const order = [];
    (function dfs(u) { T.children[u].forEach(dfs); c.add("visit"); take[u] = T.w[u]; skip[u] = 0; T.children[u].forEach(v => { take[u] += skip[v]; skip[u] += Math.max(take[v], skip[v]); }); order.push(u); if (rec) rec({ u, take: take.slice(), skip: skip.slice() }); })(0);
    const chosen = []; (function pick(u, mayTake) { const t = mayTake && take[u] >= skip[u]; if (t) chosen.push(u); T.children[u].forEach(v => pick(v, !t)); })(0, true);
    return { best: Math.max(take[0], skip[0]), take, skip, chosen, order };
  }
  function treeMwisBrute(T) { const n = T.n; let best = -INF, count = 0; for (let mask = 0; mask < (1 << n); mask++) { count++; let ok = true, s = 0; for (let u = 0; u < n && ok; u++) if (mask >> u & 1) { s += T.w[u]; if (u > 0 && (mask >> T.parent[u] & 1)) ok = false; } if (ok && s > best) best = s; } return { best, count }; }
  /* tree diameter (unweighted edges): returns {diam, through, height[]} with c.add("visit") */
  function treeDiameter(T, c) { c = c || nop; const h = new Array(T.n).fill(0); let diam = 0, through = 0; (function dfs(u) { c.add("visit"); let b1 = -1, b2 = -1; T.children[u].forEach(v => { dfs(v); const hv = h[v] + 1; if (hv > b1) { b2 = b1; b1 = hv; } else if (hv > b2) b2 = hv; }); h[u] = Math.max(0, b1); const cand = (b1 < 0 ? 0 : b1) + (b2 < 0 ? 0 : b2); if (cand > diam) { diam = cand; through = u; } })(0); return { diam, through, height: h }; }
  function treeSizes(T) { const sz = new Array(T.n).fill(1); (function dfs(u) { T.children[u].forEach(v => { dfs(v); sz[u] += sz[v]; }); })(0); return sz; }
  /* rerooting: sum of distances from every node, in O(n) via the standard two-pass, checked against BFS from each node */
  function treeSumDist(T) { const n = T.n, sz = treeSizes(T), down = new Array(n).fill(0), all = new Array(n).fill(0); const post = []; (function dfs(u) { T.children[u].forEach(v => { dfs(v); down[u] += down[v] + sz[v]; }); post.push(u); })(0); all[0] = down[0]; (function pre(u) { T.children[u].forEach(v => { all[v] = all[u] - sz[v] + (n - sz[v]); pre(v); }); })(0); return all; }
  function treeSumDistBrute(T) { const n = T.n, adj = Array.from({ length: n }, () => []); for (let u = 1; u < n; u++) { adj[u].push(T.parent[u]); adj[T.parent[u]].push(u); } return Array.from({ length: n }, (_, s) => { const d = new Array(n).fill(-1); d[s] = 0; const q = [s]; while (q.length) { const u = q.shift(); adj[u].forEach(v => { if (d[v] < 0) { d[v] = d[u] + 1; q.push(v); } }); } return d.reduce((a, b) => a + b, 0); }); }
  function treeFromParents(parent, w) { const n = parent.length; const children = Array.from({ length: n }, () => []); for (let u = 1; u < n; u++) children[parent[u]].push(u); return { n, parent, w, children }; }

  /* ── §21 DAG DP ─────────────────────────────────────────────────────── */
  /* G = {n, edges:[{u,v,w}]}. Kahn topological order; then one pass. mode: "short" | "long" | "count". c.add("relax") per edge. */
  function dagTopo(G) { const n = G.n, indeg = new Array(n).fill(0), adj = Array.from({ length: n }, () => []); G.edges.forEach(e => { adj[e.u].push(e); indeg[e.v]++; }); const q = []; for (let i = 0; i < n; i++) if (!indeg[i]) q.push(i); const order = []; while (q.length) { const u = q.shift(); order.push(u); adj[u].forEach(e => { if (--indeg[e.v] === 0) q.push(e.v); }); } return { order, adj, ok: order.length === n }; }
  function dagDP(G, src, mode, c, rec) {
    c = c || nop; const { order, adj } = dagTopo(G); const n = G.n;
    const d = new Array(n).fill(mode === "count" ? 0 : (mode === "long" ? -INF : INF)); const pred = new Array(n).fill(-1);
    d[src] = mode === "count" ? 1 : 0;
    order.forEach(u => { adj[u].forEach(e => { c.add("relax"); if (mode === "count") { d[e.v] += d[u]; } else if (mode === "short") { if (d[u] + e.w < d[e.v]) { d[e.v] = d[u] + e.w; pred[e.v] = u; } } else { if (d[u] > -INF && d[u] + e.w > d[e.v]) { d[e.v] = d[u] + e.w; pred[e.v] = u; } } if (rec) rec({ u, v: e.v, d: d.slice() }); }); });
    return { d, pred, order };
  }
  /* brute force: enumerate every path from src to every vertex */
  function dagBrute(G, src) { const n = G.n, adj = Array.from({ length: n }, () => []); G.edges.forEach(e => adj[e.u].push(e)); const cnt = new Array(n).fill(0), sh = new Array(n).fill(INF), lo = new Array(n).fill(-INF); let paths = 0; (function go(u, len) { cnt[u]++; paths++; if (len < sh[u]) sh[u] = len; if (len > lo[u]) lo[u] = len; adj[u].forEach(e => go(e.v, len + e.w)); })(src, 0); return { count: cnt, short: sh, long: lo, paths }; }
  /* Bellman-Ford as DP over "at most k edges": dist_k(v) = min(dist_{k-1}(v), min_(u,v) dist_{k-1}(u) + w) */
  function bellmanFordK(G, src, c) { c = c || nop; const n = G.n; let prev = new Array(n).fill(INF); prev[src] = 0; const rounds = [prev.slice()]; for (let k = 1; k <= n - 1; k++) { const cur = prev.slice(); G.edges.forEach(e => { c.add("relax"); if (prev[e.u] + e.w < cur[e.v]) cur[e.v] = prev[e.u] + e.w; }); rounds.push(cur.slice()); prev = cur; } return { d: prev, rounds }; }
  /* Floyd-Warshall as DP over intermediate vertices ⊆ {0..k} */
  function floydWarshall(G, c) { c = c || nop; const n = G.n; const d = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 0 : INF)); G.edges.forEach(e => { if (e.w < d[e.u][e.v]) d[e.u][e.v] = e.w; }); for (let k = 0; k < n; k++) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { c.add("triple"); if (d[i][k] + d[k][j] < d[i][j]) d[i][j] = d[i][k] + d[k][j]; } return d; }

  /* ── §22 Held–Karp ──────────────────────────────────────────────────── */
  /* cities 0..n-1, start at 0. C[S][j] with S a bitmask containing bit 0. c.add("triple") per (S, j, i) examined.
     i ranges over S∖{0, j} when |S| ≥ 3; when S = {0, j} the only predecessor is 0. */
  function heldKarp(D, c, rec) {
    c = c || nop; const n = D.length; const FULL = (1 << n) - 1; const C = Array.from({ length: 1 << n }, () => new Array(n).fill(INF)); const P = Array.from({ length: 1 << n }, () => new Array(n).fill(-1));
    C[1][0] = 0; let states = 0;
    for (let s = 2; s <= n; s++) for (let S = 1; S <= FULL; S += 2) { if (popcount(S) !== s) continue;
      for (let j = 1; j < n; j++) { if (!(S >> j & 1)) continue; const Sj = S & ~(1 << j); states++;
        if (Sj === 1) { c.add("triple"); C[S][j] = D[0][j]; P[S][j] = 0; }
        else for (let i = 1; i < n; i++) { if (!(Sj >> i & 1)) continue; c.add("triple"); const v = C[Sj][i] + D[i][j]; if (v < C[S][j]) { C[S][j] = v; P[S][j] = i; } }
        if (rec) rec({ s, S, j, C: C[S].slice() }); } }
    let best = INF, last = -1; for (let j = 1; j < n; j++) { c.add("close"); const v = C[FULL][j] + D[j][0]; if (v < best) { best = v; last = j; } }
    const tour = []; let S = FULL, j = last; while (j !== -1 && j !== 0) { tour.push(j); const pj = P[S][j]; S &= ~(1 << j); j = pj; } tour.push(0); tour.reverse();
    return { best, tour, C, states };
  }
  function popcount(x) { let k = 0; while (x) { k += x & 1; x >>= 1; } return k; }
  /* brute force over all (n−1)! tours from city 0 */
  function tspBrute(D) { const n = D.length; let best = INF, count = 0, bestTour = null; const rest = []; for (let i = 1; i < n; i++) rest.push(i); (function go(perm, used, len, last) { if (perm.length === n - 1) { count++; const t = len + D[last][0]; if (t < best) { best = t; bestTour = [0].concat(perm); } return; } for (let i = 1; i < n; i++) if (!(used >> i & 1)) go(perm.concat([i]), used | (1 << i), len + D[last][i], i); })([], 1, 0, 0); return { best, count, tour: bestTour }; }
  function hkTriples(n) { let t = n - 1; for (let s = 3; s <= n; s++) t += binom(n - 1, s - 1) * (s - 1) * (s - 2); return t; }
  function binom(n, k) { let r = 1; for (let i = 1; i <= k; i++) r = r * (n - k + i) / i; return Math.round(r); }
  function factorial(n) { let r = 1; for (let i = 2; i <= n; i++) r *= i; return r; }

  /* ── §23 digit DP ───────────────────────────────────────────────────── */
  /* count integers in [0, N] whose digit sum equals k. State (pos, tight, sum). c.add("state") per distinct memo entry computed, c.add("call") per call. rec per state resolved. */
  function digitSumDP(N, k, c, rec) {
    c = c || nop; const ds = String(N).split("").map(Number); const L = ds.length; const memo = new Map(); const visited = [];
    function f(pos, tight, sum) {
      c.add("call"); if (sum > k) return 0;
      if (pos === L) return sum === k ? 1 : 0;
      const key = pos + "," + tight + "," + sum; if (memo.has(key)) { c.add("hit"); return memo.get(key); }
      c.add("state"); const lim = tight ? ds[pos] : 9; let tot = 0;
      for (let d = 0; d <= lim; d++) tot += f(pos + 1, tight && d === lim, sum + d);
      memo.set(key, tot); visited.push({ pos, tight, sum, count: tot }); if (rec) rec({ pos, tight, sum, count: tot });
      return tot;
    }
    const total = f(0, true, 0); return { total, states: visited, digits: ds };
  }
  function digitSumBrute(N, k) { let cnt = 0, hits = []; for (let x = 0; x <= N; x++) { let s = 0, y = x; while (y) { s += y % 10; y = Math.floor(y / 10); } if (s === k) { cnt++; hits.push(x); } } return { count: cnt, hits, checked: N + 1 }; }
  /* count integers in [0, N] with no two adjacent equal digits (leading zeros are not digits) */
  function digitNoAdjDP(N, c, rec) {
    c = c || nop; const ds = String(N).split("").map(Number); const L = ds.length; const memo = new Map(); const visited = [];
    function f(pos, tight, started, prev) {
      c.add("call"); if (pos === L) return 1;
      const key = [pos, tight, started, prev].join(","); if (memo.has(key)) { c.add("hit"); return memo.get(key); }
      c.add("state"); const lim = tight ? ds[pos] : 9; let tot = 0;
      for (let d = 0; d <= lim; d++) { const st = started || d > 0; if (st && started && d === prev) continue; tot += f(pos + 1, tight && d === lim, st, st ? d : -1); }
      memo.set(key, tot); visited.push({ pos, tight, started, prev, count: tot }); if (rec) rec({ pos, tight, started, prev, count: tot });
      return tot;
    }
    const total = f(0, true, false, -1); return { total, states: visited, digits: ds };
  }
  function digitNoAdjBrute(N) { let cnt = 0; for (let x = 0; x <= N; x++) { const s = String(x); let ok = true; for (let i = 1; i < s.length; i++) if (s[i] === s[i - 1]) { ok = false; break; } if (ok) cnt++; } return { count: cnt, checked: N + 1 }; }

  return { mcTable, mcParens, mcTree, mcBrute, catalan, mcCostOfTree,
           obstTable, obstTree, obstCostOf, obstBrute, obstRootMonotone,
           lpsTable, lpsRecon, lpsBrute, lpSubstring, isPal, burstTable, burstOrder, burstBrute, burstFirstWrong,
           treeMwis, treeMwisBrute, treeDiameter, treeSizes, treeSumDist, treeSumDistBrute, treeFromParents,
           dagTopo, dagDP, dagBrute, bellmanFordK, floydWarshall,
           heldKarp, tspBrute, hkTriples, binom, factorial, popcount,
           digitSumDP, digitSumBrute, digitNoAdjDP, digitNoAdjBrute };
})());
Object.assign(DI, {
  mcDims: [30, 35, 15, 5, 10, 20, 25],
  obstP: [0, 0.15, 0.10, 0.05, 0.10, 0.20],
  obstQ: [0.05, 0.10, 0.05, 0.05, 0.05, 0.10],
  lpsWord: "bbabcbcab",
  burstVals: [3, 1, 5, 8],
  /* a rooted tree of 9 nodes: parent[] (node 0 is the root) and weights */
  treeParent: [-1, 0, 0, 1, 1, 2, 4, 4, 5],
  treeW: [4, 6, 3, 5, 2, 7, 1, 8, 3],
  /* a small weighted DAG on 6 vertices */
  dagG: { n: 6, edges: [{ u: 0, v: 1, w: 2 }, { u: 0, v: 2, w: 5 }, { u: 1, v: 2, w: 1 }, { u: 1, v: 3, w: 6 }, { u: 2, v: 3, w: 2 }, { u: 2, v: 4, w: 4 }, { u: 3, v: 4, w: 1 }, { u: 3, v: 5, w: 3 }, { u: 4, v: 5, w: 2 }] },
  /* five cities, symmetric distances (city 0 is home) */
  tspD: [[0, 2, 9, 10, 7], [2, 0, 6, 4, 3], [9, 6, 0, 8, 5], [10, 4, 8, 0, 6], [7, 3, 5, 6, 0]],
  tspXY: [[60, 60], [200, 40], [340, 120], [280, 230], [120, 200]]
});

/* ── chunk D routines ── */
/* ── chunk D routines: checkpointed LCS (§24), the monotone-deque recurrence (§25),
   value iteration / policy iteration on a gridworld and Viterbi / forward on a
   trellis (§26) ──────────────────────────────────────────────────────────── */
Object.assign(DR, (function () {
  const nop = { add: function () {}, get: function () { return 0; } };
  const INF = Infinity;

  /* ── §24 reconstruction after rolling: checkpointing ────────────────────
     LCS of X, Y keeping only every r-th row of the table (plus the current pair of
     rows). Forward pass: fill the table two rows at a time, store row i whenever
     i % r === 0. Traceback: for each block of r rows, rebuild the block from its
     stored checkpoint row, walk back through it, discard it. c.add("cell") per cell
     computed (forward and rebuild), c.add("stored") per checkpoint row kept.
     Returns {len, s, blocks, live} — live = the largest number of rows held at once. */
  function lcsCheckpoint(X, Y, r, c) {
    c = c || nop; const m = X.length, n = Y.length; r = Math.max(1, r | 0);
    const cp = new Map(); cp.set(0, Array(n + 1).fill(0)); c.add("stored");
    let prev = Array(n + 1).fill(0), cur = Array(n + 1).fill(0);
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) { c.add("cell"); cur[j] = X[i - 1] === Y[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]); }
      if (i % r === 0 || i === m) { cp.set(i, cur.slice()); c.add("stored"); }
      const t = prev; prev = cur; cur = t;
    }
    const len = cp.get(m)[n];
    /* traceback block by block from row m down to row 0 */
    const out = []; let i = m, j = n, blocks = 0, live = 0;
    while (i > 0 && j > 0) {
      const top = Math.floor((i - 1) / r) * r;              // the checkpoint row at or above i − 1
      /* rebuild rows top+1 .. i from the stored row `top` */
      const rows = [cp.get(top)]; c.add("rebuild");
      for (let ii = top + 1; ii <= i; ii++) { const row = Array(n + 1).fill(0); const p = rows[rows.length - 1];
        for (let jj = 1; jj <= n; jj++) { c.add("cell"); row[jj] = X[ii - 1] === Y[jj - 1] ? p[jj - 1] + 1 : Math.max(p[jj], row[jj - 1]); }
        rows.push(row); }
      live = Math.max(live, rows.length + cp.size); blocks++;
      /* walk back inside the block (prefer up on ties, as lcsTraceback does) */
      while (i > top && j > 0) { c.add("trace"); const ri = i - top;
        if (X[i - 1] === Y[j - 1]) { out.push(X[i - 1]); i--; j--; }
        else if (rows[ri - 1][j] >= rows[ri][j - 1]) i--; else j--; }
    }
    return { len, s: out.reverse().join(""), blocks, live, stored: cp.size };
  }

  /* ── §25 the sliding-window recurrence  dp[i] = cost[i] + min over j ∈ [i−k, i) of dp[j],  dp[0] = cost[0]
     ("reach position n−1 from 0 in jumps of at most k, paying cost[i] on landing") ── */
  /* naive: examine every j in the window — c.add("cmp") per candidate examined */
  function mqNaive(cost, k, c) {
    c = c || nop; const n = cost.length, dp = [cost[0]], from = [-1];
    for (let i = 1; i < n; i++) { let best = INF, arg = -1;
      for (let j = Math.max(0, i - k); j < i; j++) { c.add("cmp"); if (dp[j] < best) { best = dp[j]; arg = j; } }
      dp[i] = cost[i] + best; from[i] = arg; }
    return { dp, from, val: dp[n - 1] };
  }
  /* monotone deque of indices with increasing dp values: front = the window minimum.
     c.add("cmp") per comparison at the back, c.add("push") / c.add("pop") / c.add("expire").
     rec(frame) after each i with the deque's contents. */
  function mqDeque(cost, k, c, rec) {
    c = c || nop; const n = cost.length, dp = [cost[0]], from = [-1]; const dq = [0]; c.add("push");
    if (rec) rec({ i: 0, dp: dp.slice(), dq: dq.slice(), popped: [], expired: null, min: -1 });
    for (let i = 1; i < n; i++) {
      let expired = null;
      if (dq[0] < i - k) { expired = dq.shift(); c.add("expire"); }        // at most one index leaves the window per step
      const minJ = dq[0]; dp[i] = cost[i] + dp[minJ]; from[i] = minJ;
      const popped = [];
      while (dq.length) { c.add("cmp"); if (dp[dq[dq.length - 1]] >= dp[i]) { popped.push(dq.pop()); c.add("pop"); } else break; }
      dq.push(i); c.add("push");
      if (rec) rec({ i, dp: dp.slice(), dq: dq.slice(), popped, expired, min: minJ });
    }
    return { dp, from, val: dp[n - 1] };
  }
  /* brute force: every sequence of jumps of length ≤ k from 0 to n−1 */
  function mqBrute(cost, k) {
    const n = cost.length; let best = INF, paths = 0;
    (function go(i, acc) { if (i === n - 1) { paths++; if (acc < best) best = acc; return; } for (let s = 1; s <= k && i + s < n; s++) go(i + s, acc + cost[i + s]); })(0, cost[0]);
    return { val: best, paths };
  }
  /* the path from from[] */
  function mqPath(from, n) { const p = []; for (let i = n - 1; i >= 0; i = from[i]) { p.push(i); if (i === 0) break; } return p.reverse(); }

  /* ── §26 value iteration on a gridworld ─────────────────────────────────
     grid: {rows, cols, goal:[r,c], pit:[r,c], walls:[[r,c]…], step (reward per move), goalR, pitR, slip}
     actions 0..3 = up, right, down, left; the intended move happens with prob 1 − 2·slip and
     each perpendicular move with prob slip; moving into a wall or off the grid stays put.
     Terminal states (goal, pit) have V = 0 and the reward is paid on ENTERING them. */
  function gwModel(g) {
    const R = g.rows, C = g.cols, S = R * C; const id = (r, c) => r * C + c;
    const wall = new Set((g.walls || []).map(([r, c]) => id(r, c)));
    const goal = id(g.goal[0], g.goal[1]), pit = id(g.pit[0], g.pit[1]);
    const term = s => s === goal || s === pit;
    const D = [[-1, 0], [0, 1], [1, 0], [0, -1]];
    function move(s, a) { const r = Math.floor(s / C), c = s % C; const nr = r + D[a][0], nc = c + D[a][1]; if (nr < 0 || nr >= R || nc < 0 || nc >= C || wall.has(id(nr, nc))) return s; return id(nr, nc); }
    /* P[s][a] = list of [s', prob, reward] */
    const P = [];
    for (let s = 0; s < S; s++) { P[s] = [];
      for (let a = 0; a < 4; a++) { const out = new Map();
        if (term(s) || wall.has(s)) { out.set(s, 1); }
        else { [[a, 1 - 2 * g.slip], [(a + 1) % 4, g.slip], [(a + 3) % 4, g.slip]].forEach(([aa, pr]) => { if (pr <= 0) return; const t = move(s, aa); out.set(t, (out.get(t) || 0) + pr); }); }
        P[s][a] = [...out.entries()].map(([t, pr]) => [t, pr, term(s) ? 0 : (t === goal ? g.goalR : t === pit ? g.pitR : g.step)]); } }
    return { S, R, C, P, goal, pit, term, wall, id };
  }
  /* one Bellman backup of state s under V: returns {q:[…], best, arg}; c.add("backup") per (s, a) */
  function gwBackup(M, V, s, gamma, c) {
    const q = []; let best = -INF, arg = 0;
    for (let a = 0; a < 4; a++) { c.add("backup"); let v = 0; M.P[s][a].forEach(([t, pr, r]) => { v += pr * (r + gamma * V[t]); }); q[a] = v; if (v > best + 1e-12) { best = v; arg = a; } }
    return { q, best, arg };
  }
  /* value iteration: synchronous sweeps until max residual < eps; rec(k, V, residual) per sweep.
     Returns {V, policy, sweeps, residuals}. */
  function valueIteration(M, gamma, eps, c, rec, maxSweeps) {
    c = c || nop; maxSweeps = maxSweeps || 10000; let V = Array(M.S).fill(0); const residuals = []; let k = 0;
    while (k < maxSweeps) { const W = V.slice(); let res = 0;
      for (let s = 0; s < M.S; s++) { if (M.term(s) || M.wall.has(s)) continue; W[s] = gwBackup(M, V, s, gamma, c).best; res = Math.max(res, Math.abs(W[s] - V[s])); }
      c.add("sweep"); k++; residuals.push(res); V = W; if (rec) rec({ k, V: V.slice(), res });
      if (res < eps) break; }
    const policy = Array(M.S).fill(-1); for (let s = 0; s < M.S; s++) if (!M.term(s) && !M.wall.has(s)) policy[s] = gwBackup(M, V, s, gamma, nop).arg;
    return { V, policy, sweeps: k, residuals };
  }
  /* exact policy evaluation by iterating the linear Bellman-expectation update to tolerance */
  function policyEval(M, policy, gamma, eps) {
    let V = Array(M.S).fill(0);
    for (let it = 0; it < 100000; it++) { const W = V.slice(); let res = 0;
      for (let s = 0; s < M.S; s++) { if (M.term(s) || M.wall.has(s)) continue; let v = 0; M.P[s][policy[s]].forEach(([t, pr, r]) => { v += pr * (r + gamma * V[t]); }); W[s] = v; res = Math.max(res, Math.abs(v - V[s])); }
      V = W; if (res < eps) break; }
    return V;
  }
  /* policy iteration: evaluate, improve, until the policy is stable; c.add("piter") per round */
  function policyIteration(M, gamma, eps, c) {
    c = c || nop; let policy = Array(M.S).fill(0); for (let s = 0; s < M.S; s++) if (M.term(s) || M.wall.has(s)) policy[s] = -1;
    let rounds = 0, V = null;
    while (rounds < 100) { V = policyEval(M, policy, gamma, eps); rounds++; c.add("piter"); let changed = false;
      for (let s = 0; s < M.S; s++) { if (policy[s] < 0) continue; const b = gwBackup(M, V, s, gamma, nop); if (b.q[b.arg] > b.q[policy[s]] + 1e-9) { policy[s] = b.arg; changed = true; } }
      if (!changed) break; }
    return { V, policy, rounds };
  }

  /* ── §26 Viterbi and the forward algorithm on a trellis ─────────────────
     hmm = {pi:[S], A:[S][S], B:[S][K]}, obs = [t0, t1, …] symbol indices.
     c.add("mult") per multiplication; c.add("max") per max over predecessors. */
  function viterbi(hmm, obs, c, rec) {
    c = c || nop; const S = hmm.pi.length, T = obs.length; const delta = [], psi = [];
    delta[0] = []; psi[0] = [];
    for (let j = 0; j < S; j++) { c.add("mult"); delta[0][j] = hmm.pi[j] * hmm.B[j][obs[0]]; psi[0][j] = -1; }
    if (rec) rec({ t: 0, delta: delta[0].slice(), psi: psi[0].slice() });
    for (let t = 1; t < T; t++) { delta[t] = []; psi[t] = [];
      for (let j = 0; j < S; j++) { let best = -1, arg = -1;
        for (let i = 0; i < S; i++) { c.add("mult"); const v = delta[t - 1][i] * hmm.A[i][j]; if (v > best) { best = v; arg = i; } }
        c.add("max"); c.add("mult"); delta[t][j] = best * hmm.B[j][obs[t]]; psi[t][j] = arg; }
      if (rec) rec({ t, delta: delta[t].slice(), psi: psi[t].slice() }); }
    let best = -1, last = -1; for (let j = 0; j < S; j++) if (delta[T - 1][j] > best) { best = delta[T - 1][j]; last = j; }
    const path = [last]; for (let t = T - 1; t > 0; t--) path.push(psi[t][path[path.length - 1]]); path.reverse();
    return { delta, psi, prob: best, path };
  }
  function forward(hmm, obs, c) {
    c = c || nop; const S = hmm.pi.length, T = obs.length; let alpha = [];
    for (let j = 0; j < S; j++) { c.add("mult"); alpha[j] = hmm.pi[j] * hmm.B[j][obs[0]]; }
    const alphas = [alpha.slice()];
    for (let t = 1; t < T; t++) { const nx = [];
      for (let j = 0; j < S; j++) { let s = 0; for (let i = 0; i < S; i++) { c.add("mult"); s += alpha[i] * hmm.A[i][j]; } c.add("mult"); nx[j] = s * hmm.B[j][obs[t]]; }
      alpha = nx; alphas.push(alpha.slice()); }
    return { total: alpha.reduce((a, b) => a + b, 0), alphas };
  }
  /* brute force over all S^T state sequences: the best one and the total probability */
  function hmmBrute(hmm, obs) {
    const S = hmm.pi.length, T = obs.length; let best = -1, bestPath = null, total = 0, count = 0;
    (function go(t, prev, p, path) {
      if (t === T) { count++; total += p; if (p > best) { best = p; bestPath = path.slice(); } return; }
      for (let j = 0; j < S; j++) { const q = p * (t === 0 ? hmm.pi[j] : hmm.A[prev][j]) * hmm.B[j][obs[t]]; path.push(j); go(t + 1, j, q, path); path.pop(); }
    })(0, -1, 1, []);
    return { prob: best, path: bestPath, total, count };
  }
  /* probability of one given path (for re-scoring the Viterbi path independently) */
  function hmmPathProb(hmm, obs, path) { let p = hmm.pi[path[0]] * hmm.B[path[0]][obs[0]]; for (let t = 1; t < obs.length; t++) p *= hmm.A[path[t - 1]][path[t]] * hmm.B[path[t]][obs[t]]; return p; }
  /* beam search with width k over the same trellis — NOT exact; returns the best surviving path */
  function hmmBeam(hmm, obs, k) {
    const S = hmm.pi.length, T = obs.length; let beam = [];
    for (let j = 0; j < S; j++) beam.push({ p: hmm.pi[j] * hmm.B[j][obs[0]], path: [j] });
    beam.sort((a, b) => b.p - a.p); beam = beam.slice(0, k);
    for (let t = 1; t < T; t++) { const nx = [];
      beam.forEach(h => { for (let j = 0; j < S; j++) nx.push({ p: h.p * hmm.A[h.path[t - 1]][j] * hmm.B[j][obs[t]], path: h.path.concat([j]) }); });
      nx.sort((a, b) => b.p - a.p); beam = nx.slice(0, k); }
    return beam[0];
  }

  return { lcsCheckpoint, mqNaive, mqDeque, mqBrute, mqPath,
           gwModel, gwBackup, valueIteration, policyEval, policyIteration,
           viterbi, forward, hmmBrute, hmmPathProb, hmmBeam };
})());
Object.assign(DI, {
  /* §25: landing costs for the sliding-window recurrence (n = 12) */
  mqCosts: [3, 7, 2, 9, 4, 8, 1, 6, 5, 7, 2, 3],
  /* §26: the 4 × 4 gridworld — goal top-right, pit just below it, one wall; rewards on entering */
  gridworld: { rows: 4, cols: 4, goal: [0, 3], pit: [1, 3], walls: [[1, 1]], step: -0.04, goalR: 1, pitR: -1, slip: 0.1 },
  /* §26: a three-state HMM over two symbols, and the six observations of the figure */
  hmm: { names: ["S₁", "S₂", "S₃"], pi: [0.5, 0.3, 0.2], A: [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]], B: [[0.7, 0.3], [0.4, 0.6], [0.1, 0.9]], syms: ["a", "b"] },
  hmmObs: [0, 1, 1, 0, 1, 1],
  /* §26: the hand-worked two-state, three-step example */
  hmm2: { names: ["A", "B"], pi: [0.6, 0.4], A: [[0.7, 0.3], [0.4, 0.6]], B: [[0.5, 0.5], [0.1, 0.9]], syms: ["u", "v"] },
  hmm2Obs: [0, 1, 1]
});

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

/* ── 01  #rod-tree-svg  rod cutting's call tree grown call by call ───── */
(function () {
  if (!SX.has("rod-tree-svg")) return;
  const W = 680, H = 300;
  function build() {
    const n = +SX.val("rt-n", 4), mode = SX.val("rt-mode", "naive");
    const p = DI.rodPrices; const c = AL.counter(); const ev = [];
    const v = mode === "naive" ? DR.rodNaive(p, n, c, e => ev.push(e)) : DR.rodMemo(p, n, c, e => ev.push(e));
    /* lay out the tree: children in call order, positions by leaf count */
    const nodes = new Map(); const rootIds = [];
    ev.forEach(e => { if (e.type === "call") { nodes.set(e.id, { id: e.id, n: e.n, parent: e.parent, hit: e.hit, kids: [], depth: 0 }); if (e.parent >= 0) { const par = nodes.get(e.parent); par.kids.push(e.id); nodes.get(e.id).depth = par.depth + 1; } else rootIds.push(e.id); } });
    let leafX = 0;
    function place(id) { const nd = nodes.get(id); if (!nd.kids.length) { nd.x = leafX++; return; } nd.kids.forEach(place); nd.x = (nodes.get(nd.kids[0]).x + nodes.get(nd.kids[nd.kids.length - 1]).x) / 2; }
    rootIds.forEach(place);
    const maxDepth = Math.max(...[...nodes.values()].map(d => d.depth));
    const frames = []; ev.forEach(e => { if (e.type === "call") frames.push(e.id); });
    const svg = d3.select("#rod-tree-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 10, r: 10, t: 14, b: 10 });
    const xs = d3.scaleLinear().domain([-0.5, Math.max(leafX - 0.5, 0.5)]).range([0, F.iw]);
    const ys = d3.scaleLinear().domain([0, Math.max(maxDepth, 1)]).range([12, F.ih - 14]);
    const r = Math.max(4, Math.min(11, F.iw / (2.4 * Math.max(leafX, 1))));
    function render(f, k) {
      F.g.selectAll("*").remove();
      const shown = new Set(frames.slice(0, k + 1));
      nodes.forEach(nd => { if (!shown.has(nd.id) || nd.parent < 0) return; const par = nodes.get(nd.parent);
        F.g.append("line").attr("x1", xs(par.x)).attr("y1", ys(par.depth)).attr("x2", xs(nd.x)).attr("y2", ys(nd.depth)).attr("stroke", AC.line); });
      nodes.forEach(nd => { if (!shown.has(nd.id)) return; const cur = nd.id === f;
        F.g.append("circle").attr("cx", xs(nd.x)).attr("cy", ys(nd.depth)).attr("r", cur ? r + 2 : r).attr("fill", cur ? AC.a2 : nd.hit ? AC.good : AC.accent).attr("stroke", AC.line);
        if (r >= 6) F.g.append("text").attr("x", xs(nd.x)).attr("y", ys(nd.depth) + 3.5).attr("text-anchor", "middle").attr("font-size", Math.min(11, r + 2)).attr("fill", AC.bg).text(nd.n); });
      const done = frames.slice(0, k + 1); const hits = done.filter(id => nodes.get(id).hit).length;
      F.g.append("text").attr("x", F.iw).attr("y", 4).attr("text-anchor", "end").attr("font-size", 11).attr("fill", AC.muted).text(`calls so far ${done.length}${mode === "memo" ? ", hits " + hits : ""} · calling CUT(${nodes.get(f).n})`);
    }
    AL.stepper(svg, { frames, render, delay: 350, label: "call" });
    const calls = c.get("call"), hits = c.get("hit"), comp = c.get("compute");
    const brute = DR.rodBrute(p, n);
    SX.setHtml("rod-tree-readout", mode === "naive"
      ? `naive CUT(${n}) on the price table of §04: value <b>${v}</b> · measured <b>${SX.int(calls)}</b> calls = 2ⁿ = ${SX.int(1 << n)} ${SX.flag(calls === (1 << n))} · leaves ${SX.int([...nodes.values()].filter(d => !d.kids.length).length)} = 2ⁿ⁻¹ = ${1 << (n - 1)}, one per cutting ${SX.flag([...nodes.values()].filter(d => !d.kids.length).length === (1 << (n - 1)))} · distinct subproblems only ${n + 1} · brute force over all ${brute.count} cuttings: <b>${brute.best}</b> ${SX.flag(brute.best === v)}`
      : `memoized CUT(${n}): value <b>${v}</b> · measured <b>${calls}</b> calls = 1 + n(n+1)/2 = ${1 + n * (n + 1) / 2} ${SX.flag(calls === 1 + n * (n + 1) / 2)}, of which <b>${hits}</b> memo hits and <b>${comp}</b> computations = n + 1 = ${n + 1} ${SX.flag(comp === n + 1)} · naive would make ${SX.int(1 << n)} calls · brute force over ${brute.count} cuttings: <b>${brute.best}</b> ${SX.flag(brute.best === v)}`);
  }
  SX.on("rt-n", "input", build); SX.on("rt-mode", "change", build);
  build();
})();

/* ── chunk A figures ── */
/* ── 02  #rod-table-svg  rod cutting's r[] and s[] filled candidate by candidate, the cutting reconstructed ── */
(function () {
  if (!SX.has("rod-table-svg")) return;
  const W = 680, H = 320;
  function build() {
    const n = +SX.val("rtb-n", 7), form = SX.val("rtb-form", "first");
    const p = DI.rodPrices; const c = AL.counter();
    const split = form !== "first", sym = form === "sym";
    const T = split ? DR.rodTableSplit(p, n, c, null, sym) : DR.rodTable(p, n, c);
    const rc = AL.counter();
    const cuts = split ? DR.rodCutsSplit(T.s, n, rc) : DR.rodCuts(T.s, n, rc);
    const brute = DR.rodOptCount(p, n);
    /* frames: for each j, one per candidate (recomputed from the fixed r[] the routine returned), then a fix frame; then reconstruction frames */
    const frames = [];
    for (let j = 1; j <= n; j++) {
      const cands = [];
      if (!split) for (let i = 1; i <= j; i++) cands.push({ label: `i=${i}`, expr: `${p[i]}+r[${j - i}]`, v: p[i] + T.r[j - i] });
      else { cands.push({ label: "whole", expr: `p${j}`, v: p[j] }); const top = sym ? Math.floor(j / 2) : j - 1; for (let i = 1; i <= top; i++) cands.push({ label: `i=${i}`, expr: `r[${i}]+r[${j - i}]`, v: T.r[i] + T.r[j - i] }); }
      for (let k = 0; k < cands.length; k++) frames.push({ type: "cand", j, k, cands });
      frames.push({ type: "fix", j, k: cands.length - 1, cands });
    }
    /* reconstruction: the sequence of s[] look-ups, as the piece list grows */
    const pieces = [];
    if (!split) { let m = n; while (m > 0) { pieces.push({ at: m, len: T.s[m] }); m -= T.s[m]; } }
    else { (function go(j) { if (T.s[j] === 0) { pieces.push({ at: j, len: j }); return; } go(T.s[j]); go(j - T.s[j]); })(n); }
    for (let k = 1; k <= pieces.length; k++) frames.push({ type: "recon", j: n, k, cands: [] });
    const svg = d3.select("#rod-table-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 44, r: 10, t: 14, b: 8 });
    const cw = Math.min(40, Math.floor((F.iw - 10) / (n + 1)) - 3);
    function render(f) {
      F.g.selectAll("*").remove();
      const fixedUpTo = f.type === "fix" || f.type === "recon" ? f.j : f.j - 1;
      const reconSet = new Set(f.type === "recon" ? pieces.slice(0, f.k).map(q => q.at) : []);
      const rows = ["p", "r", "s"];
      SX.grid(F.g, 3, n + 1, { x: 0, y: 12, w: cw, h: 22, gap: 3, rowLabel: i => rows[i] + "[j]", colLabel: j => "j=" + j,
        text: (i, j) => i === 0 ? (j === 0 ? "–" : p[j]) : j <= fixedUpTo ? (i === 1 ? T.r[j] : (j === 0 ? "–" : (split && T.s[j] === 0 ? "∅" : T.s[j]))) : "",
        fill: (i, j) => reconSet.has(j) && i > 0 ? AC.good : (j === f.j && f.type !== "recon" && i > 0 ? AC.a2 : (i > 0 && j <= fixedUpTo && j > 0 ? AC.accent : null)) });
      /* candidate strip */
      const y0 = 12 + 3 * 25 + 22;
      if (f.type !== "recon") {
        F.g.append("text").attr("x", 0).attr("y", y0).attr("font-size", 11).attr("fill", AC.muted)
          .text(`candidates for r[${f.j}]` + (split ? "  (whole rod, then one cut at i)" : "  (first piece i, then the solved remainder)"));
        let best = -Infinity, bestK = -1;
        for (let k = 0; k <= f.k; k++) if (f.cands[k].v > best) { best = f.cands[k].v; bestK = k; }
        const bw = Math.min(88, Math.floor(F.iw / Math.max(f.cands.length, 1)) - 4);
        f.cands.forEach((cd, k) => {
          const x = k * (bw + 4), shown = k <= f.k;
          F.g.append("rect").attr("x", x).attr("y", y0 + 8).attr("width", bw).attr("height", 40).attr("rx", 4)
            .attr("fill", !shown ? AC.panel2 : k === bestK ? AC.good : k === f.k ? AC.a2 : AC.panel2).attr("stroke", AC.line);
          if (shown) { F.g.append("text").attr("x", x + bw / 2).attr("y", y0 + 24).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", k === bestK || k === f.k ? AC.bg : AC.muted).text(cd.label + ": " + cd.expr);
            F.g.append("text").attr("x", x + bw / 2).attr("y", y0 + 41).attr("text-anchor", "middle").attr("font-size", 13).attr("fill", k === bestK || k === f.k ? AC.bg : AC.ink).text(cd.v); }
        });
        F.g.append("text").attr("x", F.iw).attr("y", y0).attr("text-anchor", "end").attr("font-size", 11).attr("fill", AC.muted)
          .text(f.type === "fix" ? `r[${f.j}] = ${T.r[f.j]},  s[${f.j}] = ${split && T.s[f.j] === 0 ? "∅ (whole)" : T.s[f.j]}` : `examined ${f.k + 1} of ${f.cands.length} · best so far ${best}`);
      } else {
        F.g.append("text").attr("x", 0).attr("y", y0).attr("font-size", 11).attr("fill", AC.muted)
          .text(split ? `reconstruction: follow s[] recursively — s[j] = ∅ means "sell the piece whole"` : `reconstruction: piece s[${n}] = ${T.s[n]}, then s[${n - T.s[n]}], … until nothing is left`);
        F.g.append("text").attr("x", F.iw).attr("y", y0).attr("text-anchor", "end").attr("font-size", 11).attr("fill", AC.muted).text(`pieces so far: ${pieces.slice(0, f.k).map(q => q.len).join(" + ")}`);
      }
      /* the rod, drawn in pieces at the end */
      const yr = y0 + 66, unit = Math.min(60, (F.iw - 4) / Math.max(n, 1));
      F.g.append("rect").attr("x", 0).attr("y", yr).attr("width", unit * n).attr("height", 30).attr("rx", 4).attr("fill", AC.panel2).attr("stroke", AC.line);
      if (f.type === "recon") { let x = 0; pieces.slice(0, f.k).forEach((q, k) => {
        F.g.append("rect").attr("x", x + 1).attr("y", yr + 1).attr("width", unit * q.len - 2).attr("height", 28).attr("rx", 3).attr("fill", k === f.k - 1 ? AC.a2 : AC.good).attr("stroke", AC.line);
        F.g.append("text").attr("x", x + unit * q.len / 2).attr("y", yr + 20).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", AC.bg).text(`${q.len} → ${p[q.len]}`); x += unit * q.len; }); }
      F.g.append("text").attr("x", 0).attr("y", yr + 46).attr("font-size", 11).attr("fill", AC.muted).text(f.type === "recon" ? `rod of length ${n}: revenue ${T.r[n]} = ${pieces.slice(0, f.k).map(q => p[q.len]).join(" + ")}${f.k < pieces.length ? " + …" : ""}` : `rod of length ${n} — the cutting is drawn once the table is complete`);
    }
    AL.stepper(svg, { frames, render, delay: 260, label: "step" });
    const cells = c.get("cell"), expect = split ? (sym ? Array.from({ length: n }, (_, j) => 1 + Math.floor((j + 1) / 2)).reduce((a, b) => a + b, 0) : n * (n + 1) / 2) : n * (n + 1) / 2;
    const sumP = cuts.reduce((a, l) => a + p[l], 0);
    SX.setHtml("rod-table-readout",
      `r[${n}] = <b>${T.r[n]}</b> · cutting from s[]: <b>${cuts.join(" + ")}</b> (prices ${cuts.map(l => p[l]).join(" + ")} = ${sumP}) ${SX.flag(sumP === T.r[n])} in <b>${rc.get("recon")}</b> reconstruction steps · measured candidate examinations <b>${cells}</b> = ${split ? (sym ? "∑ (1 + ⌊j/2⌋)" : "n(n+1)/2") : "n(n+1)/2"} = ${expect} ${SX.flag(cells === expect)} · brute force over all ${SX.int(brute.count)} cuttings: best <b>${brute.best}</b> ${SX.flag(brute.best === T.r[n])}, attained by ${brute.opt} cutting${brute.opt === 1 ? "" : "s"}`);
  }
  SX.on("rtb-n", "input", build); SX.on("rtb-form", "change", build);
  build();
})();

/* ── 03  #subgraph-svg  the subproblem graph in tabulation order, the DFS from the goal stepped ── */
(function () {
  if (!SX.has("subgraph-svg")) return;
  const W = 680, H = 300;
  function build() {
    const kind = SX.val("sg-kind", "rod"), n = +SX.val("sg-n", 6);
    const G = kind === "rod" ? DR.sgRod(n) : kind === "fib" ? DR.sgFib(n) : DR.sgLcs(DI.lcsSmall[0], DI.lcsSmall[1]);
    const cd = AL.counter(); const D = DR.sgDfs(G, cd);
    const ct = AL.counter(); const topo = DR.sgTopo(G, ct);
    const topoPos = new Map(); topo.forEach((id, k) => topoPos.set(id, k));
    /* the real memoized routine, for the call-count cross-check */
    const cm = AL.counter();
    if (kind === "rod") DR.rodMemo(DI.rodPrices, n, cm); else if (kind === "fib") DR.fibMemo(n, cm); else DR.sgLcsMemoCalls(DI.lcsSmall[0], DI.lcsSmall[1], cm);
    const svg = d3.select("#subgraph-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 16, r: 16, t: 16, b: 8 });
    /* positions */
    const pos = new Map(); let R = 13;
    if (kind !== "lcs") {
      const xs = d3.scaleLinear().domain([0, Math.max(n, 1)]).range([R + 4, F.iw - R - 4]);
      G.V.forEach(v => pos.set(v.id, { x: xs(v.id), y: F.ih - 40 }));
      R = Math.max(8, Math.min(13, (F.iw / (n + 1)) / 3));
    } else {
      const m = G.m, nn = G.n, sx = Math.min(70, (F.iw - 60) / nn), sy = Math.min(44, (F.ih - 30) / m);
      const x0 = (F.iw - sx * nn) / 2, y0 = 14;
      G.V.forEach(v => pos.set(v.id, { x: x0 + v.j * sx, y: y0 + v.i * sy })); R = 11;
    }
    const frames = D.events; const vById = new Map(G.V.map(v => [v.id, v]));
    function render(f, k) {
      F.g.selectAll("*").remove();
      const done = frames.slice(0, k + 1);
      const entered = new Set(done.filter(e => e.type === "enter").map(e => e.id));
      const fixedOrder = new Map(); done.filter(e => e.type === "fix").forEach((e, i) => fixedOrder.set(e.id, i + 1));
      const hitEdge = f.type === "hit" ? f.from + "→" + f.id : null;
      /* edges */
      G.E.forEach(([u, v]) => {
        const a = pos.get(u), b = pos.get(v), key = u + "→" + v;
        const used = entered.has(u) && (entered.has(v) || fixedOrder.has(v));
        const col = key === hitEdge ? AC.good : used ? AC.accent : AC.line;
        if (kind !== "lcs") {
          const dist = Math.abs(a.x - b.x), h = Math.min(F.ih - 60, 18 + dist * 0.55);
          F.g.append("path").attr("d", `M${a.x},${a.y - R} Q${(a.x + b.x) / 2},${a.y - R - h} ${b.x},${b.y - R}`).attr("fill", "none").attr("stroke", col).attr("stroke-width", key === hitEdge ? 2.5 : 1.2).attr("opacity", used ? 0.95 : 0.6);
          const t = 0.94, mx = (1 - t) * (1 - t) * a.x + 2 * (1 - t) * t * (a.x + b.x) / 2 + t * t * b.x, my = (1 - t) * (1 - t) * (a.y - R) + 2 * (1 - t) * t * (a.y - R - h) + t * t * (b.y - R);
          AL.arrow(F.g, mx, my, b.x, b.y - R, { color: col, w: 1, head: 5 });
        } else {
          const dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy), ux = dx / L, uy = dy / L;
          AL.arrow(F.g, a.x + ux * R, a.y + uy * R, b.x - ux * (R + 1), b.y - uy * (R + 1), { color: col, w: key === hitEdge ? 2.5 : 1.2, head: 5 });
        }
      });
      /* vertices */
      G.V.forEach(v => {
        const q = pos.get(v.id), cur = f.id === v.id && f.type === "enter", fixed = fixedOrder.has(v.id), ent = entered.has(v.id);
        F.g.append("circle").attr("cx", q.x).attr("cy", q.y).attr("r", cur ? R + 2 : R).attr("fill", cur ? AC.a2 : fixed ? AC.accent : ent ? AC.panel2 : AC.panel).attr("stroke", cur ? AC.a2 : ent ? AC.accent : AC.line).attr("stroke-width", v.id === G.goal ? 2.5 : 1);
        F.g.append("text").attr("x", q.x).attr("y", q.y + 3.5).attr("text-anchor", "middle").attr("font-size", kind === "lcs" ? 9 : 10).attr("fill", cur || fixed ? AC.bg : AC.ink).text(v.label);
        if (fixed) F.g.append("text").attr("x", q.x + R + 2).attr("y", q.y - R + 2).attr("font-size", 9).attr("fill", AC.good).text("#" + fixedOrder.get(v.id));
        if (kind !== "lcs") F.g.append("text").attr("x", q.x).attr("y", q.y + R + 12).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", AC.muted).text("t" + topoPos.get(v.id));
      });
      if (kind === "lcs") { const [x, y] = DI.lcsSmall; const p0 = pos.get(0);
        for (let j = 1; j <= G.n; j++) F.g.append("text").attr("x", pos.get(j).x).attr("y", p0.y - R - 4).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.muted).text(y[j - 1]);
        for (let i = 1; i <= G.m; i++) F.g.append("text").attr("x", p0.x - R - 8).attr("y", pos.get(i * (G.n + 1)).y + 3).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.muted).text(x[i - 1]); }
      const msg = f.type === "enter" ? `enter ${vById.get(f.id).label} — first visit: compute it` : f.type === "hit" ? `${vById.get(f.from).label} consults ${vById.get(f.id).label}: already fixed — memo hit` : `fix ${vById.get(f.id).label} (post-order #${fixedOrder.get(f.id)})`;
      F.g.append("text").attr("x", 0).attr("y", F.ih + 2).attr("font-size", 11).attr("fill", AC.muted).text(msg + (kind !== "lcs" ? "   ·   t = position in the tabulation (reverse topological) order" : ""));
    }
    AL.stepper(svg, { frames, render, delay: 320, label: "event" });
    const Vn = G.V.length, En = G.E.length;
    const closedV = kind === "lcs" ? (G.m + 1) * (G.n + 1) : n + 1;
    const closedE = kind === "rod" ? n * (n + 1) / 2 : kind === "fib" ? Math.max(0, 2 * (n - 1)) : null;
    const eLabel = kind === "rod" ? "n(n+1)/2" : kind === "fib" ? "2(n−1)" : "between mn and 2mn";
    const edgesOK = kind === "lcs" ? (En >= G.m * G.n && En <= 2 * G.m * G.n) : En === closedE;
    const calls = cm.get("call");
    SX.setHtml("subgraph-readout",
      `measured |V| = <b>${Vn}</b> = ${kind === "lcs" ? "(m+1)(n+1)" : "n + 1"} = ${closedV} ${SX.flag(Vn === closedV)} · |E| = <b>${En}</b> ${kind === "lcs" ? "(" + eLabel + " = " + G.m * G.n + "…" + 2 * G.m * G.n + ")" : "= " + eLabel + " = " + closedE} ${SX.flag(edgesOK)} · Kahn's algorithm popped ${ct.get("pop")} vertices${kind !== "lcs" ? " in the order " + topo.join(" ") : " (row-major is one valid order)"} · DFS from the goal reached <b>${D.reached.size}</b> of ${Vn} vertices following <b>${cd.get("edge")}</b> edges → 1 + edges = ${1 + cd.get("edge")} calls; the real memoized routine made <b>${calls}</b> calls with ${cm.get("hit")} hits ${SX.flag(calls === 1 + cd.get("edge"))}`);
  }
  SX.on("sg-kind", "change", build); SX.on("sg-n", "input", build);
  build();
})();

/* ── 04  #mt-svg  memoization vs tabulation measured across n; the sparse recurrence ── */
(function () {
  if (!SX.has("mt-svg")) return;
  const W = 680, H = 300;
  function pricesUpTo(n) { const p = [0]; for (let i = 1; i <= n; i++) p[i] = DI.rodPrices[((i - 1) % 10) + 1]; return p; }
  function build() {
    const prob = SX.val("mt-prob", "rod"), maxN = +SX.val("mt-max", 40);
    const svg = d3.select("#mt-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 56, r: 16, t: 14, b: 36 });
    let series = [], readout = "";
    if (prob !== "sparse") {
      const ns = d3.range(1, maxN + 1); const rows = [];
      let allAgree = true, bruteMsg = "";
      ns.forEach(n => {
        const cm = AL.counter(), ct = AL.counter(); let vm, vt;
        if (prob === "rod") { const p = pricesUpTo(n); vm = DR.rodMemo(p, n, cm); vt = DR.rodTable(p, n, ct).r[n]; if (n <= 12) { const b = DR.rodBrute(p, n); if (b.best !== vt) allAgree = false; bruteMsg = `brute force over all 2ⁿ⁻¹ cuttings agrees for every n ≤ 12`; } }
        else { vm = DR.fibMemo(n, cm); vt = DR.fibTable(n, ct); const vi = DR.fibIter(n); if (vi !== vt) allAgree = false; bruteMsg = `the plain iterative pair-update agrees for every n`; }
        if (vm !== vt) allAgree = false;
        rows.push({ n, calls: cm.get("call"), hits: cm.get("hit"), computed: cm.get("call") - cm.get("hit"), cells: ct.get("cell") });
      });
      series = [
        { key: "calls", label: "memo calls", color: AC.a2 }, { key: "hits", label: "memo hits", color: AC.rose },
        { key: "computed", label: "memo computed states", color: AC.good }, { key: "cells", label: "table cells", color: AC.accent, dash: "6,4" }];
      const x = d3.scaleLinear().domain([1, maxN]).range([0, F.iw]);
      const ymax = d3.max(rows, r => Math.max(r.calls, r.cells));
      const y = d3.scaleLinear().domain([0, ymax]).nice().range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 5); AL.axisB(F.g, x, F.ih, 8, "n"); AL.axisL(F.g, y, 5, "measured count");
      series.forEach(s => { const ln = d3.line().x(r => x(r.n)).y(r => y(r[s.key]));
        const path = F.g.append("path").datum(rows).attr("d", ln).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", 2); if (s.dash) path.attr("stroke-dasharray", s.dash);
        F.g.selectAll(null).data(rows.filter((_, i) => i % Math.max(1, Math.floor(rows.length / 20)) === 0)).join("circle").attr("cx", r => x(r.n)).attr("cy", r => y(r[s.key])).attr("r", 2.2).attr("fill", s.color); });
      AL.legend(F.g, series, 8, 14);
      const L = rows[rows.length - 1], n = L.n;
      const exp = prob === "rod" ? { calls: 1 + n * (n + 1) / 2, hits: n * (n - 1) / 2, computed: n + 1, cells: n * (n + 1) / 2, f: "1 + n(n+1)/2, n(n−1)/2, n + 1, n(n+1)/2" }
                                 : { calls: 2 * n - 1, hits: n - 2, computed: n + 1, cells: n - 1, f: "2n − 1, n − 2, n + 1, n − 1" };
      readout = `at n = ${n}: memo calls <b>${SX.int(L.calls)}</b>, hits <b>${SX.int(L.hits)}</b>, computed <b>${L.computed}</b>; table cells <b>${SX.int(L.cells)}</b> — closed forms (${exp.f}) = ${SX.int(exp.calls)}, ${SX.int(exp.hits)}, ${exp.computed}, ${SX.int(exp.cells)} ${SX.flag(L.calls === exp.calls && L.hits === exp.hits && L.computed === exp.computed && L.cells === exp.cells)} · the two evaluators return the same value at every n from 1 to ${n} ${SX.flag(allAgree)}; ${bruteMsg} · the memo's recursion is ${prob === "rod" ? n + 1 : n} frames deep at n = ${n}; the loop's is 0`;
    } else {
      const E = Math.min(7, Math.max(3, Math.round(maxN / 10) + 2)); const rows = []; let allAgree = true;
      for (let e = 1; e <= E; e++) { const n = Math.pow(10, e); const cm = AL.counter(), ct = AL.counter();
        const r = DR.sparseMemo(n, cm); const vt = DR.sparseTable(n, ct); if (vt !== r.value) allAgree = false;
        rows.push({ n, states: r.states, calls: cm.get("call"), cells: ct.get("cell"), value: r.value }); }
      series = [{ key: "cells", label: "table cells = n", color: AC.accent, dash: "6,4" }, { key: "calls", label: "memo calls", color: AC.a2 }, { key: "states", label: "memo computed states", color: AC.good }];
      const x = d3.scaleLog().domain([10, Math.pow(10, E)]).range([0, F.iw]);
      const y = d3.scaleLog().domain([1, Math.pow(10, E)]).range([F.ih, 0]);
      AL.axisB(F.g, x, F.ih, E, "n (log scale)", d3.format("~s")); AL.axisL(F.g, y, E, "measured count (log scale)", d3.format("~s"));
      series.forEach(s => { const ln = d3.line().x(r => x(r.n)).y(r => y(Math.max(1, r[s.key])));
        const path = F.g.append("path").datum(rows).attr("d", ln).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", 2); if (s.dash) path.attr("stroke-dasharray", s.dash);
        F.g.selectAll(null).data(rows).join("circle").attr("cx", r => x(r.n)).attr("cy", r => y(Math.max(1, r[s.key]))).attr("r", 3).attr("fill", s.color);
        rows.forEach(r => F.g.append("text").attr("x", x(r.n) + 5).attr("y", y(Math.max(1, r[s.key])) + 3).attr("font-size", 9).attr("fill", s.color).text(SX.int(r[s.key]))); });
      AL.legend(F.g, series, 8, 14);
      const L = rows[rows.length - 1];
      const big = DR.sparseMemo(1e9, AL.counter());
      readout = `at n = 10${"⁰¹²³⁴⁵⁶⁷"[E]}: the memo computed <b>${L.states}</b> states in <b>${L.calls}</b> calls (= 1 + 3·(states − 1) = ${1 + 3 * (L.states - 1)} ${SX.flag(L.calls === 1 + 3 * (L.states - 1))}); the table filled <b>${SX.int(L.cells)}</b> cells; both return f(n) = <b>${SX.int(L.value)}</b> ${SX.flag(allAgree)} (agreement checked at every power of ten) · at n = 10⁹ the memo computes <b>${big.states}</b> states (f = ${SX.int(big.value)}); a table would need 10⁹ + 1 cells, not attempted · upper bound (log₂n + 1)(log₃n + 1) on reachable states at 10${"⁰¹²³⁴⁵⁶⁷"[E]}: ${Math.floor((Math.log2(L.n) + 1) * (Math.log(L.n) / Math.log(3) + 1))} ${SX.flag(L.states <= (Math.log2(L.n) + 1) * (Math.log(L.n) / Math.log(3) + 1))}`;
    }
    SX.setHtml("mt-readout", readout);
  }
  SX.on("mt-prob", "change", build); SX.on("mt-max", "input", build);
  build();
})();

/* ── 05  #lin-svg  four 1-D strips filled left to right, each checked against brute force ── */
(function () {
  if (!SX.has("lin-svg")) return;
  const W = 680, H = 330;
  function build() {
    const n = +SX.val("lin-n", 10), seed = +SX.val("lin-seed", 0);
    let rv = DI.robberVals, kv = DI.kadaneVals;
    if (seed > 0) { const r = AL.rng(1000 + seed); rv = Array.from({ length: 7 }, () => AL.randInt(r, 1, 9)); kv = Array.from({ length: 9 }, () => AL.randInt(r, -5, 6)); }
    /* run the four routines under counters */
    const cf = AL.counter(), cs = AL.counter(), cr = AL.counter(), ck = AL.counter();
    const fib = []; for (let i = 0; i <= n; i++) fib[i] = DR.fibIter(i); DR.fibTable(n, cf);
    const st = DR.stairs(n, cs); const rb = DR.robber(rv, cr); const kd = DR.kadane(kv, ck);
    const cmx = AL.counter(); const fm = DR.fibMatrix(n, cmx); const fc = DR.fibClosed(n);
    const sb = DR.stairsBrute(n), rbb = DR.robberBrute(rv), kb = DR.kadaneBrute(kv);
    /* robber reconstruction: walk back */
    const taken = []; { let i = rv.length - 1; while (i >= 0) { if (i === 0 || rb[i] !== rb[i - 1]) { taken.push(i); i -= 2; } else i--; } taken.reverse(); }
    /* kadane endpoints */
    let kEnd = 0; for (let i = 1; i < kv.length; i++) if (kd.end[i] > kd.end[kEnd]) kEnd = i; let kStart = kEnd; while (kStart > 0 && kd.end[kStart - 1] > 0) kStart--;
    const maxLen = Math.max(n + 1, rv.length, kv.length);
    const frames = d3.range(0, maxLen);
    const svg = d3.select("#lin-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 78, r: 10, t: 10, b: 6 });
    const cw = Math.min(40, Math.floor(F.iw / maxLen) - 3);
    function strip(y, label, vals, upTo, cur, opt) {
      const o = Object.assign({ above: null, below: null, mark: null }, opt || {});
      const shown = vals.map((v, i) => i <= upTo ? v : null);
      AL.row(F.g, shown, { x: 0, y, w: cw, h: 26, gap: 3, label, fontSize: 11, index: true,
        mark: (i, v) => v === null ? null : (o.mark && o.mark(i) ? AC.good : i === cur ? AC.a2 : AC.accent), text: AC.ink });
      if (o.above) vals.forEach((_, i) => F.g.append("text").attr("x", i * (cw + 3) + cw / 2).attr("y", y - 4).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", AC.muted).text(o.above(i)));
      if (o.below) vals.forEach((_, i) => { if (i <= upTo) F.g.append("text").attr("x", i * (cw + 3) + cw / 2).attr("y", y + 26 + 24).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", AC.good).text(o.below(i)); });
    }
    function render(i, k) {
      F.g.selectAll("*").remove();
      const last = k === frames.length - 1;
      strip(8, "Fibonacci F[i]", fib, Math.min(i, n), i);
      strip(66, "stairs W[i]", st, Math.min(i, n), i);
      strip(136, "robber B[i]", rb, Math.min(i, rv.length - 1), i, { above: j => "v=" + rv[j], mark: j => last && taken.includes(j) });
      strip(216, "Kadane E[i]", kd.end, Math.min(i, kv.length - 1), i, { above: j => "a=" + kv[j], below: j => "max " + kd.bestArr[j], mark: j => last && j >= kStart && j <= kEnd });
      F.g.append("text").attr("x", 0).attr("y", F.ih + 2).attr("font-size", 11).attr("fill", AC.muted)
        .text(last ? `done — robber: houses ${taken.join(", ")} taken (green); Kadane: the best run is a[${kStart}..${kEnd}] (green), found from the argmax of E and a walk back while E[i−1] > 0` : `writing cell ${i} of each strip that has one`);
    }
    AL.stepper(svg, { frames, render, delay: 420, label: "cell" });
    SX.setHtml("lin-readout",
      `Fibonacci: F[${n}] = <b>${fib[n]}</b> in <b>${cf.get("cell")}</b> cell updates (= n − 1 ${SX.flag(cf.get("cell") === n - 1)}); matrix power gives ${fm} in ${cmx.get("mult")} 2×2 products ${SX.flag(fm === fib[n])}, closed form gives ${fc} ${SX.flag(fc === fib[n])} · stairs: W[${n}] = <b>${st[n]}</b> = F[n+1] = ${fib[n] + (n >= 1 ? fib[n - 1] : 0)} ${SX.flag(st[n] === DR.fibIter(n + 1))}; enumerating every 1/2 sequence finds ${sb} ${SX.flag(sb === st[n])} · robber on [${rv.join(", ")}]: <b>${rb[rb.length - 1]}</b> = ${taken.map(j => rv[j]).join(" + ")} in ${cr.get("cell")} cells; brute force over the ${rbb.legal} adjacency-free subsets of ${rbb.total}: ${rbb.best} ${SX.flag(rbb.best === rb[rb.length - 1] && taken.reduce((s, j) => s + rv[j], 0) === rbb.best)} · Kadane on [${kv.join(", ")}]: <b>${kd.best}</b> = a[${kStart}..${kEnd}] in ${ck.get("cell")} cells; brute force over all ${kb.count} subarrays: ${kb.best} ${SX.flag(kb.best === kd.best && kv.slice(kStart, kEnd + 1).reduce((a, b) => a + b, 0) === kd.best)}`);
  }
  SX.on("lin-n", "input", build); SX.on("lin-seed", "change", build);
  build();
})();

/* ── 06  #coin-svg  coin change under three loop nests, stepped; counts checked by enumeration ── */
(function () {
  if (!SX.has("coin-svg")) return;
  const W = 680, H = 260;
  function build() {
    const coins = DI.coinSets[SX.val("cc-set", "1,2,5")] || [1, 2, 5], A = +SX.val("cc-amt", 5), prob = SX.val("cc-prob", "comb");
    const c1 = AL.counter(), c2 = AL.counter(), c3 = AL.counter(); const fr = { comb: [], comp: [], min: [] };
    const comb = DR.coinComb(coins, A, c1, f => fr.comb.push(f));
    const comp = DR.coinComp(coins, A, c2, f => fr.comp.push(f));
    const mn = DR.coinMin(coins, A, c3, f => fr.min.push(f));
    const bb = DR.coinCombBrute(coins, A), bc = DR.coinCompBrute(coins, A);
    const greedy = DR.coinGreedy(coins, A), recon = DR.coinMinRecon(mn.from, A);
    const frames = [{ outer: null, inner: null, dp: prob === "min" ? [0].concat(new Array(A).fill(Infinity)) : [1].concat(new Array(A).fill(0)), init: true }].concat(fr[prob]);
    const finals = [{ label: "(i) combinations", d: comb, live: prob === "comb" }, { label: "(ii) compositions", d: comp, live: prob === "comp" }, { label: "(iii) fewest coins", d: mn.d, live: prob === "min" }];
    const svg = d3.select("#coin-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 118, r: 10, t: 10, b: 6 });
    const cw = Math.min(40, Math.floor(F.iw / (A + 1)) - 3);
    function render(f) {
      F.g.selectAll("*").remove();
      const a = prob === "comb" ? f.inner : f.outer, k = prob === "comb" ? f.outer : f.inner;
      AL.row(F.g, f.dp.map(SX.inf), { x: 0, y: 8, w: cw, h: 28, gap: 3, label: "live d[a]", fontSize: 12, index: true,
        mark: i => f.init ? null : i === a ? AC.a2 : i === a - k ? AC.good : null });
      const msg = f.init ? `base: d[0] = ${prob === "min" ? "0" : "1"}${prob === "min" ? ", others ∞" : " (the empty " + (prob === "comb" ? "multiset" : "sequence") + "), others 0"}`
        : prob === "comb" ? `outer coin k = ${k}, inner amount a = ${a}:   d[${a}] += d[${a - k}]   → ${f.dp[a]}     (multisets from {${coins.filter(x => x <= k).join(", ")}} summing to ${a})`
        : prob === "comp" ? `outer amount a = ${a}, inner coin k = ${k}:   d[${a}] += d[${a - k}]   → ${f.dp[a]}     ("the last coin is ${k}")`
        : `amount a = ${a}, coin k = ${k}:   d[${a}] = min(d[${a}], 1 + d[${a - k}])   → ${SX.inf(f.dp[a])}`;
      F.g.append("text").attr("x", 0).attr("y", 66).attr("font-size", 11).attr("fill", AC.muted).text(msg);
      finals.forEach((row, r) => { const y = 92 + r * 44;
        AL.row(F.g, row.d.map(SX.inf), { x: 0, y, w: cw, h: 24, gap: 3, label: row.label, fontSize: 11, index: false, mark: i => i === A ? (row.live ? AC.accent : AC.panel2) : null, stroke: row.live ? AC.accent : AC.line }); });
      F.g.append("text").attr("x", 0).attr("y", F.ih + 2).attr("font-size", 10).attr("fill", AC.muted).text(`finished rows for coins {${coins.join(", ")}}: the same ${c1.get("cell")} updates under nest (i) and nest (ii) give ${comb[A]} and ${comp[A]} at a = ${A}`);
    }
    AL.stepper(svg, { frames, render, delay: 380, label: "update" });
    const N = mn.d[A];
    SX.setHtml("coin-readout",
      `coins {${coins.join(", ")}}, A = ${A} · (i) combinations <b>${comb[A]}</b> in ${c1.get("cell")} updates; enumerating every multiset: ${bb.count} ${SX.flag(bb.count === comb[A])} · (ii) compositions <b>${comp[A]}</b> in ${c2.get("cell")} updates; enumerating every sequence: ${bc} ${SX.flag(bc === comp[A])} · both nests did the same number of updates ${SX.flag(c1.get("cell") === c2.get("cell"))}${comb[A] !== comp[A] ? " and gave different counts — the loop-order bug in one line" : " (the counts coincide on this instance; pick an amount two coins both fit)"} · (iii) fewest coins <b>${SX.inf(N)}</b>${recon ? " = " + recon.join(" + ") : ""} in ${c3.get("cell")} updates; the smallest multiset enumerated has ${SX.inf(bb.fewest)} coin${SX.inf(bb.fewest) === 1 ? "" : "s"} ${SX.flag(bb.fewest === N)} · largest-coin-first: ${greedy ? greedy.join(" + ") + " = " + greedy.length + " coin" + (greedy.length === 1 ? "" : "s") : "stranded (no representation found)"}${greedy && greedy.length === N ? " — matches the DP" : greedy ? " — <b>worse than the DP</b>: this coin system is not canonical at this amount" : N === Infinity ? " — and the DP confirms the amount is unreachable" : " — but the DP finds a representation"}`);
  }
  SX.on("cc-set", "change", build); SX.on("cc-amt", "input", build); SX.on("cc-prob", "change", build);
  build();
})();

/* ── chunk B figures ── */
/* ── 07  #knap-svg  0/1 knapsack table filled row by row, reconstruction walked back ── */
(function () {
  if (!SX.has("knap-svg")) return;
  const W = 680, H = 300;
  function instance(which) {
    if (which === "greedy") return { items: DI.knapGreedyItems, W: DI.knapGreedyW };
    if (which === "greedy5") return { items: DI.knapGreedyItems.map(it => ({ id: it.id, w: it.w / 5, v: it.v })), W: 10 };
    if (which === "rand") { const r = AL.rng(31); return { items: Array.from({ length: 5 }, (_, i) => ({ id: String(i + 1), w: 1 + Math.floor(r() * 6), v: 5 + Math.floor(r() * 30) })), W: 12 }; }
    return { items: DI.knapSmall, W: DI.knapSmallW };
  }
  function build() {
    const inst = instance(SX.val("knap-inst", "small")); const items = inst.items, W0 = inst.W, n = items.length;
    const c = AL.counter(); const rows = [];
    const res = DR.knap01(items, W0, c, e => rows.push(e));
    const brute = DR.knapBrute(items, W0);
    /* frames: one per row filled, then one per reconstruction step */
    const frames = [];
    rows.forEach(e => frames.push({ kind: "fill", i: e.i }));
    for (let k = 0; k < res.path.length - 1; k++) frames.push({ kind: "recon", k });
    const svg = d3.select("#knap-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 92, r: 12, t: 26, b: 14 });
    const gap = 2, cw = Math.min(30, (F.iw - gap * W0) / (W0 + 1)), ch = Math.min(26, (F.ih - gap * n) / (n + 1));
    const vmax = Math.max(1, res.val);
    function render(f, k) {
      F.g.selectAll("*").remove();
      const filled = f.kind === "fill" ? f.i : n, cur = f.kind === "fill" ? f.i : -1;
      const reconSteps = f.kind === "recon" ? f.k + 1 : 0;
      const onPath = new Map(); for (let s = 0; s < reconSteps; s++) { const [i, w] = res.path[s]; onPath.set(i + "," + w, res.keep[i][w] ? "take" : "skip"); }
      if (reconSteps > 0) { const [i, w] = res.path[reconSteps]; onPath.set(i + "," + w, "end"); }
      const G = SX.grid(F.g, n + 1, W0 + 1, { x: 0, y: 0, w: cw, h: ch, gap, fontSize: cw >= 24 ? 11 : 9,
        text: (i, j) => (cw >= 18 && i <= filled) ? res.dp[i][j] : "",
        fill: (i, j) => { const p = onPath.get(i + "," + j); if (p === "take") return AC.good; if (p === "skip") return AC.violet; if (p === "end") return AC.teal; if (i === cur) return AC.a2; if (i > filled) return null; if (cw < 18) return d3.interpolateRgb(AC.panel2, AC.accent)(0.15 + 0.85 * res.dp[i][j] / vmax); return null; },
        rowLabel: i => i === 0 ? "i = 0" : `${i}: (${items[i - 1].w}, ${items[i - 1].v})`,
        colLabel: j => (cw >= 18 || j % 5 === 0) ? j : "" });
      /* arrows along the reconstruction path */
      for (let s = 0; s < reconSteps; s++) { const [i, w] = res.path[s], [i2, w2] = res.path[s + 1]; AL.arrow(G.g, G.cx(w), G.cy(i) - ch / 2 + 2, G.cx(w2), G.cy(i2) + ch / 2 - 2, { color: res.keep[i][w] ? AC.good : AC.violet, w: 2 }); }
      F.g.append("text").attr("x", -86).attr("y", -12).attr("font-size", 10).attr("fill", AC.muted).text("w →");
      const msg = f.kind === "fill" ? `row ${f.i}: item ${items[f.i - 1].id} (w = ${items[f.i - 1].w}, v = ${items[f.i - 1].v}) — dp[${f.i}][w] = max(dp[${f.i - 1}][w], dp[${f.i - 1}][w − ${items[f.i - 1].w}] + ${items[f.i - 1].v})`
        : (() => { const [i, w] = res.path[f.k]; return res.keep[i][w] ? `walk-back at (${i}, ${w}): ${res.dp[i][w]} ≠ dp[${i - 1}][${w}] = ${res.dp[i - 1][w]} → item ${items[i - 1].id} TAKEN, jump to (${i - 1}, ${w - items[i - 1].w})` : `walk-back at (${i}, ${w}): ${res.dp[i][w]} = dp[${i - 1}][${w}] → item ${items[i - 1].id} skipped, up to (${i - 1}, ${w})`; })();
      F.g.append("text").attr("x", F.iw).attr("y", -12).attr("text-anchor", "end").attr("font-size", 10.5).attr("fill", AC.muted).text(msg);
    }
    AL.stepper(svg, { frames, render, delay: 900, label: "step" });
    SX.setHtml("knap-readout", `${n} items, W = ${W0}: dp[${n}][${W0}] = <b>${res.val}</b>, chosen {${res.take.join(", ")}} (weight ${res.take.reduce((s, id) => s + items.find(it => it.id === id).w, 0)}) · measured <b>${SX.int(c.get("cell"))}</b> cells filled = n·(W + 1) = ${n}·${W0 + 1} = ${SX.int(n * (W0 + 1))} ${SX.flag(c.get("cell") === n * (W0 + 1))} · walk-back <b>${c.get("recon")}</b> steps = n ${SX.flag(c.get("recon") === n)} · brute force over all 2^${n} = ${brute.count} subsets: <b>${brute.val}</b> {${brute.take.join(", ")}} ${SX.flag(brute.val === res.val)}`);
  }
  SX.on("knap-inst", "change", build);
  build();
})();

/* ── 08  #knap1d-svg  the rolling 1-D knapsack array, direction switchable ── */
(function () {
  if (!SX.has("knap1d-svg")) return;
  const W = 680, H = 260;
  function instance(which) {
    if (which === "greedy5") return { items: DI.knapGreedyItems.map(it => ({ id: it.id, w: it.w / 5, v: it.v })), W: 10 };
    if (which === "w7") return { items: DI.knapSmall, W: 7 };
    if (which === "rand") { const r = AL.rng(37); return { items: Array.from({ length: 5 }, (_, i) => ({ id: String(i + 1), w: 1 + Math.floor(r() * 6), v: 5 + Math.floor(r() * 30) })), W: 12 }; }
    return { items: DI.knapSmall, W: DI.knapSmallW };
  }
  function build() {
    const inst = instance(SX.val("k1-inst", "small")); const items = inst.items, W0 = inst.W, n = items.length;
    const dir = SX.val("k1-dir", "down") === "up" ? "up" : "down";
    const c = AL.counter(); const frames = [];
    const res = DR.knap1d(items, W0, dir, c, e => frames.push(e));
    const other = DR.knap1d(items, W0, dir === "down" ? "up" : "down");
    const b01 = DR.knapBrute(items, W0), bUnb = DR.knapUnboundedBrute(items, W0);
    const down = dir === "down" ? res.val : other.val, up = dir === "down" ? other.val : res.val;
    const svg = d3.select("#knap1d-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 16, r: 16, t: 14, b: 14 });
    const cw = Math.min(40, (F.iw - 3 * W0) / (W0 + 1));
    function render(f, k) {
      F.g.selectAll("*").remove();
      /* the item strip */
      items.forEach((it, i) => { const x = i * (F.iw / n); const cur = i === f.i; F.g.append("rect").attr("x", x).attr("y", 0).attr("width", F.iw / n - 6).attr("height", 26).attr("rx", 4).attr("fill", cur ? AC.a2 : i < f.i ? AC.panel2 : AC.panel).attr("stroke", AC.line); F.g.append("text").attr("x", x + 8).attr("y", 17).attr("font-size", 11).attr("fill", cur ? AC.bg : AC.ink).text(`item ${it.id}: w = ${it.w}, v = ${it.v}${i < f.i ? " · done" : cur ? " · in progress" : ""}`); });
      /* which cells were rewritten in this item's pass so far */
      const it = items[f.i]; const touched = new Set();
      if (dir === "down") { for (let w = W0; w > f.w; w--) touched.add(w); } else { for (let w = it.w; w < f.w; w++) touched.add(w); }
      const R = AL.row(F.g, f.arr, { x: 0, y: 70, w: cw, h: 32, gap: 3, fontSize: cw >= 30 ? 12 : 10, label: "f[w]",
        mark: (i) => i === f.w ? AC.a2 : i === f.w - it.w ? AC.teal : touched.has(i) ? AC.good : null });
      /* the read arrow, and the sweep direction */
      AL.arrow(F.g, R.cellX(f.w - it.w), 70 + 32 + 22, R.cellX(f.w), 70 + 32 + 6, { color: AC.teal, w: 1.5 });
      F.g.append("text").attr("x", R.cellX(f.w - it.w)).attr("y", 70 + 32 + 36).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.teal).text(`read f[${f.w - it.w}] = ${f.arr[f.w - it.w]}${touched.has(f.w - it.w) ? " — already rewritten by item " + it.id + " (reuse!)" : ""}`);
      const y2 = 70 + 32 + 60;
      if (dir === "down") AL.arrow(F.g, R.cellX(W0), y2, R.cellX(it.w), y2, { color: AC.muted, w: 1.5 }); else AL.arrow(F.g, R.cellX(it.w), y2, R.cellX(W0), y2, { color: AC.muted, w: 1.5 });
      F.g.append("text").attr("x", 0).attr("y", y2 + 16).attr("font-size", 10.5).attr("fill", AC.muted).text(dir === "down" ? `w runs ${W0} → ${it.w}: f[w − ${it.w}] is still ROWNAME row when read — item ${it.id} enters at most once`.replace("ROWNAME", f.i === 0 ? "the all-zero" : "item " + items[f.i - 1].id + "'s") : `w runs ${it.w} → ${W0}: f[w − ${it.w}] may already hold item ${it.id} — item ${it.id} can be stacked`);
      F.g.append("text").attr("x", F.iw).attr("y", y2 + 16).attr("text-anchor", "end").attr("font-size", 10.5).attr("fill", AC.muted).text(`f[${f.w}] ← max(${f.old}, ${f.arr[f.w - it.w]} + ${it.v}) = ${f.arr[f.w]}${f.changed ? " (updated)" : " (unchanged)"}`);
    }
    AL.stepper(svg, { frames, render, delay: 500, label: "update" });
    const upIsUnb = up === bUnb.val, downIs01 = down === b01.val;
    SX.setHtml("knap1d-readout", `this pass (${dir === "down" ? "downward" : "upward"}): f[${W0}] = <b>${res.val}</b> · measured <b>${c.get("cell")}</b> updates = ∑ᵢ (W − wᵢ + 1) = ${items.reduce((s, it) => s + Math.max(0, W0 - it.w + 1), 0)} ${SX.flag(c.get("cell") === items.reduce((s, it) => s + Math.max(0, W0 - it.w + 1), 0))} · <b>downward</b> answer ${down}: 0/1 brute force over 2^${n} = ${b01.count} subsets gives ${b01.val} {${b01.take.join(", ")}} ${SX.flag(downIs01)} · <b>upward</b> answer ${up}: unbounded brute force over ${bUnb.count} multisets gives ${bUnb.val} {${bUnb.take.join(", ")}} ${SX.flag(upIsUnb)} · ${up === down ? "the two directions coincide on this instance (no item is worth repeating)" : "the upward pass is <b>not</b> the 0/1 answer: it silently solved the unbounded problem"}`);
  }
  SX.on("k1-inst", "change", build); SX.on("k1-dir", "change", build);
  build();
})();

/* ── 09  #wis-svg  weighted interval scheduling on a timeline ─────────── */
(function () {
  if (!SX.has("wis-svg")) return;
  const W = 680, H = 330;
  function instance(which) {
    if (which === "unit") return DI.wisIntervals.map(a => ({ id: a.id, s: a.s, f: a.f, v: 1 }));
    if (which === "rand") { const r = AL.rng(41); return Array.from({ length: 10 }, () => { const s = Math.floor(r() * 30); return { s, f: s + 2 + Math.floor(r() * 9), v: 1 + Math.floor(r() * 9) }; }).sort((a, b) => a.f - b.f || a.s - b.s).map((a, i) => ({ id: i + 1, s: a.s, f: a.f, v: a.v })); }
    return DI.wisIntervals;
  }
  function build() {
    const ivs = instance(SX.val("wis-inst", "book")); const n = ivs.length;
    const c = AL.counter(); const fills = [];
    const res = DR.wisSolve(ivs, c, e => fills.push(e));
    const brute = DR.wisBrute(ivs), gEF = DR.wisGreedyEF(ivs);
    const frames = fills.map(e => ({ kind: "fill", j: e.j }));
    { let j = n; while (j > 0) { frames.push({ kind: "recon", j, take: res.choice[j] }); j = res.choice[j] ? res.p[j] : j - 1; } frames.push({ kind: "done" }); }
    const order = res.order, tmax = Math.max(...order.map(a => a.f));
    const svg = d3.select("#wis-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 30, r: 190, t: 12, b: 34 });
    const x = d3.scaleLinear().domain([0, tmax]).range([0, F.iw]);
    const rowH = Math.min(26, (F.ih - 8) / n);
    const yOf = j => (j - 1) * rowH;                     // 1-based j
    function render(f, k) {
      F.g.selectAll("*").remove();
      AL.gridX(F.g, x, n * rowH, 8); AL.axisB(F.g, x, n * rowH + 4, 8, "time");
      const filled = f.kind === "fill" ? f.j : n;
      const status = new Map(); const seen = frames.slice(0, k + 1).filter(fr => fr.kind === "recon"); seen.forEach(fr => status.set(fr.j, fr.take ? "take" : "skip"));
      /* p(j) hooks */
      order.forEach((a, idx) => { const j = idx + 1; if (res.p[j] > 0 && j <= filled) { const q = order[res.p[j] - 1]; AL.arrow(F.g, x(a.s) - 1, yOf(j) + rowH / 2, x(q.f) + 1, yOf(res.p[j]) + rowH / 2, { color: f.kind === "fill" && f.j === j ? AC.a2 : AC.muted, w: 1, dash: "3 2", head: 4 }); } });
      order.forEach((a, idx) => { const j = idx + 1; const cur = f.kind === "fill" && f.j === j; const st = status.get(j);
        const col = cur ? AC.a2 : st === "take" ? AC.good : st === "skip" ? AC.violet : j <= filled ? AC.panel2 : AC.panel;
        F.g.append("rect").attr("x", x(a.s)).attr("y", yOf(j) + 3).attr("width", Math.max(2, x(a.f) - x(a.s))).attr("height", rowH - 6).attr("rx", 3).attr("fill", col).attr("stroke", AC.line).attr("opacity", st === "skip" ? 0.7 : 1);
        F.g.append("text").attr("x", x(a.s) + 4).attr("y", yOf(j) + rowH / 2 + 4).attr("font-size", 10).attr("fill", cur || st === "take" ? AC.bg : AC.ink).text(`${j}: v=${a.v}`);
        F.g.append("text").attr("x", -6).attr("y", yOf(j) + rowH / 2 + 4).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.muted).text(`p=${res.p[j]}`); });
      /* the OPT column on the right */
      const cx = F.iw + 14;
      F.g.append("text").attr("x", cx).attr("y", -2).attr("font-size", 10).attr("fill", AC.muted).text("OPT(j) = max(OPT(j−1), vⱼ + OPT(p(j)))");
      for (let j = 1; j <= n; j++) { const cur = f.kind === "fill" && f.j === j; const st = status.get(j);
        F.g.append("rect").attr("x", cx).attr("y", yOf(j) + 3).attr("width", 164).attr("height", rowH - 6).attr("rx", 3).attr("fill", cur ? AC.a2 : st === "take" ? AC.good : j <= filled ? AC.panel2 : AC.panel).attr("stroke", AC.line);
        if (j <= filled) { const a = order[j - 1]; F.g.append("text").attr("x", cx + 4).attr("y", yOf(j) + rowH / 2 + 4).attr("font-size", 10).attr("fill", cur || st === "take" ? AC.bg : AC.ink).text(`max(${res.opt[j - 1]}, ${a.v}+${res.opt[res.p[j]]}) = ${res.opt[j]}`); } }
      const msg = f.kind === "fill" ? `OPT(${f.j}) = ${res.opt[f.j]}` : f.kind === "recon" ? (f.take ? `walk-back at j = ${f.j}: take (${order[f.j - 1].v} + OPT(${res.p[f.j]}) = ${order[f.j - 1].v + res.opt[res.p[f.j]]} > OPT(${f.j - 1}) = ${res.opt[f.j - 1]}) → jump to p(${f.j}) = ${res.p[f.j]}` : `walk-back at j = ${f.j}: skip (${order[f.j - 1].v} + OPT(${res.p[f.j]}) = ${order[f.j - 1].v + res.opt[res.p[f.j]]} ≤ OPT(${f.j - 1}) = ${res.opt[f.j - 1]}) → j = ${f.j - 1}`) : `chosen {${res.take.join(", ")}}, weight ${res.val}`;
      F.g.append("text").attr("x", 0).attr("y", F.ih + 30).attr("font-size", 10.5).attr("fill", AC.muted).text(msg);
    }
    AL.stepper(svg, { frames, render, delay: 800, label: "step" });
    const probeBound = order.reduce((s, _, idx) => s + Math.ceil(Math.log2(idx + 1)), 0);
    SX.setHtml("wis-readout", `${n} intervals: OPT(${n}) = <b>${res.val}</b>, chosen {${res.take.join(", ")}} · measured: <b>${c.get("probe")}</b> binary-search probes for p (bound ∑ⱼ⌈log₂ j⌉ = ${probeBound} ${SX.flag(c.get("probe") <= probeBound)}), <b>${c.get("cell")}</b> OPT cells = n ${SX.flag(c.get("cell") === n)}, <b>${c.get("recon")}</b> walk-back steps, ${c.get("sortcmp")} sort comparisons · brute force over 2^${n} = ${SX.int(brute.count)} subsets: <b>${brute.val}</b> {${brute.take.join(", ")}} ${SX.flag(brute.val === res.val)} · unweighted earliest-finish greedy on the same intervals: {${gEF.take.join(", ")}} weighing ${gEF.val} ${gEF.val === res.val ? "— optimal here" : "— <b>not</b> optimal"}`);
  }
  SX.on("wis-inst", "change", build);
  build();
})();

/* ── 10  #lcs-svg  LCS table stepped, traceback lit ───────────────────── */
(function () {
  if (!SX.has("lcs-svg")) return;
  const W = 680, H = 330;
  const PAIRS = { book: [DI.lcsX, DI.lcsY], dna: ["ACCGGTCGAGTG", "GTCGTTCGGAATGC"], none: ["ABCDE", "FGHIJ"], same: ["BANANA", "BANANA"] };
  function build() {
    const [X, Y] = PAIRS[SX.val("lcs-pair", "book")] || PAIRS.book; const pref = SX.val("lcs-tie", "up") === "left" ? "left" : "up";
    const m = X.length, n = Y.length;
    const c = AL.counter(); const fills = [];
    const t = DR.lcsTable(X, Y, c, e => fills.push(e));
    const ct = AL.counter(); const tb = DR.lcsTraceback(X, Y, t, ct, pref);
    const other = DR.lcsTraceback(X, Y, t, null, pref === "up" ? "left" : "up");
    const cn = AL.counter(); DR.lcsNaive(X, Y, cn);
    const cs = AL.counter(); const sp = DR.lcsLenSpace(X, Y, cs);
    const brute = Math.min(m, n) <= 14 ? DR.lcsBrute(X, Y) : null;
    const all = (m * n <= 200) ? DR.lcsAll(X, Y, t) : null;
    const frames = fills.map(e => ({ kind: "fill", i: e.i, j: e.j })).concat(tb.path.map((p, k) => ({ kind: "trace", k })), [{ kind: "done" }]);
    const svg = d3.select("#lcs-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 40, r: 12, t: 30, b: 16 });
    const gap = 2, cw = Math.min(34, (F.iw - gap * n) / (n + 1)), ch = Math.min(26, (F.ih - gap * m) / (m + 1));
    function render(f, k) {
      F.g.selectAll("*").remove();
      const done = f.kind === "fill" ? fills.findIndex(e => e.i === f.i && e.j === f.j) : fills.length;   // cells filled so far (index)
      const filledSet = new Set(fills.slice(0, f.kind === "fill" ? done + 1 : fills.length).map(e => e.i + "," + e.j));
      const steps = f.kind === "trace" ? f.k + 1 : f.kind === "done" ? tb.path.length : 0;
      const onPath = new Map(); for (let s = 0; s < steps; s++) { const [i, j] = tb.path[s]; onPath.set(i + "," + j, X[i - 1] === Y[j - 1] ? "match" : "move"); }
      const cur = f.kind === "fill" ? f.i + "," + f.j : null;
      const G = SX.grid(F.g, m + 1, n + 1, { x: 0, y: 0, w: cw, h: ch, gap, fontSize: cw >= 26 ? 11 : 9,
        text: (i, j) => (i === 0 || j === 0) ? "0" : filledSet.has(i + "," + j) ? (cw >= 26 ? t.L[i][j] + " " + t.dirs[i][j] : t.L[i][j]) : "",
        fill: (i, j) => { const key = i + "," + j; if (key === cur) return AC.a2; const p = onPath.get(key); if (p === "match") return AC.good; if (p === "move") return AC.violet; return null; },
        rowLabel: i => i === 0 ? "i=0" : `${i} ${X[i - 1]}`, colLabel: j => j === 0 ? "j=0" : `${Y[j - 1]}` });
      if (f.kind === "fill") { [[f.i - 1, f.j - 1], [f.i - 1, f.j], [f.i, f.j - 1]].forEach(([i, j]) => G.g.append("rect").attr("x", G.cx(j) - cw / 2).attr("y", G.cy(i) - ch / 2).attr("width", cw).attr("height", ch).attr("rx", 3).attr("fill", "none").attr("stroke", AC.teal).attr("stroke-width", 1.5)); }
      const emitted = tb.path.slice(0, steps).filter(([i, j]) => X[i - 1] === Y[j - 1]).map(([i]) => X[i - 1]).reverse().join("");
      const msg = f.kind === "fill" ? (X[f.i - 1] === Y[f.j - 1] ? `L[${f.i}][${f.j}]: ${X[f.i - 1]} = ${Y[f.j - 1]} → L[${f.i - 1}][${f.j - 1}] + 1 = ${t.L[f.i][f.j]}` : `L[${f.i}][${f.j}]: ${X[f.i - 1]} ≠ ${Y[f.j - 1]} → max(L[${f.i - 1}][${f.j}] = ${t.L[f.i - 1][f.j]}, L[${f.i}][${f.j - 1}] = ${t.L[f.i][f.j - 1]}) = ${t.L[f.i][f.j]}`)
        : f.kind === "trace" ? (() => { const [i, j] = tb.path[f.k]; return X[i - 1] === Y[j - 1] ? `traceback at (${i}, ${j}): ${X[i - 1]} = ${Y[j - 1]} → emit ${X[i - 1]}, go diagonal · so far "${emitted}"` : `traceback at (${i}, ${j}): ${X[i - 1]} ≠ ${Y[j - 1]} → ${(f.k + 1 < tb.path.length && tb.path[f.k + 1][0] === i - 1) ? "up" : "left"} (tie rule: prefer ${pref})`; })()
        : `LCS = "${tb.s}" (length ${t.len})`;
      F.g.append("text").attr("x", 0).attr("y", -18).attr("font-size", 10.5).attr("fill", AC.muted).text(msg);
    }
    AL.stepper(svg, { frames, render, delay: 350, label: "step" });
    SX.setHtml("lcs-readout", `|X| = ${m}, |Y| = ${n}: L[${m}][${n}] = <b>${t.len}</b>, traceback (prefer ${pref}) → <b>"${tb.s}"</b> in ${ct.get("trace")} steps; the other tie rule gives "${other.s}"${other.s !== tb.s ? " — a <b>different</b> LCS of the same length" : " — the same string here"}${all ? ` · all distinct LCSs (every optimal traceback): {${all.map(s => '"' + s + '"').join(", ")}}` : ""} · measured <b>${c.get("cell")}</b> cells = mn = ${m * n} ${SX.flag(c.get("cell") === m * n)} · naive recursion: <b>${SX.int(cn.get("call"))}</b> calls · two-row routine: length ${sp.len} ${SX.flag(sp.len === t.len)} in ${sp.words} words${brute ? ` · brute force over all 2^${Math.min(m, n)} = ${SX.int(brute.count)} subsequences of the shorter string: length <b>${brute.len}</b> ("${brute.s}") ${SX.flag(brute.len === t.len)}` : ""}`);
  }
  SX.on("lcs-pair", "change", build); SX.on("lcs-tie", "change", build);
  build();
})();

/* ── 11  #ed-svg  edit-distance table with weighted operations, alignment drawn ── */
(function () {
  if (!SX.has("ed-svg")) return;
  const W = 680, H = 360;
  const PAIRS = { kitten: [DI.edA, DI.edB], exp: ["EXPONENTIAL", "POLYNOMIAL"], sunday: ["sunday", "saturday"], abc: ["abc", "yabd"] };
  const OPCOL = { match: AC.good, sub: AC.a2, ins: AC.rose, del: AC.violet };
  function build() {
    const [a, b] = PAIRS[SX.val("ed-pair", "kitten")] || PAIRS.kitten; const m = a.length, n = b.length;
    const K = { ins: +SX.val("ed-ins", 1) || 1, del: +SX.val("ed-del", 1) || 1, sub: +SX.val("ed-sub", 1) || 1 };
    const c = AL.counter(); const fills = [];
    const t = DR.edTable(a, b, K, c, e => fills.push(e));
    const steps = DR.edAlign(a, b, t); const alignCost = DR.edAlignCost(steps, K);
    const cd = AL.counter(); const dj = DR.edDijkstra(a, b, K, cd);
    const unit = K.ins === 1 && K.del === 1 && K.sub === 1;
    let naive = null, cn = null; if (m * n <= 56) { cn = AL.counter(); naive = DR.edNaive(a, b, K, cn); }
    const bfs = (unit && t.dist <= 4 && m + n <= 14) ? DR.edBFS(a, b, t.dist) : null;
    /* the walk-back path, from (m, n) backwards */
    const path = []; { let i = m, j = n; for (let s = steps.length - 1; s >= 0; s--) { path.push([i, j, steps[s].op]); if (steps[s].op === "ins") j--; else if (steps[s].op === "del") i--; else { i--; j--; } } }
    const frames = fills.map(e => ({ kind: "fill", i: e.i, j: e.j })).concat(path.map((p, k) => ({ kind: "walk", k })), [{ kind: "done" }]);
    const svg = d3.select("#ed-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 40, r: 12, t: 30, b: 10 });
    const gap = 2, tableH = F.ih - 70;
    const cw = Math.min(34, (F.iw - gap * n) / (n + 1)), ch = Math.min(24, (tableH - gap * m) / (m + 1));
    function render(f, k) {
      F.g.selectAll("*").remove();
      const fi = f.kind === "fill" ? fills.findIndex(e => e.i === f.i && e.j === f.j) + 1 : fills.length;
      const filled = new Set(fills.slice(0, fi).map(e => e.i + "," + e.j));
      const walked = f.kind === "walk" ? f.k + 1 : f.kind === "done" ? path.length : 0;
      const onPath = new Map(); for (let s = 0; s < walked; s++) onPath.set(path[s][0] + "," + path[s][1], path[s][2]);
      if (walked === path.length && walked > 0) onPath.set("0,0", "match");
      const cur = f.kind === "fill" ? f.i + "," + f.j : null;
      const G = SX.grid(F.g, m + 1, n + 1, { x: 0, y: 0, w: cw, h: ch, gap, fontSize: cw >= 26 ? 11 : 9,
        text: (i, j) => (i === 0 || j === 0 || filled.has(i + "," + j)) ? t.D[i][j] : "",
        fill: (i, j) => { const key = i + "," + j; if (key === cur) return AC.a2; const o = onPath.get(key); return o ? OPCOL[o] : null; },
        rowLabel: i => i === 0 ? "i=0" : `${i} ${a[i - 1]}`, colLabel: j => j === 0 ? "j=0" : b[j - 1] });
      if (f.kind === "fill") [[f.i - 1, f.j - 1], [f.i - 1, f.j], [f.i, f.j - 1]].forEach(([i, j]) => G.g.append("rect").attr("x", G.cx(j) - cw / 2).attr("y", G.cy(i) - ch / 2).attr("width", cw).attr("height", ch).attr("rx", 3).attr("fill", "none").attr("stroke", AC.teal).attr("stroke-width", 1.5));
      /* the alignment, revealed from the right as the walk-back proceeds */
      const shown = steps.slice(steps.length - walked); const y0 = (m + 1) * (ch + gap) + 18; const bw = Math.min(26, F.iw / Math.max(steps.length, 1));
      F.g.append("text").attr("x", 0).attr("y", y0 - 4).attr("font-size", 10).attr("fill", AC.muted).text("alignment (read right to left as the walk-back reveals it):");
      shown.forEach((st, q) => { const x = (steps.length - shown.length + q) * bw;
        F.g.append("rect").attr("x", x).attr("y", y0).attr("width", bw - 3).attr("height", 40).attr("rx", 3).attr("fill", OPCOL[st.op]).attr("opacity", 0.85);
        F.g.append("text").attr("x", x + (bw - 3) / 2).attr("y", y0 + 15).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", AC.bg).text(st.x);
        F.g.append("text").attr("x", x + (bw - 3) / 2).attr("y", y0 + 33).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", AC.bg).text(st.y); });
      const msg = f.kind === "fill" ? (() => { const i = f.i, j = f.j, same = a[i - 1] === b[j - 1]; return `D[${i}][${j}] = min(${t.D[i - 1][j - 1]} + ${same ? "0 (" + a[i - 1] + " = " + b[j - 1] + ")" : K.sub + " (" + a[i - 1] + " → " + b[j - 1] + ")"}, ${t.D[i - 1][j]} + ${K.del} (delete ${a[i - 1]}), ${t.D[i][j - 1]} + ${K.ins} (insert ${b[j - 1]})) = ${t.D[i][j]}`; })()
        : f.kind === "walk" ? (() => { const [i, j, o] = path[f.k]; return `walk-back at (${i}, ${j}): ${o === "match" ? "match " + a[i - 1] : o === "sub" ? "substitute " + a[i - 1] + " → " + b[j - 1] : o === "ins" ? "insert " + b[j - 1] : "delete " + a[i - 1]} → ${o === "ins" ? `(${i}, ${j - 1})` : o === "del" ? `(${i - 1}, ${j})` : `(${i - 1}, ${j - 1})`}`; })()
        : `distance ${t.dist}; alignment cost re-summed from its columns = ${alignCost}`;
      F.g.append("text").attr("x", 0).attr("y", -18).attr("font-size", 10.5).attr("fill", AC.muted).text(msg);
    }
    AL.stepper(svg, { frames, render, delay: 350, label: "step" });
    SX.setHtml("ed-readout", `${a} → ${b} with costs (insert ${K.ins}, delete ${K.del}, substitute ${K.sub}): D[${m}][${n}] = <b>${t.dist}</b> · measured <b>${c.get("cell")}</b> cells = mn = ${m * n} ${SX.flag(c.get("cell") === m * n)} · alignment of ${steps.length} columns re-summed: ${alignCost} ${SX.flag(alignCost === t.dist)} · Dijkstra on the edit graph (${(m + 1) * (n + 1)} vertices, ${cd.get("relax")} relaxations): <b>${dj}</b> ${SX.flag(dj === t.dist)}${naive !== null ? ` · naive recursion, no table: <b>${naive}</b> after ${SX.int(cn.get("call"))} calls ${SX.flag(naive === t.dist)}` : ""}${bfs ? ` · BFS over single-character edits: distance <b>${bfs.dist}</b> after expanding ${SX.int(bfs.expanded)} strings ${SX.flag(bfs.dist === t.dist)}` : unit ? "" : " · (BFS applies to unit costs only)"}${K.sub >= K.ins + K.del ? " · substitute ≥ insert + delete, so no substitution appears in the alignment" : ""}`);
  }
  ["ed-pair", "ed-ins", "ed-del", "ed-sub"].forEach(id => SX.on(id, id === "ed-pair" ? "change" : "input", build));
  build();
})();

/* ── 12  #hirsch-svg  Hirschberg's recursion drawn on the edit grid ──── */
(function () {
  if (!SX.has("hirsch-svg")) return;
  const W = 680, H = 340;
  const PAIRS = { kitten: [DI.edA, DI.edB], exp: ["EXPONENTIAL", "POLYNOMIAL"], sunday: ["sunday", "saturday"], pow: ["abcdefgh", "abdcefhg"] };
  function bound(m, n) { let s = 0; for (let k = 0; ; k++) { const t = Math.ceil(m / Math.pow(2, k)); s += t; if (t <= 1) break; } return s * n; }
  function pathOf(steps) { const pts = [[0, 0]]; let i = 0, j = 0; steps.forEach(s => { if (s.op === "ins") j++; else if (s.op === "del") i++; else { i++; j++; } pts.push([i, j]); }); return pts; }
  function build() {
    const [a, b] = PAIRS[SX.val("hi-pair", "kitten")] || PAIRS.kitten; const m = a.length, n = b.length;
    const c = AL.counter(); const splits = [];
    const h = DR.hirschberg(a, b, null, c, e => splits.push(e));
    const cf = AL.counter(); const full = DR.edTable(a, b, null, cf); const fullSteps = DR.edAlign(a, b, full);
    const hPath = pathOf(h.steps), fPath = pathOf(fullSteps);
    /* frames: one per split (in the order the recursion made them), then the finished alignment */
    const frames = splits.map((s, k) => ({ kind: "split", k })).concat([{ kind: "done" }]);
    const svg = d3.select("#hirsch-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 40, r: 12, t: 30, b: 12 });
    const cw = Math.min(40, F.iw / (n + 1)), ch = Math.min(28, (F.ih - 8) / (m + 1));
    const X = j => j * cw, Y = i => i * ch;
    function render(f, k) {
      F.g.selectAll("*").remove();
      /* the grid of vertices */
      for (let i = 0; i <= m; i++) for (let j = 0; j <= n; j++) F.g.append("circle").attr("cx", X(j)).attr("cy", Y(i)).attr("r", 2).attr("fill", AC.line);
      for (let i = 1; i <= m; i++) F.g.append("text").attr("x", -10).attr("y", Y(i) + 4).attr("text-anchor", "end").attr("font-size", 11).attr("fill", AC.muted).text(a[i - 1]);
      for (let j = 1; j <= n; j++) F.g.append("text").attr("x", X(j)).attr("y", -8).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.muted).text(b[j - 1]);
      /* the full-table path, dashed */
      F.g.append("path").attr("d", d3.line()(fPath.map(([i, j]) => [X(j), Y(i)]))).attr("fill", "none").attr("stroke", AC.muted).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.5);
      const upto = f.kind === "split" ? f.k + 1 : splits.length;
      /* earlier splits: faint rectangles and their chosen crossing points */
      splits.slice(0, upto).forEach((s, q) => { const cur = f.kind === "split" && q === f.k;
        const x0 = X(s.j0), y0 = Y(s.i0), x1 = X(s.j0 + s.b.length), y1 = Y(s.i0 + s.a.length);
        F.g.append("rect").attr("x", x0 - 4).attr("y", y0 - 4).attr("width", x1 - x0 + 8).attr("height", y1 - y0 + 8).attr("rx", 4).attr("fill", cur ? AC.a2 : "none").attr("fill-opacity", cur ? 0.12 : 0).attr("stroke", cur ? AC.a2 : AC.line).attr("stroke-width", cur ? 1.5 : 1);
        const ym = Y(s.i0 + s.mid);
        if (cur) { F.g.append("line").attr("x1", x0).attr("x2", x1).attr("y1", ym).attr("y2", ym).attr("stroke", AC.a2).attr("stroke-width", 1);
          for (let j = 0; j <= s.b.length; j++) { const v = s.fwd[j] + s.bwd[s.b.length - j]; F.g.append("text").attr("x", X(s.j0 + j)).attr("y", ym - 6).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", j === s.split ? AC.good : AC.a2).attr("font-weight", j === s.split ? "bold" : null).text(v); } }
        F.g.append("circle").attr("cx", X(s.j0 + s.split)).attr("cy", ym).attr("r", cur ? 6 : 4).attr("fill", AC.good).attr("stroke", AC.bg); });
      /* the alignment path once done */
      if (f.kind === "done") F.g.append("path").attr("d", d3.line()(hPath.map(([i, j]) => [X(j), Y(i)]))).attr("fill", "none").attr("stroke", AC.good).attr("stroke-width", 2.5);
      const msg = f.kind === "split" ? (() => { const s = splits[f.k]; return `split "${s.a}" | "${s.b}": mid = ${s.mid} ("${s.a.slice(0, s.mid)}" + "${s.a.slice(s.mid)}"), min F[j] + R[j] = ${s.cost} at j = ${s.split} → ("${s.a.slice(0, s.mid)}" | "${s.b.slice(0, s.split)}") and ("${s.a.slice(s.mid)}" | "${s.b.slice(s.split)}")`; })()
        : `alignment: ${h.steps.map(s => s.x).join("")} / ${h.steps.map(s => s.y).join("")} — cost ${h.dist}`;
      F.g.append("text").attr("x", 0).attr("y", -20).attr("font-size", 10.5).attr("fill", AC.muted).text(msg);
    }
    AL.stepper(svg, { frames, render, delay: 1100, label: "split" });
    const cells = c.get("cell"), bnd = bound(m, n);
    SX.setHtml("hirsch-readout", `${a} → ${b} (m = ${m}, n = ${n}): Hirschberg alignment cost <b>${h.dist}</b>, full-table distance ${full.dist} ${SX.flag(h.dist === full.dist)} · measured cells touched <b>${cells}</b> vs the full table's mn = ${m * n} (ratio ${SX.f2(cells / (m * n))}) · bound n·∑⌈m/2ᵏ⌉ = ${bnd} ${SX.flag(cells <= bnd)}; bare 2mn = ${2 * m * n} ${cells <= 2 * m * n ? "(within it)" : "(exceeded — rounding, m not a power of two)"} · ${splits.length} splits, recursion depth ${Math.ceil(Math.log2(Math.max(m, 1)))} · the two alignments are ${h.steps.map(s => s.op + s.x + s.y).join() === fullSteps.map(s => s.op + s.x + s.y).join() ? "identical" : "different paths of equal cost"}`);
  }
  SX.on("hi-pair", "change", build);
  build();
})();

/* ── 13  #lis-svg  LIS: the Θ(n²) dp and the O(n log n) tails method, stepped side by side ── */
(function () {
  if (!SX.has("lis-svg")) return;
  const W = 680, H = 300;
  function instance(which) {
    if (which === "trap") return [2, 6, 3, 4, 1];
    if (which === "sorted") return [1, 2, 3, 4, 5, 6, 7, 8];
    if (which === "rev") return [8, 7, 6, 5, 4, 3, 2, 1];
    if (which === "rand") { const r = AL.rng(53); return AL.shuffle(Array.from({ length: 12 }, (_, i) => i + 1), r); }
    return DI.lisA;
  }
  function build() {
    const a = instance(SX.val("lis-inst", "book")); const n = a.length;
    const cq = AL.counter(), ct = AL.counter(); const qf = [], tf = [];
    const q = DR.lisQuad(a, cq, e => qf.push(e)), t = DR.lisTails(a, ct, e => tf.push(e));
    const brute = n <= 16 ? DR.lisBrute(a) : null; const cl = AL.counter(); const viaLcs = DR.lisViaLcs(a, cl);
    const tailsIsSub = DR.isSubsequence(t.tails.join(","), a.join(","));
    const frames = a.map((_, i) => ({ kind: "step", i })).concat([{ kind: "done" }]);
    const svg = d3.select("#lis-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 60, r: 12, t: 22, b: 8 });
    const cw = Math.min(40, (F.iw - 4 * n) / n);
    function render(f, k) {
      F.g.selectAll("*").remove();
      const i = f.kind === "step" ? f.i : n - 1, done = f.kind === "done";
      const qs = qf[i], ts = tf[i];
      const qSet = new Set(done ? q.idx : []), tSet = new Set(done ? t.idx : []);
      F.g.append("text").attr("x", -56).attr("y", -8).attr("font-size", 10).attr("fill", AC.muted).text("method 1 — Θ(n²): dp[i] = 1 + max{dp[j] : j < i, aⱼ < aᵢ}");
      const R1 = AL.row(F.g, a, { x: 0, y: 0, w: cw, h: 28, gap: 4, label: "aᵢ", index: true, mark: (idx) => idx === i && !done ? AC.a2 : qSet.has(idx) ? AC.good : idx < i ? AC.panel2 : AC.panel });
      const R2 = AL.row(F.g, a.map((_, idx) => idx <= i ? qs.dp[idx] : null), { x: 0, y: 48, w: cw, h: 24, gap: 4, label: "dp[i]", index: false, fontSize: 12, mark: (idx) => idx === i && !done ? AC.a2 : qSet.has(idx) ? AC.good : null });
      /* predecessor arcs for the elements done so far */
      for (let idx = 0; idx <= i; idx++) if (qs.prev[idx] >= 0) { const x1 = R1.cellX(qs.prev[idx]), x2 = R1.cellX(idx); const lit = qSet.has(idx) && qSet.has(qs.prev[idx]);
        F.g.append("path").attr("d", `M${x1},${48 + 24} Q${(x1 + x2) / 2},${48 + 24 + 14 + (x2 - x1) / 10} ${x2},${48 + 24}`).attr("fill", "none").attr("stroke", lit ? AC.good : AC.line).attr("stroke-width", lit ? 2 : 1); }
      F.g.append("text").attr("x", -56).attr("y", 118).attr("font-size", 10).attr("fill", AC.muted).text("method 2 — O(n log n): tails[k] = smallest tail of an increasing subsequence of length k + 1");
      const R3 = AL.row(F.g, ts.tails, { x: 0, y: 126, w: cw, h: 28, gap: 4, label: "tails", index: true, mark: (kk) => kk === ts.k && !done ? AC.a2 : done ? AC.teal : null });
      AL.row(F.g, ts.tailIdx.map(v => "a" + (v + 1)), { x: 0, y: 172, w: cw, h: 20, gap: 4, label: "held by", index: false, fontSize: 10 });
      /* the tails method's chosen subsequence, lit on the input row via a bracket */
      if (done) t.idx.forEach(idx => F.g.append("rect").attr("x", R1.cellX(idx) - cw / 2 + 1).attr("y", -3).attr("width", cw - 2).attr("height", 34).attr("rx", 4).attr("fill", "none").attr("stroke", AC.teal).attr("stroke-width", 2));
      const msg = done ? `method 1 → [${q.seq.join(", ")}] (green)   ·   method 2 → [${t.seq.join(", ")}] (teal outline)   ·   both length ${q.len}` : `a${i + 1} = ${a[i]}:   dp = ${qs.dp[i]}${qs.prev[i] >= 0 ? " (after a" + (qs.prev[i] + 1) + " = " + a[qs.prev[i]] + ")" : " (starts fresh)"}   ·   tails: ${ts.k === ts.tails.length - 1 && (i === 0 || tf[i - 1].tails.length < ts.tails.length) ? "append at k = " + ts.k : "replace tails[" + ts.k + "]"} → [${ts.tails.join(", ")}]${ts.pred[i] >= 0 ? ", pred = a" + (ts.pred[i] + 1) : ""}`;
      F.g.append("text").attr("x", -56).attr("y", 222).attr("font-size", 10.5).attr("fill", AC.muted).text(msg);
      if (done && !tailsIsSub) F.g.append("text").attr("x", -56).attr("y", 240).attr("font-size", 10.5).attr("fill", AC.rose).text(`the final tails array [${t.tails.join(", ")}] is NOT a subsequence of the input — its length is the answer, its contents are not`);
    }
    AL.stepper(svg, { frames, render, delay: 900, label: "element" });
    SX.setHtml("lis-readout", `n = ${n}: method 1 length <b>${q.len}</b> → [${q.seq.join(", ")}] after <b>${cq.get("cmp")}</b> comparisons = n(n−1)/2 = ${n * (n - 1) / 2} ${SX.flag(cq.get("cmp") === n * (n - 1) / 2)} · method 2 length <b>${t.len}</b> → [${t.seq.join(", ")}] after <b>${ct.get("probe")}</b> binary-search probes; lengths equal ${SX.flag(q.len === t.len)} · final tails [${t.tails.join(", ")}] ${tailsIsSub ? "happens to be a subsequence of the input" : "is <b>not</b> a subsequence of the input"} · LCS(a, sorted a): length ${viaLcs.len} in ${cl.get("cell")} cells ${SX.flag(viaLcs.len === q.len)}${brute ? ` · brute force over all 2^${n} = ${SX.int(brute.count)} subsequences: <b>${brute.len}</b> ${SX.flag(brute.len === q.len)}` : ""}`);
  }
  SX.on("lis-inst", "change", build);
  build();
})();

/* ── chunk C figures ── */
/* ── 14  #mc-svg  matrix chain m/s tables by length, parenthesization tree ── */
(function () {
  if (!SX.has("mc-svg")) return;
  const W = 680, H = 300;
  const CHAINS = { six: DI.mcDims, three: [10, 100, 5, 50], four: [5, 10, 3, 12, 5], seven: [4, 10, 3, 12, 20, 7, 8, 6] };
  /* layout of the parenthesization tree: leaves in chain order, internal x = midpoint of children */
  function layoutTree(t) { let leaf = 0; const nodes = []; (function place(t, d) { t.d = d; if (t.leaf) { t.x = leaf++; nodes.push(t); return; } place(t.left, d + 1); place(t.right, d + 1); t.x = (t.left.x + t.right.x) / 2; nodes.push(t); })(t, 0); return { nodes, leaves: leaf, depth: Math.max(...nodes.map(n => n.d)) }; }
  function build() {
    const which = SX.val("mc-inst", "six");
    const p = which === "rand" ? (function () { const r = AL.rng(17); return Array.from({ length: 8 }, () => AL.randInt(r, 2, 40)); })() : CHAINS[which];
    const n = p.length - 1; const c = AL.counter(); const frames = [];
    const T = DR.mcTable(p, c, f => frames.push(f));
    const iters = c.get("iter"), formula = (n * n * n - n) / 6;
    const brute = DR.mcBrute(p); const cat = DR.catalan(n - 1);
    const parens = DR.mcParens(T.s, 1, n); const tree = DR.mcTree(T.s, 1, n); const lay = layoutTree(tree);
    const svg = d3.select("#mc-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 26, r: 8, t: 26, b: 8 });
    const cw = n <= 4 ? 52 : n <= 6 ? 44 : 38, ch = Math.min(26, Math.floor((F.ih - 20) / n)), sw = n <= 6 ? 20 : 17;
    function render(f, k) {
      F.g.selectAll("*").remove();
      const done = new Set(); for (let t = 0; t <= k; t++) done.add(frames[t].i + "," + frames[t].j);
      const cur = f.i + "," + f.j;
      F.g.append("text").attr("x", 0).attr("y", -14).attr("font-size", 11).attr("fill", AC.muted).text(`m[i][j]  (chain length l = ${f.l})`);
      SX.grid(F.g, n, n, { x: 0, y: 0, w: cw, h: ch, fontSize: 10,
        rowLabel: i => "i=" + (i + 1), colLabel: j => "j=" + (j + 1),
        text: (i, j) => i > j ? "" : i === j ? "0" : done.has((i + 1) + "," + (j + 1)) ? SX.int(f.m[i + 1][j + 1]) : "",
        fill: (i, j) => i > j ? AC.bg : (i + 1) + "," + (j + 1) === cur ? AC.a2 : (j - i + 1 === f.l && done.has((i + 1) + "," + (j + 1))) ? AC.accent : null });
      const sx = n * (cw + 2) + 40;
      F.g.append("text").attr("x", sx).attr("y", -14).attr("font-size", 11).attr("fill", AC.muted).text("s[i][j]");
      SX.grid(F.g, n, n, { x: sx, y: 0, w: sw, h: ch, fontSize: 10, colLabel: j => j + 1,
        text: (i, j) => i >= j ? "" : done.has((i + 1) + "," + (j + 1)) ? f.s[i + 1][j + 1] : "",
        fill: (i, j) => i >= j ? AC.bg : (i + 1) + "," + (j + 1) === cur ? AC.a2 : null });
      /* tree (only once the table is complete) */
      const tx = sx + n * (sw + 2) + 30, tw = F.iw - tx;
      if (k === frames.length - 1 && tw > 60) {
        const xs = d3.scaleLinear().domain([-0.5, lay.leaves - 0.5]).range([tx, F.iw]);
        const ys = d3.scaleLinear().domain([0, Math.max(1, lay.depth)]).range([10, F.ih - 16]);
        lay.nodes.forEach(nd => { if (nd.leaf) return; [nd.left, nd.right].forEach(ch2 => F.g.append("line").attr("x1", xs(nd.x)).attr("y1", ys(nd.d)).attr("x2", xs(ch2.x)).attr("y2", ys(ch2.d)).attr("stroke", AC.line)); });
        const rr = Math.max(6, Math.min(11, tw / lay.leaves / 2 - 1));
        lay.nodes.forEach(nd => { F.g.append("circle").attr("cx", xs(nd.x)).attr("cy", ys(nd.d)).attr("r", rr).attr("fill", nd.leaf ? AC.panel2 : AC.good).attr("stroke", AC.line);
          F.g.append("text").attr("x", xs(nd.x)).attr("y", ys(nd.d) + 3).attr("text-anchor", "middle").attr("font-size", rr >= 10 ? 9 : 7).attr("fill", nd.leaf ? AC.ink : AC.bg).text(nd.leaf ? "A" + nd.i : "k" + nd.k); });
        F.g.append("text").attr("x", tx).attr("y", -14).attr("font-size", 11).attr("fill", AC.muted).text("parenthesization tree");
      } else if (tw > 60) F.g.append("text").attr("x", tx).attr("y", -14).attr("font-size", 11).attr("fill", AC.muted).text("tree appears at the last step");
      F.g.append("text").attr("x", 0).attr("y", F.ih + 2).attr("font-size", 11).attr("fill", AC.ink).text(`m[${f.i}][${f.j}] = ${SX.int(f.m[f.i][f.j])}, split k = ${f.s[f.i][f.j]}  ·  cells fixed ${k + 1} / ${frames.length}`);
    }
    const st = AL.stepper(svg, { frames, render, delay: 500, label: "cell" });
    st.go(frames.length - 1);
    SX.setHtml("mc-readout", `p = ⟨${p.join(", ")}⟩, n = ${n} matrices · m[1][${n}] = <b>${SX.int(T.m[1][n])}</b> scalar multiplications, order <b>${parens}</b> · measured inner iterations <b>${iters}</b> = (n³ − n)/6 = ${formula} ${SX.flag(iters === formula)} · brute force over all <b>${brute.count}</b> parenthesizations = C(${n - 1}) = ${cat} ${SX.flag(brute.count === cat)}: minimum ${SX.int(brute.best)} ${SX.flag(brute.best === T.m[1][n])}${which === "three" ? " · the two orders cost 7 500 and 75 000" : ""}`);
  }
  SX.on("mc-inst", "change", build);
  build();
})();

/* ── 15  #obst-svg  optimal BST: e/root tables by length, the tree, vs all shapes ── */
(function () {
  if (!SX.has("obst-svg")) return;
  const W = 680, H = 320;
  const INST = {
    classic: { p: DI.obstP, q: DI.obstQ },
    uniform: { p: [0, .2, .2, .2, .2, .2], q: [0, 0, 0, 0, 0, 0] },
    skew: { p: [0, .05, .05, .05, .05, .70], q: [.02, .02, .02, .02, .01, .01] },
    four: { p: [0, .10, .30, .10, .20], q: [.05, .05, .10, .05, .05] }
  };
  /* in-order layout: dummies and keys alternate; x = in-order slot, y = depth */
  function layout(t) { let slot = 0; const nodes = []; (function walk(t, d) { if (t.dummy !== undefined) { t.x = slot++; t.d = d; nodes.push(t); return; } walk(t.left, d + 1); t.x = slot++; t.d = d; nodes.push(t); walk(t.right, d + 1); })(t, 0); return { nodes, slots: slot, depth: Math.max(...nodes.map(n => n.d)) }; }
  function build() {
    const which = SX.val("obst-inst", "classic"), knuth = SX.checked("obst-knuth", false);
    const p = INST[which].p, q = INST[which].q, n = p.length - 1;
    const c = AL.counter(); const frames = [];
    const T = DR.obstTable(p, q, c, f => frames.push(f), { knuth });
    const iters = c.get("iter"), full = n * (n + 1) * (n + 2) / 6;
    const tree = DR.obstTree(T.root, 1, n); const costFromDepths = DR.obstCostOf(tree, p, q);
    const brute = DR.obstBrute(p, q); const cat = DR.catalan(n); const mono = DR.obstRootMonotone(T.root, n);
    /* how many shapes hit the minimum */
    let atMin = 0;
    const shapes = (function all(i, j) { if (i > j) return [{ dummy: j }]; const out = []; for (let r = i; r <= j; r++) for (const L of all(i, r - 1)) for (const R of all(r + 1, j)) out.push({ key: r, left: L, right: R }); return out; })(1, n);
    shapes.forEach(s => { if (Math.abs(DR.obstCostOf(s, p, q) - brute.best) < 1e-9) atMin++; });
    const lay = layout(tree);
    const svg = d3.select("#obst-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 28, r: 8, t: 26, b: 8 });
    const cw = 36, ch = Math.min(24, Math.floor((F.ih - 24) / (n + 1))), rw = 20;
    function render(f, k) {
      F.g.selectAll("*").remove();
      const done = new Set(); for (let t = 0; t <= k; t++) done.add(frames[t].i + "," + frames[t].j);
      const cur = f.i + "," + f.j;
      F.g.append("text").attr("x", 0).attr("y", -14).attr("font-size", 11).attr("fill", AC.muted).text(`e[i][j]  (l = ${f.l} keys)`);
      SX.grid(F.g, n + 1, n + 1, { x: 0, y: 0, w: cw, h: ch, fontSize: 10, rowLabel: i => "i=" + (i + 1), colLabel: j => "j=" + j,
        text: (i, j) => { const ii = i + 1, jj = j; if (jj < ii - 1) return ""; if (jj === ii - 1) return SX.f2(f.e[ii][jj]); return done.has(ii + "," + jj) ? SX.f2(f.e[ii][jj]) : ""; },
        fill: (i, j) => { const ii = i + 1, jj = j; if (jj < ii - 1) return AC.bg; if (ii + "," + jj === cur) return AC.a2; if (jj === ii - 1) return AC.panel; if (jj - ii + 1 === f.l && done.has(ii + "," + jj)) return AC.accent; return null; } });
      const rx = (n + 1) * (cw + 2) + 40;
      F.g.append("text").attr("x", rx).attr("y", -14).attr("font-size", 11).attr("fill", AC.muted).text("root[i][j]");
      SX.grid(F.g, n, n, { x: rx, y: 0, w: rw, h: ch, fontSize: 10, colLabel: j => j + 1,
        text: (i, j) => i > j ? "" : done.has((i + 1) + "," + (j + 1)) ? f.root[i + 1][j + 1] : "",
        fill: (i, j) => i > j ? AC.bg : (i + 1) + "," + (j + 1) === cur ? AC.a2 : null });
      const tx = rx + n * (rw + 2) + 24, tw = F.iw - tx;
      if (k === frames.length - 1 && tw > 80) {
        const xs = d3.scaleLinear().domain([-0.5, lay.slots - 0.5]).range([tx, F.iw]);
        const ys = d3.scaleLinear().domain([0, Math.max(1, lay.depth)]).range([12, F.ih - 12]);
        const pmax = Math.max(...p.slice(1));
        lay.nodes.forEach(nd => { if (nd.dummy !== undefined) return; [nd.left, nd.right].forEach(ch2 => F.g.append("line").attr("x1", xs(nd.x)).attr("y1", ys(nd.d)).attr("x2", xs(ch2.x)).attr("y2", ys(ch2.d)).attr("stroke", AC.line)); });
        lay.nodes.forEach(nd => { if (nd.dummy !== undefined) { F.g.append("rect").attr("x", xs(nd.x) - 5).attr("y", ys(nd.d) - 5).attr("width", 10).attr("height", 10).attr("fill", AC.panel2).attr("stroke", AC.line); F.g.append("text").attr("x", xs(nd.x)).attr("y", ys(nd.d) + 14).attr("text-anchor", "middle").attr("font-size", 8).attr("fill", AC.muted).text("d" + nd.dummy); return; }
          const rr = 7 + 7 * p[nd.key] / pmax;
          F.g.append("circle").attr("cx", xs(nd.x)).attr("cy", ys(nd.d)).attr("r", rr).attr("fill", AC.good).attr("stroke", AC.line);
          F.g.append("text").attr("x", xs(nd.x)).attr("y", ys(nd.d) + 3.5).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.bg).text("k" + nd.key); });
        F.g.append("text").attr("x", tx).attr("y", -14).attr("font-size", 11).attr("fill", AC.muted).text("the tree from root[][]");
      } else if (tw > 80) F.g.append("text").attr("x", tx).attr("y", -14).attr("font-size", 11).attr("fill", AC.muted).text("tree appears at the last step");
      F.g.append("text").attr("x", 0).attr("y", F.ih + 4).attr("font-size", 11).attr("fill", AC.ink).text(`e[${f.i}][${f.j}] = ${SX.f2(f.e[f.i][f.j])}, w = ${SX.f2(f.w[f.i][f.j])}, root = k${f.root[f.i][f.j]}  ·  cells fixed ${k + 1} / ${frames.length}`);
    }
    const st = AL.stepper(svg, { frames, render, delay: 500, label: "cell" });
    st.go(frames.length - 1);
    SX.setHtml("obst-readout", `n = ${n} keys, ∑p + ∑q = ${SX.f2(p.reduce((a, b) => a + b, 0) + q.reduce((a, b) => a + b, 0))} · e[1][${n}] = <b>${SX.f4(T.e[1][n])}</b> expected nodes per search, root k${T.root[1][n]} · cost of the drawn tree recomputed from depths ${SX.f4(costFromDepths)} ${SX.flag(Math.abs(costFromDepths - T.e[1][n]) < 1e-9)} · brute force over all <b>${brute.count}</b> shapes = C(${n}) = ${cat} ${SX.flag(brute.count === cat)}: minimum ${SX.f4(brute.best)} ${SX.flag(Math.abs(brute.best - T.e[1][n]) < 1e-9)}, achieved by ${atMin} shape${atMin === 1 ? "" : "s"} · candidate roots examined <b>${iters}</b>${knuth ? ` with Knuth's window (full loop: n(n+1)(n+2)/6 = ${full})` : ` = n(n+1)(n+2)/6 = ${full} ${SX.flag(iters === full)}`} · root monotonicity root[i][j−1] ≤ root[i][j] ≤ root[i+1][j] on all ${mono.checked} pairs: ${SX.flag(mono.ok)}`);
  }
  SX.on("obst-inst", "change", build); SX.on("obst-knuth", "change", build);
  build();
})();

/* ── 16  #ivl-svg  interval DP by length: palindromic subsequence / burst balloons ── */
(function () {
  if (!SX.has("ivl-svg")) return;
  const W = 680, H = 320;
  const WORDS = { a: DI.lpsWord, b: "agbdba", c: "character" };
  const BALLS = { a: DI.burstVals, b: [2, 3, 4, 5, 6], c: [1, 5] };
  function build() {
    const prob = SX.val("ivl-prob", "lps"), inst = SX.val("ivl-inst", "a");
    const svg = d3.select("#ivl-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 30, r: 8, t: 44, b: 8 });
    if (prob === "lps") {
      const s = WORDS[inst], n = s.length; const c = AL.counter(); const frames = [];
      const L = DR.lpsTable(s, c, f => frames.push(f));
      const cells = c.get("cell"), formula = n * (n + 1) / 2; const rec = DR.lpsRecon(s, L, 0, n - 1); const brute = DR.lpsBrute(s); const sub = DR.lpSubstring(s);
      /* which indices the reconstruction used (walk the same decisions) */
      const used = new Set(); (function walk(i, j) { if (i > j) return; if (i === j) { used.add(i); return; } if (s[i] === s[j]) { used.add(i); used.add(j); walk(i + 1, j - 1); } else if (L[i + 1][j] >= L[i][j - 1]) walk(i + 1, j); else walk(i, j - 1); })(0, n - 1);
      const lcsRev = (function () { const t = s.split("").reverse().join(""); const m = []; for (let i = 0; i <= n; i++) m.push(new Array(n + 1).fill(0)); for (let i = 1; i <= n; i++) for (let j = 1; j <= n; j++) m[i][j] = s[i - 1] === t[j - 1] ? m[i - 1][j - 1] + 1 : Math.max(m[i - 1][j], m[i][j - 1]); return m[n][n]; })();
      const cw = Math.min(30, Math.floor((F.iw - 200) / n)), ch = Math.min(24, Math.floor((F.ih - 10) / n));
      function render(f, k) {
        F.g.selectAll("*").remove();
        const last = k === frames.length - 1;
        AL.row(F.g, s.split(""), { x: 0, y: -40, w: cw, h: 22, gap: 2, index: false, fontSize: 12, mark: i => last && used.has(i) ? AC.good : null });
        const cur = f.i !== undefined ? f.i + "," + f.j : "";
        SX.grid(F.g, n, n, { x: 0, y: 0, w: cw, h: ch, fontSize: 10, rowLabel: i => i, colLabel: j => j,
          text: (i, j) => i > j ? "" : (f.L[i][j] > 0 ? f.L[i][j] : ""),
          fill: (i, j) => i > j ? AC.bg : i + "," + j === cur ? AC.a2 : (f.L[i][j] > 0 && j - i + 1 === f.l) ? AC.accent : null });
        const tx = n * (cw + 2) + 30;
        const lines = [`length l = ${f.l}` + (f.i !== undefined ? `: L[${f.i}][${f.j}] = ${f.L[f.i][f.j]}` : ": every single character"), f.i !== undefined ? (s[f.i] === s[f.j] ? `s[${f.i}] = s[${f.j}] = '${s[f.i]}' → L[${f.i + 1}][${f.j - 1}] + 2` : `'${s[f.i]}' ≠ '${s[f.j]}' → max(L[${f.i + 1}][${f.j}], L[${f.i}][${f.j - 1}])`) : ""];
        if (last) lines.push("", `L[0][${n - 1}] = ${L[0][n - 1]}`, `reconstructed: ${rec}`, `longest palindromic SUBSTRING: ${s.substr(sub.start, sub.len)} (${sub.len})`);
        lines.forEach((t, r) => F.g.append("text").attr("x", tx).attr("y", 12 + r * 16).attr("font-size", 11).attr("fill", r >= 3 ? AC.good : AC.ink).text(t));
      }
      const st = AL.stepper(svg, { frames, render, delay: 400, label: "cell" }); st.go(frames.length - 1);
      SX.setHtml("ivl-readout", `s = ${s}, n = ${n} · longest palindromic subsequence <b>${L[0][n - 1]}</b> (${rec}) · measured cells <b>${cells}</b> = n(n+1)/2 = ${formula} ${SX.flag(cells === formula)} · brute force over all 2ⁿ = ${brute.count} subsequences: ${brute.best} ${SX.flag(brute.best === L[0][n - 1])} · longest palindromic substring ${sub.len} (${s.substr(sub.start, sub.len)}) — a different problem · LCS(s, reverse s) = ${lcsRev} ${SX.flag(lcsRev === L[0][n - 1])}`);
    } else {
      const v = BALLS[inst], n = v.length; const c = AL.counter(); const frames = [];
      const T = DR.burstTable(v, c, f => frames.push(f)); const a = T.a, N = a.length;
      const cells = c.get("cell"), iters = c.get("iter"); const brute = DR.burstBrute(v); const wrong = DR.burstFirstWrong(v);
      const order = DR.burstOrder(T.last, 0, N - 1);
      /* earnings per burst, replaying the order */
      const alive = a.slice(); const earn = order.map(k => { let l = k - 1; while (alive[l] === null) l--; let r = k + 1; while (alive[r] === null) r++; const e = alive[l] * alive[k] * alive[r]; const s = `${alive[l]}·${alive[k]}·${alive[r]} = ${e}`; alive[k] = null; return { k, e, s }; });
      const cw = Math.min(40, Math.floor((F.iw - 220) / N)), ch = Math.min(26, Math.floor((F.ih - 10) / N));
      function render(f, k) {
        F.g.selectAll("*").remove();
        const last = k === frames.length - 1;
        AL.row(F.g, a, { x: 0, y: -40, w: cw, h: 22, gap: 2, index: false, fontSize: 12, mark: i => (i === 0 || i === N - 1) ? AC.panel : null });
        const cur = f.i + "," + f.j;
        SX.grid(F.g, N, N, { x: 0, y: 0, w: cw, h: ch, fontSize: 10, rowLabel: i => i, colLabel: j => j,
          text: (i, j) => j <= i + 1 ? "" : (f.dp[i][j] > 0 || (i + "," + j) === cur ? f.dp[i][j] : ""),
          fill: (i, j) => j <= i + 1 ? AC.bg : i + "," + j === cur ? AC.a2 : (j - i === f.len && f.dp[i][j] > 0) ? AC.accent : null });
        const tx = N * (cw + 2) + 30;
        const lines = [`range (${f.i}, ${f.j}), ${f.len - 1} balloon${f.len > 2 ? "s" : ""} inside`, `B[${f.i}][${f.j}] = ${f.dp[f.i][f.j]}, last = a[${T.last[f.i][f.j]}] = ${a[T.last[f.i][f.j]]}`];
        if (last) { lines.push("", `B[0][${N - 1}] = ${T.best}`, "burst order (value: earned):"); earn.forEach(e => lines.push(`  ${a[e.k]}: ${e.s}`)); }
        lines.forEach((t, r) => F.g.append("text").attr("x", tx).attr("y", 12 + r * 15).attr("font-size", 11).attr("fill", r >= 3 ? AC.good : AC.ink).text(t));
      }
      const st = AL.stepper(svg, { frames, render, delay: 500, label: "cell" }); st.go(frames.length - 1);
      const formulaCells = n * (n + 1) / 2;
      const itersFormula = (function () { let t = 0; for (let m = 1; m <= n; m++) t += (n - m + 1) * m; return t; })();
      SX.setHtml("ivl-readout", `balloons [${v.join(", ")}], padded a = [${a.join(", ")}] · maximum coins <b>${T.best}</b>, burst order ${order.map(k => a[k]).join(" → ")}, earnings ${earn.map(e => e.e).join(" + ")} = ${earn.reduce((s, e) => s + e.e, 0)} ${SX.flag(earn.reduce((s, e) => s + e.e, 0) === T.best)} · measured states <b>${cells}</b> = n(n+1)/2 = ${formulaCells} ${SX.flag(cells === formulaCells)}, candidates <b>${iters}</b> = ∑ₘ (n−m+1)·m = n(n+1)(n+2)/6 = ${itersFormula} ${SX.flag(iters === itersFormula)} · brute force over all n! = ${brute.count} orders: ${brute.best} ${SX.flag(brute.best === T.best)} · the "burst k FIRST" recurrence returns ${wrong} ${wrong === T.best ? "(equal here — with ≤ 2 balloons the sides never interact)" : "— wrong, because the two sides share a moving boundary"}`);
    }
  }
  SX.on("ivl-prob", "change", build); SX.on("ivl-inst", "change", build);
  build();
})();

/* ── 17  #tree-svg  tree DP: (take, skip) per node in post-order, chosen set lit ── */
(function () {
  if (!SX.has("tree-svg")) return;
  const W = 680, H = 320;
  const INST = {
    nine: () => [DI.treeParent, DI.treeW],
    path: () => [[-1, 0, 1, 2, 3, 4, 5], [5, 3, 4, 7, 2, 6, 1]],
    star: () => [[-1, 0, 0, 0, 0, 0, 0], [10, 3, 3, 3, 3, 3, 3]],
    rand: () => { const r = AL.rng(23); const par = [-1], w = [AL.randInt(r, 1, 9)]; for (let u = 1; u < 10; u++) { par.push(AL.randInt(r, 0, u - 1)); w.push(AL.randInt(r, 1, 9)); } return [par, w]; }
  };
  /* layered layout: x by leaf order (children in index order), y by depth */
  function layout(T) { const x = new Array(T.n).fill(0), d = new Array(T.n).fill(0); let leaf = 0; (function place(u, depth) { d[u] = depth; if (!T.children[u].length) { x[u] = leaf++; return; } T.children[u].forEach(v => place(v, depth + 1)); x[u] = (x[T.children[u][0]] + x[T.children[u][T.children[u].length - 1]]) / 2; })(0, 0); return { x, d, leaves: leaf, depth: Math.max(...d) }; }
  function build() {
    const which = SX.val("tree-inst", "nine"); const [par, w] = INST[which](); const T = DR.treeFromParents(par, w); const n = T.n;
    const c = AL.counter(); const frames = [];
    const R = DR.treeMwis(T, c, f => frames.push(f));
    const visits = c.get("visit"); const brute = DR.treeMwisBrute(T);
    const cd = AL.counter(); const diam = DR.treeDiameter(T, cd); const sizes = DR.treeSizes(T); const sd = DR.treeSumDist(T), sdb = DR.treeSumDistBrute(T);
    const lay = layout(T); const chosen = new Set(R.chosen);
    const allFrames = frames.concat([{ final: true, take: R.take, skip: R.skip }]);
    const svg = d3.select("#tree-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 20, r: 200, t: 16, b: 30 });
    const xs = d3.scaleLinear().domain([-0.5, lay.leaves - 0.5]).range([0, F.iw]);
    const ys = d3.scaleLinear().domain([0, Math.max(1, lay.depth)]).range([14, F.ih - 20]);
    function render(f, k) {
      F.g.selectAll("*").remove();
      const done = new Set(); for (let t = 0; t <= Math.min(k, frames.length - 1); t++) done.add(frames[t].u);
      for (let u = 1; u < n; u++) F.g.append("line").attr("x1", xs(lay.x[T.parent[u]])).attr("y1", ys(lay.d[T.parent[u]])).attr("x2", xs(lay.x[u])).attr("y2", ys(lay.d[u])).attr("stroke", AC.line).attr("stroke-width", 1.5);
      for (let u = 0; u < n; u++) {
        const cur = !f.final && f.u === u;
        F.g.append("circle").attr("cx", xs(lay.x[u])).attr("cy", ys(lay.d[u])).attr("r", 14).attr("fill", f.final ? (chosen.has(u) ? AC.good : AC.panel2) : cur ? AC.a2 : done.has(u) ? AC.accent : AC.panel2).attr("stroke", AC.line);
        F.g.append("text").attr("x", xs(lay.x[u])).attr("y", ys(lay.d[u]) + 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", (f.final && chosen.has(u)) || cur || done.has(u) ? AC.bg : AC.ink).text(w[u]);
        F.g.append("text").attr("x", xs(lay.x[u]) + 16).attr("y", ys(lay.d[u]) - 8).attr("font-size", 8).attr("fill", AC.muted).text("#" + u);
        if (done.has(u) || f.final) F.g.append("text").attr("x", xs(lay.x[u])).attr("y", ys(lay.d[u]) + 26).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", cur ? AC.a2 : AC.ink).text(`${f.take[u]} | ${f.skip[u]}`);
      }
      const px = F.iw + 16;
      const lines = f.final ? ["reconstruction, top-down:", `best = max(${R.take[0]}, ${R.skip[0]}) = ${R.best}`, `chosen {${R.chosen.slice().sort((a, b) => a - b).join(", ")}}`, `weight ${R.chosen.reduce((s, u) => s + w[u], 0)}`] : [`post-order step ${k + 1}: node #${f.u}`, `take = ${w[f.u]} + ∑ skip(children) = ${f.take[f.u]}`, `skip = ∑ max(take, skip) = ${f.skip[f.u]}`, `children: ${T.children[f.u].length ? T.children[f.u].map(v => "#" + v).join(", ") : "none (leaf)"}`];
      lines.forEach((t, r) => F.g.append("text").attr("x", px).attr("y", 14 + r * 16).attr("font-size", 11).attr("fill", f.final && r === 2 ? AC.good : AC.ink).text(t));
    }
    const st = AL.stepper(svg, { frames: allFrames, render, delay: 600, label: "node" }); st.go(allFrames.length - 1);
    SX.setHtml("tree-readout", `n = ${n}, root #0 · maximum-weight independent set <b>${R.best}</b> = {${R.chosen.slice().sort((a, b) => a - b).join(", ")}} · measured node visits <b>${visits}</b> = n ${SX.flag(visits === n)} · brute force over all 2ⁿ = ${SX.int(brute.count)} subsets: ${brute.best} ${SX.flag(brute.best === R.best)} · diameter <b>${diam.diam}</b> edges (topmost node #${diam.through}), heights ⟨${diam.height.join(", ")}⟩, ${cd.get("visit")} visits · subtree sizes ⟨${sizes.join(", ")}⟩ · sum of distances from each node by re-rooting ⟨${sd.join(", ")}⟩ vs a BFS from every node ${SX.flag(sd.every((x, i) => x === sdb[i]))}${which === "path" ? " · on a path, taking every other node gives " + Math.max(w.filter((_, i) => i % 2 === 0).reduce((a, b) => a + b, 0), w.filter((_, i) => i % 2 === 1).reduce((a, b) => a + b, 0)) + "; the optimum skips two in a row" : ""}${which === "star" ? " · the centre's 10 loses to the six leaves' 18" : ""}`);
  }
  SX.on("tree-inst", "change", build);
  build();
})();

/* ── 18  #dag-svg  DP on a DAG: shortest / longest / count in topological order ── */
(function () {
  if (!SX.has("dag-svg")) return;
  const W = 680, H = 280;
  function randDag() { const r = AL.rng(41); const n = 7, edges = []; for (let v = 1; v < n; v++) { const must = AL.randInt(r, Math.max(0, v - 3), v - 1); for (let u = 0; u < v; u++) if (u === must || (v - u <= 3 && r() < 0.45)) edges.push({ u, v, w: AL.randInt(r, 1, 9) }); } return { n, edges }; }
  function build() {
    const mode = SX.val("dag-mode", "short"), inst = SX.val("dag-inst", "six");
    const G = inst === "six" ? DI.dagG : randDag(); const n = G.n;
    const c = AL.counter(); const frames = [];
    const R = DR.dagDP(G, 0, mode, c, f => frames.push(f));
    const relax = c.get("relax"); const brute = DR.dagBrute(G, 0);
    const key = mode === "short" ? "short" : mode === "long" ? "long" : "count";
    const agree = R.d.every((x, v) => x === brute[key][v]);
    const cb = AL.counter(); const bf = DR.bellmanFordK(G, 0, cb); let conv = bf.rounds.length - 1; for (let k = 1; k < bf.rounds.length; k++) if (bf.rounds[k].every((x, i) => x === bf.rounds[k - 1][i])) { conv = k - 1; break; }
    const cf = AL.counter(); const fw = DR.floydWarshall(G, cf);
    const svg = d3.select("#dag-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 30, r: 30, t: 20, b: 40 });
    const order = R.order; const pos = {}; order.forEach((v, i) => { pos[v] = { x: F.iw * i / Math.max(1, n - 1), y: F.ih / 2 + (i % 2 ? -1 : 1) * 55 }; });
    const fmt = x => x === Infinity ? "∞" : x === -Infinity ? "−∞" : x;
    const allFrames = [{ init: true, d: (function () { const d = new Array(n).fill(mode === "count" ? 0 : mode === "long" ? -Infinity : Infinity); d[0] = mode === "count" ? 1 : 0; return d; })() }].concat(frames);
    function edgePath(e) { const a = pos[e.u], b = pos[e.v]; const span = Math.abs(order.indexOf(e.v) - order.indexOf(e.u)); const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2 + (span > 1 ? (order.indexOf(e.u) % 2 ? 1 : -1) * 28 * (span - 1) : 0); return { d: `M${a.x},${a.y} Q${mx},${my} ${b.x},${b.y}`, mx, my, tx: b.x - (b.x - mx) * 0.2, ty: b.y - (b.y - my) * 0.2 }; }
    function render(f, k) {
      F.g.selectAll("*").remove();
      G.edges.forEach((e, i) => { const p = edgePath(e); const cur = !f.init && f.u === e.u && f.v === e.v; const done = !f.init && frames.indexOf(f) >= 0 && frames.slice(0, frames.indexOf(f) + 1).some(g => g.u === e.u && g.v === e.v);
        F.g.append("path").attr("d", p.d).attr("fill", "none").attr("stroke", cur ? AC.a2 : done ? AC.accent : AC.line).attr("stroke-width", cur ? 3 : 1.5);
        const a = pos[e.u], b = pos[e.v]; const ang = Math.atan2(b.y - p.my, b.x - p.mx); const hx = b.x - 16 * Math.cos(ang), hy = b.y - 16 * Math.sin(ang);
        F.g.append("path").attr("d", `M${hx},${hy} L${hx - 6 * Math.cos(ang - 0.45)},${hy - 6 * Math.sin(ang - 0.45)} L${hx - 6 * Math.cos(ang + 0.45)},${hy - 6 * Math.sin(ang + 0.45)} Z`).attr("fill", cur ? AC.a2 : done ? AC.accent : AC.muted);
        F.g.append("text").attr("x", (a.x + b.x) / 2 + (p.mx - (a.x + b.x) / 2) / 2).attr("y", (a.y + b.y) / 2 + (p.my - (a.y + b.y) / 2) / 2 - 3).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", cur ? AC.a2 : AC.muted).text(e.w); });
      for (let v = 0; v < n; v++) { const cur = !f.init && f.v === v; const src = !f.init && f.u === v;
        F.g.append("circle").attr("cx", pos[v].x).attr("cy", pos[v].y).attr("r", 15).attr("fill", cur ? AC.a2 : src ? AC.accent : AC.panel2).attr("stroke", AC.line);
        F.g.append("text").attr("x", pos[v].x).attr("y", pos[v].y + 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", cur || src ? AC.bg : AC.ink).text(v);
        F.g.append("text").attr("x", pos[v].x).attr("y", pos[v].y + (pos[v].y > F.ih / 2 ? 30 : -22)).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", cur ? AC.a2 : AC.good).text(`d = ${fmt(f.d[v])}`); }
      F.g.append("text").attr("x", 0).attr("y", F.ih + 30).attr("font-size", 11).attr("fill", AC.ink).text(f.init ? `initialised: d[0] = ${fmt(f.d[0])}, others ${fmt(f.d[1])} · topological order ${order.join(" ")}` : `relax ${f.u} → ${f.v}: d[${f.v}] = ${fmt(f.d[f.v])} · relaxations so far ${k}`);
    }
    const st = AL.stepper(svg, { frames: allFrames, render, delay: 600, label: "relaxation" }); st.go(allFrames.length - 1);
    const label = mode === "short" ? "shortest" : mode === "long" ? "longest" : "path count";
    SX.setHtml("dag-readout", `V = ${n}, E = ${G.edges.length}, topological order ${order.join(" ")} · ${label} from 0: ⟨${R.d.map(fmt).join(", ")}⟩ · measured relaxations <b>${relax}</b> = |E| ${SX.flag(relax === G.edges.length)} · brute force over all ${brute.paths - 1} paths leaving 0 (${brute.paths} counting the empty path at 0): ⟨${brute[key].map(fmt).join(", ")}⟩ ${SX.flag(agree)} · Bellman-Ford on the same graph (no order used): ${cb.get("relax")} relaxations = (V − 1)·E = ${(n - 1) * G.edges.length} ${SX.flag(cb.get("relax") === (n - 1) * G.edges.length)}, rows stable after round ${conv}, shortest ⟨${bf.d.map(fmt).join(", ")}⟩ ${SX.flag(bf.d.every((x, v) => x === brute.short[v]))} · Floyd-Warshall: ${cf.get("triple")} triples = V³ = ${n * n * n} ${SX.flag(cf.get("triple") === n * n * n)}, row 0 ⟨${fw[0].map(fmt).join(", ")}⟩ ${SX.flag(fw[0].every((x, v) => x === brute.short[v]))}`);
  }
  SX.on("dag-mode", "change", build); SX.on("dag-inst", "change", build);
  build();
})();

/* ── 19  #tsp-svg  Held–Karp on 4–5 cities: C(S, j) by |S|, the tour drawn ── */
(function () {
  if (!SX.has("tsp-svg")) return;
  const W = 680, H = 330;
  function build() {
    const n = +SX.val("tsp-n", 5);
    const D = DI.tspD.slice(0, n).map(r => r.slice(0, n)); const XY = DI.tspXY.slice(0, n);
    const c = AL.counter(); const frames = [];
    const R = DR.heldKarp(D, c, f => frames.push(f));
    const triples = c.get("triple"), closing = c.get("close"), formula = DR.hkTriples(n), statesF = (n - 1) * Math.pow(2, n - 2);
    const brute = DR.tspBrute(D);
    const tourLen = R.tour.reduce((s, v, i) => s + D[v][R.tour[(i + 1) % n]], 0);
    /* rows: subsets containing 0, ordered by popcount then value */
    const rows = []; for (let S = 1; S < (1 << n); S += 2) rows.push(S); rows.sort((a, b) => DR.popcount(a) - DR.popcount(b) || a - b);
    const setName = S => "{" + Array.from({ length: n }, (_, i) => i).filter(i => S >> i & 1).join(",") + "}";
    const svg = d3.select("#tsp-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 10, r: 10, t: 22, b: 8 });
    const mapW = 300; const sx = d3.scaleLinear().domain([0, 400]).range([10, mapW - 10]); const sy = d3.scaleLinear().domain([0, 280]).range([F.ih - 10, 20]);
    const tx = mapW + 70, cw = 34, ch = Math.min(18, Math.floor((F.ih - 4) / rows.length));
    function render(f, k) {
      F.g.selectAll("*").remove();
      const last = k === frames.length - 1;
      /* map */
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) F.g.append("line").attr("x1", sx(XY[i][0])).attr("y1", sy(XY[i][1])).attr("x2", sx(XY[j][0])).attr("y2", sy(XY[j][1])).attr("stroke", AC.grid);
      if (last) for (let i = 0; i < n; i++) { const a = R.tour[i], b = R.tour[(i + 1) % n]; F.g.append("line").attr("x1", sx(XY[a][0])).attr("y1", sy(XY[a][1])).attr("x2", sx(XY[b][0])).attr("y2", sy(XY[b][1])).attr("stroke", AC.good).attr("stroke-width", 3); F.g.append("text").attr("x", (sx(XY[a][0]) + sx(XY[b][0])) / 2).attr("y", (sy(XY[a][1]) + sy(XY[b][1])) / 2 - 4).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.good).text(D[a][b]); }
      else { /* the partial path for the current state: reconstruct through P is not exposed; show S and j */ }
      for (let i = 0; i < n; i++) { const inS = !last && (f.S >> i & 1); F.g.append("circle").attr("cx", sx(XY[i][0])).attr("cy", sy(XY[i][1])).attr("r", 13).attr("fill", !last && f.j === i ? AC.a2 : inS ? AC.accent : i === 0 ? AC.panel : AC.panel2).attr("stroke", AC.line); F.g.append("text").attr("x", sx(XY[i][0])).attr("y", sy(XY[i][1]) + 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", (!last && (f.j === i || inS)) ? AC.bg : AC.ink).text(i); }
      F.g.append("text").attr("x", 10).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text(last ? `tour ${R.tour.join(" → ")} → 0, length ${tourLen}` : `C(${setName(f.S)}, ${f.j}) = ${f.C[f.j]}: visited S in blue, end city j in amber`);
      /* table */
      const done = new Set(); for (let t = 0; t <= k; t++) done.add(frames[t].S + "," + frames[t].j);
      const valOf = (S, j) => { for (let t = k; t >= 0; t--) if (frames[t].S === S) return frames[t].C[j]; return Infinity; };
      F.g.append("text").attr("x", tx - 60).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted).text("S");
      for (let j = 1; j < n; j++) F.g.append("text").attr("x", tx + (j - 1) * (cw + 2) + cw / 2).attr("y", -8).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.muted).text("j=" + j);
      rows.forEach((S, r) => { F.g.append("text").attr("x", tx - 6).attr("y", r * (ch + 1) + ch / 2 + 4).attr("text-anchor", "end").attr("font-size", 9).attr("fill", AC.muted).text(setName(S));
        for (let j = 1; j < n; j++) { const inS = S >> j & 1; const key = S + "," + j; const cur = !last && f.S === S && f.j === j; const v = inS && done.has(key) ? valOf(S, j) : null;
          F.g.append("rect").attr("x", tx + (j - 1) * (cw + 2)).attr("y", r * (ch + 1)).attr("width", cw).attr("height", ch).attr("rx", 2).attr("fill", !inS ? AC.bg : cur ? AC.a2 : v !== null ? AC.panel2 : AC.panel).attr("stroke", AC.line);
          if (v !== null && v !== Infinity) F.g.append("text").attr("x", tx + (j - 1) * (cw + 2) + cw / 2).attr("y", r * (ch + 1) + ch / 2 + 3.5).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", cur ? AC.bg : AC.ink).text(v); } });
    }
    const st = AL.stepper(svg, { frames, render, delay: 450, label: "state" }); st.go(frames.length - 1);
    SX.setHtml("tsp-readout", `n = ${n} cities, home 0 · optimal tour <b>${R.tour.join(" → ")} → 0</b>, length <b>${R.best}</b> (edges re-added: ${tourLen} ${SX.flag(tourLen === R.best)}) · states (S, j) computed <b>${R.states}</b> = (n − 1)·2ⁿ⁻² = ${statesF} ${SX.flag(R.states === statesF)} · measured (S, j, i) triples <b>${triples}</b> = (n − 1) + ∑ₛ₌₃ⁿ C(n−1, s−1)(s−1)(s−2) = ${formula} ${SX.flag(triples === formula)}, plus ${closing} closing candidates · brute force over all (n − 1)! = ${brute.count} tours: ${brute.best} (${brute.tour.join(" → ")} → 0) ${SX.flag(brute.best === R.best)}${brute.tour.join() !== R.tour.join() ? " — a different tour of the same length (on a symmetric instance a tour and its reversal tie)" : ""} · versus n²·2ⁿ = ${n * n * Math.pow(2, n)} and n! = ${DR.factorial(n)}`);
  }
  SX.on("tsp-n", "change", build);
  build();
})();

/* ── 20  #digit-svg  digit DP state lattice: position × summary, tight spine ── */
(function () {
  if (!SX.has("digit-svg")) return;
  const W = 680, H = 320;
  function build() {
    const prop = SX.val("digit-prop", "sum"); let N = Math.floor(+SX.val("digit-n", 253)); if (!(N >= 0)) N = 0; if (N > 99999) N = 99999; const k = +SX.val("digit-k", 7);
    SX.setText("digit-k-out", k);
    const c = AL.counter(); const frames = [];
    const R = prop === "sum" ? DR.digitSumDP(N, k, c, f => frames.push(f)) : DR.digitNoAdjDP(N, c, f => frames.push(f));
    const states = c.get("state"), calls = c.get("call"), hits = c.get("hit");
    const brute = prop === "sum" ? DR.digitSumBrute(N, k) : DR.digitNoAdjBrute(N);
    const L = R.digits.length;
    /* summary axis */
    const summ = prop === "sum" ? (f => f.sum) : (f => f.started ? f.prev : -1);
    const rowsMax = prop === "sum" ? k : 9, rowsMin = prop === "sum" ? 0 : -1;
    const bound = prop === "sum" ? L * 2 * (k + 1) : L * 2 * 2 * 10 + 0;
    const svg = d3.select("#digit-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 60, r: 130, t: 24, b: 30 });
    const xs = d3.scaleLinear().domain([0, L]).range([0, F.iw]);
    const ys = d3.scaleLinear().domain([rowsMin, Math.max(rowsMax, rowsMin + 1)]).range([0, F.ih]);
    const key = f => f.pos + "," + (f.tight ? 1 : 0) + "," + summ(f);
    const isTight = f => f.tight;
    /* edges: from each state to the states it reads (children with the digit choices) */
    function children(f) { const out = []; const lim = f.tight ? R.digits[f.pos] : 9; for (let d = 0; d <= lim; d++) { if (prop === "sum") { const s = f.sum + d; if (s > k) continue; out.push({ pos: f.pos + 1, tight: f.tight && d === lim, sum: s }); } else { const st = f.started || d > 0; if (st && f.started && d === f.prev) continue; out.push({ pos: f.pos + 1, tight: f.tight && d === lim, started: st, prev: st ? d : -1 }); } } return out; }
    function render(f, kk) {
      F.g.selectAll("*").remove();
      const shown = frames.slice(0, kk + 1); const shownKeys = new Set(shown.map(key));
      for (let p = 0; p <= L; p++) F.g.append("text").attr("x", xs(p)).attr("y", -10).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.muted).text(p < L ? `pos ${p} (n=${R.digits[p]})` : "end");
      F.g.append("text").attr("x", -8).attr("y", ys(rowsMin) - 6).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.muted).text(prop === "sum" ? "sum" : "prev");
      for (let r = rowsMin; r <= rowsMax; r++) F.g.append("text").attr("x", -8).attr("y", ys(r) + 3).attr("text-anchor", "end").attr("font-size", 9).attr("fill", AC.muted).text(r === -1 ? "not started" : r);
      /* edges of shown states to shown/base children */
      shown.forEach(s => children(s).forEach(ch => { const ck = key(ch); if (ch.pos === L || shownKeys.has(ck)) F.g.append("line").attr("x1", xs(s.pos)).attr("y1", ys(summ(s))).attr("x2", xs(ch.pos)).attr("y2", ys(summ(ch))).attr("stroke", s === f ? AC.a2 : AC.grid).attr("stroke-width", s === f ? 1.5 : 1); }));
      /* base column */
      const baseOk = prop === "sum" ? (s => s === k) : (() => true);
      for (let r = rowsMin; r <= rowsMax; r++) { const ok = baseOk(r); F.g.append("rect").attr("x", xs(L) - 7).attr("y", ys(r) - 7).attr("width", 14).attr("height", 14).attr("rx", 2).attr("fill", ok ? AC.good : AC.panel).attr("stroke", AC.line); F.g.append("text").attr("x", xs(L)).attr("y", ys(r) + 3).attr("text-anchor", "middle").attr("font-size", 8).attr("fill", ok ? AC.bg : AC.muted).text(ok ? 1 : 0); }
      const rr = Math.min(11, Math.max(4, (ys(rowsMin + 1) - ys(rowsMin)) / 2 - 1));
      shown.forEach(s => { const cur = s === f;
        F.g.append("circle").attr("cx", xs(s.pos)).attr("cy", ys(summ(s))).attr("r", cur ? rr + 2 : rr).attr("fill", cur ? AC.a2 : AC.accent).attr("stroke", isTight(s) ? AC.a2 : AC.line).attr("stroke-width", isTight(s) ? 3 : 1);
        if (rr >= 8) F.g.append("text").attr("x", xs(s.pos)).attr("y", ys(summ(s)) + 3.5).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", AC.bg).text(s.count); });
      const px = F.iw + 14;
      const lines = [`state ${kk + 1} / ${frames.length}`, `pos ${f.pos}, ${f.tight ? "TIGHT (lim " + R.digits[f.pos] + ")" : "loose (lim 9)"}`, prop === "sum" ? `sum so far ${f.sum}` : (f.started ? `prev digit ${f.prev}` : "not started"), `count = ${f.count}`];
      if (kk === frames.length - 1) lines.push("", `f(0, tight, ${prop === "sum" ? 0 : "—"}) = ${R.total}`);
      lines.forEach((t, r) => F.g.append("text").attr("x", px).attr("y", 8 + r * 16).attr("font-size", 11).attr("fill", r === 5 ? AC.good : AC.ink).text(t));
      F.g.append("text").attr("x", 0).attr("y", F.ih + 24).attr("font-size", 10).attr("fill", AC.muted).text("amber ring = tight state (prefix equals N's); green squares = accepting base cases; edges = the digit choices a state sums over");
    }
    const st = AL.stepper(svg, { frames, render, delay: 450, label: "state" }); st.go(frames.length - 1);
    SX.setHtml("digit-readout", `N = ${SX.int(N)} (${L} digit${L > 1 ? "s" : ""}), property: ${prop === "sum" ? "digit sum = " + k : "no two adjacent digits equal"} · count in [0, N] = <b>${SX.int(R.total)}</b> · measured memo states <b>${states}</b> of at most ${bound} (${prop === "sum" ? "L × 2 × (k + 1)" : "L × 2 × 2 × 10"}), ${calls} calls, ${hits} memo hits · brute force over all ${SX.int(brute.checked)} integers in [0, N]: ${SX.int(brute.count)} ${SX.flag(brute.count === R.total)}${prop === "sum" && N === 253 && k === 7 ? " · the 21: " + brute.hits.join(", ") : ""}`);
  }
  SX.on("digit-prop", "change", build); SX.on("digit-n", "input", build); SX.on("digit-k", "input", build);
  build();
})();

/* ── chunk D figures ── */
/* ── 21  #mq-svg  the sliding-window recurrence: naive window scan vs the monotone deque, stepped ── */
(function () {
  if (!SX.has("mq-svg")) return;
  const W = 680, H = 300;
  function build() {
    const k = +SX.val("mq-k", 3), inst = SX.val("mq-inst", "shared");
    const cost = inst === "shared" ? DI.mqCosts : (function () { const r = AL.rng(1200 + (+inst || 1)); return Array.from({ length: 12 }, () => AL.randInt(r, 1, 9)); })();
    const n = cost.length;
    const cn = AL.counter(), cd = AL.counter(); const frames = [];
    const nv = DR.mqNaive(cost, k, cn); const dq = DR.mqDeque(cost, k, cd, f => frames.push(f));
    const brute = DR.mqBrute(cost, k); const path = DR.mqPath(dq.from, n);
    let naiveExpect = 0; for (let i = 1; i < n; i++) naiveExpect += Math.min(i, k);
    const svg = d3.select("#mq-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 70, r: 12, t: 12, b: 8 });
    const cw = Math.min(40, Math.floor(F.iw / n) - 3);
    function render(f, idx) {
      F.g.selectAll("*").remove();
      const last = idx === frames.length - 1; const onPath = new Set(last ? path : []);
      const lo = Math.max(0, f.i - k);
      AL.row(F.g, cost, { x: 0, y: 0, w: cw, h: 26, gap: 3, label: "cost[i]", index: true, fontSize: 11,
        mark: i => i === f.i ? AC.a2 : (i >= lo && i < f.i) ? AC.panel : null });
      AL.row(F.g, cost.map((_, i) => i <= f.i ? f.dp[i] : null), { x: 0, y: 56, w: cw, h: 26, gap: 3, label: "dp[i]", index: false, fontSize: 11,
        mark: i => i === f.i ? AC.a2 : onPath.has(i) ? AC.good : (i === f.min ? AC.teal : (i <= f.i ? AC.accent : null)) });
      /* the window bracket */
      if (f.i > 0) { const x0 = lo * (cw + 3), x1 = f.i * (cw + 3) - 3; F.g.append("path").attr("d", `M${x0},${52} L${x0},${48} L${x1},${48} L${x1},${52}`).attr("fill", "none").attr("stroke", AC.muted);
        F.g.append("text").attr("x", (x0 + x1) / 2).attr("y", 45).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", AC.muted).text(`window [${lo}, ${f.i})`); }
      /* the deque, drawn as boxes of indices with their dp values */
      const y0 = 116;
      F.g.append("text").attr("x", -64).attr("y", y0 + 16).attr("font-size", 11).attr("fill", AC.muted).text("deque");
      const items = f.dq.map(j => ({ j, kind: "in" })).concat(f.popped.map(j => ({ j, kind: "pop" })));
      if (f.expired !== null) items.unshift({ j: f.expired, kind: "exp" });
      items.forEach((it, q) => { const x = q * 62; const col = it.kind === "in" ? (it.j === f.i ? AC.a2 : AC.accent) : it.kind === "pop" ? AC.rose : AC.violet;
        F.g.append("rect").attr("x", x).attr("y", y0).attr("width", 56).attr("height", 34).attr("rx", 4).attr("fill", col).attr("stroke", AC.line).attr("opacity", it.kind === "in" ? 1 : 0.55);
        F.g.append("text").attr("x", x + 28).attr("y", y0 + 14).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.bg).text(`j = ${it.j}`);
        F.g.append("text").attr("x", x + 28).attr("y", y0 + 28).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.bg).text(`dp ${f.dp[it.j]}${it.kind === "pop" ? " ✗" : it.kind === "exp" ? " out" : ""}`); });
      F.g.append("text").attr("x", 0).attr("y", y0 + 50).attr("font-size", 10).attr("fill", AC.muted).text("front = the window minimum · back = the newest index · rose = popped (dominated by dp[i]) · violet = expired (left the window)");
      const lines = f.i === 0 ? [`dp[0] = cost[0] = ${f.dp[0]}; push 0`]
        : [`i = ${f.i}: ${f.expired !== null ? "index " + f.expired + " expired (< i − k = " + (f.i - k) + "); " : ""}front j = ${f.min} is the minimum of the window → dp[${f.i}] = cost[${f.i}] + dp[${f.min}] = ${cost[f.i]} + ${f.dp[f.min]} = ${f.dp[f.i]}`,
           `${f.popped.length ? "pop " + f.popped.map(j => j).join(", ") + " from the back (dp ≥ " + f.dp[f.i] + ": never the minimum again while " + f.i + " is in the window); " : "back dp < " + f.dp[f.i] + ", nothing popped; "}push ${f.i}`];
      if (last) lines.push(`done: dp[${n - 1}] = ${dq.val}, path ${path.join(" → ")} (green)`);
      lines.forEach((t, r) => F.g.append("text").attr("x", 0).attr("y", y0 + 72 + r * 16).attr("font-size", 10.5).attr("fill", r === 2 ? AC.good : AC.ink).text(t));
    }
    AL.stepper(svg, { frames, render, delay: 700, label: "index" });
    SX.setHtml("mq-readout", `n = ${n}, k = ${k}: dp[${n - 1}] = <b>${dq.val}</b> by the deque, <b>${nv.val}</b> by the naive scan ${SX.flag(dq.val === nv.val)}; path ${path.join(" → ")} costs ${path.map(i => cost[i]).join(" + ")} = ${path.reduce((s, i) => s + cost[i], 0)} ${SX.flag(path.reduce((s, i) => s + cost[i], 0) === dq.val)} · brute force over all <b>${SX.int(brute.paths)}</b> jump sequences: ${brute.val} ${SX.flag(brute.val === dq.val)} · measured: naive <b>${cn.get("cmp")}</b> candidate examinations = ∑ᵢ min(i, k) = ${naiveExpect} ${SX.flag(cn.get("cmp") === naiveExpect)} (Θ(nk)); deque <b>${cd.get("cmp")}</b> comparisons, ${cd.get("push")} pushes = n ${SX.flag(cd.get("push") === n)}, ${cd.get("pop")} pops + ${cd.get("expire")} expiries ≤ n ${SX.flag(cd.get("pop") + cd.get("expire") <= n)} — each index enters once and leaves at most once, so the total is O(n) whatever k is`);
  }
  SX.on("mq-k", "input", build); SX.on("mq-inst", "change", build);
  build();
})();

/* ── 22  #vi-svg  value iteration on a 4 × 4 gridworld: the heat-map converging, the greedy policy, the residuals ── */
(function () {
  if (!SX.has("vi-svg")) return;
  const W = 680, H = 340;
  function build() {
    const gamma = +SX.val("vi-gamma", 0.9), slip = +SX.val("vi-slip", 0.1);
    SX.setText("vi-gamma-out", gamma.toFixed(2));
    const g = Object.assign({}, DI.gridworld, { slip }); const M = DR.gwModel(g);
    const c = AL.counter(); const frames = []; const eps = 1e-6;
    const R = DR.valueIteration(M, gamma, eps, c, f => frames.push(f));
    const cp = AL.counter(); const PI = DR.policyIteration(M, gamma, 1e-10, cp);
    const Vpi = DR.policyEval(M, R.policy, gamma, 1e-12);
    const nonterm = M.S - 2 - M.wall.size;
    const contr = R.residuals.every((r, i) => i === 0 || r <= gamma * R.residuals[i - 1] + 1e-12);
    const polAgree = R.policy.filter((a, s) => a >= 0 && a === PI.policy[s]).length;
    /* where the two policies differ, the two actions must be equally valued under V* (a tie, not a disagreement) */
    const tieOk = R.policy.every((a, s) => a < 0 || a === PI.policy[s] || (function () { const q = DR.gwBackup(M, R.V, s, gamma, { add() {} }).q; return Math.abs(q[a] - q[PI.policy[s]]) < 1e-6; })());
    const vdiff = Math.max(...R.V.map((v, s) => Math.abs(v - PI.V[s]))), vpiDiff = Math.max(...R.V.map((v, s) => Math.abs(v - Vpi[s])));
    const svg = d3.select("#vi-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 16, r: 16, t: 18, b: 8 });
    const cell = Math.min(64, Math.floor((F.ih - 30) / g.rows)); const gx = 0, gy = 0;
    const ARR = ["↑", "→", "↓", "←"];
    const vmax = Math.max(1, ...frames[frames.length - 1].V.map(Math.abs));
    const rx = g.cols * (cell + 3) + 40, rw = F.iw - rx;
    const xs = d3.scaleLinear().domain([1, Math.max(2, frames.length)]).range([0, rw]);
    const ys = d3.scaleLog().domain([Math.max(1e-7, Math.min(...R.residuals) / 2), Math.max(1, R.residuals[0] * 1.5)]).range([F.ih - 40, 0]);
    function render(f, idx) {
      F.g.selectAll("*").remove();
      for (let s = 0; s < M.S; s++) { const r = Math.floor(s / g.cols), cc = s % g.cols; const x = gx + cc * (cell + 3), y = gy + r * (cell + 3);
        const v = f.V[s]; const col = s === M.goal ? AC.good : s === M.pit ? AC.rose : M.wall.has(s) ? AC.line : d3.interpolateRgb(AC.panel2, v >= 0 ? AC.accent : AC.bad)(Math.min(1, Math.abs(v) / vmax));
        F.g.append("rect").attr("x", x).attr("y", y).attr("width", cell).attr("height", cell).attr("rx", 4).attr("fill", col).attr("stroke", AC.line);
        if (s === M.goal || s === M.pit) F.g.append("text").attr("x", x + cell / 2).attr("y", y + cell / 2 + 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.bg).text(s === M.goal ? `goal +${g.goalR}` : `pit ${g.pitR}`);
        else if (M.wall.has(s)) F.g.append("text").attr("x", x + cell / 2).attr("y", y + cell / 2 + 4).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.muted).text("wall");
        else { F.g.append("text").attr("x", x + cell / 2).attr("y", y + cell / 2 - 4).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.ink).text(v.toFixed(3));
          const b = DR.gwBackup(M, f.V, s, gamma, { add() {} }); F.g.append("text").attr("x", x + cell / 2).attr("y", y + cell / 2 + 16).attr("text-anchor", "middle").attr("font-size", 15).attr("fill", AC.a2).text(ARR[b.arg]); } }
      F.g.append("text").attr("x", gx).attr("y", g.rows * (cell + 3) + 14).attr("font-size", 10.5).attr("fill", AC.muted).text(`sweep ${f.k} of ${frames.length}: max residual ‖V${f.k} − V${f.k - 1}‖∞ = ${f.res.toExponential(2)}${f.k > 1 ? " ≤ γ · " + R.residuals[f.k - 2].toExponential(2) + " = " + (gamma * R.residuals[f.k - 2]).toExponential(2) + (f.res <= gamma * R.residuals[f.k - 2] + 1e-12 ? " ✓" : " ✗") : ""}`);
      /* residual chart */
      const G = F.g.append("g").attr("transform", `translate(${rx},0)`);
      G.append("text").attr("x", 0).attr("y", -6).attr("font-size", 10.5).attr("fill", AC.muted).text("max residual per sweep (log scale) and the γᵏ envelope");
      AL.axisB(G, xs, F.ih - 40, 6, "sweep k"); AL.axisL(G, ys, 4, "", d3.format(".0e"));
      const env = R.residuals.map((_, i) => [i + 1, R.residuals[0] * Math.pow(gamma, i)]);
      G.append("path").attr("d", d3.line().x(d => xs(d[0])).y(d => ys(Math.max(1e-7, d[1])))(env)).attr("fill", "none").attr("stroke", AC.muted).attr("stroke-dasharray", "4 3");
      G.append("path").attr("d", d3.line().x((d, i) => xs(i + 1)).y(d => ys(Math.max(1e-7, d)))(R.residuals.slice(0, f.k))).attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 2);
      G.selectAll(null).data(R.residuals.slice(0, f.k)).join("circle").attr("cx", (d, i) => xs(i + 1)).attr("cy", d => ys(Math.max(1e-7, d))).attr("r", 2.5).attr("fill", AC.a2);
      AL.legend(G, [{ label: "measured residual", color: AC.a2 }, { label: "r₁ · γᵏ⁻¹ (the contraction envelope)", color: AC.muted, dash: "4 3" }], 4, F.ih - 8);
    }
    const st = AL.stepper(svg, { frames, render, delay: 260, label: "sweep" }); st.go(frames.length - 1);
    SX.setHtml("vi-readout", `4 × 4 gridworld, γ = ${gamma.toFixed(2)}, slip ${slip} each way, step reward ${g.step}: value iteration converged (‖V₊ − V‖∞ &lt; 10⁻⁶) after <b>${R.sweeps}</b> sweeps = <b>${SX.int(c.get("backup"))}</b> Bellman backups (= sweeps × ${nonterm} states × 4 actions = ${SX.int(R.sweeps * nonterm * 4)} ${SX.flag(c.get("backup") === R.sweeps * nonterm * 4)}) · residuals r₁ = ${R.residuals[0].toFixed(4)}, r₂ = ${(R.residuals[1] || 0).toFixed(4)}, … r${R.sweeps} = ${R.residuals[R.sweeps - 1].toExponential(2)}; the contraction bound rₖ₊₁ ≤ γ·rₖ holds on every sweep ${SX.flag(contr)} · V*(start, bottom-left) = <b>${R.V[M.id(g.rows - 1, 0)].toFixed(4)}</b> · policy iteration on the same model: <b>${PI.rounds}</b> evaluate-improve rounds, values within ${vdiff.toExponential(1)} of value iteration's ${SX.flag(vdiff < 1e-4)}, greedy policy identical on <b>${polAgree}/${nonterm}</b> states${polAgree < nonterm ? " — on the other " + (nonterm - polAgree) + " the two chosen actions are tied in value" : ""} ${SX.flag(tieOk)} · the greedy policy read off V*, evaluated exactly, is worth V* within ${vpiDiff.toExponential(1)} ${SX.flag(vpiDiff < 1e-4)}`);
  }
  SX.on("vi-gamma", "input", build); SX.on("vi-slip", "change", build);
  build();
})();

/* ── 23  #vit-svg  Viterbi on a 3-state × 6-step trellis, stepped column by column, the best path lit ── */
(function () {
  if (!SX.has("vit-svg")) return;
  const W = 680, H = 300;
  const OBS = { abbabb: DI.hmmObs, aaaaaa: [0, 0, 0, 0, 0, 0], bbbbbb: [1, 1, 1, 1, 1, 1], ababab: [0, 1, 0, 1, 0, 1], bbaaab: [1, 1, 0, 0, 0, 1] };
  function build() {
    const obs = OBS[SX.val("vit-obs", "abbabb")] || DI.hmmObs; const beamK = +SX.val("vit-beam", 1);
    const h = DI.hmm, S = h.pi.length, T = obs.length;
    const c = AL.counter(), cf = AL.counter(); const frames = [];
    const V = DR.viterbi(h, obs, c, f => frames.push(f)); const Fw = DR.forward(h, obs, cf); const B = DR.hmmBrute(h, obs);
    const rescored = DR.hmmPathProb(h, obs, V.path); const beam = DR.hmmBeam(h, obs, beamK);
    const multF = (T - 1) * S * S + T * S;
    const allFrames = frames.concat([{ t: T - 1, done: true }]);
    const svg = d3.select("#vit-svg"); SX.clearControls(svg.node());
    const F = AL.frame(svg, W, H, { l: 40, r: 150, t: 26, b: 8 });
    const xs = d3.scaleLinear().domain([0, T - 1]).range([30, F.iw - 30]); const ys = d3.scaleLinear().domain([0, S - 1]).range([24, F.ih - 40]);
    const fmt = x => x.toPrecision(3);
    function render(f, idx) {
      F.g.selectAll("*").remove();
      const upto = f.done ? T - 1 : f.t; const onPath = new Set(f.done ? V.path.map((j, t) => t + "," + j) : []);
      for (let t = 0; t < T; t++) F.g.append("text").attr("x", xs(t)).attr("y", -10).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.muted).text(`t = ${t + 1}: ${h.syms[obs[t]]}`);
      for (let j = 0; j < S; j++) F.g.append("text").attr("x", 0).attr("y", ys(j) + 4).attr("text-anchor", "end").attr("font-size", 11).attr("fill", AC.muted).text(h.names[j]);
      /* all transitions faintly, argmax predecessors solid, the best path green */
      for (let t = 1; t <= upto; t++) for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) { const arg = V.psi[t][j] === i; const lit = onPath.has((t - 1) + "," + i) && onPath.has(t + "," + j);
        F.g.append("line").attr("x1", xs(t - 1) + 14).attr("y1", ys(i)).attr("x2", xs(t) - 14).attr("y2", ys(j)).attr("stroke", lit ? AC.good : arg ? (t === f.t && !f.done ? AC.a2 : AC.accent) : AC.grid).attr("stroke-width", lit ? 3 : arg ? 1.6 : 1); }
      for (let t = 0; t <= upto; t++) for (let j = 0; j < S; j++) { const cur = !f.done && t === f.t; const lit = onPath.has(t + "," + j);
        F.g.append("circle").attr("cx", xs(t)).attr("cy", ys(j)).attr("r", 13).attr("fill", lit ? AC.good : cur ? AC.a2 : AC.panel2).attr("stroke", AC.line);
        F.g.append("text").attr("x", xs(t)).attr("y", ys(j) + 24).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", lit ? AC.good : cur ? AC.a2 : AC.ink).text(fmt(V.delta[t][j])); }
      const px = F.iw + 14;
      const lines = f.done ? ["best path (green):", V.path.map(j => h.names[j]).join(" "), `probability ${V.prob.toPrecision(4)}`, "", `forward total`, Fw.total.toPrecision(4)]
        : (f.t === 0 ? ["t = 1: δ₁(j) = πⱼ · bⱼ(o₁)"].concat(V.delta[0].map((d, j) => `${h.names[j]}: ${h.pi[j]} · ${h.B[j][obs[0]]} = ${fmt(d)}`))
          : [`t = ${f.t + 1}: δ(j) = max ᵢ δ(i)·aᵢⱼ · bⱼ(o)`].concat(V.delta[f.t].map((d, j) => `${h.names[j]}: from ${h.names[V.psi[f.t][j]]} → ${fmt(d)}`)));
      lines.forEach((t, r) => F.g.append("text").attr("x", px).attr("y", 8 + r * 15).attr("font-size", 10.5).attr("fill", f.done && (r === 1 || r === 5) ? AC.good : AC.ink).text(t));
      F.g.append("text").attr("x", 0).attr("y", F.ih - 6).attr("font-size", 10).attr("fill", AC.muted).text("nodes: δₜ(j) beneath each state · solid edge = the argmax predecessor ψₜ(j) · the best path is lit after the last column");
    }
    const st = AL.stepper(svg, { frames: allFrames, render, delay: 700, label: "column" }); st.go(allFrames.length - 1);
    SX.setHtml("vit-readout", `S = ${S} states, T = ${T} observations "${obs.map(o => h.syms[o]).join("")}": Viterbi path <b>${V.path.map(j => h.names[j]).join(" ")}</b> with probability <b>${V.prob.toPrecision(4)}</b>; re-scoring that path from π, A, B: ${rescored.toPrecision(4)} ${SX.flag(Math.abs(rescored - V.prob) < 1e-15)} · measured multiplications <b>${c.get("mult")}</b> = (T − 1)·S² + T·S = ${multF} ${SX.flag(c.get("mult") === multF)} (leading term T·S²), ${c.get("max")} maxima · brute force over all Sᵀ = <b>${SX.int(B.count)}</b> state sequences: best ${B.prob.toPrecision(4)} (${B.path.map(j => h.names[j]).join(" ")}) ${SX.flag(Math.abs(B.prob - V.prob) < 1e-15)} · forward algorithm (sum-product on the same trellis, ${cf.get("mult")} multiplications): P(observations) = <b>${Fw.total.toPrecision(4)}</b>; brute-force sum over the ${SX.int(B.count)} sequences ${B.total.toPrecision(4)} ${SX.flag(Math.abs(B.total - Fw.total) < 1e-12)} · beam search of width ${beamK} keeps ${beam.path.map(j => h.names[j]).join(" ")} at ${beam.p.toPrecision(4)} — ${Math.abs(beam.p - V.prob) < 1e-15 ? "the Viterbi path, on this instance" : "<b>not</b> the best path: it discarded the prefix the optimum needed"}`);
  }
  SX.on("vit-obs", "change", build); SX.on("vit-beam", "input", build);
  build();
})();

/* ── DA  the load-time audit: every DP family on the page against an independent computation over random instances ── */
(function () {
  if (!SX.has("audit-readout")) return;
  const out = []; const c = AL.counter(); const t0 = (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();
  const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
  /* 1. rod cutting: table = brute force over all cuttings; memo = table */
  { const T = 200; let ok = 0; const r = AL.rng(101);
    for (let t = 0; t < T; t++) { const n = AL.randInt(r, 1, 9); const p = [0]; for (let i = 1; i <= n; i++) p[i] = AL.randInt(r, 1, 3 * i); const R = DR.rodTable(p, n, c); const cuts = DR.rodCuts(R.s, n); const b = DR.rodBrute(p, n); const m = DR.rodMemo(p, n, c); if (R.r[n] === b.best && m === b.best && cuts.reduce((s, l) => s + p[l], 0) === b.best) ok++; }
    out.push(`rod cutting, ${T} random price tables (n ≤ 9): table = memo = brute force over all 2ⁿ⁻¹ cuttings, and the reconstructed cutting re-sums to the value, on <b>${ok}/${T}</b> ${SX.flag(ok === T)}`); }
  /* 2. house robber, line and ring; Kadane */
  { const T = 200; let okR = 0, okRing = 0, okK = 0; const r = AL.rng(102);
    for (let t = 0; t < T; t++) { const n = AL.randInt(r, 1, 10); const v = Array.from({ length: n }, () => AL.randInt(r, 0, 9)); const d = DR.robber(v, c); if (d[n - 1] === DR.robberBrute(v).best) okR++; if (DR.robberRing(v, c) === DR.robberRingBrute(v)) okRing++;
      const a = Array.from({ length: n }, () => AL.randInt(r, -6, 6)); if (DR.kadane(a, c).best === DR.kadaneBrute(a).best) okK++; }
    out.push(`house robber, ${T} random lines (n ≤ 10): = brute force over adjacency-free subsets <b>${okR}/${T}</b> ${SX.flag(okR === T)}; on a ring <b>${okRing}/${T}</b> ${SX.flag(okRing === T)}; Kadane = every subarray <b>${okK}/${T}</b> ${SX.flag(okK === T)}`); }
  /* 3. coin change: combinations, compositions, fewest */
  { const T = 150; let okC = 0, okP = 0, okM = 0, differ = 0; const r = AL.rng(103);
    for (let t = 0; t < T; t++) { const set = new Set(); while (set.size < AL.randInt(r, 2, 3)) set.add(AL.randInt(r, 1, 6)); const coins = [...set].sort((a, b) => a - b); const A = AL.randInt(r, 1, 12);
      const cb = DR.coinComb(coins, A, c)[A], cp = DR.coinComp(coins, A, c)[A], mn = DR.coinMin(coins, A, c).d[A]; const bb = DR.coinCombBrute(coins, A), bc = DR.coinCompBrute(coins, A);
      if (cb === bb.count) okC++; if (cp === bc) okP++; if (mn === bb.fewest) okM++; if (cb !== cp) differ++; }
    out.push(`coin change, ${T} random systems (2–3 coins ≤ 6, A ≤ 12): combinations = enumerated multisets <b>${okC}/${T}</b> ${SX.flag(okC === T)}; compositions = enumerated sequences <b>${okP}/${T}</b> ${SX.flag(okP === T)}; fewest = smallest multiset <b>${okM}/${T}</b> ${SX.flag(okM === T)}; the two counts differed on ${differ} — the loop-order bug would have been visible on each`); }
  /* 4. knapsack: 0/1 table, 1-D downward, unbounded (K and the upward array), binary splitting, subset sum */
  { const T = 150; let ok01 = 0, okDown = 0, okUnb = 0, okUp = 0, okB = 0, okSS = 0; const r = AL.rng(104);
    for (let t = 0; t < T; t++) { const n = AL.randInt(r, 1, 6), W0 = AL.randInt(r, 3, 15); const items = Array.from({ length: n }, (_, i) => ({ id: String(i + 1), w: AL.randInt(r, 1, 7), v: AL.randInt(r, 1, 20), k: AL.randInt(r, 1, 4) }));
      const b = DR.knapBrute(items, W0); const R = DR.knap01(items, W0, c); if (R.val === b.val && R.take.reduce((s, id) => s + items.find(it => it.id === id).v, 0) === b.val) ok01++;
      if (DR.knap1d(items, W0, "down", c).val === b.val) okDown++;
      const ub = DR.knapUnboundedBrute(items, W0); if (DR.knapUnbounded(items, W0, c).val === ub.val) okUnb++; if (DR.knap1d(items, W0, "up", c).val === ub.val) okUp++;
      const bb = DR.knapBoundedBrute(items, W0); if (DR.knap01(DR.knapBinarySplit(items), W0, c).val === bb.val) okB++;
      const a = Array.from({ length: AL.randInt(r, 1, 8) }, () => AL.randInt(r, 1, 12)); const Ts = AL.randInt(r, 1, 30); if (DR.subsetSum(a, Ts, c)[Ts] === DR.subsetSumBrute(a, Ts).ok) okSS++; }
    out.push(`knapsack, ${T} random instances (n ≤ 6, W ≤ 15): 0/1 table = brute force over 2ⁿ subsets and the chosen set re-sums <b>${ok01}/${T}</b> ${SX.flag(ok01 === T)}; 1-D array swept downward = 0/1 <b>${okDown}/${T}</b> ${SX.flag(okDown === T)}; K(w) = unbounded brute force over multisets <b>${okUnb}/${T}</b> ${SX.flag(okUnb === T)}; the array swept upward = unbounded <b>${okUp}/${T}</b> ${SX.flag(okUp === T)}; binary splitting (kᵢ ≤ 4) = bounded brute force <b>${okB}/${T}</b> ${SX.flag(okB === T)}; subset sum = brute force <b>${okSS}/${T}</b> ${SX.flag(okSS === T)}`); }
  /* 5. weighted interval scheduling */
  { const T = 200; let ok = 0, okBound = 0; const r = AL.rng(105);
    for (let t = 0; t < T; t++) { const n = AL.randInt(r, 1, 8); const ivs = Array.from({ length: n }, (_, i) => { const s = AL.randInt(r, 0, 20); return { id: i + 1, s, f: s + AL.randInt(r, 1, 8), v: AL.randInt(r, 1, 9) }; }); const cc = AL.counter(); const R = DR.wisSolve(ivs, cc); const b = DR.wisBrute(ivs); if (R.val === b.val) ok++; let bound = 0; for (let j = 1; j <= n; j++) bound += Math.ceil(Math.log2(j)); if (cc.get("probe") <= bound) okBound++; c.add("probe", cc.get("probe")); }
    out.push(`weighted interval scheduling, ${T} random sets (n ≤ 8): OPT = brute force over 2ⁿ subsets <b>${ok}/${T}</b> ${SX.flag(ok === T)}; binary-search probes ≤ ∑⌈log₂ j⌉ on <b>${okBound}/${T}</b> ${SX.flag(okBound === T)}`); }
  /* 6. LCS: table = brute force; two-row length; checkpointed reconstruction; naive recursion on the small ones */
  { const T = 150; let okB = 0, okSp = 0, okCp = 0, okAll = 0; const r = AL.rng(106);
    for (let t = 0; t < T; t++) { const X = Array.from({ length: AL.randInt(r, 1, 8) }, () => "ABC"[AL.randInt(r, 0, 2)]).join(""), Y = Array.from({ length: AL.randInt(r, 1, 8) }, () => "ABC"[AL.randInt(r, 0, 2)]).join("");
      const tb = DR.lcsTable(X, Y, c); const b = DR.lcsBrute(X, Y); if (tb.len === b.len) okB++; if (DR.lcsLenSpace(X, Y, c).len === tb.len) okSp++;
      const cp = DR.lcsCheckpoint(X, Y, AL.randInt(r, 1, 4), c); const tr = DR.lcsTraceback(X, Y, tb, c, "up"); if (cp.len === tb.len && cp.s === tr.s) okCp++;
      const all = DR.lcsAll(X, Y, tb); if (all.includes(tr.s) && all.every(s => s.length === tb.len && DR.isSubsequence(s, X) && DR.isSubsequence(s, Y))) okAll++; }
    out.push(`LCS, ${T} random pairs over {A, B, C} (lengths ≤ 8): length = brute force over all subsequences <b>${okB}/${T}</b> ${SX.flag(okB === T)}; two-row routine agrees <b>${okSp}/${T}</b> ${SX.flag(okSp === T)}; checkpointed traceback (every r-th row, r ≤ 4) returns the full table's string <b>${okCp}/${T}</b> ${SX.flag(okCp === T)}; every enumerated optimal traceback is a common subsequence of full length <b>${okAll}/${T}</b> ${SX.flag(okAll === T)}`); }
  /* 7. edit distance: table = Dijkstra on the edit graph; alignment re-sums; Hirschberg = table and within its bound; BFS on tiny unit-cost cases */
  { const T = 150; let okD = 0, okA = 0, okH = 0, okBnd = 0, okBfs = 0, nBfs = 0; const r = AL.rng(107);
    for (let t = 0; t < T; t++) { const a = Array.from({ length: AL.randInt(r, 1, 7) }, () => "abc"[AL.randInt(r, 0, 2)]).join(""), b = Array.from({ length: AL.randInt(r, 1, 7) }, () => "abc"[AL.randInt(r, 0, 2)]).join(""); const K = { ins: AL.randInt(r, 1, 3), del: AL.randInt(r, 1, 3), sub: AL.randInt(r, 1, 3) };
      const tb = DR.edTable(a, b, K, c); if (DR.edDijkstra(a, b, K, c) === tb.dist) okD++; if (DR.edAlignCost(DR.edAlign(a, b, tb), K) === tb.dist) okA++;
      const ch = AL.counter(); const h = DR.hirschberg(a, b, K, ch); c.add("cell", ch.get("cell")); if (h.dist === tb.dist) okH++; let bnd = 0; for (let k = 0; ; k++) { const q = Math.ceil(a.length / Math.pow(2, k)); bnd += q; if (q <= 1) break; } if (ch.get("cell") <= bnd * b.length + a.length * b.length) okBnd++;
      if (a.length + b.length <= 8) { const u = DR.edTable(a, b, null, c).dist; if (u <= 3) { nBfs++; if (DR.edBFS(a, b, u).dist === u) okBfs++; } } }
    /* Hirschberg on the §13 LCS pair: insert/delete 1, substitution 2 makes the distance m + n − 2·LCS and the split the LCS split; the §24 table quotes these two numbers */
    const chL = AL.counter(); const hL = DR.hirschberg("ABCBDAB", "BDCABA", { ins: 1, del: 1, sub: 2 }, chL); const hLc = chL.get("cell"); c.add("cell", hLc); const hLb = 6 * (7 + 4 + 2 + 1);
    out.push(`edit distance, ${T} random pairs over {a, b, c} (lengths ≤ 7, costs 1–3): table = Dijkstra on the edit graph <b>${okD}/${T}</b> ${SX.flag(okD === T)}; alignment re-sums to the distance <b>${okA}/${T}</b> ${SX.flag(okA === T)}; Hirschberg = full table <b>${okH}/${T}</b> ${SX.flag(okH === T)}, cells within the ceiling bound <b>${okBnd}/${T}</b> ${SX.flag(okBnd === T)}; BFS over unit edits on the ${nBfs} tiny cases <b>${okBfs}/${nBfs}</b> ${SX.flag(okBfs === nBfs)}; Hirschberg on the §13 LCS pair ABCBDAB / BDCABA (insert/delete 1, substitution 2, leftmost-minimum split): distance <b>${hL.dist}</b> = 7 + 6 − 2·4 ${SX.flag(hL.dist === 5)}, <b>${hLc}</b> cells against the bound 6·(7 + 4 + 2 + 1) = ${hLb} ${SX.flag(hLc <= hLb)}`); }
  /* 8. LIS: quadratic = tails = brute force; both reconstructions are increasing subsequences of full length */
  { const T = 200; let ok = 0, okRec = 0; const r = AL.rng(108);
    for (let t = 0; t < T; t++) { const n = AL.randInt(r, 1, 10); const a = AL.shuffle(Array.from({ length: 20 }, (_, i) => i + 1), r).slice(0, n); const q = DR.lisQuad(a, c), tl = DR.lisTails(a, c), b = DR.lisBrute(a); if (q.len === b.len && tl.len === b.len) ok++;
      const inc = s => s.every((x, i) => i === 0 || s[i - 1] < x); const sub = s => DR.isSubsequence(s.join(","), a.join(",")); if (inc(q.seq) && inc(tl.seq) && q.seq.length === b.len && tl.seq.length === b.len && sub(q.seq) && sub(tl.seq)) okRec++; }
    out.push(`longest increasing subsequence, ${T} random sequences (n ≤ 10, distinct): Θ(n²) = tails = brute force over 2ⁿ subsequences <b>${ok}/${T}</b> ${SX.flag(ok === T)}; both reconstructed sequences are increasing subsequences of the input of full length <b>${okRec}/${T}</b> ${SX.flag(okRec === T)}`); }
  /* 9. matrix chain and optimal BST (plain and Knuth) vs brute force over all Catalan(n) shapes */
  { const T = 120; let okM = 0, okIt = 0, okO = 0, okK = 0, okMono = 0; const r = AL.rng(109);
    for (let t = 0; t < T; t++) { const n = AL.randInt(r, 2, 6); const p = Array.from({ length: n + 1 }, () => AL.randInt(r, 1, 30)); const cc = AL.counter(); const M = DR.mcTable(p, cc); c.add("iter", cc.get("iter")); const b = DR.mcBrute(p); if (M.m[1][n] === b.best) okM++; if (cc.get("iter") === (n * n * n - n) / 6) okIt++;
      const m = AL.randInt(r, 1, 5); const raw = Array.from({ length: 2 * m + 1 }, () => AL.randInt(r, 1, 9)); const tot = raw.reduce((s, x) => s + x, 0); const pp = [0], qq = [raw[0] / tot]; for (let i = 1; i <= m; i++) { pp.push(raw[2 * i - 1] / tot); qq.push(raw[2 * i] / tot); }
      const O = DR.obstTable(pp, qq, c); const ob = DR.obstBrute(pp, qq); if (Math.abs(O.e[1][m] - ob.best) < 1e-9) okO++; const OK = DR.obstTable(pp, qq, c, null, { knuth: true }); if (Math.abs(OK.e[1][m] - ob.best) < 1e-9) okK++; if (DR.obstRootMonotone(O.root, m).ok) okMono++; }
    out.push(`matrix chain, ${T} random chains (n ≤ 6): m[1][n] = brute force over all C(n−1) parenthesizations <b>${okM}/${T}</b> ${SX.flag(okM === T)}, inner iterations = (n³ − n)/6 <b>${okIt}/${T}</b> ${SX.flag(okIt === T)}; optimal BST, ${T} random distributions (n ≤ 5): e[1][n] = brute force over all C(n) shapes <b>${okO}/${T}</b> ${SX.flag(okO === T)}, with Knuth's window <b>${okK}/${T}</b> ${SX.flag(okK === T)}, root monotonicity holds <b>${okMono}/${T}</b> ${SX.flag(okMono === T)}`); }
  /* 10. palindromic subsequence and burst balloons */
  { const T = 150; let okP = 0, okB = 0, wrongFirst = 0; const r = AL.rng(110);
    for (let t = 0; t < T; t++) { const s = Array.from({ length: AL.randInt(r, 1, 9) }, () => "abc"[AL.randInt(r, 0, 2)]).join(""); const L = DR.lpsTable(s, c); const rec = DR.lpsRecon(s, L, 0, s.length - 1); if (L[0][s.length - 1] === DR.lpsBrute(s).best && DR.isPal(rec) && rec.length === L[0][s.length - 1] && DR.isSubsequence(rec, s)) okP++;
      const v = Array.from({ length: AL.randInt(r, 1, 5) }, () => AL.randInt(r, 1, 9)); const B = DR.burstTable(v, c); const bb = DR.burstBrute(v); if (B.best === bb.best) okB++; if (DR.burstFirstWrong(v) !== bb.best) wrongFirst++; }
    out.push(`palindromic subsequence, ${T} random strings (n ≤ 9): = brute force, and the reconstruction is a palindromic subsequence of that length <b>${okP}/${T}</b> ${SX.flag(okP === T)}; burst balloons, ${T} random rows (n ≤ 5): last-burst table = brute force over all n! orders <b>${okB}/${T}</b> ${SX.flag(okB === T)}; the first-burst recurrence was wrong on ${wrongFirst}`); }
  /* 11. tree DP: independent set, diameter (vs all-pairs BFS), distance sums */
  { const T = 150; let okI = 0, okD = 0, okS = 0; const r = AL.rng(111);
    for (let t = 0; t < T; t++) { const n = AL.randInt(r, 1, 10); const par = [-1], w = [AL.randInt(r, 1, 9)]; for (let u = 1; u < n; u++) { par.push(AL.randInt(r, 0, u - 1)); w.push(AL.randInt(r, 1, 9)); } const Tn = DR.treeFromParents(par, w);
      const R = DR.treeMwis(Tn, c); const b = DR.treeMwisBrute(Tn); const ch = new Set(R.chosen); const legal = R.chosen.every(u => u === 0 || !ch.has(Tn.parent[u])); if (R.best === b.best && legal && R.chosen.reduce((s, u) => s + w[u], 0) === b.best) okI++;
      const sd = DR.treeSumDist(Tn), sdb = DR.treeSumDistBrute(Tn); if (same(sd, sdb)) okS++;
      /* diameter by BFS from every node: the largest eccentricity */
      const adj = Array.from({ length: n }, () => []); for (let u = 1; u < n; u++) { adj[u].push(par[u]); adj[par[u]].push(u); } let ecc = 0; for (let s0 = 0; s0 < n; s0++) { const d = new Array(n).fill(-1); d[s0] = 0; const q = [s0]; while (q.length) { const u = q.shift(); adj[u].forEach(v => { if (d[v] < 0) { d[v] = d[u] + 1; q.push(v); } }); } ecc = Math.max(ecc, ...d); } if (DR.treeDiameter(Tn, c).diam === ecc) okD++; }
    out.push(`tree DP, ${T} random trees (n ≤ 10): max-weight independent set = brute force over 2ⁿ subsets, chosen set legal and re-sums <b>${okI}/${T}</b> ${SX.flag(okI === T)}; diameter = largest eccentricity by BFS from every node <b>${okD}/${T}</b> ${SX.flag(okD === T)}; re-rooted distance sums = BFS from every node <b>${okS}/${T}</b> ${SX.flag(okS === T)}`); }
  /* 12. DAG DP: shortest / longest / count vs path enumeration; Bellman-Ford and Floyd-Warshall agree */
  { const T = 150; let ok = 0, okBF = 0, okFW = 0; const r = AL.rng(112);
    for (let t = 0; t < T; t++) { const n = AL.randInt(r, 2, 7); const edges = []; for (let v = 1; v < n; v++) { const must = AL.randInt(r, 0, v - 1); for (let u = 0; u < v; u++) if (u === must || r() < 0.4) edges.push({ u, v, w: AL.randInt(r, -3, 9) }); } const G = { n, edges };
      const b = DR.dagBrute(G, 0); const s = DR.dagDP(G, 0, "short", c).d, l = DR.dagDP(G, 0, "long", c).d, k = DR.dagDP(G, 0, "count", c).d; if (same(s, b.short) && same(l, b.long) && same(k, b.count)) ok++;
      if (same(DR.bellmanFordK(G, 0, c).d, b.short)) okBF++; if (same(DR.floydWarshall(G, c)[0], b.short)) okFW++; }
    out.push(`DAG DP, ${T} random DAGs (V ≤ 7, weights −3 … 9): shortest, longest and path count = enumeration of every path <b>${ok}/${T}</b> ${SX.flag(ok === T)}; Bellman-Ford agrees <b>${okBF}/${T}</b> ${SX.flag(okBF === T)}; Floyd-Warshall's row 0 agrees <b>${okFW}/${T}</b> ${SX.flag(okFW === T)}`); }
  /* 13. Held–Karp vs all (n−1)! tours; triples = the closed form */
  { const T = 100; let ok = 0, okTr = 0; const r = AL.rng(113);
    for (let t = 0; t < T; t++) { const n = AL.randInt(r, 3, 6); const D = Array.from({ length: n }, () => new Array(n).fill(0)); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) D[i][j] = AL.randInt(r, 1, 20); const cc = AL.counter(); const R = DR.heldKarp(D, cc); c.add("triple", cc.get("triple")); const b = DR.tspBrute(D); const len = R.tour.reduce((s, v, i) => s + D[v][R.tour[(i + 1) % n]], 0); if (R.best === b.best && len === b.best) ok++; if (cc.get("triple") === DR.hkTriples(n)) okTr++; }
    out.push(`Held–Karp, ${T} random asymmetric distance matrices (n = 3 … 6): tour = brute force over all (n − 1)! tours and the reconstructed tour re-sums <b>${ok}/${T}</b> ${SX.flag(ok === T)}; triples = the closed form <b>${okTr}/${T}</b> ${SX.flag(okTr === T)}`); }
  /* 14. digit DP vs brute force over 0 … N */
  { const T = 100; let okS = 0, okA = 0; const r = AL.rng(114);
    for (let t = 0; t < T; t++) { const N = AL.randInt(r, 0, 3000), k = AL.randInt(r, 0, 20); if (DR.digitSumDP(N, k, c).total === DR.digitSumBrute(N, k).count) okS++; if (DR.digitNoAdjDP(N, c).total === DR.digitNoAdjBrute(N).count) okA++; }
    out.push(`digit DP, ${T} random N ≤ 3 000: digit-sum count = brute force over 0 … N <b>${okS}/${T}</b> ${SX.flag(okS === T)}; no-two-adjacent-equal count <b>${okA}/${T}</b> ${SX.flag(okA === T)}`); }
  /* 15. monotone deque vs the naive window scan vs every jump sequence; the O(n) operation count */
  { const T = 200; let ok = 0, okOps = 0; const r = AL.rng(115);
    for (let t = 0; t < T; t++) { const n = AL.randInt(r, 1, 10), k = AL.randInt(r, 1, 4); const cost = Array.from({ length: n }, () => AL.randInt(r, 1, 9)); const cd = AL.counter(); const d = DR.mqDeque(cost, k, cd); const nv = DR.mqNaive(cost, k, c), b = DR.mqBrute(cost, k); c.add("cmp", cd.get("cmp")); if (d.val === nv.val && d.val === b.val) ok++; if (cd.get("push") === n && cd.get("pop") + cd.get("expire") <= n) okOps++; }
    out.push(`sliding-window recurrence, ${T} random cost rows (n ≤ 10, k ≤ 4): deque = naive scan = brute force over every jump sequence <b>${ok}/${T}</b> ${SX.flag(ok === T)}; pushes = n and pops + expiries ≤ n <b>${okOps}/${T}</b> ${SX.flag(okOps === T)}`); }
  /* 16. Viterbi and forward vs all S^T sequences; beam width 1 failures counted */
  { const T = 100; let okV = 0, okF = 0, beamWrong = 0; const r = AL.rng(116);
    for (let t = 0; t < T; t++) { const S = AL.randInt(r, 2, 3), K = 2, Tn = AL.randInt(r, 3, 5); const norm = a => { const s = a.reduce((x, y) => x + y, 0); return a.map(x => x / s); }; const h = { pi: norm(Array.from({ length: S }, () => AL.randInt(r, 1, 9))), A: Array.from({ length: S }, () => norm(Array.from({ length: S }, () => AL.randInt(r, 1, 9)))), B: Array.from({ length: S }, () => norm(Array.from({ length: K }, () => AL.randInt(r, 1, 9)))) }; const obs = Array.from({ length: Tn }, () => AL.randInt(r, 0, K - 1));
      const V = DR.viterbi(h, obs, c), F = DR.forward(h, obs, c), b = DR.hmmBrute(h, obs); if (Math.abs(V.prob - b.prob) < 1e-15 && Math.abs(DR.hmmPathProb(h, obs, V.path) - b.prob) < 1e-15) okV++; if (Math.abs(F.total - b.total) < 1e-12) okF++; if (Math.abs(DR.hmmBeam(h, obs, 1).p - b.prob) > 1e-15) beamWrong++; }
    out.push(`Viterbi, ${T} random models (S ≤ 3, T ≤ 5): best path = brute force over all Sᵀ sequences <b>${okV}/${T}</b> ${SX.flag(okV === T)}; forward total = brute-force sum <b>${okF}/${T}</b> ${SX.flag(okF === T)}; a width-1 beam missed the best path on ${beamWrong}/${T} — it is not a DP`); }
  /* 17. value iteration vs policy iteration on random small gridworlds; the contraction on every sweep */
  { const T = 40; let okPI = 0, okC = 0, okFix = 0; const r = AL.rng(117);
    for (let t = 0; t < T; t++) { const rows = AL.randInt(r, 2, 4), cols = AL.randInt(r, 2, 4); const cells = AL.shuffle(Array.from({ length: rows * cols }, (_, i) => [Math.floor(i / cols), i % cols]), r); const gamma = 0.5 + 0.45 * r();
      const g = { rows, cols, goal: cells[0], pit: cells[1], walls: rows * cols > 4 && r() < 0.5 ? [cells[2]] : [], step: -0.04, goalR: 1, pitR: -1, slip: [0, 0.1, 0.2][AL.randInt(r, 0, 2)] }; const M = DR.gwModel(g);
      const cv = AL.counter(); const R = DR.valueIteration(M, gamma, 1e-7, cv); c.add("backup", cv.get("backup")); const PI = DR.policyIteration(M, gamma, 1e-11, c);
      if (R.V.every((v, s) => Math.abs(v - PI.V[s]) < 1e-4)) okPI++; if (R.residuals.every((x, i) => i === 0 || x <= gamma * R.residuals[i - 1] + 1e-12)) okC++;
      const Vpi = DR.policyEval(M, R.policy, gamma, 1e-12); if (R.V.every((v, s) => Math.abs(v - Vpi[s]) < 1e-4)) okFix++; }
    out.push(`value iteration, ${T} random gridworlds (≤ 4 × 4, γ ∈ [0.5, 0.95], slip 0–0.2): values = policy iteration's <b>${okPI}/${T}</b> ${SX.flag(okPI === T)}; residuals obey rₖ₊₁ ≤ γ·rₖ on every sweep <b>${okC}/${T}</b> ${SX.flag(okC === T)}; the greedy policy, evaluated exactly, attains V* <b>${okFix}/${T}</b> ${SX.flag(okFix === T)}`); }
  const ms = ((typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now()) - t0;
  const allOk = !out.some(s => s.includes("DISAGREE"));
  SX.setHtml("audit-readout", `<b>Load-time audit</b> — ${out.length} families of random instances, every DP answer recomputed independently (brute force over all cuttings, subsets, multisets, sequences, subsequences, parenthesizations, shapes, orders, paths, tours, integers and state sequences; a second algorithm where enumeration is not the natural check):<br>` + out.map(s => "· " + s).join("<br>") + `<br>· overall: ${allOk ? '<span style="color:' + AC.good + '">all audits agree</span>' : '<span style="color:' + AC.bad + '">SOME AUDIT DISAGREES</span>'} · total instrumented work in the audit: ${SX.int(Object.values(c.all()).reduce((a, b) => a + b, 0))} counted operations in ${ms.toFixed(0)} ms`);
  SX.setHtml("audit-cell", allOk ? "all " + out.length + " families agree — details in the readout below" : "a family DISAGREES — see below");
})();

/* @@FIGURES-END@@ */
})();
