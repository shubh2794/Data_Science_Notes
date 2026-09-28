/* online-randomized.viz.js — figures for dsa/algorithms/online-randomized.html
   (part 10 of the Algorithm Design & Analysis series).

   Loaded after ../../data.js, ../../notes.js and ../dsa-viz.js, so the globals
   AC (palette) and AL (rng/frame/axisB/axisL/legend/counter/…) are available.

   House rule obeyed throughout: every cost, fault count, ratio, success rate or
   period this page DISPLAYS is produced by running the real routine — under
   AL.counter() where it is a count — and reading the result back. Nothing in a
   caption is a typed-in constant. Every random experiment is driven by a SEEDED
   generator (AL.rng / the local mulberry32), so every reader sees the same
   picture and a stale number in the prose announces itself.

   Layout of this file:
     OR — the instrumented routines, pure and DOM-free (also loadable in node).
     OX — small DOM helpers shared by the figures.
     figures — one block per <svg>, in page order. */

/* ═══════════════════════════════════════════════════════════════════════════
   OR — the routines
   ═══════════════════════════════════════════════════════════════════════════ */
const OR = (function () {
  const nop = { add: function () {}, get: function () { return 0; } };
  const ctr = c => c || nop;
  function rnd(seed) {                      // mulberry32, identical to AL.rng
    let a = (seed >>> 0) || 1;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const ri = (r, n) => Math.floor(r() * n);         // uniform integer in [0, n)
  function harmonic(n) { let s = 0; for (let i = 1; i <= n; i++) s += 1 / i; return s; }
  function fyPerm(n, r, c) {                        // Fisher–Yates on 0..n−1
    c = ctr(c);
    const a = new Array(n); for (let i = 0; i < n; i++) a[i] = i;
    for (let i = n - 1; i > 0; i--) { const j = ri(r, i + 1); c.add("draw"); const t = a[i]; a[i] = a[j]; a[j] = t; }
    return a;
  }

  /* ── §04–05  SKI RENTAL ────────────────────────────────────────────────
     Rent costs 1 per day, buying costs B. The skier does not know the number
     of days d. "Buy on day t" = rent on days 1…t−1, buy at the start of day t. */
  function skiCost(t, d, B, c) {
    c = ctr(c); c.add("eval");
    return d < t ? d : (t - 1) + B;
  }
  function skiOpt(d, B) { return Math.min(d, B); }
  /* the adversary's best response: the stop day d ≤ dMax maximising ALG/OPT */
  function skiWorst(t, B, dMax, c) {
    let best = { ratio: 0, d: 1 };
    for (let d = 1; d <= dMax; d++) {
      const r = skiCost(t, d, B, c) / skiOpt(d, B);
      if (r > best.ratio + 1e-12) best = { ratio: r, d: d };
    }
    return best;
  }
  /* the optimal randomised rule for integer B:
     P(buy on day i) ∝ ((B−1)/B)^(B−i),  i = 1…B */
  function skiDist(B) {
    const p = []; let s = 0;
    for (let i = 1; i <= B; i++) { const v = Math.pow((B - 1) / B, B - i); p.push(v); s += v; }
    return p.map(v => v / s);
  }
  function skiRandExpected(d, B, p, c) {
    let e = 0;
    for (let i = 1; i <= B; i++) e += p[i - 1] * skiCost(i, d, B, c);
    return e;
  }

  /* ── §06–10  PAGING ─────────────────────────────────────────────────────
     simulate(policy, k, seq, seed, counter) serves the request sequence with a
     cache of k slots and returns {faults, snaps}: snaps[t] = the slot contents
     after request t, and whether it faulted. Every policy keeps its pages in a
     fixed array of slots so the figure can draw "which slot changed". */
  const POLICIES = ["LRU", "FIFO", "LFU", "FWF", "CLOCK", "MARK", "OPT"];
  function nextUse(seq) {
    const nxt = new Array(seq.length), last = new Map();
    for (let t = seq.length - 1; t >= 0; t--) { nxt[t] = last.has(seq[t]) ? last.get(seq[t]) : Infinity; last.set(seq[t], t); }
    return nxt;
  }
  function makeCache(policy, k, seed, nxt, c) {
    c = ctr(c);
    const r = rnd(seed || 1);
    const slot = [];                       // page in each slot
    const where = new Map();               // page → slot index
    const meta = [];                       // per-slot bookkeeping
    const nextOf = new Map();              // OPT: page → next use of the copy in cache
    let hand = 0;
    const st = { faults: 0, phases: 1, slot: slot, where: where };
    st.access = function (p, t) {
      c.add("req");
      let hit = where.has(p), evicted = null, into = -1;
      if (hit) {
        const s = where.get(p);
        if (policy === "LRU") meta[s] = t;
        else if (policy === "LFU") { meta[s].f++; meta[s].t = t; }
        else if (policy === "CLOCK" || policy === "MARK") meta[s] = 1;
        if (policy === "OPT") nextOf.set(p, nxt[t]);
        into = s;
      } else {
        st.faults++; c.add("fault");
        if (slot.length < k) { into = slot.length; slot.push(p); }
        else {
          if (policy === "LRU" || policy === "FIFO") { into = 0; for (let s = 1; s < k; s++) { c.add("scan"); if (meta[s] < meta[into]) into = s; } }
          else if (policy === "LFU") { into = 0; for (let s = 1; s < k; s++) { c.add("scan"); const a = meta[s], b = meta[into]; if (a.f < b.f || (a.f === b.f && a.t < b.t)) into = s; } }
          else if (policy === "FWF") {       // flush when full: empty every slot, restart
            for (let s = 0; s < k; s++) { where.delete(slot[s]); c.add("flush"); }
            slot.length = 0; meta.length = 0; into = 0; slot.push(p); evicted = "all"; st.phases++;
          }
          else if (policy === "CLOCK") {
            for (;;) { c.add("scan"); if (meta[hand] === 1) { meta[hand] = 0; hand = (hand + 1) % k; } else { into = hand; hand = (hand + 1) % k; break; } }
          }
          else if (policy === "MARK") {
            let unmarked = []; for (let s = 0; s < k; s++) if (!meta[s]) unmarked.push(s);
            if (!unmarked.length) { for (let s = 0; s < k; s++) { meta[s] = 0; unmarked.push(s); } st.phases++; c.add("phase"); }
            into = unmarked[ri(r, unmarked.length)]; c.add("coin");
          }
          else if (policy === "OPT") {
            into = 0; let far = -1;
            for (let s = 0; s < k; s++) { c.add("scan"); const u = nextOf.get(slot[s]); if (u > far) { far = u; into = s; } if (u === Infinity) break; }
          }
          if (evicted !== "all") { evicted = slot[into]; where.delete(evicted); slot[into] = p; }
        }
        where.set(p, into);
        if (policy === "LRU" || policy === "FIFO") meta[into] = t;
        else if (policy === "LFU") meta[into] = { f: 1, t: t };
        else if (policy === "CLOCK" || policy === "MARK") meta[into] = 1;
        else if (policy === "FWF") meta[into] = 0;
        if (policy === "OPT") nextOf.set(p, nxt[t]);
      }
      return { req: p, hit: hit, into: into, evicted: evicted };
    };
    return st;
  }
  function simulate(policy, k, seq, seed, c, keepSnaps) {
    const cache = makeCache(policy, k, seed, policy === "OPT" ? nextUse(seq) : null, c);
    const snaps = keepSnaps ? [] : null;
    for (let t = 0; t < seq.length; t++) {
      const o = cache.access(seq[t], t);
      if (snaps) { o.slot = cache.slot.slice(); snaps.push(o); }
    }
    return { faults: cache.faults, snaps: snaps, phases: cache.phases };
  }
  /* request patterns; "adv" builds the sequence against a DETERMINISTIC policy
     by always requesting the one page (of k+1) that the policy does not hold */
  function pattern(kind, k, L, seed, advPolicy) {
    const r = rnd(seed), s = [];
    if (kind === "uniform") { const N = 2 * k; for (let i = 0; i < L; i++) s.push(ri(r, N)); }
    else if (kind === "zipf") {
      const N = 5 * k, w = []; let tot = 0;
      for (let i = 1; i <= N; i++) { tot += 1 / i; w.push(tot); }
      for (let i = 0; i < L; i++) { const u = r() * tot; let j = 0; while (w[j] < u) j++; s.push(j); }
    }
    else if (kind === "phases") {             // a working set of ≈ k pages that drifts
      let base = 0;
      for (let i = 0; i < L; i++) { if (i % 40 === 39) base += Math.max(1, Math.floor(k / 2)); s.push(base + ri(r, k)); }
    }
    else if (kind === "loop") { for (let i = 0; i < L; i++) s.push(i % (k + 1)); }
    else if (kind === "scan") {               // a hot set of k−1 pages, interrupted by one-shot scans
      let fresh = 1000;
      for (let i = 0; i < L; i++) {
        if (i % 60 >= 40) s.push(fresh++); else s.push(ri(r, Math.max(1, k - 1)));
      }
    }
    else if (kind === "adv") {
      const pol = advPolicy === "MARK" || advPolicy === "OPT" ? "LRU" : advPolicy;
      const cache = makeCache(pol, k, 1, null, null);
      for (let i = 0; i < L; i++) {
        let q = 0; while (cache.where.has(q)) q++;      // the one page of 0…k it does not hold
        s.push(q); cache.access(q, i);
      }
    }
    return s;
  }

  /* ── §09  BELADY'S ANOMALY: FIFO faults as a function of k ─────────────── */
  function faultCurve(policy, seq, kMax, c) {
    const out = [];
    for (let k = 1; k <= kMax; k++) out.push(simulate(policy, k, seq, 1, c).faults);
    return out;
  }
  /* search a seeded family of random strings for one on which FIFO is anomalous */
  function findAnomaly(nPages, len, seed, tries, c) {
    c = ctr(c);
    const r = rnd(seed);
    for (let t = 0; t < tries; t++) {
      c.add("tried");
      const s = []; for (let i = 0; i < len; i++) s.push(ri(r, nPages));
      const f = faultCurve("FIFO", s, nPages, null);
      for (let k = 1; k < f.length; k++) if (f[k] > f[k - 1]) return { seq: s, tried: t + 1 };
    }
    return null;
  }

  /* ── §11  LIST UPDATE ───────────────────────────────────────────────────
     Accessing the item at position i (1-based) costs i. Free exchanges move the
     accessed item forward. MTF = move to front; TRANS = swap with predecessor;
     FC = keep ordered by access count; STATIC = never move (the offline list
     ordered by true frequency — the best static list). */
  function listServe(policy, n, seq, init, c) {
    c = ctr(c);
    const L = init.slice(), cnt = new Array(n).fill(0);
    const costs = [];
    for (let t = 0; t < seq.length; t++) {
      const x = seq[t]; let i = 0;
      while (L[i] !== x) { i++; c.add("probe"); }
      c.add("probe");
      costs.push(i + 1);
      cnt[x]++;
      if (policy === "MTF") { L.splice(i, 1); L.unshift(x); }
      else if (policy === "TRANS") { if (i > 0) { L[i] = L[i - 1]; L[i - 1] = x; } }
      else if (policy === "FC") { let j = i; while (j > 0 && cnt[L[j - 1]] < cnt[x]) { L[j] = L[j - 1]; j--; } L[j] = x; }
    }
    return { costs: costs, final: L };
  }
  /* inversions between two lists = the potential Φ of the proof */
  function inversions(A, B) {
    const pos = new Map(); B.forEach((v, i) => pos.set(v, i));
    let inv = 0;
    for (let i = 0; i < A.length; i++) for (let j = i + 1; j < A.length; j++) if (pos.get(A[i]) > pos.get(A[j])) inv++;
    return inv;
  }
  /* per-step check of the amortised inequality  cost_MTF + ΔΦ ≤ 2·cost_A − 1
     for a STATIC offline list A (no exchanges). Returns counts of steps checked
     and steps that held, plus the running sums. */
  function mtfPotentialCheck(n, seq, init, staticList, c) {
    c = ctr(c);
    const M = init.slice(); let phi = inversions(M, staticList);
    const posA = new Map(); staticList.forEach((v, i) => posA.set(v, i + 1));
    let held = 0, sumM = 0, sumA = 0; const phi0 = phi;
    for (let t = 0; t < seq.length; t++) {
      const x = seq[t], k = M.indexOf(x) + 1, i = posA.get(x);
      M.splice(k - 1, 1); M.unshift(x);
      const phiNew = inversions(M, staticList); c.add("check");
      if (k + (phiNew - phi) <= 2 * i - 1) held++;
      sumM += k; sumA += i; phi = phiNew;
    }
    return { held: held, steps: seq.length, sumM: sumM, sumA: sumA, phi0: phi0, phiEnd: phi };
  }
  function listPattern(kind, n, m, seed) {
    const r = rnd(seed), s = [];
    if (kind === "zipf") {
      const w = []; let tot = 0; const lab = fyPerm(n, r);
      for (let i = 1; i <= n; i++) { tot += 1 / i; w.push(tot); }
      for (let t = 0; t < m; t++) { const u = r() * tot; let j = 0; while (w[j] < u) j++; s.push(lab[j]); }
    } else if (kind === "bursty") {           // runs of repeated items — locality MTF exploits
      while (s.length < m) { const x = ri(r, n), len = 1 + ri(r, 8); for (let q = 0; q < len && s.length < m; q++) s.push(x); }
    } else if (kind === "uniform") { for (let t = 0; t < m; t++) s.push(ri(r, n)); }
    else if (kind === "transbad") {           // alternate the two LAST items: TRANS swaps them forever at the back
      for (let t = 0; t < m; t++) s.push(t % 2 === 0 ? n - 1 : n - 2);
    }
    else if (kind === "cyclic") { for (let t = 0; t < m; t++) s.push((n - 1) - (t % n)); }
    return s;
  }
  function staticByFreq(n, seq) {
    const f = new Array(n).fill(0); seq.forEach(x => f[x]++);
    return Array.from({ length: n }, (_, i) => i).sort((a, b) => f[b] - f[a] || a - b);
  }

  /* ── §14  HIRING ─────────────────────────────────────────────────────── */
  function hireRun(rank, c) {               // rank: candidate qualities in interview order
    c = ctr(c); let best = -1, hires = 0;
    for (let i = 0; i < rank.length; i++) { c.add("interview"); if (rank[i] > best) { best = rank[i]; hires++; c.add("hire"); } }
    return hires;
  }
  /* ── §15  SECRETARY: observe the first `cut`, then take the first record ─ */
  function secretaryRun(rank, cut, c) {
    c = ctr(c); const n = rank.length; let best = -1;
    for (let i = 0; i < cut; i++) { c.add("observe"); if (rank[i] > best) best = rank[i]; }
    for (let i = cut; i < n; i++) { c.add("observe"); if (rank[i] > best) return rank[i] === n - 1; }
    return rank[n - 1] === n - 1;   // no record appeared: forced to take the last candidate
  }
  /* exact success probability of cutoff r (observe r, then first record) */
  function secretaryExact(n, r) {
    if (r === 0) return 1 / n;
    let s = 0; for (let i = r + 1; i <= n; i++) s += 1 / (i - 1);
    return (r / n) * s;
  }

  /* ── §17  KARGER'S CONTRACTION ──────────────────────────────────────── */
  function kargerRun(n, edges, r, c) {
    c = ctr(c);
    const par = Array.from({ length: n }, (_, i) => i);
    const find = x => { while (par[x] !== x) { par[x] = par[par[x]]; x = par[x]; } return x; };
    let comps = n;
    const E = edges.slice();
    while (comps > 2) {
      const i = ri(r, E.length); c.add("pick");
      const a = find(E[i][0]), b = find(E[i][1]);
      if (a === b) { E[i] = E[E.length - 1]; E.pop(); continue; }   // a self-loop: discard
      par[a] = b; comps--; c.add("contract");
    }
    let cut = 0; for (const e of E) if (find(e[0]) !== find(e[1])) cut++;
    return cut;
  }
  function minCutBrute(n, edges, c) {
    c = ctr(c); let best = Infinity, count = 0;
    for (let mask = 1; mask < (1 << (n - 1)); mask++) {          // vertex n−1 always on side 0
      c.add("subset");
      let cut = 0; for (const e of edges) if (((mask >> e[0]) & 1) !== ((mask >> e[1]) & 1)) cut++;
      if (cut < best) { best = cut; count = 1; } else if (cut === best) count++;
    }
    return { value: best, count: count };
  }
  function kGraph(kind, seed) {
    const r = rnd(seed), E = [];
    if (kind === "clusters") {                  // two K5's joined by two edges
      for (let a = 0; a < 5; a++) for (let b = a + 1; b < 5; b++) { E.push([a, b]); E.push([a + 5, b + 5]); }
      E.push([0, 5]); E.push([2, 8]);
      return { n: 10, edges: E };
    }
    if (kind === "cycle") { for (let i = 0; i < 10; i++) E.push([i, (i + 1) % 10]); return { n: 10, edges: E }; }
    /* random connected graph: a random spanning tree plus extra edges */
    const n = 10, have = new Set();
    const add = (a, b) => { const k = a < b ? a * 16 + b : b * 16 + a; if (a === b || have.has(k)) return; have.add(k); E.push([a, b]); };
    const p = fyPerm(n, r); for (let i = 1; i < n; i++) add(p[i], p[ri(r, i)]);
    while (E.length < 20) add(ri(r, n), ri(r, n));
    return { n: n, edges: E };
  }

  /* ── §19–20  TAILS, BALLS AND BINS ─────────────────────────────────── */
  function binomTail(n, a) {                // exact P(X ≥ a), X ~ Bin(n, 1/2)
    const lg = []; lg[0] = 0; for (let i = 1; i <= n; i++) lg[i] = lg[i - 1] + Math.log(i);
    let s = 0; for (let x = Math.ceil(a - 1e-9); x <= n; x++) s += Math.exp(lg[n] - lg[x] - lg[n - x] - n * Math.LN2);
    return Math.min(1, s);
  }
  function flips(n, r, c) { c = ctr(c); let h = 0; for (let i = 0; i < n; i++) { c.add("flip"); if (r() < 0.5) h++; } return h; }
  function coupon(n, r, c) {
    c = ctr(c); const seen = new Uint8Array(n); let got = 0, draws = 0;
    while (got < n) { draws++; c.add("draw"); const j = ri(r, n); if (!seen[j]) { seen[j] = 1; got++; } }
    return draws;
  }
  function firstCollision(m, r, c) {
    c = ctr(c); const seen = new Uint8Array(m); let t = 0;
    for (;;) { t++; c.add("draw"); const j = ri(r, m); if (seen[j]) return t; seen[j] = 1; }
  }
  function birthdayExact(m, n) { let p = 1; for (let i = 0; i < n; i++) p *= (m - i) / m; return 1 - p; }
  function maxLoad(n, d, r, c) {
    c = ctr(c); const load = new Uint32Array(n); let mx = 0;
    for (let b = 0; b < n; b++) {
      let best = ri(r, n); c.add("probe");
      for (let q = 1; q < d; q++) { const j = ri(r, n); c.add("probe"); if (load[j] < load[best]) best = j; }
      load[best]++; if (load[best] > mx) mx = load[best];
    }
    return mx;
  }

  /* ── §21  GENERATORS ───────────────────────────────────────────────── */
  function lcg(a, c0, m, seed) {           // exact integer LCG for m ≤ 2³¹ via BigInt-free splitting
    let x = seed % m;
    return function () {
      /* a·x can exceed 2⁵³; split a into 16-bit halves to stay exact */
      const ah = Math.floor(a / 65536), al = a % 65536;
      x = ((((ah * x) % m) * 65536) % m + al * x + c0) % m;
      return x;
    };
  }
  function period(next, cap, c) {           // length of the cycle the state sequence falls into
    c = ctr(c); const seen = new Map();
    for (let t = 0; t <= cap; t++) {
      const x = next(); c.add("step");
      if (seen.has(x)) return t - seen.get(x);
      seen.set(x, t);
    }
    return Infinity;
  }
  /* period of bit j of x_{k+1} = (a·x_k + c) mod 2³², measured on the sequence */
  function lowBitPeriod(a, c0, j, cap, seed, c) {
    c = ctr(c);
    let x = seed >>> 0; const bits = [];
    for (let t = 0; t < cap; t++) { x = (Math.imul(a, x) + c0) >>> 0; bits.push((x >>> j) & 1); c.add("step"); }
    for (let p = 1; p <= cap / 2; p++) {
      let ok = true; for (let t = 0; t + p < cap; t++) if (bits[t] !== bits[t + p]) { ok = false; break; }
      if (ok) return p;
    }
    return Infinity;
  }

  /* ── §22  SHUFFLES: exact position distribution by exhaustive enumeration ─ */
  /* returns M[i][j] = P(item i ends at position j), plus the per-permutation counts */
  function shuffleExact(kind, n, c) {
    c = ctr(c);
    const M = Array.from({ length: n }, () => new Array(n).fill(0));
    const perms = new Map(); let paths = 0;
    const a0 = Array.from({ length: n }, (_, i) => i);
    const finish = a => { paths++; c.add("path"); for (let p = 0; p < n; p++) M[a[p]][p]++; const key = a.join(""); perms.set(key, (perms.get(key) || 0) + 1); };
    /* the random choice at step i ranges over choices(i); recurse over all of them */
    const steps = kind === "naive" ? a0.map(i => ({ i: i, lo: 0, hi: n - 1 }))
                : kind === "sattolo" ? a0.slice(1).reverse().map(i => ({ i: i, lo: 0, hi: i - 1 }))
                : a0.slice(1).reverse().map(i => ({ i: i, lo: 0, hi: i }));      // Fisher–Yates
    (function rec(a, s) {
      if (s === steps.length) { finish(a); return; }
      const st = steps[s];
      for (let j = st.lo; j <= st.hi; j++) { const b = a.slice(); const t = b[st.i]; b[st.i] = b[j]; b[j] = t; rec(b, s + 1); }
    })(a0, 0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[i][j] /= paths;
    return { M: M, paths: paths, perms: perms };
  }
  /* sort-by-coin-flip comparator, run through a real merge sort: Monte Carlo */
  function shuffleBySort(n, trials, r, c) {
    c = ctr(c);
    const M = Array.from({ length: n }, () => new Array(n).fill(0));
    const perms = new Map();
    const msort = a => {
      if (a.length < 2) return a;
      const m = a.length >> 1, L = msort(a.slice(0, m)), R = msort(a.slice(m)), o = [];
      let i = 0, j = 0;
      while (i < L.length && j < R.length) { c.add("cmp"); if (r() < 0.5) o.push(L[i++]); else o.push(R[j++]); }
      while (i < L.length) o.push(L[i++]); while (j < R.length) o.push(R[j++]);
      return o;
    };
    for (let t = 0; t < trials; t++) {
      const a = msort(Array.from({ length: n }, (_, i) => i));
      for (let p = 0; p < n; p++) M[a[p]][p]++;
      const key = a.join(""); perms.set(key, (perms.get(key) || 0) + 1);
    }
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[i][j] /= trials;
    return { M: M, paths: trials, perms: perms };
  }

  /* ── §23  RESERVOIR SAMPLING ───────────────────────────────────────── */
  function reservoir(kind, N, k, r, c) {
    c = ctr(c); const R = [];
    for (let i = 0; i < N; i++) {
      c.add("item");
      if (i < k) { R.push(i); continue; }
      if (kind === "R") { const j = ri(r, i + 1); c.add("draw"); if (j < k) R[j] = i; }
      else if (kind === "half") { c.add("draw"); if (r() < 0.5) R[ri(r, k)] = i; }
      else if (kind === "fixed") { c.add("draw"); if (r() < k / N) R[ri(r, k)] = i; }
    }
    return R;
  }

  return {
    rnd, ri, harmonic, fyPerm,
    skiCost, skiOpt, skiWorst, skiDist, skiRandExpected,
    POLICIES, simulate, pattern, faultCurve, findAnomaly,
    listServe, inversions, mtfPotentialCheck, listPattern, staticByFreq,
    hireRun, secretaryRun, secretaryExact,
    kargerRun, minCutBrute, kGraph,
    binomTail, flips, coupon, firstCollision, birthdayExact, maxLoad,
    lcg, period, lowBitPeriod,
    shuffleExact, shuffleBySort, reservoir
  };
})();
if (typeof module !== "undefined") module.exports = OR;

/* ═══════════════════════════════════════════════════════════════════════════
   OX — DOM helpers shared by the figures
   ═══════════════════════════════════════════════════════════════════════════ */
const OX = {
  int: v => (typeof d3 !== "undefined" ? d3.format(",")(v) : String(v)),
  f2: v => (+v).toFixed(2), f3: v => (+v).toFixed(3), f4: v => (+v).toFixed(4),
  flag: ok => ok ? '<span style="color:' + AC.good + '">✓ holds</span>' : '<span style="color:' + AC.bad + '">✗ VIOLATED</span>',
  has: id => typeof d3 !== "undefined" && typeof document !== "undefined" && !d3.select("#" + id).empty(),
  on: (id, ev, fn) => { const s = d3.select("#" + id); if (!s.empty()) s.on(ev, fn); },
  val: (id, dflt) => { const s = d3.select("#" + id); return s.empty() ? dflt : s.property("value"); },
  setText: (id, t) => { const s = d3.select("#" + id); if (!s.empty()) s.text(t); },
  setHtml: (id, t) => { const s = d3.select("#" + id); if (!s.empty()) s.html(t); },
  txt: (g, x, y, t, o) => {
    o = o || {};
    return g.append("text").attr("x", x).attr("y", y).attr("font-size", o.size || 11)
      .attr("fill", o.fill || AC.muted).attr("text-anchor", o.anchor || "start").text(t);
  },
  line: (g, data, x, y, color, dash, w) => {
    const p = g.append("path").datum(data).attr("fill", "none").attr("stroke", color)
      .attr("stroke-width", w || 2).attr("d", d3.line().x(d => x(d[0])).y(d => y(d[1])));
    if (dash) p.attr("stroke-dasharray", dash);
    return p;
  },
  hline: (g, x0, x1, yv, color, dash) => g.append("line").attr("x1", x0).attr("x2", x1).attr("y1", yv).attr("y2", yv)
    .attr("stroke", color).attr("stroke-width", 1.4).attr("stroke-dasharray", dash || "5 4")
};
const OR_POLICY_COLOR = typeof AC === "undefined" ? {} : { LRU: AC.accent, FIFO: AC.a2, LFU: AC.rose, FWF: AC.muted, CLOCK: AC.teal, MARK: AC.violet, OPT: AC.good };
const OR_POLICY_NAME = { LRU: "LRU", FIFO: "FIFO", LFU: "LFU", FWF: "FWF (flush when full)", CLOCK: "CLOCK (second chance)", MARK: "Marker (randomised)", OPT: "Belady (offline OPT)" };

/* ── 04  #ski-svg  the rent-or-buy cost ratio, and the adversary's stop day ── */
(function () {
  if (!OX.has("ski-svg")) return;
  const W = 720, H = 360;
  function build() {
    const B = +OX.val("ski-B", 10), mode = OX.val("ski-mode", "ratio");
    const t = Math.min(+OX.val("ski-t", 10), 3 * B);
    OX.setText("ski-B-val", B); OX.setText("ski-t-val", t);
    const F = AL.frame("#ski-svg", W, H, { l: 52, r: 190, t: 20, b: 42 });
    const c = AL.counter(), cb = c.bump("eval");
    const cc = { add: () => cb() };
    const dMax = 3 * B;
    const p = OR.skiDist(B);
    const worstT = OR.skiWorst(t, B, 4 * B, cc);
    const worstBE = OR.skiWorst(B, B, 4 * B, cc);
    let randWorst = 0, randBest = Infinity;
    for (let d = 1; d <= 4 * B; d++) { const r = OR.skiRandExpected(d, B, p, cc) / OR.skiOpt(d, B); randWorst = Math.max(randWorst, r); randBest = Math.min(randBest, r); }
    const ee = Math.E / (Math.E - 1), disc = 1 / (1 - Math.pow(1 - 1 / B, B));

    if (mode === "dist") {
      const x = d3.scaleBand().domain(d3.range(1, B + 1)).range([0, F.iw]).padding(0.15);
      const y = d3.scaleLinear().domain([0, d3.max(p) * 1.15]).range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 5);
      F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x).tickValues(x.domain().filter(i => B <= 20 || i % 5 === 0 || i === 1)));
      AL.axisL(F.g, y, 5, "P(buy on day i)", d3.format(".2f"));
      OX.txt(F.g, F.iw, F.ih + 34, "buy day i", { anchor: "end" });
      p.forEach((v, i) => F.g.append("rect").attr("x", x(i + 1)).attr("y", y(v)).attr("width", x.bandwidth()).attr("height", F.ih - y(v)).attr("rx", 2).attr("fill", AC.violet));
      AL.legend(F.g, [{ label: "P(buy on day i) ∝ ((B−1)/B)^(B−i)", color: AC.violet }], F.iw + 12, 20);
      OX.txt(F.g, F.iw + 12, 60, `expected ratio, every d:`); OX.txt(F.g, F.iw + 12, 76, `min ${OX.f4(randBest)} · max ${OX.f4(randWorst)}`, { fill: AC.ink });
      OX.setHtml("ski-readout", `the randomised rule for B = ${B}: it buys on day i with probability proportional to ((B−1)/B)^(B−i), so it is most likely to buy late and least likely to buy on day 1 · evaluated against every stop day d = 1…${4 * B}, its expected ratio E[ALG(d)]/OPT(d) ranges only from <b>${OX.f4(randBest)}</b> to <b>${OX.f4(randWorst)}</b> — it is an <i>equaliser</i>: the adversary gains nothing by choosing d · that common value is 1/(1 − (1 − 1/B)^B) = ${OX.f4(disc)} ${OX.flag(Math.abs(randWorst - disc) < 1e-9)} and tends to e/(e − 1) = ${OX.f4(ee)} as B grows · ${OX.int(c.get("eval"))} cost evaluations`);
      return;
    }

    if (mode === "worst") {
      const ts = d3.range(1, 3 * B + 1);
      const pts = ts.map(tt => [tt, OR.skiWorst(tt, B, 4 * B, cc).ratio]);
      const x = d3.scaleLinear().domain([1, 3 * B]).range([0, F.iw]);
      const y = d3.scaleLinear().domain([1, Math.min(B + 1, d3.max(pts, q => q[1]) * 1.05)]).range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 6); AL.axisB(F.g, x, F.ih, 8, "the day t on which the deterministic rule buys"); AL.axisL(F.g, y, 6, "worst-case ratio over every stop day");
      OX.line(F.g, pts.filter(q => q[1] <= y.domain()[1]), x, y, AC.accent, null, 2.4);
      OX.hline(F.g, 0, F.iw, y(randWorst), AC.violet);
      const best = pts.reduce((a, b) => b[1] < a[1] ? b : a);
      F.g.append("circle").attr("cx", x(best[0])).attr("cy", y(best[1])).attr("r", 6).attr("fill", AC.good);
      OX.txt(F.g, x(best[0]) + 8, y(best[1]) - 8, `t = ${best[0]}: ${OX.f4(best[1])}`, { fill: AC.good });
      if (t <= 3 * B) { const q = pts[t - 1]; if (q[1] <= y.domain()[1]) F.g.append("circle").attr("cx", x(t)).attr("cy", y(q[1])).attr("r", 4.5).attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 2); }
      AL.legend(F.g, [{ label: "worst ratio of 'buy on day t'", color: AC.accent }, { label: "randomised rule (expected)", color: AC.violet, dash: "5 4" }, { label: "best deterministic t", color: AC.good }], F.iw + 12, 20);
      OX.setHtml("ski-readout", `for every buy day t = 1…${3 * B} the adversary's best stop day was found by trying every d ≤ ${4 * B} · the minimum worst-case ratio is at <b>t = ${best[0]}</b> (= B, the break-even day) with ratio <b>${OX.f4(best[1])}</b> = 2 − 1/B = ${OX.f4(2 - 1 / B)} ${OX.flag(best[0] === B && Math.abs(best[1] - (2 - 1 / B)) < 1e-9)} · no deterministic rule does better — the curve IS the lower-bound argument, evaluated · the randomised rule's worst expected ratio is ${OX.f4(randWorst)} · ${OX.int(c.get("eval"))} cost evaluations`);
      return;
    }

    /* mode "ratio": ALG/OPT as a function of the stop day */
    const ds = d3.range(1, dMax + 1);
    const cur = ds.map(d => [d, OR.skiCost(t, d, B, cc) / OR.skiOpt(d, B)]);
    const be = ds.map(d => [d, OR.skiCost(B, d, B, cc) / OR.skiOpt(d, B)]);
    const rd = ds.map(d => [d, OR.skiRandExpected(d, B, p, cc) / OR.skiOpt(d, B)]);
    const ymax = Math.max(2.1, d3.max(cur, q => q[1]) * 1.05);
    const x = d3.scaleLinear().domain([1, dMax]).range([0, F.iw]);
    const y = d3.scaleLinear().domain([0.9, ymax]).range([F.ih, 0]);
    AL.gridY(F.g, y, F.iw, 6); AL.axisB(F.g, x, F.ih, 8, "d = the day the season ends (chosen by the adversary)"); AL.axisL(F.g, y, 6, "cost ratio  ALG / OPT");
    F.g.append("line").attr("x1", x(B)).attr("x2", x(B)).attr("y1", 0).attr("y2", F.ih).attr("stroke", AC.line).attr("stroke-dasharray", "3 3");
    OX.txt(F.g, x(B) + 4, 12, "d = B");
    OX.line(F.g, be, x, y, AC.good, "6 4", 2);
    OX.line(F.g, rd, x, y, AC.violet, null, 2);
    OX.line(F.g, cur, x, y, AC.accent, null, 2.6);
    F.g.append("circle").attr("cx", x(Math.min(worstT.d, dMax))).attr("cy", y(worstT.ratio)).attr("r", 6).attr("fill", AC.bad);
    OX.txt(F.g, x(Math.min(worstT.d, dMax)) + 8, y(worstT.ratio) - 6, `adversary stops on day ${worstT.d}`, { fill: AC.bad });
    AL.legend(F.g, [{ label: `buy on day t = ${t}`, color: AC.accent }, { label: "break-even (t = B)", color: AC.good, dash: "6 4" }, { label: "randomised rule (expected)", color: AC.violet }], F.iw + 12, 20);
    OX.setHtml("ski-readout", `B = ${B}, rule "rent ${t - 1} day(s), buy on day ${t}" · the adversary's best stop day is <b>d = ${worstT.d}</b> — the day the rule buys — giving ALG = ${OR.skiCost(t, worstT.d, B)} against OPT = ${OR.skiOpt(worstT.d, B)}, ratio <b>${OX.f4(worstT.ratio)}</b> · break-even (t = B) worst ratio ${OX.f4(worstBE.ratio)} = 2 − 1/B ${OX.flag(Math.abs(worstBE.ratio - (2 - 1 / B)) < 1e-9)} · randomised rule's worst expected ratio ${OX.f4(randWorst)} (limit e/(e−1) = ${OX.f4(ee)}) · ${OX.int(c.get("eval"))} cost evaluations`);
  }
  ["ski-B", "ski-t"].forEach(id => OX.on(id, "input", build));
  OX.on("ski-mode", "change", build);
  build();
})();

/* ── 08  #paging-svg  the paging simulator: seven policies on one request string ── */
(function () {
  if (!OX.has("paging-svg")) return;
  const W = 720, H = 470, L = 300, SHOW = 36, MSEEDS = 30;
  let seed = 7;
  function build() {
    const pol = OX.val("paging-policy", "LRU"), k = +OX.val("paging-k", 4), pat = OX.val("paging-pattern", "zipf");
    OX.setText("paging-k-val", k);
    const seq = OR.pattern(pat, k, L, seed, pol);
    const svg = d3.select("#paging-svg"); svg.selectAll("*").remove();
    const res = {}, ctr = {};
    OR.POLICIES.forEach(P => {
      const c = AL.counter(), cc = { add: k2 => c.add(k2) };
      if (P === "MARK") {
        let tot = 0; for (let s = 1; s <= MSEEDS; s++) tot += OR.simulate(P, k, seq, s, cc).faults;
        res[P] = tot / MSEEDS;
      } else res[P] = OR.simulate(P, k, seq, 1, cc).faults;
      ctr[P] = c;
    });
    const opt = res.OPT;
    /* top: bars */
    const g = svg.append("g").attr("transform", "translate(170,16)");
    const bw = 400, x = d3.scaleLinear().domain([0, L]).range([0, bw]);
    OR.POLICIES.forEach((P, i) => {
      const yy = i * 21;
      g.append("text").attr("x", -8).attr("y", yy + 13).attr("text-anchor", "end").attr("font-size", 11)
        .attr("fill", P === pol ? AC.ink : AC.muted).attr("font-weight", P === pol ? 700 : 400).text(OR_POLICY_NAME[P]);
      g.append("rect").attr("x", 0).attr("y", yy + 2).attr("width", bw).attr("height", 15).attr("rx", 3).attr("fill", AC.panel2);
      g.append("rect").attr("x", 0).attr("y", yy + 2).attr("width", x(res[P])).attr("height", 15).attr("rx", 3).attr("fill", OR_POLICY_COLOR[P]).attr("opacity", P === pol || P === "OPT" ? 1 : 0.7);
      g.append("text").attr("x", x(res[P]) + 6).attr("y", yy + 13).attr("font-size", 11).attr("fill", AC.ink)
        .text(`${P === "MARK" ? res[P].toFixed(1) : res[P]} faults · ×${opt > 0 ? (res[P] / opt).toFixed(2) : "—"} OPT`);
    });
    OX.txt(g, 0, 7 * 21 + 12, `${L} requests · faults include the first ${k} cold misses · Marker averaged over ${MSEEDS} seeds`);
    /* bottom: the timeline of the chosen policy against OPT */
    const snaps = OR.simulate(pol, k, seq, 1, null, true).snaps;
    const optS = OR.simulate("OPT", k, seq, 1, null, true).snaps;
    const top = 190, cw = 15.5, ch = Math.min(18, Math.floor(230 / (k + 2))), gx = 84;
    const tg = svg.append("g").attr("transform", `translate(${gx},${top})`);
    OX.txt(tg, -8, ch - 4, "request", { anchor: "end" });
    for (let s = 0; s < k; s++) OX.txt(tg, -8, (s + 2) * ch - 4, `slot ${s + 1}`, { anchor: "end" });
    OX.txt(tg, -8, (k + 2) * ch + 10, "OPT fault", { anchor: "end" });
    for (let t = 0; t < SHOW && t < snaps.length; t++) {
      const sn = snaps[t], xx = t * cw;
      tg.append("rect").attr("x", xx).attr("y", 0).attr("width", cw - 1.5).attr("height", ch - 2).attr("rx", 2).attr("fill", sn.hit ? AC.panel2 : AC.bad).attr("opacity", sn.hit ? 1 : 0.85);
      tg.append("text").attr("x", xx + cw / 2 - 0.7).attr("y", ch - 6).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", sn.hit ? AC.ink : AC.bg).text(sn.req > 999 ? "s" : sn.req);
      for (let s = 0; s < k; s++) {
        const v = sn.slot[s], changed = !sn.hit && (s === sn.into || sn.evicted === "all");
        tg.append("rect").attr("x", xx).attr("y", (s + 1) * ch).attr("width", cw - 1.5).attr("height", ch - 2).attr("rx", 2)
          .attr("fill", changed ? OR_POLICY_COLOR[pol] : (sn.hit && s === sn.into ? "#24324a" : AC.panel)).attr("stroke", AC.grid);
        if (v !== undefined) tg.append("text").attr("x", xx + cw / 2 - 0.7).attr("y", (s + 2) * ch - 6).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", changed ? AC.bg : AC.ink).text(v > 999 ? "s" : v);
      }
      if (!optS[t].hit) tg.append("circle").attr("cx", xx + cw / 2 - 0.7).attr("cy", (k + 2) * ch + 6).attr("r", 3.5).attr("fill", AC.good);
    }
    OX.txt(tg, 0, (k + 2) * ch + 30, `first ${SHOW} requests served by ${OR_POLICY_NAME[pol]}: a red request is a fault, the coloured slot is where the page went${pol === "FWF" ? " (FWF empties every slot)" : ""}; "s" = a one-shot scan page · green dots are Belady's faults on the same requests`);
    const conservative = ["LRU", "FIFO", "FWF", "CLOCK"];
    const checks = conservative.map(P => `${P} ${res[P]} ≤ k·OPT + k = ${k * opt + k} ${OX.flag(res[P] <= k * opt + k)}`).join(" · ");
    const optMin = OR.POLICIES.every(P => res[P] >= opt - 1e-9);
    OX.setHtml("paging-readout", `k = ${k}, pattern "${pat}"${pat === "adv" ? ` (built against ${pol === "MARK" || pol === "OPT" ? "LRU — an oblivious adversary cannot see the marker's coins" : pol})` : ""} · Belady's furthest-in-future: <b>${opt}</b> faults, no policy beat it ${OX.flag(optMin)} · ${OR_POLICY_NAME[pol]}: <b>${pol === "MARK" ? res[pol].toFixed(1) : res[pol]}</b> faults, measured ratio <b>${(res[pol] / opt).toFixed(3)}</b> · the k-competitive bound for the conservative policies: ${checks} · Marker's bound 2H_k·OPT = ${(2 * OR.harmonic(k) * opt).toFixed(1)}; measured ${res.MARK.toFixed(1)} · ${OX.int(ctr[pol].get("scan") + ctr[pol].get("coin"))} eviction scans/coins for ${pol}`);
  }
  OX.on("paging-policy", "change", build); OX.on("paging-pattern", "change", build); OX.on("paging-k", "input", build);
  OX.on("paging-seed", "click", () => { seed = seed * 31 % 997 + 1; build(); });
  build();
})();

/* ── 09  #anomaly-svg  faults against cache size: FIFO can go UP ── */
(function () {
  if (!OX.has("anomaly-svg")) return;
  const W = 720, H = 330;
  const found = OR.findAnomaly(5, 14, 11, 20000);
  function build() {
    const which = OX.val("anomaly-seq", "classic");
    let seq, name;
    if (which === "classic") { seq = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]; name = "1 2 3 4 1 2 5 1 2 3 4 5"; }
    else if (which === "found") { seq = found ? found.seq : [0]; name = seq.join(" ") + (found ? ` (the first anomalous string among ${found.tried} seeded random strings)` : ""); }
    else { seq = OR.pattern("zipf", 3, 120, 5); name = "120 Zipf-skewed requests over 15 pages"; }
    const distinct = new Set(seq).size, kMax = Math.min(distinct, 8);
    const c = AL.counter(), cc = { add: k2 => c.add(k2) };
    const curves = { FIFO: OR.faultCurve("FIFO", seq, kMax, cc), LRU: OR.faultCurve("LRU", seq, kMax, cc), OPT: OR.faultCurve("OPT", seq, kMax, cc) };
    const F = AL.frame("#anomaly-svg", W, H, { l: 52, r: 170, t: 20, b: 42 });
    const x = d3.scaleLinear().domain([1, kMax]).range([0, F.iw]);
    const y = d3.scaleLinear().domain([0, d3.max(curves.FIFO.concat(curves.LRU)) * 1.08]).range([F.ih, 0]);
    AL.gridY(F.g, y, F.iw, 6); AL.axisB(F.g, x, F.ih, kMax - 1, "cache size k (frames)", d3.format("d")); AL.axisL(F.g, y, 6, "page faults");
    const anomalies = [];
    [["OPT", AC.good, "3 3"], ["LRU", AC.accent, null], ["FIFO", AC.a2, null]].forEach(([P, col, dash]) => {
      const pts = curves[P].map((f, i) => [i + 1, f]);
      OX.line(F.g, pts, x, y, col, dash, 2.4);
      pts.forEach((q, i) => {
        const up = i > 0 && q[1] > pts[i - 1][1];
        if (up) anomalies.push(`${P}: k = ${i} → ${i + 1} faults ${pts[i - 1][1]} → ${q[1]}`);
        F.g.append("circle").attr("cx", x(q[0])).attr("cy", y(q[1])).attr("r", up ? 7 : 3.5).attr("fill", up ? AC.bad : col);
      });
    });
    AL.legend(F.g, [{ label: "FIFO", color: AC.a2 }, { label: "LRU (a stack algorithm)", color: AC.accent }, { label: "Belady OPT (a stack algorithm)", color: AC.good, dash: "3 3" }, { label: "more frames, MORE faults", color: AC.bad }], F.iw + 12, 20);
    const mono = P => curves[P].every((f, i) => i === 0 || f <= curves[P][i - 1]);
    OX.setHtml("anomaly-readout", `requests: ${name} · FIFO faults for k = 1…${kMax}: [${curves.FIFO.join(", ")}] · LRU: [${curves.LRU.join(", ")}] · OPT: [${curves.OPT.join(", ")}] · ${anomalies.length ? "<b>anomaly</b> — " + anomalies.join("; ") : "no anomaly on this string"} · LRU non-increasing in k ${OX.flag(mono("LRU"))}, OPT non-increasing ${OX.flag(mono("OPT"))} · ${OX.int(c.get("req"))} requests simulated`);
  }
  OX.on("anomaly-seq", "change", build);
  build();
})();

/* ── 10  #marker-svg  deterministic k against randomised H_k, measured ── */
(function () {
  if (!OX.has("marker-svg")) return;
  const W = 720, H = 340, L = 600, SEEDS = 16;
  function build() {
    const pat = OX.val("marker-pattern", "loop");
    const ks = d3.range(2, 17);
    const c = AL.counter(), cc = { add: k2 => c.add(k2) };
    const rows = ks.map(k => {
      const seq = pat === "loop" ? OR.pattern("loop", k, L, 1) : (function () { const r = OR.rnd(100 + k), s = []; for (let i = 0; i < L; i++) s.push(OR.ri(r, k + 1)); return s; })();
      const opt = OR.simulate("OPT", k, seq, 1, cc).faults - k, lru = OR.simulate("LRU", k, seq, 1, cc).faults - k;
      let m = 0; for (let s = 1; s <= SEEDS; s++) m += OR.simulate("MARK", k, seq, s, cc).faults - k;
      m /= SEEDS;
      return { k: k, lru: lru / opt, mark: m / opt, H: OR.harmonic(k) };
    });
    const F = AL.frame("#marker-svg", W, H, { l: 52, r: 190, t: 20, b: 42 });
    const x = d3.scaleLinear().domain([2, 16]).range([0, F.iw]);
    const y = d3.scaleLinear().domain([0, 17]).range([F.ih, 0]);
    AL.gridY(F.g, y, F.iw, 6); AL.axisB(F.g, x, F.ih, 14, "cache size k", d3.format("d")); AL.axisL(F.g, y, 6, "faults ÷ OPT faults (after warm-up)");
    OX.line(F.g, rows.map(r => [r.k, r.k]), x, y, AC.muted, "3 3", 1.4);
    OX.line(F.g, rows.map(r => [r.k, 2 * r.H]), x, y, AC.violet, "6 4", 1.6);
    OX.line(F.g, rows.map(r => [r.k, r.H]), x, y, AC.good, "2 3", 1.6);
    OX.line(F.g, rows.map(r => [r.k, r.lru]), x, y, AC.accent, null, 2.4);
    OX.line(F.g, rows.map(r => [r.k, r.mark]), x, y, AC.violet, null, 2.4);
    rows.forEach(r => { F.g.append("circle").attr("cx", x(r.k)).attr("cy", y(r.lru)).attr("r", 3).attr("fill", AC.accent); F.g.append("circle").attr("cx", x(r.k)).attr("cy", y(r.mark)).attr("r", 3).attr("fill", AC.violet); });
    AL.legend(F.g, [{ label: "LRU (measured)", color: AC.accent }, { label: "Marker (measured, mean)", color: AC.violet }, { label: "k", color: AC.muted, dash: "3 3" }, { label: "2·H_k  (Marker's bound)", color: AC.violet, dash: "6 4" }, { label: "H_k  (randomised lower bound)", color: AC.good, dash: "2 3" }], F.iw + 12, 20);
    const last = rows[rows.length - 1];
    const within = rows.every(r => r.mark <= 2 * r.H + 0.05);
    OX.setHtml("marker-readout", `${pat === "loop" ? "the cyclic string 0, 1, …, k, 0, 1, … over k + 1 pages — the adversary's sequence for every deterministic policy" : "uniformly random requests over k + 1 pages"} · ${L} requests, the k cold misses subtracted from every count, Marker averaged over ${SEEDS} seeds · at k = 16: LRU <b>${last.lru.toFixed(2)}×</b> OPT, Marker <b>${last.mark.toFixed(2)}×</b> OPT, against H₁₆ = ${last.H.toFixed(3)} and 2H₁₆ = ${(2 * last.H).toFixed(3)} · Marker inside 2H_k at every k ${OX.flag(within)} · ${OX.int(c.get("req"))} requests simulated, ${OX.int(c.get("coin"))} coins flipped`);
  }
  OX.on("marker-pattern", "change", build);
  build();
})();

/* ── 11  #mtf-svg  list update: cumulative cost, and the potential-function bound ── */
(function () {
  if (!OX.has("mtf-svg")) return;
  const W = 720, H = 360, M = 400;
  function build() {
    const n = +OX.val("mtf-n", 10), pat = OX.val("mtf-pattern", "zipf");
    OX.setText("mtf-n-val", n);
    const seq = OR.listPattern(pat, n, M, 3), init = d3.range(n), st = OR.staticByFreq(n, seq);
    const c = AL.counter(), cc = { add: k2 => c.add(k2) };
    const runs = { MTF: OR.listServe("MTF", n, seq, init, cc), TRANS: OR.listServe("TRANS", n, seq, init, cc), FC: OR.listServe("FC", n, seq, init, cc), STATIC: OR.listServe("STATIC", n, seq, st, cc) };
    const chk = OR.mtfPotentialCheck(n, seq, init, st, cc);
    const cum = a => { let s = 0; return a.map((v, i) => [i + 1, (s += v)]); };
    const C = {}; Object.keys(runs).forEach(P => C[P] = cum(runs[P].costs));
    const bound = C.STATIC.map(q => [q[0], 2 * q[1] - q[0] + chk.phi0]);
    const F = AL.frame("#mtf-svg", W, H, { l: 60, r: 200, t: 20, b: 42 });
    const x = d3.scaleLinear().domain([1, M]).range([0, F.iw]);
    const ymax = d3.max([C.MTF, C.TRANS, C.FC, bound], a => a[a.length - 1][1]) * 1.05;
    const y = d3.scaleLinear().domain([0, ymax]).range([F.ih, 0]);
    AL.gridY(F.g, y, F.iw, 6); AL.axisB(F.g, x, F.ih, 8, "requests served"); AL.axisL(F.g, y, 6, "cumulative access cost", d3.format("~s"));
    const col = { MTF: AC.accent, TRANS: AC.rose, FC: AC.a2, STATIC: AC.good };
    OX.line(F.g, bound, x, y, AC.violet, "6 4", 1.8);
    Object.keys(C).forEach(P => OX.line(F.g, C[P], x, y, col[P], P === "STATIC" ? "3 3" : null, P === "MTF" ? 2.8 : 2));
    AL.legend(F.g, [{ label: "move-to-front", color: AC.accent }, { label: "transpose", color: AC.rose }, { label: "frequency count", color: AC.a2 }, { label: "best static list (offline)", color: AC.good, dash: "3 3" }, { label: "2·STATIC − m + Φ₀  (the bound)", color: AC.violet, dash: "6 4" }], F.iw + 12, 20);
    const tot = P => C[P][M - 1][1];
    OX.setHtml("mtf-readout", `${n} items, ${M} requests ("${pat}") · total access cost: MTF <b>${OX.int(tot("MTF"))}</b>, transpose ${OX.int(tot("TRANS"))}, frequency count ${OX.int(tot("FC"))}, best static list ${OX.int(tot("STATIC"))} · ratio MTF/STATIC = <b>${(tot("MTF") / tot("STATIC")).toFixed(3)}</b>, TRANS/STATIC = ${(tot("TRANS") / tot("STATIC")).toFixed(3)} · the amortised step inequality cost_MTF + ΔΦ ≤ 2·cost_A − 1 checked on every request: held on <b>${chk.held} / ${chk.steps}</b> ${OX.flag(chk.held === chk.steps)} · summed: ${OX.int(chk.sumM)} ≤ 2·${OX.int(chk.sumA)} − ${M} + Φ₀ − Φ_end = ${OX.int(2 * chk.sumA - M + chk.phi0 - chk.phiEnd)} ${OX.flag(chk.sumM <= 2 * chk.sumA - M + chk.phi0 - chk.phiEnd)} (Φ₀ = ${chk.phi0}, Φ_end = ${chk.phiEnd}) · ${OX.int(c.get("probe"))} list probes`);
  }
  OX.on("mtf-n", "input", build); OX.on("mtf-pattern", "change", build);
  build();
})();

/* ── 14  #hire-svg  the hiring problem: how many hires in a random order ── */
(function () {
  if (!OX.has("hire-svg")) return;
  const W = 720, H = 320, T = 2000;
  function build() {
    const n = +OX.val("hire-n", 100);
    OX.setText("hire-n-val", n);
    const r = AL.rng(2024 + n), c = AL.counter(), cc = { add: k2 => c.add(k2) };
    const counts = [];
    for (let t = 0; t < T; t++) counts.push(OR.hireRun(OR.fyPerm(n, r), cc));
    const worstC = AL.counter(); const worst = OR.hireRun(d3.range(n), { add: k2 => worstC.add(k2) });
    const Hn = OR.harmonic(n), mean = d3.mean(counts), mx = d3.max(counts);
    const hist = d3.range(1, mx + 1).map(h => [h, counts.filter(v => v === h).length / T]);
    const F = AL.frame("#hire-svg", W, H, { l: 52, r: 190, t: 20, b: 42 });
    const x = d3.scaleBand().domain(hist.map(q => q[0])).range([0, F.iw]).padding(0.12);
    const y = d3.scaleLinear().domain([0, d3.max(hist, q => q[1]) * 1.15]).range([F.ih, 0]);
    AL.gridY(F.g, y, F.iw, 5);
    F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x).tickValues(x.domain().filter(h => mx <= 20 || h % 2 === 1)));
    AL.axisL(F.g, y, 5, "fraction of runs", d3.format(".0%"));
    OX.txt(F.g, F.iw, F.ih + 34, "number of hires in one run", { anchor: "end" });
    hist.forEach(q => F.g.append("rect").attr("x", x(q[0])).attr("y", y(q[1])).attr("width", x.bandwidth()).attr("height", F.ih - y(q[1])).attr("rx", 2).attr("fill", AC.accent));
    const xs = v => x(1) + (v - 1) * x.step() + x.bandwidth() / 2;
    F.g.append("line").attr("x1", xs(Hn)).attr("x2", xs(Hn)).attr("y1", 0).attr("y2", F.ih).attr("stroke", AC.good).attr("stroke-width", 2).attr("stroke-dasharray", "5 4");
    F.g.append("line").attr("x1", xs(mean)).attr("x2", xs(mean)).attr("y1", 0).attr("y2", F.ih).attr("stroke", AC.a2).attr("stroke-width", 2);
    AL.legend(F.g, [{ label: `measured mean ${mean.toFixed(3)}`, color: AC.a2 }, { label: `Hₙ = ${Hn.toFixed(3)}`, color: AC.good, dash: "5 4" }], F.iw + 12, 20);
    OX.setHtml("hire-readout", `n = ${n} candidates in ${OX.int(T)} uniformly random orders · mean hires <b>${mean.toFixed(3)}</b> against E[X] = Hₙ = ${Hn.toFixed(3)} (ln n + 0.5772 = ${(Math.log(n) + 0.5772).toFixed(3)}) ${OX.flag(Math.abs(mean - Hn) < 0.12 * Math.sqrt(Hn) + 0.05)} · most hires in any run: ${mx} · the adversarial order (increasing quality) hires <b>${worst}</b> = n every time · ${OX.int(c.get("interview"))} interviews, ${OX.int(c.get("hire"))} hires counted`);
  }
  OX.on("hire-n", "input", build);
  build();
})();

/* ── 15  #sec-svg  the secretary rule: observe r, then take the first record ── */
(function () {
  if (!OX.has("sec-svg")) return;
  const W = 720, H = 340;
  function build() {
    const n = +OX.val("sec-n", 50);
    const cut = Math.min(+OX.val("sec-cut", 18), n - 1);
    OX.setText("sec-n-val", n); OX.setText("sec-cut-val", cut);
    const r = AL.rng(77 + n), c = AL.counter(), cc = { add: k2 => c.add(k2) };
    const exact = d3.range(0, n).map(k => [k, OR.secretaryExact(n, k)]);
    const grid = Array.from(new Set(d3.range(0, 21).map(i => Math.round(i * (n - 1) / 20))));
    const TR = 800;
    const mc = grid.map(k => { let w = 0; for (let t = 0; t < TR; t++) if (OR.secretaryRun(OR.fyPerm(n, r), k, cc)) w++; return [k, w / TR]; });
    let wCut = 0; const TC = 4000; for (let t = 0; t < TC; t++) if (OR.secretaryRun(OR.fyPerm(n, r), cut, cc)) wCut++;
    const best = exact.reduce((a, b) => b[1] > a[1] ? b : a);
    const F = AL.frame("#sec-svg", W, H, { l: 52, r: 190, t: 20, b: 42 });
    const x = d3.scaleLinear().domain([0, n - 1]).range([0, F.iw]);
    const y = d3.scaleLinear().domain([0, Math.max(0.5, d3.max(mc, q => q[1]) * 1.1)]).range([F.ih, 0]);
    AL.gridY(F.g, y, F.iw, 5); AL.axisB(F.g, x, F.ih, 8, "r = how many candidates are observed and rejected"); AL.axisL(F.g, y, 5, "P(the best candidate is chosen)", d3.format(".0%"));
    OX.hline(F.g, 0, F.iw, y(1 / Math.E), AC.muted, "3 3"); OX.txt(F.g, 4, y(1 / Math.E) - 4, "1/e");
    F.g.append("line").attr("x1", x(n / Math.E)).attr("x2", x(n / Math.E)).attr("y1", 0).attr("y2", F.ih).attr("stroke", AC.muted).attr("stroke-dasharray", "3 3");
    OX.txt(F.g, x(n / Math.E) + 4, 12, "n/e");
    OX.line(F.g, exact, x, y, AC.good, null, 2.2);
    mc.forEach(q => F.g.append("circle").attr("cx", x(q[0])).attr("cy", y(q[1])).attr("r", 3.5).attr("fill", AC.accent));
    F.g.append("circle").attr("cx", x(cut)).attr("cy", y(wCut / TC)).attr("r", 7).attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 2.5);
    AL.legend(F.g, [{ label: "exact  (r/n)·Σ 1/(i−1)", color: AC.good }, { label: `Monte Carlo, ${TR} runs/point`, color: AC.accent }, { label: `your cutoff, ${TC} runs`, color: AC.a2 }], F.iw + 12, 20);
    OX.setHtml("sec-readout", `n = ${n}, observe r = ${cut} then take the first candidate better than all of them · measured success <b>${(100 * wCut / TC).toFixed(1)}%</b> over ${OX.int(TC)} random orders, exact ${(100 * OR.secretaryExact(n, cut)).toFixed(1)}% · the best cutoff for this n is <b>r = ${best[0]}</b> (n/e = ${(n / Math.E).toFixed(1)}) with success ${(100 * best[1]).toFixed(2)}%, against the limit 1/e = 36.79% · ${OX.int(c.get("observe"))} candidates observed`);
  }
  OX.on("sec-n", "input", build); OX.on("sec-cut", "input", build);
  build();
})();

/* ── 17  #karger-svg  contraction, and amplification by repetition ── */
(function () {
  if (!OX.has("karger-svg")) return;
  const W = 720, H = 340, RUNS = 3000;
  let seed = 5;
  function build() {
    const kind = OX.val("karger-graph", "clusters");
    const G = OR.kGraph(kind, 3), n = G.n;
    const bc = AL.counter(), brute = OR.minCutBrute(n, G.edges, { add: k2 => bc.add(k2) });
    const r = AL.rng(seed), kc = AL.counter(), kk = { add: k2 => kc.add(k2) };
    const hits = []; for (let i = 0; i < RUNS; i++) hits.push(OR.kargerRun(n, G.edges, r, kk) === brute.value ? 1 : 0);
    const p = d3.mean(hits), pb = 2 / (n * (n - 1));
    const ts = d3.range(1, 41);
    const meas = ts.map(t => { const blocks = Math.floor(RUNS / t); let ok = 0; for (let b = 0; b < blocks; b++) { let any = 0; for (let j = 0; j < t; j++) any |= hits[b * t + j]; ok += any; } return [t, ok / blocks]; });
    const svg = d3.select("#karger-svg"); svg.selectAll("*").remove();
    /* left: the graph on a circle */
    const gg = svg.append("g").attr("transform", "translate(120,170)");
    const pos = i => { const a = kind === "clusters" ? (i < 5 ? Math.PI * (0.55 + 0.9 * i / 4) : Math.PI * (-0.45 + 0.9 * (i - 5) / 4)) : 2 * Math.PI * i / n - Math.PI / 2; const R = kind === "clusters" ? 55 : 90; const off = kind === "clusters" ? (i < 5 ? -40 : 40) : 0; return [off + R * Math.cos(a), R * Math.sin(a)]; };
    G.edges.forEach(e => { const a = pos(e[0]), b = pos(e[1]); gg.append("line").attr("x1", a[0]).attr("y1", a[1]).attr("x2", b[0]).attr("y2", b[1]).attr("stroke", AC.line).attr("stroke-width", 1.6); });
    d3.range(n).forEach(i => { const q = pos(i); gg.append("circle").attr("cx", q[0]).attr("cy", q[1]).attr("r", 9).attr("fill", AC.panel2).attr("stroke", AC.accent); gg.append("text").attr("x", q[0]).attr("y", q[1] + 4).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.ink).text(i); });
    OX.txt(gg, 0, 140, `n = ${n}, m = ${G.edges.length}, min cut ${brute.value} (${brute.count} min cut${brute.count > 1 ? "s" : ""})`, { anchor: "middle" });
    /* right: success of the best of t runs */
    const g = svg.append("g").attr("transform", "translate(300,20)"), iw = 400, ih = 270;
    const x = d3.scaleLinear().domain([1, 40]).range([0, iw]), y = d3.scaleLinear().domain([0, 1]).range([ih, 0]);
    AL.gridY(g, y, iw, 5); AL.axisB(g, x, ih, 8, "t = independent runs, keep the smallest cut"); AL.axisL(g, y, 5, "P(some run finds a minimum cut)", d3.format(".0%"));
    OX.line(g, ts.map(t => [t, 1 - Math.pow(1 - pb, t)]), x, y, AC.muted, "5 4", 1.8);
    OX.line(g, ts.map(t => [t, 1 - Math.pow(1 - p, t)]), x, y, AC.good, null, 2);
    meas.forEach(q => g.append("circle").attr("cx", x(q[0])).attr("cy", y(q[1])).attr("r", 3.2).attr("fill", AC.accent));
    AL.legend(g, [{ label: "measured (disjoint blocks of t runs)", color: AC.accent }, { label: "1 − (1 − p̂)ᵗ", color: AC.good }, { label: "guarantee 1 − (1 − 2/n(n−1))ᵗ", color: AC.muted, dash: "5 4" }], iw - 205, ih - 50);
    const need = Math.ceil(n * (n - 1) / 2 * Math.log(n));
    OX.setHtml("karger-readout", `exhaustive search over ${OX.int(bc.get("subset"))} bipartitions: min cut <b>${brute.value}</b>, attained by ${brute.count} cut(s) · ${OX.int(RUNS)} contraction runs: single-run success p̂ = <b>${p.toFixed(4)}</b> against the guarantee 2/(n(n−1)) = ${pb.toFixed(4)} ${OX.flag(p >= pb)} · the guarantee alone says C(n,2)·ln n = ${need} runs push the failure probability below 1/n; with the measured p̂ the same target needs only ${p > 0 ? Math.ceil(Math.log(n) / -Math.log(1 - Math.min(p, 0.999999))) : "∞"} · ${OX.int(kc.get("contract"))} contractions, ${OX.int(kc.get("pick"))} edge picks`);
  }
  OX.on("karger-graph", "change", build);
  OX.on("karger-reseed", "click", () => { seed = seed * 17 % 1009 + 2; build(); });
  build();
})();

/* ── 19  #tail-svg  Markov, Chebyshev, Chernoff against the exact binomial tail ── */
(function () {
  if (!OX.has("tail-svg")) return;
  const W = 720, H = 350, TR = 4000;
  function build() {
    const n = +OX.val("tail-n", 100), dl = +OX.val("tail-delta", 0.2);
    OX.setText("tail-n-val", n); OX.setText("tail-delta-val", dl.toFixed(2));
    const mu = n / 2;
    const r = AL.rng(31 + n), c = AL.counter(), cc = { add: k2 => c.add(k2) };
    const X = []; for (let t = 0; t < TR; t++) X.push(OR.flips(n, r, cc));
    const ds = AL.linspace(0.02, 1, 50);
    const curve = f => ds.map(d => [d, Math.max(1e-14, Math.min(1, f(d)))]);
    const exact = curve(d => OR.binomTail(n, (1 + d) * mu));
    const markov = curve(d => 1 / (1 + d));
    const cheb = curve(d => (n / 4) / Math.pow(d * mu, 2));
    const chern = curve(d => Math.exp(-d * d * mu / (2 + d)));
    const mc = ds.filter((d, i) => i % 3 === 0).map(d => [d, X.filter(v => v >= (1 + d) * mu - 1e-9).length / TR]).filter(q => q[1] > 0);
    const F = AL.frame("#tail-svg", W, H, { l: 58, r: 200, t: 20, b: 42 });
    const x = d3.scaleLinear().domain([0, 1]).range([0, F.iw]);
    const y = d3.scaleLog().domain([1e-14, 1.5]).range([F.ih, 0]).clamp(true);
    AL.gridY(F.g, y, F.iw, 7); AL.axisB(F.g, x, F.ih, 10, "d  (threshold (1 + d)·μ, μ = n/2)");
    F.g.append("g").attr("class", "axis").call(d3.axisLeft(y).tickValues([1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]).tickFormat(d3.format(".0e")));
    OX.txt(F.g, 0, -6, "P(X ≥ (1 + d)μ)  — log scale");
    OX.line(F.g, markov, x, y, AC.muted, "3 3", 1.8);
    OX.line(F.g, cheb, x, y, AC.a2, null, 2);
    OX.line(F.g, chern, x, y, AC.violet, null, 2);
    OX.line(F.g, exact, x, y, AC.good, null, 2.6);
    mc.forEach(q => F.g.append("circle").attr("cx", x(q[0])).attr("cy", y(q[1])).attr("r", 3.5).attr("fill", AC.accent));
    F.g.append("line").attr("x1", x(dl)).attr("x2", x(dl)).attr("y1", 0).attr("y2", F.ih).attr("stroke", AC.line).attr("stroke-dasharray", "3 3");
    AL.legend(F.g, [{ label: "exact binomial tail", color: AC.good }, { label: `Monte Carlo, ${TR} runs`, color: AC.accent }, { label: "Markov  1/(1+d)", color: AC.muted, dash: "3 3" }, { label: "Chebyshev  1/(d²n)", color: AC.a2 }, { label: "Chernoff  e^(−d²μ/(2+d))", color: AC.violet }], F.iw + 12, 20);
    const ex = OR.binomTail(n, (1 + dl) * mu), mk = 1 / (1 + dl), cb = Math.min(1, (n / 4) / Math.pow(dl * mu, 2)), ch = Math.exp(-dl * dl * mu / (2 + dl));
    const mcv = X.filter(v => v >= (1 + dl) * mu - 1e-9).length / TR;
    const valid = exact.every((q, i) => q[1] <= markov[i][1] + 1e-12 && q[1] <= cheb[i][1] + 1e-12 && q[1] <= chern[i][1] + 1e-12);
    const e = v => v < 1e-3 ? v.toExponential(2) : v.toFixed(4);
    OX.setHtml("tail-readout", `X = heads in n = ${n} fair flips, μ = ${mu} · at d = ${dl.toFixed(2)} (X ≥ ${((1 + dl) * mu).toFixed(1)}): exact <b>${e(ex)}</b>, measured ${e(mcv)} over ${OX.int(TR)} runs · Markov ${e(mk)} · Chebyshev ${e(cb)} · Chernoff <b>${e(ch)}</b> · every bound ≥ the exact tail at every d ${OX.flag(valid)} · Markov ignores n entirely, Chebyshev falls like 1/n, Chernoff like e^(−cn) · ${OX.int(c.get("flip"))} coin flips`);
  }
  OX.on("tail-n", "input", build); OX.on("tail-delta", "input", build);
  build();
})();

/* ── 20  #bins-svg  birthday, coupon collector, maximum load ── */
(function () {
  if (!OX.has("bins-svg")) return;
  const W = 720, H = 340;
  function build() {
    const mode = OX.val("bins-mode", "birthday"), m = +OX.val("bins-m", 365);
    OX.setText("bins-m-val", m);
    const c = AL.counter(), cc = { add: k2 => c.add(k2) };
    const F = AL.frame("#bins-svg", W, H, { l: 56, r: 200, t: 20, b: 42 });
    if (mode === "birthday") {
      const r = AL.rng(11 + m), TR = 3000, T = [];
      for (let t = 0; t < TR; t++) T.push(OR.firstCollision(m, r, cc));
      const nMax = Math.ceil(3.2 * Math.sqrt(m)) + 2, ns = d3.range(1, nMax + 1);
      const x = d3.scaleLinear().domain([1, nMax]).range([0, F.iw]), y = d3.scaleLinear().domain([0, 1]).range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 5); AL.axisB(F.g, x, F.ih, 8, "balls thrown (people in the room)"); AL.axisL(F.g, y, 5, "P(some bin holds two)", d3.format(".0%"));
      OX.line(F.g, ns.map(k => [k, 1 - Math.exp(-k * (k - 1) / (2 * m))]), x, y, AC.violet, "5 4", 1.8);
      OX.line(F.g, ns.map(k => [k, OR.birthdayExact(m, k)]), x, y, AC.good, null, 2.4);
      ns.filter(k => k % Math.max(1, Math.floor(nMax / 24)) === 0).forEach(k => F.g.append("circle").attr("cx", x(k)).attr("cy", y(T.filter(v => v <= k).length / TR)).attr("r", 3.2).attr("fill", AC.accent));
      let half = 1; while (OR.birthdayExact(m, half) < 0.5) half++;
      OX.hline(F.g, 0, F.iw, y(0.5), AC.muted, "3 3");
      AL.legend(F.g, [{ label: "exact 1 − Π(1 − i/m)", color: AC.good }, { label: "approx 1 − e^(−n(n−1)/2m)", color: AC.violet, dash: "5 4" }, { label: `measured, ${TR} runs`, color: AC.accent }], F.iw + 12, 20);
      OX.setHtml("bins-readout", `m = ${m} bins · the first collision came, on average, after <b>${d3.mean(T).toFixed(2)}</b> balls (√(πm/2) = ${Math.sqrt(Math.PI * m / 2).toFixed(2)}) · a collision is more likely than not from <b>n = ${half}</b> balls on (√(2m ln 2) = ${Math.sqrt(2 * m * Math.LN2).toFixed(2)}) — about √m, not m/2 · ${OX.int(c.get("draw"))} balls thrown`);
    } else if (mode === "coupon") {
      const nC = Math.max(5, Math.min(200, Math.round(m / 5)));
      const r = AL.rng(13 + nC), TR = 1500, T = [];
      for (let t = 0; t < TR; t++) T.push(OR.coupon(nC, r, cc));
      const nH = nC * OR.harmonic(nC);
      const bins = d3.bin().thresholds(30)(T);
      const x = d3.scaleLinear().domain([bins[0].x0, bins[bins.length - 1].x1]).range([0, F.iw]);
      const y = d3.scaleLinear().domain([0, d3.max(bins, b => b.length / TR) * 1.15]).range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 5); AL.axisB(F.g, x, F.ih, 8, "draws needed to see all n coupons"); AL.axisL(F.g, y, 5, "fraction of runs", d3.format(".0%"));
      bins.forEach(b => F.g.append("rect").attr("x", x(b.x0) + 1).attr("y", y(b.length / TR)).attr("width", Math.max(0, x(b.x1) - x(b.x0) - 2)).attr("height", F.ih - y(b.length / TR)).attr("fill", AC.accent));
      [[nH, AC.good, "5 4"], [d3.mean(T), AC.a2, null], [nC * Math.log(nC), AC.muted, "2 3"]].forEach(([v, col, dash]) => { const l = F.g.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", 0).attr("y2", F.ih).attr("stroke", col).attr("stroke-width", 2); if (dash) l.attr("stroke-dasharray", dash); });
      AL.legend(F.g, [{ label: `n·Hₙ = ${nH.toFixed(1)}`, color: AC.good, dash: "5 4" }, { label: `measured mean ${d3.mean(T).toFixed(1)}`, color: AC.a2 }, { label: `n ln n = ${(nC * Math.log(nC)).toFixed(1)}`, color: AC.muted, dash: "2 3" }], F.iw + 12, 20);
      const tailC = 2, thr = nC * Math.log(nC) + tailC * nC, frac = T.filter(v => v > thr).length / TR;
      OX.setHtml("bins-readout", `n = ${nC} coupons (the slider ÷ 5) · mean draws to collect all: <b>${d3.mean(T).toFixed(1)}</b> against n·Hₙ = ${nH.toFixed(1)} · runs needing more than n ln n + 2n = ${thr.toFixed(0)} draws: ${(100 * frac).toFixed(2)}%, within the bound e⁻² = 13.53% ${OX.flag(frac <= Math.exp(-tailC) + 0.02)} · ${OX.int(c.get("draw"))} draws`);
    } else {
      const ex = d3.range(4, 15), TR = 12;
      const rows = ex.map(e => { const N = 1 << e, r = AL.rng(97 + e); let a = 0, b = 0; for (let t = 0; t < TR; t++) { a += OR.maxLoad(N, 1, r, cc); b += OR.maxLoad(N, 2, r, cc); } return { e: e, N: N, one: a / TR, two: b / TR }; });
      const x = d3.scaleLinear().domain([4, 14]).range([0, F.iw]), y = d3.scaleLinear().domain([0, 9]).range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 6); AL.axisB(F.g, x, F.ih, 10, "log₂ n   (n balls into n bins)", d3.format("d")); AL.axisL(F.g, y, 6, "maximum load (mean of 12 runs)");
      OX.line(F.g, rows.map(q => [q.e, Math.log(q.N) / Math.log(Math.log(q.N))]), x, y, AC.accent, "5 4", 1.6);
      OX.line(F.g, rows.map(q => [q.e, Math.log(Math.log(q.N)) / Math.LN2]), x, y, AC.good, "5 4", 1.6);
      OX.line(F.g, rows.map(q => [q.e, q.one]), x, y, AC.accent, null, 2.4);
      OX.line(F.g, rows.map(q => [q.e, q.two]), x, y, AC.good, null, 2.4);
      AL.legend(F.g, [{ label: "one random bin (measured)", color: AC.accent }, { label: "ln n / ln ln n", color: AC.accent, dash: "5 4" }, { label: "less loaded of two (measured)", color: AC.good }, { label: "ln ln n / ln 2", color: AC.good, dash: "5 4" }], F.iw + 12, 20);
      const lastR = rows[rows.length - 1];
      OX.setHtml("bins-readout", `n = ${OX.int(lastR.N)} balls into ${OX.int(lastR.N)} bins: the fullest bin holds <b>${lastR.one.toFixed(2)}</b> on average with one random choice, <b>${lastR.two.toFixed(2)}</b> when each ball goes to the less loaded of two random bins — the "power of two choices" · the slider does not apply in this mode · ${OX.int(c.get("probe"))} bin probes`);
    }
  }
  OX.on("bins-mode", "change", build); OX.on("bins-m", "input", build);
  build();
})();

/* ── 21  #lcg-svg  what a linear congruential generator's output looks like ── */
(function () {
  if (!OX.has("lcg-svg")) return;
  const W = 720, H = 360;
  function gcd(a, b) { while (b) { const t = a % b; a = b; b = t; } return a; }
  function hullDobell(a, c0, m) {
    const pf = []; let q = m; for (let p = 2; p * p <= q; p++) if (q % p === 0) { pf.push(p); while (q % p === 0) q /= p; } if (q > 1) pf.push(q);
    return gcd(c0, m) === 1 && pf.every(p => (a - 1) % p === 0) && (m % 4 !== 0 || (a - 1) % 4 === 0);
  }
  function build() {
    const gen = OX.val("lcg-gen", "good");
    const c = AL.counter(), cc = { add: k2 => c.add(k2) };
    const F = AL.frame("#lcg-svg", W, H, { l: 60, r: 250, t: 20, b: 42 });
    if (gen === "lowbits") {
      const js = d3.range(0, 12), per = js.map(j => OR.lowBitPeriod(1664525, 1013904223, j, 8192, 12345, cc));
      const x = d3.scaleBand().domain(js).range([0, F.iw]).padding(0.15), y = d3.scaleLog().domain([1, 8192]).range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 6);
      F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x));
      F.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6, "~s"));
      OX.txt(F.g, 0, -6, "measured period of bit j (log scale)"); OX.txt(F.g, F.iw, F.ih + 34, "bit j of the 32-bit state (0 = lowest)", { anchor: "end" });
      js.forEach((j, i) => F.g.append("rect").attr("x", x(j)).attr("y", y(per[i])).attr("width", x.bandwidth()).attr("height", F.ih - y(per[i])).attr("fill", AC.bad));
      OX.setHtml("lcg-readout", `x ← (1664525·x + 1013904223) mod 2³² — a full-period generator, period 2³² · but bit j of its output repeats with measured period [${per.join(", ")}] = 2^(j+1) ${OX.flag(per.every((p, j) => p === Math.pow(2, j + 1)))}: the lowest bit simply alternates 0, 1, 0, 1 · so <code>rand() % 2</code> on such a generator is a metronome, and <code>rand() % 2ᵏ</code> has period 2ᵏ · use the HIGH bits · ${OX.int(c.get("step"))} generator steps`);
      return;
    }
    let pts = [], title = "", extra = "", xs = "uₖ", ys = "uₖ₊₁", yDom = [0, 1];
    if (gen === "good" || gen === "poor" || gen === "short") {
      const P = { good: [137, 187], poor: [5, 1], short: [4, 1] }[gen], m = 256;
      const g1 = OR.lcg(P[0], P[1], m, 1), per = OR.period(OR.lcg(P[0], P[1], m, 1), 10000, cc);
      let prev = g1(); for (let i = 0; i < 256; i++) { const v = g1(); pts.push([prev / m, v / m]); prev = v; }
      title = `x ← (${P[0]}·x + ${P[1]}) mod 256`;
      extra = `measured period <b>${per}</b> of a possible 256 · Hull–Dobell conditions ${hullDobell(P[0], P[1], m) ? "met" : "NOT met"} ${OX.flag((per === 256) === hullDobell(P[0], P[1], m))} · every pair (uₖ, uₖ₊₁) lies on a handful of parallel lines — ${gen === "poor" ? "here 5 steep ones, so the next output is almost a function of the last" : gen === "short" ? "except that here the state falls into a cycle of that length and the generator is stuck" : "a full period does not mean a good lattice"}`;
    } else {
      const m = 2147483648;
      const nxt = gen === "randu" ? (function () { const g = OR.lcg(65539, 0, m, 1); return () => g() / m; })() : AL.rng(20240901);
      let a = nxt(), b = nxt(); const levels = new Set();
      for (let i = 0; i < 2500; i++) { const cc2 = nxt(); cc.add("step"); const v = 9 * a - 6 * b + cc2; pts.push([a, v]); levels.add(Math.round(v * 1000) / 1000); a = b; b = cc2; }
      ys = "9uₖ − 6uₖ₊₁ + uₖ₊₂"; yDom = [-6.5, 10.5];
      title = gen === "randu" ? "RANDU: x ← 65539·x mod 2³¹" : "mulberry32 (this page's generator)";
      const ints = pts.filter(q => Math.abs(q[1] - Math.round(q[1])) < 1e-6).length;
      extra = gen === "randu" ? `because 65539 = 2¹⁶ + 3, x_{k+2} = 6x_{k+1} − 9x_k (mod 2³¹): the combination plotted is ALWAYS an integer — measured ${ints} of ${pts.length} triples on an integer, spread over <b>${new Set(pts.map(q => Math.round(q[1]))).size}</b> distinct values, i.e. every triple lies on one of that many planes in the unit cube ${OX.flag(ints === pts.length)}`
        : `the same combination is spread continuously: ${ints} of ${pts.length} triples land on an integer, ${OX.int(levels.size)} distinct values to three decimals`;
    }
    const x = d3.scaleLinear().domain([0, 1]).range([0, F.iw > F.ih ? F.ih : F.iw]), y = d3.scaleLinear().domain(yDom).range([F.ih, 0]);
    AL.axisB(F.g, x, F.ih, 5, xs); AL.axisL(F.g, y, 6, ys);
    F.g.selectAll("circle.pt").data(pts).join("circle").attr("class", "pt").attr("cx", q => x(q[0])).attr("cy", q => y(q[1])).attr("r", gen === "randu" || gen === "mulberry" ? 1.3 : 2.2).attr("fill", gen === "mulberry" ? AC.good : AC.accent).attr("opacity", 0.8);
    const tx = F.ih + 30;
    OX.txt(F.g, tx, 12, title, { fill: AC.ink, size: 12 });
    OX.setHtml("lcg-readout", `${title} · ${extra} · ${OX.int(c.get("step"))} generator steps`);
  }
  OX.on("lcg-gen", "change", build);
  build();
})();

/* ── 22  #shuffle-svg  where each item lands: Fisher–Yates against the classic bugs ── */
(function () {
  if (!OX.has("shuffle-svg")) return;
  const W = 720, H = 360;
  function build() {
    const kind = OX.val("shuffle-kind", "naive"), n = +OX.val("shuffle-n", 4);
    OX.setText("shuffle-n-val", n);
    const c = AL.counter(), cc = { add: k2 => c.add(k2) };
    const R = kind === "sort" ? OR.shuffleBySort(n, 20000, AL.rng(99 + n), cc) : OR.shuffleExact(kind, n, cc);
    const svg = d3.select("#shuffle-svg"); svg.selectAll("*").remove();
    const cs = Math.min(40, Math.floor(260 / n)), g = svg.append("g").attr("transform", "translate(70,50)");
    const col = d3.scaleDiverging(d3.interpolateRdBu).domain([0, 1, 2]);
    OX.txt(g, n * cs / 2, -28, "final position j →", { anchor: "middle" });
    OX.txt(g, -40, n * cs / 2, "item i", { anchor: "middle" });
    let dev = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const v = R.M[i][j]; dev = Math.max(dev, Math.abs(v - 1 / n));
      g.append("rect").attr("x", j * cs).attr("y", i * cs).attr("width", cs - 2).attr("height", cs - 2).attr("rx", 3).attr("fill", col(v * n));
      if (cs >= 30) g.append("text").attr("x", j * cs + cs / 2 - 1).attr("y", i * cs + cs / 2 + 3).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#111").text((100 * v).toFixed(1));
    }
    for (let j = 0; j < n; j++) OX.txt(g, j * cs + cs / 2 - 1, -8, j, { anchor: "middle" });
    for (let i = 0; i < n; i++) OX.txt(g, -8, i * cs + cs / 2 + 3, i, { anchor: "end" });
    OX.txt(g, 0, n * cs + 18, `cell = P(item i ends at j), %; fair = ${(100 / n).toFixed(1)}%`);
    OX.txt(g, 0, n * cs + 32, "blue = too likely, red = too unlikely");
    /* right: every permutation's probability, sorted */
    let fact = 1; for (let i = 2; i <= n; i++) fact *= i;
    const probs = []; R.perms.forEach(v => probs.push(v / R.paths));
    while (probs.length < fact) probs.push(0);
    probs.sort((a, b) => b - a);
    const rg = svg.append("g").attr("transform", "translate(400,40)"), iw = 290, ih = 240;
    const x = d3.scaleLinear().domain([0, fact]).range([0, iw]), y = d3.scaleLinear().domain([0, d3.max(probs) * 1.15]).range([ih, 0]);
    AL.gridY(rg, y, iw, 4); AL.axisL(rg, y, 4, "P(permutation)", d3.format(".3f"));
    AL.axisB(rg, x, ih, 5, `the ${fact} permutations, most likely first`);
    rg.append("path").datum(probs).attr("fill", AC.accent).attr("opacity", 0.8)
      .attr("d", d3.area().x((p, i) => x(i + 0.5)).y0(ih).y1(p => y(p)).curve(d3.curveStep));
    OX.hline(rg, 0, iw, y(1 / fact), AC.good, "5 4");
    OX.txt(rg, iw, y(1 / fact) - 4, "1/n!", { anchor: "end", fill: AC.good });
    const pmin = probs[probs.length - 1], pmax = probs[0];
    const names = { fy: "Fisher–Yates (j uniform in 0…i)", naive: "naive (j uniform in 0…n−1 at every i)", sattolo: "off-by-one (j uniform in 0…i−1)", sort: "sort with a coin-flip comparator (merge sort)" };
    const how = kind === "sort" ? `${OX.int(R.paths)} Monte Carlo runs, ${OX.int(c.get("cmp"))} comparator coin flips` : `all <b>${OX.int(R.paths)}</b> equally likely paths of random choices enumerated exactly`;
    OX.setHtml("shuffle-readout", `${names[kind]}, n = ${n} · ${how} · permutations reached: <b>${R.perms.size}</b> of n! = ${fact} · most likely / least likely permutation: ${pmin > 0 ? (pmax / pmin).toFixed(3) : "∞ (some are impossible)"} · worst |P(i→j) − 1/n| = <b>${dev.toFixed(4)}</b> ${kind === "fy" ? OX.flag(dev < 1e-12 && R.perms.size === fact) : ""}${kind === "naive" ? ` · ${OX.int(R.paths)} = nⁿ paths cannot split evenly over ${fact} permutations when n! ∤ nⁿ` : ""}${kind === "sattolo" ? " · every item always moves: this is Sattolo's algorithm, which produces only the (n−1)! cyclic permutations — correct for a random cycle, wrong for a shuffle" : ""}`);
  }
  OX.on("shuffle-kind", "change", build); OX.on("shuffle-n", "input", build);
  build();
})();

/* ── 23  #res-svg  reservoir sampling: each stream position's inclusion rate ── */
(function () {
  if (!OX.has("res-svg")) return;
  const W = 720, H = 320, TR = 4000;
  function build() {
    const kind = OX.val("res-kind", "R"), N = +OX.val("res-N", 60), k = Math.min(+OX.val("res-k", 5), N);
    OX.setText("res-N-val", N); OX.setText("res-k-val", k);
    const r = AL.rng(4242 + N * 13 + k), c = AL.counter(), cc = { add: k2 => c.add(k2) };
    const inc = new Array(N).fill(0);
    for (let t = 0; t < TR; t++) OR.reservoir(kind, N, k, r, cc).forEach(i => inc[i]++);
    const f = inc.map(v => v / TR);
    const F = AL.frame("#res-svg", W, H, { l: 56, r: 190, t: 20, b: 42 });
    const x = d3.scaleBand().domain(d3.range(N)).range([0, F.iw]).padding(0.1);
    const y = d3.scaleLinear().domain([0, Math.max(1.5 * k / N, d3.max(f) * 1.1)]).range([F.ih, 0]);
    AL.gridY(F.g, y, F.iw, 5); AL.axisL(F.g, y, 5, "P(item i is in the final sample)", d3.format(".0%"));
    F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x).tickValues(x.domain().filter(i => i % Math.ceil(N / 12) === 0)));
    OX.txt(F.g, F.iw, F.ih + 34, "position i in the stream", { anchor: "end" });
    f.forEach((v, i) => F.g.append("rect").attr("x", x(i)).attr("y", y(v)).attr("width", x.bandwidth()).attr("height", F.ih - y(v)).attr("fill", AC.accent));
    OX.hline(F.g, 0, F.iw, y(k / N), AC.good, "5 4");
    AL.legend(F.g, [{ label: "measured inclusion rate", color: AC.accent }, { label: `k/N = ${(100 * k / N).toFixed(1)}%`, color: AC.good, dash: "5 4" }], F.iw + 12, 20);
    const dev = d3.max(f, v => Math.abs(v - k / N)), sd = Math.sqrt((k / N) * (1 - k / N) / TR);
    const names = { R: "reservoir sampling (keep the i-th item with probability k/i, into a random slot)", half: "biased: replace a random slot with probability 1/2", fixed: "biased: replace with fixed probability k/N" };
    OX.setHtml("res-readout", `${names[kind]} · N = ${N}, k = ${k}, ${OX.int(TR)} passes over the stream · first item kept ${(100 * f[0]).toFixed(1)}%, last item kept ${(100 * f[N - 1]).toFixed(1)}%, target ${(100 * k / N).toFixed(1)}% · worst deviation ${(100 * dev).toFixed(2)} points against a sampling s.d. of ${(100 * sd).toFixed(2)} ${kind === "R" ? OX.flag(dev < 5 * sd + 1e-9) : ""} · ${OX.int(c.get("draw"))} random draws for ${OX.int(c.get("item"))} items`);
  }
  OX.on("res-kind", "change", build); ["res-N", "res-k"].forEach(id => OX.on(id, "input", build));
  build();
})();
