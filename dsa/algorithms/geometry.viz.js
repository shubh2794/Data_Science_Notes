/* geometry.viz.js — figures for dsa/algorithms/geometry.html
   (part 8 of the Algorithm Design & Analysis series: Computational Geometry).

   Loaded after ../../data.js, ../../notes.js and ../dsa-viz.js, so the globals
   AC (palette) and AL (rng/frame/counter/stepper/…) are available.

   House rule obeyed throughout: every count this page DISPLAYS — orientation
   tests, sort comparisons, stack pops, events processed, distance
   evaluations, tree nodes visited — is produced by running the real routine
   under AL.counter() and reading the counter back. Every ANSWER is
   cross-checked in the same figure against an independent computation
   (brute force over all pairs / all points, exact BigInt arithmetic, or a
   convexity certificate) and printed with an agree / DISAGREE flag.

   Layout of this file:
     GR — the instrumented routines, pure and DOM-free (loadable in node:
          `require("./geometry.viz.js")` returns GR).
     GX — small DOM helpers shared by the figures.
     figures — one block per <svg>, in page order:
        #rb-svg   floating-point orientation on a near-degenerate grid
        #or-svg   orientation explorer: three draggable points
        #br-svg   Bresenham's line, two draggable endpoints (appended at the end, block GB)
        #sg-svg   segment-intersection explorer: four draggable points
        #sw-svg   sweep-line (Bentley–Ottmann) intersection reporting, stepped
        #hl-svg   convex hull: Graham / monotone chain / Jarvis / quickhull, stepped
        #cp-svg   closest pair: divide and conquer against the randomized grid
        #pp-svg   polygons: shoelace area, ray casting against winding number
        #kd-svg   kd-tree build + orthogonal range query, with the √n line-query curve */

/* ═══════════════════════════════════════════════════════════════════════════
   GR — the instrumented routines
   ═══════════════════════════════════════════════════════════════════════════ */
const GR = (function () {
  const nop = { add: function () {}, get: function () { return 0; } };
  const ctr = c => c || nop;
  function rnd(seed) {                      // mulberry32, so this loads in node without AL
    let a = (seed >>> 0) || 1;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0; let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* ── primitives ───────────────────────────────────────────────────────── */
  /* orient(a, b, c) = (b − a) × (c − a): twice the signed area of triangle abc.
     > 0  c is LEFT of the directed line a→b (counter-clockwise turn)
     < 0  c is RIGHT (clockwise turn)      = 0  collinear.
     Exact whenever the inputs are integers below 2^25 in magnitude. */
  function orient(a, b, c, k) {
    ctr(k).add("orient");
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  }
  const sgn = v => (v > 0) - (v < 0);
  const d2 = (a, b) => (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
  const same = (a, b) => a.x === b.x && a.y === b.y;

  /* ── §02 robustness: three ways to evaluate the same sign ─────────────── */
  const EPS = Math.pow(2, -53);
  const ERRB = (3 + 16 * EPS) * EPS;        // static error bound for orient2d's first stage
  function orientFloatP(p, q, r) { return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x); }
  function orientFloatR(p, q, r) { return (p.x - r.x) * (q.y - r.y) - (p.y - r.y) * (q.x - r.x); }
  /* exact: every coordinate used here is an integer multiple of 2^-53, so
     scaling by 2^53 gives integers and BigInt does the rest exactly */
  const SC = Math.pow(2, 53);
  function big(v) { return BigInt(v * SC); }
  function orientExact(p, q, r) {
    const px = big(p.x), py = big(p.y), qx = big(q.x), qy = big(q.y), rx = big(r.x), ry = big(r.y);
    const d = (qx - px) * (ry - py) - (qy - py) * (rx - px);
    return d > 0n ? 1 : (d < 0n ? -1 : 0);
  }
  /* float first; fall back to exact only when |det| is inside the proven error bound */
  function orientFiltered(p, q, r, k) {
    const l = (p.x - r.x) * (q.y - r.y), rr = (p.y - r.y) * (q.x - r.x), det = l - rr;
    const bound = ERRB * (Math.abs(l) + Math.abs(rr));
    if (det > bound || -det > bound) { ctr(k).add("fast"); return sgn(det); }
    ctr(k).add("exact");
    return orientExact(p, q, r);
  }

  /* ── §07 segment intersection ─────────────────────────────────────────── */
  function onSeg(a, b, p) {                  // p collinear with ab: is it within the box?
    return Math.min(a.x, b.x) <= p.x && p.x <= Math.max(a.x, b.x) &&
           Math.min(a.y, b.y) <= p.y && p.y <= Math.max(a.y, b.y);
  }
  /* returns {hit, kind, o:[o1..o4]} — kind names which case fired */
  function segIntersect(a, b, c, d, k) {
    const o1 = sgn(orient(a, b, c, k)), o2 = sgn(orient(a, b, d, k));
    const o3 = sgn(orient(c, d, a, k)), o4 = sgn(orient(c, d, b, k));
    const o = [o1, o2, o3, o4];
    if (o1 * o2 < 0 && o3 * o4 < 0) return { hit: true, kind: "proper", o };
    if (o1 === 0 && onSeg(a, b, c)) return { hit: true, kind: "c on ab", o };
    if (o2 === 0 && onSeg(a, b, d)) return { hit: true, kind: "d on ab", o };
    if (o3 === 0 && onSeg(c, d, a)) return { hit: true, kind: "a on cd", o };
    if (o4 === 0 && onSeg(c, d, b)) return { hit: true, kind: "b on cd", o };
    return { hit: false, kind: (o1 === 0 && o2 === 0) ? "collinear, disjoint" : "none", o };
  }
  function interPoint(s, t) {               // intersection of the supporting lines
    const r = { x: s.b.x - s.a.x, y: s.b.y - s.a.y }, q = { x: t.b.x - t.a.x, y: t.b.y - t.a.y };
    const den = r.x * q.y - r.y * q.x;
    const u = ((t.a.x - s.a.x) * q.y - (t.a.y - s.a.y) * q.x) / den;
    return { x: s.a.x + u * r.x, y: s.a.y + u * r.y };
  }

  /* ── §08–§10 the sweep line ───────────────────────────────────────────── */
  function randomSegments(n, seed) {
    const r = rnd(seed * 7919 + 13), segs = [];
    for (let i = 0; i < n; i++) {
      let x1 = 0.03 + 0.94 * r(), x2 = 0.03 + 0.94 * r();
      const y1 = 0.05 + 0.9 * r(), y2 = 0.05 + 0.9 * r();
      if (Math.abs(x1 - x2) < 0.08) x2 = x1 < 0.5 ? x1 + 0.25 : x1 - 0.25;
      const a = { x: x1, y: y1 }, b = { x: x2, y: y2 };
      segs.push(a.x < b.x ? { id: i, a, b } : { id: i, a: b, b: a });
    }
    return segs;
  }
  /* a binary heap keyed by x: the event queue */
  function Heap(k) {
    const h = [];
    const less = (i, j) => { k.add("pqCmp"); return h[i].x < h[j].x; };
    const sw = (i, j) => { const t = h[i]; h[i] = h[j]; h[j] = t; };
    return {
      size: () => h.length,
      items: () => h.slice(),
      push(e) { h.push(e); let i = h.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (!less(i, p)) break; sw(i, p); i = p; } },
      pop() {
        const top = h[0], last = h.pop();
        if (h.length) { h[0] = last; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i;
          if (l < h.length && less(l, m)) m = l; if (r < h.length && less(r, m)) m = r;
          if (m === i) break; sw(i, m); i = m; } }
        return top;
      }
    };
  }
  /* Bentley–Ottmann in general position. Status = the segments crossing the
     sweep line, bottom to top. A sorted array stands in for the balanced BST:
     the comparisons its binary search makes are exactly the O(log n) key
     comparisons a BST insertion would make, and those are what is counted. */
  function sweep(segs, k, frames) {
    k = ctr(k);
    const yAt = (s, x) => s.a.y + (s.b.y - s.a.y) * (x - s.a.x) / (s.b.x - s.a.x);
    const slope = s => (s.b.y - s.a.y) / (s.b.x - s.a.x);
    const pq = Heap(k), status = [], found = [], scheduled = new Set();
    let curX = -Infinity, firstHitEvent = -1, ev = 0;
    segs.forEach(s => { pq.push({ x: s.a.x, y: s.a.y, type: "left", s }); pq.push({ x: s.b.x, y: s.b.y, type: "right", s }); });
    function check(s, t) {
      if (!s || !t) return;
      k.add("pairTests");
      if (!segIntersect(s.a, s.b, t.a, t.b, k).hit) return;
      if (firstHitEvent < 0) firstHitEvent = ev;
      const key = Math.min(s.id, t.id) + "-" + Math.max(s.id, t.id);
      if (scheduled.has(key)) return;
      const p = interPoint(s, t);
      if (p.x < curX) return;
      scheduled.add(key);
      pq.push({ x: p.x, y: p.y, type: "cross", s, t });
    }
    const snap = e => frames && frames.push({
      x: e.x, type: e.type, ids: e.type === "cross" ? [e.s.id, e.t.id] : [e.s.id],
      status: status.map(s => s.id), found: found.slice(),
      pending: pq.items().filter(q => q.type === "cross").map(q => ({ x: q.x, y: q.y })),
      c: { events: k.get("events"), orient: k.get("orient"), statusCmp: k.get("statusCmp"), pairTests: k.get("pairTests") }
    });
    while (pq.size()) {
      const e = pq.pop(); curX = e.x; ev++; k.add("events");
      if (e.type === "left") {
        let lo = 0, hi = status.length;
        const y = e.y, m0 = slope(e.s);
        while (lo < hi) {
          const mid = (lo + hi) >> 1; k.add("statusCmp");
          const ym = yAt(status[mid], curX);
          if (ym < y || (ym === y && slope(status[mid]) < m0)) lo = mid + 1; else hi = mid;
        }
        status.splice(lo, 0, e.s);
        check(status[lo - 1], e.s); check(e.s, status[lo + 1]);
      } else if (e.type === "right") {
        const i = status.indexOf(e.s);
        status.splice(i, 1);
        check(status[i - 1], status[i]);
      } else {
        found.push({ x: e.x, y: e.y, a: e.s.id, b: e.t.id });
        let i = status.indexOf(e.s), j = status.indexOf(e.t);
        if (i > j) { const t = i; i = j; j = t; }
        const t = status[i]; status[i] = status[j]; status[j] = t;
        check(status[i - 1], status[i]); check(status[j], status[j + 1]);
      }
      snap(e);
    }
    return { found, firstHitEvent };
  }
  function bruteIntersections(segs, k) {
    const out = [];
    for (let i = 0; i < segs.length; i++) for (let j = i + 1; j < segs.length; j++) {
      ctr(k).add("pairTests");
      if (segIntersect(segs[i].a, segs[i].b, segs[j].a, segs[j].b, k).hit) out.push(i + "-" + j);
    }
    return out;
  }

  /* ── §11–§17 convex hull, strict (no collinear points on the output) ──── */
  function randomPoints(n, seed, dist) {
    const r = rnd(seed * 104729 + 7), pts = [];
    for (let i = 0; i < n; i++) {
      let x, y;
      if (dist === "convex") { x = 0; y = 0; }            // replaced below
      else if (dist === "circle") { const t = 2 * Math.PI * r(); x = Math.round(50 + 44 * Math.cos(t)); y = Math.round(50 + 44 * Math.sin(t)); }
      else if (dist === "disk") { const t = 2 * Math.PI * r(), q = 44 * Math.sqrt(r()); x = Math.round(50 + q * Math.cos(t)); y = Math.round(50 + q * Math.sin(t)); }
      else if (dist === "grid") { x = 20 + 15 * Math.floor(r() * 5); y = 20 + 15 * Math.floor(r() * 5); }
      else { x = Math.round(6 + 88 * r()); y = Math.round(6 + 88 * r()); }
      pts.push({ x, y, i });
    }
    if (dist === "convex") {
      /* integer points in STRICTLY convex position: n/2 distinct primitive
         directions and their negations, sorted by angle and summed. Distinct
         directions ⇒ every turn is strictly left ⇒ h = n exactly. */
      const gcd = (a, b) => b ? gcd(b, a % b) : Math.abs(a);
      const dirs = [];
      for (let a = -5; a <= 5; a++) for (let b = 0; b <= 5; b++)
        if ((b > 0 || a > 0) && gcd(Math.abs(a), b) === 1) dirs.push([a, b]);
      for (let i = dirs.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = dirs[i]; dirs[i] = dirs[j]; dirs[j] = t; }
      const half = dirs.slice(0, Math.max(2, Math.floor(n / 2)));
      const all = half.concat(half.map(v => [-v[0], -v[1]])).sort((u, v) => Math.atan2(u[1], u[0]) - Math.atan2(v[1], v[0]));
      let cx = 0, cy = 0; const raw = all.map(v => { cx += v[0]; cy += v[1]; return [cx, cy]; });
      const xs = raw.map(p => p[0]), ys = raw.map(p => p[1]);
      const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) || 1;
      const sc = Math.max(1, Math.floor(84 / span));
      const ox = Math.round(50 - sc * (Math.max(...xs) + Math.min(...xs)) / 2), oy = Math.round(50 - sc * (Math.max(...ys) + Math.min(...ys)) / 2);
      const out = raw.map((p, i) => ({ x: ox + sc * p[0], y: oy + sc * p[1], i }));
      for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = out[i]; out[i] = out[j]; out[j] = t; }
      return out.slice(0, n).map((p, i) => ({ x: p.x, y: p.y, i }));
    }
    return pts;
  }
  /* Graham scan. Anchor = lowest point (leftmost among ties); every other point
     lies in the closed upper half-plane, so the polar order can be decided by
     orientation alone — no angles, no atan2. Ties (collinear with the anchor)
     are broken by distance, nearer first. */
  function graham(P, k, fr) {
    k = ctr(k);
    let a = P[0];
    P.forEach(p => { if (p.y < a.y || (p.y === a.y && p.x < a.x)) a = p; });
    const rest = P.filter(p => p !== a);
    rest.sort((p, q) => {
      k.add("cmp");
      const o = orient(a, p, q, k);
      if (o !== 0) return o > 0 ? -1 : 1;
      return d2(a, p) - d2(a, q);
    });
    const st = [a];
    fr && fr.push({ st: st.slice(), note: "anchor chosen, " + rest.length + " points sorted by angle", order: rest.map(p => p.i) });
    for (const p of rest) {
      while (st.length >= 2) {
        const o = orient(st[st.length - 2], st[st.length - 1], p, k);
        fr && fr.push({ st: st.slice(), cand: p, test: [st[st.length - 2], st[st.length - 1], p], o });
        if (o > 0) break;
        st.pop(); k.add("pop");
      }
      st.push(p); k.add("push");
      fr && fr.push({ st: st.slice(), cand: p, note: "push" });
    }
    return st;
  }
  function monotone(P, k, fr) {
    k = ctr(k);
    const S = P.slice().sort((p, q) => { k.add("cmp"); return p.x - q.x || p.y - q.y; });
    const half = (seq, name) => {
      const h = [];
      for (const p of seq) {
        while (h.length >= 2) {
          const o = orient(h[h.length - 2], h[h.length - 1], p, k);
          fr && fr.push({ st: h.slice(), done: name === "upper" ? lowerDone : null, cand: p, test: [h[h.length - 2], h[h.length - 1], p], o, note: name + " chain" });
          if (o > 0) break;
          h.pop(); k.add("pop");
        }
        h.push(p); k.add("push");
      }
      return h;
    };
    let lowerDone = null;
    const lower = half(S, "lower"); lowerDone = lower.slice();
    const upper = half(S.slice().reverse(), "upper");
    lower.pop(); upper.pop();
    const hull = lower.concat(upper);
    if (hull.length === 2 && same(hull[0], hull[1])) hull.pop();
    return hull;
  }
  /* Jarvis march (gift wrapping): O(n) per hull vertex, O(nh) in total */
  function jarvis(P, k, fr) {
    k = ctr(k);
    let s = P[0];
    P.forEach(p => { if (p.x < s.x || (p.x === s.x && p.y < s.y)) s = p; });
    const hull = [];
    let cur = s;
    for (let guard = 0; guard <= P.length; guard++) {
      hull.push(cur);
      let cand = null;
      for (const r of P) {
        if (same(r, cur)) continue;
        if (!cand) { cand = r; continue; }
        const o = orient(cur, cand, r, k);
        if (o < 0 || (o === 0 && d2(cur, r) > d2(cur, cand))) cand = r;
      }
      fr && fr.push({ st: hull.slice(), cand, note: "wrap: next hull vertex found" });
      if (!cand || same(cand, s)) break;
      cur = cand;
    }
    return hull;
  }
  function quickhull(P, k) {
    k = ctr(k);
    let A = P[0], B = P[0];
    P.forEach(p => {
      if (p.x < A.x || (p.x === A.x && p.y < A.y)) A = p;
      if (p.x > B.x || (p.x === B.x && p.y > B.y)) B = p;
    });
    if (same(A, B)) return [A];
    /* S = points strictly RIGHT of a→b (outside the current edge); returns the
       hull chain strictly between a and b, in counter-clockwise order */
    function rec(a, b, S) {
      if (!S.length) return [];
      k.add("calls");
      let far = null, fd = 0;
      /* ties in distance are broken toward a, so the point taken is a hull VERTEX,
         never a point in the middle of a hull edge parallel to a→b */
      const along = p => (p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y);
      S.forEach(p => { const v = -orient(a, b, p, k); if (v > fd || (v === fd && along(p) < along(far))) { fd = v; far = p; } });
      const L1 = S.filter(p => orient(a, far, p, k) < 0), L2 = S.filter(p => orient(far, b, p, k) < 0);
      return rec(a, far, L1).concat([far], rec(far, b, L2));
    }
    const below = P.filter(p => orient(A, B, p, k) < 0), above = P.filter(p => orient(B, A, p, k) < 0);
    return [A].concat(rec(A, B, below), [B], rec(B, A, above));
  }
  /* independent certificate: CCW, strictly convex, and every input point on or
     to the left of every hull edge */
  function hullCertificate(H, P) {
    const h = H.length;
    if (h < 3) return h > 0;
    for (let i = 0; i < h; i++) if (orient(H[i], H[(i + 1) % h], H[(i + 2) % h]) <= 0) return false;
    for (let i = 0; i < h; i++) for (const p of P) if (orient(H[i], H[(i + 1) % h], p) < 0) return false;
    return true;
  }
  const hullKey = H => H.map(p => p.x + "," + p.y).sort().join(" ");

  /* ── §18 closest pair ─────────────────────────────────────────────────── */
  function randomFloatPoints(n, seed, box) {
    const r = rnd(seed * 31337 + 5), B = box || 100;
    return Array.from({ length: n }, (_, i) => ({ x: B * r(), y: B * r(), i }));
  }
  const dist = (a, b, k) => { ctr(k).add("dist"); return Math.sqrt(d2(a, b)); };
  function closestBrute(P, k) {
    let best = Infinity, pair = null;
    for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) {
      const d = dist(P[i], P[j], k); if (d < best) { best = d; pair = [P[i], P[j]]; }
    }
    return { d: best, pair };
  }
  function closestDC(P, k) {
    k = ctr(k);
    const X = P.slice().sort((a, b) => a.x - b.x);
    let maxPartners = 0;
    function rec(lo, hi) {                   // returns {d, pair, ys: points of X[lo..hi) by y}
      const n = hi - lo;
      if (n <= 3) {
        let best = Infinity, pair = null;
        for (let i = lo; i < hi; i++) for (let j = i + 1; j < hi; j++) { const d = dist(X[i], X[j], k); if (d < best) { best = d; pair = [X[i], X[j]]; } }
        return { d: best, pair, ys: X.slice(lo, hi).sort((a, b) => a.y - b.y) };
      }
      const mid = (lo + hi) >> 1, xm = X[mid].x;
      const L = rec(lo, mid), R = rec(mid, hi);
      let d = L.d, pair = L.pair; if (R.d < d) { d = R.d; pair = R.pair; }
      const ys = []; let i = 0, j = 0;                         // merge, as in mergesort
      while (i < L.ys.length || j < R.ys.length) ys.push(j >= R.ys.length || (i < L.ys.length && L.ys[i].y <= R.ys[j].y) ? L.ys[i++] : R.ys[j++]);
      const strip = ys.filter(p => Math.abs(p.x - xm) < d);
      for (let a = 0; a < strip.length; a++) {
        let partners = 0;
        for (let b = a + 1; b < strip.length && strip[b].y - strip[a].y < d; b++) {
          partners++; k.add("stripDist");
          const dd = dist(strip[a], strip[b], k); if (dd < d) { d = dd; pair = [strip[a], strip[b]]; }
        }
        if (partners > maxPartners) maxPartners = partners;
      }
      return { d, pair, ys };
    }
    const r = rec(0, X.length);
    return { d: r.d, pair: r.pair, maxPartners };
  }
  /* randomized incremental grid: insert in random order; keep a hash grid of
     cell side δ = closest distance so far; a new point only needs its 3×3
     neighbourhood; when δ shrinks, rebuild. Expected O(n). */
  function closestGrid(P, seed, k) {
    k = ctr(k);
    const r = rnd(seed * 17 + 3), O = P.slice();
    for (let i = O.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = O[i]; O[i] = O[j]; O[j] = t; }
    let d = dist(O[0], O[1], k), pair = [O[0], O[1]], grid = new Map(), maxCell = 0, maxNbr = 0, rebuilds = 0;
    const key = (p) => Math.floor(p.x / d) + "," + Math.floor(p.y / d);
    const ins = p => { const kk = key(p); if (!grid.has(kk)) grid.set(kk, []); grid.get(kk).push(p); maxCell = Math.max(maxCell, grid.get(kk).length); k.add("gridInsert"); };
    const rebuild = m => { grid = new Map(); for (let i = 0; i < m; i++) ins(O[i]); rebuilds++; };
    rebuild(2);
    for (let i = 2; i < O.length; i++) {
      const p = O[i], cx = Math.floor(p.x / d), cy = Math.floor(p.y / d);
      let best = d, bp = null, nb = 0;
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
        const cell = grid.get((cx + dx) + "," + (cy + dy)); if (!cell) continue;
        for (const q of cell) { nb++; const dd = dist(p, q, k); if (dd < best) { best = dd; bp = q; } }
      }
      if (nb > maxNbr) maxNbr = nb;
      if (bp) { d = best; pair = [p, bp]; rebuild(i + 1); } else ins(p);
    }
    return { d, pair, rebuilds, maxCell, maxNbr, cell: d };
  }

  /* ── §19–§20 polygons ─────────────────────────────────────────────────── */
  function shoelace(V, k) {                  // twice the signed area; > 0 ⇔ counter-clockwise
    let s = 0;
    for (let i = 0; i < V.length; i++) { const a = V[i], b = V[(i + 1) % V.length]; ctr(k).add("term"); s += a.x * b.y - a.y * b.x; }
    return s;
  }
  /* even–odd: count edges crossing the horizontal ray to the right of q.
     Half-open rule (a.y ≤ q.y < b.y, or the reverse) counts a vertex once. */
  function rayCrossings(V, q, k) {
    let c = 0; const xs = [];
    for (let i = 0; i < V.length; i++) {
      const a = V[i], b = V[(i + 1) % V.length];
      ctr(k).add("edge");
      if ((a.y <= q.y) !== (b.y <= q.y)) {
        const o = orient(a, b, q, k);
        if ((b.y > a.y && o > 0) || (b.y < a.y && o < 0)) { c++; xs.push(a.x + (q.y - a.y) * (b.x - a.x) / (b.y - a.y)); }
      }
    }
    return { c, xs };
  }
  /* winding number: +1 for each upward edge q is left of, −1 for each downward edge q is right of */
  function winding(V, q, k) {
    let w = 0;
    for (let i = 0; i < V.length; i++) {
      const a = V[i], b = V[(i + 1) % V.length];
      ctr(k).add("edge");
      if (a.y <= q.y) { if (b.y > q.y && orient(a, b, q, k) > 0) w++; }
      else if (b.y <= q.y && orient(a, b, q, k) < 0) w--;
    }
    return w;
  }
  function onBoundary(V, q) {
    for (let i = 0; i < V.length; i++) { const a = V[i], b = V[(i + 1) % V.length]; if (orient(a, b, q) === 0 && onSeg(a, b, q)) return true; }
    return false;
  }
  /* convexity: every turn has the same sign (zeros allowed) AND the boundary
     winds exactly once — the second clause is what rejects a pentagram */
  function convexity(V, k) {
    let pos = 0, neg = 0, flips = 0, prev = 0;
    for (let i = 0; i < V.length; i++) {
      const a = V[i], b = V[(i + 1) % V.length], c = V[(i + 2) % V.length];
      const o = orient(a, b, c, k); if (o > 0) pos++; else if (o < 0) neg++;
    }
    for (let i = 0; i <= V.length; i++) {         // sign changes of Δx around the boundary
      const a = V[i % V.length], b = V[(i + 1) % V.length], s = sgn(b.x - a.x);
      if (s !== 0) { if (prev !== 0 && s !== prev) flips++; prev = s; }
    }
    const sameSign = !(pos && neg);
    return { pos, neg, flips, sameSign, convex: sameSign && flips <= 2 };
  }
  /* O(log n) point-in-convex-polygon: binary search for the wedge at V[0] */
  function inConvex(V, q, k) {
    const n = V.length;
    if (orient(V[0], V[1], q, k) < 0 || orient(V[0], V[n - 1], q, k) > 0) return false;
    let lo = 1, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (orient(V[0], V[m], q, k) >= 0) lo = m; else hi = m; }
    return orient(V[lo], V[lo + 1], q, k) >= 0;
  }

  /* ── §21–§22 range searching ──────────────────────────────────────────── */
  function kdBuild(P, k, depth, box) {
    k = ctr(k); depth = depth || 0; box = box || { x0: 0, y0: 0, x1: 100, y1: 100 };
    if (!P.length) return null;
    const ax = depth % 2 ? "y" : "x";
    const S = P.slice().sort((a, b) => { k.add("cmp"); return a[ax] - b[ax]; });
    const m = S.length >> 1, p = S[m];
    k.add("nodes");
    const lb = Object.assign({}, box), rb = Object.assign({}, box);
    if (ax === "x") { lb.x1 = p.x; rb.x0 = p.x; } else { lb.y1 = p.y; rb.y0 = p.y; }
    return { p, ax, box, depth, left: kdBuild(S.slice(0, m), k, depth + 1, lb), right: kdBuild(S.slice(m + 1), k, depth + 1, rb) };
  }
  const inRect = (p, R) => p.x >= R.x0 && p.x <= R.x1 && p.y >= R.y0 && p.y <= R.y1;
  function kdQuery(t, R, k, vis) {
    const out = [];
    (function go(t) {
      if (!t) return;
      ctr(k).add("visited"); vis && vis.push(t);
      if (inRect(t.p, R)) out.push(t.p);
      const lo = t.ax === "x" ? R.x0 : R.y0, hi = t.ax === "x" ? R.x1 : R.y1, v = t.p[t.ax];
      if (lo <= v) go(t.left);
      if (hi >= v) go(t.right);
    })(t);
    return out;
  }
  /* 2-D range tree: a balanced tree on x whose every node carries its
     subtree's points sorted by y. Query = O(log n) canonical nodes, one
     binary search in each. */
  function rtBuild(P, k) {
    k = ctr(k);
    const X = P.slice().sort((a, b) => { k.add("cmp"); return a.x - b.x; });
    function b(lo, hi) {
      k.add("nodes");
      if (hi - lo === 1) return { lo: X[lo].x, hi: X[lo].x, ys: [X[lo]] };
      const m = (lo + hi) >> 1, L = b(lo, m), Rr = b(m, hi);
      const ys = []; let i = 0, j = 0;
      while (i < L.ys.length || j < Rr.ys.length) { k.add("merge"); ys.push(j >= Rr.ys.length || (i < L.ys.length && L.ys[i].y <= Rr.ys[j].y) ? L.ys[i++] : Rr.ys[j++]); }
      k.add("stored", ys.length);
      return { lo: X[lo].x, hi: X[hi - 1].x, ys, left: L, right: Rr };
    }
    return X.length ? b(0, X.length) : null;
  }
  function rtQuery(t, R, k) {
    k = ctr(k); const out = [];
    (function go(v) {
      if (!v) return;
      k.add("visited");
      if (v.hi < R.x0 || v.lo > R.x1) return;
      if (R.x0 <= v.lo && v.hi <= R.x1) {           // canonical node: 1-D query on its y-list
        k.add("canonical");
        let lo = 0, hi = v.ys.length;
        while (lo < hi) { const m = (lo + hi) >> 1; k.add("bsearch"); if (v.ys[m].y < R.y0) lo = m + 1; else hi = m; }
        for (let i = lo; i < v.ys.length && v.ys[i].y <= R.y1; i++) out.push(v.ys[i]);
        return;
      }
      go(v.left); go(v.right);
    })(t);
    return out;
  }
  function gridQuery(P, R, g, k) {
    k = ctr(k); const cs = 100 / g, cells = new Map();
    P.forEach(p => { const kk = Math.min(g - 1, Math.floor(p.x / cs)) + "," + Math.min(g - 1, Math.floor(p.y / cs)); if (!cells.has(kk)) cells.set(kk, []); cells.get(kk).push(p); });
    const out = [];
    const i0 = Math.max(0, Math.floor(R.x0 / cs)), i1 = Math.min(g - 1, Math.floor(R.x1 / cs));
    const j0 = Math.max(0, Math.floor(R.y0 / cs)), j1 = Math.min(g - 1, Math.floor(R.y1 / cs));
    for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
      k.add("cells");
      (cells.get(i + "," + j) || []).forEach(p => { k.add("examined"); if (inRect(p, R)) out.push(p); });
    }
    return out;
  }

  return { rnd, orient, sgn, d2, same, EPS, ERRB, orientFloatP, orientFloatR, orientExact, orientFiltered,
           onSeg, segIntersect, interPoint, randomSegments, sweep, bruteIntersections,
           randomPoints, graham, monotone, jarvis, quickhull, hullCertificate, hullKey,
           randomFloatPoints, closestBrute, closestDC, closestGrid,
           shoelace, rayCrossings, winding, onBoundary, convexity, inConvex,
           kdBuild, kdQuery, inRect, rtBuild, rtQuery, gridQuery };
})();
if (typeof module !== "undefined" && module.exports) module.exports = GR;

/* ═══════════════════════════════════════════════════════════════════════════
   GX — DOM helpers, and the figures (browser only)
   ═══════════════════════════════════════════════════════════════════════════ */
if (typeof document !== "undefined" && typeof d3 !== "undefined") {

const GX = {
  int: d3.format(","),
  table: function (sel, head, rows) {
    const h = d3.select(sel); if (h.empty()) return;
    h.selectAll("*").remove();
    const t = h.append("table").attr("class", "cmp").style("margin", "10px 0 0");
    const hr = t.append("tr"); head.forEach(c => hr.append("th").html(c));
    rows.forEach(r => { const tr = t.append("tr"); r.forEach(c => tr.append("td").html(c)); });
  },
  flag: ok => ok ? '<span style="color:' + AC.good + '">✓ agree</span>'
                 : '<span style="color:' + AC.bad + '">✗ DISAGREE</span>',
  clearControls: node => node.parentNode.querySelectorAll('div[role="group"]').forEach(el => el.remove()),
  txt: (g, x, y, s, o) => {
    o = o || {};
    return g.append("text").attr("x", x).attr("y", y).attr("font-size", o.size || 12)
      .attr("fill", o.fill || AC.ink).attr("text-anchor", o.anchor || "start")
      .attr("font-family", o.mono ? "SF Mono, Menlo, monospace" : null).text(s);
  },
  /* a y-up plane: math orientation, so a left turn LOOKS counter-clockwise */
  plane: (x0, y0, w, h, dx, dy) => ({
    sx: d3.scaleLinear().domain(dx).range([x0, x0 + w]),
    sy: d3.scaleLinear().domain(dy).range([y0 + h, y0])
  }),
  gridLines: (g, P, xs, ys) => {
    xs.forEach(v => g.append("line").attr("x1", P.sx(v)).attr("x2", P.sx(v)).attr("y1", P.sy.range()[0]).attr("y2", P.sy.range()[1]).attr("stroke", AC.grid));
    ys.forEach(v => g.append("line").attr("y1", P.sy(v)).attr("y2", P.sy(v)).attr("x1", P.sx.range()[0]).attr("x2", P.sx.range()[1]).attr("stroke", AC.grid));
  },
  sgnWord: s => s > 0 ? "+ (left / counter-clockwise)" : (s < 0 ? "− (right / clockwise)" : "0 (collinear)"),
  segColor: i => d3.schemeTableau10[i % 10]
};

/* ══ FIGURE 1 — floating-point orientation on a near-degenerate grid ═══════
   p = (0.5 + i·u, 0.5 + j·u) on a 48 × 48 grid of spacing u, q = (12, 12),
   r = (24, 24). The exact sign is sign(j − i): the diagonal is collinear.
   Each cell is coloured by what the chosen predicate RETURNS; the right panel
   marks every cell where that differs from the BigInt-exact sign.          */
(function () {
  const svg = d3.select("#rb-svg"); if (svg.empty()) return;
  const N = 48, CS = 6;
  const LINES = { a: [{ x: 12, y: 12 }, { x: 24, y: 24 }], b: [{ x: 12.1, y: 12.1 }, { x: 24.3, y: 24.3 }] };
  const col = s => s > 0 ? AC.teal : (s < 0 ? AC.accent : AC.a2);
  function build() {
    const pred = d3.select("#rb-pred").property("value"), sp = +d3.select("#rb-space").property("value");
    const u = Math.pow(2, -53) * sp, k = AL.counter();
    const [q, r] = LINES[d3.select("#rb-line").property("value")];
    svg.selectAll("*").remove();
    const L = svg.append("g").attr("transform", "translate(40,34)"), R = svg.append("g").attr("transform", "translate(380,34)");
    GX.txt(svg, 40, 22, "sign returned by the predicate", { fill: AC.muted });
    GX.txt(svg, 380, 22, "cells where it disagrees with exact arithmetic", { fill: AC.muted });
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const p = { x: 0.5 + i * u, y: 0.5 + j * u };
      const ex = GR.orientExact(p, q, r); k.add("exactCalls");
      let s;
      if (pred === "floatP") { s = GR.sgn(GR.orientFloatP(p, q, r)); k.add("float"); }
      else if (pred === "floatR") { s = GR.sgn(GR.orientFloatR(p, q, r)); k.add("float"); }
      else if (pred === "filtered") s = GR.orientFiltered(p, q, r, k);
      else s = ex;
      const wrong = s !== ex; if (wrong) k.add("wrong"); if (wrong && s * ex < 0) k.add("flipped");
      const X = i * CS, Y = (N - 1 - j) * CS;
      L.append("rect").attr("x", X).attr("y", Y).attr("width", CS).attr("height", CS).attr("fill", col(s));
      R.append("rect").attr("x", X).attr("y", Y).attr("width", CS).attr("height", CS)
        .attr("fill", wrong ? AC.bad : AC.panel2);
    }
    [L, R].forEach(g => {
      g.append("rect").attr("width", N * CS).attr("height", N * CS).attr("fill", "none").attr("stroke", AC.line);
      g.append("line").attr("x1", 0).attr("y1", N * CS).attr("x2", N * CS).attr("y2", 0).attr("stroke", AC.ink).attr("stroke-dasharray", "2 3").attr("opacity", 0.5);
      GX.txt(g, 0, N * CS + 14, "i → (p.x = 0.5 + i·u)", { size: 10, fill: AC.muted });
    });
    const lg = AL.legend(svg, [{ label: "+ left of q→r", color: AC.teal }, { label: "− right of q→r", color: AC.accent }, { label: "0 collinear", color: AC.a2 }, { label: "wrong sign", color: AC.bad }], 40, 352, { gap: 0 });
    lg.selectAll("g").attr("transform", (d, i) => `translate(${i * 150},0)`);
    const tot = N * N;
    d3.select("#rb-readout").html(
      "q = (" + q.x + ", " + q.y + "), r = (" + r.x + ", " + r.y + "), u = " + sp + " × 2⁻⁵³ &nbsp;·&nbsp; " + GX.int(tot) + " orientation queries &nbsp;·&nbsp; wrong sign <b>" + GX.int(k.get("wrong")) + "</b>"
      + " (" + (100 * k.get("wrong") / tot).toFixed(1) + "%): a spurious 0 on <b>" + GX.int(k.get("wrong") - k.get("flipped")) + "</b>, the OPPOSITE sign on <b>" + GX.int(k.get("flipped")) + "</b>"
      + (pred === "filtered" ? " &nbsp;·&nbsp; answered by the float filter <b>" + GX.int(k.get("fast")) + "</b>, fell back to exact <b>" + GX.int(k.get("exact")) + "</b>" : "")
      + " &nbsp;·&nbsp; the exact answer is 0 on the " + N + " diagonal cells and sign(j − i) elsewhere"
      + " &nbsp;·&nbsp; " + (k.get("wrong") === 0 ? GX.flag(true) + " with exact" : GX.flag(false) + " with exact on " + GX.int(k.get("wrong")) + " cells"));
  }
  d3.select("#rb-pred").on("change", build);
  d3.select("#rb-space").on("change", build);
  d3.select("#rb-line").on("change", build);
  build();
})();

/* ══ FIGURES 2 & 3 — the orientation and segment-intersection explorers ════
   Integer coordinates, so every orientation value shown is exact.          */
function dragExplorer(opt) {
  const svg = d3.select(opt.svg); if (svg.empty()) return;
  const P = GX.plane(30, 16, 420, 270, [0, 14], [0, 9]);
  const pts = opt.presets[0].pts.map(p => ({ x: p[0], y: p[1] }));
  svg.selectAll("*").remove();
  const grid = svg.append("g");
  GX.gridLines(grid, P, d3.range(0, 15), d3.range(0, 10));
  d3.range(0, 15, 2).forEach(v => GX.txt(grid, P.sx(v), P.sy(0) + 14, v, { size: 9, fill: AC.muted, anchor: "middle" }));
  d3.range(0, 10, 2).forEach(v => GX.txt(grid, P.sx(0) - 6, P.sy(v) + 3, v, { size: 9, fill: AC.muted, anchor: "end" }));
  const cid = opt.svg.slice(1) + "-clip";
  svg.append("defs").append("clipPath").attr("id", cid).append("rect").attr("x", 30).attr("y", 16).attr("width", 420).attr("height", 270);
  const layer = svg.append("g").attr("clip-path", "url(#" + cid + ")"), side = svg.append("g").attr("transform", "translate(474,24)");
  const handles = svg.append("g").selectAll("g").data(pts).join("g").style("cursor", "grab");
  handles.append("circle").attr("r", 9).attr("fill", AC.panel).attr("stroke", AC.ink).attr("stroke-width", 1.5);
  handles.append("text").attr("text-anchor", "middle").attr("y", 4).attr("font-size", 11).attr("fill", AC.ink).text((d, i) => opt.names[i]);
  handles.call(d3.drag().on("drag", function (ev, d) {
    d.x = AL.clamp(Math.round(P.sx.invert(ev.x)), 0, 14); d.y = AL.clamp(Math.round(P.sy.invert(ev.y)), 0, 9); draw();
  }));
  function draw() {
    handles.attr("transform", d => `translate(${P.sx(d.x)},${P.sy(d.y)})`);
    layer.selectAll("*").remove(); side.selectAll("*").remove();
    opt.draw(pts, P, layer, side);
  }
  const sel = d3.select(opt.select);
  sel.selectAll("option").data(opt.presets).join("option").attr("value", (d, i) => i).text(d => d.name);
  sel.on("change", function () { opt.presets[+this.value].pts.forEach((p, i) => { pts[i].x = p[0]; pts[i].y = p[1]; }); draw(); });
  draw();
}

dragExplorer({
  svg: "#or-svg", select: "#or-preset", names: ["a", "b", "c"],
  presets: [
    { name: "left turn", pts: [[2, 2], [9, 4], [6, 8]] },
    { name: "right turn", pts: [[2, 2], [9, 4], [11, 1]] },
    { name: "collinear, c beyond b", pts: [[1, 1], [5, 3], [11, 6]] },
    { name: "collinear, c between a and b", pts: [[1, 1], [11, 6], [5, 3]] },
    { name: "almost collinear (area ½)", pts: [[1, 1], [12, 7], [2, 2]] },
    { name: "a = b (degenerate)", pts: [[5, 5], [5, 5], [9, 2]] }
  ],
  draw: function (pts, P, g, side) {
    const [a, b, c] = pts, k = AL.counter();
    const o = GR.orient(a, b, c, k), s = GR.sgn(o);
    const fill = s > 0 ? AC.teal : (s < 0 ? AC.accent : AC.a2);
    g.append("path").attr("d", `M${P.sx(a.x)},${P.sy(a.y)}L${P.sx(b.x)},${P.sy(b.y)}L${P.sx(c.x)},${P.sy(c.y)}Z`)
      .attr("fill", fill).attr("fill-opacity", 0.18).attr("stroke", "none");
    /* the directed line a→b, extended, splits the plane */
    const dx = b.x - a.x, dy = b.y - a.y;
    if (dx || dy) g.append("line").attr("x1", P.sx(a.x - 30 * dx)).attr("y1", P.sy(a.y - 30 * dy)).attr("x2", P.sx(a.x + 30 * dx)).attr("y2", P.sy(a.y + 30 * dy))
      .attr("stroke", AC.muted).attr("stroke-dasharray", "3 4");
    AL.arrow(g, P.sx(a.x), P.sy(a.y), P.sx(b.x), P.sy(b.y), { color: AC.ink, w: 2, head: 7 });
    AL.arrow(g, P.sx(a.x), P.sy(a.y), P.sx(c.x), P.sy(c.y), { color: fill, w: 2, head: 7 });
    const L = [
      ["u = b − a", "(" + dx + ", " + dy + ")"],
      ["v = c − a", "(" + (c.x - a.x) + ", " + (c.y - a.y) + ")"],
      ["u × v", dx + "·" + (c.y - a.y) + " − " + dy + "·" + (c.x - a.x)],
      ["", "= " + o],
      ["sign", GX.sgnWord(s)],
      ["|triangle abc|", (Math.abs(o) / 2) + "  (= |u × v| / 2)"],
      ["multiplications", "2, subtractions 5"],
      ["angles / slopes", "none, no division"]
    ];
    L.forEach((r, i) => { GX.txt(side, 0, i * 30, r[0], { size: 11, fill: AC.muted }); GX.txt(side, 0, i * 30 + 14, r[1], { size: 12, mono: true, fill: i === 4 ? fill : AC.ink }); });
    d3.select("#or-readout").html("orient(a, b, c) = <b>" + o + "</b> → " + GX.sgnWord(s)
      + " &nbsp;·&nbsp; orientation tests evaluated: <b>" + k.get("orient") + "</b>"
      + " &nbsp;·&nbsp; swapping b and c flips the sign: orient(a, c, b) = <b>" + GR.orient(a, c, b) + "</b>"
      + " &nbsp;·&nbsp; the cyclic shift keeps it: orient(b, c, a) = <b>" + GR.orient(b, c, a) + "</b> "
      + GX.flag(GR.orient(b, c, a) === o && GR.orient(a, c, b) === -o));
  }
});

dragExplorer({
  svg: "#sg-svg", select: "#sg-preset", names: ["a", "b", "c", "d"],
  presets: [
    { name: "proper crossing", pts: [[1, 1], [12, 7], [2, 7], [11, 2]] },
    { name: "straddles one way only", pts: [[1, 1], [12, 7], [5, 6], [3, 8]] },
    { name: "endpoint touches (T)", pts: [[1, 1], [11, 6], [5, 3], [7, 8]] },
    { name: "collinear, overlapping", pts: [[1, 1], [9, 5], [5, 3], [13, 7]] },
    { name: "collinear, disjoint", pts: [[1, 1], [5, 3], [9, 5], [13, 7]] },
    { name: "parallel", pts: [[1, 2], [9, 6], [3, 1], [11, 5]] },
    { name: "shared endpoint", pts: [[2, 2], [8, 6], [8, 6], [13, 1]] }
  ],
  draw: function (pts, P, g, side) {
    const [a, b, c, d] = pts, k = AL.counter();
    const r = GR.segIntersect(a, b, c, d, k);
    const line = (p, q, col) => g.append("line").attr("x1", P.sx(p.x)).attr("y1", P.sy(p.y)).attr("x2", P.sx(q.x)).attr("y2", P.sy(q.y)).attr("stroke", col).attr("stroke-width", 3).attr("stroke-linecap", "round");
    line(a, b, AC.accent); line(c, d, AC.a2);
    if (r.hit && r.kind === "proper") { const x = GR.interPoint({ a, b }, { a: c, b: d }); g.append("circle").attr("cx", P.sx(x.x)).attr("cy", P.sy(x.y)).attr("r", 5).attr("fill", AC.good); }
    const sym = v => v > 0 ? "+" : (v < 0 ? "−" : "0");
    const rows = [["orient(a, b, c)", r.o[0]], ["orient(a, b, d)", r.o[1]], ["orient(c, d, a)", r.o[2]], ["orient(c, d, b)", r.o[3]]];
    rows.forEach((row, i) => { GX.txt(side, 0, i * 24, row[0], { size: 12, mono: true, fill: AC.muted }); GX.txt(side, 150, i * 24, sym(row[1]), { size: 14, mono: true, fill: row[1] ? AC.ink : AC.a2 }); });
    GX.txt(side, 0, 112, "c, d on opposite sides of ab:", { size: 11, fill: AC.muted }); GX.txt(side, 190, 112, r.o[0] * r.o[1] < 0 ? "yes" : "no", { size: 12, fill: r.o[0] * r.o[1] < 0 ? AC.good : AC.bad });
    GX.txt(side, 0, 134, "a, b on opposite sides of cd:", { size: 11, fill: AC.muted }); GX.txt(side, 190, 134, r.o[2] * r.o[3] < 0 ? "yes" : "no", { size: 12, fill: r.o[2] * r.o[3] < 0 ? AC.good : AC.bad });
    GX.txt(side, 0, 170, "verdict", { size: 11, fill: AC.muted });
    GX.txt(side, 0, 190, r.hit ? "INTERSECT" : "DISJOINT", { size: 15, fill: r.hit ? AC.good : AC.bad });
    GX.txt(side, 0, 210, "case: " + r.kind, { size: 12, fill: AC.ink });
    /* independent check: solve a + t(b − a) = c + s(d − c) exactly in integers,
       with no orientation predicate — a different algorithm for the same answer */
    const alg = (function () {
      const cr = (u, v) => u.x * v.y - u.y * v.x;
      const r = { x: b.x - a.x, y: b.y - a.y }, sv = { x: d.x - c.x, y: d.y - c.y }, w = { x: c.x - a.x, y: c.y - a.y };
      let den = cr(r, sv), tn = cr(w, sv), sn = cr(w, r);
      if (den !== 0) { if (den < 0) { den = -den; tn = -tn; sn = -sn; } return tn >= 0 && tn <= den && sn >= 0 && sn <= den; }
      if (cr(w, r) !== 0) return false;                        // parallel, not collinear
      const ax = (r.x !== 0 || sv.x !== 0) ? "x" : "y";        // collinear: overlap of 1-D intervals
      const lo1 = Math.min(a[ax], b[ax]), hi1 = Math.max(a[ax], b[ax]), lo2 = Math.min(c[ax], d[ax]), hi2 = Math.max(c[ax], d[ax]);
      return Math.max(lo1, lo2) <= Math.min(hi1, hi2);
    })();
    d3.select("#sg-readout").html("orientation tests: <b>" + k.get("orient") + "</b> &nbsp;·&nbsp; general case needs both straddles; a zero sign sends the test to the collinear branch, which checks the bounding box"
      + " &nbsp;·&nbsp; verdict <b>" + (r.hit ? "intersect" : "disjoint") + "</b> (" + r.kind + ")"
      + " &nbsp;·&nbsp; exact parametric solve (no orientation tests) says <b>" + (alg ? "intersect" : "disjoint") + "</b> " + GX.flag(alg === r.hit));
  }
});

/* ══ FIGURE 4 — the sweep line, stepped event by event ════════════════════ */
(function () {
  const svg = d3.select("#sw-svg"); if (svg.empty()) return;
  const P = GX.plane(24, 18, 440, 300, [0, 1], [0, 1]);
  function build() {
    const n = +d3.select("#sw-n").property("value"), seed = +d3.select("#sw-seed").property("value");
    d3.select("#sw-n-out").text(n); d3.select("#sw-seed-out").text(seed);
    const segs = GR.randomSegments(n, seed), k = AL.counter(), frames = [];
    const res = GR.sweep(segs, k, frames);
    const kb = AL.counter(), bf = GR.bruteIntersections(segs, kb);
    const got = res.found.map(f => Math.min(f.a, f.b) + "-" + Math.max(f.a, f.b)).sort().join();
    const ok = got === bf.slice().sort().join();
    GX.clearControls(svg.node());
    function render(f) {
      svg.selectAll("*").remove();
      svg.append("rect").attr("x", 24).attr("y", 18).attr("width", 440).attr("height", 300).attr("fill", "none").attr("stroke", AC.line);
      const active = new Set(f.status);
      segs.forEach(s => {
        const done = s.b.x < f.x, act = active.has(s.id);
        svg.append("line").attr("x1", P.sx(s.a.x)).attr("y1", P.sy(s.a.y)).attr("x2", P.sx(s.b.x)).attr("y2", P.sy(s.b.y))
          .attr("stroke", GX.segColor(s.id)).attr("stroke-width", act ? 2.6 : 1.4).attr("opacity", act ? 1 : (done ? 0.25 : 0.55));
        GX.txt(svg, P.sx(s.a.x) - 4, P.sy(s.a.y) + 4, s.id, { size: 10, anchor: "end", fill: GX.segColor(s.id) });
      });
      f.pending.forEach(p => svg.append("circle").attr("cx", P.sx(p.x)).attr("cy", P.sy(p.y)).attr("r", 5).attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 1.5));
      f.found.forEach(p => svg.append("circle").attr("cx", P.sx(p.x)).attr("cy", P.sy(p.y)).attr("r", 4.5).attr("fill", AC.good));
      svg.append("line").attr("x1", P.sx(f.x)).attr("x2", P.sx(f.x)).attr("y1", 18).attr("y2", 318).attr("stroke", AC.ink).attr("stroke-width", 1.5).attr("stroke-dasharray", "5 3");
      GX.txt(svg, P.sx(f.x), 334, f.type === "cross" ? "crossing " + f.ids.join("×") : f.type + " end of " + f.ids[0], { size: 11, anchor: "middle", fill: AC.a2 });
      /* the status structure, top of the sweep line first */
      const sx = 490;
      GX.txt(svg, sx, 30, "status (top → bottom)", { size: 11, fill: AC.muted });
      const rh = Math.min(19, 280 / Math.max(1, f.status.length));
      f.status.slice().reverse().forEach((id, i) => {
        const hot = f.ids.indexOf(id) >= 0;
        svg.append("rect").attr("x", sx).attr("y", 40 + i * rh).attr("width", 60).attr("height", rh - 3).attr("rx", 3)
          .attr("fill", AC.panel2).attr("stroke", hot ? AC.a2 : AC.line);
        svg.append("rect").attr("x", sx + 4).attr("y", 40 + i * rh + (rh - 3) / 2 - 2).attr("width", 12).attr("height", 4).attr("fill", GX.segColor(id));
        GX.txt(svg, sx + 40, 40 + i * rh + (rh - 3) / 2 + 4, "s" + id, { size: Math.min(11, rh - 4), anchor: "middle" });
      });
      const cx = 570, L = [["events", f.c.events], ["orientation tests", f.c.orient], ["status comparisons", f.c.statusCmp], ["neighbour pair tests", f.c.pairTests], ["found so far", f.found.length], ["queued crossings", f.pending.length]];
      L.forEach((r, i) => { GX.txt(svg, cx, 40 + i * 34, r[0], { size: 10, fill: AC.muted }); GX.txt(svg, cx, 55 + i * 34, r[1], { size: 13, mono: true }); });
    }
    AL.stepper(svg, { frames, render, delay: 650, label: "event" });
    const pairs = n * (n - 1) / 2, kk = res.found.length;
    d3.select("#sw-readout").html("n = " + n + " segments, k = <b>" + kk + "</b> intersections &nbsp;·&nbsp; sweep: <b>" + GX.int(k.get("events")) + "</b> events (2n + k = " + (2 * n + kk) + "), <b>"
      + GX.int(k.get("pairTests")) + "</b> neighbour pair tests, <b>" + GX.int(k.get("statusCmp")) + "</b> status comparisons, <b>" + GX.int(k.get("pqCmp")) + "</b> heap comparisons, <b>" + GX.int(k.get("orient")) + "</b> orientation tests"
      + " &nbsp;·&nbsp; brute force: <b>" + GX.int(kb.get("pairTests")) + "</b> pair tests (= n(n−1)/2 = " + pairs + "), <b>" + GX.int(kb.get("orient")) + "</b> orientation tests"
      + " &nbsp;·&nbsp; same " + bf.length + " pairs " + GX.flag(ok)
      + " &nbsp;·&nbsp; any-segments-intersect would stop at event <b>" + (res.firstHitEvent < 0 ? "— (none: it scans all 2n)" : res.firstHitEvent) + "</b>");
  }
  d3.select("#sw-n").on("input", build);
  d3.select("#sw-seed").on("input", build);
  build();
})();

/* ══ FIGURE 5 — convex hull, stepped ═══════════════════════════════════════ */
(function () {
  const svg = d3.select("#hl-svg"); if (svg.empty()) return;
  const P = GX.plane(24, 14, 330, 330, [0, 100], [0, 100]);
  const ALG = { graham: "Graham scan", monotone: "monotone chain", jarvis: "Jarvis march", quickhull: "quickhull" };
  function build() {
    const alg = d3.select("#hl-alg").property("value"), dist = d3.select("#hl-dist").property("value");
    const n = +d3.select("#hl-n").property("value"), seed = +d3.select("#hl-seed").property("value");
    d3.select("#hl-n-out").text(n); d3.select("#hl-seed-out").text(seed);
    const pts = GR.randomPoints(n, seed, dist);
    /* every algorithm on the same set, each under its own counter */
    const runs = {};
    Object.keys(ALG).forEach(a => { const k = AL.counter(); const fr = []; const h = GR[a](pts, k, a === "quickhull" ? null : fr); runs[a] = { k, h, fr }; });
    const ref = GR.hullKey(runs.graham.h);
    const frames = runs[alg].fr.concat([{ st: runs[alg].h, final: true, note: "hull complete: h = " + runs[alg].h.length }]);
    GX.clearControls(svg.node());
    const distinct = new Set(pts.map(p => p.x + "," + p.y)).size;
    function render(f) {
      svg.selectAll("*").remove();
      svg.append("rect").attr("x", 24).attr("y", 14).attr("width", 330).attr("height", 330).attr("fill", "none").attr("stroke", AC.line);
      if (f.done) svg.append("path").attr("d", d3.line().x(p => P.sx(p.x)).y(p => P.sy(p.y))(f.done)).attr("fill", "none").attr("stroke", AC.good).attr("stroke-width", 2).attr("opacity", 0.6);
      if (f.final) svg.append("path").attr("d", d3.line().x(p => P.sx(p.x)).y(p => P.sy(p.y))(f.st.concat([f.st[0]]))).attr("fill", AC.good).attr("fill-opacity", 0.08).attr("stroke", AC.good).attr("stroke-width", 2.2);
      else svg.append("path").attr("d", d3.line().x(p => P.sx(p.x)).y(p => P.sy(p.y))(f.st)).attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 2.2);
      pts.forEach(p => svg.append("circle").attr("cx", P.sx(p.x)).attr("cy", P.sy(p.y)).attr("r", 3.5).attr("fill", AC.muted));
      f.st.forEach(p => svg.append("circle").attr("cx", P.sx(p.x)).attr("cy", P.sy(p.y)).attr("r", 4.5).attr("fill", f.final ? AC.good : AC.accent));
      if (f.test) {
        const [a, b, c] = f.test, ok = f.o > 0;
        svg.append("line").attr("x1", P.sx(b.x)).attr("y1", P.sy(b.y)).attr("x2", P.sx(c.x)).attr("y2", P.sy(c.y)).attr("stroke", ok ? AC.good : AC.bad).attr("stroke-width", 2).attr("stroke-dasharray", "4 3");
        [a, b].forEach(p => svg.append("circle").attr("cx", P.sx(p.x)).attr("cy", P.sy(p.y)).attr("r", 7).attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 1.5));
      }
      if (f.cand) svg.append("circle").attr("cx", P.sx(f.cand.x)).attr("cy", P.sy(f.cand.y)).attr("r", 7).attr("fill", "none").attr("stroke", AC.rose).attr("stroke-width", 2);
      const x0 = 380;
      GX.txt(svg, x0, 30, ALG[alg], { size: 14, fill: AC.a2 });
      GX.txt(svg, x0, 52, f.test ? ("orient(below-top, top, next) = " + f.o) : (f.note || ""), { size: 11.5, mono: !!f.test });
      if (f.test) GX.txt(svg, x0, 70, f.o > 0 ? "left turn → keep the stack, push next" : (f.o === 0 ? "collinear → pop (strict hull)" : "right turn → pop the top"), { size: 11.5, fill: f.o > 0 ? AC.good : AC.bad });
      GX.txt(svg, x0, 100, (alg === "jarvis" ? "hull so far" : "stack") + " (" + f.st.length + "):", { size: 11, fill: AC.muted });
      const ids = f.st.map(p => "(" + p.x + "," + p.y + ")");
      for (let r = 0; r * 4 < ids.length && r < 8; r++) GX.txt(svg, x0, 118 + r * 16, ids.slice(r * 4, r * 4 + 4).join(" ") + (r === 7 && ids.length > 32 ? " …" : ""), { size: 10.5, mono: true });
      GX.txt(svg, x0, 268, "n = " + n + " (" + distinct + " distinct), h = " + runs[alg].h.length, { size: 11, fill: AC.muted });
      const kk = runs[alg].k;
      GX.txt(svg, x0, 288, "whole run: " + kk.get("orient") + " orientation tests, " + kk.get("cmp") + " sort comparisons", { size: 11, fill: AC.muted });
      GX.txt(svg, x0, 306, "pushes " + kk.get("push") + ", pops " + kk.get("pop") + (alg === "graham" || alg === "monotone" ? "  (pops ≤ pushes)" : ""), { size: 11, fill: AC.muted });
    }
    AL.stepper(svg, { frames, render, delay: alg === "jarvis" ? 700 : 260, label: alg === "jarvis" ? "wrap" : "step" });
    const rows = Object.keys(ALG).map(a => {
      const r = runs[a], cert = GR.hullCertificate(r.h, pts);
      return ["<b>" + ALG[a] + "</b>" + (a === alg ? " ◀" : ""), GX.int(r.k.get("orient")), GX.int(r.k.get("cmp")), a === "quickhull" ? r.k.get("calls") + " calls" : (a === "jarvis" ? "— (no stack)" : r.k.get("push") + " / " + r.k.get("pop")),
              r.h.length, (cert ? '<span style="color:' + AC.good + '">✓ convex, contains all</span>' : '<span style="color:' + AC.bad + '">✗ fails</span>'), GX.flag(GR.hullKey(r.h) === ref)];
    });
    GX.table("#hl-table", ["algorithm", "orientation tests", "sort comparisons", "push / pop", "h", "certificate", "same vertex set as Graham"], rows);
    const nh = n * runs.graham.h.length;
    d3.select("#hl-readout").html("n·h = " + GX.int(nh) + " &nbsp;·&nbsp; n·log₂n ≈ " + GX.int(Math.round(n * Math.log2(n)))
      + " &nbsp;·&nbsp; Jarvis measured <b>" + GX.int(runs.jarvis.k.get("orient")) + "</b> orientation tests, Graham <b>" + GX.int(runs.graham.k.get("orient")) + "</b> (sort included)"
      + " &nbsp;·&nbsp; in convex position every point is a hull vertex, h = n, and Jarvis goes quadratic");
  }
  ["#hl-alg", "#hl-dist"].forEach(s => d3.select(s).on("change", build));
  ["#hl-n", "#hl-seed"].forEach(s => d3.select(s).on("input", build));
  build();
})();

/* ══ FIGURE 6 — closest pair: divide and conquer against the random grid ══ */
(function () {
  const svg = d3.select("#cq-svg"); if (svg.empty()) return;
  const P = GX.plane(24, 14, 310, 310, [0, 100], [0, 100]);
  function build() {
    const n = +d3.select("#cq-n").property("value"), seed = +d3.select("#cq-seed").property("value");
    d3.select("#cq-n-out").text(n); d3.select("#cq-seed-out").text(seed);
    const pts = GR.randomFloatPoints(n, seed);
    const kb = AL.counter(), kd = AL.counter(), kg = AL.counter();
    const B = GR.closestBrute(pts, kb), D = GR.closestDC(pts, kd), G = GR.closestGrid(pts, seed, kg);
    svg.selectAll("*").remove();
    svg.append("rect").attr("x", 24).attr("y", 14).attr("width", 310).attr("height", 310).attr("fill", "none").attr("stroke", AC.line);
    /* the 3×3 block of grid cells (side δ) around the first point of the pair */
    const d = G.d, p0 = G.pair[0], cx = Math.floor(p0.x / d), cy = Math.floor(p0.y / d);
    for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) {
      const x0 = (cx + i) * d, y0 = (cy + j) * d;
      svg.append("rect").attr("x", P.sx(x0)).attr("y", P.sy(y0 + d)).attr("width", P.sx(x0 + d) - P.sx(x0)).attr("height", P.sy(y0) - P.sy(y0 + d))
        .attr("fill", i === 0 && j === 0 ? AC.a2 : "none").attr("fill-opacity", 0.15).attr("stroke", AC.a2).attr("stroke-width", 0.8);
    }
    pts.forEach(p => svg.append("circle").attr("cx", P.sx(p.x)).attr("cy", P.sy(p.y)).attr("r", n > 300 ? 1.8 : 2.6).attr("fill", AC.muted));
    svg.append("line").attr("x1", P.sx(B.pair[0].x)).attr("y1", P.sy(B.pair[0].y)).attr("x2", P.sx(B.pair[1].x)).attr("y2", P.sy(B.pair[1].y)).attr("stroke", AC.good).attr("stroke-width", 2.5);
    B.pair.forEach(p => svg.append("circle").attr("cx", P.sx(p.x)).attr("cy", P.sy(p.y)).attr("r", 4).attr("fill", AC.good));
    /* the packing schematic: the δ × 2δ box above a strip point p */
    const g = svg.append("g").attr("transform", "translate(400,40)"), u = 110;
    GX.txt(g, 0, -14, "the box a strip point p must search", { size: 11, fill: AC.muted });
    g.append("rect").attr("width", 2 * u).attr("height", u).attr("fill", "none").attr("stroke", AC.ink);
    g.append("line").attr("x1", u).attr("x2", u).attr("y1", -6).attr("y2", u + 6).attr("stroke", AC.a2).attr("stroke-dasharray", "4 3");
    [u / 2, 3 * u / 2].forEach(x => g.append("line").attr("x1", x).attr("x2", x).attr("y1", 0).attr("y2", u).attr("stroke", AC.line));
    g.append("line").attr("x1", 0).attr("x2", 2 * u).attr("y1", u / 2).attr("y2", u / 2).attr("stroke", AC.line);
    [[0, 0], [u, 0], [0, u], [u, u], [u, 0], [2 * u, 0], [u, u], [2 * u, u]].forEach((c, i) =>
      g.append("circle").attr("cx", c[0] + (i >= 4 && c[0] === u ? 5 : (i < 4 && c[0] === u ? -5 : 0))).attr("cy", c[1]).attr("r", 4).attr("fill", i < 4 ? AC.accent : AC.teal));
    GX.txt(g, u / 2, u + 18, "L: ≤ 4", { size: 11, anchor: "middle", fill: AC.accent });
    GX.txt(g, 3 * u / 2, u + 18, "R: ≤ 4", { size: 11, anchor: "middle", fill: AC.teal });
    GX.txt(g, -6, u / 2 + 4, "δ", { size: 12, anchor: "end" });
    GX.txt(g, u, u + 34, "width 2δ, height δ — each δ × δ half holds ≤ 4 points", { size: 10.5, anchor: "middle", fill: AC.muted });
    GX.txt(g, u, u + 48, "pairwise ≥ δ apart, so ≤ 7 partners besides p", { size: 10.5, anchor: "middle", fill: AC.muted });
    GX.txt(g, u, u + 74, "measured: most forward partners any strip point", { size: 10.5, anchor: "middle", fill: AC.muted });
    GX.txt(g, u, u + 90, "examined = " + D.maxPartners + "   ·   most points in one grid cell = " + G.maxCell, { size: 11.5, anchor: "middle", fill: AC.a2 });
    GX.txt(g, u, u + 106, "most points in a 3×3 neighbourhood = " + G.maxNbr + "  (bound 36)", { size: 11.5, anchor: "middle", fill: AC.a2 });
    const ok = B.d === D.d && B.d === G.d;
    GX.table("#cq-table", ["method", "distance evaluations", "other work (measured)", "δ found", "agrees with brute force"], [
      ["brute force, all pairs", GX.int(kb.get("dist")), "— (= n(n−1)/2 = " + GX.int(n * (n - 1) / 2) + ")", B.d.toFixed(4), "—"],
      ["divide &amp; conquer + strip", GX.int(kd.get("dist")), GX.int(kd.get("stripDist")) + " of them in strips; max partners per strip point " + D.maxPartners, D.d.toFixed(4), GX.flag(B.d === D.d)],
      ["randomized incremental grid", GX.int(kg.get("dist")), G.rebuilds + " rebuilds, " + GX.int(kg.get("gridInsert")) + " grid insertions (" + (kg.get("gridInsert") / n).toFixed(2) + " per point)", G.d.toFixed(4), GX.flag(B.d === G.d)]
    ]);
    d3.select("#cq-readout").html("n = " + n + " &nbsp;·&nbsp; δ = <b>" + B.d.toFixed(4) + "</b> &nbsp;·&nbsp; distance evaluations: brute <b>" + GX.int(kb.get("dist")) + "</b>, D&amp;C <b>" + GX.int(kd.get("dist")) + "</b>, grid <b>" + GX.int(kg.get("dist")) + "</b>"
      + " &nbsp;·&nbsp; grid work per point (insertions + distances) <b>" + ((kg.get("gridInsert") + kg.get("dist")) / n).toFixed(2) + "</b> — flat in n is what expected O(n) looks like &nbsp;·&nbsp; " + GX.flag(ok));
  }
  d3.select("#cq-n").on("input", build);
  d3.select("#cq-seed").on("input", build);
  build();
})();

/* ══ FIGURE 7 — polygons: area, ray casting, winding number, convexity ════ */
(function () {
  const svg = d3.select("#pp-svg"); if (svg.empty()) return;
  const P = GX.plane(24, 16, 400, 280, [0, 40], [0, 28]);
  const star = [0, 2, 4, 1, 3].map(i => { const t = Math.PI / 2 + i * 2 * Math.PI / 5; return { x: Math.round(20 + 13 * Math.cos(t)), y: Math.round(14 + 13 * Math.sin(t)) }; });
  const oct = d3.range(8).map(i => { const t = Math.PI / 8 + i * Math.PI / 4; return { x: Math.round(20 + 12 * Math.cos(t)), y: Math.round(14 + 12 * Math.sin(t)) }; });
  const POLY = {
    comb: [[4, 4], [36, 4], [36, 24], [30, 24], [30, 10], [24, 10], [24, 24], [18, 24], [18, 10], [12, 10], [12, 24], [4, 24]].map(p => ({ x: p[0], y: p[1] })),
    star: star, octagon: oct,
    collinear: [[6, 4], [20, 4], [34, 4], [34, 24], [6, 24]].map(p => ({ x: p[0], y: p[1] })),
    clockwise: [[4, 4], [36, 4], [36, 24], [30, 24], [30, 10], [24, 10], [24, 24], [18, 24], [18, 10], [12, 10], [12, 24], [4, 24]].reverse().map(p => ({ x: p[0], y: p[1] }))
  };
  const q = { x: 21, y: 14 };
  const layer = svg.append("g");
  const handle = svg.append("circle").attr("r", 7).attr("fill", AC.a2).attr("stroke", AC.ink).style("cursor", "grab")
    .call(d3.drag().on("drag", ev => { q.x = AL.clamp(Math.round(P.sx.invert(ev.x)), 0, 40); q.y = AL.clamp(Math.round(P.sy.invert(ev.y)), 0, 28); syncSliders(); draw(); }));
  function syncSliders() { d3.select("#pp-qx").property("value", q.x); d3.select("#pp-qy").property("value", q.y); d3.select("#pp-qx-out").text(q.x); d3.select("#pp-qy-out").text(q.y); }
  function draw() {
    const which = d3.select("#pp-poly").property("value"), rule = d3.select("#pp-rule").property("value");
    const V = POLY[which];
    layer.selectAll("*").remove();
    GX.gridLines(layer, P, d3.range(0, 41, 4), d3.range(0, 29, 4));
    layer.append("path").attr("d", "M" + V.map(p => P.sx(p.x) + "," + P.sy(p.y)).join("L") + "Z")
      .attr("fill", AC.accent).attr("fill-opacity", 0.22).attr("fill-rule", rule).attr("stroke", AC.accent).attr("stroke-width", 2);
    V.forEach((p, i) => { layer.append("circle").attr("cx", P.sx(p.x)).attr("cy", P.sy(p.y)).attr("r", 3).attr("fill", AC.ink);
      GX.txt(layer, P.sx(p.x) + 5, P.sy(p.y) - 5, i, { size: 9, fill: AC.muted }); });
    const kr = AL.counter(), kw = AL.counter(), ka = AL.counter(), kc = AL.counter();
    const R = GR.rayCrossings(V, q, kr), w = GR.winding(V, q, kw), A2 = GR.shoelace(V, ka), cv = GR.convexity(V, kc);
    const onB = GR.onBoundary(V, q);
    layer.append("line").attr("x1", P.sx(q.x)).attr("y1", P.sy(q.y)).attr("x2", P.sx(40)).attr("y2", P.sy(q.y)).attr("stroke", AC.a2).attr("stroke-dasharray", "4 3");
    R.xs.forEach(x => layer.append("circle").attr("cx", P.sx(x)).attr("cy", P.sy(q.y)).attr("r", 4.5).attr("fill", "none").attr("stroke", AC.rose).attr("stroke-width", 2));
    handle.attr("cx", P.sx(q.x)).attr("cy", P.sy(q.y)).raise();
    const s = layer.append("g").attr("transform", "translate(450,30)");
    const L = [
      ["signed area (shoelace ÷ 2)", (A2 / 2) + (A2 > 0 ? "  → counter-clockwise" : "  → clockwise")],
      ["ray crossings", R.c + "  → even–odd: " + (R.c % 2 ? "INSIDE" : "outside")],
      ["winding number", w + "  → nonzero: " + (w !== 0 ? "INSIDE" : "outside")],
      ["on the boundary?", onB ? "yes — both rules undefined here" : "no"],
      ["turn signs (+ / −)", cv.pos + " / " + cv.neg + (cv.sameSign ? "  all one sign" : "  mixed")],
      ["Δx direction changes", cv.flips + (cv.flips <= 2 ? "  (≤ 2)" : "  (> 2: winds more than once)")],
      ["convex?", cv.convex ? "yes" : "no"]
    ];
    L.forEach((r, i) => { GX.txt(s, 0, i * 34, r[0], { size: 10.5, fill: AC.muted }); GX.txt(s, 0, i * 34 + 15, r[1], { size: 12, mono: true, fill: (i === 1 || i === 2) ? AC.a2 : AC.ink }); });
    let convexNote = "";
    if (cv.convex) {
      const ccw = A2 > 0 ? V : V.slice().reverse(), kl = AL.counter();
      const inC = GR.inConvex(ccw, q, kl);
      convexNote = " &nbsp;·&nbsp; O(log n) wedge search: <b>" + kl.get("orient") + "</b> orientation tests → " + (inC ? "inside" : "outside") + " " + GX.flag(onB || inC === (w !== 0));
    }
    d3.select("#pp-readout").html("q = (" + q.x + ", " + q.y + ") &nbsp;·&nbsp; shoelace: <b>" + ka.get("term") + "</b> cross terms &nbsp;·&nbsp; ray casting: <b>" + kr.get("edge") + "</b> edges, <b>" + kr.get("orient") + "</b> orientation tests"
      + " &nbsp;·&nbsp; winding: <b>" + kw.get("edge") + "</b> edges, <b>" + kw.get("orient") + "</b> orientation tests &nbsp;·&nbsp; the two rules "
      + ((R.c % 2 === 1) === (w !== 0) ? "agree" : "<b>disagree</b> (|winding| = " + Math.abs(w) + " is even — only possible for a self-intersecting boundary)")
      + " &nbsp;·&nbsp; parity check: crossings ≡ winding (mod 2) " + GX.flag((R.c - w) % 2 === 0) + convexNote);
  }
  ["#pp-poly", "#pp-rule"].forEach(sel => d3.select(sel).on("change", draw));
  d3.select("#pp-qx").on("input", function () { q.x = +this.value; syncSliders(); draw(); });
  d3.select("#pp-qy").on("input", function () { q.y = +this.value; syncSliders(); draw(); });
  syncSliders(); draw();
})();

/* ══ FIGURE 8 — kd-tree build + range query, and the √n line-query curve ═══ */
(function () {
  const svg = d3.select("#kd-svg"); if (svg.empty()) return;
  const P = GX.plane(20, 20, 330, 330, [0, 100], [0, 100]);
  const QUERIES = {
    square: { x0: 30, x1: 48, y0: 40, y1: 58 },
    strip: { x0: 49, x1: 51, y0: 0, y1: 100 },
    band: { x0: 5, x1: 95, y0: 44, y1: 52 },
    big: { x0: 20, x1: 80, y0: 20, y1: 80 }
  };
  /* measured once at load: kd nodes visited by a zero-width vertical line query */
  const curve = [16, 64, 256, 1024, 4096, 16384].map(n => {
    const pts = GR.randomFloatPoints(n, 1), t = GR.kdBuild(pts), k = AL.counter();
    GR.kdQuery(t, { x0: 50.123, x1: 50.123, y0: 0, y1: 100 }, k);
    return { n, v: k.get("visited") };
  });
  function build() {
    const n = +d3.select("#kd-n").property("value"), seed = +d3.select("#kd-seed").property("value"), qs = d3.select("#kd-q").property("value");
    d3.select("#kd-n-out").text(n); d3.select("#kd-seed-out").text(seed);
    const R = QUERIES[qs], pts = GR.randomFloatPoints(n, seed);
    const kb = AL.counter(), t = GR.kdBuild(pts, kb), kq = AL.counter(), vis = [];
    const out = GR.kdQuery(t, R, kq, vis);
    const krb = AL.counter(), rt = GR.rtBuild(pts, krb), krq = AL.counter(), outR = GR.rtQuery(rt, R, krq);
    const g = Math.max(1, Math.round(Math.sqrt(n / 2))), kg = AL.counter(), outG = GR.gridQuery(pts, R, g, kg);
    const bf = pts.filter(p => GR.inRect(p, R)).map(p => p.i).sort().join();
    const key = a => a.map(p => p.i).sort().join();
    svg.selectAll("*").remove();
    svg.append("rect").attr("x", 20).attr("y", 20).attr("width", 330).attr("height", 330).attr("fill", "none").attr("stroke", AC.line);
    const visited = new Set(vis);
    (function lines(node) {
      if (!node) return;
      const b = node.box, on = visited.has(node);
      const l = node.ax === "x" ? [node.p.x, b.y0, node.p.x, b.y1] : [b.x0, node.p.y, b.x1, node.p.y];
      svg.append("line").attr("x1", P.sx(l[0])).attr("y1", P.sy(l[1])).attr("x2", P.sx(l[2])).attr("y2", P.sy(l[3]))
        .attr("stroke", on ? AC.a2 : AC.line).attr("stroke-width", on ? 1.3 : 0.8).attr("opacity", on ? 0.9 : 0.8);
      lines(node.left); lines(node.right);
    })(t);
    svg.append("rect").attr("x", P.sx(R.x0)).attr("y", P.sy(R.y1)).attr("width", Math.max(1.5, P.sx(R.x1) - P.sx(R.x0))).attr("height", P.sy(R.y0) - P.sy(R.y1))
      .attr("fill", AC.good).attr("fill-opacity", 0.08).attr("stroke", AC.good).attr("stroke-width", 1.5);
    const hit = new Set(out);
    pts.forEach(p => svg.append("circle").attr("cx", P.sx(p.x)).attr("cy", P.sy(p.y)).attr("r", n > 300 ? 1.8 : 2.6).attr("fill", hit.has(p) ? AC.good : AC.muted));
    /* right: log–log plot of the measured line-query cost */
    const F = { x: 420, y: 40, w: 250, h: 250 };
    const x = d3.scaleLog().domain([16, 16384]).range([F.x, F.x + F.w]), y = d3.scaleLog().domain([2, 400]).range([F.y + F.h, F.y]);
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${F.y + F.h})`).call(d3.axisBottom(x).tickValues([16, 64, 256, 1024, 4096, 16384]).tickFormat(d3.format("~s")));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${F.x},0)`).call(d3.axisLeft(y).tickValues([2, 5, 10, 20, 50, 100, 200]).tickFormat(d3.format("~s")));
    GX.txt(svg, F.x, F.y - 14, "kd nodes visited by a vertical LINE query", { size: 11, fill: AC.muted });
    GX.txt(svg, F.x + F.w, F.y + F.h + 30, "n (log scale)", { size: 10.5, anchor: "end", fill: AC.muted });
    const ref = d3.range(16, 16385, 64).concat([16384]);
    [[1, "√n", "3 3"], [2, "2√n", null]].forEach(([c, lab, dash]) => {
      const pth = svg.append("path").attr("d", d3.line().x(v => x(v)).y(v => y(c * Math.sqrt(v)))(ref)).attr("fill", "none").attr("stroke", AC.muted).attr("stroke-width", 1.2);
      if (dash) pth.attr("stroke-dasharray", dash);
      GX.txt(svg, x(16384) - 4, y(c * 128) - 6, lab, { size: 10, anchor: "end", fill: AC.muted });
    });
    svg.append("path").attr("d", d3.line().x(d => x(d.n)).y(d => y(d.v))(curve)).attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 2);
    curve.forEach(d => svg.append("circle").attr("cx", x(d.n)).attr("cy", y(d.v)).attr("r", 3.5).attr("fill", AC.a2));
    GX.txt(svg, F.x + 8, F.y + 14, "measured (seed 1)", { size: 10.5, fill: AC.a2 });
    GX.table("#kd-table", ["structure", "build work (measured)", "space", "query work (measured)", "reported k", "agrees with brute force"], [
      ["kd-tree", GX.int(kb.get("cmp")) + " comparisons (sort per level)", n + " nodes", "<b>" + GX.int(kq.get("visited")) + "</b> nodes visited", out.length, GX.flag(key(out) === bf)],
      ["2-D range tree", GX.int(krb.get("cmp") + krb.get("merge")) + " comparisons + merges", GX.int(krb.get("stored")) + " stored y-entries (n log n)", "<b>" + GX.int(krq.get("visited")) + "</b> nodes, " + krq.get("canonical") + " canonical, " + GX.int(krq.get("bsearch")) + " binary-search steps", outR.length, GX.flag(key(outR) === bf)],
      ["uniform grid " + g + "×" + g, "n bucket insertions", n + " + " + (g * g) + " cells", "<b>" + GX.int(kg.get("cells")) + "</b> cells, " + GX.int(kg.get("examined")) + " points examined", outG.length, GX.flag(key(outG) === bf)],
      ["brute force", "—", "n", GX.int(n) + " points examined", bf ? bf.split(",").length : 0, "—"]
    ]);
    const last = curve[curve.length - 1];
    d3.select("#kd-readout").html("n = " + n + ", k = <b>" + out.length + "</b> &nbsp;·&nbsp; kd-tree visited <b>" + kq.get("visited") + "</b> of " + n + " nodes (√n = " + Math.sqrt(n).toFixed(1) + ")"
      + " &nbsp;·&nbsp; line-query curve: at n = " + GX.int(last.n) + " the measured count is " + last.v + " = " + (last.v / Math.sqrt(last.n)).toFixed(2) + "·√n");
  }
  d3.select("#kd-q").on("change", build);
  ["#kd-n", "#kd-seed"].forEach(s => d3.select(s).on("input", build));
  build();
})();

}

/* ═══════════════════════════════════════════════════════════════════════════
   GB — appended with §06: Bresenham's line algorithm. Integer-only; the
   counter records c.add("add") per integer addition/subtraction in the loop,
   c.add("cmp") per sign test of the error term, c.add("mul") per
   multiplication (the doublings 2·A, 2·B are done as additions, so 0).
   Node-loadable: the routines are added to module.exports alongside GR.
   ═══════════════════════════════════════════════════════════════════════════ */
const GB = (function () {
  const nop = { add: function () {}, get: function () { return 0; } };
  /* A = the major-axis length, B = the minor; D = 2·F(midpoint), where
     F(x, y) = B·(x − x₀) − A·(y − y₀) in the octant-normalised frame.
     D > 0: the midpoint is below the line, step the minor axis too. */
  function bresenham(x0, y0, x1, y1, c) {
    c = c || nop;
    const dx = Math.abs(x1 - x0), dy = Math.abs(y1 - y0), sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    const steep = dy > dx, A = steep ? dy : dx, B = steep ? dx : dy;
    const twoA = A + A, twoB = B + B; c.add("add", 2);
    let D = twoB - A; c.add("add");
    let x = x0, y = y0; const pts = [];
    for (let i = 0; i <= A; i++) {
      const p = { x: x, y: y, D: D, minor: false };
      pts.push(p);
      if (i === A) break;
      c.add("cmp");
      if (D > 0) { if (steep) x += sx; else y += sy; D -= twoA; c.add("add", 2); p.minor = true; }
      D += twoB; c.add("add");
      if (steep) y += sy; else x += sx; c.add("add");
    }
    return { pts: pts, A: A, B: B, steep: steep };
  }
  /* the floating-point DDA a first attempt writes: t = i/A, round the lerp */
  function floatDDA(x0, y0, x1, y1) {
    const A = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0)), pts = [];
    for (let i = 0; i <= A; i++) { const t = A ? i / A : 0; pts.push({ x: Math.round(x0 + t * (x1 - x0)), y: Math.round(y0 + t * (y1 - y0)) }); }
    return pts;
  }
  /* exact check: the minor coordinate of step i is within ½ of the true line,
     i.e. |2·(k·A − i·B)| ≤ A where k is the minor offset — integers only. */
  function withinHalf(r, x0, y0) {
    return r.pts.every((p, i) => { const k = r.steep ? Math.abs(p.x - x0) : Math.abs(p.y - y0); return Math.abs(2 * (k * r.A - i * r.B)) <= r.A; });
  }
  return { bresenham, floatDDA, withinHalf };
})();
if (typeof module !== "undefined" && module.exports) Object.assign(module.exports, GB);

/* ══ FIGURE 9 — Bresenham's line: drag the endpoints, watch the error term ══ */
if (typeof document !== "undefined" && typeof d3 !== "undefined") (function () {
  const svg = d3.select("#br-svg"); if (svg.empty()) return;
  const XM = 24, YM = 14;
  const sx = d3.scaleLinear().domain([-0.5, XM + 0.5]).range([24, 24 + 25 * 17.5]);
  const sy = d3.scaleLinear().domain([-0.5, YM + 0.5]).range([16 + 15 * 17.5, 16]);
  const U = 17.5;
  const flag = ok => ok ? '<span style="color:' + AC.good + '">✓ agree</span>' : '<span style="color:' + AC.bad + '">✗ DISAGREE</span>';
  const txt = (g, x, y, s, o) => { o = o || {}; return g.append("text").attr("x", x).attr("y", y).attr("font-size", o.size || 11).attr("fill", o.fill || AC.ink).attr("text-anchor", o.anchor || "start").attr("font-family", o.mono ? "SF Mono, Menlo, monospace" : null).attr("xml:space", o.mono ? "preserve" : null).style("white-space", o.mono ? "pre" : null).text(s); };
  const PRESETS = [
    { name: "shallow, first octant", p: [[2, 3], [21, 10]] },
    { name: "exact ties: dx = 2·dy", p: [[2, 2], [18, 10]] },
    { name: "steep", p: [[5, 1], [11, 13]] },
    { name: "negative slope", p: [[3, 12], [22, 4]] },
    { name: "right to left", p: [[21, 10], [2, 3]] },
    { name: "horizontal", p: [[3, 7], [20, 7]] },
    { name: "diagonal", p: [[4, 2], [16, 14]] }
  ];
  const pts = PRESETS[0].p.map(q => ({ x: q[0], y: q[1] }));
  svg.selectAll("*").remove();
  const grid = svg.append("g"), layer = svg.append("g"), side = svg.append("g").attr("transform", "translate(482,22)");
  for (let i = 0; i <= XM; i++) for (let j = 0; j <= YM; j++)
    grid.append("rect").attr("x", sx(i) - U / 2 + 0.5).attr("y", sy(j) - U / 2 + 0.5).attr("width", U - 1).attr("height", U - 1).attr("fill", AC.panel2).attr("stroke", AC.grid);
  for (let i = 0; i <= XM; i += 4) txt(grid, sx(i), sy(0) + 20, i, { size: 9, fill: AC.muted, anchor: "middle" });
  for (let j = 0; j <= YM; j += 2) txt(grid, sx(0) - 13, sy(j) + 3, j, { size: 9, fill: AC.muted, anchor: "end" });
  const handles = svg.append("g").selectAll("g").data(pts).join("g").style("cursor", "grab");
  handles.append("circle").attr("r", 8).attr("fill", AC.panel).attr("stroke", AC.ink).attr("stroke-width", 1.5);
  handles.append("text").attr("text-anchor", "middle").attr("y", 4).attr("font-size", 10).attr("fill", AC.ink).text((d, i) => "p" + i);
  handles.call(d3.drag().on("drag", function (ev, d) {
    d.x = AL.clamp(Math.round(sx.invert(ev.x)), 0, XM); d.y = AL.clamp(Math.round(sy.invert(ev.y)), 0, YM); draw();
  }));
  function draw() {
    const [a, b] = pts, c = AL.counter();
    const r = GB.bresenham(a.x, a.y, b.x, b.y, c);
    layer.selectAll("*").remove(); side.selectAll("*").remove();
    r.pts.forEach(p => layer.append("rect").attr("x", sx(p.x) - U / 2 + 1).attr("y", sy(p.y) - U / 2 + 1).attr("width", U - 2).attr("height", U - 2)
      .attr("rx", 2).attr("fill", AC.accent).attr("fill-opacity", 0.55).attr("stroke", AC.accent));
    /* midpoints the loop tested: half a pixel off the major axis, between the two candidates */
    r.pts.slice(0, -1).forEach((p, i) => {
      const mx = r.steep ? p.x + (b.x >= a.x ? 0.5 : -0.5) : p.x + (b.x >= a.x ? 1 : -1);
      const my = r.steep ? p.y + (b.y >= a.y ? 1 : -1) : p.y + (b.y >= a.y ? 0.5 : -0.5);
      layer.append("circle").attr("cx", sx(mx)).attr("cy", sy(my)).attr("r", 2.2).attr("fill", p.D > 0 ? AC.a2 : (p.D === 0 ? AC.rose : AC.muted));
    });
    layer.append("line").attr("x1", sx(a.x)).attr("y1", sy(a.y)).attr("x2", sx(b.x)).attr("y2", sy(b.y)).attr("stroke", AC.ink).attr("stroke-width", 1.4);
    handles.attr("transform", d => `translate(${sx(d.x)},${sy(d.y)})`);

    const dx = b.x - a.x, dy = b.y - a.y;
    txt(side, 0, 0, "dx = " + dx + ",  dy = " + dy + (r.steep ? "   y major" : "   x major"), { mono: true, size: 11 });
    txt(side, 0, 16, "A = " + r.A + ",  B = " + r.B + ",  D₀ = 2B − A = " + (2 * r.B - r.A), { mono: true, size: 11 });
    txt(side, 0, 36, "step   pixel      D  decision", { mono: true, size: 10, fill: AC.muted });
    txt(side, 0, 226, "stay:       D += 2B", { mono: true, size: 10, fill: AC.muted });
    txt(side, 0, 240, "minor step: D += 2B − 2A", { mono: true, size: 10, fill: AC.muted });
    const show = r.pts.slice(0, -1), MAXR = 11;
    show.slice(0, MAXR).forEach((p, i) => {
      const dec = p.D > 0 ? "minor step" : (p.D === 0 ? "tie: stay" : "stay");
      txt(side, 0, 52 + i * 14, String(i).padStart(3) + "  (" + String(p.x).padStart(2) + "," + String(p.y).padStart(2) + ")  " + String(p.D).padStart(4) + "   " + dec,
        { mono: true, size: 10, fill: p.D > 0 ? AC.a2 : (p.D === 0 ? AC.rose : AC.ink) });
    });
    if (show.length > MAXR) txt(side, 0, 52 + MAXR * 14, "… " + (show.length - MAXR) + " more step" + (show.length - MAXR === 1 ? "" : "s"), { size: 10, fill: AC.muted });
    txt(side, 0, 258, "dots = the midpoints tested:", { size: 9, fill: AC.muted });
    txt(side, 0, 270, "amber D > 0 · grey D < 0 · rose D = 0", { size: 9, fill: AC.muted });

    const ref = GB.floatDDA(a.x, a.y, b.x, b.y);
    const tieAt = i => i > 0 && r.pts[i - 1].D === 0;            // pixel i was chosen by a tie at step i − 1
    const diffIdx = r.pts.map((p, i) => i).filter(i => r.pts[i].x !== ref[i].x || r.pts[i].y !== ref[i].y);
    const diff = diffIdx.length, offTie = diffIdx.filter(i => !tieAt(i)).length;
    const ties = r.pts.slice(0, -1).filter(p => p.D === 0).length;
    const back = GB.bresenham(b.x, b.y, a.x, a.y), key = q => q.x + "," + q.y;
    const same = new Set(back.pts.map(key)), revDiff = r.pts.filter(p => !same.has(key(p))).length;
    d3.select("#br-readout").html("pixels lit <b>" + r.pts.length + "</b> = max(|dx|, |dy|) + 1 = " + (r.A + 1) + " " + flag(r.pts.length === r.A + 1)
      + " · integer additions <b>" + c.get("add") + "</b>, sign tests <b>" + c.get("cmp") + "</b>, multiplications <b>" + c.get("mul") + "</b>, divisions 0"
      + " · every pixel within ½ of the true line along the minor axis (checked in integers) " + flag(GB.withinHalf(r, a.x, a.y))
      + " · exact ties (D = 0): <b>" + ties + "</b> · the float DDA (t = i/A, round the lerp) lights " + (diff ? "<b>" + diff + "</b> different pixel" + (diff === 1 ? "" : "s") + " (Math.round rounds a half upward; the loop keeps the minor coordinate) — differences away from a tie: " + offTie + " " + flag(offTie === 0) : "the same pixels")
      + " · drawn from p1 to p0 instead: " + (revDiff ? "<b>" + revDiff + "</b> pixel" + (revDiff === 1 ? "" : "s") + " differ — the tie rule depends on direction" : "the same pixel set"));
  }
  const sel = d3.select("#br-preset");
  sel.selectAll("option").data(PRESETS).join("option").attr("value", (d, i) => i).text(d => d.name);
  sel.on("change", function () { PRESETS[+this.value].p.forEach((q, i) => { pts[i].x = q[0]; pts[i].y = q[1]; }); draw(); });
  draw();
})();
