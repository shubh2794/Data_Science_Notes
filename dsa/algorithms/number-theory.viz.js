/* number-theory.viz.js — figures for dsa/algorithms/number-theory.html
   (part 11 of the Algorithm Design & Analysis series: Number-Theoretic Algorithms).

   Loaded after ../../data.js, ../../notes.js and ../dsa-viz.js, so the globals
   AC (palette) and AL (rng/frame/counter/stepper/…) are available.

   House rule obeyed throughout: every count this page DISPLAYS — division
   steps, squarings, multiplications, trial divisions, liar counts, rho
   iterations — is produced by running the real routine under a counter and
   reading the counter back. Every ANSWER is cross-checked in the same figure
   against an independent computation (brute-force search, a second algorithm,
   or re-multiplying) and printed with an agree / DISAGREE flag.

   Values that can exceed 2⁵³ are BigInt. Routines that only ever see small
   moduli use Number arithmetic guarded by mulmod(), which switches to BigInt
   when a product could lose precision.

   Layout of this file:
     NT — the instrumented routines, pure and DOM-free (loadable in node:
          `require("./number-theory.viz.js")` returns NT).
     NX — small DOM helpers shared by the figures.
     figures — one block per <svg>, in page order:
        #bits-svg  cost against input size in bits: trial division vs the polynomial routines
        #eu-svg    Euclid: the square tiling, the Fibonacci worst case, extended-Euclid table
        #pw-svg    the power table aᵏ mod n: orders, Euler's theorem, primitive roots
        #me-svg    modular exponentiation by repeated squaring, stepped
        #crt-svg   the Chinese-remainder grid Z_m × Z_n ↔ Z_mn
        #mr-svg    Miller–Rabin: every base of one n, and liar fractions for all odd composites
        #rsa-svg   toy RSA: the permutation m ↦ mᵉ mod N and the CRT speed-up
        #rho-svg   Pollard's rho: the sequence mod the hidden factor, and iterations vs √p */

/* ═══════════════════════════════════════════════════════════════════════════
   NT — the instrumented routines
   ═══════════════════════════════════════════════════════════════════════════ */
const NT = (function () {
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
  /* a uniformly random BigInt of exactly `bits` bits (top bit set) */
  function randBig(bits, r) {
    let x = 1n;
    for (let i = 1; i < bits; i++) x = (x << 1n) | (r() < 0.5 ? 1n : 0n);
    return x;
  }
  const bitLen = x => (typeof x === "bigint" ? x.toString(2).length : (x === 0 ? 0 : Math.floor(Math.log2(x)) + 1));
  const popcount = x => { let c = 0; for (const ch of BigInt(x).toString(2)) if (ch === "1") c++; return c; };

  /* exact a·b mod n for Numbers: fast path while a·b < 2⁵³ */
  function mulmod(a, b, n) {
    if (a * b < 9007199254740992) return (a * b) % n;
    return Number((BigInt(a) * BigInt(b)) % BigInt(n));
  }
  function powmod(a, e, n) {                // Number, right-to-left; small n only
    let r = 1 % n, s = a % n;
    while (e > 0) { if (e & 1) r = mulmod(r, s, n); e = Math.floor(e / 2); if (e > 0) s = mulmod(s, s, n); }
    return r;
  }

  /* ── §05 Euclid ───────────────────────────────────────────────────────── */
  /* returns gcd and the division chain; k counts "div" (one a mod b each) */
  function euclid(a, b, k) {
    const rows = [];
    while (b !== 0) {
      ctr(k).add("div");
      const q = Math.floor(a / b), r = a - q * b;
      rows.push({ a, b, q, r });
      a = b; b = r;
    }
    return { g: a, rows };
  }
  function gcd(a, b) { while (b) { const t = a % b; a = b; b = t; } return a; }
  function gcdBig(a, b) { while (b) { const t = a % b; a = b; b = t; } return a < 0n ? -a : a; }

  /* ── §07 binary gcd: only shifts, subtractions and parity tests ───────── */
  function binaryGcd(a, b, k) {
    k = ctr(k);
    if (a === 0) return b; if (b === 0) return a;
    let sh = 0;
    while (((a | b) & 1) === 0) { a /= 2; b /= 2; sh++; k.add("shift"); }
    while ((a & 1) === 0) { a /= 2; k.add("shift"); }
    while (b !== 0) {
      while ((b & 1) === 0) { b /= 2; k.add("shift"); }
      if (a > b) { const t = a; a = b; b = t; }
      b -= a; k.add("sub");
    }
    return a * Math.pow(2, sh);
  }

  /* ── §08 extended Euclid, recursive, exactly as the page writes it ────── */
  /* rows are top-down: (a, b, ⌊a/b⌋) on the way down, (d, x, y) filled on the way up */
  function extEuclid(a, b, k, rows) {
    ctr(k).add("call");
    const row = { a, b, q: b ? Math.floor(a / b) : null, d: 0, x: 0, y: 0 };
    if (rows) rows.push(row);
    if (b === 0) { row.d = a; row.x = 1; row.y = 0; return { d: a, x: 1, y: 0 }; }
    const s = extEuclid(b, a % b, k, rows);
    const out = { d: s.d, x: s.y, y: s.x - Math.floor(a / b) * s.y };
    row.d = out.d; row.x = out.x; row.y = out.y;
    return out;
  }
  function extEuclidBig(a, b, k) {          // iterative, BigInt
    let [r0, r1, x0, x1, y0, y1] = [a, b, 1n, 0n, 0n, 1n];
    while (r1 !== 0n) {
      ctr(k).add("div");
      const q = r0 / r1;
      [r0, r1] = [r1, r0 - q * r1]; [x0, x1] = [x1, x0 - q * x1]; [y0, y1] = [y1, y0 - q * y1];
    }
    return { d: r0, x: x0, y: y0 };
  }
  function modInverse(a, n, k) {            // Number; null when gcd(a, n) ≠ 1
    const e = extEuclid(((a % n) + n) % n, n, k);
    return e.d !== 1 ? null : ((e.x % n) + n) % n;
  }
  function modInverseBig(a, n, k) {
    const e = extEuclidBig(((a % n) + n) % n, n, k);
    return e.d !== 1n ? null : ((e.x % n) + n) % n;
  }
  /* a·x ≡ b (mod n): all solutions in [0, n) */
  function solveLinear(a, b, n, k) {
    const e = extEuclid(((a % n) + n) % n, n, k), d = e.d;
    if (((b % d) + d) % d !== 0) return { d, sols: [] };
    const x0 = (((e.x * (b / d)) % n) + n) % n, sols = [];
    for (let i = 0; i < d; i++) sols.push((x0 + i * (n / d)) % n);
    return { d, sols };
  }

  /* ── §09–§12 the structure of Z_n ────────────────────────────────────── */
  function factor(n) {                      // trial division, small n
    const f = []; let m = n;
    for (let p = 2; p * p <= m; p++) { if (m % p === 0) { let e = 0; while (m % p === 0) { m /= p; e++; } f.push([p, e]); } }
    if (m > 1) f.push([m, 1]);
    return f;
  }
  function phi(n) { let r = n; factor(n).forEach(([p]) => { r = r / p * (p - 1); }); return r; }
  function order(a, n) {                    // multiplicative order, 0 if a not a unit
    if (gcd(a, n) !== 1) return 0;
    let x = a % n, k = 1;
    while (x !== 1 % n) { x = mulmod(x, a, n); k++; if (k > n) return 0; }
    return k;
  }
  function powerTable(n) {                  // rows a = 1..n−1, columns k = 1..n−1
    const rows = [];
    for (let a = 1; a < n; a++) {
      const r = []; let x = 1;
      for (let k = 1; k < n; k++) { x = mulmod(x, a, n); r.push(x); }
      rows.push(r);
    }
    return rows;
  }

  /* ── §13 modular exponentiation (BigInt), counted, with frames ───────── */
  function powLTR(a, b, n, k, frames) {
    k = ctr(k);
    if (b === 0n) return 1n % n;
    const bits = b.toString(2);
    let r = a % n;
    if (frames) frames.push({ i: 0, op: "start", r, bits });
    for (let i = 1; i < bits.length; i++) {
      r = (r * r) % n; k.add("sq");
      if (frames) frames.push({ i, op: "square", r, bits });
      if (bits[i] === "1") { r = (r * a) % n; k.add("mul"); if (frames) frames.push({ i, op: "multiply", r, bits }); }
    }
    return r;
  }
  function powRTL(a, b, n, k, frames) {
    k = ctr(k);
    const bits = b.toString(2);
    let r = 1n % n, s = a % n, i = bits.length - 1;
    while (b > 0n) {
      if (b & 1n) { r = (r * s) % n; k.add("mul"); if (frames) frames.push({ i, op: "multiply", r, s, bits }); }
      b >>= 1n;
      if (b > 0n) { s = (s * s) % n; k.add("sq"); if (frames) frames.push({ i: i - 1, op: "square", r, s, bits }); }
      i--;
    }
    return r;
  }
  function powNaive(a, b, n, k) { let r = 1n % n; for (let i = 0n; i < b; i++) { r = (r * a) % n; ctr(k).add("mul"); } return r; }

  /* ── §14 Chinese remainder theorem ───────────────────────────────────── */
  /* x ≡ rᵢ (mod mᵢ) for pairwise-coprime mᵢ (Numbers, small) */
  function crt(rs, ms, k) {
    const M = ms.reduce((p, m) => p * m, 1);
    let x = 0; const terms = [];
    for (let i = 0; i < ms.length; i++) {
      const Mi = M / ms[i], inv = modInverse(Mi % ms[i], ms[i], k);
      if (inv === null) return null;
      const c = Mi * inv;                     // ≡ 1 mod mᵢ, ≡ 0 mod every other mⱼ
      terms.push({ Mi, inv, c });
      x = (x + mulmod(rs[i] % ms[i], c % M, M)) % M;
    }
    return { x, M, terms };
  }

  /* ── §16–§17 primality ───────────────────────────────────────────────── */
  function trialDivision(n, k) {            // Number < 2⁵³; returns smallest factor or n
    k = ctr(k);
    if (n % 2 === 0) { k.add("div"); return n === 2 ? n : 2; }
    for (let d = 3; d * d <= n; d += 2) { k.add("div"); if (n % d === 0) return d; }
    return n;
  }
  function fermatLiar(a, n) { return powmod(a, n - 1, n) === 1; }
  function decompose(n) { let d = n - 1, s = 0; while (d % 2 === 0) { d /= 2; s++; } return { s, d }; }
  /* one strong-probable-prime round, Number n; true = "a says probably prime" */
  function strongLiar(a, n, sd) {
    const { s, d } = sd || decompose(n);
    let x = powmod(a, d, n);
    if (x === 1 || x === n - 1) return true;
    for (let r = 1; r < s; r++) { x = mulmod(x, x, n); if (x === n - 1) return true; }
    return false;
  }
  /* Miller–Rabin on BigInt with given bases; k counts mulmods ("mul") and rounds */
  function millerRabin(n, bases, k) {
    k = ctr(k);
    if (n < 2n) return false;
    for (const p of [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n]) { if (n === p) return true; if (n % p === 0n) return false; }
    let d = n - 1n, s = 0;
    while ((d & 1n) === 0n) { d >>= 1n; s++; }
    for (const b0 of bases) {
      const a = BigInt(b0) % n; if (a === 0n) continue;
      k.add("round");
      const kk = { add: (key) => k.add("mul") };
      let x = powRTL(a, d, n, kk);
      if (x === 1n || x === n - 1n) continue;
      let ok = false;
      for (let r = 1; r < s; r++) { x = (x * x) % n; k.add("mul"); if (x === n - 1n) { ok = true; break; } }
      if (!ok) return false;
    }
    return true;
  }
  const BASES64 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
  function isPrimeBig(n) { return millerRabin(n, BASES64); }
  /* Korselt: n composite, squarefree, and (p − 1) | (n − 1) for every prime p | n */
  function isCarmichael(n) {
    const f = factor(n);
    if (f.length < 2 || f.some(([, e]) => e > 1)) return false;
    return f.every(([p]) => (n - 1) % (p - 1) === 0);
  }
  /* census of one odd n over all bases 1..n−1 */
  function baseCensus(n) {
    const sd = decompose(n), cls = new Array(n);
    let fermat = 0, strong = 0, units = 0;
    for (let a = 1; a < n; a++) {
      if (gcd(a, n) !== 1) { cls[a] = 0; continue; }
      units++;
      const f = fermatLiar(a, n), s = f && strongLiar(a, n, sd);
      if (f) fermat++; if (s) strong++;
      cls[a] = s ? 3 : (f ? 2 : 1);           // 1 witness, 2 Fermat liar only, 3 strong liar (also Fermat)
    }
    return { cls, fermat, strong, units };
  }

  /* ── §18 random prime generation ─────────────────────────────────────── */
  function randomPrime(bits, r, k) {
    k = ctr(k);
    for (;;) {
      const c = randBig(bits, r) | 1n; k.add("candidate");
      if (millerRabin(c, BASES64)) return c;
    }
  }

  /* ── §19 RSA (small, BigInt) ─────────────────────────────────────────── */
  function lcm(a, b) { return a / gcdBig(a, b) * b; }
  function rsaKeys(p, q, e) {
    p = BigInt(p); q = BigInt(q); e = BigInt(e);
    const N = p * q, ph = (p - 1n) * (q - 1n), lam = lcm(p - 1n, q - 1n);
    const d = modInverseBig(e, lam), dPhi = modInverseBig(e, ph);
    if (d === null) return null;
    return { p, q, N, phi: ph, lam, e, d, dPhi, dp: d % (p - 1n), dq: d % (q - 1n), qInv: modInverseBig(q, p) };
  }
  function rsaDecryptCRT(c, K, k) {
    const m1 = powRTL(c % K.p, K.dp, K.p, k), m2 = powRTL(c % K.q, K.dq, K.q, k);
    const h = (K.qInv * (((m1 - m2) % K.p) + K.p)) % K.p;   // Garner recombination
    return m2 + h * K.q;
  }

  /* ── §22 Pollard's rho (Floyd cycle finding), BigInt ─────────────────── */
  function pollardRho(N, c, k, x0) {
    k = ctr(k);
    const f = x => (x * x + c) % N;
    let x = x0 === undefined ? 2n : x0, y = x, d = 1n;
    while (d === 1n) {
      x = f(x); y = f(f(y)); k.add("iter"); k.add("gcd");
      d = gcdBig(x > y ? x - y : y - x, N);
    }
    return d;                                 // d === N means failure: retry with another c
  }
  /* tail length μ and cycle length λ of x ↦ x² + c mod p, from x₀ */
  function rhoShape(p, c, x0) {
    const seen = new Map(), seq = [];
    let x = x0 % p;
    while (!seen.has(x)) { seen.set(x, seq.length); seq.push(x); x = (x * x + c) % p; }
    const mu = seen.get(x);
    return { seq, mu, lam: seq.length - mu };
  }

  return {
    rnd, randBig, bitLen, popcount, mulmod, powmod,
    euclid, gcd, gcdBig, binaryGcd, extEuclid, extEuclidBig, modInverse, modInverseBig, solveLinear,
    factor, phi, order, powerTable,
    powLTR, powRTL, powNaive,
    crt,
    trialDivision, fermatLiar, decompose, strongLiar, millerRabin, BASES64, isPrimeBig, isCarmichael, baseCensus,
    randomPrime, lcm, rsaKeys, rsaDecryptCRT,
    pollardRho, rhoShape
  };
})();
if (typeof module !== "undefined" && module.exports) module.exports = NT;

/* ═══════════════════════════════════════════════════════════════════════════
   NX — DOM helpers, and the figures (browser only)
   ═══════════════════════════════════════════════════════════════════════════ */
if (typeof document !== "undefined" && typeof d3 !== "undefined") {

const NX = {
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
  holds: ok => ok ? '<span style="color:' + AC.good + '">✓ holds</span>'
                  : '<span style="color:' + AC.bad + '">✗ VIOLATED</span>',
  clearControls: node => node.parentNode.querySelectorAll('div[role="group"]').forEach(el => el.remove()),
  txt: (g, x, y, s, o) => {
    o = o || {};
    return g.append("text").attr("x", x).attr("y", y).attr("font-size", o.size || 12)
      .attr("fill", o.fill || AC.ink).attr("text-anchor", o.anchor || "start")
      .attr("font-family", o.mono ? "SF Mono, Menlo, monospace" : null)
      .attr("font-weight", o.bold ? 600 : null).text(s);
  },
  /* a BigInt shown compactly: all digits when short, else head…tail and the digit count */
  big: (x, keep) => { const s = x.toString(); keep = keep || 12; return s.length <= 2 * keep + 3 ? s : s.slice(0, keep) + "…" + s.slice(-keep) + " (" + s.length + " digits)"; },
  val: sel => +d3.select(sel).property("value"),
  FIB: (function () { const F = [0, 1]; for (let i = 2; i < 80; i++) F.push(F[i - 1] + F[i - 2]); return F; })(),
  PHI: (1 + Math.sqrt(5)) / 2
};

/* ══ FIGURE 1 — cost against input size in bits ═══════════════════════════
   For every β, n_β = the largest prime below 2^β (found by Miller–Rabin and
   confirmed by the trial division that is being counted). Four measured
   counts: trial divisions to certify n_β prime; Euclid divisions on the worst
   β-bit pair (consecutive Fibonacci numbers); multiplications in 2^(n_β − 1)
   mod n_β by repeated squaring; multiplications in 12-base Miller–Rabin.   */
(function () {
  const svg = d3.select("#bits-svg"); if (svg.empty()) return;
  const W = 720, H = 360, F = { x: 60, y: 24, w: 470, h: 290 };
  const data = [];
  for (let b = 4; b <= 44; b += 2) {
    let n = (1n << BigInt(b)) - 1n; while (!NT.isPrimeBig(n)) n -= 2n;
    const kt = AL.counter(), ke = AL.counter(), km = AL.counter(), kr = AL.counter();
    const sf = NT.trialDivision(Number(n), kt);
    let i = 2; while (NX.FIB[i + 1] < Math.pow(2, b)) i++;
    NT.euclid(NX.FIB[i], NX.FIB[i - 1], ke);
    NT.powRTL(2n, n - 1n, n, km);
    NT.millerRabin(n, NT.BASES64, kr);
    data.push({ b, n, certified: sf === Number(n), trial: kt.get("div"), euclid: ke.get("div"), fibK: i,
                modexp: km.get("sq") + km.get("mul"), mr: kr.get("mul") });
  }
  const series = [
    { key: "trial", label: "trial divisions", color: AC.bad },
    { key: "mr", label: "Miller–Rabin, 12 bases", color: AC.violet },
    { key: "modexp", label: "2ⁿ⁻¹ mod n, squaring", color: AC.accent },
    { key: "euclid", label: "Euclid, worst β-bit pair", color: AC.good }
  ];
  function build() {
    const maxB = NX.val("#bits-max"), scale = d3.select("#bits-scale").property("value");
    d3.select("#bits-max-val").text(maxB);
    const D = data.filter(d => d.b <= maxB);
    svg.selectAll("*").remove();
    const x = d3.scaleLinear().domain([4, maxB]).range([F.x, F.x + F.w]);
    const top = d3.max(D, d => d.trial);
    const y = scale === "log" ? d3.scaleLog().domain([1, Math.max(10, top * 1.5)]).range([F.y + F.h, F.y]).clamp(true)
                              : d3.scaleLinear().domain([0, top * 1.05]).range([F.y + F.h, F.y]);
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${F.y + F.h})`).call(d3.axisBottom(x).ticks(8));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${F.x},0)`)
      .call(scale === "log" ? d3.axisLeft(y).ticks(6, "~s") : d3.axisLeft(y).ticks(6).tickFormat(d3.format("~s")));
    NX.txt(svg, F.x + F.w, F.y + F.h + 30, "β = bits of n (the INPUT SIZE)", { size: 11, anchor: "end", fill: AC.muted });
    NX.txt(svg, F.x, F.y - 10, "operations on β-bit numbers (measured, " + scale + " scale)", { size: 11, fill: AC.muted });
    series.forEach(s => {
      svg.append("path").attr("d", d3.line().x(d => x(d.b)).y(d => y(Math.max(1, d[s.key])))(D))
        .attr("fill", "none").attr("stroke", s.color).attr("stroke-width", 2);
      D.forEach(d => svg.append("circle").attr("cx", x(d.b)).attr("cy", y(Math.max(1, d[s.key]))).attr("r", 2.6).attr("fill", s.color));
    });
    AL.legend(svg, series.map(s => ({ label: s.label, color: s.color })), F.x + F.w + 14, F.y + 20, { gap: 20 });
    const L = D[D.length - 1], P = D[D.length - 2] || L;
    const side = svg.append("g").attr("transform", `translate(${F.x + F.w + 14},${F.y + 120})`);
    [["at β = " + L.b, ""], ["n = " + NX.big(L.n, 8), ""], ["trial divisions", NX.int(L.trial)], ["Miller–Rabin mults", NX.int(L.mr)],
     ["modexp mults", NX.int(L.modexp)], ["Euclid divisions", NX.int(L.euclid)]].forEach((r, i) => {
      NX.txt(side, 0, i * 20, r[0], { size: 11, fill: AC.muted }); NX.txt(side, 170, i * 20, r[1], { size: 12, anchor: "end", mono: true });
    });
    d3.select("#bits-readout").html("n = largest prime below 2^" + L.b + " = <b>" + L.n + "</b> "
      + "(" + (L.certified ? "trial division confirms it prime " + NX.flag(true) : NX.flag(false)) + ")"
      + " &nbsp;·&nbsp; trial division: <b>" + NX.int(L.trial) + "</b> divisions, ×" + (L.trial / P.trial).toFixed(2) + " for the last 2 extra bits (√4 = 2: exponential in β)"
      + " &nbsp;·&nbsp; Miller–Rabin: <b>" + NX.int(L.mr) + "</b> multiplications, modexp: <b>" + L.modexp + "</b> (≤ 2β = " + (2 * L.b) + "; n − 1 is almost all 1-bits)"
      + " &nbsp;·&nbsp; Euclid on (F" + L.fibK + ", F" + (L.fibK - 1) + "): <b>" + L.euclid + "</b> divisions (≈ 1.44·β = " + (1.44 * L.b).toFixed(0) + ")");
  }
  d3.select("#bits-max").on("input", build);
  d3.select("#bits-scale").on("change", build);
  build();
})();

/* ══ FIGURE 2 — Euclid: tiling, the Fibonacci worst case, extended Euclid ═══
   Left: the a × b rectangle cut into squares, one colour per division step —
   each step removes ⌊a/b⌋ squares of side b, and the last square has side
   gcd(a, b). Right: the MAXIMUM number of divisions over all pairs b < a ≤ n,
   measured by running Euclid on every pair up to 987, against Lamé's bound.  */
(function () {
  const svg = d3.select("#eu-svg"); if (svg.empty()) return;
  const NMAX = 987, maxSteps = new Array(NMAX + 1).fill(0), firstNeed = {};
  { const k = AL.counter();
    for (let a = 2; a <= NMAX; a++) {
      let m = maxSteps[a - 1];
      for (let b = 1; b < a; b++) { k.reset(); NT.euclid(a, b, k); const s = k.get("div"); if (s > m) m = s; if (!(s in firstNeed)) firstNeed[s] = a; }
      maxSteps[a] = m;
    }
  }
  let seed = 1;
  const colr = i => [AC.accent, AC.a2, AC.teal, AC.violet, AC.rose, AC.good][i % 6];
  function build() {
    const A = NX.val("#eu-a"), B = NX.val("#eu-b");
    d3.select("#eu-a-val").text(A); d3.select("#eu-b-val").text(B);
    svg.selectAll("*").remove();
    const k = AL.counter(), kb = AL.counter(), ke = AL.counter();
    const E = NT.euclid(A, B, k), g = E.g;
    const bg = NT.binaryGcd(A, B, kb);
    const rows = []; const X = NT.extEuclid(A, B, ke, rows);
    /* ── left: tiling ── */
    const s = Math.min(380 / A, 290 / Math.max(B, 1));
    const L = svg.append("g").attr("transform", "translate(20,40)");
    NX.txt(svg, 20, 24, A + " × " + B + " rectangle, cut into squares by Euclid", { size: 11, fill: AC.muted });
    L.append("rect").attr("width", A * s).attr("height", Math.max(B, 0.5) * s).attr("fill", AC.panel2).attr("stroke", AC.line);
    let x = 0, y = 0, horiz = true;
    E.rows.forEach((r, i) => {
      for (let j = 0; j < r.q; j++) {
        const sx = horiz ? x + j * r.b : x, sy = horiz ? y : y + j * r.b;
        L.append("rect").attr("x", sx * s).attr("y", sy * s).attr("width", r.b * s).attr("height", r.b * s)
          .attr("fill", colr(i)).attr("fill-opacity", 0.55).attr("stroke", AC.bg).attr("stroke-width", 0.8);
        if (r.b * s > 22 && j === 0) NX.txt(L, sx * s + r.b * s / 2, sy * s + r.b * s / 2 + 4, r.b, { size: Math.min(13, r.b * s / 3), anchor: "middle" });
      }
      if (horiz) x += r.q * r.b; else y += r.q * r.b;
      horiz = !horiz;
    });
    /* ── right: worst case over all pairs ── */
    const P = { x: 470, y: 40, w: 225, h: 250 };
    const xs = d3.scaleLinear().domain([1, NMAX]).range([P.x, P.x + P.w]);
    const ys = d3.scaleLinear().domain([0, 16]).range([P.y + P.h, P.y]);
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${P.y + P.h})`).call(d3.axisBottom(xs).ticks(4));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${P.x},0)`).call(d3.axisLeft(ys).ticks(8));
    NX.txt(svg, P.x - 30, 24, "max divisions over all b < a ≤ n (measured)", { size: 11, fill: AC.muted });
    NX.txt(svg, P.x + P.w, P.y + P.h + 30, "n", { size: 11, anchor: "end", fill: AC.muted });
    const pts = d3.range(2, NMAX + 1).map(n => [n, maxSteps[n]]);
    svg.append("path").attr("d", d3.line().curve(d3.curveStepAfter).x(d => xs(d[0])).y(d => ys(d[1]))(pts)).attr("fill", "none").attr("stroke", AC.good).attr("stroke-width", 2);
    const lb = d3.range(2, NMAX + 1, 4).map(n => [n, 1 + Math.log(n) / Math.log(NX.PHI)]);
    svg.append("path").attr("d", d3.line().x(d => xs(d[0])).y(d => ys(d[1]))(lb)).attr("fill", "none").attr("stroke", AC.muted).attr("stroke-dasharray", "4 3");
    NX.txt(svg, P.x + P.w - 4, ys(1 + Math.log(NMAX) / Math.log(NX.PHI)) - 6, "1 + log_φ n", { size: 10, anchor: "end", fill: AC.muted });
    Object.keys(firstNeed).forEach(st => { if (+st >= 3) svg.append("circle").attr("cx", xs(firstNeed[st])).attr("cy", ys(+st)).attr("r", 3).attr("fill", AC.a2); });
    NX.txt(svg, P.x + 6, P.y + 10, "● first n needing k steps", { size: 10, fill: AC.a2 });
    const steps = k.get("div");
    svg.append("circle").attr("cx", xs(Math.max(A, B, 1))).attr("cy", ys(steps)).attr("r", 5).attr("fill", "none").attr("stroke", AC.ink).attr("stroke-width", 1.6);
    NX.txt(svg, P.x + 6, P.y + 26, "○ your pair", { size: 10, fill: AC.ink });
    /* ── table + readout ── */
    NX.table("#eu-table", ["a", "b", "⌊a/b⌋", "d", "x", "y", "a·x + b·y"],
      rows.map(r => [r.a, r.b, r.q === null ? "—" : r.q, r.d, r.x, r.y, (r.a * r.x + r.b * r.y) + (r.a * r.x + r.b * r.y === r.d ? " ✓" : " ✗")]));
    const hi = Math.max(A, B), lo = Math.min(A, B);
    const k2 = AL.counter(); NT.euclid(hi, lo, k2); const kk = k2.get("div");
    const lame = lo === 0 || hi === lo || (lo >= NX.FIB[kk + 1] && hi >= NX.FIB[kk + 2]);
    const firsts = Object.keys(firstNeed).map(Number).filter(v => v >= 1);
    const fibOK = firsts.every(st => firstNeed[st] === NX.FIB[st + 2]);
    d3.select("#eu-readout").html("gcd(" + A + ", " + B + ") = <b>" + g + "</b> &nbsp;·&nbsp; Euclid: <b>" + steps + "</b> divisions"
      + (A < B ? " (the first only swaps, since a &lt; b)" : "")
      + " &nbsp;·&nbsp; binary gcd: " + kb.get("shift") + " shifts + " + kb.get("sub") + " subtractions, result " + bg + " " + NX.flag(bg === g)
      + " &nbsp;·&nbsp; extended Euclid: " + ke.get("call") + " calls, " + A + "·(" + X.x + ") + " + B + "·(" + X.y + ") = " + (A * X.x + B * X.y) + " " + NX.flag(A * X.x + B * X.y === g)
      + " &nbsp;·&nbsp; Lamé (" + kk + " divisions on (" + hi + ", " + lo + ") ⟹ smaller ≥ F" + (kk + 1) + " = " + NX.FIB[kk + 1] + "): " + NX.holds(lame)
      + " &nbsp;·&nbsp; over all " + NX.int(NMAX * (NMAX - 1) / 2) + " pairs up to " + NMAX + ", the first a needing k divisions is F(k+2) for every k = 1…" + d3.max(firsts) + ": " + NX.holds(fibOK));
  }
  d3.select("#eu-a").on("input", build);
  d3.select("#eu-b").on("input", build);
  d3.select("#eu-fib").on("click", () => {
    const A = Math.max(2, NX.val("#eu-a")); let i = 2; while (NX.FIB[i + 1] <= A) i++;
    d3.select("#eu-a").property("value", NX.FIB[i]); d3.select("#eu-b").property("value", NX.FIB[i - 1]); build();
  });
  d3.select("#eu-rand").on("click", () => {
    const r = AL.rng(++seed * 977);
    d3.select("#eu-a").property("value", AL.randInt(r, 50, NMAX)); d3.select("#eu-b").property("value", AL.randInt(r, 10, NMAX)); build();
  });
  build();
})();

/* ══ FIGURE 3 — the power table aᵏ mod n ═══════════════════════════════════
   Row a, column k holds aᵏ mod n. A unit's row returns to 1 with period
   ord(a), which divides φ(n) (Lagrange); a non-unit never reaches 1. The φ(n)
   column is all 1s on unit rows (Euler); the n − 1 column is all 1s when n is
   prime (Fermat). A primitive root's row visits every unit before repeating. */
(function () {
  const svg = d3.select("#pw-svg"); if (svg.empty()) return;
  function build() {
    const n = NX.val("#pw-n"), mode = d3.select("#pw-colour").property("value");
    d3.select("#pw-n-val").text(n);
    svg.selectAll("*").remove();
    const T = NT.powerTable(n), ph = NT.phi(n), m = n - 1;
    const cs = Math.max(6, Math.min(18, Math.floor(340 / Math.max(m, 1))));
    const G = svg.append("g").attr("transform", "translate(46,40)");
    const vcol = d3.scaleSequential(d3.interpolateViridis).domain([0, Math.max(1, n - 1)]);
    const units = [], prim = [], orders = {};
    let euler = true, fermat = 0;
    for (let a = 1; a < n; a++) {
      const ord = NT.order(a, n), row = T[a - 1];
      if (ord) { units.push(a); orders[ord] = (orders[ord] || 0) + 1; if (ord === ph) prim.push(a); if (row[ph - 1] !== 1) euler = false; if (ord && ph % ord !== 0) euler = false; }
      if (row[n - 2] === 1) fermat++;
      row.forEach((v, j) => {
        let f;
        if (mode === "value") f = vcol(v);
        else f = v === 1 ? AC.good : (v === 0 ? AC.bg : (ord ? AC.panel2 : "#2a2230"));
        G.append("rect").attr("x", j * cs).attr("y", (a - 1) * cs).attr("width", cs - 0.6).attr("height", cs - 0.6).attr("fill", f);
        if (cs >= 16) NX.txt(G, j * cs + cs / 2, (a - 1) * cs + cs / 2 + 3.5, v, { size: 8.5, anchor: "middle", fill: mode === "value" ? AC.bg : (v === 1 ? AC.bg : AC.muted) });
      });
      if (cs >= 9 || a % 5 === 0 || a === 1) NX.txt(G, -6, (a - 1) * cs + cs / 2 + 3.5, a, { size: Math.min(10, cs), anchor: "end", fill: ord ? AC.ink : AC.muted });
      if (ord === ph && ph > 0) G.append("rect").attr("x", -0.5).attr("y", (a - 1) * cs - 0.5).attr("width", m * cs).attr("height", cs).attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 1.4);
    }
    G.append("rect").attr("x", (ph - 1) * cs - 1).attr("y", -1).attr("width", cs + 1).attr("height", m * cs + 1).attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 1.6);
    NX.txt(G, (ph - 1) * cs + cs / 2, -6, "k = φ(n)", { size: 10, anchor: ph < m * 0.8 ? "middle" : "end", fill: AC.accent });
    if (ph !== m) G.append("rect").attr("x", (m - 1) * cs - 1).attr("y", -1).attr("width", cs + 1).attr("height", m * cs + 1).attr("fill", "none").attr("stroke", AC.muted).attr("stroke-dasharray", "3 2");
    NX.txt(svg, 46, 18, "row a, column k = aᵏ mod " + n + "   (k = 1 … " + m + ")", { size: 11, fill: AC.muted });
    const side = svg.append("g").attr("transform", "translate(430,50)");
    const isPrime = ph === n - 1;
    const lines = [
      ["n", n + (isPrime ? "  (prime)" : "  = " + NT.factor(n).map(([p, e]) => p + (e > 1 ? "^" + e : "")).join("·"))],
      ["φ(n) = |Zₙ*|", ph],
      ["units", units.length <= 10 ? units.join(", ") : units.slice(0, 9).join(", ") + ", …"],
      ["orders seen", Object.keys(orders).map(o => o + (orders[o] > 1 ? "×" + orders[o] : "")).join("  ")],
      ["primitive roots", prim.length ? (prim.length <= 8 ? prim.join(", ") : prim.slice(0, 7).join(", ") + ", …") + "  (" + prim.length + ")" : "none — Zₙ* is not cyclic"],
      ["φ(φ(n))", NT.phi(Math.max(1, ph))],
      ["rows with aⁿ⁻¹ ≡ 1", fermat + " of " + m]
    ];
    lines.forEach((r, i) => { NX.txt(side, 0, i * 38, r[0], { size: 11, fill: AC.muted }); NX.txt(side, 0, i * 38 + 16, String(r[1]), { size: 12.5, mono: true }); });
    AL.legend(svg, [{ label: "value 1", color: AC.good }, { label: "primitive-root row", color: AC.a2 }, { label: "column φ(n)", color: AC.accent }], 430, 330, { gap: 16 });
    const primOK = prim.length === 0 || prim.length === NT.phi(ph);
    d3.select("#pw-readout").html("n = " + n + ", φ(n) = <b>" + ph + "</b> units &nbsp;·&nbsp; every unit's order divides φ(n) and a^φ(n) ≡ 1 on every unit row (Lagrange, Euler): " + NX.holds(euler)
      + " &nbsp;·&nbsp; primitive roots: <b>" + prim.length + "</b>" + (prim.length ? " = φ(φ(n)) = " + NT.phi(ph) + " " + NX.holds(primOK) : " (Zₙ* is cyclic only for n = 2, 4, pᵏ, 2pᵏ)")
      + " &nbsp;·&nbsp; bases with aⁿ⁻¹ ≡ 1: <b>" + fermat + "</b> of " + m + (isPrime ? " — all of them, as Fermat's little theorem says" : " — the composite's Fermat liars (§16)"));
  }
  d3.select("#pw-n").on("input", build);
  d3.select("#pw-colour").on("change", build);
  build();
})();

/* ══ FIGURE 4 — modular exponentiation by repeated squaring, stepped ═══════
   a, n are random 64-bit numbers (n odd), b a random exponent with exactly
   the chosen number of bits. Every frame is emitted by the real routine; the
   counts in the readout are the routine's counter.                          */
(function () {
  const svg = d3.select("#me-svg"); if (svg.empty()) return;
  let seed = 3;
  /* measured total multiplications for one random exponent of every length */
  function curve(dir) {
    const r = NT.rnd(101), out = [];
    for (let bits = 1; bits <= 64; bits++) {
      const b = NT.randBig(bits, r), k = AL.counter();
      (dir === "ltr" ? NT.powLTR : NT.powRTL)(7n, b, 1000000007n, k);
      out.push({ bits, total: k.get("sq") + k.get("mul") });
    }
    return out;
  }
  const CURVE = { ltr: curve("ltr"), rtl: curve("rtl") };
  function build() {
    NX.clearControls(svg.node());
    const bits = NX.val("#me-bits"), dir = d3.select("#me-dir").property("value");
    d3.select("#me-bits-val").text(bits);
    const r = NT.rnd(seed * 7919);
    const a = NT.randBig(64, r), n = NT.randBig(64, r) | 1n, b = NT.randBig(bits, r);
    const k = AL.counter(), k2 = AL.counter(), frames = [];
    const res = (dir === "ltr" ? NT.powLTR : NT.powRTL)(a, b, n, k, frames);
    const other = (dir === "ltr" ? NT.powRTL : NT.powLTR)(a, b, n, k2);
    const naive = bits <= 14 ? NT.powNaive(a, b, n) : null;
    const bstr = b.toString(2), L = bstr.length;
    const cw = Math.min(26, Math.floor(640 / L) - 1);
    const C = { x: 60, y: 210, w: 600, h: 90 };
    const xs = d3.scaleLinear().domain([1, 64]).range([C.x, C.x + C.w]), ys = d3.scaleLinear().domain([0, 128]).range([C.y + C.h, C.y]);
    function draw(f, fi) {
      svg.selectAll("*").remove();
      NX.txt(svg, 20, 20, "b = " + NX.big(b, 10) + " in binary (" + L + " bits, " + NT.popcount(b) + " ones) — " + (dir === "ltr" ? "scanned left to right" : "scanned right to left"), { size: 11, fill: AC.muted });
      AL.row(svg, bstr.split(""), { x: 20, y: 30, w: cw, h: 22, gap: 1, index: false, fontSize: Math.min(12, cw - 1),
        mark: (i) => i === f.i ? (f.op === "multiply" ? AC.a2 : AC.accent) : ((dir === "ltr" ? i < f.i : i > f.i) ? "#1f2a3d" : null) });
      const desc = f.op === "start" ? "start: r = a mod n (the leading 1 bit)"
        : f.op === "square" ? (dir === "ltr" ? "bit " + (L - 1 - f.i) + ": square  r ← r² mod n" : "square the running power  s ← s² mod n  (s = a^(2^" + (L - 1 - f.i) + "))")
        : (dir === "ltr" ? "bit is 1: multiply  r ← r·a mod n" : "bit " + (L - 1 - f.i) + " is 1: multiply  r ← r·s mod n");
      NX.txt(svg, 20, 84, "step " + (fi + 1) + " of " + frames.length + ":  " + desc, { size: 12.5, fill: f.op === "multiply" ? AC.a2 : AC.accent });
      NX.txt(svg, 20, 110, "r = " + NX.big(f.r, 14), { size: 12, mono: true });
      if (f.s !== undefined) NX.txt(svg, 20, 130, "s = " + NX.big(f.s, 14), { size: 12, mono: true, fill: AC.muted });
      NX.txt(svg, 20, 156, "a = " + NX.big(a, 10) + "     n = " + NX.big(n, 10), { size: 11, mono: true, fill: AC.muted });
      NX.txt(svg, 20, 176, "every intermediate value stays below n: at most " + n.toString(2).length + " bits, never " + "b·log₂a", { size: 11, fill: AC.muted });
      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${C.y + C.h})`).call(d3.axisBottom(xs).ticks(8));
      svg.append("g").attr("class", "axis").attr("transform", `translate(${C.x},0)`).call(d3.axisLeft(ys).ticks(4));
      NX.txt(svg, C.x + C.w, C.y + C.h + 28, "bits of the exponent b", { size: 10.5, anchor: "end", fill: AC.muted });
      NX.txt(svg, C.x + 4, C.y - 4, "measured multiplications (squarings + multiplies), one random b per length", { size: 10.5, fill: AC.muted });
      [[1, "⌊log₂ b⌋ + 1"], [2, "2⌊log₂ b⌋ + 1"]].forEach(([c, lab]) => {
        svg.append("line").attr("x1", xs(1)).attr("y1", ys(c * 0 + 1)).attr("x2", xs(64)).attr("y2", ys(c * 63 + 1)).attr("stroke", AC.muted).attr("stroke-dasharray", "4 3");
        NX.txt(svg, xs(64) + 4, ys(c * 63 + 1) + 4, lab, { size: 10, fill: AC.muted });
      });
      svg.append("path").attr("d", d3.line().x(d => xs(d.bits)).y(d => ys(d.total))(CURVE[dir])).attr("fill", "none").attr("stroke", AC.teal).attr("stroke-width", 1.8);
      svg.append("circle").attr("cx", xs(L)).attr("cy", ys(k.get("sq") + k.get("mul"))).attr("r", 5).attr("fill", AC.a2);
    }
    AL.stepper(svg, { frames: frames, render: draw, delay: 450, label: "step" });
    const sq = k.get("sq"), mu = k.get("mul"), pc = NT.popcount(b);
    const expSq = L - 1, expMu = dir === "ltr" ? pc - 1 : pc;
    const bigBits = Number(b) * Math.log2(Number(a));
    d3.select("#me-readout").html("measured: <b>" + sq + "</b> squarings + <b>" + mu + "</b> multiplies = <b>" + (sq + mu) + "</b> modular multiplications"
      + " &nbsp;·&nbsp; predicted ⌊log₂ b⌋ = " + expSq + " squarings and " + (dir === "ltr" ? "popcount(b) − 1" : "popcount(b)") + " = " + expMu + " multiplies " + NX.holds(sq === expSq && mu === expMu)
      + " &nbsp;·&nbsp; left-to-right and right-to-left give the same aᵇ mod n " + NX.flag(res === other)
      + (naive !== null ? " &nbsp;·&nbsp; naive b − 1 = " + NX.int(Number(b) - 1) + " multiplications gives the same " + NX.flag(naive === res) : " &nbsp;·&nbsp; naive repeated multiplication would need b − 1 ≈ 2^" + (L - 1) + " multiplications")
      + " &nbsp;·&nbsp; computing aᵇ before reducing would produce a number of ≈ 2^" + Math.log2(bigBits).toFixed(1) + " bits");
  }
  d3.select("#me-bits").on("input", build);
  d3.select("#me-dir").on("change", build);
  d3.select("#me-seed").on("click", () => { seed++; build(); });
  build();
})();

/* ══ FIGURE 5 — the Chinese-remainder grid ═════════════════════════════════
   Each x in 0 … m·n − 1 is written in the cell (x mod m, x mod n). When
   gcd(m, n) = 1 every cell receives exactly one number: the map is a
   bijection Z_mn → Z_m × Z_n. When gcd = d > 1 only 1/d of the cells are hit,
   each by d numbers. The target cell's number is computed by the constructive
   formula, with inverses from extended Euclid, and checked by search.       */
(function () {
  const svg = d3.select("#crt-svg"); if (svg.empty()) return;
  function build() {
    const m = NX.val("#crt-m"), n = NX.val("#crt-n");
    const r1 = NX.val("#crt-r1") % m, r2 = NX.val("#crt-r2") % n, walk = d3.select("#crt-walk").property("checked");
    d3.select("#crt-m-val").text(m); d3.select("#crt-n-val").text(n);
    d3.select("#crt-r1-val").text(r1); d3.select("#crt-r2-val").text(r2);
    svg.selectAll("*").remove();
    const d = NT.gcd(m, n), cs = Math.floor(Math.min(34, 400 / n, 300 / m));
    const G = svg.append("g").attr("transform", "translate(56,46)");
    const cell = {};
    for (let x = 0; x < m * n; x++) { const key = (x % m) + "," + (x % n); (cell[key] = cell[key] || []).push(x); }
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
      const xsv = cell[i + "," + j] || [], tgt = i === r1 && j === r2;
      G.append("rect").attr("x", j * cs).attr("y", i * cs).attr("width", cs - 1).attr("height", cs - 1).attr("rx", 3)
        .attr("fill", tgt ? AC.a2 : (xsv.length ? AC.panel2 : AC.bg)).attr("stroke", xsv.length > 1 ? AC.bad : AC.line);
      if (xsv.length) NX.txt(G, j * cs + cs / 2, i * cs + cs / 2 + 4, xsv.length > 1 ? xsv[0] + "+" : xsv[0], { size: Math.min(12, cs / 2.4), anchor: "middle", fill: tgt ? AC.bg : AC.ink });
    }
    for (let j = 0; j < n; j++) NX.txt(G, j * cs + cs / 2, -6, j, { size: 10, anchor: "middle", fill: AC.muted });
    for (let i = 0; i < m; i++) NX.txt(G, -6, i * cs + cs / 2 + 4, i, { size: 10, anchor: "end", fill: AC.muted });
    NX.txt(svg, 56, 18, "column = x mod " + n + ",  row = x mod " + m + ",  cell shows x ∈ [0, " + (m * n - 1) + "]", { size: 11, fill: AC.muted });
    if (walk) {
      const pts = d3.range(0, Math.min(m * n, 4 * m * n)).map(x => [(x % n) * cs + cs / 2, (x % m) * cs + cs / 2]);
      for (let t = 0; t + 1 < pts.length; t++) {
        const [a, b] = [pts[t], pts[t + 1]];
        if (Math.abs(a[0] - b[0]) <= cs + 1 && Math.abs(a[1] - b[1]) <= cs + 1)
          G.append("line").attr("x1", a[0]).attr("y1", a[1]).attr("x2", b[0]).attr("y2", b[1]).attr("stroke", AC.teal).attr("stroke-opacity", 0.55).attr("stroke-width", 1.4);
      }
    }
    const side = svg.append("g").attr("transform", `translate(${Math.max(56 + n * cs + 30, 470)},56)`);
    const k = AL.counter();
    const sol = d === 1 ? NT.crt([r1, r2], [m, n], k) : null;
    const brute = []; for (let x = 0; x < m * n; x++) if (x % m === r1 && x % n === r2) brute.push(x);
    let lines;
    if (sol) {
      const [t1, t2] = sol.terms;
      lines = [["solve", "x ≡ " + r1 + " (mod " + m + "),  x ≡ " + r2 + " (mod " + n + ")"],
        ["M₁ = " + n + ",  M₁⁻¹ mod " + m, t1.inv], ["c₁ = M₁·M₁⁻¹", t1.c + "   (≡1 mod " + m + ", ≡0 mod " + n + ")"],
        ["M₂ = " + m + ",  M₂⁻¹ mod " + n, t2.inv], ["c₂ = M₂·M₂⁻¹", t2.c + "   (≡0 mod " + m + ", ≡1 mod " + n + ")"],
        ["x = r₁c₁ + r₂c₂ mod " + (m * n), r1 + "·" + t1.c + " + " + r2 + "·" + t2.c + " ≡ " + sol.x]];
    } else {
      lines = [["gcd(" + m + ", " + n + ") = " + d, "not coprime: the map is not onto"],
        ["cells hit", (m * n / d) + " of " + (m * n) + ", each by " + d + " numbers"],
        ["solvable iff", "r₁ ≡ r₂ (mod " + d + ")"],
        ["this target", brute.length ? "solutions " + brute.join(", ") : "no solution"]];
    }
    lines.forEach((r, i) => { NX.txt(side, 0, i * 38, r[0], { size: 11, fill: AC.muted }); NX.txt(side, 0, i * 38 + 16, String(r[1]), { size: 12, mono: true }); });
    const filled = Object.keys(cell).length;
    d3.select("#crt-readout").html("m = " + m + ", n = " + n + ", gcd = <b>" + d + "</b> &nbsp;·&nbsp; cells hit: <b>" + filled + "</b> of " + (m * n)
      + (d === 1 ? " — every cell exactly once, so x ↦ (x mod m, x mod n) is a bijection " + NX.holds(filled === m * n)
                 : " — only m·n/d = " + (m * n / d) + " " + NX.holds(filled === m * n / d))
      + (sol ? " &nbsp;·&nbsp; CRT construction gives x = <b>" + sol.x + "</b> using " + k.get("call") + " extended-Euclid calls; exhaustive search finds " + brute.join(", ") + " " + NX.flag(brute.length === 1 && brute[0] === sol.x)
             : " &nbsp;·&nbsp; target (" + r1 + ", " + r2 + "): " + (brute.length ? brute.length + " solutions below m·n (one per lcm = " + (m * n / d) + ")" : "no solution, since " + r1 + " ≢ " + r2 + " (mod " + d + ")")));
  }
  ["#crt-m", "#crt-n", "#crt-r1", "#crt-r2"].forEach(s => d3.select(s).on("input", build));
  d3.select("#crt-walk").on("change", build);
  build();
})();

/* ══ FIGURE 6 — Miller–Rabin: every base of one n, and liar fractions ══════
   Left: each base a ∈ [1, n − 1] classified by running the real tests.
   Right: for EVERY odd composite n up to NMAX, the fraction of bases that are
   strong liars (and, faint, Fermat liars), measured by the same census.    */
(function () {
  const svg = d3.select("#mr-svg"); if (svg.empty()) return;
  const NMAX = 2999, census = [];
  for (let n = 9; n <= NMAX; n += 2) {
    if (NT.trialDivision(n) === n) continue;
    const c = NT.baseCensus(n);
    census.push({ n, s: c.strong / (n - 1), f: c.fermat / (n - 1), strong: c.strong, units: c.units, carm: NT.isCarmichael(n) });
  }
  const worst = census.filter(c => c.n > 9).reduce((p, c) => (c.strong / c.units > p.strong / p.units ? c : p));
  const COL = { 0: "#3a3f4b", 1: AC.accent, 2: AC.a2, 3: AC.bad, 4: AC.good };
  function build(fromPreset) {
    if (fromPreset) d3.select("#mr-n").property("value", d3.select("#mr-preset").property("value"));
    let n = NX.val("#mr-n"); if (n % 2 === 0) n += 1;
    d3.select("#mr-n-val").text(n);
    svg.selectAll("*").remove();
    const isPrime = NT.trialDivision(n) === n, c = NT.baseCensus(n);
    const cls = c.cls.map(v => (isPrime && v === 3 ? 4 : v));
    const cols = n < 400 ? 20 : 60, cs = Math.max(3, Math.min(14, Math.floor(Math.min(390 / cols, 300 / Math.ceil((n - 1) / cols)) * 10) / 10));
    const G = svg.append("g").attr("transform", "translate(20,40)");
    NX.txt(svg, 20, 20, "bases a = 1 … " + (n - 1) + " of n = " + n + ", " + cols + " per row", { size: 11, fill: AC.muted });
    let rects = 0;
    for (let a = 1; a < n;) {                 // run-length encode each row: one <rect> per colour run
      const row = Math.floor((a - 1) / cols), col = (a - 1) % cols, v = cls[a];
      let len = 1; while (a + len < n && cls[a + len] === v && (a + len - 1) % cols !== 0) len++;
      G.append("rect").attr("x", col * cs).attr("y", row * cs).attr("width", len * cs - 0.5).attr("height", cs - 0.5).attr("fill", COL[v]);
      rects++; a += len;
    }
    AL.legend(svg, [{ label: "shares a factor with n", color: COL[0] }, { label: "witness (Fermat catches it)", color: COL[1] },
      { label: "Fermat liar, strong witness", color: COL[2] }, { label: "strong liar", color: COL[3] }, { label: "n prime: passes", color: COL[4] }], 20, 360, { gap: 13 });
    /* right: census */
    const P = { x: 470, y: 40, w: 230, h: 280 };
    const xs = d3.scaleLinear().domain([0, NMAX]).range([P.x, P.x + P.w]), ys = d3.scaleLinear().domain([0, 1]).range([P.y + P.h, P.y]);
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${P.y + P.h})`).call(d3.axisBottom(xs).ticks(4));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${P.x},0)`).call(d3.axisLeft(ys).ticks(5).tickFormat(d3.format(".0%")));
    NX.txt(svg, P.x - 30, 20, "liar fraction, every odd composite ≤ " + NMAX, { size: 11, fill: AC.muted });
    const dotPath = (arr, fy) => arr.map(d => "M" + xs(d.n).toFixed(1) + "," + ys(fy(d)).toFixed(1) + "h0.01").join("");
    svg.append("path").attr("d", dotPath(census, d => d.f)).attr("stroke", AC.a2).attr("stroke-opacity", 0.35).attr("stroke-width", 2.4).attr("stroke-linecap", "round");
    svg.append("path").attr("d", dotPath(census, d => d.s)).attr("stroke", AC.bad).attr("stroke-width", 2.6).attr("stroke-linecap", "round");
    census.filter(d => d.carm).forEach(d => svg.append("circle").attr("cx", xs(d.n)).attr("cy", ys(d.f)).attr("r", 5).attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 1.4));
    svg.append("line").attr("x1", P.x).attr("x2", P.x + P.w).attr("y1", ys(0.25)).attr("y2", ys(0.25)).attr("stroke", AC.ink).attr("stroke-dasharray", "4 3");
    NX.txt(svg, P.x + P.w, ys(0.25) - 5, "¼", { size: 12, anchor: "end" });
    NX.txt(svg, P.x + 6, P.y + 12, "● strong liars   ● Fermat liars   ○ Carmichael", { size: 10, fill: AC.muted });
    svg.append("line").attr("x1", xs(n)).attr("x2", xs(n)).attr("y1", P.y).attr("y2", P.y + P.h).attr("stroke", AC.teal).attr("stroke-opacity", 0.7);
    const maxS = d3.max(census, d => d.s);
    const rabin = census.every(d => d.n <= 9 || d.strong <= d.units / 4);
    const p1 = isPrime ? 0 : Math.max(0, c.strong - 2) / Math.max(1, n - 3);     // over bases 2 … n − 2
    const carm = !isPrime && NT.isCarmichael(n);
    const fstr = isPrime ? "prime" : NT.factor(n).map(([p, e]) => p + (e > 1 ? "^" + e : "")).join("·");
    d3.select("#mr-readout").html("n = <b>" + n + "</b> = " + fstr + (carm ? " — a <b>Carmichael number</b>" : "")
      + " &nbsp;·&nbsp; φ(n) = " + c.units + " &nbsp;·&nbsp; Fermat liars <b>" + (isPrime ? "—" : c.fermat) + "</b>, strong liars <b>" + (isPrime ? "—" : c.strong) + "</b>"
      + (isPrime ? " — n is prime, so every base passes both tests (no liars to count)"
                 : " (" + (100 * c.strong / (n - 1)).toFixed(2) + "% of all bases; ≤ φ(n)/4 = " + (c.units / 4).toFixed(1) + " " + NX.holds(n <= 9 || c.strong <= c.units / 4) + ")"
                   + " &nbsp;·&nbsp; one round with a uniform base in [2, n − 2] errs with probability " + p1.toFixed(4) + "; ten independent rounds: " + Math.pow(p1, 10).toExponential(2))
      + " &nbsp;·&nbsp; census of " + NX.int(census.length) + " odd composites ≤ " + NMAX + ": largest strong-liar share of [1, n − 1] = " + (100 * maxS).toFixed(1) + "%, worst ratio to φ(n) for n &gt; 9 is " + worst.strong + "/" + worst.units + " at n = " + worst.n + ", strong ≤ φ(n)/4 for every n &gt; 9: " + NX.holds(rabin)
      + " &nbsp;·&nbsp; drawn with " + rects + " run-length rectangles");
  }
  d3.select("#mr-n").on("input", () => build(false));
  d3.select("#mr-preset").on("change", () => build(true));
  build(false);
})();

/* ══ FIGURE 7 — toy RSA: the permutation m ↦ mᵉ mod N ═════════════════════
   All N messages are encrypted and decrypted with the real routines; the plot
   is the permutation, and the fixed points (mᵉ = m) are highlighted.       */
(function () {
  const svg = d3.select("#rsa-svg"); if (svg.empty()) return;
  function build() {
    let p = NX.val("#rsa-p"), q = NX.val("#rsa-q"), e = NX.val("#rsa-e");
    svg.selectAll("*").remove();
    if (p === q) {
      NX.txt(svg, 20, 40, "p = q: N = p² is not an RSA modulus — pick two different primes.", { size: 13, fill: AC.bad });
      d3.select("#rsa-readout").html("p and q must differ: with N = p², φ(N) = p(p − 1) and p is found as √N in one step.");
      return;
    }
    const K = NT.rsaKeys(p, q, e);
    if (!K) {
      NX.txt(svg, 20, 40, "e = " + e + " shares a factor with λ(N) = lcm(" + (p - 1) + ", " + (q - 1) + ") — no inverse d exists.", { size: 13, fill: AC.bad });
      d3.select("#rsa-readout").html("gcd(e, λ(N)) = " + NT.gcd(e, Number(NT.lcm(BigInt(p - 1), BigInt(q - 1)))) + " ≠ 1, so mᵉ is not a permutation of Z_N and cannot be undone. Key generation rejects this e (or these primes).");
      return;
    }
    const N = Number(K.N), m = NX.val("#rsa-m") % N;
    const kPlain = AL.counter(), kCRT = AL.counter();
    let allOK = true, fixed = 0, coll = 0;
    const seen = new Uint8Array(N), pts = [];
    for (let x = 0; x < N; x++) {
      const c = NT.powRTL(BigInt(x), K.e, K.N);
      const back = NT.powRTL(c, K.d, K.N);
      if (back !== BigInt(x)) allOK = false;
      const ci = Number(c); if (seen[ci]) coll++; seen[ci] = 1;
      if (ci === x) fixed++;
      pts.push([x, ci]);
    }
    const c = NT.powRTL(BigInt(m), K.e, K.N);
    const dPlain = NT.powRTL(c, K.d, K.N, kPlain), dCRT = NT.rsaDecryptCRT(c, K, kCRT);
    /* weight each multiplication by (operand bits)², the schoolbook cost of one product */
    const kP = AL.counter(), kQ = AL.counter();
    NT.powRTL(c % K.p, K.dp, K.p, kP); NT.powRTL(c % K.q, K.dq, K.q, kQ);
    const tot = kk => kk.get("sq") + kk.get("mul"), bl = x => NT.bitLen(x);
    const wPlain = tot(kPlain) * bl(K.N) * bl(K.N), wCRT = tot(kP) * bl(K.p) * bl(K.p) + tot(kQ) * bl(K.q) * bl(K.q);
    const P = { x: 50, y: 30, w: 300, h: 300 };
    const xs = d3.scaleLinear().domain([0, N]).range([P.x, P.x + P.w]), ys = d3.scaleLinear().domain([0, N]).range([P.y + P.h, P.y]);
    svg.append("rect").attr("x", P.x).attr("y", P.y).attr("width", P.w).attr("height", P.h).attr("fill", AC.panel2);
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${P.y + P.h})`).call(d3.axisBottom(xs).ticks(4));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${P.x},0)`).call(d3.axisLeft(ys).ticks(4));
    NX.txt(svg, P.x + P.w, P.y + P.h + 30, "message m", { size: 10.5, anchor: "end", fill: AC.muted });
    NX.txt(svg, P.x, P.y - 8, "ciphertext c = mᵉ mod N", { size: 10.5, fill: AC.muted });
    const path = arr => arr.map(d => "M" + xs(d[0]).toFixed(1) + "," + ys(d[1]).toFixed(1) + "h0.01").join("");
    svg.append("path").attr("d", path(pts)).attr("stroke", AC.accent).attr("stroke-width", N > 1500 ? 1.6 : 2.4).attr("stroke-linecap", "round").attr("stroke-opacity", 0.8);
    svg.append("path").attr("d", path(pts.filter(d => d[0] === d[1]))).attr("stroke", AC.bad).attr("stroke-width", 5).attr("stroke-linecap", "round");
    svg.append("circle").attr("cx", xs(m)).attr("cy", ys(Number(c))).attr("r", 6).attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 2);
    const side = svg.append("g").attr("transform", "translate(390,40)");
    [["public key (N, e)", "(" + N + ", " + e + ")"], ["φ(N) = (p−1)(q−1)", String(K.phi)], ["λ(N) = lcm(p−1, q−1)", String(K.lam)],
     ["private d = e⁻¹ mod λ(N)", String(K.d) + "   (mod φ(N): " + K.dPhi + ")"], ["CRT: d_p, d_q, q⁻¹ mod p", K.dp + ", " + K.dq + ", " + K.qInv],
     ["m → c → m", m + " → " + c + " → " + dPlain], ["fixed points mᵉ = m", fixed + "   (red)"]].forEach((r, i) => {
      NX.txt(side, 0, i * 40, r[0], { size: 11, fill: AC.muted }); NX.txt(side, 0, i * 40 + 17, r[1], { size: 12.5, mono: true });
    });
    const predFixed = (1 + NT.gcd(e - 1, p - 1)) * (1 + NT.gcd(e - 1, q - 1));
    d3.select("#rsa-readout").html("N = " + p + "·" + q + " = <b>" + N + "</b>, e = " + e + ", d = <b>" + K.d + "</b> (e·d mod λ(N) = " + (K.e * K.d % K.lam) + ")"
      + " &nbsp;·&nbsp; decrypt(encrypt(m)) = m for all " + NX.int(N) + " messages, including the " + (N - Number(K.phi)) + " not coprime to N: " + NX.holds(allOK && coll === 0)
      + " &nbsp;·&nbsp; decrypting c = " + c + ": plain <b>" + tot(kPlain) + "</b> multiplications of " + bl(K.N) + "-bit numbers, CRT <b>" + tot(kCRT) + "</b> of " + bl(K.p) + "- and " + bl(K.q) + "-bit numbers (d_p, d_q have " + bl(K.dp) + " and " + bl(K.dq) + " bits against d's " + bl(K.d) + ")"
      + "; weighted by bits², CRT costs <b>" + (wCRT / wPlain).toFixed(2) + "</b>× plain; answers " + NX.flag(dPlain === dCRT && kCRT.get("sq") + kCRT.get("mul") === tot(kP) + tot(kQ))
      + " &nbsp;·&nbsp; fixed points: " + fixed + ", predicted (1 + gcd(e−1, p−1))(1 + gcd(e−1, q−1)) = " + predFixed + " " + NX.flag(fixed === predFixed));
  }
  ["#rsa-p", "#rsa-q", "#rsa-e"].forEach(s => d3.select(s).on("change", build));
  d3.select("#rsa-m").on("input", build);
  build();
})();

/* ══ FIGURE 8 — Pollard's rho ══════════════════════════════════════════════
   Left: the sequence xᵢ₊₁ = xᵢ² + c reduced mod the hidden prime factor p —
   a tail of μ values, then a cycle of λ. Floyd's method stops at the first
   i ≥ 1 with x_i ≡ x_2i (mod p), which is the first multiple of λ that is ≥ μ.
   Right: measured iterations against √p for 120 random semiprimes.         */
(function () {
  const svg = d3.select("#rho-svg"); if (svg.empty()) return;
  const floydStop = (mu, lam) => { let i = lam; while (i < mu) i += lam; return i; };
  const r = NT.rnd(2024), cloud = [];
  const primeIn = (lo, hi) => { for (;;) { let v = lo + Math.floor(r() * (hi - lo)); v |= 1; if (NT.trialDivision(v) === v) return v; } };
  for (let t = 0; t < 120; t++) {
    const p = primeIn(30, 3e5), q = primeIn(3e5, 1e6), k = AL.counter();
    const d = NT.pollardRho(BigInt(p) * BigInt(q), 1n, k);
    if (d !== BigInt(p) * BigInt(q)) cloud.push({ p: Math.min(p, q), it: k.get("iter") });
  }
  function build() {
    const N = BigInt(d3.select("#rho-N").property("value")), c = NX.val("#rho-c");
    d3.select("#rho-c-val").text(c);
    svg.selectAll("*").remove();
    const k = AL.counter();
    const d = NT.pollardRho(N, BigInt(c), k);
    const p1 = NT.trialDivision(Number(N)), p2 = Number(N) / p1;
    const sh1 = NT.rhoShape(p1, c, 2), sh2 = NT.rhoShape(p2, c, 2);
    const st1 = floydStop(sh1.mu, sh1.lam), st2 = floydStop(sh2.mu, sh2.lam);
    const pred = Math.min(st1, st2), p = st1 <= st2 ? p1 : p2, sh = st1 <= st2 ? sh1 : sh2;
    /* rho layout: tail on a vertical stem, cycle on a circle above it */
    const cx = 200, cy = 150, R = Math.min(115, 20 + sh.lam * 2.2), stem = Math.min(150, 12 + sh.mu * 6);
    const pos = i => {
      if (i < sh.mu) { const t = sh.mu === 0 ? 0 : (sh.mu - i) / sh.mu; return [cx, cy + R + stem * t]; }
      const j = i - sh.mu, a = Math.PI / 2 + 2 * Math.PI * j / sh.lam;
      return [cx + R * Math.cos(a), cy + R * Math.sin(a)];
    };
    const all = d3.range(sh.seq.length + 1).map(i => pos(i < sh.seq.length ? i : sh.mu));
    svg.append("path").attr("d", d3.line()(all)).attr("fill", "none").attr("stroke", AC.muted).attr("stroke-width", 1);
    const rad = sh.seq.length > 300 ? 1.6 : 3.2;
    sh.seq.forEach((v, i) => { const [x, y] = pos(i); svg.append("circle").attr("cx", x).attr("cy", y).attr("r", i === 0 || i === sh.mu ? rad + 2 : rad).attr("fill", i < sh.mu ? AC.a2 : AC.accent); });
    const [sx, sy] = pos(pred < sh.mu ? pred : sh.mu + ((pred - sh.mu) % sh.lam));
    svg.append("circle").attr("cx", sx).attr("cy", sy).attr("r", 8).attr("fill", "none").attr("stroke", AC.good).attr("stroke-width", 2);
    NX.txt(svg, 20, 20, "xᵢ mod p for p = " + p + ":  tail μ = " + sh.mu + " (orange), cycle λ = " + sh.lam + " (blue)", { size: 11, fill: AC.muted });
    NX.txt(svg, 20, 360, "○ green: where Floyd's tortoise meets the hare (i = " + pred + ")", { size: 10.5, fill: AC.good });
    /* right: iterations vs √p */
    const P = { x: 470, y: 40, w: 225, h: 270 };
    const xs = d3.scaleLog().domain([4, 1100]).range([P.x, P.x + P.w]), ys = d3.scaleLog().domain([1, 5000]).range([P.y + P.h, P.y]);
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${P.y + P.h})`).call(d3.axisBottom(xs).ticks(4, "~s"));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${P.x},0)`).call(d3.axisLeft(ys).ticks(4, "~s"));
    NX.txt(svg, P.x - 30, 20, "iterations until a factor, 120 random semiprimes", { size: 11, fill: AC.muted });
    NX.txt(svg, P.x + P.w, P.y + P.h + 30, "√p (p = smaller factor)", { size: 10.5, anchor: "end", fill: AC.muted });
    svg.append("line").attr("x1", xs(4)).attr("y1", ys(4)).attr("x2", xs(1100)).attr("y2", ys(1100)).attr("stroke", AC.muted).attr("stroke-dasharray", "4 3");
    NX.txt(svg, xs(900), ys(900) + 16, "= √p", { size: 10, fill: AC.muted, anchor: "end" });
    cloud.forEach(e => svg.append("circle").attr("cx", xs(Math.sqrt(e.p))).attr("cy", ys(Math.max(1, e.it))).attr("r", 2.6).attr("fill", AC.teal).attr("fill-opacity", 0.75));
    const ratios = cloud.map(e => e.it / Math.sqrt(e.p)).sort((a, b) => a - b);
    const med = ratios[Math.floor(ratios.length / 2)];
    const ok = d !== 1n && d !== N && N % d === 0n;
    d3.select("#rho-readout").html("N = " + N + " = " + p1 + " × " + p2 + ", c = " + c
      + " &nbsp;·&nbsp; " + (ok ? "found factor <b>" + d + "</b> after <b>" + k.get("iter") + "</b> iterations (" + k.get("gcd") + " gcds); " + d + " × " + (N / d) + " = N " + NX.flag(d * (N / d) === N)
                             : "<b>failed</b>: gcd reached N after " + k.get("iter") + " iterations — both factors cycled together; retry with another c")
      + " &nbsp;·&nbsp; mod p = " + p + ": μ = " + sh.mu + ", λ = " + sh.lam + ", μ + λ = " + (sh.mu + sh.lam) + " against √(πp/2) = " + Math.sqrt(Math.PI * p / 2).toFixed(0)
      + " &nbsp;·&nbsp; Floyd predicts a stop at i = " + pred + " (first multiple of λ ≥ μ, taken over both primes) " + NX.flag(!ok || pred === k.get("iter"))
      + " &nbsp;·&nbsp; trial division would need up to " + NX.int(Math.floor(Math.min(p1, p2) / 2)) + " divisions &nbsp;·&nbsp; cloud: median iterations / √p = " + med.toFixed(2));
  }
  d3.select("#rho-N").on("change", build);
  d3.select("#rho-c").on("input", build);
  build();
})();

}
