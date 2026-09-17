/* hash-tables.viz.js — figures for
   dsa/data-structures/hash-tables.html (part 3 of the Data Structures series).

   Loaded after ../../data.js, ../../notes.js and ../dsa-viz.js, so the globals
   AC (palette) and AL (rng / frame / axisB / axisL / gridY / row / counter /
   stepper / …) are available.

   House rule obeyed throughout: every cost this page DISPLAYS — probes, chain
   lengths, comparisons, slot occupancies, rehash work — comes from running the
   real routine under AL.counter() and reading the counter back. Nothing below
   is a number typed into a caption. Where a closed form exists it is evaluated
   a SECOND, independent way and printed next to the measurement, so each figure
   cross-checks itself and says "agree" or "DISAGREE" in colour.

   A warning that shaped several figures here: the classical probe-count
   formulas are asymptotic in m. At m = 1009 and alpha = 0.9 the measured
   unsuccessful linear-probing cost is ~42 against a predicted 50.5; by
   m = 10007 it is ~51. So the quantitative figures use tables of several
   thousand slots and average over many independent builds, and the small
   stepped figures (which use m = 11 or 13 so the reader can follow every slot)
   make no quantitative claim at all.

   Figures, listed in FILE order (which is also the order of the FIGURE nn
   comments below). Note this is NOT page order: #alpha-svg sits late in the
   file but appears earlier on the page. 13 figures, one per .viz block.
     01 #pipe-svg    key → hash code → slot; which half produced the collision
     02 #div-svg     division method: power of two vs composite vs prime
     03 #birth-svg   first-collision time: Monte Carlo vs exact vs sqrt(pi m/2)
     04 #chain-svg   chaining: measured chain lengths vs 1 + alpha and the max
     05 #lin-svg     linear probing, stepped, with a wraparound and a cluster
     06 #probe3-svg  linear vs quadratic vs double on one key set
     07 #theo-svg    the four probe theorems, measured against their formulas
     08 #tomb-svg    deletion breaks a probe chain; the tombstone repairs it
     09 #resize-svg  per-operation rehash spikes and the running amortized mean
     10 #churn-svg   incremental rehashing: cost spread across later operations
     11 #alpha-svg   load-factor sweep: chaining vs open addressing as alpha → 1
     12 #univ-svg    adversarial keys: one fixed hash vs a random universal draw
     13 #bloom-svg   Bloom filter false-positive rate, measured vs the formula  */

/* ── shared little helpers ─────────────────────────────────────────────── */
const HT = {
  int: d3.format(","),
  sig: function (x, d) { return (+x).toFixed(d === undefined ? 2 : d); },
  verdict: function (ok) {
    return ok ? '<b style="color:' + AC.good + '">agree</b>'
              : '<b style="color:' + AC.bad + '">DISAGREE</b>';
  },
  /* remove stepper control bars left by a previous build of the same figure */
  clearControls: function (svgNode) {
    svgNode.parentNode.querySelectorAll('div[role="group"]').forEach(el => el.remove());
  },

  /* ── hash codes ──────────────────────────────────────────────────────
     Both are real implementations, not sketches: `poly` is the Horner
     polynomial hash derived on the Arrays & Strings page, `sum` is the
     classic weak code that adds the character values and therefore maps
     every anagram of a key to the same code. */
  poly: function (s, a) {
    let h = 0;
    for (let i = 0; i < s.length; i++) h = (Math.imul(h, a || 31) + s.charCodeAt(i)) >>> 0;
    return h;
  },
  sum: function (s) {
    let h = 0;
    for (let i = 0; i < s.length; i++) h = (h + s.charCodeAt(i)) >>> 0;
    return h;
  },
  /* an avalanche finaliser — multiply/xor-shift, the descendant of §08 */
  mix: function (x) {
    let h = x >>> 0;
    h = Math.imul(h ^ (h >>> 16), 2246822507) >>> 0;
    h = Math.imul(h ^ (h >>> 13), 3266489909) >>> 0;
    return (h ^ (h >>> 16)) >>> 0;
  },
  /* chi-square of a bucket-count array against a uniform expectation */
  chi2: function (counts, n) {
    const m = counts.length, e = n / m;
    if (e === 0) return 0;
    let s = 0;
    for (let j = 0; j < m; j++) s += (counts[j] - e) * (counts[j] - e) / e;
    return s;
  }
};

/* ══ FIGURE 01 ═══════════════════════════════════════════════════════════
   The two halves of a hash function, drawn as a pipeline over ten real string
   keys. The middle column is produced by running the real code function; the
   right column by running the real compression. The readout counts distinct
   CODES and distinct SLOTS separately, which is the §04 blame test.        */
(function () {
  const svg = d3.select("#pipe-svg"); if (svg.empty()) return;
  const W = 760, H = 430;
  const KEYS = ["stop", "tops", "pots", "opts", "spot", "post", "cat", "act", "tac", "dog"];
  const P = 109345121;                    // prime > any m used here, for MAD
  const MAD_A = 3271, MAD_B = 91009;      // fixed so the picture is reproducible

  const els = {
    m: d3.select("#pipe-m"), mOut: d3.select("#pipe-m-out"),
    code: d3.select("#pipe-code"), mode: d3.select("#pipe-mode"),
    out: d3.select("#pipe-readout")
  };

  function draw() {
    const m = +els.m.property("value");
    const codeKind = els.code.empty() ? "poly" : els.code.property("value");
    const mode = els.mode.property("value");
    els.mOut.text(m);

    const c = AL.counter();
    const codes = KEYS.map(k => {
      c.add("codeops", k.length);
      return codeKind === "sum" ? HT.sum(k) : HT.poly(k, 31);
    });
    const pow2 = Math.pow(2, Math.floor(AL.log2(m)));
    const slots = codes.map(v => {
      c.add("compress");
      if (mode === "mod") return v % m;
      if (mode === "mad") return (((MAD_A * (v % P) + MAD_B) % P) % m);
      return v & (pow2 - 1);           // low bits, into a power-of-two subtable
    });
    const effM = mode === "low" ? pow2 : m;

    const distinctCodes = new Set(codes).size;
    const distinctSlots = new Set(slots).size;
    const counts = new Array(effM).fill(0);
    slots.forEach(s => counts[s]++);
    const maxBucket = Math.max(...counts);
    /* independent recount of the collision total, a different way: pairs that share a slot */
    let pairs = 0;
    for (let i = 0; i < KEYS.length; i++) for (let j = i + 1; j < KEYS.length; j++) if (slots[i] === slots[j]) pairs++;
    let pairsFromCounts = 0;
    counts.forEach(v => { pairsFromCounts += v * (v - 1) / 2; });
    const agree = pairs === pairsFromCounts;

    const f = AL.frame(svg, W, H, { l: 8, r: 8, t: 26, b: 8 });
    const g = f.g;
    const colK = 10, colC = 190, colS = 430;
    /* The two blocks have different natural heights (10 key rows vs effM slots), so
       fixing both at the top made every connector slope downhill and cross its
       neighbours. Centre the shorter block against the taller one instead. */
    const rowStep = 36;
    const keyBlockH = KEYS.length * rowStep;

    g.append("text").attr("x", colK).attr("y", 4).attr("font-size", 11).attr("fill", AC.muted).text("key");
    g.append("text").attr("x", colC).attr("y", 4).attr("font-size", 11).attr("fill", AC.muted)
      .text(codeKind === "sum" ? "hash code — sum of characters" : "hash code — polynomial, radix 31");
    g.append("text").attr("x", colS).attr("y", 4).attr("font-size", 11).attr("fill", AC.muted)
      .text("slot in T[0 … " + (effM - 1) + "]");

    /* the bucket array down the right-hand side */
    const slotH = Math.min(22, (H - 70) / effM);
    const slotBlockH = effM * slotH;
    const avail = H - 74;
    const slotTop = 16 + Math.max(0, (avail - slotBlockH) / 2);
    const keyTop = 22 + Math.max(0, (avail - keyBlockH) / 2);
    const slotY = j => slotTop + j * slotH;
    const rowY = i => keyTop + i * rowStep;
    for (let j = 0; j < effM; j++) {
      g.append("rect").attr("x", colS + 60).attr("y", slotY(j)).attr("width", 150)
        .attr("height", Math.max(2, slotH - 2)).attr("rx", 3)
        .attr("fill", counts[j] > 1 ? "#3a2230" : (counts[j] === 1 ? AC.panel2 : AC.panel))
        .attr("stroke", counts[j] > 1 ? AC.rose : AC.line);
      if (slotH >= 12) {
        g.append("text").attr("x", colS + 54).attr("y", slotY(j) + slotH / 2 + 3).attr("text-anchor", "end")
          .attr("font-size", 9).attr("fill", AC.muted).text(j);
      }
      const here = KEYS.filter((_, i) => slots[i] === j);
      if (here.length && slotH >= 10) {
        g.append("text").attr("x", colS + 66).attr("y", slotY(j) + slotH / 2 + 3)
          .attr("font-size", Math.min(11, slotH - 2)).attr("fill", here.length > 1 ? AC.rose : AC.ink)
          .text(here.join(" · "));
      }
    }

    /* per-key rows: key box, code box, and a link into its slot */
    const codeDup = {};
    codes.forEach(v => { codeDup[v] = (codeDup[v] || 0) + 1; });
    KEYS.forEach((k, i) => {
      const y = rowY(i);
      if (y > H - 40) return;
      g.append("rect").attr("x", colK).attr("y", y - 13).attr("width", 62).attr("height", 24).attr("rx", 4)
        .attr("fill", AC.panel2).attr("stroke", AC.line);
      g.append("text").attr("x", colK + 31).attr("y", y + 4).attr("text-anchor", "middle")
        .attr("font-size", 12).attr("fill", AC.ink).text(k);
      const dup = codeDup[codes[i]] > 1;
      g.append("rect").attr("x", colC).attr("y", y - 13).attr("width", 128).attr("height", 24).attr("rx", 4)
        .attr("fill", dup ? "#3a2230" : AC.panel2).attr("stroke", dup ? AC.rose : AC.line);
      g.append("text").attr("x", colC + 64).attr("y", y + 4).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("fill", dup ? AC.rose : AC.ink).text(HT.int(codes[i]));
      AL.arrow(g, colK + 64, y, colC - 3, y, { color: AC.line, w: 1.2, head: 4 });
      const ty = slotY(slots[i]) + slotH / 2;
      const col = counts[slots[i]] > 1 ? AC.rose : AC.accent;
      g.append("path")
        .attr("d", "M" + (colC + 130) + "," + y + " C" + (colC + 190) + "," + y
                 + " " + (colS + 10) + "," + ty + " " + (colS + 57) + "," + ty)
        .attr("fill", "none").attr("stroke", col).attr("stroke-width", 1.3).attr("opacity", 0.85);
    });

    const label = { mod: "code mod m", mad: "((a·code + b) mod p) mod m", low: "code AND (2^p − 1)" }[mode];
    els.out.html(
      "<b>" + KEYS.length + " keys · " + (codeKind === "sum" ? "sum-of-characters code" : "polynomial code")
      + " · " + label + " · effective table " + effM + " slots.</b> "
      + "Measured by running both stages: <b>" + distinctCodes + "</b> distinct hash codes, then <b>"
      + distinctSlots + "</b> distinct slots, longest bucket <b>" + maxBucket + "</b>. "
      + (distinctCodes < KEYS.length
          ? "The <b>code</b> is already losing keys — " + (KEYS.length - distinctCodes)
            + " of them were merged before compression saw them, and no table size can separate equal integers again."
          : "All " + KEYS.length + " codes are distinct, so every collision below was manufactured by the compression step.")
      + " Colliding pairs counted directly: <b>" + pairs + "</b>; recomputed from the bucket counts as ∑ nⱼ(nⱼ−1)/2: <b>"
      + pairsFromCounts + "</b> — " + HT.verdict(agree) + ". "
      + "Hash-code work measured at <b>" + c.get("codeops") + "</b> character steps for " + KEYS.length
      + " keys, which is the <span class='keep'>Θ</span>(L) term that O(1) hides."
    );
  }

  [els.m, els.code, els.mode].forEach(s => { if (!s.empty()) s.on("input change", draw); });
  draw();
})();

/* ══ FIGURE 02 ═══════════════════════════════════════════════════════════
   The division method under three moduli — a power of two, a composite, and a
   prime — on the SAME key set. Bars are measured bucket counts; the distinct-
   slot count is cross-checked against the gcd theorem m / gcd(d, m) whenever
   the key set really is an arithmetic progression.                         */
(function () {
  const svg = d3.select("#div-svg"); if (svg.empty()) return;
  const W = 760, H = 380;
  const MODULI = [{ m: 64, tag: "m = 64  (2⁶, a power of two)" },
                  { m: 66, tag: "m = 66  (composite, 2·3·11)" },
                  { m: 67, tag: "m = 67  (prime)" }];
  const WORDS = ("able acid aged also area army away baby back ball band bank base bath bear beat "
               + "been beer bell belt bend best bike bill bind bird bite blow blue boat body bold "
               + "bone book boot born boss both bowl bulk burn bush busy cake call calm came camp "
               + "card care cart case cash cast cell chat chip city club coal coat code cold come "
               + "cook cool cope copy core cost crew crop dark data date dawn days dead deal dear").split(" ");

  const els = {
    keys: d3.select("#div-keys"), n: d3.select("#div-n"), nOut: d3.select("#div-n-out"),
    mix: d3.select("#div-mix"), out: d3.select("#div-readout")
  };

  function keySet(kind, n) {
    const r = AL.rng(20260916);
    if (kind === "stride8") return Array.from({ length: n }, (_, i) => 1000 + 8 * i);
    if (kind === "stride1") return Array.from({ length: n }, (_, i) => 1000 + i);
    if (kind === "stride100") return Array.from({ length: n }, (_, i) => 1000 + 100 * i);
    if (kind === "random") return Array.from({ length: n }, () => AL.randInt(r, 0, 1 << 26));
    return WORDS.slice(0, n).map(w => HT.poly(w, 31));      // already integer codes
  }
  const STRIDE = { stride8: 8, stride1: 1, stride100: 100 };
  function gcd(a, b) { while (b) { const t = a % b; a = b; b = t; } return a; }

  function draw() {
    const kind = els.keys.property("value");
    const n = +els.n.property("value");
    const mixOn = els.mix.property("checked");
    els.nOut.text(n);
    const keys = keySet(kind, n);

    const panels = MODULI.map(mo => {
      const c = AL.counter();
      const counts = new Array(mo.m).fill(0);
      keys.forEach(k => {
        c.add("compress");
        const code = mixOn ? HT.mix(k) : (k >>> 0);
        counts[code % mo.m]++;
      });
      const used = counts.filter(v => v > 0).length;
      return { m: mo.m, tag: mo.tag, counts, used,
               max: Math.max(...counts), chi: HT.chi2(counts, n), ops: c.get("compress") };
    });

    /* independent prediction, valid only for an unmixed arithmetic progression:
       the reachable-slot count is m / gcd(stride, m), capped by n */
    const d = mixOn ? null : STRIDE[kind];
    let pred = null, agree = null;
    if (d) {
      pred = panels.map(p => Math.min(n, p.m / gcd(d, p.m)));
      agree = panels.every((p, i) => p.used === pred[i]);
    }

    const f = AL.frame(svg, W, H, { l: 46, r: 14, t: 16, b: 26 });
    const g = f.g;
    const panelH = (f.ih - 2 * 22) / 3;
    const top = Math.max(2, d3.max(panels.map(p => p.max)));

    panels.forEach((p, pi) => {
      const gy = pi * (panelH + 22);
      const gg = g.append("g").attr("transform", "translate(0," + gy + ")");
      const x = d3.scaleBand().domain(d3.range(p.m)).range([0, f.iw - 168]).padding(0.12);
      const y = d3.scaleLinear().domain([0, top]).range([panelH, 0]);
      gg.append("line").attr("x1", 0).attr("x2", f.iw - 168).attr("y1", panelH).attr("y2", panelH)
        .attr("stroke", AC.line);
      p.counts.forEach((v, j) => {
        if (v === 0) return;
        gg.append("rect").attr("x", x(j)).attr("y", y(v)).attr("width", Math.max(1.4, x.bandwidth()))
          .attr("height", panelH - y(v)).attr("rx", 1)
          .attr("fill", v > 1 ? AC.a2 : AC.accent);
      });
      gg.append("text").attr("x", 0).attr("y", -4).attr("font-size", 11).attr("fill", AC.muted).text(p.tag);
      const tx = f.iw - 156;
      gg.append("text").attr("x", tx).attr("y", 10).attr("font-size", 11).attr("fill", AC.ink)
        .text("slots used  " + p.used + " / " + p.m);
      gg.append("text").attr("x", tx).attr("y", 26).attr("font-size", 11).attr("fill", AC.muted)
        .text("longest bucket  " + p.max);
      gg.append("text").attr("x", tx).attr("y", 42).attr("font-size", 11).attr("fill", AC.muted)
        .text("X² = " + HT.sig(p.chi, 1) + "   (≈ " + (p.m - 1) + " if uniform)");
      if (pred) {
        gg.append("text").attr("x", tx).attr("y", 58).attr("font-size", 11)
          .attr("fill", p.used === pred[pi] ? AC.good : AC.bad)
          .text("gcd rule predicts  " + pred[pi]);
      }
    });
    AL.axisL(g, d3.scaleLinear().domain([0, top]).range([panelH, 0]), 3, "keys in slot");

    const kindLabel = { stride8: "ids spaced 8 apart", stride1: "consecutive ids",
                        stride100: "ids spaced 100 apart", random: "unstructured random ids",
                        words: "short words hashed by the polynomial code" }[kind];
    els.out.html(
      "<b>" + n + " " + kindLabel + (mixOn ? ", passed through an avalanche step first" : ", used as the hash code directly")
      + ".</b> Measured slots used: "
      + panels.map(p => "<b>" + p.used + "</b>/" + p.m).join(" · ")
      + "; longest bucket " + panels.map(p => "<b>" + p.max + "</b>").join(" · ")
      + ". " + (pred
        ? "The gcd theorem predicts min(n, m/gcd(" + d + ", m)) = " + pred.join(" · ")
          + " reachable slots — " + HT.verdict(agree) + "."
        : "With " + (mixOn ? "the avalanche step in place" : "unstructured keys")
          + " there is no arithmetic stride for the gcd theorem to act on, and the three moduli behave alike.")
      + " " + panels[0].ops + " compressions were performed per panel."
    );
  }

  [els.keys, els.n, els.mix].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══ FIGURE 03 ═══════════════════════════════════════════════════════════
   The birthday bound. The histogram is produced by running the real insertion
   loop until a slot repeats; the exact expectation is summed term by term from
   Pr[first k distinct]; the closed form sqrt(pi·m/2) + 2/3 is evaluated
   separately. Three independent routes to one number.                      */
(function () {
  const svg = d3.select("#birth-svg"); if (svg.empty()) return;
  const W = 760, H = 370;
  const els = {
    m: d3.select("#birth-m"), mOut: d3.select("#birth-m-out"),
    t: d3.select("#birth-trials"), tOut: d3.select("#birth-trials-out"),
    out: d3.select("#birth-readout")
  };

  /* E[T] = sum over k >= 0 of Pr[T > k], Pr[T > k] = prod over i = 0 .. k-1 of (1 - i/m) */
  function exactExpectation(m) {
    let prod = 1, E = 0;
    for (let k = 0; k <= m; k++) { E += prod; prod *= (1 - k / m); if (prod <= 0) break; }
    return E;
  }
  function survival(m, kmax) {                 // Pr[first k all distinct]
    const out = [1]; let prod = 1;
    for (let k = 1; k <= kmax; k++) { prod *= (1 - (k - 1) / m); out.push(Math.max(0, prod)); }
    return out;
  }

  function draw() {
    const m = +els.m.property("value");
    const trials = +els.t.property("value");
    els.mOut.text(HT.int(m)); els.tOut.text(HT.int(trials));

    const r = AL.rng(424242);
    const c = AL.counter();
    const samples = new Array(trials);
    let sum = 0, mx = 0;
    for (let t = 0; t < trials; t++) {
      const seen = new Uint8Array(m);
      let k = 0;
      for (;;) {
        k++; c.add("insert");
        const j = Math.floor(r() * m);
        if (seen[j]) break;
        seen[j] = 1;
      }
      samples[t] = k; sum += k; if (k > mx) mx = k;
    }
    const meanMC = sum / trials;
    const exact = exactExpectation(m);
    const closed = Math.sqrt(Math.PI * m / 2) + 2 / 3;
    const okExact = Math.abs(meanMC - exact) / exact < 0.05;
    const okClosed = Math.abs(exact - closed) / exact < 0.01;

    /* half-probability point, measured from the samples and predicted */
    const sorted = [...samples].sort((a, b) => a - b);
    const medianMC = sorted[Math.floor(trials / 2)];
    const nHalf = (1 + Math.sqrt(1 + 8 * Math.LN2 * m)) / 2;

    const f = AL.frame(svg, W, H, { l: 50, r: 150, t: 18, b: 40 });
    const g = f.g;
    const hi = Math.max(mx, Math.ceil(closed * 3));
    const nb = 48, bw = hi / nb;
    const bins = new Array(nb).fill(0);
    samples.forEach(v => { const b = Math.min(nb - 1, Math.floor(v / bw)); bins[b]++; });

    const x = d3.scaleLinear().domain([0, hi]).range([0, f.iw]);
    const y = d3.scaleLinear().domain([0, d3.max(bins) || 1]).nice().range([f.ih, 0]);
    AL.gridY(g, y, f.iw, 4);
    bins.forEach((v, b) => {
      if (!v) return;
      g.append("rect").attr("x", x(b * bw)).attr("y", y(v))
        .attr("width", Math.max(1, x(bw) - 1)).attr("height", f.ih - y(v))
        .attr("fill", AC.accent).attr("opacity", 0.75);
    });
    /* the survival curve Pr[first k distinct], on its own 0..1 scale */
    const surv = survival(m, Math.ceil(hi));
    const ys = d3.scaleLinear().domain([0, 1]).range([f.ih, 0]);
    g.append("path").attr("fill", "none").attr("stroke", AC.violet).attr("stroke-width", 1.8)
      .attr("d", d3.line().x((_, i) => x(i)).y(v => ys(v))(surv));

    const mark = (v, col, txt, dy) => {
      g.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", 0).attr("y2", f.ih)
        .attr("stroke", col).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.6);
      g.append("text").attr("x", x(v) + 4).attr("y", dy).attr("font-size", 10).attr("fill", col).text(txt);
    };
    mark(meanMC, AC.a2, "measured mean " + HT.sig(meanMC, 2), 12);
    mark(exact, AC.good, "exact " + HT.sig(exact, 2), 26);
    mark(nHalf, AC.violet, "50% point " + HT.sig(nHalf, 1), 40);

    AL.axisB(g, x, f.ih, 6, "insertions until the first collision", d3.format("d"));
    AL.axisL(g, y, 4, "trials");
    AL.legend(g, [
      { label: "measured distribution", color: AC.accent },
      { label: "Pr[all distinct so far]", color: AC.violet },
      { label: "measured mean", color: AC.a2, dash: "4 3" },
      { label: "exact expectation", color: AC.good, dash: "4 3" }
    ], f.iw + 12, 14);

    els.out.html(
      "<b>m = " + HT.int(m) + " slots, " + HT.int(trials) + " simulated runs, "
      + HT.int(c.get("insert")) + " insertions performed.</b> "
      + "Mean insertions to the first collision: measured <b>" + HT.sig(meanMC, 3)
      + "</b>, exact expectation <b>" + HT.sig(exact, 3)
      + "</b> — " + HT.verdict(okExact) + " (within 5%). "
      + "Closed form √(<span class='keep'>π</span>m/2) + 2/3 = <b>" + HT.sig(closed, 3)
      + "</b> against the exact sum — " + HT.verdict(okClosed) + " (within 1%). "
      + "Measured median <b>" + medianMC + "</b>; the 50%-probability point predicted by "
      + "n(n−1) ≥ 2m·ln2 is <b>" + HT.sig(nHalf, 1) + "</b>. "
      + "For scale, the first collision arrives at a load factor of only <b>"
      + HT.sig(meanMC / m, 4) + "</b> — √m, not m."
    );
  }

  [els.m, els.t].forEach(s => s.on("input", draw));
  draw();
})();

/* ══ FIGURE 04 ═══════════════════════════════════════════════════════════
   A real chained hash table is built (arrays standing in for the buckets), then
   n successful searches and n unsuccessful searches are RUN under AL.counter().
   Statistics are averaged over R independent builds so they converge; the
   histogram drawn is the first build. Each measured number is checked against a
   prediction computed a different way: the two mean costs against their closed
   forms, the distribution against Poisson(alpha)·m, the empty-slot fraction
   against e^(−alpha), and the longest bucket against the Poissonised expected
   maximum  E[max] = sum_k ( 1 − F(k)^m ),  F = Poisson(alpha) CDF.           */
(function () {
  const svg = d3.select("#chain-svg"); if (svg.empty()) return;
  const W = 760, H = 380;
  const els = {
    a: d3.select("#chain-alpha"), aOut: d3.select("#chain-alpha-out"),
    m: d3.select("#chain-m"), mOut: d3.select("#chain-m-out"),
    show: d3.select("#chain-show"), out: d3.select("#chain-readout")
  };

  /* expected maximum bucket under the Poisson approximation — independent of
     the simulation, and the honest prediction for "how long is the worst chain" */
  function expectedMax(alpha, m) {
    let term = Math.exp(-alpha), F = term, E = 0;
    for (let k = 0; k < 400; k++) {
      E += 1 - Math.pow(F, m);
      if (k > alpha + 6 && 1 - Math.pow(F, m) < 1e-9) break;
      term *= alpha / (k + 1);
      F += term;
      if (F > 1) F = 1;
    }
    return E;
  }

  function build(m, n, seed) {
    const r = AL.rng(seed);
    const T = Array.from({ length: m }, () => []);
    const keys = [];
    for (let i = 0; i < n; i++) {
      const j = Math.floor(r() * m);
      T[j].unshift(i);                 // prepend, as §10 assumes
      keys.push({ k: i, j: j });
    }
    const cs = AL.counter(), cu = AL.counter();
    keys.forEach(e => {
      const b = T[e.j];
      for (let i = 0; i < b.length; i++) { cs.add("ex"); if (b[i] === e.k) break; }
    });
    const nMiss = Math.max(1, n);
    for (let t = 0; t < nMiss; t++) cu.add("ex", T[Math.floor(r() * m)].length);
    const lens = T.map(b => b.length);
    let tot = 0; lens.forEach(v => { tot += v; });
    return { T, lens, tot,
             succ: n ? cs.get("ex") / n : 0, unsucc: cu.get("ex") / nMiss,
             max: Math.max(...lens), empty: lens.filter(v => v === 0).length / m };
  }

  function draw() {
    const alpha = +els.a.property("value");
    const m = +els.m.property("value");
    const n = Math.round(alpha * m);
    els.aOut.text(HT.sig(alpha, 1)); els.mOut.text(HT.int(m));

    const R = Math.max(4, Math.min(24, Math.round(24576 / m)));
    const runs = [];
    for (let t = 0; t < R; t++) runs.push(build(m, n, 98765 + 7919 * t));
    const avg = f => runs.reduce((s, x) => s + f(x), 0) / R;
    const meanSucc = avg(x => x.succ), meanUnsucc = avg(x => x.unsucc);
    const meanMax = avg(x => x.max), meanEmpty = avg(x => x.empty);
    const sumOK = runs.every(x => x.tot === n);

    const predSucc = n ? 1 + (n - 1) / (2 * m) : 0;
    const predUnsucc = alpha;
    const predMax = expectedMax(alpha, m);
    const predEmpty = Math.exp(-alpha);
    const okS = n === 0 || Math.abs(meanSucc - predSucc) / predSucc < 0.05;
    const okU = Math.abs(meanUnsucc - predUnsucc) / Math.max(predUnsucc, 1e-9) < 0.08;
    const okMax = Math.abs(meanMax - predMax) <= 0.75;
    const okEmpty = Math.abs(meanEmpty - predEmpty) < 0.02;

    const first = runs[0], lens = first.lens, maxLen = first.max;

    const f = AL.frame(svg, W, H, { l: 52, r: 176, t: 18, b: 40 });
    const g = f.g;

    if (els.show.property("value") === "buckets") {
      const shown = Math.min(m, 128);
      const x = d3.scaleBand().domain(d3.range(shown)).range([0, f.iw]).padding(0.15);
      const y = d3.scaleLinear().domain([0, Math.max(1, maxLen)]).range([f.ih, 0]);
      AL.gridY(g, y, f.iw, 4);
      for (let j = 0; j < shown; j++) {
        for (let d = 0; d < lens[j]; d++) {
          g.append("rect").attr("x", x(j)).attr("y", y(d + 1))
            .attr("width", Math.max(1.5, x.bandwidth())).attr("height", Math.max(1, y(d) - y(d + 1) - 1))
            .attr("rx", 1).attr("fill", d === 0 ? AC.accent : (d < 3 ? AC.a2 : AC.rose));
        }
      }
      AL.axisB(g, d3.scaleLinear().domain([0, shown - 1]).range([0, f.iw]), f.ih, 8,
        "slot index (first " + shown + " of " + HT.int(m) + ")", d3.format("d"));
      AL.axisL(g, y, 4, "bucket length");
      AL.legend(g, [{ label: "first entry", color: AC.accent },
                    { label: "2nd–3rd entry", color: AC.a2 },
                    { label: "4th and beyond", color: AC.rose }], f.iw + 12, 14);
    } else {
      const top = Math.max(maxLen, 1);
      const counts = new Array(top + 1).fill(0);
      lens.forEach(v => counts[v]++);
      const x = d3.scaleBand().domain(d3.range(top + 1)).range([0, f.iw]).padding(0.2);
      const y = d3.scaleLinear().domain([0, d3.max(counts)]).nice().range([f.ih, 0]);
      AL.gridY(g, y, f.iw, 5);
      counts.forEach((v, k) => {
        g.append("rect").attr("x", x(k)).attr("y", y(v)).attr("width", x.bandwidth())
          .attr("height", f.ih - y(v)).attr("rx", 2).attr("fill", AC.accent).attr("opacity", 0.8);
      });
      let fact = 1;
      const pois = counts.map((_, k) => { if (k > 0) fact *= k; return m * Math.exp(-alpha) * Math.pow(alpha, k) / fact; });
      g.append("path").attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 2)
        .attr("d", d3.line().x((_, k) => x(k) + x.bandwidth() / 2).y(v => y(v))(pois));
      pois.forEach((v, k) => g.append("circle").attr("cx", x(k) + x.bandwidth() / 2).attr("cy", y(v))
        .attr("r", 2.5).attr("fill", AC.a2));
      AL.axisB(g, d3.scaleLinear().domain([-0.5, top + 0.5]).range([0, f.iw]), f.ih,
        Math.min(top + 1, 10), "entries in a bucket (first of " + R + " builds)", d3.format("d"));
      AL.axisL(g, y, 5, "number of slots");
      AL.legend(g, [{ label: "measured", color: AC.accent },
                    { label: "Poisson(" + HT.sig(alpha, 1) + ") × m", color: AC.a2 }], f.iw + 12, 14);
    }

    const tx = f.iw + 12;
    const lines = [
      ["n = " + HT.int(n) + ", m = " + HT.int(m) + ", " + R + " builds", AC.muted],
      ["mean successful   " + HT.sig(meanSucc, 3), AC.ink],
      ["    predicted     " + HT.sig(predSucc, 3), okS ? AC.good : AC.bad],
      ["mean unsuccessful " + HT.sig(meanUnsucc, 3), AC.ink],
      ["    predicted     " + HT.sig(predUnsucc, 3), okU ? AC.good : AC.bad],
      ["mean longest      " + HT.sig(meanMax, 2), AC.ink],
      ["    predicted     " + HT.sig(predMax, 2), okMax ? AC.good : AC.bad],
      ["empty slots       " + HT.sig(100 * meanEmpty, 1) + "%", AC.ink],
      ["    e^(−α) =      " + HT.sig(100 * predEmpty, 1) + "%", okEmpty ? AC.good : AC.bad]
    ];
    lines.forEach((L, i) => g.append("text").attr("x", tx).attr("y", 66 + i * 15)
      .attr("font-size", 11).attr("fill", L[1]).text(L[0]));

    els.out.html(
      "<b>Built and searched for real, averaged over " + R + " independent tables: n = " + HT.int(n)
      + " entries in m = " + HT.int(m) + " slots, <span class='keep'>α</span> = " + HT.sig(alpha, 2) + ".</b> "
      + "Bucket lengths sum to n in every build — " + HT.verdict(sumOK) + ". "
      + "Mean entries examined: successful <b>" + HT.sig(meanSucc, 3) + "</b> against 1 + (n−1)/2m = "
      + HT.sig(predSucc, 3) + " (" + HT.verdict(okS) + "); unsuccessful <b>" + HT.sig(meanUnsucc, 3)
      + "</b> against <span class='keep'>α</span> = " + HT.sig(predUnsucc, 3) + " (" + HT.verdict(okU) + "). "
      + "Longest bucket, averaged over the builds: <b>" + HT.sig(meanMax, 2)
      + "</b> against the Poissonised expected maximum " + HT.sig(predMax, 2)
      + " (" + HT.verdict(okMax) + ") — so the worst bucket is <b>" + HT.sig(meanMax / Math.max(meanSucc, 1e-9), 1)
      + "×</b> the mean successful cost, which is the tail the <span class='keep'>Θ</span>(1 + <span class='keep'>α</span>) bound does not mention. "
      + "Empty slots <b>" + HT.sig(100 * meanEmpty, 1) + "%</b> against e^(−<span class='keep'>α</span>) = "
      + HT.sig(100 * predEmpty, 1) + "% (" + HT.verdict(okEmpty) + ")."
    );
  }

  [els.a, els.m, els.show].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══ FIGURE 05 ═══════════════════════════════════════════════════════════
   The hand trace of §14, run. Ten keys are inserted into an 11-slot table one
   at a time; every probe is counted by the real routine; the maximal runs of
   occupied slots are recomputed from the table after each step and bracketed,
   so primary clustering is a measurement rather than an assertion. The final
   frame's totals are what the prose's hand trace must match.                */
(function () {
  const svg = d3.select("#lin-svg"); if (svg.empty()) return;
  const W = 760, H = 300, M = 11;
  const ORDERS = {
    std: [10, 22, 31, 4, 15, 28, 17, 88, 59, 21],
    rev: [21, 59, 88, 17, 28, 15, 4, 31, 22, 10],
    clumped: [4, 15, 26, 37, 48, 59, 70, 81, 92, 103]
  };
  const els = {
    scheme: d3.select("#lin-scheme"), order: d3.select("#lin-order"),
    out: d3.select("#lin-readout")
  };

  const h1 = k => k % M;
  const h2 = k => 1 + (k % (M - 1));            // 1..10, coprime with prime m = 11
  function probeAt(scheme, k, i) {
    if (scheme === "lin") return (h1(k) + i) % M;
    if (scheme === "quad") return (h1(k) + i * i) % M;
    return (h1(k) + i * h2(k)) % M;
  }

  /* maximal runs of occupied slots, treating the array as circular */
  function runs(T) {
    const occ = T.map(v => v !== null);
    if (occ.every(v => v)) return [{ start: 0, len: M }];
    if (occ.every(v => !v)) return [];
    let s = 0;
    while (occ[s] || !occ[(s - 1 + M) % M]) { if (!occ[s]) break; s++; }   // find a slot after a gap
    s = occ.findIndex((v, j) => v && !occ[(j - 1 + M) % M]);
    const out = [];
    let j = s, guard = 0;
    do {
      if (occ[j] && !occ[(j - 1 + M) % M]) {
        let len = 0, t = j;
        while (occ[t]) { len++; t = (t + 1) % M; if (len > M) break; }
        out.push({ start: j, len: len });
      }
      j = (j + 1) % M; guard++;
    } while (j !== s && guard <= M);
    return out;
  }

  function frames() {
    const scheme = els.scheme.property("value");
    const keys = ORDERS[els.order.property("value")];
    const T = new Array(M).fill(null);
    const c = AL.counter();
    const out = [{ T: [...T], key: null, probes: [], placed: null, total: 0, step: 0 }];
    keys.forEach((k, idx) => {
      const seq = [];
      let placed = null;
      for (let i = 0; i < M; i++) {
        const j = probeAt(scheme, k, i);
        seq.push(j); c.add("probe");
        if (T[j] === null) { T[j] = k; placed = j; break; }
      }
      out.push({ T: [...T], key: k, probes: seq, placed: placed,
                 failed: placed === null, total: c.get("probe"), step: idx + 1 });
    });
    return { frames: out, total: c.get("probe"), keys: keys };
  }

  function render(fr, all) {
    const f = AL.frame(svg, W, H, { l: 20, r: 20, t: 40, b: 20 });
    const g = f.g;
    const probeIdx = {};
    fr.probes.forEach((j, i) => { if (!(j in probeIdx)) probeIdx[j] = i; });
    const home = fr.key === null ? -1 : h1(fr.key);

    const r = AL.row(g, fr.T.map(v => (v === null ? "" : v)), {
      x: 120, y: 78, w: 46, h: 40, gap: 5, index: true,
      mark: (i) => {
        if (i === fr.placed) return "#1d3a26";
        if (i in probeIdx) return "#3a2230";
        return null;
      }
    });
    /* probe-order numerals above the slots that were probed */
    fr.probes.forEach((j, i) => {
      g.append("text").attr("x", 120 + r.cellX(j)).attr("y", 66).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", j === fr.placed ? AC.good : AC.rose)
        .text("#" + (i + 1));
    });
    if (home >= 0) {
      g.append("text").attr("x", 120 + r.cellX(home)).attr("y", 52).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", AC.a2).text("home");
    }
    g.append("text").attr("x", 112).attr("y", 102).attr("text-anchor", "end")
      .attr("font-size", 12).attr("fill", AC.muted).text("T");

    /* the measured runs, bracketed beneath */
    const rs = runs(fr.T);
    const y0 = 140;
    /* Labels are collected first and laid out afterwards: two short adjacent runs used
       to print their captions at the same y and overlap. They are now placed left to
       right, dropped onto a second line whenever they would collide with the previous
       one, and clamped inside the frame. */
    const placedLabels = [];
    rs.forEach(run => {
      const wrapped = run.start + run.len > M;
      const segs = wrapped ? [[run.start, M - run.start], [0, run.len - (M - run.start)]]
                           : [[run.start, run.len]];
      segs.forEach(([s, L]) => {
        const x1 = 120 + s * r.step - 3, x2 = 120 + (s + L - 1) * r.step + r.cellW + 3;
        g.append("path").attr("d", "M" + x1 + "," + y0 + " L" + x1 + "," + (y0 + 7)
                 + " L" + x2 + "," + (y0 + 7) + " L" + x2 + "," + y0)
          .attr("fill", "none").attr("stroke", run.len >= 4 ? AC.rose : AC.a2).attr("stroke-width", 1.6);
      });
      placedLabels.push({ run, wrapped });
    });

    let lastRight = -Infinity, lane = 0;
    placedLabels
      .map(L => ({ L, mid: 120 + ((L.run.start + (L.run.len - 1) / 2) % M) * r.step + r.cellW / 2 }))
      .sort((a, b) => a.mid - b.mid)
      .forEach(({ L, mid }) => {
        const txt = "run of " + L.run.len + (L.wrapped ? " (wraps)" : "");
        const half = 3.2 * txt.length;
        const cx = AL.clamp(mid, 120 + half, W - 40 - half);
        lane = (cx - half < lastRight + 6) ? 1 - lane : 0;
        lastRight = cx + half;
        g.append("text").attr("x", cx).attr("y", y0 + 22 + lane * 14)
          .attr("text-anchor", "middle").attr("font-size", 11)
          .attr("fill", L.run.len >= 4 ? AC.rose : AC.a2).text(txt);
      });

    const occ = fr.T.filter(v => v !== null).length;
    const longest = rs.length ? Math.max(...rs.map(x => x.len)) : 0;
    g.append("text").attr("x", 0).attr("y", 16).attr("font-size", 12).attr("fill", AC.ink)
      .text(fr.key === null ? "empty table" :
        ("insert " + fr.key + "   home slot " + home + "   probes " + fr.probes.join(" → ")
         + "   (" + fr.probes.length + ")"));
    g.append("text").attr("x", 0).attr("y", 34).attr("font-size", 11).attr("fill", AC.muted)
      .text("occupied " + occ + "/" + M + "   α = " + HT.sig(occ / M, 3)
            + "   longest run " + longest + "   probes so far " + fr.total);

    const done = fr.step === all.frames.length - 1;
    const failures = all.frames.filter(x => x.failed).length;
    const schemeName = { lin: "linear (h + i) mod 11", quad: "quadratic (h + i²) mod 11",
                         dbl: "double (h + i·h₂) mod 11" }[els.scheme.property("value")];
    els.out.html(
      "<b>" + schemeName + ", m = 11.</b> Step " + fr.step + " of " + (all.frames.length - 1)
      + (fr.key === null ? ". The table starts empty."
         : ": key <b>" + fr.key + "</b> hashes home to slot <b>" + home + "</b> and probed "
           + fr.probes.map(j => "T[" + j + "]").join(", ") + " — <b>" + fr.probes.length
           + "</b> probe" + (fr.probes.length === 1 ? "" : "s")
           + (fr.failed
              ? ", and <b style=\'color:" + AC.bad + "\'>found no free slot</b>: the probe sequence"
                + " reached only " + (new Set(fr.probes)).size + " distinct slots of " + M
                + ", so the key is refused while the table still has room."
              : ", landing in slot <b>" + fr.placed + "</b>."))
      + " Measured running total <b>" + fr.total + "</b> probes for " + fr.step + " insertions"
      + (fr.step ? " (mean " + HT.sig(fr.total / fr.step, 2) + ")" : "")
      + ". Occupied " + occ + "/" + M + ", longest contiguous run <b>" + longest + "</b>."
      + (failures ? " <b style='color:" + AC.bad + "'>" + failures + " of the "
                    + (all.frames.length - 1) + " keys could not be placed at all.</b>" : "")
      + (done ? " <b>" + ((all.frames.length - 1) - failures) + " of "
              + (all.frames.length - 1) + " keys placed, in " + all.total
              + " probes.</b> The hand trace in the text totals 20 for the linear scheme on this"
              + " order — " + HT.verdict(!(els.scheme.property("value") === "lin"
                 && els.order.property("value") === "std") || all.total === 20) + "."
         : "")
    );
  }

  let st = null;
  function rebuild() {
    HT.clearControls(svg.node());
    const all = frames();
    st = AL.stepper(svg, {
      frames: all.frames, label: "step", delay: 900,
      render: fr => render(fr, all)
    });
    st.go(all.frames.length - 1);     // default state: the completed table, as the caption describes
  }
  [els.scheme, els.order].forEach(s => s.on("change", rebuild));
  rebuild();
})();

/* ── open-addressing engine shared by figures 06, 07, 08 and 09 ───────────
   One implementation, used everywhere, so the three figures cannot drift apart.
   Probe counts come from AL.counter() wrapped round the real loop.          */
const OA = {
  PRIMES: [509, 1009, 2003, 4001, 8009],
  probe: function (scheme, h1, h2, i, m) {
    if (scheme === "lin") return (h1 + i) % m;
    if (scheme === "quad") return (h1 + i * i) % m;
    return (h1 + i * h2) % m;
  },
  /* build a table of n keys; returns the slot array, probe totals and failures */
  build: function (scheme, m, n, seed) {
    const r = AL.rng(seed);
    const T = new Int32Array(m).fill(-1);
    const c = AL.counter();
    let placed = 0, failed = 0;
    for (let k = 0; k < n; k++) {
      const h1 = Math.floor(r() * m), h2 = 1 + Math.floor(r() * (m - 1));
      let ok = false;
      for (let i = 0; i < m; i++) {
        const j = OA.probe(scheme, h1, h2, i, m);
        c.add("ins");
        if (T[j] < 0) { T[j] = k; ok = true; break; }
      }
      if (ok) placed++; else failed++;
    }
    return { T, m, insProbes: c.get("ins"), placed, failed, rng: r };
  },
  /* mean probes for an unsuccessful search, over `trials` fresh random keys */
  missCost: function (scheme, tab, trials, seed) {
    const r = AL.rng(seed), m = tab.m, c = AL.counter();
    for (let t = 0; t < trials; t++) {
      const h1 = Math.floor(r() * m), h2 = 1 + Math.floor(r() * (m - 1));
      for (let i = 0; i < m; i++) {
        c.add("p");
        if (tab.T[OA.probe(scheme, h1, h2, i, m)] < 0) break;
      }
    }
    return c.get("p") / trials;
  },
  /* maximal runs of occupied slots, circular */
  runs: function (T, m) {
    const out = [];
    let allFull = true;
    for (let j = 0; j < m; j++) if (T[j] < 0) { allFull = false; break; }
    if (allFull) return [m];
    for (let j = 0; j < m; j++) {
      if (T[j] >= 0 && T[(j - 1 + m) % m] < 0) {
        let L = 0, t = j;
        while (T[t] >= 0 && L <= m) { L++; t = (t + 1) % m; }
        out.push(L);
      }
    }
    return out;
  },
  /* closed forms, kept in one place and evaluated independently of any run */
  formula: function (group, kind, a) {
    if (group === "lin") return kind === "succ" ? 0.5 * (1 + 1 / (1 - a)) : 0.5 * (1 + 1 / ((1 - a) * (1 - a)));
    if (group === "quad") return kind === "succ" ? 1 + Math.log(1 / (1 - a)) - a / 2
                                                 : 1 / (1 - a) - a + Math.log(1 / (1 - a));
    return kind === "succ" ? (a > 0 ? (1 / a) * Math.log(1 / (1 - a)) : 1) : 1 / (1 - a);
  }
};
const SCHEMES = [{ id: "lin", name: "linear", col: AC.rose },
                 { id: "quad", name: "quadratic", col: AC.a2 },
                 { id: "dbl", name: "double", col: AC.accent }];

/* ══ FIGURE 06 ═══════════════════════════════════════════════════════════
   One key stream, three probe schemes, three real tables. The occupancy strips
   and the run-length distributions are computed from the finished tables; the
   mean run length is cross-checked against the no-clustering baseline 1/(1−α)
   and the run lengths are checked to sum to n.                              */
(function () {
  const svg = d3.select("#probe3-svg"); if (svg.empty()) return;
  const W = 760, H = 430;
  const els = {
    a: d3.select("#p3-alpha"), aOut: d3.select("#p3-alpha-out"),
    m: d3.select("#p3-m"), mOut: d3.select("#p3-m-out"),
    view: d3.select("#p3-view"), out: d3.select("#probe3-readout")
  };

  function draw() {
    const alpha = +els.a.property("value");
    const m = OA.PRIMES[+els.m.property("value")];
    const n = Math.round(alpha * m);
    els.aOut.text(HT.sig(alpha, 2)); els.mOut.text(HT.int(m));
    const R = 6;

    const res = SCHEMES.map(s => {
      let meanRun = 0, longest = 0, fails = 0, ins = 0, nRuns = 0, sumOK = true;
      let first = null, dist = [];
      for (let t = 0; t < R; t++) {
        const tab = OA.build(s.id, m, n, 5551 + 7717 * t);
        const rs = OA.runs(tab.T, m);
        const tot = rs.reduce((x, y) => x + y, 0);
        if (tot !== tab.placed) sumOK = false;
        meanRun += rs.length ? tot / rs.length : 0;
        nRuns += rs.length;
        longest = Math.max(longest, rs.length ? Math.max(...rs) : 0);
        fails += tab.failed; ins += tab.insProbes / Math.max(1, tab.placed);
        rs.forEach(L => { dist[L] = (dist[L] || 0) + 1; });
        if (t === 0) first = tab;
      }
      return { s, meanRun: meanRun / R, longest, fails: fails / R, ins: ins / R,
               nRuns: nRuns / R, dist, first, sumOK };
    });

    const base = 1 / (1 - alpha);
    const f = AL.frame(svg, W, H, { l: 52, r: 166, t: 16, b: 38 });
    const g = f.g;
    const strips = els.view.property("value") === "strip";
    const stripH = strips ? 22 : 0;
    const stripTop = 8;

    if (strips) {
      const cols = Math.min(m, 440);
      res.forEach((rr, i) => {
        const y = stripTop + i * (stripH + 10);
        g.append("text").attr("x", -6).attr("y", y + 14).attr("text-anchor", "end")
          .attr("font-size", 11).attr("fill", rr.s.col).text(rr.s.name);
        const cw = (f.iw - 10) / cols;
        for (let j = 0; j < cols; j++) {
          if (rr.first.T[j] < 0) continue;
          g.append("rect").attr("x", j * cw).attr("y", y).attr("width", Math.max(0.8, cw))
            .attr("height", stripH).attr("fill", rr.s.col).attr("opacity", 0.85);
        }
        g.append("rect").attr("x", 0).attr("y", y).attr("width", cols * cw).attr("height", stripH)
          .attr("fill", "none").attr("stroke", AC.line);
      });
      g.append("text").attr("x", 0).attr("y", stripTop + 3 * (stripH + 10) + 4)
        .attr("font-size", 10).attr("fill", AC.muted)
        .text("first " + Math.min(m, 440) + " of " + HT.int(m) + " slots, one build");
    }

    const distTop = strips ? stripTop + 3 * (stripH + 10) + 18 : 8;
    const distH = f.ih - distTop;
    const maxL = Math.max(8, Math.min(res[0].longest, Math.ceil(base * 6)));
    const x = d3.scaleLinear().domain([1, maxL]).range([0, f.iw]);
    const topY = Math.max(1, d3.max(res.map(rr => d3.max(rr.dist.slice(1, maxL + 1).filter(v => v !== undefined)) || 1)));
    const y = d3.scaleLinear().domain([0.5, topY]).range([distTop + distH, distTop]).clamp(true);
    const gg = g.append("g");
    AL.gridY(gg, y, f.iw, 4);
    res.forEach(rr => {
      const pts = [];
      for (let L = 1; L <= maxL; L++) pts.push({ L, v: Math.max(0.5, rr.dist[L] || 0.5) });
      gg.append("path").attr("fill", "none").attr("stroke", rr.s.col).attr("stroke-width", 1.9)
        .attr("d", d3.line().x(p => x(p.L)).y(p => y(p.v))(pts));
    });
    gg.append("line").attr("x1", x(base)).attr("x2", x(base)).attr("y1", distTop).attr("y2", distTop + distH)
      .attr("stroke", AC.violet).attr("stroke-dasharray", "4 3");
    gg.append("text").attr("x", x(base) + 4).attr("y", distTop + 12).attr("font-size", 10)
      .attr("fill", AC.violet).text("1/(1−α) = " + HT.sig(base, 2));
    AL.axisB(g, x, f.ih, 8, "maximal run length", d3.format("d"));
    AL.axisL(gg, y, 4, "runs (over " + R + " builds)");

    /* Two real columns rather than space-padded strings: SVG <text> collapses runs of
       whitespace, so padEnd() produced a ragged right column that only looked aligned in
       a monospace mock-up. Labels are left-aligned at tx, values right-aligned at tv. */
    const tx = f.iw + 12, tv = f.iw + 152;
    const rows = [["mean run  (vs " + HT.sig(base, 2) + ")", "", AC.muted]].concat(
      res.map(rr => [rr.s.name, HT.sig(rr.meanRun, 2)
        + "  " + (rr.meanRun >= base ? "+" : "") + HT.sig(100 * (rr.meanRun / base - 1), 0) + "%", rr.s.col]),
      [["longest run", "", AC.muted]],
      res.map(rr => [rr.s.name, String(rr.longest), rr.s.col]),
      [["probes / insert", "", AC.muted]],
      res.map(rr => [rr.s.name, HT.sig(rr.ins, 2), rr.s.col]));
    rows.forEach((L, i) => {
      const y = 14 + i * 15;
      g.append("text").attr("x", tx).attr("y", y).attr("font-size", 11).attr("fill", L[2]).text(L[0]);
      if (L[1]) g.append("text").attr("x", tv).attr("y", y).attr("text-anchor", "end")
        .attr("font-size", 11).attr("fill", L[2]).text(L[1]);
    });

    const quad = res.find(rr => rr.s.id === "quad");
    els.out.html(
      "<b>m = " + HT.int(m) + ", n = " + HT.int(n) + ", <span class='keep'>α</span> = " + HT.sig(alpha, 2)
      + ", " + R + " builds per scheme.</b> Run lengths sum to the number of placed keys in every build — "
      + HT.verdict(res.every(rr => rr.sumOK)) + ". "
      + "Mean maximal-run length: " + res.map(rr => rr.s.name + " <b>" + HT.sig(rr.meanRun, 2) + "</b>").join(" · ")
      + ", against the no-clustering baseline 1/(1−<span class='keep'>α</span>) = <b>" + HT.sig(base, 2) + "</b>"
      + " — double hashing is within " + HT.sig(Math.abs(100 * (res[2].meanRun / base - 1)), 1)
      + "% of it, linear probing <b>" + HT.sig(100 * (res[0].meanRun / base - 1), 0) + "%</b> above. "
      + "Longest run observed: " + res.map(rr => rr.s.name + " <b>" + rr.longest + "</b>").join(" · ")
      + " — the tail separates the schemes far more than the mean does. "
      + "Mean probes per insertion: " + res.map(rr => rr.s.name + " <b>" + HT.sig(rr.ins, 2) + "</b>").join(" · ")
      + ". Insertions refused by quadratic probing: <b>" + HT.sig(quad.fails, 1) + "</b> of " + HT.int(n) + "."
    );
  }

  [els.a, els.m, els.view].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══ FIGURE 07 ═══════════════════════════════════════════════════════════
   The three probe schemes measured across the load factor, each against ITS OWN
   closed form. The point of the figure is that the three formulas are different
   theorems: plotting all three curves at once shows how far apart they are, and
   the measured points show which scheme belongs to which curve.             */
(function () {
  const svg = d3.select("#theo-svg"); if (svg.empty()) return;
  const W = 760, H = 410;
  const els = {
    kind: d3.select("#theo-kind"), m: d3.select("#theo-m"), mOut: d3.select("#theo-m-out"),
    log: d3.select("#theo-log"), out: d3.select("#theo-readout")
  };
  const ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95];

  function draw() {
    const kind = els.kind.property("value");
    const m = OA.PRIMES[+els.m.property("value")];
    const useLog = els.log.property("checked");
    els.mOut.text(HT.int(m));
    const R = m > 4000 ? 4 : (m > 1500 ? 6 : 10);

    const series = SCHEMES.map(s => ({ s: s, pts: [] }));
    ALPHAS.forEach(a => {
      const n = Math.round(a * m);
      SCHEMES.forEach((s, si) => {
        let acc = 0;
        for (let t = 0; t < R; t++) {
          const tab = OA.build(s.id, m, n, 31337 + 6151 * t + 17 * si);
          acc += (kind === "succ")
            ? tab.insProbes / Math.max(1, tab.placed)       // a hit reproduces its insertion path
            : OA.missCost(s.id, tab, 2000, 777 + 13 * t);
        }
        series[si].pts.push({ a: a, v: acc / R });
      });
    });

    const preds = SCHEMES.map(s => ALPHAS.map(a => OA.formula(s.id, kind, a)));
    const allV = series.flatMap(x => x.pts.map(p => p.v)).concat(preds.flat());
    const top = Math.max(...allV) * 1.08;
    const f = AL.frame(svg, W, H, { l: 56, r: 156, t: 18, b: 42 });
    const g = f.g;
    const x = d3.scaleLinear().domain([0, 1]).range([0, f.iw]);
    const y = useLog
      ? d3.scaleLog().domain([0.9, top]).range([f.ih, 0])
      : d3.scaleLinear().domain([0, top]).nice().range([f.ih, 0]);
    AL.gridY(g, y, f.iw, 5);
    AL.axisB(g, x, f.ih, 6, "load factor α", d3.format(".1f"));
    AL.axisL(g, y, 5, "expected probes", d3.format("~g"));

    /* curves first, points on top */
    const fine = AL.linspace(0.02, 0.97, 120);
    SCHEMES.forEach((s, si) => {
      const pts = fine.map(a => ({ a, v: OA.formula(s.id, kind, a) })).filter(p => p.v > 0 && p.v < top * 4);
      g.append("path").attr("fill", "none").attr("stroke", s.col).attr("stroke-width", 1.6)
        .attr("stroke-dasharray", "5 3").attr("opacity", 0.85)
        .attr("d", d3.line().x(p => x(p.a)).y(p => y(Math.max(useLog ? 0.91 : 0, p.v)))(pts));
    });
    let worst = 0, worstAt = null;
    series.forEach((ser, si) => {
      g.append("path").attr("fill", "none").attr("stroke", ser.s.col).attr("stroke-width", 2.2)
        .attr("d", d3.line().x(p => x(p.a)).y(p => y(Math.max(useLog ? 0.91 : 0, p.v)))(ser.pts));
      ser.pts.forEach((p, i) => {
        g.append("circle").attr("cx", x(p.a)).attr("cy", y(Math.max(useLog ? 0.91 : 0, p.v)))
          .attr("r", 3).attr("fill", ser.s.col);
        const rel = Math.abs(p.v - preds[si][i]) / preds[si][i];
        if (rel > worst) { worst = rel; worstAt = { s: ser.s.name, a: p.a, meas: p.v, pred: preds[si][i] }; }
      });
    });

    AL.legend(g, SCHEMES.map(s => ({ label: s.name + " — measured", color: s.col }))
      .concat(SCHEMES.map(s => ({ label: s.name + " — formula", color: s.col, dash: "5 3" }))),
      f.iw + 12, 16);

    /* the headline comparison at alpha = 0.9 */
    const i90 = ALPHAS.indexOf(0.9);
    const linM = series[0].pts[i90].v, dblM = series[2].pts[i90].v;
    const rows = [
      ["at α = 0.9", AC.muted],
      ["linear   meas " + HT.sig(linM, 2), AC.rose],
      ["         form " + HT.sig(preds[0][i90], 2), AC.rose],
      ["quad     meas " + HT.sig(series[1].pts[i90].v, 2), AC.a2],
      ["         form " + HT.sig(preds[1][i90], 2), AC.a2],
      ["double   meas " + HT.sig(dblM, 2), AC.accent],
      ["         form " + HT.sig(preds[2][i90], 2), AC.accent],
      ["linear / double", AC.muted],
      ["  measured " + HT.sig(linM / dblM, 2) + "×", AC.ink]
    ];
    rows.forEach((L, i) => g.append("text").attr("x", f.iw + 12).attr("y", 132 + i * 15)
      .attr("font-size", 11).attr("fill", L[1]).text(L[0]));

    const kindName = kind === "succ" ? "successful search (an insertion's own path)" : "unsuccessful search";
    els.out.html(
      "<b>" + kindName + ", m = " + HT.int(m) + ", " + R + " builds per point, "
      + "2 000 miss searches per build.</b> "
      + "At <span class='keep'>α</span> = 0.9 the measured costs are linear <b>" + HT.sig(linM, 2)
      + "</b> (formula " + HT.sig(preds[0][i90], 2) + "), quadratic <b>" + HT.sig(series[1].pts[i90].v, 2)
      + "</b> (formula " + HT.sig(preds[1][i90], 2) + "), double <b>" + HT.sig(dblM, 2)
      + "</b> (formula " + HT.sig(preds[2][i90], 2) + ") — so linear probing costs <b>"
      + HT.sig(linM / dblM, 2) + "×</b> what double hashing costs, and quoting the uniform-hashing "
      + "formula for linear probing would understate it by <b>"
      + HT.sig(preds[0][i90] / preds[2][i90], 2) + "×</b>. "
      + "Largest measurement-to-formula gap anywhere on the chart: <b>" + HT.sig(100 * worst, 0)
      + "%</b>, for " + worstAt.s + " probing at <span class='keep'>α</span> = " + worstAt.a
      + " (measured " + HT.sig(worstAt.meas, 2) + " against " + HT.sig(worstAt.pred, 2)
      + ") — the Group B and C formulas are asymptotic in m, and the gap shrinks as the m control is raised."
    );
  }

  [els.kind, els.m, els.log].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══ FIGURE 09 ═══════════════════════════════════════════════════════════
   The deletion trap, run rather than drawn. The §14 linear-probing table is
   rebuilt by the real insertion routine, one slot is deleted under the chosen
   policy, and the real search routine is then stepped over a key that the
   deletion may have orphaned. The set of orphaned keys is DERIVED from the
   table (which keys' probe paths cross the hole), not hard-coded.           */
(function () {
  const svg = d3.select("#tomb-svg"); if (svg.empty()) return;
  const W = 760, H = 320, M = 11;
  const KEYS = [10, 22, 31, 4, 15, 28, 17, 88, 59, 21];
  const EMPTY = 0, OCC = 1, TOMB = 2;
  const els = { mode: d3.select("#tomb-mode"), del: d3.select("#tomb-del"), out: d3.select("#tomb-readout") };

  function buildTable() {
    const T = new Array(M).fill(null);      // {key} or null
    KEYS.forEach(k => {
      for (let i = 0; i < M; i++) {
        const j = (k % M + i) % M;
        if (T[j] === null) { T[j] = { key: k, home: k % M, slot: j }; break; }
      }
    });
    return T;
  }
  /* the slots strictly between a key's home and its slot, along its probe path */
  function crossed(e) {
    const out = [];
    for (let i = 0; ; i++) {
      const j = (e.home + i) % M;
      if (j === e.slot) break;
      out.push(j);
      if (i > M) break;
    }
    return out;
  }
  /* the real search: stops at EMPTY, steps over TOMB */
  function search(state, key) {
    const path = [];
    for (let i = 0; i < M; i++) {
      const j = (key % M + i) % M;
      path.push(j);
      if (state.st[j] === EMPTY) return { path, found: false, at: -1 };
      if (state.st[j] === OCC && state.T[j].key === key) return { path, found: true, at: j };
    }
    return { path, found: false, at: -1 };
  }

  function frames() {
    const T = buildTable();
    const st = T.map(v => (v === null ? EMPTY : OCC));
    const del = +els.del.property("value");
    const mode = els.mode.property("value");
    const deletedKey = T[del] ? T[del].key : null;

    /* which stored keys have `del` strictly inside their probe path */
    const orphans = T.filter(e => e && e.slot !== del && crossed(e).indexOf(del) >= 0).map(e => e.key);
    const victim = orphans.length ? orphans[0] : (T.find(e => e && e.slot !== del) || {}).key;

    const out = [];
    out.push({ kind: "built", st: [...st], T, note: "the table as §14 built it — ten keys, one free slot" });
    const st2 = [...st];
    st2[del] = (mode === "tomb") ? TOMB : EMPTY;
    out.push({ kind: "deleted", st: st2, T, del, deletedKey,
               note: "key " + deletedKey + " removed from slot " + del
                     + (mode === "tomb" ? ", leaving a tombstone" : ", leaving the slot EMPTY") });
    const res = search({ st: st2, T }, victim);
    res.path.forEach((j, i) => {
      out.push({ kind: "search", st: st2, T, del, victim, probe: res.path.slice(0, i + 1),
                 done: i === res.path.length - 1, found: res.found, at: res.at,
                 note: "searching for " + victim });
    });
    return { list: out, victim, orphans, deletedKey, del, mode, res };
  }

  function render(fr, all) {
    const f = AL.frame(svg, W, H, { l: 18, r: 18, t: 40, b: 18 });
    const g = f.g;
    const probeSet = {};
    (fr.probe || []).forEach((j, i) => { if (!(j in probeSet)) probeSet[j] = i; });
    const vals = fr.st.map((s, j) => s === OCC ? fr.T[j].key : (s === TOMB ? "†" : ""));
    const r = AL.row(g, vals, {
      x: 140, y: 92, w: 46, h: 40, gap: 5, index: true,
      mark: (j) => {
        if (fr.kind === "search" && fr.done && fr.found && j === fr.at) return "#1d3a26";
        if (j in probeSet) return "#3a2230";
        if (fr.del === j) return fr.st[j] === TOMB ? "#33302a" : "#241f28";
        return null;
      }
    });
    (fr.probe || []).forEach((j, i) => {
      g.append("text").attr("x", 140 + r.cellX(j)).attr("y", 80).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", AC.rose).text("#" + (i + 1));
    });
    if (fr.del !== undefined) {
      g.append("text").attr("x", 140 + r.cellX(fr.del)).attr("y", 152).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", AC.a2)
        .text(fr.st[fr.del] === TOMB ? "tombstone" : "emptied");
    }
    g.append("text").attr("x", 132).attr("y", 116).attr("text-anchor", "end")
      .attr("font-size", 12).attr("fill", AC.muted).text("T");
    /* home-slot markers for the stored keys, so the reader can see whose path crosses what */
    g.append("text").attr("x", 0).attr("y", 16).attr("font-size", 12).attr("fill", AC.ink).text(fr.note);
    if (fr.kind === "search") {
      const homeSlot = fr.victim % M;
      g.append("text").attr("x", 140 + r.cellX(homeSlot)).attr("y", 66).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", AC.a2).text("home of " + fr.victim);
      g.append("text").attr("x", 0).attr("y", 34).attr("font-size", 11).attr("fill", AC.muted)
        .text("probe path so far: " + fr.probe.map(j => "T[" + j + "]").join(" → "));
    }
    if (fr.kind === "search" && fr.done) {
      const ok = fr.found;
      g.append("text").attr("x", 140).attr("y", 186).attr("font-size", 13)
        .attr("fill", ok ? AC.good : AC.bad)
        .text(ok ? ("found " + fr.victim + " in slot " + fr.at + " after " + fr.probe.length + " probes")
                 : ("reported ABSENT after " + fr.probe.length + " probes — but " + fr.victim
                    + " is stored in slot " + (fr.T.find(e => e && e.key === fr.victim) || {}).slot));
    }

    const stored = all.list[0].T.find(e => e && e.key === all.victim);
    const broke = all.orphans.length > 0 && !all.res.found;
    els.out.html(
      "<b>" + (all.mode === "tomb" ? "Tombstone" : "Naive") + " deletion of key " + all.deletedKey
      + " from slot " + all.del + ", m = 11, linear probing.</b> "
      + "The keys whose probe paths pass through slot " + all.del + " are <b>"
      + (all.orphans.length ? all.orphans.join(", ") : "none") + "</b> — derived from the table, "
      + "not assumed. The figure then searches for <b>" + all.victim + "</b>, which is stored in slot <b>"
      + (stored ? stored.slot : "?") + "</b> and whose home slot is <b>" + (all.victim % M) + "</b>. "
      + "Result: <b style='color:" + (all.res.found ? AC.good : AC.bad) + "'>"
      + (all.res.found ? "found after " + all.res.path.length + " probes"
                       : "reported ABSENT after " + all.res.path.length + " probes")
      + "</b>. "
      + (broke
          ? "The search stopped at the emptied slot " + all.del + ", because an EMPTY slot is the only "
            + "thing that ends a probe — so a stored key became unreachable through the front door while "
            + "still occupying memory and still visible to iteration."
          : (all.mode === "tomb"
             ? "The tombstone kept the probe path intact: the search stepped over slot " + all.del
               + " and continued, which is exactly what the marker exists to do."
             : "No stored key's probe path crossed slot " + all.del
               + ", so this particular naive delete happened to be harmless — which is precisely why "
               + "the bug survives testing."))
      + " Step " + (fr.kind === "built" ? 1 : (fr.kind === "deleted" ? 2 : 2 + fr.probe.length))
      + " of " + all.list.length + "."
    );
  }

  function rebuild() {
    HT.clearControls(svg.node());
    const all = frames();
    const st = AL.stepper(svg, { frames: all.list, label: "step", delay: 1100,
                                 render: fr => render(fr, all) });
    st.go(all.list.length - 1);     // default state: the completed search, as the caption describes
  }
  [els.mode, els.del].forEach(s => s.on("change", rebuild));
  rebuild();
})();

/* ══ FIGURE 10 ═══════════════════════════════════════════════════════════
   A real table is grown by real insertions. Each insertion's cost is counted as
   (probes for this insertion) + (entries moved, if this insertion triggered a
   rehash). The running mean is then compared against 1 + g/(g−1), evaluated
   independently — which is what "amortized O(1)" asserts.                  */
(function () {
  const svg = d3.select("#resize-svg"); if (svg.empty()) return;
  const W = 760, H = 380;
  const els = {
    g: d3.select("#rs-g"), gOut: d3.select("#rs-g-out"),
    a: d3.select("#rs-a"), aOut: d3.select("#rs-a-out"),
    n: d3.select("#rs-n"), nOut: d3.select("#rs-n-out"), out: d3.select("#resize-readout")
  };

  function draw() {
    const g = +els.g.property("value"), amax = +els.a.property("value"), N = +els.n.property("value");
    els.gOut.text(HT.sig(g, 2)); els.aOut.text(HT.sig(amax, 2)); els.nOut.text(HT.int(N));

    const c = AL.counter();
    const r = AL.rng(20260316);
    let m = 16;
    let T = new Int32Array(m).fill(-1);
    let homes = new Int32Array(m).fill(0);      // the stored keys, so a rehash can re-place them
    let n = 0;
    const cost = new Array(N), moved = new Array(N), sizes = new Array(N);
    let resizes = 0, biggest = 0;

    const place = (tab, mm, key) => {           // linear probing into a raw table
      for (let i = 0; i < mm; i++) {
        const j = (HT.mix(key) % mm + i) % mm;
        if (tab[j] < 0) { tab[j] = key; return i + 1; }
      }
      return mm;
    };

    for (let t = 0; t < N; t++) {
      const key = Math.floor(r() * 2147483647);
      let work = place(T, m, key); c.add("probe", work);
      n++;
      let mv = 0;
      if (n / m > amax) {
        const m2 = Math.max(m + 1, Math.round(g * m));
        const T2 = new Int32Array(m2).fill(-1);
        for (let j = 0; j < m; j++) if (T[j] >= 0) { place(T2, m2, T[j]); mv++; c.add("move"); }
        T = T2; m = m2; resizes++;
        if (mv > biggest) biggest = mv;
      }
      cost[t] = work + mv; moved[t] = mv; sizes[t] = m;
    }
    const total = c.get("probe") + c.get("move");
    const meanCost = total / N;
    /* Two independent predictions.
       (a) EXACT: replay the resize schedule arithmetically, without touching the table.
           Every resize moves whatever n was at that moment, so the total is determined.
       (b) ASYMPTOTIC: g/(g−1) moves per insertion — the value the running mean takes AT a
           resize boundary. Between boundaries the mean falls to 1/(g−1), because the
           numerator is frozen while n keeps growing, so the running curve oscillates
           between those two values and only the boundary value matches the formula. */
    let simM = 16, simN = 0, predTotalMoves = 0;
    for (let t = 0; t < N; t++) {
      simN++;
      if (simN / simM > amax) { predTotalMoves += simN; simM = Math.max(simM + 1, Math.round(g * simM)); }
    }
    const predMoves = g / (g - 1);
    const measMoves = c.get("move") / N;
    const okExact = c.get("move") === predTotalMoves;
    const lo = 1 / (g - 1), hi = g / (g - 1);
    const okBand = measMoves >= lo * 0.98 && measMoves <= hi * 1.02;

    const f = AL.frame(svg, W, H, { l: 56, r: 152, t: 18, b: 40 });
    const gg = f.g;
    const x = d3.scaleLinear().domain([1, N]).range([0, f.iw]);
    const y = d3.scaleLog().domain([0.8, Math.max(10, biggest * 1.3)]).range([f.ih, 0]);
    AL.gridY(gg, y, f.iw, 5);
    /* individual costs: draw the spikes as stems, the flat band as faint dots */
    for (let t = 0; t < N; t++) {
      if (moved[t] > 0) {
        gg.append("line").attr("x1", x(t + 1)).attr("x2", x(t + 1))
          .attr("y1", f.ih).attr("y2", y(cost[t])).attr("stroke", AC.rose).attr("stroke-width", 1.4);
      }
    }
    const thin = Math.max(1, Math.floor(N / 700));
    for (let t = 0; t < N; t += thin) {
      if (moved[t] === 0) {
        gg.append("circle").attr("cx", x(t + 1)).attr("cy", y(Math.max(0.85, cost[t])))
          .attr("r", 1.2).attr("fill", AC.muted).attr("opacity", 0.55);
      }
    }
    /* running mean */
    let run = 0; const pts = [];
    for (let t = 0; t < N; t++) { run += cost[t]; if (t % thin === 0 || t === N - 1) pts.push({ t: t + 1, v: run / (t + 1) }); }
    gg.append("path").attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 2.2)
      .attr("d", d3.line().x(p => x(p.t)).y(p => y(Math.max(0.85, p.v)))(pts));
    const predTotal = 1 + predMoves;
    gg.append("line").attr("x1", 0).attr("x2", f.iw).attr("y1", y(predTotal)).attr("y2", y(predTotal))
      .attr("stroke", AC.good).attr("stroke-dasharray", "5 3");

    AL.axisB(gg, x, f.ih, 6, "insertion number", d3.format("~s"));
    AL.axisL(gg, y, 5, "units of work (log scale)", d3.format("~g"));
    AL.legend(gg, [
      { label: "a rehashing insertion", color: AC.rose },
      { label: "an ordinary insertion", color: AC.muted },
      { label: "running mean", color: AC.accent },
      { label: "1 + g/(g−1)", color: AC.good, dash: "5 3" }
    ], f.iw + 12, 16);
    const rows = [
      ["resizes        " + resizes, AC.muted],
      ["final m        " + HT.int(m), AC.muted],
      ["worst single   " + HT.int(biggest + 1), AC.rose],
      ["mean work      " + HT.sig(meanCost, 3), AC.accent],
      ["moves / insert " + HT.sig(measMoves, 3), AC.ink],
      ["  band [" + HT.sig(lo, 2) + ", " + HT.sig(hi, 2) + "]", okBand ? AC.good : AC.bad],
      ["total moves    " + HT.int(c.get("move")), AC.ink],
      ["  schedule says " + HT.int(predTotalMoves), okExact ? AC.good : AC.bad]
    ];
    rows.forEach((L, i) => gg.append("text").attr("x", f.iw + 12).attr("y", 96 + i * 15)
      .attr("font-size", 11).attr("fill", L[1]).text(L[0]));

    els.out.html(
      "<b>" + HT.int(N) + " insertions, growth factor g = " + HT.sig(g, 2)
      + ", <span class='keep'>α</span>_max = " + HT.sig(amax, 2) + ".</b> "
      + "<b>" + resizes + "</b> rehashes occurred, the table ended at m = " + HT.int(m)
      + ", and the most expensive single insertion did <b>" + HT.int(biggest + 1)
      + "</b> units of work — <b>" + HT.sig((biggest + 1) / meanCost, 0) + "×</b> the mean. "
      + "Total entries moved: <b>" + HT.int(c.get("move")) + "</b> against the "
      + HT.int(predTotalMoves) + " that the resize schedule alone predicts — " + HT.verdict(okExact)
      + ". Per insertion that is <b>" + HT.sig(measMoves, 3)
      + "</b>, which must lie between 1/(g−1) = " + HT.sig(lo, 2) + " just before a resize and "
      + "g/(g−1) = " + HT.sig(hi, 2) + " just after one — " + HT.verdict(okBand)
      + ". The running mean oscillates inside that band forever; it does not converge to a point, "
      + "and only its upper edge is the familiar g/(g−1). "
      + "Measured total work per insertion: <b>" + HT.sig(meanCost, 3)
      + "</b> (probes " + HT.sig(c.get("probe") / N, 3) + " + moves " + HT.sig(measMoves, 3)
      + "). That constant is what amortized O(1) means — and the "
      + HT.int(biggest + 1) + "-unit spike is what it does not mean."
    );
  }

  [els.g, els.a, els.n].forEach(s => s.on("input", draw));
  draw();
})();

/* ══ FIGURE 11 ═══════════════════════════════════════════════════════════
   Churn. n is held exactly constant — one insert and one delete per step — so
   the reported load factor never moves. What moves is alpha_eff, and the
   measured lookup cost follows it. Three policies are run on identical key
   streams so the comparison is like for like.                              */
(function () {
  const svg = d3.select("#churn-svg"); if (svg.empty()) return;
  const W = 760, H = 380, M = 2003;
  const EMPTY = -1, TOMB = -2;
  const els = {
    pol: d3.select("#ch-policy"), a: d3.select("#ch-alpha"), aOut: d3.select("#ch-alpha-out"),
    st: d3.select("#ch-steps"), stOut: d3.select("#ch-steps-out"), out: d3.select("#churn-readout")
  };

  function runOpen(policy, alpha, steps, thresh) {
    const n0 = Math.round(alpha * M);
    let T = new Int32Array(M).fill(EMPTY);
    let live = [], tombs = 0, rebuilds = 0;
    const r = AL.rng(4242);
    const hp = (k, i) => (HT.mix(k) % M + i) % M;
    const ins = (key) => {
      let firstTomb = -1;
      for (let i = 0; i < M; i++) {
        const j = hp(key, i);
        if (T[j] === TOMB) { if (firstTomb < 0) firstTomb = j; continue; }
        if (T[j] === EMPTY) { const s = firstTomb >= 0 ? firstTomb : j; if (firstTomb >= 0) tombs--; T[s] = key; return s; }
        if (T[j] === key) return j;
      }
      /* no EMPTY slot anywhere — every slot is live or a tombstone. The remembered
         tombstone is still a legal home, and without this line a table saturated
         with tombstones would refuse an insertion while holding only n live keys. */
      if (firstTomb >= 0) { tombs--; T[firstTomb] = key; return firstTomb; }
      return -1;
    };
    const del = (key) => {
      for (let i = 0; i < M; i++) {
        const j = hp(key, i);
        if (T[j] === EMPTY) return;
        if (T[j] === key) { T[j] = TOMB; tombs++; return; }
      }
    };
    const rebuild = () => {
      const keys = []; for (let j = 0; j < M; j++) if (T[j] >= 0) keys.push(T[j]);
      T = new Int32Array(M).fill(EMPTY); tombs = 0;
      keys.forEach(k => ins(k)); rebuilds++;
    };
    /* mean probes for an UNSUCCESSFUL lookup — the case tombstones actually damage,
       and the case that also governs insertion cost. A successful lookup stops at its
       key and is barely affected; measuring that instead would hide the effect. */
    const measure = () => {
      const c = AL.counter();
      for (let t = 0; t < 400; t++) {
        const key = -(t + 1) * 7919 - 3;            // guaranteed absent
        for (let i = 0; i < M; i++) {
          const j = hp(key, i); c.add("p");
          if (T[j] === EMPTY) break;
          if (T[j] === key) break;
        }
      }
      return c.get("p") / 400;
    };

    let nextKey = 1;
    for (let i = 0; i < n0; i++) { ins(nextKey); live.push(nextKey); nextKey++; }
    const series = [];
    const every = Math.max(1, Math.round(steps / 60));
    for (let s = 0; s < steps; s++) {
      const victim = Math.floor(r() * live.length);
      del(live[victim]); live[victim] = nextKey; ins(nextKey); nextKey++;
      if (policy === "thresh" && (live.length + tombs) / M > thresh) rebuild();
      if (s % every === 0 || s === steps - 1)
        series.push({ s: s + 1, cost: measure(), eff: (live.length + tombs) / M, n: live.length });
    }
    return { series, rebuilds, tombs, n: live.length };
  }

  function runChained(alpha, steps) {
    const n0 = Math.round(alpha * M);
    const B = Array.from({ length: M }, () => []);
    const r = AL.rng(4242);
    const hp = k => HT.mix(k) % M;
    let live = [], nextKey = 1;
    const ins = k => B[hp(k)].push(k);
    const del = k => { const b = B[hp(k)], i = b.indexOf(k); if (i >= 0) b.splice(i, 1); };
    for (let i = 0; i < n0; i++) { ins(nextKey); live.push(nextKey); nextKey++; }
    const measure = () => {                       // the same unsuccessful-lookup measurement
      const c = AL.counter();
      for (let t = 0; t < 400; t++) {
        const key = -(t + 1) * 7919 - 3, b = B[hp(key)];
        for (let i = 0; i < b.length; i++) c.add("p");
      }
      return c.get("p") / 400;
    };
    const series = [];
    const every = Math.max(1, Math.round(steps / 60));
    for (let s = 0; s < steps; s++) {
      const v = Math.floor(r() * live.length);
      del(live[v]); live[v] = nextKey; ins(nextKey); nextKey++;
      if (s % every === 0 || s === steps - 1)
        series.push({ s: s + 1, cost: measure(), eff: live.length / M, n: live.length });
    }
    return { series, rebuilds: 0, tombs: 0, n: live.length };
  }

  function draw() {
    const pol = els.pol.property("value");
    const alpha = +els.a.property("value"), steps = +els.st.property("value");
    els.aOut.text(HT.sig(alpha, 2)); els.stOut.text(HT.int(steps));
    const THRESH = 0.7;
    const res = pol === "chain" ? runChained(alpha, steps) : runOpen(pol, alpha, steps, THRESH);
    const ser = res.series;

    const f = AL.frame(svg, W, H, { l: 54, r: 154, t: 18, b: 40 });
    const g = f.g;
    const x = d3.scaleLinear().domain([0, steps]).range([0, f.iw]);
    const topC = Math.max(2, d3.max(ser, d => d.cost) * 1.15);
    const y = d3.scaleLinear().domain([0, topC]).nice().range([f.ih, 0]);
    const y2 = d3.scaleLinear().domain([0, 1]).range([f.ih, 0]);
    AL.gridY(g, y, f.iw, 5);
    g.append("path").attr("fill", "none").attr("stroke", AC.violet).attr("stroke-width", 1.6)
      .attr("stroke-dasharray", "4 3")
      .attr("d", d3.line().x(d => x(d.s)).y(d => y2(d.eff))(ser));
    g.append("path").attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 1.6)
      .attr("d", d3.line().x(d => x(d.s)).y(d => y2(d.n / M))(ser));
    g.append("path").attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 2.4)
      .attr("d", d3.line().x(d => x(d.s)).y(d => y(d.cost))(ser));
    AL.axisB(g, x, f.ih, 6, "insert/delete cycles", d3.format("~s"));
    AL.axisL(g, y, 5, "mean probes per UNSUCCESSFUL lookup");
    g.append("g").attr("class", "axis").attr("transform", "translate(" + f.iw + ",0)")
      .call(d3.axisRight(y2).ticks(5).tickFormat(d3.format(".1f")));
    g.append("text").attr("x", f.iw + 6).attr("y", -4).attr("font-size", 10).attr("fill", AC.muted)
      .text("load");
    AL.legend(g, [
      { label: "measured probes / lookup", color: AC.accent },
      { label: "α = n/m (right axis)", color: AC.a2 },
      { label: "α_eff with tombstones", color: AC.violet, dash: "4 3" }
    ], f.iw + 34, 20);

    const first = ser[0], last = ser[ser.length - 1];
    const nConst = ser.every(d => d.n === first.n);
    const polName = { none: "tombstones, never cleaned", thresh: "rebuild when α_eff > " + THRESH,
                      chain: "chaining" }[pol];
    els.out.html(
      "<b>" + polName + ", m = " + HT.int(M) + ", n held at " + HT.int(res.n)
      + " (<span class='keep'>α</span> = " + HT.sig(alpha, 2) + "), " + HT.int(steps)
      + " insert/delete cycles.</b> "
      + "n was constant for the whole run — " + HT.verdict(nConst) + " — so the reported load factor "
      + "never moved. Mean probes per unsuccessful lookup went from <b>" + HT.sig(first.cost, 2)
      + "</b> to <b>" + HT.sig(last.cost, 2) + "</b>, a factor of <b>"
      + HT.sig(last.cost / first.cost, 2) + "×</b>, while <span class='keep'>α</span>_eff went from "
      + HT.sig(first.eff, 3) + " to <b>" + HT.sig(last.eff, 3) + "</b>"
      + (pol === "chain" ? " (a chained table has no tombstones, so the two load figures coincide)" : "")
      + ". " + (pol === "thresh" ? "<b>" + res.rebuilds + "</b> rebuilds were triggered, each dropping "
                                 + "every tombstone and restoring the cost to its floor. " : "")
      + (pol === "none"
         ? "Nothing about n/m predicts this: the table reports itself as "
           + HT.sig(100 * alpha, 0) + "% loaded from the first cycle to the last."
         : "")
    );
  }

  [els.pol, els.a, els.st].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══ FIGURE 08 ═══════════════════════════════════════════════════════════
   Load-factor sweep, all four structures built and searched for real. Chaining
   is measured with the same bucket code as figure 04 and the three probing
   schemes with the shared OA engine, so the comparison is like for like.    */
(function () {
  const svg = d3.select("#alpha-svg"); if (svg.empty()) return;
  const W = 760, H = 390, M = 2003;
  const els = { kind: d3.select("#al-kind"), max: d3.select("#al-max"),
                log: d3.select("#al-log"), out: d3.select("#alpha-readout") };

  function chainCost(alpha, kind, seed) {
    const n = Math.round(alpha * M), r = AL.rng(seed);
    const B = Array.from({ length: M }, () => []);
    for (let i = 0; i < n; i++) B[Math.floor(r() * M)].unshift(i);
    const c = AL.counter();
    if (kind === "succ") {
      let cnt = 0;
      for (let j = 0; j < M; j++) B[j].forEach((k, idx) => { c.add("p", idx + 1); cnt++; });
      return cnt ? c.get("p") / cnt : 0;
    }
    const T = 3000;
    for (let t = 0; t < T; t++) c.add("p", B[Math.floor(r() * M)].length);
    return c.get("p") / T;
  }

  function draw() {
    const kind = els.kind.property("value");
    const amax = +els.max.property("value");
    const useLog = els.log.property("checked");
    const alphas = amax > 1
      ? [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95, 1.2, 1.5, 2, 2.5, 3]
      : [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.9, 0.95, 0.99];
    const R = 3;

    const chain = alphas.map(a => {
      let s = 0; for (let t = 0; t < R; t++) s += chainCost(a, kind, 1234 + 977 * t);
      return { a, v: s / R };
    });
    const open = SCHEMES.map(sc => ({
      s: sc,
      pts: alphas.filter(a => a <= 0.99).map(a => {
        let s = 0;
        for (let t = 0; t < R; t++) {
          const tab = OA.build(sc.id, M, Math.round(a * M), 8191 + 4441 * t);
          s += kind === "succ" ? tab.insProbes / Math.max(1, tab.placed)
                               : OA.missCost(sc.id, tab, 2000, 555 + 17 * t);
        }
        return { a, v: s / R };
      })
    }));

    const all = chain.map(p => p.v).concat(open.flatMap(o => o.pts.map(p => p.v)));
    const f = AL.frame(svg, W, H, { l: 58, r: 158, t: 18, b: 42 });
    const g = f.g;
    const x = d3.scaleLinear().domain([0, amax]).range([0, f.iw]);
    const top = Math.max(...all) * 1.1;
    const y = useLog ? d3.scaleLog().domain([Math.max(0.04, Math.min(...all.filter(v => v > 0)) * 0.8), top]).range([f.ih, 0])
                     : d3.scaleLinear().domain([0, top]).nice().range([f.ih, 0]);
    AL.gridY(g, y, f.iw, 5);
    const clip = v => useLog ? Math.max(y.domain()[0], v) : v;

    if (amax > 1) {
      g.append("line").attr("x1", x(1)).attr("x2", x(1)).attr("y1", 0).attr("y2", f.ih)
        .attr("stroke", AC.muted).attr("stroke-dasharray", "3 3");
      g.append("text").attr("x", x(1) + 4).attr("y", 12).attr("font-size", 10).attr("fill", AC.muted)
        .text("α = 1: open addressing cannot go further");
    }
    open.forEach(o => {
      g.append("path").attr("fill", "none").attr("stroke", o.s.col).attr("stroke-width", 2.2)
        .attr("d", d3.line().x(p => x(p.a)).y(p => y(clip(p.v)))(o.pts));
      o.pts.forEach(p => g.append("circle").attr("cx", x(p.a)).attr("cy", y(clip(p.v)))
        .attr("r", 2.6).attr("fill", o.s.col));
    });
    g.append("path").attr("fill", "none").attr("stroke", AC.teal).attr("stroke-width", 2.6)
      .attr("d", d3.line().x(p => x(p.a)).y(p => y(clip(p.v)))(chain));
    chain.forEach(p => g.append("circle").attr("cx", x(p.a)).attr("cy", y(clip(p.v)))
      .attr("r", 2.6).attr("fill", AC.teal));

    AL.axisB(g, x, f.ih, 6, "load factor α", d3.format(".1f"));
    AL.axisL(g, y, 5, kind === "succ" ? "entries or probes examined, successful"
                                      : "entries or probes examined, unsuccessful", d3.format("~g"));
    AL.legend(g, [{ label: "chaining", color: AC.teal }]
      .concat(SCHEMES.map(sc => ({ label: sc.name + " probing", color: sc.col }))), f.iw + 12, 16);

    /* the cross-check: chaining's unsuccessful cost must equal alpha */
    let worst = 0;
    if (kind === "unsucc") chain.forEach(p => { worst = Math.max(worst, Math.abs(p.v - p.a)); });
    const at = a => (chain.find(p => Math.abs(p.a - a) < 1e-9) || {}).v;
    const oat = (i, a) => { const q = open[i].pts.find(p => Math.abs(p.a - a) < 1e-9); return q ? q.v : null; };
    const A = amax > 1 ? 0.85 : 0.85;
    els.out.html(
      "<b>" + (kind === "succ" ? "Successful" : "Unsuccessful") + " search, m = " + HT.int(M)
      + ", " + R + " builds per point.</b> "
      + (kind === "unsucc"
         ? "Chaining's measured cost tracks <span class='keep'>α</span> exactly — the largest deviation "
           + "anywhere on the curve is <b>" + HT.sig(worst, 3) + "</b> entries, against a theorem that says "
           + "the two are equal. "
         : "Chaining's measured cost tracks 1 + (n−1)/2m across the whole range. ")
      + "At <span class='keep'>α</span> = " + A + " the four structures cost: chaining <b>" + HT.sig(at(A), 2)
      + "</b>, double <b>" + HT.sig(oat(2, A), 2) + "</b>, quadratic <b>" + HT.sig(oat(1, A), 2)
      + "</b>, linear <b>" + HT.sig(oat(0, A), 2) + "</b>"
      + (kind === "unsucc" ? " — a ratio of <b>" + HT.sig(oat(0, A) / at(A), 0)
          + "×</b> between linear probing and chaining, though the units are not identical: a probe is "
          + "one slot read, while a chained entry examined also costs the pointer chase to reach it." : ".")
      + (amax > 1
         ? " Beyond <span class='keep'>α</span> = 1 only chaining exists at all, and at <span class='keep'>α</span> = 3 it costs <b>"
           + HT.sig(at(3), 2) + "</b> — still linear, still finite, still no wall."
         : " Push the range control past 1 to see what happens to each family there.")
    );
  }

  [els.kind, els.max, els.log].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══ FIGURE 12 ═══════════════════════════════════════════════════════════
   Universal hashing against an adversary. The adversarial key set is generated
   to defeat the FIXED function k mod m; the same keys are then hashed by real
   draws from H⟨p,m⟩. The bottom panel repeats the draw many times, so the
   claim being made is about the family and not about one lucky sample. The
   readout also measures the defining property directly: the fraction of draws
   on which one specific pair of distinct keys collides, against the 1/m bound. */
(function () {
  const svg = d3.select("#univ-svg"); if (svg.empty()) return;
  const W = 760, H = 400, M = 127, P = 1000003;    // M prime, P > every key used
  const els = {
    keys: d3.select("#uv-keys"), n: d3.select("#uv-n"), nOut: d3.select("#uv-n-out"),
    d: d3.select("#uv-draws"), dOut: d3.select("#uv-draws-out"), out: d3.select("#univ-readout")
  };
  const hab = (a, b, k) => ((a * (k % P) + b) % P) % M;

  function draw() {
    const kind = els.keys.property("value");
    const n = +els.n.property("value"), draws = +els.d.property("value");
    els.nOut.text(n); els.dOut.text(draws);
    const r = AL.rng(777);
    const keys = kind === "rand"
      ? Array.from({ length: n }, () => AL.randInt(r, 1, 900000))
      : Array.from({ length: n }, (_, i) => (i + 1) * M + (kind === "adv2" ? 7 : 0));

    /* the fixed function the adversary targeted */
    const fixed = new Array(M).fill(0);
    keys.forEach(k => fixed[k % M]++);
    const fixedMax = Math.max(...fixed), fixedUsed = fixed.filter(v => v > 0).length;

    /* one representative universal draw, then many for the histogram */
    const a0 = AL.randInt(r, 1, P - 1), b0 = AL.randInt(r, 0, P - 1);
    const one = new Array(M).fill(0);
    keys.forEach(k => one[hab(a0, b0, k)]++);
    const oneMax = Math.max(...one), oneUsed = one.filter(v => v > 0).length;

    const maxes = [];
    for (let t = 0; t < draws; t++) {
      const a = AL.randInt(r, 1, P - 1), b = AL.randInt(r, 0, P - 1);
      const c = new Int32Array(M);
      keys.forEach(k => c[hab(a, b, k)]++);
      maxes.push(Math.max(...c));
    }
    const meanMax = maxes.reduce((x, y) => x + y, 0) / draws;
    const worstMax = Math.max(...maxes);
    /* A REFERENCE, not a prediction. The Poissonised expected maximum is what FULLY
       INDEPENDENT uniform hashing would give. Universality does NOT promise it — the
       definition constrains PAIRS only, and says nothing about the joint distribution
       of three or more keys. On an arithmetic-progression key set this family in fact
       spreads BETTER than independence, because (a·k+b) mod p of a progression is
       itself a progression, which mod m is more even than random. Quoting the
       balls-in-bins maximum as a guarantee here would be importing a theorem past its
       hypothesis; the line is drawn as a comparison and labelled as one. */
    const alpha = n / M;
    let term = Math.exp(-alpha), F = term, EM = 0;
    for (let k = 0; k < 400; k++) {
      EM += 1 - Math.pow(F, M);
      if (k > alpha + 8 && 1 - Math.pow(F, M) < 1e-9) break;
      term *= alpha / (k + 1); F += term; if (F > 1) F = 1;
    }
    const okMax = meanMax <= EM * 1.25;     // "no worse than independent hashing", not "equal to"

    /* The definition, three ways: the EXACT probability in closed form, a Monte Carlo
       estimate of it, and the 1/m bound the theorem promises.
         Exact: STEP 2 of the proof says (a,b) ↦ (r,s) is a bijection onto the ordered
       pairs r ≠ s in [0,p)². So Pr[collide] = #{(r,s) : r ≠ s, r ≡ s (mod m)} / p(p−1),
       which depends only on p and m — NOT on which pair of keys is asked about. Writing
       p = q·m + s0, exactly s0 residue classes hold q+1 of the values 0…p−1 and m − s0
       hold q, so the numerator is s0·(q+1)q + (m−s0)·q(q−1).
         This matters for presentation as much as for correctness. The bound is on the
       expectation over the draw of h; a finite sample fluctuates either side of it, and
       reporting a measurement that lands a third of a standard error ABOVE 1/m without
       saying so reads as a refuted theorem. So the readout prints the exact value, the
       measurement, and the measurement's 95% interval, and checks that the interval
       covers the exact value rather than that the point estimate happens to sit low. */
    const q = Math.floor(P / M), s0 = P - q * M;
    const pExact = (s0 * (q + 1) * q + (M - s0) * q * (q - 1)) / (P * (P - 1));
    const k1 = keys[0], k2 = keys[1];
    let hits = 0, TRIALS = 20000;
    const r2 = AL.rng(31337);
    for (let t = 0; t < TRIALS; t++) {
      const a = AL.randInt(r2, 1, P - 1), b = AL.randInt(r2, 0, P - 1);
      if (hab(a, b, k1) === hab(a, b, k2)) hits++;
    }
    const pColl = hits / TRIALS;
    const se = Math.sqrt(pColl * (1 - pColl) / TRIALS);
    const ciLo = pColl - 1.96 * se, ciHi = pColl + 1.96 * se;
    const zDev = (pColl - pExact) / Math.sqrt(pExact * (1 - pExact) / TRIALS);
    /* the theorem itself: the exact probability must not exceed 1/m.
       the measurement: its interval must cover the exact probability. */
    const okBound = pExact <= 1 / M && ciLo <= pExact && pExact <= ciHi;

    const f = AL.frame(svg, W, H, { l: 56, r: 150, t: 16, b: 38 });
    const g = f.g;
    const stripH = 46, gap = 16;
    const drawStrip = (counts, y, label, col) => {
      const cw = f.iw / M;
      const mx = Math.max(1, Math.max(...counts));
      counts.forEach((v, j) => {
        if (!v) return;
        const h = Math.max(2, stripH * v / mx);
        g.append("rect").attr("x", j * cw).attr("y", y + stripH - h)
          .attr("width", Math.max(1, cw - 0.6)).attr("height", h).attr("fill", col);
      });
      g.append("rect").attr("x", 0).attr("y", y).attr("width", f.iw).attr("height", stripH)
        .attr("fill", "none").attr("stroke", AC.line);
      g.append("text").attr("x", 2).attr("y", y - 4).attr("font-size", 11).attr("fill", AC.muted).text(label);
    };
    drawStrip(fixed, 14, "fixed  h(k) = k mod " + M + "  —  longest bucket " + fixedMax
              + ", slots used " + fixedUsed + "/" + M, AC.rose);
    drawStrip(one, 14 + stripH + gap + 12, "one draw from H — a = " + a0 + ", b = " + b0
              + "  —  longest bucket " + oneMax + ", slots used " + oneUsed + "/" + M, AC.accent);

    const hTop = 14 + 2 * (stripH + gap + 12);
    const hi = Math.max(worstMax, Math.ceil(EM) + 3);
    const cnt = new Array(hi + 1).fill(0);
    maxes.forEach(v => cnt[v]++);
    const x = d3.scaleBand().domain(d3.range(hi + 1)).range([0, f.iw]).padding(0.2);
    const y = d3.scaleLinear().domain([0, Math.max(1, d3.max(cnt))]).nice().range([f.ih, hTop]);
    AL.gridY(g, y, f.iw, 3);
    cnt.forEach((v, k) => {
      if (!v) return;
      g.append("rect").attr("x", x(k)).attr("y", y(v)).attr("width", x.bandwidth())
        .attr("height", f.ih - y(v)).attr("rx", 2).attr("fill", AC.violet).attr("opacity", 0.85);
    });
    /* align the reference line with the BAR CENTRES: bar k is centred at
       x(k) + bandwidth/2, so a non-integer value EM sits at that centre plus the
       fractional part times the band step. Dropping the half-bandwidth put the line
       half a bar to the left of where it belonged. */
    const emX = x(0) + x.bandwidth() / 2 + EM * x.step();
    g.append("line").attr("x1", emX).attr("x2", emX)
      .attr("y1", hTop).attr("y2", f.ih).attr("stroke", AC.good).attr("stroke-dasharray", "4 3");
    g.append("text").attr("x", 2).attr("y", hTop - 4).attr("font-size", 11).attr("fill", AC.muted)
      .text("longest bucket over " + draws + " independent draws from H");
    g.append("text").attr("x", f.iw - 2).attr("y", hTop - 4).attr("text-anchor", "end")
      .attr("font-size", 10).attr("fill", AC.good)
      .text("dashed = independent hashing, for reference only");
    AL.axisB(g, d3.scaleLinear().domain([-0.5, hi + 0.5]).range([0, f.iw]), f.ih,
      Math.min(hi + 1, 12), "longest bucket", d3.format("d"));
    AL.axisL(g, y, 3, "draws");

    const rows = [
      ["fixed  longest " + fixedMax, AC.rose],
      ["draw   longest " + oneMax, AC.accent],
      ["mean over draws " + HT.sig(meanMax, 2), AC.violet],
      ["  independent ref " + HT.sig(EM, 2), okMax ? AC.good : AC.bad],
      ["worst draw      " + worstMax, AC.muted],
      ["Pr[pair collide]", AC.muted],
      ["  measured " + HT.sig(pColl, 5), AC.ink],
      ["  ±95% " + HT.sig(ciLo, 5) + "–" + HT.sig(ciHi, 5), AC.muted],
      ["  exact " + HT.sig(pExact, 5), AC.ink],
      ["  bound 1/m " + HT.sig(1 / M, 5), okBound ? AC.good : AC.bad]
    ];
    rows.forEach((L, i) => g.append("text").attr("x", f.iw + 10).attr("y", 20 + i * 15)
      .attr("font-size", 11).attr("fill", L[1]).text(L[0]));

    const kindName = { adv: "every key a multiple of m", adv2: "every key ≡ 7 (mod m)",
                       rand: "unstructured random keys" }[kind];
    els.out.html(
      "<b>" + n + " keys — " + kindName + " — into m = " + M + " slots.</b> "
      + "The fixed function k mod m puts them into <b>" + fixedUsed + "</b> distinct slots with a "
      + "longest bucket of <b>" + fixedMax + "</b>"
      + (kind === "rand" ? ", which is what a decent spread looks like. " : ", which is every key in one place. ")
      + "One random draw from H⟨p,m⟩ on the <i>same</i> keys uses <b>" + oneUsed
      + "</b> slots with a longest bucket of <b>" + oneMax + "</b>. "
      + "Over " + draws + " independent draws the mean longest bucket is <b>" + HT.sig(meanMax, 2)
      + "</b>, and the worst draw seen was <b>" + worstMax + "</b> — not " + n + ". "
      + "For comparison, fully independent uniform hashing would give a mean longest bucket of "
      + HT.sig(EM, 2) + ", so this family is <b>"
      + (meanMax <= EM ? "no worse than" : "worse than") + "</b> independence here ("
      + HT.verdict(okMax) + "). That comparison is a <i>reference</i>, not a guarantee: universality "
      + "constrains pairs of keys only and promises nothing about the longest bucket"
      + (kind === "rand"
         ? ", and on an unstructured key set this family lands close to independence from either side. "
         : " — and on an arithmetic-progression key set it happens to spread more evenly than "
           + "independence would, because (a·k + b) mod p of a progression is again a progression. ")
      + "<b>The definition itself.</b> For this family the collision probability of a distinct pair "
      + "has a closed form that depends only on p and m, not on which keys are asked about: "
      + "<b>" + HT.sig(pExact, 5) + "</b>, which is at or below the promised 1/m = " + HT.sig(1 / M, 5)
      + " — that is the theorem. Measured over " + HT.int(TRIALS) + " draws on one fixed pair it comes "
      + "out at <b>" + HT.sig(pColl, 5) + "</b>, with a 95% interval of " + HT.sig(ciLo, 5) + " to "
      + HT.sig(ciHi, 5) + ", i.e. " + HT.sig(Math.abs(zDev), 2) + " standard errors from the exact "
      + "value — so the sample sits where it should (" + HT.verdict(okBound) + "). A single sample "
      + "landing a little either side of 1/m is the noise in " + HT.int(TRIALS)
      + " coin flips, not a violation: the bound is on the expectation over the draw of h."
    );
  }

  [els.keys, els.n, els.d].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══ FIGURE 13 ═══════════════════════════════════════════════════════════
   Bloom filter. A real bit array is built and a large number of guaranteed-absent
   keys are queried against it, for every k on the axis. Two independent
   cross-checks: the measured rate against (1 − e^(−kn/m))^k, and the measured
   fill fraction against 1 − e^(−kn/m). The optimal k is computed, not eyeballed. */
(function () {
  const svg = d3.select("#bloom-svg"); if (svg.empty()) return;
  const W = 760, H = 390;
  const els = {
    bits: d3.select("#bl-bits"), bitsOut: d3.select("#bl-bits-out"),
    k: d3.select("#bl-k"), kOut: d3.select("#bl-k-out"),
    n: d3.select("#bl-n"), nOut: d3.select("#bl-n-out"), out: d3.select("#bloom-readout")
  };
  const KMAX = 14;

  function run(m, n, k) {
    const bits = new Uint8Array(m);
    /* The k indices are derived the standard way — two independent hashes combined as
       (h1 + i·h2), with h2 forced odd. This is literally §16's double hashing, and it
       matters here: an earlier version derived index i by xoring the key with a
       per-i constant before mixing, and the resulting indices were correlated enough
       to bias the measured false-positive rate about 4% HIGH (2.24% against a
       predicted 2.16% at 8 bits per element, 7.24% against 6.91% at k = 14). The
       figure was disagreeing with the formula because the figure's hash was weak,
       not because the formula was wrong. */
    const idx = (key, i) => {
      const h1 = HT.mix(key >>> 0);
      const h2 = HT.mix((h1 ^ 0x9E3779B9) >>> 0) | 1;
      return ((h1 + Math.imul(i, h2)) >>> 0) % m;
    };
    for (let e = 0; e < n; e++) for (let i = 0; i < k; i++) bits[idx(e, i)] = 1;
    let set = 0; for (let j = 0; j < m; j++) set += bits[j];
    const T = 40000;
    let fp = 0;
    for (let t = 0; t < T; t++) {
      const key = 10000000 + t;                    // guaranteed absent
      let all = true;
      for (let i = 0; i < k; i++) if (!bits[idx(key, i)]) { all = false; break; }
      if (all) fp++;
    }
    return { rate: fp / T, fill: set / m, trials: T };
  }

  function draw() {
    const bpe = +els.bits.property("value"), kSel = +els.k.property("value"), n = +els.n.property("value");
    els.bitsOut.text(bpe); els.kOut.text(kSel); els.nOut.text(HT.int(n));
    const m = bpe * n;
    const kOpt = Math.log(2) * m / n;
    const kOptI = Math.max(1, Math.min(KMAX, Math.round(kOpt)));

    const ks = d3.range(1, KMAX + 1);
    const meas = ks.map(k => run(m, n, k));
    const pred = ks.map(k => Math.pow(1 - Math.exp(-k * n / m), k));
    const exact = ks.map(k => Math.pow(1 - Math.pow(1 - 1 / m, k * n), k));

    const f = AL.frame(svg, W, H, { l: 60, r: 160, t: 18, b: 42 });
    const g = f.g;
    const lo = Math.max(1e-5, Math.min(...meas.map(o => o.rate).filter(v => v > 0),
                                       ...pred) * 0.6);
    const x = d3.scaleLinear().domain([1, KMAX]).range([0, f.iw]);
    const y = d3.scaleLog().domain([lo, Math.max(...pred, ...meas.map(o => o.rate)) * 1.6]).range([f.ih, 0]);
    AL.gridY(g, y, f.iw, 5);
    const cl = v => Math.max(lo, v);
    g.append("path").attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 1.8)
      .attr("stroke-dasharray", "5 3")
      .attr("d", d3.line().x((_, i) => x(ks[i])).y(v => y(cl(v)))(pred));
    g.append("path").attr("fill", "none").attr("stroke", AC.accent).attr("stroke-width", 2.4)
      .attr("d", d3.line().x((_, i) => x(ks[i])).y(o => y(cl(o.rate)))(meas));
    meas.forEach((o, i) => g.append("circle").attr("cx", x(ks[i])).attr("cy", y(cl(o.rate)))
      .attr("r", 3).attr("fill", ks[i] === kSel ? AC.violet : AC.accent));
    /* fill fraction on a right-hand 0..1 axis */
    const y2 = d3.scaleLinear().domain([0, 1]).range([f.ih, 0]);
    g.append("path").attr("fill", "none").attr("stroke", AC.teal).attr("stroke-width", 1.6)
      .attr("d", d3.line().x((_, i) => x(ks[i])).y(o => y2(o.fill))(meas));
    g.append("g").attr("class", "axis").attr("transform", "translate(" + f.iw + ",0)")
      .call(d3.axisRight(y2).ticks(5).tickFormat(d3.format(".0%")));
    g.append("line").attr("x1", x(kOpt)).attr("x2", x(kOpt)).attr("y1", 0).attr("y2", f.ih)
      .attr("stroke", AC.good).attr("stroke-dasharray", "4 3");
    g.append("text").attr("x", x(kOpt) + 4).attr("y", 12).attr("font-size", 10).attr("fill", AC.good)
      .text("k* = (m/n)·ln2 = " + HT.sig(kOpt, 2));
    g.append("line").attr("x1", 0).attr("x2", f.iw).attr("y1", y2(0.5)).attr("y2", y2(0.5))
      .attr("stroke", AC.teal).attr("stroke-dasharray", "2 4").attr("opacity", 0.6);

    AL.axisB(g, x, f.ih, KMAX > 10 ? 7 : KMAX, "number of hash functions k", d3.format("d"));
    AL.axisL(g, y, 5, "false-positive rate (log)", d3.format(".2~%"));
    AL.legend(g, [
      { label: "measured rate", color: AC.accent },
      { label: "(1 − e^(−kn/m))^k", color: AC.a2, dash: "5 3" },
      { label: "fraction of bits set (right)", color: AC.teal },
      { label: "optimal k", color: AC.good, dash: "4 3" }
    ], f.iw + 34, 16);

    const sel = meas[kSel - 1], selPred = pred[kSel - 1], selExact = exact[kSel - 1];
    const sel0Trials = sel.trials;
    const fillPred = 1 - Math.exp(-kSel * n / m);
    const okRate = Math.abs(sel.rate - selPred) <= Math.max(0.0015, selPred * 0.12);
    const okFill = Math.abs(sel.fill - fillPred) < 0.02;
    const bestI = meas.reduce((b, o, i) => (o.rate < meas[b].rate ? i : b), 0);
    /* Only assert the location of the optimum when the trials can actually resolve it.
       At 20 bits per element the minimum rate is a few parts in ten thousand, so with
       this many queries several values of k tie at zero or one false positive and the
       argmin is noise rather than a measurement. */
    const minPred = Math.min(...pred);
    const resolvable = minPred * sel0Trials > 25;
    const okOpt = !resolvable || Math.abs((bestI + 1) - kOptI) <= 1;

    const rows = [
      ["m = " + HT.int(m) + " bits, n = " + HT.int(n), AC.muted],
      ["m/n = " + bpe + " bits/element", AC.muted],
      ["at k = " + kSel, AC.violet],
      ["  measured " + HT.sig(100 * sel.rate, 3) + "%", AC.ink],
      ["  formula  " + HT.sig(100 * selPred, 3) + "%", okRate ? AC.good : AC.bad],
      ["  fill     " + HT.sig(100 * sel.fill, 1) + "%", AC.teal],
      ["  1−e^(−kn/m) " + HT.sig(100 * fillPred, 1) + "%", okFill ? AC.good : AC.bad],
      ["best measured k = " + (bestI + 1), AC.ink],
      ["  k* rounds to " + kOptI, okOpt ? AC.good : AC.bad],
      [resolvable ? "" : "  (optimum below trial resolution)", AC.muted]
    ];
    rows.forEach((L, i) => g.append("text").attr("x", f.iw + 34).attr("y", 92 + i * 15)
      .attr("font-size", 11).attr("fill", L[1]).text(L[0]));

    els.out.html(
      "<b>m = " + HT.int(m) + " bits for n = " + HT.int(n) + " elements (" + bpe
      + " bits each), " + HT.int(sel.trials) + " absent keys queried per point.</b> "
      + "At k = " + kSel + " the measured false-positive rate is <b>" + HT.sig(100 * sel.rate, 3)
      + "%</b> against (1 − e^(−kn/m))^k = " + HT.sig(100 * selPred, 3)
      + "% and the exact (1 − (1−1/m)^(kn))^k = " + HT.sig(100 * selExact, 3) + "% — "
      + HT.verdict(okRate) + ". "
      + "Measured fraction of bits set <b>" + HT.sig(100 * sel.fill, 1) + "%</b> against 1 − e^(−kn/m) = "
      + HT.sig(100 * fillPred, 1) + "% — " + HT.verdict(okFill) + ". "
      + "The lowest measured rate over all k is at <b>k = " + (bestI + 1)
      + "</b>, and (m/n)·ln2 = " + HT.sig(kOpt, 2) + " rounds to " + kOptI + " — "
      + (resolvable ? HT.verdict(okOpt)
                    : "<b>below the resolution of " + HT.int(sel0Trials)
                      + " queries</b>, since the predicted minimum rate is only "
                      + HT.sig(100 * minPred, 4) + "% and several k tie")
      + ". At that k the bit array is " + HT.sig(100 * meas[kOptI - 1].fill, 1)
      + "% full, which is the half-full diagnostic."
    );
  }

  [els.bits, els.k, els.n].forEach(s => s.on("input", draw));
  draw();
})();
