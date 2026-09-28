/* dl-in-practice.viz.js — the visualizations on deep-learning/dl-in-practice.html.
   Loaded after ../data.js → ../notes.js → dl-viz.js (DC palette, DL helpers).
   Each block is an IIFE that exits quietly if its <svg> is not on the page.

     1  #bc-svg     broadcasting checker
     2  #tape-svg   define-by-run tape: forward records, backward replays
     3  #pipe-svg   input-pipeline timeline, GPU idle vs prefetch
     4  #lrf-svg    LR finder on a simulated run, with suggestions
     5  #oc-svg     one-cycle schedule (LR and momentum)
     6  #diag-svg   learning-curve diagnoser                                  */

(function () {
  "use strict";
  const $ = id => document.getElementById(id);
  const mono = "SF Mono, Menlo, monospace";
  const sup = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
  const pow10 = e => "10" + String(e).split("").map(c => sup[c] || c).join("");
  const fe = (x, d) => {           // 3.2e−4 style
    if (!isFinite(x)) return "—";
    if (x === 0) return "0";
    const s = x.toExponential(d === undefined ? 1 : d).split("e");
    return s[0] + "e" + s[1].replace("+", "").replace("-", "−");
  };

  /* ═══ 1 · broadcasting checker ═══════════════════════════════════════ */
  (function () {
    const svg = d3.select("#bc-svg");
    if (svg.empty()) return;
    const W = 760, H = 230;
    const parse = s => {
      const t = s.replace(/[()\[\]]/g, "").split(/[\s,×x*]+/).filter(Boolean);
      if (!t.length) return null;
      const v = t.map(Number);
      return v.every(n => Number.isInteger(n) && n >= 0) ? v : null;
    };
    function broadcast(a, b) {
      const n = Math.max(a.length, b.length);
      const A = Array(n - a.length).fill(null).concat(a), B = Array(n - b.length).fill(null).concat(b);
      const cols = [], out = [];
      let ok = true;
      for (let i = 0; i < n; i++) {
        const x = A[i], y = B[i];
        const xa = x === null ? 1 : x, yb = y === null ? 1 : y;
        let st, r;
        if (xa === yb) { st = "eq"; r = xa; }
        else if (xa === 1) { st = "A"; r = yb; }
        else if (yb === 1) { st = "B"; r = xa; }
        else { st = "bad"; r = null; ok = false; }
        cols.push({ x: x, y: y, st: st, r: r });
        out.push(r);
      }
      return { cols: cols, ok: ok, out: out };
    }
    function draw() {
      svg.selectAll("*").remove();
      const a = parse($("bc-a").value), b = parse($("bc-b").value);
      const ro = $("bc-readout");
      if (!a || !b) {
        svg.append("text").attr("x", W / 2).attr("y", H / 2).attr("text-anchor", "middle")
          .attr("fill", DC.bad).attr("font-size", 13).text("could not parse a shape — use e.g. 64,3,32,32");
        ro.innerHTML = "";
        return;
      }
      const R = broadcast(a, b), n = R.cols.length;
      const bw = Math.min(90, (W - 200) / n - 10), gap = 10;
      const x0 = W - 30 - n * (bw + gap);
      const rows = [{ k: "A", y: 30, get: c => c.x, str: c => c.st === "A" }, { k: "B", y: 90, get: c => c.y, str: c => c.st === "B" },
        { k: "A ⊕ B", y: 165, get: c => c.r, str: () => false }];
      rows.forEach(row => {
        svg.append("text").attr("x", x0 - 16).attr("y", row.y + 25).attr("text-anchor", "end")
          .attr("fill", DC.ink).attr("font-size", 13).attr("font-weight", 600).text(row.k);
        R.cols.forEach((c, i) => {
          const v = row.get(c), x = x0 + i * (bw + gap);
          const missing = v === null && row.k !== "A ⊕ B";
          const bad = c.st === "bad";
          const stretched = row.str(c);
          let stroke = DC.line, fill = DC.panel2, txt = DC.ink;
          if (missing) { stroke = DC.muted; fill = "none"; txt = DC.muted; }
          if (stretched) { stroke = DC.a2; fill = "rgba(255,180,84,.15)"; }
          if (bad) { stroke = DC.bad; fill = "rgba(248,113,113,.15)"; txt = DC.bad; }
          if (row.k === "A ⊕ B" && !bad) { stroke = DC.accent; fill = "rgba(91,156,255,.12)"; }
          svg.append("rect").attr("x", x).attr("y", row.y).attr("width", bw).attr("height", 38).attr("rx", 6)
            .attr("fill", fill).attr("stroke", stroke).attr("stroke-width", 1.4)
            .attr("stroke-dasharray", missing ? "4 3" : null);
          svg.append("text").attr("x", x + bw / 2).attr("y", row.y + 24).attr("text-anchor", "middle")
            .attr("font-family", mono).attr("font-size", 14).attr("fill", txt)
            .text(v === null ? (row.k === "A ⊕ B" ? "✕" : "(1)") : v);
          if (stretched) svg.append("text").attr("x", x + bw / 2).attr("y", row.y + 50).attr("text-anchor", "middle")
            .attr("font-size", 10).attr("fill", DC.a2).text("stretched → " + c.r);
          if (row.k === "A ⊕ B") svg.append("text").attr("x", x + bw / 2).attr("y", row.y - 8).attr("text-anchor", "middle")
            .attr("font-size", 10).attr("fill", DC.muted).text("axis " + (i - n));
        });
      });
      svg.append("text").attr("x", 14).attr("y", 20).attr("font-size", 11).attr("fill", DC.muted)
        .text("right-aligned: trailing axes compared first");
      const prod = v => v.reduce((s, q) => s * q, 1);
      if (R.ok) {
        const nA = prod(a), nB = prod(b), nO = prod(R.out);
        let msg = `result shape <b>(${R.out.join(", ")})</b> · ${nO.toLocaleString()} elements from ${nA.toLocaleString()} and ${nB.toLocaleString()} — no copy is made`;
        if (nO > Math.max(nA, nB)) msg += ` · <b>the result is larger than both inputs</b> — intended (an outer operation) or a silent bug?`;
        ro.innerHTML = msg;
      } else {
        const i = R.cols.findIndex(c => c.st === "bad");
        ro.innerHTML = `<b>error</b>: axis ${i - n} has sizes ${R.cols[i].x} and ${R.cols[i].y}; neither is 1 · add a size-1 axis (keepdim=True, x[:, None]) where you meant to broadcast`;
      }
    }
    $("bc-preset").addEventListener("change", e => {
      const p = e.target.value.split("|"); $("bc-a").value = p[0]; $("bc-b").value = p[1]; draw();
    });
    $("bc-a").addEventListener("input", draw);
    $("bc-b").addEventListener("input", draw);
    draw();
  })();

  /* ═══ 2 · autograd tape ═════════════════════════════════════════════ */
  (function () {
    const svg = d3.select("#tape-svg");
    if (svg.empty()) return;
    const X = 2, B = 1, T = 5;
    let step = 0;
    const OPS = [
      { id: "u", op: "u = w·x", fn: "MulBackward0", cx: 200 },
      { id: "z", op: "z = u + b", fn: "AddBackward0", cx: 310 },
      { id: "y", op: "y = relu(z)", fn: "ReluBackward0", cx: 420 },
      { id: "d", op: "d = y − t", fn: "SubBackward0", cx: 530 },
      { id: "L", op: "L = d²", fn: "PowBackward0", cx: 640 }
    ];
    function compute(w) {
      const u = w * X, z = u + B, y = Math.max(0, z), d = y - T, L = d * d;
      const gL = 1, gd = 2 * d, gy = gd, gz = gy * (z > 0 ? 1 : 0), gu = gz, gb = gz, gw = gu * X;
      return { v: { w: w, x: X, b: B, t: T, u: u, z: z, y: y, d: d, L: L },
        g: { L: gL, d: gd, y: gy, z: gz, u: gu, b: gb, w: gw } };
    }
    const fmt = v => (Math.round(v * 100) / 100).toString().replace("-", "−");
    function draw() {
      svg.selectAll("*").remove();
      const w = +$("tp-w").value, mode = $("tp-mode").value;
      $("tp-wv").textContent = w.toFixed(1);
      const C = compute(w);
      // which ops are recorded
      const recorded = OPS.map((o, i) => mode === "normal" ? true : mode === "detach" ? i < 3 : false);
      const nRec = recorded.filter(Boolean).length;
      const canBack = mode === "normal";
      const fwdDone = Math.min(step, 5);
      const bwdDone = canBack ? Math.max(0, Math.min(step - 5, 5)) : 0;
      const Y = 95;
      // leaves
      const leaves = [{ k: "w", x: 70, y: 40, grad: true }, { k: "x", x: 70, y: 150, grad: false }, { k: "b", x: 200, y: 175, grad: true }, { k: "t", x: 420, y: 175, grad: false }];
      const edge = (x1, y1, x2, y2, col) => DL.arrow(svg, x1, y1, x2, y2, { color: col || DC.line, w: 1.3, head: 5 });
      edge(95, 48, 172, Y - 6); edge(95, 142, 172, Y + 6);
      edge(200, 160, 290, Y + 16); edge(420, 160, 510, Y + 16);
      for (let i = 0; i < 4; i++) edge(OPS[i].cx + 30, Y, OPS[i + 1].cx - 32, Y);
      leaves.forEach(l => {
        const shown = true;
        const gOn = l.grad && canBack && bwdDone === 5;
        svg.append("rect").attr("x", l.x - 26).attr("y", l.y - 18).attr("width", 52).attr("height", 36).attr("rx", 18)
          .attr("fill", DC.panel2).attr("stroke", l.grad && mode !== "nograd" ? DC.accent : DC.line);
        svg.append("text").attr("x", l.x).attr("y", l.y - 2).attr("text-anchor", "middle").attr("font-size", 11)
          .attr("fill", DC.ink).text(l.k + " = " + fmt(C.v[l.k]));
        svg.append("text").attr("x", l.x).attr("y", l.y + 12).attr("text-anchor", "middle").attr("font-size", 9.5)
          .attr("fill", gOn ? DC.a2 : DC.muted).text(l.grad ? (gOn ? "grad " + fmt(C.g[l.k]) : "requires_grad") : "constant");
        void shown;
      });
      OPS.forEach((o, i) => {
        const done = i < fwdDone;
        const gIdx = 4 - i;                               // backward order
        const gOn = canBack && gIdx < bwdDone;
        svg.append("rect").attr("x", o.cx - 30).attr("y", Y - 22).attr("width", 60).attr("height", 44).attr("rx", 7)
          .attr("fill", done ? DC.panel2 : "none").attr("stroke", gOn ? DC.a2 : (done ? (recorded[i] ? DC.accent : DC.muted) : DC.line))
          .attr("stroke-dasharray", done ? null : "3 3");
        svg.append("text").attr("x", o.cx).attr("y", Y - 28).attr("text-anchor", "middle").attr("font-size", 10)
          .attr("fill", DC.muted).text(o.op);
        svg.append("text").attr("x", o.cx).attr("y", Y - 3).attr("text-anchor", "middle").attr("font-size", 12).attr("font-family", mono)
          .attr("fill", done ? DC.ink : DC.muted).text(done ? fmt(C.v[o.id]) : "·");
        svg.append("text").attr("x", o.cx).attr("y", Y + 14).attr("text-anchor", "middle").attr("font-size", 10).attr("font-family", mono)
          .attr("fill", DC.a2).text(gOn ? "∂L/∂" + o.id + " " + fmt(C.g[o.id]) : "");
      });
      // tape strip
      const ty = 230;
      svg.append("text").attr("x", 20).attr("y", ty - 16).attr("font-size", 11).attr("fill", DC.ink).attr("font-weight", 600)
        .text("the tape (grad_fn chain), in recording order →");
      const recNow = OPS.map((o, i) => recorded[i] && i < fwdDone);
      let k = 0;
      OPS.forEach((o, i) => {
        if (!recNow[i]) return;
        const x = 20 + k * 142; k++;
        const consumed = canBack && (4 - i) < bwdDone;
        const current = canBack && (4 - i) === bwdDone - 1;
        svg.append("rect").attr("x", x).attr("y", ty).attr("width", 132).attr("height", 34).attr("rx", 5)
          .attr("fill", current ? "rgba(255,180,84,.18)" : DC.panel2).attr("stroke", consumed ? DC.a2 : DC.accent)
          .attr("opacity", consumed && !current ? 0.55 : 1);
        svg.append("text").attr("x", x + 66).attr("y", ty + 21).attr("text-anchor", "middle").attr("font-family", mono)
          .attr("font-size", 11).attr("fill", DC.ink).text(o.fn);
      });
      if (k === 0) svg.append("text").attr("x", 20).attr("y", ty + 21).attr("font-size", 11).attr("fill", DC.muted)
        .text(fwdDone === 0 ? "(empty — step forward to record operations)" : "(nothing recorded)");
      // readout
      const ro = $("tape-readout");
      if (step === 0) ro.innerHTML = "press <b>step</b> to run the forward pass one operation at a time";
      else if (step <= 5) ro.innerHTML = `forward: ${OPS[step - 1].op} = <b>${fmt(C.v[OPS[step - 1].id])}</b> · ${recorded[step - 1] ? "recorded as " + OPS[step - 1].fn : "<b>not recorded</b>"} · tape length ${recNow.filter(Boolean).length}`;
      else if (!canBack) ro.innerHTML = `<b>L.backward() → RuntimeError</b>: element 0 of tensors does not require grad and does not have a grad_fn · ${mode === "detach" ? "detach() cut the path from L back to w and b (" + nRec + " ops recorded, none reach L)" : "no_grad() recorded nothing"} — w.grad stays None`;
      else if (bwdDone < 5) {
        const o = OPS[4 - (bwdDone - 1)];
        let extra = "";
        if (o.id === "z" && C.v.z <= 0) extra = " · <b>ReLU is off (z ≤ 0): gradient 0 from here down</b>";
        ro.innerHTML = `backward reaches ${o.id}: ∂L/∂${o.id} = <b>${fmt(C.g[o.id])}</b> · next, ${o.fn} multiplies it by its local derivative and passes it to its inputs${extra}`;
      } else ro.innerHTML = `done: <b>w.grad = ${fmt(C.g.w)}</b>, <b>b.grad = ${fmt(C.g.b)}</b> (x and t are constants: no grad) · the tape is freed; a second backward() would need retain_graph=True`;
    }
    $("tp-step").addEventListener("click", () => { step = Math.min(10, step + 1); draw(); });
    $("tp-all").addEventListener("click", () => { step = 10; draw(); });
    $("tp-reset").addEventListener("click", () => { step = 0; draw(); });
    $("tp-w").addEventListener("input", draw);
    $("tp-mode").addEventListener("change", () => { step = 0; draw(); });
    draw();
  })();

  /* ═══ 3 · input pipeline timeline ═══════════════════════════════════ */
  (function () {
    const svg = d3.select("#pipe-svg");
    if (svg.empty()) return;
    const W = 760, H = 300;
    function simulate(mode, tl, tg, Wk, pf, N) {
      const load = [], gpu = [];
      if (mode === "sync") {
        let t = 0;
        for (let i = 0; i < N; i++) {
          load.push({ i: i, w: 0, s: t, e: t + tl }); t += tl;
          gpu.push({ i: i, s: t, e: t + tg }); t += tg;
        }
        return { load: load, gpu: gpu, lanes: 1 };
      }
      const Q = pf * Wk, wFree = Array(Wk).fill(0), gStart = [];
      let gEnd = 0;
      for (let i = 0; i < N; i++) {
        const w = i % Wk;                                  // round-robin, as DataLoader does
        const room = i - Q >= 0 ? gStart[i - Q] : 0;       // queue slot frees when batch i−Q is taken
        const s = Math.max(wFree[w], room), e = s + tl;
        wFree[w] = e;
        load.push({ i: i, w: w, s: s, e: e });
        const gs = Math.max(e, gEnd);
        gStart.push(gs); gEnd = gs + tg;
        gpu.push({ i: i, s: gs, e: gEnd });
      }
      return { load: load, gpu: gpu, lanes: Wk };
    }
    function draw() {
      const mode = $("pp-mode").value, tl = +$("pp-load").value, tg = +$("pp-gpu").value,
        Wk = +$("pp-w").value, pf = +$("pp-pf").value;
      $("pp-loadv").textContent = tl + " ms"; $("pp-gpuv").textContent = tg + " ms";
      $("pp-wv").textContent = Wk; $("pp-pfv").textContent = pf;
      $("pp-w").disabled = $("pp-pf").disabled = mode === "sync";
      const S = simulate(mode, tl, tg, Wk, pf, 60);
      const horizon = 480;
      const F = DL.frame(svg, W, H, { l: 96, r: 16, t: 16, b: 34 });
      const g = F.g;
      const x = d3.scaleLinear().domain([0, horizon]).range([0, F.iw]);
      const lanes = S.lanes + 1, lh = Math.min(34, (F.ih - 10) / lanes), gap = 6;
      const laneY = k => k * (lh + gap);
      DL.axisB(g, x, F.ih, 8, "time (ms)");
      const clipId = "pp-clip";
      svg.append("defs").append("clipPath").attr("id", clipId).append("rect").attr("width", F.iw).attr("height", F.ih);
      const body = g.append("g").attr("clip-path", `url(#${clipId})`);
      for (let k = 0; k < S.lanes; k++) {
        g.append("text").attr("x", -8).attr("y", laneY(k) + lh / 2 + 4).attr("text-anchor", "end").attr("font-size", 10.5)
          .attr("fill", DC.muted).text(mode === "sync" ? "main: load" : "worker " + k);
      }
      const gy = laneY(S.lanes);
      g.append("text").attr("x", -8).attr("y", gy + lh / 2 + 4).attr("text-anchor", "end").attr("font-size", 11)
        .attr("fill", DC.ink).attr("font-weight", 600).text("GPU");
      S.load.filter(b => b.s < horizon).forEach(b => {
        body.append("rect").attr("x", x(b.s)).attr("y", laneY(b.w)).attr("width", Math.max(1, x(b.e) - x(b.s) - 1)).attr("height", lh)
          .attr("rx", 3).attr("fill", DC.teal).attr("fill-opacity", 0.35).attr("stroke", DC.teal).attr("stroke-width", 0.8);
        if (x(b.e) - x(b.s) > 16) body.append("text").attr("x", x(b.s) + 4).attr("y", laneY(b.w) + lh / 2 + 4)
          .attr("font-size", 9.5).attr("fill", DC.ink).text(b.i);
      });
      body.append("rect").attr("x", 0).attr("y", gy).attr("width", F.iw).attr("height", lh).attr("fill", DC.bad).attr("fill-opacity", 0.28);
      S.gpu.filter(b => b.s < horizon).forEach(b => {
        body.append("rect").attr("x", x(b.s)).attr("y", gy).attr("width", Math.max(1, x(b.e) - x(b.s) - 0.5)).attr("height", lh)
          .attr("fill", DC.accent).attr("fill-opacity", 0.85);
        if (x(b.e) - x(b.s) > 14) body.append("text").attr("x", x(b.s) + 3).attr("y", gy + lh / 2 + 4)
          .attr("font-size", 9.5).attr("fill", DC.bg).text(b.i);
      });
      // steady-state stats from the last 30 batches
      const a = S.gpu[29], z = S.gpu[59];
      const per = (z.e - a.e) / 30, util = tg / per;
      const need = Math.ceil(tl / tg);
      $("pipe-readout").innerHTML =
        `steady state: <b>${per.toFixed(1)} ms / batch</b> · ${(1000 / per).toFixed(1)} batches/s · GPU busy <b>${Math.round(util * 100)} %</b>` +
        (mode === "sync" ? ` · every load stalls the GPU (t_load + t_gpu = ${tl + tg} ms)` :
          (util > 0.98 ? ` · GPU-bound: the input pipeline keeps up (≥ ${need} worker${need > 1 ? "s" : ""} needed)` :
            ` · input-bound: need ≈ ⌈t_load / t_gpu⌉ = ${need} workers`)) +
        ` · <span style="color:${DC.bad}">red</span> = GPU idle`;
    }
    ["pp-mode", "pp-load", "pp-gpu", "pp-w", "pp-pf"].forEach(id => $(id).addEventListener("input", draw));
    draw();
  })();

  /* ═══ 4 · LR finder ════════════════════════════════════════════════ */
  (function () {
    const svg = d3.select("#lrf-svg");
    if (svg.empty()) return;
    const W = 760, H = 340, N = 100, LO = -7, HI = 1;
    let seed = 7;
    const sigm = z => 1 / (1 + Math.exp(-z));
    const PROFILE = {
      fresh: { L0: 2.35, A: 1.75, c: -2.6, s: 0.45, d: -0.4, k: 3.2, B: 0.25 },
      unfrozen: { L0: 0.42, A: 0.07, c: -5.4, s: 0.5, d: -3.4, k: 2.6, B: 0.06 }
    };
    function run(state, beta, noise) {
      const P = PROFILE[state], r = DL.rng(seed), pts = [];
      let avg = 0, best = Infinity;
      for (let k = 0; k < N; k++) {
        const le = LO + (HI - LO) * k / (N - 1), lr = Math.pow(10, le);
        const base = P.L0 - P.A * sigm((le - P.c) / P.s) + P.B * (Math.exp((le - P.d) * P.k) - 0);
        const raw = Math.max(0.001, base * (1 + noise * DL.randn(r)));
        avg = beta * avg + (1 - beta) * raw;
        const sm = avg / (1 - Math.pow(beta, k + 1));
        pts.push({ k: k, lr: lr, le: le, raw: raw, s: sm });
        if (sm > 4 * best || !isFinite(sm)) break;
        best = Math.min(best, sm);
      }
      return pts;
    }
    function suggest(pts) {
      // exclude the final (diverged) point
      const P = pts.length > 3 ? pts.slice(0, -1) : pts;
      let iMin = 0; P.forEach((p, i) => { if (p.s < P[iMin].s) iMin = i; });
      let iSteep = 0, best = Infinity;
      for (let i = 8; i < iMin; i++) {                        // slope over a 4-step window, skipping warm-up of the EWMA
        const g = (P[i].s - P[i - 4].s) / (P[i].le - P[i - 4].le);
        if (g < best) { best = g; iSteep = i; }
      }
      // valley: longest decreasing subsequence, then a point partway down it
      const n = P.length, lds = Array(n).fill(1), st = P.map((_, i) => i);
      for (let i = 1; i < n; i++) for (let j = 0; j < i; j++)
        if (P[i].s < P[j].s && lds[j] + 1 > lds[i]) { lds[i] = lds[j] + 1; st[i] = st[j]; }
      let end = 0; lds.forEach((v, i) => { if (v > lds[end]) end = i; });
      const s0 = st[end], sec = (end - s0) / 3;
      const iVal = Math.min(n - 1, s0 + Math.floor(sec) + Math.floor(sec / 2));
      return { min: P[iMin].lr, min10: P[iMin].lr / 10, steep: P[iSteep].lr, valley: P[iVal].lr, iMin: iMin };
    }
    function draw() {
      const state = $("lf-state").value, beta = +$("lf-beta").value, noise = +$("lf-noise").value, showRaw = $("lf-raw").checked;
      $("lf-betav").textContent = beta.toFixed(2); $("lf-noisev").textContent = noise.toFixed(2);
      const pts = run(state, beta, noise), S = suggest(pts);
      const F = DL.frame(svg, W, H, { l: 50, r: 150, t: 16, b: 36 }), g = F.g;
      const x = d3.scaleLog().domain([Math.pow(10, LO), Math.pow(10, HI)]).range([0, F.iw]);
      const ymax = state === "fresh" ? 3.2 : 1.0;
      const y = d3.scaleLinear().domain([0, ymax]).range([F.ih, 0]).clamp(true);
      DL.gridY(g, y, F.iw, 5);
      g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`)
        .call(d3.axisBottom(x).tickValues(d3.range(LO, HI + 1).map(e => Math.pow(10, e))).tickFormat(v => pow10(Math.round(Math.log10(v)))));
      g.append("text").attr("x", F.iw).attr("y", F.ih + 32).attr("text-anchor", "end").attr("font-size", 11).attr("fill", DC.muted).text("learning rate (log)");
      DL.axisL(g, y, 5, "loss");
      if (showRaw) g.selectAll(".rawpt").data(pts).join("circle").attr("cx", d => x(d.lr)).attr("cy", d => y(d.raw))
        .attr("r", 2).attr("fill", DC.muted).attr("fill-opacity", 0.45);
      g.append("path").datum(pts).attr("fill", "none").attr("stroke", DC.accent).attr("stroke-width", 2.2)
        .attr("d", d3.line().x(d => x(d.lr)).y(d => y(d.s)));
      const last = pts[pts.length - 1];
      if (last.le < HI - 1e-9) {
        g.append("text").attr("x", x(last.lr) + 4).attr("y", 14).attr("font-size", 10).attr("fill", DC.bad).text("stopped: diverged");
      }
      const marks = [
        { k: "minimum", v: S.min, c: DC.bad, dash: "2 3" },
        { k: "minimum / 10", v: S.min10, c: DC.good, dash: null },
        { k: "steepest", v: S.steep, c: DC.violet, dash: "6 3" },
        { k: "valley", v: S.valley, c: DC.a2, dash: null }
      ];
      marks.forEach((m, i) => {
        if (!(m.v > 0)) return;
        g.append("line").attr("x1", x(m.v)).attr("x2", x(m.v)).attr("y1", 0).attr("y2", F.ih)
          .attr("stroke", m.c).attr("stroke-width", 1.6).attr("stroke-dasharray", m.dash);
        const lg = g.append("g").attr("transform", `translate(${F.iw + 14},${20 + i * 40})`);
        lg.append("line").attr("x1", 0).attr("x2", 16).attr("y1", 0).attr("y2", 0).attr("stroke", m.c).attr("stroke-width", 2).attr("stroke-dasharray", m.dash);
        lg.append("text").attr("x", 22).attr("y", 4).attr("font-size", 11).attr("fill", DC.ink).text(m.k);
        lg.append("text").attr("x", 22).attr("y", 18).attr("font-size", 11).attr("font-family", mono).attr("fill", m.c).text(fe(m.v));
      });
      const spread = Math.log10(Math.max(S.min10, S.steep, S.valley) / Math.min(S.min10, S.steep, S.valley));
      $("lrf-readout").innerHTML = `run stopped after <b>${pts.length}</b> of ${N} steps · minimum at ${fe(S.min)} · min/10 ${fe(S.min10)}, steepest ${fe(S.steep)}, valley ${fe(S.valley)} — the three "safe" suggestions span <b>${spread.toFixed(2)} decades</b>` +
        (beta > 0 ? ` · smoothing lags the curve right by ≈ β/(1−β) = ${(beta / (1 - beta)).toFixed(0)} steps ≈ ${(beta / (1 - beta) * (HI - LO) / (N - 1)).toFixed(2)} decades` : "") +
        (state === "unfrozen" ? " · pretrained curve: no real descent — the steepest point is noise; take a rate well left of the rise" :
          " · pick on the descending slope, well left of the minimum");
    }
    ["lf-state", "lf-beta", "lf-noise", "lf-raw"].forEach(id => $(id).addEventListener("input", draw));
    $("lf-rerun").addEventListener("click", () => { seed = (seed * 1103515245 + 12345) % 2147483647; draw(); });
    draw();
  })();

  /* ═══ 5 · one-cycle schedule ═══════════════════════════════════════ */
  (function () {
    const svg = d3.select("#oc-svg");
    if (svg.empty()) return;
    const W = 760, H = 380, T = 1000;
    const ann = (kind, a, b, p) => kind === "cos" ? b + (a - b) * (1 + Math.cos(Math.PI * p)) / 2 : a + (b - a) * p;
    function draw() {
      const le = +$("oc-max").value, pct = +$("oc-pct").value, div = +$("oc-div").value, fdiv = +$("oc-fdiv").value,
        kind = $("oc-ann").value, logY = $("oc-log").checked;
      let mb = +$("oc-mb").value, mm = +$("oc-mm").value;
      if (mb > mm) { mb = mm; }
      const maxLr = Math.pow(10, le), lr0 = maxLr / div, lrMin = lr0 / fdiv;
      $("oc-maxv").textContent = fe(maxLr, 1).replace(/^1\.0e/, "1e");
      $("oc-pctv").textContent = pct.toFixed(2); $("oc-mbv").textContent = mb.toFixed(2); $("oc-mmv").textContent = mm.toFixed(2);
      const s1 = pct * T - 1;
      const data = d3.range(T).map(t => {
        let lr, m;
        if (t <= s1) { const p = s1 > 0 ? t / s1 : 1; lr = ann(kind, lr0, maxLr, p); m = ann(kind, mm, mb, p); }
        else { const p = (t - s1) / (T - 1 - s1); lr = ann(kind, maxLr, lrMin, p); m = ann(kind, mb, mm, p); }
        return { t: t, lr: lr, m: m };
      });
      svg.selectAll("*").remove();
      const L = 70, R = 20, iw = W - L - R, h1 = 190, h2 = 110, top1 = 16, top2 = 16 + h1 + 36;
      const x = d3.scaleLinear().domain([0, T]).range([0, iw]);
      const g1 = svg.append("g").attr("transform", `translate(${L},${top1})`);
      const y1 = logY ? d3.scaleLog().domain([Math.max(lrMin, 1e-12), maxLr * 1.3]).range([h1, 0])
        : d3.scaleLinear().domain([0, maxLr * 1.08]).range([h1, 0]);
      DL.gridY(g1, y1, iw, 4);
      g1.append("g").attr("class", "axis").call(d3.axisLeft(y1).ticks(4, logY ? "~e" : "~e"));
      g1.append("g").attr("class", "axis").attr("transform", `translate(0,${h1})`).call(d3.axisBottom(x).ticks(10).tickFormat(() => ""));
      g1.append("text").attr("x", 0).attr("y", -4).attr("font-size", 11).attr("fill", DC.muted).text("learning rate");
      g1.append("path").datum(data).attr("fill", "none").attr("stroke", DC.accent).attr("stroke-width", 2.2)
        .attr("d", d3.line().x(d => x(d.t)).y(d => y1(Math.max(d.lr, 1e-12))));
      const g2 = svg.append("g").attr("transform", `translate(${L},${top2})`);
      const y2 = d3.scaleLinear().domain([Math.min(mb, 0.7) - 0.01, 1]).range([h2, 0]);
      DL.gridY(g2, y2, iw, 3);
      g2.append("g").attr("class", "axis").call(d3.axisLeft(y2).ticks(3));
      DL.axisB(g2, x, h2, 10, "training step");
      g2.append("text").attr("x", 0).attr("y", -4).attr("font-size", 11).attr("fill", DC.muted).text("momentum (β₁ for Adam)");
      g2.append("path").datum(data).attr("fill", "none").attr("stroke", DC.a2).attr("stroke-width", 2.2)
        .attr("d", d3.line().x(d => x(d.t)).y(d => y2(d.m)));
      [g1, g2].forEach((gg, i) => gg.append("line").attr("x1", x(s1 + 1)).attr("x2", x(s1 + 1)).attr("y1", 0).attr("y2", i ? h2 : h1)
        .attr("stroke", DC.muted).attr("stroke-dasharray", "4 4"));
      g1.append("text").attr("x", x(s1 + 1) + 5).attr("y", 12).attr("font-size", 10).attr("fill", DC.muted).text("peak at step " + Math.round(s1 + 1));
      const at = t => data[t].lr;
      $("oc-readout").innerHTML = `start <b>${fe(lr0)}</b> → peak <b>${fe(maxLr)}</b> at step ${Math.round(s1 + 1)} → end <b>${fe(lrMin)}</b> ` +
        `(${Math.log10(maxLr / lrMin).toFixed(1)} decades below peak) · LR at 50 % of the run: ${fe(at(500))}, at 90 %: ${fe(at(900))}` +
        (+$("oc-mb").value > +$("oc-mm").value ? " · base momentum clipped to the max" : "");
    }
    ["oc-max", "oc-pct", "oc-div", "oc-fdiv", "oc-ann", "oc-mb", "oc-mm", "oc-log"].forEach(id => $(id).addEventListener("input", draw));
    draw();
  })();

  /* ═══ 6 · learning-curve diagnoser ═════════════════════════════════ */
  (function () {
    const svg = d3.select("#diag-svg");
    if (svg.empty()) return;
    const W = 760, H = 320, E = 50, LNC = Math.log(10);
    const ex = (e, tau) => Math.exp(-e / tau);
    const S = {
      healthy: { tr: e => 0.22 + 2.08 * ex(e, 8), va: e => 0.34 + 1.96 * ex(e, 8.5) + 0.0005 * Math.max(0, e - 34) ** 1.4, amp: 0.02,
        d: ["Both curves fall, level off close together; validation a little above training.", "Nothing is wrong. The gap is ordinary generalisation error.", "Keep the best-validation checkpoint; if you want more, try a bigger model or longer schedule and see whether the gap grows."] },
      underfit: { tr: e => 1.15 + 1.15 * ex(e, 5), va: e => 1.2 + 1.1 * ex(e, 5), amp: 0.02,
        d: ["Training loss plateaus high and validation tracks it closely.", "Too little capacity, too much regularisation, too few epochs — or features that cannot predict the target.", "Overfit one batch first (§15). Then increase width/depth, remove dropout/weight decay, train longer, raise the LR with the finder. Do not add regularisation."] },
      overfit: { tr: e => 0.03 + 2.27 * ex(e, 6.5), va: e => 0.62 + 1.68 * ex(e, 5) + 0.0045 * Math.max(0, e - 9) ** 1.45, amp: 0.025,
        d: ["Training keeps falling toward zero; validation bottoms out and then rises.", "The model is memorising the training set; capacity exceeds what the data supports.", "Early-stop at the marked epoch; then more data / stronger augmentation, weight decay, dropout, a smaller model — one at a time (§09 stage 5)."] },
      lrhigh: { tr: e => 1.35 + 0.9 * ex(e, 2.5), va: e => 1.45 + 0.85 * ex(e, 2.5), amp: 0.16, osc: 0.18,
        d: ["Loss drops quickly, then bounces around a high plateau; large epoch-to-epoch swings.", "The step size exceeds the curvature bound: updates overshoot the valley floor.", "Rerun the LR finder and take a smaller rate; add warmup and a decaying schedule (one-cycle); clip gradients; check that batch size and LR were not changed independently."] },
      lrlow: { tr: e => 2.3 - 1.05 * (1 - ex(e, 70)), va: e => 2.32 - 1.0 * (1 - ex(e, 70)), amp: 0.01,
        d: ["Both curves descend in a slow, almost straight line and are nowhere near flat at the end.", "Learning rate far below the useful range, or a schedule that decays too early.", "LR finder: move up to the descending part of the curve. Check the scheduler is stepped per batch, not per epoch, and that warmup is not the whole run."] },
      nan: { tr: e => e < 19 ? 0.4 + 1.9 * ex(e, 6) : (e < 22 ? 0.4 + 1.9 * ex(19, 6) + (e - 18) ** 3 * 0.12 : NaN),
        va: e => e < 19 ? 0.5 + 1.8 * ex(e, 6) : (e < 21 ? 0.5 + 1.8 * ex(19, 6) + (e - 18) ** 3 * 0.18 : NaN), amp: 0.02,
        d: ["A normal run that suddenly spikes and becomes NaN.", "An overflow or invalid operation: LR spike, log(0) or division by a zero std, fp16 overflow, a bad example in the data.", "Resume from the last good checkpoint with anomaly detection / check_numerics on; clip gradients; use the fused stable loss; under fp16 use a GradScaler or switch to bf16; scan inputs for NaN/Inf."] },
      stuck: { tr: e => LNC - 0.004 * (1 - ex(e, 3)), va: e => LNC + 0.002, amp: 0.004,
        d: ["Both losses sit at ln C from the first epoch to the last.", "The model outputs the uniform distribution: no gradient reaches the weights (detached graph, frozen parameters, optimiser over the wrong parameters), LR of zero, or labels unrelated to inputs (shuffled separately).", "Overfit one batch; print per-layer gradient norms; confirm opt.param_groups contains the model's parameters; display a batch with its labels."] },
      noise: { tr: e => 1.05 + 1.25 * ex(e, 5) - 0.012 * Math.max(0, e - 14) ** 1.15, va: e => 1.28 + 1.02 * ex(e, 5) + 0.0035 * Math.max(0, e - 12) ** 1.45, amp: 0.02,
        d: ["Training plateaus at a high floor, then slowly creeps down again while validation rises.", "Many labels are wrong: the floor is the irreducible loss on noisy labels, and the late descent is the network memorising them.", "Inspect the highest-loss training examples and fix labels; early-stop before memorisation; label smoothing or noise-robust losses; more epochs will make it worse."] },
      leak: { tr: e => 0.55 + 1.75 * ex(e, 6), va: e => 0.22 + 1.9 * ex(e, 5), amp: 0.015,
        d: ["Validation is below training throughout — and suspiciously good.", "Benign causes first: dropout/augmentation make training harder than validation, and the training loss is averaged over an epoch while the weights improved during it. If the gap is large or the score too good: leakage — duplicates or the same entities in both splits, or a feature that encodes the target.", "Evaluate the training set in eval mode to remove the benign causes. Deduplicate across splits, split by entity/time, audit features computed after the event."] },
      shift: { tr: e => 0.2 + 2.1 * ex(e, 7), va: e => 1.25 + 1.05 * ex(e, 4), amp: 0.02,
        d: ["Validation stalls far above training almost from the start, with a gap that never closes.", "Validation data are not drawn from the training distribution (different source, time period, preprocessing), or the validation pipeline differs from training's.", "Compare preprocessing code paths; look at validation examples side by side with training ones; if the shift is real, collect training data like the target, or use a train-dev split to separate variance from mismatch (see ML Strategy)."] }
    };
    function draw() {
      const key = $("dg-scen").value, sc = S[key], nz = +$("dg-noise").value, showEs = $("dg-es").checked;
      $("dg-noisev").textContent = nz.toFixed(2);
      const r = DL.rng(11);
      const pts = d3.range(0, E + 1).map(e => {
        const osc = sc.osc ? sc.osc * Math.sin(e * 1.7) * (e > 3 ? 1 : 0) : 0;
        const a1 = sc.amp * nz * 3, t = sc.tr(e), v = sc.va(e);
        return { e: e, tr: isFinite(t) ? t + osc * 0.8 + a1 * DL.randn(r) * (sc.osc ? 1.5 : 1) : NaN,
          va: isFinite(v) ? v + osc + a1 * 1.3 * DL.randn(r) * (sc.osc ? 1.5 : 1) : NaN };
      });
      const F = DL.frame(svg, W, H, { l: 48, r: 110, t: 16, b: 36 }), g = F.g;
      const x = d3.scaleLinear().domain([0, E]).range([0, F.iw]);
      const y = d3.scaleLinear().domain([0, 3]).range([F.ih, 0]).clamp(true);
      DL.gridY(g, y, F.iw, 6);
      DL.axisB(g, x, F.ih, 10, "epoch");
      DL.axisL(g, y, 6, "loss");
      g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(LNC)).attr("y2", y(LNC)).attr("stroke", DC.muted).attr("stroke-dasharray", "5 4");
      g.append("text").attr("x", F.iw + 6).attr("y", y(LNC) + 4).attr("font-size", 10).attr("fill", DC.muted).text("ln 10 (uniform)");
      const line = k => d3.line().defined(d => isFinite(d[k])).x(d => x(d.e)).y(d => y(d[k]));
      g.append("path").datum(pts).attr("fill", "none").attr("stroke", DC.accent).attr("stroke-width", 2).attr("d", line("tr"));
      g.append("path").datum(pts).attr("fill", "none").attr("stroke", DC.a2).attr("stroke-width", 2).attr("d", line("va"));
      const lastT = pts.filter(p => isFinite(p.tr)).pop(), lastV = pts.filter(p => isFinite(p.va)).pop();
      if (key === "nan") g.append("text").attr("x", x(Math.max(lastT.e, lastV.e)) + 6).attr("y", 14).attr("font-size", 12)
        .attr("font-weight", 600).attr("fill", DC.bad).text("NaN →");
      DL.legend(g, [{ label: "training loss", color: DC.accent }, { label: "validation loss", color: DC.a2 }], F.iw + 8, 12);
      let best = null;
      pts.forEach(p => { if (isFinite(p.va) && (!best || p.va < best.va)) best = p; });
      if (showEs && best) {
        g.append("line").attr("x1", x(best.e)).attr("x2", x(best.e)).attr("y1", 0).attr("y2", F.ih).attr("stroke", DC.good).attr("stroke-dasharray", "3 3");
        g.append("circle").attr("cx", x(best.e)).attr("cy", y(best.va)).attr("r", 4.5).attr("fill", DC.good);
        g.append("text").attr("x", x(best.e) + (best.e > 40 ? -6 : 6)).attr("y", F.ih - 8).attr("text-anchor", best.e > 40 ? "end" : "start")
          .attr("font-size", 10).attr("fill", DC.good).text("best val · epoch " + best.e);
      }
      const gap = key === "nan" ? NaN : (lastV.va - lastT.tr);
      $("diag-readout").innerHTML =
        `<div><span class="k">What you see.</span> ${sc.d[0]}</div>` +
        `<div><span class="k">Likely cause.</span> ${sc.d[1]}</div>` +
        `<div><span class="k">Try first.</span> ${sc.d[2]}</div>` +
        `<div style="margin-top:6px;font-family:${mono};font-size:12px">final train ${key === "nan" ? "NaN" : lastT.tr.toFixed(3)} · final val ${key === "nan" ? "NaN" : lastV.va.toFixed(3)} · gap ${isFinite(gap) ? gap.toFixed(3) : "—"} · best val ${best ? best.va.toFixed(3) + " @ epoch " + best.e : "—"}</div>`;
    }
    ["dg-scen", "dg-noise", "dg-es"].forEach(id => $(id).addEventListener("input", draw));
    draw();
  })();
})();
