/* model-merging.viz.js — figures for model-merging.html.
   Loaded after data.js / notes.js. One IIFE per figure; each guards its own elements.
   All randomness is seeded so the figures (and the headless harness) are deterministic. */

function mmRng(seed){            // mulberry32
  let a = seed >>> 0;
  return function(){
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------- Figure 1: loss landscape, lerp vs SLERP, shared vs independent init ---------- */
(function(){
  const svgEl = document.getElementById("mm-land-svg");
  if (!svgEl || typeof d3 === "undefined") return;
  const scnSel = document.getElementById("mm-scn"), pathSel = document.getElementById("mm-path"),
        tIn = document.getElementById("mm-t"), tV = document.getElementById("mm-t-v"),
        read = document.getElementById("mm-land-read");
  if (!scnSel || !pathSel || !tIn) return;

  const C = {accent:"#2dd4bf", a2:"#ffb454", bad:"#f87171", ink:"#e6e9ef", muted:"#9aa3b2", line:"#2a2f3a", blue:"#5b9cff"};
  const svg = d3.select(svgEl);
  const R = 1.2, DEG = Math.PI/180;
  const pol = (r, deg) => [r*Math.cos(deg*DEG), r*Math.sin(deg*DEG)];
  const A = pol(R, 15), Bshared = pol(R, 62), Bind = pol(R, 82), Balign = pol(R, 30), PRE = pol(1.13, 38);

  // seeded small bumps — texture only, kept modest so the story is set by the valley
  const rnd = mmRng(7);
  const bumps = d3.range(7).map(() => ({x: -0.1 + rnd()*1.5, y: -0.1 + rnd()*1.5, a: (rnd()-0.5)*0.06, s: 0.03 + rnd()*0.04}));

  function loss(x, y, scn){
    const r = Math.hypot(x, y), phi = Math.atan2(y, x)/DEG;
    let L = 0.3 + 9*(r - R)*(r - R);                       // ring valley: weights of the "right" size
    L += 0.10*(1 - Math.exp(-Math.pow((phi - 45)/45, 2)));  // mild preference for the sector
    const B = scn === "indep" ? Bind : (scn === "aligned" ? Balign : Bshared);
    [A, B].forEach(P => { const d2 = (x-P[0])*(x-P[0]) + (y-P[1])*(y-P[1]); L -= 0.06*Math.exp(-d2/0.03); });
    if (scn !== "shared"){                                  // a ridge between unrelated basins
      L += 0.9*Math.exp(-Math.pow((phi - 54)/7, 2))*Math.exp(-Math.pow((r - R)/0.4, 2));
    }
    bumps.forEach(b => { const d2 = (x-b.x)*(x-b.x) + (y-b.y)*(y-b.y); L += b.a*Math.exp(-d2/b.s); });
    return L;
  }

  function lerp(P, Q, t){ return [(1-t)*P[0] + t*Q[0], (1-t)*P[1] + t*Q[1]]; }
  function slerp(P, Q, t){
    const nP = Math.hypot(P[0], P[1]), nQ = Math.hypot(Q[0], Q[1]);
    const c = Math.max(-1, Math.min(1, (P[0]*Q[0] + P[1]*Q[1])/(nP*nQ)));
    const Om = Math.acos(c);
    if (Om < 1e-6) return lerp(P, Q, t);
    const s = Math.sin(Om), a = Math.sin((1-t)*Om)/s, b = Math.sin(t*Om)/s;
    return [a*P[0] + b*Q[0], a*P[1] + b*Q[1]];
  }

  // map: world [-0.15, 1.45]² → pixels; map panel 300×300 at (20, 20)
  const X0 = 20, Y0 = 20, S = 300, lo = -0.15, hi = 1.45;
  const px = x => X0 + (x - lo)/(hi - lo)*S, py = y => Y0 + (hi - y)/(hi - lo)*S;
  const N = 80;

  const cache = {};
  function grid(scn){
    if (cache[scn]) return cache[scn];
    const v = new Array(N*N);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++){
      const x = lo + (i + 0.5)/N*(hi - lo), y = hi - (j + 0.5)/N*(hi - lo);
      v[j*N + i] = loss(x, y, scn);
    }
    return (cache[scn] = v);
  }

  const color = d3.scaleSequential(d3.interpolateViridis).domain([1.3, 0.1]).clamp(true);

  function draw(){
    svg.selectAll("*").remove();
    const scn = scnSel.value, mode = pathSel.value, t = +tIn.value;
    if (tV) tV.textContent = t.toFixed(2);
    const Bp = scn === "indep" ? Bind : (scn === "aligned" ? Balign : Bshared);

    // contour map
    const vals = grid(scn);
    const thr = d3.range(0.1, 1.35, 0.06);
    const cs = d3.contours().size([N, N]).thresholds(thr)(vals);
    const gp = d3.geoPath(d3.geoIdentity().scale(S/N));
    const gm = svg.append("g").attr("transform", `translate(${X0},${Y0})`);
    gm.append("rect").attr("width", S).attr("height", S).attr("fill", color(1.3));
    gm.selectAll("path").data(cs).join("path").attr("d", gp).attr("fill", d => color(d.value)).attr("stroke", "#0f1117").attr("stroke-opacity", 0.25).attr("stroke-width", 0.5);
    svg.append("rect").attr("x", X0).attr("y", Y0).attr("width", S).attr("height", S).attr("fill", "none").attr("stroke", C.line);

    // origin + norm circle (the sphere SLERP walks on)
    svg.append("circle").attr("cx", px(0)).attr("cy", py(0)).attr("r", 3).attr("fill", C.ink);
    svg.append("text").attr("x", px(0) + 6).attr("y", py(0) - 5).attr("font-size", 9.5).attr("fill", C.ink).text("0");

    // both paths, chosen one solid
    const ts = d3.range(0, 1.0001, 0.01);
    const linePts = ts.map(s => lerp(A, Bp, s)), arcPts = ts.map(s => slerp(A, Bp, s));
    const ln = d3.line().x(p => px(p[0])).y(p => py(p[1]));
    svg.append("path").attr("d", ln(linePts)).attr("fill", "none").attr("stroke", "#fff")
      .attr("stroke-width", mode === "lerp" ? 2.2 : 1).attr("stroke-dasharray", mode === "lerp" ? null : "4 3").attr("opacity", mode === "lerp" ? 1 : 0.6);
    svg.append("path").attr("d", ln(arcPts)).attr("fill", "none").attr("stroke", C.a2)
      .attr("stroke-width", mode === "slerp" ? 2.2 : 1).attr("stroke-dasharray", mode === "slerp" ? null : "4 3").attr("opacity", mode === "slerp" ? 1 : 0.7);

    // pretrained init and the fine-tuning moves (shared scenario only)
    if (scn === "shared"){
      [A, Bp].forEach(P => svg.append("line").attr("x1", px(PRE[0])).attr("y1", py(PRE[1])).attr("x2", px(P[0])).attr("y2", py(P[1]))
        .attr("stroke", C.blue).attr("stroke-width", 1.2).attr("stroke-dasharray", "2 2"));
      svg.append("circle").attr("cx", px(PRE[0])).attr("cy", py(PRE[1])).attr("r", 5).attr("fill", C.blue).attr("stroke", "#0f1117");
      svg.append("text").attr("x", px(PRE[0]) - 8).attr("y", py(PRE[1]) + 4).attr("text-anchor", "end").attr("font-size", 10).attr("fill", "#fff").text("θ_pre");
    }
    if (scn === "aligned"){
      svg.append("circle").attr("cx", px(Bind[0])).attr("cy", py(Bind[1])).attr("r", 5).attr("fill", "none").attr("stroke", C.bad).attr("stroke-dasharray", "2 2");
      svg.append("line").attr("x1", px(Bind[0])).attr("y1", py(Bind[1])).attr("x2", px(Balign[0])).attr("y2", py(Balign[1]))
        .attr("stroke", C.bad).attr("stroke-dasharray", "3 3");
      svg.append("text").attr("x", px(Bind[0]) + 8).attr("y", py(Bind[1]) + 3).attr("font-size", 9.5).attr("fill", C.bad).text("B before alignment");
    }
    const pts = [{p: A, n: "A"}, {p: Bp, n: scn === "aligned" ? "P·B" : "B"}];
    pts.forEach(o => {
      svg.append("circle").attr("cx", px(o.p[0])).attr("cy", py(o.p[1])).attr("r", 5.5).attr("fill", "#fff").attr("stroke", "#0f1117");
      svg.append("text").attr("x", px(o.p[0]) + 8).attr("y", py(o.p[1]) + 4).attr("font-size", 11).attr("font-weight", 700).attr("fill", "#fff").text(o.n);
    });
    const cur = mode === "lerp" ? lerp(A, Bp, t) : slerp(A, Bp, t);
    svg.append("circle").attr("cx", px(cur[0])).attr("cy", py(cur[1])).attr("r", 6).attr("fill", C.bad).attr("stroke", "#fff").attr("stroke-width", 1.5);

    // loss profile panel
    const PX = 370, PW = 250, PY = 30, PH = 230;
    const LA = loss(A[0], A[1], scn), LB = loss(Bp[0], Bp[1], scn);
    const prof = ts.map(s => {
      const pl = lerp(A, Bp, s), pa = slerp(A, Bp, s);
      return {t: s, lin: loss(pl[0], pl[1], scn), arc: loss(pa[0], pa[1], scn), base: (1-s)*LA + s*LB};
    });
    const yMax = Math.max(0.6, d3.max(prof, d => Math.max(d.lin, d.arc)) * 1.08);
    const yMin = Math.min(0, d3.min(prof, d => Math.min(d.lin, d.arc)) - 0.03);
    const xs = d3.scaleLinear().domain([0, 1]).range([PX, PX + PW]);
    const ys = d3.scaleLinear().domain([yMin, yMax]).range([PY + PH, PY]);
    const g = svg.append("g");
    g.append("g").attr("transform", `translate(0,${PY + PH})`).call(d3.axisBottom(xs).ticks(5)).attr("color", C.muted);
    g.append("g").attr("transform", `translate(${PX},0)`).call(d3.axisLeft(ys).ticks(5)).attr("color", C.muted);
    g.append("text").attr("x", PX + PW/2).attr("y", PY + PH + 30).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", C.muted).text("t  (0 = A, 1 = B)");
    g.append("text").attr("x", PX).attr("y", PY - 10).attr("font-size", 10).attr("fill", C.muted).text("loss along the path");
    const mk = k => d3.line().x(d => xs(d.t)).y(d => ys(d[k]));
    g.append("path").attr("d", mk("base")(prof)).attr("fill", "none").attr("stroke", C.muted).attr("stroke-dasharray", "2 3");
    g.append("path").attr("d", mk("lin")(prof)).attr("fill", "none").attr("stroke", "#fff").attr("stroke-width", mode === "lerp" ? 2 : 1).attr("opacity", mode === "lerp" ? 1 : 0.55);
    g.append("path").attr("d", mk("arc")(prof)).attr("fill", "none").attr("stroke", C.a2).attr("stroke-width", mode === "slerp" ? 2 : 1).attr("opacity", mode === "slerp" ? 1 : 0.65);
    const curL = loss(cur[0], cur[1], scn);
    g.append("line").attr("x1", xs(t)).attr("x2", xs(t)).attr("y1", PY).attr("y2", PY + PH).attr("stroke", C.bad).attr("stroke-opacity", 0.5);
    g.append("circle").attr("cx", xs(t)).attr("cy", ys(curL)).attr("r", 4.5).attr("fill", C.bad);
    // legend
    [["lerp (chord)", "#fff", null], ["SLERP (arc)", C.a2, null], ["endpoint mix", C.muted, "2 3"]].forEach((l, i) => {
      const lx = PX + 6 + i*84, ly = PY + PH + 50;
      g.append("line").attr("x1", lx).attr("x2", lx + 16).attr("y1", ly - 3).attr("y2", ly - 3).attr("stroke", l[1]).attr("stroke-width", 2).attr("stroke-dasharray", l[2]);
      g.append("text").attr("x", lx + 20).attr("y", ly).attr("font-size", 9.5).attr("fill", C.muted).text(l[0]);
    });

    const barrier = d3.max(prof, d => (mode === "lerp" ? d.lin : d.arc) - d.base);
    const norm = Math.hypot(cur[0], cur[1]);
    const note = scn === "shared" ? "both fine-tunes started from θ_pre, so they share one basin; the angle between them is exaggerated here — real fine-tunes of one base are far closer, so lerp's dip is much smaller"
               : scn === "indep" ? "no shared init: a ridge separates the basins and no interpolation path avoids it"
               : "B's hidden units were permuted to match A's: same function, now in A's basin";
    if (read) read.innerHTML = `${mode === "lerp" ? "lerp" : "SLERP"} at t = <b>${t.toFixed(2)}</b> · loss <b>${curL.toFixed(3)}</b> · ‖θ(t)‖ = <b>${norm.toFixed(3)}</b> (endpoints ≈ ${R.toFixed(2)}) · barrier on this path <b style="color:${barrier > 0.12 ? C.bad : C.accent}">${Math.max(0, barrier).toFixed(3)}</b><br>${note}`;
  }
  [scnSel, pathSel].forEach(el => el.addEventListener("change", draw));
  tIn.addEventListener("input", draw);
  draw();
})();

/* ---------- Figure 2: TIES / DARE explorer ---------- */
(function(){
  const svgEl = document.getElementById("mm-ties-svg");
  if (!svgEl || typeof d3 === "undefined") return;
  const ruleSel = document.getElementById("mm-rule"), trimIn = document.getElementById("mm-trim"),
        pIn = document.getElementById("mm-p"), lamIn = document.getElementById("mm-lam"),
        resBtn = document.getElementById("mm-resample"), read = document.getElementById("mm-ties-read");
  if (!ruleSel || !trimIn || !pIn || !lamIn) return;
  const setV = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };

  const C = {accent:"#2dd4bf", bad:"#f87171", ink:"#e6e9ef", muted:"#9aa3b2", line:"#2a2f3a"};
  const TC = ["#5b9cff", "#c084fc", "#a3e635"];
  const NP = 24, NT = 3;
  const svg = d3.select(svgEl);

  // seeded task vectors: small noise everywhere, a few large entries per task, some in conflict
  const rnd = mmRng(42);
  const gauss = () => { const u = Math.max(1e-9, rnd()), v = rnd(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };
  const tau = d3.range(NT).map(() => d3.range(NP).map(() => 0.12*gauss()));
  for (let t = 0; t < NT; t++) for (let i = 0; i < NP; i++){
    if (rnd() < 0.3) tau[t][i] += (rnd() < 0.5 ? -1 : 1)*(0.5 + 0.5*rnd());
  }
  // force a few clear conflicts so the election has work to do
  [[3, 0.8, -0.7, 0.2], [9, -0.6, 0.9, -0.5], [15, 0.7, 0.6, -0.9], [20, -0.8, 0.1, 0.75]].forEach(r => {
    for (let t = 0; t < NT; t++) tau[t][r[0]] = r[t + 1];
  });

  let maskSeed = 1;
  function process(){
    const rule = ruleSel.value, trim = +trimIn.value/100, p = +pIn.value, lam = +lamIn.value;
    const mr = mmRng(1000 + maskSeed);
    // 1. DARE
    const dare = tau.map(row => row.map(v => (p > 0 && mr() < p) ? 0 : v/(1 - p)));
    // 2. trim (per task, keep top-k by magnitude among nonzero entries)
    const keepN = Math.round(NP*(1 - trim));
    const trimmed = dare.map(row => {
      const order = row.map((v, i) => [Math.abs(v), i]).sort((a, b) => b[0] - a[0]);
      const keep = new Set(order.slice(0, keepN).map(d => d[1]));
      return row.map((v, i) => keep.has(i) ? v : 0);
    });
    // 3. merge
    let conflicts = 0, outvoted = 0;
    const merged = d3.range(NP).map(i => {
      const col = trimmed.map(r => r[i]);
      const nz = col.filter(v => v !== 0);
      if (nz.some(v => v > 0) && nz.some(v => v < 0)) conflicts++;
      if (rule === "sum") return lam*d3.sum(col);
      if (rule === "mean") return lam*d3.sum(col)/NT;
      const g = Math.sign(d3.sum(col));
      const agree = nz.filter(v => Math.sign(v) === g);
      outvoted += nz.length - agree.length;
      return agree.length ? lam*d3.sum(agree)/agree.length : 0;
    });
    // raw (no DARE/trim) reference for the expectation check
    const rawSum = d3.sum(tau, r => d3.sum(r)), dareSum = d3.sum(dare, r => d3.sum(r));
    const survivors = d3.sum(trimmed, r => r.filter(v => v !== 0).length);
    return {rule, trim, p, lam, dare, trimmed, merged, conflicts, outvoted, rawSum, dareSum, survivors};
  }

  function draw(){
    svg.selectAll("*").remove();
    const s = process();
    setV("mm-trim-v", (s.trim*100).toFixed(0)); setV("mm-p-v", s.p.toFixed(1)); setV("mm-lam-v", s.lam.toFixed(1));
    const L = 40, Rm = 14, T = 22, B = 46, W = 640, H = 300;
    const band = (W - L - Rm)/NP;
    const absMax = Math.max(0.5, d3.max(s.dare.flat().map(Math.abs).concat(s.merged.map(Math.abs))) || 1);
    const y = d3.scaleLinear().domain([-absMax, absMax]).range([H - B, T]).nice();
    svg.append("g").attr("transform", `translate(${L},0)`).call(d3.axisLeft(y).ticks(6)).attr("color", C.muted);
    svg.append("line").attr("x1", L).attr("x2", W - Rm).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", C.muted).attr("stroke-width", 0.8);
    const bar = (v) => ({y: Math.min(y(0), y(v)), h: Math.abs(y(v) - y(0))});

    for (let i = 0; i < NP; i++){
      const x0 = L + i*band;
      // merged delta: wide outline
      const m = bar(s.merged[i]);
      svg.append("rect").attr("x", x0 + 1.5).attr("width", band - 3).attr("y", m.y).attr("height", Math.max(0.5, m.h))
        .attr("fill", C.accent).attr("fill-opacity", 0.16).attr("stroke", C.accent).attr("stroke-width", 1.4);
      // per-task bars
      const sub = (band - 8)/NT;
      for (let t = 0; t < NT; t++){
        const orig = s.dare[t][i], kept = s.trimmed[t][i];
        const shown = kept !== 0 ? kept : (orig !== 0 ? orig : tau[t][i]);
        const b = bar(shown);
        svg.append("rect").attr("x", x0 + 4 + t*sub).attr("width", Math.max(1, sub - 1)).attr("y", b.y).attr("height", Math.max(0.5, b.h))
          .attr("fill", TC[t]).attr("fill-opacity", kept !== 0 ? 0.9 : 0.14);
      }
      const nz = s.trimmed.map(r => r[i]).filter(v => v !== 0);
      if (nz.some(v => v > 0) && nz.some(v => v < 0))
        svg.append("circle").attr("cx", x0 + band/2).attr("cy", H - B + 12).attr("r", 3.2).attr("fill", C.bad);
      if (i % 4 === 0) svg.append("text").attr("x", x0 + band/2).attr("y", H - B + 26).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", C.muted).text(i);
    }
    svg.append("text").attr("x", L).attr("y", 13).attr("font-size", 10).attr("fill", C.muted).text("delta value per parameter (after rescaling)");
    ["task 1", "task 2", "task 3", "merged"].forEach((n, k) => {
      const lx = W - Rm - 270 + k*68, ly = H - 8;
      svg.append("rect").attr("x", lx).attr("y", ly - 9).attr("width", 10).attr("height", 10).attr("rx", 2)
        .attr("fill", k < 3 ? TC[k] : C.accent).attr("fill-opacity", k < 3 ? 0.9 : 0.3).attr("stroke", k < 3 ? "none" : C.accent);
      svg.append("text").attr("x", lx + 14).attr("y", ly).attr("font-size", 10).attr("fill", C.muted).text(n);
    });
    svg.append("text").attr("x", L).attr("y", H - 8).attr("font-size", 10).attr("fill", C.muted).text("parameter index");

    const ruleTxt = s.rule === "sum" ? "sum" : s.rule === "mean" ? "mean" : "TIES disjoint mean";
    const tiesTxt = s.rule === "ties" ? ` · election discarded <b>${s.outvoted}</b> dissenting entries` : " · conflicts are averaged straight through";
    const dareTxt = s.p > 0 ? ` · DARE: total delta ${s.rawSum.toFixed(2)} → ${s.dareSum.toFixed(2)} after drop + 1/(1−p) rescale (equal only in expectation; resample to see the spread)` : "";
    if (read) read.innerHTML = `${ruleTxt}, λ = ${s.lam.toFixed(1)} · surviving task entries <b>${s.survivors}</b>/${NP*NT} · parameters with a sign conflict <b style="color:${s.conflicts ? C.bad : C.accent}">${s.conflicts}</b>${tiesTxt} · ‖merged‖₂ = <b>${Math.hypot(...s.merged).toFixed(2)}</b>${dareTxt}`;
  }

  [ruleSel].forEach(el => el.addEventListener("change", draw));
  [trimIn, pIn, lamIn].forEach(el => el.addEventListener("input", draw));
  if (resBtn) resBtn.addEventListener("click", () => { maskSeed++; draw(); });
  draw();
})();
