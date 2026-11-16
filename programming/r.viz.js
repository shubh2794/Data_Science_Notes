/* r.viz.js — interactive figure for r.html.
   Formula -> design matrix explorer: parses an R model formula right-hand side,
   expands terms in R's order, applies treatment contrasts with R's margin rule,
   builds X for a fixed 8-row dataset and fits least squares with aliasing (NA)
   detection in column order, as lm() does. Loaded after data.js / notes.js. */
(function(){
  const svgEl = document.getElementById("fm-svg");
  if (!svgEl) return;
  const C = {accent:"#5b9cff", a2:"#ffb454", good:"#4ade80", bad:"#f87171", muted:"#9aa3b2", ink:"#e6e9ef", line:"#2a2f3a", panel:"#1e222d"};
  const W = 640, H = 340;
  const svg = d3.select(svgEl);

  // ---------- data ----------
  const X0 = [1.2, 2.5, 3.1, 4.0, 4.8, 5.5, 6.3, 7.4];
  const G0 = ["a", "b", "c", "a", "b", "c", "a", "b"];
  const H0 = ["lo", "lo", "lo", "hi", "hi", "hi", "hi", "lo"];
  const NOISE = [0.3, -0.2, 0.1, -0.4, 0.2, 0.0, -0.1, 0.25];
  const gEff = {a: 0, b: 1.5, c: -1}, hEff = {lo: 0, hi: 0.8};
  const Y = X0.map((x, i) => +(2 + 0.9*x + gEff[G0[i]] + hEff[H0[i]] + 0.35*x*(G0[i] === "b" ? 1 : 0) + NOISE[i]).toFixed(2));
  const N = Y.length;
  const numeric = {
    "x": X0,
    "I(x^2)": X0.map(v => v*v),
    "I(x^3)": X0.map(v => v*v*v),
    "log(x)": X0.map(v => Math.log(v))
  };
  function factorLevels(name, ref){
    if (name === "g") { const L = ["a", "b", "c"]; return [ref].concat(L.filter(l => l !== ref)); }
    return ["lo", "hi"];                                       // h: levels set explicitly, lo is the reference
  }
  const factorData = {g: G0, h: H0};
  const isFactor = v => v === "g" || v === "h";
  const known = v => isFactor(v) || v in numeric;

  // ---------- parser ----------
  // term = array of variable names (ordered by first appearance); a formula value = {terms:[...], int: null|true|false}
  let order = [];                                             // global first-appearance order of variables
  const key = t => t.slice().sort((a, b) => order.indexOf(a) - order.indexOf(b)).join(":");
  function norm(t){ const u = Array.from(new Set(t)); return u.sort((a, b) => order.indexOf(a) - order.indexOf(b)); }
  function uniq(ts){ const seen = new Set(), out = []; ts.forEach(t => { const k = key(t); if (!seen.has(k)) { seen.add(k); out.push(norm(t)); } }); return out; }
  const V = (terms, int) => ({terms: uniq(terms), int: int === undefined ? null : int});
  const plus = (a, b) => V(a.terms.concat(b.terms), b.int !== null ? b.int : a.int);
  const minus = (a, b) => { const drop = new Set(b.terms.map(key));
    let int = a.int; if (b.int === true) int = false;          // "- 1" removes the intercept
    return V(a.terms.filter(t => !drop.has(key(t))), int); };
  const colon = (a, b) => { const out = []; a.terms.forEach(s => b.terms.forEach(t => out.push(s.concat(t)))); return V(out, null); };
  const star = (a, b) => plus(plus(a, b), colon(a, b));
  const slash = (a, b) => { const all = Array.from(new Set(a.terms.flat())); return plus(a, colon(V([all]), b)); };

  function tokenize(s){
    const toks = []; let i = 0;
    while (i < s.length) {
      const c = s[i];
      if (/\s/.test(c)) { i++; continue; }
      if ("+-*:/^()".includes(c)) { toks.push(c); i++; continue; }
      if (/[0-9]/.test(c)) { let j = i; while (j < s.length && /[0-9.]/.test(s[j])) j++; toks.push(s.slice(i, j)); i = j; continue; }
      if (/[A-Za-z_.]/.test(c)) {
        let j = i; while (j < s.length && /[A-Za-z0-9_.]/.test(s[j])) j++;
        let name = s.slice(i, j);
        if (s[j] === "(") {                                    // function call: capture up to the matching paren
          let depth = 0, k = j;
          for (; k < s.length; k++) { if (s[k] === "(") depth++; else if (s[k] === ")") { depth--; if (depth === 0) break; } }
          if (depth !== 0) throw new Error("unbalanced parentheses in " + name + "(…)");
          name = (name + s.slice(j, k + 1)).replace(/\s+/g, "");
          j = k + 1;
        }
        toks.push(name); i = j; continue;
      }
      throw new Error(`unexpected character "${c}"`);
    }
    return toks;
  }

  function parse(src){
    const toks = tokenize(src); let p = 0;
    const peek = () => toks[p], next = () => toks[p++];
    function sum(){
      let left;
      if (peek() === "-") { next(); left = minus(V([], null), prod()); } else left = prod();
      while (peek() === "+" || peek() === "-") { const op = next(); const r = prod(); left = op === "+" ? plus(left, r) : minus(left, r); }
      return left;
    }
    function prod(){
      let left = inter();
      while (peek() === "*" || peek() === "/") { const op = next(); const r = inter(); left = op === "*" ? star(left, r) : slash(left, r); }
      return left;
    }
    function inter(){
      let left = power();
      while (peek() === ":") { next(); left = colon(left, power()); }
      return left;
    }
    function power(){
      let base = atom();
      if (peek() === "^") { next(); const k = parseInt(next(), 10);
        if (!(k >= 1 && k <= 4)) throw new Error("^ needs a small integer order, e.g. (x + g)^2");
        let acc = base; for (let i = 1; i < k; i++) acc = star(acc, base);
        base = V(acc.terms.filter(t => t.length <= k), acc.int); }
      return base;
    }
    function atom(){
      const t = next();
      if (t === undefined) throw new Error("formula ends unexpectedly");
      if (t === "(") { const v = sum(); if (next() !== ")") throw new Error("missing )"); return v; }
      if (t === "1") return V([], true);
      if (t === "0") return V([], false);
      if (t === ".") throw new Error("'.' needs a data frame; spell the columns out here (x, g, h)");
      if (!known(t)) throw new Error(`unknown variable "${t}" — available: x, g, h, I(x^2), I(x^3), log(x)`);
      if (!order.includes(t)) order.push(t);
      return V([[t]], null);
    }
    const v = sum();
    if (p < toks.length) throw new Error(`unexpected "${toks[p]}"`);
    return v;
  }

  // ---------- model matrix ----------
  function build(src, ref){
    order = [];
    let rhs = src.replace(/^\s*y\s*~/, "");
    // R: a leading "0 +" or "+ 0" anywhere means no intercept; parse() handles 0 as int:false
    const f = parse(rhs);
    const intercept = f.int === null ? true : f.int;
    // R orders terms by degree, keeping first-appearance order within a degree
    const terms = f.terms.map((t, i) => ({t, i})).sort((a, b) => a.t.length - b.t.length || a.i - b.i).map(o => o.t);
    const cols = [];
    if (intercept) cols.push({name: "(Intercept)", v: new Array(N).fill(1), term: "(Intercept)"});
    // R's TermCode heuristic: a factor in term t is contrast-coded if its margin (t without it) is empty,
    // or is contained in some PRECEDING term; otherwise it gets full indicator coding. With no intercept,
    // the first factor of the first factor-containing term is switched to full indicator coding.
    let firstFactorTerm = -1, firstFactorVar = null;
    if (!intercept) {
      for (let k = 0; k < terms.length && firstFactorTerm < 0; k++) {
        const fv = terms[k].filter(isFactor).sort((a, b) => order.indexOf(a) - order.indexOf(b));
        if (fv.length) { firstFactorTerm = k; firstFactorVar = fv[0]; }
      }
    }
    terms.forEach((t, k) => {
      // per variable: list of {label, values}
      const parts = t.map(vname => {
        if (!isFactor(vname)) return [{lab: vname, v: numeric[vname]}];
        const lv = factorLevels(vname, ref);
        const margin = t.filter(u => u !== vname);
        let contrast = margin.length === 0 || terms.slice(0, k).some(p => margin.every(u => p.includes(u)));
        if (k === firstFactorTerm && vname === firstFactorVar) contrast = false;
        const use = contrast ? lv.slice(1) : lv;
        return use.map(l => ({lab: vname + l, v: factorData[vname].map(d => d === l ? 1 : 0)}));
      });
      // cartesian product, first variable varying fastest (R's column order)
      let combos = [{lab: [], v: new Array(N).fill(1)}];
      parts.forEach(opts => {
        const nxt = [];
        opts.forEach(o => combos.forEach(c => nxt.push({lab: c.lab.concat(o.lab), v: c.v.map((a, i) => a * o.v[i])})));
        combos = nxt;
      });
      combos.forEach(c => cols.push({name: c.lab.join(":"), v: c.v, term: t.join(":")}));
    });
    return {terms: terms.map(t => t.join(":")), intercept, cols};
  }

  // Gram-Schmidt in column order: a column whose residual is ~0 is aliased (NA), as with lm()'s pivoting
  function fit(cols){
    const Q = [], kept = [];
    cols.forEach((c, j) => {
      let r = c.v.slice();
      Q.forEach(q => { const d = d3.sum(r, (a, i) => a * q[i]); r = r.map((a, i) => a - d * q[i]); });
      const nrm = Math.sqrt(d3.sum(r, a => a*a)), base = Math.sqrt(d3.sum(c.v, a => a*a)) || 1;
      if (nrm / base > 1e-8) { Q.push(r.map(a => a / nrm)); kept.push(j); c.aliased = false; } else c.aliased = true;
    });
    // solve normal equations on kept columns (Gaussian elimination with partial pivoting)
    const p = kept.length, A = [], bvec = [];
    for (let a = 0; a < p; a++) {
      A.push([]); const ca = cols[kept[a]].v;
      for (let b = 0; b < p; b++) { const cb = cols[kept[b]].v; A[a].push(d3.sum(ca, (u, i) => u * cb[i])); }
      bvec.push(d3.sum(ca, (u, i) => u * Y[i]));
    }
    for (let k = 0; k < p; k++) {
      let piv = k; for (let r = k + 1; r < p; r++) if (Math.abs(A[r][k]) > Math.abs(A[piv][k])) piv = r;
      [A[k], A[piv]] = [A[piv], A[k]]; [bvec[k], bvec[piv]] = [bvec[piv], bvec[k]];
      for (let r = k + 1; r < p; r++) { const m = A[r][k] / A[k][k]; for (let c = k; c < p; c++) A[r][c] -= m * A[k][c]; bvec[r] -= m * bvec[k]; }
    }
    const beta = new Array(p).fill(0);
    for (let k = p - 1; k >= 0; k--) { let s = bvec[k]; for (let c = k + 1; c < p; c++) s -= A[k][c] * beta[c]; beta[k] = s / A[k][k]; }
    cols.forEach(c => c.beta = null);
    kept.forEach((j, a) => cols[j].beta = beta[a]);
    const yhat = Y.map((_, i) => d3.sum(kept, (j, a) => cols[j].v[i] * beta[a]));
    const rss = d3.sum(Y, (y, i) => (y - yhat[i]) ** 2), ybar = d3.mean(Y), tss = d3.sum(Y, y => (y - ybar) ** 2);
    return {rank: p, rss, r2: 1 - rss / tss};
  }

  // ---------- drawing ----------
  const gData = svg.append("g"), gMat = svg.append("g");
  const rowY0 = 92, rowH = 26;
  const dataCols = [{k: "y", v: Y}, {k: "x", v: X0}, {k: "g", v: G0}, {k: "h", v: H0}];
  const dW = 36, dX0 = 8;
  gData.append("text").attr("x", dX0).attr("y", 14).attr("fill", C.muted).attr("font-size", 10).text("data (8 rows)");
  dataCols.forEach((c, j) => {
    gData.append("text").attr("x", dX0 + j*dW + dW/2).attr("y", rowY0 - 10).attr("text-anchor", "middle").attr("fill", j === 0 ? C.a2 : C.ink).attr("font-size", 11).attr("font-weight", 600).text(c.k);
    c.v.forEach((val, i) => {
      gData.append("text").attr("x", dX0 + j*dW + dW/2).attr("y", rowY0 + i*rowH + 17).attr("text-anchor", "middle")
        .attr("fill", typeof val === "string" ? C.a2 : C.ink).attr("font-size", 11).attr("font-family", "monospace")
        .text(typeof val === "string" ? val : val.toFixed(1));
    });
  });
  gData.append("line").attr("x1", dX0 + 4*dW + 6).attr("x2", dX0 + 4*dW + 6).attr("y1", 20).attr("y2", H - 10).attr("stroke", C.line);
  const mX0 = dX0 + 4*dW + 18, mW = W - mX0 - 8;

  const colour = d3.scaleOrdinal(["#5b9cff", "#ffb454", "#c084fc", "#4ade80", "#f472b6", "#22d3ee", "#facc15"]);

  function render(){
    const src = d3.select("#fm-input").property("value");
    const ref = d3.select("#fm-ref").property("value");
    gMat.selectAll("*").remove();
    let model;
    try { model = build(src, ref); }
    catch (e) {
      gMat.append("text").attr("x", mX0).attr("y", 60).attr("fill", C.bad).attr("font-size", 12).text("parse error: " + e.message);
      d3.select("#fm-read").html(`<b style="color:${C.bad}">Could not parse "y ~ ${src.replace(/</g, "&lt;")}".</b> ${e.message.replace(/</g, "&lt;")}`);
      return;
    }
    const cols = model.cols, p = cols.length;
    if (p === 0) { gMat.append("text").attr("x", mX0).attr("y", 60).attr("fill", C.a2).attr("font-size", 12).text("empty model: no intercept and no terms"); d3.select("#fm-read").html("The model has no columns."); return; }
    if (p > 16) { gMat.append("text").attr("x", mX0).attr("y", 60).attr("fill", C.a2).attr("font-size", 12).text(`${p} columns: too many to draw here`); d3.select("#fm-read").html(`${p} columns for ${N} rows — hopelessly over-parameterised.`); return; }
    const res = fit(cols);
    const cw = Math.min(64, mW / p);
    colour.domain([]);
    gMat.append("text").attr("x", mX0).attr("y", 14).attr("fill", C.muted).attr("font-size", 10).text(`design matrix X: ${N} × ${p}   (model.matrix output)`);

    cols.forEach((c, j) => {
      const cx = mX0 + j*cw, tcol = c.term === "(Intercept)" ? C.muted : colour(c.term);
      // header (rotated)
      gMat.append("text").attr("transform", `translate(${cx + cw/2 + 3},${rowY0 - 8}) rotate(-40)`)
        .attr("fill", c.aliased ? C.bad : tcol).attr("font-size", 10.5).attr("font-family", "monospace").text(c.name);
      gMat.append("rect").attr("x", cx + 1).attr("y", rowY0 - 4).attr("width", cw - 2).attr("height", 3).attr("fill", tcol);
      const intCol = c.v.every(Number.isInteger);
      c.v.forEach((val, i) => {
        const nz = Math.abs(val) > 1e-12;
        gMat.append("rect").attr("x", cx + 1).attr("y", rowY0 + i*rowH + 1).attr("width", cw - 2).attr("height", rowH - 2).attr("rx", 3)
          .attr("fill", nz ? tcol : C.panel).attr("fill-opacity", nz ? (val === 1 ? 0.32 : 0.18) : 1)
          .attr("stroke", c.aliased ? C.bad : "none").attr("stroke-dasharray", c.aliased ? "3 2" : null);
        const txt = (intCol || val === 0) ? String(val) : (Math.abs(val) >= 100 ? val.toFixed(0) : val.toFixed(cw < 34 ? 1 : 2));
        gMat.append("text").attr("x", cx + cw/2).attr("y", rowY0 + i*rowH + 17).attr("text-anchor", "middle")
          .attr("fill", nz ? C.ink : "#5a6170").attr("font-size", cw < 34 ? 9 : 10.5).attr("font-family", "monospace").text(txt);
      });
      // coefficient row
      const by = rowY0 + N*rowH + 18;
      gMat.append("text").attr("x", cx + cw/2).attr("y", by).attr("text-anchor", "middle").attr("font-size", cw < 34 ? 9 : 10.5).attr("font-family", "monospace")
        .attr("fill", c.aliased ? C.bad : C.good).text(c.aliased ? "NA" : c.beta.toFixed(2));
    });
    gMat.append("text").attr("x", mX0 - 6).attr("y", rowY0 + N*rowH + 18).attr("text-anchor", "end").attr("fill", C.good).attr("font-size", 11).text("β̂");

    const aliased = cols.filter(c => c.aliased).map(c => c.name);
    const dfRes = N - res.rank;
    const hint = interpret(model, ref);
    d3.select("#fm-read").html(
      `terms: <b>${model.intercept ? "1 + " : "0 + "}${model.terms.join(" + ") || "(none)"}</b><br>` +
      `n = ${N}, columns p = ${p}, rank = <b style="color:${res.rank < p ? C.bad : C.good}">${res.rank}</b>, residual df = ${dfRes}` +
      (dfRes > 0 ? `, R² = ${res.r2.toFixed(3)}` : `, R² = 1 (saturated: the fit interpolates every row)`) +
      (aliased.length ? `<br><b style="color:${C.bad}">aliased (NA):</b> ${aliased.join(", ")} — each is a linear combination of earlier columns` : "") +
      (hint ? `<br><span style="color:${C.muted}">${hint}</span>` : "")
    );
  }

  function interpret(model, ref){
    const t = model.terms.join(" + ");
    const others = ["a", "b", "c"].filter(l => l !== ref);
    if (model.intercept && t === "g") return `(Intercept) is the mean of reference group ${ref}; g${others[0]} and g${others[1]} are differences from it.`;
    if (!model.intercept && t === "g") return "No intercept, so g is fully indicator-coded: each coefficient is one group's own mean.";
    if (model.intercept && t === "x + g") return `Parallel lines: one shared slope for x, and g${others[0]}, g${others[1]} shift the intercept relative to group ${ref}.`;
    if (model.intercept && t === "x + g + x:g") return `Separate lines per group: x is group ${ref}'s slope; x:g${others[0]} is how much steeper group ${others[0]} is. Change the reference level: the coefficients change, the fitted values do not.`;
    if (model.intercept && t === "x:g") return "No x main effect, so the margin rule gives x:g all three columns — one slope per group, sharing a single intercept.";
    if (model.intercept && t === "g + h + g:h") return "Two-way ANOVA with interaction: 1 + 2 + 1 + 2 = 6 columns, one free mean per (g, h) cell.";
    return "";
  }

  d3.select("#fm-input").on("input", render);
  d3.select("#fm-ref").on("change", render);
  d3.selectAll("#fm-presets button").on("click", function(){ d3.select("#fm-input").property("value", this.getAttribute("data-f")); render(); });
  render();
})();
