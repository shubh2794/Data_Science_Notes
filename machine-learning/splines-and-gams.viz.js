/* splines-and-gams.viz.js
   Loaded after data.js / notes.js.
   Interactive spline regression explorer: noisy data, adjustable knots,
   cubic vs natural spline toggle showing boundary behavior. */

(function(){
  /* ── palette ── */
  var C = {accent:"#ffb454", blue:"#5b9cff", good:"#4ade80", bad:"#f87171",
           muted:"#9aa3b2", ink:"#e6e9ef", line:"#2a2f3a"};

  /* ── layout ── */
  var W = 640, H = 320;
  var m = {t: 22, r: 24, b: 36, l: 48};

  /* ── deterministic PRNG (mulberry32) ── */
  var seed = 20241105;
  function rnd(){
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }
  function gauss(){
    var u = 0, v = 0;
    while (u === 0) u = rnd();
    while (v === 0) v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* ── true function ── */
  function trueFn(x){ return 2 * Math.sin(0.9 * x) + 0.3 * x; }

  /* ── data generation ── */
  var N = 55;
  var SIGMA = 0.85;
  var data = [];

  function generateData(){
    seed = Math.floor(rnd() * 2147483647) + 1;
    data = [];
    for (var i = 0; i < N; i++){
      var x = 0.5 + 9.0 * rnd();
      var y = trueFn(x) + SIGMA * gauss();
      data.push({x: x, y: y});
    }
    data.sort(function(a, b){ return a.x - b.x; });
  }

  /* ── Gaussian elimination with partial pivoting ── */
  function solve(Ain, bin){
    var n = bin.length;
    var A = Ain.map(function(r){ return r.slice(); });
    var b = bin.slice();
    for (var i = 0; i < n; i++){
      var maxR = i;
      for (var r = i + 1; r < n; r++){
        if (Math.abs(A[r][i]) > Math.abs(A[maxR][i])) maxR = r;
      }
      var tmpA = A[i]; A[i] = A[maxR]; A[maxR] = tmpA;
      var tmpB = b[i]; b[i] = b[maxR]; b[maxR] = tmpB;
      var pivot = A[i][i] || 1e-14;
      for (var r2 = i + 1; r2 < n; r2++){
        var fac = A[r2][i] / pivot;
        for (var c = i; c < n; c++) A[r2][c] -= fac * A[i][c];
        b[r2] -= fac * b[i];
      }
    }
    var x = new Array(n).fill(0);
    for (var i2 = n - 1; i2 >= 0; i2--){
      var s = b[i2];
      for (var c2 = i2 + 1; c2 < n; c2++) s -= A[i2][c2] * x[c2];
      x[i2] = s / (A[i2][i2] || 1e-14);
    }
    return x;
  }

  /* ── Truncated power basis (cubic spline) ──
     Basis: {1, x, x^2, x^3, (x-k1)_+^3, ..., (x-kK)_+^3}
     Total: K + 4 functions. */
  function tpBasis(x, knots){
    var row = [1, x, x * x, x * x * x];
    for (var j = 0; j < knots.length; j++){
      var d = x - knots[j];
      row.push(d > 0 ? d * d * d : 0);
    }
    return row;
  }

  /* ── Natural spline basis (ESL 5.2.1, eq 5.4-5.5) ──
     Basis: {1, x, d_1(x)-d_{K-1}(x), d_2(x)-d_{K-1}(x), ..., d_{K-2}(x)-d_{K-1}(x)}
     where d_k(x) = [(x-xi_k)_+^3 - (x-xi_K)_+^3] / (xi_K - xi_k)
     Total: K functions. For K < 3, only {1, x} (linear). */
  function nsBasis(x, knots){
    var K = knots.length;
    if (K < 3) return [1, x];

    function dk(xv, k){
      var a = xv - knots[k];
      var b = xv - knots[K - 1];
      var num = (a > 0 ? a * a * a : 0) - (b > 0 ? b * b * b : 0);
      var den = knots[K - 1] - knots[k];
      return den === 0 ? 0 : num / den;
    }

    var row = [1, x];
    var dKm1 = dk(x, K - 2);       /* d_{K-1}(x) using 0-indexed K-2 */
    for (var k = 0; k < K - 2; k++){
      row.push(dk(x, k) - dKm1);
    }
    return row;
  }

  /* ── Fit a spline by OLS on the basis expansion ── */
  function fitSpline(pts, knots, type){
    var basisFn = (type === "natural") ? nsBasis : tpBasis;
    var n = pts.length;
    /* determine p from a probe evaluation */
    var p = basisFn(0, knots).length;

    /* form X'X and X'y */
    var XtX = [];
    for (var ii = 0; ii < p; ii++){
      XtX.push(new Array(p).fill(0));
    }
    var Xty = new Array(p).fill(0);

    for (var i = 0; i < n; i++){
      var bi = basisFn(pts[i].xn, knots);
      for (var j = 0; j < p; j++){
        Xty[j] += bi[j] * pts[i].y;
        for (var k = j; k < p; k++){
          XtX[j][k] += bi[j] * bi[k];
        }
      }
    }
    /* symmetrize */
    for (var j2 = 0; j2 < p; j2++){
      for (var k2 = j2 + 1; k2 < p; k2++){
        XtX[k2][j2] = XtX[j2][k2];
      }
    }
    /* ridge for numerical stability */
    for (var j3 = 0; j3 < p; j3++) XtX[j3][j3] += 1e-6;

    var beta = solve(XtX, Xty);

    /* return a prediction function (takes normalized x) */
    return function(xn){
      var b = basisFn(xn, knots);
      var v = 0;
      for (var jj = 0; jj < p; jj++) v += beta[jj] * b[jj];
      return v;
    };
  }

  /* ── quantile helper ── */
  function quantile(sortedArr, q){
    var pos = q * (sortedArr.length - 1);
    var lo = Math.floor(pos), hi = Math.ceil(pos);
    if (lo === hi) return sortedArr[lo];
    return sortedArr[lo] + (sortedArr[hi] - sortedArr[lo]) * (pos - lo);
  }

  /* ── D3 rendering ── */
  var svg = d3.select("#spline-svg");
  if (svg.empty()) return;      /* guard: page may not have the element */

  /* scales — x extends beyond data to show boundary behavior */
  var xDomMin = -0.8, xDomMax = 10.8;
  var sx = d3.scaleLinear().domain([xDomMin, xDomMax]).range([m.l, W - m.r]);
  /* y domain auto-computed per draw */
  var sy = d3.scaleLinear().range([H - m.b, m.t]);

  /* evaluation grid for smooth curves */
  var grid = d3.range(xDomMin, xDomMax + 0.02, 0.08);

  function draw(){
    var nKnots = +d3.select("#spline-knots").property("value");
    var type   = d3.select("#spline-type").property("value");
    d3.select("#spline-knots-val").text(nKnots);

    /* ---- normalize x to [0,1] for numerical stability ---- */
    var xs = data.map(function(d){ return d.x; });
    var xMin = Math.min.apply(null, xs);
    var xMax = Math.max.apply(null, xs);
    var xRange = xMax - xMin || 1;
    function norm(x){ return (x - xMin) / xRange; }

    var normPts = data.map(function(d){
      return {xn: norm(d.x), y: d.y};
    });

    /* ---- place knots at quantiles of data x (normalized) ---- */
    var sortedXn = normPts.map(function(d){ return d.xn; }).sort(function(a,b){return a-b;});
    var knots = [];
    for (var i = 0; i < nKnots; i++){
      var q = (i + 1) / (nKnots + 1);
      knots.push(quantile(sortedXn, q));
    }

    /* ---- fit ---- */
    var predict = fitSpline(normPts, knots, type);

    /* ---- compute y domain from data + fitted curve ---- */
    var allY = data.map(function(d){ return d.y; });
    grid.forEach(function(gx){
      var pv = predict(norm(gx));
      if (isFinite(pv)) allY.push(pv);
      allY.push(trueFn(gx));
    });
    var yLo = d3.min(allY), yHi = d3.max(allY);
    var yPad = (yHi - yLo) * 0.12 || 1;
    sy.domain([yLo - yPad, yHi + yPad]);

    /* ---- clear & redraw ---- */
    svg.selectAll("*").remove();

    /* background shading for extrapolation zone */
    var xDataMin = d3.min(data, function(d){return d.x;});
    var xDataMax = d3.max(data, function(d){return d.x;});
    svg.append("rect")
      .attr("x", sx(xDomMin)).attr("y", m.t)
      .attr("width", sx(xDataMin) - sx(xDomMin))
      .attr("height", H - m.b - m.t)
      .attr("fill", "#1a1520").attr("opacity", 0.35);
    svg.append("rect")
      .attr("x", sx(xDataMax)).attr("y", m.t)
      .attr("width", sx(xDomMax) - sx(xDataMax))
      .attr("height", H - m.b - m.t)
      .attr("fill", "#1a1520").attr("opacity", 0.35);

    /* axes */
    svg.append("g").attr("class","axis")
      .attr("transform","translate(0,"+(H - m.b)+")")
      .call(d3.axisBottom(sx).ticks(10).tickSize(-H + m.t + m.b))
      .selectAll("line").attr("stroke","#1e222d");
    svg.append("g").attr("class","axis")
      .attr("transform","translate("+m.l+",0)")
      .call(d3.axisLeft(sy).ticks(6).tickSize(-W + m.l + m.r))
      .selectAll("line").attr("stroke","#1e222d");
    svg.selectAll(".axis path").attr("stroke", C.line);
    svg.selectAll(".axis text").attr("fill", C.muted).attr("font-size", 10);

    /* axis labels */
    svg.append("text").attr("x", (m.l + W - m.r) / 2).attr("y", H - 4)
      .attr("text-anchor","middle").attr("fill", C.muted).attr("font-size", 10)
      .text("x");
    svg.append("text").attr("x", 12).attr("y", (m.t + H - m.b) / 2)
      .attr("text-anchor","middle").attr("fill", C.muted).attr("font-size", 10)
      .attr("transform","rotate(-90,12,"+(m.t + H - m.b)/2+")")
      .text("y");

    /* knot vertical lines */
    knots.forEach(function(kn){
      var kx = kn * xRange + xMin;      /* un-normalize for plotting */
      svg.append("line")
        .attr("x1", sx(kx)).attr("y1", m.t)
        .attr("x2", sx(kx)).attr("y2", H - m.b)
        .attr("stroke", "#4a5060").attr("stroke-width", 1)
        .attr("stroke-dasharray", "4 4").attr("opacity", 0.7);
    });

    /* true function (dashed blue) */
    var trueLine = d3.line()
      .x(function(gx){ return sx(gx); })
      .y(function(gx){ return sy(trueFn(gx)); });
    svg.append("path")
      .attr("d", trueLine(grid))
      .attr("fill","none").attr("stroke", C.blue)
      .attr("stroke-width", 1.8).attr("stroke-dasharray","6 4");

    /* fitted spline (solid accent) — clamp to avoid SVG blow-up */
    var yClipLo = sy.domain()[0] - 5;
    var yClipHi = sy.domain()[1] + 5;
    var fitLine = d3.line()
      .defined(function(gx){
        var pv = predict(norm(gx));
        return isFinite(pv) && pv > yClipLo && pv < yClipHi;
      })
      .x(function(gx){ return sx(gx); })
      .y(function(gx){ return sy(predict(norm(gx))); });
    svg.append("path")
      .attr("d", fitLine(grid))
      .attr("fill","none").attr("stroke", C.accent)
      .attr("stroke-width", 2.6);

    /* data points */
    svg.selectAll(".pt").data(data).join("circle")
      .attr("class","pt")
      .attr("cx", function(d){ return sx(d.x); })
      .attr("cy", function(d){ return sy(d.y); })
      .attr("r", 3.2)
      .attr("fill", C.muted).attr("fill-opacity", 0.6)
      .attr("stroke", C.muted).attr("stroke-opacity", 0.3).attr("stroke-width", 0.5);

    /* ---- readout ---- */
    var rss = 0;
    data.forEach(function(d){
      var r = d.y - predict(norm(d.x));
      rss += r * r;
    });
    var df = (type === "natural") ? nKnots : nKnots + 4;
    d3.select("#spline-read").html(
      "knots <b>" + nKnots + "</b> &middot; " +
      "type <b>" + type + "</b> &middot; " +
      "effective df <b>" + df + "</b> &middot; " +
      "RSS <b>" + rss.toFixed(2) + "</b> &middot; " +
      "RSS/n <b>" + (rss / data.length).toFixed(3) + "</b>"
    );
  }

  /* ── event wiring ── */
  d3.select("#spline-knots").on("input", draw);
  d3.select("#spline-type").on("change", draw);
  d3.select("#spline-regen").on("click", function(){ generateData(); draw(); });

  /* ── initial render ── */
  generateData();
  draw();
})();
