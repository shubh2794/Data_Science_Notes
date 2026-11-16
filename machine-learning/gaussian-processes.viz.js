/* gaussian-processes.viz.js — Interactive GP regression visualization.
   Loaded after data.js / notes.js.  Uses Cholesky decomposition to
   generate exact GP prior and posterior samples with uncertainty bands.

   Click to add observations, drag to reposition, double-click to remove.
   Controls: kernel type, length scale, signal variance, noise variance. */

(function () {
  /* ================================================================
     1. Math utilities
     ================================================================ */

  /** Box-Muller transform: returns a standard normal variate. */
  function randn() {
    let u, v, s;
    do {
      u = 2 * Math.random() - 1;
      v = 2 * Math.random() - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    return u * Math.sqrt(-2 * Math.log(s) / s);
  }

  /** Cholesky decomposition A = L Lᵀ.  Returns lower-triangular L.
      Falls back gracefully when A is numerically non-PD. */
  function cholesky(A) {
    var n = A.length;
    var L = [];
    for (var i = 0; i < n; i++) { L[i] = new Float64Array(n); }
    for (var i = 0; i < n; i++) {
      for (var j = 0; j <= i; j++) {
        var sum = 0;
        for (var k = 0; k < j; k++) sum += L[i][k] * L[j][k];
        if (i === j) {
          var diag = A[i][i] - sum;
          L[i][j] = diag > 1e-12 ? Math.sqrt(diag) : 1e-8;
        } else {
          L[i][j] = L[j][j] > 1e-20 ? (A[i][j] - sum) / L[j][j] : 0;
        }
      }
    }
    return L;
  }

  /** Forward substitution: solve L x = b where L is lower-triangular. */
  function forwardSolve(L, b) {
    var n = b.length, x = new Float64Array(n);
    for (var i = 0; i < n; i++) {
      var s = 0;
      for (var j = 0; j < i; j++) s += L[i][j] * x[j];
      x[i] = L[i][i] > 1e-20 ? (b[i] - s) / L[i][i] : 0;
    }
    return x;
  }

  /** Full Cholesky solve: (L Lᵀ) x = b.  Forward then backward. */
  function choleskySolve(L, b) {
    var z = forwardSolve(L, b);
    var n = z.length, x = new Float64Array(n);
    for (var i = n - 1; i >= 0; i--) {
      var s = 0;
      for (var j = i + 1; j < n; j++) s += L[j][i] * x[j];
      x[i] = L[i][i] > 1e-20 ? (z[i] - s) / L[i][i] : 0;
    }
    return x;
  }

  /** Build kernel matrix K[i][j] = kernelFn(xs1[i], xs2[j]). */
  function buildKernelMatrix(xs1, xs2, kernelFn) {
    var n1 = xs1.length, n2 = xs2.length;
    var K = [];
    for (var i = 0; i < n1; i++) {
      K[i] = new Float64Array(n2);
      for (var j = 0; j < n2; j++) K[i][j] = kernelFn(xs1[i], xs2[j]);
    }
    return K;
  }

  /* ================================================================
     2. Kernel functions
     ================================================================ */

  function rbfKernel(x1, x2, sf, ell) {
    var d = x1 - x2;
    return sf * Math.exp(-0.5 * d * d / (ell * ell));
  }

  function matern32Kernel(x1, x2, sf, ell) {
    var r = Math.abs(x1 - x2);
    var s = Math.sqrt(3) * r / ell;
    return sf * (1 + s) * Math.exp(-s);
  }

  function periodicKernel(x1, x2, sf, ell) {
    var period = 4;
    var s = Math.sin(Math.PI * (x1 - x2) / period);
    return sf * Math.exp(-2 * s * s / (ell * ell));
  }

  /* ================================================================
     3. GP computation
     ================================================================ */

  /**
   * Compute the GP prior or posterior.
   * Returns { mean, std, L } where L is the Cholesky of the covariance
   * (prior or posterior) for sampling.
   */
  function computeGP(xTest, obs, kFn, noiseVar) {
    var nt = xTest.length, no = obs.length;

    if (no === 0) {
      /* --- Prior --- */
      var Kss = buildKernelMatrix(xTest, xTest, kFn);
      for (var i = 0; i < nt; i++) Kss[i][i] += 1e-6;          // jitter
      var L = cholesky(Kss);
      var mean = new Float64Array(nt);                           // zero mean
      var std = new Float64Array(nt);
      for (var i = 0; i < nt; i++) std[i] = Math.sqrt(Math.max(0, Kss[i][i]));
      return { mean: mean, std: std, L: L };
    }

    /* --- Posterior --- */
    var xObs = [], yObs = [];
    for (var i = 0; i < no; i++) { xObs[i] = obs[i][0]; yObs[i] = obs[i][1]; }

    // K(X,X) + σ²ₙ I
    var Kxx = buildKernelMatrix(xObs, xObs, kFn);
    for (var i = 0; i < no; i++) Kxx[i][i] += noiseVar + 1e-6;

    // K(x*,X)  — each row is the covariance of one test point with all training points
    var Ksx = buildKernelMatrix(xTest, xObs, kFn);

    // K(x*,x*)
    var Kss = buildKernelMatrix(xTest, xTest, kFn);

    // Cholesky of K(X,X) + noise
    var Lxx = cholesky(Kxx);

    // α = (K + σ²ₙI)⁻¹ y
    var alpha = choleskySolve(Lxx, yObs);

    // Posterior mean: μ* = Ksx · α
    var mean = new Float64Array(nt);
    for (var i = 0; i < nt; i++) {
      var s = 0;
      for (var j = 0; j < no; j++) s += Ksx[i][j] * alpha[j];
      mean[i] = s;
    }

    // V[i] = L⁻¹ · Ksx[i,:]  (forward solve for each test point)
    var V = [];
    for (var i = 0; i < nt; i++) V[i] = forwardSolve(Lxx, Ksx[i]);

    // Posterior covariance: Σ* = Kss − V Vᵀ
    var Kpost = [];
    for (var i = 0; i < nt; i++) {
      Kpost[i] = new Float64Array(nt);
      for (var j = 0; j <= i; j++) {
        var dot = 0;
        for (var k = 0; k < no; k++) dot += V[i][k] * V[j][k];
        var val = Kss[i][j] - dot;
        Kpost[i][j] = val;
        if (i !== j) Kpost[j][i] = val;
      }
      Kpost[i][i] = Math.max(1e-8, Kpost[i][i]) + 1e-6;        // jitter
    }

    var std = new Float64Array(nt);
    for (var i = 0; i < nt; i++) std[i] = Math.sqrt(Math.max(0, Kpost[i][i]));

    var Lpost = cholesky(Kpost);
    return { mean: mean, std: std, L: Lpost };
  }

  /** Generate samples: f = mean + L z  for each stored z-vector. */
  function sampleGP(mean, L, zVecs) {
    var n = mean.length, ns = zVecs.length;
    var samples = [];
    for (var s = 0; s < ns; s++) {
      var f = new Float64Array(n);
      for (var i = 0; i < n; i++) {
        var val = mean[i];
        for (var j = 0; j <= i; j++) val += L[i][j] * zVecs[s][j];
        f[i] = val;
      }
      samples.push(f);
    }
    return samples;
  }

  /* ================================================================
     4. D3 setup
     ================================================================ */

  var svg = d3.select("#gp-svg");
  if (svg.empty()) return;                                       // guard

  var W = 700, H = 380;
  var mg = { t: 20, r: 20, b: 35, l: 45 };
  var xDomain = [-5, 5], yDomain = [-4, 4];
  var xS = d3.scaleLinear().domain(xDomain).range([mg.l, W - mg.r]);
  var yS = d3.scaleLinear().domain(yDomain).range([H - mg.b, mg.t]);

  // Clip path — keeps samples/bands inside the plot area
  svg.append("defs").append("clipPath").attr("id", "gp-clip")
    .append("rect").attr("x", mg.l).attr("y", mg.t)
    .attr("width", W - mg.l - mg.r).attr("height", H - mg.t - mg.b);

  // Layers (bottom → top)
  var gAxis = svg.append("g");
  var gPlot = svg.append("g").attr("clip-path", "url(#gp-clip)");
  var gBand = gPlot.append("path");
  var gSamples = gPlot.append("g");
  var gMean = gPlot.append("path");
  var gObs = svg.append("g");                                    // on top, unclipped

  // Axes
  gAxis.append("g").attr("class", "axis")
    .attr("transform", "translate(0," + (H - mg.b) + ")")
    .call(d3.axisBottom(xS).ticks(10));
  gAxis.append("g").attr("class", "axis")
    .attr("transform", "translate(" + mg.l + ",0)")
    .call(d3.axisLeft(yS).ticks(8));

  // Subtle zero line
  gPlot.append("line")
    .attr("x1", xS(-5)).attr("y1", yS(0))
    .attr("x2", xS(5)).attr("y2", yS(0))
    .attr("stroke", "#2a2f3a").attr("stroke-width", 1);

  // D3 generators
  var areaGen = d3.area()
    .x(function (d) { return d.px; })
    .y0(function (d) { return d.lo; })
    .y1(function (d) { return d.hi; })
    .curve(d3.curveNatural);

  var lineGen = d3.line()
    .x(function (d) { return d[0]; })
    .y(function (d) { return d[1]; })
    .curve(d3.curveNatural);

  /* ================================================================
     5. State
     ================================================================ */

  var NTEST = 80;
  var NSAMPLES = 3;
  var xTest = d3.range(NTEST).map(function (i) {
    return xDomain[0] + (xDomain[1] - xDomain[0]) * i / (NTEST - 1);
  });
  var observations = [];        // array of [x, y] pairs
  var zVectors = [];            // random z vectors for sampling, reused across redraws

  function regenerateZ() {
    zVectors = [];
    for (var s = 0; s < NSAMPLES; s++) {
      var z = [];
      for (var i = 0; i < NTEST; i++) z.push(randn());
      zVectors.push(z);
    }
  }
  regenerateZ();

  /* ================================================================
     6. Draw
     ================================================================ */

  function draw() {
    // --- Read controls ---
    var ell = parseFloat(d3.select("#gp-length").property("value"));
    var sigF = parseFloat(d3.select("#gp-var").property("value"));
    var sigN = parseFloat(d3.select("#gp-noise").property("value"));
    var kType = d3.select("#gp-kernel").property("value");

    d3.select("#gp-length-val").text(ell.toFixed(1));
    d3.select("#gp-var-val").text(sigF.toFixed(1));
    d3.select("#gp-noise-val").text(sigN.toFixed(2));

    // --- Build kernel function ---
    var kFn;
    if (kType === "matern32") {
      kFn = function (x1, x2) { return matern32Kernel(x1, x2, sigF, ell); };
    } else if (kType === "periodic") {
      kFn = function (x1, x2) { return periodicKernel(x1, x2, sigF, ell); };
    } else {
      kFn = function (x1, x2) { return rbfKernel(x1, x2, sigF, ell); };
    }

    // --- Compute GP ---
    var gp = computeGP(xTest, observations, kFn, sigN);
    var samples = sampleGP(gp.mean, gp.L, zVectors);

    // --- Band data (±2σ) ---
    var bandData = [];
    for (var i = 0; i < NTEST; i++) {
      bandData.push({
        px: xS(xTest[i]),
        hi: yS(gp.mean[i] + 2 * gp.std[i]),
        lo: yS(gp.mean[i] - 2 * gp.std[i])
      });
    }
    gBand.datum(bandData).attr("d", areaGen)
      .attr("fill", C.A).attr("fill-opacity", 0.15).attr("stroke", "none");

    // --- Sample lines ---
    var sData = samples.map(function (f) {
      var pts = [];
      for (var i = 0; i < NTEST; i++) pts.push([xS(xTest[i]), yS(f[i])]);
      return pts;
    });
    var sPaths = gSamples.selectAll("path").data(sData);
    sPaths.enter().append("path").attr("fill", "none")
      .attr("stroke", C.A).attr("stroke-width", 1.5).attr("stroke-opacity", 0.4)
      .merge(sPaths).attr("d", lineGen);
    sPaths.exit().remove();

    // --- Mean line ---
    var mPts = [];
    for (var i = 0; i < NTEST; i++) mPts.push([xS(xTest[i]), yS(gp.mean[i])]);
    gMean.datum(mPts).attr("d", lineGen)
      .attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 2.5);

    // --- Observation points ---
    var circles = gObs.selectAll("circle").data(observations);
    circles.enter().append("circle")
      .attr("r", 7)
      .attr("fill", C.good).attr("stroke", "#0f1117").attr("stroke-width", 2)
      .attr("cursor", "grab")
      .merge(circles)
      .attr("cx", function (d) { return xS(d[0]); })
      .attr("cy", function (d) { return yS(d[1]); });
    circles.exit().remove();

    // Drag behaviour
    gObs.selectAll("circle").call(
      d3.drag().on("drag", function (ev, d) {
        d[0] = Math.max(xDomain[0], Math.min(xDomain[1], xS.invert(ev.x)));
        d[1] = Math.max(yDomain[0], Math.min(yDomain[1], yS.invert(ev.y)));
        draw();
      })
    );

    // --- Readout ---
    var state = observations.length === 0 ? "Prior" : "Posterior (n = " + observations.length + ")";
    d3.select("#gp-readout").html(
      "State: <b>" + state + "</b> &middot; " +
      "Kernel: <b>" + kType + "</b> &middot; " +
      "&#8467; = <b>" + ell.toFixed(1) + "</b> &middot; " +
      "&sigma;&sup2; = <b>" + sigF.toFixed(1) + "</b> &middot; " +
      "&sigma;&sup2;&#8345; = <b>" + sigN.toFixed(2) + "</b>"
    );
  }

  /* ================================================================
     7. Event handlers
     ================================================================ */

  // Click to add an observation
  svg.on("click", function (ev) {
    if (ev.target.tagName === "circle") return;
    var pt = d3.pointer(ev);
    var px = xS.invert(pt[0]), py = yS.invert(pt[1]);
    if (px < xDomain[0] || px > xDomain[1]) return;
    if (py < yDomain[0] || py > yDomain[1]) return;
    if (observations.length >= 20) return;                       // cap for performance
    observations.push([px, py]);
    draw();
  });

  // Double-click to remove an observation
  svg.on("dblclick", function (ev) {
    if (ev.target.tagName !== "circle") return;
    ev.preventDefault();
    var datum = d3.select(ev.target).datum();
    observations = observations.filter(function (o) { return o !== datum; });
    draw();
  });

  // Controls
  d3.select("#gp-length").on("input", draw);
  d3.select("#gp-var").on("input", draw);
  d3.select("#gp-noise").on("input", draw);
  d3.select("#gp-kernel").on("change", draw);

  d3.select("#gp-resample").on("click", function () {
    regenerateZ();
    draw();
  });

  d3.select("#gp-clear").on("click", function () {
    observations = [];
    draw();
  });

  // Initial draw
  draw();
})();
