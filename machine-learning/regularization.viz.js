/* regularization.viz.js — extracted from regularization.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* Interactive D3 polynomial regression fitter & weights bar chart */
(function(){
  const svg = d3.select("#poly-svg"), W = 460, H = 280, m = { t: 16, r: 16, b: 30, l: 36 };
  const x = d3.scaleLinear().domain([0, 10]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0, 10]).range([H - m.b, m.t]);
  
  // Weights plot setup
  const wSvg = d3.select("#weights-svg"), wW = 280, wH = 280, wm = { t: 25, r: 16, b: 35, l: 40 };
  const wx = d3.scaleBand().domain(d3.range(1, 10).map(String)).range([wm.l, wW - wm.r]).padding(0.2);
  const wy = d3.scaleLinear().domain([-1.2, 1.2]).range([wH - wm.b, wm.t]); // Normalized weight range
  
  let pts = [
    { x: 1.2, y: 3.5 }, { x: 2.1, y: 5.8 }, { x: 3.2, y: 7.2 }, { x: 4.3, y: 6.8 },
    { x: 5.1, y: 4.5 }, { x: 6.0, y: 2.4 }, { x: 7.2, y: 2.1 }, { x: 8.5, y: 4.2 },
    { x: 9.3, y: 6.5 }
  ];
  
  // Create axis grids
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6));
  
  wSvg.append("g").attr("class", "axis").attr("transform", `translate(0,${wH - wm.b})`).call(d3.axisBottom(wx));
  wSvg.append("g").attr("class", "axis").attr("transform", `translate(${wm.l},0)`).call(d3.axisLeft(wy).ticks(5));
  
  // Zero line for weights chart
  wSvg.append("line")
    .attr("x1", wm.l).attr("y1", wy(0))
    .attr("x2", wW - wm.r).attr("y2", wy(0))
    .attr("stroke", "#3a4150")
    .attr("stroke-width", 1.5)
    .attr("stroke-dasharray", "3 2");

  // Plot elements
  const curvePath = svg.append("path").attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 2.5);
  const gPts = svg.append("g");
  const gBars = wSvg.append("g");
  const readout = d3.select("#poly-readout");
  const lamDisplay = d3.select("#lam-display");
  
  // Coordinate Descent solver for Polynomial Lasso/Ridge
  function fitPolynomial(points, degree, regType, lambda) {
    const N = points.length;
    const D = degree;
    
    // Normalize coordinates to [-1.1, 1.1] for matrix stability
    const X = [];
    const targetY = [];
    for (let i = 0; i < N; i++) {
      const px = (points[i].x - 5) / 4.5;
      const py = (points[i].y - 5) / 4.5;
      targetY.push(py);
      const row = [];
      for (let j = 0; j <= D; j++) {
        row.push(Math.pow(px, j));
      }
      X.push(row);
    }
    
    // Initialize weights
    const w = new Array(D + 1).fill(0);
    const maxIter = 1000;
    
    for (let iter = 0; iter < maxIter; iter++) {
      for (let j = 0; j <= D; j++) {
        let rho = 0;
        let z = 0;
        
        for (let i = 0; i < N; i++) {
          let pred_no_j = 0;
          for (let k = 0; k <= D; k++) {
            if (k !== j) {
              pred_no_j += w[k] * X[i][k];
            }
          }
          const r_i = targetY[i] - pred_no_j;
          rho += X[i][j] * r_i;
          z += X[i][j] * X[i][j];
        }
        
        if (j === 0) {
          w[j] = z === 0 ? 0 : rho / z;
        } else {
          if (regType === 'l1') {
            const gamma = N * lambda;
            const abs_rho = Math.abs(rho);
            if (abs_rho <= gamma) {
              w[j] = 0;
            } else {
              w[j] = (Math.sign(rho) * (abs_rho - gamma)) / z;
            }
          } else if (regType === 'l2') {
            const denom = z + N * lambda;
            w[j] = denom === 0 ? 0 : rho / denom;
          } else {
            // None
            w[j] = z === 0 ? 0 : rho / z;
          }
        }
      }
    }
    return w;
  }
  
  function draw() {
    const deg = parseInt(d3.select("#deg-val").property("value"));
    const reg = d3.select("#reg-type").property("value");
    const sliderVal = parseFloat(d3.select("#lam-val").property("value"));
    
    // lambda mapping: exponential slider for finer control at small values
    let lambda = sliderVal === 0 ? 0 : Math.pow(10, (sliderVal / 100) * 3 - 3.5);
    if (reg === 'none') lambda = 0;
    
    lamDisplay.text(reg === 'none' ? 'λ = 0' : `λ = ${lambda.toFixed(4)}`);
    
    // Fit normalized weights
    const w = fitPolynomial(pts, deg, reg, lambda);
    
    // 1. Draw fitted curve
    const pathPoints = d3.range(0, 10.1, 0.15).map(xi => {
      const px = (xi - 5) / 4.5;
      let py = 0;
      for (let j = 0; j <= deg; j++) {
        py += w[j] * Math.pow(px, j);
      }
      const yi = py * 4.5 + 5;
      const clampedY = Math.max(-2, Math.min(12, yi)); // Keep within reasonable plotting bounds
      return `${x(xi)},${y(clampedY)}`;
    });
    curvePath.attr("d", "M" + pathPoints.join(" L"));
    
    // 2. Render points
    const sel = gPts.selectAll("circle").data(pts);
    sel.enter().append("circle").attr("r", 6).attr("fill", C.B).attr("stroke", "#0f1117").attr("stroke-width", 1.5).attr("cursor", "grab")
      .merge(sel).attr("cx", d => x(d.x)).attr("cy", d => y(d.y));
    sel.exit().remove();
    
    gPts.selectAll("circle").call(d3.drag()
      .on("drag", function(ev, d) {
        d.x = Math.max(0.1, Math.min(9.9, x.invert(ev.x)));
        d.y = Math.max(0.1, Math.min(9.9, y.invert(ev.y)));
        draw();
      }));
      
    // 3. Render Weights Bar Chart
    // Create an array for weights 1 to 9 (padded with 0 if degree is smaller)
    const weightsData = d3.range(1, 10).map(j => {
      return { id: String(j), val: j <= deg ? w[j] : 0 };
    });
    
    // Dynamically adjust y scale if weights are huge, but cap minimum range at [-1.2, 1.2]
    const maxWeightVal = d3.max(weightsData, d => Math.abs(d.val)) || 0;
    const yLimit = Math.max(1.2, Math.min(25, maxWeightVal * 1.1));
    wy.domain([-yLimit, yLimit]);
    wSvg.select("g.axis.axis-y").remove(); // Re-draw axis
    wSvg.append("g").attr("class", "axis axis-y").attr("transform", `translate(${wm.l},0)`).call(d3.axisLeft(wy).ticks(5));
    
    // Update baseline
    wSvg.select("line").attr("y1", wy(0)).attr("y2", wy(0));

    const bars = gBars.selectAll("rect").data(weightsData);
    bars.enter().append("rect")
      .merge(bars)
      .attr("x", d => wx(d.id))
      .attr("y", d => d.val >= 0 ? wy(d.val) : wy(0))
      .attr("width", wx.bandwidth())
      .attr("height", d => Math.abs(wy(d.val) - wy(0)))
      .attr("fill", d => Math.abs(d.val) < 1e-4 ? "#2a2f3a" : (reg === 'l1' ? C.good : (reg === 'l2' ? C.A : C.B)))
      .attr("opacity", d => d.val === 0 ? 0.2 : 0.95);
    bars.exit().remove();
    
    // Compute stats
    let mse = 0;
    pts.forEach(p => {
      const px = (p.x - 5) / 4.5;
      let py = 0;
      for (let j = 0; j <= deg; j++) {
        py += w[j] * Math.pow(px, j);
      }
      const actualY = (p.y - 5) / 4.5;
      mse += Math.pow(py - actualY, 2);
    });
    mse = mse / pts.length;
    
    let penalty = 0;
    for (let j = 1; j <= deg; j++) {
      penalty += (reg === 'l1') ? Math.abs(w[j]) : (w[j]*w[j]);
    }
    
    readout.html(`
      MSE (normalized): <b>${mse.toFixed(4)}</b><br>
      Penalty sum: <b>${penalty.toFixed(4)}</b><br>
      Non-zero weights (w₁–w₉): <b>${weightsData.filter(d => Math.abs(d.val) > 1e-4).length} / 9</b>
    `);
  }
  
  // Click background to add a point
  svg.on("click", function(ev) {
    if (ev.target.tagName === "circle") return;
    const [mx, my] = d3.pointer(ev);
    if (pts.length >= 18) return; // limit to 18 points
    pts.push({ x: Math.max(0.1, Math.min(9.9, x.invert(mx))), y: Math.max(0.1, Math.min(9.9, y.invert(my))) });
    draw();
  });
  
  // Double-click point to delete
  svg.on("dblclick", function(ev) {
    if (ev.target.tagName !== "circle") return;
    const circleData = d3.select(ev.target).datum();
    pts = pts.filter(p => p !== circleData);
    draw();
  });
  
  d3.select("#poly-reset").on("click", () => {
    pts = [
      { x: 1.2, y: 3.5 }, { x: 2.1, y: 5.8 }, { x: 3.2, y: 7.2 }, { x: 4.3, y: 6.8 },
      { x: 5.1, y: 4.5 }, { x: 6.0, y: 2.4 }, { x: 7.2, y: 2.1 }, { x: 8.5, y: 4.2 },
      { x: 9.3, y: 6.5 }
    ];
    draw();
  });
  
  d3.select("#deg-val").on("input", draw);
  d3.select("#reg-type").on("change", draw);
  d3.select("#lam-val").on("input", draw);
  
  draw();
})();
