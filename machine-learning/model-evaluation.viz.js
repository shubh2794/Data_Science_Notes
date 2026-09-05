/* model-evaluation.viz.js — extracted from model-evaluation.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* Interactive D3 Model Evaluation widget: threshold + prevalence → confusion matrix, ROC and PR */
(function(){
  const svg = d3.select("#dist-svg"), W = 460, H = 230, m = { t: 16, r: 16, b: 30, l: 30 };
  const x = d3.scaleLinear().domain([0, 10]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0, 0.4]).range([H - m.b, m.t]);

  // ROC + PR mini-plots share a geometry
  const rW = 160, rH = 140, rm = { t: 12, r: 12, b: 24, l: 28 };
  const rx = d3.scaleLinear().domain([0, 1]).range([rm.l, rW - rm.r]);
  const ry = d3.scaleLinear().domain([0, 1]).range([rH - rm.b, rm.t]);
  const rSvg = d3.select("#roc-svg"), pSvg = d3.select("#pr-svg");

  // Normal density
  function pdf(val, mean, std) {
    return (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((val - mean) / std, 2));
  }
  // Normal CDF via an erf approximation
  function cdf(val, mean, std) {
    const xVal = (val - mean) / (std * Math.sqrt(2));
    const t = 1 / (1 + 0.5 * Math.abs(xVal));
    const tau = t * Math.exp(-xVal * xVal - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    const erfVal = xVal >= 0 ? 1 - tau : tau - 1;
    return 0.5 * (1 + erfVal);
  }

  // Class parameters
  const meanNeg = 3.8, stdNeg = 1.25;
  const meanPos = 6.2, stdPos = 1.25;
  const N_NEG = 100;                              // negatives are held fixed; the slider sets n positives

  // Axes
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(8));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4));

  [rSvg, pSvg].forEach(s => {
    s.append("g").attr("class", "axis").attr("transform", `translate(0,${rH - rm.b})`).call(d3.axisBottom(rx).ticks(3));
    s.append("g").attr("class", "axis").attr("transform", `translate(${rm.l},0)`).call(d3.axisLeft(ry).ticks(3));
  });
  rSvg.append("text").attr("x", rW / 2).attr("y", 10).attr("text-anchor", "middle")
    .attr("fill", C.muted).attr("font-size", 9).text("ROC · TPR vs FPR");
  pSvg.append("text").attr("x", rW / 2).attr("y", 10).attr("text-anchor", "middle")
    .attr("fill", C.muted).attr("font-size", 9).text("PR · Precision vs Recall");

  // Chance line for ROC (diagonal); PR chance line is horizontal at the prevalence
  rSvg.append("line")
    .attr("x1", rx(0)).attr("y1", ry(0))
    .attr("x2", rx(1)).attr("y2", ry(1))
    .attr("stroke", "#3a4150").attr("stroke-dasharray", "2 2");
  const prBase = pSvg.append("line").attr("stroke", "#3a4150").attr("stroke-dasharray", "2 2");

  // Threshold grid used for both curves
  const thGrid = d3.range(0.05, 10.0, 0.05);

  // ROC is prevalence-free, so it can be precomputed once
  const rocPoints = thGrid.map(th => [1 - cdf(th, meanNeg, stdNeg), 1 - cdf(th, meanPos, stdPos)])
                          .sort((a, b) => a[0] - b[0]);
  const lineGen = d3.line().x(d => rx(d[0])).y(d => ry(d[1]));
  rSvg.append("path").datum(rocPoints)
    .attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 2)
    .attr("d", lineGen);

  // PR curve is redrawn whenever prevalence changes
  const prPath = pSvg.append("path").attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 2);

  // Shaded regions of the score distributions
  const areaTN = svg.append("path").attr("fill", C.A).attr("fill-opacity", 0.16);
  const areaFP = svg.append("path").attr("fill", C.bad).attr("fill-opacity", 0.22);
  const areaFN = svg.append("path").attr("fill", C.bad).attr("fill-opacity", 0.22);
  const areaTP = svg.append("path").attr("fill", C.good).attr("fill-opacity", 0.22);

  const lineNeg = svg.append("path").attr("fill", "none").attr("stroke", C.A).attr("stroke-width", 2);
  const linePos = svg.append("path").attr("fill", "none").attr("stroke", C.B).attr("stroke-width", 2);

  const threshLine = svg.append("line")
    .attr("stroke", "#ef4444").attr("stroke-width", 2.5).attr("stroke-dasharray", "3 1")
    .attr("cursor", "ew-resize");

  const rocDot = rSvg.append("circle").attr("r", 4.5).attr("fill", "#ef4444").attr("stroke", "#0f1117").attr("stroke-width", 1);
  const prDot  = pSvg.append("circle").attr("r", 4.5).attr("fill", "#ef4444").attr("stroke", "#0f1117").attr("stroke-width", 1);

  // Static density outlines
  const pathRange = d3.range(0, 10.05, 0.05);
  lineNeg.attr("d", "M" + pathRange.map(xi => `${x(xi)},${y(pdf(xi, meanNeg, stdNeg))}`).join(" L"));
  linePos.attr("d", "M" + pathRange.map(xi => `${x(xi)},${y(pdf(xi, meanPos, stdPos))}`).join(" L"));

  function update() {
    const threshVal = parseFloat(d3.select("#thresh-val").property("value")) / 10;
    const nPos = parseFloat(d3.select("#prev-val").property("value"));
    const prevalence = nPos / (nPos + N_NEG);

    threshLine.attr("x1", x(threshVal)).attr("y1", y(0)).attr("x2", x(threshVal)).attr("y2", y(0.38));

    // Shaded areas
    const step = 0.05;
    function areaPath(from, to, mean, std) {
      const pts = d3.range(from, to + step, step).map(xi => `${x(xi)},${y(pdf(xi, mean, std))}`);
      pts.unshift(`${x(from)},${y(0)}`);
      pts.push(`${x(to)},${y(0)}`);
      return "M" + pts.join(" L");
    }
    areaTN.attr("d", areaPath(0, threshVal, meanNeg, stdNeg));
    areaFP.attr("d", areaPath(threshVal, 10, meanNeg, stdNeg));
    areaFN.attr("d", areaPath(0, threshVal, meanPos, stdPos));
    areaTP.attr("d", areaPath(threshVal, 10, meanPos, stdPos));

    // Rates (prevalence-free) then counts (prevalence-dependent)
    const tnRate = cdf(threshVal, meanNeg, stdNeg), fpRate = 1 - tnRate;
    const fnRate = cdf(threshVal, meanPos, stdPos), tpRate = 1 - fnRate;

    const TN = Math.round(tnRate * N_NEG);
    const FP = Math.round(fpRate * N_NEG);
    const FN = Math.round(fnRate * nPos);
    const TP = Math.round(tpRate * nPos);

    document.getElementById("m-tn").textContent = TN;
    document.getElementById("m-fp").textContent = FP;
    document.getElementById("m-fn").textContent = FN;
    document.getElementById("m-tp").textContent = TP;

    const accuracy  = (TP + TN) / (nPos + N_NEG);
    const precision = (TP + FP) === 0 ? 0 : TP / (TP + FP);
    const recall    = (TP + FN) === 0 ? 0 : TP / (TP + FN);
    const f1 = (precision + recall) === 0 ? 0 : 2 * precision * recall / (precision + recall);
    const majority = Math.max(nPos, N_NEG) / (nPos + N_NEG);   // always-predict-the-common-class baseline

    // PR curve for the current prevalence: precision = π·TPR / (π·TPR + (1−π)·FPR)
    const prPoints = thGrid.map(th => {
      const tpr = 1 - cdf(th, meanPos, stdPos);
      const fpr = 1 - cdf(th, meanNeg, stdNeg);
      const denom = prevalence * tpr + (1 - prevalence) * fpr;
      return [tpr, denom === 0 ? 1 : prevalence * tpr / denom];
    }).sort((a, b) => a[0] - b[0]);
    prPath.datum(prPoints).attr("d", lineGen);
    prBase.attr("x1", rx(0)).attr("y1", ry(prevalence)).attr("x2", rx(1)).attr("y2", ry(prevalence));

    rocDot.attr("cx", rx(fpRate)).attr("cy", ry(tpRate));
    prDot.attr("cx", rx(recall)).attr("cy", ry(precision));

    d3.select("#eval-readout").html(`
      prevalence π = <b>${(prevalence * 100).toFixed(1)}%</b> &nbsp;·&nbsp; ${nPos} pos / ${N_NEG} neg<br>
      FPR: <b style="color:${C.bad}">${fpRate.toFixed(3)}</b> &nbsp; TPR (Recall): <b style="color:${C.good}">${tpRate.toFixed(3)}</b> <span style="color:${C.muted}">(both fixed as π moves)</span><br>
      Accuracy:  <b>${accuracy.toFixed(3)}</b> <span style="color:${C.muted}">(majority baseline ${majority.toFixed(3)})</span><br>
      Precision: <b style="color:${C.B}">${precision.toFixed(3)}</b> <span style="color:${C.muted}">(chance ${prevalence.toFixed(3)})</span><br>
      F₁-Score:  <b>${f1.toFixed(3)}</b>
    `);
  }

  // Drag the threshold line directly
  threshLine.call(d3.drag().on("drag", function(ev) {
    const val = Math.max(1, Math.min(9, x.invert(ev.x)));
    d3.select("#thresh-val").property("value", Math.round(val * 10));
    update();
  }));

  d3.select("#thresh-val").on("input", update);
  d3.select("#prev-val").on("input", update);
  update();
})();
