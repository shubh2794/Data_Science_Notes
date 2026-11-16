/* semi-supervised-learning.viz.js
   Interactive label-propagation demo on two-moons data.
   Loaded after data.js / notes.js (which defines global C). */

(function(){
  var svg = d3.select("#ssl-svg");
  if (svg.empty()) return;

  var W = 640, H = 380, PAD = 30;
  var pts = [], edges = [];
  var labeledIdx = [], nLabel = 4, kNeighbors = 6;
  var method = "supervised";
  var animTimer = null;

  /* ---- Colour helpers ---- */
  var colA   = C.A;       // #5b9cff (blue, class A)
  var colB   = C.B;       // #ffb454 (orange, class B)
  var colMut = C.muted;   // #9aa3b2 (unlabeled)
  var colGood = C.good;   // #4ade80 (propagated)

  /* ---- Two-moons generator ---- */
  function makeMoons(n, noise){
    var out = [];
    var half = Math.floor(n / 2);
    for (var i = 0; i < half; i++){
      var angle = Math.PI * i / half;
      out.push({
        ox: Math.cos(angle) + (Math.random() - 0.5) * noise,
        oy: Math.sin(angle) + (Math.random() - 0.5) * noise,
        trueClass: 0, label: -1, soft: 0.5
      });
    }
    for (var i = 0; i < n - half; i++){
      var angle = Math.PI * i / (n - half);
      out.push({
        ox: 1 - Math.cos(angle) + (Math.random() - 0.5) * noise,
        oy: 0.5 - Math.sin(angle) + (Math.random() - 0.5) * noise,
        trueClass: 1, label: -1, soft: 0.5
      });
    }
    return out;
  }

  /* ---- Scales ---- */
  var xScale = d3.scaleLinear().range([PAD, W - PAD]);
  var yScale = d3.scaleLinear().range([H - PAD, PAD]);

  /* ---- Layers ---- */
  var gEdge = svg.append("g").attr("class", "edges");
  var gBound = svg.append("g").attr("class", "boundary");
  var gPts  = svg.append("g").attr("class", "points");

  /* ---- Generate fresh data ---- */
  function generate(){
    if (animTimer) { clearInterval(animTimer); animTimer = null; }
    nLabel = +d3.select("#ssl-nlabel").property("value");
    pts = makeMoons(120, 0.18);

    /* Fit scales */
    var xExt = d3.extent(pts, function(d){ return d.ox; });
    var yExt = d3.extent(pts, function(d){ return d.oy; });
    var xPad = (xExt[1] - xExt[0]) * 0.08;
    var yPad = (yExt[1] - yExt[0]) * 0.08;
    xScale.domain([xExt[0] - xPad, xExt[1] + xPad]);
    yScale.domain([yExt[0] - yPad, yExt[1] + yPad]);

    /* Pick labeled points: nLabel per class, spread out */
    labeledIdx = [];
    for (var c = 0; c < 2; c++){
      var cands = [];
      pts.forEach(function(p, i){ if (p.trueClass === c) cands.push(i); });
      /* shuffle and take nLabel */
      cands.sort(function(){ return Math.random() - 0.5; });
      for (var j = 0; j < Math.min(nLabel, cands.length); j++){
        labeledIdx.push(cands[j]);
        pts[cands[j]].label = c;
      }
    }

    /* Build kNN graph */
    buildGraph();
    updateMethod();
  }

  /* ---- kNN graph ---- */
  function dist2(a, b){ return (a.ox - b.ox) * (a.ox - b.ox) + (a.oy - b.oy) * (a.oy - b.oy); }

  function buildGraph(){
    edges = [];
    var n = pts.length;
    for (var i = 0; i < n; i++){
      var dists = [];
      for (var j = 0; j < n; j++){
        if (i !== j) dists.push({ j: j, d: dist2(pts[i], pts[j]) });
      }
      dists.sort(function(a, b){ return a.d - b.d; });
      for (var k = 0; k < Math.min(kNeighbors, dists.length); k++){
        edges.push({ s: i, t: dists[k].j, w: Math.exp(-dists[k].d * 12) });
      }
    }
    /* make symmetric */
    var seen = {};
    var symEdges = [];
    edges.forEach(function(e){
      var key = Math.min(e.s, e.t) + "-" + Math.max(e.s, e.t);
      if (!seen[key]) {
        seen[key] = true;
        symEdges.push({ s: e.s, t: e.t, w: e.w });
      }
    });
    edges = symEdges;
  }

  /* ---- Label propagation ---- */
  function propagateLabels(steps){
    /* initialize soft labels */
    pts.forEach(function(p){
      if (p.label === 0) p.soft = 0;
      else if (p.label === 1) p.soft = 1;
      else p.soft = 0.5;
    });

    for (var s = 0; s < steps; s++){
      var newSoft = pts.map(function(p){ return p.soft; });
      for (var i = 0; i < pts.length; i++){
        if (pts[i].label >= 0) continue; // clamped
        var wSum = 0, valSum = 0;
        edges.forEach(function(e){
          if (e.s === i) { wSum += e.w; valSum += e.w * pts[e.t].soft; }
          if (e.t === i) { wSum += e.w; valSum += e.w * pts[e.s].soft; }
        });
        if (wSum > 0) newSoft[i] = valSum / wSum;
      }
      for (var i = 0; i < pts.length; i++){
        pts[i].soft = newSoft[i];
      }
    }
  }

  /* ---- Supervised-only boundary (simple linear on labeled pts) ---- */
  function supervisedBoundary(){
    /* Compute centroid of each class from labeled points, draw perpendicular bisector */
    var cA = { x: 0, y: 0, n: 0 }, cB = { x: 0, y: 0, n: 0 };
    pts.forEach(function(p){
      if (p.label === 0) { cA.x += p.ox; cA.y += p.oy; cA.n++; }
      else if (p.label === 1) { cB.x += p.ox; cB.y += p.oy; cB.n++; }
    });
    if (cA.n === 0 || cB.n === 0) return null;
    cA.x /= cA.n; cA.y /= cA.n;
    cB.x /= cB.n; cB.y /= cB.n;
    /* midpoint and normal */
    var mx = (cA.x + cB.x) / 2, my = (cA.y + cB.y) / 2;
    var dx = cB.x - cA.x, dy = cB.y - cA.y;
    /* perpendicular direction */
    var nx = -dy, ny = dx;
    var len = Math.sqrt(nx * nx + ny * ny);
    if (len < 1e-9) return null;
    nx /= len; ny /= len;
    /* extend to edges */
    var ext = 5;
    return {
      x1: xScale(mx - nx * ext), y1: yScale(my - ny * ext),
      x2: xScale(mx + nx * ext), y2: yScale(my + ny * ext)
    };
  }

  /* ---- Draw ---- */
  function draw(showEdges, animateEdges){
    /* Edges */
    var edgeData = (method === "propagation" && showEdges) ? edges : [];
    var eSel = gEdge.selectAll("line").data(edgeData, function(d){ return d.s + "-" + d.t; });
    eSel.exit().remove();
    var eEnter = eSel.enter().append("line")
      .attr("stroke", C.line)
      .attr("stroke-width", 0.5)
      .attr("opacity", 0);
    eEnter.merge(eSel)
      .attr("x1", function(d){ return xScale(pts[d.s].ox); })
      .attr("y1", function(d){ return yScale(pts[d.s].oy); })
      .attr("x2", function(d){ return xScale(pts[d.t].ox); })
      .attr("y2", function(d){ return yScale(pts[d.t].oy); })
      .transition().duration(animateEdges ? 600 : 0)
      .attr("opacity", function(d){ return d.w * 0.35; });

    /* Boundary */
    gBound.selectAll("*").remove();
    if (method === "supervised"){
      var b = supervisedBoundary();
      if (b) {
        gBound.append("line")
          .attr("x1", b.x1).attr("y1", b.y1)
          .attr("x2", b.x2).attr("y2", b.y2)
          .attr("stroke", C.bad).attr("stroke-width", 2.5)
          .attr("stroke-dasharray", "8 5").attr("opacity", 0.8);
        gBound.append("text")
          .attr("x", W - PAD - 4).attr("y", PAD + 14)
          .attr("text-anchor", "end").attr("fill", C.bad)
          .attr("font-size", 11).attr("font-weight", 600)
          .text("supervised boundary");
      }
    } else {
      /* For propagation, draw a contour-like boundary using soft labels */
      /* Approximate: draw a dashed line through the 0.5 iso-contour */
      /* We'll use a grid-based approach */
      var step = 6;
      var gridPts = [];
      for (var gx = PAD; gx < W - PAD; gx += step){
        for (var gy = PAD; gy < H - PAD; gy += step){
          var ox = xScale.invert(gx), oy = yScale.invert(gy);
          /* predict by weighted nearest neighbors using soft labels */
          var wSum = 0, valSum = 0;
          pts.forEach(function(p){
            var d2 = (p.ox - ox) * (p.ox - ox) + (p.oy - oy) * (p.oy - oy);
            var w = Math.exp(-d2 * 12);
            wSum += w;
            valSum += w * p.soft;
          });
          var val = wSum > 0 ? valSum / wSum : 0.5;
          gridPts.push({ gx: gx, gy: gy, val: val });
        }
      }
      /* Draw boundary cells near 0.5 */
      gridPts.forEach(function(gp){
        if (Math.abs(gp.val - 0.5) < 0.06){
          gBound.append("rect")
            .attr("x", gp.gx - step / 2).attr("y", gp.gy - step / 2)
            .attr("width", step).attr("height", step)
            .attr("fill", colGood).attr("opacity", 0.25);
        }
      });
    }

    /* Points */
    var pSel = gPts.selectAll("circle").data(pts);
    pSel.exit().remove();
    var pEnter = pSel.enter().append("circle");

    pEnter.merge(pSel)
      .attr("cx", function(d){ return xScale(d.ox); })
      .attr("cy", function(d){ return yScale(d.oy); })
      .attr("r", function(d){ return d.label >= 0 ? 7 : 4; })
      .attr("stroke", function(d){ return d.label >= 0 ? "#0f1117" : "none"; })
      .attr("stroke-width", function(d){ return d.label >= 0 ? 2 : 0; })
      .attr("fill", function(d){
        if (d.label === 0) return colA;
        if (d.label === 1) return colB;
        if (method === "propagation"){
          /* interpolate from class A to class B through soft label */
          if (d.soft < 0.5) return d3.interpolateRgb(colA, colGood)(d.soft * 2);
          return d3.interpolateRgb(colGood, colB)((d.soft - 0.5) * 2);
        }
        return colMut;
      })
      .attr("opacity", function(d){ return d.label >= 0 ? 1 : 0.75; });

    /* Readout */
    if (method === "supervised"){
      var acc = 0, tot = 0;
      /* accuracy if we classified all by perpendicular bisector */
      var cA = { x: 0, y: 0, n: 0 }, cB = { x: 0, y: 0, n: 0 };
      pts.forEach(function(p){
        if (p.label === 0) { cA.x += p.ox; cA.y += p.oy; cA.n++; }
        else if (p.label === 1) { cB.x += p.ox; cB.y += p.oy; cB.n++; }
      });
      if (cA.n > 0 && cB.n > 0) {
        cA.x /= cA.n; cA.y /= cA.n; cB.x /= cB.n; cB.y /= cB.n;
        var dx = cB.x - cA.x, dy = cB.y - cA.y;
        var mx = (cA.x + cB.x) / 2, my = (cA.y + cB.y) / 2;
        pts.forEach(function(p){
          var proj = (p.ox - mx) * dx + (p.oy - my) * dy;
          var pred = proj > 0 ? 1 : 0;
          if (pred === p.trueClass) acc++;
          tot++;
        });
      }
      d3.select("#ssl-readout").html(
        "method: <b>supervised only</b> (" + (nLabel * 2) + " labeled pts) &mdash; accuracy on all pts: <b>" +
        (tot > 0 ? (100 * acc / tot).toFixed(0) : "?") + "%</b>"
      );
    } else {
      /* propagation accuracy */
      var acc = 0;
      pts.forEach(function(p){
        var pred = p.soft < 0.5 ? 0 : 1;
        if (pred === p.trueClass) acc++;
      });
      d3.select("#ssl-readout").html(
        "method: <b>label propagation</b> (" + (nLabel * 2) + " labeled pts) &mdash; accuracy on all pts: <b>" +
        (100 * acc / pts.length).toFixed(0) + "%</b>"
      );
    }
  }

  /* ---- Animated propagation ---- */
  function animatePropagation(){
    if (animTimer) { clearInterval(animTimer); animTimer = null; }
    /* Reset soft labels */
    pts.forEach(function(p){
      if (p.label === 0) p.soft = 0;
      else if (p.label === 1) p.soft = 1;
      else p.soft = 0.5;
    });
    draw(true, true);

    var step = 0, maxSteps = 25;
    animTimer = setInterval(function(){
      propagateLabels(1);
      draw(true, false);
      step++;
      if (step >= maxSteps) {
        clearInterval(animTimer);
        animTimer = null;
      }
    }, 350);
  }

  /* ---- Method toggle ---- */
  function updateMethod(){
    method = d3.select("#ssl-method").property("value");
    var animBtn = d3.select("#ssl-animate");
    if (method === "propagation"){
      animBtn.style("display", "inline-block");
      propagateLabels(30);
      draw(true, true);
    } else {
      animBtn.style("display", "none");
      /* reset soft labels */
      pts.forEach(function(p){ p.soft = 0.5; });
      draw(false, false);
    }
  }

  /* ---- Wire controls ---- */
  d3.select("#ssl-method").on("change", updateMethod);
  d3.select("#ssl-nlabel").on("change", generate);
  d3.select("#ssl-reset").on("click", generate);
  d3.select("#ssl-animate").on("click", animatePropagation);

  /* ---- Init ---- */
  generate();
})();
