/* variational-inference.viz.js — Mean-field CAVI on a 2D Gaussian posterior.
   Loaded after data.js / notes.js; uses the global C colour palette and d3.  */

(function(){
  /* ------------------------------------------------------------------ */
  /*  Colour constants (from notes.js global C)                         */
  /* ------------------------------------------------------------------ */
  var blue  = C.A,       // #5b9cff — true posterior
      amber = C.B,       // #ffb454 — q approximation
      good  = C.good,    // #4ade80
      bad   = C.bad,     // #f87171
      ink   = C.ink,     // #e6e9ef
      muted = C.muted,   // #9aa3b2
      line  = C.line;    // #2a2f3a

  /* ------------------------------------------------------------------ */
  /*  Math helpers                                                      */
  /* ------------------------------------------------------------------ */

  /* 2-D Gaussian pdf value (un-normalised is fine for contours). */
  function gauss2d(x, y, mx, my, sx, sy, rho){
    var dx = (x - mx)/sx, dy = (y - my)/sy;
    var z = dx*dx - 2*rho*dx*dy + dy*dy;
    return Math.exp(-z / (2*(1 - rho*rho)));
  }

  /* Axis-aligned (mean-field) 2-D Gaussian — rho forced to 0. */
  function gaussMF(x, y, mx, my, sx, sy){
    var dx = (x - mx)/sx, dy = (y - my)/sy;
    return Math.exp(-0.5*(dx*dx + dy*dy));
  }

  /* Analytic ELBO for Gaussian q vs Gaussian p.
     p = N(mu_p, Sigma_p),  q = N(mu_q, diag(s1^2, s2^2)).
     ELBO = E_q[log p(z)] - E_q[log q(z)]
          = -0.5 { log det Sigma_p + tr(Sigma_p^{-1} Sigma_q)
                    + (mu_q - mu_p)^T Sigma_p^{-1} (mu_q - mu_p) - 2 }
            + 0.5 { log(s1^2) + log(s2^2) + 2 }
     For 2D with Sigma_p parameterised by (sx, sy, rho).  */
  function elbo(muPx, muPy, sx, sy, rho, mqx, mqy, sq1, sq2){
    var det = sx*sx * sy*sy * (1 - rho*rho);
    /* Sigma_p inverse for 2d: */
    var ixx = 1 / (sx*sx * (1 - rho*rho));
    var iyy = 1 / (sy*sy * (1 - rho*rho));
    var ixy = rho / (sx * sy * (1 - rho*rho));
    /* trace term: tr(Sigma_p^{-1} diag(sq1^2, sq2^2)) */
    var tr = ixx * sq1*sq1 + iyy * sq2*sq2;
    /* quadratic term */
    var dx = mqx - muPx, dy = mqy - muPy;
    var quad = ixx*dx*dx + iyy*dy*dy - 2*ixy*dx*dy;
    /* log det */
    var logDet = Math.log(det);
    /* Cross-entropy: E_q[log p] */
    var cross = -0.5*(logDet + tr + quad + 2);
    /* Entropy: H(q) = 0.5 log((2 pi e)^2 s1^2 s2^2) */
    var entropy = 0.5*(2*Math.log(2*Math.PI*Math.E) + Math.log(sq1*sq1) + Math.log(sq2*sq2));
    return cross + entropy;
  }

  /* Analytic CAVI updates for mean-field Gaussian q vs full Gaussian p.
     Optimal q1(z1) and q2(z2) given the other.
     p(z) = N([mx,my], [[sx^2, rho sx sy],[rho sx sy, sy^2]])
     q(z) = q1(z1) q2(z2), each Gaussian.

     Optimal q1: mean = mx + rho*(sx/sy)*(E_q2[z2] - my),
                 var  = sx^2*(1 - rho^2)
     Optimal q2: mean = my + rho*(sy/sx)*(E_q1[z1] - mx),
                 var  = sy^2*(1 - rho^2)                     */
  function caviStep(muPx, muPy, sx, sy, rho, mqx, mqy, sq1, sq2){
    /* Update factor 1 */
    var newMq1 = muPx + rho*(sx/sy)*(mqy - muPy);
    var newSq1 = sx * Math.sqrt(1 - rho*rho);
    /* Update factor 2 (using the NEW q1 mean) */
    var newMq2 = muPy + rho*(sy/sx)*(newMq1 - muPx);
    var newSq2 = sy * Math.sqrt(1 - rho*rho);
    return { mqx: newMq1, mqy: newMq2, sq1: newSq1, sq2: newSq2 };
  }

  /* ------------------------------------------------------------------ */
  /*  Contour SVG (left panel)                                          */
  /* ------------------------------------------------------------------ */
  var W = 420, H = 420, pad = 40;
  var svg = d3.select("#vi-svg");

  /* Value range for plotting */
  var lo = -4, hi = 4;
  var xS = d3.scaleLinear().domain([lo, hi]).range([pad, W - pad]);
  var yS = d3.scaleLinear().domain([lo, hi]).range([H - pad, pad]);

  /* Axes */
  svg.append("g").attr("class","axis").attr("transform","translate(0,"+(H-pad)+")")
    .call(d3.axisBottom(xS).ticks(5));
  svg.append("g").attr("class","axis").attr("transform","translate("+pad+",0)")
    .call(d3.axisLeft(yS).ticks(5));
  svg.append("text").attr("x",W/2).attr("y",H-4).attr("text-anchor","middle").attr("fill",muted).attr("font-size",12).text("z₁");
  svg.append("text").attr("x",12).attr("y",H/2).attr("text-anchor","middle").attr("fill",muted).attr("font-size",12)
    .attr("transform","rotate(-90,12,"+H/2+")").text("z₂");

  /* Groups for layering */
  var gTrueContour = svg.append("g");
  var gQContour    = svg.append("g");
  var gTrueMean    = svg.append("g");
  var gQMean       = svg.append("g");

  /* Contour grid */
  var gridN = 80;
  function makeGrid(evalFn){
    var vals = [];
    var step = (hi - lo)/gridN;
    for(var j=0; j<gridN; j++){
      for(var i=0; i<gridN; i++){
        vals.push(evalFn(lo + (i+0.5)*step, lo + (j+0.5)*step));
      }
    }
    return vals;
  }

  function drawContours(group, vals, color, opacity){
    var step = (hi - lo)/gridN;
    var thresholds = d3.range(0.05, 1.0, 0.12);
    var contours = d3.contours().size([gridN, gridN]).thresholds(thresholds)(vals);
    var t = d3.geoTransform({ point: function(px, py){
      this.stream.point(xS(lo + (px+0.5)*step), yS(lo + (py+0.5)*step));
    }});
    var path = d3.geoPath().projection(t);
    group.selectAll("path").remove();
    group.selectAll("path").data(contours).enter().append("path")
      .attr("d", path)
      .attr("fill","none")
      .attr("stroke", color)
      .attr("stroke-width", 1.4)
      .attr("stroke-opacity", opacity);
  }

  /* ------------------------------------------------------------------ */
  /*  ELBO plot (right panel)                                           */
  /* ------------------------------------------------------------------ */
  var eW = 260, eH = 200, eP = {t:24,r:16,b:32,l:48};
  var eSvg = d3.select("#vi-elbo-svg");
  eSvg.append("text").attr("x",eP.l + (eW-eP.l-eP.r)/2).attr("y",14).attr("text-anchor","middle")
    .attr("fill",ink).attr("font-size",13).attr("font-weight",600).text("ELBO per iteration");
  var eX = d3.scaleLinear().range([eP.l, eW - eP.r]);
  var eY = d3.scaleLinear().range([eH - eP.b, eP.t]);
  var eAxisX = eSvg.append("g").attr("class","axis").attr("transform","translate(0,"+(eH-eP.b)+")");
  var eAxisY = eSvg.append("g").attr("class","axis").attr("transform","translate("+eP.l+",0)");
  var eLine = eSvg.append("path").attr("fill","none").attr("stroke",amber).attr("stroke-width",2.5);
  var eDots = eSvg.append("g");
  eSvg.append("text").attr("x",eP.l+(eW-eP.l-eP.r)/2).attr("y",eH-4).attr("text-anchor","middle")
    .attr("fill",muted).attr("font-size",11).text("iteration");

  function drawElbo(history){
    if(history.length === 0) return;
    var maxIter = Math.max(history.length - 1, 1);
    eX.domain([0, maxIter]);
    var ext = d3.extent(history);
    var margin = (ext[1] - ext[0]) * 0.15 || 1;
    eY.domain([ext[0] - margin, ext[1] + margin]);
    eAxisX.call(d3.axisBottom(eX).ticks(Math.min(maxIter, 8)).tickFormat(d3.format("d")));
    eAxisY.call(d3.axisLeft(eY).ticks(5).tickFormat(d3.format(".2f")));
    var lineGen = d3.line().x(function(d,i){return eX(i);}).y(function(d){return eY(d);});
    eLine.attr("d", lineGen(history));
    var dots = eDots.selectAll("circle").data(history);
    dots.enter().append("circle").attr("r",4).attr("fill",amber).attr("stroke","#0f1117").attr("stroke-width",1)
      .merge(dots).attr("cx",function(d,i){return eX(i);}).attr("cy",function(d){return eY(d);});
    dots.exit().remove();
  }

  /* ------------------------------------------------------------------ */
  /*  State                                                             */
  /* ------------------------------------------------------------------ */
  /* True posterior: N([0,0], [[1, rho],[rho, 1]]) — unit variances for simplicity */
  var muPx = 0, muPy = 0, sigX = 1.4, sigY = 1.4;
  var rho = 0.70;
  /* q parameters (mean-field: diagonal covariance) */
  var mqx, mqy, sq1, sq2;
  var elboHist = [];
  var timer = null;
  var iteration = 0;

  function initQ(){
    /* Start q far from the optimum so the animation is visible */
    mqx = -2.5; mqy = 2.0; sq1 = 0.5; sq2 = 2.0;
    elboHist = [elbo(muPx, muPy, sigX, sigY, rho, mqx, mqy, sq1, sq2)];
    iteration = 0;
  }

  function draw(){
    /* True posterior contours */
    var trueVals = makeGrid(function(x,y){ return gauss2d(x,y, muPx, muPy, sigX, sigY, rho); });
    drawContours(gTrueContour, trueVals, blue, 0.7);

    /* q contours */
    var qVals = makeGrid(function(x,y){ return gaussMF(x,y, mqx, mqy, sq1, sq2); });
    drawContours(gQContour, qVals, amber, 0.85);

    /* True posterior mean marker */
    gTrueMean.selectAll("*").remove();
    gTrueMean.append("circle").attr("cx",xS(muPx)).attr("cy",yS(muPy)).attr("r",5)
      .attr("fill",blue).attr("stroke","#0f1117").attr("stroke-width",1.5);
    gTrueMean.append("text").attr("x",xS(muPx)+10).attr("y",yS(muPy)-8)
      .attr("fill",blue).attr("font-size",11).attr("font-weight",600).text("p mean");

    /* q mean marker */
    gQMean.selectAll("*").remove();
    gQMean.append("circle").attr("cx",xS(mqx)).attr("cy",yS(mqy)).attr("r",5)
      .attr("fill",amber).attr("stroke","#0f1117").attr("stroke-width",1.5);
    gQMean.append("text").attr("x",xS(mqx)+10).attr("y",yS(mqy)+14)
      .attr("fill",amber).attr("font-size",11).attr("font-weight",600).text("q mean");

    /* ELBO plot */
    drawElbo(elboHist);

    /* Readout */
    var klVal = Math.log(sigX*sigY*Math.sqrt(1-rho*rho)) - Math.log(sq1*sq2)
      + 0.5*( (sq1*sq1)/(sigX*sigX*(1-rho*rho)) + (sq2*sq2)/(sigY*sigY*(1-rho*rho))
        + ((mqx-muPx)*(mqx-muPx))/(sigX*sigX*(1-rho*rho))
        + ((mqy-muPy)*(mqy-muPy))/(sigY*sigY*(1-rho*rho))
        - 2*rho*(mqx-muPx)*(mqy-muPy)/(sigX*sigY*(1-rho*rho)) - 2 );
    d3.select("#vi-readout").html(
      "Iteration <b>" + iteration + "</b> &nbsp;|&nbsp; " +
      "q mean = (<b>" + mqx.toFixed(2) + "</b>, <b>" + mqy.toFixed(2) + "</b>) &nbsp;|&nbsp; " +
      "q std = (<b>" + sq1.toFixed(2) + "</b>, <b>" + sq2.toFixed(2) + "</b>) &nbsp;|&nbsp; " +
      "ELBO = <b>" + elboHist[elboHist.length-1].toFixed(3) + "</b> &nbsp;|&nbsp; " +
      "KL = <b>" + klVal.toFixed(3) + "</b>"
    );
  }

  function step(){
    var res = caviStep(muPx, muPy, sigX, sigY, rho, mqx, mqy, sq1, sq2);
    mqx = res.mqx; mqy = res.mqy; sq1 = res.sq1; sq2 = res.sq2;
    iteration++;
    elboHist.push(elbo(muPx, muPy, sigX, sigY, rho, mqx, mqy, sq1, sq2));
    draw();
  }

  /* ------------------------------------------------------------------ */
  /*  Controls                                                          */
  /* ------------------------------------------------------------------ */
  d3.select("#vi-rho").on("input", function(){
    rho = +this.value;
    d3.select("#vi-rho-val").text(rho.toFixed(2));
    if(timer){ clearInterval(timer); timer = null; }
    initQ();
    draw();
  });

  d3.select("#vi-step").on("click", function(){
    if(timer){ clearInterval(timer); timer = null; }
    step();
  });

  d3.select("#vi-run").on("click", function(){
    if(timer){ clearInterval(timer); timer = null; return; }
    timer = setInterval(function(){
      step();
      if(iteration >= 15){ clearInterval(timer); timer = null; }
    }, 500);
  });

  d3.select("#vi-reset").on("click", function(){
    if(timer){ clearInterval(timer); timer = null; }
    initQ();
    draw();
  });

  /* ------------------------------------------------------------------ */
  /*  Initial draw                                                      */
  /* ------------------------------------------------------------------ */
  initQ();
  draw();
})();
