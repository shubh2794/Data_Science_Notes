/* em-algorithm.viz.js — Interactive EM for 2D Gaussian Mixture Models.
   Loaded after data.js / notes.js; uses the global C palette from notes.js
   and d3 from vendor/d3.min.js. */

(function(){
  /* ---- palette & layout ---- */
  const pal = ["#5b9cff", "#ffb454", "#4ade80", "#c084fc"];
  const W = 480, H = 380, m = {t:16, r:16, b:16, l:16};
  const LLW = 200, LLH = 180;

  const svg  = d3.select("#em-svg");
  const llSvg = d3.select("#em-ll");
  if (svg.empty()) return;           // guard: element not on page

  const xSc = d3.scaleLinear().range([m.l, W - m.r]);
  const ySc = d3.scaleLinear().range([H - m.b, m.t]);

  /* ---- state ---- */
  let pts = [], comps = [], K = 3, iter = 0, timer = null, llHistory = [];

  /* ---- helpers ---- */
  function randn(){ let u=0,v=0; while(!u) u=Math.random(); v=Math.random(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }

  /* Generate data from K ground-truth blobs */
  function genData(){
    K = +d3.select("#em-k").property("value");
    d3.select("#em-kval").text(K);
    pts = [];
    const centers = [];
    for(let k=0; k<Math.max(K,3); k++){
      const cx = 0.15 + Math.random()*0.7;
      const cy = 0.15 + Math.random()*0.7;
      const sx = 0.03 + Math.random()*0.06;
      const sy = 0.03 + Math.random()*0.06;
      const angle = Math.random()*Math.PI;
      centers.push({cx,cy,sx,sy,angle});
      const n = 25 + Math.floor(Math.random()*20);
      for(let i=0; i<n; i++){
        let u = randn()*sx, v = randn()*sy;
        const x = cx + u*Math.cos(angle) - v*Math.sin(angle);
        const y = cy + u*Math.sin(angle) + v*Math.cos(angle);
        pts.push({x: Math.max(0.01, Math.min(0.99, x)),
                  y: Math.max(0.01, Math.min(0.99, y)),
                  r: new Array(K).fill(1/K)});   // uniform responsibilities
      }
    }
    xSc.domain([0, 1]);
    ySc.domain([0, 1]);
    initComps();
    iter = 0;
    llHistory = [];
    updateReadout();
  }

  /* Initialize GMM components (random) */
  function initComps(){
    comps = [];
    for(let k=0; k<K; k++){
      comps.push({
        mu: [0.2 + Math.random()*0.6, 0.2 + Math.random()*0.6],
        sigma: [[0.005, 0],[0, 0.005]],   // start with small isotropic covariance
        pi: 1/K
      });
    }
  }

  /* 2x2 matrix inverse */
  function inv2(S){
    const det = S[0][0]*S[1][1] - S[0][1]*S[1][0];
    if(Math.abs(det) < 1e-14) return [[1e6,0],[0,1e6]];
    const id = 1/det;
    return [[ S[1][1]*id, -S[0][1]*id],
            [-S[1][0]*id,  S[0][0]*id]];
  }
  function det2(S){ return S[0][0]*S[1][1] - S[0][1]*S[1][0]; }

  /* Evaluate N(x | mu, Sigma) for 2D */
  function gaussPdf(x, mu, sigma){
    const dx = x.x - mu[0], dy = x.y - mu[1];
    const Si = inv2(sigma);
    const maha = dx*Si[0][0]*dx + dx*Si[0][1]*dy + dy*Si[1][0]*dx + dy*Si[1][1]*dy;
    const d = det2(sigma);
    const norm = 1 / (2*Math.PI * Math.sqrt(Math.max(d, 1e-14)));
    return norm * Math.exp(-0.5 * maha);
  }

  /* ---- E-step ---- */
  function eStep(){
    pts.forEach(p => {
      const raw = comps.map((c,k) => c.pi * gaussPdf(p, c.mu, c.sigma));
      const s = raw.reduce((a,b)=>a+b, 0) || 1e-30;
      p.r = raw.map(v => v/s);
    });
  }

  /* ---- M-step ---- */
  function mStep(){
    const N = pts.length;
    comps.forEach((c,k) => {
      const Nk = pts.reduce((s,p) => s + p.r[k], 0);
      if(Nk < 1e-6){ // collapsed component: re-initialize
        c.mu = [0.2+Math.random()*0.6, 0.2+Math.random()*0.6];
        c.sigma = [[0.005,0],[0,0.005]];
        c.pi = 1/K;
        return;
      }
      // update mean
      let mx=0, my=0;
      pts.forEach(p => { mx += p.r[k]*p.x; my += p.r[k]*p.y; });
      c.mu = [mx/Nk, my/Nk];
      // update covariance
      let s00=0, s01=0, s11=0;
      pts.forEach(p => {
        const dx = p.x - c.mu[0], dy = p.y - c.mu[1];
        s00 += p.r[k]*dx*dx;
        s01 += p.r[k]*dx*dy;
        s11 += p.r[k]*dy*dy;
      });
      const reg = 1e-5;  // regularization to prevent singularity
      c.sigma = [[ s00/Nk + reg, s01/Nk ],
                 [ s01/Nk,       s11/Nk + reg ]];
      // update weight
      c.pi = Nk / N;
    });
  }

  /* ---- log-likelihood ---- */
  function logLikelihood(){
    let ll = 0;
    pts.forEach(p => {
      let s = 0;
      comps.forEach((c,k) => { s += c.pi * gaussPdf(p, c.mu, c.sigma); });
      ll += Math.log(Math.max(s, 1e-300));
    });
    return ll;
  }

  /* ---- draw scatter + ellipses ---- */
  const gEllipses = svg.append("g");
  const gPts      = svg.append("g");
  const gCenters  = svg.append("g");

  function dominantColor(r){
    let best=0, bi=0;
    r.forEach((v,i) => { if(v>best){best=v; bi=i;} });
    // blend: use dominant color at full opacity proportional to responsibility
    return pal[bi % pal.length];
  }
  function dominantOpacity(r){
    return 0.35 + 0.65 * Math.max(...r);
  }

  /* Compute ellipse path from 2x2 covariance matrix (2 std devs) */
  function ellipsePath(mu, sigma, nstd){
    nstd = nstd || 2;
    // eigendecomposition of 2x2 symmetric matrix
    const a = sigma[0][0], b = sigma[0][1], d = sigma[1][1];
    const tr = a + d;
    const det = a*d - b*b;
    const disc = Math.sqrt(Math.max(0, tr*tr/4 - det));
    const l1 = tr/2 + disc;
    const l2 = Math.max(tr/2 - disc, 1e-8);
    // eigenvector for l1
    let angle = 0;
    if(Math.abs(b) > 1e-12) angle = Math.atan2(l1 - a, b);
    else if(a >= d) angle = 0;
    else angle = Math.PI/2;

    const rx = nstd * Math.sqrt(l1);
    const ry = nstd * Math.sqrt(l2);
    const cx = xSc(mu[0]), cy = ySc(mu[1]);
    const deg = angle * 180 / Math.PI;

    // build ellipse as a path using arc
    const cos = Math.cos(angle), sin = Math.sin(angle);
    const nPts = 60;
    const path = [];
    for(let i=0; i<=nPts; i++){
      const t = (i/nPts) * 2 * Math.PI;
      const ex = rx * Math.cos(t);
      const ey = ry * Math.sin(t);
      const px = cx + ex*cos - ey*sin;
      const py = cy - (ex*sin + ey*cos); // minus because y-axis is flipped in SVG
      path.push((i===0?"M":"L") + px.toFixed(1) + "," + py.toFixed(1));
    }
    path.push("Z");
    return path.join(" ");
  }

  function draw(animate){
    const dur = animate ? 400 : 0;

    // points
    const p = gPts.selectAll("circle").data(pts);
    p.enter().append("circle").attr("r", 3.5)
      .merge(p)
      .transition().duration(dur)
      .attr("cx", d => xSc(d.x))
      .attr("cy", d => ySc(d.y))
      .attr("fill", d => dominantColor(d.r))
      .attr("opacity", d => dominantOpacity(d.r));
    p.exit().remove();

    // ellipses (2-sigma contour)
    const ell = gEllipses.selectAll("path.ell").data(comps);
    ell.enter().append("path").attr("class","ell")
      .attr("fill","none").attr("stroke-width", 2).attr("stroke-dasharray","6 3")
      .merge(ell)
      .transition().duration(dur)
      .attr("d", d => ellipsePath(d.mu, d.sigma, 2))
      .attr("stroke", (d,i) => pal[i % pal.length])
      .attr("opacity", 0.85);
    ell.exit().remove();

    // 1-sigma filled ellipse (faint)
    const ell1 = gEllipses.selectAll("path.ell1").data(comps);
    ell1.enter().append("path").attr("class","ell1")
      .attr("stroke","none")
      .merge(ell1)
      .transition().duration(dur)
      .attr("d", d => ellipsePath(d.mu, d.sigma, 1))
      .attr("fill", (d,i) => pal[i % pal.length])
      .attr("opacity", 0.10);
    ell1.exit().remove();

    // centers (cross marks)
    const ctr = gCenters.selectAll("g.ctr").data(comps);
    const ctrEnter = ctr.enter().append("g").attr("class","ctr");
    ctrEnter.append("line").attr("class","cx").attr("stroke-width",2.5);
    ctrEnter.append("line").attr("class","cy").attr("stroke-width",2.5);
    const ctrMerge = ctrEnter.merge(ctr);
    ctrMerge.select("line.cx")
      .transition().duration(dur)
      .attr("x1", d => xSc(d.mu[0])-6).attr("x2", d => xSc(d.mu[0])+6)
      .attr("y1", d => ySc(d.mu[1])).attr("y2", d => ySc(d.mu[1]))
      .attr("stroke", (d,i) => pal[i % pal.length]);
    ctrMerge.select("line.cy")
      .transition().duration(dur)
      .attr("x1", d => xSc(d.mu[0])).attr("x2", d => xSc(d.mu[0]))
      .attr("y1", d => ySc(d.mu[1])-6).attr("y2", d => ySc(d.mu[1])+6)
      .attr("stroke", (d,i) => pal[i % pal.length]);
    ctr.exit().remove();

    drawLL();
    drawLegend();
  }

  /* ---- log-likelihood side plot ---- */
  function drawLL(){
    llSvg.selectAll("*").remove();
    if(llHistory.length < 1) {
      llSvg.append("text").attr("x",LLW/2).attr("y",LLH/2)
        .attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11)
        .text("Log-likelihood plot");
      return;
    }
    const mLL = {t:28, r:12, b:28, l:50};
    const xLL = d3.scaleLinear().domain([0, Math.max(llHistory.length-1, 1)]).range([mLL.l, LLW-mLL.r]);
    const yLL = d3.scaleLinear().domain(d3.extent(llHistory)).nice().range([LLH-mLL.b, mLL.t]);

    // axes
    llSvg.append("g").attr("class","axis").attr("transform",`translate(0,${LLH-mLL.b})`)
      .call(d3.axisBottom(xLL).ticks(Math.min(llHistory.length, 6)).tickFormat(d3.format("d")));
    llSvg.append("g").attr("class","axis").attr("transform",`translate(${mLL.l},0)`)
      .call(d3.axisLeft(yLL).ticks(4).tickFormat(d => d.toFixed(0)));

    // labels
    llSvg.append("text").attr("x",LLW/2).attr("y",LLH-3)
      .attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("iteration");
    llSvg.append("text").attr("x", mLL.l - 6).attr("y", mLL.t - 8)
      .attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10).text("log L");

    // line
    const line = d3.line().x((d,i)=>xLL(i)).y(d=>yLL(d));
    llSvg.append("path").datum(llHistory)
      .attr("fill","none").attr("stroke",C.good).attr("stroke-width",2)
      .attr("d", line);

    // dots
    llSvg.selectAll("circle.lldot").data(llHistory).enter()
      .append("circle").attr("class","lldot")
      .attr("cx",(d,i)=>xLL(i)).attr("cy",d=>yLL(d))
      .attr("r",3).attr("fill",C.good);
  }

  /* ---- legend ---- */
  function drawLegend(){
    const leg = d3.select("#em-legend");
    leg.html(comps.map((c,i) =>
      `<span><span class="dot" style="background:${pal[i%pal.length]}"></span>` +
      `Component ${i+1} (&pi;=${c.pi.toFixed(2)})</span>`
    ).join(""));
  }

  function updateReadout(){
    const ll = llHistory.length > 0 ? llHistory[llHistory.length-1].toFixed(1) : "—";
    d3.select("#em-iter").html(`iter <b>${iter}</b> · log L &asymp; <b>${ll}</b>`);
  }

  /* ---- step: one E+M cycle ---- */
  function step(){
    eStep();
    mStep();
    iter++;
    llHistory.push(logLikelihood());
    draw(true);
    updateReadout();
  }

  /* ---- controls ---- */
  d3.select("#em-step").on("click", step);

  d3.select("#em-auto").on("click", function(){
    if(timer){ clearInterval(timer); timer=null; d3.select(this).text("Auto-play"); return; }
    d3.select(this).text("Pause");
    let n=0;
    timer = setInterval(()=>{
      step();
      n++;
      if(n >= 40){ clearInterval(timer); timer=null; d3.select("#em-auto").text("Auto-play"); }
    }, 500);
  });

  d3.select("#em-reset").on("click", function(){
    if(timer){ clearInterval(timer); timer=null; d3.select("#em-auto").text("Auto-play"); }
    genData();
    draw(false);
  });

  d3.select("#em-k").on("input", function(){
    if(timer){ clearInterval(timer); timer=null; d3.select("#em-auto").text("Auto-play"); }
    genData();
    draw(false);
  });

  /* ---- init ---- */
  genData();
  draw(false);
})();
