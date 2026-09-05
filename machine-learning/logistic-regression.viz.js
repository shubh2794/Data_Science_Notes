/* logistic-regression.viz.js — extracted from logistic-regression.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ---------- 2 · convex vs non-convex loss landscape ---------- */
(function(){
  const svg = d3.select("#loss-svg"), W=680, H=290;
  const m = {t:30, r:16, b:38, l:48}, gapX = 60;
  const pw = (W - m.l - m.r - gapX) / 2, ph = H - m.t - m.b;
  const clamp = (v,a,b) => Math.max(a, Math.min(b, v));
  const sig = z => 1/(1+Math.exp(-z));

  /* nine points, one of them mislabelled — enough to make MSE∘σ non-convex */
  const D = [{x:1,y:0},{x:2,y:0},{x:3,y:0},{x:4,y:0},
             {x:6,y:1},{x:7,y:1},{x:8,y:1},{x:9,y:1},
             {x:-3,y:1}];
  const B0 = -5, n = D.length, B_LO = -3, B_HI = 3;

  const logloss = b => D.reduce((s,d)=>{ const p = sig(B0+b*d.x);
      return s - (d.y*Math.log(p+1e-12) + (1-d.y)*Math.log(1-p+1e-12)); },0)/n;
  const msesig  = b => D.reduce((s,d)=> s + (sig(B0+b*d.x)-d.y)**2, 0)/n;
  const gLog = b => D.reduce((s,d)=> s + (sig(B0+b*d.x)-d.y)*d.x, 0)/n;
  const gMse = b => D.reduce((s,d)=>{ const p = sig(B0+b*d.x); return s + 2*(p-d.y)*p*(1-p)*d.x; },0)/n;

  const xs = d3.scaleLinear().domain([B_LO,B_HI]).range([0,pw]);
  const curve = f => d3.range(B_LO, B_HI+1e-9, 0.02).map(b=>[b, f(b)]);
  const cLog = curve(logloss), cMse = curve(msesig);

  const panels = [
    { key:"log", g:svg.append("g").attr("transform",`translate(${m.l},${m.t})`),
      title:"log-loss  ·  convex", loss:logloss, grad:gLog, lr:0.05, data:cLog,
      ys:d3.scaleLinear().domain([0, d3.max(cLog,d=>d[1])*1.05]).range([ph,0]), col:C.good, b:-1.5 },
    { key:"mse", g:svg.append("g").attr("transform",`translate(${m.l+pw+gapX},${m.t})`),
      title:"MSE ∘ σ  ·  non-convex", loss:msesig, grad:gMse, lr:0.5, data:cMse,
      ys:d3.scaleLinear().domain([d3.min(cMse,d=>d[1])*0.9, d3.max(cMse,d=>d[1])*1.04]).range([ph,0]), col:C.bad, b:-1.5 }
  ];

  let timer = null;

  panels.forEach(P=>{
    P.g.append("text").attr("x",0).attr("y",-12).attr("font-size",12).attr("font-weight",600)
       .attr("fill", P.col).text(P.title);
    P.g.append("g").attr("class","axis").attr("transform",`translate(0,${ph})`).call(d3.axisBottom(xs).ticks(5));
    P.g.append("g").attr("class","axis").call(d3.axisLeft(P.ys).ticks(4));
    P.g.append("text").attr("x",pw/2).attr("y",ph+30).attr("text-anchor","middle")
       .attr("font-size",11).attr("fill",C.muted).text("β₁");
    P.g.append("path").attr("fill","none").attr("stroke",P.col).attr("stroke-width",2.2)
       .attr("d", d3.line().x(d=>xs(d[0])).y(d=>P.ys(d[1]))(P.data));
    P.drop = P.g.append("line").attr("stroke",C.B).attr("stroke-width",1).attr("stroke-dasharray","3 3");
    P.dot  = P.g.append("circle").attr("class","dragpt").attr("r",10)
       .attr("fill",C.B).attr("stroke","#0f1117").attr("stroke-width",2).style("cursor","grab")
       .call(d3.drag()
         .on("start", function(){ stop(); d3.select(this).style("cursor","grabbing"); })
         .on("drag",  function(ev){ const b = clamp(xs.invert(ev.x), B_LO, B_HI);
                                    panels.forEach(Q=>Q.b=b); render(); })
         .on("end",   function(){ d3.select(this).style("cursor","grab"); }));
  });

  function stop(){ if(timer){ timer.stop(); timer=null; } }

  function render(){
    panels.forEach(P=>{
      const L = P.loss(P.b);
      P.dot.attr("cx", xs(P.b)).attr("cy", P.ys(L));
      P.drop.attr("x1", xs(P.b)).attr("x2", xs(P.b)).attr("y1", P.ys(L)).attr("y2", ph);
    });
    const [A,B] = panels;
    d3.select("#loss-read").html(
      `log-loss: β₁ = <b>${A.b.toFixed(2)}</b>, L = <b>${A.loss(A.b).toFixed(3)}</b>, |∂L/∂β₁| = <b>${Math.abs(A.grad(A.b)).toFixed(3)}</b>` +
      ` &nbsp;&nbsp;|&nbsp;&nbsp; MSE ∘ σ: β₁ = <b>${B.b.toFixed(2)}</b>, L = <b>${B.loss(B.b).toFixed(3)}</b>, |∂L/∂β₁| = <b>${Math.abs(B.grad(B.b)).toFixed(3)}</b>`);
  }

  d3.select("#loss-run").on("click", ()=>{
    stop();
    const traj = panels.map(P=>{
      let b = P.b; const t=[b];
      for(let i=0;i<600;i++){ b -= P.lr*P.grad(b); if(!isFinite(b)) break; t.push(clamp(b,B_LO,B_HI)); }
      return t;
    });
    let k = 0;
    timer = d3.interval(()=>{
      k += 4;
      panels.forEach((P,i)=>{ P.b = traj[i][Math.min(k, traj[i].length-1)]; });
      render();
      if(k >= d3.max(traj,t=>t.length)) stop();
    }, 16);
  });
  d3.select("#loss-reset").on("click", ()=>{ stop(); panels.forEach(P=>P.b=-1.5); render(); });
  render();
})();

/* ---------- 3 · logistic fit + draggable decision threshold ---------- */
(function(){
  const svg = d3.select("#logi-svg"), W=680, H=380, m={t:22, r:58, b:40, l:52};
  const x = d3.scaleLinear().domain([0,10]).range([m.l, W-m.r]);
  const y = d3.scaleLinear().domain([-0.10, 1.10]).range([H-m.b, m.t]);
  const clamp = (v,a,b) => Math.max(a, Math.min(b, v));
  const sig = z => 1/(1+Math.exp(-z));
  const SC = 2.5, CT = 5;             // fixed standardisation, keeps GD well conditioned
  const LAM = 1e-4;                   // tiny L2 — stops the MLE diverging when classes separate
  const NPC = 13;                     // points per class

  let sep = +d3.select("#logi-sep").property("value");
  let t = 0.5, pts = [], B0 = 0, B1 = 1;

  const gauss = () => (Math.random()+Math.random()+Math.random()+Math.random()+Math.random()+Math.random()-3);
  function sample(){
    pts = [];
    for(let i=0;i<NPC;i++) pts.push({ x: clamp(5 - sep + gauss(), 0.3, 9.7), y: 0 });
    for(let i=0;i<NPC;i++) pts.push({ x: clamp(5 + sep + gauss(), 0.3, 9.7), y: 1 });
  }
  /* honest fit: batch gradient descent on log-loss + L2, in standardised coordinates */
  function fit(){
    const N = pts.length, xz = pts.map(p=>(p.x-CT)/SC), yv = pts.map(p=>p.y);
    let b0 = 0, b1 = 0;
    for(let it=0; it<2000; it++){
      let g0=0, g1=0;
      for(let i=0;i<N;i++){ const e = sig(b0 + b1*xz[i]) - yv[i]; g0 += e; g1 += e*xz[i]; }
      g0 /= N; g1 = g1/N + LAM*b1;
      b0 -= 0.8*g0; b1 -= 0.8*g1;
    }
    B1 = b1/SC; B0 = b0 - b1*CT/SC;   // back to raw-x coordinates
  }

  const gAx = svg.append("g");
  gAx.append("g").attr("class","axis").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x).ticks(6));
  gAx.append("g").attr("class","axis").attr("transform",`translate(${m.l},0)`)
     .call(d3.axisLeft(y).tickValues([0,0.25,0.5,0.75,1]).tickFormat(d3.format(".2f")));
  [0,1].forEach(v => gAx.append("line").attr("x1",x(0)).attr("x2",x(10))
     .attr("y1",y(v)).attr("y2",y(v)).attr("stroke",C.line).attr("stroke-width",1));
  svg.append("text").attr("transform",`translate(14,${(m.t + H - m.b)/2}) rotate(-90)`)
     .attr("text-anchor","middle").attr("font-size",11).attr("fill",C.muted).text("P(y = 1 | x)");
  svg.append("text").attr("x", W-m.r+4).attr("y", H-m.b+16).attr("font-size",11).attr("fill",C.muted).text("x");

  const posRect = svg.append("rect").attr("fill",C.A).attr("opacity",0.07);
  const curve   = svg.append("path").attr("fill","none").attr("stroke",C.good).attr("stroke-width",2.6);
  const tLine   = svg.append("line").attr("stroke",C.B).attr("stroke-width",1.8).attr("stroke-dasharray","6 4");
  const bLine   = svg.append("line").attr("stroke",C.B).attr("stroke-width",1.8).attr("stroke-dasharray","6 4");
  const bLabel  = svg.append("text").attr("font-size",11).attr("fill",C.B).attr("text-anchor","middle");
  const gPts    = svg.append("g");
  const handle  = svg.append("circle").attr("class","dragpt").attr("r",11)
      .attr("cx", W-m.r+20).attr("fill",C.B).attr("stroke","#0f1117").attr("stroke-width",2).style("cursor","grab")
      .call(d3.drag()
        .on("start", function(){ d3.select(this).style("cursor","grabbing"); })
        .on("drag",  function(ev){ t = clamp(y.invert(ev.y), 0.02, 0.98); render(); })
        .on("end",   function(){ d3.select(this).style("cursor","grab"); }));
  const tLabel = svg.append("text").attr("x", W-m.r+20).attr("font-size",10)
      .attr("fill",C.B).attr("text-anchor","middle");

  function render(){
    const zt = Math.log(t/(1-t));
    const xstar = Math.abs(B1) < 1e-6 ? NaN : (zt - B0)/B1;

    curve.attr("d", d3.line().x(d=>x(d)).y(d=>y(sig(B0+B1*d)))(d3.range(0,10.001,0.05)));

    if(isFinite(xstar)){
      /* shade the predicted-positive side: right of x* if β₁ > 0, left of it if β₁ < 0 */
      const x0 = clamp(xstar, 0, 10);
      const a = B1 > 0 ? x(x0) : x(0), b = B1 > 0 ? x(10) : x(x0);
      posRect.attr("x", a).attr("y", m.t).attr("width", Math.max(0, b-a)).attr("height", (H-m.b)-m.t).attr("opacity",0.07);
    } else posRect.attr("opacity",0);

    tLine.attr("x1", x(0)).attr("x2", x(10)).attr("y1", y(t)).attr("y2", y(t));
    handle.attr("cy", y(t));
    tLabel.attr("y", y(t)+24).text("t = "+t.toFixed(2));

    if(isFinite(xstar) && xstar >= 0 && xstar <= 10){
      bLine.attr("opacity",1).attr("x1",x(xstar)).attr("x2",x(xstar)).attr("y1",m.t).attr("y2",y(-0.10));
      bLabel.attr("opacity",1).attr("x", x(xstar)).attr("y", m.t-6).text("x* = "+xstar.toFixed(2));
    } else { bLine.attr("opacity",0); bLabel.attr("opacity",0); }

    let TP=0, FP=0, TN=0, FN=0;
    pts.forEach(p=>{
      p.p = sig(B0 + B1*p.x);
      p.pred = p.p >= t ? 1 : 0;
      if(p.pred===1 && p.y===1) TP++; else if(p.pred===1 && p.y===0) FP++;
      else if(p.pred===0 && p.y===0) TN++; else FN++;
    });

    gPts.selectAll("circle").data(pts).join("circle")
      .attr("r",6.5)
      .attr("cx", d=>x(d.x)).attr("cy", d=>y(d.y))
      .attr("fill", d=> d.y===1 ? C.A : C.muted)
      .attr("stroke", d=> d.pred===d.y ? "#0f1117" : C.bad)
      .attr("stroke-width", d=> d.pred===d.y ? 1.5 : 2.6);

    const N = pts.length;
    const acc  = (TP+TN)/N;
    const prec = (TP+FP) ? TP/(TP+FP) : NaN;
    const rec  = (TP+FN) ? TP/(TP+FN) : NaN;
    const f = v => isNaN(v) ? "—" : v.toFixed(3);
    d3.select("#logi-read").html(
      `fit: σ(<b>${B0.toFixed(2)}</b> + <b>${B1.toFixed(2)}</b>·x) &nbsp;·&nbsp; |β₁| = <b>${Math.abs(B1).toFixed(2)}</b> (steepness) &nbsp;·&nbsp; ` +
      `threshold t = <b>${t.toFixed(2)}</b> &nbsp;·&nbsp; boundary x* = <b>${isFinite(xstar)?xstar.toFixed(2):"—"}</b><br>` +
      `TP <b>${TP}</b> &nbsp; FP <b>${FP}</b> &nbsp; TN <b>${TN}</b> &nbsp; FN <b>${FN}</b> &nbsp;·&nbsp; ` +
      `accuracy <b>${f(acc)}</b> &nbsp;·&nbsp; precision <b>${f(prec)}</b> &nbsp;·&nbsp; recall <b>${f(rec)}</b>`);
  }

  function rebuild(){ sample(); fit(); render(); }
  d3.select("#logi-sep").on("input", function(){ sep = +this.value; rebuild(); });
  d3.select("#logi-resample").on("click", rebuild);
  d3.select("#logi-half").on("click", ()=>{ t = 0.5; render(); });
  rebuild();
})();
