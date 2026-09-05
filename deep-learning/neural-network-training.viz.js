/* neural-network-training.viz.js — extracted from neural-network-training.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ───────────────────────── 04 · gradient descent on a 1-D loss ───────────────────────── */
(function(){
  const C={accent:"#f87171",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#nnt-svg"),W=640,H=280;
  const a=0.18,c=1.2, f=x=>a*(x-c)*(x-c)+0.4, grad=x=>2*a*(x-c);
  const xmin=-6,xmax=9, px=40,py=24,pw=W-80,ph=210;
  const X=v=>px+(v-xmin)/(xmax-xmin)*pw, Xc=v=>Math.max(px-12,Math.min(px+pw+12,X(v)));
  const ymax=Math.max(f(xmin),f(xmax)), Y=v=>py+ph-(Math.min(v,ymax)/ymax)*ph;
  let x=7, traj=[7];
  function draw(){
    svg.selectAll("*").remove();
    const xs=d3.range(xmin,xmax+0.01,0.1);
    svg.append("path").attr("d",d3.line().x(d=>X(d)).y(d=>Y(f(d)))(xs)).attr("fill","none").attr("stroke","#3a4150").attr("stroke-width",2);
    svg.append("circle").attr("cx",X(c)).attr("cy",Y(f(c))).attr("r",3).attr("fill",C.good);
    svg.append("text").attr("x",X(c)).attr("y",Y(f(c))+18).attr("text-anchor","middle").attr("fill",C.good).attr("font-size",9).text("min");
    traj.forEach((tx,i)=>svg.append("circle").attr("cx",Xc(tx)).attr("cy",Y(f(tx))).attr("r",3).attr("fill",C.a2).attr("opacity",0.25+0.7*i/Math.max(traj.length,1)));
    svg.append("circle").attr("cx",Xc(x)).attr("cy",Y(f(x))).attr("r",7).attr("fill",C.accent).attr("stroke","#0f1117").attr("stroke-width",1.5);
    svg.append("text").attr("x",px).attr("y",16).attr("fill",C.muted).attr("font-size",11).text("loss surface (1-D)");
    const lr=+d3.select("#nnt-lr").property("value");
    const div=Math.abs(x)>25, conv=Math.abs(grad(x))<0.02;
    d3.select("#nnt-read").html(`lr <b>${lr.toFixed(1)}</b> · x = <b>${x.toFixed(2)}</b> · loss = <b>${(div?'∞':f(x).toFixed(3))}</b> ${div?'<span style="color:#f87171">— diverging! lr too high</span>':conv?'<span style="color:#4ade80">— converged ✓</span>':''}`);
  }
  d3.select("#nnt-step").on("click",()=>{ const lr=+d3.select("#nnt-lr").property("value"); x=x-lr*grad(x); traj.push(x); if(traj.length>40)traj.shift(); draw(); });
  d3.select("#nnt-reset").on("click",()=>{ x=7; traj=[7]; draw(); });
  d3.select("#nnt-lr").on("input",draw);
  draw();
})();

/* ───────────────────────── 06 · learning-rate schedule explorer ───────────────────────── */
(function(){
  const svg=d3.select("#sched-svg"), W=640, H=250;
  const m={t:26,r:24,b:36,l:58}, iw=W-m.l-m.r, ih=H-m.t-m.b;
  const g=svg.append("g").attr("transform",`translate(${m.l},${m.t})`);
  const T=1000, PEAK=1;                       // rates drawn relative to the peak
  const x=d3.scaleLinear().domain([1,T]).range([0,iw]);
  const y=d3.scaleLinear().domain([0,1.08]).range([ih,0]);
  const xAxisG=g.append("g").attr("transform",`translate(0,${ih})`).attr("color",C.muted);
  const yAxisG=g.append("g").attr("color",C.muted);
  const warmBand=g.append("rect").attr("y",0).attr("height",ih).attr("fill",C.B).attr("fill-opacity",.10);
  const area=g.append("path").attr("fill",C.A).attr("fill-opacity",.13).attr("stroke","none");
  const path=g.append("path").attr("fill","none").attr("stroke",C.A).attr("stroke-width",2.5);
  const marker=g.append("line").attr("y1",0).attr("y2",ih).attr("stroke",C.good).attr("stroke-dasharray","3 3");
  const dot=g.append("circle").attr("r",5).attr("fill",C.good);
  g.append("text").attr("x",iw).attr("y",ih+30).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10.5).text("training step →");
  g.append("text").attr("transform","rotate(-90)").attr("x",-ih/2).attr("y",-42).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10.5).text("η / η_max");
  const warmLabel=g.append("text").attr("y",-8).attr("fill",C.B).attr("font-size",10).text("warmup");

  const MILE=[0.5,0.75,0.9];
  function lr(t,kind,Tw){
    if(Tw>0 && t<Tw) return PEAK*t/Tw;
    const p=(t-Tw)/Math.max(T-Tw,1);          // progress after warmup, 0..1
    if(kind==="constant") return PEAK;
    if(kind==="cosine")   return PEAK*0.5*(1+Math.cos(Math.PI*Math.min(p,1)));
    if(kind==="onecycle") return PEAK*Math.max(1-Math.min(p,1),0.01);
    if(kind==="step")     return PEAK*Math.pow(0.1, MILE.filter(mi=>t/T>=mi).length);
    if(kind==="invsqrt")  return PEAK*Math.sqrt(Math.max(Tw,1)/Math.max(t,Math.max(Tw,1)));
    return PEAK;
  }
  const NAME={cosine:"warmup + cosine decay",onecycle:"one-cycle (triangular)",step:"step decay ×0.1 at 50/75/90%",invsqrt:"inverse square root",constant:"constant"};
  const RULE={
    cosine:"ηₜ = ½·η_max·(1 + cos(π·p))  after linear warmup",
    onecycle:"linear ramp to η_max, then linear anneal to ≈0",
    step:"ηₜ = η_max · 0.1^(milestones passed)",
    invsqrt:"ηₜ = η_max · √(T_w / t)",
    constant:"ηₜ = η_max"
  };

  function draw(){
    const kind=d3.select("#sched-kind").property("value");
    let wpct=+d3.select("#sched-warm").property("value");
    if(kind==="onecycle" && wpct<10){ /* one-cycle wants a long ramp; leave the user's value but hint it */ }
    document.getElementById("sched-wval").textContent=wpct;
    const Tw=Math.round(T*wpct/100);
    const t=+d3.select("#sched-t").property("value");
    const data=d3.range(1,T+1,2).map(s=>({s, v:lr(s,kind,Tw)}));
    path.datum(data).attr("d",d3.line().x(d=>x(d.s)).y(d=>y(d.v)));
    area.datum(data).attr("d",d3.area().x(d=>x(d.s)).y0(ih).y1(d=>y(d.v)));
    warmBand.attr("x",x(1)).attr("width",Math.max(0,x(Math.max(Tw,1))-x(1)));
    warmLabel.attr("x",x(1)+4).style("opacity",Tw>20?1:0);
    xAxisG.call(d3.axisBottom(x).ticks(6).tickFormat(d3.format("d")));
    yAxisG.call(d3.axisLeft(y).ticks(5));
    const v=lr(t,kind,Tw);
    marker.attr("x1",x(t)).attr("x2",x(t));
    dot.attr("cx",x(t)).attr("cy",y(v));
    const phase = (Tw>0 && t<Tw) ? "warming up" : (v<0.05 ? "annealed" : "decaying");
    d3.select("#sched-read").html(
      `<b>${NAME[kind]}</b> · warmup <b>${Tw}</b> steps · at step <b>${t}</b> the rate is <b>${(v*100).toFixed(1)}%</b> of η<sub>max</sub> — <b style="color:${C.good}">${phase}</b>`+
      `<br>${RULE[kind]}`
    );
  }
  d3.select("#sched-kind").on("change",draw);
  ["#sched-warm","#sched-t"].forEach(id=>d3.select(id).on("input",draw));
  draw();
})();
