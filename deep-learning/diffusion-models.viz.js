/* diffusion-models.viz.js — extracted from diffusion-models.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ── shared schedule helpers (cosine + linear ᾱ tables) ───────────────────── */
function abarCosine(T, s){ s = (s===undefined) ? 0.008 : s;
  const f = u => Math.pow(Math.cos(((u + s)/(1 + s)) * Math.PI/2), 2);
  const f0 = f(0), out = new Float64Array(T+1);
  for(let t=0;t<=T;t++) out[t] = Math.max(1e-9, Math.min(1, f(t/T)/f0));
  return out;                                   // out[0] = 1 (clean), out[T] ≈ 0
}
function abarLinear(T, bmin, bmax){
  bmin = (bmin===undefined) ? 1e-4 : bmin; bmax = (bmax===undefined) ? 0.02 : bmax;
  const out = new Float64Array(T+1); let acc = 1; out[0] = 1;
  for(let t=1;t<=T;t++){ const b = bmin + (t-1)/Math.max(1,T-1)*(bmax-bmin); acc *= (1-b); out[t] = Math.max(1e-9, acc); }
  return out;
}
function gauss(){ let u=0,v=0; while(u===0)u=Math.random(); while(v===0)v=Math.random();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }

/* ───────────────── 02 · forward noising ↔ reverse denoising ───────────────── */
(function(){
  const svg=d3.select("#diff-svg"), N=16, cell=12, ox=(640-N*cell)/2, oy=14;
  // a small stand-in "image": a ring, two dots and a bar — enough structure to watch dissolve
  const base=d3.range(N).map((r)=>d3.range(N).map((c)=>{
    const cx=N/2, cy=N/2, d=Math.hypot(r-cy+1,c-cx+0.5);
    let v = (d<6 && d>4) ? 0.85 : 0.12;
    if(r===6 && (c===5||c===10)) v=0.95;
    if(r>=9 && r<=10 && c>=5 && c<=10 && Math.abs(c-7.5)<3) v=0.9;
    return v*2-1;                                  // centre to roughly [-1, 1]
  }));
  const noise=base.map(row=>row.map(()=>gauss()));  // one fixed ε draw, so the slider is smooth
  const g=svg.append("g").attr("transform","translate("+ox+","+oy+")");
  const col=d3.scaleLinear().domain([-2.2,2.2]).range([0,1]).clamp(true);
  const cells=[]; for(let r=0;r<N;r++) for(let c=0;c<N;c++) cells.push({r,c});
  g.selectAll("rect").data(cells).enter().append("rect")
    .attr("x",d=>d.c*cell).attr("y",d=>d.r*cell).attr("width",cell-1).attr("height",cell-1);

  const STEPS=100, tables={cosine:abarCosine(STEPS), linear:abarLinear(STEPS,1e-3,0.2)};
  const lbl=svg.append("text").attr("x",320).attr("y",oy+N*cell+22).attr("text-anchor","middle")
    .attr("fill",C.muted).attr("font-size",11);
  // signal / noise energy bar
  const barY=oy+N*cell+38, barX=140, barW=360;
  svg.append("text").attr("x",barX-8).attr("y",barY+11).attr("text-anchor","end").attr("font-size",10).attr("fill",C.muted).text("energy");
  const barSig=svg.append("rect").attr("x",barX).attr("y",barY).attr("height",14).attr("rx",3).attr("fill",C.A).attr("fill-opacity",.85);
  const barNoi=svg.append("rect").attr("y",barY).attr("height",14).attr("rx",3).attr("fill",C.bad).attr("fill-opacity",.6);
  svg.append("rect").attr("x",barX).attr("y",barY).attr("width",barW).attr("height",14).attr("rx",3)
    .attr("fill","none").attr("stroke",C.line);

  const out=document.getElementById("diff-readout");
  let timer=null;
  function render(t){
    const sched=document.getElementById("diff-sched").value, ab=tables[sched][t];
    const sa=Math.sqrt(ab), sn=Math.sqrt(1-ab);
    g.selectAll("rect").attr("fill",d=> d3.interpolateViridis( col( sa*base[d.r][d.c] + sn*noise[d.r][d.c] ) ));
    d3.select("#diff-tval").text(t);
    lbl.text(t===0 ? "clean data x₀" : (t>=STEPS ? "pure noise x_T ~ 𝒩(0, I)" : "noised xₜ = √ᾱₜ x₀ + √(1 − ᾱₜ) ε"));
    barSig.attr("width",Math.max(0,barW*ab));
    barNoi.attr("x",barX+barW*ab).attr("width",Math.max(0,barW*(1-ab)));
    const snr=ab/Math.max(1e-9,1-ab);
    out.innerHTML="ᾱₜ = <b>"+ab.toFixed(3)+"</b> &nbsp;·&nbsp; signal coeff √ᾱₜ = <b>"+sa.toFixed(3)+
      "</b> &nbsp;·&nbsp; noise coeff √(1−ᾱₜ) = <b>"+sn.toFixed(3)+
      "</b> &nbsp;·&nbsp; log SNR = <b>"+Math.log(snr).toFixed(2)+"</b>";
  }
  function stop(){ if(timer){clearInterval(timer); timer=null;} }
  d3.select("#diff-t").on("input",function(){ stop(); render(+this.value); });
  d3.select("#diff-sched").on("change",()=>render(+d3.select("#diff-t").property("value")));
  function anim(dir){ if(timer){stop();return;} let t=+d3.select("#diff-t").property("value");
    if(dir>0 && t>=STEPS) t=0; if(dir<0 && t<=0) t=STEPS;
    timer=setInterval(()=>{ t+=dir*3; if(t<=0){t=0;stop();} if(t>=STEPS){t=STEPS;stop();}
      d3.select("#diff-t").property("value",t); render(t); },70); }
  d3.select("#diff-play").on("click",()=>anim(1));
  d3.select("#diff-rev").on("click",()=>anim(-1));
  render(0);
})();

/* ───────────────── 03 · noise-schedule explorer ───────────────── */
(function(){
  const svg=d3.select("#sch-svg"), W=640, H=300, top=34, ph=196, pw=170, gap=30, x0=52;
  const panels=[
    {key:"beta",  title:"βₜ  (log scale)",  log:true,  dom:[1e-4,1]},
    {key:"abar",  title:"ᾱₜ  (signal kept)",log:false, dom:[0,1]},
    {key:"snr",   title:"log SNR",          log:false, dom:[-12,12]}
  ];
  const Tel=document.getElementById("sch-T"), tEl=document.getElementById("sch-t"),
        Tval=document.getElementById("sch-Tval"), tval=document.getElementById("sch-tval"),
        out=document.getElementById("sch-readout");
  const root=svg.append("g");

  function seriesFor(T){
    const cos=abarCosine(T), lin=abarLinear(T);
    const mk=(ab)=>{ const rows=[];
      for(let t=1;t<=T;t++){
        const beta=Math.min(0.999, Math.max(1e-6, 1 - ab[t]/ab[t-1]));
        rows.push({u:t/T, beta:beta, abar:ab[t], snr:Math.log(ab[t]/Math.max(1e-12,1-ab[t]))});
      } return rows; };
    return {cosine:mk(cos), linear:mk(lin), cosTab:cos, linTab:lin};
  }
  function draw(){
    const T=+Tel.value, u=(+tEl.value)/100; Tval.textContent=T; tval.textContent=u.toFixed(2);
    const S=seriesFor(T);
    root.selectAll("*").remove();
    panels.forEach((p,i)=>{
      const gx=x0+i*(pw+gap), g=root.append("g").attr("transform","translate("+gx+",0)");
      const x=d3.scaleLinear().domain([0,1]).range([0,pw]);
      const y=p.log ? d3.scaleLog().domain(p.dom).range([top+ph,top]).clamp(true)
                    : d3.scaleLinear().domain(p.dom).range([top+ph,top]).clamp(true);
      g.append("text").attr("x",pw/2).attr("y",18).attr("text-anchor","middle")
        .attr("font-size",11.5).attr("fill",C.ink).attr("font-weight",600).text(p.title);
      // frame + gridlines
      g.append("rect").attr("x",0).attr("y",top).attr("width",pw).attr("height",ph)
        .attr("fill","none").attr("stroke",C.line);
      const ticks = p.log ? [1e-4,1e-3,1e-2,1e-1,1] : y.ticks(5);
      ticks.forEach(tv=>{ const yy=y(tv);
        g.append("line").attr("x1",0).attr("y1",yy).attr("x2",pw).attr("y2",yy).attr("stroke","#1b2130");
        g.append("text").attr("x",-6).attr("y",yy+3).attr("text-anchor","end").attr("font-size",9)
          .attr("fill",C.muted).text(p.log ? d3.format(".0e")(tv) : d3.format("~g")(tv));
      });
      if(!p.log && p.dom[0]<0){ g.append("line").attr("x1",0).attr("y1",y(0)).attr("x2",pw).attr("y2",y(0)).attr("stroke","#3a4150").attr("stroke-dasharray","2 3"); }
      const line=d3.line().x(d=>x(d.u)).y(d=>y(d[p.key]));
      g.append("path").datum(S.linear).attr("fill","none").attr("stroke",C.B)
        .attr("stroke-width",1.8).attr("stroke-dasharray","5 4").attr("d",line);
      g.append("path").datum(S.cosine).attr("fill","none").attr("stroke",C.A)
        .attr("stroke-width",2).attr("d",line);
      // marker
      g.append("line").attr("x1",x(u)).attr("y1",top).attr("x2",x(u)).attr("y2",top+ph)
        .attr("stroke",C.good).attr("stroke-width",1.2).attr("stroke-opacity",.8);
      g.append("text").attr("x",pw/2).attr("y",top+ph+16).attr("text-anchor","middle")
        .attr("font-size",9.5).attr("fill",C.muted).text("t / T  →");
    });
    const idx=Math.max(1,Math.round(u*T));
    const cb=S.cosTab[idx], lb=S.linTab[idx];
    const lsnr=a=>Math.log(a/Math.max(1e-12,1-a)).toFixed(2);
    out.innerHTML="at t = <b>"+idx+"</b> / "+T+" &nbsp;·&nbsp; cosine ᾱₜ = <b>"+cb.toFixed(3)+
      "</b> (log SNR "+lsnr(cb)+") &nbsp;·&nbsp; linear ᾱₜ = <b>"+lb.toFixed(3)+
      "</b> (log SNR "+lsnr(lb)+") &nbsp;·&nbsp; terminal ᾱ_T: cosine <b>"+S.cosTab[T].toExponential(1)+
      "</b>, linear <b>"+S.linTab[T].toExponential(1)+"</b>";
  }
  Tel.addEventListener("input",draw); tEl.addEventListener("input",draw); draw();
})();

/* ───────────────── 07 · reverse sampler playground (1-D toy) ───────────────── */
(function(){
  const svg=d3.select("#rev-svg"), W=640, H=300, m={t:22,r:16,b:34,l:46};
  const XLO=-5, XHI=5, NB=64, NP=240, T=1000, AB=abarCosine(T);
  const MODES=[{mu:-2.2,w:0.5},{mu:2.0,w:0.5}], SD=0.35, COND=1;   // guided class = right mode
  const x=d3.scaleLinear().domain([XLO,XHI]).range([m.l,W-m.r]);
  const y=d3.scaleLinear().domain([0,0.2]).range([H-m.b,m.t]).clamp(true);

  // axes
  svg.append("g").attr("transform","translate(0,"+(H-m.b)+")").attr("class","axis")
    .call(d3.axisBottom(x).ticks(9));
  svg.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-6).attr("text-anchor","middle")
    .attr("font-size",10.5).attr("fill",C.muted).text("x  (1-D sample space)");
  // target densities
  const dens=(xv,list)=>list.reduce((a,c)=>a + c.w*Math.exp(-0.5*Math.pow((xv-c.mu)/SD,2))/(SD*Math.sqrt(2*Math.PI)),0);
  const grid=d3.range(XLO,XHI+0.01,0.05);
  const area=d3.line().x(d=>x(d)).y(d=>y(dens(d,MODES)/6));
  svg.append("path").datum(grid).attr("fill","none").attr("stroke",C.good).attr("stroke-width",1.6)
    .attr("stroke-opacity",.75).attr("d",area);
  svg.append("text").attr("x",W-m.r).attr("y",m.t+2).attr("text-anchor","end").attr("font-size",10)
    .attr("fill",C.good).text("target p(x₀)  (scaled)");
  svg.append("line").attr("x1",x(MODES[1].mu)).attr("x2",x(MODES[1].mu)).attr("y1",m.t).attr("y2",H-m.b)
    .attr("stroke",C.B).attr("stroke-dasharray","3 4").attr("stroke-opacity",.7);
  svg.append("text").attr("x",x(MODES[1].mu)+6).attr("y",m.t+12).attr("font-size",10)
    .attr("fill",C.B).text("guided mode c");
  const gBars=svg.append("g");
  const BW=(XHI-XLO)/NB, THR=d3.range(NB+1).map(i=>XLO+i*BW);
  const out=document.getElementById("rev-readout");
  const stepsEl=document.getElementById("rev-steps"), wEl=document.getElementById("rev-w"),
        etaEl=document.getElementById("rev-eta"),
        stepsVal=document.getElementById("rev-stepsval"), wVal=document.getElementById("rev-wval");

  let parts=null, timer=null;
  function reset(){ stop(); parts=d3.range(NP).map(()=>gauss()); histo("start: x_T ~ 𝒩(0, I) — pure noise"); }
  function stop(){ if(timer){clearInterval(timer); timer=null;} }
  function histo(msg){
    const bins=d3.bin().domain([XLO,XHI]).thresholds(THR)(parts);
    const px=(x(XHI)-x(XLO))/NB;
    const dhat=d=>d.length/(NP*BW)/6;         // same 1/6 scaling as the target curve
    gBars.selectAll("rect").data(bins).join("rect")
      .attr("x",d=>x(d.x0)+1).attr("width",Math.max(1,px-2))
      .attr("y",d=>y(dhat(d)))
      .attr("height",d=>Math.max(0,(H-m.b)-y(dhat(d))))
      .attr("fill",C.A).attr("fill-opacity",.55).attr("stroke",C.A).attr("stroke-opacity",.35);
    const near=(mu)=>parts.filter(p=>Math.abs(p-mu)<0.8).length/NP;
    const L=near(MODES[0].mu), Rm=near(MODES[1].mu);
    out.innerHTML=msg+" &nbsp;·&nbsp; left mode <b>"+(100*L).toFixed(0)+"%</b> · right mode <b>"+
      (100*Rm).toFixed(0)+"%</b> · stranded between <b>"+(100*Math.max(0,1-L-Rm)).toFixed(0)+
      "%</b> &nbsp;·&nbsp; overall σ = <b>"+(d3.deviation(parts)||0).toFixed(2)+"</b>";
  }
  // exact ε for the toy: the noisy marginal of a Gaussian mixture is a Gaussian mixture
  function epsFor(xv, ab, list){
    const sa=Math.sqrt(ab), v=ab*SD*SD + (1-ab);
    let num=0, den=1e-12;
    list.forEach(c=>{ const mu=sa*c.mu, p=c.w*Math.exp(-0.5*Math.pow(xv-mu,2)/v);
      den+=p; num+=p*(-(xv-mu)/v); });
    const score=num/den;
    return -Math.sqrt(1-ab)*score;                 // ε = −√(1−ᾱ) · ∇ₓ log q(xₜ)
  }
  function run(){
    stop();
    const steps=+stepsEl.value, w=+wEl.value, eta=+etaEl.value;
    parts=d3.range(NP).map(()=>gauss());
    const seq=d3.range(steps).map(i=>Math.max(1,Math.round(T - i*(T-1)/steps)));  // T … ~1
    let k=0;
    timer=setInterval(()=>{
      if(k>=steps){ stop(); histo("done — "+steps+" steps, w = "+w.toFixed(2)+", η = "+eta); return; }
      const tCur=seq[k], tNext=(k+1<steps)?seq[k+1]:0;
      const at=AB[tCur], as=AB[tNext];
      const sig = eta * Math.sqrt(Math.max(0,(1-as)/(1-at))) * Math.sqrt(Math.max(0,1-at/as));
      const rest = Math.sqrt(Math.max(0,1-as-sig*sig));
      parts=parts.map(p=>{
        const eu=epsFor(p,at,MODES), ec=epsFor(p,at,[{mu:MODES[COND].mu,w:1}]);
        const e = eu + w*(ec-eu);                          // classifier-free guidance
        const x0h = (p - Math.sqrt(1-at)*e)/Math.sqrt(at); // predicted x₀
        return Math.sqrt(as)*x0h + rest*e + sig*gauss();   // DDIM / DDPM step
      });
      histo("step <b>"+(k+1)+"</b>/"+steps+" &nbsp; t = "+tCur+" &nbsp; ᾱₜ = "+at.toFixed(3));
      k++;
    },45);
  }
  stepsEl.addEventListener("input",()=>{stepsVal.textContent=stepsEl.value;});
  wEl.addEventListener("input",()=>{wVal.textContent=(+wEl.value).toFixed(2);});
  document.getElementById("rev-run").addEventListener("click",run);
  document.getElementById("rev-reset").addEventListener("click",reset);
  stepsVal.textContent=stepsEl.value; wVal.textContent=(+wEl.value).toFixed(2);
  reset();
})();
