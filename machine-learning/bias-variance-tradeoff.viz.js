/* bias-variance-tradeoff.viz.js — extracted from bias-variance-tradeoff.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ── error vs complexity, with the double-descent toggle ── */
(function(){
  const C={accent:"#ffb454",blue:"#5b9cff",good:"#4ade80",bad:"#f87171",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#bv-svg"),W=640,H=280;
  const bias2=c=>3*Math.exp(-0.35*c);
  const varc=c=>0.05*Math.exp(0.20*c);
  const noise=0.4;
  const classic=c=>noise+bias2(c)+varc(c);
  const peakC=12, peakV=noise+bias2(peakC)+varc(peakC);
  const dd=c=> c<=peakC ? classic(c) : peakV*Math.exp(-0.14*(c-peakC))+0.55;
  const px=46,py=22,pw=W-92,ph=210, xmin=1,xmax=19;
  const X=v=>px+(v-xmin)/(xmax-xmin)*pw;
  const ymax=4.2, Y=v=>py+ph-Math.min(v,ymax)/ymax*ph;
  function draw(){
    svg.selectAll("*").remove();
    const c=+d3.select("#bv-c").property("value"), ddon=d3.select("#bv-dd").property("checked");
    svg.append("line").attr("x1",px).attr("y1",py+ph).attr("x2",px+pw).attr("y2",py+ph).attr("stroke","#3a4150");
    svg.append("text").attr("x",px+pw/2).attr("y",py+ph+22).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("model complexity →");
    svg.append("text").attr("x",px-4).attr("y",py-8).attr("fill",C.muted).attr("font-size",10).text("error");
    // noise floor
    svg.append("line").attr("x1",px).attr("y1",Y(noise)).attr("x2",px+pw).attr("y2",Y(noise)).attr("stroke",C.muted).attr("stroke-opacity",.45).attr("stroke-dasharray","2 4");
    svg.append("text").attr("x",px+pw-2).attr("y",Y(noise)-5).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",9).text("irreducible noise σ²");
    const xs=d3.range(xmin,xmax+0.01,0.25);
    const line=(f,col,wid,dash)=>svg.append("path").attr("d",d3.line().x(X).y(d=>Y(f(d)))(xs)).attr("fill","none").attr("stroke",col).attr("stroke-width",wid).attr("stroke-dasharray",dash||null);
    if(!ddon){
      line(bias2,C.blue,1.6,"4 3"); line(varc,C.bad,1.6,"4 3"); line(classic,C.accent,2.6);
      svg.append("text").attr("x",X(2)).attr("y",Y(bias2(2))-6).attr("fill",C.blue).attr("font-size",10).text("bias²");
      svg.append("text").attr("x",X(17)).attr("y",Y(varc(17))-6).attr("text-anchor","end").attr("fill",C.bad).attr("font-size",10).text("variance");
      svg.append("text").attr("x",X(9.2)).attr("y",Y(classic(9.2))-10).attr("fill",C.accent).attr("font-size",10).text("total error");
    } else {
      svg.append("line").attr("x1",X(peakC)).attr("y1",py).attr("x2",X(peakC)).attr("y2",py+ph).attr("stroke","#3a4150").attr("stroke-dasharray","2 3");
      svg.append("text").attr("x",X(peakC)).attr("y",py+10).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9).text("interpolation");
      line(dd,C.accent,2.6);
    }
    const f=ddon?dd:classic;
    // marker
    svg.append("line").attr("x1",X(c)).attr("y1",py).attr("x2",X(c)).attr("y2",py+ph).attr("stroke",C.good).attr("stroke-dasharray","3 3");
    svg.append("circle").attr("cx",X(c)).attr("cy",Y(f(c))).attr("r",5).attr("fill",C.good);
    // optimum
    let best=xmin,bv=1e9; xs.forEach(x=>{const v=f(x); if(v<bv){bv=v;best=x;}});
    const region = c<best-1.5?"underfitting (high bias)":c>best+1.5?(ddon&&c>peakC?"second descent — generalizing again":"overfitting (high variance)"):"≈ sweet spot";
    const parts = ddon ? "" : ` · bias² <b>${bias2(c).toFixed(2)}</b> + var <b>${varc(c).toFixed(2)}</b> + σ² <b>${noise.toFixed(2)}</b>`;
    d3.select("#bv-read").html(`complexity <b>${(+c).toFixed(1)}</b> · total error <b>${f(c).toFixed(2)}</b>${parts} · <b style="color:${C.good}">${region}</b>`);
  }
  d3.select("#bv-c").on("input",draw); d3.select("#bv-dd").on("change",draw);
  draw();
})();

/* ── twelve polynomial fits on twelve resampled training sets ── */
(function(){
  const C={accent:"#ffb454",blue:"#5b9cff",good:"#4ade80",bad:"#f87171",muted:"#9aa3b2",ink:"#e6e9ef"};
  const svg=d3.select("#bv2-svg"),W=640,H=300,L=46,R=14,T=16,B=32;
  const R_SETS=12, N_PTS=16;
  const truth=x=>Math.sin(2.6*x)+0.35*x;
  const sx=d3.scaleLinear().domain([-1,1]).range([L,W-R]);
  const sy=d3.scaleLinear().domain([-2.3,2.3]).range([H-B,T]);
  const grid=d3.range(-1,1.0001,1/60);

  let seed=20240904;
  function rnd(){ seed|=0; seed=seed+0x6D2B79F5|0; let t=Math.imul(seed^seed>>>15,1|seed); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }
  function gauss(){ let u=0,v=0; while(u===0)u=rnd(); while(v===0)v=rnd(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }

  function solve(A,b){
    const n=b.length;
    for(let i=0;i<n;i++){
      let p=i; for(let r=i+1;r<n;r++) if(Math.abs(A[r][i])>Math.abs(A[p][i])) p=r;
      const tA=A[i]; A[i]=A[p]; A[p]=tA; const tb=b[i]; b[i]=b[p]; b[p]=tb;
      const d=A[i][i]||1e-12;
      for(let r=i+1;r<n;r++){ const m=A[r][i]/d; for(let c=i;c<n;c++) A[r][c]-=m*A[i][c]; b[r]-=m*b[i]; }
    }
    const x=new Array(n).fill(0);
    for(let i=n-1;i>=0;i--){ let s=b[i]; for(let c=i+1;c<n;c++) s-=A[i][c]*x[c]; x[i]=s/(A[i][i]||1e-12); }
    return x;
  }
  function fitPoly(pts,deg){
    const m=deg+1, A=Array.from({length:m},()=>new Array(m).fill(0)), b=new Array(m).fill(0);
    pts.forEach(p=>{ const pw=[]; let v=1; for(let j=0;j<m;j++){ pw.push(v); v*=p[0]; }
      for(let i=0;i<m;i++){ b[i]+=pw[i]*p[1]; for(let j=0;j<m;j++) A[i][j]+=pw[i]*pw[j]; } });
    for(let i=0;i<m;i++) A[i][i]+=1e-7;
    return solve(A,b);
  }
  const evalPoly=(c,x)=>{ let s=0,v=1; for(let j=0;j<c.length;j++){ s+=c[j]*v; v*=x; } return s; };

  let sets=[];
  function makeSets(sig){
    sets=[];
    for(let r=0;r<R_SETS;r++){
      const p=[];
      for(let i=0;i<N_PTS;i++){ const x=-1+2*rnd(); p.push([x, truth(x)+sig*gauss()]); }
      sets.push(p);
    }
  }
  function draw(regen){
    const deg=+d3.select("#bv2-deg").property("value");
    const sig=+d3.select("#bv2-noise").property("value");
    if(regen||sets.length===0) makeSets(sig);
    else { // keep the same x's, re-noise to the current sigma
      sets = sets.map(p=>p.map(q=>[q[0], truth(q[0])+sig*gauss()]));
    }
    svg.selectAll("*").remove();
    // axes
    svg.append("line").attr("x1",L).attr("y1",sy(0)).attr("x2",W-R).attr("y2",sy(0)).attr("stroke","#2a2f3a");
    svg.append("line").attr("x1",L).attr("y1",T).attr("x2",L).attr("y2",H-B).attr("stroke","#3a4150");
    svg.append("text").attr("x",(L+W-R)/2).attr("y",H-8).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("x →");
    svg.append("text").attr("x",L-6).attr("y",T+2).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10).text("y");

    const drawCurve=(fn,col,wid,dash,op)=>svg.append("path")
      .attr("d",d3.line().x(d=>sx(d)).y(d=>sy(Math.max(-2.3,Math.min(2.3,fn(d)))))(grid))
      .attr("fill","none").attr("stroke",col).attr("stroke-width",wid)
      .attr("stroke-dasharray",dash||null).attr("stroke-opacity",op==null?1:op);

    // the sample points of the first dataset, faintly
    svg.append("g").selectAll("circle").data(sets[0]).join("circle")
      .attr("cx",d=>sx(d[0])).attr("cy",d=>sy(Math.max(-2.3,Math.min(2.3,d[1])))).attr("r",3)
      .attr("fill",C.muted).attr("fill-opacity",.5);

    const coefs=sets.map(p=>fitPoly(p,deg));
    coefs.forEach(c=>drawCurve(x=>evalPoly(c,x),C.blue,1.1,null,.42));
    const mean=x=>d3.mean(coefs,c=>evalPoly(c,x));
    drawCurve(truth,C.accent,2,"5 4",.95);
    drawCurve(mean,C.good,2.6);

    // empirical decomposition over the grid
    let b2=0,vr=0;
    grid.forEach(x=>{ const mu=mean(x); b2+=(mu-truth(x))**2; vr+=d3.mean(coefs,c=>(evalPoly(c,x)-mu)**2); });
    b2/=grid.length; vr/=grid.length;
    const tot=b2+vr+sig*sig;
    const verdict = b2>3*vr?"bias-dominated (underfitting)":vr>3*b2?"variance-dominated (overfitting)":"balanced";
    d3.select("#bv2-read").html(
      `degree <b>${deg}</b> · σ² <b>${(sig*sig).toFixed(3)}</b> &nbsp;|&nbsp; bias² <b style="color:${C.accent}">${b2.toFixed(3)}</b> + variance <b style="color:${C.blue}">${vr.toFixed(3)}</b> + noise <b>${(sig*sig).toFixed(3)}</b> = <b style="color:${C.good}">${tot.toFixed(3)}</b> &nbsp;·&nbsp; ${verdict}`);
  }
  d3.select("#bv2-deg").on("input",()=>draw(false));
  d3.select("#bv2-noise").on("input",()=>draw(false));
  d3.select("#bv2-new").on("click",()=>draw(true));
  draw(true);
})();
