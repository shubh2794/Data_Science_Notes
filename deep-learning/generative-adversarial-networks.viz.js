/* generative-adversarial-networks.viz.js — extracted from generative-adversarial-networks.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ═══════════ 01 · training dynamics + generator gradient scales ═══════════ */
(function(){
  const svg=d3.select("#gan-svg"), W=640, H=300, m={t:22,r:212,b:34,l:46};
  const x=d3.scaleLinear().domain([0,120]).range([m.l,W-m.r]);
  const y=d3.scaleLinear().domain([0,1]).range([H-m.b,m.t]);
  svg.append("g").attr("class","axis").attr("transform","translate(0,"+(H-m.b)+")").call(d3.axisBottom(x).ticks(5));
  svg.append("g").attr("class","axis").attr("transform","translate("+m.l+",0)").call(d3.axisLeft(y).ticks(5));
  svg.append("text").attr("x",(m.l+(W-m.r))/2).attr("y",H-6).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.muted).text("training step");
  svg.append("line").attr("x1",x(0)).attr("x2",x(120)).attr("y1",y(0.5)).attr("y2",y(0.5)).attr("stroke",C.good).attr("stroke-dasharray","5 5").attr("stroke-opacity",.7);
  svg.append("text").attr("x",x(120)).attr("y",y(0.5)-6).attr("text-anchor","end").attr("fill",C.good).attr("font-size",10).text("equilibrium D = 0.5");
  const lineR=svg.append("path").attr("fill","none").attr("stroke",C.A).attr("stroke-width",2);
  const lineF=svg.append("path").attr("fill","none").attr("stroke",C.B).attr("stroke-width",2);
  svg.append("text").attr("x",m.l+8).attr("y",m.t+2).attr("fill",C.A).attr("font-size",11).text("D(real)");
  svg.append("text").attr("x",m.l+8).attr("y",m.t+18).attr("fill",C.B).attr("font-size",11).text("D(fake)");

  // right-hand gradient-scale panel
  const bx=W-m.r+34, bw=52, bBase=H-m.b, bTop=m.t+30;
  const by=d3.scaleLinear().domain([0,1]).range([bBase,bTop]);
  const gBars=svg.append("g");
  svg.append("text").attr("x",bx).attr("y",m.t+8).attr("font-size",11).attr("fill",C.ink).text("generator ∇ scale");
  svg.append("line").attr("x1",bx-6).attr("x2",W-14).attr("y1",bBase).attr("y2",bBase).attr("stroke",C.line);
  const bars=[{k:"sat",label:"saturating",sub:"= D(fake)",col:C.bad,x:bx},
              {k:"non",label:"non-sat.",sub:"= 1 − D(fake)",col:C.good,x:bx+82}];
  bars.forEach(b=>{
    b.rect=gBars.append("rect").attr("x",b.x).attr("width",bw).attr("fill",b.col).attr("fill-opacity",.55).attr("stroke",b.col);
    b.val=gBars.append("text").attr("x",b.x+bw/2).attr("text-anchor","middle").attr("font-size",11).attr("fill",C.ink);
    svg.append("text").attr("x",b.x+bw/2).attr("y",bBase+14).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.muted).text(b.label);
    svg.append("text").attr("x",b.x+bw/2).attr("y",bBase+26).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.muted).text(b.sub);
  });

  const ld=d3.line().x(d=>x(d.t)).y(d=>y(d.v)).curve(d3.curveMonotoneX);
  let dR=[],dF=[],t=0,timer=null;
  const regime=()=>document.getElementById("gan-regime").value;
  function state(t){
    const n=()=>(Math.random()-0.5)*0.05;
    if(regime()==="dstrong")  return {r:Math.min(0.995,0.80+0.19*(1-Math.exp(-t/18)))+n()*0.3, f:Math.max(0.005,0.20*Math.exp(-t/18))+n()*0.3};
    if(regime()==="osc")      return {r:0.5+0.40*Math.cos(t/9)+n(), f:0.5-0.40*Math.cos(t/9+0.7)+n()};
    const d=Math.exp(-t/38);
    return {r:0.5+0.42*d*(0.6+0.4*Math.cos(t/11))+n()*d, f:0.5-0.42*d*(0.6+0.4*Math.cos(t/11+0.4))+n()*d};
  }
  function clamp(v){ return Math.max(0.005,Math.min(0.995,v)); }
  function bar(f){
    const vals={sat:f, non:1-f};
    bars.forEach(b=>{
      const v=vals[b.k];
      b.rect.attr("y",by(v)).attr("height",Math.max(1,bBase-by(v)));
      b.val.attr("y",by(v)-6).text(v.toFixed(2));
    });
  }
  function render(){
    lineR.attr("d",ld(dR)); lineF.attr("d",ld(dF));
    const f=dF.length?dF[dF.length-1].v:0.5, r=dR.length?dR[dR.length-1].v:0.5;
    bar(f);
    const warn = f<0.08 ? " — saturating loss has no gradient left" : (Math.abs(f-0.5)<0.06 ? " — balanced" : "");
    d3.select("#gan-readout").html("step <b>"+t+"</b> &nbsp;·&nbsp; D(real) <b>"+r.toFixed(2)+"</b> &nbsp;·&nbsp; D(fake) <b>"+f.toFixed(2)+"</b> &nbsp;·&nbsp; ∇ scale: saturating <b>"+f.toFixed(2)+"</b>, non-saturating <b>"+(1-f).toFixed(2)+"</b>"+warn);
  }
  function reset(){ if(timer){clearInterval(timer);timer=null;} t=0; const s=state(0); dR=[{t:0,v:clamp(s.r)}]; dF=[{t:0,v:clamp(s.f)}]; render(); }
  function train(){ if(timer) return; if(t>=120) reset();
    timer=setInterval(()=>{ t+=2; const s=state(t);
      dR.push({t,v:clamp(s.r)}); dF.push({t,v:clamp(s.f)}); render();
      if(t>=120){clearInterval(timer);timer=null;}
    },110);
  }
  d3.select("#gan-train").on("click",train);
  d3.select("#gan-reset").on("click",reset);
  d3.select("#gan-regime").on("change",reset);
  reset();
})();

/* ═══════════ 02 · p_data vs p_g, optimal discriminator, JSD ═══════════ */
(function(){
  const svg=d3.select("#jsd-svg"), W=640, H=330, m={t:22,r:52,b:40,l:52};
  const XLO=-5.2, XHI=5.2, N=261, dx=(XHI-XLO)/(N-1);
  const xs=d3.range(N).map(i=>XLO+i*dx);
  const g1=(x,mu,s)=>Math.exp(-0.5*Math.pow((x-mu)/s,2))/(s*Math.sqrt(2*Math.PI));
  const MODES=[{mu:-1.7,s:0.62,w:0.5},{mu:1.8,s:0.55,w:0.5}];
  const pdata=x=>MODES.reduce((a,k)=>a+k.w*g1(x,k.mu,k.s),0);

  const x=d3.scaleLinear().domain([XLO,XHI]).range([m.l,W-m.r]);
  const yD=d3.scaleLinear().domain([0,0.55]).range([H-m.b,m.t]);
  const yP=d3.scaleLinear().domain([0,1]).range([H-m.b,m.t]);
  svg.append("g").attr("class","axis").attr("transform","translate(0,"+(H-m.b)+")").call(d3.axisBottom(x).ticks(7));
  svg.append("g").attr("class","axis").attr("transform","translate("+m.l+",0)").call(d3.axisLeft(yD).ticks(4));
  svg.append("g").attr("class","axis").attr("transform","translate("+(W-m.r)+",0)").call(d3.axisRight(yP).ticks(3));
  svg.append("text").attr("x",m.l-6).attr("y",m.t-8).attr("text-anchor","end").attr("font-size",10).attr("fill",C.muted).text("density");
  svg.append("text").attr("x",W-m.r+6).attr("y",m.t-8).attr("font-size",10).attr("fill",C.good).text("D*(x)");
  svg.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-8).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.muted).text("x");
  svg.append("line").attr("x1",x(XLO)).attr("x2",x(XHI)).attr("y1",yP(0.5)).attr("y2",yP(0.5)).attr("stroke",C.good).attr("stroke-dasharray","4 4").attr("stroke-opacity",.4);

  const area=d3.area().x(d=>x(d.x)).y0(yD(0)).y1(d=>yD(d.v)).curve(d3.curveBasis);
  const line=d3.line().x(d=>x(d.x)).y(d=>yP(d.v)).curve(d3.curveBasis);
  const aReal=svg.append("path").attr("fill",C.A).attr("fill-opacity",.28).attr("stroke",C.A).attr("stroke-width",1.6);
  const aGen =svg.append("path").attr("fill",C.B).attr("fill-opacity",.28).attr("stroke",C.B).attr("stroke-width",1.6);
  const lDs  =svg.append("path").attr("fill","none").attr("stroke",C.good).attr("stroke-width",2.2);
  svg.append("text").attr("x",m.l+8).attr("y",m.t+2).attr("fill",C.A).attr("font-size",11).text("p_data");
  svg.append("text").attr("x",m.l+8).attr("y",m.t+18).attr("fill",C.B).attr("font-size",11).text("p_g (generator)");
  svg.append("text").attr("x",m.l+8).attr("y",m.t+34).attr("fill",C.good).attr("font-size",11).text("D*(x) = p_data / (p_data + p_g)");
  const handle=svg.append("circle").attr("r",8).attr("fill",C.B).attr("stroke","#0f1117").attr("stroke-width",2).style("cursor","ew-resize");

  const muEl=document.getElementById("jsd-mu"), sigEl=document.getElementById("jsd-sig"), out=document.getElementById("jsd-readout");
  function render(){
    const mu=+muEl.value, s=+sigEl.value;
    const P=xs.map(v=>pdata(v)), Q=xs.map(v=>g1(v,mu,s));
    aReal.attr("d",area(xs.map((v,i)=>({x:v,v:P[i]}))));
    aGen .attr("d",area(xs.map((v,i)=>({x:v,v:Q[i]}))));
    const D=xs.map((v,i)=>{ const d=P[i]+Q[i]; return d<1e-12?0.5:P[i]/d; });
    lDs.attr("d",line(xs.map((v,i)=>({x:v,v:D[i]}))));
    handle.attr("cx",x(mu)).attr("cy",yD(Math.min(0.55,g1(mu,mu,s))));
    let jsd=0, satW=0, qMass=0;
    for(let i=0;i<N;i++){
      const p=P[i], q=Q[i], mm=0.5*(p+q);
      if(p>1e-12&&mm>1e-12) jsd+=0.5*p*Math.log(p/mm)*dx;
      if(q>1e-12&&mm>1e-12) jsd+=0.5*q*Math.log(q/mm)*dx;
      satW+=q*D[i]*dx; qMass+=q*dx;
    }
    const sat=satW/Math.max(qMass,1e-9);
    const bits=jsd/Math.LN2, V=-Math.log(4)+2*jsd;
    const col = bits>0.85 ? C.bad : (bits<0.15 ? C.good : C.B);
    out.innerHTML="JSD = <b style=\"color:"+col+"\">"+bits.toFixed(3)+"</b> / 1.000 bits &nbsp;·&nbsp; V(G, D*) = <b>"+V.toFixed(3)+"</b> (floor −1.386) &nbsp;·&nbsp; 𝔼<sub>p_g</sub>[D*] = <b>"+sat.toFixed(3)+"</b> — the saturating-loss gradient scale &nbsp;·&nbsp; non-saturating = <b>"+(1-sat).toFixed(3)+"</b>";
  }
  muEl.addEventListener("input",render); sigEl.addEventListener("input",render);
  handle.call(d3.drag().on("drag",e=>{
    const v=Math.max(-4.5,Math.min(4.5,x.invert(e.x)));
    muEl.value=v.toFixed(2); render();
  }));
  document.getElementById("jsd-fit").addEventListener("click",()=>{ muEl.value="1.80"; sigEl.value="0.55"; render(); });
  render();
})();

/* ═══════════ 04 · mode collapse on a ring of 8 modes ═══════════ */
(function(){
  const svg=d3.select("#mc-svg"), W=640, H=330, K=8, NP=280;
  const cx=190, cy=168, R=112;
  const modes=d3.range(K).map(i=>{ const a=-Math.PI/2+2*Math.PI*i/K; return {i,a,x:cx+R*Math.cos(a),y:cy+R*Math.sin(a)}; });
  // fixed randomness so the slider animates smoothly
  let seed=42; const rnd=()=>{ seed=(seed*1103515245+12345)&0x7fffffff; return seed/0x7fffffff; };
  const pts=d3.range(NP).map(()=>({u:rnd(), jx:(rnd()+rnd()+rnd()-1.5)*13, jy:(rnd()+rnd()+rnd()-1.5)*13}));

  svg.append("text").attr("x",cx).attr("y",26).attr("text-anchor","middle").attr("font-size",11).attr("fill",C.muted).text("data modes (rings) · generator samples (dots)");
  const gRings=svg.append("g"), gPts=svg.append("g");
  modes.forEach(mo=>{
    gRings.append("circle").attr("cx",mo.x).attr("cy",mo.y).attr("r",20).attr("fill","none").attr("stroke",C.A).attr("stroke-opacity",.75).attr("stroke-dasharray","3 3");
  });

  // right-hand bar chart of generator mass per mode
  const bx0=390, bw=26, bBase=H-56, bTop=64;
  const byS=d3.scaleLinear().domain([0,0.6]).range([bBase,bTop]);
  svg.append("text").attr("x",bx0).attr("y",44).attr("font-size",11).attr("fill",C.ink).text("generator mass per mode");
  svg.append("line").attr("x1",bx0-6).attr("x2",bx0+K*bw+4).attr("y1",bBase).attr("y2",bBase).attr("stroke",C.line);
  const target=svg.append("line").attr("x1",bx0-6).attr("x2",bx0+K*bw+4).attr("y1",byS(1/K)).attr("y2",byS(1/K)).attr("stroke",C.good).attr("stroke-dasharray","4 3").attr("stroke-opacity",.7);
  svg.append("text").attr("x",bx0+K*bw+8).attr("y",byS(1/K)+4).attr("font-size",9).attr("fill",C.good).text("1/8");
  const gBars=svg.append("g");

  const cEl=document.getElementById("mc-c"), out=document.getElementById("mc-readout");
  function render(){
    const c=+cEl.value, kappa=9*c;
    const raw=modes.map(mo=>Math.exp(kappa*Math.cos(mo.a+Math.PI/2)));
    const tot=d3.sum(raw), w=raw.map(v=>v/tot);
    const cum=[]; let acc=0; w.forEach(v=>{acc+=v; cum.push(acc);});
    const counts=new Array(K).fill(0);
    const placed=pts.map(p=>{
      let k=0; while(k<K-1&&p.u>cum[k]) k++;
      counts[k]++;
      return {x:modes[k].x+p.jx, y:modes[k].y+p.jy, k};
    });
    gPts.selectAll("circle").data(placed).join("circle")
      .attr("cx",d=>d.x).attr("cy",d=>d.y).attr("r",2.4).attr("fill",C.B).attr("fill-opacity",.75);
    gBars.selectAll("rect").data(w).join("rect")
      .attr("x",(d,i)=>bx0+i*bw+2).attr("width",bw-4)
      .attr("y",d=>byS(Math.min(0.6,d))).attr("height",d=>Math.max(1,bBase-byS(Math.min(0.6,d))))
      .attr("fill",(d,i)=>counts[i]>=NP*0.02?C.B:C.bad).attr("fill-opacity",.6).attr("stroke",(d,i)=>counts[i]>=NP*0.02?C.B:C.bad);
    const covered=counts.filter(v=>v>=NP*0.02).length;
    const dom=100*d3.max(w);
    const col = covered>=7?C.good : covered>=4?C.B : C.bad;
    out.innerHTML="modes covered <b style=\"color:"+col+"\">"+covered+" / 8</b> &nbsp;·&nbsp; largest mode holds <b>"+dom.toFixed(0)+"%</b> of generator mass &nbsp;·&nbsp; "+(covered>=7?"healthy coverage":covered>=4?"partial collapse — samples still look fine, diversity is halved":"severe collapse — a beautiful, useless generator");
  }
  cEl.addEventListener("input",render);
  document.getElementById("mc-reset").addEventListener("click",()=>{ cEl.value=0; render(); });
  render();
})();
