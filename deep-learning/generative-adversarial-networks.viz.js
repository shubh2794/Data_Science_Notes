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

/* ═══════════ 11 · PatchGAN receptive field over a 256×256 input ═══════════ */
(function(){
  const svgEl=document.getElementById("patch-svg"), nEl=document.getElementById("patch-n"), out=document.getElementById("patch-readout");
  if(!svgEl||!nEl||!out) return;
  const svg=d3.select(svgEl), IMG=256, ix=28, iy=40, ox=372, oy=58, OS=220;
  // seeded toy "scene" so the figure is identical on every load
  let seed=7; const rnd=()=>{ seed=(seed*1103515245+12345)&0x7fffffff; return seed/0x7fffffff; };
  const scene=svg.append("g");
  scene.append("rect").attr("x",ix).attr("y",iy).attr("width",IMG).attr("height",IMG*0.55).attr("fill","#1f3a5c");
  scene.append("rect").attr("x",ix).attr("y",iy+IMG*0.55).attr("width",IMG).attr("height",IMG*0.45).attr("fill","#2c4a2a");
  scene.append("circle").attr("cx",ix+200).attr("cy",iy+48).attr("r",22).attr("fill","#d9a441").attr("fill-opacity",.85);
  for(let i=0;i<7;i++){ const w=18+rnd()*34, h=30+rnd()*70, x0=ix+6+rnd()*(IMG-w-12);
    scene.append("rect").attr("x",x0).attr("y",iy+IMG*0.55-h).attr("width",w).attr("height",h).attr("fill","#46506a").attr("stroke","#6b7690").attr("stroke-width",.6); }
  for(let i=0;i<5;i++){ const cx=ix+20+rnd()*(IMG-40), cy=iy+IMG*0.62+rnd()*IMG*0.32;
    scene.append("circle").attr("cx",cx).attr("cy",cy).attr("r",6+rnd()*10).attr("fill","#3f6b37"); }
  scene.append("rect").attr("x",ix).attr("y",iy).attr("width",IMG).attr("height",IMG).attr("fill","none").attr("stroke",C.line);
  svg.append("text").attr("x",ix).attr("y",iy-10).attr("font-size",11).attr("fill",C.muted).text("input x ⊕ output G(x), 256 × 256");
  const clip=svg.append("clipPath").attr("id","patch-clip"); clip.append("rect").attr("x",ix).attr("y",iy).attr("width",IMG).attr("height",IMG);
  const rfFull=svg.append("rect").attr("fill","none").attr("stroke",C.B).attr("stroke-dasharray","4 3").attr("stroke-opacity",.55);
  const rfIn=svg.append("rect").attr("clip-path","url(#patch-clip)").attr("fill",C.B).attr("fill-opacity",.22).attr("stroke",C.B).attr("stroke-width",2);
  const ctr=svg.append("circle").attr("r",2.5).attr("fill",C.B);
  svg.append("text").attr("x",ox).attr("y",oy-26).attr("font-size",11).attr("fill",C.muted).text("discriminator output: one real/fake score per cell");
  const oTitle=svg.append("text").attr("x",ox).attr("y",oy-10).attr("font-size",11).attr("fill",C.ink);
  svg.append("rect").attr("x",ox).attr("y",oy).attr("width",OS).attr("height",OS).attr("fill","#161a22").attr("stroke",C.line);
  const gGrid=svg.append("g"), sel=svg.append("rect").attr("fill",C.B).attr("fill-opacity",.8).attr("stroke",C.ink).attr("stroke-width",.8);
  const hit=svg.append("rect").attr("x",ox).attr("y",oy).attr("width",OS).attr("height",OS).attr("fill","transparent").style("cursor","crosshair");
  let ci=null, cj=null;
  const geom=n=>{ // n stride-2 4×4 convs (pad 1) then two stride-1 4×4 convs (pad 1)
    let r=4; r=1*(r-1)+4; for(let k=0;k<n;k++) r=2*(r-1)+4;
    const J=Math.pow(2,n), W=IMG/J-2, start=-3*J+1; return {r,J,W,start}; };
  function render(){
    const n=+nEl.value, g=geom(n);
    if(ci===null||ci>=g.W||cj>=g.W){ ci=Math.floor(g.W/2); cj=Math.floor(g.W/2); }
    const cs=OS/g.W;
    const lines=g.W<=62?d3.range(1,g.W):[];
    gGrid.selectAll("line").data(lines.flatMap(k=>[{k,v:0},{k,v:1}])).join("line")
      .attr("x1",d=>d.v?ox:ox+d.k*cs).attr("x2",d=>d.v?ox+OS:ox+d.k*cs)
      .attr("y1",d=>d.v?oy+d.k*cs:oy).attr("y2",d=>d.v?oy+d.k*cs:oy+OS)
      .attr("stroke",C.line).attr("stroke-width",.6);
    sel.attr("x",ox+ci*cs).attr("y",oy+cj*cs).attr("width",Math.max(1.5,cs)).attr("height",Math.max(1.5,cs));
    const x0=ci*g.J+g.start, y0=cj*g.J+g.start;
    rfFull.attr("x",ix+x0).attr("y",iy+y0).attr("width",g.r).attr("height",g.r);
    rfIn.attr("x",ix+x0).attr("y",iy+y0).attr("width",g.r).attr("height",g.r);
    ctr.attr("cx",ix+x0+g.r/2).attr("cy",iy+y0+g.r/2);
    oTitle.text(g.W+" × "+g.W+" = "+(g.W*g.W)+" overlapping patch judgements");
    const vis=Math.max(0,Math.min(IMG,x0+g.r)-Math.max(0,x0))*Math.max(0,Math.min(IMG,y0+g.r)-Math.max(0,y0));
    const name=g.r===16?"a small-patch critic: sharp but prone to tiling":g.r===70?"the standard 70×70 PatchGAN":g.r===286?"larger than the image: effectively a whole-image critic":g.r===34?"a mid-sized patch critic":"a near-global critic";
    out.innerHTML="receptive field <b>"+g.r+" × "+g.r+"</b> px &nbsp;·&nbsp; output map <b>"+g.W+" × "+g.W+"</b> (stride "+g.J+" px between neighbouring cells) &nbsp;·&nbsp; cell ("+ci+", "+cj+") sees <b>"+(100*vis/(IMG*IMG)).toFixed(1)+"%</b> of the image"+(vis<g.r*g.r?" ("+(100*(1-vis/(g.r*g.r))).toFixed(0)+"% of its patch is zero padding)":"")+" &nbsp;·&nbsp; "+name;
  }
  function pick(e){ const [mx,my]=d3.pointer(e,svgEl); const g=geom(+nEl.value), cs=OS/g.W;
    ci=Math.max(0,Math.min(g.W-1,Math.floor((mx-ox)/cs))); cj=Math.max(0,Math.min(g.W-1,Math.floor((my-oy)/cs))); render(); }
  hit.on("mousemove",pick).on("click",pick);
  nEl.addEventListener("input",()=>{ ci=null; render(); });
  const cb=document.getElementById("patch-center"); if(cb) cb.addEventListener("click",()=>{ ci=null; render(); });
  render();
})();

/* ═══════════ 12 · CycleGAN loss explorer: 1-D domains, five candidate mappings ═══════════ */
(function(){
  const svgEl=document.getElementById("cyc-svg"), lcEl=document.getElementById("cyc-lc"), liEl=document.getElementById("cyc-li"), out=document.getElementById("cyc-readout");
  if(!svgEl||!lcEl||!liEl||!out) return;
  const svg=d3.select(svgEl);
  const MX=4.4, SX=1.0, MY=5.6, SY=1.1, NQ=800, NL=26;
  // standard normal CDF (Abramowitz–Stegun 7.1.26) and inverse CDF (Acklam's rational approximation)
  const Phi=z=>{ const t=1/(1+0.3275911*Math.abs(z)/Math.SQRT2), y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-z*z/2); return z>=0?0.5*(1+y):0.5*(1-y); };
  const iPhi=p=>{ p=Math.min(1-1e-6,Math.max(1e-6,p));
    const a=[-39.69683028665376,220.9460984245205,-275.9285104469687,138.3577518672690,-30.66479806614716,2.506628277459239],
          b=[-54.47609879822406,161.5858368580409,-155.6989798598866,66.80131188771466,-13.28068155288572],
          c=[-0.007784894002430293,-0.3223964580411365,-2.400758277161838,-2.549732539343734,4.374664141464968,2.938163982698783],
          d=[0.007784695709041462,0.3224671290700398,2.445134137142996,3.754408661907416];
    if(p<0.02425){ const q=Math.sqrt(-2*Math.log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); }
    if(p>1-0.02425){ const q=Math.sqrt(-2*Math.log(1-p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); }
    const q=p-0.5, r=q*q; return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1); };
  const uX=x=>Phi((x-MX)/SX), uY=y=>Phi((y-MY)/SY), qX=u=>MX+SX*iPhi(u), qY=u=>MY+SY*iPhi(u);
  // each candidate: h maps X-quantiles to Y-quantiles (measure-preserving except the collapse); hinv is the best inverse branch available to F
  const CANDS=[
    {k:"mono", name:"monotone", tag:["order","kept"], note:"order-preserving bijection", h:u=>u, hinv:v=>v},
    {k:"rev",  name:"reversed", tag:["order","reversed"], note:"order-reversing bijection (the 'recoloured' map)", h:u=>1-u, hinv:v=>1-v},
    {k:"fold2",name:"2-to-1 fold", tag:["2-to-1","fold"], note:"tent map: two inputs per output", h:u=>1-Math.abs(2*u-1), hinv:v=>v/2},
    {k:"fold3",name:"3-to-1 fold", tag:["3-to-1","fold"], note:"zig-zag: three inputs per output", h:u=>u<1/3?3*u:(u<2/3?2-3*u:3*u-2), hinv:v=>v/3},
    {k:"coll", name:"collapse", tag:["","collapse"], note:"every input to the same output", G:()=>MY, F:()=>MX}
  ];
  CANDS.forEach(c=>{ if(!c.G){ c.G=x=>qY(c.h(uX(x))); c.F=y=>qX(c.hinv(uY(y))); } });
  // stratified quantile grids make the expectations deterministic
  const xs=d3.range(NQ).map(i=>qX((i+0.5)/NQ)), ys=d3.range(NQ).map(i=>qY((i+0.5)/NQ));
  CANDS.forEach(c=>{
    const gx=xs.map(c.G).sort((a,b)=>a-b);
    c.adv=d3.mean(gx.map((v,i)=>Math.abs(v-ys[i])));        // 1-D W₁ between G(X) and Y: a stand-in for a perfect critic
    c.cyc=d3.mean(xs,x=>Math.abs(c.F(c.G(x))-x))+d3.mean(ys,y=>Math.abs(c.G(c.F(y))-y));
    c.id =d3.mean(ys,y=>Math.abs(c.G(y)-y))+d3.mean(xs,x=>Math.abs(c.F(x)-x));
  });

  // layout: mapping panel left, loss bars right
  const L0=40, L1=372, ax=d3.scaleLinear().domain([0,10]).range([L0,L1]);
  const rows=[{y:62,lab:"x ∈ X",col:C.A},{y:160,lab:"G(x) ∈ Y",col:C.B},{y:258,lab:"F(G(x))",col:C.A}];
  rows.forEach(r=>{ svg.append("line").attr("x1",L0).attr("x2",L1).attr("y1",r.y).attr("y2",r.y).attr("stroke",C.line);
    svg.append("text").attr("x",L0-6).attr("y",r.y+4).attr("text-anchor","end").attr("font-size",10).attr("fill",r.col).text(r.lab.split(" ")[0]); });
  svg.append("g").attr("class","axis").attr("transform","translate(0,272)").call(d3.axisBottom(ax).ticks(6));
  const dens=(m,s)=>d3.range(0,10.01,0.1).map(v=>[v,Math.exp(-0.5*((v-m)/s)**2)/(s*Math.sqrt(2*Math.PI))]);
  const ar=(y0,flip)=>d3.area().x(d=>ax(d[0])).y0(y0).y1(d=>y0-(flip?-1:1)*d[1]*60).curve(d3.curveBasis);
  svg.append("path").attr("d",ar(rows[0].y)(dens(MX,SX))).attr("fill",C.A).attr("fill-opacity",.18).attr("stroke",C.A).attr("stroke-opacity",.6);
  svg.append("path").attr("d",ar(rows[1].y)(dens(MY,SY))).attr("fill",C.B).attr("fill-opacity",.18).attr("stroke",C.B).attr("stroke-opacity",.6);
  svg.append("path").attr("d",ar(rows[2].y)(dens(MX,SX))).attr("fill",C.A).attr("fill-opacity",.10).attr("stroke",C.A).attr("stroke-opacity",.35);
  svg.append("text").attr("x",L0).attr("y",16).attr("font-size",11).attr("fill",C.muted).text("domain X (blue) → G → domain Y (orange) → F → back to X");
  const mapTitle=svg.append("text").attr("x",L0).attr("y",302).attr("font-size",11).attr("fill",C.ink);
  const gLinks=svg.append("g");
  const pts=d3.range(NL).map(i=>xs[Math.floor((i+0.5)*NQ/NL)]);
  const colr=d3.scaleLinear().domain([0,NL-1]).range([C.A,"#e879f9"]).interpolate(d3.interpolateRgb);

  const B0=404, BW=226, BT=40, BB=262, bx=d3.scaleBand().domain(CANDS.map(c=>c.k)).range([B0,B0+BW]).padding(0.22);
  svg.append("text").attr("x",B0).attr("y",16).attr("font-size",11).attr("fill",C.muted).text("total loss per candidate (lowest wins)");
  [["adversarial",C.muted],["λ·cycle",C.A],["λ·identity",C.B]].forEach((d,i)=>{
    svg.append("rect").attr("x",B0+i*72).attr("y",24).attr("width",9).attr("height",9).attr("fill",d[1]).attr("fill-opacity",.7);
    svg.append("text").attr("x",B0+i*72+13).attr("y",32).attr("font-size",9.5).attr("fill",C.muted).text(d[0]); });
  svg.append("line").attr("x1",B0).attr("x2",B0+BW).attr("y1",BB).attr("y2",BB).attr("stroke",C.line);
  const gBars=svg.append("g"), gLab=svg.append("g"), gWin=svg.append("g");
  CANDS.forEach(c=>{ gLab.append("text").attr("x",bx(c.k)+bx.bandwidth()/2).attr("y",BB+13).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.muted).text(c.tag[0]);
    gLab.append("text").attr("x",bx(c.k)+bx.bandwidth()/2).attr("y",BB+24).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.muted).text(c.tag[1]); });

  let pinned=null;
  function render(){
    const lc=+lcEl.value, li=+liEl.value;
    CANDS.forEach(c=>{ c.parts=[c.adv, lc*c.cyc, li*c.id]; c.tot=c.adv+lc*c.cyc+li*c.id; });
    const best=d3.min(CANDS,c=>c.tot), tied=CANDS.filter(c=>c.tot-best<0.02);
    const shown=pinned?CANDS.find(c=>c.k===pinned):tied[0];
    const by=d3.scaleLinear().domain([0,Math.max(1,d3.max(CANDS,c=>c.tot))*1.08]).range([BB,BT+10]);
    const segs=CANDS.flatMap(c=>{ let acc=0; return c.parts.map((v,j)=>{ const s={c,j,y0:acc,y1:acc+v}; acc+=v; return s; }); });
    gBars.selectAll("rect").data(segs).join("rect")
      .attr("x",d=>bx(d.c.k)).attr("width",bx.bandwidth())
      .attr("y",d=>by(d.y1)).attr("height",d=>Math.max(0,by(d.y0)-by(d.y1)))
      .attr("fill",d=>[C.muted,C.A,C.B][d.j]).attr("fill-opacity",d=>d.c===shown?.85:.5)
      .attr("stroke",d=>d.c===shown?C.ink:"none").attr("stroke-width",.8).style("cursor","pointer")
      .on("click",(e,d)=>{ pinned=d.c.k; render(); });
    gWin.selectAll("text").data(CANDS).join("text")
      .attr("x",c=>bx(c.k)+bx.bandwidth()/2).attr("y",c=>by(c.tot)-5).attr("text-anchor","middle").attr("font-size",9.5)
      .attr("fill",c=>tied.includes(c)?C.good:C.muted).text(c=>(tied.includes(c)?"★ ":"")+c.tot.toFixed(2));
    // mappings for the shown candidate
    const L=pts.map((x,i)=>{ const g=shown.G(x), f=shown.F(g); return {i,x,g,f}; });
    const segL=L.flatMap(d=>[{d,a:[ax(d.x),rows[0].y],b:[ax(Math.max(0,Math.min(10,d.g))),rows[1].y]},{d,a:[ax(Math.max(0,Math.min(10,d.g))),rows[1].y],b:[ax(Math.max(0,Math.min(10,d.f))),rows[2].y]}]);
    gLinks.selectAll("line").data(segL).join("line")
      .attr("x1",s=>s.a[0]).attr("y1",s=>s.a[1]).attr("x2",s=>s.b[0]).attr("y2",s=>s.b[1])
      .attr("stroke",s=>colr(s.d.i)).attr("stroke-opacity",.75).attr("stroke-width",1.1);
    gLinks.selectAll("circle").data(L.flatMap(d=>[{d,x:d.x,y:rows[0].y},{d,x:d.f,y:rows[2].y}])).join("circle")
      .attr("cx",p=>ax(Math.max(0,Math.min(10,p.x)))).attr("cy",p=>p.y).attr("r",2.2).attr("fill",p=>colr(p.d.i));
    mapTitle.text("showing: "+shown.name+" — "+shown.note+(pinned?" (pinned)":""));
    let msg;
    if(tied.length>1) msg="<b style=\"color:"+C.bad+"\">tie</b> between "+tied.map(c=>c.name).join(", ")+" — "+(tied.some(c=>c.k.startsWith("fold"))?"the adversarial term alone cannot tell a bijection from a many-to-one fold":"cycle consistency cannot choose between two bijections; only the identity term can");
    else msg="winner <b style=\"color:"+C.good+"\">"+tied[0].name+"</b>"+(tied[0].k==="coll"?" — identity weight with no cycle weight rewards doing nothing; raise λ<sub>cyc</sub>":tied[0].k==="mono"?" — invertible <i>and</i> faithful":"");
    out.innerHTML=msg+" &nbsp;·&nbsp; "+shown.name+": adversarial <b>"+shown.adv.toFixed(3)+"</b>, cycle <b>"+shown.cyc.toFixed(3)+"</b>, identity <b>"+shown.id.toFixed(3)+"</b> &nbsp;·&nbsp; adversarial term = W₁ between G(X) and Y (an idealised critic)";
  }
  lcEl.addEventListener("input",()=>{ pinned=null; render(); });
  liEl.addEventListener("input",()=>{ pinned=null; render(); });
  const pb=document.getElementById("cyc-paper"); if(pb) pb.addEventListener("click",()=>{ lcEl.value=10; liEl.value=5; pinned=null; render(); });
  const rb=document.getElementById("cyc-reset"); if(rb) rb.addEventListener("click",()=>{ lcEl.value=0; liEl.value=0; pinned=null; render(); });
  render();
})();

/* ═══════════ 13 · Gram matrices ignore spatial arrangement ═══════════ */
(function(){
  const svgEl=document.getElementById("gram-svg"), pEl=document.getElementById("gram-p"), out=document.getElementById("gram-readout");
  if(!svgEl||!pEl||!out) return;
  const svg=d3.select(svgEl), N=16, PX=8, M=N*N, K=5;
  // deterministic toy scene: sky gradient, sun, ground, house with a roof
  const img=[];
  for(let r=0;r<N;r++) for(let c=0;c<N;c++){
    let col=[0.25+0.03*r,0.45+0.02*r,0.85-0.02*r];
    if(r>=11) col=[0.25,0.55-0.02*(r-11),0.2];
    if((r-3)**2+(c-12)**2<=4) col=[0.98,0.85,0.3];
    if(r>=7&&r<=11&&c>=3&&c<=8) col=[0.75,0.35,0.25];
    if(r>=4&&r<=6&&c>=3+(6-r)&&c<=8-(6-r)) col=[0.45,0.2,0.2];
    if(r>=9&&r<=11&&c>=5&&c<=6) col=[0.3,0.2,0.12];
    img.push(col);
  }
  const lum=p=>0.299*p[0]+0.587*p[1]+0.114*p[2];
  // features: R, G, B (per pixel) and |∂x L|, |∂y L| (forward differences, replicate border: a 2-pixel receptive field)
  function feats(im){
    const F=d3.range(K).map(()=>new Float64Array(M));
    for(let r=0;r<N;r++) for(let c=0;c<N;c++){
      const i=r*N+c, p=im[i], L=lum(p), Lx=lum(im[r*N+Math.min(N-1,c+1)]), Ly=lum(im[Math.min(N-1,r+1)*N+c]);
      F[0][i]=p[0]; F[1][i]=p[1]; F[2][i]=p[2]; F[3][i]=Math.abs(Lx-L)*3; F[4][i]=Math.abs(Ly-L)*3;
    }
    return F;
  }
  const gram=F=>d3.range(K).map(a=>d3.range(K).map(b=>{ let s=0; for(let k=0;k<M;k++) s+=F[a][k]*F[b][k]; return s/M; }));
  const F0=feats(img), G0=gram(F0);
  let perm=d3.range(M), shuffles=0;
  function shuffled(p){ // permute whole p×p patches with a seeded Fisher–Yates; seed advances per click
    const nb=N/p, blocks=d3.range(nb*nb); let s=1234+977*shuffles;
    const rnd=()=>{ s=(s*1103515245+12345)&0x7fffffff; return s/0x7fffffff; };
    for(let i=blocks.length-1;i>0;i--){ const j=Math.floor(rnd()*(i+1)); [blocks[i],blocks[j]]=[blocks[j],blocks[i]]; }
    const P=new Array(M);
    for(let b=0;b<blocks.length;b++){ const br=Math.floor(b/nb), bc=b%nb, sr=Math.floor(blocks[b]/nb), sc=blocks[b]%nb;
      for(let dr=0;dr<p;dr++) for(let dc=0;dc<p;dc++) P[(br*p+dr)*N+bc*p+dc]=(sr*p+dr)*N+sc*p+dc; }
    return P;
  }
  const rgb=p=>d3.rgb(255*p[0],255*p[1],255*p[2]).toString();
  const panels=[{x:18,lab:"content image"},{x:166,lab:"patch-shuffled"}];
  const gImg=panels.map(pn=>{ svg.append("text").attr("x",pn.x).attr("y",40).attr("font-size",11).attr("fill",C.muted).text(pn.lab);
    return svg.append("g").attr("transform","translate("+pn.x+",50)"); });
  const GS=22, gx=[330,482], chans=["R","G","B","∂x","∂y"];
  const hm=d3.scaleLinear().domain([0,d3.max(G0.flat())]).range(["#141821",C.B]).interpolate(d3.interpolateRgb).clamp(true);
  const gG=gx.map((x,t)=>{ svg.append("text").attr("x",x).attr("y",40).attr("font-size",11).attr("fill",C.muted).text(t?"Gram, shuffled":"Gram, original");
    chans.forEach((ch,i)=>{ svg.append("text").attr("x",x-4).attr("y",50+i*GS+GS/2+4).attr("text-anchor","end").attr("font-size",9).attr("fill",C.muted).text(ch);
      svg.append("text").attr("x",x+i*GS+GS/2).attr("y",50+K*GS+12).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.muted).text(ch); });
    return svg.append("g").attr("transform","translate("+x+",50)"); });
  const note=svg.append("text").attr("x",18).attr("y",214).attr("font-size",10.5).attr("fill",C.ink);
  const note2=svg.append("text").attr("x",18).attr("y",232).attr("font-size",10.5).attr("fill",C.muted);
  svg.append("text").attr("x",330).attr("y",214).attr("font-size",10).attr("fill",C.muted).text("G_ij = (1/M) ∑_k F_ik F_jk over all 256 positions");
  svg.append("text").attr("x",330).attr("y",232).attr("font-size",10).attr("fill",C.muted).text("cells in the R/G/B block can never change");
  function draw(g,im){ g.selectAll("rect").data(im).join("rect").attr("x",(d,i)=>(i%N)*PX).attr("y",(d,i)=>Math.floor(i/N)*PX).attr("width",PX).attr("height",PX).attr("fill",rgb); }
  function drawG(g,G){ g.selectAll("rect").data(G.flat()).join("rect").attr("x",(d,i)=>(i%K)*GS).attr("y",(d,i)=>Math.floor(i/K)*GS).attr("width",GS-1).attr("height",GS-1).attr("fill",hm); }
  draw(gImg[0],img); drawG(gG[0],G0);
  function render(){
    const im=perm.map(j=>img[j]), F=feats(im), G=gram(F);
    draw(gImg[1],im); drawG(gG[1],G);
    let content=0; for(let a=0;a<K;a++) for(let k=0;k<M;k++) content+=(F[a][k]-F0[a][k])**2; content/=M;
    let dc=0,nc=0,de=0,ne=0;
    for(let a=0;a<K;a++) for(let b=0;b<K;b++){ const d2=(G[a][b]-G0[a][b])**2, n2=G0[a][b]**2; if(a<3&&b<3){dc+=d2;nc+=n2;} else {de+=d2;ne+=n2;} }
    const rc=Math.sqrt(dc/nc), re=Math.sqrt(de/ne);
    const p=+pEl.value, moved=perm.filter((j,i)=>j!==i).length;
    note.text(shuffles?("shuffle #"+shuffles+" with "+p+"×"+p+" patches: "+moved+" of 256 pixels moved"):"unshuffled: the two images are identical");
    note2.text(shuffles?"the scene is destroyed; the colour statistics are not":"press Shuffle patches");
    out.innerHTML="content loss (mean squared feature difference, same positions) <b>"+content.toFixed(4)+"</b> &nbsp;·&nbsp; Gram change, colour block <b style=\"color:"+C.good+"\">"+(rc<1e-9?"0 (exact)":(100*rc).toExponential(1)+"%")+"</b> &nbsp;·&nbsp; Gram change, edge entries <b>"+(100*re).toFixed(1)+"%</b> (only at the cut lines; larger patches → fewer cuts)";
  }
  document.getElementById("gram-shuffle").addEventListener("click",()=>{ shuffles++; perm=shuffled(+pEl.value); render(); });
  document.getElementById("gram-reset").addEventListener("click",()=>{ shuffles=0; perm=d3.range(M); render(); });
  pEl.addEventListener("change",()=>{ if(shuffles){ perm=shuffled(+pEl.value); } render(); });
  render();
})();
