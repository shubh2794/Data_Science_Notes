/* linear-algebra-projections.viz.js — extracted from linear-algebra-projections.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ───────────────────────── Ch13 · projection onto a line ───────────────────────── */
(function(){
  const svg=d3.select("#proj-svg"),W=640,H=360,ox=W/2,oy=H/2,U=46;
  axes(svg,W,H,ox,oy,U); arrowDefs(svg,[["pb",PC.a2]]);
  const px=v=>ox+v.x*U, py=v=>oy-v.y*U;
  let b={x:1.4,y:2.1}, a={x:2.6,y:0.7};
  const line=svg.append("line").attr("stroke",PC.accent).attr("stroke-width",1.5).attr("stroke-opacity",.55);
  const err=svg.append("line").attr("stroke",PC.bad).attr("stroke-width",1.8).attr("stroke-dasharray","4 3");
  const pvec=svg.append("line").attr("stroke",PC.good).attr("stroke-width",3.5).attr("stroke-opacity",.85);
  const bvec=svg.append("line").attr("stroke",PC.a2).attr("stroke-width",2.5).attr("marker-end","url(#pb)");
  const rt=svg.append("path").attr("fill","none").attr("stroke",PC.bad).attr("stroke-opacity",.7).attr("stroke-width",1.2);
  const pdot=svg.append("circle").attr("r",5).attr("fill",PC.good).attr("stroke","#0f1117").attr("stroke-width",1.5);
  const out=document.getElementById("proj-readout");
  function handle(v,col,r){ const h=svg.append("circle").attr("class","dragpt").attr("r",r).attr("fill",col).attr("stroke","#0f1117").attr("stroke-width",2).style("cursor","grab");
    h.call(d3.drag().on("drag",e=>{ v.x=(e.x-ox)/U; v.y=(oy-e.y)/U; update(); })); return h; }
  function update(){
    const na2=a.x*a.x+a.y*a.y, t=(a.x*b.x+a.y*b.y)/(na2||1e-9), p={x:a.x*t,y:a.y*t};
    // infinite line through origin along a
    const ext={x:a.x/Math.hypot(a.x,a.y)*6,y:a.y/Math.hypot(a.x,a.y)*6};
    line.attr("x1",ox-ext.x*U).attr("y1",oy+ext.y*U).attr("x2",ox+ext.x*U).attr("y2",oy-ext.y*U);
    bvec.attr("x1",ox).attr("y1",oy).attr("x2",px(b)).attr("y2",py(b));
    pvec.attr("x1",ox).attr("y1",oy).attr("x2",px(p)).attr("y2",py(p));
    err.attr("x1",px(b)).attr("y1",py(b)).attr("x2",px(p)).attr("y2",py(p));
    pdot.attr("cx",px(p)).attr("cy",py(p));
    ha.attr("cx",px(a)).attr("cy",py(a)); hb.attr("cx",px(b)).attr("cy",py(b));
    // little right-angle square at the foot p
    const ua={x:a.x/Math.hypot(a.x,a.y),y:a.y/Math.hypot(a.x,a.y)};
    const ue={x:(b.x-p.x),y:(b.y-p.y)}, ne=Math.hypot(ue.x,ue.y)||1e-9; ue.x/=ne; ue.y/=ne;
    const s=0.28, c1={x:p.x+ua.x*s,y:p.y+ua.y*s}, c2={x:c1.x+ue.x*s,y:c1.y+ue.y*s}, c3={x:p.x+ue.x*s,y:p.y+ue.y*s};
    rt.attr("d",`M${px(c1)},${py(c1)} L${px(c2)},${py(c2)} L${px(c3)},${py(c3)}`);
    const errLen=Math.hypot(b.x-p.x,b.y-p.y);
    out.innerHTML=`projection p = (aᵀb/aᵀa)·a = (<b style="color:${PC.good}">${p.x.toFixed(2)}, ${p.y.toFixed(2)}</b>) &nbsp;·&nbsp; error ‖b − p‖ = <b style="color:${PC.bad}">${errLen.toFixed(2)}</b> &nbsp;(⟂ to the line)`;
  }
  const hb=handle(b,PC.a2,10), ha=handle(a,PC.accent,8); update();
})();
/* ─────────────── Ch13 · orthogonal vs oblique projection ─────────────── */
(function(){
  const svg=d3.select("#obl-svg"),W=640,H=380,ox=W/2,oy=H/2,U=48;
  axes(svg,W,H,ox,oy,U); arrowDefs(svg,[["obx",PC.a2],["oba",PC.accent],["obc",PC.muted]]);
  const px=v=>ox+v.x*U, py=v=>oy-v.y*U;
  const D0={x:{x:0.2,y:2.3}, a:{x:1.5,y:0.32}, c:{x:-2.0,y:1.4}};
  let x={...D0.x}, a={...D0.a}, c={...D0.c};
  const aline=svg.append("line").attr("stroke",PC.accent).attr("stroke-width",2).attr("stroke-opacity",.75);
  const cguide=svg.append("line").attr("stroke",PC.muted).attr("stroke-width",1.2).attr("stroke-dasharray","3 4").attr("stroke-opacity",.55);
  const drop=svg.append("line").attr("stroke",PC.good).attr("stroke-width",1.7).attr("stroke-dasharray","4 3");
  const slide=svg.append("line").attr("stroke",PC.a2).attr("stroke-width",1.7).attr("stroke-dasharray","4 3");
  const cvec=svg.append("line").attr("stroke",PC.muted).attr("stroke-width",2).attr("marker-end","url(#obc)");
  const xvec=svg.append("line").attr("stroke",PC.muted).attr("stroke-width",1.6).attr("stroke-opacity",.45);
  const lab=svg.append("g");
  lab.append("rect").attr("x",8).attr("y",6).attr("width",286).attr("height",66).attr("rx",6).attr("fill","#0f1117").attr("opacity",.72);
  lab.append("text").attr("x",14).attr("y",18).attr("fill",PC.accent).attr("font-size",11).text("S₁ = span(a) — the target line");
  lab.append("text").attr("x",14).attr("y",34).attr("fill",PC.muted).attr("font-size",11).text("S₂ = span(c) — the direction we project along");
  lab.append("text").attr("x",14).attr("y",50).attr("fill",PC.good).attr("font-size",11).text("● orthogonal foot (drop a perpendicular)");
  lab.append("text").attr("x",14).attr("y",66).attr("fill",PC.a2).attr("font-size",11).text("● oblique foot (slide along c)");
  const rt=svg.append("path").attr("fill","none").attr("stroke",PC.good).attr("stroke-opacity",.75).attr("stroke-width",1.2);
  const pOrth=svg.append("circle").attr("r",6).attr("fill",PC.good).attr("stroke","#0f1117").attr("stroke-width",1.6);
  const pObl =svg.append("circle").attr("r",6).attr("fill",PC.a2).attr("stroke","#0f1117").attr("stroke-width",1.6);
  const out=document.getElementById("obl-readout");
  function handle(v,col,r){
    const h=svg.append("circle").attr("class","dragpt").attr("r",r).attr("fill",col)
      .attr("stroke","#0f1117").attr("stroke-width",2).style("cursor","grab");
    h.call(d3.drag().on("drag",e=>{ v.x=(e.x-ox)/U; v.y=(oy-e.y)/U; update(); }));
    return h;
  }
  const T=9;
  function spanLine(sel,v,base){
    const n=Math.hypot(v.x,v.y)||1e-9, d={x:v.x/n*T, y:v.y/n*T};
    sel.attr("x1",px({x:base.x-d.x,y:base.y-d.y})).attr("y1",py({x:base.x-d.x,y:base.y-d.y}))
       .attr("x2",px({x:base.x+d.x,y:base.y+d.y})).attr("y2",py({x:base.x+d.x,y:base.y+d.y}));
  }
  function update(){
    const na=Math.hypot(a.x,a.y)||1e-9, nc=Math.hypot(c.x,c.y)||1e-9;
    // orthogonal projection of x onto span(a)
    const t0=(a.x*x.x+a.y*x.y)/(na*na), po={x:a.x*t0, y:a.y*t0};
    // oblique: solve [a c][t s]ᵀ = x, keep t·a
    const det=a.x*c.y-a.y*c.x;
    const degen=Math.abs(det)<1e-3*na*nc;
    const t1=degen?t0:( x.x*c.y-x.y*c.x)/det;
    const pb=degen?{...po}:{x:a.x*t1, y:a.y*t1};
    spanLine(aline,a,{x:0,y:0});
    spanLine(cguide,c,{x:0,y:0});
    cvec.attr("x1",ox).attr("y1",oy).attr("x2",px(c)).attr("y2",py(c));
    xvec.attr("x1",ox).attr("y1",oy).attr("x2",px(x)).attr("y2",py(x));
    drop.attr("x1",px(x)).attr("y1",py(x)).attr("x2",px(po)).attr("y2",py(po));
    slide.attr("x1",px(x)).attr("y1",py(x)).attr("x2",px(pb)).attr("y2",py(pb));
    pOrth.attr("cx",px(po)).attr("cy",py(po));
    pObl.attr("cx",px(pb)).attr("cy",py(pb)).attr("opacity",degen?0:1);
    // right-angle mark at the orthogonal foot
    const ua={x:a.x/na,y:a.y/na};
    let ex=x.x-po.x, ey=x.y-po.y; const ne=Math.hypot(ex,ey)||1e-9; ex/=ne; ey/=ne;
    const s=0.3, c1={x:po.x+ua.x*s,y:po.y+ua.y*s}, c2={x:c1.x+ex*s,y:c1.y+ey*s}, c3={x:po.x+ex*s,y:po.y+ey*s};
    rt.attr("d",`M${px(c1)},${py(c1)} L${px(c2)},${py(c2)} L${px(c3)},${py(c3)}`);
    ha.attr("cx",px(a)).attr("cy",py(a)); hc.attr("cx",px(c)).attr("cy",py(c)); hx.attr("cx",px(x)).attr("cy",py(x));
    // angle between the two directions, and the oblique projector norm 1/sin θ
    const sin=Math.abs(det)/(na*nc), cos=(a.x*c.x+a.y*c.y)/(na*nc);
    let deg=Math.atan2(Math.abs(det), a.x*c.x+a.y*c.y)*180/Math.PI;
    if(deg>90) deg=180-deg;                          // angle between the two LINES, in (0°, 90°]
    const pnorm=sin>1e-9?1/sin:Infinity;
    const gap=Math.hypot(pb.x-po.x,pb.y-po.y);
    out.innerHTML = degen
      ? `<b style="color:${PC.bad}">c is parallel to a</b> — the two subspaces no longer complement each other and the oblique projection is undefined.`
      : `orthogonal foot <b style="color:${PC.good}">(${po.x.toFixed(2)}, ${po.y.toFixed(2)})</b> &nbsp;·&nbsp; `+
        `oblique foot <b style="color:${PC.a2}">(${pb.x.toFixed(2)}, ${pb.y.toFixed(2)})</b> &nbsp;·&nbsp; gap ${gap.toFixed(2)}<br>`+
        `angle between the two subspaces: <b>${deg.toFixed(1)}°</b> &nbsp;·&nbsp; ‖P_oblique‖₂ = 1/sin θ = `+
        `<b style="color:${pnorm>3?PC.bad:PC.good}">${pnorm.toFixed(2)}</b> `+
        `<span style="opacity:.7">(‖P_orthogonal‖₂ = 1 always)</span>`+
        (Math.abs(cos)<1e-3?` &nbsp;— <b style="color:${PC.good}">c ⟂ a, so the two projections coincide</b>`:"");
  }
  const hx=handle(x,PC.ink,10), ha=handle(a,PC.accent,8), hc=handle(c,PC.muted,8);
  document.getElementById("ob-perp").addEventListener("click",()=>{
    const n=Math.hypot(a.x,a.y)||1e-9, L=Math.hypot(c.x,c.y)||1;
    c.x=-a.y/n*L; c.y=a.x/n*L; update();
  });
  document.getElementById("ob-reset").addEventListener("click",()=>{
    x={...D0.x}; a={...D0.a}; c={...D0.c}; update();
  });
  update();
})();
/* ───────────────────────── Ch14 · least-squares line fit ───────────────────────── */
(function(){
  const svg=d3.select("#ls-svg"),W=640,H=360,pad=40;
  const pts=[[1,1.6],[2,2.1],[3,2.0],[4,3.4],[5,3.1],[6,4.4],[7,4.2],[8,5.6],[9,5.3]];
  const sx=d3.scaleLinear().domain([0,10]).range([pad,W-pad]);
  const sy=d3.scaleLinear().domain([0,7]).range([H-pad,pad]);
  // optimal least-squares fit
  const n=pts.length, mx=d3.mean(pts,d=>d[0]), my=d3.mean(pts,d=>d[1]);
  const mOpt=d3.sum(pts,d=>(d[0]-mx)*(d[1]-my))/d3.sum(pts,d=>(d[0]-mx)**2), kOpt=my-mOpt*mx;
  const sseOpt=d3.sum(pts,d=>(d[1]-(mOpt*d[0]+kOpt))**2);
  const g=svg.append("g");
  g.append("line").attr("x1",pad).attr("y1",H-pad).attr("x2",W-pad).attr("y2",H-pad).attr("stroke","#3a4150");
  g.append("line").attr("x1",pad).attr("y1",H-pad).attr("x2",pad).attr("y2",pad).attr("stroke","#3a4150");
  const resid=svg.append("g"), lineEl=svg.append("line").attr("stroke",PC.accent).attr("stroke-width",2.5);
  svg.append("g").selectAll("circle").data(pts).join("circle").attr("cx",d=>sx(d[0])).attr("cy",d=>sy(d[1])).attr("r",5).attr("fill",PC.a2).attr("stroke","#0f1117").attr("stroke-width",1.5);
  const mEl=document.getElementById("ls-m"), kEl=document.getElementById("ls-k"), out=document.getElementById("ls-readout");
  function update(){
    const m=+mEl.value, k=+kEl.value;
    lineEl.attr("x1",sx(0)).attr("y1",sy(k)).attr("x2",sx(10)).attr("y2",sy(m*10+k));
    resid.selectAll("line").data(pts).join("line")
      .attr("x1",d=>sx(d[0])).attr("y1",d=>sy(d[1])).attr("x2",d=>sx(d[0])).attr("y2",d=>sy(m*d[0]+k))
      .attr("stroke",PC.muted).attr("stroke-opacity",.6).attr("stroke-dasharray","2 2");
    const sse=d3.sum(pts,d=>(d[1]-(m*d[0]+k))**2);
    const best=Math.abs(sse-sseOpt)<0.02;
    out.innerHTML=`y = <b>${m.toFixed(2)}</b>x + <b>${k.toFixed(2)}</b> &nbsp;·&nbsp; SSE = <b style="color:${best?PC.good:PC.a2}">${sse.toFixed(2)}</b> &nbsp; (minimum ${sseOpt.toFixed(2)} at m=${mOpt.toFixed(2)}, k=${kOpt.toFixed(2)})`;
  }
  mEl.addEventListener("input",update); kEl.addEventListener("input",update);
  document.getElementById("ls-best").addEventListener("click",()=>{ mEl.value=mOpt.toFixed(2); kEl.value=kOpt.toFixed(2); update(); });
  update();
})();
