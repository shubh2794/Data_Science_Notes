/* linear-algebra-quadratic-pca.viz.js — extracted from linear-algebra-quadratic-pca.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ───────────────────────── Ch19 · PCA on a 2-D cloud ───────────────────────── */
(function(){
  const svg=d3.select("#pca-svg"),W=640,H=360,ox=W/2,oy=H/2,U=26;
  axes(svg,W,H,ox,oy,U); arrowDefs(svg,[["pc1","#4ade80"],["pc2","#818cf8"]]);
  // deterministic correlated cloud (seeded LCG)
  let seed=42; const rnd=()=>{ seed=(seed*1103515245+12345)&0x7fffffff; return seed/0x7fffffff; };
  const gauss=()=>{ let s=0; for(let i=0;i<6;i++) s+=rnd(); return s-3; };
  const pts=d3.range(60).map(()=>{ const t=gauss()*3.2, n=gauss()*1.0; return {x:t*0.9+n*0.5, y:t*0.5-n*0.7}; });
  const mx=d3.mean(pts,p=>p.x), my=d3.mean(pts,p=>p.y);
  pts.forEach(p=>{ p.x-=mx; p.y-=my; });
  const cxx=d3.mean(pts,p=>p.x*p.x), cyy=d3.mean(pts,p=>p.y*p.y), cxy=d3.mean(pts,p=>p.x*p.y);
  const E=eigSym2(cxx,cxy,cyy);
  const px=p=>ox+p.x*U, py=p=>oy-p.y*U;
  const projLine=svg.append("g"), dots=svg.append("g");
  dots.selectAll("circle").data(pts).join("circle").attr("cx",px).attr("cy",py).attr("r",3.5).attr("fill",PC.a2).attr("fill-opacity",.85);
  const v1=E.v1, sd1=Math.sqrt(E.l1)*2.2, sd2=Math.sqrt(E.l2)*2.2;
  svg.append("line").attr("x1",ox-v1.x*sd1*U).attr("y1",oy+v1.y*sd1*U).attr("x2",ox+v1.x*sd1*U).attr("y2",oy-v1.y*sd1*U).attr("stroke","#4ade80").attr("stroke-width",2.5).attr("marker-end","url(#pc1)");
  svg.append("line").attr("x1",ox).attr("y1",oy).attr("x2",ox+E.v2.x*sd2*U).attr("y2",oy-E.v2.y*sd2*U).attr("stroke","#818cf8").attr("stroke-width",2).attr("marker-end","url(#pc2)");
  const out=document.getElementById("pca-readout");
  const ve=E.l1/(E.l1+E.l2)*100;
  function show(proj){
    if(proj){
      projLine.selectAll("line").data(pts).join("line")
        .attr("x1",px).attr("y1",py)
        .attr("x2",p=>{ const t=p.x*v1.x+p.y*v1.y; return px({x:v1.x*t,y:v1.y*t}); })
        .attr("y2",p=>{ const t=p.x*v1.x+p.y*v1.y; return py({x:v1.x*t,y:v1.y*t}); })
        .attr("stroke",PC.muted).attr("stroke-opacity",.45).attr("stroke-dasharray","2 2");
      out.innerHTML=`Projected onto PC1 → 2-D reduced to 1-D, keeping <b style="color:${PC.good}">${ve.toFixed(1)}%</b> of the variance`;
    } else { projLine.selectAll("line").remove();
      out.innerHTML=`<b style="color:#4ade80">PC1</b> explains <b>${ve.toFixed(1)}%</b> of variance, <b style="color:#818cf8">PC2</b> the remaining <b>${(100-ve).toFixed(1)}%</b> (eigenvalues ${E.l1.toFixed(2)}, ${E.l2.toFixed(2)})`; }
  }
  document.getElementById("pca-proj").addEventListener("click",()=>show(true));
  document.getElementById("pca-reset").addEventListener("click",()=>show(false));
  show(false);
})();

/* ─────────────── Ch17 · quadratic-form contours + Rayleigh quotient ─────────────── */
(function(){
  const svg=d3.select("#qf-svg"); if(svg.empty()) return;
  const W=640,H=360,ox=W/2,oy=H/2,U=52;          // U px per unit
  const N=130;                                    // contour grid resolution
  const vx=(W/2-10)/U, vy=(H/2-10)/U;             // visible half-extent, data units
  const PAD=1.35, xr=vx*PAD, yr=vy*PAD;           // sample wider than we draw, so the
  axes(svg,W,H,ox,oy,U);                          // contours' closing edges land off-canvas
  svg.append("clipPath").attr("id","qf-clip").append("rect")
     .attr("x",ox-vx*U).attr("y",oy-vy*U).attr("width",2*vx*U).attr("height",2*vy*U);
  arrowDefs(svg,[["qf1","#4ade80"],["qf2","#818cf8"]]);

  const gCont=svg.append("g").attr("clip-path","url(#qf-clip)"), gAxesEig=svg.append("g"), gHandle=svg.append("g");
  const sx=(2*xr*U)/(N-1), sy=(2*yr*U)/(N-1);
  const path=d3.geoPath(d3.geoTransform({point:function(x,y){
    this.stream.point(ox-xr*U+x*sx, oy-yr*U+y*sy);   // grid → svg (y already flipped by grid order)
  }}));
  const LV=[0.5,1,2,4,8,16,32];
  const thresholds=LV.slice().reverse().map(v=>-v).concat([0]).concat(LV);

  // draggable direction handle, kept on a circle (R depends on direction only)
  const R0=2.6; let ang=0.6;
  const els={a:document.getElementById("qf-a"),b:document.getElementById("qf-b"),d:document.getElementById("qf-d")};
  const out=document.getElementById("qf-readout");
  svg.append("circle").attr("cx",ox).attr("cy",oy).attr("r",R0*U).attr("fill","none")
     .attr("stroke",PC.line).attr("stroke-dasharray","3 4");

  function category(l1,l2){
    const eps=1e-6, mn=Math.min(l1,l2), mx=Math.max(l1,l2);
    if(mn>eps)  return ["positive definite","bowl — climbs in every direction",PC.good];
    if(mx<-eps) return ["negative definite","dome — falls in every direction",PC.bad];
    if(mn<-eps&&mx>eps) return ["indefinite","saddle — climbs one way, falls the other","#818cf8"];
    if(Math.abs(mn)<=eps&&Math.abs(mx)<=eps) return ["both PSD and NSD","the zeros matrix — flat everywhere",PC.muted];
    if(mn>=-eps&&mx>eps) return ["positive semidefinite","bowl with a flat trough — singular",PC.good];
    return ["negative semidefinite","dome with a flat ridge — singular",PC.bad];
  }

  function render(){
    const a=+els.a.value, b=+els.b.value, d=+els.d.value;
    const q=(x,y)=>a*x*x+2*b*x*y+d*y*y;

    // sample q on the grid, rows top→bottom so grid y maps straight to svg y
    const vals=new Array(N*N);
    for(let j=0;j<N;j++){ const y=yr-(2*yr)*j/(N-1);
      for(let i=0;i<N;i++){ const x=-xr+(2*xr)*i/(N-1); vals[j*N+i]=q(x,y); } }

    const cont=d3.contours().size([N,N]).thresholds(thresholds)(vals);
    gCont.selectAll("path").data(cont).join("path")
      .attr("d",path)
      .attr("fill","none")
      .attr("stroke",c=>c.value>1e-9?PC.good:(c.value<-1e-9?PC.bad:PC.muted))
      .attr("stroke-width",c=>Math.abs(c.value)<1e-9?1.8:1.1)
      .attr("stroke-dasharray",c=>Math.abs(c.value)<1e-9?"5 4":null)
      .attr("stroke-opacity",c=>Math.abs(c.value)<1e-9?.9:Math.max(.25,.75-Math.log2(Math.abs(c.value)+1)*.11));

    // eigenvectors = principal axes of the contours
    const E=eigSym2(a,b,d);
    const L=3.0;
    const ax=[[E.v1,"#4ade80","qf1",E.l1],[E.v2,"#818cf8","qf2",E.l2]];
    gAxesEig.selectAll("line").data(ax).join("line")
      .attr("x1",ox).attr("y1",oy)
      .attr("x2",p=>ox+p[0].x*L*U).attr("y2",p=>oy-p[0].y*L*U)
      .attr("stroke",p=>p[1]).attr("stroke-width",2.2).attr("marker-end",p=>`url(#${p[2]})`);
    gAxesEig.selectAll("text").data(ax).join("text")
      .attr("x",p=>ox+p[0].x*L*U*0.55-p[0].y*16).attr("y",p=>oy-p[0].y*L*U*0.55-p[0].x*16)
      .attr("fill",p=>p[1]).attr("font-size",11.5)
      .text((p,i)=>`λ${i===0?"₁":"₂"} = ${p[3].toFixed(2)}`);

    // the draggable direction handle + its Rayleigh value
    const u={x:Math.cos(ang),y:Math.sin(ang)};
    const hx=ox+u.x*R0*U, hy=oy-u.y*R0*U;
    const R=q(u.x,u.y);                       // ‖u‖ = 1, so this IS the Rayleigh quotient
    const ray=gHandle.selectAll("line.ray").data([0]).join("line").attr("class","ray")
      .attr("x1",ox).attr("y1",oy).attr("x2",hx).attr("y2",hy)
      .attr("stroke",PC.a2).attr("stroke-width",1.6).attr("stroke-opacity",.8);
    const h=gHandle.selectAll("circle.dragpt").data([0]).join("circle").attr("class","dragpt")
      .attr("cx",hx).attr("cy",hy).attr("r",7).attr("fill",PC.a2)
      .attr("stroke","#0f1117").attr("stroke-width",2).style("cursor","grab");
    h.call(d3.drag().on("drag",e=>{ ang=Math.atan2(oy-e.y,e.x-ox); render(); }));

    const mn=Math.min(E.l1,E.l2), mx=Math.max(E.l1,E.l2);
    const cat=category(E.l1,E.l2);
    const det=a*d-b*b;
    out.innerHTML =
      `A = [[${a.toFixed(1)}, ${b.toFixed(1)}], [${b.toFixed(1)}, ${d.toFixed(1)}]]`
      + ` &nbsp;·&nbsp; tr = ${(a+d).toFixed(2)} &nbsp;·&nbsp; det = ${det.toFixed(2)}`
      + ` &nbsp;·&nbsp; λ = ${E.l1.toFixed(2)}, ${E.l2.toFixed(2)}`
      + `<br><b style="color:${cat[2]}">${cat[0]}</b> — ${cat[1]}`
      + `<br>direction x = [${u.x.toFixed(2)}, ${u.y.toFixed(2)}]ᵀ &nbsp;⟹&nbsp; R(x) = xᵀAx/xᵀx = <b>${R.toFixed(3)}</b>`
      + ` &nbsp;·&nbsp; bounded by λ_min = ${mn.toFixed(2)} ≤ R ≤ λ_max = ${mx.toFixed(2)} ✓`;
  }

  ["a","b","d"].forEach(k=>els[k].addEventListener("input",render));
  document.querySelectorAll("#viz-quad button[data-qf]").forEach(btn=>{
    btn.addEventListener("click",()=>{ const [a,b,d]=btn.dataset.qf.split(",");
      els.a.value=a; els.b.value=b; els.d.value=d; render(); });
  });
  render();
})();

/* ─────────────── Ch18 · live covariance / correlation on a draggable cloud ─────────────── */
(function(){
  const svg=d3.select("#cov-svg"); if(svg.empty()) return;
  const W=640,H=360,ox=W/2,oy=H/2,U=34;
  axes(svg,W,H,ox,oy,U);
  const px=p=>ox+p.x*U, py=p=>oy-p.y*U;

  /* deliberately off-centre, so "Mean-center" visibly does something */
  const START=[[-1.4,-1.1],[-0.6,0.3],[0.2,-0.4],[1.0,1.3],[1.8,0.6],
               [2.4,1.9],[3.2,1.6],[3.9,3.0],[4.7,2.3],[5.5,3.7]];
  let pts=START.map(p=>({x:p[0],y:p[1]}));

  const gEll=svg.append("g"), gFit=svg.append("g"), gMean=svg.append("g"), gPts=svg.append("g");
  const out=document.getElementById("cov-readout");

  function stats(){
    const n=pts.length;
    const mx=d3.mean(pts,p=>p.x), my=d3.mean(pts,p=>p.y);
    let cxx=0,cyy=0,cxy=0;
    pts.forEach(p=>{ const dx=p.x-mx, dy=p.y-my; cxx+=dx*dx; cyy+=dy*dy; cxy+=dx*dy; });
    const den=n-1;
    return {n,mx,my,cxx:cxx/den,cyy:cyy/den,cxy:cxy/den};
  }

  function render(){
    const s=stats();
    const sdx=Math.sqrt(s.cxx), sdy=Math.sqrt(s.cyy);
    const r=(sdx>1e-9&&sdy>1e-9)?s.cxy/(sdx*sdy):0;
    const E=eigSym2(s.cxx,s.cxy,s.cyy);

    // 1-σ ellipse: axes along the eigenvectors, half-lengths √λ
    const rot=Math.atan2(E.v1.y,E.v1.x)*180/Math.PI;
    gEll.selectAll("ellipse").data([0]).join("ellipse")
      .attr("cx",0).attr("cy",0)
      .attr("rx",Math.sqrt(Math.max(E.l1,0))*U).attr("ry",Math.sqrt(Math.max(E.l2,0))*U)
      .attr("fill",PC.accent).attr("fill-opacity",.07)
      .attr("stroke",PC.accent).attr("stroke-width",1.6).attr("stroke-opacity",.75)
      .attr("transform",`translate(${px({x:s.mx,y:s.my})},${py({x:s.mx,y:s.my})}) rotate(${-rot})`);

    // least-squares fit line y = ȳ + (c_xy/σx²)(x − x̄)  — the regression of y on x
    const slope=s.cxx>1e-9?s.cxy/s.cxx:0, X=9;
    gFit.selectAll("line").data([0]).join("line")
      .attr("x1",px({x:s.mx-X,y:0})).attr("y1",py({x:0,y:s.my-slope*X}))
      .attr("x2",px({x:s.mx+X,y:0})).attr("y2",py({x:0,y:s.my+slope*X}))
      .attr("stroke",PC.good).attr("stroke-width",1.6).attr("stroke-dasharray","6 4").attr("stroke-opacity",.8);

    // the mean, marked
    gMean.selectAll("path").data([0]).join("path")
      .attr("d","M-7,0H7M0,-7V7")
      .attr("transform",`translate(${px({x:s.mx,y:s.my})},${py({x:s.mx,y:s.my})})`)
      .attr("stroke",PC.a2).attr("stroke-width",2);

    gPts.selectAll("circle").data(pts).join("circle")
      .attr("class","dragpt")
      .attr("cx",px).attr("cy",py).attr("r",6)
      .attr("fill",PC.a2).attr("fill-opacity",.9)
      .attr("stroke","#0f1117").attr("stroke-width",1.5).style("cursor","grab")
      .call(d3.drag().on("drag",function(e,p){ p.x=(e.x-ox)/U; p.y=(oy-e.y)/U; render(); }));

    const rc=Math.abs(r)>0.85?PC.good:(Math.abs(r)<0.25?PC.bad:PC.a2);
    const centered=Math.hypot(s.mx,s.my)<1e-6;
    out.innerHTML =
      `mean = (${s.mx.toFixed(2)}, ${s.my.toFixed(2)})${centered?' <span style="color:'+PC.good+'">· centered</span>':""}`
      + ` &nbsp;·&nbsp; σ_x = ${sdx.toFixed(2)}, σ_y = ${sdy.toFixed(2)} &nbsp;·&nbsp; c_xy = ${s.cxy.toFixed(2)}`
      + `<br>C = [[${s.cxx.toFixed(2)}, ${s.cxy.toFixed(2)}], [${s.cxy.toFixed(2)}, ${s.cyy.toFixed(2)}]]`
      + ` &nbsp;·&nbsp; λ = ${E.l1.toFixed(2)}, ${E.l2.toFixed(2)} &nbsp;·&nbsp; ${E.l2<1e-3?'<b style="color:'+PC.bad+'">singular — the cloud is a line</b>':"PSD, full rank"}`
      + `<br>r = c_xy/(σ_x σ_y) = cos θ = <b style="color:${rc}">${r.toFixed(3)}</b>`
      + ` &nbsp;·&nbsp; regression slope c_xy/σ_x² = ${slope.toFixed(2)}`;
  }

  document.getElementById("cov-center").addEventListener("click",()=>{
    const s=stats(); pts.forEach(p=>{ p.x-=s.mx; p.y-=s.my; }); render(); });
  document.getElementById("cov-z").addEventListener("click",()=>{
    const s=stats(), sdx=Math.sqrt(s.cxx)||1, sdy=Math.sqrt(s.cyy)||1;
    pts.forEach(p=>{ p.x=(p.x-s.mx)/sdx; p.y=(p.y-s.my)/sdy; }); render(); });
  document.getElementById("cov-reset").addEventListener("click",()=>{
    pts=START.map(p=>({x:p[0],y:p[1]})); render(); });
  render();
})();
