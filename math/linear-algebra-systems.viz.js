/* linear-algebra-systems.viz.js — extracted from linear-algebra-systems.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ───────────────────────── Ch10 · two-line system ───────────────────────── */
(function(){
  const svg=d3.select("#sys-svg"),W=640,H=360,ox=W/2,oy=H/2,U=40;
  axes(svg,W,H,ox,oy,U);
  const px=x=>ox+x*U, py=y=>oy-y*U, X=(W/2)/U;   // x ranges roughly [-X, X]
  const l1=svg.append("line").attr("stroke",PC.accent).attr("stroke-width",2.5);
  const l2=svg.append("line").attr("stroke",PC.a2).attr("stroke-width",2.5);
  const dot=svg.append("circle").attr("r",6).attr("fill",PC.good).attr("stroke","#0f1117").attr("stroke-width",2);
  const out=document.getElementById("sys-readout");
  const g=id=>+document.getElementById(id).value;
  function update(){
    const m1=g("sy-m1"),k1=g("sy-k1"),m2=g("sy-m2"),k2=g("sy-k2");
    l1.attr("x1",px(-X)).attr("y1",py(m1*-X+k1)).attr("x2",px(X)).attr("y2",py(m1*X+k1));
    l2.attr("x1",px(-X)).attr("y1",py(m2*-X+k2)).attr("x2",px(X)).attr("y2",py(m2*X+k2));
    if(Math.abs(m1-m2)<1e-6){
      dot.attr("opacity",0);
      const same=Math.abs(k1-k2)<1e-6;
      out.innerHTML = same
        ? `<b style="color:${PC.a2}">Identical lines → infinitely many solutions</b> (rank &lt; n)`
        : `<b style="color:${PC.bad}">Parallel lines → no solution</b> (inconsistent)`;
    } else {
      const xs=(k2-k1)/(m1-m2), ys=m1*xs+k1;
      dot.attr("opacity",1).attr("cx",px(xs)).attr("cy",py(ys));
      out.innerHTML=`<b style="color:${PC.good}">Unique solution</b> at (x, y) = (<b>${xs.toFixed(2)}</b>, <b>${ys.toFixed(2)}</b>)`;
    }
  }
  ["sy-m1","sy-k1","sy-m2","sy-k2"].forEach(id=>document.getElementById(id).addEventListener("input",update));
  update();
})();
/* ──────────────── conditioning · two nearly-parallel lines ──────────────── */
(function(){
  const svg=d3.select("#cond-svg"),W=640,H=380,ox=W/2,oy=H/2+18,U=52;
  axes(svg,W,H,ox,oy,U);
  const px=x=>ox+x*U, py=y=>oy-y*U;
  const l1=svg.append("line").attr("stroke",PC.accent).attr("stroke-width",2.2).attr("stroke-opacity",.9);
  const l2=svg.append("line").attr("stroke",PC.a2).attr("stroke-width",2.2).attr("stroke-opacity",.9);
  const l2p=svg.append("line").attr("stroke",PC.bad).attr("stroke-width",2).attr("stroke-dasharray","7 4").attr("stroke-opacity",.95);
  const shift=svg.append("line").attr("stroke",PC.bad).attr("stroke-width",1.4).attr("stroke-dasharray","2 3").attr("stroke-opacity",.8);
  const dot=svg.append("circle").attr("r",9).attr("fill","none").attr("stroke",PC.good).attr("stroke-width",2.5);
  const dotp=svg.append("circle").attr("r",4.5).attr("fill",PC.bad).attr("stroke","#0f1117").attr("stroke-width",1.2);
  const bar=svg.append("g"), lab=svg.append("g");
  lab.append("rect").attr("x",8).attr("y",6).attr("width",250).attr("height",66).attr("rx",6).attr("fill","#0f1117").attr("opacity",.72);
  lab.append("text").attr("x",14).attr("y",18).attr("fill",PC.accent).attr("font-size",11).text("line 1");
  lab.append("text").attr("x",14).attr("y",34).attr("fill",PC.a2).attr("font-size",11).text("line 2");
  lab.append("text").attr("x",14).attr("y",50).attr("fill",PC.bad).attr("font-size",11).text("line 2, perturbed");
  lab.append("text").attr("x",14).attr("y",66).attr("fill",PC.good).attr("font-size",11).text("○ true solution   • perturbed solution");
  const out=document.getElementById("cond-readout");
  const angEl=document.getElementById("cd-ang"), perEl=document.getElementById("cd-pert");
  const CLAMP=7.2;                                   // keep drawing inside the canvas
  function drawLine(sel,n,c){                        // line { x : n·x = c }, n a unit normal
    // point on the line closest to the origin, plus its direction
    const p={x:n.x*c, y:n.y*c}, d={x:-n.y, y:n.x}, T=14;
    sel.attr("x1",px(p.x-d.x*T)).attr("y1",py(p.y-d.y*T)).attr("x2",px(p.x+d.x*T)).attr("y2",py(p.y+d.y*T));
  }
  function solve2(A,b){                              // A = [[a,b],[c,d]] as rows n1,n2
    const det=A[0].x*A[1].y-A[0].y*A[1].x;
    return {x:( b[0]*A[1].y-b[1]*A[0].y)/det, y:(A[0].x*b[1]-A[1].x*b[0])/det};
  }
  function cond2(n1,n2){                             // spectral condition number of [[n1],[n2]]
    const a=n1.x,b=n1.y,c=n2.x,d=n2.y;
    const p=a*a+c*c, q=a*b+c*d, r=b*b+d*d;           // AᵀA = [[p,q],[q,r]]
    const e=eigSym2(p,q,r);
    const s1=Math.sqrt(Math.max(e.l1,0)), s2=Math.sqrt(Math.max(e.l2,0));
    return s2>1e-12 ? s1/s2 : Infinity;
  }
  function update(){
    const phi=+angEl.value, rel=+perEl.value, rad=phi*Math.PI/180;
    const base=-Math.PI/9;                                   // tilt the pair for a nicer picture
    const n1={x:Math.cos(base),           y:Math.sin(base)};
    const n2={x:Math.cos(base+rad),       y:Math.sin(base+rad)};
    const xs={x:1.6,y:1.1};                                   // the intended solution, held fixed
    const b=[n1.x*xs.x+n1.y*xs.y, n2.x*xs.x+n2.y*xs.y];
    const nb=Math.hypot(b[0],b[1]);
    const db=[0, rel*nb];                                     // perturb only the 2nd constant
    const bp=[b[0]+db[0], b[1]+db[1]];
    const xp=solve2([n1,n2],bp);
    drawLine(l1,n1,b[0]); drawLine(l2,n2,b[1]); drawLine(l2p,n2,bp[1]);
    dot.attr("cx",px(xs.x)).attr("cy",py(xs.y));
    const cl=v=>Math.max(-CLAMP,Math.min(CLAMP,v));
    const off=Math.abs(xp.x)>CLAMP||Math.abs(xp.y)>CLAMP;
    dotp.attr("cx",px(cl(xp.x))).attr("cy",py(cl(xp.y))).attr("opacity",rel>0?1:0).attr("r",off?4:5);
    shift.attr("x1",px(xs.x)).attr("y1",py(xs.y)).attr("x2",px(cl(xp.x))).attr("y2",py(cl(xp.y))).attr("opacity",rel>0?1:0);
    const k=cond2(n1,n2), cot=1/Math.tan(rad/2);
    const relx=Math.hypot(xp.x-xs.x,xp.y-xs.y)/Math.hypot(xs.x,xs.y);
    const amp=rel>0 ? relx/rel : 0;
    // bound bar: how much of the worst case κ this particular perturbation realises
    const frac=k>0&&isFinite(k)?Math.min(1,amp/k):0;
    bar.selectAll("*").remove();
    const BY=H-16;
    bar.append("rect").attr("x",30).attr("y",BY).attr("width",W-60).attr("height",9).attr("rx",4).attr("fill","#1b2130");
    bar.append("rect").attr("x",30).attr("y",BY).attr("width",(W-60)*frac).attr("height",9).attr("rx",4).attr("fill",PC.bad).attr("opacity",.85);
    bar.append("text").attr("x",30).attr("y",BY-6).attr("fill",PC.muted).attr("font-size",10)
       .text("realised amplification, as a fraction of the worst case κ₂");
    out.innerHTML =
      `φ = <b>${phi.toFixed(1)}°</b> &nbsp;·&nbsp; κ₂(A) = <b style="color:${k>100?PC.bad:PC.good}">${isFinite(k)?k.toFixed(1):"∞"}</b> `+
      `<span style="opacity:.7">(= cot(φ/2) = ${cot.toFixed(1)})</span><br>`+
      `‖δb‖/‖b‖ = <b>${(rel*100).toFixed(1)}%</b> &nbsp;⟶&nbsp; ‖δx‖/‖x‖ = <b style="color:${PC.bad}">${(relx*100).toFixed(1)}%</b> `+
      `&nbsp;·&nbsp; amplification <b>${amp.toFixed(1)}×</b> &nbsp;<span style="opacity:.7">(bound: ≤ κ₂ = ${isFinite(k)?k.toFixed(1):"∞"})</span>`+
      (off?`<br><span style="color:${PC.bad}">perturbed solution is off-canvas — clamped to the edge</span>`:"");
  }
  angEl.addEventListener("input",update); perEl.addEventListener("input",update);
  document.getElementById("cd-reset").addEventListener("click",()=>{angEl.value=60;perEl.value=0.01;update();});
  update();
})();
