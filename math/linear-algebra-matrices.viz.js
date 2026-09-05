/* linear-algebra-matrices.viz.js — extracted from linear-algebra-matrices.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ───────────────────────── Ch6 · 2×2 transformation ───────────────────────── */
(function(){
  const svg=d3.select("#xf-svg"),W=640,H=360,ox=W/2,oy=H/2,U=46;
  axes(svg,W,H,ox,oy,U); arrowDefs(svg,[["xi",PC.accent],["xj",PC.a2]]);
  const px=(x,y)=>ox+x*U, py=(x,y)=>oy-y*U;
  const square=svg.append("polygon").attr("fill",PC.accent).attr("fill-opacity",.10).attr("stroke",PC.accent).attr("stroke-opacity",.4).attr("stroke-dasharray","4 3");
  const image=svg.append("polygon").attr("fill",PC.good).attr("fill-opacity",.16).attr("stroke",PC.good).attr("stroke-width",1.5);
  const ivec=svg.append("line").attr("stroke",PC.accent).attr("stroke-width",2.5).attr("marker-end","url(#xi)");
  const jvec=svg.append("line").attr("stroke",PC.a2).attr("stroke-width",2.5).attr("marker-end","url(#xj)");
  const out=document.getElementById("xf-readout");
  const ids=["xf-a","xf-b","xf-c","xf-d"], defs=[1,0.5,0.3,1];
  function update(){
    const [a,b,c,d]=ids.map(id=>+document.getElementById(id).value);
    square.attr("points",`${px(0,0)},${py(0,0)} ${px(1,0)},${py(1,0)} ${px(1,1)},${py(1,1)} ${px(0,1)},${py(0,1)}`);
    // columns of [[a,b],[c,d]] are the images of i=(1,0) and j=(0,1)
    const i=[a,c], j=[b,d], ij=[a+b,c+d];
    image.attr("points",`${px(0,0)},${py(0,0)} ${px(i[0],i[1])},${py(i[0],i[1])} ${px(ij[0],ij[1])},${py(ij[0],ij[1])} ${px(j[0],j[1])},${py(j[0],j[1])}`);
    ivec.attr("x1",ox).attr("y1",oy).attr("x2",px(i[0],i[1])).attr("y2",py(i[0],i[1]));
    jvec.attr("x1",ox).attr("y1",oy).attr("x2",px(j[0],j[1])).attr("y2",py(j[0],j[1]));
    const det=a*d-b*c, near0=Math.abs(det)<0.06;
    out.innerHTML=`[[${a}, ${b}], [${c}, ${d}]] &nbsp;·&nbsp; det = <b style="color:${near0?PC.bad:PC.good}">${det.toFixed(2)}</b> (area scale) &nbsp;·&nbsp; `
      + (near0?`<b style="color:${PC.bad}">det ≈ 0 → rank-deficient: the square collapsed to a line, no inverse</b>`:`rank 2 · invertible`);
  }
  ids.forEach(id=>document.getElementById(id).addEventListener("input",update));
  document.getElementById("xf-reset").addEventListener("click",()=>{ ids.forEach((id,k)=>document.getElementById(id).value=defs[k]); update(); });
  update();
})();
/* ───────────────────────── similarity · one map, many matrices ───────────────────────── */
(function(){
  const svg=d3.select("#sim-svg"),W=640,H=400,ox=W/2,oy=H/2,U=38;
  axes(svg,W,H,ox,oy,U); arrowDefs(svg,[["sb1",PC.accent],["sb2",PC.a2],["si1",PC.good],["si2",PC.bad]]);
  const px=v=>ox+v.x*U, py=v=>oy-v.y*U;
  let b1={x:2,y:0}, b2={x:0,y:2}, A={p:2,q:1,r:2};
  const tie1=svg.append("line").attr("stroke",PC.good).attr("stroke-width",1).attr("stroke-dasharray","2 4").attr("stroke-opacity",.45);
  const tie2=svg.append("line").attr("stroke",PC.bad).attr("stroke-width",1).attr("stroke-dasharray","2 4").attr("stroke-opacity",.45);
  const l1=svg.append("line").attr("stroke",PC.accent).attr("stroke-width",3).attr("marker-end","url(#sb1)");
  const l2=svg.append("line").attr("stroke",PC.a2).attr("stroke-width",3).attr("marker-end","url(#sb2)");
  /* images drawn last, dashed, so a coincident image (an eigenvector!) stays visible over its basis arrow */
  const i1=svg.append("line").attr("stroke",PC.good).attr("stroke-width",2).attr("stroke-dasharray","7 4").attr("marker-end","url(#si1)");
  const i2=svg.append("line").attr("stroke",PC.bad).attr("stroke-width",2).attr("stroke-dasharray","7 4").attr("marker-end","url(#si2)");
  const out=document.getElementById("sim-readout");
  function handle(v,col){
    const h=svg.append("circle").attr("class","dragpt").attr("r",10).attr("fill",col).attr("stroke","#0f1117").attr("stroke-width",2).style("cursor","grab");
    h.call(d3.drag().on("drag",e=>{ v.x=Math.round((e.x-ox)/U*10)/10; v.y=Math.round((oy-e.y)/U*10)/10; update(); }));
    return h;
  }
  const h1=handle(b1,PC.accent), h2=handle(b2,PC.a2);
  const f=n=>(Math.round(n*100)/100).toFixed(2);
  const mul=v=>({x:A.p*v.x+A.q*v.y, y:A.q*v.x+A.r*v.y});
  function update(){
    const a1=mul(b1), a2=mul(b2), dP=b1.x*b2.y-b2.x*b1.y;
    l1.attr("x1",ox).attr("y1",oy).attr("x2",px(b1)).attr("y2",py(b1));
    l2.attr("x1",ox).attr("y1",oy).attr("x2",px(b2)).attr("y2",py(b2));
    i1.attr("x1",ox).attr("y1",oy).attr("x2",px(a1)).attr("y2",py(a1));
    i2.attr("x1",ox).attr("y1",oy).attr("x2",px(a2)).attr("y2",py(a2));
    tie1.attr("x1",px(b1)).attr("y1",py(b1)).attr("x2",px(a1)).attr("y2",py(a1));
    tie2.attr("x1",px(b2)).attr("y1",py(b2)).attr("x2",px(a2)).attr("y2",py(a2));
    h1.attr("cx",px(b1)).attr("cy",py(b1)); h2.attr("cx",px(b2)).attr("cy",py(b2));
    const trA=A.p+A.r, detA=A.p*A.r-A.q*A.q;
    if(Math.abs(dP)<0.08){
      out.innerHTML='b₁ and b₂ are <b style="color:'+PC.bad+'">linearly dependent</b> (det P ≈ 0), so they are not a basis and P⁻¹ does not exist — there is no Â to compute.'
        +'<br>A = [['+f(A.p)+', '+f(A.q)+'], ['+f(A.q)+', '+f(A.r)+']] &nbsp;·&nbsp; tr = <b>'+f(trA)+'</b> &nbsp; det = <b>'+f(detA)+'</b>';
      return;
    }
    const m11=( b2.y*a1.x - b2.x*a1.y)/dP, m21=(-b1.y*a1.x + b1.x*a1.y)/dP;
    const m12=( b2.y*a2.x - b2.x*a2.y)/dP, m22=(-b1.y*a2.x + b1.x*a2.y)/dP;
    const diag=Math.abs(m12)<0.04 && Math.abs(m21)<0.04;
    out.innerHTML='Â = P⁻¹AP = <b style="color:'+(diag?PC.good:PC.accent)+'">[['+f(m11)+', '+f(m12)+'], ['+f(m21)+', '+f(m22)+']]</b>'
      + ' &nbsp;·&nbsp; tr(Â) = <b>'+f(m11+m22)+'</b> = tr(A) = '+f(trA)
      + ' &nbsp;·&nbsp; det(Â) = <b>'+f(m11*m22-m12*m21)+'</b> = det(A) = '+f(detA)
      + '<br>column j of Â = the coordinates of A·b<sub>j</sub> in the basis ⟨b₁, b₂⟩'
      + (diag ? ' &nbsp;·&nbsp; <b style="color:'+PC.good+'">Â is diagonal — each image lies along its own basis vector, so b₁, b₂ are eigenvectors with eigenvalues '+f(m11)+' and '+f(m22)+'.</b>'
              : ' &nbsp;·&nbsp; off-diagonals ≠ 0: the images spill onto the other basis direction.');
  }
  document.getElementById("sim-A").addEventListener("change",e=>{
    const v=e.target.value.split(",").map(Number); A={p:v[0],q:v[1],r:v[2]}; update();
  });
  document.getElementById("sim-std").addEventListener("click",()=>{ b1={x:2,y:0}; b2={x:0,y:2}; update(); });
  document.getElementById("sim-eig").addEventListener("click",()=>{
    const e=eigSym2(A.p,A.q,A.r);
    b1={x:Math.round(e.v1.x*2*10)/10, y:Math.round(e.v1.y*2*10)/10};
    b2={x:Math.round(e.v2.x*2*10)/10, y:Math.round(e.v2.y*2*10)/10};
    update();
  });
  update();
})();
