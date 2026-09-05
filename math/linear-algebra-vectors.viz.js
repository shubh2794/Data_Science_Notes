/* linear-algebra-vectors.viz.js — extracted from linear-algebra-vectors.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ───────────────────────── Ch2 · vector playground ───────────────────────── */
(function(){
  const svg=d3.select("#vec-svg"),W=640,H=360,ox=W/2,oy=H/2,U=46;
  axes(svg,W,H,ox,oy,U); arrowDefs(svg,[["va",PC.accent],["vb",PC.a2],["vs",PC.good]]);
  const px=v=>ox+v.x*U, py=v=>oy-v.y*U;
  let a={x:2,y:1}, b={x:-1,y:1.6};
  const lsum=svg.append("line").attr("stroke",PC.good).attr("stroke-width",2.5).attr("marker-end","url(#vs)");
  const p1=svg.append("line").attr("stroke",PC.muted).attr("stroke-dasharray","3 3").attr("stroke-opacity",.5);
  const p2=svg.append("line").attr("stroke",PC.muted).attr("stroke-dasharray","3 3").attr("stroke-opacity",.5);
  const la=svg.append("line").attr("stroke",PC.accent).attr("stroke-width",2.5).attr("marker-end","url(#va)");
  const lb=svg.append("line").attr("stroke",PC.a2).attr("stroke-width",2.5).attr("marker-end","url(#vb)");
  const out=document.getElementById("vec-readout");
  function handle(v,col){ const h=svg.append("circle").attr("class","dragpt").attr("r",10).attr("fill",col).attr("stroke","#0f1117").attr("stroke-width",2).style("cursor","grab");
    h.call(d3.drag().on("drag",e=>{ v.x=Math.round((e.x-ox)/U*2)/2; v.y=Math.round((oy-e.y)/U*2)/2; update(); })); return h; }
  function update(){
    const s={x:a.x+b.x,y:a.y+b.y};
    la.attr("x1",ox).attr("y1",oy).attr("x2",px(a)).attr("y2",py(a));
    lb.attr("x1",ox).attr("y1",oy).attr("x2",px(b)).attr("y2",py(b));
    lsum.attr("x1",ox).attr("y1",oy).attr("x2",px(s)).attr("y2",py(s));
    p1.attr("x1",px(a)).attr("y1",py(a)).attr("x2",px(s)).attr("y2",py(s));
    p2.attr("x1",px(b)).attr("y1",py(b)).attr("x2",px(s)).attr("y2",py(s));
    ha.attr("cx",px(a)).attr("cy",py(a)); hb.attr("cx",px(b)).attr("cy",py(b));
    const mag=v=>Math.hypot(v.x,v.y).toFixed(2);
    out.innerHTML=`a=(${a.x}, ${a.y}) &nbsp;‖a‖=<b>${mag(a)}</b> &nbsp;·&nbsp; b=(${b.x}, ${b.y}) &nbsp;‖b‖=<b>${mag(b)}</b> &nbsp;·&nbsp; <b style="color:${PC.good}">a+b=(${s.x}, ${s.y})</b>`;
  }
  const ha=handle(a,PC.accent), hb=handle(b,PC.a2); update();
})();
/* ───────────────────────── Ch3 · dot product geometry ───────────────────────── */
(function(){
  const svg=d3.select("#dot-svg"),W=640,H=360,ox=W/2,oy=H/2,U=46;
  axes(svg,W,H,ox,oy,U); arrowDefs(svg,[["da",PC.accent],["db",PC.a2]]);
  const px=v=>ox+v.x*U, py=v=>oy-v.y*U;
  let a={x:2.2,y:0.6}, b={x:0.8,y:1.8};
  const proj=svg.append("line").attr("stroke",PC.muted).attr("stroke-width",1.5).attr("stroke-dasharray","4 3");
  const projv=svg.append("line").attr("stroke",PC.good).attr("stroke-width",3).attr("stroke-opacity",.8);
  const la=svg.append("line").attr("stroke",PC.accent).attr("stroke-width",2.5).attr("marker-end","url(#da)");
  const lb=svg.append("line").attr("stroke",PC.a2).attr("stroke-width",2.5).attr("marker-end","url(#db)");
  const out=document.getElementById("dot-readout");
  function handle(v,col){ const h=svg.append("circle").attr("class","dragpt").attr("r",10).attr("fill",col).attr("stroke","#0f1117").attr("stroke-width",2).style("cursor","grab");
    h.call(d3.drag().on("drag",e=>{ v.x=(e.x-ox)/U; v.y=(oy-e.y)/U; update(); })); return h; }
  function update(){
    la.attr("x1",ox).attr("y1",oy).attr("x2",px(a)).attr("y2",py(a));
    lb.attr("x1",ox).attr("y1",oy).attr("x2",px(b)).attr("y2",py(b));
    ha.attr("cx",px(a)).attr("cy",py(a)); hb.attr("cx",px(b)).attr("cy",py(b));
    const dot=a.x*b.x+a.y*b.y, na=Math.hypot(a.x,a.y), nb=Math.hypot(b.x,b.y);
    const cos=dot/((na*nb)||1e-9), ang=Math.acos(Math.max(-1,Math.min(1,cos)))*180/Math.PI;
    const t=dot/((na*na)||1e-9), foot={x:a.x*t,y:a.y*t};   // projection of b onto a
    projv.attr("x1",ox).attr("y1",oy).attr("x2",px(foot)).attr("y2",py(foot));
    proj.attr("x1",px(b)).attr("y1",py(b)).attr("x2",px(foot)).attr("y2",py(foot));
    const cc=dot>0.05?PC.good:dot<-0.05?PC.bad:PC.a2, tag=Math.abs(dot)<0.05?"orthogonal":dot>0?"acute":"obtuse";
    out.innerHTML=`a · b = <b style="color:${cc}">${dot.toFixed(2)}</b> &nbsp;(${tag}) &nbsp;·&nbsp; angle <b>${ang.toFixed(0)}°</b> &nbsp;·&nbsp; ‖a‖‖b‖cos θ = <b>${(na*nb*cos).toFixed(2)}</b>`;
  }
  const ha=handle(a,PC.accent), hb=handle(b,PC.a2); update();
})();
/* ───────────────────────── Ch4 · basis & coordinates explorer ───────────────────────── */
(function(){
  const svg=d3.select("#basis-svg"),W=640,H=400,ox=W/2,oy=H/2,U=44;
  axes(svg,W,H,ox,oy,U); arrowDefs(svg,[["kb1",PC.accent],["kb2",PC.a2]]);
  const px=v=>ox+v.x*U, py=v=>oy-v.y*U;
  let b1={x:2,y:0.5}, b2={x:-0.5,y:1.8}, p={x:2.5,y:2.2};
  const grid=svg.append("g").attr("opacity",.9);
  const spanline=svg.append("line").attr("stroke",PC.bad).attr("stroke-width",2).attr("stroke-dasharray","7 5").attr("opacity",0);
  const leg1=svg.append("line").attr("stroke",PC.accent).attr("stroke-width",7).attr("stroke-opacity",.30).attr("stroke-linecap","round");
  const leg2=svg.append("line").attr("stroke",PC.a2).attr("stroke-width",7).attr("stroke-opacity",.30).attr("stroke-linecap","round");
  const l1=svg.append("line").attr("stroke",PC.accent).attr("stroke-width",2.5).attr("marker-end","url(#kb1)");
  const l2=svg.append("line").attr("stroke",PC.a2).attr("stroke-width",2.5).attr("marker-end","url(#kb2)");
  const out=document.getElementById("basis-readout");
  function handle(v,col,r){
    const h=svg.append("circle").attr("class","dragpt").attr("r",r).attr("fill",col).attr("stroke","#0f1117").attr("stroke-width",2).style("cursor","grab");
    h.call(d3.drag().on("drag",e=>{ v.x=Math.round((e.x-ox)/U*10)/10; v.y=Math.round((oy-e.y)/U*10)/10; update(); }));
    return h;
  }
  const h1=handle(b1,PC.accent,10), h2=handle(b2,PC.a2,10), hp=handle(p,PC.good,8);
  function fmt(n){ return (Math.round(n*100)/100).toFixed(2); }
  function update(){
    const det=b1.x*b2.y-b1.y*b2.x, dep=Math.abs(det)<0.08;
    grid.selectAll("*").remove();
    if(!dep){
      const K=9;
      for(let k=-K;k<=K;k++){
        const A1={x:b1.x*k-b2.x*K, y:b1.y*k-b2.y*K}, B1={x:b1.x*k+b2.x*K, y:b1.y*k+b2.y*K};
        grid.append("line").attr("x1",px(A1)).attr("y1",py(A1)).attr("x2",px(B1)).attr("y2",py(B1))
            .attr("stroke",k===0?PC.a2:"#26314a").attr("stroke-opacity",k===0?.55:1);
        const A2={x:b2.x*k-b1.x*K, y:b2.y*k-b1.y*K}, B2={x:b2.x*k+b1.x*K, y:b2.y*k+b1.y*K};
        grid.append("line").attr("x1",px(A2)).attr("y1",py(A2)).attr("x2",px(B2)).attr("y2",py(B2))
            .attr("stroke",k===0?PC.accent:"#26314a").attr("stroke-opacity",k===0?.55:1);
      }
    }
    l1.attr("x1",ox).attr("y1",oy).attr("x2",px(b1)).attr("y2",py(b1));
    l2.attr("x1",ox).attr("y1",oy).attr("x2",px(b2)).attr("y2",py(b2));
    h1.attr("cx",px(b1)).attr("cy",py(b1)); h2.attr("cx",px(b2)).attr("cy",py(b2));
    hp.attr("cx",px(p)).attr("cy",py(p));
    if(dep){
      const d=(Math.hypot(b1.x,b1.y)>1e-6)?b1:b2, n=Math.hypot(d.x,d.y)||1;
      const e={x:d.x/n*9, y:d.y/n*9}, f={x:-e.x,y:-e.y};
      spanline.attr("opacity",.9).attr("x1",px(f)).attr("y1",py(f)).attr("x2",px(e)).attr("y2",py(e));
      leg1.attr("opacity",0); leg2.attr("opacity",0);
      const cross=Math.abs(p.x*d.y-p.y*d.x), onLine=cross<0.12*n;
      out.innerHTML='det = <b style="color:'+PC.bad+'">'+fmt(det)+' ≈ 0</b> &nbsp;·&nbsp; b₁ and b₂ are <b style="color:'+PC.bad+'">linearly dependent</b>: the span has collapsed from a plane to a line, so this is <b>not a basis</b>.'
        + '<br>' + (onLine
            ? 'p lies on that line, so it has <b>infinitely many</b> coordinate pairs — uniqueness is exactly what independence buys.'
            : 'p is off the line, so it is <b>unreachable</b>: no weights on b₁, b₂ can build it.');
      return;
    }
    spanline.attr("opacity",0);
    leg1.attr("opacity",1); leg2.attr("opacity",1);
    const c1=(p.x*b2.y-p.y*b2.x)/det, c2=(b1.x*p.y-b1.y*p.x)/det;
    const m={x:b1.x*c1, y:b1.y*c1};
    leg1.attr("x1",ox).attr("y1",oy).attr("x2",px(m)).attr("y2",py(m));
    leg2.attr("x1",px(m)).attr("y1",py(m)).attr("x2",px(p)).attr("y2",py(p));
    out.innerHTML='b₁=('+fmt(b1.x)+', '+fmt(b1.y)+') &nbsp; b₂=('+fmt(b2.x)+', '+fmt(b2.y)+') &nbsp; p=('+fmt(p.x)+', '+fmt(p.y)+')'
      + ' &nbsp;·&nbsp; det = <b>'+fmt(det)+'</b> (≠ 0 ⟹ independent ⟹ a basis)'
      + '<br>coordinates <b style="color:'+PC.good+'">p[B] = ('+fmt(c1)+', '+fmt(c2)+')</b>'
      + ' &nbsp; check: '+fmt(c1)+'·b₁ + '+fmt(c2)+'·b₂ = ('+fmt(c1*b1.x+c2*b2.x)+', '+fmt(c1*b1.y+c2*b2.y)+') ✓';
  }
  function set(a,b){ b1.x=a[0]; b1.y=a[1]; b2.x=b[0]; b2.y=b[1]; update(); }
  document.getElementById("bs-std").addEventListener("click",()=>set([1,0],[0,1]));
  document.getElementById("bs-skew").addEventListener("click",()=>set([2,0.5],[-0.5,1.8]));
  document.getElementById("bs-dep").addEventListener("click",()=>set([2,1],[-1,-0.5]));
  update();
})();
