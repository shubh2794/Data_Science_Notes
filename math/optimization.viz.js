/* optimization.viz.js — extracted from optimization.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

const PC={accent:"#5b9cff",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};

function arrowDefs(svg,specs){ const defs=svg.append("defs");
  specs.forEach(([id,c])=>defs.append("marker").attr("id",id).attr("viewBox","0 0 10 10").attr("refX",8).attr("refY",5)
    .attr("markerWidth",7).attr("markerHeight",7).attr("orient","auto-start-reverse").append("path").attr("d","M0,0L10,5L0,10").attr("fill",c)); }

/* ───────────────────────── §02 · 1-D loss landscape ───────────────────────── */
(function(){
  const svg=d3.select("#land-svg"),W=640,H=360,m={l:40,r:20,t:24,b:36};
  arrowDefs(svg,[["lg",PC.bad]]);
  // a non-convex curve with a local min, a saddle-ish bump, and a global min
  const f =x=> 0.06*x*x*x*x - 0.18*x*x*x - 1.2*x*x + 0.6*x + 6;
  const df=x=> 0.24*x*x*x - 0.54*x*x - 2.4*x + 0.6;
  const xs=d3.scaleLinear().domain([-4,5]).range([m.l,W-m.r]);
  let lo=Infinity,hi=-Infinity; for(let x=-4;x<=5;x+=0.05){const y=f(x);lo=Math.min(lo,y);hi=Math.max(hi,y);}
  const ys=d3.scaleLinear().domain([lo-1,hi+1]).range([H-m.b,m.t]);
  svg.append("line").attr("x1",m.l).attr("y1",H-m.b).attr("x2",W-m.r).attr("y2",H-m.b).attr("stroke",PC.line);
  svg.append("text").attr("x",W-m.r).attr("y",H-m.b+24).attr("text-anchor","end").attr("fill",PC.muted).attr("font-size",11).text("θ");
  svg.append("text").attr("x",m.l-6).attr("y",m.t+4).attr("text-anchor","end").attr("fill",PC.muted).attr("font-size",11).text("J(θ)");
  const pts=[]; for(let x=-4;x<=5;x+=0.04) pts.push([xs(x),ys(f(x))]);
  svg.append("path").attr("d",d3.line()(pts)).attr("fill","none").attr("stroke",PC.accent).attr("stroke-width",2.5);
  const tang=svg.append("line").attr("stroke",PC.a2).attr("stroke-width",2);
  const arr=svg.append("line").attr("stroke",PC.bad).attr("stroke-width",2.5).attr("marker-end","url(#lg)");
  const dot=svg.append("circle").attr("r",8).attr("fill",PC.good).attr("stroke","#0f1117").attr("stroke-width",2).style("cursor","grab");
  const out=document.getElementById("land-readout");
  let cx=-2.2;
  function update(){
    const y=f(cx),s=df(cx);
    dot.attr("cx",xs(cx)).attr("cy",ys(y));
    const dx=0.9, x1=cx-dx,x2=cx+dx;            // tangent segment
    tang.attr("x1",xs(x1)).attr("y1",ys(y-s*dx)).attr("x2",xs(x2)).attr("y2",ys(y+s*dx));
    const dir=-Math.sign(s)||0;                  // descent direction along θ
    arr.attr("x1",xs(cx)).attr("y1",ys(y)-0).attr("x2",xs(cx+dir*0.9)).attr("y2",ys(y));
    const cls=Math.abs(s)<0.06?"critical (slope ≈ 0)":s>0?"slope > 0 → step left":"slope < 0 → step right";
    out.innerHTML=`θ = <b>${cx.toFixed(2)}</b> &nbsp;·&nbsp; J = <b>${y.toFixed(2)}</b> &nbsp;·&nbsp; J′ = <b style="color:${PC.a2}">${s.toFixed(2)}</b> &nbsp;·&nbsp; <b style="color:${PC.bad}">${cls}</b>`;
  }
  dot.call(d3.drag().on("drag",e=>{ cx=Math.max(-4,Math.min(5,xs.invert(e.x))); update(); }));
  update();
})();

/* shared elliptical-bowl helpers for §04, §06, §07 */
function bowlViz(svgId,readoutId,opts){
  const svg=d3.select(svgId),W=640,H=360,ox=W/2,oy=H/2,U=opts.U||34;
  const out=document.getElementById(readoutId);
  const px=p=>ox+p.x*U, py=p=>oy-p.y*U;
  // J = 0.5(a x² + b y²); gradient = (a x, b y)
  function drawContours(a,b,g){
    g.selectAll("*").remove();
    for(let k=1;k<=6;k++){
      const lvl=k*k*0.55;                 // level value
      const rx=Math.sqrt(2*lvl/a), ry=Math.sqrt(2*lvl/b);
      g.append("ellipse").attr("cx",ox).attr("cy",oy).attr("rx",rx*U).attr("ry",ry*U)
        .attr("fill","none").attr("stroke","#243044").attr("stroke-width",1);
    }
    g.append("line").attr("x1",10).attr("y1",oy).attr("x2",W-10).attr("y2",oy).attr("stroke","#2a3344");
    g.append("line").attr("x1",ox).attr("y1",10).attr("x2",ox).attr("y2",H-10).attr("stroke","#2a3344");
    g.append("circle").attr("cx",ox).attr("cy",oy).attr("r",4).attr("fill",PC.good);
  }
  return {svg,W,H,ox,oy,U,px,py,out,drawContours};
}

/* ───────────────────────── §04 · gradient descent on a bowl ───────────────────────── */
(function(){
  const a=1.0,b=2.6;                          // mildly stretched bowl
  const v=bowlViz("#gd-svg","gd-readout",{});
  const cg=v.svg.append("g"); v.drawContours(a,b,cg);
  v.svg.append("text").attr("x",v.ox).attr("y",v.oy-8).attr("text-anchor","middle").attr("fill",PC.good).attr("font-size",11).text("min");
  const pathG=v.svg.append("g");
  let start={x:-3.6,y:1.7}, cur={x:start.x,y:start.y}, trail=[];
  const startDot=v.svg.append("circle").attr("r",8).attr("fill",PC.a2).attr("stroke","#0f1117").attr("stroke-width",2).style("cursor","grab");
  const eta=document.getElementById("gd-eta");
  function J(p){return 0.5*(a*p.x*p.x+b*p.y*p.y);}
  function redraw(){
    startDot.attr("cx",v.px(start)).attr("cy",v.py(start));
    pathG.selectAll("*").remove();
    if(trail.length>1) pathG.append("path").attr("d",d3.line().x(p=>v.px(p)).y(p=>v.py(p))(trail))
      .attr("fill","none").attr("stroke",PC.accent).attr("stroke-width",2).attr("stroke-opacity",.85);
    trail.forEach((p,i)=>pathG.append("circle").attr("cx",v.px(p)).attr("cy",v.py(p)).attr("r",i===trail.length-1?5:2.5)
      .attr("fill",i===trail.length-1?PC.bad:PC.accent));
    const e=+eta.value, diverged=Math.hypot(cur.x,cur.y)>30;
    v.out.innerHTML=`η = <b>${e.toFixed(2)}</b> &nbsp;·&nbsp; steps <b>${Math.max(0,trail.length-1)}</b> &nbsp;·&nbsp; θ = (${cur.x.toFixed(2)}, ${cur.y.toFixed(2)}) &nbsp;·&nbsp; J = <b style="color:${diverged?PC.bad:PC.good}">${diverged?"diverging ✗":J(cur).toFixed(3)}</b> &nbsp;<span style="color:var(--muted)">(stable η &lt; ${(2/b).toFixed(2)})</span>`;
  }
  function reset(){cur={x:start.x,y:start.y};trail=[{...cur}];redraw();}
  function step(){const e=+eta.value; cur={x:cur.x-e*a*cur.x, y:cur.y-e*b*cur.y}; trail.push({...cur}); if(trail.length>200)trail.shift(); redraw();}
  startDot.call(d3.drag().on("drag",e=>{ start.x=(e.x-v.ox)/v.U; start.y=(v.oy-e.y)/v.U; reset(); }));
  document.getElementById("gd-step").onclick=step;
  document.getElementById("gd-run").onclick=()=>{for(let i=0;i<40;i++)step();};
  document.getElementById("gd-reset").onclick=reset;
  reset();
})();

/* ───────────────────────── §06 · plain vs momentum ───────────────────────── */
(function(){
  const a=0.35,b=4.2;                          // strongly stretched valley
  const v=bowlViz("#mom-svg","mom-readout",{U:30});
  const cg=v.svg.append("g"); v.drawContours(a,b,cg);
  const gPlain=v.svg.append("g"), gMom=v.svg.append("g");
  const eta=document.getElementById("mom-eta"), beta=document.getElementById("mom-beta");
  const start={x:-7.5,y:1.4};
  function J(p){return 0.5*(a*p.x*p.x+b*p.y*p.y);}
  function runPath(useMom){
    const e=+eta.value, B=+beta.value;
    let p={...start}, vel={x:0,y:0}, t=[{...p}];
    for(let i=0;i<80;i++){
      const g={x:a*p.x,y:b*p.y};
      if(useMom){vel={x:B*vel.x+g.x,y:B*vel.y+g.y}; p={x:p.x-e*vel.x,y:p.y-e*vel.y};}
      else{p={x:p.x-e*g.x,y:p.y-e*g.y};}
      t.push({...p}); if(Math.hypot(p.x,p.y)<0.04)break;
    }
    return t;
  }
  function draw(g,t,col){
    g.selectAll("*").remove();
    g.append("path").attr("d",d3.line().x(p=>v.px(p)).y(p=>v.py(p))(t)).attr("fill","none").attr("stroke",col).attr("stroke-width",2).attr("stroke-opacity",.9);
    t.forEach((p,i)=>{ if(i%4===0||i===t.length-1) g.append("circle").attr("cx",v.px(p)).attr("cy",v.py(p)).attr("r",2.5).attr("fill",col);});
  }
  function run(){
    const tp=runPath(false), tm=runPath(true);
    draw(gPlain,tp,PC.bad); draw(gMom,tm,PC.good);
    v.out.innerHTML=`<b style="color:${PC.bad}">plain GD</b>: ${tp.length-1} steps to center &nbsp;·&nbsp; <b style="color:${PC.good}">momentum (β=${(+beta.value).toFixed(2)})</b>: ${tm.length-1} steps &nbsp;<span style="color:var(--muted)">— momentum cancels the cross-valley zig-zag</span>`;
  }
  function reset(){gPlain.selectAll("*").remove();gMom.selectAll("*").remove();
    v.svg.selectAll(".startc").remove();
    v.svg.append("circle").attr("class","startc").attr("cx",v.px(start)).attr("cy",v.py(start)).attr("r",6).attr("fill",PC.a2).attr("stroke","#0f1117").attr("stroke-width",2);
    v.out.innerHTML=`start = (${start.x}, ${start.y}) &nbsp;·&nbsp; press <b>Run both</b> to compare`;}
  document.getElementById("mom-run").onclick=()=>{reset();run();};
  document.getElementById("mom-reset").onclick=reset;
  reset();
})();

/* ───────────────────────── §07 · condition number ───────────────────────── */
(function(){
  const v=bowlViz("#cond-svg","cond-readout",{U:28});
  const cg=v.svg.append("g"), pathG=v.svg.append("g");
  const kS=document.getElementById("cond-k");
  const start={x:-7,y:6};
  function ab(){const k=+kS.value; return {a:1, b:k};}   // λmin=1, λmax=k → κ=k
  function redraw(){
    const {a,b}=ab(); v.drawContours(a,b,cg);
    pathG.selectAll("*").remove();
    v.svg.selectAll(".startc2").remove();
    v.svg.append("circle").attr("class","startc2").attr("cx",v.px(start)).attr("cy",v.py(start)).attr("r",6).attr("fill",PC.a2).attr("stroke","#0f1117").attr("stroke-width",2);
    v.out.innerHTML=`κ = λ_max/λ_min = <b>${(+kS.value).toFixed(1)}</b> &nbsp;·&nbsp; optimal η = 2/(λ_min+λ_max) = <b>${(2/(1+ +kS.value)).toFixed(3)}</b> &nbsp;·&nbsp; GD factor (κ−1)/(κ+1) = <b style="color:${PC.bad}">${(((+kS.value)-1)/((+kS.value)+1)).toFixed(3)}</b>`;
  }
  function run(){
    const {a,b}=ab(); const e=2/(a+b);            // optimal fixed rate
    let p={...start}, t=[{...p}];
    for(let i=0;i<120;i++){ p={x:p.x-e*a*p.x, y:p.y-e*b*p.y}; t.push({...p}); if(Math.hypot(p.x,p.y)<0.05)break; }
    pathG.selectAll("*").remove();
    pathG.append("path").attr("d",d3.line().x(p=>v.px(p)).y(p=>v.py(p))(t)).attr("fill","none").attr("stroke",PC.accent).attr("stroke-width",2);
    t.forEach((p,i)=>pathG.append("circle").attr("cx",v.px(p)).attr("cy",v.py(p)).attr("r",i===t.length-1?5:2).attr("fill",i===t.length-1?PC.good:PC.accent));
    v.out.innerHTML=`κ = <b>${(+kS.value).toFixed(1)}</b> &nbsp;·&nbsp; converged in <b>${t.length-1}</b> steps &nbsp;<span style="color:var(--muted)">(round bowl → straight shot; stretched bowl → zig-zag)</span>`;
  }
  kS.oninput=redraw;
  document.getElementById("cond-run").onclick=run;
  document.getElementById("cond-reset").onclick=redraw;
  redraw();
})();
