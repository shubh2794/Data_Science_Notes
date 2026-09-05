/* k-nearest-neighbors.viz.js — extracted from k-nearest-neighbors.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

function dist(a,b){ return Math.hypot(a.x-b.x, a.y-b.y); }

/* Interactive kNN decision regions */
(function(){
  const svg=d3.select("#knn-svg"), W=640, H=360;
  let addClass=0;
  let data=[
    {x:140,y:120,c:0},{x:180,y:90,c:0},{x:210,y:160,c:0},{x:120,y:200,c:0},
    {x:470,y:230,c:1},{x:510,y:200,c:1},{x:440,y:280,c:1},{x:540,y:270,c:1}
  ];
  const bg=svg.append("g"), gPts=svg.append("g");
  const step=16;
  function classify(px,py,k){
    const ds=data.map(d=>({d:Math.hypot(d.x-px,d.y-py),c:d.c})).sort((a,b)=>a.d-b.d).slice(0,Math.min(k,data.length));
    let s=0; ds.forEach(o=>s+=o.c?1:-1); return s>0?1:0;
  }
  function draw(){
    const k=+d3.select("#knn-k").property("value");
    d3.select("#knn-kval").text(k);
    const cells=[];
    for(let gx=0;gx<W;gx+=step) for(let gy=0;gy<H;gy+=step) cells.push({gx,gy,c:classify(gx+step/2,gy+step/2,k)});
    const r=bg.selectAll("rect").data(cells);
    r.enter().append("rect").attr("width",step).attr("height",step)
      .merge(r).attr("x",d=>d.gx).attr("y",d=>d.gy).attr("fill",d=>d.c?C.B:C.A).attr("opacity",0.16);
    r.exit().remove();
    const p=gPts.selectAll("circle").data(data);
    p.enter().append("circle").attr("r",7).attr("stroke","#0f1117").attr("stroke-width",2)
      .merge(p).attr("cx",d=>d.x).attr("cy",d=>d.y).attr("fill",d=>d.c?C.B:C.A);
    p.exit().remove();
  }
  svg.on("click",function(ev){
    const [mx,my]=d3.pointer(ev);
    data.push({x:mx,y:my,c:addClass}); draw();
  });
  d3.select("#knn-k").on("input",draw);
  d3.select("#knn-class").on("click",function(){ addClass=addClass?0:1; d3.select("#knn-classname").text(addClass?"Class B":"Class A"); });
  d3.select("#knn-reset").on("click",()=>{ data=[{x:140,y:120,c:0},{x:180,y:90,c:0},{x:210,y:160,c:0},{x:120,y:200,c:0},{x:470,y:230,c:1},{x:510,y:200,c:1},{x:440,y:280,c:1},{x:540,y:270,c:1}]; draw(); });
  draw();
})();
