/* decision-trees.viz.js — extracted from decision-trees.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

function dist(a,b){ return Math.hypot(a.x-b.x, a.y-b.y); }

/* Interactive recursive tree */
(function(){
  const svg=d3.select("#tree-svg"), W=640, H=320;
  const g=svg.append("g");
  const feats=["x₁>0.5","x₂>0.3","x₁>0.8","x₂>0.7","x₁>0.2","x₂>0.6","x₁>0.4"];
  function build(depth){
    g.selectAll("*").remove();
    const root={};
    let id=0;
    function make(d){ const n={id:id++}; if(d>0){ n.children=[make(d-1),make(d-1)]; } return n; }
    const tree=make(depth);
    const hierarchy=d3.hierarchy(tree);
    const layout=d3.tree().size([W-60,H-70]);
    layout(hierarchy);
    g.attr("transform","translate(30,24)");
    g.selectAll("line.link").data(hierarchy.links()).enter().append("line").attr("class","link")
      .attr("x1",d=>d.source.x).attr("y1",d=>d.source.y).attr("x2",d=>d.target.x).attr("y2",d=>d.target.y)
      .attr("stroke",C.line).attr("stroke-width",1.5);
    const node=g.selectAll("g.node").data(hierarchy.descendants()).enter().append("g").attr("transform",d=>`translate(${d.x},${d.y})`);
    node.append("circle").attr("r",d=>d.children?16:14)
      .attr("fill",d=>d.children?"#1e222d": (d.data.id%2?C.A:C.B)).attr("stroke",d=>d.children?C.A:"#0f1117").attr("stroke-width",2);
    node.append("text").attr("text-anchor","middle").attr("dy",4).attr("font-size",10).attr("fill",C.ink)
      .text(d=>d.children?feats[d.data.id%feats.length]:(d.data.id%2?"A":"B"));
    const purity=Math.min(99, 60+depth*10);
    g.append("text").attr("x",0).attr("y",H-60).attr("fill",C.muted).attr("font-size",11).text(`Leaf nodes: ${Math.pow(2,depth)} · est. leaf purity ≈ ${purity}%`);
  }
  d3.select("#tree-depth").on("input",function(){ d3.select("#tree-dval").text(this.value); build(+this.value); });
  build(2);
})();
