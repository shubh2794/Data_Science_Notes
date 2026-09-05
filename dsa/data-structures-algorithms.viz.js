/* data-structures-algorithms.viz.js — extracted from data-structures-algorithms.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const svg=d3.select("#map-svg"), W=660;
  const tiles=[
    {id:"bigo",  label:"Big-O Complexity",   x:30,  y:30, role:"The yardstick: how cost grows with n.",        cx:"common classes: O(1), O(log n), O(n), O(n log n), O(n²)"},
    {id:"arr",   label:"Arrays & Strings",   x:240, y:30, role:"Contiguous, index-addressable storage.",       cx:"index access O(1) · middle insert/delete O(n)"},
    {id:"tree",  label:"Trees & Graphs",     x:450, y:30, role:"Hierarchy and pairwise relationships.",        cx:"BST search O(log n) balanced · BFS/DFS O(V+E)"},
    {id:"sort",  label:"Sorting & Searching",x:135, y:170,role:"Order data, then locate it fast.",             cx:"comparison sort Ω(n log n) · binary search O(log n)"},
    {id:"dp",    label:"Dynamic Programming",x:360, y:170,role:"Cache overlapping subproblems; solve once.",    cx:"e.g. Fibonacci O(2ⁿ) naive → O(n) with DP"}
  ];
  const tw=180, th=80;
  const g=svg.selectAll("g.tile").data(tiles).enter().append("g")
    .attr("class","tile").attr("cursor","pointer")
    .attr("transform",d=>`translate(${d.x},${d.y})`);
  g.append("rect").attr("width",tw).attr("height",th).attr("rx",12)
    .attr("fill","#1a2030").attr("stroke",C.line).attr("stroke-width",1.5);
  g.append("text").attr("x",tw/2).attr("y",th/2+1).attr("text-anchor","middle")
    .attr("fill",C.ink).attr("font-size",14).attr("font-weight",600).text(d=>d.label);
  g.on("click",function(ev,d){
    svg.selectAll("g.tile rect").attr("stroke",C.line).attr("stroke-width",1.5).attr("fill","#1a2030");
    d3.select(this).select("rect").attr("stroke",C.A).attr("stroke-width",2.5).attr("fill","#1d2740");
    d3.select("#map-readout").html(`<b>${d.label}</b> — ${d.role}<br>${d.cx}`);
  });
  d3.select("#map-reset").on("click",()=>{
    svg.selectAll("g.tile rect").attr("stroke",C.line).attr("stroke-width",1.5).attr("fill","#1a2030");
    d3.select("#map-readout").text("Click a tile to explore one of the five pillars");
  });
})();
