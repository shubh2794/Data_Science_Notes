/* trees-graphs.viz.js — extracted from trees-graphs.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const svg=d3.select("#bfs-svg");
  const nodes=[
    {id:0,x:90, y:170},{id:1,x:210,y:80}, {id:2,x:210,y:260},
    {id:3,x:340,y:50}, {id:4,x:340,y:170},{id:5,x:340,y:290},
    {id:6,x:470,y:110},{id:7,x:470,y:230},{id:8,x:590,y:110},{id:9,x:590,y:230}
  ];
  const edges=[[0,1],[0,2],[1,3],[1,4],[2,4],[2,5],[3,6],[4,6],[4,7],[5,7],[6,8],[7,9],[8,9]];
  const adj=nodes.map(()=>[]);
  edges.forEach(([a,b])=>{ adj[a].push(b); adj[b].push(a); });
  adj.forEach(l=>l.sort((p,q)=>p-q));

  let mode="BFS", front=[], visited=new Set(), order=[], current=null, started=false;

  const gE=svg.append("g"), gN=svg.append("g");
  gE.selectAll("line").data(edges).enter().append("line")
    .attr("x1",d=>nodes[d[0]].x).attr("y1",d=>nodes[d[0]].y)
    .attr("x2",d=>nodes[d[1]].x).attr("y2",d=>nodes[d[1]].y)
    .attr("stroke",C.line).attr("stroke-width",1.6);
  const gn=gN.selectAll("g").data(nodes).enter().append("g")
    .attr("transform",d=>`translate(${d.x},${d.y})`);
  gn.append("circle").attr("r",17).attr("fill","#1a2030").attr("stroke",C.line).attr("stroke-width",2);
  gn.append("text").attr("text-anchor","middle").attr("dy",5).attr("fill",C.ink).attr("font-size",14).attr("font-weight",600).text(d=>d.id);

  function paint(){
    gn.select("circle")
      .attr("fill",d=> d.id===current ? "#3a2e16" : visited.has(d.id) ? "#173a26" : front.includes(d.id) ? "#15233a" : "#1a2030")
      .attr("stroke",d=> d.id===current ? C.B : visited.has(d.id) ? C.good : front.includes(d.id) ? C.A : C.line)
      .attr("stroke-width",d=> (d.id===current||visited.has(d.id)||front.includes(d.id))?2.8:2);
    const struct = mode==="BFS" ? "queue" : "stack";
    d3.select("#bfs-readout").html(
      `${struct}: [${front.join(", ")||"·"}] &nbsp;·&nbsp; visited order: ${order.join(" → ")||"·"}`);
  }
  function reset(){
    front=[0]; visited=new Set(); order=[]; current=null; started=true;
    paint();
    d3.select("#bfs-readout").html(`${mode==="BFS"?"queue":"stack"} seeded with source 0 — press Step`);
  }
  function step(){
    if(!started) reset();
    if(front.length===0){ current=null; paint(); d3.select("#bfs-readout").html(`done — visited ${order.length} nodes in order: ${order.join(" → ")}`); return; }
    // BFS pops front (queue), DFS pops back (stack)
    const node = mode==="BFS" ? front.shift() : front.pop();
    if(visited.has(node)){ paint(); return step(); }
    visited.add(node); order.push(node); current=node;
    const neigh = adj[node].filter(x=>!visited.has(x) && !front.includes(x));
    if(mode==="BFS"){ neigh.forEach(x=>front.push(x)); }
    else { neigh.slice().reverse().forEach(x=>front.push(x)); }
    paint();
  }
  d3.select("#bfs-mode").on("click",function(){
    mode = mode==="BFS" ? "DFS" : "BFS";
    d3.select(this).text("Mode: "+mode);
    started=false; front=[]; visited=new Set(); order=[]; current=null; paint();
    d3.select("#bfs-readout").html(`switched to ${mode} — press Step to begin from source 0`);
  });
  d3.select("#bfs-step").on("click",step);
  d3.select("#bfs-reset").on("click",reset);
  reset(); started=false; front=[]; visited=new Set(); paint();
  d3.select("#bfs-readout").html(`BFS uses a queue, DFS a stack — press Step to begin from source 0`);
})();
