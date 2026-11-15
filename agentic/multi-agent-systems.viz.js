/* multi-agent-systems.viz.js — interactive topology diagram.
   Loaded after data.js / notes.js; uses d3 from ../vendor/d3.min.js. */

(function(){
  const C = {
    accent: "#fb923c", a2: "#ffb454", good: "#4ade80", bad: "#f87171",
    ink: "#e6e9ef", muted: "#9aa3b2", line: "#2a2f3a",
    coord: "#818cf8", worker: "#fb923c", deleg: "#2dd4bf", msg: "#fcd34d"
  };
  const svg = d3.select("#topo-svg"), W = 640, H = 300;

  /* ── marker ─────────────────────────────────────── */
  function ensureMarker(){
    const d = svg.append("defs");
    d.append("marker").attr("id","ma").attr("viewBox","0 -5 10 10")
      .attr("refX",9).attr("refY",0).attr("markerWidth",7).attr("markerHeight",7)
      .attr("orient","auto")
      .append("path").attr("d","M0,-4L8,0L0,4").attr("fill",C.muted);
    d.append("marker").attr("id","ma-msg").attr("viewBox","0 -5 10 10")
      .attr("refX",9).attr("refY",0).attr("markerWidth",7).attr("markerHeight",7)
      .attr("orient","auto")
      .append("path").attr("d","M0,-4L8,0L0,4").attr("fill",C.msg);
  }

  /* ── drawing helpers ────────────────────────────── */
  function agent(x, y, label, role){
    const r = 24;
    const fill = role === "coord" ? "rgba(129,140,248,.2)" :
                 role === "deleg" ? "rgba(45,212,191,.2)" :
                 role === "worker" ? "rgba(251,146,60,.2)" :
                 "rgba(129,140,248,.1)";
    const stroke = role === "coord" ? C.coord :
                   role === "deleg" ? C.deleg :
                   role === "worker" ? C.worker : C.muted;
    svg.append("circle").attr("cx",x).attr("cy",y).attr("r",r)
      .attr("fill",fill).attr("stroke",stroke).attr("stroke-width",1.5);
    svg.append("text").attr("x",x).attr("y",y+4).attr("text-anchor","middle")
      .attr("font-size",10).attr("fill",C.ink).text(label);
    return {x, y, r};
  }

  function edge(a, b, dashed){
    const dx = b.x - a.x, dy = b.y - a.y;
    const dist = Math.sqrt(dx*dx + dy*dy);
    const ux = dx/dist, uy = dy/dist;
    const x1 = a.x + ux * a.r, y1 = a.y + uy * a.r;
    const x2 = b.x - ux * b.r, y2 = b.y - uy * b.r;
    const l = svg.append("line")
      .attr("x1",x1).attr("y1",y1).attr("x2",x2).attr("y2",y2)
      .attr("stroke",C.muted).attr("stroke-width",1.2)
      .attr("marker-end","url(#ma)");
    if(dashed) l.attr("stroke-dasharray","4,3");
    return {x1,y1,x2,y2};
  }

  function biEdge(a, b){
    // offset two lines slightly
    const dx = b.x - a.x, dy = b.y - a.y;
    const dist = Math.sqrt(dx*dx + dy*dy);
    const ux = dx/dist, uy = dy/dist;
    const px = -uy * 3, py = ux * 3; // perpendicular offset
    const x1a = a.x + ux*a.r + px, y1a = a.y + uy*a.r + py;
    const x2a = b.x - ux*b.r + px, y2a = b.y - uy*b.r + py;
    const x1b = b.x - ux*b.r - px, y1b = b.y - uy*b.r - py;
    const x2b = a.x + ux*a.r - px, y2b = a.y + uy*a.r - py;
    svg.append("line").attr("x1",x1a).attr("y1",y1a).attr("x2",x2a).attr("y2",y2a)
      .attr("stroke",C.muted).attr("stroke-width",1).attr("marker-end","url(#ma)");
    svg.append("line").attr("x1",x1b).attr("y1",y1b).attr("x2",x2b).attr("y2",y2b)
      .attr("stroke",C.muted).attr("stroke-width",1).attr("marker-end","url(#ma)");
  }

  /* ── topology layouts ───────────────────────────── */
  const topos = {
    flat(){
      const cx = W/2, cy = H/2, R = 100, n = 5;
      const agents = [];
      for(let i = 0; i < n; i++){
        const a = 2 * Math.PI * i / n - Math.PI/2;
        const x = cx + R * Math.cos(a), y = cy + R * Math.sin(a);
        agents.push(agent(x, y, "Agent "+(i+1), "peer"));
      }
      // all-to-all edges
      for(let i = 0; i < n; i++)
        for(let j = i+1; j < n; j++)
          biEdge(agents[i], agents[j]);
      // label
      svg.append("text").attr("x",cx).attr("y",28).attr("text-anchor","middle")
        .attr("font-size",11).attr("fill",C.muted).text("All peers — vote or discuss as equals");
      return "Democratic / flat: all agents communicate with all others. Good for debate and voting, but O(n\u00B2) messages per round.";
    },

    hub(){
      const mgr = agent(W/2, 80, "Manager", "coord");
      const workers = [];
      const xs = [120, 260, 400, 540];
      xs.forEach((x, i) => {
        const w = agent(x, 220, "Worker "+(i+1), "worker");
        workers.push(w);
        biEdge(mgr, w);
      });
      svg.append("text").attr("x",W/2).attr("y",28).attr("text-anchor","middle")
        .attr("font-size",11).attr("fill",C.muted).text("Central manager routes tasks and integrates results");
      return "Manager / hub-spoke: one manager distributes tasks and collects results. Simple and debuggable, but the manager is a bottleneck and single point of failure.";
    },

    hier(){
      const top = agent(W/2, 50, "Lead", "coord");
      const mid1 = agent(200, 150, "Sub-mgr A", "deleg");
      const mid2 = agent(440, 150, "Sub-mgr B", "deleg");
      edge(top, mid1); edge(top, mid2);
      const w1 = agent(110, 250, "W1", "worker");
      const w2 = agent(200, 250, "W2", "worker");
      const w3 = agent(290, 250, "W3", "worker");
      edge(mid1, w1); edge(mid1, w2); edge(mid1, w3);
      const w4 = agent(380, 250, "W4", "worker");
      const w5 = agent(470, 250, "W5", "worker");
      const w6 = agent(540, 250, "W6", "worker");
      edge(mid2, w4); edge(mid2, w5); edge(mid2, w6);
      svg.append("text").attr("x",W/2).attr("y",22).attr("text-anchor","middle")
        .attr("font-size",11).attr("fill",C.muted).text("Nested managers — each level owns a sub-problem");
      return "Hierarchical / nested: the lead delegates to sub-managers, each managing their own workers. Scales to complex tasks but adds latency per level.";
    },

    pipe(){
      const stages = ["Planner","Researcher","Writer","Reviewer","Publisher"];
      const gap = (W - 80) / (stages.length - 1);
      const agents = stages.map((s, i) => {
        const role = i === 0 ? "coord" : i === stages.length-1 ? "coord" : "worker";
        return agent(50 + i * gap, H/2, s, role);
      });
      for(let i = 0; i < agents.length - 1; i++){
        edge(agents[i], agents[i+1]);
      }
      // feedback arrow from reviewer back to writer
      const rev = agents[3], wri = agents[2];
      const arc = svg.append("path")
        .attr("d", `M${rev.x - rev.r},${rev.y - 10} C${(rev.x+wri.x)/2},${rev.y - 70} ${(rev.x+wri.x)/2},${rev.y - 70} ${wri.x + wri.r},${wri.y - 10}`)
        .attr("fill","none").attr("stroke",C.bad).attr("stroke-width",1.2)
        .attr("stroke-dasharray","4,3").attr("marker-end","url(#ma)");
      svg.append("text").attr("x",(rev.x+wri.x)/2).attr("y",rev.y - 62)
        .attr("text-anchor","middle").attr("font-size",9).attr("fill",C.bad).text("revise");
      svg.append("text").attr("x",W/2).attr("y",28).attr("text-anchor","middle")
        .attr("font-size",11).attr("fill",C.muted).text("Sequential handoff — each stage refines the artifact");
      return "Pipeline / assembly-line: work flows left to right through specialized stages, with optional feedback loops. Latency is the sum of all stages.";
    }
  };

  /* ── message animation ──────────────────────────── */
  function animateMessages(){
    const lines = svg.selectAll("line").nodes();
    if(!lines.length) return;
    lines.forEach((el, i) => {
      const l = d3.select(el);
      const x1 = +l.attr("x1"), y1 = +l.attr("y1");
      const x2 = +l.attr("x2"), y2 = +l.attr("y2");
      const delay = i * 180;
      const dot = svg.append("circle").attr("r",4).attr("fill",C.msg).attr("opacity",0.9)
        .attr("cx",x1).attr("cy",y1);
      dot.transition().delay(delay).duration(700).ease(d3.easeCubicInOut)
        .attr("cx",x2).attr("cy",y2)
        .transition().duration(200).attr("opacity",0).remove();
    });
  }

  /* ── draw / redraw ──────────────────────────────── */
  function draw(){
    svg.selectAll("*").remove();
    ensureMarker();
    const k = d3.select("#topo-sel").property("value");
    const note = topos[k]();
    d3.select("#topo-read").html(note);
  }

  d3.select("#topo-sel").on("change", draw);
  d3.select("#topo-play").on("click", animateMessages);
  draw();
})();
