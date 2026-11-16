/* agent-evaluation.viz.js — interactive radar/spider chart comparing
   Agent v1 and Agent v2 across five evaluation axes.
   Loaded after data.js / notes.js; uses d3 from ../vendor/d3.min.js. */

(function(){
  const C = {
    accent: "#a3e635", good: "#4ade80", bad: "#f87171",
    a2: "#ffb454", blue: "#60a5fa", purple: "#c084fc",
    muted: "#9aa3b2", ink: "#e6e9ef", line: "#2a2f3a", panel: "#15181f"
  };

  const svg = d3.select("#eval-radar-svg");
  if(svg.empty()) return;
  const W = 520, H = 400;
  const cx = W / 2, cy = H / 2 + 10;
  const R = 150; // max radius for the chart

  const axes = [
    { label: "Tool Accuracy",   key: "tool"    },
    { label: "Plan Quality",    key: "plan"    },
    { label: "Memory Recall",   key: "memory"  },
    { label: "Task Completion",  key: "task"    },
    { label: "Cost Efficiency",  key: "cost"    }
  ];
  const N = axes.length;

  /* scores 0-100 */
  var v1 = { tool: 62, plan: 55, memory: 70, task: 58, cost: 45 };
  var v2 = { tool: 85, plan: 78, memory: 82, task: 90, cost: 72 };

  var showV1 = true, showV2 = true;

  /* angle for each axis (starting from top, going clockwise) */
  function angle(i){ return (2 * Math.PI * i / N) - Math.PI / 2; }

  /* point on the radar at given axis index and value 0-100 */
  function pt(i, val){
    var a = angle(i);
    var r = (val / 100) * R;
    return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
  }

  /* build the polygon path string for a data object */
  function polygonPath(data){
    var points = axes.map(function(ax, i){ return pt(i, data[ax.key]); });
    return points.map(function(p, i){ return (i === 0 ? "M" : "L") + p.x + "," + p.y; }).join("") + "Z";
  }

  function draw(){
    svg.selectAll("*").remove();

    /* ── grid rings ── */
    [20, 40, 60, 80, 100].forEach(function(val){
      var r = (val / 100) * R;
      var ringPts = [];
      for(var i = 0; i < N; i++){
        var a = angle(i);
        ringPts.push({ x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) });
      }
      var ringPath = ringPts.map(function(p, i){ return (i === 0 ? "M" : "L") + p.x + "," + p.y; }).join("") + "Z";
      svg.append("path").attr("d", ringPath)
        .attr("fill", "none").attr("stroke", C.line).attr("stroke-width", 1)
        .attr("stroke-opacity", val === 100 ? 0.6 : 0.35);
      /* ring label at the top axis */
      var labelPt = pt(0, val);
      svg.append("text").attr("x", labelPt.x + 6).attr("y", labelPt.y + 3)
        .attr("font-size", 9).attr("fill", C.muted).attr("fill-opacity", 0.7)
        .text(val);
    });

    /* ── axis spokes and labels ── */
    axes.forEach(function(ax, i){
      var tip = pt(i, 100);
      svg.append("line")
        .attr("x1", cx).attr("y1", cy).attr("x2", tip.x).attr("y2", tip.y)
        .attr("stroke", C.line).attr("stroke-width", 1).attr("stroke-opacity", 0.5);
      /* label */
      var labelR = R + 22;
      var a = angle(i);
      var lx = cx + labelR * Math.cos(a);
      var ly = cy + labelR * Math.sin(a);
      var anchor = "middle";
      if(Math.abs(a) < 0.1 || Math.abs(a - Math.PI) < 0.1) anchor = "middle";
      else if(a > 0 && a < Math.PI) anchor = "start";
      else if(a > -Math.PI && a < 0) anchor = "start";
      /* refine */
      if(Math.cos(a) < -0.3) anchor = "end";
      else if(Math.cos(a) > 0.3) anchor = "start";
      else anchor = "middle";
      svg.append("text").attr("x", lx).attr("y", ly + 4)
        .attr("text-anchor", anchor)
        .attr("font-size", 11.5).attr("font-weight", 600).attr("fill", C.ink)
        .text(ax.label);
    });

    /* ── data polygons ── */
    if(showV1){
      svg.append("path").attr("d", polygonPath(v1))
        .attr("fill", C.blue).attr("fill-opacity", 0.15)
        .attr("stroke", C.blue).attr("stroke-width", 2);
      /* dots */
      axes.forEach(function(ax, i){
        var p = pt(i, v1[ax.key]);
        svg.append("circle").attr("cx", p.x).attr("cy", p.y).attr("r", 4)
          .attr("fill", C.blue).attr("stroke", C.panel).attr("stroke-width", 1.5);
      });
    }
    if(showV2){
      svg.append("path").attr("d", polygonPath(v2))
        .attr("fill", C.accent).attr("fill-opacity", 0.15)
        .attr("stroke", C.accent).attr("stroke-width", 2);
      /* dots */
      axes.forEach(function(ax, i){
        var p = pt(i, v2[ax.key]);
        svg.append("circle").attr("cx", p.x).attr("cy", p.y).attr("r", 4)
          .attr("fill", C.accent).attr("stroke", C.panel).attr("stroke-width", 1.5);
      });
    }

    /* ── legend ── */
    var legY = 18;
    if(showV1){
      svg.append("rect").attr("x", 12).attr("y", legY - 8).attr("width", 14).attr("height", 14).attr("rx", 3)
        .attr("fill", C.blue).attr("fill-opacity", 0.3).attr("stroke", C.blue).attr("stroke-width", 1.5);
      svg.append("text").attr("x", 32).attr("y", legY + 3).attr("font-size", 11).attr("fill", C.blue).text("Agent v1 (baseline)");
    }
    if(showV2){
      var legX = showV1 ? 180 : 12;
      svg.append("rect").attr("x", legX).attr("y", legY - 8).attr("width", 14).attr("height", 14).attr("rx", 3)
        .attr("fill", C.accent).attr("fill-opacity", 0.3).attr("stroke", C.accent).attr("stroke-width", 1.5);
      svg.append("text").attr("x", legX + 20).attr("y", legY + 3).attr("font-size", 11).attr("fill", C.accent).text("Agent v2 (improved)");
    }

    /* ── readout ── */
    var readout = "";
    if(showV1 && showV2){
      var v1avg = Math.round((v1.tool + v1.plan + v1.memory + v1.task + v1.cost) / N);
      var v2avg = Math.round((v2.tool + v2.plan + v2.memory + v2.task + v2.cost) / N);
      readout = "v1 avg: <b>" + v1avg + "</b> &middot; v2 avg: <b>" + v2avg + "</b> &middot; v2 improves on every axis, largest gain in <b>Task Completion</b> (+32) and <b>Cost Efficiency</b> (+27)";
    } else if(showV1){
      readout = "Showing <b>Agent v1</b> (baseline) only &mdash; weakest axis: <b>Cost Efficiency</b> (45)";
    } else if(showV2){
      readout = "Showing <b>Agent v2</b> (improved) only &mdash; strongest axis: <b>Task Completion</b> (90)";
    } else {
      readout = "No agent selected &mdash; click a toggle above";
    }
    d3.select("#eval-radar-read").html(readout);
  }

  /* ── button handlers ── */
  function setActive(id){
    d3.selectAll("#eval-show-v1,#eval-show-v2,#eval-show-both").classed("active", false);
    d3.select(id).classed("active", true);
  }

  d3.select("#eval-show-v1").on("click", function(){
    showV1 = true; showV2 = false; setActive("#eval-show-v1"); draw();
  });
  d3.select("#eval-show-v2").on("click", function(){
    showV1 = false; showV2 = true; setActive("#eval-show-v2"); draw();
  });
  d3.select("#eval-show-both").on("click", function(){
    showV1 = true; showV2 = true; setActive("#eval-show-both"); draw();
  });

  draw();
})();
