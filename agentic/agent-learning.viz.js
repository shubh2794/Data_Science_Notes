/* agent-learning.viz.js — Learning Ladder visualization.
   Loaded after data.js / notes.js; IIFE-wrapped. */

(function(){
  var C = {
    accent:"#a3e635", good:"#4ade80", bad:"#f87171", a2:"#ffb454",
    blue:"#60a5fa", purple:"#c084fc", muted:"#9aa3b2", ink:"#e6e9ef",
    line:"#2a2f3a", panel:"#15181f"
  };

  var svg = d3.select("#learn-ladder-svg");
  var W = 640, H = 340;

  /* ── rung data ─────────────────────────────────────── */
  var rungs = [
    { label:"Exemplar Learning",   desc:"Retrieve successful examples as few-shot context",     perf:0.42 },
    { label:"Reflexion",           desc:"Self-critique failures and inject lessons learned",     perf:0.60 },
    { label:"Experience Replay",   desc:"Mine full trajectories for reusable strategies",        perf:0.74 },
    { label:"Fine-Tuning",         desc:"Change model weights on curated trajectories",          perf:0.91 }
  ];

  var rungW = 230, rungH = 52, rungGap = 12;
  var ladderX = 36;
  var ladderTopY = 24;
  /* compute Y so the stack grows bottom-up: rung 0 at the bottom */
  function rungY(i){ return ladderTopY + (rungs.length - 1 - i) * (rungH + rungGap); }

  /* performance curve area */
  var curveX0 = ladderX + rungW + 60;
  var curveX1 = W - 24;
  var curveY0 = ladderTopY;
  var curveY1 = rungY(0) + rungH;

  /* baseline perf before any learning */
  var baseline = 0.22;

  /* animation state */
  var active = -1;  /* index of highest activated rung; -1 = none */

  /* ── helpers ───────────────────────────────────────── */
  function rungFill(i){
    if (i > active) return C.panel;
    var fills = [C.blue, C.a2, C.purple, C.accent];
    return fills[i];
  }
  function rungStroke(i){
    return i <= active ? "none" : C.line;
  }
  function rungTextFill(i){
    return i <= active ? "#0f1117" : C.ink;
  }

  /* ── draw ──────────────────────────────────────────── */
  function draw(){
    svg.selectAll("*").remove();

    /* ── ladder rungs ──────────────────────────────── */
    /* vertical rail lines connecting rungs */
    var railX1 = ladderX + 18;
    var railX2 = ladderX + rungW - 18;
    for (var r = 0; r < rungs.length - 1; r++){
      var yBot = rungY(r);
      var yTop = rungY(r + 1) + rungH;
      svg.append("line").attr("x1", railX1).attr("y1", yTop).attr("x2", railX1).attr("y2", yBot)
        .attr("stroke", C.line).attr("stroke-width", 2);
      svg.append("line").attr("x1", railX2).attr("y1", yTop).attr("x2", railX2).attr("y2", yBot)
        .attr("stroke", C.line).attr("stroke-width", 2);
    }

    for (var i = 0; i < rungs.length; i++){
      var d = rungs[i];
      var y = rungY(i);

      /* rung rectangle */
      svg.append("rect")
        .attr("x", ladderX).attr("y", y).attr("width", rungW).attr("height", rungH)
        .attr("rx", 10).attr("fill", rungFill(i)).attr("stroke", rungStroke(i))
        .attr("stroke-width", 1.5)
        .attr("fill-opacity", i <= active ? 0.92 : 0.45);

      /* rung label */
      svg.append("text")
        .attr("x", ladderX + rungW / 2).attr("y", y + 20)
        .attr("text-anchor", "middle").attr("font-size", 12).attr("font-weight", 700)
        .attr("fill", rungTextFill(i))
        .text(d.label);

      /* rung description */
      svg.append("text")
        .attr("x", ladderX + rungW / 2).attr("y", y + 38)
        .attr("text-anchor", "middle").attr("font-size", 9)
        .attr("fill", i <= active ? "#1a1f2e" : C.muted)
        .attr("opacity", i <= active ? 0.8 : 0.6)
        .text(d.desc);
    }

    /* ── performance curve (staircase) ─────────────── */
    /* axes */
    svg.append("line").attr("x1", curveX0).attr("y1", curveY1).attr("x2", curveX1).attr("y2", curveY1)
      .attr("stroke", C.line).attr("stroke-width", 1);
    svg.append("line").attr("x1", curveX0).attr("y1", curveY0).attr("x2", curveX0).attr("y2", curveY1)
      .attr("stroke", C.line).attr("stroke-width", 1);

    /* axis labels */
    svg.append("text").attr("x", (curveX0 + curveX1) / 2).attr("y", curveY1 + 20)
      .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", C.muted).text("technique applied");
    svg.append("text")
      .attr("transform", "translate(" + (curveX0 - 18) + "," + ((curveY0 + curveY1) / 2) + ") rotate(-90)")
      .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", C.muted).text("performance");

    /* Y-axis ticks */
    var yScale = d3.scaleLinear().domain([0, 1]).range([curveY1, curveY0]);
    [0, 0.25, 0.50, 0.75, 1.0].forEach(function(v){
      svg.append("line").attr("x1", curveX0 - 4).attr("x2", curveX0).attr("y1", yScale(v)).attr("y2", yScale(v))
        .attr("stroke", C.muted).attr("stroke-width", 1);
      svg.append("text").attr("x", curveX0 - 8).attr("y", yScale(v) + 3)
        .attr("text-anchor", "end").attr("font-size", 9).attr("fill", C.muted)
        .text(d3.format(".0%")(v));
    });

    /* build the staircase points */
    var segW = (curveX1 - curveX0) / (rungs.length + 1);
    var points = [];
    /* baseline segment */
    points.push({ x: curveX0, y: yScale(baseline) });
    points.push({ x: curveX0 + segW * 0.9, y: yScale(baseline) });

    for (var j = 0; j < rungs.length; j++){
      var sx = curveX0 + segW * (j + 1);
      var perf = rungs[j].perf;
      if (j <= active) {
        /* riser */
        points.push({ x: sx, y: yScale(perf) });
        /* tread */
        points.push({ x: sx + segW * 0.9, y: yScale(perf) });
      }
    }

    /* draw staircase line */
    if (points.length > 1) {
      var lineGen = d3.line().x(function(p){ return p.x; }).y(function(p){ return p.y; });
      svg.append("path")
        .datum(points)
        .attr("fill", "none")
        .attr("stroke", C.accent)
        .attr("stroke-width", 2.5)
        .attr("d", lineGen);
    }

    /* draw area fill under the staircase */
    if (points.length > 1) {
      var areaPoints = points.slice();
      areaPoints.push({ x: points[points.length - 1].x, y: curveY1 });
      areaPoints.push({ x: curveX0, y: curveY1 });
      var areaGen = d3.line().x(function(p){ return p.x; }).y(function(p){ return p.y; });
      svg.append("path")
        .datum(areaPoints)
        .attr("fill", C.accent)
        .attr("fill-opacity", 0.08)
        .attr("stroke", "none")
        .attr("d", areaGen);
    }

    /* dots on the staircase at each plateau */
    /* baseline dot */
    svg.append("circle").attr("cx", curveX0 + segW * 0.45).attr("cy", yScale(baseline))
      .attr("r", 4).attr("fill", C.muted);

    var fills = [C.blue, C.a2, C.purple, C.accent];
    for (var k = 0; k <= active; k++){
      var dx = curveX0 + segW * (k + 1) + segW * 0.45;
      var dy = yScale(rungs[k].perf);
      svg.append("circle").attr("cx", dx).attr("cy", dy)
        .attr("r", 5).attr("fill", fills[k]);

      /* label on plateau */
      svg.append("text").attr("x", dx).attr("y", dy - 10)
        .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", fills[k])
        .text(d3.format(".0%")(rungs[k].perf));
    }

    /* baseline label */
    svg.append("text")
      .attr("x", curveX0 + segW * 0.45).attr("y", yScale(baseline) - 10)
      .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", C.muted)
      .text(d3.format(".0%")(baseline));

    /* ── animated dot ──────────────────────────────── */
    if (active >= 0) {
      /* dot on the ladder at the active rung */
      var dotY = rungY(active) + rungH / 2;
      var dotX = ladderX + rungW + 14;
      svg.append("circle")
        .attr("cx", dotX).attr("cy", dotY)
        .attr("r", 7).attr("fill", fills[active])
        .attr("opacity", 0)
        .transition().duration(400).attr("opacity", 1);
      svg.append("text")
        .attr("x", dotX + 12).attr("y", dotY + 4)
        .attr("font-size", 10).attr("font-weight", 700).attr("fill", fills[active])
        .attr("opacity", 0)
        .transition().duration(400).attr("opacity", 1)
        .text("active");
    }

    /* ── readout ───────────────────────────────────── */
    var readout = d3.select("#learn-ladder-read");
    if (active < 0) {
      readout.html("Baseline agent: <b>" + d3.format(".0%")(baseline) + "</b> task success. Click <b>Climb</b> to add learning techniques.");
    } else if (active < rungs.length - 1) {
      readout.html("<b>" + rungs[active].label + "</b> applied → performance at <b style=\"color:" + fills[active] + "\">" +
        d3.format(".0%")(rungs[active].perf) + "</b>. Click <b>Climb</b> to add the next technique.");
    } else {
      readout.html("All techniques applied. Performance: <b style=\"color:" + C.accent + "\">" +
        d3.format(".0%")(rungs[active].perf) + "</b>. Click <b>Climb</b> to reset.");
    }
  }

  /* ── interaction ───────────────────────────────────── */
  d3.select("#learn-ladder-btn").on("click", function(){
    if (active >= rungs.length - 1) {
      active = -1;
    } else {
      active++;
    }
    draw();
  });

  /* initial draw */
  draw();
})();
