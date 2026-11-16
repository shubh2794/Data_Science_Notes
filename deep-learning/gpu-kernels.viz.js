/* gpu-kernels.viz.js — GPU memory hierarchy visualization.
   Loaded after data.js / notes.js. IIFE, no globals. */
(function(){
  var C = {
    accent:"#a3e635", good:"#4ade80", bad:"#f87171", a2:"#ffb454",
    blue:"#60a5fa", purple:"#c084fc", muted:"#9aa3b2", ink:"#e6e9ef",
    line:"#2a2f3a", panel:"#15181f"
  };

  var svg = d3.select("#gpu-mem-svg");
  var W = 640, H = 400;
  var fused = false;

  /* Memory hierarchy levels — innermost (fastest) first */
  var levels = [
    { name:"Registers",           size:"~256 KB/SM", bw:"fastest",       color:"#f87171", opacity:0.85 },
    { name:"Shared Memory / L1",  size:"228 KB/SM",  bw:"~19 TB/s agg",  color:"#ffb454", opacity:0.75 },
    { name:"L2 Cache",            size:"50 MB",      bw:"~12 TB/s",      color:"#a3e635", opacity:0.55 },
    { name:"Global Memory (HBM)", size:"80 GB",      bw:"3.35 TB/s",     color:"#60a5fa", opacity:0.40 }
  ];

  /* Layout: concentric rounded rectangles, centered */
  var cx = W / 2;
  var cy = H / 2 - 10;
  var paddings = [
    { hw: 70,  hh: 40 },   /* registers — innermost */
    { hw: 140, hh: 75 },   /* shared / L1 */
    { hw: 220, hh: 115 },  /* L2 */
    { hw: 300, hh: 160 }   /* HBM — outermost */
  ];

  /* Operation labels for the data-flow arrows */
  var ops = ["LayerNorm", "Dropout", "Add"];

  function draw() {
    svg.selectAll("*").remove();

    var defs = svg.append("defs");
    /* Arrow markers */
    defs.append("marker").attr("id","gm-arrow")
      .attr("viewBox","0 -5 10 10").attr("refX",8).attr("refY",0)
      .attr("markerWidth",7).attr("markerHeight",7).attr("orient","auto")
      .append("path").attr("d","M0,-4L9,0L0,4").attr("fill", fused ? C.good : C.bad);

    defs.append("marker").attr("id","gm-arrow-in")
      .attr("viewBox","0 -5 10 10").attr("refX",8).attr("refY",0)
      .attr("markerWidth",7).attr("markerHeight",7).attr("orient","auto")
      .append("path").attr("d","M0,-4L9,0L0,4").attr("fill", fused ? C.good : C.a2);

    /* Draw nested rectangles — outermost first so inner layers draw on top */
    for (var i = levels.length - 1; i >= 0; i--) {
      var lev = levels[i];
      var p = paddings[i];
      var x = cx - p.hw;
      var y = cy - p.hh;
      var w = p.hw * 2;
      var h = p.hh * 2;

      /* Background rect */
      svg.append("rect")
        .attr("x", x).attr("y", y).attr("width", w).attr("height", h)
        .attr("rx", 14).attr("ry", 14)
        .attr("fill", lev.color).attr("fill-opacity", lev.opacity * 0.18)
        .attr("stroke", lev.color).attr("stroke-opacity", lev.opacity)
        .attr("stroke-width", i === 0 ? 2 : 1.5);

      /* Label: name on the left edge inside the rect */
      var labelY = y + 16;
      var labelX = x + 10;
      /* For outer levels, put label at the top-left of the band */
      svg.append("text")
        .attr("x", labelX).attr("y", labelY)
        .attr("fill", lev.color).attr("fill-opacity", Math.min(1, lev.opacity + 0.3))
        .attr("font-size", i === 0 ? 11 : 10.5)
        .attr("font-weight", 600)
        .text(lev.name);

      /* Size + bandwidth annotation on the right */
      svg.append("text")
        .attr("x", x + w - 10).attr("y", labelY)
        .attr("text-anchor", "end")
        .attr("fill", C.muted).attr("fill-opacity", 0.9)
        .attr("font-size", 9.5)
        .text(lev.size + "  ·  " + lev.bw);
    }

    /* Speed gradient legend */
    svg.append("text").attr("x", cx - 90).attr("y", H - 10)
      .attr("fill", C.bad).attr("font-size", 10).attr("font-weight", 600).text("FAST (inner)");
    svg.append("text").attr("x", cx + 20).attr("y", H - 10)
      .attr("fill", C.blue).attr("font-size", 10).attr("font-weight", 600).text("SLOW (outer)");
    svg.append("line").attr("x1", cx - 92).attr("y1", H - 16).attr("x2", cx + 90).attr("y2", H - 16)
      .attr("stroke","url(#gm-grad)").attr("stroke-width", 3);

    /* Gradient for the speed legend bar */
    var grad = defs.append("linearGradient").attr("id","gm-grad");
    grad.append("stop").attr("offset","0%").attr("stop-color", C.bad);
    grad.append("stop").attr("offset","50%").attr("stop-color", C.a2);
    grad.append("stop").attr("offset","100%").attr("stop-color", C.blue);

    /* ---- Data flow arrows ---- */
    if (!fused) {
      drawUnfusedFlow();
    } else {
      drawFusedFlow();
    }

    /* Readout */
    if (fused) {
      d3.select("#gpu-mem-read").html(
        "<b>Fused kernel:</b> data loaded from HBM once → all 3 ops execute in Shared Memory / Registers → result written to HBM once. " +
        "<span style='color:" + C.good + "'>2 HBM accesses</span> (1 read + 1 write)."
      );
    } else {
      d3.select("#gpu-mem-read").html(
        "<b>Unfused kernels:</b> each op reads from HBM, computes, writes back to HBM. 3 ops = " +
        "<span style='color:" + C.bad + "'>6 HBM accesses</span> (3 reads + 3 writes). Most time spent waiting for memory."
      );
    }
  }

  function drawUnfusedFlow() {
    /* Unfused: 3 separate ops, each arrow goes from center out to HBM and back.
       Show them as a sequence on the right side of the diagram. */
    var rightX = cx + 80;
    var hbmTop = cy - paddings[3].hh;
    var regY = cy;
    var startY = cy - 60;
    var stepY = 44;

    for (var i = 0; i < ops.length; i++) {
      var yBase = startY + i * stepY;
      var arrowX = rightX + i * 50;

      /* Down arrow: HBM → Registers (read) */
      svg.append("line")
        .attr("x1", arrowX).attr("y1", hbmTop + 24)
        .attr("x2", arrowX).attr("y2", yBase)
        .attr("stroke", C.a2).attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "4,3")
        .attr("marker-end", "url(#gm-arrow-in)");

      /* Op label */
      svg.append("rect")
        .attr("x", arrowX - 32).attr("y", yBase - 8)
        .attr("width", 64).attr("height", 16).attr("rx", 8)
        .attr("fill", C.panel).attr("stroke", C.bad).attr("stroke-opacity", 0.6);
      svg.append("text")
        .attr("x", arrowX).attr("y", yBase + 4)
        .attr("text-anchor", "middle")
        .attr("fill", C.bad).attr("font-size", 9).attr("font-weight", 600)
        .text(ops[i]);

      /* Up arrow: Registers → HBM (write back) */
      svg.append("line")
        .attr("x1", arrowX + 14).attr("y1", yBase)
        .attr("x2", arrowX + 14).attr("y2", hbmTop + 24)
        .attr("stroke", C.bad).attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "4,3")
        .attr("marker-end", "url(#gm-arrow)");
    }

    /* Annotation */
    svg.append("text")
      .attr("x", rightX + 50).attr("y", hbmTop + 14)
      .attr("text-anchor", "middle")
      .attr("fill", C.muted).attr("font-size", 9)
      .text("HBM round-trips");
  }

  function drawFusedFlow() {
    /* Fused: one read from HBM, ops stay in shared/registers, one write back */
    var rightX = cx + 80;
    var hbmTop = cy - paddings[3].hh;
    var sharedTop = cy - paddings[1].hh;
    var regY = cy;

    /* Single down arrow: HBM → Shared Memory (one read) */
    svg.append("line")
      .attr("x1", rightX).attr("y1", hbmTop + 24)
      .attr("x2", rightX).attr("y2", sharedTop + 20)
      .attr("stroke", C.good).attr("stroke-width", 2)
      .attr("marker-end", "url(#gm-arrow)");

    svg.append("text")
      .attr("x", rightX + 8).attr("y", hbmTop + 50)
      .attr("fill", C.muted).attr("font-size", 9)
      .text("1 read");

    /* Ops chained inside shared memory area */
    var opY = cy - 30;
    var opStep = 28;
    for (var i = 0; i < ops.length; i++) {
      var y = opY + i * opStep;
      svg.append("rect")
        .attr("x", rightX - 32).attr("y", y - 8)
        .attr("width", 64).attr("height", 16).attr("rx", 8)
        .attr("fill", C.panel).attr("stroke", C.good).attr("stroke-opacity", 0.7);
      svg.append("text")
        .attr("x", rightX).attr("y", y + 4)
        .attr("text-anchor", "middle")
        .attr("fill", C.good).attr("font-size", 9).attr("font-weight", 600)
        .text(ops[i]);

      /* Chain arrow between ops (within shared mem) */
      if (i < ops.length - 1) {
        svg.append("line")
          .attr("x1", rightX).attr("y1", y + 9)
          .attr("x2", rightX).attr("y2", y + opStep - 9)
          .attr("stroke", C.good).attr("stroke-width", 1.2)
          .attr("stroke-opacity", 0.6)
          .attr("marker-end", "url(#gm-arrow)");
      }
    }

    /* Single up arrow: Shared Memory → HBM (one write) */
    var lastOpY = opY + (ops.length - 1) * opStep;
    svg.append("line")
      .attr("x1", rightX + 40).attr("y1", lastOpY)
      .attr("x2", rightX + 40).attr("y2", hbmTop + 24)
      .attr("stroke", C.good).attr("stroke-width", 2)
      .attr("stroke-dasharray", "4,3")
      .attr("marker-end", "url(#gm-arrow)");

    svg.append("text")
      .attr("x", rightX + 48).attr("y", hbmTop + 50)
      .attr("fill", C.muted).attr("font-size", 9)
      .text("1 write");

    /* Bracket around shared mem area */
    svg.append("text")
      .attr("x", rightX + 70).attr("y", cy + 4)
      .attr("fill", C.good).attr("font-size", 9).attr("font-weight", 600)
      .text("stays in");
    svg.append("text")
      .attr("x", rightX + 70).attr("y", cy + 16)
      .attr("fill", C.good).attr("font-size", 9).attr("font-weight", 600)
      .text("SRAM");
  }

  /* Toggle button */
  d3.select("#gpu-mem-toggle").on("click", function() {
    fused = !fused;
    d3.select(this).text("mode: " + (fused ? "fused" : "unfused"));
    draw();
  });

  draw();
})();
