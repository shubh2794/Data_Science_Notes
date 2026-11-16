/* kg-question-answering.viz.js — Agentic KG-QA pipeline visualization.
   Loaded after data.js / notes.js; draws an animated pipeline in #qa-pipe-svg. */

(function(){
  var C = {
    accent: "#a3e635",
    good:   "#4ade80",
    bad:    "#f87171",
    a2:     "#ffb454",
    blue:   "#60a5fa",
    purple: "#c084fc",
    muted:  "#9aa3b2",
    ink:    "#e6e9ef",
    line:   "#2a2f3a",
    panel:  "#15181f"
  };

  var svg = d3.select("#qa-pipe-svg");
  if (svg.empty()) return;

  var W = 720, H = 260;
  var readout = d3.select("#qa-pipe-read");
  var btn = d3.select("#qa-pipe-btn");

  /* ── stage definitions ───────────────────────────────── */
  var stages = [
    { id: "intent",  label: "Intent",        short: "Classify question", color: C.purple },
    { id: "schema",  label: "Schema",        short: "Extract schema",    color: C.blue   },
    { id: "query",   label: "Text-to-Query", short: "Generate Cypher",   color: C.a2     },
    { id: "execute", label: "Execute",       short: "Run query",         color: C.accent },
    { id: "route",   label: "Route",         short: "Format result",     color: C.muted  },
    { id: "summary", label: "Summary",       short: "Summarize",         color: C.good   }
  ];

  var N = stages.length;
  var boxW = 88, boxH = 48;
  var padX = 22;
  var totalW = N * boxW + (N - 1) * padX;
  var offsetX = (W - totalW) / 2;
  var baseY = 90;

  /* Compute x positions */
  stages.forEach(function(s, i) {
    s.x = offsetX + i * (boxW + padX);
    s.cx = s.x + boxW / 2;
    s.cy = baseY + boxH / 2;
  });

  /* ── arrow marker ────────────────────────────────────── */
  var defs = svg.append("defs");
  defs.append("marker")
    .attr("id", "qa-arrow")
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 9).attr("refY", 0)
    .attr("markerWidth", 7).attr("markerHeight", 7)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,-4L9,0L0,4")
    .attr("fill", C.muted);

  defs.append("marker")
    .attr("id", "qa-arrow-bad")
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 9).attr("refY", 0)
    .attr("markerWidth", 7).attr("markerHeight", 7)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,-4L9,0L0,4")
    .attr("fill", C.bad);

  /* ── draw static elements ────────────────────────────── */
  var gEdges = svg.append("g").attr("class", "edges");
  var gBoxes = svg.append("g").attr("class", "boxes");
  var gTokens = svg.append("g").attr("class", "tokens");

  /* Forward edges */
  for (var i = 0; i < N - 1; i++) {
    gEdges.append("line")
      .attr("x1", stages[i].x + boxW)
      .attr("y1", stages[i].cy)
      .attr("x2", stages[i + 1].x)
      .attr("y2", stages[i + 1].cy)
      .attr("stroke", C.line)
      .attr("stroke-width", 1.5)
      .attr("marker-end", "url(#qa-arrow)");
  }

  /* Retry arc (Execute → Text-to-Query) drawn as a curved path below the boxes */
  var execStage = stages[3]; /* execute */
  var queryStage = stages[2]; /* text-to-query */
  var retryArcY = baseY + boxH + 55;
  var retryPath = "M" + execStage.cx + "," + (baseY + boxH) +
                  " C" + execStage.cx + "," + retryArcY +
                  " " + queryStage.cx + "," + retryArcY +
                  " " + queryStage.cx + "," + (baseY + boxH);

  var retryEdge = gEdges.append("path")
    .attr("d", retryPath)
    .attr("fill", "none")
    .attr("stroke", C.bad)
    .attr("stroke-width", 1.5)
    .attr("stroke-dasharray", "5,4")
    .attr("marker-end", "url(#qa-arrow-bad)")
    .attr("opacity", 0.35);

  /* Retry label */
  var retryLabel = gEdges.append("text")
    .attr("x", (execStage.cx + queryStage.cx) / 2)
    .attr("y", retryArcY - 4)
    .attr("text-anchor", "middle")
    .attr("font-size", 10)
    .attr("fill", C.bad)
    .attr("opacity", 0.35)
    .text("retry (error)");

  /* Stage boxes */
  stages.forEach(function(s) {
    var g = gBoxes.append("g").attr("id", "qa-box-" + s.id);
    g.append("rect")
      .attr("x", s.x).attr("y", baseY)
      .attr("width", boxW).attr("height", boxH)
      .attr("rx", 10)
      .attr("fill", C.panel)
      .attr("stroke", s.color)
      .attr("stroke-width", 1.5)
      .attr("stroke-opacity", 0.6);
    g.append("text")
      .attr("x", s.cx).attr("y", baseY + boxH / 2 - 4)
      .attr("text-anchor", "middle")
      .attr("font-size", 11)
      .attr("font-weight", 600)
      .attr("fill", s.color)
      .text(s.label);
    g.append("text")
      .attr("x", s.cx).attr("y", baseY + boxH / 2 + 12)
      .attr("text-anchor", "middle")
      .attr("font-size", 8.5)
      .attr("fill", C.muted)
      .text(s.short);
  });

  /* Title above the pipeline */
  svg.append("text")
    .attr("x", W / 2).attr("y", 28)
    .attr("text-anchor", "middle")
    .attr("font-size", 13)
    .attr("fill", C.ink)
    .attr("font-weight", 600)
    .text("Question → Intent → Schema → Query → Execute → Answer");

  /* "user question" label to the left */
  svg.append("text")
    .attr("x", stages[0].x - 10).attr("y", stages[0].cy + 4)
    .attr("text-anchor", "end")
    .attr("font-size", 10)
    .attr("fill", C.muted)
    .text("“Who manages …?”");

  readout.html("Press <b>Run query</b> to animate the pipeline.");

  /* ── animation ───────────────────────────────────────── */
  var running = false;

  function highlightBox(idx, color) {
    var s = stages[idx];
    svg.select("#qa-box-" + s.id + " rect")
      .transition().duration(200)
      .attr("stroke", color)
      .attr("stroke-width", 2.5)
      .attr("stroke-opacity", 1);
  }

  function resetBox(idx) {
    var s = stages[idx];
    svg.select("#qa-box-" + s.id + " rect")
      .transition().duration(300)
      .attr("stroke", s.color)
      .attr("stroke-width", 1.5)
      .attr("stroke-opacity", 0.6);
  }

  function resetAllBoxes() {
    stages.forEach(function(_, i) { resetBox(i); });
  }

  function animateToken(fromIdx, toIdx, color, dur, cb) {
    var from = stages[fromIdx];
    var to = stages[toIdx];
    var token = gTokens.append("circle")
      .attr("cx", from.cx)
      .attr("cy", from.cy)
      .attr("r", 5)
      .attr("fill", color)
      .attr("opacity", 0.9);
    token.transition()
      .duration(dur)
      .attr("cx", to.cx)
      .attr("cy", to.cy)
      .on("end", function() {
        token.remove();
        if (cb) cb();
      });
  }

  function animateRetryToken(cb) {
    /* Animate along the retry arc path */
    var from = stages[3]; /* execute */
    var to = stages[2];   /* text-to-query */
    var token = gTokens.append("circle")
      .attr("r", 5)
      .attr("fill", C.bad)
      .attr("opacity", 0.9);

    /* Create a temporary hidden path for getTotalLength */
    var tempPath = svg.append("path")
      .attr("d", retryPath)
      .attr("fill", "none")
      .attr("stroke", "none");
    var pathLen = tempPath.node().getTotalLength();

    token.transition()
      .duration(800)
      .attrTween("cx", function() {
        return function(t) {
          return tempPath.node().getPointAtLength(t * pathLen).x;
        };
      })
      .attrTween("cy", function() {
        return function(t) {
          return tempPath.node().getPointAtLength(t * pathLen).y;
        };
      })
      .on("end", function() {
        token.remove();
        tempPath.remove();
        if (cb) cb();
      });
  }

  function runPipeline() {
    if (running) return;
    running = true;
    btn.attr("disabled", true).text("Running…");
    resetAllBoxes();
    gTokens.selectAll("*").remove();

    /* Decide randomly: succeed on first try (60%) or need one retry (40%) */
    var needsRetry = Math.random() < 0.4;
    var didRetry = needsRetry;
    var stepDelay = 500;
    var messages = [
      "Intent detected: table, multi-hop",
      "Schema filtered: Person, Organization, WORKS_FOR, MANAGES",
      "Generated Cypher: MATCH (mgr)-[:MANAGES]->(emp)-[:WORKS_FOR]->(o) …",
      needsRetry ? "Execution error: unknown relationship type MANAGED_BY" : "Query executed: 5 records returned",
      needsRetry ? "Retrying: feeding error back to Text-to-Query…" : "Routing: format as table",
      "", /* filled dynamically */
    ];

    function step(idx) {
      if (idx >= N) {
        /* Done */
        readout.html("<b>Pipeline complete.</b> " +
          (didRetry ? "Query succeeded after 1 retry." : "Query succeeded on first attempt.") +
          " 5 records returned and summarized.");
        btn.attr("disabled", null).text("Run query");
        running = false;
        setTimeout(resetAllBoxes, 2000);
        return;
      }

      /* Highlight current stage */
      highlightBox(idx, stages[idx].color);
      readout.html("<b>" + stages[idx].label + ":</b> " + messages[idx]);

      /* Stage 3 (Execute) — handle error path */
      if (idx === 3 && needsRetry) {
        setTimeout(function() {
          highlightBox(3, C.bad);
          readout.html("<b>Execute:</b> " + messages[3]);

          /* Flash retry arc */
          retryEdge.transition().duration(300).attr("opacity", 1);
          retryLabel.transition().duration(300).attr("opacity", 1);

          setTimeout(function() {
            /* Animate token back to Text-to-Query */
            animateRetryToken(function() {
              highlightBox(2, C.a2);
              readout.html("<b>Text-to-Query:</b> Regenerating query with error context…");

              setTimeout(function() {
                resetBox(2);
                resetBox(3);
                /* Now re-run Execute successfully */
                needsRetry = false;
                messages[3] = "Query executed: 5 records returned (attempt 2)";
                messages[4] = "Routing: format as table";
                messages[5] = "Summary: 3 of 5 managers oversee cross-department teams.";

                /* Animate token forward from query to execute again */
                animateToken(2, 3, C.a2, stepDelay, function() {
                  highlightBox(3, C.accent);
                  readout.html("<b>Execute:</b> " + messages[3]);

                  retryEdge.transition().duration(500).attr("opacity", 0.35);
                  retryLabel.transition().duration(500).attr("opacity", 0.35);

                  setTimeout(function() {
                    resetBox(3);
                    /* Continue from Route (idx 4) */
                    animateToken(3, 4, C.accent, stepDelay, function() {
                      step(4);
                    });
                  }, stepDelay);
                });
              }, stepDelay);
            });
          }, 400);
        }, stepDelay);
        return;
      }

      /* Normal forward flow */
      if (idx === 5) {
        messages[5] = "Summary: 3 of 5 managers oversee cross-department teams.";
        readout.html("<b>" + stages[idx].label + ":</b> " + messages[idx]);
      }

      setTimeout(function() {
        resetBox(idx);
        if (idx < N - 1) {
          animateToken(idx, idx + 1, stages[idx].color, stepDelay, function() {
            step(idx + 1);
          });
        } else {
          step(N); /* triggers "done" */
        }
      }, stepDelay);
    }

    step(0);
  }

  btn.on("click", runPipeline);
})();
