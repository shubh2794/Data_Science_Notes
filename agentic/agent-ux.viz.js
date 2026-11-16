/* agent-ux.viz.js — Autonomy slider visualization.
   Loaded after data.js / notes.js. IIFE-wrapped. */

(function(){
  var C = {
    accent: "#a3e635", good: "#4ade80", bad: "#f87171",
    a2: "#ffb454", blue: "#60a5fa", purple: "#c084fc",
    muted: "#9aa3b2", ink: "#e6e9ef", line: "#2a2f3a", panel: "#15181f"
  };

  var svg = d3.select("#autonomy-svg");
  var W = 700, H = 310;
  var active = 2; // default to "Approve" (index 2)

  var stops = [
    {
      label: "Manual",
      desc: "Human does everything; agent provides information on request",
      risk: 0.05, speed: 0.10, effort: 0.95
    },
    {
      label: "Suggest",
      desc: "Agent drafts actions or answers; human edits and executes",
      risk: 0.15, speed: 0.35, effort: 0.70
    },
    {
      label: "Approve",
      desc: "Agent proposes an action and waits for human yes/no",
      risk: 0.35, speed: 0.55, effort: 0.45
    },
    {
      label: "Monitor",
      desc: "Agent acts; human reviews a log and can intervene",
      risk: 0.65, speed: 0.80, effort: 0.20
    },
    {
      label: "Autonomous",
      desc: "Agent acts independently without notifying",
      risk: 0.90, speed: 0.95, effort: 0.05
    }
  ];

  // Layout constants
  var sliderY   = 52;
  var sliderX0  = 50;
  var sliderX1  = 400;
  var barX0     = 460;
  var barW      = 190;
  var barStartY = 50;
  var barH      = 22;
  var barGap    = 38;

  var barScale = d3.scaleLinear().domain([0, 1]).range([0, barW]);

  var indicators = [
    { key: "risk",   label: "Risk",         color: C.bad,  hi: "higher" },
    { key: "speed",  label: "Speed",        color: C.good, hi: "higher" },
    { key: "effort", label: "Human effort", color: C.blue, hi: "lower" }
  ];

  function stopX(i) {
    return sliderX0 + i * (sliderX1 - sliderX0) / (stops.length - 1);
  }

  function draw(animate) {
    svg.selectAll("*").remove();
    var dur = animate ? 400 : 0;

    // --- Slider track ---
    svg.append("line")
      .attr("x1", sliderX0).attr("x2", sliderX1)
      .attr("y1", sliderY).attr("y2", sliderY)
      .attr("stroke", C.line).attr("stroke-width", 4)
      .attr("stroke-linecap", "round");

    // Active track fill
    svg.append("line")
      .attr("x1", sliderX0).attr("x2", stopX(active))
      .attr("y1", sliderY).attr("y2", sliderY)
      .attr("stroke", C.accent).attr("stroke-width", 4)
      .attr("stroke-linecap", "round")
      .attr("stroke-opacity", 0.6);

    // --- Stop circles and labels ---
    stops.forEach(function(s, i) {
      var cx = stopX(i);
      var isActive = (i === active);

      // Clickable area (invisible, larger)
      svg.append("circle")
        .attr("cx", cx).attr("cy", sliderY)
        .attr("r", 18)
        .attr("fill", "transparent")
        .attr("cursor", "pointer")
        .on("click", function() { active = i; draw(true); });

      // Visible circle
      svg.append("circle")
        .attr("cx", cx).attr("cy", sliderY)
        .attr("r", isActive ? 10 : 7)
        .attr("fill", isActive ? C.accent : C.panel)
        .attr("stroke", isActive ? C.accent : C.muted)
        .attr("stroke-width", isActive ? 2.5 : 1.5)
        .attr("cursor", "pointer")
        .on("click", function() { active = i; draw(true); });

      // Label above
      svg.append("text")
        .attr("x", cx).attr("y", sliderY - 20)
        .attr("text-anchor", "middle")
        .attr("font-size", isActive ? 12.5 : 11)
        .attr("font-weight", isActive ? 700 : 400)
        .attr("fill", isActive ? C.ink : C.muted)
        .text(s.label);

      // Description below (only for active)
      if (isActive) {
        svg.append("text")
          .attr("x", cx).attr("y", sliderY + 28)
          .attr("text-anchor", "middle")
          .attr("font-size", 10)
          .attr("fill", C.muted)
          .text(s.desc);
      }
    });

    // --- Indicator bars ---
    var d = stops[active];

    indicators.forEach(function(ind, idx) {
      var by = barStartY + idx * barGap;
      var val = d[ind.key];

      // Label
      svg.append("text")
        .attr("x", barX0).attr("y", by - 6)
        .attr("font-size", 10.5).attr("font-weight", 600)
        .attr("fill", C.ink)
        .text(ind.label);

      // Background track
      svg.append("rect")
        .attr("x", barX0).attr("y", by)
        .attr("width", barW).attr("height", barH)
        .attr("rx", 4)
        .attr("fill", C.line);

      // Value bar (animated)
      var bar = svg.append("rect")
        .attr("x", barX0).attr("y", by)
        .attr("height", barH)
        .attr("rx", 4)
        .attr("fill", ind.color)
        .attr("fill-opacity", 0.85);

      if (dur > 0) {
        bar.attr("width", 0)
          .transition().duration(dur).ease(d3.easeCubicOut)
          .attr("width", barScale(val));
      } else {
        bar.attr("width", barScale(val));
      }

      // Value percentage
      svg.append("text")
        .attr("x", barX0 + barW + 10).attr("y", by + barH / 2 + 4)
        .attr("font-size", 11).attr("font-weight", 600)
        .attr("fill", ind.color)
        .text(Math.round(val * 100) + "%");
    });

    // --- Title label for bars section ---
    svg.append("text")
      .attr("x", barX0 + barW / 2).attr("y", 28)
      .attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", C.muted)
      .text("indicators at this level");

    // --- Guidance text at bottom ---
    var guideY = 210;
    var guides = [
      { level: 0, text: "Best for: exploration and learning. The agent is a reference tool, not an actor." },
      { level: 1, text: "Best for: creative work, drafting. The human shapes the output; the agent accelerates." },
      { level: 2, text: "Best for: medium-stakes actions (code changes, emails). The human is a gate." },
      { level: 3, text: "Best for: reversible, frequent tasks. Speed matters; the undo button is the safety net." },
      { level: 4, text: "Best for: low-stakes, well-tested routines (formatting, filing, spam filtering)." }
    ];

    var guide = guides[active];

    svg.append("rect")
      .attr("x", 30).attr("y", guideY)
      .attr("width", W - 60).attr("height", 42)
      .attr("rx", 8)
      .attr("fill", C.panel)
      .attr("stroke", C.accent).attr("stroke-opacity", 0.3);

    svg.append("text")
      .attr("x", W / 2).attr("y", guideY + 17)
      .attr("text-anchor", "middle")
      .attr("font-size", 11).attr("font-weight", 600)
      .attr("fill", C.accent)
      .text(stops[active].label + " mode");

    svg.append("text")
      .attr("x", W / 2).attr("y", guideY + 33)
      .attr("text-anchor", "middle")
      .attr("font-size", 10.5)
      .attr("fill", C.muted)
      .text(guide.text);

    // --- Arrow hint ---
    svg.append("text")
      .attr("x", (sliderX0 + sliderX1) / 2).attr("y", H - 20)
      .attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", C.muted)
      .text("← less autonomy          more autonomy →");

    // --- Readout ---
    d3.select("#autonomy-read").html(
      "Level: <b>" + stops[active].label + "</b> — " +
      "Risk " + Math.round(d.risk * 100) + "% · " +
      "Speed " + Math.round(d.speed * 100) + "% · " +
      "Human effort " + Math.round(d.effort * 100) + "%"
    );
  }

  draw(false);
})();
