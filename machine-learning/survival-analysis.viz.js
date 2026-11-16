/* survival-analysis.viz.js — Kaplan-Meier curves with two groups,
   censored observations, and a live log-rank test.
   Loaded after data.js / notes.js as the last <script> in
   survival-analysis.html. */

(function () {
  /* ── palette ── */
  var C = {
    accent: "#ffb454", blue: "#5b9cff", good: "#4ade80",
    bad: "#f87171", muted: "#9aa3b2", ink: "#e6e9ef", line: "#2a2f3a"
  };

  var svg = d3.select("#km-svg");
  if (svg.empty()) return;           /* bail if the page has no viz slot */

  var W = 640, H = 340;
  var m = { t: 24, r: 24, b: 42, l: 54 };

  /* ── seeded PRNG (same hash-mix as the bias-variance viz) ── */
  var seed = 20241116;
  function rnd() {
    seed |= 0;
    seed = seed + 0x6D2B79F5 | 0;
    var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }
  function expRand(rate) {
    var u = 0;
    while (u === 0) u = rnd();
    return -Math.log(u) / rate;
  }

  /* ── generate synthetic survival data ── */
  function generateData(n, baseRate, hazardRatio, censorRate) {
    var data = [];
    var i;
    for (i = 0; i < n; i++) {
      var tA = expRand(baseRate);
      var cA = expRand(censorRate);
      data.push({ time: Math.min(tA, cA), event: tA <= cA ? 1 : 0, group: "A" });
    }
    for (i = 0; i < n; i++) {
      var tB = expRand(baseRate * hazardRatio);
      var cB = expRand(censorRate);
      data.push({ time: Math.min(tB, cB), event: tB <= cB ? 1 : 0, group: "B" });
    }
    return data;
  }

  /* ── Kaplan-Meier computation ── */
  function computeKM(obs) {
    obs = obs.slice().sort(function (a, b) {
      return a.time - b.time || b.event - a.event; /* events before censorings at same time */
    });
    var steps = [{ time: 0, survival: 1.0 }];
    var censored = [];
    var nAtRisk = obs.length;
    var curS = 1.0;
    var i = 0;
    while (i < obs.length) {
      var t = obs[i].time;
      var d = 0, c = 0;
      while (i < obs.length && Math.abs(obs[i].time - t) < 1e-10) {
        if (obs[i].event) d++; else c++;
        i++;
      }
      if (d > 0) {
        curS *= (1 - d / nAtRisk);
        steps.push({ time: t, survival: curS });
      }
      if (c > 0) {
        censored.push({ time: t, survival: curS });
      }
      nAtRisk -= (d + c);
    }
    return { steps: steps, censored: censored };
  }

  /* ── log-rank test (two groups) ── */
  function logRankTest(dataA, dataB) {
    /* pool all observations and find distinct event times */
    var all = dataA.concat(dataB);
    var eventTimes = [];
    var seen = {};
    all.forEach(function (d) {
      if (d.event && !seen[d.time]) { seen[d.time] = true; eventTimes.push(d.time); }
    });
    eventTimes.sort(function (a, b) { return a - b; });

    var sumOE = 0, sumV = 0;

    eventTimes.forEach(function (t) {
      /* at risk = observation time >= t */
      var n1 = 0, n2 = 0, d1 = 0, d2 = 0;
      dataA.forEach(function (d) {
        if (d.time >= t) { n1++; if (d.event && Math.abs(d.time - t) < 1e-10) d1++; }
      });
      dataB.forEach(function (d) {
        if (d.time >= t) { n2++; if (d.event && Math.abs(d.time - t) < 1e-10) d2++; }
      });
      var n = n1 + n2, d = d1 + d2;
      if (n > 1 && d > 0) {
        var e1 = d * n1 / n;
        var v = d * n1 * n2 * (n - d) / (n * n * (n - 1));
        sumOE += (d1 - e1);
        sumV += v;
      }
    });

    var chiSq = sumV > 0 ? (sumOE * sumOE) / sumV : 0;
    return { chiSq: chiSq, pValue: chiSqP1(chiSq) };
  }

  /* ── chi-square p-value for 1 df (Abramowitz & Stegun erfc) ── */
  function chiSqP1(x) {
    if (x <= 0) return 1;
    var z = Math.sqrt(x / 2);
    var t = 1 / (1 + 0.3275911 * z);
    var poly = t * (0.254829592 + t * (-0.284496736 +
      t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    return Math.min(1, poly * Math.exp(-z * z));
  }

  /* ── main draw ── */
  function draw() {
    var hr = +d3.select("#km-hr").property("value");
    var cens = +d3.select("#km-cens").property("value");
    var showB = d3.select("#km-showb").property("checked");
    var n = 50;

    var data = generateData(n, 0.05, hr, cens);
    var dataA = data.filter(function (d) { return d.group === "A"; });
    var dataB = data.filter(function (d) { return d.group === "B"; });

    var kmA = computeKM(dataA);
    var kmB = computeKM(dataB);

    /* axis domains */
    var maxTime = d3.max(data, function (d) { return d.time; }) * 1.08;
    var sx = d3.scaleLinear().domain([0, maxTime]).range([m.l, W - m.r]);
    var sy = d3.scaleLinear().domain([0, 1.02]).range([H - m.b, m.t]);

    svg.selectAll("*").remove();

    /* ── axes ── */
    svg.append("g").attr("class", "axis")
      .attr("transform", "translate(0," + (H - m.b) + ")")
      .call(d3.axisBottom(sx).ticks(7).tickFormat(function (d) { return d.toFixed(0); }));

    svg.append("g").attr("class", "axis")
      .attr("transform", "translate(" + m.l + ",0)")
      .call(d3.axisLeft(sy).ticks(5).tickFormat(d3.format(".1f")));

    /* axis labels */
    svg.append("text")
      .attr("x", (m.l + W - m.r) / 2).attr("y", H - 4)
      .attr("text-anchor", "middle").attr("fill", C.muted).attr("font-size", 11)
      .text("Time");
    svg.append("text")
      .attr("x", 14).attr("y", (m.t + H - m.b) / 2)
      .attr("text-anchor", "middle").attr("fill", C.muted).attr("font-size", 11)
      .attr("transform", "rotate(-90, 14," + ((m.t + H - m.b) / 2) + ")")
      .text("Ŝ(t)");   /* Ŝ rendered as literal unicode */

    /* ── 50% reference line ── */
    svg.append("line")
      .attr("x1", m.l).attr("y1", sy(0.5))
      .attr("x2", W - m.r).attr("y2", sy(0.5))
      .attr("stroke", C.line).attr("stroke-dasharray", "3 4")
      .attr("stroke-opacity", 0.6);
    svg.append("text")
      .attr("x", W - m.r + 2).attr("y", sy(0.5) + 4)
      .attr("fill", C.muted).attr("font-size", 9).text("50%");

    /* ── draw a KM curve ── */
    function drawKM(km, color) {
      var steps = km.steps;
      var cens = km.censored;

      /* build step-function path */
      var path = "M" + sx(steps[0].time) + "," + sy(steps[0].survival);
      for (var i = 1; i < steps.length; i++) {
        path += "H" + sx(steps[i].time) + "V" + sy(steps[i].survival);
      }
      /* extend to max time */
      path += "H" + sx(maxTime);

      svg.append("path").attr("d", path)
        .attr("fill", "none").attr("stroke", color)
        .attr("stroke-width", 2.3).attr("stroke-linecap", "round");

      /* censored tick marks (small vertical lines on the curve) */
      cens.forEach(function (c) {
        svg.append("line")
          .attr("x1", sx(c.time)).attr("y1", sy(c.survival) - 6)
          .attr("x2", sx(c.time)).attr("y2", sy(c.survival) + 6)
          .attr("stroke", color).attr("stroke-width", 1.5)
          .attr("stroke-opacity", 0.65);
      });
    }

    /* always draw group A */
    drawKM(kmA, C.accent);
    if (showB) drawKM(kmB, C.blue);

    /* ── log-rank test ── */
    var lr = showB ? logRankTest(dataA, dataB) : null;

    /* ── readout ── */
    var evA = dataA.filter(function (d) { return d.event; }).length;
    var evB = dataB.filter(function (d) { return d.event; }).length;
    var txt = "Group A: <b>" + n + "</b> subjects, <b style=\"color:" + C.accent + "\">" + evA + "</b> events, <b>" + (n - evA) + "</b> censored";
    if (showB) {
      txt += " &nbsp;|&nbsp; Group B: <b>" + n + "</b> subjects, <b style=\"color:" + C.blue + "\">" + evB + "</b> events, <b>" + (n - evB) + "</b> censored";
      txt += "<br>Log-rank χ² = <b>" + lr.chiSq.toFixed(2) + "</b>";
      var pStr = lr.pValue < 0.001 ? "< 0.001" : lr.pValue.toFixed(3);
      var pCol = lr.pValue < 0.05 ? C.good : C.muted;
      txt += ", p = <b style=\"color:" + pCol + "\">" + pStr + "</b>";
      txt += lr.pValue < 0.05
        ? " · <b style=\"color:" + C.good + "\">significant difference</b>"
        : " · <b style=\"color:" + C.muted + "\">not significant</b>";
    }
    d3.select("#km-read").html(txt);
  }

  /* ── controls ── */
  d3.select("#km-hr").on("input", function () {
    d3.select("#km-hr-val").text((+this.value).toFixed(1));
    seed = 20241116;
    draw();
  });
  d3.select("#km-cens").on("input", function () {
    d3.select("#km-cens-val").text((+this.value).toFixed(2));
    seed = 20241116;
    draw();
  });
  d3.select("#km-showb").on("change", function () {
    seed = 20241116;
    draw();
  });
  d3.select("#km-new").on("click", function () {
    seed = Math.floor(Math.random() * 1e9);
    draw();
  });

  /* ── initial render ── */
  draw();
})();
