/* constrained-generation.viz.js — logits masking interactive visualization.
   Loaded after data.js / notes.js. IIFE-wrapped. Uses ../vendor/d3.min.js. */

(function(){
  /* ---- guard ---- */
  var svgEl = document.getElementById("constrained-svg");
  var readEl = document.getElementById("constrained-read");
  var btnEl  = document.getElementById("constrained-btn");
  var togEl  = document.getElementById("constrained-toggle");
  if(!svgEl || !readEl || !btnEl || !togEl || typeof d3 === "undefined") return;

  /* ---- palette ---- */
  var C = {
    valid:   "#60a5fa",   /* blue — valid token */
    masked:  "#f87171",   /* red — masked token */
    sampled: "#a3e635",   /* green accent — sampled token */
    ink:     "#e6e9ef",
    muted:   "#9aa3b2",
    line:    "#2a2f3a",
    panel:   "#11141b",
    bg:      "#181c24"
  };

  /* ---- data: simulated next-token distribution ---- */
  /* Context: "Our new product is better than" — the next token distribution */
  var RAW_TOKENS = [
    { token: '"ever"',      logit: 4.2,  competitor: false },
    { token: '"expected"',  logit: 3.8,  competitor: false },
    { token: '"Google"',    logit: 3.5,  competitor: true  },
    { token: '"the"',       logit: 3.3,  competitor: false },
    { token: '"Apple"',     logit: 3.0,  competitor: true  },
    { token: '"most"',      logit: 2.7,  competitor: false },
    { token: '"Samsung"',   logit: 2.4,  competitor: true  },
    { token: '"anything"',  logit: 2.1,  competitor: false },
    { token: '"before"',    logit: 1.6,  competitor: false },
    { token: '"Amazon"',    logit: 1.2,  competitor: true  }
  ];

  /* ---- layout ---- */
  var W = 640, H = 300;
  var m = { l: 90, r: 24, t: 36, b: 42 };
  var barH = 20, gap = 4;
  var svg = d3.select(svgEl);

  /* ---- helpers ---- */
  function softmax(logits) {
    var maxL = d3.max(logits);
    var exps = logits.map(function(l){ return Math.exp(l - maxL); });
    var sum  = d3.sum(exps);
    return exps.map(function(e){ return e / sum; });
  }

  /* Seeded PRNG for reproducible but interesting samples */
  function mulberry32(a) {
    return function() {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function sampleFrom(probs, rng) {
    var r = rng();
    var cum = 0;
    for (var i = 0; i < probs.length; i++) {
      cum += probs[i];
      if (r < cum) return i;
    }
    return probs.length - 1;
  }

  var seed = 42;
  var rng = mulberry32(seed);

  /* ---- state ---- */
  var sampledIdx = -1;

  function getData() {
    var constrained = togEl.checked;
    var tokens = RAW_TOKENS.map(function(d, i) {
      var isMasked = constrained && d.competitor;
      return {
        token: d.token,
        logit: d.logit,
        competitor: d.competitor,
        masked: isMasked,
        idx: i
      };
    });

    /* compute probabilities */
    var logitsForSoftmax = tokens.map(function(d) {
      return d.masked ? -Infinity : d.logit;
    });
    var probs = softmax(logitsForSoftmax);
    tokens.forEach(function(d, i) { d.prob = probs[i]; });

    /* also compute unconstrained probs for reference */
    var rawProbs = softmax(RAW_TOKENS.map(function(d){ return d.logit; }));
    tokens.forEach(function(d, i) { d.rawProb = rawProbs[i]; });

    return tokens;
  }

  /* ---- draw ---- */
  function draw() {
    svg.selectAll("*").remove();
    var data = getData();
    var constrained = togEl.checked;

    var maxProb = d3.max(data, function(d){ return Math.max(d.prob, d.rawProb); });
    var x = d3.scaleLinear()
      .domain([0, Math.ceil(maxProb * 100) / 100 + 0.02])
      .range([m.l, W - m.r]);

    var totalH = data.length * (barH + gap);
    var yStart = m.t + (H - m.t - m.b - totalH) / 2;

    /* title */
    svg.append("text")
      .attr("x", W / 2).attr("y", 18)
      .attr("text-anchor", "middle")
      .attr("fill", C.muted).attr("font-size", 11)
      .text("Next-token probabilities — context: \"Our new product is better than ...\"");

    /* x axis */
    svg.append("g")
      .attr("transform", "translate(0," + (H - m.b + 6) + ")")
      .call(d3.axisBottom(x).ticks(6).tickFormat(d3.format(".0%")))
      .call(function(g) {
        g.selectAll("text").attr("fill", C.muted);
        g.selectAll("line,path").attr("stroke", C.line);
      });
    svg.append("text")
      .attr("x", (m.l + W - m.r) / 2).attr("y", H - 4)
      .attr("text-anchor", "middle")
      .attr("fill", C.muted).attr("font-size", 11)
      .text("probability");

    /* bars */
    var bars = svg.selectAll(".bar-g")
      .data(data)
      .enter().append("g")
      .attr("class", "bar-g")
      .attr("transform", function(d, i) {
        return "translate(0," + (yStart + i * (barH + gap)) + ")";
      });

    /* reference bars (unconstrained, faded) when constrained is on */
    if (constrained) {
      bars.append("rect")
        .attr("x", m.l)
        .attr("y", 0)
        .attr("height", barH)
        .attr("rx", 3)
        .attr("width", function(d) { return Math.max(0, x(d.rawProb) - m.l); })
        .attr("fill", C.muted)
        .attr("opacity", 0.15);
    }

    /* main bars */
    bars.append("rect")
      .attr("x", m.l)
      .attr("y", 0)
      .attr("height", barH)
      .attr("rx", 3)
      .attr("width", function(d) { return Math.max(0, x(d.prob) - m.l); })
      .attr("fill", function(d) {
        if (d.idx === sampledIdx) return C.sampled;
        if (d.masked) return C.masked;
        return C.valid;
      })
      .attr("opacity", function(d) { return d.masked ? 0.35 : 0.85; });

    /* strikethrough on masked */
    bars.filter(function(d) { return d.masked; })
      .append("line")
      .attr("x1", m.l)
      .attr("x2", function(d) { return Math.max(m.l, x(d.rawProb)); })
      .attr("y1", barH / 2)
      .attr("y2", barH / 2)
      .attr("stroke", C.masked)
      .attr("stroke-width", 2)
      .attr("opacity", 0.7);

    /* "X" marker on masked bars */
    bars.filter(function(d) { return d.masked; })
      .append("text")
      .attr("x", function(d) { return Math.max(m.l + 8, x(d.rawProb) + 6); })
      .attr("y", barH / 2 + 4)
      .attr("fill", C.masked)
      .attr("font-size", 11)
      .attr("font-weight", 700)
      .text("MASKED");

    /* probability labels on valid bars */
    bars.filter(function(d) { return !d.masked; })
      .append("text")
      .attr("x", function(d) { return x(d.prob) + 5; })
      .attr("y", barH / 2 + 4)
      .attr("fill", function(d) { return d.idx === sampledIdx ? C.sampled : C.ink; })
      .attr("font-size", 10.5)
      .attr("font-weight", function(d) { return d.idx === sampledIdx ? 700 : 400; })
      .text(function(d) { return (d.prob * 100).toFixed(1) + "%"; });

    /* sampled marker */
    if (sampledIdx >= 0) {
      bars.filter(function(d) { return d.idx === sampledIdx; })
        .append("text")
        .attr("x", function(d) { return x(d.prob) + 42; })
        .attr("y", barH / 2 + 4)
        .attr("fill", C.sampled)
        .attr("font-size", 10.5)
        .attr("font-weight", 700)
        .text("SAMPLED");
    }

    /* token labels (left) */
    bars.append("text")
      .attr("x", m.l - 6)
      .attr("y", barH / 2 + 4)
      .attr("text-anchor", "end")
      .attr("fill", function(d) {
        if (d.idx === sampledIdx) return C.sampled;
        if (d.masked) return C.masked;
        return C.ink;
      })
      .attr("font-size", 11.5)
      .attr("font-family", "monospace")
      .attr("opacity", function(d) { return d.masked ? 0.5 : 1; })
      .attr("text-decoration", function(d) { return d.masked ? "line-through" : "none"; })
      .text(function(d) { return d.token; });

    /* legend */
    var lx = m.l, ly = H - m.b + 28;
    var legend = [
      { label: "Valid", color: C.valid },
      { label: "Masked", color: C.masked },
      { label: "Sampled", color: C.sampled }
    ];
    legend.forEach(function(item, i) {
      var offset = i * 100;
      svg.append("rect")
        .attr("x", lx + offset).attr("y", ly - 8)
        .attr("width", 10).attr("height", 10).attr("rx", 2)
        .attr("fill", item.color).attr("opacity", 0.8);
      svg.append("text")
        .attr("x", lx + offset + 14).attr("y", ly + 1)
        .attr("fill", C.muted).attr("font-size", 10)
        .text(item.label);
    });

    /* readout */
    updateReadout(data);
  }

  function updateReadout(data) {
    var constrained = togEl.checked;
    var validCount = data.filter(function(d){ return !d.masked; }).length;
    var maskedCount = data.length - validCount;
    var msg = "";
    if (constrained) {
      msg = maskedCount + " competitor-name tokens masked. Remaining " + validCount + " tokens renormalised.";
    } else {
      msg = "No constraints active. All " + data.length + " tokens eligible for sampling.";
    }
    if (sampledIdx >= 0) {
      var d = data[sampledIdx];
      if (d.masked) {
        msg += " Previously sampled token is now masked.";
      } else {
        msg += " Sampled: " + d.token + " (" + (d.prob * 100).toFixed(1) + "%).";
      }
    }
    readEl.textContent = msg;
  }

  /* ---- events ---- */
  togEl.addEventListener("change", function() {
    sampledIdx = -1;
    draw();
  });

  btnEl.addEventListener("click", function() {
    var data = getData();
    var probs = data.map(function(d){ return d.prob; });
    sampledIdx = sampleFrom(probs, rng);
    /* advance seed for next sample */
    seed += 7;
    rng = mulberry32(seed);
    draw();
  });

  /* initial draw */
  draw();

})();
