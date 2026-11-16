/* probabilistic-graphical-models.viz.js
   Interactive Bayesian-network d-separation checker.
   Loaded after data.js / notes.js (which define the colour palette C). */

(function () {
  /* ---- colour palette (from notes.js global C) ---- */
  var accent = "#ffb454",  blue = "#5b9cff", good = "#4ade80",
      bad    = "#f87171",  muted = "#9aa3b2", ink  = "#e6e9ef",
      line   = "#2a2f3a",  bg    = "#1e222d";

  /* ---- graph definition: a small "Student" Bayesian network ---- */
  var NODES = [
    { id: "D", label: "Difficulty", x: 140, y: 60  },
    { id: "I", label: "Intelligence", x: 400, y: 60  },
    { id: "G", label: "Grade", x: 270, y: 180 },
    { id: "S", label: "SAT", x: 530, y: 180 },
    { id: "L", label: "Letter", x: 270, y: 310 },
    { id: "J", label: "Job", x: 430, y: 310 }
  ];

  var EDGES = [
    { from: "D", to: "G" },
    { from: "I", to: "G" },
    { from: "I", to: "S" },
    { from: "G", to: "L" },
    { from: "L", to: "J" },
    { from: "S", to: "J" }
  ];

  /* ---- state ---- */
  var observed = {};   /* id -> true */
  var source   = null; /* id or null */
  var target   = null; /* id or null */

  /* ---- helpers: build adjacency ---- */
  var children = {}, parents = {};
  NODES.forEach(function (n) { children[n.id] = []; parents[n.id] = []; });
  EDGES.forEach(function (e) { children[e.from].push(e.to); parents[e.to].push(e.from); });

  /* ---- descendants of a node ---- */
  function descendants(id) {
    var visited = {}, stack = [id];
    while (stack.length) {
      var cur = stack.pop();
      children[cur].forEach(function (c) { if (!visited[c]) { visited[c] = true; stack.push(c); } });
    }
    return visited;
  }

  /* ---- d-separation via the Bayes-Ball algorithm ---- */
  /*  Returns { separated: bool, activePaths: [[id,...], ...] }
      We use a simplified reachability approach:
      Schedule: (node, direction) pairs; direction = "up" (from child) or "down" (from parent).
      A node is visited from below (child sent us up) or from above (parent sent us down).  */
  function dSeparation(src, tgt, obs) {
    if (!src || !tgt || src === tgt) return { separated: false, reachable: {} };

    /* phase 1: find all ancestors of observed nodes (needed for collider activation) */
    var obsAnc = {};
    function markAnc(id) {
      if (obsAnc[id]) return;
      obsAnc[id] = true;
      parents[id].forEach(markAnc);
    }
    Object.keys(obs).forEach(function (id) { if (obs[id]) markAnc(id); });

    /* phase 2: Bayes-Ball reachability */
    var visited = {};  /* "id,dir" -> true */
    var reachable = {};
    var queue = [];

    /* start: schedule src sending messages in both directions */
    queue.push({ id: src, dir: "up" });
    queue.push({ id: src, dir: "down" });

    while (queue.length) {
      var cur = queue.shift();
      var key = cur.id + "," + cur.dir;
      if (visited[key]) continue;
      visited[key] = true;
      reachable[cur.id] = true;

      if (cur.dir === "up" && !obs[cur.id]) {
        /* came from a child, node is NOT observed: pass to parents (up) and children (down) */
        parents[cur.id].forEach(function (p) { queue.push({ id: p, dir: "up" }); });
        children[cur.id].forEach(function (c) { queue.push({ id: c, dir: "down" }); });
      }
      if (cur.dir === "down") {
        /* came from a parent */
        if (!obs[cur.id]) {
          /* not observed: pass down to children */
          children[cur.id].forEach(function (c) { queue.push({ id: c, dir: "down" }); });
        }
        if (obs[cur.id] || obsAnc[cur.id]) {
          /* observed or ancestor of observed: pass up to parents (collider activation) */
          parents[cur.id].forEach(function (p) { queue.push({ id: p, dir: "up" }); });
        }
      }
    }

    return { separated: !reachable[tgt], reachable: reachable };
  }

  /* ---- SVG setup ---- */
  var svg = d3.select("#pgm-svg");
  var W = 680, H = 380;
  var defs = svg.append("defs");

  /* arrowhead marker */
  defs.append("marker")
    .attr("id", "pgm-arrow")
    .attr("viewBox", "0 0 10 10")
    .attr("refX", 28).attr("refY", 5)
    .attr("markerWidth", 8).attr("markerHeight", 8)
    .attr("orient", "auto-start-reverse")
    .append("path").attr("d", "M0,0 L10,5 L0,10 Z").attr("fill", muted);

  defs.append("marker")
    .attr("id", "pgm-arrow-active")
    .attr("viewBox", "0 0 10 10")
    .attr("refX", 28).attr("refY", 5)
    .attr("markerWidth", 8).attr("markerHeight", 8)
    .attr("orient", "auto-start-reverse")
    .append("path").attr("d", "M0,0 L10,5 L0,10 Z").attr("fill", good);

  defs.append("marker")
    .attr("id", "pgm-arrow-blocked")
    .attr("viewBox", "0 0 10 10")
    .attr("refX", 28).attr("refY", 5)
    .attr("markerWidth", 8).attr("markerHeight", 8)
    .attr("orient", "auto-start-reverse")
    .append("path").attr("d", "M0,0 L10,5 L0,10 Z").attr("fill", bad);

  /* background */
  svg.append("rect").attr("width", W).attr("height", H).attr("fill", "transparent");

  /* edge group (drawn first so nodes are on top) */
  var edgeG = svg.append("g");
  var edgeElems = edgeG.selectAll("line").data(EDGES).enter().append("line")
    .attr("x1", function (d) { return nodeById(d.from).x; })
    .attr("y1", function (d) { return nodeById(d.from).y; })
    .attr("x2", function (d) { return nodeById(d.to).x; })
    .attr("y2", function (d) { return nodeById(d.to).y; })
    .attr("stroke", muted).attr("stroke-width", 2)
    .attr("marker-end", "url(#pgm-arrow)");

  /* node group */
  var nodeG = svg.append("g");
  var nodeElems = nodeG.selectAll("g").data(NODES).enter().append("g")
    .attr("transform", function (d) { return "translate(" + d.x + "," + d.y + ")"; })
    .style("cursor", "pointer");

  nodeElems.append("circle")
    .attr("r", 26)
    .attr("fill", bg)
    .attr("stroke", muted)
    .attr("stroke-width", 2.5);

  nodeElems.append("text")
    .attr("text-anchor", "middle").attr("dy", 4)
    .attr("fill", ink).attr("font-size", 14).attr("font-weight", "bold")
    .text(function (d) { return d.id; });

  /* small label below */
  nodeElems.append("text")
    .attr("text-anchor", "middle").attr("dy", 44)
    .attr("fill", muted).attr("font-size", 10)
    .text(function (d) { return d.label; });

  /* ---- interaction ---- */
  /* left-click: toggle observed */
  nodeElems.on("click", function (event, d) {
    if (event && event.preventDefault) event.preventDefault();
    var id = d.id;
    if (id === source || id === target) return; /* don't observe source/target */
    observed[id] = !observed[id];
    if (!observed[id]) delete observed[id];
    update();
  });

  /* right-click: cycle through source -> target -> unset */
  nodeElems.on("contextmenu", function (event, d) {
    if (event && event.preventDefault) event.preventDefault();
    var id = d.id;
    if (source === id) {
      source = null;
    } else if (target === id) {
      target = null;
    } else if (!source) {
      source = id;
      delete observed[id];
    } else if (!target) {
      target = id;
      delete observed[id];
    } else {
      /* both set, replace target */
      target = id;
      delete observed[id];
    }
    update();
  });

  /* reset button */
  d3.select("#pgm-reset").on("click", function () {
    observed = {}; source = null; target = null;
    update();
  });

  /* ---- lookup helper ---- */
  function nodeById(id) {
    for (var i = 0; i < NODES.length; i++) { if (NODES[i].id === id) return NODES[i]; }
    return null;
  }

  /* ---- update / redraw ---- */
  function update() {
    var result = (source && target) ? dSeparation(source, target, observed) : null;
    var separated = result ? result.separated : null;
    var reachable = result ? result.reachable : {};

    /* update node circles */
    nodeElems.select("circle")
      .transition().duration(200)
      .attr("fill", function (d) {
        if (d.id === source) return blue;
        if (d.id === target) return accent;
        if (observed[d.id]) return "#6e7a8a";
        return bg;
      })
      .attr("stroke", function (d) {
        if (d.id === source) return blue;
        if (d.id === target) return accent;
        if (observed[d.id]) return ink;
        if (result && reachable[d.id]) return good;
        return muted;
      })
      .attr("stroke-width", function (d) {
        if (d.id === source || d.id === target) return 3.5;
        if (observed[d.id]) return 3;
        return 2.5;
      });

    /* update node id text colour */
    nodeElems.select("text")
      .attr("fill", function (d) {
        if (d.id === source || d.id === target) return "#0f1117";
        return ink;
      });

    /* "observed" badge */
    nodeElems.selectAll(".obs-badge").remove();
    nodeElems.filter(function (d) { return observed[d.id]; })
      .append("text").attr("class", "obs-badge")
      .attr("text-anchor", "middle").attr("dy", -34)
      .attr("fill", ink).attr("font-size", 10).attr("font-weight", "bold")
      .text("observed");

    /* update edges */
    edgeElems
      .transition().duration(200)
      .attr("stroke", function (d) {
        if (!result) return muted;
        /* highlight edges along reachable sub-graph */
        if (reachable[d.from] && reachable[d.to]) return separated ? bad : good;
        return muted;
      })
      .attr("stroke-width", function (d) {
        if (result && reachable[d.from] && reachable[d.to]) return 3;
        return 2;
      })
      .attr("marker-end", function (d) {
        if (!result) return "url(#pgm-arrow)";
        if (reachable[d.from] && reachable[d.to]) {
          return separated ? "url(#pgm-arrow-blocked)" : "url(#pgm-arrow-active)";
        }
        return "url(#pgm-arrow)";
      });

    /* update labels */
    d3.select("#pgm-source-label").html("Source: <b>" + (source || "none") + "</b>");
    d3.select("#pgm-target-label").html("Target: <b>" + (target || "none") + "</b>");

    /* readout */
    var obsSet = Object.keys(observed).filter(function (k) { return observed[k]; });
    var obsStr = obsSet.length ? "{" + obsSet.join(", ") + "}" : "∅ (empty set)";

    if (!source || !target) {
      d3.select("#pgm-readout").html("Right-click nodes to set source (blue) and target (orange). Left-click to observe/unobserve.");
    } else if (separated) {
      d3.select("#pgm-readout").html(
        "<b style='color:" + good + "'>" + source + " ⊥ " + target + " | " + obsStr + "</b> &mdash; " +
        "d-separated (conditionally independent). Every path is blocked."
      );
    } else {
      d3.select("#pgm-readout").html(
        "<b style='color:" + bad + "'>" + source + " ⊥̸ " + target + " | " + obsStr + "</b> &mdash; " +
        "d-connected (NOT conditionally independent). At least one active path exists."
      );
    }
  }

  update();
})();
