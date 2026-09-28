/* index.viz.js — figures for deep-learning/classical-nn/index.html
   (overview of the Classical Neural Networks series).
   Loaded after ../../data.js → ../../notes.js. Self-contained.

     1  #tl-svg   timeline 1943 → 2006, one lane per family, click / step / play
     2  #ln-svg   classical family → modern mechanism lineage map              */

const CX = {
  ink: "#e6e9ef", muted: "#9aa3b2", line: "#2a2f3a", panel2: "#1e222d", faint: "#5d6675",
  accent: "#5b9cff", orange: "#ffb454", good: "#4ade80",
  lanes: ["#5b9cff", "#c084fc", "#f472b6", "#4ade80", "#ffb454", "#2dd4bf", "#e6e9ef"]
};

/* ═════════ 1 · #tl-svg — timeline ═════════ */
(function () {
  const svg = d3.select("#tl-svg");
  if (svg.empty()) return;
  const lanes = [
    "Learning rules",
    "Memory & energy",
    "Competitive & unsupervised",
    "Function approximation",
    "Training, identification & control",
    "Evolution & hardware",
    "Deep-learning mainline"
  ];
  const P = n => ["perceptrons.html", "hopfield-associative-memory.html", "boltzmann-machines-rbms.html", "self-organizing-maps.html",
    "rbf-networks.html", "neuro-fuzzy.html", "kalman-filter-training.html", "neural-system-identification.html",
    "neuroevolution.html", "neural-hardware.html"][n - 1];
  const PN = n => "Part " + n;
  /* [year, lane, title, description, href, link label] */
  const E = [
    [1943, 0, "McCulloch–Pitts neuron", "A binary threshold unit with excitatory inputs and absolute inhibition; networks of them compute any finite logical expression.", P(1), PN(1)],
    [1949, 0, "Hebb's rule", "Connections strengthen when the cells on both ends are active together — the first learning rule, Δw ∝ y·x.", P(1), PN(1)],
    [1958, 0, "Rosenblatt's perceptron", "A trainable threshold unit on random association features, with an error-correction rule and (1962) a convergence theorem.", P(1), PN(1)],
    [1960, 0, "Adaline and the LMS rule", "Widrow and Hoff learn from the linear output: stochastic gradient descent on squared error, the delta rule.", P(1), PN(1)],
    [1962, 0, "Novikoff's mistake bound; Madaline", "The perceptron makes at most (R/γ)² mistakes on separable data; layers of Adalines with a fixed logic output.", P(1), PN(1)],
    [1965, 0, "Cover's counting theorem", "A linear threshold unit can separate about two random patterns per weight — the first capacity result.", P(1) + "#separability", PN(1)],
    [1969, 0, "Minsky & Papert, Perceptrons", "Single-trainable-layer perceptrons cannot compute parity or connectedness with local features; funding moves away.", P(1) + "#xor", PN(1)],
    [1972, 1, "Linear associative memories", "Kohonen and Anderson store pattern pairs in a correlation (outer-product) matrix — memory as a weight matrix.", P(2), PN(2)],
    [1974, 6, "Backpropagation derived", "Werbos derives reverse-mode gradient training for multilayer networks in his thesis; it goes largely unnoticed.", "../neural-network-training.html#backprop", "Neural Network Training"],
    [1975, 5, "Genetic algorithms", "Holland formalises selection, crossover and mutation — the search engine later used to evolve networks.", P(9), PN(9)],
    [1980, 6, "Neocognitron", "Fukushima's hierarchy of local feature detectors with pooling — the architectural ancestor of CNNs.", "../convolutional-networks.html", "Convolutional Networks"],
    [1982, 1, "Hopfield network", "A symmetric recurrent network with an energy that asynchronous updates never increase; memories are its minima.", P(2), PN(2)],
    [1982, 2, "Kohonen's self-organizing map", "Competitive learning with a lattice neighbourhood produces topology-preserving maps of the input space.", P(4), PN(4)],
    [1985, 1, "Boltzmann machine", "Ackley, Hinton and Sejnowski: stochastic units, a Boltzmann distribution over states, and a learning rule with hidden units.", P(3), PN(3)],
    [1985, 3, "Takagi–Sugeno fuzzy model", "Fuzzy rules whose consequents are linear functions of the inputs — the rule form ANFIS later trains.", P(6), PN(6)],
    [1986, 6, "Backprop popularised", "Rumelhart, Hinton and Williams show hidden layers learn useful internal representations with the generalised delta rule.", "../neural-network-training.html#backprop", "Neural Network Training"],
    [1986, 1, "Harmonium (the RBM)", "Smolensky's bipartite restriction of the Boltzmann machine — no visible–visible or hidden–hidden connections.", P(3), PN(3)],
    [1987, 2, "Adaptive resonance theory (ART1)", "Carpenter and Grossberg: new categories are created only when no existing one matches well enough — stability with plasticity.", P(4), PN(4)],
    [1988, 1, "Bidirectional associative memory", "Kosko's two-layer hetero-associative extension of the Hopfield idea, with its own energy function.", P(2), PN(2)],
    [1988, 3, "RBF networks", "Broomhead and Lowe frame networks of radial basis functions as multivariable interpolation with a linear readout.", P(5), PN(5)],
    [1989, 4, "EKF training", "Singhal and Wu train multilayer networks with the extended Kalman filter: fewer epochs, much more work per step.", P(7), PN(7)],
    [1989, 6, "Universal approximation", "Cybenko and Hornik et al.: one hidden layer of sigmoids can approximate any continuous function on a compact set.", "../neural-networks.html#universal", "Feedforward Networks"],
    [1989, 5, "Analog VLSI neural systems", "Mead's analog neuromorphic circuits and the first commercial neural chips — networks cast directly in silicon.", P(10), PN(10)],
    [1990, 4, "Neural identification and control", "Narendra and Parthasarathy set out series-parallel and parallel identification models and neural adaptive control.", P(8), PN(8)],
    [1990, 5, "Cascade-correlation; optimal brain damage", "Networks that grow one hidden unit at a time, and networks pruned by a second-order saliency estimate.", P(9), PN(9)],
    [1991, 4, "Decoupled EKF", "Puskorius and Feldkamp split the covariance into per-neuron blocks, making Kalman training practical for larger and recurrent nets.", P(7), PN(7)],
    [1993, 3, "ANFIS", "Jang lays a Takagi–Sugeno fuzzy system out as a five-layer network trained by a hybrid of gradient descent and least squares.", P(6), PN(6)],
    [1997, 6, "LSTM", "Hochreiter and Schmidhuber's gated memory cell fixes the vanishing gradient in recurrent networks.", "../rnns-lstms.html", "RNNs & LSTMs"],
    [1998, 6, "Gradient-based learning for documents (LeNet-5)", "Convolutional networks trained end to end by backprop read cheques commercially.", "../convolutional-networks.html", "Convolutional Networks"],
    [2000, 4, "Unscented Kalman filter training", "Wan and van der Merwe replace linearisation with sigma points — derivative-free Kalman training.", P(7), PN(7)],
    [2002, 5, "NEAT", "Stanley and Miikkulainen evolve topology and weights together, using historical markings and speciation to protect innovation.", P(9), PN(9)],
    [2002, 1, "Contrastive divergence", "Hinton's CD-k: a few Gibbs steps from the data replace the intractable model expectation — RBMs become trainable.", P(3), PN(3)],
    [2006, 1, "Deep belief networks", "Hinton, Osindero and Teh stack greedily trained RBMs to initialise a deep network — the result that relaunched deep learning.", P(3), PN(3)]
  ].map((e, i) => ({ i, year: e[0], lane: e[1], title: e[2], desc: e[3], href: e[4], hl: e[5] }));
  const eras = [[1943, 1969, "first wave"], [1969, 1982, "the quiet years"], [1982, 1995, "the revival"], [1995, 2006, "the kernel decade"]];

  const sel = document.getElementById("tl-lane");
  lanes.forEach((l, i) => { const o = document.createElement("option"); o.value = i; o.textContent = l; sel.appendChild(o); });
  const W = 760, H = 360, L = 16, R = 12, top = 30, laneH = 40;
  const x = d3.scaleLinear().domain([1940, 2008]).range([L, W - R]);
  let cur = 0, timer = null;

  /* stack events that share a lane and a year, so no dot hides another */
  const slot = {};
  E.forEach(e => { const k = e.lane + ":" + e.year; e.k = slot[k] = (slot[k] === undefined ? 0 : slot[k] + 1); });
  const cy = e => top + e.lane * laneH + laneH / 2 + (e.k ? 9 * e.k : 0) - (slot[e.lane + ":" + e.year] ? 4.5 : 0);

  const gBack = svg.append("g"), gAxis = svg.append("g"), gDots = svg.append("g"), gSel = svg.append("g");
  eras.forEach((er, i) => {
    gBack.append("rect").attr("x", x(er[0])).attr("y", top - 18).attr("width", x(er[1]) - x(er[0])).attr("height", lanes.length * laneH + 18)
      .attr("fill", i % 2 ? "#ffffff" : CX.accent).attr("fill-opacity", i % 2 ? 0.025 : 0.05);
    gBack.append("text").attr("x", (x(er[0]) + x(er[1])) / 2).attr("y", top - 6).attr("text-anchor", "middle")
      .attr("font-size", 10).attr("fill", CX.muted).text(er[2]);
  });
  lanes.forEach((l, i) => {
    gBack.append("line").attr("x1", L).attr("x2", W - R).attr("y1", top + i * laneH + laneH / 2).attr("y2", top + i * laneH + laneH / 2)
      .attr("stroke", CX.line);
    gBack.append("text").attr("x", L + 2).attr("y", top + i * laneH + 11).attr("font-size", 9.5).attr("fill", CX.lanes[i]).attr("fill-opacity", 0.9).text(l);
  });
  const ax = gAxis.attr("transform", "translate(0," + (top + lanes.length * laneH + 4) + ")")
    .call(d3.axisBottom(x).tickValues(d3.range(1940, 2010, 5)).tickFormat(d3.format("d")));
  ax.selectAll("path,line").attr("stroke", CX.faint);
  ax.selectAll("text").attr("fill", CX.muted).attr("font-size", 9.5);

  const dots = gDots.selectAll("g.ev").data(E).join("g").attr("class", "ev").style("cursor", "pointer")
    .attr("transform", e => "translate(" + x(e.year) + "," + cy(e) + ")")
    .on("click", (ev, e) => { stop(); cur = e.i; update(); })
    .on("mouseenter", (ev, e) => { stop(); cur = e.i; update(); });
  dots.append("circle").attr("r", 5.5).attr("fill", e => CX.lanes[e.lane]).attr("stroke", "#0f1117").attr("stroke-width", 1.2);
  dots.append("title").text(e => e.year + " · " + e.title);

  function visible() {
    const v = sel.value;
    return E.filter(e => v === "all" || e.lane === +v);
  }
  function update() {
    const v = sel.value;
    dots.attr("opacity", e => (v === "all" || e.lane === +v) ? 1 : 0.12);
    gSel.selectAll("*").remove();
    const e = E[cur];
    gSel.append("line").attr("x1", x(e.year)).attr("x2", x(e.year)).attr("y1", top - 2).attr("y2", top + lanes.length * laneH)
      .attr("stroke", CX.ink).attr("stroke-opacity", 0.35).attr("stroke-dasharray", "2 3");
    gSel.append("circle").attr("cx", x(e.year)).attr("cy", cy(e)).attr("r", 10).attr("fill", "none").attr("stroke", CX.ink).attr("stroke-width", 1.6);
    const right = x(e.year) > W - 240;
    gSel.append("text").attr("x", x(e.year) + (right ? -14 : 14)).attr("y", cy(e) + 20).attr("text-anchor", right ? "end" : "start")
      .attr("font-size", 11).attr("font-weight", 600).attr("fill", CX.ink).attr("paint-order", "stroke").attr("stroke", "#171a23").attr("stroke-width", 3)
      .text(e.year + " · " + e.title);
    document.getElementById("tl-readout").innerHTML = "<b>" + e.year + " — " + e.title + "</b> <span style=\"color:" + CX.lanes[e.lane] + "\">(" + lanes[e.lane] + ")</span><br>" +
      e.desc + " → <a href=\"" + e.href + "\">" + e.hl + "</a>";
  }
  function move(d) {
    const vis = visible();
    if (!vis.length) return;
    let k = vis.findIndex(e => e.i === cur);
    if (k < 0) k = d > 0 ? -1 : vis.length;
    k = (k + d + vis.length) % vis.length;
    cur = vis[k].i;
    update();
  }
  function stop() { if (timer) { timer.stop(); timer = null; } document.getElementById("tl-play").textContent = "▶ Play"; }
  document.getElementById("tl-prev").addEventListener("click", () => { stop(); move(-1); });
  document.getElementById("tl-next").addEventListener("click", () => { stop(); move(1); });
  document.getElementById("tl-play").addEventListener("click", () => {
    if (timer) { stop(); return; }
    document.getElementById("tl-play").textContent = "❚❚ Pause";
    const vis = visible();
    if (vis.length && cur === vis[vis.length - 1].i) cur = vis[0].i; else move(1);
    update();
    timer = d3.interval(() => {
      const v = visible();
      if (!v.length || cur === v[v.length - 1].i) { stop(); return; }
      move(1);
    }, 1600);
  });
  sel.addEventListener("change", () => {
    stop();
    const vis = visible();
    if (vis.length && !vis.some(e => e.i === cur)) cur = vis[0].i;
    update();
  });
  update();
})();

/* ═════════ 2 · #ln-svg — lineage map ═════════ */
(function () {
  const svg = d3.select("#ln-svg");
  if (svg.empty()) return;
  const left = [
    ["Perceptrons & early rules", "perceptrons.html"],
    ["Hopfield & associative memory", "hopfield-associative-memory.html"],
    ["Boltzmann machines & RBMs", "boltzmann-machines-rbms.html"],
    ["SOMs & competitive learning", "self-organizing-maps.html"],
    ["RBF networks", "rbf-networks.html"],
    ["Neuro-fuzzy systems", "neuro-fuzzy.html"],
    ["Kalman filter training", "kalman-filter-training.html"],
    ["System identification & control", "neural-system-identification.html"],
    ["Neuroevolution & growing nets", "neuroevolution.html"],
    ["Neural hardware", "neural-hardware.html"]
  ];
  const right = [
    ["Dense layers + backprop", "../neural-networks.html"],
    ["SVMs & kernel methods", "../../machine-learning/support-vector-machines.html"],
    ["Attention", "../attention.html"],
    ["Energy-based & diffusion models", "../diffusion-models.html"],
    ["Autoencoders, VAEs & VQ codebooks", "../autoencoders-vaes.html"],
    ["Self-supervised pretraining", "../contrastive-learning.html"],
    ["Clustering & vector quantization", "../../machine-learning/clustering.html"],
    ["Mixture-of-experts routing", "../mixture-of-experts.html"],
    ["Preconditioned optimizers", "../neural-network-training.html#optimizers"],
    ["RNNs & state-space models", "../state-space-models.html"],
    ["World models & model-based control", "../world-models-jepa.html"],
    ["Architecture & hyperparameter search", "../../machine-learning/hyperparameter-tuning.html"],
    ["Low precision & accelerators", "../neural-network-training.html#precision"]
  ];
  /* [left index, right index, why] */
  const links = [
    [0, 0, "The weighted-sum unit is unchanged; replacing the step with a smooth nonlinearity turned the delta rule into backpropagation."],
    [0, 1, "The perceptron criterion max(0, −t·s) with a margin of 1 and a norm penalty is the hinge-loss SVM."],
    [1, 2, "A modern (continuous, exponential-energy) Hopfield update is softmax attention over the stored patterns."],
    [1, 3, "Recall as descent on an energy landscape is the template for energy-based models."],
    [2, 3, "p(x) ∝ e^(−E(x)) is still the model; score-based diffusion learns the gradient of log p and avoids the partition function."],
    [2, 4, "RBM hidden units as a learned latent code — autoencoders and VAEs learn the same kind of code with an explicit encoder."],
    [2, 5, "Greedy unsupervised pretraining then fine-tuning (2006) is the recipe self-supervised pretraining scaled; CD's positive/negative phases echo contrastive objectives."],
    [3, 6, "Competitive learning without a neighbourhood is online k-means; the SOM adds a lattice that preserves topology."],
    [3, 4, "Nearest-prototype assignment to a codebook, with the winner pulled toward its input, is the VQ-VAE quantizer."],
    [3, 7, "Winner-take-all competition among units reappears as top-k routing among experts."],
    [4, 1, "A Gaussian RBF is the RBF kernel; an RBF net with a centre on every point is a kernel machine."],
    [4, 2, "A normalised RBF readout is Nadaraya–Watson smoothing: a softmax-weighted average of values with a distance score."],
    [5, 7, "A Takagi–Sugeno rule base is a soft-gated mixture of local linear experts."],
    [6, 8, "The Kalman gain preconditions the gradient by curvature and uncertainty — the job of natural-gradient and Kronecker-factored optimizers."],
    [7, 9, "NARX models on tapped delay lines are recurrent sequence models; linear dynamical systems are the core of state-space layers."],
    [7, 10, "Learn a forward model of a plant, then plan or control through it: the loop of world models and model-predictive control."],
    [8, 11, "Evolving or growing architectures with a fitness signal is architecture search; population-based training evolves hyperparameters."],
    [9, 12, "Fixed-point MAC arrays and systolic dataflow, designed for neural nets decades ago, are what tensor cores, TPUs and int8/fp8 inference use."]
  ];
  const W = 760, H = 470, lx = 18, rx = 500, bw = 236, top = 14;
  const ly = i => top + i * (H - 2 * top) / left.length + 18;
  const ry = i => top + i * (H - 2 * top) / right.length + 15;
  const gL = svg.append("g"), gN = svg.append("g");
  const paths = gL.selectAll("path").data(links).join("path")
    .attr("d", d => {
      const x0 = lx + bw, y0 = ly(d[0]), x1 = rx, y1 = ry(d[1]);
      return "M" + x0 + "," + y0 + " C" + (x0 + 90) + "," + y0 + " " + (x1 - 90) + "," + y1 + " " + x1 + "," + y1;
    })
    .attr("fill", "none").attr("stroke", CX.accent).attr("stroke-opacity", 0.35).attr("stroke-width", 1.4);
  function node(g, x, y, text, href, side, i) {
    const a = g.append("a").attr("href", href);
    const grp = a.append("g").attr("class", "ln-node").attr("data-side", side).attr("data-i", i).style("cursor", "pointer");
    grp.append("rect").attr("x", x).attr("y", y - 12).attr("width", bw).attr("height", 24).attr("rx", 6)
      .attr("fill", CX.panel2).attr("stroke", side === "L" ? CX.orange : CX.accent).attr("stroke-opacity", 0.7);
    grp.append("text").attr("x", x + 10).attr("y", y + 4).attr("font-size", 11).attr("fill", CX.ink).text((side === "L" ? (i + 1) + " · " : "") + text);
    grp.on("mouseenter", () => focus(side, i)).on("focus", () => focus(side, i));
    return grp;
  }
  left.forEach((d, i) => node(gN, lx, ly(i), d[0], d[1], "L", i));
  right.forEach((d, i) => node(gN, rx, ry(i), d[0], d[1], "R", i));
  function focus(side, i) {
    const on = l => side === "L" ? l[0] === i : l[1] === i;
    paths.attr("stroke-opacity", l => on(l) ? 0.95 : 0.08).attr("stroke-width", l => on(l) ? 2.4 : 1.2)
      .attr("stroke", l => on(l) ? CX.orange : CX.accent);
    gN.selectAll("g.ln-node").attr("opacity", function () {
      const s = this.getAttribute("data-side"), k = +this.getAttribute("data-i");
      if (s === side && k === i) return 1;
      return links.some(l => on(l) && ((s === "L" && l[0] === k) || (s === "R" && l[1] === k))) ? 1 : 0.35;
    });
    const mine = links.filter(on);
    const name = side === "L" ? left[i][0] : right[i][0];
    document.getElementById("ln-readout").innerHTML = "<b>" + name + "</b><br>" + mine.map(l =>
      (side === "L" ? "→ <a href=\"" + right[l[1]][1] + "\">" + right[l[1]][0] + "</a>" : "← <a href=\"" + left[l[0]][1] + "\">" + left[l[0]][0] + "</a>") + ": " + l[2]).join("<br>");
  }
  svg.on("mouseleave", () => {
    paths.attr("stroke-opacity", 0.35).attr("stroke-width", 1.4).attr("stroke", CX.accent);
    gN.selectAll("g.ln-node").attr("opacity", 1);
  });
  focus("L", 1);
})();
