/* linked-stacks-queues.viz.js — figures for
   dsa/data-structures/linked-stacks-queues.html (part 2 of the Data Structures
   series).

   Loaded after ../../data.js, ../../notes.js and ../dsa-viz.js, so the globals
   AC (palette) and AL (rng / frame / axisB / axisL / gridY / row / bars /
   counter / stepper / …) are available.

   House rule obeyed throughout: every cost this page DISPLAYS — pointer
   dereferences, link writes, branch tests, cache-line touches, stack frames,
   element moves — is produced by running the real routine under AL.counter()
   and reading the counter back afterwards. Nothing below is a number typed
   into a caption. Wherever a closed form exists it is evaluated a second,
   independent way and printed next to the measurement, so every figure is its
   own cross-check: if the two disagree the figure says so, in red.

   Figures, in page order:
     01 #node-svg    node anatomy; insert / delete rewiring, stepped and counted
     02 #cost-svg    array shift vs list walk at every position, both measured
     03 #cache-svg   line touches: packed block vs scattered nodes, stated model
     04 #del-svg     the predecessor problem; the copy-successor trick and its bug
     05 #sent-svg    sentinel vs no sentinel: measured boundary branches
     06 #rev-svg     iterative reversal, three pointers, link writes counted
     07 #floyd-svg   Floyd's cycle detection + the entry-point phase, measured
     08 #stk-svg     array-backed vs node-backed stack: measured work per push
     09 #frame-svg   the call stack of a real recursion, depth measured
     10 #delim-svg   the balanced-delimiter checker, stepped
     11 #shunt-svg   infix to postfix: the operator stack, stepped
     12 #ring-svg    the circular buffer: wraparound, full-vs-empty, measured
     13 #two-svg     the two-stack queue: per-op spikes and the running average */

/* ── shared little helpers ─────────────────────────────────────────────── */
const LS = {
  table: function (sel, head, rows) {
    const h = d3.select(sel);
    if (h.empty()) return null;
    h.selectAll("*").remove();
    const t = h.append("table").attr("class", "cmp").style("margin", "10px 0 0");
    const hr = t.append("tr");
    head.forEach(c => hr.append("th").html(c));
    rows.forEach(r => {
      const tr = t.append("tr");
      r.forEach(c => tr.append("td").html(c));
    });
    return t;
  },
  int: d3.format(","),
  sig: function (x, d) { return (+x).toFixed(d === undefined ? 2 : d); },
  /* remove any stepper control bars a previous build left behind, so rebuilding
     a figure on a control change does not stack duplicate button rows */
  clearControls: function (svgNode) {
    svgNode.parentNode.querySelectorAll('div[role="group"]').forEach(el => el.remove());
  },
  /* an agree / disagree verdict, so a cross-check is visible rather than implied */
  verdict: function (ok) {
    return ok ? '<b style="color:' + AC.good + '">agree</b>'
              : '<b style="color:' + AC.bad + '">DISAGREE</b>';
  },

  /* ── the workhorse of this page: draw a chain of nodes ─────────────────
     Every figure that shows linked storage calls this. A node is drawn as a
     two-compartment box — the payload on the left, the link field on the
     right — because that split IS the definition the page opens with.

       slots : [{id, v, row, col, dim, fill, tag}]   fixed display positions
       links : [{from, to, color, dash, label}]      to === null ⇒ the null slash
       ptrs  : [{name, at, color, row}]              named pointers above a slot
     Returns {pos(id), linkPt(id), W, H} so a caller can hang its own marks. */
  chain: function (g, slots, links, ptrs, opt) {
    const o = Object.assign({
      x: 0, y: 0, w: 62, dataW: 38, h: 34, colGap: 26, rowGap: 74,
      nullMark: true, font: 13
    }, opt || {});
    const gg = g.append("g").attr("transform", "translate(" + o.x + "," + o.y + ")");
    const step = o.w + o.colGap;
    const byId = {};
    slots.forEach(s => { byId[s.id] = s; });
    const X = s => s.col * step;
    const Y = s => s.row * o.rowGap;
    const pos = id => { const s = byId[id]; return s ? { x: X(s), y: Y(s), w: o.w, h: o.h } : null; };
    /* the point an outgoing link leaves from: the centre of the link field */
    const linkPt = id => {
      const s = byId[id]; if (!s) return null;
      return { x: X(s) + o.dataW + (o.w - o.dataW) / 2, y: Y(s) + o.h / 2 };
    };
    const inPt = id => { const s = byId[id]; if (!s) return null; return { x: X(s), y: Y(s) + o.h / 2 }; };

    /* links first, so the boxes sit on top of the arrow tails */
    (links || []).forEach(L => {
      const a = linkPt(L.from); if (!a) return;
      const col = L.color || AC.muted;
      if (L.to === null || L.to === undefined || !byId[L.to]) {
        if (!o.nullMark) return;
        const s = byId[L.from];
        gg.append("line")
          .attr("x1", X(s) + o.dataW + 3).attr("y1", Y(s) + o.h - 3)
          .attr("x2", X(s) + o.w - 3).attr("y2", Y(s) + 3)
          .attr("stroke", col).attr("stroke-width", 1.6);
        return;
      }
      const b = inPt(L.to);
      const sameRow = byId[L.from].row === byId[L.to].row;
      const forward = b.x >= a.x;
      if (sameRow && forward && Math.abs(b.x - a.x) < step) {
        AL.arrow(gg, a.x, a.y, b.x - 2, b.y, { color: col, w: 1.7, dash: L.dash, head: 5.5 });
      } else {
        /* a curved link — the picture of an insert or a delete, where the chain
           order and the storage order have come apart */
        const mx = (a.x + b.x) / 2;
        const my = Math.min(a.y, b.y) - (sameRow ? 24 : 0) + (sameRow ? 0 : (b.y - a.y) / 2);
        const path = gg.append("path")
          .attr("d", "M" + a.x + "," + a.y + " Q" + mx + "," + my + " " + (b.x - 4) + "," + b.y)
          .attr("fill", "none").attr("stroke", col).attr("stroke-width", 1.7);
        if (L.dash) path.attr("stroke-dasharray", L.dash);
        const ang = Math.atan2(b.y - my, (b.x - 4) - mx);
        gg.append("path")
          .attr("d", "M" + (b.x - 2) + "," + b.y
            + " L" + ((b.x - 2) - 6 * Math.cos(ang - 0.4)) + "," + (b.y - 6 * Math.sin(ang - 0.4))
            + " L" + ((b.x - 2) - 6 * Math.cos(ang + 0.4)) + "," + (b.y - 6 * Math.sin(ang + 0.4)) + " Z")
          .attr("fill", col);
      }
      if (L.label) {
        gg.append("text").attr("x", (a.x + b.x) / 2).attr("y", Math.min(a.y, b.y) - 12)
          .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", col).text(L.label);
      }
    });

    /* the node boxes */
    slots.forEach(s => {
      const x = X(s), y = Y(s);
      const n = gg.append("g").attr("transform", "translate(" + x + "," + y + ")")
        .attr("opacity", s.dim ? 0.3 : 1);
      n.append("rect").attr("width", o.w).attr("height", o.h).attr("rx", 5)
        .attr("fill", s.fill || AC.panel2).attr("stroke", s.stroke || AC.line)
        .attr("stroke-width", s.stroke ? 2 : 1);
      n.append("line").attr("x1", o.dataW).attr("y1", 0).attr("x2", o.dataW).attr("y2", o.h)
        .attr("stroke", AC.line);
      n.append("text").attr("x", o.dataW / 2).attr("y", o.h / 2 + 4.5).attr("text-anchor", "middle")
        .attr("font-size", o.font).attr("fill", s.ink || AC.ink).text(s.v);
      if (s.tag) {
        n.append("text").attr("x", o.w / 2).attr("y", o.h + 13).attr("text-anchor", "middle")
          .attr("font-size", 10).attr("fill", s.tagColor || AC.muted).text(s.tag);
      }
    });

    /* named pointers, drawn as a label with an arrow down onto the node */
    (ptrs || []).forEach(p => {
      const s = byId[p.at];
      const col = p.color || AC.a2;
      if (!s) {                                   // a pointer holding null
        gg.append("text").attr("x", p.x === undefined ? -14 : p.x)
          .attr("y", (p.row === undefined ? 0 : p.row) * o.rowGap - 16)
          .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", col)
          .text(p.name + " = ∅");
        return;
      }
      const cx = X(s) + o.dataW / 2, top = Y(s);
      AL.arrow(gg, cx, top - 20, cx, top - 3, { color: col, w: 1.6, head: 5 });
      gg.append("text").attr("x", cx).attr("y", top - 25).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("fill", col).text(p.name);
    });

    return { g: gg, pos: pos, linkPt: linkPt, inPt: inPt, step: step, w: o.w, h: o.h,
             dataW: o.dataW, rowGap: o.rowGap };
  }
};

/* ══ FIGURE 01 ═══════════════════════════════════════════════════════════
   Node anatomy and the rewiring of a singly linked list. Each frame is a
   snapshot taken from INSIDE the real routine, so the pointer pictures and the
   step counts come from the same run. The cross-check is the closed form for
   the number of cursor advances — k for an insert after position k, k−1 for a
   delete at position k — evaluated separately and compared.               */
(function () {
  const svg = d3.select("#node-svg"); if (svg.empty()) return;
  const W = 720, H = 268;
  const VALUES = [31, 7, 19, 4, 23, 12];

  const els = {
    op: d3.select("#node-op"), k: d3.select("#node-k"),
    kOut: d3.select("#node-k-out"), out: d3.select("#node-readout")
  };

  /* ── the list, as real objects: payload + a reference, nothing else ──── */
  function mkList(values) {
    const nodes = values.map((v, i) => ({ id: "n" + i, v: v, next: null, col: i, row: 0 }));
    for (let i = 0; i < nodes.length - 1; i++) nodes[i].next = nodes[i + 1];
    return { head: nodes[0] || null, all: nodes.slice() };
  }

  /* a snapshot: walk the layout array (NOT the chain), so a node keeps its
     position on screen even after it is unlinked — which is the point */
  function snap(L, frames, note, hi, ptrs, counts) {
    const slots = L.all.map(nd => ({
      id: nd.id, v: nd.v, row: nd.row, col: nd.col,
      dim: !!nd.freed,
      fill: nd.fresh ? "#2a3550" : (nd.freed ? AC.panel : AC.panel2),
      stroke: hi && hi.indexOf(nd.id) >= 0 ? AC.a2 : null,
      tag: nd.freed ? "unreachable" : (nd.fresh ? "new node" : null),
      tagColor: nd.freed ? AC.bad : AC.good
    }));
    const links = L.all.filter(nd => !nd.freed || nd.showLink)
      .map(nd => ({ from: nd.id, to: nd.next ? nd.next.id : null,
                    color: nd.linkHot ? AC.good : AC.muted,
                    dash: nd.freed ? "3 3" : null }));
    frames.push({ slots: slots, links: links, note: note,
                  ptrs: (ptrs || []).map(p => Object.assign({}, p)),
                  counts: Object.assign({}, counts) });
  }

  function build(op, k) {
    const L = mkList(VALUES);
    const c = AL.counter();
    const frames = [];
    const snapshot = (note, hi, ptrs) => snap(L, frames, note, hi, ptrs, c.all());
    const HEAD = { name: "head", at: L.head.id, color: AC.accent };

    if (op === "insert-head") {
      snapshot("A node is a payload and a reference. head names the first node; the last node's link field is null.", [], [HEAD]);
      const nd = { id: "nx", v: 55, next: null, col: 0, row: 1, fresh: true };
      L.all.push(nd); c.add("alloc");
      snapshot("Allocate one node. Its link field is not yet set — nothing in the chain points at it and it points at nothing.", ["nx"], [HEAD]);
      nd.next = L.head; c.add("write"); nd.linkHot = true;
      snapshot("new.next = head — one field write. The new node now leads the old chain, but head still names the old first node.", ["nx"], [HEAD]);
      L.head = nd; c.add("write");
      snapshot("head = new — the second write. Two writes, zero traversal: this is the O(1) that a linked list buys and an array cannot.", ["nx"], [{ name: "head", at: "nx", color: AC.accent }]);

    } else if (op === "insert-k") {
      let cur = L.head;
      snapshot("Insert after position " + k + ". There is no address arithmetic, so the only way to reach position " + k + " is to walk to it.", [cur.id],
        [HEAD, { name: "cur", at: cur.id, color: AC.a2 }]);
      for (let j = 1; j <= k; j++) {
        cur = cur.next; c.add("step");
        snapshot("cur = cur.next — cursor advance " + j + " of " + k + ". Each advance is one link read; none of them can be skipped.",
          [cur.id], [HEAD, { name: "cur", at: cur.id, color: AC.a2 }]);
      }
      const nd = { id: "nx", v: 55, next: null, col: k + 0.5, row: 1, fresh: true };
      L.all.push(nd); c.add("alloc");
      snapshot("Allocate the node. Note where it sits: physically anywhere, logically between " + cur.v + " and " + (cur.next ? cur.next.v : "null") + ".",
        ["nx"], [HEAD, { name: "cur", at: cur.id, color: AC.a2 }]);
      nd.next = cur.next; c.add("write"); nd.linkHot = true;
      snapshot("new.next = cur.next — write 1. Do this one FIRST: the other order would drop the tail of the list on the floor.",
        ["nx"], [HEAD, { name: "cur", at: cur.id, color: AC.a2 }]);
      cur.next = nd; c.add("write"); cur.linkHot = true;
      snapshot("cur.next = new — write 2. The splice itself was " + 2 + " writes; the " + k + " cursor advance" + (k === 1 ? "" : "s") + " before it " + (k === 1 ? "was" : "were") + " the whole cost.",
        ["nx", cur.id], [HEAD, { name: "cur", at: cur.id, color: AC.a2 }]);

    } else if (op === "delete-head") {
      snapshot("Delete the first node. No predecessor is needed, because head IS the predecessor.", [L.head.id], [HEAD]);
      const old = L.head;
      L.head = L.head.next; c.add("write");
      old.showLink = true; old.freed = true;
      snapshot("head = head.next — one write. The old first node is now unreachable from head; whether its memory is reclaimed is the runtime's business, not the list's.",
        [], [{ name: "head", at: L.head.id, color: AC.accent }]);
      old.showLink = false;
      snapshot("The unlinked node's own link field still points into the list. That stale reference is harmless here and lethal if anyone keeps hold of it.",
        [], [{ name: "head", at: L.head.id, color: AC.accent }]);

    } else {                                  /* delete-k */
      let prev = L.head;
      snapshot("Delete the node at position " + k + ". A singly linked node cannot reach its predecessor, so the walk stops one node SHORT of the target.",
        [prev.id], [HEAD, { name: "prev", at: prev.id, color: AC.a2 }]);
      for (let j = 1; j <= k - 1; j++) {
        prev = prev.next; c.add("step");
        snapshot("prev = prev.next — advance " + j + " of " + (k - 1) + ". Stop when prev.next is the node to remove.",
          [prev.id], [HEAD, { name: "prev", at: prev.id, color: AC.a2 }]);
      }
      const target = prev.next;
      snapshot("prev.next is the target (" + target.v + "). Everything the deletion needs is now in hand.",
        [target.id], [HEAD, { name: "prev", at: prev.id, color: AC.a2 },
                      { name: "target", at: target.id, color: AC.bad }]);
      prev.next = target.next; c.add("write"); prev.linkHot = true;
      target.freed = true; target.showLink = true;
      snapshot("prev.next = target.next — one write, and the node is out. Nothing moved; nothing after it shifted. Compare an array, where removal at position " + k + " moves every later element.",
        [prev.id], [HEAD, { name: "prev", at: prev.id, color: AC.a2 }]);
    }
    return { frames: frames, c: c };
  }

  function draw() {
    const op = els.op.property("value");
    const k = +els.k.property("value");
    els.kOut.text(k);
    d3.select("#node-k-wrap").style("opacity", (op === "insert-k" || op === "delete-k") ? 1 : 0.35);

    const built = build(op, k);
    LS.clearControls(svg.node());

    const render = (fr, idx) => {
      const f = AL.frame(svg, W, H, { l: 30, r: 16, t: 16, b: 10 });
      const g = f.g;
      LS.chain(g, fr.slots, fr.links, fr.ptrs, { x: 8, y: 54, w: 62, dataW: 38, h: 34, colGap: 26, rowGap: 78 });

      g.append("text").attr("x", 0).attr("y", 10).attr("font-size", 11).attr("fill", AC.muted)
        .text("payload │ link  —  the two compartments of a node");

      /* the running measurement, top right */
      const cc = fr.counts;
      g.append("text").attr("x", f.iw).attr("y", 10).attr("text-anchor", "end")
        .attr("font-size", 11).attr("fill", AC.ink)
        .text("so far — advances: " + (cc.step || 0) + "   link writes: " + (cc.write || 0)
              + "   nodes allocated: " + (cc.alloc || 0));

      /* the step note, wrapped by hand across at most three lines */
      const words = fr.note.split(" ");
      const lines = []; let cur = "";
      words.forEach(wd => {
        if ((cur + " " + wd).length > 108) { lines.push(cur); cur = wd; }
        else cur = cur ? cur + " " + wd : wd;
      });
      if (cur) lines.push(cur);
      lines.slice(0, 3).forEach((ln, i) => {
        g.append("text").attr("x", 0).attr("y", 200 + i * 15).attr("font-size", 11.5)
          .attr("fill", i === 0 ? AC.ink : AC.muted).text(ln);
      });
    };

    AL.stepper(svg, { frames: built.frames, render: render, delay: 1500, label: "step" });

    /* independent cross-check of the traversal cost */
    const c = built.c;
    const predicted = op === "insert-head" ? 0 : op === "delete-head" ? 0
                    : op === "insert-k" ? k : k - 1;
    const predWrites = op === "insert-head" ? 2 : op === "insert-k" ? 2 : 1;
    const ok = (c.get("step") === predicted) && (c.get("write") === predWrites);
    const name = { "insert-head": "insert at the head", "insert-k": "insert after position " + k,
                   "delete-head": "delete the head", "delete-k": "delete at position " + k }[op];

    els.out.html(
      "<b>" + name + "</b> on a 6-node list — measured by running the routine: <b>"
      + c.get("step") + "</b> cursor advance" + (c.get("step") === 1 ? "" : "s") + ", <b>"
      + c.get("write") + "</b> link write" + (c.get("write") === 1 ? "" : "s") + ", <b>"
      + c.get("alloc") + "</b> allocation" + (c.get("alloc") === 1 ? "" : "s") + ". "
      + "The splice is the same two-or-one writes wherever it happens; only the walk to get there grows. "
      + "Cross-check against the closed form — advances should be <code>" + predicted
      + "</code> and writes <code>" + predWrites + "</code>: " + LS.verdict(ok) + "."
    );
  }

  [els.op, els.k].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══ FIGURE 02 ═══════════════════════════════════════════════════════════
   The cost contract, position by position. Neither series is the plot of a
   formula: for every k the real array routine and the real list routine are
   run under AL.counter() and the bar is what the counter returned. The two
   closed forms are then evaluated separately and the figure states whether
   they agree.                                                              */
(function () {
  const svg = d3.select("#cost-svg"); if (svg.empty()) return;
  const W = 720, H = 340;

  const els = {
    n: d3.select("#cost-n"), nOut: d3.select("#cost-n-out"),
    op: d3.select("#cost-op"), out: d3.select("#cost-readout")
  };

  /* ── the array routines, on a real block with one spare cell ─────────── */
  function arrInsert(A, n, k, v, c) {
    for (let j = n; j > k; j--) { c.add("touch"); A[j] = A[j - 1]; }
    A[k] = v; c.add("write");
    return n + 1;
  }
  function arrDelete(A, n, k, c) {
    for (let j = k; j < n - 1; j++) { c.add("touch"); A[j] = A[j + 1]; }
    return n - 1;
  }
  function arrRead(A, k, c) { c.add("write", 0); return A[k]; }   // address arithmetic, no element touched

  /* ── the list routines, on a real chain ──────────────────────────────── */
  function mkChain(n) {
    let head = null;
    for (let i = n - 1; i >= 0; i--) head = { v: i, next: head };
    return head;
  }
  function listInsert(head, k, v, c) {
    if (k === 0) { c.add("write", 2); return { v: v, next: head }; }
    let cur = head;
    for (let j = 1; j <= k; j++) { c.add("touch"); cur = cur.next; }
    const nd = { v: v, next: cur.next }; c.add("write");
    cur.next = nd; c.add("write");
    return head;
  }
  function listDelete(head, k, c) {
    if (k === 0) { c.add("write"); return head.next; }
    let prev = head;
    for (let j = 1; j <= k - 1; j++) { c.add("touch"); prev = prev.next; }
    prev.next = prev.next.next; c.add("write");
    return head;
  }
  function listRead(head, k, c) {
    let cur = head;
    for (let j = 1; j <= k; j++) { c.add("touch"); cur = cur.next; }
    return cur.v;
  }

  function draw() {
    const n = +els.n.property("value");
    const op = els.op.property("value");
    els.nOut.text(n);

    const arr = [], lst = [];
    for (let k = 0; k < n; k++) {
      const ca = AL.counter(), cl = AL.counter();
      const A = Array.from({ length: n + 1 }, (_, i) => i);
      const head = mkChain(n);
      if (op === "insert") { arrInsert(A, n, k, 99, ca); listInsert(head, k, 99, cl); }
      else if (op === "delete") { arrDelete(A, n, k, ca); listDelete(head, k, cl); }
      else { arrRead(A, k, ca); listRead(head, k, cl); }
      arr.push(ca.get("touch")); lst.push(cl.get("touch"));
    }

    /* independent closed forms, evaluated without touching the routines */
    const fArr = k => op === "insert" ? n - k : op === "delete" ? n - k - 1 : 0;
    const fLst = k => op === "insert" ? k : op === "delete" ? Math.max(0, k - 1) : k;
    let agree = true;
    for (let k = 0; k < n; k++) if (arr[k] !== fArr(k) || lst[k] !== fLst(k)) agree = false;

    const sum = a => a.reduce((s, x) => s + x, 0);
    const meanA = sum(arr) / n, meanL = sum(lst) / n;

    const f = AL.frame(svg, W, H, { l: 52, r: 150, t: 22, b: 46 });
    const g = f.g;
    const x = d3.scaleBand().domain(d3.range(n)).range([0, f.iw]).padding(0.22);
    const top = Math.max(1, d3.max([...arr, ...lst]));
    const y = d3.scaleLinear().domain([0, top]).nice().range([f.ih, 0]);

    AL.gridY(g, y, f.iw, 5);
    AL.axisB(g, d3.scaleLinear().domain([0, n - 1]).range([x(0) + x.bandwidth() / 2, x(n - 1) + x.bandwidth() / 2]),
      f.ih, Math.min(n, 12), "position k", d3.format("d"));
    AL.axisL(g, y, 5, "elements touched");

    const bw = x.bandwidth() / 2 - 1;
    for (let k = 0; k < n; k++) {
      g.append("rect").attr("x", x(k)).attr("y", y(arr[k])).attr("width", bw)
        .attr("height", f.ih - y(arr[k])).attr("fill", AC.accent).attr("rx", 1.5);
      g.append("rect").attr("x", x(k) + bw + 2).attr("y", y(lst[k])).attr("width", bw)
        .attr("height", f.ih - y(lst[k])).attr("fill", AC.a2).attr("rx", 1.5);
    }
    /* the crossover, where the two costs are equal */
    let cross = -1;
    for (let k = 0; k < n; k++) if (arr[k] <= lst[k]) { cross = k; break; }
    if (cross > 0) {
      g.append("line").attr("x1", x(cross) - 3).attr("x2", x(cross) - 3)
        .attr("y1", 0).attr("y2", f.ih).attr("stroke", AC.violet).attr("stroke-dasharray", "4 3");
      g.append("text").attr("x", x(cross) + 2).attr("y", 10).attr("font-size", 10)
        .attr("fill", AC.violet).text("k = " + cross);
    }

    AL.legend(g, [
      { label: "contiguous array — shift", color: AC.accent },
      { label: "linked chain — walk", color: AC.a2 },
      { label: "crossover", color: AC.violet, dash: "4 3" }
    ], f.iw + 14, 14);

    g.append("text").attr("x", f.iw + 14).attr("y", 76).attr("font-size", 11).attr("fill", AC.muted)
      .text("mean over all k");
    g.append("text").attr("x", f.iw + 14).attr("y", 92).attr("font-size", 12).attr("fill", AC.accent)
      .text("array  " + LS.sig(meanA, 2));
    g.append("text").attr("x", f.iw + 14).attr("y", 108).attr("font-size", 12).attr("fill", AC.a2)
      .text("list   " + LS.sig(meanL, 2));

    const label = { insert: "insert at k", delete: "delete at k", read: "read element k" }[op];
    els.out.html(
      "<b>" + label + ", n = " + n + "</b> — measured, not plotted from a formula. "
      + "The array touches <b>" + arr[0] + "</b> elements at k = 0 and <b>" + arr[n - 1]
      + "</b> at k = " + (n - 1) + "; the list touches <b>" + lst[0] + "</b> and <b>" + lst[n - 1]
      + "</b>. Means over all positions: array <b>" + LS.sig(meanA, 2) + "</b>, list <b>"
      + LS.sig(meanL, 2) + "</b> — "
      + (op === "read"
         ? "for reading, the array is flat at 0 touched elements at every k while the list rises linearly; there is no crossover and never will be."
         : "both are <span class='keep'>Θ</span>(n) on average, which is exactly why 'a linked list makes insertion fast' is false as stated: it is fast only once you are already there.")
      + " Cross-check against the closed forms <code>" + (op === "read" ? "0" : op === "insert" ? "n − k" : "n − k − 1")
      + "</code> and <code>" + (op === "read" ? "k" : op === "insert" ? "k" : "max(0, k − 1)")
      + "</code> at every k: " + LS.verdict(agree) + "."
    );
  }

  [els.n, els.op].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══ FIGURE 03 ═══════════════════════════════════════════════════════════
   What the RAM model cannot see. A stated, explicit machine model — 64-byte
   lines, a fully associative LRU cache of 64 lines, cold start — is simulated
   under AL.counter(), and the figure reports LINE FETCHES, not step counts.
   Two of the three curves have exact closed forms (a packed traversal touches
   ceil(bytes / 64) lines) and those are evaluated independently as a check. */
(function () {
  const svg = d3.select("#cache-svg"); if (svg.empty()) return;
  const W = 720, H = 350;
  const LINE = 64, CACHE_LINES = 64, LINKW = 8, GRAIN = 16;

  const els = {
    p: d3.select("#cache-p"), n: d3.select("#cache-n"),
    nOut: d3.select("#cache-n-out"), out: d3.select("#cache-readout"),
    tbl: d3.select("#cache-table")
  };

  /* a fully associative LRU cache of CACHE_LINES lines; touch() bumps the
     counter only when the line was not resident */
  function mkCache(c) {
    const resident = new Map();          // line -> recency stamp
    let clock = 0;
    return {
      touch: function (addr, bytes) {
        const first = Math.floor(addr / LINE), last = Math.floor((addr + bytes - 1) / LINE);
        for (let L = first; L <= last; L++) {
          if (!resident.has(L)) {
            c.add("fetch");
            if (resident.size >= CACHE_LINES) {         // evict least recently used
              let victim = null, best = Infinity;
              resident.forEach((t, k) => { if (t < best) { best = t; victim = k; } });
              resident.delete(victim);
            }
          }
          resident.set(L, clock++);
        }
      }
    };
  }

  const roundUp = (x, g) => Math.ceil(x / g) * g;

  /* traversal of a packed block: element i lives at i × p */
  function scanArray(n, p) {
    const c = AL.counter(), cache = mkCache(c);
    for (let i = 0; i < n; i++) cache.touch(i * p, p);
    return c.get("fetch");
  }
  /* traversal of a chain whose nodes sit at the addresses `addr` gives, visited
     in chain order 0, 1, …, n−1 — the node is read whole (payload + link) */
  function scanChain(n, p, addrOf) {
    const s = roundUp(p + LINKW, GRAIN);
    const c = AL.counter(), cache = mkCache(c);
    for (let i = 0; i < n; i++) cache.touch(addrOf(i, s), s);
    return { fetches: c.get("fetch"), nodeBytes: s };
  }

  function draw() {
    const p = +els.p.property("value");
    const nSel = +els.n.property("value");
    els.nOut.text(nSel);

    const NS = [];
    for (let n = 8; n <= 512; n += 8) NS.push(n);

    const r = AL.rng(20260916);
    const PERM = AL.perm(512, r).map(x => x - 1);      // a fixed scatter, same for every n

    const series = { arr: [], seq: [], scat: [] };
    let nodeBytes = 0;
    NS.forEach(n => {
      /* the scatter is a permutation of the list's OWN n slots — the nodes occupy
         exactly the bytes they need, only the visiting order is shuffled. That is
         the conservative version of the claim; a real heap does worse. */
      const permN = PERM.filter(v => v < n);
      series.arr.push(scanArray(n, p));
      const a = scanChain(n, p, (i, s) => i * s);                    // allocated in visit order
      const b = scanChain(n, p, (i, s) => permN[i] * s);             // same bytes, visited scattered
      nodeBytes = a.nodeBytes;
      series.seq.push(a.fetches);
      series.scat.push(b.fetches);
    });

    /* independent closed forms for the two contiguous cases, and two-sided
       bracketing for the scattered one, which has no closed form. The upper
       bracket is the total number of lines the n nodes SPAN counted with
       multiplicity — a node whose bytes straddle a line boundary costs two,
       which is why the bound is not simply n. */
    const predArr = n => Math.ceil(n * p / LINE);
    const nodeB = roundUp(p + LINKW, GRAIN);
    const predSeq = n => Math.ceil(n * nodeB / LINE);
    const spanTotal = n => {
      let t = 0;
      for (let i = 0; i < n; i++)
        t += Math.floor((i * nodeB + nodeB - 1) / LINE) - Math.floor(i * nodeB / LINE) + 1;
      return t;
    };
    let agree = true;
    NS.forEach((n, i) => {
      if (series.arr[i] !== predArr(n)) agree = false;
      if (series.seq[i] !== predSeq(n)) agree = false;
      if (series.scat[i] < predSeq(n) || series.scat[i] > spanTotal(n)) agree = false;
    });

    const f = AL.frame(svg, W, H, { l: 56, r: 168, t: 20, b: 46 });
    const g = f.g;
    const x = d3.scaleLinear().domain([0, 512]).range([0, f.iw]);
    const ymax = d3.max(series.scat);
    const y = d3.scaleLinear().domain([0, ymax]).nice().range([f.ih, 0]);
    AL.gridY(g, y, f.iw, 6);
    AL.axisB(g, x, f.ih, 8, "n — elements traversed", d3.format("d"));
    AL.axisL(g, y, 6, "64-byte line fetches");

    const line = d3.line().x((d, i) => x(NS[i])).y(d => y(d));
    const curves = [
      { k: "arr", col: AC.accent, lbl: "packed array, " + p + " B/element" },
      { k: "seq", col: AC.teal, lbl: "nodes, allocated in order" },
      { k: "scat", col: AC.bad, lbl: "nodes, shuffled visit order" }
    ];
    curves.forEach(cu => {
      g.append("path").datum(series[cu.k]).attr("fill", "none")
        .attr("stroke", cu.col).attr("stroke-width", 2.2).attr("d", line);
    });

    /* where the chain's own footprint stops fitting in the modelled cache —
       below this n, scatter is free; above it, scatter is the whole story */
    const nFit = Math.floor(CACHE_LINES * LINE / nodeBytes);
    if (nFit < 512) {
      g.append("line").attr("x1", x(nFit)).attr("x2", x(nFit)).attr("y1", 0).attr("y2", f.ih)
        .attr("stroke", AC.a2).attr("stroke-width", 1.2).attr("stroke-dasharray", "2 4");
      g.append("text").attr("x", x(nFit) + 4).attr("y", 24).attr("font-size", 10)
        .attr("fill", AC.a2).text("chain stops fitting in cache — n = " + nFit);
    }

    const idx = NS.indexOf(nSel);
    g.append("line").attr("x1", x(nSel)).attr("x2", x(nSel)).attr("y1", 0).attr("y2", f.ih)
      .attr("stroke", AC.violet).attr("stroke-dasharray", "4 3");
    curves.forEach(cu => {
      g.append("circle").attr("cx", x(nSel)).attr("cy", y(series[cu.k][idx])).attr("r", 4)
        .attr("fill", cu.col).attr("stroke", AC.bg).attr("stroke-width", 1.5);
    });

    AL.legend(g, curves.map(cu => ({ label: cu.lbl, color: cu.col })), f.iw + 14, 16);
    g.append("text").attr("x", f.iw + 14).attr("y", 72).attr("font-size", 11).attr("fill", AC.muted)
      .text("the model, stated");
    ["64-byte lines", "64 lines, fully assoc.", "LRU, cold start", "node = " + p + " + " + LINKW + " B",
     "→ " + nodeBytes + " B after 16-B grain"].forEach((t, i) => {
      g.append("text").attr("x", f.iw + 14).attr("y", 88 + i * 14).attr("font-size", 10.5)
        .attr("fill", AC.muted).text(t);
    });

    const a = series.arr[idx], sq = series.seq[idx], sc = series.scat[idx];
    els.out.html(
      "At <b>n = " + nSel + "</b>, " + p + "-byte payloads, under the model stated in the panel: the packed array fetches <b>"
      + LS.int(a) + "</b> lines, the same data as nodes allocated in visit order fetches <b>" + LS.int(sq)
      + "</b> (<b>" + LS.sig(sq / a, 2) + "×</b>), and as scattered nodes <b>" + LS.int(sc)
      + "</b> (<b>" + LS.sig(sc / a, 2) + "×</b>). All three traversals are <span class='keep'>Θ</span>(n) and they are not the same cost. "
      + (sc === sq
         ? "At this n the whole chain still fits in the modelled cache, so shuffling the visit order costs <b>nothing</b> — the honest caveat, and worth stating with its condition attached: the scatter here is a shuffled visit order over nodes that still occupy one compact run, so once that run is resident the order cannot matter. Nodes genuinely strewn one-per-line across a heap pay at every n. Push n past " + nFit + " and the two curves separate. "
         : "Past n = " + nFit + " the chain no longer fits in the modelled cache, and the shuffled visit order starts re-fetching lines it has already evicted — which is where the two node curves separate. ")
      + "Cross-check — a packed scan must fetch exactly <code>ceil(n × bytes-per-element / 64)</code> lines; evaluated separately "
      + "for both contiguous cases, and the scattered count bracketed between that figure and the total number of lines the n nodes span, at all "
      + NS.length + " values of n: " + LS.verdict(agree) + "."
    );

    LS.table("#cache-table",
      ["layout", "bytes per element", "line fetches at n = " + nSel, "vs packed array", "closed form"],
      [
        ["packed array", p + " B", LS.int(a), "1.00×", "ceil(n·" + p + "/64) = " + LS.int(predArr(nSel))],
        ["nodes, allocated in order", nodeBytes + " B", LS.int(sq), LS.sig(sq / a, 2) + "×",
         "ceil(n·" + nodeBytes + "/64) = " + LS.int(predSeq(nSel))],
        ["nodes, shuffled visit order", nodeBytes + " B", LS.int(sc), LS.sig(sc / a, 2) + "×",
         "bracketed: " + LS.int(predSeq(nSel)) + " ≤ fetches ≤ " + LS.int(spanTotal(nSel))]
      ]);
  }

  [els.p, els.n].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══ FIGURE 04 ═══════════════════════════════════════════════════════════
   The predecessor problem, and the O(1) trick that dodges it. Three modes,
   all stepped from inside the real routine. The third mode is the one worth
   having a figure for: it runs the trick at the tail, where it cannot work,
   and reports the failure rather than describing it.                       */
(function () {
  const svg = d3.select("#del-svg"); if (svg.empty()) return;
  const W = 720, H = 268;
  const VALUES = [31, 7, 19, 4, 23, 12];

  const els = {
    mode: d3.select("#del-mode"), k: d3.select("#del-k"),
    kOut: d3.select("#del-k-out"), kWrap: d3.select("#del-k-wrap"),
    out: d3.select("#del-readout")
  };

  function mkList(values) {
    const nodes = values.map((v, i) => ({ id: "n" + i, v: v, next: null, col: i, row: 0 }));
    for (let i = 0; i < nodes.length - 1; i++) nodes[i].next = nodes[i + 1];
    return { head: nodes[0], all: nodes };
  }
  function snap(L, frames, note, hi, ptrs, counts, bad) {
    frames.push({
      slots: L.all.map(nd => ({
        id: nd.id, v: nd.v, row: nd.row, col: nd.col, dim: !!nd.freed,
        fill: nd.copied ? "#2a3550" : (nd.freed ? AC.panel : AC.panel2),
        stroke: hi && hi.indexOf(nd.id) >= 0 ? (bad ? AC.bad : AC.a2) : null,
        tag: nd.freed ? "unlinked" : (nd.copied ? "payload overwritten" : null),
        tagColor: nd.freed ? AC.bad : AC.good
      })),
      links: L.all.filter(nd => !nd.freed || nd.showLink).map(nd => ({
        from: nd.id, to: nd.next ? nd.next.id : null,
        color: nd.linkHot ? AC.good : AC.muted, dash: nd.freed ? "3 3" : null
      })),
      note: note, ptrs: (ptrs || []).map(p => Object.assign({}, p)),
      counts: Object.assign({}, counts), bad: !!bad
    });
  }

  function build(mode, k) {
    const L = mkList(VALUES);
    const c = AL.counter();
    const frames = [];
    const S = (note, hi, ptrs, bad) => snap(L, frames, note, hi, ptrs, c.all(), bad);
    const HEAD = { name: "head", at: "n0", color: AC.accent };
    let ok = true, why = "the cost is the search, not the splice: the write itself is O(1) and the walk in front of it is not.";

    if (mode === "walk") {
      const target = L.all[k];
      S("You are handed a reference to the node holding " + target.v + " and told to remove it. A singly linked node has no way to name whoever points at it.",
        [target.id], [HEAD, { name: "target", at: target.id, color: AC.bad }]);
      let prev = L.head;
      S("So start again from the head and walk until prev.next is the target. This walk is the entire cost of the operation.",
        [prev.id], [HEAD, { name: "prev", at: prev.id, color: AC.a2 }, { name: "target", at: target.id, color: AC.bad }]);
      while (prev.next !== target) {
        prev = prev.next; c.add("step");
        S("prev = prev.next — advance " + c.get("step") + ". Still not the predecessor.",
          [prev.id], [HEAD, { name: "prev", at: prev.id, color: AC.a2 }, { name: "target", at: target.id, color: AC.bad }]);
      }
      S("Found it: prev.next === target after " + c.get("step") + " advance" + (c.get("step") === 1 ? "" : "s") + ".",
        [prev.id, target.id], [HEAD, { name: "prev", at: prev.id, color: AC.a2 }, { name: "target", at: target.id, color: AC.bad }]);
      prev.next = target.next; c.add("write"); prev.linkHot = true;
      target.freed = true; target.showLink = true;
      S("prev.next = target.next — one write and it is out. Correct, and Θ(k): the splice was O(1), the search for the predecessor was not.",
        [prev.id], [HEAD, { name: "prev", at: prev.id, color: AC.a2 }]);

    } else if (mode === "copy") {
      const target = L.all[k], succ = target.next;
      S("Same problem, no walk allowed. The trick: do not remove this node — remove the NEXT one, after stealing its payload.",
        [target.id], [HEAD, { name: "target", at: target.id, color: AC.bad }]);
      S("The successor holds " + succ.v + ". Both nodes are reachable from target alone, which is the whole point.",
        [target.id, succ.id], [HEAD, { name: "target", at: target.id, color: AC.bad },
                               { name: "succ", at: succ.id, color: AC.a2 }]);
      target.v = succ.v; c.add("copy"); target.copied = true;
      S("target.data = target.next.data — one payload copy. The node holding " + VALUES[k] + " now holds " + succ.v + ", so the VALUE " + VALUES[k] + " is gone from the list.",
        [target.id], [HEAD, { name: "target", at: target.id, color: AC.bad },
                      { name: "succ", at: succ.id, color: AC.a2 }]);
      target.next = succ.next; c.add("write"); target.linkHot = true;
      succ.freed = true; succ.showLink = true;
      S("target.next = target.next.next — one link write, and the successor node is unlinked. Total: O(1), no walk, at any position.",
        [target.id], [HEAD, { name: "target", at: target.id, color: AC.bad }]);
      why = "the sequence of VALUES is right, but the node OBJECT holding the deleted value survived and a different node died — any outside reference to the successor is now stale or aliased.";

    } else {                                   /* tail — the bug */
      const target = L.all[VALUES.length - 1];
      S("The same trick, applied to the LAST node. It holds " + target.v + " and its link field is null.",
        [target.id], [HEAD, { name: "target", at: target.id, color: AC.bad }], true);
      S("Step one wants target.next.data. But target.next is null — there is no successor to steal a payload from.",
        [target.id], [HEAD, { name: "target", at: target.id, color: AC.bad }], true);
      c.add("fault");
      S("The routine faults. This is not an edge case you can patch inside the trick: the trick works by sacrificing the successor, and the tail has none.",
        [target.id], [HEAD, { name: "target", at: target.id, color: AC.bad }], true);
      S("The only repairs are structural — walk to the predecessor after all (Θ(n)), keep a back-pointer (§06), or give the list a trailer sentinel (§07) so no real node is ever last.",
        [target.id], [HEAD, { name: "target", at: target.id, color: AC.bad }], true);
      ok = false;
      why = "the trick has no defined behaviour at the tail, so a list that uses it must guarantee by construction that the tail is never the target.";
    }
    return { frames: frames, c: c, ok: ok, why: why };
  }

  function draw() {
    const mode = els.mode.property("value");
    let k = +els.k.property("value");
    if (mode === "copy" && k > VALUES.length - 2) k = VALUES.length - 2;
    els.kOut.text(mode === "tail" ? VALUES.length - 1 : k);
    els.kWrap.style("opacity", mode === "tail" ? 0.35 : 1);

    const built = build(mode, k);
    LS.clearControls(svg.node());

    const render = (fr) => {
      const f = AL.frame(svg, W, H, { l: 30, r: 16, t: 16, b: 10 });
      const g = f.g;
      LS.chain(g, fr.slots, fr.links, fr.ptrs, { x: 8, y: 62, w: 62, dataW: 38, h: 34, colGap: 26, rowGap: 78 });
      const cc = fr.counts;
      g.append("text").attr("x", 0).attr("y", 10).attr("font-size", 11).attr("fill", AC.muted)
        .text(fr.bad ? "removing the tail with the copy-the-successor trick" : "removing one node, given only a reference to it");
      g.append("text").attr("x", f.iw).attr("y", 10).attr("text-anchor", "end").attr("font-size", 11)
        .attr("fill", fr.bad ? AC.bad : AC.ink)
        .text("advances: " + (cc.step || 0) + "   link writes: " + (cc.write || 0)
              + "   payload copies: " + (cc.copy || 0) + (cc.fault ? "   FAULT" : ""));
      const words = fr.note.split(" ");
      const lines = []; let cur = "";
      words.forEach(wd => {
        if ((cur + " " + wd).length > 106) { lines.push(cur); cur = wd; }
        else cur = cur ? cur + " " + wd : wd;
      });
      if (cur) lines.push(cur);
      lines.slice(0, 3).forEach((ln, i) => {
        g.append("text").attr("x", 0).attr("y", 202 + i * 15).attr("font-size", 11.5)
          .attr("fill", i === 0 ? (fr.bad ? AC.bad : AC.ink) : AC.muted).text(ln);
      });
    };
    AL.stepper(svg, { frames: built.frames, render: render, delay: 1600, label: "step" });

    const c = built.c;
    /* cross-check: starting at the head (position 0), reaching the target's
       predecessor at position k−1 must take exactly k−1 advances */
    const predSteps = mode === "walk" ? k - 1 : 0;
    const okCount = (c.get("step") === predSteps);
    const names = { walk: "walk to the predecessor", copy: "copy the successor, interior node",
                    tail: "copy the successor, at the tail" };
    els.out.html(
      "<b>" + names[mode] + "</b> — measured: <b>" + c.get("step") + "</b> cursor advance"
      + (c.get("step") === 1 ? "" : "s") + ", <b>" + c.get("write") + "</b> link write"
      + (c.get("write") === 1 ? "" : "s") + ", <b>" + c.get("copy") + "</b> payload cop"
      + (c.get("copy") === 1 ? "y" : "ies") + (c.get("fault") ? ", and <b style='color:" + AC.bad + "'>1 fault</b>" : "")
      + ". " + (built.ok ? (mode === "walk" ? "Correct and identity-preserving — " : "The catch is that ") + built.why
                          : "It does not complete: " + built.why)
      + " Cross-check — the predecessor walk from the head to position " + (k - 1) + " must take exactly <code>"
      + predSteps + "</code> advances: " + LS.verdict(okCount) + "."
    );
  }

  [els.mode, els.k].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══ FIGURE 05 ═══════════════════════════════════════════════════════════
   Sentinels, priced. Two REAL doubly linked implementations — one with null
   ends and head/tail pointers, one with a header and a trailer node — are run
   against the same operations on the same list state, and every boundary test
   either of them performs bumps a counter. The bars are what the counters
   returned; the accompanying write counts are there so the figure states the
   cost of sentinels as well as the benefit.                                 */
(function () {
  const svg = d3.select("#sent-svg"); if (svg.empty()) return;
  const W = 720, H = 372;

  const els = {
    state: d3.select("#sent-state"), out: d3.select("#sent-readout"),
    tbl: d3.select("#sent-table")
  };

  /* ── implementation A: null ends, head and tail pointers ─────────────── */
  function Plain(vals) {
    const L = { head: null, tail: null };
    vals.forEach(v => {                       // built without instrumentation
      const nd = { v: v, prev: L.tail, next: null };
      if (L.tail === null) L.head = nd; else L.tail.next = nd;
      L.tail = nd;
    });
    L.pushFront = (v, c) => {
      const nd = { v: v, prev: null, next: L.head }; c.add("write", 2);
      c.add("branch");                        // is the list empty?
      if (L.head === null) { L.tail = nd; c.add("write"); }
      else { L.head.prev = nd; c.add("write"); }
      L.head = nd; c.add("write");
    };
    L.pushBack = (v, c) => {
      const nd = { v: v, prev: L.tail, next: null }; c.add("write", 2);
      c.add("branch");
      if (L.tail === null) { L.head = nd; c.add("write"); }
      else { L.tail.next = nd; c.add("write"); }
      L.tail = nd; c.add("write");
    };
    L.popFront = c => {
      c.add("precond"); if (L.head === null) return null;   // empty?
      const nd = L.head;
      L.head = nd.next; c.add("write");
      c.add("branch");                        // did the list just become empty?
      if (L.head === null) { L.tail = null; c.add("write"); }
      else { L.head.prev = null; c.add("write"); }
      return nd.v;
    };
    L.popBack = c => {
      c.add("precond"); if (L.tail === null) return null;
      const nd = L.tail;
      L.tail = nd.prev; c.add("write");
      c.add("branch");
      if (L.tail === null) { L.head = null; c.add("write"); }
      else { L.tail.next = null; c.add("write"); }
      return nd.v;
    };
    L.remove = (x, c) => {
      c.add("branch");                        // is x the first node?
      if (x.prev === null) { L.head = x.next; c.add("write"); }
      else { x.prev.next = x.next; c.add("write"); }
      c.add("branch");                        // is x the last node?
      if (x.next === null) { L.tail = x.prev; c.add("write"); }
      else { x.next.prev = x.prev; c.add("write"); }
      return x.v;
    };
    L.insertAfter = (x, v, c) => {
      const nd = { v: v, prev: x, next: x.next }; c.add("write", 2);
      c.add("branch");                        // is x the last node?
      if (x.next === null) { L.tail = nd; c.add("write"); }
      else { x.next.prev = nd; c.add("write"); }
      x.next = nd; c.add("write");
    };
    L.mid = () => { let m = L.head; if (!m) return null; let k = 0, p = L.head;
      while (p) { k++; p = p.next; } for (let i = 0; i < Math.floor(k / 2); i++) m = m.next; return m; };
    return L;
  }

  /* ── implementation B: a header and a trailer node that always exist ──── */
  function Sent(vals) {
    const H = { v: "H", prev: null, next: null }, T = { v: "T", prev: null, next: null };
    H.next = T; T.prev = H;
    const L = { H: H, T: T };
    const raw = (p, q, v) => { const nd = { v: v, prev: p, next: q }; p.next = nd; q.prev = nd; return nd; };
    vals.forEach(v => raw(T.prev, T, v));
    /* the ONLY splice either operation needs — no case analysis at all */
    L.insertBetween = (p, q, v, c) => {
      const nd = { v: v, prev: p, next: q }; c.add("write", 2);
      p.next = nd; c.add("write");
      q.prev = nd; c.add("write");
      return nd;
    };
    L.pushFront = (v, c) => L.insertBetween(H, H.next, v, c);
    L.pushBack = (v, c) => L.insertBetween(T.prev, T, v, c);
    L.remove = (x, c) => {
      x.prev.next = x.next; c.add("write");
      x.next.prev = x.prev; c.add("write");
      return x.v;
    };
    L.popFront = c => { c.add("precond"); if (H.next === T) return null; return L.remove(H.next, c); };
    L.popBack = c => { c.add("precond"); if (T.prev === H) return null; return L.remove(T.prev, c); };
    L.insertAfter = (x, v, c) => L.insertBetween(x, x.next, v, c);
    L.mid = () => { let k = 0, p = H.next; while (p !== T) { k++; p = p.next; }
      if (k === 0) return null; let m = H.next;
      for (let i = 0; i < Math.floor(k / 2); i++) m = m.next; return m; };
    return L;
  }

  const STATES = { empty: [], one: [31], six: [31, 7, 19, 4, 23, 12] };

  function measure(vals) {
    const OPS = [
      { k: "push front", run: (L, c) => L.pushFront(99, c), needs: 0 },
      { k: "push back", run: (L, c) => L.pushBack(99, c), needs: 0 },
      { k: "pop front", run: (L, c) => L.popFront(c), needs: 0 },
      { k: "pop back", run: (L, c) => L.popBack(c), needs: 0 },
      { k: "remove a held node", run: (L, c) => { const m = L.mid(); if (m) L.remove(m, c); }, needs: 1 },
      { k: "insert after a node", run: (L, c) => { const m = L.mid(); if (m) L.insertAfter(m, 99, c); }, needs: 1 }
    ];
    return OPS.filter(o => vals.length >= o.needs).map(o => {
      const ca = AL.counter(), cb = AL.counter();
      o.run(Plain(vals), ca);
      o.run(Sent(vals), cb);
      return { k: o.k, a: ca.all(), b: cb.all() };
    });
  }

  /* a small doubly linked picture: forward links above, back links below */
  function dchain(g, labels, opt) {
    const o = Object.assign({ x: 0, y: 0, w: 46, h: 26, gap: 30, title: "", sent: [] }, opt || {});
    const gg = g.append("g").attr("transform", "translate(" + o.x + "," + o.y + ")");
    if (o.title) gg.append("text").attr("x", 0).attr("y", -20).attr("font-size", 11)
      .attr("fill", AC.muted).text(o.title);
    const step = o.w + o.gap;
    labels.forEach((L, i) => {
      const isS = o.sent.indexOf(i) >= 0;
      const isNull = L === "∅";
      if (isNull) {
        gg.append("text").attr("x", i * step + o.w / 2).attr("y", o.h / 2 + 4)
          .attr("text-anchor", "middle").attr("font-size", 12).attr("fill", AC.bad).text("null");
      } else {
        gg.append("rect").attr("x", i * step).attr("y", 0).attr("width", o.w).attr("height", o.h)
          .attr("rx", 4).attr("fill", isS ? AC.panel : AC.panel2)
          .attr("stroke", isS ? AC.violet : AC.line).attr("stroke-dasharray", isS ? "3 2" : null);
        gg.append("text").attr("x", i * step + o.w / 2).attr("y", o.h / 2 + 4.5)
          .attr("text-anchor", "middle").attr("font-size", 12)
          .attr("fill", isS ? AC.violet : AC.ink).text(L);
      }
      if (i < labels.length - 1) {
        const a = i * step + o.w + 2, b = (i + 1) * step - 2;
        AL.arrow(gg, a, o.h * 0.3, b, o.h * 0.3, { color: AC.muted, w: 1.4, head: 4.5 });
        AL.arrow(gg, b, o.h * 0.72, a, o.h * 0.72, { color: "#5a6375", w: 1.2, head: 4.5 });
      }
    });
    return gg;
  }

  function draw() {
    const st = els.state.property("value");
    const vals = STATES[st];
    const rows = measure(vals);

    const f = AL.frame(svg, W, H, { l: 52, r: 150, t: 18, b: 46 });
    const g = f.g;

    const show = vals.length > 3 ? vals.slice(0, 3).concat(["…"]) : vals.slice();
    dchain(g, ["head"].concat(show.map(String), ["∅"]), {
      x: 0, y: 24, w: 44, h: 24, gap: 26, title: "null ends: head and tail pointers, and a null at each end"
    });
    dchain(g, ["H"].concat(show.map(String), ["T"]), {
      x: 340, y: 24, w: 44, h: 24, gap: 26, sent: [0, show.length + 1],
      title: "sentinels: a header and a trailer node that hold no data"
    });

    const y0 = 96;
    const x = d3.scaleBand().domain(rows.map(r => r.k)).range([0, f.iw]).padding(0.25);
    const top = Math.max(1, d3.max(rows, r => Math.max(r.a.branch || 0, r.b.branch || 0,
      r.a.write || 0, r.b.write || 0)));
    const y = d3.scaleLinear().domain([0, top]).nice().range([f.ih - y0, 0]);
    const gb = g.append("g").attr("transform", "translate(0," + y0 + ")");
    AL.gridY(gb, y, f.iw, 4);
    gb.append("g").attr("class", "axis").attr("transform", "translate(0," + (f.ih - y0) + ")")
      .call(d3.axisBottom(x).tickSize(3))
      .selectAll("text").attr("font-size", 9.5).attr("transform", "rotate(-12)").attr("text-anchor", "end");
    AL.axisL(gb, y, 4, "operations");

    const bw = x.bandwidth() / 4 - 1;
    rows.forEach(r => {
      const bx = x(r.k);
      const items = [
        { v: r.a.branch || 0, col: AC.bad, i: 0 },
        { v: r.b.branch || 0, col: AC.good, i: 1 },
        { v: r.a.write || 0, col: "#5a6375", i: 2 },
        { v: r.b.write || 0, col: AC.violet, i: 3 }
      ];
      items.forEach(it => {
        gb.append("rect").attr("x", bx + it.i * (bw + 1)).attr("y", y(it.v))
          .attr("width", bw).attr("height", (f.ih - y0) - y(it.v)).attr("fill", it.col).attr("rx", 1.5);
        if (it.v === 0) gb.append("text").attr("x", bx + it.i * (bw + 1) + bw / 2)
          .attr("y", (f.ih - y0) - 3).attr("text-anchor", "middle").attr("font-size", 9)
          .attr("fill", it.col).text("0");
      });
    });

    AL.legend(g, [
      { label: "boundary branches — null ends", color: AC.bad },
      { label: "boundary branches — sentinels", color: AC.good },
      { label: "link writes — null ends", color: "#5a6375" },
      { label: "link writes — sentinels", color: AC.violet }
    ], f.iw + 12, y0 + 12);

    const sumA = rows.reduce((s, r) => s + (r.a.branch || 0), 0);
    const sumB = rows.reduce((s, r) => s + (r.b.branch || 0), 0);
    const wA = rows.reduce((s, r) => s + (r.a.write || 0), 0);
    const wB = rows.reduce((s, r) => s + (r.b.write || 0), 0);

    /* cross-check: the sentinel implementation contains no boundary test at all
       in its source, so its measured branch total must be exactly zero */
    const ok = (sumB === 0);

    const stName = { empty: "an empty list", one: "a one-element list", six: "a six-element list" }[st];
    els.out.html(
      "Over the " + rows.length + " operations applicable to <b>" + stName + "</b>, measured by running both real implementations: "
      + "the null-ended version executes <b>" + sumA + "</b> boundary branches, the sentinel version <b>" + sumB
      + "</b>. Link writes over the same operations: <b>" + wA + "</b> against <b>" + wB + "</b>"
      + (wA === wB ? " — identical, so the branches were bought for nothing but two permanently allocated nodes per list."
                   : wB < wA ? " — the sentinel version does <i>fewer</i> writes as well, because it never has to repair a head or tail pointer."
                             : " — the sentinel version does " + (wB - wA) + " more, which is what the removed case analysis costs.")
      + " Sentinels do not change any asymptotic bound; they remove case analysis, and case analysis is where the bugs are. "
      + "Cross-check — the sentinel routines contain no null test anywhere in their source, so the measured branch count must be exactly 0: "
      + LS.verdict(ok) + "."
    );

    LS.table("#sent-table",
      ["operation", "branches, null ends", "branches, sentinels", "writes, null ends", "writes, sentinels"],
      rows.map(r => [r.k, String(r.a.branch || 0), String(r.b.branch || 0),
                     String(r.a.write || 0), String(r.b.write || 0)])
        .concat([["<b>total</b>", "<b>" + sumA + "</b>", "<b>" + sumB + "</b>",
                  "<b>" + wA + "</b>", "<b>" + wB + "</b>"]]));
  }

  els.state.on("input change", draw);
  draw();
})();

/* ══ FIGURE 06 ═══════════════════════════════════════════════════════════
   Reversal, both ways. The frames are snapshots taken from inside the real
   routines; the link writes, pointer assignments and — for the recursive
   version — the live stack depth are all read back off AL.counter(). The
   cross-check is that a correct reversal performs exactly n link writes and
   that the resulting order is the exact reverse of the input.              */
(function () {
  const svg = d3.select("#rev-svg"); if (svg.empty()) return;
  const W = 720, H = 300;
  const VALUES = [31, 7, 19, 4, 23, 12];

  const els = { mode: d3.select("#rev-mode"), out: d3.select("#rev-readout") };

  function mkList() {
    const nodes = VALUES.map((v, i) => ({ id: "n" + i, v: v, next: null, col: i, row: 0 }));
    for (let i = 0; i < nodes.length - 1; i++) nodes[i].next = nodes[i + 1];
    return { head: nodes[0], all: nodes };
  }

  function build(mode) {
    const L = mkList();
    const c = AL.counter();
    const frames = [];
    const S = (note, ptrs, depth, stack) => frames.push({
      slots: L.all.map(nd => ({
        id: nd.id, v: nd.v, row: nd.row, col: nd.col,
        fill: AC.panel2,
        stroke: (ptrs || []).some(p => p.at === nd.id && p.name === "cur") ? AC.a2 : null
      })),
      links: L.all.map(nd => ({ from: nd.id, to: nd.next ? nd.next.id : null,
                                color: nd.flipped ? AC.good : AC.muted })),
      note: note, ptrs: (ptrs || []).map(p => Object.assign({}, p)),
      counts: c.all(), depth: depth || 0, stack: (stack || []).slice()
    });

    if (mode === "iter") {
      let prev = null, cur = L.head;
      S("Three references and no extra memory: prev behind, cur at the node being flipped, and next saved so the flip cannot lose the rest of the list.",
        [{ name: "prev", at: null, color: AC.violet }, { name: "cur", at: cur.id, color: AC.a2 }]);
      while (cur !== null) {
        const nxt = cur.next; c.add("read"); c.add("assign");
        S("nxt = cur.next — save the rest of the list FIRST. The next line is about to destroy the only reference to it.",
          [prev ? { name: "prev", at: prev.id, color: AC.violet } : { name: "prev", at: null, color: AC.violet },
           { name: "cur", at: cur.id, color: AC.a2 },
           nxt ? { name: "nxt", at: nxt.id, color: AC.teal } : { name: "nxt", at: null, color: AC.teal }]);
        cur.next = prev; c.add("write"); cur.flipped = true;
        prev = cur; c.add("assign");
        cur = nxt; c.add("assign");
        S("cur.next = prev — the flip, one link write. Then prev and cur both step forward. Link writes so far: " + c.get("write") + ".",
          [{ name: "prev", at: prev.id, color: AC.violet },
           cur ? { name: "cur", at: cur.id, color: AC.a2 } : { name: "cur", at: null, color: AC.a2 }]);
      }
      L.head = prev; c.add("assign");
      S("cur is null, so every node has been flipped. head = prev. Total: " + c.get("write")
        + " link writes, " + c.get("assign") + " pointer assignments, O(1) extra space — nothing was allocated and nothing was copied.",
        [{ name: "head", at: L.head.id, color: AC.accent }]);

    } else {
      const stack = [];
      let maxDepth = 0;
      function rev(node) {
        stack.push(node.v); maxDepth = Math.max(maxDepth, stack.length);
        c.add("call");
        if (node.next === null) {
          S("Base case: node " + node.v + " has no successor, so it is the new head. This is the deepest frame — " + stack.length + " live calls.",
            [{ name: "node", at: node.id, color: AC.a2 }], stack.length, stack);
          stack.pop();
          return node;
        }
        S("Descend: reverse the rest of the list first, then fix up this node's link on the way back out. Live frames: " + stack.length + ".",
          [{ name: "node", at: node.id, color: AC.a2 }], stack.length, stack);
        const newHead = rev(node.next);        // returns with this frame still live
        node.next.next = node; c.add("write"); node.next.flipped = true;
        node.next = null; c.add("write"); node.flipped = true;
        S("Unwinding at " + node.v + ": node.next.next = node flips the link, node.next = null cuts the old one. Two writes per frame, and " + stack.length + " frames still live.",
          [{ name: "node", at: node.id, color: AC.a2 }], stack.length, stack);
        stack.pop();
        return newHead;
      }
      S("The recursive version keeps no explicit pointers. It keeps a stack frame per node instead — which is O(n) space, invisibly.",
        [{ name: "head", at: L.head.id, color: AC.accent }], 0, []);
      L.head = rev(L.head);
      c.add("maxdepth", maxDepth);
      S("Done. " + c.get("write") + " link writes — but the deepest point held " + maxDepth
        + " live stack frames, so this version is O(n) space where the iterative one is O(1).",
        [{ name: "head", at: L.head.id, color: AC.accent }], 0, []);
    }

    /* independent verification: read the finished list out and compare */
    const outOrder = [];
    let p = L.head, guard = 0;
    while (p && guard++ < 100) { outOrder.push(p.v); p = p.next; }
    const expected = VALUES.slice().reverse();
    const ok = outOrder.length === expected.length && outOrder.every((v, i) => v === expected[i]);
    return { frames: frames, c: c, out: outOrder, ok: ok };
  }

  function draw() {
    const mode = els.mode.property("value");
    const built = build(mode);
    LS.clearControls(svg.node());

    const render = (fr) => {
      const f = AL.frame(svg, W, H, { l: 30, r: 16, t: 16, b: 10 });
      const g = f.g;
      LS.chain(g, fr.slots, fr.links, fr.ptrs, { x: 8, y: 74, w: 62, dataW: 38, h: 34, colGap: 26, rowGap: 78 });
      const cc = fr.counts;
      g.append("text").attr("x", 0).attr("y", 10).attr("font-size", 11).attr("fill", AC.muted)
        .text(mode === "iter" ? "reverse in place — prev, cur, nxt" : "reverse recursively — one frame per node");
      g.append("text").attr("x", f.iw).attr("y", 10).attr("text-anchor", "end").attr("font-size", 11)
        .attr("fill", AC.ink)
        .text("link writes: " + (cc.write || 0) + "   pointer assignments: " + (cc.assign || 0)
              + (mode === "rec" ? "   calls: " + (cc.call || 0) : ""));

      /* the live stack, drawn as a column, for the recursive mode */
      if (mode === "rec") {
        const sx = f.iw - 90, sy = 132;
        g.append("text").attr("x", sx).attr("y", sy - 6).attr("font-size", 10).attr("fill", AC.muted)
          .text("live frames: " + fr.stack.length);
        fr.stack.slice().reverse().forEach((v, i) => {
          g.append("rect").attr("x", sx).attr("y", sy + i * 13).attr("width", 74).attr("height", 11)
            .attr("rx", 2).attr("fill", i === 0 ? AC.a2 : AC.panel2).attr("stroke", AC.line);
          g.append("text").attr("x", sx + 37).attr("y", sy + i * 13 + 9).attr("text-anchor", "middle")
            .attr("font-size", 9).attr("fill", i === 0 ? AC.bg : AC.muted).text("rev(node " + v + ")");
        });
      }

      const words = fr.note.split(" ");
      const lines = []; let cur = "";
      words.forEach(wd => {
        if ((cur + " " + wd).length > 104) { lines.push(cur); cur = wd; }
        else cur = cur ? cur + " " + wd : wd;
      });
      if (cur) lines.push(cur);
      lines.slice(0, 3).forEach((ln, i) => {
        g.append("text").attr("x", 0).attr("y", 228 + i * 15).attr("font-size", 11.5)
          .attr("fill", i === 0 ? AC.ink : AC.muted).text(ln);
      });
    };
    AL.stepper(svg, { frames: built.frames, render: render, delay: 1200, label: "step" });

    const c = built.c;
    const okWrites = mode === "iter" ? (c.get("write") === VALUES.length)
                                     : (c.get("write") === 2 * (VALUES.length - 1));
    els.out.html(
      "<b>" + (mode === "iter" ? "iterative reversal" : "recursive reversal") + "</b>, n = " + VALUES.length
      + " — measured: <b>" + c.get("write") + "</b> link writes"
      + (mode === "rec" ? ", <b>" + c.get("call") + "</b> calls, deepest stack <b>" + c.get("maxdepth")
         + "</b> live frames — O(n) space that no line of the source mentions"
         : ", <b>" + c.get("assign") + "</b> pointer assignments, <b>0</b> allocations and <b>O(1)</b> extra space")
      + ". Result read back off the finished list: <code>[" + built.out.join(", ") + "]</code>, "
      + "which " + LS.verdict(built.ok) + " with the input reversed. "
      + "Second check — " + (mode === "iter"
        ? "the iterative version must write exactly one link per node, n = " + VALUES.length
        : "the recursive version writes two links per non-base frame, 2(n − 1) = " + (2 * (VALUES.length - 1)))
      + ": " + LS.verdict(okWrites) + "."
    );
  }

  els.mode.on("input change", draw);
  draw();
})();

/* ══ FIGURE 07 ═══════════════════════════════════════════════════════════
   Floyd's cycle detection, both phases, on a rho-shaped list whose tail
   length mu and cycle length lam are yours to set. The step counts come from
   running the real two-pointer routine under AL.counter(); three independent
   predictions are then evaluated and compared — the meeting step must be the
   least positive multiple of the cycle length that is at least the tail
   length, phase two must take exactly the tail length in steps, and the node
   it lands on must be the entry node the generator actually built.         */
(function () {
  const svg = d3.select("#floyd-svg"); if (svg.empty()) return;
  const W = 720, H = 360;

  const els = {
    mu: d3.select("#floyd-mu"), muOut: d3.select("#floyd-mu-out"),
    lam: d3.select("#floyd-lam"), lamOut: d3.select("#floyd-lam-out"),
    out: d3.select("#floyd-readout")
  };

  function build(mu, lam) {
    const N = mu + lam;
    const next = i => (i === N - 1 ? mu : i + 1);
    const c = AL.counter();
    const frames = [];
    const S = (note, marks, phase) => frames.push({
      note: note, marks: marks, phase: phase, counts: c.all()
    });

    S("Two references start at the head. The slow one advances one link per round, the fast one two. If the list ends, the fast one falls off and there is no cycle; if it does not end, the fast one must eventually lap the slow one.",
      { slow: 0, fast: 0 }, 1);

    let slow = 0, fast = 0, round = 0;
    do {
      slow = next(slow); c.add("slow");
      fast = next(next(fast)); c.add("fast", 2);
      round++; c.add("round");
      S("Round " + round + ": slow is at node " + slow + ", fast at node " + fast
        + (slow === fast ? " — they have met, so the list contains a cycle." : ". The gap between them shrinks by exactly one link per round, so a meeting is unavoidable."),
        { slow: slow, fast: fast }, 1);
    } while (slow !== fast);
    const meet = slow;

    S("Phase 1 done: they met at node " + meet + " after " + round
      + " rounds. Note what this does NOT tell you — the meeting point is generally not the entry of the cycle.",
      { slow: meet, fast: meet }, 1);

    let p = 0, q = meet, steps = 0;
    S("Phase 2: put one reference back at the head, leave the other at the meeting point, and now advance BOTH one link at a time.",
      { p: p, q: q }, 2);
    while (p !== q) {
      p = next(p); q = next(q); steps++; c.add("phase2");
      S("Step " + steps + ": p at node " + p + ", q at node " + q
        + (p === q ? " — they have met, and this node is the entry of the cycle." : "."),
        { p: p, q: q }, 2);
    }
    S("They meet at node " + p + " after " + steps + " step" + (steps === 1 ? "" : "s")
      + ". That node is where the tail joins the cycle, and " + steps + " is the length of the tail.",
      { p: p, q: q, entry: p }, 2);

    /* three independent predictions */
    const predMeetRound = lam * Math.max(1, Math.ceil(mu / lam));
    const okMeet = (round === predMeetRound);
    const okPhase2 = (steps === mu);
    const okEntry = (p === mu);
    return { frames: frames, c: c, meet: meet, entry: p, steps: steps, round: round,
             predMeetRound: predMeetRound, ok: okMeet && okPhase2 && okEntry,
             okMeet: okMeet, okPhase2: okPhase2, okEntry: okEntry, N: N, next: next };
  }

  function draw() {
    const mu = +els.mu.property("value"), lam = +els.lam.property("value");
    els.muOut.text(mu); els.lamOut.text(lam);
    const built = build(mu, lam);
    LS.clearControls(svg.node());

    const N = built.N;
    const R = Math.max(42, 13 * lam);
    const cy = 150;
    const tailStep = 62;
    const cx = 34 + mu * tailStep + R;
    const pos = i => {
      if (i < mu) return { x: 34 + i * tailStep, y: cy };
      const j = i - mu, th = Math.PI - 2 * Math.PI * j / lam;
      return { x: cx + R * Math.cos(th), y: cy - R * Math.sin(th) };
    };

    const render = (fr) => {
      const f = AL.frame(svg, W, H, { l: 12, r: 12, t: 14, b: 10 });
      const g = f.g;

      g.append("text").attr("x", 0).attr("y", 8).attr("font-size", 11).attr("fill", AC.muted)
        .text("a rho-shaped list: a tail of " + mu + " node" + (mu === 1 ? "" : "s")
              + " running into a cycle of " + lam);

      /* edges */
      for (let i = 0; i < N; i++) {
        const a = pos(i), b = pos(built.next(i));
        const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1;
        AL.arrow(g, a.x + 15 * dx / d, a.y + 15 * dy / d,
                 b.x - 16 * dx / d, b.y - 16 * dy / d, { color: AC.line, w: 1.4, head: 5 });
      }
      /* nodes */
      for (let i = 0; i < N; i++) {
        const p = pos(i);
        const isEntry = (i === mu);
        g.append("circle").attr("cx", p.x).attr("cy", p.y).attr("r", 14)
          .attr("fill", isEntry ? "#2a3550" : AC.panel2)
          .attr("stroke", isEntry ? AC.violet : AC.line).attr("stroke-width", isEntry ? 2 : 1);
        g.append("text").attr("x", p.x).attr("y", p.y + 4).attr("text-anchor", "middle")
          .attr("font-size", 11).attr("fill", AC.ink).text(i);
      }
      g.append("text").attr("x", pos(mu).x).attr("y", pos(mu).y - 22).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", AC.violet).text("entry");
      AL.arrow(g, 10, cy, 34 - 16, cy, { color: AC.accent, w: 1.6 });
      g.append("text").attr("x", 4).attr("y", cy - 10).attr("font-size", 10).attr("fill", AC.accent).text("head");

      /* the pointers */
      const marker = (idx, label, col, dy) => {
        if (idx === undefined || idx === null) return;
        const p = pos(idx);
        g.append("circle").attr("cx", p.x).attr("cy", p.y).attr("r", 18.5)
          .attr("fill", "none").attr("stroke", col).attr("stroke-width", 2);
        g.append("text").attr("x", p.x).attr("y", p.y + dy).attr("text-anchor", "middle")
          .attr("font-size", 10.5).attr("fill", col).text(label);
      };
      if (fr.phase === 1) {
        marker(fr.marks.slow, "slow", AC.a2, -24);
        marker(fr.marks.fast, "fast", AC.bad, 32);
      } else {
        marker(fr.marks.p, "p (from head)", AC.teal, -24);
        marker(fr.marks.q, "q (from meeting)", AC.a2, 32);
      }

      /* the algebra panel */
      const px = 0, py = 236;
      g.append("text").attr("x", px).attr("y", py).attr("font-size", 11).attr("fill", AC.muted)
        .text(fr.phase === 1 ? "phase 1 — why a meeting is forced" : "phase 2 — why the meeting point plus the head give the entry");
      const alg = fr.phase === 1 ? [
        "after i rounds:  slow has taken i links,  fast has taken 2i",
        "both inside the cycle  ⇒  they coincide when  2i − mu  ≡  i − mu  (mod lam)  ⇔  i ≡ 0 (mod lam)",
        "so the first meeting is at  i = lam · max(1, ceil(mu / lam))  =  " + built.predMeetRound
          + "   (measured: " + (fr.counts.round || 0) + " so far)"
      ] : [
        "the meeting happened at  i = " + built.round + ",  a multiple of lam = " + lam,
        "p from the head reaches the entry after exactly mu = " + mu + " steps",
        "q, already at cycle offset (i − mu) mod lam, reaches offset (i − mu + mu) mod lam = i mod lam = 0 — the entry, in the same mu steps"
      ];
      alg.forEach((t, i) => {
        g.append("text").attr("x", px).attr("y", py + 16 + i * 14).attr("font-size", 10.5)
          .attr("fill", i === 2 ? AC.ink : AC.muted).text(t);
      });

      /* running counts */
      const cc = fr.counts;
      g.append("text").attr("x", f.iw).attr("y", 8).attr("text-anchor", "end").attr("font-size", 11)
        .attr("fill", AC.ink)
        .text("rounds: " + (cc.round || 0) + "   slow links: " + (cc.slow || 0)
              + "   fast links: " + (cc.fast || 0) + "   phase-2 steps: " + (cc.phase2 || 0));

      const words = fr.note.split(" ");
      const lines = []; let cur = "";
      words.forEach(wd => {
        if ((cur + " " + wd).length > 110) { lines.push(cur); cur = wd; }
        else cur = cur ? cur + " " + wd : wd;
      });
      if (cur) lines.push(cur);
      lines.slice(0, 2).forEach((ln, i) => {
        g.append("text").attr("x", px).attr("y", 300 + i * 15).attr("font-size", 11.5)
          .attr("fill", i === 0 ? AC.ink : AC.muted).text(ln);
      });
    };
    AL.stepper(svg, { frames: built.frames, render: render, delay: 1000, label: "step" });

    const c = built.c;
    els.out.html(
      "<b>tail " + mu + ", cycle " + lam + "</b> — measured by running the routine: the pointers met after <b>"
      + built.round + "</b> rounds at node <b>" + built.meet + "</b>; phase two then took <b>" + built.steps
      + "</b> step" + (built.steps === 1 ? "" : "s") + " and landed on node <b>" + built.entry
      + "</b>. Total link traversals: <b>" + (c.get("slow") + c.get("fast") + 2 * c.get("phase2"))
      + "</b>, and the extra space used is two references — O(1), whatever the list's length. "
      + "Three independent cross-checks: the meeting round should be the least positive multiple of "
      + lam + " that is at least " + mu + ", i.e. <code>" + built.predMeetRound + "</code> — " + LS.verdict(built.okMeet)
      + "; phase two should take exactly <code>" + mu + "</code> steps — " + LS.verdict(built.okPhase2)
      + "; and it should land on the node the generator made the entry, node <code>" + mu + "</code> — "
      + LS.verdict(built.okEntry) + "."
    );
  }

  [els.mu, els.lam].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══ FIGURE 08 ═══════════════════════════════════════════════════════════
   The two stack implementations, run side by side. Both are real: one pushes
   into a doubling block, one allocates a node per push. Every bar is the
   measured write count of ONE push; the totals underneath are measured too.
   The point is not which is cheaper on average — it is that one of them has
   a flat worst case and the other does not.                                */
(function () {
  const svg = d3.select("#stk-svg"); if (svg.empty()) return;
  const W = 720, H = 330;

  const els = {
    n: d3.select("#stk-n"), nOut: d3.select("#stk-n-out"),
    out: d3.select("#stk-readout"), tbl: d3.select("#stk-table")
  };

  /* array-backed: a doubling block, exactly as a dynamic array grows */
  function arrayStack() {
    let data = new Array(1), size = 0, cap = 1;
    return {
      push: (v, c) => {
        if (size === cap) {
          const nd = new Array(cap * 2);
          for (let i = 0; i < size; i++) { nd[i] = data[i]; c.add("write"); c.add("copy"); }
          data = nd; cap *= 2; c.add("allocCall");
        }
        data[size++] = v; c.add("write");
      },
      pop: c => { c.add("write"); return data[--size]; },
      cap: () => cap, size: () => size
    };
  }
  /* node-backed: one allocation per element, links rewritten at the top */
  function nodeStack() {
    let top = null, size = 0;
    return {
      push: (v, c) => { c.add("allocCall"); const nd = { v: v, next: top }; c.add("write", 2); top = nd; size++; },
      pop: c => { const v = top.v; top = top.next; c.add("write"); size--; return v; },
      size: () => size
    };
  }

  function draw() {
    const n = +els.n.property("value");
    els.nOut.text(n);

    const A = arrayStack(), B = nodeStack();
    const perA = [], perB = [];
    const ca = AL.counter(), cb = AL.counter();
    for (let i = 0; i < n; i++) {
      const before = ca.get("write"); A.push(i, ca); perA.push(ca.get("write") - before);
      const beforeB = cb.get("write"); B.push(i, cb); perB.push(cb.get("write") - beforeB);
    }

    /* independent cross-check: growing a block by doubling from capacity 1 to
       hold n elements copies 1 + 2 + 4 + … = 2^ceil(log2 n) − 1 elements */
    const predCopies = n <= 1 ? 0 : Math.pow(2, Math.ceil(Math.log2(n))) - 1;
    const okCopies = (ca.get("copy") === predCopies);
    const predAllocA = n <= 1 ? 0 : Math.ceil(Math.log2(n));
    const okAllocA = (ca.get("allocCall") === predAllocA);
    const okAllocB = (cb.get("allocCall") === n);

    const f = AL.frame(svg, W, H, { l: 52, r: 152, t: 20, b: 44 });
    const g = f.g;
    const x = d3.scaleLinear().domain([0, n]).range([0, f.iw]);
    const top = Math.max(4, d3.max(perA));
    const y = d3.scaleLinear().domain([0, top]).nice().range([f.ih, 0]);
    AL.gridY(g, y, f.iw, 5);
    AL.axisB(g, x, f.ih, 8, "push number", d3.format("d"));
    AL.axisL(g, y, 5, "memory writes performed by that one push");

    const bw = Math.max(1, f.iw / n - 1);
    perA.forEach((v, i) => {
      g.append("rect").attr("x", x(i)).attr("y", y(v)).attr("width", bw)
        .attr("height", f.ih - y(v)).attr("fill", v > 1 ? AC.bad : AC.accent).attr("rx", 1);
    });
    /* the node stack: a flat line, because every push costs the same */
    g.append("line").attr("x1", 0).attr("x2", f.iw).attr("y1", y(perB[0])).attr("y2", y(perB[0]))
      .attr("stroke", AC.a2).attr("stroke-width", 2.2);
    /* the array's running average */
    let run = 0;
    const avg = perA.map((v, i) => { run += v; return run / (i + 1); });
    g.append("path").datum(avg).attr("fill", "none").attr("stroke", AC.teal)
      .attr("stroke-width", 1.8).attr("stroke-dasharray", "4 3")
      .attr("d", d3.line().x((d, i) => x(i)).y(d => y(d)));

    AL.legend(g, [
      { label: "array push — ordinary", color: AC.accent },
      { label: "array push — a resize fired", color: AC.bad },
      { label: "array, running average", color: AC.teal, dash: "4 3" },
      { label: "node push — every push", color: AC.a2 }
    ], f.iw + 12, 16);

    const peakA = d3.max(perA), meanA = d3.mean(perA);
    g.append("text").attr("x", f.iw + 12).attr("y", 84).attr("font-size", 11).attr("fill", AC.muted)
      .text("worst single push");
    g.append("text").attr("x", f.iw + 12).attr("y", 100).attr("font-size", 12).attr("fill", AC.bad)
      .text("array  " + peakA + " writes");
    g.append("text").attr("x", f.iw + 12).attr("y", 116).attr("font-size", 12).attr("fill", AC.a2)
      .text("nodes  " + perB[0] + " writes");

    els.out.html(
      "<b>" + n + " pushes</b>, both implementations measured. The array-backed stack wrote <b>"
      + LS.int(ca.get("write")) + "</b> words in total — a mean of <b>" + LS.sig(meanA, 2)
      + "</b> per push — but its worst single push wrote <b>" + peakA
      + "</b>, and that spike grows with the stack. The node-backed stack wrote <b>"
      + LS.int(cb.get("write")) + "</b> in total, <b>" + perB[0] + "</b> on <i>every</i> push, worst case included. "
      + "Allocator calls are the other half of the story: <b>" + ca.get("allocCall") + "</b> for the array against <b>"
      + cb.get("allocCall") + "</b> for the nodes — one per element, each of which is far more expensive than a word write. "
      + "Cross-checks — doubling from capacity 1 must copy <code>2^ceil(log₂ n) − 1 = " + LS.int(predCopies)
      + "</code> elements: " + LS.verdict(okCopies) + "; it must reallocate <code>ceil(log₂ n) = " + predAllocA
      + "</code> times: " + LS.verdict(okAllocA) + "; the node stack must allocate exactly <code>n = " + n
      + "</code> times: " + LS.verdict(okAllocB) + "."
    );

    LS.table("#stk-table",
      ["", "array-backed (doubling block)", "node-backed (one node per element)"],
      [
        ["push, amortized", "O(1) — " + LS.sig(meanA, 2) + " writes measured", "O(1) — " + perB[0] + " writes, always"],
        ["push, worst case", "<span style='color:" + AC.bad + "'>Θ(n)</span> — " + peakA + " writes measured here",
          "<b style='color:" + AC.good + "'>O(1)</b> — " + perB[0] + " writes"],
        ["pop", "O(1)", "O(1)"],
        ["allocator calls for n pushes", LS.int(ca.get("allocCall")) + " = ceil(log₂ n)", LS.int(cb.get("allocCall")) + " = n"],
        ["memory per element", "one payload, plus up to 100% unused slack", "payload + one link + allocator header"],
        ["locality", "one block, sequential", "n blocks, wherever the allocator put them"],
        ["reference stability", "invalidated by every resize", "a node's address never changes"]
      ]);
  }

  els.n.on("input change", draw);
  draw();
})();

/* ══ FIGURE 09 ═══════════════════════════════════════════════════════════
   The call stack, running a real recursion. Frames are pushed and popped by
   an actual interpreter loop over the recursion, not by a hand-built script;
   the total call count and the peak depth are read off AL.counter() and then
   checked against two closed forms.                                        */
(function () {
  const svg = d3.select("#frame-svg"); if (svg.empty()) return;
  const W = 720, H = 360;

  const els = {
    fn: d3.select("#frame-fn"), n: d3.select("#frame-n"),
    nOut: d3.select("#frame-n-out"), out: d3.select("#frame-readout")
  };

  function build(fn, n) {
    const c = AL.counter();
    const frames = [];
    const stack = [];
    let peak = 0;
    const snap = (note, kind) => {
      peak = Math.max(peak, stack.length);
      frames.push({ stack: stack.map(f => Object.assign({}, f)), note: note, kind: kind,
                    counts: c.all(), peak: peak });
    };

    if (fn === "fib") {
      const fib = (k) => {
        stack.push({ name: "fib(" + k + ")", locals: "k = " + k, state: "entered", ret: null });
        c.add("call");
        snap("Call fib(" + k + "): a frame is pushed holding the parameter, room for the locals, and the address to return to.", "push");
        if (k <= 1) {
          stack[stack.length - 1].state = "base case"; stack[stack.length - 1].ret = k;
          snap("Base case: fib(" + k + ") = " + k + ". The frame is about to be popped and its space reused by the next call.", "ret");
          stack.pop(); c.add("ret");
          return k;
        }
        stack[stack.length - 1].state = "waiting on fib(" + (k - 1) + ")";
        snap("fib(" + k + ") suspends here. Its frame stays live — with its own copy of k — while the callee runs.", "wait");
        const a = fib(k - 1);
        stack[stack.length - 1].locals = "k = " + k + ", a = " + a;
        stack[stack.length - 1].state = "waiting on fib(" + (k - 2) + ")";
        snap("fib(" + (k - 1) + ") returned " + a + ". It is stored in THIS frame's local slot — which is why each frame needs its own.", "wait");
        const b = fib(k - 2);
        const v = a + b;
        stack[stack.length - 1].locals = "k = " + k + ", a = " + a + ", b = " + b;
        stack[stack.length - 1].state = "returns " + v;
        stack[stack.length - 1].ret = v;
        snap("fib(" + k + ") = " + a + " + " + b + " = " + v + ". Pop the frame and hand the value back to the caller.", "ret");
        stack.pop(); c.add("ret");
        return v;
      };
      const val = fib(n);
      snap("Stack empty — every frame that was pushed has been popped. fib(" + n + ") = " + val + ".", "done");
      /* closed forms: calls = 2·F(n+1) − 1, peak depth = n (for n ≥ 1) */
      const F = [0, 1]; for (let i = 2; i <= n + 2; i++) F.push(F[i - 1] + F[i - 2]);
      return { frames: frames, c: c, peak: peak, val: val,
               predCalls: 2 * F[n + 1] - 1, predDepth: Math.max(1, n) };

    } else {                                  /* towers of hanoi */
      const moves = [];
      const hanoi = (k, from, to, via) => {
        stack.push({ name: "move(" + k + ", " + from + "→" + to + ")", locals: "via " + via, state: "entered", ret: null });
        c.add("call");
        snap("Call move(" + k + ", " + from + "→" + to + "): the frame records which disk, which pegs, and where to resume.", "push");
        if (k === 1) {
          moves.push(from + "→" + to); c.add("move");
          stack[stack.length - 1].state = "base: move disk 1 " + from + "→" + to;
          snap("Base case: move the single disk " + from + "→" + to + ". Move number " + c.get("move") + ".", "ret");
          stack.pop(); c.add("ret");
          return;
        }
        stack[stack.length - 1].state = "move " + (k - 1) + " off to " + via;
        snap("Suspend: first shift the top " + (k - 1) + " disks out of the way, onto " + via + ".", "wait");
        hanoi(k - 1, from, via, to);
        moves.push(from + "→" + to); c.add("move");
        stack[stack.length - 1].state = "moved disk " + k + " " + from + "→" + to;
        snap("Now move disk " + k + " itself, " + from + "→" + to + ". Move number " + c.get("move") + ".", "wait");
        stack[stack.length - 1].state = "move " + (k - 1) + " back onto " + to;
        snap("Then bring the " + (k - 1) + " disks from " + via + " onto " + to + ". Still only one frame per level of disk.", "wait");
        hanoi(k - 1, via, to, from);
        stack[stack.length - 1].state = "done";
        snap("move(" + k + ") is finished. Pop.", "ret");
        stack.pop(); c.add("ret");
      };
      hanoi(n, "A", "C", "B");
      snap("Stack empty. " + c.get("move") + " disk moves, and the deepest the stack ever got was " + peak + " frames.", "done");
      return { frames: frames, c: c, peak: peak, val: c.get("move"),
               predCalls: Math.pow(2, n) - 1, predDepth: n };
    }
  }

  function draw() {
    const fn = els.fn.property("value");
    const n = +els.n.property("value");
    els.nOut.text(n);
    const built = build(fn, n);
    LS.clearControls(svg.node());

    const render = (fr) => {
      const f = AL.frame(svg, W, H, { l: 20, r: 16, t: 18, b: 12 });
      const g = f.g;
      const baseY = f.ih - 26, fh = 30, fw = 330;

      g.append("text").attr("x", 0).attr("y", 6).attr("font-size", 11).attr("fill", AC.muted)
        .text("the call stack — the newest frame is the one on top, and it is the only one running");

      /* the region the stack grows into */
      g.append("rect").attr("x", 0).attr("y", 18).attr("width", fw).attr("height", baseY - 18 + 8)
        .attr("fill", AC.panel).attr("stroke", AC.line).attr("rx", 4).attr("opacity", 0.5);
      g.append("text").attr("x", 4).attr("y", baseY + 20).attr("font-size", 10).attr("fill", AC.muted)
        .text("base of the stack — the caller that started it all");

      fr.stack.forEach((sf, i) => {
        const yy = baseY - (i + 1) * fh;
        const isTop = i === fr.stack.length - 1;
        g.append("rect").attr("x", 6).attr("y", yy).attr("width", fw - 12).attr("height", fh - 3)
          .attr("rx", 4).attr("fill", isTop ? "#2a3550" : AC.panel2)
          .attr("stroke", isTop ? AC.a2 : AC.line).attr("stroke-width", isTop ? 2 : 1);
        g.append("text").attr("x", 14).attr("y", yy + 12).attr("font-size", 11.5)
          .attr("fill", isTop ? AC.a2 : AC.ink).text(sf.name);
        g.append("text").attr("x", 14).attr("y", yy + 23).attr("font-size", 10)
          .attr("fill", AC.muted).text(sf.locals + "  ·  " + sf.state);
      });

      /* what a frame holds — the static panel */
      const px = fw + 24;
      g.append("text").attr("x", px).attr("y", 6).attr("font-size", 11).attr("fill", AC.muted)
        .text("what one frame holds");
      ["· the return address — where to resume",
       "· the caller's saved frame pointer",
       "· this call's own parameters",
       "· this call's own local variables",
       "· temporaries and saved registers"].forEach((t, i) => {
        g.append("text").attr("x", px).attr("y", 24 + i * 15).attr("font-size", 10.5)
          .attr("fill", AC.muted).text(t);
      });
      g.append("text").attr("x", px).attr("y", 116).attr("font-size", 11).attr("fill", AC.muted)
        .text("measured, live");
      const cc = fr.counts;
      [["calls made so far", cc.call || 0],
       ["returns so far", cc.ret || 0],
       ["frames live now", fr.stack.length],
       ["deepest so far", fr.peak]].forEach((t, i) => {
        g.append("text").attr("x", px).attr("y", 134 + i * 16).attr("font-size", 11).attr("fill", AC.muted)
          .text(t[0]);
        g.append("text").attr("x", px + 160).attr("y", 134 + i * 16).attr("text-anchor", "end")
          .attr("font-size", 12).attr("fill", i === 3 ? AC.a2 : AC.ink).text(t[1]);
      });

      const words = fr.note.split(" ");
      const lines = []; let cur = "";
      words.forEach(wd => {
        if ((cur + " " + wd).length > 108) { lines.push(cur); cur = wd; }
        else cur = cur ? cur + " " + wd : wd;
      });
      if (cur) lines.push(cur);
      lines.slice(0, 2).forEach((ln, i) => {
        g.append("text").attr("x", px).attr("y", 220 + i * 15).attr("font-size", 11)
          .attr("fill", i === 0 ? AC.ink : AC.muted).text(ln);
      });
    };
    AL.stepper(svg, { frames: built.frames, render: render, delay: 700, label: "step" });

    const c = built.c;
    const okCalls = (c.get("call") === built.predCalls);
    const okDepth = (built.peak === built.predDepth);
    const okBalance = (c.get("call") === c.get("ret"));
    els.out.html(
      "<b>" + (fn === "fib" ? "fib(" + n + ")" : "towers of hanoi, " + n + " disks") + "</b> — measured: <b>"
      + c.get("call") + "</b> calls, <b>" + c.get("ret") + "</b> returns, and the stack was never deeper than <b>"
      + built.peak + "</b> frames"
      + (fn === "fib"
         ? ". That gap is the whole point: the number of calls grows exponentially while the stack depth grows linearly, so recursion depth and recursion COST are different quantities."
         : ". Every disk move corresponds to a base-case frame; " + c.get("move") + " moves from " + c.get("call") + " calls.")
      + " Cross-checks — calls should be <code>" + built.predCalls + "</code> ("
      + (fn === "fib" ? "2·F(n+1) − 1" : "2ⁿ − 1") + "): " + LS.verdict(okCalls)
      + "; peak depth should be <code>" + built.predDepth + "</code>: " + LS.verdict(okDepth)
      + "; and every call must be matched by exactly one return: " + LS.verdict(okBalance) + "."
    );
  }

  [els.fn, els.n].forEach(s => s.on("input change", draw));
  draw();
})();

/* ══ FIGURE 10 ═══════════════════════════════════════════════════════════
   The balanced-delimiter checker: the smallest problem a stack solves that a
   counter cannot. Stepped through the real scan. The verdict is cross-checked
   against a completely different algorithm — repeatedly deleting adjacent
   matching pairs until nothing changes, which leaves the empty string exactly
   when the input was balanced.                                             */
(function () {
  const svg = d3.select("#delim-svg"); if (svg.empty()) return;
  const W = 720, H = 300;

  const els = {
    inp: d3.select("#delim-input"), preset: d3.select("#delim-preset"),
    out: d3.select("#delim-readout")
  };
  const OPEN = "([{", CLOSE = ")]}";
  const MATCH = { ")": "(", "]": "[", "}": "{" };

  /* algorithm A — the stack scan, instrumented */
  function scan(s, c) {
    const st = [], frames = [];
    const snap = (i, note, kind) => frames.push({
      i: i, stack: st.slice(), note: note, kind: kind, counts: c.all()
    });
    snap(-1, "Start with an empty stack. Every opener is a promise; the stack is where the unkept promises are recorded, most recent first.", "ok");
    let verdict = "balanced", at = -1;
    for (let i = 0; i < s.length; i++) {
      const ch = s[i];
      if (OPEN.indexOf(ch) >= 0) {
        st.push({ ch: ch, i: i }); c.add("push");
        snap(i, "'" + ch + "' is an opener — push it. The stack now owes " + st.length + " closer" + (st.length === 1 ? "" : "s") + ".", "ok");
      } else if (CLOSE.indexOf(ch) >= 0) {
        c.add("test");
        if (st.length === 0) {
          verdict = "unmatched closer"; at = i;
          snap(i, "'" + ch + "' closes something, but the stack is empty — there is no opener for it. Reject at index " + i + ".", "bad");
          break;
        }
        const t = st[st.length - 1]; c.add("compare");
        if (t.ch !== MATCH[ch]) {
          verdict = "mismatched pair"; at = i;
          snap(i, "'" + ch + "' expects '" + MATCH[ch] + "' on top, but the top is '" + t.ch
            + "' from index " + t.i + ". Nesting violated — reject at index " + i + ".", "bad");
          break;
        }
        st.pop(); c.add("pop");
        snap(i, "'" + ch + "' matches the '" + t.ch + "' opened at index " + t.i
          + " — pop it. The most recent unkept promise was exactly the right one, which is what LIFO means here.", "ok");
      } else {
        c.add("skip");
        snap(i, "'" + ch + "' is not a delimiter — ignore it and move on.", "dim");
      }
    }
    if (verdict === "balanced" && st.length > 0) {
      verdict = "unclosed opener"; at = st[st.length - 1].i;
      snap(s.length, "End of input with " + st.length + " opener" + (st.length === 1 ? "" : "s")
        + " still on the stack — the earliest unclosed one is at index " + st[0].i + ". Reject.", "bad");
    } else if (verdict === "balanced") {
      snap(s.length, "End of input and the stack is empty: every opener was closed, by the right closer, in the right order. Accept.", "good");
    }
    return { frames: frames, verdict: verdict, at: at };
  }

  /* algorithm B — an entirely different test, used only as a cross-check */
  function reduceTest(s) {
    let t = s.replace(/[^()\[\]{}]/g, "");
    let prev = null;
    while (t !== prev) { prev = t; t = t.replace(/\(\)|\[\]|\{\}/g, ""); }
    return t.length === 0;
  }

  function draw() {
    const s = (els.inp.property("value") || "").slice(0, 34);
    const c = AL.counter();
    const run = scan(s, c);
    LS.clearControls(svg.node());

    const render = (fr) => {
      const f = AL.frame(svg, W, H, { l: 22, r: 16, t: 18, b: 12 });
      const g = f.g;
      g.append("text").attr("x", 0).attr("y", 6).attr("font-size", 11).attr("fill", AC.muted)
        .text("scanning left to right — one pass, no backtracking");

      const cw = Math.min(26, (f.iw - 10) / Math.max(1, s.length));
      AL.row(g, s.split(""), {
        x: 0, y: 20, w: cw - 2, h: 28, gap: 2, index: true, fontSize: 13,
        mark: (k) => k === fr.i ? (fr.kind === "bad" ? AC.bad : AC.a2)
                   : (k < fr.i ? AC.panel : null)
      });

      /* the stack, drawn growing upward */
      const sx = 0, sBase = 214, bw = 40, bh = 22;
      g.append("text").attr("x", sx).attr("y", sBase + 22).attr("font-size", 10).attr("fill", AC.muted)
        .text("bottom of the stack");
      fr.stack.forEach((e, i) => {
        g.append("rect").attr("x", sx).attr("y", sBase - (i + 1) * bh).attr("width", bw).attr("height", bh - 2)
          .attr("rx", 3).attr("fill", i === fr.stack.length - 1 ? "#2a3550" : AC.panel2)
          .attr("stroke", i === fr.stack.length - 1 ? AC.a2 : AC.line);
        g.append("text").attr("x", sx + bw / 2).attr("y", sBase - (i + 1) * bh + 15)
          .attr("text-anchor", "middle").attr("font-size", 13).attr("fill", AC.ink).text(e.ch);
        g.append("text").attr("x", sx + bw + 6).attr("y", sBase - (i + 1) * bh + 15)
          .attr("font-size", 9.5).attr("fill", AC.muted).text("opened at index " + e.i);
      });
      if (fr.stack.length === 0) {
        g.append("text").attr("x", sx).attr("y", sBase - 8).attr("font-size", 11).attr("fill", AC.muted)
          .text("(stack empty)");
      }

      const cc = fr.counts;
      g.append("text").attr("x", f.iw).attr("y", 6).attr("text-anchor", "end").attr("font-size", 11)
        .attr("fill", AC.ink)
        .text("pushes: " + (cc.push || 0) + "   pops: " + (cc.pop || 0)
              + "   top comparisons: " + (cc.compare || 0));
      g.append("text").attr("x", f.iw).attr("y", 24).attr("text-anchor", "end").attr("font-size", 10.5)
        .attr("fill", AC.muted).text("stack height now: " + fr.stack.length);

      const words = fr.note.split(" ");
      const lines = []; let cur = "";
      words.forEach(wd => {
        if ((cur + " " + wd).length > 104) { lines.push(cur); cur = wd; }
        else cur = cur ? cur + " " + wd : wd;
      });
      if (cur) lines.push(cur);
      lines.slice(0, 2).forEach((ln, i) => {
        g.append("text").attr("x", 0).attr("y", 248 + i * 15).attr("font-size", 11.5)
          .attr("fill", i === 0 ? (fr.kind === "bad" ? AC.bad : fr.kind === "good" ? AC.good : AC.ink) : AC.muted)
          .text(ln);
      });
    };
    AL.stepper(svg, { frames: run.frames, render: render, delay: 900, label: "character" });

    const stackSaysOk = (run.verdict === "balanced");
    const reduceSaysOk = reduceTest(s);
    const agree = (stackSaysOk === reduceSaysOk);
    const maxDepth = Math.max(0, ...run.frames.map(fr => fr.stack.length));

    els.out.html(
      "<b>“" + (s || "(empty)") + "”</b> — the scan reports <b>"
      + (stackSaysOk ? "<span style='color:" + AC.good + "'>balanced</span>"
                     : "<span style='color:" + AC.bad + "'>" + run.verdict + " at index " + run.at + "</span>")
      + "</b> after <b>" + c.get("push") + "</b> pushes, <b>" + c.get("pop") + "</b> pops and <b>"
      + c.get("compare") + "</b> comparisons against the top. Peak stack height <b>" + maxDepth
      + "</b> — that is the deepest nesting in the string, and it is the space the algorithm needs: O(n) worst case, on an input like <code>((((…</code>. "
      + "Cross-check against a completely different test — repeatedly delete adjacent matching pairs until nothing changes, and see whether the string vanishes; it says <b>"
      + (reduceSaysOk ? "balanced" : "not balanced") + "</b>: " + LS.verdict(agree) + "."
    );
  }

  els.preset.on("change", () => { els.inp.property("value", els.preset.property("value")); draw(); });
  els.inp.on("input change", draw);
  draw();
})();

/* ══ FIGURE 11 ═══════════════════════════════════════════════════════════
   Infix to postfix with one operator stack, stepped. The cross-check is the
   strongest on the page: the postfix string the conversion produces is
   evaluated by a stack machine, the ORIGINAL infix string is evaluated by an
   independent recursive-descent parser, and the two numbers are compared. If
   the conversion were wrong in any way that changes meaning, they would part
   company.                                                                  */
(function () {
  const svg = d3.select("#shunt-svg"); if (svg.empty()) return;
  const W = 720, H = 330;

  const els = {
    inp: d3.select("#shunt-input"), preset: d3.select("#shunt-preset"),
    out: d3.select("#shunt-readout")
  };
  const PREC = { "+": 2, "-": 2, "*": 3, "/": 3, "^": 4 };
  const RIGHT = { "^": true };

  function tokenize(s) {
    const t = [], re = /\d+(?:\.\d+)?|[-+*/^()]/g;
    let m; while ((m = re.exec(s)) !== null) t.push(m[0]);
    return t;
  }

  /* the conversion, instrumented and snapshotted */
  function shunt(toks, c) {
    const out = [], st = [], frames = [];
    const snap = (i, note, kind) => frames.push({
      i: i, out: out.slice(), stack: st.slice(), note: note, kind: kind, counts: c.all()
    });
    snap(-1, "Operands go straight to the output. Operators wait on the stack until an operator of the same or higher precedence arrives — which is exactly when they can safely be emitted.", "ok");
    let bad = null;
    for (let i = 0; i < toks.length && !bad; i++) {
      const t = toks[i];
      if (/^\d/.test(t)) {
        out.push(t); c.add("emit");
        snap(i, "'" + t + "' is a number — emit it immediately. Operands never wait, because postfix keeps them in their original order.", "ok");
      } else if (t === "(") {
        st.push(t); c.add("push");
        snap(i, "'(' pushes a barrier. Nothing below it may be popped until the matching ')' arrives.", "ok");
      } else if (t === ")") {
        let popped = 0;
        while (st.length && st[st.length - 1] !== "(") { out.push(st.pop()); c.add("pop"); c.add("emit"); popped++; }
        if (!st.length) { bad = "unmatched ')' at token " + i; snap(i, bad, "bad"); break; }
        st.pop(); c.add("pop");
        snap(i, "')' flushes " + popped + " operator" + (popped === 1 ? "" : "s") + " back to the output, then discards the '(' — the parentheses themselves never appear in postfix, because the order already encodes them.", "ok");
      } else if (PREC[t]) {
        let popped = 0;
        while (st.length && st[st.length - 1] !== "(" &&
               (PREC[st[st.length - 1]] > PREC[t] ||
                (PREC[st[st.length - 1]] === PREC[t] && !RIGHT[t]))) {
          out.push(st.pop()); c.add("pop"); c.add("emit"); popped++; c.add("preccmp");
        }
        c.add("preccmp");
        st.push(t); c.add("push");
        snap(i, "'" + t + "' has precedence " + PREC[t] + (RIGHT[t] ? " and is right-associative" : " and is left-associative")
          + ", so it first pops " + popped + " operator" + (popped === 1 ? "" : "s") + " that must be applied before it, then goes on the stack.", "ok");
      } else {
        bad = "unexpected token '" + t + "'"; snap(i, bad, "bad"); break;
      }
    }
    if (!bad) {
      while (st.length) {
        if (st[st.length - 1] === "(") { bad = "unmatched '('"; break; }
        out.push(st.pop()); c.add("pop"); c.add("emit");
      }
      snap(toks.length, bad ? bad : "Input exhausted — drain the stack. The operators come off in exactly the order they must be applied, deepest-binding first.", bad ? "bad" : "good");
    }
    return { frames: frames, out: out, bad: bad };
  }

  /* evaluator A — a stack machine over the postfix output */
  function evalPostfix(rpn) {
    const s = [];
    for (const t of rpn) {
      if (/^\d/.test(t)) s.push(parseFloat(t));
      else {
        const b = s.pop(), a = s.pop();
        if (a === undefined || b === undefined) return NaN;
        s.push(t === "+" ? a + b : t === "-" ? a - b : t === "*" ? a * b
             : t === "/" ? a / b : Math.pow(a, b));
      }
    }
    return s.length === 1 ? s[0] : NaN;
  }
  /* evaluator B — an independent recursive-descent parser over the INFIX text */
  function evalInfix(toks) {
    let k = 0;
    const peek = () => toks[k];
    function primary() {
      const t = toks[k];
      if (t === "(") { k++; const v = expr(0); if (toks[k] !== ")") throw 0; k++; return v; }
      if (/^\d/.test(t)) { k++; return parseFloat(t); }
      throw 0;
    }
    function expr(minp) {
      let lhs = primary();
      while (k < toks.length && PREC[peek()] !== undefined && PREC[peek()] >= minp) {
        const op = toks[k++];
        const nextMin = RIGHT[op] ? PREC[op] : PREC[op] + 1;
        const rhs = expr(nextMin);
        lhs = op === "+" ? lhs + rhs : op === "-" ? lhs - rhs : op === "*" ? lhs * rhs
            : op === "/" ? lhs / rhs : Math.pow(lhs, rhs);
      }
      return lhs;
    }
    try { const v = expr(0); return k === toks.length ? v : NaN; } catch (e) { return NaN; }
  }

  function draw() {
    const src = (els.inp.property("value") || "").replace(/\s+/g, "").slice(0, 26);
    const toks = tokenize(src);
    const c = AL.counter();
    const run = shunt(toks, c);
    LS.clearControls(svg.node());

    const render = (fr) => {
      const f = AL.frame(svg, W, H, { l: 22, r: 16, t: 18, b: 12 });
      const g = f.g;
      g.append("text").attr("x", 0).attr("y", 6).attr("font-size", 11).attr("fill", AC.muted)
        .text("input, read once left to right");
      const cw = Math.min(30, (f.iw - 180) / Math.max(1, toks.length));
      AL.row(g, toks, {
        x: 0, y: 18, w: cw - 2, h: 26, gap: 2, index: false, fontSize: 12.5,
        mark: k => k === fr.i ? (fr.kind === "bad" ? AC.bad : AC.a2) : (k < fr.i ? AC.panel : null)
      });

      g.append("text").attr("x", 0).attr("y", 78).attr("font-size", 11).attr("fill", AC.muted)
        .text("output — postfix, built once and never reordered");
      AL.row(g, fr.out, {
        x: 0, y: 90, w: 28, h: 26, gap: 2, index: false, fontSize: 12.5,
        mark: () => "#1d2b22", stroke: "#2f4a38"
      });
      if (fr.out.length === 0) g.append("text").attr("x", 0).attr("y", 108).attr("font-size", 11)
        .attr("fill", AC.muted).text("(empty)");

      /* the operator stack */
      const sx = 0, sBase = 240, bw = 34, bh = 22;
      g.append("text").attr("x", sx).attr("y", sBase + 20).attr("font-size", 10).attr("fill", AC.muted)
        .text("bottom of the operator stack");
      fr.stack.forEach((e, i) => {
        g.append("rect").attr("x", sx).attr("y", sBase - (i + 1) * bh).attr("width", bw).attr("height", bh - 2)
          .attr("rx", 3).attr("fill", i === fr.stack.length - 1 ? "#2a3550" : AC.panel2)
          .attr("stroke", i === fr.stack.length - 1 ? AC.a2 : AC.line);
        g.append("text").attr("x", sx + bw / 2).attr("y", sBase - (i + 1) * bh + 15)
          .attr("text-anchor", "middle").attr("font-size", 13).attr("fill", AC.ink).text(e);
        g.append("text").attr("x", sx + bw + 6).attr("y", sBase - (i + 1) * bh + 15)
          .attr("font-size", 9.5).attr("fill", AC.muted)
          .text(e === "(" ? "barrier" : "precedence " + PREC[e] + (RIGHT[e] ? ", right-assoc" : ""));
      });
      if (fr.stack.length === 0) g.append("text").attr("x", sx).attr("y", sBase - 8)
        .attr("font-size", 11).attr("fill", AC.muted).text("(stack empty)");

      const cc = fr.counts;
      g.append("text").attr("x", f.iw).attr("y", 6).attr("text-anchor", "end").attr("font-size", 11)
        .attr("fill", AC.ink)
        .text("pushes: " + (cc.push || 0) + "   pops: " + (cc.pop || 0)
              + "   tokens emitted: " + (cc.emit || 0));

      const words = fr.note.split(" ");
      const lines = []; let cur = "";
      words.forEach(wd => {
        if ((cur + " " + wd).length > 104) { lines.push(cur); cur = wd; }
        else cur = cur ? cur + " " + wd : wd;
      });
      if (cur) lines.push(cur);
      lines.slice(0, 3).forEach((ln, i) => {
        g.append("text").attr("x", 0).attr("y", 268 + i * 15).attr("font-size", 11.5)
          .attr("fill", i === 0 ? (fr.kind === "bad" ? AC.bad : fr.kind === "good" ? AC.good : AC.ink) : AC.muted)
          .text(ln);
      });
    };
    AL.stepper(svg, { frames: run.frames, render: render, delay: 1000, label: "token" });

    const vRPN = run.bad ? NaN : evalPostfix(run.out);
    const vINF = run.bad ? NaN : evalInfix(toks);
    const agree = (Number.isFinite(vRPN) && Number.isFinite(vINF)
                   && Math.abs(vRPN - vINF) <= 1e-9 * Math.max(1, Math.abs(vINF)));

    els.out.html(
      run.bad
        ? "<b style='color:" + AC.bad + "'>" + run.bad + "</b> — the conversion cannot complete, which is the parser's way of rejecting malformed input."
        : "<b>" + src + "</b> &nbsp;→&nbsp; <b>" + run.out.join(" ") + "</b> &nbsp;·&nbsp; measured: <b>"
          + c.get("push") + "</b> stack pushes, <b>" + c.get("pop") + "</b> pops, <b>" + c.get("preccmp")
          + "</b> evaluations of the pop rule (one per pop, plus one exit test per operator token — some of which "
          + "short-circuit on an empty stack or a '(' barrier without reading a precedence at all), over "
          + toks.length + " tokens — one pass, O(n) time, and O(n) stack in the worst case. "
          + "Cross-check — evaluating the postfix output on a stack machine gives <b>" + LS.sig(vRPN, 6)
          + "</b>; evaluating the original infix string with an independent recursive-descent parser gives <b>"
          + LS.sig(vINF, 6) + "</b>: " + LS.verdict(agree)
          + ". Note that the postfix form contains <b>no parentheses at all</b> — the order alone carries the grouping."
    );
  }

  els.preset.on("change", () => { els.inp.property("value", els.preset.property("value")); draw(); });
  els.inp.on("input change", draw);
  draw();
})();

/* ══ FIGURE 12 ═══════════════════════════════════════════════════════════
   The circular buffer. The ring on the left is the idea; the flat row on the
   right is what is actually in memory, and they are drawn from the same state
   so the wraparound is visibly an index computation rather than a move. The
   three resolutions of the full-versus-empty ambiguity are real, separate
   implementations, so switching the control changes what the buffer DOES.
   After every operation the contents are read out by the modular walk and
   compared against a reference queue built with plain array push and shift. */
(function () {
  const svg = d3.select("#ring-svg"); if (svg.empty()) return;
  const W = 720, H = 396;
  const C = 8;

  const els = {
    mode: d3.select("#ring-mode"), out: d3.select("#ring-readout"),
    tbl: d3.select("#ring-table")
  };

  /* mode "count" : front + size            · full ring, one extra word
     mode "gap"   : front + back, never full · one cell always wasted
     mode "flag"  : front + back + a boolean · full ring, one extra bit      */
  function mkRing(mode) {
    const data = new Array(C).fill(null);
    let front = 0, back = 0, size = 0, full = false;
    const api = {
      data: data,
      front: () => front,
      back: () => (mode === "count" ? (front + size) % C : back),
      size: () => (mode === "count" ? size : (full ? C : (back - front + C) % C)),
      isEmpty: () => (mode === "count" ? size === 0
                    : mode === "gap" ? front === back
                    : (front === back && !full)),
      isFull: () => (mode === "count" ? size === C
                   : mode === "gap" ? (back + 1) % C === front
                   : (front === back && full)),
      enqueue: (v, c) => {
        if (api.isFull()) { c.add("reject"); return false; }
        /* one genuine modulo evaluation per operation, in every variant:
           "count" computes back from front + size, the others advance back */
        const at = api.back(); if (mode === "count") c.add("modop");
        data[at] = v; c.add("write");
        if (mode === "count") size++;
        else { back = (back + 1) % C; c.add("modop"); if (mode === "flag" && back === front) full = true; }
        return true;
      },
      dequeue: c => {
        if (api.isEmpty()) { c.add("reject"); return null; }
        const v = data[front]; data[front] = null; c.add("write");
        front = (front + 1) % C; c.add("modop");
        if (mode === "count") size--; else if (mode === "flag") full = false;
        return v;
      },
      /* the contents, read the only way a ring can be read */
      contents: () => {
        const out = [], n = api.size();
        for (let i = 0; i < n; i++) out.push(data[(front + i) % C]);
        return out;
      },
      usable: () => (mode === "gap" ? C - 1 : C)
    };
    return api;
  }

  /* the scripted run: fill, drain part way, wrap, fill completely, drain */
  const SCRIPT = (() => {
    const s = [];
    for (let i = 1; i <= 5; i++) s.push({ op: "E", v: i });
    for (let i = 0; i < 3; i++) s.push({ op: "D" });
    for (let i = 6; i <= 12; i++) s.push({ op: "E", v: i });
    for (let i = 0; i < 9; i++) s.push({ op: "D" });
    return s;
  })();

  function build(mode) {
    const R = mkRing(mode);
    const ref = [];                      // the reference queue: push / shift
    const c = AL.counter();
    const frames = [];
    let mismatch = false;
    const snap = (note, kind, hi) => {
      const got = R.contents(), want = ref.slice();
      const same = got.length === want.length && got.every((v, i) => v === want[i]);
      if (!same) mismatch = true;
      frames.push({
        cells: R.data.slice(), front: R.front(), back: R.back(), size: R.size(),
        empty: R.isEmpty(), full: R.isFull(), note: note, kind: kind, hi: hi,
        counts: c.all(), got: got, want: want, same: same
      });
    };
    snap("An empty ring. front and back both sit at index 0 — and with nothing else recorded, that state is indistinguishable from a completely full one.",
      "ok", -1);
    SCRIPT.forEach(step => {
      if (step.op === "E") {
        const at = R.back();
        const ok = R.enqueue(step.v, c);
        if (ok) { ref.push(step.v);
          snap("enqueue(" + step.v + ") writes into cell " + at + ", then back = (" + at + " + 1) mod " + C
            + " = " + R.back() + ". Nothing else moved.", "e", at);
        } else {
          snap("enqueue(" + step.v + ") REJECTED — the buffer reports full at size " + R.size()
            + (mode === "gap" ? ", one short of its " + C + " cells, because this variant always keeps one cell empty." : " of " + C + "."),
            "bad", -1);
        }
      } else {
        const at = R.front();
        const v = R.dequeue(c);
        if (v !== null) { ref.shift();
          snap("dequeue() returns " + v + " from cell " + at + ", then front = (" + at + " + 1) mod " + C
            + " = " + R.front() + ". The element was not moved; the window moved.", "d", at);
        } else {
          snap("dequeue() REJECTED — the buffer is empty.", "bad", -1);
        }
      }
    });
    return { frames: frames, c: c, ok: !mismatch, usable: R.usable() };
  }

  function draw() {
    const mode = els.mode.property("value");
    const built = build(mode);
    LS.clearControls(svg.node());

    const cxr = 128, cyr = 160, rad = 92;
    const cellAngle = i => -Math.PI / 2 + 2 * Math.PI * i / C;

    const render = (fr) => {
      const f = AL.frame(svg, W, H, { l: 16, r: 14, t: 16, b: 10 });
      const g = f.g;

      g.append("text").attr("x", 0).attr("y", 6).attr("font-size", 11).attr("fill", AC.muted)
        .text("the ring — how it is thought about");
      g.append("text").attr("x", 262).attr("y", 6).attr("font-size", 11).attr("fill", AC.muted)
        .text("the array — what is actually in memory");

      /* the ring */
      for (let i = 0; i < C; i++) {
        const th = cellAngle(i), x = cxr + rad * Math.cos(th), y = cyr + rad * Math.sin(th);
        const occupied = fr.cells[i] !== null;
        g.append("circle").attr("cx", x).attr("cy", y).attr("r", 19)
          .attr("fill", i === fr.hi ? AC.a2 : occupied ? "#2a3550" : AC.panel2)
          .attr("stroke", occupied ? AC.accent : AC.line).attr("stroke-width", occupied ? 1.8 : 1);
        g.append("text").attr("x", x).attr("y", y + 4).attr("text-anchor", "middle")
          .attr("font-size", 12).attr("fill", i === fr.hi ? AC.bg : AC.ink)
          .text(occupied ? fr.cells[i] : "");
        g.append("text").attr("x", cxr + (rad + 30) * Math.cos(th)).attr("y", cyr + (rad + 30) * Math.sin(th) + 3)
          .attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", AC.muted).text(i);
      }
      /* the wrap edge, drawn so it is visible that C−1 → 0 is an ordinary step */
      const thA = cellAngle(C - 1), thB = cellAngle(0);
      AL.arrow(g, cxr + (rad - 26) * Math.cos(thA), cyr + (rad - 26) * Math.sin(thA),
        cxr + (rad - 26) * Math.cos(thB), cyr + (rad - 26) * Math.sin(thB),
        { color: AC.violet, w: 1.4, dash: "3 2" });
      g.append("text").attr("x", cxr).attr("y", cyr - 4).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", AC.violet).text("(i + 1) mod " + C);
      g.append("text").attr("x", cxr).attr("y", cyr + 12).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("fill", fr.full ? AC.bad : fr.empty ? AC.a2 : AC.muted)
        .text(fr.full ? "FULL" : fr.empty ? "EMPTY" : "size " + fr.size);

      /* front / back markers on the ring */
      const mark = (idx, label, col, off) => {
        const th = cellAngle(idx);
        const x1 = cxr + (rad + off + 22) * Math.cos(th), y1 = cyr + (rad + off + 22) * Math.sin(th);
        const x2 = cxr + (rad + 21) * Math.cos(th), y2 = cyr + (rad + 21) * Math.sin(th);
        AL.arrow(g, x1, y1, x2, y2, { color: col, w: 1.6, head: 5 });
        g.append("text").attr("x", cxr + (rad + off + 34) * Math.cos(th))
          .attr("y", cyr + (rad + off + 34) * Math.sin(th) + 3).attr("text-anchor", "middle")
          .attr("font-size", 10).attr("fill", col).text(label);
      };
      mark(fr.front, "front", AC.good, 8);
      mark(fr.back, "back", AC.a2, 26);

      /* the same state as a flat array */
      const rx = 262, ry = 22, cw = 50;
      AL.row(g, fr.cells.map(v => v === null ? "" : v), {
        x: rx, y: ry, w: cw - 4, h: 34, gap: 4, index: true, fontSize: 13,
        mark: i => i === fr.hi ? AC.a2 : (fr.cells[i] !== null ? "#2a3550" : null)
      });
      const cellMid = i => rx + i * cw + (cw - 4) / 2;
      AL.arrow(g, cellMid(fr.front), ry - 24, cellMid(fr.front), ry - 4, { color: AC.good, w: 1.6, head: 5 });
      g.append("text").attr("x", cellMid(fr.front)).attr("y", ry - 28).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", AC.good).text("front");
      AL.arrow(g, cellMid(fr.back), ry + 74, cellMid(fr.back), ry + 54, { color: AC.a2, w: 1.6, head: 5 });
      g.append("text").attr("x", cellMid(fr.back)).attr("y", ry + 86).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", AC.a2).text("back");

      /* state panel */
      const px = rx, py = 132;
      const cc = fr.counts;
      [["front", fr.front], ["back", fr.back], ["size", fr.size],
       ["isEmpty()", fr.empty ? "true" : "false"], ["isFull()", fr.full ? "true" : "false"],
       ["index computations so far", cc.modop || 0],
       ["elements MOVED so far", 0],
       ["rejected operations", cc.reject || 0]].forEach((t, i) => {
        g.append("text").attr("x", px).attr("y", py + i * 16).attr("font-size", 11).attr("fill", AC.muted).text(t[0]);
        g.append("text").attr("x", px + 240).attr("y", py + i * 16).attr("text-anchor", "end")
          .attr("font-size", 11.5).attr("fill", i === 6 ? AC.good : AC.ink).text(t[1]);
      });
      g.append("text").attr("x", px).attr("y", py + 8 * 16 + 6).attr("font-size", 10.5)
        .attr("fill", fr.same ? AC.good : AC.bad)
        .text("contents by modular walk: [" + fr.got.join(", ") + "]  "
              + (fr.same ? "= reference queue ✓" : "≠ reference queue ✗"));

      const words = fr.note.split(" ");
      const lines = []; let cur = "";
      words.forEach(wd => {
        if ((cur + " " + wd).length > 112) { lines.push(cur); cur = wd; }
        else cur = cur ? cur + " " + wd : wd;
      });
      if (cur) lines.push(cur);
      lines.slice(0, 2).forEach((ln, i) => {
        g.append("text").attr("x", 0).attr("y", 322 + i * 15).attr("font-size", 11.5)
          .attr("fill", i === 0 ? (fr.kind === "bad" ? AC.bad : AC.ink) : AC.muted).text(ln);
      });
    };
    AL.stepper(svg, { frames: built.frames, render: render, delay: 900, label: "operation" });

    const c = built.c;
    const names = { count: "front + a size counter", gap: "front + back, one cell always empty",
                    flag: "front + back + a full flag" };
    els.out.html(
      "<b>" + names[mode] + "</b>, capacity " + C + " cells, usable <b>" + built.usable
      + "</b>. Over the whole scripted run: <b>" + c.get("modop") + "</b> index computations, <b>"
      + c.get("write") + "</b> cell writes, <b>" + c.get("reject") + "</b> rejected operation"
      + (c.get("reject") === 1 ? "" : "s") + ", and <b>0</b> elements moved — which is the entire point, "
      + "because the naive array queue moves n of them on every single dequeue. "
      + "Cross-check — after every operation the contents read out by the modular walk are compared with a reference queue built from plain array push and shift: "
      + LS.verdict(built.ok) + "."
    );

    LS.table("#ring-table",
      ["resolution", "extra state", "usable cells", "empty test", "full test"],
      [
        ["a size counter", "one integer", String(C), "size = 0", "size = capacity"],
        ["leave one cell empty", "none", String(C - 1), "front = back", "(back + 1) mod capacity = front"],
        ["a full flag", "one boolean", String(C), "front = back and not full", "front = back and full"]
      ]);
  }

  els.mode.on("input change", draw);
  draw();
})();

/* ══ FIGURE 13 ═══════════════════════════════════════════════════════════
   The two-stack queue and its amortization, measured rather than asserted.
   A real implementation is run over a real operation sequence; every stack
   push and pop it performs is counted, operation by operation. The potential
   function Phi = 2·|inbox| is evaluated independently, and the figure checks the
   defining identity of the potential method at EVERY prefix of the run:
       (sum of amortized charges) − (sum of actual costs)  =  Phi − Phi_0.
   If the accounting were wrong anywhere, that identity would break.        */
(function () {
  const svg = d3.select("#two-svg"); if (svg.empty()) return;
  const W = 720, H = 400;

  const els = {
    seq: d3.select("#two-seq"), n: d3.select("#two-n"),
    nOut: d3.select("#two-n-out"), out: d3.select("#two-readout"),
    tbl: d3.select("#two-table")
  };

  function twoStackQueue() {
    const inbox = [], outbox = [];
    return {
      inbox: inbox, outbox: outbox,
      enqueue: (v, c) => { inbox.push(v); c.add("op"); },
      dequeue: c => {
        if (outbox.length === 0) {
          if (inbox.length === 0) return undefined;
          while (inbox.length) { const v = inbox.pop(); c.add("op"); outbox.push(v); c.add("op"); }
        }
        const v = outbox.pop(); c.add("op");
        return v;
      },
      size: () => inbox.length + outbox.length
    };
  }

  function build(seq, n) {
    const Q = twoStackQueue(), ref = [];
    const c = AL.counter();
    const rows = [];
    let orderOk = true, phi = 0, sumA = 0, sumAm = 0, identityOk = true;
    let next = 1;

    const ops = [];
    if (seq === "alt") { for (let i = 0; i < n; i++) ops.push(i % 2 === 0 ? "E" : "D"); }
    else if (seq === "batch") {
      for (let i = 0; i < Math.floor(n / 2); i++) ops.push("E");
      for (let i = Math.floor(n / 2); i < n; i++) ops.push("D");
    } else {
      const r = AL.rng(4242);
      for (let i = 0; i < n; i++) ops.push(r() < 0.55 ? "E" : "D");
    }

    ops.forEach((op, i) => {
      const before = c.get("op");
      const inBefore = Q.inbox.length;
      let amort;
      if (op === "E") { const v = next++; Q.enqueue(v, c); ref.push(v); amort = 3; }
      else {
        const got = Q.dequeue(c);
        const want = ref.length ? ref.shift() : undefined;
        if (got !== want) orderOk = false;
        amort = (got === undefined) ? 0 : 1;
      }
      const actual = c.get("op") - before;
      phi = 2 * Q.inbox.length;
      sumA += actual; sumAm += amort;
      if (sumAm - sumA !== phi) identityOk = false;
      rows.push({ op: op, actual: actual, amort: amort, phi: phi, sumA: sumA, sumAm: sumAm,
                  transfer: (op === "D" && inBefore > 0 && actual > 1),
                  inbox: Q.inbox.length, outbox: Q.outbox.length });
    });
    return { rows: rows, c: c, orderOk: orderOk, identityOk: identityOk, ops: ops };
  }

  function draw() {
    const seq = els.seq.property("value");
    const n = +els.n.property("value");
    els.nOut.text(n);
    const B = build(seq, n);
    const rows = B.rows;

    const f = AL.frame(svg, W, H, { l: 54, r: 148, t: 20, b: 40 });
    const g = f.g;
    const hTop = 150, gap = 46;
    const x = d3.scaleLinear().domain([0, n]).range([0, f.iw]);

    /* ── top panel: what each single operation actually cost ───────────── */
    const gT = g.append("g");
    const yTop = d3.scaleLinear().domain([0, Math.max(4, d3.max(rows, r => r.actual))]).nice()
      .range([hTop, 0]);
    AL.gridY(gT, yTop, f.iw, 4);
    gT.append("g").attr("class", "axis").attr("transform", "translate(0," + hTop + ")")
      .call(d3.axisBottom(x).ticks(8).tickFormat(d3.format("d")));
    AL.axisL(gT, yTop, 4, "stack ops in that one operation");
    const bw = Math.max(1.2, f.iw / n - 1);
    rows.forEach((r, i) => {
      gT.append("rect").attr("x", x(i)).attr("y", yTop(r.actual)).attr("width", bw)
        .attr("height", hTop - yTop(r.actual)).attr("rx", 1)
        .attr("fill", r.transfer ? AC.bad : (r.op === "E" ? AC.accent : AC.teal));
    });
    gT.append("path").datum(rows).attr("fill", "none").attr("stroke", AC.a2)
      .attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3")
      .attr("d", d3.line().x((d, i) => x(i) + bw / 2).y(d => yTop(d.amort)).curve(d3.curveStepAfter));

    /* ── bottom panel: the accounting ──────────────────────────────────── */
    const gB = g.append("g").attr("transform", "translate(0," + (hTop + gap) + ")");
    const hBot = f.ih - hTop - gap;
    const yB = d3.scaleLinear().domain([0, Math.max(1, d3.max(rows, r => r.sumAm))]).nice()
      .range([hBot, 0]);
    AL.gridY(gB, yB, f.iw, 4);
    gB.append("g").attr("class", "axis").attr("transform", "translate(0," + hBot + ")")
      .call(d3.axisBottom(x).ticks(8).tickFormat(d3.format("d")));
    AL.axisL(gB, yB, 4, "cumulative cost");
    gB.append("text").attr("x", f.iw).attr("y", hBot + 32).attr("text-anchor", "end")
      .attr("font-size", 11).attr("fill", AC.muted).text("operation number");
    /* the gap between the two curves IS the stored potential */
    gB.append("path").datum(rows).attr("fill", AC.violet).attr("opacity", 0.18)
      .attr("d", d3.area().x((d, i) => x(i)).y0(d => yB(d.sumA)).y1(d => yB(d.sumAm)));
    gB.append("path").datum(rows).attr("fill", "none").attr("stroke", AC.ink).attr("stroke-width", 2)
      .attr("d", d3.line().x((d, i) => x(i)).y(d => yB(d.sumA)));
    gB.append("path").datum(rows).attr("fill", "none").attr("stroke", AC.a2).attr("stroke-width", 2)
      .attr("stroke-dasharray", "5 3")
      .attr("d", d3.line().x((d, i) => x(i)).y(d => yB(d.sumAm)));

    AL.legend(g, [
      { label: "enqueue — actual", color: AC.accent },
      { label: "dequeue — actual", color: AC.teal },
      { label: "dequeue that transferred", color: AC.bad },
      { label: "amortized charge (3 / 1)", color: AC.a2, dash: "4 3" }
    ], f.iw + 10, 12);
    AL.legend(g, [
      { label: "cumulative actual", color: AC.ink },
      { label: "cumulative amortized", color: AC.a2, dash: "5 3" },
      { label: "the gap = potential 2·|inbox|", color: AC.violet }
    ], f.iw + 10, hTop + gap + 16);

    const last = rows[rows.length - 1];
    const peak = d3.max(rows, r => r.actual);
    const meanA = d3.mean(rows, r => r.actual);
    els.out.html(
      "<b>" + n + " operations</b> (" + B.c.get("op") + " stack pushes and pops in total, measured). "
      + "The most expensive single operation cost <b>" + peak + "</b> stack ops — that is the transfer, and it is <span class='keep'>Θ</span>(n) in the worst case — "
      + "while the mean is <b>" + LS.sig(meanA, 2) + "</b>. The dashed cumulative line never dips below the solid one, "
      + "and the shaded gap between them is exactly the stored potential <b>2·|inbox| = " + last.phi + "</b> at the end. "
      + "Cross-checks — the potential-method identity <code>Σamortized − Σactual = Φ</code> must hold at every one of the "
      + n + " prefixes: " + LS.verdict(B.identityOk)
      + "; and every dequeue must return what a reference FIFO would: " + LS.verdict(B.orderOk) + "."
    );

    LS.table("#two-table",
      ["operation", "actual stack ops", "change in <span class='keep'>Φ</span> = 2·|inbox|", "amortized = actual + <span class='keep'>Δ</span><span class='keep'>Φ</span>"],
      [
        ["enqueue", "1 — one push onto the inbox", "+2", "<b>3</b>"],
        ["dequeue, outbox non-empty", "1 — one pop off the outbox", "0", "<b>1</b>"],
        ["dequeue, outbox empty, k in inbox", "2k + 1 — k pops, k pushes, one pop", "−2k", "<b>1</b>"],
        ["<b>therefore, any sequence of m operations</b>", "<b>≤ 3m stack ops in total</b>", "<b><span class='keep'>Φ</span> ≥ 0 always, <span class='keep'>Φ</span>₀ = 0</b>", "<b>O(1) amortized</b>"]
      ]);
  }

  [els.seq, els.n].forEach(s => s.on("input change", draw));
  draw();
})();
