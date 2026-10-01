/* git.viz.js — the nine visualizations on programming/git.html.
   Loaded after ../data.js → ../notes.js (palette C). Everything page-local lives under the single
   namespace GV. Each figure is an IIFE that exits quietly if its <svg> is not on the page; the pure
   parts (SHA-1, git object hashing, the toy repository, merge bases, bisect's pick rule) are exposed
   on GV so the build's node checks can call them without a DOM.

   Honesty contract. Any hash a figure displays is computed here exactly as git computes it, or was
   copied from a real run of git 2.54 with author/committer "Student <student@example.com>" and pinned
   dates (2026-01-01 UTC). The build verified: the cascade's defaults reproduce a real repository
   byte for byte; the toy repository matched real git on 150 random sessions (39 merge commits, 32
   replaying rebases); merge bases matched `git merge-base --all` on all 45 pairs of the criss-cross
   history; bisect's pick rule matched `git bisect run` in 118 of 118 cases and its progress message
   formula matched the real messages. No figure uses randomness.

     1  #ca-svg  content-addressing cascade, live SHA-1                      (§02)
     2  #fp-svg  the four pointers: edit/add/commit/restore/reset             (§03)
     3  #dg-svg  commit-DAG sandbox: commit/branch/switch/merge/rebase/detach (§05)
     4  #mb-svg  merge bases in a criss-cross DAG + line-level three-way merge (§06)
     5  #mr-svg  merge vs rebase from one real repository                     (§07)
     6  #rl-svg  the reflog safety net                                         (§08)
     7  #rg-svg  range expressions as sets                                     (§11)
     8  #bs-svg  bisect as binary search                                       (§11)
     9  #pk-svg  packfiles and delta chains                                    (§13)          */

const GV = {};

/* ── palette (notes.js's C when present) and small helpers ─────────────────────────────────── */
GV.C = (typeof C !== "undefined" && C) ? C : { A: "#5b9cff", B: "#ffb454", good: "#4ade80", bad: "#f87171", ink: "#e6e9ef", muted: "#9aa3b2", line: "#2a2f3a" };
GV.panel = "#1e222d";
GV.DUR = (typeof window !== "undefined" && window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) ? 0 : 350;
GV.short = h => (h ? h.slice(0, 7) : "");
GV.esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
GV.text = (g, x, y, t, o) => {
  o = o || {};
  const e = g.append("text").attr("x", x).attr("y", y).attr("font-size", o.size || 11)
    .attr("fill", o.color || GV.C.muted).text(t);
  if (o.anchor) e.attr("text-anchor", o.anchor);
  if (o.weight) e.attr("font-weight", o.weight);
  if (o.mono) e.attr("font-family", "SF Mono, Menlo, monospace");
  return e;
};
GV.arrowDef = (svg, id, color) => {
  svg.append("defs").append("marker").attr("id", id).attr("viewBox", "0 0 10 10").attr("refX", 9).attr("refY", 5)
    .attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto-start-reverse")
    .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", color);
};

/* ── SHA-1 over a Uint8Array, returns 40 hex digits (FIPS 180-4) ───────────────────────────── */
GV.sha1 = function (bytes) {
  const ml = bytes.length, nBlocks = ((ml + 8) >> 6) + 1, words = new Uint32Array(nBlocks * 16);
  for (let i = 0; i < ml; i++) words[i >> 2] |= bytes[i] << (24 - (i % 4) * 8);
  words[ml >> 2] |= 0x80 << (24 - (ml % 4) * 8);
  const bits = ml * 8;
  words[nBlocks * 16 - 1] = bits >>> 0;
  words[nBlocks * 16 - 2] = Math.floor(bits / 4294967296);
  let h0 = 0x67452301, h1 = 0xefcdab89, h2 = 0x98badcfe, h3 = 0x10325476, h4 = 0xc3d2e1f0;
  const w = new Uint32Array(80), rol = (x, n) => (x << n) | (x >>> (32 - n));
  for (let b = 0; b < nBlocks; b++) {
    for (let t = 0; t < 16; t++) w[t] = words[b * 16 + t];
    for (let t = 16; t < 80; t++) w[t] = rol(w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16], 1);
    let a = h0, bb = h1, c = h2, d = h3, e = h4;
    for (let t = 0; t < 80; t++) {
      let f, k;
      if (t < 20) { f = (bb & c) | (~bb & d); k = 0x5a827999; }
      else if (t < 40) { f = bb ^ c ^ d; k = 0x6ed9eba1; }
      else if (t < 60) { f = (bb & c) | (bb & d) | (c & d); k = 0x8f1bbcdc; }
      else { f = bb ^ c ^ d; k = 0xca62c1d6; }
      const tmp = (rol(a, 5) + f + e + k + w[t]) >>> 0;
      e = d; d = c; c = rol(bb, 30) >>> 0; bb = a; a = tmp;
    }
    h0 = (h0 + a) >>> 0; h1 = (h1 + bb) >>> 0; h2 = (h2 + c) >>> 0; h3 = (h3 + d) >>> 0; h4 = (h4 + e) >>> 0;
  }
  return [h0, h1, h2, h3, h4].map(x => x.toString(16).padStart(8, "0")).join("");
};

/* ── git object ids: id = SHA-1("<type> <len>\0" + body) ───────────────────────────────────── */
GV.utf8 = s => new TextEncoder().encode(s);
GV.cat = arrs => { const n = arrs.reduce((s, a) => s + a.length, 0), o = new Uint8Array(n); let p = 0; arrs.forEach(a => { o.set(a, p); p += a.length; }); return o; };
GV.hex2bytes = h => { const o = new Uint8Array(h.length / 2); for (let i = 0; i < o.length; i++) o[i] = parseInt(h.substr(i * 2, 2), 16); return o; };
GV.objId = (type, body) => GV.sha1(GV.cat([GV.utf8(type + " " + body.length + "\0"), body]));
GV.blobId = text => GV.objId("blob", GV.utf8(text));
/* entries: [{mode:"100644"|"40000", name, id}]; git sorts a subtree as if its name ended in "/" */
GV.treeId = entries => {
  const key = e => e.name + (e.mode === "40000" ? "/" : "");
  const sorted = entries.slice().sort((a, b) => (key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : 0));
  return GV.objId("tree", GV.cat(sorted.flatMap(e => [GV.utf8(e.mode + " " + e.name + "\0"), GV.hex2bytes(e.id)])));
};
GV.WHO = "Student <student@example.com>";
GV.commitText = (tree, parents, aT, cT, msg) =>
  "tree " + tree + "\n" + parents.map(p => "parent " + p + "\n").join("") +
  "author " + GV.WHO + " " + aT + " +0000\ncommitter " + GV.WHO + " " + cT + " +0000\n\n" + msg + "\n";
GV.commitId = (tree, parents, aT, cT, msg) => GV.objId("commit", GV.utf8(GV.commitText(tree, parents, aT, cT, msg)));

/* ── toy repository. Every commit adds one new file "<msg>.txt" containing "<msg>\n", so merges never
   conflict and every tree is a union of files; the clock advances 60 s per new commit. With those
   rules the hashes are the ones real git computes for the same operations (verified in the build). */
GV.T0 = 1767258000;
GV.newRepo = () => ({ commits: {}, order: [], branches: {}, head: { ref: "main" }, clock: GV.T0, n: 0, lanes: { main: 0 }, nextLane: 1 });
GV.tick = r => (r.clock += 60);
GV.headId = r => (r.head.ref ? r.branches[r.head.ref] || null : r.head.id);
GV.treeOf = files => GV.treeId(Object.keys(files).map(n => ({ mode: "100644", name: n, id: GV.blobId(files[n]) })));
GV.mkCommit = function (r, files, parents, aT, cT, msg, lane, extra) {
  const tree = GV.treeOf(files), id = GV.commitId(tree, parents, aT, cT, msg);
  if (!r.commits[id]) { r.commits[id] = Object.assign({ id, tree, parents, files, aT, cT, msg, lane, seq: r.order.length }, extra || {}); r.order.push(id); }
  return id;
};
GV.setHead = (r, id) => { if (r.head.ref) r.branches[r.head.ref] = id; else r.head.id = id; };
GV.laneOfHead = r => (r.head.ref ? r.lanes[r.head.ref] : (r.detLane === undefined ? (r.detLane = r.nextLane++) : r.detLane));
GV.commit = function (r) {
  r.n += 1; const msg = "C" + r.n, t = GV.tick(r), p = GV.headId(r);
  const files = Object.assign({}, p ? r.commits[p].files : {}); files[msg + ".txt"] = msg + "\n";
  const id = GV.mkCommit(r, files, p ? [p] : [], t, t, msg, GV.laneOfHead(r));
  GV.setHead(r, id); return id;
};
GV.ancestors = function (r, id) {                 // inclusive set of commits reachable from id
  const seen = new Set(), st = id ? [id] : [];
  while (st.length) { const c = st.pop(); if (seen.has(c)) continue; seen.add(c); r.commits[c].parents.forEach(p => st.push(p)); }
  return seen;
};
GV.isAnc = (r, a, b) => GV.ancestors(r, b).has(a);   // is a reachable from b?
/* best common ancestors: common ancestors that are not ancestors of another common ancestor */
GV.mergeBases = function (r, a, b) {
  const A = GV.ancestors(r, a), B = GV.ancestors(r, b), common = [...A].filter(x => B.has(x));
  return common.filter(c => !common.some(d => d !== c && GV.isAnc(r, c, d)));
};
GV.branch = function (r, name) { const h = GV.headId(r); if (!h || r.branches[name]) return false; r.branches[name] = h; r.lanes[name] = r.nextLane++; return true; };
GV.switchTo = function (r, name) { if (!(name in r.branches)) return false; r.head = { ref: name }; return true; };
GV.detach = function (r, id) { r.head = { id }; r.detLane = undefined; return true; };
/* merge <name> into HEAD → "uptodate" | "ff" | "merge" */
GV.merge = function (r, name, noff) {
  const h = GV.headId(r), o = r.branches[name];
  if (!h || !o) return "none";
  if (GV.isAnc(r, o, h)) return "uptodate";
  if (GV.isAnc(r, h, o) && !noff) { GV.setHead(r, o); return "ff"; }
  const t = GV.tick(r), files = Object.assign({}, r.commits[h].files, r.commits[o].files);
  const msg = r.head.ref && r.head.ref !== "main" ? "Merge branch '" + name + "' into " + r.head.ref : "Merge branch '" + name + "'";
  GV.setHead(r, GV.mkCommit(r, files, [h, o], t, t, msg, GV.laneOfHead(r), { merge: true }));
  return "merge";
};
/* rebase HEAD onto <name>: replay commits reachable from HEAD but not from <name>, oldest first;
   author date kept, committer date = now. Linear branches only (a merge inside → "merges"). */
GV.rebase = function (r, name) {
  const h = GV.headId(r), u = r.branches[name];
  if (!h || !u) return { kind: "none" };
  if (GV.isAnc(r, u, h)) return { kind: "uptodate" };
  if (GV.isAnc(r, h, u)) { GV.setHead(r, u); return { kind: "ff" }; }
  const U = GV.ancestors(r, u), todo = [...GV.ancestors(r, h)].filter(c => !U.has(c));
  if (todo.some(c => r.commits[c].parents.length > 1)) return { kind: "merges" };
  todo.sort((a, b) => r.commits[a].seq - r.commits[b].seq);
  const t = GV.tick(r), lane = GV.laneOfHead(r), map = [];
  let base = u;
  todo.forEach(c => {
    const old = r.commits[c], files = Object.assign({}, r.commits[base].files);
    files[old.msg + ".txt"] = old.files[old.msg + ".txt"];
    base = GV.mkCommit(r, files, [base], old.aT, t, old.msg, lane, { replayOf: c });
    map.push([c, base]);
  });
  GV.setHead(r, base);
  return { kind: "rebased", map };
};
GV.reachableAll = function (r) {
  const s = new Set();
  Object.values(r.branches).forEach(id => GV.ancestors(r, id).forEach(x => s.add(x)));
  if (!r.head.ref && r.head.id) GV.ancestors(r, r.head.id).forEach(x => s.add(x));
  return s;
};

/* ── git bisect on a linear history 1..n (1 known good, n known bad). Pick rule reproduced from
   real runs: weight w = candidates reachable (inclusive); test the oldest candidate other than the
   very oldest with |2w − N| ≤ 1; otherwise the best min(w, N − w), newest first. ─────────────── */
GV.bisectPick = function (g, b) {
  const nr = b - g;
  for (let c = g + 2; c <= b; c++) { const w = c - g; if (Math.abs(2 * w - nr) <= 1) return c; }
  let best = null, bd = -1;
  for (let c = b; c > g; c--) { const w = c - g, d = Math.min(w, nr - w); if (d > bd) { bd = d; best = c; } }
  return best;
};
GV.bisectEstimate = all => { if (all < 3) return 0; const n = Math.floor(Math.log2(all)), e = Math.pow(2, n), x = all - e; return e < 3 * x ? n : n - 1; };

if (typeof module !== "undefined" && module.exports) module.exports = GV;

/* ═══ 1 · content-addressing cascade (§02) ═══════════════════════════════════════════════════ */
(function () {
  if (typeof document === "undefined" || !document.getElementById("ca-svg")) return;
  const svg = d3.select("#ca-svg"), out = document.getElementById("ca-readout");
  const inp = document.getElementById("ca-readme"), sl = document.getElementById("ca-seed"), sv = document.getElementById("ca-seedv");
  const trainPy = s => "import numpy as np\n\nrng = np.random.default_rng(" + s + ")\nprint(rng.normal())\n";
  const AT = 1767258060, MSG = "Add README and training script";
  const compute = (readme, seed) => {
    const b1 = GV.blobId(readme + "\n"), b2 = GV.blobId(trainPy(seed));
    const src = GV.treeId([{ mode: "100644", name: "train.py", id: b2 }]);
    const root = GV.treeId([{ mode: "100644", name: "README.md", id: b1 }, { mode: "40000", name: "src", id: src }]);
    return { b1, b2, src, root, c: GV.commitId(root, [], AT, AT, MSG) };
  };
  const DEF = compute("hello", 42);
  GV.caCompute = compute;
  GV.arrowDef(svg, "ca-ar", GV.C.muted);
  const boxes = [
    { k: "c", type: "commit", name: "Add README and…", x: 14, y: 118 },
    { k: "root", type: "tree", name: "/ (root)", x: 180, y: 118 },
    { k: "b1", type: "blob", name: "README.md", x: 350, y: 40 },
    { k: "src", type: "tree", name: "src/", x: 350, y: 196 },
    { k: "b2", type: "blob", name: "train.py", x: 520, y: 196 }
  ];
  const BW = 146, BH = 62, edges = [["c", "root"], ["root", "b1"], ["root", "src"], ["src", "b2"]];
  const gE = svg.append("g"), gB = svg.append("g"), gT = svg.append("g");
  const pos = Object.fromEntries(boxes.map(b => [b.k, b]));
  edges.forEach(([a, b]) => {
    const A = pos[a], B = pos[b];
    gE.append("path").attr("d", `M${A.x + BW},${A.y + BH / 2} C${A.x + BW + 25},${A.y + BH / 2} ${B.x - 25},${B.y + BH / 2} ${B.x - 2},${B.y + BH / 2}`)
      .attr("fill", "none").attr("stroke", GV.C.line).attr("stroke-width", 1.5).attr("marker-end", "url(#ca-ar)");
  });
  const nodes = boxes.map(b => {
    const g = gB.append("g").attr("transform", `translate(${b.x},${b.y})`);
    const rect = g.append("rect").attr("width", BW).attr("height", BH).attr("rx", 8).attr("fill", GV.panel).attr("stroke", GV.C.line).attr("stroke-width", 1.5);
    GV.text(g, 10, 16, b.type, { size: 10, color: b.type === "commit" ? GV.C.A : b.type === "tree" ? GV.C.good : GV.C.B, weight: 600 });
    GV.text(g, 10, 30, b.name, { size: 10.5, color: GV.C.ink });
    const h = GV.text(g, 10, 47, "", { size: 12, color: GV.C.ink, mono: true, weight: 600 });
    const was = GV.text(g, 10, 58, "", { size: 9, color: GV.C.muted, mono: true });
    return { b, rect, h, was };
  });
  const bytesNote = GV.text(gT, 14, 286, "", { size: 10.5, mono: true });
  GV.text(gT, 14, 24, "metadata held fixed: author/committer Student, 1767258060 +0000, no parent", { size: 10 });
  function draw() {
    const readme = inp.value, seed = +sl.value; sv.textContent = seed;
    const now = compute(readme, seed);
    let changed = 0;
    nodes.forEach(n => {
      const v = now[n.b.k], d = DEF[n.b.k], ch = v !== d;
      if (ch) changed++;
      n.h.text(GV.short(v));
      n.was.text(ch ? "default " + GV.short(d) : "= default");
      n.rect.transition().duration(GV.DUR).attr("stroke", ch ? GV.C.B : GV.C.line).attr("stroke-width", ch ? 2.5 : 1.5);
    });
    const len = GV.utf8(readme + "\n").length;
    bytesNote.text(`README blob hashes the bytes:  "blob ${len}␀${readme}⏎"   (${len} content bytes, UTF-8)`);
    out.innerHTML = `commit <b>${now.c}</b><br>` +
      (changed ? `${changed} of 5 objects differ from the default; the README blob ${now.b1 === DEF.b1 ? "is unchanged (shared)" : "changed"}, train.py blob ${now.b2 === DEF.b2 ? "is unchanged (shared)" : "changed"}.`
               : `all 5 objects equal the real repository: blobs ce01362, ad36062 · trees cbf3ce5, 32dae24 · commit 363ab0d.`);
  }
  inp.addEventListener("input", draw);
  sl.addEventListener("input", draw);
  document.getElementById("ca-reset").addEventListener("click", () => { inp.value = "hello"; sl.value = 42; draw(); });
  draw();
})();

/* ═══ 2 · the four pointers (§03) ════════════════════════════════════════════════════════════ */
(function () {
  if (typeof document === "undefined" || !document.getElementById("fp-svg")) return;
  const svg = d3.select("#fp-svg"), out = document.getElementById("fp-readout");
  GV.arrowDef(svg, "fp-ar", GV.C.B);
  let S, last;
  /* default = the real repository of §03: C1..C3 with f.txt = v1..v3, committed at T0+120, +180, +240 */
  function init() {
    S = { commits: [], main: -1, wt: 0, idx: 0, maxV: 3, clock: 1767258060, n: 0 };
    for (let v = 1; v <= 3; v++) mk(v);
    S.wt = S.idx = 3; last = { moved: [], msg: "default: main at C3 (7953662), f.txt = v3 everywhere" };
  }
  function mk(v) {
    S.clock += 60; S.n += 1;
    const parent = S.main >= 0 ? [S.commits[S.main].id] : [];
    const tree = GV.treeOf({ "f.txt": "v" + v + "\n" });
    const id = GV.commitId(tree, parent, S.clock, S.clock, "C" + S.n);
    S.commits.push({ id, v, msg: "C" + S.n, parent: S.main });
    S.main = S.commits.length - 1;
    return id;
  }
  const headV = () => S.commits[S.main].v;
  const reach = () => { const s = new Set(); for (let i = S.main; i >= 0; i = S.commits[i].parent) s.add(i); return s; };
  const status = () => {
    const a = S.idx !== headV() ? "M" : " ", b = S.wt !== S.idx ? "M" : " ";
    return (a + b).trim() ? `"${a}${b} f.txt"` : "(clean)";
  };
  const act = {
    edit() { S.maxV += 1; S.wt = S.maxV; last = { moved: ["wt"], msg: `edit: working tree f.txt = v${S.wt}` }; },
    add() { if (S.idx === S.wt) { last = { moved: [], msg: "git add: index already matches the working tree" }; return; } S.idx = S.wt; last = { moved: ["idx"], msg: `git add f.txt: index f.txt → blob of v${S.idx}` }; },
    commit() {
      if (S.idx === headV()) { last = { moved: [], msg: "git commit: nothing to commit (index = HEAD's tree)" }; return; }
      if (S.commits.length >= 9) { last = { moved: [], msg: "figure full: press Default" }; return; }
      const id = mk(S.idx); last = { moved: ["head"], msg: `[main ${GV.short(id)}] ${S.commits[S.main].msg}  (f.txt = v${S.idx})` };
    },
    rs() { if (S.idx === headV()) { last = { moved: [], msg: "git restore --staged: nothing staged" }; return; } S.idx = headV(); last = { moved: ["idx"], msg: `git restore --staged f.txt: index ← HEAD (v${S.idx}); working tree keeps v${S.wt}` }; },
    rw() { if (S.wt === S.idx) { last = { moved: [], msg: "git restore: working tree already matches the index" }; return; } const lost = S.wt; S.wt = S.idx; last = { moved: ["wt"], msg: `git restore f.txt: working tree ← index (v${S.wt}); unstaged v${lost} is gone` }; },
    reset(mode) {
      const p = S.commits[S.main].parent;
      if (p < 0) { last = { moved: [], msg: "fatal: HEAD~1 does not exist (main is at the root commit)" }; return; }
      S.main = p; const moved = ["head"];
      if (mode !== "soft") { S.idx = headV(); moved.push("idx"); }
      if (mode === "hard") { S.wt = headV(); moved.push("wt"); }
      last = { moved, msg: `git reset --${mode} HEAD~1: main → ${GV.short(S.commits[S.main].id)} (${S.commits[S.main].msg})` };
    }
  };
  function draw() {
    svg.selectAll("g.fp").remove();
    const g = svg.append("g").attr("class", "fp"), cols = [
      { k: "wt", t: "working tree", v: S.wt, x: 30 },
      { k: "idx", t: "index (staging area)", v: S.idx, x: 250 },
      { k: "head", t: "HEAD → main → commit", v: headV(), x: 470 }];
    cols.forEach(c => {
      const hot = last.moved.includes(c.k);
      g.append("rect").attr("x", c.x).attr("y", 20).attr("width", 180).attr("height", 84).attr("rx", 9)
        .attr("fill", GV.panel).attr("stroke", hot ? GV.C.B : GV.C.line).attr("stroke-width", hot ? 2.5 : 1.5);
      GV.text(g, c.x + 12, 40, c.t, { size: 11, color: GV.C.ink, weight: 600 });
      GV.text(g, c.x + 12, 74, "f.txt = v" + c.v, { size: 17, color: hot ? GV.C.B : GV.C.ink, mono: true, weight: 600 });
      GV.text(g, c.x + 12, 94, c.k === "head" ? "commit " + GV.short(S.commits[S.main].id) : (c.k === "idx" ? "blob " + GV.short(GV.blobId("v" + c.v + "\n")) : "a file on disk"), { size: 10, mono: true });
    });
    const st = status();
    GV.text(g, 30, 128, "git status --short:  " + st, { size: 11, color: GV.C.ink, mono: true });
    GV.text(g, 360, 128, "col 1 = index vs HEAD · col 2 = working tree vs index", { size: 10 });
    // commit row
    const R = reach(), x0 = 40, dx = Math.min(78, 600 / Math.max(1, S.commits.length - 1)), y = 220;
    S.commits.forEach((c, i) => {
      if (c.parent >= 0) {
        const px = x0 + c.parent * dx;
        g.append("line").attr("x1", px + 12).attr("y1", y).attr("x2", x0 + i * dx - 12).attr("y2", y)
          .attr("stroke", GV.C.line).attr("stroke-width", 2).attr("opacity", R.has(i) ? 1 : 0.35);
      }
    });
    S.commits.forEach((c, i) => {
      const x = x0 + i * dx, on = R.has(i), isTip = i === S.main;
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", 12).attr("fill", on ? GV.C.A : GV.panel)
        .attr("stroke", isTip ? GV.C.ink : GV.C.line).attr("stroke-width", isTip ? 2.5 : 1).attr("opacity", on ? 1 : 0.4);
      GV.text(g, x, y + 4, c.msg, { size: 9, anchor: "middle", color: on ? "#06101f" : GV.C.muted, weight: 700 });
      GV.text(g, x, y + 28, GV.short(c.id), { size: 9, anchor: "middle", mono: true, color: on ? GV.C.ink : GV.C.muted });
      GV.text(g, x, y + 40, "v" + c.v, { size: 9, anchor: "middle" });
      if (isTip) {
        const hot = last.moved.includes("head");
        g.append("rect").attr("x", x - 22).attr("y", y - 52).attr("width", 44).attr("height", 16).attr("rx", 4).attr("fill", hot ? GV.C.B : GV.C.good);
        GV.text(g, x, y - 40, "main", { size: 10, anchor: "middle", color: "#06101f", weight: 700 });
        GV.text(g, x, y - 58, "HEAD", { size: 9.5, anchor: "middle", color: GV.C.ink, weight: 600 });
        g.append("line").attr("x1", x).attr("y1", y - 36).attr("x2", x).attr("y2", y - 15).attr("stroke", hot ? GV.C.B : GV.C.good).attr("stroke-width", 1.5).attr("marker-end", hot ? "url(#fp-ar)" : null);
      }
    });
    const un = S.commits.length - R.size;
    GV.text(g, 30, 292, un ? `${un} commit${un > 1 ? "s" : ""} no longer reachable from main (faded): still in the object store and the reflog` : "all commits reachable from main", { size: 10 });
    out.innerHTML = `<b>${GV.esc(last.msg)}</b><br>status ${GV.esc(st)} · main = ${GV.short(S.commits[S.main].id)} · ${S.commits.length} commits in the store`;
  }
  const on = (id, f) => document.getElementById(id).addEventListener("click", () => { f(); draw(); });
  on("fp-edit", act.edit); on("fp-add", act.add); on("fp-commit", act.commit); on("fp-rs", act.rs); on("fp-rw", act.rw);
  on("fp-soft", () => act.reset("soft")); on("fp-mixed", () => act.reset("mixed")); on("fp-hard", () => act.reset("hard"));
  on("fp-reset", init);
  init(); draw();
})();

/* ═══ 3 · commit-DAG sandbox (§05) ═══════════════════════════════════════════════════════════ */
(function () {
  if (typeof document === "undefined" || !document.getElementById("dg-svg")) return;
  const svg = d3.select("#dg-svg"), out = document.getElementById("dg-readout"), sel = document.getElementById("dg-target");
  const NAMES = ["feature", "fix", "exp", "dev"], MAX = 24, W = 680;
  let r, selected, msg;
  GV.arrowDef(svg, "dg-ar", GV.C.muted);
  const gEdges = svg.append("g"), gNodes = svg.append("g"), gTags = svg.append("g");
  function init() {
    r = GV.newRepo(); selected = GV.commit(r);
    msg = `[main (root-commit) ${GV.short(selected)}] C1`;
  }
  function refreshSelect(prefer) {
    const names = Object.keys(r.branches), keep = prefer || sel.value;
    sel.innerHTML = names.map(n => `<option value="${n}">${n}</option>`).join("");
    sel.value = names.includes(keep) ? keep : names[0];
  }
  const where = () => (r.head.ref ? r.head.ref : "detached HEAD");
  const full = () => r.order.length >= MAX;
  const leaving = before => {        // commits that become unreachable by moving HEAD away from a detached position
    const now = GV.reachableAll(r); return [...before].filter(x => !now.has(x));
  };
  const act = {
    commit() {
      if (full()) { msg = "sandbox full (24 commits): press Default"; return; }
      const id = GV.commit(r); selected = id; msg = `[${where()} ${GV.short(id)}] ${r.commits[id].msg}`;
    },
    branch() {
      const name = NAMES.find(n => !(n in r.branches));
      if (!name) { msg = "this sandbox allows four extra branches"; return; }
      GV.branch(r, name); refreshSelect(name);
      msg = `git branch ${name}: '${name}' now points at ${GV.short(GV.headId(r))}; HEAD is still on ${where()} (switch to it to commit there)`;
    },
    switch() {
      const t = sel.value; if (!t) return;
      if (r.head.ref === t) { msg = `Already on '${t}'`; return; }
      const before = GV.reachableAll(r); GV.switchTo(r, t); const lost = leaving(before);
      msg = (lost.length ? `Warning: you are leaving ${lost.length} commit${lost.length > 1 ? "s" : ""} behind, not connected to any of your branches. ` : "") + `Switched to branch '${t}'`;
      selected = GV.headId(r);
    },
    merge() {
      const t = sel.value; if (!t) return;
      if (full()) { msg = "sandbox full (24 commits): press Default"; return; }
      const h0 = GV.headId(r), k = GV.merge(r, t, document.getElementById("dg-noff").checked), h = GV.headId(r);
      if (k === "uptodate") msg = "Already up to date.";
      else if (k === "ff") msg = `Updating ${GV.short(h0)}..${GV.short(h)}  Fast-forward (no new commit: ${where()} just moved)`;
      else { const c = r.commits[h]; msg = `Merge made by the 'ort' strategy → ${GV.short(h)} with parents ${c.parents.map(GV.short).join(", ")} (first = ${where()}, second = ${t})`; }
      selected = h;
    },
    rebase() {
      const t = sel.value; if (!t) return;
      if (full()) { msg = "sandbox full (24 commits): press Default"; return; }
      const res = GV.rebase(r, t);
      if (res.kind === "uptodate") msg = `Current branch ${where()} is up to date.`;
      else if (res.kind === "ff") msg = `${where()} was an ancestor of ${t}: fast-forwarded to ${GV.short(GV.headId(r))}, nothing replayed`;
      else if (res.kind === "merges") msg = `the commits to replay include a merge commit; git would drop the merge and replay the rest in a straight line. This sandbox only replays linear branches.`;
      else msg = `Successfully rebased ${where()} onto ${t}: ` + res.map.map(([o, n]) => `${r.commits[o].msg} ${GV.short(o)} → ${GV.short(n)}`).join(", ");
      selected = GV.headId(r);
    },
    detach() {
      if (!selected || !r.commits[selected]) return;
      GV.detach(r, selected); msg = `HEAD is now at ${GV.short(selected)} ${r.commits[selected].msg} (detached: HEAD holds a hash, not a branch name)`;
    }
  };
  function draw() {
    const ids = r.order, n = ids.length, R = GV.reachableAll(r), head = GV.headId(r);
    const dx = Math.min(56, (W - 150) / Math.max(1, n - 1));
    const X = id => 30 + r.commits[id].seq * dx, Y = id => 42 + (r.commits[id].lane % 6) * 44;
    const edges = [];
    ids.forEach(id => r.commits[id].parents.forEach(p => edges.push({ k: p + id, s: p, t: id })));
    const path = e => {
      const x1 = X(e.s), y1 = Y(e.s), x2 = X(e.t), y2 = Y(e.t);
      return y1 === y2 ? `M${x1 + 10},${y1} L${x2 - 11},${y2}` : `M${x1 + 8},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2 - 11},${y2}`;
    };
    const ej = gEdges.selectAll("path").data(edges, d => d.k);
    ej.exit().remove();
    ej.enter().append("path").attr("fill", "none").attr("stroke-width", 1.6).attr("marker-end", "url(#dg-ar)")
      .merge(ej).attr("stroke", d => R.has(d.t) ? GV.C.muted : GV.C.line).attr("opacity", d => R.has(d.t) ? 1 : 0.45)
      .transition().duration(GV.DUR).attr("d", path);
    const nj = gNodes.selectAll("g.n").data(ids, d => d);
    nj.exit().remove();
    const ne = nj.enter().append("g").attr("class", "n").style("cursor", "pointer")
      .attr("transform", d => `translate(${X(d)},${Y(d)})`)
      .on("click", (ev, d) => { selected = d; msg = `selected ${GV.short(d)} ${r.commits[d].msg}: "detach at selected" will put HEAD here`; draw(); });
    ne.append("circle").attr("r", 10);
    ne.append("text").attr("class", "m").attr("text-anchor", "middle").attr("y", 3.5).attr("font-size", 8).attr("font-weight", 700);
    ne.append("text").attr("class", "h").attr("text-anchor", "middle").attr("y", 23).attr("font-size", 8.5).attr("font-family", "SF Mono, Menlo, monospace");
    const nm = ne.merge(nj);
    nm.transition().duration(GV.DUR).attr("transform", d => `translate(${X(d)},${Y(d)})`).attr("opacity", d => R.has(d) ? 1 : 0.3);
    nm.select("circle").attr("fill", d => r.commits[d].merge ? GV.C.B : GV.C.A)
      .attr("stroke", d => d === head ? GV.C.ink : d === selected ? GV.C.good : "none")
      .attr("stroke-width", d => d === head ? 3 : 2).attr("stroke-dasharray", d => d === head ? null : "3 2");
    nm.select("text.m").attr("fill", "#06101f").text(d => r.commits[d].merge ? "M" : r.commits[d].msg);
    nm.select("text.h").attr("fill", d => R.has(d) ? GV.C.ink : GV.C.muted).text(d => GV.short(d));
    // branch + HEAD tags
    gTags.selectAll("*").remove();
    const at = {};
    Object.keys(r.branches).forEach(b => { const id = r.branches[b]; (at[id] = at[id] || []).push(b); });
    if (!r.head.ref) (at[head] = at[head] || []).push("HEAD");
    Object.keys(at).forEach(id => {
      at[id].forEach((b, i) => {
        const isHead = b === "HEAD" || b === r.head.ref, label = b === "HEAD" ? "HEAD (detached)" : (isHead ? "HEAD → " + b : b);
        const w = label.length * 6 + 10, x = X(id) - w / 2, y = Y(id) - 30 - i * 15;
        gTags.append("rect").attr("x", x).attr("y", y - 10).attr("width", w).attr("height", 13).attr("rx", 3)
          .attr("fill", isHead ? GV.C.B : GV.C.good).attr("opacity", 0.95);
        GV.text(gTags, X(id), y, label, { size: 9.5, anchor: "middle", color: "#06101f", weight: 700 });
      });
    });
    const un = n - R.size;
    out.innerHTML = `<b>${GV.esc(msg)}</b><br>` +
      `HEAD → ${r.head.ref ? r.head.ref + " → " : ""}${GV.short(head)} · ` +
      Object.keys(r.branches).map(b => `${b} ${GV.short(r.branches[b])}`).join(" · ") +
      ` · ${n} commit${n > 1 ? "s" : ""} in the store${un ? `, ${un} unreachable (faded)` : ""}`;
  }
  const on = (id, f) => document.getElementById(id).addEventListener("click", () => { f(); draw(); });
  on("dg-commit", act.commit); on("dg-branch", act.branch); on("dg-switch", act.switch);
  on("dg-merge", act.merge); on("dg-rebase", act.rebase); on("dg-detach", act.detach);
  on("dg-reset", () => { gNodes.selectAll("*").remove(); gEdges.selectAll("*").remove(); init(); refreshSelect("main"); });
  init(); refreshSelect("main"); draw();
})();

/* ═══ 4 · merge bases in a criss-cross DAG + line-level three-way merge (§06) ══════════════════ */
/* The DAG is rebuilt here with the toy repository from a fixed script; its 45 pairwise merge-base
   sets were checked against `git merge-base --all`. Scenario results are real git output (zdiff3). */
GV.mbRepo = function () {
  const r = GV.newRepo();
  GV.commit(r); GV.commit(r); GV.branch(r, "feature"); GV.commit(r);           // C1 C2 | C3 on main
  GV.switchTo(r, "feature"); GV.commit(r); GV.commit(r);                         // C4 C5 on feature
  GV.switchTo(r, "main"); GV.commit(r);                                          // C6 on main
  GV.branch(r, "x"); GV.branch(r, "y");
  GV.switchTo(r, "x"); GV.merge(r, "feature", true);                             // M1 = C6 + C5
  GV.switchTo(r, "feature"); GV.branch(r, "z"); GV.switchTo(r, "z"); GV.merge(r, "main", true); // M2 = C5 + C6
  GV.switchTo(r, "main"); GV.merge(r, "x"); GV.switchTo(r, "feature"); GV.merge(r, "z");      // fast-forwards
  GV.commit(r); GV.switchTo(r, "main"); GV.commit(r);                            // C7 on feature, C8 on main
  return r;
};
GV.scen = {
  base: ['model = "logreg"', "lr = 0.1", "seed = 0", "batch_size = 32", "test_size = 0.2", "epochs = 10"],
  sc: {
    "separated": { ours: { 3: "batch_size = 64" }, theirs: { 1: "lr = 0.01", 5: "epochs = 30" }, exit: 0,
      result: ['model = "logreg"', "lr = 0.01", "seed = 0", "batch_size = 64", "test_size = 0.2", "epochs = 30"] },
    "gap1": { ours: { 1: "lr = 0.01" }, theirs: { 3: "batch_size = 64" }, exit: 0,
      result: ['model = "logreg"', "lr = 0.01", "seed = 0", "batch_size = 64", "test_size = 0.2", "epochs = 10"] },
    "adjacent": { ours: { 2: "seed = 1" }, theirs: { 3: "batch_size = 64" }, exit: 1,
      result: ['model = "logreg"', "lr = 0.1", "<<<<<<< HEAD", "seed = 1", "batch_size = 32", "||||||| d670a93", "seed = 0", "batch_size = 32", "=======", "seed = 0", "batch_size = 64", ">>>>>>> theirs", "test_size = 0.2", "epochs = 10"] },
    "same": { ours: { 1: "lr = 0.01" }, theirs: { 1: "lr = 0.01" }, exit: 0,
      result: ['model = "logreg"', "lr = 0.01", "seed = 0", "batch_size = 32", "test_size = 0.2", "epochs = 10"] },
    "conflict": { ours: { 1: "lr = 0.05" }, theirs: { 1: "lr = 0.01" }, exit: 1,
      result: ['model = "logreg"', "<<<<<<< HEAD", "lr = 0.05", "||||||| d670a93", "lr = 0.1", "=======", "lr = 0.01", ">>>>>>> theirs", "seed = 0", "batch_size = 32", "test_size = 0.2", "epochs = 10"] },
    "delete-vs-edit": { ours: { 5: null }, theirs: { 5: "epochs = 30" }, exit: 1,
      result: ['model = "logreg"', "lr = 0.1", "seed = 0", "batch_size = 32", "test_size = 0.2", "<<<<<<< HEAD", "||||||| d670a93", "epochs = 10", "=======", "epochs = 30", ">>>>>>> theirs"] }
  }
};
(function () {
  if (typeof document === "undefined" || !document.getElementById("mb-svg")) return;
  const svg = d3.select("#mb-svg"), out = document.getElementById("mb-readout");
  const sx = document.getElementById("mb-x"), sy = document.getElementById("mb-y"), ssc = document.getElementById("mb-sc");
  const r = GV.mbRepo(), ids = r.order;
  const label = id => { const c = r.commits[id]; return c.merge ? (c.parents[0] === r.branches.y ? "M1" : "M2") : c.msg; };
  const P = { C1: [0, 1], C2: [1, 1], C3: [2, 0], C6: [3, 0], M1: [4, 0], C8: [5, 0], C4: [2, 2], C5: [3, 2], M2: [4, 2], C7: [5, 2] };
  const X = id => 60 + P[label(id)][0] * 104, Y = id => 34 + P[label(id)][1] * 62;
  const tipName = id => (id === r.branches.main ? " · main" : id === r.branches.feature ? " · feature" : "");
  const opts = ids.map(id => `<option value="${id}">${label(id)} ${GV.short(id)}${tipName(id)}</option>`).join("");
  sx.innerHTML = opts; sy.innerHTML = opts;
  sx.value = r.branches.main; sy.value = r.branches.feature;
  GV.arrowDef(svg, "mb-ar", GV.C.muted);
  const gTop = svg.append("g"), gBot = svg.append("g");
  function drawTop() {
    gTop.selectAll("*").remove();
    const a = sx.value, b = sy.value, A = GV.ancestors(r, a), B = GV.ancestors(r, b);
    const common = new Set([...A].filter(x => B.has(x))), best = new Set(GV.mergeBases(r, a, b));
    ids.forEach(id => r.commits[id].parents.forEach(p => {
      const x1 = X(p), y1 = Y(p), x2 = X(id), y2 = Y(id), cross = r.commits[id].merge && p !== r.commits[id].parents[0];
      gTop.append("path").attr("d", `M${x1 + 13},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2 - 14},${y2}`)
        .attr("fill", "none").attr("stroke", cross ? GV.C.B : GV.C.muted).attr("stroke-width", 1.5).attr("opacity", cross ? 0.8 : 1).attr("marker-end", "url(#mb-ar)");
    }));
    ids.forEach(id => {
      const x = X(id), y = Y(id), isB = best.has(id), isC = common.has(id);
      gTop.append("circle").attr("cx", x).attr("cy", y).attr("r", 13)
        .attr("fill", isB ? GV.C.B : isC ? "rgba(91,156,255,.35)" : GV.panel)
        .attr("stroke", isB ? GV.C.ink : (id === a || id === b) ? GV.C.good : GV.C.line).attr("stroke-width", isB || id === a || id === b ? 2.5 : 1.2);
      GV.text(gTop, x, y + 4, label(id), { size: 9.5, anchor: "middle", color: isB ? "#06101f" : GV.C.ink, weight: 700 });
      GV.text(gTop, x, y + 25, GV.short(id), { size: 8.5, anchor: "middle", mono: true });
      const tags = [];
      if (id === a) tags.push("X"); if (id === b) tags.push("Y");
      if (id === r.branches.main) tags.push("main"); if (id === r.branches.feature) tags.push("feature");
      if (tags.length) GV.text(gTop, x + 17, y - 10, tags.join(" · "), { size: 9.5, color: GV.C.good, weight: 600 });
    });
    GV.text(gTop, 600, 34, "■ best common", { size: 9.5, color: GV.C.B });
    GV.text(gTop, 600, 48, "  ancestor", { size: 9.5, color: GV.C.B });
    GV.text(gTop, 600, 66, "■ common", { size: 9.5, color: GV.C.A });
    GV.text(gTop, 600, 80, "  ancestor", { size: 9.5, color: GV.C.A });
    const bl = [...best].map(id => `${label(id)} ${GV.short(id)}`).join(", ");
    let note;
    if (a === b) note = "X and Y are the same commit.";
    else if (best.has(a)) note = "X is an ancestor of Y: merging Y into X would fast-forward.";
    else if (best.has(b)) note = "Y is an ancestor of X: X already contains Y (\"Already up to date\").";
    else if (best.size > 1) note = "criss-cross: more than one best common ancestor; ort first merges them into a virtual base.";
    else note = "one best common ancestor: the three-way merge uses it as the base.";
    return `git merge-base --all ${label(a)} ${label(b)} → <b>${bl}</b> · ${common.size} common ancestor${common.size > 1 ? "s" : ""} · ${note}`;
  }
  function drawBot() {
    gBot.selectAll("*").remove();
    const S = GV.scen, sc = S.sc[ssc.value], y0 = 216, rh = 13, cw = 162;
    const side = ch => S.base.map((t, i) => (i in ch ? { t: ch[i], ch: true } : { t, ch: false }));
    const cols = [
      { t: "base (d670a93)", lines: S.base.map(t => ({ t })) },
      { t: "ours (HEAD)", lines: side(sc.ours), c: GV.C.A },
      { t: "theirs", lines: side(sc.theirs), c: GV.C.B },
      { t: sc.exit ? "git result: CONFLICT" : "git result: clean", lines: sc.result.map(t => ({ t })), res: true }];
    gBot.append("line").attr("x1", 10).attr("x2", 670).attr("y1", y0 - 18).attr("y2", y0 - 18).attr("stroke", GV.C.line);
    const oursCh = new Set(Object.values(sc.ours)), thCh = new Set(Object.values(sc.theirs));
    cols.forEach((c, j) => {
      const x = 14 + j * cw;
      GV.text(gBot, x, y0, c.t, { size: 10.5, weight: 600, color: c.res ? (sc.exit ? GV.C.bad : GV.C.good) : (c.c || GV.C.ink) });
      c.lines.forEach((l, i) => {
        const y = y0 + 16 + i * rh;
        let col = GV.C.muted, t = l.t;
        if (t === null) { t = "(line deleted)"; col = GV.C.bad; }
        else if (c.res) {
          if (/^(<<<<<<<|\|\|\|\|\|\|\||=======|>>>>>>>)/.test(t)) col = GV.C.bad;
          else if (oursCh.has(t) && !S.base.includes(t)) col = thCh.has(t) ? GV.C.good : GV.C.A;
          else if (thCh.has(t) && !S.base.includes(t)) col = GV.C.B;
          else col = GV.C.ink;
        } else if (l.ch) col = c.c;
        GV.text(gBot, x, y, t, { size: 10, mono: true, color: col });
      });
    });
    return `scenario: ${ssc.options[ssc.selectedIndex].text} → real git ${sc.exit ? "stopped with a conflict (exit 1)" : "merged cleanly (exit 0)"}`;
  }
  function draw() { out.innerHTML = drawTop() + "<br>" + drawBot(); }
  [sx, sy, ssc].forEach(e => e.addEventListener("change", draw));
  draw();
})();

/* ═══ 5 · merge vs rebase from one real repository (§07) ═════════════════════════════════════ */
(function () {
  if (typeof document === "undefined" || !document.getElementById("mr-svg")) return;
  const svg = d3.select("#mr-svg"), out = document.getElementById("mr-readout"), cb = document.getElementById("mr-patch");
  GV.arrowDef(svg, "mr-ar", GV.C.muted); GV.arrowDef(svg, "mr-ar2", GV.C.B);
  const PID = { cdc6627: "fcc9f48c", "7f5a18b": "a921363f", b38c7e6: "fcc9f48c", "64f3344": "a921363f" };
  let view = "both";
  const g = svg.append("g");
  function panel(x0, title, after, kind) {
    const L = x0 + 28, dx = 60, y0 = 120, y1 = 215;
    const nodes = [{ id: "7de8772", x: 0, y: y0, m: "config" }, { id: "290ee3c", x: 1, y: y0, m: "loader" }, { id: "7a583f0", x: 2, y: y0, m: "batch 64" }];
    const edges = [["7de8772", "290ee3c"], ["290ee3c", "7a583f0"]];
    const tags = {}, dash = [];
    let fade = false;
    if (kind === "merge" || !after) {
      nodes.push({ id: "cdc6627", x: 2, y: y1, m: "lr 0.01" }, { id: "7f5a18b", x: 3, y: y1, m: "epochs 30" });
      edges.push(["290ee3c", "cdc6627"], ["cdc6627", "7f5a18b"]);
      tags["7f5a18b"] = "tune";
      if (after) { nodes.push({ id: "05bb6e3", x: 4, y: y0, m: "merge", merge: true, nw: true, tree: "2505393" }); edges.push(["7a583f0", "05bb6e3"], ["7f5a18b", "05bb6e3"]); tags["05bb6e3"] = "main"; }
      else tags["7a583f0"] = "main";
    } else {
      fade = true;
      nodes.push({ id: "cdc6627", x: 2, y: y1, m: "lr 0.01", old: true }, { id: "7f5a18b", x: 3, y: y1, m: "epochs 30", old: true });
      edges.push(["290ee3c", "cdc6627"], ["cdc6627", "7f5a18b"]);
      nodes.push({ id: "b38c7e6", x: 3, y: y0, m: "lr 0.01", nw: true }, { id: "64f3344", x: 4, y: y0, m: "epochs 30", nw: true, tree: "2505393" });
      edges.push(["7a583f0", "b38c7e6"], ["b38c7e6", "64f3344"]);
      dash.push(["cdc6627", "b38c7e6"], ["7f5a18b", "64f3344"]);
      tags["7a583f0"] = "main"; tags["64f3344"] = "tune";
    }
    const pos = Object.fromEntries(nodes.map(n => [n.id, n]));
    g.append("rect").attr("x", x0).attr("y", 8).attr("width", 330).attr("height", 304).attr("rx", 10).attr("fill", "none").attr("stroke", GV.C.line);
    GV.text(g, x0 + 12, 28, title + (after ? "" : "  (before)"), { size: 11.5, color: GV.C.ink, weight: 600 });
    edges.forEach(([s, t]) => {
      const a = pos[s], b = pos[t], x1 = L + a.x * dx, x2 = L + b.x * dx;
      g.append("path").attr("d", `M${x1 + 11},${a.y} C${(x1 + x2) / 2},${a.y} ${(x1 + x2) / 2},${b.y} ${x2 - 12},${b.y}`)
        .attr("fill", "none").attr("stroke", GV.C.muted).attr("stroke-width", 1.5).attr("opacity", (a.old || b.old) ? 0.35 : 1).attr("marker-end", "url(#mr-ar)");
    });
    dash.forEach(([s, t]) => {
      const a = pos[s], b = pos[t];
      g.append("line").attr("x1", L + a.x * dx + 6).attr("y1", a.y - 12).attr("x2", L + b.x * dx - 6).attr("y2", b.y + 13)
        .attr("stroke", GV.C.B).attr("stroke-dasharray", "4 3").attr("stroke-width", 1.3).attr("marker-end", "url(#mr-ar2)");
    });
    nodes.forEach(n => {
      const x = L + n.x * dx, o = n.old ? 0.4 : 1;
      g.append("circle").attr("cx", x).attr("cy", n.y).attr("r", 11).attr("fill", n.merge ? GV.C.B : GV.C.A).attr("opacity", o)
        .attr("stroke", n.nw ? GV.C.B : "none").attr("stroke-width", 3);
      GV.text(g, x, n.y + 25, n.id, { size: 9, anchor: "middle", mono: true, color: n.nw ? GV.C.B : GV.C.ink }).attr("opacity", o);
      GV.text(g, x, n.y + 37, n.m, { size: 8.5, anchor: "middle" }).attr("opacity", o);
      if (cb.checked && PID[n.id]) GV.text(g, x, n.y + 49, "Δ " + PID[n.id], { size: 8.5, anchor: "middle", mono: true, color: GV.C.good }).attr("opacity", n.old ? 0.7 : 1);
      if (n.tree) GV.text(g, x, n.y - 30, "tree " + n.tree, { size: 8.5, anchor: "middle", mono: true, color: GV.C.good });
      if (tags[n.id]) {
        const w = tags[n.id].length * 6.5 + 10;
        g.append("rect").attr("x", x - w / 2).attr("y", n.y - 25).attr("width", w).attr("height", 12).attr("rx", 3).attr("fill", GV.C.good);
        GV.text(g, x, n.y - 16, tags[n.id], { size: 9, anchor: "middle", color: "#06101f", weight: 700 });
      }
    });
    if (fade) GV.text(g, x0 + 12, 296, "faded: originals, unreachable but in the reflog", { size: 9.5 });
    if (kind === "merge" && after) GV.text(g, x0 + 12, 296, "05bb6e3 has two parents: 7a583f0 and 7f5a18b", { size: 9.5 });
  }
  function draw() {
    g.selectAll("*").remove();
    const m = view === "merge" || view === "both", rb = view === "rebase" || view === "both";
    panel(4, "git merge tune  (on main)", m, "merge");
    panel(346, "git rebase main  (on tune)", rb, "rebase");
    const parts = [];
    parts.push(m ? "merge: one new commit 05bb6e3, nothing rewritten" : "merge future not applied");
    parts.push(rb ? "rebase: cdc6627 → b38c7e6, 7f5a18b → 64f3344 (new hashes, same changes)" : "rebase future not applied");
    if (m && rb) parts.push("both tips hold tree 2505393: same snapshot, different history");
    if (cb.checked) parts.push("Δ = zero-context patch id: identical for each original and its replay");
    out.innerHTML = `<b>${view}</b> · ` + parts.join(" · ");
  }
  ["before", "merge", "rebase", "both"].forEach(v => document.getElementById("mr-" + v).addEventListener("click", () => { view = v; draw(); }));
  cb.addEventListener("change", draw);
  draw();
})();

/* ═══ 6 · the reflog safety net (§08) — hashes and messages from the real run ═════════════════ */
(function () {
  if (typeof document === "undefined" || !document.getElementById("rl-svg")) return;
  const svg = d3.select("#rl-svg"), out = document.getElementById("rl-readout");
  const C3 = [{ id: "1b283b8", m: "Load raw data" }, { id: "1fb8dc6", m: "Clean missing values" }, { id: "e06ede0", m: "Fit baseline model" }];
  const idx = Object.fromEntries(C3.map((c, i) => [c.id, i]));
  let S, last;
  function init() {
    S = { main: "e06ede0", rescue: null, log: [
      { h: "e06ede0", m: "commit: Fit baseline model" }, { h: "1fb8dc6", m: "commit: Clean missing values" }, { h: "1b283b8", m: "commit (initial): Load raw data" }] };
    last = "default: main at e06ede0 with all three commits reachable";
  }
  const reach = () => { const s = new Set(); [S.main, S.rescue].forEach(t => { if (t) for (let i = idx[t]; i >= 0; i--) s.add(C3[i].id); }); return s; };
  const act = {
    reset() {
      const i = idx[S.main]; if (i < 2) { last = "fatal: ambiguous argument 'HEAD~2': unknown revision (main has fewer than two ancestors)"; return; }
      S.main = C3[i - 2].id; S.log.unshift({ h: S.main, m: "reset: moving to HEAD~2" });
      last = `HEAD is now at ${S.main} ${C3[idx[S.main]].m}`;
    },
    branch() {
      if (S.rescue) { last = "fatal: a branch named 'rescue' already exists"; return; }
      S.rescue = S.log[1].h; last = `git branch rescue HEAD@{1} → rescue at ${S.rescue} (HEAD's reflog is unchanged: HEAD did not move)`;
    },
    back() {
      const h = S.log[1].h; S.main = h; S.log.unshift({ h, m: "reset: moving to HEAD@{1}" });
      last = `HEAD is now at ${h} ${C3[idx[h]].m}`;
    }
  };
  function draw() {
    svg.selectAll("*").remove();
    const R = reach();
    GV.text(svg, 20, 24, "git log --oneline --all (reachable) vs the object store", { size: 11, color: GV.C.ink, weight: 600 });
    C3.forEach((c, i) => {
      const x = 50 + i * 100, y = 120, on = R.has(c.id);
      if (i) svg.append("line").attr("x1", x - 88).attr("x2", x - 13).attr("y1", y).attr("y2", y).attr("stroke", GV.C.muted).attr("opacity", on ? 1 : 0.35).attr("stroke-width", 2);
      svg.append("circle").attr("cx", x).attr("cy", y).attr("r", 13).attr("fill", GV.C.A).attr("opacity", on ? 1 : 0.3);
      GV.text(svg, x, y + 30, c.id, { size: 9.5, anchor: "middle", mono: true, color: on ? GV.C.ink : GV.C.muted });
      GV.text(svg, x, y + 43, c.m, { size: 8.5, anchor: "middle" });
      const tags = [];
      if (S.main === c.id) tags.push("HEAD → main"); if (S.rescue === c.id) tags.push("rescue");
      tags.forEach((t, j) => {
        const w = t.length * 6.2 + 10;
        svg.append("rect").attr("x", x - w / 2).attr("y", y - 34 - j * 16).attr("width", w).attr("height", 13).attr("rx", 3).attr("fill", t[0] === "H" ? GV.C.B : GV.C.good);
        GV.text(svg, x, y - 24 - j * 16, t, { size: 9, anchor: "middle", color: "#06101f", weight: 700 });
      });
    });
    const un = C3.filter(c => !R.has(c.id)).length;
    GV.text(svg, 20, 210, un ? `${un} commit${un > 1 ? "s" : ""} unreachable (faded) — still in .git/objects, named by the reflog` : "every commit reachable from a branch", { size: 10, color: un ? GV.C.B : GV.C.muted });
    GV.text(svg, 350, 24, "git reflog (HEAD), newest first", { size: 11, color: GV.C.ink, weight: 600 });
    S.log.slice(0, 12).forEach((e, i) => {
      const hot = !R.has(e.h);
      GV.text(svg, 350, 48 + i * 19, `${e.h} HEAD@{${i}}: ${e.m}`, { size: 10, mono: true, color: hot ? GV.C.B : GV.C.ink });
    });
    out.innerHTML = `<b>${GV.esc(last)}</b><br>main = ${S.main}${S.rescue ? " · rescue = " + S.rescue : ""} · HEAD@{1} = ${S.log[1].h} · reflog entries: ${S.log.length}`;
  }
  const on = (id, f) => document.getElementById(id).addEventListener("click", () => { f(); draw(); });
  on("rl-reset", act.reset); on("rl-branch", act.branch); on("rl-back", act.back); on("rl-default", init);
  init(); draw();
})();

/* ═══ 7 · range expressions as sets (§11) — the real five-commit history of §05 ═══════════════ */
(function () {
  if (typeof document === "undefined" || !document.getElementById("rg-svg")) return;
  const svg = d3.select("#rg-svg"), out = document.getElementById("rg-readout"), se = document.getElementById("rg-expr");
  const N = [
    { id: "7de8772", p: [], t: 1, m: "Add training config", x: 70, y: 115 },
    { id: "290ee3c", p: ["7de8772"], t: 2, m: "Add data loader", x: 190, y: 115 },
    { id: "cdc6627", p: ["290ee3c"], t: 3, m: "Lower learning rate", x: 330, y: 170 },
    { id: "7f5a18b", p: ["cdc6627"], t: 4, m: "Train for more epochs", x: 470, y: 170 },
    { id: "7a583f0", p: ["290ee3c"], t: 5, m: "Double batch size", x: 330, y: 60 }];
  const r = { commits: Object.fromEntries(N.map(n => [n.id, { parents: n.p }])) };
  const A = GV.ancestors(r, "7a583f0"), B = GV.ancestors(r, "7f5a18b");
  const sets = {
    ab: [...B].filter(x => !A.has(x)), ba: [...A].filter(x => !B.has(x)),
    sym: [...A, ...B].filter(x => !(A.has(x) && B.has(x))), a: [...A], b: [...B], mb: GV.mergeBases(r, "7a583f0", "7f5a18b") };
  GV.rgSets = sets;
  GV.arrowDef(svg, "rg-ar", GV.C.muted);
  const pos = Object.fromEntries(N.map(n => [n.id, n]));
  function draw() {
    svg.selectAll("g.rg").remove();
    const g = svg.append("g").attr("class", "rg"), S = new Set(sets[se.value]);
    N.forEach(n => n.p.forEach(p => { const a = pos[p];
      g.append("path").attr("d", `M${a.x + 13},${a.y} C${(a.x + n.x) / 2},${a.y} ${(a.x + n.x) / 2},${n.y} ${n.x - 14},${n.y}`)
        .attr("fill", "none").attr("stroke", GV.C.muted).attr("stroke-width", 1.5).attr("marker-end", "url(#rg-ar)"); }));
    N.forEach(n => {
      const on = S.has(n.id);
      g.append("circle").attr("cx", n.x).attr("cy", n.y).attr("r", 13).attr("fill", on ? GV.C.B : GV.panel).attr("stroke", on ? GV.C.ink : GV.C.line).attr("stroke-width", on ? 2 : 1.3);
      GV.text(g, n.x, n.y + 27, n.id, { size: 9.5, anchor: "middle", mono: true, color: on ? GV.C.ink : GV.C.muted });
      GV.text(g, n.x, n.y + 39, n.m, { size: 8.5, anchor: "middle" });
      const mark = (A.has(n.id) ? "∈main " : "") + (B.has(n.id) ? "∈tune" : "");
      GV.text(g, n.x, n.y - 18, mark, { size: 8.5, anchor: "middle", color: GV.C.A });
    });
    [["7a583f0", "main"], ["7f5a18b", "tune"]].forEach(([id, t]) => {
      const n = pos[id]; g.append("rect").attr("x", n.x + 18).attr("y", n.y - 7).attr("width", 36).attr("height", 14).attr("rx", 3).attr("fill", GV.C.good);
      GV.text(g, n.x + 36, n.y + 4, t, { size: 9.5, anchor: "middle", color: "#06101f", weight: 700 });
    });
    const order = [...S].sort((x, y) => pos[y].t - pos[x].t);
    const formula = { ab: "anc(tune) ∖ anc(main)", ba: "anc(main) ∖ anc(tune)", sym: "(anc(main) ∪ anc(tune)) ∖ (anc(main) ∩ anc(tune))", a: "anc(main)", b: "anc(tune)", mb: "best common ancestors" }[se.value];
    GV.text(g, 20, 220, formula, { size: 10.5, mono: true, color: GV.C.ink });
    out.innerHTML = `<b>${GV.esc(se.options[se.selectedIndex].text)}</b> = ${formula} → ${order.length} commit${order.length === 1 ? "" : "s"}: ` + order.map(id => `${id} ${pos[id].m}`).join(" · ");
  }
  se.addEventListener("change", draw);
  draw();
})();

/* ═══ 8 · bisect as binary search (§11) ═════════════════════════════════════════════════════ */
(function () {
  if (typeof document === "undefined" || !document.getElementById("bs-svg")) return;
  const svg = d3.select("#bs-svg"), out = document.getElementById("bs-readout");
  const sn = document.getElementById("bs-n"), sk = document.getElementById("bs-k");
  let S;
  function init() {
    const n = +sn.value; sk.max = n; if (+sk.value > n) sk.value = n;
    document.getElementById("bs-nv").textContent = n; document.getElementById("bs-kv").textContent = sk.value;
    S = { n, k: +sk.value, g: 1, b: n, tested: {}, msgs: [] };
    S.next = S.b - S.g > 1 ? GV.bisectPick(S.g, S.b) : null;
    if (S.next) S.msgs.push(msgFor(S.g, S.b, S.next));
  }
  function msgFor(g, b, c) {
    const all = b - g, nr = all - (c - g) - 1, e = GV.bisectEstimate(all);
    return `Bisecting: ${nr} revision${nr === 1 ? "" : "s"} left to test after this (roughly ${e} step${e === 1 ? "" : "s"})  [C${c}]`;
  }
  function step() {
    if (!S.next) return;
    const c = S.next, bad = c >= S.k; S.tested[c] = bad ? "bad" : "good";
    if (bad) S.b = c; else S.g = c;
    S.next = S.b - S.g > 1 ? GV.bisectPick(S.g, S.b) : null;
    if (S.next) S.msgs.push(msgFor(S.g, S.b, S.next));
  }
  function draw() {
    svg.selectAll("*").remove();
    const n = S.n, x0 = 30, span = 620, dx = span / (n - 1), X = i => x0 + (i - 1) * dx, w = Math.max(3, Math.min(14, dx * 0.75)), y = 80;
    svg.append("rect").attr("x", X(S.g) + dx / 2 - w / 2).attr("y", y - 14).attr("width", Math.max(2, X(S.b) - X(S.g) - dx / 2 + w)).attr("height", 64)
      .attr("fill", "rgba(91,156,255,.12)").attr("stroke", GV.C.A).attr("stroke-dasharray", "3 3");
    for (let i = 1; i <= n; i++) {
      const st = i === 1 ? "good" : (i === n && !S.tested[i] ? "bad" : S.tested[i]);
      svg.append("rect").attr("x", X(i) - w / 2).attr("y", y).attr("width", w).attr("height", 36).attr("rx", 2)
        .attr("fill", st === "good" ? GV.C.good : st === "bad" ? GV.C.bad : GV.C.line)
        .attr("stroke", i === S.next ? GV.C.ink : "none").attr("stroke-width", 2);
      if (S.tested[i] || i === 1 || i === n) GV.text(svg, X(i), y + 50, "C" + i, { size: 8.5, anchor: "middle", color: GV.C.ink });
    }
    if (S.next) {
      svg.append("path").attr("d", `M${X(S.next) - 6},${y - 26} L${X(S.next) + 6},${y - 26} L${X(S.next)},${y - 17} z`).attr("fill", GV.C.ink);
      GV.text(svg, X(S.next), y - 31, "test C" + S.next, { size: 9.5, anchor: "middle", color: GV.C.ink });
    } else {
      GV.text(svg, X(S.b), y - 22, "first bad: C" + S.b, { size: 10.5, anchor: "middle", color: GV.C.bad, weight: 700 });
    }
    GV.text(svg, x0, 22, `C1 known good · C${n} known bad · candidates N = ${n - 1} · interval (C${S.g}, C${S.b}]`, { size: 10.5, color: GV.C.ink });
    const used = Object.keys(S.tested).length, N = n - 1, lo = Math.floor(Math.log2(N)), hi = Math.ceil(Math.log2(N));
    const bx = 160, bw = 420, by = 168, sc = bw / Math.max(hi, n - 2);
    GV.text(svg, x0, by + 12, "tests used", { size: 10 });
    svg.append("rect").attr("x", bx).attr("y", by).attr("width", Math.max(1, used * sc)).attr("height", 14).attr("fill", GV.C.A);
    GV.text(svg, bx + used * sc + 6, by + 11, String(used), { size: 10, color: GV.C.ink });
    GV.text(svg, x0, by + 36, "⌈log₂ N⌉ bound", { size: 10 });
    svg.append("rect").attr("x", bx).attr("y", by + 24).attr("width", hi * sc).attr("height", 14).attr("fill", GV.C.B);
    GV.text(svg, bx + hi * sc + 6, by + 35, `${hi}  (⌊log₂ N⌋ = ${lo})`, { size: 10, color: GV.C.ink });
    GV.text(svg, x0, by + 60, "linear scan", { size: 10 });
    svg.append("rect").attr("x", bx).attr("y", by + 48).attr("width", (n - 2) * sc).attr("height", 14).attr("fill", GV.C.line);
    GV.text(svg, bx + (n - 2) * sc + 6, by + 59, `up to ${n - 2}`, { size: 10 });
    const lastMsg = S.next ? S.msgs[S.msgs.length - 1] : `C${S.b} is the first bad commit (${used} test${used === 1 ? "" : "s"})`;
    out.innerHTML = `<b>${GV.esc(lastMsg)}</b><br>tests so far ${used} · N = ${N} candidates · bound ⌈log₂ ${N}⌉ = ${hi}` + (S.next ? "" : (used <= hi && used >= lo ? " · within ⌊log₂ N⌋…⌈log₂ N⌉" : ""));
  }
  sn.addEventListener("input", () => { init(); draw(); });
  sk.addEventListener("input", () => { init(); draw(); });
  document.getElementById("bs-step").addEventListener("click", () => { step(); draw(); });
  document.getElementById("bs-run").addEventListener("click", () => { while (S.next) step(); draw(); });
  document.getElementById("bs-reset").addEventListener("click", () => { sn.value = 32; sk.max = 32; sk.value = 21; init(); draw(); });
  init(); draw();
})();

/* ═══ 9 · packfiles and delta chains (§13) — numbers from `git verify-pack -v` on the real repos ═ */
GV.pk = {
  shrink: { loose: 532480, pack: 40661, idx: 1912, vers: [
    ["128a5c9", 186890, 38294, 0, null], ["f3e0ea1", 186868, 51, 1, 0], ["809ffc8", 186846, 50, 2, 1], ["4217620", 186823, 48, 3, 2],
    ["5f6250a", 186800, 45, 4, 3], ["9d523e5", 186777, 46, 5, 4], ["ac33dba", 186754, 56, 5, 4], ["4d62529", 186731, 45, 6, 6],
    ["e64e922", 186708, 42, 7, 7], ["af97595", 186686, 53, 7, 7]] },
  grow: { loose: 532480, pack: 40547, idx: 1912, vers: [
    ["128a5c9", 186890, 25, 1, 9], ["88c9af7", 186926, 25, 1, 9], ["d407890", 186962, 25, 1, 9], ["596d200", 186998, 25, 1, 9],
    ["aefe43f", 187034, 26, 1, 9], ["84cbda1", 187070, 26, 1, 9], ["f4708a9", 187106, 26, 1, 9], ["7edd723", 187142, 25, 1, 9],
    ["1839fd1", 187178, 25, 1, 9], ["0030383", 187216, 38382, 0, null]] }
};
(function () {
  if (typeof document === "undefined" || !document.getElementById("pk-svg")) return;
  const svg = d3.select("#pk-svg"), out = document.getElementById("pk-readout"), se = document.getElementById("pk-exp");
  GV.arrowDef(svg, "pk-ar", GV.C.B);
  const kb = b => (b / 1024).toFixed(1) + " KiB";
  function draw() {
    svg.selectAll("g.pk").remove();
    const g = svg.append("g").attr("class", "pk"), D = GV.pk[se.value];
    const raw = D.vers.reduce((s, v) => s + v[1], 0);
    const bars = [["raw content (10 versions)", raw, GV.C.line], ["loose objects (zlib each)", D.loose, GV.C.A], ["packfile", D.pack, GV.C.good]];
    const xs = d3.scaleLog().domain([10000, 3000000]).range([0, 200]);
    GV.text(g, 14, 24, "total size (log scale)", { size: 11, color: GV.C.ink, weight: 600 });
    bars.forEach(([t, v, c], i) => {
      const y = 46 + i * 52;
      GV.text(g, 14, y, t, { size: 10 });
      g.append("rect").attr("x", 14).attr("y", y + 6).attr("width", xs(v)).attr("height", 16).attr("fill", c);
      GV.text(g, 20 + xs(v), y + 19, kb(v), { size: 10, color: GV.C.ink, mono: true });
    });
    GV.text(g, 14, 214, `pack ≈ ${(raw / D.pack).toFixed(0)}× smaller than raw`, { size: 10.5, color: GV.C.good });
    GV.text(g, 14, 232, `(+ ${kb(D.idx)} index file)`, { size: 9.5 });
    // chain diagram
    const x0 = 300, dx = 40, y = 190;
    GV.text(g, x0, 24, "versions in commit order: whole object vs deltas", { size: 11, color: GV.C.ink, weight: 600 });
    D.vers.forEach((v, i) => {
      if (v[4] === null) return;
      const x1 = x0 + i * dx + 14, x2 = x0 + v[4] * dx + 14, h = 18 + Math.abs(i - v[4]) * 9;
      g.append("path").attr("d", `M${x1},${y - 14} C${x1},${y - 14 - h} ${x2},${y - 14 - h} ${x2},${y - 16}`)
        .attr("fill", "none").attr("stroke", GV.C.B).attr("stroke-width", 1.2).attr("opacity", 0.8).attr("marker-end", "url(#pk-ar)");
    });
    D.vers.forEach((v, i) => {
      const x = x0 + i * dx, whole = v[4] === null;
      g.append("rect").attr("x", x).attr("y", y - 14).attr("width", 28).attr("height", 28).attr("rx", 4)
        .attr("fill", whole ? GV.C.good : GV.panel).attr("stroke", whole ? GV.C.ink : GV.C.B).attr("stroke-width", whole ? 2 : 1.2);
      GV.text(g, x + 14, y + 4, "v" + (i + 1), { size: 9.5, anchor: "middle", color: whole ? "#06101f" : GV.C.ink, weight: 700 });
      GV.text(g, x + 14, y + 30, whole ? "whole" : "d=" + v[3], { size: 8.5, anchor: "middle", color: whole ? GV.C.good : GV.C.muted });
      GV.text(g, x + 14, y + 42, whole ? kb(v[2]) : v[2] + " B", { size: 8, anchor: "middle", mono: true });
    });
    GV.text(g, x0, 272, "arrow: delta → its base · d = delta depth · sizes = bytes in the pack", { size: 9.5 });
    const wi = D.vers.findIndex(v => v[4] === null), dep = Math.max(...D.vers.map(v => v[3]));
    out.innerHTML = `<b>${se.value === "shrink" ? "file shrinks" : "file grows"}</b>: v${wi + 1} (${D.vers[wi][0]}, the largest version) stored whole · 9 deltas of ${Math.min(...D.vers.filter(v => v[4] !== null).map(v => v[2]))}–${Math.max(...D.vers.filter(v => v[4] !== null).map(v => v[2]))} bytes · max chain depth ${dep} · raw ${kb(raw)} → loose ${kb(D.loose)} → pack ${kb(D.pack)}`;
  }
  se.addEventListener("change", draw);
  draw();
})();
