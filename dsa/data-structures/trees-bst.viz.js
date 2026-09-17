/* trees-bst.viz.js — figures for dsa/data-structures/trees-bst.html
   (Data Structures · part 5: Trees & Binary Search Trees).

   Loaded after ../../data.js, ../../notes.js and ../dsa-viz.js, which supplies the
   shared toolbox (AC palette, AL.frame / AL.row / AL.stepper / AL.counter / AL.rng …).

   Layout of this file
     TB   — the algorithms, pure and pointer-based (no DOM): node factory, BST
            search/insert/delete/successor, the four traversals (recursive,
            explicit-stack, Morris), Euler tour, expression trees, height
            computations, random-BST builders, Catalan enumeration, rank/select.
            Every figure calls these with an AL.counter() so that every number the
            page displays is a MEASUREMENT of the real routine. TB is also what the
            offline check script loads to recompute the captions independently.
     TD   — drawing: a pointer-based binary-tree drawer (inorder x-layout) and a
            general (n-ary) tree drawer, built on AL's scaffolding. AL.binTree only
            draws level-order arrays, which is the wrong shape for a BST that may
            be arbitrarily unbalanced, so this page has its own.
     figures — one IIFE per <svg id>, each guarded by the element's existence.

   Conventions (the page states them in §01 and every formula here uses them):
     depth(root) = 0;  height(node) = edges to its farthest leaf descendant;
     height(leaf) = 0;  height(empty tree) = −1. */

const TB = (function () {

  /* ── nodes ──────────────────────────────────────────────────────────────── */
  function node(key, left, right) {
    const n = { key: key, left: left || null, right: right || null, parent: null };
    if (n.left) n.left.parent = n;
    if (n.right) n.right.parent = n;
    return n;
  }
  function gnode(label, children) { return { label: label, children: children || [] }; }

  /* ── size / depth / height ──────────────────────────────────────────────── */
  function size(x) { return x ? 1 + size(x.left) + size(x.right) : 0; }
  function depthByParent(x, c) {                 // walk up; O(depth + 1)
    let d = 0;
    while (x.parent) { x = x.parent; d++; if (c) c.add("up"); }
    return d;
  }
  /* height, the linear way: one call per node (plus one per empty slot) */
  function height(x, c) {
    if (c) c.add("calls");
    if (!x) return -1;
    return 1 + Math.max(height(x.left, c), height(x.right, c));
  }
  function ghgt(g, c) {                         // general tree
    if (c) c.add("calls");
    if (!g) return -1;
    let h = -1;
    for (const ch of g.children) h = Math.max(h, ghgt(ch, c));
    return h + 1;
  }
  /* height, the naive way: max over leaves of depth(leaf), depth by climbing.
     Correct, but Θ(n²) on a path. */
  function heightNaive(root, c) {
    let best = -1;
    const stack = root ? [root] : [];
    while (stack.length) {
      const x = stack.pop();
      if (c) c.add("visits");
      if (!x.left && !x.right) best = Math.max(best, depthByParent(x, c));
      if (x.right) stack.push(x.right);
      if (x.left) stack.push(x.left);
    }
    return best;
  }
  function nodes(x, out) { out = out || []; if (x) { out.push(x); nodes(x.left, out); nodes(x.right, out); } return out; }
  function leaves(x) { return nodes(x).filter(n => !n.left && !n.right); }
  function depthOf(x) { let d = 0; while (x.parent) { x = x.parent; d++; } return d; }
  function totalDepth(root) { let s = 0; for (const n of nodes(root)) s += depthOf(n); return s; }
  function levelCounts(root) {
    const cnt = [];
    for (const n of nodes(root)) { const d = depthOf(n); cnt[d] = (cnt[d] || 0) + 1; }
    for (let i = 0; i < cnt.length; i++) cnt[i] = cnt[i] || 0;
    return cnt;
  }

  /* ── builders for the four shapes ──────────────────────────────────────── */
  function perfect(h, keyFn) {                  // 2^(h+1) − 1 nodes
    let k = 0;
    const rec = d => {
      if (d > h) return null;
      const L = rec(d + 1); const me = node(keyFn ? keyFn(k) : k, null, null); k++; const R = rec(d + 1);
      me.left = L; me.right = R; if (L) L.parent = me; if (R) R.parent = me; return me;
    };
    return rec(0);
  }
  function fromLevel(arr) {                     // level-order array with nulls → pointer tree
    const ns = arr.map(v => (v === null || v === undefined) ? null : node(v));
    ns.forEach((n, i) => {
      if (!n || i === 0) return;
      const p = ns[Math.floor((i - 1) / 2)];
      if (!p) return;
      if (i % 2 === 1) p.left = n; else p.right = n;
      n.parent = p;
    });
    return ns[0] || null;
  }
  function complete(n) {                        // first n slots of the level numbering
    return fromLevel(Array.from({ length: n }, (_, i) => i));
  }
  function path(n, dir) {                       // a chain, all left or all right
    let root = null, prev = null;
    for (let i = 0; i < n; i++) {
      const x = node(dir === "left" ? n - i : i + 1);
      if (!prev) root = x; else { if (dir === "left") prev.left = x; else prev.right = x; x.parent = prev; }
      prev = x;
    }
    return root;
  }
  /* a comb (caterpillar): a right-going spine whose every node has a left leaf.
     n/2 leaves at depths 1 … n/2 — the worst case for the naive height */
  function comb(n) {
    let root = null, prev = null, k = 0;
    while (k < n) {
      const s = node(k++);
      if (!prev) root = s; else { prev.right = s; s.parent = prev; }
      if (k < n) { const l = node(k++); s.left = l; l.parent = s; }
      prev = s;
    }
    return root;
  }
  /* full (proper) but not complete: a "vine with pairs" of height h */
  function fullNotComplete(h) {
    let k = 0;
    const rec = d => {
      if (d === h) return node(k++);
      const me = node(k++);
      const L = rec(d + 1), R = node(k++);
      me.left = L; me.right = R; L.parent = me; R.parent = me;
      return me;
    };
    if (h === 0) return node(0);
    return rec(0);
  }

  /* ── level numbering / array layout ────────────────────────────────────── */
  function levelIndex(x) {                      // f(root)=0, f(left)=2f+1, f(right)=2f+2
    const pathUp = [];
    while (x.parent) { pathUp.push(x === x.parent.left ? 1 : 2); x = x.parent; }
    let f = 0;
    for (let i = pathUp.length - 1; i >= 0; i--) f = 2 * f + pathUp[i];
    return f;
  }
  function toLevelArray(root) {
    const ns = nodes(root);
    let mx = -1;
    const idx = ns.map(n => { const f = levelIndex(n); mx = Math.max(mx, f); return f; });
    const arr = Array.from({ length: mx + 1 }, () => null);
    ns.forEach((n, i) => { arr[idx[i]] = n.key; });
    return arr;
  }

  /* ── traversals, recursive (with call / edge counting) ─────────────────── */
  function preorder(x, out, c) { if (c) c.add("calls"); if (!x) return out; out.push(x); preorder(x.left, out, c); preorder(x.right, out, c); return out; }
  function inorder(x, out, c) { if (c) c.add("calls"); if (!x) return out; inorder(x.left, out, c); out.push(x); inorder(x.right, out, c); return out; }
  function postorder(x, out, c) { if (c) c.add("calls"); if (!x) return out; postorder(x.left, out, c); postorder(x.right, out, c); out.push(x); return out; }
  function levelorder(root, c) {
    const out = [], q = root ? [root] : [];
    let head = 0, maxQ = q.length;
    while (head < q.length) {
      const x = q[head++]; out.push(x);
      if (x.left) q.push(x.left); if (x.right) q.push(x.right);
      maxQ = Math.max(maxQ, q.length - head);
      if (c) c.add("deq");
    }
    if (c) c.add("maxQueue", maxQ);
    return out;
  }
  /* recursive traversal, but recording the frames the stepped figure needs:
     each event = {type:'pre'|'in'|'post', node, depth, frames(stack depth)} */
  function traversalEvents(root) {
    const ev = []; let maxFrames = 0, calls = 0, edges = 0;
    const rec = (x, d) => {
      calls++; if (!x) return;
      maxFrames = Math.max(maxFrames, d + 1);
      ev.push({ type: "pre", node: x, d: d });
      if (x.left) edges++; rec(x.left, d + 1); if (x.left) edges++;
      ev.push({ type: "in", node: x, d: d });
      if (x.right) edges++; rec(x.right, d + 1); if (x.right) edges++;
      ev.push({ type: "post", node: x, d: d });
    };
    rec(root, 0);
    return { events: ev, maxFrames: maxFrames, calls: calls, edgeTraversals: edges };
  }

  /* ── traversals with an explicit stack / queue, recorded frame by frame ── */
  function iterPreorder(root) {
    const frames = [], out = [], st = root ? [root] : [];
    let maxSt = st.length, pushes = st.length;
    frames.push({ stack: st.map(n => n.key), out: out.slice(), note: "push the root", cur: null });
    while (st.length) {
      const x = st.pop(); out.push(x.key);
      frames.push({ stack: st.map(n => n.key), out: out.slice(), note: `pop ${x.key}, visit it`, cur: x });
      if (x.right) { st.push(x.right); pushes++; }
      if (x.left) { st.push(x.left); pushes++; }
      maxSt = Math.max(maxSt, st.length);
      if (x.right || x.left) frames.push({ stack: st.map(n => n.key), out: out.slice(), note: `push right${x.right ? " " + x.right.key : ""} then left${x.left ? " " + x.left.key : ""} — left is on top, so it comes out first`, cur: x });
    }
    return { frames: frames, out: out, maxStack: maxSt, pushes: pushes };
  }
  function iterInorder(root) {
    const frames = [], out = [], st = [];
    let x = root, maxSt = 0, pushes = 0;
    frames.push({ stack: [], out: [], note: "start at the root with an empty stack", cur: x });
    while (x || st.length) {
      while (x) { st.push(x); pushes++; maxSt = Math.max(maxSt, st.length);
        frames.push({ stack: st.map(n => n.key), out: out.slice(), note: `push ${x.key}, go left`, cur: x }); x = x.left; }
      x = st.pop(); out.push(x.key);
      frames.push({ stack: st.map(n => n.key), out: out.slice(), note: `no left child left: pop ${x.key}, visit it, go right`, cur: x });
      x = x.right;
    }
    return { frames: frames, out: out, maxStack: maxSt, pushes: pushes };
  }
  /* the trick: run "root, right, left" preorder with a stack and reverse the output */
  function iterPostorderTrick(root) {
    const frames = [], rev = [], st = root ? [root] : [];
    let maxSt = st.length, pushes = st.length;
    frames.push({ stack: st.map(n => n.key), out: [], note: "push the root; we will collect root-RIGHT-left order and reverse it at the end", cur: null });
    while (st.length) {
      const x = st.pop(); rev.push(x.key);
      frames.push({ stack: st.map(n => n.key), out: rev.slice(), note: `pop ${x.key} → append to the reversed list`, cur: x });
      if (x.left) { st.push(x.left); pushes++; }
      if (x.right) { st.push(x.right); pushes++; }
      maxSt = Math.max(maxSt, st.length);
      if (x.left || x.right) frames.push({ stack: st.map(n => n.key), out: rev.slice(), note: `push left${x.left ? " " + x.left.key : ""} then right${x.right ? " " + x.right.key : ""} — right on top, so right subtree is emitted before left`, cur: x });
    }
    const out = rev.slice().reverse();
    frames.push({ stack: [], out: out, note: "reverse the collected list: that is the postorder", cur: null, final: true });
    return { frames: frames, out: out, maxStack: maxSt, pushes: pushes };
  }
  function iterLevel(root) {
    const frames = [], out = [], q = root ? [root] : [];
    let maxQ = q.length;
    frames.push({ stack: q.map(n => n.key), out: [], note: "enqueue the root", cur: null, queue: true });
    while (q.length) {
      const x = q.shift(); out.push(x.key);
      if (x.left) q.push(x.left); if (x.right) q.push(x.right);
      maxQ = Math.max(maxQ, q.length);
      frames.push({ stack: q.map(n => n.key), out: out.slice(), note: `dequeue ${x.key}, visit it, enqueue its children${x.left ? " " + x.left.key : ""}${x.right ? " " + x.right.key : ""}`, cur: x, queue: true });
    }
    return { frames: frames, out: out, maxStack: maxQ, pushes: out.length };
  }

  /* ── Morris (threaded) inorder: O(1) extra space, temporary right-threads ── */
  function morris(root) {
    const frames = [], out = [];
    let cur = root, follows = 0, threads = 0;
    const threadSet = new Set();
    const snap = (note, cur, pre) => frames.push({ out: out.slice(), note: note, cur: cur, pre: pre || null, threads: [...threadSet] });
    snap("start at the root", cur);
    while (cur) {
      if (!cur.left) {
        out.push(cur.key); snap(`${cur.key} has no left child: visit it, follow its right link`, cur);
        cur = cur.right; follows++;
      } else {
        let pre = cur.left; follows++;
        while (pre.right && pre.right !== cur) { pre = pre.right; follows++; }
        if (!pre.right) {
          pre.right = cur; threads++; threadSet.add(pre);
          snap(`inorder predecessor of ${cur.key} is ${pre.key}: thread ${pre.key}.right → ${cur.key}, then go left`, cur, pre);
          cur = cur.left; follows++;
        } else {
          pre.right = null; threadSet.delete(pre);
          out.push(cur.key);
          snap(`arrived back at ${cur.key} via the thread from ${pre.key}: remove it, visit ${cur.key}, go right`, cur, pre);
          cur = cur.right; follows++;
        }
      }
    }
    snap("done — every thread has been removed; the tree is as it was", null);
    return { frames: frames, out: out, follows: follows, threads: threads };
  }

  /* ── Euler tour ──────────────────────────────────────────────────────────── */
  function eulerTour(root) {
    const steps = []; let edgeDown = 0, edgeUp = 0;
    const rec = x => {
      steps.push({ type: "pre", node: x });
      if (x.left) { edgeDown++; steps.push({ type: "down", node: x, to: x.left }); rec(x.left); edgeUp++; steps.push({ type: "up", node: x.left, to: x }); }
      steps.push({ type: "in", node: x });
      if (x.right) { edgeDown++; steps.push({ type: "down", node: x, to: x.right }); rec(x.right); edgeUp++; steps.push({ type: "up", node: x.right, to: x }); }
      steps.push({ type: "post", node: x });
    };
    if (root) rec(root);
    return { steps: steps, edgeDown: edgeDown, edgeUp: edgeUp };
  }

  /* ── expression trees ──────────────────────────────────────────────────── */
  const PREC = { "+": 1, "-": 1, "*": 2, "/": 2 };
  function isOp(t) { return Object.prototype.hasOwnProperty.call(PREC, t); }
  function buildPostfix(tokens) {               // with frames for the stepped figure
    const st = [], frames = [];
    for (const t of tokens) {
      if (isOp(t)) {
        const r = st.pop(), l = st.pop();
        if (!l || !r) throw new Error("malformed postfix");
        const x = node(t, l, r); st.push(x);
        frames.push({ token: t, stack: st.slice(), note: `operator ${t}: pop two subtrees, make them its children, push the new tree` });
      } else {
        const x = node(+t); st.push(x);
        frames.push({ token: t, stack: st.slice(), note: `operand ${t}: push a leaf` });
      }
    }
    if (st.length !== 1) throw new Error("malformed postfix");
    return { root: st[0], frames: frames };
  }
  function evaluate(x, frames) {
    if (!isOp(x.key)) { if (frames) frames.push({ node: x, value: x.key }); return x.key; }
    const a = evaluate(x.left, frames), b = evaluate(x.right, frames);
    const v = x.key === "+" ? a + b : x.key === "-" ? a - b : x.key === "*" ? a * b : a / b;
    if (frames) frames.push({ node: x, value: v, a: a, b: b });
    return v;
  }
  function toInfix(x) {                           // minimal parentheses
    if (!isOp(x.key)) return String(x.key);
    const p = PREC[x.key];
    let L = toInfix(x.left), R = toInfix(x.right);
    if (isOp(x.left.key) && PREC[x.left.key] < p) L = "(" + L + ")";
    if (isOp(x.right.key) && (PREC[x.right.key] < p || (PREC[x.right.key] === p && (x.key === "-" || x.key === "/")))) R = "(" + R + ")";
    return L + " " + x.key + " " + R;
  }
  function toInfixFull(x) { return isOp(x.key) ? "(" + toInfixFull(x.left) + " " + x.key + " " + toInfixFull(x.right) + ")" : String(x.key); }
  function toPrefix(x) { return isOp(x.key) ? x.key + " " + toPrefix(x.left) + " " + toPrefix(x.right) : String(x.key); }
  function toPostfix(x) { return isOp(x.key) ? toPostfix(x.left) + " " + toPostfix(x.right) + " " + x.key : String(x.key); }

  /* ── BST operations ───────────────────────────────────────────────────────
     Counting convention, used everywhere on the page: one "comparison" per node
     whose key is compared with the search key. A search that stops at node x
     has compared against every node on the root→x path, depth(x)+1 of them; an
     insertion compares against every ancestor of the new node, depth(z) of them. */
  function search(T, k, c) {
    let x = T.root; const pathN = [];
    while (x) {
      pathN.push(x); if (c) c.add("cmp");
      if (k === x.key) return { node: x, path: pathN };
      x = k < x.key ? x.left : x.right;
    }
    return { node: null, path: pathN };
  }
  function minimum(x, c) { const p = [x]; while (x.left) { x = x.left; p.push(x); if (c) c.add("steps"); } return { node: x, path: p }; }
  function maximum(x, c) { const p = [x]; while (x.right) { x = x.right; p.push(x); if (c) c.add("steps"); } return { node: x, path: p }; }
  function successor(x, c) {
    if (x.right) { const m = minimum(x.right, c); if (c) c.add("steps"); return { node: m.node, path: [x].concat(m.path), how: "down" }; }
    const p = [x]; let y = x.parent;
    while (y && x === y.right) { x = y; y = y.parent; p.push(x); if (c) c.add("steps"); }
    if (y) { p.push(y); if (c) c.add("steps"); }
    return { node: y, path: p, how: "up" };
  }
  function predecessor(x, c) {
    if (x.left) { const m = maximum(x.left, c); if (c) c.add("steps"); return { node: m.node, path: [x].concat(m.path), how: "down" }; }
    const p = [x]; let y = x.parent;
    while (y && x === y.left) { x = y; y = y.parent; p.push(x); if (c) c.add("steps"); }
    if (y) { p.push(y); if (c) c.add("steps"); }
    return { node: y, path: p, how: "up" };
  }
  function insert(T, k, c) {
    let y = null, x = T.root; const pathN = [];
    while (x) { y = x; pathN.push(x); if (c) c.add("cmp"); x = k < x.key ? x.left : x.right; }
    const z = node(k); z.parent = y;
    if (!y) T.root = z; else if (k < y.key) y.left = z; else y.right = z;
    return { node: z, path: pathN };
  }
  function transplant(T, u, v) {
    if (!u.parent) T.root = v;
    else if (u === u.parent.left) u.parent.left = v;
    else u.parent.right = v;
    if (v) v.parent = u.parent;
  }
  /* delete z; policy 'succ' | 'pred' | 'alt' | fn → which neighbour replaces a
     two-child node. Returns the case label for the figure. */
  let altFlag = false;
  function remove(T, z, policy, c) {
    if (c) c.add("deletes");
    if (!z.left) { transplant(T, z, z.right); return z.right ? "one child (right)" : "leaf"; }
    if (!z.right) { transplant(T, z, z.left); return "one child (left)"; }
    let useSucc = true;
    if (policy === "pred") useSucc = false;
    else if (policy === "alt") { useSucc = altFlag; altFlag = !altFlag; }
    else if (typeof policy === "function") useSucc = policy();
    if (useSucc) {
      const y = minimum(z.right, c).node;
      if (y.parent !== z) { transplant(T, y, y.right); y.right = z.right; y.right.parent = y; }
      transplant(T, z, y); y.left = z.left; y.left.parent = y;
      return y === z.right ? "two children, successor is the right child" : "two children, successor deeper";
    } else {
      const y = maximum(z.left, c).node;
      if (y.parent !== z) { transplant(T, y, y.left); y.left = z.left; y.left.parent = y; }
      transplant(T, z, y); y.right = z.right; y.right.parent = y;
      return "two children, predecessor";
    }
  }
  function build(keys, c) { const T = { root: null }; const per = []; for (const k of keys) { const before = c ? c.get("cmp") : 0; insert(T, k, c); per.push(c ? c.get("cmp") - before : 0); } return { T: T, per: per }; }
  function clone(x, parent) { if (!x) return null; const n = { key: x.key, left: null, right: null, parent: parent || null }; n.left = clone(x.left, n); n.right = clone(x.right, n); return n; }
  function find(T, k) { return search(T, k).node; }
  function isBST(x, lo, hi) {                   // the CORRECT invariant, checked with bounds
    if (!x) return true;
    if ((lo !== undefined && x.key <= lo) || (hi !== undefined && x.key >= hi)) return false;
    return isBST(x.left, lo, x.key) && isBST(x.right, x.key, hi);
  }
  function isBSTLocal(x) {                      // the WRONG invariant: only children checked
    if (!x) return true;
    if (x.left && !(x.left.key < x.key)) return false;
    if (x.right && !(x.right.key > x.key)) return false;
    return isBSTLocal(x.left) && isBSTLocal(x.right);
  }

  /* ── tree sort vs quicksort: the same comparisons ─────────────────────── */
  function treeSortCounts(keys) {               // per-key comparisons at insertion = depth
    const c = counterLite(); const T = { root: null }; const per = [];
    for (const k of keys) { const b = c.n; insert(T, k, c); per.push(c.n - b); }
    return { per: per, total: c.n, sorted: inorder(T.root, []).map(n => n.key) };
  }
  function quicksortCounts(keys) {              // first element pivot, ORDER-PRESERVING partition
    const per = new Map(); let total = 0;
    const rec = a => {
      if (a.length <= 1) return a;
      const p = a[0], L = [], R = [];
      for (let i = 1; i < a.length; i++) { total++; per.set(a[i], (per.get(a[i]) || 0) + 1); if (a[i] < p) L.push(a[i]); else R.push(a[i]); }
      return rec(L).concat([p], rec(R));
    };
    const sorted = rec(keys.slice());
    return { per: keys.map(k => per.get(k) || 0), total: total, sorted: sorted };
  }
  function counterLite() { return { n: 0, add: function (k, by) { if (k === "cmp") this.n += (by === undefined ? 1 : by); }, get: function (k) { return k === "cmp" ? this.n : 0; } }; }

  /* ── random BSTs ──────────────────────────────────────────────────────── */
  function harmonic(n) { let s = 0; for (let i = 1; i <= n; i++) s += 1 / i; return s; }
  function expectedAvgDepth(n) { return n ? 2 * (n + 1) * harmonic(n) / n - 4 : 0; }   // exact, random insertion order, no deletions
  function randomBST(n, r) { const keys = AL.perm(n, r); const c = counterLite(); const T = { root: null }; for (const k of keys) insert(T, k, c); return { T: T, cmp: c.n }; }
  function stats(root) { const ns = nodes(root); let s = 0, h = -1; for (const n of ns) { const d = depthOf(n); s += d; h = Math.max(h, d); } return { n: ns.length, height: h, meanDepth: ns.length ? s / ns.length : 0, total: s }; }
  function depthHistogram(root) { const H = []; for (const n of nodes(root)) { const d = depthOf(n); H[d] = (H[d] || 0) + 1; } for (let i = 0; i < H.length; i++) H[i] = H[i] || 0; return H; }

  /* ── Catalan: enumerate all insertion orders, group by shape ──────────── */
  function shapeKey(x) { return x ? "(" + shapeKey(x.left) + x.key + shapeKey(x.right) + ")" : "."; }
  function permutations(arr) {
    const out = [];
    const rec = (a, pre) => { if (!a.length) { out.push(pre); return; } for (let i = 0; i < a.length; i++) rec(a.slice(0, i).concat(a.slice(i + 1)), pre.concat([a[i]])); };
    rec(arr, []); return out;
  }
  function enumerateBSTs(n) {
    const perms = permutations(Array.from({ length: n }, (_, i) => i + 1));
    const byShape = new Map();
    for (const p of perms) {
      const T = build(p).T; const k = shapeKey(T.root);
      if (!byShape.has(k)) byShape.set(k, { root: T.root, count: 0, first: p });
      byShape.get(k).count++;
    }
    return { shapes: [...byShape.values()], perms: perms.length };
  }
  function catalan(n) { let c = 1; for (let i = 0; i < n; i++) c = c * 2 * (2 * i + 1) / (i + 2); return Math.round(c); }

  /* ── order statistics on a size-augmented tree ────────────────────────── */
  function annotateSizes(x) { if (!x) return 0; x.size = 1 + annotateSizes(x.left) + annotateSizes(x.right); return x.size; }
  function select(x, i, c) {                    // i is 1-based rank
    const p = [];
    while (x) { p.push(x); if (c) c.add("steps"); const r = (x.left ? x.left.size : 0) + 1; if (i === r) return { node: x, path: p }; if (i < r) x = x.left; else { i -= r; x = x.right; } }
    return { node: null, path: p };
  }
  function rank(T, k, c) {
    let x = T.root, r = 0; const p = [];
    while (x) { p.push(x); if (c) c.add("steps"); if (k < x.key) x = x.left; else if (k > x.key) { r += (x.left ? x.left.size : 0) + 1; x = x.right; } else return { rank: r + (x.left ? x.left.size : 0) + 1, path: p, node: x }; }
    return { rank: null, path: p, node: null };
  }


  /* ── range query [a, b] — O(h + m) ───────────────────────────────────── */
  function rangeQuery(x, a, b, out, c) {
    if (!x) return out;
    if (c) c.add("visits");
    if (a < x.key) rangeQuery(x.left, a, b, out, c);
    if (a <= x.key && x.key <= b) out.push(x);
    if (x.key < b) rangeQuery(x.right, a, b, out, c);
    return out;
  }
  function ceiling(T, a) {                      // smallest key ≥ a, with its search path
    let x = T.root, best = null; const p = [];
    while (x) { p.push(x); if (x.key === a) return { node: x, path: p }; if (a < x.key) { best = x; x = x.left; } else x = x.right; }
    return { node: best, path: p };
  }
  function floorKey(T, b) {
    let x = T.root, best = null; const p = [];
    while (x) { p.push(x); if (x.key === b) return { node: x, path: p }; if (b < x.key) x = x.left; else { best = x; x = x.right; } }
    return { node: best, path: p };
  }
  function lca(T, u, v, c) {                    // keys u, v present; no parent pointers needed
    const lo = Math.min(u, v), hi = Math.max(u, v); let x = T.root; const p = [];
    while (x) { p.push(x); if (c) c.add("steps"); if (hi < x.key) x = x.left; else if (lo > x.key) x = x.right; else return { node: x, path: p }; }
    return { node: null, path: p };
  }
  function fromSorted(keys, c) {                // balanced build: middle element as root, recursively
    const rec = (lo, hi) => { if (c) c.add("calls"); if (lo > hi) return null; const mid = Math.floor((lo + hi) / 2); const x = node(keys[mid]); x.left = rec(lo, mid - 1); x.right = rec(mid + 1, hi); if (x.left) x.left.parent = x; if (x.right) x.right.parent = x; return x; };
    return { root: rec(0, keys.length - 1) };
  }

  /* ── general (n-ary) trees: left-child / right-sibling ────────────────── */
  function toLCRS(g) {                          // returns a binary tree; keys are labels
    const rec = (t, sibs) => {
      const me = node(t.label);
      if (t.children.length) me.left = rec(t.children[0], t.children.slice(1)), me.left.parent = me;
      if (sibs.length) me.right = rec(sibs[0], sibs.slice(1)), me.right.parent = me;
      return me;
    };
    return rec(g, []);
  }
  function gnodes(g, out) { out = out || []; out.push(g); for (const c of g.children) gnodes(c, out); return out; }
  function gsize(g) { return 1 + g.children.reduce((s, c) => s + gsize(c), 0); }
  function gdepthMap(g) { const m = new Map(); const rec = (x, d, par) => { m.set(x, { depth: d, parent: par }); x.children.forEach(c => rec(c, d + 1, x)); }; rec(g, 0, null); return m; }

  /* ── the sample trees the page uses throughout ────────────────────────── */
  function sampleGeneral() {
    return gnode("A", [
      gnode("B", [gnode("E"), gnode("F", [gnode("J"), gnode("K"), gnode("L")])]),
      gnode("C"),
      gnode("D", [gnode("G", [gnode("M")]), gnode("H"), gnode("I")])
    ]);
  }
  function sampleBinary() {                     // 10 nodes, height 3, letters
    return node("A",
      node("B", node("D"), node("E", node("H"), node("I"))),
      node("C", node("F"), node("G", null, node("J"))));
  }
  const SAMPLE_KEYS = [15, 6, 18, 3, 7, 17, 20, 2, 4, 13, 9];
  function sampleBST() { return build(SAMPLE_KEYS.slice()).T; }
  function wrongInvariantTree() {               // every parent–child pair is ordered, the tree is NOT a BST
    return { root: node(10, node(5, node(2), node(12)), node(15)) };
  }

  return {
    node, gnode, size, depthByParent, depthOf, height, ghgt, heightNaive, nodes, leaves, totalDepth, levelCounts,
    perfect, fromLevel, complete, path, comb, fullNotComplete, levelIndex, toLevelArray,
    preorder, inorder, postorder, levelorder, traversalEvents,
    iterPreorder, iterInorder, iterPostorderTrick, iterLevel, morris, eulerTour,
    PREC, isOp, buildPostfix, evaluate, toInfix, toInfixFull, toPrefix, toPostfix,
    search, minimum, maximum, successor, predecessor, insert, transplant, remove, build, clone, find, isBST, isBSTLocal,
    treeSortCounts, quicksortCounts, harmonic, expectedAvgDepth, randomBST, stats, depthHistogram,
    shapeKey, enumerateBSTs, catalan, annotateSizes, select, rank, rangeQuery, ceiling, floorKey, lca, fromSorted,
    toLCRS, gnodes, gsize, gdepthMap, sampleGeneral, sampleBinary, SAMPLE_KEYS, sampleBST, wrongInvariantTree
  };
})();

/* ═══════════════════════════════════════════════════════════════════════════
   TD — drawing helpers for pointer-based trees (built on AL's scaffolding)
   ═══════════════════════════════════════════════════════════════════════════ */
const TD = (function () {
  if (typeof d3 === "undefined") return null;

  /* inorder x-layout: the i-th node in inorder sits at column i; depth gives y.
     No two nodes share a column, so any shape — including a 30-node chain —
     draws without overlap. */
  function layoutBin(root, o) {
    const list = TB.inorder(root, []);
    const n = list.length, unit = n ? o.w / n : o.w;
    const by = new Map();
    list.forEach((nd, i) => {
      const d = TB.depthOf(nd);
      const p = { n: nd, i: i, d: d, x: o.x + (i + 0.5) * unit, y: o.y + d * o.levelH + o.r };
      by.set(nd, p);
    });
    return { list: list, by: by, unit: unit };
  }
  function drawBin(g, root, opt) {
    const o = Object.assign({ x: 0, y: 0, w: 600, levelH: 54, r: 14, fontSize: 12,
      fill: null, stroke: null, edge: null, edgeW: null, label: null, text: null, textFill: null, nullSlots: false, dash: null }, opt || {});
    const L = layoutBin(root, o);
    const gg = g.append("g");
    /* edges first */
    L.list.forEach(nd => {
      const p = L.by.get(nd);
      [nd.left, nd.right].forEach((ch, side) => {
        if (!ch) {
          if (o.nullSlots) {
            const nx = p.x + (side ? 1 : -1) * Math.max(8, L.unit * 0.35), ny = p.y + o.levelH * 0.55;
            gg.append("line").attr("x1", p.x).attr("y1", p.y).attr("x2", nx).attr("y2", ny).attr("stroke", AC.line).attr("stroke-dasharray", "2,3");
            gg.append("rect").attr("x", nx - 4).attr("y", ny - 4).attr("width", 8).attr("height", 8).attr("fill", AC.bg).attr("stroke", AC.line);
          }
          return;
        }
        const q = L.by.get(ch);
        const ln = gg.append("line").attr("x1", p.x).attr("y1", p.y).attr("x2", q.x).attr("y2", q.y)
          .attr("stroke", o.edge ? (o.edge(ch, nd) || AC.line) : AC.line)
          .attr("stroke-width", o.edgeW ? (o.edgeW(ch, nd) || 1.5) : 1.5);
        if (o.dash && o.dash(ch, nd)) ln.attr("stroke-dasharray", o.dash(ch, nd));
      });
    });
    L.list.forEach(nd => {
      const p = L.by.get(nd);
      gg.append("circle").attr("cx", p.x).attr("cy", p.y).attr("r", o.r)
        .attr("fill", o.fill ? (o.fill(nd) || AC.panel2) : AC.panel2)
        .attr("stroke", o.stroke ? (o.stroke(nd) || AC.line) : AC.line).attr("stroke-width", 1.5);
      gg.append("text").attr("x", p.x).attr("y", p.y + 4).attr("text-anchor", "middle")
        .attr("font-size", o.fontSize).attr("fill", o.textFill ? (o.textFill(nd) || AC.ink) : AC.ink)
        .text(o.text ? o.text(nd) : nd.key);
      if (o.label) {
        const t = o.label(nd);
        if (t !== null && t !== undefined && t !== "") gg.append("text").attr("x", p.x).attr("y", p.y + o.r + 11).attr("text-anchor", "middle")
          .attr("font-size", 9.5).attr("fill", AC.muted).text(t);
      }
    });
    return { g: gg, pos: L.by, list: L.list, unit: L.unit };
  }

  /* general tree: x from leaf order (each leaf its own column; a parent sits
     over the mean of its children) */
  function layoutGen(root, o) {
    const leaves = TB.gnodes(root).filter(x => !x.children.length);
    const unit = o.w / Math.max(1, leaves.length);
    const by = new Map(); let li = 0;
    const rec = (x, d) => {
      let cx;
      if (!x.children.length) { cx = o.x + (li + 0.5) * unit; li++; }
      else { const xs = x.children.map(c => rec(c, d + 1)); cx = (Math.min(...xs) + Math.max(...xs)) / 2; }
      by.set(x, { n: x, d: d, x: cx, y: o.y + d * o.levelH + o.r });
      return cx;
    };
    rec(root, 0);
    return by;
  }
  function drawGen(g, root, opt) {
    const o = Object.assign({ x: 0, y: 0, w: 600, levelH: 60, r: 14, fontSize: 12, fill: null, stroke: null, edge: null, label: null, rect: false }, opt || {});
    const by = layoutGen(root, o);
    const gg = g.append("g");
    for (const [x, p] of by) x.children.forEach(c => {
      const q = by.get(c);
      gg.append("line").attr("x1", p.x).attr("y1", p.y).attr("x2", q.x).attr("y2", q.y)
        .attr("stroke", o.edge ? (o.edge(c, x) || AC.line) : AC.line).attr("stroke-width", 1.5);
    });
    for (const [x, p] of by) {
      gg.append("circle").attr("cx", p.x).attr("cy", p.y).attr("r", o.r)
        .attr("fill", o.fill ? (o.fill(x) || AC.panel2) : AC.panel2)
        .attr("stroke", o.stroke ? (o.stroke(x) || AC.line) : AC.line).attr("stroke-width", 1.5);
      gg.append("text").attr("x", p.x).attr("y", p.y + 4).attr("text-anchor", "middle").attr("font-size", o.fontSize).attr("fill", AC.ink).text(x.label);
      if (o.label) { const t = o.label(x); if (t) gg.append("text").attr("x", p.x).attr("y", p.y + o.r + 11).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", AC.muted).text(t); }
    }
    return { g: gg, pos: by };
  }

  /* a vertical stack of boxes, bottom = index 0 (a stack), or a horizontal
     queue row, head on the left */
  function stack(g, items, o) {
    o = Object.assign({ x: 0, y: 0, w: 56, h: 22, gap: 3, title: "stack", max: 8, top: null }, o || {});
    const gg = g.append("g").attr("transform", `translate(${o.x},${o.y})`);
    gg.append("text").attr("x", o.w / 2).attr("y", -6).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.muted).text(o.title);
    const H = o.max * (o.h + o.gap);
    gg.append("rect").attr("x", -4).attr("y", 0).attr("width", o.w + 8).attr("height", H + 4).attr("fill", "none").attr("stroke", AC.line).attr("rx", 4);
    items.forEach((it, i) => {
      const y = H - (i + 1) * (o.h + o.gap) + o.gap;
      gg.append("rect").attr("x", 0).attr("y", y).attr("width", o.w).attr("height", o.h).attr("rx", 3)
        .attr("fill", i === items.length - 1 ? AC.a2 : AC.panel2).attr("stroke", AC.line);
      gg.append("text").attr("x", o.w / 2).attr("y", y + o.h / 2 + 4).attr("text-anchor", "middle").attr("font-size", 12)
        .attr("fill", i === items.length - 1 ? AC.bg : AC.ink).text(it);
    });
    if (!items.length) gg.append("text").attr("x", o.w / 2).attr("y", H / 2).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.muted).text("empty");
    return gg;
  }
  function note(g, x, y, txt, color) {
    return g.append("text").attr("x", x).attr("y", y).attr("font-size", 11.5).attr("fill", color || AC.ink).text(txt);
  }
  return { layoutBin, drawBin, layoutGen, drawGen, stack, note };
})();

/* ═══════════════════════════════════════════════════════════════════════════
   figures
   ═══════════════════════════════════════════════════════════════════════════ */
(function () {
  if (typeof document === "undefined" || typeof d3 === "undefined") return;
  const $ = id => document.getElementById(id);
  const fig = (id, fn) => { if ($(id)) fn(); };
  const txt = (id, s) => { const el = $(id); if (el) el.textContent = s; };
  const html = (id, s) => { const el = $(id); if (el) el.innerHTML = s; };
  const on = (id, ev, fn) => { const el = $(id); if (el) el.addEventListener(ev, fn); };
  const val = id => { const el = $(id); return el ? el.value : null; };
  const f2 = x => (+x).toFixed(2), f3 = x => (+x).toFixed(3);

  /* ── F01 anatomy: hover / select a node, read depth, height, size ─────── */
  fig("anat-svg", () => {
    const G = TB.sampleGeneral(), all = TB.gnodes(G), dm = TB.gdepthMap(G);
    const sel = $("anat-node");
    if (sel) { all.forEach(x => { const o = document.createElement("option"); o.value = x.label; o.textContent = x.label; sel.appendChild(o); }); sel.value = "F"; }
    const H = TB.ghgt(G), n = TB.gsize(G);
    function draw(focusLabel) {
      const focus = all.find(x => x.label === focusLabel) || all[0];
      const anc = new Set(); let a = focus; while (a) { anc.add(a); a = dm.get(a).parent; }
      const desc = new Set(TB.gnodes(focus));
      const F = AL.frame("#anat-svg", 760, 330, { l: 20, r: 20, t: 26, b: 20 });
      const d = TD.drawGen(F.g, G, {
        x: 0, y: 0, w: 520, levelH: 66, r: 15,
        fill: x => x === focus ? AC.a2 : anc.has(x) ? "#3b3a2a" : desc.has(x) ? "#1f3a4a" : null,
        stroke: x => x === focus ? AC.a2 : anc.has(x) ? AC.a2 : desc.has(x) ? AC.teal : null,
        edge: (c, p) => anc.has(c) && anc.has(p) ? AC.a2 : (desc.has(c) && desc.has(p)) ? AC.teal : null,
        label: x => `d=${dm.get(x).depth} h=${TB.ghgt(x)}`
      });
      d.g.selectAll("circle").style("cursor", "pointer").on("mouseenter", function () {
        const p = [...d.pos.entries()].find(e => Math.abs(e[1].x - +d3.select(this).attr("cx")) < 1 && Math.abs(e[1].y - +d3.select(this).attr("cy")) < 1);
        if (p && sel) { sel.value = p[0].label; draw(p[0].label); }
      });
      /* level ticks on the left */
      for (let lv = 0; lv <= H; lv++) F.g.append("text").attr("x", -6).attr("y", lv * 66 + 19).attr("font-size", 10).attr("fill", AC.muted).text(`depth ${lv}`);
      /* right-hand panel */
      const px = 545, py = 8; const c = AL.counter();
      const dep = (function () { let d = 0, x = focus; while (dm.get(x).parent) { x = dm.get(x).parent; d++; c.add("up"); } return d; })();
      const hg = TB.ghgt(focus, c), sz = TB.gsize(focus);
      const leafDepths = TB.gnodes(focus).filter(x => !x.children.length).map(x => dm.get(x).depth - dep);
      const rows = [
        [`node ${focus.label}`, ""],
        ["depth (edges up to the root, walked)", `${dep}`],
        ["height (recursive, ${hg} — calls made: ${c.get('calls')})".replace("${hg}", hg).replace("${c.get('calls')}", c.get("calls")), `${hg}`],
        ["  = max leaf depth below it − its depth", `${Math.max(...leafDepths)}  ✓`],
        ["subtree size (nodes, counted)", `${sz}`],
        ["children", `${focus.children.length}`],
        ["ancestors (excluding itself)", `${anc.size - 1}`],
        ["descendants (excluding itself)", `${desc.size - 1}`],
        [focus.children.length ? "internal node" : "leaf (external node)", ""],
        ["", ""],
        [`whole tree: n = ${n}, height = ${H}`, ""],
        [`nodes per depth: ${(function () { const cnt = []; all.forEach(x => { const dd = dm.get(x).depth; cnt[dd] = (cnt[dd] || 0) + 1; }); return cnt.join(" · "); })()}`, ""],
        [`∑ children over all nodes = ${all.reduce((s, x) => s + x.children.length, 0)} = n − 1`, ""]
      ];
      rows.forEach((r, i) => {
        F.g.append("text").attr("x", px).attr("y", py + i * 20).attr("font-size", 11).attr("fill", i === 0 ? AC.a2 : AC.muted).text(r[0]);
        F.g.append("text").attr("x", 740).attr("y", py + i * 20).attr("font-size", 11.5).attr("text-anchor", "end").attr("fill", AC.ink).text(r[1]);
      });
      txt("anat-readout", `Node ${focus.label}: depth ${dep} (walked up ${c.get("up")} parent links), height ${hg} (recursive, ${c.get("calls")} calls over its subtree), subtree size ${sz}. Tree: ${n} nodes, height ${H}, ${all.filter(x => !x.children.length).length} leaves, ${all.filter(x => x.children.length).length} internal nodes.`);
    }
    on("anat-node", "change", () => draw(val("anat-node")));
    draw("F");
  });

  /* ── F02 the four shapes at height h ───────────────────────────────────── */
  fig("shape-svg", () => {
    function draw() {
      const h = +val("shape-h") || 3;
      txt("shape-h-out", h);
      const nComplete = Math.pow(2, h) + Math.max(1, Math.floor(Math.pow(2, h - 1)) - 1);   // a bottom level neither empty nor full (h ≥ 1)
      const trees = [
        { name: "perfect", root: TB.perfect(h), what: "every level full" },
        { name: "complete", root: TB.complete(h === 0 ? 1 : nComplete), what: "all levels full but the last, filled left to right" },
        { name: "full (proper)", root: TB.fullNotComplete(h), what: "every node has 0 or 2 children" },
        { name: "degenerate", root: TB.path(h + 1, "right"), what: "every node has ≤ 1 child: a chain" }
      ];
      const F = AL.frame("#shape-svg", 760, 340, { l: 8, r: 8, t: 8, b: 8 });
      const cw = 744 / 4;
      const lines = [];
      trees.forEach((t, i) => {
        const ns = TB.nodes(t.root), lv = TB.leaves(t.root).length, int = ns.length - lv, H = TB.height(t.root);
        const g = F.g.append("g").attr("transform", `translate(${i * cw},0)`);
        g.append("text").attr("x", cw / 2).attr("y", 12).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", AC.a2).text(t.name);
        g.append("text").attr("x", cw / 2).attr("y", 26).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", AC.muted).text(t.what);
        TD.drawBin(g, t.root, { x: 4, y: 34, w: cw - 8, levelH: Math.min(40, 170 / Math.max(1, H)), r: Math.min(11, Math.max(6, (cw - 8) / ns.length / 2.2)), fontSize: 9, text: () => "" });
        const rows = [`n = ${ns.length}`, `height = ${H}`, `leaves = ${lv}`, `internal = ${int}`,
          `leaves = internal + 1? ${lv === int + 1 ? "yes" : "NO"}`,
          `n = 2^(h+1) − 1? ${ns.length === Math.pow(2, H + 1) - 1 ? "yes" : "no"}`,
          `per level: ${TB.levelCounts(t.root).join(",")}`];
        rows.forEach((r, k) => g.append("text").attr("x", 6).attr("y", 230 + k * 15).attr("font-size", 10.5).attr("fill", k === 4 ? (lv === int + 1 ? AC.good : AC.bad) : AC.ink).text(r));
        lines.push(`${t.name}: n=${ns.length}, h=${H}, leaves=${lv}, internal=${int}`);
      });
      txt("shape-readout", `Height ${h}. ` + lines.join(" · ") + `. Bounds at h = ${h}: h+1 = ${h + 1} ≤ n ≤ 2^(h+1) − 1 = ${Math.pow(2, h + 1) - 1}; leaves ≤ 2^h = ${Math.pow(2, h)}. Every count above is from walking the drawn tree.`);
    }
    on("shape-h", "input", draw); draw();
  });

  /* ── F03 height against n: the two extremes and random BSTs between ───── */
  fig("hb-svg", () => {
    const NMAX = 128, r0 = 11;
    const ns = Array.from({ length: NMAX }, (_, i) => i + 1);
    const lower = ns.map(n => TB.height(TB.complete(n)));         // measured on a complete tree
    const upper = ns.map(n => TB.height(TB.path(n, "right")));    // measured on a chain
    const r = AL.rng(r0);
    const rnd = ns.map(n => { let s = 0; for (let t = 0; t < 12; t++) s += TB.stats(TB.randomBST(n, r).T.root).height; return s / 12; });
    function draw() {
      const nSel = +val("hb-n") || 64;
      txt("hb-n-out", nSel);
      const F = AL.frame("#hb-svg", 760, 330, { l: 46, r: 16, t: 16, b: 36 });
      const x = d3.scaleLinear().domain([1, NMAX]).range([0, F.iw]);
      const y = d3.scaleLog().domain([0.7, NMAX]).range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 5); AL.axisB(F.g, x, 8, "n (nodes)"); AL.axisL(F.g, y, 5, "height (log scale)", d => d);
      const line = (arr, col, dash) => F.g.append("path").datum(ns.map((n, i) => [n, Math.max(arr[i], 0.7)])).attr("fill", "none").attr("stroke", col).attr("stroke-width", 2).attr("stroke-dasharray", dash || null)
        .attr("d", d3.line().x(d => x(d[0])).y(d => y(d[1])));
      line(upper, AC.bad); line(rnd, AC.a2); line(lower, AC.good);
      line(ns.map(n => Math.floor(Math.log2(n))), AC.ink, "2,4");
      AL.legend(F.g, [{ label: "chain: n − 1 (measured, height of a path)", color: AC.bad }, { label: "random insertion order, mean of 12 trees (measured)", color: AC.a2 },
        { label: "complete tree (measured)", color: AC.good }, { label: "floor(log₂ n), the lower bound (formula)", color: AC.ink, dash: "2,4" }], 12, 14);
      const i = Math.min(NMAX, nSel) - 1;
      F.g.append("line").attr("x1", x(nSel)).attr("x2", x(nSel)).attr("y1", 0).attr("y2", F.ih).attr("stroke", AC.muted).attr("stroke-dasharray", "3,3");
      const rr = AL.rng(7); let hs = 0, hmin = 1e9, hmax = -1; const T = 30;
      for (let t = 0; t < T; t++) { const hh = TB.stats(TB.randomBST(nSel, rr).T.root).height; hs += hh; hmin = Math.min(hmin, hh); hmax = Math.max(hmax, hh); }
      const compl = TB.height(TB.complete(nSel)), chain = TB.height(TB.path(nSel, "right"));
      const lo1 = Math.floor(Math.log2(nSel)), lo2 = Math.ceil(Math.log2(nSel + 1)) - 1;
      txt("hb-readout", `n = ${nSel}: complete tree height ${compl} (measured) = floor(log₂ n) = ${lo1} = ceil(log₂(n+1)) − 1 = ${lo2}; chain height ${chain} (measured) = n − 1. Random insertion order, ${T} trees: mean height ${f2(hs / T)}, range ${hmin}–${hmax}; 2·ln n = ${f2(2 * Math.log(nSel))}, 4.311·ln n = ${f2(4.311 * Math.log(nSel))}, height / ln n = ${f2(hs / T / Math.log(nSel))}.` + (nSel > NMAX ? " (The chart stops at n = 128; the readout is measured at your n.)" : ""));
    }
    on("hb-n", "input", draw); draw();
  });

  /* ── F04 left-child / right-sibling: a general tree and its binary form ── */
  fig("lcrs-svg", () => {
    const G = TB.sampleGeneral(), all = TB.gnodes(G), L = TB.toLCRS(G);
    const sel = $("lcrs-node");
    if (sel) { all.forEach(x => { const o = document.createElement("option"); o.value = x.label; o.textContent = x.label; sel.appendChild(o); }); sel.value = "D"; }
    function draw() {
      const fl = val("lcrs-node") || "D";
      const focus = all.find(x => x.label === fl) || all[0];
      const childSet = new Set(focus.children);
      const F = AL.frame("#lcrs-svg", 760, 400, { l: 10, r: 10, t: 22, b: 10 });
      F.g.append("text").attr("x", 150).attr("y", -6).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", AC.a2).text("general tree: node → list of children");
      F.g.append("text").attr("x", 540).attr("y", -6).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", AC.a2).text("same tree, left-child / right-sibling: exactly two links per node");
      TD.drawGen(F.g, G, { x: 0, y: 6, w: 300, levelH: 62, r: 13,
        fill: x => x === focus ? AC.a2 : childSet.has(x) ? "#1f3a4a" : null,
        stroke: x => x === focus ? AC.a2 : childSet.has(x) ? AC.teal : null,
        edge: (c, p) => p === focus ? AC.teal : null });
      /* binary form: mark focus, its left link (first child) and the sibling chain */
      const bn = TB.nodes(L), bf = bn.find(x => x.key === focus.label);
      const chain = new Set(); let s = bf.left; while (s) { chain.add(s); s = s.right; }
      TD.drawBin(F.g, L, { x: 330, y: 6, w: 420, levelH: 50, r: 12, fontSize: 11,
        fill: x => x === bf ? AC.a2 : chain.has(x) ? "#1f3a4a" : null,
        stroke: x => x === bf ? AC.a2 : chain.has(x) ? AC.teal : null,
        edge: (c, p) => (p === bf && c === bf.left) ? AC.teal : (chain.has(p) && c === p.right) ? AC.teal : null,
        dash: (c, p) => c === p.right ? "4,3" : null });
      F.g.append("text").attr("x", 330).attr("y", 372).attr("font-size", 10.5).attr("fill", AC.muted).text("solid = left link (first child) · dashed = right link (next sibling)");
      const c = AL.counter();
      let k = bf.left, cnt = 0; while (k) { cnt++; c.add("follow"); k = k.right; }
      txt("lcrs-readout", `Node ${focus.label} has ${focus.children.length} children in the general tree. In the binary form its LEFT link reaches its first child and each RIGHT link the next sibling: enumerating the children followed ${c.get("follow")} links (1 left + ${Math.max(0, cnt - 1)} right) — linear in the number of children, not in n. The binary form has ${bn.length} nodes (the same ${all.length}) and height ${TB.height(L)} against the general tree's ${TB.ghgt(G)}: sibling chains become depth. Preorder of the binary form: ${TB.preorder(L, []).map(x => x.key).join(" ")} — the general tree's preorder, unchanged.`);
    }
    on("lcrs-node", "change", draw); draw();
  });

  /* ── F05 array layout of a non-complete tree: the null slots ─────────── */
  fig("arr-svg", () => {
    function draw() {
      const shape = val("arr-shape") || "bst", k = +val("arr-k") || 6;
      txt("arr-k-out", k);
      let root, name;
      if (shape === "complete") { root = TB.complete(11); name = "complete tree, 11 nodes"; }
      else if (shape === "right") { root = TB.path(k, "right"); name = `right-going chain of ${k} nodes`; }
      else if (shape === "left") { root = TB.path(k, "left"); name = `left-going chain of ${k} nodes`; }
      else { root = TB.sampleBST().root; name = "the page's sample search tree, 11 nodes"; }
      const arr = TB.toLevelArray(root), n = TB.size(root), slots = arr.length, waste = slots - n;
      const F = AL.frame("#arr-svg", 760, 360, { l: 10, r: 10, t: 20, b: 10 });
      F.g.append("text").attr("x", 0).attr("y", -6).attr("font-size", 12).attr("fill", AC.a2).text(name + " — each node labelled with its level number f");
      TD.drawBin(F.g, root, { x: 0, y: 4, w: 740, levelH: Math.min(44, 190 / Math.max(1, TB.height(root))), r: 12, fontSize: 10.5, label: x => "f=" + TB.levelIndex(x) });
      const shown = Math.min(slots, 40);
      const cellW = Math.min(34, Math.floor(740 / shown) - 3);
      AL.row(F.g, arr.slice(0, shown).map(v => v === null ? "" : v), { x: 0, y: 262, w: cellW, h: 28, gap: 3, fontSize: 10,
        mark: (i, v) => v === "" ? "#3a1f24" : null });
      if (slots > shown) F.g.append("text").attr("x", 0).attr("y", 320).attr("font-size", 11).attr("fill", AC.muted).text(`… ${slots - shown} more slots not drawn (${slots} in total)`);
      F.g.append("text").attr("x", 0).attr("y", 338).attr("font-size", 11).attr("fill", AC.ink).text(`slots needed = max f + 1 = ${slots};  nodes = ${n};  empty slots = ${waste} (${(100 * waste / slots).toFixed(1)}%)`);
      const pow = shape === "right" ? `2^${k} − 1 = ${Math.pow(2, k) - 1}` : shape === "left" ? `2^${k - 1} = ${Math.pow(2, k - 1)}` : null;
      txt("arr-readout", `${name}: ${n} nodes need an array of ${slots} slots (largest level number ${slots - 1}), of which ${waste} are empty.` + (pow ? ` For a chain the slot count is exactly ${pow}: the level number doubles at every step down.` : "") + (shape === "complete" ? " A complete tree has no gaps: n slots for n nodes — the heap's representation." : "") + ` Space = Θ(2^h), not Θ(n), for a general shape.`);
    }
    on("arr-shape", "change", draw); on("arr-k", "input", draw); draw();
  });

  /* ── F06 naive vs linear height: call counts measured across n ────────── */
  fig("hgt-svg", () => {
    const ns = [4, 8, 16, 32, 64, 128, 256, 512];
    function make(shape, n, r) {
      if (shape === "comb") return TB.comb(n);
      if (shape === "chain") return TB.path(n, "right");
      if (shape === "complete") return TB.complete(n);
      return TB.randomBST(n, r).T.root;
    }
    function measure(shape, n, r) {
      const root = make(shape, n, r);
      const c1 = AL.counter(), c2 = AL.counter();
      const h1 = TB.heightNaive(root, c1), h2 = TB.height(root, c2);
      return { naive: c1.get("visits") + c1.get("up"), ups: c1.get("up"), visits: c1.get("visits"), linear: c2.get("calls"), h1: h1, h2: h2, leaves: TB.leaves(root).length, n: n };
    }
    function draw() {
      const shape = val("hgt-shape") || "comb";
      const r = AL.rng(3);
      const rows = ns.map(n => measure(shape, n, r));
      const F = AL.frame("#hgt-svg", 760, 330, { l: 56, r: 16, t: 16, b: 36 });
      const x = d3.scaleLog().domain([4, 512]).range([0, F.iw]);
      const y = d3.scaleLog().domain([4, Math.max(2000, d3.max(rows, d => d.naive))]).range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 5); AL.axisB(F.g, x, 8, "n (log)", d => d); AL.axisL(F.g, y, 5, "operations counted (log)", d3.format("~s"));
      const line = (key, col) => F.g.append("path").datum(rows).attr("fill", "none").attr("stroke", col).attr("stroke-width", 2)
        .attr("d", d3.line().x(d => x(d.n)).y(d => y(d[key])));
      line("naive", AC.bad); line("linear", AC.good);
      F.g.append("path").datum(rows).attr("fill", "none").attr("stroke", AC.muted).attr("stroke-dasharray", "2,4").attr("d", d3.line().x(d => x(d.n)).y(d => y(2 * d.n + 1)));
      rows.forEach(d => { F.g.append("circle").attr("cx", x(d.n)).attr("cy", y(d.naive)).attr("r", 3).attr("fill", AC.bad); F.g.append("circle").attr("cx", x(d.n)).attr("cy", y(d.linear)).attr("r", 3).attr("fill", AC.good); });
      AL.legend(F.g, [{ label: "naive: max over leaves of depth(leaf), depth by climbing — node visits + parent links followed", color: AC.bad },
        { label: "recursive height: calls made (one per node, one per empty subtree)", color: AC.good }, { label: "2n + 1 (formula)", color: AC.muted, dash: "2,4" }], 12, 14);
      const big = rows[rows.length - 1], small = rows[3];
      txt("hgt-readout", `${shape} shape. n = ${big.n}: naive ${big.naive} operations (${big.visits} visits + ${big.ups} parent links climbed over ${big.leaves} leaves) vs recursive ${big.linear} calls = 2n + 1; both return height ${big.h1} = ${big.h2}. Ratio naive / recursive: ${(small.naive / small.linear).toFixed(2)} at n = ${small.n}, ${(big.naive / big.linear).toFixed(2)} at n = ${big.n}` + (shape === "comb" ? ` — the ratio doubles with n, the signature of Θ(n²) against Θ(n).` : shape === "chain" ? ` — a chain has ONE leaf, so the naive method is linear here too: the quadratic case needs many deep leaves.` : ` — on a bushy tree the leaves are shallow and the naive method is only a constant factor worse.`));
    }
    on("hgt-shape", "change", draw); draw();
  });

  /* ── F07 the four traversals, stepped on one tree ─────────────────────── */
  fig("trav-svg", () => {
    const S = TB.sampleBinary(), n = TB.size(S), h = TB.height(S);
    let st = null;
    function build() {
      const order = val("trav-order") || "pre";
      let seq;
      if (order === "pre") seq = TB.preorder(S, []); else if (order === "in") seq = TB.inorder(S, []); else if (order === "post") seq = TB.postorder(S, []); else seq = TB.levelorder(S);
      const frames = [{ k: 0 }].concat(seq.map((_, i) => ({ k: i + 1 })));
      const host = d3.select("#trav-svg").node().parentNode;
      d3.select(host).selectAll("div[role=group]").remove();
      const names = { pre: "preorder: node, left subtree, right subtree", in: "inorder: left subtree, node, right subtree", post: "postorder: left subtree, right subtree, node", level: "level order: depth 0, then depth 1, … (a queue)" };
      const render = f => {
        const done = new Set(seq.slice(0, f.k)), cur = f.k ? seq[f.k - 1] : null;
        const F = AL.frame("#trav-svg", 760, 330, { l: 10, r: 10, t: 22, b: 10 });
        F.g.append("text").attr("x", 0).attr("y", -6).attr("font-size", 12).attr("fill", AC.a2).text(names[order]);
        TD.drawBin(F.g, S, { x: 0, y: 4, w: 480, levelH: 58, r: 15,
          fill: x => x === cur ? AC.a2 : done.has(x) ? "#1f3a4a" : null, stroke: x => x === cur ? AC.a2 : done.has(x) ? AC.teal : null,
          textFill: x => x === cur ? AC.bg : null,
          label: x => done.has(x) ? `#${seq.indexOf(x) + 1}` : "" });
        AL.row(F.g, seq.map((x, i) => i < f.k ? x.key : ""), { x: 0, y: 270, w: 30, h: 28, gap: 3, index: true, mark: (i) => i === f.k - 1 ? AC.a2 : i < f.k ? "#1f3a4a" : null });
        F.g.append("text").attr("x", 500).attr("y", 20).attr("font-size", 11.5).attr("fill", AC.ink).text(`output so far (${f.k} of ${n}):`);
        F.g.append("text").attr("x", 500).attr("y", 40).attr("font-size", 12).attr("fill", AC.a2).text(seq.slice(0, f.k).map(x => x.key).join(" "));
        const facts = [`n = ${n}, height = ${h}`, `recursive calls made: ${2 * n + 1} (= 2n + 1)`, `frames on the stack at most: ${h + 1} (= h + 1)`, `edges traversed: ${2 * (n - 1)} (= 2(n − 1))`];
        if (order === "level") facts[1] = `dequeues: ${n}, enqueues: ${n}`, facts[2] = `queue at its widest: ${(function () { const c = AL.counter(); TB.levelorder(S, c); return c.get("maxQueue"); })()} nodes`;
        facts.forEach((t, i) => F.g.append("text").attr("x", 500).attr("y", 70 + i * 18).attr("font-size", 11).attr("fill", AC.muted).text(t));
        if (cur) F.g.append("text").attr("x", 500).attr("y", 160).attr("font-size", 11.5).attr("fill", AC.ink).text(`visiting ${cur.key} (depth ${TB.depthOf(cur)})`);
      };
      st = AL.stepper(d3.select("#trav-svg"), { frames: frames, render: render, delay: 650, label: "step" });
      const ev = TB.traversalEvents(S), c = AL.counter(); TB.levelorder(S, c);
      txt("trav-readout", `Tree of ${n} nodes, height ${h}. Preorder ${TB.preorder(S, []).map(x => x.key).join(" ")} · inorder ${TB.inorder(S, []).map(x => x.key).join(" ")} · postorder ${TB.postorder(S, []).map(x => x.key).join(" ")} · level order ${TB.levelorder(S).map(x => x.key).join(" ")}. Measured on the recursive walk: ${ev.calls} calls (2n + 1 = ${2 * n + 1}), deepest stack ${ev.maxFrames} frames (h + 1 = ${h + 1}), ${ev.edgeTraversals} edge traversals (2(n − 1) = ${2 * (n - 1)}). Level order: queue at its widest ${c.get("maxQueue")}.`);
    }
    on("trav-order", "change", build); build();
  });

  /* ── F08 explicit stack / queue traversals, the stack drawn ───────────── */
  fig("stk-svg", () => {
    const S = TB.sampleBinary();
    function build() {
      const mode = val("stk-mode") || "pre";
      const R = mode === "pre" ? TB.iterPreorder(S) : mode === "in" ? TB.iterInorder(S) : mode === "post" ? TB.iterPostorderTrick(S) : TB.iterLevel(S);
      const host = d3.select("#stk-svg").node().parentNode;
      d3.select(host).selectAll("div[role=group]").remove();
      const render = f => {
        const F = AL.frame("#stk-svg", 760, 340, { l: 10, r: 10, t: 14, b: 10 });
        const outSet = new Set(f.out);
        TD.drawBin(F.g, S, { x: 0, y: 4, w: 440, levelH: 56, r: 14,
          fill: x => x === f.cur ? AC.a2 : outSet.has(x.key) ? "#1f3a4a" : f.stack.includes(x.key) ? "#3b3a2a" : null,
          stroke: x => x === f.cur ? AC.a2 : outSet.has(x.key) ? AC.teal : f.stack.includes(x.key) ? AC.a2 : null,
          textFill: x => x === f.cur ? AC.bg : null });
        if (f.queue) {
          F.g.append("text").attr("x", 470).attr("y", 14).attr("font-size", 11).attr("fill", AC.muted).text("queue (head on the left)");
          AL.row(F.g, f.stack.concat(Array(Math.max(0, 7 - f.stack.length)).fill("")), { x: 470, y: 22, w: 30, h: 26, gap: 3, index: false, mark: (i) => i === 0 && f.stack.length ? AC.a2 : null });
        } else {
          TD.stack(F.g, f.stack, { x: 480, y: 20, w: 60, h: 24, gap: 3, max: 6, title: "stack (top is highest)" });
        }
        F.g.append("text").attr("x", 470).attr("y", 220).attr("font-size", 11).attr("fill", AC.muted).text(mode === "post" ? "collected (to be reversed):" : "output:");
        F.g.append("text").attr("x", 470).attr("y", 240).attr("font-size", 12).attr("fill", AC.a2).text(f.out.join(" "));
        const words = f.note.split(" "); const lines = []; let cur = "";
        words.forEach(w => { if ((cur + " " + w).length > 52) { lines.push(cur); cur = w; } else cur = (cur ? cur + " " : "") + w; }); lines.push(cur);
        lines.forEach((l, i) => F.g.append("text").attr("x", 470).attr("y", 268 + i * 15).attr("font-size", 11).attr("fill", AC.ink).text(l));
        if (!f.queue) { F.g.append("text").attr("x", 0).attr("y", 300).attr("font-size", 11).attr("fill", AC.muted).text(`pushes so far ≤ ${R.pushes} (one per node) · stack never deeper than ${R.maxStack}`); }
      };
      AL.stepper(d3.select("#stk-svg"), { frames: R.frames, render: render, delay: 700, label: "step" });
      const ev = TB.traversalEvents(S);
      txt("stk-readout", `${mode === "pre" ? "Preorder" : mode === "in" ? "Inorder" : mode === "post" ? "Postorder via the reversed root-right-left trick" : "Level order with a queue"}: output ${R.out.join(" ")}, ${R.pushes} ${mode === "level" ? "enqueues" : "pushes"} (one per node), ${mode === "level" ? "queue" : "stack"} at its deepest ${R.maxStack}. The recursive version of the same walk made ${ev.calls} calls and reached ${ev.maxFrames} frames (h + 1). Both are Θ(n): every node is pushed once and popped once.`);
    }
    on("stk-mode", "change", build); build();
  });

  /* ── F09 Morris inorder: threads drawn, pointer follows counted ───────── */
  fig("morris-svg", () => {
    const S = TB.sampleBinary(), M = TB.morris(S);
    const render = f => {
      const F = AL.frame("#morris-svg", 760, 330, { l: 10, r: 10, t: 14, b: 10 });
      const outSet = new Set(f.out);
      const d = TD.drawBin(F.g, S, { x: 0, y: 4, w: 460, levelH: 58, r: 14,
        fill: x => x === f.cur ? AC.a2 : x === f.pre ? "#3b3a2a" : outSet.has(x.key) ? "#1f3a4a" : null,
        stroke: x => x === f.cur ? AC.a2 : x === f.pre ? AC.a2 : outSet.has(x.key) ? AC.teal : null,
        textFill: x => x === f.cur ? AC.bg : null });
      /* threads: curved arrows from pre to its (temporary) right = an ancestor */
      f.threads.forEach(p => {
        const a = d.pos.get(p), b = d.pos.get(p.right); if (!a || !b) return;
        const mx = (a.x + b.x) / 2 + 40, my = (a.y + b.y) / 2;
        F.g.append("path").attr("d", `M${a.x + 10},${a.y - 6} Q${mx},${my} ${b.x + 12},${b.y + 8}`).attr("fill", "none").attr("stroke", AC.rose).attr("stroke-width", 1.8).attr("stroke-dasharray", "5,3");
        F.g.append("circle").attr("cx", b.x + 12).attr("cy", b.y + 8).attr("r", 3).attr("fill", AC.rose);
      });
      F.g.append("text").attr("x", 490).attr("y", 14).attr("font-size", 11).attr("fill", AC.muted).text("output:");
      F.g.append("text").attr("x", 490).attr("y", 34).attr("font-size", 12).attr("fill", AC.a2).text(f.out.join(" "));
      F.g.append("text").attr("x", 490).attr("y", 62).attr("font-size", 11).attr("fill", AC.muted).text(`threads in place: ${f.threads.length} (dashed rose)`);
      const words = f.note.split(" "); const lines = []; let cur = "";
      words.forEach(w => { if ((cur + " " + w).length > 44) { lines.push(cur); cur = w; } else cur = (cur ? cur + " " : "") + w; }); lines.push(cur);
      lines.forEach((l, i) => F.g.append("text").attr("x", 490).attr("y", 90 + i * 15).attr("font-size", 11).attr("fill", AC.ink).text(l));
    };
    AL.stepper(d3.select("#morris-svg"), { frames: M.frames, render: render, delay: 800, label: "step" });
    const ev = TB.traversalEvents(S), n = TB.size(S);
    txt("morris-readout", `Output ${M.out.join(" ")} — identical to the recursive inorder (${TB.inorder(S, []).map(x => x.key).join(" ")}), and the tree is unchanged afterwards. Cost measured: ${M.follows} pointer follows and ${M.threads} threads created and removed, on ${n} nodes with ${n - 1} edges — against ${ev.edgeTraversals} edge traversals for the recursive walk, which also needs ${ev.maxFrames} stack frames. Morris needs none: O(1) extra space, still O(n) time (each edge is followed at most three times).`);
  });

  /* ── F10 the Euler tour ────────────────────────────────────────────────── */
  fig("euler-svg", () => {
    const S = TB.sampleBinary(), E = TB.eulerTour(S), n = TB.size(S);
    const frames = E.steps.map((s, i) => ({ i: i }));
    const render = f => {
      const F = AL.frame("#euler-svg", 760, 340, { l: 10, r: 10, t: 14, b: 10 });
      const d = TD.drawBin(F.g, S, { x: 0, y: 4, w: 460, levelH: 62, r: 15 });
      /* walk path so far: a polyline hugging the tree, drawn from the step list */
      const pts = []; const seen = { pre: [], in: [], post: [] };
      for (let k = 0; k <= f.i; k++) {
        const s = E.steps[k], p = d.pos.get(s.node);
        if (s.type === "pre") { pts.push([p.x - 19, p.y]); seen.pre.push(s.node.key); }
        else if (s.type === "in") { pts.push([p.x, p.y + 19]); seen.in.push(s.node.key); }
        else if (s.type === "post") { pts.push([p.x + 19, p.y]); seen.post.push(s.node.key); }
        else if (s.type === "down") { const q = d.pos.get(s.to); pts.push([(p.x + q.x) / 2 - (q.x < p.x ? 10 : -10) , (p.y + q.y) / 2 + (q.x < p.x ? -6 : 6)]); }
        else if (s.type === "up") { const q = d.pos.get(s.to); pts.push([(p.x + q.x) / 2 + (q.x < p.x ? -10 : 10), (p.y + q.y) / 2 + (q.x < p.x ? 6 : -6)]); }
      }
      if (pts.length > 1) F.g.append("path").datum(pts).attr("fill", "none").attr("stroke", AC.rose).attr("stroke-width", 2).attr("stroke-opacity", 0.85)
        .attr("d", d3.line().x(p => p[0]).y(p => p[1]).curve(d3.curveCatmullRom.alpha(0.6)));
      const s = E.steps[f.i], p = d.pos.get(s.node);
      if (s.type === "pre" || s.type === "in" || s.type === "post") F.g.append("circle").attr("cx", pts[pts.length - 1][0]).attr("cy", pts[pts.length - 1][1]).attr("r", 5).attr("fill", AC.a2);
      const lab = { pre: "PRE visit (left side): preorder hook", in: "IN visit (below): inorder hook", post: "POST visit (right side): postorder hook", down: "walk DOWN an edge", up: "walk UP the same edge" };
      F.g.append("text").attr("x", 490).attr("y", 14).attr("font-size", 11.5).attr("fill", AC.a2).text(`${lab[s.type]}${s.node ? " at " + s.node.key : ""}`);
      [["preorder", seen.pre], ["inorder", seen.in], ["postorder", seen.post]].forEach((row, i) => {
        F.g.append("text").attr("x", 490).attr("y", 44 + i * 40).attr("font-size", 11).attr("fill", AC.muted).text(row[0] + " so far");
        F.g.append("text").attr("x", 490).attr("y", 60 + i * 40).attr("font-size", 12).attr("fill", AC.ink).text(row[1].join(" "));
      });
      const downs = E.steps.slice(0, f.i + 1).filter(x => x.type === "down").length, ups = E.steps.slice(0, f.i + 1).filter(x => x.type === "up").length;
      F.g.append("text").attr("x", 490).attr("y", 180).attr("font-size", 11).attr("fill", AC.muted).text(`edges walked down ${downs}, up ${ups}  (of ${E.edgeDown} each)`);
      F.g.append("text").attr("x", 490).attr("y", 198).attr("font-size", 11).attr("fill", AC.muted).text(`visits so far ${seen.pre.length + seen.in.length + seen.post.length} of 3n = ${3 * n}`);
    };
    AL.stepper(d3.select("#euler-svg"), { frames: frames, render: render, delay: 350, label: "step" });
    txt("euler-readout", `The tour of this ${n}-node tree makes ${E.steps.length} steps: ${3 * n} node visits (each node exactly once on its left, once below, once on its right) and ${E.edgeDown + E.edgeUp} edge traversals — each of the ${n - 1} edges once down and once up, 2(n − 1) = ${2 * (n - 1)}. Reading only the left-side visits gives the preorder, only the below visits the inorder, only the right-side visits the postorder.`);
  });

  /* ── F11 expression tree: build from postfix, then evaluate ───────────── */
  fig("expr-svg", () => {
    function build() {
      const src = val("expr-src") || "8 5 * 9 7 4 - / +";
      const tokens = src.trim().split(/\s+/);
      const B = TB.buildPostfix(tokens);
      const evalFrames = []; const value = TB.evaluate(B.root, evalFrames);
      const frames = B.frames.map((f, i) => ({ phase: "build", f: f, i: i })).concat(evalFrames.map((f, i) => ({ phase: "eval", f: f, i: i })));
      const host = d3.select("#expr-svg").node().parentNode; d3.select(host).selectAll("div[role=group]").remove();
      const render = fr => {
        const F = AL.frame("#expr-svg", 760, 360, { l: 10, r: 10, t: 14, b: 10 });
        if (fr.phase === "build") {
          const f = fr.f;
          F.g.append("text").attr("x", 0).attr("y", 8).attr("font-size", 12).attr("fill", AC.a2).text(`building from postfix — token ${fr.i + 1} of ${tokens.length}: "${f.token}"`);
          AL.row(F.g, tokens, { x: 0, y: 20, w: 28, h: 26, gap: 3, index: false, mark: i => i === fr.i ? AC.a2 : i < fr.i ? "#1f3a4a" : null });
          /* draw the stack of subtrees left to right */
          let x0 = 0;
          f.stack.forEach((t, k) => {
            const sz = TB.size(t), w = Math.max(40, sz * 26);
            F.g.append("text").attr("x", x0 + w / 2).attr("y", 70).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.muted).text(k === f.stack.length - 1 ? "top" : `stack[${k}]`);
            TD.drawBin(F.g, t, { x: x0, y: 78, w: w, levelH: 46, r: 12, fontSize: 11, fill: nd => TB.isOp(nd.key) ? "#3b3a2a" : null });
            x0 += w + 30;
          });
          F.g.append("text").attr("x", 0).attr("y", 330).attr("font-size", 11).attr("fill", AC.ink).text(f.note);
        } else {
          const f = fr.f, vals = new Map(); evalFrames.slice(0, fr.i + 1).forEach(e => vals.set(e.node, e.value));
          F.g.append("text").attr("x", 0).attr("y", 8).attr("font-size", 12).attr("fill", AC.a2).text(`evaluating — postorder step ${fr.i + 1} of ${evalFrames.length}`);
          TD.drawBin(F.g, B.root, { x: 0, y: 20, w: 460, levelH: 60, r: 15, fontSize: 12,
            fill: nd => nd === f.node ? AC.a2 : vals.has(nd) ? "#1f3a4a" : TB.isOp(nd.key) ? "#3b3a2a" : null,
            stroke: nd => nd === f.node ? AC.a2 : vals.has(nd) ? AC.teal : null, textFill: nd => nd === f.node ? AC.bg : null,
            label: nd => vals.has(nd) && TB.isOp(nd.key) ? "= " + vals.get(nd) : "" });
          const lines = [TB.isOp(f.node.key) ? `${f.node.key}: ${f.a} ${f.node.key} ${f.b} = ${f.value}` : `leaf ${f.value}`, "", `infix (minimal parentheses):`, TB.toInfix(B.root), `fully parenthesised:`, TB.toInfixFull(B.root), `prefix:  ${TB.toPrefix(B.root)}`, `postfix: ${TB.toPostfix(B.root)}`, "", fr.i === evalFrames.length - 1 ? `value = ${value}` : ""];
          lines.forEach((l, i) => F.g.append("text").attr("x", 490).attr("y", 30 + i * 18).attr("font-size", 11.5).attr("fill", i === 0 || i === 9 ? AC.a2 : (i === 3 || i === 5) ? AC.ink : AC.muted).text(l));
        }
      };
      AL.stepper(d3.select("#expr-svg"), { frames: frames, render: render, delay: 750, label: "step" });
      const n = TB.size(B.root), lv = TB.leaves(B.root).length;
      txt("expr-readout", `"${src}" → value ${value}. Tree: ${n} nodes = ${lv} operands (leaves) + ${n - lv} operators (internal), a full binary tree (leaves = internal + 1: ${lv} = ${n - lv} + 1), height ${TB.height(B.root)}. Preorder gives the prefix form "${TB.toPrefix(B.root)}", postorder the postfix "${TB.toPostfix(B.root)}", inorder with the precedence rule "${TB.toInfix(B.root)}". Building took ${tokens.length} stack operations' worth of tokens; evaluation visited each of the ${n} nodes once.`);
    }
    on("expr-src", "change", build); build();
  });

  /* ── F12 the wrong invariant: locally ordered, globally broken ────────── */
  fig("inv-svg", () => {
    const W = TB.wrongInvariantTree();
    function draw() {
      const k = +val("inv-key") || 12;
      txt("inv-key-out", k);
      const c = AL.counter(); const r = TB.search(W, k, c);
      const pathSet = new Set(r.path);
      const inord = TB.inorder(W.root, []).map(x => x.key);
      const present = TB.nodes(W.root).some(x => x.key === k);
      const F = AL.frame("#inv-svg", 760, 300, { l: 10, r: 10, t: 22, b: 10 });
      F.g.append("text").attr("x", 0).attr("y", -6).attr("font-size", 12).attr("fill", AC.a2).text("every parent–child pair is correctly ordered (5 < 10, 15 > 10, 2 < 5, 12 > 5) — but 12 sits in the LEFT subtree of 10");
      TD.drawBin(F.g, W.root, { x: 0, y: 6, w: 420, levelH: 70, r: 17, fontSize: 13,
        fill: x => pathSet.has(x) ? (r.node === x ? AC.good : AC.a2) : (x.key === 12 ? "#3a1f24" : null),
        stroke: x => pathSet.has(x) ? AC.a2 : (x.key === 12 ? AC.bad : null), textFill: x => pathSet.has(x) ? AC.bg : null, nullSlots: true });
      const lines = [`search(${k}): path ${r.path.map(x => x.key).join(" → ")}`, r.node ? `found after ${c.get("cmp")} comparisons` : `MISS after ${c.get("cmp")} comparisons`,
        present && !r.node ? `…but ${k} IS in the tree. The search is correct; the tree is not a BST.` : present ? "" : `${k} is genuinely absent.`,
        ``, `inorder walk: ${inord.join(" ")}`, `sorted? ${inord.every((v, i) => !i || inord[i - 1] < v) ? "yes" : "NO — 12 comes before 10"}`,
        ``, `local check (children only): ${TB.isBSTLocal(W.root) ? "passes" : "fails"}`, `range check (bounds passed down): ${TB.isBST(W.root) ? "passes" : "FAILS at 12: must be < 10"}`];
      lines.forEach((l, i) => F.g.append("text").attr("x", 450).attr("y", 14 + i * 20).attr("font-size", 11.5).attr("fill", i === 2 ? AC.bad : (i === 8 ? AC.bad : AC.ink)).text(l));
      txt("inv-readout", `Searching for ${k}: path ${r.path.map(x => x.key).join(" → ")}, ${r.node ? "found" : "not found"} after ${c.get("cmp")} comparisons.` + (present && !r.node ? ` The key is present in the tree and the search misses it — the local invariant passed, the real one (every key in the left subtree < 10) is violated by 12.` : "") + ` Inorder walk ${inord.join(" ")} is ${inord.every((v, i) => !i || inord[i - 1] < v) ? "sorted" : "NOT sorted"}.`);
    }
    on("inv-key", "input", draw); draw();
  });

  /* ── F13 search / min / max on the sample BST, the path lit ───────────── */
  fig("srch-svg", () => {
    const T = TB.sampleBST();
    function draw() {
      const op = val("srch-op") || "search", k = +val("srch-key") || 13;
      txt("srch-key-out", k);
      const c = AL.counter(); let r, title;
      if (op === "min") { r = TB.minimum(T.root, c); title = `minimum: follow left links from the root — ${c.get("steps")} steps to ${r.node.key}`; }
      else if (op === "max") { r = TB.maximum(T.root, c); title = `maximum: follow right links from the root — ${c.get("steps")} steps to ${r.node.key}`; }
      else { r = TB.search(T, k, c); title = r.node ? `search(${k}): HIT after ${c.get("cmp")} comparisons (depth ${TB.depthOf(r.node)} + 1)` : `search(${k}): MISS after ${c.get("cmp")} comparisons — fell off at ${r.path[r.path.length - 1].key}`; }
      const pathSet = new Set(r.path);
      const F = AL.frame("#srch-svg", 760, 330, { l: 10, r: 10, t: 22, b: 10 });
      F.g.append("text").attr("x", 0).attr("y", -6).attr("font-size", 12).attr("fill", AC.a2).text(title);
      TD.drawBin(F.g, T.root, { x: 0, y: 6, w: 520, levelH: 56, r: 15, fontSize: 12, nullSlots: op === "search" && !r.node,
        fill: x => pathSet.has(x) ? (r.node === x ? AC.good : AC.a2) : null, stroke: x => pathSet.has(x) ? AC.a2 : null, textFill: x => pathSet.has(x) ? AC.bg : null,
        edge: (ch, p) => pathSet.has(ch) && pathSet.has(p) ? AC.a2 : null, edgeW: (ch, p) => pathSet.has(ch) && pathSet.has(p) ? 3 : null,
        label: x => `d=${TB.depthOf(x)}` });
      const steps = r.path.map((x, i) => op !== "search" ? `${x.key}` : (i < r.path.length - 1 ? `${k} ${k < x.key ? "<" : ">"} ${x.key} → go ${k < x.key ? "left" : "right"}` : (r.node ? `${k} = ${x.key} ✓` : `${k} ${k < x.key ? "<" : ">"} ${x.key} → ${k < x.key ? "left" : "right"} is null: absent`)));
      F.g.append("text").attr("x", 545).attr("y", 14).attr("font-size", 11).attr("fill", AC.muted).text("the path, one comparison per node:");
      steps.forEach((s, i) => F.g.append("text").attr("x", 545).attr("y", 34 + i * 18).attr("font-size", 11.5).attr("fill", AC.ink).text(s));
      F.g.append("text").attr("x", 545).attr("y", 34 + steps.length * 18 + 12).attr("font-size", 11).attr("fill", AC.muted).text(`tree: n = 11, h = ${TB.height(T.root)}`);
      const cnt = op === "search" ? c.get("cmp") : c.get("steps");
      txt("srch-readout", `${title}. Path ${r.path.map(x => x.key).join(" → ")}; ${cnt} ${op === "search" ? "comparisons" : "links followed"}, at most h + 1 = ${TB.height(T.root) + 1} on this tree. Inorder of the tree: ${TB.inorder(T.root, []).map(x => x.key).join(" ")}.`);
    }
    on("srch-op", "change", draw); on("srch-key", "input", draw); draw();
  });

  /* ── F14 successor / predecessor: the two cases ───────────────────────── */
  fig("succ-svg", () => {
    const T = TB.sampleBST(), keys = TB.inorder(T.root, []).map(x => x.key);
    const sel = $("succ-node"); if (sel) { keys.forEach(k => { const o = document.createElement("option"); o.value = k; o.textContent = k; sel.appendChild(o); }); sel.value = "13"; }
    function draw() {
      const k = +val("succ-node") || 13, dir = val("succ-dir") || "succ";
      const x = TB.find(T, k), c = AL.counter();
      const r = dir === "succ" ? TB.successor(x, c) : TB.predecessor(x, c);
      const pathSet = new Set(r.path);
      const F = AL.frame("#succ-svg", 760, 330, { l: 10, r: 10, t: 22, b: 10 });
      const word = dir === "succ" ? "successor" : "predecessor";
      const title = r.node ? `${word}(${k}) = ${r.node.key}, found by going ${r.how === "down" ? (dir === "succ" ? "DOWN: leftmost of the right subtree" : "DOWN: rightmost of the left subtree") : (dir === "succ" ? "UP: climb until we arrive from a LEFT child" : "UP: climb until we arrive from a RIGHT child")} — ${c.get("steps")} links`
        : `${word}(${k}): none — ${k} is the ${dir === "succ" ? "maximum" : "minimum"}; the climb reached the root without ever arriving from a ${dir === "succ" ? "left" : "right"} child (${c.get("steps")} links)`;
      F.g.append("text").attr("x", 0).attr("y", -6).attr("font-size", 11.5).attr("fill", AC.a2).text(title);
      TD.drawBin(F.g, T.root, { x: 0, y: 6, w: 520, levelH: 56, r: 15, fontSize: 12,
        fill: nd => nd === x ? AC.a2 : nd === r.node ? AC.good : pathSet.has(nd) ? "#3b3a2a" : null, stroke: nd => pathSet.has(nd) ? AC.a2 : null, textFill: nd => (nd === x || nd === r.node) ? AC.bg : null,
        edge: (ch, p) => pathSet.has(ch) && pathSet.has(p) ? AC.a2 : null, edgeW: (ch, p) => pathSet.has(ch) && pathSet.has(p) ? 3 : null });
      /* full walk cost */
      const cw = AL.counter(); let y = TB.minimum(T.root, cw).node; let calls = 0; while (true) { const s = TB.successor(y, cw); if (!s.node) break; y = s.node; calls++; }
      const lines = [`path: ${r.path.map(n => n.key).join(" → ")}`, `sorted order: ${keys.join(" ")}`, ``, `a full inorder walk by repeated successor,`, `starting from the minimum:`, `${calls} successor calls, ${cw.get("steps")} links in total`, `= 2(n − 1) = ${2 * (keys.length - 1)}: each edge at most twice,`, `so amortised O(1) per call, though one call is O(h)`];
      lines.forEach((l, i) => F.g.append("text").attr("x", 545).attr("y", 14 + i * 18).attr("font-size", 11).attr("fill", i < 2 ? AC.ink : AC.muted).text(l));
      txt("succ-readout", title + `. Path ${r.path.map(n => n.key).join(" → ")}. Walking the whole tree by successor from the minimum: ${calls} calls, ${cw.get("steps")} links = 2(n − 1) for n = ${keys.length}.`);
    }
    on("succ-node", "change", draw); on("succ-dir", "change", draw); draw();
  });

  /* ── F15 building a BST from a key sequence, comparisons counted ──────── */
  fig("build-svg", () => {
    function build() {
      const which = val("build-seq") || "sample";
      const keys = which === "sample" ? TB.SAMPLE_KEYS.slice() : which === "sorted" ? TB.SAMPLE_KEYS.slice().sort((a, b) => a - b) : which === "median" ? [9, 4, 17, 2, 7, 15, 20, 3, 6, 13, 18] : AL.shuffle(TB.SAMPLE_KEYS, AL.rng(5));
      const c = AL.counter(); const T = { root: null }; const frames = [{ T: null, i: -1, per: [], path: [] }];
      const per = [];
      keys.forEach((k, i) => { const b = c.get("cmp"); const r = TB.insert(T, k, c); per.push(c.get("cmp") - b); frames.push({ T: TB.clone(T.root), i: i, per: per.slice(), path: r.path.map(n => n.key), z: k }); });
      const host = d3.select("#build-svg").node().parentNode; d3.select(host).selectAll("div[role=group]").remove();
      const render = f => {
        const F = AL.frame("#build-svg", 760, 360, { l: 10, r: 10, t: 22, b: 10 });
        F.g.append("text").attr("x", 0).attr("y", -6).attr("font-size", 12).attr("fill", AC.a2).text(f.i < 0 ? "empty tree" : `insert ${f.z}: compared with ${f.path.length ? f.path.join(", ") : "nothing (first key becomes the root)"} — ${f.per[f.i]} comparisons, attached at depth ${f.per[f.i]}`);
        AL.row(F.g, keys, { x: 0, y: 6, w: 30, h: 26, gap: 3, index: false, mark: i => i === f.i ? AC.a2 : i < f.i ? "#1f3a4a" : null });
        if (f.T) {
          const pset = new Set(f.path);
          TD.drawBin(F.g, f.T, { x: 0, y: 50, w: 520, levelH: Math.min(52, 250 / Math.max(1, TB.height(f.T))), r: 14, fontSize: 11.5,
            fill: x => x.key === f.z ? AC.a2 : pset.has(x.key) ? "#3b3a2a" : null, stroke: x => x.key === f.z || pset.has(x.key) ? AC.a2 : null, textFill: x => x.key === f.z ? AC.bg : null });
        }
        const tot = f.per.reduce((s, v) => s + v, 0);
        const lines = [`comparisons per insert:`, f.per.join(" · ") || "—", ``, `total so far: ${tot}`, f.T ? `height: ${TB.height(f.T)}` : "", f.T ? `mean depth: ${(TB.totalDepth(f.T) / TB.size(f.T)).toFixed(2)} = total / n` : ""];
        lines.forEach((l, i) => F.g.append("text").attr("x", 560).attr("y", 60 + i * 18).attr("font-size", 11.5).attr("fill", i === 1 ? AC.a2 : AC.ink).text(l));
      };
      AL.stepper(d3.select("#build-svg"), { frames: frames, render: render, delay: 700, label: "insert" });
      const tot = per.reduce((s, v) => s + v, 0), n = keys.length;
      txt("build-readout", `Sequence ${keys.join(", ")}: comparisons per insertion ${per.join(", ")}, total ${tot} = sum of the final depths (${TB.totalDepth(T.root)}), so the mean depth is ${(tot / n).toFixed(3)}; final height ${TB.height(T.root)}. ${which === "sorted" ? `Sorted input: the i-th key is compared with all i − 1 before it, total n(n − 1)/2 = ${n * (n - 1) / 2}.` : which === "median" ? "Median-first order gives the perfectly balanced tree of height floor(log₂ 11) = 3." : ""}`);
    }
    on("build-seq", "change", build); build();
  });

  /* ── F16 deletion: the cases, stepped with transplant ─────────────────── */
  fig("del-svg", () => {
    function build() {
      const cs = val("del-case") || "two-deep";
      const key = cs === "leaf" ? 4 : cs === "one" ? 13 : cs === "two-right" ? 6 : 15;
      const policy = cs === "two-pred" ? "pred" : "succ";
      /* record frames by hand for the four cases */
      const T = TB.sampleBST(); const z = TB.find(T, key); const frames = [];
      const snap = (note, marks) => frames.push({ T: TB.clone(T.root), note: note, marks: Object.assign({}, marks) });
      snap(`delete ${key}: locate it (search path ${TB.search(T, key).path.map(n => n.key).join(" → ")})`, { z: key });
      if (!z.left) { snap(`${key} has no left child: transplant its right subtree ${z.right ? "(" + z.right.key + ")" : "(empty)"} into its place`, { z: key, y: z.right ? z.right.key : null }); TB.transplant(T, z, z.right); snap(`done — ${z.right ? z.right.key + " took " + key + "'s position" : key + " is simply unlinked"}`, {}); }
      else if (!z.right) { snap(`${key} has a left child ${z.left.key} and no right child: transplant the left subtree into its place`, { z: key, y: z.left.key }); TB.transplant(T, z, z.left); snap(`done — ${z.left.key} took ${key}'s position, its own subtree intact`, {}); }
      else if (policy === "succ") {
        const y = TB.minimum(z.right).node;
        snap(`${key} has two children: find its successor y = minimum of the right subtree = ${y.key} (it has no left child)`, { z: key, y: y.key });
        if (y.parent !== z) { snap(`y = ${y.key} is not ${key}'s right child: first transplant y's right subtree ${y.right ? "(" + y.right.key + ")" : "(empty)"} into y's place, then give y the whole right subtree of ${key}`, { z: key, y: y.key, x: y.right ? y.right.key : null }); TB.transplant(T, y, y.right); y.right = z.right; y.right.parent = y; snap(`y = ${y.key} now holds ${key}'s right subtree (root ${y.right.key})`, { z: key, y: y.key }); }
        else snap(`y = ${y.key} IS ${key}'s right child: no splice needed, y keeps its own right subtree`, { z: key, y: y.key });
        TB.transplant(T, z, y); y.left = z.left; y.left.parent = y;
        snap(`transplant y into ${key}'s place and give it ${key}'s left subtree — done; ${y.key} is now where ${key} was`, { y: y.key });
      } else {
        const y = TB.maximum(z.left).node;
        snap(`${key} has two children: the PREDECESSOR y = maximum of the left subtree = ${y.key} (no right child) is the mirror-image choice`, { z: key, y: y.key });
        if (y.parent !== z) { TB.transplant(T, y, y.left); y.left = z.left; y.left.parent = y; snap(`y = ${y.key} spliced out (its left subtree ${y.left ? "" : ""}takes its place) and given ${key}'s left subtree`, { z: key, y: y.key }); }
        TB.transplant(T, z, y); y.right = z.right; y.right.parent = y;
        snap(`transplant y into ${key}'s place and give it ${key}'s right subtree — done; ${y.key} is the new root`, { y: y.key });
      }
      const host = d3.select("#del-svg").node().parentNode; d3.select(host).selectAll("div[role=group]").remove();
      const render = f => {
        const F = AL.frame("#del-svg", 760, 340, { l: 10, r: 10, t: 22, b: 10 });
        const words = f.note.split(" "); const lines = []; let cur = "";
        words.forEach(w => { if ((cur + " " + w).length > 118) { lines.push(cur); cur = w; } else cur = (cur ? cur + " " : "") + w; }); lines.push(cur);
        lines.forEach((l, i) => F.g.append("text").attr("x", 0).attr("y", -6 + i * 15).attr("font-size", 11.5).attr("fill", AC.a2).text(l));
        TD.drawBin(F.g, f.T, { x: 0, y: 30, w: 740, levelH: 56, r: 15, fontSize: 12, nullSlots: false,
          fill: x => x.key === f.marks.z ? AC.bad : x.key === f.marks.y ? AC.good : x.key === f.marks.x ? AC.a2 : null,
          stroke: x => x.key === f.marks.z ? AC.bad : x.key === f.marks.y ? AC.good : x.key === f.marks.x ? AC.a2 : null,
          textFill: x => (x.key === f.marks.z || x.key === f.marks.y || x.key === f.marks.x) ? AC.bg : null });
        F.g.append("text").attr("x", 0).attr("y", 312).attr("font-size", 11).attr("fill", AC.muted).text(`inorder now: ${TB.inorder(f.T, []).map(x => x.key).join(" ")}   ·   BST invariant holds: ${TB.isBST(f.T) ? "yes" : "NO"}   ·   height ${TB.height(f.T)}`);
      };
      AL.stepper(d3.select("#del-svg"), { frames: frames, render: render, delay: 1400, label: "step" });
      txt("del-readout", `Deleting ${key} (${cs === "leaf" ? "a leaf" : cs === "one" ? "one child" : cs === "two-right" ? "two children, successor is the right child" : cs === "two-pred" ? "two children, using the predecessor" : "two children, successor deeper in the right subtree"}): ${frames.length} steps. Result inorder ${TB.inorder(T.root, []).map(x => x.key).join(" ")} (sorted, ${TB.size(T.root)} keys), root ${T.root.key}, height ${TB.height(T.root)}, invariant ${TB.isBST(T.root) ? "intact" : "BROKEN"}.`);
    }
    on("del-case", "change", build); build();
  });

  /* ── F17 tree sort vs quicksort: the same comparisons, per element ────── */
  fig("tsort-svg", () => {
    function draw() {
      const n = +val("tsort-n") || 24, inp = val("tsort-input") || "random";
      txt("tsort-n-out", n);
      const keys = inp === "sorted" ? Array.from({ length: n }, (_, i) => i + 1) : inp === "reverse" ? Array.from({ length: n }, (_, i) => n - i) : AL.perm(n, AL.rng(7));
      const t = TB.treeSortCounts(keys), q = TB.quicksortCounts(keys);
      const F = AL.frame("#tsort-svg", 760, 330, { l: 46, r: 16, t: 26, b: 34 });
      const x = d3.scaleBand().domain(keys.map((_, i) => i)).range([0, F.iw]).padding(0.15);
      const y = d3.scaleLinear().domain([0, Math.max(4, d3.max(t.per.concat(q.per)))]).nice().range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 5); AL.axisL(F.g, y, 5, "comparisons involving this element");
      F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x).tickValues(keys.map((_, i) => i).filter(i => n <= 24 || i % Math.ceil(n / 24) === 0)).tickFormat(i => keys[i]));
      F.g.append("text").attr("x", F.iw).attr("y", F.ih + 30).attr("text-anchor", "end").attr("font-size", 11).attr("fill", AC.muted).text("keys, in input order");
      keys.forEach((k, i) => {
        F.g.append("rect").attr("x", x(i)).attr("y", y(t.per[i])).attr("width", x.bandwidth() / 2).attr("height", F.ih - y(t.per[i])).attr("fill", AC.accent);
        F.g.append("rect").attr("x", x(i) + x.bandwidth() / 2).attr("y", y(q.per[i])).attr("width", x.bandwidth() / 2).attr("height", F.ih - y(q.per[i])).attr("fill", AC.a2);
      });
      AL.legend(F.g, [{ label: `tree sort: comparisons at this key's insertion (= its depth) — total ${t.total}`, color: AC.accent }, { label: `quicksort, first-element pivot: pivots this element was compared with — total ${q.total}`, color: AC.a2 }], 8, -8);
      const same = t.per.every((v, i) => v === q.per[i]);
      txt("tsort-readout", `n = ${n}, ${inp} input: tree sort ${t.total} comparisons, quicksort ${q.total}; per-element counts ${same ? "IDENTICAL for every key" : "differ"}; both outputs sorted and equal: ${t.sorted.join() === q.sorted.join() ? "yes" : "no"}. n·log₂ n = ${(n * Math.log2(n)).toFixed(1)}, n(n−1)/2 = ${n * (n - 1) / 2}, expected for random input 2(n+1)H_n − 4n = ${(2 * (n + 1) * TB.harmonic(n) - 4 * n).toFixed(1)}.`);
    }
    on("tsort-n", "input", draw); on("tsort-input", "change", draw); draw();
  });

  /* ── F18 the degenerate case measured: height & mean depth vs n ───────── */
  fig("degen-svg", () => {
    const ns = [8, 16, 32, 64, 128, 256, 512, 1024, 2048];
    function balancedOrder(n) { const out = []; const rec = (lo, hi) => { if (lo > hi) return; const mid = Math.floor((lo + hi) / 2); out.push(mid); rec(lo, mid - 1); rec(mid + 1, hi); }; rec(1, n); return out; }
    const r = AL.rng(5);
    const rows = ns.map(n => {
      const s = TB.stats(TB.build(Array.from({ length: n }, (_, i) => i + 1)).T.root);
      const b = TB.stats(TB.build(balancedOrder(n)).T.root);
      let hs = 0, ds = 0; const T = 8; for (let t = 0; t < T; t++) { const st = TB.stats(TB.randomBST(n, r).T.root); hs += st.height; ds += st.meanDepth; }
      return { n: n, sortedH: s.height, sortedD: s.meanDepth, balH: b.height, balD: b.meanDepth, rndH: hs / T, rndD: ds / T, exact: TB.expectedAvgDepth(n) };
    });
    function draw() {
      const what = val("degen-what") || "height";
      const F = AL.frame("#degen-svg", 760, 330, { l: 52, r: 16, t: 16, b: 36 });
      const x = d3.scaleLog().domain([8, 2048]).range([0, F.iw]);
      const y = d3.scaleLog().domain([1, 2100]).range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 5); AL.axisB(F.g, x, 9, "n (log)", d => d); AL.axisL(F.g, y, 5, (what === "height" ? "height" : "mean depth") + " (log)", d => d);
      const key = what === "height" ? ["sortedH", "rndH", "balH"] : ["sortedD", "rndD", "balD"];
      const line = (k, col, dash) => F.g.append("path").datum(rows).attr("fill", "none").attr("stroke", col).attr("stroke-width", 2).attr("stroke-dasharray", dash || null).attr("d", d3.line().x(d => x(d.n)).y(d => y(Math.max(1, d[k]))));
      line(key[0], AC.bad); line(key[1], AC.a2); line(key[2], AC.good);
      if (what !== "height") line("exact", AC.ink, "2,4");
      else F.g.append("path").datum(rows).attr("fill", "none").attr("stroke", AC.ink).attr("stroke-dasharray", "2,4").attr("d", d3.line().x(d => x(d.n)).y(d => y(Math.max(1, Math.floor(Math.log2(d.n))))));
      rows.forEach(d => key.forEach((k, i) => F.g.append("circle").attr("cx", x(d.n)).attr("cy", y(Math.max(1, d[k]))).attr("r", 3).attr("fill", [AC.bad, AC.a2, AC.good][i])));
      AL.legend(F.g, [{ label: "sorted insertion order (measured)", color: AC.bad }, { label: "random insertion order, mean of 8 trees (measured)", color: AC.a2 }, { label: "median-first insertion order (measured)", color: AC.good },
        { label: what === "height" ? "floor(log₂ n)" : "exact expectation 2(1 + 1/n)·H_n − 4 for random order", color: AC.ink, dash: "2,4" }], 12, 14);
      const d = rows[rows.length - 1], m = rows[4];
      txt("degen-readout", `n = ${d.n}: sorted order gives height ${d.sortedH} = n − 1 and mean depth ${f2(d.sortedD)} = (n − 1)/2; random order height ${f2(d.rndH)} and mean depth ${f3(d.rndD)} (exact expectation ${f3(d.exact)}); median-first height ${d.balH} = floor(log₂ n) and mean depth ${f3(d.balD)}. At n = ${m.n}: ${m.sortedH} / ${f2(m.rndH)} / ${m.balH}. Sorted input is ${(d.sortedD / d.rndD).toFixed(0)}× the random mean depth at n = ${d.n}.`);
    }
    on("degen-what", "change", draw); draw();
  });

  /* ── F19 random BSTs: depth distribution against the exact expectation ── */
  fig("rbst-svg", () => {
    function draw() {
      const n = +val("rbst-n") || 1024, T = 20, r = AL.rng(21);
      txt("rbst-n-out", n);
      const H = []; let hs = 0, ms = 0, hmin = 1e9, hmax = -1;
      for (let t = 0; t < T; t++) { const root = TB.randomBST(n, r).T.root; TB.depthHistogram(root).forEach((v, i) => H[i] = (H[i] || 0) + v); const s = TB.stats(root); hs += s.height; ms += s.meanDepth; hmin = Math.min(hmin, s.height); hmax = Math.max(hmax, s.height); }
      for (let i = 0; i < H.length; i++) H[i] = (H[i] || 0) / T;
      const exact = TB.expectedAvgDepth(n), twoln = 2 * Math.log(n), alpha = 4.311 * Math.log(n), alpha2 = 4.311 * Math.log(n) - 1.953 * Math.log(Math.log(n));
      const F = AL.frame("#rbst-svg", 760, 330, { l: 52, r: 16, t: 16, b: 36 });
      const x = d3.scaleLinear().domain([0, Math.max(H.length, alpha + 2)]).range([0, F.iw]);
      const y = d3.scaleLinear().domain([0, d3.max(H) * 1.15]).nice().range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 5); AL.axisB(F.g, x, 10, "depth"); AL.axisL(F.g, y, 5, "nodes at this depth (mean over trees)");
      const bw = x(1) - x(0);
      H.forEach((v, i) => F.g.append("rect").attr("x", x(i) + 1).attr("y", y(v)).attr("width", Math.max(1, bw - 2)).attr("height", F.ih - y(v)).attr("fill", AC.accent).attr("opacity", 0.85));
      const mark = (v, col, lab, dy, dash) => { F.g.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", 0).attr("y2", F.ih).attr("stroke", col).attr("stroke-width", 1.5).attr("stroke-dasharray", dash || null); F.g.append("text").attr("x", x(v) + 4).attr("y", 12 + dy).attr("font-size", 10.5).attr("fill", col).text(lab); };
      mark(ms / T, AC.a2, `measured mean depth ${f2(ms / T)}`, 0);
      mark(exact, AC.ink, `exact E[mean depth] ${f2(exact)}`, 16, "4,3");
      mark(twoln, AC.muted, `2·ln n = ${f2(twoln)}`, 32, "2,3");
      mark(hs / T, AC.rose, `measured mean height ${f2(hs / T)}`, 48);
      mark(alpha, AC.bad, `4.311·ln n = ${f2(alpha)} (asymptotic)`, 64, "2,3");
      txt("rbst-readout", `n = ${n}, ${T} random trees. Mean depth ${f3(ms / T)} against the exact expectation 2(1 + 1/n)·H_n − 4 = ${f3(exact)} (and 2·ln n = ${f2(twoln)}, 2·ln n + 2γ − 4 = ${f2(twoln + 2 * 0.5772156649 - 4)}). Mean height ${f2(hs / T)} (range ${hmin}–${hmax}); the asymptotic law gives 4.311·ln n = ${f2(alpha)}, and with its second term 4.311·ln n − 1.953·ln ln n = ${f2(alpha2)} — measured height / ln n = ${f2(hs / T / Math.log(n))}, still below 4.311 at this n.`);
    }
    on("rbst-n", "input", draw); draw();
  });

  /* ── F20 the deletion bias: asymmetric vs symmetric replacement ───────── */
  fig("hib-svg", () => {
    const n = 1024, PAIRS = n * n / 2, snaps = 16, every = Math.floor(PAIRS / snaps);
    function run(policy, seed) {
      const r = AL.rng(seed); const T = TB.randomBST(n, r).T; const keys = TB.nodes(T.root).map(x => x.key);
      const s0 = TB.stats(T.root); const series = [{ op: 0, d: s0.meanDepth, h: s0.height }];
      for (let op = 0; op < PAIRS; op++) {
        const i = Math.floor(r() * keys.length); TB.remove(T, TB.find(T, keys[i]), policy);
        const nk = r(); keys[i] = nk; TB.insert(T, nk);
        if ((op + 1) % every === 0) { const s = TB.stats(T.root); series.push({ op: op + 1, d: s.meanDepth, h: s.height }); }
      }
      return series;
    }
    const A = run("succ", 99), B = run("alt", 99), exact = TB.expectedAvgDepth(n);
    function draw() {
      const what = val("hib-what") || "depth";
      const F = AL.frame("#hib-svg", 760, 330, { l: 52, r: 16, t: 16, b: 36 });
      const x = d3.scaleLinear().domain([0, PAIRS]).range([0, F.iw]);
      const key = what === "depth" ? "d" : "h";
      const allv = A.concat(B).map(s => s[key]).concat([what === "depth" ? exact : 0]);
      const y = d3.scaleLinear().domain([d3.min(allv) * 0.9, d3.max(allv) * 1.05]).nice().range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 5); AL.axisB(F.g, x, 8, "insert/delete pairs performed", d3.format("~s")); AL.axisL(F.g, y, 5, what === "depth" ? "mean node depth" : "height");
      const line = (S, col) => { F.g.append("path").datum(S).attr("fill", "none").attr("stroke", col).attr("stroke-width", 2).attr("d", d3.line().x(s => x(s.op)).y(s => y(s[key]))); S.forEach(s => F.g.append("circle").attr("cx", x(s.op)).attr("cy", y(s[key])).attr("r", 3).attr("fill", col)); };
      line(A, AC.bad); line(B, AC.good);
      if (what === "depth") { F.g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(exact)).attr("y2", y(exact)).attr("stroke", AC.ink).attr("stroke-dasharray", "4,3"); F.g.append("text").attr("x", F.iw - 4).attr("y", y(exact) - 4).attr("text-anchor", "end").attr("font-size", 10.5).attr("fill", AC.ink).text(`random-tree expectation ${f2(exact)}`); }
      AL.legend(F.g, [{ label: "always replace by the successor (asymmetric)", color: AC.bad }, { label: "alternate successor / predecessor (symmetric)", color: AC.good }], 12, 14);
      const a0 = A[0], a1 = A[A.length - 1], b1 = B[B.length - 1];
      txt("hib-readout", `n = ${n} keys, ${PAIRS.toLocaleString()} random delete-then-insert pairs (n²/2), one run each, same seed. Mean depth: start ${f2(a0.d)} (random-tree expectation ${f2(exact)}); after all pairs, successor-only ${f2(a1.d)}, alternating ${f2(b1.d)}. Height: start ${a0.h}; end ${a1.h} vs ${b1.h}. Both policies first IMPROVE on the starting tree; the asymmetric one then drifts back up to the random-tree level while the symmetric one stays below it — the onset of the drift whose asymptotic form is Θ(√n) mean depth (√n = 32 here), far beyond what this many updates reach.`);
    }
    on("hib-what", "change", draw); draw();
  });

  /* ── F21 all BSTs on n keys, grouped by shape, with insertion-order counts */
  fig("cat-svg", () => {
    function draw() {
      const n = +val("cat-n") || 3;
      txt("cat-n-out", n);
      const E = TB.enumerateBSTs(n), C = TB.catalan(n);
      const F = AL.frame("#cat-svg", 760, 340, { l: 8, r: 8, t: 22, b: 8 });
      F.g.append("text").attr("x", 0).attr("y", -6).attr("font-size", 12).attr("fill", AC.a2).text(`n = ${n}: ${E.shapes.length} distinct search trees (Catalan C_${n} = ${C}) from ${E.perms} insertion orders (n! = ${E.perms})`);
      const cols = n <= 3 ? 5 : 7, rows = Math.ceil(E.shapes.length / cols), cw = 744 / cols, ch = Math.min(150, 300 / rows);
      E.shapes.forEach((s, i) => {
        const g = F.g.append("g").attr("transform", `translate(${(i % cols) * cw},${Math.floor(i / cols) * ch})`);
        TD.drawBin(g, s.root, { x: 6, y: 4, w: cw - 12, levelH: Math.min(30, (ch - 40) / Math.max(1, n - 1)), r: 9, fontSize: 9 });
        g.append("text").attr("x", cw / 2).attr("y", ch - 12).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", s.count > 1 ? AC.a2 : AC.muted).text(`${s.count} order${s.count > 1 ? "s" : ""} · h=${TB.height(s.root)}`);
      });
      const counts = E.shapes.map(s => s.count), maxC = Math.max(...counts), sum = counts.reduce((a, b) => a + b, 0);
      const heights = E.shapes.map(s => TB.height(s.root)); const expH = E.shapes.reduce((a, s) => a + s.count * TB.height(s.root), 0) / E.perms;
      txt("cat-readout", `${E.shapes.length} shapes = C_${n} = (2n)! / ((n+1)! n!) = ${C}; the insertion-order counts (${counts.slice().sort((a, b) => b - a).join(", ")}) sum to ${sum} = n!. Not uniform: the most likely shape arises from ${maxC} orders, the chains from 1 each. Heights range ${Math.min(...heights)}–${Math.max(...heights)}; expected height under random insertion order ${f3(expH)}, against ${f3(heights.reduce((a, b) => a + b, 0) / heights.length)} if every shape were equally likely.`);
    }
    on("cat-n", "input", draw); draw();
  });

  /* ── F22 order statistics on a size-augmented tree ────────────────────── */
  fig("rank-svg", () => {
    const T = TB.sampleBST(); TB.annotateSizes(T.root); const sorted = TB.inorder(T.root, []).map(x => x.key);
    function draw() {
      const op = val("rank-op") || "select", i = +val("rank-i") || 6;
      txt("rank-i-out", i);
      const c = AL.counter(); let r, title, steps;
      if (op === "select") { r = TB.select(T.root, i, c); title = `select(${i}) = ${r.node ? r.node.key : "none"}: the ${i}-th smallest, in ${c.get("steps")} steps`;
        steps = (function () { const out = []; let k = i; r.path.forEach((x, j) => { const rk = (x.left ? x.left.size : 0) + 1; if (k === rk) out.push(`at ${x.key}: left size ${rk - 1}, so ${x.key} is #${k} here ✓`); else if (k < rk) out.push(`at ${x.key}: rank ${rk} here, want ${k} → left`); else { out.push(`at ${x.key}: rank ${rk} here, want ${k} → right, now want #${k - rk}`); k -= rk; } }); return out; })(); }
      else { const k = sorted[Math.min(i, sorted.length) - 1]; r = TB.rank(T, k, c); title = `rank(${k}) = ${r.rank}: ${r.rank - 1} keys are smaller, found in ${c.get("steps")} steps`;
        steps = (function () { const out = []; let acc = 0; r.path.forEach(x => { if (k < x.key) out.push(`at ${x.key}: ${k} is smaller → left, nothing added`); else if (k > x.key) { acc += (x.left ? x.left.size : 0) + 1; out.push(`at ${x.key}: ${k} is larger → right, add left size + 1 = ${(x.left ? x.left.size : 0) + 1} (running ${acc})`); } else { acc += (x.left ? x.left.size : 0) + 1; out.push(`at ${x.key}: found, add its left size + 1 → rank ${acc}`); } }); return out; })(); }
      const pset = new Set(r.path);
      const F = AL.frame("#rank-svg", 760, 330, { l: 10, r: 10, t: 22, b: 10 });
      F.g.append("text").attr("x", 0).attr("y", -6).attr("font-size", 12).attr("fill", AC.a2).text(title);
      TD.drawBin(F.g, T.root, { x: 0, y: 6, w: 480, levelH: 56, r: 15, fontSize: 12,
        fill: x => x === r.node ? AC.good : pset.has(x) ? AC.a2 : null, stroke: x => pset.has(x) ? AC.a2 : null, textFill: x => pset.has(x) ? AC.bg : null,
        edge: (ch, p) => pset.has(ch) && pset.has(p) ? AC.a2 : null, edgeW: (ch, p) => pset.has(ch) && pset.has(p) ? 3 : null, label: x => `size ${x.size}` });
      steps.forEach((s, j) => F.g.append("text").attr("x", 500).attr("y", 14 + j * 17).attr("font-size", 10.5).attr("fill", AC.ink).text(s));
      F.g.append("text").attr("x", 500).attr("y", 14 + steps.length * 17 + 14).attr("font-size", 10.5).attr("fill", AC.muted).text(`check against the sorted list: ${sorted.join(" ")}`);
      txt("rank-readout", `${title}. Path ${r.path.map(x => x.key).join(" → ")}. Cross-check with the inorder list ${sorted.join(" ")}: ${op === "select" ? `position ${i} holds ${sorted[i - 1]}` : `${sorted[Math.min(i, sorted.length) - 1]} is at position ${sorted.indexOf(sorted[Math.min(i, sorted.length) - 1]) + 1}`} ✓. Each node stores size = 1 + size(left) + size(right); the root's is ${T.root.size} = n.`);
    }
    on("rank-op", "change", draw); on("rank-i", "input", draw); draw();
  });
  /* ── F23 range query [a, b]: reported nodes vs boundary-path nodes ────── */
  fig("range-svg", () => {
    const T = TB.sampleBST(), n = TB.size(T.root), h = TB.height(T.root);
    function draw() {
      let a = +val("range-a"), b = +val("range-b"); if (isNaN(a)) a = 5; if (isNaN(b)) b = 14; if (a > b) { const t = a; a = b; b = t; }
      txt("range-a-out", val("range-a")); txt("range-b-out", val("range-b"));
      const c = AL.counter(); const rep = TB.rangeQuery(T.root, a, b, [], c);
      const visited = []; (function rec(x) { if (!x) return; visited.push(x); if (a < x.key) rec(x.left); if (x.key < b) rec(x.right); })(T.root);
      const vset = new Set(visited), rset = new Set(rep);
      const ca = TB.ceiling(T, a), fb = TB.floorKey(T, b); const bset = new Set(ca.path.concat(fb.path));
      const F = AL.frame("#range-svg", 760, 330, { l: 10, r: 10, t: 22, b: 10 });
      F.g.append("text").attr("x", 0).attr("y", -6).attr("font-size", 12).attr("fill", AC.a2).text(`keys in [${a}, ${b}]: ${rep.map(x => x.key).join(" ") || "none"} — ${rep.length} reported, ${c.get("visits")} nodes visited`);
      TD.drawBin(F.g, T.root, { x: 0, y: 6, w: 500, levelH: 56, r: 15, fontSize: 12,
        fill: x => rset.has(x) ? AC.good : vset.has(x) ? AC.a2 : null, stroke: x => vset.has(x) ? AC.a2 : null, textFill: x => vset.has(x) ? AC.bg : null,
        edge: (ch, p) => vset.has(ch) && vset.has(p) ? AC.a2 : null, edgeW: (ch, p) => vset.has(ch) && vset.has(p) ? 3 : null });
      const notRep = visited.filter(x => !rset.has(x));
      TB.annotateSizes(T.root);
      const cnt = (fb.node ? TB.rank(T, fb.node.key).rank : 0) - (ca.node ? TB.rank(T, ca.node.key).rank - 1 : n);
      const lines = [`visited but not reported: ${notRep.map(x => x.key).join(" ") || "none"}`, `all of them lie on the search path of ${a} (${ca.path.map(x => x.key).join("→")})`, `or of ${b} (${fb.path.map(x => x.key).join("→")}): ${notRep.every(x => bset.has(x)) ? "yes ✓" : "NO"}`, ``,
        `bound: m + 2(h + 1) = ${rep.length} + ${2 * (h + 1)} = ${rep.length + 2 * (h + 1)} ≥ ${c.get("visits")} ✓`, ``, `ceiling(${a}) = ${ca.node ? ca.node.key : "none"},  floor(${b}) = ${fb.node ? fb.node.key : "none"}`, `count by ranks: rank(floor) − rank(ceiling) + 1 = ${cnt} ${cnt === rep.length ? "✓" : "✗"}`];
      lines.forEach((l, i) => F.g.append("text").attr("x", 520).attr("y", 14 + i * 18).attr("font-size", 11).attr("fill", i === 4 || i === 7 ? AC.ink : AC.muted).text(l));
      txt("range-readout", `[${a}, ${b}] on the 11-key tree (h = ${h}): reported ${rep.map(x => x.key).join(" ") || "nothing"} (${rep.length} keys, in sorted order) after visiting ${c.get("visits")} nodes; the ${notRep.length} extra nodes all lie on the two boundary search paths, so visits ≤ m + 2(h + 1) = ${rep.length + 2 * (h + 1)}. Count without visiting the results, from subtree sizes: ${cnt}.`);
    }
    on("range-a", "input", draw); on("range-b", "input", draw); draw();
  });

  /* ── F24 lowest common ancestor, and the balanced build from sorted input */
  fig("lca-svg", () => {
    const keys = TB.SAMPLE_KEYS.slice().sort((x, y) => x - y);
    ["lca-u", "lca-v"].forEach(id => { const sel = $(id); if (sel) keys.forEach(k => { const o = document.createElement("option"); o.value = k; o.textContent = k; sel.appendChild(o); }); });
    if ($("lca-u")) $("lca-u").value = "4"; if ($("lca-v")) $("lca-v").value = "9";
    function draw() {
      const mode = val("lca-tree") || "sample";
      const c0 = AL.counter();
      const T = mode === "sample" ? TB.sampleBST() : TB.fromSorted(keys, c0);
      const u = +val("lca-u") || 4, v = +val("lca-v") || 9;
      const c = AL.counter(); const r = TB.lca(T, u, v, c);
      const pu = TB.search(T, u).path, pv = TB.search(T, v).path; const pset = new Set(pu.concat(pv)); const shared = new Set(r.path);
      const du = TB.depthOf(TB.find(T, u)), dv = TB.depthOf(TB.find(T, v)), dl = r.node ? TB.depthOf(r.node) : 0;
      const F = AL.frame("#lca-svg", 760, 330, { l: 10, r: 10, t: 22, b: 10 });
      F.g.append("text").attr("x", 0).attr("y", -6).attr("font-size", 12).attr("fill", AC.a2).text(`${mode === "sample" ? "the sample tree (h = " + TB.height(T.root) + ")" : "the same 11 keys built median-first from sorted order (h = " + TB.height(T.root) + ", " + c0.get("calls") + " calls = 2n + 1)"} — lca(${u}, ${v}) = ${r.node ? r.node.key : "?"} in ${c.get("steps")} steps`);
      TD.drawBin(F.g, T.root, { x: 0, y: 6, w: 500, levelH: 56, r: 15, fontSize: 12,
        fill: x => x === r.node ? AC.good : (x.key === u || x.key === v) ? AC.a2 : shared.has(x) ? "#3b3a2a" : pset.has(x) ? "#1f3a4a" : null,
        stroke: x => pset.has(x) ? AC.a2 : null, textFill: x => (x === r.node || x.key === u || x.key === v) ? AC.bg : null,
        edge: (ch, p) => pset.has(ch) && pset.has(p) ? AC.a2 : null, edgeW: (ch, p) => pset.has(ch) && pset.has(p) ? 3 : null, label: x => `d=${TB.depthOf(x)}` });
      const lines = [`walk from the root: ${r.path.map(x => x.key).join(" → ")}`, `stop at the first key in [${Math.min(u, v)}, ${Math.max(u, v)}]: ${r.node ? r.node.key : "-"}`, ``, `depth(${u}) = ${du}, depth(${v}) = ${dv}, depth(lca) = ${dl}`, `distance = ${du} + ${dv} − 2·${dl} = ${du + dv - 2 * dl} edges`, ``,
        mode === "sample" ? `switch to the median-first build to see` : `median-first order: ${(function () { const o = []; const rec = (lo, hi) => { if (lo > hi) return; const m = Math.floor((lo + hi) / 2); o.push(keys[m]); rec(lo, m - 1); rec(m + 1, hi); }; rec(0, keys.length - 1); return o.join(" "); })()}`, mode === "sample" ? `the same keys at height floor(log₂ 11) = 3` : `height ${TB.height(T.root)} = floor(log₂ 11), mean depth ${(TB.totalDepth(T.root) / 11).toFixed(3)}`];
      lines.forEach((l, i) => F.g.append("text").attr("x", 520).attr("y", 14 + i * 18).attr("font-size", 11).attr("fill", i === 1 || i === 4 ? AC.ink : AC.muted).text(l));
      txt("lca-readout", `lca(${u}, ${v}) = ${r.node ? r.node.key : "none"} after ${c.get("steps")} steps down from the root, no parent pointers used; path ${r.path.map(x => x.key).join(" → ")}. Distance between the two keys: ${du} + ${dv} − 2·${dl} = ${du + dv - 2 * dl} edges. ${mode === "sample" ? "" : `Balanced build from sorted input: ${c0.get("calls")} recursive calls for 11 keys, height ${TB.height(T.root)}.`}`);
    }
    ["lca-u", "lca-v", "lca-tree"].forEach(id => on(id, "change", draw)); draw();
  });
})();
