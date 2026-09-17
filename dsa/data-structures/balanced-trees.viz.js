/* balanced-trees.viz.js — figures for dsa/data-structures/balanced-trees.html
   (Data Structures · part 6: Balanced & Multiway Trees).

   Loaded after ../../data.js, ../../notes.js and ../dsa-viz.js, which supplies the
   shared toolbox (AC palette, AL.frame / AL.row / AL.stepper / AL.counter / AL.rng …).

   Layout of this file
     BT   — the structures, pure (no DOM): AVL, red-black (nil-sentinel, parent
            pointers), B-tree with minimum degree t, red-black → 2-3-4 view, splay
            (bottom-up, with the potential Φ = Σ log₂ size), skip list (seeded coin),
            trie (array / map / compressed / ternary cost models), segment tree
            (recursive, any n, with lazy range-add), Fenwick tree, treap, and an
            order-statistic + interval-augmented AVL. Every routine takes an
            AL.counter() so that every number the page displays is a MEASUREMENT,
            and every structure has a check*() that re-verifies its invariant from
            scratch — the figures run it after every mutation they show.
     BD   — drawing: pointer binary tree (inorder x-layout, colour / label hooks),
            multiway node rows, skip-list towers, array-plus-tree twin views.
     figures — one IIFE per <svg id>, guarded by the element's existence.

   Conventions (shared with Trees & Binary Search Trees): depth(root) = 0; height in
   EDGES; height(leaf) = 0; height(∅) = −1; one comparison per node compared. */

const BT = (function () {

  /* ═══════════════════ AVL ═══════════════════════════════════════════════ */
  function anode(key) { return { key: key, left: null, right: null, h: 0 }; }
  const H = x => x ? x.h : -1;
  const bf = x => x ? H(x.left) - H(x.right) : 0;
  function upd(x) { x.h = 1 + Math.max(H(x.left), H(x.right)); if (x.size !== undefined) augUpd(x); }
  function rotR(x, c, tr) {          // right rotation at x: left child rises
    const y = x.left; x.left = y.right; if (c) c.add("write"); y.right = x; if (c) c.add("write"); upd(x); upd(y); if (c) c.add("write");   // third write: the caller stores y in x's old slot
    if (c) c.add("rot"); if (tr) tr({ ev: "rotate", dir: "right", at: x.key, up: y.key });
    return y;
  }
  function rotL(x, c, tr) {
    const y = x.right; x.right = y.left; if (c) c.add("write"); y.left = x; if (c) c.add("write"); upd(x); upd(y); if (c) c.add("write");
    if (c) c.add("rot"); if (tr) tr({ ev: "rotate", dir: "left", at: x.key, up: y.key });
    return y;
  }
  function rebalance(x, c, tr) {
    const b = bf(x);
    if (b > 1) {
      if (bf(x.left) >= 0) { if (tr) tr({ ev: "case", at: x.key, kind: "LL", bf: b, bfChild: bf(x.left) }); if (c) c.add("restructure"); return rotR(x, c, tr); }
      if (tr) tr({ ev: "case", at: x.key, kind: "LR", bf: b, bfChild: bf(x.left) }); if (c) c.add("restructure");
      x.left = rotL(x.left, c, tr); return rotR(x, c, tr);
    }
    if (b < -1) {
      if (bf(x.right) <= 0) { if (tr) tr({ ev: "case", at: x.key, kind: "RR", bf: b, bfChild: bf(x.right) }); if (c) c.add("restructure"); return rotL(x, c, tr); }
      if (tr) tr({ ev: "case", at: x.key, kind: "RL", bf: b, bfChild: bf(x.right) }); if (c) c.add("restructure");
      x.right = rotR(x.right, c, tr); return rotL(x, c, tr);
    }
    return x;
  }
  function avlInsert(x, key, c, tr) {
    if (!x) { if (c) c.add("visits"); return anode(key); }
    if (c) { c.add("visits"); c.add("cmp"); }
    if (key < x.key) x.left = avlInsert(x.left, key, c, tr);
    else if (key > x.key) x.right = avlInsert(x.right, key, c, tr);
    else return x;                                   // duplicate: no change
    upd(x);
    return rebalance(x, c, tr);
  }
  function avlMin(x) { while (x.left) x = x.left; return x; }
  function avlDelete(x, key, c, tr) {
    if (!x) return null;
    if (c) { c.add("visits"); c.add("cmp"); }
    if (key < x.key) x.left = avlDelete(x.left, key, c, tr);
    else if (key > x.key) x.right = avlDelete(x.right, key, c, tr);
    else {
      if (!x.left) return x.right;
      if (!x.right) return x.left;
      const s = avlMin(x.right);                     // successor
      if (tr) tr({ ev: "successor", at: x.key, succ: s.key });
      const nk = s.key; x.right = avlDelete(x.right, nk, c, tr); x.key = nk;
      if (x.lo !== undefined) { x.lo = s.lo; x.hi = s.hi; }
    }
    upd(x);
    return rebalance(x, c, tr);
  }
  function avlSearch(x, key, c) {
    const path = [];
    while (x) { path.push(x); if (c) { c.add("visits"); c.add("cmp"); } if (key === x.key) return { node: x, path }; x = key < x.key ? x.left : x.right; }
    return { node: null, path };
  }
  /* the invariant, re-verified from scratch: BST order, stored heights, |bf| ≤ 1 */
  function checkAVL(root) {
    let ok = true, n = 0;
    const rec = (x, lo, hi) => {
      if (!x) return -1;
      n++;
      if (!(lo < x.key && x.key < hi)) ok = false;
      const hl = rec(x.left, lo, x.key), hr = rec(x.right, x.key, hi);
      if (x.h !== 1 + Math.max(hl, hr)) ok = false;
      if (Math.abs(hl - hr) > 1) ok = false;
      return 1 + Math.max(hl, hr);
    };
    const h = rec(root, -Infinity, Infinity);
    return { ok, n, h };
  }
  function avlFromKeys(keys, c) { let r = null; for (const k of keys) r = avlInsert(r, k, c); return r; }
  /* minimum-size AVL tree of height h (the Fibonacci tree): N(h) = N(h−1) + N(h−2) + 1 */
  function fibTree(h, next) {
    next = next || { k: 1 };
    if (h < 0) return null;
    const L = fibTree(h - 1, next);
    const me = anode(next.k++);
    const R = fibTree(h - 2, next);
    me.left = L; me.right = R; upd(me); return me;
  }
  function Nmin(h) { return h < 0 ? 0 : h === 0 ? 1 : Nmin(h - 1) + Nmin(h - 2) + 1; }
  const PHI = (1 + Math.sqrt(5)) / 2;
  function avlBound(n) { return Math.log(n + 2) / Math.log(PHI) + Math.log(Math.sqrt(5)) / Math.log(PHI) - 3; }   // h (edges) < this
  function fib(k) { let a = 0, b = 1; for (let i = 0; i < k; i++) { const t = a + b; a = b; b = t; } return a; }

  /* ── plain BST insert (no balancing) for the comparison curves ─────────── */
  function bstInsertPlain(x, key, c) {
    if (!x) return anode(key);
    if (c) c.add("cmp");
    if (key < x.key) x.left = bstInsertPlain(x.left, key, c);
    else if (key > x.key) x.right = bstInsertPlain(x.right, key, c);
    return x;
  }
  function isBST(x) { let ok = true; const rec = (y, lo, hi) => { if (!y) return; if (!(lo < y.key && y.key < hi)) ok = false; rec(y.left, lo, y.key); rec(y.right, y.key, hi); }; rec(x, -Infinity, Infinity); return ok; }
  function heightOf(x) { return x ? 1 + Math.max(heightOf(x.left), heightOf(x.right)) : -1; }
  function sizeOf(x) { return x ? 1 + sizeOf(x.left) + sizeOf(x.right) : 0; }
  function inorder(x, out) { out = out || []; if (x) { inorder(x.left, out); out.push(x); inorder(x.right, out); } return out; }
  function clone(x) { if (!x) return null; const y = Object.assign({}, x); y.left = clone(x.left); y.right = clone(x.right); return y; }

  /* ── augmentation: size (order statistics) and max-high (interval tree) ── */
  function augInit(z) { z.size = 1; z.lo = z.key; z.hi = z.key; z.max = z.hi; }
  function augUpd(x) {
    x.size = 1 + (x.left ? x.left.size : 0) + (x.right ? x.right.size : 0);
    x.max = Math.max(x.hi, x.left ? x.left.max : -Infinity, x.right ? x.right.max : -Infinity);
  }
  function augInsert(x, lo, hi, c, tr) {           // key = lo (interval low endpoint)
    if (!x) { const z = anode(lo); z.lo = lo; z.hi = hi; augInit(z); z.hi = hi; z.max = hi; if (c) c.add("visits"); return z; }
    if (c) { c.add("visits"); c.add("cmp"); }
    if (lo < x.key) x.left = augInsert(x.left, lo, hi, c, tr);
    else if (lo > x.key) x.right = augInsert(x.right, lo, hi, c, tr);
    else return x;
    upd(x); return rebalance(x, c, tr);
  }
  function checkAug(root) {                          // recompute size and max from scratch
    let ok = true;
    const rec = x => {
      if (!x) return { s: 0, m: -Infinity };
      const l = rec(x.left), r = rec(x.right);
      const s = 1 + l.s + r.s, m = Math.max(x.hi, l.m, r.m);
      if (x.size !== s || x.max !== m) ok = false;
      return { s, m };
    };
    rec(root); return ok;
  }
  function select(x, i, c) { while (x) { if (c) c.add("visits"); const r = (x.left ? x.left.size : 0) + 1; if (i === r) return x; if (i < r) x = x.left; else { i -= r; x = x.right; } } return null; }
  function intervalSearch(x, lo, hi, c) {           // first overlapping interval, or null
    while (x && (x.hi < lo || hi < x.lo)) { if (c) c.add("visits"); if (x.left && x.left.max >= lo) x = x.left; else x = x.right; }
    if (x && c) c.add("visits"); return x;
  }

  /* ═══════════════════ red-black (nil sentinel, parent pointers) ═════════ */
  function RB() {
    const nil = { key: null, color: "B", left: null, right: null, p: null, nil: true };
    nil.left = nil.right = nil.p = nil;
    return { nil, root: nil, n: 0 };
  }
  function rbNode(T, key) { return { key, color: "R", left: T.nil, right: T.nil, p: T.nil }; }
  function rbLeft(T, x, c, tr) {
    const y = x.right; x.right = y.left; if (y.left !== T.nil) y.left.p = x;
    y.p = x.p; if (x.p === T.nil) T.root = y; else if (x === x.p.left) x.p.left = y; else x.p.right = y;
    y.left = x; x.p = y; if (c) c.add("rot"); if (tr) tr({ ev: "rotate", dir: "left", at: x.key, up: y.key });
  }
  function rbRight(T, x, c, tr) {
    const y = x.left; x.left = y.right; if (y.right !== T.nil) y.right.p = x;
    y.p = x.p; if (x.p === T.nil) T.root = y; else if (x === x.p.right) x.p.right = y; else x.p.left = y;
    y.right = x; x.p = y; if (c) c.add("rot"); if (tr) tr({ ev: "rotate", dir: "right", at: x.key, up: y.key });
  }
  function rbInsert(T, key, c, tr) {
    let y = T.nil, x = T.root;
    while (x !== T.nil) { if (c) { c.add("visits"); c.add("cmp"); } y = x; if (key === x.key) return null; x = key < x.key ? x.left : x.right; }
    let z = rbNode(T, key); z.p = y;
    if (y === T.nil) T.root = z; else if (key < y.key) y.left = z; else y.right = z;
    T.n++; if (tr) tr({ ev: "attach", key });
    // fixup
    while (z.p.color === "R") {
      if (z.p === z.p.p.left) {
        const u = z.p.p.right;
        if (u.color === "R") { z.p.color = "B"; u.color = "B"; z.p.p.color = "R"; if (c) { c.add("recolor", 3); c.add("case1"); } if (tr) tr({ ev: "case", n: 1, z: z.key, g: z.p.p.key }); z = z.p.p; }
        else {
          if (z === z.p.right) { z = z.p; if (c) c.add("case2"); if (tr) tr({ ev: "case", n: 2, z: z.key }); rbLeft(T, z, c, tr); }
          z.p.color = "B"; z.p.p.color = "R"; if (c) { c.add("recolor", 2); c.add("case3"); } if (tr) tr({ ev: "case", n: 3, z: z.key, g: z.p.p.key }); rbRight(T, z.p.p, c, tr);
        }
      } else {
        const u = z.p.p.left;
        if (u.color === "R") { z.p.color = "B"; u.color = "B"; z.p.p.color = "R"; if (c) { c.add("recolor", 3); c.add("case1"); } if (tr) tr({ ev: "case", n: 1, z: z.key, g: z.p.p.key }); z = z.p.p; }
        else {
          if (z === z.p.left) { z = z.p; if (c) c.add("case2"); if (tr) tr({ ev: "case", n: 2, z: z.key }); rbRight(T, z, c, tr); }
          z.p.color = "B"; z.p.p.color = "R"; if (c) { c.add("recolor", 2); c.add("case3"); } if (tr) tr({ ev: "case", n: 3, z: z.key, g: z.p.p.key }); rbLeft(T, z.p.p, c, tr);
        }
      }
    }
    if (T.root.color === "R") { T.root.color = "B"; if (c) c.add("recolor"); if (tr) tr({ ev: "rootblack" }); }
    return z;
  }
  function rbTransplant(T, u, v) { if (u.p === T.nil) T.root = v; else if (u === u.p.left) u.p.left = v; else u.p.right = v; v.p = u.p; }
  function rbMin(T, x) { while (x.left !== T.nil) x = x.left; return x; }
  function rbFind(T, key, c) { let x = T.root; while (x !== T.nil) { if (c) { c.add("visits"); c.add("cmp"); } if (key === x.key) return x; x = key < x.key ? x.left : x.right; } return null; }
  function rbDelete(T, key, c, tr) {
    const z = rbFind(T, key, c); if (!z) return false;
    let y = z, yOrig = y.color, x;
    if (z.left === T.nil) { x = z.right; rbTransplant(T, z, z.right); }
    else if (z.right === T.nil) { x = z.left; rbTransplant(T, z, z.left); }
    else {
      y = rbMin(T, z.right); yOrig = y.color; x = y.right;
      if (y.p === z) x.p = y; else { rbTransplant(T, y, y.right); y.right = z.right; y.right.p = y; }
      rbTransplant(T, z, y); y.left = z.left; y.left.p = y; y.color = z.color;
      if (tr) tr({ ev: "successor", at: z.key, succ: y.key });
    }
    T.n--;
    if (yOrig === "B") {
      while (x !== T.root && x.color === "B") {
        if (x === x.p.left) {
          let w = x.p.right;
          if (w.color === "R") { w.color = "B"; x.p.color = "R"; if (c) { c.add("recolor", 2); c.add("dcase1"); } if (tr) tr({ ev: "dcase", n: 1, x: x.key, w: w.key }); rbLeft(T, x.p, c, tr); w = x.p.right; }
          if (w.left.color === "B" && w.right.color === "B") { w.color = "R"; if (c) { c.add("recolor"); c.add("dcase2"); } if (tr) tr({ ev: "dcase", n: 2, x: x.key, w: w.key }); x = x.p; }
          else {
            if (w.right.color === "B") { w.left.color = "B"; w.color = "R"; if (c) { c.add("recolor", 2); c.add("dcase3"); } if (tr) tr({ ev: "dcase", n: 3, x: x.key, w: w.key }); rbRight(T, w, c, tr); w = x.p.right; }
            w.color = x.p.color; x.p.color = "B"; w.right.color = "B"; if (c) { c.add("recolor", 3); c.add("dcase4"); } if (tr) tr({ ev: "dcase", n: 4, x: x.key, w: w.key }); rbLeft(T, x.p, c, tr); x = T.root;
          }
        } else {
          let w = x.p.left;
          if (w.color === "R") { w.color = "B"; x.p.color = "R"; if (c) { c.add("recolor", 2); c.add("dcase1"); } if (tr) tr({ ev: "dcase", n: 1, x: x.key, w: w.key }); rbRight(T, x.p, c, tr); w = x.p.left; }
          if (w.right.color === "B" && w.left.color === "B") { w.color = "R"; if (c) { c.add("recolor"); c.add("dcase2"); } if (tr) tr({ ev: "dcase", n: 2, x: x.key, w: w.key }); x = x.p; }
          else {
            if (w.left.color === "B") { w.right.color = "B"; w.color = "R"; if (c) { c.add("recolor", 2); c.add("dcase3"); } if (tr) tr({ ev: "dcase", n: 3, x: x.key, w: w.key }); rbLeft(T, w, c, tr); w = x.p.left; }
            w.color = x.p.color; x.p.color = "B"; w.left.color = "B"; if (c) { c.add("recolor", 3); c.add("dcase4"); } if (tr) tr({ ev: "dcase", n: 4, x: x.key, w: w.key }); rbRight(T, x.p, c, tr); x = T.root;
          }
        }
      }
      if (x.color === "R" && c) c.add("recolor");
      x.color = "B";
    }
    return true;
  }
  /* the five properties, re-verified from scratch; returns bh (blacks below root incl. nil) */
  function checkRB(T) {
    let ok = true, n = 0;
    if (T.root.color !== "B") ok = false;                                   // property 2
    if (T.nil.color !== "B") ok = false;                                    // property 3
    const rec = (x, lo, hi) => {
      if (x === T.nil) return 1;                                            // the nil leaf counts as one black
      n++;
      if (!(lo < x.key && x.key < hi)) ok = false;
      if (x.color === "R" && (x.left.color === "R" || x.right.color === "R")) ok = false;   // property 4
      const bl = rec(x.left, lo, x.key), br = rec(x.right, x.key, hi);
      if (bl !== br) ok = false;                                            // property 5
      return bl + (x.color === "B" ? 1 : 0);
    };
    const bhRoot = rec(T.root, -Infinity, Infinity);
    const bh = bhRoot - (T.root !== T.nil && T.root.color === "B" ? 1 : 0);  // blacks BELOW the root, nil included
    return { ok, n, bh, h: rbHeight(T), hNil: rbHeight(T) + 1 };
  }
  function rbHeight(T) { const rec = x => x === T.nil ? -1 : 1 + Math.max(rec(x.left), rec(x.right)); return rec(T.root); }
  function rbSize(T) { const rec = x => x === T.nil ? 0 : 1 + rec(x.left) + rec(x.right); return rec(T.root); }
  function rbToPlain(T) {                           // a nil-free copy for drawing
    const rec = (x) => x === T.nil ? null : { key: x.key, color: x.color, left: rec(x.left), right: rec(x.right) };
    return rec(T.root);
  }
  function rbFromKeys(keys, c) { const T = RB(); for (const k of keys) rbInsert(T, k, c); return T; }
  /* red-black → 2-3-4 view: each black node absorbs its red children into one multiway node */
  function rbTo234(T) {
    const rec = (x) => {
      if (x === T.nil) return null;
      // x is black (or a red root, impossible); gather keys from red children
      const keys = [], kids = [];
      const take = (y) => {
        if (y === T.nil) { kids.push(null); return; }
        if (y.color === "R") { take(y.left); keys.push(y.key); take(y.right); }
        else kids.push(rec(y));
      };
      take(x.left); keys.push(x.key); take(x.right);
      const node = { keys, children: kids.map(k => k), leaf: kids.every(k => k === null) };
      if (node.leaf) node.children = [];
      return node;
    };
    return rec(T.root);
  }

  /* ═══════════════════ B-tree, minimum degree t ═════════════════════════ */
  function bnode(leaf) { return { keys: [], children: [], leaf: leaf }; }
  function BTree(t) { return { t: t, root: bnode(true), n: 0 }; }
  function btSearch(T, k, c) {
    let x = T.root, path = [];
    for (;;) {
      if (c) c.add("reads"); path.push(x);
      let i = 0; while (i < x.keys.length && k > x.keys[i]) { i++; if (c) c.add("cmp"); }
      if (i < x.keys.length) { if (c) c.add("cmp"); if (k === x.keys[i]) return { node: x, i, path, found: true }; }
      if (x.leaf) return { node: null, i, path, found: false };
      x = x.children[i];
    }
  }
  function btSplitChild(T, x, i, c, tr) {           // x non-full, x.children[i] full
    const t = T.t, y = x.children[i], z = bnode(y.leaf);
    const mid = y.keys[t - 1];
    z.keys = y.keys.splice(t, t - 1); y.keys.length = t - 1;   // y keeps t−1, z gets t−1, mid rises
    if (!y.leaf) { z.children = y.children.splice(t, t); }
    x.children.splice(i + 1, 0, z); x.keys.splice(i, 0, mid);
    if (c) c.add("splits"); if (tr) tr({ ev: "split", mid, into: x.keys.slice() });
  }
  function btInsert(T, k, c, tr) {
    const t = T.t;
    if (btSearch(T, k).found) return false;
    let r = T.root;
    if (r.keys.length === 2 * t - 1) {
      const s = bnode(false); s.children.push(r); T.root = s; if (c) c.add("reads");
      if (tr) tr({ ev: "rootsplit" });
      btSplitChild(T, s, 0, c, tr);
    }
    let x = T.root;
    for (;;) {
      if (c) c.add("reads");
      let i = x.keys.length - 1;
      if (x.leaf) {
        while (i >= 0 && k < x.keys[i]) { i--; if (c) c.add("cmp"); } if (c) c.add("cmp");
        x.keys.splice(i + 1, 0, k); T.n++; if (tr) tr({ ev: "place", key: k, into: x.keys.slice() }); return true;
      }
      while (i >= 0 && k < x.keys[i]) { i--; if (c) c.add("cmp"); } if (c) c.add("cmp");
      i++;
      if (x.children[i].keys.length === 2 * t - 1) {
        btSplitChild(T, x, i, c, tr);
        if (k > x.keys[i]) i++; if (c) c.add("cmp");
      }
      x = x.children[i];
    }
  }
  function btMax(x, c) { while (!x.leaf) { x = x.children[x.children.length - 1]; if (c) c.add("reads"); } return x.keys[x.keys.length - 1]; }
  function btMin(x, c) { while (!x.leaf) { x = x.children[0]; if (c) c.add("reads"); } return x.keys[0]; }
  function btMerge(T, x, i, c, tr) {                 // merge children i and i+1 around key i
    const y = x.children[i], z = x.children[i + 1];
    y.keys.push(x.keys[i]); y.keys = y.keys.concat(z.keys); y.children = y.children.concat(z.children);
    x.keys.splice(i, 1); x.children.splice(i + 1, 1);
    if (c) c.add("merges"); if (tr) tr({ ev: "merge", keys: y.keys.slice() });
    return y;
  }
  function btDeleteRec(T, x, k, c, tr) {
    const t = T.t; if (c) c.add("reads");
    let i = 0; while (i < x.keys.length && k > x.keys[i]) { i++; if (c) c.add("cmp"); }
    if (i < x.keys.length && k === x.keys[i]) {
      if (x.leaf) { x.keys.splice(i, 1); if (tr) tr({ ev: "case", n: "1", key: k }); return; }
      const y = x.children[i], z = x.children[i + 1];
      if (y.keys.length >= t) { const p = btMax(y, c); x.keys[i] = p; if (tr) tr({ ev: "case", n: "2a", key: k, by: p }); btDeleteRec(T, y, p, c, tr); }
      else if (z.keys.length >= t) { const s = btMin(z, c); x.keys[i] = s; if (tr) tr({ ev: "case", n: "2b", key: k, by: s }); btDeleteRec(T, z, s, c, tr); }
      else { if (tr) tr({ ev: "case", n: "2c", key: k }); const m = btMerge(T, x, i, c, tr); btDeleteRec(T, m, k, c, tr); }
      return;
    }
    if (x.leaf) return;                              // not present
    let ch = x.children[i];
    if (ch.keys.length === t - 1) {
      const L = i > 0 ? x.children[i - 1] : null, R = i < x.keys.length ? x.children[i + 1] : null;
      if (L && L.keys.length >= t) {                 // 3a: borrow from the left sibling through x
        ch.keys.unshift(x.keys[i - 1]); x.keys[i - 1] = L.keys.pop(); if (!L.leaf) ch.children.unshift(L.children.pop());
        if (c) c.add("transfers"); if (tr) tr({ ev: "case", n: "3a", key: k, dir: "from left" });
      } else if (R && R.keys.length >= t) {
        ch.keys.push(x.keys[i]); x.keys[i] = R.keys.shift(); if (!R.leaf) ch.children.push(R.children.shift());
        if (c) c.add("transfers"); if (tr) tr({ ev: "case", n: "3a", key: k, dir: "from right" });
      } else {                                       // 3b: merge with a sibling
        if (tr) tr({ ev: "case", n: "3b", key: k });
        if (R) ch = btMerge(T, x, i, c, tr); else ch = btMerge(T, x, i - 1, c, tr);
      }
    }
    btDeleteRec(T, ch, k, c, tr);
  }
  function btDelete(T, k, c, tr) {
    if (!btSearch(T, k).found) return false;
    btDeleteRec(T, T.root, k, c, tr); T.n--;
    if (T.root.keys.length === 0 && !T.root.leaf) { T.root = T.root.children[0]; if (tr) tr({ ev: "shrink" }); }
    return true;
  }
  function btHeight(T) { let h = 0, x = T.root; while (!x.leaf) { x = x.children[0]; h++; } return h; }
  function btNodes(x, out) { out = out || []; out.push(x); for (const ch of x.children) btNodes(ch, out); return out; }
  function btKeys(x, out) { out = out || []; for (let i = 0; i < x.keys.length; i++) { if (!x.leaf) btKeys(x.children[i], out); out.push(x.keys[i]); } if (!x.leaf) btKeys(x.children[x.keys.length], out); return out; }
  function checkBT(T) {                              // the definition, re-verified
    const t = T.t; let ok = true, leafDepth = -1, count = 0;
    const rec = (x, d, lo, hi, isRoot) => {
      count += x.keys.length;
      for (let i = 1; i < x.keys.length; i++) if (!(x.keys[i - 1] < x.keys[i])) ok = false;
      if (x.keys.length && !(lo < x.keys[0] && x.keys[x.keys.length - 1] < hi)) ok = false;
      if (!isRoot && x.keys.length < t - 1) ok = false;
      if (x.keys.length > 2 * t - 1) ok = false;
      if (isRoot && x.keys.length < 1 && (T.n > 0)) ok = false;
      if (x.leaf) { if (x.children.length) ok = false; if (leafDepth < 0) leafDepth = d; else if (leafDepth !== d) ok = false; return; }
      if (x.children.length !== x.keys.length + 1) ok = false;
      for (let i = 0; i <= x.keys.length; i++) rec(x.children[i], d + 1, i === 0 ? lo : x.keys[i - 1], i === x.keys.length ? hi : x.keys[i], false);
    };
    rec(T.root, 0, -Infinity, Infinity, true);
    if (count !== T.n) ok = false;
    const keys = btKeys(T.root); for (let i = 1; i < keys.length; i++) if (keys[i - 1] >= keys[i]) ok = false;
    return { ok, n: count, h: btHeight(T), nodes: btNodes(T.root).length };
  }
  function btClone(x) { return { keys: x.keys.slice(), leaf: x.leaf, children: x.children.map(btClone) }; }
  function btFromKeys(t, keys, c) { const T = BTree(t); for (const k of keys) btInsert(T, k, c); return T; }
  function btBound(n, t) { return Math.log((n + 1) / 2) / Math.log(t); }

  /* ═══════════════════ splay (bottom-up, parent pointers) ═══════════════ */
  function snode(key) { return { key, left: null, right: null, p: null }; }
  function SP() { return { root: null }; }
  function spSize(x) { return x ? 1 + spSize(x.left) + spSize(x.right) : 0; }
  function spPhi(x) { if (!x) return 0; return Math.log2(spSize(x)) + spPhi(x.left) + spPhi(x.right); }   // Φ = Σ log₂ size
  function spRotL(T, x, c) { const y = x.right; x.right = y.left; if (y.left) y.left.p = x; y.p = x.p; if (!x.p) T.root = y; else if (x === x.p.left) x.p.left = y; else x.p.right = y; y.left = x; x.p = y; if (c) c.add("rot"); }
  function spRotR(T, x, c) { const y = x.left; x.left = y.right; if (y.right) y.right.p = x; y.p = x.p; if (!x.p) T.root = y; else if (x === x.p.left) x.p.left = y; else x.p.right = y; y.right = x; x.p = y; if (c) c.add("rot"); }
  function splay(T, x, c, tr) {
    while (x.p) {
      const p = x.p, g = p.p;
      if (!g) { if (tr) tr({ ev: "zig", x: x.key, p: p.key }); if (x === p.left) spRotR(T, p, c); else spRotL(T, p, c); }
      else if ((x === p.left) === (p === g.left)) {   // zig-zig: rotate g first, then p
        if (tr) tr({ ev: "zig-zig", x: x.key, p: p.key, g: g.key });
        if (x === p.left) { spRotR(T, g, c); spRotR(T, p, c); } else { spRotL(T, g, c); spRotL(T, p, c); }
      } else {                                         // zig-zag: rotate p, then g
        if (tr) tr({ ev: "zig-zag", x: x.key, p: p.key, g: g.key });
        if (x === p.left) { spRotR(T, p, c); spRotL(T, g, c); } else { spRotL(T, p, c); spRotR(T, g, c); }
      }
      if (c) c.add("substeps");
      if (tr) tr({ ev: "after" });
    }
  }
  function spSearch(T, key, c, tr) {
    let x = T.root, last = null;
    while (x) { if (c) { c.add("visits"); c.add("cmp"); } last = x; if (key === x.key) break; x = key < x.key ? x.left : x.right; }
    if (last) splay(T, last, c, tr);
    return x;
  }
  function spInsert(T, key, c, tr) {
    let y = null, x = T.root;
    while (x) { if (c) { c.add("visits"); c.add("cmp"); } y = x; if (key === x.key) { splay(T, x, c, tr); return x; } x = key < x.key ? x.left : x.right; }
    const z = snode(key); z.p = y; if (!y) T.root = z; else if (key < y.key) y.left = z; else y.right = z;
    splay(T, z, c, tr); return z;
  }
  function spDelete(T, key, c, tr) {
    const x = spSearch(T, key, c, tr); if (!x || x.key !== key) return false;
    const L = x.left, R = x.right; if (L) L.p = null; if (R) R.p = null;
    if (!L) { T.root = R; return true; }
    T.root = L; let m = L; while (m.right) { m = m.right; if (c) c.add("visits"); }
    splay(T, m, c, tr); m.right = R; if (R) R.p = m; return true;
  }
  function spInsertNoSplay(T, key) {                 // to build a deliberately bad shape
    let y = null, x = T.root; while (x) { y = x; x = key < x.key ? x.left : x.right; }
    const z = snode(key); z.p = y; if (!y) T.root = z; else if (key < y.key) y.left = z; else y.right = z; return z;
  }
  function checkSP(T) { let ok = true; const rec = (x, lo, hi, p) => { if (!x) return; if (!(lo < x.key && x.key < hi)) ok = false; if (x.p !== p) ok = false; rec(x.left, lo, x.key, x); rec(x.right, x.key, hi, x); }; rec(T.root, -Infinity, Infinity, null); return ok; }
  function spHeight(x) { return x ? 1 + Math.max(spHeight(x.left), spHeight(x.right)) : -1; }
  function spDepth(x) { let d = 0; while (x.p) { x = x.p; d++; } return d; }
  function spFind(T, key) { let x = T.root; while (x && x.key !== key) x = key < x.key ? x.left : x.right; return x; }

  /* ═══════════════════ skip list ════════════════════════════════════════ */
  function SL(p, rnd, maxLevel) {
    const head = { key: -Infinity, next: [] };
    return { head, p: p === undefined ? 0.5 : p, rnd: rnd || Math.random, maxLevel: maxLevel || 32, levels: 1, n: 0 };
  }
  function slRandomLevel(S, c) {                     // number of levels the tower occupies, ≥ 1
    let lvl = 1; while (S.rnd() < S.p && lvl < S.maxLevel) { lvl++; if (c) c.add("heads"); } if (c) c.add("flips", lvl);   // the last flip was a tail (or the cap)
    return lvl;
  }
  function slSearch(S, key, c, tr) {                 // returns the last node with key < search key at level 0, plus the update vector
    let x = S.head; const update = new Array(S.levels), path = [];
    for (let i = S.levels - 1; i >= 0; i--) {
      while (x.next[i] && x.next[i].key < key) { x = x.next[i]; if (c) { c.add("cmp"); c.add("follows"); } path.push({ level: i, key: x.key, move: "right" }); }
      if (x.next[i]) { if (c) c.add("cmp"); }        // the comparison that stopped the scan
      update[i] = x; path.push({ level: i, key: x.key, move: "down" });
      if (tr) tr({ ev: "level", i, at: x.key });
    }
    const y = x.next[0];
    return { found: y && y.key === key ? y : null, update, path, pred: x };
  }
  function slInsert(S, key, c, tr) {
    const r = slSearch(S, key, c, tr); if (r.found) return false;
    const lvl = slRandomLevel(S, c);
    if (lvl > S.levels) { for (let i = S.levels; i < lvl; i++) r.update[i] = S.head; S.levels = lvl; }
    const z = { key, next: new Array(lvl), lvl };
    for (let i = 0; i < lvl; i++) { z.next[i] = r.update[i].next[i]; r.update[i].next[i] = z; if (c) c.add("links"); }
    S.n++; if (tr) tr({ ev: "insert", key, lvl }); return true;
  }
  function slDelete(S, key, c, tr) {
    const r = slSearch(S, key, c, tr); if (!r.found) return false;
    const z = r.found;
    for (let i = 0; i < z.lvl; i++) { if (r.update[i].next[i] === z) { r.update[i].next[i] = z.next[i]; if (c) c.add("links"); } }
    while (S.levels > 1 && !S.head.next[S.levels - 1]) S.levels--;
    S.n--; return true;
  }
  function slNodes(S) { const out = []; let x = S.head.next[0]; while (x) { out.push(x); x = x.next[0]; } return out; }
  function checkSL(S) {                              // every level a sorted subsequence of level 0
    let ok = true; const base = slNodes(S).map(x => x.key);
    for (let i = 1; i < base.length; i++) if (!(base[i - 1] < base[i])) ok = false;
    for (let lv = 0; lv < S.levels; lv++) { let x = S.head, prev = -Infinity; while (x.next[lv]) { x = x.next[lv]; if (!(x.key > prev)) ok = false; prev = x.key; if (x.lvl <= lv) ok = false; if (base.indexOf(x.key) < 0) ok = false; } }
    if (base.length !== S.n) ok = false;
    return { ok, n: S.n, levels: S.levels, pointers: slNodes(S).reduce((s, x) => s + x.lvl, 0) };
  }

  /* ═══════════════════ trie ════════════════════════════════════════════ */
  function Trie() { return { root: { ch: {}, end: false, cnt: 0 }, words: 0 }; }
  function trieInsert(T, w, c) {
    let x = T.root; x.cnt++;
    for (const ch of w) { if (c) c.add("steps"); if (!x.ch[ch]) { x.ch[ch] = { ch: {}, end: false, cnt: 0 }; if (c) c.add("newnodes"); } x = x.ch[ch]; x.cnt++; }
    if (!x.end) { x.end = true; T.words++; }
  }
  function trieSearch(T, w, c) { let x = T.root; const path = [x]; for (const ch of w) { if (c) c.add("steps"); x = x.ch[ch]; if (!x) return { found: false, path, prefixNode: null }; path.push(x); } return { found: x.end, path, prefixNode: x }; }
  function trieCollect(x, prefix, out, c) { out = out || []; if (c) c.add("collect"); if (x.end) out.push(prefix); for (const k of Object.keys(x.ch).sort()) trieCollect(x.ch[k], prefix + k, out, c); return out; }
  function trieNodes(x, out) { out = out || []; out.push(x); for (const k of Object.keys(x.ch)) trieNodes(x.ch[k], out); return out; }
  /* compressed (radix) trie: collapse every unary non-terminal chain into one edge */
  function trieCompress(x, label) {
    const keys = Object.keys(x.ch);
    if (keys.length === 1 && !x.end && label !== undefined) { const k = keys[0]; return trieCompress(x.ch[k], label + k); }
    const node = { label: label === undefined ? "" : label, end: x.end, ch: [] };
    for (const k of keys.sort()) node.ch.push(trieCompress(x.ch[k], k));
    return node;
  }
  function radixNodes(x, out) { out = out || []; out.push(x); for (const ch of x.ch) radixNodes(ch, out); return out; }
  function tstCount(T) {                             // ternary search trie: one node per (node, child-char) pair of the trie
    return trieNodes(T.root).reduce((s, x) => s + Object.keys(x.ch).length, 0);
  }

  /* ═══════════════════ segment tree (recursive over [lo, hi], any n) ═══ */
  function Seg(a, op, id) {
    const n = a.length, S = { n, a: a.slice(), t: new Array(4 * Math.max(1, n)).fill(id), lazy: new Array(4 * Math.max(1, n)).fill(0), op: op || ((x, y) => x + y), id: id === undefined ? 0 : id, nodes: 0 };
    const build = (v, lo, hi) => { S.nodes++; if (lo === hi) { S.t[v] = a[lo]; return; } const m = (lo + hi) >> 1; build(2 * v, lo, m); build(2 * v + 1, m + 1, hi); S.t[v] = S.op(S.t[2 * v], S.t[2 * v + 1]); };
    if (n) build(1, 0, n - 1);
    return S;
  }
  function segQuery(S, l, r, c, tr) {
    const rec = (v, lo, hi) => {
      if (c) c.add("visits");
      if (l <= lo && hi <= r) { if (tr) tr({ v, lo, hi, kind: "canonical", val: S.t[v] }); if (c) c.add("canonical"); return S.t[v]; }
      if (tr) tr({ v, lo, hi, kind: "partial" });
      const m = (lo + hi) >> 1; let res = S.id;
      if (l <= m) res = S.op(res, rec(2 * v, lo, m));
      if (r > m) res = S.op(res, rec(2 * v + 1, m + 1, hi));
      return res;
    };
    return rec(1, 0, S.n - 1);
  }
  function segUpdate(S, i, val, c, tr) {              // point assignment
    const rec = (v, lo, hi) => { if (c) c.add("visits"); if (tr) tr({ v, lo, hi }); if (lo === hi) { S.t[v] = val; S.a[i] = val; return; } const m = (lo + hi) >> 1; if (i <= m) rec(2 * v, lo, m); else rec(2 * v + 1, m + 1, hi); S.t[v] = S.op(S.t[2 * v], S.t[2 * v + 1]); };
    rec(1, 0, S.n - 1);
  }
  /* lazy range add, range sum (op = +) */
  function segPush(S, v, lo, hi, c) {
    if (S.lazy[v]) { const m = (lo + hi) >> 1; for (const [w, a, b] of [[2 * v, lo, m], [2 * v + 1, m + 1, hi]]) { S.t[w] += S.lazy[v] * (b - a + 1); S.lazy[w] += S.lazy[v]; } S.lazy[v] = 0; if (c) c.add("pushes"); }
  }
  function segRangeAdd(S, l, r, d, c, tr) {
    const rec = (v, lo, hi) => {
      if (c) c.add("visits");
      if (r < lo || hi < l) return;
      if (l <= lo && hi <= r) { S.t[v] += d * (hi - lo + 1); S.lazy[v] += d; if (tr) tr({ v, lo, hi, kind: "tagged" }); return; }
      if (tr) tr({ v, lo, hi, kind: "partial" });
      segPush(S, v, lo, hi, c); const m = (lo + hi) >> 1; rec(2 * v, lo, m); rec(2 * v + 1, m + 1, hi); S.t[v] = S.t[2 * v] + S.t[2 * v + 1];
    };
    rec(1, 0, S.n - 1); for (let i = l; i <= r; i++) S.a[i] += d;
  }
  function segLazyQuery(S, l, r, c, tr) {
    const rec = (v, lo, hi) => {
      if (c) c.add("visits");
      if (r < lo || hi < l) return 0;
      if (l <= lo && hi <= r) { if (tr) tr({ v, lo, hi, kind: "canonical", val: S.t[v] }); return S.t[v]; }
      if (tr) tr({ v, lo, hi, kind: "partial" });
      segPush(S, v, lo, hi, c); const m = (lo + hi) >> 1; return rec(2 * v, lo, m) + rec(2 * v + 1, m + 1, hi);
    };
    return rec(1, 0, S.n - 1);
  }
  function segNodes(S) { const out = []; const rec = (v, lo, hi, d) => { out.push({ v, lo, hi, d, val: S.t[v], lazy: S.lazy[v] }); if (lo === hi) return; const m = (lo + hi) >> 1; rec(2 * v, lo, m, d + 1); rec(2 * v + 1, m + 1, hi, d + 1); }; if (S.n) rec(1, 0, S.n - 1, 0); return out; }
  function checkSeg(S) {                             // every internal value = op(children); every leaf = a[i] (after pushing all tags)
    let ok = true;
    const rec = (v, lo, hi, add) => {
      const val = S.t[v] + add * (hi - lo + 1);
      if (lo === hi) { if (val !== S.a[lo]) ok = false; return val; }
      const m = (lo + hi) >> 1; const l = rec(2 * v, lo, m, add + S.lazy[v]), r = rec(2 * v + 1, m + 1, hi, add + S.lazy[v]);
      if (S.op(l, r) !== val && S.op === undefined) ok = false;
      if (Math.abs(S.op(l, r) - val) > 1e-9) ok = false;
      return val;
    };
    if (S.n) rec(1, 0, S.n - 1, 0); return ok;
  }
  function plainRange(a, l, r, op, id) { let s = id === undefined ? 0 : id; for (let i = l; i <= r; i++) s = (op || ((x, y) => x + y))(s, a[i]); return s; }

  /* ═══════════════════ Fenwick (binary indexed) tree, 1-based ══════════ */
  const lowbit = i => i & (-i);
  function Fen(a) {                                  // a is 0-based input; tree is 1-based, tree[0] unused
    const n = a.length, t = new Array(n + 1).fill(0);
    for (let i = 1; i <= n; i++) t[i] = a[i - 1];
    for (let i = 1; i <= n; i++) { const j = i + lowbit(i); if (j <= n) t[j] += t[i]; }   // O(n) build
    return { n, t, a: a.slice() };
  }
  function fenPrefix(F, i, c, tr) { let s = 0; while (i > 0) { if (c) c.add("visits"); s += F.t[i]; if (tr) tr({ i, add: F.t[i], next: i - lowbit(i) }); i -= lowbit(i); } return s; }
  function fenUpdate(F, i, d, c, tr) { F.a[i - 1] += d; while (i <= F.n) { if (c) c.add("visits"); F.t[i] += d; if (tr) tr({ i, next: i + lowbit(i) }); i += lowbit(i); } }
  function fenRange(F, l, r, c) { return fenPrefix(F, r, c) - fenPrefix(F, l - 1, c); }
  function checkFen(F) { let ok = true; for (let i = 1; i <= F.n; i++) { let s = 0; for (let j = i - lowbit(i) + 1; j <= i; j++) s += F.a[j - 1]; if (s !== F.t[i]) ok = false; } return ok; }

  /* ═══════════════════ treap ══════════════════════════════════════════ */
  function tnode(key, pr) { return { key, pr, left: null, right: null }; }
  function trRotR(x, c) { const y = x.left; x.left = y.right; y.right = x; if (c) c.add("rot"); return y; }
  function trRotL(x, c) { const y = x.right; x.right = y.left; y.left = x; if (c) c.add("rot"); return y; }
  function treapInsert(x, key, pr, c) {
    if (!x) return tnode(key, pr);
    if (c) c.add("cmp");
    if (key < x.key) { x.left = treapInsert(x.left, key, pr, c); if (x.left.pr < x.pr) x = trRotR(x, c); }
    else if (key > x.key) { x.right = treapInsert(x.right, key, pr, c); if (x.right.pr < x.pr) x = trRotL(x, c); }
    return x;
  }
  function treapDelete(x, key, c) {
    if (!x) return null; if (c) c.add("cmp");
    if (key < x.key) x.left = treapDelete(x.left, key, c);
    else if (key > x.key) x.right = treapDelete(x.right, key, c);
    else {
      if (!x.left) return x.right; if (!x.right) return x.left;
      if (x.left.pr < x.right.pr) { x = trRotR(x, c); x.right = treapDelete(x.right, key, c); }
      else { x = trRotL(x, c); x.left = treapDelete(x.left, key, c); }
    }
    return x;
  }
  function checkTreap(root) { let ok = true; const rec = (x, lo, hi) => { if (!x) return; if (!(lo < x.key && x.key < hi)) ok = false; if (x.left && x.left.pr < x.pr) ok = false; if (x.right && x.right.pr < x.pr) ok = false; rec(x.left, lo, x.key); rec(x.right, x.key, hi); }; rec(root, -Infinity, Infinity); return ok; }

  /* ═══════════════════ shared helpers ════════════════════════════════════ */
  function randomBSTHeight(n, r) { let root = null; for (const k of AL.perm(n, r)) root = bstInsertPlain(root, k); return heightOf(root); }
  const log2 = x => Math.log2(x);

  return {
    // avl
    anode, H, bf, upd, rotR, rotL, avlInsert, avlDelete, avlSearch, checkAVL, avlFromKeys, fibTree, Nmin, PHI, avlBound, fib,
    bstInsertPlain, isBST, heightOf, sizeOf, inorder, clone, augInsert, checkAug, select, intervalSearch, augInit, augUpd,
    // red-black
    RB, rbInsert, rbDelete, rbFind, checkRB, rbHeight, rbSize, rbToPlain, rbFromKeys, rbTo234,
    // b-tree
    BTree, btSearch, btInsert, btDelete, btHeight, btNodes, btKeys, checkBT, btClone, btFromKeys, btBound, bnode,
    // splay
    SP, splay, spSearch, spInsert, spDelete, spInsertNoSplay, checkSP, spHeight, spDepth, spFind, spPhi, spSize, snode,
    // skip list
    SL, slSearch, slInsert, slDelete, slNodes, checkSL, slRandomLevel,
    // trie
    Trie, trieInsert, trieSearch, trieCollect, trieNodes, trieCompress, radixNodes, tstCount,
    // segment / fenwick
    Seg, segQuery, segUpdate, segRangeAdd, segLazyQuery, segNodes, checkSeg, plainRange,
    Fen, fenPrefix, fenUpdate, fenRange, checkFen, lowbit,
    // treap
    treapInsert, treapDelete, checkTreap,
    randomBSTHeight, log2
  };
})();

/* ═══════════════════════════════════════════════════════════════════════════
   BD — drawing helpers (this page's own; AL.binTree only draws level-order
   arrays, which is the wrong shape for trees that are the SUBJECT of balance)
   ═══════════════════════════════════════════════════════════════════════════ */
const BD = (function () {
  if (typeof d3 === "undefined") return null;

  /* pointer binary tree, inorder x-layout: the i-th node in inorder sits in column i,
     depth gives y. Works for any shape and needs no parent pointers. */
  function layoutBin(root, o) {
    const list = [], by = new Map();
    const rec = (x, d) => { if (!x) return; rec(x.left, d + 1); list.push(x); by.set(x, { n: x, i: list.length - 1, d }); rec(x.right, d + 1); };
    rec(root, 0);
    const n = list.length, unit = n ? o.w / n : o.w;
    for (const p of by.values()) { p.x = o.x + (p.i + 0.5) * unit; p.y = o.y + p.d * o.levelH + o.r; }
    return { list, by, unit };
  }
  function drawBin(g, root, opt) {
    const o = Object.assign({ x: 0, y: 0, w: 600, levelH: 54, r: 14, fontSize: 12, fill: null, stroke: null, edge: null, edgeW: null,
      label: null, text: null, textFill: null, nullSlots: false, dash: null, labelAbove: null, sw: null }, opt || {});
    const L = layoutBin(root, o), gg = g.append("g");
    L.list.forEach(nd => {
      const p = L.by.get(nd);
      [nd.left, nd.right].forEach((ch, side) => {
        if (!ch) {
          if (o.nullSlots) {
            const nx = p.x + (side ? 1 : -1) * Math.max(8, L.unit * 0.35), ny = p.y + o.levelH * 0.55;
            gg.append("line").attr("x1", p.x).attr("y1", p.y).attr("x2", nx).attr("y2", ny).attr("stroke", AC.line).attr("stroke-dasharray", "2,3");
            gg.append("rect").attr("x", nx - 4).attr("y", ny - 4).attr("width", 8).attr("height", 8).attr("fill", "#111").attr("stroke", AC.line);
          }
          return;
        }
        const q = L.by.get(ch);
        const ln = gg.append("line").attr("x1", p.x).attr("y1", p.y).attr("x2", q.x).attr("y2", q.y)
          .attr("stroke", o.edge ? (o.edge(ch, nd) || AC.line) : AC.line).attr("stroke-width", o.edgeW ? (o.edgeW(ch, nd) || 1.5) : 1.5);
        if (o.dash && o.dash(ch, nd)) ln.attr("stroke-dasharray", o.dash(ch, nd));
      });
    });
    L.list.forEach(nd => {
      const p = L.by.get(nd);
      gg.append("circle").attr("cx", p.x).attr("cy", p.y).attr("r", o.r)
        .attr("fill", o.fill ? (o.fill(nd) || AC.panel2) : AC.panel2)
        .attr("stroke", o.stroke ? (o.stroke(nd) || AC.line) : AC.line).attr("stroke-width", o.sw ? (o.sw(nd) || 1.5) : 1.5);
      gg.append("text").attr("x", p.x).attr("y", p.y + 4).attr("text-anchor", "middle").attr("font-size", o.fontSize)
        .attr("fill", o.textFill ? (o.textFill(nd) || AC.ink) : AC.ink).text(o.text ? o.text(nd) : nd.key);
      if (o.label) { const t = o.label(nd); if (t !== null && t !== undefined && t !== "") gg.append("text").attr("x", p.x).attr("y", p.y + o.r + 11).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", AC.muted).text(t); }
      if (o.labelAbove) { const t = o.labelAbove(nd); if (t) gg.append("text").attr("x", p.x).attr("y", p.y - o.r - 4).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", AC.a2).text(t); }
    });
    return { g: gg, pos: L.by, list: L.list, unit: L.unit };
  }

  /* multiway tree: each node a row of key cells; x from leaf order, a parent centred over its children */
  function drawMulti(g, root, opt) {
    const o = Object.assign({ x: 0, y: 0, w: 700, levelH: 70, cw: 26, ch: 24, gap: 10, fontSize: 11, fill: null, keyFill: null, stroke: null, label: null }, opt || {});
    const leaves = []; const rec0 = x => { if (!x.children.length) leaves.push(x); else x.children.forEach(rec0); }; rec0(root);
    const by = new Map();
    // width needed per leaf = its cell count; distribute proportionally
    const leafW = leaves.map(l => Math.max(1, l.keys.length) * o.cw + o.gap);
    const total = leafW.reduce((a, b) => a + b, 0), scale = total > o.w ? o.w / total : 1;
    let cursor = o.x + (o.w - total * scale) / 2;
    const rec = (x, d) => {
      let cx;
      if (!x.children.length) { const wdt = Math.max(1, x.keys.length) * o.cw; cx = cursor + (leafW[leaves.indexOf(x)] * scale) / 2; cursor += leafW[leaves.indexOf(x)] * scale; }
      else { const xs = x.children.map(c => rec(c, d + 1)); cx = (Math.min(...xs) + Math.max(...xs)) / 2; }
      by.set(x, { n: x, d, x: cx, y: o.y + d * o.levelH, w: Math.max(1, x.keys.length) * o.cw * Math.min(1, scale * 1.15) });
      return cx;
    };
    rec(root, 0);
    const gg = g.append("g");
    for (const [x, p] of by) x.children.forEach((c, i) => {
      const q = by.get(c); const cellW = p.w / Math.max(1, x.keys.length);
      const sx = p.x - p.w / 2 + i * cellW;              // the i-th child pointer sits between cells i−1 and i
      gg.append("line").attr("x1", sx).attr("y1", p.y + o.ch).attr("x2", q.x).attr("y2", q.y).attr("stroke", AC.line).attr("stroke-width", 1.3);
    });
    for (const [x, p] of by) {
      const cellW = p.w / Math.max(1, x.keys.length);
      const f = o.fill ? o.fill(x) : null;
      if (!x.keys.length) gg.append("rect").attr("x", p.x - p.w / 2).attr("y", p.y).attr("width", p.w).attr("height", o.ch).attr("rx", 4).attr("fill", f || AC.panel2).attr("stroke", o.stroke ? (o.stroke(x) || AC.line) : AC.line).attr("stroke-dasharray", "3,3");
      x.keys.forEach((k, i) => {
        gg.append("rect").attr("x", p.x - p.w / 2 + i * cellW).attr("y", p.y).attr("width", cellW).attr("height", o.ch).attr("rx", i === 0 || i === x.keys.length - 1 ? 4 : 0)
          .attr("fill", (o.keyFill && o.keyFill(x, i, k)) || f || AC.panel2).attr("stroke", o.stroke ? (o.stroke(x) || AC.line) : AC.line).attr("stroke-width", 1.3);
        gg.append("text").attr("x", p.x - p.w / 2 + (i + 0.5) * cellW).attr("y", p.y + o.ch / 2 + 4).attr("text-anchor", "middle").attr("font-size", o.fontSize).attr("fill", AC.ink).text(k);
      });
      if (o.label) { const t = o.label(x); if (t) gg.append("text").attr("x", p.x).attr("y", p.y + o.ch + 11).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", AC.muted).text(t); }
    }
    return { g: gg, pos: by };
  }

  /* skip list as towers: column per key (plus head), one row per level, level 0 at the bottom */
  function drawSkip(g, S, opt) {
    const o = Object.assign({ x: 0, y: 0, w: 700, rowH: 26, cw: 30, mark: null, arrow: null, showKeysOnTop: true }, opt || {});
    const nodes = BT.slNodes(S), cols = [S.head].concat(nodes), L = S.levels;
    const unit = Math.min(o.cw + 12, o.w / (cols.length + 1));
    const gg = g.append("g").attr("transform", `translate(${o.x},${o.y})`);
    const cx = j => (j + 0.5) * unit, cy = lv => (L - 1 - lv) * o.rowH;
    // horizontal links per level
    for (let lv = 0; lv < L; lv++) {
      let x = S.head, j = 0;
      while (x.next[lv]) { const nj = cols.indexOf(x.next[lv]); gg.append("line").attr("x1", cx(j) + unit * 0.3).attr("y1", cy(lv) + o.rowH / 2).attr("x2", cx(nj) - unit * 0.3).attr("y2", cy(lv) + o.rowH / 2).attr("stroke", o.arrow ? (o.arrow(lv, x, x.next[lv]) || AC.line) : AC.line).attr("stroke-width", o.arrow && o.arrow(lv, x, x.next[lv]) ? 2.5 : 1.2); x = x.next[lv]; j = nj; }
      gg.append("text").attr("x", -4).attr("y", cy(lv) + o.rowH / 2 + 4).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.muted).text(`L${lv}`);
    }
    cols.forEach((nd, j) => {
      const lv = nd === S.head ? L : nd.lvl;
      for (let l = 0; l < lv; l++) {
        const m = o.mark ? o.mark(l, nd) : null;
        gg.append("rect").attr("x", cx(j) - unit * 0.3).attr("y", cy(l) + 3).attr("width", unit * 0.6).attr("height", o.rowH - 6).attr("rx", 3).attr("fill", m || (nd === S.head ? AC.panel : AC.panel2)).attr("stroke", m ? m : AC.line);
        gg.append("text").attr("x", cx(j)).attr("y", cy(l) + o.rowH / 2 + 4).attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", m ? AC.bg : AC.ink).text(nd === S.head ? "−∞" : nd.key);
      }
    });
    return { g: gg, cx, cy, unit, cols };
  }

  /* an array row plus a binary tree of intervals above it (segment tree) */
  function drawSeg(g, S, opt) {
    const o = Object.assign({ x: 0, y: 0, w: 700, levelH: 46, cw: 30, fill: null, text: null, stroke: null }, opt || {});
    const nodes = BT.segNodes(S), depth = Math.max(...nodes.map(n => n.d));
    const unit = Math.min(o.cw + 8, o.w / S.n);
    const gg = g.append("g").attr("transform", `translate(${o.x},${o.y})`);
    const pos = n => ({ x: ((n.lo + n.hi + 1) / 2) * unit, y: n.d * o.levelH + 12 });
    const byV = new Map(nodes.map(n => [n.v, n]));
    nodes.forEach(n => { if (n.v > 1) { const p = pos(byV.get(n.v >> 1)), q = pos(n); gg.append("line").attr("x1", p.x).attr("y1", p.y + 9).attr("x2", q.x).attr("y2", q.y - 9).attr("stroke", AC.line); } });
    nodes.forEach(n => {
      const p = pos(n), wdt = Math.max(unit * 0.9, (n.hi - n.lo + 1) * unit * 0.9);
      gg.append("rect").attr("x", p.x - wdt / 2).attr("y", p.y - 10).attr("width", wdt).attr("height", 20).attr("rx", 4).attr("fill", o.fill ? (o.fill(n) || AC.panel2) : AC.panel2).attr("stroke", o.stroke ? (o.stroke(n) || AC.line) : AC.line);
      gg.append("text").attr("x", p.x).attr("y", p.y + 4).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.ink).text(o.text ? o.text(n) : n.val);
    });
    const ay = (depth + 1) * o.levelH + 10;
    AL.row(gg, S.a, { x: unit * 0.05, y: ay, w: unit * 0.9, gap: unit * 0.1, h: 24, index: true });
    return { g: gg, pos, unit, arrayY: ay };
  }

  function note(g, x, y, txt, color, size) { return g.append("text").attr("x", x).attr("y", y).attr("font-size", size || 11.5).attr("fill", color || AC.ink).text(txt); }
  function lines(g, x, y, arr, opt) {
    const o = Object.assign({ lh: 17, size: 11, hi: [] }, opt || {});
    arr.forEach((t, i) => g.append("text").attr("x", x).attr("y", y + i * o.lh).attr("font-size", o.size).attr("fill", (o.hi.indexOf(i) >= 0 || (o.hiFn && o.hiFn(t, i))) ? AC.ink : AC.muted).text(t));
  }
  return { layoutBin, drawBin, drawMulti, drawSkip, drawSeg, note, lines };
})();

/* ═══════════════════════════════════════════════════════════════════════════
   figures
   ═══════════════════════════════════════════════════════════════════════════ */
(function () {
  if (typeof document === "undefined" || typeof d3 === "undefined") return;
  const $ = id => document.getElementById(id);
  const fig = (id, fn) => { if ($(id)) fn(); };
  const txt = (id, s) => { const el = $(id); if (el) el.textContent = s; };
  const on = (id, ev, fn) => { const el = $(id); if (el) el.addEventListener(ev, fn); };
  const val = id => { const el = $(id); return el ? el.value : null; };
  const f1 = x => (+x).toFixed(1), f2 = x => (+x).toFixed(2), f3 = x => (+x).toFixed(3);
  const ok = b => b ? "✓" : "✗";

  /* ── F01 heights of the families against n ────────────────────────────── */
  fig("fam-svg", () => {
    const NS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
    const cache = {};
    function measure(order) {
      if (cache[order]) return cache[order];
      const rows = NS.map(n => {
        const keys = order === "sorted" ? Array.from({ length: n }, (_, i) => i + 1) : AL.perm(n, AL.rng(1000 + n));
        let bst = null; for (const k of keys) bst = BT.bstInsertPlain(bst, k);
        const avl = BT.avlFromKeys(keys), rb = BT.rbFromKeys(keys), b2 = BT.btFromKeys(2, keys);
        const sl = BT.SL(0.5, AL.rng(77 + n)); for (const k of keys) BT.slInsert(sl, k);
        const r2 = AL.rng(5 + n); let tp = null; for (const k of keys) tp = BT.treapInsert(tp, k, r2());
        const ca = BT.checkAVL(avl), cr = BT.checkRB(rb), cb = BT.checkBT(b2), cs = BT.checkSL(sl);
        return { n, bst: BT.heightOf(bst), avl: ca.h, avlOk: ca.ok, rb: cr.h, rbOk: cr.ok, b2: cb.h, b2Ok: cb.ok, skip: sl.levels, skipOk: cs.ok, treap: BT.heightOf(tp), treapOk: BT.checkTreap(tp),
                 avlB: BT.avlBound(n), rbB: 2 * Math.log2(n + 1), b2B: BT.btBound(n, 2) };
      });
      cache[order] = rows; return rows;
    }
    function draw() {
      const order = val("fam-order") || "sorted", rows = measure(order);
      const F = AL.frame("#fam-svg", 760, 340, { l: 50, r: 250, t: 16, b: 36 });
      const x = d3.scaleLog().domain([16, 4096]).range([0, F.iw]);
      const y = d3.scaleLog().domain([1, 5000]).range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 4); AL.axisB(F.g, x, F.ih, 5, "n (keys)", d3.format("~s")); AL.axisL(F.g, y, 4, "height (edges)", d3.format("~s"));
      const series = [
        { k: "bst", label: order === "sorted" ? "plain BST, sorted input (= n − 1)" : "plain BST, random order", color: AC.bad },
        { k: "treap", label: "treap (random-BST shape)", color: AC.rose },
        { k: "skip", label: "skip list, levels", color: AC.violet },
        { k: "rb", label: "red-black", color: AC.a2 },
        { k: "avl", label: "AVL", color: AC.accent },
        { k: "b2", label: "2-3-4 tree (B-tree, t = 2)", color: AC.good }
      ];
      const line = k => d3.line().x(d => x(d.n)).y(d => y(Math.max(1, d[k])));
      series.forEach(s => { F.g.append("path").datum(rows).attr("d", line(s.k)).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", 2); F.g.selectAll(null).data(rows).join("circle").attr("cx", d => x(d.n)).attr("cy", d => y(Math.max(1, d[s.k]))).attr("r", 2.5).attr("fill", s.color); });
      F.g.append("path").datum(rows).attr("d", d3.line().x(d => x(d.n)).y(d => y(Math.floor(Math.log2(d.n))))).attr("fill", "none").attr("stroke", AC.muted).attr("stroke-dasharray", "4,3");
      AL.legend(F.g, series.map(s => ({ label: s.label, color: s.color })).concat([{ label: "floor(log₂ n), the perfect tree", color: AC.muted, dash: "4,3" }]), 6, 14, { gap: 14 });
      const L = rows[rows.length - 1];
      BD.lines(F.g, F.iw + 14, 8, [
        `at n = ${L.n}, ${order} order:`,
        `plain BST          ${L.bst}`,
        `AVL                ${L.avl}   bound < ${f2(L.avlB)}   ${ok(L.avlOk)}`,
        `red-black          ${L.rb}   bound ≤ ${f2(L.rbB)} (nil)  ${ok(L.rbOk)}`,
        `2-3-4              ${L.b2}   bound ≤ ${f2(L.b2B)}   ${ok(L.b2Ok)}`,
        `skip list levels   ${L.skip}   ${ok(L.skipOk)}`,
        `treap              ${L.treap}   ${ok(L.treapOk)}`,
        `floor(log₂ n)      ${Math.floor(Math.log2(L.n))}`,
        ``,
        `✓ = invariant re-verified from`,
        `scratch on the finished tree`,
        `(order, heights, colours, fan-out,`,
        `levels, heap order)`
      ], { hi: [0, 1, 2, 3, 4, 5, 6], size: 10.5, lh: 16 });
      txt("fam-readout", `${order} insertion, n = ${L.n}: plain BST height ${L.bst}; AVL ${L.avl} (bound ${f2(L.avlB)}); red-black ${L.rb} (bound ${f2(L.rbB)} to the nil leaves, ${L.rb + 1} measured that way); 2-3-4 tree ${L.b2} (bound ${f2(L.b2B)}); skip list ${L.skip} levels; treap ${L.treap}; perfect tree ${Math.floor(Math.log2(L.n))}. Invariants re-checked on every structure at every n: ${rows.every(r => r.avlOk && r.rbOk && r.b2Ok && r.skipOk && r.treapOk) ? "all pass" : "FAILURE"}.`);
    }
    on("fam-order", "change", draw); draw();
  });

  /* ── F02 a rotation and its inverse, stepped ──────────────────────────── */
  fig("rot-svg", () => {
    const mk = () => { const n = k => BT.anode(k); const a = n(20); a.left = n(10); const t = n(50); t.left = n(30); t.left.left = a; t.left.right = n(40); t.right = n(60); t.right.right = n(70); return t; };
    const frames = [];
    let T = mk(); frames.push({ title: "before: root 50, left child 30 — α = {10, 20}, β = {40}, γ = {60, 70}", root: BT.clone(T), rising: 30, writes: 0 });
    const c = AL.counter(); T = BT.rotR(T, c);
    frames.push({ title: "right rotation at 50: 30 rises, 50 sinks, β = {40} changes parent from 30 to 50", root: BT.clone(T), rising: 30, writes: c.get("write") });
    T = BT.rotL(T, c);
    frames.push({ title: "left rotation at 30: the inverse — the original tree is back exactly", root: BT.clone(T), rising: 50, writes: c.get("write") });
    const grp = k => (k <= 20 ? "α" : k === 40 ? "β" : k >= 60 ? "γ" : null);
    function render(f, i) {
      const F = AL.frame("#rot-svg", 760, 330, { l: 10, r: 10, t: 24, b: 10 });
      F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 12).attr("fill", AC.a2).text(`step ${i + 1}: ${f.title}`);
      const depthOf = new Map(); (function rec(x, d) { if (!x) return; depthOf.set(x.key, d); rec(x.left, d + 1); rec(x.right, d + 1); })(f.root, 0);
      BD.drawBin(F.g, f.root, { x: 0, y: 4, w: 470, levelH: 60, r: 16, fontSize: 12,
        fill: nd => grp(nd.key) === "α" ? "#1f3a4a" : grp(nd.key) === "β" ? "#3b2a4a" : grp(nd.key) === "γ" ? "#2a3b2a" : (nd.key === f.rising ? AC.a2 : null),
        stroke: nd => grp(nd.key) === "α" ? AC.teal : grp(nd.key) === "β" ? AC.violet : grp(nd.key) === "γ" ? AC.good : AC.a2,
        textFill: nd => nd.key === f.rising ? AC.bg : null, label: nd => grp(nd.key) ? `${grp(nd.key)} · d=${depthOf.get(nd.key)}` : `d=${depthOf.get(nd.key)}` });
      const ino = BT.inorder(f.root).map(x => x.key).join(" ");
      const dA = depthOf.get(20), dB = depthOf.get(40), dG = depthOf.get(60);
      BD.lines(F.g, 490, 14, [`inorder: ${ino}`, ``, `depth of subtree roots:`, `  α (root 20): ${dA}`, `  β (root 40): ${dB}`, `  γ (root 60): ${dG}`, ``, `child-pointer writes so far: ${f.writes}`, `(3 per rotation; +3 parent`, ` pointers if nodes carry them)`, ``, `BST order re-checked: ${ok(BT.isBST(f.root))}`], { hi: [0, 3, 4, 5, 7] });
      txt("rot-readout", `Step ${i + 1}: ${f.title}. Inorder ${ino} — unchanged. Subtree-root depths: α ${dA}, β ${dB}, γ ${dG}. Child-pointer writes so far: ${f.writes} (measured: 3 per rotation).`);
    }
    AL.stepper(d3.select("#rot-svg"), { frames, render, delay: 1200, label: "step" });
  });

  /* helpers shared by the AVL figures: balance factors computed from the DRAWN tree, not from stored fields */
  const realH = x => x ? 1 + Math.max(realH(x.left), realH(x.right)) : -1;
  const realBf = x => realH(x.left) - realH(x.right);
  const findParent = (root, key) => { let p = null, x = root; while (x && x.key !== key) { p = x; x = key < x.key ? x.left : x.right; } return { p, x }; };
  const rotateAt = (root, key, dir, c) => {           // rotate at the node with this key, on a tree without parent pointers
    const { p, x } = findParent(root, key); if (!x) return root;
    const y = dir === "right" ? BT.rotR(x, c) : BT.rotL(x, c);
    if (!p) return y; if (p.left === x) p.left = y; else p.right = y; return root;
  };
  const fixHeights = x => { if (!x) return -1; x.h = 1 + Math.max(fixHeights(x.left), fixHeights(x.right)); return x.h; };

  /* ── F03 the minimum AVL trees of heights 0..4 ───────────────────────── */
  fig("fibt-svg", () => {
    const F = AL.frame("#fibt-svg", 760, 330, { l: 8, r: 8, t: 20, b: 8 });
    const widths = [40, 60, 90, 150, 240], gap = 22; let x0 = 0; const rows = [];
    for (let h = 0; h <= 4; h++) {
      const t = BT.fibTree(h), ck = BT.checkAVL(t), n = ck.n;
      BD.drawBin(F.g, t, { x: x0, y: 0, w: widths[h], levelH: 40, r: 11, fontSize: 10, label: nd => `${realBf(nd) > 0 ? "+" : ""}${realBf(nd)}`,
        fill: nd => realBf(nd) === 0 ? null : "#2a3b2a", stroke: nd => realBf(nd) === 0 ? null : AC.good });
      F.g.append("text").attr("x", x0 + widths[h] / 2).attr("y", -6).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.a2).text(`h = ${h}: N = ${n}`);
      rows.push({ h, n, F: BT.fib(h + 3) - 1, hm: ck.h, bound: BT.avlBound(n), ok: ck.ok, bfs: BT.inorder(t).map(realBf) });
      x0 += widths[h] + gap;
    }
    const ty = 232;
    const cols = ["h", "nodes (counted)", "F(h+3) − 1", "height (walked)", "bound at that n", "all |bf| ≤ 1", "leaf depths"];
    const leafDepths = t => { const out = []; (function rec(x, d) { if (!x) return; if (!x.left && !x.right) out.push(d); rec(x.left, d + 1); rec(x.right, d + 1); })(t, 0); return [...new Set(out)].sort((a, b) => a - b).join(","); };
    const cx = [0, 60, 200, 300, 420, 540, 640];
    cols.forEach((c, i) => F.g.append("text").attr("x", cx[i]).attr("y", ty).attr("font-size", 10.5).attr("fill", AC.muted).text(c));
    rows.forEach((r, j) => { const t = BT.fibTree(r.h); const vals = [r.h, r.n, r.F, r.hm, f2(r.bound), ok(r.ok), leafDepths(t)]; vals.forEach((v, i) => F.g.append("text").attr("x", cx[i]).attr("y", ty + 17 * (j + 1)).attr("font-size", 11).attr("fill", AC.ink).text(v)); });
    txt("fibt-readout", `Minimum AVL trees: sizes ${rows.map(r => r.n).join(", ")} for heights 0–4, each equal to F(h+3) − 1 = ${rows.map(r => r.F).join(", ")}; the bound 1.44042·log₂(n+2) − 1.32772 evaluates to ${rows.map(r => f2(r.bound)).join(", ")}, above each measured height by less than one; invariant checks: ${rows.every(r => r.ok) ? "all pass" : "FAIL"}. Balance factors of the height-4 tree in inorder: ${rows[4].bfs.join(" ")}.`);
  });

  /* ── F04 AVL insertion: the four cases ────────────────────────────────── */
  fig("avlins-svg", () => {
    const SEQ = [60, 50, 10, 20, 30, 40, 80, 70];
    const frames = []; let root = null; const c = AL.counter();
    SEQ.forEach(k => {
      const ev = []; const before = BT.clone(root);
      // attach without rebalancing, on a clone, to show the imbalance
      let attached = BT.clone(root); attached = (function ins(x) { if (!x) return BT.anode(k); if (k < x.key) x.left = ins(x.left); else x.right = ins(x.right); return x; })(attached); fixHeights(attached);
      const bad = BT.inorder(attached).find(x => Math.abs(realBf(x)) === 2 && !BT.inorder(x.left).concat(BT.inorder(x.right)).some(y => Math.abs(realBf(y)) === 2));
      root = BT.avlInsert(root, k, c, e => ev.push(e));
      const cs = ev.find(e => e.ev === "case");
      const chk = BT.checkAVL(root);
      if (!cs) frames.push({ tree: BT.clone(root), title: `insert ${k}: attached as a leaf; every balance factor stays within ±1 — no rotation`, k, ok: chk.ok, cmp: c.get("cmp"), rot: c.get("rot"), rs: c.get("restructure") });
      else {
        frames.push({ tree: attached, title: `insert ${k}: attached — node ${bad.key} now has bf ${realBf(bad) > 0 ? "+" : ""}${realBf(bad)}, its ${cs.kind[0] === "L" ? "left" : "right"} child bf ${cs.bfChild > 0 ? "+" : ""}${cs.bfChild}: case ${cs.kind}`, k, bad: bad.key, ok: null, cmp: c.get("cmp"), rot: c.get("rot") - ev.filter(e => e.ev === "rotate").length, rs: c.get("restructure") - 1, pivot: cs.kind === "LL" ? bad.left.key : cs.kind === "RR" ? bad.right.key : cs.kind === "LR" ? bad.left.right.key : bad.right.left.key });
        if (cs.kind === "LR" || cs.kind === "RL") {
          const mid = BT.clone(attached); const inner = cs.kind === "LR" ? bad.left.key : bad.right.key;
          rotateAt(mid, inner, cs.kind === "LR" ? "left" : "right"); fixHeights(mid);
          frames.push({ tree: mid, title: `case ${cs.kind}, first rotation: ${cs.kind === "LR" ? "left" : "right"} rotation at ${inner} straightens the zig-zag (node ${bad.key} still at bf ${realBf(bad) > 0 ? "+" : ""}${realBf(bad)})`, k, bad: bad.key, ok: null, cmp: c.get("cmp"), rot: c.get("rot") - 1, rs: c.get("restructure") - 1, pivot: cs.kind === "LR" ? bad.left.right.key : bad.right.left.key });
        }
        const rots = ev.filter(e => e.ev === "rotate").map(e => `${e.dir} at ${e.at}`).join(", then ");
        frames.push({ tree: BT.clone(root), title: `case ${cs.kind} done: ${rots} — ${ev.filter(e => e.ev === "rotate").pop().up} rises; height back to what it was before the insertion`, k, ok: chk.ok, cmp: c.get("cmp"), rot: c.get("rot"), rs: c.get("restructure"), pivot: ev.filter(e => e.ev === "rotate").pop().up });
      }
    });
    function render(f, i) {
      const F = AL.frame("#avlins-svg", 760, 340, { l: 10, r: 10, t: 24, b: 10 });
      F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 12).attr("fill", AC.a2).text(`step ${i + 1}: ${f.title}`);
      BD.drawBin(F.g, f.tree, { x: 0, y: 6, w: 500, levelH: 62, r: 16, fontSize: 12,
        fill: nd => nd.key === f.bad && Math.abs(realBf(nd)) === 2 ? AC.bad : nd.key === f.pivot ? AC.a2 : nd.key === f.k ? "#1f3a4a" : null,
        stroke: nd => nd.key === f.k ? AC.teal : null, textFill: nd => (nd.key === f.bad && Math.abs(realBf(nd)) === 2) || nd.key === f.pivot ? AC.bg : null,
        label: nd => `bf ${realBf(nd) > 0 ? "+" : ""}${realBf(nd)}` });
      const h = realH(f.tree), n = BT.sizeOf(f.tree);
      BD.lines(F.g, 520, 14, [`keys so far: ${SEQ.slice(0, SEQ.indexOf(f.k) + 1).join(" ")}`, `n = ${n}, height ${h}`, `bound: 1.4404·log₂(${n}+2) − 1.328 = ${f2(BT.avlBound(n))}`, ``, `comparisons so far   ${f.cmp}`, `rotations so far     ${f.rot}`, `restructurings       ${f.rs}`, ``, f.ok === null ? `(mid-repair: invariant broken at ${f.bad})` : `invariant re-checked: ${ok(f.ok)}`, ``, `LL / RR: one rotation`, `LR / RL: two rotations (double)`], { hi: [1, 4, 5, 6, 8] });
      txt("avlins-readout", `Step ${i + 1}: ${f.title}. n = ${n}, height ${h}; comparisons ${f.cmp}, rotations ${f.rot}, restructurings ${f.rs} so far. ${f.ok === null ? "Invariant temporarily broken at " + f.bad + "." : "Invariant re-verified: " + ok(f.ok) + "."}`);
    }
    AL.stepper(d3.select("#avlins-svg"), { frames, render, delay: 1400, label: "step" });
  });

  /* ── F05 AVL deletion cascade on the height-6 Fibonacci tree ─────────── */
  fig("avldel-svg", () => {
    const base = BT.fibTree(6), n = BT.sizeOf(base);
    // find the key whose deletion needs the most restructurings (measured)
    let cascadeKey = 1, most = -1;
    for (let k = 1; k <= n; k++) { const x = BT.avlSearch(base, k).node; if (x.left || x.right) continue; const c = AL.counter(); BT.avlDelete(BT.clone(base), k, c); if (c.get("restructure") > most) { most = c.get("restructure"); cascadeKey = k; } }   // leaves only: the stepped walk below removes a leaf
    // the same measurement on Fibonacci trees of other heights (worst leaf deletion), for the text's table
    const cascadeByH = [4, 6, 8, 10, 12].map(h => { const t = BT.fibTree(h), m = BT.sizeOf(t); let best = 0; for (let k = 1; k <= m; k++) { const x = BT.avlSearch(t, k).node; if (x.left || x.right) continue; const c = AL.counter(); BT.avlDelete(BT.clone(t), k, c); best = Math.max(best, c.get("restructure")); } return `h = ${h} (n = ${m}): ${best}`; });
    function build(mode) {
      const key = mode === "cascade" ? cascadeKey : 1;
      const frames = []; let t = BT.clone(base);
      frames.push({ tree: BT.clone(t), title: `the minimum AVL tree of height 6: n = ${n}, keys 1 … ${n}; about to delete ${key}`, key });
      // step 1: remove the leaf (both chosen keys are leaves)
      const { p, x } = findParent(t, key);
      if (x && !x.left && !x.right) { if (!p) t = null; else if (p.left === x) p.left = null; else p.right = null; }
      fixHeights(t);
      frames.push({ tree: BT.clone(t), title: `leaf ${key} removed; balance factors recomputed up its path`, key, changed: true });
      // walk up the path from the removed leaf's parent, rotating where |bf| = 2, exactly as rebalance would
      const path = []; { let y = t; while (y && y.key !== key) { path.push(y.key); y = key < y.key ? y.left : y.right; } }
      const c = AL.counter();
      for (let i = path.length - 1; i >= 0; i--) {
        const nd = findParent(t, path[i]).x; if (!nd) continue;
        const b = realBf(nd);
        if (Math.abs(b) === 2) {
          const child = b > 0 ? nd.left : nd.right, cb = realBf(child);
          const kind = b > 0 ? (cb >= 0 ? "LL" : "LR") : (cb <= 0 ? "RR" : "RL");
          const hBefore = realH(nd);
          frames.push({ tree: BT.clone(t), title: `node ${nd.key} has bf ${b > 0 ? "+" : ""}${b}, child ${child.key} bf ${cb > 0 ? "+" : ""}${cb}: case ${kind}`, key, bad: nd.key });
          if (kind === "LR") rotateAt(t, child.key, "left", c); if (kind === "RL") rotateAt(t, child.key, "right", c);
          t = rotateAt(t, nd.key, b > 0 ? "right" : "left", c); fixHeights(t);
          const top = findParent(t, path[i - 1] === undefined ? t.key : path[i - 1]).x; // parent after rotation
          const newSub = i === 0 ? t : (key < findParent(t, path[i - 1]).x.key ? findParent(t, path[i - 1]).x.left : findParent(t, path[i - 1]).x.right);
          frames.push({ tree: BT.clone(t), title: `rotated: subtree height ${hBefore} → ${realH(newSub)} — ${realH(newSub) < hBefore ? "one shorter, so the parent must be re-examined" : "unchanged, the walk stops"}`, key, pivot: newSub.key, rots: c.get("rot") });
        }
      }
      // cross-check against the recursive routine on a fresh copy
      const c2 = AL.counter(); const ref = BT.avlDelete(BT.clone(base), key, c2);
      const same = (a, b) => (!a && !b) || (a && b && a.key === b.key && same(a.left, b.left) && same(a.right, b.right));
      const chk = BT.checkAVL(t);
      frames.push({ tree: BT.clone(t), title: `done: ${c.get("rot")} rotation(s), height ${realH(t)}; identical to the recursive delete's result (${c2.get("restructure")} restructurings there): ${ok(same(t, ref))}; invariant ${ok(chk.ok)}`, key, final: true, rots: c.get("rot"), h: realH(t), same: same(t, ref), ok: chk.ok, refRs: c2.get("restructure") });
      return frames;
    }
    let st = null;
    function setup() {
      const mode = val("avldel-key") || "cascade";
      const frames = build(mode);
      d3.select("#avldel-svg").node().parentNode.querySelectorAll("[role=group]").forEach(e => e.remove());
      st = AL.stepper(d3.select("#avldel-svg"), { frames, render, delay: 1400, label: "step" });
    }
    function render(f, i) {
      const F = AL.frame("#avldel-svg", 760, 360, { l: 6, r: 6, t: 24, b: 6 });
      F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11.5).attr("fill", AC.a2).text(`step ${i + 1}: ${f.title}`);
      BD.drawBin(F.g, f.tree, { x: 0, y: 4, w: 748, levelH: 46, r: 10, fontSize: 9.5,
        fill: nd => nd.key === f.bad ? AC.bad : nd.key === f.pivot ? AC.a2 : null, textFill: nd => nd.key === f.bad || nd.key === f.pivot ? AC.bg : null,
        label: nd => `${realBf(nd) > 0 ? "+" : ""}${realBf(nd)}` });
      txt("avldel-readout", `Step ${i + 1}: ${f.title}.${f.final ? ` Deleting ${f.key} from the 33-node Fibonacci tree of height 6: ${f.rots} rotations by the stepped walk, ${f.refRs} restructurings by the recursive routine, final height ${f.h}, trees identical ${ok(f.same)}, invariant ${ok(f.ok)}.` : ""} Worst single leaf deletion on Fibonacci trees, restructurings measured over every leaf: ${cascadeByH.join("; ")}.`);
    }
    on("avldel-key", "change", setup); setup();
  });

  /* ── F06 heights vs n: AVL, red-black, plain BST, bound, perfect ─────── */
  fig("avlh-svg", () => {
    const NS = [64, 128, 256, 512, 1024, 2048, 4096], T = 10;
    const rows = NS.map(n => {
      let sa = 0, sb = 0, sr = 0, mn = 1e9, mx = 0, allOk = true;
      for (let t = 0; t < T; t++) {
        const keys = AL.perm(n, AL.rng(100 * n + t));
        const a = BT.avlFromKeys(keys), ca = BT.checkAVL(a); sa += ca.h; mn = Math.min(mn, ca.h); mx = Math.max(mx, ca.h); allOk = allOk && ca.ok;
        let b = null; for (const k of keys) b = BT.bstInsertPlain(b, k); sb += BT.heightOf(b);
        const rb = BT.rbFromKeys(keys), cr = BT.checkRB(rb); sr += cr.h; allOk = allOk && cr.ok;
      }
      return { n, avl: sa / T, mn, mx, bst: sb / T, rb: sr / T, bound: BT.avlBound(n), perfect: Math.floor(Math.log2(n)), ok: allOk };
    });
    const F = AL.frame("#avlh-svg", 760, 320, { l: 46, r: 230, t: 14, b: 36 });
    const x = d3.scaleLog().domain([64, 4096]).range([0, F.iw]), y = d3.scaleLinear().domain([0, 30]).range([F.ih, 0]);
    AL.gridY(F.g, y, F.iw, 6); AL.axisB(F.g, x, F.ih, 7, "n (keys, random order)", d3.format("~s")); AL.axisL(F.g, y, 6, "height (edges)");
    F.g.append("path").datum(rows).attr("d", d3.area().x(d => x(d.n)).y0(d => y(d.mn)).y1(d => y(d.mx))).attr("fill", AC.accent).attr("opacity", 0.18);
    const L = (k, color, dash) => { const p = F.g.append("path").datum(rows).attr("d", d3.line().x(d => x(d.n)).y(d => y(d[k]))).attr("fill", "none").attr("stroke", color).attr("stroke-width", 2); if (dash) p.attr("stroke-dasharray", dash); };
    L("bst", AC.bad); L("rb", AC.a2); L("avl", AC.accent); L("bound", AC.violet, "2,3"); L("perfect", AC.muted, "4,3");
    AL.legend(F.g, [{ label: "plain BST (mean of 10)", color: AC.bad }, { label: "red-black (mean)", color: AC.a2 }, { label: "AVL (mean; band = min–max)", color: AC.accent }, { label: "AVL bound 1.4404·log₂(n+2) − 1.328", color: AC.violet, dash: "2,3" }, { label: "floor(log₂ n)", color: AC.muted, dash: "4,3" }], 8, 14, { gap: 14 });
    const Lr = rows[rows.length - 1];
    BD.lines(F.g, F.iw + 12, 8, [`at n = ${Lr.n} (10 permutations):`, `plain BST     ${f1(Lr.bst)}`, `red-black     ${f1(Lr.rb)}`, `AVL           ${f1(Lr.avl)}  (${Lr.mn}–${Lr.mx})`, `AVL bound     ${f2(Lr.bound)}`, `perfect       ${Lr.perfect}`, ``, `invariants re-checked on`, `every AVL and red-black`, `tree: ${rows.every(r => r.ok) ? "all pass ✓" : "FAIL"}`], { hi: [1, 2, 3, 4, 5], size: 10.5, lh: 16 });
    txt("avlh-readout", `Random insertion order, mean over 10 seeded permutations per n. At n = 4096: plain BST ${f1(Lr.bst)}, red-black ${f1(Lr.rb)}, AVL ${f1(Lr.avl)} (range ${Lr.mn}–${Lr.mx}), bound ${f2(Lr.bound)}, perfect ${Lr.perfect}. Per-n AVL means: ${rows.map(r => `${r.n}: ${f2(r.avl)}`).join("; ")}. Red-black means: ${rows.map(r => f2(r.rb)).join(", ")}.`);
  });

  /* ── F07 rotations per operation, the distributions ───────────────────── */
  fig("avlrot-svg", () => {
    const n = 4096, r = AL.rng(42), keys = AL.perm(n, r); let root = null; const ins = [], del = [], insRs = [], delRs = []; let allOk = true;
    for (const k of keys) { const c = AL.counter(); root = BT.avlInsert(root, k, c); ins.push(c.get("rot")); insRs.push(c.get("restructure")); allOk = allOk && BT.checkAVL(root).ok; }
    for (const k of AL.shuffle(keys, r)) { const c = AL.counter(); root = BT.avlDelete(root, k, c); del.push(c.get("rot")); delRs.push(c.get("restructure")); allOk = allOk && BT.checkAVL(root).ok; }
    const hist = a => { const h = []; a.forEach(v => h[v] = (h[v] || 0) + 1); for (let i = 0; i < h.length; i++) h[i] = h[i] || 0; return h; };
    const hi = hist(ins), hd = hist(del), mean = a => a.reduce((s, v) => s + v, 0) / a.length;
    const F = AL.frame("#avlrot-svg", 760, 300, { l: 46, r: 220, t: 22, b: 34 });
    const cells = AL.cells(F.iw, F.ih, 2, 1, { l: 34, r: 8, t: 18, b: 26 }, { x: 30 });
    [["insertions", hi, AC.accent], ["deletions", hd, AC.a2]].forEach(([name, h, color], i) => {
      const cell = cells[i], g = F.g.append("g").attr("transform", `translate(${cell.x + cell.m.l},${cell.y + cell.m.t})`);
      const x = d3.scaleBand().domain(d3.range(7)).range([0, cell.iw]).padding(0.2), y = d3.scaleLinear().domain([0, n]).range([cell.ih, 0]);
      g.append("g").attr("class", "axis").attr("transform", `translate(0,${cell.ih})`).call(d3.axisBottom(x)); g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4).tickFormat(d3.format("~s")));
      g.selectAll("rect").data(d3.range(7)).join("rect").attr("x", d => x(d)).attr("y", d => y(h[d] || 0)).attr("width", x.bandwidth()).attr("height", d => cell.ih - y(h[d] || 0)).attr("fill", color);
      g.selectAll(".lab").data(d3.range(7)).join("text").attr("x", d => x(d) + x.bandwidth() / 2).attr("y", d => y(h[d] || 0) - 3).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", AC.ink).text(d => h[d] ? h[d] : "");
      g.append("text").attr("x", cell.iw / 2).attr("y", -6).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.muted).text(`${name}: rotations per operation`);
      g.append("text").attr("x", cell.iw).attr("y", cell.ih + 26).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.muted).text("rotations");
    });
    BD.lines(F.g, F.iw + 10, 4, [`${n} random inserts, then ${n} deletes`, ``, `insert: mean ${f2(mean(ins))} rot/op`, `        max ${Math.max(...ins)} rotations`, `        max ${Math.max(...insRs)} restructuring`, `        ${hi[0]} needed none`, ``, `delete: mean ${f2(mean(del))} rot/op`, `        max ${Math.max(...del)} rotations`, `        max ${Math.max(...delRs)} restructurings`, `        ${hd[0]} needed none`, ``, `invariant after every op: ${allOk ? "✓" : "✗"}`], { hi: [2, 3, 7, 8, 12], size: 10.5, lh: 16 });
    txt("avlrot-readout", `${n} random insertions: rotation counts ${hi.map((v, i) => `${i}: ${v}`).join(", ")} (mean ${f3(mean(ins))}, max ${Math.max(...ins)}, at most ${Math.max(...insRs)} restructuring each). ${n} deletions: ${hd.map((v, i) => `${i}: ${v}`).join(", ")} (mean ${f3(mean(del))}, max ${Math.max(...del)} rotations, ${Math.max(...delRs)} restructurings). Invariant re-checked after all ${2 * n} operations: ${allOk ? "all pass" : "FAIL"}.`);
  });

  /* helpers for the red-black figures */
  const rbFill = nd => nd.color === "R" ? "#7f1d1d" : "#0b0d12";
  const rbStroke = nd => nd.color === "R" ? AC.bad : "#4b5563";
  const plainFind = (root, key) => { let x = root; while (x && x.key !== key) x = key < x.key ? x.left : x.right; return x; };
  const plainNodes = root => { const out = []; (function rec(x) { if (!x) return; out.push(x); rec(x.left); rec(x.right); })(root); return out; };
  const RB_SEQ = [10, 34, 14, 30, 6, 38, 26, 18, 22];

  /* ── F08 the five properties and black-height, live ───────────────────── */
  fig("rbprop-svg", () => {
    const keys = AL.perm(15, AL.rng(1)).map(k => k * 3);
    const T = BT.rbFromKeys(keys), P = BT.rbToPlain(T), all = plainNodes(P).sort((a, b) => a.key - b.key);
    const sel = $("rbprop-node"); if (sel) { all.forEach(x => { const o = document.createElement("option"); o.value = x.key; o.textContent = `${x.key} (${x.color === "R" ? "red" : "black"})`; sel.appendChild(o); }); sel.value = P.key; }
    function draw() {
      const key = +val("rbprop-node") || P.key, x = plainFind(P, key);
      const sub = new Set(plainNodes(x));
      // every path from x to a nil, with its black count
      const paths = []; (function rec(y, acc, blacks) { if (!y) { paths.push({ keys: acc.slice(), blacks: blacks + 1 }); return; } const b = blacks + (y !== x && y.color === "B" ? 1 : 0); rec(y.left, acc.concat([y.key]), b); rec(y.right, acc.concat([y.key]), b); })(x, [], 0);
      const bh = paths[0].blacks, allEq = paths.every(p => p.blacks === bh), inner = sub.size;
      const ck = BT.checkRB(T);
      const F = AL.frame("#rbprop-svg", 760, 340, { l: 6, r: 6, t: 20, b: 6 });
      BD.drawBin(F.g, P, { x: 0, y: 4, w: 470, levelH: 56, r: 13, fontSize: 11, nullSlots: true,
        fill: rbFill, stroke: nd => nd === x ? AC.a2 : sub.has(nd) ? AC.teal : rbStroke(nd), sw: nd => nd === x ? 3 : sub.has(nd) ? 2 : 1.5,
        edge: (c, p) => sub.has(c) && sub.has(p) ? AC.teal : null, edgeW: (c, p) => sub.has(c) && sub.has(p) ? 2.5 : null });
      const rows = [`x = ${x.key} (${x.color === "R" ? "red" : "black"})`, `paths from x to a nil, black count each:`].concat(paths.slice(0, 8).map(p => `  ${p.keys.join("→")}→nil : ${p.blacks}`)).concat(paths.length > 8 ? [`  … ${paths.length} paths in all`] : []).concat([``, `bh(x) = ${bh}  (all equal: ${ok(allEq)})`, `internal nodes in subtree: ${inner} ≥ 2^${bh} − 1 = ${Math.pow(2, bh) - 1}  ${ok(inner >= Math.pow(2, bh) - 1)}`, ``, `whole tree: n = ${ck.n}, h to nil = ${ck.hNil} ≤ 2·log₂(n+1) = ${f2(2 * Math.log2(ck.n + 1))} ${ok(ck.hNil <= 2 * Math.log2(ck.n + 1))}`, `properties 1–5 re-verified: ${ok(ck.ok)}`]);
      BD.lines(F.g, 486, 10, rows, { hi: [0, rows.length - 5, rows.length - 4, rows.length - 2, rows.length - 1], size: 10.5, lh: 15.5 });
      txt("rbprop-readout", `Node ${x.key}: ${paths.length} paths to nil leaves, black counts ${paths.map(p => p.blacks).join(" ")} — all ${bh}, so bh = ${bh}; subtree holds ${inner} internal nodes ≥ 2^${bh} − 1 = ${Math.pow(2, bh) - 1}. Whole tree: n = ${ck.n}, black-height ${ck.bh}, height ${ck.h} to the deepest key and ${ck.hNil} to the nil leaves against 2·log₂(n+1) = ${f2(2 * Math.log2(ck.n + 1))}. Five properties: ${ok(ck.ok)}.`);
    }
    on("rbprop-node", "change", draw); draw();
  });

  /* ── F09 red-black insertion, stepped ─────────────────────────────────── */
  fig("rbins-svg", () => {
    const T = BT.RB(), frames = [], c = AL.counter();
    RB_SEQ.forEach(k => {
      const snap = (title, z, extra) => { const ck = BT.checkRB(T); frames.push(Object.assign({ tree: BT.rbToPlain(T), title, z, k, rot: c.get("rot"), rec: c.get("recolor"), bh: ck.bh, h: ck.h, n: ck.n, ok: ck.ok }, extra || {})); };
      BT.rbInsert(T, k, c, e => {
        if (e.ev === "attach") { const zp = BT.rbFind(T, k).p; snap(`insert ${k}: attached red${zp !== T.nil && zp.color === "R" ? " — its parent " + zp.key + " is red: property 4 violated" : zp === T.nil ? " as the root" : "; parent " + zp.key + " is black: nothing to fix"}`, k, { mid: true }); }
        else if (e.ev === "case") { const zn = BT.rbFind(T, e.z); const u = zn.p === zn.p.p.left ? zn.p.p.right : zn.p.p.left; snap(`z = ${e.z}, uncle ${u === T.nil ? "nil" : u.key} is ${u.color === "R" ? "red" : "black"}${e.n === 1 ? "" : (zn === zn.p.left) === (zn.p === zn.p.p.left) ? ", z and its parent line up" : ", z is the inner grandchild"}: case ${e.n}${e.n === 1 ? " — recolour, z jumps to " + e.g : e.n === 2 ? " — rotate at " + e.z + "'s parent to straighten" : " — recolour and rotate at grandparent " + e.g}`, e.z, { mid: true, uncle: u === T.nil ? null : u.key, caseN: e.n }); }
        else if (e.ev === "rotate") snap(`${e.dir} rotation at ${e.at}: ${e.up} rises`, e.up, { mid: true });
        else if (e.ev === "rootblack") snap(`the root is red after case 1 reached it: recolour it black (black-height grows)`, null, { mid: true });
      });
      snap(`insert ${k} complete — properties re-verified`, null);
    });
    function render(f, i) {
      const F = AL.frame("#rbins-svg", 760, 340, { l: 6, r: 6, t: 24, b: 6 });
      F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11.5).attr("fill", AC.a2).text(`step ${i + 1}: ${f.title}`);
      BD.drawBin(F.g, f.tree, { x: 0, y: 4, w: 500, levelH: 62, r: 15, fontSize: 12, fill: rbFill, stroke: nd => nd.key === f.z ? AC.a2 : nd.key === f.uncle ? AC.violet : rbStroke(nd), sw: nd => nd.key === f.z || nd.key === f.uncle ? 3 : 1.5,
        label: nd => nd.key === f.z ? "z" : nd.key === f.uncle ? "uncle" : (nd.color === "R" && ((nd.left && nd.left.color === "R") || (nd.right && nd.right.color === "R"))) ? "red-red!" : "" });
      BD.lines(F.g, 520, 14, [`inserted so far: ${RB_SEQ.slice(0, RB_SEQ.indexOf(f.k) + 1).join(" ")}`, `n = ${f.n}, height ${f.h}, black-height ${f.bh}`, `bound: 2·log₂(n+1) − 1 = ${f2(2 * Math.log2(f.n + 1) - 1)}`, ``, `rotations so far      ${f.rot}`, `recolourings so far   ${f.rec}`, ``, f.mid ? `(mid-repair)` : `five properties: ${ok(f.ok)}`, ``, `case 1: recolour, climb 2 levels`, `case 2: rotate to straighten`, `case 3: recolour + rotate, stop`], { hi: [1, 4, 5, 7] });
      txt("rbins-readout", `Step ${i + 1}: ${f.title}. n = ${f.n}, height ${f.h}, black-height ${f.bh}; ${f.rot} rotations and ${f.rec} recolourings so far. ${f.mid ? "" : "Five properties re-verified: " + ok(f.ok) + "."}`);
    }
    AL.stepper(d3.select("#rbins-svg"), { frames, render, delay: 1400, label: "step" });
  });

  /* ── F10 red-black deletion, stepped ──────────────────────────────────── */
  fig("rbdel-svg", () => {
    const T = BT.rbFromKeys(RB_SEQ), frames = [], DELS = [34, 10, 38, 26];
    const c0 = BT.checkRB(T);
    frames.push({ tree: BT.rbToPlain(T), title: `the 9-key tree of §09: n = ${c0.n}, black-height ${c0.bh}; about to delete ${DELS.join(", then ")}`, rot: 0, rec: 0, n: c0.n, bh: c0.bh, h: c0.h, ok: c0.ok, done: [] });
    let done = [];
    DELS.forEach(k => {
      const c = AL.counter();
      const snap = (title, marks, mid) => { const ck = BT.checkRB(T); frames.push({ tree: BT.rbToPlain(T), title, marks: marks || {}, rot: c.get("rot"), rec: c.get("recolor"), n: ck.n, bh: ck.bh, h: ck.h, ok: ck.ok, mid, done: done.slice(), k }); };
      const z = BT.rbFind(T, k); const zc = z.color, two = z.left !== T.nil && z.right !== T.nil;
      BT.rbDelete(T, k, c, e => {
        if (e.ev === "successor") snap(`delete ${k}: two children, so its successor ${e.succ} is spliced out and takes ${k}'s place and colour`, { w: null }, true);
        else if (e.ev === "dcase") snap(`x = ${e.x === null ? "nil" : e.x} carries the extra black; sibling w = ${e.w}: case ${e.n}${e.n === 1 ? " (w red) — recolour, rotate at the parent" : e.n === 2 ? " (w black, black nephews) — w turns red, x climbs" : e.n === 3 ? " (w's far nephew black) — recolour, rotate at w" : " (w's far nephew red) — recolour, rotate at the parent, done"}`, { w: e.w, x: e.x }, true);
        else if (e.ev === "rotate") snap(`${e.dir} rotation at ${e.at}: ${e.up} rises`, {}, true);
      });
      done.push(k);
      snap(`delete ${k} complete: ${two ? "successor spliced, " : ""}${zc === "R" && !two ? "a red node was removed — nothing to repair; " : ""}${c.get("rot")} rotation(s), ${c.get("recolor")} recolouring(s) — properties re-verified`, {}, false);
    });
    function render(f, i) {
      const F = AL.frame("#rbdel-svg", 760, 340, { l: 6, r: 6, t: 24, b: 6 });
      F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11.5).attr("fill", AC.a2).text(`step ${i + 1}: ${f.title}`);
      const m = f.marks || {};
      BD.drawBin(F.g, f.tree, { x: 0, y: 4, w: 500, levelH: 62, r: 15, fontSize: 12, nullSlots: true, fill: rbFill, stroke: nd => nd.key === m.w ? AC.violet : nd.key === m.x ? AC.a2 : rbStroke(nd), sw: nd => nd.key === m.w || nd.key === m.x ? 3 : 1.5, label: nd => nd.key === m.w ? "w" : nd.key === m.x ? "x" : "" });
      BD.lines(F.g, 520, 14, [`deleted so far: ${f.done.length ? f.done.join(" ") : "—"}`, `n = ${f.n}, height ${f.h}, black-height ${f.bh}`, ``, f.k ? `this deletion: ${f.rot} rotation(s)` : ``, f.k ? `               ${f.rec} recolouring(s)` : ``, ``, f.mid ? `(mid-repair)` : `five properties: ${ok(f.ok)}`, ``, `case 1: w red → rotate, then 2/3/4`, `case 2: climb, no rotation`, `case 3: rotate at w → case 4`, `case 4: rotate at parent, stop`, `≤ 3 rotations per deletion`], { hi: [1, 3, 4, 6] });
      txt("rbdel-readout", `Step ${i + 1}: ${f.title}. n = ${f.n}, height ${f.h}, black-height ${f.bh}. ${f.mid ? "" : "Five properties re-verified: " + ok(f.ok) + "."}`);
    }
    AL.stepper(d3.select("#rbdel-svg"), { frames, render, delay: 1400, label: "step" });
  });

  /* ── F11 twin view: red-black ↔ 2-3-4 ─────────────────────────────────── */
  fig("rb234-svg", () => {
    function draw() {
      const k = +val("rb234-k") || 9; txt("rb234-k-out", k);
      const T = BT.rbFromKeys(RB_SEQ.slice(0, k)), P = BT.rbToPlain(T), M = BT.rbTo234(T), ck = BT.checkRB(T);
      const depths = []; const cnt = { 2: 0, 3: 0, 4: 0 }; let mh = 0;
      (function rec(x, d) { if (!x) return; cnt[x.keys.length + 1]++; mh = Math.max(mh, d); if (!x.children.length) depths.push(d); x.children.forEach(ch => rec(ch, d + 1)); })(M, 0);
      const same = depths.every(d => d === depths[0]);
      const F = AL.frame("#rb234-svg", 760, 330, { l: 6, r: 6, t: 26, b: 6 });
      F.g.append("text").attr("x", 0).attr("y", -10).attr("font-size", 11.5).attr("fill", AC.a2).text(`red-black after ${k} insertion${k > 1 ? "s" : ""}: ${RB_SEQ.slice(0, k).join(" ")}`);
      F.g.append("text").attr("x", 400).attr("y", -10).attr("font-size", 11.5).attr("fill", AC.a2).text(`the same tree as a 2-3-4 tree (reds merged into their black parent)`);
      BD.drawBin(F.g, P, { x: 0, y: 4, w: 370, levelH: 58, r: 14, fontSize: 11.5, fill: rbFill, stroke: rbStroke });
      BD.drawMulti(F.g, M, { x: 400, y: 10, w: 350, levelH: 74, cw: 30, ch: 26, fill: () => AC.panel2 });
      BD.lines(F.g, 400, 200, [`black-height ${ck.bh} = 2-3-4 height ${mh} + 1  ${ok(ck.bh === mh + 1)}`, `2-nodes ${cnt[2]}, 3-nodes ${cnt[3]}, 4-nodes ${cnt[4]}`, `all 2-3-4 leaves at depth ${depths[0]}: ${ok(same)}`, `red-black properties: ${ok(ck.ok)}`], { hi: [0, 1, 2, 3], size: 10.5, lh: 16 });
      txt("rb234-readout", `After ${k} insertions (${RB_SEQ.slice(0, k).join(" ")}): red-black black-height ${ck.bh}, height ${ck.h}; 2-3-4 height ${mh} with ${cnt[2]} two-nodes, ${cnt[3]} three-nodes and ${cnt[4]} four-nodes; all multiway leaves at depth ${depths[0]} ${ok(same)}; bh = 2-3-4 height + 1 ${ok(ck.bh === mh + 1)}.`);
    }
    on("rb234-k", "input", draw); draw();
  });

  /* ── F12 AVL vs red-black on the same workload ────────────────────────── */
  fig("avlrb-svg", () => {
    const cache = {};
    function measure(order) {
      if (cache[order]) return cache[order];
      const n = 4096, r = AL.rng(42), keys = order === "sorted" ? Array.from({ length: n }, (_, i) => i + 1) : AL.perm(n, r);
      const searches = AL.sample(keys, n, r), dels = AL.shuffle(keys, r);
      const out = {};
      { let root = null; const ci = AL.counter(); for (const k of keys) root = BT.avlInsert(root, k, ci); const ok1 = BT.checkAVL(root).ok; const cs = AL.counter(); for (const k of searches) BT.avlSearch(root, k, cs); let sd = 0; (function rec(x, d) { if (!x) return; sd += d; rec(x.left, d + 1); rec(x.right, d + 1); })(root, 0); const h = BT.heightOf(root); const cd = AL.counter(); for (const k of dels) root = BT.avlDelete(root, k, cd);
        out.avl = { h, depth: sd / n, scmp: cs.get("cmp") / n, icmp: ci.get("cmp") / n, irot: ci.get("rot") / n, drot: cd.get("rot") / n, irec: 0, drec: 0, ok: ok1 && root === null }; }
      { const T = BT.RB(); const ci = AL.counter(); for (const k of keys) BT.rbInsert(T, k, ci); const ck = BT.checkRB(T); const cs = AL.counter(); for (const k of searches) BT.rbFind(T, k, cs); let sd = 0; (function rec(x, d) { if (x === T.nil) return; sd += d; rec(x.left, d + 1); rec(x.right, d + 1); })(T.root, 0); const cd = AL.counter(); for (const k of dels) BT.rbDelete(T, k, cd);
        out.rb = { h: ck.h, depth: sd / n, scmp: cs.get("cmp") / n, icmp: ci.get("cmp") / n, irot: ci.get("rot") / n, drot: cd.get("rot") / n, irec: ci.get("recolor") / n, drec: cd.get("recolor") / n, ok: ck.ok && T.root === T.nil }; }
      cache[order] = out; return out;
    }
    function draw() {
      const order = val("avlrb-order") || "random", m = measure(order);
      const F = AL.frame("#avlrb-svg", 760, 320, { l: 44, r: 236, t: 16, b: 60 });
      const measures = [["height", "h"], ["mean depth", "depth"], ["cmp / search", "scmp"], ["cmp / insert", "icmp"], ["rot / insert", "irot"], ["rot / delete", "drot"]];
      const x0 = d3.scaleBand().domain(measures.map(m => m[0])).range([0, F.iw]).padding(0.25), x1 = d3.scaleBand().domain(["avl", "rb"]).range([0, x0.bandwidth()]).padding(0.1);
      const y = d3.scaleLinear().domain([0, Math.max(m.avl.h, m.rb.h, 12) * 1.15]).range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 5); F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x0)).selectAll("text").attr("transform", "rotate(-20)").attr("text-anchor", "end"); AL.axisL(F.g, y, 5, "per operation (mean)");
      measures.forEach(([lab, key]) => ["avl", "rb"].forEach(s => { const v = m[s][key]; F.g.append("rect").attr("x", x0(lab) + x1(s)).attr("y", y(v)).attr("width", x1.bandwidth()).attr("height", F.ih - y(v)).attr("fill", s === "avl" ? AC.accent : AC.a2); F.g.append("text").attr("x", x0(lab) + x1(s) + x1.bandwidth() / 2).attr("y", y(v) - 3).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", AC.ink).text(f2(v)); }));
      AL.legend(F.g, [{ label: "AVL", color: AC.accent }, { label: "red-black", color: AC.a2 }], 6, 12);
      BD.lines(F.g, F.iw + 12, 6, [`${order} insertion, n = 4096`, `4096 searches, 4096 deletions`, ``, `AVL: height ${m.avl.h}, mean depth ${f2(m.avl.depth)}`, `  cmp/search ${f2(m.avl.scmp)}`, `  rot/insert ${f2(m.avl.irot)}, rot/delete ${f2(m.avl.drot)}`, ``, `red-black: height ${m.rb.h}, depth ${f2(m.rb.depth)}`, `  cmp/search ${f2(m.rb.scmp)}`, `  rot/insert ${f2(m.rb.irot)}, rot/delete ${f2(m.rb.drot)}`, `  recolour/insert ${f2(m.rb.irec)}`, `  recolour/delete ${f2(m.rb.drec)}`, ``, `invariants ✓ after build, and`, `both trees empty after deletes: ${ok(m.avl.ok && m.rb.ok)}`], { hi: [3, 4, 5, 7, 8, 9, 10, 11], size: 10.5, lh: 15.5 });
      txt("avlrb-readout", `${order} insertion of 4096 keys, then 4096 searches and 4096 deletions. AVL: height ${m.avl.h}, mean depth ${f2(m.avl.depth)}, ${f2(m.avl.scmp)} comparisons per search, ${f2(m.avl.icmp)} per insert, ${f2(m.avl.irot)} rotations per insert, ${f2(m.avl.drot)} per delete. Red-black: height ${m.rb.h}, mean depth ${f2(m.rb.depth)}, ${f2(m.rb.scmp)} comparisons per search, ${f2(m.rb.icmp)} per insert, ${f2(m.rb.irot)} rotations per insert, ${f2(m.rb.drot)} per delete, ${f2(m.rb.irec)} recolourings per insert, ${f2(m.rb.drec)} per delete. Invariants and final emptiness: ${ok(m.avl.ok && m.rb.ok)}.`);
    }
    on("avlrb-order", "change", draw); draw();
  });

  /* ── F13 B-tree height vs n for several t ─────────────────────────────── */
  fig("bth-svg", () => {
    const NS = [64, 256, 1024, 4096, 16384, 65536], TS = [2, 3, 8, 64], cache = {};
    function measure(order) {
      if (cache[order]) return cache[order];
      const out = TS.map(t => ({ t, rows: NS.map(n => {
        const keys = order === "sorted" ? Array.from({ length: n }, (_, i) => i + 1) : AL.perm(n, AL.rng(n + t));
        const T = BT.btFromKeys(t, keys), ck = BT.checkBT(T), cs = AL.counter(); BT.btSearch(T, keys[Math.floor(n / 2)], cs);
        return { n, h: ck.h, ok: ck.ok, nodes: ck.nodes, fill: n / (ck.nodes * (2 * t - 1)), reads: cs.get("reads"), bound: BT.btBound(n, t) };
      }) }));
      cache[order] = out; return out;
    }
    function draw() {
      const order = val("bth-order") || "random", data = measure(order);
      const F = AL.frame("#bth-svg", 760, 330, { l: 44, r: 262, t: 14, b: 36 });
      const x = d3.scaleLog().domain([64, 65536]).range([0, F.iw]), y = d3.scaleLinear().domain([0, 16]).range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 4); AL.axisB(F.g, x, F.ih, 6, "n (keys)", d3.format("~s")); AL.axisL(F.g, y, 4, "height (levels below the root)");
      const colors = [AC.bad, AC.a2, AC.accent, AC.good];
      data.forEach((s, i) => {
        F.g.append("path").datum(s.rows).attr("d", d3.line().x(d => x(d.n)).y(d => y(d.h))).attr("fill", "none").attr("stroke", colors[i]).attr("stroke-width", 2);
        F.g.append("path").datum(s.rows).attr("d", d3.line().x(d => x(d.n)).y(d => y(d.bound))).attr("fill", "none").attr("stroke", colors[i]).attr("stroke-dasharray", "3,3").attr("opacity", 0.8);
        F.g.selectAll(null).data(s.rows).join("circle").attr("cx", d => x(d.n)).attr("cy", d => y(d.h)).attr("r", 2.5).attr("fill", colors[i]);
      });
      AL.legend(F.g, TS.map((t, i) => ({ label: `t = ${t}: measured (solid), bound log_t((n+1)/2) (dashed)`, color: colors[i] })), 6, 12, { gap: 14 });
      const L = data.map(s => s.rows[s.rows.length - 1]);
      BD.lines(F.g, F.iw + 12, 6, [`${order} insertion, n = 65 536:`, ``].concat(data.map((s, i) => `t = ${s.t}: h ${L[i].h} (bound ${f2(L[i].bound)}), ${L[i].nodes} nodes, fill ${Math.round(L[i].fill * 100)}%, ${L[i].reads} reads/search ${ok(L[i].ok)}`)).concat([``, `reads/search = nodes visited searching`, `for the median key (≤ h + 1)`, `✓ = size and depth invariants re-verified`]), { hi: [0, 2, 3, 4, 5], size: 10, lh: 15 });
      txt("bth-readout", `${order} insertion, n = 65 536: ` + data.map((s, i) => `t = ${s.t}: height ${L[i].h} against bound ${f2(L[i].bound)}, ${L[i].nodes} nodes, ${Math.round(L[i].fill * 100)}% full, ${L[i].reads} block reads per search`).join("; ") + `. Invariant checks on all ${TS.length * NS.length} trees: ${data.every(s => s.rows.every(r => r.ok)) ? "all pass" : "FAIL"}.`);
    }
    on("bth-order", "change", draw); draw();
  });

  /* ── F14 B-tree insertion with splits, stepped ────────────────────────── */
  const BT_SEQ = [10, 20, 30, 40, 50, 60, 70, 80, 90, 25, 35, 15];
  fig("btins-svg", () => {
    function build(t) {
      const T = BT.BTree(t), frames = [], c = AL.counter(); let total = 0;
      frames.push({ tree: BT.btClone(T.root), title: `empty B-tree, t = ${t}: nodes hold ${t - 1}–${2 * t - 1} keys`, h: 0, reads: 0, splits: 0, total: 0, ok: true });
      BT_SEQ.forEach(k => {
        const c1 = AL.counter(); let pending = null;
        BT.btInsert(T, k, c1, e => {
          if (e.ev === "rootsplit") pending = "the root is full: a new empty root is created above it, and the old root is split";
          else if (e.ev === "split") { frames.push({ tree: BT.btClone(T.root), title: `insert ${k}: ${pending ? pending + " — " : "child is full: split it on the way down — "}${e.mid} rises into [${e.into.join(" ")}]`, k, mid: e.mid, reads: c1.get("reads"), splits: c1.get("splits"), h: BT.btHeight(T), total: total + c1.get("reads"), mid2: true }); pending = null; }
        });
        total += c1.get("reads"); const ck = BT.checkBT(T);
        frames.push({ tree: BT.btClone(T.root), title: `insert ${k} placed in a leaf — ${c1.get("reads")} block read${c1.get("reads") > 1 ? "s" : ""}, ${c1.get("splits")} split${c1.get("splits") === 1 ? "" : "s"}; height ${ck.h}; invariant ${ok(ck.ok)}`, k, reads: c1.get("reads"), splits: c1.get("splits"), h: ck.h, ok: ck.ok, total, allSplits: (frames.filter(f => f.mid2).length) });
      });
      return { frames, T };
    }
    let cur = null;
    function setup() {
      const t = +val("btins-t") || 2; cur = build(t);
      d3.select("#btins-svg").node().parentNode.querySelectorAll("[role=group]").forEach(e => e.remove());
      AL.stepper(d3.select("#btins-svg"), { frames: cur.frames, render: (f, i) => render(f, i, t), delay: 1300, label: "step" });
    }
    function render(f, i, t) {
      const F = AL.frame("#btins-svg", 760, 330, { l: 6, r: 6, t: 24, b: 6 });
      F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11.5).attr("fill", AC.a2).text(`step ${i + 1}: ${f.title}`);
      BD.drawMulti(F.g, f.tree, { x: 0, y: 10, w: 520, levelH: 76, cw: 28, ch: 26, keyFill: (x, j, k) => k === f.k ? "#1f3a4a" : (k === f.mid ? "#3b3a2a" : null), stroke: x => x.keys.length === 2 * t - 1 ? AC.bad : null, label: x => x.keys.length === 2 * t - 1 ? "full" : "" });
      const splitsSoFar = cur.frames.slice(0, i + 1).filter(g => g.mid2).length;
      BD.lines(F.g, 540, 14, [`t = ${t}; keys: ${BT_SEQ.slice(0, f.k ? BT_SEQ.indexOf(f.k) + 1 : 0).join(" ") || "—"}`, `height ${f.h}`, `n = ${BT.btKeys(f.tree).length}, bound log_${t}((n+1)/2) = ${f2(BT.btBound(Math.max(1, BT.btKeys(f.tree).length), t))}`, ``, `this insertion: ${f.reads} block reads`, `splits so far: ${splitsSoFar}`, `block reads so far: ${f.total}`, ``, f.mid2 ? "(mid-insertion)" : `size & depth invariants: ${ok(f.ok)}`, ``, `red outline = full node (${2 * t - 1} keys)`, `a full node is split BEFORE it is entered`], { hi: [1, 4, 5, 6, 8] });
      txt("btins-readout", `Step ${i + 1}: ${f.title}. Height ${f.h}; ${splitsSoFar} splits and ${f.total} block reads so far.`);
    }
    on("btins-t", "change", setup); setup();
  });

  /* ── F15 B-tree deletion, stepped ─────────────────────────────────────── */
  fig("btdel-svg", () => {
    const T = BT.btFromKeys(2, BT_SEQ), frames = [], DELS = [80, 40, 20, 35];
    frames.push({ tree: BT.btClone(T.root), title: `the 12-key tree of the previous figure (t = 2), height ${BT.btHeight(T)}; about to delete ${DELS.join(", ")}`, reads: 0, merges: 0, ok: true, h: BT.btHeight(T) });
    const names = { "1": "case 1 — key in a leaf: remove it", "2a": "case 2a — left child has ≥ t keys: replace by the predecessor", "2b": "case 2b — right child has ≥ t keys: replace by the successor", "2c": "case 2c — both children minimal: merge them around the key", "3a": "case 3a — child minimal, a sibling can lend: transfer through the parent", "3b": "case 3b — child and siblings minimal: merge child with a sibling" };
    DELS.forEach(k => {
      const c = AL.counter();
      BT.btDelete(T, k, c, e => {
        if (e.ev === "case") frames.push({ tree: BT.btClone(T.root), title: `delete ${k}: ${names[e.n]}${e.by ? " " + e.by : ""}${e.dir ? " (" + e.dir + ")" : ""}`, k, by: e.by, reads: c.get("reads"), merges: c.get("merges"), mid: true, h: BT.btHeight(T) });
        else if (e.ev === "merge") frames.push({ tree: BT.btClone(T.root), title: `merged into [${e.keys.join(" ")}]`, k, reads: c.get("reads"), merges: c.get("merges"), mid: true, h: BT.btHeight(T) });
        else if (e.ev === "shrink") frames.push({ tree: BT.btClone(T.root), title: `the root is empty: its only child becomes the root — the tree shrinks to height ${BT.btHeight(T)}`, k, reads: c.get("reads"), merges: c.get("merges"), mid: true, h: BT.btHeight(T) });
      });
      const ck = BT.checkBT(T);
      frames.push({ tree: BT.btClone(T.root), title: `delete ${k} complete — ${c.get("reads")} block reads, ${c.get("merges")} merge${c.get("merges") === 1 ? "" : "s"}, ${c.get("transfers")} transfer${c.get("transfers") === 1 ? "" : "s"}; height ${ck.h}; invariant ${ok(ck.ok)}`, k, reads: c.get("reads"), merges: c.get("merges"), ok: ck.ok, h: ck.h });
    });
    function render(f, i) {
      const F = AL.frame("#btdel-svg", 760, 330, { l: 6, r: 6, t: 24, b: 6 });
      F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11.5).attr("fill", AC.a2).text(`step ${i + 1}: ${f.title}`);
      BD.drawMulti(F.g, f.tree, { x: 0, y: 10, w: 520, levelH: 76, cw: 28, ch: 26, keyFill: (x, j, k) => k === f.k ? AC.bad : k === f.by ? "#3b3a2a" : null, label: x => x.keys.length === 1 ? "minimal" : "" });
      BD.lines(F.g, 540, 14, [`n = ${BT.btKeys(f.tree).length}, height ${f.h}`, ``, f.k ? `this deletion: ${f.reads} block reads` : ``, f.k ? `               ${f.merges} merge(s)` : ``, ``, f.mid ? "(mid-deletion)" : `size & depth invariants: ${ok(f.ok)}`, ``, `red = the key being deleted`, `amber = its replacement`, `"minimal" = t − 1 = 1 key: cannot`, `  lose a key without repair`], { hi: [0, 2, 3, 5] });
      txt("btdel-readout", `Step ${i + 1}: ${f.title}. n = ${BT.btKeys(f.tree).length}, height ${f.h}.`);
    }
    AL.stepper(d3.select("#btdel-svg"), { frames, render, delay: 1300, label: "step" });
  });

  /* ── F16 B⁺-tree with leaf chain and a range scan ─────────────────────── */
  fig("bplus-svg", () => {
    const N = 48, L = 4, FAN = 4, keys = Array.from({ length: N }, (_, i) => i + 1);
    // bulk load: leaves of L keys; internal nodes over FAN children with separators = first key of each child but the first
    const leaves = []; for (let i = 0; i < N; i += L) leaves.push({ keys: keys.slice(i, i + L), children: [], leaf: true, id: leaves.length });
    for (let i = 0; i < leaves.length - 1; i++) leaves[i].next = leaves[i + 1];
    const firstKey = x => x.leaf ? x.keys[0] : firstKey(x.children[0]);   // a separator = the smallest key in the subtree to its right
    let level = leaves; while (level.length > 1) { const up = []; for (let i = 0; i < level.length; i += FAN) { const ch = level.slice(i, i + FAN); up.push({ keys: ch.slice(1).map(firstKey), children: ch, leaf: false }); } level = up; }
    const root = level[0];
    const height = (x => { let h = 0; while (!x.leaf) { x = x.children[0]; h++; } return h; })(root);
    const BTt = BT.btFromKeys(3, keys);
    function btScan(x, a, b, c) { c.add("visits"); const out = []; for (let i = 0; i < x.keys.length; i++) { if (!x.leaf && a < x.keys[i] && (i === 0 || x.keys[i - 1] < b)) out.push(...btScan(x.children[i], a, b, c)); if (x.keys[i] >= a && x.keys[i] <= b) out.push(x.keys[i]); } if (!x.leaf && b > x.keys[x.keys.length - 1]) out.push(...btScan(x.children[x.keys.length], a, b, c)); return out; }
    function draw() {
      let a = +val("bplus-a") || 14, b = +val("bplus-b") || 31; if (a > b) { const t = a; a = b; b = t; } txt("bplus-a-out", a); txt("bplus-b-out", b);
      const c = AL.counter(); const path = []; let x = root;
      while (!x.leaf) { c.add("reads"); path.push(x); let i = 0; while (i < x.keys.length && a >= x.keys[i]) i++; x = x.children[i]; }
      const walked = []; const res = []; while (x) { c.add("reads"); walked.push(x); for (const k of x.keys) if (k >= a && k <= b) res.push(k); if (x.keys[x.keys.length - 1] >= b) break; x = x.next; }
      const cb = AL.counter(); const res2 = btScan(BTt.root, a, b, cb);
      // separator check
      let sepOk = true; (function rec(nd) { if (nd.leaf) return; nd.children.forEach((ch, i) => { const lo = i === 0 ? -Infinity : nd.keys[i - 1], hi = i === nd.keys.length ? Infinity : nd.keys[i]; const ks = BT.btKeys(ch); if (!(ks[0] >= lo && ks[ks.length - 1] < hi)) sepOk = false; rec(ch); }); })(root);
      const pset = new Set(path), wset = new Set(walked);
      const F = AL.frame("#bplus-svg", 760, 320, { l: 6, r: 6, t: 22, b: 6 });
      const d = BD.drawMulti(F.g, root, { x: 0, y: 6, w: 748, levelH: 84, cw: 14, ch: 22, fontSize: 9, fill: nd => pset.has(nd) ? "#3b3a2a" : wset.has(nd) ? "#1f3a2a" : null, stroke: nd => pset.has(nd) ? AC.a2 : wset.has(nd) ? AC.good : null, keyFill: (nd, j, k) => nd.leaf && k >= a && k <= b ? AC.good : null });
      // leaf chain arrows
      leaves.forEach(lf => { if (!lf.next) return; const p = d.pos.get(lf), q = d.pos.get(lf.next); AL.arrow(F.g, p.x + p.w / 2 + 1, p.y + 11, q.x - q.w / 2 - 1, q.y + 11, { color: wset.has(lf) && wset.has(lf.next) ? AC.good : AC.muted, w: 1.2, head: 4 }); });
      F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11.5).attr("fill", AC.a2).text(`scan [${a}, ${b}]: descend ${path.length} internal node${path.length > 1 ? "s" : ""} (amber), then walk ${walked.length} leaf${walked.length > 1 ? "ves" : ""} along the chain (green) — ${c.get("reads")} block reads, ${res.length} results`);
      BD.lines(F.g, 0, 262, [`B⁺-tree: ${leaves.length} leaves of ${L} keys, fan-out ${FAN}, height ${height}; separators partition correctly: ${ok(sepOk)}`, `same scan on a B-tree (t = 3, same keys, in-order walk pruned to the range): ${cb.get("visits")} node reads, ${res2.length} results — same answer: ${ok(res.join() === res2.join())}`, `single-key lookup on the B⁺-tree always costs h + 1 = ${height + 1} reads; a B-tree may stop at an internal node`], { hi: [0, 1], size: 10.5, lh: 16 });
      txt("bplus-readout", `Range [${a}, ${b}]: B⁺-tree reads ${path.length} internal nodes then ${walked.length} leaves along the chain, ${c.get("reads")} blocks for ${res.length} results; the B-tree (t = 3) built from the same keys reads ${cb.get("visits")} nodes for the same ${res2.length} results (answers identical: ${ok(res.join() === res2.join())}). Separators verified: ${ok(sepOk)}.`);
    }
    on("bplus-a", "input", draw); on("bplus-b", "input", draw); draw();
  });

  /* ── F17 a splay stepped, with Φ ──────────────────────────────────────── */
  fig("splay-svg", () => {
    const SEQ = [8, 6, 7, 2, 1, 5, 4, 3], KEY = 3;
    const T = BT.SP(); for (const k of SEQ) BT.spInsertNoSplay(T, k);
    const x = BT.spFind(T, KEY), n = BT.spSize(T.root);
    const rx = Math.log2(BT.spSize(x)), rt = Math.log2(n), d0 = BT.spDepth(x), h0 = BT.spHeight(T.root);
    const plain = () => { const rec = y => y ? { key: y.key, left: rec(y.left), right: rec(y.right) } : null; return rec(T.root); };
    const frames = [{ tree: plain(), title: `built without splaying: key ${KEY} at depth ${d0}, height ${h0}; Φ = ${f3(BT.spPhi(T.root))}`, phi: BT.spPhi(T.root), x: KEY, rots: 0, am: 0, sum: 0 }];
    let phi = BT.spPhi(T.root), sum = 0; const c = AL.counter(); let pendingTitle = null, marks = null, rotBefore = 0;
    BT.splay(T, x, c, e => {
      if (e.ev === "after") { const phi2 = BT.spPhi(T.root), rots = c.get("rot") - rotBefore, am = rots + phi2 - phi; sum += am; frames.push({ tree: plain(), title: `${pendingTitle}: ${rots} rotation${rots > 1 ? "s" : ""}; Φ ${f3(phi)} → ${f3(phi2)} (ΔΦ ${f3(phi2 - phi)}); amortised ${f3(am)}`, phi: phi2, x: KEY, marks, rots: c.get("rot"), am, sum }); phi = phi2; rotBefore = c.get("rot"); }
      else { pendingTitle = `${e.ev} at x = ${e.x}, p = ${e.p}${e.g ? ", g = " + e.g : ""}`; marks = { p: e.p, g: e.g }; frames.push({ tree: plain(), title: `next: ${pendingTitle} — ${e.ev === "zig-zig" ? "rotate at g, then at p" : e.ev === "zig-zag" ? "rotate at p, then at g" : "one rotation at p"}`, phi, x: KEY, marks, rots: c.get("rot"), am: null, sum, pre: true }); }
    });
    const bound = 3 * (rt - rx) + 1;
    frames.push({ tree: plain(), title: `done: ${c.get("rot")} rotations in ${c.get("substeps")} steps; Σ amortised = ${f3(sum)} ≤ 3·(r(root) − r(x)) + 1 = ${f3(bound)} ${ok(sum <= bound + 1e-9)}; height ${h0} → ${BT.spHeight(T.root)}; BST order ${ok(BT.checkSP(T))}`, phi, x: KEY, rots: c.get("rot"), am: null, sum, final: true });
    function render(f, i) {
      const F = AL.frame("#splay-svg", 760, 330, { l: 6, r: 6, t: 24, b: 6 });
      F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.a2).text(`step ${i + 1}: ${f.title}`);
      const m = f.marks || {};
      const sizes = new Map(); (function rec(y) { if (!y) return 0; const s = 1 + rec(y.left) + rec(y.right); sizes.set(y.key, s); return s; })(f.tree);
      BD.drawBin(F.g, f.tree, { x: 0, y: 4, w: 470, levelH: 48, r: 14, fontSize: 12, fill: nd => nd.key === f.x ? AC.a2 : nd.key === m.p ? "#3b2a4a" : nd.key === m.g ? "#1f3a4a" : null, stroke: nd => nd.key === m.p ? AC.violet : nd.key === m.g ? AC.teal : null, textFill: nd => nd.key === f.x ? AC.bg : null, label: nd => `size ${sizes.get(nd.key)} · r=${f2(Math.log2(sizes.get(nd.key)))}` });
      BD.lines(F.g, 490, 14, [`Φ = Σ log₂ size = ${f3(f.phi)}`, `rotations so far: ${f.rots}`, `Σ amortised so far: ${f3(f.sum)}`, ``, `r(x) before = log₂ ${BT.spSize(x) ? Math.pow(2, rx) : 1} = ${f2(rx)}`, `r(root) = log₂ ${n} = ${f2(rt)}`, `access lemma: ≤ 3·(${f2(rt)} − ${f2(rx)}) + 1`, `             = ${f2(bound)}`, ``, `amortised = rotations + ΔΦ`, `zig-zig / zig-zag ≤ 3·Δr(x)`, `zig ≤ 3·Δr(x) + 1`], { hi: [0, 1, 2, 7] });
      txt("splay-readout", `Step ${i + 1}: ${f.title}. Φ = ${f3(f.phi)}, rotations so far ${f.rots}, Σ amortised ${f3(f.sum)} against the bound ${f2(bound)}.`);
    }
    AL.stepper(d3.select("#splay-svg"), { frames, render, delay: 1500, label: "step" });
  });

  /* ── F18 cost per access over a sequence ──────────────────────────────── */
  fig("splayseq-svg", () => {
    const n = 128;
    function run(mode) {
      const T = BT.SP(); for (let k = 1; k <= n; k++) BT.spInsert(T, k);
      const phi0 = BT.spPhi(T.root), h0 = BT.spHeight(T.root);
      const r = AL.rng(3); const costs = []; const seq = [];
      for (let i = 0; i < n; i++) { const k = mode === "seq" ? i + 1 : mode === "rand" ? AL.randInt(r, 1, n) : 1 + (i % 4); seq.push(k); const c = AL.counter(); BT.spSearch(T, k, c); costs.push(c.get("rot")); }
      const total = costs.reduce((a, b) => a + b, 0);
      return { costs, total, phi0, h0, bound: n * (3 * Math.log2(n) + 1) + phi0, okBST: BT.checkSP(T), hEnd: BT.spHeight(T.root) };
    }
    function draw() {
      const mode = val("splayseq-mode") || "seq", R = run(mode);
      const F = AL.frame("#splayseq-svg", 760, 300, { l: 44, r: 230, t: 16, b: 34 });
      const x = d3.scaleBand().domain(d3.range(n)).range([0, F.iw]).padding(0.15), y = d3.scaleLinear().domain([0, Math.max(30, Math.max(...R.costs))]).range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 5); AL.axisB(F.g, x.copy().domain(d3.range(0, n, 16)).range([0, F.iw]), F.ih, 8, "access number"); AL.axisL(F.g, y, 5, "rotations for this access");
      F.g.selectAll("rect").data(R.costs).join("rect").attr("x", (d, i) => x(i)).attr("y", d => y(d)).attr("width", x.bandwidth()).attr("height", d => F.ih - y(d)).attr("fill", (d, i) => i === 0 && mode === "seq" ? AC.bad : AC.accent);
      const amort = 3 * Math.log2(n) + 1; F.g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(amort)).attr("y2", y(amort)).attr("stroke", AC.a2).attr("stroke-dasharray", "4,3");
      F.g.append("text").attr("x", F.iw - 4).attr("y", y(amort) - 4).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.a2).text(`3·log₂ n + 1 = ${f1(amort)} (amortised bound per access)`);
      const mean = R.total / n;
      BD.lines(F.g, F.iw + 12, 6, [`n = ${n}; start: chain of height ${R.h0}`, `(inserted 1 … ${n} in order, splaying)`, ``, `first access: ${R.costs[0]} rotations`, `max per access: ${Math.max(...R.costs)}`, `total: ${R.total} rotations`, `mean per access: ${f2(mean)}`, ``, `actual total ≤ m(3log₂n+1) + Φ₀`, `  ${R.total} ≤ ${f1(R.bound)} ${ok(R.total <= R.bound)}`, ``, `first 8: ${R.costs.slice(0, 8).join(" ")}`, `height at the end: ${R.hEnd}`, `BST order: ${ok(R.okBST)}`], { hi: [3, 4, 5, 6, 9], size: 10.5, lh: 15.5 });
      txt("splayseq-readout", `${mode === "seq" ? "Sequential access 1…128" : mode === "rand" ? "128 random accesses" : "Working set of 4 keys, 128 accesses"} on a splay tree that starts as a chain of height ${R.h0}: first access ${R.costs[0]} rotations, maximum ${Math.max(...R.costs)}, total ${R.total}, mean ${f2(mean)} per access; the amortised-bound total m·(3·log₂ n + 1) + Φ₀ = ${f1(R.bound)} holds ${ok(R.total <= R.bound)}. First eight costs: ${R.costs.slice(0, 8).join(", ")}. Final height ${R.hEnd}.`);
    }
    on("splayseq-mode", "change", draw); draw();
  });

  /* ── F19 skip list: build, search stepped, insert, delete ─────────────── */
  fig("skip-svg", () => {
    const KEYS = [3, 7, 12, 19, 25, 31, 44, 50, 58, 66, 71, 89];
    function build() { const S = BT.SL(0.5, AL.rng(24)); const flips = {}; for (const k of KEYS) { const c = AL.counter(); BT.slInsert(S, k, c); flips[k] = c.get("flips"); } return { S, flips }; }
    let st = null;
    function setup() {
      const op = val("skip-op") || "search"; const { S, flips } = build();
      const frames = []; const snapshot = (title, hi, extra) => { const ck = BT.checkSL(S); frames.push(Object.assign({ title, hi: hi || [], levels: S.levels, ok: ck.ok, pointers: ck.pointers, n: ck.n, flips: Object.assign({}, flips) }, extra || {})); };
      const stepSearch = (key) => { const c = AL.counter(); const path = []; const r = BT.slSearch(S, key, c, e => { path.push({ level: e.i, at: e.at }); snapshot(`search ${key}: level ${e.i} — stopped at ${e.at === -Infinity ? "−∞" : e.at}${S.head.next[e.i] === null && e.at === -Infinity ? " (level empty)" : ""}, next is ${(function () { let x = S.head; while (x.key !== e.at) x = x.next[0]; return x.next[e.i] ? x.next[e.i].key : "null"; })()} — drop`, [], { path: path.slice(), cmp: c.get("cmp"), follows: c.get("follows"), key }); }); snapshot(`search ${key}: at level 0 the next node is ${r.found ? r.found.key + " — FOUND" : "not " + key + " — absent"}; ${c.get("cmp")} comparisons, ${c.get("follows")} pointer follows`, [], { path: path.slice(), cmp: c.get("cmp"), follows: c.get("follows"), key, found: !!r.found, done: true }); };
      // draw the towers as a static first frame
      snapshot(`12 keys inserted with p = ½ (seed fixed): ${S.levels} levels, ${BT.checkSL(S).pointers} forward pointers = ${f2(BT.checkSL(S).pointers / 12)} per node`, [], { intro: true });
      if (op === "search") stepSearch(58);
      else if (op === "insert") { const c = AL.counter(); BT.slInsert(S, 40, c); flips[40] = c.get("flips"); snapshot(`insert 40: ${c.get("flips")} flip${c.get("flips") > 1 ? "s" : ""} (${c.get("heads")} heads, then tails) → tower of ${c.get("flips")}; ${c.get("links")} pointer splices; invariant ${ok(BT.checkSL(S).ok)}`, [], { newKey: 40 }); stepSearch(40); }
      else { const c = AL.counter(); const z = BT.slNodes(S).find(x => x.key === 31); BT.slDelete(S, 31, c); snapshot(`delete 31: a ${z.lvl}-level tower unlinked at each level (${c.get("links")} splices); levels now ${S.levels}; invariant ${ok(BT.checkSL(S).ok)}`, [], {}); stepSearch(58); }
      d3.select("#skip-svg").node().parentNode.querySelectorAll("[role=group]").forEach(e => e.remove());
      const Sref = S;
      st = AL.stepper(d3.select("#skip-svg"), { frames, render: (f, i) => render(f, i, Sref), delay: 1300, label: "step" });
    }
    function render(f, i, S) {
      const F = AL.frame("#skip-svg", 760, 300, { l: 26, r: 6, t: 24, b: 6 });
      F.g.append("text").attr("x", -20).attr("y", -8).attr("font-size", 11).attr("fill", AC.a2).text(`step ${i + 1}: ${f.title}`);
      const path = f.path || [], last = path[path.length - 1];
      const visited = new Set(path.map(p => `${p.level}:${p.at}`));
      BD.drawSkip(F.g, S, { x: 0, y: 6, w: 500, rowH: 26, cw: 30, mark: (lv, nd) => (f.newKey && nd.key === f.newKey) ? AC.good : visited.has(`${lv}:${nd.key}`) ? (last && last.level === lv && last.at === nd.key ? AC.a2 : "#7c5a1a") : (f.done && f.found && nd.key === f.key && lv === 0) ? AC.good : null });
      const flipsList = Object.keys(f.flips).map(k => +k).sort((a, b) => a - b).map(k => `${k}:${f.flips[k]}`).join("  ");
      BD.lines(F.g, 505, 8, [`levels ${f.levels}, n = ${f.n}`, `forward pointers ${f.pointers} (${f2(f.pointers / f.n)} / node)`, `expected 1/(1−p) = 2`, ``, f.cmp !== undefined ? `comparisons so far: ${f.cmp}` : ``, f.follows !== undefined ? `pointer follows so far: ${f.follows}` : ``, ``, `invariant re-checked: ${ok(f.ok)}`, ``, `tower height per key (flips):`].concat(flipsList.match(/.{1,26}(\s|$)/g) || []), { hi: [0, 1, 4, 5, 7], size: 10, lh: 15 });
      txt("skip-readout", `Step ${i + 1}: ${f.title}. ${f.levels} levels, ${f.pointers} pointers over ${f.n} keys; invariant ${ok(f.ok)}.${f.cmp !== undefined ? ` Comparisons ${f.cmp}, pointer follows ${f.follows}.` : ""}`);
    }
    on("skip-op", "change", setup); setup();
  });

  /* ── F20 skip-list height distribution and costs ──────────────────────── */
  fig("skiph-svg", () => {
    const n = 1024, T = 200, cache = {};
    function measure(p) {
      if (cache[p]) return cache[p];
      const hist = {}; let sl = 0, mx = 0, sc = 0, sf = 0, sp = 0, allOk = true;
      for (let t = 0; t < T; t++) { const S = BT.SL(p, AL.rng(1000 * n + t)); const keys = AL.perm(n, AL.rng(t)); for (const k of keys) BT.slInsert(S, k); const lv = S.levels; sl += lv; mx = Math.max(mx, lv); hist[lv] = (hist[lv] || 0) + 1; const c = AL.counter(); for (const k of keys) BT.slSearch(S, k, c); sc += c.get("cmp") / n; sf += c.get("follows") / n; const ck = BT.checkSL(S); sp += ck.pointers / n; allOk = allOk && ck.ok; }
      const bound = Math.log(n) / Math.log(1 / p) + 1 / (1 - p);
      cache[p] = { hist, mean: sl / T, max: mx, cmp: sc / T, follows: sf / T, ptr: sp / T, bound, allOk }; return cache[p];
    }
    function draw() {
      const p = +val("skiph-p") || 0.5, M = measure(p);
      const F = AL.frame("#skiph-svg", 760, 300, { l: 44, r: 250, t: 16, b: 34 });
      const lv = Object.keys(M.hist).map(Number), lo = Math.min(...lv), hi = Math.max(...lv);
      const x = d3.scaleBand().domain(d3.range(lo, hi + 1)).range([0, F.iw]).padding(0.15), y = d3.scaleLinear().domain([0, Math.max(...Object.values(M.hist)) * 1.15]).range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 4); F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x)); AL.axisL(F.g, y, 4, `lists (of ${T})`);
      F.g.append("text").attr("x", F.iw).attr("y", F.ih + 30).attr("text-anchor", "end").attr("font-size", 11).attr("fill", AC.muted).text("number of levels (including level 0)");
      x.domain().forEach(l => { const v = M.hist[l] || 0; F.g.append("rect").attr("x", x(l)).attr("y", y(v)).attr("width", x.bandwidth()).attr("height", F.ih - y(v)).attr("fill", AC.violet); if (v) F.g.append("text").attr("x", x(l) + x.bandwidth() / 2).attr("y", y(v) - 3).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", AC.ink).text(v); });
      const bx = x(Math.floor(M.bound + 1)) !== undefined ? x(Math.floor(M.bound + 1)) + x.bandwidth() * ((M.bound + 1) % 1) : F.iw;
      F.g.append("line").attr("x1", bx).attr("x2", bx).attr("y1", 0).attr("y2", F.ih).attr("stroke", AC.a2).attr("stroke-dasharray", "4,3");
      F.g.append("text").attr("x", bx + 4).attr("y", 12).attr("font-size", 10).attr("fill", AC.a2).text(`E[levels] bound = log_{1/p} n + 1/(1−p) + 1 = ${f2(M.bound + 1)}`);
      BD.lines(F.g, F.iw + 12, 6, [`p = ${p}, n = ${n}, ${T} lists`, ``, `levels: mean ${f2(M.mean)}, max ${M.max}`, `bound on the mean: ${f2(M.bound + 1)}`, ``, `per search (all keys, all lists):`, `  comparisons ${f2(M.cmp)}`, `  pointer follows ${f2(M.follows)}`, `  2·log₂ n = ${f1(2 * Math.log2(n))}`, ``, `pointers per node ${f3(M.ptr)}`, `  expected 1/(1−p) = ${f3(1 / (1 - p))}`, ``, `invariant on all ${T}: ${ok(M.allOk)}`], { hi: [2, 3, 6, 7, 10, 11], size: 10.5, lh: 15.5 });
      txt("skiph-readout", `p = ${p}: over ${T} lists of ${n} keys, mean ${f2(M.mean)} levels (bound on the mean ${f2(M.bound + 1)}), maximum ${M.max}; histogram ${Object.keys(M.hist).sort((a, b) => a - b).map(k => `${k}: ${M.hist[k]}`).join(", ")}. Searching every key: ${f2(M.cmp)} comparisons and ${f2(M.follows)} pointer follows per search; ${f3(M.ptr)} forward pointers per node against 1/(1−p) = ${f3(1 / (1 - p))}. Invariant on all lists: ${ok(M.allOk)}.`);
    }
    on("skiph-p", "change", draw); draw();
  });

  /* ── F21 trie: search / prefix, space model, compressed view ─────────── */
  fig("trie-svg", () => {
    const WORDS = ["a", "an", "and", "ant", "car", "card", "cat", "in", "inn", "tea", "ten", "to", "tree", "trie", "try"];
    const T = BT.Trie(); for (const w of WORDS) BT.trieInsert(T, w);
    const nodes = BT.trieNodes(T.root), edges = nodes.length - 1, terms = nodes.filter(x => x.end).length, chars = WORDS.reduce((s, w) => s + w.length, 0);
    const R = BT.trieCompress(T.root), rnodes = BT.radixNodes(R);
    const bytes = { array: nodes.length * (26 * 8 + 1), map: nodes.length * 16 + edges * 16, tst: BT.tstCount(T) * 32, radix: rnodes.reduce((s, x) => s + 16 + x.label.length + 16 * x.ch.length, 0) };
    // general-tree layout for both views
    function layout(root, kids, w, levelH) {
      const by = new Map(); let li = 0; const leaves = []; (function cnt(x) { const k = kids(x); if (!k.length) leaves.push(x); k.forEach(cnt); })(root);
      const unit = w / Math.max(1, leaves.length);
      (function rec(x, d) { const k = kids(x); let cx; if (!k.length) { cx = (li + 0.5) * unit; li++; } else { const xs = k.map(c => rec(c, d + 1)); cx = (Math.min(...xs) + Math.max(...xs)) / 2; } by.set(x, { x: cx, y: d * levelH + 14, d }); return cx; })(root, 0);
      return by;
    }
    function draw() {
      const view = val("trie-view") || "trie", q = val("trie-q") || "tre";
      const isPrefix = q === "tre" || q === "an";
      const c = AL.counter(); const sr = BT.trieSearch(T, q, c);
      let out = []; if (isPrefix && sr.prefixNode) out = BT.trieCollect(sr.prefixNode, q, [], c);
      const pathSet = new Set(sr.path);
      const F = AL.frame("#trie-svg", 760, 360, { l: 6, r: 6, t: 20, b: 6 });
      if (view === "trie") {
        const by = layout(T.root, x => Object.keys(x.ch).sort().map(k => x.ch[k]), 520, 60);
        for (const [x, p] of by) Object.keys(x.ch).sort().forEach(k => { const ch = x.ch[k], qq = by.get(ch); const onPath = pathSet.has(x) && pathSet.has(ch); F.g.append("line").attr("x1", p.x).attr("y1", p.y).attr("x2", qq.x).attr("y2", qq.y).attr("stroke", onPath ? AC.a2 : AC.line).attr("stroke-width", onPath ? 2.5 : 1.2); F.g.append("text").attr("x", (p.x + qq.x) / 2 + 5).attr("y", (p.y + qq.y) / 2 + 3).attr("font-size", 10.5).attr("fill", onPath ? AC.a2 : AC.muted).text(k); });
        for (const [x, p] of by) { const inOut = isPrefix && sr.prefixNode && out.length && (function isDesc(y) { if (y === x) return true; return Object.values(y.ch).some(isDesc); })(sr.prefixNode) && (function isDesc2(y) { if (y === x) return true; return Object.values(y.ch).some(isDesc2); })(sr.prefixNode); F.g.append("circle").attr("cx", p.x).attr("cy", p.y).attr("r", x.end ? 7 : 5).attr("fill", pathSet.has(x) ? AC.a2 : x.end ? AC.good : AC.panel2).attr("stroke", x.end ? AC.good : AC.line).attr("stroke-width", x.end ? 2 : 1); }
      } else {
        const by = layout(R, x => x.ch, 520, 60);
        // path in the radix view: the nodes whose accumulated label is a prefix of q or q a prefix of it
        const acc = new Map(); (function rec(x, s) { acc.set(x, s); x.ch.forEach(ch => rec(ch, s + ch.label)); })(R, "");
        const onP = x => { const s = acc.get(x); return q.startsWith(s) || (s.startsWith(q) && acc.get(x).length - x.label.length < q.length); };
        for (const [x, p] of by) x.ch.forEach(ch => { const qq = by.get(ch); const on = onP(x) && onP(ch); F.g.append("line").attr("x1", p.x).attr("y1", p.y).attr("x2", qq.x).attr("y2", qq.y).attr("stroke", on ? AC.a2 : AC.line).attr("stroke-width", on ? 2.5 : 1.2); F.g.append("text").attr("x", (p.x + qq.x) / 2 + 5).attr("y", (p.y + qq.y) / 2 + 3).attr("font-size", 10.5).attr("fill", on ? AC.a2 : AC.muted).text(ch.label); });
        for (const [x, p] of by) F.g.append("circle").attr("cx", p.x).attr("cy", p.y).attr("r", x.end ? 7 : 5).attr("fill", onP(x) ? AC.a2 : x.end ? AC.good : AC.panel2).attr("stroke", x.end ? AC.good : AC.line).attr("stroke-width", x.end ? 2 : 1);
      }
      const result = isPrefix ? `prefix "${q}": ${sr.prefixNode ? out.length + " word(s): " + out.join(", ") : "no such prefix"}` : `search "${q}": ${sr.found ? "FOUND" : sr.prefixNode ? "node exists but is not terminal — absent as a key" : "path breaks after " + c.get("steps") + " character(s) — absent"}`;
      BD.lines(F.g, 540, 8, [`${WORDS.length} words, ${chars} characters`, `trie: ${nodes.length} nodes, ${edges} edges, ${terms} terminal`, `radix trie: ${rnodes.length} nodes`, ``, result.length > 34 ? result.slice(0, 34) : result, result.length > 34 ? result.slice(34) : ``, `edges followed: ${c.get("steps")}${isPrefix ? `, nodes enumerated: ${c.get("collect")}` : ""}`, ``, `space model (bytes):`, `  array children: 26×8 + 1 per node`, `    = ${bytes.array}`, `  map children: 16/node + 16/edge`, `    = ${bytes.map}`, `  ternary search trie: 32/char node`, `    = ${bytes.tst}`, `  radix: 16/node + label + 16/edge`, `    = ${bytes.radix}`], { hi: [1, 2, 4, 5, 6, 10, 12, 14, 16], size: 10, lh: 15 });
      txt("trie-readout", `${result}; ${c.get("steps")} edges followed${isPrefix ? `, ${c.get("collect")} nodes enumerated` : ""}. Trie: ${nodes.length} nodes, ${edges} edges, ${terms} terminal, ${chars} characters in ${WORDS.length} words; compressed: ${rnodes.length} nodes. Space model: array-children ${bytes.array} B, map-children ${bytes.map} B, ternary search trie ${bytes.tst} B, radix ${bytes.radix} B.`);
    }
    on("trie-view", "change", draw); on("trie-q", "change", draw); draw();
  });

  /* ── F22 segment tree query decomposition ─────────────────────────────── */
  fig("seg-svg", () => {
    const A8 = [5, 3, 7, 1, 4, 6, 2, 8], A11 = [5, 3, 7, 1, 4, 6, 2, 8, 9, 0, 3];
    function draw() {
      const n = +val("seg-n") || 8, a = (n === 8 ? A8 : A11).slice();
      const lEl = $("seg-l"), rEl = $("seg-r"); if (lEl) lEl.max = n - 1; if (rEl) rEl.max = n - 1;
      let l = Math.min(+val("seg-l"), n - 1), r = Math.min(+val("seg-r"), n - 1); if (isNaN(l)) l = 1; if (isNaN(r)) r = 6; if (l > r) { const t = l; l = r; r = t; }
      const nv = +val("seg-upd"); txt("seg-l-out", l); txt("seg-r-out", r); txt("seg-upd-out", nv);
      const S = BT.Seg(a); const c = AL.counter(); const canon = [], visited = []; const ans = BT.segQuery(S, l, r, c, e => { visited.push(e.v); if (e.kind === "canonical") canon.push(e); });
      const plain = BT.plainRange(a, l, r);
      const cu = AL.counter(); const upath = []; BT.segUpdate(S, 3, nv, cu, e => upath.push(e.v)); const a2 = a.slice(); a2[3] = nv;
      const c2 = AL.counter(); const ans2 = BT.segQuery(S, l, r, c2); const plain2 = BT.plainRange(a2, l, r);
      const H = Math.ceil(Math.log2(n));
      const F = AL.frame("#seg-svg", 760, 330, { l: 6, r: 6, t: 20, b: 6 });
      const S0 = BT.Seg(a);  // pre-update values for the drawing
      const cset = new Set(canon.map(e => e.v)), vset = new Set(visited);
      BD.drawSeg(F.g, S0, { x: 0, y: 0, w: 520, levelH: 46, cw: 28, fill: nd => cset.has(nd.v) ? "#1f3a2a" : vset.has(nd.v) ? "#3b3a2a" : null, stroke: nd => cset.has(nd.v) ? AC.good : vset.has(nd.v) ? AC.a2 : null, text: nd => `${nd.val}` });
      BD.lines(F.g, 540, 8, [`query [${l}, ${r}] on n = ${n}`, `canonical nodes (${canon.length}):`].concat(canon.map(e => `  [${e.lo},${e.hi}] = ${e.val}`)).concat([`sum = ${ans}; plain scan = ${plain} ${ok(ans === plain)}`, `visited ${c.get("visits")} ≤ 4·ceil(log₂ n) = ${4 * H} ${ok(c.get("visits") <= 4 * H)}`, `canonical ${canon.length} ≤ 2·ceil(log₂ n) = ${2 * H} ${ok(canon.length <= 2 * H)}`, ``, `update a[3] ← ${nv}: ${cu.get("visits")} nodes`, `  recomputed (leaf → root)`, `query again: ${ans2}; scan ${plain2} ${ok(ans2 === plain2)}`, `tree re-verified: ${ok(BT.checkSeg(S))}`]), { hi: [0, 2 + canon.length, 3 + canon.length, 4 + canon.length, 6 + canon.length, 8 + canon.length], size: 10, lh: 15 });
      txt("seg-readout", `Query [${l}, ${r}] over ${n} values: ${canon.length} canonical nodes ${canon.map(e => `[${e.lo},${e.hi}]=${e.val}`).join(" ")}, sum ${ans} (plain scan ${plain} ${ok(ans === plain)}); ${c.get("visits")} nodes visited against the bound ${4 * H}. Setting a[3] to ${nv} recomputed ${cu.get("visits")} nodes; the query then gives ${ans2} (scan ${plain2} ${ok(ans2 === plain2)}). Every internal node re-verified as the sum of its children: ${ok(BT.checkSeg(S))}.`);
    }
    ["seg-n", "seg-l", "seg-r", "seg-upd"].forEach(id => { on(id, "input", draw); on(id, "change", draw); }); draw();
  });

  /* ── F23 lazy propagation, stepped ────────────────────────────────────── */
  fig("lazy-svg", () => {
    const a = [5, 3, 7, 1, 4, 6, 2, 8]; const S = BT.Seg(a); const plain = a.slice();
    const frames = []; const snap = (title, marks, c) => frames.push({ title, marks: marks || {}, nodes: BT.segNodes(S), a: S.a.slice(), visits: c ? c.get("visits") : 0, pushes: c ? c.get("pushes") : 0, ok: BT.checkSeg(S) && S.a.every((v, i) => v === plain[i]) });
    snap("the segment tree of sums over [5, 3, 7, 1, 4, 6, 2, 8], no tags");
    const ops = [["add", 2, 6, 3], ["query", 4, 7], ["query", 5, 5], ["add", 0, 5, 2], ["query", 2, 3]];
    ops.forEach(op => {
      const c = AL.counter(); const tagged = [], partial = [], canon = [];
      if (op[0] === "add") { BT.segRangeAdd(S, op[1], op[2], op[3], c, e => { if (e.kind === "tagged") tagged.push(e.v); else partial.push(e.v); }); for (let i = op[1]; i <= op[2]; i++) plain[i] += op[3]; snap(`a[${op[1]}..${op[2]}] += ${op[3]}: tagged ${tagged.length} canonical node(s), updated ${partial.length} split node(s) above — ${c.get("visits")} visits, ${c.get("pushes")} push(es)`, { tagged, partial }, c); }
      else { const res = BT.segLazyQuery(S, op[1], op[2], c, e => { if (e.kind === "canonical") canon.push(e.v); else partial.push(e.v); }); const pl = BT.plainRange(plain, op[1], op[2]); snap(`query [${op[1]}, ${op[2]}] = ${res} (plain array: ${pl} ${ok(res === pl)}): ${canon.length} canonical node(s) read, ${c.get("pushes")} push(es) on the way down`, { canon, partial }, c); }
    });
    function render(f, i) {
      const F = AL.frame("#lazy-svg", 760, 330, { l: 6, r: 6, t: 22, b: 6 });
      F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.a2).text(`step ${i + 1}: ${f.title}`);
      const m = f.marks, tg = new Set(m.tagged || []), pa = new Set(m.partial || []), cn = new Set(m.canon || []);
      const fake = { n: 8, a: f.a, t: [], lazy: [] }; f.nodes.forEach(nd => { fake.t[nd.v] = nd.val; fake.lazy[nd.v] = nd.lazy; });
      BD.drawSeg(F.g, fake, { x: 0, y: 0, w: 540, levelH: 50, cw: 30, fill: nd => tg.has(nd.v) || cn.has(nd.v) ? "#1f3a2a" : pa.has(nd.v) ? "#3b3a2a" : null, stroke: nd => tg.has(nd.v) || cn.has(nd.v) ? AC.good : pa.has(nd.v) ? AC.a2 : (nd.lazy ? AC.violet : null), text: nd => nd.lazy ? `${nd.val} (+${nd.lazy})` : `${nd.val}` });
      BD.lines(F.g, 560, 8, [`this operation:`, `  visits ${f.visits}`, `  pushes ${f.pushes}`, ``, `"(+c)" = pending tag for`, `  the node's children`, ``, `green = canonical / tagged`, `amber = split, entered`, `violet outline = holds a tag`, ``, `full re-verification (push`, `all tags, compare with the`, `plain array): ${ok(f.ok)}`], { hi: [1, 2, 13], size: 10, lh: 15 });
      txt("lazy-readout", `Step ${i + 1}: ${f.title}. Verification against a fully pushed recomputation and the plain array: ${ok(f.ok)}.`);
    }
    AL.stepper(d3.select("#lazy-svg"), { frames, render, delay: 1600, label: "step" });
  });

  /* ── F24 Fenwick tree: prefix query and update stepped ────────────────── */
  fig("fen-svg", () => {
    const A = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8];
    const bin = i => i.toString(2).padStart(4, "0");
    function setup() {
      const qi = +val("fen-i") || 11, ui = +val("fen-u") || 5; txt("fen-i-out", qi); txt("fen-u-out", ui);
      const F0 = BT.Fen(A); const frames = [];
      const snap = (title, marks, extra) => frames.push(Object.assign({ title, t: F0.t.slice(), a: F0.a.slice(), marks: marks || {}, ok: BT.checkFen(F0) }, extra || {}));
      snap(`the tree array: t[i] = sum of a over (i − lowbit(i), i]; t[0] unused — every entry re-verified against its definition`);
      const c = AL.counter(); const steps = []; let s = 0;
      BT.fenPrefix(F0, qi, c, e => { s += e.add; steps.push(e.i); snap(`prefix(${qi}): i = ${e.i} = ${bin(e.i)}₂, lowbit ${BT.lowbit(e.i)}: add t[${e.i}] = ${e.add} → ${s}; next i = ${e.i} − ${BT.lowbit(e.i)} = ${e.next}`, { visited: steps.slice(), cur: e.i }, { running: s }); });
      const plain = A.slice(0, qi).reduce((x, y) => x + y, 0);
      snap(`prefix(${qi}) = ${s} after ${c.get("visits")} entries (one per set bit of ${bin(qi)}₂); plain loop over a[1..${qi}] = ${plain} ${ok(s === plain)}`, { visited: steps.slice() }, { done: true, result: s, plain });
      const cu = AL.counter(); const ust = [];
      BT.fenUpdate(F0, ui, 2, cu, e => { ust.push(e.i); snap(`update(${ui}, +2): i = ${e.i} = ${bin(e.i)}₂, lowbit ${BT.lowbit(e.i)}: t[${e.i}] += 2; next i = ${e.i} + ${BT.lowbit(e.i)} = ${e.next}${e.next > 12 ? " > n: stop" : ""}`, { updated: ust.slice(), cur: e.i }); });
      const c3 = AL.counter(); const s2 = BT.fenPrefix(F0, qi, c3); const plain2 = F0.a.slice(0, qi).reduce((x, y) => x + y, 0);
      snap(`after the update (${cu.get("visits")} entries changed): prefix(${qi}) = ${s2}, plain loop ${plain2} ${ok(s2 === plain2)}; all 12 entries re-verified against their definition: ${ok(BT.checkFen(F0))}`, { updated: ust.slice() }, { done: true });
      d3.select("#fen-svg").node().parentNode.querySelectorAll("[role=group]").forEach(e => e.remove());
      AL.stepper(d3.select("#fen-svg"), { frames, render, delay: 1500, label: "step" });
    }
    function render(f, i) {
      const F = AL.frame("#fen-svg", 760, 330, { l: 40, r: 6, t: 22, b: 6 });
      F.g.append("text").attr("x", -34).attr("y", -8).attr("font-size", 11).attr("fill", AC.a2).text(`step ${i + 1}: ${f.title}`);
      const n = 12, cw = 46, step = 50;
      const vis = new Set(f.marks.visited || []), upd = new Set(f.marks.updated || []);
      // brackets: range of each t[i]
      const by = 120; const depthOf = i => Math.log2(BT.lowbit(i));
      for (let i = 1; i <= n; i++) { const lo = i - BT.lowbit(i) + 1, hi = i; const y = by - 12 - depthOf(i) * 16; const x1 = (lo - 1) * step + 2, x2 = (hi - 1) * step + cw - 2; const col = vis.has(i) ? AC.a2 : upd.has(i) ? AC.good : AC.muted; F.g.append("path").attr("d", `M${x1},${y + 6} L${x1},${y} L${x2},${y} L${x2},${y + 6}`).attr("fill", "none").attr("stroke", col).attr("stroke-width", vis.has(i) || upd.has(i) ? 2 : 1); F.g.append("text").attr("x", (x1 + x2) / 2).attr("y", y - 2).attr("text-anchor", "middle").attr("font-size", 8.5).attr("fill", col).text(`t[${i}]`); }
      AL.row(F.g, [null].concat(f.a), { x: -step, y: by, w: cw, gap: step - cw, h: 26, label: "a", mark: (j) => j === 0 ? "#111" : null });
      AL.row(F.g, f.t, { x: -step, y: by + 60, w: cw, gap: step - cw, h: 26, label: "t", mark: (j) => j === 0 ? "#111" : (f.marks.cur === j ? AC.a2 : vis.has(j) ? "#3b3a2a" : upd.has(j) ? "#1f3a2a" : null) });
      F.g.append("text").attr("x", -step + cw / 2).attr("y", by + 60 + 40).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", AC.muted).text("unused");
      BD.lines(F.g, -34, by + 130, [f.running !== undefined ? `running sum ${f.running}` : f.result !== undefined ? `result ${f.result} = plain ${f.plain} ${ok(f.result === f.plain)}` : ``, `lowbit(i) = i & (−i); query walks i −= lowbit(i), update walks i += lowbit(i)`, `entries re-verified against their definition: ${ok(f.ok)}`], { hi: [0, 2], size: 10.5, lh: 16 });
      txt("fen-readout", `Step ${i + 1}: ${f.title}. Tree array: ${f.t.slice(1).join(" ")}. Entries verified: ${ok(f.ok)}.`);
    }
    ["fen-i", "fen-u"].forEach(id => on(id, "input", setup)); setup();
  });

  /* ── F25 augmentation under rotations: size + max on an AVL interval tree */
  fig("aug-svg", () => {
    const IV = [[5, 8], [15, 18], [25, 30], [10, 12], [17, 19], [7, 10], [20, 22], [26, 26], [3, 4]];
    let root = null; const log = []; let allOk = true;
    const findByKey = (r, k) => { let x = r; while (x && x.key !== k) x = k < x.key ? x.left : x.right; return x; };
    IV.forEach(([lo, hi]) => {
      const c = AL.counter(); let before = null;
      root = BT.augInsert(root, lo, hi, c, e => {
        if (e.ev === "case") { const x = findByKey(root, e.at); const ch = e.kind[0] === "L" ? x.left : x.right; before = { at: e.at, kind: e.kind, xs: x.size, xm: x.max, cs: ch.size, cm: ch.max, child: ch.key }; }
      });
      if (before) { const top = findByKey(root, before.kind === "LL" || before.kind === "RR" ? before.child : (before.kind === "LR" ? findByKey(root, before.child) && null : null)) ; }
      const ok1 = BT.checkAug(root) && BT.checkAVL(root).ok; allOk = allOk && ok1;
      if (before) {
        // after the restructure, the node that rose is the new root of that subtree; find the old node x and its new fields
        const x = findByKey(root, before.at);
        log.push(`insert [${lo},${hi}]: case ${before.kind} at ${before.at} — before: size ${before.xs} max ${before.xm}; after: size ${x.size} max ${x.max}; ${c.get("rot")} rotation(s); fields ${ok(ok1)}`);
      }
    });
    const sortedLows = IV.map(v => v[0]).sort((a, b) => a - b);
    function draw() {
      const q = val("aug-q") || "ov-11-14";
      const c = AL.counter(); const path = []; let answer = null, title = "";
      if (q.startsWith("ov")) {
        const [, a, b] = q.split("-").map(Number); let x = root;
        while (x && (x.hi < a || b < x.lo)) { path.push(x); c.add("visits"); x = (x.left && x.left.max >= a) ? x.left : x.right; }
        if (x) { path.push(x); c.add("visits"); answer = x; }
        const plain = IV.filter(v => !(v[1] < a || b < v[0]));
        title = `overlap search [${a}, ${b}]: ${answer ? "found [" + answer.lo + "," + answer.hi + "]" : "none"} after ${c.get("visits")} visits — plain scan finds ${plain.length} overlapping interval(s): ${answer ? (plain.length > 0 ? "✓" : "✗") : (plain.length === 0 ? "✓" : "✗")}`;
      } else {
        const i = +q.split("-")[1]; let x = root, k = i;
        while (x) { path.push(x); c.add("visits"); const r = (x.left ? x.left.size : 0) + 1; if (k === r) { answer = x; break; } if (k < r) x = x.left; else { k -= r; x = x.right; } }
        title = `select(${i}): [${answer.lo},${answer.hi}] after ${c.get("visits")} visits — sorted lows ${sortedLows.join(" ")}: position ${i} is ${sortedLows[i - 1]} ${ok(sortedLows[i - 1] === answer.lo)}`;
      }
      const pset = new Set(path);
      const F = AL.frame("#aug-svg", 760, 340, { l: 6, r: 6, t: 22, b: 6 });
      F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.a2).text(title.length > 118 ? title.slice(0, 118) + "…" : title);
      BD.drawBin(F.g, root, { x: 0, y: 4, w: 460, levelH: 66, r: 20, fontSize: 9.5, text: nd => `${nd.lo},${nd.hi}`, fill: nd => nd === answer ? AC.good : pset.has(nd) ? "#3b3a2a" : null, stroke: nd => pset.has(nd) ? AC.a2 : null, textFill: nd => nd === answer ? AC.bg : null, label: nd => `size ${nd.size} · max ${nd.max}`, edge: (ch, p) => pset.has(ch) && pset.has(p) ? AC.a2 : null, edgeW: (ch, p) => pset.has(ch) && pset.has(p) ? 3 : null });
      BD.lines(F.g, 470, 8, [`rotations during the build (${log.length} restructurings):`].concat(log.map(l => l.replace(/^insert /, "").slice(0, 52))).concat([``, `all size/max fields re-verified from`, `scratch after every insertion: ${ok(allOk)}`, ``, `each node: [low, high] · size · max high`]), { hi: [0, log.length + 3], size: 9.5, lh: 14.5 });
      txt("aug-readout", `${title}. Build: ${log.length} restructurings — ${log.join("; ")}. All fields verified after every insertion: ${ok(allOk)}.`);
    }
    on("aug-q", "change", draw); draw();
  });

  /* ── F26 treap heights vs random BSTs ─────────────────────────────────── */
  fig("treap-svg", () => {
    const n = 1024, T = 100; const ht = [], hb = []; let rotI = 0, rotD = 0, okAll = true;
    for (let t = 0; t < T; t++) {
      const rr = AL.rng(t + 1); let tp = null; const ci = AL.counter();
      for (let k = 1; k <= n; k++) tp = BT.treapInsert(tp, k, rr(), ci);
      okAll = okAll && BT.checkTreap(tp); ht.push(BT.heightOf(tp)); rotI += ci.get("rot");
      if (t < 10) { const cd = AL.counter(); for (const k of AL.shuffle(Array.from({ length: n }, (_, i) => i + 1), rr)) tp = BT.treapDelete(tp, k, cd); rotD += cd.get("rot") / n; }
      let b = null; for (const k of AL.perm(n, AL.rng(t + 500))) b = BT.bstInsertPlain(b, k); hb.push(BT.heightOf(b));
    }
    const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
    const lo = Math.min(...ht, ...hb), hi = Math.max(...ht, ...hb);
    const hist = a => { const h = {}; a.forEach(v => h[v] = (h[v] || 0) + 1); return h; }; const H1 = hist(ht), H2 = hist(hb);
    const F = AL.frame("#treap-svg", 760, 300, { l: 44, r: 240, t: 16, b: 34 });
    const x = d3.scaleBand().domain(d3.range(lo, hi + 1)).range([0, F.iw]).padding(0.1), x1 = d3.scaleBand().domain([0, 1]).range([0, x.bandwidth()]);
    const y = d3.scaleLinear().domain([0, Math.max(...Object.values(H1), ...Object.values(H2)) * 1.15]).range([F.ih, 0]);
    AL.gridY(F.g, y, F.iw, 4); F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x)); AL.axisL(F.g, y, 4, `trees (of ${T})`);
    F.g.append("text").attr("x", F.iw).attr("y", F.ih + 30).attr("text-anchor", "end").attr("font-size", 11).attr("fill", AC.muted).text("height");
    x.domain().forEach(h => { [[H1[h] || 0, AC.rose, 0], [H2[h] || 0, AC.bad, 1]].forEach(([v, col, k]) => { F.g.append("rect").attr("x", x(h) + x1(k)).attr("y", y(v)).attr("width", x1.bandwidth()).attr("height", F.ih - y(v)).attr("fill", col).attr("opacity", 0.85); }); });
    AL.legend(F.g, [{ label: "treaps, keys 1…1024 inserted IN ORDER", color: AC.rose }, { label: "plain BSTs, random permutations", color: AC.bad }], 6, 12);
    BD.lines(F.g, F.iw + 12, 6, [`n = ${n}, ${T} seeds each`, ``, `treap height: mean ${f2(mean(ht))}`, `  min ${Math.min(...ht)}, max ${Math.max(...ht)}`, `plain BST height: mean ${f2(mean(hb))}`, `  min ${Math.min(...hb)}, max ${Math.max(...hb)}`, ``, `treap rotations per insert ${f2(rotI / (T * n))}`, `treap rotations per delete ${f2(rotD / 10)}`, `  (deletes measured on 10 trees)`, ``, `4.31·ln n = ${f1(4.31 * Math.log(n))} (asymptotic)`, `floor(log₂ n) = ${Math.floor(Math.log2(n))}`, ``, `key + heap order on all ${T}: ${ok(okAll)}`], { hi: [2, 3, 4, 5, 7, 8, 14], size: 10.5, lh: 15.5 });
    txt("treap-readout", `Treaps from sorted input (n = ${n}, ${T} seeds): height mean ${f2(mean(ht))}, min ${Math.min(...ht)}, max ${Math.max(...ht)}; plain BSTs from random permutations: mean ${f2(mean(hb))}, min ${Math.min(...hb)}, max ${Math.max(...hb)}. Rotations per treap insertion ${f3(rotI / (T * n))}, per deletion ${f3(rotD / 10)}. Both invariants verified on every treap: ${ok(okAll)}.`);
  });

  /* ── F27 one workload, six structures ─────────────────────────────────── */
  fig("cmp-svg", () => {
    const n = 4096, r = AL.rng(42), keys = AL.perm(n, r), searches = AL.sample(keys, n, r), dels = AL.shuffle(keys, r);
    const per = (c, k) => c.get(k) / n;
    const rows = [];
    { let root = null; const ci = AL.counter(); for (const k of keys) root = BT.avlInsert(root, k, ci); const okk = BT.checkAVL(root).ok; const cs = AL.counter(); for (const k of searches) BT.avlSearch(root, k, cs); const cd = AL.counter(); for (const k of dels) root = BT.avlDelete(root, k, cd); rows.push({ name: "AVL", scmp: per(cs, "cmp"), svis: per(cs, "visits"), icmp: per(ci, "cmp"), istruct: per(ci, "rot"), dstruct: per(cd, "rot"), unit: "rotations", ok: okk }); }
    { const T = BT.RB(); const ci = AL.counter(); for (const k of keys) BT.rbInsert(T, k, ci); const okk = BT.checkRB(T).ok; const cs = AL.counter(); for (const k of searches) BT.rbFind(T, k, cs); const cd = AL.counter(); for (const k of dels) BT.rbDelete(T, k, cd); rows.push({ name: "red-black", scmp: per(cs, "cmp"), svis: per(cs, "visits"), icmp: per(ci, "cmp"), istruct: per(ci, "rot"), dstruct: per(cd, "rot"), unit: "rotations", ok: okk }); }
    for (const t of [2, 8]) { const T = BT.BTree(t); const ci = AL.counter(); for (const k of keys) BT.btInsert(T, k, ci); const okk = BT.checkBT(T).ok; const cs = AL.counter(); for (const k of searches) BT.btSearch(T, k, cs); const cd = AL.counter(); for (const k of dels) BT.btDelete(T, k, cd); rows.push({ name: `B-tree t=${t}`, scmp: per(cs, "cmp"), svis: per(cs, "reads"), icmp: per(ci, "cmp"), istruct: per(ci, "splits"), dstruct: per(cd, "merges") + per(cd, "transfers"), unit: "splits / merges+transfers", ok: okk }); }
    { const S = BT.SL(0.25, AL.rng(7)); const ci = AL.counter(); for (const k of keys) BT.slInsert(S, k, ci); const okk = BT.checkSL(S).ok; const cs = AL.counter(); for (const k of searches) BT.slSearch(S, k, cs); const cd = AL.counter(); for (const k of dels) BT.slDelete(S, k, cd); rows.push({ name: "skip list p=¼", scmp: per(cs, "cmp"), svis: per(cs, "follows"), icmp: per(ci, "cmp"), istruct: per(ci, "links"), dstruct: per(cd, "links"), unit: "pointer splices", ok: okk }); }
    { const S = BT.SP(); const ci = AL.counter(); for (const k of keys) BT.spInsert(S, k, ci); const okk = BT.checkSP(S); const cs = AL.counter(); for (const k of searches) BT.spSearch(S, k, cs); const cd = AL.counter(); for (const k of dels) BT.spDelete(S, k, cd); rows.push({ name: "splay", scmp: per(cs, "cmp"), svis: per(cs, "visits"), icmp: per(ci, "cmp"), istruct: per(ci, "rot"), dstruct: per(cd, "rot"), srot: per(cs, "rot"), unit: "rotations", ok: okk }); }
    { let root = null; const rr = AL.rng(9); const ci = AL.counter(); for (const k of keys) root = BT.treapInsert(root, k, rr(), ci); const okk = BT.checkTreap(root); const cs = AL.counter(); for (const k of searches) BT.avlSearch(root, k, cs); const cd = AL.counter(); for (const k of dels) root = BT.treapDelete(root, k, cd); rows.push({ name: "treap", scmp: per(cs, "cmp"), svis: per(cs, "visits"), icmp: per(ci, "cmp"), istruct: per(ci, "rot"), dstruct: per(cd, "rot"), unit: "rotations", ok: okk }); }
    const labels = { scmp: "comparisons per search", svis: "node visits / block reads / pointer follows per search", icmp: "comparisons per insertion", istruct: "structural work per insertion", dstruct: "structural work per deletion" };
    function draw() {
      const m = val("cmp-metric") || "scmp";
      const F = AL.frame("#cmp-svg", 760, 300, { l: 44, r: 250, t: 16, b: 50 });
      const x = d3.scaleBand().domain(rows.map(r => r.name)).range([0, F.iw]).padding(0.25), y = d3.scaleLinear().domain([0, Math.max(...rows.map(r => r[m])) * 1.2 || 1]).range([F.ih, 0]);
      AL.gridY(F.g, y, F.iw, 4); F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x)).selectAll("text").attr("transform", "rotate(-18)").attr("text-anchor", "end"); AL.axisL(F.g, y, 4, labels[m]);
      rows.forEach((r, i) => { F.g.append("rect").attr("x", x(r.name)).attr("y", y(r[m])).attr("width", x.bandwidth()).attr("height", F.ih - y(r[m])).attr("fill", [AC.accent, AC.a2, AC.good, AC.teal, AC.violet, AC.rose, AC.bad][i]); F.g.append("text").attr("x", x(r.name) + x.bandwidth() / 2).attr("y", y(r[m]) - 4).attr("text-anchor", "middle").attr("font-size", 10).attr("fill", AC.ink).text(f2(r[m])); });
      BD.lines(F.g, F.iw + 10, 4, [`n = 4096 random inserts, searches, deletes`, ``].concat(rows.map(r => `${r.name}: ${f1(r.scmp)} cmp, ${f1(r.svis)} visits /search; ${f2(r.istruct)} ${r.unit.split(" ")[0]}/ins${r.srot !== undefined ? `; ${f1(r.srot)} rot/search` : ""}`)).concat([``, `structural = rotations (binary trees),`, `splits / merges (B-trees), splices (skip)`, `invariants after build: ${ok(rows.every(r => r.ok))}`]), { hi: [2, 3, 4, 5, 6, 7, 8], size: 9.5, lh: 14.5 });
      txt("cmp-readout", `${labels[m]}: ` + rows.map(r => `${r.name} ${f2(r[m])}`).join(", ") + `. Full table — ` + rows.map(r => `${r.name}: ${f2(r.scmp)} cmp/search, ${f2(r.svis)} visits/search, ${f2(r.icmp)} cmp/insert, ${f2(r.istruct)} ${r.unit.split(" ")[0]} per insert, ${f2(r.dstruct)} per delete${r.srot !== undefined ? `, ${f2(r.srot)} rotations per search` : ""}`).join("; ") + `. Invariants after build: ${ok(rows.every(r => r.ok))}.`);
    }
    on("cmp-metric", "change", draw); draw();
  });

})();
/* END-FIGURES */
