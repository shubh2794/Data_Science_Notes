/* sorting-searching.viz.js — figures for dsa/algorithms/sorting-searching.html
   (part 3 of the Algorithm Design & Analysis series).

   Loaded after ../../data.js, ../../notes.js and ../dsa-viz.js, so the globals
   AC (palette) and AL (rng/frame/axisB/axisL/gridY/row/bars/binTree/counter/
   stepper/…) are available.

   House rule obeyed throughout: every count this page DISPLAYS — comparisons,
   element moves, swaps, passes, recursion depth, probes — is produced by running
   the real routine under AL.counter() and reading the counter back. Nothing
   below is a constant typed into a caption. Wherever a closed form exists it is
   evaluated independently (a recurrence solved numerically, an exact sum, an
   exhaustive enumeration) and printed next to the measurement with an
   agree / DISAGREE flag, so each figure is its own cross-check.

   Layout of this file:
     SS  — the instrumented routines, pure and DOM-free. They take a counter with
           the AL.counter() interface, and an optional frame recorder, and are also
           loadable from Node (module.exports at the bottom) so the numbers quoted
           in the page's prose were reproduced offline from THIS code.
     figures — one block per <svg>, in page order:
       01 #es-svg   elementary sorts animated as bars, counters live
       02 #iv-svg   insertion sort cost vs number of inversions, measured
       03 #sh-svg   shellsort comparisons vs n for four gap sequences, fitted exponents
       04 #dt-svg   the decision tree of insertion sort for n = 3, plus log₂(n!) exact vs Stirling
       05 #mr-svg   one merge stepped, with a stability tie visible
       06 #mc-svg   mergesort comparisons vs n against the exact worst-case formula
       07 #pt-svg   Lomuto vs Hoare partition stepped on the same array
       08 #tw-svg   three-way partition stepped; Lomuto/Hoare/three-way on duplicate-heavy input
       09 #pv-svg   pivot strategies compared on adversarial and random input
       10 #qc-svg   quicksort comparisons vs the exact average recurrence and 2n·ln n
       11 #qd-svg   recursion-depth distribution over random pivots, with and without smaller-first
       12 #hh-svg   mergesort / quicksort / heapsort head to head: comparisons and moves per n·log₂n
       13 #cs-svg   counting sort stepped, prefix-sum array and the backwards pass
       14 #rx-svg   LSD radix sort stepped by digit
       15 #bk-svg   bucket sort: occupancy and measured cost on uniform vs skewed input
       16 #bs-svg   binary search stepped, lo/mid/hi and the invariant
       17 #lb-svg   lower_bound / upper_bound stepped on an array with duplicates
       18 #ls-svg   linear vs binary vs interpolation search probes across n */

/* ═══════════════════════════════════════════════════════════════════════════
   SS — the instrumented routines
   ═══════════════════════════════════════════════════════════════════════════
   Conventions:
     c.add("cmp")  one key comparison (a[i] vs a[j], or a[i] vs pivot / target)
     c.add("mov")  one element WRITE into an array slot or buffer slot
     c.add("swp")  one swap (also counted as 3 moves by the swap helper)
     c.add("pass") one outer pass (bubble, shell, LSD radix)
     c.add("probe") one array inspection in a search
   Ranges are HALF-OPEN [lo, hi) everywhere in this file. */
const SS = (function () {
  const nop = { add: function () {}, get: function () { return 0; } };
  function swap(a, i, j, c) {
    if (i === j) return;
    const t = a[i]; a[i] = a[j]; a[j] = t;
    c.add("swp"); c.add("mov", 3);
  }
  function key(v) { return (v !== null && typeof v === "object") ? v.k : v; }

  /* ── elementary sorts ────────────────────────────────────────────────── */

  /* Insertion sort, linear scan. Invariant: a[0..i) is sorted and is a
     permutation of the original a[0..i). Stable because the scan stops on
     equality (strict >). */
  function insertionSort(a, c, rec) {
    c = c || nop;
    const n = a.length;
    if (rec) rec({ a: a.slice(), i: 1, j: 1, phase: "start" });
    for (let i = 1; i < n; i++) {
      const v = a[i];
      let j = i;
      c.add("mov");                                  // v = a[i]
      while (j > 0) {
        c.add("cmp");
        if (key(a[j - 1]) > key(v)) { a[j] = a[j - 1]; c.add("mov"); j--; }
        else break;
        if (rec) rec({ a: a.slice(), i, j, phase: "shift" });
      }
      a[j] = v; c.add("mov");
      if (rec) rec({ a: a.slice(), i, j, phase: "place" });
    }
    return a;
  }

  /* Binary insertion sort: finds the insertion point with an upper_bound
     search (so equal keys go AFTER their equals — stable), then shifts.
     Saves comparisons, not moves. */
  function binaryInsertionSort(a, c, rec) {
    c = c || nop;
    const n = a.length;
    for (let i = 1; i < n; i++) {
      const v = a[i]; c.add("mov");
      let lo = 0, hi = i;
      while (lo < hi) {
        const mid = lo + ((hi - lo) >> 1);
        c.add("cmp");
        if (key(a[mid]) <= key(v)) lo = mid + 1; else hi = mid;
      }
      for (let j = i; j > lo; j--) { a[j] = a[j - 1]; c.add("mov"); }
      a[lo] = v; c.add("mov");
      if (rec) rec({ a: a.slice(), i, j: lo, phase: "place" });
    }
    return a;
  }

  /* Selection sort. Always n(n−1)/2 comparisons; at most n−1 swaps.
     NOT stable: the swap can carry an element past an equal one. */
  function selectionSort(a, c, rec) {
    c = c || nop;
    const n = a.length;
    for (let i = 0; i < n - 1; i++) {
      let m = i;
      for (let j = i + 1; j < n; j++) {
        c.add("cmp");
        if (key(a[j]) < key(a[m])) m = j;
        if (rec) rec({ a: a.slice(), i, j, m, phase: "scan" });
      }
      swap(a, i, m, c);
      if (rec) rec({ a: a.slice(), i, j: m, m, phase: "swap" });
    }
    return a;
  }

  /* Bubble sort with the early-exit flag. Stable (strict >). */
  function bubbleSort(a, c, rec) {
    c = c || nop;
    const n = a.length;
    for (let pass = 0; pass < n - 1; pass++) {
      c.add("pass");
      let swapped = false;
      for (let j = 0; j < n - 1 - pass; j++) {
        c.add("cmp");
        if (key(a[j]) > key(a[j + 1])) { swap(a, j, j + 1, c); swapped = true; }
        if (rec) rec({ a: a.slice(), i: n - 1 - pass, j, phase: "cmp" });
      }
      if (!swapped) break;
    }
    return a;
  }

  /* Shellsort with a supplied gap sequence (largest first, must end in 1). */
  const GAPS = {
    shell: n => { const g = []; for (let h = Math.floor(n / 2); h >= 1; h = Math.floor(h / 2)) g.push(h); return g.length ? g : [1]; },
    hibbard: n => { const g = []; for (let k = 1; (1 << k) - 1 < n; k++) g.push((1 << k) - 1); return g.reverse().length ? g : [1]; },
    knuth: n => { const g = []; for (let h = 1; h < Math.ceil(n / 3); h = 3 * h + 1) g.push(h); if (!g.length) g.push(1); return g.reverse(); },
    ciura: n => { const base = [1, 4, 10, 23, 57, 132, 301, 701]; const g = base.filter(h => h < n); let h = 701; while (true) { h = Math.floor(h * 2.25); if (h >= n) break; g.push(h); } return g.length ? g.reverse() : [1]; }
  };
  function shellSort(a, c, gaps, rec) {
    c = c || nop;
    const n = a.length;
    const seq = (typeof gaps === "function") ? gaps(n) : (gaps || GAPS.knuth(n));
    for (const h of seq) {
      c.add("pass");
      for (let i = h; i < n; i++) {
        const v = a[i]; c.add("mov");
        let j = i;
        while (j >= h) {
          c.add("cmp");
          if (key(a[j - h]) > key(v)) { a[j] = a[j - h]; c.add("mov"); j -= h; }
          else break;
        }
        a[j] = v; c.add("mov");
      }
      if (rec) rec({ a: a.slice(), h, phase: "gap" });
    }
    return a;
  }

  /* ── mergesort family ────────────────────────────────────────────────── */

  /* merge a[lo..mid) with a[mid..hi) through the buffer. Ties go LEFT (≤),
     which is what makes the sort stable. Comparisons: m − t where t is the
     length of the uncompared tail. Moves: m into the buffer + m back = 2m. */
  function merge(a, lo, mid, hi, buf, c, rec) {
    for (let k = lo; k < hi; k++) { buf[k] = a[k]; c.add("mov"); }
    let i = lo, j = mid, k = lo;
    while (i < mid && j < hi) {
      c.add("cmp");
      if (key(buf[i]) <= key(buf[j])) { a[k] = buf[i++]; }
      else { a[k] = buf[j++]; }
      c.add("mov"); k++;
      if (rec) rec({ a: a.slice(), lo, mid, hi, i, j, k, phase: "emit" });
    }
    while (i < mid) { a[k++] = buf[i++]; c.add("mov"); if (rec) rec({ a: a.slice(), lo, mid, hi, i, j, k, phase: "copyL" }); }
    while (j < hi) { a[k++] = buf[j++]; c.add("mov"); if (rec) rec({ a: a.slice(), lo, mid, hi, i, j, k, phase: "copyR" }); }
  }
  function mergeSort(a, c, rec) {
    c = c || nop;
    const buf = new Array(a.length);
    let depth = 0, maxDepth = 0;
    (function rec_(lo, hi) {
      depth++; if (depth > maxDepth) maxDepth = depth;
      if (hi - lo > 1) {
        const mid = lo + ((hi - lo) >> 1);
        rec_(lo, mid); rec_(mid, hi);
        merge(a, lo, mid, hi, buf, c, rec);
        if (rec) rec({ a: a.slice(), lo, mid, hi, phase: "merged" });
      }
      depth--;
    })(0, a.length);
    c.add("depth", maxDepth);
    return a;
  }
  /* Bottom-up: widths 1, 2, 4, …; the last run of a pass may be short. */
  function mergeSortBottomUp(a, c, rec) {
    c = c || nop;
    const n = a.length, buf = new Array(n);
    for (let w = 1; w < n; w *= 2) {
      c.add("pass");
      for (let lo = 0; lo + w < n; lo += 2 * w) {
        const mid = lo + w, hi = Math.min(lo + 2 * w, n);
        merge(a, lo, mid, hi, buf, c, rec);
      }
      if (rec) rec({ a: a.slice(), w, phase: "pass" });
    }
    return a;
  }
  /* Natural: find the runs already present, merge neighbouring runs until one. */
  function mergeSortNatural(a, c, rec) {
    c = c || nop;
    const n = a.length, buf = new Array(n);
    if (n < 2) return a;
    let runs = [];
    let s = 0;
    for (let i = 1; i <= n; i++) {
      if (i === n) { runs.push([s, n]); break; }
      c.add("cmp");
      if (key(a[i]) < key(a[i - 1])) { runs.push([s, i]); s = i; }
    }
    c.add("runs", runs.length);
    while (runs.length > 1) {
      c.add("pass");
      const next = [];
      for (let r = 0; r + 1 < runs.length; r += 2) {
        merge(a, runs[r][0], runs[r][1], runs[r + 1][1], buf, c, rec);
        next.push([runs[r][0], runs[r + 1][1]]);
      }
      if (runs.length % 2 === 1) next.push(runs[runs.length - 1]);
      runs = next;
      if (rec) rec({ a: a.slice(), runs: runs.slice(), phase: "pass" });
    }
    return a;
  }
  /* The exact worst-case comparison count of top-down mergesort, from the
     recurrence W(n) = W(floor(n/2)) + W(ceil(n/2)) + n − 1 — an independent
     check of the closed form n·ceil(log₂n) − 2^ceil(log₂n) + 1. */
  function mergeWorstRec(n, memo) {
    memo = memo || {};
    if (n <= 1) return 0;
    if (memo[n] !== undefined) return memo[n];
    const f = Math.floor(n / 2);
    const v = mergeWorstRec(f, memo) + mergeWorstRec(n - f, memo) + n - 1;
    memo[n] = v; return v;
  }
  function mergeWorstClosed(n) {
    if (n <= 1) return 0;
    const k = Math.ceil(Math.log2(n));
    return n * k - Math.pow(2, k) + 1;
  }
  /* An input that forces every merge of top-down mergesort to its maximum
     (m − 1 comparisons): un-merge a sorted array by dealing alternately. */
  function mergeAdversary(n) {
    const sorted = Array.from({ length: n }, (_, i) => i + 1);
    function build(s) {
      const m = s.length;
      if (m <= 1) return s;
      const f = Math.floor(m / 2);
      // left half gets f elements, right half m − f; interleave so that the
      // final two elements of the merge come from different sides.
      const left = [], right = [];
      for (let i = 0; i < m; i++) {
        // right needs m − f = ceil; give right the even positions (has the max)
        if (i % 2 === 0) right.push(s[i]); else left.push(s[i]);
      }
      // right has ceil(m/2), left floor(m/2): matches mid = lo + floor(m/2)
      return build(left).concat(build(right));
    }
    return build(sorted);
  }

  /* ── quicksort family ────────────────────────────────────────────────── */

  /* Lomuto partition of a[lo..hi) around pivot a[hi−1]. Returns the pivot's
     final index. Exactly (hi − lo − 1) comparisons. Invariant:
       a[lo..i) ≤ p,  a[i..j) > p,  a[j..hi−1) unexamined,  a[hi−1] = p. */
  function lomuto(a, lo, hi, c, rec) {
    const p = key(a[hi - 1]);
    let i = lo;
    if (rec) rec({ a: a.slice(), lo, hi, i, j: lo, pivot: hi - 1, phase: "start" });
    for (let j = lo; j < hi - 1; j++) {
      c.add("cmp");
      if (key(a[j]) <= p) { swap(a, i, j, c); i++; }
      if (rec) rec({ a: a.slice(), lo, hi, i, j: j + 1, pivot: hi - 1, phase: "scan" });
    }
    swap(a, i, hi - 1, c);
    if (rec) rec({ a: a.slice(), lo, hi, i, j: hi - 1, pivot: i, phase: "final" });
    return i;
  }
  /* Hoare partition of a[lo..hi) around pivot value a[lo]. Returns j such that
     a[lo..j] ≤ p ≤ a[j+1..hi) — the pivot is NOT necessarily at j. Both
     scans stop on keys equal to the pivot. */
  function hoare(a, lo, hi, c, rec) {
    const p = key(a[lo]);
    let i = lo - 1, j = hi;
    if (rec) rec({ a: a.slice(), lo, hi, i, j, pivotVal: p, phase: "start" });
    while (true) {
      do { j--; c.add("cmp"); } while (key(a[j]) > p);
      do { i++; c.add("cmp"); } while (key(a[i]) < p);
      if (i < j) { swap(a, i, j, c); if (rec) rec({ a: a.slice(), lo, hi, i, j, pivotVal: p, phase: "swap" }); }
      else { if (rec) rec({ a: a.slice(), lo, hi, i, j, pivotVal: p, phase: "cross" }); return j; }
    }
  }
  /* Three-way (Dutch-flag) partition of a[lo..hi) around pivot value a[lo].
     Returns [lt, gt): a[lo..lt) < p, a[lt..gt) = p, a[gt..hi) > p. */
  function threeWay(a, lo, hi, c, rec) {
    const p = key(a[lo]);
    let lt = lo, i = lo + 1, gt = hi;
    if (rec) rec({ a: a.slice(), lo, hi, lt, i, gt, pivotVal: p, phase: "start" });
    while (i < gt) {
      c.add("cmp");
      const v = key(a[i]);
      if (v < p) { swap(a, lt, i, c); lt++; i++; }
      else { c.add("cmp"); if (v > p) { gt--; swap(a, i, gt, c); } else i++; }
      if (rec) rec({ a: a.slice(), lo, hi, lt, i, gt, pivotVal: p, phase: "scan" });
    }
    return [lt, gt];
  }
  /* Pivot selection: moves the chosen pivot into the slot the scheme reads.
     Lomuto reads a[hi−1]; Hoare and three-way read a[lo]. */
  function choosePivot(a, lo, hi, how, rng, c) {
    const n = hi - lo;
    let idx;
    if (how === "first") idx = lo;
    else if (how === "last") idx = hi - 1;
    else if (how === "middle") idx = lo + (n >> 1);
    else if (how === "random") idx = lo + Math.floor(rng() * n);
    else if (how === "median3") idx = median3(a, lo, lo + (n >> 1), hi - 1, c);
    else if (how === "ninther") {
      if (n < 9) idx = median3(a, lo, lo + (n >> 1), hi - 1, c);
      else {
        const s = Math.floor(n / 8);
        const m1 = median3(a, lo, lo + s, lo + 2 * s, c);
        const m2 = median3(a, lo + (n >> 1) - s, lo + (n >> 1), lo + (n >> 1) + s, c);
        const m3 = median3(a, hi - 1 - 2 * s, hi - 1 - s, hi - 1, c);
        idx = median3(a, m1, m2, m3, c);
      }
    } else idx = hi - 1;
    return idx;
  }
  function median3(a, i, j, k, c) {           // index of the median, 2–3 comparisons
    const x = key(a[i]), y = key(a[j]), z = key(a[k]);
    c.add("cmp");
    if (x < y) {
      c.add("cmp");
      if (y < z) return j;
      c.add("cmp");
      return (x < z) ? k : i;
    } else {
      c.add("cmp");
      if (x < z) return i;
      c.add("cmp");
      return (y < z) ? k : j;
    }
  }
  /* Quicksort with every knob: scheme lomuto|hoare|threeway, pivot rule,
     small-subarray cutoff (0 = none; the leftover is insertion-sorted at the
     end), smallerFirst (recurse on the smaller side, loop on the larger, to
     bound the stack), and a depth limit for introsort (falls back to heapsort). */
  function quickSort(a, c, opt) {
    c = c || nop;
    const o = Object.assign({ scheme: "lomuto", pivot: "last", cutoff: 0, smallerFirst: false,
                              seed: 1, depthLimit: 0, rec: null }, opt || {});
    const rng = AL_rng(o.seed);
    let maxDepth = 0, partitions = 0, heapFallbacks = 0;
    function sortRange(lo, hi, depth) {
      while (hi - lo > Math.max(1, o.cutoff)) {
        if (depth > maxDepth) maxDepth = depth;
        if (o.depthLimit && depth > o.depthLimit) { heapFallbacks++; heapSortRange(a, lo, hi, c); return; }
        partitions++;
        let l, r;                                   // the two sub-ranges [lo,l) and [r,hi)
        if (o.scheme === "lomuto") {
          const pi = choosePivot(a, lo, hi, o.pivot, rng, c);
          swap(a, pi, hi - 1, c);
          const q = lomuto(a, lo, hi, c, o.rec);
          l = q; r = q + 1;
        } else if (o.scheme === "hoare") {
          const pi = choosePivot(a, lo, hi, o.pivot, rng, c);
          swap(a, pi, lo, c);                       // Hoare reads its pivot VALUE from a[lo]
          const j = hoare(a, lo, hi, c, o.rec);
          l = j + 1; r = j + 1;
        } else {
          const pi = choosePivot(a, lo, hi, o.pivot, rng, c);
          swap(a, pi, lo, c);                       // three-way reads its pivot VALUE from a[lo]
          const lg = threeWay(a, lo, hi, c, o.rec);
          l = lg[0]; r = lg[1];
        }
        if (o.smallerFirst) {
          if (l - lo < hi - r) { sortRange(lo, l, depth + 1); lo = r; }
          else { sortRange(r, hi, depth + 1); hi = l; }
        } else {
          sortRange(lo, l, depth + 1);
          lo = r;                                     // tail position: loop on the right side
        }
      }
    }
    sortRange(0, a.length, 1);
    if (o.cutoff > 1) insertionSort(a, c);        // one final pass over the nearly-sorted array
    c.add("depth", maxDepth); c.add("partitions", partitions); c.add("heapFallbacks", heapFallbacks);
    return a;
  }
  /* mulberry32, duplicated here so SS has no dependency on AL when loaded in Node */
  function AL_rng(seed) {
    let s = (seed >>> 0) || 1;
    return function () {
      s = (s + 0x6D2B79F5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  /* The exact expected comparison count of quicksort with a uniformly random
     pivot on DISTINCT keys, from the recurrence
       C(n) = part(n) + (2/n)·∑_{i<n} C(i),   C(0) = C(1) = 0,
     solved numerically for a given per-partition cost part(n).
     Lomuto: part(n) = n − 1.  Hoare (this file's version): n + 1. */
  function quickExpected(nMax, part) {
    const C = new Float64Array(nMax + 1);
    let S = 0;                                   // ∑_{i<n} C(i)
    for (let n = 2; n <= nMax; n++) {
      S += C[n - 1];
      C[n] = part(n) + 2 * S / n;
    }
    return C;
  }
  function harmonic(n) { let h = 0; for (let k = 1; k <= n; k++) h += 1 / k; return h; }

  /* ── heapsort (classic, for the head-to-head table only) ─────────────── */
  function heapSortRange(a, lo, hi, c) {
    const n = hi - lo;
    const sift = (i, size) => {
      const v = a[lo + i];
      while (true) {
        let l = 2 * i + 1, r = l + 1, m = l;
        if (l >= size) break;
        if (r < size) { c.add("cmp"); if (key(a[lo + r]) > key(a[lo + l])) m = r; }
        c.add("cmp");
        if (key(a[lo + m]) > key(v)) { a[lo + i] = a[lo + m]; c.add("mov"); i = m; }
        else break;
      }
      a[lo + i] = v; c.add("mov");
    };
    for (let i = (n >> 1) - 1; i >= 0; i--) sift(i, n);
    for (let end = n - 1; end > 0; end--) { swap(a, lo, lo + end, c); sift(0, end); }
  }
  function heapSort(a, c) { c = c || nop; heapSortRange(a, 0, a.length, c); return a; }

  /* ── linear-time sorts ───────────────────────────────────────────────── */

  /* Counting sort on keys in 0..k−1; keyOf reads the digit. Stable because the
     placement pass runs BACKWARDS over the input. Returns a new array. */
  function countingSort(a, k, keyOf, c, rec) {
    c = c || nop;
    keyOf = keyOf || key;
    const n = a.length, cnt = new Array(k).fill(0), out = new Array(n);
    for (let j = 0; j < n; j++) { cnt[keyOf(a[j])]++; c.add("tally"); if (rec) rec({ phase: "count", j, cnt: cnt.slice(), out: out.slice() }); }
    for (let i = 1; i < k; i++) { cnt[i] += cnt[i - 1]; c.add("prefix"); if (rec) rec({ phase: "prefix", i, cnt: cnt.slice(), out: out.slice() }); }
    for (let j = n - 1; j >= 0; j--) {
      const d = keyOf(a[j]);
      cnt[d]--; out[cnt[d]] = a[j]; c.add("mov");
      if (rec) rec({ phase: "place", j, d, pos: cnt[d], cnt: cnt.slice(), out: out.slice() });
    }
    return out;
  }
  /* LSD radix sort on non-negative integers with `digits` base-`base` digits. */
  function radixLSD(a, digits, base, c, rec) {
    c = c || nop;
    let cur = a.slice();
    for (let d = 0; d < digits; d++) {
      c.add("pass");
      const div = Math.pow(base, d);
      cur = countingSort(cur, base, v => Math.floor(key(v) / div) % base, c);
      if (rec) rec({ phase: "digit", d, a: cur.slice() });
    }
    return cur;
  }
  /* MSD radix sort: bucket by the leading digit, recurse into each bucket. */
  function radixMSD(a, digits, base, c) {
    c = c || nop;
    function go(arr, d) {
      if (arr.length <= 1 || d < 0) return arr;
      const div = Math.pow(base, d);
      const buckets = Array.from({ length: base }, () => []);
      for (const v of arr) { buckets[Math.floor(key(v) / div) % base].push(v); c.add("mov"); }
      c.add("buckets", base);
      let out = [];
      for (const b of buckets) out = out.concat(go(b, d - 1));
      return out;
    }
    return go(a.slice(), digits - 1);
  }
  /* Bucket sort on reals in [0,1): n buckets, insertion sort inside each. */
  function bucketSort(a, c) {
    c = c || nop;
    const n = a.length, B = Array.from({ length: n }, () => []);
    for (const v of a) { B[Math.min(n - 1, Math.floor(v * n))].push(v); c.add("mov"); }
    const sizes = B.map(b => b.length);
    let out = [];
    for (const b of B) { insertionSort(b, c); out = out.concat(b); }
    return { out, sizes };
  }

  /* ── searching ───────────────────────────────────────────────────────── */
  function linearSearch(a, x, c) {
    c = c || nop;
    for (let i = 0; i < a.length; i++) { c.add("probe"); if (key(a[i]) === x) return i; }
    return -1;
  }
  /* Membership binary search on the half-open range [lo, hi). Returns an index
     holding x, or −1. One probe = one inspection of a[mid]. */
  function binarySearch(a, x, c, rec) {
    c = c || nop;
    let lo = 0, hi = a.length;
    while (lo < hi) {
      const mid = lo + ((hi - lo) >> 1);
      c.add("probe");
      if (rec) rec({ lo, hi, mid, v: key(a[mid]), phase: "probe" });
      if (key(a[mid]) < x) lo = mid + 1;
      else if (key(a[mid]) > x) hi = mid;
      else { if (rec) rec({ lo, hi, mid, v: key(a[mid]), phase: "found" }); return mid; }
    }
    if (rec) rec({ lo, hi, mid: -1, phase: "absent" });
    return -1;
  }
  /* first index i in [0, n] with a[i] ≥ x  (n if none) */
  function lowerBound(a, x, c, rec) {
    c = c || nop;
    let lo = 0, hi = a.length;
    while (lo < hi) {
      const mid = lo + ((hi - lo) >> 1);
      c.add("probe");
      const goRight = key(a[mid]) < x;
      if (rec) rec({ lo, hi, mid, v: key(a[mid]), goRight, phase: "probe" });
      if (goRight) lo = mid + 1; else hi = mid;
    }
    if (rec) rec({ lo, hi, mid: -1, phase: "done" });
    return lo;
  }
  /* first index i in [0, n] with a[i] > x  (n if none) */
  function upperBound(a, x, c, rec) {
    c = c || nop;
    let lo = 0, hi = a.length;
    while (lo < hi) {
      const mid = lo + ((hi - lo) >> 1);
      c.add("probe");
      const goRight = key(a[mid]) <= x;
      if (rec) rec({ lo, hi, mid, v: key(a[mid]), goRight, phase: "probe" });
      if (goRight) lo = mid + 1; else hi = mid;
    }
    if (rec) rec({ lo, hi, mid: -1, phase: "done" });
    return lo;
  }
  /* Exponential (galloping) search: find a bound by doubling, then binary
     search inside [b/2, min(b, n)). Cost O(log i) where i is the answer's index. */
  function exponentialSearch(a, x, c) {
    c = c || nop;
    const n = a.length;
    if (n === 0) return -1;
    let b = 1;
    c.add("probe");
    if (key(a[0]) === x) return 0;
    while (b < n) { c.add("probe"); if (key(a[b]) >= x) break; b *= 2; }
    const lo = b >> 1, hi = Math.min(b + 1, n);
    const sub = a.slice(lo, hi);
    const r = binarySearch(sub, x, c);
    return r < 0 ? -1 : lo + r;
  }
  /* Interpolation search. Expected O(log log n) probes on UNIFORM keys; can be
     Θ(n) on skewed keys. Half-open guard: keys must be numeric. */
  function interpolationSearch(a, x, c) {
    c = c || nop;
    let lo = 0, hi = a.length - 1;
    while (lo <= hi) {
      const alo = key(a[lo]), ahi = key(a[hi]);   // the end keys are kept in registers;
      if (x < alo || x > ahi) return -1;          // only the interpolated slot counts as a probe
      let pos;
      if (ahi === alo) pos = lo;
      else pos = lo + Math.floor((hi - lo) * (x - alo) / (ahi - alo));
      c.add("probe");
      const v = key(a[pos]);
      if (v === x) return pos;
      if (v < x) lo = pos + 1; else hi = pos - 1;
    }
    return -1;
  }

  /* ── small utilities the figures and the offline check share ─────────── */
  function inversions(a) {                       // O(n²) direct count — the independent way
    let k = 0;
    for (let i = 0; i < a.length; i++) for (let j = i + 1; j < a.length; j++) if (key(a[i]) > key(a[j])) k++;
    return k;
  }
  function isSorted(a) { for (let i = 1; i < a.length; i++) if (key(a[i - 1]) > key(a[i])) return false; return true; }
  function isStable(a) {                         // a = tagged records {k, tag}; tags increase within equal keys
    for (let i = 1; i < a.length; i++) if (a[i - 1].k === a[i].k && a[i - 1].tag > a[i].tag) return false;
    return true;
  }
  function log2Factorial(n) { let s = 0; for (let k = 2; k <= n; k++) s += Math.log2(k); return s; }
  function stirlingLog2(n) {                     // log₂(n!) ≈ n·log₂n − n·log₂e + ½·log₂(2πn)
    return n * Math.log2(n) - n * Math.LOG2E + 0.5 * Math.log2(2 * Math.PI * n);
  }
  function mkCounter() {                          // same interface as AL.counter(), for Node
    const n = Object.create(null);
    return { add: (k, by) => { n[k] = (n[k] || 0) + (by === undefined ? 1 : by); },
             get: k => n[k] || 0, all: () => Object.assign({}, n), reset: () => { for (const k in n) delete n[k]; } };
  }

  return {
    swap, key, insertionSort, binaryInsertionSort, selectionSort, bubbleSort, shellSort, GAPS,
    merge, mergeSort, mergeSortBottomUp, mergeSortNatural, mergeWorstRec, mergeWorstClosed, mergeAdversary,
    lomuto, hoare, threeWay, median3, choosePivot, quickSort, quickExpected, harmonic,
    heapSort, countingSort, radixLSD, radixMSD, bucketSort,
    linearSearch, binarySearch, lowerBound, upperBound, exponentialSearch, interpolationSearch,
    inversions, isSorted, isStable, log2Factorial, stirlingLog2, mkCounter, rng: AL_rng
  };
})();

if (typeof module !== "undefined" && module.exports) module.exports = SS;

/* ═══════════════════════════════════════════════════════════════════════════
   FIGURES — only when running in a page with d3 and the toolbox present
   ═══════════════════════════════════════════════════════════════════════════ */
if (typeof d3 !== "undefined" && typeof AL !== "undefined") (function () {

/* ── shared little helpers ─────────────────────────────────────────────── */
const SX = {
  table: function (sel, head, rows) {
    const h = d3.select(sel);
    if (h.empty()) return null;
    h.selectAll("*").remove();
    const t = h.append("table").attr("class", "cmp").style("margin", "10px 0 0");
    const hr = t.append("tr");
    head.forEach(c => hr.append("th").html(c));
    rows.forEach(r => { const tr = t.append("tr"); r.forEach(c => tr.append("td").html(c)); });
    return t;
  },
  int: d3.format(","),
  f1: d3.format(".1f"), f2: d3.format(".2f"), f3: d3.format(".3f"),
  flag: ok => ok ? '<span style="color:' + AC.good + '">✓ agree</span>' : '<span style="color:' + AC.bad + '">✗ DISAGREE</span>',
  clearControls: function (svgNode) {
    d3.select(svgNode.parentNode).selectAll('div[role="group"]').remove();
  },
  has: id => !d3.select("#" + id).empty(),
  on: (id, ev, fn) => { const s = d3.select("#" + id); if (!s.empty()) s.on(ev, fn); },
  val: (id, dflt) => { const s = d3.select("#" + id); return s.empty() ? dflt : s.property("value"); },
  checked: (id, dflt) => { const s = d3.select("#" + id); return s.empty() ? dflt : s.property("checked"); },
  setText: (id, t) => { const s = d3.select("#" + id); if (!s.empty()) s.text(t); },
  setHtml: (id, t) => { const s = d3.select("#" + id); if (!s.empty()) s.html(t); },
  inputs: {
    random: (n, seed) => AL.perm(n, AL.rng(seed || 11)),
    sorted: n => Array.from({ length: n }, (_, i) => i + 1),
    reversed: n => Array.from({ length: n }, (_, i) => n - i),
    nearly: (n, seed) => { const a = Array.from({ length: n }, (_, i) => i + 1); const r = AL.rng(seed || 5); for (let t = 0; t < Math.max(1, n >> 3); t++) { const i = Math.floor(r() * (n - 1)); const x = a[i]; a[i] = a[i + 1]; a[i + 1] = x; } return a; },
    fewUnique: (n, seed) => { const r = AL.rng(seed || 7); return Array.from({ length: n }, () => 1 + Math.floor(r() * 4)); },
    allEqual: n => Array.from({ length: n }, () => 5),
    organPipe: n => Array.from({ length: n }, (_, i) => i < n / 2 ? i + 1 : n - i)
  }
};

/* FIGURE BLOCKS ARE APPENDED BELOW, ONE AT A TIME */

/* ── 01  #es-svg  elementary sorts animated as bars, counters live ─────── */
(function () {
  if (!SX.has("es-svg")) return;
  const W = 680, H = 250;
  let stepper = null;
  function build() {
    const scheme = SX.val("es-scheme", "insertion"), input = SX.val("es-input", "random");
    const n = +SX.val("es-n", 24);
    const a0 = SX.inputs[input](n, 11);
    const c = AL.counter();
    const frames = [];
    const rec = f => frames.push(Object.assign(f, { cnt: c.all() }));
    const a = a0.slice();
    rec({ a: a.slice(), phase: "start" });
    if (scheme === "insertion") SS.insertionSort(a, c, rec);
    else if (scheme === "binary") SS.binaryInsertionSort(a, c, rec);
    else if (scheme === "selection") SS.selectionSort(a, c, rec);
    else if (scheme === "bubble") SS.bubbleSort(a, c, rec);
    else SS.shellSort(a, c, SS.GAPS.knuth, rec);
    rec({ a: a.slice(), phase: "done" });
    const tot = c.all();
    const inv = SS.inversions(a0);
    /* independent checks, per scheme */
    let check = "";
    if (scheme === "insertion") {
      /* comparisons = inversions + (n − 1) − z, z = insertions that ran to the front */
      let z = 0; const b = a0.slice();
      for (let i = 1; i < n; i++) { const v = b[i]; let j = i; while (j > 0 && b[j - 1] > v) { b[j] = b[j - 1]; j--; } b[j] = v; if (j === 0) z++; }
      const pred = inv + (n - 1) - z;
      check = `inversions I = ${inv} (counted directly); I + (n − 1) − z = ${inv} + ${n - 1} − ${z} = ${pred} ${SX.flag(pred === tot.cmp)} · moves = 2(n − 1) + I = ${2 * (n - 1) + inv} ${SX.flag(2 * (n - 1) + inv === tot.mov)}`;
    } else if (scheme === "selection") {
      check = `n(n − 1)/2 = ${n * (n - 1) / 2} ${SX.flag(n * (n - 1) / 2 === tot.cmp)} · swaps ≤ n − 1 = ${n - 1} ${SX.flag((tot.swp || 0) <= n - 1)}`;
    } else if (scheme === "bubble") {
      check = `swaps = inversions = ${inv} ${SX.flag(inv === (tot.swp || 0))} (each adjacent swap removes exactly one inversion) · passes ${tot.pass} ≤ n − 1`;
    } else if (scheme === "binary") {
      let ub = 0; for (let i = 1; i < n; i++) ub += Math.ceil(Math.log2(i + 1));
      check = `comparisons ≤ ∑ ceil(log₂(i + 1)) = ${ub} ${SX.flag(tot.cmp <= ub)} · moves = 2(n − 1) + I = ${2 * (n - 1) + inv} ${SX.flag(2 * (n - 1) + inv === tot.mov)} — same moves as linear insertion`;
    } else {
      const gaps = SS.GAPS.knuth(n);
      check = `gaps ${gaps.join(", ")} · ${tot.pass} passes · inversions before ${inv}; insertion sort alone would need ≥ ${inv} comparisons`;
    }
    const svg = d3.select("#es-svg");
    SX.clearControls(svg.node());
    const render = (f, k) => {
      const F = AL.frame(svg, W, H, { l: 16, r: 16, t: 26, b: 20 });
      const g = F.g;
      const sortedPrefix = (scheme === "insertion" || scheme === "binary") ? (f.i !== undefined ? f.i + (f.phase === "place" ? 1 : 0) : 0)
                         : (scheme === "selection") ? (f.i !== undefined ? f.i + (f.phase === "swap" ? 1 : 0) : 0) : 0;
      const sortedSuffix = (scheme === "bubble" && f.i !== undefined) ? n - 1 - f.i : 0;
      AL.bars(g, f.a, { w: F.iw, h: F.ih - 20, y: 0, mark: (i, v) => {
        if (f.phase === "done") return AC.good;
        if (scheme === "bubble" && (i === f.j || i === f.j + 1)) return AC.a2;
        if (scheme === "selection" && i === f.m) return AC.violet;
        if (scheme === "selection" && i === f.j) return AC.a2;
        if ((scheme === "insertion" || scheme === "binary") && i === f.j && f.phase !== "start") return AC.a2;
        if (i < sortedPrefix) return AC.good;
        if (sortedSuffix && i >= n - sortedSuffix) return AC.good;
        return null;
      } });
      const so = f.cnt || {};
      g.append("text").attr("x", 0).attr("y", F.ih - 2).attr("font-size", 11).attr("fill", AC.muted)
        .text(`step ${k + 1}/${frames.length} · ${f.phase}${f.h ? " (gap " + f.h + ")" : ""} · so far: ${so.cmp || 0} comparisons, ${so.mov || 0} moves${so.swp ? ", " + so.swp + " swaps" : ""}`);
    };
    stepper = AL.stepper(svg, { frames, render, delay: scheme === "insertion" || scheme === "binary" || scheme === "shell" ? 220 : 90, label: "step" });
    /* stability, tested live: 300 arrays of 40 tagged records over 5 key values */
    let unstable = 0;
    for (let t = 0; t < 300; t++) {
      const rr = AL.rng(t + 1);
      const rec40 = Array.from({ length: 40 }, (_, i) => ({ k: 1 + Math.floor(rr() * 5), tag: i }));
      const nc = AL.counter();
      if (scheme === "insertion") SS.insertionSort(rec40, nc); else if (scheme === "binary") SS.binaryInsertionSort(rec40, nc);
      else if (scheme === "selection") SS.selectionSort(rec40, nc); else if (scheme === "bubble") SS.bubbleSort(rec40, nc); else SS.shellSort(rec40, nc, SS.GAPS.knuth);
      if (!SS.isStable(rec40)) unstable++;
    }
    SX.setHtml("es-readout", `<b>${scheme === "binary" ? "binary insertion" : scheme}</b> on ${n} ${input} values — whole run, measured: <b>${tot.cmp} comparisons</b>, <b>${tot.mov} moves</b>${tot.swp ? ", " + tot.swp + " swaps" : ""}${tot.pass ? ", " + tot.pass + " passes" : ""} · sorted afterwards: ${SX.flag(SS.isSorted(a))} · ${check} · stability, 300 tagged-record runs of 40: order of equal keys violated on <b>${unstable}</b>/300`);
  }
  ["es-scheme", "es-input"].forEach(id => SX.on(id, "change", build));
  SX.on("es-n", "input", () => { SX.setText("es-n-out", SX.val("es-n", 24)); build(); });
  SX.setText("es-n-out", SX.val("es-n", 24));
  build();
})();

/* ── 02  #iv-svg  insertion sort cost vs inversions, measured ──────────── */
(function () {
  if (!SX.has("iv-svg")) return;
  const W = 680, H = 300;
  function build() {
    const n = +SX.val("iv-n", 256);
    const r = AL.rng(23);
    const pts = [];
    const K = 60;
    for (let t = 0; t < K; t++) {
      /* controlled disorder: m random swaps on a sorted array, m spread from 0 to about 2n */
      const a = Array.from({ length: n }, (_, i) => i + 1);
      const m = Math.round(2 * n * t / (K - 1));
      for (let s = 0; s < m; s++) { const i = Math.floor(r() * n), j = Math.floor(r() * n); const x = a[i]; a[i] = a[j]; a[j] = x; }
      const I = SS.inversions(a);
      const c1 = AL.counter(); SS.insertionSort(a.slice(), c1);
      const c2 = AL.counter(); SS.binaryInsertionSort(a.slice(), c2);
      pts.push({ I, cmp: c1.get("cmp"), mov: c1.get("mov"), bcmp: c2.get("cmp"), bmov: c2.get("mov") });
    }
    /* the identities, checked on every point */
    let okC = 0, okM = 0, okB = 0;
    let binUB = 0; for (let i = 1; i < n; i++) binUB += Math.ceil(Math.log2(i + 1));
    pts.forEach(p => { if (p.cmp >= p.I && p.cmp <= p.I + n - 1) okC++; if (p.mov === 2 * (n - 1) + p.I) okM++; if (p.bcmp <= binUB && p.bmov === p.mov) okB++; });
    const F = AL.frame("#iv-svg", W, H, { l: 60, r: 16, t: 14, b: 40 });
    const maxI = d3.max(pts, p => p.I), maxY = d3.max(pts, p => Math.max(p.mov, p.cmp));
    const x = d3.scaleLinear().domain([0, maxI]).range([0, F.iw]).nice();
    const y = d3.scaleLinear().domain([0, maxY]).range([F.ih, 0]).nice();
    AL.gridY(F.g, y, F.iw); AL.axisB(F.g, x, 6, "inversions I in the input (counted directly)"); AL.axisL(F.g, y, 5, "operations, measured");
    /* the two bounding lines for comparisons and the exact moves line */
    const line = (fn, col, dash) => F.g.append("path").datum([0, maxI]).attr("fill", "none").attr("stroke", col).attr("stroke-width", 1.5).attr("stroke-dasharray", dash || null)
      .attr("d", d3.line().x(d => x(d)).y(d => y(fn(d))));
    line(I => I + n - 1, AC.muted, "4 3"); line(I => I, AC.muted, "2 3"); line(I => 2 * (n - 1) + I, AC.a2, "4 3");
    F.g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(binUB)).attr("y2", y(binUB)).attr("stroke", AC.violet).attr("stroke-dasharray", "4 3");
    const dot = (k, col, sym) => F.g.selectAll(".d" + k).data(pts).join("circle").attr("cx", p => x(p.I)).attr("cy", p => y(p[k])).attr("r", 3).attr("fill", col).attr("opacity", 0.9);
    dot("mov", AC.a2); dot("cmp", AC.accent); dot("bcmp", AC.violet);
    AL.legend(F.g, [{ label: "insertion sort comparisons (between I and I + n − 1, dotted/dashed grey)", color: AC.accent },
                    { label: "insertion sort moves = 2(n − 1) + I (dashed amber)", color: AC.a2 },
                    { label: "binary insertion comparisons — flat, ≤ ∑ ceil(log₂(i + 1)) (dashed violet)", color: AC.violet }], 8, 14);
    const lo = pts[0], hi = pts.reduce((m, p) => (p.I > m.I ? p : m), pts[0]);
    /* the average case, Monte Carlo vs the exact formula n(n−1)/4 + n − Hₙ */
    const RUNS = 400; let sumAvg = 0;
    for (let t = 0; t < RUNS; t++) { const cc = AL.counter(); SS.insertionSort(AL.perm(n, AL.rng(1000 + t)), cc); sumAvg += cc.get("cmp"); }
    const exactAvg = n * (n - 1) / 4 + n - SS.harmonic(n);
    SX.setHtml("iv-readout", `n = ${n}, ${K} inputs from 0 to ${SX.int(maxI)} inversions · comparisons within [I, I + n − 1] on ${okC}/${K} inputs ${SX.flag(okC === K)} · moves = 2(n − 1) + I on ${okM}/${K} ${SX.flag(okM === K)} · binary insertion: comparisons ≤ ${SX.int(binUB)} and moves identical to linear on ${okB}/${K} ${SX.flag(okB === K)} · sorted input: ${lo.cmp} comparisons, ${lo.mov} moves; most disordered input here (I = ${SX.int(hi.I)}): ${SX.int(hi.cmp)} comparisons against n(n − 1)/2 = ${SX.int(n * (n - 1) / 2)}, binary insertion ${SX.int(hi.bcmp)} · average over ${RUNS} random permutations: <b>${SX.f1(sumAvg / RUNS)}</b> comparisons against the exact n(n − 1)/4 + n − Hₙ = ${SX.f1(exactAvg)} (${SX.f2(100 * Math.abs(sumAvg / RUNS - exactAvg) / exactAvg)}% off)`);
  }
  SX.on("iv-n", "input", () => { SX.setText("iv-n-out", SX.val("iv-n", 256)); build(); });
  SX.setText("iv-n-out", SX.val("iv-n", 256));
  build();
})();

/* ── 03  #sh-svg  shellsort comparisons vs n for four gap sequences ────── */
(function () {
  if (!SX.has("sh-svg")) return;
  const W = 680, H = 320;
  const NS = [64, 128, 256, 512, 1024, 2048, 4096];
  function build() {
    const input = SX.val("sh-input", "random");
    const seqs = [["shell", "Shell: floor(n/2ᵏ)", AC.bad], ["hibbard", "Hibbard: 2ᵏ − 1", AC.a2], ["knuth", "Knuth: (3ᵏ − 1)/2", AC.accent], ["ciura", "Ciura: 1, 4, 10, 23, 57, …", AC.good]];
    const series = seqs.map(s => ({ key: s[0], label: s[1], color: s[2], pts: [] }));
    const ins = { key: "insertion", label: "insertion sort (gap 1 only)", color: AC.muted, pts: [] };
    NS.forEach(n => {
      let a;
      if (input === "random") a = AL.perm(n, AL.rng(21));
      else { a = []; for (let i = 0; i < n; i++) a.push(i % 2 === 0 ? n / 2 + i / 2 + 1 : (i + 1) / 2); }   // evens hold the large half
      series.forEach(s => { const c = AL.counter(); const b = SS.shellSort(a.slice(), c, SS.GAPS[s.key]); s.pts.push({ n, v: c.get("cmp"), ok: SS.isSorted(b) }); });
      if (n <= 2048 || input !== "random") { const c = AL.counter(); SS.insertionSort(a.slice(), c); ins.pts.push({ n, v: c.get("cmp") }); }
    });
    const all = series.concat([ins]);
    /* fitted exponent: least-squares slope of log v on log n over the measured points */
    const slope = pts => { const xs = pts.map(p => Math.log(p.n)), ys = pts.map(p => Math.log(p.v)); const mx = d3.mean(xs), my = d3.mean(ys); let num = 0, den = 0; xs.forEach((xv, i) => { num += (xv - mx) * (ys[i] - my); den += (xv - mx) * (xv - mx); }); return num / den; };
    const F = AL.frame("#sh-svg", W, H, { l: 64, r: 16, t: 14, b: 40 });
    const x = d3.scaleLog().domain([NS[0], NS[NS.length - 1]]).range([0, F.iw]);
    const y = d3.scaleLog().domain([d3.min(all, s => d3.min(s.pts, p => p.v)), d3.max(all, s => d3.max(s.pts, p => p.v))]).range([F.ih, 0]).nice();
    F.g.append("g").attr("class", "gridlines").selectAll("line").data(y.ticks(6)).join("line").attr("x1", 0).attr("x2", F.iw).attr("y1", d => y(d)).attr("y2", d => y(d)).attr("stroke", AC.grid);
    AL.axisB(F.g, x, 7, "n (log scale)", d3.format("~s")); AL.axisL(F.g, y, 6, "comparisons (log scale)", d3.format("~s"));
    all.forEach(s => {
      F.g.append("path").datum(s.pts).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", 2).attr("d", d3.line().x(p => x(p.n)).y(p => y(p.v)));
      F.g.selectAll(".p" + s.key).data(s.pts).join("circle").attr("cx", p => x(p.n)).attr("cy", p => y(p.v)).attr("r", 3).attr("fill", s.color);
    });
    AL.legend(F.g, all.map(s => ({ label: `${s.label} — fitted exponent ${SX.f2(slope(s.pts))}`, color: s.color })), 8, 14);
    const at = (s, n) => s.pts.find(p => p.n === n);
    const sortedAll = series.every(s => s.pts.every(p => p.ok));
    SX.setHtml("sh-readout", `${input === "random" ? "one random permutation per n (fixed seed)" : "the interleaved input: large keys in even positions, small in odd"} · at n = 4096: Shell ${SX.int(at(series[0], 4096).v)} · Hibbard ${SX.int(at(series[1], 4096).v)} · Knuth ${SX.int(at(series[2], 4096).v)} · Ciura ${SX.int(at(series[3], 4096).v)} comparisons${ins.pts.length && at(ins, 4096) ? " · insertion " + SX.int(at(ins, 4096).v) : (at(ins, 2048) ? " · insertion at n = 2048: " + SX.int(at(ins, 2048).v) : "")} · every run re-checked sorted ${SX.flag(sortedAll)} · exponents are least-squares slopes on the log–log points shown, not theorems`);
  }
  SX.on("sh-input", "change", build);
  build();
})();

/* ── 04  #dt-svg  the decision tree of a real sort for n = 3 or 4 ───────── */
(function () {
  if (!SX.has("dt-svg")) return;
  const W = 680, H = 330;
  function perms(n) { const out = []; const a = Array.from({ length: n }, (_, i) => i); (function go(k) { if (k === n) { out.push(a.slice()); return; } for (let i = k; i < n; i++) { const t = a[k]; a[k] = a[i]; a[i] = t; go(k + 1); const u = a[k]; a[k] = a[i]; a[i] = u; } })(0); return out; }
  /* run the chosen sort on labelled records, logging each comparison as "x>y" and its answer */
  function trace(scheme, p) {
    const lab = "abcd";
    const a = p.map((v, i) => ({ k: v, tag: lab[i] }));
    const seq = [];
    const c = { add: () => {} };
    const gt = (x, y) => { const r = x.k > y.k; seq.push({ q: x.tag + ">" + y.tag, ans: r }); return r; };
    const n = a.length;
    if (scheme === "insertion") {
      for (let i = 1; i < n; i++) { const v = a[i]; let j = i; while (j > 0 && gt(a[j - 1], v)) { a[j] = a[j - 1]; j--; } a[j] = v; }
    } else if (scheme === "selection") {
      for (let i = 0; i < n - 1; i++) { let m = i; for (let j = i + 1; j < n; j++) if (gt(a[m], a[j])) m = j; const t = a[i]; a[i] = a[m]; a[m] = t; }
    } else {
      for (let pass = 0; pass < n - 1; pass++) { let sw = false; for (let j = 0; j < n - 1 - pass; j++) if (gt(a[j], a[j + 1])) { const t = a[j]; a[j] = a[j + 1]; a[j + 1] = t; sw = true; } if (!sw) break; }
    }
    return { seq, out: a.map(x => x.tag).join("") };
  }
  function build() {
    const scheme = SX.val("dt-scheme", "insertion"), n = +SX.val("dt-n", 3);
    const P = perms(n);
    const root = { name: "", kids: {}, leaf: null, count: 0 };
    const leaves = [];
    P.forEach(p => {
      const t = trace(scheme, p);
      let node = root;
      t.seq.forEach((s, d) => {
        node.name = s.q;
        const key = s.ans ? "Y" : "N";
        if (!node.kids[key]) node.kids[key] = { name: "", kids: {}, leaf: null, count: 0 };
        node = node.kids[key];
      });
      node.leaf = (node.leaf ? node.leaf + " " : "") + t.out; node.count++;
      leaves.push(t.seq.length);
    });
    const h = d3.hierarchy(root, d => ["N", "Y"].map(k => d.kids[k]).filter(Boolean).map(k => Object.assign(k, { _edge: k === d.kids.Y ? "yes" : "no" })));
    const F = AL.frame("#dt-svg", W, H, { l: 10, r: 10, t: 22, b: 10 });
    const layout = d3.tree().size([F.iw, F.ih - 30]);
    layout(h);
    F.g.selectAll(".e").data(h.links()).join("line").attr("x1", d => d.source.x).attr("y1", d => d.source.y).attr("x2", d => d.target.x).attr("y2", d => d.target.y).attr("stroke", AC.line);
    F.g.selectAll(".el").data(h.links()).join("text").attr("x", d => (d.source.x + d.target.x) / 2).attr("y", d => (d.source.y + d.target.y) / 2 - 3).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", AC.muted).text(d => d.target.data._edge);
    const nodes = F.g.selectAll(".n").data(h.descendants()).join("g").attr("transform", d => `translate(${d.x},${d.y})`);
    nodes.filter(d => !d.data.leaf).append("circle").attr("r", n === 3 ? 15 : 11).attr("fill", AC.panel2).attr("stroke", AC.accent);
    nodes.filter(d => !d.data.leaf).append("text").attr("y", 4).attr("text-anchor", "middle").attr("font-size", n === 3 ? 11 : 9).attr("fill", AC.ink).text(d => d.data.name.replace(">", " > "));
    nodes.filter(d => d.data.leaf).append("rect").attr("x", n === 3 ? -22 : -13).attr("y", -9).attr("width", n === 3 ? 44 : 26).attr("height", 18).attr("rx", 4).attr("fill", d => d.data.count > 1 ? AC.bad : AC.good).attr("opacity", 0.85);
    nodes.filter(d => d.data.leaf).append("text").attr("y", 4).attr("text-anchor", "middle").attr("font-size", n === 3 ? 10 : 8).attr("fill", AC.bg).text(d => d.data.leaf.split(" ")[0] + (d.data.count > 1 ? " ×" + d.data.count : ""));
    const height = d3.max(leaves), minD = d3.min(leaves), avg = d3.mean(leaves);
    const nLeaves = h.leaves().length, lb = Math.log2(P.length);
    F.g.append("text").attr("x", 0).attr("y", -8).attr("font-size", 11).attr("fill", AC.muted)
      .text(`${scheme} sort, n = ${n}: ${nLeaves} leaves for ${P.length} permutations · height ${height} · log₂(${n}!) = ${SX.f3(lb)} ⇒ ≥ ${Math.ceil(lb)}`);
    SX.setHtml("dt-readout", `every root-to-leaf path is the comparison sequence of the real routine on one input · leaves reached: <b>${nLeaves}</b> (permutations: ${P.length}${nLeaves < P.length ? "; red leaves are reached by more than one input — impossible for a correct sort, so this cannot happen" : ""}) · <b>height ${height}</b> = worst-case comparisons, shortest path ${minD}, mean depth ${SX.f3(avg)} · bound: height ≥ ceil(log₂ ${n}!) = ${Math.ceil(lb)} ${SX.flag(height >= Math.ceil(lb))}, mean depth ≥ log₂ ${n}! = ${SX.f3(lb)} ${SX.flag(avg >= lb)}${scheme === "selection" ? " · selection sort asks all n(n − 1)/2 = " + n * (n - 1) / 2 + " questions on every input: some answer combinations are never realised, so the tree has fewer than 2^height leaves" : ""}`);
    /* the table: exact log₂(n!) vs Stirling vs n·log₂n vs mergesort's worst */
    const rows = [2, 3, 4, 5, 8, 10, 12, 16, 32, 64, 100, 1000, 10000, 1000000].map(m => {
      const ex = SS.log2Factorial(m), st = SS.stirlingLog2(m);
      return [SX.int(m), SX.f3(ex), SX.int(Math.ceil(ex)), SX.f3(st), (ex - st).toExponential(1), SX.f1(m * Math.log2(m)), SX.f3(ex / (m * Math.log2(m))), SX.int(SS.mergeWorstClosed(m))];
    });
    SX.table("#dt-table", ["n", "log₂(n!) exact (sum of logs)", "⇒ any comparison sort needs ≥", "Stirling: n·log₂n − 1.4427n + ½·log₂(2πn)", "exact − Stirling", "n·log₂n", "log₂(n!) / (n·log₂n)", "mergesort worst (§10)"], rows);
  }
  ["dt-scheme", "dt-n"].forEach(id => SX.on(id, "change", build));
  build();
})();

/* ── 05  #mr-svg  one merge, stepped, with the stability tie visible ───── */
(function () {
  if (!SX.has("mr-svg")) return;
  const W = 680, H = 210;
  const presets = {
    tie: { L: [[2, "2"], [5, "5a"], [7, "7"]], R: [[3, "3"], [5, "5b"], [9, "9"]] },
    worked: { L: [[3, "3"], [27, "27"], [38, "38"], [43, "43"]], R: [[1, "1"], [9, "9"], [10, "10"], [82, "82"]] },
    disjoint: { L: [[1, "1"], [2, "2"], [3, "3"], [4, "4"]], R: [[5, "5"], [6, "6"], [7, "7"], [8, "8"]] },
    interleaved: { L: [[1, "1"], [3, "3"], [5, "5"], [7, "7"]], R: [[2, "2"], [4, "4"], [6, "6"], [8, "8"]] }
  };
  function build() {
    const pr = presets[SX.val("mr-preset", "tie")];
    const mk = arr => arr.map(x => ({ k: x[0], tag: x[1] }));
    const L = mk(pr.L), R = mk(pr.R);
    const a = L.concat(R), mid = L.length, hi = a.length, buf = new Array(hi);
    const c = AL.counter(), frames = [];
    frames.push({ a: a.slice(), i: 0, j: mid, k: 0, phase: "start", cnt: {} });
    SS.merge(a, 0, mid, hi, buf, c, f => frames.push(Object.assign(f, { cnt: c.all() })));
    const cmp = c.get("cmp"), m = hi;
    /* the identity: comparisons = m − t, t = length of the trailing single-side run of the OUTPUT */
    const src = a.map(x => (L.indexOf(x) >= 0 ? "L" : "R"));
    let t = 1; while (t < m && src[m - 1 - t] === src[m - 1]) t++;
    const stable = SS.isStable(a.map((x, i) => ({ k: x.k, tag: (x.tag.endsWith("b") ? 1 : 0) })));
    const svg = d3.select("#mr-svg");
    SX.clearControls(svg.node());
    const render = (f, idx) => {
      const F = AL.frame(svg, W, H, { l: 70, r: 10, t: 14, b: 8 });
      const g = F.g;
      const lab = x => x.tag;
      AL.row(g, L.map(lab), { x: 0, y: 0, w: 40, label: "left run", mark: (i) => i < f.i - 0 ? AC.panel : (i === f.i && f.i < mid && f.phase !== "start" ? AC.a2 : (i === f.i && f.phase === "start" ? AC.a2 : null)), index: false });
      AL.row(g, R.map(lab), { x: 0, y: 50, w: 40, label: "right run", mark: (i) => i + mid < f.j ? AC.panel : ((i + mid === f.j && f.j < hi) ? AC.a2 : null), index: false });
      const out = f.a.slice(0, f.k).map(lab);
      while (out.length < m) out.push("");
      AL.row(g, out, { x: 0, y: 110, w: 40, label: "output", mark: (i) => i < f.k ? (f.a[i].tag.endsWith("a") || f.a[i].tag.endsWith("b") ? AC.violet : AC.good) : null });
      const so = f.cnt || {};
      g.append("text").attr("x", 0).attr("y", 175).attr("font-size", 11).attr("fill", AC.muted)
        .text(f.phase === "start" ? "cursors at the front of each run; the smaller front element is emitted, ties go LEFT (≤)" :
              f.phase === "emit" ? `compared front elements → emitted output[${f.k - 1}] · comparisons so far ${so.cmp || 0}` :
              `${f.phase === "copyL" ? "right" : "left"} run exhausted: copying without comparing · comparisons ${so.cmp || 0}`);
    };
    AL.stepper(svg, { frames, render, delay: 800, label: "step" });
    SX.setHtml("mr-readout", `merge of ${L.length} + ${R.length} = ${m} elements — measured <b>${cmp} comparisons</b>, ${c.get("mov")} moves (${m} into the buffer, ${m} back) · identity m − t: trailing run of ${t} element${t > 1 ? "s" : ""} from one side, so m − t = ${m} − ${t} = ${m - t} ${SX.flag(m - t === cmp)} · bounds min(p, q) = ${Math.min(L.length, R.length)} ≤ ${cmp} ≤ m − 1 = ${m - 1} ${SX.flag(cmp >= Math.min(L.length, R.length) && cmp <= m - 1)} · output ${a.map(x => x.tag).join(" ")} · equal keys kept left-before-right: ${SX.flag(stable)}`);
  }
  SX.on("mr-preset", "change", build);
  build();
})();

/* ── 06  #mc-svg  mergesort comparisons vs n against the exact formulas ── */
(function () {
  if (!SX.has("mc-svg")) return;
  const W = 680, H = 320;
  const NS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024];
  function build() {
    const runs = +SX.val("mc-runs", 20);
    const S = { tdAdv: [], tdRand: [], tdSorted: [], buRand: [], buAdv: [], closed: [], buBound: [], nat: [] };
    let agree = 0;
    NS.forEach(n => {
      const adv = SS.mergeAdversary(n);
      const c1 = AL.counter(); SS.mergeSort(adv.slice(), c1); S.tdAdv.push({ n, v: c1.get("cmp") / n });
      const cl = SS.mergeWorstClosed(n); S.closed.push({ n, v: cl / n }); if (cl === c1.get("cmp")) agree++;
      const c2 = AL.counter(); SS.mergeSortBottomUp(adv.slice(), c2); S.buAdv.push({ n, v: c2.get("cmp") / n });
      let wbu = 0; for (let w = 1; w < n; w *= 2) for (let lo = 0; lo + w < n; lo += 2 * w) wbu += Math.min(lo + 2 * w, n) - lo - 1; S.buBound.push({ n, v: wbu / n });
      const c3 = AL.counter(); SS.mergeSort(Array.from({ length: n }, (_, i) => i + 1), c3); S.tdSorted.push({ n, v: c3.get("cmp") / n });
      let s1 = 0, s2 = 0, s3 = 0;
      for (let t = 0; t < runs; t++) {
        const a = AL.perm(n, AL.rng(100 * n + t));
        const d1 = AL.counter(); SS.mergeSort(a.slice(), d1); s1 += d1.get("cmp");
        const d2 = AL.counter(); SS.mergeSortBottomUp(a.slice(), d2); s2 += d2.get("cmp");
        const d3c = AL.counter(); SS.mergeSortNatural(a.slice(), d3c); s3 += d3c.get("cmp");
      }
      S.tdRand.push({ n, v: s1 / runs / n }); S.buRand.push({ n, v: s2 / runs / n }); S.nat.push({ n, v: s3 / runs / n });
    });
    const F = AL.frame("#mc-svg", W, H, { l: 56, r: 16, t: 14, b: 40 });
    const x = d3.scaleLog().domain([2, 1024]).range([0, F.iw]);
    const y = d3.scaleLinear().domain([0, d3.max(S.buBound.concat(S.nat), p => p.v)]).range([F.ih, 0]).nice();
    AL.gridY(F.g, y, F.iw); AL.axisB(F.g, x, 10, "n (log scale)", d3.format("~s")); AL.axisL(F.g, y, 6, "comparisons ÷ n");
    const path = (pts, col, dash, w) => F.g.append("path").datum(pts).attr("fill", "none").attr("stroke", col).attr("stroke-width", w || 2).attr("stroke-dasharray", dash || null).attr("d", d3.line().x(p => x(p.n)).y(p => y(p.v)));
    const dots = (pts, col) => F.g.selectAll(".x" + col.slice(1)).data(pts).join("circle").attr("cx", p => x(p.n)).attr("cy", p => y(p.v)).attr("r", 2.5).attr("fill", col);
    path(S.closed, AC.ink, "5 3", 1.5); dots(S.tdAdv, AC.bad);
    path(S.buBound, AC.rose, "2 3", 1.5); dots(S.buAdv, AC.rose);
    path(S.tdRand, AC.accent); path(S.buRand, AC.a2, "6 3"); path(S.nat, AC.violet, "3 2"); path(S.tdSorted, AC.good);
    /* the reference n·log₂n line, i.e. log₂ n on this axis */
    path(NS.map(n => ({ n, v: Math.log2(n) })), AC.muted, "1 3", 1);
    AL.legend(F.g, [
      { label: "top-down on the adversarial input (dots) vs n·ceil(log₂n) − 2^ceil(log₂n) + 1 (dashed)", color: AC.bad },
      { label: "bottom-up on the same input (dots) vs its own bound ∑(m − 1) (dotted rose)", color: AC.rose },
      { label: "top-down, random permutations (mean)", color: AC.accent },
      { label: "bottom-up, random (mean)", color: AC.a2 },
      { label: "natural mergesort, random (mean, incl. run detection)", color: AC.violet },
      { label: "top-down, sorted input (best case)", color: AC.good },
      { label: "n·log₂n reference", color: AC.muted }], 8, 14);
    const at = (s, n) => S[s].find(p => p.n === n);
    SX.setHtml("mc-readout", `adversarial input meets the closed form exactly at <b>${agree}/${NS.length}</b> sizes ${SX.flag(agree === NS.length)} · at n = 1024: worst ${SX.int(at("tdAdv", 1024).v * 1024)} (= ${SX.f3(at("tdAdv", 1024).v)}·n), random top-down ${SX.f3(at("tdRand", 1024).v)}·n, bottom-up random ${SX.f3(at("buRand", 1024).v)}·n, natural ${SX.f3(at("nat", 1024).v)}·n, sorted ${SX.f3(at("tdSorted", 1024).v)}·n, against log₂n = 10 · bottom-up's worst bound ∑(m − 1) exceeds top-down's at n = ${NS.filter(n => at("buBound", n).v > at("closed", n).v).join(", ")} among the sizes plotted, and the two coincide at every power of two · ${runs} random permutations per size`);
  }
  SX.on("mc-runs", "input", () => { SX.setText("mc-runs-out", SX.val("mc-runs", 20)); build(); });
  SX.setText("mc-runs-out", SX.val("mc-runs", 20));
  build();
})();

/* ── 07  #pt-svg  Lomuto vs Hoare partition stepped on the same array ──── */
(function () {
  if (!SX.has("pt-svg")) return;
  const W = 680, H = 230;
  const presets = { worked: [2, 8, 7, 1, 3, 5, 6, 4], sorted: [1, 2, 3, 4, 5, 6, 7, 8], equal: [4, 4, 4, 4, 4, 4, 4, 4], dup: [4, 1, 4, 2, 4, 3, 4, 4] };
  function build() {
    const base = presets[SX.val("pt-preset", "worked")];
    const n = base.length;
    /* Lomuto: pivot = last element. Hoare: the SAME pivot value, first moved to the front. */
    const aL = base.slice(), cL = AL.counter(), fL = [];
    fL.push({ a: aL.slice(), phase: "start", i: 0, j: 0, pivot: n - 1, cnt: {} });
    const q = SS.lomuto(aL, 0, n, cL, f => fL.push(Object.assign(f, { cnt: cL.all() })));
    const aH = base.slice(), cH = AL.counter(), fH = [];
    SS.swap(aH, n - 1, 0, cH);
    fH.push({ a: aH.slice(), phase: "moved pivot to front", i: -1, j: n, pivotVal: aH[0], cnt: cH.all() });
    const jH = SS.hoare(aH, 0, n, cH, f => fH.push(Object.assign(f, { cnt: cH.all() })));
    const p = base[n - 1];
    /* checks: Lomuto's postcondition and Hoare's, verified on the outputs */
    const okL = aL.slice(0, q).every(v => v <= p) && aL[q] === p && aL.slice(q + 1).every(v => v > p);
    const okH = aH.slice(0, jH + 1).every(v => v <= p) && aH.slice(jH + 1).every(v => v >= p) && jH < n - 1;
    const frames = []; const m = Math.max(fL.length, fH.length);
    for (let k = 0; k < m; k++) frames.push({ L: fL[Math.min(k, fL.length - 1)], H: fH[Math.min(k, fH.length - 1)], k });
    const svg = d3.select("#pt-svg");
    SX.clearControls(svg.node());
    const render = (f) => {
      const F = AL.frame(svg, W, H, { l: 90, r: 10, t: 16, b: 6 });
      const g = F.g;
      const L = f.L, Hh = f.H;
      g.append("text").attr("x", -84).attr("y", -4).attr("font-size", 11).attr("fill", AC.muted).text(`Lomuto, pivot a[hi−1] = ${p}`);
      AL.row(g, L.a, { x: 0, y: 8, w: 44, mark: (i, v) => {
        if (L.phase === "final") return i === L.pivot ? AC.a2 : (i < L.pivot ? AC.good : AC.bad);
        if (i === L.pivot) return AC.a2;
        if (i < L.i) return AC.good; if (i < L.j) return AC.bad; return null; } });
      g.append("text").attr("x", 0).attr("y", 62).attr("font-size", 10).attr("fill", AC.muted)
        .text(L.phase === "start" ? "i = lo (≤ region empty), j = lo; scan begins" : L.phase === "final" ? `swap a[i] ↔ pivot: pivot fixed at index ${L.pivot}; returns ${L.pivot} · ${L.cnt.cmp || 0} comparisons, ${L.cnt.swp || 0} swaps` : `i = ${L.i} (a[lo..i) ≤ ${p}, green), j = ${L.j} (a[i..j) > ${p}, red) · ${L.cnt.cmp || 0} comparisons, ${L.cnt.swp || 0} swaps`);
      g.append("text").attr("x", -84).attr("y", 96).attr("font-size", 11).attr("fill", AC.muted).text(`Hoare, pivot value ${p}`);
      AL.row(g, Hh.a, { x: 0, y: 108, w: 44, mark: (i, v) => {
        if (Hh.phase === "cross") return i <= Hh.j ? AC.good : AC.bad;
        if (i === Hh.i || i === Hh.j) return AC.a2;
        if (i < Hh.i) return AC.good; if (i > Hh.j) return AC.bad; return null; } });
      g.append("text").attr("x", 0).attr("y", 162).attr("font-size", 10).attr("fill", AC.muted)
        .text(Hh.phase === "moved pivot to front" ? `pivot ${p} swapped to a[lo]; i = lo − 1, j = hi; scans move inward` : Hh.phase === "cross" ? `i = ${Hh.i} ≥ j = ${Hh.j}: crossed — returns j = ${Hh.j}; a[lo..j] ≤ ${p} ≤ a[j+1..hi); the pivot is at index ${Hh.a.indexOf(p)}, NOT fixed · ${Hh.cnt.cmp || 0} comparisons, ${Hh.cnt.swp || 0} swaps` : `scans stopped at i = ${Hh.i} (a[i] ≥ ${p}) and j = ${Hh.j} (a[j] ≤ ${p}); i < j so swap · ${Hh.cnt.cmp || 0} comparisons, ${Hh.cnt.swp || 0} swaps`);
      g.append("text").attr("x", 0).attr("y", 192).attr("font-size", 10).attr("fill", AC.muted).text(`step ${f.k + 1} / ${m}`);
    };
    AL.stepper(svg, { frames, render, delay: 900, label: "step" });
    SX.setHtml("pt-readout", `input [${base.join(", ")}], pivot value ${p} · <b>Lomuto</b>: ${cL.get("cmp")} comparisons (= n − 1 = ${n - 1} ${SX.flag(cL.get("cmp") === n - 1)}), ${cL.get("swp") || 0} swaps, pivot fixed at ${q}, postcondition a[0..q) ≤ p = a[q] &lt; a[q+1..n) ${SX.flag(okL)} · <b>Hoare</b>: ${cH.get("cmp")} comparisons, ${cH.get("swp") || 0} swaps incl. the initial move, returns j = ${jH} with a[0..j] ≤ p ≤ a[j+1..n) and j &lt; n − 1 ${SX.flag(okH)} · result Lomuto [${aL.join(", ")}] · Hoare [${aH.join(", ")}]`);
  }
  SX.on("pt-preset", "change", build);
  build();
})();

/* ── 08  #tw-svg  three-way partition stepped; the three schemes on duplicates */
(function () {
  if (!SX.has("tw-svg")) return;
  const W = 680, H = 170;
  function build() {
    const base = [3, 5, 1, 3, 2, 3, 4, 3, 1, 3];
    const a = base.slice(), c = AL.counter(), frames = [];
    frames.push({ a: a.slice(), lt: 0, i: 1, gt: a.length, phase: "start", cnt: {} });
    const lg = SS.threeWay(a, 0, a.length, c, f => frames.push(Object.assign(f, { cnt: c.all() })));
    const p = base[0];
    const ok = a.slice(0, lg[0]).every(v => v < p) && a.slice(lg[0], lg[1]).every(v => v === p) && a.slice(lg[1]).every(v => v > p);
    const svg = d3.select("#tw-svg");
    SX.clearControls(svg.node());
    const render = (f, k) => {
      const F = AL.frame(svg, W, H, { l: 16, r: 10, t: 16, b: 6 });
      const g = F.g;
      AL.row(g, f.a, { x: 0, y: 6, w: 44, mark: (i) => {
        if (i < f.lt) return AC.good; if (i < f.i) return AC.a2; if (i >= f.gt) return AC.bad; return null; } });
      g.append("text").attr("x", 0).attr("y", 64).attr("font-size", 10).attr("fill", AC.muted)
        .text(`lt = ${f.lt}, i = ${f.i}, gt = ${f.gt} · a[0..lt) < ${p} (green) · a[lt..i) = ${p} (amber) · a[i..gt) unexamined · a[gt..n) > ${p} (red) · comparisons ${f.cnt.cmp || 0}, swaps ${f.cnt.swp || 0}`);
      g.append("text").attr("x", 0).attr("y", 84).attr("font-size", 10).attr("fill", AC.muted).text(`step ${k + 1} / ${frames.length}${k === frames.length - 1 ? " — done: returns [lt, gt) = [" + lg[0] + ", " + lg[1] + "); only a[0..lt) and a[gt..n) are recursed on" : ""}`);
    };
    AL.stepper(svg, { frames, render, delay: 800, label: "step" });
    /* the three schemes on duplicate-heavy inputs, measured */
    const n = +SX.val("tw-n", 1024);
    const inputs = { "all equal": SX.inputs.allEqual(n), "4 distinct keys": SX.inputs.fewUnique(n, 7), "16 distinct keys": Array.from({ length: n }, (_, i) => (i * 7919) % 16 + 1), "distinct (random)": AL.perm(n, AL.rng(3)) };
    const rows = Object.entries(inputs).map(([nm, arr]) => {
      const cells = [nm];
      ["lomuto", "hoare", "threeway"].forEach(s => { const cc = AL.counter(); const b = SS.quickSort(arr.slice(), cc, { scheme: s, pivot: "last" }); cells.push(`${SX.int(cc.get("cmp"))} cmp · ${SX.int(cc.get("swp") || 0)} swaps · depth ${cc.get("depth")} · ${SS.isSorted(b) ? "sorted ✓" : "NOT SORTED"}`); });
      return cells;
    });
    SX.table("#tw-table", ["input, n = " + n, "Lomuto (last pivot)", "Hoare (same pivot value)", "three-way (same pivot value)"], rows);
    SX.setHtml("tw-readout", `trace: pivot ${p} on [${base.join(", ")}] → [${a.join(", ")}], <b>${c.get("cmp")} comparisons</b> (this formulation asks up to two questions per element), ${c.get("swp") || 0} swaps, [lt, gt) = [${lg[0]}, ${lg[1]}), the ${lg[1] - lg[0]} copies of ${p} are final ${SX.flag(ok)} · table: n(n − 1)/2 = ${SX.int(n * (n - 1) / 2)}, n·log₂n = ${SX.int(Math.round(n * Math.log2(n)))}`);
  }
  SX.on("tw-n", "change", build);
  build();
})();

/* ── 09  #pv-svg  pivot strategies on adversarial and random inputs ─────── */
(function () {
  if (!SX.has("pv-svg")) return;
  const W = 680, H = 300;
  function build() {
    const n = +SX.val("pv-n", 1024);
    const inputs = [["sorted", SX.inputs.sorted(n)], ["reversed", SX.inputs.reversed(n)], ["organ pipe", SX.inputs.organPipe(n)], ["sawtooth (16 keys)", Array.from({ length: n }, (_, i) => (i % 16) + 1)], ["random", AL.perm(n, AL.rng(3))]];
    const strategies = [["last", "last element", AC.bad], ["middle", "middle element", AC.a2], ["median3", "median of three", AC.accent], ["ninther", "ninther", AC.violet], ["random", "random", AC.good]];
    const data = inputs.map(([nm, arr]) => ({ name: nm, vals: strategies.map(([s]) => { const c = AL.counter(); SS.quickSort(arr.slice(), c, { scheme: "lomuto", pivot: s, seed: 5 }); return { s, cmp: c.get("cmp"), depth: c.get("depth") }; }) }));
    const F = AL.frame("#pv-svg", W, H, { l: 60, r: 16, t: 14, b: 44 });
    const x0 = d3.scaleBand().domain(inputs.map(i => i[0])).range([0, F.iw]).padding(0.2);
    const x1 = d3.scaleBand().domain(strategies.map(s => s[0])).range([0, x0.bandwidth()]).padding(0.08);
    const ymax = d3.max(data, d => d3.max(d.vals, v => v.cmp));
    const y = d3.scaleLog().domain([n, ymax]).range([F.ih, 0]).nice();
    F.g.append("g").attr("class", "gridlines").selectAll("line").data(y.ticks(5)).join("line").attr("x1", 0).attr("x2", F.iw).attr("y1", d => y(d)).attr("y2", d => y(d)).attr("stroke", AC.grid);
    AL.axisB(F.g, x0, 5, "input shape"); AL.axisL(F.g, y, 5, "comparisons (log scale)", d3.format("~s"));
    const ref = v => F.g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(v)).attr("y2", y(v)).attr("stroke", AC.muted).attr("stroke-dasharray", "4 3");
    ref(n * (n - 1) / 2); ref(2 * n * Math.log(n));
    F.g.append("text").attr("x", F.iw).attr("y", y(n * (n - 1) / 2) - 3).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.muted).text("n(n − 1)/2");
    F.g.append("text").attr("x", F.iw).attr("y", y(2 * n * Math.log(n)) - 3).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.muted).text("2n·ln n");
    const grp = F.g.selectAll(".grp").data(data).join("g").attr("transform", d => `translate(${x0(d.name)},0)`);
    grp.selectAll("rect").data(d => d.vals).join("rect").attr("x", v => x1(v.s)).attr("y", v => y(v.cmp)).attr("width", x1.bandwidth()).attr("height", v => F.ih - y(v.cmp)).attr("fill", v => strategies.find(s => s[0] === v.s)[2]).attr("rx", 2);
    AL.legend(F.g, strategies.map(s => ({ label: s[1], color: s[2] })), 8, 14);
    const cell = (inp, s) => data.find(d => d.name === inp).vals.find(v => v.s === s);
    SX.setHtml("pv-readout", `n = ${n}, Lomuto partition, no cutoff · sorted input: last-element pivot <b>${SX.int(cell("sorted", "last").cmp)}</b> (= n(n − 1)/2 = ${SX.int(n * (n - 1) / 2)} ${SX.flag(cell("sorted", "last").cmp === n * (n - 1) / 2)}, depth ${cell("sorted", "last").depth}), middle ${SX.int(cell("sorted", "middle").cmp)}, median-of-three ${SX.int(cell("sorted", "median3").cmp)}, random ${SX.int(cell("sorted", "random").cmp)} · organ pipe: median-of-three <b>${SX.int(cell("organ pipe", "median3").cmp)}</b> at depth ${cell("organ pipe", "median3").depth} — its killer input — against ninther ${SX.int(cell("organ pipe", "ninther").cmp)} and random ${SX.int(cell("organ pipe", "random").cmp)} · random input: all five within ${SX.int(d3.min(data[4].vals, v => v.cmp))}–${SX.int(d3.max(data[4].vals, v => v.cmp))} · 2n·ln n = ${SX.int(Math.round(2 * n * Math.log(n)))}`);
  }
  SX.on("pv-n", "change", build);
  build();
})();

/* ── 10  #qc-svg  quicksort comparisons vs the exact average recurrence ── */
(function () {
  if (!SX.has("qc-svg")) return;
  const W = 680, H = 320;
  const NS = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
  function build() {
    const runs = +SX.val("qc-runs", 30);
    const EL = SS.quickExpected(4096, n => n - 1);
    const S = { lom: [], hoare: [], three: [], rec: [], closed: [], twoNlnN: [], c139: [], sortedLast: [] };
    let maxDev = 0;
    NS.forEach(n => {
      let sl = 0, sh = 0, st = 0;
      for (let t = 0; t < runs; t++) {
        const a = AL.perm(n, AL.rng(7 * n + t));
        const c1 = AL.counter(); SS.quickSort(a.slice(), c1, { scheme: "lomuto", pivot: "random", seed: t + 1 }); sl += c1.get("cmp");
        const c2 = AL.counter(); SS.quickSort(a.slice(), c2, { scheme: "hoare", pivot: "random", seed: t + 1 }); sh += c2.get("cmp");
        const c3 = AL.counter(); SS.quickSort(a.slice(), c3, { scheme: "threeway", pivot: "random", seed: t + 1 }); st += c3.get("cmp");
      }
      S.lom.push({ n, v: sl / runs / n }); S.hoare.push({ n, v: sh / runs / n }); S.three.push({ n, v: st / runs / n });
      S.rec.push({ n, v: EL[n] / n });
      const cl = 2 * (n + 1) * SS.harmonic(n) - 4 * n; S.closed.push({ n, v: cl / n });
      maxDev = Math.max(maxDev, Math.abs(sl / runs - EL[n]) / EL[n]);
      S.twoNlnN.push({ n, v: 2 * Math.log(n) }); S.c139.push({ n, v: 1.39 * Math.log2(n) });
      const c4 = AL.counter(); SS.quickSort(SX.inputs.sorted(n), c4, { scheme: "lomuto", pivot: "last" }); S.sortedLast.push({ n, v: c4.get("cmp") / n });
    });
    const F = AL.frame("#qc-svg", W, H, { l: 56, r: 16, t: 14, b: 40 });
    const x = d3.scaleLog().domain([4, 4096]).range([0, F.iw]);
    const y = d3.scaleLog().domain([1, d3.max(S.sortedLast, p => p.v)]).range([F.ih, 0]).nice();
    F.g.append("g").attr("class", "gridlines").selectAll("line").data(y.ticks(5)).join("line").attr("x1", 0).attr("x2", F.iw).attr("y1", d => y(d)).attr("y2", d => y(d)).attr("stroke", AC.grid);
    AL.axisB(F.g, x, 8, "n (log scale)", d3.format("~s")); AL.axisL(F.g, y, 5, "comparisons ÷ n (log scale)", d3.format("~s"));
    const path = (pts, col, dash, w) => F.g.append("path").datum(pts).attr("fill", "none").attr("stroke", col).attr("stroke-width", w || 2).attr("stroke-dasharray", dash || null).attr("d", d3.line().x(p => x(p.n)).y(p => y(p.v)));
    const dots = (pts, col) => F.g.selectAll(".q" + col.slice(1)).data(pts).join("circle").attr("cx", p => x(p.n)).attr("cy", p => y(p.v)).attr("r", 3).attr("fill", col);
    path(S.rec, AC.ink, "5 3", 1.5); dots(S.lom, AC.accent);
    path(S.twoNlnN, AC.muted, "2 3", 1.2); path(S.c139, AC.muted, "6 2", 1.2);
    path(S.hoare, AC.a2); path(S.three, AC.violet); path(S.sortedLast, AC.bad);
    AL.legend(F.g, [
      { label: "Lomuto, random pivot: measured mean (dots) vs the recurrence C(n) = n − 1 + (2/n)∑C(i) (dashed)", color: AC.accent },
      { label: "Hoare, random pivot: measured mean", color: AC.a2 },
      { label: "three-way (two questions per key ≥ pivot), random pivot: measured mean", color: AC.violet },
      { label: "Lomuto, LAST pivot on sorted input: n(n − 1)/2 — the worst case", color: AC.bad },
      { label: "2n·ln n (dotted) and 1.39·n·log₂n (dashed) references", color: AC.muted }], 8, 14);
    const at = (s, n) => S[s].find(p => p.n === n);
    const closedOK = S.rec.every((p, i) => Math.abs(p.v - S.closed[i].v) < 1e-9);
    SX.setHtml("qc-readout", `${runs} random permutations per size · recurrence solved numerically vs the closed form 2(n + 1)Hₙ − 4n: identical at every n ${SX.flag(closedOK)} · largest gap between the Lomuto measurement and the recurrence over the eleven sizes: <b>${SX.f2(100 * maxDev)}%</b> · at n = 4096: Lomuto ${SX.f2(at("lom", 4096).v)}·n measured vs ${SX.f2(at("rec", 4096).v)}·n exact, Hoare ${SX.f2(at("hoare", 4096).v)}·n, three-way ${SX.f2(at("three", 4096).v)}·n, 2·ln n = ${SX.f2(2 * Math.log(4096))}, 1.39·log₂n = ${SX.f2(1.39 * 12)}; sorted input with the last pivot ${SX.f1(at("sortedLast", 4096).v)}·n = ${SX.int(at("sortedLast", 4096).v * 4096)} · Hoare ÷ Lomuto: ${SX.f2(at("hoare", 64).v / at("lom", 64).v)} at n = 64, ${SX.f2(at("hoare", 4096).v / at("lom", 4096).v)} at n = 4096 — the ratio falls toward 1 as the shared 2n·ln n term dominates`);
  }
  SX.on("qc-runs", "input", () => { SX.setText("qc-runs-out", SX.val("qc-runs", 30)); build(); });
  SX.setText("qc-runs-out", SX.val("qc-runs", 30));
  build();
})();

/* ── 11  #qd-svg  recursion-depth distribution over random pivots ───────── */
(function () {
  if (!SX.has("qd-svg")) return;
  const W = 680, H = 260;
  function build() {
    const n = +SX.val("qd-n", 1024), T = 1000;
    const hA = {}, hB = {}; let mA = 0, mB = 0, sA = 0, sB = 0;
    for (let t = 0; t < T; t++) {
      const a = AL.perm(n, AL.rng(11 * n + t));
      const c1 = AL.counter(); SS.quickSort(a.slice(), c1, { scheme: "lomuto", pivot: "random", seed: t + 1 });
      const c2 = AL.counter(); SS.quickSort(a.slice(), c2, { scheme: "lomuto", pivot: "random", seed: t + 1, smallerFirst: true });
      const d1 = c1.get("depth"), d2 = c2.get("depth");
      hA[d1] = (hA[d1] || 0) + 1; hB[d2] = (hB[d2] || 0) + 1; mA = Math.max(mA, d1); mB = Math.max(mB, d2); sA += d1; sB += d2;
    }
    const cs1 = AL.counter(); SS.quickSort(SX.inputs.sorted(n), cs1, { scheme: "lomuto", pivot: "last" });
    const cs2 = AL.counter(); SS.quickSort(SX.inputs.sorted(n), cs2, { scheme: "lomuto", pivot: "last", smallerFirst: true });
    const lim = 2 * Math.floor(Math.log2(n));
    const ci = AL.counter(); const bi = SS.quickSort(SX.inputs.sorted(n), ci, { scheme: "lomuto", pivot: "last", depthLimit: lim });
    const F = AL.frame("#qd-svg", W, H, { l: 50, r: 16, t: 14, b: 40 });
    const maxD = Math.max(mA, mB) + 1, bound = Math.floor(Math.log2(n)) + 1;
    const x = d3.scaleLinear().domain([0, maxD]).range([0, F.iw]);
    const y = d3.scaleLinear().domain([0, Math.max(d3.max(Object.values(hA)), d3.max(Object.values(hB)))]).range([F.ih, 0]).nice();
    AL.gridY(F.g, y, F.iw); AL.axisB(F.g, x, 10, "maximum recursion depth (partitioning frames live at once)"); AL.axisL(F.g, y, 5, "runs out of " + T);
    const bw = x(1) - x(0);
    Object.entries(hA).forEach(([d, v]) => F.g.append("rect").attr("x", x(+d) - bw * 0.45).attr("y", y(v)).attr("width", bw * 0.42).attr("height", F.ih - y(v)).attr("fill", AC.accent).attr("rx", 2));
    Object.entries(hB).forEach(([d, v]) => F.g.append("rect").attr("x", x(+d) + bw * 0.03).attr("y", y(v)).attr("width", bw * 0.42).attr("height", F.ih - y(v)).attr("fill", AC.good).attr("rx", 2));
    F.g.append("line").attr("x1", x(bound)).attr("x2", x(bound)).attr("y1", 0).attr("y2", F.ih).attr("stroke", AC.a2).attr("stroke-dasharray", "4 3");
    F.g.append("text").attr("x", x(bound) + 3).attr("y", 12).attr("font-size", 10).attr("fill", AC.a2).text(`floor(log₂n) + 1 = ${bound}: the smaller-first guarantee`);
    AL.legend(F.g, [{ label: "recurse on the left part, loop on the right (naive)", color: AC.accent }, { label: "recurse on the SMALLER part, loop on the larger", color: AC.good }], F.iw - 300, 30);
    SX.setHtml("qd-readout", `n = ${n}, ${T} random permutations, random pivot · naive: max depth <b>${mA}</b>, mean ${SX.f2(sA / T)} · smaller-first: max <b>${mB}</b>, mean ${SX.f2(sB / T)}, never above floor(log₂n) + 1 = ${bound} ${SX.flag(mB <= bound)} · sorted input, last-element pivot: naive depth <b>${cs1.get("depth")}</b> frames (= n − 1 ${SX.flag(cs1.get("depth") === n - 1)}) against smaller-first <b>${cs2.get("depth")}</b> — with the SAME ${SX.int(cs2.get("cmp"))} comparisons: the trick bounds the stack, not the time · introsort with depth limit 2·floor(log₂n) = ${lim} on that input: ${SX.int(ci.get("cmp"))} comparisons, ${ci.get("heapFallbacks")} heapsort fallback, sorted ${SX.flag(SS.isSorted(bi))}`);
  }
  SX.on("qd-n", "change", build);
  build();
})();

/* ── §17  #co-table  the small-subarray cutoff sweep (a table, no svg) ──── */
(function () {
  if (d3.select("#co-table").empty()) return;
  const n = 4096, T = 10, cuts = [0, 4, 8, 12, 16, 24, 32, 64];
  const perms = Array.from({ length: T }, (_, t) => AL.perm(n, AL.rng(40 + t)));
  const res = cuts.map(M => {
    const r = { M };
    ["hoare", "lomuto"].forEach(s => {
      let sc = 0, sm = 0, ok = true;
      perms.forEach(a => { const c = AL.counter(); const b = SS.quickSort(a.slice(), c, { scheme: s, pivot: "median3", cutoff: M }); sc += c.get("cmp"); sm += c.get("mov"); ok = ok && SS.isSorted(b); });
      r[s] = { cmp: sc / T, mov: sm / T, tot: (sc + sm) / T, ok };
    });
    return r;
  });
  const best = s => res.reduce((m, r) => (r[s].tot < m[s].tot ? r : m), res[0]);
  const bh = best("hoare"), bl = best("lomuto");
  SX.table("#co-table", ["cutoff M", "Hoare + median-of-three: comparisons", "moves", "total", "vs best", "Lomuto + median-of-three: comparisons", "moves", "total", "vs best"],
    res.map(r => [r.M, SX.int(Math.round(r.hoare.cmp)), SX.int(Math.round(r.hoare.mov)), (r === bh ? "<b>" : "") + SX.int(Math.round(r.hoare.tot)) + (r === bh ? "</b>" : ""), "+" + SX.f1(100 * (r.hoare.tot / bh.hoare.tot - 1)) + "%",
                   SX.int(Math.round(r.lomuto.cmp)), SX.int(Math.round(r.lomuto.mov)), (r === bl ? "<b>" : "") + SX.int(Math.round(r.lomuto.tot)) + (r === bl ? "</b>" : ""), "+" + SX.f1(100 * (r.lomuto.tot / bl.lomuto.tot - 1)) + "%"]));
  const okAll = res.every(r => r.hoare.ok && r.lomuto.ok);
  SX.setHtml("co-readout", `n = ${n}, mean over ${T} random permutations, the final insertion sort included · Hoare: best total at M = <b>${bh.M}</b>, ${SX.f1(100 * (1 - bh.hoare.tot / res[0].hoare.tot))}% below M = 0 · Lomuto: ${bl.M === 0 ? "best with <b>no cutoff at all</b> (M = 0); M = 8–16 costs about " + SX.f1(100 * (res.find(r => r.M === 12).lomuto.tot / bl.lomuto.tot - 1)) + "% more" : "best at M = <b>" + bl.M + "</b>, " + SX.f1(100 * (1 - bl.lomuto.tot / res[0].lomuto.tot)) + "% below M = 0"} · every hybrid re-checked sorted ${SX.flag(okAll)}`);
})();

/* ── 12  #hh-svg  the Θ(n log n) sorts head to head ─────────────────────── */
(function () {
  if (!SX.has("hh-svg")) return;
  const W = 680, H = 300;
  const NS = [64, 128, 256, 512, 1024, 2048, 4096, 8192];
  const algos = [
    ["merge", "mergesort (top-down, buffer)", AC.accent, (a, c) => SS.mergeSort(a, c)],
    ["quick", "quicksort (Hoare, median-of-three, cutoff 12)", AC.a2, (a, c) => SS.quickSort(a, c, { scheme: "hoare", pivot: "median3", cutoff: 12, smallerFirst: true })],
    ["quickL", "quicksort (Lomuto, random pivot, no cutoff)", AC.rose, (a, c) => SS.quickSort(a, c, { scheme: "lomuto", pivot: "random", seed: 9 })],
    ["heap", "heapsort (classic sift-down)", AC.violet, (a, c) => SS.heapSort(a, c)]
  ];
  let cache = null;
  function measure() {
    if (cache) return cache;
    const input = SX.val("hh-input", "random");
    cache = algos.map(al => ({ key: al[0], label: al[1], color: al[2], pts: [] }));
    NS.forEach(n => {
      const runs = n >= 4096 ? 2 : 4;
      const sums = algos.map(() => ({ cmp: 0, mov: 0 }));
      for (let t = 0; t < runs; t++) {
        const a = input === "random" ? AL.perm(n, AL.rng(50 * n + t)) : input === "sorted" ? SX.inputs.sorted(n) : input === "reversed" ? SX.inputs.reversed(n) : SX.inputs.fewUnique(n, 3 + t);
        algos.forEach((al, k) => { const c = AL.counter(); const b = al[3](a.slice(), c); if (!SS.isSorted(b)) throw new Error("not sorted: " + al[0]); sums[k].cmp += c.get("cmp"); sums[k].mov += c.get("mov"); });
      }
      cache.forEach((s, k) => s.pts.push({ n, cmp: sums[k].cmp / runs, mov: sums[k].mov / runs, nl: n * Math.log2(n) }));
    });
    return cache;
  }
  function build() {
    const what = SX.val("hh-what", "cmp");
    const S = measure();
    const F = AL.frame("#hh-svg", W, H, { l: 56, r: 16, t: 14, b: 40 });
    const x = d3.scaleLog().domain([64, 8192]).range([0, F.iw]);
    const y = d3.scaleLinear().domain([0, d3.max(S, s => d3.max(s.pts, p => p[what] / p.nl))]).range([F.ih, 0]).nice();
    AL.gridY(F.g, y, F.iw); AL.axisB(F.g, x, 8, "n (log scale)", d3.format("~s")); AL.axisL(F.g, y, 5, (what === "cmp" ? "comparisons" : "element moves") + " ÷ n·log₂n");
    S.forEach(s => {
      F.g.append("path").datum(s.pts).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", 2).attr("d", d3.line().x(p => x(p.n)).y(p => y(p[what] / p.nl)));
      F.g.selectAll(".h" + s.key).data(s.pts).join("circle").attr("cx", p => x(p.n)).attr("cy", p => y(p[what] / p.nl)).attr("r", 3).attr("fill", s.color);
    });
    [1, 2].forEach(v => { if (v <= y.domain()[1]) F.g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(v)).attr("y2", y(v)).attr("stroke", AC.muted).attr("stroke-dasharray", "4 3"); });
    AL.legend(F.g, S.map(s => ({ label: s.label, color: s.color })), 8, 14);
    const at = (k, n) => S.find(s => s.key === k).pts.find(p => p.n === n);
    const props = { merge: ["Θ(n log n) always", "yes", "Θ(n) buffer", "sequential"], quick: ["Θ(n²) without a depth limit", "no", "O(log n) stack", "sequential"], quickL: ["Θ(n²) on few-distinct keys (scheme); vanishing probability on distinct keys", "no", "O(log n) expected stack", "sequential"], heap: ["Θ(n log n) always", "no", "O(1)", "scattered"] };
    SX.table("#hh-table", ["algorithm", "comparisons at n = 8192", "÷ n·log₂n", "moves at n = 8192", "÷ n·log₂n", "worst case", "stable", "extra memory", "access pattern"],
      S.map(s => { const p = at(s.key, 8192); return [s.label, SX.int(Math.round(p.cmp)), SX.f2(p.cmp / p.nl), SX.int(Math.round(p.mov)), SX.f2(p.mov / p.nl)].concat(props[s.key]); }));
    SX.setHtml("hh-readout", `input: ${SX.val("hh-input", "random")} · at n = 8192, comparisons ÷ n·log₂n: mergesort <b>${SX.f2(at("merge", 8192).cmp / at("merge", 8192).nl)}</b>, quicksort (Hoare/median-of-three/cutoff) <b>${SX.f2(at("quick", 8192).cmp / at("quick", 8192).nl)}</b>, quicksort (Lomuto/random) <b>${SX.f2(at("quickL", 8192).cmp / at("quickL", 8192).nl)}</b>, heapsort <b>${SX.f2(at("heap", 8192).cmp / at("heap", 8192).nl)}</b> · moves ÷ n·log₂n: ${SX.f2(at("merge", 8192).mov / at("merge", 8192).nl)}, ${SX.f2(at("quick", 8192).mov / at("quick", 8192).nl)}, ${SX.f2(at("quickL", 8192).mov / at("quickL", 8192).nl)}, ${SX.f2(at("heap", 8192).mov / at("heap", 8192).nl)} · every output re-checked sorted · a swap counts as 3 moves; mergesort's moves are buffer-in plus merge-out`);
  }
  SX.on("hh-what", "change", build);
  SX.on("hh-input", "change", () => { cache = null; build(); });
  build();
})();

/* ── 13  #cs-svg  counting sort stepped: tally, prefix sums, backwards pass */
(function () {
  if (!SX.has("cs-svg")) return;
  const W = 680, H = 250;
  function build() {
    const preset = SX.val("cs-preset", "worked");
    const base = preset === "worked" ? [2, 5, 3, 0, 2, 3, 0, 3] : preset === "sorted" ? [0, 0, 2, 2, 3, 3, 3, 5] : [5, 5, 5, 5, 5, 5, 5, 5];
    const k = 6;
    /* tagged records so stability is visible: equal keys carry a, b, c … by input order */
    const seen = {};
    const a = base.map(v => { seen[v] = (seen[v] || 0) + 1; return { k: v, tag: seen[v], lab: v + (seen[v] > 1 || base.filter(x => x === v).length > 1 ? "abcdefgh"[seen[v] - 1] : "") }; });
    const c = AL.counter(), frames = [];
    frames.push({ phase: "start", cnt: new Array(k).fill(0), out: new Array(a.length).fill(null), j: -1, so: {} });
    const out = SS.countingSort(a, k, null, c, f => frames.push(Object.assign(f, { so: c.all() })));
    const stable = SS.isStable(out);
    const svg = d3.select("#cs-svg");
    SX.clearControls(svg.node());
    const render = (f, idx) => {
      const F = AL.frame(svg, W, H, { l: 80, r: 10, t: 14, b: 6 });
      const g = F.g;
      AL.row(g, a.map(x => x.lab), { x: 0, y: 0, w: 40, label: "input A", mark: (i) => (f.phase === "count" && i === f.j) || (f.phase === "place" && i === f.j) ? AC.a2 : null });
      AL.row(g, f.cnt, { x: 0, y: 60, w: 40, label: f.phase === "prefix" || f.phase === "place" || f.phase === "start" && idx > 0 ? "C (prefix)" : "C (counts)", mark: (i) => (f.phase === "count" && i === a[f.j].k) || (f.phase === "prefix" && i === f.i) || (f.phase === "place" && i === f.d) ? AC.violet : null });
      AL.row(g, f.out.map(x => x === null || x === undefined ? "" : x.lab), { x: 0, y: 120, w: 40, label: "output B", mark: (i) => f.phase === "place" && i === f.pos ? AC.good : (f.out[i] ? AC.panel : null) });
      const txt = f.phase === "start" ? "counts C[0..k) start at zero; k = 6 possible key values 0..5" :
        f.phase === "count" ? `tally: A[${f.j}] = ${a[f.j].k}, so C[${a[f.j].k}] becomes ${f.cnt[a[f.j].k]}` :
        f.phase === "prefix" ? `prefix sum: C[${f.i}] += C[${f.i - 1}] → ${f.cnt[f.i]} = number of keys ≤ ${f.i}` :
        `place, scanning A BACKWARDS: A[${f.j}] = ${a[f.j].lab}, C[${f.d}] was ${f.cnt[f.d] + 1} → decrement to ${f.cnt[f.d]}, write B[${f.pos}]`;
      g.append("text").attr("x", 0).attr("y", 180).attr("font-size", 11).attr("fill", AC.muted).text(txt);
      g.append("text").attr("x", 0).attr("y", 198).attr("font-size", 10).attr("fill", AC.muted).text(`step ${idx + 1} / ${frames.length} · ${f.so.tally || 0} tallies, ${f.so.prefix || 0} prefix additions, ${f.so.mov || 0} placements — and 0 key comparisons`);
    };
    AL.stepper(svg, { frames, render, delay: 700, label: "step" });
    SX.setHtml("cs-readout", `n = ${a.length}, k = ${k} · measured: ${c.get("tally")} tallies + ${c.get("prefix")} prefix additions + ${c.get("mov")} placements = ${c.get("tally") + c.get("prefix") + c.get("mov")} steps, against n + (k − 1) + n = ${2 * a.length + k - 1} ${SX.flag(c.get("tally") + c.get("prefix") + c.get("mov") === 2 * a.length + k - 1)} · comparisons between keys: <b>0</b> · output ${out.map(x => x.lab).join(" ")} · sorted ${SX.flag(SS.isSorted(out))} · equal keys in input order (stable) ${SX.flag(stable)}`);
  }
  SX.on("cs-preset", "change", build);
  build();
})();

/* ── 14  #rx-svg  LSD radix sort stepped by digit ────────────────────────── */
(function () {
  if (!SX.has("rx-svg")) return;
  const W = 680, H = 230;
  function build() {
    const preset = SX.val("rx-preset", "worked");
    const base = preset === "worked" ? [329, 457, 657, 839, 436, 720, 355] : preset === "ties" ? [120, 121, 21, 20, 220, 221, 320] : [999, 111, 555, 333, 777, 222, 888, 444, 666, 100];
    const a = base.map((v, i) => ({ k: v, tag: i }));
    const c = AL.counter(), frames = [];
    frames.push({ phase: "start", d: -1, a: a.slice(), so: {} });
    const out = SS.radixLSD(a, 3, 10, c, f => frames.push(Object.assign(f, { so: c.all() })));
    const ok = SS.isSorted(out), stable = SS.isStable(out);
    /* the MSD count on the same input, for the readout */
    const cm = AL.counter(); const outM = SS.radixMSD(a, 3, 10, cm);
    const svg = d3.select("#rx-svg");
    SX.clearControls(svg.node());
    const digitOf = (v, d) => Math.floor(v / Math.pow(10, d)) % 10;
    const render = (f, idx) => {
      const F = AL.frame(svg, W, H, { l: 90, r: 10, t: 14, b: 6 });
      const g = F.g;
      const cols = [AC.a2, AC.violet, AC.teal];
      /* the row: each key drawn with its digits, the digit just sorted on highlighted */
      const step = 60, w = 54;
      f.a.forEach((rec, i) => {
        const s = String(rec.k).padStart(3, "0");
        const cell = g.append("g").attr("transform", `translate(${i * step},0)`);
        cell.append("rect").attr("width", w).attr("height", 34).attr("rx", 4).attr("fill", AC.panel2).attr("stroke", AC.line);
        [0, 1, 2].forEach(pos => {
          const d = 2 - pos;   // digit index from the right
          cell.append("text").attr("x", 10 + pos * 17).attr("y", 23).attr("font-size", 15).attr("font-family", "monospace")
            .attr("fill", f.d >= 0 && d === f.d ? cols[d] : (f.d >= 0 && d < f.d ? AC.muted : AC.ink)).attr("font-weight", f.d >= 0 && d === f.d ? "bold" : "normal").text(s[pos]);
        });
        cell.append("text").attr("x", w / 2).attr("y", 48).attr("text-anchor", "middle").attr("font-size", 9).attr("fill", AC.muted).text("in " + rec.tag);
      });
      g.append("text").attr("x", -84).attr("y", 22).attr("font-size", 11).attr("fill", AC.muted).text(f.phase === "start" ? "input" : `after digit ${f.d}`);
      /* the invariant text */
      const inv = f.phase === "start" ? "LSD radix sort: sort by the LEAST significant digit first, stably; then the next; the last pass is on the most significant digit" :
        f.d === 0 ? "pass 1 (units): the array is sorted by the last digit — and by nothing else" :
        f.d === 1 ? "pass 2 (tens): sorted by the last TWO digits; keys with equal tens digit kept their units order because the digit sort was STABLE" :
        "pass 3 (hundreds): sorted by all three digits — done. Every pass preserved the order established by the passes before it";
      g.append("text").attr("x", 0).attr("y", 90).attr("font-size", 11).attr("fill", AC.muted).text(inv);
      /* check: is the array sorted by the low d+1 digits? */
      if (f.d >= 0) {
        const low = v => v % Math.pow(10, f.d + 1);
        let okLow = true; for (let i = 1; i < f.a.length; i++) if (low(f.a[i - 1].k) > low(f.a[i].k)) okLow = false;
        g.append("text").attr("x", 0).attr("y", 112).attr("font-size", 11).attr("fill", okLow ? AC.good : AC.bad).text(`invariant checked: sorted by the low ${f.d + 1} digit${f.d ? "s" : ""}: ${okLow ? "✓" : "✗"}`);
      }
      g.append("text").attr("x", 0).attr("y", 140).attr("font-size", 10).attr("fill", AC.muted).text(`step ${idx + 1} / ${frames.length} · so far ${f.so.pass || 0} passes, ${f.so.tally || 0} tallies, ${f.so.prefix || 0} prefix additions, ${f.so.mov || 0} placements · 0 key comparisons`);
    };
    AL.stepper(svg, { frames, render, delay: 1200, label: "step" });
    SX.setHtml("rx-readout", `n = ${a.length}, d = 3 digits, base 10 · measured: ${c.get("pass")} passes, ${c.get("tally")} tallies, ${c.get("prefix")} prefix additions, ${c.get("mov")} placements — work ${c.get("tally") + c.get("prefix") + c.get("mov")} against d·(2n + k − 1) = ${3 * (2 * a.length + 9)} ${SX.flag(c.get("tally") + c.get("prefix") + c.get("mov") === 3 * (2 * a.length + 9))} · sorted ${SX.flag(ok)} · stable ${SX.flag(stable)} · MSD on the same keys: ${cm.get("mov")} bucket placements, ${cm.get("buckets") / 10} recursive bucketings creating ${cm.get("buckets")} bucket arrays, result ${SS.isSorted(outM) ? "sorted ✓" : "NOT sorted"}`);
  }
  SX.on("rx-preset", "change", build);
  build();
})();

/* ── 15  #bk-svg  bucket sort: occupancy and cost on uniform vs skewed input */
(function () {
  if (!SX.has("bk-svg")) return;
  const W = 680, H = 250;
  function build() {
    const n = +SX.val("bk-n", 256);
    const r = AL.rng(3);
    const U = Array.from({ length: n }, () => r());
    const S = U.map(v => v * v * v);            // cubed: the same numbers, pushed toward 0
    const cu = AL.counter(), bu = SS.bucketSort(U, cu);
    const cs = AL.counter(), bs = SS.bucketSort(S, cs);
    const F = AL.frame("#bk-svg", W, H, { l: 46, r: 16, t: 14, b: 36 });
    const half = (F.iw - 20) / 2;
    const draw = (sizes, x0, col, title) => {
      const x = d3.scaleLinear().domain([0, n]).range([x0, x0 + half]);
      const y = d3.scaleLinear().domain([0, Math.max(d3.max(bu.sizes), d3.max(bs.sizes))]).range([F.ih, 0]).nice();
      if (x0 === 0) AL.axisL(F.g, y, 5, "elements in bucket");
      F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x).ticks(4));
      const bw = Math.max(1, half / n);
      sizes.forEach((s, i) => F.g.append("rect").attr("x", x(i)).attr("y", y(s)).attr("width", bw).attr("height", F.ih - y(s)).attr("fill", col));
      F.g.append("text").attr("x", x0 + half / 2).attr("y", -2).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", AC.muted).text(title);
    };
    draw(bu.sizes, 0, AC.accent, "uniform keys on [0, 1): bucket occupancy");
    draw(bs.sizes, half + 20, AC.bad, "the same keys cubed (skewed toward 0)");
    F.g.append("text").attr("x", F.iw).attr("y", F.ih + 30).attr("text-anchor", "end").attr("font-size", 11).attr("fill", AC.muted).text("bucket index (n buckets)");
    const sq = sizes => sizes.reduce((s, v) => s + v * v, 0);
    SX.setHtml("bk-readout", `n = ${n} keys into ${n} buckets, insertion sort inside each · uniform: largest bucket <b>${d3.max(bu.sizes)}</b>, ∑nᵢ² = ${SX.int(sq(bu.sizes))} (expected under uniformity n·(2 − 1/n) = ${SX.int(Math.round(n * (2 - 1 / n)))}), insertion-sort comparisons <b>${SX.int(cu.get("cmp"))}</b> · cubed: largest bucket <b>${d3.max(bs.sizes)}</b>, ∑nᵢ² = ${SX.int(sq(bs.sizes))}, comparisons <b>${SX.int(cs.get("cmp"))}</b> — ${SX.f1(cs.get("cmp") / cu.get("cmp"))}× the uniform cost on the same n · both outputs re-checked sorted ${SX.flag(SS.isSorted(bu.out) && SS.isSorted(bs.out))} · for scale, n·log₂n = ${SX.int(Math.round(n * Math.log2(n)))}`);
  }
  SX.on("bk-n", "change", build);
  build();
})();

/* ── 16  #bs-svg  binary search stepped, lo/mid/hi and the invariant ────── */
(function () {
  if (!SX.has("bs-svg")) return;
  const W = 680, H = 200;
  const A = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91];
  function build() {
    const x = +SX.val("bs-x", 56);
    const c = AL.counter(), frames = [];
    const res = SS.binarySearch(A, x, c, f => frames.push(Object.assign({}, f, { so: c.all() })));
    const linear = AL.counter(); const lr = SS.linearSearch(A, x, linear);
    /* the worst-case bound and an independent check of the answer */
    const bound = Math.floor(Math.log2(A.length)) + 1;
    const truth = A.indexOf(x);
    const svg = d3.select("#bs-svg");
    SX.clearControls(svg.node());
    const render = (f, idx) => {
      const F = AL.frame(svg, W, H, { l: 16, r: 10, t: 30, b: 6 });
      const g = F.g;
      const R = AL.row(g, A, { x: 0, y: 0, w: 44, mark: (i) => f.phase === "found" && i === f.mid ? AC.good : (i === f.mid ? AC.a2 : (i >= f.lo && i < f.hi ? AC.panel2 : AC.panel)) });
      /* brackets for [lo, hi) */
      if (f.hi > f.lo) {
        g.append("line").attr("x1", R.cellX(f.lo) - 22).attr("x2", R.cellX(f.hi - 1) + 22).attr("y1", -8).attr("y2", -8).attr("stroke", AC.accent).attr("stroke-width", 2);
        g.append("text").attr("x", R.cellX(f.lo) - 22).attr("y", -12).attr("font-size", 10).attr("fill", AC.accent).text(`lo = ${f.lo}`);
        g.append("text").attr("x", R.cellX(f.hi - 1) + 22).attr("y", -12).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.accent).text(`hi = ${f.hi}`);
      } else {
        g.append("text").attr("x", 0).attr("y", -12).attr("font-size", 10).attr("fill", AC.bad).text(`lo = hi = ${f.lo}: the range [lo, hi) is EMPTY`);
      }
      const txt = f.phase === "probe" ? `probe ${f.so.probe}: mid = ${f.lo} + floor((${f.hi} − ${f.lo})/2) = ${f.mid}, A[mid] = ${f.v} ${f.v < x ? "< " + x + " ⇒ lo = mid + 1 = " + (f.mid + 1) : f.v > x ? "> " + x + " ⇒ hi = mid = " + f.mid : "= " + x + " ⇒ found"}` :
        f.phase === "found" ? `found at index ${f.mid} after ${f.so.probe} probe${f.so.probe > 1 ? "s" : ""}` :
        `range empty after ${f.so.probe} probes: ${x} is ABSENT; lo = ${f.lo} is where it would be inserted`;
      g.append("text").attr("x", 0).attr("y", 78).attr("font-size", 11).attr("fill", AC.ink).text(txt);
      g.append("text").attr("x", 0).attr("y", 98).attr("font-size", 11).attr("fill", AC.muted).text(`invariant: if ${x} is in A at all, it is in A[lo .. hi) = A[${f.lo} .. ${f.hi})  —  ${f.hi > f.lo ? "range size " + (f.hi - f.lo) : "vacuous: nothing is in an empty range, so " + x + " is nowhere"}`);
      g.append("text").attr("x", 0).attr("y", 118).attr("font-size", 10).attr("fill", AC.muted).text(`step ${idx + 1} / ${frames.length}`);
    };
    AL.stepper(svg, { frames, render, delay: 900, label: "step" });
    SX.setHtml("bs-readout", `A = [${A.join(", ")}], n = ${A.length}, searching for ${x} · result <b>${res < 0 ? "absent" : "index " + res}</b> in <b>${c.get("probe")} probes</b> (worst case floor(log₂n) + 1 = ${bound} ${SX.flag(c.get("probe") <= bound)}) · independent check by linear scan: ${lr < 0 ? "absent" : "index " + lr} after ${linear.get("probe")} probes ${SX.flag((res < 0) === (lr < 0) && (res < 0 || A[res] === x))}${res < 0 ? " · insertion point " + frames[frames.length - 1].lo + ": A[" + (frames[frames.length - 1].lo - 1) + "] = " + (A[frames[frames.length - 1].lo - 1] === undefined ? "−∞" : A[frames[frames.length - 1].lo - 1]) + " < " + x + " < " + (A[frames[frames.length - 1].lo] === undefined ? "+∞" : A[frames[frames.length - 1].lo]) + " = A[" + frames[frames.length - 1].lo + "]" : ""}`);
  }
  SX.on("bs-x", "change", build);
  build();
})();

/* ── 17  #lb-svg  lower_bound / upper_bound stepped on an array with duplicates */
(function () {
  if (!SX.has("lb-svg")) return;
  const W = 680, H = 200;
  const A = [1, 2, 2, 2, 3, 5, 5, 8];
  function build() {
    const x = +SX.val("lb-x", 2), which = SX.val("lb-which", "lower");
    const c = AL.counter(), frames = [];
    const fn = which === "lower" ? SS.lowerBound : SS.upperBound;
    const res = fn(A, x, c, f => frames.push(Object.assign({}, f, { so: c.all() })));
    const c2 = AL.counter(); const other = (which === "lower" ? SS.upperBound : SS.lowerBound)(A, x, c2);
    const lb = which === "lower" ? res : other, ub = which === "lower" ? other : res;
    /* independent: linear scan definitions */
    const lbLin = (() => { for (let i = 0; i < A.length; i++) if (A[i] >= x) return i; return A.length; })();
    const ubLin = (() => { for (let i = 0; i < A.length; i++) if (A[i] > x) return i; return A.length; })();
    const svg = d3.select("#lb-svg");
    SX.clearControls(svg.node());
    const pred = which === "lower" ? `A[mid] < ${x}` : `A[mid] ≤ ${x}`;
    const render = (f, idx) => {
      const F = AL.frame(svg, W, H, { l: 16, r: 10, t: 30, b: 6 });
      const g = F.g;
      const R = AL.row(g, A, { x: 0, y: 0, w: 44, mark: (i) => f.phase === "done" ? (i === f.lo ? AC.good : (A[i] === x ? AC.violet : AC.panel)) : (i === f.mid ? AC.a2 : (i >= f.lo && i < f.hi ? AC.panel2 : AC.panel)) });
      if (f.hi > f.lo) {
        g.append("line").attr("x1", R.cellX(f.lo) - 22).attr("x2", R.cellX(f.hi - 1) + 22).attr("y1", -8).attr("y2", -8).attr("stroke", AC.accent).attr("stroke-width", 2);
        g.append("text").attr("x", R.cellX(f.lo) - 22).attr("y", -12).attr("font-size", 10).attr("fill", AC.accent).text(`lo = ${f.lo}`);
        g.append("text").attr("x", R.cellX(f.hi - 1) + 22).attr("y", -12).attr("text-anchor", "end").attr("font-size", 10).attr("fill", AC.accent).text(`hi = ${f.hi}`);
      }
      const txt = f.phase === "probe" ? `probe ${f.so.probe}: mid = ${f.mid}, A[mid] = ${f.v}: ${pred} is ${f.goRight ? "TRUE ⇒ the answer is right of mid: lo = " + (f.mid + 1) : "FALSE ⇒ mid could be the answer: hi = " + f.mid}` :
        `lo = hi = ${f.lo}: ${which === "lower" ? "lower_bound" : "upper_bound"}(${x}) = ${f.lo}${f.lo < A.length ? " — A[" + f.lo + "] = " + A[f.lo] + " is the first element " + (which === "lower" ? "≥" : ">") + " " + x : " — no element is " + (which === "lower" ? "≥" : ">") + " " + x + ", so the answer is n"}`;
      g.append("text").attr("x", 0).attr("y", 78).attr("font-size", 11).attr("fill", AC.ink).text(txt);
      g.append("text").attr("x", 0).attr("y", 98).attr("font-size", 11).attr("fill", AC.muted).text(`invariant: every index < lo has ${pred.replace("mid", "i")} ; every index ≥ hi does NOT ; the answer is in [lo, hi]`);
      g.append("text").attr("x", 0).attr("y", 118).attr("font-size", 10).attr("fill", AC.muted).text(`step ${idx + 1} / ${frames.length}`);
    };
    AL.stepper(svg, { frames, render, delay: 900, label: "step" });
    SX.setHtml("lb-readout", `A = [${A.join(", ")}], x = ${x} · <b>lower_bound = ${lb}</b>, <b>upper_bound = ${ub}</b>, so ${x} occurs <b>${ub - lb}</b> time${ub - lb === 1 ? "" : "s"}, at indices [${lb}, ${ub})${ub > lb ? "" : " — absent; " + lb + " is its insertion point"} · probes: ${c.get("probe")} for the ${which} bound, ${c2.get("probe")} for the other (each always floor(log₂n) + 1 = ${Math.floor(Math.log2(A.length)) + 1} or fewer, never early-exits) · independent check by linear scan: lower ${lbLin}, upper ${ubLin} ${SX.flag(lbLin === lb && ubLin === ub)}`);
  }
  ["lb-x", "lb-which"].forEach(id => SX.on(id, "change", build));
  build();
})();

/* ── 18  #ls-svg  linear vs binary vs interpolation vs exponential probes ─ */
(function () {
  if (!SX.has("ls-svg")) return;
  const W = 680, H = 320;
  const NS = [16, 64, 256, 1024, 4096, 16384, 65536];
  function build() {
    const dist = SX.val("ls-dist", "uniform");
    const S = { linear: [], binary: [], interp: [], expo: [], log2: [], loglog: [] };
    let interpMax = 0;
    NS.forEach(n => {
      const r = AL.rng(9);
      let A;
      if (dist === "uniform") A = Array.from({ length: n }, () => Math.floor(r() * n * 8)).sort((a, b) => a - b);
      else if (dist === "power") A = Array.from({ length: n }, () => Math.floor(n * 8 * Math.pow(r(), 4))).sort((a, b) => a - b);
      else A = Array.from({ length: n }, (_, i) => Math.floor(Math.pow(2, 40 * i / n))).sort((a, b) => a - b);
      const T = Math.min(n, 400);
      let l = 0, b = 0, ip = 0, ex = 0;
      for (let t = 0; t < T; t++) {
        const i = Math.floor(r() * n), x = A[i];
        const c1 = AL.counter(); const r1 = SS.linearSearch(A, x, c1); l += c1.get("probe");
        const c2 = AL.counter(); const r2 = SS.binarySearch(A, x, c2); b += c2.get("probe");
        const c3 = AL.counter(); const r3 = SS.interpolationSearch(A, x, c3); ip += c3.get("probe"); interpMax = Math.max(interpMax, c3.get("probe"));
        const c4 = AL.counter(); const r4 = SS.exponentialSearch(A, x, c4); ex += c4.get("probe");
        if (A[r1] !== x || A[r2] !== x || A[r3] !== x || A[r4] !== x) throw new Error("search returned a wrong index");
      }
      S.linear.push({ n, v: l / T }); S.binary.push({ n, v: b / T }); S.interp.push({ n, v: ip / T }); S.expo.push({ n, v: ex / T });
      S.log2.push({ n, v: Math.log2(n) }); S.loglog.push({ n, v: Math.log2(Math.log2(n)) });
    });
    const F = AL.frame("#ls-svg", W, H, { l: 56, r: 16, t: 14, b: 40 });
    const x = d3.scaleLog().domain([16, 65536]).range([0, F.iw]);
    const y = d3.scaleLog().domain([1, d3.max(S.linear, p => p.v)]).range([F.ih, 0]).nice();
    F.g.append("g").attr("class", "gridlines").selectAll("line").data(y.ticks(5)).join("line").attr("x1", 0).attr("x2", F.iw).attr("y1", d => y(d)).attr("y2", d => y(d)).attr("stroke", AC.grid);
    AL.axisB(F.g, x, 7, "n (log scale)", d3.format("~s")); AL.axisL(F.g, y, 5, "mean probes, successful search (log scale)", d3.format("~s"));
    const path = (pts, col, dash, w) => F.g.append("path").datum(pts).attr("fill", "none").attr("stroke", col).attr("stroke-width", w || 2).attr("stroke-dasharray", dash || null).attr("d", d3.line().x(p => x(p.n)).y(p => y(Math.max(1, p.v))));
    const dots = (pts, col) => F.g.selectAll(".s" + col.slice(1)).data(pts).join("circle").attr("cx", p => x(p.n)).attr("cy", p => y(Math.max(1, p.v))).attr("r", 3).attr("fill", col);
    path(S.log2, AC.muted, "4 3", 1.2); path(S.loglog, AC.muted, "1 3", 1.2);
    path(S.linear, AC.bad); dots(S.linear, AC.bad); path(S.binary, AC.accent); dots(S.binary, AC.accent);
    path(S.expo, AC.a2); dots(S.expo, AC.a2); path(S.interp, AC.good); dots(S.interp, AC.good);
    AL.legend(F.g, [{ label: "linear scan — ≈ n/2", color: AC.bad }, { label: "binary search — ≈ log₂n (dashed grey)", color: AC.accent },
                    { label: "exponential search from the front — ≈ 2·log₂(position)", color: AC.a2 },
                    { label: "interpolation search — ≈ log₂log₂n on uniform keys (dotted grey), far worse on skewed", color: AC.good }], 8, 14);
    const at = (s, n) => S[s].find(p => p.n === n);
    SX.setHtml("ls-readout", `keys: ${dist === "uniform" ? "uniform integers" : dist === "power" ? "skewed (uniform to the 4th power)" : "exponentially spaced"} · at n = 65 536, mean probes for a present key: linear <b>${SX.f1(at("linear", 65536).v)}</b>, binary <b>${SX.f2(at("binary", 65536).v)}</b> (log₂n = 16), exponential <b>${SX.f2(at("expo", 65536).v)}</b>, interpolation <b>${SX.f2(at("interp", 65536).v)}</b> (log₂log₂n = 4.00; worst single search seen ${SX.int(interpMax)} probes) · every result cross-checked against the target · 400 searches per size`);
  }
  SX.on("ls-dist", "change", build);
  build();
})();

})();
