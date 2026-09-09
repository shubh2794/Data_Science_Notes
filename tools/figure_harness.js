#!/usr/bin/env node
/* figure_harness.js — run a notes page's figures headlessly and report anything that breaks.
 *
 * The per-page gate (verify_page.py) reads the HTML as text; it cannot tell you that a
 * figure threw on load, rendered nothing, or crashes when a slider reaches its maximum.
 * This does. It caught four real problems on vision/frequency-domain.html alone, including
 * a leakage demo whose weak tone sat exactly on a Dirichlet null so the figure silently
 * proved the opposite of its caption.
 *
 * usage:
 *   npm i --prefix /tmp/harness-deps jsdom canvas        # once; both are native-ish, ~1 min
 *   node tools/figure_harness.js <page.html> [depsDir]
 *     page.html  path relative to the ml-notes root, e.g. vision/features.html
 *     depsDir    where jsdom/canvas live (default /tmp/harness-deps)
 *
 * exits non-zero if any script threw, any control throws, or any <svg> was never drawn.
 */
const fs = require("fs");
const path = require("path");
const ROOT = path.resolve(__dirname, "..");

const page = process.argv[2];
const deps = process.argv[3] || "/tmp/harness-deps";
if (!page) { console.error("usage: node tools/figure_harness.js <page.html> [depsDir]"); process.exit(2); }

let JSDOM, createCanvas;
try {
  ({ JSDOM } = require(path.join(deps, "node_modules/jsdom")));
  ({ createCanvas } = require(path.join(deps, "node_modules/canvas")));
} catch (e) {
  console.error(`missing deps in ${deps}\n  npm i --prefix ${deps} jsdom canvas`);
  process.exit(2);
}

const html = fs.readFileSync(path.join(ROOT, page), "utf8");
const dom = new JSDOM(html, { pretendToBeVisual: true, runScripts: "outside-only" });
const w = dom.window;
global.window = w; global.document = w.document; global.navigator = w.navigator;
global.self = w; global.HTMLElement = w.HTMLElement;
global.requestAnimationFrame = cb => setTimeout(cb, 0);

/* jsdom has no 2-D canvas; several pages raster images into one rather than emitting
   one <rect> per pixel, which is unusable above ~2500 cells. */
const origCreate = w.document.createElement.bind(w.document);
w.document.createElement = function (tag) {
  if (String(tag).toLowerCase() === "canvas") {
    const c = createCanvas(1, 1);
    c.toDataURL = c.toDataURL || (() => "data:,");
    return c;
  }
  return origCreate(tag);
};

/* Load exactly the scripts the page loads, in the page's own order, skipping data.js and
   notes.js (they need the sidebar/graph and are not what we are testing). */
const srcs = [...w.document.querySelectorAll("script[src]")].map(s => s.getAttribute("src"));
const wanted = srcs.filter(s => !/\/?(data|notes)\.js$/.test(s));
const dir = path.dirname(path.join(ROOT, page));
const files = ["vendor/d3.min.js", ...wanted.map(s => path.relative(ROOT, path.resolve(dir, s)))]
  .filter((v, i, a) => a.indexOf(v) === i)
  .filter(f => fs.existsSync(path.join(ROOT, f)));
console.log("loading:", files.join(" → "));

let fatal = null;
try { (0, eval)(files.map(f => fs.readFileSync(path.join(ROOT, f), "utf8")).join("\n;\n")); }
catch (e) { fatal = e.stack.split("\n").slice(0, 6).join("\n   "); }
if (fatal) { console.log("TOP-LEVEL THROW:\n   " + fatal); process.exit(1); }

/* every <svg> must have been drawn into */
const svgs = [...w.document.querySelectorAll("svg")].map(s => s.id).filter(Boolean);
const empty = svgs.filter(id => {
  const el = w.document.getElementById(id);
  return el && el.childNodes.length === 0;
});
console.log(`svgs: ${svgs.length}` + (empty.length ? `  EMPTY (never drawn): ${empty.join(", ")}` : "  all drawn"));

/* every readout must say something */
const blankRo = [];
[...w.document.querySelectorAll(".readout")].forEach(r => {
  if (!(r.textContent || "").trim()) blankRo.push(r.id || "(unnamed)");
});
if (blankRo.length) console.log("EMPTY READOUTS: " + blankRo.join(", "));

/* drive every control through its range */
let bad = 0;
const fire = (el, type) => {
  try { el.dispatchEvent(new w.Event(type, { bubbles: true })); }
  catch (e) { bad++; console.log(`THROW ${el.id || el.tagName} ${type}: ${e.message} @ ${(e.stack||"").split("\n")[1]}`); }
};
[...w.document.querySelectorAll("select")].forEach(sel => {
  const orig = sel.value;
  [...sel.options].forEach(o => { sel.value = o.value; fire(sel, "change"); });
  sel.value = orig; fire(sel, "change");
});
[...w.document.querySelectorAll("input[type=range]")].forEach(r => {
  const orig = r.value, lo = +r.min, hi = +r.max, st = +(r.step || 1) || 1;
  const snap = v => lo + Math.round((v - lo) / st) * st;   // browsers sanitise to the step
  [lo, snap((lo + hi) / 2), hi].forEach(v => { r.value = String(v); fire(r, "input"); });
  r.value = orig; fire(r, "input");
});
[...w.document.querySelectorAll("input[type=checkbox]")].forEach(c => {
  c.checked = !c.checked; fire(c, "change"); c.checked = !c.checked; fire(c, "change");
});
[...w.document.querySelectorAll("input[type=number]")].forEach(c => { fire(c, "input"); fire(c, "change"); });
[...w.document.querySelectorAll("button")].forEach(b => fire(b, "click"));

/* NaN / undefined leaking into a readout is nearly always a real bug */
const nan = [];
[...w.document.querySelectorAll(".readout")].forEach(r => {
  const t = r.textContent || "";
  if (/\b(NaN|undefined|Infinity)\b/.test(t)) nan.push(`${r.id}: ${t.replace(/\s+/g," ").slice(0,90)}`);
});
if (nan.length) console.log("BAD VALUES IN READOUTS:\n  " + nan.join("\n  "));

console.log(bad ? `${bad} CONTROL THROWS` : "all controls exercised cleanly");
process.exit((bad || empty.length || nan.length || blankRo.length) ? 1 : 0);
