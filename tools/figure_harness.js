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

let JSDOM, VirtualConsole, createCanvas;
try {
  ({ JSDOM, VirtualConsole } = require(path.join(deps, "node_modules/jsdom")));
  ({ createCanvas } = require(path.join(deps, "node_modules/canvas")));
} catch (e) {
  console.error(`missing deps in ${deps}\n  npm i --prefix ${deps} jsdom canvas`);
  process.exit(2);
}

const html = fs.readFileSync(path.join(ROOT, page), "utf8");
/* jsdom does NOT rethrow an exception thrown inside an event listener: dispatchEvent
   returns normally and the error goes to the virtual console as a "jsdomError". Without
   this hook a control handler that throws still reads "all controls exercised cleanly". */
let handlerErrs = 0;
const vc = new VirtualConsole();
vc.forwardTo(console, { jsdomErrors: "none" });   // jsdom ≥ 25 API (was sendTo/omitJSDOMErrors)
vc.on("jsdomError", e => {
  handlerErrs++;
  const err = e && e.detail || e;
  const frame = ((err && err.stack) || "").split("\n").find(l => /\.viz\.js|viz\.js/.test(l)) || "";
  console.log(`HANDLER THROW: ${err && err.message || e.message || e} @ ${frame.trim()}`);
});
const dom = new JSDOM(html, { pretendToBeVisual: true, runScripts: "outside-only", virtualConsole: vc });
const w = dom.window;
global.window = w; global.document = w.document; global.navigator = w.navigator;
global.self = w; global.HTMLElement = w.HTMLElement;
global.requestAnimationFrame = cb => setTimeout(cb, 0);
global.cancelAnimationFrame = id => clearTimeout(id);
w.requestAnimationFrame = global.requestAnimationFrame; w.cancelAnimationFrame = global.cancelAnimationFrame;

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

/* notes.js is skipped, but the older pages read its palette `C` (a plain const on its
   first lines). Preload just that declaration so those pages are tested, not the skip. */
const bodies = files.map(f => fs.readFileSync(path.join(ROOT, f), "utf8"));
const declaresC = bodies.some(b => /^\s*(const|let|var)\s+C\b/m.test(b));   // a page with its own C wins
const palette = declaresC ? "" : (fs.readFileSync(path.join(ROOT, "notes.js"), "utf8").match(/^const C = \{[^\n]*\};/m) || [""])[0];
let fatal = null;
try { (0, eval)([palette, ...bodies].join("\n;\n")); }
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

bad += handlerErrs;
console.log(bad ? `${bad} CONTROL THROWS` : "all controls exercised cleanly");
process.exit((bad || empty.length || nan.length || blankRo.length) ? 1 : 0);
