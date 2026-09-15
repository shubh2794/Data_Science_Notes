/* graph.js — the landing knowledge graph, rendered in 3D.
   Loaded after data.js. Rendering comes from vendor/force-graph-3d.min.js
   (window.FG3D: 3d-force-graph + three + three-spritetext + d3-force-3d). */

/* ============================================================
   Obsidian-style knowledge graph — one flat force web of data.js,
   laid out in three dimensions. Degree-sized spheres, neighbour
   spotlight, live physics, tunable. Orbit to look around it.
   ============================================================ */
const { ForceGraph3D, THREE, SpriteText, force: F3 } = window.FG3D;
const BG = "#161618", ACCENT = "#a78bfa";
const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = id => document.getElementById(id);

/* ---- data prep: flat node + link arrays ---- */
const domList = Object.keys(tree);
const topicDomain = {}; domList.forEach(d => tree[d].forEach(t => topicDomain[t]=d));
// level-3 sub-topics inherit their parent topic's domain slot
Object.entries(subtree).forEach(([top,kids]) => kids.forEach(k => topicDomain[k] = topicDomain[top]));
const degree = {}; nodes.forEach(n=> degree[n.id] = ADJ[n.id].size);

/* One working copy per node, made once and reused across rebuilds, so a node
   that survives a depth change keeps its position and its scene object. */
const ALL = nodes.map(n => Object.assign({}, n));
const byIdAll = {}; ALL.forEach(n => byIdAll[n.id] = n);

/* each domain gets a slot on a sphere — a *gentle* anchor that fans the
   clusters apart so the dense cross-links don't collapse into a central hairball,
   while topology (links + charge) still does the real organic placement.
   Slots are spread evenly with a Fibonacci spiral. */
const ANCHOR_R = 320;
const domAnchor = {};
domList.forEach((d,i)=>{
  const n = domList.length, y = 1 - (i + 0.5) / n * 2, r = Math.sqrt(1 - y*y), th = i * 2.399963;
  domAnchor[d] = [Math.cos(th)*r*ANCHOR_R, y*ANCHOR_R, Math.sin(th)*r*ANCHOR_R];
});
function anchorOf(d){
  if(d.level===0) return [0,0,0];
  const dom = d.level===1 ? d.id : topicDomain[d.id];
  return domAnchor[dom] || [0,0,0];
}

function buildGraphData(){
  // The legend never filters — but the depth control deliberately does, because
  // hiding a whole level is how you get from 162 nodes to a readable overview.
  const N = ALL.filter(n=> n.level <= maxDepth);
  const byId = {}; N.forEach(n=> byId[n.id]=n);
  // seed positions near each node's domain slot so the sim settles into clusters quickly
  N.forEach(n=>{
    if(n.x!=null) return;   // keep where a previous build left it
    const [ax,ay,az]=anchorOf(n);
    n.x = ax + (Math.random()-.5)*60; n.y = ay + (Math.random()-.5)*60; n.z = az + (Math.random()-.5)*60;
  });
  const L = [];
  Object.entries(tree).forEach(([dom,kids])=>{
    if(byId[dom]) L.push({source:"Data Science",target:dom,kind:"tree"});
    kids.forEach(k=>{ if(byId[k]&&byId[dom]) L.push({source:dom,target:k,kind:"tree"}); });
  });
  Object.entries(subtree).forEach(([top,kids])=>{
    kids.forEach(k=>{ if(byId[k]&&byId[top]) L.push({source:top,target:k,kind:"tree"}); });
  });
  cross.forEach(([s,t,kind])=>{ if(byId[s]&&byId[t]) L.push({source:s,target:t,kind:kind||"cross"}); });
  return {nodes:N, links:L};
}
/* links carry ids until the engine resolves them to node objects; accept both */
const endOf = x => (typeof x === "object" && x) ? x : byIdAll[x];

/* node radius scales with connection count, like Obsidian */
let sizeScale = 1;        // "Node size" slider, 0.5x .. 2x
let maxDepth = 3;         // "Show down to": 1 domains, 2 topics, 3 sub-topics
let linkOpacity = 1;      // multiplier on every link's opacity
function rN(n){
  if(n.level===0) return 13*sizeScale;
  const base = n.level===1 ? 6 : n.level===2 ? 3 : 2.4;   // sub-topics slightly smaller than topics
  return (base + Math.sqrt(degree[n.id]||1)*1.5) * sizeScale;
}

/* ============================================================
   scene objects — one Group per node: sphere + halo + label sprite.
   All later state changes (spotlight, legend, search, stubs) mutate
   these materials in place; nothing is rebuilt for a hover.
   ============================================================ */
const sphereGeo = new THREE.SphereGeometry(1, 24, 16);
const FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';
function labelHeight(n){ return n.level===0 ? 22 : n.level===1 ? 17 : n.level===2 ? 12 : 10.5; }   // world units; the layout spans ~1200
function labelColor(n){ return n.level===0 ? "#e8e9ed" : n.level===1 ? GROUPS[n.group].color : "#c8c9ce"; }

function nodeObject(n){
  const g = new THREE.Group();
  const col = new THREE.Color(GROUPS[n.group].color);
  const mat = new THREE.MeshLambertMaterial({ color: col, transparent: true, opacity: 1 });
  // hubs + notes-bearing nodes glow softly (a depth cue, like the SVG glow filter)
  mat.emissive = col.clone().multiplyScalar((n.level<=1 || n.link) ? 0.45 : 0.12);
  const sphere = new THREE.Mesh(sphereGeo, mat);
  g.add(sphere);

  // halo: the "ring" for pinned / matched / lit-domain states, plus a faint
  // glow for hubs and written notes. Additive blending reads as light on the dark ground.
  const haloMat = new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0,
    depthWrite: false, blending: THREE.AdditiveBlending });
  const halo = new THREE.Mesh(sphereGeo, haloMat);
  halo.raycast = () => {};   // never steals a hover from the sphere
  g.add(halo);

  const label = new SpriteText(n.id, labelHeight(n), labelColor(n));
  label.fontFace = FONT;
  label.fontWeight = n.level<=1 ? "600" : "500";
  label.strokeWidth = 0.55; label.strokeColor = BG;    // outlined, as the SVG labels were
  label.material.depthWrite = false;
  label.raycast = () => {};
  g.add(label);

  n.__sphere = sphere; n.__halo = halo; n.__label = label; n.__mat = mat; n.__haloMat = haloMat; n.__color = col;
  sizeNode(n);
  paintNode(n);
  return g;
}
function sizeNode(n){
  if(!n.__sphere) return;
  const r = rN(n);
  n.__sphere.scale.setScalar(r);
  n.__halo.scale.setScalar(r * 1.55);
  n.__label.position.y = r + labelHeight(n) * 0.65 + 1.5;
}

/* ---- per-node visual state -----------------------------------------------
   flags: _dim (outside a spotlight), _gdim/_gsel (domain legend), _match (search),
   _pinned (clicked). paintNode resolves them into material values. */
let fadeStubs = false;
let spotlightOn = false, searchOn = false;
function paintNode(n){
  if(!n.__mat) return;
  let op = (n.level>=2 && !n.link) ? 0.85 : 1;
  if(n._dim) op = 0.10;
  else if(n._gdim) op = 0.16;
  if(fadeStubs && !n.link) op *= 0.28;
  n.__mat.opacity = op;
  n.__mat.emissiveIntensity = (n._pinned || n._match) ? 1.6 : 1;

  const ring = n._pinned || n._match || n._gsel;
  const glow = n.level<=1 || n.link;
  n.__haloMat.color.set(n._gsel && !(n._pinned||n._match) ? "#ffffff" : (ring ? ACCENT : n.__color));
  n.__haloMat.opacity = n._gsel && !(n._pinned||n._match) ? 0.22 : ring ? 0.38 : (glow ? 0.14 : 0);
  if(n._dim) n.__haloMat.opacity *= 0.15;
  if(fadeStubs && !n.link && !ring) n.__haloMat.opacity *= 0.28;
}
function paintAll(){ N.forEach(paintNode); }

/* ---- links: colour carries the state (alpha = opacity) ---- */
/* plain hex → rgba string (NOT via THREE.Color, which would hand back linear-space channels) */
function rgba(hex, a){ const v = parseInt(hex.slice(1), 16); return `rgba(${v>>16&255},${v>>8&255},${v&255},${a.toFixed(3)})`; }
function linkColor(l){
  const on = $("c-colorlinks").checked, s = endOf(l.source), t = endOf(l.target);
  if(l._hot) return rgba(ACCENT, 0.95);
  let base, a;
  if(l.kind==="isa"){ base = "#b9bac4"; a = 0.72; }
  else if(l.kind==="cross"){ base = on ? GROUPS[s.group].color : "#5a5b63"; a = on ? 0.32 : 0.42; }
  else { base = on ? GROUPS[t.group].color : "#5a5b63"; a = on ? 0.48 : 0.55; }
  if(l._dim) a = 0.04;
  return rgba(base, Math.min(1, a * linkOpacity));
}
/* re-running the accessor is how 3d-force-graph refreshes link materials */
function repaintLinks(){ if(Graph){ Graph.linkColor(linkColor); Graph.linkDirectionalArrowColor(arrowColor); } }
function arrowColor(l){ return rgba("#b9bac4", l._dim ? 0.06 : 0.85); }

/* ---- detail panel ---- */
const detail = document.getElementById("detail");
function selectNode(d){
  const g = GROUPS[d.group];
  document.getElementById("d-kicker").textContent = g.label;
  document.getElementById("d-kicker").style.color = g.color;
  document.getElementById("d-title").textContent = d.id;
  document.getElementById("d-desc").textContent = d.desc || "";
  let html = d.link ? `<a class="opennotes" href="${d.link}">Open interactive notes →</a>`
                    : (d.level===1 ? "" : `<span class="soon">📝 Notes coming soon</span>`);
  document.getElementById("d-link").innerHTML = html;
  const chips = document.getElementById("d-chips"); chips.innerHTML="";
  Array.from(ADJ[d.id]||[]).forEach(nid=>{
    const c=document.createElement("span"); c.className="chip"; c.textContent=nid;
    c.onclick=(ev)=>{ ev.stopPropagation(); if(byIdAll[nid]) navigateTo(byIdAll[nid]); };
    chips.appendChild(c);
  });
  detail.classList.add("open");
}
document.getElementById("detail-close").onclick=()=>{ detail.classList.remove("open"); unpin(); };

/* ---- chip-hop history: browsing via Linked-notes chips can be walked back ---- */
const navStack = [];
const backBtn = document.getElementById("detail-back");
function updateBack(){ backBtn.style.display = navStack.length ? "block" : "none"; }
function navigateTo(d){ if(pinnedId && pinnedId!==d.id) navStack.push(pinnedId); activate(d); flyTo(d); updateBack(); }
function resetNav(){ navStack.length=0; updateBack(); }
backBtn.onclick=(e)=>{
  e.stopPropagation();
  const id=navStack.pop();
  if(id && byIdAll[id]){ activate(byIdAll[id]); flyTo(byIdAll[id]); }
  updateBack();
};

let Graph = null, N=[], L=[];
let pinnedId = null;   // node whose spotlight stays on after click, until you click elsewhere
let hoverId = null;    // node under the pointer, if any
function highlight(id){
  const nbr=ADJ[id]||new Set();
  spotlightOn = true;
  N.forEach(n=>{ n._gdim = false; n._dim = n.id!==id && !nbr.has(n.id); n._pinned = n.id===pinnedId; });   // a node spotlight takes over from the domain wash
  L.forEach(l=>{ const hot = endOf(l.source).id===id || endOf(l.target).id===id; l._hot = hot; l._dim = !hot; });
  paintAll(); repaintLinks(); updateLabels();
}
function clearHighlight(){
  spotlightOn = false;
  N.forEach(n=>{ n._dim = false; });
  L.forEach(l=>{ l._dim = false; l._hot = false; });
  // The resting state is whatever the legend says: a lit domain if one is selected,
  // a clean graph otherwise. applyGroupSel handles both, including clearing its own
  // flags when nothing is lit — so this must NOT be conditional on a selection.
  applyGroupSel();
}

/* click = pin the spotlight + open detail (same highlight as hover, but it persists) */
function activate(d){ pinnedId = d.id; selectNode(d); highlight(d.id); }
/* hover-out reverts to the pinned node's spotlight (or clears if nothing pinned) */
function restore(){ if(pinnedId && byIdAll[pinnedId]) highlight(pinnedId); else if(searchOn) applySearchPaint(); else clearHighlight(); }
/* clicking empty space / closing the panel releases the pin */
function unpin(){ pinnedId = null; N.forEach(n=> n._pinned=false); clearHighlight(); resetNav(); }

/* ---- label visibility (camera distance + importance, tunable) ----------
   In 3D "zoom level" is how close the camera is: a node's apparent scale is
   LABEL_REF / distance, which stands in for the SVG zoom factor k. */
let labelThreshold = 0.45;  // 0..1 from slider; higher = labels appear sooner
const LABEL_REF = 800;
const _v = new THREE.Vector3();
function nodeScaleK(n){
  if(!Graph) return 1;
  const cam = Graph.camera().position;
  const d = _v.set(n.x||0, n.y||0, n.z||0).distanceTo(cam);
  return LABEL_REF / Math.max(1, d);
}
function labelVisible(n){
  if(n.level<=1) return true;   // root + domain labels always shown — they orient the view
  // leaf labels fade in as you approach, sooner for well-connected notes and at higher density
  const need = 1.55 - (degree[n.id]||1)*0.05 - labelThreshold*0.9;
  return nodeScaleK(n) > Math.max(0.85, need);
}
/* Resolves every label's visibility + opacity from the current interaction
   state. Cheap (a distance per node), so it runs on every camera move. */
function updateLabels(){
  const groupSel = groupSelActive();
  const spot = spotlightOn ? (ADJ[hoverId || pinnedId] || new Set()) : null;
  const spotId = hoverId || pinnedId;
  N.forEach(n=>{
    const lab = n.__label; if(!lab) return;
    let show, op = 1;
    if(spot){ const inSpot = n.id===spotId || spot.has(n.id); show = inSpot || labelVisible(n); op = inSpot ? 1 : 0.25; }
    else if(searchOn){ show = !!n._match; }
    else if(groupSel){ show = n._gsel ? true : labelVisible(n); op = n._gsel ? 1 : 0.2; }
    else { show = labelVisible(n); }
    if(fadeStubs && !n.link) op *= 0.35;
    lab.visible = show;
    lab.material.opacity = op;
    lab.material.color.set(n._match ? ACCENT : "#ffffff");
  });
}

/* ---- collapsible panel sections ---------------------------------------
   Each <button class="sechead"> owns the .secbody immediately after it; the
   CSS keys off aria-expanded, so the state lives in one place. Remembered
   per browser, and failures to reach storage are non-fatal. */
function section(headId, storeKey, dflt){
  const head = document.getElementById(headId);
  if(!head) return;
  let open = dflt;
  try{ const v = localStorage.getItem(storeKey); if(v!==null) open = v === "1"; }catch(e){}
  const paint = () => head.setAttribute("aria-expanded", open ? "true" : "false");
  paint();
  head.addEventListener("click", ()=>{
    open = !open; paint();
    try{ localStorage.setItem(storeKey, open ? "1" : "0"); }catch(e){}
  });
}
section("settings-head", "kg.settings.open", true);
section("legend-head",   "kg.legend.open",   true);

/* ---- domain legend: SELECT to highlight, never to hide ------------------
   Clicking a domain spotlights its nodes and fades the rest; clicking again
   releases it. Several domains can be lit at once. Nothing is removed from
   the graph, so the shape of the whole map stays readable while you look at
   one part of it. */
const selectedGroups = new Set();
const legtoggle = document.getElementById("legtoggle");
const legBtns = {};
const legTally = document.getElementById("leg-tally");
const legClear = document.getElementById("leg-clear");

function groupSelActive(){ return selectedGroups.size > 0; }

function paintGroupSel(){
  Object.entries(legBtns).forEach(([k,el]) => el.classList.toggle("on", selectedGroups.has(k)));
  const n = selectedGroups.size;
  if(legTally) legTally.textContent = n ? n + " lit" : "";
  if(legClear) legClear.classList.toggle("show", n > 0);
}

/* the graph-side half of a domain selection */
function applyGroupSel(){
  if(!Graph) return;
  if(!groupSelActive()){
    N.forEach(n=>{ n._gsel = false; n._gdim = false; });
    L.forEach(l=>{ l._dim = false; });
  } else {
    const inSel = n => selectedGroups.has(n.group);
    N.forEach(n=>{ n._gsel = inSel(n); n._gdim = !inSel(n); });
    // a link stays lit only when BOTH ends are in the selection
    L.forEach(l=>{ l._dim = !(selectedGroups.has(endOf(l.source).group) && selectedGroups.has(endOf(l.target).group)); });
  }
  paintAll(); repaintLinks(); updateLabels();
}

/* Nodes with no page yet are the graph's scaffolding. Fading them lets you see
   what is actually written without losing the shape of the plan. */
function applyStubFade(){ paintAll(); updateLabels(); }

function toggleGroup(key){
  if(selectedGroups.has(key)) selectedGroups.delete(key); else selectedGroups.add(key);
  paintGroupSel();
  restore();
}
function clearGroupSel(){ selectedGroups.clear(); paintGroupSel(); restore(); }

Object.entries(GROUPS).forEach(([key,g])=>{
  if(key==="root") return;
  const b = document.createElement("button");
  b.type = "button"; b.title = "Highlight " + g.label + " in the graph"; b.setAttribute("aria-pressed","false");
  b.addEventListener("click", ()=>{ toggleGroup(key); b.setAttribute("aria-pressed", selectedGroups.has(key) ? "true" : "false"); });
  const sw = document.createElement("span"); sw.className = "sw"; sw.style.background = g.color; sw.style.color = g.color;
  const tx = document.createElement("span"); tx.textContent = g.short || g.label.split(" ")[0].replace("&","");
  b.append(sw, tx); legtoggle.appendChild(b);
  legBtns[key] = b;
});
paintGroupSel();
if(legClear) legClear.addEventListener("click", ()=>{
  clearGroupSel();
  Object.values(legBtns).forEach(el => el.setAttribute("aria-pressed","false"));
});

/* ---- camera: fly to a node, keeping the current viewing direction ---- */
let userMovedView = false;   // set once the reader orbits / zooms / pans by hand
function flyTo(d, ms){
  if(!Graph || d.x==null) return;
  settleFitDone = true;   // a deliberate focus must not be undone by the opening re-fit
  const cam = Graph.camera().position;
  const dir = new THREE.Vector3(cam.x - d.x, cam.y - d.y, cam.z - d.z);
  if(dir.lengthSq() < 1) dir.set(0, 0, 1);
  dir.normalize().multiplyScalar(620);   // close enough to read the neighbourhood, far enough to keep it in frame
  Graph.cameraPosition({ x: d.x + dir.x, y: d.y + dir.y, z: d.z + dir.z }, { x: d.x, y: d.y, z: d.z }, ms == null ? 800 : ms);
}

/* ---- search: highlight matches (title + description), dropdown, keyboard ---- */
const searchBox = document.getElementById("search");
const resultsEl = document.getElementById("search-results");
let searchMatches = [], searchSel = -1;

function jumpTo(d){ if(!d) return; resetNav(); activate(d); flyTo(d); closeResults(); }
function closeResults(){ resultsEl.classList.remove("open"); resultsEl.innerHTML=""; searchSel=-1; }

/* lower rank = stronger match: exact title → prefix → title substring → description → domain */
function rank(n, q){
  const t=n.id.toLowerCase();
  if(t===q) return 0;
  if(t.startsWith(q)) return 1;
  if(t.includes(q)) return 2;
  if((n.desc||"").toLowerCase().includes(q)) return 3;
  if(((GROUPS[n.group]||{}).label||"").toLowerCase().includes(q)) return 4;
  return 99;
}
function runSearch(){
  const q = searchBox.value.trim().toLowerCase();
  const clr=document.getElementById("search-clear"), cnt=document.getElementById("search-count");
  if(!q){
    searchOn = false;
    N.forEach(n=> n._match=false);
    restore();
    clr.style.display="none"; cnt.textContent=""; closeResults(); return;
  }
  clr.style.display="block";
  searchMatches = N.map(n=>({n,r:rank(n,q)})).filter(o=>o.r<99)
    .sort((a,b)=> a.r-b.r || (degree[b.n.id]||0)-(degree[a.n.id]||0) || a.n.id.localeCompare(b.n.id))
    .map(o=>o.n);
  const hit = new Set(searchMatches.map(n=>n.id));
  searchOn = true;
  N.forEach(n=> n._match = hit.has(n.id));
  applySearchPaint();
  cnt.textContent = searchMatches.length+" match"+(searchMatches.length===1?"":"es");
  renderResults();
}
/* HIGHLIGHT the matches (accent halo) and dim the rest */
function applySearchPaint(){
  spotlightOn = false;
  N.forEach(n=>{ n._dim = !n._match; n._gdim = false; n._pinned = false; });
  L.forEach(l=>{ l._dim = true; l._hot = false; });
  paintAll(); repaintLinks(); updateLabels();
}
function renderResults(){
  resultsEl.innerHTML="";
  const top = searchMatches.slice(0,8);
  if(!top.length){ closeResults(); return; }
  searchSel = 0;
  top.forEach((n,i)=>{
    const g=GROUPS[n.group]||{};
    const row=document.createElement("div");
    row.className="sr-row"+(i===0?" sel":""); row.setAttribute("role","option");
    row.innerHTML = `<span class="sr-dot" style="background:${g.color||'#888'}"></span>`+
      `<span class="sr-title">${n.id}</span>`+
      `<span class="sr-dom">${g.label||""}${n.link?"":" · soon"}</span>`;
    row.onmousedown=(e)=>{ e.preventDefault(); jumpTo(n); };   // fire before the input blurs
    row.onmouseenter=()=>{ searchSel=i; paintSel(); };
    resultsEl.appendChild(row);
  });
  resultsEl.classList.add("open");
}
function paintSel(){ [...resultsEl.children].forEach((r,i)=> r.classList.toggle("sel", i===searchSel)); }

searchBox.addEventListener("input", runSearch);
searchBox.addEventListener("focus", ()=>{ if(searchBox.value.trim() && searchMatches.length) renderResults(); });
searchBox.addEventListener("blur", ()=> setTimeout(closeResults, 130));
searchBox.addEventListener("keydown", e=>{
  const top = searchMatches.slice(0,8);
  if(e.key==="ArrowDown"){ e.preventDefault(); if(top.length){ searchSel=(searchSel+1)%top.length; paintSel(); } }
  else if(e.key==="ArrowUp"){ e.preventDefault(); if(top.length){ searchSel=(searchSel-1+top.length)%top.length; paintSel(); } }
  else if(e.key==="Enter"){ e.preventDefault(); jumpTo(top[searchSel<0?0:searchSel] || searchMatches[0]); }
  else if(e.key==="Escape"){ searchBox.value=""; runSearch(); searchBox.blur(); }
});
document.getElementById("search-clear").addEventListener("click",()=>{ searchBox.value=""; runSearch(); searchBox.focus(); });

/* ---- force config: the control panel is the single source of truth ---- */
const cfg = {
  force:   () => +$("c-force").value,
  link:    () => +$("c-link").value,
  cluster: () => +$("c-cluster").value/100,   // 0..1
};
function chargeStrength(d){ const f=cfg.force(); return d.level===0?-f*1.5 : d.level===1?-f*1.1 : -f*0.42; }
function linkDistance(l){ const v=cfg.link(); return l.kind==="tree"? v*0.62 : v*1.45; }   // tree short (tight clusters), cross long
function linkStrength(l){ return l.kind==="tree"?0.72 : l.kind==="isa"?0.05 : 0.035; }       // cross/isa weak → no hairball
function clusterStrength(d){ const c=cfg.cluster(); return d.level===0?0.18 : d.level===1?c*0.16 : c*0.06; }

function applyForces(reheat){
  if(!Graph) return;
  Graph.d3Force("charge").strength(chargeStrength);
  Graph.d3Force("link").distance(linkDistance).strength(linkStrength);
  Graph.d3Force("x").strength(clusterStrength).x(d=>anchorOf(d)[0]);
  Graph.d3Force("y").strength(clusterStrength).y(d=>anchorOf(d)[1]);
  Graph.d3Force("z").strength(clusterStrength).z(d=>anchorOf(d)[2]);
  if(reheat) Graph.d3ReheatSimulation();
}

/* ---- controls wiring + live value readouts ---- */
function setVal(id,v){ $(id).textContent = v; }
function syncReadouts(){ setVal("v-force",cfg.force()); setVal("v-link",cfg.link()); setVal("v-cluster",Math.round(cfg.cluster()*100)); setVal("v-label",$("c-label").value); setVal("v-size",$("c-size").value+"%"); setVal("v-linkop",$("c-linkop").value+"%"); }
$("c-force").addEventListener("input",()=>{ setVal("v-force",cfg.force()); applyForces(true); });
$("c-link").addEventListener("input",()=>{ setVal("v-link",cfg.link()); applyForces(true); });
$("c-cluster").addEventListener("input",()=>{ setVal("v-cluster",Math.round(cfg.cluster()*100)); applyForces(true); });
$("c-label").addEventListener("input",e=>{ labelThreshold = e.target.value/100; setVal("v-label",e.target.value); updateLabels(); });
$("c-arrows").addEventListener("change",()=> applyArrows());
$("c-colorlinks").addEventListener("change",()=> repaintLinks());

/* re-size in place: cheaper and far less jarring than rebuilding the scene */
function applySize(){
  N.forEach(sizeNode);
  if(Graph){ Graph.d3Force("collide").radius(d=>rN(d)+5); Graph.d3ReheatSimulation(); }
}
$("c-size").addEventListener("input",e=>{ sizeScale = +e.target.value/100; setVal("v-size", e.target.value+"%"); applySize(); });
$("c-linkop").addEventListener("input",e=>{ linkOpacity = +e.target.value/100; setVal("v-linkop", e.target.value+"%"); repaintLinks(); });
$("c-depth").addEventListener("change",e=>{ maxDepth = +e.target.value; build(); });
$("c-stubs").addEventListener("change",e=>{ fadeStubs = e.target.checked; applyStubFade(); });
$("c-freeze").addEventListener("change",e=>{
  if(!Graph) return;
  // cooldownTicks(0) parks the engine at the next tick; Infinity + reheat wakes it
  if(e.target.checked){ Graph.cooldownTicks(0); } else { Graph.cooldownTicks(Infinity); Graph.d3ReheatSimulation(); }
});
$("c-rotate").addEventListener("change",e=>{ if(Graph) Graph.controls().autoRotate = e.target.checked; });
$("c-refit").addEventListener("click",()=>{ userMovedView = false; fitGraph(650); });
$("c-reset").addEventListener("click",()=>{
  const d = {"c-force":1400,"c-link":160,"c-cluster":0,"c-label":45,"c-size":100,"c-linkop":100};
  Object.entries(d).forEach(([k,v])=>{ $(k).value = v; });
  $("c-depth").value = 3; $("c-arrows").checked = true; $("c-colorlinks").checked = true;
  $("c-stubs").checked = false; $("c-freeze").checked = false; $("c-rotate").checked = false;
  sizeScale = 1; linkOpacity = 1; maxDepth = 3; fadeStubs = false;
  labelThreshold = 0.45;
  if(Graph){ Graph.controls().autoRotate = false; Graph.cooldownTicks(Infinity); }
  clearGroupSel();
  build();                       // depth may have changed, so rebuild rather than patch
  syncReadouts();
});

function applyArrows(){
  if(!Graph) return;
  const on = $("c-arrows").checked;
  Graph.linkDirectionalArrowLength(l => (on && l.kind==="isa") ? 4.5 : 0);
}

/* ============================================================
   build / rebuild
   ============================================================ */
let settleFitDone = false, firstTickDone = false;
let lastClick = { id:null, t:0 };
/* Re-frame the graph once after the opening moments — but never steal the
   view back from a reader who has already orbited or zoomed, and never again
   after that (dragging a node must not yank the camera). Hence the one-way latch. */
function settleFit(){
  if(settleFitDone) return;
  settleFitDone = true;
  if(!userMovedView) fitGraph(650);
}
/* Frame the whole graph: back the camera off along its current line of sight
   until the layout's bounding sphere fills the narrower field of view. (The
   library's zoomToFit frames the bounding *box*, which leaves a cloud of nodes
   floating small in the middle of the screen.) */
function fitGraph(ms){
  if(!Graph || !N.length) return;
  const bb = Graph.getGraphBbox(); if(!bb) return;
  const c = { x:(bb.x[0]+bb.x[1])/2, y:(bb.y[0]+bb.y[1])/2, z:(bb.z[0]+bb.z[1])/2 };
  const R = 0.5 * Math.max(bb.x[1]-bb.x[0], bb.y[1]-bb.y[0], bb.z[1]-bb.z[0]) * 0.88;
  const cam = Graph.camera();
  const vfov = cam.fov * Math.PI/180, hfov = 2*Math.atan(Math.tan(vfov/2) * cam.aspect);
  const dist = Math.max(120, R / Math.sin(Math.min(vfov, hfov)/2) * 1.22);   // a little slack: near nodes project large
  const dir = new THREE.Vector3(cam.position.x - c.x, cam.position.y - c.y, cam.position.z - c.z);
  if(dir.lengthSq() < 1) dir.set(0, 0, 1);
  dir.normalize().multiplyScalar(dist);
  Graph.cameraPosition({ x:c.x+dir.x, y:c.y+dir.y, z:c.z+dir.z }, c, ms);
}

function makeGraph(){
  const el = document.getElementById("graph");
  Graph = new ForceGraph3D(el, { controlType: "orbit" })
    .backgroundColor(BG)
    .showNavInfo(false)
    .nodeId("id")
    .nodeThreeObject(nodeObject)
    .nodeLabel(()=>"")             // our own sprite labels + the detail panel replace the tooltip
    .linkColor(linkColor)
    .linkOpacity(1)                 // per-link alpha lives in linkColor
    .linkWidth(0)                   // 1px lines: crisp, and cheap for a few hundred links
    .linkDirectionalArrowRelPos(1)
    .linkDirectionalArrowColor(arrowColor)
    .warmupTicks(230)               // pre-warm for a settled first paint
    .cooldownTicks(Infinity)
    .cooldownTime(reduce ? 0 : 16000)
    .d3AlphaDecay(0.022)
    .d3VelocityDecay(0.42)
    .onNodeHover(n=>{ hoverId = n ? n.id : null; if(n) highlight(n.id); else restore(); })
    .onNodeClick(n=>{
      // 3d-force-graph has no dblclick; a quick second click on the same node opens its notes
      const now = Date.now();
      if(lastClick.id===n.id && now-lastClick.t < 380 && n.link){ location.href = n.link; return; }
      lastClick = { id:n.id, t:now };
      resetNav(); activate(n);
    })
    .onBackgroundClick(()=>{ detail.classList.remove("open"); unpin(); })
    .onEngineTick(()=>{
      // the warm-up has already settled the layout, so the first frame can be framed for real
      if(!firstTickDone){ firstTickDone = true; fitGraph(0); }
    });

  // forces that the 2D version had and 3d-force-graph does not add by itself
  Graph.d3Force("charge", F3.forceManyBody().distanceMax(620));
  Graph.d3Force("collide", F3.forceCollide().radius(d=>rN(d)+5).strength(0.85));
  Graph.d3Force("x", F3.forceX()); Graph.d3Force("y", F3.forceY()); Graph.d3Force("z", F3.forceZ());

  // depth cue: things far from the camera sink into the background (range follows the camera, see updateFog)
  Graph.scene().fog = new THREE.Fog(BG, 900, 2600);

  const controls = Graph.controls();
  controls.autoRotateSpeed = 0.6;
  /* Library quirk: when a node drag/click ends, 3d-force-graph dispatches a
     synthetic touch `pointerup` on the document to reset fly controls. Three's
     OrbitControls also listens on the document, can't match that fake pointer to
     the mouse pointer it is tracking, and throws. The real pointerup that follows
     does the actual cleanup, so the synthetic one can simply be swallowed. */
  document.addEventListener("pointerup", e=>{ if(!e.isTrusted && e.pointerType==="touch") e.stopImmediatePropagation(); }, true);
  controls.addEventListener("start", ()=>{ userMovedView = true; });
  controls.addEventListener("change", updateLabels);
}

function build(){
  settleFitDone = false; firstTickDone = false;
  const data = buildGraphData(); N = data.nodes; L = data.links;
  if(!Graph) makeGraph();
  Graph.graphData(data);
  applyForces(false);   // pull distance/strength/charge/cluster from the control panel
  applyArrows();
  applyGroupSel();      // a lit domain survives a rebuild of the scene (also paints nodes + links)
  // one gentle re-frame after any residual drift, unless the reader has taken the camera
  setTimeout(settleFit, 2500);
}

window.addEventListener("resize",()=>{ if(Graph) Graph.width(window.innerWidth).height(window.innerHeight); });

/* The fog band sits just behind the point the camera orbits: the near half of
   the graph stays crisp, the far half recedes — whatever the zoom level. */
function updateFog(){
  if(!Graph) return;
  const fog = Graph.scene().fog, cam = Graph.camera().position, tgt = Graph.controls().target;
  const d = cam.distanceTo(tgt);
  fog.near = d * 0.9; fog.far = d * 2.1;
}
/* labels + fog track the camera; the controls' change event covers user moves, this
   covers our own fly-to / fit tweens and the layout drifting under a still camera */
(function labelLoop(){
  let i = 0;
  (function tick(){ if(++i % 4 === 0){ updateFog(); updateLabels(); } requestAnimationFrame(tick); })();
})();

labelThreshold = +$("c-label").value/100;
syncReadouts();
build();

/* deep-link: index.html?focus=<id> */
(function(){
  const f=new URLSearchParams(location.search).get("focus");
  if(f&&byIdAll[f]) setTimeout(()=>{ activate(byIdAll[f]); flyTo(byIdAll[f]); }, 900);
})();
