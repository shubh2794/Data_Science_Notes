/* graph.js — extracted from index.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ============================================================
   Obsidian-style knowledge graph — one flat force web of data.js.
   Degree-sized nodes, neighbour spotlight, live physics, tunable.
   ============================================================ */
const svg = d3.select("#graph");
let width = window.innerWidth, height = window.innerHeight;
svg.attr("viewBox",[0,0,width,height]);
const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const defs = svg.append("defs");
defs.append("marker").attr("id","arrow").attr("viewBox","0 -5 10 10").attr("refX",14).attr("refY",0)
  .attr("markerWidth",8).attr("markerHeight",8).attr("orient","auto").attr("markerUnits","userSpaceOnUse")
  .append("path").attr("d","M0,-4L8,0L0,4").attr("fill","#8a8b94");
// soft glow for domain hubs + notes-bearing nodes (depth cue)
const glow = defs.append("filter").attr("id","glow").attr("x","-70%").attr("y","-70%").attr("width","240%").attr("height","240%");
glow.append("feGaussianBlur").attr("stdDeviation","2.4").attr("result","b");
const gm = glow.append("feMerge"); gm.append("feMergeNode").attr("in","b"); gm.append("feMergeNode").attr("in","SourceGraphic");

const root = svg.append("g");
/* Set once the reader pans or zooms by hand. A programmatic transform (our own
   fit / centre-on transitions) carries no sourceEvent, so it never trips this. */
let userMovedView = false;
const zoom = d3.zoom().scaleExtent([0.2,6]).on("zoom",e=>{
  if(e.sourceEvent) userMovedView = true;
  curK = e.transform.k;
  root.attr("transform",e.transform);
  applyLabels();
});
svg.call(zoom).on("dblclick.zoom",null);
svg.on("click",()=>{ detail.classList.remove("open"); unpin(); });
let curK = 1;

/* ---- data prep: flat node + link arrays ---- */
const domList = Object.keys(tree);
const topicDomain = {}; domList.forEach(d => tree[d].forEach(t => topicDomain[t]=d));
// level-3 sub-topics inherit their parent topic's domain slot
Object.entries(subtree).forEach(([top,kids]) => kids.forEach(k => topicDomain[k] = topicDomain[top]));
const degree = {}; nodes.forEach(n=> degree[n.id] = ADJ[n.id].size);

/* each domain gets an angular slot on a ring — a *gentle* anchor that fans the
   clusters apart so the dense cross-links don't collapse into a central hairball,
   while topology (links + charge) still does the real organic placement. */
const ANCHOR_R = 300;
const domAngle = {}; domList.forEach((d,i)=> domAngle[d] = i/domList.length*2*Math.PI - Math.PI/2);
function anchorOf(d){
  if(d.level===0) return [0,0];
  const dom = d.level===1 ? d.id : topicDomain[d.id];
  const a = domAngle[dom]; return [Math.cos(a)*ANCHOR_R, Math.sin(a)*ANCHOR_R];
}

function buildGraphData(){
  // The legend never filters — but the depth control deliberately does, because
  // hiding a whole level is how you get from 162 nodes to a readable overview.
  const N = nodes.filter(n=> n.level <= maxDepth).map(n=> Object.assign({}, n));
  const byId = {}; N.forEach(n=> byId[n.id]=n);
  // seed positions near each node's domain slot so the sim settles into clusters quickly
  N.forEach(n=>{ const [ax,ay]=anchorOf(n); n.x = ax + (Math.random()-.5)*60; n.y = ay + (Math.random()-.5)*60; });
  const L = [];
  Object.entries(tree).forEach(([dom,kids])=>{
    if(byId[dom]) L.push({source:"Data Science",target:dom,kind:"tree"});
    kids.forEach(k=>{ if(byId[k]&&byId[dom]) L.push({source:dom,target:k,kind:"tree"}); });
  });
  Object.entries(subtree).forEach(([top,kids])=>{
    kids.forEach(k=>{ if(byId[k]&&byId[top]) L.push({source:top,target:k,kind:"tree"}); });
  });
  cross.forEach(([s,t,kind])=>{ if(byId[s]&&byId[t]) L.push({source:s,target:t,kind:kind||"cross"}); });
  return {N,L};
}

/* node radius scales with connection count, like Obsidian */
let sizeScale = 1;        // "Node size" slider, 0.5x .. 2x
let maxDepth = 3;         // "Show down to": 1 domains, 2 topics, 3 sub-topics
let linkOpacity = 1;      // multiplier on every link's stroke opacity
function rN(n){
  if(n.level===0) return 13*sizeScale;
  const base = n.level===1 ? 6 : n.level===2 ? 3 : 2.4;   // sub-topics slightly smaller than topics
  return (base + Math.sqrt(degree[n.id]||1)*1.5) * sizeScale;
}
/* re-radius in place: cheaper and far less jarring than rebuilding the scene */
function applySize(){
  if(!nodeSel) return;
  nodeSel.select("circle").attr("r", rN);
  labelSel.attr("dy", d => -(rN(d)+5));
  if(sim){ sim.force("collide").radius(d=>rN(d)+5); sim.alpha(Math.max(sim.alpha(),0.08)).restart(); }
}

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
    c.onclick=(ev)=>{ ev.stopPropagation(); if(meta[nid]) navigateTo(meta[nid]); };
    chips.appendChild(c);
  });
  detail.classList.add("open");
}
document.getElementById("detail-close").onclick=()=>{ detail.classList.remove("open"); unpin(); };

/* ---- chip-hop history: browsing via Linked-notes chips can be walked back ---- */
const navStack = [];
const backBtn = document.getElementById("detail-back");
function updateBack(){ backBtn.style.display = navStack.length ? "block" : "none"; }
function navigateTo(d){ if(pinnedId && pinnedId!==d.id) navStack.push(pinnedId); activate(d); updateBack(); }
function resetNav(){ navStack.length=0; updateBack(); }
backBtn.onclick=(e)=>{
  e.stopPropagation();
  const id=navStack.pop();
  if(id && meta[id]) activate(meta[id]);
  updateBack();
};

let nodeSel, linkSel, labelSel, sim, N=[], L=[];
let pinnedId = null;   // node whose spotlight stays on after click, until you click elsewhere
function highlight(id){
  const nbr=ADJ[id]||new Set();
  nodeSel.classed("gdim", false); labelSel.style("opacity", null);   // a node spotlight takes over from the domain wash
  nodeSel.classed("dim", n=> n.id!==id && !nbr.has(n.id)).classed("pinned", n=> n.id===pinnedId);
  labelSel.style("display", n=> (n.id===id||nbr.has(n.id)) ? null : (labelVisible(n)?null:"none"));
  linkSel.classed("hot", l=> l.source.id===id||l.target.id===id)
    .classed("dim", l=> !(l.source.id===id||l.target.id===id));
}
function clearHighlight(){
  nodeSel.classed("dim",false); linkSel.classed("dim",false).classed("hot",false);
  // The resting state is whatever the legend says: a lit domain if one is selected,
  // a clean graph otherwise. applyGroupSel handles both, including clearing its own
  // classes when nothing is lit — so this must NOT be conditional on a selection.
  if(typeof applyGroupSel === "function") applyGroupSel(); else applyLabels();
}

/* click = pin the spotlight + open detail (same highlight as hover, but it persists) */
function activate(d){ pinnedId = d.id; selectNode(d); highlight(d.id); nodeSel.classed("pinned", n=> n.id===pinnedId); }
/* hover-out reverts to the pinned node's spotlight (or clears if nothing pinned) */
function restore(){ if(pinnedId && meta[pinnedId]) highlight(pinnedId); else clearHighlight(); }
/* clicking empty space / closing the panel releases the pin */
function unpin(){ pinnedId = null; nodeSel && nodeSel.classed("pinned",false); clearHighlight(); resetNav(); }

/* ---- label visibility (zoom + importance, tunable) ---- */
let labelThreshold = 0.45;  // 0..1 from slider; higher = labels appear sooner
function labelVisible(n){
  if(n.level<=1) return true;   // root + domain labels always shown — they orient the view
  // leaf labels fade in as you zoom, sooner for well-connected notes and at higher density
  const need = 1.55 - (degree[n.id]||1)*0.05 - labelThreshold*0.9;
  return curK > Math.max(0.85, need);
}
function applyLabels(){ if(labelSel) labelSel.style("display", n=> labelVisible(n)?null:"none"); }

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
const legtoggle = d3.select("#legtoggle");
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
  if(!nodeSel) return;
  if(!groupSelActive()){
    nodeSel.classed("gsel", false).classed("gdim", false);
    linkSel.classed("dim", false);
    labelSel.style("opacity", null);
    applyLabels();
    return;
  }
  const inSel = n => selectedGroups.has(n.group);
  nodeSel.classed("gsel", inSel).classed("gdim", n => !inSel(n));
  // a link stays lit only when BOTH ends are in the selection
  linkSel.classed("dim", l => !(selectedGroups.has(l.source.group) && selectedGroups.has(l.target.group)));
  // labels live in their own layer, so they need dimming of their own — otherwise
  // the faded half of the graph still shouts its names. Lit nodes are always named.
  labelSel.style("display", n => inSel(n) ? null : (labelVisible(n) ? null : "none"))
          .style("opacity",  n => inSel(n) ? 1 : 0.2);
}

/* Nodes with no page yet are the graph's scaffolding. Fading them lets you see
   what is actually written without losing the shape of the plan. */
let fadeStubs = false;
function applyStubFade(){ if(nodeSel) nodeSel.classed("stub-faded", n => fadeStubs && !n.link); }

function toggleGroup(key){
  if(selectedGroups.has(key)) selectedGroups.delete(key); else selectedGroups.add(key);
  paintGroupSel();
  restore();
}
function clearGroupSel(){ selectedGroups.clear(); paintGroupSel(); restore(); }

Object.entries(GROUPS).forEach(([key,g])=>{
  if(key==="root") return;
  const b = legtoggle.append("button").attr("type","button")
    .attr("title","Highlight " + g.label + " in the graph")
    .attr("aria-pressed","false")
    .on("click",function(){
      toggleGroup(key);
      this.setAttribute("aria-pressed", selectedGroups.has(key) ? "true" : "false");
    });
  b.append("span").attr("class","sw").style("background",g.color).style("color",g.color);
  b.append("span").text(g.short || g.label.split(" ")[0].replace("&",""));
  legBtns[key] = b.node();
});
paintGroupSel();
if(legClear) legClear.addEventListener("click", ()=>{
  clearGroupSel();
  Object.values(legBtns).forEach(el => el.setAttribute("aria-pressed","false"));
});

/* ---- search: highlight matches (title + description), dropdown, keyboard ---- */
const searchBox = document.getElementById("search");
const resultsEl = document.getElementById("search-results");
let searchMatches = [], searchSel = -1;

function centerOn(d, k){
  if(d.x==null || d.y==null) return;
  k = k || 1.5;
  svg.transition().duration(650).call(zoom.transform,
    d3.zoomIdentity.translate(width/2 - d.x*k, height/2 - d.y*k).scale(k));
}
function jumpTo(d){ if(!d) return; resetNav(); activate(d); centerOn(d); closeResults(); }
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
    nodeSel && nodeSel.classed("match",false);
    labelSel && labelSel.classed("match",false);
    restore(); applyLabels();
    clr.style.display="none"; cnt.textContent=""; closeResults(); return;
  }
  clr.style.display="block";
  searchMatches = N.map(n=>({n,r:rank(n,q)})).filter(o=>o.r<99)
    .sort((a,b)=> a.r-b.r || (degree[b.n.id]||0)-(degree[a.n.id]||0) || a.n.id.localeCompare(b.n.id))
    .map(o=>o.n);
  const hit = new Set(searchMatches.map(n=>n.id));
  // HIGHLIGHT the matches (accent ring + glow) and dim the rest
  nodeSel.classed("match", n=> hit.has(n.id)).classed("dim", n=> !hit.has(n.id));
  labelSel.classed("match", n=> hit.has(n.id)).style("display", n=> hit.has(n.id)?null:"none");
  linkSel.classed("dim",true).classed("hot",false);
  cnt.textContent = searchMatches.length+" match"+(searchMatches.length===1?"":"es");
  renderResults();
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
const $ = id => document.getElementById(id);
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
  if(!sim) return;
  sim.force("charge").strength(chargeStrength);
  sim.force("link").distance(linkDistance).strength(linkStrength);
  sim.force("x").strength(clusterStrength).x(d=>anchorOf(d)[0]);
  sim.force("y").strength(clusterStrength).y(d=>anchorOf(d)[1]);
  if(reheat) sim.alpha(0.45).restart();
}

/* ---- controls wiring + live value readouts ---- */
function setVal(id,v){ $(id).textContent = v; }
function syncReadouts(){ setVal("v-force",cfg.force()); setVal("v-link",cfg.link()); setVal("v-cluster",Math.round(cfg.cluster()*100)); setVal("v-label",$("c-label").value); setVal("v-size",$("c-size").value+"%"); setVal("v-linkop",$("c-linkop").value+"%"); }
$("c-force").addEventListener("input",()=>{ setVal("v-force",cfg.force()); applyForces(true); });
$("c-link").addEventListener("input",()=>{ setVal("v-link",cfg.link()); applyForces(true); });
$("c-cluster").addEventListener("input",()=>{ setVal("v-cluster",Math.round(cfg.cluster()*100)); applyForces(true); });
$("c-label").addEventListener("input",e=>{ labelThreshold = e.target.value/100; setVal("v-label",e.target.value); applyLabels(); });
$("c-arrows").addEventListener("change",e=>{
  linkSel.filter(l=>l.kind==="isa").attr("marker-end", e.target.checked?"url(#arrow)":null)
    .attr("stroke-dasharray", e.target.checked?"2 4":null);
});
$("c-colorlinks").addEventListener("change",e=>paintLinks(e.target.checked));

$("c-size").addEventListener("input",e=>{ sizeScale = +e.target.value/100; setVal("v-size", e.target.value+"%"); applySize(); });
$("c-linkop").addEventListener("input",e=>{ linkOpacity = +e.target.value/100; setVal("v-linkop", e.target.value+"%"); paintLinks($("c-colorlinks").checked); });
$("c-depth").addEventListener("change",e=>{ maxDepth = +e.target.value; build(); });
$("c-stubs").addEventListener("change",e=>{ fadeStubs = e.target.checked; applyStubFade(); });
$("c-freeze").addEventListener("change",e=>{
  if(!sim) return;
  if(e.target.checked){ sim.stop(); } else { sim.alpha(0.25).restart(); }
});
$("c-refit").addEventListener("click",()=>{ userMovedView = false; fitGraph(); });
$("c-reset").addEventListener("click",()=>{
  const d = {"c-force":1400,"c-link":160,"c-cluster":0,"c-label":45,"c-size":100,"c-linkop":100};
  Object.entries(d).forEach(([k,v])=>{ $(k).value = v; });
  $("c-depth").value = 3; $("c-arrows").checked = true; $("c-colorlinks").checked = true;
  $("c-stubs").checked = false; $("c-freeze").checked = false;
  sizeScale = 1; linkOpacity = 1; maxDepth = 3; fadeStubs = false;
  labelThreshold = 0.45;
  clearGroupSel();
  build();                       // depth may have changed, so rebuild rather than patch
  syncReadouts();
});

/* ============================================================
   build / rebuild
   ============================================================ */
let settleFitDone = false;
/* Re-frame the graph once the opening layout stops moving — but never steal the
   view back from a reader who has already panned or zoomed, and never re-fit on
   the later settles that follow dragging a node. Hence the one-way latch. */
function settleFit(){
  if(settleFitDone) return;
  settleFitDone = true;
  if(!userMovedView) fitGraph();
}
function build(){
  settleFitDone = false;
  if(sim){ sim.stop(); }
  root.selectAll("g.scene").remove();
  const scene = root.append("g").attr("class","scene");
  const data = buildGraphData(); N=data.N; L=data.L;

  const gLink = scene.append("g"), gNode = scene.append("g"), gLabel = scene.append("g");

  linkSel = gLink.selectAll("line").data(L).join("line")
    .attr("class",d=> "glink"+(d.kind==="isa"?" isa":""))
    .attr("stroke-width",d=> d.kind==="tree"?1.1:d.kind==="isa"?1.4:0.9)
    .attr("stroke-dasharray",d=> d.kind==="isa"?"2 4":null)
    .attr("marker-end",d=> d.kind==="isa"?"url(#arrow)":null);

  nodeSel = gNode.selectAll("g.node").data(N, d=>d.id).join("g").attr("class","node")
    .on("click",(e,d)=>{ e.stopPropagation(); resetNav(); activate(d); })
    .on("dblclick",(e,d)=>{ if(d.link) location.href=d.link; })
    .on("mouseover",(e,d)=>highlight(d.id)).on("mouseout",restore)
    .call(d3.drag()
      .on("start",(e,d)=>{ if(!e.active) sim.alphaTarget(.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on("drag",(e,d)=>{ d.fx=e.x; d.fy=e.y; })
      .on("end",(e,d)=>{ if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }));

  nodeSel.append("circle").attr("r",rN)
    .attr("fill",d=>GROUPS[d.group].color)
    .attr("fill-opacity",d=>d.level>=2 && !d.link?0.78:1)
    .attr("stroke",d=>d.link?"#ffffff":"#161618").attr("stroke-width",d=>d.link?1.5:1)
    .attr("filter",d=>(d.level<=1 || d.link)?"url(#glow)":null);  // hubs + notes glow softly

  nodeSel.append("title").text(d=>d.id);

  labelSel = gLabel.selectAll("text").data(N, d=>d.id).join("text").attr("class","nlabel")
    .attr("text-anchor","middle").attr("dy",d=>-(rN(d)+5))
    .attr("font-size",d=>d.level===0?14:d.level===1?12:d.level===2?9.5:8.5)
    .style("fill",d=>d.level===0?"#e8e9ed":d.level===1?GROUPS[d.group].color:"#c8c9ce")
    .text(d=>d.id);

  sim = d3.forceSimulation(N)
    .force("link", d3.forceLink(L).id(d=>d.id))
    .force("charge", d3.forceManyBody().distanceMax(620))
    .force("collide", d3.forceCollide().radius(d=>rN(d)+5).strength(0.85))
    .force("x", d3.forceX())
    .force("y", d3.forceY())
    .velocityDecay(0.42)
    .on("tick", ticked);
  applyForces(false);   // pull distance/strength/charge/cluster from the control panel
  paintLinks($("c-colorlinks").checked);   // after forceLink resolves source/target to node objects

  function ticked(){
    linkSel.attr("x1",l=>l.source.x).attr("y1",l=>l.source.y).attr("x2",l=>l.target.x).attr("y2",l=>l.target.y);
    nodeSel.attr("transform",d=>`translate(${d.x},${d.y})`);
    labelSel.attr("transform",d=>`translate(${d.x},${d.y})`);
    /* The opening fit is computed while the layout is still expanding, so by the
       time the graph settles it is framed wrong. Re-fit ONCE, as soon as the
       motion has mostly died down. */
    if(!settleFitDone && sim.alpha() < 0.12) settleFit();
  }

  // pre-warm for a settled first paint, then keep gentle live physics (Obsidian feel)
  for(let i=0;i<230;i++) sim.tick();
  ticked(); applyLabels();
  if(reduce){ sim.stop(); } else { sim.alpha(0.42).alphaDecay(0.022).restart(); }
  applyGroupSel();   // a lit domain survives a rebuild of the scene
  applyStubFade();
  fitGraph();
  // if this tab is throttled, ticks may never reach the alpha threshold above
  setTimeout(settleFit, 2500);
}

/* tree links tinted by their cluster's colour (constellations); cross/isa stay neutral */
function paintLinks(on){
  if(!linkSel) return;
  linkSel
    .attr("stroke", l=>{
      if(l.kind==="isa")   return "#9a9ba4";
      if(l.kind==="cross") return on ? GROUPS[l.source.group].color : "#3a3b42";
      return on ? GROUPS[l.target.group].color : "#3a3b42";
    })
    .attr("stroke-opacity", l=> linkOpacity * (
      l.kind==="isa" ? 0.72 : l.kind==="cross" ? (on?0.30:0.42) : (on?0.45:0.55)));
}

function fitGraph(){
  const xs=N.map(n=>n.x), ys=N.map(n=>n.y);
  const minx=d3.min(xs),maxx=d3.max(xs),miny=d3.min(ys),maxy=d3.max(ys);
  const w=Math.max(1,maxx-minx)+120, h=Math.max(1,maxy-miny)+120;
  const k=Math.min(1.0, 0.9*Math.min(width/w, height/h));
  const cx=(minx+maxx)/2, cy=(miny+maxy)/2;
  curK=k;
  svg.transition().duration(650).call(zoom.transform, d3.zoomIdentity.translate(width/2-cx*k,height/2-cy*k).scale(k));
}

window.addEventListener("resize",()=>{ width=window.innerWidth; height=window.innerHeight; svg.attr("viewBox",[0,0,width,height]); });

labelThreshold = +$("c-label").value/100;
syncReadouts();
build();

/* deep-link: index.html?focus=<id> */
(function(){
  const f=new URLSearchParams(location.search).get("focus");
  if(f&&meta[f]) setTimeout(()=>activate(meta[f]),500);
})();
