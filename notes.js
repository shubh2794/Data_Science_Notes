/* notes.js — shared chrome for every topic notes page.
   Requires data.js (GROUPS, tree, subtree, PARENT, meta, nodeForPath) loaded first.
   Responsibilities:
     • C  — color palette used by the inline per-page viz scripts
     • build the sidebar (sibling topics + "On this page") from data.js
     • scrollspy that highlights the current section
   The page only needs:
     <nav class="sidebar" id="sidebar"></nav>   (empty; filled here)
   and to load:  ../data.js  →  ../notes.js  →  its own inline <script>. */

const C = { A:"#5b9cff", B:"#ffb454", good:"#4ade80", bad:"#f87171", ink:"#e6e9ef", muted:"#9aa3b2", line:"#2a2f3a" };

(function(){
  const nav = document.getElementById("sidebar");
  const cur = (typeof nodeForPath === "function") ? nodeForPath(location.pathname) : null;

  /* Depth-aware links. Every `link` in data.js is relative to the ml-notes root
     ("math/statistics/descriptive.html"), but a page may sit at any depth, so
     resolve each href against the CURRENT page's folder instead of assuming one
     level. For the depth-1 pages this returns exactly what the old code did:
     a sibling in the same folder collapses to its bare filename, the graph is
     "../index.html", a page in another folder is "../nlp/embeddings.html". */
  const rel = (to) => {
    const from = (cur && cur.link ? cur.link : "x.html").split("/").slice(0, -1);
    const t = to.split("/");
    let i = 0;
    while (i < from.length && i < t.length - 1 && from[i] === t[i]) i++;
    return "../".repeat(from.length - i) + t.slice(i).join("/");
  };
  const ROOT = () => rel("index.html");

  if (nav && cur) {
    const g = GROUPS[cur.group];
    const head = [
      `<div class="navtitle">${g.label}</div>`,
      `<div class="sub">Interactive notes · a collection from various sources</div>`,
      `<a class="tocitem graphlink" href="${ROOT()}?focus=${encodeURIComponent(cur.id)}"><span aria-hidden="true">🕸</span> Graph view</a>`,
    ];
    // On this page — section outline (primary, open by default)
    const secs = [...document.querySelectorAll("section.topic[id]")];
    const onpage = secs.length > 1
      ? secs.map(s => `<a class="onpage" href="#${s.id}">${sectionLabel(s)}</a>`).join("") : "";
    // sibling topics: same domain for a topic page; same parent (plus the parent hub) for a level-3 sub-topic page
    const parent = (typeof PARENT !== "undefined") ? PARENT[cur.id] : null;
    const sibIds = (cur.level === 3 && typeof subtree !== "undefined" && subtree[parent]) ? subtree[parent] : (tree[g.label] || []);
    const item = id => `<a class="tocitem${id===cur.id?" active":""}" href="${rel(meta[id].link)}">${id}</a>`;
    const sibs = (cur.level === 3 && meta[parent] && meta[parent].link ? `<a class="tocitem" href="${rel(meta[parent].link)}">↑ ${parent} · overview</a>` : "")
      + sibIds.filter(id => meta[id] && meta[id].link).map(item).join("");
    const sibCount = sibIds.filter(id => meta[id] && meta[id].link).length + (cur.level === 3 ? 1 : 0);
    // sub-topics (level 3) of a hub page
    const kids = (typeof subtree !== "undefined" && subtree[cur.id]) ? subtree[cur.id].filter(id => meta[id] && meta[id].link) : [];
    const subs = kids.map(item).join("");
    // connections (bidirectional links from data.js)
    const nbrs = (typeof ADJ !== "undefined") ? [...(ADJ[cur.id] || [])].filter(id => meta[id]).sort() : [];
    const conns = nbrs.map(id => { const n = meta[id];
      return n.link ? `<a class="tocitem" href="${rel(n.link)}">${id}</a>`
                    : `<a class="onpage" href="${ROOT()}?focus=${encodeURIComponent(id)}">${id} 🕸</a>`; }).join("");

    const groups = [];
    if (onpage) groups.push(grp("On this page", onpage, true, secs.length));
    if (subs)   groups.push(grp("Sub-topics", subs, true, kids.length));
    if (sibs)   groups.push(grp(cur.level === 3 ? parent : "Topics", sibs, cur.level === 3, sibCount));
    if (conns)  groups.push(grp("Connections", conns, false, nbrs.length));
    groups.push(grp("Local graph", `<svg id="local-graph" width="244" height="188" viewBox="0 0 244 188"></svg>`, false));

    nav.innerHTML = head.join("\n") + groups.join("");
    let lgRendered = false;
    nav.querySelectorAll(".group.toggle").forEach(h => h.addEventListener("click", () => {
      const open = h.parentElement.classList.toggle("open");
      h.setAttribute("aria-expanded", open ? "true" : "false");
      // lazily render the local graph the first time its group opens (avoids rendering into display:none)
      if (open && !lgRendered && h.parentElement.querySelector("#local-graph")) {
        lgRendered = true; try { renderLocalGraph(cur); } catch(e){}
      }
    }));
    try { setupChrome(); } catch(e){}
  }

  function grp(title, items, open, count){
    const badge = count ? `<span class="gcount">${count}</span>` : "";
    return `<div class="navgroup${open?" open":""}"><button type="button" class="group toggle" aria-expanded="${open?"true":"false"}"><span>${title}${badge}</span><span class="chev" aria-hidden="true">▸</span></button><div class="groupitems">${items}</div></div>`;
  }

  function setupChrome(){
    // skip-link + <main> landmark id
    const main = document.querySelector("main");
    if (main && !main.id) main.id = "main";
    if (main && !document.querySelector(".skiplink")){
      const sk = document.createElement("a");
      sk.className = "skiplink"; sk.href = "#main"; sk.textContent = "Skip to content";
      document.body.insertBefore(sk, document.body.firstChild);
    }
    // mobile hamburger + scrim → off-canvas sidebar
    if (document.querySelector(".navtoggle")) return;
    const btn = document.createElement("button");
    btn.className = "navtoggle"; btn.type = "button";
    btn.setAttribute("aria-label","Toggle navigation"); btn.setAttribute("aria-expanded","false");
    btn.innerHTML = "☰";
    const scrim = document.createElement("div"); scrim.className = "navscrim";
    const close = () => { document.body.classList.remove("navopen"); btn.setAttribute("aria-expanded","false"); };
    const open  = () => { document.body.classList.add("navopen"); btn.setAttribute("aria-expanded","true"); };
    btn.addEventListener("click", () => document.body.classList.contains("navopen") ? close() : open());
    scrim.addEventListener("click", close);
    document.addEventListener("keydown", e => { if (e.key === "Escape") close(); });
    document.querySelectorAll("nav.sidebar a").forEach(a => a.addEventListener("click", close));
    document.body.appendChild(btn); document.body.appendChild(scrim);
  }

  function renderLocalGraph(cur){
    if (typeof d3 === "undefined" || typeof ADJ === "undefined") return;
    const svg = d3.select("#local-graph"); if (svg.empty()) return;
    svg.selectAll("*").remove();
    const all = [...(ADJ[cur.id] || [])].filter(id => meta[id]);
    const nbrs = all.slice(0, 10);
    const W=244, H=188, cx=W/2, cy=H/2, R=64;
    const short = s => s.length > 15 ? s.slice(0,14) + "…" : s;
    const pos = nbrs.map((id,i)=>{ const a=(i/Math.max(nbrs.length,1))*2*Math.PI - Math.PI/2; return [cx+R*Math.cos(a), cy+R*Math.sin(a)]; });
    pos.forEach(p => svg.append("line").attr("x1",cx).attr("y1",cy).attr("x2",p[0]).attr("y2",p[1]).attr("stroke","#2a2f3a").attr("stroke-width",1));
    nbrs.forEach((id,i)=>{
      const n=meta[id], col=(GROUPS[n.group]||{}).color||"#888", p=pos[i], left=p[0] < cx-2;
      const a=svg.append("a").attr("href", n.link ? rel(n.link) : (ROOT()+"?focus="+encodeURIComponent(id)));
      a.append("circle").attr("cx",p[0]).attr("cy",p[1]).attr("r",5).attr("fill",col).attr("fill-opacity",n.link?0.95:0.45)
        .attr("stroke",n.link?"#fff":"#555").attr("stroke-width",1).style("cursor","pointer");
      a.append("title").text(id + (n.link?"":" (in graph)"));
      a.append("text").attr("x",p[0]+(left?-8:8)).attr("y",p[1]+3).attr("text-anchor",left?"end":"start")
        .attr("font-size",7.5).attr("fill",n.link?"#c2cad6":"#7b8494").style("cursor","pointer").text(short(id));
    });
    const cg=GROUPS[cur.group]||{};
    svg.append("circle").attr("cx",cx).attr("cy",cy).attr("r",7).attr("fill",cg.color||"#888").attr("stroke","#0f1117").attr("stroke-width",2).append("title").text(cur.id);
    svg.append("text").attr("x",cx).attr("y",cy-11).attr("text-anchor","middle").attr("font-size",8).attr("font-weight",700).attr("fill","#e6e9ef").attr("paint-order","stroke").attr("stroke","#171a23").attr("stroke-width",2).text(short(cur.id));
    if (all.length>nbrs.length) svg.append("text").attr("x",cx).attr("y",H-2).attr("text-anchor","middle").attr("font-size",8).attr("fill","#5d6675").text("+"+(all.length-nbrs.length)+" more");
  }

  // touch: draggable viz handles (cursor:grab set by the inline d3 scripts) must not
  // turn a finger-drag into a page scroll — runs after the inline scripts have built the vizzes
  window.addEventListener("load", () => {
    document.querySelectorAll(".viz svg *").forEach(el => {
      if (el.style && (el.style.cursor === "grab" || el.style.cursor === "pointer")) el.style.touchAction = "none";
    });
  });

  // scrollspy — highlight the section currently nearest the top (only real #anchors)
  const onpageLinks = [...document.querySelectorAll('nav.sidebar a.onpage[href^="#"]')];
  if (onpageLinks.length) {
    const secEls = [...document.querySelectorAll("section.topic[id]")];
    let ticking = false;
    function spy(){
      ticking = false;
      let curId = secEls.length ? secEls[0].id : null;
      for (const s of secEls){ if (s.getBoundingClientRect().top <= 130) curId = s.id; else break; }
      onpageLinks.forEach(a => a.classList.toggle("cur", a.getAttribute("href") === "#" + curId));
    }
    window.addEventListener("scroll", () => { if (!ticking){ ticking = true; requestAnimationFrame(spy); } }, { passive: true });
    onpageLinks.forEach(a => a.addEventListener("click", () => setTimeout(spy, 60)));
    spy();
  }

  // section label: explicit data-nav wins; else the <h3> text minus its index chip
  function sectionLabel(s){
    const explicit = s.getAttribute("data-nav");
    if (explicit) return explicit;
    const h = s.querySelector("h3");
    if (!h) return s.id;
    const clone = h.cloneNode(true);
    clone.querySelectorAll(".idx").forEach(n => n.remove());
    return clone.textContent.trim();
  }
})();
