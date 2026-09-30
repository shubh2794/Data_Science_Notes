/* donut.viz.js — every interactive figure on models/donut.html.
   Loaded after data.js / notes.js (palette C comes from notes.js when present).
   Each figure is its own IIFE-style block wrapped in try/catch so one failure cannot blank the rest.
   House rule: every number a figure displays is computed here from a spec or from a data
   array labelled with its source — never typed into a label. */

const DV = (function(){
  const P = Object.assign({}, (typeof C !== "undefined" ? C : {A:"#5b9cff",B:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"}),
    { panel:"#15181f", paper:"#e9e4d8", paperInk:"#2b2b2b", teal:"#2dd4bf", purple:"#c084fc", pink:"#f472b6", grey:"#6b7280" });

  function safe(name, fn){ try { fn(); } catch(e){ if (typeof console !== "undefined") console.error("[donut.viz] "+name+" failed:", e); } }
  function txt(g, x, y, s, o){
    o = o || {};
    return g.append("text").attr("x",x).attr("y",y).attr("font-size",o.size||11).attr("fill",o.fill||P.ink)
      .attr("text-anchor",o.anchor||"start").attr("font-family",o.mono?"SF Mono,Menlo,monospace":null)
      .attr("font-weight",o.bold?600:null).text(s);
  }
  function lcg(seed){ let s = seed >>> 0; return () => { s = (1664525*s + 1013904223) >>> 0; return s/4294967296; }; }
  function gauss(r){ const u = Math.max(1e-12, r()), v = r(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }
  function comma(x){ return Math.round(x).toLocaleString("en-US"); }
  function sci(x){ if (x === 0) return "0"; const e = Math.floor(Math.log10(Math.abs(x))); return (x/Math.pow(10,e)).toFixed(2)+"e"+e; }
  function bytes(b){ if (b >= 1e9) return (b/1e9).toFixed(1)+" GB"; if (b >= 1e6) return (b/1e6).toFixed(1)+" MB"; return (b/1e3).toFixed(1)+" kB"; }
  function $(id){ return document.getElementById(id); }
  function val(id){ const e = $(id); return e ? e.value : null; }
  function on(id, ev, fn){ const e = $(id); if (e) e.addEventListener(ev, fn); }

  /* ── Swin stage geometry and cost (Swin paper Eq. 1–2; MLP adds 8hwC² per block).
        Donut: patch 4, C₀ = 128, heads 4·2^(k−1), feature maps padded up to a multiple of M. ── */
  const DEPTHS = [2, 2, 14, 2];          // donut-base config.json "depths"
  function swinStages(H, W, M, depths, C0){
    depths = depths || DEPTHS; C0 = C0 || 128;
    const out = [];
    let h = Math.ceil(H/4), w = Math.ceil(W/4);
    for (let k = 0; k < 4; k++){
      if (k > 0){ h = Math.ceil(h/2); w = Math.ceil(w/2); }
      const c = C0 * Math.pow(2, k), heads = 4 * Math.pow(2, k);
      const hp = Math.ceil(h/M)*M, wp = Math.ceil(w/M)*M;
      const n = h*w, np = hp*wp;
      const wins = (hp/M)*(wp/M);
      const wmsa = 4*np*c*c + 2*M*M*np*c;
      const msa  = 4*n*c*c + 2*n*n*c;
      const mlp  = 8*n*c*c;
      out.push({k:k+1, h, w, hp, wp, n, np, c, heads, wins, wmsa, msa, mlp, blocks:depths[k],
        memW: wins*heads*M*M*M*M*2, memG: n*n*heads*2});
    }
    return out;
  }
  function encoderMACs(H, W, M){ return swinStages(H, W, M).reduce((s,st) => s + st.blocks*(st.wmsa + st.mlp), 0); }

  /* ── Page fitting, following transformers DonutImageProcessor: optional rotate, resize shortest edge
        to min(H, W) (int truncation), thumbnail into H × W, pad centred. Sizes are w × h here. ── */
  function fitPage(pw, ph, H, W, align){
    let w = pw, h = ph, rotated = false;
    if (align && ((W < H && w > h) || (W > H && w < h))){ const t = w; w = h; h = t; rotated = true; }
    const S = Math.min(H, W);
    let rw, rh;
    if (w <= h){ rw = S; rh = Math.floor(h * S / w); } else { rh = S; rw = Math.floor(w * S / h); }
    let tw = rw, th = rh;
    const hh = Math.min(rh, H), ww = Math.min(rw, W);
    if (!(hh === rh && ww === rw)){
      if (rh > rw){ th = hh; tw = Math.floor(rw * hh / rh); }
      else if (rw > rh){ tw = ww; th = Math.floor(rh * ww / rw); }
      else { tw = ww; th = hh; }
    }
    const scale = tw / w;
    return {w: tw, h: th, rotated, scale, padL: Math.floor((W - tw)/2), padT: Math.floor((H - th)/2), padFrac: 1 - (tw*th)/(W*H), rw, rh};
  }

  /* ── Sub-word splits produced by the released naver-clova-ix/donut-base tokenizer
        (XLMRobertaTokenizer, 57,525 entries), run once for every string used in these figures. ── */
  const TOK = {"Latte":["▁La","tte"],"2":["▁2"],"9,000":["▁9",",","000"],"Cookie":["▁Cookie"],"1":["▁1"],"2,500":["▁2",",","500"],
    "11,500":["▁11",",","500"],"20,000":["▁","20,000"],"8,500":["▁8",",","500"],
    "3002-Kyoto Choco Mochi":["▁300","2-","K","yo","to","▁Cho","co","▁Mo","chi"],"14,000":["▁1","4,000"],"28,000":["▁28,","000"],
    "50,000":["▁","50,000"],"4":["▁4"],
    "what is the price of choco mochi?":["▁what","▁is","▁the","▁price","▁of","▁cho","co","▁mo","chi","?"],
    "2017年11月15日":["▁2017","年","11","月","15","日"],"福田站":["▁","福","田","站"],"珂":["▁","珂"],"二等座":["▁二","等","座"],
    "广州南站":["▁","广州","南","站"],"C068987":["▁C","06","89","87"],"¥82.0元":["▁","¥","8","2.0","元"],"G79":["▁G","79"],
    "Le":["▁Le"],"Lette":["▁Let","te"],"memo":["▁me","mo"],"9,800":["▁9",",","800"]};
  /* Categorical value tokens added by the released train.py (RVL-CDIP classes, DocVQA yes/no). */
  const CAT = ["advertisement","budget","email","file_folder","form","handwritten","invoice","letter","memo","news_article",
    "presentation","questionnaire","resume","scientific_publication","scientific_report","specification","yes","no"];

  /* ── Ports of the released DonutModel.json2token / token2json ── */
  function json2token(obj, sort){
    if (obj !== null && typeof obj === "object" && !Array.isArray(obj)){
      const keys = Object.keys(obj);
      if (keys.length === 1 && keys[0] === "text_sequence") return String(obj.text_sequence);
      const ks = sort ? keys.slice().sort().reverse() : keys;
      return ks.map(k => "<s_"+k+">" + json2token(obj[k], sort) + "</s_"+k+">").join("");
    }
    if (Array.isArray(obj)) return obj.map(it => json2token(it, sort)).join("<sep/>");
    const s = String(obj);
    return CAT.indexOf(s) >= 0 ? "<"+s+"/>" : s;
  }
  function esc(s){ return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
  function token2json(tokens, inner){
    const output = {};
    let guard = 0;
    while (tokens && guard++ < 500){
      const st = tokens.match(/<s_(.*?)>/i);
      if (!st) break;
      const key = st[1], stTok = st[0];
      const et = tokens.match(new RegExp("</s_"+esc(key)+">", "i"));
      if (!et){ tokens = tokens.split(stTok).join(""); continue; }
      const etTok = et[0];
      const content = tokens.match(new RegExp(esc(stTok)+"([\\s\\S]*?)"+esc(etTok), "i"));
      if (content){
        const c = content[1].trim();
        if (c.indexOf("<s_") >= 0 && c.indexOf("</s_") >= 0){
          const v = token2json(c, true);
          if (v && v.length) output[key] = v.length === 1 ? v[0] : v;
        } else {
          let leaves = c.split("<sep/>").map(l => { l = l.trim(); const m = l.match(/^<([a-z_]+)\/>$/); return (m && CAT.indexOf(m[1]) >= 0) ? m[1] : l; });
          output[key] = leaves.length === 1 ? leaves[0] : leaves;
        }
      }
      tokens = tokens.slice(tokens.indexOf(etTok) + etTok.length).trim();
      if (tokens.slice(0,6) === "<sep/>") return [output].concat(token2json(tokens.slice(6), true));
    }
    if (Object.keys(output).length) return inner ? [output] : output;
    return inner ? [] : {text_sequence: tokens};
  }
  /* split a serialised string into display chunks: special tokens and text leaves */
  function chunks(seq){
    const parts = seq.split(/(<\/?s_[^>]*>|<sep\/>|<[a-z_]+\/>|<\/s>)/).filter(x => x !== "");
    return parts.map(p => {
      if (/^<\/?s_/.test(p)) return {t:p, kind:"sp"};
      if (p === "<sep/>" || /^<[a-z_]+\/>$/.test(p)) return {t:p, kind:"sp"};
      if (p === "</s>") return {t:p, kind:"ctl"};
      return {t:p, kind:"text", pieces: TOK[p] || null};
    });
  }

  /* ── The released JSONParseEvaluator (donut/util.py), ported: normalize_dict, flatten, cal_f1,
        construct_tree_from_dict, and zss tree edit distance with its costs. ── */
  function isDict(x){ return x !== null && typeof x === "object" && !Array.isArray(x); }
  function normalize(data){
    if (data === null || data === undefined || data === "" || (Array.isArray(data) && !data.length) || (isDict(data) && !Object.keys(data).length)) return {};
    if (isDict(data)){
      const out = {};
      Object.keys(data).sort((a,b) => (a.length - b.length) || (a < b ? -1 : a > b ? 1 : 0)).forEach(k => {
        let v = normalize(data[k]);
        const empty = (v === null) || (Array.isArray(v) && !v.length) || (isDict(v) && !Object.keys(v).length);
        if (!empty){ if (!Array.isArray(v)) v = [v]; out[k] = v; }
      });
      return out;
    }
    if (Array.isArray(data)){
      if (data.every(isDict)) return data.map(normalize).filter(x => Object.keys(x).length);
      return data.filter(x => ["string","number"].indexOf(typeof x) >= 0 && String(x).trim()).map(x => String(x).trim());
    }
    return [String(data).trim()];
  }
  function flatten(data){
    const out = [];
    (function rec(v, key){
      if (isDict(v)) Object.keys(v).forEach(k => rec(v[k], key ? key+"."+k : k));
      else if (Array.isArray(v)) v.forEach(it => rec(it, key));
      else out.push(key+"\u0000"+v);
    })(data, "");
    return out;
  }
  function f1(pred, gt){
    const p = flatten(normalize(pred)), a = flatten(normalize(gt));
    let tp = 0, fnfp = 0;
    p.forEach(f => { const i = a.indexOf(f); if (i >= 0){ tp++; a.splice(i,1); } else fnfp++; });
    fnfp += a.length;
    return (tp + fnfp/2) === 0 ? 0 : tp / (tp + fnfp/2);
  }
  function tree(data, name){
    const node = {label: name || "<root>", children: []};
    if (isDict(data)) Object.keys(data).forEach(k => node.children.push(tree(data[k], k)));
    else if (Array.isArray(data)){
      if (data.length && data.every(isDict)) data.forEach(it => node.children.push(tree(it, "<subtree>")));
      else data.forEach(it => node.children.push({label:"<leaf>"+it, children:[]}));
    }
    return node;
  }
  function isLeaf(n){ return n.label.indexOf("<leaf>") === 0; }
  function leafText(n){ return n.label.replace("<leaf>",""); }
  function lev(a, b){
    const m = a.length, n = b.length; if (!m) return n; if (!n) return m;
    let prev = []; for (let j=0;j<=n;j++) prev[j] = j;
    for (let i=1;i<=m;i++){ const cur = [i]; for (let j=1;j<=n;j++) cur[j] = Math.min(prev[j]+1, cur[j-1]+1, prev[j-1] + (a[i-1]===b[j-1]?0:1)); prev = cur; }
    return prev[n];
  }
  function insDel(n){ return isLeaf(n) ? leafText(n).length : 1; }
  function upd(a, b){
    const la = isLeaf(a), lb = isLeaf(b);
    if (la && lb) return lev(leafText(a), leafText(b));
    if (!la && lb) return 1 + leafText(b).length;
    if (la && !lb) return 1 + leafText(a).length;
    return a.label === b.label ? 0 : 1;
  }
  function prep(root){
    const nodes = [], l = [];
    (function rec(n){
      let lm = -1;
      n.children.forEach((c, k) => { const r = rec(c); if (k === 0) lm = r; });
      nodes.push(n); const idx = nodes.length - 1;
      l[idx] = lm === -1 ? idx : lm;
      return l[idx];
    })(root);
    const kr = [], seen = {};
    for (let i = nodes.length - 1; i >= 0; i--) if (!(l[i] in seen)){ kr.push(i); seen[l[i]] = 1; }
    kr.sort((a,b) => a-b);
    return {nodes, l, kr};
  }
  /* Zhang–Shasha tree edit distance (same algorithm as the zss package) */
  function ted(A, B){
    const a = prep(A), b = prep(B), n1 = a.nodes.length, n2 = b.nodes.length;
    const TD = []; for (let i=0;i<n1;i++){ TD.push(new Array(n2).fill(0)); }
    a.kr.forEach(i1 => { b.kr.forEach(j1 => {
      const li = a.l[i1], lj = b.l[j1], m = i1 - li + 2, n = j1 - lj + 2;
      const fd = []; for (let x=0;x<m;x++) fd.push(new Array(n).fill(0));
      for (let x=1;x<m;x++) fd[x][0] = fd[x-1][0] + insDel(a.nodes[x+li-1]);
      for (let y=1;y<n;y++) fd[0][y] = fd[0][y-1] + insDel(b.nodes[y+lj-1]);
      for (let x=1;x<m;x++) for (let y=1;y<n;y++){
        const p = x+li-1, q = y+lj-1;
        if (a.l[p] === li && b.l[q] === lj){
          fd[x][y] = Math.min(fd[x-1][y] + insDel(a.nodes[p]), fd[x][y-1] + insDel(b.nodes[q]), fd[x-1][y-1] + upd(a.nodes[p], b.nodes[q]));
          TD[p][q] = fd[x][y];
        } else {
          fd[x][y] = Math.min(fd[x-1][y] + insDel(a.nodes[p]), fd[x][y-1] + insDel(b.nodes[q]), fd[a.l[p]-li][b.l[q]-lj] + TD[p][q]);
        }
      }
    }); });
    return TD[n1-1][n2-1];
  }
  function tedAcc(pred, gt){
    const tp = tree(normalize(pred)), tg = tree(normalize(gt)), te = tree(normalize({}));
    const d = ted(tp, tg), d0 = ted(te, tg);
    return {d, d0, acc: d0 === 0 ? 0 : Math.max(0, 1 - d/d0)};
  }

  return {P, safe, txt, lcg, gauss, comma, sci, bytes, $, val, on, DEPTHS, swinStages, encoderMACs, fitPage,
    TOK, CAT, json2token, token2json, chunks, normalize, flatten, f1, tree, ted, tedAcc, lev};
})();

/* ───────────────────────── 02 · one receipt through two systems ───────────────────────── */
DV.safe("pipe", function(){
  const {P, txt, $, val, on, f1} = DV;
  const svg = d3.select("#pipe-svg");
  /* Receipt content from the Donut paper's Figure 1 (item, count, unit price, price; total and quantity),
     flattened to one level of keys for this figure. Word boxes are on a 1000 × 1400 mock receipt. */
  const WORDS = [
    {t:"3002-Kyoto", f:"name", line:1, x:60,  y:300}, {t:"Choco", f:"name", line:1, x:330, y:300}, {t:"Mochi", f:"name", line:1, x:500, y:300},
    {t:"2", f:"count", line:2, x:60, y:380}, {t:"14,000", f:"unitprice", line:2, x:380, y:380}, {t:"28,000", f:"price", line:2, x:700, y:380},
    {t:"TOTAL", f:null, line:3, x:60, y:700}, {t:"50,000", f:"total_price", line:3, x:700, y:700},
    {t:"QTY", f:null, line:4, x:60, y:780}, {t:"4", f:"menuqty_cnt", line:4, x:700, y:780}
  ];
  const GT = {item:{name:"3002-Kyoto Choco Mochi", count:"2", unitprice:"14,000", price:"28,000"}, total:{total_price:"50,000", menuqty_cnt:"4"}};
  function ocr(err){
    let w = WORDS.map(o => Object.assign({}, o));
    if (err === "sub")   w.forEach(o => { if (o.t === "28,000"){ o.t = "28,O00"; o.bad = true; } });
    if (err === "drop")  w = w.filter(o => o.t !== "Choco");
    if (err === "split") w.forEach(o => { if (o.t === "Mochi"){ o.line = 1.5; o.bad = true; } });
    if (err === "order"){ const i = w.findIndex(o => o.t === "Choco"), j = w.findIndex(o => o.t === "Mochi"); const t = w[i]; w[i] = w[j]; w[j] = t; w[i].bad = w[j].bad = true; }
    return w;
  }
  /* oracle tagger + grouping: a name word that starts a new line opens a new item group */
  function parse(words){
    const items = []; const total = {}; let cur = null, lastLine = null;
    words.forEach(o => {
      if (!o.f) { lastLine = o.line; return; }
      if (o.f === "total_price" || o.f === "menuqty_cnt"){ total[o.f] = o.t; lastLine = o.line; return; }
      if (o.f === "name" && (cur === null || (o.line !== lastLine && cur.name !== undefined && lastLine !== null && o.line !== 2))){ cur = {}; items.push(cur); }
      if (!cur){ cur = {}; items.push(cur); }
      if (o.f === "name") cur.name = cur.name ? cur.name + " " + o.t : o.t;
      else if (cur[o.f] === undefined) cur[o.f] = o.t;
      else { cur = {}; items.push(cur); cur[o.f] = o.t; }
      lastLine = o.line;
    });
    const out = {};
    out.item = items.length === 1 ? items[0] : items;
    out.total = total;
    return out;
  }
  function flatPairs(o){ const r = []; (function rec(v, k){ if (Array.isArray(v)) v.forEach(x => rec(x, k)); else if (v && typeof v === "object") Object.keys(v).forEach(kk => rec(v[kk], k ? k+"."+kk : kk)); else r.push([k, v]); })(o, ""); return r; }
  function draw(){
    svg.selectAll("*").remove();
    const err = val("pipe-err");
    const words = ocr(err), pred = parse(words);
    const g = svg.append("g");
    /* receipt */
    const rx = 10, ry = 22, rw = 130, rh = 182, sx = v => rx + v/1000*rw, sy = v => ry + v/1400*rh;
    txt(g, rx, 14, "page image", {size:10.5, fill:P.muted});
    g.append("rect").attr("x",rx).attr("y",ry).attr("width",rw).attr("height",rh).attr("fill",P.paper).attr("rx",2);
    WORDS.forEach(o => txt(g, sx(o.x), sy(o.y), o.t, {size:7.5, fill:P.paperInk, mono:true}));
    /* pipeline stages */
    const cols = [{x:160, t:"① OCR: words in reading order"}, {x:330, t:"② tagger (oracle) + grouping"}, {x:500, t:"③ JSON fields"}];
    cols.forEach(c => txt(g, c.x, 14, c.t, {size:10.5, fill:P.A}));
    words.forEach((o, i) => {
      txt(g, cols[0].x, 32 + i*16, (i+1)+". "+o.t + "  (line "+o.line+")", {size:10, fill:o.bad ? P.bad : P.ink, mono:true});
      txt(g, cols[1].x, 32 + i*16, o.t+" → "+(o.f || "O"), {size:10, fill:o.bad ? P.bad : P.ink, mono:true});
    });
    const pp = flatPairs(pred), gp = flatPairs(GT).map(p => p.join("="));
    pp.forEach((p, i) => { const ok = gp.indexOf(p.join("=")) >= 0;
      txt(g, cols[2].x, 32 + i*16, (ok ? "✓ " : "✗ ") + p[0] + ": " + p[1], {size:10, fill:ok ? P.good : P.bad, mono:true}); });
    /* OCR-free */
    g.append("line").attr("x1",10).attr("x2",650).attr("y1",222).attr("y2",222).attr("stroke",P.line);
    txt(g, 10, 240, "OCR-free: page image → Swin encoder → decoder → JSON (no OCR stage for errors to enter)", {size:10.5, fill:P.B});
    flatPairs(GT).forEach((p, i) => txt(g, 20 + (i%3)*215, 262 + Math.floor(i/3)*18, "✓ "+p[0]+": "+p[1], {size:10, fill:P.good, mono:true}));
    const fPipe = f1(pred, GT);
    const wrong = pp.filter(p => gp.indexOf(p.join("=")) < 0).length;
    $("pipe-read").textContent = "pipeline: " + pp.length + " fields produced, " + wrong + " wrong, field F1 = " + fPipe.toFixed(3) +
      " · OCR-free column: OCR errors cannot enter (field F1 against ground truth = " + f1(GT, GT).toFixed(3) + ", its own misreads not simulated)";
  }
  on("pipe-err", "change", draw);
  draw();
});

/* ───────────────────────── 04 · encoder → decoder block diagram ───────────────────────── */
DV.safe("arch", function(){
  const {P, txt, $, val, on, swinStages, comma} = DV;
  const svg = d3.select("#arch-svg");
  /* decoder settings from naver-clova-ix/donut-base config.json */
  const DEC = {layers:4, d:1024, heads:16, ffn:4096, vocab:57525};
  function draw(){
    svg.selectAll("*").remove();
    const [H, W] = val("arch-res").split("x").map(Number);
    const st = swinStages(H, W, 10);
    const g = svg.append("g");
    const bx = 10, by = 40;
    g.append("rect").attr("x",bx).attr("y",by).attr("width",70).attr("height",130).attr("fill",P.paper).attr("rx",3);
    txt(g, bx+35, by+60, "page", {anchor:"middle", fill:P.paperInk, size:11});
    txt(g, bx+35, by+76, H+" × "+W, {anchor:"middle", fill:P.paperInk, size:10});
    const maxN = Math.log10(st[0].n);
    st.forEach((s, i) => {
      const x = 100 + i*95, hgt = 30 + 100 * Math.log10(s.n)/maxN, y = by + (130 - hgt)/2;
      const r = g.append("rect").attr("x",x).attr("y",y).attr("width",78).attr("height",hgt).attr("fill",P.A).attr("opacity",0.25+0.15*i).attr("rx",4);
      r.append("title").text("stage "+s.k+": "+s.h+" × "+s.w+" tokens × C = "+s.c+", "+s.heads+" heads, "+s.blocks+" blocks");
      txt(g, x+39, y-6, "stage "+s.k+" ×"+s.blocks, {anchor:"middle", size:10, fill:P.muted});
      txt(g, x+39, by+150, s.h+"×"+s.w, {anchor:"middle", size:10, mono:true});
      txt(g, x+39, by+164, comma(s.n)+" tok", {anchor:"middle", size:10, mono:true});
      txt(g, x+39, by+178, "C = "+s.c, {anchor:"middle", size:10, mono:true, fill:P.muted});
    });
    const dx = 500;
    g.append("rect").attr("x",dx).attr("y",by).attr("width",140).attr("height",130).attr("fill",P.B).attr("opacity",0.3).attr("rx",4)
      .append("title").text("decoder: "+DEC.layers+" layers, d = "+DEC.d+", "+DEC.heads+" heads, FFN "+DEC.ffn);
    txt(g, dx+70, by+22, "decoder ×"+DEC.layers, {anchor:"middle", size:11, bold:true});
    ["causal self-attn", "cross-attn → z", "FFN "+DEC.ffn].forEach((t, i) => txt(g, dx+70, by+48+i*20, t, {anchor:"middle", size:10}));
    txt(g, dx+70, by+118, "d = "+DEC.d+", "+DEC.heads+" heads", {anchor:"middle", size:10, fill:P.muted});
    const s4 = st[3];
    g.append("path").attr("d", "M"+(100+3*95+78)+","+(by+65)+" L"+dx+","+(by+65)).attr("stroke",P.B).attr("stroke-width",2).attr("marker-end",null);
    txt(g, (100+3*95+78+dx)/2, by+58, "z: "+comma(s4.n)+"×"+s4.c, {anchor:"middle", size:9.5, fill:P.B});
    txt(g, dx+70, by+150, "→ tokens → JSON", {anchor:"middle", size:10.5, fill:P.good});
    txt(g, dx+70, by+166, "vocab "+comma(DEC.vocab), {anchor:"middle", size:10, fill:P.muted});
    const kv = DEC.layers * 2 * s4.n * DEC.d;
    $("arch-read").textContent = "canvas " + H + " × " + W + ": stage-1 " + comma(st[0].n) + " tokens → stage-4 " + comma(s4.n) +
      " tokens (" + (st[0].n / s4.n) + "× fewer) handed to the decoder · cross-attention K/V cache " + comma(kv) + " values = " + (kv*2/1e6).toFixed(1) + " MB in fp16";
  }
  on("arch-res", "change", draw);
  draw();
});

/* ───────────────────────── 05 · fitting a page onto the canvas ───────────────────────── */
DV.safe("prep", function(){
  const {P, txt, $, val, on, fitPage} = DV;
  const svg = d3.select("#prep-svg");
  /* physical size × scan resolution (pixels are w × h) */
  const PAGES = {
    a4:      {name:"A4 at 300 dpi",            w:Math.round(210/25.4*300), h:Math.round(297/25.4*300), dpi:300},
    letter:  {name:"US Letter at 200 dpi",     w:Math.round(8.5*200),      h:Math.round(11*200),       dpi:200},
    receipt: {name:"80 × 200 mm receipt, 300 dpi", w:Math.round(80/25.4*300), h:Math.round(200/25.4*300), dpi:300},
    slide:   {name:"13.33 × 7.5 in slide, 144 dpi", w:Math.round(13.333*144), h:Math.round(7.5*144), dpi:144},
    ticket:  {name:"87 × 55 mm ticket, 300 dpi", w:Math.round(87/25.4*300), h:Math.round(55/25.4*300), dpi:300}
  };
  function draw(){
    svg.selectAll("*").remove();
    const pg = PAGES[val("prep-page")], [H, W] = val("prep-canvas").split("x").map(Number), align = $("prep-align").checked;
    const f = fitPage(pg.w, pg.h, H, W, align);
    const g = svg.append("g");
    const k = 270 / Math.max(H, W*270/270), cw = W*k, ch = H*k;
    const ox = 20, oy = 20;
    g.append("rect").attr("x",ox).attr("y",oy).attr("width",cw).attr("height",ch).attr("fill","#2b3040").attr("stroke",P.muted).attr("stroke-dasharray","3,3");
    g.append("rect").attr("x",ox + f.padL*k).attr("y",oy + f.padT*k).attr("width",f.w*k).attr("height",f.h*k).attr("fill",P.paper);
    for (let i=0;i<6;i++) g.append("rect").attr("x",ox + f.padL*k + f.w*k*0.1).attr("y",oy + f.padT*k + f.h*k*(0.1+0.13*i)).attr("width",f.w*k*(0.5+0.3*((i*7)%3)/2)).attr("height",Math.max(1, f.h*k*0.03)).attr("fill","#9b958a");
    txt(g, ox, oy+ch+16, "canvas "+H+" × "+W+" (H × W); grey = padding", {size:10.5, fill:P.muted});
    /* 10-pt em zoom */
    const em = 10/72 * pg.dpi * f.scale, patches = em/4;
    const zx = 360, zy = 40, zs = Math.min(200, em*5);
    txt(g, zx, 26, "a 10-pt em on the canvas, with 4 × 4-pixel patches", {size:10.5, fill:P.A});
    const cell = zs / em * 4;
    g.append("rect").attr("x",zx).attr("y",zy).attr("width",zs).attr("height",zs).attr("fill",P.paper);
    for (let v = 0; v <= zs + 0.01; v += cell){
      g.append("line").attr("x1",zx+v).attr("x2",zx+v).attr("y1",zy).attr("y2",zy+zs).attr("stroke","#8a8577").attr("stroke-width",0.5);
      g.append("line").attr("y1",zy+v).attr("y2",zy+v).attr("x1",zx).attr("x2",zx+zs).attr("stroke","#8a8577").attr("stroke-width",0.5);
    }
    txt(g, zx+zs/2, zy+zs*0.78, "Ag", {anchor:"middle", size:zs*0.7, fill:P.paperInk});
    txt(g, zx, zy+zs+18, em.toFixed(1)+" px = "+patches.toFixed(1)+" patches per em", {size:11, mono:true});
    $("prep-read").textContent = pg.name + " (" + pg.w + " × " + pg.h + " px, w × h)" + (f.rotated ? " rotated 90° →" : " →") +
      " shorter side to " + Math.min(H, W) + ": " + f.rw + " × " + f.rh + " → thumbnail " + f.w + " × " + f.h +
      " → pad left " + f.padL + ", top " + f.padT + " · scale " + f.scale.toFixed(3) + " · padding " + (100*f.padFrac).toFixed(1) + "% of canvas · 10-pt em " + em.toFixed(1) + " px";
  }
  ["prep-page","prep-canvas","prep-align"].forEach(id => on(id, "change", draw));
  draw();
});

/* ───────────────────────── 06 · stages, tokens and cost ───────────────────────── */
DV.safe("swin", function(){
  const {P, txt, $, val, on, swinStages, comma, sci, bytes} = DV;
  const svg = d3.select("#swin-svg");
  function draw(){
    svg.selectAll("*").remove();
    const [H, W] = val("swin-res").split("x").map(Number), M = +val("swin-m"), glob = $("swin-global").checked;
    $("swin-mv").textContent = M + " × " + M;
    const st = swinStages(H, W, M);
    const g = svg.append("g");
    const all = [];
    st.forEach(s => { all.push(s.n, s.wmsa); if (glob) all.push(s.msa); });
    const x = d3.scaleLog().domain([1e3, d3.max(all)*1.5]).range([150, 640]);
    x.ticks(6).filter(t => Math.log10(t) % 1 === 0).forEach(t => {
      g.append("line").attr("x1",x(t)).attr("x2",x(t)).attr("y1",18).attr("y2",280).attr("stroke",P.line);
      txt(g, x(t), 292, "1e"+Math.round(Math.log10(t)), {anchor:"middle", size:9.5, fill:P.muted});
    });
    st.forEach((s, i) => {
      const y0 = 22 + i*66;
      txt(g, 10, y0+12, "stage "+s.k+" · "+s.h+"×"+s.w, {size:11, bold:true});
      txt(g, 10, y0+27, "C "+s.c+" · "+s.heads+" heads · ×"+s.blocks, {size:10, fill:P.muted});
      txt(g, 10, y0+41, comma(s.wins)+" windows"+(s.np !== s.n ? " (padded)" : ""), {size:10, fill:P.muted});
      const bars = [["tokens", s.n, P.A], ["W-MSA MACs / block", s.wmsa, P.good]];
      if (glob) bars.push(["global MSA MACs / block", s.msa, P.bad]);
      bars.forEach((b, j) => {
        const y = y0 + j*16;
        g.append("rect").attr("x",150).attr("y",y).attr("width",Math.max(1, x(b[1]) - 150)).attr("height",12).attr("fill",b[2]).attr("opacity",0.75);
        txt(g, x(b[1]) + 4, y+10, b[0]+" "+(j === 0 ? comma(b[1]) : sci(b[1])), {size:9.5, mono:true});
      });
    });
    const tot = st.reduce((a,s) => a + s.blocks*(s.wmsa + s.mlp), 0);
    const totG = st.reduce((a,s) => a + s.blocks*(s.msa + s.mlp), 0);
    const kv = 4*2*st[3].n*1024*2;
    $("swin-read").textContent = "encoder attention + MLP ≈ " + sci(tot) + " MACs per page with " + M + "×" + M + " windows" +
      (glob ? " vs " + sci(totG) + " with global attention (" + (totG/tot).toFixed(1) + "×)" : "") +
      " · largest attention-score buffer " + bytes(d3.max(st, s => s.memW)) + (glob ? " vs " + bytes(d3.max(st, s => s.memG)) + " global" : "") +
      " · decoder sees " + comma(st[3].n) + " vectors (cross-attn K/V " + bytes(kv) + ")";
  }
  ["swin-res","swin-global"].forEach(id => on(id, "change", draw));
  on("swin-m", "input", draw);
  draw();
});

/* ───────────────────────── 07 · regular and shifted windows ───────────────────────── */
DV.safe("window", function(){
  const {P, txt, $, val, on} = DV;
  const svg = d3.select("#win-svg");
  const GW = 20, GH = 15;           // illustrative token grid (columns × rows)
  let sel = {r:4, c:4};
  const fl = Math.floor;
  function wid(r, c, M, shift){ const s = shift ? fl(M/2) : 0; return fl((r - s)/M) + ":" + fl((c - s)/M); }
  function setFor(r, c, M, shift){
    const id = wid(r, c, M, shift), out = [];
    for (let rr=0; rr<GH; rr++) for (let cc=0; cc<GW; cc++) if (wid(rr, cc, M, shift) === id) out.push(rr*GW+cc);
    return out;
  }
  function draw(){
    svg.selectAll("*").remove();
    const mode = val("win-mode"), M = +val("win-m"), s = fl(M/2);
    let set;
    if (mode === "w") set = setFor(sel.r, sel.c, M, false);
    else if (mode === "sw") set = setFor(sel.r, sel.c, M, true);
    else { const u = {}; setFor(sel.r, sel.c, M, true).forEach(k => setFor(fl(k/GW), k%GW, M, false).forEach(j => { u[j] = 1; })); set = Object.keys(u).map(Number); }
    const inSet = {}; set.forEach(k => { inSet[k] = 1; });
    const g = svg.append("g"), cs = 17, ox = 12, oy = 18;
    for (let r=0; r<GH; r++) for (let c=0; c<GW; c++){
      const k = r*GW+c, isSel = (r === sel.r && c === sel.c);
      g.append("rect").attr("x",ox+c*cs).attr("y",oy+r*cs).attr("width",cs-1).attr("height",cs-1)
        .attr("fill", isSel ? P.B : inSet[k] ? P.A : "#2b3040").attr("opacity", isSel ? 1 : inSet[k] ? 0.8 : 1)
        .style("cursor","pointer").on("click", () => { sel = {r, c}; draw(); });
    }
    const shifts = mode === "w" ? [0] : mode === "sw" ? [s] : [0, s];
    shifts.forEach((sh, j) => {
      const col = sh === 0 ? P.good : P.pink;
      for (let v = sh % M; v <= GW; v += M) if (v > 0 && v < GW) g.append("line").attr("x1",ox+v*cs-0.5).attr("x2",ox+v*cs-0.5).attr("y1",oy).attr("y2",oy+GH*cs).attr("stroke",col).attr("stroke-width",2).attr("stroke-dasharray", j ? "4,3" : null);
      for (let v = sh % M; v <= GH; v += M) if (v > 0 && v < GH) g.append("line").attr("y1",oy+v*cs-0.5).attr("y2",oy+v*cs-0.5).attr("x1",ox).attr("x2",ox+GW*cs).attr("stroke",col).attr("stroke-width",2).attr("stroke-dasharray", j ? "4,3" : null);
    });
    const regular = Math.ceil(GH/M)*Math.ceil(GW/M);
    const ids = {}; for (let r=0;r<GH;r++) for (let c=0;c<GW;c++) ids[wid(r,c,M,true)] = 1;
    const naive = Object.keys(ids).length;
    const lx = 370;
    txt(g, lx, 30, "green lines: regular windows", {size:10.5, fill:P.good});
    txt(g, lx, 48, "pink lines: shifted by floor(M/2) = "+s, {size:10.5, fill:P.pink});
    txt(g, lx, 66, "amber: query token; blue: tokens it reaches", {size:10.5, fill:P.muted});
    txt(g, lx, 100, "regular windows: "+regular, {size:11, mono:true});
    txt(g, lx, 118, "shifted, naive partition: "+naive, {size:11, mono:true});
    txt(g, lx, 136, "shifted, after cyclic shift: "+regular, {size:11, mono:true});
    txt(g, lx, 170, "tokens reached: "+set.length+" of "+(GW*GH), {size:11, mono:true, fill:P.A});
    $("win-read").textContent = "token (row " + sel.r + ", col " + sel.c + "), M = " + M + ": " +
      (mode === "w" ? "W-MSA window has " : mode === "sw" ? "SW-MSA (masked) window has " : "after W-MSA then SW-MSA it can draw on ") + set.length +
      " tokens; a full window holds " + (M*M) + " · windows per block: " + regular + " regular, " + naive + " in a naive shifted partition, " + regular + " after the cyclic shift";
  }
  ["win-mode","win-m"].forEach(id => on(id, "change", draw));
  draw();
});

/* ───────────────────────── 09 · what one decoding step looks at ───────────────────────── */
DV.safe("xattn", function(){
  const {P, txt, $, val, on, TOK} = DV;
  const svg = d3.select("#xa-svg");
  /* Invented receipt layout on a 1920 × 2560 canvas (w × h). Stage-4 grid: 60 × 80 cells of 32 px. */
  const ITEMS = [
    {t:"CAFE DONUT",             b:[420,150,1500,330]},
    {t:"3002-Kyoto Choco Mochi", b:[150,700,1250,800]},
    {t:"2",                      b:[1300,700,1360,800]},
    {t:"28,000",                 b:[1500,700,1800,800]},
    {t:"Latte",                  b:[150,900,420,1000]},
    {t:"4,500",                  b:[1560,900,1800,1000]},
    {t:"TOTAL",                  b:[150,1500,500,1600]},
    {t:"50,000",                 b:[1450,1500,1800,1600]}
  ];
  /* output steps: a piece of a string (box split in proportion to characters), or a structure token (broad) */
  const STEPS = [];
  function addPieces(str){
    const it = ITEMS.find(o => o.t === str), pcs = TOK[str], total = pcs.reduce((a,p) => a + p.replace("▁","").length, 0) || 1;
    let acc = 0;
    pcs.forEach(p => { const L = p.replace("▁","").length || 0.5;
      const x0 = it.b[0] + (it.b[2]-it.b[0]) * acc/total, x1 = it.b[0] + (it.b[2]-it.b[0]) * Math.min(1, (acc+L)/total);
      STEPS.push({label:p+"  (from \""+str+"\")", b:[x0, it.b[1], Math.max(x1, x0+24), it.b[3]], sigma:48}); acc += L; });
  }
  addPieces("3002-Kyoto Choco Mochi");
  STEPS.push({label:"</s_nm>  (structure token)", b:ITEMS[1].b, sigma:260});
  addPieces("28,000");
  addPieces("50,000");
  const sel = $("xa-tok");
  if (sel && !sel.options.length) STEPS.forEach((s, i) => { const o = document.createElement("option"); o.value = i; o.textContent = s.label; sel.appendChild(o); });
  const GW = 60, GH = 80, CELL = 32;
  function draw(){
    svg.selectAll("*").remove();
    const st = STEPS[+val("xa-tok") || 0], focus = +val("xa-sharp");
    const logits = [], b = st.b;
    for (let r=0; r<GH; r++) for (let c=0; c<GW; c++){
      const cx = (c+0.5)*CELL, cy = (r+0.5)*CELL;
      const dx = Math.max(b[0]-cx, 0, cx-b[2]), dy = Math.max(b[1]-cy, 0, cy-b[3]);
      const d = Math.hypot(dx, dy);
      logits.push(-(focus/2) * (d/st.sigma) * (d/st.sigma));
    }
    const mx = d3.max(logits), ex = logits.map(v => Math.exp(v - mx)), Z = d3.sum(ex), a = ex.map(v => v/Z);
    let inside = 0, H = 0, top = 0;
    a.forEach((v, i) => { const r = Math.floor(i/GW), c = i%GW, cx = (c+0.5)*CELL, cy = (r+0.5)*CELL;
      if (cx >= b[0] && cx <= b[2] && cy >= b[1] && cy <= b[3]) inside += v; if (v > 0) H -= v*Math.log(v); if (v > a[top]) top = i; });
    const g = svg.append("g"), k = 3.9, ox = 10, oy = 8;
    g.append("rect").attr("x",ox).attr("y",oy).attr("width",GW*k).attr("height",GH*k).attr("fill",P.paper);
    ITEMS.forEach(o => txt(g, ox + o.b[0]/CELL*k, oy + o.b[3]/CELL*k - 3, o.t, {size:Math.max(7, (o.b[3]-o.b[1])/CELL*k*0.75), fill:P.paperInk, mono:true}));
    const amax = a[top];
    a.forEach((v, i) => { if (v > amax/40){ const r = Math.floor(i/GW), c = i%GW;
      g.append("rect").attr("x",ox+c*k).attr("y",oy+r*k).attr("width",k).attr("height",k).attr("fill",P.bad).attr("opacity", 0.1 + 0.75*v/amax); } });
    g.append("rect").attr("x",ox+b[0]/CELL*k).attr("y",oy+b[1]/CELL*k).attr("width",(b[2]-b[0])/CELL*k).attr("height",(b[3]-b[1])/CELL*k).attr("fill","none").attr("stroke",P.A).attr("stroke-width",1.2);
    const lx = 270;
    txt(g, lx, 24, "generating: "+st.label, {size:11, bold:true});
    txt(g, lx, 50, "grid: "+GW+" × "+GH+" = "+(GW*GH)+" stage-4 cells (32 px each)", {size:10.5, fill:P.muted});
    txt(g, lx, 74, "mass inside the true box: "+(100*inside).toFixed(1)+"%", {size:11, mono:true});
    txt(g, lx, 94, "entropy: "+H.toFixed(2)+" nats (uniform: "+Math.log(GW*GH).toFixed(2)+")", {size:11, mono:true});
    txt(g, lx, 114, "effective cells attended: "+Math.exp(H).toFixed(1), {size:11, mono:true});
    txt(g, lx, 134, "peak weight: "+amax.toFixed(4), {size:11, mono:true});
    txt(g, lx, 170, "blue outline: the text's box · red: attention weight", {size:10, fill:P.muted});
    $("xa-read").textContent = "step " + ((+val("xa-tok") || 0) + 1) + " of " + STEPS.length + ": " + (100*inside).toFixed(1) + "% of attention inside the box, effective " +
      Math.exp(H).toFixed(1) + " of " + (GW*GH) + " cells, peak cell (row " + Math.floor(top/GW) + ", col " + (top%GW) + ")";
  }
  on("xa-tok", "change", draw);
  on("xa-sharp", "input", draw);
  draw();
});

/* ───────────────────────── 10 · parameter calculator ───────────────────────── */
DV.safe("params", function(){
  const {P, txt, $, val, on, comma} = DV;
  const svg = d3.select("#par-svg");
  /* presets from the Hub configurations (vocab_size: donut-base 57,525; cord-v2 57,580; donut-proto 57,524) */
  const PRE = {
    base:  {d3:14, M:10, nl:4, pos:1536, V:57525},
    cord:  {d3:14, M:10, nl:4, pos:768,  V:57580},
    proto: {d3:18, M:8,  nl:4, pos:768,  V:57524},
    swinb: {d3:18, M:7,  nl:4, pos:1536, V:57525}
  };
  const PAPER_143M = 143e6;   // Donut paper Tables 1–2 (vocabulary omitted)
  let V = 57525;
  function count(cfg){
    const C0 = 128, depths = [2, 2, cfg.d3, 2], d = 1024, f = 4096;
    const blk = (c, h) => 12*c*c + 13*c + (2*cfg.M - 1)*(2*cfg.M - 1)*h;
    const parts = {embed: 48*C0 + C0 + 2*C0, stages: []};
    for (let k=0; k<4; k++){
      const c = C0*Math.pow(2,k), h = 4*Math.pow(2,k);
      const merge = k < 3 ? 8*c + 8*c*c : 0;
      parts.stages.push(depths[k]*blk(c, h) + merge);
    }
    const enc = parts.embed + d3.sum(parts.stages);
    const layer = 8*d*d + 8*d + 2*d*f + f + d + 6*d;
    const tok = cfg.V*d, pos = (cfg.pos + 2)*d, norms = 4*d, layers = cfg.nl*layer;
    return {enc, parts, tok, pos, norms, layers, dec: tok + pos + norms + layers, total: enc + tok + pos + norms + layers};
  }
  function setSliders(p){ $("par-d3").value = p.d3; $("par-m").value = p.M; $("par-nl").value = p.nl; $("par-pos").value = String(p.pos); V = p.V; }
  function draw(){
    svg.selectAll("*").remove();
    const cfg = {d3:+val("par-d3"), M:+val("par-m"), nl:+val("par-nl"), pos:+val("par-pos"), V};
    const r = count(cfg);
    const segs = [["patch embed", r.parts.embed, P.A], ["stage 1", r.parts.stages[0], P.A], ["stage 2", r.parts.stages[1], P.A], ["stage 3", r.parts.stages[2], P.A], ["stage 4", r.parts.stages[3], P.A],
      ["token embedding", r.tok, P.B], ["positions", r.pos, P.purple], ["decoder layers", r.layers, P.good], ["norms", r.norms, P.grey]];
    const g = svg.append("g"), x = d3.scaleLinear().domain([0, Math.max(r.total, 3.4e8)]).range([10, 650]);
    let acc = 0;
    segs.forEach((s, i) => {
      g.append("rect").attr("x",x(acc)).attr("y",40).attr("width",Math.max(0.5, x(acc+s[1]) - x(acc))).attr("height",40).attr("fill",s[2]).attr("opacity", 0.45 + 0.1*(i%5)).attr("stroke","#0b0d12");
      acc += s[1];
    });
    txt(g, 10, 30, "encoder (blue) · token embedding (amber) · positions (violet) · decoder layers (green)", {size:10.5, fill:P.muted});
    g.append("line").attr("x1",x(PAPER_143M)).attr("x2",x(PAPER_143M)).attr("y1",90).attr("y2",100).attr("stroke",P.ink);
    txt(g, x(PAPER_143M), 114, "paper's 143M", {anchor:"middle", size:10});
    const rows = [["encoder", r.enc], ["  of which stage 3", r.parts.stages[2]], ["token embedding "+comma(V)+" × 1,024", r.tok], ["position embedding ("+cfg.pos+" + 2) × 1,024", r.pos], [cfg.nl+" decoder layers", r.layers], ["total", r.total], ["total without token embedding", r.total - r.tok]];
    rows.forEach((row, i) => { txt(g, 10, 140 + i*14, row[0], {size:10.5}); txt(g, 330, 140 + i*14, comma(row[1]), {size:10.5, mono:true, anchor:"end"}); });
    $("par-read").textContent = "total " + comma(r.total) + " (" + (r.total/1e6).toFixed(1) + "M); without the token embedding " + comma(r.total - r.tok) +
      " (" + ((r.total - r.tok)/1e6).toFixed(1) + "M, " + (100*((r.total - r.tok)/PAPER_143M - 1)).toFixed(1) + "% vs the paper's 143M); encoder " + (100*r.enc/r.total).toFixed(1) + "% of all parameters";
  }
  on("par-preset", "change", () => { const p = PRE[val("par-preset")]; if (p) setSliders(p); draw(); });
  ["par-d3","par-m","par-nl"].forEach(id => on(id, "input", () => { $("par-preset").value = "custom"; draw(); }));
  on("par-pos", "change", () => { $("par-preset").value = "custom"; draw(); });
  setSliders(PRE.base);
  draw();
});

/* ───────────────────────── 12 · JSON ↔ token sequence ───────────────────────── */
DV.safe("json", function(){
  const {P, txt, $, val, on, json2token, token2json, chunks, normalize} = DV;
  const svg = d3.select("#js-svg");
  const PRE = {
    cord1:  {prompt:"<s_cord-v2>", obj:{menu:{nm:"Latte", cnt:"2", price:"9,000"}, total:{total_price:"9,000"}}},
    cord2:  {prompt:"<s_cord-v2>", obj:{menu:[{nm:"Latte", cnt:"2", price:"9,000"}, {nm:"Cookie", cnt:"1", price:"2,500"}], sub_total:{subtotal_price:"11,500"}, total:{total_price:"11,500", cashprice:"20,000", changeprice:"8,500"}}},
    /* the Donut paper's Figure 1 record, keys renamed to CORD's */
    kyoto:  {prompt:"<s_cord-v2>", obj:{menu:{nm:"3002-Kyoto Choco Mochi", cnt:"2", unitprice:"14,000", price:"28,000"}, total:{total_price:"50,000", menuqty_cnt:"4"}}},
    cls:    {prompt:"<s_rvlcdip>", obj:{"class":"memo"}},
    vqa:    {prompt:"<s_docvqa>", obj:{question:"what is the price of choco mochi?", answer:"14,000"}},
    /* the Donut paper's appendix Figure B, first ticket */
    ticket: {prompt:"<s_zhtrainticket>", obj:{date:"2017年11月15日", destination_station:"福田站", name:"珂", seat_category:"二等座", starting_station:"广州南站", ticket_num:"C068987", ticket_rates:"¥82.0元", train_num:"G79"}}
  };
  let cur = PRE.cord1;
  function render(obj, prompt){
    const sort = $("js-sort").checked, brk = $("js-break").checked;
    let body = json2token(obj, sort);
    let dropped = null;
    if (brk){ const all = body.match(/<\/s_[^>]*>/g) || []; if (all.length){ dropped = all[all.length-1]; const i = body.lastIndexOf(dropped); body = body.slice(0, i) + body.slice(i + dropped.length); } }
    const seq = prompt + body + "</s>";
    const ch = [{t:prompt, kind:"ctl"}].concat(chunks(body + "</s>"));
    const box = $("js-toks"); box.innerHTML = "";
    let nKey = 0, nText = 0, nUnk = 0;
    ch.forEach(c => {
      if (c.kind === "text"){
        if (c.pieces){ c.pieces.forEach(p => { const s = document.createElement("span"); s.textContent = p; box.appendChild(s); nText++; }); }
        else { const s = document.createElement("span"); s.className = "bad"; s.textContent = c.t + " ?"; box.appendChild(s); nUnk++; }
      } else { const s = document.createElement("span"); s.className = c.kind; s.textContent = c.t; box.appendChild(s); if (c.kind === "sp") nKey++; }
    });
    const parsed = token2json(body);
    $("js-back").textContent = "token2json → " + JSON.stringify(parsed, null, 1);
    const same = JSON.stringify(normalize(parsed)) === JSON.stringify(normalize(obj));
    svg.selectAll("*").remove();
    const g = svg.append("g"), total = 2 + nKey + nText + nUnk, x = d3.scaleLinear().domain([0, total]).range([10, 650]);
    let acc = 0;
    [["prompt + end", 2, P.purple], ["key / categorical tokens", nKey, P.B], ["text pieces", nText, P.A], ["untokenized text", nUnk, P.bad]].forEach(s => {
      if (s[1] > 0){ g.append("rect").attr("x",x(acc)).attr("y",8).attr("width",x(acc+s[1]) - x(acc)).attr("height",20).attr("fill",s[2]).attr("opacity",0.7);
        txt(g, x(acc)+3, 44, s[0]+" "+s[1], {size:10}); }
      acc += s[1];
    });
    const keys = {}; (body.match(/<\/?s_[^>]*>/g) || []).forEach(t => { const k = t.replace(/[<>\/]/g,"").replace(/^s_/,""); keys[k] = 1; });
    $("js-read").textContent = (nUnk ? "(pieces not counted for " + nUnk + " typed string" + (nUnk > 1 ? "s" : "") + ") " : "") + total + " tokens in total: " + nKey + " key/categorical, " + nText + " text pieces, 2 prompt/end · " +
      Math.round(100*(nKey + 1)/(total - 1)) + "% of the predicted tokens are structure · " + Object.keys(keys).length + " distinct keys → " + (2*Object.keys(keys).length) +
      " added tokens · " + (dropped ? "dropped " + dropped + " → " : "") + "round trip " + (same ? "identical after normalisation" : "NOT identical: a field was lost or garbled");
  }
  function loadPreset(){ cur = PRE[val("js-preset")] || PRE.cord1; $("js-text").value = JSON.stringify(cur.obj); render(cur.obj, cur.prompt); }
  on("js-preset", "change", loadPreset);
  ["js-sort","js-break"].forEach(id => on(id, "change", () => render(cur.obj, cur.prompt)));
  on("js-apply", "click", () => {
    try { const o = JSON.parse($("js-text").value); if (o === null || typeof o !== "object") throw new Error("top level must be an object"); cur = {prompt:cur.prompt, obj:o}; render(o, cur.prompt); }
    catch(e){ $("js-read").textContent = "could not parse the JSON: " + e.message; }
  });
  loadPreset();
});

/* ───────────────────────── 13 · decoding a record one token at a time ───────────────────────── */
DV.safe("decode", function(){
  const {P, txt, $, val, on, token2json} = DV;
  const svg = d3.select("#dec-svg");
  const PROMPT = "<s_cord-v2>";
  /* real tokenization of {"menu":{"nm":"Latte","cnt":"2"},"total":{"total_price":"9,000"}} (donut-base tokenizer) */
  const TGT = ["<s_menu>","<s_nm>","▁La","tte","</s_nm>","<s_cnt>","▁2","</s_cnt>","</s_menu>","<s_total>","<s_total_price>","▁9",",","000","</s_total_price>","</s_total>","</s>"];
  /* illustrative next-token distributions (top candidates; the rest of the mass is "other") */
  const DIST = [
    [["<s_menu>",0.97],["<s_total>",0.02]], [["<s_nm>",0.99],["<s_cnt>",0.005]], [["▁La",0.62],["▁Le",0.30],["▁Lat",0.05]],
    [["tte",0.91],["te",0.06]], [["</s_nm>",0.98],["▁L",0.01]], [["<s_cnt>",0.95],["<s_price>",0.04]], [["▁2",0.80],["▁1",0.12],["▁3",0.05]],
    [["</s_cnt>",0.99],["▁x",0.005]], [["</s_menu>",0.93],["<s_price>",0.06]], [["<s_total>",0.96],["<s_sub_total>",0.03]],
    [["<s_total_price>",0.97],["<s_cashprice>",0.02]], [["▁9",0.70],["▁8",0.20],["▁6",0.06]], [[",",0.94],[".",0.05]],
    [["000",0.97],["500",0.02]], [["</s_total_price>",0.99],["▁",0.005]], [["</s_total>",0.99],["<s_cashprice>",0.008]], [["</s>",0.98],["<s_menu>",0.01]]
  ];
  /* misread mode: at step 3 the model prefers ▁Le; step 4 is conditioned on that prefix (illustrative) */
  const ERR = {2: [["▁Le",0.55],["▁La",0.35],["▁Lat",0.05]], 3: [["tte",0.88],["te",0.09]]};
  let t = 0, out = [];
  function distAt(i, mode){ return (mode === "err" && ERR[i]) ? ERR[i] : DIST[i]; }
  function stepOnce(){
    const mode = val("dec-mode");
    if (t >= TGT.length) return;
    const d = distAt(t, mode);
    const pick = mode === "tf" ? TGT[t] : d.reduce((a, b) => b[1] > a[1] ? b : a)[0];
    out.push(pick); t++;
  }
  function draw(){
    svg.selectAll("*").remove();
    const mode = val("dec-mode"), g = svg.append("g");
    txt(g, 10, 18, mode === "tf" ? "decoder input = prompt + ground-truth prefix (teacher forcing)" : "decoder input = prompt + its own previous outputs", {size:10.5, fill:P.muted});
    const shown = [PROMPT].concat(mode === "tf" ? TGT.slice(0, t) : out);
    let x = 10, y = 38;
    shown.forEach((s, i) => {
      const w = 8 + s.length*6.2;
      if (x + w > 650){ x = 10; y += 22; }
      const wrong = mode !== "tf" && i > 0 && s !== TGT[i-1];
      g.append("rect").attr("x",x).attr("y",y-12).attr("width",w).attr("height",17).attr("rx",3)
        .attr("fill", i === 0 ? P.purple : wrong ? P.bad : /^<\/?s_|^<\/s>/.test(s) ? P.B : P.A).attr("opacity",0.35);
      txt(g, x+4, y+1, s, {size:10, mono:true});
      x += w + 3;
    });
    const cy = 150;
    if (t < TGT.length){
      const d = distAt(t, mode);
      txt(g, 10, cy - 12, "step " + (t+1) + ": next-token candidates (target " + TGT[t] + ")", {size:11, bold:true});
      const xs = d3.scaleLinear().domain([0,1]).range([0, 360]);
      d.forEach((c, i) => {
        const yy = cy + i*22;
        g.append("rect").attr("x",150).attr("y",yy).attr("width",xs(c[1])).attr("height",16).attr("fill", c[0] === TGT[t] ? P.good : P.muted).attr("opacity",0.7);
        txt(g, 144, yy+12, c[0], {size:10, mono:true, anchor:"end"});
        txt(g, 156 + xs(c[1]), yy+12, c[1].toFixed(3), {size:10, mono:true});
      });
      const other = 1 - d3.sum(d, c => c[1]);
      txt(g, 144, cy + d.length*22 + 12, "other", {size:10, mono:true, anchor:"end", fill:P.muted});
      txt(g, 156, cy + d.length*22 + 12, other.toFixed(3) + " spread over the rest of the vocabulary", {size:10, mono:true, fill:P.muted});
    } else {
      txt(g, 10, cy, "finished: " + (mode === "tf" ? TGT.length : out.length) + " tokens generated", {size:11, bold:true});
    }
    let msg;
    if (mode === "tf"){
      const nll = TGT.slice(0, t).map((tok, i) => -Math.log(DIST[i].find(c => c[0] === tok)[1]));
      const s = d3.sum(nll);
      msg = t === 0 ? "teacher forcing: press step · loss accumulates −ln p(target) at each position" :
        "step " + t + ": −ln p(" + TGT[t-1] + ") = " + nll[t-1].toFixed(4) + " · sum " + s.toFixed(4) + " · mean L = " + (s/t).toFixed(4) +
        " · perplexity " + Math.exp(s/t).toFixed(3) + " · sequence probability " + Math.exp(-s).toFixed(4);
    } else {
      const errs = out.filter((s, i) => s !== TGT[i]).length;
      const body = out.join("").replace("</s>", "").replace(/▁/g, " ");
      msg = t === 0 ? "greedy decoding: press step · the argmax token is appended and fed back" :
        "step " + t + ": picked " + out[t-1] + " · tokens differing from the target so far: " + errs +
        (t >= TGT.length ? " · token2json → " + JSON.stringify(token2json(body)) : "");
    }
    $("dec-read").textContent = msg;
  }
  on("dec-step", "click", () => { stepOnce(); draw(); });
  on("dec-run", "click", () => { while (t < TGT.length) stepOnce(); draw(); });
  on("dec-reset", "click", () => { t = 0; out = []; draw(); });
  on("dec-mode", "change", () => { t = 0; out = []; draw(); });
  draw();
});

/* ───────────────────────── 15 · SynthDoG layer composer ───────────────────────── */
DV.safe("synthdog", function(){
  const {P, txt, $, val, on, lcg} = DV;
  const svg = d3.select("#sd-svg");
  /* Ranges from clovaai/donut synthdog/config_en.yaml */
  const CFG = {quality:[50,95], landscape:0.5, short_size:[720,1024], aspect_ratio:[1,2],
    bg_blur_sigma:[0,10], doc_fullscreen:0.5, doc_short_size:[480,1024], paper_alpha:[0,0.2],
    text_scale:[0.0334,0.1], max_row:10, max_col:3, margin:[0,0.1],
    perspective_weights:[750,50,50,25,25,25,25,50], perspective_pct:[0.75,1],
    shadow_intensity:[0,160], contrast_alpha:[1,1.5], brightness_beta:[-48,0], motion_blur_prob:0.5, motion_blur_k:[3,5], gauss_sigma:[0,1.5]};
  /* Placeholder phrases written for this figure (not from the Wikipedia corpora SynthDoG samples). */
  const TEXT = {
    en:["annual rainfall in river valleys","the committee met on tuesday","notes on early printing","library opening hours","a short history of bridges","results of the spring survey"],
    zh:["城市交通发展报告","春季调查结果","图书馆开放时间","河谷年降雨量","印刷术简史","委员会会议纪要"],
    ja:["東京の天気と季節","図書館の開館時間","春の調査結果","川の流域の降水量","印刷の歴史","委員会の議事録"],
    ko:["서울 도서관 운영 안내","봄 조사 결과","강 유역 연평균 강수량","인쇄술의 짧은 역사","위원회 회의록","다리의 역사"]
  };
  let seed = 7;
  function U(r, a){ return a[0] + (a[1]-a[0])*r(); }
  function draw(){
    svg.selectAll("*").remove();
    const r = lcg(seed*2654435761 >>> 0), lang = val("sd-lang");
    const land = r() < CFG.landscape, short = Math.round(U(r, CFG.short_size)), ar = U(r, CFG.aspect_ratio);
    const cw = land ? Math.round(short*ar) : short, ch = land ? short : Math.round(short*ar);
    const full = r() < CFG.doc_fullscreen, dshort = Math.round(U(r, CFG.doc_short_size));
    const alpha = U(r, CFG.paper_alpha), ts = U(r, CFG.text_scale), bgs = U(r, CFG.bg_blur_sigma);
    const rows = 1 + Math.floor(r()*CFG.max_row), cols = 1 + Math.floor(r()*CFG.max_col);
    const wsum = CFG.perspective_weights.reduce((a,b) => a+b, 0); let pick = r()*wsum, pat = 0;
    while (pick > CFG.perspective_weights[pat]){ pick -= CFG.perspective_weights[pat]; pat++; }
    const corners = [0,1,2,3].map(() => pat === 7 ? 1 : U(r, CFG.perspective_pct));
    const shadow = U(r, CFG.shadow_intensity), contrast = U(r, CFG.contrast_alpha), bright = U(r, CFG.brightness_beta);
    const mblur = r() < CFG.motion_blur_prob ? Math.round(U(r, CFG.motion_blur_k)) : 0, gsig = U(r, CFG.gauss_sigma), q = Math.round(U(r, CFG.quality));
    const g = svg.append("g");
    const k = Math.min(380/cw, 300/ch), ox = 10, oy = 10, W = cw*k, H = ch*k;
    if ($("sd-bg").checked){
      g.append("rect").attr("x",ox).attr("y",oy).attr("width",W).attr("height",H).attr("fill","#3b4a3a");
      for (let i=0;i<14;i++) g.append("circle").attr("cx",ox+r()*W).attr("cy",oy+r()*H).attr("r",10+r()*50).attr("fill",d3.hsl(r()*360,0.35,0.35).toString()).attr("opacity",0.55);
    } else g.append("rect").attr("x",ox).attr("y",oy).attr("width",W).attr("height",H).attr("fill","none").attr("stroke",P.line);
    const dw = full ? W*0.94 : Math.min(W*0.9, dshort*k*(ch > cw ? 1 : 1.3)), dh = full ? H*0.94 : Math.min(H*0.9, dshort*k*(ch > cw ? 1.3 : 1));
    const dx = ox + (W - dw)/2, dy = oy + (H - dh)/2;
    const quad = [[dx + dw*(1-corners[0])/2, dy + dh*(1-corners[0])/2], [dx + dw - dw*(1-corners[1])/2, dy + dh*(1-corners[1])/2],
      [dx + dw - dw*(1-corners[2])/2, dy + dh - dh*(1-corners[2])/2], [dx + dw*(1-corners[3])/2, dy + dh - dh*(1-corners[3])/2]];
    if ($("sd-paper").checked){
      g.append("path").attr("d","M"+quad.map(p => p.join(",")).join("L")+"Z").attr("fill",P.paper);
      for (let i=0;i<30;i++) g.append("circle").attr("cx",dx+r()*dw).attr("cy",dy+r()*dh).attr("r",1+r()*3).attr("fill","#b8b09c").attr("opacity",alpha*3);
    }
    const lines = [];
    const m = 0.06, cellW = dw*(1-2*m)/cols, cellH = dh*(1-2*m)/rows, lh = Math.max(4, ts*Math.min(dw, dh));
    const phrases = TEXT[lang];
    for (let rr=0; rr<rows; rr++) for (let cc=0; cc<cols; cc++){
      if (r() < 0.25) continue;
      const bx = dx + dw*m + cc*cellW, by = dy + dh*m + rr*cellH;
      if ($("sd-layout").checked) g.append("rect").attr("x",bx+2).attr("y",by+2).attr("width",cellW-4).attr("height",cellH-4).attr("fill","none").attr("stroke",P.A).attr("stroke-dasharray","3,2");
      const nl = Math.max(1, Math.floor((cellH-4)/(lh*1.3)));
      for (let li=0; li<nl; li++){
        const ph = phrases[Math.floor(r()*phrases.length)];
        lines.push({y:by+li*lh*1.3, x:bx, t:ph});
        if ($("sd-text").checked) txt(g, bx+4, by+4+lh + li*lh*1.3, ph, {size:Math.max(5, lh*0.8), fill:d3.hsl(0,0,0.1 + 0.2*r()).toString()});
      }
    }
    if ($("sd-fx").checked){
      const grad = svg.append("defs").append("linearGradient").attr("id","sd-shadow").attr("x1","0").attr("x2","1");
      grad.append("stop").attr("offset","0").attr("stop-color","#000").attr("stop-opacity", shadow/255*0.8);
      grad.append("stop").attr("offset","1").attr("stop-color","#000").attr("stop-opacity",0);
      g.append("rect").attr("x",ox).attr("y",oy).attr("width",W).attr("height",H).attr("fill","url(#sd-shadow)");
    }
    const lx = 410, ps = [["canvas", cw+" × "+ch+(land ? " (landscape)" : "")], ["document", full ? "full-screen" : "short side "+dshort+" px"], ["background blur σ", bgs.toFixed(2)],
      ["paper alpha", alpha.toFixed(3)], ["perspective corners", corners.map(c => (100*c).toFixed(0)+"%").join(" ")], ["grid", rows+" rows × "+cols+" cols"], ["text scale", ts.toFixed(4)],
      ["shadow intensity", shadow.toFixed(0)], ["contrast α / brightness β", contrast.toFixed(2)+" / "+bright.toFixed(0)], ["motion blur k", mblur ? String(mblur) : "none"], ["Gaussian blur σ", gsig.toFixed(2)], ["JPEG quality", String(q)]];
    ps.forEach((p, i) => { txt(g, lx, 20 + i*22, p[0], {size:10, fill:P.muted}); txt(g, lx, 31 + i*22, p[1], {size:10.5, mono:true}); });
    lines.sort((a, b) => (a.y - b.y) || (a.x - b.x));
    const tr = lines.map(l => l.t).join(" ");
    $("sd-read").textContent = "seed " + seed + " · " + lines.length + " text lines · target (reading order, top-left to bottom-right): \"" + (tr.length > 160 ? tr.slice(0, 160) + " …" : tr) + "\"";
  }
  ["sd-bg","sd-paper","sd-layout","sd-text","sd-fx","sd-lang"].forEach(id => on(id, "change", draw));
  on("sd-new", "click", () => { seed++; draw(); });
  draw();
});

/* ───────────────────────── 17 · tree edit distance ───────────────────────── */
DV.safe("ted", function(){
  const {P, txt, $, val, on, tree, normalize, tedAcc, f1} = DV;
  const svg = d3.select("#ted-svg");
  const GT = {menu:[{nm:"Latte", cnt:"2", price:"9,000"}], total:{total_price:"9,000"}};
  const PRED = {
    exact: GT,
    "char": {menu:[{nm:"Latte", cnt:"2", price:"9,800"}], total:{total_price:"9,000"}},
    key:   {menu:[{nm:"Latte", cnt:"2", unitprice:"9,000"}], total:{total_price:"9,000"}},
    miss:  {menu:[{nm:"Latte", price:"9,000"}], total:{total_price:"9,000"}},
    extra: {menu:[{nm:"Latte", cnt:"2", price:"9,000"}, {nm:"Cookie", cnt:"1", price:"2,500"}], total:{total_price:"9,000"}},
    flat:  {nm:"Latte", cnt:"2", price:"9,000", total_price:"9,000"},
    empty: {}
  };
  function paths(t){ const out = []; (function rec(n, p){ const q = p + "/" + n.label; out.push(q); n.children.forEach(c => rec(c, q)); })(t, ""); return out; }
  function drawTree(g, t, x0, w, other){
    const h = d3.hierarchy(t, d => d.children), lay = d3.tree().size([w, 200]);
    lay(h);
    const op = paths(other).slice();
    const mine = {}; (function rec(n, p){ const q = p + "/" + n.data.label; const i = op.indexOf(q); mine[q + "#" + (n.x|0)] = i >= 0; if (i >= 0) op.splice(i, 1); (n.children || []).forEach(c => rec(c, q)); })(h, "");
    h.links().forEach(l => g.append("line").attr("x1",x0+l.source.x).attr("y1",40+l.source.y).attr("x2",x0+l.target.x).attr("y2",40+l.target.y).attr("stroke",P.line));
    (function rec(n, p){
      const q = p + "/" + n.data.label, ok = mine[q + "#" + (n.x|0)], leaf = n.data.label.indexOf("<leaf>") === 0;
      const lab = leaf ? "\"" + n.data.label.slice(6) + "\"" : n.data.label.replace("<subtree>","{…}").replace("<root>","root");
      g.append("circle").attr("cx",x0+n.x).attr("cy",40+n.y).attr("r",4).attr("fill", ok ? (leaf ? P.A : P.good) : P.bad);
      txt(g, x0+n.x, 40+n.y+14, lab, {size:9, anchor:"middle", mono:true, fill: ok ? P.ink : P.bad});
      (n.children || []).forEach(c => rec(c, q));
    })(h, "");
  }
  function draw(){
    svg.selectAll("*").remove();
    const pred = PRED[val("ted-pred")] || GT;
    const tg = tree(normalize(GT)), tp = tree(normalize(pred));
    const g = svg.append("g");
    txt(g, 10, 18, "ground truth", {size:11, bold:true}); txt(g, 340, 18, "prediction", {size:11, bold:true});
    drawTree(g, tg, 10, 300, tp);
    drawTree(g, tp, 340, 300, tg);
    const r = tedAcc(pred, GT), F = f1(pred, GT);
    txt(g, 10, 292, "TED(pr, gt) = " + r.d + "   TED(∅, gt) = " + r.d0 + "   accuracy = " + r.acc.toFixed(3) + "   field F1 = " + F.toFixed(3), {size:11, mono:true});
    $("ted-read").textContent = "TED accuracy = max(0, 1 − " + r.d + "/" + r.d0 + ") = " + r.acc.toFixed(4) + " · field F1 = " + F.toFixed(4) +
      " · red nodes appear in only one tree (or with a different label)";
  }
  on("ted-pred", "change", draw);
  draw();
});

/* ───────────────────────── 18 · accuracy against time ───────────────────────── */
DV.safe("results", function(){
  const {P, txt, $, val, on} = DV;
  const svg = d3.select("#res-svg");
  /* Donut paper (arXiv 2111.15664 v5): Table 1 (RVL-CDIP, ms), Table 2 (IE, s; params exclude vocabulary), Table 3 (DocVQA test, ms).
     [model, params in M, time in s, score, uses OCR] */
  const T1 = [["BERT",110,1.392,89.81,1],["RoBERTa",125,1.392,90.06,1],["LayoutLM",113,1.396,91.78,1],["LayoutLM (w/ image)",160,1.426,94.42,1],["LayoutLMv2",200,1.489,95.25,1],["Donut",143,0.752,95.30,0]];
  const T2 = { /* time, F1, Acc per dataset */
    cord:   [["BERT",86,1.6,73.0,65.5,1],["BROS",86,1.7,74.7,70.0,1],["LayoutLM",89,1.7,78.4,81.3,1],["LayoutLMv2",179,1.7,78.9,82.4,1],["Donut",143,1.2,84.1,90.9,0],["SPADE",93,4.0,74.0,75.8,1],["WYVERN",106,1.2,43.3,46.9,1]],
    ticket: [["BERT",86,1.7,74.3,82.4,1],["LayoutLMv2",179,1.8,87.2,90.1,1],["Donut",143,0.6,94.1,98.7,0],["SPADE",93,4.5,14.9,29.4,1],["WYVERN",106,1.5,41.8,54.8,1]],
    bc:     [["BERT",86,1.5,40.8,72.1,1],["LayoutLMv2",179,1.6,52.2,83.0,1],["Donut",143,1.4,57.8,84.4,0],["SPADE",93,4.3,32.3,51.3,1],["WYVERN",106,1.7,29.9,51.5,1]],
    rec:    [["BERT",86,2.5,70.3,54.1,1],["LayoutLMv2",179,2.6,72.9,78.0,1],["Donut",143,1.9,78.6,88.6,0],["SPADE",93,7.3,64.1,53.2,1],["WYVERN",106,3.4,71.5,82.9,1]]
  };
  const T3 = [["BERT",110,1.517,63.5,1],["LayoutLM",113,1.519,69.8,1],["LayoutLMv2",200,1.610,78.1,1],["Donut",176,0.782,67.5,0],["LayoutLMv2-Large-QG",390,1.698,86.7,1]];
  function rows(key){
    if (key === "rvl") return {rows:T1.map(r => ({m:r[0], p:r[1], t:r[2], s:r[3], ocr:r[4]})), y:"accuracy (%)", src:"Table 1"};
    if (key === "docvqa") return {rows:T3.map(r => ({m:r[0], p:r[1], t:r[2], s:r[3], ocr:r[4]})), y:"ANLS", src:"Table 3"};
    const [ds, met] = key.split("-"), idx = met === "f1" ? 3 : 4;
    return {rows:T2[ds].map(r => ({m:r[0], p:r[1], t:r[2], s:r[idx], ocr:r[5]})), y: met === "f1" ? "field F1" : "TED accuracy", src:"Table 2"};
  }
  function draw(){
    svg.selectAll("*").remove();
    const D = rows(val("res-set")), g = svg.append("g");
    const x = d3.scaleLinear().domain([0, d3.max(D.rows, r => r.t)*1.12]).range([60, 640]);
    const y = d3.scaleLinear().domain([Math.min(40, d3.min(D.rows, r => r.s)) - 5, 100]).range([260, 20]);
    const rad = d3.scaleSqrt().domain([0, 400]).range([0, 16]);
    g.append("g").attr("transform","translate(0,260)").call(d3.axisBottom(x).ticks(6)).call(s => s.selectAll("text").attr("fill",P.muted)).call(s => s.selectAll("line,path").attr("stroke",P.line));
    g.append("g").attr("transform","translate(60,0)").call(d3.axisLeft(y).ticks(6)).call(s => s.selectAll("text").attr("fill",P.muted)).call(s => s.selectAll("line,path").attr("stroke",P.line));
    txt(g, 350, 292, "seconds per document (OCR-based rows include OCR)", {anchor:"middle", size:10.5, fill:P.muted});
    txt(g, 14, 14, D.y + " · " + D.src, {size:10.5, fill:P.muted});
    D.rows.forEach(r => {
      g.append("circle").attr("cx",x(r.t)).attr("cy",y(r.s)).attr("r",rad(r.p)).attr("fill", r.ocr ? P.A : P.B).attr("opacity",0.45)
        .attr("stroke", r.ocr ? P.A : P.B).attr("stroke-dasharray", r.ocr ? "3,2" : null);
      txt(g, x(r.t) + rad(r.p) + 3, y(r.s) + 4, r.m, {size:10});
    });
    const donut = D.rows.find(r => r.m === "Donut"), base = D.rows.filter(r => r.ocr), best = base.reduce((a, b) => b.s > a.s ? b : a);
    $("res-read").textContent = "Donut " + donut.s + " at " + donut.t + " s vs best OCR-based " + best.m + " " + best.s + " at " + best.t + " s: " +
      (donut.s - best.s >= 0 ? "+" : "") + (donut.s - best.s).toFixed(2) + " points; the baseline takes " + (best.t/donut.t).toFixed(2) + "× as long · " + base.length + " OCR-based rows";
  }
  on("res-set", "change", draw);
  draw();
});

/* ───────────────────────── 19 · resolution: legibility versus cost ───────────────────────── */
DV.safe("resolution", function(){
  const {P, txt, $, val, on, encoderMACs, swinStages, fitPage, comma} = DV;
  const svg = d3.select("#rs-svg");
  /* canvases (H × W): Figure 7(c) sizes, donut-proto's 2048 × 1536, and one larger hypothetical (3840 × 2880) */
  const RES = [[640,640],[960,960],[1280,960],[2048,1536],[2560,1920],[3840,2880]];
  /* stated CORD measurements: Donut paper §3.3 text (1280 × 960) and Table 2 (2560 × 1920) */
  const CORD = {"1280x960":{acc:91.1, t:0.7}, "2560x1920":{acc:90.9, t:1.2}};
  const A4 = {w:Math.round(210/25.4*300), h:Math.round(297/25.4*300), dpi:300};
  function draw(){
    svg.selectAll("*").remove();
    const i = +val("rs-i"), pt = +val("rs-pt"), [H, W] = RES[i];
    $("rs-v").textContent = H + " × " + W;
    const ref = encoderMACs(2560, 1920, 10);
    const data = RES.map(([h, w]) => { const f = fitPage(A4.w, A4.h, h, w, false); const xh = pt/2/72*A4.dpi*f.scale;
      return {h, w, rel: encoderMACs(h, w, 10)/ref, tok: swinStages(h, w, 10)[3].n, xh, patches: xh/4, key: h+"x"+w}; });
    const g = svg.append("g");
    const bw = 44, x0 = 60;
    const yc = d3.scaleLinear().domain([0, d3.max(data, d => d.rel)]).range([0, 150]);
    txt(g, 10, 16, "encoder cost relative to 2560 × 1920", {size:10.5, fill:P.muted});
    data.forEach((d, j) => {
      const x = x0 + j*(bw+8);
      g.append("rect").attr("x",x).attr("y",180 - yc(d.rel)).attr("width",bw).attr("height",yc(d.rel)).attr("fill", j === i ? P.B : P.A).attr("opacity", j === i ? 0.9 : 0.5);
      txt(g, x + bw/2, 176 - yc(d.rel), d.rel.toFixed(2)+"×", {anchor:"middle", size:9.5, mono:true});
      txt(g, x + bw/2, 194, d.h+"×"+d.w, {anchor:"middle", size:8.5, fill:P.muted});
      const c = CORD[d.key]; if (c) txt(g, x + bw/2, 210, "CORD "+c.acc+" · "+c.t+" s", {anchor:"middle", size:8.5, fill:P.good});
    });
    const cur = data[i], lx = 400;
    txt(g, lx, 16, pt + "-pt text on an A4 page (300 dpi) at " + H + " × " + W, {size:10.5, fill:P.muted});
    const s = Math.min(140, Math.max(6, cur.xh*4));
    g.append("rect").attr("x",lx).attr("y",30).attr("width",s).attr("height",s).attr("fill",P.paper);
    const cell = s / cur.xh * 4;
    for (let v = 0; v <= s + 0.01; v += cell){
      g.append("line").attr("x1",lx+v).attr("x2",lx+v).attr("y1",30).attr("y2",30+s).attr("stroke","#8a8577").attr("stroke-width",0.5);
      g.append("line").attr("y1",30+v).attr("y2",30+v).attr("x1",lx).attr("x2",lx+s).attr("stroke","#8a8577").attr("stroke-width",0.5);
    }
    txt(g, lx + s/2, 30 + s*0.85, "x", {anchor:"middle", size:s, fill:P.paperInk});
    txt(g, lx, 30 + s + 18, "x-height " + cur.xh.toFixed(1) + " px = " + cur.patches.toFixed(1) + " stage-1 patches", {size:10.5, mono:true});
    txt(g, lx, 30 + s + 34, "stage-4 tokens: " + comma(cur.tok), {size:10.5, mono:true});
    const c = CORD[cur.key];
    $("rs-read").textContent = H + " × " + W + ": encoder cost " + cur.rel.toFixed(2) + "× that of 2560 × 1920, " + comma(cur.tok) + " stage-4 tokens, a " + pt +
      "-pt lowercase letter ≈ " + cur.xh.toFixed(1) + " px (" + cur.patches.toFixed(1) + " patches) tall" + (c ? " · paper: CORD TED accuracy " + c.acc + " at " + c.t + " s/image" : " · no accuracy value printed for this size");
  }
  on("rs-i", "input", draw);
  on("rs-pt", "change", draw);
  draw();
});

/* ───────────────────────── 23 · linear CKA ───────────────────────── */
DV.safe("cka", function(){
  const {P, txt, $, val, on, lcg, gauss} = DV;
  const svg = d3.select("#cka-svg");
  /* Illustrative teacher activations: 8 inputs × 3 features. Each column is a permutation of the same
     zero-mean values, scaled by 4, 2, 1 → column variances in the ratio 16 : 4 : 1. */
  const BASE = [1,-1,0.5,-0.5,0.8,-0.8,0.2,-0.2];
  const PERM2 = [2,5,0,7,1,4,3,6], PERM3 = [6,1,3,0,7,2,5,4];
  const X = BASE.map((v, i) => [4*v, 2*BASE[PERM2[i]], BASE[PERM3[i]]]);
  const m = X.length;
  function center(A){ const d = A[0].length, mu = []; for (let j=0;j<d;j++) mu[j] = A.reduce((s, r) => s + r[j], 0)/A.length; return A.map(r => r.map((v, j) => v - mu[j])); }
  function gram(A){ return A.map(r => A.map(q => r.reduce((s, v, j) => s + v*q[j], 0))); }
  function centerGram(K){ const n = K.length, rm = K.map(r => r.reduce((a,b) => a+b, 0)/n), all = rm.reduce((a,b) => a+b, 0)/n;
    return K.map((r, i) => r.map((v, j) => v - rm[i] - rm[j] + all)); }
  function hsic(K, L){ let s = 0; for (let i=0;i<m;i++) for (let j=0;j<m;j++) s += K[i][j]*L[i][j]; return s/((m-1)*(m-1)); }
  function matmul(A, B){ return A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v*B[k][j], 0))); }
  function fro2(A){ return A.reduce((s, r) => s + r.reduce((t, v) => t + v*v, 0), 0); }
  function T(A){ return A[0].map((_, j) => A.map(r => r[j])); }
  function student(op, noise, sp){
    const r = lcg(12345);
    let Y;
    if (op === "rot"){ const a = 40*Math.PI/180, R = [[Math.cos(a), -Math.sin(a), 0],[Math.sin(a), Math.cos(a), 0],[0,0,1]]; Y = matmul(X, R); }
    else if (op === "scale") Y = X.map(row => row.map(v => 3*v));
    else if (op === "keephi") Y = X.map(row => [row[0]]);
    else if (op === "keeplo") Y = X.map(row => [row[2]]);
    else if (op === "prune"){
      const Wm = [[0,0,0],[0,0,0],[0,0,0]].map(row => row.map(() => gauss(r)));
      const flat = []; Wm.forEach((row, i) => row.forEach((v, j) => flat.push([Math.abs(v), i, j])));
      flat.sort((a, b) => a[0] - b[0]);
      const nz = Math.round(9*sp/100); for (let k=0;k<nz;k++) Wm[flat[k][1]][flat[k][2]] = 0;
      Y = matmul(X, Wm);
    }
    else Y = X.map(() => [gauss(r), gauss(r), gauss(r)]);
    const sd0 = Math.sqrt(X.reduce((s, row) => s + row[0]*row[0], 0)/m);
    const rn = lcg(999);
    return Y.map(row => row.map(v => v + (noise/10)*sd0*gauss(rn)));
  }
  function heat(g, K, x0, y0, title){
    const mx = Math.max(1e-12, d3.max(K, r => d3.max(r, v => Math.abs(v)))), s = 22;
    txt(g, x0, y0 - 8, title, {size:10.5, fill:P.muted});
    K.forEach((row, i) => row.forEach((v, j) => g.append("rect").attr("x",x0 + j*s).attr("y",y0 + i*s).attr("width",s-1).attr("height",s-1)
      .attr("fill", v >= 0 ? P.B : P.A).attr("opacity", 0.08 + 0.9*Math.abs(v)/mx)));
  }
  function draw(){
    svg.selectAll("*").remove();
    const op = val("cka-op"), noise = +val("cka-noise"), sp = +val("cka-sp");
    const Y = student(op, noise, sp);
    const Kc = centerGram(gram(X)), Lc = centerGram(gram(Y));
    const hkl = hsic(Kc, Lc), hkk = hsic(Kc, Kc), hll = hsic(Lc, Lc);
    const ckaK = hll > 1e-12 ? hkl/Math.sqrt(hkk*hll) : 0;
    const Xc = center(X), Yc = center(Y), yx = fro2(matmul(T(Yc), Xc)), xx = Math.sqrt(fro2(matmul(T(Xc), Xc))), yy = Math.sqrt(fro2(matmul(T(Yc), Yc)));
    const ckaF = yy > 1e-12 ? yx/(xx*yy) : 0;
    const g = svg.append("g");
    heat(g, Kc, 20, 30, "teacher K′ = HXXᵀH");
    heat(g, Lc, 230, 30, "student L′ = HYYᵀH (" + Y[0].length + " feature" + (Y[0].length > 1 ? "s" : "") + ")");
    txt(g, 440, 60, "CKA = " + ckaK.toFixed(4), {size:16, bold:true, fill:P.good});
    txt(g, 440, 88, "HSIC₀(K, L) = " + hkl.toFixed(3), {size:10.5, mono:true});
    txt(g, 440, 106, "HSIC₀(K, K) = " + hkk.toFixed(3), {size:10.5, mono:true});
    txt(g, 440, 124, "HSIC₀(L, L) = " + hll.toFixed(3), {size:10.5, mono:true});
    txt(g, 440, 150, "feature form: " + ckaF.toFixed(4), {size:10.5, mono:true, fill:P.muted});
    txt(g, 440, 176, "amber +, blue −; " + m + " inputs", {size:10, fill:P.muted});
    $("cka-read").textContent = "CKA(teacher, student) = " + ckaK.toFixed(4) + " (kernel/HSIC form) = " + ckaF.toFixed(4) + " (feature form)" +
      (op === "prune" ? " · " + Math.round(9*sp/100) + " of 9 map entries zeroed" : "") + (noise ? " · noise " + (noise/10).toFixed(1) + " × the largest column s.d." : "");
  }
  on("cka-op", "change", draw);
  ["cka-noise","cka-sp"].forEach(id => on(id, "input", draw));
  draw();
});
