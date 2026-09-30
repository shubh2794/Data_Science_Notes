/* bert.viz.js — every interactive figure on models/bert.html.
   Loaded after data.js / notes.js (palette C comes from notes.js).
   Each figure is its own function wrapped in try/catch so one failure cannot blank the rest.
   House rule: every number a figure displays is computed here from a spec or from the
   published table rows listed in the code, never typed into a label. */

const BV = (function(){
  const P = Object.assign({}, (typeof C !== "undefined" ? C : {A:"#5b9cff",B:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"}),
    { panel:"#15181f", teal:"#2dd4bf", purple:"#c084fc", pink:"#f472b6" });

  /* exact BertModel parameter count (HF / original layout) */
  function params(c){
    const V = c.V, H = c.H, L = c.L, I = c.I || 4*H, Pn = c.P || 512, T = (c.T == null ? 2 : c.T);
    const tok = V*H, posseg = Pn*H + T*H, embLN = 2*H;
    const attn = L*4*(H*H + H), ffn = L*(2*H*I + I + H), ln = L*4*H;
    const pool = c.pool === false ? 0 : H*H + H;
    const heads = c.heads ? (H*H + H + 2*H + V) + (c.T === 0 ? 0 : 2*H + 2) : 0;
    const total = tok + posseg + embLN + attn + ffn + ln + pool + heads;
    return {tok, posseg, embLN, attn, ffn, ln, pool, heads, total, perLayer: 4*(H*H+H) + 2*H*I + I + H + 4*H,
            nonEmb: attn + ffn + ln, approx: 12*L*H*H + V*H};
  }
  const CFG = {
    tiny:  {name:"BERT-Tiny",   L:2,  H:128,  V:30522},
    mini:  {name:"BERT-Mini",   L:4,  H:256,  V:30522},
    small: {name:"BERT-Small",  L:4,  H:512,  V:30522},
    medium:{name:"BERT-Medium", L:8,  H:512,  V:30522},
    base:  {name:"BERT-Base",   L:12, H:768,  V:30522, paper:110e6, paperLabel:"110M (paper)"},
    large: {name:"BERT-Large",  L:24, H:1024, V:30522, paper:340e6, paperLabel:"340M (paper)"},
    distil:{name:"DistilBERT",  L:6,  H:768,  V:30522, T:0, pool:false, paper:66e6, paperLabel:"66M (DistilBERT Table 3)"},
    mbert: {name:"Multilingual BERT-Base", L:12, H:768, V:119547}
  };
  function fmtN(x){
    const a = Math.abs(x);
    if (a >= 1e12) return (x/1e12).toFixed(2)+"T";
    if (a >= 1e9)  return (x/1e9).toFixed(2)+"B";
    if (a >= 1e6)  return (x/1e6).toFixed(2)+"M";
    if (a >= 1e3)  return (x/1e3).toFixed(1)+"k";
    return String(Math.round(x));
  }
  function sup(e){ const m={"-":"⁻","0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹"}; return String(e).split("").map(ch=>m[ch]).join(""); }
  function sci(x, p){ if (!isFinite(x) || x===0) return "0"; const e = Math.floor(Math.log10(Math.abs(x))); return (x/Math.pow(10,e)).toFixed(p==null?2:p)+" × 10"+sup(e); }
  function softmax(z){ const m = Math.max(...z), e = z.map(v=>Math.exp(v-m)), s = e.reduce((a,b)=>a+b,0); return e.map(v=>v/s); }
  function txt(g, x, y, s, o){
    o = o || {};
    return g.append("text").attr("x",x).attr("y",y).attr("font-size",o.size||11).attr("fill",o.fill||P.ink)
      .attr("text-anchor",o.anchor||"start").attr("font-family",o.mono?"SF Mono,Menlo,monospace":null)
      .attr("font-weight",o.bold?600:null).text(s);
  }
  function safe(name, fn){ try { fn(); } catch(e){ if (typeof console!=="undefined") console.error("[bert.viz] "+name+" failed:", e); } }
  function lcg(seed){ let s = seed >>> 0; return () => { s = (1664525*s + 1013904223) >>> 0; return s/4294967296; }; }
  function hash(str){ let h = 2166136261 >>> 0; for (let i=0;i<str.length;i++){ h ^= str.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; } return h; }
  /* forward FLOPs of one sequence: weight matmuls + attention scores/mixing */
  function fwdFlops(L, H, n){ return {lin: 24*L*H*H*n, quad: 4*L*n*n*H}; }
  return {P, params, CFG, fmtN, sci, softmax, txt, safe, lcg, hash, fwdFlops};
})();

/* ───────────────────────── 02 · ELMo / GPT / BERT receptive field ───────────────────────── */
BV.safe("context", function(){
  const {P, txt} = BV;
  const svg = d3.select("#ctx-svg"), toks = ["he","sat","by","the","river","bank","to","fish"], n = toks.length;
  let sel = 5;
  function vis(model, i){
    if (model === "bert") return {fwd: d3.range(n), bwd: [], joint: d3.range(n)};
    if (model === "gpt")  return {fwd: d3.range(i+1), bwd: [], joint: d3.range(i+1)};
    return {fwd: d3.range(i+1), bwd: d3.range(i, n), joint: [i]};   /* elmo: two separate streams */
  }
  function draw(){
    svg.selectAll("*").remove();
    const model = d3.select("#ctx-model").property("value"), g = svg.append("g");
    const x0 = 40, cw = 70, yIn = 200, yOut = 60, bw = 58;
    const v = vis(model, sel), cx = j => x0 + j*cw + bw/2;
    txt(g, 14, yOut+4, "output", {size:10, fill:P.muted});
    txt(g, 14, yIn+26, "input", {size:10, fill:P.muted});
    /* arcs */
    const arcs = model === "elmo" ? v.fwd.map(j=>({j, col:P.A, off:-6})).concat(v.bwd.map(j=>({j, col:P.B, off:6}))) : v.joint.map(j=>({j, col:model==="gpt"?P.A:P.purple, off:0}));
    arcs.forEach(a=>{
      const x1 = cx(a.j)+a.off, x2 = cx(sel)+a.off;
      g.append("path").attr("d", `M${x1},${yIn} C${x1},${(yIn+yOut)/2} ${x2},${(yIn+yOut)/2} ${x2},${yOut+26}`)
        .attr("fill","none").attr("stroke",a.col).attr("stroke-opacity",.55).attr("stroke-width",1.6);
    });
    toks.forEach((t,j)=>{
      const on = arcs.some(a=>a.j===j);
      const gg = g.append("g").style("cursor","pointer").on("click",()=>{ sel=j; draw(); });
      gg.append("rect").attr("x",x0+j*cw).attr("y",yIn).attr("width",bw).attr("height",26).attr("rx",5)
        .attr("fill", on ? "rgba(91,156,255,.18)" : P.panel).attr("stroke", j===sel ? P.good : (on ? P.A : P.line));
      txt(gg, cx(j), yIn+17, t, {anchor:"middle", mono:true, size:11, fill: on ? P.ink : P.muted});
    });
    g.append("rect").attr("x",x0+sel*cw).attr("y",yOut).attr("width",bw).attr("height",26).attr("rx",5).attr("fill","rgba(74,222,128,.15)").attr("stroke",P.good);
    txt(g, cx(sel), yOut+17, "h("+toks[sel]+")", {anchor:"middle", mono:true, size:10.5, fill:P.good});
    if (model === "elmo"){
      txt(g, 330, 24, "blue: forward LSTM (left context) · orange: backward LSTM (right context)", {anchor:"middle", size:10, fill:P.muted});
      txt(g, 330, 40, "the two streams are concatenated only at the top", {anchor:"middle", size:10, fill:P.muted});
    } else txt(g, 330, 30, model==="gpt" ? "causal mask: every layer sees positions ≤ i" : "no mask: every layer sees all positions", {anchor:"middle", size:10, fill:P.muted});
    const right = d3.range(sel+1, n), rightSeenJoint = right.filter(j=>v.joint.includes(j));
    const name = {elmo:"ELMo", gpt:"GPT-1", bert:"BERT"}[model];
    let msg = `<b>${name}</b>, position ${sel+1} ("${toks[sel]}"): `;
    if (model === "elmo") msg += `forward stream sees <b>${v.fwd.length}</b> tokens, backward stream sees <b>${v.bwd.length}</b>; inside any single LSTM layer the two never meet, so jointly conditioned context is <b>${v.joint.length}</b> token.`;
    else msg += `every layer's representation depends on <b>${v.joint.length}</b> of ${n} tokens; <b>${rightSeenJoint.length}</b> of the ${right.length} tokens to its right${right.length ? " (" + right.map(j=>toks[j]).join(", ") + ")" : ""}.`;
    d3.select("#ctx-read").html(msg);
  }
  d3.select("#ctx-model").on("change", draw);
  draw();
});

/* ───────────────────────── 03 · the "see itself" leak, as path counts ───────────────────────── */
BV.safe("leak", function(){
  const {P, txt} = BV;
  const svg = d3.select("#lk-svg");
  function draw(){
    svg.selectAll("*").remove();
    const n = +d3.select("#lk-n").property("value"), L = +d3.select("#lk-L").property("value");
    const causal = d3.select("#lk-att").property("value") === "causal", masked = d3.select("#lk-mask").property("checked");
    d3.select("#lk-nv").text(n); d3.select("#lk-Lv").text(L);
    const t = Math.floor(n/2), out = causal ? t-1 : t;
    const allowed = (i,j) => causal ? j <= i : true;
    /* forward reach from input t, backward reach from output 'out'; path count by DP */
    const fwd = [], bwd = [], cnt = [];
    fwd[0] = d3.range(n).map(j=>j===t); cnt[0] = d3.range(n).map(j=>j===t?1:0);
    for (let l=1;l<=L;l++){
      fwd[l] = d3.range(n).map(i=>d3.range(n).some(j=>allowed(i,j) && fwd[l-1][j]));
      cnt[l] = d3.range(n).map(i=>d3.range(n).reduce((a,j)=>a + (allowed(i,j) ? cnt[l-1][j] : 0), 0));
    }
    bwd[L] = d3.range(n).map(i=>i===out);
    for (let l=L-1;l>=0;l--) bwd[l] = d3.range(n).map(j=>d3.range(n).some(i=>allowed(i,j) && bwd[l+1][i]));
    const x = j => 90 + j*(500/(n-1)), y = l => 250 - l*(210/L);
    const g = svg.append("g");
    for (let l=1;l<=L;l++) for (let i=0;i<n;i++) for (let j=0;j<n;j++){
      if (!allowed(i,j)) continue;
      const hot = fwd[l-1][j] && bwd[l][i] && !(masked && !causal);
      g.append("line").attr("x1",x(j)).attr("y1",y(l-1)).attr("x2",x(i)).attr("y2",y(l))
        .attr("stroke", hot ? P.bad : P.line).attr("stroke-opacity", hot ? .8 : .5).attr("stroke-width", hot ? 1.6 : 0.8);
    }
    for (let l=0;l<=L;l++){
      txt(g, 14, y(l)+4, l===0 ? "input" : (l===L ? "output" : "layer "+l), {size:10, fill:P.muted});
      for (let j=0;j<n;j++){
        const isT = l===0 && j===t, isO = l===L && j===out;
        g.append("circle").attr("cx",x(j)).attr("cy",y(l)).attr("r",isT||isO?8:5.5)
          .attr("fill", isT ? (masked && !causal ? P.purple : P.bad) : (isO ? P.good : P.panel)).attr("stroke", P.muted);
      }
    }
    txt(g, x(t), y(0)+22, masked && !causal ? "[MASK]" : "x"+(t+1)+" (target)", {anchor:"middle", size:10, mono:true, fill: masked && !causal ? P.purple : P.bad});
    txt(g, x(out), y(L)-14, causal ? "predicts x"+(t+1) : "predicts x"+(t+1), {anchor:"middle", size:10, mono:true, fill:P.good});
    const raw = cnt[L][out], carrying = (masked && !causal) ? 0 : raw;
    const expect = causal ? 0 : Math.pow(n, L-1);
    let msg;
    if (causal) msg = `Causal attention: the prediction of x${t+1} is made at position ${out+1}, whose inputs are positions 1…${out+1}. Paths from x${t+1} to it: <b>${raw}</b> — nothing leaks, at any depth.`;
    else if (masked) msg = `Full attention with the target replaced by [MASK]: the graph still has <b>${raw.toLocaleString("en-US")}</b> paths from position ${t+1}'s input to its output (n^(L−1) = ${expect.toLocaleString("en-US")}), but they carry the mask symbol, so paths carrying the answer = <b>${carrying}</b>. This is the MLM set-up.`;
    else msg = `Full attention with the target visible: <b>${raw.toLocaleString("en-US")}</b> paths carry x${t+1} into its own prediction (n^(L−1) = ${expect.toLocaleString("en-US")}${L===1?"; with one layer it is the direct self-edge":""}). Training would reward copying — the "see itself" problem.`;
    d3.select("#lk-read").html(msg);
  }
  ["#lk-n","#lk-L"].forEach(s=>d3.select(s).on("input", draw));
  d3.select("#lk-att").on("change", draw); d3.select("#lk-mask").on("change", draw);
  draw();
});

/* ───────────────────────── 04 · input = token + segment + position (numeric) ───────────────────────── */
BV.safe("input", function(){
  const {P, txt, lcg, hash} = BV;
  const svg = d3.select("#bert-input-svg"), D = 6;
  const ex = [
    {toks:["[CLS]","the","cat","sat","[SEP]"], seg:[0,0,0,0,0]},
    {toks:["[CLS]","my","dog","is","cute","[SEP]","he","likes","play","##ing","[SEP]"], seg:[0,0,0,0,0,0,1,1,1,1,1]}
  ];
  function vec(seed){ const r = lcg(seed); return d3.range(D).map(()=> +(2*r()-1).toFixed(2)); }
  const tokV = t => vec(hash("tok:"+t)), segV = s => vec(hash("seg:"+s)), posV = i => vec(hash("pos:"+i));
  function ln(v){ const m = d3.mean(v), va = d3.mean(v.map(a=>(a-m)*(a-m))); return v.map(a=>(a-m)/Math.sqrt(va+1e-12)); }
  const col = d3.scaleLinear().domain([-1.5,0,1.5]).range([P.bad, P.panel, P.A]).clamp(true);
  let hover = 2;
  const fmt = v => "[" + v.map(a=>(a>=0?" ":"")+a.toFixed(2)).join(",") + "]";
  function draw(){
    svg.selectAll("*").remove();
    const e = ex[+d3.select("#bert-input-sel").property("value")], n = e.toks.length;
    if (hover >= n) hover = n-1;
    const x0 = 118, cw = Math.min(58, Math.floor(510/n)-3), gap = 3;
    const rows = [
      {key:"tok", y:78,  lab:"E_tok token",    f:i=>tokV(e.toks[i])},
      {key:"seg", y:120, lab:"E_seg segment",  f:i=>segV(e.seg[i])},
      {key:"pos", y:162, lab:"E_pos position", f:i=>posV(i)},
      {key:"sum", y:216, lab:"sum e",           f:i=>tokV(e.toks[i]).map((a,k)=>a+segV(e.seg[i])[k]+posV(i)[k])},
      {key:"ln",  y:258, lab:"LayerNorm(e)",    f:i=>ln(tokV(e.toks[i]).map((a,k)=>a+segV(e.seg[i])[k]+posV(i)[k]))}
    ];
    const g = svg.append("g");
    txt(g, x0, 16, "each cell column = one position; colour = value of each of 6 toy dimensions (blue +, red −)", {size:10, fill:P.muted});
    e.toks.forEach((t,i)=>{
      const x = x0 + i*(cw+gap), on = hover===i;
      g.append("rect").attr("x",x).attr("y",28).attr("width",cw).attr("height",24).attr("rx",4).attr("fill",on?"rgba(192,132,252,.22)":P.panel).attr("stroke",on?P.purple:P.line);
      txt(g, x+cw/2, 44, t, {anchor:"middle", size:Math.min(10.5, cw/4.2), mono:true, fill:on?P.purple:P.ink});
      txt(g, x+cw/2, 66, (e.seg[i]===0?"A":"B")+" · "+i, {anchor:"middle", size:9, mono:true, fill:P.muted});
    });
    rows.forEach(r=>{
      txt(g, x0-8, r.y+15, r.lab, {anchor:"end", size:10, mono:true, fill: r.key==="ln" ? P.good : P.muted});
      e.toks.forEach((t,i)=>{
        const x = x0 + i*(cw+gap), v = r.f(i), on = hover===i;
        v.forEach((a,k)=> g.append("rect").attr("x",x+k*(cw/D)).attr("y",r.y).attr("width",cw/D-1).attr("height",22).attr("fill",col(a)).attr("fill-opacity",on?1:.7));
        if (on) g.append("rect").attr("x",x-1).attr("y",r.y-1).attr("width",cw+1).attr("height",24).attr("fill","none").attr("stroke",P.purple);
      });
    });
    txt(g, x0-8, 112, "+", {anchor:"end", size:13, fill:P.muted}); txt(g, x0-8, 154, "+", {anchor:"end", size:13, fill:P.muted});
    txt(g, x0-8, 208, "=", {anchor:"end", size:13, fill:P.muted});
    e.toks.forEach((t,i)=>{
      const x = x0 + i*(cw+gap);
      g.append("rect").attr("x",x).attr("y",26).attr("width",cw).attr("height",260).attr("fill","transparent").style("cursor","pointer")
        .on("mouseenter",()=>{ hover=i; draw(); });
    });
    const i = hover, tv = rows[0].f(i), sv = rows[1].f(i), pv = rows[2].f(i), su = rows[3].f(i), lv = rows[4].f(i);
    const twin = e.toks.findIndex((t,k)=>k!==i && t===e.toks[i]);
    const m = d3.mean(su), sd = Math.sqrt(d3.mean(su.map(a=>(a-m)*(a-m))));
    txt(g, x0, 306, `sum: mean ${m.toFixed(3)}, std ${sd.toFixed(3)}  →  LayerNorm: mean ${d3.mean(lv).toFixed(3)}, std ${Math.sqrt(d3.mean(lv.map(a=>a*a))).toFixed(3)}`, {size:10, mono:true, fill:P.muted});
    d3.select("#bert-input-read").html(`<b>${e.toks[i]}</b> at position ${i}, segment ${e.seg[i]===0?"A":"B"}:<br><code>E_tok ${fmt(tv)}</code><br><code>E_seg ${fmt(sv)}</code><br><code>E_pos ${fmt(pv)}</code><br><code>sum&nbsp; ${fmt(su)}</code><br><code>LN&nbsp;&nbsp; ${fmt(lv)}</code>` +
      (twin >= 0 ? `<br>"${e.toks[i]}" also appears at position ${twin}: same token row, different position${e.seg[twin]!==e.seg[i]?" and segment":""} row, so a different input vector.` : ""));
  }
  d3.select("#bert-input-sel").on("change", ()=>{ hover = 2; draw(); });
  draw();
});

/* ───────────────────────── 05 · WordPiece greedy longest match ───────────────────────── */
BV.safe("wordpiece", function(){
  const {P, txt} = BV;
  const VOCAB = ["un","##aff","##able","play","##ing","hair","##y","flight","##less","em","##bed","##ding","##s","naive","q","a","##a","##e"];
  const VS = new Set(VOCAB);
  function strip(w){ return w.normalize("NFD").replace(/[̀-ͯ]/g, ""); }
  function wp(word){
    const w = strip(word), steps = [], out = [];
    let start = 0;
    while (start < w.length){
      let end = w.length, cur = null;
      while (start < end){
        let sub = w.slice(start, end); if (start > 0) sub = "##" + sub;
        const hit = VS.has(sub); steps.push({sub, hit});
        if (hit){ cur = sub; break; }
        end--;
      }
      if (cur === null) return {w, steps, out:["[UNK]"], unk:true};
      out.push(cur); start = end;
    }
    return {w, steps, out, unk:false};
  }
  const svg = d3.select("#wp-svg");
  function draw(){
    svg.selectAll("*").remove();
    const word = d3.select("#wp-word").property("value"), r = wp(word), g = svg.append("g");
    txt(g, 16, 18, `input "${word}"` + (r.w !== word ? ` → after accent stripping "${r.w}"` : ""), {size:11, bold:true});
    const per = 13;
    r.steps.forEach((s,k)=>{
      const col = Math.floor(k/per), row = k%per, x = 16 + col*200, y = 40 + row*17;
      txt(g, x, y, `${String(k+1).padStart(2," ")}. ${s.sub}`, {mono:true, size:10.5, fill: s.hit ? P.good : P.muted});
      txt(g, x+160, y, s.hit ? "✓" : "✗", {size:10.5, fill: s.hit ? P.good : P.bad});
    });
    const vx = 440;
    txt(g, vx, 40, "toy vocabulary", {size:10, fill:P.muted});
    VOCAB.forEach((v,k)=> txt(g, vx + (k%2)*90, 58 + Math.floor(k/2)*16, v, {mono:true, size:10.5, fill: r.out.includes(v) ? P.B : P.ink}));
    let x = 16; const y = 268;
    txt(g, 16, y-10, "output pieces", {size:10, fill:P.muted});
    r.out.forEach(p=>{
      const w = Math.max(34, p.length*8+14);
      g.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",24).attr("rx",4).attr("fill", r.unk ? "rgba(248,113,113,.2)" : "rgba(255,180,84,.18)").attr("stroke", r.unk ? P.bad : P.B);
      txt(g, x+w/2, y+16, p, {anchor:"middle", mono:true, size:11}); x += w + 6;
    });
    const hits = r.steps.filter(s=>s.hit).length;
    d3.select("#wp-read").html(r.unk
      ? `No vocabulary entry matches the remainder after ${hits} piece${hits===1?"":"s"}, so the <b>whole word</b> becomes <b>[UNK]</b> — ${r.steps.length} lookups were spent before giving up.`
      : `<b>${r.out.join(" ")}</b> — ${r.out.length} piece${r.out.length===1?"":"s"} from ${r.steps.length} vocabulary lookups (${r.steps.length - hits} misses). Every piece after the first carries <code>##</code>.`);
  }
  d3.select("#wp-word").on("change", draw);
  draw();
});

/* ───────────────────────── 08 · parameter calculator ───────────────────────── */
BV.safe("params", function(){
  const {P, params, CFG, fmtN, txt} = BV;
  const svg = d3.select("#bp-svg");
  let key = "base", st = Object.assign({}, CFG.base);
  function sync(){
    [["L",st.L],["H",st.H],["V",st.V]].forEach(([k,v])=>{ d3.select("#bp-"+k).property("value", v); d3.select("#bp-"+k+"v").text(v); });
    d3.select("#bp-pool").property("checked", st.pool !== false);
  }
  function draw(){
    svg.selectAll("*").remove();
    const heads = d3.select("#bp-mlm").property("checked");
    const c = Object.assign({}, st, {pool: d3.select("#bp-pool").property("checked"), heads});
    const r = params(c), g = svg.append("g");
    const parts = [["token embeddings",r.tok,P.B],["positions + segments",r.posseg + r.embLN,P.pink],["attention Q,K,V,O",r.attn,P.A],["feed-forward",r.ffn,P.teal],["layer LayerNorms",r.ln,P.purple],["pooler",r.pool,P.good],["MLM + NSP heads",r.heads,P.muted]];
    const x0 = 20, W = 600, y = 36;
    txt(g, x0, 20, `${key ? CFG[key].name : "custom"} — L ${c.L}, H ${c.H}, A ${Math.max(1, Math.round(c.H/64))} (H/64), I ${4*c.H}, V ${c.V.toLocaleString("en-US")}${c.T===0?", no segment table":""}`, {size:11, bold:true});
    let x = x0;
    parts.forEach(([lab,v,colr])=>{ const w = W*v/r.total; g.append("rect").attr("x",x).attr("y",y).attr("width",Math.max(0,w)).attr("height",30).attr("fill",colr).attr("fill-opacity",.75); x += w; });
    parts.forEach(([lab,v,colr],i)=>{
      const yy = 84 + i*22;
      g.append("rect").attr("x",x0).attr("y",yy).attr("width",11).attr("height",11).attr("fill",colr).attr("fill-opacity",.75);
      txt(g, x0+18, yy+10, lab, {size:10.5});
      txt(g, x0+230, yy+10, v.toLocaleString("en-US"), {size:10.5, mono:true, anchor:"end"});
      txt(g, x0+285, yy+10, (100*v/r.total).toFixed(1)+"%", {size:10.5, mono:true, anchor:"end", fill:P.muted});
    });
    const cfg = key ? CFG[key] : null;
    txt(g, 340, 96, "exact total", {size:10.5, fill:P.muted}); txt(g, 610, 96, r.total.toLocaleString("en-US"), {size:12, mono:true, anchor:"end", bold:true});
    txt(g, 340, 120, "12·L·H² + V·H", {size:10.5, fill:P.muted}); txt(g, 610, 120, fmtN(r.approx), {size:11, mono:true, anchor:"end"});
    txt(g, 340, 144, "one encoder layer", {size:10.5, fill:P.muted}); txt(g, 610, 144, r.perLayer.toLocaleString("en-US"), {size:11, mono:true, anchor:"end"});
    if (cfg && cfg.paper){ txt(g, 340, 168, "reported", {size:10.5, fill:P.muted}); txt(g, 610, 168, cfg.paperLabel, {size:11, mono:true, anchor:"end", fill:P.B}); }
    let rep = "";
    if (cfg && cfg.paper && !heads){ const d = r.total - cfg.paper; rep = ` Against ${cfg.paperLabel}: ${d>=0?"+":"−"}${fmtN(Math.abs(d))} (${(100*d/cfg.paper).toFixed(1)}%).`; }
    d3.select("#bp-read").html(`exact <b>${r.total.toLocaleString("en-US")}</b> (${fmtN(r.total)}); embeddings are <b>${(100*(r.tok+r.posseg+r.embLN)/r.total).toFixed(1)}%</b>; the approximation 12·L·H² + V·H is off by ${(100*(r.approx-r.total)/r.total).toFixed(2)}%.${rep}`);
  }
  d3.select("#bp-preset").on("change", function(){ key = this.value; st = Object.assign({}, CFG[key]); sync(); draw(); });
  ["L","H","V"].forEach(k=> d3.select("#bp-"+k).on("input", function(){ st[k] = +this.value; d3.select("#bp-"+k+"v").text(this.value);
    const c = CFG[key]; if (c && (c.L!==st.L || c.H!==st.H || c.V!==st.V)) key = null; draw(); }));
  d3.select("#bp-pool").on("change", draw); d3.select("#bp-mlm").on("change", draw);
  sync(); draw();
});

/* ───────────────────────── 09 · MLM corruption, run for real ───────────────────────── */
BV.safe("mlm", function(){
  const {P, txt, lcg} = BV;
  const svg = d3.select("#bert-svg");
  const SENTS = [
    ["[CLS]","my","dog","is","hairy","and","he","likes","play","##ing","in","the","park","[SEP]"],
    ["[CLS]","the","man","went","to","the","store","[SEP]","he","bought","a","gallon","of","milk","[SEP]"],
    ["[CLS]","paris","is","the","capital","of","france","and","its","largest","city","[SEP]"]
  ];
  const RANDOM_POOL = ["apple","river","seven","blue","ran","##ly","window","of","chair","sing","cloud","fast"];
  const MAXPRED = 20, SPECIAL = new Set(["[CLS]","[SEP]"]);
  let seed = 7;
  /* the released generator's rule: k = min(cap, max(1, round(n·rate))), candidates exclude specials */
  function corrupt(toks, rate, rnd){
    const cand = toks.map((t,i)=>i).filter(i=>!SPECIAL.has(toks[i]));
    for (let i=cand.length-1;i>0;i--){ const j = Math.floor(rnd()*(i+1)); [cand[i],cand[j]] = [cand[j],cand[i]]; }
    const k = Math.min(MAXPRED, Math.max(1, Math.round(toks.length*rate)), cand.length);
    const chosen = cand.slice(0,k), shown = toks.slice(), kind = toks.map(()=>null);
    chosen.forEach(i=>{ const u = rnd(); if (u < 0.8){ shown[i] = "[MASK]"; kind[i] = "mask"; } else if (u < 0.9){ shown[i] = RANDOM_POOL[Math.floor(rnd()*RANDOM_POOL.length)]; kind[i] = "rnd"; } else kind[i] = "same"; });
    return {shown, kind, k};
  }
  const tallyCache = {};
  function tally(rate){
    const key = rate.toFixed(2); if (tallyCache[key]) return tallyCache[key];
    const rnd = lcg(12345), n = 128, T = 20000, toks = ["[CLS]"].concat(d3.range(n-2).map(i=>"w"+i), ["[SEP]"]);
    const c = {mask:0, rnd:0, same:0, none:0};
    for (let s=0;s<T;s++){ const r = corrupt(toks, rate, rnd); r.kind.forEach(kd=>{ c[kd || "none"]++; }); }
    const tot = n*T; const out = {mask:c.mask/tot, rnd:c.rnd/tot, same:c.same/tot, none:c.none/tot, k:Math.min(MAXPRED, Math.max(1, Math.round(n*rate)))};
    tallyCache[key] = out; return out;
  }
  function row(g, y, toks, colorFn, label){
    txt(g, 10, y+15, label, {size:10, fill:P.muted});
    let x = 78; const w = Math.min(40, Math.floor(548/toks.length)-3);
    toks.forEach((t,i)=>{
      const c = colorFn(i);
      g.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",22).attr("rx",4).attr("fill",c.fill).attr("stroke",c.stroke);
      txt(g, x+w/2, y+15, t, {anchor:"middle", mono:true, size: Math.min(9.5, 60/Math.max(3,t.length)+3), fill:c.ink||P.ink});
      x += w + 3;
    });
  }
  function draw(){
    svg.selectAll("*").remove();
    const toks = SENTS[+d3.select("#bert-sel").property("value")], rate = +d3.select("#bert-rate").property("value");
    d3.select("#bert-ratev").text(rate.toFixed(2));
    const r = corrupt(toks, rate, lcg(seed*7919 + 1)), g = svg.append("g");
    const kc = {mask:{fill:"rgba(192,132,252,.25)",stroke:P.purple}, rnd:{fill:"rgba(248,113,113,.22)",stroke:P.bad}, same:{fill:"rgba(74,222,128,.18)",stroke:P.good}};
    row(g, 14, toks, ()=>({fill:P.panel, stroke:P.line}), "original");
    row(g, 46, r.shown, i=> r.kind[i] ? kc[r.kind[i]] : {fill:P.panel, stroke:P.line}, "input x̃");
    row(g, 78, toks.map((t,i)=> r.kind[i] ? t : "·"), i=> r.kind[i] ? {fill:"rgba(255,180,84,.18)", stroke:P.B} : {fill:"transparent", stroke:P.line, ink:P.muted}, "target");
    txt(g, 78, 118, "purple: [MASK] · red: random token · green: unchanged · orange: positions that carry the loss", {size:9.5, fill:P.muted});
    /* tally */
    const t = tally(rate), items = [["[MASK]", t.mask, 0.8*rate, P.purple],["random", t.rnd, 0.1*rate, P.bad],["unchanged", t.same, 0.1*rate, P.good],["no loss", t.none, 1-rate, P.muted]];
    txt(g, 16, 146, `20,000 sequences of 128 tokens (k = ${t.k} predictions each, cap ${MAXPRED}): share of ALL positions`, {size:10.5, bold:true});
    const x = d3.scaleLinear().domain([0,1]).range([120, 520]);
    items.forEach(([lab, obs, exp, colr], i)=>{
      const yy = 162 + i*38;
      txt(g, 112, yy+14, lab, {anchor:"end", size:10.5});
      g.append("rect").attr("x",x(0)).attr("y",yy).attr("width",x(obs)-x(0)).attr("height",18).attr("fill",colr).attr("fill-opacity",.7);
      g.append("line").attr("x1",x(exp)).attr("x2",x(exp)).attr("y1",yy-3).attr("y2",yy+21).attr("stroke",P.ink).attr("stroke-dasharray","3,2");
      txt(g, 528, yy+14, (100*obs).toFixed(2)+"%  (nominal "+(100*exp).toFixed(1)+"%)", {size:10, mono:true, fill:P.muted});
    });
    const nm = r.kind.filter(k=>k==="mask").length, nr = r.kind.filter(k=>k==="rnd").length, ns = r.kind.filter(k=>k==="same").length;
    const chosenFrac = t.mask + t.rnd + t.same;
    d3.select("#bert-read").html(`This draw (seed ${seed}): n = ${toks.length} tokens → k = <b>${r.k}</b> chosen (${nm} [MASK], ${nr} random, ${ns} unchanged); loss is computed at those ${r.k} positions only. ` +
      `Over the 20,000-sequence tally, <b>${(100*chosenFrac).toFixed(2)}%</b> of positions are chosen (not exactly ${(100*rate).toFixed(0)}%: round(128 × ${rate.toFixed(2)}) = ${Math.round(128*rate)}${Math.round(128*rate) > MAXPRED ? `, capped at ${MAXPRED}` : ""}, out of 128 positions including [CLS] and [SEP]).`);
  }
  d3.select("#bert-sel").on("change", draw);
  d3.select("#bert-resample").on("click", ()=>{ seed++; draw(); });
  d3.select("#bert-rate").on("input", draw);
  draw();
});

/* ───────────────────────── 10 · masking-strategy ablation (Table 8) ───────────────────────── */
BV.safe("maskabl", function(){
  const {P, txt} = BV;
  /* BERT paper Table 8: MASK / SAME / RND probabilities and dev results */
  const ROWS = [
    {m:80, s:10, r:10, mnli:84.2, nerft:95.4, nerfb:94.9},
    {m:100,s:0,  r:0,  mnli:84.3, nerft:94.9, nerfb:94.0},
    {m:80, s:0,  r:20, mnli:84.1, nerft:95.2, nerfb:94.6},
    {m:80, s:20, r:0,  mnli:84.4, nerft:95.2, nerfb:94.7},
    {m:0,  s:20, r:80, mnli:83.7, nerft:94.8, nerfb:94.6},
    {m:0,  s:0,  r:100,mnli:83.6, nerft:94.9, nerfb:94.6}
  ];
  const svg = d3.select("#ma-svg");
  function draw(){
    svg.selectAll("*").remove();
    const k = d3.select("#ma-metric").property("value"), ref = ROWS[0][k], g = svg.append("g");
    const d = ROWS.map(r=>r[k]-ref), lim = Math.max(1, d3.max(d.map(Math.abs)));
    const x = d3.scaleLinear().domain([-lim, lim]).range([250, 600]);
    g.append("line").attr("x1",x(0)).attr("x2",x(0)).attr("y1",16).attr("y2",236).attr("stroke",P.muted);
    txt(g, x(0), 250, "Δ vs 80/10/10 (points)", {anchor:"middle", size:10, fill:P.muted});
    [-lim, lim].forEach(v=> txt(g, x(v), 250, (v>0?"+":"")+v.toFixed(1), {anchor:"middle", size:9.5, fill:P.muted}));
    ROWS.forEach((r,i)=>{
      const y = 22 + i*36, dv = d[i];
      txt(g, 16, y+13, `MASK ${r.m}% · SAME ${r.s}% · RND ${r.r}%`, {size:10.5, mono:true, fill: i===0 ? P.B : P.ink});
      g.append("rect").attr("x", Math.min(x(0), x(dv))).attr("y", y).attr("width", Math.abs(x(dv)-x(0))).attr("height", 18)
        .attr("fill", i===0 ? P.B : (dv >= 0 ? P.good : P.bad)).attr("fill-opacity", .75);
      txt(g, x(0) + (dv>=0?6:-6) + (dv>=0 ? Math.abs(x(dv)-x(0)) : -Math.abs(x(dv)-x(0))), y+13, r[k].toFixed(1), {anchor: dv>=0?"start":"end", size:10, mono:true, fill:P.muted});
    });
    const vals = ROWS.map(r=>r[k]), best = ROWS[d3.maxIndex(vals)], worst = ROWS[d3.minIndex(vals)];
    const name = {mnli:"MNLI (fine-tune)", nerft:"NER (fine-tune)", nerfb:"NER (feature-based)"}[k];
    d3.select("#ma-read").html(`<b>${name}</b>: spread across the six strategies is <b>${(d3.max(vals)-d3.min(vals)).toFixed(1)}</b> points. Best: ${best.m}/${best.s}/${best.r} (${best[k].toFixed(1)}); worst: ${worst.m}/${worst.s}/${worst.r} (${worst[k].toFixed(1)}). 100% MASK is ${(ROWS[1][k]-ref>=0?"+":"")}${(ROWS[1][k]-ref).toFixed(1)} relative to BERT's choice.`);
  }
  d3.select("#ma-metric").on("change", draw);
  draw();
});

/* ───────────────────────── 12 · NSP vs SOP with a word-overlap baseline ───────────────────────── */
BV.safe("nsp", function(){
  const {P, txt} = BV;
  const DOCS = [
    ["preheat the oven and grease the cake tray","mix flour and sugar for the cake batter","pour the cake batter into the tray","bake the cake until the oven timer rings"],
    ["the striker took the ball into the box","the striker beat two defenders in the box","her shot beat the keeper at the near post","the keeper could not stop the shot"],
    ["the telescope tracked a faint comet overnight","the comet tail pointed away from the sun","astronomers measured the comet orbit around the sun","the orbit brings the comet back every century"],
    ["the bank raised its interest rate today","savers earn more interest at the bank","borrowers pay the bank a higher rate on each loan","each loan now costs more than last year"]
  ];
  const STOP = new Set("the a an and of to in into at on for from its her she it is was be by with than more each could not until every away around back now last two near this that".split(" "));
  const bag = s => new Set(s.split(" ").filter(w=>!STOP.has(w)));
  const jac = (a,b)=>{ const A = bag(a), B = bag(b); let i=0; A.forEach(w=>{ if (B.has(w)) i++; }); return i / (A.size + B.size - i); };
  const pos = [], nspNeg = [], sopNeg = [];
  DOCS.forEach((d,di)=> d.forEach((s,si)=>{
    if (si < d.length-1){
      pos.push({a:s, b:d[si+1], v:jac(s, d[si+1])});
      sopNeg.push({a:d[si+1], b:s, v:jac(d[si+1], s)});
      DOCS.forEach((d2,dj)=>{ if (dj!==di) d2.forEach(s2=> nspNeg.push({a:s, b:s2, v:jac(s, s2)})); });
    }
  }));
  function bacc(neg, th){ const tpr = pos.filter(p=>p.v > th).length/pos.length, tnr = neg.filter(p=>p.v <= th).length/neg.length; return {tpr, tnr, b:(tpr+tnr)/2}; }
  const svg = d3.select("#nsp-svg");
  function draw(){
    svg.selectAll("*").remove();
    const task = d3.select("#nsp-task").property("value"), th = +d3.select("#nsp-th").property("value");
    d3.select("#nsp-thv").text(th.toFixed(2));
    const neg = task === "nsp" ? nspNeg : sopNeg, g = svg.append("g");
    const bins = d3.range(0, 0.6001, 0.05);
    const hist = arr => bins.slice(0,-1).map((b,i)=> arr.filter(p=> p.v >= b && (i === bins.length-2 ? p.v <= bins[i+1] : p.v < bins[i+1])).length / arr.length);
    const hp = hist(pos), hn = hist(neg);
    const x = d3.scaleLinear().domain([0,0.6]).range([60, 600]), y = d3.scaleLinear().domain([0, Math.max(d3.max(hp), d3.max(hn))]).range([220, 30]);
    const bw = (x(0.05)-x(0))/2 - 1;
    hp.forEach((v,i)=> g.append("rect").attr("x",x(bins[i])+1).attr("y",y(v)).attr("width",bw).attr("height",220-y(v)).attr("fill",P.good).attr("fill-opacity",.75));
    hn.forEach((v,i)=> g.append("rect").attr("x",x(bins[i])+bw+2).attr("y",y(v)).attr("width",bw).attr("height",220-y(v)).attr("fill",P.bad).attr("fill-opacity",.65));
    g.append("line").attr("x1",60).attr("x2",600).attr("y1",220).attr("y2",220).attr("stroke",P.muted);
    [0,0.1,0.2,0.3,0.4,0.5,0.6].forEach(v=> txt(g, x(v), 236, v.toFixed(1), {anchor:"middle", size:9.5, fill:P.muted}));
    txt(g, 330, 256, "word overlap (Jaccard of content words)", {anchor:"middle", size:10, fill:P.muted});
    txt(g, 56, 34, "share of pairs", {anchor:"end", size:9.5, fill:P.muted});
    g.append("line").attr("x1",x(th)).attr("x2",x(th)).attr("y1",24).attr("y2",222).attr("stroke",P.B).attr("stroke-dasharray","4,3");
    txt(g, x(th)+4, 24, "threshold", {size:9.5, fill:P.B});
    txt(g, 440, 44, `■ positives (${pos.length})`, {size:10, fill:P.good});
    txt(g, 440, 60, `■ ${task==="nsp"?"random-document":"swapped-order"} negatives (${neg.length})`, {size:10, fill:P.bad});
    const r = bacc(neg, th);
    const best = d3.range(0, 0.61, 0.01).map(t=>({t, b:bacc(neg, t).b})).reduce((a,b)=> b.b > a.b ? b : a);
    d3.select("#nsp-read").html(`${task.toUpperCase()}: predicting "positive" when overlap &gt; ${th.toFixed(2)} gets <b>${(100*r.tpr).toFixed(0)}%</b> of positives and <b>${(100*r.tnr).toFixed(0)}%</b> of negatives right — balanced accuracy <b>${(100*r.b).toFixed(1)}%</b>. The best threshold on this corpus reaches ${(100*best.b).toFixed(1)}%` +
      (task === "sop" ? " — exactly chance, because every swapped pair has the same overlap as its original." : (best.b >= 0.999 ? " with no model at all: topic words alone separate every random negative here." : " with no model at all: topic words alone separate most random negatives.")));
  }
  d3.select("#nsp-task").on("change", draw); d3.select("#nsp-th").on("input", draw);
  draw();
});

/* ───────────────────────── 13 · objective ablations: BERT Table 5 vs RoBERTa Table 2 ───────────────────────── */
BV.safe("ablate", function(){
  const {P, txt} = BV;
  const SRC = {
    bert: {tasks:["MNLI-m","QNLI","MRPC","SST-2","SQuAD 1.1 F1"], ref:0, rows:[
      ["BERT-Base (MLM + NSP)", [84.4,88.4,86.7,92.7,88.5]],
      ["No NSP", [83.9,84.9,86.5,92.6,87.9]],
      ["LTR & No NSP", [82.1,84.3,77.5,92.1,77.8]],
      ["+ BiLSTM on top", [82.1,84.1,75.7,91.6,84.9]]]},
    roberta: {tasks:["SQuAD 1.1 F1","SQuAD 2.0 F1","MNLI-m","SST-2","RACE"], ref:0, rows:[
      ["SEGMENT-PAIR + NSP", [90.4,78.7,84.0,92.9,64.2]],
      ["SENTENCE-PAIR + NSP", [88.7,76.2,82.9,92.1,63.0]],
      ["FULL-SENTENCES, no NSP", [90.4,79.1,84.7,92.5,64.8]],
      ["DOC-SENTENCES, no NSP", [90.6,79.7,84.7,92.7,65.6]],
      ["BERT-Base as published", [88.5,76.3,84.3,92.8,64.3]]]}
  };
  const svg = d3.select("#ab-svg");
  function fillTasks(){
    const s = SRC[d3.select("#ab-src").property("value")];
    const sel = d3.select("#ab-task"); sel.selectAll("option").remove();
    s.tasks.forEach((t,i)=> sel.append("option").attr("value", i).text(t));
    sel.property("value", "0");
  }
  function draw(){
    svg.selectAll("*").remove();
    const s = SRC[d3.select("#ab-src").property("value")], ti = +d3.select("#ab-task").property("value") || 0, g = svg.append("g");
    const vals = s.rows.map(r=>r[1][ti]), lo = Math.floor(d3.min(vals) - 2), hi = Math.ceil(d3.max(vals) + 1);
    const x = d3.scaleLinear().domain([lo, hi]).range([210, 590]);
    s.rows.forEach((r,i)=>{
      const y = 20 + i*44, v = r[1][ti], dv = v - s.rows[s.ref][1][ti];
      txt(g, 200, y+15, r[0], {anchor:"end", size:10.5, fill: i===s.ref ? P.B : P.ink});
      g.append("rect").attr("x",x(lo)).attr("y",y).attr("width",x(v)-x(lo)).attr("height",20).attr("fill", i===s.ref ? P.B : (dv >= 0 ? P.good : P.A)).attr("fill-opacity",.7);
      txt(g, x(v)+6, y+15, v.toFixed(1) + (i===s.ref ? "" : `  (${dv>=0?"+":""}${dv.toFixed(1)})`), {size:10, mono:true, fill:P.muted});
    });
    txt(g, x(lo), 246, `axis starts at ${lo} (not zero) to show differences`, {size:9.5, fill:P.muted});
    const refName = s.rows[s.ref][0];
    const ds = s.rows.map(r=>r[1][ti] - s.rows[s.ref][1][ti]);
    const worst = s.rows[d3.minIndex(ds)], bestAlt = s.rows.filter((r,i)=>i!==s.ref).reduce((a,b)=> b[1][ti] > a[1][ti] ? b : a);
    d3.select("#ab-read").html(`<b>${s.tasks[ti]}</b>, relative to "${refName}": largest drop <b>${d3.min(ds).toFixed(1)}</b> (${worst[0]}); best alternative "${bestAlt[0]}" at ${(bestAlt[1][ti] - s.rows[s.ref][1][ti] >= 0 ? "+" : "")}${(bestAlt[1][ti] - s.rows[s.ref][1][ti]).toFixed(1)}.`);
  }
  d3.select("#ab-src").on("change", ()=>{ fillTasks(); draw(); });
  d3.select("#ab-task").on("change", draw);
  fillTasks(); draw();
});

/* ───────────────────────── 14 · pretraining schedule and token accounting ───────────────────────── */
BV.safe("sched", function(){
  const {P, txt, fmtN, sci, fwdFlops} = BV;
  const STEPS = 1e6, BATCH = 256, WARM = 1e4, LR = 1e-4, WORDS = 3.3e9, DAYS = 4;
  const M = {base:{L:12, H:768, chips:16}, large:{L:24, H:1024, chips:64}};
  const svg = d3.select("#sc-svg");
  function draw(){
    svg.selectAll("*").remove();
    const f = +d3.select("#sc-f").property("value"), m = M[d3.select("#sc-model").property("value")], g = svg.append("g");
    d3.select("#sc-fv").text(f.toFixed(2));
    const x = d3.scaleLinear().domain([0, STEPS]).range([60, 610]);
    const yl = d3.scaleLinear().domain([0, LR]).range([150, 24]);
    /* learning rate: linear warm-up then linear decay to 0 */
    const lr = s => s < WARM ? LR*s/WARM : LR*(STEPS - s)/(STEPS - WARM);
    const pts = [0, WARM].concat(d3.range(50000, STEPS+1, 50000));
    g.append("path").attr("d", d3.line().x(s=>x(s)).y(s=>yl(lr(s)))(pts)).attr("fill","none").attr("stroke",P.A).attr("stroke-width",2);
    txt(g, 56, yl(LR)+4, "1e-4", {anchor:"end", size:9.5, fill:P.muted}); txt(g, 56, 154, "0", {anchor:"end", size:9.5, fill:P.muted});
    txt(g, x(WARM)+6, yl(LR)+2, "warm-up ends at 10k steps", {size:9.5, fill:P.A});
    txt(g, 14, 90, "lr", {size:10, fill:P.muted});
    /* sequence-length phases, bar height ∝ tokens per step */
    const s1 = f*STEPS, yb = 250, hmax = 70;
    g.append("rect").attr("x",x(0)).attr("y",yb - hmax*128/512).attr("width",x(s1)-x(0)).attr("height",hmax*128/512).attr("fill",P.teal).attr("fill-opacity",.6);
    g.append("rect").attr("x",x(s1)).attr("y",yb - hmax).attr("width",x(STEPS)-x(s1)).attr("height",hmax).attr("fill",P.purple).attr("fill-opacity",.6);
    if (f > 0.08) txt(g, (x(0)+x(s1))/2, yb-hmax*128/512-5, "length 128", {anchor:"middle", size:10, fill:P.teal});
    if (f < 0.92) txt(g, (x(s1)+x(STEPS))/2, yb-hmax-5, "length 512", {anchor:"middle", size:10, fill:P.purple});
    [0,250000,500000,750000,1000000].forEach(s=> txt(g, x(s), 266, (s/1000)+"k", {anchor:"middle", size:9.5, fill:P.muted}));
    txt(g, 14, yb-30, "tokens", {size:10, fill:P.muted}); txt(g, 14, yb-18, "per step", {size:10, fill:P.muted});
    const t1 = s1*BATCH*128, t2 = (STEPS-s1)*BATCH*512, tot = t1 + t2, full = STEPS*BATCH*512;
    const F1 = fwdFlops(m.L, m.H, 128), F2 = fwdFlops(m.L, m.H, 512);
    const trainF = 3*BATCH*(s1*(F1.lin+F1.quad) + (STEPS-s1)*(F2.lin+F2.quad));
    const perChip = trainF / (m.chips * DAYS * 86400);
    d3.select("#sc-read").html(`Tokens processed: ${fmtN(t1)} at length 128 + ${fmtN(t2)} at 512 = <b>${fmtN(tot)}</b> → <b>${(tot/WORDS).toFixed(1)}</b> passes over 3.3B words. ` +
      `If every step used 512-token sequences: ${fmtN(full)} → ${(full/WORDS).toFixed(1)} passes (the paper's "approximately 40 epochs"; its "128,000 tokens/batch" is 256 × 512 = ${(BATCH*512).toLocaleString("en-US")}). ` +
      `Matrix-multiply training FLOPs ≈ 3 × forward = <b>${sci(trainF)}</b>, i.e. an implied sustained ${sci(perChip,1)} FLOP/s per chip over ${DAYS} days on ${m.chips} chips — an estimate that ignores the embedding and MLM output layers.`);
  }
  d3.select("#sc-f").on("input", draw); d3.select("#sc-model").on("change", draw);
  draw();
});

/* ───────────────────────── 15 · FLOPs vs sequence length ───────────────────────── */
BV.safe("flops", function(){
  const {P, txt, sci, fwdFlops, params} = BV;
  const M = {base:{L:12, H:768, name:"BERT-Base"}, large:{L:24, H:1024, name:"BERT-Large"}};
  const svg = d3.select("#fl-svg"), W = 640, H0 = 270, m = {l:64, r:20, t:18, b:36};
  function draw(){
    svg.selectAll("*").remove();
    const mm = M[d3.select("#fl-model").property("value")], n = +d3.select("#fl-n").property("value"), g = svg.append("g");
    d3.select("#fl-nv").text(n);
    const ns = d3.range(Math.log2(16), Math.log2(8192)+0.001, 0.1).map(e=>Math.pow(2,e));
    const lin = ns.map(k=>fwdFlops(mm.L, mm.H, k).lin), quad = ns.map(k=>fwdFlops(mm.L, mm.H, k).quad);
    const x = d3.scaleLog().domain([16, 8192]).range([m.l, W-m.r]);
    const y = d3.scaleLog().domain([d3.min(quad), d3.max(lin.map((v,i)=>v+quad[i]))*1.2]).range([H0-m.b, m.t]);
    [16,64,256,512,1024,4096,8192].forEach(v=>{ g.append("line").attr("x1",x(v)).attr("x2",x(v)).attr("y1",m.t).attr("y2",H0-m.b).attr("stroke",P.line).attr("stroke-dasharray","2,3"); txt(g, x(v), H0-m.b+14, String(v), {anchor:"middle", size:9.5, fill:P.muted}); });
    y.ticks(5).filter(v=>Math.log10(v)%1===0).forEach(v=> txt(g, m.l-6, y(v)+4, sci(v,0).replace("1 × ",""), {anchor:"end", size:9.5, fill:P.muted}));
    g.append("rect").attr("x",x(512)).attr("y",m.t).attr("width",x(8192)-x(512)).attr("height",H0-m.b-m.t).attr("fill",P.bad).attr("fill-opacity",.06);
    txt(g, x(2048), m.t+12, "beyond the 512-position table", {anchor:"middle", size:9.5, fill:P.bad});
    const line = d3.line().x((d,i)=>x(ns[i])).y(d=>y(d));
    g.append("path").attr("d", line(lin)).attr("fill","none").attr("stroke",P.A).attr("stroke-width",2);
    g.append("path").attr("d", line(quad)).attr("fill","none").attr("stroke",P.B).attr("stroke-width",2);
    g.append("path").attr("d", line(lin.map((v,i)=>v+quad[i]))).attr("fill","none").attr("stroke",P.ink).attr("stroke-dasharray","4,3");
    const cross = 6*mm.H;
    g.append("line").attr("x1",x(cross)).attr("x2",x(cross)).attr("y1",m.t).attr("y2",H0-m.b).attr("stroke",P.good).attr("stroke-width",1.5);
    txt(g, x(cross)-4, H0-m.b-8, "n = 6H = "+cross, {anchor:"end", size:9.5, fill:P.good});
    const f = fwdFlops(mm.L, mm.H, n);
    g.append("circle").attr("cx",x(n)).attr("cy",y(f.lin+f.quad)).attr("r",5).attr("fill",P.good);
    txt(g, m.l+8, m.t+14, "— weights (24·L·H²·n)", {size:10, fill:P.A});
    txt(g, m.l+8, m.t+28, "— attention scores (4·L·n²·H)", {size:10, fill:P.B});
    txt(g, m.l+8, m.t+42, "- - total", {size:10, fill:P.ink});
    txt(g, (m.l+W-m.r)/2, H0-4, "sequence length n (log)", {anchor:"middle", size:10, fill:P.muted});
    const share = f.quad/(f.lin+f.quad), N = params({L:mm.L, H:mm.H, V:30522}).nonEmb;
    d3.select("#fl-read").html(`<b>${mm.name}</b>, n = ${n}: forward ≈ <b>${sci(f.lin+f.quad)}</b> FLOPs; attention scores are <b>${(100*share).toFixed(1)}%</b> of it${n>512?" (this length is not possible with the released checkpoints)":""}. ` +
      `Per token that is ${sci((f.lin+f.quad)/n)} FLOPs, against 2 × ${(N/1e6).toFixed(1)}M non-embedding parameters = ${sci(2*N)}. Attention equals the weight matmuls at n = 6H = <b>${cross.toLocaleString("en-US")}</b>.`);
  }
  d3.select("#fl-model").on("change", draw); d3.select("#fl-n").on("input", draw);
  draw();
});

/* ───────────────────────── 17 · task heads ───────────────────────── */
BV.safe("heads", function(){
  const {P, txt, params, fmtN} = BV;
  const T = {
    single:{name:"single-sentence classification", toks:["[CLS]","the","film","was","great","[SEP]"], seg:"A", reads:"pool", out:K=>"K = "+K+" classes", np:(H,K)=>K*H+K, passes:1, cls:"BertForSequenceClassification"},
    pair:  {name:"sentence-pair classification", toks:["[CLS]","a","man","plays","guitar","[SEP]","someone","makes","music","[SEP]"], seg:"AB", reads:"pool", out:K=>"K = "+K+" classes", np:(H,K)=>K*H+K, passes:1, cls:"BertForSequenceClassification"},
    reg:   {name:"pair regression", toks:["[CLS]","a","man","plays","guitar","[SEP]","someone","makes","music","[SEP]"], seg:"AB", reads:"pool", out:()=>"1 score (MSE)", np:(H)=>H+1, passes:1, cls:"BertForSequenceClassification (num_labels = 1)"},
    tok:   {name:"token classification", toks:["[CLS]","john","lives","in","new","york","[SEP]"], seg:"A", reads:"tokens", labels:["","B-PER","O","O","B-LOC","I-LOC",""], out:K=>"K = "+K+" tags per token", np:(H,K)=>K*H+K, passes:1, cls:"BertForTokenClassification"},
    qa:    {name:"span question answering", toks:["[CLS]","who","plays","?","[SEP]","a","man","plays","guitar","[SEP]"], seg:"AB", reads:"passage", out:()=>"start + end logits", np:(H)=>2*H+2, paperNp:(H)=>2*H, passes:1, cls:"BertForQuestionAnswering"},
    mc:    {name:"multiple choice (×4 sequences)", toks:["[CLS]","she","opened","the","door","[SEP]","and","walked","in","[SEP]"], seg:"AB", reads:"pool", out:()=>"1 score per choice, softmax over 4", np:(H)=>H+1, passes:4, cls:"BertForMultipleChoice"}
  };
  const svg = d3.select("#hd-svg");
  function draw(){
    svg.selectAll("*").remove();
    const t = T[d3.select("#hd-task").property("value")], H = d3.select("#hd-model").property("value") === "base" ? 768 : 1024;
    const L = H === 768 ? 12 : 24, K = +d3.select("#hd-K").property("value"); d3.select("#hd-Kv").text(K);
    const g = svg.append("g"), n = t.toks.length, cw = Math.min(56, Math.floor(560/n) - 4), x0 = 40;
    let segB = false;
    const segOf = t.toks.map((tk,i)=>{ const s = segB ? 1 : 0; if (tk === "[SEP]" && t.seg === "AB") segB = true; return s; });
    const readIdx = t.reads === "pool" ? [0] : t.reads === "tokens" ? d3.range(1, n-1) : d3.range(segOf.indexOf(1), n-1);
    t.toks.forEach((tk,i)=>{
      const x = x0 + i*(cw+4);
      g.append("rect").attr("x",x).attr("y",236).attr("width",cw).attr("height",24).attr("rx",4).attr("fill", segOf[i] ? "rgba(255,180,84,.16)" : "rgba(91,156,255,.14)").attr("stroke",P.line);
      txt(g, x+cw/2, 252, tk, {anchor:"middle", mono:true, size:Math.min(10, cw/3.6)});
      txt(g, x+cw/2, 274, segOf[i] ? "B" : "A", {anchor:"middle", size:9, fill:P.muted});
      const on = readIdx.includes(i);
      g.append("rect").attr("x",x).attr("y",120).attr("width",cw).attr("height",22).attr("rx",4).attr("fill", on ? "rgba(74,222,128,.22)" : P.panel).attr("stroke", on ? P.good : P.line);
      txt(g, x+cw/2, 135, i===0 ? "C" : "T"+i, {anchor:"middle", mono:true, size:9.5, fill: on ? P.good : P.muted});
      if (t.labels && t.labels[i]) txt(g, x+cw/2, 112, t.labels[i], {anchor:"middle", mono:true, size:9, fill:P.B});
    });
    g.append("rect").attr("x",x0).attr("y",156).attr("width",n*(cw+4)-4).attr("height",66).attr("rx",8).attr("fill",P.panel).attr("stroke",P.A);
    txt(g, x0 + (n*(cw+4))/2, 194, `BERT encoder: ${L} layers × H ${H}  (${fmtN(params({L, H, V:30522}).total)} parameters, all fine-tuned)`, {anchor:"middle", size:11, fill:P.A});
    const hx = t.reads === "pool" ? x0 : x0 + readIdx[0]*(cw+4), hw = t.reads === "pool" ? 220 : (readIdx.length*(cw+4)-4);
    g.append("rect").attr("x",hx).attr("y",40).attr("width",Math.max(hw, 200)).attr("height",44).attr("rx",6).attr("fill","rgba(74,222,128,.10)").attr("stroke",P.good);
    txt(g, hx+8, 58, (t.reads === "pool" ? "pooler tanh(W_p·C) → " : "shared linear per token → ") + t.out(K), {size:10.5, fill:P.good});
    txt(g, hx+8, 76, t.cls, {size:9.5, mono:true, fill:P.muted});
    txt(g, 16, 22, t.name + (t.passes > 1 ? " — one forward pass per choice" : ""), {size:11, bold:true});
    const np = t.np(H, K), bb = params({L, H, V:30522}).total;
    d3.select("#hd-read").html(`New parameters for this head: <b>${np.toLocaleString("en-US")}</b>` + (t.paperNp ? ` (library, with biases; the paper's S and E vectors alone are ${t.paperNp(H).toLocaleString("en-US")})` : "") +
      ` — <b>${(100*np/bb).toFixed(4)}%</b> of the ${fmtN(bb)} backbone. Forward passes per example: <b>${t.passes}</b>. ` +
      (t.reads === "pool" ? "Reads the pooled [CLS] state." : t.reads === "tokens" ? "Reads every token; score only the first sub-token of each word." : "Reads the passage tokens (plus [CLS] for SQuAD 2.0's no-answer)."));
  }
  d3.select("#hd-task").on("change", draw); d3.select("#hd-model").on("change", draw); d3.select("#hd-K").on("input", draw);
  draw();
});

/* ───────────────────────── 18 · span decoding with a no-answer threshold ───────────────────────── */
BV.safe("span", function(){
  const {P, txt, softmax} = BV;
  const PASSAGE = ["the","telescope","was","built","in","chile","by","a","team","of","engineers","in","1998","."];
  /* illustrative start / end scores for [CLS] followed by the passage tokens */
  const EX = [
    {q:"where was the telescope built ?", s:[1.0, -1,0.2,-1,-0.5,0.4,3.6,-0.8,0.9,0.3,-1,0.6,-0.4,1.8,-2], e:[0.8, -1,0.1,-1,0.3,-0.6,3.1,-0.5,-0.9,0.7,-1,2.2,-0.8,1.9,-0.5]},
    {q:"who paid for the telescope ?",    s:[2.9, -1,0.3,-1,-0.6,0.2,0.9,-0.7,1.4,1.1,-1,0.7,-0.5,0.4,-2], e:[3.0, -1,0.2,-1,0.1,-0.4,0.8,-0.5,-0.7,0.9,-1,1.5,-0.9,0.5,-0.5]}
  ];
  const svg = d3.select("#sp-svg");
  function draw(){
    svg.selectAll("*").remove();
    const ex = EX[+d3.select("#sp-ex").property("value")], tau = +d3.select("#sp-tau").property("value"), maxLen = +d3.select("#sp-len").property("value");
    d3.select("#sp-tauv").text(tau.toFixed(2)); d3.select("#sp-lenv").text(maxLen);
    const toks = ["[CLS]"].concat(PASSAGE), ps = softmax(ex.s), pe = softmax(ex.e), g = svg.append("g");
    let best = {i:-1, j:-1, v:-Infinity};
    for (let i=1;i<toks.length;i++) for (let j=i;j<toks.length && j-i+1<=maxLen;j++){ const v = ex.s[i] + ex.e[j]; if (v > best.v) best = {i, j, v}; }
    const sNull = ex.s[0] + ex.e[0], answer = best.v > sNull + tau;
    const cw = 40, x0 = 22, y = d3.scaleLinear().domain([0, Math.max(d3.max(ps), d3.max(pe))]).range([0, 80]);
    txt(g, 16, 18, "Q: " + ex.q, {size:11, bold:true});
    toks.forEach((t,i)=>{
      const x = x0 + i*cw, inSpan = answer && i >= best.i && i <= best.j, isNull = !answer && i === 0;
      g.append("rect").attr("x",x+2).attr("y",130-y(ps[i])).attr("width",15).attr("height",y(ps[i])).attr("fill",P.A).attr("fill-opacity",.75);
      g.append("rect").attr("x",x+19).attr("y",130-y(pe[i])).attr("width",15).attr("height",y(pe[i])).attr("fill",P.B).attr("fill-opacity",.75);
      g.append("rect").attr("x",x).attr("y",140).attr("width",cw-3).attr("height",24).attr("rx",4)
        .attr("fill", inSpan || isNull ? "rgba(74,222,128,.22)" : P.panel).attr("stroke", inSpan || isNull ? P.good : P.line);
      txt(g, x+(cw-3)/2, 156, t, {anchor:"middle", mono:true, size: t.length > 6 ? 8 : 9.5});
    });
    txt(g, 22, 186, "■ P_start", {size:10, fill:P.A}); txt(g, 100, 186, "■ P_end", {size:10, fill:P.B});
    const lines = [
      `best span ŝ = S·T${best.i} + E·T${best.j} = ${ex.s[best.i].toFixed(2)} + ${ex.e[best.j].toFixed(2)} = ${best.v.toFixed(2)}   ("${toks.slice(best.i, best.j+1).join(" ")}")`,
      `no-answer s_null = S·C + E·C = ${ex.s[0].toFixed(2)} + ${ex.e[0].toFixed(2)} = ${sNull.toFixed(2)}`,
      `rule: answer iff ŝ > s_null + τ  →  ${best.v.toFixed(2)} ${answer ? ">" : "≤"} ${(sNull+tau).toFixed(2)}`
    ];
    lines.forEach((l,i)=> txt(g, 22, 214 + i*20, l, {size:10.5, mono:true, fill: i===2 ? (answer ? P.good : P.bad) : P.ink}));
    const amax = d3.maxIndex(ps.slice(1))+1, bmax = d3.maxIndex(pe.slice(1))+1;
    d3.select("#sp-read").html((answer ? `Answer: <b>"${toks.slice(best.i, best.j+1).join(" ")}"</b>` : `Prediction: <b>no answer</b>`) +
      ` · P_start × P_end of the best span = ${(ps[best.i]*pe[best.j]).toFixed(3)}; the argmax start alone is "${toks[amax]}" and the argmax end alone is "${toks[bmax]}"${bmax < amax ? " — an invalid pair (end before start), which is why decoding searches j ≥ i" : ""}.`);
  }
  d3.select("#sp-ex").on("change", draw); d3.select("#sp-tau").on("input", draw); d3.select("#sp-len").on("input", draw);
  draw();
});

/* ───────────────────────── 21 · attention heads as parsers (Clark et al., Table 1) ───────────────────────── */
BV.safe("probe", function(){
  const {P, txt} = BV;
  /* relation, best head (layer-head), accuracy, baseline accuracy, baseline offset */
  const ROWS = [["All","7-6",34.5,26.3,"1"],["prep","7-4",66.7,61.8,"-1"],["pobj","9-6",76.3,34.6,"-2"],["det","8-11",94.3,51.7,"1"],["nn","4-10",70.4,70.2,"1"],
    ["nsubj","8-2",58.5,45.5,"1"],["amod","4-10",75.6,68.3,"1"],["dobj","8-10",86.8,40.0,"-2"],["advmod","7-6",48.8,40.2,"1"],["aux","4-10",81.1,71.5,"1"],
    ["poss","7-6",80.5,47.7,"1"],["auxpass","4-10",82.5,40.5,"1"],["ccomp","8-1",48.8,12.4,"-2"],["mark","8-2",50.7,14.5,"2"],["prt","6-7",99.1,91.4,"-1"]];
  const svg = d3.select("#pr-svg");
  function draw(){
    svg.selectAll("*").remove();
    const s = d3.select("#pr-sort").property("value"), g = svg.append("g");
    const rows = ROWS.map(r=>({rel:r[0], head:r[1], acc:r[2], base:r[3], off:r[4], gain:r[2]-r[3]}));
    rows.sort((a,b)=> s === "gain" ? b.gain - a.gain : s === "acc" ? b.acc - a.acc : a.rel.localeCompare(b.rel));
    const x = d3.scaleLinear().domain([0,100]).range([150, 560]);
    rows.forEach((r,i)=>{
      const y = 10 + i*20;
      txt(g, 140, y+12, `${r.rel} (${r.head})`, {anchor:"end", size:10, mono:true, fill: r.rel==="All" ? P.muted : P.ink});
      g.append("rect").attr("x",x(0)).attr("y",y+2).attr("width",x(r.acc)-x(0)).attr("height",13).attr("fill", r.gain >= 20 ? P.good : P.A).attr("fill-opacity",.7);
      g.append("line").attr("x1",x(r.base)).attr("x2",x(r.base)).attr("y1",y).attr("y2",y+17).attr("stroke",P.B).attr("stroke-width",2);
      txt(g, x(r.acc)+5, y+13, `${r.acc.toFixed(1)} (+${r.gain.toFixed(1)})`, {size:9.5, mono:true, fill:P.muted});
    });
    [0,25,50,75,100].forEach(v=> txt(g, x(v), 314, v+"%", {anchor:"middle", size:9.5, fill:P.muted}));
    txt(g, 572, 20, "| baseline", {size:9.5, fill:P.B});
    const spec = rows.filter(r=>r.rel !== "All"), big = spec.filter(r=>r.gain >= 20);
    const top = spec.reduce((a,b)=> b.gain > a.gain ? b : a);
    d3.select("#pr-read").html(`Largest gain: <b>${top.rel}</b> (head ${top.head}) ${top.acc.toFixed(1)}% vs baseline ${top.base.toFixed(1)}% (+${top.gain.toFixed(1)}). <b>${big.length}</b> of ${spec.length} relations have a single head beating the positional baseline by 20 points or more; mean gain ${(spec.reduce((a,r)=>a+r.gain,0)/spec.length).toFixed(1)} points. The "All" row gains only ${(ROWS[0][2]-ROWS[0][3]).toFixed(1)}: no single head parses everything.`);
  }
  d3.select("#pr-sort").on("change", draw);
  draw();
});

/* ───────────────────────── 22/23 · GLUE Table 1 and the model-size Table 6 ───────────────────────── */
BV.safe("glue", function(){
  const {P, txt, params, fmtN} = BV;
  const TASKS = ["MNLI-m","MNLI-mm","QQP","QNLI","SST-2","CoLA","STS-B","MRPC","RTE"];
  /* BERT paper Table 1 rows (test server), with the paper's own Average column for checking */
  const SYS = [
    {name:"Pre-OpenAI SOTA", v:[80.6,80.1,66.1,82.3,93.2,35.0,81.0,86.0,61.7], avg:74.0, col:"#6b7280", prior:true},
    {name:"BiLSTM+ELMo+Attn", v:[76.4,76.1,64.8,79.8,90.4,36.0,73.3,84.9,56.8], avg:71.0, col:P.pink, prior:true},
    {name:"OpenAI GPT", v:[82.1,81.4,70.3,87.4,91.3,45.4,80.0,82.3,56.0], avg:75.1, col:P.teal, prior:true},
    {name:"BERT-Base", v:[84.6,83.4,71.2,90.5,93.5,52.1,85.8,88.9,66.4], avg:79.6, col:P.A},
    {name:"BERT-Large", v:[86.7,85.9,72.1,92.7,94.9,60.5,86.5,89.3,70.1], avg:82.1, col:P.B}
  ];
  /* Table 6: L, H, A, MLM ppl, MNLI-m, MRPC, SST-2 (dev, mean of 5 restarts) */
  const SIZE = [[3,768,12,5.84,77.9,79.8,88.4],[6,768,3,5.24,80.6,82.2,90.7],[6,768,12,4.68,81.9,84.8,91.3],[12,768,12,3.99,84.4,86.7,92.9],[12,1024,16,3.54,85.7,86.9,93.3],[24,1024,16,3.23,86.6,87.8,93.7]];
  const svg = d3.select("#gl-svg");
  function drawGlue(g){
    const show = d3.select("#gl-gpt").property("checked"), sys = SYS.filter(s=>show || !s.prior);
    const cols = TASKS.concat(["Avg"]), gw = 580/cols.length, bw = Math.max(3, (gw-8)/sys.length);
    const y = d3.scaleLinear().domain([30, 100]).range([240, 30]);
    [40,60,80,100].forEach(v=>{ g.append("line").attr("x1",44).attr("x2",620).attr("y1",y(v)).attr("y2",y(v)).attr("stroke",P.line).attr("stroke-dasharray","2,3"); txt(g, 40, y(v)+4, String(v), {anchor:"end", size:9.5, fill:P.muted}); });
    const avgs = sys.map(s=> d3.mean(s.v));
    cols.forEach((c,ci)=>{
      const gx = 46 + ci*gw;
      sys.forEach((s,si)=>{
        const v = ci < TASKS.length ? s.v[ci] : avgs[si];
        g.append("rect").attr("x",gx+si*bw).attr("y",y(v)).attr("width",bw-1).attr("height",240-y(v)).attr("fill",s.col).attr("fill-opacity", ci===cols.length-1 ? .95 : .7);
      });
      txt(g, gx+(sys.length*bw)/2, 256, c, {anchor:"middle", size:9, fill: c==="Avg" ? P.ink : P.muted});
    });
    sys.forEach((s,si)=> txt(g, 50 + si*118, 18, "■ "+s.name, {size:9.5, fill:s.col}));
    txt(g, 44, 276, "axis starts at 30", {size:9, fill:P.muted});
    const checks = sys.map((s,si)=>`${s.name} ${avgs[si].toFixed(2)} (paper ${s.avg.toFixed(1)}${Math.abs(avgs[si]-s.avg) < 0.1 ? (Math.abs(avgs[si]-s.avg) < 0.05 ? " ✓" : " ✓ within rounding of the per-task scores") : " ✗"})`);
    const gpt = SYS[2], L = SYS[4], B = SYS[3];
    d3.select("#gl-read").html(`Recomputed averages over the nine columns: ${checks.join("; ")}. BERT-Large − GPT: <b>+${(d3.mean(L.v)-d3.mean(gpt.v)).toFixed(1)}</b>; BERT-Base − GPT: <b>+${(d3.mean(B.v)-d3.mean(gpt.v)).toFixed(1)}</b>. Largest single-task gain of Large over GPT: ${TASKS[d3.maxIndex(L.v.map((v,i)=>v-gpt.v[i]))]} (+${d3.max(L.v.map((v,i)=>v-gpt.v[i])).toFixed(1)}).`);
  }
  function drawSize(g){
    const ti = {mnli:4, mrpc:5, sst:6}[d3.select("#gl-task").property("value")], name = {4:"MNLI-m",5:"MRPC",6:"SST-2"}[ti];
    const pts = SIZE.map(r=>({L:r[0], H:r[1], A:r[2], ppl:r[3], v:r[ti], n:params({L:r[0], H:r[1], V:30522}).total}));
    const x = d3.scaleLog().domain([35e6, 420e6]).range([70, 600]), y = d3.scaleLinear().domain([d3.min(pts,p=>p.v)-2, d3.max(pts,p=>p.v)+1.5]).range([240, 30]);
    [50e6,100e6,200e6,400e6].forEach(v=> txt(g, x(v), 256, fmtN(v), {anchor:"middle", size:9.5, fill:P.muted}));
    y.ticks(5).forEach(v=>{ g.append("line").attr("x1",70).attr("x2",600).attr("y1",y(v)).attr("y2",y(v)).attr("stroke",P.line).attr("stroke-dasharray","2,3"); txt(g, 64, y(v)+4, v.toFixed(0), {anchor:"end", size:9.5, fill:P.muted}); });
    g.append("path").attr("d", d3.line().x(p=>x(p.n)).y(p=>y(p.v))(pts.slice().sort((a,b)=>a.n-b.n || a.A-b.A))).attr("fill","none").attr("stroke",P.A).attr("stroke-opacity",.5);
    pts.forEach((p,i)=>{
      g.append("circle").attr("cx",x(p.n)).attr("cy",y(p.v)).attr("r",5).attr("fill", p.A === 3 ? P.bad : P.A);
      txt(g, x(p.n)+7, y(p.v) + (i===2 ? -8 : (i===1 ? 14 : 4)), `${p.L}/${p.H}/${p.A}`, {size:9.5, mono:true, fill:P.muted});
    });
    txt(g, 335, 276, "parameters (computed from L, H; log scale)", {anchor:"middle", size:10, fill:P.muted});
    txt(g, 70, 20, name + " dev accuracy (Table 6)", {size:10.5, bold:true});
    const first = pts[0], last = pts[pts.length-1], same = [pts[1], pts[2]];
    d3.select("#gl-read").html(`${name}: from ${first.L}/${first.H}/${first.A} (${fmtN(first.n)}) to ${last.L}/${last.H}/${last.A} (${fmtN(last.n)}) is <b>+${(last.v-first.v).toFixed(1)}</b> points for ${(last.n/first.n).toFixed(1)}× the parameters. The two 6/768 rows have identical counts (${same[0].n.toLocaleString("en-US")}) and differ only in heads: 12 heads is ${(same[1].v-same[0].v>=0?"+":"")}${(same[1].v-same[0].v).toFixed(1)} over 3 heads, with MLM perplexity ${same[1].ppl} vs ${same[0].ppl}.`);
  }
  function draw(){
    svg.selectAll("*").remove();
    const g = svg.append("g");
    if (d3.select("#gl-view").property("value") === "glue") drawGlue(g); else drawSize(g);
  }
  d3.select("#gl-view").on("change", draw); d3.select("#gl-task").on("change", draw); d3.select("#gl-gpt").on("change", draw);
  draw();
});

/* ───────────────────────── 32 · GELU exact vs tanh approximation ───────────────────────── */
BV.safe("gelu", function(){
  /* erf via Abramowitz–Stegun 7.1.26 (|error| < 1.5e-7, far below the difference being measured) */
  function erf(x){ const s = x < 0 ? -1 : 1, a = Math.abs(x), t = 1/(1+0.3275911*a);
    const y = 1 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t - 0.284496736)*t + 0.254829592)*t*Math.exp(-a*a); return s*y; }
  const exact = x => 0.5*x*(1 + erf(x/Math.SQRT2));
  const approx = x => 0.5*x*(1 + Math.tanh(Math.sqrt(2/Math.PI)*(x + 0.044715*x*x*x)));
  let best = {d:0, x:0};
  for (let i=-8000;i<=8000;i++){ const x = i/1000, d = Math.abs(exact(x)-approx(x)); if (d > best.d) best = {d, x}; }
  const el = document.getElementById("gelu-diff");
  if (el) el.textContent = `Computed on a grid of step 0.001 over [−8, 8]: the largest |exact − tanh| is ${BV.sci(best.d, 2)}, at x ≈ ${best.x.toFixed(2)}; at x = 1 the two give ${exact(1).toFixed(5)} and ${approx(1).toFixed(5)}.`;
});
