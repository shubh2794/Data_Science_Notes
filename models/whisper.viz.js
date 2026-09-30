/* whisper.viz.js — every interactive figure on models/whisper.html.
   Loaded after data.js / notes.js (palette C comes from notes.js when present).
   Each figure is its own block wrapped in try/catch so one failure cannot blank the rest.
   House rule: every number a figure displays is computed here from a specification, or from a data
   array labelled with its source (a table of the Whisper paper, a released config, a released file) —
   never typed into a label. */

const WV = (function(){
  const P = Object.assign({}, (typeof C !== "undefined" ? C : {A:"#5b9cff",B:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"}),
    { panel:"#15181f", teal:"#2dd4bf", purple:"#c084fc", pink:"#f472b6", grey:"#6b7280" });

  function safe(name, fn){ try { fn(); } catch(e){ if (typeof console !== "undefined") console.error("[whisper.viz] "+name+" failed:", e); } }
  function txt(g, x, y, s, o){
    o = o || {};
    return g.append("text").attr("x",x).attr("y",y).attr("font-size",o.size||11).attr("fill",o.fill||P.ink)
      .attr("text-anchor",o.anchor||"start").attr("font-family",o.mono?"SF Mono,Menlo,monospace":null)
      .attr("font-weight",o.bold?600:null).text(s);
  }
  function lcg(seed){ let s = seed >>> 0; return () => { s = (1664525*s + 1013904223) >>> 0; return s/4294967296; }; }
  function comma(x){ return Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
  function fmtN(x){
    const a = Math.abs(x);
    if (a >= 1e9) return (x/1e9).toFixed(2)+"B";
    if (a >= 1e6) return (x/1e6).toFixed(a >= 1e8 ? 0 : 1)+"M";
    if (a >= 1e3) return (x/1e3).toFixed(1)+"k";
    return String(Math.round(x));
  }
  function $(id){ return document.getElementById(id); }
  function val(id){ const e = $(id); return e ? e.value : null; }
  function on(id, ev, fn){ const e = $(id); if (e) e.addEventListener(ev, fn); }
  function mean(a){ return a.reduce((s,v)=>s+v,0)/a.length; }

  /* ── Audio constants (openai/whisper audio.py; identical in transformers WhisperFeatureExtractor) ── */
  const AUD = { sr:16000, nfft:400, hop:160, chunk:30, nSamples:480000, nFrames:3000, nPos:1500, textCtx:448 };

  /* ── Released configurations (config.json of each checkpoint on the Hugging Face Hub) plus the
        size label the paper / README gives, and the parameter total the Hub reports for the
        safetensors file (used only to check the formula below, never as a displayed input). ── */
  const CFG = {
    tiny:   {name:"tiny",            d:384,  Le:4,  Ld:4,  H:6,  mels:80,  V:51865, label:39e6,   hub:37760640,   fam:"multi"},
    tinyen: {name:"tiny.en",         d:384,  Le:4,  Ld:4,  H:6,  mels:80,  V:51864, label:39e6,   hub:37760256,   fam:"en"},
    base:   {name:"base",            d:512,  Le:6,  Ld:6,  H:8,  mels:80,  V:51865, label:74e6,   hub:72593920,   fam:"multi"},
    small:  {name:"small",           d:768,  Le:12, Ld:12, H:12, mels:80,  V:51865, label:244e6,  hub:241734912,  fam:"multi"},
    medium: {name:"medium",          d:1024, Le:24, Ld:24, H:16, mels:80,  V:51865, label:769e6,  hub:763857920,  fam:"multi"},
    large2: {name:"large-v2",        d:1280, Le:32, Ld:32, H:20, mels:80,  V:51865, label:1550e6, hub:1543304960, fam:"multi"},
    large3: {name:"large-v3",        d:1280, Le:32, Ld:32, H:20, mels:128, V:51866, label:1550e6, hub:1543490560, fam:"v3"},
    turbo:  {name:"large-v3-turbo",  d:1280, Le:32, Ld:4,  H:20, mels:128, V:51866, label:809e6,  hub:808878080,  fam:"v3"},
    dl3:    {name:"distil-large-v3", d:1280, Le:32, Ld:2,  H:20, mels:128, V:51866, label:756e6,  hub:756405760,  fam:"v3"},
    dmen:   {name:"distil-medium.en",d:1024, Le:24, Ld:2,  H:16, mels:80,  V:51864, label:394e6,  hub:394375168,  fam:"en"},
    dsen:   {name:"distil-small.en", d:768,  Le:12, Ld:4,  H:12, mels:80,  V:51864, label:166e6,  hub:166132224,  fam:"en"}
  };
  /* Parameter count of the Transformers WhisperForConditionalGeneration layout:
     conv1 (mels→d, k3) + conv2 (d→d, k3) with biases; 1,500 × d sinusoidal table (stored as a frozen weight);
     encoder block = q,k,v,o (k has no bias) + FFN d→4d→d + 2 LayerNorms = 12d² + 12d; final LN;
     decoder block adds cross-attention and a third LN = 16d² + 17d; final LN;
     token embedding V × d shared with the output projection; 448 × d learned positions. */
  function params(c, Ld){
    const d = c.d, L = (Ld == null ? c.Ld : Ld);
    const parts = [
      {k:"conv stem",               v: c.mels*d*3 + d + d*d*3 + d, col:P.teal},
      {k:"encoder blocks",          v: c.Le*(12*d*d + 12*d) + 2*d, col:P.A},
      {k:"encoder positions (fixed)", v: AUD.nPos*d,              col:P.grey},
      {k:"decoder blocks",          v: L*(16*d*d + 17*d) + 2*d,   col:P.B},
      {k:"token embedding (tied)",  v: c.V*d,                     col:P.purple},
      {k:"decoder positions",       v: AUD.textCtx*d,             col:P.pink}
    ];
    return {parts, total: parts.reduce((a,p)=>a+p.v,0)};
  }

  /* ── Tokenizer layout (openai/whisper tokenizer.py get_encoding: specials appended after the BPE ranks) ── */
  const LANGS = ["en","zh","de","es","ru","ko","fr","ja","pt","tr","pl","ca","nl","ar","sv","it","id","hi","fi","vi","he","uk","el","ms","cs","ro","da","hu","ta","no","th","ur","hr","bg","lt","la","mi","ml","cy","sk","te","fa","lv","bn","sr","az","sl","kn","et","mk","br","eu","is","hy","ne","mn","bs","kk","sq","sw","gl","mr","pa","si","km","sn","yo","so","af","oc","ka","be","tg","sd","gu","am","yi","lo","uz","fo","ht","ps","tk","nn","mt","sa","lb","my","bo","tl","mg","as","tt","haw","ln","ha","ba","jw","su","yue"];
  const FAM = {
    multi: {name:"multilingual tiny … large-v2", eot:50257, nLang:99, bpe:"multilingual BPE (refit, 50,257 ranks)", nospeechName:"<|nocaptions|>"},
    v3:    {name:"large-v3 / large-v3-turbo",    eot:50257, nLang:100, bpe:"multilingual BPE (refit, 50,257 ranks)", nospeechName:"<|nospeech|>"},
    en:    {name:"English-only (.en)",           eot:50256, nLang:99, bpe:"GPT-2 BPE (50,256 ranks)", nospeechName:"<|nocaptions|>"}
  };
  function layout(famKey){
    const f = FAM[famKey], sot = f.eot + 1;
    const L = {eot:f.eot, sot, langBase:sot+1, nLang:f.nLang};
    L.translate = sot + 1 + f.nLang; L.transcribe = L.translate + 1; L.startoflm = L.translate + 2;
    L.startofprev = L.translate + 3; L.nospeech = L.translate + 4; L.notimestamps = L.translate + 5;
    L.tsBegin = L.translate + 6; L.nTs = Math.round(AUD.chunk / 0.02) + 1; L.vocab = L.tsBegin + L.nTs;
    return L;
  }

  /* ── Levenshtein alignment on word lists (for WER) ── */
  function align(ref, hyp){
    const m = ref.length, n = hyp.length, D = [];
    for (let i=0;i<=m;i++){ D.push(new Array(n+1).fill(0)); D[i][0] = i; }
    for (let j=0;j<=n;j++) D[0][j] = j;
    for (let i=1;i<=m;i++) for (let j=1;j<=n;j++)
      D[i][j] = Math.min(D[i-1][j]+1, D[i][j-1]+1, D[i-1][j-1] + (ref[i-1]===hyp[j-1]?0:1));
    const ops = []; let i = m, j = n;
    while (i>0 || j>0){
      if (i>0 && j>0 && D[i][j] === D[i-1][j-1] + (ref[i-1]===hyp[j-1]?0:1)){ ops.push({op: ref[i-1]===hyp[j-1] ? "=" : "S", r:ref[i-1], h:hyp[j-1]}); i--; j--; }
      else if (i>0 && D[i][j] === D[i-1][j]+1){ ops.push({op:"D", r:ref[i-1], h:null}); i--; }
      else { ops.push({op:"I", r:null, h:hyp[j-1]}); j--; }
    }
    ops.reverse();
    const c = {S:0, D:0, I:0, N:m};
    ops.forEach(o => { if (o.op !== "=") c[o.op]++; });
    c.wer = m ? (c.S + c.D + c.I)/m : 0;
    return {ops, c};
  }

  /* ── DEFLATE size estimator (LZ77 with a 32 KiB window and hash chains, one-step lazy matching,
        then the cheaper of fixed and dynamic Huffman coding) + zlib's 2-byte header and 4-byte checksum.
        It stands in for Python's zlib.compress, which the released transcribe() uses. ── */
  function deflateSize(bytes){
    const n = bytes.length;
    const LBASE=[3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258];
    const LEXT=[0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0];
    const DBASE=[1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577];
    const DEXT=[0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13];
    function lcode(len){ for (let i=LBASE.length-1;i>=0;i--) if (len>=LBASE[i]) return i; return 0; }
    function dcode(d){ for (let i=DBASE.length-1;i>=0;i--) if (d>=DBASE[i]) return i; return 0; }
    const head = new Map(), prev = new Int32Array(Math.max(1,n)).fill(-1);
    const key = i => bytes[i]*65536 + bytes[i+1]*256 + bytes[i+2];
    function insert(i){ if (i+2<n){ const k=key(i); prev[i] = head.has(k) ? head.get(k) : -1; head.set(k,i); } }
    function best(i){
      let bl=0, bd=0; if (i+2>=n) return [0,0];
      let j = head.has(key(i)) ? head.get(key(i)) : -1, chain = 128;
      while (j>=0 && i-j<=32768 && chain-- > 0){
        let l=0; while (l<258 && i+l<n && bytes[j+l]===bytes[i+l]) l++;
        if (l>bl){ bl=l; bd=i-j; if (l===258) break; }
        j = prev[j];
      }
      return bl>=3 ? [bl,bd] : [0,0];
    }
    const syms = []; let i = 0;
    while (i<n){
      let [l,d] = best(i);
      if (l>=3 && l<32 && i+1<n){
        insert(i); const [l2,d2] = best(i+1);
        if (l2>l){ syms.push({lit:bytes[i]}); i++; l=l2; d=d2; for (let k=1;k<l;k++) insert(i+k); insert(i); syms.push({len:l,dist:d}); i+=l; continue; }
        for (let k=1;k<l;k++) insert(i+k); syms.push({len:l,dist:d}); i+=l; continue;
      }
      if (l>=3){ for (let k=0;k<l;k++) insert(i+k); syms.push({len:l,dist:d}); i+=l; }
      else { insert(i); syms.push({lit:bytes[i]}); i++; }
    }
    const lf = new Array(286).fill(0), df = new Array(30).fill(0); lf[256] = 1;
    let extra = 0;
    syms.forEach(s => { if (s.lit !== undefined) lf[s.lit]++; else { const c=lcode(s.len); lf[257+c]++; extra+=LEXT[c]; const e=dcode(s.dist); df[e]++; extra+=DEXT[e]; } });
    let fixed = 3 + extra;
    for (let s=0;s<286;s++) fixed += lf[s]*(s<144?8 : s<256?9 : s<280?7 : 8);
    for (let s=0;s<30;s++) fixed += df[s]*5;
    function huff(freq, limit){
      const L = new Array(freq.length).fill(0);
      let heap = []; freq.forEach((f,s) => { if (f>0) heap.push({f, syms:[s]}); });
      if (!heap.length) return L;
      if (heap.length === 1){ L[heap[0].syms[0]] = 1; return L; }
      const depth = {}; heap.forEach(h => depth[h.syms[0]] = 0);
      while (heap.length > 1){ heap.sort((a,b)=>a.f-b.f); const a=heap.shift(), b=heap.shift(); const s=a.syms.concat(b.syms); s.forEach(x=>depth[x]++); heap.push({f:a.f+b.f, syms:s}); }
      Object.keys(depth).forEach(s => L[s] = Math.min(limit, depth[s]));
      return L;
    }
    const ll = huff(lf,15), dl = huff(df,15);
    let dyn = 3 + 14 + extra;
    for (let s=0;s<286;s++) dyn += lf[s]*ll[s];
    for (let s=0;s<30;s++) dyn += df[s]*dl[s];
    let hlit = 286; while (hlit>257 && ll[hlit-1]===0) hlit--;
    let hdist = 30; while (hdist>1 && dl[hdist-1]===0) hdist--;
    const seq = ll.slice(0,hlit).concat(dl.slice(0,hdist)), rle = [];
    for (let k=0;k<seq.length;){
      let r=1; while (k+r<seq.length && seq[k+r]===seq[k]) r++;
      const v = seq[k];
      if (v===0 && r>=3){ let left=r; while (left>=3){ const t=Math.min(left,138); rle.push(t>=11?[18,7]:[17,3]); left-=t; } for (let q=0;q<left;q++) rle.push([0,0]); }
      else if (r>=4){ rle.push([v,0]); let left=r-1; while (left>=3){ const t=Math.min(left,6); rle.push([16,2]); left-=t; } for (let q=0;q<left;q++) rle.push([v,0]); }
      else for (let q=0;q<r;q++) rle.push([v,0]);
      k += r;
    }
    const cf = new Array(19).fill(0); rle.forEach(x => cf[x[0]]++);
    const cl = huff(cf,7), ORDER=[16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15];
    let hclen = 19; while (hclen>4 && cl[ORDER[hclen-1]]===0) hclen--;
    dyn += 3*hclen; rle.forEach(x => dyn += cl[x[0]] + x[1]);
    const stored = 3 + 8*(4 + n) + 7;
    return 2 + Math.ceil(Math.min(fixed, dyn, stored)/8) + 4;
  }
  function utf8(s){ return (typeof TextEncoder !== "undefined") ? Array.from(new TextEncoder().encode(s)) : Array.from(unescape(encodeURIComponent(s))).map(c=>c.charCodeAt(0)); }
  function compressionRatio(text){ const b = utf8(text); return b.length ? b.length / deflateSize(b) : 0; }

  /* ── The released decision rules of transcribe(): temperature fallback and the no-speech skip ── */
  const DEC = { temps:[0.0,0.2,0.4,0.6,0.8,1.0], crThr:2.4, lpThr:-1.0, nsThr:0.6, promptResetAbove:0.5 };
  function needsFallback(r){
    let nf = false;
    if (r.cr > DEC.crThr) nf = true;
    if (r.lp < DEC.lpThr) nf = true;
    if (r.ns > DEC.nsThr && r.lp < DEC.lpThr) nf = false;   // treated as silence, not as failure
    return nf;
  }
  function shouldSkip(r){ let s = r.ns > DEC.nsThr; if (r.lp > DEC.lpThr) s = false; return s; }

  return {P, safe, txt, lcg, comma, fmtN, $, val, on, mean, AUD, CFG, params, LANGS, FAM, layout, align, compressionRatio, DEC, needsFallback, shouldSkip};
})();

/* ───────────────────────── 03 · the 680,000 hours ───────────────────────── */
WV.safe("data", function(){
  const {P, txt, $, val, on, comma} = WV;
  const svg = d3.select("#data-svg");
  /* Whisper paper, Appendix E (Training Dataset Statistics): the three components in hours */
  const PARTS = [{k:"English transcription", h:438218, col:P.A}, {k:"X → English translation", h:125739, col:P.B}, {k:"non-English transcription", h:117113, col:P.teal}];
  /* Other corpora named in the paper's introduction and Section 5 (hours) */
  const SCALE = [{k:"LibriSpeech (960 h)", h:960}, {k:"SpeechStew, 7 datasets", h:5140}, {k:"GigaSpeech", h:10000}, {k:"People's Speech", h:30000}];
  /* Appendix E, hours per language (the figure's bars; languages listed in the paper's order) */
  const ASR = [["Chinese",23446],["German",13344],["Spanish",11100],["Russian",9761],["French",9752],["Portuguese",8573],["Korean",7993],["Japanese",7054],["Turkish",4333],["Polish",4278],["Italian",2585],["Swedish",2119],["Dutch",2077],["Catalan",1883],["Finnish",1066],["Indonesian",1014],["Arabic",739],["Ukrainian",697],["Vietnamese",691],["Hebrew",688],["Greek",529],["Danish",473],["Malay",382],["Hungarian",379],["Romanian",356],["Norwegian",266],["Thai",226],["Czech",192],["Tamil",136],["Urdu",104],["Croatian",91],["Slovak",90],["Bulgarian",86],["Tagalog",75],["Welsh",73],["Lithuanian",67],["Latvian",65],["Azerbaijani",47],["Estonian",41],["Slovenian",41]];
  const TR = [["Korean",19938],["Chinese",11731],["Japanese",8860],["Welsh",8263],["Russian",7687],["Spanish",6693],["Hindi",5438],["French",4481],["German",4309],["Portuguese",3620],["Arabic",2286],["Turkish",2241],["Polish",2200],["Italian",2145],["Urdu",1990],["Bengali",1988],["Norwegian Nynorsk",1889],["Dutch",1767],["Vietnamese",1719],["Malay",1691],["Thai",1635],["Latin",1614],["Tamil",1484],["Maori",1381],["Indonesian",1174],["Swedish",1055],["Telugu",987],["Greek",968],["Tagalog",894],["Malayalam",892]];
  function draw(){
    svg.selectAll("*").remove();
    const mode = val("data-mode"), g = svg.append("g");
    const total = PARTS.reduce((s,p)=>s+p.h,0);
    if (mode === "mix"){
      txt(g, 20, 20, "composition of the training set (hours)", {size:11, fill:P.muted});
      let x = 20; const W = 620;
      PARTS.forEach(p => {
        const w = W * p.h / total;
        g.append("rect").attr("x",x).attr("y",30).attr("width",w).attr("height",34).attr("fill",p.col).attr("opacity",0.75)
          .append("title").text(p.k+": "+comma(p.h)+" h");
        txt(g, x+4, 52, (100*p.h/total).toFixed(1)+"%", {size:11, bold:true, fill:"#0b0d12"});
        x += w;
      });
      PARTS.forEach((p,i) => { g.append("rect").attr("x",20+i*210).attr("y",74).attr("width",10).attr("height",10).attr("fill",p.col);
        txt(g, 34+i*210, 83, p.k+" · "+comma(p.h)+" h", {size:10.5}); });
      /* scale comparison, log axis */
      const rows = SCALE.concat([{k:"Whisper", h:total}]);
      const xs = d3.scaleLog().domain([500, 1e6]).range([180, 640]);
      txt(g, 20, 118, "labelled speech corpora, log scale (hours)", {size:11, fill:P.muted});
      rows.forEach((r,i) => {
        const y = 128 + i*24;
        txt(g, 170, y+13, r.k, {anchor:"end", size:10.5, fill: r.k==="Whisper" ? P.B : P.ink});
        g.append("rect").attr("x",180).attr("y",y).attr("width",xs(r.h)-180).attr("height",16).attr("fill", r.k==="Whisper" ? P.B : P.A).attr("opacity",0.7);
        txt(g, xs(r.h)+4, y+13, comma(r.h)+(r.k==="Whisper" ? "" : "  (Whisper = "+(total/r.h).toFixed(0)+"×)"), {size:10, mono:true});
      });
      [1e3,1e4,1e5,1e6].forEach(t => { g.append("line").attr("x1",xs(t)).attr("x2",xs(t)).attr("y1",126).attr("y2",250).attr("stroke",P.line);
        txt(g, xs(t), 262, d3.format("~s")(t), {anchor:"middle", size:9.5, fill:P.muted}); });
      $("data-read").textContent = "total " + comma(total) + " h = " + PARTS.map(p => (100*p.h/total).toFixed(1)+"% "+p.k).join(" + ") +
        " · non-English audio (translation + non-English transcription) = " + comma(PARTS[1].h+PARTS[2].h) + " h, " + (100*(PARTS[1].h+PARTS[2].h)/total).toFixed(1) + "%";
    } else {
      const task = val("data-task"), data = task === "asr" ? ASR : TR, N = +val("data-n");
      const rows = data.slice(0, N), bh = Math.min(16, 230/N);
      const xs = d3.scaleLog().domain([10, 30000]).range([130, 620]);
      txt(g, 20, 16, (task==="asr" ? "non-English transcription" : "X → English translation") + " hours per language, log scale (Appendix E)", {size:11, fill:P.muted});
      rows.forEach((r,i) => {
        const y = 26 + i*bh;
        txt(g, 124, y+bh*0.75, r[0], {anchor:"end", size:Math.min(10, bh*0.8)});
        g.append("rect").attr("x",130).attr("y",y).attr("width",Math.max(1,xs(r[1])-130)).attr("height",bh*0.8)
          .attr("fill", r[0]==="Welsh" && task==="tr" ? P.bad : (task==="asr" ? P.teal : P.B)).attr("opacity",0.75)
          .append("title").text(r[0]+": "+comma(r[1])+" h");
      });
      [10,100,1000,10000].forEach(t => { g.append("line").attr("x1",xs(t)).attr("x2",xs(t)).attr("y1",24).attr("y2",26+N*bh).attr("stroke",P.line).attr("opacity",0.6);
        txt(g, xs(t), 38+N*bh, comma(t), {anchor:"middle", size:9.5, fill:P.muted}); });
      const shown = rows.reduce((s,r)=>s+r[1],0), comp = task==="asr" ? PARTS[2].h : PARTS[1].h;
      const top1 = data[0][1], med = d3.median(data.map(r=>r[1]));
      $("data-read").textContent = "top " + N + " languages hold " + comma(shown) + " of " + comma(comp) + " h (" + (100*shown/comp).toFixed(1) + "%) · largest " + data[0][0] + " " + comma(top1) +
        " h, " + (top1/data[Math.min(N, data.length)-1][1]).toFixed(0) + "× the " + N + "th" + (task==="tr" ? " · Welsh (red) ranks " + (data.findIndex(r=>r[0]==="Welsh")+1) + " — mostly English audio mislabelled by the language detector" : " · median of the " + data.length + " listed: " + comma(med) + " h");
    }
    $("data-task").disabled = mode === "mix"; $("data-n").disabled = mode === "mix";
  }
  ["data-mode","data-task","data-n"].forEach(id => on(id, id==="data-n" ? "input" : "change", draw));
  draw();
});

/* ───────────────────────── 05 · architecture ───────────────────────── */
WV.safe("arch", function(){
  const {P, txt, $, val, on, CFG, params, comma, fmtN, AUD} = WV;
  const svg = d3.select("#arch-svg");
  function draw(){
    svg.selectAll("*").remove();
    const c = CFG[val("arch-size")], g = svg.append("g"), pr = params(c);
    const box = (x,y,w,h,col,label,sub,title) => {
      const r = g.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",h).attr("rx",5).attr("fill",col).attr("opacity",0.28).attr("stroke",col);
      if (title) r.append("title").text(title);
      txt(g, x+w/2, y+h/2-(sub?3:-4), label, {anchor:"middle", size:11, bold:true});
      if (sub) txt(g, x+w/2, y+h/2+12, sub, {anchor:"middle", size:9.5, fill:P.muted, mono:true});
    };
    box(10, 20, 120, 44, P.grey, "waveform", "30 s × "+comma(AUD.sr)+" Hz");
    box(10, 80, 120, 44, P.teal, "log-mel", c.mels+" × "+comma(AUD.nFrames));
    box(10, 140, 120, 44, P.teal, "2 × Conv1D + GELU", "→ "+comma(AUD.nPos)+" × "+c.d, "conv1 k=3 s=1, conv2 k=3 s=2");
    box(10, 200, 120, 40, P.grey, "+ sinusoidal pos.", comma(AUD.nPos)+" × "+c.d);
    [[70,64,70,80],[70,124,70,140],[70,184,70,200]].forEach(a => g.append("line").attr("x1",a[0]).attr("y1",a[1]).attr("x2",a[2]).attr("y2",a[3]).attr("stroke",P.muted));
    /* encoder stack */
    const ex = 160, ew = 170;
    for (let i=0;i<Math.min(c.Le,8);i++) g.append("rect").attr("x",ex+ i*3).attr("y",40 - i*3).attr("width",ew).attr("height",190).attr("rx",6).attr("fill",P.A).attr("opacity",0.08);
    box(ex, 40, ew, 190, P.A, "encoder × "+c.Le, null, "pre-LN blocks: self-attention (bidirectional) + MLP 4d; final LayerNorm");
    ["self-attention ("+c.H+" heads)", "MLP "+c.d+" → "+(4*c.d)+" → "+c.d, "d = "+c.d+", head dim "+(c.d/c.H), "output: "+comma(AUD.nPos)+" × "+c.d].forEach((s,i) => txt(g, ex+ew/2, 150+i*16, s, {anchor:"middle", size:10, mono:true}));
    g.append("line").attr("x1",130).attr("y1",220).attr("x2",ex).attr("y2",200).attr("stroke",P.muted);
    /* decoder stack */
    const dx = 400, dw = 180;
    for (let i=0;i<Math.min(c.Ld,8);i++) g.append("rect").attr("x",dx+ i*3).attr("y",40 - i*3).attr("width",dw).attr("height",190).attr("rx",6).attr("fill",P.B).attr("opacity",0.08);
    box(dx, 40, dw, 190, P.B, "decoder × "+c.Ld, null, "pre-LN blocks: causal self-attention + cross-attention to the encoder output + MLP");
    ["causal self-attention", "cross-attention → encoder", "MLP "+c.d+" → "+(4*c.d)+" → "+c.d, "learned pos. "+AUD.textCtx+" × "+c.d, "vocab "+comma(c.V)+" (tied)"].forEach((s,i) => txt(g, dx+dw/2, 140+i*16, s, {anchor:"middle", size:10, mono:true}));
    g.append("path").attr("d","M"+(ex+ew)+",135 L"+dx+",135").attr("stroke",P.B).attr("stroke-width",2);
    txt(g, (ex+ew+dx)/2, 128, "K, V", {anchor:"middle", size:10, fill:P.B});
    txt(g, dx+dw/2, 256, "tokens in: <|startoftranscript|> <|lang|> <|task|> …", {anchor:"middle", size:9.5, fill:P.muted, mono:true});
    txt(g, dx+dw/2, 18, "next-token distribution over "+comma(c.V), {anchor:"middle", size:10, fill:P.good});
    const enc = pr.parts[0].v + pr.parts[1].v + pr.parts[2].v, dec = pr.total - enc;
    const kvCross = c.Ld * 2 * AUD.nPos * c.d, kvSelf = c.Ld * 2 * AUD.textCtx * c.d;
    $("arch-read").textContent = c.name + ": " + c.Le + " encoder + " + c.Ld + " decoder layers, d = " + c.d + ", " + c.H + " heads of " + (c.d/c.H) +
      " · " + comma(pr.total) + " parameters (" + fmtN(enc) + " encoder side, " + fmtN(dec) + " decoder side incl. the tied embedding)" +
      " · cross-attention K/V cache " + comma(kvCross) + " values = " + (kvCross*2/1e6).toFixed(1) + " MB in fp16, self-attention cache at the full 448 tokens " + (kvSelf*2/1e6).toFixed(1) + " MB";
  }
  on("arch-size", "change", draw);
  draw();
});

/* ───────────────────────── 06 · log-mel front end ───────────────────────── */
WV.safe("mel", function(){
  const {P, txt, $, val, on, AUD, lcg} = WV;
  const svg = d3.select("#mel-svg");
  const SR = AUD.sr, N = AUD.nfft, HOP = AUD.hop, DUR = 3.0, L = Math.round(DUR*SR), NB = N/2 + 1;
  /* Slaney mel scale and area-normalised triangular filters, as librosa.filters.mel(sr=16000, n_fft=400, n_mels) — the matrices stored in whisper/assets/mel_filters.npz */
  const fsp = 200/3, minLogHz = 1000, minLogMel = minLogHz/fsp, logstep = Math.log(6.4)/27;
  const hz2mel = f => f < minLogHz ? f/fsp : minLogMel + Math.log(f/minLogHz)/logstep;
  const mel2hz = m => m < minLogMel ? m*fsp : minLogHz*Math.exp(logstep*(m - minLogMel));
  function filters(nm){
    const mmax = hz2mel(SR/2), pts = []; for (let i=0;i<nm+2;i++) pts.push(mel2hz(mmax*i/(nm+1)));
    const F = [];
    for (let m=0;m<nm;m++){
      const lo = pts[m], ce = pts[m+1], hi = pts[m+2], row = new Float64Array(NB), en = 2/(hi-lo);
      for (let k=0;k<NB;k++){ const f = k*SR/N; row[k] = en*Math.max(0, Math.min((f-lo)/(ce-lo), (hi-f)/(hi-ce))); }
      F.push({row, lo, ce, hi});
    }
    return F;
  }
  /* test signals (16 kHz) */
  const FORM = {a:[730,1090,2440], i:[270,2290,3010], u:[300,870,2240], e:[530,1840,2480], o:[570,840,2410]};
  function signal(kind){
    const x = new Float64Array(L), r = lcg(7);
    const g = () => { const u = Math.max(1e-12, r()), v = r(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };
    if (kind === "vowels"){
      const seq = ["a","i","u","e","o"];
      for (let n=0;n<L;n++){
        const t = n/SR, idx = Math.floor(t/0.55), local = t - idx*0.55;
        let s = 0;
        if (idx < seq.length && local < 0.42){
          const f0 = 120 + 40*Math.sin(2*Math.PI*0.6*t), F = FORM[seq[idx]], env = Math.sin(Math.PI*local/0.42);
          for (let h=1; h*f0 < 4000; h++){ const f = h*f0; let a = 0; F.forEach((fc,j) => a += Math.exp(-0.5*Math.pow((f-fc)/(80+40*j),2))/(1+j)); s += (a + 0.02) * Math.sin(2*Math.PI*f*t)/h; }
          s *= env;
        }
        if (t > 2.78 && t < 2.98) s += 0.25*g();
        x[n] = 0.3*s + 0.0005*g();
      }
      for (let n=L-1;n>0;n--) if (n/SR > 2.78 && n/SR < 2.98) x[n] = x[n] - 0.95*x[n-1];   // crude high-pass on the "s" burst
    } else if (kind === "chirp"){
      const f1 = 100, f2 = 7000;
      for (let n=0;n<L;n++){ const t = n/SR; x[n] = 0.3*Math.sin(2*Math.PI*(f1*t + (f2-f1)*t*t/(2*DUR))) + 0.0005*g(); }
    } else {
      for (let n=0;n<L;n++){ const t = n/SR; x[n] = (t > 1.5 ? 0.3*Math.sin(2*Math.PI*1000*t) : 0) + 1e-5*g(); }
    }
    return x;
  }
  /* STFT as torch.stft(center=True, reflect padding, periodic Hann), power, last frame dropped */
  const win = new Float64Array(N); for (let n=0;n<N;n++) win[n] = 0.5 - 0.5*Math.cos(2*Math.PI*n/N);
  const COS = [], SIN = [];
  for (let k=0;k<NB;k++){ const c = new Float64Array(N), s = new Float64Array(N); for (let n=0;n<N;n++){ c[n] = Math.cos(2*Math.PI*k*n/N); s[n] = Math.sin(2*Math.PI*k*n/N); } COS.push(c); SIN.push(s); }
  function stft(x){
    const pad = N/2, xp = new Float64Array(L + 2*pad);
    for (let n=0;n<xp.length;n++){ let j = n - pad; if (j < 0) j = -j; if (j >= L) j = 2*(L-1) - j; xp[n] = x[j]; }
    const nFr = 1 + Math.floor((xp.length - N)/HOP), out = [], fr = new Float64Array(N);
    for (let t=0;t<nFr-1;t++){
      for (let n=0;n<N;n++) fr[n] = xp[t*HOP+n]*win[n];
      const p = new Float64Array(NB);
      for (let k=0;k<NB;k++){ let re=0, im=0; const c=COS[k], s=SIN[k]; for (let n=0;n<N;n++){ re += fr[n]*c[n]; im -= fr[n]*s[n]; } p[k] = re*re + im*im; }
      out.push(p);
    }
    return out;
  }
  const cache = {};
  function draw(){
    svg.selectAll("*").remove();
    const kind = val("mel-sig"), nm = +val("mel-bins"), clamp = $("mel-clamp").checked;
    if (!cache[kind]) cache[kind] = stft(signal(kind));
    const S = cache[kind], T = S.length, F = filters(nm);
    const logm = [];
    let mx = -Infinity;
    S.forEach(p => { const col = new Float64Array(nm); for (let m=0;m<nm;m++){ let e = 0; const row = F[m].row; for (let k=0;k<NB;k++) e += row[k]*p[k]; col[m] = Math.log10(Math.max(e, 1e-10)); if (col[m] > mx) mx = col[m]; } logm.push(col); });
    let raised = 0, lo = Infinity, hi = -Infinity, rawLo = Infinity;
    logm.forEach(col => { for (let m=0;m<nm;m++){ rawLo = Math.min(rawLo, col[m]); if (clamp && col[m] < mx - 8){ col[m] = mx - 8; raised++; } col[m] = (col[m] + 4)/4; lo = Math.min(lo, col[m]); hi = Math.max(hi, col[m]); } });
    /* raster into a canvas, placed as an <image> */
    const cv = document.createElement("canvas"); cv.width = T; cv.height = nm;
    const ctx = cv.getContext && cv.getContext("2d");
    const col = d3.scaleSequential(d3.interpolateMagma).domain([lo, hi]);
    if (ctx){
      const img = ctx.createImageData(T, nm);
      for (let t=0;t<T;t++) for (let m=0;m<nm;m++){ const c = d3.rgb(col(logm[t][m])), o = 4*((nm-1-m)*T + t); img.data[o]=c.r; img.data[o+1]=c.g; img.data[o+2]=c.b; img.data[o+3]=255; }
      ctx.putImageData(img, 0, 0);
    }
    const g = svg.append("g"), X0 = 50, W = 440, Y0 = 20, H = 200;
    const href = (cv.toDataURL ? cv.toDataURL() : "");
    g.append("image").attr("x",X0).attr("y",Y0).attr("width",W).attr("height",H).attr("preserveAspectRatio","none").attr("href", href);
    g.append("rect").attr("x",X0).attr("y",Y0).attr("width",W).attr("height",H).attr("fill","none").attr("stroke",P.line);
    const xs = d3.scaleLinear().domain([0, T*HOP/SR]).range([X0, X0+W]);
    [0,0.5,1,1.5,2,2.5,3].forEach(s => txt(g, xs(s), Y0+H+14, s+" s", {anchor:"middle", size:9.5, fill:P.muted}));
    [0, Math.floor(nm/2), nm-1].forEach(m => txt(g, X0-4, Y0 + H - (m+0.5)*H/nm + 3, "mel "+m, {anchor:"end", size:9, fill:P.muted}));
    txt(g, X0, 14, T+" frames × "+nm+" mel bins (log10, "+(clamp ? "floored at max − 8" : "no floor")+", then (x + 4) / 4)", {size:10.5, fill:P.muted});
    /* filterbank panel: centre frequency of each filter */
    const fx = 520, fw = 130, fy = Y0, fh = H, fs = d3.scaleLinear().domain([0, SR/2]).range([fy+fh, fy]);
    txt(g, fx, 14, "filter centres (Hz)", {size:10.5, fill:P.muted});
    F.forEach((f,m) => g.append("line").attr("x1",fx).attr("x2",fx + fw*(m/nm)).attr("y1",fs(f.ce)).attr("y2",fs(f.ce)).attr("stroke",P.teal).attr("opacity",0.5));
    [0,1000,2000,4000,8000].forEach(h => txt(g, fx+fw+2, fs(h)+3, h, {size:9, fill:P.muted}));
    /* filters that touch at most one FFT bin (bin spacing 40 Hz) */
    const narrow = F.filter(f => f.row.filter(v => v > 0).length <= 1).length;
    const below1k = F.filter(f => f.ce < 1000).length;
    $("mel-read").textContent = "raw log10 range [" + rawLo.toFixed(2) + ", " + mx.toFixed(2) + "]" + (clamp ? " → floor " + (mx-8).toFixed(2) + " raised " + (100*raised/(T*nm)).toFixed(1) + "% of cells" : "") +
      " → model input range [" + lo.toFixed(2) + ", " + hi.toFixed(2) + "] · " + below1k + " of " + nm + " filters centred below 1 kHz · FFT bin spacing " + (SR/N) + " Hz; filters with ≤ 1 non-zero FFT bin: " + narrow;
  }
  ["mel-sig","mel-bins","mel-clamp"].forEach(id => on(id, "change", draw));
  draw();
});

/* ───────────────────────── 07 · 30-second windows ───────────────────────── */
WV.safe("frames", function(){
  const {P, txt, $, val, on, AUD, comma} = WV;
  const svg = d3.select("#fr-svg");
  const conv = (n, k, s, p) => Math.floor((n + 2*p - k)/s) + 1;
  function draw(){
    svg.selectAll("*").remove();
    const len = +val("fr-len"), mode = val("fr-mode"), g = svg.append("g");
    $("fr-lenv").textContent = len.toFixed(1) + " s";
    const n = Math.round(len*AUD.sr);
    const span = Math.max(len, AUD.chunk), xs = d3.scaleLinear().domain([0, Math.max(span, 30)]).range([20, 640]);
    txt(g, 20, 16, "timeline (s)", {size:10.5, fill:P.muted});
    g.append("rect").attr("x",xs(0)).attr("y",24).attr("width",xs(len)-xs(0)).attr("height",18).attr("fill",P.A).attr("opacity",0.7);
    txt(g, xs(0)+4, 37, "audio " + len.toFixed(1) + " s", {size:10, fill:"#0b0d12", bold:true});
    let windows = 1, padS = 0, lost = 0;
    if (mode === "short"){
      if (len < AUD.chunk){ padS = AUD.chunk - len; g.append("rect").attr("x",xs(len)).attr("y",24).attr("width",xs(AUD.chunk)-xs(len)).attr("height",18).attr("fill",P.grey).attr("opacity",0.5); }
      if (len > AUD.chunk){ lost = len - AUD.chunk; g.append("rect").attr("x",xs(AUD.chunk)).attr("y",24).attr("width",xs(len)-xs(AUD.chunk)).attr("height",18).attr("fill",P.bad).attr("opacity",0.6); txt(g, xs(AUD.chunk)+4, 37, "truncated", {size:10, fill:"#0b0d12"}); }
      g.append("rect").attr("x",xs(0)).attr("y",50).attr("width",xs(AUD.chunk)-xs(0)).attr("height",10).attr("fill","none").attr("stroke",P.B);
      txt(g, xs(0), 72, "one 30-s window", {size:10, fill:P.B});
    } else {
      windows = Math.max(1, Math.ceil(len/AUD.chunk));
      for (let w=0; w<windows; w++){ const a = w*AUD.chunk, b = Math.min(len, a+AUD.chunk);
        g.append("rect").attr("x",xs(a)).attr("y",50 + (w%2)*6).attr("width",xs(a+AUD.chunk)-xs(a)).attr("height",8).attr("fill","none").attr("stroke",P.B);
        if (b - a < AUD.chunk) padS += AUD.chunk - (b - a); }
      txt(g, xs(0), 78, windows + " window(s) if each advanced a full 30 s (the sequential decoder usually advances less; see long-form)", {size:10, fill:P.B});
    }
    /* stage bars (one window) */
    const frames = AUD.nSamples / AUD.hop, c1 = conv(frames, 3, 1, 1), c2 = conv(c1, 3, 2, 1);
    const stages = [{k:"samples", v:AUD.nSamples, s:"16 kHz × 30 s"}, {k:"mel frames", v:frames, s:"hop 160 = 10 ms"}, {k:"conv1 (k3, s1)", v:c1, s:"same length"}, {k:"conv2 (k3, s2)", v:c2, s:"20 ms per position"}];
    const ls = d3.scaleLog().domain([1000, 600000]).range([0, 400]);
    stages.forEach((st,i) => { const y = 100 + i*30;
      txt(g, 150, y+13, st.k, {anchor:"end", size:10.5});
      g.append("rect").attr("x",160).attr("y",y).attr("width",ls(st.v)).attr("height",18).attr("fill",i===0?P.grey:(i===1?P.teal:P.A)).attr("opacity",0.7);
      txt(g, 166+ls(st.v), y+13, comma(st.v) + "  · " + st.s, {size:10, mono:true}); });
    const speechFrac = Math.min(len, AUD.chunk)/AUD.chunk;
    /* receptive field of one encoder position through the stem, in samples */
    const rfFrames = 1 + (3-1)*1 + (3-1)*1, rfSamples = (rfFrames-1)*AUD.hop + AUD.nfft;
    txt(g, 20, 238, "one encoder position sees " + rfFrames + " mel frames = " + rfSamples + " samples = " + (1000*rfSamples/AUD.sr).toFixed(0) + " ms of audio before any attention", {size:10, fill:P.muted});
    $("fr-read").textContent = comma(n) + " samples · " + (mode === "short" ? "padded/trimmed to " + comma(AUD.nSamples) : "each window padded to " + comma(AUD.nSamples)) +
      " → " + comma(frames) + " frames → " + comma(c2) + " encoder positions per window · " +
      (mode === "short" ? (lost > 0 ? lost.toFixed(1) + " s of audio silently dropped" : (100*(1-speechFrac)).toFixed(1) + "% of the encoder positions are padding (" + padS.toFixed(1) + " s)") :
      windows + " window(s); encoder work " + comma(windows*c2) + " positions for " + len.toFixed(1) + " s of audio");
  }
  on("fr-len", "input", draw); on("fr-mode", "change", draw);
  draw();
});

/* ───────────────────────── 11 · the multitask token sequence ───────────────────────── */
WV.safe("tokens", function(){
  const {P, txt, $, val, on, layout, LANGS, FAM, comma} = WV;
  const svg = d3.select("#tok-svg");
  /* example utterances (source text and its English translation), for display only */
  const EX = {
    en:{src:"Ask not what your country can do for you.", tr:"Ask not what your country can do for you."},
    es:{src:"El rápido zorro marrón salta sobre el perro.", tr:"The quick brown fox jumps over the dog."},
    fr:{src:"Il fait beau aujourd'hui.", tr:"The weather is nice today."},
    de:{src:"Wo ist der Bahnhof?", tr:"Where is the train station?"},
    ja:{src:"今日は雨が降っています。", tr:"It is raining today."},
    hi:{src:"मुझे चाय पसंद है।", tr:"I like tea."},
    yue:{src:"你食咗飯未呀？", tr:"Have you eaten yet?"}
  };
  function draw(){
    const fam = val("tok-model"), lang = val("tok-lang"), task = val("tok-task"), ts = $("tok-ts").checked, prev = $("tok-prev").checked, silent = $("tok-silent").checked;
    const Lt = layout(fam), F = FAM[fam], row = $("tok-row");
    row.innerHTML = "";
    const li = LANGS.indexOf(lang), inVocab = li >= 0 && li < Lt.nLang;
    const seq = [];
    const push = (t, id, kind) => seq.push({t, id, kind});
    if (prev){ push("<|startofprev|>", Lt.startofprev, "ctl"); push("…previous window's text…", null, "prev"); }
    push("<|startoftranscript|>", Lt.sot, "sp");
    const english = fam === "en";
    if (silent && !english){ push(F.nospeechName, Lt.nospeech, "sp"); push("<|endoftext|>", Lt.eot, "sp"); }
    else {
      if (!english){ push("<|"+lang+"|>", inVocab ? Lt.langBase + li : null, inVocab ? "sp" : "bad"); push(task==="translate" ? "<|translate|>" : "<|transcribe|>", task==="translate" ? Lt.translate : Lt.transcribe, "sp"); }
      const ex = EX[lang] || EX.en, text = english ? EX.en.src : (task === "translate" ? ex.tr : ex.src);
      if (!ts){ push("<|notimestamps|>", Lt.notimestamps, "sp"); push(text, null, "text"); }
      else {
        /* two segments with times quantised to 20 ms */
        const words = Array.from(text), cut = Math.ceil(words.length/2), a = words.slice(0,cut).join(""), b = words.slice(cut).join("");
        const T = [0.0, 1.84, 1.84, 3.62], tok = t => "<|" + t.toFixed(2) + "|>", tid = t => Lt.tsBegin + Math.round(t/0.02);
        push(tok(T[0]), tid(T[0]), "ts"); push(a, null, "text"); push(tok(T[1]), tid(T[1]), "ts"); push(tok(T[2]), tid(T[2]), "ts"); push(b, null, "text"); push(tok(T[3]), tid(T[3]), "ts");
      }
      push("<|endoftext|>", Lt.eot, "sp");
    }
    seq.forEach(s => { const sp = document.createElement("span"); sp.className = s.kind === "text" ? "" : (s.kind === "ts" ? "ts" : (s.kind === "prev" ? "prev" : s.kind));
      sp.textContent = s.t + (s.id != null ? " · " + s.id : ""); row.appendChild(sp); });
    /* vocabulary layout bars */
    svg.selectAll("*").remove();
    const g = svg.append("g"), W = 620, X0 = 20, xs = d3.scaleLinear().domain([0, Lt.vocab]).range([X0, X0+W]);
    txt(g, X0, 14, "vocabulary of " + comma(Lt.vocab) + " ids", {size:10.5, fill:P.muted});
    const blocks = [{a:0, b:Lt.eot, k:F.bpe, c:P.A}, {a:Lt.eot, b:Lt.langBase, k:"EOT, SOT", c:P.grey}, {a:Lt.langBase, b:Lt.translate, k:Lt.nLang+" language tags", c:P.teal},
      {a:Lt.translate, b:Lt.tsBegin, k:"task / control", c:P.purple}, {a:Lt.tsBegin, b:Lt.vocab, k:comma(Lt.nTs)+" timestamps", c:P.B}];
    blocks.forEach(b => g.append("rect").attr("x",xs(b.a)).attr("y",22).attr("width",Math.max(1,xs(b.b)-xs(b.a))).attr("height",16).attr("fill",b.c).attr("opacity",0.7).append("title").text(b.k+": ids "+b.a+"–"+(b.b-1)));
    /* zoom on the specials */
    const zs = d3.scaleLinear().domain([Lt.eot - 2, Lt.tsBegin + 12]).range([X0, X0+W]);
    txt(g, X0, 60, "zoom: ids " + (Lt.eot-2) + " … " + (Lt.tsBegin+11), {size:10.5, fill:P.muted});
    blocks.slice(1).forEach(b => g.append("rect").attr("x",zs(Math.max(b.a, Lt.eot-2))).attr("y",68).attr("width",Math.max(1, zs(Math.min(b.b, Lt.tsBegin+12)) - zs(Math.max(b.a, Lt.eot-2)))).attr("height",16).attr("fill",b.c).attr("opacity",0.35));
    seq.filter(s => s.id != null && s.id >= Lt.eot - 2 && s.id < Lt.tsBegin + 12).forEach(s => {
      g.append("line").attr("x1",zs(s.id+0.5)).attr("x2",zs(s.id+0.5)).attr("y1",66).attr("y2",88).attr("stroke",P.ink).attr("stroke-width",2); });
    [["EOT",Lt.eot],["SOT",Lt.sot],["translate",Lt.translate],["notimestamps",Lt.notimestamps],["<|0.00|>",Lt.tsBegin]].forEach((p,i) =>
      txt(g, zs(p[1]+0.5), 102 + (i%2)*12, p[0]+" "+p[1], {anchor:"middle", size:9, mono:true, fill:P.muted}));
    const nSpecial = seq.filter(s => s.id != null).length;
    $("tok-read").textContent = F.name + ": EOT " + Lt.eot + ", SOT " + Lt.sot + ", languages " + Lt.langBase + "–" + (Lt.translate-1) + ", translate " + Lt.translate + ", transcribe " + Lt.transcribe +
      ", no-speech " + Lt.nospeech + ", notimestamps " + Lt.notimestamps + ", timestamps " + Lt.tsBegin + "–" + (Lt.vocab-1) + " → vocab " + comma(Lt.vocab) +
      " · " + nSpecial + " special tokens in this sequence" + (english ? " · English-only checkpoints take no language or task token (the openai tokenizer sets both to None)" : "") +
      (!inVocab && !english && !silent ? " · <|" + lang + "|> is not in this vocabulary (Cantonese was added in large-v3)" : "") + (english && task==="translate" ? " · English-only checkpoints cannot translate" : "");
  }
  ["tok-model","tok-lang","tok-task","tok-ts","tok-prev","tok-silent"].forEach(id => on(id, "change", draw));
  draw();
});

/* ───────────────────────── 12 · timestamps as tokens ───────────────────────── */
WV.safe("ts", function(){
  const {P, txt, $, val, on, layout} = WV;
  const svg = d3.select("#ts-svg");
  /* an example 30-s window: speech segments (start, end) in seconds */
  const SEG = [[0.52,4.13],[4.61,9.07],[9.70,15.38],[16.95,21.40],[22.02,26.47],[27.31,33.80]];
  function draw(){
    svg.selectAll("*").remove();
    const t = +val("ts-t"), fam = val("ts-model"), cross = $("ts-cross").checked, Lt = layout(fam);
    const q = v => Math.round(v/0.02), id = v => Lt.tsBegin + q(v);
    $("ts-tv").textContent = t.toFixed(3) + " s";
    const g = svg.append("g"), xs = d3.scaleLinear().domain([0, 34]).range([20, 640]);
    g.append("rect").attr("x",xs(0)).attr("y",20).attr("width",xs(30)-xs(0)).attr("height",36).attr("fill","none").attr("stroke",P.B).attr("stroke-dasharray","4,3");
    txt(g, xs(30), 16, "window end 30.00 s", {anchor:"middle", size:9.5, fill:P.B});
    const segs = cross ? SEG : SEG.slice(0, 5);
    segs.forEach((s,i) => { const inside = s[1] <= 30;
      g.append("rect").attr("x",xs(s[0])).attr("y",26).attr("width",xs(s[1])-xs(s[0])).attr("height",24).attr("fill",inside?P.A:P.bad).attr("opacity",0.6)
        .append("title").text("segment "+(i+1)+": "+s[0].toFixed(2)+"–"+s[1].toFixed(2)+" s");
      txt(g, xs(s[0])+2, 42, "seg "+(i+1), {size:9.5, fill:"#0b0d12"}); });
    for (let s=0;s<=34;s+=2) txt(g, xs(s), 70, s, {anchor:"middle", size:9, fill:P.muted});
    /* quantisation strip around t */
    const k = q(Math.min(30, Math.max(0, t))), zx = d3.scaleLinear().domain([k*0.02-0.1, k*0.02+0.1]).range([120, 540]);
    for (let j=k-5;j<=k+5;j++){ if (j<0 || j>1500) continue; const v = j*0.02;
      g.append("line").attr("x1",zx(v)).attr("x2",zx(v)).attr("y1",92).attr("y2",112).attr("stroke", j===k ? P.B : P.muted);
      if ((j-k)%2===0) txt(g, zx(v), 124, "<|"+v.toFixed(2)+"|>", {anchor:"middle", size:9, mono:true, fill: j===k ? P.B : P.muted}); }
    if (t <= 30) g.append("circle").attr("cx",zx(Math.min(t, zx.domain()[1]))).attr("cy",102).attr("r",4).attr("fill",P.good);
    txt(g, 20, 104, "20-ms grid", {size:10, fill:P.muted});
    /* the target token sequence for this window */
    const toks = [];
    segs.forEach((s,i) => {
      if (s[1] <= 30){ toks.push("<|"+(q(s[0])*0.02).toFixed(2)+"|>", "text"+(i+1), "<|"+(q(s[1])*0.02).toFixed(2)+"|>"); }
      else toks.push("<|"+(q(s[0])*0.02).toFixed(2)+"|>");
    });
    toks.push("<|endoftext|>");
    const row = $("ts-row"); row.innerHTML = "";
    toks.forEach(s => { const sp = document.createElement("span"); sp.className = /^<\|\d/.test(s) ? "ts" : (s === "<|endoftext|>" ? "sp" : ""); sp.textContent = s; row.appendChild(sp); });
    const last = segs[segs.length-1], partial = last[1] > 30;
    const nextSeek = partial ? q(last[0])*0.02 : 30;
    g.append("line").attr("x1",xs(nextSeek)).attr("x2",xs(nextSeek)).attr("y1",140).attr("y2",176).attr("stroke",P.good).attr("stroke-width",2);
    txt(g, xs(nextSeek), 190, "next window starts at " + nextSeek.toFixed(2) + " s", {anchor:"middle", size:10, fill:P.good});
    const nTs = toks.filter(s => /^<\|\d/.test(s)).length;
    const err = Math.abs(t - k*0.02) * 1000;
    $("ts-read").textContent = (t > 30 ? t.toFixed(3) + " s is outside the window (last token <|30.00|>)" : t.toFixed(3) + " s → <|" + (k*0.02).toFixed(2) + "|> = id " + id(t) + " (index " + k + " of " + Lt.nTs + "), rounding error " + err.toFixed(1) + " ms") +
      " · this window's target has " + nTs + " timestamp tokens" + (partial ? "; the last segment crosses 30 s, so only its start token is emitted and decoding resumes from it" : "; every segment ends inside the window, so the next window starts at 30.00 s");
  }
  on("ts-t", "input", draw); ["ts-model","ts-cross"].forEach(id => on(id, "change", draw));
  draw();
});

/* ───────────────────────── 14 · parameters, recomputed ───────────────────────── */
WV.safe("params", function(){
  const {P, txt, $, val, on, CFG, params, comma, fmtN} = WV;
  const svg = d3.select("#par-svg"); let last = null;
  function draw(){
    svg.selectAll("*").remove();
    const key = val("par-model"), c = CFG[key], dec = $("par-dec");
    if (key !== last){ last = key; dec.value = c.Ld; }
    const Ld = +dec.value; $("par-decv").textContent = Ld;
    const pr = params(c, Ld), g = svg.append("g");
    /* stacked bar for the selection */
    const xs = d3.scaleLinear().domain([0, 1.6e9]).range([20, 640]);
    txt(g, 20, 16, c.name + " with " + Ld + " decoder layers: " + comma(pr.total), {size:11, bold:true});
    let x = 0;
    pr.parts.forEach((p,i) => { g.append("rect").attr("x",xs(x)).attr("y",24).attr("width",Math.max(0.5, xs(x+p.v)-xs(x))).attr("height",22).attr("fill",p.col).attr("opacity",0.8).append("title").text(p.k+": "+comma(p.v)); x += p.v; });
    pr.parts.forEach((p,i) => { const lx = 20 + (i%3)*210, ly = 62 + Math.floor(i/3)*14;
      g.append("rect").attr("x",lx).attr("y",ly-9).attr("width",9).attr("height",9).attr("fill",p.col);
      txt(g, lx+13, ly, p.k+" "+fmtN(p.v)+" ("+(100*p.v/pr.total).toFixed(1)+"%)", {size:9.5}); });
    /* ladder of all released sizes at their own decoder depth */
    txt(g, 20, 104, "released checkpoints (bar = formula; tick = paper / README label)", {size:10.5, fill:P.muted});
    const keys = Object.keys(CFG), lx = d3.scaleLinear().domain([0, 1.6e9]).range([150, 600]);
    keys.forEach((k,i) => { const cc = CFG[k], t = params(cc).total, y = 112 + i*13;
      txt(g, 145, y+9, cc.name, {anchor:"end", size:9.5, fill: k===key ? P.B : P.ink});
      g.append("rect").attr("x",150).attr("y",y).attr("width",lx(t)-150).attr("height",10).attr("fill", k===key ? P.B : P.A).attr("opacity",0.6)
        .append("title").text(cc.name+": formula "+comma(t)+", Hub file "+comma(cc.hub)+", label "+fmtN(cc.label));
      g.append("line").attr("x1",lx(cc.label)).attr("x2",lx(cc.label)).attr("y1",y-1).attr("y2",y+11).attr("stroke",P.ink);
      txt(g, Math.max(lx(t), lx(cc.label)) + 6, y+9, fmtN(t), {size:9, mono:true, fill:P.muted}); });
    const full = params(c).total, check = full === c.hub;
    const encSide = pr.parts[0].v + pr.parts[1].v + pr.parts[2].v;
    $("par-read").textContent = c.name + " at its released depth: " + comma(full) + (check ? " = the Hub's safetensors total exactly" : " vs Hub " + comma(c.hub)) +
      " · label " + fmtN(c.label) + " (formula is " + (100*(full/c.label - 1)).toFixed(1) + "% from it)" +
      " · with " + Ld + " decoder layers: " + comma(pr.total) + ", encoder side " + (100*encSide/pr.total).toFixed(1) + "%, decoder blocks " + (100*pr.parts[3].v/pr.total).toFixed(1) + "%" +
      (Ld !== c.Ld ? " · Δ vs released " + (pr.total > full ? "+" : "−") + comma(Math.abs(pr.total - full)) : "");
  }
  on("par-model", "change", draw); on("par-dec", "input", draw);
  draw();
});

/* ───────────────────────── 15 · where the compute goes ───────────────────────── */
WV.safe("cost", function(){
  const {P, txt, $, val, on, CFG, AUD, comma} = WV;
  const svg = d3.select("#cost-svg");
  /* multiply–accumulates, counted layer by layer; FLOPs = 2 × MACs */
  function macs(c, nTok, beams){
    const d = c.d, n = AUD.nPos;
    const stem = n*2*(c.mels*d*3) + n*(d*d*3);            // conv1 runs on 3,000 frames, conv2 on 1,500 outputs
    const encLayer = n*(4*d*d) + 2*n*n*d + n*(8*d*d);
    const enc = stem + c.Le*encLayer;
    const crossKV = c.Ld * n * 2*d*d;                        // cross-attention K, V computed once per window
    let dec = 0;
    for (let t=1;t<=nTok;t++){
      const perLayer = 4*d*d + 2*t*d + 2*d*d + 2*n*d + 8*d*d; // self-attn proj + scores; cross q,o + scores; MLP
      dec += c.Ld*perLayer + c.V*d;                          // plus the output projection over the vocabulary
    }
    return {enc, crossKV, dec: dec*beams, total: enc + crossKV + dec*beams};
  }
  function draw(){
    svg.selectAll("*").remove();
    const nTok = +val("cost-tok"), beams = +val("cost-beam"); $("cost-tokv").textContent = nTok;
    const keys = ["tiny","base","small","medium","large3","turbo","dl3"], g = svg.append("g");
    const rows = keys.map(k => ({k, c:CFG[k], m:macs(CFG[k], nTok, beams)}));
    const mx = d3.max(rows, r => r.m.total)*2;
    const xs = d3.scaleLinear().domain([0, mx]).range([130, 560]);
    txt(g, 20, 16, "GFLOPs per 30-s window: encoder (blue) + cross-attention K/V (grey) + decoder for " + nTok + " tokens × " + beams + " beam(s) (orange)", {size:10.5, fill:P.muted});
    rows.forEach((r,i) => { const y = 28 + i*26;
      txt(g, 124, y+13, r.c.name, {anchor:"end", size:10.5});
      let x = 0;
      [[r.m.enc,P.A],[r.m.crossKV,P.grey],[r.m.dec,P.B]].forEach(p => { g.append("rect").attr("x",xs(2*x)).attr("y",y).attr("width",Math.max(0.5, xs(2*(x+p[0]))-xs(2*x))).attr("height",18).attr("fill",p[1]).attr("opacity",0.8); x += p[0]; });
      txt(g, xs(2*r.m.total)+4, y+13, (2*r.m.total/1e9).toFixed(0) + " · dec " + (100*r.m.dec/r.m.total).toFixed(0) + "%", {size:9.5, mono:true}); });
    const L3 = rows.find(r=>r.k==="large3"), T = rows.find(r=>r.k==="turbo"), D = rows.find(r=>r.k==="dl3");
    const perTokL = L3.m.dec/(nTok*beams), perTokT = T.m.dec/(nTok*beams);
    $("cost-read").textContent = "large-v3: " + (2*L3.m.enc/1e9).toFixed(0) + " GFLOPs encoder, " + (2*perTokL/1e9).toFixed(2) + " GFLOPs per decoded token; turbo " + (2*perTokT/1e9).toFixed(2) +
      " per token (" + (perTokL/perTokT).toFixed(1) + "× less) · whole window: turbo " + (L3.m.total/T.m.total).toFixed(2) + "× and distil-large-v3 " + (L3.m.total/D.m.total).toFixed(2) + "× fewer FLOPs than large-v3 · decoder steps: " + comma(nTok) + " sequential passes per beam";
  }
  on("cost-tok", "input", draw); on("cost-beam", "change", draw);
  draw();
});

/* ───────────────────────── 18 · WER and the text normaliser ───────────────────────── */
WV.safe("wer", function(){
  const {P, txt, $, val, on, align} = WV;
  const svg = d3.select("#wer-svg");
  const PRE = {
    contr:{r:"you are going to see that it is twenty five percent", h:"You're going to see that it's 25%."},
    money:{r:"the company lost sixty eight million dollars last year", h:"The company lost $68 million last year."},
    brit: {r:"the colour of the theatre was grey", h:"The color of the theater was gray."},
    fill: {r:"so um we went home [laughter] after the show", h:"So we went home after the show."},
    real: {r:"the cat sat on the mat by the door", h:"The cat sat on a hat by the door."}
  };
  /* A subset of the paper's Appendix C English normaliser (the released EnglishTextNormalizer does much more) */
  const FILL = new Set(["hmm","mm","mhm","mmm","uh","um"]);
  const CONTR = {"you're":"you are","we're":"we are","they're":"they are","i'm":"i am","it's":"it is","that's":"that is","what's":"what is","let's":"let us",
    "don't":"do not","doesn't":"does not","didn't":"did not","can't":"can not","won't":"will not","isn't":"is not","aren't":"are not","wasn't":"was not","i've":"i have","you've":"you have","we'll":"we will","i'll":"i will"};
  const BRIT = {colour:"color", theatre:"theater", grey:"gray", centre:"center", organise:"organize", recognise:"recognize", favourite:"favorite", labour:"labor"};
  const ONES = {zero:0,one:1,two:2,three:3,four:4,five:5,six:6,seven:7,eight:8,nine:9,ten:10,eleven:11,twelve:12,thirteen:13,fourteen:14,fifteen:15,sixteen:16,seventeen:17,eighteen:18,nineteen:19};
  const TENS = {twenty:20,thirty:30,forty:40,fifty:50,sixty:60,seventy:70,eighty:80,ninety:90};
  const MULT = {hundred:100, thousand:1e3, million:1e6, billion:1e9};
  function numbers(words){
    const out = []; let i = 0;
    while (i < words.length){
      const w = words[i];
      const isNumWord = x => x in ONES || x in TENS || x in MULT;
      const isDigit = x => /^\$?\d+(\.\d+)?%?$/.test(x);
      if (isNumWord(w) || (isDigit(w) && i+1 < words.length && words[i+1] in MULT)){
        let total = 0, cur = 0, cur$ = false, pct = false;
        while (i < words.length && (isNumWord(words[i]) || (isDigit(words[i]) && cur === 0))){
          const x = words[i];
          if (isDigit(x)){ if (x[0]==="$") cur$ = true; if (x.slice(-1)==="%") pct = true; cur = parseFloat(x.replace(/[$%]/g,"")); }
          else if (x in ONES) cur += ONES[x]; else if (x in TENS) cur += TENS[x];
          else if (x === "hundred") cur *= 100; else { total += cur * MULT[x]; cur = 0; }
          i++;
        }
        let num = total + cur, s = String(num);
        if (i < words.length && (words[i] === "percent")){ pct = true; i++; }
        if (i < words.length && (words[i] === "dollars" || words[i] === "dollar")){ cur$ = true; i++; }
        out.push((cur$ ? "$" : "") + s + (pct ? "%" : ""));
      } else if (isDigit(w) && i+1 < words.length && (words[i+1] === "percent" || words[i+1] === "dollars")){
        out.push((words[i+1]==="dollars" ? "$" : "") + w + (words[i+1]==="percent" ? "%" : "")); i += 2;
      } else { out.push(w); i++; }
    }
    return out;
  }
  function norm(s, level){
    if (level === "none") return s.split(/\s+/).filter(Boolean);
    let t = s.toLowerCase();
    if (level === "basic") return t.replace(/[^\p{L}\p{N}\s']/gu, " ").split(/\s+/).filter(Boolean);
    t = t.replace(/\[[^\]]*\]/g, " ").replace(/\([^)]*\)/g, " ").replace(/’/g, "'");
    t = t.replace(/\s+'/g, "'");
    t = t.replace(/(\d),(\d)/g, "$1$2").replace(/\.(?!\d)/g, " ");
    let w = t.replace(/[^\p{L}\p{N}\s'$%.]/gu, " ").split(/\s+/).filter(Boolean);
    w = w.filter(x => !FILL.has(x)).map(x => CONTR[x] || x).join(" ").split(" ");
    w = numbers(w).map(x => BRIT[x] || x).map(x => /^[$\d]/.test(x) ? x : x.replace(/'/g, ""));
    return w.filter(Boolean);
  }
  function draw(){
    const lvl = val("wer-norm"), r = norm($("wer-ref").value, lvl), h = norm($("wer-hyp").value, lvl);
    const A = align(r, h);
    const raw = align(norm($("wer-ref").value,"none"), norm($("wer-hyp").value,"none")).c.wer;
    svg.selectAll("*").remove();
    const g = svg.append("g"); let x = 20, y = 30;
    txt(g, 20, 14, "alignment after '" + lvl + "' normalisation (top: reference, bottom: hypothesis)", {size:10.5, fill:P.muted});
    const COL = {"=":P.good, S:P.B, D:P.bad, I:P.purple};
    A.ops.forEach(o => {
      const w = Math.max((o.r||"").length, (o.h||"").length, 1)*6.6 + 10;
      if (x + w > 640){ x = 20; y += 52; }
      g.append("rect").attr("x",x).attr("y",y).attr("width",w-4).attr("height",40).attr("rx",4).attr("fill",COL[o.op]).attr("opacity",0.18).attr("stroke",COL[o.op]);
      txt(g, x+(w-4)/2, y+15, o.r || "∅", {anchor:"middle", size:10.5, mono:true});
      txt(g, x+(w-4)/2, y+32, o.h || "∅", {anchor:"middle", size:10.5, mono:true, fill: o.op==="=" ? P.ink : COL[o.op]});
      x += w;
    });
    svg.attr("viewBox", "0 0 660 " + Math.max(120, y + 56));
    const c = A.c;
    $("wer-read").textContent = "reference " + c.N + " words · S " + c.S + ", D " + c.D + ", I " + c.I + " → WER = (" + c.S + " + " + c.D + " + " + c.I + ") / " + c.N + " = " + (100*c.wer).toFixed(1) + "%" +
      " · with no normalisation it would be " + (100*raw).toFixed(1) + "%";
  }
  on("wer-preset", "change", () => { const p = PRE[val("wer-preset")]; $("wer-ref").value = p.r; $("wer-hyp").value = p.h; draw(); });
  on("wer-norm", "change", draw); on("wer-ref", "input", draw); on("wer-hyp", "input", draw);
  const p0 = PRE[val("wer-preset")]; $("wer-ref").value = p0.r; $("wer-hyp").value = p0.h;
  draw();
});

/* ───────────────────────── 19 · effective robustness ───────────────────────── */
WV.safe("robust", function(){
  const {P, txt, $, val, on, mean} = WV;
  const svg = d3.select("#rob-svg");
  /* Whisper paper, Appendix D.1.1, Table 8: English WER (%), greedy decoding, after the paper's normaliser.
     Columns: LS clean, LS other, TED-LIUM3, WSJ, CallHome, Switchboard, CommonVoice5.1, Artie, CORAAL, CHiME6, AMI-IHM, AMI-SDM1, VoxPopuli.en, Fleurs.en_us */
  const COLS = ["LibriSpeech clean","LibriSpeech other","TED-LIUM3","WSJ","CallHome","Switchboard","CommonVoice5.1","Artie","CORAAL","CHiME6","AMI-IHM","AMI-SDM1","VoxPopuli.en","Fleurs.en"];
  const T8 = [
    ["Whisper tiny.en",5.6,14.6,6.0,5.0,24.1,17.8,26.3,20.0,23.9,41.3,23.7,50.3,11.7,11.6,"w"],
    ["Whisper tiny",7.6,16.9,7.0,6.7,30.0,22.8,29.6,23.9,31.0,49.6,27.6,58.1,12.7,13.7,"w"],
    ["Whisper base.en",4.2,10.2,4.9,4.6,20.9,15.2,19.0,13.4,22.6,36.4,20.5,46.7,10.0,7.6,"w"],
    ["Whisper base",5.0,12.4,5.5,5.1,23.0,16.8,21.6,16.9,26.0,40.2,22.0,49.9,10.0,10.1,"w"],
    ["Whisper small.en",3.1,7.4,4.0,3.3,18.2,15.7,13.1,9.7,20.2,27.6,17.5,38.0,8.1,6.0,"w"],
    ["Whisper small",3.4,7.6,4.3,4.0,17.5,14.5,13.5,10.3,18.1,29.3,19.0,39.6,8.3,6.6,"w"],
    ["Whisper medium.en",3.1,6.3,4.1,3.3,16.2,14.1,10.6,7.6,17.5,25.3,16.4,37.2,7.4,5.0,"w"],
    ["Whisper medium",2.9,5.9,3.8,2.9,16.4,14.0,10.3,7.2,16.6,26.4,16.6,36.0,7.4,5.4,"w"],
    ["Whisper large",2.7,5.6,4.0,3.1,15.8,13.1,9.5,6.7,19.4,25.6,16.4,36.9,7.3,4.6,"w"],
    ["Whisper large-v2",2.7,5.2,4.0,3.9,17.6,13.8,9.0,6.2,16.2,25.5,16.9,36.4,7.3,4.4,"w"],
    ["wav2vec2-base-100h",6.0,13.4,17.8,13.9,46.9,40.2,47.4,40.8,47.0,79.9,48.1,81.2,28.9,23.1,"l"],
    ["wav2vec2-base-960h",3.3,8.5,12.8,8.9,40.6,32.9,36.4,30.9,39.9,68.5,40.2,71.9,21.4,17.4,"l"],
    ["wav2vec2-large-960h-lv60-self",1.8,3.8,7.4,4.4,29.1,22.2,19.9,15.8,29.2,56.3,30.8,57.0,13.0,10.2,"l"],
    ["wav2vec2-large-960h",2.7,6.2,10.5,7.7,34.8,28.3,29.9,24.5,35.6,65.8,37.0,67.6,17.9,14.6,"l"],
    ["wav2vec2-large-robust-ft-libri-960h",2.6,5.3,9.2,6.1,23.4,19.8,20.3,16.2,29.4,58.1,31.7,61.6,15.1,11.8,"l"],
    ["asr-crdnn-rnnlm-librispeech",3.0,9.7,17.7,10.7,59.7,56.1,43.7,33.3,83.8,81.0,57.2,85.8,30.6,32.4,"l"],
    ["asr-transformer-transformerlm-librispeech",2.1,5.4,11.9,7.4,38.9,33.0,30.6,23.5,44.9,79.5,44.5,75.4,17.8,17.0,"l"],
    ["hubert-large-ls960-ft",2.0,4.1,8.4,5.4,29.6,22.8,20.8,16.0,32.0,60.0,33.7,59.1,14.4,10.9,"l"],
    ["hubert-xlarge-ls960-ft",1.9,3.5,8.3,5.4,29.3,22.2,19.8,14.8,31.5,58.5,33.3,58.9,14.2,10.5,"l"],
    ["s2t-large-librispeech-asr",3.3,8.1,14.9,9.4,54.5,40.3,38.1,30.7,50.2,79.2,53.4,79.5,21.6,18.0,"l"],
    ["s2t-medium-librispeech-asr",3.6,8.2,15.7,9.7,58.1,42.4,39.3,31.3,52.6,79.8,60.3,85.3,22.9,19.7,"l"],
    ["stt_en_conformer_ctc_large",2.1,4.2,4.4,2.1,11.3,8.2,7.4,4.0,13.5,30.5,15.9,39.9,6.7,8.2,"m"],
    ["stt_en_conformer_transducer_xlarge",1.5,2.8,4.3,1.2,12.0,7.4,4.3,1.5,19.9,36.8,20.5,48.6,6.0,6.3,"m"],
    ["unispeech-sat-base-100h-libri-ft",5.7,13.8,17.7,13.6,46.5,40.0,45.3,38.6,44.7,74.8,47.8,77.7,29.8,22.4,"l"]
  ];
  const SETS = { ood12:[2,3,4,5,6,7,8,9,10,11,12,13], table2:[1,2,3,4,5,6,7,8,9,10,11,12,13], fig2:[6,9,2] };
  function draw(){
    svg.selectAll("*").remove();
    const set = SETS[val("rob-y")], g = svg.append("g"), m = {l:50, r:20, t:14, b:36}, W = 660, H = 300;
    const pts = T8.map(r => ({n:r[0], x:r[1], y:mean(set.map(i => r[i+1])), kind:r[15]}));
    const xs = d3.scaleLinear().domain([0, 8]).range([m.l, W-m.r]), ys = d3.scaleLinear().domain([0, d3.max(pts, p=>p.y)*1.08]).range([H-m.b, m.t]);
    g.append("g").attr("transform","translate(0,"+(H-m.b)+")").call(d3.axisBottom(xs).ticks(8)).attr("color",P.muted);
    g.append("g").attr("transform","translate("+m.l+",0)").call(d3.axisLeft(ys).ticks(6)).attr("color",P.muted);
    g.append("line").attr("x1",xs(0)).attr("y1",ys(0)).attr("x2",xs(Math.min(8, ys.domain()[1]))).attr("y2",ys(Math.min(8, ys.domain()[1]))).attr("stroke",P.muted).attr("stroke-dasharray","4,3");
    txt(g, xs(7.9), ys(7.9)-4, "y = x", {anchor:"end", size:9.5, fill:P.muted});
    txt(g, (m.l+W-m.r)/2, H-4, "WER on LibriSpeech test-clean (%)", {anchor:"middle", size:10.5, fill:P.muted});
    txt(g, 12, m.t+4, "mean WER on the chosen set (%)", {size:10.5, fill:P.muted});
    const COL = {w:P.B, l:P.A, m:P.teal};
    pts.forEach(p => g.append("circle").attr("cx",xs(p.x)).attr("cy",ys(p.y)).attr("r", p.n==="Whisper large-v2" || p.n==="wav2vec2-large-960h" ? 6 : 4).attr("fill",COL[p.kind]).attr("opacity",0.85)
      .attr("stroke", p.n==="Whisper large-v2" || p.n==="wav2vec2-large-960h" ? P.ink : "none").append("title").text(p.n+": clean "+p.x+", mean "+p.y.toFixed(1)));
    [["Whisper (zero-shot)",P.B],["LibriSpeech-trained",P.A],["NVIDIA STT (mixture incl. LibriSpeech)",P.teal]].forEach((l,i) => { g.append("circle").attr("cx",m.l+20).attr("cy",m.t+18+i*14).attr("r",4).attr("fill",l[1]); txt(g, m.l+28, m.t+22+i*14, l[0], {size:9.5}); });
    /* Table 2 pair: wav2vec2-large-960h vs Whisper large-v2 */
    const a = T8.find(r => r[0]==="wav2vec2-large-960h"), b = T8.find(r => r[0]==="Whisper large-v2");
    const idx = SETS.table2, rer = idx.map(i => 100*(1 - b[i+1]/a[i+1]));
    const mA = mean(idx.map(i=>a[i+1])), mB = mean(idx.map(i=>b[i+1]));
    const ws = pts.filter(p=>p.kind==="w"), ls = pts.filter(p=>p.kind==="l");
    $("rob-read").textContent = "Table 2 pair (both " + a[1] + "% on test-clean): mean over its 13 other sets " + mA.toFixed(1) + " vs " + mB.toFixed(1) +
      " · mean of per-dataset relative error reductions " + mean(rer).toFixed(1) + "% (the paper's 55.2), reduction of the mean WER " + (100*(1-mB/mA)).toFixed(1) + "%" +
      " · on this axis the best LibriSpeech-trained model scores " + d3.min(ls, p=>p.y).toFixed(1) + ", the smallest Whisper (tiny) " + ws.find(p=>p.n==="Whisper tiny").y.toFixed(1);
  }
  on("rob-y", "change", draw);
  draw();
});

/* ───────────────────────── 20 · scaling with data ───────────────────────── */
WV.safe("scaling", function(){
  const {P, txt, $, val, on} = WV;
  const svg = d3.select("#sc-svg");
  /* hours: Whisper paper Appendix E (multilingual speech recognition); WER: Table 13, Whisper large-v2 on Fleurs */
  const LANG = [["Chinese",23446,14.7],["German",13344,4.5],["Spanish",11100,3.0],["Russian",9761,5.6],["French",9752,8.3],["Portuguese",8573,4.3],["Korean",7993,14.3],["Japanese",7054,5.3],
    ["Turkish",4333,8.4],["Polish",4278,5.4],["Italian",2585,4.0],["Swedish",2119,8.5],["Dutch",2077,6.7],["Catalan",1883,7.3],["Finnish",1066,9.7],["Indonesian",1014,7.1],["Arabic",739,16.0],
    ["Ukrainian",697,8.6],["Vietnamese",691,10.3],["Hebrew",688,27.1],["Greek",529,12.5],["Danish",473,13.8],["Malay",382,8.7],["Hungarian",379,17.0],["Romanian",356,14.4],["Thai",226,11.5],
    ["Czech",192,13.3],["Tamil",136,17.5],["Urdu",104,22.6],["Croatian",91,13.4],["Slovak",90,11.7],["Bulgarian",86,14.6],["Tagalog",75,13.8],["Welsh",73,33.0],["Lithuanian",67,28.1],
    ["Latvian",65,23.1],["Azerbaijani",47,23.4],["Estonian",41,21.9],["Slovenian",41,23.1],["Serbian",28,33.9],["Persian",24,32.9],["Icelandic",16,38.2],["Macedonian",16,16.5],["Armenian",13,44.6],
    ["Kazakh",12,37.7],["Hindi",12,21.5],["Bosnian",11,15.7],["Galician",8.9,15.4],["Swahili",5.4,39.3],["Telugu",4.3,99.0],["Afrikaans",4.1,36.7],["Belarusian",2.4,45.4],["Khmer",1.3,99.7],
    ["Bengali",1.3,104.1],["Maltese",1.1,76.6],["Punjabi",0.8,102.4],["Marathi",0.6,38.3],["Nepali",0.6,47.1],["Georgian",0.6,105.0],["Malayalam",0.5,100.7],["Uzbek",0.3,90.2],["Gujarati",0.3,102.7],
    ["Tajik",0.3,85.8],["Burmese",0.1,115.7],["Lao",0.1,101.5]];
  /* Table 6: medium-sized models trained on subsets (hours) */
  const T6 = {h:[3405,6811,13621,27243,54486,681070], en:[30.5,19.6,14.4,12.3,10.9,9.9], ml:[92.4,72.7,56.6,45.0,36.4,29.2], bleu:[0.2,1.7,7.9,13.9,19.2,24.8]};
  function fit(pts){
    const X = pts.map(p=>Math.log10(p[0])), Y = pts.map(p=>Math.log10(p[1])), n = X.length;
    const mx = X.reduce((a,b)=>a+b,0)/n, my = Y.reduce((a,b)=>a+b,0)/n;
    let sxy=0, sxx=0, syy=0; for (let i=0;i<n;i++){ sxy += (X[i]-mx)*(Y[i]-my); sxx += (X[i]-mx)**2; syy += (Y[i]-my)**2; }
    const b = sxy/sxx; return {b, a: my - b*mx, r2: sxy*sxy/(sxx*syy)};
  }
  function draw(){
    svg.selectAll("*").remove();
    const mode = val("sc-mode"), g = svg.append("g"), m = {l:52, r:20, t:16, b:36}, W = 660, H = 300;
    $("sc-metric").disabled = mode === "lang"; $("sc-ex").disabled = mode !== "lang";
    if (mode === "lang"){
      const ex = $("sc-ex").checked, pts = LANG.filter(r => !ex || r[2] < 100);
      const xs = d3.scaleLog().domain([0.05, 40000]).range([m.l, W-m.r]), ys = d3.scaleLog().domain([2, 130]).range([H-m.b, m.t]);
      g.append("g").attr("transform","translate(0,"+(H-m.b)+")").call(d3.axisBottom(xs).ticks(6, "~g")).attr("color",P.muted);
      g.append("g").attr("transform","translate("+m.l+",0)").call(d3.axisLeft(ys).ticks(6, "~g")).attr("color",P.muted);
      LANG.forEach(r => { const used = !ex || r[2] < 100;
        g.append("circle").attr("cx",xs(r[1])).attr("cy",ys(r[2])).attr("r",3.6).attr("fill", used ? P.teal : P.grey).attr("opacity", used ? 0.85 : 0.4).append("title").text(r[0]+": "+r[1]+" h, WER "+r[2]); });
      const f = fit(pts.map(r => [r[1], r[2]]));
      const line = d3.line().x(d=>xs(d)).y(d=>ys(Math.pow(10, f.a + f.b*Math.log10(d))));
      g.append("path").attr("d", line([0.1, 30000])).attr("stroke",P.B).attr("stroke-width",2).attr("fill","none");
      txt(g, (m.l+W-m.r)/2, H-4, "hours of transcribed training audio (log)", {anchor:"middle", size:10.5, fill:P.muted});
      txt(g, 12, m.t+2, "Fleurs WER, large-v2 (log)", {size:10.5, fill:P.muted});
      const k = Math.pow(0.5, 1/f.b);
      $("sc-read").textContent = pts.length + " languages · fit log WER = " + f.a.toFixed(2) + " + (" + f.b.toFixed(3) + ") · log hours, r² = " + f.r2.toFixed(2) +
        " · WER halves for every " + k.toFixed(0) + "× more data on this subset (the paper, fitting all its Fleurs languages: r² = 0.83, 16×)";
    } else {
      const met = val("sc-metric"), Y = T6[met];
      const xs = d3.scaleLog().domain([2000, 1e6]).range([m.l, W-m.r]), ys = d3.scaleLinear().domain([0, d3.max(Y)*1.1]).range([H-m.b, m.t]);
      g.append("g").attr("transform","translate(0,"+(H-m.b)+")").call(d3.axisBottom(xs).ticks(5, "~s")).attr("color",P.muted);
      g.append("g").attr("transform","translate("+m.l+",0)").call(d3.axisLeft(ys).ticks(6)).attr("color",P.muted);
      const line = d3.line().x((d,i)=>xs(T6.h[i])).y(d=>ys(d));
      g.append("path").attr("d", line(Y)).attr("stroke",P.B).attr("stroke-width",2).attr("fill","none");
      Y.forEach((v,i) => { g.append("circle").attr("cx",xs(T6.h[i])).attr("cy",ys(v)).attr("r",4.5).attr("fill",P.B);
        txt(g, xs(T6.h[i]), ys(v)-8, v, {anchor:"middle", size:10, mono:true}); });
      txt(g, (m.l+W-m.r)/2, H-4, "training hours (medium model, subsampled dataset)", {anchor:"middle", size:10.5, fill:P.muted});
      const lab = {en:"English WER, 12-set mean", ml:"multilingual WER (Fleurs)", bleu:"X→en BLEU (CoVoST2)"}[met];
      txt(g, 12, m.t+2, lab, {size:10.5, fill:P.muted});
      const gains = Y.slice(1).map((v,i) => v - Y[i]);
      $("sc-read").textContent = lab + ": " + Y.map((v,i)=>T6.h[i].toLocaleString("en-US")+" h → "+v).join(", ") +
        " · last step multiplies data by " + (T6.h[5]/T6.h[4]).toFixed(1) + "× for a change of " + gains[4].toFixed(1) + " (the step before, " + (T6.h[4]/T6.h[3]).toFixed(1) + "×, gave " + gains[3].toFixed(1) + ")";
    }
  }
  ["sc-mode","sc-metric","sc-ex"].forEach(id => on(id, "change", draw));
  draw();
});

/* ───────────────────────── 24 · long-form: sequential vs chunked ───────────────────────── */
WV.safe("longform", function(){
  const {P, txt, $, val, on, lcg} = WV;
  const svg = d3.select("#lf-svg"); let seed = 11;
  function speech(len, sd){
    const r = lcg(sd), segs = []; let t = 0.3 + r();
    while (t < len){ const d = 1.5 + 7.5*r(); const e = Math.min(len, t + d); segs.push([t, e]); t = e + 0.2 + (r() < 0.12 ? 4 + 6*r() : 1.3*r()); }
    return segs;
  }
  const q = v => Math.round(v/0.02)*0.02;
  function sequential(segs, len){
    const wins = []; let seek = 0, guard = 0;
    while (seek < len - 0.05 && guard++ < 500){
      const end = seek + 30, inside = segs.filter(s => s[1] > seek + 1e-9 && s[0] < end);
      const crossing = inside.find(s => s[1] > end && s[0] > seek + 1e-9);
      const next = crossing ? q(crossing[0]) : Math.min(len, end);
      wins.push({a:seek, b:Math.min(end, len), next, partial:!!crossing});
      if (next <= seek) { seek = end; } else seek = next;
    }
    return wins;
  }
  function chunked(len, CL, S){
    const step = CL - 2*S, out = [];
    for (let a = 0; a < len; a += step){ const b = a + CL; out.push({a, b:Math.min(b, len), sl: a === 0 ? 0 : S, sr: b >= len ? 0 : S}); if (b >= len) break; }
    return out;
  }
  function draw(){
    svg.selectAll("*").remove();
    const len = +val("lf-len"), alg = val("lf-alg"), CL = +val("lf-chunk");
    let S = +val("lf-stride"); const maxS = Math.floor((CL-1)/2*10)/10; if (S > maxS){ S = maxS; $("lf-stride").value = S; }
    $("lf-lenv").textContent = len + " s"; $("lf-chunkv").textContent = CL + " s"; $("lf-stridev").textContent = S.toFixed(1) + " s";
    $("lf-chunk").disabled = alg !== "chunk"; $("lf-stride").disabled = alg !== "chunk";
    const segs = speech(len, seed), g = svg.append("g"), xs = d3.scaleLinear().domain([0, len]).range([20, 640]);
    txt(g, 20, 14, "speech segments (synthetic, seeded)", {size:10.5, fill:P.muted});
    segs.forEach(s => g.append("rect").attr("x",xs(s[0])).attr("y",20).attr("width",Math.max(1, xs(s[1])-xs(s[0]))).attr("height",14).attr("fill",P.A).attr("opacity",0.6));
    let encSec = 0, n = 0;
    if (alg === "seq"){
      const W = sequential(segs, len); n = W.length;
      W.forEach((w,i) => { const y = 46 + (i % 8)*22;
        g.append("rect").attr("x",xs(w.a)).attr("y",y).attr("width",Math.max(1, xs(w.a+30 > len ? len : w.a+30)-xs(w.a))).attr("height",14).attr("fill","none").attr("stroke",P.B);
        g.append("rect").attr("x",xs(w.a)).attr("y",y).attr("width",Math.max(1, xs(w.next)-xs(w.a))).attr("height",14).attr("fill",P.B).attr("opacity",0.35);
        if (w.partial) g.append("line").attr("x1",xs(w.next)).attr("x2",xs(w.next)).attr("y1",y-2).attr("y2",y+16).attr("stroke",P.good).attr("stroke-width",2);
        encSec += 30; });
      const adv = W.map(w => w.next - w.a);
      txt(g, 20, 238, "filled = audio consumed by the window; green tick = seek to the start of a segment that crossed the window end", {size:9.5, fill:P.muted});
      $("lf-read").textContent = "sequential: " + n + " windows, one after another (each needs the previous window's timestamps) · mean advance " + d3.mean(adv).toFixed(1) + " s of 30 · " +
        W.filter(w=>w.partial).length + " windows ended on a partial segment · encoder processes " + encSec + " s for " + len + " s of audio (" + (encSec/len).toFixed(2) + "×)";
    } else {
      const Cs = chunked(len, CL, S); n = Cs.length;
      Cs.forEach((c,i) => { const y = 46 + (i % 8)*22;
        g.append("rect").attr("x",xs(c.a)).attr("y",y).attr("width",Math.max(1, xs(c.b)-xs(c.a))).attr("height",14).attr("fill",P.teal).attr("opacity",0.35).attr("stroke",P.teal);
        if (c.sl) g.append("rect").attr("x",xs(c.a)).attr("y",y).attr("width",xs(c.a+c.sl)-xs(c.a)).attr("height",14).attr("fill",P.grey).attr("opacity",0.7);
        if (c.sr) g.append("rect").attr("x",xs(c.b-c.sr)).attr("y",y).attr("width",xs(c.b)-xs(c.b-c.sr)).attr("height",14).attr("fill",P.grey).attr("opacity",0.7);
        encSec += 30; });
      txt(g, 20, 238, "grey = stride regions: decoded, then discarded when neighbouring chunks' tokens are merged", {size:9.5, fill:P.muted});
      $("lf-read").textContent = "chunked: chunk " + CL + " s, stride " + S.toFixed(1) + " s each side → step " + (CL-2*S).toFixed(1) + " s · " + n + " chunks, all independent (batchable) · encoder processes " + encSec + " s (each chunk padded to 30 s) for " + len + " s of audio (" + (encSec/len).toFixed(2) + "×)";
    }
    for (let s=0; s<=len; s += (len > 300 ? 60 : 30)) txt(g, xs(s), 226, s, {anchor:"middle", size:9, fill:P.muted});
  }
  ["lf-len","lf-chunk","lf-stride"].forEach(id => on(id, "input", draw)); on("lf-alg", "change", draw);
  on("lf-seed", "click", () => { seed = (seed * 7 + 3) % 1000; draw(); });
  draw();
});

/* ───────────────────────── 25 · compression ratio ───────────────────────── */
WV.safe("compress", function(){
  const {P, txt, $, val, on, compressionRatio, DEC} = WV;
  const svg = d3.select("#cr-svg");
  const PRE = {
    speech:{base:"So what we found was that the model got better at recognising accents and handling background noise as we added more data.", rep:""},
    thanks:{base:"", rep:"Thank you. "},
    loop:{base:"And then we went to the market and ", rep:"we went to the market and "},
    sub:{base:"That's all for today. ", rep:"Thanks for watching and please subscribe. "}
  };
  function build(){ const p = PRE[val("cr-preset")], k = +val("cr-rep"); $("cr-repv").textContent = k; return (p.base + (p.rep ? p.rep.repeat(k) : "")).trim(); }
  function draw(fromText){
    if (!fromText) $("cr-text").value = build();
    const text = $("cr-text").value, cr = compressionRatio(text), p = PRE[val("cr-preset")];
    svg.selectAll("*").remove();
    const g = svg.append("g"), m = {l:50, r:20, t:16, b:34}, W = 660, H = 220;
    const curve = []; for (let k=1;k<=30;k++) curve.push([k, compressionRatio((p.base + (p.rep ? p.rep.repeat(k) : "")).trim())]);
    const xs = d3.scaleLinear().domain([1, 30]).range([m.l, W-m.r]), ys = d3.scaleLinear().domain([0, Math.max(4, d3.max(curve, d=>d[1])*1.1, cr*1.1)]).range([H-m.b, m.t]);
    g.append("g").attr("transform","translate(0,"+(H-m.b)+")").call(d3.axisBottom(xs).ticks(10)).attr("color",P.muted);
    g.append("g").attr("transform","translate("+m.l+",0)").call(d3.axisLeft(ys).ticks(5)).attr("color",P.muted);
    g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",ys(DEC.crThr)).attr("y2",ys(DEC.crThr)).attr("stroke",P.bad).attr("stroke-dasharray","5,3");
    txt(g, W-m.r, ys(DEC.crThr)-4, "threshold " + DEC.crThr, {anchor:"end", size:9.5, fill:P.bad});
    g.append("path").attr("d", d3.line().x(d=>xs(d[0])).y(d=>ys(d[1]))(curve)).attr("stroke",P.A).attr("stroke-width",2).attr("fill","none");
    curve.forEach(d => g.append("circle").attr("cx",xs(d[0])).attr("cy",ys(d[1])).attr("r",2.4).attr("fill", d[1] > DEC.crThr ? P.bad : P.A));
    txt(g, (m.l+W-m.r)/2, H-4, "repetitions of the preset's repeated phrase", {anchor:"middle", size:10, fill:P.muted});
    txt(g, 12, m.t, "compression ratio", {size:10, fill:P.muted});
    const bytes = (typeof TextEncoder !== "undefined") ? new TextEncoder().encode(text).length : text.length;
    const first = curve.find(d => d[1] > DEC.crThr);
    $("cr-read").textContent = bytes + " bytes of UTF-8 → ratio " + cr.toFixed(2) + (cr > DEC.crThr ? " > " + DEC.crThr + ": this window would be re-decoded at a higher temperature" : " ≤ " + DEC.crThr + ": passes the repetition check") +
      (p.rep ? " · for this phrase the ratio first exceeds " + DEC.crThr + (first ? " at " + first[0] + " repetitions" : " beyond 30 repetitions") : "");
  }
  ["cr-preset"].forEach(id => on(id, "change", () => draw(false))); on("cr-rep", "input", () => draw(false)); on("cr-text", "input", () => draw(true));
  draw(false);
});

/* ───────────────────────── 25 · temperature fallback and the no-speech rule ───────────────────────── */
WV.safe("fallback", function(){
  const {P, txt, $, val, on, DEC, needsFallback, shouldSkip} = WV;
  const svg = d3.select("#fb-svg");
  /* Illustrative per-attempt outcomes (avg log-prob, compression ratio, no-speech probability) — inputs to the
     released decision rules, not measurements. The rules themselves are ported from transcribe(). */
  const SC = {
    normal:  [{lp:-0.35, cr:1.6, ns:0.02}],
    loop:    [{lp:-0.15, cr:4.1, ns:0.01}, {lp:-0.30, cr:3.2, ns:0.01}, {lp:-0.55, cr:2.1, ns:0.02}],
    unsure:  [{lp:-1.35, cr:1.4, ns:0.10}, {lp:-1.28, cr:1.5, ns:0.10}, {lp:-1.22, cr:1.4, ns:0.10}, {lp:-1.12, cr:1.3, ns:0.10}, {lp:-0.95, cr:1.4, ns:0.10}],
    silence: [{lp:-1.60, cr:1.9, ns:0.85}],
    music:   [{lp:-0.80, cr:1.7, ns:0.75}],
    fail:    [{lp:-1.9, cr:3.0, ns:0.05}, {lp:-1.7, cr:2.8, ns:0.05}, {lp:-1.6, cr:2.6, ns:0.05}, {lp:-1.5, cr:2.5, ns:0.05}, {lp:-1.45, cr:2.6, ns:0.05}, {lp:-1.4, cr:2.5, ns:0.05}]
  };
  let custom = null;
  function draw(){
    svg.selectAll("*").remove();
    const scen = val("fb-scen"), nsOver = +val("fb-ns"); $("fb-nsv").textContent = nsOver.toFixed(2);
    const atts = (scen === "custom" ? [custom || {lp:-0.5, cr:1.8}] : SC[scen]).map(a => Object.assign({}, a, {ns:nsOver}));
    const g = svg.append("g"), m = {l:52, r:170, t:16, b:36}, W = 660, H = 290;
    const xs = d3.scaleLinear().domain([-2.5, 0]).range([m.l, W-m.r]), ys = d3.scaleLinear().domain([0.8, 4.5]).range([H-m.b, m.t]);
    /* shade the decision regions on a grid for the current no-speech probability */
    const ns = nsOver;
    for (let i=0;i<50;i++) for (let j=0;j<37;j++){
      const lp = -2.5 + 2.5*(i+0.5)/50, cr = 0.8 + 3.7*(j+0.5)/37, r = {lp, cr, ns};
      const col = needsFallback(r) ? P.bad : (shouldSkip(r) ? P.grey : P.good);
      g.append("rect").attr("x",xs(-2.5+2.5*i/50)).attr("y",ys(0.8+3.7*(j+1)/37)).attr("width",(W-m.l-m.r)/50+0.5).attr("height",(H-m.t-m.b)/37+0.5).attr("fill",col).attr("opacity",0.16);
    }
    g.append("g").attr("transform","translate(0,"+(H-m.b)+")").call(d3.axisBottom(xs).ticks(6)).attr("color",P.muted);
    g.append("g").attr("transform","translate("+m.l+",0)").call(d3.axisLeft(ys).ticks(6)).attr("color",P.muted);
    g.append("line").attr("x1",xs(DEC.lpThr)).attr("x2",xs(DEC.lpThr)).attr("y1",m.t).attr("y2",H-m.b).attr("stroke",P.muted).attr("stroke-dasharray","4,3");
    g.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",ys(DEC.crThr)).attr("y2",ys(DEC.crThr)).attr("stroke",P.muted).attr("stroke-dasharray","4,3");
    txt(g, (m.l+W-m.r)/2, H-4, "average log-probability of the decoded tokens", {anchor:"middle", size:10, fill:P.muted});
    txt(g, 12, m.t, "compression ratio", {size:10, fill:P.muted});
    [["accept",P.good],["re-decode at next temperature",P.bad],["accept, then skip as silence",P.grey]].forEach((l,i) => { g.append("rect").attr("x",W-m.r+12).attr("y",m.t+i*16).attr("width",10).attr("height",10).attr("fill",l[1]).attr("opacity",0.6); txt(g, W-m.r+26, m.t+9+i*16, l[0], {size:9.5}); });
    /* walk the ladder */
    const log = []; let final = null, used = 0;
    for (let k=0;k<DEC.temps.length && k<atts.length;k++){
      const r = atts[k], T = DEC.temps[k], nf = needsFallback(r); used = k;
      log.push("T=" + T.toFixed(1) + ": lp " + r.lp.toFixed(2) + ", ratio " + r.cr.toFixed(2) + (nf ? " → retry" : " → keep"));
      final = {r, T};
      if (!nf) break;
    }
    atts.slice(0, used+1).forEach((r,k) => {
      if (k > 0){ const p = atts[k-1]; g.append("line").attr("x1",xs(p.lp)).attr("y1",ys(Math.min(4.5,p.cr))).attr("x2",xs(r.lp)).attr("y2",ys(Math.min(4.5,r.cr))).attr("stroke",P.ink).attr("opacity",0.6); }
      g.append("circle").attr("cx",xs(r.lp)).attr("cy",ys(Math.min(4.5, r.cr))).attr("r",5).attr("fill", k===used ? P.B : P.ink);
      txt(g, xs(r.lp)+7, ys(Math.min(4.5, r.cr))-6, "T " + DEC.temps[k].toFixed(1), {size:9.5, mono:true});
    });
    const skip = shouldSkip(final.r), reset = final.T > DEC.promptResetAbove;
    txt(g, W-m.r+12, m.t+74, "ladder: " + DEC.temps.join(", "), {size:9, fill:P.muted});
    log.forEach((l,i) => txt(g, W-m.r+12, m.t+94+i*14, l, {size:9, mono:true}));
    $("fb-read").textContent = "kept the result at T = " + final.T.toFixed(1) + " after " + (used+1) + " attempt(s)" + (needsFallback(final.r) && used === atts.length-1 ? " (every attempt failed; the last one is kept anyway)" : "") +
      " · " + (skip ? "no-speech " + final.r.ns.toFixed(2) + " > " + DEC.nsThr + " and log-prob " + final.r.lp.toFixed(2) + " ≤ " + DEC.lpThr + ": window skipped, seek += 30 s" : "window's text emitted") +
      " · " + (skip ? "nothing is added to the prompt" : "next window " + (reset ? "does NOT get this text as prompt (T > 0.5)" : "is conditioned on this text"));
  }
  on("fb-scen", "change", () => { const s = val("fb-scen"); if (SC[s]) $("fb-ns").value = SC[s][0].ns; draw(); }); on("fb-ns", "input", draw);
  svg.on("click", function(ev){
    const pt = d3.pointer(ev, this), lp = -2.5 + 2.5*(pt[0]-52)/(660-52-170), cr = 0.8 + 3.7*((290-36) - pt[1])/(290-16-36);
    if (lp < -2.5 || lp > 0 || cr < 0.8 || cr > 4.5) return;
    custom = {lp, cr, ns:+val("fb-ns")}; $("fb-scen").value = "custom"; draw();
  });
  draw();
});
