/* t5.viz.js — every interactive figure on models/t5.html.
   Loaded after data.js / notes.js (palette C comes from notes.js when present).
   Each figure is its own block wrapped in try/catch so one failure cannot blank the rest.
   House rule: every number a figure displays is computed here from a spec, a port of the released
   code, or a data array labelled with its source — never typed into a label. */

const TV = (function(){
  const P = Object.assign({}, (typeof C !== "undefined" ? C : {A:"#5b9cff",B:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"}),
    { panel:"#15181f", teal:"#2dd4bf", purple:"#c084fc", pink:"#f472b6", grey:"#6b7280", box:"#1d2230" });

  function safe(name, fn){ try { fn(); } catch(e){ if (typeof console !== "undefined") console.error("[t5.viz] "+name+" failed:", e); } }
  function txt(g, x, y, s, o){
    o = o || {};
    return g.append("text").attr("x",x).attr("y",y).attr("font-size",o.size||11).attr("fill",o.fill||P.ink)
      .attr("text-anchor",o.anchor||"start").attr("font-family",o.mono?"SF Mono,Menlo,monospace":null)
      .attr("font-weight",o.bold?600:null).text(s);
  }
  function lcg(seed){ let s = seed >>> 0; return () => { s = (1664525*s + 1013904223) >>> 0; return s/4294967296; }; }
  function comma(x){ return Math.round(x).toLocaleString("en-US"); }
  function big(x){
    const a = Math.abs(x);
    if (a >= 1e12) return (x/1e12).toFixed(2)+"T";
    if (a >= 1e9) return (x/1e9).toFixed(2)+"B";
    if (a >= 1e6) return (x/1e6).toFixed(1)+"M";
    if (a >= 1e3) return (x/1e3).toFixed(1)+"k";
    return String(Math.round(x));
  }
  function bytes(b){ if (b >= 1e9) return (b/1e9).toFixed(2)+" GB"; if (b >= 1e6) return (b/1e6).toFixed(1)+" MB"; return (b/1e3).toFixed(1)+" kB"; }
  /* Python's round() and tf.round() both round half to even; the released length helpers depend on it. */
  function rnd(x){ const f = Math.floor(x), d = x - f; if (Math.abs(d - 0.5) < 1e-9) return (f % 2 === 0) ? f : f + 1; return Math.round(x); }
  function $(id){ return document.getElementById(id); }
  function val(id){ const e = $(id); return e ? e.value : null; }
  function on(id, ev, fn){ const e = $(id); if (e) e.addEventListener(ev, fn); }

  /* ── Released configurations (config.json on the Hugging Face Hub). gated = feed_forward_proj "gated-gelu";
        untied = tie_word_embeddings false. rep = the paper's own size label, where it gives one. ── */
  const CFG = {
    "t5-small":  {name:"T5-Small",       d:512,  ff:2048,  kv:64,  h:8,   Le:6,  Ld:6,  V:32128,  gated:false, untied:false, rep:60e6,   repL:"60M"},
    "t5-base":   {name:"T5-Base",        d:768,  ff:3072,  kv:64,  h:12,  Le:12, Ld:12, V:32128,  gated:false, untied:false, rep:220e6,  repL:"220M"},
    "t5-large":  {name:"T5-Large",       d:1024, ff:4096,  kv:64,  h:16,  Le:24, Ld:24, V:32128,  gated:false, untied:false, rep:770e6,  repL:"770M"},
    "t5-3b":     {name:"T5-3B",          d:1024, ff:16384, kv:128, h:32,  Le:24, Ld:24, V:32128,  gated:false, untied:false, rep:2.8e9,  repL:"2.8B"},
    "t5-11b":    {name:"T5-11B",         d:1024, ff:65536, kv:128, h:128, Le:24, Ld:24, V:32128,  gated:false, untied:false, rep:11e9,   repL:"11B"},
    "v11-small": {name:"v1.1 Small",     d:512,  ff:1024,  kv:64,  h:6,   Le:8,  Ld:8,  V:32128,  gated:true,  untied:true},
    "v11-base":  {name:"v1.1 Base",      d:768,  ff:2048,  kv:64,  h:12,  Le:12, Ld:12, V:32128,  gated:true,  untied:true},
    "v11-large": {name:"v1.1 Large",     d:1024, ff:2816,  kv:64,  h:16,  Le:24, Ld:24, V:32128,  gated:true,  untied:true},
    "v11-xl":    {name:"v1.1 XL",        d:2048, ff:5120,  kv:64,  h:32,  Le:24, Ld:24, V:32128,  gated:true,  untied:true},
    "v11-xxl":   {name:"v1.1 XXL",       d:4096, ff:10240, kv:64,  h:64,  Le:24, Ld:24, V:32128,  gated:true,  untied:true},
    "mt5-small": {name:"mT5-Small",      d:512,  ff:1024,  kv:64,  h:6,   Le:8,  Ld:8,  V:250112, gated:true,  untied:true, rep:300e6, repL:"300M"},
    "mt5-base":  {name:"mT5-Base",       d:768,  ff:2048,  kv:64,  h:12,  Le:12, Ld:12, V:250112, gated:true,  untied:true, rep:580e6, repL:"580M"},
    "mt5-xxl":   {name:"mT5-XXL",        d:4096, ff:10240, kv:64,  h:64,  Le:24, Ld:24, V:250112, gated:true,  untied:true, rep:13e9,  repL:"13B"},
    "byt5-small":{name:"ByT5-Small",     d:1472, ff:3584,  kv:64,  h:6,   Le:12, Ld:4,  V:384,    gated:true,  untied:true},
    "byt5-xxl":  {name:"ByT5-XXL",       d:4672, ff:12352, kv:64,  h:64,  Le:36, Ld:12, V:384,    gated:true,  untied:true}
  };
  const BUCKETS = 32;
  function params(c){
    const inner = c.h * c.kv, k = c.gated ? 3 : 2;
    const attn = 4 * c.d * inner, ffp = k * c.d * c.ff;
    const p = {inner, attn, ffp,
      emb: c.V * c.d, head: c.untied ? c.V * c.d : 0,
      encAttn: c.Le * attn, encFF: c.Le * ffp,
      decSelf: c.Ld * attn, decCross: c.Ld * attn, decFF: c.Ld * ffp,
      norms: c.Le * 2 * c.d + c.Ld * 3 * c.d + 2 * c.d,
      rel: 2 * BUCKETS * c.h,
      encBlock: attn + ffp + 2 * c.d, decBlock: 2 * attn + ffp + 3 * c.d};
    p.total = p.emb + p.head + p.encAttn + p.encFF + p.decSelf + p.decCross + p.decFF + p.norms + p.rel;
    p.ckpt = p.total + BUCKETS * c.h;          /* unused cross-attention bias table stored in the checkpoints */
    p.nonEmb = p.total - p.emb - p.head;
    return p;
  }
  function cfgOptions(id, keys, sel){
    const s = $(id); if (!s) return;
    keys.forEach(k => { const o = document.createElement("option"); o.value = k; o.textContent = CFG[k].name; if (k === sel) o.selected = true; s.appendChild(o); });
  }

  /* ── Port of transformers' T5Attention._relative_position_bucket (adapted from Mesh TensorFlow). r = key − query. ── */
  function bucket(r, bidirectional, nb, maxd){
    nb = nb || 32; maxd = maxd || 128;
    let b = 0, a;
    if (bidirectional){ nb = Math.floor(nb/2); if (r > 0) b += nb; a = Math.abs(r); }
    else a = -Math.min(r, 0);
    const me = Math.floor(nb/2);
    if (a < me) return b + a;
    const large = me + Math.floor(Math.log(a/me) / Math.log(maxd/me) * (nb - me));
    return b + Math.min(large, nb - 1);
  }

  /* ── Port of t5.data.preprocessors.random_spans_helper (one sentinel per span, EOS on both sides). ── */
  function spanHelper(L, r, mu){
    function f(T){ const n = rnd(T*r), s = rnd(n/mu); return [T - n + s + 1, n + s + 1, n, s]; }
    let T = L - 1;
    while (f(T+1)[0] <= L) T++;
    let [inp, tgt, n, s] = f(T);
    if (Math.abs(r - 0.5) < 1e-9 && tgt > inp){ T--; tgt--; }
    return {raw:T, inp, tgt, noise:n, spans:s};
  }

  /* ── Port of random_spans_noise_mask (random_roll = False): alternating kept / noise spans, kept first. ── */
  function randomSeg(items, segs, rand){
    /* choose segs−1 cut points among items−1 gaps, uniformly */
    const gaps = []; for (let i = 0; i < items - 1; i++) gaps.push(i < segs - 1 ? 1 : 0);
    for (let i = gaps.length - 1; i > 0; i--){ const j = Math.floor(rand()*(i+1)); const t = gaps[i]; gaps[i] = gaps[j]; gaps[j] = t; }
    const lens = []; let cur = 1;
    for (let i = 0; i < gaps.length; i++){ if (gaps[i]){ lens.push(cur); cur = 1; } else cur++; }
    lens.push(cur);
    return lens;
  }
  function noiseMask(T, r, mu, rand){
    let noise = rnd(T*r); noise = Math.min(Math.max(noise, 1), T - 1);
    let spans = Math.max(rnd(noise/mu), 1);
    const keep = T - noise;
    spans = Math.min(spans, noise, keep);
    const nl = randomSeg(noise, spans, rand), kl = randomSeg(keep, spans, rand);
    const mask = [];
    for (let i = 0; i < spans; i++){ for (let j = 0; j < kl[i]; j++) mask.push(false); for (let j = 0; j < nl[i]; j++) mask.push(true); }
    return {mask, noise, spans, keep};
  }

  /* ── Pieces and ids produced once with the released google-t5/t5-small tokenizer.json (no </s>). ── */
  const TOK = {"translate English to German: That is good.":[["▁translate",13959],["▁English",1566],["▁to",12],["▁German",2968],[":",10],["▁That",466],["▁is",19],["▁good",207],[".",5]],"Das ist gut.":[["▁Das",644],["▁ist",229],["▁gut",1806],[".",5]],"cola sentence: The course is jumping well.":[["▁",3],["cola",12600],["▁sentence",7142],[":",10],["▁The",37],["▁course",503],["▁is",19],["▁jumping",15539],["▁well",168],[".",5]],"not acceptable":[["▁not",59],["▁acceptable",9961]],"unacceptable":[["▁unacceptable",29452]],"acceptable":[["▁acceptable",9961]],"stsb sentence1: The rhino grazed on the grass. sentence2: A rhino is grazing in a field.":[["▁",3],["s",7],["t",17],["s",7],["b",115],["▁sentence",7142],["1",536],[":",10],["▁The",37],["▁rhino",29586],["▁",3],["gra",3484],["ze",776],["d",26],["▁on",30],["▁the",8],["▁grass",5956],[".",5],["▁sentence",7142],["2",357],[":",10],["▁A",71],["▁rhino",29586],["▁is",19],["▁",3],["gra",3484],["zing",8128],["▁in",16],["▁",3],["a",9],["▁field",1057],[".",5]],"3.8":[["▁",3],["3.8",26195]],"summarize: state authorities dispatched emergency crews tuesday to survey the damage after an onslaught of severe weather in mississippi":[["▁summarize",21603],[":",10],["▁state",538],["▁authorities",5779],["▁dispatch",17648],["e",15],["d",26],["▁emergency",3583],["▁crew",4627],["s",7],["▁",3],["t",17],["u",76],["e",15],["s",7],["day",1135],["▁to",12],["▁survey",3719],["▁the",8],["▁damage",1783],["▁after",227],["▁an",46],["▁on",30],["s",7],["l",40],["aught",9313],["▁of",13],["▁severe",5274],["▁weather",1969],["▁in",16],["▁miss",3041],["is",159],["s",7],["i",23],["pp",1572],["i",23]],"six people hospitalized after a storm in attala county.":[["▁six",1296],["▁people",151],["▁hospital",2833],["ized",1601],["▁after",227],["▁",3],["a",9],["▁storm",5536],["▁in",16],["▁at",44],["tal",1947],["a",9],["▁county",5435],[".",5]],"mnli hypothesis: My feelings towards pigeons are filled with animosity. premise: I hate pigeons.":[["▁",3],["m",51],["n",29],["l",40],["i",23],["▁hypothesis",22455],[":",10],["▁My",499],["▁feelings",6382],["▁towards",1587],["▁",3],["pig",9905],["e",15],["on",106],["s",7],["▁are",33],["▁filled",3353],["▁with",28],["▁ani",1667],["mos",3972],["ity",485],[".",5],["▁",3],["premise",17398],[":",10],["▁I",27],["▁hate",5591],["▁",3],["pig",9905],["e",15],["on",106],["s",7],[".",5]],"entailment":[["▁",3],["en",35],["tail",5756],["ment",297]],"rte sentence1: Slovenia has 3,000 inhabitants.":[["▁",3],["r",52],["t",17],["e",15],["▁sentence",7142],["1",536],[":",10],["▁Slovenia",27425],["▁has",65],["▁",3],["3,000",11212],["▁inhabitants",21155],[".",5]],"not_entailment":[["▁not",59],["_",834],["en",35],["tail",5756],["ment",297]],"wsc: The stable was very roomy, with four good stalls; a large swinging window opened into the yard , which made *it* pleasant and airy.":[["▁",3],["w",210],["s",7],["c",75],[":",10],["▁The",37],["▁stable",5711],["▁was",47],["▁very",182],["▁room",562],["y",63],[",",6],["▁with",28],["▁four",662],["▁good",207],["▁",3],["stall",9176],["s",7],[";",117],["▁",3],["a",9],["▁large",508],["▁swing",7180],["ing",53],["▁window",2034],["▁opened",2946],["▁into",139],["▁the",8],["▁yard",6178],["▁",3],[",",6],["▁which",84],["▁made",263],["▁*",1429],["it",155],["*",1935],["▁pleasant",8714],["▁and",11],["▁air",799],["y",63],[".",5]],"stable":[["▁stable",5711]],"question: What does increased oxygen concentrations in the patient’s lungs displace? context: Increased O 2 concentration in the lungs helps to displace carbon monoxide from the heme group of hemoglobin.":[["▁question",822],[":",10],["▁What",363],["▁does",405],["▁increased",1936],["▁oxygen",11035],["▁concentration",6145],["s",7],["▁in",16],["▁the",8],["▁patient",1868],["’",22],["s",7],["▁",3],["lungs",17454],["▁dis",1028],["place",4687],["?",58],["▁context",2625],[":",10],["▁Increase",18840],["d",26],["▁O",411],["▁2",204],["▁concentration",6145],["▁in",16],["▁the",8],["▁",3],["lungs",17454],["▁helps",1691],["▁to",12],["▁dis",1028],["place",4687],["▁carbon",4146],["▁mon",1911],["oxid",6778],["e",15],["▁from",45],["▁the",8],["▁",3],["hem",6015],["e",15],["▁group",563],["▁of",13],["▁hemo",24731],["glob",14063],["in",77],[".",5]],"carbon monoxide":[["▁carbon",4146],["▁mon",1911],["oxid",6778],["e",15]],"Thank you for inviting me to your party last week.":[["▁Thank",1562],["▁you",25],["▁for",21],["▁inviting",14256],["▁me",140],["▁to",12],["▁your",39],["▁party",1088],["▁last",336],["▁week",471],[".",5]],"def f(x): { return x ~ 2 }":[["▁de",20],["f",89],["▁",3],["f",89],["(",599],["x",226],[")",61],[":",10],["▁",3],["{",2],["▁return",1205],["▁",3],["x",226],["▁",3],["~",2],["▁2",204],["▁",3],["}",2]],"if a < b then":[["▁",3],["if",99],["▁",3],["a",9],["▁",3],["<",2],["▁",3],["b",115],["▁then",258]],"naïve café":[["▁",3],["n",29],["a",9],["ï",2],["ve",162],["▁café",11949]],"東京":[["▁",3],["東京",2]],"Привет мир":[["▁",3],["П",2],["ри",14709],["в",6609],["ет",15042],["▁",3],["ми",21325],["р",8452]],"Grüße aus Köln":[["▁Gr",3796],["üß",11984],["e",15],["▁aus",403],["▁Köln",18552]],"line one\nline two":[["▁line",689],["▁one",80],["▁line",689],["▁two",192]],"  spaces   everywhere  ":[["▁spaces",4856],["▁everywhere",6531]],"3.25":[["▁3.",1877],["25",1828]],"</s>":[["</s>",1]],"<pad>":[["<pad>",0]],"The <extra_id_0> walks in <extra_id_1> park":[["▁The",37],["<extra_id_0>",32099],["▁walks",10681],["▁in",16],["<extra_id_1>",32098],["▁park",2447]]};

  /* ── The systematic study's validation scores (arXiv 1910.10683 Tables 1, 2, 4–13, 15; Table 10's
        d = 2048 row from appendix Table 16). Columns: GLUE, CNNDM, SQuAD, SGLUE, EnDe, EnFr, EnRo. ── */
  const BASE = [83.28, 19.24, 80.88, 71.36, 26.98, 39.82, 27.65];
  const SIGMA = [0.235, 0.065, 0.343, 0.416, 0.112, 0.090, 0.108];
  const METRICS = ["GLUE", "CNNDM", "SQuAD", "SuperGLUE", "WMT EnDe", "WMT EnFr", "WMT EnRo"];
  const STUDY = [
    {t:"Table 1 · baseline vs no pretraining", rows:[
      ["Baseline average", BASE], ["No pre-training", [66.22,17.60,50.31,53.04,25.86,39.77,24.04]]]},
    {t:"Table 2 · architectures × objectives", rows:[
      ["Enc–dec, denoising", BASE], ["Enc–dec shared, denoising", [82.81,18.78,80.63,70.73,26.72,39.03,27.46]],
      ["Enc–dec 6+6, denoising", [80.88,18.97,77.59,68.42,26.38,38.40,26.95]], ["LM, denoising", [74.70,17.93,61.14,55.02,25.09,35.28,25.86]],
      ["Prefix LM, denoising", [81.82,18.61,78.94,68.11,26.43,37.98,27.39]], ["Enc–dec, LM", [79.56,18.59,76.02,64.29,26.27,39.17,26.86]],
      ["Enc–dec shared, LM", [79.60,18.13,76.35,63.50,26.62,39.17,27.05]], ["Enc–dec 6+6, LM", [78.67,18.26,75.32,64.06,26.13,38.42,26.89]],
      ["LM, LM", [73.78,17.54,53.81,56.51,25.23,34.31,25.38]], ["Prefix LM, LM", [79.68,17.84,76.87,64.86,26.28,37.51,26.76]]]},
    {t:"Table 4 · objective families", rows:[
      ["Prefix LM", [80.69,18.94,77.99,65.27,26.86,39.73,27.49]], ["BERT-style", [82.96,19.17,80.65,69.85,26.78,40.03,27.41]],
      ["Deshuffling", [73.17,18.59,67.61,58.47,26.11,39.30,25.62]]]},
    {t:"Table 5 · BERT-style variants", rows:[
      ["BERT-style", [82.96,19.17,80.65,69.85,26.78,40.03,27.41]], ["MASS-style", [82.32,19.16,80.10,69.28,26.79,39.89,27.55]],
      ["Replace corrupted spans", BASE], ["Drop corrupted tokens", [84.44,19.31,80.52,68.67,27.07,39.76,27.82]]]},
    {t:"Table 6 · corruption rate", rows:[
      ["10%", [82.82,19.00,80.38,69.55,26.87,39.28,27.44]], ["15%", BASE],
      ["25%", [83.00,19.54,80.96,70.48,27.04,39.83,27.47]], ["50%", [81.27,19.32,79.80,70.33,27.01,39.90,27.49]]]},
    {t:"Table 7 · mean span length", rows:[
      ["i.i.d.", BASE], ["2", [83.54,19.39,82.09,72.20,26.76,39.99,27.63]], ["3", [83.49,19.62,81.84,72.53,26.86,39.65,27.62]],
      ["5", [83.40,19.24,82.05,72.23,26.88,39.40,27.53]], ["10", [82.85,19.33,81.84,70.44,26.79,39.49,27.69]]]},
    {t:"Table 8 · pretraining data sets", rows:[
      ["C4", BASE], ["C4, unfiltered", [81.46,19.14,78.78,68.04,26.55,39.34,27.21]], ["RealNews-like", [83.83,19.23,80.39,72.38,26.75,39.90,27.48]],
      ["WebText-like", [84.03,19.31,81.42,71.40,26.80,39.74,27.59]], ["Wikipedia", [81.85,19.31,81.29,68.01,26.94,39.69,27.67]],
      ["Wikipedia + TBC", [83.65,19.28,82.08,73.24,26.77,39.63,27.57]]]},
    {t:"Table 9 · repeated data", rows:[
      ["Full data set", BASE], ["2^29 tokens (64×)", [82.87,19.19,80.97,72.03,26.83,39.74,27.63]], ["2^27 (256×)", [82.62,19.20,79.78,69.97,27.02,39.71,27.33]],
      ["2^25 (1,024×)", [79.55,18.57,76.27,64.76,26.38,39.56,26.80]], ["2^23 (4,096×)", [76.34,18.33,70.92,59.29,26.37,38.84,25.81]]]},
    {t:"Table 10 · partial fine-tuning", rows:[
      ["All parameters", BASE], ["Adapters d=32", [80.52,15.08,79.32,60.40,13.84,17.88,15.54]], ["Adapters d=128", [81.51,16.62,79.47,63.03,19.83,27.50,22.63]],
      ["Adapters d=512", [81.54,17.78,79.18,64.30,23.45,33.98,25.81]], ["Adapters d=2048 (T16)", [82.62,18.30,79.40,68.61,25.64,36.92,26.93]],
      ["Gradual unfreezing", [82.50,18.95,79.17,70.79,26.71,39.02,26.93]]]},
    {t:"Table 11 · multi-task mixing", rows:[
      ["Pretrain / fine-tune", BASE], ["Equal", [76.13,19.02,76.51,63.37,23.89,34.31,26.78]],
      ["Ex.-prop. K=2^16", [80.45,19.04,77.25,69.95,24.35,34.99,27.10]], ["K=2^17", [81.56,19.12,77.00,67.91,24.36,35.00,27.25]],
      ["K=2^18", [81.67,19.07,78.17,67.94,24.57,35.19,27.39]], ["K=2^19", [81.42,19.24,79.78,67.30,25.21,36.30,27.76]],
      ["K=2^20", [80.80,19.24,80.36,67.38,25.66,36.93,27.68]], ["K=2^21", [79.83,18.79,79.50,65.10,25.82,37.22,27.13]],
      ["Temperature T=2", [81.90,19.28,79.42,69.92,25.42,36.72,27.20]], ["T=4", [80.56,19.22,77.99,69.54,25.04,35.82,27.45]],
      ["T=8", [77.21,19.10,77.14,66.07,24.55,35.35,27.17]]]},
    {t:"Table 12 · multi-task + fine-tuning", rows:[
      ["Unsup. pretrain + fine-tune", BASE], ["Multi-task training", [81.42,19.24,79.78,67.30,25.21,36.30,27.76]],
      ["Multi-task pretrain + fine-tune", [83.11,19.12,80.26,71.03,27.08,39.80,28.07]], ["Leave-one-out", [81.98,19.05,79.97,71.68,26.93,39.79,27.87]],
      ["Supervised multi-task pretrain", [79.93,18.96,77.38,65.36,26.81,40.13,28.04]]]},
    {t:"Table 13 · scaling with 4× compute", rows:[
      ["Baseline", BASE], ["1× size, 4× steps", [85.33,19.33,82.45,74.72,27.08,40.66,27.93]], ["1× size, 4× batch", [84.60,19.42,82.52,74.64,27.07,40.60,27.84]],
      ["2× size, 2× steps", [86.18,19.66,84.18,77.18,27.52,41.03,28.19]], ["4× size, 1× steps", [85.91,19.73,83.86,78.04,27.47,40.71,28.10]],
      ["4× ensembled", [84.77,20.10,83.09,71.74,28.05,40.53,28.57]], ["4× ensembled, fine-tune only", [84.05,19.57,82.36,71.55,27.55,40.22,28.09]]]},
    {t:"Table 15 · baseline vs T5-Base", rows:[
      ["Baseline (34B tokens)", BASE], ["Baseline-1T", [84.80,19.62,83.01,73.90,27.46,40.30,28.34]], ["T5-Base", [85.97,20.90,85.44,75.64,28.37,41.37,28.98]]]}
  ];

  return {P, safe, txt, lcg, comma, big, bytes, rnd, $, val, on, CFG, params, cfgOptions, bucket, spanHelper, noiseMask, TOK,
    BASE, SIGMA, METRICS, STUDY};
})();

/* ── shared: draw a row of token chips that wraps inside [x0, x1]; returns the y below the last row ── */
function tvChips(g, pieces, x0, y0, x1, o){
  o = o || {};
  const P = TV.P, cw = o.cw || 6.6, hgt = o.h || 18, gap = 3;
  let x = x0, y = y0;
  pieces.forEach(p => {
    const label = p.t, w = Math.max(14, label.length*cw + 8);
    if (x + w > x1 && x > x0){ x = x0; y += hgt + gap; }
    const fill = p.kind === "unk" ? "rgba(248,113,113,.28)" : p.kind === "sp" ? "rgba(192,132,252,.28)" : p.kind === "noise" ? "rgba(255,180,84,.30)" : p.kind === "eos" ? "rgba(154,163,178,.22)" : "rgba(91,156,255,.18)";
    const r = g.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",hgt).attr("rx",3).attr("fill",fill);
    if (p.title) r.append("title").text(p.title);
    TV.txt(g, x+4, y+hgt-5, label, {size:o.size||10.5, mono:true, fill: p.kind === "unk" ? "#fecaca" : p.kind === "sp" ? "#e9d5ff" : p.kind === "noise" ? "#ffd9a3" : "#dbe6ff"});
    x += w + gap;
  });
  return y + hgt;
}

/* ───────────────────────── 02 · tasks as strings ───────────────────────── */
TV.safe("task", function(){
  const {P, txt, $, val, on, TOK} = TV;
  const svg = d3.select("#task-svg");
  /* Strings from the paper's Figure 1, Section 2.4 and Appendix D (summary and SQuAD inputs shortened). */
  const TASKS = {
    ende:  {inp:"translate English to German: That is good.", tgt:"Das ist gut."},
    sum:   {inp:"summarize: state authorities dispatched emergency crews tuesday to survey the damage after an onslaught of severe weather in mississippi", tgt:"six people hospitalized after a storm in attala county."},
    cola:  {inp:"cola sentence: The course is jumping well.", tgt:"unacceptable", alt:"not acceptable"},
    stsb:  {inp:"stsb sentence1: The rhino grazed on the grass. sentence2: A rhino is grazing in a field.", tgt:"3.8"},
    mnli:  {inp:"mnli hypothesis: My feelings towards pigeons are filled with animosity. premise: I hate pigeons.", tgt:"entailment"},
    rte:   {inp:"rte sentence1: Slovenia has 3,000 inhabitants.", tgt:"not_entailment"},
    wsc:   {inp:"wsc: The stable was very roomy, with four good stalls; a large swinging window opened into the yard , which made *it* pleasant and airy.", tgt:"stable"},
    squad: {inp:"question: What does increased oxygen concentrations in the patient’s lungs displace? context: Increased O 2 concentration in the lungs helps to displace carbon monoxide from the heme group of hemoglobin.", tgt:"carbon monoxide"}
  };
  function chips(s){ const t = TOK[s] || []; return t.map(([p, id]) => ({t:p, kind: id === 2 ? "unk" : (id >= 32000 ? "sp" : "tok"), title:"id "+id})).concat([{t:"</s>", kind:"eos", title:"id 1"}]); }
  function draw(){
    svg.selectAll("*").remove();
    const k = val("task-pick"), T = TASKS[k], g = svg.append("g");
    const score = +val("task-sts") / 100, q = Math.round(5 * score) / 5, qs = q.toFixed(1);
    $("task-stsv").textContent = score.toFixed(2);
    const tgtStr = k === "stsb" ? qs : T.tgt;
    txt(g, 10, 14, "encoder input", {size:10.5, fill:P.A});
    const ci = chips(T.inp);
    let y = tvChips(g, ci, 10, 20, 650, {size:10, cw:6.2, h:17});
    y += 18;
    txt(g, 10, y, "decoder target", {size:10.5, fill:P.good});
    const tp = k === "stsb" ? [{t:"▁", kind:"tok"}, {t:qs, kind:"tok"}, {t:"</s>", kind:"eos"}] : chips(tgtStr);
    y = tvChips(g, tp, 10, y+6, 650, {size:10, cw:6.2, h:17});
    /* STS-B rounding strip */
    const sy = 240, x = d3.scaleLinear().domain([1, 5]).range([40, 620]);
    txt(g, 10, sy-14, "STS-B: gold score → nearest 0.2 → string (" + (Math.round((5-1)/0.2) + 1) + " classes)", {size:10.5, fill:P.muted});
    g.append("line").attr("x1",x(1)).attr("x2",x(5)).attr("y1",sy).attr("y2",sy).attr("stroke",P.line);
    for (let i = 0; i <= 20; i++){ const v = 1 + i*0.2; g.append("line").attr("x1",x(v)).attr("x2",x(v)).attr("y1",sy-4).attr("y2",sy+4).attr("stroke",P.muted);
      if (i % 5 === 0) txt(g, x(v), sy+18, v.toFixed(1), {size:9.5, anchor:"middle", fill:P.muted}); }
    g.append("circle").attr("cx",x(score)).attr("cy",sy).attr("r",5).attr("fill",P.B);
    g.append("path").attr("d","M"+x(score)+","+(sy-8)+" Q"+((x(score)+x(q))/2)+","+(sy-26)+" "+x(q)+","+(sy-8)).attr("fill","none").attr("stroke",P.good);
    g.append("rect").attr("x",x(q)-3).attr("y",sy-6).attr("width",6).attr("height",12).attr("fill",P.good);
    txt(g, x(q), sy+34, "\"" + qs + "\"", {size:11, anchor:"middle", mono:true, fill:P.good});
    const nIn = ci.length, nOut = tp.length;
    $("task-read").textContent = "input " + nIn + " tokens (incl. </s>) → target " + nOut + " tokens · target/input = " + (nOut/nIn).toFixed(2) +
      (T.alt ? " · the paper's Figure 1 writes \"" + T.alt + "\" (" + (TOK[T.alt] || []).length + " pieces); the dataset label is \"" + T.tgt + "\" (" + (TOK[T.tgt] || []).length + " piece)" : "") +
      " · STS-B " + score.toFixed(2) + " → \"" + qs + "\" (off by " + Math.abs(score - q).toFixed(2) + ")";
  }
  ["task-pick"].forEach(id => on(id, "change", draw));
  on("task-sts", "input", draw);
  draw();
});

/* ───────────────────────── 03 · blocks and shapes ───────────────────────── */
TV.safe("arch", function(){
  const {P, txt, $, val, on, CFG, params, cfgOptions, comma} = TV;
  const svg = d3.select("#arch-svg");
  cfgOptions("arch-cfg", Object.keys(CFG), "t5-base");
  function box(g, x, y, w, h, label, sub, fill, tip){
    const r = g.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",h).attr("rx",4).attr("fill",fill).attr("opacity",0.85);
    r.append("title").text(tip);
    txt(g, x+w/2, y+(sub ? 14 : h/2+4), label, {size:10.5, anchor:"middle", bold:true, fill:"#0f1115"});
    if (sub) txt(g, x+w/2, y+27, sub, {size:9.5, anchor:"middle", mono:true, fill:"#1b1f29"});
  }
  function draw(){
    svg.selectAll("*").remove();
    const k = val("arch-cfg"), c = CFG[k], p = params(c), g = svg.append("g");
    const inner = p.inner, ffName = c.gated ? "GEGLU" : "ReLU";
    const q = c.d+"×"+inner, ffS = c.gated ? "2×("+c.d+"×"+c.ff+") + "+c.ff+"×"+c.d : c.d+"×"+c.ff+", "+c.ff+"×"+c.d;
    const cols = [{x:20, t:"encoder block × "+c.Le, rows:[
        ["RMS norm", "g: "+c.d, P.muted, "scale-only norm, "+c.d+" gains"],
        ["self-attention (full) + bias", "Q,K,V: "+q+"  O: "+inner+"×"+c.d, P.A, "4 × "+c.d+" × "+inner+" = "+comma(p.attn)+" params; relative bias from layer 1 of the stack"],
        ["RMS norm", "g: "+c.d, P.muted, c.d+" gains"],
        ["feed-forward ("+ffName+")", ffS, P.B, comma(p.ffp)+" params"]]},
      {x:340, t:"decoder block × "+c.Ld, rows:[
        ["causal self-attention + bias", "Q,K,V: "+q+"  O: "+inner+"×"+c.d, P.A, comma(p.attn)+" params"],
        ["cross-attention → encoder", "Q: "+q+"  K,V from z", P.teal, comma(p.attn)+" params; no position bias"],
        ["feed-forward ("+ffName+")", ffS, P.B, comma(p.ffp)+" params"],
        ["3 RMS norms", "g: 3 × "+c.d, P.muted, 3*c.d+" gains"]]}];
    cols.forEach(col => {
      txt(g, col.x, 16, col.t, {size:11.5, bold:true});
      col.rows.forEach((r, i) => {
        const y = 26 + i*48;
        box(g, col.x, y, 290, 38, r[0], r[1], r[2], r[3]);
        if (i < col.rows.length - 1) g.append("line").attr("x1",col.x+145).attr("x2",col.x+145).attr("y1",y+38).attr("y2",y+48).attr("stroke",P.muted);
      });
      g.append("path").attr("d","M"+(col.x-8)+",26 L"+(col.x-8)+","+(26+4*48-10)).attr("stroke",P.good).attr("stroke-dasharray","3,3").attr("fill","none");
    });
    txt(g, 12, 234, "residual stream (dashed) is never normalised until the final norm of each stack", {size:10, fill:P.good});
    txt(g, 20, 256, "embedding "+c.V.toLocaleString("en-US")+" × "+c.d+(c.untied ? " + separate output matrix (untied)" : " shared with the output softmax (tied, output rescaled by d^−½)"), {size:10.5, fill:P.muted});
    txt(g, 20, 276, "inner attention width "+inner+" = "+(inner/c.d).toFixed(inner % c.d ? 2 : 0)+" × d_model · d_ff = "+(c.ff/c.d).toFixed(c.ff % c.d ? 2 : 0)+" × d_model", {size:10.5, fill:(inner !== c.d || c.ff/c.d > 4) ? P.B : P.muted});
    $("arch-read").textContent = c.name + ": encoder block " + comma(p.encBlock) + " params, decoder block " + comma(p.decBlock) +
      " (" + (p.decBlock/p.encBlock).toFixed(2) + "× — the cross-attention) · " + c.Le + " + " + c.Ld + " layers · total " + comma(p.total) + " parameters";
  }
  on("arch-cfg", "change", draw);
  draw();
});

/* ───────────────────────── 05 · relative position buckets ───────────────────────── */
TV.safe("rel", function(){
  const {P, txt, $, val, on, bucket} = TV;
  const svg = d3.select("#rel-svg");
  function draw(){
    svg.selectAll("*").remove();
    const bi = val("rel-dir") === "bi", nb = +val("rel-nb"), md = +val("rel-md");
    $("rel-nbv").textContent = nb; $("rel-mdv").textContent = md;
    const g = svg.append("g"), R = 160;
    const x = d3.scaleLinear().domain([-R, R]).range([50, 640]), y = d3.scaleLinear().domain([0, nb-1]).range([200, 20]);
    for (let b = 0; b < nb; b += Math.max(1, nb/8)){ g.append("line").attr("x1",50).attr("x2",640).attr("y1",y(b)).attr("y2",y(b)).attr("stroke",P.line);
      txt(g, 44, y(b)+3, b, {size:9.5, anchor:"end", fill:P.muted}); }
    [-160,-128,-64,0,64,128,160].forEach(v => txt(g, x(v), 216, v, {size:9.5, anchor:"middle", fill:P.muted}));
    txt(g, 345, 232, "relative offset r = key position − query position", {size:10, anchor:"middle", fill:P.muted});
    const pts = []; for (let r = -R; r <= R; r++) pts.push([r, bucket(r, bi, nb, md)]);
    const line = d3.line().x(d => x(d[0])).y(d => y(d[1])).curve(d3.curveStepAfter);
    g.append("path").attr("d", line(pts.filter(d => bi || d[0] <= 0))).attr("fill","none").attr("stroke",P.A).attr("stroke-width",2);
    if (!bi) g.append("path").attr("d", line(pts.filter(d => d[0] >= 0))).attr("fill","none").attr("stroke",P.grey).attr("stroke-dasharray","3,3");
    /* bucket starts on the "past" side */
    const starts = []; let prev = -1;
    for (let a = 0; a <= 4096; a++){ const b = bucket(-a, bi, nb, md); if (b !== prev){ starts.push([b, a]); prev = b; } }
    starts.forEach(([b, a]) => { if (a <= R && a >= Math.floor((bi ? nb/2 : nb)/2)) g.append("circle").attr("cx",x(-a)).attr("cy",y(b)).attr("r",2.5).attr("fill",P.B); });
    const lastStart = starts[starts.length-1][1];
    if (lastStart <= R) g.append("line").attr("x1",x(-lastStart)).attr("x2",x(-lastStart)).attr("y1",20).attr("y2",200).attr("stroke",P.B).attr("stroke-dasharray","2,3");
    if (md <= R) g.append("line").attr("x1",x(-md)).attr("x2",x(-md)).attr("y1",20).attr("y2",200).attr("stroke",P.pink).attr("stroke-dasharray","5,3");
    txt(g, 56, 30, (bi ? "encoder: keys after the query use buckets " + (nb/2) + "–" + (nb-1) : "decoder: past offsets only (future dashed = masked)"), {size:10, fill:P.muted});
    /* one query's row, coloured by bucket */
    const col = d3.scaleSequential(d3.interpolateViridis).domain([0, nb-1]), cw = (640-50)/(2*R+1);
    txt(g, 10, 256, "row i", {size:10, fill:P.muted});
    pts.forEach(([r, b]) => g.append("rect").attr("x",x(r)-cw/2).attr("y",246).attr("width",cw+0.3).attr("height",16).attr("fill",col(b)).attr("opacity", (!bi && r > 0) ? 0.2 : 1));
    txt(g, x(0), 278, "query", {size:9.5, anchor:"middle", fill:P.ink});
    const distinct = new Set(pts.map(d => d[1])).size;
    $("rel-read").textContent = (bi ? "encoder" : "decoder") + ", " + nb + " buckets, max distance " + md + ": exact distances 0–" +
      (Math.floor((bi ? nb/2 : nb)/2) - 1) + ", then logarithmic buckets start at distance " +
      starts.filter(s => s[1] >= Math.floor((bi ? nb/2 : nb)/2)).map(s => s[1]).join(", ") +
      " · last bucket covers distance ≥ " + lastStart + " · " + distinct + " distinct buckets used between −" + R + " and +" + R;
  }
  on("rel-dir", "change", draw);
  ["rel-nb","rel-md"].forEach(id => on(id, "input", draw));
  draw();
});

/* ───────────────────────── 07 · ReLU vs GEGLU ───────────────────────── */
TV.safe("ff", function(){
  const {P, txt, $, val, on, comma, CFG} = TV;
  const svg = d3.select("#ff-svg");
  const gelu = x => 0.5*x*(1 + Math.tanh(Math.sqrt(2/Math.PI)*(x + 0.044715*x*x*x)));   /* gelu_new */
  const PAIRS = {base:["t5-base","v11-base"], large:["t5-large","v11-large"], small:["t5-small","v11-small"]};
  function draw(){
    svg.selectAll("*").remove();
    const b = +val("ff-b")/10; $("ff-bv").textContent = b.toFixed(1);
    const g = svg.append("g");
    const x = d3.scaleLinear().domain([-4, 4]).range([40, 380]), y = d3.scaleLinear().domain([-4, 4]).range([250, 20]);
    [-4,-2,0,2,4].forEach(v => { g.append("line").attr("x1",x(v)).attr("x2",x(v)).attr("y1",20).attr("y2",250).attr("stroke",P.line);
      g.append("line").attr("x1",40).attr("x2",380).attr("y1",y(v)).attr("y2",y(v)).attr("stroke",P.line);
      txt(g, x(v), 264, v, {size:9.5, anchor:"middle", fill:P.muted}); txt(g, 34, y(v)+3, v, {size:9.5, anchor:"end", fill:P.muted}); });
    const xs = d3.range(-4, 4.001, 0.05);
    const series = [["ReLU(a)", a => Math.max(0, a), P.muted], ["GELU(a)", gelu, P.A], ["GELU(a) · "+b.toFixed(1), a => gelu(a)*b, P.B]];
    series.forEach(([n, f, c], i) => {
      g.append("path").attr("d", d3.line().x(a => x(a)).y(a => y(Math.max(-4, Math.min(4, f(a)))))(xs)).attr("fill","none").attr("stroke",c).attr("stroke-width",i === 2 ? 2.5 : 1.6);
      txt(g, 48, 34 + i*15, n, {size:10.5, fill:c, mono:true});
    });
    txt(g, 210, 278, "pre-activation a = x·W₀ (gate branch)", {size:10, anchor:"middle", fill:P.muted});
    /* parameter bars */
    const [k1, k2] = PAIRS[val("ff-cfg")], c1 = CFG[k1], c2 = CFG[k2];
    const p1 = 2*c1.d*c1.ff, p2 = 3*c2.d*c2.ff, match = 2*c1.ff/3;
    const bx = d3.scaleLinear().domain([0, Math.max(p1, p2)*1.1]).range([0, 200]);
    txt(g, 410, 40, "one feed-forward block", {size:11, bold:true});
    [[c1.name+" ReLU", p1, P.muted, c1.d+"×"+c1.ff+" ×2"], [c2.name+" GEGLU", p2, P.B, c2.d+"×"+c2.ff+" ×3"]].forEach(([n, v, c, s], i) => {
      const yy = 60 + i*56;
      txt(g, 410, yy, n, {size:10.5});
      g.append("rect").attr("x",410).attr("y",yy+6).attr("width",bx(v)).attr("height",18).attr("fill",c).attr("rx",3);
      txt(g, 414 + bx(v), yy+19, comma(v), {size:10, mono:true});
      txt(g, 410, yy+38, s, {size:9.5, mono:true, fill:P.muted});
    });
    txt(g, 410, 190, "d_ff′ matching v1.0 = 2/3 · "+c1.ff+" = "+match.toFixed(1), {size:10.5, fill:P.good});
    $("ff-read").textContent = "at a = 1: ReLU " + Math.max(0,1).toFixed(3) + ", GELU " + gelu(1).toFixed(3) + ", GEGLU unit " + (gelu(1)*b).toFixed(3) +
      " · at a = −1: ReLU 0, GELU " + gelu(-1).toFixed(3) + ", GEGLU " + (gelu(-1)*b).toFixed(3) +
      " · " + c2.name + " feed-forward has " + (p2/p1).toFixed(3) + "× the parameters of " + c1.name + "'s (released d_ff " + c2.ff + " vs 2/3-matched " + match.toFixed(1) + (c2.d !== c1.d ? "; d_model also differs" : "") + ")";
  }
  on("ff-b", "input", draw); on("ff-cfg", "change", draw);
  draw();
});

/* ───────────────────────── 09 · parameter breakdown ───────────────────────── */
TV.safe("params", function(){
  const {P, txt, $, val, on, CFG, params, cfgOptions, comma, big} = TV;
  const svg = d3.select("#par-svg");
  cfgOptions("par-cfg", Object.keys(CFG), "t5-base");
  const PARTS = [["emb","embedding",P.purple],["head","output matrix",P.pink],["encAttn","enc self-attn",P.A],["encFF","enc FF",P.B],
    ["decSelf","dec self-attn","#8bb8ff"],["decCross","cross-attn",P.teal],["decFF","dec FF","#ffcf8a"],["norms","norms + bias tables",P.grey]];
  function draw(){
    svg.selectAll("*").remove();
    const k = val("par-cfg"), c = CFG[k], p = params(c), g = svg.append("g"), logS = $("par-log").checked;
    p.normsRel = p.norms + p.rel;
    txt(g, 10, 16, c.name + " — " + comma(p.total) + " parameters", {size:11.5, bold:true});
    const sx = d3.scaleLinear().domain([0, p.total]).range([10, 650]);
    let acc = 0;
    PARTS.forEach(([key, name, col], i) => {
      const v = key === "norms" ? p.normsRel : p[key];
      if (v > 0){ g.append("rect").attr("x",sx(acc)).attr("y",24).attr("width",Math.max(0.5, sx(acc+v)-sx(acc))).attr("height",26).attr("fill",col)
        .append("title").text(name+": "+comma(v)+" ("+(100*v/p.total).toFixed(1)+"%)"); }
      acc += v;
      const lx = 10 + (i % 4)*160, ly = 66 + Math.floor(i/4)*16;
      g.append("rect").attr("x",lx).attr("y",ly-9).attr("width",9).attr("height",9).attr("fill",col);
      txt(g, lx+13, ly, name+" "+(100*v/p.total).toFixed(1)+"%", {size:9.5, fill:P.muted});
    });
    /* family */
    const keys = Object.keys(CFG), all = keys.map(kk => params(CFG[kk]).total);
    const fx = logS ? d3.scaleLog().domain([3e7, 2e10]).range([120, 500]) : d3.scaleLinear().domain([0, d3.max(all)*1.05]).range([120, 500]);
    keys.forEach((kk, i) => {
      const cc = CFG[kk], tot = all[i], yy = 104 + i*14;
      txt(g, 114, yy+9, cc.name, {size:9.5, anchor:"end", fill: kk === k ? P.ink : P.muted, bold: kk === k});
      g.append("rect").attr("x",120).attr("y",yy+1).attr("width",Math.max(1, fx(tot)-120)).attr("height",10).attr("fill", kk === k ? P.good : P.A).attr("opacity",0.8);
      txt(g, fx(tot)+4, yy+9, big(tot) + (cc.rep ? "  (paper " + cc.repL + ", " + (tot >= cc.rep ? "+" : "") + (100*(tot/cc.rep - 1)).toFixed(1) + "%)" : ""), {size:9, mono:true, fill:P.muted});
    });
    const embShare = (p.emb + p.head)/p.total;
    $("par-read").textContent = c.name + ": " + comma(p.total) + " (library count) · checkpoint file " + comma(p.ckpt) + " with the unused cross-attention bias table · embeddings " +
      (100*embShare).toFixed(1) + "% · non-embedding " + comma(p.nonEmb) + (c.rep ? " · paper's label " + c.repL + " is " + (100*(c.rep/p.total - 1)).toFixed(1) + "% off the recount" : "");
  }
  on("par-cfg", "change", draw); on("par-log", "change", draw);
  draw();
});

/* ───────────────────────── 10 · tokenizer ───────────────────────── */
TV.safe("tok", function(){
  const {P, txt, $, val, on, TOK} = TV;
  const svg = d3.select("#tok-svg");
  const PICK = ["Thank you for inviting me to your party last week.", "The <extra_id_0> walks in <extra_id_1> park", "def f(x): { return x ~ 2 }", "if a < b then",
    "naïve café", "東京", "Привет мир", "Grüße aus Köln", "line one\nline two", "  spaces   everywhere  ", "3.25", "entailment", "not_entailment"];
  const s = $("tok-pick");
  PICK.forEach((p, i) => { const o = document.createElement("option"); o.value = i; o.textContent = p.replace("\n", "⏎"); s.appendChild(o); });
  function draw(){
    svg.selectAll("*").remove();
    const str = PICK[+val("tok-pick")], t = TOK[str] || [], g = svg.append("g");
    txt(g, 10, 16, "text: " + JSON.stringify(str), {size:10.5, mono:true, fill:P.muted});
    const chips = t.map(([p, id]) => ({t: id === 2 ? p + "→<unk>" : p, kind: id === 2 ? "unk" : (id >= 32000 ? "sp" : "tok"), title:"id "+id}));
    const yEnd = tvChips(g, chips, 10, 28, 650, {size:10.5, cw:6.6, h:19});
    const idsRow = t.map(([, id]) => id).join(" ");
    txt(g, 10, yEnd + 22, "ids: " + (idsRow.length > 100 ? idsRow.slice(0, 100) + " …" : idsRow), {size:10, mono:true, fill:P.muted});
    const unk = t.filter(([, id]) => id === 2).length, sp = t.filter(([, id]) => id >= 32000).length;
    const chars = [...str.replace(/\s+/g, " ").trim()].length;
    $("tok-read").textContent = t.length + " pieces for " + chars + " characters after whitespace normalisation (" + (t.length ? (chars / t.length).toFixed(2) : "0") +
      " chars/piece) · " + unk + " unknown (id 2)" + (unk ? " — " + (100*unk/t.length).toFixed(0) + "% of pieces lost" : "") + (sp ? " · " + sp + " sentinel ids" : "") +
      (/\n/.test(str) ? " · the newline became an ordinary space" : "");
  }
  on("tok-pick", "change", draw);
  draw();
});

/* ───────────────────────── 11 · C4 cleaning ───────────────────────── */
TV.safe("c4", function(){
  const {P, txt, $, val, on} = TV;
  const svg = d3.select("#c4-svg");
  /* Mock pages invented for this figure. */
  const PAGES = {
    blog: ["Easy Weeknight Pasta", "Posted by Anna on March 3", "This pasta takes twenty minutes and uses only pantry staples.",
      "Boil the water and salt it generously before adding the pasta.", "Meanwhile, warm the olive oil and garlic in a wide pan.",
      "Toss everything together and serve with grated cheese!", "Yum!", "This site uses cookies to improve your experience.",
      "Please enable JavaScript to view the comments.", "Read more of my weeknight recipes here..."],
    code: ["Loops in JavaScript", "A for loop repeats a block of code a fixed number of times.", "for (let i = 0; i < 3; i++) {", "  console.log(i);", "}",
      "The loop above prints the numbers zero, one and two.", "Objects are written with braces, like {a: 1}, in this language.",
      "JavaScript runs in every modern web browser today."],
    shop: ["Home | Shop | Contact", "Blue Ceramic Mug", "$12.99", "Add to cart", "This mug holds 350 ml and is safe for the dishwasher.",
      "Free shipping on all orders over $50.", "By using this site you agree to our Terms of Use.", "Please read our privacy policy for details.",
      "Copyright 2019 Example Shop. All rights reserved."],
    wiki: ["The river rises in the northern hills and flows south.[1]", "It is 212 km long and drains an area of 3,400 square km.[2][citation needed]",
      "The town at its mouth was founded in 1804 by fishermen.[3]", "See also", "References"]
  };
  const POLICY = ["terms of use", "privacy policy", "cookie policy", "uses cookies", "use of cookies", "use cookies"];
  const END = [".", "?", "!", "\""];
  function sentences(line){ const m = line.match(/[.!?]+["']?(\s|$)/g); return m ? m.length : 0; }
  function clean(lines, R){
    const out = []; let nSent = 0, killed = null;
    for (const raw of lines){
      let line = raw.trim(), why = null;
      if (line.split(/\s+/).some(w => w.length > 1000)) why = "word > 1,000 chars";
      if (!why && R.cite) line = line.replace(/\[\d*\]|\[edit\]|\[citation needed\]/g, "");
      if (!why && R.end && (!END.some(e => line.endsWith(e)) || line.endsWith("..."))) why = line.endsWith("...") ? "ends in ellipsis" : "no terminal punctuation";
      if (!why && R.words && line.split(/\s+/).filter(Boolean).length < 5) why = "fewer than 5 words";
      const low = line.toLowerCase();
      if (!why && low.indexOf("lorem ipsum") >= 0){ killed = "lorem ipsum"; out.push({raw, why:"lorem ipsum → page removed"}); break; }
      if (!why && R.js && low.indexOf("javascript") >= 0) why = "contains \"javascript\"";
      if (!why && R.brace && line.indexOf("{") >= 0){ killed = "curly bracket"; out.push({raw, why:"\"{\" → page removed"}); break; }
      if (!why && R.policy && POLICY.some(p => low.indexOf(p) >= 0)) why = "policy boilerplate";
      if (!why){ nSent += sentences(line); out.push({raw, kept:line}); }
      else out.push({raw, why});
    }
    if (killed) return {out, nSent, verdict:"page removed (" + killed + ")", ok:false};
    if (R.sent && nSent < 3) return {out, nSent, verdict:"page removed (" + nSent + " sentence" + (nSent === 1 ? "" : "s") + " < 3)", ok:false};
    return {out, nSent, verdict:"page kept", ok:true};
  }
  function draw(){
    svg.selectAll("*").remove();
    const R = {end:$("c4-end").checked, words:$("c4-words").checked, js:$("c4-js").checked, brace:$("c4-brace").checked,
      policy:$("c4-policy").checked, cite:$("c4-cite").checked, sent:$("c4-sent").checked};
    const lines = PAGES[val("c4-page")], res = clean(lines, R), g = svg.append("g");
    lines.forEach((l, i) => {
      const y = 20 + i*28, r = res.out[i];
      const state = !r ? "not reached" : (r.kept !== undefined ? "kept" : r.why);
      const ok = r && r.kept !== undefined;
      g.append("rect").attr("x",6).attr("y",y-13).attr("width",648).attr("height",24).attr("rx",3).attr("fill", ok ? "rgba(74,222,128,.10)" : (r ? "rgba(248,113,113,.10)" : "rgba(154,163,178,.08)"));
      txt(g, 14, y+3, ok ? "✓" : (r ? "✗" : "·"), {size:12, fill: ok ? P.good : (r ? P.bad : P.muted)});
      const shown = l.replace(/\n/g, " ");
      txt(g, 32, y+3, shown.length > 62 ? shown.slice(0, 61) + "…" : shown, {size:10.5, mono:true, fill: ok ? P.ink : P.muted});
      txt(g, 646, y+3, state, {size:9.5, anchor:"end", fill: ok ? P.good : P.bad});
    });
    const yv = 20 + lines.length*28 + 8;
    txt(g, 14, yv, "verdict: " + res.verdict + " · " + res.nSent + " sentence(s) in kept lines", {size:11.5, bold:true, fill: res.ok ? P.good : P.bad});
    const kept = res.out.filter(r => r.kept !== undefined).length;
    $("c4-read").textContent = kept + " of " + lines.length + " lines kept · " + res.nSent + " sentences · " + res.verdict +
      (res.ok ? " · text written to C4: " + res.out.filter(r => r.kept !== undefined).map(r => r.kept).join(" ").length + " characters" : "");
  }
  ["c4-page","c4-end","c4-words","c4-js","c4-brace","c4-policy","c4-cite","c4-sent"].forEach(id => on(id, "change", draw));
  draw();
});

/* ───────────────────────── 12 · span-corruption sampler ───────────────────────── */
TV.safe("span", function(){
  const {P, txt, $, val, on, lcg, noiseMask} = TV;
  const svg = d3.select("#sp-svg");
  /* the first words of the paper's abstract, one word per token for display */
  const WORDS = ("Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique " +
    "in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, " +
    "we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based language problems into a text-to-text format.").split(" ");
  let seed = 7;
  on("sp-seed", "click", () => { seed += 1; draw(); });
  function draw(){
    svg.selectAll("*").remove();
    const T = Math.min(+val("sp-len"), WORDS.length), r = +val("sp-rate")/100, mu = +val("sp-mu");
    $("sp-lenv").textContent = T; $("sp-ratev").textContent = Math.round(r*100) + "%"; $("sp-muv").textContent = mu;
    const m = noiseMask(T, r, mu, lcg(seed*9973 + T*31 + mu));
    const toks = WORDS.slice(0, T), g = svg.append("g");
    const orig = toks.map((w, i) => ({t:w, kind: m.mask[i] ? "noise" : "tok"}));
    const inp = [], tgt = []; let sid = 0;
    for (let i = 0; i < T; i++){
      if (m.mask[i]){
        if (i === 0 || !m.mask[i-1]){ inp.push({t:"<X"+sid+">", kind:"sp", title:"<extra_id_"+sid+"> id "+(32099-sid)}); tgt.push({t:"<X"+sid+">", kind:"sp", title:"<extra_id_"+sid+">"}); sid++; }
        tgt.push({t:toks[i], kind:"noise"});
      } else inp.push({t:toks[i], kind:"tok"});
    }
    inp.push({t:"</s>", kind:"eos"}); tgt.push({t:"</s>", kind:"eos"});
    txt(g, 8, 12, "original text (" + T + " tokens; orange = dropped)", {size:10.5, fill:P.muted});
    let y = tvChips(g, orig, 8, 17, 652, {size:9.5, cw:5.7, h:15});
    txt(g, 8, y+14, "encoder input (" + inp.length + ")", {size:10.5, fill:P.A});
    y = tvChips(g, inp, 8, y+19, 652, {size:9.5, cw:5.7, h:15});
    txt(g, 8, y+14, "decoder target (" + tgt.length + ")", {size:10.5, fill:P.good});
    tvChips(g, tgt, 8, y+19, 652, {size:9.5, cw:5.7, h:15});
    const runs = []; let cur = 0; m.mask.forEach(b => { if (b) cur++; else if (cur){ runs.push(cur); cur = 0; } }); if (cur) runs.push(cur);
    $("sp-read").textContent = "noise = round(" + (r*T).toFixed(2) + ") = " + m.noise + " tokens in " + m.spans + " spans (lengths " + runs.join(", ") + "; mean " +
      (m.noise/m.spans).toFixed(2) + ") · input " + (T - m.noise) + " + " + m.spans + " sentinels + </s> = " + inp.length + " · target " + m.noise + " + " + m.spans +
      " sentinels + </s> = " + tgt.length + " · target/input " + (tgt.length/inp.length).toFixed(2);
  }
  ["sp-len","sp-rate","sp-mu"].forEach(id => on(id, "input", draw));
  draw();
});

/* ───────────────────────── 13 · length helper ───────────────────────── */
TV.safe("len", function(){
  const {P, txt, $, val, on, spanHelper, comma} = TV;
  const svg = d3.select("#len-svg");
  function draw(){
    svg.selectAll("*").remove();
    const L = +val("len-in"), r = +val("len-rate")/100, mu = +val("len-mu");
    $("len-inv").textContent = L; $("len-ratev").textContent = Math.round(r*100) + "%"; $("len-muv").textContent = mu;
    const h = spanHelper(L, r, mu), g = svg.append("g");
    const bx = d3.scaleLinear().domain([0, Math.max(h.raw, h.inp, h.tgt)*1.1]).range([0, 230]);
    [["raw text", h.raw, P.muted], ["encoder input", h.inp, P.A], ["decoder target", h.tgt, P.good]].forEach(([n, v, c], i) => {
      const y = 30 + i*44;
      txt(g, 10, y, n, {size:10.5});
      g.append("rect").attr("x",10).attr("y",y+6).attr("width",bx(v)).attr("height",18).attr("rx",3).attr("fill",c);
      txt(g, 14 + bx(v), y+20, comma(v), {size:10.5, mono:true});
    });
    txt(g, 10, 172, "noise " + h.noise + " · spans " + h.spans, {size:10.5, fill:P.muted, mono:true});
    txt(g, 10, 190, "BERT-style target would be " + L, {size:10.5, fill:P.muted});
    /* ratio curve across densities */
    const xs = d3.range(0.05, 0.501, 0.05), pts = xs.map(d => [d, (() => { const q = spanHelper(L, d, mu); return q.tgt/q.inp; })()]);
    const x = d3.scaleLinear().domain([0.05, 0.5]).range([330, 640]), y = d3.scaleLinear().domain([0, Math.max(1.05, d3.max(pts, p => p[1])*1.05)]).range([250, 30]);
    [0, 0.25, 0.5, 0.75, 1].forEach(v => { g.append("line").attr("x1",330).attr("x2",640).attr("y1",y(v)).attr("y2",y(v)).attr("stroke",P.line); txt(g, 324, y(v)+3, v.toFixed(2), {size:9.5, anchor:"end", fill:P.muted}); });
    [0.1, 0.2, 0.3, 0.4, 0.5].forEach(v => txt(g, x(v), 266, Math.round(v*100) + "%", {size:9.5, anchor:"middle", fill:P.muted}));
    g.append("path").attr("d", d3.line().x(p => x(p[0])).y(p => y(p[1]))(pts)).attr("fill","none").attr("stroke",P.good).attr("stroke-width",2);
    g.append("circle").attr("cx",x(r)).attr("cy",y(h.tgt/h.inp)).attr("r",4.5).attr("fill",P.B);
    txt(g, 485, 22, "target / input vs noise density (mean span " + mu + ")", {size:10.5, anchor:"middle", fill:P.muted});
    txt(g, 485, 282, "noise density", {size:10, anchor:"middle", fill:P.muted});
    $("len-read").textContent = "raw " + h.raw + " tokens → noise round(" + (h.raw*r).toFixed(2) + ") = " + h.noise + ", spans round(" + (h.noise/mu).toFixed(2) + ") = " + h.spans +
      " → input " + (h.raw - h.noise) + " + " + h.spans + " + 1 = " + h.inp + " · target " + h.noise + " + " + h.spans + " + 1 = " + h.tgt +
      " · target/input " + (100*h.tgt/h.inp).toFixed(1) + "% · target/raw " + (100*h.tgt/h.raw).toFixed(1) + "%";
  }
  ["len-in","len-rate","len-mu"].forEach(id => on(id, "input", draw));
  draw();
});

/* ───────────────────────── 14 · study explorer ───────────────────────── */
TV.safe("study", function(){
  const {P, txt, $, val, on, STUDY, BASE, SIGMA, METRICS} = TV;
  const svg = d3.select("#st-svg");
  const s = $("st-table");
  STUDY.forEach((t, i) => { const o = document.createElement("option"); o.value = i; o.textContent = t.t; if (i === 1) o.selected = true; s.appendChild(o); });
  function draw(){
    svg.selectAll("*").remove();
    const T = STUDY[+val("st-table")], m = +val("st-metric"), g = svg.append("g");
    const vals = T.rows.map(r => r[1][m]), mx = d3.max(vals), thr = mx - 2*SIGMA[m];
    const diffs = vals.map(v => v - BASE[m]);
    let lo = Math.min(0, d3.min(diffs), thr - BASE[m]), hi = Math.max(0, d3.max(diffs), mx - BASE[m]);
    const padX = Math.max(0.3, (hi - lo)*0.08); lo -= padX; hi += padX;
    const x = d3.scaleLinear().domain([lo, hi]).range([220, 540]);
    const rowH = Math.min(26, 290 / T.rows.length);
    g.append("rect").attr("x",x(thr - BASE[m])).attr("y",14).attr("width",Math.max(0, x(mx - BASE[m]) - x(thr - BASE[m]))).attr("height",rowH*T.rows.length+6).attr("fill",P.B).attr("opacity",0.10);
    g.append("line").attr("x1",x(0)).attr("x2",x(0)).attr("y1",12).attr("y2",rowH*T.rows.length+22).attr("stroke",P.muted);
    T.rows.forEach((r, i) => {
      const v = r[1][m], d = v - BASE[m], y = 18 + i*rowH, bold = v >= thr - 1e-9;
      txt(g, 212, y + rowH*0.6, r[0], {size:10, anchor:"end", fill: bold ? P.ink : P.muted, bold});
      g.append("rect").attr("x", Math.min(x(0), x(d))).attr("y", y+2).attr("width", Math.max(bold ? 3 : 1, Math.abs(x(d)-x(0)))).attr("height", rowH-6).attr("rx",2)
        .attr("fill", bold ? P.B : (d < 0 ? P.bad : P.A)).attr("opacity", bold ? 0.95 : 0.6);
      txt(g, 548, y + rowH*0.6, v.toFixed(2) + " (" + (d >= 0 ? "+" : "") + d.toFixed(2) + ")", {size:9.5, mono:true, fill:P.muted});
    });
    const yb = 18 + T.rows.length*rowH + 18;
    txt(g, x(0), yb, "baseline " + BASE[m].toFixed(2), {size:9.5, anchor:"middle", fill:P.muted});
    txt(g, 220, yb+14, "difference from the baseline in " + METRICS[m] + " · shaded band = within 2σ of the best", {size:9.5, fill:P.muted});
    const bolds = T.rows.filter(r => r[1][m] >= thr - 1e-9).map(r => r[0]);
    const best = T.rows[vals.indexOf(mx)][0];
    $("st-read").textContent = T.t + " · " + METRICS[m] + ": best " + best + " " + mx.toFixed(2) + " · 2σ = " + (2*SIGMA[m]).toFixed(3) +
      " · within 2σ of the best (bold in the paper's rule): " + bolds.join("; ") + " · range " + (d3.max(vals) - d3.min(vals)).toFixed(2) + " points = " +
      ((d3.max(vals) - d3.min(vals))/SIGMA[m]).toFixed(1) + "σ";
  }
  on("st-table", "change", draw); on("st-metric", "change", draw);
  draw();
});

/* ───────────────────────── 15 · attention masks ───────────────────────── */
TV.safe("mask", function(){
  const {P, txt, $, val, on} = TV;
  const svg = d3.select("#mk-svg");
  function mat(g, x0, y0, rows, cols, allow, title, rl, cl, cell){
    txt(g, x0, y0 - 8, title, {size:10.5, fill:P.muted});
    let n = 0;
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++){
      const ok = allow(i, j); if (ok) n++;
      g.append("rect").attr("x",x0 + j*cell).attr("y",y0 + i*cell).attr("width",cell-2).attr("height",cell-2).attr("rx",2).attr("fill", ok ? P.A : "#232838");
    }
    for (let i = 0; i < rows; i++) txt(g, x0 - 4, y0 + i*cell + cell*0.65, rl(i), {size:9, anchor:"end", fill:P.muted});
    for (let j = 0; j < cols; j++) txt(g, x0 + j*cell + cell/2 - 1, y0 + rows*cell + 11, cl(j), {size:9, anchor:"middle", fill:P.muted});
    return n;
  }
  function draw(){
    svg.selectAll("*").remove();
    const a = val("mk-arch"), n = +val("mk-in"), m = +val("mk-out"), g = svg.append("g");
    $("mk-inv").textContent = n; $("mk-outv").textContent = m;
    const xl = i => "x" + (i+1), yl = i => "y" + (i+1), both = i => i < n ? xl(i) : yl(i - n);
    let vis = 0, prods = 0, full = 0, desc;
    if (a === "encdec"){
      const cell = Math.min(24, 200 / Math.max(n, m));
      vis += mat(g, 40, 30, n, n, () => true, "encoder self-attn", xl, xl, cell);
      vis += mat(g, 280, 30, m, m, (i, j) => j <= i, "decoder self-attn", yl, yl, cell);
      vis += mat(g, 470, 30, m, n, () => true, "cross-attn", yl, xl, cell);
      prods = n*n + m*m + m*n; full = n; desc = "two stacks (2P parameters)";
    } else {
      const N = n + m, cell = Math.min(24, 240 / N);
      const allow = a === "lm" ? (i, j) => j <= i : (i, j) => (i < n ? j < n : j <= i);
      vis = mat(g, 60, 30, N, N, allow, a === "lm" ? "one stack, causal throughout" : "one stack, fully visible prefix", both, both, cell);
      prods = N*N; full = a === "lm" ? 1 : n; desc = "one stack (P parameters)";
      g.append("rect").attr("x",60 + n*cell - 1).attr("y",26).attr("width",1.5).attr("height",N*cell+6).attr("fill",P.B);
    }
    txt(g, 40, 290, "blue = may attend · rows are outputs (queries), columns inputs (keys)", {size:10, fill:P.muted});
    $("mk-read").textContent = desc + " · " + vis + " visible query–key pairs of " + prods + " products computed per layer per head · " +
      full + " of " + n + " input positions see the whole input" + (a === "lm" ? " (only the last input position does in a causal model)" : "");
  }
  on("mk-arch", "change", draw); ["mk-in","mk-out"].forEach(id => on(id, "input", draw));
  draw();
});

/* ───────────────────────── 18 · mixing rates ───────────────────────── */
TV.safe("mix", function(){
  const {P, txt, $, val, on, big} = TV;
  const svg = d3.select("#mx-svg");
  /* approximate training-set sizes in examples (TensorFlow Datasets versions; GLUE and SuperGLUE concatenated) */
  const TASKS = [["C4 (unsupervised)", Infinity], ["WMT En–Fr", 40.8e6], ["WMT En–De", 4.5e6], ["GLUE", 949e3], ["WMT En–Ro", 610e3],
    ["CNN/DM", 287e3], ["SuperGLUE", 148e3], ["SQuAD", 87.6e3]];
  function rates(mode, K, T){
    if (mode === "equal") return TASKS.map(() => 1/TASKS.length);
    const Kc = mode === "temp" ? Math.pow(2, 21) : K;
    let r = TASKS.map(t => Math.min(t[1], Kc)); const s = d3.sum(r); r = r.map(v => v/s);
    if (mode === "temp"){ r = r.map(v => Math.pow(v, 1/T)); const s2 = d3.sum(r); r = r.map(v => v/s2); }
    return r;
  }
  function draw(){
    svg.selectAll("*").remove();
    const mode = val("mx-mode"), lk = +val("mx-k"), K = Math.pow(2, lk), T = +val("mx-t");
    $("mx-kv").textContent = "2^" + lk + " = " + big(K); $("mx-tv").textContent = T;
    const r = rates(mode, K, T), g = svg.append("g");
    const x = d3.scaleLinear().domain([0, Math.max(0.2, d3.max(r))*1.1]).range([150, 560]);
    TASKS.forEach((t, i) => {
      const y = 20 + i*32;
      txt(g, 144, y+14, t[0], {size:10.5, anchor:"end"});
      g.append("rect").attr("x",150).attr("y",y+2).attr("width",Math.max(1, x(r[i]) - 150)).attr("height",18).attr("rx",3).attr("fill", i === 0 ? P.purple : P.A).attr("opacity",0.85);
      txt(g, x(r[i]) + 5, y+15, (100*r[i]).toFixed(2) + "%", {size:10, mono:true});
      txt(g, 650, y+15, isFinite(t[1]) ? big(t[1]) + " ex." : "unlimited", {size:9.5, anchor:"end", fill:P.muted});
    });
    txt(g, 150, 285, mode === "prop" ? "rₘ = min(eₘ, K) / Σ min(eₙ, K)" : mode === "temp" ? "rₘ ∝ (min(eₘ, 2²¹) / Σ)^(1/T)" : "rₘ = 1 / " + TASKS.length, {size:10.5, mono:true, fill:P.muted});
    const sup = 1 - r[0];
    $("mx-read").textContent = (mode === "prop" ? "examples-proportional, K = 2^" + lk : mode === "temp" ? "temperature T = " + T + " with K = 2^21" : "equal mixing") +
      " · unsupervised share " + (100*r[0]).toFixed(1) + "%, supervised " + (100*sup).toFixed(1) + "% · largest/smallest task rate " + (d3.max(r)/d3.min(r)).toFixed(1) + "× · SQuAD gets " +
      (100*r[7]).toFixed(2) + "% of examples";
  }
  on("mx-mode", "change", draw); ["mx-k","mx-t"].forEach(id => on(id, "input", draw));
  draw();
});

/* ───────────────────────── 20 · schedule and optimiser memory ───────────────────────── */
TV.safe("lr", function(){
  const {P, txt, $, val, on, CFG, params, cfgOptions, comma, big, bytes} = TV;
  const svg = d3.select("#lr-svg");
  cfgOptions("lr-cfg", ["t5-small","t5-base","t5-large","t5-3b","t5-11b","v11-xl","v11-xxl"], "t5-11b");
  const PRE = Math.pow(2, 19), FT = Math.pow(2, 18);
  /* Adafactor (factored, no momentum): rows + cols per weight matrix, full state for vectors */
  function adafactorState(c){
    const inner = c.h*c.kv, mats = [], vecs = [];
    mats.push([c.V, c.d]); if (c.untied) mats.push([c.V, c.d]);
    const ffm = c.gated ? [[c.d, c.ff], [c.d, c.ff], [c.ff, c.d]] : [[c.d, c.ff], [c.ff, c.d]];
    for (let i = 0; i < c.Le; i++){ for (let j = 0; j < 3; j++) mats.push([c.d, inner]); mats.push([inner, c.d]); ffm.forEach(x => mats.push(x)); vecs.push(c.d, c.d); }
    for (let i = 0; i < c.Ld; i++){ for (let j = 0; j < 6; j++) mats.push([c.d, inner]); mats.push([inner, c.d], [inner, c.d]); ffm.forEach(x => mats.push(x)); vecs.push(c.d, c.d, c.d); }
    mats.push([32, c.h], [32, c.h]); vecs.push(c.d, c.d);
    /* the loop above counts 3 input projections + 1 output per attention; decoder has two attentions: 6 inputs + 2 outputs */
    return d3.sum(mats, m => m[0] + m[1]) + d3.sum(vecs);
  }
  function draw(){
    svg.selectAll("*").remove();
    const lk = +val("lr-k"), k = Math.pow(10, lk), logx = $("lr-log").checked, c = CFG[val("lr-cfg")], p = params(c);
    $("lr-kv").textContent = "k = " + comma(k);
    const g = svg.append("g");
    const total = PRE + FT, x = logx ? d3.scaleLog().domain([1, total]).range([50, 420]) : d3.scaleLinear().domain([0, total]).range([50, 420]);
    const ymax = 1/Math.sqrt(k) * 1.1, y = d3.scaleLinear().domain([0, ymax]).range([250, 20]);
    y.ticks(5).forEach(v => { g.append("line").attr("x1",50).attr("x2",420).attr("y1",y(v)).attr("y2",y(v)).attr("stroke",P.line); txt(g, 46, y(v)+3, v.toFixed(3), {size:9, anchor:"end", fill:P.muted}); });
    const steps = logx ? d3.range(0, 200).map(i => Math.pow(total, i/199)) : d3.range(1, total, total/400);
    const lr = n => n <= PRE ? 1/Math.sqrt(Math.max(n, k)) : 1e-3;
    g.append("path").attr("d", d3.line().x(n => x(n)).y(n => y(lr(n)))(steps.filter(n => n <= PRE))).attr("fill","none").attr("stroke",P.A).attr("stroke-width",2);
    g.append("path").attr("d", d3.line().x(n => x(n)).y(() => y(1e-3))([PRE, total])).attr("fill","none").attr("stroke",P.B).attr("stroke-width",2);
    g.append("line").attr("x1",x(PRE)).attr("x2",x(PRE)).attr("y1",20).attr("y2",250).attr("stroke",P.muted).attr("stroke-dasharray","3,3");
    txt(g, x(PRE)-4, 34, "pretrain 2¹⁹", {size:9.5, anchor:"end", fill:P.A}); txt(g, x(PRE)+4, 34, "fine-tune", {size:9.5, fill:P.B});
    (logx ? [1, 100, 1e4, 1e6] : [0, 2e5, 4e5, 6e5]).forEach(v => { if (v <= total) txt(g, x(Math.max(v, logx ? 1 : 0)), 264, logx ? "1e" + Math.round(Math.log10(v)) : big(v), {size:9, anchor:"middle", fill:P.muted}); });
    txt(g, 235, 280, "step", {size:10, anchor:"middle", fill:P.muted});
    /* optimiser state */
    const adam = 2*p.total, ada = adafactorState(c);
    const bx = d3.scaleLog().domain([1e5, 1e11]).range([0, 190]);
    txt(g, 450, 40, "optimiser state, " + c.name, {size:11, bold:true});
    [["Adam (m, v)", adam, P.bad], ["Adafactor, factored", ada, P.good]].forEach(([n, v, col], i) => {
      const yy = 60 + i*60;
      txt(g, 450, yy, n, {size:10.5});
      g.append("rect").attr("x",450).attr("y",yy+6).attr("width",Math.max(2, bx(v))).attr("height",18).attr("rx",3).attr("fill",col);
      txt(g, 450, yy+40, big(v) + " numbers = " + bytes(4*v) + " fp32", {size:9.5, mono:true, fill:P.muted});
    });
    txt(g, 450, 200, "log scale, 10⁵ … 10¹¹", {size:9.5, fill:P.muted});
    const tokens = PRE * Math.pow(2, 16);
    $("lr-read").textContent = "lr = " + (1/Math.sqrt(k)).toFixed(4) + " for the first " + comma(k) + " steps, " + lr(PRE).toFixed(5) + " at step 2^19 (" +
      (lr(Math.max(k, 1))/lr(PRE)).toFixed(2) + "× lower) · pretraining tokens 2^19 × 2^16 = " + comma(tokens) + " · " + c.name + ": Adam state " + big(adam) +
      " vs factored Adafactor " + big(ada) + " (" + comma(adam/ada) + "× smaller)";
  }
  on("lr-k", "input", draw); on("lr-cfg", "change", draw); on("lr-log", "change", draw);
  draw();
});

/* ───────────────────────── 22 · results by size ───────────────────────── */
TV.safe("res", function(){
  const {P, txt, $, val, on, CFG, params} = TV;
  const svg = d3.select("#res-svg");
  /* Table 14 (test sets; SQuAD validation). Order: Small, Base, Large, 3B, 11B; prev = previous best as of Oct 24, 2019. */
  const R = [
    ["GLUE average", [77.4,82.7,86.4,88.5,90.3], 89.4], ["SuperGLUE average", [63.3,76.2,82.3,86.4,88.9], 84.6],
    ["SQuAD EM", [79.10,85.44,86.66,88.53,91.26], 90.1], ["SQuAD F1", [87.24,92.08,93.79,94.95,96.22], 95.5],
    ["CNN/DM ROUGE-2", [19.56,20.34,20.68,21.02,21.55], 20.30], ["CNN/DM ROUGE-L", [38.35,39.40,39.75,39.94,40.69], 40.63],
    ["WMT En–De BLEU", [26.7,30.9,32.0,31.8,32.1], 33.8], ["WMT En–Fr BLEU", [36.0,41.2,41.5,42.6,43.4], 43.8], ["WMT En–Ro BLEU", [26.8,28.0,28.1,28.2,28.1], 38.5],
    ["CoLA Matthews", [41.0,51.1,61.2,67.1,71.6], 69.2], ["RTE (GLUE) accuracy", [69.9,80.1,87.2,91.1,92.8], 89.2], ["MNLI-m accuracy", [82.4,87.1,89.9,91.4,92.2], 91.3],
    ["COPA accuracy", [46.0,71.2,83.4,92.0,94.8], 90.6], ["WiC accuracy", [66.9,68.3,69.3,72.1,76.9], 69.9], ["ReCoRD F1", [56.3,75.0,86.8,91.2,94.1], 90.6]];
  const KEYS = ["t5-small","t5-base","t5-large","t5-3b","t5-11b"], N = KEYS.map(k => params(CFG[k]).total);
  const s = $("res-metric");
  R.forEach((r, i) => { const o = document.createElement("option"); o.value = i; o.textContent = r[0]; s.appendChild(o); });
  function draw(){
    svg.selectAll("*").remove();
    const r = R[+val("res-metric")], showPrev = $("res-prev").checked, g = svg.append("g");
    const lo = Math.min(d3.min(r[1]), showPrev ? r[2] : Infinity), hi = Math.max(d3.max(r[1]), showPrev ? r[2] : -Infinity), pad = (hi - lo)*0.12 + 0.5;
    const x = d3.scaleLog().domain([4e7, 1.6e10]).range([60, 620]), y = d3.scaleLinear().domain([lo - pad, hi + pad]).range([240, 20]);
    y.ticks(6).forEach(v => { g.append("line").attr("x1",60).attr("x2",620).attr("y1",y(v)).attr("y2",y(v)).attr("stroke",P.line); txt(g, 54, y(v)+3, v, {size:9.5, anchor:"end", fill:P.muted}); });
    g.append("path").attr("d", d3.line().x((v, i) => x(N[i])).y(v => y(v))(r[1])).attr("fill","none").attr("stroke",P.A).attr("stroke-width",2);
    r[1].forEach((v, i) => { g.append("circle").attr("cx",x(N[i])).attr("cy",y(v)).attr("r",4.5).attr("fill",P.A);
      txt(g, x(N[i]), y(v)-9, v, {size:9.5, anchor:"middle", mono:true}); txt(g, x(N[i]), 258, CFG[KEYS[i]].name.replace("T5-", ""), {size:9.5, anchor:"middle", fill:P.muted}); });
    if (showPrev){ g.append("line").attr("x1",60).attr("x2",620).attr("y1",y(r[2])).attr("y2",y(r[2])).attr("stroke",P.B).attr("stroke-dasharray","6,4");
      txt(g, 616, y(r[2])-5, "previous best " + r[2], {size:9.5, anchor:"end", fill:P.B}); }
    txt(g, 340, 276, "parameters (recount, log scale)", {size:10, anchor:"middle", fill:P.muted});
    const first = r[1].findIndex(v => v > r[2]);
    const perDbl = (r[1][4] - r[1][0]) / Math.log2(N[4]/N[0]);
    $("res-read").textContent = r[0] + ": " + r[1][0] + " → " + r[1][4] + " from " + (N[0]/1e6).toFixed(1) + "M to " + (N[4]/1e9).toFixed(2) + "B parameters · " +
      perDbl.toFixed(2) + " points per doubling on average · " + (first >= 0 ? "first above the previous best at " + CFG[KEYS[first]].name : "never above the previous best (" + r[2] + ")") +
      " · largest single-step gain " + d3.max(d3.range(4), i => r[1][i+1] - r[1][i]).toFixed(2);
  }
  on("res-metric", "change", draw); on("res-prev", "change", draw);
  draw();
});

/* ───────────────────────── 29 · compute arithmetic ───────────────────────── */
TV.safe("cc", function(){
  const {P, txt, $, val, on, CFG, cfgOptions, big, bytes} = TV;
  const svg = d3.select("#cc-svg");
  cfgOptions("cc-cfg", ["t5-small","t5-base","t5-large","t5-3b","t5-11b","v11-base","v11-xl","v11-xxl"], "t5-base");
  function macs(c, n, m){
    const inner = c.h*c.kv, k = c.gated ? 3 : 2, lin = 4*c.d*inner + k*c.d*c.ff;
    const enc = c.Le * (n*lin + 2*n*n*inner);
    const dec = c.Ld * (m*(6*c.d*inner + k*c.d*c.ff) + n*2*c.d*inner + 2*m*m*inner + 2*m*n*inner);
    const one = (L) => L * ((n+m)*lin + 2*(n+m)*(n+m)*inner);
    const blockEnc = 4*c.d*inner + k*c.d*c.ff + 2*c.d, blockDec = 8*c.d*inner + k*c.d*c.ff + 3*c.d;
    return {ed: enc + dec, dN: one(c.Le), d2N: one(2*c.Le),
      pEd: c.Le*blockEnc + c.Ld*blockDec, pN: c.Le*blockEnc, p2N: 2*c.Le*blockEnc,
      kvEd: c.Ld * 2 * (m + n) * inner * 2, kvN: c.Le * 2 * (n + m) * inner * 2, kv2N: 2*c.Le * 2 * (n + m) * inner * 2};
  }
  function draw(){
    svg.selectAll("*").remove();
    const n = +val("cc-in"), m = +val("cc-out"), c = CFG[val("cc-cfg")];
    $("cc-inv").textContent = n; $("cc-outv").textContent = m;
    const r = macs(c, n, m), g = svg.append("g");
    const groups = [["forward FLOPs (2 × MACs)", [2*r.ed, 2*r.dN, 2*r.d2N]], ["block parameters", [r.pEd, r.pN, r.p2N]], ["KV cache at last token", [r.kvEd, r.kvN, r.kv2N]]];
    const names = [c.Le + " + " + c.Ld + " enc–dec", c.Le + "-layer decoder-only", (2*c.Le) + "-layer decoder-only"], cols = [P.good, P.A, P.purple];
    groups.forEach((gr, gi) => {
      const x0 = 20 + gi*215, mx = d3.max(gr[1]);
      txt(g, x0, 18, gr[0], {size:10.5, bold:true});
      gr[1].forEach((v, i) => {
        const hgt = 170 * v / mx, y = 210 - hgt;
        g.append("rect").attr("x",x0 + i*62).attr("y",y).attr("width",50).attr("height",Math.max(1, hgt)).attr("rx",3).attr("fill",cols[i]).attr("opacity",0.85)
          .append("title").text(names[i] + ": " + (gi === 2 ? bytes(v) : big(v)));
        txt(g, x0 + i*62 + 25, y - 5, gi === 2 ? bytes(v) : big(v), {size:9.5, anchor:"middle", mono:true});
      });
    });
    names.forEach((nm, i) => { g.append("rect").attr("x",20 + i*210).attr("y",238).attr("width",10).attr("height",10).attr("fill",cols[i]); txt(g, 34 + i*210, 247, nm, {size:10, fill:P.muted}); });
    txt(g, 20, 272, "shapes of " + c.name + " · input " + n + ", target " + m + " tokens", {size:10, fill:P.muted});
    $("cc-read").textContent = "enc–dec " + big(2*r.ed) + " FLOPs vs " + c.Le + "-layer decoder-only " + big(2*r.dN) + " (ratio " + (r.ed/r.dN).toFixed(2) + ") and " + (2*c.Le) +
      "-layer " + big(2*r.d2N) + " (ratio " + (r.ed/r.d2N).toFixed(2) + ") · parameters " + big(r.pEd) + " vs " + big(r.pN) + " / " + big(r.p2N) +
      " · KV cache " + bytes(r.kvEd) + " vs " + bytes(r.kvN) + " / " + bytes(r.kv2N);
  }
  ["cc-in","cc-out"].forEach(id => on(id, "input", draw)); on("cc-cfg", "change", draw);
  draw();
});
