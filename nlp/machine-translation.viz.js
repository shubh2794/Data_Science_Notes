/* machine-translation.viz.js — extracted from machine-translation.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#mt-svg"),W=640;
  const pairs=[
    {src:["Le","chat","noir"], tgt:["The","black","cat"], A:[[0.9,0.05,0.05],[0.04,0.06,0.9],[0.05,0.9,0.05]]},
    {src:["Je","t'","aime"], tgt:["I","love","you"], A:[[0.92,0.04,0.04],[0.05,0.05,0.9],[0.05,0.88,0.07]]},
  ];
  function draw(){
    svg.selectAll("*").remove();
    const p=pairs[+d3.select("#mt-sel").property("value")];
    const top=46,left=120,cell=44;
    svg.append("text").attr("x",left+p.src.length*cell/2).attr("y",22).attr("text-anchor","middle").attr("fill","#5b9cff").attr("font-size",11).text("source →");
    svg.append("text").attr("x",26).attr("y",top+p.tgt.length*cell/2).attr("fill",C.accent).attr("font-size",11).attr("transform",`rotate(-90,30,${top+p.tgt.length*cell/2})`).text("target");
    p.src.forEach((s,j)=>svg.append("text").attr("x",left+j*cell+cell/2).attr("y",top-8).attr("text-anchor","middle").attr("font-size",11).attr("fill",C.muted).text(s));
    p.tgt.forEach((t,i)=>{
      svg.append("text").attr("x",left-10).attr("y",top+i*cell+cell/2+4).attr("text-anchor","end").attr("font-size",11).attr("fill",C.muted).text(t);
      p.src.forEach((_,j)=>{
        const w=p.A[i][j];
        svg.append("rect").attr("x",left+j*cell).attr("y",top+i*cell).attr("width",cell-3).attr("height",cell-3).attr("rx",3)
          .attr("fill",d3.interpolatePurples(0.15+w*0.8));
        if(w>0.4) svg.append("text").attr("x",left+j*cell+cell/2-1).attr("y",top+i*cell+cell/2+3).attr("text-anchor","middle").attr("font-size",9).attr("fill","#fff").text(w.toFixed(2));
      });
    });
    d3.select("#mt-read").html(`each target word attends mostly to its translation — alignment is <b>non-monotonic</b> (word order differs between languages)`);
  }
  d3.select("#mt-sel").on("change",draw);
  draw();
})();

/* ───────────────────────── 07 · live BLEU scoring ───────────────────────── */
(function(){
  const C={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",bad:"#f87171",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#bleu-svg"),W=640,H=240,N=4;
  const reference="the cat sat on the mat";
  const candidates=[
    {label:"near-perfect", text:"the cat sat on the mat"},
    {label:"paraphrase",   text:"a cat was sitting on the mat"},
    {label:"too short",    text:"the cat"},
    {label:"repeats word", text:"the the the the the the"},
    {label:"reordered",    text:"on the mat the cat sat"},
  ];
  const sel=d3.select("#bleu-sel");
  sel.selectAll("option").data(candidates).join("option").attr("value",(d,i)=>i).text(d=>d.label+" — “"+d.text+"”");
  d3.select("#bleu-ref").html("reference: <b>“"+reference+"”</b>");

  const tok=s=>s.toLowerCase().trim().split(/\s+/).filter(Boolean);
  function ngrams(arr,n){ const m=new Map(); for(let i=0;i+n<=arr.length;i++){ const k=arr.slice(i,i+n).join(" "); m.set(k,(m.get(k)||0)+1); } return m; }
  // modified (clipped) precision for order n
  function precision(cand,ref,n){
    const cg=ngrams(cand,n), rg=ngrams(ref,n);
    let match=0,total=0;
    cg.forEach((cnt,k)=>{ total+=cnt; match+=Math.min(cnt, rg.get(k)||0); });
    return {match,total,p: total>0 ? match/total : 0};
  }

  const m={t:30,r:24,b:46,l:48};
  const x=d3.scaleBand().domain(d3.range(1,N+1)).range([m.l,W-m.r]).padding(0.32);
  const y=d3.scaleLinear().domain([0,1]).range([H-m.b,m.t]);
  const gAxis=svg.append("g"), gBars=svg.append("g"), gLab=svg.append("g");
  // y gridlines
  [0,0.25,0.5,0.75,1].forEach(t=>{
    gAxis.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(t)).attr("y2",y(t)).attr("stroke",C.line).attr("stroke-opacity",.5);
    gAxis.append("text").attr("x",m.l-8).attr("y",y(t)+4).attr("text-anchor","end").attr("font-size",10).attr("fill",C.muted).text(t.toFixed(2));
  });
  gAxis.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-12).attr("text-anchor","middle").attr("font-size",11).attr("fill",C.muted).text("n-gram order  →  modified precision pₙ");

  function draw(){
    const cand=tok(candidates[+sel.property("value")].text), ref=tok(reference);
    const rows=d3.range(1,N+1).map(n=>{ const r=precision(cand,ref,n); return {n,...r}; });
    // BLEU = BP · exp( (1/N) Σ ln p_n )  with p_n=0 → BLEU 0
    const c=cand.length, r=ref.length;
    const BP = c>r ? 1 : Math.exp(1 - r/Math.max(c,1));
    let logsum=0, anyZero=false;
    rows.forEach(d=>{ if(d.p>0) logsum+=Math.log(d.p); else anyZero=true; });
    const geo = anyZero ? 0 : Math.exp(logsum/N);
    const bleu = BP*geo;

    const bars=gBars.selectAll("rect.bar").data(rows,d=>d.n);
    bars.join(
      en=>en.append("rect").attr("class","bar").attr("x",d=>x(d.n)).attr("width",x.bandwidth()).attr("y",y(0)).attr("height",0),
      up=>up,
      ex=>ex.remove()
    ).transition().duration(450)
      .attr("x",d=>x(d.n)).attr("width",x.bandwidth())
      .attr("y",d=>y(d.p)).attr("height",d=>y(0)-y(d.p)).attr("rx",4)
      .attr("fill",d=>d.p>0?C.accent:C.bad).attr("fill-opacity",.85);

    const labs=gLab.selectAll("g.bl").data(rows,d=>d.n);
    const en=labs.enter().append("g").attr("class","bl");
    en.append("text").attr("class","pv");
    en.append("text").attr("class","mt");
    en.append("text").attr("class","nx");
    const all=en.merge(labs);
    all.select("text.pv").attr("x",d=>x(d.n)+x.bandwidth()/2).attr("y",d=>y(d.p)-7).attr("text-anchor","middle").attr("font-size",11).attr("font-weight",700).attr("fill",d=>d.p>0?C.good:C.bad).text(d=>d.p.toFixed(2));
    all.select("text.mt").attr("x",d=>x(d.n)+x.bandwidth()/2).attr("y",H-m.b+16).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.muted).text(d=>d.match+"/"+d.total);
    all.select("text.nx").attr("x",d=>x(d.n)+x.bandwidth()/2).attr("y",H-m.b+30).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.ink).text(d=>(d.n)+"-gram");
    labs.exit().remove();

    const bc = bleu>0.5?C.good : bleu>0.2?C.a2 : C.bad;
    d3.select("#bleu-read").html(
      `len c=<b>${c}</b>, r=<b>${r}</b> &nbsp;·&nbsp; BP=<b>${BP.toFixed(3)}</b> &nbsp;·&nbsp; geo-mean pₙ=<b>${geo.toFixed(3)}</b>`+
      (anyZero?` <span style="color:${C.bad}">(a pₙ=0 zeroes the product)</span>`:``)+
      ` &nbsp;⇒&nbsp; BLEU = <b style="color:${bc}">${bleu.toFixed(3)}</b>`
    );
  }
  sel.on("change",draw);
  draw();
})();
