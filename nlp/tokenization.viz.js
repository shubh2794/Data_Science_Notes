/* tokenization.viz.js — extracted from tokenization.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

const PC={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};

/* ───────────────────────── 01 · greedy sub-word tokenizer ───────────────────────── */
(function(){
  const words=["the","a","to","of","and","is","in","it","make","makes","power","powerful","trans","former","transform","transformer","transformers","token","tokens","tokenize","tokenization","model","models","language","learn","learning","deep","neural","network","word","embed","embedding","attention","context","vector","data","gener","generate","un","imag","ine","able","ly","ing","ed","er","ers","s","ization","ation","ize"];
  const subs=["tion","ize","iza","ing","ness","ment","able","ful","un","re","pre","over","under","form","port","struct","work","net","sub"];
  let vocab=Array.from(new Set([...words,...subs]));
  "abcdefghijklmnopqrstuvwxyz".split("").forEach(c=>{ if(!vocab.includes(c)) vocab.push(c); });
  const vmap=new Map(vocab.map((v,i)=>[v,i+5])); // ids offset for "specials"
  const pal=["#c084fc","#5b9cff","#4ade80","#ffb454","#f472b6","#22d3ee","#fbbf24"];

  function tokenizeWord(w){
    const out=[]; let i=0; const lw=w.toLowerCase();
    while(i<lw.length){
      let end=lw.length, piece=null;
      while(end>i){ const cand=lw.slice(i,end); if(vmap.has(cand)){piece=cand;break;} end--; }
      if(piece===null){ out.push({t:lw[i],id:vmap.get(lw[i]),cont:i>0}); i++; }
      else { out.push({t:piece,id:vmap.get(piece),cont:i>0}); i=end; }
    }
    return out;
  }
  const box=d3.select("#tok-out");
  function render(){
    const text=d3.select("#tok-input").property("value");
    const wordList=text.split(/(\s+)/).filter(s=>s.length);
    const chips=[]; let ci=0;
    wordList.forEach(seg=>{
      if(/^\s+$/.test(seg)) return;
      tokenizeWord(seg).forEach(tk=>{
        chips.push({label:(tk.cont?"##":"")+tk.t, id:tk.id, col:pal[(ci++)%pal.length]});
      });
    });
    box.selectAll("span.tok").data(chips).join("span")
      .attr("class","tok").attr("title",d=>"id "+d.id)
      .attr("style",d=>`display:inline-block;padding:3px 9px;border-radius:7px;font-size:13px;font-family:'SF Mono',Menlo,monospace;background:${d.col}22;color:${d.col};border:1px solid ${d.col}55;`)
      .text(d=>d.label);
    const chars=text.replace(/\s/g,"").length, total=chips.length;
    d3.select("#tok-readout").html(`<b>${total}</b> tokens from <b>${chars}</b> characters · ratio ≈ <b>${chars?(chars/total).toFixed(1):0}</b> chars/token (hover a token for its id)`);
  }
  d3.select("#tok-input").on("input",render);
  render();
})();

/* ───────────────────────── 03 · BPE merge stepper ───────────────────────── */
(function(){
  const svg=d3.select("#bpe-svg"),W=640;
  const stepBtn=document.getElementById("bpe-step"),
        resetBtn=document.getElementById("bpe-reset"),
        readout=document.getElementById("bpe-readout"),
        rulesEl=document.getElementById("bpe-rules");
  // tiny corpus: word + frequency (· = end-of-word marker)
  const CORPUS=[["low",5],["lower",2],["newest",6],["widest",3]];
  let words, rules, lastPair;

  function reset(){
    words=CORPUS.map(([w,f])=>({sym:w.split("").concat("·"),f}));
    rules=[]; lastPair=null;
    stepBtn.disabled=false; stepBtn.style.opacity=1;
    render();
  }
  function pairCounts(){
    const m=new Map();
    words.forEach(({sym,f})=>{
      for(let i=0;i<sym.length-1;i++){
        const k=sym[i]+"\u0001"+sym[i+1];
        m.set(k,(m.get(k)||0)+f);
      }
    });
    return m;
  }
  function step(){
    const m=pairCounts();
    if(m.size===0){ finish(); return; }
    let best=null,bc=-1;
    m.forEach((c,k)=>{ if(c>bc){bc=c;best=k;} });
    const [a,b]=best.split("\u0001");
    lastPair={a,b,c:bc};
    words.forEach(o=>{
      const ns=[];
      for(let i=0;i<o.sym.length;i++){
        if(i<o.sym.length-1 && o.sym[i]===a && o.sym[i+1]===b){ ns.push(a+b); i++; }
        else ns.push(o.sym[i]);
      }
      o.sym=ns;
    });
    rules.push({pair:a+" + "+b, res:a+b, c:bc});
    render();
    if(pairCounts().size===0) finish();
  }
  function finish(){ stepBtn.disabled=true; stepBtn.style.opacity=.45; lastPair=null; render(); }

  function render(){
    const g=svg.html("").append("g");
    const rowH=46, padL=20;
    words.forEach((o,r)=>{
      const y=22+r*rowH;
      g.append("text").attr("x",padL).attr("y",y+4).attr("font-size",11).attr("fill",PC.muted)
        .attr("font-family","'SF Mono',Menlo,monospace").text("×"+o.f);
      let x=padL+44;
      o.sym.forEach((s)=>{
        const wpx=Math.max(20, s.length*9+14);
        const isNew=lastPair && s===lastPair.a+lastPair.b;
        const isEow=s==="·";
        const col=isNew?PC.good:isEow?PC.muted:PC.accent;
        g.append("rect").attr("x",x).attr("y",y-15).attr("width",wpx).attr("height",26).attr("rx",6)
          .attr("fill",col+(isNew?"33":"1f")).attr("stroke",col+(isNew?"":"66"))
          .attr("stroke-width",isNew?1.8:1);
        g.append("text").attr("x",x+wpx/2).attr("y",y+3).attr("text-anchor","middle")
          .attr("font-size",12.5).attr("font-family","'SF Mono',Menlo,monospace")
          .attr("font-weight",isNew?700:500).attr("fill",isNew?PC.good:isEow?PC.muted:PC.ink).text(s);
        x+=wpx+6;
      });
    });
    const ntok=words.reduce((s,o)=>s+o.sym.length,0);
    const vocab=new Set(); words.forEach(o=>o.sym.forEach(s=>vocab.add(s)));
    readout.innerHTML = lastPair
      ? `merged <b style="color:${PC.good}">${lastPair.a} + ${lastPair.b}</b> (count ${lastPair.c}) · <b>${rules.length}</b> rules · <b>${ntok}</b> symbols · |V| <b>${vocab.size}</b>`
      : (rules.length? `done — no adjacent pair left · <b>${rules.length}</b> rules · |V| <b>${vocab.size}</b>`
                     : `start: every word is single characters (· = end-of-word) · <b>${ntok}</b> symbols`);
    rulesEl.innerHTML = rules.length
      ? "merge rules: "+rules.map(r=>`<b style="color:${PC.accent}">${r.pair} → ${r.res}</b>`).join(" &nbsp;·&nbsp; ")
      : "merge rules: <i>none yet — click “Merge next pair”</i>";
  }

  stepBtn.addEventListener("click",step);
  resetBtn.addEventListener("click",reset);
  reset();
})();
