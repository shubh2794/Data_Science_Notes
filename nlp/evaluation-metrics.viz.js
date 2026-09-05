/* evaluation-metrics.viz.js — extracted from evaluation-metrics.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

const PC={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};

/* ───────────────────────── 05 · LLM-as-a-Judge position bias ───────────────────────── */
(function(){
  const svg=d3.select("#bias-svg"),W=640;
  // Truth: B is genuinely better. A biased judge favors the FIRST-shown answer.
  // order = [first, second]; with averaging we run both orders and combine.
  let order=["A","B"], avg=false;
  const gSlots=svg.append("g").attr("transform","translate(0,40)");
  const out=document.getElementById("bias-readout");
  const verdict=svg.append("text").attr("x",W/2).attr("y",150).attr("text-anchor","middle").attr("font-size",14).attr("font-weight",700);
  function judge(first){ return first; } // biased: picks whoever is shown first
  function render(){
    const data=[{slot:"shown first",ans:order[0],x:160},{slot:"shown second",ans:order[1],x:480}];
    const g=gSlots.selectAll("g.slot").data(data,d=>d.slot).join(enter=>{
        const e=enter.append("g").attr("class","slot");
        e.append("rect").attr("x",-110).attr("width",220).attr("height",64).attr("rx",10).attr("fill","#222733").attr("stroke",PC.line);
        e.append("text").attr("class","ans").attr("y",30).attr("text-anchor","middle").attr("font-size",20).attr("font-weight",700);
        e.append("text").attr("class","slt").attr("y",52).attr("text-anchor","middle").attr("font-size",11).attr("fill",PC.muted);
        return e;});
    g.attr("transform",d=>`translate(${d.x},0)`);
    g.select(".slt").text(d=>d.slot);
    let winner, text;
    if(avg){
      // run both orders; biased judge picks first each time → each answer wins once → tie broken by truth (B)
      winner="B"; text="Order-averaged: A and B each win once when first → tie; true winner B prevails.";
      g.select(".ans").attr("fill",d=>d.ans==="B"?PC.good:PC.ink).text(d=>d.ans);
    } else {
      const pick=judge(order[0]); winner=pick;
      const correct = pick==="B";
      text = correct ? `Judge picks ${pick} (shown first) — happens to match truth (B).`
                     : `Judge picks ${pick} (shown first) — WRONG, truth is B. Position bias.`;
      g.select(".ans").attr("fill",d=>d.ans===pick?(pick==="B"?PC.good:PC.bad):PC.ink).text(d=>d.ans);
    }
    verdict.attr("fill", winner==="B"?PC.good:PC.bad).text("Verdict: "+winner+(winner==="B"?"  ✓ (truth)":"  ✗ (truth is B)"));
    out.innerHTML=text;
  }
  document.getElementById("bias-swap").onclick=()=>{avg=false; order=[order[1],order[0]]; render();};
  document.getElementById("bias-avg").onclick=()=>{avg=!avg; render();};
  render();
})();

/* ───────────────────────── 06 · BLEU vs ROUGE vs BERTScore ───────────────────────── */
(function(){
  const reference=["the","quick","brown","fox","jumps","over","the","lazy","dog"];
  // synonym similarity table for the stand-in BERTScore (1.0 = same word)
  const SIM={ "quick":{"fast":0.92}, "fast":{"quick":0.92}, "jumps":{"leaps":0.9}, "leaps":{"jumps":0.9},
             "lazy":{"sleepy":0.88}, "sleepy":{"lazy":0.88}, "brown":{"tan":0.8}, "tan":{"brown":0.8} };
  function sim(a,b){ if(a===b)return 1; if(SIM[a]&&SIM[a][b]!=null)return SIM[a][b]; return 0.15; }
  function ngrams(arr,n){ const o=[]; for(let i=0;i+n<=arr.length;i++)o.push(arr.slice(i,i+n).join(" ")); return o; }
  function bleu(cand){
    let logsum=0, ok=true;
    for(let n=1;n<=2;n++){ // use n=1,2 for the toy
      const cg=ngrams(cand,n), rg=ngrams(reference,n);
      if(cg.length===0){ok=false;break;}
      const rc={}; rg.forEach(g=>rc[g]=(rc[g]||0)+1);
      let match=0; const used={};
      cg.forEach(g=>{ used[g]=(used[g]||0); if((rc[g]||0)>used[g]){match++; used[g]++;} });
      const p=match/cg.length; if(p===0){ok=false;break;} logsum+=0.5*Math.log(p);
    }
    let bp=1; const c=cand.length,r=reference.length; if(c<=r) bp=Math.exp(1-r/c);
    return ok?bp*Math.exp(logsum):0;
  }
  function rouge1(cand){ // recall-oriented unigram overlap
    const rc={}; reference.forEach(w=>rc[w]=(rc[w]||0)+1);
    let match=0; const used={};
    cand.forEach(w=>{ used[w]=used[w]||0; if((rc[w]||0)>used[w]){match++;used[w]++;} });
    return match/reference.length;
  }
  function bertscore(cand){ // greedy-match precision/recall with synonym similarity → F1
    const P=cand.reduce((a,w)=>a+Math.max(...reference.map(r=>sim(w,r))),0)/cand.length;
    const R=reference.reduce((a,w)=>a+Math.max(...cand.map(c=>sim(w,c))),0)/reference.length;
    return 2*P*R/((P+R)||1);
  }
  const variants={
    exact:{cand:reference.slice(), note:"Identical to reference → all metrics ≈ 1."},
    syn:{cand:["the","fast","brown","fox","leaps","over","the","sleepy","dog"], note:"Synonyms swapped: BLEU & ROUGE drop (different words), BERTScore holds (same meaning)."},
    drop:{cand:["the","quick","fox","jumps","over","the","lazy","dog"], note:"A word dropped: ROUGE recall falls, BLEU dips via brevity penalty, BERTScore mostly intact."},
    reorder:{cand:["the","lazy","dog","the","quick","brown","fox","jumps","over"], note:"Reordered: bigram precision (BLEU) suffers most; unigram-based ROUGE/BERTScore barely move."},
  };
  const svg=d3.select("#cmp-svg"),W=640,H=280,m={t:30,r:24,b:40,l:96};
  const metrics=[{k:"BLEU",c:PC.accent},{k:"ROUGE-1",c:PC.a2},{k:"BERTScore",c:PC.good}];
  const x=d3.scaleLinear().domain([0,1]).range([m.l,W-m.r]);
  const y=d3.scaleBand().domain(metrics.map(d=>d.k)).range([m.t,H-m.b]).padding(0.4);
  const gRow=svg.append("g");
  // candidate sentence display
  svg.append("text").attr("id","cmp-cand").attr("x",m.l).attr("y",20).attr("font-size",12).attr("fill",PC.ink).attr("font-style","italic");
  function render(key){
    const v=variants[key], cand=v.cand;
    svg.select("#cmp-cand").text("candidate: “"+cand.join(" ")+"”");
    const vals={"BLEU":bleu(cand),"ROUGE-1":rouge1(cand),"BERTScore":bertscore(cand)};
    const rows=metrics.map(mt=>({k:mt.k,c:mt.c,v:vals[mt.k]}));
    const g=gRow.selectAll("g.row").data(rows,d=>d.k).join(enter=>{
        const e=enter.append("g").attr("class","row");
        e.append("text").attr("class","lbl").attr("x",m.l-10).attr("text-anchor","end").attr("font-size",12).attr("fill",PC.muted);
        e.append("rect").attr("class","track").attr("x",m.l).attr("rx",5).attr("height",y.bandwidth()).attr("fill","#222733");
        e.append("rect").attr("class","bar").attr("x",m.l).attr("rx",5).attr("height",y.bandwidth());
        e.append("text").attr("class","val").attr("font-size",12).attr("dy",y.bandwidth()/2+4).attr("fill",PC.ink);
        return e;});
    g.attr("transform",d=>`translate(0,${y(d.k)})`);
    g.select(".lbl").attr("y",y.bandwidth()/2+4).text(d=>d.k);
    g.select(".track").attr("width",x(1)-x(0));
    g.select(".bar").transition().duration(450).attr("width",d=>Math.max(0,x(d.v)-x(0))).attr("fill",d=>d.c);
    g.select(".val").transition().duration(450).attr("x",d=>x(d.v)+8).text(d=>d.v.toFixed(2));
    d3.select("#cmp-readout").html("<b>"+key.toUpperCase()+":</b> "+v.note);
  }
  document.getElementById("cmp-syn").onclick=()=>render("syn");
  document.getElementById("cmp-drop").onclick=()=>render("drop");
  document.getElementById("cmp-reorder").onclick=()=>render("reorder");
  document.getElementById("cmp-reset").onclick=()=>render("exact");
  render("exact");
})();
