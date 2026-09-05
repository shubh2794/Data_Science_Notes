/* hidden-markov-models-crfs.viz.js — extracted from hidden-markov-models-crfs.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

const PC={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",bad:"#f87171",blue:"#5b9cff",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};

/* ───────────────────────── 04 · Viterbi trellis ───────────────────────── */
(function(){
  const C=PC;
  const svg=d3.select("#hmm-svg"),W=640,H=250;
  const states=["Det","Noun","Verb"];
  const data=[
    {words:["the","dog","runs"], best:[0,1,2]},   // Det Noun Verb
    {words:["she","can","fish"], best:[1,2,2]},    // Noun(Pron) Verb Verb
  ];
  function draw(){
    svg.selectAll("*").remove();
    const d=data[+d3.select("#hmm-sel").property("value")];
    const n=d.words.length, colX=i=>110+i*160, rowY=j=>60+j*56;
    // words header
    d.words.forEach((w,i)=>svg.append("text").attr("x",colX(i)).attr("y",30).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",13).attr("font-family","SF Mono, monospace").text(w));
    svg.append("text").attr("x",30).attr("y",30).attr("fill",C.muted).attr("font-size",11).text("word:");
    // transitions (all + best)
    for(let i=0;i<n-1;i++){
      for(let a=0;a<states.length;a++) for(let b=0;b<states.length;b++){
        const onPath=(d.best[i]===a && d.best[i+1]===b);
        svg.append("line").attr("x1",colX(i)+16).attr("y1",rowY(a)).attr("x2",colX(i+1)-16).attr("y2",rowY(b))
          .attr("stroke",onPath?C.accent:"#222836").attr("stroke-width",onPath?2.4:1).attr("opacity",onPath?1:0.5);
      }
    }
    // state nodes
    d.words.forEach((w,i)=>{
      states.forEach((s,j)=>{
        const on=d.best[i]===j;
        svg.append("circle").attr("cx",colX(i)).attr("cy",rowY(j)).attr("r",15).attr("fill",on?C.accent:"#1a1f29").attr("fill-opacity",on?0.9:1).attr("stroke",on?"#fff":"#2a2f3a").attr("stroke-width",on?2:1);
        svg.append("text").attr("x",colX(i)).attr("y",rowY(j)+4).attr("text-anchor","middle").attr("font-size",10).attr("fill",on?"#1a1207":C.muted).attr("font-weight",on?700:400).text(s);
      });
    });
    // state labels left
    states.forEach((s,j)=>svg.append("text").attr("x",30).attr("y",rowY(j)+4).attr("fill",C.muted).attr("font-size",10).text(s));
    const path=d.best.map((j,i)=>states[j]);
    d3.select("#hmm-read").html(`Viterbi best path: `+d.words.map((w,i)=>`${w}/<b style="color:${C.accent}">${path[i]}</b>`).join(" "));
  }
  d3.select("#hmm-sel").on("change",draw);
  draw();
})();

/* ───────────────────────── 05 · label bias: local vs global ───────────────────────── */
(function(){
  const C=PC;
  const svg=d3.select("#lb-svg");
  const readEl=d3.select("#lb-read");
  const bA=document.getElementById("lb-memm"), bB=document.getElementById("lb-crf");
  let mode="local";

  // Two branches over three steps. Branch A passes through a state with ONE successor,
  // branch B through a state with THREE. The potentials psi are identical in both modes;
  // only the normalizer changes.
  const obs=["r","o","b"];
  const branches=[
    {name:"A", label:["S1","A2","A3"], psi:[1.0,0.1,1.0], succ:[2,1,1], col:C.bad},
    {name:"B", label:["S1","B2","B3"], psi:[1.0,0.9,1.0], succ:[2,3,1], col:C.good}
  ];
  const B_SIBLINGS=[0.9,0.05,0.05];   // competitors at B's middle state (local normalization only)

  function scores(){
    if(mode==="local"){
      const sib=B_SIBLINGS.reduce((s,v)=>s+v,0);
      const midA=1.0, midB=0.9/sib;               // A's edge is forced to 1 — one successor
      return {A:0.5*midA*1.0, B:0.5*midB*1.0, mid:{A:midA,B:midB}};
    }
    const sA=branches[0].psi.reduce((s,v)=>s*v,1);
    const sB=branches[1].psi.reduce((s,v)=>s*v,1);
    const Z=sA+sB;
    return {A:sA/Z, B:sB/Z, mid:{A:branches[0].psi[1], B:branches[1].psi[1]}};
  }

  function draw(){
    svg.selectAll("*").remove();
    const s=scores();
    const colX=i=>120+i*140, rowY=k=>96+k*76;

    svg.append("text").attr("x",26).attr("y",34).attr("fill",C.muted).attr("font-size",11).text("observed:");
    obs.forEach((o,i)=>svg.append("text").attr("x",colX(i)).attr("y",34).attr("text-anchor","middle")
      .attr("fill",i===1?C.a2:C.ink).attr("font-size",13).attr("font-weight",i===1?700:400)
      .attr("font-family","SF Mono, monospace").text(o));
    svg.append("text").attr("x",colX(1)).attr("y",50).attr("text-anchor","middle").attr("fill",C.a2)
      .attr("font-size",9.5).text("↑ the decisive observation");
    svg.append("text").attr("x",470).attr("y",34).attr("font-size",10).attr("fill",C.muted).text("P(path | x)");

    branches.forEach((br,k)=>{
      const y=rowY(k), other=(br.name==="A"?"B":"A"), win=(s[br.name]>=s[other]);
      for(let i=0;i<obs.length-1;i++){
        const shown = (mode==="local") ? (i===1 ? s.mid[br.name] : 0.5) : br.psi[i+1];
        svg.append("line").attr("x1",colX(i)+18).attr("y1",y).attr("x2",colX(i+1)-18).attr("y2",y)
          .attr("stroke",br.col).attr("stroke-width",1+3.2*shown).attr("opacity",.75);
        svg.append("text").attr("x",(colX(i)+colX(i+1))/2).attr("y",y-10).attr("text-anchor","middle")
          .attr("font-size",10).attr("font-family","SF Mono, monospace").attr("fill",br.col)
          .text((mode==="local"?"p=":"ψ=")+shown.toFixed(2));
      }
      br.label.forEach((lab,i)=>{
        svg.append("circle").attr("cx",colX(i)).attr("cy",y).attr("r",17)
          .attr("fill",br.col).attr("fill-opacity",.18).attr("stroke",br.col).attr("stroke-width",1.4);
        svg.append("text").attr("x",colX(i)).attr("y",y+4).attr("text-anchor","middle")
          .attr("font-size",10.5).attr("fill",C.ink).text(lab);
        if(i===1) svg.append("text").attr("x",colX(i)).attr("y",y+32).attr("text-anchor","middle")
          .attr("font-size",9).attr("fill",C.muted).text(br.succ[1]+" successor"+(br.succ[1]>1?"s":""));
      });
      const bx=470, bw=Math.max(2,130*s[br.name]);
      svg.append("rect").attr("x",bx).attr("y",y-11).attr("width",bw).attr("height",22).attr("rx",4)
        .attr("fill",br.col).attr("fill-opacity",win?.62:.24).attr("stroke",br.col).attr("stroke-width",win?1.6:1);
      svg.append("text").attr("x",bx+bw+8).attr("y",y+5).attr("font-size",11)
        .attr("font-family","SF Mono, monospace").attr("fill",win?br.col:C.muted)
        .attr("font-weight",win?700:400).text(s[br.name].toFixed(3));
      svg.append("text").attr("x",26).attr("y",y+4).attr("font-size",11).attr("fill",br.col).text("path "+br.name);
    });

    readEl.html(mode==="local"
      ? `<b style="color:${C.bad}">Local normalization (MEMM).</b> Path A's middle state has one successor, so its probability is forced to <code>1.00</code> — the observation <code>o</code> is ignored entirely. A wins (<b>${s.A.toFixed(3)}</b> vs ${s.B.toFixed(3)}) on <i>structure</i>, not evidence. That is <b>label bias</b>.`
      : `<b style="color:${C.good}">Global normalization (CRF).</b> The same potentials are multiplied unnormalised, then divided by <code>Z(x)</code> once over whole paths. A's poor match at step 2 (<code>ψ = 0.10</code>) now genuinely costs it, so B wins (<b>${s.B.toFixed(3)}</b> vs ${s.A.toFixed(3)}) — evidence decides.`);

    bA.classList.toggle("ghost", mode!=="local");
    bB.classList.toggle("ghost", mode==="local");
  }
  bA.addEventListener("click",()=>{mode="local";draw();});
  bB.addEventListener("click",()=>{mode="global";draw();});
  draw();
})();
