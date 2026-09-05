/* named-entity-recognition.viz.js — extracted from named-entity-recognition.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

const PC={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",bad:"#f87171",blue:"#5b9cff",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};

/* ───────────────────────── 02 · BIO tags & spans ───────────────────────── */
(function(){
  const C={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#ner-svg"),W=640;
  const TYPES={PER:"#5b9cff",LOC:"#4ade80",ORG:"#fb923c",DATE:"#fbbf24"};
  const sents=[
    {toks:["Tim","Cook","visited","Paris","with","Apple","."], tags:["B-PER","I-PER","O","B-LOC","O","B-ORG","O"]},
    {toks:["Ada","Lovelace","was","born","in","London","in","1815","."], tags:["B-PER","I-PER","O","O","O","B-LOC","O","B-DATE","O"]},
    {toks:["The","World","Health","Organization","met","in","Geneva","."], tags:["O","B-ORG","I-ORG","I-ORG","O","O","B-LOC","O"]},
  ];
  function draw(){
    svg.selectAll("*").remove();
    const s=sents[+d3.select("#ner-sel").property("value")];
    let x=20; const y=60;
    s.toks.forEach((t,i)=>{
      const tag=s.tags[i], type=tag==="O"?null:tag.slice(2), col=type?TYPES[type]:"#3a4150";
      const w=Math.max(30,t.length*9+18);
      svg.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",30).attr("rx",6)
        .attr("fill",type?d3.color(col).copy({opacity:0.18}):"#15181f").attr("stroke",col);
      svg.append("text").attr("x",x+w/2).attr("y",y+20).attr("text-anchor","middle").attr("font-size",12).attr("fill",C.ink).text(t);
      svg.append("text").attr("x",x+w/2).attr("y",y-8).attr("text-anchor","middle").attr("font-size",9).attr("fill",col).attr("font-family","SF Mono, monospace").text(tag);
      x+=w+7;
    });
    // legend
    let lx=20; const ly=140;
    Object.entries(TYPES).forEach(([k,c])=>{ svg.append("rect").attr("x",lx).attr("y",ly-9).attr("width",10).attr("height",10).attr("rx",2).attr("fill",c);
      svg.append("text").attr("x",lx+15).attr("y",ly).attr("font-size",10).attr("fill",C.muted).text(k); lx+=70; });
    const ents=[]; let cur=null;
    s.tags.forEach((tg,i)=>{ if(tg.startsWith("B-")){ if(cur)ents.push(cur); cur={type:tg.slice(2),toks:[s.toks[i]]}; } else if(tg.startsWith("I-")&&cur){ cur.toks.push(s.toks[i]); } else { if(cur){ents.push(cur);cur=null;} } });
    if(cur)ents.push(cur);
    d3.select("#ner-read").html(`<b>${ents.length}</b> entities: `+ents.map(e=>`<span style="color:${TYPES[e.type]}">${e.toks.join(" ")} (${e.type})</span>`).join(" · "));
  }
  d3.select("#ner-sel").on("change",draw);
  draw();
})();

/* ───────────────────────── 04 · word labels → sub-word positions ───────────────────────── */
(function(){
  const C=PC;
  const svg=d3.select("#align-svg");
  const readEl=d3.select("#align-read");
  const btn=document.getElementById("align-mode");
  const TYPES={PER:"#5b9cff",LOC:"#4ade80",ORG:"#fb923c"};
  let mode="first";   // "first" = first-subword labelling, "prop" = propagate B-/I- to every piece

  // words with their gold tag and a plausible sub-word split
  const data=[
    {words:[
      {w:"Ada",       pieces:["Ada"],                tag:"B-PER"},
      {w:"Lovelace",  pieces:["Love","##lace"],      tag:"I-PER"},
      {w:"visited",   pieces:["visited"],            tag:"O"},
      {w:"Reykjavik", pieces:["Rey","##kja","##vik"],tag:"B-LOC"}
    ]},
    {words:[
      {w:"Cook",         pieces:["Cook"],                  tag:"B-PER"},
      {w:"joined",       pieces:["joined"],                tag:"O"},
      {w:"Apple",        pieces:["Apple"],                 tag:"B-ORG"},
      {w:"Incorporated", pieces:["Incor","##por","##ated"],tag:"I-ORG"}
    ]}
  ];

  function pieceTag(word, k){
    if(mode==="first") return k===0 ? word.tag : "−100";
    if(word.tag==="O") return "O";
    return k===0 ? word.tag : "I-"+word.tag.slice(2);
  }

  function draw(){
    svg.selectAll("*").remove();
    const d=data[+d3.select("#align-sel").property("value")];
    const yWord=48, yPiece=132;

    svg.append("text").attr("x",14).attr("y",yWord+5).attr("font-size",10).attr("fill",C.muted).text("words");
    svg.append("text").attr("x",14).attr("y",yPiece+5).attr("font-size",10).attr("fill",C.muted).text("pieces");

    let px=74, masked=0, supervised=0;
    const wordBoxes=[];
    d.words.forEach(word=>{
      const start=px, pieceBoxes=[];
      word.pieces.forEach((p,k)=>{
        const tg=pieceTag(word,k);
        const type=(tg==="O"||tg==="−100")?null:tg.slice(2);
        const col=type?TYPES[type]:(tg==="−100"?"#3a4150":"#4a5262");
        const w=Math.max(34,p.length*7.6+14);
        const ignored=(tg==="−100");
        if(ignored) masked++; else supervised++;
        svg.append("rect").attr("x",px).attr("y",yPiece).attr("width",w).attr("height",28).attr("rx",6)
          .attr("fill",ignored?"#13161d":d3.color(col).copy({opacity:.18}))
          .attr("stroke",col).attr("stroke-dasharray",ignored?"3 2":null);
        svg.append("text").attr("x",px+w/2).attr("y",yPiece+19).attr("text-anchor","middle")
          .attr("font-size",11).attr("font-family","SF Mono, monospace")
          .attr("fill",ignored?"#5c6474":C.ink).text(p);
        svg.append("text").attr("x",px+w/2).attr("y",yPiece+46).attr("text-anchor","middle")
          .attr("font-size",9).attr("font-family","SF Mono, monospace").attr("fill",col).text(tg);
        pieceBoxes.push(px+w/2);
        px+=w+6;
      });
      wordBoxes.push({word,start,end:px-6,mids:pieceBoxes});
      px+=10;
    });

    wordBoxes.forEach(b=>{
      const type=b.word.tag==="O"?null:b.word.tag.slice(2);
      const col=type?TYPES[type]:"#3a4150";
      const w=b.end-b.start;
      svg.append("rect").attr("x",b.start).attr("y",yWord-14).attr("width",w).attr("height",28).attr("rx",6)
        .attr("fill",type?d3.color(col).copy({opacity:.16}):"#15181f").attr("stroke",col);
      svg.append("text").attr("x",b.start+w/2).attr("y",yWord+5).attr("text-anchor","middle")
        .attr("font-size",11.5).attr("fill",C.ink).text(b.word.w);
      svg.append("text").attr("x",b.start+w/2).attr("y",yWord-20).attr("text-anchor","middle")
        .attr("font-size",9).attr("font-family","SF Mono, monospace").attr("fill",col).text(b.word.tag);
      // link word → the piece(s) that receive supervision
      b.mids.forEach((mx,k)=>{
        const carries = (mode==="first") ? (k===0) : true;
        svg.append("line").attr("x1",b.start+w/2).attr("y1",yWord+15).attr("x2",mx).attr("y2",yPiece-2)
          .attr("stroke",carries?col:"#252a35").attr("stroke-width",carries?1.6:1)
          .attr("stroke-dasharray",carries?null:"3 3").attr("opacity",carries?.85:.6);
      });
    });

    readEl.html(mode==="first"
      ? `<b>First-subword labelling.</b> <b style="color:${C.a2}">${supervised}</b> positions supervised, <b>${masked}</b> continuation pieces set to the ignore index <code>−100</code> and dropped from the loss. At inference the first piece's tag is the word's tag — no reconciliation needed.`
      : `<b>Propagate-to-all.</b> All <b style="color:${C.a2}">${supervised}</b> positions are supervised (continuations become <code>I-</code>), so there is more training signal — but pieces of one word can now disagree at decode time and must be reconciled (majority vote, or trust the first piece).`);
    btn.textContent = "show: " + (mode==="first" ? "first-subword" : "propagate-to-all");
  }
  d3.select("#align-sel").on("change",draw);
  btn.addEventListener("click",()=>{ mode = (mode==="first") ? "prop" : "first"; draw(); });
  draw();
})();
