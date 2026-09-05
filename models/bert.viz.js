/* bert.viz.js — extracted from bert.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ───────────────────────── 03 · masked language modelling ───────────────────────── */
(function(){
  const C={accent:"#c084fc",good:"#4ade80",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#bert-svg"),W=640,H=280;
  const sents=[
    {toks:["The","[MASK]","sat","on","the","mat"], mi:1, preds:[["cat",0.41],["dog",0.17],["man",0.11],["boy",0.08],["girl",0.06]]},
    {toks:["Paris","is","the","[MASK]","of","France"], mi:3, preds:[["capital",0.79],["heart",0.06],["city",0.05],["center",0.04],["pride",0.02]]},
    {toks:["I","love","[MASK]","learning","models"], mi:2, preds:[["machine",0.55],["deep",0.23],["transfer",0.07],["reinforcement",0.05],["statistical",0.04]]},
  ];
  function draw(){
    svg.selectAll("*").remove();
    const s=sents[+d3.select("#bert-sel").property("value")];
    // sentence
    let x=24; const y=46;
    s.toks.forEach((t,i)=>{
      const masked=i===s.mi, w=Math.max(40,t.length*9+18);
      svg.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",32).attr("rx",6)
        .attr("fill",masked?"rgba(192,132,252,.2)":"#15181f").attr("stroke",masked?C.accent:"#2a2f3a").attr("stroke-width",masked?2:1);
      svg.append("text").attr("x",x+w/2).attr("y",y+21).attr("text-anchor","middle").attr("font-size",12).attr("fill",masked?C.accent:C.ink).attr("font-family","SF Mono,monospace").text(t);
      // bidirectional arrows into the mask
      if(!masked){ const mx=24+s.toks.slice(0,s.mi).reduce((a,t)=>a+Math.max(40,t.length*9+18)+6,0)+Math.max(40,s.toks[s.mi].length*9+18)/2;
        svg.append("line").attr("x1",x+w/2).attr("y1",y+34).attr("x2",mx).attr("y2",y+44).attr("stroke",C.accent).attr("stroke-opacity",.25); }
      x+=w+6;
    });
    svg.append("text").attr("x",W/2).attr("y",96).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("↑ context flows in from BOTH sides of the mask");
    // predictions
    const top=120, bw=300, x0=170;
    svg.append("text").attr("x",x0).attr("y",top-8).attr("fill",C.muted).attr("font-size",11).text("predicted fillers");
    s.preds.forEach((p,i)=>{
      const yy=top+i*28;
      svg.append("text").attr("x",x0-12).attr("y",yy+13).attr("text-anchor","end").attr("font-size",11).attr("fill",i===0?C.good:C.ink).attr("font-family","SF Mono,monospace").text(p[0]);
      svg.append("rect").attr("x",x0).attr("y",yy).attr("width",bw).attr("height",16).attr("rx",4).attr("fill","#15181f");
      svg.append("rect").attr("x",x0).attr("y",yy).attr("width",p[1]*bw).attr("height",16).attr("rx",4).attr("fill",i===0?C.good:C.accent).attr("fill-opacity",i===0?0.9:0.55);
      svg.append("text").attr("x",x0+p[1]*bw+8).attr("y",yy+13).attr("font-size",10).attr("fill",i===0?C.good:C.muted).text((p[1]*100).toFixed(0)+"%");
    });
    d3.select("#bert-read").html(`top prediction for [MASK]: <b style="color:${C.good}">${s.preds[0][0]}</b> (${(s.preds[0][1]*100).toFixed(0)}%) — chosen using words on <b>both</b> sides`);
  }
  d3.select("#bert-sel").on("change",draw);
  draw();
})();

/* ───────────────────────── 05 · input = token + segment + position ───────────────────────── */
(function(){
  const C={tok:"#c084fc",seg:"#ffb454",pos:"#2dd4bf",sum:"#4ade80",muted:"#9aa3b2",ink:"#e6e9ef",panel:"#15181f",line:"#2a2f3a"};
  const svg=d3.select("#bert-input-svg");
  const ex=[
    {toks:["[CLS]","the","cat","sat","[SEP]"], seg:[0,0,0,0,0]},
    {toks:["[CLS]","i","like","cats","[SEP]","me","too","[SEP]"], seg:[0,0,0,0,0,1,1,1]}
  ];
  const x0=118, cw=58, gap=4;
  const rows=[
    {key:"tok", y:104, col:C.tok,  lab:"E_tok  token"},
    {key:"seg", y:146, col:C.seg,  lab:"E_seg  segment"},
    {key:"pos", y:188, col:C.pos,  lab:"E_pos  position"},
    {key:"sum", y:240, col:C.sum,  lab:"h⁰  input vector"}
  ];
  // deterministic pseudo-random stripe pattern so each cell looks like a distinct vector
  function stripes(seed,n){ const out=[]; let s=seed*9301+49297; for(let i=0;i<n;i++){ s=(s*9301+49297)%233280; out.push(0.25+0.7*(s/233280)); } return out; }
  let hover=-1;

  function draw(){
    svg.selectAll("*").remove();
    const e=ex[+d3.select("#bert-input-sel").property("value")];
    const n=e.toks.length;

    // row labels
    rows.forEach(r=>{
      svg.append("text").attr("x",x0-12).attr("y",r.y+18).attr("text-anchor","end")
        .attr("font-size",10.5).attr("fill",r.key==="sum"?C.sum:C.muted)
        .attr("font-family","SF Mono,monospace").text(r.lab);
    });
    // plus / equals column markers
    [ [124,"+"],[166,"+"],[214,"="] ].forEach(([yy,ch])=>{
      svg.append("text").attr("x",x0-98).attr("y",yy+18).attr("font-size",13).attr("fill",C.muted).text("");
    });

    // token strip
    svg.append("text").attr("x",x0-12).attr("y",62).attr("text-anchor","end").attr("font-size",10.5)
      .attr("fill",C.muted).attr("font-family","SF Mono,monospace").text("tokens");
    e.toks.forEach((t,i)=>{
      const x=x0+i*(cw+gap), on=(hover===i);
      svg.append("rect").attr("x",x).attr("y",44).attr("width",cw).attr("height",26).attr("rx",5)
        .attr("fill",on?"rgba(192,132,252,.22)":C.panel).attr("stroke",on?C.tok:C.line);
      svg.append("text").attr("x",x+cw/2).attr("y",62).attr("text-anchor","middle").attr("font-size",10.5)
        .attr("fill",on?C.tok:C.ink).attr("font-family","SF Mono,monospace").text(t);
    });

    // embedding rows
    e.toks.forEach((t,i)=>{
      const x=x0+i*(cw+gap), on=(hover===i);
      rows.forEach(r=>{
        const seed = r.key==="tok" ? (t.length*7+i) : r.key==="seg" ? (e.seg[i]*13+3) : r.key==="pos" ? (i*5+11) : (t.length*7+i)+(e.seg[i]*13+3)+(i*5+11);
        const vals=stripes(seed,6);
        svg.append("rect").attr("x",x).attr("y",r.y).attr("width",cw).attr("height",30).attr("rx",5)
          .attr("fill",C.panel).attr("stroke",on?r.col:C.line).attr("stroke-width",on?1.8:1);
        vals.forEach((v,k)=>{
          svg.append("rect").attr("x",x+5+k*((cw-10)/6)).attr("y",r.y+6)
            .attr("width",(cw-10)/6-1.5).attr("height",18).attr("rx",2)
            .attr("fill",r.col).attr("fill-opacity",on?v:v*0.45);
        });
      });
      // caption under each column: segment id / position id
      svg.append("text").attr("x",x+cw/2).attr("y",rows[1].y+43).attr("text-anchor","middle")
        .attr("font-size",9.5).attr("fill",on?C.seg:C.muted).attr("font-family","SF Mono,monospace").text(e.seg[i]===0?"A":"B");
      svg.append("text").attr("x",x+cw/2).attr("y",rows[2].y+43).attr("text-anchor","middle")
        .attr("font-size",9.5).attr("fill",on?C.pos:C.muted).attr("font-family","SF Mono,monospace").text(i);
    });

    // + and = signs at the left margin between rows
    svg.append("text").attr("x",x0-6).attr("y",rows[1].y-6).attr("text-anchor","end").attr("font-size",13).attr("fill",C.muted).text("+");
    svg.append("text").attr("x",x0-6).attr("y",rows[2].y-6).attr("text-anchor","end").attr("font-size",13).attr("fill",C.muted).text("+");
    svg.append("text").attr("x",x0-6).attr("y",rows[3].y-8).attr("text-anchor","end").attr("font-size",13).attr("fill",C.sum).text("=");

    // hover targets across the whole column
    e.toks.forEach((t,i)=>{
      const x=x0+i*(cw+gap);
      svg.append("rect").attr("x",x).attr("y",40).attr("width",cw).attr("height",236)
        .attr("fill","transparent").style("cursor","pointer")
        .on("mouseenter",()=>{hover=i;draw();})
        .on("mouseleave",()=>{hover=-1;draw();});
    });

    svg.append("text").attr("x",x0).attr("y",22).attr("font-size",10).attr("fill",C.muted)
      .text("every position is the sum of three learned table lookups, then LayerNorm");

    const msg = hover<0
      ? `hover a column — segment ids: <b style="color:${C.seg}">A</b> before the first [SEP], <b style="color:${C.seg}">B</b> after it`
      : `<b style="color:${C.tok}">${e.toks[hover]}</b> at position <b style="color:${C.pos}">${hover}</b> in segment <b style="color:${C.seg}">${e.seg[hover]===0?"A":"B"}</b> → h⁰ = E_tok[${e.toks[hover]}] + E_seg[${e.seg[hover]}] + E_pos[${hover}]`;
    d3.select("#bert-input-read").html(msg);
  }
  d3.select("#bert-input-sel").on("change",()=>{hover=-1;draw();});
  draw();
})();
