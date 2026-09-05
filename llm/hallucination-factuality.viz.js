/* hallucination-factuality.viz.js — extracted from hallucination-factuality.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#hal-svg"),W=640,H=280;
  // claims: confidence + ground-truth correctness (calibrated-ish: higher conf → more likely correct)
  const claims=[0.96,0.93,0.9,0.88,0.84,0.8,0.78,0.72,0.68,0.62,0.58,0.52,0.48,0.42,0.36,0.3,0.26,0.2,0.15,0.1]
    .map((c,i)=>({c, ok: Math.random()<c }));
  function draw(){
    svg.selectAll("*").remove();
    const th=+d3.select("#hal-th").property("value");
    const px=46,py=30,pw=W-92,ph=150;
    const X=v=>px+v*pw;
    // axis
    svg.append("line").attr("x1",px).attr("y1",py+ph).attr("x2",px+pw).attr("y2",py+ph).attr("stroke","#3a4150");
    svg.append("text").attr("x",px+pw/2).attr("y",py+ph+24).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("model confidence →");
    // threshold band
    svg.append("rect").attr("x",X(th)).attr("y",py).attr("width",px+pw-X(th)).attr("height",ph).attr("fill","rgba(45,212,191,.06)");
    svg.append("line").attr("x1",X(th)).attr("y1",py-6).attr("x2",X(th)).attr("y2",py+ph+6).attr("stroke",C.accent).attr("stroke-dasharray","3 3");
    svg.append("text").attr("x",X(th)).attr("y",py-10).attr("text-anchor","middle").attr("fill",C.accent).attr("font-size",10).text("answer ▶");
    let ans=0,corr=0;
    claims.forEach((d,i)=>{
      const answered=d.c>=th; if(answered){ans++; if(d.ok)corr++;}
      const y=py+ph-30 - (i%5)*22;
      svg.append("circle").attr("cx",X(d.c)).attr("cy",y).attr("r",6)
        .attr("fill",d.ok?C.good:C.bad).attr("fill-opacity",answered?0.95:0.18)
        .attr("stroke",answered?"#0f1117":"none").attr("stroke-width",1);
    });
    const cov=ans/claims.length, acc=ans?corr/ans:0;
    // bars
    const by=py+ph+44, bw=200;
    [["coverage",cov,C.a2],["accuracy of answered",acc,C.good]].forEach((r,i)=>{
      const y=by+i*26;
      svg.append("text").attr("x",px).attr("y",y+11).attr("font-size",10.5).attr("fill",C.muted).text(r[0]);
      svg.append("rect").attr("x",px+150).attr("y",y).attr("width",bw).attr("height",14).attr("rx",3).attr("fill","#15181f");
      svg.append("rect").attr("x",px+150).attr("y",y).attr("width",r[1]*bw).attr("height",14).attr("rx",3).attr("fill",r[2]);
      svg.append("text").attr("x",px+150+bw+8).attr("y",y+11).attr("font-size",10).attr("fill",r[2]).text((r[1]*100).toFixed(0)+"%");
    });
    svg.append("circle").attr("cx",px+6).attr("cy",by-12).attr("r",5).attr("fill",C.good); svg.append("text").attr("x",px+16).attr("y",by-8).attr("font-size",9.5).attr("fill",C.muted).text("correct");
    svg.append("circle").attr("cx",px+86).attr("cy",by-12).attr("r",5).attr("fill",C.bad); svg.append("text").attr("x",px+96).attr("y",by-8).attr("font-size",9.5).attr("fill",C.muted).text("wrong/hallucinated");
    d3.select("#hal-read").html(`threshold <b>${th.toFixed(2)}</b> → answers <b>${ans}/${claims.length}</b> (coverage ${(cov*100).toFixed(0)}%), of which <b>${(acc*100).toFixed(0)}%</b> correct · raise it to fabricate less, lower it to answer more`);
  }
  d3.select("#hal-th").on("input",draw);
  draw();
})();
