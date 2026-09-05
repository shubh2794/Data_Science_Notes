/* learning-paradigms.viz.js — extracted from learning-paradigms.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#ffb454",blue:"#5b9cff",good:"#4ade80",bad:"#f87171",violet:"#c084fc",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#lp-svg"),W=640,H=230;
  function dot(x,y,c){ svg.append("circle").attr("cx",x).attr("cy",y).attr("r",6).attr("fill",c); }
  function lbl(x,y,t,c,s){ svg.append("text").attr("x",x).attr("y",y).attr("text-anchor","middle").attr("fill",c||C.muted).attr("font-size",s||11).text(t); }
  const draws={
    sup(){
      lbl(W/2,28,"labeled examples → learn X → Y",C.ink,12);
      const xs=[120,180,240,300]; xs.forEach((x,i)=>{ dot(x,90,i<2?C.blue:C.bad); lbl(x,118,i<2?"class A":"class B",i<2?C.blue:C.bad,9); });
      svg.append("text").attr("x",420).attr("y",95).attr("font-size",13).attr("fill",C.muted).text("→ predict label for new X");
      return "Supervised: every example carries a target label; the model learns the input→output mapping.";
    },
    unsup(){
      lbl(W/2,28,"no labels → find structure",C.ink,12);
      const pts=[[160,80],[180,100],[150,110],[200,90],[420,150],[450,130],[440,160],[470,145]];
      pts.forEach(p=>dot(p[0],p[1],C.muted));
      svg.append("ellipse").attr("cx",172).attr("cy",95).attr("rx",46).attr("ry",36).attr("fill","none").attr("stroke",C.good).attr("stroke-dasharray","4 3");
      svg.append("ellipse").attr("cx",452).attr("cy",145).attr("rx",42).attr("ry",32).attr("fill","none").attr("stroke",C.good).attr("stroke-dasharray","4 3");
      lbl(172,150,"cluster 1",C.good,9); lbl(452,195,"cluster 2",C.good,9);
      return "Unsupervised: no labels — discover groups (clusters), low-dimensional structure, or density.";
    },
    self(){
      lbl(W/2,28,"hide part of the input → predict it",C.ink,12);
      const toks=["the","cat","____","on","the","mat"]; let x=110;
      toks.forEach((t,i)=>{ const masked=t==="____"; svg.append("rect").attr("x",x).attr("y",80).attr("width",64).attr("height",30).attr("rx",5).attr("fill",masked?"rgba(192,132,252,.2)":"#15181f").attr("stroke",masked?C.violet:"#2a2f3a"); svg.append("text").attr("x",x+32).attr("y",100).attr("text-anchor","middle").attr("font-size",11).attr("fill",masked?C.violet:C.ink).attr("font-family","SF Mono,monospace").text(masked?"?":t); x+=72; });
      lbl(W/2,150,"label = the real hidden token ('sat') — free supervision",C.violet,10);
      return "Self-supervised: the data supplies its own labels by masking/predicting parts — powers LLM pretraining.";
    },
    rl(){
      lbl(W/2,28,"act → observe reward → improve",C.ink,12);
      svg.append("rect").attr("x",110).attr("y",80).attr("width",110).attr("height",40).attr("rx",8).attr("fill","rgba(255,180,84,.15)").attr("stroke",C.accent);
      lbl(165,105,"Agent",C.accent,12);
      svg.append("rect").attr("x",420).attr("y",80).attr("width",120).attr("height",40).attr("rx",8).attr("fill","#15181f").attr("stroke",C.line);
      lbl(480,105,"Environment",C.ink,11);
      svg.append("text").attr("x",315).attr("y",95).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.muted).text("action →");
      svg.append("text").attr("x",315).attr("y",150).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.good).text("← state, reward");
      svg.append("line").attr("x1",220).attr("y1",92).attr("x2",420).attr("y2",92).attr("stroke",C.muted);
      svg.append("line").attr("x1",420).attr("y1",112).attr("x2",220).attr("y2",112).attr("stroke",C.good);
      return "Reinforcement: no labels — an agent learns from reward feedback to maximize long-term return.";
    },
  };
  function draw(){ svg.selectAll("*").remove(); const k=d3.select("#lp-sel").property("value"); const note=draws[k](); document.getElementById("lp-read").innerHTML=note; }
  d3.select("#lp-sel").on("change",draw);
  draw();
})();
