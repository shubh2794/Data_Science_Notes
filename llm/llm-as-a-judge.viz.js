/* llm-as-a-judge.viz.js — extracted from llm-as-a-judge.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#jd-svg"),W=640,H=250;
  const qa=0.62, qb=0.66;   // true quality; B is actually slightly better
  let firstIsA=true;
  function draw(){
    svg.selectAll("*").remove();
    const bias=+d3.select("#jd-bias").property("value");
    // judge score = true quality + bias toward whichever is shown first
    const sa=qa + (firstIsA?bias:0);
    const sb=qb + (firstIsA?0:bias);
    const order = firstIsA? [["A",qa,sa,true],["B",qb,sb,false]] : [["B",qb,sb,false],["A",qa,sa,true]];
    svg.append("text").attr("x",24).attr("y",26).attr("fill",C.muted).attr("font-size",11).text("shown to judge in this order ↓   (B is truly a bit better)");
    const top=46, bh=150, slot=180, scale=bh;
    order.forEach((o,i)=>{
      const x=60+i*220;
      // true quality (faint) and judged score (solid)
      svg.append("rect").attr("x",x).attr("y",top+bh-o[1]*scale).attr("width",60).attr("height",o[1]*scale).attr("rx",4).attr("fill","#3a4150").attr("fill-opacity",.5);
      svg.append("rect").attr("x",x+66).attr("y",top+bh-o[2]*scale).attr("width",60).attr("height",o[2]*scale).attr("rx",4).attr("fill",o[3]?"#5b9cff":"#c084fc");
      svg.append("text").attr("x",x+63).attr("y",top+bh+18).attr("text-anchor","middle").attr("fill",o[3]?"#5b9cff":"#c084fc").attr("font-size",12).attr("font-weight",600).text("Answer "+o[0]+(i===0?"  (1st)":"  (2nd)"));
      svg.append("text").attr("x",x+30).attr("y",top+bh-o[1]*scale-6).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.muted).text("true");
      svg.append("text").attr("x",x+96).attr("y",top+bh-o[2]*scale-6).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.ink).text("judged");
    });
    svg.append("line").attr("x1",40).attr("y1",top+bh).attr("x2",W-40).attr("y2",top+bh).attr("stroke","#3a4150");
    const winner = sa>sb?"A":"B";
    const correct = winner==="B"; // B truly better
    d3.select("#jd-read").html(`judge picks <b style="color:${winner==='A'?'#5b9cff':'#c084fc'}">Answer ${winner}</b> — ${correct?'<span style="color:#4ade80">✓ matches true quality</span>':'<span style="color:#f87171">✗ wrong: position bias flipped it</span>'} · swap the order or lower the bias`);
  }
  d3.select("#jd-bias").on("input",draw);
  d3.select("#jd-swap").on("click",()=>{ firstIsA=!firstIsA; draw(); });
  draw();
})();
