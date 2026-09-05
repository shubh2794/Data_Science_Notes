/* federated-learning.viz.js — extracted from federated-learning.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

const C={accent:"#a3e635",blue:"#5b9cff",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};

/* ───────── 03 · FedAvg round ───────── */
(function(){
  const svg=d3.select("#fl-svg"),W=640,H=250;
  let round=0, acc=55;
  function draw(){
    svg.selectAll("*").remove();
    const sx=W/2, sy=46;
    svg.append("rect").attr("x",sx-90).attr("y",26).attr("width",180).attr("height",40).attr("rx",8).attr("fill","rgba(163,230,53,.15)").attr("stroke",C.accent);
    svg.append("text").attr("x",sx).attr("y",51).attr("text-anchor","middle").attr("fill",C.accent).attr("font-size",12).text("Server · global model");
    const cx=[110,270,430,560];
    cx.forEach((x,i)=>{
      const y=180;
      svg.append("rect").attr("x",x-45).attr("y",y).attr("width",90).attr("height",40).attr("rx",8).attr("fill","#15181f").attr("stroke",C.line);
      svg.append("text").attr("x",x).attr("y",y+18).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",10).text("Client "+(i+1));
      svg.append("text").attr("x",x).attr("y",y+32).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",8).text("local data");
      svg.append("line").attr("x1",sx-20).attr("y1",66).attr("x2",x).attr("y2",y).attr("stroke",C.blue).attr("stroke-width",1).attr("stroke-opacity",.5);
      svg.append("line").attr("x1",x+6).attr("y1",y).attr("x2",sx+6).attr("y2",66).attr("stroke",C.good).attr("stroke-width",1).attr("stroke-opacity",.5);
    });
    svg.append("text").attr("x",sx-110).attr("y",120).attr("text-anchor","end").attr("fill",C.blue).attr("font-size",9).text("broadcast ↓");
    svg.append("text").attr("x",sx+120).attr("y",120).attr("fill",C.good).attr("font-size",9).text("↑ updates only");
    svg.append("text").attr("x",60).attr("y",H-12).attr("fill",C.muted).attr("font-size",10).text("global accuracy");
    svg.append("rect").attr("x",170).attr("y",H-24).attr("width",380).attr("height",14).attr("rx",4).attr("fill","#15181f");
    svg.append("rect").attr("x",170).attr("y",H-24).attr("width",380*acc/100).attr("height",14).attr("rx",4).attr("fill",C.accent);
    svg.append("text").attr("x",560).attr("y",H-13).attr("fill",C.accent).attr("font-size",10).text(acc.toFixed(0)+"%");
    d3.select("#fl-read").html(`round <b>${round}</b> · global accuracy <b>${acc.toFixed(0)}%</b> ${round===0?"— click to run federated rounds":acc>90?"— converged, raw data never left any client ✓":"— improving from averaged client updates"}`);
  }
  d3.select("#fl-round").on("click",()=>{ round++; acc=Math.min(94, acc + (94-acc)*0.4 + 2); draw(); });
  d3.select("#fl-reset").on("click",()=>{ round=0; acc=55; draw(); });
  draw();
})();

/* ───────── 04 · local epochs: comms vs drift ───────── */
(function(){
  const svg=d3.select("#ft-svg"),W=640,H=250,m={t:26,r:56,b:44,l:56};
  // illustrative model: progress per round saturates in E; drift caps final accuracy for large E
  const rounds =E=>Math.round(210/(1-Math.exp(-E/3)))/1;
  const finalAcc=E=>94-1.05*Math.pow(Math.max(0,E-4),1.25);
  const MB=5*4/1e6*1e6; // 5M params × 4 bytes
  const bytes =E=>rounds(E)*500*5e6*4*2;     // R × m clients × params × bytes × (down+up)
  const Es=d3.range(1,21);
  const x=d3.scaleLinear().domain([1,20]).range([m.l,W-m.r]);
  const yR=d3.scaleLinear().domain([0,d3.max(Es,rounds)]).range([H-m.b,m.t]);
  const yA=d3.scaleLinear().domain([70,96]).range([H-m.b,m.t]);
  const g=svg.append("g");
  [1,5,10,15,20].forEach(v=>{
    g.append("line").attr("x1",x(v)).attr("y1",m.t).attr("x2",x(v)).attr("y2",H-m.b).attr("stroke","#171c26");
    g.append("text").attr("x",x(v)).attr("y",H-m.b+16).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.muted).text(v);
  });
  g.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-8).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.muted).text("local epochs per round (E)");
  const lineR=d3.line().x(d=>x(d)).y(d=>yR(rounds(d))).curve(d3.curveMonotoneX);
  const lineA=d3.line().x(d=>x(d)).y(d=>yA(finalAcc(d))).curve(d3.curveMonotoneX);
  g.append("path").datum(Es).attr("d",lineR).attr("fill","none").attr("stroke",C.blue).attr("stroke-width",2.2);
  g.append("path").datum(Es).attr("d",lineA).attr("fill","none").attr("stroke",C.a2).attr("stroke-width",2.2);
  g.append("text").attr("x",m.l+8).attr("y",m.t+12).attr("font-size",10).attr("fill",C.blue).text("rounds to target (lower = cheaper)");
  g.append("text").attr("x",W-m.r-8).attr("y",m.t+12).attr("text-anchor","end").attr("font-size",10).attr("fill",C.a2).text("final accuracy under skew");
  const dotR=g.append("circle").attr("r",5.5).attr("fill",C.blue).attr("stroke","#0f1117").attr("stroke-width",2);
  const dotA=g.append("circle").attr("r",5.5).attr("fill",C.a2).attr("stroke","#0f1117").attr("stroke-width",2);
  const rule=g.append("line").attr("stroke",C.muted).attr("stroke-opacity",.4).attr("stroke-dasharray","3 3").attr("y1",m.t).attr("y2",H-m.b);
  const el=document.getElementById("ft-e"), out=document.getElementById("ft-read");
  function render(){
    const E=+el.value, R=rounds(E), A=finalAcc(E), TB=bytes(E)/1e12;
    dotR.attr("cx",x(E)).attr("cy",yR(R)); dotA.attr("cx",x(E)).attr("cy",yA(A));
    rule.attr("x1",x(E)).attr("x2",x(E));
    const verdict = E<=2 ? "communication-bound: many cheap rounds, little drift"
                  : E<=6 ? "near the sweet spot for moderately skewed clients"
                         : "drift-bound: local models diverge faster than averaging can fix";
    out.innerHTML=`E = <b>${E}</b> · rounds ≈ <b>${R}</b> · aggregate traffic ≈ <b>${TB.toFixed(1)} TB</b> · final accuracy ≈ <b style="color:${A>90?C.good:C.bad}">${A.toFixed(1)}%</b> — ${verdict}`;
  }
  el.addEventListener("input",render); render();
})();
