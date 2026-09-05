/* policy-preference-optimization.viz.js — extracted from policy-preference-optimization.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#ppo-svg"),W=640,H=300,beta=1.0;
  const sig=x=>1/(1+Math.exp(-x));
  function draw(){
    svg.selectAll("*").remove();
    const m=+d3.select("#ppo-m").property("value");
    // left: sigmoid curve
    const px=46,py=34,pw=320,ph=200, xmin=-5,xmax=5;
    const X=v=>px+(v-xmin)/(xmax-xmin)*pw, Y=p=>py+ph-p*ph;
    svg.append("text").attr("x",px).attr("y",py-12).attr("fill",C.muted).attr("font-size",11).text("P(prefer chosen) = σ(margin)");
    [0,0.5,1].forEach(g=>{ svg.append("line").attr("x1",px).attr("y1",Y(g)).attr("x2",px+pw).attr("y2",Y(g)).attr("stroke","#222836").attr("stroke-dasharray","2 5");
      svg.append("text").attr("x",px-8).attr("y",Y(g)+3).attr("text-anchor","end").attr("font-size",9).attr("fill",C.muted).text(g.toFixed(1)); });
    svg.append("line").attr("x1",X(0)).attr("y1",py).attr("x2",X(0)).attr("y2",py+ph).attr("stroke","#2a2f3a");
    const xs=d3.range(xmin,xmax+0.01,0.2);
    svg.append("path").attr("d",d3.line().x(d=>X(d)).y(d=>Y(sig(d)))(xs)).attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2);
    svg.append("line").attr("x1",X(m)).attr("y1",py).attr("x2",X(m)).attr("y2",py+ph).attr("stroke",C.a2).attr("stroke-dasharray","3 3");
    svg.append("circle").attr("cx",X(m)).attr("cy",Y(sig(m))).attr("r",5).attr("fill",C.a2);
    svg.append("text").attr("x",px+pw/2).attr("y",py+ph+18).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("margin  r_chosen − r_rejected");
    // right: reward bars
    const bx=430, mid=140, scale=26, bw=46;
    svg.append("text").attr("x",bx+bw).attr("y",30).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("rewards");
    svg.append("line").attr("x1",bx-10).attr("y1",mid).attr("x2",bx+2*bw+30).attr("y2",mid).attr("stroke","#3a4150");
    const rc=m/2, rr=-m/2;
    // chosen
    svg.append("rect").attr("x",bx).attr("y",rc>=0?mid-rc*scale:mid).attr("width",bw).attr("height",Math.abs(rc*scale)).attr("fill",C.good).attr("fill-opacity",.8);
    svg.append("text").attr("x",bx+bw/2).attr("y",mid+ (rc>=0?16:-6)).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.good).text("chosen");
    // rejected
    const rxx=bx+bw+30;
    svg.append("rect").attr("x",rxx).attr("y",rr>=0?mid-rr*scale:mid).attr("width",bw).attr("height",Math.abs(rr*scale)).attr("fill",C.bad).attr("fill-opacity",.8);
    svg.append("text").attr("x",rxx+bw/2).attr("y",mid+(rr>=0?16:-6)).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.bad).text("rejected");
    const p=sig(beta*m), loss=-Math.log(Math.max(1e-9,p));
    d3.select("#ppo-read").html(`margin <b>${m.toFixed(2)}</b> → P(prefer chosen) = <b>${(p*100).toFixed(1)}%</b> · DPO/RM loss −log σ(βm) = <b>${loss.toFixed(3)}</b> ${m>2.5?"· chosen clearly preferred":m<0?"· ⚠ rejected currently winning":""}`);
  }
  d3.select("#ppo-m").on("input",draw);
  d3.select("#ppo-step").on("click",()=>{ const v=Math.min(4,+d3.select("#ppo-m").property("value")+0.4); d3.select("#ppo-m").property("value",v.toFixed(1)); draw(); });
  d3.select("#ppo-reset").on("click",()=>{ d3.select("#ppo-m").property("value",0.6); draw(); });
  draw();
})();
