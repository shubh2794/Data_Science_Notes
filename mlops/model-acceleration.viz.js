/* model-acceleration.viz.js — extracted from model-acceleration.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#a3e635",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#acc-svg"),W=640,H=220;
  function draw(){
    svg.selectAll("*").remove();
    const q=d3.select("#acc-q").property("checked"), p=d3.select("#acc-p").property("checked"), d=d3.select("#acc-d").property("checked");
    let size=100, lat=100, acc=100;
    if(q){ size*=0.28; lat*=0.6; acc-=1.0; }
    if(p){ size*=0.6; lat*=0.7; acc-=2.0; }
    if(d){ size*=0.45; lat*=0.55; acc-=2.5; }
    const metrics=[["model size",size,C.accent],["latency",lat,C.a2],["accuracy",acc,C.good]];
    const x0=150, bw=W-x0-70, top=30, rowH=52;
    metrics.forEach((m,i)=>{
      const y=top+i*rowH;
      svg.append("text").attr("x",x0-12).attr("y",y+16).attr("text-anchor","end").attr("font-size",11).attr("fill",C.muted).text(m[0]);
      svg.append("rect").attr("x",x0).attr("y",y).attr("width",bw).attr("height",22).attr("rx",4).attr("fill","#15181f");
      const w = (m[0]==="accuracy"? m[1]/100 : m[1]/100)*bw;
      svg.append("rect").attr("x",x0).attr("y",y).attr("width",Math.max(2,w)).attr("height",22).attr("rx",4).attr("fill",m[2]).attr("fill-opacity",.85);
      svg.append("text").attr("x",x0+Math.max(2,w)+8).attr("y",y+16).attr("font-size",11).attr("fill",m[2]).text(m[0]==="accuracy"?m[1].toFixed(1)+"%":Math.round(m[1])+"%");
    });
    const sp=(100/lat).toFixed(1), sh=(100/size).toFixed(1);
    d3.select("#acc-read").html(`<b>${sh}×</b> smaller · <b>${sp}×</b> faster · accuracy <b>${acc.toFixed(1)}%</b> ${(!q&&!p&&!d)?"— full model baseline":""}`);
  }
  ["#acc-q","#acc-p","#acc-d"].forEach(s=>d3.select(s).on("change",draw));
  draw();
})();
