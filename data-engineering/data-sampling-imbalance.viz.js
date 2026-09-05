/* data-sampling-imbalance.viz.js — extracted from data-sampling-imbalance.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#22d3ee",maj:"#5b9cff",min:"#fb923c",synth:"#fbbf24",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#imb-svg"),W=640,H=250;
  function draw(){
    svg.selectAll("*").remove();
    const k=d3.select("#imb-sel").property("value");
    let maj=900, min=100, synth=0, syntFlag=false;
    if(k==="over"){ min=900; }
    else if(k==="under"){ maj=100; }
    else if(k==="smote"){ min=100; synth=800; syntFlag=true; }
    // bars
    const x0=70, bw=110, base=150, scale=130/900;
    const bars=[["majority",maj,C.maj], ["minority",min+synth,syntFlag?C.min:C.min]];
    bars.forEach((b,i)=>{
      const x=x0+i*200, h=b[1]*scale;
      svg.append("rect").attr("x",x).attr("y",base-h).attr("width",bw).attr("height",h).attr("rx",4).attr("fill",b[2]).attr("fill-opacity",.85);
      if(syntFlag && i===1){ const sh=synth*scale; svg.append("rect").attr("x",x).attr("y",base-h).attr("width",bw).attr("height",sh).attr("rx",4).attr("fill",C.synth).attr("fill-opacity",.7);
        svg.append("text").attr("x",x+bw/2).attr("y",base-h+14).attr("text-anchor","middle").attr("font-size",9).attr("fill","#1a1207").text("synthetic"); }
      svg.append("text").attr("x",x+bw/2).attr("y",base-h-8).attr("text-anchor","middle").attr("font-size",12).attr("fill",b[2]).text(b[1]);
      svg.append("text").attr("x",x+bw/2).attr("y",base+18).attr("text-anchor","middle").attr("font-size",11).attr("fill",C.muted).text(b[0]);
    });
    svg.append("line").attr("x1",40).attr("y1",base).attr("x2",W-40).attr("y2",base).attr("stroke","#3a4150");
    const ratio=(maj/(min+synth)).toFixed(1);
    const note={none:"99% accuracy by predicting only majority — minority ignored",over:"minority duplicated up to parity",under:"majority thrown away down to parity (data lost)",smote:"minority grown with synthetic points between neighbours"}[k];
    d3.select("#imb-read").html(`ratio majority:minority = <b>${ratio} : 1</b> · ${note}`);
  }
  d3.select("#imb-sel").on("change",draw);
  draw();
})();

/* ── threshold moving under 1% prevalence ── */
(function(){
  const C={accent:"#22d3ee",good:"#4ade80",bad:"#f87171",maj:"#5b9cff",min:"#fb923c",muted:"#9aa3b2"};
  const svg=d3.select("#thr-svg"),W=640,H=270,px=44,py=26,pw=W-88,ph=170;
  // deterministic pseudo-random so the page reads the same every visit
  let seed=20240917; const rnd=()=>{ seed=(seed*1103515245+12345)&0x7fffffff; return seed/0x7fffffff; };
  const gauss=(m,s)=>{ const u=Math.max(1e-9,rnd()), v=rnd(); return m+s*Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };
  const clip=v=>Math.min(0.999,Math.max(0.001,v));
  const NEG=[],POS=[];
  for(let i=0;i<1980;i++) NEG.push(clip(gauss(0.28,0.15)));
  for(let i=0;i<20;i++)   POS.push(clip(gauss(0.66,0.17)));
  const x=d3.scaleLinear().domain([0,1]).range([px,px+pw]);
  const binner=d3.bin().domain([0,1]).thresholds(40);
  const bn=binner(NEG), bp=binner(POS);
  const mn=d3.max(bn,b=>b.length)||1, mp=d3.max(bp,b=>b.length)||1;
  const y=d3.scaleLinear().domain([0,1]).range([py+ph,py]);
  function stats(t){
    const TP=POS.filter(v=>v>=t).length, FN=POS.length-TP;
    const FP=NEG.filter(v=>v>=t).length, TN=NEG.length-FP;
    const prec=TP+FP?TP/(TP+FP):0, rec=TP/(TP+FN||1);
    const f1=prec+rec?2*prec*rec/(prec+rec):0;
    const acc=(TP+TN)/(POS.length+NEG.length), bal=(rec+TN/(TN+FP||1))/2;
    return {TP,FN,FP,TN,prec,rec,f1,acc,bal};
  }
  let bestT=0.5,bestF=-1;
  for(let t=0.02;t<=0.98;t+=0.01){ const s=stats(t); if(s.f1>bestF){bestF=s.f1;bestT=t;} }
  function draw(){
    svg.selectAll("*").remove();
    const t=+d3.select("#thr-t").property("value"), s=stats(t);
    // per-class densities, each normalised to its own peak so 20 positives stay visible
    const area=(bins,max,col,op)=>svg.append("path")
      .attr("d",d3.area().x(d=>x((d.x0+d.x1)/2)).y0(y(0)).y1(d=>y(d.length/max)).curve(d3.curveBasis)(bins))
      .attr("fill",col).attr("fill-opacity",op).attr("stroke",col).attr("stroke-opacity",.8).attr("stroke-width",1.5);
    area(bn,mn,C.maj,.22); area(bp,mp,C.min,.28);
    svg.append("line").attr("x1",px).attr("y1",y(0)).attr("x2",px+pw).attr("y2",y(0)).attr("stroke","#3a4150");
    [0,0.25,0.5,0.75,1].forEach(v=>svg.append("text").attr("x",x(v)).attr("y",y(0)+16).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.muted).text(v.toFixed(2)));
    svg.append("text").attr("x",px).attr("y",py-10).attr("font-size",10).attr("fill",C.maj).text("negatives (1,980)");
    svg.append("text").attr("x",px+pw).attr("y",py-10).attr("text-anchor","end").attr("font-size",10).attr("fill",C.min).text("positives (20)");
    svg.append("line").attr("x1",x(t)).attr("y1",py-2).attr("x2",x(t)).attr("y2",y(0)).attr("stroke",C.good).attr("stroke-width",2);
    svg.append("text").attr("x",x(t)).attr("y",py+ph+34).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.good).text("t = "+t.toFixed(2));
    d3.select("#thr-read").html(
      `TP <b style="color:${C.min}">${s.TP}</b> · FN <b>${s.FN}</b> · FP <b style="color:${C.bad}">${s.FP}</b> · TN <b>${s.TN}</b>` +
      ` &nbsp;|&nbsp; precision <b>${s.prec.toFixed(3)}</b> · recall <b>${s.rec.toFixed(2)}</b> · F1 <b>${s.f1.toFixed(3)}</b>` +
      ` &nbsp;|&nbsp; accuracy <b>${(100*s.acc).toFixed(1)}%</b> (all-negative scores 99.0%) · balanced acc <b>${s.bal.toFixed(3)}</b>`);
  }
  d3.select("#thr-t").on("input",draw);
  d3.select("#thr-f1").on("click",()=>{ d3.select("#thr-t").property("value",bestT.toFixed(2)); draw(); });
  draw();
})();
