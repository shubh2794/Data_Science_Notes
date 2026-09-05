/* mixture-of-experts.viz.js — extracted from mixture-of-experts.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const svg=d3.select("#moe-svg"), W=640, H=360, T=12;
  let logits=[];
  function reroute(){
    const N=+d3.select("#moe-n").property("value");
    logits = d3.range(T).map(()=> d3.range(N).map(()=> Math.random()));
    draw();
  }
  function draw(){
    const N=+d3.select("#moe-n").property("value");
    const K=Math.min(+d3.select("#moe-k").property("value"), N);
    svg.selectAll("*").remove();
    const tx=95, ex=470, tTop=54, eTop=44;
    const tGap=(H-80)/(T-1), eGap=(H-70)/(Math.max(N-1,1));
    const color=d3.scaleSequential(d3.interpolateTurbo).domain([-0.6,N-0.4]);
    const ty=i=>tTop+i*tGap, ey=j=>eTop+j*eGap;
    svg.append("text").attr("x",tx).attr("y",26).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("tokens");
    svg.append("text").attr("x",(tx+ex)/2).attr("y",26).attr("text-anchor","middle").attr("fill",C.B).attr("font-size",11).text(`router · top-${K}`);
    svg.append("text").attr("x",ex).attr("y",26).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text(`${N} experts`);
    const load=new Array(N).fill(0);
    logits.forEach((row,i)=>{
      const idx=d3.range(N).sort((a,b)=>row[b]-row[a]).slice(0,K);
      const m=Math.max(...idx.map(j=>row[j]));
      const exps=idx.map(j=>Math.exp(row[j]-m)); const s=d3.sum(exps);
      idx.forEach((j,r)=>{
        const w=exps[r]/s; load[j]++;
        svg.append("path")
          .attr("d",`M${tx+7},${ty(i)} C${(tx+ex)/2},${ty(i)} ${(tx+ex)/2},${ey(j)} ${ex-11},${ey(j)}`)
          .attr("fill","none").attr("stroke",color(j)).attr("stroke-width",0.7+w*2.6).attr("stroke-opacity",.5);
      });
    });
    logits.forEach((_,i)=>{
      svg.append("circle").attr("cx",tx).attr("cy",ty(i)).attr("r",4.5).attr("fill","#2a2f3a").attr("stroke",C.muted).attr("stroke-width",1);
    });
    const maxLoad=Math.max(1,...load);
    for(let j=0;j<N;j++){
      svg.append("rect").attr("x",ex-10).attr("y",ey(j)-9).attr("width",20).attr("height",18).attr("rx",4).attr("fill",color(j)).attr("stroke","#0f1117").attr("stroke-width",1.3);
      svg.append("rect").attr("x",ex+18).attr("y",ey(j)-5).attr("width",load[j]/maxLoad*78).attr("height",10).attr("rx",3).attr("fill",color(j)).attr("fill-opacity",.55);
      svg.append("text").attr("x",ex+18+load[j]/maxLoad*78+6).attr("y",ey(j)+3.5).attr("font-size",9).attr("fill",C.muted).text(load[j]);
    }
    const activePct=Math.round(K/N*100);
    const fold=N/K, foldStr=Number.isInteger(fold)?fold.toFixed(0):fold.toFixed(1);
    const mean=d3.mean(load), sd=Math.sqrt(d3.mean(load.map(l=>(l-mean)**2))), cv=mean?sd/mean:0;
    d3.select("#moe-read").html(`active ≈ <b>${K}/${N} = ${activePct}%</b> of experts &nbsp;·&nbsp; sparsity <b>${foldStr}×</b> &nbsp;·&nbsp; load imbalance (CV) <b>${cv.toFixed(2)}</b> ${cv>0.5?"⚠ skewed — would need balancing":"✓ fairly even"}`);
  }
  d3.select("#moe-n").on("input",reroute);
  d3.select("#moe-k").on("change",draw);
  d3.select("#moe-reroute").on("click",reroute);
  reroute();
})();
