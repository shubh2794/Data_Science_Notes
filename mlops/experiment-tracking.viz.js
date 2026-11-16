/* experiment-tracking.viz.js — interactive figures for experiment-tracking.html.
   Loaded after data.js / notes.js. */

/* ---- Viz 1: brushable parallel coordinates over a synthetic sweep ---- */
(function(){
  const C={accent:"#a3e635",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#pc-svg"); if(svg.empty()) return;
  const W=680,H=300,m={t:28,r:40,b:22,l:40};
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  const g=svg.append("g").attr("transform",`translate(${m.l},${m.t})`);
  const NR=80;

  const dims=[
    {key:"lr",label:"learning rate",log:true,dom:[1e-5,1e-1]},
    {key:"batch",label:"batch size",log:true,dom:[16,512]},
    {key:"dropout",label:"dropout",log:false,dom:[0,0.6]},
    {key:"layers",label:"layers",log:false,dom:[2,12]},
    {key:"wd",label:"weight decay",log:true,dom:[1e-6,1e-2]},
    {key:"acc",label:"val accuracy",log:false,dom:null}
  ];
  const rnd=d3.randomUniform(0,1), gauss=d3.randomNormal(0,1);
  function logU(a,b){return Math.exp(Math.log(a)+rnd()*(Math.log(b)-Math.log(a)));}

  let runs=[];
  function gen(){
    runs=d3.range(NR).map(i=>{
      const lr=logU(1e-5,1e-1), batch=Math.pow(2,Math.floor(4+rnd()*6)), dropout=rnd()*0.6,
            layers=2+Math.floor(rnd()*11), wd=logU(1e-6,1e-2);
      // synthetic response: strong interior optimum in log-lr, moderate dropout effect, weak others
      const llr=Math.log10(lr);
      let acc=0.90-0.035*(llr+3.2)**2-0.12*(dropout-0.25)**2-0.0015*Math.abs(layers-7)
              -0.004*Math.abs(Math.log10(wd)+4)+0.004*gauss();
      if(llr>-1.6 && rnd()<0.6) acc=0.5+0.1*rnd(); // divergence at high lr
      return {id:i,lr,batch,dropout,layers,wd,acc:Math.max(0.45,acc)};
    });
  }

  const x=d3.scalePoint().domain(dims.map(d=>d.key)).range([0,iw]);
  const ys={};
  const color=d3.scaleSequential(d3.interpolateRgb("#4b5263","#a3e635")).clamp(true);
  const lineG=g.append("g"), axG=g.append("g");
  const brushes={};
  const sel={};

  function setScales(){
    dims.forEach(d=>{
      const dom = d.dom || d3.extent(runs,r=>r[d.key]);
      ys[d.key]=(d.log?d3.scaleLog():d3.scaleLinear()).domain(dom).range([ih,0]);
      if(!d.log) ys[d.key].nice();
    });
    const accs=runs.map(r=>r.acc).sort(d3.ascending);
    color.domain([d3.quantileSorted(accs,0.4),accs[accs.length-1]]);
  }
  const path=r=>d3.line()(dims.map(d=>[x(d.key),ys[d.key](r[d.key])]));

  function active(r){
    return dims.every(d=>{const s=sel[d.key]; if(!s) return true; const v=r[d.key]; return v>=s[0]&&v<=s[1];});
  }

  function render(){
    setScales();
    lineG.selectAll("path").data(runs,r=>r.id).join("path")
      .attr("d",path).attr("fill","none").attr("stroke-width",1.2);
    axG.selectAll("*").remove();
    dims.forEach(d=>{
      const a=axG.append("g").attr("class","axis").attr("transform",`translate(${x(d.key)},0)`);
      const ax=d3.axisLeft(ys[d.key]).ticks(5,d.log?"~e":"~g");
      if(d.key==="batch") ax.tickValues([16,32,64,128,256,512]).tickFormat(d3.format("d"));
      a.call(ax);
      a.selectAll("text").attr("fill",C.muted).attr("font-size",9);
      a.selectAll("path,line").attr("stroke","#3a4150");
      a.append("text").attr("y",-12).attr("text-anchor","middle").attr("fill",d.key==="acc"?C.accent:C.ink).attr("font-size",10.5).text(d.label);
      const br=d3.brushY().extent([[-10,0],[10,ih]]).on("brush end",ev=>{
        if(!ev.selection){delete sel[d.key];}
        else{const [y0,y1]=ev.selection; const s=ys[d.key]; sel[d.key]=[s.invert(y1),s.invert(y0)];}
        restyle();
      });
      const bg=a.append("g").attr("class","brush").call(br);
      brushes[d.key]={br,bg};
      if(sel[d.key]){const s=ys[d.key]; bg.call(br.move,[s(sel[d.key][1]),s(sel[d.key][0])]);}
    });
    restyle();
  }

  function restyle(){
    const n=runs.filter(active).length;
    lineG.selectAll("path")
      .attr("stroke",r=>active(r)?color(r.acc):"#2a2f3a")
      .attr("stroke-opacity",r=>active(r)?0.9:0.35);
    lineG.selectAll("path").filter(r=>active(r)).raise();
    const act=runs.filter(active);
    let txt=`${n} / ${NR} runs selected`;
    if(act.length){
      const best=d3.greatest(act,r=>r.acc);
      const lrE=d3.extent(act,r=>Math.log10(r.lr)), bE=d3.extent(act,r=>Math.log2(r.batch));
      txt+=` · best val acc <b>${best.acc.toFixed(3)}</b> · mean <b>${d3.mean(act,r=>r.acc).toFixed(3)}</b>`+
           `<br>selection spans <b>${(lrE[1]-lrE[0]).toFixed(1)}</b> decades of lr (of 4) and <b>${(bE[1]-bE[0]).toFixed(0)}</b> doublings of batch size (of 5)`+
           `<br><span class="lbl">a parameter whose selected range stays wide after brushing the metric axis barely matters</span>`;
    }
    d3.select("#pc-read").html(txt);
  }

  function clearAll(){Object.keys(sel).forEach(k=>delete sel[k]); render();}
  function top10(){
    Object.keys(sel).forEach(k=>delete sel[k]);
    const q=d3.quantile(runs.map(r=>r.acc).sort(d3.ascending),0.9);
    sel.acc=[q,d3.max(runs,r=>r.acc)];
    render();
  }
  d3.select("#pc-new").on("click",()=>{gen();clearAll();});
  d3.select("#pc-clear").on("click",clearAll);
  d3.select("#pc-top").on("click",top10);
  gen(); render();
})();

/* ---- Viz 2: winner's curse — best of N on a finite validation set ---- */
(function(){
  const C={accent:"#a3e635",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#wc-svg"); if(svg.empty()) return;
  const W=640,H=260,m={t:14,r:16,b:34,l:58};
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  const g=svg.append("g").attr("transform",`translate(${m.l},${m.t})`);
  const P0=0.80, gauss=d3.randomNormal(0,1);

  const x=d3.scaleLinear().range([0,iw]), y=d3.scaleLinear().range([ih,0]);
  const xa=g.append("g").attr("class","axis").attr("transform",`translate(0,${ih})`), ya=g.append("g").attr("class","axis");
  const diag=g.append("line").attr("stroke",C.muted).attr("stroke-dasharray","4 3");
  const ptsG=g.append("g");
  const win=g.append("circle").attr("r",6).attr("fill","none").attr("stroke",C.bad).attr("stroke-width",2);
  const winLbl=g.append("text").attr("fill",C.bad).attr("font-size",10);
  g.append("text").attr("x",iw/2).attr("y",ih+28).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("true accuracy (what the test set would converge to)");
  g.append("text").attr("transform","rotate(-90)").attr("x",-ih/2).attr("y",-48).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("observed validation accuracy");

  let seed=0;
  function trial(N,nv,spread){
    const truth=d3.range(N).map(()=>P0+spread*gauss());
    const obs=truth.map(t=>{const se=Math.sqrt(Math.max(t*(1-t),1e-6)/nv); return t+se*gauss();});
    const iw_=d3.maxIndex(obs);
    return {truth,obs,iw:iw_,optimism:obs[iw_]-truth[iw_],regret:d3.max(truth)-truth[iw_]};
  }
  function draw(){
    const N=+d3.select("#wc-n").property("value");
    const nv=Math.round(Math.pow(10,+d3.select("#wc-v").property("value")));
    const spread=+d3.select("#wc-s").property("value");
    d3.select("#wc-n-v").text(N);
    d3.select("#wc-v-v").text(nv.toLocaleString());
    d3.select("#wc-s-v").text("±"+(spread*100).toFixed(1)+" pts");

    const t=trial(N,nv,spread);
    // average over repeated trials for a stable readout
    let opt=0,reg=0; const R=150;
    for(let k=0;k<R;k++){const tr=trial(N,nv,spread); opt+=tr.optimism; reg+=tr.regret;}
    opt/=R; reg/=R;
    const se=Math.sqrt(P0*(1-P0)/nv);
    const approx=N>1? se*Math.sqrt(2*Math.log(N)) : 0;

    const all=t.truth.concat(t.obs);
    const lo=d3.min(all)-0.004, hi=d3.max(all)+0.004;
    x.domain([lo,hi]); y.domain([lo,hi]);
    xa.call(d3.axisBottom(x).ticks(6).tickFormat(d3.format(".1%")));
    ya.call(d3.axisLeft(y).ticks(6).tickFormat(d3.format(".1%")));
    g.selectAll(".axis text").attr("fill",C.muted).attr("font-size",9);
    g.selectAll(".axis path,.axis line").attr("stroke","#3a4150");
    diag.attr("x1",x(lo)).attr("y1",y(lo)).attr("x2",x(hi)).attr("y2",y(hi));

    const d=t.truth.map((v,i)=>({tv:v,ov:t.obs[i],i}));
    ptsG.selectAll("circle").data(d).join("circle")
      .attr("cx",p=>x(p.tv)).attr("cy",p=>y(p.ov)).attr("r",2.6)
      .attr("fill",C.accent).attr("fill-opacity",0.55);
    const w=d[t.iw];
    win.attr("cx",x(w.tv)).attr("cy",y(w.ov));
    winLbl.attr("x",Math.min(x(w.tv)+9,iw-120)).attr("y",y(w.ov)+4).text("selected: val "+(w.ov*100).toFixed(2)+"%");

    d3.select("#wc-read").html(
      `noise per score σ̂ ≈ <b>${(se*100).toFixed(2)}</b> pts · `+
      `mean optimism of the winner (val − true) over ${R} repeats: <b style="color:${opt>se?C.bad:C.good}">+${(opt*100).toFixed(2)}</b> pts`+
      ` · asymptotic σ̂√(2 ln N): <b>${(approx*100).toFixed(2)}</b> pts`+
      `<br>mean regret (best true − selected true): <b>${(reg*100).toFixed(2)}</b> pts`+
      `<br><span class="lbl">points above the dashed line got lucky on validation; the winner is almost always one of them</span>`
    );
  }
  ["#wc-n","#wc-v","#wc-s"].forEach(id=>d3.select(id).on("input",draw));
  d3.select("#wc-re").on("click",()=>{seed++;draw();});
  draw();
})();
