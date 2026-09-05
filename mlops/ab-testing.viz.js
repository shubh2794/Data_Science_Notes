/* ab-testing.viz.js — extracted from ab-testing.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

const C={accent:"#a3e635",blue:"#5b9cff",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};

/* ───────── 04 · A vs B confidence intervals ───────── */
(function(){
  const svg=d3.select("#ab-svg"),W=640,H=220;
  const pA=0.10, pB=0.12;
  function draw(){
    svg.selectAll("*").remove();
    const n=+d3.select("#ab-n").property("value");
    const seA=Math.sqrt(pA*(1-pA)/n), seB=Math.sqrt(pB*(1-pB)/n);
    const ciA=1.96*seA, ciB=1.96*seB;
    const px=60,pw=W-120, lo=0.06, hi=0.16;
    const X=v=>px+(v-lo)/(hi-lo)*pw;
    svg.append("line").attr("x1",px).attr("y1",170).attr("x2",px+pw).attr("y2",170).attr("stroke","#3a4150");
    [0.06,0.08,0.10,0.12,0.14,0.16].forEach(t=>{ svg.append("line").attr("x1",X(t)).attr("y1",60).attr("x2",X(t)).attr("y2",170).attr("stroke","#1e2430"); svg.append("text").attr("x",X(t)).attr("y",188).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.muted).text((t*100).toFixed(0)+"%"); });
    function arm(p,ci,y,col,lbl){
      svg.append("line").attr("x1",X(p-ci)).attr("y1",y).attr("x2",X(p+ci)).attr("y2",y).attr("stroke",col).attr("stroke-width",2);
      [p-ci,p+ci].forEach(e=>svg.append("line").attr("x1",X(e)).attr("y1",y-6).attr("x2",X(e)).attr("y2",y+6).attr("stroke",col).attr("stroke-width",2));
      svg.append("circle").attr("cx",X(p)).attr("cy",y).attr("r",6).attr("fill",col);
      svg.append("text").attr("x",px-12).attr("y",y+4).attr("text-anchor","end").attr("font-size",12).attr("fill",col).attr("font-weight",600).text(lbl);
    }
    arm(pA,ciA,90,C.blue,"A");
    arm(pB,ciB,130,C.accent,"B");
    const se=Math.sqrt(pA*(1-pA)/n + pB*(1-pB)/n);
    const z=(pB-pA)/se;
    const sig=Math.abs(z)>1.96;
    svg.append("text").attr("x",W/2).attr("y",40).attr("text-anchor","middle").attr("fill",sig?C.good:C.bad).attr("font-size",12).attr("font-weight",600)
      .text(sig?"✓ B significantly beats A":"✗ not yet significant — CIs overlap");
    d3.select("#ab-read").html(`n = <b>${n.toLocaleString()}</b>/arm · observed lift <b>+2.0 pts</b> · z = <b>${z.toFixed(2)}</b> · ${sig?'<span style="color:#4ade80">p &lt; 0.05 — ship B</span>':'<span style="color:#f87171">p &gt; 0.05 — need more data</span>'}`);
  }
  d3.select("#ab-n").on("input",draw);
  draw();
})();

/* ───────── 06 · peeking inflates the false-positive rate ───────── */
(function(){
  const svg=d3.select("#peek-svg"),W=640,H=240,m={t:26,r:24,b:44,l:56};
  // approximate two-sided type-I error when stopping at the first of k equally spaced looks (alpha = .05)
  const tbl=[[1,.050],[2,.083],[3,.107],[5,.142],[10,.193],[20,.248],[50,.320],[100,.374]];
  function fpr(k){
    k=Math.max(1,Math.min(100,k));
    for(let i=0;i<tbl.length-1;i++){
      const [k0,v0]=tbl[i],[k1,v1]=tbl[i+1];
      if(k<=k1){ const t=(Math.log(k)-Math.log(k0))/(Math.log(k1)-Math.log(k0)); return v0+t*(v1-v0); }
    }
    return tbl[tbl.length-1][1];
  }
  const x=d3.scaleLog().domain([1,100]).range([m.l,W-m.r]);
  const y=d3.scaleLinear().domain([0,0.42]).range([H-m.b,m.t]);
  // static frame
  const gAx=svg.append("g");
  [0,0.1,0.2,0.3,0.4].forEach(v=>{
    gAx.append("line").attr("x1",m.l).attr("y1",y(v)).attr("x2",W-m.r).attr("y2",y(v)).attr("stroke","#1e2430");
    gAx.append("text").attr("x",m.l-8).attr("y",y(v)+4).attr("text-anchor","end").attr("font-size",9).attr("fill",C.muted).text((v*100).toFixed(0)+"%");
  });
  [1,2,5,10,20,50,100].forEach(v=>{
    gAx.append("line").attr("x1",x(v)).attr("y1",m.t).attr("x2",x(v)).attr("y2",H-m.b).attr("stroke","#171c26");
    gAx.append("text").attr("x",x(v)).attr("y",H-m.b+16).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.muted).text(v);
  });
  gAx.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-8).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.muted).text("number of times you look at the result (log scale)");
  // nominal alpha line = a valid sequential test
  gAx.append("line").attr("x1",m.l).attr("y1",y(0.05)).attr("x2",W-m.r).attr("y2",y(0.05)).attr("stroke",C.good).attr("stroke-width",1.6).attr("stroke-dasharray","5 4");
  gAx.append("text").attr("x",W-m.r-4).attr("y",y(0.05)-7).attr("text-anchor","end").attr("font-size",10).attr("fill",C.good).text("valid sequential test — stays at α = 5%");
  // naive curve
  const pts=[]; for(let i=0;i<=200;i++){ const k=Math.pow(10,i/100); pts.push([k,fpr(k)]); }
  const line=d3.line().x(d=>x(d[0])).y(d=>y(d[1])).curve(d3.curveMonotoneX);
  gAx.append("path").datum(pts).attr("d",line).attr("fill","none").attr("stroke",C.bad).attr("stroke-width",2.2);
  gAx.append("text").attr("x",m.l+10).attr("y",m.t+12).attr("font-size",10).attr("fill",C.bad).text("naive: stop the first time p < 0.05");
  const mk=svg.append("g");
  const dot=mk.append("circle").attr("r",6).attr("fill",C.bad).attr("stroke","#0f1117").attr("stroke-width",2);
  const drop=mk.append("line").attr("stroke",C.bad).attr("stroke-opacity",.45).attr("stroke-dasharray","3 3");
  const kEl=document.getElementById("peek-k"), out=document.getElementById("peek-read");
  function render(){
    const k=+kEl.value, v=fpr(k);
    dot.attr("cx",x(k)).attr("cy",y(v));
    drop.attr("x1",x(k)).attr("y1",y(v)).attr("x2",x(k)).attr("y2",H-m.b);
    const infl=(v/0.05).toFixed(1);
    out.innerHTML=`looks = <b>${k}</b> · true false-positive rate ≈ <b style="color:${v>0.08?C.bad:C.a2}">${(v*100).toFixed(1)}%</b> · that is <b>${infl}×</b> the 5% you think you are running at · ${k===1?"one look at the pre-computed sample size is the only honest fixed-horizon option":"a group-sequential or always-valid test keeps this pinned at 5%"}`;
  }
  kEl.addEventListener("input",render); render();
})();
