/* stream-processing.viz.js — interactive figures for stream-processing.html.
   Loaded after data.js / notes.js. Uses d3 from ../vendor. */

/* ---- Viz 1: event time × processing time, windows, watermark, late data ---- */
(function(){
  const C={accent:"#22d3ee",good:"#4ade80",bad:"#f87171",warm:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",panel:"#0e1016",line:"#2a2f3a",ev:"#5b9cff"};
  const svg=d3.select("#sw-svg");
  if(svg.empty()) return;
  // [event time, arrival (processing) time] in seconds — fixed so the figure is reproducible
  const EV=[[1,2],[3,4.5],[4,6],[7,8],[8,15],[11,12],[12,13.5],[14,16],[16,17],[18,27],[21,22],[23,24.5],[24,26],
            [27,28],[29,30],[9,30.5],[33,34],[34,36],[38,39],[26,41],[41,42],[44,45],[36,47],[46,48]]
    .map(([te,ta],i)=>({i,te,ta})).sort((a,b)=>a.ta-b.ta);
  const TMAX=50, PMAX=52, GAP=4;
  const X=d3.scaleLinear().domain([0,TMAX]).range([52,606]);
  const Y=d3.scaleLinear().domain([0,PMAX]).range([18,214]);
  function wmAt(p,d){ // watermark just before processing time p
    const seen=EV.filter(e=>e.ta<p); return seen.length? d3.max(seen,e=>e.te)-d : -Infinity; }
  function windowsOf(te,type){
    if(type==="tumble"){ const a=Math.floor(te/10)*10; return [[a,a+10]]; }
    if(type==="slide"){ const out=[]; for(let k=Math.floor(te/5);k*5>te-10;k--) if(k>=0) out.push([k*5,k*5+10]); return out; }
    return null;
  }
  function draw(){
    svg.selectAll("*").remove();
    const type=d3.select("#sw-type").property("value"), d=+d3.select("#sw-delay").property("value"), pt=+d3.select("#sw-pt").property("value");
    d3.select("#sw-delayv").text(d+"s"); d3.select("#sw-ptv").text(pt+"s");
    const W=wmAt(pt+1e-9,d);
    // classify each arrived event
    EV.forEach(e=>{
      e.arrived=e.ta<=pt;
      const w=wmAt(e.ta,d);
      if(type==="session") e.late = e.te < w;
      else { const ws=windowsOf(e.te,type); e.openWins=ws.filter(([a,b])=>b>w); e.late=e.openWins.length===0; e.partial=!e.late&&e.openWins.length<ws.length; }
    });
    // axes
    svg.append("g").attr("class","axis").attr("transform","translate(0,214)").call(d3.axisBottom(X).ticks(10).tickFormat(t=>t+"s"));
    svg.append("g").attr("class","axis").attr("transform","translate(52,0)").call(d3.axisLeft(Y).ticks(6).tickFormat(t=>t+"s"));
    svg.append("text").attr("x",606).attr("y",208).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",9.5).text("event time →");
    svg.append("text").attr("x",12).attr("y",120).attr("fill",C.muted).attr("font-size",9.5).attr("transform","rotate(-90 12 120)").attr("text-anchor","middle").text("processing time ↓");
    // ideal line: arrival = event time
    svg.append("line").attr("x1",X(0)).attr("y1",Y(0)).attr("x2",X(TMAX)).attr("y2",Y(TMAX)).attr("stroke","#3a4150").attr("stroke-dasharray","2 4");
    svg.append("text").attr("x",X(TMAX)-4).attr("y",Y(TMAX)-6).attr("text-anchor","end").attr("fill","#5f6878").attr("font-size",9).text("zero skew");
    // watermark step curve up to pt
    const pts=[]; const steps=d3.range(0,pt+0.01,0.25);
    steps.forEach(p=>{ const w=wmAt(p+1e-9,d); if(w>-Infinity) pts.push([X(Math.max(0,w)),Y(p)]); });
    if(pts.length) svg.append("path").attr("d",d3.line().curve(d3.curveStepBefore)(pts)).attr("fill","none").attr("stroke",C.warm).attr("stroke-width",2);
    if(W>-Infinity) svg.append("text").attr("x",X(Math.max(0,W))+4).attr("y",Y(pt)-4).attr("fill",C.warm).attr("font-size",10).text("watermark "+Math.max(0,W).toFixed(0)+"s");
    // "now" line
    svg.append("line").attr("x1",X(0)).attr("x2",X(TMAX)).attr("y1",Y(pt)).attr("y2",Y(pt)).attr("stroke",C.accent).attr("stroke-opacity",.5);
    // events
    EV.forEach(e=>{
      if(!e.arrived){ svg.append("circle").attr("cx",X(e.te)).attr("cy",Y(e.ta)).attr("r",3).attr("fill","none").attr("stroke","#3a4150"); return; }
      svg.append("line").attr("x1",X(e.te)).attr("x2",X(e.te)).attr("y1",Y(e.te)).attr("y2",Y(e.ta)).attr("stroke",e.late?C.bad:"#2f3a52").attr("stroke-width",1);
      svg.append("circle").attr("cx",X(e.te)).attr("cy",Y(e.ta)).attr("r",4.5).attr("fill",e.late?C.bad:e.partial?C.warm:C.ev).attr("stroke","#0e1016");
    });
    // windows band
    const by=236;
    let results=[];
    if(type==="session"){
      const acc=EV.filter(e=>e.arrived&&!e.late).sort((a,b)=>a.te-b.te);
      const ses=[]; acc.forEach(e=>{ const s=ses[ses.length-1]; if(s&&e.te-s.last<GAP){s.last=e.te;s.n++;} else ses.push({first:e.te,last:e.te,n:1}); });
      ses.forEach(s=>{ const a=s.first,b=s.last+GAP, fired=W>=b; results.push({a,b,n:s.n,fired});
        svg.append("rect").attr("x",X(a)).attr("y",by).attr("width",X(Math.min(b,TMAX))-X(a)).attr("height",22).attr("rx",4)
          .attr("fill",fired?"rgba(74,222,128,.22)":"rgba(34,211,238,.10)").attr("stroke",fired?C.good:C.accent);
        svg.append("text").attr("x",(X(a)+X(Math.min(b,TMAX)))/2).attr("y",by+15).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",9.5).text(s.n); });
    } else {
      const wins = type==="tumble" ? d3.range(0,TMAX,10).map(a=>[a,a+10]) : d3.range(0,TMAX,5).map(a=>[a,a+10]);
      wins.forEach(([a,b],k)=>{
        const n=EV.filter(e=>e.arrived&&!e.late&&e.te>=a&&e.te<b&&(e.openWins||[]).some(w=>w[0]===a)).length;
        const fired=W>=b; results.push({a,b,n,fired});
        const row=type==="slide"?(((a%10)+10)%10===0?0:1):0, h=type==="slide"?20:24;
        const x0=X(Math.max(0,a)), x1=X(Math.min(b,TMAX));
        svg.append("rect").attr("x",x0+1).attr("y",by+row*(h+3)).attr("width",x1-x0-2).attr("height",h).attr("rx",4)
          .attr("fill",fired?"rgba(74,222,128,.22)":"rgba(34,211,238,.08)").attr("stroke",fired?C.good:"#2e4a55");
        svg.append("text").attr("x",(x0+x1)/2).attr("y",by+row*(h+3)+h/2+4).attr("text-anchor","middle").attr("fill",fired?C.ink:C.muted).attr("font-size",9.5)
          .text(fired?`[${a},${b}) = ${n}`:`[${a},${b}) … ${n}`);
      });
    }
    const late=EV.filter(e=>e.arrived&&e.late);
    const fired=results.filter(r=>r.fired).length;
    d3.select("#sw-read").html(`watermark at t=${pt}s: <b>${W>-Infinity?Math.max(0,W).toFixed(0)+"s":"none yet"}</b> (max event time seen − ${d}s) · windows fired <b>${fired}</b>/${results.length} · `+
      `<span style="color:${C.bad}">late &amp; dropped: ${late.length}</span>`+(late.length?` (event times ${late.map(e=>e.te+"s").join(", ")})`:"")+
      (type==="slide"?` · <span style="color:${C.warm}">amber = counted in some windows, too late for others</span>`:"")+
      (type==="session"?` · sessions close when the watermark passes last event + ${GAP}s`:""));
  }
  d3.selectAll("#sw-type").on("change",draw);
  d3.selectAll("#sw-delay,#sw-pt").on("input",draw);
  draw();
})();

/* ---- Viz 2: Kafka partitions, keys, and a consumer group ---- */
(function(){
  const C={accent:"#22d3ee",good:"#4ade80",bad:"#f87171",warm:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",panel:"#0e1016",line:"#2a2f3a"};
  const PC=["#5b9cff","#ffb454","#c084fc","#4ade80","#f472b6","#22d3ee","#facc15","#fb923c"];
  const svg=d3.select("#kf-svg");
  if(svg.empty()) return;
  const KEYS=d3.range(12).map(i=>"user-"+(101+i*7));
  const RATE=1200, CAP=600, SECS=60;
  const hash=s=>{let h=0;for(const ch of s)h=(Math.imul(h,31)+ch.charCodeAt(0))>>>0;return h;};
  function draw(){
    svg.selectAll("*").remove();
    const P=+d3.select("#kf-p").property("value"), Cn=+d3.select("#kf-c").property("value");
    const hot=+d3.select("#kf-hot").property("value"), slow=d3.select("#kf-slow").property("checked");
    d3.select("#kf-pv").text(P); d3.select("#kf-cv").text(Cn); d3.select("#kf-hotv").text(Math.round(hot*100)+"%");
    const share=KEYS.map((k,i)=>i===0?hot+(1-hot)/KEYS.length:(1-hot)/KEYS.length);
    const kp=KEYS.map(k=>hash(k)%P);
    const prate=d3.range(P).map(p=>RATE*d3.sum(KEYS.map((k,i)=>kp[i]===p?share[i]:0)));
    // assignment: partition p → consumer p mod C (members beyond P sit idle)
    const owner=d3.range(P).map(p=>p%Cn);
    const cons=d3.range(Cn).map(c=>{ const parts=d3.range(P).filter(p=>owner[p]===c); const cap=CAP*(slow&&c===0?0.5:1);
      const load=d3.sum(parts,p=>prate[p]); return {c,parts,cap,load}; });
    const lag=d3.range(P).fill(0);
    cons.forEach(o=>{ const excess=Math.max(0,o.load-o.cap); o.parts.forEach(p=>{ lag[p]=o.load>0?excess*SECS*prate[p]/o.load:0; }); });
    // layout
    const ky=i=>26+i*21, ph=Math.min(44,(250-(P-1)*6)/P), py=p=>22+p*(ph+6);
    const PX0=170, PX1=400, CX=470;
    svg.append("text").attr("x",14).attr("y",14).attr("fill",C.muted).attr("font-size",10).text("producer keys");
    svg.append("text").attr("x",PX0).attr("y",14).attr("fill",C.muted).attr("font-size",10).text("topic partitions (append-only logs)");
    svg.append("text").attr("x",CX).attr("y",14).attr("fill",C.muted).attr("font-size",10).text("consumer group");
    KEYS.forEach((k,i)=>{
      svg.append("path").attr("d",d3.linkHorizontal()({source:[92,ky(i)],target:[PX0-2,py(kp[i])+ph/2]}))
        .attr("fill","none").attr("stroke",PC[kp[i]]).attr("stroke-opacity",.35).attr("stroke-width",1+share[i]*14);
      svg.append("rect").attr("x",14).attr("y",ky(i)-8).attr("width",78).attr("height",16).attr("rx",4).attr("fill",C.panel).attr("stroke",PC[kp[i]]);
      svg.append("text").attr("x",20).attr("y",ky(i)+4).attr("fill",i===0&&hot>0?C.bad:C.ink).attr("font-size",9.5).attr("font-family","monospace").text(k+(i===0&&hot>0?" ★":""));
    });
    const maxRate=Math.max(RATE/2,d3.max(prate));
    d3.range(P).forEach(p=>{
      const y=py(p);
      svg.append("rect").attr("x",PX0).attr("y",y).attr("width",PX1-PX0).attr("height",ph).attr("rx",5).attr("fill",C.panel).attr("stroke",PC[p]);
      // offset tape
      const cells=18, cw=(PX1-PX0-8)/cells;
      const lagCells=Math.min(cells,Math.ceil(lag[p]/2000));
      for(let j=0;j<cells;j++){
        const isLag=j>=cells-lagCells;
        svg.append("rect").attr("x",PX0+4+j*cw).attr("y",y+ph-12).attr("width",cw-2).attr("height",8).attr("rx",1)
          .attr("fill",isLag?C.bad:PC[p]).attr("fill-opacity",isLag?.85:.35);
      }
      svg.append("text").attr("x",PX0+6).attr("y",y+Math.min(14,ph-14)).attr("fill",C.ink).attr("font-size",9.5)
        .text(`p${p} · ${Math.round(prate[p])} msg/s`+(lag[p]>0?` · lag ${d3.format(",")(Math.round(lag[p]))}`:""));
      svg.append("line").attr("x1",PX1).attr("y1",y+ph/2).attr("x2",CX-4).attr("y2",(()=>{ const c=owner[p]; return 22+c*(Math.min(44,(250-(Cn-1)*6)/Cn)+6)+Math.min(44,(250-(Cn-1)*6)/Cn)/2; })())
        .attr("stroke",PC[p]).attr("stroke-opacity",.6);
    });
    const ch=Math.min(44,(250-(Cn-1)*6)/Cn);
    cons.forEach(o=>{
      const y=22+o.c*(ch+6), idle=o.parts.length===0, over=o.load>o.cap;
      svg.append("rect").attr("x",CX).attr("y",y).attr("width",156).attr("height",ch).attr("rx",5)
        .attr("fill",C.panel).attr("stroke",idle?"#3a4150":over?C.bad:C.good).attr("stroke-dasharray",idle?"3 3":null);
      svg.append("text").attr("x",CX+6).attr("y",y+Math.min(14,ch-6)).attr("fill",idle?C.muted:C.ink).attr("font-size",9.5)
        .text(idle?`consumer ${o.c} · idle (no partition)`:`consumer ${o.c} · p${o.parts.join(",p")}`);
      if(!idle && ch>=22){
        const u=Math.min(1.5,o.load/o.cap);
        svg.append("rect").attr("x",CX+6).attr("y",y+ch-9).attr("width",144).attr("height",5).attr("fill","#1b2030");
        svg.append("rect").attr("x",CX+6).attr("y",y+ch-9).attr("width",144*Math.min(1,u)).attr("height",5).attr("fill",over?C.bad:C.good);
      }
    });
    const idle=cons.filter(o=>!o.parts.length).length, totalLag=d3.sum(lag);
    const worst=d3.max(cons,o=>o.load/o.cap);
    d3.select("#kf-read").html(`max parallelism = min(partitions, consumers) = <b>${Math.min(P,Cn)}</b>`+(idle?` · <span style="color:${C.warm}">${idle} idle consumer${idle>1?"s":""}: more members than partitions</span>`:"")+
      ` · busiest consumer at <b>${Math.round(worst*100)}%</b> of capacity · `+(totalLag>0?`<span style="color:${C.bad}">lag after ${SECS}s: ${d3.format(",")(Math.round(totalLag))} records and growing</span>`:`<span style="color:${C.good}">all consumers keep up</span>`)+
      (hot>0.3?` · a hot key cannot be split across partitions without changing the key`:""));
  }
  d3.selectAll("#kf-p,#kf-c,#kf-hot").on("input",draw);
  d3.select("#kf-slow").on("change",draw);
  draw();
})();
