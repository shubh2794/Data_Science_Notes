/* model-deployment.viz.js — interactive figures for model-deployment.html.
   Loaded after data.js / notes.js. */

/* ---- Viz 1: discrete-event simulation of dynamic batching across replicas ---- */
(function(){
  const C={accent:"#a3e635",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#bt-svg"); if(svg.empty()) return;
  const W=640,H=230,m={t:16,r:16,b:34,l:46};
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  const g=svg.append("g").attr("transform",`translate(${m.l},${m.t})`);
  const S0=12, S1=1.2;          // ms
  const NREQ=4000;

  // deterministic PRNG so the histogram does not flicker between slider ticks
  function mulberry(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}

  function simulate(lambda,R,B,TO){
    const rng=mulberry(12345);
    // generate Poisson arrivals (ms) and route each to a uniformly random replica
    const per=d3.range(R).map(()=>[]);
    let t=0;
    for(let i=0;i<NREQ;i++){t+=-Math.log(1-rng())/(lambda/1000); per[Math.floor(rng()*R)].push(t);}
    const lat=[]; let busy=0, batches=0, items=0, horizon=t;
    per.forEach(arr=>{
      let free=0, idx=0;
      while(idx<arr.length){
        const t0=Math.max(free,arr[idx]);
        const deadline=Math.max(t0,arr[idx]+TO);
        let j=idx;
        while(j<arr.length && j-idx<B && arr[j]<=deadline) j++;
        let dispatch = (j-idx===B) ? Math.max(t0,arr[j-1]) : deadline;
        if(j===idx){j=idx+1; dispatch=t0;}
        const b=j-idx, sv=S0+S1*b, fin=dispatch+sv;
        for(let k=idx;k<j;k++) lat.push(fin-arr[k]);
        busy+=sv; batches++; items+=b; free=fin; idx=j;
        horizon=Math.max(horizon,fin);
      }
    });
    lat.sort(d3.ascending);
    return {lat,p50:d3.quantileSorted(lat,0.5),p99:d3.quantileSorted(lat,0.99),
            util:busy/(R*horizon),meanB:items/batches};
  }

  const x=d3.scaleLinear().range([0,iw]), y=d3.scaleLinear().range([ih,0]);
  const xa=g.append("g").attr("class","axis").attr("transform",`translate(0,${ih})`);
  const ya=g.append("g").attr("class","axis");
  const bars=g.append("g");
  const mk=(col,dash)=>g.append("line").attr("stroke",col).attr("stroke-width",1.6).attr("stroke-dasharray",dash).attr("y1",0).attr("y2",ih);
  const l50=mk(C.good,"none"), l99=mk(C.a2,"none"), lslo=mk(C.bad,"5 4");
  const t50=g.append("text").attr("fill",C.good).attr("font-size",10).attr("y",10);
  const t99=g.append("text").attr("fill",C.a2).attr("font-size",10).attr("y",22);
  const tslo=g.append("text").attr("fill",C.bad).attr("font-size",10).attr("y",34);
  g.append("text").attr("x",iw/2).attr("y",ih+28).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("request latency (ms) — queue wait + batch compute");

  function draw(){
    const L=+d3.select("#bt-l").property("value"), R=+d3.select("#bt-r").property("value"),
          B=+d3.select("#bt-b").property("value"), TO=+d3.select("#bt-t").property("value"),
          SLO=+d3.select("#bt-s").property("value");
    d3.select("#bt-l-v").text(L); d3.select("#bt-r-v").text(R); d3.select("#bt-b-v").text(B);
    d3.select("#bt-t-v").text(TO); d3.select("#bt-s-v").text(SLO);

    const cap=1000*B/(S0+S1*B);           // req/s per replica at full batches
    const unstable = L/R >= cap;
    const s=simulate(L,R,B,TO);
    const xmax=Math.max(SLO*1.6, Math.min(s.p99*1.15, 2000));
    x.domain([0,xmax]);
    const bins=d3.bin().domain([0,xmax]).thresholds(50)(s.lat.filter(v=>v<=xmax));
    y.domain([0,d3.max(bins,b=>b.length)||1]);
    xa.call(d3.axisBottom(x).ticks(8)); ya.call(d3.axisLeft(y).ticks(4));
    g.selectAll(".axis text").attr("fill",C.muted).attr("font-size",9);
    g.selectAll(".axis path,.axis line").attr("stroke","#3a4150");
    bars.selectAll("rect").data(bins).join("rect")
      .attr("x",b=>x(b.x0)+0.5).attr("width",b=>Math.max(0,x(b.x1)-x(b.x0)-1))
      .attr("y",b=>y(b.length)).attr("height",b=>ih-y(b.length))
      .attr("fill","rgba(163,230,53,.35)").attr("stroke",C.accent).attr("stroke-opacity",.5);
    const place=(ln,tx,v,label)=>{const xv=Math.min(x(v),iw); ln.attr("x1",xv).attr("x2",xv);
      tx.attr("x",Math.min(xv+4,iw-110)).text(label+" "+(v>2000?">2000":v.toFixed(0))+" ms");};
    place(l50,t50,s.p50,"p50"); place(l99,t99,s.p99,"p99"); place(lslo,tslo,SLO,"SLO");

    const ok=!unstable && s.p99<=SLO;
    const st=unstable?["OVERLOADED — queue grows without bound",C.bad]:ok?["MEETS SLO",C.good]:["SLO MISSED — tail too slow",C.a2];
    d3.select("#bt-state").html(`<span style="color:${st[1]}">● ${st[0]}</span>`);
    d3.select("#bt-read").html(
      `capacity per replica at batch ${B}: <b>${cap.toFixed(0)}</b> req/s · offered per replica: <b>${(L/R).toFixed(0)}</b> req/s · `+
      `utilization <b>${(Math.min(s.util,1)*100).toFixed(0)}%</b> · mean batch <b>${s.meanB.toFixed(1)}</b>`+
      `<br>p50 <b>${s.p50.toFixed(1)}</b> ms · p99 <b>${s.p99>99999?"&gt;1e5":s.p99.toFixed(1)}</b> ms · replica-hours per 1M req: <b>${(R*1e6/L/3600).toFixed(2)}</b>`
    );
  }
  ["#bt-l","#bt-r","#bt-b","#bt-t","#bt-s"].forEach(id=>d3.select(id).on("input",draw));
  draw();
})();

/* ---- Viz 2: progressive canary with per-step two-proportion test ---- */
(function(){
  const C={accent:"#a3e635",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#cn-svg"); if(svg.empty()) return;
  const W=640,H=250,m={t:18,r:16,b:46,l:50};
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  const g=svg.append("g").attr("transform",`translate(${m.l},${m.t})`);
  const STEPS=[0.01,0.05,0.25,0.5,1.0], NWIN=50000, PB=0.010, ZCRIT=2.33;
  const gauss=d3.randomNormal(0,1);

  function binom(n,p){ // normal approximation, clipped
    if(n<=0) return 0;
    const mu=n*p, sd=Math.sqrt(n*p*(1-p));
    return Math.max(0,Math.round(mu+sd*gauss()));
  }

  const x=d3.scaleBand().domain(STEPS.map(String)).range([0,iw]).padding(0.25);
  const y=d3.scaleLinear().domain([0,0.04]).range([ih,0]);
  const xa=g.append("g").attr("class","axis").attr("transform",`translate(0,${ih})`);
  const ya=g.append("g").attr("class","axis");
  xa.call(d3.axisBottom(x).tickFormat(d=>d3.format(".0%")(+d)+" canary"));
  ya.call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".1%")));
  g.selectAll(".axis text").attr("fill",C.muted).attr("font-size",9);
  g.selectAll(".axis path,.axis line").attr("stroke","#3a4150");
  g.append("text").attr("transform","rotate(-90)").attr("x",-ih/2).attr("y",-38).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("observed error rate (95% CI)");
  const base=g.append("line").attr("x1",0).attr("x2",iw).attr("y1",y(PB)).attr("y2",y(PB)).attr("stroke",C.muted).attr("stroke-dasharray","3 3");
  const trueC=g.append("line").attr("x1",0).attr("x2",iw).attr("stroke",C.bad).attr("stroke-dasharray","2 4").attr("stroke-opacity",.7);
  const stepG=g.append("g");
  const lg=g.append("g").attr("transform",`translate(0,${ih+30})`);
  [["baseline",C.muted],["canary",C.accent]].forEach((d,i)=>{
    lg.append("circle").attr("cx",i*90+5).attr("cy",0).attr("r",4).attr("fill",d[1]);
    lg.append("text").attr("x",i*90+13).attr("y",4).attr("fill",C.muted).attr("font-size",10).text(d[0]);
  });
  lg.append("text").attr("x",200).attr("y",4).attr("fill",C.muted).attr("font-size",10).text("dashed red = candidate's true error rate (hidden from the analysis)");

  let results=[], status="idle", timer=null;
  function pc(){return +d3.select("#cn-e").property("value")/100;}

  function step(){
    if(status==="rolled back"||status==="promoted") return;
    const i=results.length, f=STEPS[i];
    const nc=Math.round(NWIN*f), nb=NWIN-nc;
    const ec=binom(nc,pc()), eb=nb>0?binom(nb,PB):binom(NWIN,PB);
    const nbEff=nb>0?nb:NWIN; // at 100% compare to a held-back baseline window
    const phc=ec/nc, phb=eb/nbEff, pbar=(ec+eb)/(nc+nbEff);
    const z=(phc-phb)/Math.sqrt(Math.max(pbar*(1-pbar),1e-9)*(1/nc+1/nbEff));
    const ci=(p,n)=>1.96*Math.sqrt(Math.max(p*(1-p),1e-9)/n);
    const abort=z>ZCRIT;
    results.push({f,nc,nb:nbEff,phc,phb,z,abort,cic:ci(phc,nc),cib:ci(phb,nbEff),excess:nc*(pc()-PB)});
    if(abort) status="rolled back";
    else if(i===STEPS.length-1) status="promoted";
    else status="running";
    render();
  }
  function render(){
    trueC.attr("y1",y(pc())).attr("y2",y(pc()));
    const sel=stepG.selectAll("g.st").data(results).join(enter=>{
      const s=enter.append("g").attr("class","st");
      s.append("rect").attr("class","bg"); s.append("line").attr("class","cb"); s.append("circle").attr("class","pb");
      s.append("line").attr("class","cc"); s.append("circle").attr("class","pcn"); s.append("text").attr("class","zt");
      return s;});
    sel.attr("transform",d=>`translate(${x(String(d.f))},0)`);
    const bw=x.bandwidth();
    sel.select("rect.bg").attr("x",0).attr("width",bw).attr("y",0).attr("height",ih)
      .attr("fill",d=>d.abort?"rgba(248,113,113,.12)":"rgba(74,222,128,.07)");
    const yc=v=>y(Math.max(0,Math.min(0.04,v)));
    sel.select("line.cb").attr("x1",bw*0.3).attr("x2",bw*0.3).attr("y1",d=>yc(d.phb-d.cib)).attr("y2",d=>yc(d.phb+d.cib)).attr("stroke",C.muted).attr("stroke-width",2);
    sel.select("circle.pb").attr("cx",bw*0.3).attr("cy",d=>yc(d.phb)).attr("r",4).attr("fill",C.muted);
    sel.select("line.cc").attr("x1",bw*0.7).attr("x2",bw*0.7).attr("y1",d=>yc(d.phc-d.cic)).attr("y2",d=>yc(d.phc+d.cic)).attr("stroke",C.accent).attr("stroke-width",2);
    sel.select("circle.pcn").attr("cx",bw*0.7).attr("cy",d=>yc(d.phc)).attr("r",4).attr("fill",C.accent);
    sel.select("text.zt").attr("x",bw/2).attr("y",12).attr("text-anchor","middle").attr("font-size",10)
      .attr("fill",d=>d.abort?C.bad:C.good).text(d=>(d.abort?"ABORT ":"pass ")+"z="+d.z.toFixed(2));

    const excess=d3.sum(results,d=>Math.max(0,d.excess));
    const col=status==="rolled back"?C.bad:status==="promoted"?C.good:C.a2;
    const msg={idle:"READY — advance to send 1% of traffic",running:`STEP ${results.length} PASSED — next: ${STEPS[results.length]*100}%`,
      "rolled back":`ROLLED BACK at ${results.length?results[results.length-1].f*100:0}% canary`,promoted:"FULLY PROMOTED"}[status];
    d3.select("#cn-state").html(`<span style="color:${col}">● ${msg}</span>`);
    const last=results[results.length-1];
    d3.select("#cn-read").html(
      (last?`last step: canary n=<b>${last.nc.toLocaleString()}</b> · observed <b>${(last.phc*100).toFixed(2)}%</b> vs baseline <b>${(last.phb*100).toFixed(2)}%</b> · z=<b>${last.z.toFixed(2)}</b><br>`:"")+
      `expected excess failures caused by the candidate so far (blast radius): <b style="color:${excess>200?C.bad:C.ink}">${excess.toFixed(0)}</b>`+
      (pc()<=PB&&status==="promoted"?` <span class="lbl">(candidate was no worse — promotion is correct)</span>`:"")+
      (pc()>PB&&status==="promoted"?` <span class="lbl" style="color:${C.bad}">(a worse model reached 100% — the test lacked power)</span>`:"")
    );
  }
  function reset(){ if(timer){timer.stop();timer=null;} results=[]; status="idle"; stepG.selectAll("*").remove(); d3.select("#cn-e-v").text((pc()*100).toFixed(2)+"%"); render(); }
  d3.select("#cn-step").on("click",step);
  d3.select("#cn-auto").on("click",()=>{ if(timer) timer.stop(); timer=d3.interval(()=>{ if(status==="rolled back"||status==="promoted"){timer.stop();timer=null;return;} step(); },550); });
  d3.select("#cn-reset").on("click",reset);
  d3.select("#cn-e").on("input",reset);
  reset();
})();
