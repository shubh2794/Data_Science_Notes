/* hyperparameter-tuning.viz.js — extracted from hyperparameter-tuning.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ── grid vs random vs Bayesian on a 2-D validation surface ── */
(function(){
  const C={accent:"#ffb454",good:"#4ade80",bad:"#f87171",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#hp-svg"),W=640,H=290;
  const px=40,py=20,pw=420,ph=230;
  const opt=[0.62,0.40];
  function scoreFn(ridge){
    return ridge
      ? (x,y)=>Math.exp(-((x-opt[0])**2/0.02))            // only A matters
      : (x,y)=>Math.exp(-((x-opt[0])**2/0.10 + (y-opt[1])**2/0.10));
  }
  function draw(){
    svg.selectAll("*").remove();
    const k=d3.select("#hp-sel").property("value");
    const n=+d3.select("#hp-n").property("value");
    const ridge=d3.select("#hp-ridge").property("checked");
    const score=scoreFn(ridge);
    const N=28;
    for(let i=0;i<N;i++)for(let j=0;j<N;j++){
      const x=i/(N-1), y=j/(N-1);
      svg.append("rect").attr("x",px+i/N*pw).attr("y",py+(1-y)*ph - ph/N).attr("width",pw/N+1).attr("height",ph/N+1)
        .attr("fill",d3.interpolateInferno(0.1+score(x,y)*0.85)).attr("opacity",0.9);
    }
    svg.append("rect").attr("x",px).attr("y",py).attr("width",pw).attr("height",ph).attr("fill","none").attr("stroke","#3a4150");
    svg.append("text").attr("x",px+pw/2).attr("y",py+ph+34).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("hyperparameter A →");
    svg.append("text").attr("x",px-6).attr("y",py-7).attr("fill",C.muted).attr("font-size",10).text("hp B");

    let pts=[];
    if(k==="grid"){
      const s=Math.max(2,Math.round(Math.sqrt(n)));
      for(let i=1;i<=s;i++)for(let j=1;j<=s;j++) pts.push([i/(s+1),j/(s+1)]);
    } else if(k==="random"){
      for(let i=0;i<n;i++) pts.push([Math.random(),Math.random()]);
    } else {
      const seedN=Math.max(3,Math.round(n*0.25));
      for(let i=0;i<seedN;i++) pts.push([Math.random(),Math.random()]);
      // crude surrogate: shrink toward the best observed point
      for(let i=seedN;i<n;i++){
        let bx=pts[0],bs=-1; pts.forEach(p=>{const s=score(p[0],p[1]); if(s>bs){bs=s;bx=p;}});
        const w=0.45*(1-i/n)+0.08;
        pts.push([Math.min(1,Math.max(0,bx[0]+(Math.random()-0.5)*2*w)),
                  Math.min(1,Math.max(0,bx[1]+(Math.random()-0.5)*2*w))]);
      }
    }
    let best=0,bp=null;
    pts.forEach((p,i)=>{ const s=score(p[0],p[1]); if(s>best){best=s;bp=p;}
      svg.append("circle").attr("cx",px+p[0]*pw).attr("cy",py+(1-p[1])*ph).attr("r",4)
        .attr("fill","#fff").attr("stroke","#0f1117").attr("stroke-width",1).attr("opacity",0.9); });
    if(bp) svg.append("circle").attr("cx",px+bp[0]*pw).attr("cy",py+(1-bp[1])*ph).attr("r",7).attr("fill","none").attr("stroke",C.good).attr("stroke-width",2);

    // marginal ticks: distinct A values probed
    const uniq=new Set(pts.map(p=>Math.round(p[0]*1000)));
    pts.forEach(p=>svg.append("line").attr("x1",px+p[0]*pw).attr("y1",py+ph+5).attr("x2",px+p[0]*pw).attr("y2",py+ph+15)
      .attr("stroke",C.accent).attr("stroke-opacity",.75).attr("stroke-width",1.4));
    svg.append("text").attr("x",px+pw+8).attr("y",py+ph+14).attr("fill",C.muted).attr("font-size",9).text("distinct A");

    svg.append("text").attr("x",px+pw+24).attr("y",py+20).attr("fill",C.muted).attr("font-size",10).text("○ trials");
    svg.append("text").attr("x",px+pw+24).attr("y",py+40).attr("fill",C.good).attr("font-size",10).text("◎ best found");
    svg.append("text").attr("x",px+pw+24).attr("y",py+66).attr("fill",C.muted).attr("font-size",9.5).text(ridge?"surface: a ridge —":"surface: a broad");
    svg.append("text").attr("x",px+pw+24).attr("y",py+80).attr("fill",C.muted).attr("font-size",9.5).text(ridge?"B is irrelevant":"optimum in A,B");

    const note = k==='grid'?'grid re-tests the same few A values':k==='random'?'random probes a new A every trial':'Bayesian narrows toward the incumbent';
    d3.select("#hp-read").html(`<b>${pts.length}</b> trials · <b>${uniq.size}</b> distinct values of A · best score <b>${(best*100).toFixed(0)}%</b> · ${note}`);
  }
  d3.select("#hp-sel").on("change",draw);
  d3.select("#hp-n").on("input",draw);
  d3.select("#hp-ridge").on("change",draw);
  d3.select("#hp-run").on("click",draw);
  draw();
})();

/* ── successive halving: configs compete for compute across rungs ── */
(function(){
  const C={accent:"#ffb454",good:"#4ade80",bad:"#f87171",muted:"#9aa3b2",ink:"#e6e9ef"};
  const svg=d3.select("#sha-svg"),W=640,H=300,L=48,R=150,T=18,B=38;
  const N=27, RMAX=27;          // 27 configs, full budget 27 units
  let cfgs=[];
  function sample(){
    cfgs=[];
    for(let i=0;i<N;i++){
      const floorErr=0.06+0.42*Math.random();      // error it converges to
      const speed=0.25+2.6*Math.random();          // how fast it gets there
      const jitter=0.02+0.05*Math.random();
      cfgs.push({floorErr,speed,jitter,ph:Math.random()*6.28});
    }
  }
  const curve=(c,t)=>c.floorErr+(0.92-c.floorErr)*Math.exp(-c.speed*t/8)+c.jitter*Math.sin(c.ph+t*0.9)*0.35;
  const sx=d3.scaleLog().domain([1,RMAX]).range([L,W-R]);
  const sy=d3.scaleLinear().domain([0,1]).range([H-B,T]);

  function draw(){
    const eta=+d3.select("#sha-eta").property("value");
    const rungs=Math.max(1,Math.floor(Math.log(N)/Math.log(eta)));
    const rSel=d3.select("#sha-r");
    rSel.attr("max",rungs);
    let cur=Math.min(+rSel.property("value"),rungs);
    rSel.property("value",cur);

    // budgets: rung i runs from r_{i-1} to r_i where r_i = RMAX * eta^(i-rungs)
    const budget=i=>RMAX*Math.pow(eta,i-rungs);
    // rank configs by score at each rung's end, keep top 1/eta
    let alive=cfgs.map((c,i)=>i);
    const cull=[];   // cull[i] = rung index at which config i stopped
    cfgs.forEach(()=>cull.push(rungs));
    let spent=0, survivors=[alive.slice()];
    for(let i=0;i<rungs;i++){
      const lo=i===0?0:budget(i-1), hi=budget(i);
      spent+=alive.length*(hi-lo);
      const scored=alive.map(ix=>[ix,curve(cfgs[ix],hi)]).sort((a,b)=>a[1]-b[1]);
      const keep=Math.max(1,Math.floor(alive.length/eta));
      scored.slice(keep).forEach(s=>cull[s[0]]=i);
      alive=scored.slice(0,keep).map(s=>s[0]);
      survivors.push(alive.slice());
    }
    // last stretch for final survivors
    const lo=rungs===0?0:budget(rungs-1);
    spent+=alive.length*(RMAX-lo);

    const trueBest=d3.least(d3.range(N),i=>curve(cfgs[i],RMAX));
    const shownBudget = cur>=rungs ? RMAX : budget(cur);

    svg.selectAll("*").remove();
    // axes
    svg.append("line").attr("x1",L).attr("y1",H-B).attr("x2",W-R).attr("y2",H-B).attr("stroke","#3a4150");
    svg.append("line").attr("x1",L).attr("y1",T).attr("x2",L).attr("y2",H-B).attr("stroke","#3a4150");
    svg.append("text").attr("x",(L+W-R)/2).attr("y",H-6).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("training budget per config (log) →");
    svg.append("text").attr("x",L-6).attr("y",T-4).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10).text("val error");
    for(let i=0;i<=rungs;i++){
      const b=i===rungs?RMAX:budget(i);
      svg.append("line").attr("x1",sx(b)).attr("y1",T).attr("x2",sx(b)).attr("y2",H-B).attr("stroke","#2a2f3a").attr("stroke-dasharray","2 4");
      svg.append("text").attr("x",sx(b)).attr("y",T-5).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9).text(i===rungs?"full":"rung "+i);
    }
    const ln=d3.line().x(d=>sx(d[0])).y(d=>sy(Math.max(0,Math.min(1,d[1]))));
    cfgs.forEach((c,i)=>{
      const stopB = cull[i]>=rungs ? RMAX : budget(cull[i]);
      const endB = Math.min(stopB, shownBudget);
      const ts=d3.range(1,endB+0.0001,Math.max(0.15,endB/60)).map(t=>[t,curve(c,t)]);
      if(ts.length<2) return;
      const dead = cull[i]<rungs && stopB<=shownBudget;
      svg.append("path").attr("d",ln(ts)).attr("fill","none")
        .attr("stroke", i===trueBest?C.accent:(dead?C.muted:C.good))
        .attr("stroke-width", i===trueBest?2.4:1.3)
        .attr("stroke-opacity", i===trueBest?1:(dead?0.28:0.8));
      if(dead) svg.append("circle").attr("cx",sx(stopB)).attr("cy",sy(curve(c,stopB))).attr("r",2.4).attr("fill",C.bad).attr("fill-opacity",.55);
    });
    // legend
    svg.append("text").attr("x",W-R+12).attr("y",T+14).attr("fill",C.good).attr("font-size",10).text("— still running");
    svg.append("text").attr("x",W-R+12).attr("y",T+30).attr("fill",C.muted).attr("font-size",10).text("— culled");
    svg.append("text").attr("x",W-R+12).attr("y",T+46).attr("fill",C.accent).attr("font-size",10).text("— best at full budget");

    const full=N*RMAX;
    const bestSurvived = cull[trueBest]>=rungs;
    const aliveNow = cfgs.filter((c,i)=>cull[i]>=cur).length;
    d3.select("#sha-read").html(
      `η = <b>${eta}</b> · ${rungs+1} rungs · showing rung <b>${cur}</b> · <b>${aliveNow}</b> of ${N} configs still alive &nbsp;|&nbsp; budget spent <b>${Math.round(spent)}</b> vs <b>${full}</b> to train all fully (<b style="color:${C.good}">${(full/spent).toFixed(1)}×</b> cheaper) &nbsp;|&nbsp; true best <b style="color:${bestSurvived?C.good:C.bad}">${bestSurvived?"survived":"was culled at rung "+cull[trueBest]}</b>`);
  }
  d3.select("#sha-eta").on("change",draw);
  d3.select("#sha-r").on("input",draw);
  d3.select("#sha-new").on("click",()=>{sample();draw();});
  sample(); draw();
})();
