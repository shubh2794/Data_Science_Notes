/* cap-theorem.viz.js — visualizations for cap-theorem.html.
   1. partition simulator (CP vs AP)  2. quorum calculator (R + W > N). */

const CAP_C={accent:"#22d3ee",node:"#5b9cff",good:"#4ade80",bad:"#f87171",warn:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",panel:"#0e1016",line:"#2a3140"};

/* ---- Viz 1: partition simulator ---- */
(function(){
  const C=CAP_C, svg=d3.select("#cap-svg"), W=640, H=260;
  const POS={A:[90,70],B:[90,190],C:[290,130]}, SIDE={A:0,B:0,C:1};
  let S, cut, clock, nextV, log, stats, cutTs=0;
  function reset(){ S={A:{v:0,ts:0},B:{v:0,ts:0},C:{v:0,ts:0}}; cut=false; clock=0; nextV=1; log=[]; stats={ok:0,rej:0,stale:0}; draw(); }
  const mode=()=>d3.select("#cap-mode").property("value");
  const reach=n=>Object.keys(S).filter(m=>!cut||SIDE[m]===SIDE[n]);
  const latest=()=>Object.values(S).reduce((a,b)=>b.ts>a.ts?b:a);
  let globalLatest={v:0,ts:0};        // latest acknowledged write anywhere
  function push(txt,col){ log.push({txt,col}); if(log.length>8) log.shift(); }
  function write(n){
    const r=reach(n);
    if(cut && mode()==="CP" && r.length<2){ stats.rej++; push(`✗ write → ${n}: no majority reachable · unavailable`,C.bad); return draw(); }
    clock++; const v=nextV++; r.forEach(m=>S[m]={v,ts:clock}); globalLatest={v,ts:clock}; stats.ok++;
    push(`✓ write x=${v} → ${n}, acked by {${r.join(",")}}`+(r.length<3?` · ${Object.keys(S).filter(m=>!r.includes(m)).join(",")} missed it`:""),r.length<3?C.warn:C.good);
    draw();
  }
  function read(n){
    const r=reach(n);
    if(cut && mode()==="CP" && r.length<2){ stats.rej++; push(`✗ read ← ${n}: no majority · refuses rather than risk stale data`,C.bad); return draw(); }
    const val = mode()==="CP" ? r.map(m=>S[m]).reduce((a,b)=>b.ts>a.ts?b:a) : S[n];
    stats.ok++;
    if(val.ts<globalLatest.ts){ stats.stale++; push(`⚠ read ← ${n}: x=${val.v} · STALE (latest acked is x=${globalLatest.v})`,C.warn); }
    else push(`✓ read ← ${n}: x=${val.v} (latest)`,C.good);
    draw();
  }
  function toggleCut(){
    if(!cut){ cut=true; cutTs=clock; push("✂ link cut: {A,B} | {C}",C.bad); }
    else{
      cut=false;
      const left=S.A.ts>=S.B.ts?S.A:S.B, right=S.C, win=latest();
      if(left.ts>cutTs && right.ts>cutTs){          // both sides accepted writes during the partition
        const loser=left.ts===win.ts?right:left;
        push(`⟳ healed · conflict! LWW keeps x=${win.v}, write x=${loser.v} is silently discarded`,C.warn);
      } else if(left.ts!==right.ts) push(`⟳ healed · lagging replica catches up to x=${win.v}`,C.accent);
      else push("⟳ healed · replicas already agree",C.accent);
      Object.keys(S).forEach(m=>S[m]={...win});
      globalLatest={v:win.v,ts:win.ts};
    }
    d3.select("#cap-part").text(cut?"⟳ heal link":"✂ cut link");
    draw();
  }
  function draw(){
    svg.selectAll("*").remove();
    // links
    [["A","B"],["A","C"],["B","C"]].forEach(([a,b])=>{
      const broken=cut && SIDE[a]!==SIDE[b];
      svg.append("line").attr("x1",POS[a][0]).attr("y1",POS[a][1]).attr("x2",POS[b][0]).attr("y2",POS[b][1])
        .attr("stroke",broken?"#3a2328":"#2f3a4c").attr("stroke-width",2).attr("stroke-dasharray",broken?"3 5":null);
    });
    if(cut){
      svg.append("line").attr("x1",195).attr("x2",195).attr("y1",14).attr("y2",H-14).attr("stroke",C.bad).attr("stroke-width",2.5).attr("stroke-dasharray","8 5");
      svg.append("text").attr("x",195).attr("y",H-2).attr("text-anchor","middle").attr("fill",C.bad).attr("font-size",10).text("partition");
      svg.append("text").attr("x",90).attr("y",20).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9.5).text("majority side");
      svg.append("text").attr("x",290).attr("y",60).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9.5).text("minority side");
    }
    Object.keys(POS).forEach(n=>{
      const [x,y]=POS[n], stale=S[n].ts<globalLatest.ts;
      svg.append("circle").attr("cx",x).attr("cy",y).attr("r",30).attr("fill",C.panel).attr("stroke",stale?C.warn:C.node).attr("stroke-width",2.2);
      svg.append("text").attr("x",x).attr("y",y-6).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",13).attr("font-weight",700).text(n);
      svg.append("text").attr("x",x).attr("y",y+12).attr("text-anchor","middle").attr("fill",stale?C.warn:C.good).attr("font-size",11).text("x="+S[n].v);
    });
    // event log
    const lx=350;
    svg.append("rect").attr("x",lx-8).attr("y",8).attr("width",W-lx).attr("height",H-16).attr("rx",8).attr("fill","#0c0e13").attr("stroke","#222833");
    svg.append("text").attr("x",lx).attr("y",26).attr("fill",C.muted).attr("font-size",10).text("event log ("+mode()+" mode)");
    log.forEach((e,i)=>{
      const words=e.txt; const y=46+i*26;
      const t=svg.append("text").attr("x",lx).attr("y",y).attr("fill",e.col).attr("font-size",9.6);
      if(words.length>44){ const cutAt=words.lastIndexOf(" ",44); t.append("tspan").attr("x",lx).text(words.slice(0,cutAt)); t.append("tspan").attr("x",lx+10).attr("dy",11).text(words.slice(cutAt+1)); }
      else t.text(words);
    });
    if(!log.length) svg.append("text").attr("x",lx).attr("y",48).attr("fill","#5b6372").attr("font-size",10).text("cut the link, then write to A and read from C");
    d3.select("#cap-read").html(`answered <b>${stats.ok}</b> · refused <b style="color:${C.bad}">${stats.rej}</b> · stale reads <b style="color:${C.warn}">${stats.stale}</b> — `+
      (mode()==="CP"?"CP keeps every answer linearizable by refusing requests on the minority side.":"AP answers every request, so it pays with stale reads and lost concurrent writes when the partition heals."));
  }
  d3.select("#cap-part").on("click",toggleCut);
  d3.select("#cap-wa").on("click",()=>write("A"));
  d3.select("#cap-wc").on("click",()=>write("C"));
  d3.select("#cap-ra").on("click",()=>read("A"));
  d3.select("#cap-rc").on("click",()=>read("C"));
  d3.select("#cap-reset").on("click",()=>{ globalLatest={v:0,ts:0}; d3.select("#cap-part").text("✂ cut link"); reset(); });
  d3.select("#cap-mode").on("change",()=>{ globalLatest={v:0,ts:0}; d3.select("#cap-part").text("✂ cut link"); reset(); });
  reset();
})();

/* ---- Viz 2: quorum calculator ---- */
(function(){
  const C=CAP_C, svg=d3.select("#q-svg"), W=640, H=230;
  const LAT=[8,11,14,19,26,38,55,90,160];          // per-replica response times (ms), replica i
  function draw(){
    const N=+d3.select("#q-n").property("value");
    ["#q-w","#q-r"].forEach(id=>{ const el=d3.select(id); el.attr("max",N); if(+el.property("value")>N) el.property("value",N); });
    const Wq=+d3.select("#q-w").property("value"), R=+d3.select("#q-r").property("value"), sloppy=d3.select("#q-sloppy").property("checked");
    const ov=Math.max(0,R+Wq-N);
    svg.selectAll("*").remove();
    const gap=Math.min(64,(W-80)/N), x0=(W-gap*(N-1))/2, y=62;
    svg.append("text").attr("x",14).attr("y",18).attr("fill",C.muted).attr("font-size",10).text("replicas of one key (worst-case placement)");
    for(let i=0;i<N;i++){
      const x=x0+i*gap, inW=i<Wq, inR=i>=N-R, both=inW&&inR;
      svg.append("circle").attr("cx",x).attr("cy",y).attr("r",20).attr("fill",both?"rgba(34,211,238,.25)":inW?"rgba(74,222,128,.18)":C.panel)
        .attr("stroke",inR?C.node:"#2f3a4c").attr("stroke-width",inR?2.6:1.4).attr("stroke-dasharray",inR&&!both?"4 2":null);
      svg.append("text").attr("x",x).attr("y",y+4).attr("text-anchor","middle").attr("fill",both?C.accent:inW?C.good:C.muted).attr("font-size",11).attr("font-weight",600).text(inW?"v₂":"v₁");
      svg.append("text").attr("x",x).attr("y",y+36).attr("text-anchor","middle").attr("fill","#5b6372").attr("font-size",9).text("r"+(i+1));
    }
    // brackets
    const bW=[x0-22,x0+(Wq-1)*gap+22], bR=[x0+(N-R)*gap-22,x0+(N-1)*gap+22];
    svg.append("path").attr("d",`M${bW[0]},${y-26} V${y-32} H${bW[1]} V${y-26}`).attr("fill","none").attr("stroke",C.good).attr("stroke-width",1.6);
    svg.append("text").attr("x",(bW[0]+bW[1])/2).attr("y",y-36).attr("text-anchor","middle").attr("fill",C.good).attr("font-size",10).text("write set W="+Wq);
    svg.append("path").attr("d",`M${bR[0]},${y+46} V${y+52} H${bR[1]} V${y+46}`).attr("fill","none").attr("stroke",C.node).attr("stroke-width",1.6);
    svg.append("text").attr("x",(bR[0]+bR[1])/2).attr("y",y+64).attr("text-anchor","middle").attr("fill",C.node).attr("font-size",10).text("read set R="+R);
    // latency strip
    const ly=158, lx=60, lw=W-120, maxL=LAT[N-1]*1.05, X=v=>lx+v/maxL*lw;
    svg.append("text").attr("x",14).attr("y",ly-14).attr("fill",C.muted).attr("font-size",10).text("replica response times (ms) — a request waits for the k-th fastest");
    svg.append("line").attr("x1",lx).attr("x2",lx+lw).attr("y1",ly).attr("y2",ly).attr("stroke","#3a4150");
    LAT.slice(0,N).forEach((l,i)=>{ svg.append("circle").attr("cx",X(l)).attr("cy",ly).attr("r",4).attr("fill",C.muted); });
    const wl=LAT[Wq-1], rl=LAT[R-1];
    [[wl,"write ≈ "+wl+" ms",C.good,-14],[rl,"read ≈ "+rl+" ms",C.node,24]].forEach(([v,t,c,dy])=>{
      svg.append("line").attr("x1",X(v)).attr("x2",X(v)).attr("y1",ly-10).attr("y2",ly+10).attr("stroke",c).attr("stroke-width",2.4);
      svg.append("text").attr("x",X(v)).attr("y",ly+dy).attr("text-anchor","middle").attr("fill",c).attr("font-size",10).text(t);
    });
    const ok=R+Wq>N && !sloppy;
    svg.append("rect").attr("x",14).attr("y",H-30).attr("width",W-28).attr("height",24).attr("rx",6).attr("fill",ok?"rgba(74,222,128,.10)":"rgba(248,113,113,.10)").attr("stroke",ok?C.good:C.bad);
    svg.append("text").attr("x",24).attr("y",H-14).attr("fill",ok?C.good:C.bad).attr("font-size",10.5).attr("font-weight",600)
      .text(sloppy?"sloppy quorum: W acks may come from stand-in nodes outside the home N, so overlap is NOT guaranteed"
        :ok?`R + W = ${R+Wq} > N = ${N}: every read overlaps the last completed write in ≥ ${ov} replica${ov>1?"s":""}`
        :`R + W = ${R+Wq} ≤ N = ${N}: a read can miss the latest write entirely (stale reads possible)`);
    d3.select("#q-read").html(`N=${N}, W=${Wq}, R=${R} · writes tolerate <b>${N-Wq}</b> down · reads tolerate <b>${N-R}</b> down · `+
      (Wq>N/2?`W &gt; N/2 so two write quorums always intersect`:`<span style="color:${C.warn}">W ≤ N/2: two concurrent writes can be acked by disjoint sets</span>`));
  }
  d3.selectAll("#q-n,#q-w,#q-r,#q-sloppy").on("input change",draw);
  draw();
})();
