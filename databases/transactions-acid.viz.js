/* transactions-acid.viz.js — visualizations for transactions-acid.html.
   1. WAL crash recovery (redo winners / undo losers)  2. isolation anomaly simulator. */

const TX_C={accent:"#22d3ee",node:"#5b9cff",good:"#4ade80",bad:"#f87171",warn:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",panel:"#0e1016",line:"#2a3140"};

/* ---- Viz 1: WAL crash recovery ---- */
(function(){
  const C=TX_C, svg=d3.select("#wal-svg"), W=640;
  const LOG=[
    {tx:1,k:"BEGIN"},{tx:1,k:"UPD",item:"x",from:10,to:20},{tx:2,k:"BEGIN"},{tx:2,k:"UPD",item:"y",from:5,to:7},
    {tx:1,k:"UPD",item:"z",from:1,to:2},{tx:1,k:"COMMIT"},{tx:3,k:"BEGIN"},{tx:3,k:"UPD",item:"x",from:20,to:30},
    {tx:2,k:"UPD",item:"y",from:7,to:9},{tx:3,k:"COMMIT"},{tx:2,k:"UPD",item:"z",from:2,to:4},{tx:2,k:"COMMIT"}];
  const TXC={1:C.node,2:C.warn,3:"#c084fc"};
  function draw(){
    const k=+d3.select("#wal-k").property("value");
    const seen=LOG.slice(0,k), began=new Set(seen.filter(r=>r.k==="BEGIN").map(r=>r.tx)), won=new Set(seen.filter(r=>r.k==="COMMIT").map(r=>r.tx));
    const lost=[...began].filter(t=>!won.has(t));
    svg.selectAll("*").remove();
    const x0=14, bw=50, gap=2, y0=26, bh=38;
    svg.append("text").attr("x",x0).attr("y",14).attr("fill",C.muted).attr("font-size",10).text("write-ahead log (LSN →)");
    LOG.forEach((r,i)=>{
      const x=x0+i*(bw+gap), dur=i<k, winner=won.has(r.tx), col=TXC[r.tx];
      svg.append("rect").attr("x",x).attr("y",y0).attr("width",bw).attr("height",bh).attr("rx",4)
        .attr("fill",dur?(winner?"rgba(74,222,128,.13)":"rgba(248,113,113,.12)"):C.panel)
        .attr("stroke",dur?col:"#262b36").attr("stroke-opacity",dur?1:.6);
      svg.append("text").attr("x",x+bw/2).attr("y",y0+15).attr("text-anchor","middle").attr("font-size",r.k==="UPD"?9.5:8.2).attr("font-weight",600)
        .attr("fill",dur?col:"#4b5263").text("T"+r.tx+" "+(r.k==="UPD"?r.item:r.k==="BEGIN"?"BEGIN":"COMMIT"));
      if(r.k==="UPD") svg.append("text").attr("x",x+bw/2).attr("y",y0+29).attr("text-anchor","middle").attr("font-size",9).attr("fill",dur?C.ink:"#4b5263").text(r.from+"→"+r.to);
      svg.append("text").attr("x",x+bw/2).attr("y",y0+bh+11).attr("text-anchor","middle").attr("font-size",8.5).attr("fill","#5b6372").text(i+1);
    });
    const cx=x0+k*(bw+gap)-1;
    svg.append("line").attr("x1",cx).attr("x2",cx).attr("y1",y0-8).attr("y2",y0+bh+16).attr("stroke",C.bad).attr("stroke-width",2.2);
    svg.append("text").attr("x",cx).attr("y",y0-10).attr("text-anchor","middle").attr("fill",C.bad).attr("font-size",10).attr("font-weight",700).text("⚡ crash");
    if(k<LOG.length) svg.append("rect").attr("x",cx+2).attr("y",y0-2).attr("width",x0+LOG.length*(bw+gap)-cx).attr("height",bh+4).attr("fill","#0f1117").attr("opacity",.55);
    // per-transaction verdicts
    const ty=y0+bh+36;
    [1,2,3].forEach((t,i)=>{
      const y=ty+i*22, st=!began.has(t)?["never started","—",C.muted]:won.has(t)?["COMMIT record is durable","REDO (winner)",C.good]:["no COMMIT record","UNDO (loser)",C.bad];
      svg.append("circle").attr("cx",x0+6).attr("cy",y-4).attr("r",5).attr("fill",TXC[t]);
      svg.append("text").attr("x",x0+18).attr("y",y).attr("fill",C.ink).attr("font-size",11).text("T"+t);
      svg.append("text").attr("x",x0+44).attr("y",y).attr("fill",C.muted).attr("font-size",10.5).text(st[0]);
      svg.append("text").attr("x",x0+230).attr("y",y).attr("fill",st[2]).attr("font-size",10.5).attr("font-weight",700).text(st[1]);
    });
    // recovered state
    const state={x:10,y:5,z:1};
    seen.filter(r=>r.k==="UPD" && won.has(r.tx)).forEach(r=>state[r.item]=r.to);
    const sx=400;
    svg.append("text").attr("x",sx).attr("y",ty-4).attr("fill",C.muted).attr("font-size",10).text("state after recovery");
    ["x","y","z"].forEach((it,i)=>{
      const x=sx+i*74;
      svg.append("rect").attr("x",x).attr("y",ty+4).attr("width",64).attr("height",44).attr("rx",6).attr("fill",C.panel).attr("stroke",C.line);
      svg.append("text").attr("x",x+32).attr("y",ty+20).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text(it);
      svg.append("text").attr("x",x+32).attr("y",ty+40).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",16).attr("font-weight",700).text(state[it]);
    });
    svg.append("text").attr("x",sx).attr("y",ty+66).attr("fill",C.muted).attr("font-size",9.5).text("initial x=10, y=5, z=1");
    d3.select("#wal-read").html(`crash after record <b>${k}</b> · redo: <b style="color:${C.good}">${[...won].map(t=>"T"+t).join(", ")||"none"}</b> · undo: <b style="color:${C.bad}">${lost.map(t=>"T"+t).join(", ")||"none"}</b> · `+
      (lost.length?`the losers' changes may already be on disk (steal), so recovery uses the logged before-images to roll them back.`:`no transaction was mid-flight, so recovery is pure redo.`));
  }
  d3.select("#wal-k").on("input",draw);
  draw();
})();

/* ---- Viz 2: isolation anomaly simulator ---- */
(function(){
  const C=TX_C, svg=d3.select("#iso-svg"), W=640, H=250;
  const LV={RU:"Read Uncommitted",RC:"Read Committed",SI:"Snapshot Isolation",SER:"Serializable"};
  // each scenario: level → {init, steps:[{tx,a,b,col?,note}], bad, out}
  const SC={
    dirty:L=>{ const leak=L==="RU"; return {init:"x = 100",steps:[
      {tx:1,a:"BEGIN",note:"T1 starts"},
      {tx:1,a:"W(x) = 50",b:"uncommitted",note:"T1 writes x but has not committed"},
      {tx:2,a:"R(x)",b:"→ "+(leak?"50":"100"),col:leak?C.bad:C.good,note:leak?"T2 reads T1's uncommitted value":"T2 reads the last committed value"},
      {tx:1,a:"ROLLBACK",col:C.warn,note:"T1 aborts; x = 50 never existed"},
      {tx:2,a:"COMMIT",b:"used x="+(leak?"50":"100"),note:"T2 commits work based on what it read"}],
      bad:leak,out:leak?"dirty read: T2 acted on x = 50, a value that was rolled back":"no anomaly: T2 never saw uncommitted data"}; },
    fuzzy:L=>{ const bad=L==="RU"||L==="RC"; return {init:"x = 100",steps:[
      {tx:1,a:"R(x)",b:"→ 100",note:"T1 reads x"},
      {tx:2,a:"W(x) = 50",note:"T2 updates x"},
      {tx:2,a:"COMMIT",note:"T2 commits"},
      {tx:1,a:"R(x)",b:"→ "+(bad?"50":"100"),col:bad?C.bad:C.good,note:bad?"each statement sees the latest committed data":"T1 keeps reading from its snapshot"},
      {tx:1,a:"COMMIT",note:"T1 commits"}],
      bad,out:bad?"non-repeatable read: the same query returned 100 then 50 inside one transaction":"no anomaly: T1 saw a stable value throughout"}; },
    lost:L=>{ const bad=L==="RU"||L==="RC"; return {init:"counter = 10",steps:[
      {tx:1,a:"R(c)",b:"→ 10",note:"T1 reads the counter"},
      {tx:2,a:"R(c)",b:"→ 10",note:"T2 reads the counter"},
      {tx:1,a:"W(c) = 11",note:"T1 writes its increment"},
      {tx:1,a:"COMMIT",note:"T1 commits"},
      bad?{tx:2,a:"W(c) = 11",col:C.bad,note:"T2 overwrites with its stale increment"}
         :{tx:2,a:"W(c) = 11",b:"✗ conflict",col:C.warn,note:"c was changed by a concurrent committed transaction → first-committer-wins aborts T2"},
      bad?{tx:2,a:"COMMIT",note:"T2 commits"}:{tx:2,a:"ROLLBACK",b:"retry → 12",col:C.good,note:"T2 retries, reads 11 and writes 12"}],
      bad,out:bad?"lost update: two increments ran but the counter is 11":"no anomaly: one transaction was aborted and retried; counter = 12"}; },
    phantom:L=>{ const bad=L==="RU"||L==="RC"; return {init:"3 rows WHERE dept='A'",steps:[
      {tx:1,a:"COUNT(dept=A)",b:"→ 3",note:"T1 counts matching rows"},
      {tx:2,a:"INSERT dept=A",note:"T2 inserts a new matching row"},
      {tx:2,a:"COMMIT",note:"T2 commits"},
      {tx:1,a:"COUNT(dept=A)",b:"→ "+(bad?"4":"3"),col:bad?C.bad:C.good,note:bad?"the new row appears mid-transaction":"the snapshot does not contain the new row"},
      {tx:1,a:"COMMIT",note:"T1 commits"}],
      bad,out:bad?"phantom: the set of rows matching the predicate changed inside T1":"no anomaly: T1's predicate saw a stable set"}; },
    skew:L=>{ const ser=L==="SER"; return {init:"on_call: alice, bob (2)",steps:[
      {tx:1,a:"COUNT(on_call)",b:"→ 2",note:"Alice's transaction checks that someone else is on call"},
      {tx:2,a:"COUNT(on_call)",b:"→ 2",note:"Bob's transaction does the same check"},
      {tx:1,a:"W(alice off)",note:"T1 takes Alice off call (row alice)"},
      {tx:2,a:"W(bob off)",note:"T2 takes Bob off call (a different row, so no write-write conflict)"},
      {tx:1,a:"COMMIT",note:"T1 commits"},
      ser?{tx:2,a:"COMMIT",b:"✗ 40001",col:C.warn,note:"serializable: a cycle of read→write dependencies is detected, so T2 aborts"}
         :{tx:2,a:"COMMIT",col:C.bad,note:"T2 commits; nothing it wrote was touched by T1"}],
      bad:!ser,out:!ser?"write skew: each check passed on its own snapshot, and now nobody is on call (0)":"no anomaly: T2 was aborted; its retry sees 1 on call and refuses"}; }
  };
  let cur=0;
  function plan(){ return SC[d3.select("#iso-s").property("value")](d3.select("#iso-l").property("value")); }
  function draw(){
    const P=plan(), L=d3.select("#iso-l").property("value"), n=P.steps.length;
    svg.selectAll("*").remove();
    const lx=70, lw=W-lx-14, col=lw/n, lane={1:66,2:156}, bw=Math.min(92,col-8), bh=46;
    svg.append("text").attr("x",14).attr("y",18).attr("fill",C.muted).attr("font-size",10.5).text("initial: "+P.init+"   ·   level: "+LV[L]);
    [1,2].forEach(t=>{
      svg.append("line").attr("x1",lx-6).attr("x2",W-10).attr("y1",lane[t]).attr("y2",lane[t]).attr("stroke","#262b36");
      svg.append("text").attr("x",14).attr("y",lane[t]+4).attr("fill",t===1?C.node:C.warn).attr("font-size",12).attr("font-weight",700).text("T"+t);
    });
    svg.append("text").attr("x",W-12).attr("y",H-50).attr("text-anchor","end").attr("fill","#5b6372").attr("font-size",9.5).text("time →");
    P.steps.forEach((s,i)=>{
      const on=i<cur, x=lx+i*col+(col-bw)/2, y=lane[s.tx]-bh/2;
      const stroke=on?(s.col||(s.tx===1?C.node:C.warn)):"#262b36";
      svg.append("rect").attr("x",x).attr("y",y).attr("width",bw).attr("height",bh).attr("rx",6)
        .attr("fill",on?(s.col===C.bad?"rgba(248,113,113,.14)":s.col===C.good?"rgba(74,222,128,.12)":C.panel):"#0c0e13")
        .attr("stroke",stroke).attr("stroke-width",i===cur-1?2.2:1);
      svg.append("text").attr("x",x+bw/2).attr("y",y+19).attr("text-anchor","middle").attr("font-size",10).attr("font-weight",600).attr("fill",on?C.ink:"#3d4350").text(s.a);
      if(s.b) svg.append("text").attr("x",x+bw/2).attr("y",y+35).attr("text-anchor","middle").attr("font-size",9.5).attr("fill",on?(s.col||C.muted):"#3d4350").text(s.b);
      svg.append("text").attr("x",lx+i*col+col/2).attr("y",206).attr("text-anchor","middle").attr("font-size",9).attr("fill","#5b6372").text("t"+(i+1));
    });
    if(cur>=n){
      svg.append("rect").attr("x",14).attr("y",216).attr("width",W-28).attr("height",26).attr("rx",6).attr("fill",P.bad?"rgba(248,113,113,.12)":"rgba(74,222,128,.10)").attr("stroke",P.bad?C.bad:C.good);
      svg.append("text").attr("x",24).attr("y",233).attr("fill",P.bad?C.bad:C.good).attr("font-size",11).attr("font-weight",600).text((P.bad?"✗ ":"✓ ")+P.out);
    }
    const note=cur===0?"press ▶ step to run the interleaving":cur<=n?P.steps[cur-1].note:"";
    d3.select("#iso-read").html(`step <b>${Math.min(cur,n)}/${n}</b> · ${note}`+(cur>=n?` · <b style="color:${P.bad?C.bad:C.good}">${P.bad?"anomaly allowed":"anomaly prevented"}</b> under ${LV[L]}`:""));
  }
  d3.select("#iso-step").on("click",()=>{ if(cur<plan().steps.length){cur++;draw();} });
  d3.select("#iso-all").on("click",()=>{ cur=plan().steps.length; draw(); });
  d3.select("#iso-reset").on("click",()=>{ cur=0; draw(); });
  d3.selectAll("#iso-s,#iso-l").on("change",()=>{ cur=0; draw(); });
  draw();
})();
