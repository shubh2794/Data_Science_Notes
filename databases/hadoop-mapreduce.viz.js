/* hadoop-mapreduce.viz.js — interactive figures for hadoop-mapreduce.html.
   Loaded after data.js / notes.js. Uses d3 from ../vendor. */

/* ---- Viz 1: word-count dataflow (split → map → combine → shuffle → reduce → output) ---- */
(function(){
  const C={accent:"#22d3ee",good:"#4ade80",bad:"#f87171",warm:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",panel:"#0e1016",line:"#222833"};
  const RC=["#5b9cff","#ffb454","#c084fc","#4ade80"];           // one colour per reducer
  const svg=d3.select("#mr-svg");
  if(svg.empty()) return;
  const SPLITS=["the cat sat on the mat","the dog sat on the log","the cat and the dog"];
  const PHASES=["input splits","map: emit (word, 1)","combine (map-side)","shuffle: partition by hash(k) mod R","reduce: sum per key"];
  let R=2, comb=false, phase=0;
  const hash=w=>{let h=0;for(const ch of w)h=(h*31+ch.charCodeAt(0))>>>0;return h;};

  function compute(){
    const maps=SPLITS.map(s=>s.split(" ").map(w=>({k:w,v:1})));
    const combs=maps.map(m=>{
      if(!comb) return m.map(p=>({k:p.k,v:p.v}));
      const roll=d3.rollup(m,v=>d3.sum(v,d=>d.v),d=>d.k);
      return Array.from(roll,([k,v])=>({k,v})).sort((a,b)=>d3.ascending(a.k,b.k));
    });
    const red=d3.range(R).map(()=>new Map());
    combs.forEach((m,si)=>m.forEach(p=>{ p.r=hash(p.k)%R; p.si=si;
      const M=red[p.r]; if(!M.has(p.k)) M.set(p.k,[]); M.get(p.k).push(p.v); }));
    const reds=red.map(M=>Array.from(M,([k,vs])=>({k,vs})).sort((a,b)=>d3.ascending(a.k,b.k)));
    return {maps,combs,reds};
  }

  const top=30, rowH=12.5, gap=12;
  const COL={split:8, map:168, comb:256, red:392, out:548};

  function draw(){
    svg.selectAll("*").remove();
    const D=compute();
    // column headers
    const heads=[["input split",COL.split],["map → (k,1)",COL.map],[comb?"combine":"no combiner",COL.comb],["shuffle → reduce",COL.red],["output",COL.out]];
    heads.forEach(([t,x],i)=>{
      const shown = i===0 || (i===1&&phase>=1) || (i===2&&phase>=2) || (i===3&&phase>=3) || (i===4&&phase>=4);
      svg.append("text").attr("x",x).attr("y",16).attr("fill",shown?C.ink:"#3a4150").attr("font-size",10.5).attr("font-weight",600).text(t);
    });
    // split blocks (height = longest list among map / combine output for that split)
    let y=top; const blockY=[], rowY=[];
    SPLITS.forEach((s,si)=>{
      const n=Math.max(D.maps[si].length,1);
      const h=n*rowH+14;
      blockY.push([y,h]);
      svg.append("rect").attr("x",COL.split-4).attr("y",y).attr("width",COL.comb+76-COL.split).attr("height",h).attr("rx",7)
        .attr("fill",C.panel).attr("stroke",C.line);
      svg.append("text").attr("x",COL.split).attr("y",y+14).attr("fill",C.muted).attr("font-size",9.5).text("split "+si);
      const words=s.split(" ");
      // wrap the line into 2 rows for legibility
      svg.append("text").attr("x",COL.split).attr("y",y+30).attr("fill",C.ink).attr("font-size",10.5).text(words.slice(0,3).join(" "));
      svg.append("text").attr("x",COL.split).attr("y",y+44).attr("fill",C.ink).attr("font-size",10.5).text(words.slice(3).join(" "));
      if(phase>=1){
        D.maps[si].forEach((p,i)=>{
          svg.append("text").attr("x",COL.map).attr("y",y+12+i*rowH).attr("fill",comb&&phase>=2?"#5f6878":C.ink)
            .attr("font-size",10).attr("font-family","monospace").text(`(${p.k},${p.v})`);
        });
      }
      if(phase>=2){
        const list=D.combs[si];
        const rows=[];
        list.forEach((p,i)=>{
          const yy=y+12+i*rowH; rows.push(yy);
          svg.append("text").attr("x",COL.comb).attr("y",yy).attr("fill",phase>=3?RC[p.r]:C.ink)
            .attr("font-size",10).attr("font-family","monospace").text(`(${p.k},${p.v})`);
        });
        rowY.push(rows);
      }
      y+=h+gap;
    });
    // reducers
    const redTop=top;
    const sizes=D.reds.map(r=>Math.max(r.length,1)*rowH+24);
    let ry=redTop; const keyY=[];
    D.reds.forEach((list,ri)=>{
      const h=sizes[ri]; const ys=new Map();
      if(phase>=3){
        svg.append("rect").attr("x",COL.red-6).attr("y",ry).attr("width",140).attr("height",h).attr("rx",7)
          .attr("fill",C.panel).attr("stroke",RC[ri]).attr("stroke-opacity",.7);
        svg.append("text").attr("x",COL.red).attr("y",ry+12).attr("fill",RC[ri]).attr("font-size",9.5).text("reducer "+ri+"  ("+d3.sum(list,d=>d.vs.length)+" records in)");
      }
      list.forEach((kv,i)=>{
        const yy=ry+26+i*rowH; ys.set(kv.k,yy);
        if(phase>=3){
          svg.append("text").attr("x",COL.red).attr("y",yy).attr("fill",C.ink).attr("font-size",10).attr("font-family","monospace")
            .text(`${kv.k} [${kv.vs.join(",")}]`);
        }
        if(phase>=4){
          svg.append("text").attr("x",COL.out).attr("y",yy).attr("fill",C.good).attr("font-size",10.5).attr("font-family","monospace")
            .text(`${kv.k}  ${d3.sum(kv.vs)}`);
        }
      });
      keyY.push(ys);
      ry+=h+8;
    });
    // shuffle edges
    if(phase>=3){
      const link=d3.linkHorizontal();
      D.combs.forEach((list,si)=>list.forEach((p,i)=>{
        const y0=rowY[si][i]-3, y1=keyY[p.r].get(p.k)-3;
        svg.insert("path",":first-child").attr("d",link({source:[COL.comb+64,y0],target:[COL.red-8,y1]}))
          .attr("fill","none").attr("stroke",RC[p.r]).attr("stroke-opacity",.45).attr("stroke-width",1.1);
      }));
    }
    // readout
    const emitted=d3.sum(D.maps,m=>m.length), shuffled=d3.sum(D.combs,m=>m.length);
    const loads=D.reds.map(r=>d3.sum(r,d=>d.vs.length));
    const msg=[
      "Three input splits, one map task each. Nothing has run yet.",
      `Each mapper emitted one (word, 1) pair per token: <b>${emitted}</b> intermediate records in total.`,
      comb ? `The combiner pre-summed inside each mapper: <b>${emitted}</b> → <b>${shuffled}</b> records will cross the network.`
           : `No combiner: all <b>${emitted}</b> records go to the shuffle unchanged.`,
      `Partitioner hash(k) mod ${R} routes every copy of a key to one reducer. Shuffled <b>${shuffled}</b> records over ${3*R} map→reduce transfers; reducer loads [${loads.join(", ")}].`,
      `Each reducer sums the values for its keys. ${8} distinct words, ${R} output file${R>1?"s":""}. Max reducer load ${d3.max(loads)} of ${shuffled} records.`
    ];
    d3.select("#mr-read").html(`<span style="color:${C.accent}">▶</span> phase ${phase+1}/5 · ${PHASES[phase]} — ${msg[phase]}`);
  }
  d3.select("#mr-next").on("click",()=>{ phase=Math.min(4,phase+1); draw(); });
  d3.select("#mr-prev").on("click",()=>{ phase=Math.max(0,phase-1); draw(); });
  d3.select("#mr-r").on("input",function(){ R=+this.value; d3.select("#mr-rv").text(R); draw(); });
  d3.select("#mr-comb").on("change",function(){ comb=this.checked; draw(); });
  draw();
})();

/* ---- Viz 2: stragglers and speculative execution (Gantt) ---- */
(function(){
  const C={accent:"#22d3ee",good:"#4ade80",bad:"#f87171",warm:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",task:"#5b9cff"};
  const svg=d3.select("#sp-svg");
  if(svg.empty()) return;
  const BASE=[10,11,9.5,10.5,12,9,10,11];      // seconds per map task on a healthy node
  const SLOW=5;                                // index of the task on the degraded node
  const BACKUP=10.5;                           // a fresh copy on a healthy node takes ~median time
  const W=640, px=80, pw=520, py=18, rowH=24;
  function draw(){
    svg.selectAll("*").remove();
    const slow=+d3.select("#sp-slow").property("value"), spec=d3.select("#sp-on").property("checked");
    d3.select("#sp-slowv").text(slow+"×");
    const dur=BASE.map((b,i)=>i===SLOW?b*slow:b);
    // speculation trigger: once 75% of tasks have finished, back up every task still running
    const sorted=dur.slice().sort((a,b)=>a-b);
    const tTrig=sorted[Math.ceil(BASE.length*0.75)-1];
    const tasks=dur.map((d,i)=>{
      const t={i,d,end:d,backup:null,winner:"orig"};
      if(spec && d>tTrig){ const bEnd=tTrig+BACKUP; t.backup={start:tTrig,end:bEnd};
        if(bEnd<d){t.end=bEnd;t.winner="backup";} }
      return t;
    });
    const job=d3.max(tasks,t=>t.end), ideal=d3.max(BASE);
    const xmax=Math.max(40,Math.ceil(d3.max(dur)/5)*5+5);
    const X=d3.scaleLinear().domain([0,xmax]).range([px,px+pw]);
    const axisY=py+BASE.length*rowH+6;
    svg.append("g").attr("class","axis").attr("transform",`translate(0,${axisY})`).call(d3.axisBottom(X).ticks(8).tickFormat(d=>d+"s"));
    tasks.forEach(t=>{
      const y=py+t.i*rowH;
      svg.append("text").attr("x",px-8).attr("y",y+14).attr("text-anchor","end").attr("fill",t.i===SLOW?C.bad:C.muted).attr("font-size",10)
        .text((t.i===SLOW?"slow node ":"node ")+t.i);
      const killed = t.winner==="backup";
      svg.append("rect").attr("x",X(0)).attr("y",y+3).attr("height",14).attr("rx",3)
        .attr("width",X(killed?t.end:t.d)-X(0))
        .attr("fill",t.i===SLOW?C.bad:C.task).attr("fill-opacity",killed?0.35:0.85);
      if(killed){
        svg.append("text").attr("x",X(t.end)+4).attr("y",y+14).attr("fill",C.muted).attr("font-size",9).text("original killed");
      }
      if(t.backup){
        const won=t.winner==="backup";
        svg.append("rect").attr("x",X(t.backup.start)).attr("y",y+8).attr("height",9).attr("rx",3)
          .attr("width",X(won?t.backup.end:t.d)-X(t.backup.start))
          .attr("fill",C.warm).attr("fill-opacity",won?0.95:0.35).attr("stroke",C.warm).attr("stroke-dasharray",won?null:"3 2");
        if(!won) svg.append("text").attr("x",X(t.d)+4).attr("y",y+14).attr("fill",C.muted).attr("font-size",9).text("backup wasted");
      }
    });
    if(spec){
      svg.append("line").attr("x1",X(tTrig)).attr("x2",X(tTrig)).attr("y1",py-4).attr("y2",axisY).attr("stroke",C.warm).attr("stroke-dasharray","3 3");
      svg.append("text").attr("x",X(tTrig)+3).attr("y",py-6).attr("fill",C.warm).attr("font-size",9).text("75% done → launch backups");
    }
    svg.append("line").attr("x1",X(job)).attr("x2",X(job)).attr("y1",py-4).attr("y2",axisY).attr("stroke",C.accent).attr("stroke-width",2);
    svg.append("text").attr("x",X(job)+4).attr("y",axisY-4).attr("fill",C.accent).attr("font-size",10).text("job done "+job.toFixed(1)+"s");
    const extra=tasks.filter(t=>t.backup).length;
    d3.select("#sp-read").html(`map phase ends at <b>${job.toFixed(1)}s</b> (healthy cluster: ${ideal}s) · `+
      (spec?`${extra} backup task${extra===1?"":"s"} launched, costing extra slot time — `+(tasks.some(t=>t.winner==="backup")?`<span style="color:${C.good}">the backup beat the straggler</span>`:`<span style="color:${C.muted}">no straggler to beat, slots wasted</span>`)
           :`<span style="color:${C.bad}">every reducer waits on the slowest map task</span>`));
  }
  d3.select("#sp-slow").on("input",draw);
  d3.select("#sp-on").on("change",draw);
  draw();
})();
