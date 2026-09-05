/* python.viz.js — extracted from python.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

const PC={accent:"#5b9cff",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",cell:"#1b2130",cellb:"#2a2f3a"};

/* ───────────────────────── 03 · names, binding & aliasing ───────────────────────── */
(function(){
  const svg=d3.select("#alias-svg"),W=640,H=300;
  const out=document.getElementById("alias-readout");
  // heap objects: each {id, vals, x, y}
  let objs=[];          // list of objects on the heap
  let bindX=null;       // object id x points at (or null)
  let bindY=null;       // object id y points at (or null)
  let nextId=1;
  const nameX={x:60,y:90}, nameY={x:60,y:200};   // name slots (left column)
  const heapX=300;                                // heap column start

  function reset(){
    objs=[]; bindX=null; bindY=null; nextId=1; draw();
    out.innerHTML="<span style='color:var(--muted)'>No names bound yet. Press “x = [1,2,3]”.</span>";
  }
  function placeHeap(){
    // arrange objects vertically in the heap column
    objs.forEach((o,i)=>{ o.x=heapX; o.y=70+i*90; });
  }
  function newObj(vals){ const o={id:nextId++,vals:vals.slice()}; objs.push(o); placeHeap(); return o; }

  function doX(){ const o=newObj([1,2,3]); bindX=o.id; draw();
    out.innerHTML=`<b style="color:${PC.accent}">x = [1,2,3]</b> · created a list object; name <code>x</code> bound to it`; }
  function doY(){ if(bindX==null){ doX(); } bindY=bindX; draw();
    out.innerHTML=`<b style="color:${PC.a2}">y = x</b> · ALIAS — <code>x</code> and <code>y</code> point at the <b style="color:${PC.good}">SAME object</b>; mutation is visible through both`; }
  function doMut(){ if(bindY==null){ out.innerHTML="<span style='color:var(--bad)'>bind y first</span>"; return; }
    const o=objs.find(o=>o.id===bindY); o.vals.push(99); draw(true);
    const shared=(bindX===bindY);
    out.innerHTML=`<b>y.append(99)</b> · mutated in place${shared?` → the 99 is visible through <b style="color:${PC.accent}">x</b> AND <b style="color:${PC.a2}">y</b> (aliased)`:` → only <b style="color:${PC.a2}">y</b>'s object changed`}`; }
  function doRebind(){ const o=newObj([4,5]); bindY=o.id; draw();
    out.innerHTML=`<b style="color:${PC.a2}">y = [4,5]</b> · REBIND — <code>y</code>'s arrow moves to a NEW object; <b style="color:${PC.accent}">x</b> is unaffected`; }
  function doCopy(){ if(bindX==null){ out.innerHTML="<span style='color:var(--bad)'>bind x first</span>"; return; }
    const src=objs.find(o=>o.id===bindX); const o=newObj(src.vals); bindY=o.id; draw();
    out.innerHTML=`<b style="color:${PC.a2}">y = x.copy()</b> · DETACH — a new independent object; mutating one can't affect the other`; }

  function draw(flash){
    svg.selectAll("*").remove();
    // column headers
    svg.append("text").attr("x",nameX.x).attr("y",40).attr("fill",PC.muted).attr("font-size",12).text("names");
    svg.append("text").attr("x",heapX).attr("y",40).attr("fill",PC.muted).attr("font-size",12).text("heap (objects)");

    // name slots
    function nameBox(slot,label,color,bound){
      svg.append("rect").attr("x",slot.x).attr("y",slot.y-22).attr("width",70).attr("height",40).attr("rx",8)
        .attr("fill",PC.cell).attr("stroke",bound?color:PC.cellb).attr("stroke-width",bound?2:1);
      svg.append("text").attr("x",slot.x+35).attr("y",slot.y+4).attr("text-anchor","middle").attr("fill",color).attr("font-size",18).attr("font-weight",700).text(label);
    }
    nameBox(nameX,"x",PC.accent,bindX!=null);
    nameBox(nameY,"y",PC.a2,bindY!=null);

    // heap objects
    objs.forEach(o=>{
      const w=Math.max(120, o.vals.length*30+24);
      svg.append("rect").attr("class","obj-"+o.id).attr("x",o.x).attr("y",o.y-26).attr("width",w).attr("height",52).attr("rx",10)
        .attr("fill",PC.cell).attr("stroke",PC.cellb);
      o.w=w;
      o.vals.forEach((v,i)=>{
        const cx=o.x+16+i*30;
        const isNew=flash && v===99;
        svg.append("rect").attr("x",cx).attr("y",o.y-12).attr("width",24).attr("height",24).attr("rx",4)
          .attr("fill",isNew?"rgba(248,113,113,.22)":"rgba(154,163,178,.12)").attr("stroke",isNew?PC.bad:PC.cellb);
        svg.append("text").attr("x",cx+12).attr("y",o.y+5).attr("text-anchor","middle")
          .attr("fill",isNew?PC.bad:PC.ink).attr("font-size",13).attr("font-weight",isNew?700:400).text(v);
      });
      svg.append("text").attr("x",o.x).attr("y",o.y-34).attr("fill",PC.muted).attr("font-size",10).text("list @"+o.id);
    });

    // arrows from names to objects
    function arrow(slot,objId,color){
      if(objId==null) return; const o=objs.find(z=>z.id===objId); if(!o) return;
      const x1=slot.x+70, y1=slot.y-2, x2=o.x, y2=o.y;
      const mx=(x1+x2)/2;
      svg.append("path").attr("d",`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`)
        .attr("fill","none").attr("stroke",color).attr("stroke-width",2.2).attr("marker-end","url(#ah-"+(color===PC.accent?"x":"y")+")");
    }
    // arrowheads
    const defs=svg.append("defs");
    [["ah-x",PC.accent],["ah-y",PC.a2]].forEach(([id,c])=>{
      defs.append("marker").attr("id",id).attr("viewBox","0 0 10 10").attr("refX",9).attr("refY",5)
        .attr("markerWidth",7).attr("markerHeight",7).attr("orient","auto-start-reverse")
        .append("path").attr("d","M0,0 L10,5 L0,10 z").attr("fill",c);
    });
    // highlight shared object when aliased
    if(bindX!=null && bindX===bindY){
      const o=objs.find(z=>z.id===bindX);
      if(o) svg.select(".obj-"+o.id).attr("stroke",PC.good).attr("stroke-width",2.5);
      svg.append("text").attr("x",heapX).attr("y",H-14).attr("fill",PC.good).attr("font-size",11).text("x and y point at the SAME object (aliased)");
    } else if(bindX!=null && bindY!=null){
      svg.append("text").attr("x",heapX).attr("y",H-14).attr("fill",PC.muted).attr("font-size",11).text("x and y point at different objects (independent)");
    }
    arrow(nameX,bindX,PC.accent);
    arrow(nameY,bindY,PC.a2);
  }

  document.getElementById("al-x").addEventListener("click",doX);
  document.getElementById("al-y").addEventListener("click",doY);
  document.getElementById("al-mut").addEventListener("click",doMut);
  document.getElementById("al-rebind").addEventListener("click",doRebind);
  document.getElementById("al-copy").addEventListener("click",doCopy);
  document.getElementById("al-reset").addEventListener("click",reset);
  reset();
})();

/* ───────────────────────── 04 · the GIL & concurrency choice ───────────────────────── */
(function(){
  const svg=d3.select("#gil-svg"),W=640,H=300;
  const out=document.getElementById("gil-readout");
  const N=4;                              // workers
  const x0=120, y0=46, laneH=42, laneGap=10, trackW=460;
  let workload="cpu";                     // "cpu" | "io"
  let mode="thread";                      // thread | mp | async | ft
  let raf=null, t=0;

  // per (workload,mode) → {speedup, parallel, ioOverlap, note}
  function profile(){
    const k=workload+"|"+mode;
    const P={
      "cpu|thread":{spd:1.0, par:false, io:false, note:"CPU-bound + GIL: only one thread runs bytecode at a time → no scaling"},
      "cpu|mp":    {spd:N,   par:true,  io:false, note:"CPU-bound + processes: separate interpreters/GILs → true multi-core ≈ N×"},
      "cpu|async": {spd:1.0, par:false, io:false, note:"CPU-bound + asyncio: single thread, no overlap on compute → no help"},
      "cpu|ft":    {spd:N,   par:true,  io:false, note:"CPU-bound + free-threaded build: no GIL → threads run in parallel ≈ N× (3.14 supported, not default)"},
      "io|thread": {spd:N,   par:true,  io:true,  note:"I/O-bound + threads: GIL released during waits → lanes overlap, good concurrency"},
      "io|mp":     {spd:N,   par:true,  io:true,  note:"I/O-bound + processes: works, but heavier than threads for waiting work"},
      "io|async":  {spd:N,   par:true,  io:true,  note:"I/O-bound + asyncio: one thread cooperatively overlaps waits → ideal, low overhead"},
      "io|ft":     {spd:N,   par:true,  io:true,  note:"I/O-bound + free-threaded: also overlaps waits across cores"}
    };
    return P[k];
  }

  function start(){
    if(raf) cancelAnimationFrame(raf);
    t=0; const p=profile();
    function frame(){
      t+=0.012; if(t>1.05){ draw(p,1); return; }
      draw(p, Math.min(t,1)); raf=requestAnimationFrame(frame);
    }
    frame();
  }

  function draw(p,prog){
    svg.selectAll("*").remove();
    // CPU core legend
    svg.append("text").attr("x",x0).attr("y",24).attr("fill",PC.muted).attr("font-size",12)
      .text((mode==="mp"?"processes":mode==="async"?"tasks (1 thread)":"threads")+" over CPU timeline →");
    // lanes
    for(let i=0;i<N;i++){
      const ly=y0+i*(laneH+laneGap);
      svg.append("text").attr("x",x0-12).attr("y",ly+laneH/2+4).attr("text-anchor","end").attr("fill",PC.ink).attr("font-size",12)
        .text(mode==="async"?("task "+(i+1)):(mode==="mp"?("proc "+(i+1)):("thread "+(i+1))));
      // track background
      svg.append("rect").attr("x",x0).attr("y",ly).attr("width",trackW).attr("height",laneH).attr("rx",6)
        .attr("fill",PC.cell).attr("stroke",PC.cellb);

      if(workload==="io"){
        // I/O-bound: short compute block + shaded wait; overlap if parallel
        // segments: compute (solid) then I/O wait (shaded). When overlapping, all start at 0.
        const startFrac = p.par ? 0 : i/N;          // serialized if not parallel
        const compW = trackW*0.18, waitW=trackW*0.30;
        const baseX = x0 + startFrac*trackW;
        const filled = prog* (p.par? 1 : 1);
        // compute block
        const cw = Math.min(compW, Math.max(0,(prog - startFrac)*trackW));
        if(cw>0){
          svg.append("rect").attr("x",baseX).attr("y",ly+6).attr("width",Math.min(cw,compW)).attr("height",laneH-12).attr("rx",4)
            .attr("fill","rgba(91,156,255,.30)").attr("stroke",PC.accent);
        }
        // I/O wait (shaded, GIL released)
        if(prog> startFrac+compW/trackW){
          const ww=Math.min(waitW, (prog-startFrac-compW/trackW)*trackW);
          svg.append("rect").attr("x",baseX+compW).attr("y",ly+6).attr("width",Math.max(0,ww)).attr("height",laneH-12).attr("rx",4)
            .attr("fill","rgba(154,163,178,.10)").attr("stroke",PC.muted).attr("stroke-dasharray","3 3");
        }
      } else {
        // CPU-bound: solid compute block; serialized (GIL) vs parallel
        if(p.par){
          const fw=prog*trackW*0.9;
          svg.append("rect").attr("x",x0+4).attr("y",ly+6).attr("width",fw).attr("height",laneH-12).attr("rx",4)
            .attr("fill","rgba(74,222,128,.26)").attr("stroke",PC.good);
        } else {
          // each thread gets a 1/N time slice in turn — only one advances at a time
          const slice=trackW*0.9/N;
          const myStart=i/N, myEnd=(i+1)/N;
          let fw=0;
          if(prog>=myEnd) fw=slice; else if(prog>myStart) fw=(prog-myStart)*N*slice;
          if(fw>0) svg.append("rect").attr("x",x0+4+i*slice).attr("y",ly+6).attr("width",fw).attr("height",laneH-12).attr("rx",4)
            .attr("fill","rgba(255,180,84,.26)").attr("stroke",PC.a2);
        }
      }
    }
    // legend for block types (I/O)
    if(workload==="io"){
      svg.append("rect").attr("x",x0).attr("y",H-26).attr("width",14).attr("height",12).attr("fill","rgba(91,156,255,.30)").attr("stroke",PC.accent);
      svg.append("text").attr("x",x0+20).attr("y",H-16).attr("fill",PC.muted).attr("font-size",11).text("Python bytecode");
      svg.append("rect").attr("x",x0+140).attr("y",H-26).attr("width",14).attr("height",12).attr("fill","rgba(154,163,178,.10)").attr("stroke",PC.muted).attr("stroke-dasharray","3 3");
      svg.append("text").attr("x",x0+160).attr("y",H-16).attr("fill",PC.muted).attr("font-size",11).text("I/O wait (GIL released)");
    }
    const spd=p.spd;
    out.innerHTML=`<b>${workload==="cpu"?"CPU-bound":"I/O-bound"}</b> · <b style="color:${PC.accent}">${ {thread:"threads (GIL)",mp:"multiprocessing",async:"asyncio",ft:"free-threaded"}[mode] }</b> → effective speedup ≈ <b style="color:${spd>1?PC.good:PC.bad}">${spd===N?N+"×":spd.toFixed(1)+"×"}</b><br><span style="color:var(--muted)">${p.note}</span>`;
  }

  function setWorkload(w){ workload=w; start(); }
  function setMode(m){ mode=m; start(); }
  document.getElementById("gw-cpu").addEventListener("click",()=>setWorkload("cpu"));
  document.getElementById("gw-io").addEventListener("click",()=>setWorkload("io"));
  document.getElementById("gm-thread").addEventListener("click",()=>setMode("thread"));
  document.getElementById("gm-mp").addEventListener("click",()=>setMode("mp"));
  document.getElementById("gm-async").addEventListener("click",()=>setMode("async"));
  document.getElementById("gm-ft").addEventListener("click",()=>setMode("ft"));
  start();
})();
