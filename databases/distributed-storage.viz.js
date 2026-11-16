/* distributed-storage.viz.js — interactive figures for distributed-storage.html.
   Loaded after data.js / notes.js. */

/* ---- Viz 1: HDFS rack-aware block placement ---- */
(function(){
  const C={accent:"#22d3ee",node:"#5b9cff",good:"#4ade80",bad:"#f87171",q:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",panel:"#0e1016",line:"#222833"};
  const svg=d3.select("#hd-svg"); if(svg.empty()) return;
  const W=640,H=300, RACKS=3, NODES=4;
  const PAL=d3.range(30).map(i=>d3.hsl((i*47)%360,0.65,0.62).formatHex());
  let failed=new Set();                                 // node ids "r-n"
  function rng(s){ return function(){ s|=0; s=s+0x6D2B79F5|0; let t=Math.imul(s^s>>>15,1|s);
    t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
  function place(B,writerOnDN){
    const r=rng(31337), pick=n=>Math.floor(r()*n), blocks=[];
    for(let b=0;b<B;b++){
      const r1 = writerOnDN ? {rack:0,node:0} : {rack:pick(RACKS),node:pick(NODES)};
      let rack2=pick(RACKS-1); if(rack2>=r1.rack) rack2++;
      const n2=pick(NODES); let n3=pick(NODES-1); if(n3>=n2) n3++;
      blocks.push([r1,{rack:rack2,node:n2},{rack:rack2,node:n3}]);
    }
    return blocks;
  }
  function draw(){
    const size=+d3.select("#hd-size").property("value"), bs=+d3.select("#hd-bs").property("value"), wr=d3.select("#hd-wr").property("value");
    d3.select("#hd-size-v").text(size+" MB");
    const B=Math.ceil(size/bs), blocks=place(B,wr==="dn");
    const live=blocks.map(reps=>reps.filter(p=>!failed.has(p.rack+"-"+p.node)).length);
    svg.selectAll("*").remove();
    const rw=196, rx=r=>12+r*(rw+12), ny=n=>30+n*56, nh=50;
    for(let r=0;r<RACKS;r++){
      const rackDown=d3.range(NODES).every(n=>failed.has(r+"-"+n));
      svg.append("rect").attr("x",rx(r)).attr("y",6).attr("width",rw).attr("height",254).attr("rx",8).attr("fill",rackDown?"#1d1012":"#0b0d12").attr("stroke",rackDown?C.bad:C.line);
      svg.append("text").attr("x",rx(r)+10).attr("y",21).attr("fill",rackDown?C.bad:C.muted).attr("font-size",11).text("rack "+(r+1)+(rackDown?" — down":""));
      for(let n=0;n<NODES;n++){
        const id=r+"-"+n, dead=failed.has(id);
        const g=svg.append("g").style("cursor","pointer").on("click",()=>{ if(failed.has(id)) failed.delete(id); else failed.add(id); draw(); });
        g.append("rect").attr("x",rx(r)+8).attr("y",ny(n)).attr("width",rw-16).attr("height",nh).attr("rx",5).attr("fill",dead?"#2a1416":C.panel).attr("stroke",dead?C.bad:"#2a2f3a");
        g.append("text").attr("x",rx(r)+14).attr("y",ny(n)+12).attr("fill",dead?C.bad:C.muted).attr("font-size",9).text("r"+(r+1)+"-n"+(n+1)+(dead?" ✗":"")+(wr==="dn"&&r===0&&n===0?" (writer)":""));
        // replicas held on this node
        const held=[]; blocks.forEach((reps,b)=>reps.forEach(p=>{ if(p.rack===r&&p.node===n) held.push(b); }));
        held.forEach((b,i)=>{ const col=i%14, row=Math.floor(i/14);
          g.append("rect").attr("x",rx(r)+14+col*12).attr("y",ny(n)+17+row*11).attr("width",10).attr("height",9).attr("rx",2)
            .attr("fill",PAL[b%PAL.length]).attr("fill-opacity",dead?.2:.9); });
      }
    }
    // block status strip
    svg.append("text").attr("x",12).attr("y",278).attr("fill",C.muted).attr("font-size",10).text("blocks:");
    const sw=Math.min(22,(W-70)/B);
    blocks.forEach((_,b)=>{ const st=live[b]; const col=st===0?C.bad:st<3?C.q:C.good;
      svg.append("rect").attr("x",56+b*sw).attr("y",268).attr("width",sw-3).attr("height",14).attr("rx",2).attr("fill",PAL[b%PAL.length]).attr("stroke",col).attr("stroke-width",2);
      svg.append("text").attr("x",56+b*sw+(sw-3)/2).attr("y",294).attr("text-anchor","middle").attr("fill",col).attr("font-size",8).text(st); });
    const lost=live.filter(x=>x===0).length, under=live.filter(x=>x>0&&x<3).length;
    let msg=`${size} MB ÷ ${bs} MB = <b>${B}</b> blocks × 3 replicas = <b>${3*B}</b> stored blocks (${(3*size/1024).toFixed(1)} GB raw) · NameNode tracks ~${B+1} objects`;
    msg+=` · <span style="color:${C.good}">healthy ${B-lost-under}</span> · <span style="color:${C.q}">under-replicated ${under}</span> · <span style="color:${lost?C.bad:C.muted}">lost ${lost}</span>`;
    if(lost) msg+=`<br><span style="color:${C.bad}">Data loss: every replica of ${lost} block${lost>1?"s":""} sits on failed nodes — two racks (or 3 specific nodes) went down.</span>`;
    else if(under) msg+=`<br>No data lost. The NameNode would now schedule re-replication of the ${under} under-replicated block${under>1?"s":""} onto healthy nodes.`;
    if(wr==="dn") msg+=`<br>Writer on a DataNode: r1-n1 holds a replica of <b>every</b> block — fast local writes, but a hot spot. Re-balancing tools spread this out later.`;
    d3.select("#hd-read").html(msg);
  }
  d3.select("#hd-size").on("input",draw);
  d3.selectAll("#hd-bs,#hd-wr").on("change",draw);
  d3.select("#hd-r2").on("click",()=>{ const all=d3.range(NODES).every(n=>failed.has("1-"+n));
    d3.range(NODES).forEach(n=>all?failed.delete("1-"+n):failed.add("1-"+n)); draw(); });
  d3.select("#hd-heal").on("click",()=>{ failed.clear(); draw(); });
  draw();
})();

/* ---- Viz 2: replication vs erasure coding ---- */
(function(){
  const C={accent:"#22d3ee",node:"#5b9cff",good:"#4ade80",bad:"#f87171",q:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",panel:"#0e1016",line:"#222833"};
  const svg=d3.select("#ec-svg"); if(svg.empty()) return;
  const W=640,H=250;
  let fEC=new Set(), fRep=new Set();
  function logC(n,j){ let s=0; for(let i=1;i<=j;i++) s+=Math.log(n-j+i)-Math.log(i); return s; }
  function pLossEC(k,m,p){ const n=k+m; let s=0; for(let j=m+1;j<=n;j++) s+=Math.exp(logC(n,j)+j*Math.log(p)+(n-j)*Math.log(1-p)); return s; }
  const nines=P=>P<=0?30:Math.min(30,-Math.log10(P));
  function draw(){
    const k=+d3.select("#ec-k").property("value"), m=+d3.select("#ec-m").property("value"), r=+d3.select("#ec-r").property("value");
    const p=Math.pow(10,+d3.select("#ec-p").property("value"));
    d3.select("#ec-k-v").text(k); d3.select("#ec-m-v").text(m); d3.select("#ec-r-v").text(r); d3.select("#ec-p-v").text(p<0.001?p.toExponential(1):p.toFixed(3));
    fEC=new Set([...fEC].filter(i=>i<k+m)); fRep=new Set([...fRep].filter(i=>i<r));
    svg.selectAll("*").remove();
    // fragment strip
    const n=k+m, fw=Math.min(40,(W-150)/n);
    svg.append("text").attr("x",12).attr("y",30).attr("fill",C.ink).attr("font-size",11).text("RS("+k+","+m+")");
    for(let i=0;i<n;i++){ const isP=i>=k, dead=fEC.has(i);
      const g=svg.append("g").style("cursor","pointer").on("click",()=>{ dead?fEC.delete(i):fEC.add(i); draw(); });
      g.append("rect").attr("x",110+i*fw).attr("y",14).attr("width",fw-4).attr("height",26).attr("rx",4)
        .attr("fill",dead?"#2a1416":(isP?C.q:C.node)).attr("fill-opacity",dead?1:.8).attr("stroke",dead?C.bad:"none");
      g.append("text").attr("x",110+i*fw+(fw-4)/2).attr("y",31).attr("text-anchor","middle").attr("fill",dead?C.bad:"#0b0d12").attr("font-size",10).attr("font-weight",700).text(dead?"✗":(isP?"p"+(i-k+1):"d"+(i+1))); }
    const ecOK=fEC.size<=m;
    svg.append("text").attr("x",110).attr("y",56).attr("fill",ecOK?C.good:C.bad).attr("font-size",10)
      .text(ecOK?`${fEC.size} of ${n} lost — recoverable from any ${k} survivors (can lose ${m-fEC.size} more)`:`${fEC.size} lost > m = ${m} — data unrecoverable`);
    // replica strip
    svg.append("text").attr("x",12).attr("y",88).attr("fill",C.ink).attr("font-size",11).text(r+"× replication");
    const rwid=Math.min(120,(W-150)/r);
    for(let i=0;i<r;i++){ const dead=fRep.has(i);
      const g=svg.append("g").style("cursor","pointer").on("click",()=>{ dead?fRep.delete(i):fRep.add(i); draw(); });
      g.append("rect").attr("x",110+i*rwid).attr("y",72).attr("width",rwid-6).attr("height",26).attr("rx",4).attr("fill",dead?"#2a1416":C.good).attr("fill-opacity",dead?1:.75).attr("stroke",dead?C.bad:"none");
      g.append("text").attr("x",110+i*rwid+(rwid-6)/2).attr("y",89).attr("text-anchor","middle").attr("fill",dead?C.bad:"#0b0d12").attr("font-size",10).attr("font-weight",700).text(dead?"✗":"copy "+(i+1)); }
    const repOK=fRep.size<r;
    svg.append("text").attr("x",110).attr("y",114).attr("fill",repOK?C.good:C.bad).attr("font-size",10)
      .text(repOK?`${fRep.size} of ${r} copies lost — still readable`:`all copies lost — data gone`);
    // metric bars
    const ovEC=n/k, ovRep=r, PEC=pLossEC(k,m,p), PR=Math.pow(p,r), nEC=nines(PEC), nR=nines(PR);
    const bx=110, bw=200, y0=140;
    const Xo=d3.scaleLinear().domain([0,Math.max(ovRep,ovEC,3)]).range([0,bw]);
    const Xn=d3.scaleLinear().domain([0,Math.max(nEC,nR,6)]).range([0,bw]);
    svg.append("text").attr("x",bx).attr("y",y0).attr("fill",C.muted).attr("font-size",11).text("raw bytes stored per byte of data");
    svg.append("text").attr("x",bx+bw+60).attr("y",y0).attr("fill",C.muted).attr("font-size",11).text("durability (nines = −log₁₀ P(loss))");
    [["RS",ovEC,nEC,C.node],["rep",ovRep,nR,C.good]].forEach(([lab,ov,nn,col],i)=>{
      const y=y0+12+i*40;
      svg.append("text").attr("x",12).attr("y",y+16).attr("fill",col).attr("font-size",11).text(i===0?"RS("+k+","+m+")":r+"× rep");
      svg.append("rect").attr("x",bx).attr("y",y).attr("width",Xo(ov)).attr("height",22).attr("rx",3).attr("fill",col).attr("fill-opacity",.8);
      svg.append("text").attr("x",bx+Xo(ov)+5).attr("y",y+15).attr("fill",C.ink).attr("font-size",10).text(ov.toFixed(2)+"×");
      svg.append("rect").attr("x",bx+bw+60).attr("y",y).attr("width",Xn(nn)).attr("height",22).attr("rx",3).attr("fill",col).attr("fill-opacity",.8);
      svg.append("text").attr("x",bx+bw+60+Xn(nn)+5).attr("y",y+15).attr("fill",C.ink).attr("font-size",10).text(nn>=30?"≥30":nn.toFixed(1));
    });
    const verdict = (ovEC<ovRep && nEC>=nR) ? `<span style="color:${C.good}">RS(${k},${m}) is both cheaper and more durable here</span>`
      : (ovEC<ovRep) ? `RS(${k},${m}) is cheaper but less durable — raise m or speed up repair`
      : `replication is cheaper here — erasure coding pays off at larger k`;
    d3.select("#ec-read").html(`p = ${p<0.001?p.toExponential(1):p.toFixed(4)} per fragment during the repair window · RS P(loss) ≈ <b>${PEC.toExponential(2)}</b> · ${r}× P(loss) = pʳ ≈ <b>${PR.toExponential(2)}</b> · ${verdict}. Repair of one RS fragment reads <b>${k}</b> fragments vs 1 copy for replication. Assumes independent failures — correlated ones (same rack) make both far worse.`);
  }
  d3.selectAll("#ec-k,#ec-m,#ec-r,#ec-p").on("input",draw);
  draw();
})();
