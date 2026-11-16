/* indexing.viz.js — visualizations for indexing.html.
   Loaded after data.js / notes.js. Three independent widgets:
   1. B+-tree height calculator  2. leftmost-prefix scanner  3. planner cost crossover. */

const IX_C={accent:"#22d3ee",node:"#5b9cff",good:"#4ade80",bad:"#f87171",warn:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",edge:"#2a2f3a",panel:"#0e1016"};

/* ---- Viz 1: B+-tree height calculator ---- */
(function(){
  const C=IX_C, svg=d3.select("#bt-svg"), W=640, H=250, PTR=8;
  const fmt=n=>n>=1e9?(n/1e9).toFixed(n>=1e10?0:1)+"B":n>=1e6?(n/1e6).toFixed(n>=1e7?0:1)+"M":n>=1e3?(n/1e3).toFixed(n>=1e4?0:1)+"k":String(Math.round(n));
  const bytes=b=>b>=1<<30?(b/(1<<30)).toFixed(1)+" GB":b>=1<<20?(b/(1<<20)).toFixed(1)+" MB":(b/1024).toFixed(0)+" KB";
  function draw(){
    const page=+d3.select("#bt-page").property("value");
    const key=+d3.select("#bt-key").property("value");
    const N=Math.round(Math.pow(10,+d3.select("#bt-n").property("value")));
    const fill=+d3.select("#bt-fill").property("value");
    const f=Math.max(2,Math.floor(page/(key+PTR)*fill));     // inner fan-out and leaf capacity (same entry size)
    const levels=[Math.ceil(N/f)];                              // leaf count
    while(levels[levels.length-1]>1) levels.push(Math.ceil(levels[levels.length-1]/f));
    levels.reverse();                                           // root first
    const h=levels.length, innerNodes=d3.sum(levels.slice(0,-1)), leafNodes=levels[h-1];
    svg.selectAll("*").remove();
    const top=18, rowH=Math.min(46,(H-40)/Math.max(h,1)), bx=60, bw=430;
    let prev=null;
    levels.forEach((cnt,i)=>{
      const y=top+i*rowH, shown=Math.min(cnt,7), isLeaf=i===h-1;
      const w=Math.min(52,(bw-(shown-1)*8)/shown), gap=shown>1?(bw-shown*w)/(shown-1):0;
      const pathIdx=Math.floor(shown/2);
      for(let j=0;j<shown;j++){
        const x=bx+(shown===1?(bw-w)/2:j*(w+gap));
        const on=j===pathIdx;
        if(cnt>7 && j===5){ svg.append("text").attr("x",x+w/2).attr("y",y+rowH*0.45).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",14).text("…"); continue; }
        svg.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",rowH*0.55).attr("rx",4)
          .attr("fill",on?"rgba(74,222,128,.18)":C.panel).attr("stroke",on?C.good:isLeaf?"#2c4a5a":"#2a3140").attr("stroke-width",on?1.8:1);
        if(on){
          const cx=x+w/2;
          if(prev) svg.append("line").attr("x1",prev[0]).attr("y1",prev[1]).attr("x2",cx).attr("y2",y).attr("stroke",C.good).attr("stroke-width",1.8);
          prev=[cx,y+rowH*0.55];
        }
      }
      if(isLeaf && shown>1){ // leaf chain
        svg.append("line").attr("x1",bx).attr("y1",y+rowH*0.7).attr("x2",bx+bw).attr("y2",y+rowH*0.7).attr("stroke",C.accent).attr("stroke-dasharray","3 3").attr("stroke-opacity",.6);
        svg.append("text").attr("x",bx).attr("y",y+rowH*0.7+11).attr("fill",C.accent).attr("font-size",9).text("leaf chain → range scans");
      }
      svg.append("text").attr("x",bx-10).attr("y",y+rowH*0.38).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10).text(i===0?"root":isLeaf?"leaf":"L"+i);
      svg.append("text").attr("x",bx+bw+14).attr("y",y+rowH*0.38).attr("fill",isLeaf?C.ink:C.muted).attr("font-size",10.5)
        .text(fmt(cnt)+" node"+(cnt>1?"s":"")+" · "+bytes(cnt*page));
    });
    const bin=Math.ceil(Math.log2(N));
    d3.select("#bt-read").html(`fan-out <b>${f}</b> · height <b>${h}</b> levels for <b>${fmt(N)}</b> rows (a binary tree would need ${bin}) · inner levels total <b>${bytes(innerNodes*page)}</b>, leaves <b>${bytes(leafNodes*page)}</b>. `+
      (innerNodes*page<8*(1<<30)?`<span style="color:${C.good}">Inner levels fit in RAM</span> → a point lookup ≈ <b>1</b> disk read (the leaf) plus a heap fetch.`:`<span style="color:${C.warn}">Inner levels exceed ~8 GB</span>; expect the lower inner level to miss the cache.`));
  }
  d3.selectAll("#bt-page,#bt-key,#bt-n,#bt-fill").on("input change",draw);
  draw();
})();

/* ---- Viz 2: leftmost-prefix on a composite index (country, city, signup_date) ---- */
(function(){
  const C=IX_C, svg=d3.select("#lp-svg"), W=640;
  const R=[["DE","Berlin","2024-02"],["DE","Berlin","2024-08"],["DE","Munich","2024-03"],["DE","Munich","2024-09"],
    ["FR","Lyon","2024-05"],["FR","Paris","2024-01"],["FR","Paris","2024-03"],["FR","Paris","2024-07"],["FR","Paris","2024-11"],
    ["US","Austin","2024-03"],["US","Austin","2024-06"],["US","Austin","2024-08"],["US","Austin","2024-12"],
    ["US","Paris","2024-03"],["US","Paris","2024-10"],["US","Seattle","2024-02"],["US","Seattle","2024-07"]];
  const NAMES=["country","city","signup_date"];
  function preds(){
    const c=d3.select("#lp-c").property("value"), t=d3.select("#lp-t").property("value"), d=d3.select("#lp-d").property("value");
    return [
      c==="any"?null:{op:"eq",test:r=>r[0]===c,txt:`country='${c}'`},
      t==="any"?null:{op:"eq",test:r=>r[1]===t,txt:`city='${t}'`},
      d==="any"?null:d==="ge"?{op:"range",test:r=>r[2]>="2024-07",txt:"signup_date≥'2024-07'"}:{op:"eq",test:r=>r[2]==="2024-03",txt:"signup_date='2024-03'"}
    ];
  }
  function draw(){
    const P=preds(), cov=d3.select("#lp-cov").property("checked");
    const seek=[];                                   // leading equalities, then at most one range
    for(let i=0;i<3;i++){ if(!P[i]) break; seek.push(i); if(P[i].op==="range") break; }
    const filt=[0,1,2].filter(i=>P[i] && !seek.includes(i));
    const inRange=r=>seek.every(i=>P[i].test(r));
    const match=r=>P.every(p=>!p||p.test(r));
    svg.selectAll("*").remove();
    const x0=120, colX=[x0,x0+70,x0+170], top=50, rh=15.5;
    NAMES.forEach((n,i)=>{
      const role=seek.includes(i)?(P[i].op==="range"?"range seek":"seek"):filt.includes(i)?"filter":"—";
      const col=seek.includes(i)?C.good:filt.includes(i)?C.warn:C.muted;
      svg.append("text").attr("x",colX[i]).attr("y",16).attr("fill",C.ink).attr("font-size",11).attr("font-weight",600).text(n);
      svg.append("text").attr("x",colX[i]).attr("y",30).attr("fill",col).attr("font-size",9.5).text(role);
    });
    let first=-1,last=-1,scanned=0,returned=0;
    R.forEach((r,k)=>{
      const y=top+k*rh, a=inRange(r), m=match(r);
      if(a){ if(first<0) first=k; last=k; scanned++; } if(m) returned++;
      const fillc=m?"rgba(74,222,128,.16)":a?"rgba(255,180,84,.13)":"transparent";
      svg.append("rect").attr("x",x0-8).attr("y",y-11).attr("width",300).attr("height",rh-2).attr("rx",3).attr("fill",fillc);
      r.forEach((v,i)=>svg.append("text").attr("x",colX[i]).attr("y",y).attr("font-size",10.5)
        .attr("fill",m?C.good:a?C.warn:"#4b5263").text(v));
      svg.append("text").attr("x",x0+320).attr("y",y).attr("font-size",9.5).attr("fill",m?C.good:a?C.warn:"#4b5263")
        .text(m?"returned":a?"scanned, filtered out":"skipped");
      svg.append("text").attr("x",x0-14).attr("y",y).attr("text-anchor","end").attr("font-size",9).attr("fill","#4b5263").text("#"+(k+1));
    });
    if(first>=0){ // bracket for the contiguous seek range
      const y1=top+first*rh-12, y2=top+last*rh+3, bx=x0-40;
      svg.append("path").attr("d",`M${bx+8},${y1} H${bx} V${y2} H${bx+8}`).attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2);
      svg.append("text").attr("x",bx-4).attr("y",(y1+y2)/2+3).attr("text-anchor","end").attr("fill",C.accent).attr("font-size",9).text(seek.length?"seek":"full");
    }
    const any=P.some(Boolean);
    const seekTxt=seek.length?`seek on (${seek.map(i=>NAMES[i]).join(", ")})`:(any?"no predicate on the leading column → <b>cannot seek</b>: full index scan, skip scan, or a seq scan":"no predicates → full scan");
    const heap=cov?`<span style="color:${C.good}">index-only scan, 0 table fetches</span>`:`${returned} table fetch${returned===1?"":"es"}`;
    d3.select("#lp-read").html(`${seekTxt} · touches <b>${scanned}</b>/${R.length} entries · returns <b>${returned}</b>`+
      (filt.length?` · filter on ${filt.map(i=>NAMES[i]).join(", ")} evaluated on index entries`:"")+` · ${heap}`);
  }
  d3.selectAll("#lp-c,#lp-t,#lp-d,#lp-cov").on("input change",draw);
  draw();
})();

/* ---- Viz 3: planner cost crossover ---- */
(function(){
  const C=IX_C, svg=d3.select("#pl-svg"), W=640, H=250;
  const N=1e7, RPP=50, P=N/RPP, CPU=0.01, HT=4;          // 10M rows, 50 rows/page → 200k pages
  const px=62, py=16, pw=440, ph=190;
  function costs(s,r,corr){
    const k=Math.max(1,s*N);
    const seq=P*1+N*CPU;
    const index=corr? HT*r + r + (k/RPP)*1 + k*CPU : HT*r + k*r + k*CPU;
    const touched=corr? Math.min(P,Math.max(1,k/RPP)) : P*(1-Math.exp(-k/P));
    const perPage=r-(r-1)*Math.sqrt(touched/P);           // interpolate random → sequential as pages get dense
    const bitmap=HT*r + 20 + k*0.02 + touched*perPage + k*CPU;   // +fixed and per-entry cost of building/rechecking the bitmap
    return {seq,index,bitmap};
  }
  const NAMES={seq:"seq scan",index:"index scan",bitmap:"bitmap heap scan"}, COL={seq:C.bad,index:C.good,bitmap:C.accent};
  function draw(){
    const ls=+d3.select("#pl-s").property("value"), r=+d3.select("#pl-r").property("value"), corr=d3.select("#pl-corr").property("checked");
    const s=Math.pow(10,ls);
    const xs=d3.range(-6,0.001,0.05);
    const all=xs.map(x=>costs(Math.pow(10,x),r,corr));
    const ymax=Math.log10(d3.max(all,c=>Math.max(c.seq,c.index,c.bitmap))), ymin=1;
    const X=x=>px+(x+6)/6*pw, Y=c=>py+ph-(Math.log10(Math.max(c,10))-ymin)/(ymax-ymin)*ph;
    svg.selectAll("*").remove();
    svg.append("line").attr("x1",px).attr("y1",py+ph).attr("x2",px+pw).attr("y2",py+ph).attr("stroke","#3a4150");
    svg.append("line").attr("x1",px).attr("y1",py).attr("x2",px).attr("y2",py+ph).attr("stroke","#3a4150");
    [-6,-5,-4,-3,-2,-1,0].forEach(t=>{
      svg.append("text").attr("x",X(t)).attr("y",py+ph+13).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9.5)
        .text(t===0?"100%":t===-1?"10%":t===-2?"1%":"10"+["⁻⁶","⁻⁵","⁻⁴","⁻³"][t+6]);
    });
    svg.append("text").attr("x",px+pw/2).attr("y",H-6).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("selectivity s (fraction of 10M rows matched, log)");
    svg.append("text").attr("x",14).attr("y",py+ph/2).attr("fill",C.muted).attr("font-size",10).attr("transform",`rotate(-90,14,${py+ph/2})`).attr("text-anchor","middle").text("est. cost (log)");
    ["seq","index","bitmap"].forEach(key=>{
      svg.append("path").attr("d",d3.line().x((d,i)=>X(xs[i])).y(d=>Y(d[key]))(all)).attr("fill","none").attr("stroke",COL[key]).attr("stroke-width",2.2)
        .attr("stroke-dasharray",key==="bitmap"?"5 3":null);
    });
    // end-of-line labels, nudged apart so they never overlap
    const lab=["seq","index","bitmap"].map(key=>({key,y:Y(all[all.length-1][key])+4})).sort((a,b)=>a.y-b.y);
    for(let i=1;i<lab.length;i++) if(lab[i].y-lab[i-1].y<13) lab[i].y=lab[i-1].y+13;
    lab.forEach(l=>svg.append("text").attr("x",px+pw+8).attr("y",l.y).attr("fill",COL[l.key]).attr("font-size",10).text(NAMES[l.key]));
    const c=costs(s,r,corr), best=Object.keys(c).reduce((a,b)=>c[a]<=c[b]?a:b);
    svg.append("line").attr("x1",X(ls)).attr("y1",py).attr("x2",X(ls)).attr("y2",py+ph).attr("stroke",C.warn).attr("stroke-dasharray","3 3");
    Object.keys(c).forEach(key=>svg.append("circle").attr("cx",X(ls)).attr("cy",Y(c[key])).attr("r",key===best?6:3.5)
      .attr("fill",COL[key]).attr("stroke",key===best?C.ink:"none").attr("stroke-width",1.5));
    let cross=null; for(let i=1;i<xs.length;i++){ if(all[i-1].index<all[i-1].seq && all[i].index>=all[i].seq){cross=xs[i];break;} }
    const pct=v=>v>=0.01?(v*100).toFixed(1)+"%":v>=1e-4?(v*100).toFixed(3)+"%":(v*100).toExponential(1)+"%";
    d3.select("#pl-read").html(`s = <b>${pct(s)}</b> (${Math.round(s*N).toLocaleString()} rows) → planner picks <b style="color:${COL[best]}">${NAMES[best]}</b> · `+
      Object.keys(c).map(k=>`${NAMES[k]} ${Math.round(c[k]).toLocaleString()}`).join(" · ")+
      (cross!==null?` · plain index scan stops beating the seq scan near s ≈ <b>${pct(Math.pow(10,cross))}</b>`:` · the index scan never loses to the seq scan here`));
  }
  d3.selectAll("#pl-s,#pl-r,#pl-corr").on("input change",draw);
  draw();
})();
