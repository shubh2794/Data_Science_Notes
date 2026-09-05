/* sql.viz.js — extracted from sql.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

const PC={accent:"#5b9cff",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",cell:"#1b2130",cellb:"#2a2f3a"};

/* ───────────────────────── 02 · logical execution order pipeline ───────────────────────── */
(function(){
  const svg=d3.select("#pipe-svg"),W=640,H=340;
  const out=document.getElementById("pipe-readout");
  // base rows: region, amt
  const base=[
    {id:0,region:"W",amt:5},
    {id:1,region:"E",amt:2},
    {id:2,region:"W",amt:4},
    {id:3,region:"E",amt:7},
    {id:4,region:"W",amt:1},
    {id:5,region:"E",amt:3}
  ];
  // toggle state
  const on={where:false,group:false,having:false,order:false,limit:false};
  let aliasErr=false;
  const x0=40, y0=70, rh=22, rg=4, colW=150;

  // compute the staged result given toggles
  function compute(){
    const stages=[]; // {name, rows, kind}
    let rows=base.map(r=>({region:r.region,amt:r.amt,label:r.region+" · "+r.amt}));
    stages.push({name:"FROM",rows:rows.slice(),kind:"row"});
    if(on.where){ rows=rows.filter(r=>r.region==="W"); }
    stages.push({name:"WHERE",rows:rows.slice(),kind:"row",active:on.where});
    let grouped=false;
    if(on.group){
      const m={}; rows.forEach(r=>{ m[r.region]=(m[r.region]||0)+r.amt; });
      rows=Object.keys(m).sort().map(k=>({region:k,amt:m[k],sum:m[k],label:k+" · Σ="+m[k]}));
      grouped=true;
    }
    stages.push({name:"GROUP BY",rows:rows.slice(),kind:grouped?"grp":"row",active:on.group});
    if(on.having && grouped){ rows=rows.filter(r=>r.sum>8); }
    stages.push({name:"HAVING",rows:rows.slice(),kind:grouped?"grp":"row",active:on.having&&grouped});
    // SELECT (identity for our display)
    stages.push({name:"SELECT",rows:rows.slice(),kind:grouped?"grp":"row",active:true});
    if(on.order){ rows=rows.slice().sort((a,b)=> (grouped? b.sum-a.sum : b.amt-a.amt)); }
    stages.push({name:"ORDER BY",rows:rows.slice(),kind:grouped?"grp":"row",active:on.order});
    if(on.limit){ rows=rows.slice(0,2); }
    stages.push({name:"LIMIT",rows:rows.slice(),kind:grouped?"grp":"row",active:on.limit});
    return stages;
  }

  function draw(){
    svg.selectAll("*").remove();
    // arrowhead def — appended first so marker-end references resolve
    const defs=svg.append("defs");
    defs.append("marker").attr("id","pah").attr("viewBox","0 0 10 10").attr("refX",8).attr("refY",5)
      .attr("markerWidth",6).attr("markerHeight",6).attr("orient","auto")
      .append("path").attr("d","M0,0 L10,5 L0,10 z").attr("fill",PC.muted);
    if(aliasErr){
      // error panel
      svg.append("rect").attr("x",30).attr("y",120).attr("width",W-60).attr("height",90).attr("rx",10)
        .attr("fill","rgba(248,113,113,.10)").attr("stroke",PC.bad);
      svg.append("text").attr("x",W/2).attr("y",150).attr("text-anchor","middle").attr("fill",PC.bad).attr("font-size",15).attr("font-weight",700)
        .text("✗ error: column \"total\" does not exist");
      svg.append("text").attr("x",W/2).attr("y",176).attr("text-anchor","middle").attr("fill",PC.ink).attr("font-size",12)
        .text("SELECT amt*2 AS total FROM s WHERE total > 5");
      svg.append("text").attr("x",W/2).attr("y",196).attr("text-anchor","middle").attr("fill",PC.muted).attr("font-size",11)
        .text("WHERE runs BEFORE SELECT — the alias does not exist yet");
      out.innerHTML=`<b style="color:${PC.bad}">alias-in-WHERE error</b> · WHERE is evaluated before SELECT, so the alias <code>total</code> isn't defined yet. Repeat the expression, or wrap in a subquery/CTE. <span style="color:var(--muted)">Press Reset.</span>`;
      return;
    }
    const stages=compute();
    const n=stages.length;
    const sw=(W-x0-20)/n;
    let maxRows=0; stages.forEach(s=>{ if(s.rows.length>maxRows)maxRows=s.rows.length; });

    stages.forEach((s,si)=>{
      const cx=x0+si*sw;
      const active = s.active!==false;
      // stage header
      svg.append("text").attr("x",cx+sw/2-6).attr("y",y0-30).attr("text-anchor","middle")
        .attr("fill",active?PC.accent:PC.muted).attr("font-size",10.5).attr("font-weight",active?700:400).text(s.name);
      // row-count badge
      svg.append("rect").attr("x",cx+sw/2-6-14).attr("y",y0-22).attr("width",28).attr("height",16).attr("rx",8)
        .attr("fill",active?"rgba(91,156,255,.18)":PC.cell).attr("stroke",active?PC.accent:PC.cellb);
      svg.append("text").attr("x",cx+sw/2-6).attr("y",y0-10).attr("text-anchor","middle")
        .attr("fill",active?PC.accent:PC.muted).attr("font-size",10).attr("font-weight",700).text(s.rows.length);
      // rows
      s.rows.forEach((r,ri)=>{
        const isGrp = s.kind==="grp";
        const ry=y0+ri*(rh+rg);
        svg.append("rect").attr("x",cx).attr("y",ry).attr("width",sw-12).attr("height",rh).attr("rx",5)
          .attr("fill", isGrp?"rgba(74,222,128,.16)":(r.region==="W"?"rgba(91,156,255,.16)":"rgba(255,180,84,.16)"))
          .attr("stroke", isGrp?PC.good:(r.region==="W"?PC.accent:PC.a2)).attr("stroke-width", isGrp?1.6:1);
        svg.append("text").attr("x",cx+8).attr("y",ry+rh/2+4).attr("fill",PC.ink).attr("font-size",11).text(r.label);
      });
      // arrow to next stage
      if(si<n-1){
        svg.append("path").attr("d",`M${cx+sw-11},${y0+8} l8,0`).attr("stroke",PC.muted).attr("stroke-width",1.5).attr("marker-end","url(#pah)");
      }
    });

    const fin=stages[stages.length-1].rows;
    const active=Object.keys(on).filter(k=>on[k]);
    out.innerHTML = `clauses on: <b style="color:${PC.accent}">${active.length?active.map(a=>a.toUpperCase()).join(" · "):"none (FROM only)"}</b> → <b>${fin.length}</b> output row${fin.length===1?"":"s"}. `
      + (on.group?`<span style="color:${PC.good}">GROUP BY collapsed rows into per-region summaries.</span>`:`<span style="color:var(--muted)">WHERE filters rows; HAVING needs GROUP BY; window/alias filtering needs a subquery.</span>`);
  }

  function tog(k,btn){ on[k]=!on[k]; aliasErr=false; document.getElementById(btn).classList.toggle("on",on[k]); draw(); }
  document.getElementById("cl-where").addEventListener("click",()=>tog("where","cl-where"));
  document.getElementById("cl-group").addEventListener("click",()=>tog("group","cl-group"));
  document.getElementById("cl-having").addEventListener("click",()=>tog("having","cl-having"));
  document.getElementById("cl-order").addEventListener("click",()=>tog("order","cl-order"));
  document.getElementById("cl-limit").addEventListener("click",()=>tog("limit","cl-limit"));
  document.getElementById("cl-alias").addEventListener("click",()=>{ aliasErr=true; draw(); });
  document.getElementById("cl-reset").addEventListener("click",()=>{ Object.keys(on).forEach(k=>{on[k]=false;}); aliasErr=false;
    ["cl-where","cl-group","cl-having","cl-order","cl-limit"].forEach(id=>document.getElementById(id).classList.remove("on")); draw(); });
  draw();
})();

/* ───────────────────────── 03 · JOIN fan-out explorer ───────────────────────── */
(function(){
  const svg=d3.select("#join-svg"),W=640,H=330;
  const out=document.getElementById("join-readout");
  // left = orders (key, amt), right = shipments (key)
  const L=[{k:1,amt:10},{k:2,amt:20},{k:3,amt:30}];
  let dup=false; // when true, key 2 appears twice on the right (fan-out)
  function rightTbl(){ return dup ? [{k:1},{k:2},{k:2},{k:4}] : [{k:1},{k:2},{k:4}]; }
  let jtype="inner";
  const lx=70, rx=420, y0=80, rh=34, rg=12, bw=150;

  function rows(){
    const R=rightTbl();
    // build output per join type
    let out=[];
    if(jtype==="cross"){
      L.forEach(l=>R.forEach(r=>out.push({l,r})));
      return {R,out};
    }
    L.forEach(l=>{
      const m=R.filter(r=>r.k===l.k);
      if(m.length) m.forEach(r=>out.push({l,r}));
      else if(jtype==="left"||jtype==="full") out.push({l,r:null});
    });
    R.forEach(r=>{
      const m=L.filter(l=>l.k===r.k);
      if(!m.length && (jtype==="right"||jtype==="full")) out.push({l:null,r});
    });
    return {R,out};
  }

  function draw(){
    svg.selectAll("*").remove();
    const {R,out:res}=rows();
    // headers
    svg.append("text").attr("x",lx).attr("y",y0-26).attr("fill",PC.accent).attr("font-size",12).attr("font-weight",700).text("orders (key, amt)");
    svg.append("text").attr("x",rx).attr("y",y0-26).attr("fill",PC.a2).attr("font-size",12).attr("font-weight",700).text("shipments (key)");
    // left rows
    const lpos={};
    L.forEach((l,i)=>{
      const y=y0+i*(rh+rg); lpos[i]=y+rh/2;
      svg.append("rect").attr("x",lx).attr("y",y).attr("width",bw).attr("height",rh).attr("rx",6)
        .attr("fill","rgba(91,156,255,.14)").attr("stroke",PC.accent);
      svg.append("text").attr("x",lx+12).attr("y",y+rh/2+4).attr("fill",PC.ink).attr("font-size",13).text("k="+l.k+"  amt="+l.amt);
    });
    // right rows
    const rpos={};
    R.forEach((r,i)=>{
      const y=y0+i*(rh+rg); rpos[i]=y+rh/2;
      const isDup = dup && r.k===2;
      svg.append("rect").attr("x",rx).attr("y",y).attr("width",bw-20).attr("height",rh).attr("rx",6)
        .attr("fill", isDup?"rgba(248,113,113,.16)":"rgba(255,180,84,.14)").attr("stroke", isDup?PC.bad:PC.a2);
      svg.append("text").attr("x",rx+12).attr("y",y+rh/2+4).attr("fill",PC.ink).attr("font-size",13).text("k="+r.k+(isDup?"  (dup)":""));
    });
    // match lines
    res.forEach(p=>{
      if(p.l && p.r){
        const li=L.indexOf(p.l), ri=R.indexOf(p.r);
        // for cross/dup, indexOf on r is fine because objects are reused
        const y1=lpos[li], y2=rpos[ri];
        const matched = p.l.k===p.r.k;
        svg.append("path").attr("d",`M${lx+bw},${y1} C${(lx+bw+rx)/2},${y1} ${(lx+bw+rx)/2},${y2} ${rx},${y2}`)
          .attr("fill","none").attr("stroke", matched?PC.good:PC.muted).attr("stroke-opacity", matched?.8:.3)
          .attr("stroke-width", matched?2:1).attr("stroke-dasharray", matched?null:"3 3");
      }
    });
    // SUM(amt) over output — double counts on fan-out
    let sum=0; res.forEach(p=>{ if(p.l) sum+=p.l.amt; });
    const trueSum=L.reduce((a,l)=>a+l.amt,0);
    const overcount = sum>trueSum && (jtype==="inner"||jtype==="left"||jtype==="right"||jtype==="full");
    // badges
    const by=y0+Math.max(L.length,R.length)*(rh+rg)+24;
    svg.append("text").attr("x",lx).attr("y",by).attr("fill",PC.muted).attr("font-size",12)
      .text("output rows: ").append("tspan").attr("fill",PC.ink).attr("font-weight",700).text(res.length);
    svg.append("rect").attr("x",lx+260).attr("y",by-16).attr("width",250).attr("height",24).attr("rx",6)
      .attr("fill", overcount?"rgba(248,113,113,.14)":"rgba(74,222,128,.12)").attr("stroke", overcount?PC.bad:PC.good);
    svg.append("text").attr("x",lx+272).attr("y",by).attr("fill", overcount?PC.bad:PC.good).attr("font-size",12).attr("font-weight",700)
      .text(`SUM(amt) over join = ${sum}` + (overcount?`  ✗ (true = ${trueSum})`:`  ✓`));

    let note;
    if(jtype==="cross") note=`CROSS JOIN → |L|·|R| = ${L.length}·${R.length} = ${res.length} rows (every pair, no key)`;
    else if(overcount) note=`fan-out! key 2 matches twice → amt=20 counted twice. Aggregate AFTER a fan-out join double-counts (SUM should be ${trueSum}).`;
    else note=`${jtype.toUpperCase()} JOIN on unique keys → one output row per match, SUM is correct.`;
    out.innerHTML=`<b style="color:${PC.accent}">${jtype.toUpperCase()}${jtype==="cross"?"":" JOIN"}</b> · <b>${res.length}</b> rows · ${dup?"<span style='color:"+PC.bad+"'>duplicate right key ON</span>":"unique right key"}<br><span style="color:var(--muted)">${note}</span>`;
  }

  function setJ(t){ jtype=t; draw(); }
  document.getElementById("j-inner").addEventListener("click",()=>setJ("inner"));
  document.getElementById("j-left").addEventListener("click",()=>setJ("left"));
  document.getElementById("j-right").addEventListener("click",()=>setJ("right"));
  document.getElementById("j-full").addEventListener("click",()=>setJ("full"));
  document.getElementById("j-cross").addEventListener("click",()=>setJ("cross"));
  document.getElementById("j-dup").addEventListener("click",()=>{ dup=!dup; draw(); });
  draw();
})();

/* ───────────────────────── 05 · GROUP BY vs WINDOW ───────────────────────── */
(function(){
  const svg=d3.select("#win-svg"),W=640,H=330;
  const out=document.getElementById("win-readout");
  // rows: region, val (ordered as given for running-total demo)
  const data=[
    {region:"W",val:5},
    {region:"E",val:2},
    {region:"W",val:4},
    {region:"E",val:7},
    {region:"W",val:1},
    {region:"E",val:3}
  ];
  const kcol={W:PC.accent,E:PC.a2};
  let mode="group"; // group | window
  let ordered=false;
  const x0=40, y0=64, rh=26, rg=6, bw=240;

  function draw(){
    svg.selectAll("*").remove();
    svg.append("text").attr("x",x0).attr("y",y0-30).attr("fill",PC.muted).attr("font-size",12).text("input rows (region, val)");
    svg.append("text").attr("x",x0+320).attr("y",y0-30).attr("fill", mode==="group"?PC.good:PC.accent).attr("font-size",12).attr("font-weight",700)
      .text(mode==="group"?"GROUP BY region → collapse":"WINDOW OVER (PARTITION BY region) → keep");

    // left: input rows
    data.forEach((r,i)=>{
      const y=y0+i*(rh+rg);
      svg.append("rect").attr("x",x0).attr("y",y).attr("width",bw).attr("height",rh).attr("rx",6)
        .attr("fill",kcol[r.region]).attr("fill-opacity",.15).attr("stroke",kcol[r.region]);
      svg.append("text").attr("x",x0+12).attr("y",y+rh/2+4).attr("fill",PC.ink).attr("font-size",13).text(r.region+"   val="+r.val);
    });

    const rx=x0+320;
    if(mode==="group"){
      // collapse to one summary per region
      const m={}; data.forEach(r=>{ m[r.region]=(m[r.region]||0)+r.val; });
      const keys=Object.keys(m).sort();
      keys.forEach((k,i)=>{
        const y=y0+i*(rh+rg);
        svg.append("rect").attr("x",rx).attr("y",y).attr("width",bw).attr("height",rh).attr("rx",6)
          .attr("fill",PC.good).attr("fill-opacity",.20).attr("stroke",PC.good).attr("stroke-width",1.6)
          .attr("opacity",0).transition().duration(450).attr("opacity",1);
        svg.append("text").attr("x",rx+12).attr("y",y+rh/2+4).attr("fill",PC.ink).attr("font-size",13).attr("font-weight",700)
          .attr("opacity",0).transition().duration(450).attr("opacity",1).text(k+"   SUM(val)="+m[k]);
      });
      // collapse hint lines
      out.innerHTML=`<b style="color:${PC.good}">GROUP BY</b> → ${keys.length} rows (one per region). The 6 input rows are <b>gone</b> — only the summary survives.`;
    } else {
      // window: one output row per input row, append the computed value
      // running total if ordered, else partition sum
      const running={};
      data.forEach((r,i)=>{
        const y=y0+i*(rh+rg);
        let v;
        if(ordered){ running[r.region]=(running[r.region]||0)+r.val; v=running[r.region]; }
        else { v=data.filter(d=>d.region===r.region).reduce((a,d)=>a+d.val,0); }
        svg.append("rect").attr("x",rx).attr("y",y).attr("width",bw).attr("height",rh).attr("rx",6)
          .attr("fill",PC.accent).attr("fill-opacity",.16).attr("stroke",PC.accent)
          .attr("opacity",0).transition().duration(350).delay(i*40).attr("opacity",1);
        svg.append("text").attr("x",rx+12).attr("y",y+rh/2+4).attr("fill",PC.ink).attr("font-size",12.5)
          .attr("opacity",0).transition().duration(350).delay(i*40).attr("opacity",1)
          .text(r.region+"  val="+r.val+"  →  "+(ordered?"run=":"part=")+v);
        // connector
        svg.append("line").attr("x1",x0+bw).attr("y1",y+rh/2).attr("x2",rx).attr("y2",y+rh/2)
          .attr("stroke",PC.muted).attr("stroke-opacity",.3).attr("stroke-dasharray","2 3");
      });
      out.innerHTML= ordered
        ? `<b style="color:${PC.accent}">WINDOW + ORDER BY</b> → all 6 rows kept, each gets a <b>running total</b> within its region (frame = UNBOUNDED PRECEDING … CURRENT ROW).`
        : `<b style="color:${PC.accent}">WINDOW</b> (no ORDER BY) → all 6 rows kept, each gets its region's <b>partition sum</b> (frame = whole partition). Add ORDER BY for a running total.`;
    }
  }

  document.getElementById("w-group").addEventListener("click",()=>{ mode="group"; draw(); });
  document.getElementById("w-window").addEventListener("click",()=>{ mode="window"; draw(); });
  document.getElementById("w-order").addEventListener("click",()=>{ ordered=!ordered; if(mode!=="window")mode="window"; draw(); });
  document.getElementById("w-reset").addEventListener("click",()=>{ mode="group"; ordered=false; draw(); });
  draw();
})();
