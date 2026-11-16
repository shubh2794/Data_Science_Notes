/* etl-pipelines.viz.js — interactive figures for etl-pipelines.html.
   Loaded after data.js / notes.js. Uses d3 from ../vendor. */

/* ---- Viz 1: DAG run with retries; append vs overwrite under a mid-write failure ---- */
(function(){
  const C={accent:"#22d3ee",good:"#4ade80",bad:"#f87171",warm:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",panel:"#0e1016",line:"#2a2f3a",dim:"#6b7280"};
  const svg=d3.select("#etl-svg");
  if(svg.empty()) return;
  const T=[
    {id:"E1",name:"extract_orders",x:20,y:30,lvl:0},
    {id:"E2",name:"extract_customers",x:20,y:110,lvl:0},
    {id:"T",name:"transform_join",x:180,y:70,lvl:1},
    {id:"Q",name:"quality_checks",x:330,y:70,lvl:2},
    {id:"L",name:"load_mart",x:480,y:70,lvl:3}
  ];
  const EDGES=[["E1","T"],["E2","T"],["T","Q"],["Q","L"]];
  const BW=128, BH=36, EXPECTED=1000, PARTIAL=400;
  const byId=Object.fromEntries(T.map(t=>[t.id,t]));
  const COLOR={queued:C.line,running:C.accent,success:C.good,failed:C.bad,retry:C.warm,upstream:C.dim};
  const LABEL={queued:"queued",running:"running",success:"success",failed:"failed",retry:"up for retry",upstream:"upstream failed"};
  let rows=0, runNo=0, history=[], frames=[], fi=0, timer=null;
  let cur={state:Object.fromEntries(T.map(t=>[t.id,{s:"queued",a:0}])),rows:0,note:"Idle. Press run to execute the DAG for logical date 2026-11-16."};

  function simulate(){
    const fail=d3.select("#etl-fail").property("value"), retries=+d3.select("#etl-ret").property("value");
    const mode=d3.select("#etl-mode").property("value");
    const st=Object.fromEntries(T.map(t=>[t.id,{s:"queued",a:0}]));
    let r=rows; const out=[]; let retriesUsed=0, ok=true;
    const snap=note=>out.push({state:JSON.parse(JSON.stringify(st)),rows:r,note});
    snap(`run #${runNo}: scheduled for data interval 2026-11-16.`);
    for(let lvl=0;lvl<=3;lvl++){
      let pending=T.filter(t=>t.lvl===lvl).map(t=>t.id), attempt=1;
      while(pending.length){
        pending.forEach(id=>{st[id]={s:"running",a:attempt};});
        snap(`${pending.map(id=>byId[id].name).join(" + ")} running (attempt ${attempt}).`);
        const next=[]; const notes=[];
        pending.forEach(id=>{
          const fails=(fail==="persistent"&&id==="E1")||(fail==="transient"&&id==="L"&&attempt===1);
          if(id==="L"){
            if(fails){ if(mode==="append"){ r+=PARTIAL; notes.push(`load crashed after writing ${PARTIAL} rows straight into the table`); }
                       else notes.push("load crashed while writing to staging; the live partition is untouched"); }
            else { if(mode==="append"){ r+=EXPECTED; notes.push(`load appended ${EXPECTED} rows`); }
                   else { r=EXPECTED; notes.push(`load replaced partition dt=2026-11-16 atomically with ${EXPECTED} rows`); } }
          }
          if(fails){
            if(attempt<=retries){ st[id]={s:"retry",a:attempt}; next.push(id); retriesUsed++; }
            else st[id]={s:"failed",a:attempt};
          } else st[id]={s:"success",a:attempt};
        });
        snap(notes.length?notes.join("; ")+".":`attempt ${attempt} finished.`);
        pending=next; attempt++;
      }
      if(T.filter(t=>t.lvl===lvl).some(t=>st[t.id].s==="failed")){
        T.filter(t=>t.lvl>lvl).forEach(t=>{st[t.id]={s:"upstream",a:0};});
        ok=false; snap("A task exhausted its retries; downstream tasks are marked upstream-failed. The run fails loudly.");
        break;
      }
    }
    return {frames:out,final:{ok,retriesUsed,rows:r,mode}};
  }

  function draw(){
    svg.selectAll("*").remove();
    const defs=svg.append("defs");
    defs.append("marker").attr("id","etl-arr").attr("viewBox","0 0 10 10").attr("refX",9).attr("refY",5)
      .attr("markerWidth",6).attr("markerHeight",6).attr("orient","auto").append("path").attr("d","M0,0L10,5L0,10z").attr("fill",C.muted);
    EDGES.forEach(([a,b])=>{
      const A=byId[a],B=byId[b];
      svg.append("path").attr("d",d3.linkHorizontal()({source:[A.x+BW,A.y+BH/2],target:[B.x-3,B.y+BH/2]}))
        .attr("fill","none").attr("stroke",C.muted).attr("stroke-width",1.2).attr("marker-end","url(#etl-arr)");
    });
    T.forEach(t=>{
      const s=cur.state[t.id];
      svg.append("rect").attr("x",t.x).attr("y",t.y).attr("width",BW).attr("height",BH).attr("rx",7)
        .attr("fill",C.panel).attr("stroke",COLOR[s.s]).attr("stroke-width",s.s==="queued"?1:2.2)
        .attr("stroke-dasharray",s.s==="upstream"?"4 3":null);
      svg.append("text").attr("x",t.x+BW/2).attr("y",t.y+15).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",10.5).attr("font-family","monospace").text(t.name);
      svg.append("text").attr("x",t.x+BW/2).attr("y",t.y+29).attr("text-anchor","middle").attr("fill",COLOR[s.s]==C.line?C.muted:COLOR[s.s]).attr("font-size",9.5)
        .text(LABEL[s.s]+(s.a>0?` · try ${s.a}`:""));
    });
    // target table panel
    const py=178;
    svg.append("rect").attr("x",14).attr("y",py).attr("width",612).attr("height",78).attr("rx",9).attr("fill",C.panel).attr("stroke",C.line);
    svg.append("text").attr("x",26).attr("y",py+18).attr("fill",C.muted).attr("font-size",10.5).text("target: mart.daily_orders   partition dt = 2026-11-16");
    const X=d3.scaleLinear().domain([0,3600]).range([26,560]);
    svg.append("rect").attr("x",26).attr("y",py+30).attr("width",X(3600)-26).attr("height",18).attr("rx",4).attr("fill","#161a22");
    const over=cur.rows>EXPECTED;
    svg.append("rect").attr("x",26).attr("y",py+30).attr("width",Math.max(0,X(Math.min(cur.rows,3600))-26)).attr("height",18).attr("rx",4)
      .attr("fill",over?C.bad:cur.rows===EXPECTED?C.good:C.warm).attr("fill-opacity",.85);
    svg.append("line").attr("x1",X(EXPECTED)).attr("x2",X(EXPECTED)).attr("y1",py+25).attr("y2",py+54).attr("stroke",C.ink).attr("stroke-dasharray","3 2");
    svg.append("text").attr("x",X(EXPECTED)).attr("y",py+68).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",9.5).text("correct = 1,000 source rows");
    svg.append("text").attr("x",570).attr("y",py+44).attr("fill",over?C.bad:C.ink).attr("font-size",12).attr("font-weight",700).text(d3.format(",")(cur.rows));
    const hist=history.length?" · history: "+history.join(" | "):"";
    d3.select("#etl-read").html(`<span style="color:${C.accent}">▶</span> ${cur.note}${hist}`);
  }

  function play(){
    if(timer){clearInterval(timer);timer=null;}
    runNo++;
    const sim=simulate(); frames=sim.frames; fi=0; cur=frames[0]; draw();
    timer=setInterval(()=>{
      fi++;
      if(fi>=frames.length){ clearInterval(timer); timer=null;
        rows=sim.final.rows;
        const tag=sim.final.ok?(rows===EXPECTED?"✓":"⚠ duplicates"):"✗ failed";
        history.push(`#${runNo} ${sim.final.mode} → ${d3.format(",")(rows)} ${tag}`);
        if(history.length>3) history.shift();
        const verdict=!sim.final.ok?`Run failed after ${sim.final.retriesUsed} retr${sim.final.retriesUsed===1?"y":"ies"}; the partition still holds ${d3.format(",")(rows)} rows.`
          : rows===EXPECTED?`Run succeeded (${sim.final.retriesUsed} retr${sim.final.retriesUsed===1?"y":"ies"}). Partition holds exactly ${EXPECTED} rows — rerun it: it stays correct.`
          : `Run reported SUCCESS, but the partition holds ${d3.format(",")(rows)} rows for ${EXPECTED} source rows. Nothing alerted.`;
        cur={...cur,note:verdict}; draw(); return; }
      cur=frames[fi]; draw();
    },600);
  }
  d3.select("#etl-run").on("click",play);
  d3.select("#etl-reset").on("click",()=>{ if(timer){clearInterval(timer);timer=null;} rows=0; runNo=0; history=[];
    cur={state:Object.fromEntries(T.map(t=>[t.id,{s:"queued",a:0}])),rows:0,note:"Target partition emptied."}; draw(); });
  d3.select("#etl-ret").on("input",function(){ d3.select("#etl-retv").text(this.value); });
  draw();
})();

/* ---- Viz 2: SCD Type 1 vs Type 2 ---- */
(function(){
  const C={accent:"#22d3ee",good:"#4ade80",bad:"#f87171",warm:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",panel:"#0e1016",line:"#2a2f3a"};
  const svg=d3.select("#scd-svg");
  if(svg.empty()) return;
  const INIT={date:"2025-01-10",city:"Lisbon",segment:"Bronze"};
  const CHANGES=[
    {date:"2025-06-01",city:"Porto",segment:"Bronze",why:"customer moves to Porto"},
    {date:"2026-02-15",city:"Porto",segment:"Gold",why:"customer upgraded to Gold"},
    {date:"2026-09-30",city:"Berlin",segment:"Gold",why:"customer moves to Berlin"},
    {date:"2026-10-05",city:"Berlin",segment:"Gold",why:"duplicate change event re-delivered by CDC"}
  ];
  const FACT={date:"2025-08-20",amount:120};
  let k=0;
  function state(){
    let t1={...INIT,updated:INIT.date};
    let t2=[{sk:1,city:INIT.city,segment:INIT.segment,from:INIT.date,to:"9999-12-31",cur:true}];
    let lastNote="Initial load: one customer (id 42).";
    for(let i=0;i<k;i++){
      const c=CHANGES[i];
      t1={date:t1.date,city:c.city,segment:c.segment,updated:c.date};
      const curRow=t2[t2.length-1];
      if(curRow.city===c.city && curRow.segment===c.segment){
        lastNote=`${c.date}: ${c.why} — attribute hash unchanged, so Type 2 creates no new version (idempotent).`;
        continue;
      }
      curRow.to=c.date; curRow.cur=false;
      t2.push({sk:curRow.sk+1,city:c.city,segment:c.segment,from:c.date,to:"9999-12-31",cur:true});
      lastNote=`${c.date}: ${c.why} — Type 1 overwrites; Type 2 closes sk ${curRow.sk} and inserts sk ${curRow.sk+1}.`;
    }
    return {t1,t2,lastNote};
  }
  function draw(){
    svg.selectAll("*").remove();
    const S=state();
    // Type 1 table
    svg.append("text").attr("x",14).attr("y",18).attr("fill",C.warm).attr("font-size",11).attr("font-weight",600).text("Type 1 — overwrite");
    const h1=[["id",14],["city",44],["segment",104],["updated",168]];
    h1.forEach(([t,x])=>svg.append("text").attr("x",x).attr("y",40).attr("fill",C.muted).attr("font-size",10).text(t));
    svg.append("line").attr("x1",10).attr("x2",236).attr("y1",45).attr("y2",45).attr("stroke",C.line);
    [["42",14],[S.t1.city,44],[S.t1.segment,104],[S.t1.updated,168]].forEach(([t,x])=>
      svg.append("text").attr("x",x).attr("y",62).attr("fill",C.ink).attr("font-size",10.5).attr("font-family","monospace").text(t));
    // Type 2 table
    svg.append("text").attr("x",256).attr("y",18).attr("fill",C.accent).attr("font-size",11).attr("font-weight",600).text("Type 2 — versioned rows");
    const h2=[["sk",256],["city",284],["segment",344],["valid_from",410],["valid_to",494],["current",578]];
    h2.forEach(([t,x])=>svg.append("text").attr("x",x).attr("y",40).attr("fill",C.muted).attr("font-size",10).text(t));
    svg.append("line").attr("x1",252).attr("x2",630).attr("y1",45).attr("y2",45).attr("stroke",C.line);
    S.t2.forEach((r,i)=>{
      const y=62+i*20;
      const hit = r.from<=FACT.date && FACT.date<r.to;
      if(hit) svg.append("rect").attr("x",252).attr("y",y-13).attr("width",378).attr("height",18).attr("rx",3).attr("fill","rgba(74,222,128,.10)");
      [[r.sk,256],[r.city,284],[r.segment,344],[r.from,410],[r.to,494],[r.cur?"yes":"no",578]].forEach(([t,x])=>
        svg.append("text").attr("x",x).attr("y",y).attr("fill",r.cur?C.ink:C.muted).attr("font-size",10.5).attr("font-family","monospace").text(t));
    });
    // fact lookup panel
    const py=168;
    svg.append("rect").attr("x",10).attr("y",py).attr("width",620).attr("height",72).attr("rx",9).attr("fill",C.panel).attr("stroke",C.line);
    svg.append("text").attr("x",22).attr("y",py+18).attr("fill",C.muted).attr("font-size",10.5)
      .text(`fact: order on ${FACT.date}, amount ${FACT.amount} — "revenue by city at time of sale"`);
    const t2row=S.t2.find(r=>r.from<=FACT.date && FACT.date<r.to);
    const truth=t2row.city;   // Type 2 is correct by construction: the version valid on the fact date
    const t1ok=S.t1.city===truth;
    svg.append("text").attr("x",22).attr("y",py+40).attr("fill",t1ok?C.good:C.bad).attr("font-size",11)
      .text(`Type 1 report: ${S.t1.city}${t1ok?"  ✓":"  ✗ (history rewritten)"}`);
    svg.append("text").attr("x",22).attr("y",py+60).attr("fill",C.good).attr("font-size",11)
      .text(`Type 2 report (fact points at sk ${t2row.sk}): ${t2row.city}  ✓`);
    svg.append("text").attr("x",330).attr("y",py+50).attr("fill",C.muted).attr("font-size",10)
      .text(`customer lived in ${truth} on ${FACT.date}`);
    d3.select("#scd-read").html(`<span style="color:${C.accent}">▶</span> change ${k}/${CHANGES.length} · ${S.lastNote}`);
  }
  d3.select("#scd-next").on("click",()=>{ k=Math.min(CHANGES.length,k+1); draw(); });
  d3.select("#scd-reset").on("click",()=>{ k=0; draw(); });
  draw();
})();
