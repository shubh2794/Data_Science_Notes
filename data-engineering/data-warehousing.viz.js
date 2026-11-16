/* data-warehousing.viz.js — interactive figures for data-warehousing.html.
   Loaded after data.js / notes.js. Uses d3 from ../vendor. */

/* ---- Viz 1: row vs column store, with zone-map pruning ---- */
(function(){
  const C={accent:"#22d3ee",good:"#4ade80",bad:"#f87171",warm:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",panel:"#0e1016",line:"#2a2f3a",cell:"#1b2030"};
  const svg=d3.select("#cs-svg");
  if(svg.empty()) return;
  const COLS=[
    {name:"date",w:4,r:{sorted:40,unsorted:4}},
    {name:"customer",w:8,r:{sorted:3,unsorted:3}},
    {name:"product",w:8,r:{sorted:4,unsorted:4}},
    {name:"region",w:12,r:{sorted:12,unsorted:12}},
    {name:"revenue",w:8,r:{sorted:2,unsorted:2}},
    {name:"notes",w:60,r:{sorted:3,unsorted:3}}
  ];
  const G=12, N=1.2e9, NG=N/G, ROW_R=2;     // 12 row groups; row stores get ~2× general compression
  const MON=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  const gx=96, cw=72, cg=6, gy=34, rh=14.5;
  const fmt=b=>b>=1e12?(b/1e12).toFixed(2)+" TB":b>=1e9?(b/1e9).toFixed(1)+" GB":(b/1e6).toFixed(0)+" MB";
  function draw(){
    svg.selectAll("*").remove();
    const layout=d3.select("#cs-layout").property("value");
    const sel=new Set(d3.selectAll(".cs-col").nodes().filter(n=>n.checked).map(n=>+n.value));
    const from=+d3.select("#cs-from").property("value"), sorted=d3.select("#cs-sort").property("checked");
    d3.select("#cs-fromv").text(from+" ("+MON[from-1]+")");
    // which row groups can be skipped? clustered: group g holds month g+1 only; unclustered: every group spans Jan–Dec
    const keep=d3.range(G).map(g=> sorted ? (g+1)>=from : true);
    const readG=keep.filter(Boolean).length;
    // headers
    COLS.forEach((c,i)=>{
      const x=gx+i*(cw+cg);
      svg.append("text").attr("x",x+cw/2).attr("y",14).attr("text-anchor","middle").attr("fill",sel.has(i)?C.ink:C.muted).attr("font-size",10.5).attr("font-weight",sel.has(i)?700:400).text(c.name);
      svg.append("text").attr("x",x+cw/2).attr("y",26).attr("text-anchor","middle").attr("fill","#5f6878").attr("font-size",8.5).text(c.w+" B/value");
    });
    d3.range(G).forEach(g=>{
      const y=gy+g*rh;
      svg.append("text").attr("x",gx-6).attr("y",y+10).attr("text-anchor","end").attr("fill",keep[g]?C.muted:"#4b5263").attr("font-size",9)
        .text("rg "+(g+1)+" "+(sorted?MON[g]:"Jan–Dec"));
      if(layout==="row"){
        const read=keep[g];
        svg.append("rect").attr("x",gx).attr("y",y).attr("width",COLS.length*(cw+cg)-cg).attr("height",rh-3).attr("rx",2)
          .attr("fill",read?C.bad:C.cell).attr("fill-opacity",read?0.55:1);
      } else {
        COLS.forEach((c,i)=>{
          const read=keep[g]&&sel.has(i);
          svg.append("rect").attr("x",gx+i*(cw+cg)).attr("y",y).attr("width",cw).attr("height",rh-3).attr("rx",2)
            .attr("fill",read?C.good:C.cell).attr("fill-opacity",read?0.7:1);
        });
      }
      if(!keep[g]) svg.append("line").attr("x1",gx).attr("x2",gx+COLS.length*(cw+cg)-cg).attr("y1",y+5.5).attr("y2",y+5.5).attr("stroke","#4b5263").attr("stroke-dasharray","2 3");
    });
    // bytes
    const full=N*d3.sum(COLS,c=>c.w)/ROW_R;
    let bytes;
    if(layout==="row") bytes=readG*NG*d3.sum(COLS,c=>c.w)/ROW_R;
    else bytes=readG*NG*d3.sum([...sel],i=>COLS[i].w/COLS[i].r[sorted?"sorted":"unsorted"]);
    const by=gy+G*rh+14;
    const X=d3.scaleLinear().domain([0,full]).range([gx,gx+COLS.length*(cw+cg)-cg]);
    svg.append("text").attr("x",gx-6).attr("y",by+11).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",9.5).text("full scan");
    svg.append("rect").attr("x",gx).attr("y",by).attr("width",X(full)-gx).attr("height",14).attr("rx",3).attr("fill",C.cell);
    svg.append("text").attr("x",X(full)-4).attr("y",by+11).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",9.5).text(fmt(full));
    svg.append("text").attr("x",gx-6).attr("y",by+33).attr("text-anchor","end").attr("fill",C.ink).attr("font-size",9.5).text("this query");
    svg.append("rect").attr("x",gx).attr("y",by+22).attr("width",Math.max(2,X(bytes)-gx)).attr("height",14).attr("rx",3).attr("fill",layout==="row"?C.bad:C.good).attr("fill-opacity",.8);
    const lbl=fmt(bytes)+"  ("+(100*bytes/full).toFixed(bytes/full<0.01?2:1)+"% of full scan)";
    const inside=X(bytes)>380;
    svg.append("text").attr("x",inside?X(bytes)-6:Math.max(X(bytes),gx+2)+6).attr("y",by+33).attr("text-anchor",inside?"end":"start")
      .attr("fill",inside?"#06101f":C.ink).attr("font-size",10).attr("font-weight",inside?700:400).text(lbl);
    const why = layout==="row" ? "row layout reads every column of every surviving row group"
      : sel.size===0 ? "select at least one column"
      : `reads ${sel.size} of ${COLS.length} column chunks in ${readG} of ${G} row groups`;
    d3.select("#cs-read").html(`${why} · `+(sorted?`clustered: ${G-readG} row group${G-readG===1?"":"s"} skipped by min/max`:`<span style="color:${C.warm}">unclustered: every row group's date range is Jan–Dec, so nothing can be pruned</span>`)+
      (sel.has(5)&&layout==="col"?` · <span style="color:${C.warm}">the wide notes column dominates</span>`:""));
  }
  d3.selectAll("#cs-layout,.cs-col,#cs-sort").on("change",draw);
  d3.select("#cs-from").on("input",draw);
  draw();
})();

/* ---- Viz 2: lakehouse version history — COW vs MOR, time travel, vacuum ---- */
(function(){
  const C={accent:"#22d3ee",good:"#4ade80",bad:"#f87171",warm:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",panel:"#0e1016",line:"#2a2f3a",del:"#c084fc"};
  const svg=d3.select("#lh-svg");
  if(svg.empty()) return;
  const OPS=["create + append","append","DELETE customer 7","UPDATE prices","append","compact (optimize)"];
  const PLANS={
    cow:{files:[{id:"A",mb:100},{id:"B",mb:100},{id:"C",mb:100},{id:"B′",mb:99,note:"B rewritten"},{id:"A′",mb:100,note:"A rewritten"},{id:"D",mb:100},{id:"E",mb:399,note:"compacted"}],
         versions:[["A","B"],["A","B","C"],["A","B′","C"],["A′","B′","C"],["A′","B′","C","D"],["E"]],
         written:[200,100,99,100,100,399]},
    mor:{files:[{id:"A",mb:100},{id:"B",mb:100},{id:"C",mb:100},{id:"del-B",mb:1,del:true,note:"deletes in B"},{id:"del-A",mb:1,del:true,note:"old rows in A"},{id:"A-upd",mb:2,note:"new row versions"},{id:"D",mb:100},{id:"E",mb:399,note:"compacted"}],
         versions:[["A","B"],["A","B","C"],["A","B","C","del-B"],["A","B","C","del-B","del-A","A-upd"],["A","B","C","del-B","del-A","A-upd","D"],["E"]],
         written:[200,100,1,3,100,399]}
  };
  let vacuumed=false;
  function draw(){
    svg.selectAll("*").remove();
    const mode=d3.select("#lh-mode").property("value"), v=+d3.select("#lh-v").property("value");
    d3.select("#lh-vv").text(v);
    const P=PLANS[mode];
    const retained=new Set([...P.versions[4],...P.versions[5]]);
    const exists=f=>!vacuumed||retained.has(f);
    // timeline
    const tx=i=>60+i*104, ty=34;
    svg.append("line").attr("x1",tx(0)).attr("x2",tx(5)).attr("y1",ty).attr("y2",ty).attr("stroke",C.line).attr("stroke-width",2);
    OPS.forEach((op,i)=>{
      const readable=P.versions[i].every(exists);
      svg.append("circle").attr("cx",tx(i)).attr("cy",ty).attr("r",i===v?9:6)
        .attr("fill",!readable?"#3a2a2e":i===v?C.accent:"#1e222d").attr("stroke",!readable?C.bad:i===v?C.accent:C.muted).attr("stroke-width",1.5)
        .style("cursor","pointer").on("click",()=>{ d3.select("#lh-v").property("value",i); draw(); });
      svg.append("text").attr("x",tx(i)).attr("y",ty-14).attr("text-anchor","middle").attr("fill",i===v?C.ink:C.muted).attr("font-size",10).text("v"+i);
      svg.append("text").attr("x",tx(i)).attr("y",ty+24).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9).text(op);
      svg.append("text").attr("x",tx(i)).attr("y",ty+36).attr("text-anchor","middle").attr("fill",P.written[i]<10?C.good:"#6b7280").attr("font-size",8.5).text("wrote "+P.written[i]+" MB");
    });
    // files
    const live=new Set(P.versions[v]);
    const MINW=46, small=P.files.filter(f=>f.mb<MINW/2), big=P.files.filter(f=>f.mb>=MINW/2);
    const avail=600-(P.files.length-1)*6-small.length*MINW, bigTot=d3.sum(big,f=>f.mb);
    const width=f=>f.mb<MINW/2?MINW:Math.max(MINW,f.mb/bigTot*avail);
    let x=20; const fy=104;
    svg.append("text").attr("x",20).attr("y",fy-10).attr("fill",C.muted).attr("font-size",10).text("data files in object storage (width ∝ size) — highlighted = referenced by v"+v);
    P.files.forEach(f=>{
      const w=width(f);
      const gone=!exists(f.id), on=live.has(f.id);
      svg.append("rect").attr("x",x).attr("y",fy).attr("width",w).attr("height",36).attr("rx",4)
        .attr("fill",gone?"none":on?(f.del?C.del:C.good):"#1b2030").attr("fill-opacity",on?0.75:1)
        .attr("stroke",gone?C.bad:f.del?C.del:"#3a4150").attr("stroke-dasharray",gone?"3 3":null);
      svg.append("text").attr("x",x+w/2).attr("y",fy+16).attr("text-anchor","middle").attr("fill",gone?C.bad:on?"#06101f":C.ink).attr("font-size",10).attr("font-weight",600).text(f.id);
      svg.append("text").attr("x",x+w/2).attr("y",fy+29).attr("text-anchor","middle").attr("fill",gone?C.bad:on?"#06101f":C.muted).attr("font-size",8.5).text(gone?"vacuumed":f.mb+" MB");
      if(f.note) svg.append("text").attr("x",x+w/2).attr("y",fy+50+(f.mb<MINW/2&&P.files.indexOf(f)%2?11:0)).attr("text-anchor","middle").attr("fill","#6b7280").attr("font-size",8.5).text(f.note);
      x+=w+6;
    });
    // reader plan
    const py=180, readable=P.versions[v].every(exists);
    svg.append("rect").attr("x",14).attr("y",py).attr("width",612).attr("height",100).attr("rx",9).attr("fill",C.panel).attr("stroke",C.line);
    svg.append("text").attr("x",26).attr("y",py+20).attr("fill",C.ink).attr("font-size",11).attr("font-weight",600)
      .text(`SELECT … FROM sales VERSION AS OF ${v}`);
    const files=P.versions[v].map(id=>P.files.find(f=>f.id===id));
    const dels=files.filter(f=>f.del).length, data=files.length-dels;
    const lines = !readable ? [
        [`✗ cannot time-travel: v${v} references files removed by vacuum`,C.bad],
        ["vacuum kept only files needed by the latest 2 versions; older snapshots are gone for good",C.muted]]
      : [
        [`reads ${data} data file${data===1?"":"s"} (${d3.sum(files.filter(f=>!f.del),f=>f.mb)} MB)`+(dels?` + merges ${dels} delete file${dels>1?"s":""} at read time`:""),C.ink],
        [mode==="cow"?"copy-on-write: changes were paid for at write time by rewriting whole files; reads are plain scans"
                     :"merge-on-read: changes were cheap to write; every reader pays to apply delete files until compaction",C.muted],
        [v===5?"compaction rewrote everything into one file: fast reads again, old files now only serve time travel":
         `the log for v${v} lists exactly these files — readers never list the directory`,C.muted]];
    lines.forEach(([t,c],i)=>svg.append("text").attr("x",26).attr("y",py+42+i*19).attr("fill",c).attr("font-size",10.5).text(t));
    const stored=d3.sum(P.files.filter(f=>exists(f.id)),f=>f.mb);
    d3.select("#lh-read").html(`storage held: <b>${stored} MB</b> for a ~400 MB table`+(vacuumed?" (after vacuum)":" — old versions keep their files alive until vacuum")+
      ` · total bytes written over 6 commits: <b>${d3.sum(P.written)} MB</b> (${mode==="cow"?"copy-on-write":"merge-on-read"})`);
    d3.select("#lh-vac").text(vacuumed?"restore files (reset demo)":"vacuum (retain latest 2)");
  }
  d3.select("#lh-v").on("input",draw);
  d3.select("#lh-mode").on("change",draw);
  d3.select("#lh-vac").on("click",()=>{ vacuumed=!vacuumed; draw(); });
  draw();
})();
