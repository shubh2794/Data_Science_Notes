/* knowledge-graphs.viz.js — extracted from knowledge-graphs.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",blue:"#5b9cff",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#kg-svg"),W=640,H=260;
  const data=[
    {triples:[["Marie Curie","won","Nobel Prize"],["Nobel Prize","field","Physics"]],
     pos:{"Marie Curie":[130,80],"Nobel Prize":[380,80],"Physics":[500,190]}},
    {triples:[["Apple","founded by","Steve Jobs"],["Apple","located in","California"]],
     pos:{"Apple":[140,130],"Steve Jobs":[420,70],"California":[440,200]}},
  ];
  function draw(){
    svg.selectAll("*").remove();
    const d=data[+d3.select("#kg-sel").property("value")];
    // edges
    d.triples.forEach(([h,r,t])=>{
      const a=d.pos[h], b=d.pos[t];
      svg.append("line").attr("x1",a[0]).attr("y1",a[1]).attr("x2",b[0]).attr("y2",b[1]).attr("stroke","#3a4150").attr("stroke-width",1.5).attr("marker-end","url(#kga)");
      const mx=(a[0]+b[0])/2, my=(a[1]+b[1])/2;
      svg.append("rect").attr("x",mx-r.length*3.4-6).attr("y",my-9).attr("width",r.length*6.8+12).attr("height",17).attr("rx",8).attr("fill","#15181f").attr("stroke",C.accent).attr("stroke-opacity",.5);
      svg.append("text").attr("x",mx).attr("y",my+3).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.a2).text(r);
    });
    // arrow marker
    const defs=svg.append("defs"); defs.append("marker").attr("id","kga").attr("viewBox","0 -5 10 10").attr("refX",20).attr("refY",0).attr("markerWidth",7).attr("markerHeight",7).attr("orient","auto").append("path").attr("d","M0,-4L8,0L0,4").attr("fill","#3a4150");
    // nodes
    Object.entries(d.pos).forEach(([id,p])=>{
      const w=id.length*7.5+20;
      svg.append("rect").attr("x",p[0]-w/2).attr("y",p[1]-16).attr("width",w).attr("height",32).attr("rx",16).attr("fill","rgba(192,132,252,.16)").attr("stroke",C.accent).attr("stroke-width",1.5);
      svg.append("text").attr("x",p[0]).attr("y",p[1]+5).attr("text-anchor","middle").attr("font-size",12).attr("fill",C.ink).text(id);
    });
    d3.select("#kg-read").html(`<b>${Object.keys(d.pos).length}</b> entities · <b>${d.triples.length}</b> relations: `+d.triples.map(t=>`(${t[0]}, <span style="color:${C.a2}">${t[1]}</span>, ${t[2]})`).join("  "));
  }
  d3.select("#kg-sel").on("change",draw);
  draw();
})();

/* ───────────────────────── 05 · TransE scoring + link prediction ───────────────────────── */
(function(){
  const C={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",bad:"#f87171",blue:"#5b9cff",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  // entity vectors in 2-D, laid out so relations act as ~consistent translations
  const ent={
    Paris:[2.0,7.0], France:[2.4,2.0],
    Rome:[6.5,7.2], Italy:[6.9,2.2],
    Berlin:[9.6,7.4], Germany:[9.9,2.4],
    Tokyo:[5.0,8.6], Japan:[5.4,3.6],
  };
  // relations as translation vectors (approx)
  const rels={
    "capital_of":[0.4,-5.0],   // capital → country
    "located_in":[0.4,-5.0],
  };
  const heads=["Paris","Rome","Berlin","Tokyo"];
  const tails=["France","Italy","Germany","Japan"];
  const truth={Paris:"France",Rome:"Italy",Berlin:"Germany",Tokyo:"Japan"};

  const svg=d3.select("#te-svg"),W=640,H=300, plotW=380;
  const m={t:18,r:16,b:24,l:34};
  const xs=d3.scaleLinear().domain([0,12]).range([m.l,plotW-m.r]);
  const ys=d3.scaleLinear().domain([0,11]).range([H-m.b,m.t]);

  const selH=d3.select("#te-head"), selR=d3.select("#te-rel");
  selH.selectAll("option").data(heads).join("option").attr("value",d=>d).text(d=>d);
  selR.selectAll("option").data(Object.keys(rels)).join("option").attr("value",d=>d).text(d=>d);

  const gGrid=svg.append("g"), gLink=svg.append("g"), gPts=svg.append("g"), gBars=svg.append("g");
  // grid
  for(let gx=0;gx<=12;gx+=2) gGrid.append("line").attr("x1",xs(gx)).attr("x2",xs(gx)).attr("y1",ys(0)).attr("y2",ys(11)).attr("stroke",C.line).attr("stroke-opacity",.4);
  for(let gy=0;gy<=11;gy+=2) gGrid.append("line").attr("x1",xs(0)).attr("x2",xs(12)).attr("y1",ys(gy)).attr("y2",ys(gy)).attr("stroke",C.line).attr("stroke-opacity",.4);
  // arrow marker
  const defs=svg.append("defs");
  defs.append("marker").attr("id","tea").attr("viewBox","0 -5 10 10").attr("refX",8).attr("refY",0).attr("markerWidth",7).attr("markerHeight",7).attr("orient","auto").append("path").attr("d","M0,-4L9,0L0,4").attr("fill",C.a2);

  function draw(){
    const h=selH.property("value"), rel=selR.property("value");
    const hv=ent[h], rv=rels[rel];
    const target=[hv[0]+rv[0], hv[1]+rv[1]];   // h + r
    const dist=t=>Math.hypot(ent[t][0]-target[0], ent[t][1]-target[1]);
    const ranked=tails.map(t=>({t,d:dist(t),s:-dist(t)})).sort((a,b)=>a.d-b.d);
    const best=ranked[0];

    // ── plot: entities ──
    gLink.selectAll("*").remove();
    // translation arrow h -> h+r
    gLink.append("line").attr("x1",xs(hv[0])).attr("y1",ys(hv[1])).attr("x2",xs(target[0])).attr("y2",ys(target[1]))
      .attr("stroke",C.a2).attr("stroke-width",2).attr("stroke-dasharray","5 3").attr("marker-end","url(#tea)");
    // line from target to nearest tail
    gLink.append("line").attr("x1",xs(target[0])).attr("y1",ys(target[1])).attr("x2",xs(ent[best.t][0])).attr("y2",ys(ent[best.t][1]))
      .attr("stroke",C.good).attr("stroke-width",1.5).attr("stroke-opacity",.7);
    // target star (h+r)
    const star="M0,-9 L2.6,-2.6 L9,-2.6 L3.7,1.4 L5.6,8 L0,3.8 L-5.6,8 L-3.7,1.4 L-9,-2.6 L-2.6,-2.6 Z";
    gLink.append("path").attr("d",star).attr("transform",`translate(${xs(target[0])},${ys(target[1])})`).attr("fill",C.a2).attr("stroke","#0f1117").attr("stroke-width",1);
    gLink.append("text").attr("x",xs(target[0])+12).attr("y",ys(target[1])-6).attr("font-size",10).attr("fill",C.a2).text("h + r");

    const pts=gPts.selectAll("g.te").data(Object.keys(ent),d=>d);
    const en=pts.enter().append("g").attr("class","te");
    en.append("circle").attr("r",6).attr("stroke","#0f1117").attr("stroke-width",1.4);
    en.append("text").attr("font-size",10).attr("paint-order","stroke").attr("stroke","#0f1117").attr("stroke-width",3).attr("stroke-linejoin","round");
    const all=en.merge(pts);
    all.attr("transform",d=>`translate(${xs(ent[d][0])},${ys(ent[d][1])})`);
    all.select("circle").attr("fill",d=> d===h ? C.accent : d===best.t ? C.good : tails.includes(d) ? C.blue : "#3a4150");
    all.select("text").attr("x",9).attr("dy",4).attr("fill",d=> d===h||d===best.t ? C.ink : C.muted).text(d=>d);
    pts.exit().remove();

    // ── ranked bars (link prediction) ──
    const bx0=plotW+18, bw=W-bx0-90, maxD=d3.max(ranked,d=>d.d)||1;
    const by=d3.scaleBand().domain(ranked.map(d=>d.t)).range([m.t+8,H-m.b-8]).padding(0.3);
    const bx=d3.scaleLinear().domain([0,maxD*1.05]).range([0,bw]);
    svg.selectAll("text.bttl").data([0]).join("text").attr("class","bttl").attr("x",bx0).attr("y",m.t).attr("font-size",10).attr("fill",C.muted).text("candidate tails by distance ‖h+r−t‖");

    const rows=gBars.selectAll("g.br").data(ranked,d=>d.t);
    const ren=rows.enter().append("g").attr("class","br");
    ren.append("rect").attr("class","bg").attr("height",by.bandwidth()).attr("rx",3).attr("fill","#15181f");
    ren.append("rect").attr("class","fg").attr("height",by.bandwidth()).attr("rx",3);
    ren.append("text").attr("class","lab").attr("font-size",10);
    ren.append("text").attr("class","val").attr("font-size",10);
    const rall=ren.merge(rows);
    rall.attr("transform",d=>`translate(${bx0},${by(d.t)})`);
    rall.select("rect.bg").attr("width",bw);
    rall.select("rect.fg").transition().duration(400).attr("width",d=>Math.max(2,bx(d.d))).attr("fill",d=>d.t===best.t?C.good:C.blue).attr("fill-opacity",.85);
    rall.select("text.lab").attr("x",-6).attr("y",by.bandwidth()/2+3).attr("text-anchor","end").attr("fill",d=>d.t===best.t?C.good:C.muted).attr("font-weight",d=>d.t===best.t?700:400).text(d=>d.t);
    rall.select("text.val").attr("x",d=>Math.max(2,bx(d.d))+5).attr("y",by.bandwidth()/2+3).attr("fill",C.ink).text(d=>d.d.toFixed(2));
    rows.exit().remove();

    const ok = best.t===truth[h];
    d3.select("#te-read").html(`f(${h}, <span style="color:${C.a2}">${rel}</span>, t) = −‖h+r−t‖ &nbsp;·&nbsp; best <b style="color:${C.good}">${best.t}</b> &nbsp; score <b>${best.s.toFixed(2)}</b>`);
    d3.select("#te-rank").html(`predicted tail: <b style="color:${C.good}">${best.t}</b> &nbsp; ${ok?`<span style="color:${C.good}">✓ matches the known fact</span>`:`<span style="color:${C.bad}">(ground truth: ${truth[h]})</span>`}`);
  }
  selH.on("change",draw); selR.on("change",draw);
  draw();
})();

/* ───────────────────────── 04 · table row → typed triples ───────────────────────── */
(function(){
  const svgEl=document.getElementById("map-svg"), rowSel=document.getElementById("map-row"), modeSel=document.getElementById("map-mode");
  if(!svgEl||!rowSel||!modeSel||typeof d3==="undefined") return;
  const C={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",bad:"#f87171",blue:"#5b9cff",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a",panel:"#15181f"};
  const svg=d3.select(svgEl);
  const cols=["id","name","works_for","born","reports_to"];
  const people=[
    {id:101,name:"Ada Ruiz",works_for:7,born:"1984-03-02",reports_to:102},
    {id:102,name:"Ben Osei",works_for:7,born:"1979-11-20",reports_to:null},
    {id:103,name:"Chen Li",works_for:9,born:"c. 1990",reports_to:101},
  ];
  const orgs=[{id:7,name:"Orbital Labs"},{id:8,name:"Kestrel Bio"}];
  const isDate=s=>/^\d{4}-\d{2}-\d{2}$/.test(String(s));
  const personIds=new Set(people.map(p=>p.id)), orgIds=new Set(orgs.map(o=>o.id));

  function triples(r,mode){
    const out=[];
    if(mode==="direct"){
      const s=`<person/id=${r.id}>`;
      out.push({col:"id",s,p:"rdf:type",o:"<person>",kind:"iri"});
      out.push({col:"name",s,p:"person#name",o:`"${r.name}"`,kind:"lit"});
      out.push({col:"works_for",s,p:"person#ref-works_for",o:`<org/id=${r.works_for}>`,kind:"iri"});
      out.push({col:"born",s,p:"person#born",o:`"${r.born}"`,kind:"lit"});
      if(r.reports_to!=null) out.push({col:"reports_to",s,p:"person#ref-reports_to",o:`<person/id=${r.reports_to}>`,kind:"iri"});
      else out.push({col:"reports_to",s:"",p:"",o:"",kind:"none"});
      return out;
    }
    const s=`ex:person/${r.id}`;
    out.push({col:"id",s,p:"rdf:type",o:"ex:Person",kind:"iri",ok:true});
    out.push({col:"name",s,p:"ex:name",o:`"${r.name}"`,kind:"lit",ok:true});
    const okOrg=orgIds.has(r.works_for);
    out.push({col:"works_for",s,p:"ex:worksFor",o:`ex:org/${r.works_for}`,kind:"iri",ok:okOrg,why:okOrg?"":`worksFor range unmet: no ex:org/${r.works_for}`});
    const okD=isDate(r.born);
    out.push({col:"born",s,p:"ex:birthDate",o:okD?`"${r.born}"^^xsd:date`:`"${r.born}"`,kind:"lit",ok:okD,why:okD?"":`"${r.born}" is not a valid xsd:date`});
    if(r.reports_to!=null){
      const okP=personIds.has(r.reports_to);
      out.push({col:"reports_to",s,p:"ex:reportsTo",o:`ex:person/${r.reports_to}`,kind:"iri",ok:okP});
    } else out.push({col:"reports_to",s:"",p:"",o:"",kind:"none"});
    return out;
  }

  const tx=12, ty=40, cw=[38,72,70,82,72], rh=24;
  const cx=[tx]; cw.forEach((w,i)=>cx.push(cx[i]+w));
  function draw(){
    const ri=+rowSel.value, mode=modeSel.value, r=people[ri];
    const T=triples(r,mode);
    svg.selectAll("*").remove();
    // person table
    svg.append("text").attr("x",tx).attr("y",ty-10).attr("font-size",11).attr("fill",C.muted).text("table person  (PK id; FK works_for → org, reports_to → person)");
    cols.forEach((c,j)=>{
      svg.append("rect").attr("x",cx[j]).attr("y",ty).attr("width",cw[j]).attr("height",rh).attr("fill","#1c2030").attr("stroke",C.line);
      svg.append("text").attr("x",cx[j]+5).attr("y",ty+16).attr("font-size",10).attr("fill",C.ink).attr("font-weight",600).text(c);
    });
    people.forEach((p,i)=>{
      const y=ty+rh*(i+1), sel=i===ri;
      cols.forEach((c,j)=>{
        const v=p[c];
        svg.append("rect").attr("x",cx[j]).attr("y",y).attr("width",cw[j]).attr("height",rh)
          .attr("fill",sel?"rgba(192,132,252,.16)":C.panel).attr("stroke",sel?C.accent:C.line);
        svg.append("text").attr("x",cx[j]+5).attr("y",y+16).attr("font-size",10)
          .attr("fill",v==null?C.muted:(sel?C.ink:C.muted)).attr("font-style",v==null?"italic":"normal").text(v==null?"null":v);
      });
    });
    // org table
    const oy=ty+rh*5+14;
    svg.append("text").attr("x",tx).attr("y",oy-8).attr("font-size",11).attr("fill",C.muted).text("table org");
    [["id","name"]].concat(orgs.map(o=>[o.id,o.name])).forEach((row,i)=>{
      row.forEach((v,j)=>{
        const x=tx+(j?38:0), w=j?110:38;
        svg.append("rect").attr("x",x).attr("y",oy+rh*i).attr("width",w).attr("height",rh).attr("fill",i?C.panel:"#1c2030").attr("stroke",C.line);
        svg.append("text").attr("x",x+5).attr("y",oy+rh*i+16).attr("font-size",10).attr("fill",i?C.muted:C.ink).attr("font-weight",i?400:600).text(v);
      });
    });
    // triples
    const X0=372, ry=ty-10;
    svg.append("text").attr("x",X0).attr("y",ry-14).attr("font-size",11).attr("fill",C.muted).text(mode==="direct"?"emitted triples (table/column vocabulary)":"emitted triples (shared ontology vocabulary)");
    svg.append("text").attr("x",X0).attr("y",ry).attr("font-size",10).attr("font-family","SF Mono, Menlo, monospace").attr("fill",C.ink).text("s = "+(T[0]&&T[0].s));
    const selY=ty+rh*(ri+1)+rh/2;
    T.forEach((t,k)=>{
      const y=ty+8+k*44;
      const j=cols.indexOf(t.col);
      // mapping link from cell to triple
      svg.append("path").attr("d",`M${cx[j]+cw[j]/2},${selY+rh/2} C${cx[j]+cw[j]/2},${selY+60} ${X0-40},${y+12} ${X0-4},${y+12}`)
        .attr("fill","none").attr("stroke",t.kind==="none"?C.line:(t.ok===false?C.bad:C.accent)).attr("stroke-opacity",.55).attr("stroke-width",1.2)
        .attr("stroke-dasharray",t.kind==="none"?"3 3":null);
      if(t.kind==="none"){
        svg.append("text").attr("x",X0).attr("y",y+16).attr("font-size",10).attr("fill",C.muted).attr("font-style","italic").text(`${t.col} is null → no triple (unknown, not false)`);
        return;
      }
      svg.append("rect").attr("x",X0).attr("y",y).attr("width",252).attr("height",24).attr("rx",5).attr("fill",C.panel)
        .attr("stroke",t.ok===false?C.bad:(t.ok?C.good:C.line)).attr("stroke-opacity",.8);
      const txt=svg.append("text").attr("x",X0+6).attr("y",y+16).attr("font-size",9.5).attr("font-family","SF Mono, Menlo, monospace");
      txt.append("tspan").attr("fill",C.ink).text("s  ");
      txt.append("tspan").attr("fill",C.a2).text(t.p+"  ");
      txt.append("tspan").attr("fill",t.kind==="lit"?C.good:C.blue).text(t.o);
      if(t.ok!==undefined) svg.append("text").attr("x",X0+258).attr("y",y+16).attr("font-size",12).attr("fill",t.ok?C.good:C.bad).text(t.ok?"✓":"✗");
      if(t.why) svg.append("text").attr("x",X0+6).attr("y",y+36).attr("font-size",9).attr("fill",C.bad).text(t.why);
    });
    // legend
    const ly=288;
    [["IRI (entity)",C.blue],["literal (value)",C.good],["predicate",C.a2]].forEach(([l,c],i)=>{
      svg.append("circle").attr("cx",tx+6+i*118).attr("cy",ly-4).attr("r",4).attr("fill",c);
      svg.append("text").attr("x",tx+14+i*118).attr("y",ly).attr("font-size",10).attr("fill",C.muted).text(l);
    });
    const emitted=T.filter(t=>t.kind!=="none"), bad=emitted.filter(t=>t.ok===false);
    let msg=`<b>${emitted.length}</b> triples from 1 row`;
    if(mode==="direct") msg+=` · direct mapping: lossless but table-shaped — <code>person#ref-works_for</code> is not the same predicate as another source's <code>staff#company</code>, and <code>born</code> is an untyped string, so nothing can be checked.`;
    else if(bad.length) msg+=` · <span style="color:${C.bad}"><b>${bad.length}</b> shape violation${bad.length>1?"s":""}</span>: ${bad.map(b=>b.why).join("; ")}. Quarantine the row rather than load it — a reasoner would not object (open world), only the closed-world shape check does.`;
    else msg+=` · <span style="color:${C.good}">all shapes pass</span>: rows become typed entities, foreign keys become edges to other entities, values become typed literals.`;
    d3.select("#map-read").html(msg);
  }
  rowSel.addEventListener("change",draw); modeSel.addEventListener("change",draw);
  draw();
})();

/* ───────────────────────── 05 · entity linking: prior vs context vs coherence ───────────────────────── */
(function(){
  const svgEl=document.getElementById("el-svg"), sel=document.getElementById("el-sel"),
        lamEl=document.getElementById("el-lam"), tauEl=document.getElementById("el-tau"), cohBtn=document.getElementById("el-coh");
  if(!svgEl||!sel||!lamEl||!tauEl||!cohBtn||typeof d3==="undefined") return;
  const C={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",bad:"#f87171",blue:"#5b9cff",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a",panel:"#15181f"};
  const svg=d3.select(svgEl), W=640, H=360, BETA=1.0;
  // p = prior P(e|m) from an alias table; s = context similarity sim(ctx(m), desc(e)) ∈ [0,1]
  const S=[
    {text:"Jordan flew from Paris to Amman for talks.",
     M:[{m:"Jordan",c:[{e:"Jordan (basketball player)",p:.55,s:.15},{e:"Jordan (country)",p:.30,s:.55},{e:"Jordan River",p:.08,s:.35}]},
        {m:"Paris",c:[{e:"Paris (France)",p:.85,s:.50},{e:"Paris (mythology)",p:.05,s:.10},{e:"Paris, Texas",p:.04,s:.20}]},
        {m:"Amman",c:[{e:"Amman (city)",p:.95,s:.60},{e:"Amman Governorate",p:.04,s:.45}]}],
     rel:{"Jordan (country)|Amman (city)":.9,"Jordan (country)|Amman Governorate":.85,"Jordan River|Amman (city)":.4,"Jordan River|Amman Governorate":.4,
          "Jordan (country)|Paris (France)":.3,"Paris (France)|Amman (city)":.25,"Paris (France)|Amman Governorate":.1,"Jordan (basketball player)|Paris (France)":.1}},
    {text:"Jordan scored 40 as Chicago beat Boston.",
     M:[{m:"Jordan",c:[{e:"Jordan (basketball player)",p:.55,s:.70},{e:"Jordan (country)",p:.30,s:.05},{e:"Jordan River",p:.08,s:.02}]},
        {m:"Chicago",c:[{e:"Chicago (city)",p:.80,s:.15},{e:"Chicago Bulls (team)",p:.12,s:.80},{e:"Chicago (musical)",p:.05,s:.05}]},
        {m:"Boston",c:[{e:"Boston (city)",p:.85,s:.15},{e:"Boston Celtics (team)",p:.10,s:.80},{e:"Boston (band)",p:.04,s:.10}]}],
     rel:{"Jordan (basketball player)|Chicago Bulls (team)":.95,"Jordan (basketball player)|Boston Celtics (team)":.7,"Chicago Bulls (team)|Boston Celtics (team)":.9,
          "Chicago (city)|Boston (city)":.4,"Jordan (basketball player)|Chicago (city)":.3,"Jordan (basketball player)|Boston (city)":.3,
          "Chicago Bulls (team)|Boston (city)":.2,"Chicago (city)|Boston Celtics (team)":.2,"Chicago (musical)|Boston (band)":.15}},
    {text:"Apple unveiled the phone in Paris, near the Seine.",
     M:[{m:"Apple",c:[{e:"Apple (company)",p:.60,s:.70},{e:"apple (fruit)",p:.35,s:.10},{e:"Apple Records",p:.05,s:.15}]},
        {m:"Paris",c:[{e:"Paris (France)",p:.85,s:.55},{e:"Paris (mythology)",p:.05,s:.05},{e:"Paris, Texas",p:.04,s:.10}]},
        {m:"Seine",c:[{e:"Seine (river)",p:.90,s:.60},{e:"Seine (former department)",p:.10,s:.30}]}],
     rel:{"Paris (France)|Seine (river)":.9,"Paris (France)|Seine (former department)":.8,"Apple (company)|Paris (France)":.15,
          "apple (fruit)|Paris (France)":.05,"Paris, Texas|Seine (river)":.02}},
    {text:"Priya Okonkwo joined Apple's Paris office.",
     M:[{m:"Priya Okonkwo",c:[{e:"Priya (film)",p:.01,s:.05},{e:"Okonkwo (character)",p:.01,s:.12}]},
        {m:"Apple",c:[{e:"Apple (company)",p:.60,s:.80},{e:"apple (fruit)",p:.35,s:.05},{e:"Apple Records",p:.05,s:.20}]},
        {m:"Paris",c:[{e:"Paris (France)",p:.85,s:.55},{e:"Paris, Texas",p:.04,s:.15},{e:"Paris (mythology)",p:.05,s:.02}]}],
     rel:{"Apple (company)|Paris (France)":.15,"apple (fruit)|Paris (France)":.05,"Priya (film)|Paris (France)":.03,"Okonkwo (character)|Apple (company)":.02}},
  ];
  let coh=false;
  const relOf=(d,a,b)=>{ if(a===b) return 0; return d.rel[a+"|"+b] ?? d.rel[b+"|"+a] ?? 0.05; };
  const local=(c,lam)=>lam*Math.log(Math.max(c.p,.01))+(1-lam)*c.s;

  function solve(d,lam,tau){
    const n=d.M.length;
    const loc=d.M.map(x=>x.c.map(c=>local(c,lam)));
    let active=d.M.map(()=>true), pick=loc.map(L=>L.indexOf(Math.max(...L)));
    for(let pass=0;pass<3;pass++){
      if(coh){
        // exact joint MAP over active mentions (small: ≤ 3^3 combos)
        const idx=d.M.map((_,i)=>i).filter(i=>active[i]);
        let best=-Infinity, bestA=null;
        const rec=(k,acc)=>{
          if(k===idx.length){
            let v=0; idx.forEach((i,a)=>v+=loc[i][acc[a]]);
            for(let a=0;a<idx.length;a++) for(let b=a+1;b<idx.length;b++) v+=BETA*relOf(d,d.M[idx[a]].c[acc[a]].e,d.M[idx[b]].c[acc[b]].e);
            if(v>best){best=v;bestA=acc.slice();}
            return;
          }
          d.M[idx[k]].c.forEach((_,ci)=>{acc.push(ci);rec(k+1,acc);acc.pop();});
        };
        rec(0,[]);
        idx.forEach((i,a)=>pick[i]=bestA[a]);
      }
      // per-candidate totals given the other active picks
      var tot=d.M.map((x,i)=>x.c.map((c,ci)=>{
        let bonus=0;
        if(coh) d.M.forEach((y,j)=>{ if(j!==i&&active[j]) bonus+=BETA*relOf(d,c.e,y.c[pick[j]].e); });
        return {L:loc[i][ci],B:bonus,T:loc[i][ci]+bonus};
      }));
      if(!coh) d.M.forEach((x,i)=>{ const T=tot[i].map(t=>t.T); pick[i]=T.indexOf(Math.max(...T)); });
      const nextActive=d.M.map((x,i)=>tot[i][pick[i]].T>=tau);
      const changed=nextActive.some((v,i)=>v!==active[i]);
      active=nextActive;
      if(!changed||!coh) break;
    }
    return {pick,tot,active};
  }

  const xs=d3.scaleLinear().domain([-4.7,3]).clamp(true);
  function draw(){
    const d=S[+sel.value], lam=+lamEl.value, tau=+tauEl.value;
    document.getElementById("el-lamval").textContent=lam.toFixed(2);
    document.getElementById("el-tauval").textContent=(tau<0?"−":"")+Math.abs(tau).toFixed(1);
    cohBtn.textContent="coherence: "+(coh?"on (β = 1)":"off");
    const R=solve(d,lam,tau);
    svg.selectAll("*").remove();
    // sentence with mention highlights (monospace for predictable widths)
    const cw=7.4, sx=Math.max(8,(W-d.text.length*cw)/2);
    d.M.forEach(x=>{
      const k=d.text.indexOf(x.m);
      if(k<0) return;
      svg.append("rect").attr("x",sx+k*cw-2).attr("y",10).attr("width",x.m.length*cw+4).attr("height",22).attr("rx",4)
        .attr("fill","rgba(255,180,84,.14)").attr("stroke",C.a2).attr("stroke-opacity",.7);
    });
    svg.append("text").attr("x",sx).attr("y",26).attr("font-size",12.3).attr("font-family","SF Mono, Menlo, monospace").attr("fill",C.ink).text(d.text);

    const n=d.M.length, colW=W/n, top=70, rowH=46;
    const chosenPos=[];
    d.M.forEach((x,i)=>{
      const x0=i*colW+12, x1=(i+1)*colW-14;
      xs.range([x0,x1]);
      svg.append("text").attr("x",x0).attr("y",top-14).attr("font-size",12).attr("font-weight",700).attr("fill",C.a2).text(`“${x.m}”`);
      // τ line
      svg.append("line").attr("x1",xs(tau)).attr("x2",xs(tau)).attr("y1",top-4).attr("y2",top+3*rowH+2).attr("stroke",C.bad).attr("stroke-dasharray","4 3").attr("stroke-opacity",.8);
      svg.append("text").attr("x",xs(tau)+3).attr("y",top+3*rowH+12).attr("font-size",9).attr("fill",C.bad).text("τ");
      x.c.forEach((c,ci)=>{
        const y=top+ci*rowH, t=R.tot[i][ci], isPick=ci===R.pick[i], linked=isPick&&R.active[i];
        svg.append("text").attr("x",x0).attr("y",y+8).attr("font-size",10).attr("fill",linked?C.good:(isPick?C.bad:C.muted)).attr("font-weight",isPick?700:400)
          .text(c.e.length>32?c.e.slice(0,31)+"…":c.e);
        svg.append("text").attr("x",x0).attr("y",y+33).attr("font-size",8.5).attr("fill",C.muted).text(`P(e|m) = ${c.p.toFixed(2)}   sim = ${c.s.toFixed(2)}`+(t.B>0?`   coh +${t.B.toFixed(2)}`:""));
        svg.append("rect").attr("x",x0).attr("y",y+12).attr("width",x1-x0).attr("height",11).attr("rx",3).attr("fill",C.panel);
        svg.append("rect").attr("x",x0).attr("y",y+12).attr("width",Math.max(1,xs(t.L)-x0)).attr("height",11).attr("rx",3).attr("fill",linked?C.good:C.blue).attr("fill-opacity",.8);
        if(t.B>0) svg.append("rect").attr("x",xs(t.L)).attr("y",y+12).attr("width",Math.max(0,xs(t.T)-xs(t.L))).attr("height",11).attr("fill",C.accent).attr("fill-opacity",.75);
        svg.append("text").attr("x",Math.min(x1-2,xs(t.T)+3)).attr("y",y+21).attr("font-size",9).attr("fill",C.ink).attr("text-anchor",xs(t.T)>x1-30?"end":"start").text(t.T.toFixed(2));
      });
      // decision tag
      const ty=top+3*rowH+30, pk=x.c[R.pick[i]], ok=R.active[i];
      const label=ok?pk.e:"NIL";
      const tw=Math.min(colW-20,label.length*6.2+16);
      svg.append("rect").attr("x",x0).attr("y",ty-14).attr("width",tw).attr("height",22).attr("rx",11)
        .attr("fill",ok?"rgba(74,222,128,.14)":"rgba(248,113,113,.14)").attr("stroke",ok?C.good:C.bad);
      svg.append("text").attr("x",x0+8).attr("y",ty+1).attr("font-size",10).attr("font-weight",700).attr("fill",ok?C.good:C.bad).text(label.length>30?label.slice(0,29)+"…":label);
      chosenPos.push({x:x0+tw/2,y:ty+8,e:pk.e,ok});
    });
    // coherence arcs between chosen (linked) entities
    if(coh){
      for(let a=0;a<chosenPos.length;a++) for(let b=a+1;b<chosenPos.length;b++){
        const A=chosenPos[a],B=chosenPos[b];
        if(!A.ok||!B.ok) continue;
        const r=relOf(d,A.e,B.e), dip=Math.min(H-8,A.y+16+(b-a)*14);
        svg.append("path").attr("d",`M${A.x},${A.y} Q${(A.x+B.x)/2},${dip+12} ${B.x},${B.y}`).attr("fill","none")
          .attr("stroke",C.accent).attr("stroke-width",0.8+r*4).attr("stroke-opacity",.35+r*.5);
        svg.append("text").attr("x",(A.x+B.x)/2).attr("y",Math.min(H-3,dip+4)).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.accent).text(`rel ${r.toFixed(2)}`);
      }
    } else {
      svg.append("text").attr("x",W/2).attr("y",H-14).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.muted)
        .text("bars: blue = local φ = λ·log P + (1−λ)·sim  ·  purple = coherence bonus  ·  red dashed = τ");
    }
    // readout: compare with prior-only choice
    const parts=d.M.map((x,i)=>{
      const pk=x.c[R.pick[i]], prior=x.c.reduce((a,b)=>b.p>a.p?b:a);
      const res=R.active[i]?`<b style="color:${C.good}">${pk.e}</b>`:`<b style="color:${C.bad}">NIL</b>`;
      const note=R.active[i]&&pk.e!==prior.e&&prior.p>pk.p?` <span style="color:${C.a2}">(overrides prior: ${prior.e})</span>`:"";
      return `“${x.m}” → ${res} [${R.tot[i][R.pick[i]].T.toFixed(2)}]${note}`;
    });
    d3.select("#el-read").html(parts.join(" &nbsp;·&nbsp; ")+`<br>λ = ${lam.toFixed(2)}, τ = ${(tau<0?"−":"")+Math.abs(tau).toFixed(1)}, `+(coh?"collective: exact joint argmax of ∑ φ + β·∑ rel over the non-NIL mentions":"local: each mention decided independently"));
  }
  sel.addEventListener("change",draw);
  lamEl.addEventListener("input",draw);
  tauEl.addEventListener("input",draw);
  cohBtn.addEventListener("click",()=>{coh=!coh;draw();});
  draw();
})();
