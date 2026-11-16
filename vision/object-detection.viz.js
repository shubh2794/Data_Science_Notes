/* object-detection.viz.js — the two figures on vision/object-detection.html.
   Loaded after ../data.js → ../notes.js. Self-contained: needs only d3 from ../vendor.

   1. #iou-svg — two draggable, resizable boxes; intersection, union and the smallest
                 enclosing box drawn; IoU, GIoU and DIoU (and their losses) read out.
   2. #nms-svg — greedy non-maximum suppression stepped one kept box at a time over a
                 fixed scene of 15 scored candidates and 4 ground-truth objects, with an
                 adjustable IoU threshold; the right panel scores what NMS kept as a
                 precision–recall curve and COCO-style 101-point AP. */

(function(){
  const C={accent:"#5b9cff",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel2:"#1e222d",violet:"#a78bfa",teal:"#2dd4bf"};
  // boxes are [x1, y1, x2, y2] (corner form)
  const area=b=>Math.max(0,b[2]-b[0])*Math.max(0,b[3]-b[1]);
  function inter(a,b){return [Math.max(a[0],b[0]),Math.max(a[1],b[1]),Math.min(a[2],b[2]),Math.min(a[3],b[3])];}
  function iou(a,b){const i=area(inter(a,b));return i/(area(a)+area(b)-i);}
  function metrics(a,b){
    const I=area(inter(a,b)), U=area(a)+area(b)-I, J=I/U;
    const Cb=[Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[2],b[2]),Math.max(a[3],b[3])];
    const Ca=area(Cb), G=J-(Ca-U)/Ca;
    const cx=(a[0]+a[2]-b[0]-b[2])/2, cy=(a[1]+a[3]-b[1]-b[3])/2;
    const rho2=cx*cx+cy*cy, c2=(Cb[2]-Cb[0])**2+(Cb[3]-Cb[1])**2, Dd=J-rho2/c2;
    return {I,U,J,Cb,Ca,G,D:Dd,rho2,c2};
  }
  function styleAxis(sel){sel.selectAll(".domain,.tick line").attr("stroke",C.line);
    sel.selectAll(".tick text").attr("fill",C.muted).attr("font-size",9.5);}

  /* ═══ 1 · IoU / GIoU / DIoU calculator ═════════════════════════════════════ */
  (function(){
    const svg=d3.select("#iou-svg"); if(svg.empty()) return;
    const CW=470, CH=340, ox=10, oy=24;
    const presets={
      overlap:[[80,80,260,240],[170,140,340,300]],
      disjoint:[[60,90,180,210],[240,130,360,250]],
      far:[[30,40,130,140],[340,220,440,320]],
      nested:[[60,50,420,320],[180,140,280,230]],
      shifted:[[100,100,260,260],[120,120,280,280]]
    };
    let boxes=presets.overlap.map(b=>b.slice());
    const cv=svg.append("g").attr("transform",`translate(${ox},${oy})`);
    cv.append("rect").attr("width",CW).attr("height",CH).attr("fill","#11141b").attr("stroke",C.line).attr("rx",6);
    svg.append("text").attr("x",ox).attr("y",14).attr("fill",C.ink).attr("font-size",12).attr("font-weight",600).text("drag a box to move it · drag a corner to resize");
    const encl=cv.append("rect").attr("fill","none").attr("stroke",C.violet).attr("stroke-dasharray","5 4");
    const cline=cv.append("line").attr("stroke",C.teal).attr("stroke-dasharray","2 3").attr("stroke-width",1.5);
    const bG=cv.append("g"), interR=cv.append("rect").attr("fill",C.good).attr("opacity",.45).style("pointer-events","none");
    const hG=cv.append("g");
    const cols=[C.accent,C.a2], names=["A (prediction)","B (ground truth)"];

    const px=510, pw=220;
    const pg=svg.append("g").attr("transform",`translate(${px},${oy})`);
    pg.append("text").attr("x",0).attr("y",-10).attr("fill",C.ink).attr("font-size",12).attr("font-weight",600).text("overlap scores");
    const xs=d3.scaleLinear().domain([-1,1]).range([0,pw]);
    pg.append("g").attr("transform",`translate(0,${220})`).call(d3.axisBottom(xs).ticks(5)).call(styleAxis);
    pg.append("line").attr("x1",xs(0)).attr("x2",xs(0)).attr("y1",0).attr("y2",220).attr("stroke",C.muted).attr("stroke-dasharray","3 3");
    const rows=[{k:"J",n:"IoU",c:C.good},{k:"G",n:"GIoU",c:C.violet},{k:"D",n:"DIoU",c:C.teal}];
    const rowG=pg.selectAll("g.row").data(rows).join("g").attr("class","row").attr("transform",(d,i)=>`translate(0,${14+i*68})`);
    rowG.append("text").attr("x",0).attr("y",0).attr("fill",d=>d.c).attr("font-size",11.5).attr("font-weight",600).text(d=>d.n);
    rowG.append("rect").attr("class","bar").attr("y",8).attr("height",22).attr("fill",d=>d.c).attr("opacity",.8);
    rowG.append("text").attr("class","v").attr("y",24).attr("font-size",11).attr("fill",C.ink);
    const ro=d3.select("#iou-readout");

    function draw(){
      const [A,B]=boxes, m=metrics(A,B);
      bG.selectAll("rect").data(boxes).join("rect").attr("class","dragpt").style("cursor","move")
        .attr("x",d=>d[0]).attr("y",d=>d[1]).attr("width",d=>d[2]-d[0]).attr("height",d=>d[3]-d[1])
        .attr("fill",(d,i)=>cols[i]).attr("fill-opacity",.12).attr("stroke",(d,i)=>cols[i]).attr("stroke-width",2)
        .call(d3.drag().on("drag",function(ev,d){
          const w=d[2]-d[0],h=d[3]-d[1];
          let x=Math.max(0,Math.min(CW-w,d[0]+ev.dx)), y=Math.max(0,Math.min(CH-h,d[1]+ev.dy));
          d[0]=x;d[1]=y;d[2]=x+w;d[3]=y+h;draw();}));
      const I=inter(A,B);
      interR.attr("x",I[0]).attr("y",I[1]).attr("width",Math.max(0,I[2]-I[0])).attr("height",Math.max(0,I[3]-I[1]));
      encl.attr("x",m.Cb[0]).attr("y",m.Cb[1]).attr("width",m.Cb[2]-m.Cb[0]).attr("height",m.Cb[3]-m.Cb[1]);
      cline.attr("x1",(A[0]+A[2])/2).attr("y1",(A[1]+A[3])/2).attr("x2",(B[0]+B[2])/2).attr("y2",(B[1]+B[3])/2);
      const hd=boxes.flatMap((b,bi)=>[[0,1],[2,1],[0,3],[2,3]].map(([xi,yi])=>({b,bi,xi,yi})));
      hG.selectAll("circle").data(hd).join("circle").attr("class","dragpt").attr("r",6)
        .attr("cx",d=>d.b[d.xi]).attr("cy",d=>d.b[d.yi]).attr("fill",d=>cols[d.bi]).attr("stroke","#0f1117").attr("stroke-width",1.5)
        .style("cursor","nwse-resize")
        .call(d3.drag().on("drag",function(ev,d){
          const b=d.b, nx=Math.max(0,Math.min(CW,ev.x)), ny=Math.max(0,Math.min(CH,ev.y));
          if(d.xi===0) b[0]=Math.min(nx,b[2]-12); else b[2]=Math.max(nx,b[0]+12);
          if(d.yi===1) b[1]=Math.min(ny,b[3]-12); else b[3]=Math.max(ny,b[1]+12);
          draw();}));
      rowG.select("rect.bar").attr("x",d=>Math.min(xs(0),xs(m[d.k]))).attr("width",d=>Math.abs(xs(m[d.k])-xs(0)));
      rowG.select("text.v").attr("x",d=>m[d.k]>=0?xs(m[d.k])+5:xs(m[d.k])-5).attr("text-anchor",d=>m[d.k]>=0?"start":"end")
        .text(d=>m[d.k].toFixed(3));
      const lab=m.J===0?` · <b style="color:${C.bad}">IoU = 0: the IoU loss is flat here, but GIoU and DIoU still point the boxes together</b>`:"";
      ro.html(`|A∩B| = <b>${Math.round(m.I)}</b> · |A∪B| = <b>${Math.round(m.U)}</b> · |C| (dashed violet) = <b>${Math.round(m.Ca)}</b> · `+
        `centre distance² / C-diagonal² = <b>${(m.rho2/m.c2).toFixed(3)}</b><br>`+
        `losses: 1 − IoU = <b>${(1-m.J).toFixed(3)}</b> · 1 − GIoU = <b>${(1-m.G).toFixed(3)}</b> · 1 − DIoU = <b>${(1-m.D).toFixed(3)}</b>`+
        ` · matches at IoU ≥ 0.5? <b>${m.J>=0.5?"yes":"no"}</b>${lab}`);
    }
    d3.select("#iou-preset").on("change",function(){boxes=presets[this.value].map(b=>b.slice());draw();});
    // legend
    const lg=svg.append("g").attr("transform",`translate(${px},${oy+262})`);
    [[names[0],C.accent],[names[1],C.a2],["intersection",C.good],["enclosing box C",C.violet],["centre distance",C.teal]].forEach((d,i)=>{
      lg.append("rect").attr("x",(i%2)*118).attr("y",Math.floor(i/2)*17-9).attr("width",10).attr("height",10).attr("fill",d[1]);
      lg.append("text").attr("x",(i%2)*118+15).attr("y",Math.floor(i/2)*17).attr("fill",C.muted).attr("font-size",10.5).text(d[0]);});
    draw();
  })();

  /* ═══ 2 · NMS step-through + PR / AP of what survives ══════════════════════ */
  (function(){
    const svg=d3.select("#nms-svg"); if(svg.empty()) return;
    const GT=[[60,70,150,260],[100,75,190,262],[250,170,380,260],[300,40,326,62]];
    const D=[
      [62,72,148,258,.95],[55,65,155,250,.81],[70,85,145,265,.66],
      [102,78,192,260,.90],[95,70,185,255,.72],[110,90,198,268,.55],
      [252,168,378,262,.88],[240,165,370,255,.77],[258,178,390,265,.60],[250,170,320,260,.45],
      [298,38,328,64,.58],[304,44,330,66,.35],
      [180,20,240,60,.40],[10,200,60,280,.30],[330,100,390,150,.25]];
    const order=D.map((d,i)=>i).sort((i,j)=>D[j][4]-D[i][4]);

    function nmsAll(tau){
      const st=D.map(()=>"pending"); const keep=[];
      for(const i of order){ if(st[i]!=="pending") continue; st[i]="kept"; keep.push(i);
        for(const j of order) if(st[j]==="pending" && iou(D[i],D[j])>tau) st[j]="suppressed"; }
      return keep;
    }
    function evalAP(keep,t){
      const dets=keep.map(i=>D[i]).sort((a,b)=>b[4]-a[4]);
      const used=new Set(); let tp=0, fp=0; const pr=[];
      for(const d of dets){
        let best=-1, bi=-1;
        GT.forEach((g,k)=>{ if(used.has(k)) return; const v=iou(d,g); if(v>best){best=v;bi=k;} });
        if(best>=t){used.add(bi);tp++;} else fp++;
        pr.push({r:tp/GT.length,p:tp/(tp+fp),hit:best>=t});
      }
      let s=0; const env=[];
      for(let q=0;q<=100;q++){ const rr=q/100; let mx=0; for(const e of pr) if(e.r>=rr-1e-12 && e.p>mx) mx=e.p; s+=mx; env.push({r:rr,p:mx}); }
      return {ap:s/101,pr,env};
    }

    const sx=20, sy=34, SW=400, SH=300;
    const sc=svg.append("g").attr("transform",`translate(${sx},${sy})`);
    sc.append("rect").attr("width",SW).attr("height",SH).attr("fill","#11141b").attr("stroke",C.line).attr("rx",6);
    svg.append("text").attr("x",sx).attr("y",20).attr("fill",C.ink).attr("font-size",12).attr("font-weight",600).text("15 scored candidates, 4 objects (two of them overlapping people, one small bird)");
    const gtG=sc.append("g"), dG=sc.append("g"), labG=sc.append("g");

    const px=486, py=sy, PW=240, PH=230;
    const pg=svg.append("g").attr("transform",`translate(${px},${py})`);
    const xr=d3.scaleLinear().domain([0,1]).range([0,PW]), yp=d3.scaleLinear().domain([0,1.02]).range([PH,0]);
    pg.append("g").attr("transform",`translate(0,${PH})`).call(d3.axisBottom(xr).ticks(5)).call(styleAxis);
    pg.append("g").call(d3.axisLeft(yp).ticks(5)).call(styleAxis);
    pg.append("text").attr("x",PW/2).attr("y",PH+28).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10.5).text("recall");
    pg.append("text").attr("transform",`translate(-32,${PH/2}) rotate(-90)`).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10.5).text("precision");
    const ptitle=pg.append("text").attr("x",0).attr("y",-12).attr("fill",C.ink).attr("font-size",12).attr("font-weight",600);
    const envA=pg.append("path").attr("fill",C.accent).attr("opacity",.22);
    const envL=pg.append("path").attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2);
    const rawG=pg.append("g");
    const apT=pg.append("text").attr("x",PW).attr("y",PH+46).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10.5);

    let st, step=0, lastMsg="";
    function reset(){ st=D.map(()=>({s:"pending",by:-1,v:0})); step=0; lastMsg="press <b>step</b> to keep the highest-scoring pending box and suppress its overlaps"; draw(); }
    function doStep(){
      const tau=+d3.select("#nms-tau").property("value");
      const i=order.find(k=>st[k].s==="pending");
      st.forEach(x=>{ if(x.s==="current") x.s="kept"; });
      if(i===undefined){ lastMsg="done — no pending boxes left"; draw(); return false; }
      st[i].s="current"; step++;
      const gone=[];
      for(const j of order) if(st[j].s==="pending"){ const v=iou(D[i],D[j]); if(v>tau){ st[j]={s:"suppressed",by:i,v}; gone.push(`#${j} (${D[j][4].toFixed(2)}, IoU ${v.toFixed(2)})`); } }
      lastMsg=`step ${step}: keep <b>#${i}</b> (score ${D[i][4].toFixed(2)}) · suppressed: ${gone.length?gone.join(", "):"none"}`;
      draw(); return true;
    }

    function draw(){
      const tau=+d3.select("#nms-tau").property("value"), t=+d3.select("#nms-eval").property("value");
      d3.select("#nms-tauv").text(tau.toFixed(2)); d3.select("#nms-evalv").text(t.toFixed(2));
      const showGT=d3.select("#nms-gt").property("checked");
      gtG.selectAll("rect").data(showGT?GT:[]).join("rect")
        .attr("x",d=>d[0]).attr("y",d=>d[1]).attr("width",d=>d[2]-d[0]).attr("height",d=>d[3]-d[1])
        .attr("fill","none").attr("stroke",C.ink).attr("stroke-width",1).attr("stroke-dasharray","1 3");
      const sty={pending:{c:C.accent,w:1,o:.55,da:null},current:{c:C.a2,w:3,o:1,da:null},kept:{c:C.good,w:2.2,o:1,da:null},suppressed:{c:C.bad,w:1,o:.35,da:"4 3"}};
      const sel=dG.selectAll("rect").data(D.map((d,i)=>({d,i}))).join(enter=>{const r=enter.append("rect");r.append("title");return r;});
      sel.attr("x",o=>o.d[0]).attr("y",o=>o.d[1]).attr("width",o=>o.d[2]-o.d[0]).attr("height",o=>o.d[3]-o.d[1])
        .attr("fill","none").attr("stroke",o=>sty[st[o.i].s].c).attr("stroke-width",o=>sty[st[o.i].s].w)
        .attr("opacity",o=>sty[st[o.i].s].o).attr("stroke-dasharray",o=>sty[st[o.i].s].da);
      sel.select("title").text(o=>`#${o.i} · score ${o.d[4].toFixed(2)} · ${st[o.i].s}${st[o.i].by>=0?` by #${st[o.i].by} (IoU ${st[o.i].v.toFixed(2)})`:""}`);
      sel.filter(o=>st[o.i].s==="current"||st[o.i].s==="kept").raise();
      const lab=D.map((d,i)=>({d,i})).filter(o=>st[o.i].s==="current"||st[o.i].s==="kept");
      labG.selectAll("text").data(lab).join("text").attr("x",o=>o.d[0]+2).attr("y",o=>o.d[1]-3)
        .attr("font-size",9.5).attr("fill",o=>sty[st[o.i].s].c).text(o=>`#${o.i} ${o.d[4].toFixed(2)}`);

      const keep=nmsAll(tau), e=evalAP(keep,t);
      ptitle.text(`after full NMS at τ = ${tau.toFixed(2)}: ${keep.length} boxes kept`);
      envA.attr("d",d3.area().curve(d3.curveStepBefore).x(d=>xr(d.r)).y0(PH).y1(d=>yp(d.p))(e.env));
      envL.attr("d",d3.line().curve(d3.curveStepBefore).x(d=>xr(d.r)).y(d=>yp(d.p))(e.env));
      rawG.selectAll("circle").data(e.pr).join("circle").attr("r",3.5).attr("cx",d=>xr(d.r)).attr("cy",d=>yp(d.p))
        .attr("fill",d=>d.hit?C.good:C.bad).attr("stroke","#0f1117");
      let m=0; for(let k=0;k<10;k++) m+=evalAP(keep,0.5+0.05*k).ap; m/=10;
      const a50=evalAP(keep,0.5).ap, a75=evalAP(keep,0.75).ap;
      apT.text(`AP@${t.toFixed(2)} = ${e.ap.toFixed(3)}`);
      d3.select("#nms-readout").html(`${lastMsg}<br>scoring the full NMS output: AP@.50 = <b>${a50.toFixed(3)}</b> · AP@.75 = <b>${a75.toFixed(3)}</b> · AP@[.50:.95] = <b>${m.toFixed(3)}</b> · green dots = true positives, red = false positives at IoU ≥ ${t.toFixed(2)}`);
    }
    d3.select("#nms-step").on("click",doStep);
    d3.select("#nms-run").on("click",()=>{ while(doStep()){} });
    d3.select("#nms-reset").on("click",reset);
    d3.select("#nms-tau").on("input",reset);
    d3.selectAll("#nms-eval,#nms-gt").on("input",draw);
    reset();
  })();
})();
