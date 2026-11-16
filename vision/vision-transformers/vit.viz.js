/* vit.viz.js — interactive figures for vision/vision-transformers/vit.html.
   Loaded after data.js / notes.js. Three independent IIFEs:
     #viz-pos    2-D (bilinear) interpolation of a 14×14 position-embedding grid to the fine-tuning grid
     #viz-data   Table 5: ImageNet top-1 by pre-training dataset, five ViT models
     #viz-scale  Table 6: transfer accuracy vs pre-training exaFLOPs, ViT / ResNet / hybrid */

(function(){
  "use strict";
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#15181f",violet:"#a78bfa",blue:"#5b9cff"};
  const styleAxis=a=>{a.selectAll("text").attr("fill",C.muted);a.selectAll("line,path").attr("stroke",C.line);};

  /* ═══════════════ 1. position-embedding interpolation ═══════════════ */
  (function(){
    const svg=d3.select("#pos-svg"); if(svg.empty()) return;
    const W=640,H=320, G0=14, P=16, DIM=24;
    // synthetic stand-in for a learned table: smooth sinusoidal features of row and column,
    // so similarity falls with distance and same-row / same-column patches stay similar
    function emb(r,c){
      const v=[];
      for(let f=1;f<=3;f++){
        const a=Math.PI*f*r/(G0-1), b=Math.PI*f*c/(G0-1);
        v.push(Math.cos(a),Math.sin(a),Math.cos(b),Math.sin(b));
        v.push(0.5*Math.cos(a+b),0.5*Math.sin(a-b)*0.6+0.3*Math.cos(0.7*f*(r-c)));
      }
      return v.slice(0,DIM);
    }
    const src=[]; for(let r=0;r<G0;r++){src.push([]);for(let c=0;c<G0;c++)src[r].push(emb(r,c));}
    const cos=(a,b)=>{let d=0,na=0,nb=0;for(let i=0;i<a.length;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}return d/Math.sqrt(na*nb||1);};
    const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
    // half-pixel-aligned bilinear sample: target cell centre mapped to the same image location
    function sampleCoord(i,g){return clamp((i+0.5)*G0/g-0.5,0,G0-1);}
    function interp(ti,tj,g){
      const u=sampleCoord(ti,g), v=sampleCoord(tj,g);
      const r0=Math.floor(u), c0=Math.floor(v), r1=Math.min(G0-1,r0+1), c1=Math.min(G0-1,c0+1);
      const fr=u-r0, fc=v-c0;
      const nb=[{r:r0,c:c0,w:(1-fr)*(1-fc)},{r:r0,c:c1,w:(1-fr)*fc},{r:r1,c:c0,w:fr*(1-fc)},{r:r1,c:c1,w:fr*fc}];
      // merge duplicates at the border
      const m=new Map(); nb.forEach(n=>{const k=n.r+","+n.c;m.set(k,{r:n.r,c:n.c,w:(m.has(k)?m.get(k).w:0)+n.w});});
      const parts=[...m.values()];
      const e=new Array(DIM).fill(0);
      parts.forEach(n=>{const s=src[n.r][n.c];for(let k=0;k<DIM;k++)e[k]+=n.w*s[k];});
      return {e,u,v,parts};
    }
    const col=d3.scaleSequential(d3.interpolateViridis).domain([-0.2,1]);
    const L={x:14,y:34,s:250}, R={x:340,y:34,s:250};
    svg.append("text").attr("x",L.x).attr("y",20).attr("fill",C.muted).attr("font-size",11).text("pre-trained grid, 14 × 14 (224 px)");
    const rTitle=svg.append("text").attr("x",R.x).attr("y",20).attr("fill",C.muted).attr("font-size",11);
    const gL=svg.append("g"), gR=svg.append("g"), gN=svg.append("g"), gT=svg.append("g");
    svg.append("text").attr("x",L.x).attr("y",H-10).attr("fill",C.muted).attr("font-size",10.5).text("colour = cosine similarity to the chosen interpolated embedding (yellow = 1)");
    // the left grid is fixed; draw once
    const cL=L.s/G0;
    for(let r=0;r<G0;r++)for(let c=0;c<G0;c++)
      gL.append("rect").attr("class","s").attr("x",L.x+c*cL).attr("y",L.y+r*cL).attr("width",cL-1).attr("height",cL-1).attr("rx",1.5).datum({r,c});
    let state={res:384,r:5,c:17};
    function draw(){
      const res=+d3.select("#pos-res").property("value");
      const g=Math.round(res/P);
      const tr=clamp(+d3.select("#pos-r").property("value"),0,g-1), tc=clamp(+d3.select("#pos-c").property("value"),0,g-1);
      state={res,r:tr,c:tc};
      d3.select("#pos-resv").text(`${res} px → ${g} × ${g}`);
      d3.select("#pos-rv").text(tr); d3.select("#pos-cv").text(tc);
      rTitle.text(`fine-tuning grid, ${g} × ${g} (${res} px, P = 16)`);
      const t=interp(tr,tc,g);
      gL.selectAll("rect.s").attr("fill",d=>col(cos(src[d.r][d.c],t.e)));
      // right grid
      const cR=R.s/g, cells=[];
      for(let i=0;i<g;i++)for(let j=0;j<g;j++)cells.push({i,j});
      const s=gR.selectAll("rect").data(cells,d=>g+":"+d.i+","+d.j);
      s.exit().remove();
      s.enter().append("rect").attr("rx",cR>6?1.5:0).style("cursor","pointer")
        .on("click",(ev,d)=>{d3.select("#pos-r").property("value",d.i);d3.select("#pos-c").property("value",d.j);draw();})
        .merge(s)
        .attr("x",d=>R.x+d.j*cR).attr("y",d=>R.y+d.i*cR).attr("width",Math.max(0.6,cR-(cR>6?1:0.3))).attr("height",Math.max(0.6,cR-(cR>6?1:0.3)))
        .attr("fill",d=>col(cos(interp(d.i,d.j,g).e,t.e)));
      // outlines: neighbours on the left, chosen cell on the right
      gN.selectAll("*").remove();
      t.parts.forEach(n=>{
        gN.append("rect").attr("x",L.x+n.c*cL-0.5).attr("y",L.y+n.r*cL-0.5).attr("width",cL).attr("height",cL)
          .attr("fill","none").attr("stroke",C.bad).attr("stroke-width",Math.max(1,3*n.w+0.6));
      });
      gN.append("circle").attr("cx",L.x+(t.v+0.5)*cL).attr("cy",L.y+(t.u+0.5)*cL).attr("r",3).attr("fill",C.bad).attr("stroke",C.panel);
      gN.append("rect").attr("x",R.x+tc*cR-0.5).attr("y",R.y+tr*cR-0.5).attr("width",cR).attr("height",cR)
        .attr("fill","none").attr("stroke",C.bad).attr("stroke-width",2);
      // weights text
      gT.selectAll("*").remove();
      const top=t.parts.slice().sort((a,b)=>b.w-a.w);
      gT.append("text").attr("x",L.x).attr("y",L.y+L.s+16).attr("fill",C.ink).attr("font-size",10.5)
        .text("weights: "+top.map(n=>`(${n.r},${n.c}) ${n.w.toFixed(2)}`).join("  "));
      gT.append("text").attr("x",R.x).attr("y",R.y+R.s+16).attr("fill",C.ink).attr("font-size",10.5)
        .text(`${g*g} patch slots + 1 class slot (was 196 + 1)`);
      const note = g===G0 ? "Same grid as pre-training: no interpolation is needed and each slot copies its own vector."
                 : `Each new slot is a weighted mix of at most 4 pre-trained vectors; ${g*g-196>0?(g*g-196)+" slots more than pre-training":(196-g*g)+" slots fewer than pre-training"}.`;
      d3.select("#pos-read").text(`Fine-tuning at ${res} px gives a ${g} × ${g} grid. Target patch (${tr}, ${tc}) sits at source coordinates (${t.u.toFixed(2)}, ${t.v.toFixed(2)}) on the 14 × 14 grid; its embedding = `+
        top.map(n=>`${n.w.toFixed(2)} × E_pos(${n.r},${n.c})`).join(" + ")+`. ${note}`);
    }
    d3.selectAll("#pos-res,#pos-r,#pos-c").on("input",draw);
    draw();
  })();

  /* ═══════════════ 2. Table 5: pre-training dataset ═══════════════ */
  (function(){
    const svg=d3.select("#data-svg"); if(svg.empty()) return;
    const W=640,H=290,m={l:52,r:120,t:18,b:40};
    // ImageNet top-1 after fine-tuning at 384, as printed in Table 5; null = not trained
    const sets=["ImageNet","ImageNet-21k","JFT-300M"], sizes=["1.3M","14M","303M"];
    const rows=[
      {k:"ViT-B/16",v:[77.91,83.97,84.15],col:C.blue,dy:12},
      {k:"ViT-B/32",v:[73.38,81.28,80.73],col:C.violet,dy:4},
      {k:"ViT-L/16",v:[76.53,85.15,87.12],col:C.accent,dy:6},
      {k:"ViT-L/32",v:[71.16,80.99,84.37],col:C.a2,dy:-2},
      {k:"ViT-H/14",v:[null,85.13,88.04],col:C.good,dy:0}
    ];
    const x=d3.scalePoint().domain(sets).range([m.l+20,W-m.r-10]);
    const y=d3.scaleLinear().domain([70,89]).range([H-m.b,m.t]);
    svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(x).tickFormat((d,i)=>`${d} (${sizes[i]})`)).call(styleAxis);
    svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(7)).call(styleAxis);
    svg.append("text").attr("transform",`translate(14,${(H-m.b+m.t)/2}) rotate(-90)`).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("ImageNet top-1 (%), fine-tuned");
    svg.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-6).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("pre-training dataset (images)");
    const line=d3.line().defined(d=>d.v!=null).x(d=>x(d.s)).y(d=>y(d.v));
    const g=svg.append("g");
    let sel=null;
    const btns=d3.select("#data-btns");
    ["all"].concat(rows.map(r=>r.k)).forEach(k=>btns.append("button").attr("type","button").text(k).classed("on",k==="all")
      .on("click",function(){sel=k==="all"?null:k;btns.selectAll("button").classed("on",false);d3.select(this).classed("on",true);draw();}));
    function say(r){
      const v=r.v, parts=sets.map((s,i)=>v[i]==null?`${s}: not trained`:`${s}: ${v[i].toFixed(2)}`);
      const gain=v[0]!=null?` From ImageNet to JFT-300M: ${(v[2]-v[0]>=0?"+":"")+(v[2]-v[0]).toFixed(2)} points (our arithmetic).`:"";
      d3.select("#data-read").text(`${r.k} — ${parts.join("; ")}.${gain}`);
    }
    function draw(){
      const s=g.selectAll("g.r").data(rows,r=>r.k);
      const e=s.enter().append("g").attr("class","r");
      e.append("path").attr("fill","none").attr("stroke-width",2.2);
      e.append("text").attr("class","nm").attr("font-size",11).attr("dx",8).attr("dy",r=>r.dy);
      s.merge(e).each(function(r){
        const gg=d3.select(this), on=!sel||sel===r.k;
        const pts=sets.map((st,i)=>({s:st,v:r.v[i],r}));
        gg.select("path").attr("d",line(pts)).attr("stroke",r.col).attr("opacity",on?1:0.15);
        const last=pts[2];
        gg.select("text.nm").attr("x",x(last.s)).attr("y",y(last.v)).attr("fill",on?r.col:C.line).text(`${r.k}  ${last.v.toFixed(2)}`);
        const c=gg.selectAll("circle").data(pts.filter(p=>p.v!=null));
        c.enter().append("circle").attr("r",5).attr("tabindex",0)
          .on("mouseenter focus",(ev,p)=>d3.select("#data-read").text(`${p.r.k} pre-trained on ${p.s}: ${p.v.toFixed(2)} ImageNet top-1 (Table 5).`))
          .merge(c).attr("cx",p=>x(p.s)).attr("cy",p=>y(p.v)).attr("fill",r.col).attr("stroke",C.panel).attr("stroke-width",1.5).attr("opacity",on?1:0.15);
      });
      if(sel) say(rows.find(r=>r.k===sel));
      else d3.select("#data-read").text("Pre-trained on ImageNet alone, ViT-B/16 (77.91) beats ViT-L/16 (76.53). On ImageNet-21k, L/16 (85.15) and H/14 (85.13) draw level. On JFT-300M the order follows model size: H/14 88.04 > L/16 87.12 > L/32 84.37 > B/16 84.15 > B/32 80.73.");
    }
    draw();
  })();

  /* ═══════════════ 3. Table 6: scaling study ═══════════════ */
  (function(){
    const svg=d3.select("#scale-svg"); if(svg.empty()) return;
    const W=640,H=300,m={l:52,r:20,t:22,b:42};
    // Table 6 (revised version): epochs, ImageNet, ReaL, CIFAR-10, CIFAR-100, Pets, Flowers, exaFLOPs
    const raw=[
      ["ViT","ViT-B/32",7,80.73,86.27,98.61,90.49,93.40,99.27,55],
      ["ViT","ViT-B/16",7,84.15,88.85,99.00,91.87,95.80,99.56,224],
      ["ViT","ViT-L/32",7,84.37,88.28,99.19,92.52,95.83,99.45,196],
      ["ViT","ViT-L/16",7,86.30,89.43,99.38,93.46,96.81,99.66,783],
      ["ViT","ViT-L/16",14,87.12,89.99,99.38,94.04,97.11,99.56,1567],
      ["ViT","ViT-H/14",14,88.08,90.36,99.50,94.71,97.11,99.71,4262],
      ["ResNet","ResNet50x1",7,77.54,84.56,97.67,86.07,91.11,94.26,50],
      ["ResNet","ResNet50x2",7,82.12,87.94,98.29,89.20,93.43,97.02,199],
      ["ResNet","ResNet101x1",7,80.67,87.07,98.48,89.17,94.08,95.95,96],
      ["ResNet","ResNet152x1",7,81.88,87.96,98.82,90.22,94.17,96.94,141],
      ["ResNet","ResNet152x2",7,84.97,89.69,99.06,92.05,95.37,98.62,563],
      ["ResNet","ResNet152x2",14,85.56,89.89,99.24,91.92,95.75,98.75,1126],
      ["ResNet","ResNet200x3",14,87.22,90.15,99.34,93.53,96.32,99.04,3306],
      ["hybrid","R50x1+ViT-B/32",7,84.90,89.15,99.01,92.24,95.75,99.46,106],
      ["hybrid","R50x1+ViT-B/16",7,85.58,89.65,99.14,92.63,96.65,99.40,274],
      ["hybrid","R50x1+ViT-L/32",7,85.68,89.04,99.24,92.93,96.97,99.43,246],
      ["hybrid","R50x1+ViT-L/16",7,86.60,89.72,99.18,93.64,97.03,99.40,859],
      ["hybrid","R50x1+ViT-L/16",14,87.12,89.76,99.31,93.89,97.36,99.11,1668]
    ];
    const metrics=[
      {k:"average-5",lab:"mean of ImageNet, CIFAR-10, CIFAR-100, Pets, Flowers (our arithmetic)",f:r=>(r.inet+r.c10+r.c100+r.pets+r.fl)/5},
      {k:"ImageNet",lab:"ImageNet top-1",f:r=>r.inet},
      {k:"ReaL",lab:"ImageNet ReaL",f:r=>r.real},
      {k:"CIFAR-10",lab:"CIFAR-10",f:r=>r.c10},
      {k:"CIFAR-100",lab:"CIFAR-100",f:r=>r.c100},
      {k:"Pets",lab:"Oxford-IIIT Pets",f:r=>r.pets},
      {k:"Flowers",lab:"Oxford Flowers-102",f:r=>r.fl}
    ];
    const data=raw.map(a=>({fam:a[0],name:a[1]+" ("+a[2]+" ep)",inet:a[3],real:a[4],c10:a[5],c100:a[6],pets:a[7],fl:a[8],ef:a[9]}));
    const fam={ViT:{col:C.accent,shape:d3.symbolCircle},ResNet:{col:C.a2,shape:d3.symbolSquare},hybrid:{col:C.violet,shape:d3.symbolTriangle}};
    const x=d3.scaleLog().domain([35,6000]).range([m.l,W-m.r]);
    const y=d3.scaleLinear().range([H-m.b,m.t]);
    const gx=svg.append("g").attr("transform",`translate(0,${H-m.b})`), gy=svg.append("g").attr("transform",`translate(${m.l},0)`);
    gx.call(d3.axisBottom(x).tickValues([50,100,200,500,1000,2000,5000]).tickFormat(d3.format("~s"))).call(styleAxis);
    svg.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-6).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("pre-training compute, exaFLOPs (log)");
    const yl=svg.append("text").attr("transform",`translate(14,${(H-m.b+m.t)/2}) rotate(-90)`).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11);
    // legend
    Object.keys(fam).forEach((k,i)=>{
      const lg=svg.append("g").attr("transform",`translate(${m.l+14+i*90},${m.t-8})`);
      lg.append("path").attr("d",d3.symbol().type(fam[k].shape).size(60)()).attr("fill",fam[k].col);
      lg.append("text").attr("x",9).attr("y",4).attr("font-size",11).attr("fill",C.ink).text(k);
    });
    const gl=svg.append("g"), gp=svg.append("g");
    let cur=metrics[0];
    const btns=d3.select("#scale-btns");
    metrics.forEach(mt=>btns.append("button").attr("type","button").text(mt.k).classed("on",mt===cur)
      .on("click",function(){cur=mt;btns.selectAll("button").classed("on",false);d3.select(this).classed("on",true);draw();}));
    const fmt=v=>v.toFixed(2);
    function draw(){
      const vals=data.map(d=>({d,v:cur.f(d)}));
      y.domain([Math.floor(d3.min(vals,p=>p.v)-0.3),Math.ceil(d3.max(vals,p=>p.v)+0.2)]);
      gy.call(d3.axisLeft(y).ticks(6)).call(styleAxis);
      yl.text(cur.lab.length>34?cur.k+" (%)":cur.lab+" (%)");
      // frontier line per family, sorted by compute
      const lines=Object.keys(fam).map(k=>({k,pts:vals.filter(p=>p.d.fam===k).sort((a,b)=>a.d.ef-b.d.ef)}));
      const ln=d3.line().x(p=>x(p.d.ef)).y(p=>y(p.v));
      const sl=gl.selectAll("path").data(lines,l=>l.k);
      sl.enter().append("path").attr("fill","none").attr("stroke-width",1.3).attr("opacity",0.45).merge(sl)
        .attr("stroke",l=>fam[l.k].col).attr("stroke-dasharray",l=>l.k==="hybrid"?"4 3":null).attr("d",l=>ln(l.pts));
      const sp=gp.selectAll("path.pt").data(vals,p=>p.d.name+p.d.fam);
      sp.enter().append("path").attr("class","pt").attr("tabindex",0)
        .on("mouseenter focus",(ev,p)=>d3.select("#scale-read").text(`${p.d.name}, ${p.d.fam}: ${fmt(cur.f(p.d))} on ${cur.k} at ${p.d.ef} exaFLOPs (Table 6${cur.k==="average-5"?"; the average is our arithmetic":""}).`))
        .merge(sp)
        .attr("d",p=>d3.symbol().type(fam[p.d.fam].shape).size(70)())
        .attr("fill",p=>fam[p.d.fam].col).attr("stroke",C.panel).attr("stroke-width",1)
        .attr("transform",p=>`translate(${x(p.d.ef)},${y(p.v)})`);
      // compute ratio: for the best ResNet, the cheapest ViT that matches it
      const res=vals.filter(p=>p.d.fam==="ResNet").sort((a,b)=>b.v-a.v)[0];
      const match=vals.filter(p=>p.d.fam==="ViT"&&p.v>=res.v).sort((a,b)=>a.d.ef-b.d.ef)[0];
      const best=vals.slice().sort((a,b)=>b.v-a.v)[0];
      let cmp="No ViT in the sweep matches it.";
      if(match){
        const q=res.d.ef/match.d.ef;
        cmp=`The cheapest ViT that matches it is ${match.d.name} at ${fmt(match.v)} and ${match.d.ef} exaFLOPs, `+
            (q>=1?`${q.toFixed(1)}× less compute.`:`${(1/q).toFixed(1)}× more compute — on this column alone the ResNet frontier is not beaten on cost.`);
      }
      d3.select("#scale-read").text(`${cur.k}: the best ResNet is ${res.d.name} at ${fmt(res.v)} and ${res.d.ef} exaFLOPs. ${cmp} Best overall: ${best.d.name}, ${best.d.fam}, at ${fmt(best.v)}. Ratios are our arithmetic.`);
    }
    draw();
  })();
})();
