/* latent-diffusion.viz.js — interactive figures for deep-learning/diffusion-models/latent-diffusion.html.
   Loaded after data.js / notes.js. Two independent IIFEs:
     #viz-f    downsampling-factor explorer (latent shape, size, attention cost, Table 8 reconstruction)
     #viz-cmp  reported training compute vs FID / throughput (Table 18) */

(function(){
  "use strict";
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#15181f",violet:"#a78bfa",blue:"#5b9cff"};
  const fmt=n=>n.toLocaleString("en-US");

  /* ═══════════════ 1. downsampling-factor explorer ═══════════════ */
  (function(){
    const svg=d3.select("#f-svg"); if(svg.empty()) return;
    const FS=[1,2,4,8,16,32];
    // Table 8 (first-stage autoencoders, ImageNet-val reconstructions). f = 1 is pixel space: no autoencoder.
    const AE={
      kl:{1:{c:3},2:{c:2,rfid:0.086,psnr:32.47},4:{c:3,rfid:0.27,psnr:27.53},8:{c:4,rfid:0.90,psnr:24.19},16:{c:16,rfid:0.87,psnr:24.08},32:{c:64,rfid:2.04,psnr:22.27}},
      vq:{1:{c:3},2:{c:2,z:2048,rfid:0.16,psnr:30.85},4:{c:3,z:8192,rfid:0.58,psnr:27.43},8:{c:4,z:16384,rfid:1.14,psnr:23.07},16:{c:8,z:16384,rfid:5.15,psnr:20.83},32:{c:16,z:16384,rfid:31.83,psnr:17.45}}
    };
    // Section 4.1 ablation verdicts (class-conditional ImageNet, same steps and parameter count per f)
    const VERDICT={1:["pixel space (LDM-1): trains far slower — FID gap of 38 to LDM-8 after 2M steps",C.bad],
      2:["small factor: slow training progress",C.a2],
      4:["among the best conditions (LDM-4)",C.good],
      8:["among the best conditions (LDM-8)",C.good],
      16:["inside the 4–16 balance range, not one of the best two",C.a2],
      32:["too much compression: quality stagnates early",C.bad]};
    const W=760;

    // ── static layout ──
    const IMG=150, ix=24, iy=40, lx=260;
    svg.append("text").attr("x",ix).attr("y",24).attr("fill",C.muted).attr("font-size",11).text("image x (H×W×3)");
    const imgG=svg.append("g").attr("transform",`translate(${ix},${iy})`);
    svg.append("text").attr("x",ix+IMG+22).attr("y",iy+IMG/2-6).attr("fill",C.ink).attr("font-size",13).text("ℰ");
    svg.append("line").attr("x1",ix+IMG+10).attr("x2",lx-14).attr("y1",iy+IMG/2).attr("y2",iy+IMG/2).attr("stroke",C.muted).attr("marker-end","url(#ldm-arr)");
    svg.append("defs").append("marker").attr("id","ldm-arr").attr("viewBox","0 0 10 10").attr("refX",9).attr("refY",5).attr("markerWidth",7).attr("markerHeight",7).attr("orient","auto")
      .append("path").attr("d","M0,0L10,5L0,10z").attr("fill",C.muted);
    const latTitle=svg.append("text").attr("x",lx).attr("y",24).attr("fill",C.muted).attr("font-size",11);
    const latG=svg.append("g").attr("transform",`translate(${lx},${iy})`);
    svg.append("text").attr("x",ix).attr("y",iy+IMG+40).attr("fill",C.muted).attr("font-size",10.5).text("drawn to the same spatial scale; slabs behind the latent = channels");
    const info=svg.append("g").attr("transform","translate(478,34)");

    // bar charts
    const by0=262, by1=400, b1=[50,360], b2=[440,744];
    const xb1=d3.scaleBand().domain(FS.map(String)).range(b1).padding(0.22);
    const xb2=d3.scaleBand().domain(FS.map(String)).range(b2).padding(0.22);
    const ye=d3.scaleLog().domain([1/256,1]).range([by1,by0]);
    const yr=d3.scaleLog().domain([0.05,50]).range([by1,by0]).clamp(true);
    function band(xb){const x0=xb("4"), x1=xb("8")+xb.bandwidth();
      svg.append("rect").attr("x",x0-xb.step()*xb.paddingInner()/2).attr("y",by0-6).attr("width",x1-x0+xb.step()*xb.paddingInner()).attr("height",by1-by0+6).attr("fill",C.good).attr("opacity",0.08);
      svg.append("text").attr("x",(x0+x1)/2).attr("y",by0-10).attr("text-anchor","middle").attr("fill",C.good).attr("font-size",10).text("ablation sweet spot f = 4–8");}
    band(xb1); band(xb2);
    svg.append("g").attr("transform",`translate(0,${by1})`).call(d3.axisBottom(xb1).tickFormat(d=>"f="+d)).attr("color",C.muted);
    svg.append("g").attr("transform",`translate(${b1[0]},0)`).call(d3.axisLeft(ye).tickValues([1,1/4,1/16,1/64,1/256]).tickFormat(d=>d===1?"1":"1/"+Math.round(1/d))).attr("color",C.muted);
    svg.append("g").attr("transform",`translate(0,${by1})`).call(d3.axisBottom(xb2).tickFormat(d=>"f="+d)).attr("color",C.muted);
    svg.append("g").attr("transform",`translate(${b2[0]},0)`).call(d3.axisLeft(yr).tickValues([0.1,1,10]).tickFormat(d3.format("~g"))).attr("color",C.muted);
    svg.append("text").attr("x",b1[0]).attr("y",by1+34).attr("fill",C.muted).attr("font-size",11).text("latent elements / image elements (log)");
    svg.append("text").attr("x",b2[0]).attr("y",by1+34).attr("fill",C.muted).attr("font-size",11).text("autoencoder reconstruction FID, Table 8 (log, lower = better)");
    const g1=svg.append("g"), g2=svg.append("g");

    function draw(){
      const fi=+d3.select("#f-i").property("value"), f=FS[fi], H=+d3.select("#f-res").property("value"), reg=d3.select("#f-reg").property("value");
      d3.select("#f-iv").text(f);
      const ae=AE[reg][f], c=ae.c, h=H/f, elems=h*h*c, img=H*H*3, pos=h*h;

      // image with a faint latent-cell grid
      imgG.selectAll("*").remove();
      imgG.append("rect").attr("width",IMG).attr("height",IMG).attr("fill","#1d2530").attr("stroke",C.line);
      const cell=IMG/h;
      if(cell>=3){for(let k=1;k<h;k++){imgG.append("line").attr("x1",k*cell).attr("x2",k*cell).attr("y1",0).attr("y2",IMG).attr("stroke",C.accent).attr("opacity",0.18);
        imgG.append("line").attr("y1",k*cell).attr("y2",k*cell).attr("x1",0).attr("x2",IMG).attr("stroke",C.accent).attr("opacity",0.18);}}
      imgG.append("text").attr("x",IMG/2).attr("y",IMG/2+4).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",12).text(H+"×"+H+"×3");
      imgG.append("text").attr("x",IMG/2).attr("y",IMG/2+20).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text(cell>=3?"grid = one latent position each":"latent cells too small to draw");

      // latent: same spatial scale, channel slabs stacked behind
      latG.selectAll("*").remove();
      const side=Math.max(3,IMG/f), shown=Math.min(c,12), off=Math.min(5,60/shown);
      for(let k=shown-1;k>=0;k--){
        latG.append("rect").attr("x",k*off).attr("y",-k*off+ (IMG-side)/2+ (shown-1)*off/2).attr("width",side).attr("height",side)
          .attr("fill",k===0?"#1d2a2a":"#18222a").attr("stroke",k===0?C.accent:C.muted).attr("stroke-opacity",k===0?1:0.5);
      }
      latG.append("text").attr("x",0).attr("y",IMG+18).attr("fill",C.accent).attr("font-size",12).text(f===1?"no autoencoder":h+"×"+h+"×"+c+(c>shown?"  ("+shown+" of "+c+" slabs drawn)":""));
      latTitle.text(f===1?"pixel space, f = 1 (LDM-1)":"latent z = ℰ(x), f = "+f+", "+(reg==="kl"?"KL-reg":"VQ-reg, |𝒵| = "+fmt(ae.z)));

      // info panel
      info.selectAll("*").remove();
      const rel=elems/img, att=1/Math.pow(f,4);
      const lines=[
        ["latent shape",f===1?H+"×"+H+"×3 (the image)":h+" × "+h+" × "+c],
        ["numbers per sample",fmt(elems)+"  ("+(f===1?"1×":(img/elems).toFixed(0)+"× fewer")+")"],
        ["spatial positions h·w",fmt(pos)],
        ["one full-grid self-attention*",f===1?"1 (reference)":"1/"+fmt(Math.pow(f,4))+" of pixel space"],
        ["reconstruction R-FID",ae.rfid==null?"— (no first stage)":String(ae.rfid)],
        ["reconstruction PSNR",ae.psnr==null?"—":ae.psnr.toFixed(2)+" dB"]];
      lines.forEach((l,k)=>{info.append("text").attr("x",0).attr("y",k*24).attr("fill",C.muted).attr("font-size",11).text(l[0]);
        info.append("text").attr("x",0).attr("y",k*24+13).attr("fill",C.ink).attr("font-size",12.5).text(l[1]);});
      info.append("text").attr("x",0).attr("y",lines.length*24+8).attr("fill",VERDICT[f][1]).attr("font-size",11.5).text("ablation: "+VERDICT[f][0].split(" — ")[0]);
      info.append("text").attr("x",0).attr("y",lines.length*24+24).attr("fill",C.muted).attr("font-size",9.5).text("* own arithmetic, ∝ (h·w)²; the U-Nets only attend at low resolutions");

      // bars
      const dat=FS.map(ff=>({f:ff,rel:AE[reg][ff].c/(3*ff*ff),rfid:AE[reg][ff].rfid}));
      g1.selectAll("rect").data(dat).join("rect").attr("x",d=>xb1(String(d.f))).attr("width",xb1.bandwidth())
        .attr("y",d=>ye(Math.max(d.rel,1/256))).attr("height",d=>by1-ye(Math.max(d.rel,1/256)))
        .attr("fill",d=>d.f===f?C.accent:C.muted).attr("opacity",d=>d.f===f?0.95:0.45).attr("stroke",d=>d.f===f?C.ink:"none");
      g1.selectAll("text").data(dat).join("text").attr("x",d=>xb1(String(d.f))+xb1.bandwidth()/2).attr("y",d=>ye(Math.max(d.rel,1/256))-4)
        .attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",10).text(d=>d.f===1?"1":"1/"+Math.round(1/d.rel));
      const dr=dat.filter(d=>d.rfid!=null);
      g2.selectAll("rect").data(dr).join("rect").attr("x",d=>xb2(String(d.f))).attr("width",xb2.bandwidth())
        .attr("y",d=>yr(d.rfid)).attr("height",d=>by1-yr(d.rfid))
        .attr("fill",d=>d.f===f?C.a2:C.muted).attr("opacity",d=>d.f===f?0.95:0.45).attr("stroke",d=>d.f===f?C.ink:"none");
      g2.selectAll("text").data(dat).join("text").attr("x",d=>xb2(String(d.f))+xb2.bandwidth()/2).attr("y",d=>d.rfid==null?by1-6:yr(d.rfid)-4)
        .attr("text-anchor","middle").attr("fill",d=>d.rfid==null?C.muted:C.ink).attr("font-size",10).text(d=>d.rfid==null?"n/a":d.rfid);

      d3.select("#f-readout").html(
        (f===1?`<b>f = 1</b>: diffusion directly on the ${H}×${H}×3 image — ${fmt(img)} numbers and ${fmt(pos)} positions per step`:
        `<b>f = ${f}</b>, ${reg==="kl"?"KL-reg":"VQ-reg (|𝒵| = "+fmt(ae.z)+")"}: ${H}×${H}×3 → <b>${h}×${h}×${c}</b> · ${fmt(elems)} numbers, <b>${(img/elems).toFixed(0)}× fewer</b> than the image · ${fmt(pos)} positions (${f*f}× fewer)`)+
        ` · one self-attention layer over the whole grid: <b>${f===1?"reference cost":"1/"+fmt(Math.pow(f,4))}</b> (own arithmetic, ∝ (h·w)²)`+
        `<br>Table 8 reconstruction: ${ae.rfid==null?"none — there is no first stage":"R-FID <b>"+ae.rfid+"</b>, PSNR <b>"+ae.psnr.toFixed(2)+" dB</b>"}`+
        ` · ablation verdict: <b style="color:${VERDICT[f][1]}">${VERDICT[f][0]}</b>`+
        (reg==="kl"&&f>=8?` · note: KL f = 8, 16, 32 all keep ${fmt(H*H/16)} numbers per ${H}² image — channels grow as the grid shrinks`:""));
    }
    ["#f-i","#f-res","#f-reg"].forEach(id=>d3.select(id).on("input",draw).on("change",draw));
    draw();
  })();

  /* ═══════════════ 2. compute vs quality (Table 18) ═══════════════ */
  (function(){
    const svg=d3.select("#cmp-svg"); if(svg.empty()) return;
    // [label, overall V100-days, FID, samples/s (null = not reported), params, is latent model]
    const D={
      imagenet:[["ADM, 250 steps",916,10.94,0.12,"554M",0],["ADM-G, 250 steps",962,4.59,0.07,"608M",0],["ADM-G, 25 steps",962,5.58,0.7,"608M",0],
        ["ADM-G + ADM-U, 250 steps",349,3.85,null,"n/a",0],["LDM-8, 200 steps",79,17.41,1.9,"395M",1],["LDM-8-G, 100 steps",91,8.11,1.93,"506M",1],
        ["LDM-4, 250 steps",271,10.56,0.7,"400M",1],["LDM-4-G s=1.25",271,3.95,0.4,"400M",1],["LDM-4-G s=1.5",271,3.60,0.4,"400M",1]],
      bedrooms:[["ADM, 1000 steps",232,1.9,0.03,"552M",0],["LDM-4, 200 steps",55,2.95,1.07,"274M",1]],
      churches:[["StyleGAN2",64,3.86,null,"59M",0],["LDM-8, 100 steps",18,4.02,6.80,"256M",1]],
      ffhq:[["StyleGAN2",32.13,3.8,null,"59M",0],["LDM-4, 200 steps",26,4.98,1.07,"274M",1]]
    };
    const NAME={imagenet:"ImageNet 256²",bedrooms:"LSUN-Bedrooms 256²",churches:"LSUN-Churches 256²",ffhq:"FFHQ 256²"};
    const W=760,H=360,m={l:64,r:190,t:24,b:46};
    function draw(){
      svg.selectAll("*").remove();
      const ds=d3.select("#cmp-d").property("value"), yv=d3.select("#cmp-y").property("value");
      const all=D[ds], pts=yv==="fid"?all:all.filter(p=>p[3]!=null), miss=all.filter(p=>yv==="thr"&&p[3]==null);
      const xs=d3.scaleLog().domain([d3.min(all,p=>p[1])/1.6,d3.max(all,p=>p[1])*1.6]).range([m.l,W-m.r]);
      const ys=yv==="fid"?d3.scaleLinear().domain([0,d3.max(all,p=>p[2])*1.12]).range([H-m.b,m.t]).nice()
        :d3.scaleLog().domain([d3.min(pts,p=>p[3])/2,d3.max(pts,p=>p[3])*2]).range([H-m.b,m.t]);
      svg.append("g").attr("transform",`translate(0,${H-m.b})`).call(d3.axisBottom(xs).ticks(6,"~g")).attr("color",C.muted);
      svg.append("g").attr("transform",`translate(${m.l},0)`).call(d3.axisLeft(ys).ticks(6,"~g")).attr("color",C.muted);
      svg.append("text").attr("x",(m.l+W-m.r)/2).attr("y",H-10).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("overall training compute, V100-days (log)");
      svg.append("text").attr("x",m.l).attr("y",14).attr("fill",C.muted).attr("font-size",11).text(NAME[ds]+" — "+(yv==="fid"?"FID (lower is better)":"samples / s on one A100 (log, higher is better)"));
      const yof=p=>ys(yv==="fid"?p[2]:p[3]);
      // labels with simple vertical de-overlap
      const lab=pts.map(p=>({p,x:xs(p[1]),y:yof(p),ly:yof(p)})).sort((a,b)=>a.y-b.y);
      for(let i=1;i<lab.length;i++){const q=lab[i-1];if(Math.abs(lab[i].x-q.x)<120&&lab[i].ly-q.ly<13)lab[i].ly=q.ly+13;}
      lab.forEach(L=>{
        const p=L.p, col=p[5]?C.accent:C.a2;
        svg.append("circle").attr("cx",L.x).attr("cy",L.y).attr("r",6).attr("fill",col).attr("opacity",0.9).style("cursor","pointer")
          .on("mouseenter",()=>hover(p,ds)).append("title").text(`${p[0]}: ${p[1]} V100-days, FID ${p[2]}${p[3]!=null?", "+p[3]+" samples/s":""}`);
        svg.append("text").attr("x",L.x+10).attr("y",L.ly+4).attr("fill",col).attr("font-size",10.5).text(p[0]);
      });
      svg.append("circle").attr("cx",W-170).attr("cy",40).attr("r",5).attr("fill",C.accent);
      svg.append("text").attr("x",W-160).attr("y",44).attr("fill",C.accent).attr("font-size",11).text("latent diffusion");
      svg.append("circle").attr("cx",W-170).attr("cy",60).attr("r",5).attr("fill",C.a2);
      svg.append("text").attr("x",W-160).attr("y",64).attr("fill",C.a2).attr("font-size",11).text("baseline");
      summary(ds,all,miss);
    }
    function summary(ds,all,miss){
      const L=all.filter(p=>p[5]), B=all.filter(p=>!p[5]);
      const bl=d3.least(L,p=>p[2]), bb=d3.least(B,p=>p[2]);
      d3.select("#cmp-readout").html(`${NAME[ds]}: best latent model <b>${bl[0]}</b> — ${bl[1]} V100-days, FID <b>${bl[2]}</b>`+
        ` · best baseline <b>${bb[0]}</b> — ${bb[1]} V100-days, FID <b>${bb[2]}</b> · compute ratio ≈ <b>${(bb[1]/bl[1]).toFixed(1)}×</b> (own arithmetic)`+
        (miss.length?`<br>no throughput reported for: ${miss.map(p=>p[0]).join(", ")}`:"")+"<br>hover a point for its numbers");
    }
    function hover(p,ds){
      d3.select("#cmp-readout").html(`<b>${p[0]}</b> (${NAME[ds]}, ${p[5]?"latent diffusion":"baseline"}): <b>${p[1]}</b> V100-days · FID <b>${p[2]}</b> · `+
        (p[3]!=null?`<b>${p[3]}</b> samples/s`:"throughput not reported")+` · ${p[4]} parameters`);
    }
    ["#cmp-d","#cmp-y"].forEach(id=>d3.select(id).on("change",draw));
    draw();
  })();
})();
