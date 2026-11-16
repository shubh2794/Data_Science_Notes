/* alexnet.viz.js — interactive figures for deep-learning/convolutional-networks/alexnet.html.
   Loaded after data.js / notes.js. Two independent IIFEs:
     #viz-aw  layer-by-layer walker of the eight-layer, two-GPU network: shapes, parameters,
              multiply-accumulates, neurons, and which layer boundaries cross between GPUs
     #viz-ap  overlapping (z = 3, s = 2) vs non-overlapping (z = 2, s = 2) max-pooling on a grid
   Every count is computed here from the layer specification; nothing is hard-coded except the
   specification itself (kernel sizes, strides, kernel counts and connectivity from the paper). */

(function(){
  "use strict";
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",panel:"#15181f",violet:"#a78bfa",blue:"#5b9cff"};
  const fmt=n=>Math.round(n).toLocaleString("en-US");
  const short=n=>n>=1e9?(n/1e9).toFixed(2)+" G":n>=1e6?(n/1e6).toFixed(2)+" M":n>=1e3?(n/1e3).toFixed(1)+" k":String(n);
  function rng(seed){let s=seed>>>0;return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return (s+0.5)/4294967296;};}
  function gauss(r){const u=r(),v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}

  /* ═══════════════ 1. the layer walker ═══════════════ */
  (function(){
    const svg=d3.select("#aw-svg"); if(svg.empty()) return;
    const W=760, NCOL=9;
    const xs=d3.range(NCOL).map(i=>78+i*81);
    const BW=44, ROW1=92, ROW2=196, MID=(ROW1+ROW2)/2;
    const CH0=300, CH1=432;                       // bar chart band
    let sel=1, L=[];

    function build(){
      const inp=d3.select("#aw-in").property("value");
      const dense=d3.select("#aw-mode").property("value")==="dense";
      const n0=inp==="227"?227:224, p1=inp==="224p"?2:0;
      const outN=(n,k,s,p)=>Math.floor((n+2*p-k)/s)+1, pool=n=>Math.floor((n-3)/2)+1;
      function conv(id,nIn,k,s,p,cin,cout,grp,conn,after,bias){
        const n=outN(nIn,k,s,p), g=dense?1:grp, cg=cin/g;
        return {id,type:"conv",nIn,cin,n,c:cout,k,s,p,cg,g,conn:(dense&&conn==="same")?"all":conn,after,bias,
          params:k*k*cg*cout+cout,macs:n*n*cout*k*k*cg,act:n*n*cout,nOut:after.indexOf("pool")>=0?pool(n):n};
      }
      function fc(id,nIn,cout,drop,bias,soft){
        return {id,type:"fc",nIn,c:cout,n:1,conn:"all",after:[],drop,bias,soft,params:nIn*cout+cout,macs:nIn*cout,act:cout,nOut:1};
      }
      const c1=conv("conv1",n0,11,4,p1,3,96,1,"input",["LRN","pool"],0);
      const c2=conv("conv2",c1.nOut,5,1,2,96,256,2,"same",["LRN","pool"],1);
      const c3=conv("conv3",c2.nOut,3,1,1,256,384,1,"all",[],0);
      const c4=conv("conv4",c3.n,3,1,1,384,384,2,"same",[],1);
      const c5=conv("conv5",c4.n,3,1,1,384,256,2,"same",["pool"],1);
      const f6=fc("fc6",c5.nOut*c5.nOut*256,4096,true,1,false);
      const f7=fc("fc7",4096,4096,true,1,false);
      const f8=fc("fc8",4096,1000,false,0,true);
      L=[{id:"input",type:"input",n:n0,c:3,act:n0*n0*3,params:0,macs:0,after:[],pad:p1},c1,c2,c3,c4,c5,f6,f7,f8];
      // what each GPU must receive from the other at a crossing: the other half of this layer's input
      L.forEach((l,i)=>{l.recv=0;
        if(!dense&&l.conn==="all"&&i>1){const prev=L[i-1];l.recv=(prev.type==="conv"?prev.nOut*prev.nOut*prev.c:prev.c)/2;}});
      return dense;
    }

    function blockH(l){
      if(l.type==="fc") return 10+34*Math.sqrt(l.c/4096);
      return 14+50*Math.sqrt(l.n/L[0].n);
    }

    function draw(){
      const dense=build();
      const metric=d3.select("#aw-m").property("value");
      svg.selectAll("*").remove();
      const tot={params:d3.sum(L,l=>l.params),macs:d3.sum(L,l=>l.macs),act:d3.sum(L.slice(1),l=>l.act)};

      // row labels
      if(!dense){
        svg.append("text").attr("x",6).attr("y",ROW1-40).attr("fill",C.muted).attr("font-size",10.5).text("GPU 1");
        svg.append("text").attr("x",6).attr("y",ROW2+50).attr("fill",C.muted).attr("font-size",10.5).text("GPU 2");
      }else{
        svg.append("text").attr("x",6).attr("y",MID-40).attr("fill",C.muted).attr("font-size",10.5).text("one device");
      }
      svg.append("text").attr("x",W-6).attr("y",16).attr("text-anchor","end").attr("fill",C.muted).attr("font-size",10.5)
        .text(dense?"one-GPU-equivalent: every layer sees every map below it":"amber = the GPUs exchange activations here; grey = same-GPU only");

      // positions of the block(s) in each column
      const pos=L.map((l,i)=>{const h=blockH(l);
        if(l.type==="input"||dense) return [{y:MID-h/2,h,x:xs[i]-BW/2}];
        return [{y:ROW1-h/2,h,x:xs[i]-BW/2},{y:ROW2-h/2,h,x:xs[i]-BW/2}];});

      // connections
      const gC=svg.append("g");
      for(let i=1;i<NCOL;i++){
        const A=pos[i-1], B=pos[i], l=L[i];
        A.forEach((a,ai)=>B.forEach((b,bi)=>{
          const crossing=A.length===2&&B.length===2&&ai!==bi;
          if(crossing&&l.conn!=="all") return;
          const x1=a.x+BW, y1=a.y+a.h/2, x2=b.x, y2=b.y+b.h/2;
          const hot=!dense&&l.conn==="all";
          gC.append("path").attr("d",`M${x1},${y1} C${(x1+x2)/2},${y1} ${(x1+x2)/2},${y2} ${x2},${y2}`)
            .attr("fill","none").attr("stroke",hot?C.a2:C.muted).attr("stroke-width",hot?1.8:1.1).attr("opacity",hot?0.9:0.55);
        }));
        // ops between columns
        const ops=L[i-1].after||[];
        if(ops.length){
          svg.append("text").attr("x",(xs[i-1]+xs[i])/2).attr("y",ROW2+68).attr("text-anchor","middle")
            .attr("fill",C.violet).attr("font-size",9.5).text(ops.map(o=>o==="pool"?"pool 3/2":o).join(" · "));
        }
      }

      // blocks
      L.forEach((l,i)=>{
        const g=svg.append("g").style("cursor","pointer").on("click",()=>{sel=i;draw();});
        pos[i].forEach((p,pi)=>{
          g.append("rect").attr("x",p.x).attr("y",p.y).attr("width",BW).attr("height",p.h).attr("rx",4)
            .attr("fill",i===sel?"#1d2a2a":C.panel).attr("stroke",i===sel?C.accent:(l.type==="fc"?C.blue:C.line)).attr("stroke-width",i===sel?2:1.2);
          const c=(l.type==="input"||dense)?l.c:l.c/2;
          const shape=l.type==="fc"?String(c):`${l.n}×${l.n}×${c}`;
          g.append("text").attr("x",xs[i]).attr("y",p.y+p.h+12).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9.5).text(shape);
        });
        g.append("text").attr("x",xs[i]).attr("y",34).attr("text-anchor","middle").attr("fill",i===sel?C.accent:C.ink).attr("font-size",11.5).attr("font-weight",600).text(l.id);
        g.append("title").text(l.id+": click for details");
      });

      // bar chart of the chosen metric, aligned under the columns
      const key=metric, totK=tot[key];
      const vals=L.map((l,i)=>key==="act"&&i===0?0:l[key]);
      const y=d3.scaleLinear().domain([0,d3.max(vals)||1]).range([CH1,CH0+14]);
      const lab={params:"parameters (weights + biases)",macs:"multiply-accumulates per image, forward",act:"neurons (output activations)"}[key];
      svg.append("text").attr("x",6).attr("y",CH0-6).attr("fill",C.muted).attr("font-size",11)
        .text(`${lab} · total ${fmt(totK)} (${short(totK)}) · bar labels are the share of the total`);
      svg.append("line").attr("x1",40).attr("x2",W-6).attr("y1",CH1).attr("y2",CH1).attr("stroke",C.line);
      L.forEach((l,i)=>{
        if(i===0) return;
        const v=vals[i], share=100*v/totK;
        const col=l.type==="fc"?C.blue:C.accent;
        svg.append("rect").attr("x",xs[i]-16).attr("width",32).attr("y",y(v)).attr("height",Math.max(0.5,CH1-y(v)))
          .attr("fill",col).attr("opacity",i===sel?1:0.55).style("cursor","pointer").on("click",()=>{sel=i;draw();})
          .append("title").text(`${l.id}: ${fmt(v)}`);
        svg.append("text").attr("x",xs[i]).attr("y",y(v)-4).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",10)
          .text(share>=10?share.toFixed(0)+"%":share.toFixed(1)+"%");
        svg.append("text").attr("x",xs[i]).attr("y",CH1+13).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text(l.id);
      });
      readout(dense,tot);
    }

    function readout(dense,tot){
      const l=L[sel], el=d3.select("#aw-readout");
      const conv5=L[5], fc6=L[6];
      let h="";
      if(l.type==="input"){
        h=`<b>input</b> ${l.n}×${l.n}×3 = <b>${fmt(l.act)}</b> numbers, a random crop of the 256×256 mean-subtracted image; both GPUs receive the whole crop.`+
          (l.n===224&&l.pad===0?` · with 224 and no padding, conv1 gives <b>${L[1].n}×${L[1].n}</b>, not the 55×55 the paper's figure shows, and the head becomes ${conv5.nOut}×${conv5.nOut}×256 → fc6 with <b>${fmt(fc6.params)}</b> parameters`:"")+
          (l.n===227?" · 227 (or 224 with two pixels of padding) is the size that reproduces the paper's 55×55 first map":"");
      }else if(l.type==="conv"){
        const per=dense?l.c:l.c/2;
        const padNote=l.id==="conv1"?(l.p?"2 (one of the two common fixes)":"0"):`${l.p} (not stated in the paper; implied by the map sizes it reports)`;
        h=`<b>${l.id}</b> · ${l.nIn}×${l.nIn}×${l.cin} → <b>${l.n}×${l.n}×${l.c}</b> · ${l.c} kernels of <b>${l.k}×${l.k}×${l.cg}</b>, stride ${l.s}, padding ${padNote}`+
          (dense?"":` · ${per} kernels on each GPU`)+
          `<br>connectivity: <b>${l.conn==="input"?"each GPU filters the full 3-channel image":l.conn==="all"?(dense?"all maps of the previous layer":"all maps of the previous layer, on both GPUs — a crossing"):"only the maps on the same GPU"}</b>`+
          (l.recv?` · each GPU receives <b>${fmt(l.recv)}</b> activations per image from the other (${(l.recv*4/1024).toFixed(1)} KiB in 32-bit)`:"")+
          `<br>params <b>${fmt(l.params)}</b> (${(100*l.params/tot.params).toFixed(2)}% of the network) · MACs <b>${fmt(l.macs)}</b> (${(100*l.macs/tot.macs).toFixed(1)}%) · neurons <b>${fmt(l.act)}</b>`+
          ` · then ReLU${l.after.length?" → "+l.after.map(o=>o==="pool"?"max-pool 3×3 stride 2 → "+l.nOut+"×"+l.nOut:"LRN").join(" → "):""} · bias initialised to <b>${l.bias}</b>`+
          (l.id==="conv1"&&l.n===55?` · the paper's caption lists 253,440 neurons here; 55×55×96 is ${fmt(l.act)}, a mismatch the paper does not explain`:"");
      }else{
        h=`<b>${l.id}</b> · ${fmt(l.nIn)} → <b>${fmt(l.c)}</b>, fully connected`+
          (l.id==="fc6"?` (the input is the pooled ${conv5.nOut}×${conv5.nOut}×256 map, flattened)`:"")+
          (dense?"":` · ${l.c/2} units on each GPU, each connected to all ${fmt(l.nIn)} inputs — a crossing`)+
          (l.recv?` · each GPU receives <b>${fmt(l.recv)}</b> activations per image from the other`:"")+
          `<br>params <b>${fmt(l.params)}</b> (<b>${(100*l.params/tot.params).toFixed(1)}%</b> of the network) · MACs <b>${fmt(l.macs)}</b> (${(100*l.macs/tot.macs).toFixed(1)}%)`+
          (l.drop?" · ReLU, then <b>dropout p = 0.5</b>":"")+(l.soft?" · feeds the 1000-way softmax":"")+` · bias initialised to <b>${l.bias}</b>`;
      }
      const convP=d3.sum(L.slice(1,6),z=>z.params), fcP=d3.sum(L.slice(6),z=>z.params);
      const convM=d3.sum(L.slice(1,6),z=>z.macs);
      const recv=d3.sum(L,z=>z.recv);
      h+=`<br><span style="color:var(--muted)">totals (our arithmetic): <b>${fmt(tot.params)}</b> parameters — conv ${(100*convP/tot.params).toFixed(1)}%, fully connected ${(100*fcP/tot.params).toFixed(1)}% · <b>${short(tot.macs)}</b> MACs per image — conv ${(100*convM/tot.macs).toFixed(1)}% · <b>${fmt(tot.act)}</b> neurons`+
        (dense?"":` · cross-GPU traffic ${fmt(recv)} activations per image per GPU, ≈ ${(recv*4*128/1048576).toFixed(1)} MiB per 128-image batch in 32-bit, forward only`)+
        ` · the paper reports 60 million parameters and 650,000 neurons</span>`;
      el.html(h);
    }

    d3.select("#aw-prev").on("click",()=>{sel=(sel+NCOL-1)%NCOL;draw();});
    d3.select("#aw-next").on("click",()=>{sel=(sel+1)%NCOL;draw();});
    ["#aw-m","#aw-in","#aw-mode"].forEach(id=>d3.select(id).on("change",draw));
    draw();
  })();

  /* ═══════════════ 2. overlapping vs non-overlapping pooling ═══════════════ */
  (function(){
    const svg=d3.select("#ap-svg"); if(svg.empty()) return;
    const W=760, TRIALS=300;
    let seed=5, M=null, hov=null, cacheN=-1, stab=null; const cust={};

    function scheme(){
      const s=d3.select("#ap-s").property("value");
      const custom=s==="custom";
      d3.select("#ap-z").property("disabled",!custom); d3.select("#ap-st").property("disabled",!custom);
      if(s==="ov") return {z:3,s:2};
      if(s==="non") return {z:2,s:2};
      return {z:+d3.select("#ap-z").property("value"),s:+d3.select("#ap-st").property("value")};
    }
    const outSize=(n,z,s)=>Math.floor((n-z)/s)+1;
    function makeMap(n,sd){const R=rng(sd);return d3.range(n).map(()=>d3.range(n+1).map(()=>Math.max(0,gauss(R))));}
    function pool(A,n,z,s,off){
      const m=outSize(n,z,s), O=[];
      for(let i=0;i<m;i++){const row=[];for(let j=0;j<m;j++){let v=-Infinity;
        for(let a=0;a<z;a++)for(let b=0;b<z;b++)v=Math.max(v,A[i*s+a][off+j*s+b]);row.push(v);}O.push(row);}
      return O;
    }
    function unchangedFrac(n,z,s){
      if(n<z) return NaN;
      let same=0,all=0;
      for(let t=0;t<TRIALS;t++){const A=makeMap(n,1000+t),P=pool(A,n,z,s,0),Q=pool(A,n,z,s,1);
        for(let i=0;i<P.length;i++)for(let j=0;j<P.length;j++){all++;if(P[i][j]===Q[i][j])same++;}}
      return same/all;
    }
    function coverage(n,z,s){
      const m=outSize(n,z,s), cov=d3.range(n).map(()=>new Array(n).fill(0));
      for(let i=0;i<m;i++)for(let j=0;j<m;j++)for(let a=0;a<z;a++)for(let b=0;b<z;b++)cov[i*s+a][j*s+b]++;
      return cov;
    }

    function draw(){
      const n=+d3.select("#ap-n").property("value"); d3.select("#ap-nv").text(n);
      d3.select("#ap-zv").text(d3.select("#ap-z").property("value")); d3.select("#ap-stv").text(d3.select("#ap-st").property("value"));
      const {z,s}=scheme();
      const shift=d3.select("#ap-shift").property("checked"), showCov=d3.select("#ap-cov").property("checked");
      if(!M||M.length!==n) M=makeMap(n,seed);
      const off=shift?1:0;
      const m=outSize(n,z,s);
      const P0=pool(M,n,z,s,0), P=pool(M,n,z,s,off), cov=coverage(n,z,s);
      svg.selectAll("*").remove();

      const cell=Math.min(17,330/n), ix=20, iy=44;
      const ocell=Math.min(30,260/Math.max(m,1)), ox=430, oy=44;
      const vmax=d3.max(M,r=>d3.max(r))||1;
      const col=v=>d3.interpolateRgb("#1b1f27",C.accent)(Math.min(1,v/vmax));
      const covCol=k=>k===0?"#3a1f24":["#1b1f27","#24414a","#2f6f6a","#3aa596","#4ade80"][Math.min(4,k)];

      svg.append("text").attr("x",ix).attr("y",22).attr("fill",C.muted).attr("font-size",11)
        .text(`input map ${n}×${n}${shift?" — shifted one pixel left":""}${showCov?" · colour = windows covering the cell":" · ReLU-like activations"}`);
      for(let i=0;i<n;i++)for(let j=0;j<n;j++){
        svg.append("rect").attr("x",ix+j*cell).attr("y",iy+i*cell).attr("width",cell-1).attr("height",cell-1)
          .attr("fill",showCov?covCol(cov[i][j]):col(M[i][off+j])).attr("stroke",showCov&&cov[i][j]===0?C.bad:"none");
        if(showCov&&cell>=13) svg.append("text").attr("x",ix+j*cell+cell/2-0.5).attr("y",iy+i*cell+cell/2+3.5).attr("text-anchor","middle")
          .attr("fill",C.ink).attr("font-size",9).text(cov[i][j]);
      }
      // window origins
      for(let i=0;i<m;i++)for(let j=0;j<m;j++)
        svg.append("circle").attr("cx",ix+(j*s+z/2)*cell-0.5).attr("cy",iy+(i*s+z/2)*cell-0.5).attr("r",1.6).attr("fill",C.a2).attr("opacity",0.7);
      const hv=hov&&hov[0]<m&&hov[1]<m?hov:[0,0];
      svg.append("rect").attr("x",ix+hv[1]*s*cell-1).attr("y",iy+hv[0]*s*cell-1).attr("width",z*cell+1).attr("height",z*cell+1)
        .attr("fill","none").attr("stroke",C.a2).attr("stroke-width",2);
      if(hv[1]+1<m) svg.append("rect").attr("x",ix+(hv[1]+1)*s*cell-1).attr("y",iy+hv[0]*s*cell-1).attr("width",z*cell+1).attr("height",z*cell+1)
        .attr("fill","none").attr("stroke",C.violet).attr("stroke-width",1.4).attr("stroke-dasharray","4 3");

      svg.append("text").attr("x",ox).attr("y",22).attr("fill",C.muted).attr("font-size",11)
        .text(`pooled ${m}×${m} · z = ${z}, s = ${s}${shift?" · red outline = value changed by the shift":""}`);
      for(let i=0;i<m;i++)for(let j=0;j<m;j++){
        const changed=shift&&P[i][j]!==P0[i][j];
        svg.append("rect").attr("x",ox+j*ocell).attr("y",oy+i*ocell).attr("width",ocell-2).attr("height",ocell-2)
          .attr("fill",col(P[i][j])).attr("stroke",(hv[0]===i&&hv[1]===j)?C.a2:changed?C.bad:"none").attr("stroke-width",2)
          .style("cursor","pointer").on("mouseenter",()=>{if(hov&&hov[0]===i&&hov[1]===j)return;hov=[i,j];draw();});
      }
      svg.append("text").attr("x",ox).attr("y",oy+m*ocell+18).attr("fill",C.muted).attr("font-size",10.5)
        .text("hover an output cell: amber = its window, dashed violet = its right-hand neighbour's");

      // stability over many random maps, for the two schemes the paper compares (and custom)
      if(cacheN!==n){stab={ov:unchangedFrac(n,3,2),non:unchangedFrac(n,2,2)};cacheN=n;}
      const cur=(z===3&&s===2)?stab.ov:(z===2&&s===2)?stab.non:(cust[n+"/"+z+"/"+s]??(cust[n+"/"+z+"/"+s]=unchangedFrac(n,z,s)));
      let k0=0,k1=0,kmany=0;
      cov.forEach(r=>r.forEach(k=>{if(k===0)k0++;else if(k===1)k1++;else kmany++;}));
      const chain=(z2)=>{const a=[55];for(let q=0;q<3;q++){const prev=a[a.length-1];a.push(outSize(prev,z2,2));}return a;};
      const c3=chain(3), c2=chain(2);
      const changedNow=shift?d3.sum(P,(r,i)=>d3.sum(r,(v,j)=>v!==P0[i][j]?1:0)):null;
      d3.select("#ap-readout").html(
        `output size ⌊(n − z)/s⌋ + 1 = ⌊(${n} − ${z})/${s}⌋ + 1 = <b>${m}</b> · input cells covered 0× <b>${k0}</b>, 1× <b>${k1}</b>, 2× or more <b>${kmany}</b>`+
        (k0?` — <b>${k0} cells are never seen by any window</b> (the last row/column when n − z is odd)`:"")+
        (shift?` · this map: <b>${changedNow}</b> of ${m*m} outputs changed under the one-pixel shift`:"")+
        `<br>over ${TRIALS} random maps of this size, fraction of pooled values unchanged by a one-pixel shift: overlapping z=3,s=2 <b>${(100*stab.ov).toFixed(1)}%</b> · non-overlapping z=2,s=2 <b>${(100*stab.non).toFixed(1)}%</b>`+
        ((z===3&&s===2)||(z===2&&s===2)?"":` · current z=${z},s=${s} <b>${isNaN(cur)?"—":(100*cur).toFixed(1)+"%"}</b>`)+
        `<br>the network's three pooling stages from a 55×55 map: z = 3 gives ${c3.join(" → ")}, z = 2 gives ${c2.join(" → ")} — equal output sizes, as the paper says`);
    }
    d3.select("#ap-new").on("click",()=>{seed+=1;M=null;draw();});
    ["#ap-s","#ap-shift","#ap-cov"].forEach(id=>d3.select(id).on("change",draw));
    ["#ap-n","#ap-z","#ap-st"].forEach(id=>d3.select(id).on("input",()=>{if(id==="#ap-n")M=null;draw();}));
    draw();
  })();
})();
