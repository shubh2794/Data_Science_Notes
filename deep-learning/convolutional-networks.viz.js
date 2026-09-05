/* convolutional-networks.viz.js — extracted from convolutional-networks.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ───────────────────────── 02 · kernel sweep ───────────────────────── */
(function(){
  const svg=d3.select("#cnn-svg"), N=8, cell=26, ox=18, oy=44;
  const kx=272, ky=64, kcell=32;               // kernel panel
  const oox=430, ooy=44;                       // output grid
  const presets={
    edge:[[0,-1,0],[-1,4,-1],[0,-1,0]],
    sobelx:[[1,0,-1],[2,0,-2],[1,0,-1]],
    sobely:[[1,2,1],[0,0,0],[-1,-2,-1]],
    sharpen:[[0,-1,0],[-1,5,-1],[0,-1,0]],
    blur:[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]],
    identity:[[0,0,0],[0,1,0],[0,0,0]]
  };
  const CYCLE=[0,1,2,-1,-2];                   // click a kernel cell to cycle
  let K=presets.edge.map(r=>r.slice());
  // synthetic 8x8 "image": bright diagonal band + a darker quadrant
  const img=d3.range(N).map((_,r)=>d3.range(N).map((__,c)=>
    Math.abs(r-c)<2 ? 0.92 : (r+c>9 ? 0.5 : 0.14)));
  const out=d3.range(N-2).map(()=>d3.range(N-2).fill(null));
  const readout=document.getElementById("cnn-readout");
  const grey=d3.interpolateGreys;
  let timer=null, cur=0;                       // cur = flat index of next window

  const lab=(x,y,t)=>svg.append("text").attr("x",x).attr("y",y).attr("fill",C.muted).attr("font-size",11).text(t);
  lab(ox,32,"Input  8×8");
  lab(kx,32,"Kernel 3×3");
  lab(oox,32,"Feature map  6×6");
  svg.append("text").attr("x",kx).attr("y",ky+3*kcell+22).attr("fill",C.muted).attr("font-size",10).text("click a cell to edit");

  const gIn=svg.append("g").attr("transform",`translate(${ox},${oy})`);
  const gK=svg.append("g").attr("transform",`translate(${kx},${ky})`);
  const gOut=svg.append("g").attr("transform",`translate(${oox},${ooy})`);

  const inCells=[]; for(let r=0;r<N;r++) for(let c=0;c<N;c++) inCells.push({r,c,v:img[r][c]});
  gIn.selectAll("rect").data(inCells).join("rect")
    .attr("x",d=>d.c*cell).attr("y",d=>d.r*cell).attr("width",cell-1).attr("height",cell-1)
    .attr("fill",d=>grey(d.v));
  const hi=gIn.append("rect").attr("width",3*cell-1).attr("height",3*cell-1)
    .attr("fill","none").attr("stroke",C.A).attr("stroke-width",2.5).attr("rx",2).style("opacity",0);

  function kmax(){ let s=0; for(let i=0;i<3;i++) for(let j=0;j<3;j++) s+=Math.abs(K[i][j]); return Math.max(s,1e-6); }
  function conv(r,c){ let s=0; for(let i=0;i<3;i++) for(let j=0;j<3;j++) s+=img[r+i][c+j]*K[i][j]; return s; }

  function drawK(){
    const cells=[]; for(let i=0;i<3;i++) for(let j=0;j<3;j++) cells.push({i,j,v:K[i][j]});
    const g=gK.selectAll("g.kc").data(cells).join("g").attr("class","kc")
      .attr("transform",d=>`translate(${d.j*kcell},${d.i*kcell})`).style("cursor","pointer")
      .on("click",(e,d)=>{
        const idx=CYCLE.indexOf(Math.round(d.v*100)/100);
        K[d.i][d.j]=CYCLE[(idx<0?0:idx+1)%CYCLE.length];
        d3.select("#cnn-kernel").property("value","custom");
        drawK(); reset();
      });
    g.selectAll("rect").data(d=>[d]).join("rect")
      .attr("width",kcell-2).attr("height",kcell-2).attr("rx",4)
      .attr("fill",d=>d.v>0?"rgba(91,156,255,.22)":d.v<0?"rgba(248,113,113,.20)":"#171b24")
      .attr("stroke",d=>d.v>0?C.A:d.v<0?C.bad:C.line);
    g.selectAll("text").data(d=>[d]).join("text")
      .attr("x",(kcell-2)/2).attr("y",(kcell-2)/2+4).attr("text-anchor","middle")
      .attr("font-size",11).attr("fill",C.ink)
      .text(d=>Math.abs(d.v-1/9)<1e-6?"⅑":(+d.v.toFixed(2)));
  }

  function drawOut(){
    const m=kmax();
    const cells=[]; for(let r=0;r<N-2;r++) for(let c=0;c<N-2;c++) cells.push({r,c,v:out[r][c]});
    gOut.selectAll("rect").data(cells).join("rect")
      .attr("x",d=>d.c*cell).attr("y",d=>d.r*cell).attr("width",cell-1).attr("height",cell-1)
      .attr("fill",d=>d.v===null?"#171b24":d3.interpolateRdBu(1-(0.5+d.v/(2*m))))
      .style("cursor",d=>d.v===null?"default":"pointer")
      .on("mouseover",(e,d)=>{ if(d.v!==null) explain(d.r,d.c); });
  }

  function explain(r,c){
    hi.style("opacity",1).attr("x",c*cell).attr("y",r*cell);
    const terms=[];
    for(let i=0;i<3;i++) for(let j=0;j<3;j++)
      if(Math.abs(K[i][j])>1e-9) terms.push(`${img[r+i][c+j].toFixed(2)}·${(+K[i][j].toFixed(2))}`);
    readout.innerHTML=`output[${r},${c}] = ${terms.join(" + ")||"0"} = <b>${conv(r,c).toFixed(2)}</b>`;
  }

  function reset(){
    if(timer){ clearInterval(timer); timer=null; }
    for(let r=0;r<N-2;r++) for(let c=0;c<N-2;c++) out[r][c]=null;
    cur=0; drawOut(); hi.style("opacity",0);
    readout.innerHTML="6×6 output = (8 − 3)/1 + 1 per axis. Blue = positive response, red = negative.";
  }
  function step(){
    if(cur>=(N-2)*(N-2)) return false;
    const r=Math.floor(cur/(N-2)), c=cur%(N-2);
    out[r][c]=conv(r,c); drawOut(); explain(r,c); cur++;
    return true;
  }
  function animate(){
    reset();
    timer=setInterval(()=>{ if(!step()){ clearInterval(timer); timer=null; setTimeout(()=>hi.style("opacity",0),500); } },70);
  }

  d3.select("#cnn-kernel").on("change",function(){
    const v=this.value; if(presets[v]) K=presets[v].map(r=>r.slice());
    drawK(); reset();
  });
  d3.select("#cnn-play").on("click",animate);
  d3.select("#cnn-step").on("click",()=>{ if(timer){clearInterval(timer);timer=null;} step(); });
  d3.select("#cnn-reset").on("click",reset);
  drawK(); reset();
})();

/* ───────────────────────── 03 · conv geometry + receptive field ───────────────────────── */
(function(){
  const svg=d3.select("#geo-svg"), W=640;
  const gIn=svg.append("g"), gOut=svg.append("g"), gLab=svg.append("g");
  const el=id=>document.getElementById(id);
  const iEl=el("geo-i"), kEl=el("geo-k"), sEl=el("geo-s"), pEl=el("geo-p"), dEl=el("geo-d"), posEl=el("geo-pos");
  const out=el("geo-readout");

  function render(){
    const i=+iEl.value, k=+kEl.value, s=+sEl.value, p=+pEl.value, d=+dEl.value;
    el("geo-ival").textContent=i; el("geo-sval").textContent=s;
    el("geo-pval").textContent=p; el("geo-dval").textContent=d;

    const keff=d*(k-1)+1;
    const o=Math.floor((i+2*p-keff)/s)+1;
    const valid=o>0;
    const nOut=valid?o:0;
    posEl.max=Math.max(0,nOut*nOut-1);
    if(+posEl.value>+posEl.max) posEl.value=posEl.max;
    const pos=+posEl.value, pr=nOut?Math.floor(pos/nOut):0, pc=nOut?pos%nOut:0;

    const padded=i+2*p;
    const cs=Math.min(22,Math.floor(200/Math.max(padded,1)));
    const ox=24, oy=56;
    gIn.selectAll("*").remove(); gOut.selectAll("*").remove(); gLab.selectAll("*").remove();

    // padded input grid
    for(let r=0;r<padded;r++) for(let c=0;c<padded;c++){
      const isPad=r<p||c<p||r>=p+i||c>=p+i;
      gIn.append("rect").attr("x",ox+c*cs).attr("y",oy+r*cs).attr("width",cs-1.5).attr("height",cs-1.5).attr("rx",2)
        .attr("fill",isPad?"#12151c":"#232a38")
        .attr("stroke",isPad?C.line:"#39445a").attr("stroke-dasharray",isPad?"2 2":null);
    }
    // dilated kernel taps for the selected window
    if(valid){
      for(let a=0;a<k;a++) for(let b=0;b<k;b++){
        const rr=pr*s+a*d, cc=pc*s+b*d;
        gIn.append("rect").attr("x",ox+cc*cs).attr("y",oy+rr*cs).attr("width",cs-1.5).attr("height",cs-1.5).attr("rx",2)
          .attr("fill",C.A).attr("fill-opacity",.75);
      }
      // dashed box over the effective extent
      gIn.append("rect").attr("x",ox+pc*s*cs-2).attr("y",oy+pr*s*cs-2)
        .attr("width",keff*cs+2).attr("height",keff*cs+2)
        .attr("fill","none").attr("stroke",C.A).attr("stroke-opacity",.5).attr("stroke-dasharray","3 3");
    }
    gLab.append("text").attr("x",ox).attr("y",oy-14).attr("fill",C.muted).attr("font-size",11)
      .text(`input ${i}×${i}` + (p?`  + padding ${p}  → ${padded}×${padded}`:""));

    // arrow
    const ax=ox+padded*cs+30;
    gLab.append("text").attr("x",ax+18).attr("y",oy+60).attr("fill",C.muted).attr("font-size",18).text("→");

    // output grid
    const ox2=ax+60, cs2=Math.min(26,Math.floor(180/Math.max(nOut,1)));
    if(valid){
      for(let r=0;r<nOut;r++) for(let c=0;c<nOut;c++){
        const on=(r===pr&&c===pc);
        gOut.append("rect").attr("x",ox2+c*cs2).attr("y",oy+r*cs2).attr("width",cs2-1.5).attr("height",cs2-1.5).attr("rx",2)
          .attr("fill",on?C.B:"#232a38").attr("fill-opacity",on?.85:1)
          .attr("stroke",on?C.B:"#39445a");
      }
      gLab.append("text").attr("x",ox2).attr("y",oy-14).attr("fill",C.muted).attr("font-size",11).text(`output ${o}×${o}`);
    } else {
      gLab.append("text").attr("x",ox2).attr("y",oy+20).attr("fill",C.bad).attr("font-size",12)
        .text("kernel larger than the padded input");
    }

    // receptive field of L stacked identical layers
    const rf=[]; let r=1, j=1;
    for(let L=1;L<=4;L++){ r=r+(keff-1)*j; j=j*s; rf.push(r); }
    out.innerHTML=`k<sub>eff</sub> = ${d}·(${k} − 1) + 1 = <b>${keff}</b> &nbsp;·&nbsp; `+
      `o = ⌊(${i} + 2·${p} − ${keff})/${s}⌋ + 1 = <b style="color:${valid?C.good:C.bad}">${valid?o:"invalid"}</b>`+
      `<br>receptive field after 1–4 such layers: <b>${rf.join(" → ")}</b> px &nbsp;·&nbsp; jump after 4 layers: <b>${Math.pow(s,4)}</b>`;
  }
  [iEl,sEl,pEl,dEl,posEl].forEach(e=>e.addEventListener("input",render));
  kEl.addEventListener("change",render);
  render();
})();

/* ───────────────────────── 06 · pooling vs strided conv ───────────────────────── */
(function(){
  const svg=d3.select("#pool-svg"), N=6, cell=30, ox=24, oy=52, oox=380, ooy=68;
  const gIn=svg.append("g"), gOut=svg.append("g"), gLab=svg.append("g"), gHi=svg.append("g");
  const out=document.getElementById("pool-readout");
  const grid=[[2,7,1,0,4,3],[8,3,2,9,1,0],[0,1,6,5,2,7],[4,2,3,1,8,2],[9,0,1,2,3,6],[1,5,4,0,7,2]];
  const Wk=[[0.6,-0.4],[-0.2,0.8]];              // a "learned" 2×2 strided-conv kernel
  const col=d3.scaleLinear().domain([0,9]).range(["#161a22","#3f5f9e"]);
  let mode="max";

  gLab.append("text").attr("x",ox).attr("y",34).attr("fill",C.muted).attr("font-size",11).text("input 6×6");
  gLab.append("text").attr("x",oox).attr("y",34).attr("fill",C.muted).attr("font-size",11).text("output 3×3 (stride 2)");
  gLab.append("text").attr("x",oox).attr("y",ooy+3*cell+34).attr("fill",C.muted).attr("font-size",10)
    .text("conv kernel [[0.6, −0.4], [−0.2, 0.8]]");

  for(let r=0;r<N;r++) for(let c=0;c<N;c++){
    const g=gIn.append("g").attr("transform",`translate(${ox+c*cell},${oy+r*cell})`);
    g.append("rect").attr("width",cell-2).attr("height",cell-2).attr("rx",3).attr("fill",col(grid[r][c])).attr("stroke",C.line);
    g.append("text").attr("x",(cell-2)/2).attr("y",(cell-2)/2+4).attr("text-anchor","middle")
      .attr("font-size",11).attr("fill",C.ink).text(grid[r][c]);
  }
  // window separators
  for(let r=0;r<3;r++) for(let c=0;c<3;c++)
    gIn.append("rect").attr("x",ox+c*2*cell-2).attr("y",oy+r*2*cell-2).attr("width",2*cell).attr("height",2*cell)
      .attr("fill","none").attr("stroke","#4a5570").attr("stroke-opacity",.8).attr("rx",3);

  function compute(r,c){
    const v=[grid[2*r][2*c],grid[2*r][2*c+1],grid[2*r+1][2*c],grid[2*r+1][2*c+1]];
    if(mode==="max") return {val:Math.max(...v), pick:v.indexOf(Math.max(...v))};
    if(mode==="avg") return {val:v.reduce((a,b)=>a+b,0)/4, pick:-1};
    return {val:v[0]*Wk[0][0]+v[1]*Wk[0][1]+v[2]*Wk[1][0]+v[3]*Wk[1][1], pick:-1};
  }
  function render(){
    gOut.selectAll("*").remove(); gHi.selectAll("*").remove();
    let vals=[];
    for(let r=0;r<3;r++) for(let c=0;c<3;c++){
      const {val,pick}=compute(r,c); vals.push(val);
      const g=gOut.append("g").attr("transform",`translate(${oox+c*cell},${ooy+r*cell})`);
      g.append("rect").attr("width",cell-2).attr("height",cell-2).attr("rx",3)
        .attr("fill",col(Math.max(0,Math.min(9,mode==="conv"?val+4:val)))).attr("stroke",C.B).attr("stroke-opacity",.5);
      g.append("text").attr("x",(cell-2)/2).attr("y",(cell-2)/2+4).attr("text-anchor","middle")
        .attr("font-size",10.5).attr("fill",C.ink).text(mode==="avg"||mode==="conv"?val.toFixed(1):val);
      if(pick>=0){
        const pr=2*r+Math.floor(pick/2), pc=2*c+pick%2;
        gHi.append("rect").attr("x",ox+pc*cell-1).attr("y",oy+pr*cell-1).attr("width",cell).attr("height",cell)
          .attr("fill","none").attr("stroke",C.good).attr("stroke-width",2).attr("rx",3);
      }
    }
    const txt={
      max:`<b>Max pool</b> keeps the largest value in each 2×2 window (highlighted green). No parameters; the gradient reaches only the winner. Sharp, feature-presence oriented.`,
      avg:`<b>Average pool</b> keeps the mean of each window. No parameters; the gradient is shared by all four cells. Smoother, keeps overall intensity.`,
      conv:`<b>Strided conv</b> applies <i>learned</i> weights, so it can produce negative values and can learn to imitate either pooling — at the cost of ${2*2}·C<sub>in</sub>·C<sub>out</sub> parameters.`
    }[mode];
    out.innerHTML=txt+` &nbsp;·&nbsp; output mean = <b>${(vals.reduce((a,b)=>a+b,0)/9).toFixed(2)}</b>`;
    d3.select("#pool-max").attr("class",mode==="max"?"btn":"btn ghost");
    d3.select("#pool-avg").attr("class",mode==="avg"?"btn":"btn ghost");
    d3.select("#pool-conv").attr("class",mode==="conv"?"btn":"btn ghost");
  }
  d3.select("#pool-max").on("click",()=>{mode="max";render();});
  d3.select("#pool-avg").on("click",()=>{mode="avg";render();});
  d3.select("#pool-conv").on("click",()=>{mode="conv";render();});
  render();
})();
