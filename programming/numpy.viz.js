/* numpy.viz.js — extracted from numpy.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

const PC={accent:"#5b9cff",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a",cell:"#1b2130",cellb:"#2a2f3a"};

/* ───────────────────────── 03 · views vs copies / strides playground ───────────────────────── */
(function(){
  const svg=d3.select("#views-svg"),W=640,H=300;
  const out=document.getElementById("views-readout");
  const n=8, base=[0,1,2,3,4,5,6,7].map(i=>i*2); // x = [0,2,4,6,8,10,12,14]
  let parent=base.slice();
  let mode=null;            // "slice" | "fancy" | "bool"
  let selIdx=[];            // positions selected
  let isView=false;
  let child=null;           // for copies: separate buffer values
  const cw=58, ch=46, x0=40, yP=70, yC=200;

  function selFor(m){
    if(m==="slice") return {idx:[1,3,5], view:true, label:"x[1:7:2]", note:"basic slice → VIEW (start,stop,step) · shares the buffer"};
    if(m==="fancy") return {idx:[0,3,5], view:false, label:"x[[0,3,5]]", note:"fancy index → COPY (explicit positions) · new buffer"};
    if(m==="bool")  return {idx:parent.map((v,i)=>v>4?i:-1).filter(i=>i>=0), view:false, label:"x[x>4]", note:"boolean mask → COPY · new buffer"};
    return null;
  }
  function pick(m){
    mode=m; const s=selFor(m); selIdx=s.idx; isView=s.view;
    child = isView ? null : selIdx.map(i=>parent[i]);
    draw(s);
  }
  function write(){
    if(mode==null) return;
    if(isView){ selIdx.forEach(i=>parent[i]=99); }   // view: parent changes
    else { child=selIdx.map(()=>99); }               // copy: only child changes
    draw(selFor(mode), true);
  }
  function reset(){ parent=base.slice(); mode=null; selIdx=[]; child=null; isView=false;
    svg.selectAll("*").remove(); drawParent(); out.innerHTML="<span style='color:var(--muted)'>buffer x = ["+base.join(", ")+"]. Pick a selection.</span>"; }

  function drawParent(){
    svg.append("text").attr("x",x0).attr("y",yP-16).attr("fill",PC.muted).attr("font-size",12).text("parent buffer  x");
    const g=svg.selectAll(".pc").data(parent).enter().append("g").attr("class","pc");
    g.append("rect").attr("class","prect").attr("x",(d,i)=>x0+i*cw).attr("y",yP).attr("width",cw-6).attr("height",ch)
      .attr("rx",6).attr("fill",PC.cell).attr("stroke",PC.cellb);
    g.append("text").attr("class","ptxt").attr("x",(d,i)=>x0+i*cw+(cw-6)/2).attr("y",yP+ch/2+5).attr("text-anchor","middle").attr("fill",PC.ink).attr("font-size",15).text(d=>d);
    g.append("text").attr("x",(d,i)=>x0+i*cw+(cw-6)/2).attr("y",yP+ch+14).attr("text-anchor","middle").attr("fill",PC.muted).attr("font-size",10).text((d,i)=>"["+i+"]");
  }

  function draw(s, wrote){
    svg.selectAll("*").remove();
    drawParent();
    // recolor parent cells: selected ones
    svg.selectAll(".prect").attr("fill",(d,i)=>selIdx.includes(i)?(isView?"rgba(91,156,255,.32)":"rgba(154,163,178,.18)"):PC.cell)
       .attr("stroke",(d,i)=>selIdx.includes(i)?(isView?PC.accent:PC.cellb):PC.cellb);
    svg.selectAll(".ptxt").attr("fill",(d,i)=>(wrote&&isView&&d===99)?PC.bad:PC.ink).attr("font-weight",(d,i)=>(wrote&&isView&&selIdx.includes(i))?700:400);

    // child row
    const childVals = isView ? selIdx.map(i=>parent[i]) : child;
    const cy = yC;
    const labelTxt = (isView?"VIEW  y = ":"COPY  z = ")+s.label;
    svg.append("text").attr("x",x0).attr("y",cy-16).attr("fill",isView?PC.accent:PC.a2).attr("font-size",12).attr("font-weight",600).text(labelTxt);
    if(isView){
      // stride arrows from parent selected cells down to child
      selIdx.forEach((pi,k)=>{
        const sx=x0+pi*cw+(cw-6)/2, dx=x0+k*cw+(cw-6)/2;
        svg.append("line").attr("x1",sx).attr("y1",yP+ch+20).attr("x2",dx).attr("y2",cy-6)
          .attr("stroke",PC.accent).attr("stroke-opacity",.5).attr("stroke-dasharray","3 3");
      });
    }
    const cg=svg.selectAll(".cc").data(childVals).enter().append("g");
    cg.append("rect").attr("x",(d,i)=>x0+i*cw).attr("y",cy).attr("width",cw-6).attr("height",ch).attr("rx",6)
      .attr("fill",isView?"rgba(91,156,255,.18)":"rgba(255,180,84,.16)").attr("stroke",isView?PC.accent:PC.a2)
      .attr("stroke-dasharray",isView?null:"4 3");
    cg.append("text").attr("x",(d,i)=>x0+i*cw+(cw-6)/2).attr("y",cy+ch/2+5).attr("text-anchor","middle")
      .attr("fill",(d)=>(wrote&&d===99)?PC.bad:PC.ink).attr("font-size",15).attr("font-weight",(d)=>(wrote&&d===99)?700:400).text(d=>d);

    svg.append("text").attr("x",x0).attr("y",H-14).attr("fill",PC.muted).attr("font-size",11).text(s.note);

    out.innerHTML = `selection <b>${s.label}</b> → <b style="color:${isView?PC.accent:PC.a2}">${isView?"VIEW (.base is x)":"COPY (.base is None)"}</b>`
      + (wrote ? ` · wrote 99 → parent ${isView?`<b style="color:${PC.bad}">CHANGED</b>`:`<b style="color:${PC.good}">unchanged</b>`}` : ` · now press “Write 99”`);
  }

  document.getElementById("sel-slice").addEventListener("click",()=>pick("slice"));
  document.getElementById("sel-fancy").addEventListener("click",()=>pick("fancy"));
  document.getElementById("sel-bool").addEventListener("click",()=>pick("bool"));
  document.getElementById("sel-write").addEventListener("click",write);
  document.getElementById("sel-reset").addEventListener("click",reset);
  reset();
})();

/* ───────────────────────── 04 · broadcasting explorer ───────────────────────── */
(function(){
  const svg=d3.select("#bcast-svg"),W=640,H=340;
  const out=document.getElementById("bcast-readout");
  const A=[4,3];                 // fixed operand A shape
  const shapes={ "b-3":[3], "b-41":[4,1], "b-13":[1,3], "b-2":[2] };
  const cell=30, gap=4, gridX=40, gridY=58;

  // compatibility: align right-to-left, dims compatible if equal or one is 1
  function broadcast(a,b){
    const la=a.length, lb=b.length, L=Math.max(la,lb);
    const pa=[], pb=[], res=[], ok=[];
    let allOk=true;
    for(let k=0;k<L;k++){
      const da = k<la ? a[la-1-k] : 1;
      const db = k<lb ? b[lb-1-k] : 1;
      const compat = (da===db)||da===1||db===1;
      if(!compat) allOk=false;
      pa.unshift(da); pb.unshift(db); ok.unshift(compat);
      res.unshift(compat ? Math.max(da,db) : null);
    }
    return {pa,pb,res,ok,allOk};
  }

  function drawGrid(g,rows,cols,color,ghostCols,ghostRows){
    g.selectAll("*").remove();
    for(let r=0;r<rows;r++) for(let c=0;c<cols;c++){
      const ghost = (ghostCols&&cols===1&&false); // placeholder
      g.append("rect").attr("x",c*(cell+gap)).attr("y",r*(cell+gap)).attr("width",cell).attr("height",cell)
        .attr("rx",4).attr("fill",color.fill).attr("stroke",color.stroke);
    }
  }

  function draw(key){
    const b=shapes[key];
    const bc=broadcast(A,b);
    svg.selectAll("*").remove();

    // ── A grid (4×3) ──
    const gA=svg.append("g").attr("transform",`translate(${gridX},${gridY})`);
    drawGrid(gA,A[0],A[1],{fill:"rgba(91,156,255,.14)",stroke:PC.accent});
    svg.append("text").attr("x",gridX).attr("y",gridY-12).attr("fill",PC.accent).attr("font-size",12).attr("font-weight",600).text("A  (4, 3)");

    // ── b grid, virtually stretched to its broadcast footprint over A ──
    const bRows = b.length===2 ? b[0] : 1;
    const bCols = b.length===2 ? b[1] : b[0];
    const bx=gridX+ A[1]*(cell+gap) + 70;
    const gB=svg.append("g").attr("transform",`translate(${bx},${gridY})`);
    // draw real b cells solid, then ghost-stretch size-1 axes across A's footprint
    const tRows = bc.allOk ? Math.max(A[0],bRows) : bRows;
    const tCols = bc.allOk ? Math.max(A[1],bCols) : bCols;
    for(let r=0;r<tRows;r++) for(let c=0;c<tCols;c++){
      const isReal = (r<bRows)&&(c<bCols);
      const ghost = bc.allOk && !((bRows>1&&bCols>1)) && !(r<bRows && c<bCols && bRows===tRows && bCols===tCols);
      const stretched = bc.allOk && ((bRows===1&&tRows>1)||(bCols===1&&tCols>1)) && !isReal;
      gB.append("rect").attr("x",c*(cell+gap)).attr("y",r*(cell+gap)).attr("width",cell).attr("height",cell).attr("rx",4)
        .attr("fill", isReal ? "rgba(255,180,84,.22)" : (stretched? "rgba(255,180,84,.07)":"rgba(255,180,84,.22)"))
        .attr("stroke", isReal ? PC.a2 : PC.a2)
        .attr("stroke-opacity", isReal?1:.35)
        .attr("stroke-dasharray", isReal?null:"3 3");
    }
    svg.append("text").attr("x",bx).attr("y",gridY-12).attr("fill",PC.a2).attr("font-size",12).attr("font-weight",600).text("b  ("+b.join(", ")+(b.length===1?",":"")+")");

    // ── alignment ladder (right-to-left) ──
    const ay=gridY + Math.max(A[0],tRows)*(cell+gap) + 36;
    svg.append("text").attr("x",gridX).attr("y",ay-10).attr("fill",PC.muted).attr("font-size",11).text("align right-to-left:");
    const L=bc.pa.length, bw=46, bx0=gridX+150;
    for(let k=0;k<L;k++){
      const X=bx0+k*(bw+8);
      const col = bc.ok[k]?PC.good:PC.bad;
      svg.append("rect").attr("x",X).attr("y",ay-22).attr("width",bw).attr("height",20).attr("rx",4)
        .attr("fill","none").attr("stroke",col);
      svg.append("text").attr("x",X+bw/2).attr("y",ay-8).attr("text-anchor","middle").attr("fill",PC.ink).attr("font-size",12).text(bc.pa[k]);
      svg.append("rect").attr("x",X).attr("y",ay+2).attr("width",bw).attr("height",20).attr("rx",4)
        .attr("fill","none").attr("stroke",col);
      svg.append("text").attr("x",X+bw/2).attr("y",ay+16).attr("text-anchor","middle").attr("fill",PC.ink).attr("font-size",12).text(bc.pb[k]);
      svg.append("text").attr("x",X+bw/2).attr("y",ay+38).attr("text-anchor","middle").attr("fill",col).attr("font-size",13).attr("font-weight",700)
        .text(bc.ok[k]? (bc.res[k]) : "✗");
    }
    svg.append("text").attr("x",bx0-12).attr("y",ay-8).attr("text-anchor","end").attr("fill",PC.accent).attr("font-size",11).text("A");
    svg.append("text").attr("x",bx0-12).attr("y",ay+16).attr("text-anchor","end").attr("fill",PC.a2).attr("font-size",11).text("b");

    if(bc.allOk){
      out.innerHTML = `A (${A.join(", ")}) ⊕ b (${b.join(", ")}) → <b style="color:${PC.good}">result (${bc.res.join(", ")})</b> · size-1 axes virtually stretched (stride 0, no copy)`;
    } else {
      out.innerHTML = `A (${A.join(", ")}) ⊕ b (${b.join(", ")}) → <b style="color:${PC.bad}">ValueError</b> · a trailing dim is neither equal nor 1`;
    }
  }

  Object.keys(shapes).forEach(id=>document.getElementById(id).addEventListener("click",()=>draw(id)));
  draw("b-3");
})();
