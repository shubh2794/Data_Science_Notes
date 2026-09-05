/* embeddings.viz.js — extracted from embeddings.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

const PC={accent:"#c084fc",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};

/* ───────────────────────── 01 · embedding space + analogy ───────────────────────── */
(function(){
  const data=[
    {w:"man",x:2,y:2,g:0},{w:"woman",x:2,y:5,g:0},{w:"boy",x:0.8,y:2.2,g:0},{w:"girl",x:0.8,y:4.8,g:0},
    {w:"king",x:7,y:2,g:1},{w:"queen",x:7,y:5,g:1},{w:"prince",x:6,y:2.4,g:1},{w:"princess",x:6,y:4.6,g:1},
    {w:"cat",x:3,y:9,g:2},{w:"dog",x:3.6,y:9.4,g:2},{w:"lion",x:2.4,y:9.6,g:2},{w:"kitten",x:3.1,y:8.3,g:2},
    {w:"apple",x:10.5,y:9,g:3},{w:"bread",x:11,y:9.6,g:3},{w:"pizza",x:10,y:8.4,g:3},{w:"coffee",x:11.3,y:8.7,g:3},
    {w:"data",x:11,y:1.6,g:4},{w:"code",x:11.6,y:2.2,g:4},{w:"neural",x:10.4,y:2.4,g:4},{w:"model",x:11,y:3,g:4},
  ];
  const groups=["#5b9cff","#c084fc","#ffb454","#4ade80","#f472b6"];
  const svg=d3.select("#emb-svg"),W=640,H=420,m={t:20,r:20,b:20,l:20};
  const x=d3.scaleLinear().domain([-1,13]).range([m.l,W-m.r]);
  const y=d3.scaleLinear().domain([-1,11]).range([H-m.b,m.t]);
  const gLines=svg.append("g"),gPts=svg.append("g");
  ["emb-a","emb-b","emb-c"].forEach((id)=>{ const sel=d3.select("#"+id); data.forEach(d=>sel.append("option").attr("value",d.w).text(d.w)); });
  d3.select("#emb-a").property("value","king"); d3.select("#emb-b").property("value","man"); d3.select("#emb-c").property("value","woman");
  function draw(){
    const g=gPts.selectAll("g.pt").data(data).join("g").attr("class","pt").attr("transform",d=>`translate(${x(d.x)},${y(d.y)})`);
    g.selectAll("circle").data(d=>[d]).join("circle").attr("r",6).attr("fill",d=>groups[d.g]).attr("stroke","#0f1117").attr("stroke-width",1.5).attr("cursor","pointer");
    g.selectAll("text").data(d=>[d]).join("text").attr("x",9).attr("dy",4).attr("font-size",11).attr("fill",PC.ink).attr("paint-order","stroke").attr("stroke","#0f1117").attr("stroke-width",3).attr("stroke-linejoin","round").text(d=>d.w);
    g.on("mouseover",(e,d)=>neighbors(d)).on("mouseout",()=>{ gLines.selectAll("*").remove(); d3.select("#emb-readout").text(""); });
  }
  function neighbors(d){
    gLines.selectAll("*").remove();
    const ds=data.filter(o=>o.w!==d.w).map(o=>({o,dist:Math.hypot(o.x-d.x,o.y-d.y)})).sort((a,b)=>a.dist-b.dist).slice(0,3);
    ds.forEach(({o})=>gLines.append("line").attr("x1",x(d.x)).attr("y1",y(d.y)).attr("x2",x(o.x)).attr("y2",y(o.y)).attr("stroke",PC.accent).attr("stroke-width",1.5).attr("stroke-opacity",.6));
    d3.select("#emb-readout").html(`<b>${d.w}</b> ≈ ${ds.map(n=>n.o.w).join(", ")}`);
  }
  function solve(){
    gLines.selectAll("*").remove();
    const A=data.find(d=>d.w===d3.select("#emb-a").property("value"));
    const B=data.find(d=>d.w===d3.select("#emb-b").property("value"));
    const Cc=data.find(d=>d.w===d3.select("#emb-c").property("value"));
    const vx=A.x-B.x+Cc.x, vy=A.y-B.y+Cc.y;
    const res=data.filter(d=>![A.w,B.w,Cc.w].includes(d.w)).map(d=>({d,dist:Math.hypot(d.x-vx,d.y-vy)})).sort((a,b)=>a.dist-b.dist)[0];
    const arrow=(p,q,col)=>gLines.append("line").attr("x1",x(p.x)).attr("y1",y(p.y)).attr("x2",x(q.x)).attr("y2",y(q.y)).attr("stroke",col).attr("stroke-width",2).attr("marker-end","url(#arr)");
    if(!svg.select("#arr").node()){ const defs=svg.append("defs"); defs.append("marker").attr("id","arr").attr("viewBox","0 0 10 10").attr("refX",8).attr("refY",5).attr("markerWidth",6).attr("markerHeight",6).attr("orient","auto-start-reverse").append("path").attr("d","M0,0L10,5L0,10").attr("fill",PC.a2); }
    arrow(B,A,PC.a2); arrow(Cc,res.d,PC.good);
    gLines.append("circle").attr("cx",x(vx)).attr("cy",y(vy)).attr("r",10).attr("fill","none").attr("stroke",PC.good).attr("stroke-dasharray","4 3").attr("stroke-width",2);
    d3.select("#emb-readout").html(`<b>${A.w}</b> − <b>${B.w}</b> + <b>${Cc.w}</b> ≈ <b style="color:${PC.good}">${res.d.w}</b>`);
  }
  d3.select("#emb-run").on("click",solve);
  d3.select("#emb-clear").on("click",()=>{ gLines.selectAll("*").remove(); d3.select("#emb-readout").text(""); });
  draw();
})();

/* ───────────────────────── 03 · skip-gram context window ───────────────────────── */
(function(){
  const words=["the","quick","brown","fox","jumps","over","the","lazy","dog"];
  const svg=d3.select("#sk-svg"),W=640,step=(W-40)/words.length, y0=64;
  const g=svg.append("g");
  const centerEl=document.getElementById("sk-center"), winEl=document.getElementById("sk-win"),
        winVal=document.getElementById("sk-winval"), out=document.getElementById("sk-readout");
  function render(){
    const c=+centerEl.value, win=+winEl.value; winVal.textContent=win;
    g.selectAll("*").remove();
    words.forEach((w,i)=>{
      const cx=20+step*i+step/2, inWin=i!==c && Math.abs(i-c)<=win, isC=i===c;
      const col=isC?PC.accent:inWin?PC.a2:"#222733", txt=isC?"#1a1320":inWin?"#2a1c06":PC.muted;
      g.append("rect").attr("x",cx-step/2+3).attr("y",y0-18).attr("width",step-6).attr("height",30).attr("rx",7)
        .attr("fill",col).attr("stroke",isC?"#fff":inWin?PC.a2:PC.line).attr("stroke-width",isC?1.6:1);
      g.append("text").attr("x",cx).attr("y",y0+2).attr("text-anchor","middle").attr("font-size",13)
        .attr("font-weight",isC?700:inWin?600:400).attr("fill",txt).text(w);
    });
    // bracket under the window
    const lo=Math.max(0,c-win), hi=Math.min(words.length-1,c+win);
    const x1=20+step*lo+4, x2=20+step*hi+step-4;
    g.append("path").attr("d",`M${x1},${y0+24} L${x1},${y0+30} L${x2},${y0+30} L${x2},${y0+24}`)
      .attr("fill","none").attr("stroke",PC.a2).attr("stroke-width",1.4).attr("stroke-opacity",.7);
    g.append("text").attr("x",(x1+x2)/2).attr("y",y0+44).attr("text-anchor","middle").attr("font-size",11).attr("fill",PC.muted).text("context window");
    const pairs=[];
    for(let i=lo;i<=hi;i++) if(i!==c) pairs.push(`(<b>${words[c]}</b> → ${words[i]})`);
    out.innerHTML="Training pairs: "+pairs.join(" &nbsp; ");
  }
  centerEl.addEventListener("input",render); winEl.addEventListener("input",render); render();
})();

/* ───────────────────────── 05 · cosine playground ───────────────────────── */
(function(){
  const svg=d3.select("#cos-svg"),W=640,H=360,ox=W/2,oy=H/2;
  const defs=svg.append("defs");
  [["av",PC.accent],["bv",PC.a2]].forEach(([id,c])=>defs.append("marker").attr("id",id).attr("viewBox","0 0 10 10")
    .attr("refX",8).attr("refY",5).attr("markerWidth",7).attr("markerHeight",7).attr("orient","auto-start-reverse")
    .append("path").attr("d","M0,0L10,5L0,10").attr("fill",c));
  // grid + axes
  const gg=svg.append("g");
  for(let gx=-3;gx<=3;gx++){ gg.append("line").attr("x1",ox+gx*55).attr("y1",20).attr("x2",ox+gx*55).attr("y2",H-20).attr("stroke","#1b2130"); }
  for(let gy=-2;gy<=2;gy++){ gg.append("line").attr("x1",30).attr("y1",oy+gy*55).attr("x2",W-30).attr("y2",oy+gy*55).attr("stroke","#1b2130"); }
  gg.append("line").attr("x1",30).attr("y1",oy).attr("x2",W-30).attr("y2",oy).attr("stroke","#3a4150");
  gg.append("line").attr("x1",ox).attr("y1",20).attr("x2",ox).attr("y2",H-20).attr("stroke","#3a4150");
  const arc=svg.append("path").attr("fill",PC.good).attr("fill-opacity",.14).attr("stroke",PC.good).attr("stroke-opacity",.5);
  const la=svg.append("line").attr("stroke",PC.accent).attr("stroke-width",2.5).attr("marker-end","url(#av)");
  const lb=svg.append("line").attr("stroke",PC.a2).attr("stroke-width",2.5).attr("marker-end","url(#bv)");
  const out=document.getElementById("cos-readout");
  let a={x:2.1,y:-1.1}, b={x:1.4,y:1.6};   // in grid units (1 unit = 55px, y up)
  const px=v=>ox+v.x*55, py=v=>oy-v.y*55;
  function handle(v,col,onmove){
    const h=svg.append("circle").attr("r",8).attr("fill",col).attr("stroke","#0f1117").attr("stroke-width",2).style("cursor","grab");
    h.call(d3.drag().on("drag",e=>{ v.x=(e.x-ox)/55; v.y=(oy-e.y)/55; onmove(); }));
    return h;
  }
  function update(){
    la.attr("x1",ox).attr("y1",oy).attr("x2",px(a)).attr("y2",py(a));
    lb.attr("x1",ox).attr("y1",oy).attr("x2",px(b)).attr("y2",py(b));
    ha.attr("cx",px(a)).attr("cy",py(a)); hb.attr("cx",px(b)).attr("cy",py(b));
    const dot=a.x*b.x+a.y*b.y, na=Math.hypot(a.x,a.y), nb=Math.hypot(b.x,b.y);
    const cos=dot/((na*nb)||1e-9), ang=Math.acos(Math.max(-1,Math.min(1,cos)))*180/Math.PI;
    const eu=Math.hypot(a.x-b.x,a.y-b.y);
    // angle arc between the two vectors
    const r=34, a0=Math.atan2(-a.y,a.x), a1=Math.atan2(-b.y,b.x);
    arc.attr("d",`M${ox},${oy} L${ox+r*Math.cos(a0)},${oy+r*Math.sin(a0)} A${r},${r} 0 0 ${((a1-a0+2*Math.PI)%(2*Math.PI))>Math.PI?1:0} ${ox+r*Math.cos(a1)},${oy+r*Math.sin(a1)} Z`);
    const cc=cos>0.66?PC.good:cos<0?PC.bad:PC.a2;
    out.innerHTML=`angle <b>${ang.toFixed(0)}°</b> &nbsp; cosine <b style="color:${cc}">${cos.toFixed(3)}</b> &nbsp; dot <b>${dot.toFixed(2)}</b> &nbsp; euclidean <b>${eu.toFixed(2)}</b>`;
  }
  const ha=handle(a,PC.accent,update), hb=handle(b,PC.a2,update);
  update();
})();

/* ───────────────────────── 06 · polysemy ("bank") ───────────────────────── */
(function(){
  const svg=d3.select("#poly-svg"),W=640,H=300;
  const river=[{w:"water",x:120,y:90},{w:"stream",x:90,y:160},{w:"fish",x:170,y:200},{w:"flow",x:200,y:110}];
  const money=[{w:"loan",x:470,y:90},{w:"deposit",x:520,y:160},{w:"account",x:440,y:200},{w:"cash",x:560,y:110}];
  svg.append("text").attr("x",140).attr("y",40).attr("fill","#5b9cff").attr("font-size",12).attr("font-weight",700).attr("text-anchor","middle").text("RIVER sense");
  svg.append("text").attr("x",500).attr("y",40).attr("fill",PC.good).attr("font-size",12).attr("font-weight",700).attr("text-anchor","middle").text("FINANCE sense");
  function cluster(arr,col){ arr.forEach(d=>{ const g=svg.append("g");
    g.append("circle").attr("cx",d.x).attr("cy",d.y).attr("r",5).attr("fill",col).attr("stroke","#0f1117").attr("stroke-width",1.4);
    g.append("text").attr("x",d.x+8).attr("y",d.y+4).attr("font-size",11).attr("fill",PC.muted).text(d.w); }); }
  cluster(river,"#5b9cff"); cluster(money,PC.good);
  const mid={x:W/2,y:150};
  const bank=svg.append("g");
  const bc=bank.append("circle").attr("r",9).attr("fill",PC.accent).attr("stroke","#fff").attr("stroke-width",1.8);
  const bt=bank.append("text").attr("font-size",12).attr("font-weight",700).attr("fill",PC.ink).attr("paint-order","stroke").attr("stroke","#0f1117").attr("stroke-width",3).attr("stroke-linejoin","round").text("bank");
  const cap=svg.append("text").attr("x",W/2).attr("y",H-14).attr("text-anchor","middle").attr("font-size",12).attr("fill",PC.muted);
  function move(to,label){ const t={river:{x:150,y:150},money:{x:490,y:150},static:mid}[to];
    bc.transition().duration(600).attr("cx",t.x).attr("cy",t.y);
    bt.transition().duration(600).attr("x",t.x+12).attr("y",t.y+4);
    cap.text(label); }
  document.getElementById("poly-river").onclick=()=>move("river","Contextual: near water words → the river vector");
  document.getElementById("poly-money").onclick=()=>move("money","Contextual: near finance words → the money vector");
  document.getElementById("poly-static").onclick=()=>move("static","Static: one fixed vector, stuck between both senses");
  move("static","Static: one fixed vector, stuck between both senses");
})();

/* ───────────────────────── 08 · matryoshka truncation ───────────────────────── */
(function(){
  const svg=d3.select("#mat-svg"),W=640,FULL=256,N=32,cellW=(W-40)/N;
  const g=svg.append("g");
  const dimsEl=document.getElementById("mat-dims"), out=document.getElementById("mat-readout");
  // a stand-in "importance" per cell — earlier dims carry more signal (Matryoshka ordering)
  function render(){
    const keep=+dimsEl.value, kept=Math.round(keep/FULL*N);
    g.selectAll("*").remove();
    for(let i=0;i<N;i++){
      const on=i<kept, h=18+58*Math.exp(-i/9);
      g.append("rect").attr("x",20+i*cellW+1).attr("y",96-h).attr("width",cellW-2).attr("height",h).attr("rx",2)
        .attr("fill",on?PC.accent:"#222733").attr("fill-opacity",on?0.55+0.45*Math.exp(-i/9):0.5)
        .attr("stroke",on?PC.accent:PC.line).attr("stroke-opacity",on?0.6:0.4);
    }
    g.append("text").attr("x",20).attr("y",118).attr("font-size",10).attr("fill",PC.muted).text("dim 1");
    g.append("text").attr("x",W-20).attr("y",118).attr("text-anchor","end").attr("font-size",10).attr("fill",PC.muted).text("dim "+FULL);
    const quality=100*(1-Math.exp(-keep/55));          // diminishing returns
    const bytes=keep*4, fullBytes=FULL*4, save=100*(1-bytes/fullBytes);
    out.innerHTML=`keep <b>${keep}</b>/${FULL} dims &nbsp;·&nbsp; retrieval quality ≈ <b style="color:${PC.good}">${quality.toFixed(1)}%</b> &nbsp;·&nbsp; storage <b>${bytes} B</b>/vec (<b style="color:${PC.a2}">−${save.toFixed(0)}%</b>)`;
  }
  dimsEl.addEventListener("input",render); render();
})();
