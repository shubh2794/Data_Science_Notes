/* dynamic-programming.viz.js — extracted from dynamic-programming.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const svg=d3.select("#dp-svg");
  const A="SAT", B="CATS";              // transform A → B
  const m=A.length, n=B.length;
  const dp=Array.from({length:m+1},()=>Array(n+1).fill(null));
  const cell=42, ox=120, oy=70;

  // fill order: row by row, skipping borders which we preset
  const order=[];
  for(let i=0;i<=m;i++) for(let j=0;j<=n;j++){
    if(i===0||j===0) dp[i][j]= (i===0? j : i);
    else order.push([i,j]);
  }
  let step=0, timer=null, cur=null, won=null;

  const g=svg.append("g");
  // column header (B) and row header (A)
  for(let j=0;j<n;j++) g.append("text").attr("x",ox+(j+1)*cell+cell/2).attr("y",oy-cell/2+6)
    .attr("text-anchor","middle").attr("fill",C.B).attr("font-size",16).attr("font-weight",700).text(B[j]);
  for(let i=0;i<m;i++) g.append("text").attr("x",ox-cell/2).attr("y",oy+(i+1)*cell+cell/2+6)
    .attr("text-anchor","middle").attr("fill",C.B).attr("font-size",16).attr("font-weight",700).text(A[i]);
  g.append("text").attr("x",ox-cell/2-22).attr("y",oy-cell/2+6).attr("fill",C.muted).attr("font-size",11).text("A↓");
  g.append("text").attr("x",ox+cell/2).attr("y",oy-cell-6).attr("fill",C.muted).attr("font-size",11).text("B→");

  function rects(){
    const data=[];
    for(let i=0;i<=m;i++) for(let j=0;j<=n;j++) data.push({i,j});
    const sel=g.selectAll("g.cell").data(data).enter().append("g").attr("class","cell")
      .attr("transform",d=>`translate(${ox+d.j*cell},${oy+d.i*cell})`);
    sel.append("rect").attr("class","cbox").attr("width",cell-4).attr("height",cell-4).attr("rx",6)
      .attr("fill","#161b27").attr("stroke",C.line).attr("stroke-width",1.4);
    sel.append("text").attr("class","cval").attr("x",(cell-4)/2).attr("y",(cell-4)/2+5)
      .attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",15).attr("font-weight",600).text("");
  }
  rects();

  function paint(){
    g.selectAll("g.cell").each(function(d){
      const filled = (d.i===0||d.j===0) ? (step>=0) : (dp[d.i][d.j]!==null);
      const isBorder = d.i===0||d.j===0;
      const txt = dp[d.i][d.j];
      const isCur = cur && cur[0]===d.i && cur[1]===d.j;
      const isWon = won && won.some(w=>w[0]===d.i&&w[1]===d.j);
      d3.select(this).select("text.cval").text(txt===null?"":txt)
        .attr("fill", txt===null? C.muted : isBorder? C.muted : C.ink);
      d3.select(this).select("rect.cbox")
        .attr("fill", isCur? "#3a2e16" : isWon? "#15233a" : (txt!==null&&!isBorder)? "#15233a" : (isBorder&&txt!==null)? "#1a2030" : "#161b27")
        .attr("stroke", isCur? C.B : isWon? C.A : (txt!==null&&!isBorder)? C.A : C.line)
        .attr("stroke-width", (isCur||isWon)? 2.6 : (txt!==null&&!isBorder)?1.8:1.4);
    });
  }

  function doStep(){
    if(step>=order.length){ cur=null; won=null; paint();
      d3.select("#dp-readout").html(`done · edit distance(<b>${A}</b> → <b>${B}</b>) = <b>${dp[m][n]}</b>`); stop(); return; }
    const [i,j]=order[step];
    const match = A[i-1]===B[j-1];
    if(match){ dp[i][j]=dp[i-1][j-1]; cur=[i,j]; won=[[i-1,j-1]];
      d3.select("#dp-readout").html(`(${i},${j}) '${A[i-1]}'='${B[j-1]}' match → carry diagonal ${dp[i-1][j-1]} ⇒ <b>${dp[i][j]}</b>`); }
    else {
      const up=dp[i-1][j], left=dp[i][j-1], diag=dp[i-1][j-1];
      const best=Math.min(up,left,diag); dp[i][j]=1+best; cur=[i,j];
      won = best===diag?[[i-1,j-1]] : best===up?[[i-1,j]] : [[i,j-1]];
      const src = best===diag?"substitute (diag)" : best===up?"delete (up)" : "insert (left)";
      d3.select("#dp-readout").html(`(${i},${j}) '${A[i-1]}'≠'${B[j-1]}' → 1 + min(${up},${left},${diag}) = <b>${dp[i][j]}</b> · ${src}`); }
    step++; paint();
  }
  function play(){ if(timer) return; d3.select("#dp-play").text("Pause");
    timer=setInterval(()=>{ if(step>=order.length){stop();doStep();} else doStep(); }, 600); }
  function stop(){ if(timer){clearInterval(timer);timer=null;} d3.select("#dp-play").text("Play"); }
  function reset(){
    stop(); step=0; cur=null; won=null;
    for(let i=0;i<=m;i++) for(let j=0;j<=n;j++) dp[i][j]= (i===0? j : j===0? i : null);
    paint();
    d3.select("#dp-readout").html(`borders preset: dp[i][0]=i, dp[0][j]=j · transform "${A}" → "${B}"`);
  }
  d3.select("#dp-step").on("click",()=>{ stop(); doStep(); });
  d3.select("#dp-play").on("click",()=>{ timer?stop():play(); });
  d3.select("#dp-reset").on("click",reset);
  reset();
})();
