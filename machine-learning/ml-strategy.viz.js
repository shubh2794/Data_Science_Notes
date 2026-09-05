/* ml-strategy.viz.js — extracted from ml-strategy.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ---- Viz 1: error decomposition diagnoser ---- */
(function(){
  const C={accent:"#ffb454",blue:"#5b9cff",good:"#4ade80",bad:"#f87171",violet:"#c084fc",muted:"#9aa3b2",ink:"#e6e9ef"};
  const svg=d3.select("#d-svg"),W=640,H=200;
  const px=20,py=58,pw=W-40,barH=46;
  const ids=["d-opt","d-tr","d-td","d-dv"];
  function val(id){return +d3.select("#"+id).property("value");}
  function draw(){
    svg.selectAll("*").remove();
    let opt=val("d-opt"), tr=val("d-tr"), td=val("d-td"), dv=val("d-dv");
    // keep the story monotonic for the visual, but report raw too
    const unavoid=opt;
    const avoidBias=Math.max(0,tr-opt);
    const variance=Math.max(0,td-tr);
    const mismatch=Math.max(0,dv-td);
    const total=unavoid+avoidBias+variance+mismatch;
    const xmax=Math.max(total,dv,20);
    const X=v=>px+v/xmax*pw;
    const segs=[
      {label:"unavoidable (Bayes)",v:unavoid,c:C.muted},
      {label:"avoidable bias",v:avoidBias,c:C.bad},
      {label:"variance",v:variance,c:C.blue},
      {label:"data mismatch",v:mismatch,c:C.violet},
    ];
    // axis
    svg.append("text").attr("x",px).attr("y",24).attr("fill",C.ink).attr("font-size",12).attr("font-weight",600).text("dev error ≈ "+total.toFixed(1)+"%  (stacked components)");
    let x=px;
    segs.forEach(s=>{
      if(s.v<=0){return;}
      svg.append("rect").attr("x",X(x)).attr("y",py).attr("width",X(x+s.v)-X(x)).attr("height",barH).attr("fill",s.c).attr("fill-opacity",0.85).attr("stroke","#0f1117");
      if(X(x+s.v)-X(x)>40)
        svg.append("text").attr("x",(X(x)+X(x+s.v))/2).attr("y",py+barH/2+4).attr("text-anchor","middle").attr("fill","#0f1117").attr("font-size",11).attr("font-weight",700).text(s.v.toFixed(1));
      x+=s.v;
    });
    svg.append("line").attr("x1",X(0)).attr("y1",py-6).attr("x2",X(0)).attr("y2",py+barH+6).attr("stroke","#3a4150");
    // legend
    let lx=px;
    segs.forEach(s=>{
      const g=svg.append("g");
      g.append("rect").attr("x",lx).attr("y",py+barH+18).attr("width",11).attr("height",11).attr("fill",s.c).attr("fill-opacity",0.85);
      g.append("text").attr("x",lx+16).attr("y",py+barH+28).attr("fill",C.muted).attr("font-size",10).text(s.label);
      lx+=s.label.length*5.6+34;
    });
    // recommendation
    const cand=[{k:"avoidable bias",v:avoidBias,msg:"reduce bias → bigger model, train longer, better architecture, less regularization"},
                {k:"variance",v:variance,msg:"reduce variance → more training data, regularization, early stopping"},
                {k:"data mismatch",v:mismatch,msg:"close the gap → error-analyze train↔dev difference, synthesize/collect matching data"}];
    cand.sort((a,b)=>b.v-a.v);
    let rec, col;
    if(cand[0].v<=1){rec="Already near the optimal error rate — little room left to improve."; col=C.good;}
    else { rec=`Biggest lever: <b>${cand[0].k}</b> (${cand[0].v.toFixed(1)}%) — ${cand[0].msg}.`; col=cand[0].k==="avoidable bias"?C.bad:cand[0].k==="variance"?C.blue:C.violet; }
    let warn = (tr<opt||td<tr||dv<td) ? ' <span style="color:'+C.muted+'">(note: expected opt ≤ train ≤ train-dev ≤ dev)</span>' : "";
    d3.select("#d-read").html(`<span style="color:${col}">▶</span> ${rec}${warn}`);
  }
  ids.forEach(id=>d3.select("#"+id).on("input",draw));
  draw();
})();

/* ---- Viz 2: learning-curve interpreter ---- */
(function(){
  const C={accent:"#ffb454",blue:"#5b9cff",good:"#4ade80",bad:"#f87171",muted:"#9aa3b2",ink:"#e6e9ef"};
  const svg=d3.select("#lc-svg"),W=640,H=280;
  const px=48,py=24,pw=W-96,ph=200;
  const mmax=100, desired=10;
  const X=m=>px+m/mmax*pw;
  const ymax=34, Y=v=>py+ph-Math.min(v,ymax)/ymax*ph;
  // scenario → {biasFloor (training asymptote), residualGap (dev-train at large m)}
  const SC={
    bias:{b:18,g:2, txt:"Training error has plateaued <b>above</b> the desired line and the gap to dev is small → <b>high bias</b>. More data won't reach the goal; grow the model / improve architecture."},
    var:{b:4,g:15, txt:"Training error sits <b>below</b> desired but dev is far above and still falling → <b>high variance</b>. More data (and regularization) should close the gap."},
    both:{b:16,g:12,txt:"Training error is above desired <b>and</b> the dev–train gap is large → <b>high bias + high variance</b>. You'll need to reduce both."},
    good:{b:6,g:2, txt:"Both curves settle near/below the desired line with a small gap → <b>well-fit</b>. Diminishing returns from more data."}
  };
  function curves(s){
    const te=m=> s.b*(1-Math.exp(-m/22));            // training error rises 0 → b
    const de=m=> (s.b+s.g) + (30-(s.b+s.g))*Math.exp(-m/30); // dev error falls → b+g
    return {te,de};
  }
  function draw(){
    svg.selectAll("*").remove();
    const key=d3.select('input[name="lc"]:checked').property("value"), s=SC[key];
    const {te,de}=curves(s);
    // axes
    svg.append("line").attr("x1",px).attr("y1",py+ph).attr("x2",px+pw).attr("y2",py+ph).attr("stroke","#3a4150");
    svg.append("line").attr("x1",px).attr("y1",py).attr("x2",px).attr("y2",py+ph).attr("stroke","#3a4150");
    svg.append("text").attr("x",px+pw/2).attr("y",py+ph+24).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("training set size →");
    svg.append("text").attr("x",px-6).attr("y",py-9).attr("fill",C.muted).attr("font-size",10).text("error %");
    // desired line
    svg.append("line").attr("x1",px).attr("y1",Y(desired)).attr("x2",px+pw).attr("y2",Y(desired)).attr("stroke",C.good).attr("stroke-dasharray","5 4").attr("stroke-width",1.4);
    svg.append("text").attr("x",px+pw).attr("y",Y(desired)-5).attr("text-anchor","end").attr("fill",C.good).attr("font-size",10).text("desired (≈ optimal) error");
    const ms=d3.range(2,mmax+0.1,2);
    const line=(f,col)=>svg.append("path").attr("d",d3.line().x(X).y(d=>Y(f(d)))(ms)).attr("fill","none").attr("stroke",col).attr("stroke-width",2.4);
    line(de,C.bad);   svg.append("text").attr("x",X(mmax)).attr("y",Y(de(mmax))-6).attr("text-anchor","end").attr("fill",C.bad).attr("font-size",10).text("dev error");
    line(te,C.blue);  svg.append("text").attr("x",X(mmax)).attr("y",Y(te(mmax))+14).attr("text-anchor","end").attr("fill",C.blue).attr("font-size",10).text("training error");
    d3.select("#lc-read").html(`<span style="color:${C.accent}">▶</span> ${s.txt}`);
  }
  d3.selectAll('input[name="lc"]').on("change",draw);
  draw();
})();
