/* contrastive-learning.viz.js — extracted from contrastive-learning.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const svg=d3.select("#temp-svg"), W=640, H=340;
  // ---- scene geometry: a unit-circle embedding space on the left ----
  const cx=170, cy=170, R=120;
  // fixed angular layout of points on the unit circle (anchor at 0°).
  // anchor, positive, then several negatives at varying angles (=> varying similarity).
  const anchorAng = 0;
  // negatives placed at angles whose cosine vs anchor gives a spread of similarities
  const negAngs = [38, 62, 95, 130, 168, -150, -88, -50].map(d=>d*Math.PI/180);

  function pt(ang){ return [cx + R*Math.cos(ang), cy - R*Math.sin(ang)]; }

  function draw(){
    const tau=+d3.select("#t-tau").property("value");
    const posSim=+d3.select("#t-pos").property("value");   // user-set anchor·positive cosine
    svg.selectAll("*").remove();

    // similarities: positive is set directly by slider; negatives = cos(angle) vs anchor
    const negSims = negAngs.map(a=>Math.cos(a));            // anchor at 0° → sim = cos(angle)
    // logits = sim/τ over [positive, ...negatives]; softmax for the loss denominator term
    const allSims = [posSim, ...negSims];
    const logits = allSims.map(s=>s/tau);
    const mx = Math.max(...logits);
    const exps = logits.map(l=>Math.exp(l-mx));
    const Z = exps.reduce((a,b)=>a+b,0);
    const probs = exps.map(e=>e/Z);                         // softmax incl. positive
    const pPos = probs[0];
    const negProbs = probs.slice(1);
    const loss = -Math.log(pPos);                           // ℓ(i,j) = -log softmax(positive)
    // softmax weight among negatives only (what the "push-away" gradient is distributed over)
    const negZ = negProbs.reduce((a,b)=>a+b,0) || 1e-9;
    const negWeight = negProbs.map(p=>p/negZ);

    // ---- left panel: embedding circle ----
    svg.append("text").attr("x",cx-R).attr("y",26).attr("fill",C.muted).attr("font-size",11)
       .text("embedding space (unit circle)");
    svg.append("circle").attr("cx",cx).attr("cy",cy).attr("r",R)
       .attr("fill","none").attr("stroke",C.line).attr("stroke-dasharray","3 4");

    // anchor at the positive's location? no — anchor fixed; we draw the positive at an angle
    // whose cosine equals posSim, on the upper arc, so dragging it visibly moves the point.
    const posAng = Math.acos(Math.max(-1,Math.min(1,posSim)));   // 0..π
    const [ax,ay]=pt(anchorAng);
    const [px,py]=pt(posAng);

    // lines anchor→negatives (weight = softmax weight among negatives)
    negAngs.forEach((a,i)=>{
      const [nx,ny]=pt(a);
      svg.append("line").attr("x1",ax).attr("y1",ay).attr("x2",nx).attr("y2",ny)
         .attr("stroke",C.bad).attr("stroke-opacity",0.15+0.7*negWeight[i])
         .attr("stroke-width",0.8+5*negWeight[i]);
    });
    // line anchor→positive
    svg.append("line").attr("x1",ax).attr("y1",ay).attr("x2",px).attr("y2",py)
       .attr("stroke",C.good).attr("stroke-width",3).attr("stroke-opacity",0.9);

    // negative points: radius & color ramp by softmax weight (hardest = nearest = most weight)
    negAngs.forEach((a,i)=>{
      const [nx,ny]=pt(a);
      const w=negWeight[i];
      svg.append("circle").attr("cx",nx).attr("cy",ny)
         .attr("r",4+18*w)
         .attr("fill",d3.interpolateInferno(0.25+0.6*w)).attr("fill-opacity",0.85)
         .attr("stroke",C.bad).attr("stroke-width",1);
      svg.append("title").text(`negative · sim=${negSims[i].toFixed(2)} · weight=${(w*100).toFixed(1)}%`);
    });
    // positive point
    svg.append("circle").attr("cx",px).attr("cy",py).attr("r",8)
       .attr("fill",C.good).attr("stroke","#0b0e14").attr("stroke-width",1.5);
    svg.append("text").attr("x",px+11).attr("y",py+4).attr("fill",C.good).attr("font-size",11).text("positive");
    // anchor point
    svg.append("circle").attr("cx",ax).attr("cy",ay).attr("r",8)
       .attr("fill",C.A).attr("stroke","#0b0e14").attr("stroke-width",1.5);
    svg.append("text").attr("x",ax+11).attr("y",ay+4).attr("fill",C.A).attr("font-size",11).text("anchor");

    // ---- right panel: softmax-weight bars over negatives ----
    const bx=350, bw=W-bx-30, bTop=46, rowH=24;
    svg.append("text").attr("x",bx).attr("y",26).attr("fill",C.muted).attr("font-size",11)
       .text("softmax weight among negatives  →  where the push-away gradient goes");
    // order negatives by similarity (hardest first) for a readable bar chart
    const order=negSims.map((s,i)=>i).sort((a,b)=>negSims[b]-negSims[a]);
    order.forEach((idx,row)=>{
      const y=bTop+row*rowH;
      const w=negWeight[idx];
      svg.append("rect").attr("x",bx).attr("y",y).attr("width",bw).attr("height",rowH-7).attr("rx",3)
         .attr("fill",C.line).attr("opacity",0.5);
      svg.append("rect").attr("x",bx).attr("y",y).attr("width",Math.max(2,bw*w)).attr("height",rowH-7).attr("rx",3)
         .attr("fill",d3.interpolateInferno(0.25+0.6*w));
      svg.append("text").attr("x",bx-6).attr("y",y+rowH-12).attr("text-anchor","end")
         .attr("font-size",9).attr("fill",C.muted).text(`sim ${negSims[idx].toFixed(2)}`);
      svg.append("text").attr("x",bx+bw).attr("y",y+rowH-12).attr("text-anchor","end")
         .attr("font-size",9).attr("fill",C.ink).text(`${(w*100).toFixed(1)}%`);
    });

    d3.select("#temp-readout").html(
      `τ=<b>${tau.toFixed(2)}</b> · positive sim=<b>${posSim.toFixed(2)}</b> · `+
      `softmax P(positive)=<b>${(pPos*100).toFixed(1)}%</b> · `+
      `loss ℓ(i,j) = −log P(positive) = <b>${loss.toFixed(3)}</b>. `+
      `Low τ piles weight on the nearest negative (hardest); high τ spreads it evenly.`
    );
  }
  d3.select("#t-tau").on("input",draw);
  d3.select("#t-pos").on("input",draw);
  draw();
})();
