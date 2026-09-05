/* support-vector-machines.viz.js — extracted from support-vector-machines.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

function dist(a,b){ return Math.hypot(a.x-b.x, a.y-b.y); }

/* Interactive max-margin SVM */
(function(){
  const svg=d3.select("#svm-svg"), W=640, H=340;
  let data=[
    {x:160,y:110,c:1},{x:210,y:90,c:1},{x:140,y:170,c:1},{x:240,y:150,c:1},
    {x:420,y:230,c:-1},{x:470,y:200,c:-1},{x:500,y:270,c:-1},{x:400,y:280,c:-1}
  ];
  const sx=d3.scaleLinear().domain([0,W]).range([-3,3]);
  const sy=d3.scaleLinear().domain([0,H]).range([-3,3]);
  let w=[0.1,0.1], b=0;
  function train(){ w=[0.1,-0.1]; b=0; const lr=0.001, C_=1;
    for(let it=0;it<4000;it++){ data.forEach(p=>{ const X=[sx(p.x),sy(p.y)];
      const margin=p.c*(w[0]*X[0]+w[1]*X[1]+b);
      if(margin<1){ w[0]+= lr*(C_*p.c*X[0]-w[0]/data.length); w[1]+= lr*(C_*p.c*X[1]-w[1]/data.length); b+= lr*C_*p.c; }
      else { w[0]-= lr*w[0]/data.length; w[1]-= lr*w[1]/data.length; }
    }); }
  }
  const gLines=svg.append("g"), gPts=svg.append("g");
  function lineFor(off){ // w0*X0 + w1*X1 + b = off  -> solve for two x's
    const xa=sx(0), xb=sx(W);
    const Ya = w[1]!==0 ? (off-b-w[0]*xa)/w[1] : 0;
    const Yb = w[1]!==0 ? (off-b-w[0]*xb)/w[1] : 0;
    return `M0,${sy.invert(Ya)} L${W},${sy.invert(Yb)}`;
  }
  function draw(){
    train();
    gLines.selectAll("path").remove();
    gLines.append("path").attr("d",lineFor(0)).attr("stroke",C.ink).attr("stroke-width",2.5).attr("fill","none");
    [1,-1].forEach(o=>gLines.append("path").attr("d",lineFor(o)).attr("stroke",C.muted).attr("stroke-dasharray","6 5").attr("stroke-width",1.5).attr("fill","none"));
    // support vectors: margin within ~1.15
    const svm = data.map(p=>{const X=[sx(p.x),sy(p.y)]; return Math.abs(w[0]*X[0]+w[1]*X[1]+b);});
    const p=gPts.selectAll("circle").data(data);
    p.enter().append("circle").attr("r",8).attr("stroke-width",3).attr("cursor","grab")
      .merge(p).attr("cx",d=>d.x).attr("cy",d=>d.y).attr("fill",d=>d.c>0?C.A:C.B)
      .attr("stroke",(d,i)=> svm[i]<1.2 ? C.good : "#0f1117");
    p.exit().remove();
    gPts.selectAll("circle").call(d3.drag().on("drag",function(ev,d){ d.x=Math.max(8,Math.min(W-8,ev.x)); d.y=Math.max(8,Math.min(H-8,ev.y)); draw(); }));
    d3.select("#svm-readout").html(`margin width ≈ <b>${(2/Math.hypot(w[0],w[1])).toFixed(2)}</b> (units)`);
  }
  d3.select("#svm-reset").on("click",()=>{ data=[{x:160,y:110,c:1},{x:210,y:90,c:1},{x:140,y:170,c:1},{x:240,y:150,c:1},{x:420,y:230,c:-1},{x:470,y:200,c:-1},{x:500,y:270,c:-1},{x:400,y:280,c:-1}]; draw(); });
  draw();
})();
