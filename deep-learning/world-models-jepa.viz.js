/* world-models-jepa.viz.js — extracted from world-models-jepa.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#f87171",a2:"#ffb454",good:"#4ade80",blue:"#5b9cff",violet:"#c084fc",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#wm-svg"),W=640,H=240;
  function box(x,y,w,h,label,fill,fs){ svg.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",h).attr("rx",8).attr("fill",fill).attr("stroke",C.line);
    svg.append("text").attr("x",x+w/2).attr("y",y+h/2+4).attr("text-anchor","middle").attr("font-size",fs||11).attr("fill",C.ink).text(label); }
  function arr(x1,y,x2){ svg.append("line").attr("x1",x1).attr("y1",y).attr("x2",x2).attr("y2",y).attr("stroke",C.muted).attr("stroke-width",1.5);
    svg.append("path").attr("d",`M${x2-6},${y-4}L${x2},${y}L${x2-6},${y+4}`).attr("fill","none").attr("stroke",C.muted); }
  function draw(){
    svg.selectAll("*").remove();
    const k=d3.select("#wm-sel").property("value");
    const y=80,h=70;
    box(30,y,90,h,"context","#1e222d");
    arr(120,y+h/2,160);
    box(160,y,90,h,"encoder","rgba(91,156,255,.15)");
    arr(250,y+h/2,290);
    box(290,y,90,h,"predictor","rgba(251,146,60,.15)");
    arr(380,y+h/2,420);
    if(k==="gen"){
      box(420,y,170,h,"decode → pixels","rgba(248,113,113,.15)");
      // pixel grid
      const gx=440,gy=y+12;
      for(let r=0;r<3;r++)for(let c=0;c<5;c++) svg.append("rect").attr("x",gx+c*16).attr("y",gy+r*16).attr("width",14).attr("height",14).attr("rx",2).attr("fill",d3.interpolateCividis((r+c)/7));
      svg.append("text").attr("x",505).attr("y",y+h+20).attr("text-anchor","middle").attr("font-size",9.5).attr("fill",C.muted).text("must reconstruct every pixel (incl. noise)");
      document.getElementById("wm-read").innerHTML="Generative: predict the full target pixels — high fidelity but wastes capacity on unpredictable detail.";
    } else {
      box(420,y,170,h,"target embedding","rgba(192,132,252,.15)");
      svg.append("text").attr("x",505).attr("y",y+h/2-2).attr("text-anchor","middle").attr("font-size",10).attr("fill",C.violet).text("≈ encode(target)");
      svg.append("text").attr("x",505).attr("y",y+h+20).attr("text-anchor","middle").attr("font-size",9.5).attr("fill",C.muted).text("match in representation space — ignore pixel noise");
      document.getElementById("wm-read").innerHTML="JEPA: predict the target's abstract embedding, not its pixels — efficient, robust representations.";
    }
  }
  d3.select("#wm-sel").on("change",draw);
  draw();
})();
