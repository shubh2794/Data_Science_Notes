/* donut.viz.js — extracted from donut.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ---- Viz 0: encoder–decoder architecture diagram ---- */
(function(){
  const C={enc:"#5b9cff",dec:"#e879f9",cross:"#4ade80",muted:"#9aa3b2",ink:"#e6e9ef",off:"#39404e"};
  const svg=d3.select("#arch-svg"),W=640,H=300;
  function box(x,y,w,h,t,sub,col,fill){
    const g=svg.append("g");
    g.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",h).attr("rx",7).attr("fill",fill||"#11141b").attr("stroke",col).attr("stroke-width",1.4);
    g.append("text").attr("x",x+w/2).attr("y",y+(sub?h/2-2:h/2+4)).attr("text-anchor","middle").attr("font-size",11).attr("font-weight",600).attr("fill",C.ink).text(t);
    if(sub) g.append("text").attr("x",x+w/2).attr("y",y+h/2+13).attr("text-anchor","middle").attr("font-size",8.5).attr("fill",col).text(sub);
  }
  function arrow(x1,y1,x2,y2,col,dash){
    svg.append("line").attr("x1",x1).attr("y1",y1).attr("x2",x2).attr("y2",y2).attr("stroke",col||C.off).attr("stroke-width",1.4).attr("stroke-dasharray",dash||null);
    const a=Math.atan2(y2-y1,x2-x1);
    svg.append("path").attr("d",`M${x2-8*Math.cos(a-0.4)},${y2-8*Math.sin(a-0.4)} L${x2},${y2} L${x2-8*Math.cos(a+0.4)},${y2-8*Math.sin(a+0.4)}`).attr("fill",col||C.off);
  }
  function draw(){
    svg.selectAll("*").remove();
    // ---- ENCODER (left) ----
    svg.append("text").attr("x",30).attr("y",24).attr("fill",C.enc).attr("font-size",11).attr("font-weight",700).text("Swin Transformer encoder");
    // document image with patch grid
    box(30,46,96,150,"","",C.off,"#0e1016");
    svg.append("text").attr("x",78).attr("y",40).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9).text("doc image");
    for(let r=0;r<5;r++)for(let c=0;c<3;c++){ svg.append("rect").attr("x",36+c*30).attr("y",52+r*28).attr("width",28).attr("height",26).attr("fill","none").attr("stroke","#232833"); }
    arrow(126,121,150,121,C.enc);
    box(150,70,84,102,"Swin","windowed attn · hierarchical stages",C.enc);
    // wrap the long sub manually
    svg.select("g:last-child text:last-child").remove();
    svg.append("text").attr("x",192).attr("y",128).attr("text-anchor","middle").attr("font-size",8).attr("fill",C.enc).text("windowed attn");
    svg.append("text").attr("x",192).attr("y",139).attr("text-anchor","middle").attr("font-size",8).attr("fill",C.enc).text("hierarchical");
    arrow(234,121,262,121,C.enc);
    // image memory (encoder output)
    box(262,70,70,102,"image","tokens",C.enc,"#0e1016");

    // ---- DECODER (right) ----
    svg.append("text").attr("x",380).attr("y",24).attr("fill",C.dec).attr("font-size",11).attr("font-weight",700).text("BART-style decoder");
    box(380,70,150,60,"masked self-attention",null,C.dec);
    box(380,148,150,60,"cross-attention",null,C.cross);
    box(380,226,150,46,"feed-forward → token",null,C.dec);
    arrow(455,130,455,148,C.dec);
    arrow(455,208,455,226,C.dec);
    // cross-attention pulls from image tokens
    arrow(332,150,378,170,C.cross,"5 3");
    svg.append("text").attr("x",352).attr("y",200).attr("text-anchor","middle").attr("fill",C.cross).attr("font-size",8.5).text("attends to");
    svg.append("text").attr("x",352).attr("y",211).attr("text-anchor","middle").attr("fill",C.cross).attr("font-size",8.5).text("the image");
    // autoregressive output loop
    arrow(530,249,560,249,C.dec);
    svg.append("text").attr("x",565).attr("y",240).attr("fill",C.ink).attr("font-size",10).attr("font-family","SF Mono,monospace").text("⟨s⟩");
    svg.append("text").attr("x",565).attr("y",253).attr("fill",C.ink).attr("font-size",9).attr("font-family","SF Mono,monospace").text("JSON…");
    arrow(560,266,455,266,C.off,"4 3"); svg.append("line").attr("x1",455).attr("y1",266).attr("x2",455).attr("y2",272).attr("stroke",C.off);
    svg.append("text").attr("x",500).attr("y",286).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",8.5).text("previous tokens fed back (autoregressive)");

    d3.select("#arch-read").html(`<span style="color:${C.cross}">▶</span> The <b style="color:${C.enc}">Swin encoder</b> turns page pixels into image tokens; the <b style="color:${C.dec}">decoder</b> generates output tokens, <b style="color:${C.cross}">cross-attending</b> to those image tokens — no OCR text ever exists in between.`);
  }
  draw();
})();

/* ---- Viz 1: OCR pipeline vs Donut flow ---- */
(function(){
  const C={accent:"#5b9cff",good:"#4ade80",bad:"#f87171",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef"};
  const svg=d3.select("#fl-svg"),W=640,H=250;
  function box(x,y,w,h,label,col,sub){
    const g=svg.append("g");
    g.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",h).attr("rx",7).attr("fill","#11141b").attr("stroke",col).attr("stroke-width",1.4);
    g.append("text").attr("x",x+w/2).attr("y",y+(sub?h/2-3:h/2+4)).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",11).attr("font-weight",600).text(label);
    if(sub) g.append("text").attr("x",x+w/2).attr("y",y+h/2+13).attr("text-anchor","middle").attr("fill",col).attr("font-size",10).attr("font-family","SF Mono,monospace").text(sub);
    return {x,y,w,h};
  }
  function arrow(x1,y,x2,col){ svg.append("line").attr("x1",x1).attr("y1",y).attr("x2",x2-7).attr("y2",y).attr("stroke",col||"#39404e").attr("stroke-width",1.4).attr("marker-end","url(#fa)");
    svg.append("path").attr("d",`M${x2-7},${y-4} L${x2},${y} L${x2-7},${y+4}`).attr("fill",col||"#39404e"); }
  function draw(){
    svg.selectAll("*").remove();
    const err=d3.select("#fl-err").property("checked");
    const ocrText=err?'"Total: O.00"':'"Total: 42.00"';
    const ocrCol=err?C.bad:C.good;
    // top row: OCR pipeline
    svg.append("text").attr("x",16).attr("y",24).attr("fill",C.muted).attr("font-size",11).text("OCR pipeline");
    box(16,34,86,46,"image","#39404e");
    arrow(102,57,130);
    box(130,34,110,46,"OCR engine",C.a2);
    arrow(240,57,268);
    box(268,34,150,46,"text",ocrCol,ocrText);
    arrow(418,57,446);
    box(446,34,110,46,"parser",C.accent);
    arrow(556,57,584);
    const out1=err?'{ total: "O.00" }':'{ total: "42.00" }';
    svg.append("text").attr("x",560).attr("y",100).attr("text-anchor","end").attr("fill",err?C.bad:C.good).attr("font-size",10).attr("font-family","SF Mono,monospace").text(out1);
    // bottom row: Donut
    svg.append("text").attr("x",16).attr("y",150).attr("fill",C.muted).attr("font-size",11).text("Donut (OCR-free)");
    box(16,160,86,46,"image","#39404e");
    arrow(102,183,150);
    box(150,160,200,46,"Swin enc → BART dec",C.accent);
    arrow(350,183,398);
    box(398,160,158,46,"JSON","#39404e");
    svg.append("text").attr("x",477).attr("y",228).attr("text-anchor","middle").attr("fill",C.good).attr("font-size",10).attr("font-family","SF Mono,monospace").text('{ total: "42.00" }');
    d3.select("#fl-read").html(err
      ? `<span style="color:${C.bad}">▶</span> OCR misread "42"→"O" and the error <b>propagates</b> to the output. Donut reads the pixels directly, so it isn't bottlenecked by an OCR stage.`
      : `<span style="color:${C.good}">▶</span> Both work when OCR is clean — but the pipeline adds a heavy OCR stage and a failure point. Flip the switch.`);
  }
  d3.select("#fl-err").on("change",draw);
  draw();
})();

/* ---- Viz 2: autoregressive JSON generation ---- */
(function(){
  const C={accent:"#e879f9",good:"#4ade80",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef"};
  const svg=d3.select("#gn-svg"),W=640,H=230;
  const toks=["<s_cord>","<s_menu>","Latte","<sep/>","$4.50","</s_menu>","<s_total>","$4.50","</s_total>","</s>"];
  const special=t=>t.startsWith("<");
  let n=0;
  function draw(){
    svg.selectAll("*").remove();
    // image stub
    svg.append("rect").attr("x",20).attr("y",30).attr("width",120).attr("height",150).attr("rx",8).attr("fill","#11141b").attr("stroke","#2a2f3a");
    svg.append("text").attr("x",80).attr("y",55).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("receipt");
    ["Latte    $4.50","——————","TOTAL   $4.50"].forEach((l,i)=>svg.append("text").attr("x",32).attr("y",90+i*22).attr("fill",C.ink).attr("font-size",10).attr("font-family","SF Mono,monospace").text(l));
    svg.append("text").attr("x",170).attr("y",44).attr("fill",C.muted).attr("font-size",10).text("decoder output (generated left → right):");
    // tokens
    let x=170,y=64; const maxx=W-20;
    toks.forEach((t,i)=>{
      const shown=i<n, w=t.length*7.2+14;
      if(x+w>maxx){x=170;y+=34;}
      svg.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",24).attr("rx",5)
        .attr("fill",shown?(special(t)?"rgba(232,121,249,.18)":"#15181f"):"#0e1016")
        .attr("stroke",shown?(special(t)?C.accent:"#39404e"):"#1c2026").attr("stroke-width",1)
        .attr("opacity",shown?1:0.35);
      svg.append("text").attr("x",x+w/2).attr("y",y+16).attr("text-anchor","middle").attr("font-size",10).attr("font-family","SF Mono,monospace")
        .attr("fill",shown?(special(t)?C.accent:C.ink):"#3a4150").text(t);
      if(i===n-1) svg.append("circle").attr("cx",x+w+6).attr("cy",y+12).attr("r",3).attr("fill",C.good);
      x+=w+6;
    });
    // parsed JSON when done
    if(n>=toks.length){
      svg.append("text").attr("x",170).attr("y",y+58).attr("fill",C.good).attr("font-size",11).attr("font-family","SF Mono,monospace").text('parsed → { menu:[{name:"Latte", price:"$4.50"}], total:"$4.50" }');
    }
    d3.select("#gn-read").html(n>=toks.length
      ? `<span style="color:${C.good}">▶</span> Done — the tagged sequence parses straight into a JSON record. Special <b style="color:${C.accent}">&lt;s_…&gt;</b> tokens mark fields.`
      : `<span style="color:${C.accent}">▶</span> Step the decoder — it emits structured tokens conditioned on the image (token ${n}/${toks.length}).`);
  }
  d3.select("#gn-step").on("click",()=>{ if(n<toks.length){n++;draw();} });
  d3.select("#gn-reset").on("click",()=>{ n=0;draw(); });
  draw();
})();
