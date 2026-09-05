/* layoutlm.viz.js — extracted from layoutlm.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

/* ---- Viz: interactive architecture by version ---- */
(function(){
  const C={text:"#5b9cff",layout:"#ffb454",img:"#e879f9",muted:"#9aa3b2",ink:"#e6e9ef",good:"#4ade80",off:"#3a4150"};
  const svg=d3.select("#arch-svg"),W=640,H=330;
  const SPEC={
    v1:{img:{label:"image features",sub:"Faster R-CNN ROI · fine-tuning only",active:false},
        attn:"self-attention", obj:["MVLM","(+ MDC)"],
        note:"v1: text + 2D layout go through the encoder; <b>image features are added only at fine-tuning</b> (not pretrained)."},
    v2:{img:{label:"visual tokens",sub:"ResNeXt-FPN (CNN) grid",active:true},
        attn:"spatial-aware self-attention", obj:["MVLM","TIA","TIM"],
        note:"v2: a CNN turns the page into <b>visual tokens concatenated with text+layout</b>; attention gets a 2D relative-position bias."},
    v3:{img:{label:"image patches",sub:"linear ViT-style patches · no CNN",active:true},
        attn:"spatial-aware self-attention", obj:["MLM","MIM","WPA"],
        note:"v3: image enters as <b>linear patches</b> (no detector); text &amp; image are pretrained with symmetric masking + word-patch alignment."}};
  function box(x,y,w,h,title,sub,col,active){
    const g=svg.append("g");
    g.append("rect").attr("x",x).attr("y",y).attr("width",w).attr("height",h).attr("rx",7)
      .attr("fill",active?"#11141b":"#0e1016").attr("stroke",active?col:C.off).attr("stroke-width",1.5).attr("stroke-dasharray",active?null:"4 3");
    g.append("text").attr("x",x+w/2).attr("y",y+(sub?h/2-2:h/2+4)).attr("text-anchor","middle").attr("font-size",11).attr("font-weight",600).attr("fill",active?C.ink:C.muted).text(title);
    if(sub) g.append("text").attr("x",x+w/2).attr("y",y+h/2+13).attr("text-anchor","middle").attr("font-size",8.5).attr("fill",active?col:C.off).text(sub);
  }
  function up(x1,y1,x2,y2,col){ svg.append("line").attr("x1",x1).attr("y1",y1).attr("x2",x2).attr("y2",y2-7).attr("stroke",col||C.off).attr("stroke-width",1.3);
    svg.append("path").attr("d",`M${x2-4},${y2-7} L${x2},${y2} L${x2+4},${y2-7}`).attr("fill",col||C.off); }
  function draw(){
    svg.selectAll("*").remove();
    const k=d3.select('input[name="arch"]:checked').property("value"), s=SPEC[k];
    // input row (bottom)
    const iy=242, ih=58, tx=40, lx=246, mx=452, bw=148;
    box(tx,iy,bw,ih,"text tokens","WordPiece embeddings",C.text,true);
    box(lx,iy,bw,ih,"2D layout","(x₀,y₀,x₁,y₁) box embeddings",C.layout,true);
    box(mx,iy,bw,ih,s.img.label,s.img.sub,C.img,s.img.active);
    // encoder block (middle)
    const ey=150, eh=58, ex=40, ew=560;
    box(ex,ey,ew,eh,"Multimodal Transformer encoder",s.attn,"#6ee7ff",true);
    // arrows inputs → encoder
    up(tx+bw/2,iy,tx+bw/2,ey+eh,C.text);
    up(lx+bw/2,iy,lx+bw/2,ey+eh,C.layout);
    up(mx+bw/2,iy,mx+bw/2,ey+eh, s.img.active?C.img:C.off);
    if(!s.img.active){ svg.append("text").attr("x",mx+bw/2).attr("y",ey+eh+ (iy-(ey+eh))/2 +3).attr("text-anchor","middle").attr("font-size",8).attr("fill",C.off).text("(late)"); }
    // output row (top): objectives / heads
    const oy=58, oh=50; let ox=40;
    svg.append("text").attr("x",40).attr("y",44).attr("fill",C.muted).attr("font-size",10).text("pretraining objectives");
    s.obj.forEach(o=>{ const w=o.length*8+22; box(ox,oy,w,oh,o,null,C.good,true); up(ox+w/2,ey, ox+w/2, oy+oh, C.good); ox+=w+12; });
    d3.select("#arch-read").html(`<span style="color:${C.img}">▶</span> ${s.note}`);
  }
  d3.selectAll('input[name="arch"]').on("change",draw);
  draw();
})();

/* ---- Viz: OCR bounding boxes → 2D embeddings ---- */
(function(){
  const C={accent:"#e879f9",blue:"#5b9cff",good:"#4ade80",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef"};
  const svg=d3.select("#bbox-svg"),W=640,H=300;
  // mock invoice: page area maps a 0-1000 normalized grid → svg px
  const px=24,py=20,pw=380,ph=260;          // document panel
  const NX=v=>px+v/1000*pw, NY=v=>py+v/1000*ph;
  // words with normalized boxes [x0,y0,x1,y1] (0-1000)
  const words=[
    {t:"INVOICE", b:[60,40,360,120], role:"title"},
    {t:"Bill To:", b:[60,200,260,260], role:"label"},
    {t:"Acme Corp", b:[60,280,420,340], role:"value"},
    {t:"Invoice #", b:[600,200,820,260], role:"label"},
    {t:"INV-0042", b:[600,280,860,340], role:"value"},
    {t:"Date", b:[600,420,740,480], role:"label"},
    {t:"2026-06-12", b:[600,500,900,560], role:"value"},
    {t:"Total", b:[60,720,240,790], role:"label"},
    {t:"$42.00", b:[600,720,860,790], role:"value"},
  ];
  let sel=8; // default select "$42.00"
  const roleCol={title:C.a2,label:C.blue,value:C.good};
  function draw(){
    svg.selectAll("*").remove();
    // document panel
    svg.append("rect").attr("x",px).attr("y",py).attr("width",pw).attr("height",ph).attr("rx",8).attr("fill","#11141b").attr("stroke","#2a2f3a");
    words.forEach((w,i)=>{
      const x=NX(w.b[0]),y=NY(w.b[1]),ww=NX(w.b[2])-NX(w.b[0]),hh=NY(w.b[3])-NY(w.b[1]);
      const on=i===sel, col=roleCol[w.role];
      const g=svg.append("g").style("cursor","pointer").on("click",()=>{sel=i;draw();});
      g.append("rect").attr("x",x).attr("y",y).attr("width",ww).attr("height",hh).attr("rx",3)
        .attr("fill",on?col:"#181c24").attr("fill-opacity",on?0.22:1).attr("stroke",on?col:"#39404e").attr("stroke-width",on?2:1);
      g.append("text").attr("x",x+ww/2).attr("y",y+hh/2+4).attr("text-anchor","middle").attr("font-size",Math.min(13,hh*0.5))
        .attr("fill",on?col:C.ink).attr("font-family","SF Mono,monospace").text(w.t);
    });
    svg.append("text").attr("x",px).attr("y",py+ph+16).attr("fill",C.muted).attr("font-size",10).text("scanned page (OCR words + boxes)");

    // right panel: coordinates → embeddings
    const ox=px+pw+30, w=words[sel], b=w.b;
    svg.append("text").attr("x",ox).attr("y",py+8).attr("fill",C.ink).attr("font-size",13).attr("font-weight",700).text('"'+w.t+'"');
    svg.append("text").attr("x",ox).attr("y",py+26).attr("fill",roleCol[w.role]).attr("font-size",10).text("role: "+w.role+" (illustrative)");
    const rows=[["x₀",b[0]],["y₀",b[1]],["x₁",b[2]],["y₁",b[3]],["width",b[2]-b[0]],["height",b[3]-b[1]]];
    rows.forEach((r,i)=>{
      const yy=py+52+i*30;
      svg.append("text").attr("x",ox).attr("y",yy+12).attr("fill",C.muted).attr("font-size",11).attr("font-family","SF Mono,monospace").text(r[0]);
      svg.append("text").attr("x",ox+52).attr("y",yy+12).attr("fill",C.ink).attr("font-size",11).attr("font-family","SF Mono,monospace").text(r[1]);
      const bw=70, bx=ox+92;
      svg.append("rect").attr("x",bx).attr("y",yy+1).attr("width",bw).attr("height",12).attr("rx",3).attr("fill","#15181f");
      svg.append("rect").attr("x",bx).attr("y",yy+1).attr("width",Math.max(2,r[1]/1000*bw)).attr("height",12).attr("rx",3).attr("fill",C.accent).attr("fill-opacity",0.7);
      svg.append("text").attr("x",bx+bw+8).attr("y",yy+11).attr("fill",C.muted).attr("font-size",9).text("→ emb");
    });
    d3.select("#bbox-read").html(`<b style="color:${roleCol[w.role]}">"${w.t}"</b> → box (${b.join(", ")}) on a 0–1000 grid, each coordinate embedded and <b>added</b> to the token vector.`);
  }
  draw();
})();
