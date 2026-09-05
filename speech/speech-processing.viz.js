/* speech-processing.viz.js — extracted from speech-processing.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#38bdf8",good:"#4ade80",a2:"#ffb454",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#sp-svg"),W=640,H=220;
  function draw(){
    svg.selectAll("*").remove();
    const tts=d3.select("#sp-sel").property("value")==="tts";
    // waveform (left)
    const wx=30,wy=70,ww=150,wh=80;
    svg.append("rect").attr("x",wx).attr("y",wy).attr("width",ww).attr("height",wh).attr("rx",6).attr("fill","#0f131b").attr("stroke","#2a2f3a");
    let path="M"+wx+","+(wy+wh/2);
    for(let i=0;i<=ww;i+=3){ const a=Math.sin(i*0.25)*Math.sin(i*0.06)*0.4; path+=" L"+(wx+i)+","+(wy+wh/2 - a*wh); }
    svg.append("path").attr("d",path).attr("fill","none").attr("stroke",C.accent).attr("stroke-width",1.3);
    svg.append("text").attr("x",wx+ww/2).attr("y",wy+wh+16).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("waveform");
    // spectrogram (mid)
    const sx=230,sy=58,sw=170,sh=104, cols=22, rows=10;
    for(let c=0;c<cols;c++)for(let r=0;r<rows;r++){
      const v=Math.max(0,Math.sin(c*0.5+r*0.3)*0.5+0.5 - r*0.04 + (Math.random()*0.15));
      svg.append("rect").attr("x",sx+c*sw/cols).attr("y",sy+r*sh/rows).attr("width",sw/cols+0.5).attr("height",sh/rows+0.5).attr("fill",d3.interpolateMagma(0.1+v*0.85));
    }
    svg.append("rect").attr("x",sx).attr("y",sy).attr("width",sw).attr("height",sh).attr("fill","none").attr("stroke","#2a2f3a");
    svg.append("text").attr("x",sx+sw/2).attr("y",sy+sh+16).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("mel-spectrogram");
    // text (right)
    const tx=450,ty=88;
    svg.append("rect").attr("x",tx).attr("y",ty).attr("width",160).attr("height",44).attr("rx",6).attr("fill","#15181f").attr("stroke",C.line);
    svg.append("text").attr("x",tx+80).attr("y",ty+27).attr("text-anchor","middle").attr("fill",C.ink).attr("font-size",13).attr("font-family","SF Mono,monospace").text("“hello world”");
    svg.append("text").attr("x",tx+80).attr("y",ty+58).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("text");
    // arrows
    function arrow(x1,x2,y,col){ const dir=tts?-1:1; const a=tts?x2:x1, b=tts?x1:x2;
      svg.append("line").attr("x1",a).attr("y1",y).attr("x2",b).attr("y2",y).attr("stroke",col).attr("stroke-width",2);
      svg.append("path").attr("d",`M${b-dir*7},${y-4}L${b},${y}L${b-dir*7},${y+4}`).attr("fill","none").attr("stroke",col).attr("stroke-width",2); }
    arrow(wx+ww+4, sx-4, 110, C.muted);
    arrow(sx+sw+4, tx-4, 110, C.muted);
    svg.append("text").attr("x",W/2).attr("y",28).attr("text-anchor","middle").attr("fill",C.accent).attr("font-size",12).attr("font-weight",600).text(tts?"TTS: text → spectrogram → waveform":"ASR: waveform → spectrogram → text");
    d3.select("#sp-read").html(tts?`<b>TTS</b>: a model turns text into a spectrogram, then a neural vocoder renders the waveform`:`<b>ASR</b>: frames of the spectrogram are decoded (CTC / seq2seq) into text`);
  }
  d3.select("#sp-sel").on("change",draw);
  draw();
})();
