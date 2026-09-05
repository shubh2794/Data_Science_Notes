/* parameter-efficient-fine-tuning.viz.js — extracted from parameter-efficient-fine-tuning.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#2dd4bf",a2:"#ffb454",good:"#4ade80",bad:"#f87171",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#peft-svg"),W=640,H=300,d=1024;
  function draw(){
    svg.selectAll("*").remove();
    const r=+d3.select("#peft-r").property("value");
    const full=d*d, lora=2*d*r, pct=lora/full*100;
    const cy=44, S=128;
    // frozen W
    const cx=46;
    svg.append("rect").attr("x",cx).attr("y",cy).attr("width",S).attr("height",S).attr("rx",4).attr("fill","#2a2f3a").attr("stroke","#3a4150");
    svg.append("text").attr("x",cx+S/2).attr("y",cy+S/2-2).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",11).text("W frozen");
    svg.append("text").attr("x",cx+S/2).attr("y",cy+S/2+14).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",9).text(d+"×"+d);
    svg.append("text").attr("x",cx+S+22).attr("y",cy+S/2+6).attr("font-size",22).attr("fill",C.ink).text("+");
    // B (d×r) tall narrow
    const bw=Math.max(5,Math.min(34,r/128*34)), bx=cx+S+48;
    svg.append("rect").attr("x",bx).attr("y",cy).attr("width",bw).attr("height",S).attr("rx",2).attr("fill",C.accent);
    svg.append("text").attr("x",bx+bw/2).attr("y",cy+S+13).attr("text-anchor","middle").attr("fill",C.accent).attr("font-size",9).text("B");
    svg.append("text").attr("x",bx+bw+9).attr("y",cy+S/2+6).attr("font-size",16).attr("fill",C.muted).text("×");
    // A (r×d) wide short
    const ah=Math.max(5,Math.min(34,r/128*34)), ax=bx+bw+18;
    svg.append("rect").attr("x",ax).attr("y",cy+S/2-ah/2).attr("width",S).attr("height",ah).attr("rx",2).attr("fill",C.accent);
    svg.append("text").attr("x",ax+S/2).attr("y",cy+S/2-ah/2-6).attr("text-anchor","middle").attr("fill",C.accent).attr("font-size",9).text("A   (rank r = "+r+")");
    svg.append("text").attr("x",ax+S+14).attr("y",cy+S/2+6).attr("font-size",11).attr("fill",C.muted).text("trainable");
    // params comparison bars
    const by=232, bw2=W-64;
    svg.append("text").attr("x",30).attr("y",by-10).attr("fill",C.muted).attr("font-size",11).text("trainable parameters per matrix");
    svg.append("rect").attr("x",30).attr("y",by).attr("width",bw2).attr("height",13).attr("rx",3).attr("fill","#4b5563");
    svg.append("text").attr("x",30+8).attr("y",by+10).attr("font-size",9.5).attr("fill","#e6e9ef").text("full fine-tuning  "+full.toLocaleString());
    const lw=Math.max(2,lora/full*bw2);
    svg.append("rect").attr("x",30).attr("y",by+22).attr("width",bw2).attr("height",13).attr("rx",3).attr("fill","#15181f");
    svg.append("rect").attr("x",30).attr("y",by+22).attr("width",lw).attr("height",13).attr("rx",3).attr("fill",C.accent);
    svg.append("text").attr("x",30+Math.max(lw+8,8)).attr("y",by+32).attr("font-size",9.5).attr("fill",C.accent).text("LoRA  "+lora.toLocaleString());
    d3.select("#peft-read").html(`rank <b>r=${r}</b> → LoRA trains <b>${lora.toLocaleString()}</b> params = <b>${pct.toFixed(2)}%</b> of full fine-tuning's <b>${full.toLocaleString()}</b> (per ${d}×${d} matrix)`);
  }
  d3.select("#peft-r").on("input",draw);
  draw();
})();
