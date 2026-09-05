/* dl-architectures-compared.viz.js — extracted from dl-architectures-compared.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#f87171",a2:"#ffb454",good:"#4ade80",blue:"#5b9cff",ink:"#e6e9ef",muted:"#9aa3b2",line:"#2a2f3a"};
  const svg=d3.select("#cmp-svg"),W=640,H=240;
  const props=["parallelism","long-range capture","inductive-bias strength","compute efficiency"];
  const data={
    MLP:{v:[0.9,0.4,0.1,0.7],note:"No structural assumptions — universal but data-hungry; for tabular/fixed vectors."},
    CNN:{v:[0.9,0.3,0.85,0.85],note:"Locality + translation invariance — efficient and strong on images/grids."},
    RNN:{v:[0.2,0.45,0.7,0.6],note:"Sequential recurrence — natural for sequences but slow and forgets long range."},
    Transformer:{v:[1.0,1.0,0.2,0.4],note:"Any-to-any attention, fully parallel; weak bias, O(n²) cost — wins at scale."},
    SSM:{v:[0.8,0.85,0.5,0.95],note:"Linear-time recurrence — long sequences cheaply; weaker direct recall than attention."},
    GNN:{v:[0.6,0.5,0.85,0.6],note:"Relational structure via message passing — for graphs/molecules."},
  };
  function draw(){
    svg.selectAll("*").remove();
    const k=d3.select("#cmp-sel").property("value");
    const d=data[k];
    const x0=210, top=34, rowH=44, bw=W-x0-70;
    svg.append("text").attr("x",24).attr("y",24).attr("fill",C.ink).attr("font-size",13).attr("font-weight",600).text(k);
    props.forEach((p,i)=>{
      const y=top+i*rowH;
      svg.append("text").attr("x",x0-12).attr("y",y+15).attr("text-anchor","end").attr("font-size",11).attr("fill",C.muted).text(p);
      svg.append("rect").attr("x",x0).attr("y",y+4).attr("width",bw).attr("height",16).attr("rx",4).attr("fill","#15181f");
      const col=d.v[i]>0.66?C.good:d.v[i]>0.4?C.a2:C.accent;
      svg.append("rect").attr("x",x0).attr("y",y+4).attr("width",d.v[i]*bw).attr("height",16).attr("rx",4).attr("fill",col);
      svg.append("text").attr("x",x0+d.v[i]*bw+8).attr("y",y+16).attr("font-size",9.5).attr("fill",col).text(Math.round(d.v[i]*100));
    });
    document.getElementById("cmp-read").innerHTML=d.note;
  }
  d3.select("#cmp-sel").on("change",draw);
  draw();
})();
