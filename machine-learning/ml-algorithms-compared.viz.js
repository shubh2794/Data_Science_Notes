/* ml-algorithms-compared.viz.js — extracted from ml-algorithms-compared.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#ffb454",blue:"#5b9cff",good:"#4ade80",bad:"#f87171",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#mlc-svg"),W=640,H=330;
  const props=["interpretability","training speed","inference speed","handles non-linearity","needs feature scaling","works with little data","tolerates junk features"];
  const A={
    lr:{v:[0.90,0.95,0.98,0.15,0.75,0.70,0.35],n:"One hyperplane. Fast to fit, fast to serve, and every coefficient is readable — but it only sees the linearity you put in the features."},
    knn:{v:[0.55,1.00,0.15,0.85,0.95,0.55,0.10],n:"No training at all; all the cost is at prediction time. The metric is the model, so scaling matters and irrelevant features are poison."},
    nb:{v:[0.75,0.95,0.95,0.35,0.15,0.90,0.45],n:"One counting pass. Unbeatable time-to-baseline on sparse text and tiny datasets; probabilities are overconfident when features correlate."},
    svm:{v:[0.35,0.40,0.55,0.90,0.95,0.65,0.40],n:"Smooth max-margin boundary via a kernel. Strong on clean medium-sized problems; training is super-linear in n and C/γ must be tuned jointly."},
    tree:{v:[0.95,0.80,0.95,0.85,0.05,0.55,0.65],n:"Axis-aligned boxes, readable as rules, indifferent to scaling and happy with mixed types — but high variance on its own and unable to extrapolate."},
    rf:{v:[0.45,0.60,0.65,0.90,0.05,0.60,0.85],n:"Averaged decorrelated trees: most of the tree strengths, far less variance, almost no tuning. Large in memory, and no longer a readable rule set."},
    gb:{v:[0.35,0.40,0.70,0.95,0.05,0.50,0.80],n:"Sequential small corrections. Usually the best tabular accuracy available, at the price of tuning and a training loop that will fit noise if unchecked."},
    mlp:{v:[0.15,0.25,0.75,0.98,0.90,0.20,0.55],n:"Learns its own representation given enough data. Needs scale, tuning and normalization; on small heterogeneous tables it typically trails boosting."},
  };
  function draw(){
    svg.selectAll("*").remove();
    const s1=document.getElementById("mlc-sel"), s2=document.getElementById("mlc-sel2");
    const a=A[s1.value], b=A[s2.value];
    const nameA=s1.options[s1.selectedIndex].text, nameB=s2.options[s2.selectedIndex].text;
    const x0=210, top=44, rowH=38, bw=W-x0-58;
    svg.append("rect").attr("x",24).attr("y",12).attr("width",10).attr("height",10).attr("rx",2).attr("fill",C.blue);
    svg.append("text").attr("x",40).attr("y",21).attr("fill",C.ink).attr("font-size",11.5).text(nameA);
    svg.append("rect").attr("x",24).attr("y",28).attr("width",10).attr("height",10).attr("rx",2).attr("fill",C.accent);
    svg.append("text").attr("x",40).attr("y",37).attr("fill",C.ink).attr("font-size",11.5).text(nameB);
    props.forEach((p,i)=>{
      const y=top+i*rowH;
      svg.append("text").attr("x",x0-12).attr("y",y+16).attr("text-anchor","end").attr("font-size",10.5).attr("fill",C.muted).text(p);
      svg.append("rect").attr("x",x0).attr("y",y).attr("width",bw).attr("height",11).attr("rx",3).attr("fill","#15181f");
      svg.append("rect").attr("x",x0).attr("y",y+13).attr("width",bw).attr("height",11).attr("rx",3).attr("fill","#15181f");
      svg.append("rect").attr("x",x0).attr("y",y).attr("width",Math.max(2,a.v[i]*bw)).attr("height",11).attr("rx",3).attr("fill",C.blue);
      svg.append("rect").attr("x",x0).attr("y",y+13).attr("width",Math.max(2,b.v[i]*bw)).attr("height",11).attr("rx",3).attr("fill",C.accent);
      svg.append("text").attr("x",x0+bw+8).attr("y",y+10).attr("font-size",9.5).attr("fill",C.blue).text(Math.round(a.v[i]*100));
      svg.append("text").attr("x",x0+bw+8).attr("y",y+23).attr("font-size",9.5).attr("fill",C.accent).text(Math.round(b.v[i]*100));
    });
    // biggest divergence
    let gi=0,gd=-1; props.forEach((p,i)=>{const d=Math.abs(a.v[i]-b.v[i]); if(d>gd){gd=d;gi=i;}});
    document.getElementById("mlc-read").innerHTML =
      `<b style="color:${C.blue}">${nameA}</b> — ${a.n}<br><b style="color:${C.accent}">${nameB}</b> — ${b.n}<br>Biggest gap: <b>${props[gi]}</b> (${Math.round(gd*100)} points).`;
  }
  d3.select("#mlc-sel").on("change",draw);
  d3.select("#mlc-sel2").on("change",draw);
  draw();
})();
