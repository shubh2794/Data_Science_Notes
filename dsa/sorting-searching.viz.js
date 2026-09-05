/* sorting-searching.viz.js — extracted from sorting-searching.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const svg=d3.select("#sort-svg"), W=680, H=220, base=180;
  const N=14;
  let algo="Bubble", arr, frames, fi, timer=null, comps, swaps;

  function randArr(){ return d3.range(N).map(()=>8+Math.round(Math.random()*150)); }

  // Build a list of frames; each frame = {a:[...], hi:[i,j], sorted:Set, comps, swaps, note}
  function recordBubble(a){
    const f=[]; a=a.slice(); let c=0,s=0; const sorted=new Set();
    for(let i=0;i<a.length-1;i++){
      for(let j=0;j<a.length-1-i;j++){
        c++;
        f.push({a:a.slice(),hi:[j,j+1],sorted:new Set(sorted),comps:c,swaps:s,note:`compare ${a[j]} & ${a[j+1]}`});
        if(a[j]>a[j+1]){ [a[j],a[j+1]]=[a[j+1],a[j]]; s++;
          f.push({a:a.slice(),hi:[j,j+1],sorted:new Set(sorted),comps:c,swaps:s,note:`swap → ${a[j]}, ${a[j+1]}`}); }
      }
      sorted.add(a.length-1-i);
    }
    sorted.clear(); for(let k=0;k<a.length;k++) sorted.add(k);
    f.push({a:a.slice(),hi:[],sorted:new Set(sorted),comps:c,swaps:s,note:`sorted · ${c} comparisons, ${s} swaps`});
    return f;
  }
  function recordQuick(a){
    const f=[]; a=a.slice(); let c={n:0}, s={n:0}; const sorted=new Set();
    function qs(lo,hi){
      if(lo>hi){ return; }
      if(lo===hi){ sorted.add(lo); return; }
      const pivot=a[hi]; let i=lo;
      for(let j=lo;j<hi;j++){
        c.n++;
        f.push({a:a.slice(),hi:[j,hi],sorted:new Set(sorted),comps:c.n,swaps:s.n,note:`compare ${a[j]} vs pivot ${pivot}`});
        if(a[j]<pivot){ if(i!==j){[a[i],a[j]]=[a[j],a[i]]; s.n++;
          f.push({a:a.slice(),hi:[i,j],sorted:new Set(sorted),comps:c.n,swaps:s.n,note:`swap → smaller left`});} i++; }
      }
      if(i!==hi){[a[i],a[hi]]=[a[hi],a[i]]; s.n++;}
      sorted.add(i);
      f.push({a:a.slice(),hi:[i],sorted:new Set(sorted),comps:c.n,swaps:s.n,note:`pivot ${pivot} placed at ${i}`});
      qs(lo,i-1); qs(i+1,hi);
    }
    qs(0,a.length-1);
    for(let k=0;k<a.length;k++) sorted.add(k);
    f.push({a:a.slice(),hi:[],sorted:new Set(sorted),comps:c.n,swaps:s.n,note:`sorted · ${c.n} comparisons, ${s.n} swaps`});
    return f;
  }

  const x=d3.scaleBand().domain(d3.range(N)).range([20,W-20]).padding(0.18);
  const gB=svg.append("g");

  function draw(fr){
    const bars=gB.selectAll("rect").data(fr.a);
    bars.enter().append("rect").attr("rx",3)
      .merge(bars)
      .attr("x",(d,i)=>x(i)).attr("width",x.bandwidth())
      .attr("y",d=>base-d).attr("height",d=>d)
      .attr("fill",(d,i)=> fr.sorted.has(i)?C.good : fr.hi.includes(i)?C.B : C.A);
    d3.select("#sort-readout").html(`${fr.note} &nbsp;·&nbsp; comparisons <b>${fr.comps}</b>, swaps <b>${fr.swaps}</b>`);
  }
  function build(){
    arr=randArr();
    frames = algo==="Bubble" ? recordBubble(arr) : recordQuick(arr);
    fi=0; draw(frames[0]);
  }
  function step(){ if(fi<frames.length-1){ fi++; draw(frames[fi]); } else stop(); }
  function play(){ if(timer) return; d3.select("#sort-play").text("Pause");
    timer=setInterval(()=>{ if(fi<frames.length-1){fi++;draw(frames[fi]);} else stop(); }, 140); }
  function stop(){ if(timer){clearInterval(timer);timer=null;} d3.select("#sort-play").text("Play"); }

  d3.select("#sort-algo").on("click",function(){
    stop(); algo = algo==="Bubble" ? "Quick" : "Bubble";
    d3.select(this).text("Algorithm: "+algo); build();
  });
  d3.select("#sort-step").on("click",()=>{ stop(); step(); });
  d3.select("#sort-play").on("click",()=>{ timer?stop():play(); });
  d3.select("#sort-reset").on("click",()=>{ stop(); build(); });
  build();
})();
