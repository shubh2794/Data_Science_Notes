/* llm-serving.viz.js — two interactive D3 visualizations:
   1. Roofline model showing compute-bound vs memory-bound regions
   2. Static vs continuous batching GPU utilization simulation
   Loaded after data.js / notes.js. */

(function(){
  const C={accent:"#a3e635",good:"#4ade80",bad:"#f87171",a2:"#ffb454",blue:"#60a5fa",
           purple:"#c084fc",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a",panel:"#15181f"};

  /* ================================================================
     VIZ 1 — Roofline model: prefill vs decode
     ================================================================ */
  (function roofline(){
    var svg=d3.select("#roof-svg"), W=640, H=280;
    if(svg.empty()) return;
    var m={l:56,r:24,t:24,b:44}, pw=W-m.l-m.r, ph=H-m.t-m.b;

    // H100-ish numbers (simplified for illustration)
    var peakFLOPS=990;  // TFLOPS FP16
    var peakBW=3.35;    // TB/s HBM

    // ridge point: arithmetic intensity where roofline bends
    var ridge=peakFLOPS/peakBW; // ~296 ops/byte

    // define arrowhead marker once in a persistent defs block
    var defs=svg.append("defs");
    defs.append("marker").attr("id","arrow").attr("viewBox","0 0 10 10")
       .attr("refX",9).attr("refY",5).attr("markerWidth",6).attr("markerHeight",6)
       .attr("orient","auto-start-reverse")
       .append("path").attr("d","M 0 0 L 10 5 L 0 10 z").attr("fill",C.good);

    var drawG=svg.append("g");

    function draw(){
      drawG.selectAll("*").remove();
      var bs=+d3.select("#roof-bs").property("value");
      d3.select("#roof-bs-v").text(bs);

      // log-log axes
      var xMin=0.5, xMax=2048;
      var yMin=0.5, yMax=peakFLOPS*1.3;
      var x=d3.scaleLog().domain([xMin,xMax]).range([m.l,m.l+pw]);
      var y=d3.scaleLog().domain([yMin,yMax]).range([m.t+ph,m.t]);

      // grid lines
      [1,10,100,1000].forEach(function(v){
        if(v>=xMin && v<=xMax){
          drawG.append("line").attr("x1",x(v)).attr("y1",m.t).attr("x2",x(v)).attr("y2",m.t+ph)
             .attr("stroke",C.line).attr("stroke-width",0.5);
          drawG.append("text").attr("x",x(v)).attr("y",m.t+ph+14).attr("text-anchor","middle")
             .attr("font-size",9).attr("fill",C.muted).text(v);
        }
      });
      [1,10,100,1000].forEach(function(v){
        if(v>=yMin && v<=yMax){
          drawG.append("line").attr("x1",m.l).attr("y1",y(v)).attr("x2",m.l+pw).attr("y2",y(v))
             .attr("stroke",C.line).attr("stroke-width",0.5);
          drawG.append("text").attr("x",m.l-6).attr("y",y(v)+3).attr("text-anchor","end")
             .attr("font-size",9).attr("fill",C.muted).text(v);
        }
      });

      // axis labels
      drawG.append("text").attr("x",m.l+pw/2).attr("y",m.t+ph+34).attr("text-anchor","middle")
         .attr("font-size",10).attr("fill",C.muted).text("arithmetic intensity (ops/byte)");
      drawG.append("text").attr("x",m.l-44).attr("y",m.t+ph/2).attr("text-anchor","middle")
         .attr("font-size",10).attr("fill",C.muted)
         .attr("transform","rotate(-90,"+( m.l-44)+","+(m.t+ph/2)+")").text("TFLOPS achieved");

      // roofline: two segments
      // memory-bound: y = BW * x  (up to ridge)
      // compute-bound: y = peakFLOPS (flat)
      var roofPts=[];
      var nPts=200;
      for(var i=0;i<nPts;i++){
        var xi=xMin*Math.pow(xMax/xMin, i/(nPts-1));
        var yi=Math.min(peakFLOPS, peakBW*xi);
        roofPts.push([xi,yi]);
      }
      var roofLine=d3.line().x(function(d){return x(d[0]);}).y(function(d){return y(d[1]);});
      drawG.append("path").datum(roofPts).attr("fill","none")
         .attr("stroke",C.ink).attr("stroke-width",2).attr("d",roofLine);

      // shade regions
      // memory-bound region (left of ridge)
      drawG.append("text").attr("x",x(Math.sqrt(xMin*ridge))).attr("y",m.t+14)
         .attr("text-anchor","middle").attr("font-size",10).attr("fill",C.bad).attr("opacity",0.7)
         .text("memory-bound");
      // compute-bound region (right of ridge)
      drawG.append("text").attr("x",x(Math.sqrt(ridge*xMax))).attr("y",m.t+14)
         .attr("text-anchor","middle").attr("font-size",10).attr("fill",C.blue).attr("opacity",0.7)
         .text("compute-bound");

      // ridge point indicator
      drawG.append("line").attr("x1",x(ridge)).attr("y1",m.t).attr("x2",x(ridge)).attr("y2",m.t+ph)
         .attr("stroke",C.muted).attr("stroke-dasharray","4 3").attr("opacity",0.5);

      // plot prefill and decode points
      // prefill: high arithmetic intensity (~200-1000 depending on seq len), near peak
      var prefillAI=400; // typical large prefill
      var prefillPerf=Math.min(peakFLOPS, peakBW*prefillAI)*0.7; // 70% of peak (realistic)

      // decode: low arithmetic intensity, depends on batch size
      // single token decode: AI ~ 1-2 (tiny amount of compute per weight loaded)
      // with batch size bs: AI ~ bs * base_AI
      var decodeBaseAI=1.5;
      var decodeAI=decodeBaseAI*bs;
      var decodePerf=Math.min(peakFLOPS, peakBW*decodeAI)*0.6; // 60% of achievable

      // prefill dot
      drawG.append("circle").attr("cx",x(prefillAI)).attr("cy",y(prefillPerf)).attr("r",7)
         .attr("fill",C.blue).attr("stroke",C.ink).attr("stroke-width",1.5);
      drawG.append("text").attr("x",x(prefillAI)+12).attr("y",y(prefillPerf)+4)
         .attr("font-size",11).attr("font-weight",700).attr("fill",C.blue).text("prefill");

      // decode dot
      drawG.append("circle").attr("cx",x(Math.max(xMin,decodeAI))).attr("cy",y(Math.max(yMin,decodePerf))).attr("r",7)
         .attr("fill",C.a2).attr("stroke",C.ink).attr("stroke-width",1.5);
      var decodeLabel = "decode (bs="+bs+")";
      drawG.append("text").attr("x",x(Math.max(xMin,decodeAI))+12).attr("y",y(Math.max(yMin,decodePerf))+4)
         .attr("font-size",11).attr("font-weight",700).attr("fill",C.a2).text(decodeLabel);

      // arrow showing batch size effect
      if(bs>1){
        var baseAI=decodeBaseAI;
        drawG.append("line")
           .attr("x1",x(baseAI)+8).attr("y1",y(Math.min(peakFLOPS,peakBW*baseAI)*0.6))
           .attr("x2",x(decodeAI)-8).attr("y2",y(decodePerf))
           .attr("stroke",C.good).attr("stroke-width",1.5).attr("stroke-dasharray","4 2")
           .attr("marker-end","url(#arrow)");
      }

      // readout
      var decodeRegion = decodeAI < ridge ? "memory-bound" : "compute-bound";
      var utilPct = (decodePerf/peakFLOPS*100).toFixed(1);
      d3.select("#roof-read").html(
        "Prefill: <b style='color:"+C.blue+"'>compute-bound</b> (AI="+prefillAI+") &nbsp;&middot;&nbsp; "+
        "Decode at bs="+bs+": <b style='color:"+C.a2+"'>"+decodeRegion+"</b> (AI="+decodeAI.toFixed(1)+
        ") &nbsp;&middot;&nbsp; GPU utilization: <b>"+utilPct+"%</b>"+
        (bs>1?" &nbsp;&middot;&nbsp; <span style='color:"+C.good+"'>batching moves decode right on the roofline</span>":"")
      );
    }
    d3.select("#roof-bs").on("input",draw);
    draw();
  })();

  /* ================================================================
     VIZ 2 — Static vs continuous batching simulation
     ================================================================ */
  (function batching(){
    var svg=d3.select("#batch-svg"), W=640, H=240;
    if(svg.empty()) return;
    var m={l:50,r:16,t:28,b:36};
    var pw=W-m.l-m.r, ph=H-m.t-m.b;

    var SLOTS=4;    // GPU slots
    var STEPS=20;   // time steps

    // fixed sequence lengths for consistent comparison
    var seqPool=[3,7,5,12,4,8,6,15,3,9,11,4,7,5,10,6,8,13,4,7];

    function simulate(mode){
      // grid: slots x steps, 0=empty, >0 = request id
      var grid=[];
      for(var s=0;s<SLOTS;s++){
        grid.push(new Array(STEPS).fill(0));
      }
      var colors=[];
      var reqIdx=0;
      var nextId=1;

      if(mode==="static"){
        // static: fill all slots, wait for longest to finish, then refill
        var t=0;
        while(t<STEPS){
          // fill batch
          var batchEnd=t;
          for(var s=0;s<SLOTS;s++){
            var len=seqPool[reqIdx%seqPool.length];
            reqIdx++;
            var id=nextId++;
            colors[id]=id;
            for(var dt=0;dt<len && (t+dt)<STEPS;dt++){
              grid[s][t+dt]=id;
            }
            batchEnd=Math.max(batchEnd, t+len);
          }
          t=Math.min(batchEnd, STEPS);
        }
      } else {
        // continuous: at each step, evict finished, insert new
        var active=[]; // {slot, id, remaining}
        for(var s=0;s<SLOTS;s++) active.push(null);

        for(var t=0;t<STEPS;t++){
          // decrement remaining, evict finished
          for(var s=0;s<SLOTS;s++){
            if(active[s]){
              active[s].remaining--;
              if(active[s].remaining<=0) active[s]=null;
            }
          }
          // fill empty slots
          for(var s=0;s<SLOTS;s++){
            if(!active[s]){
              var len=seqPool[reqIdx%seqPool.length];
              reqIdx++;
              var id=nextId++;
              colors[id]=id;
              active[s]={slot:s, id:id, remaining:len};
            }
          }
          // write grid
          for(var s=0;s<SLOTS;s++){
            if(active[s]) grid[s][t]=active[s].id;
          }
        }
      }
      return {grid:grid, colors:colors, totalReqs:nextId-1};
    }

    // color palette for requests
    var palette=["#60a5fa","#f87171","#a3e635","#c084fc","#ffb454","#4ade80",
                 "#fb923c","#38bdf8","#e879f9","#fcd34d","#22d3ee","#f472b6",
                 "#818cf8","#2dd4bf","#94a3b8","#fb7185","#a78bfa","#34d399",
                 "#f59e0b","#ec4899","#6366f1","#14b8a6","#eab308","#d946ef"];

    function draw(){
      svg.selectAll("*").remove();
      var mode=d3.select("input[name='batch-mode']:checked").property("value");
      var result=simulate(mode);
      var grid=result.grid;

      var cellW=pw/STEPS;
      var cellH=ph/SLOTS;
      var gap=2;

      // title
      svg.append("text").attr("x",m.l).attr("y",m.t-8).attr("font-size",11).attr("fill",C.muted)
         .text(mode==="static"?"static batching — slots idle while waiting for longest sequence"
                               :"continuous batching — finished slots immediately refilled");

      // draw grid
      var occupied=0, total=SLOTS*STEPS;
      for(var s=0;s<SLOTS;s++){
        // slot label
        svg.append("text").attr("x",m.l-8).attr("y",m.t+s*cellH+cellH/2+4)
           .attr("text-anchor","end").attr("font-size",9).attr("fill",C.muted).text("slot "+s);

        for(var t=0;t<STEPS;t++){
          var x=m.l+t*cellW;
          var yy=m.t+s*cellH;
          var id=grid[s][t];
          var fill = id>0 ? palette[(id-1)%palette.length] : C.panel;
          var op = id>0 ? 0.8 : 0.3;
          if(id>0) occupied++;
          svg.append("rect").attr("x",x+gap/2).attr("y",yy+gap/2)
             .attr("width",cellW-gap).attr("height",cellH-gap).attr("rx",3)
             .attr("fill",fill).attr("opacity",op).attr("stroke",C.line).attr("stroke-width",0.5);
        }
      }

      // time axis
      for(var t=0;t<STEPS;t+=2){
        svg.append("text").attr("x",m.l+t*cellW+cellW/2).attr("y",m.t+ph+14)
           .attr("text-anchor","middle").attr("font-size",8).attr("fill",C.muted).text(t);
      }
      svg.append("text").attr("x",m.l+pw/2).attr("y",m.t+ph+30).attr("text-anchor","middle")
         .attr("font-size",10).attr("fill",C.muted).text("decode iteration");

      // utilization
      var util=(occupied/total*100).toFixed(1);
      var utilColor = util>70 ? C.good : util>50 ? C.a2 : C.bad;
      d3.select("#batch-read").html(
        "GPU utilization: <b style='color:"+utilColor+"'>"+util+"%</b> ("+occupied+"/"+total+" slot-steps occupied) &nbsp;&middot;&nbsp; "+
        "Requests served: <b>"+result.totalReqs+"</b>"+
        (mode==="static"?" &nbsp;&middot;&nbsp; <span style='color:"+C.bad+"'>slots idle while waiting for longest sequence in batch</span>"
                        :" &nbsp;&middot;&nbsp; <span style='color:"+C.good+"'>finished slots immediately filled with new requests</span>")
      );
    }

    d3.selectAll("input[name='batch-mode']").on("change",draw);
    d3.select("#batch-run").on("click",draw);
    draw();
  })();

})();
