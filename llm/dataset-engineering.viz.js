/* dataset-engineering.viz.js — data mixing dashboard for dataset-engineering.html.
   Loaded after data.js / notes.js. IIFE pattern; guards its own elements.
   Uses d3 from ../vendor/d3.min.js (no CDN). */

(function(){
  var svgEl = document.getElementById("mix-svg");
  if (!svgEl || typeof d3 === "undefined") return;

  var readEl  = document.getElementById("mix-read");
  var btnBal  = document.getElementById("mix-balanced");
  var btnCode = document.getElementById("mix-code");
  var btnChat = document.getElementById("mix-chat");
  if (!btnBal || !btnCode || !btnChat) return;

  /* ---- colour palette ---- */
  var C = {
    accent:"#a3e635", good:"#4ade80", bad:"#f87171", a2:"#ffb454",
    blue:"#60a5fa", purple:"#c084fc", muted:"#9aa3b2", ink:"#e6e9ef",
    line:"#2a2f3a", panel:"#15181f"
  };

  /* ---- domain definitions ---- */
  var domains = ["Web","Code","Books","Math","Conversation","Other"];
  var domainColors = [C.blue, C.accent, C.a2, C.purple, C.good, C.muted];

  /* ---- preset mixes (percentages, must sum to 100) ---- */
  var presets = {
    balanced: { label:"Balanced",   mix:[50, 20, 10, 8, 7, 5] },
    code:     { label:"Code-heavy", mix:[30, 40, 5, 15, 5, 5] },
    chat:     { label:"Chat-heavy", mix:[25, 10, 10, 5, 40, 10] }
  };

  /* ---- data sources for scatter (name, tokens in B, quality 0–1) ---- */
  var sourcesByPreset = {
    balanced: [
      {name:"Common Crawl",   tokens:4200, quality:0.35},
      {name:"Refined Web",    tokens:2800, quality:0.65},
      {name:"GitHub Code",    tokens:800,  quality:0.55},
      {name:"StackExchange",  tokens:120,  quality:0.72},
      {name:"Wikipedia",      tokens:40,   quality:0.90},
      {name:"Textbooks",      tokens:25,   quality:0.95},
      {name:"ArXiv",          tokens:90,   quality:0.80},
      {name:"Books3",         tokens:200,  quality:0.60},
      {name:"OpenWebMath",    tokens:35,   quality:0.82},
      {name:"Chat logs",      tokens:60,   quality:0.45},
      {name:"Multilingual",   tokens:300,  quality:0.42}
    ],
    code: [
      {name:"Common Crawl",   tokens:2000, quality:0.35},
      {name:"Refined Web",    tokens:1500, quality:0.65},
      {name:"GitHub Code",    tokens:2500, quality:0.55},
      {name:"StackExchange",  tokens:400,  quality:0.72},
      {name:"Wikipedia",      tokens:30,   quality:0.90},
      {name:"Textbooks",      tokens:60,   quality:0.95},
      {name:"ArXiv",          tokens:200,  quality:0.80},
      {name:"Books3",         tokens:50,   quality:0.60},
      {name:"OpenWebMath",    tokens:150,  quality:0.82},
      {name:"Chat logs",      tokens:30,   quality:0.45},
      {name:"Multilingual",   tokens:100,  quality:0.42}
    ],
    chat: [
      {name:"Common Crawl",   tokens:1800, quality:0.35},
      {name:"Refined Web",    tokens:1200, quality:0.65},
      {name:"GitHub Code",    tokens:400,  quality:0.55},
      {name:"StackExchange",  tokens:300,  quality:0.72},
      {name:"Wikipedia",      tokens:50,   quality:0.90},
      {name:"Textbooks",      tokens:30,   quality:0.95},
      {name:"ArXiv",          tokens:60,   quality:0.80},
      {name:"Books3",         tokens:150,  quality:0.60},
      {name:"OpenWebMath",    tokens:20,   quality:0.82},
      {name:"Chat logs",      tokens:2200, quality:0.45},
      {name:"Multilingual",   tokens:400,  quality:0.42}
    ]
  };

  /* ---- SVG setup ---- */
  var W = 720, H = 420;
  var svg = d3.select(svgEl);
  svg.selectAll("*").remove();

  /* background */
  svg.append("rect").attr("width",W).attr("height",H).attr("fill",C.panel).attr("rx",10);

  /* ---- Stacked bar region ---- */
  var barX = 60, barY = 30, barW = 600, barH = 50;

  svg.append("text").attr("x",barX).attr("y",barY - 10)
     .attr("fill",C.ink).attr("font-size",13).attr("font-weight",600)
     .text("Domain mix (%)");

  var barG = svg.append("g").attr("class","bar-group");

  /* legend */
  var legG = svg.append("g").attr("transform","translate("+barX+","+(barY + barH + 14)+")");
  domains.forEach(function(d, i){
    var gx = i * 110;
    legG.append("rect").attr("x",gx).attr("y",0).attr("width",12).attr("height",12)
        .attr("rx",3).attr("fill",domainColors[i]);
    legG.append("text").attr("x",gx+16).attr("y",10)
        .attr("fill",C.muted).attr("font-size",11).text(d);
  });

  /* ---- Scatter region ---- */
  var scX = 60, scY = 160, scW = 600, scH = 220;
  var dotMaxR = 28, dotMinR = 5;

  svg.append("text").attr("x",scX).attr("y",scY - 8)
     .attr("fill",C.ink).attr("font-size",13).attr("font-weight",600)
     .text("Data sources: quality vs quantity");

  /* axes */
  var xScale = d3.scaleLinear().range([scX, scX + scW]);
  var yScale = d3.scaleLinear().domain([0, 1]).range([scY + scH, scY]);
  var rScale = d3.scaleSqrt().range([dotMinR, dotMaxR]);

  /* y-axis label */
  svg.append("text").attr("x",scX - 42).attr("y",scY + scH/2)
     .attr("fill",C.muted).attr("font-size",11).attr("text-anchor","middle")
     .attr("transform","rotate(-90,"+( scX - 42)+","+(scY + scH/2)+")")
     .text("Quality score");

  /* x-axis label */
  svg.append("text").attr("x",scX + scW/2).attr("y",scY + scH + 36)
     .attr("fill",C.muted).attr("font-size",11).attr("text-anchor","middle")
     .text("Tokens (billions)");

  /* gridlines */
  var gridG = svg.append("g");
  [0, 0.25, 0.5, 0.75, 1.0].forEach(function(v){
    var yy = yScale(v);
    gridG.append("line").attr("x1",scX).attr("x2",scX+scW)
         .attr("y1",yy).attr("y2",yy)
         .attr("stroke",C.line).attr("stroke-dasharray","3,3");
    gridG.append("text").attr("x",scX - 6).attr("y",yy + 4)
         .attr("fill",C.muted).attr("font-size",10).attr("text-anchor","end")
         .text(v.toFixed(2));
  });

  /* scatter group */
  var dotG = svg.append("g").attr("class","dot-group");

  /* tooltip */
  var tip = svg.append("g").attr("class","tip").style("display","none");
  tip.append("rect").attr("rx",6).attr("fill","rgba(21,24,31,0.92)").attr("stroke",C.line);
  tip.append("text").attr("fill",C.ink).attr("font-size",11);

  /* ---- draw / update ---- */
  var current = "code";

  function update(preset, animate){
    current = preset;
    var mix = presets[preset].mix;
    var sources = sourcesByPreset[preset];

    /* ---- stacked bar ---- */
    var cumul = [0];
    mix.forEach(function(v, i){ cumul.push(cumul[i] + v); });

    var bars = barG.selectAll("rect.seg").data(mix);
    bars.enter().append("rect").attr("class","seg")
        .attr("y", barY).attr("height", barH).attr("rx", 4)
        .merge(bars)
        .transition().duration(animate ? 500 : 0)
        .attr("x", function(d,i){ return barX + cumul[i] / 100 * barW; })
        .attr("width", function(d){ return Math.max(0, d / 100 * barW - 1); })
        .attr("fill", function(d,i){ return domainColors[i]; });

    /* percent labels */
    barG.selectAll("text.pct").remove();
    mix.forEach(function(v, i){
      if (v < 5) return; /* skip tiny slices */
      var cx = barX + (cumul[i] + v/2) / 100 * barW;
      barG.append("text").attr("class","pct")
          .attr("x", cx).attr("y", barY + barH/2 + 5)
          .attr("text-anchor","middle")
          .attr("fill", (v > 10) ? "#000" : C.ink)
          .attr("font-size", 12).attr("font-weight", 600)
          .text(v + "%");
    });

    /* ---- scatter ---- */
    var maxTok = d3.max(sources, function(s){ return s.tokens; });
    xScale.domain([0, maxTok * 1.15]);
    rScale.domain([0, maxTok]);

    /* x-axis ticks */
    svg.selectAll("text.xtick").remove();
    var xTicks = xScale.ticks(5);
    xTicks.forEach(function(v){
      svg.append("text").attr("class","xtick")
         .attr("x", xScale(v)).attr("y", scY + scH + 16)
         .attr("fill",C.muted).attr("font-size",10).attr("text-anchor","middle")
         .text(v >= 1000 ? (v/1000).toFixed(1) + "T" : v + "B");
    });

    var dots = dotG.selectAll("circle.src").data(sources, function(d){ return d.name; });

    dots.enter().append("circle").attr("class","src")
        .attr("cx", function(d){ return xScale(d.tokens); })
        .attr("cy", function(d){ return yScale(d.quality); })
        .attr("r", 0)
        .attr("fill", function(d){
          if (d.quality >= 0.8) return C.good;
          if (d.quality >= 0.6) return C.a2;
          if (d.quality >= 0.4) return C.blue;
          return C.bad;
        })
        .attr("fill-opacity", 0.7)
        .attr("stroke", C.ink).attr("stroke-width", 1)
        .on("mouseover", function(ev, d){
          d3.select(this).attr("fill-opacity",1).attr("stroke-width",2);
          var tx = xScale(d.tokens), ty = yScale(d.quality) - rScale(d.tokens) - 10;
          var label = d.name + "  " + d.tokens + "B tokens  quality " + d.quality.toFixed(2);
          var tt = tip.select("text");
          tt.text(label);
          var bbox = tt.node().getBBox ? tt.node().getBBox() : {width:120,height:14};
          tip.select("rect").attr("x", bbox.x - 6).attr("y", bbox.y - 4)
             .attr("width", bbox.width + 12).attr("height", bbox.height + 8);
          tip.attr("transform","translate("+tx+","+ty+")").style("display",null);
          tip.select("text").attr("text-anchor","middle");
        })
        .on("mouseout", function(){
          d3.select(this).attr("fill-opacity",0.7).attr("stroke-width",1);
          tip.style("display","none");
        })
        .merge(dots)
        .transition().duration(animate ? 500 : 0)
        .attr("cx", function(d){ return xScale(d.tokens); })
        .attr("cy", function(d){ return yScale(d.quality); })
        .attr("r",  function(d){ return rScale(d.tokens); })
        .attr("fill", function(d){
          if (d.quality >= 0.8) return C.good;
          if (d.quality >= 0.6) return C.a2;
          if (d.quality >= 0.4) return C.blue;
          return C.bad;
        });

    dots.exit().transition().duration(300).attr("r",0).remove();

    /* ---- readout ---- */
    if (readEl) {
      readEl.textContent = presets[preset].label + " preset — " +
        domains.map(function(d,i){ return d + " " + mix[i] + "%"; }).join(", ");
    }

    /* button active states */
    [btnBal, btnCode, btnChat].forEach(function(b){ b.classList.remove("active"); });
    if (preset === "balanced") btnBal.classList.add("active");
    if (preset === "code")     btnCode.classList.add("active");
    if (preset === "chat")     btnChat.classList.add("active");
  }

  /* ---- button handlers ---- */
  btnBal.addEventListener("click",  function(){ update("balanced", true); });
  btnCode.addEventListener("click", function(){ update("code", true); });
  btnChat.addEventListener("click", function(){ update("chat", true); });

  /* ---- initial render ---- */
  update("balanced", false);

})();
