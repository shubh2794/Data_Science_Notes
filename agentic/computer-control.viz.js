/* computer-control.viz.js — extracted from computer-control.html.
   Loaded after data.js / notes.js; keeps the page markup free of a large
   inline script. Same code, same load position — no behaviour change. */

(function(){
  const C={accent:"#fb923c",good:"#4ade80",blue:"#5b9cff",muted:"#9aa3b2",ink:"#e6e9ef",line:"#2a2f3a"};
  const svg=d3.select("#cc-svg"),W=640;
  const steps=[
    {act:"observe", txt:"read screenshot — a search page", target:null},
    {act:"click", txt:'click(the search box)', target:"box"},
    {act:"type", txt:'type("best ML notes")', target:"box"},
    {act:"click", txt:"click(Search)", target:"btn"},
    {act:"click", txt:"click(first result)", target:"res"},
  ];
  function draw(){
    svg.selectAll("*").remove();
    const k=+d3.select("#cc-step").property("value"); const s=steps[k];
    // mock browser window
    svg.append("rect").attr("x",30).attr("y",16).attr("width",430).attr("height",210).attr("rx",8).attr("fill","#0f131b").attr("stroke","#2a2f3a");
    svg.append("rect").attr("x",30).attr("y",16).attr("width",430).attr("height",24).attr("rx",8).attr("fill","#171a23");
    [44,60,76].forEach((cx,i)=>svg.append("circle").attr("cx",cx).attr("cy",28).attr("r",4).attr("fill",["#fb7185","#fbbf24","#4ade80"][i]));
    // search box
    const box={x:60,y:70,w:280,h:34};
    svg.append("rect").attr("x",box.x).attr("y",box.y).attr("width",box.w).attr("height",box.h).attr("rx",6).attr("fill","#15181f").attr("stroke",s.target==="box"?C.accent:"#2a2f3a").attr("stroke-width",s.target==="box"?2:1);
    svg.append("text").attr("x",box.x+10).attr("y",box.y+22).attr("font-size",12).attr("fill",k>=2?C.ink:"#5d6675").attr("font-family","SF Mono,monospace").text(k>=2?"best ML notes":"Search…");
    // button
    const btn={x:350,y:70,w:90,h:34};
    svg.append("rect").attr("x",btn.x).attr("y",btn.y).attr("width",btn.w).attr("height",btn.h).attr("rx",6).attr("fill",s.target==="btn"?"rgba(251,146,60,.25)":"rgba(91,156,255,.18)").attr("stroke",s.target==="btn"?C.accent:C.blue).attr("stroke-width",s.target==="btn"?2:1);
    svg.append("text").attr("x",btn.x+btn.w/2).attr("y",btn.y+22).attr("text-anchor","middle").attr("font-size",12).attr("fill",C.ink).text("Search");
    // results (after step 3)
    if(k>=4){ ["ML Notes — interactive","Intro to ML","ML cheatsheet"].forEach((t,i)=>{ const ry=120+i*30; svg.append("rect").attr("x",60).attr("y",ry).attr("width",380).attr("height",26).attr("rx",5).attr("fill",i===0&&s.target==="res"?"rgba(251,146,60,.18)":"#15181f").attr("stroke",i===0&&s.target==="res"?C.accent:"#2a2f3a"); svg.append("text").attr("x",70).attr("y",ry+17).attr("font-size",11).attr("fill",i===0?C.blue:C.muted).text(t); }); }
    else if(k>=1){ svg.append("text").attr("x",60).attr("y",150).attr("font-size",11).attr("fill","#5d6675").text("(results appear after search)"); }
    // cursor on target
    const tmap={box:[box.x+box.w-20,box.y+box.h/2], btn:[btn.x+btn.w/2,btn.y+btn.h/2], res:[250,133]};
    if(s.target && tmap[s.target]){ const t=tmap[s.target]; svg.append("circle").attr("cx",t[0]).attr("cy",t[1]).attr("r",10).attr("fill","none").attr("stroke",C.accent).attr("stroke-width",2);
      svg.append("text").attr("x",t[0]+4).attr("y",t[1]+22).attr("font-size",14).text("🖱"); }
    // action panel
    svg.append("rect").attr("x",478).attr("y",16).attr("width",132).attr("height",210).attr("rx",8).attr("fill","#15181f").attr("stroke","#2a2f3a");
    svg.append("text").attr("x",544).attr("y",36).attr("text-anchor","middle").attr("fill",C.muted).attr("font-size",10).text("agent");
    steps.forEach((st,i)=>{ const y=54+i*34; const on=i===k, done=i<k;
      svg.append("circle").attr("cx",492).attr("cy",y).attr("r",5).attr("fill",done?C.good:on?C.accent:"#2a2f3a");
      svg.append("text").attr("x",502).attr("y",y+4).attr("font-size",8.5).attr("fill",on?C.ink:done?C.muted:"#46506180").text(st.act); });
    d3.select("#cc-read").html(`step <b>${k+1}/5</b> · <b style="color:${C.accent}">${s.act}</b> → ${s.txt}`);
  }
  d3.select("#cc-step").on("input",draw);
  d3.select("#cc-next").on("click",()=>{ const v=Math.min(4,+d3.select("#cc-step").property("value")+1); d3.select("#cc-step").property("value",v); draw(); });
  draw();
})();

(function(){
  const C={accent:"#fb923c",good:"#4ade80",blue:"#5b9cff",muted:"#9aa3b2",ink:"#e6e9ef",dim:"#5d6675",line:"#2a2f3a"};
  const svg=d3.select("#obs-svg");
  const MONO="SF Mono,Menlo,monospace";
  const els=[
    {n:1, x:24, y:64,  w:190, h:26, kind:"textbox", label:"Search"},
    {n:2, x:222,y:64,  w:62,  h:26, kind:"button",  label:"Go"},
    {n:3, x:24, y:104, w:260, h:22, kind:"link",    label:"Interactive notes — index"},
    {n:4, x:24, y:134, w:260, h:22, kind:"link",    label:"Study plan (draft)"},
    {n:5, x:24, y:170, w:110, h:24, kind:"checkbox",label:"Only mine"},
  ];
  function screenPanel(marks){
    svg.append("rect").attr("x",14).attr("y",14).attr("width",300).attr("height",232).attr("rx",8).attr("fill","#0f131b").attr("stroke",C.line);
    svg.append("rect").attr("x",14).attr("y",14).attr("width",300).attr("height",22).attr("rx",8).attr("fill","#171a23");
    [26,40,54].forEach((cx,i)=>svg.append("circle").attr("cx",cx).attr("cy",25).attr("r",3.5).attr("fill",["#fb7185","#fbbf24","#4ade80"][i]));
    svg.append("text").attr("x",164).attr("y",29).attr("text-anchor","middle").attr("font-size",9).attr("fill",C.muted).text("notes · search");
    els.forEach(e=>{
      const g=svg.append("g").attr("transform","translate(14,14)");
      const fill = e.kind==="button" ? "rgba(91,156,255,.18)" : e.kind==="link" ? "transparent" : "#15181f";
      if(e.kind!=="link") g.append("rect").attr("x",e.x).attr("y",e.y).attr("width",e.w).attr("height",e.h).attr("rx",5).attr("fill",fill).attr("stroke",e.kind==="button"?C.blue:C.line);
      g.append("text").attr("x",e.x+(e.kind==="button"?e.w/2:8)).attr("y",e.y+e.h/2+4)
        .attr("text-anchor",e.kind==="button"?"middle":"start").attr("font-size",10)
        .attr("fill",e.kind==="link"?C.blue:e.kind==="textbox"?C.dim:C.ink).text(e.label);
      if(marks){
        g.append("rect").attr("x",e.x-4).attr("y",e.y-4).attr("width",e.w+8).attr("height",e.h+8).attr("rx",4)
          .attr("fill","none").attr("stroke",C.accent).attr("stroke-width",1).attr("stroke-dasharray","3,2");
        g.append("rect").attr("x",e.x-4).attr("y",e.y-14).attr("width",14).attr("height",12).attr("rx",2).attr("fill",C.accent);
        g.append("text").attr("x",e.x+3).attr("y",e.y-4.5).attr("text-anchor","middle").attr("font-size",8.5).attr("fill","#12151c").text(e.n);
      }
    });
  }
  function textPanel(title,lines,colorFn){
    svg.append("rect").attr("x",328).attr("y",14).attr("width",298).attr("height",232).attr("rx",8).attr("fill","#15181f").attr("stroke",C.line);
    svg.append("text").attr("x",340).attr("y",30).attr("font-size",9.5).attr("fill",C.muted).text(title);
    lines.forEach((ln,i)=>{
      svg.append("text").attr("x",340).attr("y",48+i*15.5).attr("font-size",9).attr("font-family",MONO)
        .attr("fill",colorFn?colorFn(ln,i):C.ink).text(ln);
    });
  }
  const views={
    pixels(){
      screenPanel(false);
      textPanel("what the model receives",[
        "[image  1280 × 800 → 1024 × 640]",
        "≈ 1,400 visual tokens",
        "",
        "no element list, no labels,",
        "no bounding boxes —",
        "everything must be read",
        "out of the pixels.",
        "",
        "action it must emit:",
        "  click(x=142, y=91)",
      ],(ln,i)=>i>=8?C.accent:C.ink);
      return "Pixels: universal — works on any app or remote desktop — but the model has to regress a coordinate, and small text disappears when the frame is downscaled.";
    },
    axtree(){
      screenPanel(false);
      textPanel("accessibility tree (pruned)",[
        "document \"notes · search\"",
        "  textbox  name=\"Search\"  focused",
        "  button   name=\"Go\"",
        "  link     name=\"Interactive notes…\"",
        "  link     name=\"Study plan (draft)\"",
        "  checkbox name=\"Only mine\" checked=false",
        "",
        "roles · names · states · boxes",
        "",
        "action it must emit:",
        "  click(role=button, name=\"Go\")",
      ],(ln,i)=>i>=9?C.good:(i===7?C.muted:C.ink));
      return "Accessibility tree: roles, names, and states with exact boxes — no coordinate error at all, but only as complete as the app's own accessibility support.";
    },
    dom(){
      screenPanel(false);
      textPanel("pruned DOM (interactive nodes)",[
        "<input id=q  type=text  aria-label=Search>",
        "<button class=go>Go</button>",
        "<a href=/index>Interactive notes — index</a>",
        "<a href=/plan>Study plan (draft)</a>",
        "<input id=mine type=checkbox>",
        "",
        "unpruned this page is ~12k tokens;",
        "styles, scripts and hidden nodes",
        "dominate and must be stripped.",
        "",
        "action it must emit:",
        "  click(selector=\"button.go\")",
      ],(ln,i)=>i>=10?C.blue:(i>=6&&i<=8?C.muted:C.ink));
      return "DOM: exact selectors and hidden attributes, web only — and by far the most verbose. Pruning to visible, interactive nodes is mandatory, and layout is invisible in it.";
    },
    som(){
      screenPanel(true);
      textPanel("annotated image + legend",[
        "[image with numbered boxes]",
        "",
        "1  textbox   \"Search\"",
        "2  button    \"Go\"",
        "3  link      \"Interactive notes…\"",
        "4  link      \"Study plan (draft)\"",
        "5  checkbox  \"Only mine\"",
        "",
        "layout stays visible, the action",
        "space is discrete and checkable.",
        "",
        "action it must emit:",
        "  click(elem=2)",
      ],(ln,i)=>i>=11?C.accent:(i>=8&&i<=9?C.muted:C.ink));
      return "Set-of-marks: detect the interactive elements, draw numbered boxes, pass image + legend. The model picks an index — validatable before execution — while keeping the visual layout.";
    },
  };
  function draw(){
    svg.selectAll("*").remove();
    const k=d3.select("#obs-sel").property("value");
    d3.select("#obs-read").html(views[k]());
  }
  d3.select("#obs-sel").on("change",draw);
  draw();
})();
