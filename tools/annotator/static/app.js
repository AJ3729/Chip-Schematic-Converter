/* CGHD netlist annotator (task C1).
 *
 * Keyboard-driven, autosaving, resumable. Records the AS-DRAWN topology, with
 * would-be repairs kept in a separate field.
 *
 * The tool deliberately has no access to pipeline output. There is no endpoint
 * that serves detections or predicted nets, so it cannot pre-fill or suggest
 * one even by accident -- circular evaluation is prevented structurally rather
 * than by discipline.
 */
"use strict";

const CLASSES = ["Resistor","Capacitor","Inductor","Diode","Zener Diode","GND",
  "V-DC","V-DC (one port)","V-AC","I-DC","I-AC","MOSFET-N","MOSFET-P",
  "BJT-NPN","BJT-PNP","Op-Amp","Wire Crossover"];

// Terminal counts and port names must match the pipeline's class table.
const PORTS = {
  "Resistor":["terminal 0","terminal 1"], "Capacitor":["terminal 0","terminal 1"],
  "Inductor":["terminal 0","terminal 1"], "Diode":["anode","cathode"],
  "Zener Diode":["anode","cathode"], "GND":["terminal 0"],
  "V-DC":["positive","negative"], "V-DC (one port)":["terminal 0"],
  "V-AC":["positive","negative"], "I-DC":["positive","negative"],
  "I-AC":["positive","negative"], "MOSFET-N":["drain","gate","source"],
  "MOSFET-P":["drain","gate","source"], "BJT-NPN":["collector","base","emitter"],
  "BJT-PNP":["collector","base","emitter"], "Op-Amp":["in+","in-","out"],
  "Wire Crossover":[]
};

const SITE_KINDS = {j:"junction", k:"crossing", e:"edge_group", o:"none"};
const SITE_COLOR = {junction:"#3fbf6f", crossing:"#e0574a",
                    edge_group:"#e0a33e", none:"#8b93a7"};

let items = [], idx = 0, tutorial = false, blind = false;
let state = null;              // the record being built
let mode = "box";
let curNet = "n1";
let view = {z:1, x:0, y:0};
let t0 = 0, elapsed = 0;

const $ = s => document.querySelector(s);
const canvas = $("#canvas"), img = $("#img");

const SOURCE = {
  cghd: "cghd_geometry+manual_topology",
  "digitize-hcd": "digitize_hcd_tutorial",
  "blind-review": "digitize_hcd_blind_second_pass"
};

function blankRecord(item){
  return {
    schema_version: 1,
    image: item.image + ".jpg",
    source: SOURCE[item.corpus] || "manual",
    drafter: item.drafter,
    drawing_group: item.id,
    picture: 1,
    annotator: "author",
    annotation_seconds: 0,
    pass: 1,
    components: [],
    sites: [],
    interventions: [],
    notes: ""
  };
}

async function boot(){
  const r = await (await fetch("/api/items")).json();
  items = r.items; tutorial = r.tutorial; blind = !!r.blind;
  $("#cls").innerHTML = CLASSES.map(c=>`<option>${c}</option>`).join("");
  if (tutorial) $("#tutorialbox").style.display = "block";
  await load(0);
  wire();
}

async function load(i){
  if (i < 0 || i >= items.length) return;
  if (state) await saveDraft();
  idx = i;
  const item = items[idx];
  const d = await (await fetch("/api/draft?id="+encodeURIComponent(item.id))).json();
  state = (d && d.components) ? d : blankRecord(item);
  elapsed = state.annotation_seconds || 0;
  t0 = Date.now();
  img.src = "/img/" + item.image;
  view = {z:1, x:0, y:0}; applyView();
  $("#title").textContent = `${item.id}`;
  $("#sub").textContent =
    `${idx+1} / ${items.length}  ·  ${item.corpus}` +
    (item.drafter!=null ? `  ·  drafter ${item.drafter}` : "") +
    (item.n_captures>1 ? `  ·  ${item.n_captures} captures share this netlist` : "");
  $("#notes").value = state.notes || "";
  $("#interv").value = (state.interventions||[])
      .map(x=>`${x.type}: ${x.target||""} — ${x.note||""}`).join("\n");
  $("#banner").innerHTML = item.tutorial
    ? `<div class="okbox">Tutorial circuit — ground truth exists. Annotate it
       yourself first, then reveal to calibrate.</div>`
    : `<div class="warnbox">Trace from the drawing only. Record what is
       <b>visibly drawn</b>, not a circuit that would be easier to simulate.</div>`;
  render();
}

/* ---------------------------------------------------------------- view */
function applyView(){
  canvas.style.transform = `translate(${view.x}px,${view.y}px) scale(${view.z})`;
}
$("#stage").addEventListener("wheel", e=>{
  e.preventDefault();
  const k = e.deltaY < 0 ? 1.12 : 1/1.12;
  const r = $("#stage").getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  view.x = mx - (mx - view.x) * k;
  view.y = my - (my - view.y) * k;
  view.z *= k; applyView();
}, {passive:false});

let drag = null;
$("#stage").addEventListener("mousedown", e=>{
  if (e.button === 1 || e.shiftKey) drag = {x:e.clientX-view.x, y:e.clientY-view.y};
});
addEventListener("mousemove", e=>{
  if (!drag) return;
  view.x = e.clientX-drag.x; view.y = e.clientY-drag.y; applyView();
});
addEventListener("mouseup", ()=> drag = null);

/* ------------------------------------------------------------- placing */
const at = e => {
  const r = img.getBoundingClientRect();
  return [(e.clientX - r.left) / view.z, (e.clientY - r.top) / view.z];
};

canvas.addEventListener("click", e=>{
  // A drag emits mouseup and THEN click. Box mode switches to terminal mode on
  // mouseup, so without this the trailing click lands as a terminal at the
  // corner of the box you just drew -- every component silently gains a phantom
  // terminal, and the count only looks wrong two components later.
  if (swallowClick) { swallowClick = false; return; }
  if (e.shiftKey || mode === "box") return;
  const [x,y] = at(e);
  if (mode === "terminal") addTerminal(x, y);
  else addSite(x, y);
  render();
});

/* Box mode: drag out the component's bounding box. The box is not decoration --
 * components are paired between the two annotations GEOMETRICALLY (Hungarian
 * assignment on IoU, see scripts/compare_annotations.py), so a component with no
 * box cannot be paired at all and reads as one missing plus one extra. */
let boxDrag = null, swallowClick = false;
canvas.addEventListener("mousedown", e=>{
  if (mode !== "box" || e.shiftKey || e.button !== 0) return;
  e.preventDefault();
  const [x,y] = at(e);
  boxDrag = {x0:x, y0:y, x1:x, y1:y};
  drawBoxPreview();
});
addEventListener("mousemove", e=>{
  if (!boxDrag) return;
  const [x,y] = at(e);
  boxDrag.x1 = x; boxDrag.y1 = y;
  drawBoxPreview();
});
addEventListener("mouseup", ()=>{
  if (!boxDrag) return;
  const b = boxDrag; boxDrag = null;
  const el = $("#boxpreview"); if (el) el.remove();
  swallowClick = true;
  const x = Math.min(b.x0,b.x1), y = Math.min(b.y0,b.y1);
  const w = Math.abs(b.x1-b.x0), h = Math.abs(b.y1-b.y0);
  if (w < 6 || h < 6) { render(); return; }   // a stray click, not a box
  addComponent(x, y, w, h);
  mode = "terminal";                          // boxed it; now place its terminals
  render();
});

function drawBoxPreview(){
  let el = $("#boxpreview");
  if (!el){
    el = document.createElement("div");
    el.id = "boxpreview"; el.className = "cbox preview";
    canvas.appendChild(el);
  }
  const b = boxDrag;
  el.style.left = Math.min(b.x0,b.x1)+"px";
  el.style.top = Math.min(b.y0,b.y1)+"px";
  el.style.width = Math.abs(b.x1-b.x0)+"px";
  el.style.height = Math.abs(b.y1-b.y0)+"px";
}

function addComponent(x,y,w,h){
  // Stored CENTRE-based (cx, cy, w, h). Every scorer in this repo -- both
  // benchmark.iou_center and compare_annotations.geometric_pairs -- reads boxes
  // that way, and CGHD's own annotations use it too. Storing the drag's
  // top-left corner instead would offset every box by half its own size, which
  // drops IoU below the 0.3 pairing threshold for exactly the small components
  // that are already hardest, and shows up as component disagreement rather
  // than as a bug.
  state.components.push({
    id: state.components.length, class: $("#cls").value, terminals: [],
    bbox: [Math.round(x + w/2), Math.round(y + h/2), Math.round(w), Math.round(h)],
    allow_self_short: false
  });
}

function addTerminal(x,y){
  const cls = $("#cls").value;
  let c = state.components[state.components.length-1];
  const need = PORTS[cls].length;
  if (!c || c.terminals.length >= PORTS[c.class].length){
    // No boxed component is waiting for terminals. Create one without a box and
    // let render() flag it, rather than silently dropping the terminal.
    c = {id: state.components.length, class: cls, terminals: [],
         bbox: null, allow_self_short:false};
    state.components.push(c);
  }
  c.terminals.push({index:c.terminals.length, net:curNet, xy:[Math.round(x),Math.round(y)]});
  if (c.terminals.length === PORTS[c.class].length) beep();
}

function addSite(x,y){
  state.sites.push({id: state.sites.length, xy:[Math.round(x),Math.round(y)],
                    kind: "junction"});
}

function beep(){ // visual, not audible
  $("#hint").style.color = "#3fbf6f";
  setTimeout(()=> $("#hint").style.color = "", 250);
}

/* -------------------------------------------------------------- render */
function render(){
  [...canvas.querySelectorAll(".sitedot,.termdot,.cbox:not(.preview)")]
    .forEach(n=>n.remove());
  state.components.forEach(c=>{
    if (!c.bbox) return;
    const d = document.createElement("div");
    d.className = "cbox";
    d.style.cssText += `left:${c.bbox[0] - c.bbox[2]/2}px;` +
                       `top:${c.bbox[1] - c.bbox[3]/2}px;` +
                       `width:${c.bbox[2]}px;height:${c.bbox[3]}px`;
    d.title = `#${c.id} ${c.class}`;
    canvas.appendChild(d);
  });
  state.sites.forEach(s=>{
    const d = document.createElement("div");
    d.className = "sitedot";
    d.style.cssText += `left:${s.xy[0]}px;top:${s.xy[1]}px;background:${SITE_COLOR[s.kind]}`;
    d.title = `site ${s.id}: ${s.kind}`;
    canvas.appendChild(d);
  });
  state.components.forEach(c=> c.terminals.forEach(t=>{
    const d = document.createElement("div");
    d.className = "termdot";
    d.style.cssText += `left:${t.xy[0]}px;top:${t.xy[1]}px;background:#4da3ff`;
    d.title = `${c.class} ${PORTS[c.class][t.index]||t.index} → ${t.net}`;
    canvas.appendChild(d);
  }));

  $("#comps").innerHTML = "<tr><th>#</th><th>class</th><th>terminals</th></tr>" +
    state.components.map(c=>{
      const need = PORTS[c.class].length;
      const bad = c.terminals.length !== need;
      return `<tr><td>${c.id}${c.bbox ? "" :
                 ' <span style="color:var(--bad)" title="no bounding box">▢</span>'}` +
        `</td><td>${c.class}</td><td>` +
        c.terminals.map(t=>`<span class="net">${t.net}</span>`).join(" ") +
        (bad ? ` <span style="color:var(--warn)">${c.terminals.length}/${need}</span>` : "") +
        `</td></tr>`;
    }).join("");

  const byKind = {};
  state.sites.forEach(s=> byKind[s.kind] = (byKind[s.kind]||0)+1);
  $("#sites").innerHTML = "<tr><th>kind</th><th>n</th></tr>" +
    Object.entries(byKind).map(([k,v])=>
      `<tr><td style="color:${SITE_COLOR[k]}">${k}</td><td>${v}</td></tr>`).join("");

  const noBox = state.components.filter(c=>!c.bbox).length;
  $("#mode").textContent = mode;
  $("#hint").textContent =
    `${mode} mode · net ${curNet} · ${state.components.length} components · ` +
    `${state.sites.length} sites` +
    (noBox ? ` · ${noBox} WITHOUT A BOX` : "");
  $("#timer").textContent = `${Math.round(secs())} s on this circuit`;
}

const secs = ()=> elapsed + (Date.now()-t0)/1000;

/* --------------------------------------------------------- persistence */
function collect(){
  state.notes = $("#notes").value;
  state.annotation_seconds = Math.round(secs());
  state.interventions = $("#interv").value.split("\n")
    .map(l=>l.trim()).filter(Boolean).map(l=>{
      const m = l.match(/^([a-z_]+)\s*:\s*([^—-]*)[—-]?\s*(.*)$/i);
      return m ? {type:m[1].trim(), target:m[2].trim()||null, note:m[3].trim()}
               : {type:"note", target:null, note:l};
    });
  return state;
}
async function saveDraft(){
  const rec = collect();
  await fetch("/api/draft", {method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({id: items[idx].id, ...rec})});
}
setInterval(()=>{ if (state) { saveDraft(); render(); } }, 10000);   // autosave

async function submit(){
  const rec = collect();
  const bad = rec.components.filter(c=>
    c.terminals.length !== PORTS[c.class].length);
  if (bad.length && !confirm(
      `${bad.length} component(s) have the wrong terminal count. Submit anyway?`))
    return;
  const noBox = rec.components.filter(c=>!c.bbox);
  if (noBox.length && !confirm(
      `${noBox.length} component(s) have no bounding box. Components are paired ` +
      `between annotations by box overlap, so an unboxed one cannot be matched ` +
      `and will read as a disagreement. Submit anyway?`))
    return;
  const r = await (await fetch("/api/submit", {method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({id: items[idx].id, record: rec})})).json();
  $("#banner").innerHTML =
    `<div class="okbox">submitted → ${r.written}. The sync script validates it.</div>`;
  setTimeout(()=> load(Math.min(idx+1, items.length-1)), 700);
}

/* ------------------------------------------------------------ bindings */
function wire(){
  $("#prev").onclick = ()=> load(idx-1);
  $("#next").onclick = ()=> load(idx+1);
  $("#save").onclick = saveDraft;
  $("#submit").onclick = submit;
  $("#newnet").onclick = ()=>{
    const used = new Set(); state.components.forEach(c=>c.terminals.forEach(t=>used.add(t.net)));
    let i=1; while (used.has("n"+i)) i++;
    curNet = "n"+i; $("#net").value = curNet; render();
  };
  $("#net").oninput = e=> { curNet = e.target.value.trim() || "n1"; render(); };
  // Boxing first and picking the class after is the natural order, so let a
  // class change re-label the component still waiting for its terminals.
  $("#cls").onchange = ()=>{
    const c = state.components[state.components.length-1];
    if (c && !c.terminals.length) { c.class = $("#cls").value; render(); }
  };
  $("#reveal").onclick = async ()=>{
    const t = await (await fetch("/api/truth?id="+encodeURIComponent(items[idx].image))).json();
    $("#truth").textContent = t.components
      ? t.components.map(c=>`${c.class}: ` +
          c.terminals.map(x=>x.net).join(",")).join("  |  ")
      : "no ground truth found";
  };

  addEventListener("keydown", e=>{
    if (["INPUT","TEXTAREA","SELECT"].includes(e.target.tagName)){
      if (e.key === "Escape") e.target.blur();
      return;
    }
    const k = e.key.toLowerCase();
    if (k === "b") { mode = "box"; render(); }
    else if (k === "t") { mode = "terminal"; render(); }
    else if (k === "i") { mode = "intersection"; render(); }
    else if (k in SITE_KINDS && state.sites.length){
      state.sites[state.sites.length-1].kind = SITE_KINDS[k]; render();
    }
    else if (k === "n") $("#newnet").click();
    else if (k === "0") { curNet = "0"; $("#net").value = "0"; render(); }
    else if (k === "x") {          // undo last placement
      if (mode === "intersection") state.sites.pop();
      else if (mode === "box") state.components.pop();
      else {
        const c = state.components[state.components.length-1];
        if (c){ c.terminals.pop(); if (!c.terminals.length) state.components.pop(); }
      }
      render();
    }
    else if (k === "c") $("#cls").focus();
    else if (k === "s") saveDraft();
    else if (k === "[") load(idx-1);
    else if (k === "]") load(idx+1);
    else if (k === "enter") submit();
    else if (k === "escape") { view={z:1,x:0,y:0}; applyView(); }
  });
}

boot();
