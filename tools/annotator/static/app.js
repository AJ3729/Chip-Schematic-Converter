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

let items = [], idx = 0, tutorial = false;
let state = null;              // the record being built
let mode = "terminal";
let curNet = "n1";
let view = {z:1, x:0, y:0};
let t0 = 0, elapsed = 0;

const $ = s => document.querySelector(s);
const canvas = $("#canvas"), img = $("#img");

function blankRecord(item){
  return {
    schema_version: 1,
    image: item.image + ".jpg",
    source: item.corpus === "cghd" ? "cghd_geometry+manual_topology"
                                   : "digitize_hcd_tutorial",
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
  items = r.items; tutorial = r.tutorial;
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
canvas.addEventListener("click", e=>{
  if (e.shiftKey) return;
  const r = img.getBoundingClientRect();
  const x = (e.clientX - r.left) / view.z, y = (e.clientY - r.top) / view.z;
  if (mode === "terminal") addTerminal(x, y);
  else addSite(x, y);
  render();
});

function addTerminal(x,y){
  const cls = $("#cls").value;
  let c = state.components[state.components.length-1];
  const need = PORTS[cls].length;
  if (!c || c.class !== cls || c.terminals.length >= PORTS[c.class].length){
    c = {id: state.components.length, class: cls, terminals: [],
         bbox: null, allow_self_short:false};
    state.components.push(c);
  }
  c.terminals.push({index:c.terminals.length, net:curNet, xy:[Math.round(x),Math.round(y)]});
  if (c.terminals.length === need) beep();
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
  [...canvas.querySelectorAll(".sitedot,.termdot")].forEach(n=>n.remove());
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
      return `<tr><td>${c.id}</td><td>${c.class}</td><td>` +
        c.terminals.map(t=>`<span class="net">${t.net}</span>`).join(" ") +
        (bad ? ` <span style="color:var(--warn)">${c.terminals.length}/${need}</span>` : "") +
        `</td></tr>`;
    }).join("");

  const byKind = {};
  state.sites.forEach(s=> byKind[s.kind] = (byKind[s.kind]||0)+1);
  $("#sites").innerHTML = "<tr><th>kind</th><th>n</th></tr>" +
    Object.entries(byKind).map(([k,v])=>
      `<tr><td style="color:${SITE_COLOR[k]}">${k}</td><td>${v}</td></tr>`).join("");

  $("#mode").textContent = mode;
  $("#hint").textContent =
    `${mode} mode · net ${curNet} · ${state.components.length} components · ` +
    `${state.sites.length} sites`;
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
    if (k === "t") { mode = "terminal"; render(); }
    else if (k === "i") { mode = "intersection"; render(); }
    else if (k in SITE_KINDS && state.sites.length){
      state.sites[state.sites.length-1].kind = SITE_KINDS[k]; render();
    }
    else if (k === "n") $("#newnet").click();
    else if (k === "0") { curNet = "0"; $("#net").value = "0"; render(); }
    else if (k === "x") {          // undo last placement
      if (mode === "intersection") state.sites.pop();
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
