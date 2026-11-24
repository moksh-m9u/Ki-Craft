"""The interactive pipeline diagram on the home page.

Mermaid (a JavaScript library that draws diagrams from text) renders the
flowchart inside an iframe, with zoom, pan and a note per clicked node.
"""

from __future__ import annotations

import json

# Shown when a node in the diagram is clicked.
MERMAID_NODE_NOTES: dict[str, str] = {
    "PDF": "A datasheet or application note.",
    "EXTRACT": "<code>unstructured.partition_pdf(strategy=\"hi_res\")</code> splits the PDF "
    "into typed elements: text, tables (as HTML) and figures. Page headers, footers and "
    "vendor logos are dropped.",
    "CHUNK": "<code>chunk_by_title</code> groups elements into sections of at most 3000 "
    "characters, so a figure stays with the text that explains it.",
    "SUMMARY": "A vision model reads each section's text, tables and figures and writes a "
    "plain-text summary: GLM-4.5V hosted, or Qwen3.5-4B on a Mac.",
    "EMBED": "IBM watsonx Granite turns each summary into a 768-number vector.",
    "STORE": "Postgres with pgvector keeps the summary vector, and alongside it the original text, tables "
    "and figure paths. Figures are JPEG files served by the API.",
    "QUESTION": "Your question, from the Query page or any client of the API.",
    "QEMBED": "The question is embedded with the same Granite model.",
    "SEARCH": "MMR picks 5 sections from the 20 nearest: relevant, but not five copies of "
    "the same paragraph.",
    "PROMPT": "The model gets each section's original text and tables, labelled C1, C2, ..., "
    "and up to 3 figures, labelled IMG1, IMG2, ...",
    "MODEL": "Gemini 3.5 Flash-Lite by default, or any listed model with your own key. "
    "The answer streams to the page and cites [C1] and [IMG1].",
    "CHECK": "The server reads every bracketed label in the answer and keeps those the "
    "model was given; any other label is reported as unknown.",
}


# The current pipeline, rendered client-side with mermaid.js inside a
# sandboxed iframe. Keep it in step with the README's "How it works".
MERMAID_DIAGRAM = """
flowchart TB
    subgraph IN["Once per datasheet"]
        direction LR
        PDF[("PDF")]
        EXTRACT["unstructured hi_res:<br/>text, tables, figures"]
        CHUNK["Sections by heading<br/>max 3,000 chars"]
        SUMMARY["Vision model<br/>summary per section"]
        EMBED["watsonx Granite<br/>768-number vector"]
        PDF --> EXTRACT --> CHUNK --> SUMMARY --> EMBED
    end

    STORE[("Postgres + pgvector: vectors,<br/>original text, tables, figure paths")]

    subgraph Q["For every question"]
        direction LR
        QUESTION["Question"]
        QEMBED["Embed question<br/>(same Granite)"]
        SEARCH["MMR: 20 nearest,<br/>keep 5"]
        PROMPT["Prompt: C1-C5 text + tables,<br/>IMG1-IMG3 figures"]
        MODEL["Gemini 3.5 Flash-Lite<br/>or your model"]
        CHECK["Check citations<br/>against what was sent"]
        QUESTION --> QEMBED --> SEARCH --> PROMPT --> MODEL --> CHECK
    end

    %% Linking the groups, not nodes inside them, keeps each group left to right:
    %% Mermaid drops a group's own direction when one of its nodes links outside it.
    IN --> STORE
    STORE --> Q

    classDef ingest fill:#1a73e8,color:#fff,stroke:#0b4fa8
    classDef store fill:#34a853,color:#fff,stroke:#1d6b32
    classDef query fill:#673ab7,color:#fff,stroke:#3f1d80
    class PDF,EXTRACT,CHUNK,SUMMARY,EMBED ingest
    class STORE store
    class QUESTION,QEMBED,SEARCH,PROMPT,MODEL,CHECK query
"""


_MERMAID_VIEWER = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<style>
  :root {
    --bg: #ffffff; --panel: #f7f8fa; --fg: #1c1c1c; --muted: #6b7280;
    --line: #e3e6ec; --accent: #ff4b4b;
  }
  html.dark {
    --bg: #0e1117; --panel: #171c26; --fg: #e8eaed; --muted: #98a2b3;
    --line: #272d38; --accent: #ff6b6b;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: transparent; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  #bar { display: flex; align-items: center; gap: 6px; padding: 2px 0 8px 0; }
  #bar button {
    font: inherit; font-size: 12px; font-weight: 600; line-height: 1;
    color: var(--fg); background: var(--panel); cursor: pointer;
    border: 1px solid var(--line); border-radius: 7px; padding: 6px 10px;
  }
  #bar button:hover { border-color: var(--accent); color: var(--accent); }
  #hint { margin-left: 4px; font-size: 11px; color: var(--muted); }
  #zoom {
    margin-left: auto; font-size: 11px; color: var(--muted);
    font-variant-numeric: tabular-nums;
  }
  #viewport {
    position: relative; height: 510px; overflow: hidden; cursor: grab;
    background: var(--bg); border: 1px solid var(--line); border-radius: 12px;
  }
  #viewport.dragging { cursor: grabbing; }
  #canvas {
    position: absolute; top: 0; left: 0; width: max-content;
    transform-origin: 0 0;
  }
  /* Mermaid lays the graph out against the width of this element.  Left to
     shrink-to-fit it collapses, every label wraps one word per line and the
     drawing comes out thousands of pixels tall, so give it a real canvas;
     the script below then matches it to the frame. */
  #canvas .mermaid { display: block; width: 1150px; }
  #canvas svg { max-width: none !important; height: auto !important; }
  #canvas g.node { cursor: pointer; }
  #canvas g.mmr-sel > rect, #canvas g.mmr-sel > polygon,
  #canvas g.mmr-sel > circle, #canvas g.mmr-sel > path {
    stroke: var(--accent) !important; stroke-width: 3px !important;
  }
  #note {
    position: absolute; left: 12px; right: 12px; bottom: 12px; display: none;
    padding: 10px 12px; border-radius: 10px; font-size: 12px; line-height: 1.55;
    color: var(--fg); background: var(--panel); border: 1px solid var(--line);
    box-shadow: 0 6px 20px rgba(0,0,0,.18);
  }
  #note.on { display: block; cursor: pointer; }
  #note b { color: var(--accent); }
  #note code { font-size: 11px; }
</style>
</head>
<body>
<div id="bar">
  <button id="zoomOut" title="Zoom out">&minus;</button>
  <button id="zoomIn" title="Zoom in">+</button>
  <button id="fit" title="Fit the whole diagram">Fit</button>
  <button id="actual" title="Render at 100%">1:1</button>
  <span id="hint">drag to pan &middot; double-click to fit &middot; click a node</span>
  <span id="zoom">100%</span>
</div>
<div id="viewport">
  <div id="canvas"><div class="mermaid">
__DIAGRAM__
  </div></div>
  <div id="note"></div>
</div>
<script>
(function () {
  var NOTES = __NOTES__;
  var vp = document.getElementById("viewport");
  var canvas = document.getElementById("canvas");
  var zoomLabel = document.getElementById("zoom");
  var note = document.getElementById("note");
  var svg = null, scale = 1, tx = 0, ty = 0, moved = false;
  var nat = { w: 0, h: 0 };
  var MIN = 0.05, MAX = 8;

  function isDark() {
    try {
      var w = window.parent, d = w.document;
      var els = [
        d.querySelector('[data-testid="stAppViewContainer"]'),
        d.querySelector(".stApp"),
        d.body,
        d.documentElement
      ];
      for (var i = 0; i < els.length; i++) {
        if (!els[i]) continue;
        var c = w.getComputedStyle(els[i]).backgroundColor || "";
        var m = c.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)(?:,\\s*([\\d.]+))?\\)/);
        if (!m) continue;
        if (m[4] !== undefined && parseFloat(m[4]) === 0) continue;
        return (0.299 * m[1] + 0.587 * m[2] + 0.114 * m[3]) < 128;
      }
    } catch (e) { /* parent not reachable: assume the light theme */ }
    return false;
  }

  var dark = isDark();
  if (dark) document.documentElement.classList.add("dark");

  function apply() {
    canvas.style.transform =
      "translate(" + tx + "px," + ty + "px) scale(" + scale + ")";
    zoomLabel.textContent = Math.round(scale * 100) + "%";
  }

  function natural() {
    return nat;
  }

  /* Pin the SVG to its intrinsic pixel size.  Mermaid emits width="100%" plus a
     max-width style, which resolves against the container and can collapse the
     drawing; forcing the viewBox size makes the scale factor predictable. */
  function measure() {
    if (!svg) return { w: 0, h: 0 };
    var vb = svg.viewBox && svg.viewBox.baseVal;
    var w = vb && vb.width, h = vb && vb.height;
    if (!w || !h) {
      var r = svg.getBoundingClientRect();
      w = r.width / scale;
      h = r.height / scale;
    }
    if (w > 0 && h > 0) {
      svg.removeAttribute("width");
      svg.removeAttribute("height");
      svg.style.width = w + "px";
      svg.style.height = h + "px";
    }
    nat = { w: w, h: h };
    return nat;
  }

  function center(s) {
    var r = vp.getBoundingClientRect();
    var n = natural();
    tx = (r.width - n.w * s) / 2;
    ty = (r.height - n.h * s) / 2;
    scale = s;
    apply();
  }

  function fit() {
    var r = vp.getBoundingClientRect();
    var n = natural();
    if (!n.w || !n.h) return;
    center(Math.min((r.width - 24) / n.w, (r.height - 24) / n.h, MAX));
  }

  function zoomBy(factor) {
    var next = Math.min(MAX, Math.max(MIN, scale * factor));
    var r = vp.getBoundingClientRect();
    var cx = r.width / 2, cy = r.height / 2;
    tx = cx - (cx - tx) * (next / scale);
    ty = cy - (cy - ty) * (next / scale);
    scale = next;
    apply();
  }

  function select(g) {
    var prev = canvas.querySelector("g.mmr-sel");
    if (prev) prev.classList.remove("mmr-sel");
    if (!g) { note.classList.remove("on"); return; }
    g.classList.add("mmr-sel");
    var id = (g.id || "").split("-")[1] || "";
    var text = NOTES[id];
    if (!text) { note.classList.remove("on"); return; }
    note.innerHTML = "<b>" + id + "</b> &mdash; " + text;
    note.classList.add("on");
  }

  function bind() {
    document.getElementById("zoomIn").onclick = function () { zoomBy(1.25); };
    document.getElementById("zoomOut").onclick = function () { zoomBy(0.8); };
    document.getElementById("fit").onclick = fit;
    document.getElementById("actual").onclick = function () { center(1); };
    note.onclick = function () { select(null); };

    var drag = false, sx = 0, sy = 0, ox = 0, oy = 0;
    vp.addEventListener("pointerdown", function (e) {
      drag = true; moved = false;
      sx = e.clientX; sy = e.clientY; ox = tx; oy = ty;
      vp.classList.add("dragging");
      try { vp.setPointerCapture(e.pointerId); } catch (err) {}
    });
    vp.addEventListener("pointermove", function (e) {
      if (!drag) return;
      var dx = e.clientX - sx, dy = e.clientY - sy;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) moved = true;
      tx = ox + dx; ty = oy + dy;
      apply();
    });
    ["pointerup", "pointercancel", "pointerleave"].forEach(function (ev) {
      vp.addEventListener(ev, function () {
        drag = false; vp.classList.remove("dragging");
      });
    });
    vp.addEventListener("dblclick", fit);

    Array.prototype.forEach.call(canvas.querySelectorAll("g.node"), function (g) {
      g.addEventListener("click", function () { if (!moved) select(g); });
    });
    window.addEventListener("resize", fit);
  }

  function start() {
    svg = canvas.querySelector("svg");
    bind();

    /* Mermaid finishes measuring labels asynchronously, so the intrinsic size
       can change a frame or two later.  Re-measure, then open framed. */
    function place() {
      measure();
      fit();
    }
    requestAnimationFrame(place);
    setTimeout(place, 120);
  }

  if (window.mermaid) {
    /* Draw the graph at the width of the frame so the intrinsic size of the
       diagram matches the space available, instead of sprawling into a strip
       that then has to be shrunk. */
    var box = document.querySelector(".mermaid");
    if (box) {
      var avail = vp.getBoundingClientRect().width;
      if (avail > 320) box.style.width = avail + "px";
    }

    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: "loose",
      theme: "base",
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      themeVariables: dark ? {
        fontSize: "15px",
        background: "#0e1117",
        primaryColor: "#1b2130",
        primaryTextColor: "#e8eaed",
        lineColor: "#7d8797",
        clusterBkg: "#151a24",
        clusterBorder: "#2a3140",
        edgeLabelBackground: "#0e1117"
      } : {
        fontSize: "15px",
        background: "#ffffff",
        primaryColor: "#f4f6f9",
        primaryTextColor: "#1c1c1c",
        lineColor: "#8a94a6",
        clusterBkg: "#fbfcfd",
        clusterBorder: "#dfe4ec",
        edgeLabelBackground: "#ffffff"
      },
      flowchart: {
        htmlLabels: true, curve: "basis", nodeSpacing: 34, rankSpacing: 46
      }
    });
    window.mermaid.run({ querySelector: ".mermaid" })
      .then(start)
      .catch(function () { start(); });
  }
})();
</script>
</body>
</html>
"""


def mermaid_iframe(diagram: str) -> str:
    """Build the interactive mermaid viewer document.

    The SVG is rendered at its natural size and scaled with a CSS transform
    inside a fixed viewport, so the diagram fills the frame instead of being
    shrunk by ``max-width``.  Zoom, fit, drag-to-pan and clickable nodes are all
    handled in the iframe; the theme is inherited from the Streamlit app.
    """
    return (
        _MERMAID_VIEWER.replace("__DIAGRAM__", diagram.strip()).replace(
            "__NOTES__", json.dumps(MERMAID_NODE_NOTES)
        )
    )


