"""A self-contained HTML map (Leaflet + OpenStreetMap tiles). No Python dependencies."""
import json

import numpy as np

LAYERS = [  # key, label, colour, weight, dashed
    ("raw", "Raw GPS from the phone", "#eb6834", 0, False),
    ("rts", "Cleaned GPS track", "#1baf7a", 3, False),
    ("ins", "micronav (GPS + motion sensors)", "#2a78d6", 3, False),
    ("matched", "Matched to roads", "#6b4fbb", 4, False),
    ("outage_kf", "GPS switched off: GPS alone", "#c23b22", 2, True),
    ("outage_ins", "GPS switched off: micronav", "#0b3d91", 2, True),
]


def write_html(result, path, title="micronav", intro=None):
    """intro: optional HTML shown in a box at the top left (e.g. what the viewer is looking at)."""
    layers = []
    for key, label, colour, weight, dashed in LAYERS:
        pts = result.get(key)
        if pts is None:
            continue
        pts = np.asarray(pts)
        pts = pts[np.isfinite(pts).all(1)]
        layers.append(dict(label=label, colour=colour, weight=weight, dashed=dashed,
                           pts=np.round(pts, 6).tolist(), on=key not in ("outage_kf", "outage_ins")))
    stats = result.get("stats", {})
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{height:100%;margin:0}} .box{{background:#fff;padding:8px 10px;font:13px/1.35 system-ui;border-radius:6px;max-width:320px}} @media (max-width:700px){{.box{{font-size:11.5px;max-width:62vw;padding:6px 8px}}}}</style>
</head><body><div id="map"></div><script>
const layers = {json.dumps(layers)};
const stats = {json.dumps(stats)};
const intro = {json.dumps(intro)};
const map = L.map('map');
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'}}).addTo(map);
const ctl = L.control.layers(null, null, {{collapsed: window.innerWidth < 700}}).addTo(map);
let bounds = null;
for (const l of layers) {{ const b = L.latLngBounds(l.pts); bounds = bounds ? bounds.extend(b) : b; }}
map.fitBounds(bounds);   // the map needs a view before vector layers are added
for (const l of layers) {{
  const g = l.weight === 0
    ? L.layerGroup(l.pts.map(p => L.circleMarker(p, {{radius: 2, color: l.colour, weight: 0, fillOpacity: .8}})))
    : L.polyline(l.pts, {{color: l.colour, weight: l.weight, dashArray: l.dashed ? '6 6' : null}});
  if (l.on) g.addTo(map);
  ctl.addOverlay(g, `<span style="color:${{l.colour}}">&#9632;</span> ${{l.label}}`);
}}
// a map created while hidden (zero size) never zooms in: refit once it gets a size
let fitted = false;
new ResizeObserver(() => {{ if (!fitted && map.getSize().y > 0) {{
  fitted = true; map.invalidateSize(); map.fitBounds(bounds); }} }}).observe(document.getElementById('map'));
if (intro) {{
  const ib = L.control({{position: 'topleft'}});
  ib.onAdd = () => {{ const d = L.DomUtil.create('div', 'box'); d.innerHTML = intro; return d; }};
  ib.addTo(map);
}}
if (stats.outage) {{
  const box = L.control({{position: 'bottomleft'}});
  box.onAdd = () => {{ const d = L.DomUtil.create('div', 'box');
    d.innerHTML = `<b>GPS hidden ${{stats.outage.gap}} s of every ${{stats.outage.every}} s</b><br>` +
      `GPS alone: ${{stats.outage.gps_only.toFixed(1)}} m off (median)<br>` +
      `micronav: ${{stats.outage.fused.toFixed(1)}} m off`; return d; }};
  box.addTo(map);
}}
</script></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
