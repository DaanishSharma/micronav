"""Road data from OpenStreetMap's Overpass API (chapter 5).

Free, but often busy: tries several servers and caches answers in ~/.cache/micronav.
Map data (c) OpenStreetMap contributors, ODbL.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
            "https://overpass.private.coffee/api/interpreter"]
DRIVE = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service|road|"
         "motorway_link|trunk_link|primary_link|secondary_link|tertiary_link")
USER_AGENT = "micronav (+https://github.com/DaanishSharma/micronav)"
CACHE = Path.home() / ".cache" / "micronav"


def fetch_roads(south, west, north, east, drivable=True, cache_dir=CACHE, retries=2):
    """Roads in a lat/lon box as Overpass JSON, or None if every server failed."""
    s, w = np.floor(south * 500) / 500, np.floor(west * 500) / 500      # ~200 m grid so the cache hits
    n, e = np.ceil(north * 500) / 500, np.ceil(east * 500) / 500
    f = Path(cache_dir) / f"roads_{'drive' if drivable else 'all'}_{s:.3f}_{w:.3f}_{n:.3f}_{e:.3f}.json"
    if f.exists():
        return json.loads(f.read_text())
    sel = f'["highway"~"^({DRIVE})$"]' if drivable else '["highway"]'
    q = f"[out:json][timeout:90];way{sel}({s},{w},{n},{e});out geom;"
    for _ in range(retries):
        for url in OVERPASS:
            try:
                req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(),
                                             headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=120) as r:
                    d = json.loads(r.read())
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(d))
                return d
            except (OSError, ValueError):
                continue
        time.sleep(5)
    return None
