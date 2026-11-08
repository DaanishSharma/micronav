"""The road as a measurement (chapter 8).

Map matching (chapter 5) runs *after* the filter: it moves the track onto roads.
Here the map goes *into* the filter. If the car is on a road segment, its distance
across that road is ~0 (a few metres for lane width). That is a 1-number measurement
along the road's normal, and it says nothing about where along the road the car is.

Which segment? The nearest one whose direction agrees with the car's velocity, so a
crossing road at a junction isn't picked by mistake.
"""
import numpy as np


def road_line(graph, p, v, radius=25.0, min_speed=2.0, min_cos=0.9):
    """(point on road, road direction) for the best segment near p, or None."""
    sp = np.linalg.norm(v[:2])
    if sp < min_speed:
        return None
    seg, f, q, d = graph.nearest(p, radius, k=6)
    if not len(seg):
        return None
    dirs = graph.direction(seg)
    cos = np.abs(dirs @ (v[:2] / sp))
    good = np.where(cos > min_cos)[0]
    if not len(good):
        return None
    i = good[np.argmin(d[good])]
    return q[i], dirs[i]
