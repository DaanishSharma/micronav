"""Roads as a graph (chapter 5): nodes are points in metres, edges are straight segments."""
import heapq

import numpy as np


class RoadGraph:
    def __init__(self, xy, segments):
        """xy: (n, 2) node positions in metres (east, north). segments: (m, 2) node index pairs."""
        self.xy = np.asarray(xy, float)
        seg = np.unique(np.sort(np.asarray(segments, int), axis=1), axis=0)
        seg = seg[seg[:, 0] != seg[:, 1]]
        self.u, self.v = seg[:, 0], seg[:, 1]
        self.A, self.B = self.xy[self.u], self.xy[self.v]
        self.len = np.maximum(np.linalg.norm(self.B - self.A, axis=1), 0.01)
        self.adj = [[] for _ in range(len(self.xy))]
        for a, b_, L in zip(self.u, self.v, self.len):
            self.adj[a].append((b_, L))
            self.adj[b_].append((a, L))

    @classmethod
    def from_osm(cls, osm, frame):
        """Build from an Overpass JSON answer ("out geom"), in the metres of a `LocalFrame`."""
        idx, latlon, seg = {}, [], []
        for way in osm.get("elements", []):
            if way.get("type") != "way" or len(way.get("nodes", [])) < 2:
                continue
            for nid, p in zip(way["nodes"], way["geometry"]):
                if nid not in idx:
                    idx[nid] = len(latlon)
                    latlon.append((p["lat"], p["lon"]))
            seg += [(idx[a], idx[b]) for a, b in zip(way["nodes"][:-1], way["nodes"][1:]) if a != b]
        ll = np.array(latlon)
        return cls(frame.to_enu(ll[:, 0], ll[:, 1])[:, :2], seg)

    @classmethod
    def from_polyline(cls, pts):
        pts = np.asarray(pts, float)[:, :2]
        return cls(pts, np.column_stack([np.arange(len(pts) - 1), np.arange(1, len(pts))]))

    def nearest(self, p, radius, k=8):
        """Closest point on every segment; keep the k closest within `radius`.
        Returns (segment ids, fraction along each, points, distances)."""
        AB = self.B - self.A
        f = np.clip(((p[:2] - self.A) * AB).sum(1) / self.len ** 2, 0, 1)
        q = self.A + f[:, None] * AB
        d = np.linalg.norm(q - p[:2], axis=1)
        i = np.where(d < radius)[0]
        i = i[np.argsort(d[i])][:k]
        return i, f[i], q[i], d[i]

    def direction(self, seg):
        return (self.B[seg] - self.A[seg]) / self.len[seg, None]

    def dijkstra(self, src, limit=np.inf):
        """Road distance from node `src` to every node within `limit` metres: {node: distance}."""
        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, a = heapq.heappop(heap)
            if d > dist.get(a, np.inf) or d > limit:
                continue
            for b_, L in self.adj[a]:
                nd = d + L
                if nd < dist.get(b_, np.inf) and nd <= limit:
                    dist[b_] = nd
                    heapq.heappush(heap, (nd, b_))
        return dist
