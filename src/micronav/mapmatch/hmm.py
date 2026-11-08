"""Map matching with a hidden Markov model (chapter 5; Newson & Krumm, 2009).

"Snap each fix to its nearest road" makes the track jump between parallel roads.
Instead, find the most likely *sequence* of road positions, using the fact that the car
has to drive along roads to get from one fix to the next.

- hidden state: a position on a road segment (candidates within ~50 m of the fix)
- emission score: how well a candidate explains the fix, -0.5 (d / sigma)^2
- transition score: -|road distance - straight-line distance| / beta. Driving along one
  road costs ~0; jumping to a parallel road needs a long detour, so a big penalty
- Viterbi: keep the best total score for every candidate, fix by fix, then follow the
  back-pointers from the end ("smoothed"). `current()` is the online best guess.
"""
import numpy as np


class MapMatcher:
    def __init__(self, graph, beta=5.0, k=8, radius=50.0, min_move=3.0):
        self.g, self.beta, self.k, self.radius, self.min_move = graph, beta, k, radius, min_move
        self.layers, self.last_p, self.breaks = [], None, 0

    def step(self, t, p, sigma=10.0):
        """Add one fix: p in metres (east, north), sigma its accuracy in metres."""
        p = np.asarray(p, float)[:2]
        if self.last_p is not None and np.linalg.norm(p - self.last_p) < self.min_move:
            return self.current()                      # barely moved: GPS jitter, skip
        sigma = float(np.clip(sigma, 4, 30)) if np.isfinite(sigma) else 10.0
        seg, f, q, d = self.g.nearest(p, max(self.radius, 3 * sigma), self.k)
        if not len(seg):
            return self.current()                      # no road nearby
        emis = -0.5 * (d / sigma) ** 2
        back, score = np.full(len(seg), -1), emis.copy()
        if self.layers:
            prev = self.layers[-1]
            straight = np.linalg.norm(p - self.last_p)
            route = self.route(prev, seg, f, limit=3 * straight + 200)
            tot = prev["score"][:, None] - np.abs(route - straight) / self.beta
            back = np.argmax(tot, axis=0)
            score = tot[back, np.arange(len(seg))] + emis
            if not np.isfinite(score).any():           # nothing reachable: start a new chain
                back[:], score = -1, emis.copy()
                self.breaks += 1
        self.layers.append(dict(t=t, seg=seg, f=f, q=q, score=score - score.max(), back=back, raw=p))
        self.last_p = p
        return self.current()

    def route(self, prev, seg, f, limit):
        """Road distance between every previous candidate and every new one."""
        g = self.g
        src = np.unique(np.r_[g.u[prev["seg"]], g.v[prev["seg"]]])
        D = {s: g.dijkstra(s, limit) for s in src}
        out = np.full((len(prev["seg"]), len(seg)), np.inf)
        for i, (sp, fp) in enumerate(zip(prev["seg"], prev["f"])):
            for j, (sc, fc) in enumerate(zip(seg, f)):
                if sp == sc:
                    out[i, j] = abs(fc - fp) * g.len[sp]
                    continue
                for ep, dp in ((g.u[sp], fp * g.len[sp]), (g.v[sp], (1 - fp) * g.len[sp])):
                    for ec, dc in ((g.u[sc], fc * g.len[sc]), (g.v[sc], (1 - fc) * g.len[sc])):
                        out[i, j] = min(out[i, j], dp + D[ep].get(ec, np.inf) + dc)
        return out

    def current(self):
        """Online answer: (point, segment) of the best candidate for the latest fix."""
        if not self.layers:
            return None
        L = self.layers[-1]
        j = int(np.argmax(L["score"]))
        return L["q"][j], int(L["seg"][j])

    def path(self):
        """Smoothed answer: Viterbi back-tracking. List of (t, matched point, segment, raw fix)."""
        out, j = [], -1
        for L in reversed(self.layers):
            if j < 0:
                j = int(np.argmax(L["score"]))
            out.append((L["t"], L["q"][j], int(L["seg"][j]), L["raw"]))
            j = L["back"][j]
        return out[::-1]


def nearest_road(graph, points, radius=50.0):
    """The naive baseline: snap every point to its nearest segment. Returns segment ids (-1 = none)."""
    out = []
    for p in np.asarray(points, float):
        seg, _, _, _ = graph.nearest(p, radius, 1)
        out.append(int(seg[0]) if len(seg) else -1)
    return np.array(out)
