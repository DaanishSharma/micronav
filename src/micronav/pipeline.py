"""Everything, end to end, on one phyphox recording."""
import numpy as np

from .align import estimate_mount
from .filters import held_out_errors, outage_mask, run_ins, track_gps
from .geo import LocalFrame
from .mapmatch import MapMatcher, RoadGraph, fetch_roads
from .smoothers import rts_smooth


def process(rec, roads=True, outage=(90.0, 30.0), log=print):
    """Run every chapter on a Recording. Returns a dict of tracks (lat, lon arrays) and numbers."""
    t, gyro, accel, _ = rec.imu()
    g = rec.gps
    frame = LocalFrame(float(np.mean(g.lat)), float(np.mean(g.lon)), float(np.nanmean(g.alt)))
    P = frame.to_enu(g.lat, g.lon, g.alt)
    out = dict(frame=frame, gps_t=g.t, raw=np.column_stack([g.lat, g.lon]), stats={})

    def ll(xy):
        lat, lon, _ = frame.to_lla(xy[:, 0], xy[:, 1])
        return np.column_stack([lat, lon])

    tr = track_gps(g.t, P, g.h_acc)
    xs, _ = rts_smooth(tr)
    out["kf"], out["rts"] = ll(tr["x"]), ll(xs)
    out["stats"]["median_nis"] = float(np.nanmedian(tr["nis"]))
    log(f"GPS filter: median NIS {out['stats']['median_nis']:.3f} (1.39 expected if the phone's accuracy were honest)")

    R_CI, info = estimate_mount(t, gyro, accel, g.t, g.speed)
    out["R_CI"] = R_CI
    log(f"phone mount: estimated from {info['samples']} GPS intervals, explains {100 * info['explained']:.0f}% of horizontal acceleration")

    ins = run_ins(t, gyro, accel, g.t, P, g.h_acc, g.speed, g.course, R_CI=R_CI)
    k0 = ins["k0"]
    step = max(1, int(round(0.2 / np.median(np.diff(t)))))     # keep 5 Hz for the map
    out["ins_t"] = t[k0::step]
    out["ins"] = ll(ins["p"][k0::step])

    if outage:
        every, gap = outage
        hide = outage_mask(g.t, every, gap)
        gi = np.clip(np.searchsorted(t, g.t), 0, len(t) - 1)
        e_kf = held_out_errors(track_gps(g.t, P, g.h_acc, hide=hide)["x"], P, hide)
        r = run_ins(t, gyro, accel, g.t, P, g.h_acc, g.speed, g.course, R_CI=R_CI, hide=hide)
        pe = r["p"][gi]
        m = hide & np.isfinite(pe[:, 0])
        e_ins = np.linalg.norm(pe[m, :2] - P[m, :2], axis=1)
        out["outage_hide"] = hide
        out["outage_ins"] = ll(r["p"][k0::step])
        out["outage_kf"] = ll(track_gps(g.t, P, g.h_acc, hide=hide)["x"])
        out["stats"]["outage"] = dict(every=every, gap=gap, gps_only=float(np.median(e_kf)),
                                      fused=float(np.median(e_ins)))
        log(f"GPS hidden {gap:.0f} s of every {every:.0f} s: GPS-only filter misses the hidden fixes by "
            f"{np.median(e_kf):.1f} m (median), the 15-state filter by {np.median(e_ins):.1f} m")

    if roads:
        pad = 0.003
        osm = fetch_roads(g.lat.min() - pad, g.lon.min() - pad, g.lat.max() + pad, g.lon.max() + pad)
        if osm is None:
            log("roads: Overpass unavailable, map matching skipped")
        else:
            graph = RoadGraph.from_osm(osm, frame)
            mm = MapMatcher(graph)
            for k in range(len(g.t)):
                mm.step(g.t[k], xs[k], float(np.sqrt(tr["P"][k, 0, 0])) if np.isfinite(tr["P"][k, 0, 0]) else 10.0)
            path = mm.path()
            out["matched"] = ll(np.array([q for _, q, _, _ in path]))
            log(f"map matching: {len(path)} fixes on {graph.len.sum() / 1000:.0f} km of road, {mm.breaks} chain restarts")
    return out
