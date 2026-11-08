"""Chapters 5-8 on a simulated car drive with a known path and phone mount."""
import numpy as np
import pytest

import micronav as mn
from micronav.mapmatch import nearest_road


@pytest.fixture(scope="module")
def drive():
    return mn.simulate_drive()


def run(d, **kw):
    return mn.run_ins(d.t, d.gyro, d.accel, d.gps_t, d.gps_p, d.gps_acc, d.gps_speed, d.gps_course,
                      R_CI=d.R_CI, **kw)


def test_map_matching_stays_on_the_right_road():
    # two parallel roads 15 m apart, joined only at their ends, 1 km long
    x = np.arange(0, 1001, 50.0)
    n = len(x)
    xy = np.r_[np.column_stack([x, 0 * x]), np.column_stack([x, 0 * x + 15])]
    seg = [(i, i + 1) for i in range(n - 1)] + [(n + i, n + i + 1) for i in range(n - 1)] + [(0, n), (n - 1, 2 * n - 1)]
    g = mn.RoadGraph(xy, seg)
    rng = np.random.default_rng(3)
    truth = np.column_stack([np.arange(20, 980, 10.0), np.zeros(96)])
    fixes = truth + rng.normal(0, 6, truth.shape)
    naive = nearest_road(g, fixes)
    on_upper = lambda s: (g.A[s, 1] > 7) & (g.B[s, 1] > 7)  # segments of the y = 15 road
    assert on_upper(naive).sum() > 5                      # nearest-road jumps across
    mm = mn.MapMatcher(g)
    for k, p in enumerate(fixes):
        mm.step(k, p, 6.0)
    matched = np.array([q for _, q, _, _ in mm.path()])
    assert len(matched) > 80 and np.abs(matched[:, 1]).max() < 1e-9


def test_mount_estimate(drive):
    R, info = mn.estimate_mount(drive.t, drive.gyro, drive.accel, drive.gps_t, drive.gps_speed)
    assert mn.rotation_angle_deg(R, drive.R_CI) < 3


def test_ins_tracks_and_learns_biases(drive):
    r = run(drive)
    k = r["k0"]
    assert np.median(np.linalg.norm(r["p"][k:, :2] - drive.p[k:, :2], axis=1)) < drive.gps_acc[0]
    assert np.degrees(np.abs(r["bg"][-1] - drive.gyro_bias)).max() < 0.05


def test_gps_outage_fused_beats_gps_only(drive):
    hide = mn.outage_mask(drive.gps_t, every=80, gap=30)
    gi = drive.gps_idx
    kf = mn.held_out_errors(mn.track_gps(drive.gps_t, drive.gps_p, drive.gps_acc, hide=hide)["x"], drive.p[gi], hide)
    p = run(drive, hide=hide)["p"][gi]
    m = hide & np.isfinite(p[:, 0])
    fused = np.linalg.norm(p[m, :2] - drive.p[gi][m, :2], axis=1)
    no_nhc = run(drive, hide=hide, use_nhc=False)["p"][gi]
    no_nhc = np.linalg.norm(no_nhc[m, :2] - drive.p[gi][m, :2], axis=1)
    assert np.median(fused) < np.median(no_nhc) < np.median(kf)


def test_road_constraint_helps(drive):
    road = mn.RoadGraph.from_polyline(drive.p[::200])
    err = lambda r: np.median(np.linalg.norm(r["p"][r["k0"]:, :2] - drive.p[r["k0"]:, :2], axis=1))
    assert err(run(drive, road=road)) < 0.8 * err(run(drive))
