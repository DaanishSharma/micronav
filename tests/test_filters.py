"""Chapters 2-4 and 9: every filter is scored against a simulation with a known answer."""
import numpy as np
import pytest

import micronav as mn
from micronav.filters import NIS_GATE_1


def test_kf1d_constant_is_the_running_average():
    z = 5 + np.random.default_rng(0).normal(0, 2, 100)
    x, P, K = mn.filter_1d(z, r=4.0)
    assert x[-1] == pytest.approx(z.mean(), abs=1e-5)
    assert P[-1] == pytest.approx(4.0 / 100, rel=1e-4)    # variance r / n
    assert np.all(np.diff(K) < 0)                         # each new reading counts a bit less


@pytest.fixture(scope="module")
def walk():
    return mn.simulate_walk()


def _yaw_err(R, yaw):
    est = np.array([mn.R_to_ypr(r)[0] for r in R])
    return np.degrees(np.abs(np.angle(np.exp(1j * (est - yaw)))))


def test_orientation_gravity_fixes_tilt_not_heading(walk):
    o = mn.run_orientation(walk.t, walk.gyro, walk.accel, R0=walk.R_WI[0])
    tilt = [mn.angle_between(r.T @ [0, 0, 1], rt.T @ [0, 0, 1]) for r, rt in zip(o["R"], walk.R_WI)]
    assert max(tilt[200:]) < 5
    assert o["sigma_deg"][-1, 2] > 30                     # the filter knows heading is unknown


def test_compass_with_gates_rejects_the_steel(walk):
    kw = dict(R0=walk.R_WI[0], hard_iron=walk.hard_iron, field=np.linalg.norm(walk.B_W))
    trusting = mn.run_orientation(walk.t, walk.gyro, walk.accel, mag=walk.mag, nis_gate=None, **kw)
    gated = mn.run_orientation(walk.t, walk.gyro, walk.accel, mag=walk.mag, field_gate=3.0,
                               nis_gate=NIS_GATE_1, **kw)
    assert _yaw_err(trusting["R"], walk.yaw).max() > 15   # steel at 30-36 s pulls heading off
    assert _yaw_err(gated["R"], walk.yaw).max() < 5
    assert np.degrees(np.abs(gated["bias"][-1] - walk.gyro_bias)).max() < 0.1


def test_gps_filter_and_smoother_beat_raw(walk):
    idx, fixes = mn.simulate_gps(walk)
    t = walk.t[idx]
    tr = mn.track_gps(t, fixes, np.full(len(t), 3.0))
    xs, _ = mn.rts_smooth(tr)
    err = lambda p: np.median(np.linalg.norm(p[:, :2] - walk.p[idx, :2], axis=1))
    assert err(xs) < err(tr["x"]) < err(fixes)


def test_batch_equals_rts(walk):
    idx, fixes = mn.simulate_gps(walk)
    t, acc = walk.t[idx], np.full(len(idx), 3.0)
    xr, Pr = mn.rts_smooth(mn.track_gps(t, fixes, acc, gate=None))
    xb, Pb = mn.batch_cv_smoother(t, fixes, acc)
    assert np.allclose(xr, xb, atol=1e-8)
    assert np.allclose(Pr, Pb, atol=1e-8)
