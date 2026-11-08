import numpy as np

import micronav as mn


def test_exp_log_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = rng.uniform(-1, 1, 3)
        w *= rng.uniform(0, 3.0) / np.linalg.norm(w)   # angles up to 3 rad (< pi)
        R = mn.exp_so3(w)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)
        assert np.allclose(mn.log_so3(R), w, atol=1e-9)


def test_exp_matches_rot_x():
    assert np.allclose(mn.exp_so3([0.7, 0, 0]), mn.rot_x(0.7))


def test_skew_is_cross_product():
    a, b = np.array([1.0, 2, 3]), np.array([-4.0, 0.5, 2])
    assert np.allclose(mn.skew(a) @ b, np.cross(a, b))


def test_ypr_roundtrip():
    for ypr in [(0.3, -0.2, 0.1), (-2.5, 0.4, 1.2), (3.0, -1.0, -2.0)]:
        assert np.allclose(mn.R_to_ypr(mn.ypr_to_R(*ypr)), ypr)


def test_imu_rotation_seen_from_camera():
    # 120 deg about IMU x is 120 deg about camera z when camera z = IMU x
    R_IC = np.column_stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]]).astype(float)
    w = mn.log_so3(R_IC.T @ mn.exp_so3([np.radians(120), 0, 0]) @ R_IC)
    assert np.isclose(np.degrees(np.linalg.norm(w)), 120)
    assert np.allclose(w / np.linalg.norm(w), [0, 0, 1])


def test_local_frame_roundtrip():
    f = mn.LocalFrame(30.358, 76.385, 257.0)
    enu = f.to_enu([30.367, 30.358], [76.385, 76.395], [260.0, 257.0])
    assert abs(enu[0, 1] - 1001.7) < 1.0          # 0.009 deg of latitude is ~1 km north
    assert abs(enu[0, 2] - 3.0) < 1e-9
    lat, lon, alt = f.to_lla(enu[:, 0], enu[:, 1], enu[:, 2])
    assert np.allclose(lat, [30.367, 30.358]) and np.allclose(lon, [76.385, 76.395])
