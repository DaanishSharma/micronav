"""The simulation is the ground truth for every later filter, so check it first."""
import numpy as np
import pytest

import micronav as mn


@pytest.fixture(scope="module")
def sim():
    return mn.simulate_walk()


def test_accel_reads_plus_g_when_still(sim):
    # stop just short of t = 3 s: the numerical derivative there already sees the walk starting
    assert np.allclose(sim.accel_ideal[sim.t < 2.9], [0, 0, 9.81], atol=1e-6)


def test_ideal_gyro_reproduces_true_orientation(sim):
    R = mn.integrate_gyro(sim.gyro_ideal, sim.R_WI[0], sim.dt)
    assert mn.rotation_angle_deg(R[-1], sim.R_WI[-1]) < 0.01


def test_gyro_bias_makes_heading_drift(sim):
    R = mn.integrate_gyro(sim.gyro, sim.R_WI[0], sim.dt)
    assert mn.rotation_angle_deg(R[-1], sim.R_WI[-1]) > 10   # ~0.5 deg/s bias for a minute


def test_bias_from_standing_still(sim):
    est = mn.gyro_bias_from_still(sim.gyro, sim.still)
    assert np.degrees(np.abs(est - sim.gyro_bias)).max() < 0.1
    R = mn.integrate_gyro(sim.gyro, sim.R_WI[0], sim.dt, bias=est)
    assert mn.rotation_angle_deg(R[-1], sim.R_WI[-1]) < 3


def test_dead_reckoning_perfect_vs_real(sim):
    p_ideal = mn.dead_reckon(sim.R_WI, sim.accel_ideal, sim.p[0], sim.v[0], sim.dt)
    p_real = mn.dead_reckon(sim.R_WI, sim.accel, sim.p[0], sim.v[0], sim.dt)
    err_ideal = np.linalg.norm(p_ideal[-1] - sim.p[-1])
    err_real = np.linalg.norm(p_real[-1] - sim.p[-1])
    assert err_ideal < 1.0
    # 0.05-0.06 m/s^2 accel bias alone gives about 0.5 * b * t^2 ~ 100 m after 60 s
    assert err_real > 30


def test_gps_noise_level(sim):
    idx, fixes = mn.simulate_gps(sim)
    assert len(idx) == 60
    err = fixes - sim.p[idx]
    assert 1.5 < err[:, 0].std() < 4.5
