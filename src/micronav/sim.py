"""A simulated walk with a known true path (chapter 0).

Real recordings have no ground truth, so we can't score a filter on them.
In a simulation we know exactly where the device was, so every method gets a number.

The walk: stand still for 3 s, then walk a circle (radius 15 m, 1.2 m/s), looking
left and right, nodding slightly, bobbing with each step.
"""
from dataclasses import dataclass

import numpy as np

from .imu import GRAVITY
from .so3 import log_so3, ypr_to_R


@dataclass
class Sim:
    t: np.ndarray            # (N,) seconds
    dt: float
    p: np.ndarray            # (N, 3) true position, world (east, north, up), m
    v: np.ndarray            # (N, 3) true velocity, m/s
    a: np.ndarray            # (N, 3) true acceleration, m/s^2
    R_WI: np.ndarray         # (N, 3, 3) true orientation, IMU -> world
    yaw: np.ndarray          # (N,) true heading, rad
    gyro_ideal: np.ndarray   # (N, 3) rad/s, IMU axes
    accel_ideal: np.ndarray  # (N, 3) m/s^2 specific force, IMU axes
    gyro: np.ndarray         # ideal + bias + noise
    accel: np.ndarray        # ideal + bias + noise
    gyro_bias: np.ndarray
    accel_bias: np.ndarray
    mag: np.ndarray          # (N, 3) uT, IMU axes, with hard iron, a steel disturbance and noise
    B_W: np.ndarray          # Earth's field in world axes, uT
    hard_iron: np.ndarray    # magnetometer offset fixed in IMU axes, uT
    still: np.ndarray        # (N,) True while standing still


def simulate_walk(fs=200, duration=60.0, seed=0, radius=15.0, speed=1.2,
                  gyro_bias_dps=(0.3, -0.2, 0.4), accel_bias=(0.05, -0.04, 0.06),
                  gyro_noise_dps=0.5, accel_noise=0.08, mag_noise=0.5):
    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    t = np.arange(0, duration, dt)
    N = len(t)

    # smooth start: 0 while standing (t < 3 s), ramps to 1 by t = 5 s
    ramp = np.clip((t - 3) / 2, 0, 1)
    ramp = ramp * ramp * (3 - 2 * ramp)

    theta = np.cumsum(speed * ramp / radius) * dt  # angle around the circle
    p = np.column_stack([radius * np.cos(theta), radius * np.sin(theta),
                         1.7 + 0.03 * np.sin(2 * np.pi * 1.8 * t) * ramp])
    v = np.gradient(p, dt, axis=0)
    a = np.gradient(v, dt, axis=0)

    yaw = theta + np.pi / 2 + ramp * 0.5 * np.sin(2 * np.pi * 0.15 * t)  # look +-29 deg
    pitch = ramp * 0.15 * np.sin(2 * np.pi * 0.4 * t)                    # nod +-9 deg
    roll = ramp * 0.05 * np.sin(2 * np.pi * 0.9 * t)                     # sway +-3 deg
    R_WI = np.array([ypr_to_R(*x) for x in zip(yaw, pitch, roll)])

    # ideal readings: gyro = rotation between samples, in IMU axes; accel = R^T (a - g)
    gyro_ideal = np.array([log_so3(R_WI[k].T @ R_WI[k + 1]) / dt for k in range(N - 1)] + [np.zeros(3)])
    accel_ideal = np.einsum("nji,nj->ni", R_WI, a - GRAVITY)

    gyro_bias = np.radians(gyro_bias_dps)
    accel_bias = np.asarray(accel_bias, float)
    gyro = gyro_ideal + gyro_bias + rng.normal(0, np.radians(gyro_noise_dps), (N, 3))
    accel = accel_ideal + accel_bias + rng.normal(0, accel_noise, (N, 3))

    # magnetometer: Earth's field, a steel structure at 30-36 s, hard-iron offset, noise
    B_W = np.array([0.0, 19.6, -44.5])
    hard_iron = np.array([12.0, -7.0, 4.0])
    disturb = np.zeros((N, 3))
    near = (t > 30) & (t < 36)
    disturb[near] = np.array([20.0, 10.0, -25.0]) * np.sin(np.pi * (t[near] - 30) / 6)[:, None]
    mag = np.einsum("nji,nj->ni", R_WI, B_W + disturb) + hard_iron + rng.normal(0, mag_noise, (N, 3))

    return Sim(t, dt, p, v, a, R_WI, yaw, gyro_ideal, accel_ideal, gyro, accel,
               gyro_bias, accel_bias, mag, B_W, hard_iron, t < 3)


def simulate_gps(sim, rate=1.0, sigma=(3.0, 3.0, 5.0), seed=1):
    """Noisy position fixes at `rate` Hz. Returns (sample indices, (M, 3) ENU positions)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(0, len(sim.t), int(round(1 / (rate * sim.dt))))
    return idx, sim.p[idx] + rng.normal(0, sigma, (len(idx), 3))
