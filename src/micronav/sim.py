"""A simulated walk with a known true path (chapter 0).

Real recordings have no ground truth, so we can't score a filter on them.
In a simulation we know exactly where the device was, so every method gets a number.

The walk: stand still for 3 s, then walk a circle (radius 15 m, 1.2 m/s), looking
left and right, nodding slightly, bobbing with each step.
"""
from dataclasses import dataclass

import numpy as np

from .imu import GRAVITY
from .so3 import log_so3, rot_x, rot_y, rot_z, ypr_to_R


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


@dataclass
class DriveSim:
    t: np.ndarray            # (N,) s
    dt: float
    p: np.ndarray            # (N, 3) true position, m
    v: np.ndarray            # (N, 3) true velocity, m/s
    R_WI: np.ndarray         # (N, 3, 3) true phone orientation
    R_CI: np.ndarray         # phone -> car rotation (how the phone sits in its holder)
    yaw: np.ndarray          # (N,) car heading, rad, counter-clockwise from east
    speed: np.ndarray        # (N,) m/s
    gyro: np.ndarray         # (N, 3) rad/s, phone axes, with bias + noise
    accel: np.ndarray        # (N, 3) m/s^2, phone axes, with bias + noise
    gyro_bias: np.ndarray
    accel_bias: np.ndarray
    gps_idx: np.ndarray      # (M,) sample index of each fix
    gps_p: np.ndarray        # (M, 3) noisy fixes, m
    gps_acc: np.ndarray      # (M,) reported accuracy, m
    gps_speed: np.ndarray    # (M,) m/s
    gps_course: np.ndarray   # (M,) degrees clockwise from north

    @property
    def gps_t(self):
        return self.t[self.gps_idx]


def _bumps(t, events):
    """Sum of smooth bumps: events = [(start, duration, height)]."""
    out = np.zeros_like(t)
    for t0, dur, h in events:
        x = np.clip((t - t0) / dur, 0, 1)
        out += h * np.sin(np.pi * x) ** 2
    return out


def simulate_drive(fs=100, duration=240.0, seed=0, R_CI=None, gps_sigma=4.0,
                   gyro_bias_dps=(0.2, -0.3, 0.25), accel_bias=(0.08, -0.06, 0.05),
                   gyro_noise_dps=0.3, accel_noise=0.15):
    """A car drive: start still, speed up, turns, braking, a stop. The phone sits in a holder
    at a fixed, unknown angle R_CI. Car frame: x forward, y left, z up. 1 Hz GPS."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    t = np.arange(0, duration, dt)
    # speed: accelerate to ~12 m/s, slow down for turns, stop in the middle, go again
    ramp = np.clip((t - 5) / 10, 0, 1)
    speed = 12 * ramp * ramp * (3 - 2 * ramp)
    speed -= _bumps(t, [(40, 15, 6), (90, 30, 12.0), (150, 12, 5), (195, 15, 7)])
    speed = np.clip(speed, 0, None)
    # yaw rate: left and right turns while moving
    yaw_rate = _bumps(t, [(42, 10, 0.18), (70, 8, -0.2), (120, 12, 0.15), (152, 9, -0.25), (197, 12, 0.2)])
    yaw = np.radians(30) + np.cumsum(yaw_rate) * dt
    v = np.column_stack([speed * np.cos(yaw), speed * np.sin(yaw), np.zeros_like(t)])
    p = np.cumsum(v, axis=0) * dt
    a = np.gradient(v, dt, axis=0)

    if R_CI is None:   # phone tilted back ~60 deg and turned ~25 deg in its holder
        R_CI = rot_z(np.radians(25)) @ rot_y(np.radians(-60)) @ rot_x(np.radians(5))
    R_WC = np.array([rot_z(y) for y in yaw])
    R_WI = R_WC @ R_CI
    w_car = np.column_stack([np.zeros_like(t), np.zeros_like(t), yaw_rate])
    gyro_ideal = w_car @ R_CI                          # = R_CI^T w_car, per row
    accel_ideal = np.einsum("nji,nj->ni", R_WI, a - GRAVITY)

    gyro_bias = np.radians(gyro_bias_dps)
    accel_bias = np.asarray(accel_bias, float)
    gyro = gyro_ideal + gyro_bias + rng.normal(0, np.radians(gyro_noise_dps), gyro_ideal.shape)
    accel = accel_ideal + accel_bias + rng.normal(0, accel_noise, accel_ideal.shape)

    gps_idx = np.arange(0, len(t), fs)
    gps_p = p[gps_idx] + rng.normal(0, [gps_sigma, gps_sigma, 2 * gps_sigma], (len(gps_idx), 3))
    sp = speed[gps_idx] + rng.normal(0, 0.3, len(gps_idx))
    course = np.degrees(np.pi / 2 - yaw[gps_idx]) % 360 + rng.normal(0, 2, len(gps_idx))
    course[sp < 1] = np.nan
    return DriveSim(t, dt, p, v, R_WI, R_CI, yaw, speed, gyro, accel, gyro_bias, accel_bias,
                    gps_idx, gps_p, np.full(len(gps_idx), gps_sigma), np.clip(sp, 0, None), course)
