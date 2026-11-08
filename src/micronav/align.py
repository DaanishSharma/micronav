"""How does the phone sit in the car? (chapter 6)

To use the accelerometer in a car, we need R_CI: the rotation from phone axes to car axes
(car: x forward, y left, z up). Nobody mounts a phone straight, so we estimate it.

1. "Up": over a long drive the car's own acceleration averages out, so the mean
   accelerometer reading points up (gravity's reaction). That is the car's z axis.
2. "Forward": in the horizontal plane, the phone feels
       a_horizontal = (dv/dt) * forward + (v * yaw_rate) * left
   GPS gives dv/dt (braking, speeding up); GPS speed x gyro gives v * yaw_rate (turns).
   With forward = (cos phi, sin phi) in the plane and left = forward turned 90 deg,
   the best phi has a closed form: write each sample as a complex number and
       phi = angle( sum  a_k * conj(s1_k + i s2_k) )
"""
import numpy as np


def estimate_mount(t_imu, gyro, accel, t_gps, speed, min_speed=2.0):
    """Returns (R_CI, info). R_CI @ v_phone = v_car."""
    t_imu, t_gps, speed = np.asarray(t_imu), np.asarray(t_gps), np.asarray(speed, float)
    up = accel.mean(axis=0)
    up /= np.linalg.norm(up)
    e1 = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    A, S = [], []
    for k in range(len(t_gps) - 1):
        v0, v1 = speed[k], speed[k + 1]
        if not (np.isfinite(v0) and np.isfinite(v1)) or max(v0, v1) < min_speed:
            continue
        sel = (t_imu >= t_gps[k]) & (t_imu < t_gps[k + 1])
        if sel.sum() < 5:
            continue
        a = accel[sel].mean(0)
        w_up = gyro[sel].mean(0) @ up
        a_h = a - (a @ up) * up
        A.append(complex(a_h @ e1, a_h @ e2))
        S.append(complex((v1 - v0) / (t_gps[k + 1] - t_gps[k]), (v0 + v1) / 2 * w_up))
    A, S = np.array(A), np.array(S)
    if len(A) < 10:
        raise ValueError("not enough driving to estimate the mount (need speed changes and turns)")
    z = np.sum(A * np.conj(S))
    phi = np.angle(z)
    fwd = np.cos(phi) * e1 + np.sin(phi) * e2
    left = np.cross(up, fwd)
    R_CI = np.vstack([fwd, left, up])
    # how much of the horizontal acceleration the model explains (1 = all of it)
    explained = 1 - np.sum(np.abs(A - S * np.exp(1j * phi)) ** 2) / np.sum(np.abs(A) ** 2)
    return R_CI, dict(samples=len(A), explained=float(explained))
