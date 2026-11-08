"""What the IMU alone can do, and how it fails (chapter 1).

- Gyro -> orientation: works, but drifts because of bias.
- Accelerometer -> position: error grows with time squared. Useless after seconds.

World frame: x east, y north, z up. The accelerometer measures specific force
f = R_WI^T (a - g): standing still it reads +9.81 upwards (the floor pushing up).
"""
import numpy as np

from .so3 import exp_so3

GRAVITY = np.array([0.0, 0.0, -9.81])


def integrate_gyro(gyro, R0, dt, bias=np.zeros(3)):
    """Chain tiny rotations: R[k+1] = R[k] @ exp((gyro[k] - bias) * dt).

    gyro: (N, 3) rad/s in IMU axes. R0: starting R_WI. dt: seconds (scalar or (N,) array).
    Returns (N, 3, 3) orientations R_WI.
    """
    gyro = np.asarray(gyro, float)
    dts = np.broadcast_to(np.asarray(dt, float), (len(gyro),))
    R = np.empty((len(gyro), 3, 3))
    R[0] = R0
    for k in range(len(gyro) - 1):
        R[k + 1] = R[k] @ exp_so3((gyro[k] - bias) * dts[k])
    return R


def dead_reckon(R_WI, accel, p0, v0, dt, accel_bias=np.zeros(3), g=GRAVITY):
    """Integrate the accelerometer twice to get position.

    a_world = R_WI @ (accel - bias) + g, then velocity and position.
    Returns (N, 3) positions in the world frame.
    """
    accel = np.asarray(accel, float)
    dts = np.broadcast_to(np.asarray(dt, float), (len(accel),))
    p = np.empty((len(accel), 3))
    p[0] = p0
    v = np.array(v0, float)
    for k in range(len(accel) - 1):
        a_world = R_WI[k] @ (accel[k] - accel_bias) + g
        p[k + 1] = p[k] + v * dts[k] + 0.5 * a_world * dts[k] ** 2
        v = v + a_world * dts[k]
    return p


def gyro_bias_from_still(gyro, still):
    """While the device is still, the gyro should read zero: its mean is the bias."""
    return np.asarray(gyro)[still].mean(axis=0)
