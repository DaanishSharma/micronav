"""Orientation with an error-state Kalman filter (chapter 3).

Best guess ("nominal state"): R (IMU -> world) and b (gyro bias, rad/s).
The filter tracks the *errors* of that guess, x = [dtheta (3), db (3)]:
    R_true = exp(dtheta) @ R,   b_true = b + db
Rotations don't add like vectors, but small errors do, so the filter works on errors
and folds each correction back into R. That is the "error-state" trick.

- predict: integrate the gyro (chapter 1), grow the uncertainty
- gravity update: when not accelerating, the accelerometer points "up" -> fixes tilt
- heading update: the horizontal magnetic field points north -> fixes heading
Gravity can't see heading: without a compass, heading uncertainty only grows, and the
filter says so.
"""
import numpy as np

from ..so3 import exp_so3, skew
from .kalman import NIS_GATE_1, kalman_correct

UP = np.array([0.0, 0.0, 1.0])
NORTH = np.array([0.0, 1.0, 0.0])


class OrientationKF:
    def __init__(self, R0, att0_deg=(2, 2, 10), bias0_dps=1.0,
                 gyro_noise_dps=0.3, bias_walk_dps=0.002, grav_noise=0.05):
        self.R, self.b = np.array(R0, float), np.zeros(3)
        self.P = np.diag(np.r_[np.radians(att0_deg), np.radians([bias0_dps] * 3)] ** 2)
        self.q_att = np.radians(gyro_noise_dps) ** 2
        self.q_bias = np.radians(bias_walk_dps) ** 2
        self.grav_noise = grav_noise
        self.nis = np.nan

    def predict(self, w, dt):
        self.R = self.R @ exp_so3((w - self.b) * dt)
        F = np.eye(6)
        F[:3, 3:] = -self.R * dt                       # a bias error db adds -R db dt of orientation error
        self.P = F @ self.P @ F.T
        self.P[:3, :3] += self.q_att * dt * np.eye(3)
        self.P[3:, 3:] += self.q_bias * dt * np.eye(3)

    def correct(self, y, H, Rm, gate=None):
        dx, self.P, self.nis = kalman_correct(self.P, y, H, Rm, gate)
        if dx is None:
            return False
        self.R = exp_so3(dx[:3]) @ self.R              # fold the error into the guess...
        self.b = self.b + dx[3:]                       # ...after which the error is zero again
        return True

    def update_gravity(self, f, g=9.81):
        n = np.linalg.norm(f)
        if abs(n - g) > 1.0:                           # accelerating hard: f isn't just gravity
            return False
        y = f / n - self.R.T @ UP                      # measured "up" minus predicted "up", IMU axes
        H = np.hstack([self.R.T @ skew(UP), np.zeros((3, 3))])
        sigma = self.grav_noise + abs(n - g) / g
        return self.correct(y, H, sigma ** 2 * np.eye(3))

    @property
    def sigma_deg(self):
        return np.degrees(np.sqrt(np.diag(self.P)[:3]))


class OrientationMagKF(OrientationKF):
    """Adds a compass. Only the horizontal part of the field is used."""

    def __init__(self, R0, mag_noise_deg=3.0, unstick=75, **kw):
        super().__init__(R0, **kw)
        self.mag_noise, self.unstick, self.n_rej = np.radians(mag_noise_deg), unstick, 0

    def update_heading(self, m, gate=NIS_GATE_1):
        h = self.R @ m
        h[2] = 0                                       # field in world axes, horizontal part
        y = np.array([np.arctan2(np.cross(h, NORTH)[2], h @ NORTH)])
        H = np.zeros((1, 6))
        H[0, 2] = 1                                    # measures the vertical part of dtheta
        # a gate needs a way out: after `unstick` rejections in a row, accept anyway
        ok = self.correct(y, H, np.array([[self.mag_noise ** 2]]), gate if self.n_rej < self.unstick else None)
        self.n_rej = 0 if ok else self.n_rej + 1
        return ok


def R_from_gravity(f):
    """Start-up: tilt from one accelerometer reading, heading 0."""
    u = f / np.linalg.norm(f)
    ax = np.cross(u, UP)
    s = np.linalg.norm(ax)
    return exp_so3(ax / s * np.arctan2(s, u[2])) if s > 1e-9 else np.eye(3)


def run_orientation(t, gyro, accel, mag=None, R0=None, hard_iron=np.zeros(3), field=None,
                    field_gate=None, nis_gate=NIS_GATE_1, update_every=0.04, **kw):
    """Run the filter over a whole recording on one clock.

    mag: optional magnetometer (uT). field: expected field strength, for the field gate.
    Returns dict with R (N,3,3), bias (N,3) rad/s, sigma_deg (N,3), mag_used (N,) bool.
    """
    t = np.asarray(t, float)
    R0 = R_from_gravity(accel[0]) if R0 is None else R0
    kf = OrientationMagKF(R0, **kw) if mag is not None else OrientationKF(R0, **kw)
    N = len(t)
    out = dict(R=np.empty((N, 3, 3)), bias=np.empty((N, 3)), sigma_deg=np.empty((N, 3)),
               mag_used=np.zeros(N, bool))
    t_upd = -np.inf
    for k in range(N):
        if k:
            kf.predict(gyro[k - 1], float(np.clip(t[k] - t[k - 1], 0, 0.05)))
        if t[k] - t_upd >= update_every:
            t_upd = t[k]
            kf.update_gravity(accel[k])
            if mag is not None:
                m = mag[k] - hard_iron
                if field_gate is None or field is None or abs(np.linalg.norm(m) - field) < field_gate:
                    out["mag_used"][k] = kf.update_heading(m, nis_gate)
        out["R"][k], out["bias"][k], out["sigma_deg"][k] = kf.R, kf.b, kf.sigma_deg
    return out


def hard_iron_from_sweep(mag):
    """Turn the device in all directions: readings lie on a sphere whose centre is the offset."""
    mag = np.asarray(mag)
    return (mag.max(0) + mag.min(0)) / 2
