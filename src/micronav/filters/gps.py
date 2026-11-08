"""A Kalman filter on GPS alone (chapter 4).

State x = [east, north, v_east, v_north]. Motion model: constant velocity, changed only
by random acceleration of size `accel_noise` (m/s^2). That is enough to turn noisy 1 Hz
fixes into a smooth track and to keep going through a short outage.

Each fix brings its own accuracy, so bad fixes count less. The NIS gate skips fixes
that don't fit; after `max_reject` in a row the filter restarts from GPS.
"""
import numpy as np

from .kalman import NIS_GATE_2, kalman_correct


def cv_F(dt):
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt                             # p += v dt
    return F


def cv_Q(dt, accel_noise):
    # white random acceleration: position blurs by dt^3/3, velocity by dt, linked by dt^2/2
    return accel_noise ** 2 * np.kron([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]], np.eye(2))


class PositionKF:
    def __init__(self, p0, sigma_p0, accel_noise=3.0, sigma_v0=10.0):
        self.x = np.r_[np.asarray(p0, float)[:2], 0.0, 0.0]
        self.P = np.diag([sigma_p0 ** 2, sigma_p0 ** 2, sigma_v0 ** 2, sigma_v0 ** 2])
        self.accel_noise, self.nis, self.n_rej = accel_noise, np.nan, 0

    def predict(self, dt):
        F = cv_F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + cv_Q(dt, self.accel_noise)
        return F

    def correct(self, z, H, Rm, gate=None):
        y = z - H @ self.x
        dx, self.P, self.nis = kalman_correct(self.P, y, H, Rm, gate)
        if dx is None:
            return False
        self.x = self.x + dx
        return True

    def update_gps(self, p, accuracy, gate=NIS_GATE_2, max_reject=5):
        H = np.hstack([np.eye(2), np.zeros((2, 2))])
        ok = self.correct(np.asarray(p)[:2], H, accuracy ** 2 * np.eye(2), gate)
        self.n_rej = 0 if ok else self.n_rej + 1
        if self.n_rej >= max_reject:                   # lost: restart from GPS
            self.x[:2] = p[:2]
            self.P[:2, :2] = accuracy ** 2 * np.eye(2)
            self.n_rej = 0
        return ok

    def update_velocity(self, speed, course_deg, sigma=2.0, gate=NIS_GATE_2):
        c = np.radians(course_deg)                     # course: clockwise from north
        z = speed * np.array([np.sin(c), np.cos(c)])
        return self.correct(z, np.hstack([np.zeros((2, 2)), np.eye(2)]), sigma ** 2 * np.eye(2), gate)


def track_gps(t, p, accuracy, speed=None, course=None, use_velocity=False, accel_noise=3.0,
              hide=None, gate=NIS_GATE_2, min_accuracy=3.0):
    """Forward pass over all fixes.

    t (n,), p (n, 2+) metres, accuracy (n,) metres. hide: fixes to leave out (for testing).
    Keeps what the RTS smoother needs: predictions (xp, Pp), results (x, P) and F.
    """
    t = np.asarray(t, float)
    n = len(t)
    acc = np.clip(np.nan_to_num(np.asarray(accuracy, float), nan=10.0), min_accuracy, None)
    hide = np.zeros(n, bool) if hide is None else np.asarray(hide, bool)
    k0 = int(np.argmax(~hide))
    kf = PositionKF(p[k0], acc[k0], accel_noise)
    out = dict(t=t, x=np.zeros((n, 4)), P=np.zeros((n, 4, 4)), xp=np.zeros((n, 4)),
               Pp=np.zeros((n, 4, 4)), F=np.zeros((n, 4, 4)), used=np.zeros(n, bool),
               nis=np.full(n, np.nan), accel_noise=accel_noise)
    for k in range(n):
        out["F"][k] = kf.predict(t[k] - t[k - 1] if k else 0.0)
        out["xp"][k], out["Pp"][k] = kf.x, kf.P
        if not hide[k]:
            out["used"][k] = kf.update_gps(p[k], acc[k], gate=gate)
            out["nis"][k] = kf.nis
            if (use_velocity and speed is not None and np.isfinite(speed[k])
                    and np.isfinite(course[k]) and speed[k] > 1.5):
                kf.update_velocity(speed[k], course[k])
        out["x"][k], out["P"][k] = kf.x, kf.P
    return out


def outage_mask(t, every=60.0, gap=10.0, offset=None):
    """True for fixes inside a simulated outage: the last `gap` s of every `every` s."""
    t = np.asarray(t, float)
    offset = every - gap if offset is None else offset
    return ((t - t[0]) % every) >= offset


def held_out_errors(estimate, truth, hide):
    """Horizontal distance from estimate to the hidden fixes (or to the true path in a simulation)."""
    return np.linalg.norm(np.asarray(estimate)[hide, :2] - np.asarray(truth)[hide, :2], axis=1)
