"""One filter for everything: a 15-state error-state EKF (chapter 7).

This is what drones and cars run. The IMU drives the prediction at 100 Hz; slower,
drift-free sensors correct it.

Best guess (nominal state): position p, velocity v, orientation R (IMU -> world),
accelerometer bias ba, gyro bias bg.
Error state (15 numbers): [dp (3), dv (3), dtheta (3), dba (3), dbg (3)], with
    R_true = exp(dtheta) R.

Predict (one IMU sample):
    a_world = R (f - ba) + g
    p += v dt + a_world dt^2 / 2,   v += a_world dt,   R = R exp((w - bg) dt)
Error dynamics (how errors grow):
    dp' = dv
    dv' = -[R (f - ba)]x dtheta - R dba     a tilt error leaks gravity into velocity
    dtheta' = -R dbg

Updates:
- GPS position (and optionally GPS velocity)
- non-holonomic constraint (cars don't slide sideways or fly): car-frame lateral and
  vertical velocity are ~0. Needs the mount R_CI from chapter 6. This is what keeps
  the filter on track when GPS drops out.
- zero velocity when stopped
- a road line (chapter 8): the car is on this line, so the distance across it is ~0
"""
import numpy as np

from ..so3 import exp_so3, rot_z, skew
from .kalman import NIS_GATE_2, NIS_GATE_3, kalman_correct

G = np.array([0.0, 0.0, -9.81])
P_, V_, TH, BA, BG = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15)


class INS15:
    def __init__(self, p0, v0, R0, sigma_p=5.0, sigma_v=1.0, sigma_att_deg=(2, 2, 5),
                 sigma_ba=0.2, sigma_bg_dps=0.5, accel_noise=0.2, gyro_noise_dps=0.3,
                 ba_walk=0.002, bg_walk_dps=0.002):
        self.p, self.v, self.R = np.array(p0, float), np.array(v0, float), np.array(R0, float)
        self.ba, self.bg = np.zeros(3), np.zeros(3)
        self.P = np.diag(np.r_[[sigma_p ** 2] * 3, [sigma_v ** 2] * 3,
                               np.radians(sigma_att_deg) ** 2, [sigma_ba ** 2] * 3,
                               [np.radians(sigma_bg_dps) ** 2] * 3])
        self.q = np.r_[[0.0] * 3, [accel_noise ** 2] * 3, [np.radians(gyro_noise_dps) ** 2] * 3,
                       [ba_walk ** 2] * 3, [np.radians(bg_walk_dps) ** 2] * 3]
        self.nis = np.nan

    def predict(self, w, f, dt):
        fb = f - self.ba
        a_w = self.R @ fb + G
        F = np.eye(15)
        F[P_, V_] = np.eye(3) * dt
        F[V_, TH] = -skew(self.R @ fb) * dt
        F[V_, BA] = -self.R * dt
        F[TH, BG] = -self.R * dt
        self.p = self.p + self.v * dt + 0.5 * a_w * dt ** 2
        self.v = self.v + a_w * dt
        self.R = self.R @ exp_so3((w - self.bg) * dt)
        self.P = F @ self.P @ F.T
        self.P[np.diag_indices(15)] += self.q * dt

    def correct(self, y, H, Rm, gate=None):
        dx, self.P, self.nis = kalman_correct(self.P, y, H, Rm, gate)
        if dx is None:
            return False
        self.p = self.p + dx[P_]
        self.v = self.v + dx[V_]
        self.R = exp_so3(dx[TH]) @ self.R
        self.ba = self.ba + dx[BA]
        self.bg = self.bg + dx[BG]
        return True

    def update_position(self, z, sigma_h, sigma_v=None, gate=NIS_GATE_3):
        """GPS fix z (3,) in metres. Without sigma_v only the horizontal part is used."""
        if sigma_v is None:
            H = np.zeros((2, 15))
            H[:, :2] = np.eye(2)
            return self.correct(z[:2] - self.p[:2], H, sigma_h ** 2 * np.eye(2), gate)
        H = np.zeros((3, 15))
        H[:, P_] = np.eye(3)
        return self.correct(z - self.p, H, np.diag([sigma_h ** 2, sigma_h ** 2, sigma_v ** 2]), gate)

    def update_velocity_en(self, v_en, sigma=1.0, gate=NIS_GATE_2):
        H = np.zeros((2, 15))
        H[:, 3:5] = np.eye(2)
        return self.correct(v_en - self.v[:2], H, sigma ** 2 * np.eye(2), gate)

    def update_nhc(self, R_CI, sigma=0.3):
        """Car-frame lateral and vertical velocity are ~0."""
        S = np.eye(3)[1:]                              # rows: car y (left), car z (up)
        M = S @ R_CI @ self.R.T                        # world velocity -> car (y, z)
        y = -M @ self.v
        H = np.zeros((2, 15))
        H[:, V_] = M
        H[:, TH] = M @ skew(self.v)                    # R^T exp(-dtheta) v = R^T (v + [v]x dtheta)
        return self.correct(y, H, sigma ** 2 * np.eye(2))

    def update_zero_velocity(self, sigma=0.05):
        H = np.zeros((3, 15))
        H[:, V_] = np.eye(3)
        return self.correct(-self.v, H, sigma ** 2 * np.eye(3))

    def update_line(self, q, d, sigma=2.0, gate=NIS_GATE_2 / 2):
        """The car is on the line through q with direction d (2D): distance across it is ~0."""
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        H = np.zeros((1, 15))
        H[0, :2] = n
        return self.correct(np.array([n @ (q[:2] - self.p[:2])]), H, np.array([[sigma ** 2]]), gate)

    @property
    def sigma_p(self):
        return np.sqrt(np.diag(self.P)[:3])


def course_to_yaw(course_deg):
    """GPS course (clockwise from north, degrees) -> yaw (counter-clockwise from east, radians)."""
    return np.radians(90.0 - course_deg)


def run_ins(t, gyro, accel, gps_t, gps_p, gps_acc, gps_speed=None, gps_course=None, R_CI=None,
            hide=None, use_nhc=True, use_gps_velocity=False, use_zupt=True, road=None,
            nhc_every=0.1, start_speed=3.0, **kw):
    """Run the 15-state filter over a recording.

    t, gyro, accel: IMU on one clock. gps_*: fixes (positions in metres). R_CI: the phone
    mount (chapter 6). hide: GPS fixes to leave out. road: a RoadGraph for chapter 8.
    The filter starts at the first fix where the car moves faster than `start_speed`,
    with heading from GPS course and the mount.
    Returns dict: t, p, v, R, ba, bg, sigma_p (at IMU rate), k0 (first IMU index used).
    """
    t, gps_t = np.asarray(t, float), np.asarray(gps_t, float)
    gps_p = np.asarray(gps_p, float)
    hide = np.zeros(len(gps_t), bool) if hide is None else np.asarray(hide, bool)
    sp = np.full(len(gps_t), np.nan) if gps_speed is None else np.asarray(gps_speed, float)
    co = np.full(len(gps_t), np.nan) if gps_course is None else np.asarray(gps_course, float)
    acc = np.clip(np.nan_to_num(np.asarray(gps_acc, float), nan=10.0), 3.0, None)

    ok = (sp > start_speed) & np.isfinite(co) & ~hide
    if R_CI is None or not ok.any():
        raise ValueError("run_ins needs the mount R_CI and a moving GPS fix to start from")
    j0 = int(np.argmax(ok))
    yaw0 = course_to_yaw(co[j0])
    R0 = rot_z(yaw0) @ R_CI                            # car level, pointing along the GPS course
    v0 = sp[j0] * np.array([np.cos(yaw0), np.sin(yaw0), 0.0])
    ins = INS15(gps_p[j0], v0, R0, sigma_p=acc[j0], **kw)

    k0 = int(np.searchsorted(t, gps_t[j0]))
    N = len(t)
    out = dict(t=t, p=np.full((N, 3), np.nan), v=np.full((N, 3), np.nan), R=np.full((N, 3, 3), np.nan),
               ba=np.full((N, 3), np.nan), bg=np.full((N, 3), np.nan), sigma_p=np.full((N, 3), np.nan),
               k0=k0, gps_used=np.zeros(len(gps_t), bool))
    j = j0 + 1
    t_nhc = -np.inf
    for k in range(k0, N):
        if k > k0:
            ins.predict(gyro[k - 1], accel[k - 1], float(np.clip(t[k] - t[k - 1], 0, 0.05)))
        while j < len(gps_t) and gps_t[j] <= t[k]:
            stopped = np.isfinite(sp[j]) and sp[j] < 0.3
            if not hide[j]:
                out["gps_used"][j] = ins.update_position(gps_p[j], acc[j], 2 * acc[j])
                if use_gps_velocity and np.isfinite(co[j]) and sp[j] > 1.5:
                    yaw = course_to_yaw(co[j])
                    ins.update_velocity_en(sp[j] * np.array([np.cos(yaw), np.sin(yaw)]))
            if use_zupt and stopped:
                ins.update_zero_velocity()
            if road is not None:
                _road_update(ins, road)
            j += 1
        if use_nhc and t[k] - t_nhc >= nhc_every:
            t_nhc = t[k]
            ins.update_nhc(R_CI)
        out["p"][k], out["v"][k], out["R"][k] = ins.p, ins.v, ins.R
        out["ba"][k], out["bg"][k], out["sigma_p"][k] = ins.ba, ins.bg, ins.sigma_p
    return out


def _road_update(ins, road, radius=25.0, min_speed=2.0):
    from .road_update import road_line
    line = road_line(road, ins.p, ins.v, radius=radius, min_speed=min_speed)
    if line is not None:
        ins.update_line(*line)
