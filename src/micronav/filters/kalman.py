"""The Kalman update, shared by every filter in micronav.

y = measured - predicted (the "innovation"), H = how the state error changes the
measurement, Rm = measurement noise covariance.

S = H P H^T + Rm is how big y *should* be if the filter's uncertainty is right.
NIS = y^T S^-1 y is how big it actually is, in units of that. A huge NIS means
"this reading doesn't fit": the gate skips it.
"""
import numpy as np

# chi-square thresholds: a reading this surprising happens by chance only 1 % / 0.1 % of the time
NIS_GATE_1 = 6.63    # 1 number, 99 %
NIS_GATE_2 = 13.8    # 2 numbers, 99.9 %
NIS_GATE_3 = 16.3    # 3 numbers, 99.9 %


def kalman_correct(P, y, H, Rm, gate=None):
    """Returns (dx, P_new, nis). dx is None when the gate rejected the reading."""
    y = np.atleast_1d(y)
    S = H @ P @ H.T + Rm
    nis = float(y @ np.linalg.solve(S, y))
    if gate is not None and nis > gate:
        return None, P, nis
    K = np.linalg.solve(S, H @ P).T                  # Kalman gain, P H^T S^-1
    dx = K @ y
    I_KH = np.eye(len(P)) - K @ H
    P = I_KH @ P @ I_KH.T + K @ Rm @ K.T              # Joseph form: stays symmetric and positive
    return dx, P, nis
