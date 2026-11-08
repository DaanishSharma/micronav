"""Batch smoothing: solve for the whole track at once (chapter 9, Barfoot ch. 3).

The RTS smoother is a clever recursion. The batch view is simpler to state: write down
every piece of information as a cost, and find the track that minimises the total.

    cost = |x_0 - prior|^2_P0                      what we believed at the start
         + sum_k |x_{k+1} - F x_k|^2_Q             the motion model
         + sum_k |z_k - H x_k|^2_R                 the readings
(|e|^2_S means e^T S^-1 e.)

For a linear model with Gaussian noise this is one big least-squares problem, and its
answer is *exactly* the RTS answer: two algorithms, one solution. The batch form is the
one that generalises: nonlinear models (Gauss-Newton), loop closures, and factor graphs.
"""
import numpy as np

from ..filters.gps import cv_F, cv_Q


def batch_cv_smoother(t, p, accuracy, accel_noise=3.0, sigma_v0=10.0, min_accuracy=3.0):
    """Batch MAP estimate for the constant-velocity GPS model of `track_gps`.

    Same model and start-up prior as `track_gps(..., gate=None)`, so the result matches
    `rts_smooth` of that pass. Returns (x (n, 4), P (n, 4, 4) marginal covariances).
    """
    t = np.asarray(t, float)
    p = np.asarray(p, float)[:, :2]
    acc = np.clip(np.nan_to_num(np.asarray(accuracy, float), nan=10.0), min_accuracy, None)
    n, d = len(t), 4
    A = np.zeros((n * d, n * d))                       # information matrix  (J^T W J)
    b = np.zeros(n * d)                                # information vector  (J^T W r)

    def add(rows_J, W, r):
        # one cost term |r - J x|^2_W^-1, with J given as {state index: block}
        for i, Ji in rows_J.items():
            b[i * d:(i + 1) * d] += Ji.T @ W @ r
            for j, Jj in rows_J.items():
                A[i * d:(i + 1) * d, j * d:(j + 1) * d] += Ji.T @ W @ Jj

    # prior on the first state, as in the filter's start-up
    P0 = np.diag([acc[0] ** 2, acc[0] ** 2, sigma_v0 ** 2, sigma_v0 ** 2])
    add({0: np.eye(d)}, np.linalg.inv(P0), np.r_[p[0], 0.0, 0.0])
    H = np.hstack([np.eye(2), np.zeros((2, 2))])
    for k in range(n):
        if k:
            dt = t[k] - t[k - 1]
            # x_k - F x_{k-1} = 0, with noise Q
            add({k - 1: -cv_F(dt), k: np.eye(d)}, np.linalg.inv(cv_Q(dt, accel_noise)), np.zeros(d))
        add({k: H}, np.eye(2) / acc[k] ** 2, p[k])
    x = np.linalg.solve(A, b).reshape(n, d)
    cov = np.linalg.inv(A)
    P = np.array([cov[k * d:(k + 1) * d, k * d:(k + 1) * d] for k in range(n)])
    return x, P
