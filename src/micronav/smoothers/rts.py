"""Rauch-Tung-Striebel smoother (chapter 4).

A filter only uses readings up to now: that's all you have while driving. Afterwards you
can do better: run backwards over the filter's results and let every later reading
correct the earlier estimates.

    C_k   = P_k F_{k+1}^T (P^-_{k+1})^-1
    x^s_k = x_k + C_k (x^s_{k+1} - x^-_{k+1})
    P^s_k = P_k + C_k (P^s_{k+1} - P^-_{k+1}) C_k^T
"""
import numpy as np


def rts_smooth(tr):
    """tr: output of a forward pass with keys x, P, xp, Pp, F. Returns (x_smoothed, P_smoothed)."""
    x, P = tr["x"].copy(), tr["P"].copy()
    for k in range(len(x) - 2, -1, -1):
        C = np.linalg.solve(tr["Pp"][k + 1].T, (tr["P"][k] @ tr["F"][k + 1].T).T).T
        x[k] = tr["x"][k] + C @ (x[k + 1] - tr["xp"][k + 1])
        P[k] = tr["P"][k] + C @ (P[k + 1] - tr["Pp"][k + 1]) @ C.T
    return x, P
