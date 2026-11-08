"""The Kalman filter in one dimension (chapter 2).

Everything a Kalman filter does fits in two numbers: a guess x and how unsure we are, P
(the variance: standard deviation squared).

- predict: time passes, the thing may have changed, so we get less sure: P += q * dt
- update: a reading z with noise variance r arrives. Blend guess and reading:
      K = P / (P + r)        # 0: ignore the reading, 1: trust it completely
      x = x + K (z - x)
      P = (1 - K) P          # always smaller: a reading never makes us less sure

With q = 0 (a constant) this is exactly the running average, and P = r / n.
Every bigger filter in micronav is this, with x a vector and P a matrix.
"""
import numpy as np


class KF1D:
    def __init__(self, x0, P0, q=0.0):
        self.x, self.P, self.q = float(x0), float(P0), float(q)

    def predict(self, dt=1.0):
        self.P += self.q * dt

    def update(self, z, r):
        K = self.P / (self.P + r)
        self.x += K * (z - self.x)
        self.P *= 1 - K
        return K


def filter_1d(z, r, q=0.0, x0=0.0, P0=1e6, dt=1.0):
    """Run KF1D over readings z. Returns arrays of estimates, variances and gains."""
    kf = KF1D(x0, P0, q)
    xs, Ps, Ks = [], [], []
    for zk in np.asarray(z, float):
        kf.predict(dt)
        Ks.append(kf.update(zk, r))
        xs.append(kf.x)
        Ps.append(kf.P)
    return np.array(xs), np.array(Ps), np.array(Ks)
