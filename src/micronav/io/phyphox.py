"""Read a phyphox CSV export (chapter 0).

phyphox ("Export data" -> CSV) gives a folder or .zip with one CSV per sensor:
Accelerometer.csv, Gyroscope.csv, Magnetometer.csv, Location.csv and meta/*.csv.
Every file starts its clock at 0 when you press start. Missing sensors are allowed.

Units: accelerometer m/s^2 (including gravity, i.e. specific force), gyro rad/s,
magnetometer uT, location in degrees and metres.
"""
import csv
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Series:
    """Timestamps (N,) in seconds and values (N, 3)."""
    t: np.ndarray
    xyz: np.ndarray

    @property
    def rate(self):
        return 1.0 / np.median(np.diff(self.t)) if len(self.t) > 1 else float("nan")


@dataclass
class Fixes:
    """GPS fixes. Speed/course/accuracy are NaN where the phone didn't report them."""
    t: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    speed: np.ndarray       # m/s
    course: np.ndarray      # degrees clockwise from north
    h_acc: np.ndarray       # horizontal accuracy, m
    v_acc: np.ndarray       # vertical accuracy, m


@dataclass
class Recording:
    accel: Optional[Series] = None
    gyro: Optional[Series] = None
    mag: Optional[Series] = None
    gps: Optional[Fixes] = None
    start_unix: Optional[float] = None   # wall-clock time (unix seconds) of t = 0
    device: dict = field(default_factory=dict)

    def summary(self):
        lines = [f"device: {self.device.get('deviceModel', '?')}"]
        for name in ("accel", "gyro", "mag"):
            s = getattr(self, name)
            if s is not None:
                lines.append(f"{name}: {len(s.t)} samples, {s.rate:.0f} Hz, {s.t[-1] - s.t[0]:.1f} s")
        if self.gps is not None:
            lines.append(f"gps: {len(self.gps.t)} fixes, median accuracy {np.nanmedian(self.gps.h_acc):.1f} m")
        return "\n".join(lines)

    def imu(self, t=None):
        """Gyro, accel (and mag, if present) on one shared clock, by linear interpolation.

        Each phyphox sensor has its own timestamps. Filters want one clock.
        Default clock: the gyro's, trimmed to where every sensor has data.
        Returns (t, gyro, accel, mag_or_None).
        """
        series = [s for s in (self.gyro, self.accel, self.mag) if s is not None]
        if self.gyro is None or self.accel is None:
            raise ValueError("need both gyroscope and accelerometer")
        if t is None:
            lo = max(s.t[0] for s in series)
            hi = min(s.t[-1] for s in series)
            t = self.gyro.t[(self.gyro.t >= lo) & (self.gyro.t <= hi)]

        def interp(s):
            return np.column_stack([np.interp(t, s.t, s.xyz[:, i]) for i in range(3)])

        return t, interp(self.gyro), interp(self.accel), (interp(self.mag) if self.mag is not None else None)


def _read_rows(text):
    rows = list(csv.reader(io.StringIO(text)))
    return rows[0], rows[1:]


def _numeric(rows):
    return np.array([[float(x) for x in r] for r in rows if r], float)


def load_phyphox(path):
    """Load a phyphox CSV export: a folder, or the .zip phyphox shares."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            names = {Path(n).as_posix(): n for n in z.namelist()}
            # files may sit at the root or inside one top-level folder
            files = {}
            for key, real in names.items():
                parts = key.split("/")
                short = "/".join(parts[-2:]) if len(parts) >= 2 and parts[-2] == "meta" else parts[-1]
                files[short] = z.read(real).decode("utf-8-sig")
    elif path.is_dir():
        files = {}
        for f in path.rglob("*.csv"):
            rel = f.relative_to(path).as_posix()
            files[rel if rel.startswith("meta/") else f.name] = f.read_text(encoding="utf-8-sig")
    else:
        raise FileNotFoundError(f"not a phyphox folder or .zip: {path}")

    rec = Recording()

    def series(name):
        if name not in files:
            return None
        _, rows = _read_rows(files[name])
        d = _numeric(rows)
        return Series(d[:, 0], d[:, 1:4])

    rec.accel = series("Accelerometer.csv")
    rec.gyro = series("Gyroscope.csv")
    rec.mag = series("Magnetometer.csv")

    if "Location.csv" in files:
        _, rows = _read_rows(files["Location.csv"])
        d = _numeric(rows)
        rec.gps = Fixes(t=d[:, 0], lat=d[:, 1], lon=d[:, 2], alt=d[:, 3], speed=d[:, 4],
                        course=d[:, 5], h_acc=d[:, 6], v_acc=d[:, 7])

    if "meta/time.csv" in files:
        _, rows = _read_rows(files["meta/time.csv"])
        for r in rows:
            if r and r[0] == "START":
                rec.start_unix = float(r[2])
                break

    if "meta/device.csv" in files:
        _, rows = _read_rows(files["meta/device.csv"])
        rec.device = {r[0]: r[1] for r in rows if len(r) >= 2}

    return rec
