"""Latitude/longitude <-> metres (chapter 1).

GPS gives latitude and longitude. Filters want metres. `LocalFrame` picks a reference
point and measures east (e), north (n) and up (u) from it.

It treats the Earth as flat around the reference point. Over a few km the error is
centimetres, far below phone GPS noise.
"""
from dataclasses import dataclass

import numpy as np

R_EARTH = 6378137.0  # metres, WGS-84 equator radius


@dataclass(frozen=True)
class LocalFrame:
    lat0: float
    lon0: float
    alt0: float = 0.0

    def to_enu(self, lat, lon, alt=None):
        """Degrees (+ metres) -> (N, 3) array of east, north, up in metres."""
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        alt = np.full_like(lat, self.alt0) if alt is None else np.asarray(alt, float)
        n = np.radians(lat - self.lat0) * R_EARTH
        e = np.radians(lon - self.lon0) * R_EARTH * np.cos(np.radians(self.lat0))
        return np.column_stack([e, n, alt - self.alt0])

    def to_lla(self, e, n, u=0.0):
        """Metres -> (lat, lon, alt) in degrees and metres."""
        e, n = np.asarray(e, float), np.asarray(n, float)
        lat = self.lat0 + np.degrees(n / R_EARTH)
        lon = self.lon0 + np.degrees(e / (R_EARTH * np.cos(np.radians(self.lat0))))
        return lat, lon, self.alt0 + np.asarray(u, float)
