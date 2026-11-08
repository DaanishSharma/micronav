"""Loader tests on the Oct 17 drive. Skipped when the recording isn't on this machine.

Point MICRONAV_SAMPLE at a phyphox export folder to use a different one.
"""
import os
from pathlib import Path

import numpy as np
import pytest

import micronav as mn

HERE = Path(__file__).resolve().parent
DEFAULT = HERE.parent.parent / "Mapping 2026-10-17 17-42-34"
FOLDER = Path(os.environ.get("MICRONAV_SAMPLE", DEFAULT))
ZIP = FOLDER.with_suffix(".zip")

needs_sample = pytest.mark.skipif(not FOLDER.is_dir(), reason="sample recording not found")


@needs_sample
def test_folder_loads():
    rec = mn.load_phyphox(FOLDER)
    assert rec.device["deviceModel"] == "iPhone14,5"
    assert len(rec.accel.t) > 60000 and rec.accel.xyz.shape[1] == 3
    assert 95 < rec.gyro.rate < 105
    assert len(rec.gps.t) == 698
    assert np.nanmedian(rec.gps.h_acc) < 10
    assert rec.start_unix == pytest.approx(1792238425.346, abs=1e-3)
    # phone held still-ish at the start: accelerometer magnitude near g
    assert abs(np.linalg.norm(rec.accel.xyz[:100].mean(0)) - 9.81) < 0.5


@needs_sample
@pytest.mark.skipif(not ZIP.exists(), reason="zip export not found")
def test_zip_matches_folder():
    a, b = mn.load_phyphox(FOLDER), mn.load_phyphox(ZIP)
    assert np.array_equal(a.gyro.xyz, b.gyro.xyz)
    assert np.array_equal(a.gps.lat, b.gps.lat)


@needs_sample
def test_shared_imu_clock():
    rec = mn.load_phyphox(FOLDER)
    t, gyro, accel, mag = rec.imu()
    assert len(t) == len(gyro) == len(accel) == len(mag)
    assert np.all(np.diff(t) > 0)
    assert t[0] >= rec.accel.t[0] and t[-1] <= rec.accel.t[-1]
