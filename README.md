# micronav

Sensor fusion and navigation from scratch, in plain numpy. Every formula is written out, and every filter is tested against a simulation where the true path is known.

I built this to learn sensor fusion and map matching from zero, one chapter at a time, and I'm writing up each chapter at [daanishsharma.github.io/blog](https://daanishsharma.github.io/blog).

```bash
pip install micronav
micronav run "Mapping 2026-10-17 17-42-34.zip" --out map.html   # any phyphox CSV export
```

## On a real 11-minute drive (iPhone 13, phyphox, 4.3 km)
| | |
|---|---|
| Phone mount, estimated from braking and turns | upright, camera facing the road |
| Median NIS of the GPS filter | 0.016 (1.39 if the phone's reported accuracy were honest: it's pessimistic) |
| **GPS hidden 30 s of every 90 s**: distance to the hidden fixes, median | GPS-only Kalman filter **29 m** · 15-state filter **9 m** |
| GPS hidden 60 s of every 120 s | GPS-only **55 m** · 15-state **28 m** |

## Chapters
| # | Concept | Module |
|---|---|---|
| 0 | Phyphox recordings, and simulations with a known answer | `io.phyphox`, `sim` |
| 1 | Frames, rotations, and why every sensor lies | `so3`, `geo`, `imu` |
| 2 | The Kalman filter in 1-D | `filters.kf1d` |
| 3 | Orientation: error-state Kalman filter, gravity + compass, NIS gate | `filters.orientation` |
| 4 | GPS filter, RTS smoother, held-out-fix evaluation | `filters.gps`, `smoothers.rts` |
| 5 | Map matching: HMM + Viterbi on OpenStreetMap roads | `mapmatch` |
| 6 | How the phone sits in the car | `align` |
| 7 | One 15-state error-state EKF: IMU + GPS + "cars don't slide sideways" | `filters.ins15` |
| 8 | The road as a measurement inside the filter | `filters.road_update` |
| 9 | Batch smoothing: the same answer as RTS, from one big least-squares problem | `smoothers.batch` |

```python
import micronav as mn

sim = mn.simulate_drive()                                   # known path and phone mount
R_CI, _ = mn.estimate_mount(sim.t, sim.gyro, sim.accel, sim.gps_t, sim.gps_speed)
out = mn.run_ins(sim.t, sim.gyro, sim.accel, sim.gps_t, sim.gps_p, sim.gps_acc,
                 sim.gps_speed, sim.gps_course, R_CI=R_CI)
```

Only numpy is required. Map matching downloads roads from OpenStreetMap's Overpass API and caches them in `~/.cache/micronav`. Map data © OpenStreetMap contributors.

## Development
```bash
pip install -e ".[dev]"
pytest
```

References: T. Barfoot, *State Estimation for Robotics*; Solà, Deray & Atchuthan, *A micro Lie theory for state estimation in robotics*; Newson & Krumm, *Hidden Markov map matching through noise and sparseness* (2009).

MIT licensed.
