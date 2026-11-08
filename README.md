# micronav

Sensor fusion and navigation from scratch, in plain numpy. Every formula is written out, and every filter is tested against a simulation where the true path is known.

I'm building this to learn sensor fusion and map matching from zero, one chapter at a time. Each chapter adds one module, one test and one blog post.

```bash
pip install micronav
```

```python
import micronav as mn

rec = mn.load_phyphox("Mapping 2026-10-17 17-42-34")   # a phyphox recording folder or .zip
print(rec.summary())

sim = mn.simulate_walk()                              # 60 s walk with a known true path
R = mn.integrate_gyro(sim.gyro, sim.R_WI[0], sim.dt)  # gyro -> orientation (watch it drift)
```

## Chapters
| # | Concept | Module |
|---|---|---|
| 0 | Phyphox recordings and a simulated walk with known truth | `io.phyphox`, `sim` |
| 1 | Frames, rotations, and why every sensor lies | `so3`, `geo`, `imu` |
| 2 | The Kalman filter in 1-D | coming |
| 3 | Orientation: error-state Kalman filter, gravity + compass | coming |
| 4 | GPS filter, RTS smoother, held-out-fix evaluation | coming |
| 5 | Map matching: HMM + Viterbi | coming |
| 6 | Phone-in-car alignment | coming |
| 7 | One 15-state error-state EKF for the whole drive | coming |
| 8 | The road as a measurement | coming |
| 9 | Batch smoother vs RTS | coming |

## Development
```bash
pip install -e ".[dev]"
pytest
```

MIT licensed.
