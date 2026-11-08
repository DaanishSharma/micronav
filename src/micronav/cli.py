"""Command line: `micronav info REC` and `micronav run REC --out map.html`."""
import argparse
import sys

from .io import load_phyphox


def main(argv=None):
    ap = argparse.ArgumentParser(prog="micronav", description="Sensor fusion from scratch on phyphox recordings.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_info = sub.add_parser("info", help="summarise a recording")
    p_info.add_argument("recording", help="phyphox CSV export: folder or .zip")
    p_run = sub.add_parser("run", help="run every filter and write an HTML map")
    p_run.add_argument("recording")
    p_run.add_argument("--out", default="micronav_map.html")
    p_run.add_argument("--no-roads", action="store_true", help="skip OpenStreetMap download and map matching")
    a = ap.parse_args(argv)

    rec = load_phyphox(a.recording)
    if a.cmd == "info":
        print(rec.summary())
        return 0
    if rec.gps is None or rec.gyro is None or rec.accel is None:
        print("need Location, Gyroscope and Accelerometer in the recording", file=sys.stderr)
        return 1
    from .pipeline import process
    from .viz import write_html
    result = process(rec, roads=not a.no_roads)
    write_html(result, a.out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
