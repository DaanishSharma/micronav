"""Render the micronav launch video: map animation + drive footage + the GPS-outage result.

Needs Pillow and ffmpeg. Usage:
    python scripts/make_launch_video.py RECORDING DRIVE_VIDEO OUT.mp4 [--offset 8.5] [--work DIR]
--offset: recording time (s) at which the drive video starts.
Map tiles (c) OpenStreetMap contributors, cached in ~/.cache/micronav/tiles.
"""
import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import micronav as mn
from micronav.mapmatch.osm import CACHE, USER_AGENT
from micronav.pipeline import process

W, H, FPS, Z, TILE = 1280, 720, 30, 16, 256
FONT = "/System/Library/Fonts/SFNS.ttf"
INK, BG = (30, 32, 36), (16, 18, 22)
ORANGE, BLUE, RED, PURPLE = (235, 104, 52), (42, 120, 214), (200, 50, 40), (107, 79, 187)


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default(size)


def world_px(lat, lon, z=Z):
    n = TILE * 2 ** z
    lat = np.asarray(lat, float)
    return (np.asarray(lon, float) + 180) / 360 * n, (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n


def tile(z, x, y):
    f = CACHE / "tiles" / str(z) / str(x) / f"{y}.png"
    if not f.exists():
        req = urllib.request.Request(f"https://tile.openstreetmap.org/{z}/{x}/{y}.png", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
    return Image.open(f).convert("RGB")


def basemap(cx, cy):
    x0, y0 = int(cx - W / 2), int(cy - H / 2)
    img = Image.new("RGB", (W, H))
    for tx in range(x0 // TILE, (x0 + W - 1) // TILE + 1):
        for ty in range(y0 // TILE, (y0 + H - 1) // TILE + 1):
            img.paste(tile(Z, tx, ty), (tx * TILE - x0, ty * TILE - y0))
    img = Image.blend(img, Image.new("RGB", (W, H), (255, 255, 255)), 0.35)
    d = ImageDraw.Draw(img)
    txt = "© OpenStreetMap contributors"
    d.rectangle([W - 215, H - 22, W, H], fill=(255, 255, 255))
    d.text((W - 208, H - 19), txt, fill=INK, font=font(13))
    return img, (x0, y0)


def to_px(ll, origin):
    px, py = world_px(ll[:, 0], ll[:, 1])
    return np.column_stack([px - origin[0], py - origin[1]])


def caption(d, lines, xy=(24, 20)):
    f1, f2 = font(30), font(20)
    w = max(d.textlength(lines[0], font=f1), *(d.textlength(s, font=f2) for s in lines[1:])) + 32
    h = 50 + 28 * (len(lines) - 1)
    d.rounded_rectangle([xy[0], xy[1], xy[0] + w, xy[1] + h], 10, fill=(255, 255, 255))
    d.text((xy[0] + 16, xy[1] + 8), lines[0], fill=INK, font=f1)
    for i, s in enumerate(lines[1:]):
        d.text((xy[0] + 16, xy[1] + 46 + 28 * i), s, fill=(70, 72, 78), font=f2)


def card(lines, sizes, colours=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    total = sum(s * 1.5 for s in sizes)
    y = (H - total) / 2
    for i, (s, sz) in enumerate(zip(lines, sizes)):
        f = font(sz)
        col = (colours or [None] * len(lines))[i] or (240, 240, 240)
        d.text(((W - d.textlength(s, font=f)) / 2, y), s, fill=col, font=f)
        y += sz * 1.5
    return img


def polyline(d, pts, colour, width):
    pts = pts[np.isfinite(pts).all(1)]
    if len(pts) > 1:
        d.line([tuple(p) for p in pts], fill=colour, width=width, joint="curve")


def parse_script(path):
    """Sections: a '## kind | line | line' header, then the narration. Kinds: card, drive, outage, matched, end."""
    out = []
    for block in Path(path).read_text().split("## ")[1:]:
        head, _, text = block.partition("\n")
        parts = [x.strip() for x in head.split("|")]
        out.append(dict(kind=parts[0], lines=parts[1:], text=" ".join(text.split())))
    return out


HERE = Path(__file__).resolve().parent


def speak(text, a, f):
    """Narrate `text` into file f with the chosen engine; returns its length in seconds.
    Lines are cached by (engine, voice, text), so re-renders don't spend ElevenLabs credits again."""
    import hashlib
    import shutil
    cached = CACHE / "tts" / (hashlib.sha1(f"{a.tts}|{a.voice}|{text}".encode()).hexdigest() + Path(f).suffix)
    if cached.exists():
        shutil.copy(cached, f)
    elif a.tts == "elevenlabs":
        sys.path.insert(0, str(HERE))
        from tts_elevenlabs import DEFAULT_VOICE, speak as eleven
        eleven(text, f, a.voice or DEFAULT_VOICE)
    elif a.tts == "apple":   # AVSpeechSynthesizer: uses premium voices properly, natural pace
        tool = Path(a.work) / "tts"
        if not tool.exists():
            subprocess.run(["swiftc", "-O", str(HERE / "tts.swift"), "-o", str(tool)], check=True)
        subprocess.run([str(tool), a.voice or "com.apple.voice.premium.en-US.Zoe", str(f), text], check=True)
    else:
        subprocess.run(["say", "-v", a.voice or "Samantha", "-o", str(f), text], check=True)
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, cached)
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(f)], capture_output=True, text=True, check=True).stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--offset", type=float, default=8.5, help="recording time (s) when the drive video starts")
    ap.add_argument("--work", default="launch_frames")
    ap.add_argument("--tts", choices=["elevenlabs", "apple", "say"], default="elevenlabs")
    ap.add_argument("--voice", default=None, help="voice id for the chosen engine (default: Rachel / Zoe Premium)")
    ap.add_argument("--script", default=str(Path(__file__).with_name("narration.txt")))
    ap.add_argument("--transpose", type=int, default=None,
                    help="ffmpeg transpose for a sideways drive video: 1 = 90 deg clockwise, 2 = counter-clockwise")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    for f in work.glob("*"):
        f.unlink()

    if a.voice is None:   # resolve the default, so the narration cache is keyed on the real voice
        if a.tts == "elevenlabs":
            sys.path.insert(0, str(HERE))
            from tts_elevenlabs import DEFAULT_VOICE
            a.voice = DEFAULT_VOICE
        else:
            a.voice = "com.apple.voice.premium.en-US.Zoe" if a.tts == "apple" else "Samantha"
    sections = parse_script(a.script)
    for i, sec in enumerate(sections):
        sec["audio"] = work / f"vo_{i}.{'mp3' if a.tts == 'elevenlabs' else 'caf' if a.tts == 'apple' else 'aiff'}"
        sec["dur"] = speak(sec["text"], a, sec["audio"]) + 0.8

    rec = mn.load_phyphox(a.recording)
    r = process(rec, roads=True, outage=(90.0, 30.0))
    s = r["stats"]["outage"]
    raw, ins, gps_t, ins_t = r["raw"], r["ins"], r["gps_t"], r["ins_t"]

    # centre the route in the upper-left part of the frame, leaving room for the drive video
    px, py = world_px(raw[:, 0], raw[:, 1])
    base, origin = basemap((px.min() + px.max()) / 2 + 40, (py.min() + py.max()) / 2 + 110)
    raw_px, ins_px = to_px(raw, origin), to_px(ins, origin)

    hide = r["outage_hide"]
    okf, oins = to_px(r["outage_kf"], origin), to_px(r["outage_ins"], origin)
    gaps = []                                          # [start, end] of each outage window
    for t in gps_t[hide]:
        if gaps and t - gaps[-1][1] < 1.5:
            gaps[-1][1] = t
        else:
            gaps.append([t, t])

    frame_i, drive_window = 0, None
    last_file = None

    def emit(img, n):
        nonlocal frame_i, last_file
        for _ in range(n):
            f = work / f"{frame_i:05d}.png"
            if img is None:
                f.hardlink_to(last_file)
            else:
                img.save(f, compress_level=1)
                last_file, img = f, None
            frame_i += 1

    t_now = 0.0
    for sec in sections:
        n = int(round(sec["dur"] * FPS))
        sec["start"] = t_now
        kind = sec["kind"]
        if kind == "card":
            lines = sec["lines"]
            sizes = [56] + [30] * (len(lines) - 1)
            cols = [None] + [(170, 175, 185)] * (len(lines) - 1)
            emit(card(lines, sizes, cols), n)
        elif kind == "end":
            emit(card(["pip install micronav", "open-source positioning on OpenStreetMap",
                       "github.com/DaanishSharma/micronav", "daanishsharma.github.io/blog"],
                      [60, 28, 30, 30], [(255, 255, 255), (170, 175, 185), (130, 170, 230), (130, 170, 230)]), n)
        elif kind == "drive":
            T0, T1 = a.offset, float(gps_t[-1])
            speedup = (T1 - T0) / sec["dur"]
            drive_window = (t_now, sec["dur"], speedup, T1 - T0)
            for i in range(n):
                T = T0 + i / FPS * speedup
                img = base.copy()
                d = ImageDraw.Draw(img)
                for p in raw_px[gps_t <= T]:
                    d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=ORANGE)
                polyline(d, ins_px[ins_t <= T], BLUE, 4)
                k = np.searchsorted(ins_t, T)
                if 0 < k <= len(ins_px):
                    q = ins_px[k - 1]
                    d.ellipse([q[0] - 8, q[1] - 8, q[0] + 8, q[1] + 8], fill=BLUE, outline=(255, 255, 255), width=3)
                caption(d, ["11 minutes, 4.3 km, one iPhone",
                            "orange: raw GPS   blue: micronav 15-state Kalman filter (IMU + GPS)",
                            f"{speedup:.0f}x speed"])
                emit(img, 1)
        elif kind == "outage":
            reveal = 0.62 * sec["dur"]                 # the numbers appear as the narration says them
            for i in range(n):
                u = min(1.0, i / (FPS * reveal * 0.8))
                img = base.copy()
                d = ImageDraw.Draw(img)
                for p in raw_px[~hide]:
                    d.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=(200, 200, 200))
                for g0, g1 in gaps[:max(1, int(round(u * len(gaps))))]:
                    for p in raw_px[(gps_t >= g0) & (gps_t <= g1)]:
                        d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], outline=INK, width=2)
                    polyline(d, okf[np.where((gps_t >= g0 - 1) & (gps_t <= g1 + 1))[0]], RED, 4)
                    polyline(d, oins[(ins_t >= g0 - 1) & (ins_t <= g1 + 1)], BLUE, 4)
                caption(d, [f"Now hide GPS for {s['gap']:.0f} s of every {s['every']:.0f} s",
                            "circles: the GPS fixes the filters never saw",
                            "red: GPS-only Kalman filter   blue: micronav (IMU + GPS + car constraints)"])
                if i > FPS * reveal:
                    f = font(40)
                    msg = f"GPS-only: {s['gps_only']:.0f} m off    micronav: {s['fused']:.0f} m off"
                    w = d.textlength(msg, font=f)
                    d.rounded_rectangle([(W - w) / 2 - 24, H - 130, (W + w) / 2 + 24, H - 60], 12, fill=BG)
                    d.text(((W - w) / 2, H - 122), msg, fill=(255, 255, 255), font=f)
                emit(img, 1)
        elif kind == "matched":
            img = base.copy()
            d = ImageDraw.Draw(img)
            for p in raw_px:
                d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=ORANGE)
            if r.get("matched") is not None:
                polyline(d, to_px(r["matched"], origin), PURPLE, 6)
            caption(d, ["Map matching on OpenStreetMap roads",
                        "orange: raw GPS   purple: most likely route (hidden Markov model + Viterbi)"])
            emit(img, n)
        else:
            raise ValueError(f"unknown section kind: {kind}")
        t_now += n / FPS

    vf, cmd = "", ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(work / "%05d.png")]
    if drive_window:
        t_start, drive_s, speedup, length = drive_window
        rot = f"transpose={a.transpose}," if a.transpose else ""
        pip_w, pip_h = 432, 243
        cmd += ["-i", a.video]
        vf = (f"[1:v]trim=duration={length:.2f},setpts=PTS/{speedup:.4f}+{t_start:.3f}/TB,fps={FPS},{rot}"
              f"scale={pip_w}:{pip_h}:force_original_aspect_ratio=increase,crop={pip_w}:{pip_h},"
              f"pad={pip_w + 8}:{pip_h + 8}:4:4:white[pip];"
              f"[0:v][pip]overlay={W - pip_w - 32}:{H - pip_h - 40}:"
              f"enable='between(t,{t_start:.3f},{t_start + drive_s:.3f})':eof_action=pass,format=yuv420p[v]")
    else:
        vf = "[0:v]format=yuv420p[v]"
    n_in = 2 if drive_window else 1
    for sec in sections:
        cmd += ["-i", str(sec["audio"])]
    for i, sec in enumerate(sections):
        vf += f";[{n_in + i}:a]adelay={int((sec['start'] + 0.4) * 1000)}:all=1[a{i}]"
    vf += ";" + "".join(f"[a{i}]" for i in range(len(sections)))
    vf += f"amix=inputs={len(sections)}:normalize=0,loudnorm=I=-16:TP=-1.5[a]"
    subprocess.run(cmd + ["-filter_complex", vf, "-map", "[v]", "-map", "[a]", "-c:a", "aac", "-b:a", "160k",
                          "-c:v", "libx264", "-crf", "22", "-preset", "medium", "-movflags", "+faststart",
                          "-t", f"{t_now:.2f}", a.out], check=True)
    print(f"wrote {a.out}: {t_now:.0f} s")


if __name__ == "__main__":
    main()
