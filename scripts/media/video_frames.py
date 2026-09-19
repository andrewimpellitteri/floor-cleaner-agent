#!/usr/bin/env python3
"""Turn shop videos into a small set of frames worth a vision model's attention.

Why this shape: a 60 s clip at 30 fps is 1800 frames of mostly the same thing.
Sending them all to a vision model is expensive and produces 1800 restatements
of one fact. So this does triage:

  1. probe   -- fps, duration, rotation, resolution (rotation matters: phone
                video is often stored sideways with a display matrix)
  2. extract -- frames at a chosen fps into a directory, named by timestamp so
                a frame filename IS its time coordinate
  3. montage -- contact sheets (tiled grids, labelled with timestamps) so ONE
                vision call can say "the tape is legible at 4.2s and 11.8s,
                the wand is side-on at 7.5s", and only those frames then get a
                detailed pass at full resolution

What the model needs out of a wash-bay pass video (issue #11):
  - swath width     : a frame looking along the just-cleaned stripe, tape across it
  - standoff        : wand tip to floor, needs a SIDE-ON frame with a scale in it
  - tilt            : angle of the wand off vertical, same side-on frame
  - walking speed   : two frames where the operator passes two known tape marks;
                      speed = (mark2 - mark1) / (t2 - t1), and t comes free from
                      the frame filenames
  - overlap         : consecutive stripe edges against the tape

Usage:
    python scripts/media/video_frames.py probe   IN.mp4
    python scripts/media/video_frames.py extract IN.mp4 OUTDIR --fps 2
    python scripts/media/video_frames.py montage OUTDIR --cols 5 --rows 4
    python scripts/media/video_frames.py auto    IN.mp4 OUTDIR --fps 2
"""
from __future__ import annotations
import argparse, json, pathlib, shutil, subprocess, sys


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{p.stderr[-2000:]}")
    return p.stdout


def probe(path: pathlib.Path) -> dict:
    out = _run(["ffprobe", "-v", "error", "-print_format", "json",
                "-show_streams", "-show_format", str(path)])
    d = json.loads(out)
    v = next((s for s in d["streams"] if s.get("codec_type") == "video"), None)
    if v is None:
        sys.exit(f"{path}: no video stream")
    num, den = (v.get("avg_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    # Phone video is frequently stored sideways; the rotation lives either in a
    # side_data display matrix or in a stream tag. ffmpeg's scale filter honours
    # it on decode, but we report it so a human reading frames is not confused.
    rot = 0
    for sd in v.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = rot or int(float((v.get("tags") or {}).get("rotate", 0) or 0))
    info = {
        "path": str(path),
        "width": v.get("width"), "height": v.get("height"),
        "fps": round(fps, 3), "rotation": rot,
        "duration_s": round(float(d["format"].get("duration", 0.0)), 2),
        "nb_frames": v.get("nb_frames"),
        "codec": v.get("codec_name"),
    }
    info["est_frames_at_2fps"] = int(info["duration_s"] * 2)
    return info


def _transpose_for(rotation: int) -> str:
    """No-op: ffmpeg already applies the display matrix before user filters.

    Checked empirically on ffmpeg 8.0.1 against these clips (rotation=-90):
    `-vf scale=...` alone yields the correct portrait frame, and adding a
    transpose on top double-rotates it. Kept as a named seam because older
    ffmpeg builds did NOT autorotate once -vf was supplied; if frames ever come
    out sideways again, check `ffmpeg -version` before reaching for transpose.
    """
    return ""


def extract(path: pathlib.Path, outdir: pathlib.Path, fps: float,
            width: int, start: float | None, duration: float | None,
            rotation: int = 0) -> list[pathlib.Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("f_*.jpg"):
        old.unlink()
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    # -vsync vfr + fps filter gives one frame per 1/fps of SOURCE time, so the
    # frame index maps linearly onto timestamp regardless of source fps.
    tp = _transpose_for(rotation)
    vf = f"fps={fps}," + (f"{tp}," if tp else "") + f"scale={width}:-2"
    cmd += ["-vf", vf, "-vsync", "vfr",
            "-q:v", "3", str(outdir / "f_%05d.jpg")]
    _run(cmd)
    frames = sorted(outdir.glob("f_*.jpg"))
    # Rename to embed the timestamp: f_00001.jpg -> t0.50s.jpg
    off = start or 0.0
    named = []
    for i, f in enumerate(frames):
        t = off + i / fps
        new = outdir / f"t{t:07.2f}s.jpg"
        f.rename(new)
        named.append(new)
    return named


def montage(outdir: pathlib.Path, cols: int, rows: int, tile_w: int) -> list[pathlib.Path]:
    frames = sorted(outdir.glob("t*.jpg"))
    if not frames:
        sys.exit(f"{outdir}: no extracted frames (run extract first)")
    per = cols * rows
    sheets = []
    sheetdir = outdir / "sheets"
    sheetdir.mkdir(exist_ok=True)
    for old in sheetdir.glob("sheet_*.jpg"):
        old.unlink()
    for s in range(0, len(frames), per):
        chunk = frames[s:s + per]
        listfile = sheetdir / "_inputs.txt"
        # drawtext needs the timestamp burned in, else the model cannot tell the
        # caller WHICH frame to look at more closely.
        inputs, filt = [], []
        for i, f in enumerate(chunk):
            inputs += ["-i", str(f)]
            label = f.stem  # e.g. t0012.50s
            filt.append(
                f"[{i}:v]scale={tile_w}:-2,"
                f"drawtext=text='{label}':x=8:y=8:fontsize=28:fontcolor=yellow:"
                f"box=1:boxcolor=black@0.6:boxborderw=6[v{i}]")
        chain = "".join(f"[v{i}]" for i in range(len(chunk)))
        filt.append(f"{chain}xstack=inputs={len(chunk)}:"
                    f"layout={_xstack_layout(len(chunk), cols)}:fill=black[out]")
        sheet = sheetdir / f"sheet_{s // per:03d}.jpg"
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", *inputs,
               "-filter_complex", ";".join(filt), "-map", "[out]",
               "-q:v", "3", str(sheet)]
        _run(cmd)
        sheets.append(sheet)
        if listfile.exists():
            listfile.unlink()
    return sheets


def _xstack_layout(n: int, cols: int) -> str:
    """xstack layout string: w0_0|w0_h0|... built as a simple grid."""
    parts = []
    for i in range(n):
        r, c = divmod(i, cols)
        x = "0" if c == 0 else "+".join(["w0"] * c)
        y = "0" if r == 0 else "+".join(["h0"] * r)
        parts.append(f"{x}_{y}")
    return "|".join(parts)


def main():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ffmpeg/ffprobe not on PATH (brew install ffmpeg)")
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe");   p.add_argument("video")
    e = sub.add_parser("extract"); e.add_argument("video"); e.add_argument("outdir")
    m = sub.add_parser("montage"); m.add_argument("outdir")
    a = sub.add_parser("auto");    a.add_argument("video"); a.add_argument("outdir")
    for q in (e, a):
        q.add_argument("--fps", type=float, default=2.0)
        q.add_argument("--width", type=int, default=1280)
        q.add_argument("--start", type=float, default=None)
        q.add_argument("--duration", type=float, default=None)
    for q in (m, a):
        q.add_argument("--cols", type=int, default=5)
        q.add_argument("--rows", type=int, default=4)
        q.add_argument("--tile-width", type=int, default=480)
    args = ap.parse_args()

    if args.cmd == "probe":
        print(json.dumps(probe(pathlib.Path(args.video)), indent=2))
        return
    if args.cmd == "montage":
        for s in montage(pathlib.Path(args.outdir), args.cols, args.rows, args.tile_width):
            print(s)
        return

    vid, out = pathlib.Path(args.video), pathlib.Path(args.outdir)
    info = probe(vid)
    print(json.dumps(info, indent=2))
    if info["rotation"]:
        print(f"NOTE rotation={info['rotation']} deg; ffmpeg applies it on decode.")
    frames = extract(vid, out, args.fps, args.width, args.start, args.duration,
                     rotation=info["rotation"])
    print(f"extracted {len(frames)} frames at {args.fps} fps -> {out}")
    if args.cmd == "auto":
        sheets = montage(out, args.cols, args.rows, args.tile_width)
        print(f"{len(sheets)} contact sheet(s):")
        for s in sheets:
            print(f"  {s}")
        print("\nNext: show the sheet(s) to a vision model, ask which timestamps "
              "have (a) a legible tape measure, (b) a side-on wand, (c) a clean "
              "stripe edge. Then Read those full-res frames individually.")


if __name__ == "__main__":
    main()
