"""
make.py — builds the film master: render every frame, concatenate, mux, normalise.

The renderer is pure CPU, so the work is split into contiguous frame ranges that run as
separate processes (one per core), each piping raw frames straight into its own x264
encoder.  The parts are then joined with the concat demuxer (identical encoder settings,
so `-c copy` is safe) and the voice-over/score mix is muxed with loudness normalisation.

Usage:
    python make.py                     # default: cpu_count workers, 1080x1920 @ 60 fps
    python make.py --parts 3 --ss 1.5
    python make.py --test 20           # 20 second proof
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import film

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
OUT = os.path.join(ROOT, "out")
VIDEO = os.path.join(ROOT, "video")


def ff():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run(cmd, log, cwd=SRC):
    with open(log, "w") as fh:
        p = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    return p.returncode, log


def tail(path, n=6):
    try:
        return "".join(open(path).readlines()[-n:]).strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--ss", type=float, default=1.6, help="supersampling factor")
    ap.add_argument("--trail", type=float, default=0.35, help="shutter trail blend")
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--preset", default="faster")
    ap.add_argument("--fps", type=int, default=film.FPS)
    ap.add_argument("--test", type=float, default=0.0, help="render only N seconds")
    ap.add_argument("--name", default="rahasia_tembok_besar_china_1080x1920")
    ap.add_argument("--keep-parts", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    ffmpeg = ff()
    tl = film.build_timeline(os.path.join(SRC, "story.json"),
                             os.path.join(ROOT, "assets", "audio"), ffmpeg, fps=args.fps)
    total_frames = int(round((args.test or tl.total) * args.fps))
    print(f"[make] {tl.total:.2f}s · {total_frames} frames · {args.fps} fps · "
          f"{args.parts} workers · ss={args.ss}", flush=True)

    # ---- audio (once) -------------------------------------------------------
    t0 = time.time()
    run([sys.executable, "render.py", "--audio-only"], os.path.join(OUT, "log_audio.txt"))
    print(f"[make] audio mix ready ({time.time() - t0:.0f}s)", flush=True)

    # ---- split --------------------------------------------------------------
    edges = [int(round(i * total_frames / args.parts)) for i in range(args.parts + 1)]
    jobs = [(f"part{i}", edges[i], edges[i + 1]) for i in range(args.parts)
            if edges[i + 1] > edges[i]]
    parts = []
    t_all = time.time()

    def work(job):
        pid, a, b = job
        out = os.path.join(OUT, f"{pid}.mp4")
        log = os.path.join(OUT, f"log_{pid}.txt")
        t = time.time()
        rc, _ = run([sys.executable, "render.py", "--f0", str(a), "--f1", str(b),
                     "--out", out, "--no-final", "--no-audio", "--no-subs",
                     "--ss", str(args.ss), "--trail", str(args.trail), "--crf", str(args.crf),
                     "--preset", args.preset, "--fps", str(args.fps)], log)
        print(f"[make] {pid} frames {a}-{b} {'done' if rc == 0 else 'FAILED'} "
              f"in {time.time() - t:.0f}s" + ("" if rc == 0 else "\n" + tail(log)), flush=True)
        return rc, out

    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        for rc, out in ex.map(work, jobs):
            if rc != 0:
                sys.exit(f"[make] render failed: {out}")
            parts.append(out)
    print(f"[make] all parts rendered in {(time.time() - t_all) / 60:.1f} min", flush=True)

    # ---- concat, caption burn, mux ----------------------------------------
    # three small steps instead of one giant one: the single pass used enough peak
    # memory to get OOM-killed on a 4 GB box at 90 fps
    lst = os.path.join(OUT, "parts.txt")
    with open(lst, "w") as fh:
        for p in parts:
            fh.write(f"file '{p}'\n")
    joined = os.path.join(OUT, "joined.mp4")
    subprocess.run([ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", lst, "-c", "copy", joined], check=True)

    subs = film.write_ass(tl, os.path.join(OUT, "subs.ass"))
    shutil.copyfile(subs, os.path.join(OUT, "captions.subs.ass"))
    burnt = os.path.join(OUT, "burnt.mp4")
    subprocess.run([ffmpeg, "-y", "-v", "error", "-i", joined,
                    "-vf", f"ass={subs}:fontsdir={os.path.join(ROOT, 'assets', 'fonts')}",
                    "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
                    "-tune", "animation", "-profile:v", "high", "-g", str(args.fps * 2),
                    "-pix_fmt", "yuv420p", "-threads", "2",
                    "-x264-params", "rc-lookahead=15:sync-lookahead=0", burnt], check=True)

    os.makedirs(VIDEO, exist_ok=True)
    master = os.path.join(OUT, args.name + ".mp4")
    subprocess.run([ffmpeg, "-y", "-v", "error", "-i", burnt,
                    "-i", os.path.join(OUT, "mix.wav"),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    "-af", "loudnorm=I=-14:TP=-1.2:LRA=9",
                    "-shortest", "-movflags", "+faststart", master], check=True)
    final = master if args.test else os.path.join(VIDEO, os.path.basename(master))
    if not args.test:
        os.replace(master, final)
        shutil.copyfile(os.path.join(OUT, "captions.subs.ass"),
                        os.path.join(VIDEO, "captions.subs.ass"))

    # ---- report -------------------------------------------------------------
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", final],
                           capture_output=True, text=True).stderr
    meta = [ln.strip() for ln in probe.splitlines() if "Stream #" in ln or "Duration" in ln]
    size = os.path.getsize(final) / 1e6
    print(f"[make] master: {final}  ({size:.1f} MB)", flush=True)
    for m in meta:
        print("        " + m, flush=True)
    print(f"[make] total wall time {(time.time() - t_all) / 60:.1f} min", flush=True)

    if not args.keep_parts:
        for p in parts:
            try:
                os.remove(p)
            except OSError:
                pass
        for tmp in ("parts.txt", "joined.mp4", "burnt.mp4"):
            try:
                os.remove(os.path.join(OUT, tmp))
            except OSError:
                pass


if __name__ == "__main__":
    main()
