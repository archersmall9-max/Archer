"""
render.py — renders the film: 3D frames -> h264, voice-over + score -> AAC, burned captions.

Usage: python render.py [--test SECONDS] [--fps N]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

import film
import shots as S
from engine3d import Renderer, post

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 1080, 1920


def ffmpeg_bin():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=float, default=0.0, help="render only N seconds")
    ap.add_argument("--fps", type=int, default=film.FPS)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "video_noaudio.mp4"))
    ap.add_argument("--audio-only", action="store_true")
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=0.0)
    ap.add_argument("--no-final", action="store_true")
    args = ap.parse_args()

    ff = ffmpeg_bin()
    S.W_IMG, S.H_IMG = W, H
    tl = film.build_timeline(os.path.join(ROOT, "src", "story.json"),
                             os.path.join(ROOT, "assets", "audio"), ff, fps=args.fps)
    total = args.test or tl.total
    print(f"timeline: {tl.total:.1f}s, {len(tl.segments)} shots, fps={args.fps}")

    audio = os.path.join(ROOT, "out", "mix.wav")
    t0 = time.time()
    film.build_audio(tl, ff, os.path.join(ROOT, "assets", "audio"), audio)
    print(f"audio mix: {time.time()-t0:.1f}s -> {audio}")
    clip = (args.t0, args.t1) if args.t1 > args.t0 else None
    subs = film.write_ass(tl, os.path.join(ROOT, "out", "subs_part.ass" if clip else "out" / "subs.ass") if False else
                          os.path.join(ROOT, "out", "subs_part.ass" if clip else "subs.ass"),
                          clip=clip, shift=args.t0)
    if args.audio_only:
        return

    # instantiate the shots we need, longest duration first
    need = {}
    for seg in tl.segments:
        need[seg.name] = max(need.get(seg.name, 0.0), seg.dur)
    cache = {}
    for name, d in need.items():
        cls = next(c for c in S.SHOT_CLASSES if c.name == name)
        t1 = time.time()
        cache[name] = cls(d + 0.1)
        print(f"built {name:13s} {time.time()-t1:5.2f}s")

    t_end_limit = args.t1 if args.t1 > args.t0 else total
    n_frames = int((t_end_limit - args.t0) * args.fps)
    cmd = [ff, "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(args.fps),
           "-i", "-",
           "-vf", f"ass={subs}:fontsdir={os.path.join(ROOT,'assets','fonts')}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    r = Renderer(W, H)

    def shot_frame(name, local):
        sh = cache[name]
        meshes, cam, lk = sh.frame(min(local, sh.dur - 1e-3))
        bg = r.sky_image(cam, lk.light, top=lk.sky_top, bottom=lk.sky_bot,
                         sun_glow=lk.sky_glow, sun_size=lk.sky_size, stars=lk.stars,
                         seed=int(local * 7) % 97)
        img = r.render(meshes, cam, lk.light, background=bg)
        return post(img, seed=int(local * 30), **lk.post)

    segs = tl.segments
    starts = [s.start for s in segs]
    t_start = time.time()
    for fi in range(n_frames):
        t = args.t0 + fi / args.fps
        i = int(np.searchsorted(starts, t, side="right") - 1)
        i = max(0, min(len(segs) - 1, i))
        seg = segs[i]
        img = shot_frame(seg.name, t - seg.start)
        # dissolve into the next shot
        if i + 1 < len(segs):
            X = film.DISSOLVE
            if t > seg.end - X:
                nxt = segs[i + 1]
                w_ = smoothstep((t - (seg.end - X)) / X)
                img2 = shot_frame(nxt.name, t - (seg.end - X))
                img = Image.blend(img, img2, w_)
        proc.stdin.write(img.tobytes())
        if fi % 120 == 0:
            el = time.time() - t_start
            eta = el / (fi + 1) * (n_frames - fi - 1)
            print(f"frame {fi}/{n_frames}  {el:6.1f}s elapsed  eta {eta/60:5.1f} min",
                  flush=True)
    proc.stdin.close()
    err = proc.stderr.read().decode()[-3000:]
    proc.wait()
    if proc.returncode != 0:
        print("ffmpeg failed:", err)
        sys.exit(1)
    print(f"video: {args.out}  ({time.time()-t_start:.0f}s)")

    if args.no_final or args.test:
        return
    final = os.path.join(ROOT, "out", "rahasia_tembok_besar_china.mp4")
    subprocess.run([ff, "-y", "-v", "error", "-i", args.out, "-i", audio,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-af", "loudnorm=I=-14:TP=-1.2:LRA=9",
                    "-shortest", "-movflags", "+faststart", final], check=True)
    print("final:", final)


if __name__ == "__main__":
    main()
