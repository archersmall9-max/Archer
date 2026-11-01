"""
render.py — renders the film: 3D frames -> h264, voice-over + score -> AAC, burned captions.

Pipeline per frame (see gfx.py):
    shot.frame(t) -> geometry + camera + look
    Renderer.sky_image -> procedural sky            (evaluated at 1/4 res, cubic upscale)
    Renderer.render    -> painter's-algorithm raster (supersampled 1.5x)
    gfx.resolve        -> INTER_AREA resolve to 1080x1920 (true anti-aliasing)
    gfx.grade          -> bloom/halation, filmic LUT, clarity, vignette, grain

Usage:
    python render.py                          # whole film, 1080x1920, 60 fps
    python render.py --test 6                 # first 6 seconds only
    python render.py --f0 3000 --f1 4500 --out ../out/part.mp4 --no-final
    python render.py --audio-only             # rebuild out/mix.wav only
    python ../make.py                         # parallel render + concat + mux (film master)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import zlib

import cv2
import numpy as np

import film
import gfx
import shadow
import shots as S
from engine3d import Renderer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 1080, 1920
SS = 1.6             # supersampling factor (>=1.5 removes polygonal edge shimmer)
CRF = 17
PRESET = "faster"


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
    ap.add_argument("--no-audio", action="store_true", help="reuse the existing out/mix.wav")
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=0.0)
    ap.add_argument("--f0", type=int, default=-1, help="first frame index (inclusive)")
    ap.add_argument("--f1", type=int, default=-1, help="last frame index (exclusive)")
    ap.add_argument("--ss", type=float, default=SS, help="super-sampling factor")
    ap.add_argument("--shadow", type=float, default=0.72,
                    help="baked sun-shadow strength for the static world (0=off)")
    ap.add_argument("--trail", type=float, default=0.35,
                    help="shutter trail: blend with the previous frame (0=off)")
    ap.add_argument("--ao", type=float, default=0.0, help="contact occlusion strength")
    ap.add_argument("--crf", type=int, default=CRF)
    ap.add_argument("--preset", default=PRESET)
    ap.add_argument("--no-final", action="store_true")
    ap.add_argument("--no-subs", action="store_true",
                    help="skip the caption burn (make.py burns them once on the joined master)")
    args = ap.parse_args()

    ff = ffmpeg_bin()
    S.W_IMG, S.H_IMG = W, H
    tl = film.build_timeline(os.path.join(ROOT, "src", "story.json"),
                             os.path.join(ROOT, "assets", "audio"), ff, fps=args.fps)
    print(f"timeline: {tl.total:.2f}s, {len(tl.segments)} shots, fps={args.fps}, "
          f"{int(tl.total * args.fps)} frames", flush=True)

    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    audio = os.path.join(ROOT, "out", "mix.wav")
    if not (args.no_audio and os.path.exists(audio)):
        t0 = time.time()
        film.build_audio(tl, ff, os.path.join(ROOT, "assets", "audio"), audio)
        print(f"audio mix: {time.time() - t0:.1f}s -> {audio}", flush=True)
    if args.audio_only:
        return

    # ---- frame range --------------------------------------------------------
    if args.f0 >= 0:
        f0, f1 = args.f0, args.f1
    elif args.t1 > args.t0:
        f0, f1 = int(round(args.t0 * args.fps)), int(round(args.t1 * args.fps))
    else:
        f0, f1 = 0, int(round((args.test or tl.total) * args.fps))
    n_frames = f1 - f0
    t_start, t_end = f0 / args.fps, f1 / args.fps

    if args.no_subs:
        subs = None
    else:
        subs = film.write_ass(
            tl, os.path.join(ROOT, "out", f"subs_{f0:05d}.ass"),
            clip=(t_start, t_end) if (f0 or f1 < int(tl.total * args.fps) - 1) else None,
            shift=t_start)

    # ---- shots --------------------------------------------------------------
    need = {}
    for seg in tl.segments:
        need[seg.name] = max(need.get(seg.name, 0.0), seg.dur)
    by_name = {c.name: c for c in S.SHOT_CLASSES}
    cache = {}
    for name, d in need.items():
        cache[name] = by_name[name](d + 0.5)

    # ---- bake sun shadows (sun is constant inside a shot) -------------------
    if args.shadow > 0:
        t_b = time.time()
        for name, sh in cache.items():
            try:
                _, _, lk_b = sh.frame(0.42 * sh.dur)
            except Exception:
                continue
            statics = []
            w = getattr(sh, "world", None)
            if w is not None:
                statics += list(w.static)
            for attr in ("meshes", "ribbons", "grains"):
                if attr == "grains":
                    continue
                extra = getattr(sh, attr, None)
                if isinstance(extra, list):
                    statics += [m for m in extra if m is not None]
            if statics:
                shadow.bake(statics, lk_b.light.sun.d, strength=args.shadow)
        print(f"sun shadows baked in {time.time() - t_b:.1f}s", flush=True)

    r = Renderer(W, H, ssaa=args.ss)
    segs = tl.segments
    starts = [s.start for s in segs]
    last = len(segs) - 1

    def raw_frame(name, local):
        sh = cache[name]
        meshes, cam, lk = sh.frame(min(local, sh.dur - 1e-3))
        bg = r.sky_image(cam, lk.light, top=lk.sky_top, bottom=lk.sky_bot,
                         sun_glow=lk.sky_glow, sun_size=lk.sky_size, stars=lk.stars,
                         seed=zlib.crc32(name.encode()) % 9973, time=local)
        img = r.render(meshes, cam, lk.light, background=bg, ao=args.ao)
        return gfx.resolve(np.asarray(img), W, H), lk

    # ---- encoder ------------------------------------------------------------
    cmd = [ff, "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(args.fps),
           "-i", "-"]
    if subs:
        cmd += ["-vf", f"ass={subs}:fontsdir={os.path.join(ROOT, 'assets', 'fonts')}"]
    cmd += ["-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
           "-tune", "animation", "-profile:v", "high", "-g", str(args.fps * 2),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    t0 = time.time()
    trail = max(0.0, min(0.9, args.trail))
    prev = None
    try:
        for k in range(n_frames):
            gi = f0 + k
            t = gi / args.fps
            i = int(np.searchsorted(starts, t, side="right") - 1)
            i = max(0, min(last, i))
            seg = segs[i]
            img, lk = raw_frame(seg.name, t - seg.start)
            # dissolve into the next shot
            if i < last and t > seg.end - film.DISSOLVE:
                nxt = segs[i + 1]
                w_ = smoothstep((t - (seg.end - film.DISSOLVE)) / film.DISSOLVE)
                img2, _ = raw_frame(nxt.name, t - (seg.end - film.DISSOLVE))
                img = cv2.addWeighted(img, 1.0 - w_, img2, w_, 0.0)
            # shutter trail: a 180-degree-ish shutter, so fast moves read as motion
            # instead of a sequence of stills (and it calms any residual stepping)
            if trail > 0:
                if prev is None:
                    if f0 > 0:                         # prime a part with its real
                        tp = (gi - 1) / args.fps        # predecessor frame
                        ip = int(np.searchsorted(starts, tp, side="right") - 1)
                        ip = max(0, min(last, ip))
                        prev, _ = raw_frame(segs[ip].name, tp - segs[ip].start)
                    else:
                        prev = img
                img = cv2.addWeighted(prev, trail, img, 1.0 - trail, 0.0)
                prev = img
            q = gfx.grade(img, seed=gi, **lk.post)
            proc.stdin.write(np.ascontiguousarray(q).tobytes())
            if k % 120 == 0:
                el = time.time() - t0
                eta = el / (k + 1) * (n_frames - k - 1)
                print(f"frame {gi}/{f1}  {el:6.1f}s  eta {eta / 60:5.1f} min  "
                      f"({(k + 1) / max(el, 1e-9):4.2f} fps)", flush=True)
    finally:
        if proc.stdin:
            proc.stdin.close()
    err = proc.stderr.read().decode()[-4000:]
    proc.wait()
    if proc.returncode != 0:
        print("ffmpeg failed:\n", err)
        sys.exit(1)
    print(f"video: {args.out}  ({time.time() - t0:.0f}s)", flush=True)

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
