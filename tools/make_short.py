#!/usr/bin/env python3
"""make_short.py — memotong video sumber apa pun menjadi YouTube Short 9:16.

Dipakai untuk alur "ambil video asli -> klip bagian pentingnya -> jadikan Short".

Fitur
-----
* Ambil satu atau beberapa segmen (``--segments 12.5-19,44-58``) lalu disambung.
* Reframe otomatis ke 1080x1920:
  - ``cover``  : zoom + crop (default, paling sinematik)
  - ``blur``   : video utuh di tengah, latar belakang versi blur (aman untuk
                 footage yang objeknya tersebar)
  - ``topfit`` : video ditempel di atas, sisanya ruang untuk teks besar
* Voice-over ditempel di atas audio asli yang otomatis di-duck.
* Caption/teks on-screen dari file JSON (burn-in, gaya Shorts).
* Hard-limit 180 detik sesuai syarat YouTube Shorts.

Contoh
------
    python3 tools/make_short.py \
        --src raw/sumber.mp4 \
        --segments 8.2-17.0,31.5-44.0 \
        --mode cover \
        --vo audio/vo.mp3 \
        --captions script/captions.json \
        --music audio/bed.wav \
        --out output/short.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile

W, H, FPS = 1080, 1920, 30
MAX_SECONDS = 180.0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FONT = os.path.join(
    ROOT, "projects", "timnas-final-2026", "assets", "fonts", "Roboto-Black.ttf"
)


# --------------------------------------------------------------------------- #
# util
# --------------------------------------------------------------------------- #
def ffmpeg_bin() -> str:
    env = os.environ.get("FFMPEG")
    if env and os.path.exists(env):
        return env
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    for cand in ("/tmp/vid/lib/python3.11/site-packages/imageio_ffmpeg/binaries/"
                 "ffmpeg-linux-x86_64-v7.0.2",
                 os.path.expanduser("~/.local/bin/ffmpeg"),
                 "ffmpeg"):
        if cand == "ffmpeg" or os.path.exists(cand):
            return cand
    raise SystemExit("ffmpeg tidak ditemukan — jalankan: source tools/ffmpeg_setup.sh")


FFMPEG = ffmpeg_bin()
FFPROBE = FFMPEG.replace("ffmpeg", "ffprobe")


def run(cmd: list[str], quiet: bool = True) -> None:
    if not quiet:
        print("+", " ".join(shlex.quote(c) for c in cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-4000:])
        raise SystemExit(f"ffmpeg gagal (exit {p.returncode})")


def probe_duration(path: str) -> float:
    """Durasi media, tanpa bergantung pada ffprobe (tidak ikut di imageio-ffmpeg)."""
    p = subprocess.run([FFMPEG, "-hide_banner", "-i", path], capture_output=True, text=True)
    for line in p.stderr.splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            hms = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = hms.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise SystemExit(f"tidak bisa membaca durasi: {path}")


def has_audio(path: str) -> bool:
    p = subprocess.run([FFMPEG, "-hide_banner", "-i", path], capture_output=True, text=True)
    return "Audio:" in p.stderr


def parse_segments(spec: str, src_dur: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            raise SystemExit(f"format segmen salah: {chunk!r} (pakai mulai-selesai)")
        a, b = chunk.split("-", 1)
        start, end = float(a), float(b)
        end = min(end, src_dur)
        if end <= start:
            raise SystemExit(f"segmen tidak valid: {chunk}")
        out.append((start, end))
    if not out:
        out = [(0.0, min(src_dur, 55.0))]
    return out


# --------------------------------------------------------------------------- #
# reframe 9:16
# --------------------------------------------------------------------------- #
def reframe_filter(mode: str, pan: str) -> str:
    if mode == "cover":
        xmap = {"center": "(iw-ow)/2", "left": "0", "right": "iw-ow"}
        x = xmap.get(pan, "(iw-ow)/2")
        return (
            f"scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H}:{x}:(ih-oh)/2,setsar=1"
        )
    if mode == "blur":
        return (
            f"split=2[bg][fg];"
            f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},gblur=sigma=42,eq=brightness=-0.12[bgb];"
            f"[fg]scale={W}:-2[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1"
        )
    if mode == "topfit":
        return (
            f"split=2[bg][fg];"
            f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},gblur=sigma=42,eq=brightness=-0.3[bgb];"
            f"[fg]scale={W}:-2[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:360,setsar=1"
        )
    raise SystemExit(f"mode tidak dikenal: {mode}")


# --------------------------------------------------------------------------- #
# caption
# --------------------------------------------------------------------------- #
def esc(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace("%", r"\%")
    )


def caption_filters(captions: list[dict], font: str) -> str:
    """captions: [{"t": 0.0, "d": 2.5, "text": "BARIS 1|BARIS 2", "pos": "bottom"}]"""
    parts: list[str] = []
    for c in captions:
        t0 = float(c["t"])
        t1 = t0 + float(c.get("d", 2.5))
        lines = str(c["text"]).split("|")
        pos = c.get("pos", "bottom")
        size = int(c.get("size", 84))
        color = c.get("color", "white")
        gap = int(size * 1.22)
        n = len(lines)
        base = {"bottom": H - 460, "center": H // 2 - (n * gap) // 2, "top": 300}[pos]
        for i, line in enumerate(lines):
            y = base + i * gap
            parts.append(
                f"drawtext=fontfile='{font}':text='{esc(line.upper())}'"
                f":fontsize={size}:fontcolor={color}:borderw=7:bordercolor=black@0.92"
                f":shadowx=0:shadowy=5:shadowcolor=black@0.55"
                f":x=(w-text_w)/2:y={y}"
                f":enable='between(t,{t0:.3f},{t1:.3f})'"
            )
    return ",".join(parts)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Potong video jadi YouTube Short 9:16")
    ap.add_argument("--src", required=True, help="video sumber")
    ap.add_argument("--segments", default="", help="mis. 8.2-17,31.5-44")
    ap.add_argument("--mode", default="cover", choices=["cover", "blur", "topfit"])
    ap.add_argument("--pan", default="center", choices=["center", "left", "right"])
    ap.add_argument("--vo", default="", help="file voice-over (mp3/wav)")
    ap.add_argument("--music", default="", help="bed musik")
    ap.add_argument("--captions", default="", help="file JSON caption")
    ap.add_argument("--font", default=DEFAULT_FONT)
    ap.add_argument("--src-volume", type=float, default=0.18,
                    help="volume audio asli saat ada VO (0 = bisukan)")
    ap.add_argument("--music-volume", type=float, default=0.14)
    ap.add_argument("--speed", type=float, default=1.0, help="mis. 1.15 untuk sedikit dipercepat")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src_dur = probe_duration(a.src)
    segs = parse_segments(a.segments, src_dur)
    total = sum(e - s for s, e in segs) / max(a.speed, 0.01)
    if total > MAX_SECONDS:
        raise SystemExit(
            f"total {total:.1f}s melebihi batas Shorts {MAX_SECONDS:.0f}s — "
            f"persempit --segments"
        )

    src_audio = has_audio(a.src)
    tmp = tempfile.mkdtemp(prefix="short_")
    pieces: list[str] = []

    vf = reframe_filter(a.mode, a.pan)
    print(f"sumber  : {a.src} ({src_dur:.2f}s, audio={'ya' if src_audio else 'tidak'})")
    print(f"segmen  : {segs}  -> total {total:.2f}s")

    for i, (s, e) in enumerate(segs):
        piece = os.path.join(tmp, f"p{i:02d}.mp4")
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
               "-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", a.src,
               "-vf", f"{vf},fps={FPS}",
               "-c:v", "libx264", "-preset", "medium", "-crf", "19",
               "-pix_fmt", "yuv420p"]
        if src_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
        else:
            cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                    "-shortest", "-c:a", "aac", "-b:a", "192k"]
        cmd += [piece]
        run(cmd)
        pieces.append(piece)
        print(f"  klip {i+1}/{len(segs)}  {s:.2f}s -> {e:.2f}s")

    # sambung
    listfile = os.path.join(tmp, "list.txt")
    with open(listfile, "w") as fh:
        for p in pieces:
            fh.write(f"file '{p}'\n")
    joined = os.path.join(tmp, "joined.mp4")
    run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "concat", "-safe", "0", "-i", listfile, "-c", "copy", joined])

    # susun filter akhir + audio
    inputs = [joined]
    vchain = []
    if abs(a.speed - 1.0) > 0.001:
        vchain.append(f"setpts={1/a.speed:.6f}*PTS")
    if a.captions:
        with open(a.captions) as fh:
            caps = json.load(fh)
        vchain.append(caption_filters(caps, a.font))
    vchain.append("format=yuv420p")

    amaps = []
    idx = 1
    filt = []
    if a.vo:
        inputs.append(a.vo)
        filt.append(f"[{idx}:a]aresample=48000,volume=1.0[vo]")
        amaps.append("[vo]")
        idx += 1
    if a.music:
        inputs.append(a.music)
        filt.append(f"[{idx}:a]aresample=48000,volume={a.music_volume}[mus]")
        amaps.append("[mus]")
        idx += 1

    src_vol = a.src_volume if a.vo else 1.0
    tempo = f",atempo={a.speed}" if abs(a.speed - 1.0) > 0.001 else ""
    filt.append(f"[0:a]aresample=48000{tempo},volume={src_vol}[src]")
    amaps.insert(0, "[src]")

    if len(amaps) > 1:
        filt.append(f"{''.join(amaps)}amix=inputs={len(amaps)}:duration=longest:"
                    f"normalize=0,alimiter=limit=0.95,aresample=48000[aout]")
    else:
        filt.append("[src]alimiter=limit=0.95[aout]")

    filt.append(f"[0:v]{','.join(vchain)}[vout]")

    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y"]
    for p in inputs:
        cmd += ["-i", p]
    cmd += ["-filter_complex", ";".join(filt),
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-profile:v", "high", "-level", "4.1",
            "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
            "-g", str(FPS * 2), "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-r", str(FPS), a.out]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    run(cmd)

    dur = probe_duration(a.out)
    size = os.path.getsize(a.out) / 1e6
    print(f"\nselesai : {a.out}")
    print(f"          {W}x{H} @ {FPS}fps · {dur:.2f}s · {size:.1f} MB"
          f" · {'OK <180s' if dur <= MAX_SECONDS else 'TERLALU PANJANG'}")


if __name__ == "__main__":
    main()
