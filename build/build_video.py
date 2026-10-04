#!/usr/bin/env python3
"""Rakit video stickman (short / long) dari gambar + voice-over.

Pakai: python3 build/build_video.py short
       python3 build/build_video.py long
"""
import json, os, subprocess, sys
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
FP = FF.replace("ffmpeg-", "ffprobe-")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dur(path):
    out = subprocess.run([FF, "-i", path, "-f", "null", "-"], capture_output=True, text=True).stderr
    # parse last time=
    t = [l for l in out.split() if l.startswith("time=")]
    hh, mm, ss = t[-1][5:].split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def build(mode):
    cfg = json.load(open(os.path.join(ROOT, "content", f"{mode}.json")))
    W, H = (1080, 1920) if mode == "short" else (1920, 1080)
    scenes = cfg["scenes"]
    adir = os.path.join(ROOT, "assets", mode)
    work = os.path.join(ROOT, "build", "tmp_" + mode)
    os.makedirs(work, exist_ok=True)

    XF = 0.6           # durasi transisi crossfade
    HEAD, TAIL = 0.2, 0.45   # jeda napas sebelum & sesudah VO tiap scene

    clips, durs = [], []
    for i, sc in enumerate(scenes):
        img = os.path.join(adir, sc["image"])
        vo = os.path.join(adir, sc["vo"])
        d = dur(vo) + HEAD + TAIL + (XF if i < len(scenes) - 1 else 0)
        out = os.path.join(work, f"c{i}.mp4")
        # ken burns: selang-seling zoom in / zoom out + pan halus
        zdir = "+" if i % 2 == 0 else "-"
        z = ("min(zoom+0.0012,1.20)" if i % 2 == 0 else "max(1.20-0.0012*on,1.001)")
        fps = 30
        vf = (
            f"scale={W*4}:{H*4}:force_original_aspect_ratio=increase,"
            f"crop={W*4}:{H*4},"
            f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={int(d*fps)}:s={W}x{H}:fps={fps},"
            f"format=yuv420p"
        )
        subprocess.run([FF, "-y", "-loop", "1", "-i", img, "-t", f"{d}", "-vf", vf,
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-r", "30", out],
                       check=True, capture_output=True)
        clips.append(out)
        durs.append(d)

    # rantai xfade
    inputs = []
    for c in clips:
        inputs += ["-i", c]
    fc, prev, off = [], "0:v", 0.0
    for i in range(1, len(clips)):
        off += durs[i - 1] - XF
        lbl = f"v{i}"
        fc.append(f"[{prev}][{i}:v]xfade=transition={'fade' if i%2 else 'smoothleft'}:duration={XF}:offset={off:.3f}[{lbl}]")
        prev = lbl
    vmap = f"[{prev}]" if len(clips) > 1 else "[0:v]"
    silent = os.path.join(work, "video_silent.mp4")
    subprocess.run([FF, "-y", *inputs, "-filter_complex", ";".join(fc),
                    "-map", vmap, "-c:v", "libx264", "-preset", "medium", "-crf", "19",
                    "-pix_fmt", "yuv420p", silent], check=True, capture_output=True)
    total = sum(durs) - XF * (len(clips) - 1)

    # susun trek voice-over pada posisi yang sama dengan scene-nya
    ai, af, amix = [], [], []
    t = 0.0
    for i, sc in enumerate(scenes):
        start = t + HEAD
        ai += ["-i", os.path.join(adir, sc["vo"])]
        af.append(f"[{i}:a]adelay={int(start*1000)}|{int(start*1000)},aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a{i}]")
        amix.append(f"[a{i}]")
        t += durs[i] - (XF if i < len(scenes) - 1 else 0)
    vo_mix = os.path.join(work, "vo.wav")
    subprocess.run([FF, "-y", *ai, "-filter_complex",
                    ";".join(af) + ";" + "".join(amix) + f"amix=inputs={len(scenes)}:normalize=0[out]",
                    "-map", "[out]", "-t", f"{total}", vo_mix], check=True, capture_output=True)

    # bed musik ambient lembut (sintesis) + sidechain-ish volume rendah
    bed = os.path.join(work, "bed.wav")
    subprocess.run([FF, "-y", "-f", "lavfi", "-i",
                    f"sine=frequency=110:duration={total}", "-f", "lavfi", "-i",
                    f"sine=frequency=164.81:duration={total}", "-f", "lavfi", "-i",
                    f"sine=frequency=220:duration={total}",
                    "-filter_complex",
                    "[0:a]tremolo=f=0.4:d=0.7[t0];[1:a]tremolo=f=0.27:d=0.6[t1];"
                    "[t0][t1][2:a]amix=inputs=3:normalize=1,lowpass=f=900,"
                    f"volume=-26dB,afade=t=in:d=1.5,afade=t=out:st={max(total-2,0):.2f}:d=2[b]",
                    "-map", "[b]", bed], check=True, capture_output=True)

    final = os.path.join(ROOT, "output", f"{cfg['slug']}.mp4")
    os.makedirs(os.path.dirname(final), exist_ok=True)
    subprocess.run([FF, "-y", "-i", silent, "-i", vo_mix, "-i", bed, "-filter_complex",
                    "[1:a]loudnorm=I=-14:TP=-1.5:LRA=11[v];[2:a]volume=1[m];[v][m]amix=inputs=2:normalize=0,alimiter=limit=0.95[a]",
                    "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                    "-shortest", final], check=True, capture_output=True)
    print("OK:", final, f"{total:.1f}s")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "short")
