#!/usr/bin/env python3
"""sticktomyplan — perakit video v2.

Fitur: intro berbrand, ken burns dinamis, transisi bervariasi (9 jenis),
whoosh + impact SFX tersintesis, bed musik tegang, outro berbrand,
loudness -14 LUFS. Tanpa subtitle — teks hanya wordmark channel di intro/outro.

Pakai: python3 build/build_v2.py content/short2.json
"""
import json, os, subprocess, sys
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BRAND = "sticktomyplan"

TRANS = ["fadeblack", "smoothleft", "circleopen", "wiperight", "slideup",
         "dissolve", "smoothright", "radial", "wipeleft", "fadewhite", "circlecrop"]


def wordmark(work, W, H):
    """Logo channel kecil di sudut kiri atas (branding, bukan subtitle)."""
    from PIL import Image, ImageDraw, ImageFont
    p = os.path.join(work, f"wm_{W}x{H}.png")
    if os.path.exists(p):
        return p
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    size = max(18, int(W * 0.030))
    f = ImageFont.truetype(FONT, size)
    x, y = int(W * 0.045), int(H * 0.035)
    bb = d.textbbox((0, 0), BRAND, font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = int(size * 0.55)
    d.rounded_rectangle([x - pad, y - pad, x + tw + pad, y + th + pad + int(size * 0.35)],
                        radius=int(size * 0.5), fill=(12, 12, 12, 150))
    d.rectangle([x - pad + int(size * 0.18), y - pad + int(size * 0.25),
                 x - pad + int(size * 0.18) + max(3, int(size * 0.12)),
                 y + th + pad - int(size * 0.05)], fill=(226, 72, 40, 230))
    d.text((x + int(size * 0.45), y), BRAND, font=f, fill=(255, 255, 255, 232))
    im.save(p)
    return p


def run(args):
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode:
        print(p.stderr[-3000:])
        raise SystemExit(1)


def dur(path):
    out = subprocess.run([FF, "-i", path, "-f", "null", "-"], capture_output=True, text=True).stderr
    t = [l for l in out.split() if l.startswith("time=")][-1][5:]
    hh, mm, ss = t.split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def build(cfg_path):
    cfg = json.load(open(cfg_path))
    W, H = (1080, 1920) if cfg["format"] == "9:16" else (1920, 1080)
    adir = os.path.join(ROOT, cfg["assets_dir"])
    work = os.path.join(ROOT, "build", "tmp_" + cfg["slug"])
    os.makedirs(work, exist_ok=True)
    FPS, XF = int(cfg.get("fps", 30)), 0.5
    HEAD, TAIL = 0.18, 0.38

    scenes = cfg["scenes"]
    clips, durs = [], []
    for i, sc in enumerate(scenes):
        img = os.path.join(adir, sc["image"])
        vo = os.path.join(adir, sc["vo"]) if sc.get("vo") else None
        base = dur(vo) + HEAD + TAIL if vo else sc.get("dur", 2.0)
        d = base + sc.get("pad", 0) + (XF if i < len(scenes) - 1 else 0)
        n = int(d * FPS)
        out = os.path.join(work, f"c{i:02d}.mp4")
        mv = sc.get("move", "in" if i % 2 == 0 else "out")
        if mv == "in":
            z, x, y = "min(zoom+0.0010,1.22)", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
        elif mv == "out":
            z, x, y = f"max(1.22-0.0010*on,1.001)", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
        elif mv == "left":
            z, x, y = "1.18", f"(iw-iw/zoom)*(on/{n})", "ih/2-(ih/zoom/2)"
        elif mv == "right":
            z, x, y = "1.18", f"(iw-iw/zoom)*(1-on/{n})", "ih/2-(ih/zoom/2)"
        elif mv == "up":
            z, x, y = "1.18", "iw/2-(iw/zoom/2)", f"(ih-ih/zoom)*(1-on/{n})"
        else:  # punch: zoom cepat lalu tahan
            z, x, y = "min(zoom+0.0030,1.30)", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"

        vf = (f"scale={W*3}:{H*3}:force_original_aspect_ratio=increase,crop={W*3}:{H*3},"
              f"zoompan=z='{z}':x='{x}':y='{y}':d={n}:s={W}x{H}:fps={FPS},"
              f"eq=contrast=1.06:saturation=0.9,"
              f"vignette=PI/5,format=yuv420p")
        mark = wordmark(work, W, H)
        fc_b = (f"[0:v]{vf}[bg];[1:v]format=rgba,fade=t=in:st=0:d=0.6:alpha=1[wm];"
                f"[bg][wm]overlay=0:0:format=auto,format=yuv420p[v]")
        run([FF, "-y", "-loop", "1", "-i", img, "-loop", "1", "-i", mark, "-t", f"{d}",
             "-filter_complex", fc_b, "-map", "[v]",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-r", str(FPS), out])
        clips.append(out)
        durs.append(d)

    # --- rantai transisi bervariasi (bertahap per kelompok agar hemat memori) ---
    def xfade_group(files, lens, transs, tag):
        """Gabungkan beberapa klip dengan xfade; kembalikan (path, durasi)."""
        if len(files) == 1:
            return files[0], lens[0]
        inputs = []
        for f in files:
            inputs += ["-i", f]
        fc, prev, off = [], "0:v", 0.0
        for i in range(1, len(files)):
            off += lens[i - 1] - XF
            fc.append(f"[{prev}][{i}:v]xfade=transition={transs[i]}:duration={XF}:offset={off:.3f}[v{i}]")
            prev = f"v{i}"
        out = os.path.join(work, f"g_{tag}.mp4")
        run([FF, "-y", *inputs, "-filter_complex", ";".join(fc), "-map", f"[{prev}]",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", out])
        return out, sum(lens) - XF * (len(files) - 1)

    all_trans = [None] + [scenes[i].get("trans") or TRANS[(i - 1) % len(TRANS)] for i in range(1, len(clips))]
    G = 4
    groups, gl, gt = [], [], []
    for a in range(0, len(clips), G):
        b = min(a + G, len(clips))
        f, d_ = xfade_group(clips[a:b], durs[a:b], [None] + all_trans[a + 1:b], f"{a:02d}")
        groups.append(f); gl.append(d_); gt.append(all_trans[a] if a > 0 else None)
    silent, _ = xfade_group(groups, gl, gt, "final")
    os.replace(silent, os.path.join(work, "silent.mp4"))
    silent = os.path.join(work, "silent.mp4")
    offsets, _acc = [], 0.0
    for i in range(1, len(clips)):
        _acc += durs[i - 1] - XF
        offsets.append(_acc)
    total = sum(durs) - XF * (len(clips) - 1)

    # --- voice-over diposisikan per scene ---
    ai, af, mix, t, k = [], [], [], 0.0, 0
    for i, sc in enumerate(scenes):
        if sc.get("vo"):
            start = t + HEAD
            ai += ["-i", os.path.join(adir, sc["vo"])]
            af.append(f"[{k}:a]apad=pad_dur=0.4,adelay={int(start*1000)}|{int(start*1000)},"
                      f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a{k}]")
            mix.append(f"[a{k}]")
            k += 1
        t += durs[i] - (XF if i < len(scenes) - 1 else 0)
    vo = os.path.join(work, "vo.wav")
    run([FF, "-y", *ai, "-filter_complex", ";".join(af) + ";" + "".join(mix) +
         f"amix=inputs={k}:normalize=0[o]", "-map", "[o]", "-t", f"{total}", vo])

    # --- SFX: whoosh tiap transisi + impact drum di scene tertentu ---
    sfx_src, sfx_f, sfx_mix = [], [], []
    k = 0
    for i, o in enumerate(offsets):
        sfx_src += ["-f", "lavfi", "-i", "anoisesrc=d=0.6:c=pink:a=0.35"]
        sfx_f.append(f"[{k}:a]highpass=f=400,volume='min(1,max(0,(t/0.25)))*max(0,1-(t-0.25)/0.3)':eval=frame,"
                     f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
                     f"adelay={int(o*1000)}|{int(o*1000)},volume=-19dB[s{k}]")
        sfx_mix.append(f"[s{k}]")
        k += 1
    for h in cfg.get("impacts", []):
        sfx_src += ["-f", "lavfi", "-i", "sine=frequency=62:duration=0.5"]
        sfx_f.append(f"[{k}:a]volume='max(0,1-t/0.45)':eval=frame,"
                     f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
                     f"adelay={int(h*1000)}|{int(h*1000)},volume=-8dB[s{k}]")
        sfx_mix.append(f"[s{k}]")
        k += 1
    sfx = os.path.join(work, "sfx.wav")
    run([FF, "-y", *sfx_src, "-filter_complex", ";".join(sfx_f) + ";" + "".join(sfx_mix) +
         f"amix=inputs={k}:normalize=0[o]", "-map", "[o]", "-t", f"{total}", sfx])

    # --- bed musik tegang (drone + detak) ---
    bed = os.path.join(work, "bed.wav")
    run([FF, "-y",
         "-f", "lavfi", "-i", f"sine=frequency=55:duration={total}",
         "-f", "lavfi", "-i", f"sine=frequency=82.4:duration={total}",
         "-f", "lavfi", "-i", f"sine=frequency=110:duration={total}",
         "-filter_complex",
         "[0:a]tremolo=f=0.33:d=0.6[t0];[1:a]tremolo=f=0.22:d=0.5[t1];"
         "[t0][t1][2:a]amix=inputs=3:normalize=1,lowpass=f=700,volume=-24dB,"
         f"afade=t=in:d=1.2,afade=t=out:st={max(total-2.5,0):.2f}:d=2.5[b]",
         "-map", "[b]", bed])

    out = os.path.join(ROOT, "output", cfg["slug"] + ".mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    run([FF, "-y", "-i", silent, "-i", vo, "-i", sfx, "-i", bed, "-filter_complex",
         "[1:a]loudnorm=I=-14:TP=-1.5:LRA=11[v];[v][2:a][3:a]amix=inputs=3:normalize=0,"
         "alimiter=limit=0.95[a]",
         "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
         "-shortest", out])
    print(f"OK: {out}  {total:.1f}s  {len(scenes)} scene")


if __name__ == "__main__":
    build(os.path.join(ROOT, sys.argv[1]))
