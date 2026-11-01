"""
film.py — timeline, procedural score, voice-over bus and ASS captions.

Timeline is driven by the real voice-over durations, so picture cuts always land on
narration beats. Shots inside a scene dissolve into each other.
"""
from __future__ import annotations

import json
import subprocess
import wave
from dataclasses import dataclass, field

import numpy as np

SR = 48000

# scene id -> [(shot class name, weight)]
SCENE_SHOTS = {
    "01": [("hero", 1.0)],
    "02": [("many", 1.0)],
    "03": [("orbit", 0.5), ("camo", 0.5)],
    "04": [("workers", 0.55), ("chisel", 0.45)],
    "05": [("memorial", 1.0)],
    "06": [("rice", 0.52), ("cutaway", 0.48)],
    "07": [("gate", 0.55), ("barracks", 0.45)],
    "08": [("beacon_day", 0.5), ("beacon_night", 0.5)],
    "09": [("mongol", 0.5), ("betray", 0.5)],
    "10": [("ruin", 0.34), ("villager", 0.30), ("final", 0.36)],
}

LEAD_IN = 1.7
SCENE_GAP = 0.40
TAIL = 2.4
DISSOLVE = 0.55
FPS = 90


@dataclass
class Segment:
    name: str
    start: float
    dur: float
    scene: str
    vo_start: float = 0.0
    vo_dur: float = 0.0
    narration: str = ""

    @property
    def end(self):
        return self.start + self.dur


@dataclass
class Timeline:
    segments: list = field(default_factory=list)
    scenes: list = field(default_factory=list)
    total: float = 0.0
    title: str = ""


def ffprobe_dur(path, ffmpeg_bin="ffmpeg"):
    out = subprocess.run([ffmpeg_bin, "-hide_banner", "-i", path],
                         capture_output=True, text=True)
    for tok in out.stderr.replace(",", " ").split():
        if tok.count(":") == 2 and tok[0].isdigit():
            try:
                h, m, s = tok.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
            except Exception:
                continue
    raise RuntimeError("cannot read duration of " + path)


def build_timeline(story_path, vo_dir, ffmpeg_bin, fps=FPS):
    story = json.load(open(story_path))
    scenes = story["scenes"]
    segs = []
    t = LEAD_IN
    for sc in scenes:
        vo = f"{vo_dir}/vo_{sc['id']}.mp3"
        dur = ffprobe_dur(vo, ffmpeg_bin)
        scene_len = dur + SCENE_GAP
        shots = SCENE_SHOTS[sc["id"]]
        wsum = sum(w for _, w in shots)
        cursor = t
        for k, (name, w) in enumerate(shots):
            d = scene_len * w / wsum
            segs.append(Segment(name=name, start=cursor, dur=d, scene=sc["id"],
                                vo_start=t + 0.10 if k == 0 else 0.0,
                                vo_dur=dur if k == 0 else 0.0,
                                narration=sc["narration"]))
            cursor += d
        t += scene_len
    total = t + TAIL
    tl = Timeline(segments=segs, total=total, title=story["title"])
    tl.scenes = scenes
    return tl


# ------------------------------------------------------------------ audio bus
def read_mp3(ffmpeg_bin, path, sr=SR):
    raw = subprocess.run(
        [ffmpeg_bin, "-v", "quiet", "-i", path, "-f", "f32le", "-ac", "1",
         "-ar", str(sr), "-"], capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def _env(x, sr):
    """Envelope follower: max filter over a window, then smoothing."""
    from numpy.lib.stride_tricks import sliding_window_view
    win = int(0.06 * sr)
    pad = np.pad(np.abs(x), (win, win), mode="edge")
    e = sliding_window_view(pad, 2 * win + 1).max(axis=-1)
    k = max(8, int(0.14 * sr))
    ker = np.hanning(k)
    ker /= ker.sum()
    return np.convolve(e, ker, mode="same")


def build_audio(tl: Timeline, ffmpeg_bin, vo_dir, out_wav, music_gain=0.205):
    n = int(tl.total * SR) + SR
    vo_bus = np.zeros(n, dtype=np.float32)
    for seg in tl.segments:
        if seg.vo_dur <= 0:
            continue
        x = read_mp3(ffmpeg_bin, f"{vo_dir}/vo_{seg.scene}.mp3")
        # gentle de-click + consistent level
        x = x / (np.max(np.abs(x)) + 1e-9)
        i0 = int(seg.vo_start * SR)
        vo_bus[i0:i0 + len(x)] += x
    vo_bus = vo_bus[: int(tl.total * SR)]

    music = score(tl.total)
    duck = _env(vo_bus, SR)
    duck = np.clip(duck / (duck.max() + 1e-9) * 1.6, 0, 1)
    music = music * (music_gain * (1.0 - 0.72 * duck))[:, None]

    vo = np.repeat(vo_bus[:, None], 2, axis=1) * 1.0
    mix = vo + music
    # soft limiter
    peak = np.max(np.abs(mix)) + 1e-9
    if peak > 0.98:
        mix = np.tanh(mix * (1.0 / peak) * 1.15) * 0.92
    # short fade in/out
    fi = int(0.25 * SR)
    mix[:fi] *= np.linspace(0, 1, fi)[:, None]
    fo = int(0.9 * SR)
    mix[-fo:] *= np.linspace(1, 0, fo)[:, None]

    data = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())
    return out_wav


# ------------------------------------------------------------------ score
def _note(freq, dur, sr=SR, harm=(1.0, 0.45, 0.22, 0.12), detune=0.15):
    n = int(dur * sr)
    t = np.arange(n) / sr
    sig = np.zeros(n, dtype=np.float32)
    for h, amp in enumerate(harm, start=1):
        sig += amp * np.sin(2 * np.pi * freq * h * t + np.sin(2 * np.pi * detune * t) * 0.6)
    return sig / (sum(harm) + 1e-9)


def score(total, sr=SR):
    """A slow, cinematic bed: drone chords, taiko hits, wind and shimmer."""
    n = int(total * sr)
    out = np.zeros((n, 2), dtype=np.float32)
    t = np.arange(n) / sr

    # chord bed ---------------------------------------------------------
    chords = [(55.0, [55.0, 82.41, 110.0, 164.81]),      # A minor
              (43.65, [43.65, 65.41, 87.31, 130.81]),    # F
              (65.41, [65.41, 98.0, 130.81, 196.0]),     # C
              (49.0, [49.0, 73.42, 98.0, 146.83])]       # G
    bar = 8.2
    pos = 0.0
    ci = 0
    while pos < total:
        root, freqs = chords[ci % len(chords)]
        d = min(bar + 1.4, total - pos)
        seg = np.zeros(int(d * sr), dtype=np.float32)
        for k, f in enumerate(freqs):
            amp = 0.55 if k == 0 else 0.30 / k
            seg += _note(f, d, harm=(1.0, 0.4, 0.18, 0.08)).astype(np.float32) * amp
        # slow swell
        m = len(seg)
        envl = np.sin(np.linspace(0, np.pi, m)) ** 0.6
        seg *= envl
        i0 = int(pos * sr)
        i1 = min(n, i0 + m)
        out[i0:i1, 0] += seg[: i1 - i0]
        out[i0:i1, 1] += seg[: i1 - i0] * 0.97
        pos += bar
        ci += 1

    # wind --------------------------------------------------------------
    rng = np.random.default_rng(4)
    noise = rng.standard_normal(n).astype(np.float32)
    kern = np.hanning(400).astype(np.float32)
    kern /= kern.sum()
    noise = np.convolve(noise, kern, mode="same")
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.055 * t + 0.7)
    out[:, 0] += noise * 0.055 * lfo
    out[:, 1] += noise * 0.055 * (1.0 - lfo * 0.5)

    # taiko-ish hits every 2 bars, soft heartbeat in between -------------
    def hit(at, amp=1.0, f0=96.0, f1=42.0, dur=0.9):
        m = int(dur * sr)
        tt = np.arange(m) / sr
        f = f1 + (f0 - f1) * np.exp(-tt * 26)
        body = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-tt * 5.5)
        click = rng.standard_normal(m) * np.exp(-tt * 60) * 0.25
        s = ((body + click) * amp).astype(np.float32)
        i0 = int(at * sr)
        i1 = min(n, i0 + m)
        if i1 > i0:
            out[i0:i1, 0] += s[: i1 - i0]
            out[i0:i1, 1] += s[: i1 - i0] * 0.94

    at = 2.0
    k = 0
    while at < total - 1.0:
        if k % 4 == 0:
            hit(at, amp=0.42)
        else:
            hit(at, amp=0.17, f0=72, f1=38, dur=0.5)
        at += bar / 2
        k += 1

    # shimmer -----------------------------------------------------------
    for at in np.arange(6.0, total - 3.0, 13.5):
        m = int(2.6 * sr)
        tt = np.arange(m) / sr
        f = 880.0 * rng.choice([1.0, 1.5, 2.0])
        s = (np.sin(2 * np.pi * f * tt) * np.exp(-tt * 1.6) * 0.05).astype(np.float32)
        i0 = int(at * sr)
        i1 = min(n, i0 + m)
        if i1 > i0:
            out[i0:i1, 1] += s[: i1 - i0]
            out[i0:i1, 0] += s[: i1 - i0] * 0.6

    # cheap stereo reverb via FFT convolution with a decaying noise tail
    ir_n = int(1.3 * sr)
    ir = rng.standard_normal(ir_n).astype(np.float32) * np.exp(
        -np.arange(ir_n) / (0.32 * sr)).astype(np.float32)
    ir[: int(0.02 * sr)] = 0
    ir /= np.abs(ir).sum() / 3.0
    size = 1
    while size < n + ir_n:
        size *= 2
    for ch in (0, 1):
        a = np.fft.rfft(out[:, ch], size)
        b = np.fft.rfft(ir, size)
        wet = np.fft.irfft(a * b, size)[:n].astype(np.float32)
        out[:, ch] = out[:, ch] * 0.82 + wet * 0.55

    # gentle low-shelf so it never fights the voice
    out = np.tanh(out * 0.9) * 0.9
    return out.astype(np.float32)


# ------------------------------------------------------------------ captions
WORD_PER_CUE = 3
BASE_FS = 88          # caption size that always fits three Anton words
FIT_W = 930           # usable width inside the 1080 frame
ACCENT = "&H0032D9FF"  # ASS &HAABBGGRR -> warm yellow, same as the reference
WHITE = "&H00FFFFFF"

_FONT = {}


def _font(size):
    f = _FONT.get(size)
    if f is None:
        import os
        from PIL import ImageFont
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "assets", "fonts", "Anton-Regular.ttf")
        f = ImageFont.truetype(path, size)
        _FONT[size] = f
    return f


def _fit_size(text, base=BASE_FS, width=FIT_W):
    """Shrink the caption just enough that the widest line stays inside the frame."""
    try:
        fs = base
        for _ in range(24):
            if _font(fs).getlength(text) <= width or fs <= 46:
                return fs
            fs -= 2
        return fs
    except Exception:
        return base


def _chunks(text, size=WORD_PER_CUE):
    words = [w for w in text.replace("…", " ").split() if w]
    out = []
    for i in range(0, len(words), size):
        out.append(words[i:i + size])
    return out


def _reveal(words, j, fs):
    """Words 0..j, newest popping in accent-coloured; the rest stay as invisible
    placeholders so the line never re-centres while it types on."""
    parts = []
    for k, w in enumerate(words[: j + 1]):
        w = w.upper()
        if k == j:
            parts.append(f"{{\\c{ACCENT}\\fscx114\\fscy114"
                         f"\\t(0,80,\\fscx100\\fscy100)}}{w}")
        else:
            parts.append(f"{{\\c{WHITE}}}{w}")
    txt = " ".join(parts)
    rest = [w.upper() for w in words[j + 1:]]
    if rest:
        txt += " {\\alpha&HFF&}" + " ".join(rest)
    return f"{{\\fs{fs}\\fad(40,70)}}" + txt


def _measure(words):
    return " ".join(w.upper() for w in words)


def ass_time_to_s(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def ass_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Anton,88,&H0032D9FF,&H00FFFFFF,&H00141010,&H90000000,0,0,0,0,100,100,1,0,1,8,4,2,70,70,300,1
Style: Kicker,Anton,44,&H0040E0FF,&H00FFFFFF,&H00141010,&H90000000,0,0,0,0,100,100,3,0,1,5,3,8,60,60,140,1
Style: Title,Anton,132,&H00FFFFFF,&H00FFFFFF,&H00141010,&H90000000,0,0,0,0,100,100,2,0,1,10,5,5,60,60,0,1
Style: Sub,Anton,54,&H00D9E6FF,&H00FFFFFF,&H00141010,&H90000000,0,0,0,0,100,100,2,0,1,5,3,2,60,60,300,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def write_ass(tl: Timeline, path, clip=None, shift=0.0):
    """Burned captions: kicker per scene, word-by-word reveal, title + end card."""
    lines = [ASS_HEADER]
    t0, t1 = (clip if clip else (0.0, 1e9))

    lines.append(f"Dialogue: 1,{ass_time(0.30)},{ass_time(3.85)},Title,,0,0,0,,"
                 "{\\fad(340,420)\\fscx86\\fscy86\\blur1.2"
                 "\\t(0,520,\\fscx100\\fscy100\\blur0)}"
                 "RAHASIA DI BALIK\\NTEMBOK BESAR CHINA")

    for sc in tl.scenes:
        seg = next(s for s in tl.segments if s.scene == sc["id"] and s.vo_dur > 0)
        k0 = seg.vo_start - 0.35
        k1 = min(seg.vo_start + seg.vo_dur - 0.1, k0 + 4.6)
        lines.append(f"Dialogue: 0,{ass_time(k0)},{ass_time(k1)},Kicker,,0,0,0,,"
                     "{\\fad(240,260)\\fscx104\\fscy104"
                     "\\t(0,240,\\fscx100\\fscy100)}" + sc["kicker"].upper())

        cues = _chunks(sc["narration"])
        weights = np.array([len(" ".join(c)) + 3 for c in cues], dtype=float)
        weights /= weights.sum()
        a0 = seg.vo_start + 0.06
        span = max(1.0, seg.vo_dur - 0.20)
        cur = a0
        for cue, wgt in zip(cues, weights):
            d = span * wgt
            lw = np.array([len(w) + 1.6 for w in cue], dtype=float)
            lw /= lw.sum()
            fs = _fit_size(_measure(cue))
            t = cur
            for j in range(len(cue)):
                t_end = cur + d * float(min(1.0, lw[: j + 1].sum()))
                hold = 0.10 if j == len(cue) - 1 else 0.0
                lines.append(f"Dialogue: 2,{ass_time(t)},{ass_time(t_end + hold)},Cap,,0,0,0,,"
                             + _reveal(cue, j, fs))
                t = t_end
            cur += d

    t_end = tl.total
    lines.append(f"Dialogue: 2,{ass_time(t_end - 2.15)},{ass_time(t_end - 0.05)},Cap,,0,0,0,,"
                 "{\\fad(180,300)\\fs74}IKUTI UNTUK BAGIAN BERIKUTNYA")

    if clip is not None:
        kept = [lines[0]]
        for ln in lines[1:]:
            head = ln.split(",", 9)          # keep the text field (may contain commas) intact
            a, b = ass_time_to_s(head[1]), ass_time_to_s(head[2])
            if b <= t0 or a >= t1:
                continue
            a2, b2 = max(a, t0) - shift, min(b, t1) - shift
            kept.append(",".join([head[0], ass_time(a2), ass_time(b2)] + head[3:]))
        lines = kept
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return path
