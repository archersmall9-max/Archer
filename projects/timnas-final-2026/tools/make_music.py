#!/usr/bin/env python3
"""Mensintesis bed musik original (tanpa sampel berhak cipta) untuk Short 9:16.

Gaya: sport-hype / trailer. 4-on-the-floor kick, hi-hat, sub-bass pulse,
riser menjelang drop, dan whoosh tiap pergantian scene.

Dipakai oleh build.py. Output: WAV 44.1 kHz stereo.
"""
import argparse
import math
import random
import struct
import wave

SR = 44100


def _env(n, attack, decay, sr=SR):
    """Amplop attack-decay eksponensial."""
    a = max(1, int(attack * sr))
    out = []
    for i in range(n):
        if i < a:
            out.append(i / a)
        else:
            out.append(math.exp(-(i - a) / (decay * sr)))
    return out


def kick(dur=0.32):
    n = int(dur * SR)
    env = _env(n, 0.001, 0.055)
    buf = []
    phase = 0.0
    for i in range(n):
        t = i / SR
        f = 120.0 * math.exp(-t * 28.0) + 44.0          # pitch sweep 120 -> 44 Hz
        phase += 2 * math.pi * f / SR
        s = math.sin(phase)
        s += 0.35 * math.sin(phase * 2) * math.exp(-t * 60)   # klik transient
        buf.append(s * env[i])
    return buf


def hat(dur=0.055, bright=1.0):
    n = int(dur * SR)
    env = _env(n, 0.0005, 0.012)
    prev = 0.0
    buf = []
    for i in range(n):
        w = random.uniform(-1, 1)
        hp = w - prev            # high-pass 1-pole sederhana
        prev = w
        buf.append(hp * env[i] * bright)
    return buf


def sub(freq, dur):
    n = int(dur * SR)
    env = _env(n, 0.006, dur * 0.6)
    buf = []
    for i in range(n):
        t = i / SR
        s = math.sin(2 * math.pi * freq * t)
        s += 0.22 * math.sin(2 * math.pi * freq * 2 * t)
        buf.append(s * env[i])
    return buf


def stab(freq, dur=0.5):
    """Chord stab pendek (saw-ish) untuk aksen."""
    n = int(dur * SR)
    env = _env(n, 0.004, 0.16)
    buf = []
    for i in range(n):
        t = i / SR
        s = 0.0
        for k, g in ((1, 1.0), (2, 0.45), (3, 0.3), (4, 0.18), (6, 0.1)):
            s += g * math.sin(2 * math.pi * freq * k * t)
        buf.append(s / 2.0 * env[i])
    return buf


def whoosh(dur=0.7):
    """Noise sweep untuk transisi antar scene."""
    n = int(dur * SR)
    buf = []
    lp = 0.0
    for i in range(n):
        p = i / n
        w = random.uniform(-1, 1)
        # filter low-pass yang makin terbuka -> terasa "naik"
        a = 0.02 + 0.55 * (p ** 2)
        lp += a * (w - lp)
        amp = math.sin(math.pi * p) ** 1.6
        buf.append(lp * amp)
    return buf


def riser(dur=1.6):
    n = int(dur * SR)
    buf = []
    phase = 0.0
    for i in range(n):
        p = i / n
        f = 220 * (2 ** (2.6 * p))
        phase += 2 * math.pi * f / SR
        buf.append(math.sin(phase) * (p ** 2.2) * 0.6)
    return buf


def mix_into(track, src, at, gain=1.0):
    start = int(at * SR)
    if start < 0:
        src = src[-start:]
        start = 0
    for i, s in enumerate(src):
        j = start + i
        if j >= len(track):
            break
        track[j] += s * gain


def build(duration, cuts, drop_at):
    total = int(duration * SR) + SR
    track = [0.0] * total
    bpm = 128.0
    beat = 60.0 / bpm
    bar = beat * 4

    n_beats = int(duration / beat) + 4
    for b in range(n_beats):
        t = b * beat
        if t > duration:
            break
        intensity = 0.55 if t < drop_at else 1.0
        # kick tiap ketukan
        mix_into(track, kick(), t, 0.80 * intensity)
        # off-beat hat
        mix_into(track, hat(), t + beat * 0.5, 0.13 * intensity)
        if t >= drop_at:
            mix_into(track, hat(bright=0.7), t + beat * 0.25, 0.07)
            mix_into(track, hat(bright=0.7), t + beat * 0.75, 0.07)

    # progresi bass: Am - F - C - G (minor, heroik)
    roots = [55.00, 43.65, 65.41, 49.00]   # A1, F1, C2, G1
    n_bars = int(duration / bar) + 2
    for i in range(n_bars):
        t = i * bar
        if t > duration:
            break
        f = roots[i % len(roots)]
        intensity = 0.5 if t < drop_at else 1.0
        mix_into(track, sub(f, bar * 0.92), t, 0.42 * intensity)
        if t >= drop_at:
            mix_into(track, stab(f * 4), t, 0.10)
            mix_into(track, stab(f * 4), t + beat * 2.5, 0.07)

    # riser tepat sebelum drop
    mix_into(track, riser(1.8), max(0.0, drop_at - 1.8), 0.22)

    # whoosh di tiap pergantian scene
    for c in cuts:
        mix_into(track, whoosh(0.65), max(0.0, c - 0.42), 0.26)

    # fade in / out
    fi = int(0.9 * SR)
    for i in range(min(fi, len(track))):
        track[i] *= i / fi
    fo = int(1.2 * SR)
    end = int(duration * SR)
    for i in range(min(fo, end)):
        track[end - 1 - i] *= i / fo
    for i in range(end, len(track)):
        track[i] = 0.0

    # soft clip + normalisasi
    peak = max(1e-6, max(abs(s) for s in track))
    g = 0.85 / peak
    out = []
    for s in track[:end]:
        v = math.tanh(s * g * 1.15)
        out.append(v)
    return out


def write_wav(path, mono):
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        frames = bytearray()
        for s in mono:
            v = int(max(-1.0, min(1.0, s)) * 32000)
            frames += struct.pack("<hh", v, v)
        w.writeframes(bytes(frames))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--cuts", default="", help="daftar detik pergantian scene, dipisah koma")
    ap.add_argument("--drop", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    random.seed(a.seed)
    cuts = [float(x) for x in a.cuts.split(",") if x.strip()]
    write_wav(a.out, build(a.duration, cuts, a.drop))
    print(f"musik: {a.out} ({a.duration:.2f}s, {len(cuts)} whoosh)")


if __name__ == "__main__":
    main()
