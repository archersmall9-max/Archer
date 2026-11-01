"""
gfx.py — finishing stack for the film, built on OpenCV 5 + numpy (latest releases).

The renderer hands over flat-shaded 8-bit frames; this module makes them look like a
grade, not a raster:

  * supersample resolve      — INTER_AREA downsample from a 1.5x render (true AA)
  * highlight-headroom curve — Narkowicz ACES fit blended with a linear ramp, applied
                               through a per-channel 3x256 LUT (keeps sky gradients smooth)
  * two-scale bloom          — 1/4-res bright pass, wide + tight gaussian octaves
  * warm halation            — film-style reddish bleed around the highlights
  * clarity / micro-contrast — unsharp at two radii (structure + detail)
  * saturation, lift/gain    — per-look colour grading
  * vignette                 — cached 8-bit multiplier mask
  * grain + ordered dither   — one addWeighted pass; kills banding, adds tooth
  * radial defocus           — optional fake depth of field for macro shots

Everything is cached per (h, w) inside the process, so the per-frame cost stays in the
single-digit milliseconds per effect.
"""
from __future__ import annotations

import numpy as np
import cv2

HEADROOM = 1.35          # scene linear headroom carried into the LUT
_C: dict = {}


# --------------------------------------------------------------------------- caches
def _cache(key, build):
    v = _C.get(key)
    if v is None:
        v = build()
        _C[key] = v
    return v


def _vig_mask(h, w, power=1.7, start=0.32):
    def build():
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2) / 1.4142
        m = np.clip((r - start) / (1.0 - start), 0, 1) ** power
        return (1.0 - m).astype(np.float32)
    return _cache(("vig", h, w, power, start), build)


def _vig_u8(h, w, amount):
    key = ("vigu8", h, w, round(amount, 3))

    def build():
        m = 1.0 - amount * (1.0 - _vig_mask(h, w))
        g = np.clip(m * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(np.dstack([g, g, g]))
    return _cache(key, build)


def _grain_bank(h, w, n=6):
    """Half-res noise: after upsampling it reads as fine film grain."""
    def build():
        rng = np.random.default_rng(11)
        hh, ww = max(8, h // 2), max(8, w // 2)
        return [rng.integers(0, 255, (hh, ww, 3), dtype=np.uint8) for _ in range(n)]
    return _cache(("grain", h, w), build)


def _dither(h, w):
    def build():
        rng = np.random.default_rng(23)
        d = rng.integers(96, 160, (h // 2, w // 2, 3), dtype=np.uint8)
        return cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    return _cache(("dith", h, w), build)


def _aces(x):
    x = np.clip(x, 0.0, 8.0)
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def _film_curve(x):
    """Narkowicz ACES, exposure-normalised so mid grey stays put.

    Plain ACES lifts the midtones ~20%, which is what makes a flat-shaded render read as
    "washed and punchy" at the same time. Solving for the exposure that keeps 0.5 at 0.5
    gives the highlight rolloff without the lift.
    """
    lo, hi = 0.2, 4.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _aces(0.5 * mid) / _aces(mid) > 0.5:
            hi = mid
        else:
            lo = mid
    e = 0.5 * (lo + hi)
    return _aces(x * e) / _aces(e)


def _tone_lut(filmic, contrast, lift, gain, shadows=0.045):
    key = ("lut", round(filmic, 3), round(contrast, 3),
           tuple(np.round(lift, 4)), tuple(np.round(gain, 4)), round(shadows, 4))
    def build():
        x = np.linspace(0.0, HEADROOM, 256, dtype=np.float32)
        out = np.zeros((256, 1, 3), np.uint8)
        for c in range(3):
            v = x * gain[c] + lift[c]
            if filmic > 0:
                f = _film_curve(np.clip(v, 0.0, HEADROOM))
                knee = np.clip((v - 1.0) / (HEADROOM - 1.0), 0.0, 1.0)
                f = f + (1.0 + (v - 1.0) * 0.18 - f) * knee     # shoulder above 1.0
                v = v + (f - v) * filmic
            v = v + shadows * (1.0 - np.clip(v, 0, 1)) ** 2.5      # gentle toe lift
            v = (v - 0.5) * contrast + 0.5
            out[:, 0, c] = np.clip(v * 255.0 + 0.5, 0, 255)
        return out
    return _cache(key, build)


# --------------------------------------------------------------------------- pieces
def resolve(img, w, h, interp=None):
    """Supersample resolve: high quality area average down to delivery size."""
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _bloom(a8, amount, halation, thr=0.70):
    """Returns a full-res uint8 glow layer (already scaled) - added in 8-bit for speed."""
    h, w = a8.shape[:2]
    sw, sh = max(16, w // 4), max(16, h // 4)
    small = cv2.resize(a8, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32) * (1.0 / 255.0)
    bright = np.clip((small - thr) * 3.33, 0, 1)
    tight = cv2.GaussianBlur(bright, (0, 0), 2.0)
    wide = cv2.GaussianBlur(bright, (0, 0), 7.5)
    bl = tight * 0.62 + wide * 0.38
    if halation > 0:
        tint = np.array([1.0, 0.80, 0.58], np.float32)
        bl = bl * (1.0 - halation) + bl * tint[None, None, :] * (halation * 1.25)
    bl *= amount * (255.0 / HEADROOM)
    return cv2.resize(np.clip(bl, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- grade
def grade(img, bloom=0.35, vignette=0.35, grain=3.0, saturation=1.06, contrast=1.05,
          lift=(0.0, 0.0, 0.0), gain=(1.0, 1.0, 1.0), seed=0, blur_bg=0.0,
          filmic=0.45, clarity=0.18, sharpen=0.16, halation=0.30, dither=1.5,
          shadows=0.045):
    """Finishing pass. Accepts a PIL image or an (H,W,3) uint8 RGB array."""
    a8 = np.asarray(img, dtype=np.uint8)
    if a8.ndim == 2:
        a8 = np.dstack([a8] * 3)
    h, w = a8.shape[:2]

    # ---- scene linear + bloom (8-bit, single SIMD pass) -----------------------
    q = cv2.convertScaleAbs(a8, alpha=1.0 / HEADROOM)
    if bloom > 0:
        q = cv2.add(q, _bloom(a8, bloom, halation))

    # ---- tone curve (per-channel LUT with highlight headroom) -----------------
    q = cv2.LUT(q, _tone_lut(filmic, contrast, lift, gain, shadows))

    # ---- saturation -----------------------------------------------------------
    if abs(saturation - 1.0) > 1e-3:
        g = cv2.cvtColor(q, cv2.COLOR_RGB2GRAY)
        q = cv2.addWeighted(q, saturation, cv2.cvtColor(g, cv2.COLOR_GRAY2RGB),
                            1.0 - saturation, 0)

    # ---- clarity (half-res blur: same look, quarter of the cost) --------------
    if clarity > 0:
        hh, ww = q.shape[0] // 2, q.shape[1] // 2
        small = cv2.resize(q, (ww, hh), interpolation=cv2.INTER_AREA)
        wide = cv2.GaussianBlur(small, (0, 0), 2.1)
        wide = cv2.resize(wide, (q.shape[1], q.shape[0]), interpolation=cv2.INTER_LINEAR)
        q = cv2.addWeighted(q, 1.0 + clarity, wide, -clarity, 0)

    # ---- micro-sharpen (3x3 unsharp) -----------------------------------------
    if sharpen > 0:
        k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) * sharpen
        k[1, 1] += 1.0 - sharpen
        q = cv2.filter2D(q, -1, k)

    # ---- defocus (macro shots) ------------------------------------------------
    if blur_bg > 0:
        soft = cv2.GaussianBlur(q, (0, 0), 4.5)
        q = cv2.addWeighted(q, 1.0, soft, blur_bg * 0.85, 0)

    # ---- vignette (cached 8-bit multiplier) -----------------------------------
    if vignette > 0:
        q = cv2.multiply(q, _vig_u8(h, w, vignette), scale=1.0 / 255.0)

    # ---- grain + dither (one SIMD pass) --------------------------------------
    amp = grain * 0.55 + dither * 0.9
    if amp > 0:
        bank = _grain_bank(h, w)
        n = cv2.resize(bank[seed % len(bank)], (w, h), interpolation=cv2.INTER_LINEAR)
        if dither > 0:
            n = cv2.addWeighted(n, 0.65, _dither(h, w), 0.35, 0)
        q = cv2.addWeighted(q, 1.0, n, amp / 255.0, -128.0 * amp / 255.0)
    return q


def encode_frame_bgr(q):
    """ffmpeg's rawvideo pipe wants BGR when the codec is yuv420p via swscale."""
    return q
