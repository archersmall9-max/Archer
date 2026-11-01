"""
shots.py — the film: 18 animated 3D shots with auto-framed cinematic cameras.

Each shot builds procedural geometry once, then animates camera + actors + palette
as a function of local time. Camera framing is derived from the subject radius so
every shot is guaranteed to be composed, not accidental.
"""
from __future__ import annotations

import numpy as np

import world as W
from engine3d import (Camera, Lighting, Mesh, Sun, box, ease_io, icosphere, lerp,
                      normalize, pad_faces, prism, pyramid, rot_x, rot_y, smooth)

W_IMG, H_IMG = 1080, 1920


# ------------------------------------------------------------------ palettes
class Look:
    def __init__(self, sun_dir, sun_col, sun_s, amb, sky_top, sky_bot, fog, fog_d,
                 exposure=1.0, sky_glow=0.5, sky_size=0.06, stars=0.0,
                 post=None, sky=(0.55, 0.68, 0.85), ground=(0.28, 0.26, 0.22),
                 fog_h=0.0045):
        self.sun = Sun(normalize(np.array(sun_dir, float)), sun_col, sun_s)
        self.light = Lighting(self.sun, ambient=amb, sky=sky, ground=ground,
                              fog=fog, fog_density=fog_d, fog_height=fog_h,
                              exposure=exposure)
        self.sky_top = sky_top
        self.sky_bot = sky_bot
        self.sky_glow = sky_glow
        self.sky_size = sky_size
        self.stars = stars
        self.post = post or dict(bloom=0.30, vignette=0.34, grain=2.0,
                                 saturation=1.12, contrast=1.06)


def look_golden():
    return Look((-0.66, 0.17, 0.60), (1.0, 0.80, 0.55), 1.45, (0.24, 0.25, 0.34),
                sky_top=(0.12, 0.26, 0.58), sky_bot=(1.0, 0.76, 0.56),
                fog=(0.92, 0.78, 0.68), fog_d=0.0011, exposure=1.02, sky_glow=1.0,
                sky_size=0.09, post=dict(bloom=0.40, vignette=0.36, grain=2.0,
                                         saturation=1.18, contrast=1.06, filmic=0.50,
                                         gain=(1.05, 1.0, 0.93)))


def look_day():
    return Look((-0.45, 0.62, 0.44), (1.0, 0.97, 0.92), 1.05, (0.34, 0.38, 0.46),
                sky_top=(0.18, 0.42, 0.80), sky_bot=(0.80, 0.89, 0.98),
                fog=(0.80, 0.87, 0.96), fog_d=0.0012, exposure=1.0, sky_glow=0.45,
                sky_size=0.05, post=dict(bloom=0.26, vignette=0.28, grain=1.8,
                                         saturation=1.10, contrast=1.05, filmic=0.42))


def look_dusk():
    return Look((-0.75, 0.10, 0.32), (1.0, 0.52, 0.32), 0.85, (0.22, 0.23, 0.34),
                sky_top=(0.07, 0.12, 0.28), sky_bot=(0.62, 0.40, 0.38),
                fog=(0.30, 0.32, 0.44), fog_d=0.0022, exposure=1.05, sky_glow=0.5,
                sky_size=0.05, post=dict(bloom=0.40, vignette=0.45, grain=2.6,
                                         saturation=1.02, contrast=1.08, filmic=0.35,
                                         gain=(1.02, 1.0, 1.03)))


def look_night():
    return Look((-0.35, 0.58, -0.42), (0.58, 0.70, 1.0), 0.60, (0.11, 0.14, 0.27),
                sky_top=(0.015, 0.03, 0.10), sky_bot=(0.08, 0.12, 0.24),
                fog=(0.09, 0.12, 0.22), fog_d=0.0022, exposure=1.10, sky_glow=0.35,
                sky_size=0.05, stars=1.0,
                post=dict(bloom=0.55, vignette=0.50, grain=3.0, saturation=1.05,
                          contrast=1.08, filmic=0.18, shadows=0.09,
                          gain=(0.97, 0.99, 1.06)))


def look_blue():
    return Look((-0.45, 0.30, -0.5), (0.66, 0.76, 1.0), 1.05, (0.24, 0.28, 0.44),
                sky_top=(0.02, 0.06, 0.16), sky_bot=(0.24, 0.36, 0.52),
                fog=(0.26, 0.34, 0.48), fog_d=0.0016, exposure=1.15, sky_glow=0.5,
                sky_size=0.045, stars=0.8,
                post=dict(bloom=0.45, vignette=0.46, grain=2.6, saturation=1.06,
                          contrast=1.08, filmic=0.22, shadows=0.08,
                          gain=(0.96, 1.0, 1.08)))


def look_macro():
    return Look((-0.55, 0.45, 0.55), (1.0, 0.90, 0.72), 1.25, (0.34, 0.32, 0.34),
                sky_top=(0.30, 0.42, 0.62), sky_bot=(0.92, 0.84, 0.74),
                fog=(0.70, 0.68, 0.66), fog_d=0.0030, exposure=1.05, sky_glow=0.5,
                post=dict(bloom=0.34, vignette=0.40, grain=2.2, saturation=1.10,
                          contrast=1.08, filmic=0.45, gain=(1.02, 1.0, 0.97)))


# ------------------------------------------------------------------ camera
def shake(t, amp=0.10, rate=1.0, seed=0):
    return np.array([np.sin(t * 1.7 * rate + seed),
                     np.sin(t * 2.3 * rate + 1 + seed) * 0.7,
                     np.sin(t * 1.3 * rate + 2 + seed)]) * amp


def aim(center, radius, az, el, dist, fov=50, t=0.0, shake_amp=0.09, look=None,
        rate=1.0, roll=0.0):
    """Place the camera on a sphere around `center`, looking at it (or at `look`)."""
    d = np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
    eye = np.asarray(center, float) + d * radius * dist
    tgt = np.asarray(look if look is not None else center, float)
    return Camera(eye + shake(t, shake_amp, rate), tgt, fov=fov, w=W_IMG, h=H_IMG,
                  near=0.4, roll=roll)


def _blocked(world, eye, tgt, margin=1.6, n=26):
    """True if terrain rises above the sight line between eye and target."""
    if world is None:
        return False
    for s_ in np.linspace(0.06, 0.94, n):
        p = np.asarray(eye) + (np.asarray(tgt) - np.asarray(eye)) * s_
        if world.height_at(p[0], p[2]) > p[1] - margin:
            return True
    return False


def smart_aim(world, center, radius, az, el, dist, fov=50, t=0.0, shake_amp=0.09,
              look=None, rate=1.0, roll=0.0, margin=1.6):
    """aim(), but searches nearby camera placements until the subject is unobstructed."""
    tgt = np.asarray(look if look is not None else center, float)
    d_az = [0.0, 0.18, -0.18, 0.38, -0.38, 0.62, -0.62, 0.9, -0.9, 1.25, -1.25]
    d_el = [0.0, 0.10, 0.22, -0.10, 0.36, 0.52, 0.14]
    d_d = [1.0, 1.18, 1.38, 1.6, 0.9]
    best, best_pen = None, 1e9
    for dd in d_d:
        for de in d_el:
            for da in d_az:
                c = aim(center, radius, az + da, el + de, dist * dd, fov=fov, t=t,
                        shake_amp=shake_amp, look=look, rate=rate, roll=roll)
                pen = 0.0
                for s_ in np.linspace(0.06, 0.94, 14):
                    p = c.eye + (tgt - c.eye) * s_
                    h = world.height_at(p[0], p[2]) if world is not None else -1e9
                    pen += max(0.0, h - (p[1] - margin))
                if pen < best_pen - 1e-6:
                    best, best_pen = c, pen
                if pen <= 0:
                    return c
    return best



def wall_shot(world, pts, i, lat, up, ahead, side=1, fov=50, t=0.0, shake_amp=0.08,
              look_up=6.0, back=0.0):
    """Camera offset to the side of the wall, above the walkway, looking along it."""
    i = int(max(2, min(len(pts) - 2, i)))
    a = np.asarray(pts[i], float)
    fwd, right = path_frame(pts, i)
    eye = a + right * (side * lat) + np.array([0, world.wall_h + up, 0]) - fwd * back
    eye[1] = max(eye[1], world.height_at(eye[0], eye[2]) + 6.0)
    j = int(max(1, min(len(pts) - 1, i + ahead / 4.4)))
    tgt = np.asarray(pts[j], float) + np.array([0, look_up, 0])
    return Camera(eye + shake(t, shake_amp), tgt, fov=fov, w=W_IMG, h=H_IMG, near=0.4)


def look_shot(world, center, dist, az, el, fov=50, t=0.0, shake_amp=0.08, look=None,
              lift=5.0, roll=0.0):
    d = np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
    eye = np.asarray(center, float) + d * dist
    if world is not None:
        eye[1] = max(eye[1], world.height_at(eye[0], eye[2]) + lift)
    tgt = np.asarray(look if look is not None else center, float)
    return Camera(eye + shake(t, shake_amp), tgt, fov=fov, w=W_IMG, h=H_IMG, near=0.4,
                  roll=roll)



def fit_view(world, meshes, direction, fov=50, margin=1.15, t=0.0, shake_amp=0.08,
             target=None, lift=4.0, extra=1.0):
    """Frame a subject exactly: camera distance derived from its bounding sphere."""
    V = np.concatenate([m.v for m in meshes])
    lo, hi = V.min(0), V.max(0)
    ctr = (lo + hi) / 2.0 if target is None else np.asarray(target, float)
    rad = float(np.linalg.norm(hi - lo)) / 2.0 * extra
    d = normalize(np.asarray(direction, float))
    dist = rad * margin / np.tan(fov * np.pi / 360.0)
    eye = ctr + d * dist
    if world is not None:
        eye[1] = max(eye[1], world.height_at(eye[0], eye[2]) + lift)
    return Camera(eye + shake(t, shake_amp), ctr, fov=fov, w=W_IMG, h=H_IMG, near=0.4)


def lerp_key(keys, u):
    """keys: [(u0, value...), ...] piecewise smooth interpolation of scalars."""
    us = [k[0] for k in keys]
    for i in range(len(keys) - 1):
        if us[i] <= u <= us[i + 1]:
            f = smooth((u - us[i]) / max(us[i + 1] - us[i], 1e-9))
            return [lerp(keys[i][j], keys[i + 1][j], f) for j in range(1, len(keys[i]))]
    return list(keys[-1][1:])


# ------------------------------------------------------------------ props
def tower_flags(world, t, color=W.BLUE, spacing=4.4, every=0.10, h_extra=13.6):
    pts = world.path_points(spacing)
    n = len(pts)
    idxs = np.linspace(4, n - 5, max(2, int(round(1.0 / every)))).astype(int)
    out = []
    for i in idxs:
        p = pts[i] + np.array([0, world.wall_h + h_extra, 0])
        tgh = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
        tgh[1] = 0
        yaw = float(np.arctan2(tgh[0], tgh[2]))
        out.append(box(p + np.array([0, 2.0, 0]), (0.13, 4.2, 0.13), W.WOOD))
        out.extend(W.flag(p + np.array([0, 3.4, 0]), yaw, t + i * 0.3, color, scale=1.1))
    return out


def mist_bank(center, t, count=10, spread=90.0, y=0.0, color=(0.92, 0.94, 0.98),
              alpha=0.16, seed=4):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        p = np.array(center, float) + np.array([rng.uniform(-spread, spread),
                                                y + rng.uniform(-3, 5),
                                                rng.uniform(-spread, spread)])
        p[0] += np.sin(t * 0.15 + i) * 6
        out.append(icosphere(p, rng.uniform(16, 34), color, squash=0.16, alpha=alpha))
    return out


def grave_mounds(pts, count=16, seed=8):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        j = int(rng.integers(3, len(pts) - 3))
        off = rng.normal(0, 8)
        t = pts[min(j + 1, len(pts) - 1)] - pts[j]
        t[1] = 0
        t = normalize(t)
        s = np.cross(t, np.array([0.0, 1.0, 0.0]))
        p = pts[j] + s * off
        out.append(icosphere(p + np.array([0, 0.3, 0]), rng.uniform(1.2, 2.2), W.EARTH * 0.9,
                             squash=0.42))
        if rng.random() < 0.75:
            out.append(box(p + np.array([0, 1.0, 0]), (0.06, 1.7, 0.06),
                           np.array([0.44, 0.29, 0.17])))
            out.append(icosphere(p + np.array([0, 1.9, 0]), 0.14,
                                 np.array([1.0, 0.55, 0.25]), glow=1.0))
    return out


def path_frame(pts, i, span=8):
    """Forward/right unit vectors at path sample i."""
    a = pts[max(0, i - span)]
    b = pts[min(len(pts) - 1, i + span)]
    fwd = (b - a) * np.array([1, 0, 1])
    fwd = normalize(fwd)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    return fwd, right


class Shot:
    name = "shot"
    weight = 1.0

    def __init__(self, dur):
        self.dur = dur
        self.build()

    def build(self):
        pass

    def u(self, t):
        return min(1.0, max(0.0, t / max(self.dur, 1e-3)))

    def frame(self, t):
        raise NotImplementedError


# ---------------------------------------------------------------- 1. hero
class HeroShot(Shot):
    name = "hero"

    def build(self):
        self.world = W.World(seed=3, cell=8.0, extent=760.0, wall_height=11.0,
                             trees=120, boulders=55)
        self.pts = self.world.path_points(4.4)

    def frame(self, t):
        u = ease_io(self.u(t))
        p = self.pts
        i = int(len(p) * (0.30 + 0.14 * u))
        c0 = p[i] + np.array([0, 6, 0])
        fwd, right = path_frame(p, i)
        az = float(np.arctan2(right[2], right[0])) + 0.28
        c = wall_shot(self.world, p, i, lat=40 - 9 * u, up=17 - 4 * u,
                      ahead=115 - 20 * u, fov=54, t=t, shake_amp=0.07, look_up=22)
        meshes = list(self.world.static) + tower_flags(self.world, t)
        meshes += W.cloud_bank(center=(c0[0], 240, c0[2]), t=t, count=8, scale=2.0,
                               spread=520.0)
        meshes += W.birds((c0[0] + 40, c0[1] + 55, c0[2] + 30), t, count=7, radius=38)
        meshes += mist_bank((c0[0], c0[1] - 10, c0[2]), t, count=7, spread=120, alpha=0.10)
        return meshes, c, look_golden()


# ---------------------------------------------------------------- 2. many walls
class ManyWallsShot(Shot):
    name = "many"

    def build(self):
        self.world = W.World(seed=7, cell=9.0, extent=780.0, wall_on=False,
                             trees=70, boulders=40, ridge_amp=12.0)
        base = self.world.height_at(0, 0)
        self.ribbons = []
        specs = [(11, W.EARTH * 1.05, 6.0, 1.5, 30), (23, W.EARTH * 0.85, 5.0, 1.1, -20),
                 (37, W.STONE, 9.0, 0.8, 0)]
        for seed, col, h, wig, dz in specs:
            path = W.make_path(120, length=560, wiggle=wig, seed=seed)
            path = path + np.array([0, 0, dz])
            pp = np.array([[pt[0], self.world.height_at(pt[0], pt[2]) + 14.0, pt[2]]
                           for pt in path])
            prof = [(-4.6, -8), (-3.2, h), (3.2, h), (4.6, -8)]
            self.ribbons.append(W.ribbon(pp, prof, col, closed=True))
            # beacon tower every third of the way
            for k in range(3, 118, 26):
                q = pp[k]
                self.ribbons.append(box(q + np.array([0, h + 4, 0]), (7, 12, 7), col * 1.05))
                self.ribbons.append(pyramid(q + np.array([0, h + 10, 0]), (9, 9), 4.0, W.TILE))

    def frame(self, t):
        u = ease_io(self.u(t))
        az = -0.4 + 1.0 * u
        dist = 1.45 - 0.25 * u
        c = aim((0, 30, 0), 300, az, 0.62 - 0.14 * u, dist, fov=44, t=t, shake_amp=0.05)
        meshes = list(self.world.static) + self.ribbons
        meshes += W.cloud_bank((0, 300, 0), t, count=10, scale=1.8, spread=340.0, seed=12)
        return meshes, c, look_day()


# ---------------------------------------------------------------- 3. from above
class OrbitShot(Shot):
    name = "orbit"

    def build(self):
        self.world = W.World(seed=3, cell=11.0, extent=1500.0, wall_height=11.0,
                             trees=90, boulders=40, ridge_amp=30.0, tower_every=0.09)
        self.pts = self.world.path_points(6.0)

    def frame(self, t):
        u = ease_io(self.u(t))
        p = self.pts
        c0 = p[len(p) // 2]
        az = 0.6 + 0.5 * u
        c = wall_shot(self.world, p, int(len(p) * (0.34 + 0.1 * u)), lat=170, up=190,
                      ahead=520, fov=40, t=t, shake_amp=0.04, look_up=-10)
        lk = look_day()
        lk.light.fog_density = 0.0013
        lk.light.fog = np.array([0.70, 0.79, 0.93])
        lk.sky_top = (0.10, 0.26, 0.60)
        lk.sky_bot = (0.72, 0.83, 0.95)
        lk.post = dict(bloom=0.30, vignette=0.30, grain=1.8, saturation=1.0, contrast=1.02)
        meshes = list(self.world.static)
        meshes += W.cloud_bank((c0[0], 210, c0[2]), t, count=14, scale=2.2, spread=700.0,
                               seed=21)
        return meshes, c, lk


# ---------------------------------------------------------------- 4. camouflage
class CamoShot(Shot):
    name = "camo"

    def build(self):
        self.world = W.World(seed=5, cell=7.0, extent=620.0, wall_height=10.0,
                             trees=45, boulders=110, ridge_amp=34.0)

    def frame(self, t):
        u = ease_io(self.u(t))
        p = self.world.path_points(4.4)
        i = int(len(p) * (0.40 + 0.20 * u))
        c0 = p[i] + np.array([0, 4, 0])
        fwd, right = path_frame(p, i)
        az = float(np.arctan2(fwd[2], fwd[0])) + 0.9
        c = wall_shot(self.world, p, i, lat=48, up=40, ahead=130, fov=38, t=t,
                      shake_amp=0.06, look_up=2)
        meshes = list(self.world.static)
        meshes += W.cloud_bank((c0[0], 330, c0[2]), t, count=6, scale=1.6, spread=420.0,
                               seed=33)
        lk = look_day()
        lk.light.fog_density = 0.0016
        return meshes, c, lk


# ---------------------------------------------------------------- 5. workers
class WorkerShot(Shot):
    name = "workers"

    def build(self):
        self.world = W.World(seed=3, cell=8.0, extent=700.0, wall_height=11.0,
                             trees=60, boulders=45)
        self.pts = self.world.path_points(4.4)
        self.i0 = int(len(self.pts) * 0.52)

    def frame(self, t):
        u = ease_io(self.u(t))
        p = self.pts
        i0 = self.i0
        a = p[i0]
        fwd, right = path_frame(p, i0)
        c0 = a + right * 4 + np.array([0, self.world.wall_h * 0.9, 0])
        az = float(np.arctan2(right[2], right[0])) - 0.35
        c = wall_shot(self.world, p, i0 - 1 + int(8 * u), lat=6.5, up=4.5 + 1.0 * u,
                      ahead=15, fov=50, t=t, shake_amp=0.12, look_up=11.4)
        meshes = list(self.world.static)
        walk_top = self.world.wall_h
        # labourers hauling blocks up the crest
        for k in range(8):
            ph = t * 3.2 + k * 1.9
            j = i0 - 10 + k * 4
            q = p[max(1, min(len(p) - 2, j))]
            pos = q + right * (1.4 * ((k % 2) * 2 - 1)) + np.array([0, walk_top + 0.1, 0])
            yaw = float(np.arctan2(fwd[0], fwd[2])) * (1 if k % 2 else -1)
            meshes += W.soldier(pos, yaw=yaw, phase=ph, scale=1.0,
                                armor=np.array([0.34, 0.29, 0.24]),
                                accent=np.array([0.40, 0.33, 0.25]), spear=False)
            meshes.append(box(pos + np.array([0, 1.95, 0]), (1.0, 0.5, 1.2),
                              np.array([0.60, 0.58, 0.54]), rot=rot_y(0.35)))
        # scaffolding against the wall face + stacked blocks
        for k in range(6):
            j = i0 + 6 + k * 3
            q = p[max(1, min(len(p) - 2, j))]
            for s in (-1, 1):
                meshes.append(box(q + right * (s * 6.0) + np.array([0, 5.0, 0]),
                                  (0.32, 10.0, 0.32), W.WOOD_L))
                meshes.append(box(q + right * (s * 6.0) + np.array([0, 8.8, 0]),
                                  (2.2, 0.28, 0.28), W.WOOD_L, rot=rot_y(0.25)))
        for k in range(5):
            q = p[i0 - 6] + right * (-9 - k * 1.3)
            meshes.append(box(q + np.array([0, 0.6 + 0.5 * (k % 2), 0]), (2.0, 1.0, 1.6),
                              np.array([0.58, 0.56, 0.52]), rot=rot_y(0.4 * k)))
        meshes += W.dust(a + right * 6 + np.array([0, 2, 0]), t, count=16, spread=13,
                         color=(0.80, 0.72, 0.58))
        meshes += mist_bank((a[0], a[1] - 2, a[2]), t, count=5, spread=70, alpha=0.08)
        return meshes, c, look_golden()


# ---------------------------------------------------------------- 6. chisel
class ChiselShot(Shot):
    name = "chisel"

    def build(self):
        self.world = W.World(seed=3, cell=8.0, extent=700.0, wall_height=11.0,
                             trees=40, boulders=30)
        self.pts = self.world.path_points(4.4)
        i0 = int(len(self.pts) * 0.52)
        self.i0 = i0
        fwd, right = path_frame(self.pts, i0)
        self.fwd, self.right = fwd, right
        g = self.pts[i0] + right * 27
        g[1] = self.world.height_at(g[0], g[2])
        self.g = g
        self.block_c = g + np.array([0, 0.7, 0])
        self.meshes = []
        self.meshes.append(box(self.block_c, (2.8, 1.4, 2.3),
                               np.array([0.60, 0.58, 0.54]), rot=rot_y(0.35)))
        # chisels, mallets and offcuts
        self.meshes.append(box(g + right * 1.6 + np.array([0, 0.25, 0.8]), (0.9, 0.5, 0.7),
                               np.array([0.56, 0.54, 0.50]), rot=rot_y(1.1)))
        for k in range(6):
            self.meshes.append(box(g + np.array([np.sin(k * 2.1) * 2.6, 0.12,
                                                 np.cos(k * 1.7) * 2.4]),
                                   (0.55, 0.22, 0.45),
                                   np.array([0.62, 0.60, 0.56]), rot=rot_y(k)))
        self.center = self.block_c + np.array([0, 0.6, 0])

    def frame(self, t):
        u = ease_io(self.u(t))
        g = self.g
        # two stonecutters crouched at the block
        meshes = list(self.world.static) + list(self.meshes)
        for k in range(2):
            pos = g + self.right * (-1.1 + 2.2 * k) + self.fwd * 1.5
            pos[1] = self.world.height_at(pos[0], pos[2])
            bob = 0.10 * np.sin(t * 7.0 + k * 2.0)
            meshes += W.soldier(pos + np.array([0, 0.05 + bob, 0]),
                                yaw=float(np.arctan2(-self.fwd[0], -self.fwd[2])) + 0.4 * (k - 0.5),
                                phase=0.0, scale=0.86,
                                armor=np.array([0.36, 0.30, 0.24]),
                                accent=np.array([0.42, 0.34, 0.26]), spear=False)
        tap = abs(np.sin(t * 5.0))
        meshes.append(box(self.block_c + self.right * 0.9 +
                          np.array([0, 0.75 + 0.30 * (1 - tap), 0]),
                          (0.10, 1.0, 0.10), np.array([0.72, 0.72, 0.76]), rot=rot_y(0.4)))
        meshes.append(box(self.block_c + self.right * 1.15 +
                          np.array([0, 1.35 + 0.30 * (1 - tap), 0.15]),
                          (0.40, 0.28, 0.78), W.WOOD))
        meshes += W.dust(self.block_c + self.right * 1.0, t, count=9, spread=1.1,
                         color=(0.90, 0.88, 0.84), seed=13)
        meshes += mist_bank((g[0], g[1] - 1, g[2]), t, count=4, spread=40, alpha=0.08)
        eye = self.center + self.right * (8.0 - 1.0 * u) - self.fwd * 3.4 + np.array([0, 1.5, 0])
        eye[1] = max(eye[1], self.world.height_at(eye[0], eye[2]) + 1.4)
        c = Camera(eye + shake(t, 0.05), self.center + np.array([0, 0.7, 0]),
                   fov=46, w=W_IMG, h=H_IMG, near=0.35)
        return meshes, c, look_macro()


# ---------------------------------------------------------------- 7. memorial
class MemorialShot(Shot):
    name = "memorial"

    def build(self):
        self.world = W.World(seed=3, cell=8.0, extent=720.0, wall_height=10.0,
                             trees=70, boulders=40)
        self.pts = self.world.path_points(4.4)
        self.graves = grave_mounds(self.pts, count=20, seed=8)
        self.i0 = int(len(self.pts) * 0.45)

    def frame(self, t):
        u = ease_io(self.u(t))
        p = self.pts
        i0 = self.i0
        a = p[i0]
        fwd, right = path_frame(p, i0)
        c0 = a + np.array([0, 4, 0])
        az = float(np.arctan2(right[2], right[0])) + 0.4
        c = wall_shot(self.world, p, i0, lat=52 - 10 * u, up=34 + 16 * u,
                      ahead=120 - 24 * u, fov=46, t=t, shake_amp=0.07, look_up=5)
        meshes = list(self.world.static) + self.graves
        meshes += mist_bank((a[0], a[1] - 1, a[2]), t, count=16, spread=95, alpha=0.19,
                            color=(0.80, 0.84, 0.92))
        meshes += W.birds((a[0], a[1] + 45, a[2]), t, count=4, radius=45, seed=6)
        return meshes, c, look_dusk()


# ---------------------------------------------------------------- 8. sticky rice
class RiceShot(Shot):
    name = "rice"

    def build(self):
        rng = np.random.default_rng(4)
        self.meshes = []
        bw, bh, gap = 1.5, 0.62, 0.10
        for row in range(14):
            off = (bw / 2) if row % 2 else 0.0
            for col in range(-6, 7):
                x = col * (bw + gap) + off
                y = row * (bh + gap)
                shade = 0.90 + 0.18 * rng.random()
                self.meshes.append(box((x, y + bh / 2, 0), (bw, bh, 1.2),
                                       np.array([0.56, 0.54, 0.51]) * shade))
        self.grains = []
        for row in range(14):
            off = (bw / 2) if row % 2 else 0.0
            for col in range(-6, 7):
                if rng.random() < 0.35:
                    continue
                x = col * (bw + gap) + off + rng.uniform(-0.55, 0.55)
                y = row * (bh + gap) + rng.uniform(0.04, 0.30)
                self.grains.append(icosphere((x, y, 0.62), 0.085,
                                             np.array([0.98, 0.96, 0.90]), squash=0.6))
        self.grains.append(icosphere((0, 100, 0), 0.001, (1, 1, 1)))
        self.center = np.array([0, 4.6, 0])

    def frame(self, t):
        u = ease_io(self.u(t))
        az = -1.15 + 0.85 * u
        c = aim(self.center, 5.4, az, 0.12, 1.55, fov=40, t=t, shake_amp=0.03,
                look=self.center + np.array([0.6 * (0.5 - u), 0, 0]))
        meshes = list(self.meshes) + self.grains
        meshes.append(box((0, -0.6, 0), (40, 1.0, 6), np.array([0.26, 0.24, 0.22])))
        return meshes, c, look_macro()


# ---------------------------------------------------------------- 9. cutaway
class CutawayShot(Shot):
    name = "cutaway"

    def build(self):
        L, D = 20.0, 9.0
        m = []
        y0 = -8.0
        # stepped cross-section: facing bricks on the -Z side only, so the +Z face
        # exposes the internal layers (a true cutaway)
        m.append(box((0, y0 + 2.2, -D * 0.5), (L, 4.4, 2.0), (0.34, 0.34, 0.33)))   # footing
        for k in range(7):                                                          # bricks
            m.append(box((0, y0 + 5.4 + k * 1.35, -D * 0.5), (L, 1.2, 2.4),
                         np.array([0.55, 0.53, 0.50]) * (0.94 + 0.10 * (k % 2))))
        m.append(box((0, y0 + 7.0, -D * 0.5 + 0.4), (L * 0.99, 9.0, 2.2), W.EARTH * 0.95))
        # rammed-earth core (earth bands)
        for k in range(6):
            m.append(box((0, y0 + 3.4 + k * 1.5, 0.2), (L * 0.96, 1.35, D * 0.55),
                         W.EARTH * (0.86 + 0.12 * (k % 3))))
        # rice-mortar seam
        m.append(box((0, y0 + 10.4, -1.0), (L * 0.94, 0.22, 1.2),
                     np.array([0.94, 0.90, 0.74]), glow=0.12))
        # walkway + parapets
        m.append(box((0, y0 + 11.4, 0.4), (L, 0.7, D * 0.8), (0.62, 0.61, 0.58)))
        for s in (-1, 1):
            for k in range(-7, 8):
                m.append(box((k * 1.3, y0 + 12.3, 0.4 + s * (D * 0.36)), (0.82, 1.2, 0.5),
                             (0.58, 0.57, 0.54)))
        # drainage channel on top of the core
        m.append(box((0, y0 + 10.9, 1.4), (L * 0.9, 0.4, 0.5), (0.46, 0.46, 0.45)))
        # grass cap so it sits in the world
        m.append(box((0, y0 - 0.1, 0), (L * 1.5, 0.8, D * 2.4), W.GRASS))
        self.meshes = m
        self.center = np.array([0, y0 + 7.5, 0.5])

    def frame(self, t):
        u = ease_io(self.u(t))
        az = 1.05 + 0.75 * u
        c = aim(self.center, 13.0, az, 0.18, 2.15, fov=42, t=t, shake_amp=0.04)
        meshes = list(self.meshes)
        meshes += W.cloud_bank((0, 150, 0), t, count=4, scale=0.8, spread=90, seed=9)
        return meshes, c, look_macro()


# ---------------------------------------------------------------- 10. gate
class GateBuild:
    """A gatehouse whose passage crosses the wall (local: X along the wall, Z through it)."""

    def __init__(self, world, i_frac=0.5):
        self.world = world
        self.pts = world.path_points(4.4)
        i = int(len(self.pts) * i_frac)
        self.i = i
        g = self.pts[i].copy()
        fwd, right = path_frame(self.pts, i)
        self.fwd, self.right = fwd, right
        self.yaw = float(np.arctan2(fwd[0], fwd[2]))
        self.pos = g
        R = np.stack([fwd, np.array([0.0, 1.0, 0.0]), right], axis=1)

        def L(x, y, z):
            return g + R @ np.array([x, y, z], float)

        H = 15.0
        gate = []
        for sgn in (-1, 1):
            gate.append(box(L(sgn * 10.5, H * 0.48, 0), (11.0, H + 6, 14.0), W.STONE, rot=R))
            gate.append(pyramid(L(sgn * 10.5, H + 3.0, 0), (14, 14), 5.6, W.TILE, rot=R))
            for k in np.linspace(-4.5, 4.5, 4):
                gate.append(box(L(sgn * 10.5 + 0, H + 6.2, k * 1.1), (11.2, 1.5, 1.0),
                                W.STONE, rot=R))
        # lintel + passage roof
        gate.append(box(L(0, H - 1.2, 0), (11.0, 8.5, 12.0), W.STONE, rot=R))
        for k in np.linspace(-5.2, 5.2, 6):
            gate.append(box(L(k, H + 3.4, 5.6), (1.5, 1.6, 1.0), W.STONE, rot=R))
        # dark interior of the passage
        gate.append(box(L(0, 4.4, 0), (9.6, 9.0, 11.0), np.array([0.06, 0.06, 0.08])))
        # wooden doors, swung open against the passage walls
        for sgn in (-1, 1):
            gate.append(box(L(sgn * 5.6, 5.6, 3.4), (0.5, 11.2, 7.4),
                            np.array([0.41, 0.27, 0.17]), rot=R))
        # road crossing the wall
        road = []
        for k in range(-11, 12):
            road.append(L(0, -0.45, k * 3.2))
        self.road = W.ribbon(np.array(road), [(-5.0, 0), (5.0, 0)],
                             np.array([0.62, 0.57, 0.46]))
        self.gate = gate
        self.center = g + np.array([0, 7.0, 0])
        self.R = R

class GateShot(Shot):
    name = "gate"

    def build(self):
        self.world = W.World(seed=3, cell=8.0, extent=720.0, wall_height=13.0,
                             trees=50, boulders=35, towers=False, wall_gap=(0.455, 0.545))
        self.g = GateBuild(self.world, 0.5)

    def frame(self, t):
        u = ease_io(self.u(t))
        g = self.g
        d = -g.right * 1.0 + g.fwd * 0.30 + np.array([0, 0.30 + 0.04 * u, 0])
        c = fit_view(self.world, g.gate, d, fov=46, margin=1.75 - 0.12 * u, t=t,
                     shake_amp=0.10, target=g.pos + np.array([0, 8.0, 0]))
        meshes = list(self.world.static) + g.gate + [g.road]
        # caravans queued outside
        for k in range(3):
            p = g.pos - g.fwd * (16 + k * 10) + g.right * (k * 2.5 - 2.5)
            meshes += W.cart(p, yaw=g.yaw + 0.08 * k, scale=1.0)
            meshes += W.dust(p + np.array([0, 0.4, 0]), t + k, count=5, spread=4.0,
                             color=(0.72, 0.66, 0.54), seed=20 + k)
        # traders walking in and out
        for k in range(5):
            travel = (t * 2.4 + k * 6.0) % 34
            p = g.pos - g.fwd * (18 - travel) + g.right * (3.4 - 0.9 * k)
            p = np.array([p[0], self.world.height_at(p[0], p[2]) + 0.1, p[2]])
            meshes += W.villager(p, yaw=g.yaw + np.pi, phase=t * 4.0 + k, scale=0.95)
        # guards
        for s in (-1, 1):
            meshes += W.soldier(g.pos + g.right * (s * 6.6) + g.fwd * 4.4 + np.array([0, 0.1, 0]),
                                yaw=g.yaw + np.pi, phase=0.0, scale=1.0, accent=W.RED,
                                shield=True)
        meshes += W.cloud_bank((g.pos[0], 190, g.pos[2]), t, count=6, scale=1.2,
                               spread=260, seed=15)
        return meshes, c, look_day()


# ---------------------------------------------------------------- 11. barracks
class BarracksShot(Shot):
    name = "barracks"

    def build(self):
        self.world = W.World(seed=3, cell=8.0, extent=700.0, wall_height=11.0,
                             trees=40, boulders=30)
        pts = self.world.path_points(4.4)
        self.pts = pts
        i = int(len(pts) * 0.34)
        g = pts[i]
        fwd, right = path_frame(pts, i)
        self.fwd, self.right, self.pos = fwd, right, g
        yaw = float(np.arctan2(fwd[0], fwd[2]))
        R = rot_y(yaw)
        self.yaw = yaw
        camp = []
        base = g + right * 20
        camp.append(box(base + np.array([0, 2.4, 0]), (12, 4.8, 8), W.EARTH * 1.08, rot=R))
        camp.append(box(base + np.array([0, 5.1, 0]), (13.8, 0.7, 9.2), W.TILE, rot=R))
        camp.append(box(base + R @ np.array([0, 1.5, 4.1]), (1.8, 3.0, 0.4), W.WOOD, rot=R))
        for k in range(5):
            camp.append(box(base + R @ np.array([-5 + 2.5 * k, 2.4, 4.05]), (1.4, 1.4, 0.25),
                            np.array([0.10, 0.10, 0.12]), rot=R))
        # spear rack
        for k in range(7):
            p = g + right * (8 + 0.55 * k)
            camp.append(box(p + np.array([0, 1.5, 0]), (0.09, 3.0, 0.09), W.WOOD))
            camp.append(prism(p + np.array([0, 3.2, 0]), 0.1, 0.45, 5,
                              np.array([0.78, 0.79, 0.83])))
        camp.append(icosphere(g + right * 13 + np.array([0, 0.5, 0]), 1.5, W.EARTH, squash=0.35))
        self.camp = camp
        self.center = g + right * 12 + np.array([0, 3.4, 0])

    def frame(self, t):
        u = ease_io(self.u(t))
        g, right, fwd = self.pos, self.right, self.fwd
        eye = g + right * (54 - 6 * u) + fwd * 10 + np.array([0, 15, 0])
        eye[1] = max(eye[1], self.world.height_at(eye[0], eye[2]) + 6.0)
        c = Camera(eye + shake(t, 0.11), g + right * 16 + np.array([0, 3.0, 0]),
                   fov=46, w=W_IMG, h=H_IMG, near=0.4)
        meshes = list(self.world.static) + self.camp
        # guards standing in line
        for k in range(5):
            p = g + right * (7 + k * 2.0) + fwd * (3.0 - 1.2 * k)
            meshes += W.soldier(p + np.array([0, 0.1, 0]), yaw=self.yaw + np.pi,
                                phase=0.0, scale=1.0, accent=W.RED, shield=(k % 2 == 0))
        # two patrolling
        for k in range(2):
            s = np.sin(t * 0.7 + k * 3.1)
            p = g + right * (4 + 5 * s) + fwd * (7 + 2 * k)
            meshes += W.soldier(p, yaw=float(self.yaw + (0.0 if np.cos(t * 0.7 + k * 3.1) > 0 else np.pi)),
                                phase=t * 4.2 + k, scale=1.0, accent=W.BLUE)
        meshes += W.fire_glow(g + right * 13 + np.array([0, 0.3, 0]), t, scale=0.9, seed=3)
        meshes += []
        return meshes, c, look_day()


# ---------------------------------------------------------------- 12/13. beacons
class BeaconDayShot(Shot):
    name = "beacon_day"

    def build(self):
        self.world = W.World(seed=3, cell=8.0, extent=820.0, wall_height=11.0,
                             trees=90, boulders=45, tower_every=0.12)
        self.pts = self.world.path_points(4.4)
        self.i0 = int(len(self.pts) * 0.34)

    def frame(self, t):
        u = ease_io(self.u(t))
        p, i0 = self.pts, self.i0
        a = p[i0] + np.array([0, 6, 0])
        fwd, right = path_frame(p, i0)
        az = float(np.arctan2(right[2], right[0])) + 0.35 + 0.4 * u
        c = wall_shot(self.world, p, i0 + int(6 * u), lat=52, up=34, ahead=150,
                      fov=46, t=t, shake_amp=0.06, look_up=16)
        meshes = list(self.world.static)
        top = self.world.wall_h + 13.4
        for i in range(6, len(p) - 6, 9):
            q = p[i] + np.array([0, top, 0])
            meshes += W.smoke_plume(q, t + i * 0.4, height=46, width=4.2, puffs=8,
                                    drift=(0.9, 0.5))
            meshes += W.fire_glow(q, t + i, scale=0.6, seed=i)
        meshes += tower_flags(self.world, t)
        meshes += W.birds((a[0] + 40, a[1] + 60, a[2]), t, count=6, radius=40, seed=11)
        return meshes, c, look_day()


class BeaconNightShot(Shot):
    name = "beacon_night"

    def build(self):
        self.world = W.World(seed=3, cell=8.0, extent=820.0, wall_height=11.0,
                             trees=90, boulders=45, tower_every=0.12)
        self.pts = self.world.path_points(4.4)
        self.i0 = int(len(self.pts) * 0.34)

    def frame(self, t):
        u = ease_io(self.u(t))
        p, i0 = self.pts, self.i0
        a = p[i0] + np.array([0, 6, 0])
        fwd, right = path_frame(p, i0)
        az = float(np.arctan2(right[2], right[0])) + 0.55 + 0.3 * u
        c = wall_shot(self.world, p, i0 + 14 + int(8 * u), lat=16, up=9, ahead=34,
                      fov=50, t=t, shake_amp=0.08, look_up=17)
        meshes = list(self.world.static)
        top = self.world.wall_h + 13.4
        for i in range(6, len(p) - 6, 9):
            q = p[i] + np.array([0, top, 0])
            meshes += W.fire_glow(q, t * 1.4 + i, scale=1.6, seed=i)
            meshes += W.smoke_plume(q + np.array([0, 2.0, 0]), t + i * 0.2, height=54,
                                    width=5.0, puffs=8, color=(0.26, 0.25, 0.32),
                                    drift=(0.8, 0.4))
        meshes += tower_flags(self.world, t)
        meshes += mist_bank((a[0], a[1] - 8, a[2]), t, count=8, spread=110, alpha=0.13,
                            color=(0.26, 0.30, 0.44))
        return meshes, c, look_night()


# ---------------------------------------------------------------- 14. mongols
class MongolShot(Shot):
    name = "mongol"

    def build(self):
        self.world = W.World(seed=5, cell=9.0, extent=760.0, wall_height=11.0,
                             trees=30, boulders=40, ridge_amp=18.0)
        self.pts = self.world.path_points(4.4)
        i0 = int(len(self.pts) * 0.5)
        self.wallp = self.pts[i0]
        self.fwd, self.right = path_frame(self.pts, i0)
        self.plain = self.wallp + self.right * 55
        self.plain[1] = self.world.height_at(self.plain[0], self.plain[2])

    def frame(self, t):
        u = ease_io(self.u(t))
        plain, fwd, right = self.plain, self.fwd, self.right
        c0 = plain + np.array([0, 2.5, 0])
        az = float(np.arctan2(fwd[2], fwd[0])) + 1.15
        eye = self.plain + right * 24 - fwd * 4 + np.array([0, 3.4, 0])
        eye[1] = max(eye[1], self.world.height_at(eye[0], eye[2]) + 3.5)
        tgt = self.plain + right * 2 + np.array([0, 2.6, 0])
        c = Camera(eye + shake(t, 0.20, 2.0), tgt, fov=50, w=W_IMG, h=H_IMG, near=0.4)
        meshes = list(self.world.static)
        for k in range(8):
            ph = t * 5.0 + k * 1.1
            row = k // 4
            lat = (k % 4) * 4.6 - 7.0
            advance = (t * 7.0 + row * 12.0 + k * 1.6) % 46
            pos = plain + right * lat - fwd * (34 - advance)
            pos = np.array([pos[0], self.world.height_at(pos[0], pos[2]) + 0.2, pos[2]])
            meshes += W.horse_rider(pos, yaw=float(np.arctan2(fwd[0], fwd[2])) + 0.1 * np.sin(ph),
                                    phase=ph, scale=1.0,
                                    banner=W.RED if k % 3 == 0 else None)
        meshes += W.dust(plain + right * 4 + np.array([0, 1.0, 0]) - fwd * 10, t,
                         count=24, spread=30, color=(0.80, 0.72, 0.54), seed=31)
        return meshes, c, look_day()


# ---------------------------------------------------------------- 15. betrayal
class BetrayShot(Shot):
    name = "betray"

    def build(self):
        self.world = W.World(seed=3, cell=8.0, extent=720.0, wall_height=13.0,
                             trees=50, boulders=35, towers=False, wall_gap=(0.455, 0.545))
        self.g = GateBuild(self.world, 0.5)

    def frame(self, t):
        u = ease_io(self.u(t))
        g = self.g
        az = float(np.arctan2(g.fwd[2], g.fwd[0])) + np.pi + 0.6 - 0.25 * u
        d = -g.right * 1.0 + g.fwd * 0.18 + np.array([0, 0.16, 0])
        c = fit_view(self.world, g.gate, d, fov=46, margin=1.35 - 0.08 * u, t=t,
                     shake_amp=0.16, target=g.pos + np.array([0, 7.0, 0]))
        meshes = list(self.world.static) + g.gate + [g.road]
        for s in (-1, 1):
            p = g.pos + g.right * (s * 6.4) + g.fwd * 5.0 + np.array([0, 3.4, 0])
            meshes.append(box(p, (0.3, 1.7, 0.3), W.WOOD))
            meshes += W.fire_glow(p + np.array([0, 1.0, 0]), t + s, scale=0.9,
                                  seed=int(41 + s))
        for k in range(8):
            travel = (t * 3.4 + k * 5.5) % 42
            p = g.pos - g.fwd * (24 - travel) + g.right * (2.2 * np.sin(k * 2.1))
            p = np.array([p[0], self.world.height_at(p[0], p[2]) + 0.1, p[2]])
            meshes += W.soldier(p, yaw=g.yaw + np.pi, phase=t * 4.6 + k, scale=1.0,
                                armor=W.ARMOR, accent=np.array([0.30, 0.10, 0.10]),
                                spear=True, shield=(k % 3 == 0))
        meshes += W.dust(g.pos - g.fwd * 8 + np.array([0, 1.0, 0]), t, count=18,
                         spread=18, color=(0.55, 0.50, 0.44), seed=44)
        meshes += W.cloud_bank((g.pos[0], 130, g.pos[2]), t, count=5, scale=1.2,
                               spread=150, seed=19)
        lk = look_night()
        lk.light.fog_density = 0.0028
        return meshes, c, lk


# ---------------------------------------------------------------- 16. ruin
class RuinShot(Shot):
    name = "ruin"

    def build(self):
        self.world = W.World(seed=9, cell=8.0, extent=700.0, wall_height=9.0,
                             trees=140, boulders=50)
        self.pts = self.world.path_points(4.4)
        self.ruin = self._ruined(self.pts)
        self.i0 = int(len(self.pts) * 0.5)

    def _ruined(self, pts):
        rng = np.random.default_rng(6)
        out = []
        n = len(pts)
        for i in range(4, n - 6, 4):
            if rng.random() < 0.3:
                continue
            a, b = pts[i], pts[i + 4]
            mid = (a + b) / 2
            seg = b - a
            length = float(np.linalg.norm(seg))
            yaw = float(np.arctan2(seg[0], seg[2]))
            R = rot_y(yaw)
            h = rng.uniform(1.5, 5.0) if rng.random() < 0.55 else rng.uniform(5.0, 8.0)
            out.append(box(mid + np.array([0, h * 0.5 - 1.2, 0]),
                           (length * 1.02, h + 2.0, 7.4),
                           np.array([0.50, 0.50, 0.47]) * rng.uniform(0.82, 1.06), rot=R))
            for _ in range(int(rng.integers(0, 3))):
                off = R @ np.array([rng.uniform(-4, 4), 0, rng.uniform(-8, 8)])
                out.append(box(mid + off + np.array([0, 0.3, 0]), (0.9, 0.6, 1.2),
                               np.array([0.52, 0.51, 0.48]), rot=rot_y(rng.uniform(0, 3))))
        for k in range(42):
            i = int(rng.integers(4, n - 4))
            p = pts[i] + np.array([rng.normal(0, 4), 0, rng.normal(0, 4)])
            y = self.world.height_at(p[0], p[2])
            h = rng.uniform(5, 11)
            out.append(box([p[0], y + h * 0.2, p[2]], (0.5, h * 0.45, 0.5), W.TRUNK))
            out.append(icosphere([p[0], y + h * 0.78, p[2]], h * 0.32,
                                 W.PINE if rng.random() < 0.6 else W.GRASS_L * 0.85,
                                 squash=0.85))
        return out

    def frame(self, t):
        u = ease_io(self.u(t))
        p, i0 = self.pts, self.i0
        a = p[i0] + np.array([0, 2, 0])
        fwd, right = path_frame(p, i0)
        az = float(np.arctan2(right[2], right[0])) + 0.5 + 0.45 * u
        c = wall_shot(self.world, p, i0, lat=30 - 6 * u, up=16 + 8 * u,
                      ahead=60, fov=48, t=t, shake_amp=0.08, look_up=2)
        meshes = list(self.world.static) + self.ruin
        meshes += mist_bank((a[0], a[1] - 2, a[2]), t, count=10, spread=90, alpha=0.14)
        meshes += W.birds((a[0], a[1] + 40, a[2]), t, count=4, radius=30, seed=12)
        lk = look_day()
        lk.light.fog_density = 0.0016
        return meshes, c, lk


# ---------------------------------------------------------------- 17. villager
class VillagerShot(Shot):
    name = "villager"

    def build(self):
        self.world = W.World(seed=9, cell=8.0, extent=700.0, wall_height=9.0,
                             trees=110, boulders=40)
        pts = self.world.path_points(4.4)
        self.pts = pts
        i = int(len(pts) * 0.46)
        g = pts[i] + path_frame(pts, i)[1] * 20
        g[1] = self.world.height_at(g[0], g[2])
        self.g = g
        fwd, right = path_frame(pts, i)
        self.fwd, self.right = fwd, right
        yaw = float(np.arctan2(fwd[0], fwd[2]))
        self.yaw = yaw
        R = rot_y(yaw)
        hut = []
        hut.append(box(g + np.array([0, 2.5, 0]), (10, 5.0, 8), W.EARTH * 1.1, rot=R))
        hut.append(pyramid(g + np.array([0, 5.1, 0]), (12, 10), 3.6, W.TILE, rot=R))
        hut.append(box(g + R @ np.array([0, 1.4, 4.0]), (1.7, 2.8, 0.4), W.WOOD, rot=R))
        rng = np.random.default_rng(3)
        for k in range(27):
            layer, idx = k // 9, k % 9
            off = R @ np.array([-2.4 + (idx % 3) * 1.2, 0.24 + layer * 0.46, (idx // 3) * 0.95])
            hut.append(box(g + off + R @ np.array([4.6, 0, 1.0]), (1.05, 0.44, 0.85),
                           np.array([0.55, 0.52, 0.48]) * rng.uniform(0.86, 1.08), rot=R))
        self.hut = hut
        self.center = g + np.array([2.0, 2.6, 0])

    def frame(self, t):
        u = ease_io(self.u(t))
        g = self.g
        az = float(np.arctan2(self.right[2], self.right[0])) + 0.75
        d = self.right * 1.0 + self.fwd * 0.5 + np.array([0, 0.26, 0])
        c = fit_view(self.world, self.hut, d, fov=48, margin=2.0 - 0.15 * u, t=t,
                     shake_amp=0.09, target=self.center)
        meshes = list(self.world.static) + self.hut
        for k in range(2):
            p = g + self.right * (3.2 + 2.0 * k) + self.fwd * 1.0
            meshes += W.villager(p + np.array([0, 0.1, 0]), yaw=self.yaw + np.pi * 1.5,
                                 phase=0.3 * np.sin(t * 2 + k), scale=0.95)
        meshes += W.dust(g + self.right * 4 + np.array([0, 0.6, 0]), t, count=6, spread=3,
                         color=(0.72, 0.68, 0.58), seed=51)
        meshes += W.cloud_bank((g[0], 110, g[2]), t, count=5, scale=1.0, spread=140, seed=23)
        return meshes, c, look_golden()


# ---------------------------------------------------------------- 18. final
class FinalShot(Shot):
    name = "final"

    def build(self):
        self.world = W.World(seed=3, cell=8.0, extent=820.0, wall_height=11.0,
                             trees=100, boulders=45, tower_every=0.12)
        self.pts = self.world.path_points(4.4)
        self.i0 = int(len(self.pts) * 0.42)

    def frame(self, t):
        u = ease_io(self.u(t))
        p, i0 = self.pts, self.i0
        a = p[i0] + np.array([0, 6, 0])
        fwd, right = path_frame(p, i0)
        az = float(np.arctan2(right[2], right[0])) + 0.45
        c = wall_shot(self.world, p, i0, lat=44 + 30 * u, up=24 + 24 * u,
                      ahead=95 + 35 * u, fov=52, t=t, shake_amp=0.06, look_up=12)
        meshes = list(self.world.static) + tower_flags(self.world, t, color=W.GOLD)
        for i in range(4, len(p) - 4, 6):
            q = p[i] + np.array([0, self.world.wall_h + 13.6, 0])
            meshes += W.fire_glow(q, t + i, scale=0.85, seed=i)
            meshes.append(icosphere(q + np.array([0, 1.6, 0]), 4.2,
                                    np.array([1.0, 0.62, 0.28]), glow=1.0, alpha=0.20))
        meshes += mist_bank((a[0], a[1] - 6, a[2]), t, count=12, spread=120, alpha=0.15,
                            color=(0.45, 0.52, 0.70))
        meshes += W.birds((a[0] + 30, a[1] + 40, a[2]), t, count=3, radius=26, seed=5)
        # moon
        moon = a + np.array([-260, 240, -420])
        meshes.append(icosphere(moon, 26, np.array([0.95, 0.96, 1.0]), glow=1.0))
        meshes.append(icosphere(moon, 70, np.array([0.72, 0.80, 1.0]), glow=1.0, alpha=0.10))
        return meshes, c, look_blue()


SHOT_CLASSES = [HeroShot, ManyWallsShot, OrbitShot, CamoShot, WorkerShot, ChiselShot,
                MemorialShot, RiceShot, CutawayShot, GateShot, BarracksShot,
                BeaconDayShot, BeaconNightShot, MongolShot, BetrayShot, RuinShot,
                VillagerShot, FinalShot]
