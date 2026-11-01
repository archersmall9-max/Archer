"""
world.py — procedural low-poly world: ridge terrain, the Great Wall (crenellated,
towered), and the animated props (soldiers, mongol cavalry, flags, smoke, trees,
ruins, villagers, birds, clouds).

Everything here is plain geometry generated with numpy, so any camera can render it.
"""
from __future__ import annotations

import numpy as np

from engine3d import (Mesh, box, icosphere, normalize, pad_faces, prism, pyramid,
                      ribbon, rot_x, rot_y, smooth)


# --------------------------------------------------------------------- noise
def _hash2(ix, iz, seed=0):
    h = (ix * 374761393 + iz * 668265263 + seed * 2654435761) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFF) / float(0xFFFFFF)


def vnoise(x, z, seed=0):
    ix = np.floor(x).astype(np.int64)
    iz = np.floor(z).astype(np.int64)
    fx = x - ix
    fz = z - iz
    sx = fx * fx * (3 - 2 * fx)
    sz = fz * fz * (3 - 2 * fz)
    n00 = _hash2(ix, iz, seed)
    n10 = _hash2(ix + 1, iz, seed)
    n01 = _hash2(ix, iz + 1, seed)
    n11 = _hash2(ix + 1, iz + 1, seed)
    return (n00 * (1 - sx) + n10 * sx) * (1 - sz) + (n01 * (1 - sx) + n11 * sx) * sz


def fbm(x, z, octaves=4, seed=0, gain=0.5, freq=1.0):
    a, f, s = 1.0, freq, 0.0
    norm = 0.0
    for o in range(octaves):
        s = s + a * vnoise(x * f, z * f, seed + o * 17)
        norm += a
        a *= gain
        f *= 2.05
    return s / norm


def ridge(x, z, seed=0):
    return 1 - np.abs(fbm(x, z, 3, seed) * 2 - 1)


# --------------------------------------------------------------------- palette
GRASS = np.array([0.30, 0.48, 0.20])
GRASS_D = np.array([0.20, 0.34, 0.15])
GRASS_L = np.array([0.46, 0.60, 0.26])
ROCK = np.array([0.42, 0.41, 0.39])
ROCK_D = np.array([0.34, 0.34, 0.33])
STONE = np.array([0.50, 0.49, 0.47])
STONE_D = np.array([0.41, 0.40, 0.38])
EARTH = np.array([0.55, 0.42, 0.28])
SAND = np.array([0.72, 0.63, 0.45])
WOOD = np.array([0.34, 0.24, 0.16])
WOOD_L = np.array([0.52, 0.38, 0.24])
TILE = np.array([0.25, 0.25, 0.27])
TRUNK = np.array([0.29, 0.21, 0.15])
BLOOM = np.array([0.90, 0.63, 0.70])
PINE = np.array([0.16, 0.33, 0.19])
ARMOR = np.array([0.15, 0.14, 0.16])
ARMOR_2 = np.array([0.24, 0.22, 0.25])
RED = np.array([0.62, 0.11, 0.12])
BLUE = np.array([0.16, 0.30, 0.62])
GOLD = np.array([0.85, 0.68, 0.24])
SKIN = np.array([0.82, 0.65, 0.50])
WHITE = np.array([0.93, 0.93, 0.95])


# --------------------------------------------------------------------- path
def make_path(n=140, length=600.0, wiggle=1.0, seed=3):
    """A snaking ridge path on the XZ plane."""
    t = np.linspace(0, 1, n)
    x = (t - 0.5) * length * 0.92
    z = length * 0.26 * np.sin(t * 4.4 + 0.5) * wiggle + length * 0.09 * np.sin(t * 9.7)
    x = x + 46 * np.sin(t * 6.3 + 1.1) * wiggle
    z = z + 26 * np.sin(t * 12.0) * wiggle
    return np.stack([x, np.zeros_like(x), z], 1)


class World:
    """Holds the height field + the static geometry of one scene."""

    def __init__(self, seed=3, cell=13.0, extent=760.0, path_n=118, wall_height=8.5,
                 wall_on=True, wall_material="stone", towers=True, tower_every=0.10,
                 trees=90, boulders=40, ridge_amp=26.0, seed_trees=11, snow_line=None,
                 wall_gap=None):
        self.seed = seed
        self.cell = cell
        self.extent = extent
        self.wall_on = wall_on
        self.wall_h = wall_height
        self.wall_material = wall_material
        self.path = make_path(path_n, seed=seed)
        self.ridge_amp = ridge_amp
        self.snow_line = snow_line
        self.wall_gap = wall_gap
        self.static: list[Mesh] = []

        g = np.linspace(-extent / 2, extent / 2, int(extent / cell) + 1)
        self.gx, self.gz = np.meshgrid(g, g, indexing="xy")
        self.h = self._height(self.gx, self.gz)
        self.static.append(self._terrain_mesh())

        if wall_on:
            self.static.append(self._wall_mesh())
            if towers:
                self.static.extend(self._towers(tower_every))
        if trees:
            self.static.extend(self._trees(trees, seed_trees))
        if boulders:
            self.static.extend(self._boulders(boulders, seed_trees + 5))

    # -- height field ------------------------------------------------------
    def _dist_to_path(self, X, Z):
        px, pz = self.path[:, 0], self.path[:, 2]
        d = (X[..., None] - px) ** 2 + (Z[..., None] - pz) ** 2
        return np.sqrt(d.min(axis=-1))

    def _height(self, X, Z):
        s = self.seed
        base = (26.0 * fbm(X * 0.0032, Z * 0.0032, 4, s)
                + 12.0 * fbm(X * 0.0085, Z * 0.0085, 3, s + 3)
                + 4.0 * fbm(X * 0.024, Z * 0.024, 2, s + 9))
        peaks = np.clip(ridge(X * 0.0052 + 3.1, Z * 0.0052, s + 21) - 0.30, 0, 1) ** 1.4
        base = base + peaks * 52.0
        # keep the border lower so the horizon stays clean
        r = np.sqrt(X ** 2 + Z ** 2) / (self.extent * 0.5)
        base = base * (1.0 - 0.35 * np.clip(r - 0.45, 0, 1) ** 1.5)
        if self.wall_on:
            d = self._dist_to_path(X, Z)
            bump = np.exp(-(d / (self.extent * 0.052)) ** 2)
            base = base + bump * self.ridge_amp
            base = base - np.exp(-(d / 7.0) ** 2) * 1.4  # flatten the crest line
        return base

    def height_at(self, x, z):
        return float(self._height(np.array([[float(x)]]), np.array([[float(z)]]))[0, 0])

    def path_points(self, spacing=4.6):
        """Resample the wall path and lift it onto the terrain."""
        p = self.path
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        n = max(8, int(s[-1] / spacing))
        u = np.linspace(0, s[-1], n)
        x = np.interp(u, s, p[:, 0])
        z = np.interp(u, s, p[:, 2])
        y = self._height(x, z)
        # smooth the crest so the wall doesn't zig-zag vertically
        for _ in range(3):
            y[1:-1] = (y[:-2] + 2 * y[1:-1] + y[2:]) / 4
        return np.stack([x, y, z], 1)

    # -- terrain -----------------------------------------------------------
    def _terrain_mesh(self):
        X, Z = self.gx, self.gz
        H = self.h
        nx, nz = X.shape
        verts = np.stack([X, H, Z], -1).reshape(-1, 3)
        stride = nz
        ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(nz - 1), indexing="ij")
        a = (ii * stride + jj).ravel()
        b = ((ii + 1) * stride + jj).ravel()
        c = ((ii + 1) * stride + jj + 1).ravel()
        d = (ii * stride + jj + 1).ravel()
        faces = np.stack([d, c, b, a], 1)

        # slope-based colouring
        e1 = verts[b] - verts[a]
        e2 = verts[d] - verts[a]
        nrm = normalize(np.cross(e2, e1))
        slope = 1 - np.abs(np.clip(nrm[:, 1], -1, 1))
        nz1 = fbm(X[:-1, :-1].ravel() * 0.05, Z[:-1, :-1].ravel() * 0.05, 2, self.seed + 31)

        col = np.zeros((len(faces), 3))
        grass = GRASS_D[None, :] + (GRASS_L - GRASS_D)[None, :] * (0.25 + 0.75 * nz1[:, None])
        col[:] = grass
        rock_mask = slope > 0.52
        col[rock_mask] = (ROCK[None, :] * (0.80 + 0.35 * nz1[rock_mask, None]))
        if self.snow_line is not None:
            snow = (H[:-1, :-1].ravel() > self.snow_line) & (slope < 0.55)
            col[snow] = np.array([0.92, 0.93, 0.97])[None, :] * (0.9 + 0.12 * nz1[snow, None])
        cliff = slope > 0.74
        col[cliff] = ROCK[None, :] * (0.85 + 0.3 * nz1[cliff, None]) * 0.92
        if self.wall_on:
            cx = X[:-1, :-1].ravel() + self.cell / 2
            cz = Z[:-1, :-1].ravel() + self.cell / 2
            d = self._dist_to_path(cx, cz)
            earth = np.clip(1 - (d - 5.0) / 18.0, 0, 1)[:, None] * 0.16
            col = col * (1 - earth) + SAND[None, :] * earth
        return Mesh(verts, faces, col)

    # -- wall --------------------------------------------------------------
    def _wall_mesh(self):
        pts = self.path_points(4.4)
        n = len(pts)
        stone = STONE if self.wall_material == "stone" else EARTH
        H = self.wall_h
        bw, tw = 4.9, 3.6
        par_h, par_w = 1.45, 0.58
        walk = tw - par_w
        tans = np.zeros_like(pts)
        tans[1:-1] = pts[2:] - pts[:-2]
        tans[0] = pts[1] - pts[0]
        tans[-1] = pts[-1] - pts[-2]
        tans[:, 1] = 0
        tans = normalize(tans)
        side = normalize(np.cross(tans, np.array([0.0, 1.0, 0.0])))

        # cross-section: three masonry courses per face, crenellated parapet on top
        courses = [(-7.0, bw + 0.45)]
        for f in (0.34, 0.67):
            y = H * f
            courses.append((y, bw + (tw - bw) * ((y + 7.0) / (H + 7.0))))
        courses.append((H, tw))

        rings = []
        for i in range(n):
            merlon = (i % 5) < 2
            hh = H + par_h + (0.55 if merlon else 0.0)
            ring = []
            for y, sx in courses:                       # outer face, bottom -> top
                ring.append((sx, y))
            ring.append((tw, hh))                       # parapet outer
            ring.append((walk, hh))                     # parapet cap
            ring.append((walk, H))                      # parapet inner
            ring.append((-walk, H))                     # walkway
            ring.append((-walk, hh))                    # far parapet inner
            ring.append((-tw, hh))                      # far parapet cap
            ring.append((-tw, H))                       # far parapet outer
            for y, sx in reversed(courses[:-1]):        # inner face, top -> bottom
                ring.append((-sx, y))
            rings.append([pts[i] + side[i] * a + np.array([0, b, 0]) for a, b in ring])
        rings = np.array(rings)
        npf = rings.shape[1]
        flat = rings.reshape(-1, 3)

        faces, cols = [], []
        nb = npf + 1
        for i in range(n - 1):
            if self.wall_gap is not None:
                g0, g1 = self.wall_gap
                if g0 <= i / (n - 1) <= g1:
                    continue
            for k in range(npf):
                k2 = (k + 1) % npf
                a = i * npf + k
                faces.append([a, a + 1, a + 1 + npf, a + npf])
                # masonry tinting: courses, walkway, parapet caps
                r1 = _hash2(np.int64(i // 2), np.int64(k), self.seed) * 0.28 - 0.14
                if k in (3, 4):          # parapet cap band
                    tint = 1.10 + r1 * 0.5
                elif k == 5:             # walkway
                    tint = 1.04 + r1 * 0.4
                elif 6 <= k <= 8:        # far parapet
                    tint = 1.02 + r1 * 0.5
                elif k < 3 or k > 10:    # faces: three courses
                    tint = (0.92 + 0.10 * (k % 3)) + r1
                else:
                    tint = 0.98 + r1
                cols.append(np.clip(stone * tint, 0.02, 1.0))
        faces.append(list(range(npf))[::-1])
        cols.append(stone * 0.6)
        base = (n - 1) * npf
        faces.append(list(range(base, base + npf)))
        cols.append(stone * 0.7)
        m = Mesh(flat, pad_faces(faces), np.array(cols))
        return m

    def _towers(self, every=0.10):
        pts = self.path_points(4.4)
        n = len(pts)
        out = []
        idxs = np.linspace(4, n - 5, max(2, int(round(1.0 / every)))).astype(int)
        for k, i in enumerate(idxs):
            p = pts[i]
            t = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
            t[1] = 0
            ang = float(np.arctan2(t[0], t[2]))
            R = rot_y(ang)
            H = self.wall_h
            w = 8.6 + (0.8 if k % 2 else 0.0)
            out.append(box(p + np.array([0, H * 0.5 + 1.2, 0]), (w, H + 4.0, w), STONE, rot=R))
            # upper gallery + roof
            out.append(box(p + np.array([0, H + 5.9, 0]), (w * 0.86, 3.4, w * 0.86), STONE_D, rot=R))
            out.append(pyramid(p + np.array([0, H + 7.6, 0]), (w * 1.12, w * 1.12), 4.6, TILE,
                               rot=R, overhang=1.0))
            # arrow windows (dark inset faces on the two long sides)
            for s in (-1, 1):
                for o in (-1.6, 1.6):
                    out.append(box(p + R @ np.array([o, H * 0.62, s * (w / 2 - 0.15)]),
                                   (1.5, 2.6, 0.5), np.array([0.10, 0.10, 0.12]), rot=R))
            # battlement ring around the roof base
            for s in (-1, 1):
                for o in np.linspace(-w * 0.4, w * 0.4, 3):
                    out.append(box(p + R @ np.array([o, H + 4.6, s * (w / 2 - 0.3)]),
                                   (1.3, 1.1, 0.9), STONE, rot=R))
        return out

    # -- props -------------------------------------------------------------
    def _trees(self, count, seed):
        rng = np.random.default_rng(seed)
        out = []
        pts = self.path_points(6.0)
        for _ in range(count):
            if rng.random() < 0.6:
                i = rng.integers(0, len(pts))
                off = rng.normal(0, 26)
                t = pts[min(i + 1, len(pts) - 1)] - pts[i]
                t[1] = 0
                t = normalize(t)
                side = np.cross(t, np.array([0.0, 1.0, 0.0]))
                p = pts[i] + side * off
            else:
                p = np.array([rng.uniform(-1, 1) * self.extent * 0.42, 0,
                              rng.uniform(-1, 1) * self.extent * 0.42])
            y = self.height_at(p[0], p[2])
            if y < 2:
                continue
            h = rng.uniform(4.5, 9.0)
            pine = rng.random() < 0.55
            col = PINE if pine else (BLOOM if rng.random() < 0.3 else GRASS_L * 0.9)
            out.append(box([p[0], y + h * 0.18, p[2]], (0.5, h * 0.4, 0.5), TRUNK))
            if pine:
                for k in range(2):
                    out.append(prism([p[0], y + h * (0.45 + 0.28 * k), p[2]],
                                     h * (0.30 - 0.09 * k), h * 0.55, 6, col,
                                     taper=0.15))
            else:
                out.append(icosphere([p[0], y + h * 0.72, p[2]], h * 0.34, col, squash=0.9))
        return out

    def _boulders(self, count, seed):
        rng = np.random.default_rng(seed)
        out = []
        pts = self.path_points(8.0)
        for _ in range(count):
            i = rng.integers(0, len(pts))
            p = pts[i] + np.array([rng.normal(0, 22), 0, rng.normal(0, 22)])
            y = self.height_at(p[0], p[2])
            r = rng.uniform(0.8, 3.2)
            out.append(icosphere([p[0], y + r * 0.35, p[2]], r, ROCK_D * rng.uniform(0.85, 1.1),
                                 squash=rng.uniform(0.5, 0.8)))
        return out


# --------------------------------------------------------------------- actors
def soldier(pos, yaw=0.0, phase=0.0, scale=1.0, armor=ARMOR, accent=RED, spear=True,
            shield=False, cloak=None):
    """Low-poly warrior. phase animates the walk cycle."""
    R = rot_y(yaw)
    p = np.asarray(pos, float)
    s = scale
    out = []
    swing = np.sin(phase)
    bob = abs(np.sin(phase)) * 0.05 * s

    def P(v):
        return p + R @ (np.asarray(v, float) * s) + np.array([0, bob, 0])

    # legs (swing about the hip)
    for sign in (-1, 1):
        ang = np.sin(phase + (0 if sign < 0 else np.pi)) * 0.55
        hip = np.array([0.0, 0.78, sign * 0.14])
        leg_len = 0.85
        leg = box((0, 0, 0), (0.30, leg_len, 0.34), ARMOR_2)
        leg.transform(rot_x(ang))
        leg.moved(0)
        leg.v = leg.v + (P(hip) + R @ np.array([0, -leg_len * 0.5, 0]))
        out.append(leg)
        out.append(box(P(hip + np.array([0.02, -leg_len * 0.92, 0])), np.array([0.34, 0.18, 0.44]) * s,
                       np.array([0.20, 0.17, 0.15])))
    # body
    out.append(box(P((0, 1.34, 0)), np.array([0.62, 0.72, 0.44]) * s, armor, rot=R, taper=0.86))
    out.append(box(P((0, 1.06, 0)), np.array([0.66, 0.22, 0.48]) * s, accent, rot=R))
    # arms
    for sign in (-1, 1):
        ang = -np.sin(phase + (0 if sign < 0 else np.pi)) * 0.42
        sh = np.array([0.0, 1.62, sign * 0.34])
        arm = box((0, 0, 0), np.array([0.20, 0.72, 0.20]) * s, armor)
        arm.transform(rot_x(ang))
        arm.v = arm.v + P(sh) + R @ np.array([0, -0.34 * s, 0])
        out.append(arm)
    # head + helmet
    out.append(icosphere(P((0, 1.90, 0)), 0.20 * s, SKIN, subdiv=0, squash=1.05))
    out.append(prism(P((0, 2.02, 0)), 0.24 * s, 0.26 * s, 6, armor if cloak is None else cloak,
                     taper=0.55))
    out.append(box(P((0, 2.20, 0)), np.array([0.06, 0.22, 0.06]) * s, accent))
    if spear:
        out.append(box(P((0.36, 1.30, 0.10)), np.array([0.06, 2.5, 0.06]) * s, WOOD, rot=R))
        out.append(prism(P((0.36, 2.66, 0.10)), 0.07 * s, 0.30 * s, 5, np.array([0.75, 0.76, 0.8])))
    if shield:
        out.append(box(P((-0.34, 1.30, 0.05)), np.array([0.10, 0.85, 0.62]) * s, accent, rot=R))
    return out


def horse_rider(pos, yaw=0.0, phase=0.0, scale=1.0, coat=(0.28, 0.20, 0.14),
                rider_cloak=(0.45, 0.20, 0.16), banner=None):
    R = rot_y(yaw)
    p = np.asarray(pos, float)
    s = scale
    out = []

    def P(v):
        return p + R @ (np.asarray(v, float) * s)

    gallop = np.sin(phase)
    body_y = 1.45 + 0.12 * np.sin(phase * 2)
    out.append(box(P((0, body_y, 0)), np.array([0.72, 0.78, 2.0]) * s, coat, rot=R))
    out.append(box(P((0, body_y + 0.52, 0.34)), np.array([0.5, 0.4, 1.5]) * s, coat, rot=R))
    # neck + head
    out.append(box(P((0, body_y + 0.55, 1.05)), np.array([0.38, 0.9, 0.5]) * s, coat, rot=R))
    out.append(box(P((0, body_y + 0.95, 1.42)), np.array([0.34, 0.36, 0.7]) * s, coat, rot=R))
    # tail
    out.append(box(P((0, body_y + 0.35, -1.12)), np.array([0.18, 0.8, 0.22]) * s, (0.12, 0.09, 0.07), rot=R))
    # legs
    for sign in (-1, 1):
        for fz in (0.72, -0.72):
            ang = np.sin(phase + (0 if fz > 0 else 2.2) + (0 if sign < 0 else np.pi)) * 0.85
            leg = box((0, 0, 0), np.array([0.22, 1.35, 0.24]) * s, coat)
            leg.transform(rot_x(ang))
            leg.v = leg.v + P((sign * 0.28, body_y - 0.55, fz))
            out.append(leg)
    # rider
    out.extend(soldier(P((0, body_y + 0.30, -0.05)), yaw=yaw, phase=phase * 0.0, scale=0.85 * s,
                       accent=rider_cloak, spear=False))
    if banner is not None:
        out.append(box(P((0.42, body_y + 1.9, 0.1)), np.array([0.06, 2.6, 0.06]) * s, WOOD, rot=R))
        out.extend(flag(P((0.42, body_y + 2.9, 0.1)), yaw, 0.0, banner, scale=0.8 * s))
    return out


def flag(pos, yaw, t, color=BLUE, scale=1.0, pole_h=3.0):
    """Cloth banner with a travelling wave."""
    p = np.asarray(pos, float)
    R = rot_y(yaw)
    nx, ny = 6, 3
    w, h = 2.2 * scale, 1.35 * scale
    verts = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            u = i / nx
            v = j / ny
            wave = np.sin(u * 5.0 - t * 6.0) * 0.16 * u * scale
            wave2 = np.cos(u * 4.0 - t * 4.4 + v * 2.0) * 0.10 * u * scale
            local = np.array([wave2, h * v - h * 0.5 - v * 0.35 * u * scale, w * u, 1.0])
            verts.append(p + R @ np.array([local[2], local[1], local[0]]))
    faces = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            faces.append([a, a + 1, a + 1 + nx + 1, a + nx + 1])
    m = Mesh(np.array(verts), pad_faces(faces), color)
    return [m]


def smoke_plume(origin, t, height=26.0, width=4.0, color=(0.86, 0.86, 0.88), puffs=9, drift=(1.4, 0.0)):
    out = []
    o = np.asarray(origin, float)
    for i in range(puffs):
        f = i / max(1, puffs - 1)
        ph = (t * 0.22 + f * 0.9) % 1.0
        y = o[1] + ph * height
        r = width * (0.35 + ph * 1.9)
        x = o[0] + drift[0] * ph * 9 + np.sin(ph * 6 + i) * 2.0
        z = o[2] + drift[1] * ph * 9
        alpha = 0.55 * (1 - ph) ** 0.7
        out.append(icosphere([x, y, z], r, color, squash=0.85, alpha=alpha))
    return out


def fire_glow(origin, t, scale=1.0, seed=0):
    rng = np.random.default_rng(seed + int(t * 8))
    o = np.asarray(origin, float)
    out = []
    for i in range(3):
        hh = (2.2 + 1.4 * rng.random()) * scale
        out.append(prism([o[0] + rng.normal(0, 0.3), o[1] + hh * 0.5, o[2] + rng.normal(0, 0.3)],
                         (0.9 + 0.3 * rng.random()) * scale, hh, 4,
                         np.array([1.0, 0.62, 0.18]), taper=0.05, glow=1.0))
    out.append(icosphere([o[0], o[1] + 1.1 * scale, o[2]], 2.1 * scale,
                         np.array([1.0, 0.55, 0.16]), glow=1.0, alpha=0.42))
    return out


def birds(center, t, count=6, radius=30.0, seed=2):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        a = t * 0.35 + i * (2 * np.pi / count) + rng.uniform(-0.2, 0.2)
        r = radius * rng.uniform(0.6, 1.0)
        p = np.array([center[0] + np.cos(a) * r, center[1] + rng.uniform(-4, 6), center[2] + np.sin(a) * r])
        s = 0.9
        flap = np.sin(t * 7 + i) * 0.35
        q = [p + np.array([-s, 0, 0]), p + np.array([0, 0.15 * s, 0]), p + np.array([s, 0, 0]),
             p + np.array([s, 0.1 + flap, -0.2])]
        out.append(Mesh(np.array(q), pad_faces([[0, 1, 2], [2, 1, 3]]), np.array([0.16, 0.16, 0.18])))
    return out


def cloud_bank(center, t, count=7, scale=1.0, color=(0.98, 0.98, 1.0), spread=90.0, seed=5):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        p = np.array(center) + np.array([rng.uniform(-spread, spread) + t * 2.2,
                                         rng.uniform(-8, 14),
                                         rng.uniform(-spread, spread)])
        r = scale * rng.uniform(9, 22)
        out.append(icosphere(p, r, color, squash=0.42, alpha=0.96))
        out.append(icosphere(p + np.array([r * 1.1, r * 0.1, r * 0.5]), r * 0.7, color,
                             squash=0.40, alpha=0.96))
    return out


def dust(center, t, count=14, spread=16.0, color=(0.78, 0.70, 0.55), seed=9):
    rng = np.random.default_rng(seed)
    out = []
    c = np.asarray(center, float)
    for i in range(count):
        ph = (t * 0.5 + i / count) % 1.0
        p = c + np.array([rng.uniform(-spread, spread) * (0.4 + ph),
                          rng.uniform(0, 2.5) + ph * 1.5,
                          rng.uniform(-spread, spread) * (0.4 + ph)])
        out.append(icosphere(p, 1.6 + ph * 4.0, color, squash=0.7, alpha=0.22 * (1 - ph)))
    return out


def cart(origin, yaw=0.0, scale=1.0):
    p = np.asarray(origin, float)
    R = rot_y(yaw)
    out = [box(p + R @ (np.array([0, 1.0, 0]) * scale), np.array([2.0, 1.2, 3.4]) * scale, WOOD_L, rot=R,
               taper=1.0)]
    for s in (-1, 1):
        out.append(prism(p + R @ (np.array([s * 1.05, 0.75, 0.9]) * scale), 0.75 * scale, 0.18 * scale,
                         8, WOOD, rot=R @ rot_x(np.pi / 2)))
        out.append(prism(p + R @ (np.array([s * 1.05, 0.75, -1.1]) * scale), 0.75 * scale, 0.18 * scale,
                         8, WOOD, rot=R @ rot_x(np.pi / 2)))
    out.append(box(p + R @ (np.array([0, 2.0, 0.2]) * scale), np.array([1.7, 0.9, 2.2]) * scale,
                   np.array([0.66, 0.55, 0.36]), rot=R))
    return out


def villager(pos, yaw=0.0, phase=0.0, scale=1.0, shirt=(0.42, 0.36, 0.30)):
    out = soldier(pos, yaw=yaw, phase=phase, scale=scale, armor=shirt,
                  accent=np.array([0.30, 0.28, 0.26]), spear=False)
    # straw hat
    out.append(prism(np.asarray(pos, float) + np.array([0, 2.05 * scale, 0]), 0.42 * scale,
                     0.14 * scale, 8, np.array([0.80, 0.72, 0.45]), taper=0.25))
    return out
