"""
engine3d.py — a small software 3D renderer (flat-shaded, low-poly) built on numpy + Pillow.

No GPU, no browser, no external 3D library: geometry -> camera -> painter's-algorithm
rasteriser with per-face Lambert + hemisphere lighting, distance fog, emissive faces,
translucency, and a numpy/PIL post-processing stack (bloom, vignette, grade, grain).

Coordinates: Y is up. Camera looks down +Z in camera space.
Faces are wound counter-clockwise when seen from outside.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

D2R = np.pi / 180.0


# ----------------------------------------------------------------------------- math
def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def lerp(a, b, t):
    return a + (b - a) * t


def smooth(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def ease_io(t):
    t = min(1.0, max(0.0, t))
    return 0.5 - 0.5 * np.cos(np.pi * t)


def mix_color(c1, c2, t):
    c1 = np.asarray(c1, float)
    c2 = np.asarray(c2, float)
    return c1 + (c2 - c1) * t


# ----------------------------------------------------------------------------- mesh
class Mesh:
    """World-space triangle/quad soup with per-face albedo."""

    __slots__ = ("v", "f", "c", "glow", "alpha")

    def __init__(self, verts, faces, color, glow=0.0, alpha=1.0):
        self.v = np.asarray(verts, dtype=np.float64)
        f = np.asarray(faces, dtype=np.int32)
        if f.ndim == 1:
            f = f[None, :]
        self.f = f
        n = len(f)
        col = np.asarray(color, dtype=np.float64)
        if col.ndim == 1:
            col = np.tile(col, (n, 1))
        self.c = col
        self.glow = np.broadcast_to(np.asarray(glow, float), (n,)).copy()
        self.alpha = np.broadcast_to(np.asarray(alpha, float), (n,)).copy()

    def transform(self, M, offset=(0, 0, 0)):
        self.v = self.v @ np.asarray(M, float).T + np.asarray(offset, float)
        return self

    def moved(self, d):
        self.v = self.v + np.asarray(d, float)
        return self

    def recolored(self, color):
        col = np.asarray(color, float)
        if col.ndim == 1:
            col = np.tile(col, (len(self.f), 1))
        self.c = col
        return self


def pad_faces(faces):
    """Pad polygon index lists to a rectangular (M, maxv) array with -1."""
    faces = [list(map(int, f)) for f in faces]
    m = len(faces)
    maxv = max(3, max((len(f) for f in faces), default=3))
    out = -np.ones((m, maxv), dtype=np.int32)
    for i, f in enumerate(faces):
        out[i, : len(f)] = f
    return out


# ----------------------------------------------------------------------------- primitives
def box(center, size, color, rot=None, glow=0.0, alpha=1.0, taper=1.0):
    """Axis boxes with optional yaw rotation and top taper."""
    cx, cy, cz = center
    sx, sy, sz = np.asarray(size, float) / 2.0
    v = np.array(
        [
            [-sx, -sy, -sz], [sx, -sy, -sz], [sx, -sy, sz], [-sx, -sy, sz],
            [-sx * taper, sy, -sz * taper], [sx * taper, sy, -sz * taper],
            [sx * taper, sy, sz * taper], [-sx * taper, sy, sz * taper],
        ],
        dtype=np.float64,
    )
    if rot is not None:
        v = v @ np.asarray(rot, float).T
    v += np.array([cx, cy, cz], float)
    f = [
        [0, 1, 2, 3],  # bottom (faces -y)
        [7, 6, 5, 4],  # top
        [0, 4, 5, 1],  # -z
        [3, 2, 6, 7],  # +z
        [0, 3, 7, 4],  # -x
        [1, 5, 6, 2],  # +x
    ]
    return Mesh(v, pad_faces(f), color, glow, alpha)


def prism(center, radius, height, sides, color, rot=None, taper=1.0, cap=True, glow=0.0, alpha=1.0):
    c = np.asarray(center, float)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    base = np.stack([np.cos(ang) * radius, np.zeros(sides), np.sin(ang) * radius], 1)
    top = base * np.array([taper, 0, taper]) + np.array([0, height / 2, 0])
    base = base - np.array([0, height / 2, 0])
    if rot is not None:
        base = base @ np.asarray(rot, float).T
        top = top @ np.asarray(rot, float).T
    v = np.vstack([base, top]) + c
    f = []
    for i in range(sides):
        j = (i + 1) % sides
        f.append([i, j, sides + j, sides + i])
    if cap:
        f.append(list(range(sides))[::-1])
        f.append(list(range(sides, 2 * sides)))
    return Mesh(v, pad_faces(f), color, glow, alpha)


def pyramid(center, size, height, color, rot=None, overhang=1.0, glow=0.0, alpha=1.0):
    cx, cy, cz = center
    sx, sz = np.asarray(size, float) / 2.0 * overhang
    v = np.array(
        [[-sx, 0, -sz], [sx, 0, -sz], [sx, 0, sz], [-sx, 0, sz], [0, height, 0]],
        dtype=np.float64,
    )
    if rot is not None:
        v = v @ np.asarray(rot, float).T
    v += np.array([cx, cy, cz], float)
    f = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [3, 2, 1, 0]]
    return Mesh(v, pad_faces(f), color, glow, alpha)


def icosphere(center, radius, color, subdiv=0, squash=1.0, glow=0.0, alpha=1.0):
    t = (1.0 + 5 ** 0.5) / 2.0
    v = np.array(
        [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
         [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
         [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=np.float64)
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
         [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
         [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
         [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    for _ in range(subdiv):
        cache = {}
        nf = []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k in cache:
                return cache[k]
            m = normalize(v[a] + v[b]) * radius
            v_list.append(m)
            idx = len(v_list) - 1
            cache[k] = idx
            return idx

        v_list = list(v)
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        v = np.array(v_list)
        f = nf
    v = normalize(v) * radius * np.array([1.0, squash, 1.0]) + np.asarray(center, float)
    return Mesh(v, pad_faces(f), color, glow, alpha)


def ribbon(path, profile, color, closed=False, cap=True):
    """
    Extrude a 2D cross-section profile along a 3D path.
    profile: list of (side, height) offsets in the local frame (side = horizontal normal).
    """
    path = np.asarray(path, float)
    n = len(path)
    tang = np.zeros_like(path)
    tang[1:-1] = path[2:] - path[:-2]
    tang[0] = path[1] - path[0]
    tang[-1] = path[-1] - path[-2]
    tang[:, 1] = 0.0
    tang = normalize(tang)
    up = np.array([0.0, 1.0, 0.0])
    side = normalize(np.cross(tang, up))
    rings = []
    for i in range(n):
        ring = [
            path[i] + side[i] * s + up * h for s, h in profile
        ]
        rings.append(ring)
    nprof = len(profile)
    verts = np.array(rings)  # (n, nprof, 3)
    flat = verts.reshape(-1, 3)
    faces = []
    for i in range(n - 1):
        for k in range(nprof if closed else nprof - 1):
            k2 = (k + 1) % nprof
            a = i * nprof + k
            b = i * nprof + k2
            c = (i + 1) * nprof + k2
            d = (i + 1) * nprof + k
            faces.append([a, b, c, d])
    if cap:
        faces.append(list(range(nprof))[::-1])
        base = (n - 1) * nprof
        faces.append(list(range(base, base + nprof)))
    return Mesh(flat, pad_faces(faces), color)


# ----------------------------------------------------------------------------- camera
class Camera:
    def __init__(self, eye, target, up=(0, 1, 0), fov=52, w=1080, h=1920, near=0.5, roll=0.0):
        eye = np.asarray(eye, float)
        target = np.asarray(target, float)
        f = normalize(target - eye)
        up = np.asarray(up, float)
        r = normalize(np.cross(f, up))
        u = np.cross(r, f)
        if abs(roll) > 1e-6:
            cu, su = np.cos(roll), np.sin(roll)
            r, u = r * cu + u * su, -r * su + u * cu
        self.R = np.stack([r, u, f])
        self.eye = eye
        self.w, self.h = w, h
        self.near = near
        self.focal = (h / 2.0) / np.tan(fov * D2R / 2.0)

    def to_cam(self, P):
        return (np.asarray(P, float) - self.eye) @ self.R.T


# ----------------------------------------------------------------------------- lighting
class Sun:
    def __init__(self, direction, color=(1.0, 0.96, 0.88), strength=1.0):
        self.d = normalize(np.asarray(direction, float))  # towards the sun
        self.c = np.asarray(color, float)
        self.s = strength


class Lighting:
    def __init__(self, sun, ambient=(0.42, 0.46, 0.55), sky=(0.55, 0.68, 0.85),
                 ground=(0.30, 0.28, 0.24), fog=(0.78, 0.84, 0.92), fog_density=0.0055,
                 fog_height=0.02, exposure=1.0, wrap=0.35, ambient_boost=1.0):
        self.sun = sun
        self.amb = np.asarray(ambient, float) * ambient_boost
        self.sky = np.asarray(sky, float)
        self.ground = np.asarray(ground, float)
        self.fog = np.asarray(fog, float)
        self.fog_density = fog_density
        self.fog_height = fog_height
        self.exposure = exposure
        self.wrap = wrap


# ----------------------------------------------------------------------------- renderer
class Renderer:
    def __init__(self, w=1080, h=1920, ssaa=1.0):
        self.W, self.H = w, h
        self.w, self.h = int(w * ssaa), int(h * ssaa)
        self.ssaa = ssaa

    # -- sky ------------------------------------------------------------------
    def sky_image(self, cam, light, top=(0.24, 0.46, 0.78), bottom=(0.86, 0.90, 0.95),
                  sun_glow=0.55, sun_size=0.06, stars=0.0, seed=0):
        w, h = self.w, self.h
        y = np.linspace(0, 1, h)[:, None]
        grad = (1 - y) ** 1.35
        top = np.asarray(top, float)
        bottom = np.asarray(bottom, float)
        img = bottom[None, None, :] * (1 - grad)[:, :, None] + top[None, None, :] * grad[:, :, None]
        img = np.repeat(img, w, axis=1)
        # sun glow
        sd = light.sun.d
        p = normalize(np.array([sd[0], sd[1], sd[2]]))
        # project sun direction into the camera to place the glow
        pc = cam.R @ p
        if pc[2] > 0.05:
            sx = int(w / 2 + cam.focal * (pc[0] / pc[2]) * self.ssaa)
            sy = int(h / 2 - cam.focal * (pc[1] / pc[2]) * self.ssaa)
            if -w < sx < 2 * w and -h < sy < 2 * h:
                yy, xx = np.mgrid[0:h, 0:w]
                r = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2) / (h * sun_size)
                glow = np.clip(np.exp(-r ** 1.7) * sun_glow, 0, 4)
                core = np.clip(np.exp(-(r * 3.2) ** 2) * 1.8, 0, 4)
                img = img + (glow + core)[:, :, None] * np.asarray(light.sun.c, float)[None, None, :] * 1.4
        if stars > 0:
            rng = np.random.default_rng(seed + 7)
            n = int(w * h / 5200)
            ys = rng.integers(0, int(h * 0.62), n)
            xs = rng.integers(0, w, n)
            br = rng.uniform(0.25, 1.0, n) * stars
            for x, y_, b in zip(xs, ys, br):
                img[y_, x] = img[y_, x] + b
                if x + 1 < w:
                    img[y_, x + 1] = img[y_, x + 1] + b * 0.35
        return Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8), "RGB")

    # -- geometry -------------------------------------------------------------
    def render(self, meshes, cam, light, background=None, max_faces=None):
        w, h = self.w, self.h
        meshes = [m for m in meshes if m is not None and len(m.f)]
        if not meshes:
            return background or Image.new("RGB", (w, h), (0, 0, 0))

        maxv = max(m.f.shape[1] for m in meshes)
        Fs = []
        off = 0
        for m in meshes:
            f = m.f if m.f.shape[1] == maxv else np.pad(
                m.f, ((0, 0), (0, maxv - m.f.shape[1])), constant_values=-1)
            if off:
                f = np.where(f >= 0, f + off, -1)
            Fs.append(f)
            off += len(m.v)
        V = np.concatenate([m.v for m in meshes])
        F = np.concatenate(Fs)
        C = np.concatenate([m.c for m in meshes])
        G = np.concatenate([m.glow for m in meshes])
        A = np.concatenate([m.alpha for m in meshes])

        valid = (F >= 0).sum(1) >= 3
        ok3 = (F[:, :3] >= 0).all(1)

        Pc = cam.to_cam(V)                       # (N,3) camera space
        Fi = np.clip(F, 0, len(V) - 1)
        Q = Pc[Fi]                              # (M, maxv, 3)

        a, b, c = Q[:, 0], Q[:, 1], Q[:, 2]
        nrm = normalize(np.cross(b - a, c - a))

        # visibility: normal facing the camera
        facing = np.einsum("ij,ij->i", nrm, a - cam.eye) < 0

        # depth (camera space z)
        zs = Q[:, :, 2]
        zmin = np.where(F >= 0, zs, np.inf).min(1)
        zmax = np.where(F >= 0, zs, -np.inf).max(1)
        depth = np.where(F >= 0, zs, 0).sum(1) / np.maximum((F >= 0).sum(1), 1)

        keep = valid & ok3 & facing & (zmax > cam.near) & (zmin < 1.0e4)
        idx = np.nonzero(keep)[0]
        if max_faces and len(idx) > max_faces:
            idx = idx[np.argsort(depth[idx])][:max_faces]

        # shading -------------------------------------------------------------
        nw = nrm
        ndl = nw @ light.sun.d
        diff = np.clip((ndl + light.wrap) / (1 + light.wrap), 0, 1)
        hemi = 0.5 + 0.5 * nw[:, 1]
        amb = light.amb[None, :] * (0.45 + 0.55 * hemi)[:, None]
        amb = amb + light.sky[None, :] * (0.18 * np.clip(hemi, 0, 1))[:, None]
        amb = amb + light.ground[None, :] * (0.22 * np.clip(1 - hemi, 0, 1))[:, None]
        shade = C * (amb + (light.sun.c * light.sun.s)[None, :] * diff[:, None])
        shade = 1.0 - np.exp(-shade * light.exposure * 1.35)   # filmic-ish tonemap
        glow = G[:, None]
        shade = shade * (1 - glow) + C * glow

        # fog by distance + altitude
        dist = np.linalg.norm(Q[:, 0] - cam.eye, axis=1)
        alt = np.clip(Q[:, 0, 1] - 2.0, 0, None)
        dens = light.fog_density * np.exp(-alt * light.fog_height)
        ft = 1 - np.exp(-(dist * dens) ** 1.15)
        ft = np.clip(ft, 0, 0.965)[:, None]
        shade = shade * (1 - ft) + light.fog[None, :] * ft

        # project -------------------------------------------------------------
        z = np.where(zs > cam.near, zs, cam.near)
        sx = w / 2 + cam.focal * (Q[:, :, 0] / z) * self.ssaa
        sy = h / 2 - cam.focal * (Q[:, :, 1] / z) * self.ssaa

        # clip faces crossing the near plane (python fallback, rare)
        near_faces = idx[zmin[idx] <= cam.near]
        polys = {}
        if len(near_faces):
            for fi in near_faces:
                pv = [Q[fi, k] for k in range(F.shape[1]) if F[fi, k] >= 0]
                poly = _clip_near(pv, cam.near)
                if len(poly) >= 3:
                    pts = []
                    for p in poly:
                        zz = max(p[2], cam.near)
                        pts.append((w / 2 + cam.focal * p[0] / zz * self.ssaa,
                                    h / 2 - cam.focal * p[1] / zz * self.ssaa))
                    polys[fi] = pts
                else:
                    polys[fi] = None
            idx = idx[[polys.get(fi, 1) is not None for fi in idx]]

        order = idx[np.argsort(-depth[idx])]
        cols = np.clip(shade * 255, 0, 255).astype(np.uint8)
        alpha = A

        base = background if background is not None else Image.new("RGB", (w, h), (12, 14, 18))
        draw = ImageDraw.Draw(base)
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        has_alpha = bool((alpha < 0.999).any())

        for fi in order:
            if fi in polys:
                pts = polys[fi]
            else:
                k = int((F[fi] >= 0).sum())
                pts = [(sx[fi, j], sy[fi, j]) for j in range(k)]
            col = cols[fi]
            if has_alpha and alpha[fi] < 0.999:
                odraw.polygon(pts, fill=(int(col[0]), int(col[1]), int(col[2]),
                                         int(255 * alpha[fi])))
            else:
                draw.polygon(pts, fill=(int(col[0]), int(col[1]), int(col[2])))
        if has_alpha:
            base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
        return base


def _clip_near(poly, near):
    """Sutherland-Hodgman clip of a camera-space polygon against z = near."""
    out = []
    n = len(poly)
    for i in range(n):
        A = poly[i]
        B = poly[(i + 1) % n]
        ina = A[2] >= near
        inb = B[2] >= near
        if ina:
            out.append(A)
        if ina != inb:
            t = (near - A[2]) / (B[2] - A[2])
            out.append(A + (B - A) * t)
    return out


# ----------------------------------------------------------------------------- post
_VIG_CACHE = {}
_GRAIN_BANK = {}


def _vignette_mask(h, w):
    key = (h, w)
    m = _VIG_CACHE.get(key)
    if m is None:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cy, cx = h / 2.0, w / 2.0
        r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2) / 1.4142
        m = np.clip((r - 0.35) / 0.65, 0, 1) ** 1.6
        _VIG_CACHE[key] = m.astype(np.float32)
    return m


def _grain(h, w, n=8):
    key = (h, w)
    b = _GRAIN_BANK.get(key)
    if b is None:
        rng = np.random.default_rng(7)
        b = [(rng.random((h, w), dtype=np.float32) - 0.5) * 2.0 for _ in range(n)]
        _GRAIN_BANK[key] = b
    return b


def post(img, bloom=0.35, vignette=0.35, grain=3.0, saturation=1.06, contrast=1.05,
         lift=(0.0, 0.0, 0.0), gain=(1.0, 1.0, 1.0), seed=0, blur_bg=0.0):
    w, h = img.size
    a = np.asarray(img, dtype=np.float32)
    a *= (1.0 / 255.0)

    if bloom > 0:
        sw, sh = max(8, w // 8), max(8, h // 8)
        small = img.resize((sw, sh), Image.BILINEAR)
        arr = np.asarray(small, dtype=np.float32) * (1.0 / 255.0)
        bright = np.clip((arr - 0.72) * 3.57, 0, 1)
        bimg = Image.fromarray((bright * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=3.2)).resize((w, h), Image.BILINEAR)
        b = np.asarray(bimg, dtype=np.float32) * (1.0 / 255.0)
        a += b * bloom

    if contrast != 1.0:
        a -= 0.5
        a *= contrast
        a += 0.5
    if saturation != 1.0:
        lum = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        a += (a - lum[..., None]) * (saturation - 1.0)
    if gain != (1.0, 1.0, 1.0):
        a *= np.asarray(gain, np.float32)[None, None, :]
    if lift != (0.0, 0.0, 0.0):
        a += np.asarray(lift, np.float32)[None, None, :]

    if vignette > 0:
        a *= (1.0 - vignette * _vignette_mask(h, w))[:, :, None]

    if grain > 0:
        bank = _grain(h, w)
        a += bank[seed % len(bank)][:, :, None] * (grain * (1.0 / 255.0))

    np.clip(a, 0, 1, out=a)
    a *= 255.0
    return Image.fromarray(a.astype(np.uint8), "RGB")
