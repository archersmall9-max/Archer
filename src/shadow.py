"""
shadow.py — baked sun shadows for the static world.

The sun never moves inside a shot, so shadowing is solved once per shot and stored as a
per-face sun-visibility value. The renderer multiplies only the *direct* sun term by it,
which means shadowed faces keep their blue sky light instead of just going darker — the
single cheapest trick that makes a flat-shaded scene read as lit by a real sun.

Method: orthographic depth map along the sun direction (a classic shadow map, but
rasterised with numpy: every face stamps its bounding box into a min-depth buffer,
then each face is tested against the map with a bias and a one-pixel penumbra).
"""
from __future__ import annotations

import numpy as np
import cv2

from engine3d import normalize


def _dominant_axis(d):
    up = np.array([0.0, 1.0, 0.0])
    if abs(float(d @ up)) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    u = normalize(np.cross(up, d))
    return u, np.cross(d, u)


def bake(meshes, sun_dir, res=448, strength=0.95, penumbra=1, bias_scale=1.0,
         splat=0, max_faces=60000):
    """Write per-face sun visibility (0..1) into each mesh's `sunvis`."""
    meshes = [m for m in meshes if m is not None and len(m.f)]
    if not meshes:
        return meshes
    d = normalize(np.asarray(sun_dir, float))
    u, v = _dominant_axis(d)

    V = np.concatenate([m.v for m in meshes])
    F, off = [], 0
    for m in meshes:                                  # triangle list per mesh
        k = min(3, m.f.shape[1])
        F.append(m.f[:, :k] + off)
        off += len(m.v)
    nper = [len(f) for f in F]
    tri = V[np.concatenate(F)]                        # (N,3,3)

    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    nrm = np.cross(b - a, c - a)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    ndl = np.abs(nrm @ d)

    cen = tri.mean(1)
    pu, pv, s = cen @ u, cen @ v, cen @ d
    rad = np.linalg.norm(tri - cen[:, None, :], axis=2).max(1)
    rad = np.maximum(rad, 0.3)

    lo_u, lo_v = pu.min() - 8, pv.min() - 8
    span = max(pu.max() - lo_u, pv.max() - lo_v) + 8
    cell = span / res
    iu = np.clip(((pu - lo_u) / cell).astype(np.int32), 0, res - 1)
    iv = np.clip(((pv - lo_v) / cell).astype(np.int32), 0, res - 1)

    depth = np.full(res * res, -np.inf, np.float32)
    if splat <= 0:
        np.maximum.at(depth, iv * res + iu, s.astype(np.float32))
    else:
        rpx = np.clip(np.round(rad / cell).astype(np.int32), 1, splat)
        for dx in range(-splat, splat + 1):
            for dy in range(-splat, splat + 1):
                sel = (np.abs(dx) <= rpx) & (np.abs(dy) <= rpx)
                if not sel.any():
                    continue
                x = np.clip(iu[sel] + dx, 0, res - 1)
                y = np.clip(iv[sel] + dy, 0, res - 1)
                np.maximum.at(depth, y * res + x, s[sel].astype(np.float32))
    depth = np.where(np.isfinite(depth), depth, -1e6).astype(np.float32).reshape(res, res)
    depth = cv2.dilate(depth, np.ones((3, 3), np.uint8))       # fill + soft edge
    for _ in range(int(penumbra)):
        depth = cv2.dilate(depth, np.ones((3, 3), np.uint8))

    # slope-scaled bias: grazing faces need more slack or they shadow themselves
    slope = np.sqrt(np.maximum(1.0 - ndl ** 2, 0.0)) / np.maximum(ndl, 0.18)
    bias = bias_scale * cell * (1.7 + 1.6 * np.minimum(slope, 5.0))

    # 9-tap percentage-closer style average: soft, grainy-free penumbra instead of a
    # hard polygonal cut, which matters a lot at this face density
    acc = np.zeros(len(s), np.float32)
    taps = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    for dx, dy in taps:
        x = np.clip(iu + dx, 0, res - 1)
        y = np.clip(iv + dy, 0, res - 1)
        occ = depth[y, x]
        acc += np.where(np.isfinite(occ) & (occ > s + bias), 0.0, 1.0).astype(np.float32)
    vis = acc / len(taps)
    vis = 1.0 - (1.0 - vis) * strength

    start = 0
    for m, n in zip(meshes, nper):
        m.sunvis = vis[start:start + n].copy()
        start += n
    return meshes
