#!/usr/bin/env python3
"""Generate the scene's textures and object meshes.

    python tools/make_assets.py            # writes into comp_vision_sim/assets/

Everything here is procedural: no binary assets are committed that cannot be
regenerated from this file, and no external asset library is needed. Textures
are numpy arrays saved as PNG; meshes are written as binary STL, which MuJoCo
reads directly.

The meshes are for LOOKS ONLY. They are attached with contype=0 conaffinity=0
and the bodies carry explicit <inertial>, so collision geometry, mass and
inertia are exactly what they were before -- the grasp planner still closes on
the primitive, and the pick/chore benchmarks are unaffected. A detailed mesh
would collide as its convex hull anyway, which for a key is a wedge, so using
them for contact would make grasping worse, not better.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent.parent / "assets"


# ------------------------------------------------------------------ meshes
def write_stl(path: Path, verts: np.ndarray, faces: np.ndarray):
    """Binary STL. verts (N,3) float, faces (M,3) int."""
    tris = verts[faces]                                    # (M, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, np.where(ln == 0, 1, ln))
    with path.open("wb") as f:
        f.write(b"bracketbot procedural mesh".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(faces)))
        for i in range(len(faces)):
            f.write(struct.pack("<3f", *n[i]))
            for v in tris[i]:
                f.write(struct.pack("<3f", *v))
            f.write(b"\0\0")


def extrude(profile: np.ndarray, thickness: float) -> tuple[np.ndarray, np.ndarray]:
    """A closed 2-D polygon (N,2) -> a flat solid of the given thickness."""
    n = len(profile)
    h = thickness / 2
    verts = np.vstack([np.column_stack([profile, np.full(n, -h)]),
                       np.column_stack([profile, np.full(n, +h)])])
    faces = []
    for i in range(n):                                     # side wall
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i]]
    for i in range(1, n - 1):                              # both caps, fanned
        faces += [[0, i + 1, i], [n, n + i, n + i + 1]]
    return verts, np.array(faces, np.int32)


def revolve(outline: np.ndarray, segments: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """A (radius, z) outline spun about z. Outline runs bottom to top."""
    k = len(outline)
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    verts = [np.array([0.0, 0.0, outline[0, 1]])]
    for r, z in outline:
        for a in ang:
            verts.append([r * np.cos(a), r * np.sin(a), z])
    verts.append(np.array([0.0, 0.0, outline[-1, 1]]))
    verts = np.array(verts, float)
    faces = []
    ring = lambda i, j: 1 + i * segments + (j % segments)
    for j in range(segments):                              # bottom cap
        faces.append([0, ring(0, j + 1), ring(0, j)])
    for i in range(k - 1):
        for j in range(segments):
            a, b = ring(i, j), ring(i, j + 1)
            c, d = ring(i + 1, j), ring(i + 1, j + 1)
            faces += [[a, b, d], [a, d, c]]
    top = len(verts) - 1
    for j in range(segments):                              # top cap
        faces.append([top, ring(k - 1, j), ring(k - 1, j + 1)])
    return verts, np.array(faces, np.int32)


def key_profile() -> np.ndarray:
    """A house key seen flat on: round bow, shoulder, blade with cut teeth."""
    pts = []
    for a in np.linspace(np.pi * 0.72, -np.pi * 0.72, 26):  # the bow, open at +x
        pts.append([0.0145 * np.cos(a) - 0.021, 0.0145 * np.sin(a)])
    pts += [[-0.0115, 0.0042], [-0.004, 0.0050], [-0.002, 0.0062],
            [0.0015, 0.0062], [0.0030, 0.0040]]             # shoulder
    teeth = [(0.0060, 0.0020), (0.0085, 0.0036), (0.0112, 0.0016),
             (0.0140, 0.0034), (0.0170, 0.0014), (0.0198, 0.0032),
             (0.0228, 0.0018), (0.0258, 0.0030)]            # the cut edge
    for x, y in teeth:
        pts.append([x, y])
    pts += [[0.0288, 0.0026], [0.0300, 0.0004],             # tip
            [0.0288, -0.0030], [0.0030, -0.0030],           # flat spine back
            [0.0015, -0.0062], [-0.002, -0.0062],
            [-0.004, -0.0050], [-0.0115, -0.0042]]
    return np.array(pts, float)


def ring_mesh(radius=0.0145, tube=0.0018, major=40, minor=12):
    u = np.linspace(0, 2 * np.pi, major, endpoint=False)
    v = np.linspace(0, 2 * np.pi, minor, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")
    x = (radius + tube * np.cos(V)) * np.cos(U)
    y = (radius + tube * np.cos(V)) * np.sin(U)
    z = tube * np.sin(V)
    verts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    faces = []
    idx = lambda i, j: (i % major) * minor + (j % minor)
    for i in range(major):
        for j in range(minor):
            faces += [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)],
                      [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]]
    return verts, np.array(faces, np.int32)


def mug_mesh():
    """Tapered body, hollow rim, and a handle swept round an arc."""
    outline = np.array([[0.000, 0.000], [0.036, 0.000], [0.037, 0.004],
                        [0.039, 0.030], [0.041, 0.070], [0.042, 0.096],
                        [0.042, 0.100], [0.038, 0.100], [0.037, 0.096],
                        [0.035, 0.030], [0.033, 0.006], [0.000, 0.008]])
    v, f = revolve(outline, 40)
    hv, hf = [], []
    arc = np.linspace(-1.15, 1.15, 22)
    for t in arc:
        cx, cz = 0.040 + 0.020 * np.cos(t), 0.052 + 0.031 * np.sin(t)
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            hv.append([cx + 0.005 * np.cos(a) * np.cos(t),
                       0.005 * np.sin(a),
                       cz + 0.005 * np.cos(a) * np.sin(t)])
    hv = np.array(hv)
    for i in range(len(arc) - 1):
        for j in range(8):
            a, b = i * 8 + j, i * 8 + (j + 1) % 8
            c, d = (i + 1) * 8 + j, (i + 1) * 8 + (j + 1) % 8
            hf += [[a, b, d], [a, d, c]]
    faces = np.vstack([f, np.array(hf, np.int32) + len(v)])
    return np.vstack([v, hv]), faces


def bottle_mesh():
    outline = np.array([[0.000, 0.000], [0.034, 0.000], [0.036, 0.008],
                        [0.036, 0.140], [0.033, 0.158], [0.024, 0.176],
                        [0.017, 0.192], [0.016, 0.214], [0.019, 0.216],
                        [0.019, 0.228], [0.000, 0.229]])
    return revolve(outline, 40)


def can_mesh():
    outline = np.array([[0.000, 0.000], [0.028, 0.002], [0.032, 0.008],
                        [0.033, 0.014], [0.033, 0.104], [0.032, 0.112],
                        [0.028, 0.118], [0.026, 0.120], [0.000, 0.119]])
    return revolve(outline, 36)


def uv_sphere(radius=1.0, seg=24, rings=16):
    lat = np.linspace(0, np.pi, rings + 1)[1:-1]
    verts = [[0, 0, radius]]
    for t in lat:
        for a in np.linspace(0, 2 * np.pi, seg, endpoint=False):
            verts.append([radius * np.sin(t) * np.cos(a),
                          radius * np.sin(t) * np.sin(a),
                          radius * np.cos(t)])
    verts.append([0, 0, -radius])
    verts = np.array(verts, float)
    faces = []
    idx = lambda i, j: 1 + i * seg + (j % seg)
    for j in range(seg):
        faces.append([0, idx(0, j), idx(0, j + 1)])
    for i in range(len(lat) - 1):
        for j in range(seg):
            a, b = idx(i, j), idx(i, j + 1)
            c, d = idx(i + 1, j), idx(i + 1, j + 1)
            faces += [[a, c, d], [a, d, b]]
    last = len(verts) - 1
    for j in range(seg):
        faces.append([last, idx(len(lat) - 1, j + 1), idx(len(lat) - 1, j)])
    return verts, np.array(faces, np.int32)


def foliage_mesh(seed=4):
    """A canopy of overlapping lobes, each one lumpy.

    A single smooth sphere is the giveaway that a plant is a placeholder: real
    foliage has a broken silhouette. Displacing the radius by a few low
    harmonics gives that outline, and clustering several lobes at different
    scales stops it reading as one ball.
    """
    rng = np.random.default_rng(seed)
    lobes = [((0.00, 0.00, 0.00), 0.30), ((0.17, 0.06, -0.06), 0.20),
             ((-0.15, 0.10, -0.04), 0.19), ((0.04, -0.17, -0.03), 0.18),
             ((-0.05, -0.06, 0.16), 0.17), ((0.11, 0.13, 0.10), 0.145)]
    V, F = [], []
    for (cx, cy, cz), r in lobes:
        v, f = uv_sphere(r, 20, 13)
        d = v / np.linalg.norm(v, axis=1, keepdims=True)
        bump = np.ones(len(v))
        for _ in range(4):                       # a few random harmonics
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            k = rng.uniform(2.5, 6.0)
            bump += 0.13 * np.sin(k * (d @ axis) + rng.uniform(0, 6.28))
        v = d * (r * bump)[:, None] + np.array([cx, cy, cz])
        F.append(f + sum(len(x) for x in V))
        V.append(v)
    return np.vstack(V), np.vstack(F)


def pot_mesh():
    """Tapered pot with a rolled rim -- the shape that says 'terracotta'."""
    outline = np.array([[0.000, 0.000], [0.115, 0.000], [0.120, 0.010],
                        [0.148, 0.240], [0.152, 0.276], [0.168, 0.286],
                        [0.170, 0.300], [0.150, 0.302], [0.140, 0.292],
                        [0.136, 0.284], [0.000, 0.280]])
    return revolve(outline, 40)


# ---------------------------------------------------------------- textures
def _noise(shape, octaves=4, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros(shape)
    amp = 1.0
    for o in range(octaves):
        step = max(1, 2 ** (octaves - o - 1))
        small = rng.random((shape[0] // step + 2, shape[1] // step + 2))
        big = np.kron(small, np.ones((step, step)))[:shape[0], :shape[1]]
        out += amp * big
        amp *= 0.5
    out -= out.min()
    return out / max(out.max(), 1e-9)


def oak_floor(size=1024, planks=7):
    """Oak boards: per-board tone, long grain, and dark seams across the joins."""
    y, x = np.mgrid[0:size, 0:size]
    img = np.zeros((size, size, 3))
    bw = size // planks
    rng = np.random.default_rng(7)
    grain = _noise((size, size), 5, seed=3)
    fine = _noise((size, max(size // 64, 4)), 3, seed=11)
    fine = np.repeat(fine, 64, axis=1)[:, :size]
    for p in range(planks + 1):
        lo, hi = p * bw, min((p + 1) * bw, size)
        if lo >= size:
            break
        tone = 0.82 + 0.30 * rng.random()
        shift = rng.integers(0, size)
        band = np.roll(grain[lo:hi], shift, axis=1) * 0.30 + \
            np.roll(fine[lo:hi], shift, axis=1) * 0.22
        base = np.array([0.50, 0.35, 0.21]) * tone
        img[lo:hi] = base * (0.80 + band[..., None])
        img[lo:min(lo + 2, size)] *= 0.45                   # seam between boards
        # butt joints along the board
        for j in rng.integers(0, size, 2):
            img[lo:hi, j:j + 2] *= 0.5
    return np.clip(img, 0, 1)


def plaster(size=512, tint=(0.86, 0.84, 0.80)):
    n = _noise((size, size), 5, seed=21) * 0.06 + 0.97
    return np.clip(np.array(tint) * n[..., None], 0, 1)


def woven(size=512, tint=(0.34, 0.39, 0.47), pitch=6):
    y, x = np.mgrid[0:size, 0:size]
    weave = ((x // pitch + y // pitch) % 2) * 0.10
    weave += np.sin(x / pitch * np.pi) * 0.04 + np.sin(y / pitch * np.pi) * 0.04
    n = _noise((size, size), 4, seed=33) * 0.10
    return np.clip(np.array(tint) * (0.92 + weave + n)[..., None], 0, 1)


def walnut(size=512, tint=(0.31, 0.19, 0.11)):
    y, x = np.mgrid[0:size, 0:size]
    rings = np.sin(x / 22.0 + _noise((size, size), 4, seed=5) * 5.0) * 0.5 + 0.5
    n = _noise((size, size), 5, seed=9)
    shade = 0.78 + 0.30 * rings * 0.5 + 0.22 * n
    return np.clip(np.array(tint) * shade[..., None] * 1.35, 0, 1)


def brushed(size=256, tint=(0.62, 0.63, 0.66)):
    streak = _noise((size, max(size // 128, 4)), 3, seed=17)
    streak = np.repeat(streak, 128, axis=1)[:, :size]
    return np.clip(np.array(tint) * (0.88 + 0.24 * streak)[..., None], 0, 1)


def save_png(path: Path, arr: np.ndarray):
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meshes = {
        "key_blade": extrude(key_profile(), 0.0038),
        "key_ring": ring_mesh(),
        "mug": mug_mesh(),
        "bottle": bottle_mesh(),
        "can": can_mesh(),
        "foliage": foliage_mesh(),
        "pot": pot_mesh(),
    }
    for name, (v, f) in meshes.items():
        write_stl(OUT / f"{name}.stl", v, f)
        print(f"  {name}.stl  {len(v)} verts  {len(f)} tris")

    textures = {
        "floor_oak": oak_floor(), "wall_plaster": plaster(),
        "sofa_weave": woven(), "wood_walnut": walnut(),
        "wood_oak": walnut(tint=(0.45, 0.30, 0.17)),
        "metal_brushed": brushed(),
    }
    for name, arr in textures.items():
        save_png(OUT / f"{name}.png", arr)
        print(f"  {name}.png  {arr.shape[1]}x{arr.shape[0]}")


if __name__ == "__main__":
    main()
