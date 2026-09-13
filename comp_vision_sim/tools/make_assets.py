#!/usr/bin/env python3
"""Generate the scenes' textures and object meshes.

    python tools/make_assets.py            # writes into comp_vision_sim/assets/

Everything here is procedural: no binary asset is committed that cannot be
regenerated from this file, and no external asset library is needed. Textures
are numpy arrays / PIL drawings saved as PNG.

Household items and the person
------------------------------
The seven pickable items and the person are modelled on the real thing, so an
off-the-shelf detector recognises them by what they are, not by a colour it
was told to look for:

    mug     glazed ceramic, hollow, with a looped handle and an unglazed foot
    can     printed soda can: domed base, necked top, lid and ring pull
    bottle  ribbed water bottle with a wrap-round label and a ridged cap
    remote  TV remote: numbered keypad, power key, d-pad, rockers
    keys    a bunch: car-key fob with its blade, a brass and a nickel key, ring
    ball    tennis ball: optic-yellow felt and the curved seam
    box     taped kraft carton with a shipping label
    person  clothed, with a face, one hand held out palm up

Each is a textured OBJ (texture coordinates and normals) plus one PNG atlas,
declared once in assets/household.xml for every scene to include.

They are VISUAL ONLY. The scenes attach them as class "look" geoms
(contype=0 conaffinity=0 mass=0) over the collision primitives, which are
unchanged apart from moving to group 3, which is not rendered. So mass,
inertia, contact, and the geoms the grasp planner closes on are exactly what
they were -- the pick/chore baselines are unaffected. Every visual also stays
inside the bounding box of the primitives it dresses: the depth camera
measures the visual, so a bigger visual would be a bigger object to perception,
and furniture footprints are read off every geom on a body.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "assets"


# ------------------------------------------------------------ STL meshes
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


# ------------------------------------------------------ textured meshes (OBJ)
@dataclass
class Mesh:
    v: np.ndarray       # (N, 3) positions
    uv: np.ndarray      # (N, 2) texture coordinates, OBJ convention (v = 1 is the image top)
    f: np.ndarray       # (M, 3) triangles, counter-clockwise seen from outside
    n: np.ndarray       # (N, 3) unit normals

    def moved(self, R=None, t=(0.0, 0.0, 0.0)) -> Mesh:
        R = np.eye(3) if R is None else np.asarray(R, float)
        return Mesh(self.v @ R.T + np.asarray(t, float), self.uv, self.f, self.n @ R.T)


def merge(*parts: Mesh) -> Mesh:
    off, V, UV, F, N = 0, [], [], [], []
    for p in parts:
        V.append(p.v)
        UV.append(p.uv)
        N.append(p.n)
        F.append(p.f + off)
        off += len(p.v)
    return Mesh(np.vstack(V), np.vstack(UV), np.vstack(F).astype(np.int32), np.vstack(N))


def write_obj(path: Path, m: Mesh):
    out = ["# procedural mesh, generated by comp_vision_sim/tools/make_assets.py"]
    out += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in m.v]
    out += [f"vt {u:.5f} {v:.5f}" for u, v in m.uv]
    out += [f"vn {x:.4f} {y:.4f} {z:.4f}" for x, y, z in m.n]
    out += ["f " + " ".join(f"{i}/{i}/{i}" for i in tri + 1) for tri in m.f]
    path.write_text("\n".join(out) + "\n", encoding="ascii")


def _unit(a):
    ln = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.where(ln < 1e-12, 1.0, ln)


def _grid_faces(rows, cols):
    """Two triangles per cell of a row-major rows x cols vertex grid."""
    r, c = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    a = (r * cols + c).ravel()
    b, cc = a + 1, a + cols
    d = cc + 1
    return np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, cc], 1)]).astype(np.int32)


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Atlas:
    """One texture sheet, and the pixel boxes of it each part maps onto."""

    def __init__(self, w, h, base=(0.5, 0.5, 0.5)):
        self.w, self.h = w, h
        self.img = Image.new("RGB", (w, h), c8(base))
        self.draw = ImageDraw.Draw(self.img)

    def uv(self, box, s, t):
        """Texture coords of the point s across, t down (both 0..1) pixel box (x0, y0, x1, y1)."""
        x0, y0, x1, y1 = box
        x = x0 + (x1 - x0) * np.asarray(s, float)
        y = y0 + (y1 - y0) * np.asarray(t, float)
        return np.stack([x / self.w, 1.0 - y / self.h], -1)

    def paste(self, box, arr):
        x0, y0, x1, y1 = box
        im = Image.fromarray(to8(arr))
        if im.size != (x1 - x0, y1 - y0):
            im = im.resize((x1 - x0, y1 - y0), Image.LANCZOS)
        self.img.paste(im, (x0, y0))

    def region(self, box):
        x0, y0, x1, y1 = box
        return np.asarray(self.img.crop(box), float) / 255.0

    def save(self, path):
        self.img.save(path, optimize=True)


def c8(rgb):
    return tuple(int(round(255 * min(max(c, 0.0), 1.0))) for c in rgb)


def to8(a):
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def font(px):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:                     # Pillow without FreeType
        return ImageFont.load_default()


def lathe(profile, ts, box, at: Atlas, seg=48) -> Mesh:
    """Surface of revolution about z.

    profile: (r, z) points in order, starting on the axis at the bottom and
    running outward and up, so faces point out. A point listed twice makes a
    crisp edge there. ts: where each point lands down the texture box
    (0 top row .. 1 bottom row); u runs once round.
    """
    p = np.asarray(profile, float)
    ts = np.asarray(ts, float)
    k = len(p)
    th = np.linspace(0, 2 * np.pi, seg + 1)
    tan = np.diff(p, axis=0)
    segn = _unit(np.stack([tan[:, 1], -tan[:, 0]], 1))     # tangent turned outward
    pn = _unit(np.vstack([segn[:1], segn[:-1] + segn[1:], segn[-1:]]))
    R, TH = np.meshgrid(p[:, 0], th, indexing="ij")
    Z = np.repeat(p[:, 1][:, None], seg + 1, 1)
    v = np.stack([R * np.cos(TH), R * np.sin(TH), Z], -1).reshape(-1, 3)
    NR = np.repeat(pn[:, 0][:, None], seg + 1, 1)
    NZ = np.repeat(pn[:, 1][:, None], seg + 1, 1)
    n = np.stack([NR * np.cos(TH), NR * np.sin(TH), NZ], -1).reshape(-1, 3)
    # u runs clockwise seen from above, which is left to right seen from the
    # side: counter-clockwise would print every label mirrored
    S, T = np.meshgrid(1.0 - th / (2 * np.pi), ts)
    return Mesh(v, at.uv(box, S, T).reshape(-1, 2), _grid_faces(k, seg + 1), _unit(n))


def tube(path, radii, box, at: Atlas, seg=16, flat=1.0, caps=True, ts=None) -> Mesh:
    """A tube swept along a 3-D polyline, radius per point. `flat` < 1 squashes
    the cross-section along the binormal. u runs round, v along the path."""
    P = np.asarray(path, float)
    r = np.asarray(radii, float) * np.ones(len(P))
    T = _unit(np.gradient(P, axis=0))
    a = np.array([1.0, 0, 0]) if abs(T[0, 0]) < 0.9 else np.array([0, 0, 1.0])
    N = [_unit(a - (a @ T[0]) * T[0])]
    for i in range(1, len(P)):                      # parallel transport
        N.append(_unit(N[-1] - (N[-1] @ T[i]) * T[i]))
    N = np.array(N)
    B = np.cross(T, N)
    ph = np.linspace(0, 2 * np.pi, seg + 1)
    cp, sp = np.cos(ph)[None, :, None], np.sin(ph)[None, :, None]
    v = P[:, None] + r[:, None, None] * (cp * N[:, None] + flat * sp * B[:, None])
    n = _unit(flat * cp * N[:, None] + sp * B[:, None])
    if ts is None:
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        ts = s / max(s[-1], 1e-9)
    S, TT = np.meshgrid(ph / (2 * np.pi), ts)
    m = Mesh(v.reshape(-1, 3), at.uv(box, S, TT).reshape(-1, 2),
             _grid_faces(len(P), seg + 1), n.reshape(-1, 3))
    if not caps:
        return m
    parts = [m]
    for end, sgn in ((0, -1.0), (len(P) - 1, 1.0)):
        ring = v[end]
        cv = np.vstack([P[end], ring])
        cn = np.repeat(sgn * T[end][None], len(cv), 0)
        cuv = np.repeat(at.uv(box, 0.5, ts[end])[None], len(cv), 0)
        j = np.arange(seg)
        f = (np.stack([np.zeros(seg, int), j + 2, j + 1], 1) if sgn < 0 else
             np.stack([np.zeros(seg, int), j + 1, j + 2], 1))
        parts.append(Mesh(cv, cuv, f.astype(np.int32), cn))
    return merge(*parts)


def _spow(x, e):
    return np.sign(x) * np.abs(x) ** e


def superquadric(ext, box, at: Atlas, e_lat=1.0, e_lon=1.0, nu=48, nv=24,
                 uv="sphere") -> Mesh:
    """Superellipsoid with half-extents ext. e -> 0 squares it off, 1 is an
    ellipsoid. uv "sphere": longitude across (column s is longitude
    pi - 2 pi s, so it reads left to right from outside), north pole at the
    box top; "top": planar from above, +y at the box top (sides take the edge
    texels)."""
    a, b, c = ext
    lat = np.linspace(-np.pi / 2, np.pi / 2, nv + 1)
    lon = np.linspace(-np.pi, np.pi, nu + 1)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    cl, sl, co, so = np.cos(LA), np.sin(LA), np.cos(LO), np.sin(LO)
    x = a * _spow(cl, e_lat) * _spow(co, e_lon)
    y = b * _spow(cl, e_lat) * _spow(so, e_lon)
    z = c * _spow(sl, e_lat)
    nx = _spow(cl, 2 - e_lat) * _spow(co, 2 - e_lon) / a
    ny = _spow(cl, 2 - e_lat) * _spow(so, 2 - e_lon) / b
    nz = _spow(sl, 2 - e_lat) / c
    nz = np.where(np.abs(cl) < 1e-9, np.sign(sl), nz)
    v = np.stack([x, y, z], -1).reshape(-1, 3)
    n = _unit(np.stack([nx, ny, nz], -1).reshape(-1, 3))
    if uv == "sphere":
        S, T = 1.0 - (LO + np.pi) / (2 * np.pi), 1.0 - (LA + np.pi / 2) / np.pi
    else:
        S, T = (x / a + 1) / 2, (1 - y / b) / 2
    return Mesh(v, at.uv(box, S, T).reshape(-1, 2), _grid_faces(nv + 1, nu + 1), n)


def _triangulate(poly):
    """Ear clipping for a simple counter-clockwise polygon."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p, a, b, c):
        return cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0

    idx, tris = list(range(len(poly))), []
    while len(idx) > 3:
        n = len(idx)
        for k in range(n):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % n]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if cross(a, b, c) <= 1e-14:
                continue
            if any(inside(poly[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            raise ValueError("polygon is not simple")
    tris.append(tuple(idx))
    return np.array(tris, np.int32)


def slab(poly, thickness, top_box, side_box, at: Atlas) -> Mesh:
    """A flat solid: the polygon (N, 2) extruded to `thickness`, centred on
    z = 0. Top and bottom map planar into top_box, the edge into side_box."""
    P = np.asarray(poly, float)
    if np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]) < 0:
        P = P[::-1]
    h, k = thickness / 2, len(P)
    tris = _triangulate(P)
    lo, hi = P.min(0), P.max(0)
    s, t = (P[:, 0] - lo[0]) / (hi[0] - lo[0]), (hi[1] - P[:, 1]) / (hi[1] - lo[1])
    cap_uv = at.uv(top_box, s, t)
    top = Mesh(np.column_stack([P, np.full(k, h)]), cap_uv, tris,
               np.tile([0, 0, 1.0], (k, 1)))
    bot = Mesh(np.column_stack([P, np.full(k, -h)]), cap_uv, tris[:, ::-1],
               np.tile([0, 0, -1.0], (k, 1)))
    sv, sn, suv, sf = [], [], [], []
    for i in range(k):
        p, q = P[i], P[(i + 1) % k]
        d = q - p
        nrm = _unit(np.array([d[1], -d[0], 0.0]))
        base = len(sv)
        sv += [[*p, -h], [*q, -h], [*q, h], [*p, h]]
        sn += [nrm] * 4
        suv += list(at.uv(side_box, np.array([0, 1, 1, 0]) * 0.99, np.array([1, 1, 0, 0])))
        sf += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    side = Mesh(np.array(sv), np.array(suv), np.array(sf, np.int32), np.array(sn))
    return merge(top, bot, side)


def cuboid(half, boxes, at: Atlas) -> Mesh:
    """Hard-edged box centred on the origin; boxes: texture box per face, keys
    "+x" "-x" "+y" "-y" "+z" "-z". Every face reads the right way round from
    outside: side faces have +z at their texture's top, the top +y."""
    faces = {"+z": ((0, 0, 1), (0, -1, 0)), "-z": ((0, 0, -1), (0, 1, 0)),
             "+y": ((0, 1, 0), (0, 0, -1)), "-y": ((0, -1, 0), (0, 0, -1)),
             "+x": ((1, 0, 0), (0, 0, -1)), "-x": ((-1, 0, 0), (0, 0, -1))}
    parts = []
    for key, (n, w) in faces.items():
        n, w = np.array(n, float), np.array(w, float)     # w: texture "down"
        u = np.cross(n, w)                                 # texture "right", seen from outside
        hn, hu, hw = (np.abs(half) @ np.abs(n), np.abs(half) @ np.abs(u),
                      np.abs(half) @ np.abs(w))
        st = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        v = n * hn + (2 * st[:, :1] - 1) * hu * u + (2 * st[:, 1:] - 1) * hw * w
        # u x w = -n, so the corners run clockwise seen from outside: reverse them
        parts.append(Mesh(v, at.uv(boxes[key], st[:, 0], st[:, 1]),
                          np.array([[0, 2, 1], [0, 3, 2]], np.int32), np.tile(n, (4, 1))))
    return merge(*parts)


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


def _smooth_noise(shape, cell, seed):
    """Value noise bilinearly interpolated from a grid of `cell` px."""
    rng = np.random.default_rng(seed)
    gh, gw = shape[0] // cell + 2, shape[1] // cell + 2
    g = rng.random((gh, gw))
    y = np.arange(shape[0]) / cell
    x = np.arange(shape[1]) / cell
    y0, x0 = y.astype(int), x.astype(int)
    fy, fx = (y - y0)[:, None], (x - x0)[None, :]
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    a, b = g[y0][:, x0], g[y0][:, x0 + 1]
    c, d = g[y0 + 1][:, x0], g[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def solid(h, w, rgb, grain=0.0, seed=0, blotch=0.0):
    """A flat colour with fine grain and soft blotches: nothing real is uniform."""
    arr = np.ones((h, w, 3)) * np.asarray(rgb, float)
    shade = np.ones((h, w))
    if grain:
        shade += grain * (np.random.default_rng(seed).random((h, w)) - 0.5)
    if blotch:
        shade += blotch * (_smooth_noise((h, w), max(4, min(h, w) // 4), seed + 1) - 0.5)
    return arr * shade[..., None]


def oak_floor(size=1024, planks=7):
    """Oak boards: per-board tone, long grain, and dark seams across the joins."""
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


def wicker(size=512, tint=(0.74, 0.58, 0.36), rows=16, stakes=20):
    """Basket weave: flat weft strands passing over and under round stakes,
    each strand shaded across its width and dark in the gaps."""
    y, x = np.mgrid[0:size, 0:size].astype(float)
    rh, sw = size / rows, size / stakes
    row, fy = np.floor(y / rh), (y % rh) / rh
    col, fx = np.floor(x / sw), (x % sw) / sw
    over = ((row + col) % 2) == 0                     # weft passes over the stake here
    weft = np.sin(np.pi * fy) ** 0.6                  # rounded strand
    stake = np.sin(np.pi * fx) ** 0.6
    shade = np.where(over, 0.55 + 0.45 * weft, 0.35 + 0.55 * stake * (0.4 + 0.6 * weft))
    shade *= np.where((fy < 0.07) | (fy > 0.93), 0.45, 1.0)          # gaps between rows
    fibre = _noise((size, size), 5, seed=41)
    shade *= 0.85 + 0.25 * fibre
    tone = 0.90 + 0.2 * np.random.default_rng(42).random((rows + 1, stakes + 1))
    shade *= tone[row.astype(int), col.astype(int)]
    return np.clip(np.array(tint) * shade[..., None], 0, 1)


def save_png(path: Path, arr: np.ndarray):
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


# ============================================================ the household
# Each builder returns {mesh name: Mesh} and writes its own atlas. Dimensions
# are the collision primitives' in the scenes (scene_home.xml), which these
# dress: the body origin is the item's base, centred.

def mug(out: Path) -> dict:
    """Primitives: body cylinder r 0.040 z 0..0.100; handle box at y +0.050,
    half 0.008 x 0.012 x 0.030."""
    at = Atlas(512, 512)
    BODY, HANDLE = (0, 0, 512, 400), (0, 400, 512, 512)
    prof = [(0.000, 0.0006), (0.026, 0.0006), (0.030, 0.0), (0.034, 0.0), (0.036, 0.002),
            (0.037, 0.006), (0.0385, 0.015), (0.0395, 0.035), (0.0400, 0.060), (0.0400, 0.093),
            (0.0396, 0.0985), (0.0385, 0.1000), (0.0372, 0.0995), (0.0366, 0.0965),
            (0.0362, 0.060), (0.0355, 0.025), (0.0330, 0.012), (0.0200, 0.0095), (0.000, 0.0090)]
    ts = [0.00, 0.02, 0.03, 0.035, 0.045, 0.06, 0.10, 0.20, 0.33, 0.52, 0.56, 0.58, 0.60,
          0.62, 0.72, 0.84, 0.92, 0.96, 1.00]
    h = BODY[3] - BODY[1]
    glaze = np.array([0.70, 0.09, 0.08])
    img = solid(h, 512, glaze, grain=0.05, seed=1, blotch=0.10)
    rows = np.linspace(0, 1, h)[:, None]
    img *= (0.88 + 0.12 * np.clip((rows - 0.06) / 0.5, 0, 1))[..., None]   # glaze pools lower
    img[:int(0.06 * h)] = solid(int(0.06 * h), 512, (0.80, 0.72, 0.60), grain=0.15, seed=2)
    img[int(0.56 * h):int(0.62 * h)] = solid(int(0.62 * h) - int(0.56 * h), 512,
                                             (0.86, 0.36, 0.30), grain=0.05, seed=3)
    img[int(0.62 * h):] = solid(h - int(0.62 * h), 512, (0.93, 0.90, 0.84), grain=0.04, seed=4,
                                blotch=0.06)
    img[int(0.495 * h):int(0.505 * h)] = (0.95, 0.93, 0.88)     # a thin white band near the rim
    at.paste(BODY, img)
    # the print, on the two sides away from the handle (the handle is at +y, u = 0.25)
    for u in (0.0, 0.5, 1.0):
        at.draw.text((u * 512, 0.30 * h), "GOOD", font=font(22), fill=c8((0.97, 0.95, 0.9)),
                     anchor="mm")
        at.draw.text((u * 512, 0.38 * h), "MORNING", font=font(22), fill=c8((0.97, 0.95, 0.9)),
                     anchor="mm")
    at.paste(HANDLE, solid(112, 512, glaze * 0.95, grain=0.05, seed=5, blotch=0.08))
    body = lathe(prof, ts, BODY, at, seg=64)
    # The loop's ends stop inside the wall (r 0.0362..0.0400): run them any
    # further in and they show through on the inside of the cup.
    t = np.linspace(0.0, np.pi, 28)
    path = np.stack([np.zeros_like(t), 0.0378 + 0.0180 * np.sin(t), 0.050 + 0.025 * np.cos(t)], 1)
    handle = tube(path, 0.0074, HANDLE, at, seg=18, flat=0.6)
    at.save(out / "mug.png")
    return {"mug": merge(body, handle)}


def can(out: Path) -> dict:
    """Primitive: cylinder r 0.033, z 0..0.120."""
    at = Atlas(1024, 512)
    BODY, TAB = (0, 0, 1024, 480), (0, 484, 60, 512)
    prof = [(0.000, 0.0070), (0.012, 0.0060), (0.021, 0.0035), (0.0255, 0.0005),
            (0.0275, 0.0), (0.0300, 0.0015), (0.0325, 0.007), (0.0330, 0.013),
            (0.0330, 0.105), (0.0322, 0.1105), (0.0290, 0.1165), (0.0272, 0.1188),
            (0.0276, 0.1200), (0.0262, 0.1197), (0.0256, 0.1170), (0.0240, 0.1160),
            (0.000, 0.1160)]
    ts = [0.00, 0.02, 0.04, 0.06, 0.065, 0.075, 0.09, 0.10, 0.85, 0.87, 0.90, 0.92,
          0.935, 0.95, 0.96, 0.975, 1.00]
    h = BODY[3]
    at.paste(BODY, solid(h, 1024, (0.74, 0.75, 0.77), grain=0.10, seed=6, blotch=0.12))
    # the label, drawn at the can's true aspect (circumference : height) then fitted
    lw, lh = 1400, 620
    lab = Image.new("RGB", (lw, lh))
    y = np.linspace(0, 1, lh)[:, None, None]
    grad = np.array([0.05, 0.50, 0.20]) * (1 - y) + np.array([0.02, 0.33, 0.12]) * y
    lab.paste(Image.fromarray(to8(np.broadcast_to(grad, (lh, lw, 3)))))
    d = ImageDraw.Draw(lab)
    xs = np.linspace(0, lw, 200)
    for off, col, wid in ((0.50, (0.95, 0.97, 0.90), 60), (0.62, (0.85, 0.92, 0.25), 20)):
        pts = [(x, lh * (off + 0.10 * np.sin(2 * np.pi * x / lw * 2))) for x in xs]
        d.line(pts, fill=c8(col), width=wid)
    for cx in (0.25, 0.75):
        d.text((cx * lw, 0.26 * lh), "LIMA", font=font(150), fill=c8((0.98, 0.98, 0.95)),
               anchor="mm", stroke_width=6, stroke_fill=c8((0.02, 0.25, 0.08)))
        d.text((cx * lw, 0.80 * lh), "lemon-lime soda", font=font(46),
               fill=c8((0.98, 0.95, 0.4)), anchor="mm")
        for k, (dx, dy, r) in enumerate(((-230, 250, 44), (230, 230, 34), (190, 360, 26))):
            x0, y0 = cx * lw + dx, dy
            d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=c8((0.95, 0.88, 0.20)),
                      outline=c8((0.98, 0.98, 0.9)), width=4)
    d.text((0.5 * lw, 0.74 * lh), "12 FL OZ  355 mL", font=font(30), fill=c8((1, 1, 1)),
           anchor="mm")
    d.rectangle([0.445 * lw, 0.36 * lh, 0.555 * lw, 0.64 * lh], fill=(250, 250, 250))
    for i in range(22):                                           # barcode, between the logos
        w = 2 if i % 3 else 4
        d.rectangle([0.452 * lw + i * 6.5, 0.39 * lh, 0.452 * lw + i * 6.5 + w, 0.60 * lh],
                    fill=(10, 10, 10))
    lab_box = (0, int(0.10 * h), 1024, int(0.85 * h))
    at.paste(lab_box, np.asarray(lab, float) / 255)
    at.paste(TAB, solid(28, 60, (0.80, 0.80, 0.82), grain=0.08, seed=7))
    body = lathe(prof, ts, BODY, at, seg=64)
    tab = superquadric((0.0095, 0.0055, 0.0006), TAB, at, 0.4, 0.6, nu=24, nv=8, uv="top")
    tab = tab.moved(t=(0.0085, 0.0, 0.1167))
    at.save(out / "can.png")
    return {"can": merge(body, tab)}


def bottle(out: Path) -> dict:
    """Primitives: body cylinder r 0.035 z 0..0.180, neck cylinder r 0.016 z 0.180..0.220."""
    at = Atlas(1024, 512)
    BODY = (0, 0, 1024, 512)
    prof = [(0.000, 0.0040), (0.018, 0.0030), (0.030, 0.0000), (0.0335, 0.0020),
            (0.0350, 0.0080), (0.0350, 0.0300), (0.0338, 0.0340), (0.0350, 0.0380),
            (0.0338, 0.0420), (0.0350, 0.0460), (0.0350, 0.0650), (0.0350, 0.1350),
            (0.0350, 0.1450), (0.0335, 0.1560), (0.0290, 0.1660), (0.0225, 0.1740),
            (0.0160, 0.1800), (0.0148, 0.1830), (0.0160, 0.1850), (0.0160, 0.1880),
            (0.0145, 0.1890), (0.0145, 0.1920), (0.0160, 0.1925), (0.0160, 0.2185),
            (0.0150, 0.2200), (0.000, 0.2200)]
    ts = [0.00, 0.02, 0.04, 0.05, 0.06, 0.12, 0.13, 0.14, 0.15, 0.16, 0.20, 0.55, 0.58,
          0.62, 0.67, 0.72, 0.76, 0.78, 0.80, 0.81, 0.815, 0.82, 0.84, 0.95, 0.97, 1.00]
    h, w = BODY[3], BODY[2]
    plastic = solid(h, w, (0.78, 0.87, 0.93), grain=0.03, seed=8)
    u = np.linspace(0, 1, w)[None, :]
    plastic *= (0.90 + 0.10 * np.cos(2 * np.pi * u * 3) ** 2)[..., None]   # refraction bands
    at.paste(BODY, plastic)
    cap = solid(int(0.18 * h) + 1, w, (0.10, 0.32, 0.78), grain=0.05, seed=9)
    cap *= (0.80 + 0.20 * (np.sin(2 * np.pi * u * 60) > 0))[..., None]     # knurled edge
    at.paste((0, int(0.82 * h), w, h), cap)
    lw, lh = 1400, 440
    lab = Image.new("RGB", (lw, lh))
    y = np.linspace(0, 1, lh)[:, None, None]
    grad = np.array([0.97, 0.98, 1.0]) * (1 - y) + np.array([0.70, 0.84, 0.97]) * y
    lab.paste(Image.fromarray(to8(np.broadcast_to(grad, (lh, lw, 3)))))
    d = ImageDraw.Draw(lab)
    for cx in (0.25, 0.75):
        x0 = cx * lw
        d.polygon([(x0 - 260, 0.95 * lh), (x0 - 120, 0.52 * lh), (x0 - 40, 0.70 * lh),
                   (x0 + 90, 0.40 * lh), (x0 + 260, 0.95 * lh)], fill=c8((0.20, 0.45, 0.80)))
        d.polygon([(x0 + 60, 0.49 * lh), (x0 + 90, 0.40 * lh), (x0 + 122, 0.49 * lh)],
                  fill=(250, 250, 255))
        d.text((x0, 0.22 * lh), "AQUA", font=font(130), fill=c8((0.06, 0.25, 0.62)), anchor="mm")
        d.text((x0, 0.86 * lh), "natural spring water", font=font(36), fill=(255, 255, 255),
               anchor="mm")
    d.text((0.5 * lw, 0.5 * lh), "500 mL", font=font(40), fill=c8((0.06, 0.25, 0.62)), anchor="mm")
    d.rectangle([0, 0, lw, 10], fill=c8((0.06, 0.25, 0.62)))
    d.rectangle([0, lh - 10, lw, lh], fill=c8((0.06, 0.25, 0.62)))
    at.paste((0, int(0.20 * h), w, int(0.55 * h)), np.asarray(lab, float) / 255)
    at.save(out / "bottle.png")
    return {"bottle": lathe(prof, ts, BODY, at, seg=64)}


def remote(out: Path) -> dict:
    """Primitive: box half 0.090 x 0.024 x 0.0125, z 0..0.025. +x is the top
    end (the power key); +y is the left of a person holding it."""
    at = Atlas(512, 256)
    BODY = (0, 0, 512, 136)
    A, B, C = 0.0898, 0.0238, 0.0118
    at.paste(BODY, solid(136, 512, (0.075, 0.075, 0.085), grain=0.06, seed=10, blotch=0.10))
    # small print on the body, positioned in body coordinates
    to_px = lambda x, y: ((x / A + 1) / 2 * 512, (1 - y / B) / 2 * 136)
    at.draw.text(to_px(-0.078, 0.0), "SONORA", font=font(13), fill=(205, 205, 210), anchor="mm")
    at.draw.text(to_px(0.074, 0.021), "POWER", font=font(9), fill=(190, 190, 195), anchor="mm")
    for k, lbl in enumerate(("VOL", "CH")):
        at.draw.text(to_px(-0.038, (0.0125, -0.0125)[k] + 0.0), "", font=font(9))
        at.draw.text(to_px(-0.0515, (0.0125, -0.0125)[k]), lbl, font=font(10),
                     fill=(190, 190, 195), anchor="mm")
    parts = [superquadric((A, B, C), BODY, at, e_lat=0.25, e_lon=0.35, nu=72, nv=24, uv="top")
             .moved(t=(0, 0, C))]
    # one 32 px patch per key cap, so digits can be printed on them
    patches = {}

    def patch(key, colour, label="", fg=(235, 235, 235), size=15, kind="plain"):
        i = len(patches)
        box = (32 * (i % 16), 140 + 32 * (i // 16), 32 * (i % 16) + 32, 172 + 32 * (i // 16))
        at.paste(box, solid(32, 32, colour, grain=0.06, seed=20 + i))
        if kind == "dpad":
            for dx, dy in ((0, -11), (0, 11), (-11, 0), (11, 0)):
                cx, cy = box[0] + 16 + dx, box[1] + 16 + dy
                at.draw.regular_polygon((cx, cy, 3), 3, fill=(220, 220, 220),
                                        rotation={(0, -11): 0, (0, 11): 180, (-11, 0): 90,
                                                  (11, 0): 270}[(dx, dy)])
        if label:
            at.draw.text((box[0] + 16, box[1] + 16), label, font=font(size), fill=fg, anchor="mm")
        patches[key] = box
        return box

    keys = [("power", 0.074, 0.013, (0.0048, 0.0048, 0.0011), (0.80, 0.10, 0.08), "", 1.0),
            ("input", 0.074, -0.013, (0.0045, 0.0035, 0.0010), (0.22, 0.22, 0.24), "", 0.5)]
    digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "0", "#"]
    for k, dgt in enumerate(digits):
        x = (0.054, 0.042, 0.030, 0.018)[k // 3]
        y = (0.0125, 0.0, -0.0125)[k % 3]
        keys.append((f"d{dgt}", x, y, (0.0045, 0.0045, 0.0010), (0.33, 0.33, 0.35), dgt, 0.5))
    keys += [("dpad", -0.008, 0.0, (0.0125, 0.0125, 0.0008), (0.17, 0.17, 0.19), "", 0.5),
             ("ok", -0.008, 0.0, (0.0052, 0.0052, 0.0012), (0.40, 0.40, 0.43), "OK", 0.6),
             ("vol", -0.038, 0.0125, (0.0085, 0.0042, 0.0011), (0.30, 0.30, 0.32), "+", 0.5),
             ("ch", -0.038, -0.0125, (0.0085, 0.0042, 0.0011), (0.30, 0.30, 0.32), "+", 0.5)]
    for k, col in enumerate(((0.80, 0.12, 0.10), (0.15, 0.62, 0.20), (0.90, 0.78, 0.10),
                             (0.12, 0.32, 0.80))):
        keys.append((f"c{k}", -0.066, (0.0135, 0.0045, -0.0045, -0.0135)[k],
                     (0.003, 0.0035, 0.0009), col, "", 0.5))
    for name, x, y, ext, colour, label, e in keys:
        box = patch(name, colour, label, size=11 if name == "ok" else 16,
                    kind="dpad" if name == "dpad" else "plain")
        parts.append(superquadric(ext, box, at, e_lat=0.35, e_lon=e, nu=24, nv=8, uv="top")
                     .moved(rot_z(-np.pi / 2), (x, y, 2 * C)))
    at.save(out / "remote.png")
    return {"remote": merge(*parts)}


def _key_outline(length=0.056, bow_r=0.0105, blade_w=0.0072):
    """A house key flat on, bow centred on the origin, blade along +x."""
    pts = [(bow_r * np.cos(a), bow_r * np.sin(a))
           for a in np.linspace(np.deg2rad(38), np.deg2rad(322), 34)]
    tip = length - bow_r
    half = blade_w / 2
    pts += [(0.0125, -0.0048), (0.0138, -half), (tip - 0.004, -half), (tip, -0.0008),
            (tip - 0.0018, half - 0.0012)]
    xs = np.linspace(tip - 0.005, 0.017, 9)
    for i, x in enumerate(xs):                        # the cut, tip to shoulder
        pts.append((x, half if i % 2 else half - 0.0022))
    pts += [(0.0138, half), (0.0125, 0.0048)]
    return np.array(pts)


def keys(out: Path) -> dict:
    """Primitives: box half 0.035 x 0.018 x 0.012 centred (0, 0, 0.012), and a
    ring cylinder r 0.015 h 0.008 at (0.050, 0, 0.004). Two meshes: the metal
    (keys, ring, fob blade) and the car-key fob's plastic.

    Laid out as a dropped bunch: the fob on the -y side, the nickel key flat on
    the floor beside it, and the brass key with its bow on the ring and its
    blade resting across the fob -- which is what brings the bunch up to the
    primitive's 24 mm."""
    at = Atlas(256, 256)
    BRASS, NICKEL, STEEL, EDGE = (0, 0, 128, 128), (128, 0, 256, 128), (0, 128, 128, 256), \
        (128, 128, 256, 256)
    long_key, short_key = _key_outline(0.056, 0.0095), _key_outline(0.050, 0.0100)
    for box_, outline, seed, colour in ((BRASS, short_key, 30, (0.80, 0.63, 0.30)),
                                        (NICKEL, long_key, 31, (0.74, 0.75, 0.77))):
        img = solid(128, 128, colour, grain=0.10, seed=seed, blotch=0.14)
        lo, hi = outline.min(0), outline.max(0)
        # the hole in the bow, toward its end, where the ring goes through
        yy, xx = np.mgrid[0:128, 0:128]
        hx = (-0.0040 - lo[0]) / (hi[0] - lo[0]) * 128
        hy = hi[1] / (hi[1] - lo[1]) * 128
        img[(xx - hx) ** 2 + (yy - hy) ** 2 < 5.5 ** 2] = (0.08, 0.07, 0.06)
        at.paste(box_, img)
    at.paste(STEEL, solid(128, 128, (0.62, 0.64, 0.67), grain=0.08, seed=32))
    at.paste(EDGE, solid(128, 128, (0.55, 0.52, 0.45), grain=0.08, seed=33))
    tilt = np.deg2rad(23.0)
    key1 = slab(short_key, 0.0022, BRASS, EDGE, at).moved(rot_y(tilt) @ rot_z(np.pi),
                                                          (0.036, -0.003, 0.0060))
    key2 = slab(long_key, 0.0022, NICKEL, EDGE, at).moved(rot_z(np.pi - 0.09),
                                                          (0.032, 0.0065, 0.0011))
    a = np.linspace(0, 2 * np.pi, 49)
    ring = tube(np.stack([0.050 + 0.0115 * np.cos(a), 0.0115 * np.sin(a), np.full_like(a, 0.004)], 1),
                0.0011, STEEL, at, seg=8, caps=False)
    link = tube(np.array([[0.0110, -0.007, 0.008], [0.025, -0.006, 0.007], [0.0385, -0.004, 0.0045]]),
                0.0012, STEEL, at, seg=8)
    fob_blade = np.array([(-0.0215, -0.0035), (-0.0325, -0.0035), (-0.0338, -0.001),
                          (-0.0325, 0.0035), (-0.030, 0.0012), (-0.027, 0.0035), (-0.024, 0.0012),
                          (-0.0215, 0.0035)])
    blade = slab(fob_blade, 0.0024, NICKEL, EDGE, at).moved(t=(0.0, -0.007, 0.0075))
    metal = merge(key1, key2, ring, link, blade)

    ft = Atlas(256, 176)
    FOB = (0, 0, 256, 176)
    ft.paste(FOB, solid(176, 256, (0.06, 0.06, 0.065), grain=0.05, seed=34, blotch=0.10))
    for k, (cx, glyph) in enumerate(((0.40, "lock"), (0.66, "open"))):
        x0, y0 = cx * 256, 88
        ft.draw.rounded_rectangle([x0 - 24, y0 - 30, x0 + 24, y0 + 30], 10, fill=(38, 38, 42))
        ft.draw.rectangle([x0 - 8, y0 - 2, x0 + 8, y0 + 14], fill=(215, 215, 215))
        if glyph == "lock":
            ft.draw.arc([x0 - 7, y0 - 14, x0 + 7, y0 + 4], 180, 360, fill=(215, 215, 215), width=3)
        else:
            ft.draw.arc([x0 - 2, y0 - 16, x0 + 12, y0 + 2], 180, 360, fill=(215, 215, 215), width=3)
    ft.draw.ellipse([200, 70, 236, 106], outline=(170, 170, 175), width=4)
    fob = superquadric((0.0165, 0.010, 0.0075), FOB, ft, e_lat=0.40, e_lon=0.50, nu=40, nv=16,
                       uv="top").moved(t=(-0.005, -0.007, 0.0075))
    at.save(out / "keys_metal.png")
    ft.save(out / "keys_fob.png")
    return {"keys_metal": metal, "keys_fob": fob}


def ball(out: Path) -> dict:
    """Primitive: sphere r 0.035 centred (0, 0, 0.035). A tennis ball."""
    at = Atlas(512, 256)
    BOX = (0, 0, 512, 256)
    h, w = 256, 512
    felt = solid(h, w, (0.80, 0.88, 0.20), grain=0.22, seed=35, blotch=0.10)
    lat = np.pi / 2 - np.pi * (np.arange(h) + 0.5) / h
    lon = np.pi - 2 * np.pi * (np.arange(w) + 0.5) / w          # superquadric's "sphere" uv
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    d = np.stack([np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], -1).reshape(-1, 3)
    t = np.linspace(0, 2 * np.pi, 720)
    aa, bb = 0.72, 0.28
    curve = _unit(np.stack([aa * np.cos(t) + bb * np.cos(3 * t), aa * np.sin(t) - bb * np.sin(3 * t),
                            2 * np.sqrt(aa * bb) * np.sin(2 * t)], 1))
    best = np.full(len(d), -1.0)
    for k in range(0, len(d), 16384):
        best[k:k + 16384] = (d[k:k + 16384] @ curve.T).max(1)
    ang = np.arccos(np.clip(best, -1, 1)).reshape(h, w)
    seam = np.clip(1.0 - (ang - 0.030) / 0.018, 0, 1)[..., None]
    felt = felt * (1 - seam) + np.array([0.95, 0.95, 0.88]) * seam
    groove = np.exp(-((ang - 0.052) / 0.008) ** 2)[..., None]       # felt lifts along the seam
    at.paste(BOX, felt * (1 - 0.25 * groove))
    at.save(out / "ball.png")
    return {"ball": superquadric((0.035, 0.035, 0.035), BOX, at, nu=64, nv=32)
            .moved(t=(0, 0, 0.035))}


def box(out: Path) -> dict:
    """Primitive: box half 0.060 x 0.025 x 0.025, z 0..0.050. A taped carton."""
    at = Atlas(1024, 512)
    F = {"+z": (0, 0, 480, 200), "+y": (0, 200, 480, 400), "-y": (512, 0, 992, 200),
         "-z": (512, 200, 992, 400), "+x": (0, 410, 100, 510), "-x": (120, 410, 220, 510)}
    kraft = (0.66, 0.50, 0.32)

    def face(key, seed):
        x0, y0, x1, y1 = F[key]
        img = solid(y1 - y0, x1 - x0, kraft, grain=0.10, seed=seed, blotch=0.12)
        fib = _noise((y1 - y0, (x1 - x0) // 16 + 1), 3, seed=seed + 50)
        img *= (0.94 + 0.10 * np.repeat(fib, 16, axis=1)[:, :x1 - x0])[..., None]
        img[:3], img[-3:], img[:, :3], img[:, -3:] = (img[:3] * 0.75, img[-3:] * 0.75,
                                                      img[:, :3] * 0.75, img[:, -3:] * 0.75)
        return img

    top = face("+z", 40)
    top[97:103] *= 0.55                                       # flaps meet
    top[70:130] = top[70:130] * 0.6 + np.array([0.86, 0.76, 0.58]) * 0.4   # tape
    top[70:72] *= 0.8
    top[128:130] *= 0.8
    at.paste(F["+z"], top)
    at.paste(F["-z"], face("-z", 41))
    for key, seed in (("+x", 42), ("-x", 43)):
        end = face(key, seed)
        end[:30, 35:65] = end[:30, 35:65] * 0.6 + np.array([0.86, 0.76, 0.58]) * 0.4
        at.paste(F[key], end)
    side = face("+y", 44)
    at.paste(F["+y"], side)
    x0, y0 = F["+y"][:2]
    at.draw.rectangle([x0 + 40, y0 + 30, x0 + 250, y0 + 170], fill=(246, 244, 238),
                      outline=(200, 198, 190))
    for k in range(5):
        at.draw.rectangle([x0 + 55, y0 + 45 + 14 * k, x0 + 55 + (170 - 25 * (k % 3)), y0 + 50 + 14 * k],
                          fill=(60, 60, 60))
    for k in range(34):
        bw = 2 if k % 3 else 4
        at.draw.rectangle([x0 + 55 + 5 * k, y0 + 125, x0 + 55 + 5 * k + bw, y0 + 160], fill=(15, 15, 15))
    at.draw.text((x0 + 370, y0 + 100), "SHIP TO", font=font(22), fill=(40, 30, 20), anchor="mm")
    other = face("-y", 45)
    at.paste(F["-y"], other)
    x0, y0 = F["-y"][:2]
    for dx in (60, 110):
        at.draw.line([(x0 + dx, y0 + 150), (x0 + dx, y0 + 70)], fill=(30, 25, 20), width=7)
        at.draw.polygon([(x0 + dx - 16, y0 + 78), (x0 + dx + 16, y0 + 78), (x0 + dx, y0 + 50)],
                        fill=(30, 25, 20))
    at.draw.text((x0 + 85, y0 + 175), "THIS SIDE UP", font=font(16), fill=(30, 25, 20), anchor="mm")
    at.draw.rectangle([x0 + 220, y0 + 60, x0 + 440, y0 + 140], outline=(170, 30, 25), width=5)
    at.draw.text((x0 + 330, y0 + 100), "FRAGILE", font=font(44), fill=(170, 30, 25), anchor="mm")
    at.save(out / "box.png")
    return {"box": cuboid(np.array([0.060, 0.025, 0.025]), F, at).moved(t=(0, 0, 0.025))}


def person(out: Path) -> dict:
    """Dresses scene_home.xml's person: legs box half 0.13 x 0.09 z 0..0.84,
    torso cylinder r 0.17 z 0.82..1.38, head sphere r 0.11 at z 1.52, and the
    open hand (palm box at y -0.43, z 0.615, half 0.07 x 0.06 x 0.012) held out
    toward -y. Faces -y. Everything stays inside x +-0.17, y -0.49..0.17,
    z 0..1.63 -- the footprint the place planner reads off this body."""
    at = Atlas(1024, 1024)
    SHIRT, ARM_L, ARM_R = (0, 0, 512, 512), (512, 0, 768, 512), (768, 0, 1024, 512)
    PELVIS, LEGS = (0, 512, 256, 1024), (256, 512, 512, 1024)
    SHOES, SKIN, HEAD = (512, 512, 640, 640), (640, 512, 768, 640), (512, 640, 1024, 1024)
    shirt_c, skin_c = (0.16, 0.34, 0.50), (0.84, 0.64, 0.51)
    jeans_c = (0.18, 0.26, 0.40)

    sh = solid(512, 512, shirt_c, grain=0.10, seed=60, blotch=0.14)
    folds = _smooth_noise((512, 512), 24, 61)
    sh *= (0.88 + 0.2 * folds)[..., None]
    at.paste(SHIRT, sh)
    sleeve = 0.34
    for box, seed in ((ARM_L, 62), (ARM_R, 63)):
        arm = solid(512, 256, skin_c, grain=0.04, seed=seed, blotch=0.06)
        arm[:int(sleeve * 512)] = solid(int(sleeve * 512), 256, shirt_c, grain=0.10, seed=seed + 10)
        arm[int(sleeve * 512) - 6:int(sleeve * 512)] *= 0.7      # hem
        at.paste(box, arm)
    pel = solid(512, 256, jeans_c, grain=0.18, seed=64, blotch=0.10)
    pel[120:175] = solid(55, 256, (0.25, 0.16, 0.09), grain=0.08, seed=65)   # belt
    pel[140:156, 118:138] = (0.72, 0.68, 0.55)                               # buckle
    at.paste(PELVIS, pel)
    legs = solid(512, 256, jeans_c, grain=0.22, seed=66, blotch=0.12)
    yy, xx = np.mgrid[0:512, 0:256]
    legs *= (0.92 + 0.08 * ((xx + yy) % 6 < 3))[..., None]                   # twill
    legs[:, 60:64] *= 0.75                                                    # side seams
    legs[:, 188:192] *= 0.75
    legs[470:] *= 0.85                                                        # turned-up hem
    at.paste(LEGS, legs)
    shoes = solid(128, 128, (0.13, 0.12, 0.12), grain=0.08, seed=67, blotch=0.10)
    shoes[98:] = (0.92, 0.91, 0.88)                                           # sole
    shoes[92:98] = (0.30, 0.28, 0.26)
    at.paste(SHOES, shoes)
    at.paste(SKIN, solid(128, 128, skin_c, grain=0.04, seed=68, blotch=0.05))

    # the head, painted in longitude/latitude: face toward -y (lon -90 deg)
    hh, hw = 384, 512
    lat = np.pi / 2 - np.pi * (np.arange(hh) + 0.5) / hh
    lon = np.pi - 2 * np.pi * (np.arange(hw) + 0.5) / hw          # superquadric's "sphere" uv
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    dlon = (LO + np.pi / 2 + np.pi) % (2 * np.pi) - np.pi                    # 0 = straight ahead
    head = solid(hh, hw, skin_c, grain=0.03, seed=69, blotch=0.05)
    hair_c = np.array([0.17, 0.11, 0.07])
    hair = (LA > 0.50 - 0.12 * np.cos(dlon)) | ((np.abs(dlon) > 1.75) & (LA > -0.55))
    hair |= (np.abs(dlon) > 1.25) & (LA > 0.05)
    head[hair] = hair_c * (0.8 + 0.4 * _noise((hh, hw), 4, seed=70)[hair])[:, None]
    ell = lambda l0, o0, dl, do: ((LA - l0) / dl) ** 2 + ((dlon - o0) / do) ** 2
    for side in (-1, 1):
        head[ell(0.12, side * 0.36, 0.055, 0.14) < 1] = (0.95, 0.94, 0.92)
        head[ell(0.12, side * 0.36, 0.045, 0.055) < 1] = (0.28, 0.18, 0.10)
        head[ell(0.12, side * 0.36, 0.022, 0.028) < 1] = (0.05, 0.04, 0.04)
        head[ell(0.25, side * 0.36, 0.022, 0.17) < 1] = hair_c                 # brows
    head[ell(-0.05, 0.0, 0.10, 0.07) < 1] *= 0.90                              # nose shading
    head[ell(-0.34, 0.0, 0.035, 0.22) < 1] = (0.62, 0.30, 0.28)                # mouth
    head[ell(-0.20, side * 0.45, 0.15, 0.2) < 1] *= 1.0
    at.paste(HEAD, head)

    parts = []
    for sx in (-1, 1):
        parts.append(superquadric((0.045, 0.115, 0.035), SHOES, at, e_lat=0.55, e_lon=0.60,
                                  nu=32, nv=16).moved(t=(sx * 0.072, -0.05, 0.035)))
        leg = np.array([[sx * 0.080, 0.0, 0.90], [sx * 0.079, -0.004, 0.70],
                        [sx * 0.078, -0.010, 0.48], [sx * 0.075, -0.004, 0.28],
                        [sx * 0.072, 0.0, 0.07]])
        parts.append(tube(leg, [0.070, 0.062, 0.052, 0.046, 0.040], LEGS, at, seg=24))
    parts.append(superquadric((0.155, 0.095, 0.10), PELVIS, at, e_lat=0.6, e_lon=0.7, nu=40,
                              nv=16).moved(t=(0, 0, 0.93)))
    parts.append(superquadric((0.140, 0.095, 0.25), SHIRT, at, e_lat=0.55, e_lon=0.70, nu=48,
                              nv=24).moved(t=(0, 0, 1.13)))
    parts.append(tube(np.array([[0, 0, 1.36], [0, 0, 1.45]]), 0.048, SKIN, at, seg=20))
    parts.append(superquadric((0.075, 0.090, 0.108), HEAD, at, nu=48, nv=32)
                 .moved(t=(0, -0.005, 1.52)))
    for sx in (-1, 1):
        parts.append(superquadric((0.012, 0.020, 0.028), SKIN, at, nu=12, nv=8)
                     .moved(t=(sx * 0.074, 0.0, 1.51)))
    arm_l = np.array([[-0.128, 0.0, 1.33], [-0.132, 0.01, 1.06], [-0.128, -0.01, 0.82]])
    parts.append(tube(arm_l, [0.040, 0.034, 0.028], ARM_L, at, seg=20, ts=[0, 0.5, 1]))
    parts.append(superquadric((0.020, 0.040, 0.075), SKIN, at, e_lat=0.8, e_lon=0.8, nu=16,
                              nv=10).moved(t=(-0.128, -0.01, 0.745)))
    arm_r = np.array([[0.128, -0.01, 1.33], [0.095, -0.19, 1.00], [0.022, -0.365, 0.638]])
    parts.append(tube(arm_r, [0.040, 0.034, 0.028], ARM_R, at, seg=20, ts=[0, 0.5, 1]))
    # the open hand, palm up, on the palm box
    parts.append(superquadric((0.040, 0.045, 0.011), SKIN, at, e_lat=0.6, e_lon=0.6, nu=20,
                              nv=10).moved(t=(0.0, -0.415, 0.615)))
    for fx in (-0.027, -0.009, 0.009, 0.027):
        parts.append(tube(np.array([[fx, -0.450, 0.617], [fx, -0.481, 0.617]]), 0.0085, SKIN, at,
                          seg=10))
    parts.append(tube(np.array([[0.036, -0.395, 0.615], [0.058, -0.438, 0.620]]), 0.0095, SKIN, at,
                      seg=10))
    at.save(out / "person.png")
    return {"person": merge(*parts)}


HOUSEHOLD = (mug, can, bottle, remote, keys, ball, box, person)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meshes = {"foliage": foliage_mesh(), "pot": pot_mesh()}
    for name, (v, f) in meshes.items():
        write_stl(OUT / f"{name}.stl", v, f)
        print(f"  {name}.stl  {len(v)} verts  {len(f)} tris")

    for build in HOUSEHOLD:
        for name, m in build(OUT).items():
            write_obj(OUT / f"{name}.obj", m)
            lo, hi = m.v.min(0), m.v.max(0)
            print(f"  {name}.obj  {len(m.v)} verts  {len(m.f)} tris  "
                  f"x {lo[0]:+.4f}..{hi[0]:+.4f}  y {lo[1]:+.4f}..{hi[1]:+.4f}  "
                  f"z {lo[2]:+.4f}..{hi[2]:+.4f}")

    textures = {
        "floor_oak": oak_floor(), "wall_plaster": plaster(),
        "sofa_weave": woven(), "wood_walnut": walnut(),
        "wood_oak": walnut(tint=(0.45, 0.30, 0.17)),
        "metal_brushed": brushed(), "wicker": wicker(),
    }
    for name, arr in textures.items():
        save_png(OUT / f"{name}.png", arr)
        print(f"  {name}.png  {arr.shape[1]}x{arr.shape[0]}")


if __name__ == "__main__":
    main()
