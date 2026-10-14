"""Shared helpers for the Table 6 example (4)/(5)/(6) refits (y23_t6_ex4/5/6_*.py).

Polynomial bi-quadratic patches are carried as power-coefficient arrays c[i][j] (u^i v^j)
per coordinate, so that every construction stays exact and the case truth is the patch
itself (ssi_geom.rational_poly_patch). Silhouettes are rasterised from a parameter grid
for the camera/shape fits; the final pictures use render_match.render.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "cases"))
sys.path.insert(0, os.path.join(ROOT, "scoring"))
import render_match as rm  # noqa: E402
import ssi_geom as g  # noqa: E402

PAPER = os.path.join(HERE, "paper")
COMPARE = os.path.join(HERE, "compare")

# Bernstein (degree 2) -> power basis: B_k(t) = sum_i M[k, i] t^i
M2 = np.array([[1.0, -2.0, 1.0], [0.0, 2.0, -2.0], [0.0, 0.0, 1.0]])


def bern_pow(B):
    """3x3 Bernstein coefficients b[k][l] -> power coefficients c[i][j] (u^i v^j)."""
    return M2.T @ np.asarray(B, float) @ M2


def peval(c, u, v):
    """Evaluate power coefficients c[i][j] at broadcast (u, v)."""
    c = np.asarray(c, float)
    out = 0.0
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c[i, j]:
                out = out + c[i, j] * u ** i * v ** j
    return out + 0.0 * (u + v)


def pd(c, k):
    """Partial derivative of power coefficients along u (k=0) or v (k=1)."""
    c = np.asarray(c, float)
    if k == 0:
        return c[1:] * np.arange(1, c.shape[0])[:, None] if c.shape[0] > 1 else np.zeros((1, c.shape[1]))
    return c[:, 1:] * np.arange(1, c.shape[1])[None, :] if c.shape[1] > 1 else np.zeros((c.shape[0], 1))


def grid3(P, n=41, m=None):
    """P = (X, Y, Z) power arrays -> (n, m, 3) grid on [0,1]^2."""
    m = m or n
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, m), indexing="ij")
    return np.stack([peval(c, u, v) for c in P], -1)


def patch(P, label):
    """Exact (2,2) polynomial Bezier patch from power arrays (X, Y, Z)."""
    return g.rational_poly_patch(P[0], P[1], P[2], [[1.0]], (0.0, 1.0), (0.0, 1.0), label, degree=(2, 2))


def load_panel(stem, which):
    return np.asarray(Image.open(os.path.join(PAPER, "%s_%s.png" % (stem, which))).convert("RGB"))


def grid_mask(G, cam, W, H, ss=1):
    """Filled silhouette of a surface grid (n, m, 3) from cam, W x H."""
    x, y, _ = rm.project(G, cam, W, H)
    img = Image.new("L", (W * ss, H * ss), 0)
    dr = ImageDraw.Draw(img)
    n, m = G.shape[:2]
    X, Y = x * ss, y * ss
    for i in range(n - 1):
        for j in range(m - 1):
            dr.polygon([(X[i, j], Y[i, j]), (X[i + 1, j], Y[i + 1, j]), (X[i + 1, j + 1], Y[i + 1, j + 1]),
                        (X[i, j + 1], Y[i, j + 1])], fill=255)
    a = np.asarray(img)
    if ss > 1:
        a = np.asarray(Image.fromarray(a).resize((W, H), Image.BILINEAR))
    return a > 127


def red_mask(rgb):
    a = np.asarray(rgb, float) / 255.0
    return (a[..., 0] > 0.8) & (a[..., 1] < 0.25) & (a[..., 2] < 0.25)


def curve_px(C, cam, W, H):
    x, y, _ = rm.project(np.asarray(C, float), cam, W, H)
    return np.stack([x, y], 1)


def chamfer(curves_px, red, both=True, sub=3):
    """Mean distance (px) curve->red pixels, and (both) red pixels->curve."""
    P = np.vstack(curves_px)
    H, W = red.shape
    dt = ndimage.distance_transform_edt(~red)
    xi = np.clip(np.round(P[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(P[:, 1]).astype(int), 0, H - 1)
    out = (P[:, 0] < 0) | (P[:, 0] >= W) | (P[:, 1] < 0) | (P[:, 1] >= H)
    d1 = np.where(out, 50.0, dt[yi, xi]).mean()
    if not both:
        return d1
    ry, rx = np.nonzero(red)
    ry, rx = ry[::sub], rx[::sub]
    Q = P[::2]
    d2 = np.sqrt(((rx[:, None] - Q[None, :, 0]) ** 2 + (ry[:, None] - Q[None, :, 1]) ** 2).min(1)).mean()
    return 0.5 * (d1 + d2)


def chamfer_split(vis_px, all_px, red, sub=3):
    """0.5 * (visible curves -> red pixels  +  red pixels -> all curves): curves hidden in the
    paper's picture are not pulled toward red pixels, but every red pixel must be explained."""
    H, W = red.shape
    dt = ndimage.distance_transform_edt(~red)
    P = np.vstack(vis_px)
    xi = np.clip(np.round(P[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(P[:, 1]).astype(int), 0, H - 1)
    out = (P[:, 0] < 0) | (P[:, 0] >= W) | (P[:, 1] < 0) | (P[:, 1] >= H)
    d1 = np.where(out, 50.0, dt[yi, xi]).mean()
    Q = np.vstack(all_px)[::2]
    ry, rx = np.nonzero(red)
    ry, rx = ry[::sub], rx[::sub]
    d2 = np.sqrt(((rx[:, None] - Q[None, :, 0]) ** 2 + (ry[:, None] - Q[None, :, 1]) ** 2).min(1)).mean()
    return 0.5 * (d1 + d2)


def overlay_strip(rows, out_path, gap=8):
    """rows: list of (paper_rgb, ours_rgb); each row paper | ours | 50% overlay."""
    rm.compare_strip([r[0] for r in rows], [r[1] for r in rows], out_path)


def hstack(imgs, out_path, pad=10):
    h = max(i.shape[0] for i in imgs)
    ims = [np.pad(i[..., :3], ((0, h - i.shape[0]), (0, pad), (0, 0)), constant_values=255) for i in imgs]
    Image.fromarray(np.concatenate(ims, 1)).save(out_path)


def footprint_jacobian(P, n):
    """Jacobian determinant (power coefficients) of the map (u, v) -> (P . e1, P . e2), with e1, e2
    spanning the plane normal to the unit direction n; also returns the two projected coordinates."""
    n = np.asarray(n, float)
    e1 = np.cross(n, (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    X = g.p2_add(g.p2_add(e1[0] * P[0], e1[1] * P[1]), e1[2] * P[2])
    Yc = g.p2_add(g.p2_add(e2[0] * P[0], e2[1] * P[1]), e2[2] * P[2])
    det = g.p2_add(g.p2_mul(pd(X, 0), pd(Yc, 1)), -1.0 * g.p2_mul(pd(X, 1), pd(Yc, 0)))
    return det, (X, Yc)


def certify_footprint(P, n, depth=7):
    """Certify that the patch P (power arrays on [0,1]^2) is a graph along n, i.e. its footprint
    map along n is injective:
      (1) the Jacobian determinant has one strict sign: Bernstein coefficients (degree (4,4)) of one
          sign on every cell of an adaptive subdivision (the Bernstein hull bounds the polynomial);
      (2) the boundary maps to a simple closed curve (no two non-adjacent chords of a 1600-segment
          polyline cross); with (1) the map is then a homeomorphism onto the enclosed region.
    Then every line parallel to n meets the patch at most once. Returns (sign, min coefficient)."""
    det, (X, Yc) = footprint_jacobian(P, n)
    sgn = np.sign(g.p2_eval(det, 0.5, 0.5))
    worst = np.inf
    todo = [(0.0, 1.0, 0.0, 1.0, 0)]
    while todo:
        u0, u1, v0, v1, k = todo.pop()
        b = sgn * g.bernstein_from_power_2d(g.p2_reparam(det, u0, u1, v0, v1), 4, 4)
        if (b > 0).all():
            worst = min(worst, float(b.min()))
            continue
        assert k < depth, ("Jacobian sign not certified", u0, u1, v0, v1)
        um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        todo += [(u0, um, v0, vm, k + 1), (um, u1, v0, vm, k + 1), (u0, um, vm, v1, k + 1), (um, u1, vm, v1, k + 1)]
    t = np.linspace(0, 1, 401)[:-1]
    uv = np.concatenate([np.column_stack([t, 0 * t]), np.column_stack([1 + 0 * t, t]),
                         np.column_stack([1 - t, 1 + 0 * t]), np.column_stack([0 * t, 1 - t])])
    Q = np.column_stack([g.p2_eval(X, uv[:, 0], uv[:, 1]), g.p2_eval(Yc, uv[:, 0], uv[:, 1])])
    A, B = Q, np.roll(Q, -1, 0)
    m = len(Q)
    d1 = B - A
    cr = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    for i in range(m):
        j = np.arange(i + 2, m)
        if i == 0:
            j = j[j != m - 1]
        if not len(j):
            continue
        o1 = cr(d1[i], A[j] - A[i]) * cr(d1[i], B[j] - A[i])
        o2 = cr(d1[j], A[i] - A[j]) * cr(d1[j], B[i] - A[j])
        assert not ((o1 < 0) & (o2 < 0)).any(), "footprint boundary self-intersects"
    return int(sgn), worst
