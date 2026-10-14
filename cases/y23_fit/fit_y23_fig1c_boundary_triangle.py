"""Fit camera + shape of Fig. 1(c) to the paper panels (600 dpi crops).

Model (see y23_fig1c_boundary_triangle.py), parameters (u, w) in [0, 1]^2, U = 2u - 1, V = 2w - 1:
  orange O(u, w) = (X, Y, q):  X = U (1 - p (1 - V^2)), Y = L V (1 - p (1 - U^2))  (edges bowed inward),
                               q = a20 U^2 + a11 U V + a02 V^2 + b10 U + b01 V + a22 U^2 V^2;
  blue   B(u, w) = O(u, w) + D(u, w) e_z,  D = k u w F (1 + m F^2), F = u + w + a u w - 1, on the curved triangle
         0 <= w <= g(u) = (1 - u)(1 + e1 u + e2 u^2 + e3 u^3)  (beyond the interior curve).
The plan-view map (u, w) -> (X, Y) is injective, so B(u, w) lies on O only where D(u, w) = 0: the two
shared boundary edges u = 0, w = 0 and the interior curve w = (1 - u) / (1 + a u).

Objective: 1 - IoU(union silhouette, Surfaces) + 1 - IoU(visible blue, Surfaces blue pixels)
+ 0.1 * chamfer(triangle, red pixels of the Ours panel) + 0.03 * outline chamfer; the Ours panel
has its own 2D framing (offset, scale, roll).
Run: .venv/bin/python -I cases/y23_fit/fit_y23_fig1c_boundary_triangle.py [maxiter] [x0-json]
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_match as rm  # noqa: E402

STEM = "y23_fig1c_boundary_triangle"
SURF = np.asarray(Image.open(os.path.join(HERE, "paper", STEM + "_surfaces.png")).convert("RGB"))
OURS = np.asarray(Image.open(os.path.join(HERE, "paper", STEM + "_ours.png")).convert("RGB"))
H, W = SURF.shape[:2]
FG = rm.paper_mask(SURF)
PB, PO, _ = rm.color_masks(SURF)
_, _, RED = rm.color_masks(OURS)
DT_RED = ndimage.distance_transform_edt(~RED)
RY, RX = np.nonzero(RED)
FG_EDGE = FG & ~ndimage.binary_erosion(FG)
DT_FG_EDGE = ndimage.distance_transform_edt(~FG_EDGE)
NAMES = ["el", "az", "roll", "f", "dist", "cx", "cy", "odx", "ody", "osc", "oroll",
         "L", "a20", "a11", "a02", "b10", "b01", "a", "k", "e1", "e2", "pinch", "a22", "e3", "m"]


def XY(u, w, sh):
    """Plan-view map of the parameter square: U = 2u - 1, V = 2w - 1,
    X = U (1 - p (1 - V^2)), Y = L V (1 - p (1 - U^2)) (edges bowed inward by p)."""
    L, p = sh[0], sh[10]
    U, V = 2 * u - 1, 2 * w - 1
    return U * (1 - p * (1 - V * V)), L * V * (1 - p * (1 - U * U))


def q(u, w, sh):
    L, a20, a11, a02, b10, b01 = sh[:6]
    U, V = 2 * u - 1, 2 * w - 1
    return a20 * U * U + a11 * U * V + a02 * V * V + b10 * U + b01 * V + sh[11] * U * U * V * V


def D(u, w, sh):
    a, k, m = sh[6], sh[7], sh[13]
    F = u + w + a * u * w - 1
    return k * u * w * F * (1 + m * F * F)


def g(u, sh):
    e1, e2, e3 = sh[8], sh[9], sh[12]
    return (1 - u) * (1 + e1 * u + e2 * u * u + e3 * u ** 3)


def surf(u, w, sh, blue):
    X, Y_ = XY(u, w, sh)
    Z = q(u, w, sh) + (D(u, w, sh) if blue else 0)
    return np.stack(np.broadcast_arrays(X, Y_, Z), -1)


def grids(sh, n=160):
    u = np.linspace(0, 1, n)[:, None]
    w = np.linspace(0, 1, n)[None, :]
    orange = surf(u, w, sh, False)
    blue = surf(u, w * g(u, sh), sh, True)
    return blue, orange


def curves(sh, n=200):
    a = sh[6]
    t = np.linspace(0, 1, n)
    return [surf(0 * t, t, sh, False), surf(t, 0 * t, sh, False), surf(t, (1 - t) / (1 + a * t), sh, False)]


def zsplat(P, cam):
    x, y, d = rm.project(P.reshape(-1, 3), cam, W, H)
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (d > 0)
    ix, iy, d = x[ok].astype(int), y[ok].astype(int), d[ok]
    dep = np.full((H, W), np.inf)
    np.minimum.at(dep, (iy, ix), d)
    m = np.isfinite(dep)
    m2 = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=2))
    dep = np.where(m, dep, ndimage.grey_erosion(np.where(m, dep, np.inf), size=5))
    return m2, dep


def cam_of(p):
    el, az, roll, f, dist, cx, cy = p[:7]
    return {"proj": "persp", "el": float(el), "az": float(az), "roll": float(roll), "scale": float(f),
            "dist": float(dist), "cx": float(cx), "cy": float(cy), "target": [0.0, 0.0, 0.0]}


def ours_cam(p):
    c = cam_of(p)
    c["cx"] += p[7]
    c["cy"] += p[8]
    c["scale"] *= p[9]
    c["roll"] += p[10]
    return c


def loss(p, verbose=False):
    sh = p[11:]
    cam = cam_of(p)
    a = sh[6]
    if p[4] < 1.5 or a <= -0.9 or not 0 <= sh[10] < 0.45:
        return 10.0
    uu = np.linspace(0, 1, 101)
    # g beyond the interior curve, so that D changes sign only there
    if (g(uu[1:-1], sh) <= (1 - uu[1:-1]) / (1 + a * uu[1:-1]) + 1e-4).any():
        return 10.0
    blue, orange = grids(sh)
    mb, db = zsplat(blue, cam)
    mo, do = zsplat(orange, cam)
    iou = rm.mask_iou(mb | mo, FG)
    vis_b = mb & ~(mo & (do < db - 0.003))
    iou_b = rm.mask_iou(vis_b, PB)
    oc = ours_cam(p)
    pts = np.vstack([np.stack(rm.project(Ci, oc, W, H)[:2], 1) for Ci in curves(sh)])
    ok = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
    if ok.mean() < 0.97:
        return 10.0
    d1 = DT_RED[pts[ok, 1].astype(int), pts[ok, 0].astype(int)].mean()
    d2 = np.min(np.hypot(RX[:, None] - pts[None, ::3, 0], RY[:, None] - pts[None, ::3, 1]), 1).mean()
    m = mb | mo
    edge = m & ~ndimage.binary_erosion(m)
    e1 = DT_FG_EDGE[edge].mean()
    e2 = ndimage.distance_transform_edt(~edge)[FG_EDGE].mean()
    if verbose:
        print("IoU %.4f  blue IoU %.4f  triangle chamfer %.2f / %.2f px  outline chamfer %.2f / %.2f px"
              % (iou, iou_b, d1, d2, e1, e2))
    return (1 - iou) + (1 - iou_b) + 0.1 * (d1 + d2) + 0.03 * (e1 + e2)


BOUNDS = [(20, 80), (-180, 180), (-180, 180), (200, 4000), (1.5, 20), (150, 280), (130, 250), (-30, 30), (-30, 30),
          (0.85, 1.15), (-8, 8),
          (0.5, 2.0), (-1.5, 1.5), (-2.0, 2.0), (-1.5, 1.5), (-1.0, 1.0), (-1.0, 1.0), (-0.5, 2.0), (0.02, 10.0),
          (-3.0, 8.0), (-15.0, 15.0), (0.0, 0.4), (-1.0, 1.0), (-15.0, 15.0), (0.0, 200.0)]

if __name__ == "__main__":
    x0 = np.array(json.loads(sys.argv[2])) if len(sys.argv) > 2 else None
    # optional: comma-separated shape names held at zero (lower-degree variants)
    for name in (sys.argv[3].split(",") if len(sys.argv) > 3 else []):
        BOUNDS[NAMES.index(name)] = (-1e-12, 1e-12)
        if x0 is not None:
            x0[NAMES.index(name)] = 0.0
    r = optimize.differential_evolution(loss, BOUNDS, popsize=20, maxiter=int(sys.argv[1]) if len(sys.argv) > 1 else 200,
                                        tol=1e-8, seed=4, polish=False, updating="deferred", workers=8, x0=x0)
    q_ = optimize.minimize(loss, r.x, method="Nelder-Mead",
                           options={"maxiter": 4000, "adaptive": True, "xatol": 1e-5, "fatol": 1e-8})
    p = q_.x if q_.fun < r.fun else r.x
    loss(p, verbose=True)
    print(json.dumps({"camera": cam_of(p), "ours_camera": ours_cam(p), "shape": dict(zip(NAMES[11:], map(float, p[11:]))),
                      "loss": float(min(q_.fun, r.fun)), "p": p.tolist()}))
