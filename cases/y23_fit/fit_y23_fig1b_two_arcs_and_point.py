"""Fit camera + shape of Fig. 1(b) to the paper panels (600 dpi crops).

Model (see y23_fig1b_two_arcs_and_point.py), world: x toward the viewer, y to the right, z up.
  blue   S(s, phi) = (R(s) cos phi, s, R(s) sin phi), s in [-Lb, Lb], phi all round:
         a spool (closed tube), R(s) = r0 + r2 u^2 + r4 u^4 + r6 u^6, u = s / Lb, r0 = 1.
  orange O(s, psi) = (c(s) + A cos psi, s, Bz(s) sin psi), s in [-So, So], psi in [-90, 90] deg:
         the front half of an elliptic barrel, Bz = b0 + b2 u^2 + b4 u^4, c = r0 - A + c2 u^2 + c4 u^4 (so the two touch at
         (r0, 0, 0)).
Per s the two cross-sections (circle, ellipse) meet where a quadratic in w = cos phi vanishes.
The fit samples only the front halves (the back of the spool is hidden in this view).

Objective (pixels of the 431 x 380 panels):
  1 - IoU(union silhouette, Surfaces panel) + 1 - IoU(visible blue, Surfaces panel blue pixels)
  + 0.1 * chamfer(curves, red pixels of the Ours panel) + 0.05 * |dot error|,
the Ours panel being framed with its own 2D offset and scale (the paper's rows differ by a few px).
Run: .venv/bin/python -I cases/y23_fit/fit_y23_fig1b_two_arcs_and_point.py [maxiter] [x0-json]
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

STEM = "y23_fig1b_two_arcs_and_point"
SURF = np.asarray(Image.open(os.path.join(HERE, "paper", STEM + "_surfaces.png")).convert("RGB"))
OURS = np.asarray(Image.open(os.path.join(HERE, "paper", STEM + "_ours.png")).convert("RGB"))
H, W = SURF.shape[:2]
FG = rm.paper_mask(SURF)
PB, PO, _ = rm.color_masks(SURF)
_, _, RED = rm.color_masks(OURS)
RED_C = RED.copy()
RED_C[:150] = False
RED_C[230:] = False
DOT = np.array(np.nonzero(RED_C)[::-1], float).mean(1)          # (x, y) of the isolated point
RED_L = RED & ~RED_C
DT_RED = ndimage.distance_transform_edt(~RED_L)
RY, RX = np.nonzero(RED_L)
NAMES = ["el", "az", "roll", "f", "dist", "cx", "cy", "odx", "ody", "osc",
         "r2", "r4", "r6", "A", "b0", "b2", "c2", "So", "Lb", "b4", "c4"]


def R_of(s, sh):
    r2, r4, r6 = sh[0:3]
    u = s / sh[8]
    return 1.0 + r2 * u ** 2 + r4 * u ** 4 + r6 * u ** 6        # r0 = 1 sets the unit


def orange_fns(s, sh):
    A, b0, b2, c2 = sh[3:7]
    b4, c4 = sh[9:11]
    u = s / sh[8]
    return 1.0 - A + c2 * u ** 2 + c4 * u ** 4, A + 0 * u, b0 + b2 * u ** 2 + b4 * u ** 4   # c, A, Bz


def grids(sh, n=200, m=120):
    s = sh[8] * np.linspace(-1, 1, n)[:, None]
    ph = np.linspace(-np.pi / 2, np.pi / 2, m)[None, :]
    R = R_of(s, sh)
    blue = np.stack([R * np.cos(ph), s + 0 * ph, R * np.sin(ph)], -1)
    So = sh[7] * sh[8]
    so = np.linspace(-So, So, n)[:, None]
    c, A, Bz = orange_fns(so, sh)
    orange = np.stack([c + A * np.cos(ph), so + 0 * ph, Bz * np.sin(ph)], -1)
    return blue, orange


def crossing_w(s, sh):
    """Roots w = cos(phi) in [-1, 1) of the circle/ellipse cross-section equation, the non-contact one."""
    R = R_of(s, sh)
    c, A, Bz = orange_fns(s, sh)
    a2 = R * R / (A * A) - R * R / (Bz * Bz)
    a1 = -2 * R * c / (A * A)
    a0 = c * c / (A * A) + R * R / (Bz * Bz) - 1
    disc = a1 * a1 - 4 * a2 * a0
    sq = np.sqrt(np.clip(disc, 0, None))
    w1, w2 = (-a1 - sq) / (2 * a2), (-a1 + sq) / (2 * a2)
    return np.minimum(w1, w2), np.maximum(w1, w2), disc


def curves(sh, n=200):
    So = sh[7] * sh[8]
    s = np.linspace(-So, So, n)
    w, _, disc = crossing_w(s, sh)
    if (disc < 0).any() or (w < -1).any() or (w >= 1).any():
        return None
    R = R_of(s, sh)
    z = R * np.sqrt(1 - w * w)
    return [np.stack([R * w, s, sg * z], 1) for sg in (1, -1)]


def zsplat(P, cam, Wd=W, Hd=H):
    x, y, d = rm.project(P.reshape(-1, 3), cam, Wd, Hd)
    ok = (x >= 0) & (x < Wd) & (y >= 0) & (y < Hd)
    ix, iy, d = x[ok].astype(int), y[ok].astype(int), d[ok]
    dep = np.full((Hd, Wd), np.inf)
    np.minimum.at(dep, (iy, ix), d)
    m = np.isfinite(dep)
    m2 = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=2))
    # fill depth holes with the local minimum of the neighbours
    dep2 = ndimage.grey_erosion(np.where(m, dep, np.inf), size=5)
    dep = np.where(m, dep, dep2)
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
    return c


def loss(p, verbose=False):
    sh = p[10:]
    cam = cam_of(p)
    if p[4] < 2 or sh[3] <= 0.05 or sh[4] <= 0.3:
        return 10.0
    # isolated contact at (1, 0, 0): the orange must stay inside the blue near it in both directions,
    # along s (r2 - c2 > 0) and across (A / Bz^2 > 1 / r0), with a margin
    if sh[0] - sh[6] < 0.05 or sh[3] / sh[4] ** 2 < 1.05:
        return 10.0
    if (R_of(np.linspace(0, 1, 51), sh) <= 0.2).any():
        return 10.0
    C = curves(sh)
    if C is None:
        return 10.0
    blue, orange = grids(sh)
    mb, db = zsplat(blue, cam)
    mo, do = zsplat(orange, cam)
    iou = rm.mask_iou(mb | mo, FG)
    vis_b = mb & ~(mo & (do < db - 0.01))   # tolerance: the sheets nearly coincide around the contact
    iou_b = rm.mask_iou(vis_b, PB)
    oc = ours_cam(p)
    pts = np.vstack([np.stack(rm.project(Ci, oc, W, H)[:2], 1) for Ci in C])
    ok = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
    if ok.mean() < 0.99:
        return 10.0
    d1 = DT_RED[pts[ok, 1].astype(int), pts[ok, 0].astype(int)].mean()
    d2 = np.min(np.hypot(RX[:, None] - pts[None, ::3, 0], RY[:, None] - pts[None, ::3, 1]), 1).mean()
    dx, dy, _ = rm.project(np.array([1.0, 0, 0]), oc, W, H)
    dd = float(np.hypot(dx - DOT[0], dy - DOT[1]))
    if verbose:
        print("IoU %.4f  blue IoU %.4f  curve chamfer %.2f / %.2f px  dot %.2f px" % (iou, iou_b, d1, d2, dd))
    return (1 - iou) + (1 - iou_b) + 0.1 * (d1 + d2) + 0.05 * dd


BOUNDS = [(-15, 15), (-15, 15), (-5, 5), (200, 3000), (2.5, 20), (200, 230), (175, 205), (-10, 10), (-10, 10),
          (0.9, 1.1),
          (-1.0, 2.0), (-2.0, 4.0), (-2.0, 6.0), (0.5, 6.0), (0.8, 2.0), (-1.0, 0.5), (-2.0, 0.5), (0.4, 0.98), (1.2, 2.2), (-2.0, 1.0), (-3.0, 3.0)]

if __name__ == "__main__":
    x0 = np.array(json.loads(sys.argv[2])) if len(sys.argv) > 2 else None
    # optional: comma-separated shape names held at zero (lower-degree variants)
    for name in (sys.argv[3].split(",") if len(sys.argv) > 3 else []):
        BOUNDS[NAMES.index(name)] = (-1e-12, 1e-12)
        if x0 is not None:
            x0[NAMES.index(name)] = 0.0
    r = optimize.differential_evolution(loss, BOUNDS, popsize=16, maxiter=int(sys.argv[1]) if len(sys.argv) > 1 else 150,
                                        tol=1e-8, seed=3, polish=False, updating="deferred", workers=8, x0=x0)
    q = optimize.minimize(loss, r.x, method="Nelder-Mead",
                          options={"maxiter": 4000, "adaptive": True, "xatol": 1e-5, "fatol": 1e-8})
    p = q.x if q.fun < r.fun else r.x
    loss(p, verbose=True)
    print(json.dumps({"camera": cam_of(p), "ours_camera": ours_cam(p), "shape": dict(zip(NAMES[10:], map(float, p[10:]))),
                      "loss": float(min(q.fun, r.fun)), "p": p.tolist()}))
