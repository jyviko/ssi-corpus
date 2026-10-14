"""Fit camera + spindle profile of Fig. 1(a) to the paper panels (600 dpi crops).

Model (see y23_fig1a_two_loops_cross.py): two congruent tubes, blue about the x
axis and orange about the y axis (the blue turned 90 degrees about z), with
elliptic cross-sections of half-widths a(s) (in the xy-plane) and kappa * a(s)
(along z), where the even polynomial a(s) = (1 - s^2)(c0 + c2 s^2 + c4 s^4),
s in [-1, 1], has cone points at s = +-1 and a waist at s = 0. Their
intersection contains the planar loops x = +-y, z = +-kappa sqrt(a(x)^2 - x^2).

Objective: 1 - IoU of the projected union silhouette against the "Surfaces"
panel, plus a symmetric chamfer distance (pixels) between the projected loops
and the red pixels of the "Ours" panel, weighted 0.05 / px.

Run: .venv/bin/python -I cases/y23_fit/fit_y23_fig1a_two_loops_cross.py
Prints the fitted parameters as JSON (copied into y23_fig1a_two_loops_cross.py).
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

STEM = "y23_fig1a_two_loops_cross"
SURF = np.asarray(Image.open(os.path.join(HERE, "paper", STEM + "_surfaces.png")).convert("RGB"))
OURS = np.asarray(Image.open(os.path.join(HERE, "paper", STEM + "_ours.png")).convert("RGB"))
H, W = SURF.shape[:2]
FG = rm.paper_mask(SURF)
_, _, RED = rm.color_masks(OURS)
DT_RED = ndimage.distance_transform_edt(~RED)
RY, RX = np.nonzero(RED)
FG_EDGE = FG & ~ndimage.binary_erosion(FG)
DT_FG_EDGE = ndimage.distance_transform_edt(~FG_EDGE)


def tip_widths(m, tip, dists=(6, 12, 18, 24)):
    """Widths of mask m across the lobe at given distances from a tip pixel, measured
    perpendicular to the direction from the tip to the centroid of the lobe's last 40 px."""
    ys, xs = np.nonzero(m)
    near = np.hypot(xs - tip[0], ys - tip[1]) < 40
    if near.sum() < 5:
        return np.full(len(dists), 60.0)
    u = np.array([xs[near].mean() - tip[0], ys[near].mean() - tip[1]])
    u /= np.linalg.norm(u)
    v = np.array([-u[1], u[0]])
    ts = np.arange(-60, 60, 0.5)
    out = []
    for d in dists:
        P = np.asarray(tip, float)[None] + d * u[None] + ts[:, None] * v[None]
        ok = (P[:, 0] >= 0) & (P[:, 0] < m.shape[1]) & (P[:, 1] >= 0) & (P[:, 1] < m.shape[0])
        out.append(m[P[ok, 1].astype(int), P[ok, 0].astype(int)].sum() * 0.5)
    return np.array(out)


# unclipped lobe tips of the paper panel (blue -x, blue +x, orange -y); the orange +y tip touches the frame
PAPER_TIPS = [(119, 12), (326, 372), (15, 273)]
PAPER_TIP_W = [tip_widths(FG, t) for t in PAPER_TIPS]
NAMES = ["el", "az", "roll", "f", "dist", "cx", "cy", "c0", "c2", "c4", "kappa", "sigma", "c6", "beta", "gamma", "beta5"]


def profile(s, c0, c2, c4, c6=0.0):
    return (1 - s * s) * (c0 + c2 * s * s + c4 * s ** 4 + c6 * s ** 6)


def tubes(sh, n=220, m=160):
    c0, c2, c4, ka, sg, c6, be, ga, b5 = sh
    s = np.linspace(-1, 1, n)[:, None]
    th = np.linspace(0, 2 * np.pi, m)[None, :]
    a = profile(s, c0, c2, c4, c6)
    y = a * np.cos(th)
    A = np.stack([s + 0 * th, be * s ** 3 + b5 * s ** 5 + y, ga * s * s + ka * a * np.sin(th) + sg * s * y], -1)   # blue
    B = A[..., [1, 0, 2]]                                                # mirror in x = y (orange)
    return A, B


def xstar(sh, sgn=1):
    """Half-extent of the loop in the plane y = sgn x: a(x) = |sgn x - beta x^3 - beta5 x^5|."""
    c0, c2, c4, _, _, c6, be, _, b5 = sh
    return optimize.brentq(lambda x: profile(x, c0, c2, c4, c6) - abs(sgn * x - be * x ** 3 - b5 * x ** 5),
                           1e-12, 1 - 1e-12)


def loops(sh, n=400):
    c0, c2, c4, ka, sg, c6, be, ga, b5 = sh
    t = np.linspace(0, 2 * np.pi, n)
    out = []
    for sgn in (1, -1):
        # plane y = sgn x on the blue: be s^3 + b5 s^5 + a cos(th) = sgn s
        x = xstar(sh, sgn) * np.cos(t)
        yc = sgn * x - be * x ** 3 - b5 * x ** 5                   # a cos(th)
        z = ka * np.sign(np.sin(t)) * np.sqrt(np.clip(profile(x, c0, c2, c4, c6) ** 2 - yc * yc, 0, None))
        out.append(np.stack([x, sgn * x, ga * x * x + z + sg * x * yc], 1))
    return out


def splat(P, cam):
    x, y, _ = rm.project(P.reshape(-1, 3), cam, W, H)
    m = np.zeros((H, W), bool)
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    m[y[ok].astype(int), x[ok].astype(int)] = True
    m = ndimage.binary_closing(m, iterations=2)
    return ndimage.binary_fill_holes(m)


def cam_of(p):
    el, az, roll, f, dist, cx, cy = p[:7]
    return {"proj": "persp", "el": float(el), "az": float(az), "roll": float(roll), "scale": float(f),
            "dist": float(dist), "cx": float(cx), "cy": float(cy), "target": [0.0, 0.0, 0.0]}


def loss(p, verbose=False):
    cam = cam_of(p)
    sh = p[7:16]
    ss = np.linspace(0, 1, 201)
    a = profile(ss, sh[0], sh[1], sh[2], sh[5])
    if p[4] < 1.5 or (a[:-1] <= 0.01).any() or not 0.1 < sh[3] < 2:
        return 10.0
    try:
        L = loops(sh)
    except ValueError:
        return 10.0
    A, B = tubes(sh)
    m = splat(A, cam) | splat(B, cam)
    iou = rm.mask_iou(m, FG)
    pts = np.vstack([np.stack(rm.project(Li, cam, W, H)[:2], 1) for Li in L])
    ok = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
    if ok.mean() < 0.99:
        return 10.0
    d1 = DT_RED[pts[ok, 1].astype(int), pts[ok, 0].astype(int)].mean()
    d2 = np.min(np.hypot(RX[:, None] - pts[None, ::4, 0], RY[:, None] - pts[None, ::4, 1]), 1).mean()
    tips3 = np.array([[-1.0, 0, 0], [1.0, 0, 0], [0, -1.0, 0]])
    tips3[:2, 1] = [-(sh[6] + sh[8]), sh[6] + sh[8]]       # bent axis: y = beta s^3 + beta5 s^5
    tips3[:2, 2] = sh[7]                                   # z = gamma s^2
    tips3[2] = [tips3[0, 1], -1.0, sh[7]]                  # orange = mirror of the blue in x = y
    tx, ty, _ = rm.project(tips3, cam, W, H)
    tw = np.mean([np.abs(tip_widths(m, (x_, y_)) - w).mean() for x_, y_, w in zip(tx, ty, PAPER_TIP_W)])
    edge = m & ~ndimage.binary_erosion(m)
    e1 = DT_FG_EDGE[edge].mean()
    e2 = ndimage.distance_transform_edt(~edge)[FG_EDGE].mean()
    if verbose:
        print("IoU %.4f  loop chamfer %.2f / %.2f px  outline chamfer %.2f / %.2f px  tip width err %.2f px"
              % (iou, d1, d2, e1, e2, tw))
    return (1 - iou) + 0.05 * (d1 + d2) + 0.03 * (e1 + e2) + 0.01 * tw


BOUNDS = [(50, 75), (-25, 25), (-45, -15), (700, 4000), (3, 20), (195, 215), (160, 180),
          (0.12, 0.35), (-1.0, 2.0), (-2.0, 2.0), (0.3, 1.5), (-1.0, 1.0), (-3.0, 3.0), (-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6)]

if __name__ == "__main__":
    x0 = np.array(json.loads(sys.argv[2])) if len(sys.argv) > 2 else None
    # optional: comma-separated shape names held at zero (lower-degree variants)
    for name in (sys.argv[3].split(",") if len(sys.argv) > 3 else []):
        k = NAMES.index(name)
        BOUNDS[k] = (-1e-12, 1e-12)
        if x0 is not None:
            x0[k] = 0.0
    r = optimize.differential_evolution(loss, BOUNDS, popsize=16, maxiter=int(sys.argv[1]) if len(sys.argv) > 1 else 120,
                                        tol=1e-8, seed=2, polish=False, updating="deferred", workers=8, x0=x0)
    p = r.x
    fixed = [NAMES.index(n) for n in (sys.argv[3].split(",") if len(sys.argv) > 3 else [])]
    free = [k for k in range(len(p)) if k not in fixed]

    def loss_free(q):
        x = p.copy()
        x[free] = q
        return loss(x)
    r = optimize.minimize(loss_free, p[free], method="Nelder-Mead",
                          options={"maxiter": 3000, "adaptive": True, "xatol": 1e-5, "fatol": 1e-8})
    p[free] = r.x
    loss(p, verbose=True)
    print(json.dumps({"camera": cam_of(p), "shape": dict(zip(NAMES[7:], map(float, p[7:]))), "loss": float(r.fun),
                      "p": p.tolist()}))
