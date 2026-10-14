"""Fit camera(s) + shape of Table 6 example (5) to the paper panels.

Model: see y23_t6_ex5_four_boundary_curves.py (arched sheet with a bilinear footprint and nine
free heights; the dome is the sheet with its centre control point raised). Objective: silhouette
IoU against the Surfaces panel, chamfer of the four exact boundary curves against the red
outline of the Surfaces panel (camera 1) and of the "Ours" panel (camera 2).
Run: .venv/bin/python -I cases/y23_fit/fit_t6_ex5.py
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t6_ex456_common as C  # noqa: E402
import y23_t6_ex5_four_boundary_curves as M  # noqa: E402

STEM = M.CID
SURF = C.load_panel(STEM, "surfaces")
OURS = C.load_panel(STEM, "ours")
H1, W1 = SURF.shape[:2]
H2, W2 = OURS.shape[:2]
FG = C.rm.paper_mask(SURF)
RED1 = C.red_mask(SURF)
RED2 = C.red_mask(OURS)
CAM_KEYS = ["el", "az", "roll", "scale", "cx", "cy"]
OFFSET = "orange"   # which sheet is S0 + mu n
PB = C.rm.color_masks(SURF)[0]  # blue-only pixels of the paper (above the upper arches)


def unpack(p):
    corners = np.asarray(p[:8]).reshape(4, 2).tolist()
    z = np.asarray(p[8:17]).reshape(3, 3).tolist()
    s = dict(corners=corners, z=z, c=float(p[17]))
    if len(p) > 30:
        s["n"] = [float(p[30]), float(p[31]), float(p[32])]
    if len(p) > 33:
        s["dxy"] = [float(p[33]), float(p[34])]
        s["offset"] = OFFSET
    c1 = dict(zip(CAM_KEYS, p[18:24]), proj="ortho")
    c2 = dict(zip(CAM_KEYS, p[24:30]), proj="ortho")
    return s, c1, c2


def pack(s, c1, c2):
    return np.concatenate([np.ravel(s["corners"]), np.ravel(s["z"]), [s["c"]],
                           [c1[k] for k in CAM_KEYS], [c2[k] for k in CAM_KEYS], s.get("n", (0.0, 0.0, 1.0)),
                           s.get("dxy", (0.0, 0.0))])


_UV = np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 1, 21), indexing="ij")


def valid(s):
    """Cheap screen for the fit (sampled Jacobian sign); main() certifies the final shape."""
    det, _ = M.footprint_jacobian(s)
    J = C.g.p2_eval(det, *_UV)
    return s["c"] > 0.01 and ((J > 0.05).all() or (J < -0.05).all())


def loss(p, verbose=False, use=(1, 1, 1)):
    s, c1, c2 = unpack(p)
    if not valid(s):
        return 10.0
    Po, Pb = M.surfaces(s)
    curves = M.curves(s, 100)
    t = 0.0
    # inside the orange outline the paper draws the orange in front of the dome
    Lo = C.rm._layer(C.grid3(Po, 25), c1, W1, H1, 1, C.rm.ORANGE, "surfaces")
    Lb = C.rm._layer(C.grid3(Pb, 25), c1, W1, H1, 1, C.rm.BLUE, "surfaces")
    both = (Lo[1] > 0.5) & (Lb[1] > 0.5)
    front = float((both & (Lb[2] < Lo[2])).sum()) / max(1.0, float((Lo[1] > 0.5).sum()))
    t += 2.0 * front
    # the blue seen outside the orange (paper: only above the upper arches)
    bo = (Lb[1] > 0.5) & ~(Lo[1] > 0.5)
    iou_b = C.rm.mask_iou(bo, PB)
    t += use[0] * (1 - iou_b)
    if use[0] or verbose:
        m = C.grid_mask(C.grid3(Po, 25), c1, W1, H1) | C.grid_mask(C.grid3(Pb, 25), c1, W1, H1)
        iou = C.rm.mask_iou(m, FG)
        t += use[0] * (1 - iou)
    if use[1] or verbose:
        ch1 = C.chamfer([C.curve_px(cv, c1, W1, H1) for cv in curves], RED1)
        t += use[1] * 0.05 * ch1
    if use[2] or verbose:
        ch2 = C.chamfer([C.curve_px(cv, c2, W2, H2) for cv in curves], RED2)
        t += use[2] * 0.05 * ch2
    if verbose:
        print("IoU %.4f  chamfer surfaces %.2f px  ours %.2f px  dome-in-front %.3f  blue-only IoU %.3f" % (iou, ch1, ch2, front, iou_b))
    return t


def cam_search(s, red, W, H):
    """Coarse orthographic camera search on the boundary-curve chamfer only."""
    cur = M.curves(s, 60)
    ry, rx = np.nonzero(red)
    res = []
    for el, az, roll in itertools.product(range(-80, 81, 10), range(-180, 180, 15), range(-180, 180, 15)):
        cam = dict(proj="ortho", el=el, az=az, roll=roll, scale=1.0, cx=0, cy=0)
        P = np.vstack([C.curve_px(c, cam, 1, 1) for c in cur])
        sc = np.sqrt(np.ptp(rx) * np.ptp(ry) / max(np.ptp(P[:, 0]) * np.ptp(P[:, 1]), 1e-9))
        if not np.isfinite(sc) or sc > 5000:
            continue
        cam.update(scale=sc, cx=rx.mean() - sc * P[:, 0].mean(), cy=ry.mean() - sc * P[:, 1].mean())
        res.append((C.chamfer([C.curve_px(c, cam, W, H) for c in cur], red), cam))
    res.sort(key=lambda r: r[0])
    return res
