"""Fit the shape of Table 6 example (4) to the paper's Surfaces panel (face-on view).

Model: see y23_t6_ex4_tangential_point.py (free outlines; the depth construction does not
enter the silhouettes). View coordinates: x right, y up, pixels / 100, so the
camera is fixed (orthographic, looking down -z, 100 px per unit, origin at the panel corner).
Objective: (1 - IoU) of the union silhouette, (1 - IoU) of the orange region, and the distance of
the projected contact point from the paper's red dot.
Run: .venv/bin/python -I cases/y23_fit/fit_t6_ex4.py
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t6_ex456_common as C  # noqa: E402
import y23_t6_ex4_tangential_point as M  # noqa: E402

STEM = M.CID
SURF = C.load_panel(STEM, "surfaces")
H, W = SURF.shape[:2]
FG = C.rm.paper_mask(SURF)
_, ORANGE, _ = C.rm.color_masks(SURF)
ORANGE = ndimage.binary_opening(ndimage.binary_fill_holes(ORANGE), iterations=2)
_r = C.red_mask(SURF)
_lab, _nl = ndimage.label(_r)
DOT = np.array(ndimage.center_of_mass(_r, _lab, 1 + int(np.argmax(ndimage.sum(_r, _lab, range(1, _nl + 1))))))[::-1]
CAM = M.CAMERA


def unpack(p, s0=M.SHAPE):
    s = json.loads(json.dumps(s0))
    s["xy"] = np.asarray(p[:18]).reshape(3, 3, 2).tolist()
    s["oxy"] = np.asarray(p[18:36]).reshape(3, 3, 2).tolist()
    s["st0"] = list(map(float, p[36:38]))
    return s


def pack(s):
    return np.concatenate([np.ravel(s["xy"]), np.ravel(s["oxy"]), s["st0"]])


def loss(p, verbose=False):
    s = unpack(p)
    if not (0.05 < s["st0"][0] < 0.95 and 0.05 < s["st0"][1] < 0.95):
        return 10.0
    Pb, Po = M.surfaces(s)
    mb = C.grid_mask(C.grid3(Pb, 25), CAM, W, H)
    mo = C.grid_mask(C.grid3(Po, 25), CAM, W, H)
    i1 = C.rm.mask_iou(mb | mo, FG)
    i2 = C.rm.mask_iou(mo, ORANGE)
    P = np.array([C.peval(c, *s["st0"]) for c in Po])
    px = np.array(C.rm.project(P, CAM, W, H)[:2])
    dd = float(np.linalg.norm(px - DOT))
    if verbose:
        print("IoU union %.4f  IoU orange %.4f  dot offset %.2f px" % (i1, i2, dd))
    return (1 - i1) + (1 - i2) + 0.01 * dd


if __name__ == "__main__":
    p = pack(M.SHAPE)
    print(DOT)
    loss(p, True)
    for it in range(3):
        r = optimize.minimize(loss, p, method="Nelder-Mead", options={"maxiter": 4000, "adaptive": True})
        p = r.x
        print(it, r.fun, flush=True)
    loss(p, True)
    print(json.dumps(unpack(p)))
