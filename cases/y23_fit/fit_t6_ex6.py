"""Fit camera(s) + shape of Table 6 example (6) to the paper panels.

Model: see y23_t6_ex6_four_boundary_curves.py (parabolic-cylinder trough + folded dome
whose four edges lie on the trough). Objective: silhouette IoU against the Surfaces panel,
chamfer of the four exact intersection curves against the red pixels of the Surfaces panel
(camera 1) and of the "Ours" panel (camera 2, the result rows use their own view).
Run: .venv/bin/python -I cases/y23_fit/fit_t6_ex6.py  -> prints parameters as JSON.
"""
import json
import os
import sys

import numpy as np
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t6_ex456_common as C  # noqa: E402
import y23_t6_ex6_four_boundary_curves as M  # noqa: E402

STEM = M.CID
SURF = C.load_panel(STEM, "surfaces")
OURS = C.load_panel(STEM, "ours")
H1, W1 = SURF.shape[:2]
H2, W2 = OURS.shape[:2]
FG = C.rm.paper_mask(SURF)
RED1 = C.red_mask(SURF)
RED2 = C.red_mask(OURS)

SHAPE_KEYS = ["a", "L", "h", "k1", "kf", "kt", "c", "b", "dx"]
CAM_KEYS = ["el", "az", "roll", "scale", "cx", "cy"]


def unpack(p):
    s = dict(zip(SHAPE_KEYS[:7], p[:7]))
    c1 = dict(zip(CAM_KEYS, p[7:13]), proj="ortho")
    c2 = dict(zip(CAM_KEYS, p[13:19]), proj="ortho")
    if len(p) > 19:
        s.update(b=p[19], dx=p[20])
    return s, c1, c2


def loss(p, verbose=False, use=(1, 1, 1)):
    s, c1, c2 = unpack(p)
    if not (0.05 < s["a"] < 3 and 0.1 < s["L"] < 4 and -3 < s["h"] < 3 and -0.9 < s["k1"] < 3
            and 0.05 < s["kf"] < 1 + s["k1"] and -1.9 < s["kt"] < 6 and 0.02 < s["c"] < 3):
        return 10.0
    Po, Pb = M.surfaces(s)
    Go, Gb = C.grid3(Po, 25), C.grid3(Pb, 25)
    curves = M.curves(s, 120)
    m = C.grid_mask(Go, c1, W1, H1) | C.grid_mask(Gb, c1, W1, H1)
    iou = C.rm.mask_iou(m, FG)
    px1 = [C.curve_px(cv, c1, W1, H1) for cv in curves]
    ch1 = C.chamfer_split(px1[1:], px1, RED1)  # the far arc is mostly hidden behind the dome
    ch2 = C.chamfer([C.curve_px(cv, c2, W2, H2) for cv in curves], RED2)
    if verbose:
        print("IoU %.4f  chamfer surfaces %.2f px  ours %.2f px" % (iou, ch1, ch2))
    return use[0] * (1 - iou) + use[1] * 0.05 * ch1 + use[2] * 0.02 * ch2


def run(p0, rounds=3):
    best = (loss(p0), np.array(p0, float))
    for it in range(rounds):
        r = optimize.minimize(loss, best[1], method="Nelder-Mead",
                              options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-6, "adaptive": True})
        if r.fun < best[0]:
            best = (r.fun, r.x)
        print(it, round(best[0], 5), np.round(best[1], 4).tolist(), flush=True)
    return best


if __name__ == "__main__":
    p0 = json.loads(sys.argv[1]) if len(sys.argv) > 1 else (
        [0.4, 1.0, -0.3, 0.6, 0.8, 0.5, 0.5] + [30, -90, 0, 228, 277, 200] + [30, -60, -25, 228, 340, 230])
    best = run(np.array(p0, float))
    loss(best[1], verbose=True)
    s, c1, c2 = unpack(best[1])
    print(json.dumps({"shape": s, "cam_surfaces": c1, "cam_ours": c2}))
