"""Fit camera + tube shape of Table 2 ex.5 ("two loops") to the paper's figures.

Model: y23_t2_ex5_two_loops.py. Objective (joint):
  1 - mean IoU over (union, blue-dominant, orange-dominant) masks of the Surfaces panel
  + 0.02 x mean chamfer distance (px) between our exact loops, seen from the result-panel camera, and the
    red pixels of the paper's result ("Ours") panel
  + Fig. 10 (right): the long loop spans s in [0.21, 0.79].
Both cameras are fitted together with the shape.
Run: .venv/bin/python -I cases/y23_fit/fit_t2_ex5.py [iters] [start-json]
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_match as rm  # noqa: E402
import y23_t2_ex5_two_loops as M  # noqa: E402
import fit_t2_ex5_result as R  # noqa: E402

PAPER = np.asarray(Image.open(os.path.join(HERE, "paper", M.CID + "_surfaces.png")).convert("RGB"))
H, W = PAPER.shape[:2]
FG = rm.paper_mask(PAPER)
PB, PO, PR = rm.color_masks(PAPER)

KEYS = ["x1", "x3", "z2", "z3", "a0", "a2", "b0", "b2", "c1", "k"]
CKEYS = ["el", "az", "roll", "scale", "cx", "cy"]


def score(p, cam, n=41, verbose=False):
    A, B = M.grids(p, n)
    rgb, masks = rm.render([A, B], cam, W, H, style="surfaces", ss=1, n=n, alphas=M.ALPHAS)
    ob, oo, _ = rm.color_masks(rgb)
    iu = rm.mask_iou(FG, masks[0] | masks[1])
    ib = rm.mask_iou(PB, ob)
    io = rm.mask_iou(PO, oo)
    if verbose:
        print("IoU union %.4f blue %.4f orange %.4f" % (iu, ib, io))
    return iu, ib, io


def unpack(v):
    p = dict(zip(KEYS, v[:len(KEYS)]))
    cam = dict(M.CAMERA)
    cam.update(zip(CKEYS, v[len(KEYS):len(KEYS) + 6]))
    rcam = dict(M.RESULT_CAMERA)
    rcam.update(zip(CKEYS, v[len(KEYS) + 6:]))
    return p, cam, rcam


def objective(v):
    p, cam, rcam = unpack(v)
    try:
        if not M.valid(p):
            return 2.0
        iu, ib, io = score(p, cam)
        s1 = M.loop_span(p)[0]
        ch = R.chamfer(M.truth_curves(p, 600), rcam)
    except Exception:
        return 2.0
    return 1 - (iu + ib + io) / 3 + 0.02 * ch + 2.0 * (s1 - 0.21) ** 2


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    p0, c0, r0 = dict(M.PARAMS), dict(M.CAMERA), dict(M.RESULT_CAMERA)
    if len(sys.argv) > 2:
        d = json.loads(sys.argv[2])
        p0.update(d.get("p", {}))
        c0.update(d.get("cam", {}))
        r0.update(d.get("rcam", {}))
    v0 = np.array([p0[k] for k in KEYS] + [c0[k] for k in CKEYS] + [r0[k] for k in CKEYS], float)
    print("start", objective(v0))
    r = optimize.minimize(objective, v0, method="Nelder-Mead",
                          options=dict(maxiter=iters, xatol=1e-4, fatol=1e-6, adaptive=True))
    p, cam, rcam = unpack(r.x)
    print("end", r.fun)
    score(p, cam, n=81, verbose=True)
    print("chamfer px", R.chamfer(M.truth_curves(p, 1200), rcam), "loop span", M.loop_span(p))
    rd = lambda d: {k: (round(float(v), 5) if not isinstance(v, (str, list)) else v) for k, v in d.items()}
    print(json.dumps({"p": rd(p), "cam": rd(cam), "rcam": rd(rcam)}))


if __name__ == "__main__":
    main()
