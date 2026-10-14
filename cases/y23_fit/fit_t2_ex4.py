"""Fit camera + shape of Table 2 ex.4 ("a tangential curve") to the paper's Surfaces panel and result panel.

Model: y23_t2_ex4_tangential_curve_saddle.py. Objective:
  1 - mean IoU over (union, blue-dominant, orange-dominant) masks of the Surfaces panel
  + 0.02 x mean chamfer distance (px) between our curve, seen from the result-panel camera, and the
    red pixels of the result ("Ours") panel
  + a penalty keeping the construction exact (h_s >= eta > 0, |m| < 2 k eta).
Run: .venv/bin/python -I cases/y23_fit/fit_t2_ex4.py [iters] [start-json]
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
from fit_tools_t2_ex456 import coord_search  # noqa: E402
import y23_t2_ex4_tangential_curve_saddle as M  # noqa: E402

PAPER = np.asarray(Image.open(os.path.join(HERE, "paper", M.CID + "_surfaces.png")).convert("RGB"))
H, W = PAPER.shape[:2]
FG = rm.paper_mask(PAPER)
PB, PO, PR = rm.color_masks(PAPER)
RES = np.asarray(Image.open(os.path.join(HERE, "paper", M.CID + "_result.png")).convert("RGB"))
RH, RW = RES.shape[:2]
_, _, RRED = rm.color_masks(RES)
RDT = ndimage.distance_transform_edt(~RRED)
RY, RX = np.nonzero(RRED)

HK = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0), (3, 1), (0, 1), (0, 2)]
KEYS = ["k", "to0", "to1", "tb0", "tb1", "m0", "m1"] + ["H%d%d" % ij for ij in HK]
CKEYS = ["el", "az", "roll", "scale", "cx", "cy"]


def to_params(d):
    p = {k: d[k] for k in ("k", "to0", "to1", "tb0", "tb1")}
    Hm = np.zeros((4, 4))
    for i, j in HK:
        Hm[i, j] = d["H%d%d" % (i, j)]
    p["H"] = Hm.tolist()
    p["m"] = [d["m0"], d["m1"], 0.0, 0.0]
    return p


def from_params(p):
    d = {k: p[k] for k in ("k", "to0", "to1", "tb0", "tb1")}
    Hm = np.asarray(p["H"], float)
    for i, j in HK:
        d["H%d%d" % (i, j)] = float(Hm[i, j])
    d["m0"], d["m1"] = p["m"][0], p["m"][1]
    return d


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


def chamfer(p, rcam):
    P = M.curve(p, 800)
    x, y, _ = rm.project(P, rcam, RW, RH)
    inside = (x >= 0) & (x < RW) & (y >= 0) & (y < RH)
    xi = np.clip(np.round(x).astype(int), 0, RW - 1)
    yi = np.clip(np.round(y).astype(int), 0, RH - 1)
    d1 = np.where(inside, RDT[yi, xi], 60.0).mean()
    m = np.zeros((RH, RW), bool)
    m[yi[inside], xi[inside]] = True
    if not m.any():
        return 60.0
    d2 = ndimage.distance_transform_edt(~m)[RY, RX].mean()
    return 0.5 * (d1 + d2)


def unpack(v, rcam=None):
    d = dict(zip(KEYS, v[:len(KEYS)]))
    p = to_params(d)
    cam = dict(M.CAMERA)
    cam.update(zip(CKEYS, v[len(KEYS):len(KEYS) + 6]))
    return p, cam


def penalty(p):
    e = M.eta(p)
    mm = float(np.abs(M.mval(p, np.linspace(-1, 1, 201))).max())
    return 20 * max(0.0, 0.03 - e) + 20 * max(0.0, mm - 1.6 * p["k"] * max(e, 0.0))


def objective(v, rcam):
    p, cam = unpack(v)
    try:
        if p["to0"] >= -0.1 or p["tb0"] >= -0.1 or p["to1"] <= 0.1 or p["tb1"] <= 0.1 or p["k"] <= 0:
            return 2.0
        iu, ib, io = score(p, cam)
        ch = chamfer(p, rcam)
    except Exception:
        return 2.0
    return 1 - (iu + ib + io) / 3 + 0.01 * ch + penalty(p)


def fit_rcam(p, r0):
    keys = ["el", "az", "roll", "scale", "cx", "cy"]
    best = None
    for el0 in (r0["el"], 60.0, -60.0):
        for az0 in (r0["az"], r0["az"] + 60, r0["az"] - 60, 0.0, 180.0):
            def f(v):
                c = dict(r0)
                c.update(zip(keys, v))
                return chamfer(p, c)
            rr = optimize.minimize(f, [el0, az0, r0["roll"], r0["scale"], r0["cx"], r0["cy"]], method="Nelder-Mead",
                                   options=dict(maxiter=600, adaptive=True))
            if best is None or rr.fun < best.fun:
                best = rr
    c = dict(r0)
    c.update(zip(keys, best.x))
    return c, float(best.fun)


STEPS = dict(k=0.05, to0=0.05, to1=0.05, tb0=0.05, tb1=0.05, m0=0.01, m1=0.01, el=3.0, az=3.0, roll=1.0,
             scale=3.0, cx=3.0, cy=3.0)


def main():
    outer = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    d0 = from_params(M.PARAMS)
    c0, r0 = dict(M.CAMERA), dict(M.RESULT_CAMERA)
    if len(sys.argv) > 2:
        d = json.loads(sys.argv[2])
        d0.update(d.get("p", {}))
        c0.update(d.get("cam", {}))
        r0.update(d.get("rcam", {}))
    v = np.array([d0[k] for k in KEYS] + [c0[k] for k in CKEYS], float)
    steps = [STEPS.get(k, 0.02) for k in KEYS + CKEYS]
    rcam = r0
    for it in range(outer):
        p, cam = unpack(v)
        rcam, ch = fit_rcam(p, rcam)
        print("outer", it, "rcam chamfer", ch, "objective", objective(v, rcam), flush=True)
        v, fv = coord_search(lambda x: objective(x, rcam), v, steps, cycles=5)
    p, cam = unpack(v)
    rcam, ch = fit_rcam(p, rcam)
    score(p, cam, n=81, verbose=True)
    print("chamfer px", ch, "eta", M.eta(p), "penalty", penalty(p))
    rd = lambda d: {k: (round(float(x), 5) if not isinstance(x, (str, list)) else x) for k, x in d.items()}
    print(json.dumps({"p": rd(from_params(p)), "cam": rd(cam), "rcam": rd(rcam)}))


if __name__ == "__main__":
    main()
