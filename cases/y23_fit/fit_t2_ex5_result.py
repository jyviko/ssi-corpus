"""Fit the camera of Table 2 ex.5's result panel ("Ours" column) to our exact loops (chamfer distance between
the projected truth and the paper's red pixels). Run: .venv/bin/python -I cases/y23_fit/fit_t2_ex5_result.py"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_match as rm  # noqa: E402
import y23_t2_ex5_two_loops as M  # noqa: E402

RES = np.asarray(Image.open(os.path.join(HERE, "paper", M.CID + "_result.png")).convert("RGB"))
H, W = RES.shape[:2]
_, _, RED = rm.color_masks(RES)
DT = ndimage.distance_transform_edt(~RED)
RY, RX = np.nonzero(RED)


def loops(p, n=1200):
    return M.truth_curves(p, n)


def chamfer(curves, cam):
    P = np.vstack(curves)
    x, y, _ = rm.project(P, cam, W, H)
    xi = np.clip(np.round(x).astype(int), 0, W - 1)
    yi = np.clip(np.round(y).astype(int), 0, H - 1)
    inside = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    d1 = np.where(inside, DT[yi, xi], 50.0).mean()
    m = np.zeros((H, W), bool)
    m[yi[inside], xi[inside]] = True
    d2 = ndimage.distance_transform_edt(~m)[RY, RX].mean()
    return 0.5 * (d1 + d2)


def fit(p, cam0, iters=1500):
    curves = loops(p)
    keys = ["el", "az", "roll", "scale", "cx", "cy"]

    def f(v):
        cam = dict(cam0)
        cam.update(zip(keys, v))
        return chamfer(curves, cam)
    best = None
    for el0 in (cam0["el"], 50.0, 80.0):
        for az0 in (cam0["az"], cam0["az"] + 90, cam0["az"] - 90, cam0["az"] + 180):
            v0 = [el0, az0, cam0.get("roll", 0.0), cam0["scale"], W / 2, H / 2]
            r = optimize.minimize(f, v0, method="Nelder-Mead", options=dict(maxiter=iters, adaptive=True))
            if best is None or r.fun < best.fun:
                best = r
    cam = dict(cam0)
    cam.update(zip(keys, best.x))
    return cam, float(best.fun)


if __name__ == "__main__":
    cam0 = dict(M.CAMERA)
    cam0["scale"] = 150.0
    cam, d = fit(M.PARAMS, cam0)
    print("mean chamfer px", d)
    print(json.dumps({k: (round(float(v), 4) if not isinstance(v, (str, list)) else v) for k, v in cam.items()}))
