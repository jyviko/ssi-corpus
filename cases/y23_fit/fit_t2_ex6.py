"""Fit camera + shape of Table 2 ex.6 ("a loop and a point") to the paper's Surfaces panel.

Model: y23_t2_ex6_loop_and_point.py. Objective: 1 - mean IoU over (union, blue-dominant, orange-dominant)
masks of the Surfaces panel + 0.01 x distance (px) of the projected loop centre and isolated point from the
paper's (red pixels), with penalties keeping the construction exact. Cyclic coordinate search.
Run: .venv/bin/python -I cases/y23_fit/fit_t2_ex6.py [cycles] [start-json]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_match as rm  # noqa: E402
import fit_tools_t2_ex456 as T  # noqa: E402
import y23_t2_ex6_loop_and_point as M  # noqa: E402

PAPER = T.paper(M.CID, "surfaces")
H, W = PAPER.shape[:2]
_, _, RED = rm.color_masks(PAPER)
# paper features (panel px): loop = the big red component, point = the small one
from scipy import ndimage  # noqa: E402
lab, nl = ndimage.label(RED)
sizes = ndimage.sum(RED, lab, range(1, nl + 1))
PT_LAB = int(np.argmin(sizes)) + 1  # the dot is the smallest red component; the loop may be split in two
LOOP_MASK = (lab > 0) & (lab != PT_LAB)
LOOP_PX = np.argwhere(LOOP_MASK)[:, ::-1].astype(float)
POINT_PX = np.argwhere(lab == PT_LAB)[:, ::-1].mean(0)
LOOP_DT = ndimage.distance_transform_edt(~LOOP_MASK)

FKEYS = ["f%d%d" % (i, j) for i in range(3) for j in range(3) if (i, j) != (0, 0)]
KEYS = FKEYS + ["bu0", "bu1", "bw0", "bw1", "ou0", "ou1", "ow0", "ow1", "p", "q", "l", "kap", "du", "dw", "mu", "nth",
                "nph"]
CKEYS = ["el", "az", "roll", "scale", "cx", "cy"]


def to_params(d):
    p = {k: d[k] for k in KEYS if not k.startswith("f")}
    F = np.zeros((3, 3))
    for k in FKEYS:
        F[int(k[1]), int(k[2])] = d[k]
    p["f"] = F.tolist()
    return p


def from_params(p):
    d = {k: p[k] for k in KEYS if not k.startswith("f")}
    F = np.asarray(p["f"], float)
    for k in FKEYS:
        d[k] = float(F[int(k[1]), int(k[2])])
    return d


def unpack(v):
    p = to_params(dict(zip(KEYS, v[:len(KEYS)])))
    cam = dict(M.CAMERA)
    cam.update(zip(CKEYS, v[len(KEYS):]))
    return p, cam


def features(p, cam):
    L = M.loop(p, 400)
    x, y, _ = rm.project(L, cam, W, H)
    xi = np.clip(np.round(x).astype(int), 0, W - 1)
    yi = np.clip(np.round(y).astype(int), 0, H - 1)
    m = np.zeros((H, W), bool)
    m[yi, xi] = True
    back = ndimage.distance_transform_edt(~m)[LOOP_PX[:, 1].astype(int), LOOP_PX[:, 0].astype(int)]
    dloop = 0.5 * (float(LOOP_DT[yi, xi].mean()) + float(back.mean()))
    px, py, _ = rm.project(M.point(p), cam, W, H)
    dpt = float(np.hypot(px - POINT_PX[0], py - POINT_PX[1]))
    return dloop, dpt


APEX_PX = np.array([150.0, 38.0])  # the orange's pointed upper tip in the panel


def apex_term(p, cam):
    """Mean distance (px) of the orange's far edge u = ou1 from the paper's single pointed tip: the figure's
    upper lobe ends in one point, i.e. that edge is seen nearly end-on."""
    w = np.linspace(p["ow0"], p["ow1"], 9)
    E = M.orange_pt(p, p["ou1"] + 0 * w, w)
    x, y, _ = rm.project(E, cam, W, H)
    return float(np.mean(np.hypot(x - APEX_PX[0], y - APEX_PX[1])))


def objective(v, verbose=False):
    p, cam = unpack(v)
    if not M.valid(p):
        return 2.0
    A, B = M.grids(p, 41)
    rgb, masks = rm.render([A, B], cam, W, H, ss=1, n=41, alphas=M.ALPHAS)
    iu = T.ious(rm, PAPER, rgb, masks)
    dl, dp = features(p, cam)
    out = 0.0
    for G in (A, B):  # parts of the surfaces outside the panel are not in the IoU: penalise them
        x, y, _ = rm.project(G.reshape(-1, 3), cam, W, H)
        out += float(np.mean((x < 0) | (x >= W) | (y < 0) | (y >= H)))
    ap = apex_term(p, cam)
    if verbose:
        print("outside", round(out, 4), "apex px", round(ap, 2))
    if verbose:
        print("IoU", iu, "loop px", round(dl, 2), "point px", round(dp, 2))
    return 1 - float(np.mean(iu)) + 0.01 * dl + 0.005 * dp + APEX_W * ap + out + EL_PEN * max(0.0, abs(cam["el"] - 35.0) - 15.0)


EL_PEN = float(os.environ.get("EX6_EL_PEN", "0"))
APEX_W = float(os.environ.get("EX6_APEX_W", "0.003"))


STEPS = dict(bu0=0.05, bu1=0.05, bw0=0.05, bw1=0.05, ou0=0.05, ou1=0.05, ow0=0.05, ow1=0.05, p=0.05, q=0.05,
             l=0.05, kap=0.2, du=0.03, dw=0.03, mu=0.005, nth=3.0, nph=10.0, el=3.0, az=3.0, roll=2.0, scale=3.0, cx=3.0, cy=3.0)


def main():
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    d0 = from_params(M.PARAMS)
    c0 = dict(M.CAMERA)
    if len(sys.argv) > 2:
        d = json.loads(sys.argv[2])
        d0.update(d.get("p", {}))
        c0.update(d.get("cam", {}))
    v = np.array([d0[k] for k in KEYS] + [c0[k] for k in CKEYS], float)
    print("start", objective(v, True), flush=True)
    v, fv = T.coord_search(objective, v, [STEPS.get(k, 0.03) for k in KEYS + CKEYS], cycles=cycles)
    p, cam = unpack(v)
    objective(v, True)
    rd = lambda d: {k: (round(float(x), 5) if not isinstance(x, (str, list)) else x) for k, x in d.items()}
    print(json.dumps({"p": rd(from_params(p)), "cam": rd(cam)}))


if __name__ == "__main__":
    main()
