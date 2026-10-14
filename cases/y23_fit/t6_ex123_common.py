"""Shared fitting / comparison helpers for the Table 6 ex.1-3 refits (Yang, Jia & Yan 2023).

Used by y23_t6_ex1_tangent_line.py, y23_t6_ex2_tangential_arc.py and
y23_t6_ex3_two_crossing_curves.py. Rendering for the deliverables goes through the
shared render_match.render(); the camera/shape fits use the label map below (front-most
surface per pixel by dense point splatting), which is independent of the renderer's
internals.

Scores against a paper panel (cases/y23_fit/paper/<stem>_surfaces.png):
  * silhouette IoU: union of both surfaces vs the panel's non-background pixels;
  * blue / orange IoU: pixels where that surface is the front-most one vs the panel's
    blue- / orange-dominated pixels (the panel draws the front surface's colour);
  * chamfer: symmetric mean distance (px) between the silhouette boundaries and between
    the blue-region boundaries.
Red (curve) pixels of the panel are excluded from every score.
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "cases"))
sys.path.insert(0, os.path.join(ROOT, "scoring"))
import render_match as rm  # noqa: E402

PAPER = os.path.join(HERE, "paper")
COMPARE = os.path.join(HERE, "compare")
DATA = os.path.join(ROOT, "cases", "data")


def paper_panel(stem, kind="surfaces"):
    return np.asarray(Image.open(os.path.join(PAPER, "%s_%s.png" % (stem, kind))).convert("RGB"))


class Target:
    """Label maps of one paper panel plus the boundary distance fields used by the fit."""

    def __init__(self, rgb):
        self.rgb = rgb
        self.H, self.W = rgb.shape[:2]
        self.blue, self.orange, self.red = rm.color_masks(rgb)
        self.care = ~binary_dilation(self.red, iterations=3)
        self.fg = self.blue | self.orange
        self.pe = [edges(self.fg) & self.care, edges(self.blue) & self.care]
        self.dt = [distance_transform_edt(~e) for e in self.pe]

    def scores(self, fb, fo):
        care = ~self.red
        out = dict(iou_silhouette=rm.mask_iou((fb | fo) & care, self.fg),
                   iou_blue=rm.mask_iou(fb & care, self.blue),
                   iou_orange=rm.mask_iou(fo & care, self.orange))
        ch = []
        for k, m in enumerate((fb | fo, fb)):
            e = edges(m) & self.care
            if not e.any():
                ch.append(50.0)
                continue
            ch.append(0.5 * (self.dt[k][e].mean() + distance_transform_edt(~e)[self.pe[k]].mean()))
        out["chamfer_silhouette_px"], out["chamfer_blue_px"] = ch
        return out

    def curve_loss(self, curves, cam):
        """Symmetric mean distance (px) between projected curves and the panel's red pixels."""
        if not hasattr(self, "red_dt"):
            self.red_dt = distance_transform_edt(~self.red)
        return curve_pixels_loss(curves, cam, self.red_dt, self.red, self.W, self.H)

    def loss(self, fb, fo):
        s = self.scores(fb, fo)
        return s["chamfer_silhouette_px"] + s["chamfer_blue_px"] - 10 * (s["iou_blue"] + s["iou_orange"])


def edges(m):
    m = np.asarray(m, bool)
    e = np.zeros_like(m)
    e[:-1] |= m[:-1] != m[1:]
    e[:, :-1] |= m[:, :-1] != m[:, 1:]
    return e


def front_labels(grid_fns, cam, W, H, density=0.6, nmax=900):
    """Front-most surface per pixel. grid_fns: callables n -> (n, n, 3) point grids.
    Points are splatted at ~`density` px spacing (nmax caps the grid; above the cap each
    point also covers its right/lower neighbour pixels so no holes open)."""
    D = np.full((len(grid_fns), H, W), np.inf)
    for s, G in enumerate(grid_fns):
        x, y, _ = rm.project(G(24), cam, W, H)
        ext = max(np.ptp(x), np.ptp(y))
        n = int(max(60, ext / density))
        fat = n > nmax
        P = G(min(n, nmax)).reshape(-1, 3)
        x, y, z = rm.project(P, cam, W, H)
        offs = [(0, 0), (1, 0), (0, 1), (1, 1)] if fat else [(0, 0)]
        bx, by = (np.floor(x), np.floor(y)) if fat else (np.round(x), np.round(y))
        for dx, dy in offs:
            xi, yi = bx.astype(int) + dx, by.astype(int) + dy
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) & (z > 0)
            np.minimum.at(D[s], (yi[ok], xi[ok]), z[ok])
    hit = np.isfinite(D).any(0)
    lab = np.argmin(D, 0)
    return [hit & (lab == s) for s in range(len(grid_fns))]


def nelder_mead(f, x0, steps, rounds=3, maxiter=4000, log=True):
    x0 = np.asarray(x0, float)
    for k in range(rounds):
        sim = [x0] + [x0 + np.eye(len(x0))[i] * steps[i] for i in range(len(x0))]
        r = minimize(f, x0, method="Nelder-Mead",
                     options=dict(maxiter=maxiter, initial_simplex=sim, xatol=1e-5, fatol=1e-5))
        x0 = r.x
        if log:
            print("  round", k, json.dumps([round(float(v), 5) for v in x0]), round(float(r.fun), 5), flush=True)
    return x0


def spec_grid_fn(spec):
    return lambda n: rm.surface_grid(spec, n)


def curve_pixels_loss(curves, cam, red_dt, red_pix, W, H):
    """Symmetric mean distance between the projected curves and the panel's red pixels."""
    pts = []
    for C in curves:
        x, y, _ = rm.project(np.asarray(C, float), cam, W, H)
        pts.append(np.column_stack([x, y]))
    P = np.vstack(pts)
    xi = np.clip(np.round(P[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(P[:, 1]).astype(int), 0, H - 1)
    outside = (P[:, 0] < 0) | (P[:, 0] >= W) | (P[:, 1] < 0) | (P[:, 1] >= H)
    d1 = red_dt[yi, xi].mean() + 50.0 * outside.mean()
    m = np.zeros((H, W), bool)
    m[yi, xi] = True
    d2 = distance_transform_edt(~m)[red_pix].mean()
    return 0.5 * (d1 + d2)


def fit_result_camera(curves, rgb, proj="ortho", rounds=2, dist=None):
    """Fit a camera that makes the truth curves land on the red pixels of a result panel:
    coarse multi-start over (el, az, roll) with scale and centre matched to the red pixels'
    bounding box, then Nelder-Mead on the symmetric curve/pixel distance."""
    a = rgb.astype(int)
    red = (a[..., 0] > 150) & (a[..., 1] < 110) & (a[..., 2] < 110)
    H, W = red.shape
    dt = distance_transform_edt(~red)
    ys, xs = np.nonzero(red)
    bw, bh = np.ptp(xs) + 1, np.ptp(ys) + 1
    bc = np.array([xs.min() + xs.max(), ys.min() + ys.max()]) / 2.0
    P = np.vstack([np.asarray(c, float) for c in curves])
    tgt = (P.min(0) + P.max(0)) / 2

    def cam_of(x):
        c = dict(proj=proj, el=x[0], az=x[1], roll=x[2], scale=math.exp(x[3]), cx=x[4], cy=x[5],
                 target=[float(v) for v in tgt])
        if proj == "persp":
            c["dist"] = math.exp(x[6])
        return c

    def loss(x):
        return curve_pixels_loss(curves, cam_of(x), dt, red, W, H)
    starts = []
    for el in range(-75, 90, 15):
        for az in range(-180, 180, 15):
            for roll in (-30, -15, 0, 15, 30):
                x = [el, az, roll, 0.0, W / 2, H / 2] + ([math.log(dist)] if proj == "persp" else [])
                c = cam_of(x)
                px, py, _ = rm.project(P, c, W, H)
                sc = min(bw / max(np.ptp(px), 1e-9), bh / max(np.ptp(py), 1e-9))
                x[3] = math.log(sc)
                c = cam_of(x)
                px, py, _ = rm.project(P, c, W, H)
                x[4] += bc[0] - (px.min() + px.max()) / 2
                x[5] += bc[1] - (py.min() + py.max()) / 2
                starts.append((loss(x), x))
    starts.sort(key=lambda t: t[0])
    steps = np.array([3, 3, 2, 0.04, 4, 4, 0.1])
    best = None
    for _, x0 in starts[:6]:
        x = nelder_mead(loss, x0, steps[:len(x0)], rounds, log=False)
        if best is None or loss(x) < best[0]:
            best = (loss(x), x)
    return cam_of(best[1]), best[0]


def save_compare(stem, specs, cam, curves, extra_rows=(), alphas=(0.72, 0.72)):
    """compare/<stem>.png: paper Surfaces panel | our render from the fitted camera | 50% overlay
    (+ any extra (paper_rgb, our_rgb) rows)."""
    P = paper_panel(stem)
    H, W = P.shape[:2]
    ours, _ = rm.render(specs, cam, W, H, "surfaces", curves=curves, alphas=alphas, curve_width=5.5)
    rows_p, rows_o = [P], [ours]
    for p_, o_ in extra_rows:
        rows_p.append(p_)
        rows_o.append(o_)
    out = os.path.join(COMPARE, stem + ".png")
    rm.compare_strip(rows_p, rows_o, out)
    return out


def curves_only(curves, cam, W, H, width=5.5, points=(), ss=3):
    """Result-row style: the red curves (and points) alone on white."""
    from PIL import ImageDraw
    big = Image.new("RGB", (W * ss, H * ss), (255, 255, 255))
    dr = ImageDraw.Draw(big)
    for C in curves:
        x, y, _ = rm.project(np.asarray(C, float), cam, W, H)
        dr.line(list(zip(x * ss, y * ss)), fill=rm.RED, width=int(round(width * ss)), joint="curve")
    for p in points:
        x, y, _ = rm.project(np.asarray(p, float), cam, W, H)
        R = 6.5 * ss
        dr.ellipse([x * ss - R, y * ss - R, x * ss + R, y * ss + R], fill=rm.RED)
    return np.asarray(big.resize((W, H), Image.LANCZOS))


def save_truth(stem, specs, cam, curves, result_cam=None, points=()):
    """compare/<stem>_truth.png: translucent surfaces + exact truth in red from the fitted camera;
    with result_cam also the paper's 'Ours' panel next to the truth alone from that camera."""
    P = paper_panel(stem)
    H, W = P.shape[:2]
    a, _ = rm.render(specs, cam, W, H, "result", curves=curves, points=points, curve_width=4.5)
    tiles = [a]
    if result_cam is not None:
        R = paper_panel(stem, "ours")
        Hr, Wr = R.shape[:2]
        b = curves_only(curves, result_cam, Wr, Hr, points=points)
        tiles += [R, b]
    Hm = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t[..., :3], ((0, Hm - t.shape[0]), (0, 8), (0, 0)), constant_values=255) for t in tiles]
    out = os.path.join(COMPARE, stem + "_truth.png")
    Image.fromarray(np.concatenate(tiles, 1)).save(out)
    return out


def write_case(case):
    with open(os.path.join(DATA, case["id"] + ".json"), "w") as fh:
        json.dump(case, fh)
