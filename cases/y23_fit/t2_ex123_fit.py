"""Camera and shape fitting helpers for the Table 2 examples 1-3 refits (Yang, Jia & Yan 2023).

Used by y23_t2_ex1_tangent_line.py, y23_t2_ex2_tangential_curve.py and y23_t2_ex3_loop_two_curves.py.
The renderer itself is render_match.py; this module adds what the fit needs:

  * fast per-surface silhouette and depth rasters (flat-shaded quads, no anti-aliasing),
  * the paper panel's masks: foreground, visible orange, red curve,
  * an objective built from silhouette IoU, visible-orange IoU, red-curve chamfer and
    optional pixel landmarks, minimised over camera and shape parameters,
  * the deliverables: compare strip (paper | ours | 50 % overlay) and the result-style truth render.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_match as rm  # noqa: E402

PAPER = os.path.join(HERE, "paper")
COMPARE = os.path.join(HERE, "compare")

# Table 2 "Surfaces" column: lighter, translucent surfaces than the Fig. 1 panels
T2_BLUE = np.array([0.55, 0.77, 1.0])
T2_ORANGE = np.array([1.0, 0.66, 0.42])
T2_ALPHAS = (0.72, 0.85)


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def paper_masks(rgb):
    """(foreground, visible orange, red) masks of a Table 2 panel."""
    a = np.asarray(rgb, float) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    fg = rm.paper_mask(rgb)
    red = (r > 0.75) & (g < 0.3) & (b < 0.3)
    orange = fg & (r > b + 0.08) & ~red
    fg = ndimage.binary_fill_holes(ndimage.binary_closing(fg | red, iterations=2))
    return fg, orange, red


def raster(grid, cam, W, H, ss=1):
    """Opaque silhouette mask and nearest depth of one surface grid (flat depth per triangle)."""
    n, m = grid.shape[:2]
    x, y, z = rm.project(grid, cam, W, H)
    X, Y, Z = (x * ss).ravel(), (y * ss).ravel(), z.ravel()
    i, j = np.meshgrid(np.arange(n - 1), np.arange(m - 1), indexing="ij")
    i, j = i.ravel(), j.ravel()
    quads = np.stack([i * m + j, (i + 1) * m + j, (i + 1) * m + j + 1, i * m + j + 1], 1)
    zq = Z[quads].mean(1)
    order = np.argsort(-zq)
    dep = Image.new("F", (W * ss, H * ss), float("inf"))
    dd = ImageDraw.Draw(dep)
    for t in order:
        q = quads[t]
        dd.polygon([(X[k], Y[k]) for k in q], fill=float(zq[t]), outline=float(zq[t]))
    D = np.asarray(dep, float)
    return np.isfinite(D), D


def iou(a, b):
    return rm.mask_iou(a, b)


class Target:
    """Paper panel masks plus distance transforms for chamfer terms."""

    def __init__(self, path):
        self.rgb = load_rgb(path)
        self.H, self.W = self.rgb.shape[:2]
        self.fg, self.orange, self.red = paper_masks(self.rgb)
        self.red_dt = ndimage.distance_transform_edt(~self.red) if self.red.any() else None
        self.red_px = np.argwhere(self.red)[:, ::-1].astype(float)  # (x, y)


def curve_terms(T, curves_px):
    """Mean distance of our projected curve samples to paper red pixels and back (pixels)."""
    if T.red_dt is None or not len(curves_px):
        return 0.0
    P = np.vstack(curves_px)
    xi = np.clip(np.round(P[:, 0]).astype(int), 0, T.W - 1)
    yi = np.clip(np.round(P[:, 1]).astype(int), 0, T.H - 1)
    inside = (P[:, 0] >= 0) & (P[:, 0] < T.W) & (P[:, 1] >= 0) & (P[:, 1] < T.H)
    d1 = np.where(inside, T.red_dt[yi, xi], 50.0).mean()
    from scipy.spatial import cKDTree
    d2, _ = cKDTree(P).query(T.red_px[:: max(1, len(T.red_px) // 600)])
    return 0.5 * (d1 + np.minimum(d2, 50.0).mean())


def evaluate(T, grids, cam, curves=(), weights=(1.0, 0.5, 0.02)):
    """Objective pieces for one camera/shape: (loss, silhouette IoU, visible-orange IoU, curve px)."""
    mb, db = raster(grids[0], cam, T.W, T.H)
    mo, do = raster(grids[1], cam, T.W, T.H)
    sil = mb | mo
    vis_o = mo & ~(mb & (db < do))
    i_sil = iou(sil, T.fg)
    i_or = iou(vis_o, T.orange)
    cpx = []
    for C in curves:
        x, y, _ = rm.project(np.asarray(C, float), cam, T.W, T.H)
        cpx.append(np.column_stack([x, y]))
    ct = curve_terms(T, cpx)
    loss = weights[0] * (1 - i_sil) + weights[1] * (1 - i_or) + weights[2] * ct
    return loss, i_sil, i_or, ct


CAM_KEYS = ("el", "az", "roll", "scale", "cx", "cy")


def cam_from(vec, base):
    cam = dict(base)
    for k, v in zip(CAM_KEYS, vec):
        cam[k] = float(v)
    return cam


def fit(T, build, x0, cam0, steps=(400, 400), curves_fn=None, weights=(1.0, 0.5, 0.02), fix=(), log=None):
    """Nelder-Mead over camera (CAM_KEYS) + shape vector; build(shape) -> (grid_blue, grid_orange, curves)."""
    x0 = np.asarray(x0, float)
    c0 = np.array([cam0[k] for k in CAM_KEYS], float)
    nc = len(c0)
    fixed = np.array([k in fix for k in CAM_KEYS] + [False] * len(x0))
    full0 = np.concatenate([c0, x0])
    best = [np.inf, None]

    def f(z):
        full = full0.copy()
        full[~fixed] = z
        cam = cam_from(full[:nc], cam0)
        try:
            gb, go, curves = build(full[nc:])
        except (ValueError, FloatingPointError, AssertionError):
            return 10.0
        loss = evaluate(T, (gb, go), cam, curves, weights)[0]
        if loss < best[0]:
            best[0], best[1] = loss, full.copy()
        return loss

    z0 = full0[~fixed]
    for it in steps:
        scale = np.where(np.abs(z0) > 1e-3, 0.08 * np.abs(z0), 0.05)
        simplex = [z0] + [z0 + np.eye(len(z0))[k] * scale[k] for k in range(len(z0))]
        res = optimize.minimize(f, z0, method="Nelder-Mead",
                                options=dict(maxfev=it, initial_simplex=np.array(simplex), xatol=1e-4, fatol=1e-5))
        z0 = best[1][~fixed]
        if log:
            log(best[0])
    full = best[1]
    return cam_from(full[:nc], cam0), full[nc:], best[0]


def grid_from_fn(fn, ur, wr, n=41, m=None):
    m = m or n
    u, w = np.meshgrid(np.linspace(*ur, n), np.linspace(*wr, m), indexing="ij")
    return fn(u, w)


def render_t2(specs_or_grids, cam, W, H, curves=(), style="surfaces", ss=3, n=121):
    if style == "surfaces":
        return rm.render(specs_or_grids, cam, W, H, "surfaces", curves=curves, ss=ss, n=n, alphas=T2_ALPHAS,
                         colors=(T2_BLUE, T2_ORANGE), curve_width=2.6)
    return rm.render(specs_or_grids, cam, W, H, "result", curves=curves, ss=ss, n=n, curve_width=3.2)


def write_compare(stem, case, cam, panels, out_dir=COMPARE, extra_rows=()):
    """compare/<stem>.png: each paper panel | our render from the fitted camera | 50 % overlay.
    panels: list of (paper png path, style) with style "surfaces" (no curves) or "curve" (curves only)."""
    os.makedirs(out_dir, exist_ok=True)
    curves = [np.array(b["points"]) for b in case["expected"]["branches"]]
    papers, ours = [], []
    for path, style in panels:
        P = load_rgb(path)
        H, W = P.shape[:2]
        if style == "surfaces":
            img, _ = render_t2(case["surfaces"], cam, W, H, curves=curves, style="surfaces")
        else:
            img, _ = render_t2(case["surfaces"], cam, W, H, curves=curves, style="result")
        papers.append(P)
        ours.append(img)
    for P, O in extra_rows:
        papers.append(P)
        ours.append(O)
    rm.compare_strip(papers, ours, os.path.join(out_dir, stem + ".png"))


def write_truth(stem, case, cam, W, H, out_dir=COMPARE):
    curves = [np.array(b["points"]) for b in case["expected"]["branches"]]
    pts = list(case["expected"]["isolated_points"]) + list(case["expected"]["singular_points"])
    img, _ = rm.render(case["surfaces"], cam, W, H, "result", curves=curves, points=pts, ss=3, n=121)
    Image.fromarray(img).save(os.path.join(out_dir, stem + "_truth.png"))


def score_case(T, case, cam):
    """Silhouette IoU and visible-orange IoU of the stored surfaces against a paper panel."""
    grids = [rm.surface_grid(s, 81) for s in case["surfaces"]]
    curves = [np.array(b["points"]) for b in case["expected"]["branches"]]
    _, i_sil, i_or, ct = evaluate(T, grids, cam, curves)
    return dict(iou_silhouette=round(i_sil, 4), iou_visible_orange=round(i_or, 4), curve_chamfer_px=round(ct, 2))


def auto_frame(T, grids, cam):
    """Set scale, cx, cy so that the projected sample cloud matches the paper foreground's spread and centroid."""
    P = np.vstack([g.reshape(-1, 3) for g in grids])
    c = dict(cam, scale=1.0, cx=0.0, cy=0.0)
    x, y, _ = rm.project(P, c, T.W, T.H)
    ys, xs = np.nonzero(T.fg)
    s = np.sqrt((xs.var() + ys.var()) / (x.var() + y.var() + 1e-12))
    return dict(cam, scale=float(s), cx=float(xs.mean() - s * x.mean()), cy=float(ys.mean() - s * y.mean()))


def coarse_views(T, grids, curves=(), base=None, els=range(-80, 81, 10), azs=range(-180, 180, 15),
                 rolls=range(-180, 180, 15), keep=8, weights=(1.0, 0.5, 0.02)):
    """Exhaustive orthographic orientation search with auto framing; returns the best `keep` cameras."""
    base = base or dict(proj="ortho", target=[0.0, 0.0, 0.0])
    sub = [g[::2, ::2] for g in grids]
    out = []
    for el in els:
        for az in azs:
            for roll in rolls:
                cam = auto_frame(T, sub, dict(base, el=float(el), az=float(az), roll=float(roll)))
                out.append((evaluate(T, sub, cam, curves, weights)[0], cam))
    out.sort(key=lambda t: t[0])
    return out[:keep]


def fit_curve_camera(T, curves, els=range(-90, 91, 10), azs=range(-180, 180, 10), rolls=range(-180, 180, 10)):
    """Orthographic camera for a result-row panel that shows only the red curves: orientation grid search with
    framing on the red pixels, then Nelder-Mead on the two-sided curve chamfer (pixels)."""
    P = np.vstack([np.asarray(c, float) for c in curves])
    sub = [np.asarray(c, float)[:: max(1, len(c) // 150)] for c in curves]
    ys, xs = np.nonzero(T.red)

    def frame(cam):
        c = dict(cam, scale=1.0, cx=0.0, cy=0.0)
        x, y, _ = rm.project(P, c, T.W, T.H)
        s = np.sqrt((xs.var() + ys.var()) / (x.var() + y.var() + 1e-12))
        return dict(cam, scale=float(s), cx=float(xs.mean() - s * x.mean()), cy=float(ys.mean() - s * y.mean()))

    def cost(cam, cs):
        px = []
        for C in cs:
            x, y, _ = rm.project(C, cam, T.W, T.H)
            px.append(np.column_stack([x, y]))
        return curve_terms(T, px)

    best = []
    for el in els:
        for az in azs:
            for roll in rolls:
                cam = frame(dict(proj="ortho", target=[0.0, 0.0, 0.0], el=float(el), az=float(az), roll=float(roll)))
                best.append((cost(cam, sub), cam))
    best.sort(key=lambda t: t[0])
    out = []
    for c0, cam0 in best[:5]:
        z0 = np.array([cam0[k] for k in CAM_KEYS])
        res = optimize.minimize(lambda z: cost(cam_from(z, cam0), sub), z0, method="Nelder-Mead",
                                options=dict(maxfev=1500, xatol=1e-4, fatol=1e-4))
        out.append((res.fun, cam_from(res.x, cam0)))
    out.sort(key=lambda t: t[0])
    return out[0][1], out[0][0]


def write_compare_rows(path, rows):
    """rows: list of (paper rgb, our rgb) -> paper | ours | 50 % overlay per row."""
    rm.compare_strip([r[0] for r in rows], [r[1] for r in rows], path)


def deliver(stem, case, cam):
    """Score against the Surfaces panel, fit the result-row ("Ours") camera on the truth curves, and write
    compare/<stem>.png (rows: Surfaces panel; result panel) and compare/<stem>_truth.png. Returns the numbers."""
    T = Target(os.path.join(PAPER, stem + "_surfaces.png"))
    curves = [np.array(b["points"]) for b in case["expected"]["branches"]]
    pts = list(case["expected"]["isolated_points"]) + list(case["expected"]["singular_points"])
    score = score_case(T, case, cam)
    ours, _ = render_t2(case["surfaces"], cam, T.W, T.H, curves=curves, style="surfaces")
    rows = [(T.rgb, ours)]
    res_png = os.path.join(PAPER, stem + "_ours.png")
    out = dict(score, camera={k: (round(v, 2) if isinstance(v, float) else v) for k, v in cam.items()})
    if os.path.exists(res_png):
        R = Target(res_png)
        rcam, chamfer = fit_curve_camera(R, curves)
        img, _ = rm.render(case["surfaces"], rcam, R.W, R.H, "result", curves=curves, alphas=(0.0, 0.0), ss=3,
                           curve_width=4.0)
        rows.append((R.rgb, img))
        out["result_panel_camera"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rcam.items()}
        out["result_panel_curve_chamfer_px"] = round(float(chamfer), 2)
    write_compare_rows(os.path.join(COMPARE, stem + ".png"), rows)
    img, _ = rm.render(case["surfaces"], cam, T.W, T.H, "result", curves=curves, points=pts, ss=3, n=121)
    Image.fromarray(img).save(os.path.join(COMPARE, stem + "_truth.png"))
    return out
