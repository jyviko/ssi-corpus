"""Compare images and silhouette scores for the refitted Fig. 1(e)-(g) cases.

compare(case, camera, out_stem): writes
  compare/<id>.png        rows: paper Surfaces panel | our render | 50% overlay,
                                paper "Ours" panel    | our result-style render with the exact truth | overlay
  compare/<id>_truth.png  our render with the exact truth in red, in the paper's result-row style
and returns the scores against the Surfaces panel:
  iou        foreground (any surface) mask IoU
  iou_blue   IoU of the blue-dominated pixels, iou_orange likewise
"""
import os

import numpy as np
from PIL import Image

import render_match as rm

HERE = os.path.dirname(os.path.abspath(__file__))


def paper(cid, row):
    return np.asarray(Image.open(os.path.join(HERE, "paper", f"{cid}_{row}.png")).convert("RGB"))


def scores(paper_rgb, ours_rgb):
    pm, om = rm.paper_mask(paper_rgb), rm.paper_mask(ours_rgb)
    pb, po, _ = rm.color_masks(paper_rgb)
    ob, oo, _ = rm.color_masks(ours_rgb)
    return {"iou": round(rm.mask_iou(pm, om), 4), "iou_blue": round(rm.mask_iou(pb, ob), 4),
            "iou_orange": round(rm.mask_iou(po, oo), 4)}


def truth_geometry(case):
    curves = [np.asarray(b["points"], float) for b in case["expected"]["branches"]]
    pts = [np.asarray(p, float) for p in case["expected"]["isolated_points"]]
    return curves, pts


def fit_framing(case, camera, row="ours", n=41):
    """The paper's result rows are framed slightly differently from its Surfaces row (same view, shifted
    and scaled by a few pixels). Fit only that 2D framing (cx, cy, scale) to the result-row panel."""
    from scipy.optimize import minimize
    from scipy import ndimage as ndi
    P = paper(case["id"], row)
    H, W = P.shape[:2]
    a = np.asarray(P, float) / 255.0
    pm = (a.max(-1) - a.min(-1) > 0.035) | (1 - a.mean(-1) > 0.08)
    pm[:3], pm[-3:], pm[:, :3], pm[:, -3:] = False, False, False, False
    pm = ndi.binary_opening(pm, iterations=2)
    e = lambda m: m & ~ndi.binary_erosion(m)
    dp = ndi.distance_transform_edt(~e(pm))

    def cam_of(x):
        c = dict(camera)
        c["cx"], c["cy"], c["scale"] = camera["cx"] + x[0], camera["cy"] + x[1], camera["scale"] * (1 + x[2])
        return c

    def f(x):
        _, masks = rm.render(case["surfaces"], cam_of(x), W, H, "surfaces", n=n, ss=1)
        om = masks[0] | masks[1]
        eo = e(om)
        do = ndi.distance_transform_edt(~eo)
        return 0.5 * (np.minimum(dp[eo], 40).mean() + np.minimum(do[e(pm)], 40).mean())
    x0 = np.zeros(3)
    r = minimize(f, x0, method="Nelder-Mead",
                 options=dict(initial_simplex=np.vstack([x0, np.diag([6.0, 6.0, 0.05])]), maxfev=200))
    return cam_of(r.x)


def compare(case, camera, n=181, extra_points=(), result_camera=None):
    cid = case["id"]
    P_s, P_o = paper(cid, "surfaces"), paper(cid, "ours")
    H, W = P_s.shape[:2]
    curves, pts = truth_geometry(case)
    pts = pts + [np.asarray(p, float) for p in extra_points]
    surf, _ = rm.render(case["surfaces"], camera, W, H, "surfaces", n=n)
    surf_c, _ = rm.render(case["surfaces"], camera, W, H, "surfaces", curves=curves, points=pts, n=n,
                          curve_width=2.0, point_radius=3.0)
    res, _ = rm.render(case["surfaces"], result_camera or camera, W, H, "result", curves=curves, points=pts, n=n)
    os.makedirs(os.path.join(HERE, "compare"), exist_ok=True)
    rm.compare_strip([P_s, P_o], [surf_c, res], os.path.join(HERE, "compare", cid + ".png"))
    gap = np.full((H, 8, 3), 255, np.uint8)
    Image.fromarray(np.concatenate([P_o, gap, res], 1)).save(os.path.join(HERE, "compare", cid + "_truth.png"))
    return scores(P_s, surf)
