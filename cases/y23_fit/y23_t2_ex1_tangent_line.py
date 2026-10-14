"""Table 2 example 1 of Yang, Jia & Yan 2023 ("a tangent line", bi-degree (3,3)), refitted to the paper's
Surfaces panel (cases/y23_fit/paper/y23_t2_ex1_tangent_line_*.png, cropped from p.10 at 600 dpi). Fig. 10 (left)
shows the curve as the iso-line s = 0.5 across the whole domain of one surface.

What the panel shows:
  * blue: a tall S-step sheet seen nearly face-on at its wall: the lower plateau runs out to the lower left, the
    upper plateau to the upper right (both foreshortened to thin tips); the upper edge carries a dip and a
    crest, so the sheet is wavy across as well as stepped along (a genuine bicubic, not a ruled sheet).
  * orange: a shorter sheet of the same profile covering the middle of the wall; its lower-left and upper-right
    tips stick out past the blue edges.
  * red: one straight horizontal line across the full width of the wall.
  * above the line the orange is seen through the blue, below it the orange is in front: the surfaces cross
    along the line (odd-order contact).

Construction (exact):
  * blue  B(u, w) = (u, w + p(u), H(u, w)),          u in [-1, 1],   w in [-W, W]
  * orange O(u, w) = (u, w + p(u), H(u, w) + d u^3),  u in [-Uo, Uo], w in [-Wo, Wo]
    H is bicubic with H(0, w) = c00 + c01 w (no w^2, w^3 terms at u = 0), so the contact u = 0 is the straight
    segment (0, w, c01 w), and it is the iso-line s = 0.5 of the orange across its whole w range, as in
    Fig. 10. grad(d u^3) = 0 on it, so the normals agree along the whole line; d u^3 changes sign, so the
    sheets cross (second-order contact). The step is a cubic height field in a frame turned about 45 deg to the
    wall, which the fitted camera absorbs.
  * H, p, the domains and d were fitted together with an orthographic camera (Nelder-Mead with random restarts
    on silhouette IoU + visible-orange IoU + red-curve chamfer against the panel, t2_ex123_fit.py).

Run: .venv/bin/python -I cases/y23_fit/y23_t2_ex1_tangent_line.py [--refit]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "cases"))
sys.path.insert(0, os.path.join(ROOT, "scoring"))
import gen_yang2023 as Y  # noqa: E402  (read-only helpers)
import gen_cases as gc  # noqa: E402
import ssi_geom as g  # noqa: E402
import t2_ex123_fit as F  # noqa: E402

CID = "y23_t2_ex1_tangent_line"

PARAMS = {'c01': 0.16052242064414096, 'c10': 1.0137945757300901, 'c11': -0.2842578815298231,
          'c12': -0.520083126598395, 'c13': -0.08429820048185643, 'c20': -0.11351844363431819,
          'c21': 0.17705560915190988, 'c22': -0.03614694894217198, 'c23': -0.5928720517024585,
          'c30': -0.8428719959924943, 'c31': 0.951699019546012, 'c32': 0.16817870753774278,
          'c33': -0.39323955940938926, 'p1': -0.014821426321799454, 'p2': 0.004422095186954753,
          'W': 0.5965039433539991, 'Uo': 0.3306111412748932, 'Wo': 0.5965029788805274, 'd': 0.6875154730866446}
CAMERA = {'proj': 'ortho', 'target': [0.0, 0.0, 0.0], 'el': -34.11132194464844, 'az': -51.138321908821055,
          'roll': -26.647520812441897, 'scale': 189.6443474438767, 'cx': 234.2016646870896,
          'cy': 233.51237053077205}
IOU_BEFORE = {'iou_silhouette': 0.8573, 'iou_visible_orange': 0.2631, 'curve_chamfer_px': 3.3}  # old case, own best camera

MON = [(i, j) for i in range(4) for j in range(4) if (i, j) != (0, 0) and not (i == 0 and j >= 2)]  # straight u = 0
NAMES = ['c%d%d' % m for m in MON] + ['p1', 'p2', 'W', 'Uo', 'Wo', 'd']


def H_of(p):
    H = np.zeros((4, 4))
    for i, j in MON:
        H[i, j] = p['c%d%d' % (i, j)]
    return H


def shear_of(p):
    return [0.0, p['p1'], p['p2']]


def model(x, n=25):
    """Fit-time model (grids + curve) for t2_ex123_fit.fit."""
    p = dict(zip(NAMES, x))
    H, sh = H_of(p), shear_of(p)

    def S(u, w, orange):
        P = g.sheared_point(H, sh, u.ravel(), w.ravel()).reshape(u.shape + (3,))
        if orange:
            P[..., 2] += p['d'] * u ** 3
        return P
    gb = F.grid_from_fn(lambda u, w: S(u, w, False), (-1, 1), (-p['W'], p['W']), n)
    go = F.grid_from_fn(lambda u, w: S(u, w, True), (-p['Uo'], p['Uo']), (-p['Wo'], p['Wo']), n)
    t = np.linspace(-min(p['W'], p['Wo']), min(p['W'], p['Wo']), 50)
    return gb, go, [S(0 * t, t, False)]


def build(p=PARAMS):
    H, sh = H_of(p), shear_of(p)
    Ho = g.p2_add(H, Y.mono(3, 0, p['d']))
    blue = Y.graph_patch(H, (-1.0, 1.0), (-p['W'], p['W']), "tall wavy S-step sheet", degree=(3, 3),
                         shear=sh)
    orange = Y.graph_patch(Ho, (-p['Uo'], p['Uo']), (-p['Wo'], p['Wo']), "shorter S-step sheet (blue + d u^3)",
                           degree=(3, 3), shear=sh)
    wm = min(p['W'], p['Wo'])
    curve = (lambda t: g.sheared_point(H, sh, 0 * t, t), -wm, wm)
    return blue, orange, curve, H, Ho, sh


def contact_checks(p, H, Ho, sh, n=401):
    """Angle between the normals along the curve, and the sign of the offset on either side of it."""
    w = np.linspace(-min(p['W'], p['Wo']), min(p['W'], p['Wo']), n)
    du = lambda C: (np.arange(C.shape[0])[:, None] * C)[1:]  # d/du of power coefficients c[i][j] u^i w^j
    dw = lambda C: (np.arange(C.shape[1])[None, :] * C)[:, 1:]
    pd = sh[1]  # p'(0)
    sins = []
    for C in (H, Ho):
        Su = np.stack([np.ones_like(w), pd + 0 * w, g.p2_eval(du(C), 0 * w, w)], -1)
        Sw = np.stack([0 * w, np.ones_like(w), g.p2_eval(dw(C), 0 * w, w)], -1)
        nn = np.cross(Su, Sw)
        sins.append(nn / np.linalg.norm(nn, axis=1)[:, None])
    sin = np.linalg.norm(np.cross(sins[0], sins[1]), axis=1).max()
    h = 0.05
    gap = [p['d'] * (-h) ** 3, p['d'] * h ** 3]
    return float(sin), gap


def main():
    if "--refit" in sys.argv:
        T = F.Target(os.path.join(F.PAPER, CID + "_surfaces.png"))
        cam, x, loss = F.fit(T, model, [PARAMS[k] for k in NAMES], CAMERA, steps=[3000, 3000],
                             weights=(1.0, 0.5, 0.05), log=print)
        print(repr(dict(zip(NAMES, map(float, x)))))
        print(repr(cam))
        return
    p = PARAMS
    blue, orange, curve, H, Ho, sh = build(p)
    sin, gap = contact_checks(p, H, Ho, sh)
    assert sin < 1e-12, sin
    assert gap[0] * gap[1] < 0
    cam_txt = "el %.2f, az %.2f, roll %.2f deg, orthographic, scale %.1f px/unit, target at (%.1f, %.1f) px" % (
        CAMERA['el'], CAMERA['az'], CAMERA['roll'], CAMERA['scale'], CAMERA['cx'], CAMERA['cy'])
    P0 = curve[0](np.array([curve[1], curve[2]]))
    Pm = curve[0](np.linspace(curve[1], curve[2], 201))
    dev = np.abs(np.cross(Pm - P0[0], P0[1] - P0[0])).max() / np.linalg.norm(P0[1] - P0[0])
    assert dev < 1e-12, dev
    Y.add(CID, "Table 2 ex.1 (3,3): a tangent line", blue, orange, Y.SRC + "Table 2 (example 1)",
          "'a tangent line', bi-degree (3,3); Fig. 10 (left) shows it as an iso-parameter line across the domain",
          "bi-degree (3,3) (both surfaces); topology: one open tangential straight line (deviation from its chord "
          "%.0e), the iso-line u = 0 (s = 0.5) across the whole width of both sheets, normals parallel along it "
          "(max sin %.1e), the sheets cross through each other (offset d u^3: second-order, odd contact) as the "
          "panel's depth order shows; appearance: refitted to the Surfaces panel from a fitted camera (%s): tall "
          "wavy S-step sheet and a shorter sheet of the same profile on its wall; silhouette IoU against the panel "
          "{IOU} (before: %.3f)" % (dev, sin, cam_txt, IOU_BEFORE['iou_silhouette']),
          "exact control points (the paper publishes none): fitted by eye and by silhouette/curve matching to one "
          "view; the plateau tips are seen nearly edge-on, so their length along the view is weakly constrained; "
          "the orange's tips poking out past the blue edges are only partly reproduced; the offset law d u^3 and "
          "its direction are our choice",
          "paper: " + Y.OPEN + " and " + Y.SPL + " returned nothing or an error, " + Y.COMM + " two points only",
          curves=[(curve[0], curve[1], curve[2], False, "tangent")], cov_tol=1e-2)
    case = gc.CASES[-1]
    out = F.deliver(CID, case, CAMERA)
    case["matched"] = case["matched"].replace("{IOU}", "%.3f" % out["iou_silhouette"])
    with open(os.path.join(ROOT, "cases", "data", CID + ".json"), "w") as fh:
        json.dump(case, fh)
    print(json.dumps(dict(out, contact_max_sin=sin, offset_either_side=gap, iou_before=IOU_BEFORE), indent=1))


if __name__ == "__main__":
    main()
