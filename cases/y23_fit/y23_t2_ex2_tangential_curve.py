"""Table 2 example 2 of Yang, Jia & Yan 2023 ("a tangential curve", bi-degree (3,3)), refitted to the paper's
Surfaces panel (cases/y23_fit/paper/y23_t2_ex2_tangential_curve_*.png, cropped from p.10 at 600 dpi).

What the panel shows:
  * blue: a long narrow ribbon running from upper left to lower right. Along its length it is one cubic wave
    (crest on the left, trough on the right); across its width every section is an S, so at both ends the
    ribbon turns over and shows its back (the two dark "flaps").
  * orange: a shorter ribbon of the same kind, offset below the blue on the left and above it on the right.
  * red: one S-shaped curve (dip, then crest) that crosses the ribbons from one long edge to the other.
  * above the red curve the orange is seen through the blue, below it the orange is in front: the surfaces
    cross along the curve, so the contact is of odd order (slide 5 of the talk calls this example
    "high order tangency").

Construction (exact):
  * blue  B(u, w) = (u, w + p(u), H(u, w)),          u in [-1, 1],   w in [-W, W]
  * orange O(u, w) = (u, w + p(u), H(u, w) + d u^3),  u in [-Uo, Uo], w in [-Wo, Wo]
    with H a full bicubic and p(u) = p1 u + p2 u^2. Both are sheared height fields over the same footprint
    map, so they meet exactly where d u^3 = 0: the cross-section u = 0, i.e. the curve (0, w, H(0, w)).
    grad(d u^3) = 0 there, so the normals agree along the whole curve (tangential), and d u^3 changes sign,
    so the surfaces cross through each other (second-order contact).
  * H, p, the domains and d were fitted together with an orthographic camera (Nelder-Mead on silhouette
    IoU + visible-orange IoU + red-curve chamfer against the panel, t2_ex123_fit.py), starting from a wave
    ribbon whose orientation came from a grid search over the view sphere.

Run: .venv/bin/python -I cases/y23_fit/y23_t2_ex2_tangential_curve.py [--refit]
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

CID = "y23_t2_ex2_tangential_curve"

PARAMS = {'c01': 0.26222344900678796, 'c02': -0.07929949481877849, 'c03': -1.971118982894088,
          'c10': -0.40201044501109817, 'c11': 0.12756454044597676, 'c12': -0.004333324390580192,
          'c13': -0.07450585838582277, 'c20': -0.08951064212214689, 'c21': 0.027809256258477428,
          'c22': 0.005245882615039891, 'c23': 0.14694921535657623, 'c30': 0.5321238541872245,
          'c31': -0.09312819028976943, 'c32': -0.07658228019607066, 'c33': 0.10259897794736747,
          'p1': -0.2172971287904486, 'p2': 0.15034123439189284, 'W': 0.5619919492757071,
          'Uo': 0.5287501857575841, 'Wo': 0.5387115734596768, 'd': 1.4352001473759113}
CAMERA = {'proj': 'ortho', 'target': [0.0, 0.0, 0.0], 'el': 23.495841893303073, 'az': -63.0522696490609,
          'roll': 6.825293953588882, 'scale': 222.21509934761633, 'cx': 261.18822374825754,
          'cy': 195.83502930645852}
IOU_BEFORE = {'iou_silhouette': 0.5969, 'iou_visible_orange': 0.351, 'curve_chamfer_px': 9.19}  # old case, own best camera

MON = [(i, j) for i in range(4) for j in range(4) if (i, j) != (0, 0)]
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
    blue = Y.graph_patch(H, (-1.0, 1.0), (-p['W'], p['W']), "wave ribbon with S cross-sections", degree=(3, 3),
                         shear=sh)
    orange = Y.graph_patch(Ho, (-p['Uo'], p['Uo']), (-p['Wo'], p['Wo']), "shorter wave ribbon (blue + d u^3)",
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
    Y.add(CID, "Table 2 ex.2 (3,3): a tangential curve", blue, orange, Y.SRC + "Table 2 (example 2)",
          "'a tangential curve', bi-degree (3,3)",
          "bi-degree (3,3) (both surfaces); topology: one open tangential curve crossing the ribbons edge to edge, "
          "normals parallel along it (max sin %.1e), surfaces cross through each other (offset d u^3: second-order, "
          "odd contact), as the panel's depth order shows; appearance: refitted to the Surfaces panel from a fitted "
          "camera (%s): blue a long wave ribbon whose sections are S-shaped (turned-over ends), orange a shorter "
          "ribbon of the same kind; silhouette IoU against the panel {IOU} (before: %.3f)" % (
              sin, cam_txt, IOU_BEFORE['iou_silhouette']),
          "exact control points (the paper publishes none): fitted by eye and by silhouette/curve matching to one "
          "view; the depth of the ribbons along the view direction is fixed only by shading and occlusion; the "
          "offset direction of the orange (z of the shared height field) and its law d u^3 are our choice",
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
