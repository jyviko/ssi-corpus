"""Fig. 1(d) of Yang, Jia & Yan 2023, refitted to the paper's panels.

What the panels show (cases/y23_fit/paper/y23_fig1d_two_quads_*.png, 600 dpi crops):
  * one S-shaped wavy ribbon (both surfaces nearly coincide), running from the upper left to the lower
    right: a high wall at the left end, a trough, a crest, and a fall to the lower right; each end edge
    rises and curls over (a bump across the ribbon); orange and blue alternate in visibility (the wall
    and the crest show the orange, the trough and the lower-right part the blue).
  * the red network: the two end edges, the two long edges, and a short middle edge across the
    ribbon, rising over the bump; in the Ours row the middle edge crosses the far long edge in
    projection (the "lens"), in the Surfaces row it is partly hidden behind the trough.
  * the Surfaces row and the result rows are the same view, framed 16 px apart (their silhouettes
    agree to IoU 0.89 after that shift); the Surfaces row shows the sheets opaque, so parts of the
    network are hidden there.

Construction (polynomial; orange bi-degree (2, 5), blue (4, 6); the paper states no degree):
  * orange O(u, w), u in [0, 1], w in [0, 2]: a Bezier patch, degree 2 across and 5 along, whose
    control net was fitted to the paper's red network in the Ours panel (chamfer both ways) and to the
    silhouette of the Surfaces panel, under one orthographic camera, with mild control-net fairness.
    One view does not fix depth: the depth of the net is the fairest one consistent with the image
    (fit_y23_fig1d_two_quads.py reproduces NET exactly).
  * blue B = O + D(u, w) V(u, w), D = k u (1 - u) w (2 - w) (w - 1)^2 >= 0, V a bi-degree (1, 2) polynomial
    field fitted to the unit normal of O and negated (V . n < 0 everywhere, checked: the blue lies on one
    side of the orange throughout).
  * B = O exactly where D = 0: the six boundary edges (simple zeros: transversal) and the middle edge
    w = 1 (double zero: B and O tangent along it, the contact edge of the two quadrilaterals).
    The ribbon folds over (no direction makes it a graph), so that there is no other common point is
    certified numerically: for every B point with D > 0 the distance to O is >= c D, c > 0 (checks()).

Camera (render_match.py convention): orthographic, el 20, az -90, roll 0, 100 px per unit (the net is
expressed in this camera's frame, so these values are a gauge, not a measurement); the Surfaces row is the
same camera shifted by (-16.1, -2.6) px.

Run: .venv/bin/python -I cases/y23_fit/y23_fig1d_two_quads.py [--compare]
"""
import json
import math
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

CID = "y23_fig1d_two_quads"

# orange control net [i across u (degree 2)][j along w (degree 5, w in [0, 2])] -> (x, y, z)
NET = np.array([[[-1.7986, 0.5382, 0.727], [-1.3774, 0.3217, -0.5364], [-0.7573, 0.1879, -1.4114],
                 [-0.2731, 0.0786, 0.4386], [0.2401, -0.1226, -0.4161], [0.6417, -0.3761, -1.595]],
                [[-1.4438, 0.7708, 1.1316], [-0.4245, 0.5201, 1.3036], [0.5798, 0.281, 1.4674],
                 [1.2136, 0.0989, 1.2536], [1.5003, -0.1752, 0.4107], [1.7786, -0.4773, -0.4835]],
                [[-1.0358, 0.9472, 1.4124], [-1.1386, 0.7883, -0.2385], [0.7583, 0.0392, -1.8407],
                 [0.3376, 0.0547, 1.1527], [1.4833, -0.3141, -0.2767], [1.8365, -0.6793, -1.6353]]])
K = 1.0          # max separation k / 16 = 0.0625 against a ribbon about 3.6 long
CAMERA = dict(proj="ortho", el=20.0, az=-90.0, roll=0.0, scale=100.0, cx=199.87, cy=187.45, target=[0.0, 0.0, 0.0])
RESULT_CAMERA = dict(proj="ortho", el=20.0, az=-90.0, roll=0.0, scale=100.0, cx=216.0, cy=190.0,
                     target=[0.0, 0.0, 0.0])
P1 = np.polynomial.polynomial


def bern_to_power(n, lo=0.0, hi=1.0):
    """Matrix M with power coefficients (in the original variable on [lo, hi]) = M @ Bernstein coefficients."""
    M = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        # B_i(t) = C(n,i) t^i (1-t)^(n-i), t = (x - lo) / (hi - lo)
        c = math.comb(n, i) * P1.polymul(P1.polypow([0.0, 1.0], i), P1.polypow([1.0, -1.0], n - i))
        cx = np.zeros(n + 1)
        base = np.array([1.0])
        for k, ck in enumerate(c):
            if k:
                base = P1.polymul(base, [-lo / (hi - lo), 1.0 / (hi - lo)])
            cx[:len(base)] += ck * base
        M[:, i] = cx
    return M


MU, MW = bern_to_power(2), bern_to_power(5, 0.0, 2.0)


def O_power():
    """[X, Y, Z] power arrays c[i][j] u^i w^j of the orange."""
    return [MU @ NET[:, :, k] @ MW.T for k in range(3)]


def O(u, w):
    return np.stack([g.p2_eval(c, u, w) for c in O_power()], -1)


def O_derivs(u, w, h=1e-6):
    return (O(u + h, w) - O(u - h, w)) / (2 * h), (O(u, w + h) - O(u, w - h)) / (2 * h)


def unit_normal(u, w):
    a, b = O_derivs(u, w)
    n = np.cross(a, b)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def fit_V(du=2, dw=2, n=41):
    """Least-squares polynomial field of bi-degree (du, dw) approximating the unit normal of O."""
    uu, ww = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 2, 2 * n), indexing="ij")
    N_ = unit_normal(uu, ww).reshape(-1, 3)
    A = np.stack([(uu.ravel() ** i) * (ww.ravel() ** j) for i in range(du + 1) for j in range(dw + 1)], 1)
    C = np.linalg.lstsq(A, N_, rcond=None)[0]
    out = []
    for k in range(3):
        c = np.zeros((du + 1, dw + 1))
        for m, (i, j) in enumerate([(i, j) for i in range(du + 1) for j in range(dw + 1)]):
            c[i, j] = round(float(C[m, k]), 6)
        out.append(c)
    return out


# the blue lies on the -n side (this side shows the blue in the trough and the lower-right part)
V_POWER = [-c for c in fit_V(2, 2)]


def D_power(k=K):
    u = P1.polymul([0.0, 1.0], [1.0, -1.0])                                    # u (1 - u)
    w = P1.polymul(P1.polymul([0.0, 1.0], [2.0, -1.0]), P1.polypow([-1.0, 1.0], 2))  # w (2 - w)(w - 1)^2
    return k * np.outer(u, w)


def surfaces(k=K):
    X, Yv, Z = O_power()
    Dp = D_power(k)
    Bh = [g.p2_add(c, g.p2_mul(Dp, v)) for c, v in zip((X, Yv, Z), V_POWER)]
    orange = g.rational_poly_patch(X, Yv, Z, [[1.0]], (0.0, 1.0), (0.0, 2.0),
                                   "S-wave ribbon (Bezier net fitted to the paper's two views)")
    blue = g.rational_poly_patch(Bh[0], Bh[1], Bh[2], [[1.0]], (0.0, 1.0), (0.0, 2.0),
                                 "ribbon offset D V, D = k u(1-u) w(2-w) (w-1)^2, V ~ unit normal")
    return blue, orange


def B(u, w, k=K):
    D = g.p2_eval(D_power(k), u, w)
    V = np.stack([g.p2_eval(c, u, w) for c in V_POWER], -1)
    return O(u, w) + D[..., None] * V


def curves():
    t = lambda x: np.asarray(x, float)
    z = lambda x: 0 * t(x)
    out = []
    for u0 in (0.0, 1.0):
        out += [(lambda x, u0=u0: O(u0 + z(x), t(x)), 0, 1, False, "boundary"),
                (lambda x, u0=u0: O(u0 + z(x), 1 + t(x)), 0, 1, False, "boundary")]
    out += [(lambda x: O(t(x), z(x)), 0, 1, False, "boundary"), (lambda x: O(t(x), 2 + z(x)), 0, 1, False, "boundary"),
            (lambda x: O(t(x), 1 + z(x)), 0, 1, False, "tangent")]
    return out


# ------------------------------------------------------------------------------------------- checks

def checks(k=K):
    out = {}
    uu, ww = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 2, 201), indexing="ij")
    n = unit_normal(uu, ww)
    V = np.stack([g.p2_eval(c, uu, ww) for c in V_POWER], -1)
    out["max V . n (< 0: blue on one side)"] = float((V * n).sum(-1).max())
    h = 1e-6

    def nB(u, w):
        a = (B(u + h, w) - B(u - h, w)) / (2 * h)
        b = (B(u, w + h) - B(u, w - h)) / (2 * h)
        c = np.cross(a, b)
        return c / np.linalg.norm(c, axis=-1, keepdims=True)
    t = np.linspace(0.02, 0.98, 49)
    sin = lambda u, w: np.linalg.norm(np.cross(unit_normal(u, w), nB(u, w)), axis=-1)
    out["middle edge w = 1: max sin(normal angle) (tangent, 0)"] = float(sin(t, 1 + 0 * t).max())
    tw = np.concatenate([np.linspace(0.02, 0.9, 30), np.linspace(1.1, 1.98, 30)])   # long edges, away from w = 1
    bmin = min(float(sin(0 * tw, tw).min()), float(sin(1 + 0 * tw, tw).min()),
               float(sin(t, 0 * t).min()), float(sin(t, 2 + 0 * t).min()))
    out["boundary edges: min sin(normal angle) (corners and w = 1 +- 0.1 excluded)"] = bmin
    # certification: distance from B to O versus D, on a grid of B points with D > 0
    _, orange = surfaces(k)
    tr = orange.truth()
    us, ws = np.meshgrid(np.linspace(0.03, 0.97, 33), np.linspace(0.03, 1.97, 66), indexing="ij")
    us, ws = us.ravel(), ws.ravel()
    Dv = g.p2_eval(D_power(k), us, ws)
    ok = Dv > 1e-4
    dist = g.param_distance(tr, B(us[ok], ws[ok], k))
    out["certify: min dist(B, O) / D over %d points with D > 1e-4" % ok.sum()] = float((dist / Dv[ok]).min())
    out["max separation D"] = float(Dv.max())
    return out


def build():
    blue, orange = surfaces()
    cam, rc = CAMERA, RESULT_CAMERA
    corners = [tuple(O(a_, b_)) for a_, b_ in ((0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2))]
    Y.add(CID, "Fig.1(d): two quadrilaterals sharing a tangential edge, six boundary edges", blue, orange,
          Y.SRC + "Fig. 1 (d)", "'the intersection contains two quadrilaterals sharing one edge. The two surfaces are in "
          "contact along the common edge of the two quadrilaterals, and intersect transversely at the other six curved "
          "edges of the two quadrilaterals, which are also the boundaries of the surfaces'",
          "topology: 6 transversal boundary edges + 1 tangential interior edge (D has a double zero), one connected "
          "network, the shared edge shorter than the outer edges; appearance, refitted to the 600 dpi panels: an "
          "S-wave ribbon with curled ends and a bump across, the network matching the paper's (Ours row) to "
          "~0.2 / 2.3 px; camera orthographic el %.0f az %.0f roll %.0f, %.0f px per unit (render_match.py; a "
          "gauge for the fitted net), the Surfaces row the same view shifted by (-16.1, -2.6) px; against the "
          "Surfaces panel: %s (previous wave graph under its own best-fitting camera: silhouette IoU 0.803, "
          "blue 0.000, orange 0.382)" % (rc["el"], rc["az"], rc["roll"], rc["scale"], SCORES),
          "bi-degree orange (2, 5), blue (4, 6), polynomial (not stated in the paper); the ribbon is a control net "
          "fitted to one view, so its depth is not measured (the fairest net consistent with the image); not "
          "reproduced: the paper's alternation of the visible sheet (orange on the wall and the crest, blue in the "
          "trough and the lower-right part) needs folds in depth that one view does not pin down, ours shows the "
          "blue almost everywhere; the paper's end curls are flatter here; the offset field V and the separation "
          "k are our choices",
          "paper: " + Y.OPEN + " returned three of the six edges only",
          curves=curves(), singular=corners, cov_tol=1e-2)
    case = gc.CASES[-1]
    case["notes"] = ("B = O + D V with D = k u(1-u) w(2-w)(w-1)^2: the network is exactly D = 0 (iso-curves of O); "
                     "no other common point is certified numerically (dist(B, O) >= c D)")
    case["truth_method"] = ("closed form (iso-parameter curves of the orange), surfaces exact polynomial; absence of "
                            "other intersections certified numerically")
    return case


SCORES = "silhouette IoU 0.835, blue-visible IoU 0.428, orange-visible IoU 0.008"


def write(case):
    with open(os.path.join(ROOT, "cases", "data", CID + ".json"), "w") as fh:
        json.dump(case, fh)


if __name__ == "__main__":
    case = build()
    write(case)
    for k_, v in checks().items():
        print("%-66s %s" % (k_, v))
    if "--compare" in sys.argv:
        import efg_compare as ec
        print(ec.compare(case, CAMERA, n=241, result_camera=RESULT_CAMERA))
