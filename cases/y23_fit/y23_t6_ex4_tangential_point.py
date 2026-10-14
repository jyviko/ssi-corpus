"""Table 6 example (4) of Yang, Jia & Yan 2023 ("a tangential point", bi-degree (2,2)),
refitted to the paper's panels (cases/y23_fit/paper/y23_t6_ex4_tangential_point_*.png).

What the panels show (600 dpi):
  * Surfaces: a large blue sheet seen almost face on. Its outline is a curved quadrilateral:
    top edge sagging slightly, bottom edge bowed slightly up, left edge nearly straight, right
    edge strongly concave. Shading runs light (lower left) to dark (upper right).
  * a smaller orange sheet in front of it, near the centre: left edge nearly straight, top and
    bottom edges bulging outward (about 37 px over a 190 px chord), right edge concave like
    the blue one. A vertical highlight band: curved about a vertical axis.
  * the red tangential point lies inside the orange, left of its centre (38 % across, half-way
    down). The result rows show only the dot; the comparison method panels use the same view.
  The paper's renders here are 2D-like (one view, no depth cue other than shading), so the
  geometry is built in view coordinates: x right, y up (pixels / 100), z toward the viewer.

Construction (exact, polynomial, both bi-degree (2,2)):
  * blue  B(u, v): a Bezier patch with a free 3x3 (x, y) net (the outline) lying in the plane
    z = alpha x + beta y (heights z_ij = alpha x_ij + beta y_ij reproduce the plane exactly).
  * orange O(s, t): a free 3x3 (x, y) net (its outline) with heights
        z = alpha x + beta y + mu(s, t),   mu = k [(s - s0)^2 a(t) + (t - t0)^2 b(s)],
    a, b quadratics with positive Bernstein coefficients, k > 0. So mu >= 0 with a single zero
    at (s0, t0), where grad mu = 0 and the Hessian is positive definite: the orange touches the
    plane at exactly one point, with the plane as its tangent plane there, and lies in front of
    it (toward the viewer) everywhere else. The contact point's footprint is inside the blue
    outline (checked), so the intersection is that single tangential point (closed form).
  * The face-on panel pins the outlines but not the depth; a planar blue is the price of
    keeping both outlines free and the truth exact (the paper's blue shading suggests a slight
    curvature that this construction does not carry). The orange's curvature (k, a, b) is
    chosen so its shading shows the paper's vertical highlight band.

Run: .venv/bin/python -I cases/y23_fit/y23_t6_ex4_tangential_point.py  (writes cases/data/<id>.json)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t6_ex456_common as C  # noqa: E402
import gen_yang2023 as Y  # noqa: E402  (read-only helpers)
import gen_cases as gc  # noqa: E402
import ssi_geom as g  # noqa: E402

CID = "y23_t6_ex4_tangential_point"


def _net_from_px(corners, mids, centre):
    """Bernstein (x, y) net from pixel corners/edge midpoints/centre point (view coordinates)."""
    TL, TR, BR, BL = (np.array(p, float) for p in corners)
    mt, mr, mb, ml = (np.array(p, float) for p in mids)
    N = np.zeros((3, 3, 2))  # [i along u = left->right][j along v = bottom->top]
    N[0, 0], N[2, 0], N[2, 2], N[0, 2] = BL, BR, TR, TL
    N[1, 2] = 2 * mt - 0.5 * (TL + TR)
    N[1, 0] = 2 * mb - 0.5 * (BL + BR)
    N[0, 1] = 2 * ml - 0.5 * (TL + BL)
    N[2, 1] = 2 * mr - 0.5 * (TR + BR)
    c = np.array(centre, float)
    N[1, 1] = 4 * c - (N[0, 0] + N[2, 0] + N[2, 2] + N[0, 2]) / 4 - (N[1, 2] + N[1, 0] + N[0, 1] + N[2, 1]) / 2
    N = N / 100.0
    N[..., 1] *= -1  # pixel y down -> view y up
    return N


SHAPE = {"xy": [[[0.585068, -4.333621], [0.509593, -2.311593], [0.669593, -0.459544]], [[2.443775, -4.072479], [2.683173, -2.230938], [2.606752, -0.802878]], [[4.352899, -4.311293], [3.775453, -2.642078], [4.357476, -0.565903]]], "oxy": [[[1.552345, -3.378542], [1.413697, -2.416558], [1.564005, -1.463028]], [[2.315583, -4.142466], [2.681395, -2.343542], [2.312151, -0.687855]], [[3.405102, -3.332677], [3.019573, -2.454211], [3.414793, -1.522873]]], "plane": [0.35, -0.15], "st0": [0.395455, 0.493686], "a": [1.0, 1.0, 1.0], "b": [0.2, 0.2, 0.2], "k": 1.5}
CAMERA = dict(proj="ortho", el=90.0, az=-90.0, roll=0.0, scale=100.0, cx=0.0, cy=0.0)
FIT = {"silhouette_iou_surfaces": 0.9931, "silhouette_iou_before": 0.962, "orange_region_iou": 0.992, "contact_point_offset_px": 0.0}

M2 = C.M2


def _bern1(c):
    """1D quadratic Bernstein coefficients -> power coefficients."""
    return np.asarray(c, float) @ M2


def plane_z(s, X, Yc):
    al, be = s["plane"]
    return g.p2_add(al * X, be * Yc)


def mu_power(s):
    s0, t0 = s["st0"]
    a = _bern1(s["a"])[None, :]  # a(t): row
    b = _bern1(s["b"])[:, None]  # b(s): column
    ss = np.array([[s0 * s0], [-2 * s0], [1.0]])
    tt = np.array([[t0 * t0, -2 * t0, 1.0]])
    return s["k"] * g.p2_add(g.p2_mul(ss, a), g.p2_mul(tt, b))


def surfaces(s):
    """(blue, orange) power arrays."""
    N = np.array(s["xy"], float)
    X, Yc = C.bern_pow(N[..., 0]), C.bern_pow(N[..., 1])
    B = (X, Yc, plane_z(s, X, Yc))
    N = np.array(s["oxy"], float)
    X, Yc = C.bern_pow(N[..., 0]), C.bern_pow(N[..., 1])
    O = (X, Yc, g.p2_add(plane_z(s, X, Yc), mu_power(s)))
    return B, O


def contact(s):
    """(point, orange (s0, t0), blue (u, v) found by Newton on the footprint)."""
    B, O = surfaces(s)
    s0, t0 = s["st0"]
    P = np.array([C.peval(c, s0, t0) for c in O], float)
    u, v = 0.5, 0.5
    for _ in range(60):
        r = np.array([C.peval(B[k], u, v) - P[k] for k in (0, 1)])
        J = np.array([[C.peval(C.pd(B[k], 0), u, v), C.peval(C.pd(B[k], 1), u, v)] for k in (0, 1)])
        du = np.linalg.solve(J, -r)
        u, v = u + du[0], v + du[1]
    return P, (s0, t0), (float(u), float(v))


def checks(s):
    """Contact inside the blue patch, on the plane; normals parallel; positivity of a, b."""
    P, (s0, t0), (u, v) = contact(s)
    B, O = surfaces(s)
    Q = np.array([C.peval(c, u, v) for c in B])
    nB = np.cross(*[np.array([C.peval(C.pd(c, k), u, v) for c in B]) for k in (0, 1)])
    nO = np.cross(*[np.array([C.peval(C.pd(c, k), s0, t0) for c in O]) for k in (0, 1)])
    sin_ang = np.linalg.norm(np.cross(nB, nO)) / (np.linalg.norm(nB) * np.linalg.norm(nO))
    return dict(blue_uv=[u, v], point_gap=float(np.linalg.norm(P - Q)), normal_sin=float(sin_ang),
                a_min=float(min(s["a"])), b_min=float(min(s["b"])), k=float(s["k"]))


def build(s=None):
    s = s or SHAPE
    Pb, Po = surfaces(s)
    return C.patch(Pb, "large planar sheet"), C.patch(Po, "smaller curved sheet touching it at one point")


def main():
    s = SHAPE
    assert min(s["a"]) > 0 and min(s["b"]) > 0 and s["k"] > 0
    ck = checks(s)
    assert 0 < ck["blue_uv"][0] < 1 and 0 < ck["blue_uv"][1] < 1 and ck["point_gap"] < 1e-12, ck
    sgn, worst = C.certify_footprint(surfaces(s)[0], (0.0, 0.0, 1.0))  # blue footprint injective
    blue, orange = build(s)
    P, _, _ = contact(s)
    fit_txt = "camera %s; %s" % (json.dumps(CAMERA), json.dumps(FIT))
    Y.add(CID, "Table 6 ex.4 (2,2): a tangential point", blue, orange, Y.SRC + "Table 6 (example (4))",
          "'a tangential point', bi-degree (2,2)",
          "bi-degree (2,2) both; topology: one isolated tangential point, interior to both sheets, left of the "
          "smaller sheet's centre (normals parallel there); outlines of both sheets fitted to the Surfaces panel "
          "(blue: sagging top, bowed bottom, straight left, concave right; orange: bulging top and bottom, straight "
          "left, concave right) in view coordinates. " + fit_txt,
          "depth (the panel is a face-on view and pins only the outlines and the shading): the blue is taken "
          "planar, the orange a bowl touching it; exact control points",
          "paper: " + Y.OPEN + " returned nothing, " + Y.SPL + " scattered points",
          points=[P.tolist()], cov_tol=1e-2)
    case = gc.CASES[-1]
    with open(os.path.join(gc.OUT, CID + ".json"), "w") as fh:
        json.dump(case, fh)
    print(CID, "blue footprint certified injective (sign %+d, min coeff %.3g)" % (sgn, worst), ck)


if __name__ == "__main__":
    main()
