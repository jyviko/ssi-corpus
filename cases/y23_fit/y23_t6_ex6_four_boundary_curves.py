"""Table 6 example (6) of Yang, Jia & Yan 2023 ("four boundary curves", bi-degree (2,2)),
refitted to the paper's panels (cases/y23_fit/paper/y23_t6_ex6_four_boundary_curves_*.png).

What the panels show (600 dpi):
  * Surfaces: an orange trough seen from the front and above. Its two raised top corners A
    (left) and B (right) carry short straight, nearly vertical edges A-D and B-C down to the
    lower corners. Its far top edge A-B sags gently (about 50 px over a 457 px span); its near
    lower edge D-C sags about twice as much. A blue dome sits in the trough: it bulges above
    the far edge, and its near edge is a deep arc A-B lying on the trough (sag ~143 px).
  * Ours / the comparison methods (a different camera): the intersection is a lens of two
    arcs that share both end points, plus one straight segment hanging from each end point.
    The panels are close to left-right mirror symmetric; the fit allows a small tilt b and ruling
    skew dx, which the right corner standing higher than the left calls for.

Construction (exact, polynomial, both patches bi-degree (2,2)):
  * trough (orange) on the parabolic cylinder through C(x) = (x, 0, a x^2 + b x) with rulings
    d = (dx, -L, h):  O(u, v) = C(x) + v t1(u) d,  x = 2u - 1,  t1 = 1 + k1 (1 - x^2)
    (degree (2,1), elevated). The far edge v = 0 is C itself; the near edge v = 1 sags more
    for k1 > 0 while the patch stays on the same cylinder.
  * dome (blue):  B(u, v) = C(x) + tau(u, v) d + mu(u, v) e_z with
      tau = 2 v (1 - v) (2 + kt (1 - x^2)) + v^2 kf (1 - x^2),   mu = 16 c u (1 - u) v (1 - v).
    A point C(x) + tau d + mu e_z lies on the cylinder iff mu = 0 (the vertical line through a
    cylinder point meets the cylinder once: e_z is the parabola's axis direction and is
    parallel to the plane y = 0 onto which the rulings project). mu > 0 inside,
    so the dome meets the trough exactly on its own four edges:
      v = 0:  C(x)                         = the trough's far edge A-B,
      v = 1:  C(x) + kf (1 - x^2) d        = the deep near arc A-B (inside the trough),
      u = 0, 1: C(-+1) + 4 v (1 - v) d      = the trough's side edges A-D, B-C (traversed
                                             out and back: the dome edge folds at D, C).
    Intersection: the far arc, the near arc, A-D and B-C (closed form). The two arcs meet the
    two segments at A and B (singular points); D and C are free curve ends.

Run: .venv/bin/python -I cases/y23_fit/y23_t6_ex6_four_boundary_curves.py  (writes cases/data/<id>.json)
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

CID = "y23_t6_ex6_four_boundary_curves"

# fitted shape (fit_t6_ex6.py) and cameras (render_match.py convention)
SHAPE = {"a": 0.383485, "L": 1.113876, "h": -0.109749, "k1": 0.305753, "kf": 0.657365, "kt": 0.415894, "c": 1.058405, "b": 0.031246, "dx": 0.0165}
CAM_SURFACES = {"el": 18.245815, "az": -92.956635, "roll": -0.000848, "scale": 226.363978, "cx": 272.437389, "cy": 197.463701, "proj": "ortho"}
CAM_OURS = {"el": 18.661556, "az": -89.910642, "roll": -24.862362, "scale": 225.889056, "cx": 298.851038, "cy": 251.253729, "proj": "ortho"}
FIT = {"silhouette_iou_surfaces": 0.9406, "silhouette_iou_before": 0.7171, "curve_chamfer_px_surfaces": 3.7, "curve_chamfer_px_result_view": 2.0}

X = np.array([[-1.0], [2.0]])           # x = 2u - 1
Q = np.array([[0.0], [4.0], [-4.0]])    # 4u(1-u) = 1 - x^2


def _lin(*terms):
    out = np.zeros((1, 1))
    for t in terms:
        out = g.p2_add(out, t)
    return out


def surfaces(s):
    a, L, h, k1, kf, kt, c = (s[k] for k in ("a", "L", "h", "k1", "kf", "kt", "c"))
    b, dx = s.get("b", 0.0), s.get("dx", 0.0)
    x2 = _lin(a * g.p2_mul(X, X), b * X)
    t1 = _lin([[1.0]], k1 * Q)
    to = g.p2_mul(t1, [[0.0, 1.0]])
    O = (_lin(X, dx * to), -L * to, _lin(x2, h * to))
    tau1 = _lin([[2.0]], kt * Q)
    tau = _lin(g.p2_mul(tau1, [[0.0, 2.0, -2.0]]), g.p2_mul(kf * Q, [[0.0, 0.0, 1.0]]))
    mu = 16 * c * g.p2_mul([[0.0], [1.0], [-1.0]], [[0.0, 1.0, -1.0]])
    B = (_lin(X, dx * tau), -L * tau, _lin(x2, h * tau, mu))
    return O, B


def curve_fns(s):
    a, L, h, kf = s["a"], s["L"], s["h"], s["kf"]
    b, dx = s.get("b", 0.0), s.get("dx", 0.0)
    cz = lambda x: a * x * x + b * x
    side = lambda x0: (lambda t: np.column_stack([x0 + dx * t, -L * t, cz(x0) + h * t]))
    far = lambda t: np.column_stack([t, 0 * t, cz(t)])
    near = lambda t: np.column_stack([t + dx * kf * (1 - t * t), -L * kf * (1 - t * t), cz(t) + h * kf * (1 - t * t)])
    return [(far, -1.0, 1.0), (near, -1.0, 1.0), (side(-1.0), 0.0, 1.0), (side(1.0), 0.0, 1.0)]


def curves(s, n=200):
    return [f(np.linspace(t0, t1, n)) for f, t0, t1 in curve_fns(s)]


def build(s=None):
    s = s or SHAPE
    Po, Pb = surfaces(s)
    orange = C.patch(Po, "trough (parabolic cylinder)")
    blue = C.patch(Pb, "dome sitting in the trough, its four edges on the trough")
    return blue, orange


def checks(s=None):
    """Numbers for the report: dome height above the cylinder, edge residuals."""
    s = s or SHAPE
    Po, Pb = surfaces(s)
    u, v = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 1, 201), indexing="ij")
    Bx, By, Bz = (C.peval(c, u, v) for c in Pb)
    # vertical offset of the dome above the cylinder through C(x) = (x, 0, a x^2 + b x), rulings d
    t = By / -s["L"]
    x0 = Bx - s.get("dx", 0.0) * t
    off = Bz - (s["a"] * x0 ** 2 + s.get("b", 0.0) * x0 + s["h"] * t)
    inner = (u > 0) & (u < 1) & (v > 0) & (v < 1)
    edge = ~inner
    return dict(min_interior_offset=float(off[inner].min()), max_edge_offset=float(np.abs(off[edge]).max()),
                max_offset=float(off.max()))


def main():
    s = SHAPE
    blue, orange = build(s)
    a, b = s["a"], s.get("b", 0.0)
    A, B = (-1.0, 0.0, a - b), (1.0, 0.0, a + b)
    fit_txt = ("camera (Surfaces panel) %s; camera (result rows) %s; %s" %
               (json.dumps(CAM_SURFACES), json.dumps(CAM_OURS), json.dumps(FIT)))
    Y.add(CID, "Table 6 ex.6 (2,2): four boundary curves (dome in a trough: lens of two arcs + two side edges)",
          blue, orange, Y.SRC + "Table 6 (example (6))", "'four boundary curves', bi-degree (2,2)",
          "bi-degree (2,2) both; topology: four boundary curves - the dome's four edges all lie on the trough: "
          "the trough's far edge, a deep near arc inside the trough, and the trough's two straight side edges; "
          "the two arcs share both end points (a lens) and a straight segment hangs from each (as in the paper's "
          "result rows); orange trough with raised corners and short straight sides, blue dome sitting in it; "
          "left-right mirror symmetry; fitted to the Surfaces panel and to the result-row view. " + fit_txt,
          "exact control points (the paper publishes none): the trough is taken to be a parabolic cylinder "
          "(required for the near arc to lie exactly on it at degree 2), so its near lower edge sags like a "
          "translate of the far edge plus a quadratic dip",
          "paper: " + Y.SPL + " returned an error",
          curves=[(f, t0, t1, False, "boundary") for f, t0, t1 in curve_fns(s)],
          singular=[A, B], cov_tol=1e-2)
    case = gc.CASES[-1]
    with open(os.path.join(gc.OUT, CID + ".json"), "w") as fh:
        json.dump(case, fh)
    print(CID, "branches", len(case["expected"]["branches"]), checks(s))


if __name__ == "__main__":
    main()
