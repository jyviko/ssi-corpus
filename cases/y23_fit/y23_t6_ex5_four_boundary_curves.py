"""Table 6 example (5) of Yang, Jia & Yan 2023 ("four boundary curves", bi-degree (2,2)),
refitted to the paper's panels (cases/y23_fit/paper/y23_t6_ex5_four_boundary_curves_*.png).

What the panels show (600 dpi):
  * Surfaces: an orange arched sheet outlined in red. All four of its edges arch upward,
    like a vault seen obliquely: the upper outline is an 'M' (two arches meeting at a
    raised corner, the far corner), the lower outline a second 'M' (two arches meeting at
    a low V, the near corner); the left and right corners are the outline's extreme points.
    A blue dome rises over the orange; its silhouette passes above the orange's upper arches.
  * Ours / the comparison methods: the same closed chain of four arches from a different
    camera.

Construction (exact, polynomial, both patches bi-degree (2,2)):
  * orange O(u, v): a Bezier patch with a bilinear (x, y) footprint (four corner points) and
    nine free heights z_kl (Bernstein), i.e. a height field over a convex quadrilateral.
  * blue  B(u, v) = O(u, v) + 16 c u (1 - u) v (1 - v) n  (the orange with its centre
    control point moved by 4 c along a fixed unit direction n). The footprint map of O
    projected along n is injective (its Jacobian determinant, a polynomial, has Bernstein
    coefficients of one sign, and the boundary maps to a simple curve: checked in main()), so a
    line parallel to n meets O at most once and B meets O exactly where u (1-u) v (1-v) = 0:
    the four shared edges. n points away from the camera, so the orange is in front of the
    dome inside its outline and the dome shows only above the upper arches, as in the panel. Intersection = the four boundary curves (closed form), a closed chain;
    the corners are its singular points.

Run: .venv/bin/python -I cases/y23_fit/y23_t6_ex5_four_boundary_curves.py  (writes cases/data/<id>.json)
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

CID = "y23_t6_ex5_four_boundary_curves"

# corners (x, y) of the footprint in Bernstein order (0,0), (1,0), (0,1), (1,1); heights z[k][l]; dome c
SHAPE = {"corners": [[-0.691649, -0.882472], [0.674659, -0.982305], [-0.930793, 0.682983], [0.762701, 1.049205]], "z": [[-0.019715, -0.686513, -0.01985], [-0.729877, -1.095315, -0.844199], [-0.02065, -0.921137, -0.020356]], "c": 0.24127, "n": [0.098872, 0.509631, -0.880715], "dxy": [0.009945, 0.010076], "offset": "blue"}
CAM_SURFACES = {"el": 17.017621, "az": 54.497043, "roll": 195.169377, "scale": 206.118294, "cx": 297.812373, "cy": 284.667089, "proj": "ortho"}
CAM_OURS = {"el": -21.357835, "az": 55.947723, "roll": -186.25041, "scale": 209.485317, "cx": 297.284304, "cy": 201.17557, "proj": "ortho"}
FIT = {"silhouette_iou_surfaces": 0.9482, "silhouette_iou_before": 0.7366, "curve_chamfer_px_surfaces": 2.0, "curve_chamfer_px_result_view": 1.6}


def nets(s):
    c00, c10, c01, c11 = (np.array(p, float) for p in s["corners"])
    t = np.array([0.0, 0.5, 1.0])
    xy = ((1 - t)[:, None, None] * (1 - t)[None, :, None] * c00 + t[:, None, None] * (1 - t)[None, :, None] * c10
          + (1 - t)[:, None, None] * t[None, :, None] * c01 + t[:, None, None] * t[None, :, None] * c11)
    xy[1, 1] += np.array(s.get("dxy", (0.0, 0.0)), float)
    return xy, np.array(s["z"], float)


def base(s):
    """The patch S0 that carries the four shared edges (power arrays X, Y, Z)."""
    xy, z = nets(s)
    return C.bern_pow(xy[..., 0]), C.bern_pow(xy[..., 1]), C.bern_pow(z)


def surfaces(s):
    """(orange, blue): one of them is S0, the other S0 + 16 c u(1-u) v(1-v) n."""
    S0 = base(s)
    mu = 16 * s["c"] * g.p2_mul([[0.0], [1.0], [-1.0]], [[0.0, 1.0, -1.0]])
    n = direction(s)
    S1 = tuple(g.p2_add(S0[k], n[k] * mu) for k in range(3))
    return (S1, S0) if s.get("offset") == "orange" else (S0, S1)


def direction(s):
    n = np.array(s.get("n", (0.0, 0.0, 1.0)), float)
    return n / np.linalg.norm(n)


def curve_fns(s):
    O = base(s)
    ev = lambda uf, vf: (lambda t: np.column_stack([C.peval(c, uf(t), vf(t)) for c in O]))
    z0, z1 = (lambda t: 0 * t), (lambda t: 0 * t + 1)
    idt, rev = (lambda t: t), (lambda t: 1 - t)
    # closed chain (0,0) -> (1,0) -> (1,1) -> (0,1) -> (0,0)
    return [(ev(idt, z0), 0.0, 1.0), (ev(z1, idt), 0.0, 1.0), (ev(rev, z1), 0.0, 1.0), (ev(z0, rev), 0.0, 1.0)]


def curves(s, n=200):
    return [f(np.linspace(t0, t1, n)) for f, t0, t1 in curve_fns(s)]


def corners3(s):
    O = base(s)
    return [[float(C.peval(c, u, v)) for c in O] for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))]


def footprint_jacobian(s):
    """Power coefficients of the Jacobian determinant of the footprint map of S0 along n."""
    return C.footprint_jacobian(base(s), direction(s))


def certify_footprint(s, depth=7):
    return C.certify_footprint(base(s), direction(s), depth)


def build(s=None):
    s = s or SHAPE
    Po, Pb = surfaces(s)
    return C.patch(Pb, "dome (the orange with its centre control point raised)"), C.patch(Po, "arched sheet")


def main():
    s = SHAPE
    sgn, worst = certify_footprint(s)
    blue, orange = build(s)
    fit_txt = ("camera (Surfaces panel) %s; camera (result rows) %s; %s" %
               (json.dumps(CAM_SURFACES), json.dumps(CAM_OURS), json.dumps(FIT)))
    Y.add(CID, "Table 6 ex.5 (2,2): four boundary curves (arched sheet under a dome)", blue, orange,
          Y.SRC + "Table 6 (example (5))", "'four boundary curves', bi-degree (2,2)",
          "bi-degree (2,2) both; topology: the four patch boundary curves (both patches share them), closed "
          "chain, four corners; shape: an orange sheet whose four edges all arch upward (upper 'M' meeting at the "
          "raised far corner, lower 'M' meeting at the low near corner) under a blue dome that rises above it; "
          "fitted to the Surfaces panel and to the result-row view. " + fit_txt,
          "exact control points (the paper publishes none); the footprint is taken bilinear and the dome is "
          "the orange with its centre control point raised vertically",
          "paper: " + Y.SPL + " returned nothing; " + Y.OPEN + " partial",
          curves=[(f, t0, t1, False, "boundary") for f, t0, t1 in curve_fns(s)],
          singular=corners3(s), cov_tol=1e-2)
    case = gc.CASES[-1]
    with open(os.path.join(gc.OUT, CID + ".json"), "w") as fh:
        json.dump(case, fh)
    print(CID, "branches", len(case["expected"]["branches"]), "footprint along n certified injective (Jacobian sign %+d, min Bernstein coeff %.4g)" % (sgn, worst))


if __name__ == "__main__":
    main()
