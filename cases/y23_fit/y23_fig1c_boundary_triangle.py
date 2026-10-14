"""Fig. 1(c) of Yang, Jia & Yan 2023, refitted to the paper's panels.

What the panels show (cases/y23_fit/paper/y23_fig1c_boundary_triangle_*.png, 600 dpi crops):
  * the orange is a four-sided sheet seen obliquely from above, all four edges bowed inward
    (diamond-like outline: top, left, bottom and right corners).
  * the red triangle: top corner -> left corner and left corner -> bottom corner along the orange's
    edges, closed by a long, slightly curved diagonal top -> bottom across the orange.
  * the blue lies under the orange inside the triangle and rises above it beyond the diagonal, as a
    strip from the top corner to the bottom corner that ends in its own free edge (the paper's strip
    is S-shaped with a cusp; see below).
  * the Ours row is the same view, about 4 % smaller.

Construction (exact, polynomial; orange bi-degree (2, 2), blue (6, 2); the paper states no degree):
  parameters (u, w) in [0, 1]^2, U = 2u - 1, V = 2w - 1;
  * orange O(u, w) = (X, Y, q):  X = U (1 - p (1 - V^2)), Y = L V (1 - p (1 - U^2)) (edges bowed inward),
    q = a20 U^2 + a11 U V + a02 V^2 (a shallow saddle);
  * blue B(s, t) = O(u, w) + D(u, w) e_z at u = s, w = t g(s), g = (1 - s)(1 + e1 s), (s, t) in [0, 1]^2:
    a curved triangle in the parameter square (edges u = 0, w = 0, and w = g(u) beyond the diagonal; the
    edge s = 1 collapses into the bottom corner), D = k u w F, F = u + w + a u w - 1.
  * the plan map (u, w) -> (X, Y) is injective (Jacobian > 0, checked), so B lies on O exactly where
    D = 0: the shared boundary edges u = 0 and w = 0 (B's and O's own edges: 'common boundary curves')
    and the interior curve F = 0, w = (1 - u) / (1 + a u) (transversal).

Fitted camera (render_match.py convention, fit_y23_fig1c_boundary_triangle.py): perspective, el 55.64,
az 144.27, roll 154.11, eye 3.96 units from the centre, focal 709 px; the Ours row with focal 679 px.

Run: .venv/bin/python -I cases/y23_fit/y23_fig1c_boundary_triangle.py [--compare]
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

CID = "y23_fig1c_boundary_triangle"

# fit_y23_fig1c_boundary_triangle.py (reduced variant: no linear, U^2 V^2, cubic-edge or steepening terms),
# k held at 3 so that the sheets stay within 0.10 of each other inside the triangle (k = 5.7 fits 2 % better)
PARAMS = dict(L=0.84477, a20=-0.15958, a11=-0.21819, a02=0.10869, a=0.49809, k=3.0, e1=0.60291, pinch=0.10064)
CAMERA = dict(proj="persp", el=55.6399, az=144.268, roll=154.1136, scale=709.19, dist=3.9648, cx=230.749,
              cy=207.974, target=[0.0, 0.0, 0.0])
RESULT_CAMERA = dict(CAMERA, roll=154.1241, scale=679.44, cx=230.751, cy=207.970)

pm, pa = g.p2_mul, g.p2_add


def sc(c, a):
    return np.asarray(g.p2(a), float) * c


def orange_hom(p, Uc, Vc):
    """X, Y, Z power arrays given U, V as power arrays in the patch parameters."""
    one = np.ones((1, 1))
    U2, V2 = pm(Uc, Uc), pm(Vc, Vc)
    X = pm(Uc, pa(one, sc(-p["pinch"], pa(one, sc(-1, V2)))))
    Yv = sc(p["L"], pm(Vc, pa(one, sc(-p["pinch"], pa(one, sc(-1, U2))))))
    Z = pa(pa(sc(p["a20"], U2), sc(p["a11"], pm(Uc, Vc))), sc(p["a02"], V2))
    return X, Yv, Z


def D_poly(p, uc, wc):
    one = np.ones((1, 1))
    F = pa(pa(pa(uc, wc), sc(p["a"], pm(uc, wc))), sc(-1, one))
    return sc(p["k"], pm(pm(uc, wc), F))


def surfaces(p=PARAMS):
    # orange in (u, w)
    u = np.array([[0.0], [1.0]])
    w = np.array([[0.0, 1.0]])
    Uc, Vc = pa(sc(2, u), [[-1.0]]), pa(sc(2, w), [[-1.0]])
    X, Yv, Z = orange_hom(p, Uc, Vc)
    orange = g.rational_poly_patch(X, Yv, Z, [[1.0]], (0.0, 1.0), (0.0, 1.0),
                                   "saddle sheet with inward-bowed edges (X, Y, q)(u, w)")
    # blue in (s, t): u = s, w = t g(s)
    s = np.array([[0.0], [1.0]])
    gs = np.array([[1.0], [p["e1"] - 1.0], [-p["e1"]]])          # (1 - s)(1 + e1 s)
    wt = pm(gs, np.array([[0.0, 1.0]]))
    Uc, Vc = pa(sc(2, s), [[-1.0]]), pa(sc(2, wt), [[-1.0]])
    Xb, Yb, Zb = orange_hom(p, Uc, Vc)
    Zb = pa(Zb, D_poly(p, s, wt))
    blue = g.rational_poly_patch(Xb, Yb, Zb, [[1.0]], (0.0, 1.0), (0.0, 1.0),
                                 "curved-triangle sheet O(s, t g(s)) + D e_z, D = k u w (u + w + a u w - 1)")
    return blue, orange


def O(u, w, p=PARAMS):
    U, V = 2 * np.asarray(u, float) - 1, 2 * np.asarray(w, float) - 1
    X = U * (1 - p["pinch"] * (1 - V * V))
    Yv = p["L"] * V * (1 - p["pinch"] * (1 - U * U))
    Z = p["a20"] * U * U + p["a11"] * U * V + p["a02"] * V * V
    return np.stack(np.broadcast_arrays(X, Yv, Z), -1)


def curves(p=PARAMS):
    t0 = lambda t: 0 * np.asarray(t, float)
    return [(lambda t: O(t0(t), t, p), 0, 1, False, "boundary"),
            (lambda t: O(t, t0(t), p), 0, 1, False, "boundary"),
            (lambda t: O(t, (1 - np.asarray(t, float)) / (1 + p["a"] * np.asarray(t, float)), p), 0, 1, False,
             "crossing")]


# ------------------------------------------------------------------------------------------- checks

def normal(fn, u, w, h=1e-6):
    du = (fn(u + h, w) - fn(u - h, w)) / (2 * h)
    dw = (fn(u, w + h) - fn(u, w - h)) / (2 * h)
    n = np.cross(du, dw)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def checks(p=PARAMS):
    out = {}
    uu, ww = np.meshgrid(np.linspace(0, 1, 401), np.linspace(0, 1, 401), indexing="ij")
    h = 1e-6
    Xu = (O(uu + h, ww, p) - O(uu - h, ww, p)) / (2 * h)
    Xw = (O(uu, ww + h, p) - O(uu, ww - h, p)) / (2 * h)
    jac = Xu[..., 0] * Xw[..., 1] - Xu[..., 1] * Xw[..., 0]
    out["plan map Jacobian min (> 0: injective plan view)"] = float(jac.min())
    gs = lambda s: (1 - s) * (1 + p["e1"] * s)
    s = np.linspace(0, 1, 2001)
    diag = (1 - s) / (1 + p["a"] * s)
    out["blue edge beyond the diagonal: min (g - diag) on (0, 1)"] = float((gs(s) - diag)[1:-1].min())
    out["blue domain inside the square: max g"] = float(gs(s).max())
    Dk = lambda u, w: p["k"] * u * w * (u + w + p["a"] * u * w - 1)
    B = lambda u, w: O(u, w, p) + Dk(u, w)[..., None] * np.array([0, 0, 1.0])
    tri = (uu + ww + p["a"] * uu * ww < 1)
    out["max |D| inside the triangle"] = float(np.abs(Dk(uu, ww))[tri].max())
    ext = np.ptp(O(uu, ww, p).reshape(-1, 3), axis=0)
    out["orange extent (x, y, z)"] = [round(float(e), 4) for e in ext]
    t = np.linspace(0.02, 0.98, 49)
    for name, (uu_, ww_) in {"edge u=0": (0 * t, t), "edge w=0": (t, 0 * t),
                             "diagonal": (t, (1 - t) / (1 + p["a"] * t))}.items():
        nO = normal(lambda a_, b_: O(a_, b_, p), uu_, ww_)
        nB = normal(B, uu_, ww_)
        out[name + ": min sin(normal angle)"] = float(np.linalg.norm(np.cross(nO, nB), axis=-1).min())
    return out


def build():
    blue, orange = surfaces()
    p = PARAMS
    cam = CAMERA
    Y.add(CID, "Fig.1(c): curved triangle, two edges on the shared patch boundaries", blue, orange,
          Y.SRC + "Fig. 1 (c)", "'the two surfaces are very close to each other in the triangular domain, and the "
          "intersection curve is a curved triangle, two edges of which are the boundary curves of the two surfaces and "
          "the left edge is a regular intersection curve'",
          "topology: closed curved triangle; two edges on the shared patch boundaries (u = 0, w = 0 of both), one "
          "interior transversal diagonal; appearance, refitted to the 600 dpi panels: orange a shallow saddle with "
          "all four edges bowed inward, seen obliquely from above; the blue under it inside the triangle and rising "
          "above it beyond the diagonal as a strip from the top to the bottom corner; fitted camera perspective el "
          "%.2f az %.2f roll %.2f, eye %.2f units from the centre, focal %.0f px (render_match.py); against the "
          "Surfaces panel: silhouette IoU 0.951, blue-visible IoU 0.776, orange-visible IoU 0.851 (previous quadratic "
          "graphs under their own best-fitting camera: 0.921, 0.158, 0.340); triangle within ~0.1 / 2.2 px of the paper's red curves" % (
              cam["el"], cam["az"], cam["roll"], cam["dist"], cam["scale"]),
          "bi-degree orange (2, 2), blue (6, 2), polynomial (not stated in the paper); the saddle, the inward bow "
          "of the edges and the blue's free edge are fitted to the outline, not known. Not reproduced: the "
          "paper's blue strip curls into an S with a cusp in the middle (a fold of the blue beyond the diagonal); "
          "ours is a smooth lens-shaped strip. 'Very close' inside the triangle: |D| <= 0.10 here against a "
          "2.0 x 1.7 sheet (the fit prefers a larger D, which also widens the strip)",
          "paper: " + Y.OPEN + " lost the two boundary edges",
          curves=curves(), singular=[tuple(O(a_, b_)) for a_, b_ in ((0, 0), (1, 0), (0, 1))])
    case = gc.CASES[-1]
    case["notes"] = ("B(u, w) = O(u, w) + D(u, w) e_z over an injective plan map, so the intersection is exactly "
                     "D = k u w (u + w + a u w - 1) = 0 inside the blue's domain")
    case["truth_method"] = "closed form (parametric curves on the orange), surfaces exact polynomial"
    return case


def write(case):
    with open(os.path.join(ROOT, "cases", "data", CID + ".json"), "w") as fh:
        json.dump(case, fh)


if __name__ == "__main__":
    case = build()
    write(case)
    for k, v in checks().items():
        print("%-62s %s" % (k, v))
    if "--compare" in sys.argv:
        import efg_compare as ec
        print(ec.compare(case, CAMERA, n=241, result_camera=RESULT_CAMERA))
