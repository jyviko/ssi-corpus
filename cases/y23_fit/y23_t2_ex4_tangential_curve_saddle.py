"""Table 2 ex.4 of Yang, Jia & Yan 2023 ("a tangential curve", bi-degree (3,3)), refitted to the paper's figures.

What the paper shows (cases/y23_fit/paper/y23_t2_ex4_tangential_curve_saddle_*.png, 600 dpi):
  * Surfaces panel: two sheets whose union looks like an hourglass lying on its side. Each sheet is a
    slanted band: orange from upper-left to lower-right, blue from lower-left to upper-right. Both side
    edges of each band are S-shaped (inflected) and vertical at the waist, where the two bands have the
    same horizontal extent; the four side edges meet pairwise at the two waist points. The bottom edges
    are level; the top edges rise to the right (depth seen from above), the orange top edge has a hump.
  * the red curve runs across the waist from one waist point to the other, gently wavy.
  * above the curve the orange is in front, below it the blue is in front (translucent panel: the
    overlap is orange-dominated above, blue-dominated below; the opaque AA panel shows the same): the
    sheets cross through each other at the contact, so the contact is of odd order.
  * result row: the curve alone, an inflected wave (seen from another camera).

Construction (exact), (s, t) in [-1, 1] x [t0, t1] per sheet, Z = t vertical, X horizontal, Y depth:
    orange O(s,t) = (s - k t^3, h(s,t), t)
    blue   B(s,t) = (s + k t^3, h(s,t) + t^3 m(s), t)
  with h polynomial of bi-degree <= (3,3), h_s >= eta > 0 on the domain (each sheet is a graph over the
  (X, Z) plane), m(s) polynomial with |m| < 2 k eta.
  Intersection: Z gives t' = t; X gives s' = s - 2 k t^3; then Y needs
  h(s,t) - h(s - 2kt^3, t) = t^3 m(s'), i.e. (t = 0) or 2 k h_s(xi, t) = m(s') for some xi (mean value
  theorem), which |m| < 2 k eta rules out. So the intersection is exactly the curve t = 0:
      (s, h(s, 0), 0),  s in [-1, 1]  (both sheets share the s-range at t = 0),
  from waist point to waist point. Along it O_s = B_s = (1, h_s, 0) and O_t = B_t = (0, h_t, 1): the
  tangent planes agree (normals parallel). At equal (X, Z) the depth gap is 2 k h_s t^3 + m t^3 + O(t^4),
  which changes sign across t = 0: second-order contact, the sheets cross (as drawn).
  The curve is planar (Z = 0) and inflected in the depth direction: h(s, 0) has an inflection.
  Bi-degree (3,3) for both sheets (X is (1,3), Y (3,3), Z (0,1)), polynomial.

Run: .venv/bin/python -I cases/y23_fit/y23_t2_ex4_tangential_curve_saddle.py   (writes cases/data/<id>.json)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "cases"))
sys.path.insert(0, os.path.join(ROOT, "scoring"))
import gen_yang2023 as Y  # noqa: E402  (read-only helpers)
import gen_cases as gc  # noqa: E402
import ssi_geom as g  # noqa: E402

CID = "y23_t2_ex4_tangential_curve_saddle"

# h(s,t) = sum H[i][j] s^i t^j (i = 1..3 carry the s-dependence; i = 0 a depth offset along t)
PARAMS = dict(k=0.52256, to0=-1.3012, to1=1.46856, tb0=-1.3958, tb1=1.296,
              H=[[0, -0.1095, 0.05741, 0], [-0.3013, -0.09043, -0.0013, 0], [-0.02234, -0.00288, 0, 0], [0.06, 0.0288, 0, 0]],
              m=[-0.0018, -0.01869, 0.0, 0.0])
CAMERA = dict(proj="ortho", el=-33.4256, az=-93.0, roll=0.7296, target=[0.0, 0.0, 0.0], scale=101.372, cx=274.904,
              cy=178.14)
RESULT_CAMERA = dict(proj="ortho", el=-7.44431, az=-19.1948, roll=53.07929, target=[0.0, 0.0, 0.0], scale=1238.60241,
                     cx=260.92131, cy=86.48319)
ALPHAS = (0.7, 0.7)


def hval(p, s, t, ds=0, dt=0):
    H = np.asarray(p["H"], float)
    s = np.asarray(s, float)
    t = np.asarray(t, float)
    out = 0.0
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            c = H[i, j]
            if c == 0.0 or i < ds or j < dt:
                continue
            fi = np.prod(range(i - ds + 1, i + 1)) if ds else 1
            fj = np.prod(range(j - dt + 1, j + 1)) if dt else 1
            out = out + c * fi * fj * s ** (i - ds) * t ** (j - dt)
    return out + 0 * s * t


def mval(p, s):
    return np.polyval(np.asarray(p["m"], float)[::-1], s)


def orange_pt(p, s, t):
    s, t = np.broadcast_arrays(np.asarray(s, float), np.asarray(t, float))
    return np.stack([s - p["k"] * t ** 3, hval(p, s, t), t], -1)


def blue_pt(p, s, t):
    s, t = np.broadcast_arrays(np.asarray(s, float), np.asarray(t, float))
    return np.stack([s + p["k"] * t ** 3, hval(p, s, t) + t ** 3 * mval(p, s), t], -1)


def grids(p, n=61):
    s = np.linspace(-1, 1, n)[:, None]
    to = np.linspace(p["to0"], p["to1"], n)[None, :]
    tb = np.linspace(p["tb0"], p["tb1"], n)[None, :]
    return blue_pt(p, s, tb), orange_pt(p, s, to)


def eta(p, n=81):
    """min over the domain of sgn * h_s, sgn = the sign of h_s at the centre (each sheet a graph over (X, Z))."""
    s = np.linspace(-1, 1, n)[:, None]
    t = np.linspace(min(p["to0"], p["tb0"]), max(p["to1"], p["tb1"]), n)[None, :]
    sgn = np.sign(hval(p, 0.0, 0.0, ds=1))
    return float((sgn * hval(p, s, t, ds=1)).min())


def valid(p):
    e = eta(p)
    mm = float(np.abs(mval(p, np.linspace(-1, 1, 201))).max())
    return bool(e > 0.02 and mm < 2 * p["k"] * e * 0.9 and p["k"] > 0 and p["to0"] < 0 < p["to1"]
                and p["tb0"] < 0 < p["tb1"])


def curve(p, n=2001):
    s = np.linspace(-1, 1, n)
    return orange_pt(p, s, 0 * s)
# IoU of the previous reconstruction (commit c91ad29) from its own best-fitting camera (run with --before)
BEFORE = dict(before_iou_union=0.6707, before_iou_blue=0.4954, before_iou_orange=0.5209)


# ----------------------------------------------------------------------------------- surfaces

def _poly_st(p, which):
    """Power coefficients (rows s^i, cols t^j) of X, Y, Z for one sheet."""
    k = p["k"]
    H = np.zeros((4, 4))
    H[:, :] = np.asarray(p["H"], float)
    X = np.zeros((4, 4))
    X[1, 0] = 1.0
    X[0, 3] = -k if which == "orange" else k
    Yc = H.copy()
    if which == "blue":
        Yc[:, 3] += np.asarray(p["m"], float)[:4]
    Z = np.zeros((4, 4))
    Z[0, 1] = 1.0
    return X, Yc, Z


def surfaces(p=PARAMS):
    out = []
    for which, tr, lab in (("blue", (p["tb0"], p["tb1"]), "slanted band B = (s + k t^3, h + t^3 m(s), t)"),
                           ("orange", (p["to0"], p["to1"]), "slanted band O = (s - k t^3, h(s,t), t)")):
        X, Yc, Z = _poly_st(p, which)
        out.append(g.rational_poly_patch(X, Yc, Z, Y.P([1.0]), (-1.0, 1.0), tr, lab, degree=(3, 3)))
    return out


def checks(p=PARAMS):
    out = {"eta_min_abs_h_s": eta(p), "max_abs_m": float(np.abs(mval(p, np.linspace(-1, 1, 401))).max()),
           "bound_2_k_eta": 2 * p["k"] * eta(p)}
    s = np.linspace(-1, 1, 21)
    sins = []
    for si in s:
        h = 1e-6
        nb = np.cross((blue_pt(p, si + h, 0) - blue_pt(p, si - h, 0)), (blue_pt(p, si, h) - blue_pt(p, si, -h)))
        no = np.cross((orange_pt(p, si + h, 0) - orange_pt(p, si - h, 0)), (orange_pt(p, si, h) - orange_pt(p, si, -h)))
        sins.append(np.linalg.norm(np.cross(nb, no)) / np.linalg.norm(nb) / np.linalg.norm(no))
    out["max_sin_normals_on_curve"] = float(max(sins))
    # depth gap at equal (X, Z) on both sides of the curve, X = 0: sign change = the sheets cross
    gaps = []
    for t in (-0.2, -0.1, 0.1, 0.2):
        so, sb = 0 + p["k"] * t ** 3, 0 - p["k"] * t ** 3
        gaps.append(float(blue_pt(p, sb, t)[1] - orange_pt(p, so, t)[1]))
    out["depth_gap_blue_minus_orange_at_t=-.2,-.1,.1,.2"] = gaps
    out["gap_over_t3_at_t=.1"] = gaps[2] / 0.1 ** 3
    return out


def build(p=PARAMS, fit=None):
    blue, orange = surfaces(p)
    fit = fit or {}
    cam_txt = lambda c: "%s el=%.2f az=%.2f roll=%.2f" % (c["proj"], c["el"], c["az"], c["roll"])
    Y.add(CID, "Table 2 ex.4 (3,3): a tangential curve (two slanted hourglass bands crossing at an inflected contact)",
          blue, orange, Y.SRC + "Table 2 (example 4)", "'a tangential curve', bi-degree (3,3)",
          "bi-degree (3,3), polynomial, one Bezier patch each; topology: one open tangential curve from waist point "
          "to waist point (both sheets' side edges meet there), inflected; the sheets cross each other at the contact "
          "(second-order contact), as the paper's panels show (orange in front above the curve, blue below); "
          "appearance fitted (render_match.py convention): Surfaces-panel camera %s, silhouette IoU union %s / blue "
          "%s / orange %s; result-row camera %s, mean chamfer %s px to the paper's curve"
          % (cam_txt(CAMERA), fit.get("iou_union", "?"), fit.get("iou_blue", "?"), fit.get("iou_orange", "?"),
             cam_txt(RESULT_CAMERA), fit.get("chamfer_result_px", "?")),
          "the construction (graphs over the (X, Z) plane sheared by -+ k t^3, common depth h) and all its "
          "coefficients are fitted, not published; the Surfaces camera looks from slightly below; the result-row "
          "camera looks almost along the curve (the paper's view of it is not pinned otherwise); the hump on the "
          "orange top edge and the groove in the AA panel's shading are not reproduced; the curve's inflection is "
          "weaker than drawn because h must stay monotone in s for the exact truth",
          "paper: " + Y.OPEN + ", " + Y.SPL + " and " + Y.COMM + " all returned nothing or an error",
          curves=[(lambda u: orange_pt(p, u, 0 * np.asarray(u, float)), -1.0, 1.0, False, "tangent")],
          cov_tol=1e-2,
          notes="O(s,t) = (s - k t^3, h(s,t), t), B(s,t) = (s + k t^3, h(s,t) + t^3 m(s), t), s in [-1,1]; "
                "k = %.6g; h = sum H[i][j] s^i t^j, H = %s; m = %s; t in [%.6g, %.6g] (blue), [%.6g, %.6g] (orange)"
                % (p["k"], json.dumps(p["H"]), json.dumps(p["m"]), p["tb0"], p["tb1"], p["to0"], p["to1"]))
    case = gc.CASES[-1]
    case["truth_method"] = ("closed form: the curve t = 0, (s, h(s,0), 0); no other intersection since |h_s| >= eta "
                            "> 0 and |m| < 2 k eta (mean value theorem, see the generator)")
    return case


def compare(p=PARAMS, old_case=None):
    sys.path.insert(0, HERE)
    import render_match as rm
    import fit_tools_t2_ex456 as T
    from fit_t2_ex4 import chamfer
    out = {}
    Ps, Pa, Pr = (T.paper(CID, k) for k in ("surfaces", "aagpu", "result"))
    A, B = grids(p, 121)
    H, W = Ps.shape[:2]
    cur = [curve(p, 2001)]
    rgb_s, m_s = rm.render([A, B], CAMERA, W, H, style="surfaces", alphas=ALPHAS, n=121, curves=cur)
    out["iou_union"], out["iou_blue"], out["iou_orange"] = T.ious(rm, Ps, rgb_s, m_s)
    cam_a, out["iou_aagpu_union"] = T.fit_framing(rm, [A, B], CAMERA, Pa)
    out["iou_aagpu_union"] = round(out["iou_aagpu_union"], 4)
    rgb_a, _ = rm.render([A, B], cam_a, Pa.shape[1], Pa.shape[0], style="surfaces", alphas=(1.0, 1.0), n=121,
                         colors=(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])))
    rgb_r = T.curves_only(rm, cur, RESULT_CAMERA, Pr.shape[1], Pr.shape[0])
    out["chamfer_result_px"] = round(chamfer(p, RESULT_CAMERA), 2)
    rm.compare_strip([Ps, Pa, Pr], [rgb_s, rgb_a, rgb_r], os.path.join(HERE, "compare", CID + ".png"))
    rgb_t, _ = rm.render([A, B], CAMERA, W, H, style="result", curves=cur, n=121)
    rm.compare_strip([Ps], [rgb_t], os.path.join(HERE, "compare", CID + "_truth.png"))
    if old_case is not None:
        G = [rm.surface_grid(sp, 41) for sp in old_case["surfaces"]]
        cam_o, _ = T.fit_camera(rm, G, Ps, ALPHAS)
        rgb_o, m_o = rm.render(G, cam_o, W, H, style="surfaces", alphas=ALPHAS, n=41)
        out["before_iou_union"], out["before_iou_blue"], out["before_iou_orange"] = T.ious(rm, Ps, rgb_o, m_o)
        rm.compare_strip([Ps], [rgb_o], os.path.join(HERE, "compare", CID + "_before.png"))
    return out


def main():
    import subprocess
    old = None
    if "--before" in sys.argv:
        old = json.loads(subprocess.run(["git", "-C", ROOT, "show", "c91ad29:cases/data/%s.json" % CID],
                                        capture_output=True, text=True, check=True).stdout)
    assert valid(PARAMS), "construction not exact: |h_s| >= eta > 0 and |m| < 2 k eta violated"
    fit = compare(PARAMS, old)
    if old is None:
        fit.update(BEFORE)
    print("fit", json.dumps(fit))
    case = build(PARAMS, fit)
    print(json.dumps(checks(PARAMS)))
    with open(os.path.join(ROOT, "cases", "data", CID + ".json"), "w") as fh:
        json.dump(case, fh)
    print(CID, "branches", len(case["expected"]["branches"]))


if __name__ == "__main__":
    main()
