"""Fig. 1(g) of Yang, Jia & Yan 2023, refitted to the paper's "Surfaces" panel.

What the panels show (cases/y23_fit/paper/y23_fig1g_three_points_*.png, 600 dpi crops):
  * blue: a rounded, pillow-like shell, wider than tall, pointed at the bottom (the apex). On its left a
    lens-shaped opening runs from a point on the upper-left rim (TL) down to the apex; through it the
    orange shows. The lens is bounded by two curves that both run from TL to the apex, so the blue there
    is a shell wrapped around an axis with a slit facing the viewer.
  * orange: seen only through the slit (pale result rows: orange is the front-most surface only there);
    everywhere else it is hidden inside the blue shell. Its shading changes across the lens (inside and
    outside of a curved sheet).
  * red: three isolated points, TL on the upper-left rim, TR near the upper-right rim, and the apex.
    The paper says the apex "also serves as a self-intersection point for the surfaces".

Reading adopted (from this one view; the paper gives no data): both surfaces are slit shells of
revolution about a common axis through the apex and TL, the orange nested inside the blue. The apex and
TL are the two poles (collapsed parameter edges of both patches), TR is an interior point where the
orange bulges out to touch the blue.

Construction (exact). Local frame: axis = z. Angular parameter t (theta = 2 atan t), meridian parameter s:
    r(s) = a s (1 - s),   z(s) = z1 s + z2 s^2 + z3 s^3 - z1/2,  z2 = -(z1 + 3 z3)/2  (so z'(1) = 0)
    blue   B(t, s) = A (r(s) (1 - t^2), r(s) 2t, z(s) (1 + t^2)) / (1 + t^2),   t in [tan(tb0/2), tan(tb1/2)]
    orange O(t, s) = A ((1 - D) r(s) (1 - t^2), (1 - D) r(s) 2t, z(s) (1 + t^2)) / (1 + t^2), t in [tan(to0/2), tan(to1/2)]
    D(t, s) = k (lam (t - t0)^2 + (s - s0)^2),   lam = 1 / (1 + t0^2)^2,  s in [0, 1] for both,
with A the linear map (x, y, z) -> (x, ey y + sh x, z) (elliptic cross-sections; A preserves
incidences and tangency). Before A, the orange point at (t, s) lies on the same ray from the axis and at
the same height as the blue point, at radius (1 - D) r(s). Because z is strictly increasing in s, the
blue surface of revolution meets that height in one circle, so the orange point is on the blue exactly
when D = 0 or r = 0, and the orange's angular range lies inside the blue's. With 0 <= D < 1 on the orange
domain (checked), the intersection is exactly three points:
  * TR = B(t0, s0): D = 0 with zero gradient, so the orange touches the blue tangentially from inside
    (gap quadratic in the distance);
  * TL = top pole (s = 1): smooth pole of both (z'(1) = 0), same tangent plane, gap quadratic;
  * apex = bottom pole (s = 0): a cone point of both surfaces (normals undefined there), the paper's
    self-intersection point; the orange's cone is narrower, so the gap is linear in the distance.
Bi-degree: blue (3, 3) rational, orange (4, 4) rational (the paper states no degree for Fig. 1).

Run: .venv/bin/python -I cases/y23_fit/y23_fig1g_three_points.py [--compare]
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

CID = "y23_fig1g_three_points"

# fitted (Nelder-Mead on silhouette, blue-region and orange-region chamfer plus the three red points'
# screen positions, against the Surfaces panel), rounded
PARAMS = dict(a=4.746, z1=3.121, z3=-0.8443, tb0=-93.02, tb1=101.93, to0=-54.67, to1=100.51, th0=11.49, s0=0.497,
              k=0.4361, ey=0.905, sh=-0.8135)
CAMERA = dict(proj="persp", el=36.74, az=-108.12, roll=-28.36, target=[0.0, 0.0, 0.0], scale=776.58, cx=105.26,
              cy=178.52, dist=4.5854)
# the result rows show the same view framed slightly differently (efg_compare.fit_framing)
RESULT_CAMERA = dict(CAMERA, scale=747.0, cx=107.4, cy=168.6)
# silhouette scores against the Surfaces panel; "before" = previous construction (graph domes over
# triangles) under its own best camera
FIT = dict(before=dict(iou=0.626, iou_blue=0.791, iou_orange=0.017), after=dict(iou=0.898, iou_blue=0.798, iou_orange=0.726))


def profile_coeffs(p):
    z1, z3 = p["z1"], p["z3"]
    z2 = -(z1 + 3 * z3) / 2
    r = [0.0, p["a"], -p["a"]]
    z = [-z1 / 2, z1, z2, z3]
    return r, z


def surfaces(p=PARAMS):
    r, z = profile_coeffs(p)
    t0 = math.tan(math.radians(p["th0"]) / 2)
    lam = 1.0 / (1 + t0 * t0) ** 2
    Rs = Y.P(r)                        # r(s): row t^0
    Zs = Y.P(z)
    one_m_t2 = Y.P([1.0], [0.0], [-1.0])
    two_t = Y.P([0.0], [2.0])
    Wt = Y.P([1.0], [0.0], [1.0])
    # D = k (lam (t - t0)^2 + (s - s0)^2)
    D = Y.poly(Y.mono(0, 0, p["k"] * (lam * t0 * t0 + p["s0"] ** 2)), Y.mono(1, 0, -2 * p["k"] * lam * t0),
               Y.mono(2, 0, p["k"] * lam), Y.mono(0, 1, -2 * p["k"] * p["s0"]), Y.mono(0, 2, p["k"]))
    one_m_D = g.p2_add(Y.P([1.0]), -D)

    def patch(scale, tr, label, degree):
        Xl = Y.mul(Rs, one_m_t2, scale)
        Yl = Y.mul(Rs, two_t, scale)
        Xh = Xl
        Yh = g.p2_add(p["ey"] * Yl, p["sh"] * Xl)
        Zh = Y.mul(Zs, Wt)
        T = (math.tan(math.radians(tr[0]) / 2), math.tan(math.radians(tr[1]) / 2))
        return g.rational_poly_patch(Xh, Yh, Zh, Wt, T, (0.0, 1.0), label, degree=degree)

    blue = patch(Y.P([1.0]), (p["tb0"], p["tb1"]),
                 "slit shell of revolution (elliptic sections), apex and top poles collapsed", (3, 3))
    orange = patch(one_m_D, (p["to0"], p["to1"]),
                   "inner slit shell, radius scaled by (1 - D), same poles", (4, 4))
    return blue, orange, D


def affine(p, P):
    P = np.array(P, float)
    P[..., 1] = p["ey"] * P[..., 1] + p["sh"] * P[..., 0]
    return P


def contact_points(p=PARAMS):
    r, z = profile_coeffs(p)
    zs = lambda s: sum(c * s ** i for i, c in enumerate(z))
    rs = lambda s: sum(c * s ** i for i, c in enumerate(r))
    th = math.radians(p["th0"])
    tr = [rs(p["s0"]) * math.cos(th), rs(p["s0"]) * math.sin(th), zs(p["s0"])]
    pts = affine(p, [[0.0, 0.0, zs(1.0)], tr, [0.0, 0.0, zs(0.0)]])
    return [tuple(map(float, q)) for q in pts]  # TL (top pole), TR, apex


def certify(p=PARAMS):
    """Conditions that make the three points the exact intersection, and contact measurements."""
    r, z = profile_coeffs(p)
    s = np.linspace(0, 1, 20001)
    dz = sum(i * c * s ** (i - 1) for i, c in enumerate(z) if i)
    out = {"min_dz_ds_on_[0,1)": float(dz[:-1].min()), "dz_ds_at_top": float(dz[-1])}
    blue, orange, D = surfaces(p)
    T0 = (math.tan(math.radians(p["to0"]) / 2), math.tan(math.radians(p["to1"]) / 2))
    tt, ss = np.meshgrid(np.linspace(*T0, 801), np.linspace(0, 1, 801), indexing="ij")
    Dv = g.p2_eval(D, tt, ss)
    out["D_range_on_orange"] = [float(Dv.min()), float(Dv.max())]
    out["orange_angles_inside_blue"] = bool(p["tb0"] < p["to0"] < p["th0"] < p["to1"] < p["tb1"])
    TL, TR, AP = (np.array(q) for q in contact_points(p))
    # normals at TR on both patches
    t0 = math.tan(math.radians(p["th0"]) / 2)
    S1, a1, b1 = g.param_derivs(blue.truth(), t0, p["s0"])
    S2, a2, b2 = g.param_derivs(orange.truth(), t0, p["s0"])
    n1, n2 = np.cross(a1, b1), np.cross(a2, b2)
    out["TR_on_both"] = float(max(np.linalg.norm(S1 - TR), np.linalg.norm(S2 - TR)))
    out["TR_sin_normals"] = float(np.linalg.norm(np.cross(n1, n2)) / np.linalg.norm(n1) / np.linalg.norm(n2))
    # gap growth near each point, measured before the linear map A (A changes distances by bounded factors
    # only, so the order of contact is unchanged): an orange point at (t, s) has height z(s) and radius
    # (1 - D) r(s); its distance to the blue surface of revolution is its distance, in the meridian
    # half-plane, to the profile curve (r(s), z(s)).
    S = np.linspace(0, 1, 200001)
    Rp = sum(c * S ** i for i, c in enumerate(r))
    Zp = sum(c * S ** i for i, c in enumerate(z))
    t0 = math.tan(math.radians(p["th0"]) / 2)
    rows = {}
    for name, sc, tc in (("TR", p["s0"], t0), ("TL", 1.0, t0), ("apex", 0.0, t0)):
        rho0 = float(sum(c * sc ** i for i, c in enumerate(r))) if name == "TR" else 0.0
        z0 = float(sum(c * sc ** i for i, c in enumerate(z)))
        ratios = []
        for h in (0.04, 0.02, 0.01, 0.005, 0.0025):
            sq = sc - h if sc >= 1.0 else sc + h
            rq = float(sum(c * sq ** i for i, c in enumerate(r))) * (1 - float(g.p2_eval(D, tc, sq)))
            zq = float(sum(c * sq ** i for i, c in enumerate(z)))
            gap = float(np.min(np.hypot(Rp - rq, Zp - zq)))
            dist = float(math.hypot(rq - rho0, zq - z0))
            ratios.append([round(dist, 6), gap / dist, gap / dist ** 2])
        rows[name] = ratios
    out["gap_over_dist_and_dist2"] = rows
    return out


def fmt_cam(c):
    return "%s el=%.2f az=%.2f roll=%.2f dist=%.3f (focal %.0f px, target at (%.1f, %.1f) px) in a 437x386 panel" % (
        c["proj"], c["el"], c["az"], c["roll"], c.get("dist", 0), c["scale"], c["cx"], c["cy"])


def build():
    p = PARAMS
    blue, orange, _ = surfaces(p)
    cert = certify(p)
    assert cert["min_dz_ds_on_[0,1)"] > 0 and abs(cert["dz_ds_at_top"]) < 1e-12
    assert 0 <= cert["D_range_on_orange"][0] and cert["D_range_on_orange"][1] < 1
    assert cert["orange_angles_inside_blue"] and cert["TR_on_both"] < 1e-12
    pts = contact_points(p)
    fa, fb = FIT["after"], FIT["before"]
    Y.add(CID, "Fig.1(g): nested slit shells touching at three isolated points (two poles + one interior point)",
          blue, orange, Y.SRC + "Fig. 1 (g)",
          "'the two surfaces are in contact at three isolated points'; the paper adds that one point is also a "
          "self-intersection point of the surfaces",
          "topology: three isolated contact points and nothing else: TL (top pole, tangential, smooth), TR (interior "
          "tangential bump contact, sin(normals) %.0e), apex (common cone point of both surfaces = the paper's "
          "self-intersection point, collapsed parameter edge of both patches); appearance fitted to the Surfaces panel "
          "(silhouette, blue and orange regions, the three red points' positions): a pillow-like blue shell with a "
          "lens-shaped slit from TL to the apex through which the orange shows, the orange otherwise hidden inside; "
          "camera %s; silhouette IoU %.3f (blue %.3f, orange %.3f), previous construction %.3f under its own best "
          "camera" % (cert["TR_sin_normals"], fmt_cam(CAMERA), fa["iou"] or 0, fa["iou_blue"] or 0,
                      fa["iou_orange"] or 0, fb["iou"] or 0),
          "the paper gives no data and one view: the reading as nested slit shells of revolution about one axis, the "
          "profile, elliptic sections and the camera are recovered from that view; which red point the paper's "
          "'self-intersection point' is: taken to be the apex (the point the commercial kernel missed); TL is a "
          "smooth pole (tangential contact), the apex a cone point (gap linear in the distance, normals undefined); "
          "bi-degree blue (3,3), orange (4,4), rational; not reproduced: the blue's lower-right outline is rounder "
          "than drawn and its left end slightly wider",
          "paper: " + Y.OPEN + " returned nothing, " + Y.COMM + " lost one point (the apex)",
          points=pts, cov_tol=1e-2,
          notes="contact points TL %s, TR %s, apex %s; checks %s" % (
              tuple(round(v, 12) for v in pts[0]), tuple(round(v, 12) for v in pts[1]),
              tuple(round(v, 12) for v in pts[2]),
              json.dumps({k: v for k, v in cert.items() if k != "gap_over_dist_and_dist2"})))
    case = gc.CASES[-1]
    case["truth_method"] = ("closed form: orange = blue with radius scaled by (1 - D) at equal parameters; "
                            "intersection = {D = 0} + the two poles (z strictly increasing, 0 <= D < 1)")
    return case, cert


def main():
    case, cert = build()
    assert len(case["expected"]["isolated_points"]) == 3, case["expected"]
    with open(os.path.join(ROOT, "cases", "data", CID + ".json"), "w") as fh:
        json.dump(case, fh)
    print(CID, json.dumps(cert))
    if "--compare" in sys.argv:
        import efg_compare
        print("after", efg_compare.compare(case, CAMERA, result_camera=RESULT_CAMERA))


if __name__ == "__main__":
    main()
