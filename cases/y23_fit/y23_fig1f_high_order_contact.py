"""Fig. 1(f) of Yang, Jia & Yan 2023, refitted to the paper's "Surfaces" panel.

What the panel shows (cases/y23_fit/paper/y23_fig1f_high_order_contact_*.png, 600 dpi crops):
  * orange: a ruled S-step sheet seen from just above its low part (elevation ~7 deg): the low part
    runs to a sharp tip at the lower left, a near-vertical wall, the high part runs away to a tip at
    the upper right whose underside shows. The S is point-symmetric about the middle of the wall
    (corner pairs and wall bends map onto each other under a half turn).
  * blue: covers only the wall (about 45% of the profile), spans the same rulings, and its two ends
    flare away from the orange: the lower end toward the viewer (a thin wing past the orange's left
    edge), the upper end behind (a thin wing past the right edge). Again point-symmetric.
  * red: a straight line along the rulings, half-way up the wall, spanning the orange's full width.
  * below the line the blue is in front of the orange, above it behind (brown tint): the sheets cross
    at the line, so the contact is of odd order.

Construction (exact, rational bi-degree (1, 3) for both; the paper states no degree for Fig. 1):
  * profile C(t) = (Ny, Nz)(t) / w(t), t in [0, 1]: point-symmetric rational cubic Bezier in the (y, z)
    plane, P0 = (-L, -h), P1 = (m, -h + k), P2 = (-m, h - k), P3 = (L, h), weights (1, om, om, 1).
  * orange O(x, t) = (x, C(t)), x in [-Xo, Xo], t in [0, 1].
  * blue B(x, t) = (x, (N(t) + (t - 1/2)^3 lam n) / w(t)), x in [-Xb, Xb], t in [tb0, tb1] = [tb0, 1 - tb0],
    n the unit normal of C at t = 1/2. At equal (x, t), B - O = (t - 1/2)^3 lam n / w(t): same point,
    tangent plane and normal curvature along t = 1/2, opposite sides on either side (second-order,
    crossing contact). The cubic growth of the gap is what makes the drawn wings.
  * both sheets are extrusions along x, so their intersection is {profile intersections} x {common x}.
    certify_profiles() shows the blue profile is at distance >= c |t - 1/2|^3 (c > 0) from the orange
    profile, so the only common point is t = 1/2 and the intersection is exactly the straight segment
    x -> (x, C(1/2)), |x| <= Xo, ending on the orange's side edges.

Run: .venv/bin/python -I cases/y23_fit/y23_fig1f_high_order_contact.py [--compare]
writes cases/data/y23_fig1f_high_order_contact.json (and with --compare the compare images).
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

CID = "y23_fig1f_high_order_contact"

# fitted shape parameters (Nelder-Mead on silhouette + blue-region chamfer against the Surfaces panel,
# then rounded); Xb is set a little wider than Xo so that the contact line ends on the orange's side edges
PARAMS = dict(L=1.49, h=0.854, m=0.087, k=-0.227, om=4.39, Xo=1.25, Xb=1.30, tb0=0.28, tb1=0.72, lam=-39.1)
# fitted camera (render_match.py convention): perspective, eye 12.3 units from the centre of the S
CAMERA = dict(proj="persp", el=6.78, az=-39.74, roll=3.24, target=[0.0, 0.0, 0.0], scale=1352.9, cx=237.4,
              cy=176.3, dist=12.28)
# the paper's result rows show the same view with a slightly different 2D framing (efg_compare.fit_framing)
RESULT_CAMERA = dict(CAMERA, scale=1493.3, cx=216.4, cy=165.4)


def profile(L, h, m, k=0.0, om=1.0):
    """Homogeneous control points (4, 3) = (w y, w z, w) of the point-symmetric rational cubic S in the
    (y, z) plane: P0 = (-L, -h), P1 = (m, -h + k), P2 = (-m, h - k), P3 = (L, h), weights (1, om, om, 1).
    om > 1 pulls the curve toward P1 and P2: long flat ends and a sharp turn into the wall."""
    P = np.array([[-L, -h], [m, -h + k], [-m, h - k], [L, h]], float)
    W = np.array([1.0, om, om, 1.0])
    return np.column_stack([P * W[:, None], W])


def bezier_power(Pc):
    """Cubic Bezier control points (4, d) -> power coefficients (4, d), C(t) = sum a_k t^k."""
    M = np.array([[1, 0, 0, 0], [-3, 3, 0, 0], [3, -6, 3, 0], [-1, 3, -3, 1]], float)
    return M @ Pc


def _curve(a):
    """Rational plane curve from homogeneous power coefficients a (4, 3): C, C', C'' as functions of t."""
    pv = lambda c, t, d=0: np.polyval(np.polyder(c[::-1], d) if d else c[::-1], t)

    def ev(t, d=0):
        t = np.asarray(t, float)
        H = [np.stack([pv(a[:, k], t, j) for k in range(3)], -1) for j in range(3)]
        w0, w1, w2 = H[0][..., 2:], H[1][..., 2:], H[2][..., 2:]
        C = H[0][..., :2] / w0
        if d == 0:
            return C
        C1 = (H[1][..., :2] - C * w1) / w0
        if d == 1:
            return C1
        return (H[2][..., :2] - 2 * C1 * w1 - C * w2) / w0
    return ev


def contact_normal(a):
    d = _curve(a)(0.5, 1)
    n = np.array([-d[1], d[0]])
    return n / np.linalg.norm(n)


def surfaces(p=PARAMS):
    Pc = profile(p["L"], p["h"], p["m"], p.get("k", 0.0), p.get("om", 1.0))
    a = bezier_power(Pc)  # a[k] = (wy_k, wz_k, w_k)
    n = contact_normal(a)
    wv = p["lam"] * n
    # blue homogeneous profile = orange's + (t - 1/2)^3 (wv, 0):
    # blue - orange = (t - 1/2)^3 wv / w(t), a triple zero at t = 1/2
    c3 = np.array([-0.125, 0.75, -1.5, 1.0])
    ab = a.copy()
    ab[:, 0] += wv[0] * c3
    ab[:, 1] += wv[1] * c3
    X = Y.P([0, 0, 0, 0], [1])  # x = u   (rows: u^i, columns: t^j)
    hom = lambda A, k: Y.P(list(A[:, k]))
    Xh = lambda A: g.p2_mul(X, Y.P(list(A[:, 2])))  # x * w(t), so that x = u after division
    orange = g.rational_poly_patch(Xh(a), hom(a, 0), hom(a, 1), hom(a, 2), (-p["Xo"], p["Xo"]), (0.0, 1.0),
                                   "S-step sheet: rational cubic S profile extruded along x", degree=(1, 3))
    blue = g.rational_poly_patch(Xh(ab), hom(ab, 0), hom(ab, 1), hom(ab, 2), (-p["Xb"], p["Xb"]),
                                 (p["tb0"], p["tb1"]),
                                 "wall sheet: profile C(t) + (t - 1/2)^3 w / w(t) (2nd-order contact at t = 1/2)",
                                 degree=(1, 3))
    return blue, orange, a, ab


def certify_profiles(p=PARAMS, n=20001):
    """The blue and orange profile curves meet only at t = 1/2 (as point sets).

    For every blue profile point the distance to the orange profile (dense seed + Newton on the foot
    point) is divided by |t - 1/2|^3: the ratio stays bounded away from 0, so the only common point is
    t = 1/2. Returns (min ratio, max ratio, max distance within 1e-3 of t = 1/2)."""
    _, _, a, ab = surfaces(p)
    C, Cb = _curve(a), _curve(ab)
    ts = np.linspace(0, 1, 4001)
    G = C(ts)
    tb = np.linspace(p["tb0"], p["tb1"], n)
    Bp = Cb(tb)
    j = np.argmin(((Bp[:, None, :] - G[None, ::4, :]) ** 2).sum(-1), axis=1) * 4
    t = ts[j].copy()
    for _ in range(40):  # Newton on (C(t) - B) . C'(t) = 0
        r = C(t) - Bp
        d1, d2 = C(t, 1), C(t, 2)
        f = (r * d1).sum(-1)
        fp = (d1 * d1).sum(-1) + (r * d2).sum(-1)
        t = np.clip(t - f / fp, 0, 1)
    dist = np.linalg.norm(C(t) - Bp, axis=1)
    off = np.abs(tb - 0.5) > 1e-3
    ratio = dist[off] / np.abs(tb[off] - 0.5) ** 3
    return float(ratio.min()), float(ratio.max()), float(dist[~off].max())


def contact_checks(p=PARAMS):
    """sin(normal angle) along the line, and the gap growth |B - O| / |t - 1/2|^k."""
    blue, orange, _, _ = surfaces(p)
    out = {}
    xs = np.linspace(-p["Xo"], p["Xo"], 9)
    sins = []
    for x in xs:
        S1, a1, b1 = g.param_derivs(blue.truth(), x, 0.5)
        S2, a2, b2 = g.param_derivs(orange.truth(), x, 0.5)
        n1, n2 = np.cross(a1, b1), np.cross(a2, b2)
        sins.append(np.linalg.norm(np.cross(n1, n2)) / np.linalg.norm(n1) / np.linalg.norm(n2))
        assert np.linalg.norm(S1 - S2) < 1e-14
    out["max_sin_normals_on_line"] = float(max(sins))
    return out


# silhouette scores against the Surfaces panel (efg_compare.scores), from `--compare`; "before" is the
# previous construction (z = w - w^3/3 step, D = w^4) under its own best-fitting camera
FIT = dict(before=dict(iou=0.734, iou_blue=0.206, iou_orange=0.393), after=dict(iou=0.947, iou_blue=0.878, iou_orange=0.921))


def fmt_cam(c):
    return "%s el=%.2f az=%.2f roll=%.2f dist=%.2f (focal %.0f px, centre at (%.1f, %.1f) px) in a %dx%d panel" % (
        c["proj"], c["el"], c["az"], c["roll"], c.get("dist", 0), c["scale"], c["cx"], c["cy"], 437, 386)


def build():
    blue, orange, a, _ = surfaces()
    c_mid = _curve(a)(0.5)
    p = PARAMS
    r = certify_profiles()
    ck = contact_checks()
    Y.add(CID, "Fig.1(f): high-order contact along a straight line", blue, orange,
          Y.SRC + "Fig. 1 (f)", "'the two surfaces are in high-order contact along a line'",
          "topology: one open straight tangential line (second-order, crossing contact) along the rulings, half-way "
          "up the wall, spanning the orange's full width; appearance fitted to the Surfaces panel by silhouette and "
          "blue-region matching: point-symmetric S-step sheet with a near-vertical wall (orange), a sheet covering "
          "only the wall whose ends flare away into the thin wings drawn past the orange's side edges (blue), blue "
          "in front below the line and behind above it as drawn; camera %s; silhouette IoU %.3f (blue %.3f, orange "
          "%.3f), previous construction %.3f under its own best camera" % (
              fmt_cam(CAMERA), FIT["after"]["iou"] or 0, FIT["after"]["iou_blue"] or 0, FIT["after"]["iou_orange"] or 0,
              FIT["before"]["iou"] or 0),
          "the paper gives no data: profile shape (rational cubic, weights 1, %.2f, %.2f, 1), sheet proportions and "
          "camera were recovered from one view; bi-degree (1,3) rational for both (not stated for Fig. 1); contact "
          "order 2 (the paper says only 'high-order'; its drawing of the sheets crossing at the line rules out an "
          "even order, the lowest odd one was taken); not reproduced: the slight bow of the drawn lower front edge "
          "and the curl at the upper-right tip (a ruled sheet has straight edges)" % (p["om"], p["om"]),
          "paper: both kernels returned nothing",
          curves=[(gc.line((-p["Xo"], c_mid[0], c_mid[1]), (1, 0, 0)), 0, 2 * p["Xo"], False, "tangent")],
          cov_tol=1e-2,
          notes="contact line x -> (x, %.12g, %.12g), |x| <= %g; at equal parameters blue - orange = (t - 1/2)^3 "
          "lam n / w(t); normals parallel along the line (max sin %.1e); profile distance / |t - 1/2|^3 in "
          "[%.3g, %.3g] over the blue's profile range" % (c_mid[0], c_mid[1], p["Xo"],
                                                          ck["max_sin_normals_on_line"], r[0], r[1]))
    case = gc.CASES[-1]
    case["truth_method"] = ("closed form: the straight ruling (x, C(1/2)); the two profile curves meet only at "
                            "t = 1/2 (certified by the |t - 1/2|^3 distance bound)")
    return case


def main():
    case = build()
    with open(os.path.join(ROOT, "cases", "data", CID + ".json"), "w") as fh:
        json.dump(case, fh)
    print(CID, "branches", len(case["expected"]["branches"]), case["notes"])
    if "--compare" in sys.argv:
        import efg_compare
        print("after", efg_compare.compare(case, CAMERA, result_camera=RESULT_CAMERA))


if __name__ == "__main__":
    main()
