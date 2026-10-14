"""Fig. 1(b) of Yang, Jia & Yan 2023, refitted to the paper's panels.

What the panels show (cases/y23_fit/paper/y23_fig1b_two_arcs_and_point_*.png, 600 dpi crops):
  * seen front-on and at mid-height (mirror-symmetric left/right and up/down): the blue is a horizontal
    spool: its top and bottom silhouettes are concave (half-height 122 px at the
    middle, 148 px near the ends), and its ends flare into narrow light bands with rounded corners (the
    surface turning away near the rims).
  * the orange shows only as two lens-shaped caps above and below the blue (half-height 156 px at the
    middle, ending at about 0.8 of the blue's half-length): a barrel, mostly hidden behind the blue.
  * the two red curves run just inside the blue's top and bottom silhouettes, flat in the middle and
    hooking outward at their ends; the red dot is the centre of the blue's front face.

Construction (exact; rational, blue bi-degree (6, 2), orange (4, 2); the paper states no degree):
  world: x toward the viewer, y to the right, z up; u = s / Lb.
  * blue  S(s, phi) = (R(s) cos phi, s, R(s) sin phi), |s| <= Lb, phi all round,
          R = 1 + r2 u^2 + r4 u^4 + r6 u^6 (a spool, closed tube).
  * orange O(s, psi) = (c(s) + A cos psi, s, Bz(s) sin psi), |s| <= So Lb, |psi| <= 90 deg,
          c = 1 - A + c2 u^2 + c4 u^4, Bz = b0 + b2 u^2 (front half of an elliptic barrel; its front line
          c + A touches the blue at (1, 0, 0)).
  * both surfaces have their cross-sections in the planes y = s, so per s the intersection is circle
    cap ellipse: with w = cos phi, (R w - c)^2 / A^2 + R^2 (1 - w^2) / Bz^2 = 1, a quadratic in w.
    One root gives the two branches z = +-R sqrt(1 - w^2) (rim to rim of the orange; they lie just
    behind the spool's top and bottom lines, phi ~ 96-113 deg, inside the orange's front half); the other is
    w = 1 at s = 0 and > 1 for s != 0 (checked): the isolated contact point (1, 0, 0), where both
    normals are (1, 0, 0). Near it the orange stays strictly inside the blue: across, A / Bz^2 > 1 / R;
    along s, r2 - c2 > 0.

Fitted cameras (render_match.py convention, fit_y23_fig1b_two_arcs_and_point.py): Surfaces row
perspective, el 0.61, az -0.67, roll -0.15, eye 11.5 units away, focal 1397 px; the Ours row is the
same view framed 2-4 px apart and 2 % smaller.

Run: .venv/bin/python -I cases/y23_fit/y23_fig1b_two_arcs_and_point.py [--compare]
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

CID = "y23_fig1b_two_arcs_and_point"

PARAMS = dict(r2=-0.11423, r4=1.68659, r6=-1.67288, A=5.35842, b0=1.63364, b2=-0.07698, c2=-0.19997,
              c4=0.30025, So=0.82495, Lb=1.56071)
CAMERA = dict(proj="persp", el=0.6074, az=-0.6675, roll=-0.1545, scale=1397.07, dist=11.5350, cx=213.15,
              cy=193.01, target=[0.0, 0.0, 0.0])
RESULT_CAMERA = dict(CAMERA, scale=1366.53, cx=209.24, cy=193.67)

P1 = np.polynomial.polynomial


def polys(p=PARAMS):
    """Power coefficients in s (lowest first) of R, c, A, Bz."""
    L = p["Lb"]
    R = np.array([1.0, 0, p["r2"] / L ** 2, 0, p["r4"] / L ** 4, 0, p["r6"] / L ** 6])
    c = np.array([1.0 - p["A"], 0, p["c2"] / L ** 2, 0, p["c4"] / L ** 4])
    Bz = np.array([p["b0"], 0, p["b2"] / L ** 2])
    return R, c, np.array([p["A"]]), Bz


def bernstein(c, deg, lo, hi):
    cu = np.zeros(deg + 1)
    base = np.array([1.0])
    for k, ck in enumerate(c):
        if k:
            base = P1.polymul(base, [lo, hi - lo])
        cu[:len(base)] += ck * base
    return np.array([sum(math.comb(j, i) / math.comb(deg, i) * cu[i] for i in range(j + 1)) for j in range(deg + 1)])


def tube(center_x, ax, az_, smax, deg, label, full):
    """(cx(s) + ax(s) cos t, s, az(s) sin t), |s| <= smax; t all round (full) or |t| <= 90 deg.
    Exact NURBS (deg, 2) and a parametric truth: the half uses cos, sin = (1 - v^2, 2 v) / (1 + v^2),
    v in [-1, 1]; the full circle uses v in [0, 1] with N = v - 1/2, D = v (1 - v):
    cos, sin = (D^2 - N^2, 2 N D) / (N^2 + D^2) (once round, v = 0 and 1 both at t = 180 deg)."""
    cp, cw, ck = g.arc_ctrl(0, 360) if full else g.arc_ctrl(-90, 90)
    cb, ab, zb = (bernstein(f, deg, -smax, smax) for f in (center_x, ax, az_))
    sb = bernstein(np.array([0.0, 1.0]), deg, -smax, smax)
    net = np.zeros((deg + 1, len(cp), 4))
    for i in range(deg + 1):
        for j in range(len(cp)):
            P3 = np.array([cb[i] + ab[i] * cp[j, 0], sb[i], zb[i] * cp[j, 1]])
            net[i, j] = [*(P3 * cw[j]), cw[j]]
    U = [-smax] * (deg + 1) + [smax] * (deg + 1)
    if full:
        N, D = np.array([-0.5, 1.0]), np.array([0.0, 1.0, -1.0])
        Wt = P1.polyadd(P1.polymul(N, N), P1.polymul(D, D))
        Ct = P1.polysub(P1.polymul(D, D), P1.polymul(N, N))
        St = 2 * P1.polymul(N, D)
        dom = [-smax, smax, 0.0, 1.0]
    else:
        Wt, Ct, St = np.array([1.0, 0.0, 1.0]), np.array([1.0, 0.0, -1.0]), np.array([0.0, 2.0])
        dom = [-smax, smax, -1.0, 1.0]
    m = max(len(Wt), len(Ct), len(St))
    pad = lambda a, n: np.pad(np.asarray(a, float), (0, n - len(a)))
    Wt, Ct, St = pad(Wt, m), pad(Ct, m), pad(St, m)
    Xh = np.outer(pad(center_x, deg + 1), Wt) + np.outer(pad(ax, deg + 1), Ct)
    Yh = np.outer(pad([0.0, 1.0], deg + 1), Wt)
    Zh = np.outer(pad(az_, deg + 1), St)
    Wh = np.outer(pad([1.0], deg + 1), Wt)
    truth = {"kind": "param", "hom": [h.tolist() for h in (Xh, Yh, Zh, Wh)], "domain": dom}
    if full:
        inside = lambda P_, tol: np.abs(P_[:, 1]) <= smax + tol
    else:
        inside = lambda P_, tol: (np.abs(P_[:, 1]) <= smax + tol) & (P_[:, 0] >= P1.polyval(P_[:, 1], center_x) - tol)
    return g.Nurbs(deg, 2, U, ck, net, truth, inside, label)


def surfaces(p=PARAMS):
    R, c, A, Bz = polys(p)
    blue = tube(np.zeros(1), R, R, p["Lb"], 6, "spool (closed tube), radius 1 + r2 u^2 + r4 u^4 + r6 u^6", True)
    orange = tube(c, A, Bz, p["So"] * p["Lb"], 4,
                  "front half of an elliptic barrel (c(s) + A cos t, s, Bz(s) sin t), touching the spool at (1,0,0)",
                  False)
    return blue, orange


def roots(s, p=PARAMS):
    """Both roots w = cos(phi) of the per-section quadratic, sorted."""
    R, c, A, Bz = (P1.polyval(s, f) for f in polys(p))
    a2 = R * R / (A * A) - R * R / (Bz * Bz)
    a1 = -2 * R * c / (A * A)
    a0 = c * c / (A * A) + R * R / (Bz * Bz) - 1
    sq = np.sqrt(a1 * a1 - 4 * a2 * a0)
    w1, w2 = (-a1 - sq) / (2 * a2), (-a1 + sq) / (2 * a2)
    return np.minimum(w1, w2), np.maximum(w1, w2), R


def branch(sign, p=PARAMS):
    def f(t):
        s = np.asarray(t, float)
        w, _, R = roots(s, p)
        return np.column_stack([R * w, s, sign * R * np.sqrt(1 - w * w)])
    return f


# ------------------------------------------------------------------------------------------- checks

def normals_at(Pt, p=PARAMS, h=1e-6):
    """Unit normals from the implicit cross-section equations (y = s exactly on both)."""
    R, c, A, Bz = polys(p)

    def FA(Q):
        return Q[0] ** 2 + Q[2] ** 2 - P1.polyval(Q[1], R) ** 2

    def FB(Q):
        return ((Q[0] - P1.polyval(Q[1], c)) / A[0]) ** 2 + (Q[2] / P1.polyval(Q[1], Bz)) ** 2 - 1
    grad = lambda F: np.array([(F(Pt + e) - F(Pt - e)) / (2 * h) for e in np.eye(3) * h])
    a, b = grad(FA), grad(FB)
    return a / np.linalg.norm(a), b / np.linalg.norm(b)


def checks(p=PARAMS):
    out = {}
    s = np.linspace(-p["So"] * p["Lb"], p["So"] * p["Lb"], 20001)
    w1, w2, _ = roots(s, p)
    out["branch root w1 = cos(phi): range (in (-1, 1); < 0: just behind the spool's top line)"] = (float(w1.min()), float(w1.max()))
    nz = np.abs(s) > 1e-9
    out["second root: min (w2 - 1) / s^2 for s != 0 (> 0: no real point but s = 0)"] = float(((w2[nz] - 1) / s[nz] ** 2).min())
    out["second root at s = 0 (contact)"] = float(roots(np.array([0.0]), p)[1][0])
    nA, nB = normals_at(np.array([1.0, 0.0, 0.0]), p)
    out["contact point: sin(normal angle)"] = float(np.linalg.norm(np.cross(nA, nB)))
    R, c, A, Bz = polys(p)
    out["isolation across: A / Bz(0)^2 - 1 / R(0) (> 0)"] = float(A[0] / Bz[0] ** 2 - 1.0)
    out["isolation along: r2 - c2 (> 0)"] = float(p["r2"] - p["c2"])
    sins = []
    for sg in (1, -1):
        P = branch(sg, p)(s[::200])
        sins += [np.linalg.norm(np.cross(*normals_at(Pt, p))) for Pt in P]
    out["branches: min sin(normal angle)"] = float(min(sins))
    # every branch point is on the orange's front half (psi in [-90, 90] deg)
    P = branch(1, p)(s)
    out["branches: min x - c(s) (orange psi within +-90 deg, >= 0)"] = float((P[:, 0] - P1.polyval(s, c)).min())
    return out


def build():
    blue, orange = surfaces()
    p = PARAMS
    sm = p["So"] * p["Lb"]
    cam = CAMERA
    Y.add(CID, "Fig.1(b): spool and barrel fronts, two rim-to-rim branches and an isolated contact point", blue, orange,
          Y.SRC + "Fig. 1 (b)", "'the two surfaces intersect at two separate curve branches and are in contact at an "
          "isolated point'",
          "topology: two open branches ending on the orange's rims + one isolated tangential contact point at the "
          "centre of the blue's front face (common normal (1,0,0)); appearance, refitted to the 600 dpi panels: blue "
          "a spool seen front-on at mid-height (concave top/bottom silhouettes, flared ends), "
          "orange the front half of an elliptic barrel showing as lens-shaped caps above and below the blue, the "
          "branches flat and just inside the blue's silhouettes, hooking outward at their ends; left/right and "
          "up/down mirror symmetric by construction; fitted camera perspective el %.2f az %.2f roll %.2f, eye "
          "%.2f units away, focal %.0f px (render_match.py); against the Surfaces panel: silhouette IoU 0.958, "
          "blue-visible IoU 0.946, orange-visible IoU 0.821 (previous cylinder sector + quartic sheet under its own "
          "best-fitting camera: 0.955, 0.121, 0.130); branches within ~0.3 / 2.5 px of the paper's red curves" % (
              cam["el"], cam["az"], cam["roll"], cam["dist"], cam["scale"]),
          "bi-degree blue (6, 2), orange (4, 2), rational (not stated in the paper); the spool and barrel "
          "profiles, the orange's depth and how far it reaches are fitted to the outlines and curves, not "
          "known; the paper's branch ends hook a little more sharply toward the blue's corners",
          "paper: " + Y.OPEN + " lost the isolated point",
          curves=[(branch(1), -sm, sm, False, "crossing"), (branch(-1), -sm, sm, False, "crossing")],
          points=[(1.0, 0.0, 0.0)], cov_tol=1e-2)
    case = gc.CASES[-1]
    case["notes"] = ("per section y = s the circle R(s) and the ellipse (c + A cos, Bz sin) meet where "
                     "(R w - c)^2/A^2 + R^2 (1 - w^2)/Bz^2 = 1, w = cos phi (closed form); the other root is w = 1 "
                     "only at s = 0: isolated tangential contact at (1, 0, 0)")
    case["truth_method"] = "closed form (per-section quadratic), surfaces exact rational"
    return case


def write(case):
    with open(os.path.join(ROOT, "cases", "data", CID + ".json"), "w") as fh:
        json.dump(case, fh)


if __name__ == "__main__":
    case = build()
    write(case)
    for k, v in checks().items():
        print("%-66s %s" % (k, v))
    if "--compare" in sys.argv:
        import efg_compare as ec
        print(ec.compare(case, CAMERA, n=241, result_camera=RESULT_CAMERA))
