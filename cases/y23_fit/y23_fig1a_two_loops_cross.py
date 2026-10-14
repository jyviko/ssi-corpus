"""Fig. 1(a) of Yang, Jia & Yan 2023, refitted to the paper's panels.

What the panels show (cases/y23_fit/paper/y23_fig1a_two_loops_cross_*.png, 600 dpi crops):
  * two congruent closed "petal" surfaces, blue and orange, each with two pointed lens-shaped lobes on
    either side of a waist; the orange is the blue turned about 90 degrees about the common centre,
    seen from above at about 63 degrees elevation with perspective (the near lobes are ~30 % longer).
  * fold lines inside the lobes (the other surface seen through the front one) and the rounded
    shading across each lobe: the lobes are closed tubes that end in a point, not single sheets.
  * the loops: two thin, elongated loops, one along each lobe pair's diagonal, of slightly different
    lengths, crossing four times in projection. The loop ends sit exactly where the silhouettes of
    the two surfaces cross at the waist, i.e. on each tube's side line.
  * the lobe tips are not on the lobe centre lines: each tip is swept to one side, with the same
    sense on both lobes of a surface (an S-shaped axis) and the opposite sense on the other surface.

Construction (exact; rational bi-degree (8, 2) for both; the paper states no degree for Fig. 1):
  * blue A(s, th) = (s, yc(s) + a(s) cos th, zc(s) + kappa a(s) sin th), s in [-1, 1], th in [0, 2 pi):
    a tube about the S-shaped axis (s, yc(s), zc(s)) with yc = beta s^3 + beta5 s^5 (sweeps the tips),
    zc = gamma s^2 (lifts/lowers the tips), radius a(s) = (1 - s^2)(c0 + c2 s^2 + c4 s^4 + c6 s^6)
    (cone points at the tips, a waist at s = 0) and an elliptic cross-section of ratio kappa.
  * orange B = mirror image of A in the plane x = y. A is invariant under the half turn about z
    ((x, y, z) -> (-x, -y, z): yc odd, zc and a even), so B is also the mirror image of A in x = -y.
    Each mirror plane P therefore satisfies A cap P = B cap P, so both planar loops
    A cap {y = +-x} lie on both surfaces, in closed form:
        x = s, y = +-s, z = gamma s^2 +- kappa sqrt(a(s)^2 - (+-s - yc(s))^2),  |s| <= s*(+-)
    with a(s*) = |+-s* - yc(s*)|. Both loops pass through (0, 0, +-kappa c0), where the tubes have the
    common normal (0, 0, +-1): the paper's two crossing points (tangential, saddle-type contact).
  * certify(): on a fine (s, th) grid of A, the sign changes of F_B (implicit equation of B) lie only
    on the two mirror planes, so the intersection is exactly the two loops.

Fitted camera (render_match.py convention, recovered from the panels together with the shape by
differential evolution on silhouette IoU + loop chamfer + outline chamfer + tip widths; see
fit_y23_fig1a_two_loops_cross.py): perspective, el 63.87, az 2.94, roll -30.84, eye 7.18 units from the
centre.

Run: .venv/bin/python -I cases/y23_fit/y23_fig1a_two_loops_cross.py [--compare]
writes cases/data/y23_fig1a_two_loops_cross.json (with --compare also the compare images).
"""
import json
import math
import os
import sys

import numpy as np
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "cases"))
sys.path.insert(0, os.path.join(ROOT, "scoring"))
import gen_yang2023 as Y  # noqa: E402  (read-only helpers)
import gen_cases as gc  # noqa: E402
import ssi_geom as g  # noqa: E402

CID = "y23_fig1a_two_loops_cross"
TAU = 2 * math.pi

# fitted shape (fit_y23_fig1a_two_loops_cross.py, degree-8 variant without the shear term, rounded)
PARAMS = dict(c0=0.19555, c2=0.85534, c4=-1.32101, c6=1.30006, kappa=1.03663, beta=0.26748, beta5=-0.24973,
              gamma=-0.13576)
CAMERA = dict(proj="persp", el=63.874, az=2.937, roll=-30.835, scale=1614.07, dist=7.1808, cx=206.09, cy=170.82,
              target=[0.0, 0.0, 0.0])

# ------------------------------------------------------------------------------------------- polynomials
# 1D power coefficients, lowest order first, in s


def _pc(p=PARAMS):
    a = np.polynomial.polynomial.polymul([1.0, 0.0, -1.0], [p["c0"], 0, p["c2"], 0, p["c4"], 0, p["c6"]])
    yc = np.array([0, 0, 0, p["beta"], 0, p["beta5"]], float)
    zc = np.array([0, 0, p["gamma"]], float)
    return np.trim_zeros(np.asarray(a, float), "b"), yc, zc


def pv(c, s):
    return np.polynomial.polynomial.polyval(s, c)


def bernstein(c, deg, lo=-1.0, hi=1.0):
    """Power coefficients c (in s) -> Bernstein coefficients of degree deg on s in [lo, hi]."""
    # substitute s = lo + (hi - lo) u, then power -> Bernstein in u
    P = np.polynomial.polynomial
    cu = np.zeros(deg + 1)
    base = np.array([1.0])
    lin = np.array([lo, hi - lo])
    for k, ck in enumerate(c):
        if k:
            base = P.polymul(base, lin)
        cu[:len(base)] += ck * base
    out = np.zeros(deg + 1)
    for j in range(deg + 1):
        out[j] = sum(math.comb(j, i) / math.comb(deg, i) * cu[i] for i in range(j + 1))
    return out


# ------------------------------------------------------------------------------------------- surfaces

def full_circle_truth_hom(a, yc, zc, kappa):
    """Homogeneous power coefficients in (u = s, v = t), t in [0, 1], of the blue tube, with the full
    circle parametrised once by cos th = (D^2 - N^2) / W, sin th = 2 N D / W, N = t - 1/2, D = t (1 - t),
    W = N^2 + D^2 > 0 (t = 0 and t = 1 both give th = pi)."""
    P = np.polynomial.polynomial
    N, D = np.array([-0.5, 1.0]), np.array([0.0, 1.0, -1.0])
    Wt = P.polyadd(P.polymul(N, N), P.polymul(D, D))
    Ct = P.polysub(P.polymul(D, D), P.polymul(N, N))
    St = 2 * P.polymul(N, D)
    outer = lambda cs, ct: np.outer(np.pad(cs, (0, 0)), ct)

    def add(*arrs):
        n = max(x.shape[0] for x in arrs)
        m = max(x.shape[1] for x in arrs)
        out = np.zeros((n, m))
        for x in arrs:
            out[:x.shape[0], :x.shape[1]] += x
        return out
    Xh = outer(np.array([0.0, 1.0]), Wt)
    Yh = add(outer(yc, Wt), outer(a, Ct))
    Zh = add(outer(zc, Wt), outer(kappa * np.asarray(a), St))
    Wh = outer(np.array([1.0]), Wt)
    return [Xh, Yh, Zh, Wh]


def surfaces(p=PARAMS):
    a, yc, zc = _pc(p)
    deg = 8
    ab, yb, zb, sb = (bernstein(c, deg) for c in (a, yc, zc, np.array([0.0, 1.0])))
    cp, cw, ck = g.arc_ctrl(0, 360)
    net = np.zeros((deg + 1, len(cp), 4))
    for i in range(deg + 1):
        for j in range(len(cp)):
            P3 = np.array([sb[i], yb[i] + ab[i] * cp[j, 0], zb[i] + p["kappa"] * ab[i] * cp[j, 1]])
            net[i, j] = [*(P3 * cw[j]), cw[j]]
    U = [-1.0] * (deg + 1) + [1.0] * (deg + 1)
    hom = full_circle_truth_hom(a, yc, zc, p["kappa"])
    tA = {"kind": "param", "hom": [h.tolist() for h in hom], "domain": [-1.0, 1.0, 0.0, 1.0]}
    tB = {"kind": "param", "hom": [hom[1].tolist(), hom[0].tolist(), hom[2].tolist(), hom[3].tolist()],
          "domain": [-1.0, 1.0, 0.0, 1.0]}
    insA = lambda P_, tol: np.abs(P_[:, 0]) <= 1 + tol
    insB = lambda P_, tol: np.abs(P_[:, 1]) <= 1 + tol
    A = g.Nurbs(deg, 2, U, ck, net, tA, insA,
                "petal tube about the S-axis (s, b s^3 + b5 s^5, g s^2), radius (1-s^2)q(s^2), pointed ends")
    B = g.Nurbs(deg, 2, U, ck, net[..., [1, 0, 2, 3]], tB, insB, "petal tube, mirror image of the blue in x = y")
    return A, B


def F_A(Q, p=PARAMS):
    """Implicit equation of the blue tube (zero on it): uses x = s exactly."""
    a, yc, zc = _pc(p)
    s = Q[..., 0]
    av = pv(a, s)
    cy = (Q[..., 1] - pv(yc, s)) / av
    sz = (Q[..., 2] - pv(zc, s)) / (p["kappa"] * av)
    return cy * cy + sz * sz - 1.0


def s_star(sgn, p=PARAMS):
    a, yc, _ = _pc(p)
    return optimize.brentq(lambda s: pv(a, s) - abs(sgn * s - pv(yc, s)), 1e-12, 1 - 1e-12, xtol=1e-15)


def loop(sgn, p=PARAMS):
    """Closed loop in the plane y = sgn x, as a function of t in [0, 2 pi]."""
    a, yc, zc = _pc(p)
    ss = s_star(sgn, p)

    def f(t):
        t = np.asarray(t, float)
        s = ss * np.cos(t)
        r = sgn * s - pv(yc, s)
        root = np.sqrt(np.clip(pv(a, s) ** 2 - r * r, 0, None))
        z = pv(zc, s) + p["kappa"] * np.sign(np.sin(t)) * root
        return np.column_stack([s, sgn * s, z])
    return f


# ------------------------------------------------------------------------------------------- checks

def normals(Pt, p=PARAMS, h=1e-6):
    """Unit normals of A and B at Pt from the gradients of their implicit equations."""
    def grad(F, X):
        return np.array([(F(X + e) - F(X - e)) / (2 * h) for e in np.eye(3) * h])
    FB = lambda Q: F_A(Q[..., [1, 0, 2]], p)
    nA, nB = grad(F_A, Pt), grad(FB, Pt)
    return nA / np.linalg.norm(nA), nB / np.linalg.norm(nB)


def certify(p=PARAMS, ns=2001, nt=2001):
    """Sign changes of F_B on a fine (s, th) grid of A, where B exists (|y| <= 1): report the largest
    distance of any sign-change cell from the two mirror planes |x| = |y|."""
    a, yc, zc = _pc(p)
    s = np.linspace(-1, 1, ns)[:, None]
    th = np.linspace(0, TAU, nt)[None, :]
    av = pv(a, s)
    Q = np.stack([s + 0 * th, pv(yc, s) + av * np.cos(th), pv(zc, s) + p["kappa"] * av * np.sin(th)], -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Fb = F_A(Q[..., [1, 0, 2]], p)
    valid = (np.abs(Q[..., 1]) < 1) & np.isfinite(Fb)
    sc = np.zeros_like(valid)
    for ax in (0, 1):
        d = np.diff(np.sign(Fb), axis=ax) != 0
        v = valid[:-1] & valid[1:] if ax == 0 else valid[:, :-1] & valid[:, 1:]
        sl = (slice(None, -1), slice(None)) if ax == 0 else (slice(None), slice(None, -1))
        sc[sl] |= d & v
    cells = Q[sc]
    dplane = np.minimum(np.abs(cells[:, 0] - cells[:, 1]), np.abs(cells[:, 0] + cells[:, 1])) / math.sqrt(2)
    step = float(np.max(np.linalg.norm(np.diff(Q, axis=1), axis=-1)))
    # a touching (sign-preserving) contact elsewhere would show as a small first-order distance |F_B|/|grad F_B|
    from scipy.spatial import cKDTree
    Lp = np.vstack([loop(sg, p)(np.linspace(0, TAU, 20001)) for sg in (1, -1)])
    Qv = Q[valid][::7]
    Qf = Qv[cKDTree(Lp).query(Qv)[0] > 0.02]
    h = 1e-7
    FB = lambda X: F_A(X[..., [1, 0, 2]], p)
    gr = np.stack([(FB(Qf + e) - FB(Qf - e)) / (2 * h) for e in np.eye(3) * h], -1)
    dist1 = np.abs(FB(Qf)) / np.linalg.norm(gr, axis=-1)
    return float(dplane.max()), step, int(len(cells)), float(dist1.min())


def checks(p=PARAMS):
    out = {}
    z0 = p["kappa"] * p["c0"]
    for zz in (z0, -z0):
        nA, nB = normals(np.array([0.0, 0.0, zz]), p)
        out["crossing (0,0,%+.5f): sin(normal angle)" % zz] = float(np.linalg.norm(np.cross(nA, nB)))
    for sgn in (1, -1):
        f = loop(sgn, p)
        t = np.linspace(0, TAU, 4001)
        P = f(t)
        ok = np.linalg.norm(P - np.array([0, 0, z0]), axis=1) > 0.02
        ok &= np.linalg.norm(P - np.array([0, 0, -z0]), axis=1) > 0.02
        sins = [np.linalg.norm(np.cross(*normals(Pt, p))) for Pt in P[ok][::10]]
        out["loop y=%+dx: s*" % sgn] = s_star(sgn, p)
        out["loop y=%+dx: min sin(normal angle) beyond 0.02 of the crossings" % sgn] = float(min(sins))
    dmax, step, n, dfar = certify(p)
    out["certify: min first-order distance A->B farther than 0.02 from the loops"] = dfar
    out["certify: sign-change cells of F_B on A"] = n
    out["certify: max distance of a sign change from the mirror planes"] = dmax
    out["certify: grid step on A"] = step
    return out


# ------------------------------------------------------------------------------------------- case

def build():
    A, B = surfaces()
    p = PARAMS
    z0 = p["kappa"] * p["c0"]
    cam = CAMERA
    Y.add(CID, "Fig.1(a): two petal tubes, two planar loops crossing at two points", A, B,
          Y.SRC + "Fig. 1 (a)", "cross intersections: 'the intersection curve contains two loops that are crossing "
          "at two common points'",
          "topology: two closed loops crossing at two points (tangential saddle contacts with common normal "
          "(0,0,+-1)); appearance, refitted to the 600 dpi panels: two closed petal tubes with pointed lobes and a "
          "waist, the orange the mirror image of the blue in the plane x = y (about 90 degrees apart in view), the "
          "loops thin and elongated along the diagonals, ending where the silhouettes cross; fitted camera "
          "perspective el %.2f az %.2f roll %.2f, eye %.2f units from the centre, focal %.0f px "
          "(render_match.py); against the Surfaces panel: silhouette IoU 0.946, blue-visible IoU 0.909, "
          "orange-visible IoU 0.909 (previous lobed cylinders under their own best-fitting camera: 0.796, 0.350, "
          "0.256), loops within ~1-2 px of the paper's red curves" % (cam["el"], cam["az"], cam["roll"],
                                                                          cam["dist"], cam["scale"]),
          "bi-degree (8, 2) rational (not stated in the paper; degree 6 in s fits at IoU 0.934, with blunter "
          "tips); the exact profile, the tube cross-section (round here, ratio %.3f) and the tip sweep are "
          "fitted to the outline, not known; the mirror relation between the surfaces is our choice that keeps "
          "the loops planar and exact; tips are slightly blunter than the paper's" % p["kappa"],
          "paper: all methods correct",
          curves=[(loop(1), 0, TAU, True, "crossing"), (loop(-1), 0, TAU, True, "crossing")],
          singular=[(0.0, 0.0, z0), (0.0, 0.0, -z0)])
    case = gc.CASES[-1]
    case["notes"] = ("loops in closed form: x = s, y = +-s, z = gamma s^2 +- kappa sqrt(a(s)^2 - (+-s - yc(s))^2); "
                     "crossing points (0, 0, +-kappa c0) with parallel normals")
    case["truth_method"] = "closed form (planar loops on mirror-symmetric tubes), surfaces exact rational"
    return case


def write(case):
    with open(os.path.join(ROOT, "cases", "data", CID + ".json"), "w") as fh:
        json.dump(case, fh)


if __name__ == "__main__":
    case = build()
    write(case)
    for k, v in checks().items():
        print("%-70s %s" % (k, v))
    if "--compare" in sys.argv:
        import efg_compare as ec
        print(ec.compare(case, CAMERA, n=241))
