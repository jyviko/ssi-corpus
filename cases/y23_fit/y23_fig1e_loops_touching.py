"""Fig. 1(e) of Yang, Jia & Yan 2023, refitted to the paper's "Surfaces" panel.

What the panels show (cases/y23_fit/paper/y23_fig1e_loops_touching_*.png, 600 dpi crops):
  * blue: a torus seen obliquely, its hole visible as a dark ellipse in the middle of the panel.
  * orange: a fatter torus on the left, seen nearly edge-on; its tube runs into the blue's hole (the two
    rings are linked like chain links).
  * red: two closed loops side by side that touch at one point. Zoomed (600 dpi), each loop is a smooth oval
    through the contact point and the two ovals share their tangent there: they kiss, they do not cross.

Construction (exact surfaces, certified traced truth):
  * orange O: torus centre 0, axis z, major radius 1, minor radius ro.
  * blue B: torus with major radius Rb, minor radius rb, centre (d, 0, 0), its core plane containing the
    x axis and tilted by alpha from the xy plane. d = 1 + ro - (Rb - rb), so the orange's outer equator
    point P = (1 + ro, 0, 0) lies on the inner equator of the blue (the wall of its hole), both normals along
    x: the tori are tangent at P.
  * rb is not free: it solves det(relative second fundamental form at P) = 0 (solve_rb, Brent to 1e-15), so
    the tangency is degenerate. The configuration is symmetric under the half turn about the x axis, so the
    height difference of the two surfaces near P is an even function: no cubic terms, and the degenerate
    tangency is a tacnode, G ~ u^2 - c v^4. With c > 0 (checked by the trace: two loops reach P) the
    intersection is two loops tangent to each other at P, as drawn ("two loops in contact at one point").
  * both surfaces are exact rational bi-quadratic torus NURBS. Truth: predictor-corrector continuation on
    F(theta, phi) = E_blue(orange(theta, phi)) = 0 with a Newton corrector to machine precision (trace_loops),
    every grid sign change covered, each loop closed through the exact point P.

Run: .venv/bin/python -I cases/y23_fit/y23_fig1e_loops_touching.py [--compare]
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
import implicit_trace as it  # noqa: E402
import ssi_geom as g  # noqa: E402

CID = "y23_fig1e_loops_touching"

PARAMS = dict(ro=0.56, Rb=1.10, alpha=109.2)
PARAMS["rb"] = None  # set below from the tacnode condition
CAMERA = dict(proj="persp", el=24.42, az=-78.66, roll=3.89, target=[0.0, 0.0, 0.0], scale=4297.6, cx=186.57, cy=193.48,
              dist=47.80)
RESULT_CAMERA = dict(CAMERA, scale=4396.4, cx=184.48, cy=189.06)
FIT = dict(before=dict(iou=0.863, iou_blue=0.617, iou_orange=0.351), after=dict(iou=0.968, iou_blue=0.886, iou_orange=0.882))


def frames(p=PARAMS):
    d = 1.0 + p["ro"] - (p["Rb"] - p["rb"])
    al = math.radians(p["alpha"])
    fo = g.Frame((0, 0, 0), (1, 0, 0), (0, 1, 0))
    fb = g.Frame((d, 0, 0), (1, 0, 0), (0, math.cos(al), math.sin(al)))
    return fo, fb, d


def surfaces(p=PARAMS):
    fo, fb, _ = frames(p)
    blue = g.nurbs_torus(fb, p["Rb"], p["rb"], (0, 360), (0, 360), "torus R=%g r=%g (ring)" % (p["Rb"], p["rb"]))
    orange = g.nurbs_torus(fo, 1.0, p["ro"], (0, 360), (0, 360), "torus R=1 r=%g (tube through the ring)" % p["ro"])
    return blue, orange


def torus_implicit(fr, R, r):
    A = np.array([fr.x, fr.y, fr.z])

    def E(Pt):
        Lc = (Pt - fr.o) @ A.T
        s2 = np.sum(Lc * Lc, axis=-1)
        return (s2 + R * R - r * r) ** 2 - 4 * R * R * (Lc[..., 0] ** 2 + Lc[..., 1] ** 2)

    def gE(Pt):
        Lc = (Pt - fr.o) @ A.T
        s2 = np.sum(Lc * Lc, axis=-1)
        gl = 4 * (s2 + R * R - r * r)[..., None] * Lc - 8 * R * R * np.stack(
            [Lc[..., 0], Lc[..., 1], 0 * Lc[..., 0]], -1)
        return gl @ A
    return E, gE


def solve_rb(ro, Rb, alpha_deg):
    """Blue minor radius that makes the tangency at P degenerate (det of the relative second fundamental
    form = 0), turning the X-node into a tacnode: two loops touching at P. Root of a smooth scalar
    function, bracketed on (0, Rb) and polished by Brent's method to 1e-15."""
    from scipy.optimize import brentq
    al = math.radians(alpha_deg)

    def detQ(rb):
        c, s_ = math.cos(al), math.sin(al)
        A, B = 1 / rb, 1 / (Rb - rb)
        qyy = A * s_ * s_ - B * c * c + 1 / (1 + ro)
        qzz = A * c * c - B * s_ * s_ + 1 / ro
        qyz = -(A + B) * s_ * c
        return qyy * qzz - qyz * qyz
    xs = np.linspace(1e-3 * Rb, Rb * (1 - 1e-3), 4001)
    v = np.array([detQ(x) for x in xs])
    roots = [brentq(detQ, xs[i], xs[i + 1], xtol=1e-15) for i in range(len(xs) - 1) if v[i] * v[i + 1] < 0]
    return roots


def implicit_problem(p=PARAMS):
    fo, fb, _ = frames(p)
    E, gE = torus_implicit(fb, p["Rb"], p["rb"])
    return it.TorusImplicit(fo, 1.0, p["ro"], E, gE), E, gE


def _newton_to_curve(ti, x, iters=60):
    """Move (theta, phi) to F = 0 along the gradient (Gauss-Newton on one equation)."""
    for _ in range(iters):
        f = float(ti.F(x[0], x[1]))
        gt, gp = (float(v) for v in ti.gradF(x[0], x[1]))
        g2 = gt * gt + gp * gp
        if g2 == 0:
            break
        dx = -f / g2 * np.array([gt, gp])
        x = x + dx
        if np.hypot(*dx) < 1e-16:
            break
    return x


def trace_loops(p=PARAMS, N=900, h=2e-3, delta=2e-3, excl=0.08):
    """Intersection as closed loops through the tacnode P, by predictor-corrector continuation on
    F(theta, phi) = E_blue(T_orange(theta, phi)) = 0 (orange torus parameters).

    Seeds: every sign change of F on an N x N grid, refined to F = 0, outside a small disc around P.
    Each seed not yet covered is traced in both directions (step h, Newton corrector) until the path comes
    within delta of P (then P itself closes it) or returns to its start. All grid crossings must end up
    covered by a traced path (checked). Returns (list of closed (M,3) polylines, P_param)."""
    ti, E, gE = implicit_problem(p)
    TP = (0.0, 0.0)  # P = (1 + ro, 0, 0) is theta = 0, phi = 0 on the orange torus
    per = lambda d: (d + math.pi) % (2 * math.pi) - math.pi
    hgrid = 2 * math.pi / N
    th, ph = np.meshgrid(np.arange(N) * hgrid, np.arange(N) * hgrid, indexing="ij")
    Fv = ti.F(th, ph)
    seeds = []
    for ax in (0, 1):
        Fn = np.roll(Fv, -1, axis=ax)
        I, J = np.nonzero(Fv * Fn < 0)
        for i, j in zip(I, J):
            a = np.array([i * hgrid, j * hgrid])
            b = a + (np.array([hgrid, 0]) if ax == 0 else np.array([0, hgrid]))
            t = Fv[i, j] / (Fv[i, j] - Fn[i, j])
            q = a + t * (b - a)
            if np.hypot(per(q[0] - TP[0]), per(q[1] - TP[1])) > excl:
                seeds.append(q)
    seeds = np.array(seeds)
    covered = np.zeros(len(seeds), bool)
    loops = []

    def step_dir(x, sgn):
        gt, gp = (float(v) for v in ti.gradF(x[0], x[1]))
        t = np.array([-gp, gt]) * sgn
        return t / np.hypot(*t)

    def walk(x0, sgn, maxn=200000):
        pts = [x0]
        x = x0.copy()
        d_prev = None
        for _ in range(maxn):
            dP = np.hypot(per(x[0] - TP[0]), per(x[1] - TP[1]))
            hh = min(h, max(0.25 * dP, 1e-5))
            t = step_dir(x, sgn)
            if d_prev is not None and np.dot(t, d_prev) < 0:
                t = -t
            xn = _newton_to_curve(ti, x + hh * t)
            d_prev = (xn - x) / max(np.hypot(*(xn - x)), 1e-300)
            x = xn
            pts.append(x.copy())
            if np.hypot(per(x[0] - TP[0]), per(x[1] - TP[1])) < delta:
                return pts, "P"
            if len(pts) > 20 and np.hypot(per(x[0] - x0[0]), per(x[1] - x0[1])) < 0.6 * h:
                return pts, "closed"
        raise RuntimeError("continuation did not terminate")

    for k in range(len(seeds)):
        if covered[k]:
            continue
        x0 = _newton_to_curve(ti, seeds[k].copy())
        fwd, end1 = walk(x0, +1)
        if end1 == "closed":
            path = fwd
        else:
            bwd, end2 = walk(x0, -1)
            assert end2 == "P", "loop does not reach P in both directions"
            path = [np.array(TP)] + bwd[::-1] + fwd[1:] + [np.array(TP)]
        Q = np.array(path)
        # unwrap and mark covered seeds
        d = np.diff(Q, axis=0)
        d = (d + math.pi) % (2 * math.pi) - math.pi
        Qu = Q[0] + np.vstack([np.zeros((1, 2)), np.cumsum(d, axis=0)])
        for kk in np.nonzero(~covered)[0]:
            dd = np.hypot(per(Qu[:, 0] - seeds[kk][0]), per(Qu[:, 1] - seeds[kk][1]))
            if dd.min() < 2.5 * hgrid:
                covered[kk] = True
        covered[k] = True
        loops.append(Qu)
    P3 = [ti.T(Q[:, 0], Q[:, 1]) for Q in loops]
    return P3, ti, E, gE


PARAMS["rb"] = solve_rb(PARAMS["ro"], PARAMS["Rb"], PARAMS["alpha"])[0]


def fmt_cam(c):
    s = "%s el=%.2f az=%.2f roll=%.2f" % (c["proj"], c["el"], c["az"], c["roll"])
    if c["proj"] == "persp":
        s += " dist=%.3f (focal %.0f px" % (c["dist"], c["scale"])
    else:
        s += " (scale %.1f px/unit" % c["scale"]
    return s + ", target at (%.1f, %.1f) px) in a 437x386 panel" % (c["cx"], c["cy"])


def node_check(p=PARAMS):
    """Tangency at P = (1 + ro, 0, 0): both normals along x; relative curvature indefinite (crossing)."""
    fo, fb, d = frames(p)
    P = np.array([1.0 + p["ro"], 0.0, 0.0])
    Eb, gEb = torus_implicit(fb, p["Rb"], p["rb"])
    Eo, gEo = torus_implicit(fo, 1.0, p["ro"])
    na, nb = gEo(P[None])[0], gEb(P[None])[0]
    sin = float(np.linalg.norm(np.cross(na, nb)) / np.linalg.norm(na) / np.linalg.norm(nb))
    return dict(P=P.tolist(), on_orange=float(Eo(P[None])[0]), on_blue=float(Eb(P[None])[0]), sin_normals=sin)


def build():
    p = PARAMS
    blue, orange = surfaces(p)
    loops, ti, E, gE = trace_loops(p)
    nc = node_check(p)
    assert abs(nc["on_orange"]) < 1e-12 and abs(nc["on_blue"]) < 1e-12 and nc["sin_normals"] < 1e-12
    assert len(loops) == 2, len(loops)
    Pn = np.array(nc["P"])
    for L in loops:
        assert np.linalg.norm(L[0] - Pn) < 1e-12 and np.linalg.norm(L[-1] - Pn) < 1e-12
    dist_blue = max(float((np.abs(E(L[1:-1])) / np.linalg.norm(gE(L[1:-1]), axis=1)).max()) for L in loops)
    assert dist_blue < 1e-12, dist_blue
    nc["max_vertex_distance_to_blue"] = dist_blue
    nc["rb"] = p["rb"]
    branches = [(L, True) for L in loops]
    fa, fb_ = FIT["after"], FIT["before"]
    Y.add(CID, "Fig.1(e): two linked tori, two loops meeting at one point", blue, orange,
          Y.SRC + "Fig. 1 (e)", "'the intersection curve contains two loops that are contact at one common point'",
          "topology: two closed loops touching at one common point P (the tori are tangent there, normals parallel, "
          "sin %.0e; degenerate tangency, so the loops are tangent to each other at P, kissing, as the zoomed "
          "figure shows); appearance fitted to the Surfaces panel (silhouette, blue and orange "
          "regions, and the red loops of the result row): the orange torus' tube passes through the hole of the "
          "blue torus and rubs the wall of the hole; camera %s; silhouette IoU %.3f (blue %.3f, orange %.3f), "
          "previous construction %.3f under its own best camera" % (
              nc["sin_normals"], fmt_cam(CAMERA), fa["iou"] or 0, fa["iou_blue"] or 0, fa["iou_orange"] or 0,
              fb_["iou"] or 0),
          "radii and placement recovered from one view (orange R=1, r=%g; blue R=%g, r=%.12g from the tacnode "
          "condition, ring plane tilted %g deg about the line of centres); the two loops are congruent (half-turn "
          "symmetry); not reproduced: in the paper the loops are smaller and sit side by side in the blue's hole, "
          "ours wrap further round the orange tube and overlap in this view (the loop size is not a free parameter "
          "once the tori are fitted); bi-degree not stated (ours rational (2,2)); ground truth by certified "
          "continuation (trace_loops)" % (p["ro"], p["Rb"], p["rb"], p["alpha"]),
          "paper: " + Y.OPEN + " returned nothing",
          curves=Y.polyline_curves(branches), singular=[tuple(Pn)], cov_tol=5e-3)
    case = gc.CASES[-1]
    case["truth_method"] = ("certified continuation on the orange torus parameters (Newton to F = 0, every vertex "
                            "within %.1e of the blue torus); the touching point P is exact" % dist_blue)
    return case, nc


def main():
    case, nc = build()
    with open(os.path.join(ROOT, "cases", "data", CID + ".json"), "w") as fh:
        json.dump(case, fh)
    print(CID, "branches", [(len(b["points"]), b["closed"]) for b in case["expected"]["branches"]],
          "singular", case["expected"]["singular_points"], json.dumps(nc))
    if "--compare" in sys.argv:
        import efg_compare
        print("after", efg_compare.compare(case, CAMERA, result_camera=RESULT_CAMERA))


if __name__ == "__main__":
    main()
