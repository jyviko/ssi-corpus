"""Cases derived from Yang, Jia & Yan, "Topology Guaranteed B-Spline
Surface/Surface Intersection", ACM TOG 42(6), 2023.

The paper publishes no surface data for Fig. 1, Table 2 or Table 6, so those
cases are RECONSTRUCTIONS: two polynomial (or rational) B-spline patches with
the stated bi-degree where possible and the stated intersection topology,
built so that the exact intersection is known in closed form. They are not
the authors' surfaces. Table 3/4 (torus x ellipsoid) publishes the exact
shape parameters; those cases use them directly, with ground truth computed by
our own algebra (scoring/implicit_trace.py).

Reconstructions use exact constructions whose intersection is known in closed
form (or, for tori and the appendix example, computed by our own
numerical-algebraic tracing): polynomial graphs and sheared graphs over
different footprints, rational cylinder / torus / hyperboloid / sphere
patches, a lobed rational cylinder, and patches with a collapsed apex edge.
The paper's comparison methods are named generically ("the open-source
kernel", "the commercial kernel", "the spline library").

Run: python3 -I cases/gen_yang2023.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scoring"))
import gen_cases as gc  # noqa: E402
import implicit_trace as it  # noqa: E402
import param_trace as pt  # noqa: E402
import ssi_geom as g  # noqa: E402

SRC = "reconstruction of Yang, Jia & Yan 2023, "
TAU = 2 * math.pi


def P(*rows):
    """2D power coefficients from rows c[i][j] (u^i w^j), ragged rows padded."""
    n = max(len(r) for r in rows)
    return np.array([list(r) + [0.0] * (n - len(r)) for r in rows], float)


def mono(i, j, c=1.0):
    a = np.zeros((i + 1, j + 1))
    a[i, j] = c
    return a


def poly(*terms):
    out = np.zeros((1, 1))
    for t in terms:
        out = g.p2_add(out, t)
    return out


def mul(*fs):
    out = np.ones((1, 1))
    for f in fs:
        out = g.p2_mul(out, f)
    return out


def pair(q, D, shear, ur, wr, deg, lab):
    if deg is None:  # both surfaces get the same bi-degree (exact Bezier degree elevation)
        H1 = g.p2_add(q, D)
        sd = len(shear) - 1 if shear is not None else 0
        deg = (max(g.p2_degree(H1)[0], g.p2_degree(q)[0], sd, 1), max(g.p2_degree(H1)[1], g.p2_degree(q)[1], 1))
    s1 = g.sheared_heightfield_nurbs(g.p2_add(q, D), shear, ur, wr, lab + " surface A (q + D)", degree=deg)
    s2 = g.sheared_heightfield_nurbs(q, shear, ur, wr, lab + " surface B (q)", degree=deg)
    return s1, s2


def uv_curve(q, shear, fu, fw):
    """Curve given in (u,w) by fu(t), fw(t) mapped onto surface B."""
    return lambda t: g.sheared_point(q, shear, fu(t), fw(t))


def meta(case, source, stated, matched, guessed, methods=None):
    case["source"] = source
    case["paper_topology"] = stated
    case["matched"] = matched
    case["guessed"] = guessed
    if methods:
        case["paper_other_methods"] = methods


def add(cid, title, s1, s2, src, stated, matched, guessed, methods=None, cls="Y23 reconstruction", **kw):
    gc.add_case(cid, cls, title, s1, s2, **kw)
    meta(gc.CASES[-1], src, stated, matched, guessed, methods)


OPEN = "the open-source kernel"
COMM = "the commercial kernel"
SPL = "the spline library"


def compose(Hxy, X, Y):
    """H(x, y) with x = X(u,v), y = Y(u,v) (all power-coefficient arrays)."""
    Hxy = g.p2_trim(Hxy)
    out = np.zeros((1, 1))
    xp = [np.ones((1, 1))]
    for _ in range(Hxy.shape[0]):
        xp.append(g.p2_mul(xp[-1], X))
    yp = [np.ones((1, 1))]
    for _ in range(Hxy.shape[1]):
        yp.append(g.p2_mul(yp[-1], Y))
    for i in range(Hxy.shape[0]):
        for j in range(Hxy.shape[1]):
            if Hxy[i, j]:
                out = g.p2_add(out, Hxy[i, j] * g.p2_mul(xp[i], yp[j]))
    return g.p2_trim(out)


def graph_patch(H, ur, wr, label, degree=None, shear=None):
    """z = H(u, w) over (u, w + shear(u)) - sheared height field."""
    return g.sheared_heightfield_nurbs(H, shear, ur, wr, label, degree=degree)


def fig1():
    # (a) two lobed cylinders at 90 degrees: Steinmetz-type loops crossing at two points
    L = [1.5, 1.2, 0.5, 1.2, 1.5, 1.2, 0.5, 1.2, 1.5]
    A = g.elevate_any(g.lobed_cylinder(g.Frame((0, 0, 0), (0, 1, 0), (0, 0, 1)), 1.0, L, "lobed cylinder, axis x"), 0, 1)
    B = g.elevate_any(g.lobed_cylinder(g.Frame((0, 0, 0), (1, 0, 0), (0, 0, 1)), 1.0, L, "lobed cylinder, axis y"), 0, 1)
    ell = lambda sg: (lambda t: np.column_stack([sg * np.cos(t), np.cos(t), np.sin(t)]))
    add("y23_fig1a_two_loops_cross", "Fig.1(a): two loops crossing at two points (lobed cylinders at 90 degrees)", A, B,
        SRC + "Fig. 1 (a)", "cross intersections: 'the intersection curve contains two loops that are crossing at two "
        "common points'",
        "topology: two closed loops crossing at two points; appearance: two congruent lobed (pinched) patches, one "
        "rotated 90 degrees about the common axis, loops elongated and perpendicular in top view",
        "the surfaces are lobed rational cylinder patches (exact cylinders x^2... with angle-dependent length), so the "
        "loops are the two Steinmetz ellipses x = +-y; bi-degree not stated in the paper (ours (2,2) rational)",
        "paper: all methods correct",
        curves=[(ell(1), 0, TAU, True, "crossing"), (ell(-1), 0, TAU, True, "crossing")],
        singular=[(0, 0, 1), (0, 0, -1)])
    # (b) barrel-like sheet (cylinder sector) and a protruding sheet x = h(y, z)
    al, be, ga = 0.2, 1.0, 1.6
    blue = g.elevate_any(g.nurbs_cylinder(g.Frame((0, 0, -1), (1, 0, 0), (0, 1, 0)), 1.0, (-50, 50), (0, 2),
                                          "cylinder sector (barrel-like)"), 0, 1)
    h = poly(mono(0, 0, 1.0), mono(2, 0, -al), mono(0, 2, be), mono(0, 4, -ga))  # h(y, z)
    orange = g.poly_patch(h, P([0, 0], [1]), P([0, 1]), (-1.0, 1.0), (-1.3, 1.3), "sheet x = 1 - 0.2y^2 + z^2 - 1.6z^4",
                          {"kind": "heightfield", "coeffs": h.tolist(), "axes": [1, 2, 0]},
                          lambda Q, tol: (np.abs(Q[:, 1]) <= 1.0 + tol) & (np.abs(Q[:, 2]) <= 1.3 + tol))
    ym = math.sin(math.radians(50))

    def arc(sg):
        def f(y):
            xc = np.sqrt(1 - y * y)
            Z = (be + np.sqrt(be * be + 4 * ga * (1 - al * y * y - xc))) / (2 * ga)
            return np.column_stack([xc, y, sg * np.sqrt(Z)])
        return f
    add("y23_fig1b_two_arcs_and_point", "Fig.1(b): two curve branches plus an isolated contact point", blue, orange,
        SRC + "Fig. 1 (b)", "'the two surfaces intersect at two separate curve branches and are in contact at an "
        "isolated point'",
        "topology: two open branches (rim to rim) + one isolated tangential point at the centre of the front face; "
        "appearance: barrel-like sheet seen front-on, second sheet protruding beyond its top and bottom",
        "the second sheet x = h(y,z) (bi-degree (2,4)); the barrel is a rational cylinder sector (2,2); paper states "
        "no bi-degree", "paper: " + OPEN + " lost the isolated point",
        curves=[(arc(1), -ym, ym, False, "crossing"), (arc(-1), -ym, ym, False, "crossing")],
        points=[(1.0, 0.0, 0.0)], cov_tol=1e-2)
    # (c) curved triangle (unchanged, FAITHFUL)
    a = 0.6
    q = P([0, 0, 0.5], [0, -0.7], [0.5])
    D = 0.15 * mul(mono(1, 0), mono(0, 1), poly(mono(1, 0), mono(0, 1), mono(1, 1, a), mono(0, 0, -1)))
    s1, s2 = pair(q, D, None, (0, 1), (0, 1), None, "Fig1c")
    add("y23_fig1c_boundary_triangle", "Fig.1(c): curved triangle, two edges on the patch boundaries", s1, s2,
        SRC + "Fig. 1 (c)", "'the two surfaces are very close to each other in the triangular domain, and the "
        "intersection curve is a curved triangle, two edges of which are the boundary curves of the two surfaces and "
        "the left edge is a regular intersection curve'",
        "topology: closed curved triangle; two edges on (shared) patch boundaries, one interior curve",
        "bi-degree (not stated; ours (2,2)); both patches share the boundary edges u=0, w=0",
        "paper: " + OPEN + " lost the two boundary edges",
        curves=[(uv_curve(q, None, lambda t: 0 * t, lambda t: t), 0, 1, False, "boundary"),
                (uv_curve(q, None, lambda t: t, lambda t: 0 * t), 0, 1, False, "boundary"),
                (uv_curve(q, None, lambda t: t, lambda t: (1 - t) / (1 + a * t)), 0, 1, False, "crossing")],
        singular=[g.sheared_point(q, None, [x], [y])[0] for x, y in ((0, 0), (1, 0), (0, 1))])
    # (d) two quadrilaterals sharing a tangential edge (unchanged, FAITHFUL)
    q = poly(mono(0, 3, 0.3), mono(0, 2, -0.9), mono(0, 1, 0.5), mono(2, 0, 0.3))
    D = 0.6 * mul(mono(1, 0), poly(mono(1, 0), mono(0, 0, -1)), mono(0, 1), poly(mono(0, 1), mono(0, 0, -2)),
                  poly(mono(0, 1), mono(0, 0, -1)), poly(mono(0, 1), mono(0, 0, -1)))
    s1, s2 = pair(q, D, None, (0, 1), (0, 2), None, "Fig1d")
    Lf = lambda u0, u1, w0, w1: uv_curve(q, None, lambda t: u0 + (u1 - u0) * t, lambda t: w0 + (w1 - w0) * t)
    add("y23_fig1d_two_quads", "Fig.1(d): two quadrilaterals sharing a tangential edge, six boundary edges", s1, s2,
        SRC + "Fig. 1 (d)", "'the intersection contains two quadrilaterals sharing one edge. The two surfaces are in "
        "contact along the common edge of the two quadrilaterals, and intersect transversely at the other six curved "
        "edges of the two quadrilaterals, which are also the boundaries of the surfaces'",
        "topology: 6 transversal boundary edges + 1 tangential interior edge, one connected network",
        "bi-degree (not stated; ours (2,4)), shape", "paper: " + OPEN + " returned three of the six edges only",
        curves=[(Lf(0, 0, 0, 1), 0, 1, False, "boundary"), (Lf(0, 0, 1, 2), 0, 1, False, "boundary"),
                (Lf(1, 1, 0, 1), 0, 1, False, "boundary"), (Lf(1, 1, 1, 2), 0, 1, False, "boundary"),
                (Lf(0, 1, 0, 0), 0, 1, False, "boundary"), (Lf(0, 1, 2, 2), 0, 1, False, "boundary"),
                (Lf(0, 1, 1, 1), 0, 1, False, "tangent")],
        singular=[g.sheared_point(q, None, [x], [y])[0] for x, y in ((0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2))],
        cov_tol=1e-2)
    # (e) two linked tori; the tubes meet in two loops through one tangency point (4, 0, 0)
    fa = g.Frame()
    fb = g.Frame((3.25, 0, 0), (1, 0, 0), (0, 0, 1))  # axis -y, core in the xz-plane, links through A's hole
    A = g.nurbs_torus(fa, 3.0, 1.0, (0, 360), (0, 360), "torus R=3 r=1")
    B = g.nurbs_torus(fb, 1.5, 0.75, (0, 360), (0, 360), "torus R=1.5 r=0.75 (linked)")
    E, gE = torus_implicit(fb, 1.5, 0.75)
    branches, crossing, isolated = it.trace(it.TorusImplicit(fa, 3.0, 1.0, E, gE), N=1600)
    add("y23_fig1e_loops_touching", "Fig.1(e): two linked tori, two loops meeting at one point", A, B,
        SRC + "Fig. 1 (e)", "'the intersection curve contains two loops that are contact at one common point'",
        "topology: two closed loops through one common singular (tangency) point; appearance: two interlocked "
        "torus-like tubes",
        "the tori (R=3, r=1) and (R=1.5, r=0.75), centre offset 3.25 so that the tubes are tangent at (4,0,0); "
        "the meeting point is a crossing (the surfaces are tangent there with a saddle-type contact); bi-degree not "
        "stated (ours rational (2,2)); ground truth numerical-algebraic (implicit_trace.py)",
        "paper: " + OPEN + " returned nothing",
        curves=polyline_curves(branches), points=[tuple(p) for p in isolated], singular=[tuple(p) for p in crossing],
        cov_tol=5e-3)
    gc.CASES[-1]["truth_method"] = "numerical-algebraic (implicit_trace.py), vertices on F=0 to ~1e-15 in parameter"
    # (f) high-order contact along a straight line: step sheets z = s(w) and z = s(w) + w^4
    st = poly(mono(0, 1, 1.0), mono(0, 3, -1.0 / 3.0))  # s(w) = w - w^3/3, monotone step on [-1, 1]
    orange = graph_patch(st, (-1, 1), (-1, 1), "step sheet z = w - w^3/3", degree=(1, 4))
    blue = graph_patch(g.p2_add(st, mono(0, 4, 1.0)), (-0.6, 0.6), (-0.9, 0.9), "narrower sheet z = w - w^3/3 + w^4",
                       degree=(1, 4))
    add("y23_fig1f_high_order_contact", "Fig.1(f): high-order contact along a straight line", blue, orange,
        SRC + "Fig. 1 (f)", "'the two surfaces are in high-order contact along a line'",
        "topology: one open straight tangential line; appearance: S-shaped step sheets, the blue narrower",
        "contact order: D = w^4, i.e. the surfaces agree up to third derivatives across the line (paper: 'high-order'); "
        "bi-degree not stated (ours (1,4))", "paper: both kernels returned nothing",
        curves=[(gc.line((-0.6, 0, 0), (1, 0, 0)), 0, 1.2, False, "tangent")], cov_tol=1e-2,
        notes="with D = w^4 the surfaces are within 1e-5 of each other over the band |w| < 0.056")
    # (g) dome and lens with a common apex: contact at the apex and at two rim corners
    hB = poly(mono(0, 0, 1.0), mono(2, 0, -0.35), mono(0, 2, -0.3), mono(0, 1, 0.12))  # dome z(x, y)
    A0, Pp, Qp = np.array([0.0, -1.0]), np.array([-0.8, 0.6]), np.array([0.8, 0.6])
    M = np.array([[A0[0], Pp[0], Qp[0]], [A0[1], Pp[1], Qp[1]], [1, 1, 1]])
    Minv = np.linalg.inv(M)  # barycentric b = Minv @ (x, y, 1)
    bary = [poly(mono(1, 0, Minv[i, 0]), mono(0, 1, Minv[i, 1]), mono(0, 0, Minv[i, 2])) for i in range(3)]
    # D = (b1b2)^2 + (b2b3)^2 + (b3b1)^2 >= 0, zero only at the three vertices, with zero gradient there:
    # the contacts are tangential isolated points
    sq = lambda f: mul(f, f)
    Dg = 3.0 * g.p2_add(g.p2_add(sq(mul(bary[0], bary[1])), sq(mul(bary[1], bary[2]))), sq(mul(bary[2], bary[0])))
    lens_h = g.p2_add(hB, -Dg)

    def tri_patch(H, W, Ht, label):
        X = P([0, -W], [0, 2 * W])  # x = (2u - 1) v W
        Y = P([-1.0, Ht])           # y = -1 + Ht v
        Z = compose(H, X, Y)
        ins = lambda Q, tol: ((Q[:, 1] + 1) / Ht >= -tol) & ((Q[:, 1] + 1) / Ht <= 1 + tol) & \
            (np.abs(Q[:, 0]) <= W * (Q[:, 1] + 1) / Ht + tol)
        return g.poly_patch(X, Y, Z, (0, 1), (0, 1), label, {"kind": "heightfield", "coeffs": g.p2_trim(H).tolist()},
                            ins, degree=(4, 4))
    blue = tri_patch(hB, 1.2, 2.0, "dome (apex edge collapsed)")
    orange = tri_patch(lens_h, 0.8, 1.6, "lens under the dome (apex edge collapsed)")
    pts = [(float(x), float(y), float(g.p2_eval(hB, x, y))) for x, y in (A0, Pp, Qp)]
    add("y23_fig1g_three_points", "Fig.1(g): dome and lens touching at three points (apex + two rim points)",
        blue, orange, SRC + "Fig. 1 (g)",
        "'the two surfaces are in contact at three isolated points'; the paper adds that one point is also a "
        "self-intersection point of the surfaces",
        "topology: three isolated tangential contact points (normals parallel): two on the lens's upper rim, one at "
        "the common bottom apex, where both patches have a collapsed (degenerate) edge; bi-degree (4,4)",
        "lens D = 3((b1b2)^2 + (b2b3)^2 + (b3b1)^2) in barycentric coordinates of the lens triangle, so the lens "
        "touches the dome tangentially at its three corners only; the 'self-intersection' is modelled as the collapsed "
        "apex edge of both parameterisations; degree (4,4) is above the paper's tested bi-quadratic/bi-cubic",
        "paper: " + OPEN + " returned nothing, " + COMM + " lost one point (the apex)", points=pts, cov_tol=1e-2)


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


def polyline_curves(branches):
    out = []
    for Pt, closed in branches:
        out.append((lambda t, pts=Pt: pts[np.clip(np.round(t).astype(int), 0, len(pts) - 1)], 0, len(Pt) - 1, closed,
                    "crossing"))
    return out


def table2():
    # ex1 (3,3) a tangent line: S-step sheet, contact along the straight iso-line u = 0
    step = poly(mono(1, 0, 1.2), mono(3, 0, -0.4), mono(1, 2, 0.15))  # 1.2u - 0.4u^3 + 0.15 u w^2
    blue = graph_patch(step, (-1, 1), (-1, 1), "tall S-step sheet", degree=(3, 3))
    orange = graph_patch(g.p2_add(step, mono(2, 0, 0.8)), (-0.7, 0.7), (-0.6, 0.6), "smaller sheet, same profile",
                         degree=(3, 3))
    add("y23_t2_ex1_tangent_line", "Table 2 ex.1 (3,3): a tangent line", blue, orange, SRC + "Table 2 (example 1)",
        "'a tangent line', bi-degree (3,3); Fig. 10 (left) shows it as an iso-parameter line across the domain",
        "bi-degree (3,3); topology: one open tangential straight line, which is the iso-line u = 0 (s = 0.5) across "
        "the whole domain of the smaller sheet; appearance: tall S-step sheet with a smaller sheet of matching profile",
        "surface shapes", "paper: " + OPEN + " and " + SPL + " returned nothing or an error, " + COMM + " two points only",
        curves=[(gc.line((0, -0.6, 0), (0, 1, 0)), 0, 1.2, False, "tangent")], cov_tol=1e-2)
    # ex2 (3,3) a tangential curve (unchanged, FAITHFUL)
    sh = [0.0, -0.6, 0.0, 0.5]
    q = poly(mono(0, 3, 0.4), mono(2, 0, 0.3), mono(0, 1, 0.2))
    D = mono(0, 2, 0.6)
    s1, s2 = pair(q, D, sh, (-1.2, 1.2), (-0.6, 0.6), (3, 3), "T2ex2")
    add("y23_t2_ex2_tangential_curve", "Table 2 ex.2 (3,3): a tangential curve", s1, s2,
        SRC + "Table 2 (example 2)", "'a tangential curve', bi-degree (3,3)",
        "bi-degree (3,3); topology: one open tangential curve (cubic y = 0.5x^3 - 0.6x)", "shape",
        "paper: " + OPEN + " and " + SPL + " returned nothing or an error, " + COMM + " two points only",
        curves=[(uv_curve(q, sh, lambda t: t, lambda t: 0 * t), -1.2, 1.2, False, "tangent")], cov_tol=1e-2)
    # ex3 (3,3) a small loop and two long curves: S-channel vs channel minus circle x hyperbola
    sh = [0.0, -0.45, 0.0, 0.25]
    k = 1.2
    rho, m, sig = 0.45, 0.03, 2.0
    Cc = poly(mono(2, 0), mono(0, 2), mono(0, 0, -rho * rho))
    Lh = poly(mono(1, 1), mono(0, 0, m))  # u w + m = 0: branches in quadrants II and IV
    chan = mono(0, 2, k)
    blue = graph_patch(chan, (-1.6, 1.6), (-0.7, 0.7), "S-curved channel z = 1.2 w^2", degree=(3, 3), shear=sh)
    orange = graph_patch(g.p2_add(chan, -sig * mul(Cc, Lh)), (-1.1, 1.1), (-0.6, 0.6), "shorter S-strip",
                         degree=(3, 3), shear=sh)
    on_blue = lambda fu, fw: (lambda t: g.sheared_point(chan, sh, fu(t), fw(t)))
    sm = math.sqrt(m)
    hyp = lambda sg: on_blue(lambda t: sg * sm * np.exp(t), lambda t: -sg * sm * np.exp(-t))
    disc = math.sqrt(rho ** 4 - 4 * m * m)
    sing = []
    for u2 in ((rho * rho + disc) / 2, (rho * rho - disc) / 2):
        for sg in (1, -1):
            u = sg * math.sqrt(u2)
            sing.append(g.sheared_point(chan, sh, [u], [-m / u])[0])
    add("y23_t2_ex3_loop_two_curves", "Table 2 ex.3 (3,3): a small loop and two long curves", blue, orange,
        SRC + "Table 2 (example 3)", "'a loop and two curves', bi-degree (3,3)",
        "bi-degree (3,3); topology: one small closed loop crossed twice by each of two long open curves that run "
        "boundary to boundary (4 crossing points); appearance: long S-curved channel and a shorter S-strip",
        "the two long curves are the branches of a hyperbola u w = -0.03 (they run along the strip in opposite "
        "halves rather than side by side); that they cross the loop is read from the figure",
        "paper: " + OPEN + " and " + COMM + " returned the two long curves and lost the small loop; " + SPL +
        " returned nothing",
        curves=[(on_blue(lambda t: rho * np.cos(t), lambda t: rho * np.sin(t)), 0, TAU, True, "crossing"),
                (hyp(1), -5, 5, False, "crossing"), (hyp(-1), -5, 5, False, "crossing")],
        singular=sing)
    # ex4 (3,3) a tangential curve: sheared graphs over (u, w + p(u)), p cubic; A = q + w^2 (a + b u^2) on a
    # different (u, w) box, so the silhouettes cross in an X; contact exactly on w = 0 (an inflected curve)
    sh = [0.0, -0.45, 0.0, 0.5]
    q = poly(mono(2, 0, 0.4), mono(0, 2, -0.5), mono(3, 0, 0.15))
    D = mul(mono(0, 2), poly(mono(0, 0, 0.6), mono(2, 0, 0.8)))
    blue = graph_patch(q, (-0.8, 0.8), (-0.9, 0.9), "saddle sheet B", degree=(3, 3), shear=sh)
    orange = graph_patch(g.p2_add(q, D), (-1.1, 1.1), (-0.45, 0.45), "saddle sheet A = B + w^2(0.6 + 0.8u^2)",
                         degree=(3, 3), shear=sh)
    add("y23_t2_ex4_tangential_curve_saddle", "Table 2 ex.4 (3,3): a tangential curve (inflected wave)",
        blue, orange, SRC + "Table 2 (example 4)", "'a tangential curve', bi-degree (3,3)",
        "bi-degree (3,3); topology: one open tangential curve, an inflected wave y = 0.5x^3 - 0.45x; appearance: "
        "two saddle-like sheets on different (u,w) boxes whose silhouettes cross in an X",
        "the sheets share the shear and q, so they are tangent exactly along w = 0; shapes",
        "paper: " + OPEN + ", " + SPL + " and " + COMM + " all returned nothing or an error",
        curves=[(uv_curve(q, sh, lambda t: t, lambda t: 0 * t), -0.8, 0.8, False, "tangent")], cov_tol=1e-2)
    # ex5 (3,3) two loops: two bent tube arms (torus patches, equal tube radius) crossing in an X
    R, r, al, be = 2.5, 0.35, math.radians(55), 30.0
    ca = np.array([-R * math.cos(al), 0, R * math.sin(al)])
    cb = np.array([R * math.cos(al), 0, R * math.sin(al)])
    fa = g.Frame(ca, (1, 0, 0), (0, 0, 1))
    fb = g.Frame(cb, (1, 0, 0), (0, 0, 1))
    ang_a = -math.degrees(al)
    ang_b = math.degrees(al) - 180.0
    A = g.elevate_any(g.nurbs_torus(fa, R, r, (ang_a - be, ang_a + be), (0, 360), "tube arm A"), 1, 1)
    B = g.elevate_any(g.nurbs_torus(fb, R, r, (ang_b - be, ang_b + be), (0, 360), "tube arm B"), 1, 1)
    E, gE = torus_implicit(fb, R, r)
    branches, crossing, isolated = it.trace(it.TorusImplicit(fa, R, r, E, gE), N=1600)
    keep = []
    for Pt, closed in branches:
        ins = A.inside(Pt, 1e-9) & B.inside(Pt, 1e-9)
        if ins.all():
            keep.append((Pt, closed))
        elif ins.any():
            raise RuntimeError("t2_ex5: a component crosses a patch boundary")
    in_both = lambda p: bool(A.inside(np.array([p]), 1e-9)[0] and B.inside(np.array([p]), 1e-9)[0])
    crossing = [p for p in crossing if in_both(p)]
    isolated = [p for p in isolated if in_both(p)]
    add("y23_t2_ex5_two_loops", "Table 2 ex.5 (3,3): two loops (two tube arms crossing in an X)", A, B,
        SRC + "Table 2 (example 5)", "'two loops', bi-degree (3,3)",
        "bi-degree (3,3) (rational torus patches, degree-elevated exactly); topology: two closed loops crossing at "
        "two points; appearance: two bent tube arms forming an X",
        "tube arms are torus patches (R=2.5, r=0.35) whose core circles cross at the origin, so the tubes are tangent "
        "at (0, +-0.35, 0); the loops' 'corners' in the paper's drawing are not reproduced; ground truth "
        "numerical-algebraic (implicit_trace.py)",
        "paper: " + OPEN + " and " + COMM + " returned one loop only, " + SPL + " returned nothing",
        curves=polyline_curves(keep), points=[tuple(p) for p in isolated], singular=[tuple(p) for p in crossing],
        cov_tol=5e-3)
    gc.CASES[-1]["truth_method"] = "numerical-algebraic (implicit_trace.py), vertices on F=0 to ~1e-15 in parameter"
    # ex6 (2,2) a loop and a point: bowl (blue) and a pinched sheet; D = a(u) w^2 + b(u) w + c(u) with
    # discriminant -u^2 (u - p)(u - q): loop for u in [p, q], isolated point at u = 0
    pq, qq = 0.3, 2.3
    b1sq = (-(1 - pq * qq) + math.sqrt((1 - pq * qq) ** 2 + 4 * ((pq + qq) / 2) ** 2)) / 2
    b1 = math.sqrt(b1sq)
    b0 = (pq + qq) / 2 / b1
    c0 = (b1 * b1 + 1) / 4
    lam = 0.5  # w -> lam w: stretches the loop in w into an upright leaf (width:length ~ 2:3)
    wscale = lambda H: g.p2(H) * lam ** np.arange(g.p2(H).shape[1])[None, :]
    D = wscale(poly(mono(0, 2), mono(2, 2), mono(2, 1, b1), mono(1, 1, b0), mono(2, 0, c0)))
    bowl = wscale(poly(mono(2, 0, 0.35), mono(1, 0, -0.8), mono(0, 2, 0.35), mono(0, 1, 0.3)))
    blue = graph_patch(bowl, (-0.7, 2.9), (-1.5 / lam, 0.7 / lam), "bowl", degree=(2, 2))
    orange = graph_patch(g.p2_add(bowl, 0.5 * D), (-0.25, 2.5), (-1.05 / lam, 0.25 / lam), "pinched sheet",
                         degree=(2, 2))

    def loopw(t):
        u = (pq + qq) / 2 + (qq - pq) / 2 * np.cos(t)
        a_ = 1 + u * u
        bb = u * (b1 * u + b0)
        return (-bb + u * (qq - pq) / 2 * np.sin(t)) / (2 * a_) / lam
    add("y23_t2_ex6_loop_and_point", "Table 2 ex.6 (2,2): a loop and a point", blue, orange, SRC + "Table 2 (example 6)",
        "'a loop and a point', bi-degree (2,2)",
        "bi-degree (2,2); topology: one closed loop + one isolated tangential point just beyond one end of the loop, "
        "near the lower tip of the second surface; appearance: bowl and a pinched sheet in it",
        "the loop is a leaf with smooth tips (D and the bowl use w -> 0.5 w); shapes",
        "paper: " + OPEN + ", " + SPL + " and " + COMM + " returned the loop only (lost the point)",
        curves=[(lambda t: g.sheared_point(bowl, None, (pq + qq) / 2 + (qq - pq) / 2 * np.cos(t), loopw(t)), 0, TAU,
                 True, "crossing")],
        points=[g.sheared_point(bowl, None, [0.0], [0.0])[0]], cov_tol=1e-2)


def table6():
    # (1) (3,1) a tangent line: monotone S-step, contact at the inflection, smaller second sheet
    step = poly(mono(1, 0, 1.2), mono(3, 0, -0.4))
    blue = graph_patch(step, (-1, 1), (-1, 1), "S-step sheet z = 1.2u - 0.4u^3", degree=(3, 1))
    orange = graph_patch(g.p2_add(step, mono(2, 0, 0.8)), (-0.6, 0.6), (-0.7, 0.7), "smaller sheet", degree=(3, 1))
    add("y23_t6_ex1_tangent_line", "Table 6 ex.1 (3,1): a tangent line", blue, orange, SRC + "Table 6 (example (1))",
        "'a tangent line', bi-degree (3,1)",
        "bi-degree (3,1); topology: one open tangential straight line across the steep middle of a monotone S-step "
        "(flat, steep, flat); the second sheet is smaller", "shapes",
        "paper: " + OPEN + " and " + COMM + " returned nothing, " + SPL + " scattered points",
        curves=[(gc.line((0, -0.7, 0), (0, 1, 0)), 0, 1.4, False, "tangent")], cov_tol=1e-2)
    # (2)-(5) unchanged (FAITHFUL)
    F0 = g.Frame()
    sp = g.nurbs_sphere(F0, 1.0, (-60, 60), (-40, 40), "sphere patch")
    cy = g.nurbs_cylinder(g.Frame((0, 0, -0.8), (1, 0, 0), (0, 1, 0)), 1.0, (-70, 70), (0, 1.6), "cylinder patch")
    cy = elevate_v(cy, 1)
    add("y23_t6_ex2_tangential_arc", "Table 6 ex.2 (2,2): a tangential curve (sphere patch in cylinder patch)",
        sp, cy, SRC + "Table 6 (example (2))", "'a tangential curve', bi-degree (2,2)",
        "bi-degree (2,2) (rational); topology: one open tangential circular arc",
        "the surfaces (the figure shows a dome-like patch against a cylinder-like patch)",
        "paper: " + OPEN + " and " + SPL + " returned nothing",
        curves=[(gc.circle((0, 0, 0), (0, 0, 1), 1.0), -math.pi, math.pi, True, "tangent")], cov_tol=1e-2)
    ca = g.nurbs_cylinder(g.Frame((0, 0, -0.8), (1, 0, 0), (0, 1, 0)), 1.0, (45, 135), (0, 1.6), "cylinder A")
    cb = g.nurbs_cylinder(g.Frame((-0.8, 0, 0), (0, 1, 0), (0, 0, 1)), 1.0, (-45, 45), (0, 1.6), "cylinder B")
    ca, cb = elevate_v(ca, 1), elevate_v(cb, 1)
    xc = lambda s: (lambda t: np.column_stack([np.sin(t), np.cos(t), s * np.sin(t)]))
    add("y23_t6_ex3_two_crossing_curves", "Table 6 ex.3 (2,2): two intersecting curves", ca, cb,
        SRC + "Table 6 (example (3))", "'two intersect curves', bi-degree (2,2)",
        "bi-degree (2,2) (rational); topology: two open curves crossing at one point",
        "the surfaces (figure: a cylinder-like patch and a barrel-like patch)", "paper: all methods found the X",
        curves=[(xc(1), -math.pi / 2, math.pi / 2, False, "crossing"), (xc(-1), -math.pi / 2, math.pi / 2, False,
                                                                        "crossing")],
        singular=[(0, 1, 0)])
    p1 = g.heightfield_nurbs([[0, 0, 1], [0, 0, 0], [1, 0, 0]], (-1, 1), (-1, 1), 2, 2, "z=x^2+y^2")
    p2 = g.heightfield_nurbs([[0, 0, 0.4], [0, 0, 0], [0.4, 0, 0]], (-0.6, 0.6), (-0.6, 0.6), 2, 2, "z=0.4(x^2+y^2)")
    add("y23_t6_ex4_tangential_point", "Table 6 ex.4 (2,2): a tangential point", p2, p1, SRC + "Table 6 (example (4))",
        "'a tangential point', bi-degree (2,2)", "bi-degree (2,2); topology: one isolated tangential point",
        "the surfaces (figure: a smaller bulged patch against a larger one)",
        "paper: " + OPEN + " returned nothing, " + SPL + " scattered points", points=[(0, 0, 0)], cov_tol=1e-2)
    q = poly(mono(2, 0, -1.0), mono(1, 0, 1.0), mono(0, 2, -1.0), mono(0, 1, 1.0))
    D = 0.3 * mul(mono(1, 0), poly(mono(0, 0, 1.0), mono(1, 0, -1.0)), mono(0, 1), poly(mono(0, 0, 1.0), mono(0, 1, -1.0)))
    s1, s2 = pair(q, D, None, (0, 1), (0, 1), (2, 2), "T6ex5")
    Lf = lambda u0, u1, w0, w1: uv_curve(q, None, lambda t: u0 + (u1 - u0) * t, lambda t: w0 + (w1 - w0) * t)
    add("y23_t6_ex5_four_boundary_curves", "Table 6 ex.5 (2,2): four boundary curves (dome-like)", s1, s2,
        SRC + "Table 6 (example (5))", "'four boundary curves', bi-degree (2,2)",
        "bi-degree (2,2); topology: the four patch boundary curves (both patches share them), closed chain",
        "surface shapes; that the two patches share all four boundaries",
        "paper: " + SPL + " returned nothing; " + OPEN + " partial",
        curves=[(Lf(0, 1, 0, 0), 0, 1, False, "boundary"), (Lf(1, 1, 0, 1), 0, 1, False, "boundary"),
                (Lf(1, 0, 1, 1), 0, 1, False, "boundary"), (Lf(0, 0, 1, 0), 0, 1, False, "boundary")],
        singular=[g.sheared_point(q, None, [x], [y])[0] for x, y in ((0, 0), (1, 0), (0, 1), (1, 1))])
    # (6) (2,2) four boundary curves: trough with two long sagging edges and two short straight edges,
    # dome-like second sheet sitting in it (shares all four boundaries)
    W = 0.25
    trough = poly(mono(2, 0, 1.2), mono(1, 0, -1.2), mono(0, 0, 0.3), mono(0, 1, 0.3))  # 1.2(u-1/2)^2 + 0.3w
    Dd = (0.5 * 16 / (W * W)) * mul(mono(1, 0), poly(mono(0, 0, 1.0), mono(1, 0, -1.0)), mono(0, 1),
                                     poly(mono(0, 0, W), mono(0, 1, -1.0))) * 0.25
    s1 = graph_patch(g.p2_add(trough, Dd), (0, 1), (0, W), "dome sitting in the trough", degree=(2, 2))
    s2 = graph_patch(trough, (0, 1), (0, W), "trough", degree=(2, 2))
    Lf = lambda u0, u1, w0, w1: uv_curve(trough, None, lambda t: u0 + (u1 - u0) * t, lambda t: w0 + (w1 - w0) * t)
    add("y23_t6_ex6_four_boundary_curves", "Table 6 ex.6 (2,2): four boundary curves (trough, short straight sides)",
        s1, s2, SRC + "Table 6 (example (6))", "'four boundary curves', bi-degree (2,2)",
        "bi-degree (2,2); topology: the four patch boundary curves, closed chain; appearance: two long sagging "
        "boundary arcs and two short straight edges at the raised ends, a dome-like sheet sitting in the trough",
        "that the two patches share all four boundaries; shapes",
        "paper: " + SPL + " returned an error",
        curves=[(Lf(0, 1, 0, 0), 0, 1, False, "boundary"), (Lf(1, 1, 0, W), 0, 1, False, "boundary"),
                (Lf(1, 0, W, W), 0, 1, False, "boundary"), (Lf(0, 0, W, 0), 0, 1, False, "boundary")],
        singular=[g.sheared_point(trough, None, [x], [y])[0] for x, y in ((0, 0), (1, 0), (0, W), (1, W))])


def appendix():
    """Example 2 of the paper's appendix: published exact surfaces.
    P(u,v) (Eq. 11, homogeneous weight 2) on [0,1]^2, Q(s,t) on [-1,1]x[0,2]."""
    Px = P([-11, 48, -48], [6, -96, 96], [48], [-32])     # c[i][j] u^i v^j
    Py = P([-11, 6, 48, -32], [48, -96], [-48, 96])
    Pz = P([0, 24, -24], [-24], [24])
    Pw = P([2.0])
    Qx = P([0], [2, 0, 2])                                # 2 s t^2 + 2 s
    Qy = P([3, 0, 0, -3], [0], [1, 0, 0, -1])             # -s^2 t^3 + s^2 - 3 t^3 + 3
    Qz = P([0, 6], [0], [0], [0, 2])                      # 2 s^3 t + 6 t
    Qw = P([1, 0, 1], [0], [1, 0, 1])                     # s^2 t^2 + s^2 + t^2 + 1
    Ps = g.rational_poly_patch(Px, Py, Pz, Pw, (0, 1), (0, 1), "P(u,v), Eq. (11), bicubic")
    Qs = g.rational_poly_patch(Qx, Qy, Qz, Qw, (-1, 1), (0, 2), "Q(s,t), appendix Example 2, rational bicubic")
    hom = lambda H, a, b: np.array([g.p2_eval(c, a, b) for c in H])

    def Pc(u, v):
        h = hom(Ps.hom, u, v)
        return h[:3] / h[3]

    def Qc(s, t):
        h = hom(Qs.hom, s, t)
        return h[:3] / h[3]
    # complex-safe evaluation for the continuation tracer
    def p2c(c, a, b):
        c = g.p2(c)
        out = 0 * a
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                if c[i, j]:
                    out = out + c[i, j] * a ** i * b ** j
        return out

    def G(X):
        hp = [p2c(c, X[0], X[1]) for c in Ps.hom]
        hq = [p2c(c, X[2], X[3]) for c in Qs.hom]
        return np.array([hp[k] / hp[3] - hq[k] / hq[3] for k in range(3)])
    # seeds: grid on P, s from x (x = 2s/(1+s^2) on Q), both t roots of z(1+t^2) = 2Bt
    seeds = []
    for u in np.linspace(0, 1, 241):
        for v in np.linspace(0, 1, 241):
            x, y, z = Pc(u, v)
            if abs(x) > 1 or z <= 0:
                continue
            s = 0.0 if x == 0 else (1 - math.sqrt(1 - x * x)) / x
            A_ = (s * s + 3) / (s * s + 1)
            B_ = (s ** 3 + 3) / (s * s + 1)
            for t in np.roots([z, -2 * B_, z]):
                if abs(t.imag) < 1e-12 and 0 <= t.real <= 2:
                    t = t.real
                    if abs(y * (1 + t * t) - A_ * (1 - t ** 3)) < 0.3:
                        seeds.append([u, v, s, t])
    comps = pt.trace_from_seeds(G, np.array(seeds), [(0, 1), (0, 1), (-1, 1), (0, 2)], h=1e-3, cover_tol=2e-2)
    branches = []
    for X, closed in comps:
        Pt = np.array([Pc(a, b) for a, b in X[:, :2]])
        branches.append((Pt, closed))
    add("y23_appendix_ex2", "Appendix Example 2: P (Eq. 11) x Q (published exact data)", Ps, Qs,
        "Yang, Jia & Yan 2023, appendix Example 2 (P from Example 1, Eq. 11): exact published surfaces",
        "the paper gives the surfaces and the topology graph of phi(s,t) = 0 in Q's domain (Fig. 14) with helping "
        "points; it states no topology in words",
        "the exact P(u,v) (homogeneous weight 2) and Q(s,t) as published, on [0,1]^2 and [-1,1]x[0,2] (Q stored remapped to [0,1]^2)",
        "nothing about the geometry; ground truth is our own numerical continuation in (u,v,s,t) (param_trace.py), "
        "points on both surfaces to ~1e-15; P overlaps itself, so pieces may meet in 3D on different sheets of P. "
        "The NURBS form of Q is the paper's Q(s,t) remapped onto [0,1]^2 (s' = (s+1)/2, t' = t/2), so SMLib's "
        "parameter values (and any comparison with Fig. 14's helping points) differ from the paper's (s,t)",
        "paper: used as a worked example; no comparison with other methods",
        curves=polyline_curves(branches), cov_tol=5e-3, cls="Y23 published parameters")
    gc.CASES[-1]["truth_method"] = "numerical continuation (param_trace.py), Newton-corrected to |P-Q| < 1e-12"


def elevate_v(s, t):
    """Exact degree elevation in v of a single-span-in-v NURBS (keeps the u knots)."""
    assert len(s.V) == 2 * (s.q + 1)
    s.net = np.stack([g.bezier_elevate(s.net[i], t) for i in range(s.net.shape[0])], axis=0)
    s.q += t
    s.V = np.array([s.V[0]] * (s.q + 1) + [s.V[-1]] * (s.q + 1))
    return s


TABLE4 = [
    # torus centre, normal, R, r ; ellipsoid centre, x-axis, y-axis, a, b, c
    ((0, 0, 0), (0, 0, 1), 4, 1, (0, 0, 0), (1, 0, 0), (0, 1, 0), 1, 5, 2),
    ((1, 0, 0), (0, 0, 1), 3, 1, (0, 0, 0), (0, 1, 0), (0, 0, 1), 3, 4, 2),
    ((0, 0, 0), (0, 0, 1), 3, 1, (0, 0, 0), (0, 1, 0), (0, 0, 1), 4, 3, 2),
    ((0, 0, 0), (0, 0, 1), 2, 1, (1, 0, 0), (1, 0, 0), (0, 1, 0), 2, 2, 1),
    ((0, 0, 0), (0, 0, 1), 3, 1, (0, 0, 0), (0, 1, 0), (0, 0, 1), 3, 4, 2),
]
TABLE3_PICTURES = [
    "two small figure-eight-like curves, each with a separate dot next to it",
    "one closed curve with two lobes (drawn as two loops meeting)",
    "closed curves with four marked crossing (singular) points",
    "two loops meeting at one marked point, plus a separate dot",
    "two loops crossing at two marked points",
]


def table3():
    for k, (tc, tn, R, r, ec, ex, ey, a, b, c) in enumerate(TABLE4, start=1):
        tn = np.asarray(tn, float)
        tx = np.array([1.0, 0, 0]) if abs(tn[0]) < 0.9 else np.array([0, 1.0, 0])
        tf = g.Frame(tc, tx, np.cross(tn, tx))
        tor = g.Torus(tf, R, r)
        ef = g.Frame(ec, ex, ey)
        ell = g.nurbs_ellipsoid(ef, a, b, c, "ellipsoid a=%g b=%g c=%g" % (a, b, c))
        A = np.array([ef.x, ef.y, ef.z])
        abc = np.array([a, b, c], float)

        def E(Pt, A=A, abc=abc, o=ef.o):
            L = (Pt - o) @ A.T / abc
            return np.sum(L * L, axis=-1) - 1.0

        def gradE(Pt, A=A, abc=abc, o=ef.o):
            L = (Pt - o) @ A.T / abc
            return (2 * L / abc) @ A
        ti = it.TorusImplicit(tf, R, r, E, gradE)
        branches, crossing, isolated = it.trace(ti, N=1600)
        curves = []
        for Pt, closed in branches:
            pts = Pt
            curves.append((lambda t, pts=pts: pts[np.clip(np.round(t).astype(int), 0, len(pts) - 1)], 0,
                           len(pts) - 1, closed, "crossing"))
        add("y23_t3_ex%d_torus_ellipsoid" % k, "Table 3/4 ex.%d: torus x ellipsoid (published parameters)" % k,
            tor, ell, "Yang, Jia & Yan 2023, Table 3/4 (example %d): shape parameters as published" % k,
            "intersection drawn in Table 3: %s" % TABLE3_PICTURES[k - 1],
            "exact torus and ellipsoid parameters from Table 4 (ellipsoid as an exact rational NURBS, torus analytic)",
            "nothing about the geometry; the ground-truth topology is our own computation (marching squares on "
            "F(theta,phi)=E(torus(theta,phi)) with bisection-refined vertices and Newton-located tangency points), "
            "compare it with the paper's picture in paper_topology",
            "paper: compared with " + COMM + "; both gave the same geometry",
            curves=curves, points=[tuple(p) for p in isolated], singular=[tuple(p) for p in crossing],
            cov_tol=5e-3, cls="Y23 published parameters")
        last = gc.CASES[-1]
        last["truth_method"] = "numerical-algebraic (implicit_trace.py), vertices on F=0 to ~1e-15 in parameter"


def main():
    for f in (fig1, table2, table6, table3, appendix):
        f()
    ys = [c for c in gc.CASES if c["id"].startswith("y23_")]
    os.makedirs(gc.OUT, exist_ok=True)
    for old in os.listdir(gc.OUT):
        if old.startswith("y23_") and old.endswith(".json"):
            os.remove(os.path.join(gc.OUT, old))
    for c in ys:
        with open(os.path.join(gc.OUT, c["id"] + ".json"), "w") as fh:
            json.dump(c, fh)
    idx = [{"id": c["id"], "title": c["title"], "source": c["source"], "paper_topology": c["paper_topology"],
            "surfaces": c["surface_labels"], "branches": len(c["expected"]["branches"]),
            "closed": [b["closed"] for b in c["expected"]["branches"]],
            "isolated_points": len(c["expected"]["isolated_points"]),
            "singular_points": len(c["expected"]["singular_points"])} for c in ys]
    with open(os.path.join(HERE, "index_yang2023.json"), "w") as fh:
        json.dump(idx, fh, indent=1)
    for i in idx:
        print("%-38s br=%d closed=%s pts=%d sing=%d  %s" % (i["id"], i["branches"], i["closed"], i["isolated_points"],
                                                              i["singular_points"], i["surfaces"]))


if __name__ == "__main__":
    main()
