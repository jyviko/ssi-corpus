"""Generate the SSI evidence cases with closed-form ground truth.

Every case is two surfaces (analytic primitives or exact NURBS constructions)
plus the exact expected intersection, obtained from our own closed-form maths:
the full intersection curve(s) are written parametrically, sampled, and
clipped to both trimmed patches with closed-form membership tests (clip
points refined by bisection). Output: one JSON file per case in cases/data/.

Run: python3 -I cases/gen_cases.py
"""
import json
import math
import os
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scoring"))
import ssi_geom as g  # noqa: E402

OUT = os.path.join(HERE, "data")
NS = 4001  # stored samples per expected branch

CASES = []


# ----------------------------------------------------------------------------
# curve helpers: each returns fn(t: ndarray) -> (N,3)


def line(p0, d):
    p0, d = np.asarray(p0, float), np.asarray(d, float)
    return lambda t: p0 + np.outer(t, d)


def circle(center, n, r, xdir=None):
    c = np.asarray(center, float)
    n = g.unit(n)
    if xdir is None:
        a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        xdir = a - np.dot(a, n) * n
    e1 = g.unit(xdir)
    e2 = np.cross(n, e1)
    return lambda t: c + r * (np.outer(np.cos(t), e1) + np.outer(np.sin(t), e2))


def local_curve(frame, fn):
    return lambda t: frame.to_world(fn(t))


# ----------------------------------------------------------------------------
# clipping


def clip(fn, t0, t1, closed, surfs, n=40001):
    """Sample fn on [t0,t1], keep parts inside both patches. Returns list of
    (ta, tb, closed) parameter runs with bisection-refined ends."""
    t = np.linspace(t0, t1, n)
    P = fn(t)

    def ins(P_):
        m = np.ones(len(P_), dtype=bool)
        for s in surfs:
            m &= s.inside(P_)
        return m

    m = ins(P)
    if m.all():
        return [(t0, t1, closed)]
    runs = []
    i = 0
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and m[j + 1]:
            j += 1
        runs.append([i, j])
        i = j + 1

    def refine(t_in, t_out):
        for _ in range(80):
            tm = 0.5 * (t_in + t_out)
            if ins(fn(np.array([tm])))[0]:
                t_in = tm
            else:
                t_out = tm
        return t_in

    out = []
    for i, j in runs:
        ta = t[i] if i == 0 else refine(t[i], t[i - 1])
        tb = t[j] if j == n - 1 else refine(t[j], t[j + 1])
        out.append([ta, tb])
    # merge wrap-around run of a closed curve
    if closed and len(out) > 1 and runs[0][0] == 0 and runs[-1][1] == n - 1:
        first = out.pop(0)
        last = out.pop()
        out.append([last[0], first[1] + (t1 - t0)])
    return [(a, b, False) for a, b in out]


def sample(fn, ta, tb, t0, t1, closed):
    span = t1 - t0
    t = np.linspace(ta, tb, NS)
    if closed:
        t = np.linspace(t0, t1, NS)
    else:
        t = np.where(t > t1, t - span, t)  # wrap-around runs
    return fn(t)


# ----------------------------------------------------------------------------
# case assembly


def add_case(cid, cls, title, s1, s2, curves=(), points=(), singular=(), regions=(), cov_tol=1e-3,
             feature=None, notes="", formula="", ill_cond=None, sub_tolerance=None):
    surfs = [s1, s2]
    branches = []
    for c in curves:
        fn, t0, t1, closed, kind = c
        for ta, tb, cl in clip(fn, t0, t1, closed, surfs):
            P = sample(fn, ta, tb, t0, t1, cl)
            if np.linalg.norm(P[-1] - P[0]) < 1e-12 and not cl and len(P) > 2 and abs(tb - ta) > 1e-9:
                cl = True
            if np.ptp(P, axis=0).max() < 1e-14:
                points = list(points) + [P[0].tolist()]
                continue
            branches.append({"closed": bool(cl), "kind": kind, "points": P.tolist()})
    pts = [list(map(float, p)) for p in points if all(s.inside(np.array([p]))[0] for s in surfs)]
    regs = []
    for r in regions:
        fn, t0, t1 = r
        P = fn(np.linspace(t0, t1, NS))
        assert all(s.inside(P, 1e-7).all() for s in surfs), cid
        regs.append({"boundary": P.tolist()})
    # sanity: every expected point lies on both true surfaces
    allp = [np.array(b["points"]) for b in branches] + [np.array(pts).reshape(-1, 3)] + \
           [np.array(r["boundary"]) for r in regs]
    allp = np.vstack([a for a in allp if len(a)]) if any(len(a) for a in allp) else np.zeros((0, 3))
    if len(allp):
        d = max(g.truth_distance(s.truth(), allp).max() for s in surfs)
        assert d < 1e-9, (cid, d)
    case = {
        "id": cid, "class": cls, "title": title,
        "surfaces": [s1.spec(), s2.spec()],
        "truth": [s1.truth(), s2.truth()],
        "surface_labels": [label(s1), label(s2)],
        "expected": {"branches": branches, "isolated_points": pts,
                     "singular_points": [list(map(float, p)) for p in singular],
                     "coincident_regions": regs},
        "cov_tol": cov_tol, "notes": notes, "formula": formula,
    }
    if feature is not None:
        case["feature_size"] = feature
    if ill_cond is not None:
        case["ill_cond"] = ill_cond
    if sub_tolerance is not None:
        case["sub_tolerance"] = sub_tolerance
    nurbs_check = [s.self_check() for s in surfs if isinstance(s, g.Nurbs)]
    for d, ok in nurbs_check:
        assert d < 1e-12 and ok, (cid, d, ok)
    CASES.append(case)


def label(s):
    if isinstance(s, g.Nurbs):
        nu, nv = s.net.shape[:2]
        rat = not np.allclose(s.net[:, :, 3], 1.0)
        return f"NURBS {s.label} deg({s.p},{s.q}) {nu}x{nv}cp{' rational' if rat else ''}"
    return type(s).__name__.lower()


F0 = g.Frame()
TAU = 2 * math.pi


def zframe(o=(0, 0, 0)):
    return g.Frame(o, (1, 0, 0), (0, 1, 0))


def plane_z(c, half=3.0):
    return g.Plane(zframe((0, 0, c)), (-half, half), (-half, half))


# ----------------------------------------------------------------------------
# 1. transversal

def class1():
    C = "1 transversal"
    # plane tilted z = 0.5 y through a full cylinder -> ellipse
    pf = g.Frame((0, 0, 0), (1, 0, 0), (0, 1, 0.5))
    pl = g.Plane(pf, (-3, 3), (-3, 3))
    cyl = g.Cylinder(zframe((0, 0, 0)), 1.0, (0, 360), (-2, 2))
    ell = lambda t: np.column_stack([np.cos(t), np.sin(t), 0.5 * np.sin(t)])
    add_case("c1_plane_cyl_ellipse", C, "tilted plane x full cylinder: one closed ellipse", pl, cyl,
             [(ell, 0, TAU, True, "crossing")], formula="(cos t, sin t, 0.5 sin t)")
    cyl_half = g.Cylinder(zframe((0, 0, 0)), 1.0, (0, 180), (-2, 2))
    add_case("c1_plane_halfcyl_arc", C, "tilted plane x half cylinder: one open half-ellipse", pl, cyl_half,
             [(ell, 0, TAU, True, "crossing")], formula="(cos t, sin t, 0.5 sin t), t in [0,pi]")
    sph = g.Sphere(F0, 2.0)
    add_case("c1_plane_sphere_circle", C, "plane z=1 x sphere R=2: circle r=sqrt3", plane_z(1.0), sph,
             [(circle((0, 0, 1), (0, 0, 1), math.sqrt(3)), 0, TAU, True, "crossing")])
    # two cylinders, perpendicular axes, radii 1 (axis x) and 1.5 (axis z): two separate loops
    cx = g.Cylinder(g.Frame((-3, 0, 0), (0, 1, 0), (0, 0, 1)), 1.0, (0, 360), (0, 6))
    cz = g.Cylinder(zframe((0, 0, -2)), 1.5, (0, 360), (0, 4))
    loop = lambda s: (lambda t: np.column_stack([s * np.sqrt(2.25 - np.cos(t) ** 2), np.cos(t), np.sin(t)]))
    add_case("c1_cyl_cyl_two_loops", C, "perpendicular cylinders r=1, r=1.5: two separate closed loops", cx, cz,
             [(loop(1), 0, TAU, True, "crossing"), (loop(-1), 0, TAU, True, "crossing")],
             formula="(+-sqrt(2.25-cos^2 t), cos t, sin t)")
    # sphere R=2 with off-axis cylinder (0.5,0) r=0.8: two loops (top/bottom)
    cyl2 = g.Cylinder(zframe((0.5, 0, -3)), 0.8, (0, 360), (0, 6))

    def sc(s):
        def f(t):
            x = 0.5 + 0.8 * np.cos(t)
            y = 0.8 * np.sin(t)
            return np.column_stack([x, y, s * np.sqrt(4 - x * x - y * y)])
        return f
    add_case("c1_sphere_cyl_two_loops", C, "sphere R=2 x off-axis cylinder r=0.8: two separate loops", sph, cyl2,
             [(sc(1), 0, TAU, True, "crossing"), (sc(-1), 0, TAU, True, "crossing")])
    tor = g.Torus(F0, 3.0, 1.0)
    rr = math.sqrt(1 - 0.25)
    add_case("c1_torus_plane_nested", C, "torus R=3 r=1 x plane z=0.5: two nested circles", tor, plane_z(0.5, 5),
             [(circle((0, 0, 0.5), (0, 0, 1), 3 + rr), 0, TAU, True, "crossing"),
              (circle((0, 0, 0.5), (0, 0, 1), 3 - rr), 0, TAU, True, "crossing")])
    cone = g.Cone(F0, 30.0, (0.5, 4.0))
    tn = math.tan(math.radians(30))
    pc = g.Plane(g.Frame((0, 0, 2), (1, 0, 0.3), (0, 1, 0)), (-4, 4), (-4, 4))

    def ce(t):
        h = 2.0 / (1 - 0.3 * tn * np.cos(t))
        return np.column_stack([h * tn * np.cos(t), h * tn * np.sin(t), h])
    add_case("c1_cone_plane_ellipse", C, "cone 30deg x plane z=2+0.3x: closed ellipse", cone, pc,
             [(ce, 0, TAU, True, "crossing")])
    ph = g.Plane(g.Frame((1, 0, 0), (0, 1, 0), (0, 0, 1)), (-3, 3), (-1, 5))
    ym = math.sqrt(16 * tn * tn - 1)
    hyp = lambda s: np.column_stack([np.ones_like(s), s, np.sqrt(1 + s * s) / tn])
    add_case("c1_cone_plane_hyperbola", C, "cone 30deg x plane x=1: one open hyperbola branch", cone, ph,
             [(hyp, -ym - 0.5, ym + 0.5, False, "crossing")])
    # NURBS representatives
    ncyl = g.nurbs_cylinder(F0, 1.0, (0, 360), (-2, 2), "cylinder")
    add_case("c1n_cylN_plane_ellipse", C, "rational NURBS full cylinder x tilted plane: closed ellipse", ncyl, pl,
             [(ell, 0, TAU, True, "crossing")])
    par = g.heightfield_nurbs([[0, 0, 1], [0, 0, 0], [1, 0, 0]], (-2, 2), (-2, 2), 2, 2, "paraboloid z=x^2+y^2")
    add_case("c1n_parab_plane_circle", C, "NURBS paraboloid x plane z=1: circle r=1", par, plane_z(1.0),
             [(circle((0, 0, 1), (0, 0, 1), 1.0), 0, TAU, True, "crossing")])
    sad = g.heightfield_nurbs([[0, 0], [0, 1]], (-2, 2), (-2, 2), 1, 1, "saddle z=xy")
    hb = lambda s: (lambda t: np.column_stack([s * np.exp(t), s * np.exp(-t), np.ones_like(t)]))
    add_case("c1n_saddle_plane_hyperbola", C, "bilinear saddle z=xy x plane z=1: two open hyperbola branches",
             sad, plane_z(1.0), [(hb(1), -1.0, 1.0, False, "crossing"), (hb(-1), -1.0, 1.0, False, "crossing")],
             formula="(+-e^t, +-e^-t, 1)")
    par2 = g.heightfield_nurbs([[2, 0, -1], [0, 0, 0], [-1, 0, 0]], (-2, 2), (-2, 2), 2, 2, "z=2-x^2-y^2")
    add_case("c1n_parab_parab_circle", C, "two NURBS paraboloids: circle r=1 at z=1", par, par2,
             [(circle((0, 0, 1), (0, 0, 1), 1.0), 0, TAU, True, "crossing")])
    nsph = g.nurbs_sphere(F0, 2.0, (0, 360), (-90, 90), "sphere")
    ncyl2 = g.nurbs_cylinder(zframe((0.5, 0, 0)), 0.8, (0, 360), (-3, 3), "cylinder")
    add_case("c1n_sphereN_cylN_two_loops", C, "rational NURBS sphere x NURBS cylinder: two separate loops", nsph, ncyl2,
             [(sc(1), 0, TAU, True, "crossing"), (sc(-1), 0, TAU, True, "crossing")])
    ntor = g.nurbs_torus(F0, 3.0, 1.0, (0, 360), (0, 360), "torus")
    npl = g.nurbs_plane(zframe((0, 0, 0.5)), (-5, 5), (-5, 5), 1, 1, "plane")
    add_case("c1n_torusN_planeN_nested", C, "rational NURBS torus x NURBS plane z=0.5: nested circles", ntor, npl,
             [(circle((0, 0, 0.5), (0, 0, 1), 3 + rr), 0, TAU, True, "crossing"),
              (circle((0, 0, 0.5), (0, 0, 1), 3 - rr), 0, TAU, True, "crossing")])


# ----------------------------------------------------------------------------
# 2. tangent along a curve

def class2():
    C = "2 tangent curve"
    T = 1e-2  # tangential contact: position is determined only to ~sqrt(tol*R)
    pl = plane_z(0.0)
    cyl = g.Cylinder(g.Frame((2, 0, 1), (0, 0, 1), (0, 1, 0)), 1.0, (0, 360), (0, 4))
    add_case("c2_cyl_on_plane", C, "cylinder r=1 resting on plane: tangent line", pl, cyl,
             [(line((-2, 0, 0), (1, 0, 0)), 0, 4, False, "tangent")], cov_tol=T)
    sph = g.Sphere(F0, 1.0)
    cz = g.Cylinder(zframe((0, 0, -2)), 1.0, (0, 360), (0, 4))
    add_case("c2_sphere_in_cyl", C, "sphere R=1 inside cylinder r=1: tangent circle", sph, cz,
             [(circle((0, 0, 0), (0, 0, 1), 1.0), 0, TAU, True, "tangent")], cov_tol=T)
    tor = g.Torus(F0, 3.0, 1.0)
    s2 = g.Sphere(F0, 2.0)
    add_case("c2_sphere_in_torus", C, "sphere R=2 in torus hole (R=3,r=1): tangent circle on inner equator", s2, tor,
             [(circle((0, 0, 0), (0, 0, 1), 2.0), 0, TAU, True, "tangent")], cov_tol=T)
    a = math.radians(20)
    # axes (cos a, 0, +-sin a): both cones contain the ruling along +x
    c1 = g.Cone(g.Frame((0, 0, 0), (-math.sin(a), 0, math.cos(a)), (0, -1, 0)), 20.0, (0.2, 3.0))
    c2 = g.Cone(g.Frame((0, 0, 0), (math.sin(a), 0, math.cos(a)), (0, -1, 0)), 20.0, (0.2, 3.0))
    assert np.allclose(c1.f.z, [math.cos(a), 0, math.sin(a)]) and np.allclose(c2.f.z, [math.cos(a), 0, -math.sin(a)])
    add_case("c2_cones_tangent_ruling", C, "two cones, common apex, touching along a ruling: tangent line", c1, c2,
             [(line((0, 0, 0), (1, 0, 0)), 0.2 / math.cos(a), 3.0 / math.cos(a), False, "tangent")], cov_tol=T,
             notes="substitute for 'coaxial cones': coaxial cones with equal angle are identical (class 6) "
                   "or disjoint; cones sharing apex and a ruling are tangent along it")
    ca = g.Cylinder(zframe((0, 0, -2)), 1.0, (0, 360), (0, 4))
    cb = g.Cylinder(zframe((2, 0, -2)), 1.0, (0, 360), (0, 4))
    add_case("c2_cyl_cyl_external", C, "parallel cylinders r=1, axes 2 apart: tangent line", ca, cb,
             [(line((1, 0, -2), (0, 0, 1)), 0, 4, False, "tangent")], cov_tol=T)
    pc = g.heightfield_nurbs([[0], [0], [1]], (-2, 2), (-2, 2), 2, 1, "parabolic cylinder z=x^2")
    add_case("c2n_parabcyl_plane", C, "NURBS parabolic cylinder z=x^2 on plane z=0: tangent line", pc, pl,
             [(line((0, -2, 0), (0, 1, 0)), 0, 4, False, "tangent")], cov_tol=T)
    q = [[1, 0, -2, 0, 1], [0] * 5, [-2, 0, 2, 0, 0], [0] * 5, [1, 0, 0, 0, 0]]
    qs = g.heightfield_nurbs(q, (-2, 2), (-2, 2), 4, 4, "z=(x^2+y^2-1)^2")
    add_case("c2n_quartic_circle", C, "NURBS quartic z=(x^2+y^2-1)^2 on plane z=0: tangent circle", qs, pl,
             [(circle((0, 0, 0), (0, 0, 1), 1.0), 0, TAU, True, "tangent")], cov_tol=T)
    ncyl = g.nurbs_cylinder(g.Frame((2, 0, 1), (0, 0, 1), (0, 1, 0)), 1.0, (0, 360), (0, 4), "cylinder")
    npl = g.nurbs_plane(zframe(), (-3, 3), (-3, 3), 1, 1, "plane")
    add_case("c2n_cylN_on_planeN", C, "NURBS cylinder on NURBS plane: tangent line", npl, ncyl,
             [(line((-2, 0, 0), (1, 0, 0)), 0, 4, False, "tangent")], cov_tol=T)
    # quartic contact along a parabola: z = 0.5w^3 - 0.5w + 0.2u^2 and the same + w^4 (same footprint)
    qq = np.array([[0, -0.5, 0, 0.5, 0], [0] * 5, [0.2, 0, 0, 0, 0]])
    a4 = g.sheared_heightfield_nurbs(qq + np.array([[0, 0, 0, 0, 1.0], [0] * 5, [0] * 5]), None, (-1, 1), (-1, 1),
                                     "z = 0.5w^3 - 0.5w + 0.2u^2 + w^4", degree=(2, 4))
    b4 = g.sheared_heightfield_nurbs(qq, None, (-1, 1), (-1, 1), "z = 0.5w^3 - 0.5w + 0.2u^2", degree=(2, 4))
    add_case("c2n_quartic_contact_parabola", C, "NURBS sheets in 4th-order contact (D = w^4) along a parabola", a4, b4,
             [(lambda t: np.column_stack([t, 0 * t, 0.2 * t * t]), -1, 1, False, "tangent")], cov_tol=T,
             notes="the surfaces are within 1e-5 of each other over the band |w| < 0.056")
    nsph = g.nurbs_sphere(F0, 1.0, (0, 360), (-90, 90), "sphere")
    add_case("c2n_sphereN_in_cyl", C, "NURBS sphere inside analytic cylinder: tangent circle", nsph, cz,
             [(circle((0, 0, 0), (0, 0, 1), 1.0), 0, TAU, True, "tangent")], cov_tol=T)


# ----------------------------------------------------------------------------
# 3. isolated tangent points

def class3():
    C = "3 isolated point"
    T = 1e-2
    add_case("c3_sphere_on_plane", C, "sphere touching plane: one point", g.Sphere(zframe((0, 0, 1)), 1.0),
             plane_z(0.0), points=[(0, 0, 0)], cov_tol=T)
    add_case("c3_spheres_external", C, "two spheres touching externally: one point", g.Sphere(F0, 1.0),
             g.Sphere(zframe((2, 0, 0)), 1.0), points=[(1, 0, 0)], cov_tol=T)
    add_case("c3_spheres_internal", C, "sphere touching inside a sphere: one point", g.Sphere(F0, 2.0),
             g.Sphere(g.Frame((1, 0, 0), (0, 1, 0), (0, 0, 1)), 1.0), points=[(2, 0, 0)], cov_tol=T)
    add_case("c3_sphere_torus_outer", C, "sphere touching torus outer equator: one point", g.Torus(F0, 3.0, 1.0),
             g.Sphere(g.Frame((5, 0, 0), (0, 1, 0), (0, 0, 1)), 1.0), points=[(4, 0, 0)], cov_tol=T)
    par = g.heightfield_nurbs([[0, 0, 1], [0, 0, 0], [1, 0, 0]], (-2, 2), (-2, 2), 2, 2, "paraboloid z=x^2+y^2")
    add_case("c3n_parab_apex", C, "NURBS paraboloid apex on plane z=0: one point", par, plane_z(0.0),
             points=[(0, 0, 0)], cov_tol=T)
    parn = g.heightfield_nurbs([[0, 0, -1], [0, 0, 0], [-1, 0, 0]], (-2, 2), (-2, 2), 2, 2, "z=-(x^2+y^2)")
    add_case("c3n_parab_parab_apex", C, "two opposed NURBS paraboloids: one point", par, parn,
             points=[(0, 0, 0)], cov_tol=T)
    q = [[2, 0, -2, 0, 1], [0] * 5, [-2, 0, 0, 0, 0], [0] * 5, [1, 0, 0, 0, 0]]
    qs = g.heightfield_nurbs(q, (-2, 2), (-2, 2), 4, 4, "z=(x^2-1)^2+(y^2-1)^2")
    add_case("c3n_four_points", C, "NURBS quartic z=(x^2-1)^2+(y^2-1)^2 on z=0: four points", qs, plane_z(0.0),
             points=[(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)], cov_tol=T)


# ----------------------------------------------------------------------------
# 4. singular crossing points

def class4():
    C = "4 singular"
    sad = g.heightfield_nurbs([[0, 0], [0, 1]], (-1, 1), (-1, 1), 1, 1, "saddle z=xy")
    add_case("c4n_saddle_tangent_plane", C, "saddle z=xy x its tangent plane z=0: two lines crossing at origin",
             sad, plane_z(0.0, 2), [(line((-1, 0, 0), (1, 0, 0)), 0, 2, False, "crossing"),
                                    (line((0, -1, 0), (0, 1, 0)), 0, 2, False, "crossing")],
             singular=[(0, 0, 0)])
    sph = g.Sphere(g.Frame((0, 0, 0), (-1, 0, 0), (0, 0, 1)), 2.0)
    cyl = g.Cylinder(g.Frame((1, 0, -3), (0, 1, 0), (-1, 0, 0)), 1.0, (0, 360), (0, 6))
    viv = lambda t: np.column_stack([1 + np.cos(t), np.sin(t), 2 * np.sin(t / 2)])
    add_case("c4_viviani_figure8", C, "Viviani: sphere R=2 x cylinder r=1 through centre: figure-eight", sph, cyl,
             [(viv, 0, 2 * TAU, True, "crossing")], singular=[(2, 0, 0)],
             formula="(1+cos t, sin t, 2 sin(t/2)), t in [0,4pi)")
    c1 = g.Cylinder(zframe((0, 0, -2)), 1.0, (0, 360), (0, 4))
    c2 = g.Cylinder(g.Frame((-2, 0, 0), (0, 1, 0), (0, 0, 1)), 1.0, (0, 360), (0, 4))
    e = lambda s: (lambda t: np.column_stack([np.cos(t), np.sin(t), s * np.cos(t)]))
    add_case("c4_steinmetz", C, "equal perpendicular cylinders: two ellipses crossing at two points", c1, c2,
             [(e(1), 0, TAU, True, "crossing"), (e(-1), 0, TAU, True, "crossing")],
             singular=[(0, 1, 0), (0, -1, 0)])
    cone = g.Cone(F0, 30.0, (0.0, 2.0))
    tn = math.tan(math.radians(30))
    pl = g.Plane(g.Frame((0, 0, 0), (0, 1, 0), (0, 0, 1)), (-2, 2), (-1, 3))
    add_case("c4_cone_apex_plane", C, "plane through cone axis: two lines meeting at the apex", cone, pl,
             [(line((0, 0, 0), (0, tn, 1)), 0, 2, False, "crossing"),
              (line((0, 0, 0), (0, -tn, 1)), 0, 2, False, "crossing")], singular=[(0, 0, 0)])
    cusp = g.heightfield_nurbs([[0, 0, 1], [0, 0, 0], [0, 0, 0], [-1, 0, 0]], (-1, 1.2), (-1.2, 1.2), 3, 2,
                               "z=y^2-x^3")
    add_case("c4n_cusp", C, "NURBS z=y^2-x^3 x plane z=0: cuspidal curve y^2=x^3", cusp, plane_z(0.0, 2),
             [(lambda s: np.column_stack([s * s, s ** 3, 0 * s]), -1.2, 1.2, False, "crossing")],
             singular=[(0, 0, 0)], formula="(s^2, s^3, 0)")
    nsph = g.nurbs_sphere(g.Frame((0, 0, 0), (-1, 0, 0), (0, 0, 1)), 2.0, (0, 360), (-90, 90), "sphere")
    ncyl = g.nurbs_cylinder(g.Frame((1, 0, -3), (0, 1, 0), (-1, 0, 0)), 1.0, (0, 360), (0, 6), "cylinder")
    add_case("c4n_viviani_nurbs", C, "Viviani figure-eight with rational NURBS sphere and cylinder", nsph, ncyl,
             [(viv, 0, 2 * TAU, True, "crossing")], singular=[(2, 0, 0)])


# ----------------------------------------------------------------------------
# 5. boundary cases

def class5():
    C = "5 boundary"
    A = g.Plane(zframe(), (0, 2), (0, 2))
    B = g.Plane(g.Frame((0, 0, 0), (0, 1, 0), (0, 0, 1)), (0, 2), (-1, 1))
    add_case("c5_curve_on_edge", C, "plane cuts exactly along the edge u=0 of the other patch", A, B,
             [(line((0, 0, 0), (0, 1, 0)), 0, 2, False, "crossing")])
    A1 = g.Plane(zframe(), (0, 1), (0, 1))
    Bd = g.Plane(g.Frame((1, 0, 0), (-1, 1, 0), (0, 0, 1)), (-0.5, 2), (-1, 1))
    add_case("c5_corner_to_corner", C, "intersection is the diagonal: both endpoints on patch corners", A1, Bd,
             [(line((1, 0, 0), (-1, 1, 0)), 0, 1, False, "crossing")])
    BL = g.Plane(g.Frame((0, 0, 0), (0, 1, 0), (0, 0, 1)), (0, 1), (0, 1))
    add_case("c5_shared_border_L", C, "two patches meeting at 90 deg along a shared border edge", A1, BL,
             [(line((0, 0, 0), (0, 1, 0)), 0, 1, False, "crossing")])
    hemi = g.Sphere(F0, 1.0, (0, 360), (0, 90))
    add_case("c5_hemisphere_equator", C, "hemisphere x plane of its boundary circle", hemi, plane_z(0.0, 2),
             [(circle((0, 0, 0), (0, 0, 1), 1.0), 0, TAU, True, "crossing")])
    cyl = g.Cylinder(zframe(), 1.0, (0, 360), (0, 2))
    add_case("c5_cyl_end_circle", C, "cylinder x plane of its bottom boundary circle", cyl, plane_z(0.0, 2),
             [(circle((0, 0, 0), (0, 0, 1), 1.0), 0, TAU, True, "crossing")])
    q1 = g.Cylinder(zframe(), 1.0, (0, 90), (0, 2))
    q2 = g.Cylinder(zframe(), 1.0, (90, 180), (0, 2))
    add_case("c5_cyl_quilt_seam", C, "two G1 patches of one cylinder sharing a border line", q1, q2,
             [(line((0, 1, 0), (0, 0, 1)), 0, 2, False, "coincident_border")])
    n1 = g.nurbs_cylinder(zframe(), 1.0, (0, 90), (0, 2), "cylinder 0-90")
    n2 = g.nurbs_cylinder(zframe(), 1.0, (90, 180), (0, 2), "cylinder 90-180")
    add_case("c5n_cylN_quilt_seam", C, "two G1 NURBS cylinder patches sharing a border line", n1, n2,
             [(line((0, 1, 0), (0, 0, 1)), 0, 2, False, "coincident_border")])
    nA = g.nurbs_plane(zframe(), (0, 2), (0, 2), 3, 3, "plane", refine=(2, 2))
    add_case("c5n_curve_on_edge", C, "NURBS patch: plane cuts exactly along its edge u=0", nA, B,
             [(line((0, 0, 0), (0, 1, 0)), 0, 2, False, "crossing")])


# ----------------------------------------------------------------------------
# 6. coincident regions

def rect_loop(frame, u, v):
    pts = np.array([[u[0], v[0], 0], [u[1], v[0], 0], [u[1], v[1], 0], [u[0], v[1], 0], [u[0], v[0], 0]], float)

    def f(t):
        t = np.clip(t, 0, 4)
        k = np.minimum(np.floor(t).astype(int), 3)
        a = (t - k)[:, None]
        return frame.to_world(pts[k] * (1 - a) + pts[k + 1] * a)
    return f, 0.0, 4.0


def sph_region(frame, R, lon, lat):
    def f(t):
        t = np.clip(t, 0, 4)
        k = np.minimum(np.floor(t).astype(int), 3)
        a = t - k
        lo = np.where(k == 0, lon[0] + a * (lon[1] - lon[0]), np.where(k == 1, lon[1],
             np.where(k == 2, lon[1] - a * (lon[1] - lon[0]), lon[0])))
        la = np.where(k == 0, lat[0], np.where(k == 1, lat[0] + a * (lat[1] - lat[0]),
             np.where(k == 2, lat[1], lat[1] - a * (lat[1] - lat[0]))))
        lo, la = np.radians(lo), np.radians(la)
        L = np.column_stack([R * np.cos(la) * np.cos(lo), R * np.cos(la) * np.sin(lo), R * np.sin(la)])
        return frame.to_world(L)
    return f, 0.0, 4.0


def cyl_region(frame, r, ang, h):
    def f(t):
        t = np.clip(t, 0, 4)
        k = np.minimum(np.floor(t).astype(int), 3)
        a = t - k
        an = np.where(k == 0, ang[0] + a * (ang[1] - ang[0]), np.where(k == 1, ang[1],
             np.where(k == 2, ang[1] - a * (ang[1] - ang[0]), ang[0])))
        z = np.where(k == 0, h[0], np.where(k == 1, h[0] + a * (h[1] - h[0]),
            np.where(k == 2, h[1], h[1] - a * (h[1] - h[0]))))
        an = np.radians(an)
        return frame.to_world(np.column_stack([r * np.cos(an), r * np.sin(an), z]))
    return f, 0.0, 4.0


def class6():
    C = "6 coincident"
    A = g.Plane(zframe(), (0, 2), (0, 2))
    B = g.Plane(zframe(), (1, 3), (1, 3))
    add_case("c6_planes_overlap", C, "same plane, patches overlapping in [1,2]^2", A, B,
             regions=[rect_loop(zframe(), (1, 2), (1, 2))])
    A1 = g.Plane(zframe(), (0, 1), (0, 1))
    add_case("c6_planes_identical", C, "identical plane patches", A1, g.Plane(zframe(), (0, 1), (0, 1)),
             regions=[rect_loop(zframe(), (0, 1), (0, 1))])
    s1 = g.Sphere(F0, 1.0, (0, 180), (-60, 60))
    s2 = g.Sphere(F0, 1.0, (90, 270), (-30, 80))
    add_case("c6_spheres_overlap", C, "same sphere, patches overlapping in lon[90,180] lat[-30,60]", s1, s2,
             regions=[sph_region(F0, 1.0, (90, 180), (-30, 60))])
    c1 = g.Cylinder(zframe(), 1.0, (0, 180), (0, 2))
    c2 = g.Cylinder(zframe(), 1.0, (90, 270), (1, 3))
    add_case("c6_cyl_overlap", C, "same cylinder, patches overlapping in angle[90,180] h[1,2]", c1, c2,
             regions=[cyl_region(zframe(), 1.0, (90, 180), (1, 2))])
    nB = g.nurbs_plane(zframe(), (1, 3), (0.5, 1.5), 3, 3, "plane (warped parameterisation)", warp=0.6)
    add_case("c6n_plane_vs_nurbsplane", C, "analytic plane vs NURBS plane (non-uniform param), partial overlap",
             A, nB, regions=[rect_loop(zframe(), (1, 2), (0.5, 1.5))])
    nc2 = g.nurbs_cylinder(zframe(), 1.0, (90, 270), (1, 3), "cylinder")
    add_case("c6n_cyl_vs_nurbscyl", C, "analytic cylinder vs rational NURBS cylinder, partial overlap", c1, nc2,
             regions=[cyl_region(zframe(), 1.0, (90, 180), (1, 2))])
    sp = g.Sphere(F0, 1.0, (0, 90), (0, 60))
    nsp = g.nurbs_sphere(F0, 1.0, (0, 90), (0, 60), "sphere patch")
    add_case("c6n_sphere_vs_nurbssphere", C, "analytic sphere patch vs identical rational NURBS patch", sp, nsp,
             regions=[sph_region(F0, 1.0, (0, 90), (0, 60))])


# ----------------------------------------------------------------------------
# 7. near-degenerate sweeps

ZONE_TOL = 1e-5  # SMLib default zone tolerance (SM_ZONE_TOL_3D)


def class7():
    C = "7 near-degenerate"
    # (a) almost tangent planes: angle theta between plane z=0 (analytic) and a NURBS plane
    for e in range(3, 9):
        th = 10.0 ** -e
        A = g.Plane(zframe(), (-1, 1), (-1, 1))
        Bn = g.nurbs_plane(g.Frame((0, 0, 0), (1, 0, 0), (0, math.cos(th), math.sin(th))), (-1, 1), (-1, 1), 1, 1,
                           "plane tilted 1e-%d" % e)
        add_case("c7_angle_planes_1e-%d" % e, C, "planes crossing at angle 1e-%d rad: one line" % e, A, Bn,
                 [(line((-1, 0, 0), (1, 0, 0)), 0, 2, False, "near_tangent")],
                 cov_tol=1e-3, ill_cond=1.0 / math.sin(th), feature=th, sub_tolerance=bool(math.sin(th) * 1.0 <= ZONE_TOL),
                 notes="the exact line is determined only to ~tol/angle laterally")
    # (b) cylinder slightly sunk into a plane: two lines at crossing angle ~theta
    for e in range(3, 7):
        th = 10.0 ** -e
        d = 1 - math.cos(th)  # sink depth so that the crossing angle is exactly theta
        zc = 1.0 - d
        zc_f = Fraction(zc)  # exact value of the stored double
        yy = math.sqrt(float(1 - zc_f * zc_f))
        cyl = g.Cylinder(g.Frame((2, 0, zc), (0, 0, 1), (0, 1, 0)), 1.0, (0, 360), (0, 4))
        add_case("c7_sunk_cyl_1e-%d" % e, C, "cylinder sunk into plane, crossing angle 1e-%d: two lines %.1e apart"
                 % (e, 2 * yy), plane_z(0.0), cyl,
                 [(line((-2, yy, 0), (1, 0, 0)), 0, 4, False, "near_tangent"),
                  (line((-2, -yy, 0), (1, 0, 0)), 0, 4, False, "near_tangent")],
                 cov_tol=min(1e-3, 0.25 * yy), ill_cond=1.0 / math.sin(th), feature=2 * yy,
                 sub_tolerance=bool(2 * yy < ZONE_TOL))
    # (c) tiny loops: sphere R=1 dipping into plane; loop diameter = s * model size (2)
    for e in range(4, 8):
        rho_t = 10.0 ** -e
        zc = math.sqrt(1 - rho_t * rho_t) if rho_t > 1e-7 else 1.0 - rho_t * rho_t / 2
        zf = Fraction(zc)
        rho = math.sqrt(float(1 - zf * zf))
        sph = g.Sphere(zframe((0, 0, zc)), 1.0)
        add_case("c7_tiny_loop_sphere_1e-%d" % e, C, "sphere dipping into plane: loop of diameter %.2g (1e-%d x model)"
                 % (2 * rho, e), plane_z(0.0, 1), sph,
                 [(circle((0, 0, 0), (0, 0, 1), rho), 0, TAU, True, "crossing")],
                 cov_tol=0.2 * rho, feature=2 * rho, sub_tolerance=bool(2 * rho < ZONE_TOL))
    par = g.heightfield_nurbs([[0, 0, 1], [0, 0, 0], [1, 0, 0]], (-1, 1), (-1, 1), 2, 2, "paraboloid z=x^2+y^2")
    for e in range(4, 8):
        rho = 10.0 ** -e
        eps = rho * rho
        rho = math.sqrt(eps)
        add_case("c7_tiny_loop_parab_1e-%d" % e, C, "NURBS paraboloid x plane z=%.0e: loop of diameter %.2g"
                 % (eps, 2 * rho), par, plane_z(eps, 1),
                 [(circle((0, 0, eps), (0, 0, 1), rho), 0, TAU, True, "crossing")],
                 cov_tol=0.2 * rho, feature=2 * rho, sub_tolerance=bool(2 * rho < ZONE_TOL))
    # (d) nearly coincident offsets: no intersection at all
    for e in (3, 5, 7):
        eps = 10.0 ** -e
        A = g.Plane(zframe(), (0, 1), (0, 1))
        Bn = g.nurbs_plane(zframe((0, 0, eps)), (0, 1), (0, 1), 3, 3, "plane z=1e-%d" % e, refine=(2, 2))
        add_case("c7_offset_planes_1e-%d" % e, C, "parallel planes %.0e apart: no intersection" % eps, A, Bn,
                 feature=eps, sub_tolerance=bool(eps < ZONE_TOL))
        s1 = g.Sphere(F0, 1.0, (0, 180), (-60, 60))
        s2 = g.Sphere(F0, 1.0 + eps, (0, 180), (-60, 60))
        add_case("c7_offset_spheres_1e-%d" % e, C, "concentric sphere patches, radii differ by %.0e: no intersection"
                 % eps, s1, s2, feature=eps, sub_tolerance=bool(eps < ZONE_TOL))


# ----------------------------------------------------------------------------
# 8. periodic / closed surfaces, seams, poles

def class8():
    C = "8 periodic/seam/pole"
    sph = g.Sphere(F0, 1.0)
    add_case("c8_sphere_plane_poles", C, "great circle through both poles (plane x=0)", sph,
             g.Plane(g.Frame((0, 0, 0), (0, 1, 0), (0, 0, 1)), (-2, 2), (-2, 2)),
             [(circle((0, 0, 0), (1, 0, 0), 1.0), 0, TAU, True, "crossing")])
    add_case("c8_sphere_plane_poles_seam", C, "great circle through both poles, half of it on the seam meridian (y=0)",
             sph, g.Plane(g.Frame((0, 0, 0), (1, 0, 0), (0, 0, 1)), (-2, 2), (-2, 2)),
             [(circle((0, 0, 0), (0, 1, 0), 1.0), 0, TAU, True, "crossing")])
    add_case("c8_sphere_polar_cap", C, "small circle around the pole (plane z=0.9)", sph, plane_z(0.9, 2),
             [(circle((0, 0, 0.9), (0, 0, 1), math.sqrt(1 - 0.81)), 0, TAU, True, "crossing")])
    cyl = g.Cylinder(zframe((0, 0, -2)), 1.0, (0, 360), (0, 4))
    add_case("c8_cyl_plane_seam_line", C, "plane y=0 through the axis: two lines, one on the cylinder seam", cyl,
             g.Plane(g.Frame((0, 0, 0), (1, 0, 0), (0, 0, 1)), (-2, 2), (-3, 3)),
             [(line((1, 0, -2), (0, 0, 1)), 0, 4, False, "crossing"),
              (line((-1, 0, -2), (0, 0, 1)), 0, 4, False, "crossing")])
    tor = g.Torus(F0, 3.0, 1.0)
    add_case("c8_torus_equator_seam", C, "torus x plane z=0: outer circle lies on the minor seam", tor, plane_z(0.0, 5),
             [(circle((0, 0, 0), (0, 0, 1), 4.0), 0, TAU, True, "crossing"),
              (circle((0, 0, 0), (0, 0, 1), 2.0), 0, TAU, True, "crossing")])
    add_case("c8_torus_axis_plane_seam", C, "torus x plane y=0: two meridian circles, one on the major seam", tor,
             g.Plane(g.Frame((0, 0, 0), (1, 0, 0), (0, 0, 1)), (-5, 5), (-2, 2)),
             [(circle((3, 0, 0), (0, 1, 0), 1.0), 0, TAU, True, "crossing"),
              (circle((-3, 0, 0), (0, 1, 0), 1.0), 0, TAU, True, "crossing")])
    cone = g.Cone(F0, 30.0, (0.5, 3.0))
    tn = math.tan(math.radians(30))
    add_case("c8_cone_plane_seam", C, "cone x plane y=0 (apex excluded): two rulings, one on the seam", cone,
             g.Plane(g.Frame((0, 0, 0), (1, 0, 0), (0, 0, 1)), (-3, 3), (0, 4)),
             [(line((0, 0, 0), (tn, 0, 1)), 0.5, 3.0, False, "crossing"),
              (line((0, 0, 0), (-tn, 0, 1)), 0.5, 3.0, False, "crossing")])

    cylp = g.Cylinder(zframe((0.3, 0, -2)), 0.3, (0, 360), (0, 4))

    def pl(s):
        def f(t):
            x = 0.3 + 0.3 * np.cos(t)
            y = 0.3 * np.sin(t)
            return np.column_stack([x, y, s * np.sqrt(np.maximum(1 - x * x - y * y, 0))])
        return f
    add_case("c8_sphere_cyl_through_poles", C, "sphere x cylinder touching the polar axis: two loops through the poles",
             sph, cylp, [(pl(1), 0, TAU, True, "crossing"), (pl(-1), 0, TAU, True, "crossing")])
    pf = g.Frame((0, 0, 0), (1, 0, 0), (0, 1, 0.5))
    ncyl = g.nurbs_cylinder(F0, 1.0, (0, 360), (-2, 2), "cylinder")
    ell = lambda t: np.column_stack([np.cos(t), np.sin(t), 0.5 * np.sin(t)])
    add_case("c8n_cylN_seam_line", C, "NURBS full cylinder x plane y=0: one line on the closed-u seam", ncyl,
             g.Plane(g.Frame((0, 0, 0), (1, 0, 0), (0, 0, 1)), (-2, 2), (-3, 3)),
             [(line((1, 0, -2), (0, 0, 1)), 0, 4, False, "crossing"),
              (line((-1, 0, -2), (0, 0, 1)), 0, 4, False, "crossing")])
    nsph = g.nurbs_sphere(F0, 1.0, (0, 360), (-90, 90), "sphere")
    add_case("c8n_sphereN_poles", C, "NURBS sphere (degenerate pole rows) x plane x=0: circle through both poles",
             nsph, g.Plane(g.Frame((0, 0, 0), (0, 1, 0), (0, 0, 1)), (-2, 2), (-2, 2)),
             [(circle((0, 0, 0), (1, 0, 0), 1.0), 0, TAU, True, "crossing")])
    add_case("c8n_sphereN_polar_cap", C, "NURBS sphere x plane z=0.95: small circle around the degenerate pole",
             nsph, plane_z(0.95, 2), [(circle((0, 0, 0.95), (0, 0, 1), math.sqrt(1 - 0.95 ** 2)), 0, TAU, True,
                                       "crossing")])
    ntor = g.nurbs_torus(F0, 3.0, 1.0, (0, 360), (0, 360), "torus")
    add_case("c8n_torusN_equator_seam", C, "NURBS torus x plane z=0: outer circle on the closed-v seam", ntor,
             plane_z(0.0, 5), [(circle((0, 0, 0), (0, 0, 1), 4.0), 0, TAU, True, "crossing"),
                               (circle((0, 0, 0), (0, 0, 1), 2.0), 0, TAU, True, "crossing")])


# ----------------------------------------------------------------------------
# 9. high degree, many spans, rational

def class9():
    C = "9 high degree/rational"
    c = np.zeros((7, 7))
    for i in range(4):  # (x^2+y^2)^3
        c[2 * i, 6 - 2 * i] = math.comb(3, i)
    h6 = g.heightfield_nurbs(c, (-1.2, 1.2), (-1.2, 1.2), 6, 6, "z=(x^2+y^2)^3", refine=(8, 8))
    r = 0.3 ** (1 / 6)
    add_case("c9n_deg6_circle", C, "degree-6 NURBS, 9x9 spans, z=(x^2+y^2)^3 x plane z=0.3: circle", h6,
             plane_z(0.3, 2), [(circle((0, 0, 0.3), (0, 0, 1), r), 0, TAU, True, "crossing")])
    ncyl = g.nurbs_cylinder(F0, 1.0, (0, 90), (-2, 2), "cylinder 0-90").elevate(3, 2).refine_uniform(6, 6)
    pl = g.Plane(g.Frame((0, 0, 0), (1, 0, 0), (0, 1, 0.5)), (-3, 3), (-3, 3))
    ell = lambda t: np.column_stack([np.cos(t), np.sin(t), 0.5 * np.sin(t)])
    add_case("c9n_rational_deg5_cyl_arc", C, "rational degree (5,3) cylinder patch, 7x7 spans x tilted plane: open arc",
             ncyl, pl, [(ell, 0, TAU, True, "crossing")])
    nsp = g.nurbs_sphere(F0, 2.0, (0, 90), (0, 60), "sphere patch").elevate(3, 3).refine_uniform(5, 5)
    add_case("c9n_rational_deg5_sphere_arc", C, "rational degree-5 sphere patch, 6x6 spans x plane z=1: open arc",
             nsp, plane_z(1.0), [(circle((0, 0, 1), (0, 0, 1), math.sqrt(3)), 0, TAU, True, "crossing")])
    p5 = g.heightfield_nurbs([[0, 0, 1], [0, 0, 0], [1, 0, 0]], (-2, 2), (-2, 2), 5, 5, "z=x^2+y^2 deg5",
                             refine=(10, 10))
    p4 = g.heightfield_nurbs([[2, 0, -1], [0, 0, 0], [-1, 0, 0]], (-2, 2), (-2, 2), 4, 4, "z=2-x^2-y^2 deg4",
                             refine=(7, 7))
    add_case("c9n_elevated_paraboloids", C, "degree-5 (11x11 spans) vs degree-4 (8x8 spans) paraboloids: circle",
             p5, p4, [(circle((0, 0, 1), (0, 0, 1), 1.0), 0, TAU, True, "crossing")])
    npl = g.nurbs_plane(zframe((0, 0, 0.5)), (-2.5, 2.5), (-2.5, 2.5), 3, 3, "plane, 31x31 spans", refine=(30, 30),
                        warp=0.6)
    add_case("c9n_many_spans_plane_sphere", C, "cubic NURBS plane with 31x31 spans (non-uniform) x sphere R=2",
             npl, g.Sphere(F0, 2.0), [(circle((0, 0, 0.5), (0, 0, 1), math.sqrt(4 - 0.25)), 0, TAU, True,
                                       "crossing")])
    ntor = g.nurbs_torus(F0, 3.0, 1.0, (0, 360), (0, 360), "torus").refine_uniform(12, 12)
    rho = 17.0 / 6.0
    zz = math.sqrt(9 - rho * rho)
    add_case("c9n_refined_torus_sphere", C, "rational torus refined to 16x16 spans x sphere R=3: two circles",
             ntor, g.Sphere(F0, 3.0), [(circle((0, 0, zz), (0, 0, 1), rho), 0, TAU, True, "crossing"),
                                       (circle((0, 0, -zz), (0, 0, 1), rho), 0, TAU, True, "crossing")])


def main():
    for f in (class1, class2, class3, class4, class5, class6, class7, class8, class9):
        f()
    os.makedirs(OUT, exist_ok=True)
    for old in os.listdir(OUT):
        if old.endswith(".json") and not old.startswith("y23_"):
            os.remove(os.path.join(OUT, old))
    ids = set()
    for c in CASES:
        assert c["id"] not in ids, c["id"]
        ids.add(c["id"])
        with open(os.path.join(OUT, c["id"] + ".json"), "w") as fh:
            json.dump(c, fh)
    idx = [{"id": c["id"], "class": c["class"], "title": c["title"], "surfaces": c["surface_labels"],
            "branches": len(c["expected"]["branches"]),
            "closed": [b["closed"] for b in c["expected"]["branches"]],
            "isolated_points": len(c["expected"]["isolated_points"]),
            "singular_points": len(c["expected"]["singular_points"]),
            "coincident_regions": len(c["expected"]["coincident_regions"])} for c in CASES]
    with open(os.path.join(HERE, "index.json"), "w") as fh:
        json.dump(idx, fh, indent=1)
    print(len(CASES), "cases")
    for i in idx:
        print("%-34s %-22s br=%d closed=%s pts=%d sing=%d reg=%d" % (
            i["id"], i["class"], i["branches"], i["closed"], i["isolated_points"], i["singular_points"],
            i["coincident_regions"]))


if __name__ == "__main__":
    main()
