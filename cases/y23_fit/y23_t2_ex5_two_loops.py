"""Table 2 ex.5 of Yang, Jia & Yan 2023 ("two loops", bi-degree (3,3)), refitted to the paper's figures.

What the paper shows (cases/y23_fit/paper/y23_t2_ex5_two_loops_*.png, 600 dpi):
  * Surfaces panel: two tube-like arms forming an X; blue from top-left to bottom-right, orange from
    top-right to bottom-left; each arm is S-shaped (steep at its ends, nearly horizontal in the middle),
    and the two arms run together through a long horizontal overlap in the middle. The arm ends are open
    (the lighter ovals at the tips are the openings seen at an angle). Orange is the mirror image of
    blue in the vertical plane through the middle of the X.
  * Result row: a long flat loop with a corner at each end, crossed twice by a small loop that has a
    corner at its bottom and a rounded top.
  * Fig. 10 (right), the (s,t) domain of one surface: a full vertical iso-line s = 0.5; an S-shaped
    branch from (0.21, 0) through (0.5, 0.21) [flat there], (0.79, 0.5), (0.5, 0.79) [flat] to
    (0.21, 1); and a D-shaped branch that the clipping step removes. The branch ends at t = 0 and t = 1
    meet at equal s: the patch is closed in t.

Reading of Fig. 10 used here:
  * the tube is ONE bicubic Bezier patch closed in t (first and last control columns equal), so its
    cross-section is a closed cubic with one corner, at the seam t = 0 = 1;
  * every s-iso-curve lies in a plane x = const (control columns in planes x = X_i): then the mirror
    plane x = 0 meets the blue tube exactly in the iso-curve s = 0.5, which lies on the orange (its own
    mirror image) -- the full iso-line of Fig. 10;
  * with the cross-section control polygon (-1, 0), (c1, k), (c1, -k), (-1, 0) (in (C, S)), the depth
    S(t) = 3k t(1-t)(1-2t) is extreme at t = 1/2 -+ sqrt(3)/6 = 0.211, 0.789: exactly the t where the
    paper's S-branch crosses s = 0.5. The branch is flat there iff the arm's height z(s) has no linear
    term at s = 1/2: the arms are horizontal (and tangent to each other) at the middle of the X.

Construction (exact), sigma = s - 1/2:
    blue   A(s,t) = (x(s), b(s) S(t), z(s) + a(s) C(t)),  (s,t) in [0,1]^2
    orange B(s,t) = (-x(s), b(s) S(t), z(s) + a(s) C(t))  (mirror in x = 0)
    x = x1 sigma + x3 sigma^3 (odd, strictly increasing), z = z2 sigma^2 + z3 sigma^3,
    a = a0 + a2 sigma^2, b = b0 + b2 sigma^2 (even, positive); bi-degree (3,3), polynomial.
  A(s,t) = B(s',t') forces s' = 1 - s (x odd, monotone), then S(t') = S(t) and
  a(s) (C(t) - C(t')) + 2 z3 sigma^3 = 0. Either t' = t, which gives sigma = 0: the ring s = 1/2 (the
  "small loop", with the seam corner at its bottom); or t' is the other point of the cross-section at
  the same depth: the "long loop", whose two ends are where one tube's corner meets the other tube's
  top (a corner at each end). The two loops cross at A(1/2, 1/2 -+ sqrt(3)/6), where the tubes are
  tangent (both normals along y). Nothing else: the intersection is exactly these two loops.

Ground truth: the ring is closed form; long-loop points come from the closed-form cubic roots and are
Newton-polished on the factored system (Q(t,t') = 0, a (t - t') R(t,t') + 2 z3 sigma^3 = 0) to ~1e-15.
Every stored point is evaluated on blue at (s, t) and on orange at (1 - s, t'), both are the same point.

Run: .venv/bin/python -I cases/y23_fit/y23_t2_ex5_two_loops.py   (writes cases/data/y23_t2_ex5_two_loops.json)
"""
import json
import math
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

CID = "y23_t2_ex5_two_loops"
TE = (0.5 - math.sqrt(3) / 6, 0.5 + math.sqrt(3) / 6)  # depth extremes of the cross-section

# fitted shape (fit_t2_ex5.py against the Surfaces panel + Fig. 10's loop span s in [0.21, 0.79])
PARAMS = dict(x1=2.58923, x3=0.0217, z2=0.06883, z3=-14.76281, a0=0.4011, a2=0.84338, b0=0.44347, b2=0.0,
              c1=1.16587, k=2.02012)
# fitted camera of the Surfaces panel (render_match.py convention), panel 300 x 430 px
CAMERA = dict(proj="ortho", el=9.63047, az=-64.16847, roll=5.42268, target=[0.0, 0.0, 0.0], scale=84.35911,
              cx=147.8583, cy=202.94919)
# fitted camera of the result ("Ours") panel, 520 x 260 px
RESULT_CAMERA = dict(proj="ortho", el=31.83776, az=-57.86579, roll=-4.38152, target=[0.0, 0.0, 0.0],
                     scale=341.65701, cx=262.68709, cy=114.21637)
ALPHAS = (0.75, 0.75)
# IoU of the previous reconstruction (commit c91ad29) from its own best-fitting camera (run with --before)
BEFORE = dict(before_iou_union=0.4138, before_iou_blue=0.416, before_iou_orange=0.3764)


# ----------------------------------------------------------------------------------- shape

def section_power(p):
    """Power coefficients (in t) of C(t), S(t) for the control polygon (-1,0),(c1,k),(c1,-k),(-1,0)."""
    M = np.array([[1, 0, 0, 0], [-3, 3, 0, 0], [3, -6, 3, 0], [-1, 3, -3, 1]], float)
    Q = np.array([[-1.0, 0.0], [p["c1"], p["k"]], [p["c1"], -p["k"]], [-1.0, 0.0]])
    a = M @ Q
    return a[:, 0], a[:, 1]


def sig_poly(p):
    """Power coefficients in sigma of x, z, a, b."""
    return (np.array([0.0, p["x1"], 0.0, p["x3"]]), np.array([0.0, 0.0, p["z2"], p["z3"]]),
            np.array([p["a0"], 0.0, p["a2"]]), np.array([p["b0"], 0.0, p["b2"]]))


def ev(c, x):
    return np.polyval(np.asarray(c)[::-1], x)


def blue_pt(p, s, t):
    xs, zs, as_, bs = sig_poly(p)
    Cc, Sc = section_power(p)
    sg = np.asarray(s) - 0.5
    return np.stack(np.broadcast_arrays(ev(xs, sg), ev(bs, sg) * ev(Sc, t), ev(zs, sg) + ev(as_, sg) * ev(Cc, t)), -1)


def orange_pt(p, s, t):
    P = blue_pt(p, s, t)
    return P * np.array([-1.0, 1.0, 1.0])


def grids(p, n=61):
    s = np.linspace(0, 1, n)[:, None]
    t = np.linspace(0, 1, n)[None, :]
    return blue_pt(p, s, t), orange_pt(p, s, t)


def valid(p):
    sg = np.linspace(-0.5, 0.5, 101)
    xs, zs, as_, bs = sig_poly(p)
    return bool((ev(np.polyder(xs[::-1])[::-1], sg) > 0).all() and (ev(as_, sg) > 0).all()
                and (ev(bs, sg) > 0).all() and p["c1"] > -1 and p["k"] > 0 and p["z3"] != 0
                and loop_span(p)[0] > 0.02)


def loop_span(p):
    """s-range of the long loop: 2 |z3| sigma^3 = a(sigma) (C(1/2) - C(0))."""
    Cc, _ = section_power(p)
    h = ev(Cc, 0.5) - ev(Cc, 0.0)
    r = np.roots([2 * abs(p["z3"]), -h * p["a2"], 0.0, -h * p["a0"]])
    r = [x.real for x in r if abs(x.imag) < 1e-12 and x.real > 0]
    s1 = min(r)
    return 0.5 - s1, 0.5 + s1


# ----------------------------------------------------------------------------------- surfaces

def to_s(c_sig):
    """Power coefficients in sigma = s - 1/2 -> in s."""
    c = np.zeros(len(c_sig))
    for i, ci in enumerate(c_sig):
        for j in range(i + 1):
            c[j] += ci * math.comb(i, j) * (-0.5) ** (i - j)
    return c


def surfaces(p=PARAMS):
    xs, zs, as_, bs = (to_s(c) for c in sig_poly(p))
    Cc, Sc = section_power(p)
    X = Y.P(*[[c] for c in xs])
    Yh = np.outer(bs, Sc)
    Zh = np.zeros((4, 4))
    Zh[:, 0] += zs
    Zh[:len(as_), :] += np.outer(as_, Cc)
    one = Y.P([1.0])
    blue = g.rational_poly_patch(X, Yh, Zh, one, (0.0, 1.0), (0.0, 1.0),
                                 "tube arm: bicubic patch closed in t (seam corner), sections in planes x = const",
                                 degree=(3, 3))
    orange = g.rational_poly_patch(-X, Yh, Zh, one, (0.0, 1.0), (0.0, 1.0),
                                   "tube arm: mirror image of the blue arm in x = 0", degree=(3, 3))
    return blue, orange


# ----------------------------------------------------------------------------------- truth

def partner(p, t):
    """t' != t with S(t') = S(t) on the same half of the section, in closed form.

    S(t) - S(t') = (t - t') Q(t, t') with Q = S1 + S2 (t + t') + S3 (t^2 + t t' + t'^2): a quadratic in t'."""
    _, Sc = section_power(p)
    t = np.asarray(t, float)
    a, b, c = Sc[3], Sc[2] + Sc[3] * t, Sc[1] + Sc[2] * t + Sc[3] * t * t
    disc = np.sqrt(np.clip(b * b - 4 * a * c, 0, None))
    r1, r2 = (-b + disc) / (2 * a), (-b - disc) / (2 * a)
    lo = t <= 0.5
    ok1 = np.where(lo, (r1 >= -1e-12) & (r1 <= 0.5 + 1e-12), (r1 >= 0.5 - 1e-12) & (r1 <= 1 + 1e-12))
    return np.clip(np.where(ok1, r1, r2), 0.0, 1.0)


def sigma_of(p, t, tp):
    """The real root sigma in [-1/2, 1/2] of 2 z3 sigma^3 + (a0 + a2 sigma^2) (C(t) - C(t')) = 0."""
    Cc, _ = section_power(p)
    dC = ev(Cc, t) - ev(Cc, tp)
    out = np.zeros_like(np.asarray(t, float))
    for i, d in enumerate(np.atleast_1d(dC)):
        if d == 0.0:
            continue
        r = np.roots([2 * p["z3"], p["a2"] * d, 0.0, p["a0"] * d])
        r = [x.real for x in r if abs(x.imag) < 1e-9 * max(1, abs(x.real)) and abs(x.real) <= 0.5 + 1e-12]
        assert len(r) == 1, (t[i] if np.ndim(t) else t, r)
        out.flat[i] = r[0]
    return out


def _phi_to_t(phi):
    """phi in [0, 4] -> t in [0, 1], cube-law clustered at the depth extremes TE (sigma ~ (t - te)^(1/3))."""
    knots = np.array([0.0, TE[0], 0.5, TE[1], 1.0])
    phi = np.asarray(phi, float)
    k = np.clip(np.floor(phi).astype(int), 0, 3)
    w = phi - k
    lo, hi = knots[k], knots[k + 1]
    toward_hi = (k == 0) | (k == 2)
    return np.where(toward_hi, hi - (hi - lo) * (1 - w) ** 3, lo + (hi - lo) * w ** 3)


def long_loop_params(p, phi):
    t = _phi_to_t(phi)
    tp = partner(p, t)
    tp = np.where((t == TE[0]) | (t == TE[1]), t, tp)
    return sigma_of(p, t, tp), t, tp


def long_loop_fn(p, n_dense=200001):
    """fn(tau), tau in [0, 1]: the long loop by (approximate) arc length, every point in closed form."""
    phi = np.linspace(0, 4, n_dense)
    sg, t, _ = long_loop_params(p, phi)
    P = blue_pt(p, sg + 0.5, t)
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    L /= L[-1]

    def fn(tau):
        ph = np.interp(np.asarray(tau, float) % 1.0 if np.ndim(tau) else tau, L, phi)
        ph = np.where(np.asarray(tau) >= 1.0, 4.0, ph)
        s_, t_, _ = long_loop_params(p, ph)
        return blue_pt(p, s_ + 0.5, t_)
    return fn


def ring_fn(p):
    return lambda t: blue_pt(p, 0.5 + 0 * np.asarray(t, float), t)


def loop_gap(p, n=20001):
    """max |A(s,t) - B(1-s,t')| over the long loop (both evaluations of the same stored point)."""
    sg, t, tp = long_loop_params(p, np.linspace(0, 4, n))
    return float(np.linalg.norm(blue_pt(p, 0.5 + sg, t) - orange_pt(p, 0.5 - sg, tp), axis=1).max())


def truth_curves(p=PARAMS, n=1200):
    """(long loop, ring) as polylines, for rendering and fitting."""
    fn = long_loop_fn(p, 20001)
    tau = np.linspace(0, 1, n)
    return [fn(tau), ring_fn(p)(np.linspace(0, 1, n // 2))]


def normals_sin(p, s, t, sp, tp, h=1e-6):
    def nrm(f, s_, t_):
        du = (f(p, s_ + h, t_) - f(p, s_ - h, t_)) / (2 * h)
        dv = (f(p, s_, t_ + h) - f(p, s_, t_ - h)) / (2 * h)
        n = np.cross(du, dv)
        return n / np.linalg.norm(n)
    return float(np.linalg.norm(np.cross(nrm(blue_pt, s, t), nrm(orange_pt, sp, tp))))


# ----------------------------------------------------------------------------------- case

def checks(p=PARAMS):
    out = {}
    out["long_loop_gap_A_vs_B"] = loop_gap(p)
    out["sin_normals_at_crossings"] = [normals_sin(p, 0.5, te, 0.5, te) for te in TE]
    sg, t, tp = long_loop_params(p, np.linspace(0.05, 3.95, 9))
    out["sin_normals_on_long_loop_min"] = float(min(normals_sin(p, 0.5 + a, b, 0.5 - a, c) for a, b, c in zip(sg, t, tp)
                                                   if abs(a) > 0.03))
    out["loop_span_s"] = list(loop_span(p))
    return out


def build(p=PARAMS, fit=None):
    blue, orange = surfaces(p)
    sing = [blue_pt(p, 0.5, te).tolist() for te in TE]
    fit = fit or {}
    cam_txt = lambda c: "%s el=%.2f az=%.2f roll=%.2f" % (c["proj"], c["el"], c["az"], c["roll"])
    Y.add(CID, "Table 2 ex.5 (3,3): two loops (two S-shaped tube arms crossing in an X)", blue, orange,
          Y.SRC + "Table 2 (example 5)", "'two loops', bi-degree (3,3)",
          "bi-degree (3,3), polynomial, one Bezier patch each; topology: two closed loops crossing at two points "
          "(tangential nodes), the small loop a second-order tangential curve along its whole length; appearance fitted to the paper (cameras in render_match.py convention): Surfaces "
          "panel camera %s, silhouette IoU union %s / blue %s / orange %s; result-row camera %s, mean chamfer "
          "%s px between our loops and the paper's red curves; Fig. 10 (right) reproduced: the small loop is the "
          "full iso-line s = 1/2, the long loop crosses it at t = 1/2 -+ sqrt(3)/6 = 0.211 / 0.789 with a flat "
          "tangent and spans s in [%.3f, %.3f] (paper 0.21 / 0.79); the long loop has a corner at each end and the "
          "small loop one at its bottom, as drawn (the tube's seam corner)"
          % (cam_txt(CAMERA), fit.get("iou_union", "?"), fit.get("iou_blue", "?"), fit.get("iou_orange", "?"),
             cam_txt(RESULT_CAMERA), fit.get("chamfer_result_px", "?"), *loop_span(p)),
          "the tube is one bicubic patch closed in t with its seam corner at the bottom and cross-sections in planes "
          "x = const (read from Fig. 10); the arm shape x(s), z(s), the section size a(s), b(s) and the section "
          "polygon (c1, k) are fitted to the figures, not published; the paper's D-shaped branch in Fig. 10 "
          "(removed by its clipping) is not checked",
          "paper: " + Y.OPEN + " and " + Y.COMM + " returned one loop only, " + Y.SPL + " returned nothing",
          curves=[(long_loop_fn(p), 0.0, 1.0, True, "crossing"), (ring_fn(p), 0.0, 1.0, True, "tangent")],
          singular=sing, cov_tol=5e-3,
          notes="blue A(s,t) = (x(s), b(s)S(t), z(s) + a(s)C(t)), orange = A mirrored in x = 0; sigma = s - 1/2: "
                "x = %.6g sigma + %.6g sigma^3, z = %.6g sigma^2 + %.6g sigma^3, a = %.6g + %.6g sigma^2, "
                "b = %.6g; section = cubic Bezier (-1,0), (%.6g, %.6g), (%.6g, %.6g), (-1,0) in (C, S)"
                % (p["x1"], p["x3"], p["z2"], p["z3"], p["a0"], p["a2"], p["b0"], p["c1"], p["k"], p["c1"], -p["k"]))
    case = gc.CASES[-1]
    case["truth_method"] = ("closed form: small loop = iso-line s = 1/2 of the blue (on the orange by mirror "
                            "symmetry); long loop = {(s,t): S(t') = S(t), a(s)(C(t) - C(t')) + 2 z3 (s - 1/2)^3 = 0}, "
                            "t' from a quadratic, s from a cubic, every stored point evaluated on the blue at (s, t) "
                            "and equal to the orange at (1 - s, t') to ~1e-15")
    return case


def compare(p=PARAMS, old_case=None):
    """Deliverables: compare/<CID>.png (paper | ours | overlay for the Surfaces, AA and result panels) and
    compare/<CID>_truth.png (exact truth in red, result-row style, beside the paper's result panel).
    Returns the fit numbers (IoU before/after, chamfer)."""
    sys.path.insert(0, HERE)
    import render_match as rm
    import fit_tools_t2_ex456 as T
    from fit_t2_ex5_result import chamfer
    out = {}
    Ps, Pa, Pr = (T.paper(CID, k) for k in ("surfaces", "aagpu", "result"))
    A, B = grids(p, 121)
    H, W = Ps.shape[:2]
    rgb_s, m_s = rm.render([A, B], CAMERA, W, H, style="surfaces", alphas=ALPHAS, n=121)
    out["iou_union"], out["iou_blue"], out["iou_orange"] = T.ious(rm, Ps, rgb_s, m_s)
    cam_a, out["iou_aagpu_union"] = T.fit_framing(rm, [A, B], CAMERA, Pa)
    rgb_a, _ = rm.render([A, B], cam_a, Pa.shape[1], Pa.shape[0], style="surfaces", alphas=(1.0, 1.0), n=121,
                         colors=(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])))
    cur = truth_curves(p, 3000)
    sing = [blue_pt(p, 0.5, te) for te in TE]
    rgb_r = T.curves_only(rm, cur, RESULT_CAMERA, Pr.shape[1], Pr.shape[0])
    out["chamfer_result_px"] = round(chamfer(truth_curves(p, 1200), RESULT_CAMERA), 2)
    rm.compare_strip([Ps, Pa, Pr], [rgb_s, rgb_a, rgb_r], os.path.join(HERE, "compare", CID + ".png"))
    rgb_t, _ = rm.render([A, B], RESULT_CAMERA, Pr.shape[1], Pr.shape[0], style="result", curves=cur, points=sing,
                         n=121, point_radius=4.0)
    rm.compare_strip([Pr], [rgb_t], os.path.join(HERE, "compare", CID + "_truth.png"))
    if old_case is not None:  # the previous reconstruction, from its own best-fitting camera: IoU "before"
        G = [rm.surface_grid(sp, 41) for sp in old_case["surfaces"]]
        cam_o, _ = T.fit_camera(rm, G, Ps, ALPHAS)
        rgb_o, m_o = rm.render(G, cam_o, W, H, style="surfaces", alphas=ALPHAS, n=41)
        out["before_iou_union"], out["before_iou_blue"], out["before_iou_orange"] = T.ious(rm, Ps, rgb_o, m_o)
        rm.compare_strip([Ps], [rgb_o], os.path.join(HERE, "compare", CID + "_before.png"))
    return out


def main():
    import subprocess
    try:  # the previous reconstruction, as committed in gen_yang2023.py's output
        old = json.loads(subprocess.run(["git", "-C", ROOT, "show", "c91ad29:cases/data/%s.json" % CID],
                                        capture_output=True, text=True, check=True).stdout)
    except Exception:
        old = None
    fit = compare(PARAMS, old if "--before" in sys.argv else None)
    if "--before" not in sys.argv:
        fit.update(BEFORE)
    print("fit", json.dumps(fit))
    case = build(PARAMS, fit)
    case["checks"] = checks(PARAMS)
    with open(os.path.join(ROOT, "cases", "data", CID + ".json"), "w") as fh:
        json.dump({k: v for k, v in case.items() if k != "checks"}, fh)
    print(CID, "branches", len(case["expected"]["branches"]), "singular", len(case["expected"]["singular_points"]))
    print(json.dumps(case["checks"]))


if __name__ == "__main__":
    main()
