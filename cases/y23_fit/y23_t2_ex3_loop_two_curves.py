"""Table 2 example 3 of Yang, Jia & Yan 2023 ("a loop and two curves", bi-degree (3,3)), refitted to the paper's
Surfaces panel (cases/y23_fit/paper/y23_t2_ex3_loop_two_curves_*.png, cropped from p.10 at 600 dpi).

What the panel shows:
  * blue: a long, narrow, S-curved channel seen from its convex side, so it reads as a tube (the open
    cross-section is visible only at the two cut ends).
  * orange: a shorter strip along the middle of the channel whose curved flanks stand out of it.
  * red: one long curve visible along the orange (the second long curve lies behind it in this view) and a
    small loop seen almost edge-on as a short bar that crosses the long curve.
  * the "Ours" result panel, from another view, shows the two long curves as an elongated lens open at both
    ends and the small loop crossing both: 4 crossings, each long curve running through the loop and on to
    the orange's ends.

Construction (exact, bi-degree (3,3) for both surfaces):
  * blue  B(u, w) = (u, w + p(u), h(u) + e(u) w + k w^2), a parabolic channel along the S path (p, h cubic,
    e linear), u in [-1, 1], w in [-Wb, Wb]. Its implicit form is Z = h(X) + e(X) V + k V^2, V = Y - p(X).
  * orange O(s, t) = (s, p(s) + tau, h(s) + e(s) tau + k tau^2 + p0 L M), s in [-So, So], t in [-To, To],
    with tau = g s^2 + beta t + alpha t^2 (monotone in t on the domain),
      M = t^2 + mu s t + c2 s^2 - c0^2                (long ellipse; the domain cuts it into two long curves),
      L = (s - s0)^2 / a^2 + lam (s - s0) t + t^2 / b^2 - 1   (small loop crossing both),
      p0 = -k alpha^2 b^2,  g = sqrt(-p0 c2 / (k a^2)),
    chosen so that the t^4 and s^4 terms of k tau^2 and p0 L M cancel: O is bicubic.
  * Substituting O into the blue's implicit form leaves exactly p0 L(s, t) M(s, t), so the intersection is
    {L = 0} (closed loop) union {M = 0} (two open curves), in closed form. The 4 points L = M = 0 are double
    points of the intersection; there the gradient of p0 L M vanishes, so the surfaces are tangent
    (normals parallel) and the loop is reachable from the long curves only through these points.
  * Shape and camera were fitted together (Nelder-Mead on silhouette IoU + visible-orange IoU + red-curve
    chamfer against the panel, t2_ex123_fit.py), with constraints that keep the loop and both long curves
    inside both patches.

Run: .venv/bin/python -I cases/y23_fit/y23_t2_ex3_loop_two_curves.py [--refit]
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
import t2_ex123_fit as F  # noqa: E402

CID = "y23_t2_ex3_loop_two_curves"

PARAMS = {'p1': -0.06349544008693278, 'p2': -0.005187840235238092, 'p3': 0.12372584871436335,
          'h1': 0.5588955378440115, 'h2': 0.014822221590079489, 'h3': -0.42006235350874044,
          'e0': 0.04341867160298692, 'e1': -0.17564225364451447, 'k': 5.945394407466901,
          'Wb': 0.16175196043093076, 'beta': 1.0162438017406517, 'alpha': 0.5211103959322387,
          'c0': 0.13437204842136033, 'mu': 0.008093527190119, 'a': 0.057310049438214626,
          'b': 0.14219606691601416, 'lam': 0.05274807965981358, 'So': 0.5191397559150728,
          'To': 0.18500992923340792, 's0': -0.023838901285450704}
CAMERA = {'proj': 'ortho', 'target': [0.0, 0.0, 0.0], 'el': -18.31218210063834, 'az': -88.3470827553607,
          'roll': 0.06188432119410992, 'scale': 231.09216280540832, 'cx': 278.7890380066111,
          'cy': 201.63987359448316}
IOU_BEFORE = {'iou_silhouette': 0.3289, 'iou_visible_orange': 0.1286, 'curve_chamfer_px': 4.36}  # old case, own best camera

NAMES = list(PARAMS)
# the long ellipse M = 0 reaches s = +-c0/sqrt(c2); the orange ends just inside it (So = RATIO c0/sqrt(c2)), so the
# two long curves nearly meet at both ends of the strip, as in the paper's result panel
RATIO = 0.97


def full(p):
    q = dict(p)
    q['c2'] = (RATIO * q['c0'] / q['So']) ** 2
    return q


def inside_checks(p, Q):
    """Loop and both long curves inside both patches (t within the orange, tau within the channel width)."""
    th = np.linspace(0, 2 * np.pi, 2001)
    sl, tl = loop_st(p, th)
    s = np.linspace(-p['So'], p['So'], 2001)
    for ss, tt in [(sl, tl)] + [(s, long_t(p, s, sg)) for sg in (1, -1)]:
        if not (np.abs(ss).max() <= p['So'] + 1e-12 and np.abs(tt).max() < p['To']
                and np.abs(g.p2_eval(Q['tau'], ss, tt)).max() < p['Wb']):
            raise ValueError("a curve leaves a patch")
    if p['beta'] - 2 * p['alpha'] * p['To'] <= 0.05:
        raise ValueError("tau not monotone")
    if p['lam'] ** 2 >= 4 / (p['a'] * p['b']) ** 2 or p['mu'] ** 2 >= 4 * p['c2']:
        raise ValueError("not an ellipse")


def derived(p):
    p0 = -p['k'] * p['alpha'] ** 2 * p['b'] ** 2
    gam = math.sqrt(-p0 * p['c2'] / (p['k'] * p['a'] ** 2))
    return p0, gam


def polys(p):
    """Power coefficients (rows s^i / u^i, columns t^j / w^j) of both surfaces and of L, M."""
    P, mono, poly, mul = Y.P, Y.mono, Y.poly, Y.mul
    p0, gam = derived(p)
    pp = [0.0, p['p1'], p['p2'], p['p3']]
    hh = P([0.0], [p['h1']], [p['h2']], [p['h3']])
    ee = P([p['e0']], [p['e1']])
    Hb = poly(hh, mul(ee, mono(0, 1)), mono(0, 2, p['k']))
    tau = poly(mono(2, 0, gam), mono(0, 1, p['beta']), mono(0, 2, p['alpha']))
    s0, a, b = p['s0'], p['a'], p['b']
    ds = P([-s0], [1.0])
    L = poly(mul(ds, ds, P([1 / a ** 2])), mul(ds, mono(0, 1, p['lam'])), mono(0, 2, 1 / b ** 2), P([-1.0]))
    M = poly(mono(0, 2), mono(1, 1, p['mu']), mono(2, 0, p['c2']), P([-p['c0'] ** 2]))
    Hx = P([0.0], [1.0])
    Hy = poly(P(*[[c] for c in pp]), tau)
    Hz = poly(hh, mul(ee, tau), g.p2_scale(mul(tau, tau), p['k']), g.p2_scale(mul(L, M), p0))
    # the t^4 and s^4 terms cancel by the choice of p0 and g; drop their rounding residue
    big = np.abs(Hz).max()
    Hz = np.array(Hz, float)
    assert np.abs(Hz[4:, :]).max(initial=0) < 1e-12 * big and np.abs(Hz[:, 4:]).max(initial=0) < 1e-12 * big
    Hz = g.p2_trim(np.where(np.abs(Hz) < 1e-13 * big, 0.0, Hz)[:4, :4])
    return dict(Hb=Hb, shear=pp, Hx=Hx, Hy=Hy, Hz=Hz, L=L, M=M, tau=tau, p0=p0, gam=gam)


def opoint(Q, s, t):
    s, t = np.asarray(s, float), np.asarray(t, float)
    return np.stack([g.p2_eval(Q['Hx'], s, t), g.p2_eval(Q['Hy'], s, t), g.p2_eval(Q['Hz'], s, t)], -1)


def loop_st(p, th):
    """The ellipse L = 0 in (s, t), parametrised by angle th (closed form)."""
    A, B, C = 1 / p['a'] ** 2, p['lam'], 1 / p['b'] ** 2
    M2 = np.array([[A, B / 2], [B / 2, C]])
    w, V = np.linalg.eigh(M2)
    Q = (V / np.sqrt(w)) @ np.stack([np.cos(th), np.sin(th)])
    return p['s0'] + Q[0], Q[1]


def long_t(p, s, sign):
    """t on M = 0 for given s: t^2 + mu s t + (c2 s^2 - c0^2) = 0 (closed form)."""
    disc = (p['mu'] * s) ** 2 - 4 * (p['c2'] * s * s - p['c0'] ** 2)
    return (-p['mu'] * s + sign * np.sqrt(disc)) / 2


def crossings(p, Q):
    """The 4 points L = M = 0: sign changes of M along the loop, refined by Newton on (s, t)."""
    th = np.linspace(0, 2 * np.pi, 4001)
    s, t = loop_st(p, th)
    Mv = g.p2_eval(Q['M'], s, t)
    idx = np.nonzero(np.sign(Mv[:-1]) != np.sign(Mv[1:]))[0]
    out = []
    dL = [(np.arange(Q['L'].shape[0])[:, None] * Q['L'])[1:], (np.arange(Q['L'].shape[1])[None, :] * Q['L'])[:, 1:]]
    dM = [(np.arange(Q['M'].shape[0])[:, None] * Q['M'])[1:], (np.arange(Q['M'].shape[1])[None, :] * Q['M'])[:, 1:]]
    for i in idx:
        x = np.array([s[i], t[i]])
        for _ in range(50):
            F_ = np.array([g.p2_eval(Q['L'], *x), g.p2_eval(Q['M'], *x)])
            J = np.array([[g.p2_eval(dL[0], *x), g.p2_eval(dL[1], *x)], [g.p2_eval(dM[0], *x), g.p2_eval(dM[1], *x)]])
            x = x - np.linalg.solve(J, F_)
        out.append(x)
    return np.array(out)


def normals_sin(p, Q, st):
    """sin of the angle between the two surface normals at orange parameters st (analytic derivatives)."""
    d_s = lambda C: (np.arange(C.shape[0])[:, None] * C)[1:]
    d_t = lambda C: (np.arange(C.shape[1])[None, :] * C)[:, 1:]
    s, t = st[:, 0], st[:, 1]
    Os = np.stack([g.p2_eval(d_s(Q[k]), s, t) for k in ('Hx', 'Hy', 'Hz')], -1)
    Ot = np.stack([g.p2_eval(d_t(Q[k]), s, t) for k in ('Hx', 'Hy', 'Hz')], -1)
    no = np.cross(Os, Ot)
    u, w = s, g.p2_eval(Q['tau'], s, t)
    pd = np.polyval(np.array(Q['shear'])[::-1][:-1] * np.arange(len(Q['shear']) - 1, 0, -1), u)
    Bu = np.stack([np.ones_like(u), pd, g.p2_eval(d_s(Q['Hb']), u, w)], -1)
    Bw = np.stack([0 * u, np.ones_like(u), g.p2_eval(d_t(Q['Hb']), u, w)], -1)
    nb = np.cross(Bu, Bw)
    no /= np.linalg.norm(no, axis=1)[:, None]
    nb /= np.linalg.norm(nb, axis=1)[:, None]
    return np.linalg.norm(np.cross(no, nb), axis=1)


def model(x, n=25):
    """Fit-time model (grids + curves) for t2_ex123_fit.fit."""
    p = full(dict(zip(NAMES, x)))
    for k in ('k', 'Wb', 'beta', 'alpha', 'c0', 'a', 'b', 'So', 'To'):
        if p[k] <= 0:
            raise ValueError(k)
    Q = polys(p)
    inside_checks(p, Q)
    gb = F.grid_from_fn(lambda u, w: g.sheared_point(Q['Hb'], Q['shear'], u.ravel(), w.ravel()).reshape(u.shape + (3,)),
                        (-1, 1), (-p['Wb'], p['Wb']), 2 * n, n // 2 + 1)
    go = F.grid_from_fn(lambda s, t: opoint(Q, s, t), (-p['So'], p['So']), (-p['To'], p['To']), n, n // 2 + 1)
    s = np.linspace(-p['So'], p['So'], 80)
    cv = [opoint(Q, *loop_st(p, np.linspace(0, 2 * np.pi, 120)))] + [opoint(Q, s, long_t(p, s, sg)) for sg in (1, -1)]
    return gb, go, cv


def build(p):
    Q = polys(p)
    blue = Y.graph_patch(Q['Hb'], (-1.0, 1.0), (-p['Wb'], p['Wb']), "S-curved parabolic channel", degree=(3, 3),
                         shear=Q['shear'])
    orange = g.rational_poly_patch(Q['Hx'], Q['Hy'], Q['Hz'], [[1.0]], (-p['So'], p['So']), (-p['To'], p['To']),
                                   "shorter strip lining the channel", degree=(3, 3))
    return blue, orange, Q


def main():
    if "--refit" in sys.argv:
        T = F.Target(os.path.join(F.PAPER, CID + "_surfaces.png"))
        cam, x, loss = F.fit(T, model, [PARAMS[k] for k in NAMES], CAMERA, steps=[3000, 3000],
                             weights=(1.0, 0.5, 0.04), log=print)
        print(repr(dict(zip(NAMES, map(float, x)))))
        print(repr(cam))
        return
    p = full(PARAMS)
    blue, orange, Q = build(p)
    assert blue.p == blue.q == orange.p == orange.q == 3
    inside_checks(p, Q)
    th = np.linspace(0, 2 * np.pi, 2001)
    sl, tl = loop_st(p, th)
    s = np.linspace(-p['So'], p['So'], 2001)
    X = crossings(p, Q)
    assert len(X) == 4, len(X)
    sing = opoint(Q, X[:, 0], X[:, 1])
    sin_sing = normals_sin(p, Q, X)
    # transversality of the long curves and the loop away from the 4 double points
    sin_curves = normals_sin(p, Q, np.column_stack([s, long_t(p, s, 1)]))
    sin_loop = normals_sin(p, Q, np.column_stack([sl, tl]))
    curves = [(lambda th, p=p, Q=Q: opoint(Q, *loop_st(p, th)), 0.0, 2 * np.pi, True, "crossing")]
    for sg in (1, -1):
        curves.append((lambda s, sg=sg, p=p, Q=Q: opoint(Q, s, long_t(p, s, sg)), -p['So'], p['So'], False,
                       "crossing"))
    cam_txt = "el %.2f, az %.2f, roll %.2f deg, orthographic, scale %.1f px/unit, target at (%.1f, %.1f) px" % (
        CAMERA['el'], CAMERA['az'], CAMERA['roll'], CAMERA['scale'], CAMERA['cx'], CAMERA['cy'])
    Y.add(CID, "Table 2 ex.3 (3,3): a small loop and two long curves", blue, orange, Y.SRC + "Table 2 (example 3)",
          "'a loop and two curves', bi-degree (3,3)",
          "bi-degree (3,3) (both surfaces, polynomial); topology: one small closed loop and two long open curves that "
          "each run the full length of the orange (boundary to boundary) and pass through the loop, crossing it "
          "twice: 4 double points where the surfaces are tangent (max sin %.1e); intersection exactly "
          "{L = 0} u {M = 0} in the orange's parameters; appearance: refitted to the Surfaces panel from a fitted "
          "camera (%s): long S-curved channel seen from its convex side and a shorter strip lining it whose flanks "
          "stand out; silhouette IoU against the panel {IOU} (before: %.3f)" % (sin_sing.max(), cam_txt,
                                                                                IOU_BEFORE['iou_silhouette']),
          "exact control points (the paper publishes none): fitted to one view; that the 4 crossings are true "
          "tangential double points (not projection overlaps) is read from the figure; the construction forces the "
          "orange to lie on the concave side of the channel between the long curves, and its ends flare (L grows "
          "like (s/a)^2 along the strip), which the paper's strip does not show; the loop is drawn taller than the "
          "paper's short bar; conditioning: the loop is shallow (sin between the normals along it: median %.1e, "
          "the long curves median %.1e), steeper constructions push the curves out of the channel width" % (
              np.median(sin_loop), np.median(np.concatenate([sin_curves, normals_sin(p, Q, np.column_stack(
                  [s, long_t(p, s, -1)]))]))),
          "paper: " + Y.OPEN + " and " + Y.COMM + " returned the two long curves and lost the small loop; " + Y.SPL +
          " returned nothing",
          curves=curves, singular=[tuple(map(float, q)) for q in sing], cov_tol=5e-3)
    case = gc.CASES[-1]
    assert len(case["expected"]["branches"]) == 3, [b["closed"] for b in case["expected"]["branches"]]
    out = F.deliver(CID, case, CAMERA)
    case["matched"] = case["matched"].replace("{IOU}", "%.3f" % out["iou_silhouette"])
    with open(os.path.join(ROOT, "cases", "data", CID + ".json"), "w") as fh:
        json.dump(case, fh)
    print(json.dumps(dict(out, sin_at_double_points=sin_sing.tolist(),
                          min_sin_long_curve=float(sin_curves.min()), min_sin_loop=float(sin_loop.min()),
                          singular_st=X.tolist(), iou_before=IOU_BEFORE, p0=Q['p0'], g=Q['gam']), indent=1))


if __name__ == "__main__":
    main()
