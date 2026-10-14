"""Table 2 ex.6 of Yang, Jia & Yan 2023 ("a loop and a point", bi-degree (2,2)), refitted to the paper's figures.

What the paper shows (cases/y23_fit/paper/y23_t2_ex6_loop_and_point_*.png, 600 dpi):
  * Surfaces panel: the blue sheet is a bowl seen from above and to the side: a crescent between a shallow
    upper arc and a deep U, with sharp corners at the far left and the top right. The orange sheet is a
    pinched, twisted leaf: a pointed tip at the upper left, a narrow waist where it passes through the blue,
    and a wide rounded lower lobe lying in the bowl.
  * the loop sits at the waist; the isolated point is below it, where the lower lobe of the orange touches
    the bowl near its lower right edge.
  * result row: an upright leaf-shaped loop (pointed top) and the point below it.

Construction (exact), (u, w) shared parameters:
    blue   B(u, w) = (u, w, f(u, w)),  f bi-quadratic,               (u, w) in [bu0, bu1] x [bw0, bw1]
    orange O(u, w) = B(u, w) + D(u, w) N,  N a fixed unit vector,     (u, w) in [ou0, ou1] x [ow0, ow1]
    D(u, w) = mu [(1 + u^2) W^2 + u (b1 u + b0) W + c0 u^2],  W = l w   (bi-degree (2,2))
  with b1, b0, c0 chosen so that the discriminant of D in W is -u^2 (u - p)(u - q). If the blue is a graph
  along N (its projection along N is one-to-one, checked), O(u,w) = B(u',w') forces (u',w') = (u,w), so the
  intersection is exactly {D = 0}: a closed loop over u in [p, q] and the isolated double zero (0, 0), where
  D >= 0 has a minimum: the orange touches the blue tangentially there. Bi-degree (2,2) for both.

Run: .venv/bin/python -I cases/y23_fit/y23_t2_ex6_loop_and_point.py   (writes cases/data/<id>.json)
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

CID = "y23_t2_ex6_loop_and_point"

# fitted (fit_t2_ex6.py: bowl first against the panel's blue crescent, then the orange with the camera fixed)
PARAMS = dict(f=[[0.0, -0.08717, 0.62956], [0.39538, -0.11504, -0.97479], [-0.45212, -0.67868, 0.39642]],
              bu0=-0.15, bu1=1.04816, bw0=-0.98918, bw1=0.86797, ou0=-0.12856, ou1=1.04329,
              ow0=-0.57793, ow1=0.2219, p=0.67274, q=2.17091, l=-2.06265, kap=1.93516,
              du=-0.07856, dw=-0.54593, mu=0.30838, nth=37.41539, nph=17.80843)
CAMERA = dict(proj="ortho", el=50.0, az=180.0, roll=-1.36, target=[0.0, 0.0, 0.0], scale=205.61698, cx=236.34923, cy=338.8168)
RESULT_CAMERA = dict(proj="ortho", el=-19.84590, az=-5.81063, roll=-57.01018,
                     target=[0.6561966464383219, -0.2326244635908513, 0.14756012108703734], scale=1009.74126,
                     cx=209.27017, cy=230.26723)
ALPHAS = (0.8, 0.8)


def dcoef(p):
    pq, qq = p["p"], p["q"]
    b1sq = (-(1 - pq * qq) + math.sqrt((1 - pq * qq) ** 2 + 4 * ((pq + qq) / 2) ** 2)) / 2
    b1 = math.sqrt(b1sq)
    b0 = (pq + qq) / 2 / b1
    c0 = (b1 * b1 + 1) / 4
    return b1, b0, c0


def Nvec(p):
    th, ph = math.radians(p["nth"]), math.radians(p["nph"])
    return np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])


def fval(p, u, w):
    F = np.asarray(p["f"], float)
    return sum(F[i, j] * u ** i * w ** j for i in range(3) for j in range(3))


def Dval(p, u, w):
    b1, b0, c0 = dcoef(p)
    U = p["kap"] * (u - p["du"])
    W = p["l"] * (w - p["dw"])
    return p["mu"] * ((1 + U * U) * W * W + U * (b1 * U + b0) * W + c0 * U * U)


def blue_pt(p, u, w):
    u, w = np.broadcast_arrays(np.asarray(u, float), np.asarray(w, float))
    return np.stack([u, w, fval(p, u, w)], -1)


def orange_pt(p, u, w):
    u, w = np.broadcast_arrays(np.asarray(u, float), np.asarray(w, float))
    return blue_pt(p, u, w) + Dval(p, u, w)[..., None] * Nvec(p)


def grids(p, n=61):
    bu = np.linspace(p["bu0"], p["bu1"], n)[:, None]
    bw = np.linspace(p["bw0"], p["bw1"], n)[None, :]
    ou = np.linspace(p["ou0"], p["ou1"], n)[:, None]
    ow = np.linspace(p["ow0"], p["ow1"], n)[None, :]
    return blue_pt(p, bu, bw), orange_pt(p, ou, ow)


def loop(p, n=2001):
    return loop_fn(p)(np.linspace(0, 2 * np.pi, n))


def point(p):
    return blue_pt(p, p["du"], p["dw"])


def graph_along_N_ok(p, n=161):
    """The blue's projection along N is one-to-one: the Jacobian of (u,w) -> projection keeps its sign."""
    N = Nvec(p)
    e1 = np.cross(N, [0, 0, 1.0]) if abs(N[2]) < 0.999 else np.array([1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(N, e1)
    u = np.linspace(p["bu0"], p["bu1"], n)[:, None]
    w = np.linspace(p["bw0"], p["bw1"], n)[None, :]
    h = 1e-6
    Pu = (blue_pt(p, u + h, w) - blue_pt(p, u - h, w)) / (2 * h)
    Pw = (blue_pt(p, u, w + h) - blue_pt(p, u, w - h)) / (2 * h)
    J = (Pu @ e1) * (Pw @ e2) - (Pu @ e2) * (Pw @ e1)
    return float(J.min()) * float(J.max()) > 0 and float(np.abs(J).min()) > 1e-3


def valid(p):
    return bool(p["bu0"] <= p["ou0"] < p["ou1"] <= p["bu1"] and p["bw0"] <= p["ow0"] < p["ow1"] <= p["bw1"]
                and 0 < p["p"] < p["q"] and abs(p["l"]) > 1e-3 and abs(p["kap"]) > 1e-3 and p["mu"] > 0
                and inside_orange(p, point(p)[None, :], 0.03).all() and inside_orange(p, loop(p, 401)).all()
                and graph_along_N_ok(p, 41))


def inside_orange(p, P, m=0.0):
    return ((P[:, 0] > p["ou0"] + m) & (P[:, 0] < p["ou1"] - m) & (P[:, 1] > p["ow0"] + m)
            & (P[:, 1] < p["ow1"] - m))


# ----------------------------------------------------------------------------------- surfaces / case

def D_power(p):
    """Power coefficients of D in (u, w) (rows u^i, cols w^j): D(U, W) composed with the affine U(u), W(w)."""
    b1, b0, c0 = dcoef(p)
    mu = p["mu"]
    Duw = np.zeros((3, 3))  # in (U, W)
    Duw[0, 2] = Duw[2, 2] = mu
    Duw[2, 1] = mu * b1
    Duw[1, 1] = mu * b0
    Duw[2, 0] = mu * c0
    Uu = Y.P([-p["kap"] * p["du"]], [p["kap"]])
    Ww = Y.P([-p["l"] * p["dw"], p["l"]])
    out = np.zeros((3, 3))
    c = Y.compose(Duw, Uu, Ww)
    out[:c.shape[0], :c.shape[1]] = c
    return out


def surfaces(p=PARAMS):
    F = np.asarray(p["f"], float)
    X = np.zeros((3, 3))
    X[1, 0] = 1.0
    Yc = np.zeros((3, 3))
    Yc[0, 1] = 1.0
    Dm = D_power(p)
    N = Nvec(p)
    one = Y.P([1.0])
    blue = g.rational_poly_patch(X, Yc, F, one, (p["bu0"], p["bu1"]), (p["bw0"], p["bw1"]),
                                 "bowl: graph z = f(u, w), f bi-quadratic", degree=(2, 2))
    orange = g.rational_poly_patch(X + N[0] * Dm, Yc + N[1] * Dm, F + N[2] * Dm, one, (p["ou0"], p["ou1"]),
                                   (p["ow0"], p["ow1"]), "pinched sheet: bowl + D(u,w) N", degree=(2, 2))
    return blue, orange


def checks(p=PARAMS):
    out = {"blue_graph_along_N": graph_along_N_ok(p, 201)}
    b1, b0, c0 = dcoef(p)
    out["hessian_D_at_point_det_over_mu2"] = float((4 * c0 - b0 * b0) * (p["l"] * p["kap"]) ** 2)
    h = 1e-6
    u0, w0 = p["du"], p["dw"]
    nb = np.cross(blue_pt(p, u0 + h, w0) - blue_pt(p, u0 - h, w0), blue_pt(p, u0, w0 + h) - blue_pt(p, u0, w0 - h))
    no = np.cross(orange_pt(p, u0 + h, w0) - orange_pt(p, u0 - h, w0), orange_pt(p, u0, w0 + h) - orange_pt(p, u0, w0 - h))
    out["sin_normals_at_point"] = float(np.linalg.norm(np.cross(nb, no)) / np.linalg.norm(nb) / np.linalg.norm(no))
    L = loop(p, 2001)
    out["max_D_on_loop"] = float(np.abs(Dval(p, L[:, 0], L[:, 1])).max())
    return out


def loop_fn(p):
    def fn(t):
        b1, b0, c0 = dcoef(p)
        t = np.asarray(t, float)
        U = (p["p"] + p["q"]) / 2 + (p["q"] - p["p"]) / 2 * np.cos(t)
        a = 1 + U * U
        bb = U * (b1 * U + b0)
        # disc = -U^2 (U - p)(U - q) exactly
        disc = np.clip(-U * U * (U - p["p"]) * (U - p["q"]), 0, None)
        W = (-bb + np.sign(np.sin(t)) * np.sqrt(disc)) / (2 * a)
        return blue_pt(p, p["du"] + U / p["kap"], p["dw"] + W / p["l"])
    return fn


# IoU of the previous reconstruction (commit c91ad29) from its own best-fitting camera (run with --before)
BEFORE = dict(before_iou_union=0.7852, before_iou_blue=0.6407, before_iou_orange=0.4112)


def result_chamfer(p, rcam, RES=None):
    sys.path.insert(0, HERE)
    import render_match as rm
    from scipy import ndimage
    import fit_tools_t2_ex456 as T
    RES = T.paper(CID, "result") if RES is None else RES
    H, W = RES.shape[:2]
    _, _, red = rm.color_masks(RES)
    P = np.vstack([loop(p, 600), np.repeat(point(p)[None, :], 30, 0)])
    x, y, _ = rm.project(P, rcam, W, H)
    ins = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    xi, yi = np.clip(np.round(x).astype(int), 0, W - 1), np.clip(np.round(y).astype(int), 0, H - 1)
    d1 = np.where(ins, ndimage.distance_transform_edt(~red)[yi, xi], 60.0).mean()
    m = np.zeros((H, W), bool)
    m[yi[ins], xi[ins]] = True
    if not m.any():
        return 60.0
    ry, rx = np.nonzero(red)
    d2 = ndimage.distance_transform_edt(~m)[ry, rx].mean()
    return float(0.5 * (d1 + d2))


def fit_result_camera(p, r0=RESULT_CAMERA):
    from scipy import optimize
    keys = ["el", "az", "roll", "scale", "cx", "cy"]
    c0 = 0.5 * (loop(p, 200).min(0) + loop(p, 200).max(0))
    best = None
    for el in (10.0, 40.0, 70.0, -40.0):
        for az in (-150.0, -90.0, -30.0, 30.0, 90.0, 150.0):
            for roll in (0.0, 90.0, -90.0):
                def f(v):
                    c = dict(r0, target=c0.tolist())
                    c.update(zip(keys, v))
                    return result_chamfer(p, c)
                r = optimize.minimize(f, [el, az, roll, 300.0, 150.0, 200.0], method="Nelder-Mead",
                                      options=dict(maxiter=300, adaptive=True))
                if best is None or r.fun < best.fun:
                    best = r
    c = dict(r0, target=c0.tolist())
    c.update(zip(keys, best.x))
    return c, float(best.fun)


def build(p=PARAMS, fit=None, rcam=RESULT_CAMERA):
    blue, orange = surfaces(p)
    fit = fit or {}
    cam_txt = lambda c: "%s el=%.2f az=%.2f roll=%.2f" % (c["proj"], c["el"], c["az"], c["roll"])
    Y.add(CID, "Table 2 ex.6 (2,2): a loop and a point (bowl and a sheet rising out of it)", blue, orange,
          Y.SRC + "Table 2 (example 6)", "'a loop and a point', bi-degree (2,2)",
          "bi-degree (2,2), polynomial, one Bezier patch each; topology: one closed transversal loop + one isolated "
          "tangential point (normals parallel, D has a positive-definite double zero), the point below the loop near "
          "the lower tip of the orange; appearance fitted (render_match.py convention): Surfaces-panel camera %s, "
          "silhouette IoU union %s / blue %s / orange %s; result-row camera %s, mean chamfer %s px to the paper's "
          "loop and dot" % (cam_txt(CAMERA), fit.get("iou_union", "?"), fit.get("iou_blue", "?"),
                            fit.get("iou_orange", "?"), cam_txt(rcam), fit.get("chamfer_result_px", "?")),
          "construction (orange = bowl + D N, D with the loop/point discriminant) and all coefficients are fitted, "
          "not published; the bowl matches the panel's blue crescent, but the orange is only approximate: its upper "
          "part rises as two horns (the paper: one pointed tip), the lower lobe is narrower than drawn, and the "
          "loop is smooth (the paper's has a pointed top, which a smooth pair of patches cannot produce without a "
          "further singular point); in the result-row view the point lies about 4x farther below the loop (relative to the loop's size) than drawn",
          "paper: " + Y.OPEN + ", " + Y.SPL + " and " + Y.COMM + " returned the loop only (lost the point)",
          curves=[(loop_fn(p), 0.0, 2 * np.pi, True, "crossing")], points=[point(p)], cov_tol=1e-2,
          notes="B(u,w) = (u, w, f(u,w)), f = sum F[i][j] u^i w^j, F = %s; O = B + D N, N = (%.6f, %.6f, %.6f); "
                "D = mu[(1+U^2)W^2 + U(b1 U + b0)W + c0 U^2], U = kap(u - du), W = l(w - dw), discriminant "
                "-U^2(U - p)(U - q); params %s"
                % (json.dumps(p["f"]), *Nvec(p), json.dumps({k: v for k, v in p.items() if k != "f"})))
    case = gc.CASES[-1]
    case["truth_method"] = ("closed form: loop = {D = 0} (W from a quadratic in closed form), isolated point = the "
                            "double zero of D; the blue is a graph along N, so O(u,w) = B(u',w') only for "
                            "(u',w') = (u,w)")
    return case


def compare(p=PARAMS, old_case=None, rcam=RESULT_CAMERA):
    sys.path.insert(0, HERE)
    import render_match as rm
    import fit_tools_t2_ex456 as T
    out = {}
    Ps, Pa, Pr = (T.paper(CID, k) for k in ("surfaces", "aagpu", "result"))
    A, B = grids(p, 121)
    H, W = Ps.shape[:2]
    cur = [loop(p, 1200)]
    pts = [point(p)]
    rgb_s, m_s = rm.render([A, B], CAMERA, W, H, style="surfaces", alphas=ALPHAS, n=121, curves=cur, points=pts,
                           point_radius=3.0)
    out["iou_union"], out["iou_blue"], out["iou_orange"] = T.ious(rm, Ps, rgb_s, m_s)
    cam_a, ia = T.fit_framing(rm, [A, B], CAMERA, Pa)
    out["iou_aagpu_union"] = round(ia, 4)
    rgb_a, _ = rm.render([A, B], cam_a, Pa.shape[1], Pa.shape[0], style="surfaces", alphas=(1.0, 1.0), n=121,
                         colors=(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])))
    rgb_r = T.curves_only(rm, cur, rcam, Pr.shape[1], Pr.shape[0], points=pts, width=7, radius=4)
    out["chamfer_result_px"] = round(result_chamfer(p, rcam, Pr), 2)
    rm.compare_strip([Ps, Pa, Pr], [rgb_s, rgb_a, rgb_r], os.path.join(HERE, "compare", CID + ".png"))
    rgb_t, _ = rm.render([A, B], CAMERA, W, H, style="result", curves=cur, points=pts, n=121, point_radius=3.0)
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
    assert valid(PARAMS), "construction not exact (graph along N / domains)"
    old = None
    if "--before" in sys.argv:
        old = json.loads(subprocess.run(["git", "-C", ROOT, "show", "c91ad29:cases/data/%s.json" % CID],
                                        capture_output=True, text=True, check=True).stdout)
    rcam = RESULT_CAMERA
    if "--fit-result" in sys.argv:
        rcam, ch = fit_result_camera(PARAMS)
        print("result camera", json.dumps(rcam), ch)
    fit = compare(PARAMS, old, rcam)
    if old is None:
        fit.update(BEFORE)
    print("fit", json.dumps(fit))
    case = build(PARAMS, fit, rcam)
    print(json.dumps(checks(PARAMS)))
    with open(os.path.join(ROOT, "cases", "data", CID + ".json"), "w") as fh:
        json.dump(case, fh)
    print(CID, "branches", len(case["expected"]["branches"]), "points", len(case["expected"]["isolated_points"]))


if __name__ == "__main__":
    main()
