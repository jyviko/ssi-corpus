"""Table 6 example (3) of Yang, Jia & Yan 2023 ("two intersect curves", bi-degree (2,2)),
refitted to the paper's panels.

What the panels show (cases/y23_fit/paper/y23_t6_ex3_two_crossing_curves_*.png, 600 dpi):
  * mirror-symmetric left/right and (almost exactly) top/bottom about the crossing
    point, which sits in the middle of the panel;
  * blue: a vertical band, 173 px wide x 359 px tall, straight vertical sides, top and
    bottom edges bowed outward by ~12 px with sharp corners;
  * orange: a wide barrel, 359 px wide x ~250 px tall; left/right sides bowed outward by
    ~14 px; its top/bottom edges drop steeply toward the corners (chamfer-like, 47 px over
    the outer 90 px), flat in the middle where they pass behind the blue band;
  * the AA-GPU panel (same view, opaque) shows the blue in front only inside the two
    vertical wedges of the X, the orange in front left and right of it: the surfaces
    touch at the crossing point and the blue recedes faster sideways;
  * red X: two open curves crossing once in the middle, slope dy/dx ~2.0 at the crossing
    and ~1.5 near the ends (the X opens up), ending where the band leaves the orange.
  * The edge bows that every straight-looking edge shares are perspective: the fitted
    camera is a close perspective camera on the symmetry axis.

Construction (exact, rational bi-quadratic):
  * blue   = circular cylinder patch, radius 1, vertical axis through (0, 1, 0),
             angle |phi| <= PHI_B about -y, height |z| <= H_B;
  * orange = tri-axial ellipsoid  x^2/A^2 + (y - C)^2/C^2 + z^2/Z^2 = 1  (front point at the
             origin), lat/lon patch about the x axis: |x| <= L_O (|lat| <= asin(L_O/A)),
             |angle about x from -y| <= TH_O. Rational (2,2): affine image of a sphere patch.
  * both surfaces touch at the origin with normal -y (the crossing point).
  * With s = x^2, t = z^2 the two quadrics are linear in (s, t) for given depth y:
        s = y (2 - y),   s/A^2 + t/Z^2 = y (2C - y)/C^2,
    so with y = r^2 the two branches are, in closed form,
        r -> ( r sqrt(2 - r^2),  r^2,  +- r sqrt(tau(r^2)) ),
        tau(y) = Z^2 ((2C - y)/C^2 - (2 - y)/A^2),
    smooth through r = 0 where they cross with slopes dz/dx = +-sqrt(tau(0)/2).

Run: .venv/bin/python -I cases/y23_fit/y23_t6_ex3_two_crossing_curves.py [--fit]
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t6_ex123_common as C  # noqa: E402  (sets sys.path)
import gen_cases as gc  # noqa: E402
import gen_yang2023 as Y  # noqa: E402  (read-only helpers)
import render_match as rm  # noqa: E402
import ssi_geom as g  # noqa: E402

CID = "y23_t6_ex3_two_crossing_curves"
STEM = CID

# fitted shape (blue radius 1 is the unit)
P = dict(H_B=1.025, PHI_B=31.7, A=1.417, CO=0.4947, Z=1.481, L_O=1.131, TH_O=32.2)
# fitted camera of the Surfaces panel (render_match convention, perspective)
CAM = dict(proj="persp", el=-1.53, az=-90.0, roll=0.0, target=[0.0, 0.0, 0.0], scale=311.1, dist=1.7726,
           cx=247.9, cy=204.8)
# camera fitted to the paper's 'Ours' result panel (curves only)
RESULT_CAM = None
# baseline: the previous construction (commit c91ad29) scored the same way with its own best-fitting camera
BEFORE = dict(construction="two unit cylinders (axes z and x), angle ranges 90 deg, length 1.6",
              camera=dict(proj="persp", el=-0.736, az=91.74, roll=1.483, scale=213.7, dist=1.948, cx=250.2,
                          cy=208.5, target=[0.0, 0.0, 0.0]),
              scores=dict(iou_silhouette=0.9161, iou_blue=0.6478, iou_orange=0.7681, chamfer_silhouette_px=7.34,
                          chamfer_blue_px=22.74))


def blue_surface(p=P):
    fr = g.Frame((0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0))  # z axis = +z, angle 0 towards -y
    cy = g.nurbs_cylinder(fr, 1.0, (-p["PHI_B"], p["PHI_B"]), (-p["H_B"], p["H_B"]),
                          "blue: circular cylinder patch r=1 (vertical axis)")
    return Y.elevate_v(cy, 1)


def orange_surface(p=P):
    A, Co, Z = p["A"], p["CO"], p["Z"]
    lat = math.degrees(math.asin(p["L_O"] / A))
    sph = g.nurbs_sphere(g.Frame(), 1.0, (-p["TH_O"], p["TH_O"]), (-lat, lat), "unit sphere patch")
    M = np.array([[0.0, 0.0, A], [-Co, 0.0, 0.0], [0.0, Z, 0.0]])  # unit-sphere local -> world
    t = np.array([0.0, Co, 0.0])
    net = sph.net.copy()
    w = net[:, :, 3:4]
    Pw = (net[:, :, :3] / w) @ M.T + t
    net[:, :, :3] = Pw * w
    fr = g.Frame((0.0, Co, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    truth = g.ellipsoid_truth(fr, A, Co, Z)
    th_o = p["TH_O"]

    def inside(Q, tol):
        Lx, Ly, Lz = 1.0 - Q[:, 1] / Co, Q[:, 2] / Z, Q[:, 0] / A
        la = np.degrees(np.arcsin(np.clip(Lz, -1, 1)))
        th = np.degrees(np.arctan2(Ly, Lx))
        return (np.abs(la) <= lat + 1e-7) & (np.abs(th) <= th_o + 1e-7)
    return g.Nurbs(sph.p, sph.q, sph.U, sph.V, net, truth, inside,
                   "orange: tri-axial ellipsoid patch (A, C, Z) = (%g, %g, %g)" % (A, Co, Z))


def tau(y, p=P):
    A, Co, Z = p["A"], p["CO"], p["Z"]
    return Z * Z * ((2 * Co - y) / Co ** 2 - (2 - y) / A ** 2)


def branch(sign, p=P):
    def f(r):
        r = np.asarray(r, float)
        y = r * r
        return np.column_stack([r * np.sqrt(2 - y), y, sign * r * np.sqrt(np.maximum(tau(y, p), 0.0))])
    return f


def r_max(p=P):
    A, Co = p["A"], p["CO"]
    y_star = 2 * Co * (A * A - Co) / (A * A - Co * Co)  # tau(y*) = 0
    return math.sqrt(0.98 * min(y_star, 2 * Co, 2.0))


def grid_fns(p=P):
    """Point-set grids (lat/lon parameterisation of the same patches) for the fit."""
    R = np.radians
    lat_o = math.asin(p["L_O"] / p["A"])

    def blue(n):
        f, z = np.meshgrid(np.linspace(-R(p["PHI_B"]), R(p["PHI_B"]), n), np.linspace(-p["H_B"], p["H_B"], n),
                           indexing="ij")
        return np.stack([np.sin(f), 1 - np.cos(f), z], -1)

    def orange(n):
        la, th = np.meshgrid(np.linspace(-lat_o, lat_o, n), np.linspace(-R(p["TH_O"]), R(p["TH_O"]), n),
                             indexing="ij")
        return np.stack([p["A"] * np.sin(la), p["CO"] * (1 - np.cos(la) * np.cos(th)),
                         p["Z"] * np.cos(la) * np.sin(th)], -1)
    return [blue, orange]


KEYS = ["H_B", "PHI_B", "A", "CO", "Z", "L_O", "TH_O"]


def fit():
    T = C.Target(C.paper_panel(STEM))
    cam0 = CAM

    def unpack(x):
        p = dict(zip(KEYS, x[:7]))
        c = dict(cam0, scale=math.exp(x[7]), cx=x[8], cy=x[9], dist=math.exp(x[10]), el=x[11])
        return p, c

    def loss(x):
        p, c = unpack(x)
        if min(p.values()) <= 0.01 or p["L_O"] >= 0.99 * p["A"] or p["PHI_B"] > 89 or p["TH_O"] > 89:
            return 1e3
        if tau(0.0, p) <= 0:  # no transversal crossing
            return 1e3
        return T.loss(*C.front_labels(grid_fns(p), c, T.W, T.H))
    x0 = [P[k] for k in KEYS] + [math.log(CAM["scale"]), CAM["cx"], CAM["cy"], math.log(CAM["dist"]), CAM["el"]]
    steps = np.abs(np.array(x0)) * 0.05 + 0.02
    steps[8:10] = 2.0
    x = C.nelder_mead(loss, x0, steps, rounds=3)
    p, c = unpack(x)
    print("P =", {k: round(v, 4) for k, v in p.items()})
    print("CAM =", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})


def contact_checks(p=P):
    """Numbers for the report: tangency at the crossing, crossing angle, transversality on the branches."""
    blue, orange = blue_surface(p), orange_surface(p)
    nb = lambda Q: np.column_stack([Q[:, 0], Q[:, 1] - 1.0, np.zeros(len(Q))])  # cylinder gradient
    no = lambda Q: np.column_stack([Q[:, 0] / p["A"] ** 2, (Q[:, 1] - p["CO"]) / p["CO"] ** 2,
                                    Q[:, 2] / p["Z"] ** 2])

    def sinang(a, b):
        return np.linalg.norm(np.cross(a, b), axis=1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    O = np.zeros((1, 3))
    s0 = float(sinang(nb(O + [[0, 0, 0]]), no(O))[0])
    k = math.sqrt(tau(0.0, p) / 2.0)
    ang = math.degrees(2 * math.atan(k))
    r = np.linspace(-r_max(p), r_max(p), 2001)
    worst = 1.0
    for sg in (1, -1):
        Q = branch(sg, p)(r)
        m = blue.inside(Q) & orange.inside(Q) & (np.abs(r) > 0.05)
        worst = min(worst, float(sinang(nb(Q[m]), no(Q[m])).min()))
    on = max(float(g.truth_distance(s.truth(), np.vstack([branch(sg, p)(r) for sg in (1, -1)])).max())
             for s in (blue, orange))
    return dict(sin_normals_at_crossing=s0, crossing_slope_dz_dx=k, angle_between_branches_deg=ang,
                min_sin_normals_on_branches_away_from_crossing=worst, max_truth_distance=on)


def build():
    blue, orange = blue_surface(), orange_surface()
    R = r_max()
    Y.add(CID, "Table 6 ex.3 (2,2): two intersecting curves", blue, orange, Y.SRC + "Table 6 (example (3))",
          "'two intersect curves', bi-degree (2,2)",
          "bi-degree (2,2) (rational); topology: two open curves crossing once at a tangential contact point "
          "(the crossing), endpoints where the blue band leaves the orange; appearance fitted to the Surfaces "
          "panel (camera and IoU in 'fit'): vertical blue band with outward-bowed top/bottom, wide orange barrel "
          "with bowed sides and drooping top corners, blue in front only inside the X's vertical wedges, X "
          "slope ~2 at the crossing opening to ~1.5 at the ends",
          "the surface types: blue = circular cylinder patch, orange = tri-axial ellipsoid patch (both exact "
          "rational (2,2)); the paper's control nets are not published, so the shapes are fitted to its picture "
          "with a perspective camera recovered from the same picture",
          "paper: all methods found the X",
          curves=[(branch(1), -R, R, False, "crossing"), (branch(-1), -R, R, False, "crossing")],
          singular=[(0.0, 0.0, 0.0)])
    case = gc.CASES[-1]
    case["truth_method"] = ("closed form: with y = r^2 the branches are (r sqrt(2 - r^2), r^2, +-r sqrt(tau(r^2))), "
                            "tau(y) = Z^2((2C - y)/C^2 - (2 - y)/A^2), clipped to both patches")
    return case


def scores_for(specs_or_fns, cam):
    T = C.Target(C.paper_panel(STEM))
    fns = [f if callable(f) else C.spec_grid_fn(f) for f in specs_or_fns]
    return {k: round(float(v), 4) for k, v in T.scores(*C.front_labels(fns, cam, T.W, T.H)).items()}


def main():
    if "--fit" in sys.argv:
        fit()
        return
    case = build()
    curves = [b["points"] for b in case["expected"]["branches"]]
    after = scores_for(case["surfaces"], CAM)
    res_cam, res_loss = C.fit_result_camera(curves, C.paper_panel(STEM, "ours"))
    case["contact"] = contact_checks()
    case["fit"] = dict(panel="cases/y23_fit/paper/%s_surfaces.png" % STEM, camera=CAM, params=P, scores=after,
                       result_panel_camera=res_cam, result_panel_curve_distance_px=round(float(res_loss), 3),
                       before=BEFORE)
    C.write_case(case)
    C.save_compare(STEM, case["surfaces"], CAM, curves)
    C.save_truth(STEM, case["surfaces"], CAM, curves, result_cam=res_cam)
    print(CID, "branches", len(case["expected"]["branches"]), "after", after)
    print("contact", case["contact"])
    print("result camera", res_cam, "curve distance px", res_loss)


if __name__ == "__main__":
    main()
