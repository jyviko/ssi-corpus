"""Table 6 example (2) of Yang, Jia & Yan 2023 ("a tangential curve", bi-degree (2,2)),
refitted to the paper's panels.

What the panels show (cases/y23_fit/paper/y23_t6_ex2_tangential_arc_*.png, 600 dpi):
  * blue: a four-sided, gently domed patch seen slightly from below: top edge an arch
    (peak near x = 290), bottom edge a flatter arch, left edge bulging slightly
    outward, right edge nearly straight and a little concave, sharp top-right corner;
  * orange: behind the blue everywhere (AA-GPU panel: only blue is visible except a
    thin orange sliver beyond the blue's right edge, as two small horns above and below
    the red curve); through the translucent blue its outline is a dome: arched top,
    nearly straight bottom, left edge close to the blue's left edge;
  * red: one open arch from the blue's left edge (110, 194) to its right edge
    (425, 197), peak 32 px higher near the middle: the two surfaces touch along it.

Construction (exact, rational bi-quadratic):
  * two quadrics of revolution about the z axis, r(z)^2 = 1 - k z^2 with r = sqrt(x^2+y^2):
        blue   k = KB   (prolate spheroid, semi-axes 1, 1, 1/sqrt(KB)),
        orange k = KO   (KO > KB: oblate spheroid, semi-axes 1, 1, 1/sqrt(KO));
    both contain the unit circle z = 0 and have the same (radial) normal there.
    For z != 0, r_blue(z) > r_orange(z): the orange lies strictly inside the blue, so
    the surfaces meet only on z = 0 and touch there (even-order contact).
  * patches: longitude ranges and height ranges (blue [LB0, LB1] x [ZB0, ZB1],
    orange [LO0, LO1] x [ZO0, ZO1]); each is an affine image of a sphere lat/lon patch,
    so both are exact rational (2,2) NURBS.
  * truth: the arc of the unit circle z = 0 over the common longitude range (closed form).

Run: .venv/bin/python -I cases/y23_fit/y23_t6_ex2_tangential_arc.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t6_ex123_common as C  # noqa: E402  (sets sys.path)
import gen_cases as gc  # noqa: E402
import gen_yang2023 as Y  # noqa: E402  (read-only helpers)
import ssi_geom as g  # noqa: E402

CID = "y23_t6_ex2_tangential_arc"
STEM = CID

# fitted shape (unit tangent circle)
P = dict(KB=0.1755, KO=1.2125, LB0=-55.41, LB1=57.45, ZB0=-0.9423, ZB1=0.6334, LO0=-55.24, LO1=57.92,
         ZO0=-0.6575, ZO1=0.3946)
# fitted camera of the Surfaces panel (render_match convention)
CAM = dict(proj="persp", el=-13.51, az=6.22, roll=3.96, target=[0.0, 0.0, 0.0], scale=2038.6, dist=10.95,
           cx=281.79, cy=215.0)
# baseline: the previous construction (commit c91ad29) scored the same way with its own best-fitting camera
BEFORE = dict(construction="unit sphere patch (blue, lon +-60, lat +-40) inside a unit cylinder patch (orange, "
                           "+-70 deg, z in [-0.8, 0.8])",
              camera=dict(proj="persp", el=-0.9, az=0.48, roll=5.93, scale=240.5, dist=1.849, cx=269.96, cy=226.23,
                          target=[0.0, 0.0, 0.0]),
              scores=dict(iou_silhouette=0.7756, iou_blue=0.0123, iou_orange=0.0053, chamfer_silhouette_px=24.72,
                          chamfer_blue_px=87.53),
              note="the old blue sphere lies inside the old orange cylinder, so from any outside view the orange "
                   "hides it; the best camera for the silhouette leaves almost no blue visible")


def spheroid_patch(k, lon, zr, label):
    """Patch of r^2 + k z^2 = 1 (semi-axes 1, 1, c = 1/sqrt(k)) over longitude lon and height zr."""
    c = 1.0 / math.sqrt(k)
    lat = (math.degrees(math.asin(zr[0] / c)), math.degrees(math.asin(zr[1] / c)))
    sph = g.nurbs_sphere(g.Frame(), 1.0, lon, lat, label)
    net = sph.net.copy()
    w = net[:, :, 3:4]
    net[:, :, :3] = (net[:, :, :3] / w * np.array([1.0, 1.0, c])) * w
    truth = g.ellipsoid_truth(g.Frame(), 1.0, 1.0, c)

    def inside(Q, tol):
        L = Q / np.array([1.0, 1.0, c])
        la = np.degrees(np.arcsin(np.clip(L[:, 2] / np.maximum(np.linalg.norm(L, axis=1), 1e-300), -1, 1)))
        lo = np.degrees(np.arctan2(L[:, 1], L[:, 0]))
        return (lo >= lon[0] - 1e-7) & (lo <= lon[1] + 1e-7) & (la >= lat[0] - 1e-7) & (la <= lat[1] + 1e-7)
    return g.Nurbs(sph.p, sph.q, sph.U, sph.V, net, truth, inside, label)


def surfaces(p=P):
    blue = spheroid_patch(p["KB"], (p["LB0"], p["LB1"]), (p["ZB0"], p["ZB1"]),
                          "blue: prolate spheroid patch r^2 + %g z^2 = 1" % p["KB"])
    orange = spheroid_patch(p["KO"], (p["LO0"], p["LO1"]), (p["ZO0"], p["ZO1"]),
                            "orange: oblate spheroid patch r^2 + %g z^2 = 1" % p["KO"])
    return blue, orange


def grid_fns(p=P):
    def mk(k, lon, zr):
        def f(n):
            a, z = np.meshgrid(np.radians(np.linspace(lon[0], lon[1], n)), np.linspace(zr[0], zr[1], n),
                               indexing="ij")
            r = np.sqrt(1 - k * z * z)
            return np.stack([r * np.cos(a), r * np.sin(a), z], -1)
        return f
    return [mk(p["KB"], (p["LB0"], p["LB1"]), (p["ZB0"], p["ZB1"])),
            mk(p["KO"], (p["LO0"], p["LO1"]), (p["ZO0"], p["ZO1"]))]


def arc_range(p=P):
    return math.radians(max(p["LB0"], p["LO0"])), math.radians(min(p["LB1"], p["LO1"]))


def contact_checks(p=P):
    """Normals of both surfaces along the arc (parallel), radial gap off the arc (one-sided), truth distances."""
    blue, orange = surfaces(p)
    t0, t1 = arc_range(p)
    t = np.linspace(t0, t1, 41)
    A = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    nb = A * np.array([1.0, 1.0, p["KB"]])  # gradients of r^2 + k z^2
    no = A * np.array([1.0, 1.0, p["KO"]])
    sin = np.linalg.norm(np.cross(nb, no), axis=1) / np.linalg.norm(nb, axis=1) / np.linalg.norm(no, axis=1)
    z = np.array([-0.3, -0.1, -0.01, 0.01, 0.1, 0.3])
    gap = np.sqrt(1 - p["KB"] * z * z) - np.sqrt(1 - p["KO"] * z * z)
    return dict(max_sin_normals_on_arc=float(sin.max()),
                radial_gap_blue_minus_orange={"%g" % a: float(b) for a, b in zip(z, gap)},
                contact="touching (gap >= 0 on both sides, ~ (KO - KB) z^2 / 2)",
                max_truth_distance_arc=float(max(g.truth_distance(s.truth(), A).max() for s in (blue, orange))))


def scores_for(fns, cam):
    T = C.Target(C.paper_panel(STEM))
    return {k: round(float(v), 4) for k, v in T.scores(*C.front_labels(fns, cam, T.W, T.H)).items()}


def build():
    blue, orange = surfaces()
    t0, t1 = arc_range()
    Y.add(CID, "Table 6 ex.2 (2,2): a tangential curve (orange spheroid inside a blue spheroid)", blue, orange,
          Y.SRC + "Table 6 (example (2))", "'a tangential curve', bi-degree (2,2)",
          "bi-degree (2,2) (rational); topology: one open tangential circular arc from the blue's left edge to its "
          "right edge, the orange touching the blue from behind (blue in front everywhere, as in the AA-GPU "
          "panel); appearance fitted to the Surfaces panel (camera and IoU in 'fit'): square-ish domed blue seen "
          "from slightly below, dome-shaped orange outline through it, arched red curve",
          "surface types (two coaxial spheroids sharing their equator) and all extents are fitted to the picture; "
          "the blue's rounded top-left corner, its slightly concave right edge and the orange's two small horns "
          "beyond that edge are not reproduced",
          "paper: " + Y.OPEN + " and " + Y.SPL + " returned nothing",
          curves=[(gc.circle((0, 0, 0), (0, 0, 1), 1.0), t0, t1, False, "tangent")], cov_tol=1e-2)
    case = gc.CASES[-1]
    case["truth_method"] = ("closed form: r_blue(z) > r_orange(z) for z != 0, equal at z = 0, so the intersection is "
                            "the unit circle z = 0 restricted to the common longitude range")
    return case


def main():
    case = build()
    curves = [b["points"] for b in case["expected"]["branches"]]
    after = scores_for([C.spec_grid_fn(s) for s in case["surfaces"]], CAM)
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
