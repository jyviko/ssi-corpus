"""Table 6 example (1) of Yang, Jia & Yan 2023 ("a tangent line", bi-degree (3,1)),
refitted to the paper's panels.

What the panels show (cases/y23_fit/paper/y23_t6_ex1_tangent_line_*.png, 600 dpi):
  * blue: a ruled S-step strip seen almost along its flat parts: a thin wedge runs to
    the upper-left tip (61, 45), a near-vertical wall ~85 px wide crosses the middle,
    and a second, thicker wedge runs to the lower-right tip (430, 365). The silhouette is
    close to point-symmetric about the middle of the red line (245, 206); the lower
    wedge is thicker than the upper one (perspective).
  * every horizontal slice of the wall is the same ~90 x 13 px ruling vector: the strip
    is an extrusion of one planar profile along straight, parallel rulings.
  * orange: a shorter S with the same rulings, its tails turn earlier than the blue's
    (upper tail to (130, 121), lower tail to (355, 287)), sticking out of the blue band.
  * red: one straight segment along a ruling, half-way down the wall, spanning the full
    ruling length of both sheets (both share the ruling extent).
  * above the line the blue is drawn in front (AA-GPU panel: blue with z-fighting
    speckles), below it the orange: the sheets touch along the line and cross there,
    so the contact is tangential and of odd order.

Construction (exact, polynomial bi-degree (3,1)):
  * profile frame: s along the wall tangent at the line, h along the wall normal;
        blue   h = a1 s + a2 s^2 + a3 s^3,          s in [-SB1, SB2]
        orange h = a1 s + a2 s^2 + a3o s^3,         s in [-SO1, SO2]
    rulings w in [-W, W] perpendicular to the profile plane.
    orange - blue = (a3o - a3) s^3 along h at equal (s, w): same point, tangent plane and
    normal curvature on s = 0 (second-order contact), opposite sides for s < 0 / s > 0.
    Both sheets are graphs over the same (s, w) plane, so they meet only where the gap
    vanishes: the straight ruling s = 0, w in [-W, W] (closed form, no other branch).
  * placement: the (s, w, h) frame is rotated by G about the ruling and then reflected
    through the plane through the line centre perpendicular to the fitted view direction.
    The reflection (an isometry) is what makes the blue the front sheet above the line,
    as drawn; without it the same picture shows the opposite depth order.

Run: .venv/bin/python -I cases/y23_fit/y23_t6_ex1_tangent_line.py
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

CID = "y23_t6_ex1_tangent_line"
STEM = CID

# fitted shape (profile frame units)
P = dict(G=67.07, a1=0.0505, a2=0.0081, a3=-0.2134, a3o=-0.8765, SB1=1.594, SB2=1.566, SO1=0.859, SO2=0.796,
         W=0.765)
# fitted camera of the Surfaces panel (render_match convention)
CAM = dict(proj="persp", el=-1.27, az=115.82, roll=-175.55, target=[0.0, 0.0, 0.0], scale=2305.0, dist=17.89,
           cx=246.88, cy=206.81)
# baseline: the previous construction (commit c91ad29) scored the same way with its own best-fitting camera
BEFORE = dict(construction="graph z = 1.2u - 0.4u^3 (3,1) on [-1,1]^2, orange z + 0.8u^2 on [-0.6,0.6]x[-0.7,0.7] "
                           "(touching, not crossing)",
              camera=dict(proj="ortho", el=10.11, az=120.96, roll=-164.96, scale=148.61, cx=285.84, cy=208.78,
                          target=[0.0, 0.0, 0.0]),
              scores=dict(iou_silhouette=0.8062, iou_blue=0.37, iou_orange=0.2707, chamfer_silhouette_px=6.24,
                          chamfer_blue_px=8.51))


def placement(p=P, cam=CAM):
    """Linear map (3x3) from profile-frame coordinates (s, w, h) to world: rotation by G, then the
    reflection through the plane perpendicular to the camera's view direction."""
    c, s_ = math.cos(math.radians(p["G"])), math.sin(math.radians(p["G"]))
    R = np.array([[c, 0.0, -s_], [0.0, 1.0, 0.0], [s_, 0.0, c]])  # columns: s-, w-, h- directions
    _, _, d = rm.camera_basis(cam)
    M = np.eye(3) - 2.0 * np.outer(d, d)
    return M @ R


def profile(k3, p=P):
    return np.array([[0.0], [p["a1"]], [p["a2"]], [k3]])  # h(s) power coefficients (s^i w^0)


def sheet(k3, sr, label, p=P):
    A = placement(p)
    S, Wc, Hh = Y.mono(1, 0), Y.mono(0, 1), profile(k3, p)
    H = [g.p2_add(g.p2_add(g.p2_scale(S, A[i, 0]), g.p2_scale(Wc, A[i, 1])), g.p2_scale(Hh, A[i, 2])) for i in range(3)]
    W = p["W"]

    def inside(Q, tol):
        L = Q @ A  # A is orthogonal: local (s, w, h) = A^T Q
        return (L[:, 0] >= sr[0] - tol) & (L[:, 0] <= sr[1] + tol) & (np.abs(L[:, 1]) <= W + tol)
    return g.rational_poly_patch(H[0], H[1], H[2], np.ones((1, 1)), sr, (-W, W), label, inside=inside,
                                 degree=(3, 1))


def surfaces(p=P):
    blue = sheet(p["a3"], (-p["SB1"], p["SB2"]), "blue: ruled cubic S-step h = a1 s + a2 s^2 + a3 s^3", p)
    orange = sheet(p["a3o"], (-p["SO1"], p["SO2"]), "orange: shorter S-step h = a1 s + a2 s^2 + a3o s^3", p)
    return blue, orange


def grid_fns(p=P):
    A = placement(p)

    def mk(k3, sr):
        def f(n):
            s, w = np.meshgrid(np.linspace(sr[0], sr[1], n), np.linspace(-p["W"], p["W"], n), indexing="ij")
            h = p["a1"] * s + p["a2"] * s * s + k3 * s ** 3
            return np.stack([s, w, h], -1) @ A.T
        return f
    return [mk(p["a3"], (-p["SB1"], p["SB2"])), mk(p["a3o"], (-p["SO1"], p["SO2"]))]


def line_points(n=200, p=P):
    A = placement(p)
    w = np.linspace(-p["W"], p["W"], n)
    return np.outer(w, A[:, 1])


def _normals(surf, sr, s, w, W, eps=1e-6):
    """Unit normals of a patch at profile parameters (s, w) by central differences of the NURBS itself."""
    u = (s - sr[0]) / (sr[1] - sr[0])
    v = (w + W) / (2 * W)
    out = []
    for a, b in zip(u, v):
        du = (surf.eval(min(a + eps, 1), b) - surf.eval(max(a - eps, 0), b))
        dv = (surf.eval(a, min(b + eps, 1)) - surf.eval(a, max(b - eps, 0)))
        n = np.cross(du, dv)
        out.append(n / np.linalg.norm(n))
    return np.array(out)


def contact_checks(p=P):
    """Normals of both NURBS along the line (parallel), signed gap either side (sign change = crossing),
    truth distances of the stored line."""
    W = p["W"]
    blue, orange = surfaces(p)
    srb, sro = (-p["SB1"], p["SB2"]), (-p["SO1"], p["SO2"])
    w = np.linspace(-W, W, 21)
    nb = _normals(blue, srb, np.zeros_like(w), w, W)
    no = _normals(orange, sro, np.zeros_like(w), w, W)
    out = {"max_sin_normals_on_line": float(np.linalg.norm(np.cross(nb, no), axis=1).max())}
    A = placement(p)
    gaps = {}
    for sv in (-0.4, -0.2, -0.1, 0.1, 0.2, 0.4):
        # nearest blue point to the orange point at (s, 0): along the blue normal, signed
        Po = orange.eval((sv - sro[0]) / (sro[1] - sro[0]), 0.5)
        L = A.T @ Po
        hb = p["a1"] * L[0] + p["a2"] * L[0] ** 2 + p["a3"] * L[0] ** 3
        gaps["%g" % sv] = float(L[2] - hb)
    out["gap_orange_minus_blue_along_h"] = gaps
    out["contact_order"] = 2
    L = line_points(p=p)
    out["max_truth_distance_line"] = float(max(g.truth_distance(x.truth(), L).max() for x in (blue, orange)))
    return out


def scores_for(fns, cam):
    T = C.Target(C.paper_panel(STEM))
    return {k: round(float(v), 4) for k, v in T.scores(*C.front_labels(fns, cam, T.W, T.H)).items()}


def build():
    blue, orange = surfaces()
    A = placement()
    W = P["W"]
    Y.add(CID, "Table 6 ex.1 (3,1): a tangent line", blue, orange, Y.SRC + "Table 6 (example (1))",
          "'a tangent line', bi-degree (3,1)",
          "bi-degree (3,1) (polynomial); topology: one open straight tangential line along a ruling, half-way down "
          "the wall, spanning the full ruling length of both sheets; the sheets touch there with second-order "
          "contact and cross (blue in front above the line, orange below, as drawn); appearance fitted to the "
          "Surfaces panel (camera and IoU in 'fit'): blue ruled S-step strip seen nearly along its flat parts, "
          "shorter orange S whose tails turn earlier and stick out of the blue band",
          "the paper's control nets are not published: the profile is a cubic graph in a frame aligned with the "
          "wall (fitted), the contact order (2, the lowest odd order that matches the drawn crossing) and the "
          "reflection that sets the drawn depth order are inferred from the picture",
          "paper: " + Y.OPEN + " and " + Y.COMM + " returned nothing, " + Y.SPL + " scattered points",
          curves=[(gc.line(-W * A[:, 1], A[:, 1]), 0.0, 2 * W, False, "tangent")], cov_tol=1e-2,
          notes="contact order 2 along the ruling s = 0; the gap grows like |s|^3")
    case = gc.CASES[-1]
    case["truth_method"] = ("closed form: both sheets are graphs h(s) over the same (s, w) plane with h_o - h_b = "
                            "(a3o - a3) s^3, so the intersection is exactly the ruling s = 0")
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
