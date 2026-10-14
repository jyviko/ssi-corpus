"""Render reconstructed cases from a fitted camera in the style of the paper's Fig. 1 panels.

Style (Yang, Jia & Yan 2023, Fig. 1): white background, first surface blue, second
surface orange, smooth two-sided shading, intersection in red.

  * ``style="surfaces"``: the "Surfaces" row (nearly opaque, saturated).
  * ``style="result"``:   the result rows (pale, translucent, red curves/dots on top).

Camera (dict):
  proj   "ortho" | "persp"
  el, az elevation / azimuth of the eye direction in degrees (matplotlib convention:
         eye direction = (cos el cos az, cos el sin az, sin el))
  roll   in-plane rotation of the image in degrees (counter-clockwise)
  target 3D point that lands on pixel (cx, cy)
  scale  pixels per model unit (ortho), or focal length in pixels (persp)
  dist   eye distance from target (persp only)
  cx, cy pixel position of the target in the panel

Rasterisation is a painter's algorithm per surface (triangles sorted far to near,
drawn opaque into that surface's own layer, with a flat per-triangle depth image),
followed by a per-pixel depth-ordered alpha composite of the layers. Rendering is
done at ``ss`` times the panel size and box-filtered down (anti-aliasing).

Library use:
  from render_match import Camera-free helpers: surface_grid, render, mask_iou, paper_mask
CLI:
  render_match.py <case.json> <camera.json> <out.png> [W H] [surfaces|result]
"""
import json
import math
import sys

import numpy as np
from PIL import Image, ImageDraw

BLUE = np.array([0.38, 0.58, 0.90])
ORANGE = np.array([0.98, 0.58, 0.30])
RED = (220, 20, 20)


# ---------------------------------------------------------------------------- geometry

def _basis(U, p, n, t):
    U = np.asarray(U, float)
    t = np.atleast_1d(np.asarray(t, float))
    m = len(U) - 1
    N = np.zeros((len(t), m))
    for i in range(m):
        last = U[i + 1] == U[-1] and U[i] < U[i + 1]
        N[:, i] = ((t >= U[i]) & ((t < U[i + 1]) | (last & (t <= U[i + 1])))).astype(float)
    for k in range(1, p + 1):
        M = np.zeros((len(t), m - k))
        for i in range(m - k):
            a = U[i + k] - U[i]
            b = U[i + k + 1] - U[i + 1]
            if a > 0:
                M[:, i] += (t - U[i]) / a * N[:, i]
            if b > 0:
                M[:, i] += (U[i + k + 1] - t) / b * N[:, i + 1]
        N = M
    assert N.shape[1] == n
    return N


def spec_domain(spec):
    if spec["type"] == "nurbs":
        p, q = spec["degree"]
        U, V = spec["knots_u"], spec["knots_v"]
        return (U[p], U[-p - 1]), (V[q], V[-q - 1])
    if spec["type"] == "torus":
        return tuple(np.radians(spec.get("major", [0, 360]))), tuple(np.radians(spec.get("minor", [0, 360])))
    raise ValueError(spec["type"])


def eval_spec(spec, u, v):
    """Evaluate a surface spec at scattered parameters (u, v) (same shape) -> (..., 3)."""
    u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
    sh = u.shape
    u, v = u.ravel(), v.ravel()
    if spec["type"] == "nurbs":
        p, q = spec["degree"]
        U, V = np.array(spec["knots_u"], float), np.array(spec["knots_v"], float)
        nu, nv = spec["nu"], spec["nv"]
        P = np.array(spec["points"], float).reshape(nu, nv, 3)
        W = np.array(spec.get("weights") or [1.0] * nu * nv, float).reshape(nu, nv)
        (u0, u1), (v0, v1) = spec_domain(spec)
        Nu, Nv = _basis(U, p, nu, np.clip(u, u0, u1)), _basis(V, q, nv, np.clip(v, v0, v1))
        num = np.einsum("ai,aj,ijk->ak", Nu, Nv, P * W[..., None])
        den = np.einsum("ai,aj,ij->a", Nu, Nv, W)
        return (num / den[:, None]).reshape(sh + (3,))
    if spec["type"] == "torus":
        c, x, y = (np.array(spec[k], float) for k in ("center", "xaxis", "yaxis"))
        z = np.cross(x, y)
        th, ph = u[:, None], v[:, None]
        out = c + (spec["R"] + spec["r"] * np.cos(ph)) * (np.cos(th) * x + np.sin(th) * y) + spec["r"] * np.sin(ph) * z
        return out.reshape(sh + (3,))
    raise ValueError(spec["type"])


def surface_grid(spec, n=121, m=None):
    """Evaluate a case-JSON surface spec on an n x m parameter grid -> (n, m, 3)."""
    m = m or n
    if spec["type"] == "nurbs":
        p, q = spec["degree"]
        U, V = np.array(spec["knots_u"], float), np.array(spec["knots_v"], float)
        nu, nv = spec["nu"], spec["nv"]
        P = np.array(spec["points"], float).reshape(nu, nv, 3)
        W = np.array(spec.get("weights") or [1.0] * nu * nv, float).reshape(nu, nv)
        u = np.linspace(U[p], U[-p - 1], n)
        v = np.linspace(V[q], V[-q - 1], m)
        Nu, Nv = _basis(U, p, nu, u), _basis(V, q, nv, v)
        num = np.einsum("ai,bj,ijk->abk", Nu, Nv, P * W[..., None])
        den = np.einsum("ai,bj,ij->ab", Nu, Nv, W)
        return num / den[..., None]
    if spec["type"] == "torus":
        c, x, y = (np.array(spec[k], float) for k in ("center", "xaxis", "yaxis"))
        z = np.cross(x, y)
        th = np.radians(np.linspace(*spec.get("major", [0, 360]), n))[:, None, None]
        ph = np.radians(np.linspace(*spec.get("minor", [0, 360]), m))[None, :, None]
        return c + (spec["R"] + spec["r"] * np.cos(ph)) * (np.cos(th) * x + np.sin(th) * y) + spec["r"] * np.sin(ph) * z
    raise ValueError(spec["type"])


# ---------------------------------------------------------------------------- camera

def camera_basis(cam):
    el, az, roll = (math.radians(cam.get(k, 0.0)) for k in ("el", "az", "roll"))
    d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])  # towards the eye
    up0 = np.array(cam.get("up", (0.0, 0.0, 1.0)), float)
    r = np.cross(up0, d)
    if np.linalg.norm(r) < 1e-9:
        r = np.cross((0.0, 1.0, 0.0), d)
    r /= np.linalg.norm(r)
    u = np.cross(d, r)
    cr, sr = math.cos(roll), math.sin(roll)
    r, u = cr * r + sr * u, -sr * r + cr * u
    return r, u, d


def project(P, cam, W, H):
    """3D points (..., 3) -> pixel x, y and depth (larger = farther), panel of size W x H."""
    r, u, d = camera_basis(cam)
    Q = np.asarray(P, float) - np.asarray(cam.get("target", (0, 0, 0)), float)
    xr, yu, zd = Q @ r, Q @ u, Q @ d
    cx = cam.get("cx", W / 2.0)
    cy = cam.get("cy", H / 2.0)
    if cam.get("proj", "ortho") == "persp":
        depth = cam["dist"] - zd
        f = cam["scale"]
        return cx + f * xr / depth, cy - f * yu / depth, depth
    s = cam["scale"]
    return cx + s * xr, cy - s * yu, -zd


# ---------------------------------------------------------------------------- rasterising

def _shade(nn, cam, base, style):
    """Two-sided Blinn-Phong colour for unit normals nn (..., 3)."""
    r, u, d = camera_basis(cam)
    L1 = -0.45 * r + 0.6 * u + 0.66 * d  # key light: upper left, in front
    L1 /= np.linalg.norm(L1)
    L2 = 0.7 * r + 0.1 * u + 0.7 * d     # fill light: right
    L2 /= np.linalg.norm(L2)
    nn = nn * np.sign((nn @ d) + 1e-30)[..., None]
    dif = 0.75 * np.clip(nn @ L1, 0, 1) + 0.25 * np.clip(nn @ L2, 0, 1)
    h = (L1 + d) / np.linalg.norm(L1 + d)
    spec = np.clip(nn @ h, 0, 1) ** 40
    if style == "surfaces":
        col = base * (0.35 + 0.75 * dif[..., None]) + 0.35 * spec[..., None]
    else:  # result rows: very pale, nearly opaque front surface (paper: orange ~(255,235,224), blue ~(226,235,250))
        col = base * (0.75 + 0.35 * dif[..., None]) + 0.1 * spec[..., None]
        col = 1.0 - 0.2 * (1.0 - np.clip(col, 0, 1))
    return np.clip(col, 0, 1)


def _layer(G, cam, W, H, ss, base, style):
    """Rasterise one surface grid into its own layer: RGB, coverage and per-pixel depth.

    Visibility inside the surface: triangles drawn far to near as triangle ids. Depth and normal are
    then interpolated per pixel (barycentric in screen space) so that two nearly coincident surfaces
    are ordered correctly pixel by pixel; degenerate vertex normals (collapsed edges) fall back to the
    face normal."""
    n, m = G.shape[:2]
    x, y, z = project(G, cam, W, H)
    x, y = x * ss, y * ss
    # vertex normals from the grid (one-sided differences at the borders)
    Gu = np.gradient(G, axis=0)
    Gv = np.gradient(G, axis=1)
    VN = np.cross(Gu, Gv).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n - 1), np.arange(m - 1), indexing="ij")
    i, j = i.ravel(), j.ravel()
    tri = np.concatenate([np.stack([i * m + j, (i + 1) * m + j, (i + 1) * m + j + 1], 1),
                          np.stack([i * m + j, (i + 1) * m + j + 1, i * m + j + 1], 1)])
    P = G.reshape(-1, 3)
    X, Y, Z = x.ravel(), y.ravel(), z.ravel()
    fn = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
    ln = np.linalg.norm(fn, axis=1)
    area2 = (X[tri[:, 1]] - X[tri[:, 0]]) * (Y[tri[:, 2]] - Y[tri[:, 0]]) - \
            (X[tri[:, 2]] - X[tri[:, 0]]) * (Y[tri[:, 1]] - Y[tri[:, 0]])
    ok = ln > 1e-14
    tri, fn, area2 = tri[ok], fn[ok] / ln[ok, None], area2[ok]
    zt = Z[tri].mean(1)
    order = np.argsort(-zt)
    ids = Image.new("I", (W * ss, H * ss), -1)
    dr = ImageDraw.Draw(ids)
    for t in order:
        a, b, c = tri[t]
        dr.polygon([(X[a], Y[a]), (X[b], Y[b]), (X[c], Y[c])], fill=int(t), outline=int(t))
    idm = np.asarray(ids, np.int64)
    cov = idm >= 0
    py, px = np.nonzero(cov)
    t = idm[py, px]
    A, B, C = tri[t, 0], tri[t, 1], tri[t, 2]
    ar = np.where(np.abs(area2[t]) < 1e-12, 1e-12, area2[t])
    wb = ((px - X[A]) * (Y[C] - Y[A]) - (X[C] - X[A]) * (py - Y[A])) / ar
    wc = ((X[B] - X[A]) * (py - Y[A]) - (px - X[A]) * (Y[B] - Y[A])) / ar
    wb, wc = np.clip(wb, 0, 1), np.clip(wc, 0, 1)
    s_ = wb + wc
    over = s_ > 1
    wb[over], wc[over] = wb[over] / s_[over], wc[over] / s_[over]
    wa = 1 - wb - wc
    dep = np.full(idm.shape, np.inf)
    dep[py, px] = wa * Z[A] + wb * Z[B] + wc * Z[C]
    nv = wa[:, None] * VN[A] + wb[:, None] * VN[B] + wc[:, None] * VN[C]
    nl = np.linalg.norm(nv, axis=1)
    # orient vertex normals consistently with the face normal, fall back to it when degenerate
    good = nl > 1e-10 * max(1.0, float(np.abs(VN).max()))
    nv = np.where(good[:, None], nv / np.where(nl > 0, nl, 1)[:, None], fn[t])
    img = np.ones(idm.shape + (3,))
    img[py, px] = _shade(nv, cam, base, style)
    # grid-index coordinates (i, j) of every covered pixel (for exact depth refinement)
    gi, gj = np.divmod(np.arange(n * m), m)
    ij = np.full(idm.shape + (2,), np.nan)
    ij[py, px, 0] = wa * gi[A] + wb * gi[B] + wc * gi[C]
    ij[py, px, 1] = wa * gj[A] + wb * gj[B] + wc * gj[C]
    return img, cov.astype(float), dep, ij


def _refine_close_depths(specs, layers, depths, cam, W, H, ss, n, grids, rel=0.01, iters=4):
    """Where both surfaces cover a pixel at nearly the same depth, replace the interpolated depths by the
    exact depth of each surface along that pixel's ray (Newton on the projected parametrisation), so that
    the order of near-coincident sheets (tangential contacts) is decided by the true geometry."""
    ext = max(float(np.ptp(G.reshape(-1, 3), axis=0).max()) for G in grids)
    both = np.isfinite(depths).all(0)
    close = np.zeros_like(both)
    close[both] = np.abs(depths[0][both] - depths[1][both]) < rel * ext
    if not close.any():
        return
    py, px = np.nonzero(close)
    for k, spec in enumerate(specs):
        (u0, u1), (v0, v1) = spec_domain(spec)
        ij = layers[k][3][py, px]
        u = u0 + (u1 - u0) * ij[:, 0] / (n - 1)
        v = v0 + (v1 - v0) * ij[:, 1] / (n - 1)
        hu, hv = 1e-6 * (u1 - u0), 1e-6 * (v1 - v0)
        for _ in range(iters):
            x0, y0, _ = project(eval_spec(spec, u, v), cam, W, H)
            xu, yu, _ = project(eval_spec(spec, u + hu, v), cam, W, H)
            xv, yv, _ = project(eval_spec(spec, u, v + hv), cam, W, H)
            a, b, c, d = (xu - x0) / hu, (xv - x0) / hv, (yu - y0) / hu, (yv - y0) / hv
            fx, fy = x0 - px / ss, y0 - py / ss
            det = a * d - b * c
            det = np.where(np.abs(det) < 1e-300, 1e-300, det)
            du, dv = (d * fx - b * fy) / det, (a * fy - c * fx) / det
            u = np.clip(u - du, u0, u1)
            v = np.clip(v - dv, v0, v1)
        _, _, z = project(eval_spec(spec, u, v), cam, W, H)
        layers[k][2][py, px] = z
        depths[k][py, px] = z


def render(specs_or_grids, cam, W, H, style="surfaces", curves=(), points=(), ss=3, n=121,
           alphas=None, curve_width=3.2, point_radius=6.5, colors=(BLUE, ORANGE), hidden_curves=True):
    """Render surfaces (case-JSON specs or precomputed grids) from cam. Returns (rgb uint8, [masks])."""
    grids = [s if isinstance(s, np.ndarray) else surface_grid(s, n) for s in specs_or_grids]
    if alphas is None:
        alphas = (0.92, 0.92) if style == "surfaces" else (0.85, 0.85)
    layers = [_layer(G, cam, W, H, ss, col, style) for G, col in zip(grids, colors)]
    depths = np.stack([np.where(m > 0.5, d, np.inf) for _, m, d, _ in layers])
    if len(layers) == 2 and all(isinstance(s, dict) for s in specs_or_grids):
        _refine_close_depths(specs_or_grids, layers, depths, cam, W, H, ss, n, grids)
    out = np.ones((H * ss, W * ss, 3))
    # depth-ordered composite (far first)
    order = np.argsort(-depths, axis=0)  # far to near
    for k in range(len(layers)):
        idx = order[k]
        for s, (img, m, _, _) in enumerate(layers):
            sel = (idx == s) & (m > 0.5)
            a = alphas[s]
            out[sel] = (1 - a) * out[sel] + a * img[sel]
    im = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
    masks = [np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) > 127
             for _, m, _, _ in layers]
    if curves or points:
        big = im.resize((W * ss, H * ss), Image.BICUBIC)
        dr = ImageDraw.Draw(big)
        for C in curves:
            C = np.asarray(C, float)
            cx, cy, _ = project(C, cam, W, H)
            dr.line(list(zip(cx * ss, cy * ss)), fill=RED, width=int(round(curve_width * ss)), joint="curve")
        for p in points:
            px, py, _ = project(np.asarray(p, float), cam, W, H)
            R = point_radius * ss
            dr.ellipse([px * ss - R, py * ss - R, px * ss + R, py * ss + R], fill=RED)
        im = big.resize((W, H), Image.LANCZOS)
    return np.asarray(im), masks


# ---------------------------------------------------------------------------- comparison

def paper_mask(rgb, thresh=0.06):
    """Non-background pixels of a paper panel (white background, grey for the NULL panels is not handled)."""
    a = np.asarray(rgb, float)[..., :3] / 255.0
    sat = a.max(-1) - a.min(-1)
    dark = 1.0 - a.mean(-1)
    m = (sat > thresh) | (dark > 0.12)
    # drop the panel frame (2 px)
    m[:2] = m[-2:] = False
    m[:, :2] = m[:, -2:] = False
    return m


def color_masks(rgb):
    """Pixels dominated by blue, by orange and by red (curve) in a panel."""
    a = np.asarray(rgb, float)[..., :3] / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    fg = paper_mask(rgb)
    red = (r > 0.6) & (g < 0.35) & (b < 0.35)
    blue = fg & (b > r + 0.02) & ~red
    orange = fg & (r > b + 0.02) & ~red
    return blue, orange, red


def mask_iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def compare_strip(paper_rgbs, ours_rgbs, out_path, labels=None):
    """paper | ours | 50% overlay, one row per pair; saved to out_path."""
    rows = []
    for P, O in zip(paper_rgbs, ours_rgbs):
        P = np.asarray(Image.fromarray(np.asarray(P)[..., :3]).resize(O.shape[1::-1]))
        ov = (0.5 * P.astype(float) + 0.5 * O[..., :3].astype(float)).astype(np.uint8)
        gap = np.full((O.shape[0], 8, 3), 255, np.uint8)
        rows.append(np.concatenate([P, gap, O[..., :3], gap, ov], 1))
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 6), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    Image.fromarray(np.concatenate(rows, 0)).save(out_path)


if __name__ == "__main__":
    case = json.load(open(sys.argv[1]))
    cam = json.load(open(sys.argv[2]))
    W, H = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (437, 386)
    style = sys.argv[6] if len(sys.argv) > 6 else "surfaces"
    curves = [b["points"] for b in case["expected"]["branches"]] if style == "result" else ()
    pts = case["expected"]["isolated_points"] if style == "result" else ()
    rgb, _ = render(case["surfaces"], cam, W, H, style, curves, pts)
    Image.fromarray(rgb).save(sys.argv[3])
