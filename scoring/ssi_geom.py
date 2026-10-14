"""Closed-form geometry for the SSI evidence cases (own code, numpy only).

Surfaces carry three things:
  * spec()      - the description the C++ runner turns into an SMLib surface
  * truth()     - an implicit description used by the scorer for exact distances
  * inside(P)   - closed-form membership of points in the trimmed patch

NURBS surfaces are built here from exact constructions (Bernstein conversion of
polynomial height fields, rational quadratic arcs of revolution), refined with
knot insertion and degree elevation. Their geometry is therefore known exactly:
every NURBS surface in this file is a patch of a plane, sphere, cylinder, cone,
torus or polynomial height field.
"""
import math
from math import comb

import numpy as np

# ----------------------------------------------------------------------------
# frames


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


class Frame:
    def __init__(self, origin=(0, 0, 0), xaxis=(1, 0, 0), yaxis=(0, 1, 0)):
        self.o = np.asarray(origin, dtype=float)
        self.x = unit(xaxis)
        y = np.asarray(yaxis, dtype=float)
        y = y - np.dot(y, self.x) * self.x
        self.y = unit(y)
        self.z = np.cross(self.x, self.y)

    def to_local(self, P):
        P = np.atleast_2d(P) - self.o
        return np.stack([P @ self.x, P @ self.y, P @ self.z], axis=-1)

    def to_world(self, L):
        L = np.atleast_2d(L)
        return self.o + L[:, :1] * self.x + L[:, 1:2] * self.y + L[:, 2:3] * self.z

    def dict(self):
        return {"origin": list(self.o), "xaxis": list(self.x), "yaxis": list(self.y)}


def angle_in(theta_deg, lo, hi, tol=1e-9):
    """theta (deg, any range) inside the angular interval [lo, hi] (deg, hi-lo<=360)."""
    if hi - lo >= 360 - 1e-12:
        return np.ones_like(theta_deg, dtype=bool)
    t = np.mod(theta_deg - lo, 360.0)
    w = hi - lo
    return (t <= w + tol) | (t >= 360.0 - tol)


# ----------------------------------------------------------------------------
# implicit truths: exact (or first-order for height fields) distance functions


def truth_distance(truth, P):
    P = np.atleast_2d(np.asarray(P, dtype=float))
    k = truth["kind"]
    if k == "plane":
        n = unit(truth["normal"])
        return np.abs((P - truth["point"]) @ n)
    if k == "sphere":
        return np.abs(np.linalg.norm(P - truth["center"], axis=1) - truth["radius"])
    if k in ("cylinder", "cone", "torus"):
        a = unit(truth["axis"])
        Q = P - truth["point"]
        h = Q @ a
        rho = np.linalg.norm(Q - h[:, None] * a, axis=1)
        if k == "cylinder":
            return np.abs(rho - truth["radius"])
        if k == "cone":
            al = math.radians(truth["half_angle_deg"])
            # signed distance to the nappe h>=0 (valid near the surface)
            return np.abs(rho * math.cos(al) - h * math.sin(al))
        return np.abs(np.hypot(rho - truth["R"], h) - truth["r"])
    if k == "heightfield":
        c = np.asarray(truth["coeffs"], dtype=float)  # c[i][j] a^i b^j, dependent coordinate = h(a, b)
        ax = truth.get("axes", [0, 1, 2])  # indices of (a, b, dependent)
        x, y, z = P[:, ax[0]], P[:, ax[1]], P[:, ax[2]]
        h = np.zeros(len(P))
        hx = np.zeros(len(P))
        hy = np.zeros(len(P))
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                if c[i, j] == 0:
                    continue
                h += c[i, j] * x**i * y**j
                if i > 0:
                    hx += c[i, j] * i * x ** (i - 1) * y**j
                if j > 0:
                    hy += c[i, j] * j * x**i * y ** (j - 1)
        return np.abs(z - h) / np.sqrt(1 + hx * hx + hy * hy)
    if k == "param":
        return param_distance(truth, P)
    if k == "quadric":
        A = np.asarray(truth["axes"], float)
        abc = np.asarray(truth["abc"], float)
        sg = np.asarray(truth["signs"], float)
        L = (P - truth["center"]) @ A.T / abc
        E = np.sum(sg * L * L, axis=1) - 1.0
        grad = 2 * (sg * L / abc) @ A
        return np.abs(E) / np.linalg.norm(grad, axis=1)
    if k == "ellipsoid":
        A = np.asarray(truth["axes"], float)
        abc = np.asarray(truth["abc"], float)
        L = (P - truth["center"]) @ A.T / abc  # scaled local coordinates
        E = np.sum(L * L, axis=1) - 1.0
        grad = 2 * (L / abc) @ A
        # first-order distance, refined by one Newton step along the gradient
        return np.abs(E) / np.linalg.norm(grad, axis=1)
    raise ValueError(k)


# ----------------------------------------------------------------------------
# analytic surfaces


class Plane:
    def __init__(self, frame, u, v):
        self.f, self.u, self.v = frame, u, v

    def spec(self):
        return {"type": "plane", **self.f.dict(), "u": list(self.u), "v": list(self.v)}

    def truth(self):
        return {"kind": "plane", "point": list(self.f.o), "normal": list(self.f.z)}

    def inside(self, P, tol=1e-9):
        L = self.f.to_local(P)
        return (
            (L[:, 0] >= self.u[0] - tol)
            & (L[:, 0] <= self.u[1] + tol)
            & (L[:, 1] >= self.v[0] - tol)
            & (L[:, 1] <= self.v[1] + tol)
        )

    def size(self):
        return max(self.u[1] - self.u[0], self.v[1] - self.v[0])

    def grid(self, n):
        a, b = np.meshgrid(np.linspace(*self.u, n), np.linspace(*self.v, n))
        return self.f.to_world(np.column_stack([a.ravel(), b.ravel(), 0 * a.ravel()]))


class Cylinder:
    def __init__(self, frame, r, angle=(0, 360), height=(0, 1)):
        self.f, self.r, self.angle, self.height = frame, r, angle, height

    def spec(self):
        return {"type": "cylinder", **self.f.dict(), "radius": self.r, "angle": list(self.angle),
                "height": list(self.height)}

    def truth(self):
        return {"kind": "cylinder", "point": list(self.f.o), "axis": list(self.f.z), "radius": self.r}

    def inside(self, P, tol=1e-9):
        L = self.f.to_local(P)
        th = np.degrees(np.arctan2(L[:, 1], L[:, 0]))
        return angle_in(th, *self.angle, tol=1e-7) & (L[:, 2] >= self.height[0] - tol) & (
            L[:, 2] <= self.height[1] + tol)

    def size(self):
        return max(2 * self.r, self.height[1] - self.height[0])

    def grid(self, n):
        a, h = np.meshgrid(np.radians(np.linspace(*self.angle, n)), np.linspace(*self.height, n))
        a, h = a.ravel(), h.ravel()
        return self.f.to_world(np.column_stack([self.r * np.cos(a), self.r * np.sin(a), h]))


class Cone:
    """Single nappe, apex at frame origin, opening along +z, full rotation."""

    def __init__(self, frame, half_angle_deg, height):
        self.f, self.a, self.height = frame, half_angle_deg, height

    def spec(self):
        d = self.f.dict()
        return {"type": "cone", "apex": d["origin"], "xaxis": d["xaxis"], "yaxis": d["yaxis"],
                "half_angle_deg": self.a, "height": list(self.height)}

    def truth(self):
        return {"kind": "cone", "point": list(self.f.o), "axis": list(self.f.z), "half_angle_deg": self.a}

    def inside(self, P, tol=1e-9):
        L = self.f.to_local(P)
        return (L[:, 2] >= self.height[0] - tol) & (L[:, 2] <= self.height[1] + tol)

    def size(self):
        return 2 * self.height[1] * math.tan(math.radians(self.a))

    def grid(self, n):
        a, h = np.meshgrid(np.linspace(0, 2 * math.pi, n), np.linspace(*self.height, n))
        a, h = a.ravel(), h.ravel()
        t = math.tan(math.radians(self.a))
        return self.f.to_world(np.column_stack([h * t * np.cos(a), h * t * np.sin(a), h]))


class Sphere:
    def __init__(self, frame, R, lon=(0, 360), lat=(-90, 90)):
        self.f, self.R, self.lon, self.lat = frame, R, lon, lat

    def spec(self):
        d = self.f.dict()
        return {"type": "sphere", "center": d["origin"], "xaxis": d["xaxis"], "yaxis": d["yaxis"],
                "radius": self.R, "lon": list(self.lon), "lat": list(self.lat)}

    def truth(self):
        return {"kind": "sphere", "center": list(self.f.o), "radius": self.R}

    def inside(self, P, tol=1e-9):
        L = self.f.to_local(P)
        n = np.linalg.norm(L, axis=1)
        lat = np.degrees(np.arcsin(np.clip(L[:, 2] / np.maximum(n, 1e-300), -1, 1)))
        lon = np.degrees(np.arctan2(L[:, 1], L[:, 0]))
        rho = np.hypot(L[:, 0], L[:, 1])
        lon_ok = angle_in(lon, *self.lon, tol=1e-7) | (rho < 1e-9 * self.R)
        return lon_ok & (lat >= self.lat[0] - 1e-7) & (lat <= self.lat[1] + 1e-7)

    def size(self):
        return 2 * self.R

    def grid(self, n):
        lo, la = np.meshgrid(np.radians(np.linspace(*self.lon, n)), np.radians(np.linspace(*self.lat, n)))
        lo, la = lo.ravel(), la.ravel()
        R = self.R
        return self.f.to_world(np.column_stack([R * np.cos(la) * np.cos(lo), R * np.cos(la) * np.sin(lo),
                                                R * np.sin(la)]))


class Torus:
    """minor angle 0 = outer equator, increasing towards +z (SMLib convention, probed)."""

    def __init__(self, frame, R, r, major=(0, 360), minor=(0, 360)):
        self.f, self.R, self.r, self.major, self.minor = frame, R, r, major, minor

    def spec(self):
        d = self.f.dict()
        return {"type": "torus", "center": d["origin"], "xaxis": d["xaxis"], "yaxis": d["yaxis"],
                "R": self.R, "r": self.r, "major": list(self.major), "minor": list(self.minor)}

    def truth(self):
        return {"kind": "torus", "point": list(self.f.o), "axis": list(self.f.z), "R": self.R, "r": self.r}

    def inside(self, P, tol=1e-9):
        L = self.f.to_local(P)
        rho = np.hypot(L[:, 0], L[:, 1])
        maj = np.degrees(np.arctan2(L[:, 1], L[:, 0]))
        mnr = np.degrees(np.arctan2(L[:, 2], rho - self.R))
        return angle_in(maj, *self.major, tol=1e-7) & angle_in(mnr, *self.minor, tol=1e-7)

    def size(self):
        return 2 * (self.R + self.r)

    def grid(self, n):
        A, B = np.meshgrid(np.radians(np.linspace(*self.major, n)), np.radians(np.linspace(*self.minor, n)))
        A, B = A.ravel(), B.ravel()
        rho = self.R + self.r * np.cos(B)
        return self.f.to_world(np.column_stack([rho * np.cos(A), rho * np.sin(A), self.r * np.sin(B)]))


# ----------------------------------------------------------------------------
# NURBS machinery (homogeneous coordinates, own implementation of textbook
# algorithms: Bernstein conversion, Boehm knot insertion, Bezier degree elevation)


def bspline_basis(knots, p, t):
    """All nonzero basis functions at t: returns (span, N[0..p])."""
    n = len(knots) - p - 2
    if t >= knots[n + 1]:
        span = n
    else:
        span = int(np.searchsorted(knots, t, side="right") - 1)
        span = min(max(span, p), n)
    N = np.zeros(p + 1)
    N[0] = 1.0
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    for j in range(1, p + 1):
        left[j] = t - knots[span + 1 - j]
        right[j] = knots[span + j] - t
        saved = 0.0
        for r in range(j):
            tmp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * tmp
            saved = left[j - r] * tmp
        N[j] = saved
    return span, N


def curve_insert_knot(Pw, knots, p, t):
    """Boehm: insert t once into a homogeneous curve Pw (n+1, d)."""
    k = int(np.searchsorted(knots, t, side="right") - 1)
    n = len(Pw) - 1
    Q = np.zeros((n + 2, Pw.shape[1]))
    for i in range(n + 2):
        if i <= k - p:
            Q[i] = Pw[i]
        elif i >= k + 1:
            Q[i] = Pw[i - 1]
        else:
            a = (t - knots[i]) / (knots[i + p] - knots[i])
            Q[i] = a * Pw[i] + (1 - a) * Pw[i - 1]
    return Q, np.insert(knots, k + 1, t)


def bezier_elevate(Pw, times):
    """Degree-elevate a single Bezier segment (homogeneous) `times` times."""
    for _ in range(times):
        n = len(Pw) - 1
        Q = np.zeros((n + 2, Pw.shape[1]))
        Q[0], Q[-1] = Pw[0], Pw[-1]
        for i in range(1, n + 1):
            a = i / (n + 1)
            Q[i] = a * Pw[i - 1] + (1 - a) * Pw[i]
        Pw = Q
    return Pw


class Nurbs:
    """Tensor NURBS surface; net[i][j] homogeneous (wx, wy, wz, w), i along u."""

    def __init__(self, p, q, U, V, net, truth, inside_fn, label, domain=None):
        self.p, self.q = p, q
        self.U, self.V = np.asarray(U, float), np.asarray(V, float)
        self.net = np.asarray(net, float)
        self._truth, self._inside, self.label, self.domain = truth, inside_fn, label, domain

    # refinement --------------------------------------------------------------
    def insert_u(self, ts):
        for t in ts:
            rows = []
            for j in range(self.net.shape[1]):
                Q, U2 = curve_insert_knot(self.net[:, j, :], self.U, self.p, t)
                rows.append(Q)
            self.net = np.stack(rows, axis=1)
            self.U = U2
        return self

    def insert_v(self, ts):
        for t in ts:
            cols = []
            for i in range(self.net.shape[0]):
                Q, V2 = curve_insert_knot(self.net[i], self.V, self.q, t)
                cols.append(Q)
            self.net = np.stack(cols, axis=0)
            self.V = V2
        return self

    def elevate(self, tu, tv):
        """Only for single-segment (Bezier) surfaces."""
        assert len(self.U) == 2 * (self.p + 1) and len(self.V) == 2 * (self.q + 1)
        if tu:
            self.net = np.stack([bezier_elevate(self.net[:, j, :], tu) for j in range(self.net.shape[1])], axis=1)
            self.p += tu
            self.U = np.array([self.U[0]] * (self.p + 1) + [self.U[-1]] * (self.p + 1))
        if tv:
            self.net = np.stack([bezier_elevate(self.net[i], tv) for i in range(self.net.shape[0])], axis=0)
            self.q += tv
            self.V = np.array([self.V[0]] * (self.q + 1) + [self.V[-1]] * (self.q + 1))
        return self

    def refine_uniform(self, nu, nv):
        u0, u1 = self.U[0], self.U[-1]
        v0, v1 = self.V[0], self.V[-1]
        existing_u = set(np.round(self.U, 12))
        existing_v = set(np.round(self.V, 12))
        tu = [u0 + (u1 - u0) * k / (nu + 1) for k in range(1, nu + 1)]
        tv = [v0 + (v1 - v0) * k / (nv + 1) for k in range(1, nv + 1)]
        self.insert_u([t for t in tu if round(t, 12) not in existing_u])
        self.insert_v([t for t in tv if round(t, 12) not in existing_v])
        return self

    # evaluation ---------------------------------------------------------------
    def eval(self, u, v):
        su, Nu = bspline_basis(self.U, self.p, u)
        sv, Nv = bspline_basis(self.V, self.q, v)
        blk = self.net[su - self.p: su + 1, sv - self.q: sv + 1, :]
        H = np.einsum("i,j,ijk->k", Nu, Nv, blk)
        return H[:3] / H[3]

    def grid(self, n):
        return np.array([self.eval(u, v) for u in np.linspace(self.U[0], self.U[-1], n)
                         for v in np.linspace(self.V[0], self.V[-1], n)])

    def spec(self):
        nu, nv = self.net.shape[:2]
        w = self.net[:, :, 3]
        pts = self.net[:, :, :3] / w[:, :, None]
        d = {"type": "nurbs", "label": self.label, "degree": [self.p, self.q],
             "knots_u": list(self.U), "knots_v": list(self.V), "nu": nu, "nv": nv,
             "points": pts.reshape(-1, 3).tolist(), "weights": w.reshape(-1).tolist()}
        if self.domain is not None:
            d["domain"] = list(self.domain)
        return d

    def truth(self):
        return self._truth

    def inside(self, P, tol=1e-9):
        return self._inside(np.atleast_2d(P), tol)

    def self_check(self, n=13):
        """Max distance of NURBS sample points to the implicit truth (must be ~1e-14)."""
        us = np.linspace(self.U[0], self.U[-1], n)
        vs = np.linspace(self.V[0], self.V[-1], n)
        P = np.array([self.eval(u, v) for u in us for v in vs])
        return float(truth_distance(self._truth, P).max()), bool(self.inside(P, 1e-7).all())


def heightfield_nurbs(coeffs, xr, yr, p, q, label, refine=(0, 0)):
    """z = sum c[i][j] x^i y^j over [x0,x1]x[y0,y1] as an exact polynomial Bezier
    patch of degree (p, q) (p, q >= the polynomial degrees), optionally refined."""
    c = np.asarray(coeffs, float)
    x0, x1 = xr
    y0, y1 = yr
    L, M = x1 - x0, y1 - y0
    # power coefficients a[k][l] in s,t where x = x0 + L s, y = y0 + M t
    a = np.zeros((p + 1, q + 1))
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c[i, j] == 0:
                continue
            for k in range(i + 1):
                for l in range(j + 1):
                    a[k, l] += c[i, j] * comb(i, k) * x0 ** (i - k) * L**k * comb(j, l) * y0 ** (j - l) * M**l
    b = np.zeros((p + 1, q + 1))
    for k in range(p + 1):
        for l in range(q + 1):
            s = 0.0
            for i in range(k + 1):
                for j in range(l + 1):
                    s += comb(k, i) / comb(p, i) * comb(l, j) / comb(q, j) * a[i, j]
            b[k, l] = s
    net = np.zeros((p + 1, q + 1, 4))
    for k in range(p + 1):
        for l in range(q + 1):
            net[k, l] = [x0 + L * k / p, y0 + M * l / q, b[k, l], 1.0]
    U = [0.0] * (p + 1) + [1.0] * (p + 1)
    V = [0.0] * (q + 1) + [1.0] * (q + 1)
    truth = {"kind": "heightfield", "coeffs": c.tolist()}

    def inside(P, tol):
        return (P[:, 0] >= x0 - tol) & (P[:, 0] <= x1 + tol) & (P[:, 1] >= y0 - tol) & (P[:, 1] <= y1 + tol)

    s = Nurbs(p, q, U, V, net, truth, inside, label)
    if refine[0] or refine[1]:
        s.refine_uniform(*refine)
    return s


def arc_ctrl(a0, a1):
    """Rational quadratic unit-circle arc a0..a1 (deg), split into <=90 deg pieces.
    Returns (points (k,2), weights (k,), knots)."""
    n = max(1, int(math.ceil((a1 - a0) / 90.0 - 1e-12)))
    d = math.radians(a1 - a0) / n
    w1 = math.cos(d / 2)
    pts, wts = [], []
    for s in range(n):
        t0 = math.radians(a0) + s * d
        t1 = t0 + d
        P0 = np.array([math.cos(t0), math.sin(t0)])
        P2 = np.array([math.cos(t1), math.sin(t1)])
        tm = (t0 + t1) / 2
        P1 = np.array([math.cos(tm), math.sin(tm)]) / w1
        if s == 0:
            pts.append(P0)
            wts.append(1.0)
        pts += [P1, P2]
        wts += [w1, 1.0]
    knots = [0.0] * 3
    for s in range(1, n):
        knots += [s / n, s / n]
    knots += [1.0] * 3
    return np.array(pts), np.array(wts), np.array(knots)


def revolve_nurbs(frame, meridian_pts, meridian_w, meridian_knots, meridian_deg, angle, truth, inside, label):
    """Surface of revolution about frame.z. meridian points are (rho, z) in the
    frame's xz half-plane; u = rotation, v = meridian."""
    cp, cw, ck = arc_ctrl(*angle)
    nu, nv = len(cp), len(meridian_pts)
    net = np.zeros((nu, nv, 4))
    for i in range(nu):
        for j in range(nv):
            rho, z = meridian_pts[j]
            L = np.array([[rho * cp[i, 0], rho * cp[i, 1], z]])
            Pw = frame.to_world(L)[0]
            w = cw[i] * meridian_w[j]
            net[i, j] = [Pw[0] * w, Pw[1] * w, Pw[2] * w, w]
    return Nurbs(2, meridian_deg, ck, meridian_knots, net, truth, inside, label)


def nurbs_cylinder(frame, r, angle, height, label):
    cyl = Cylinder(frame, r, angle, height)
    mer = np.array([[r, height[0]], [r, height[1]]])
    return revolve_nurbs(frame, mer, np.ones(2), np.array([0, 0, 1, 1.0]), 1, angle, cyl.truth(),
                         cyl.inside, label)


def nurbs_sphere(frame, R, lon, lat, label):
    sph = Sphere(frame, R, lon, lat)
    mp, mw, mk = arc_ctrl(lat[0], lat[1])  # meridian arc in (rho, z) = (cos, sin)
    return revolve_nurbs(frame, mp * R, mw, mk, 2, lon, sph.truth(), sph.inside, label)


def nurbs_torus(frame, R, r, major, minor, label):
    tor = Torus(frame, R, r, major, minor)
    mp, mw, mk = arc_ctrl(minor[0], minor[1])
    mer = np.column_stack([R + r * mp[:, 0], r * mp[:, 1]])
    return revolve_nurbs(frame, mer, mw, mk, 2, major, tor.truth(), tor.inside, label)


def nurbs_cone(frame, half_angle_deg, height, angle, label):
    cone = Cone(frame, half_angle_deg, height)
    t = math.tan(math.radians(half_angle_deg))
    mer = np.array([[height[0] * t, height[0]], [height[1] * t, height[1]]])
    return revolve_nurbs(frame, mer, np.ones(2), np.array([0, 0, 1, 1.0]), 1, angle, cone.truth(),
                         cone.inside, label)


def nurbs_plane(frame, u, v, p, q, label, refine=(0, 0), warp=0.0):
    """Planar patch as a degree (p,q) B-spline. warp != 0 moves interior control
    points within the plane (non-uniform parameterisation, same point set)."""
    pl = Plane(frame, u, v)
    net = np.zeros((p + 1, q + 1, 4))
    for k in range(p + 1):
        for l in range(q + 1):
            s, t = k / p, l / q
            if 0 < k < p:
                s = s + warp * s * (1 - s)
            if 0 < l < q:
                t = t + warp * t * (1 - t) * 0.5
            L = np.array([[u[0] + (u[1] - u[0]) * s, v[0] + (v[1] - v[0]) * t, 0.0]])
            P = frame.to_world(L)[0]
            net[k, l] = [*P, 1.0]
    U = [0.0] * (p + 1) + [1.0] * (p + 1)
    V = [0.0] * (q + 1) + [1.0] * (q + 1)
    s = Nurbs(p, q, U, V, net, pl.truth(), pl.inside, label)
    if refine[0] or refine[1]:
        s.refine_uniform(*refine)
    return s


def all_surfaces_selfcheck(surfs):
    out = []
    for s in surfs:
        if isinstance(s, Nurbs):
            out.append(s.self_check())
    return out


def from_spec(spec):
    """Rebuild a Python surface (geometry only) from a runner spec."""
    t = spec["type"]
    if t == "nurbs":
        nu, nv = spec["nu"], spec["nv"]
        P = np.asarray(spec["points"], float).reshape(nu, nv, 3)
        W = np.asarray(spec["weights"], float).reshape(nu, nv)
        net = np.concatenate([P * W[:, :, None], W[:, :, None]], axis=2)
        return Nurbs(spec["degree"][0], spec["degree"][1], spec["knots_u"], spec["knots_v"], net, None,
                     lambda P_, tol: np.ones(len(P_), bool), spec.get("label", ""))
    if t == "plane":
        return Plane(Frame(spec["origin"], spec["xaxis"], spec["yaxis"]), spec["u"], spec["v"])
    if t == "cylinder":
        return Cylinder(Frame(spec["origin"], spec["xaxis"], spec["yaxis"]), spec["radius"], spec["angle"],
                        spec["height"])
    if t == "cone":
        return Cone(Frame(spec["apex"], spec["xaxis"], spec["yaxis"]), spec["half_angle_deg"], spec["height"])
    if t == "sphere":
        return Sphere(Frame(spec["center"], spec["xaxis"], spec["yaxis"]), spec["radius"], spec["lon"], spec["lat"])
    if t == "torus":
        return Torus(Frame(spec["center"], spec["xaxis"], spec["yaxis"]), spec["R"], spec["r"], spec["major"],
                     spec["minor"])
    raise ValueError(t)


# ----------------------------------------------------------------------------
# 2D polynomials as coefficient arrays c[i][j] (u^i w^j), own small algebra


def p2(c):
    return np.atleast_2d(np.asarray(c, float))


def p2_add(a, b):
    a, b = p2(a), p2(b)
    out = np.zeros((max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])))
    out[: a.shape[0], : a.shape[1]] += a
    out[: b.shape[0], : b.shape[1]] += b
    return out


def p2_mul(a, b):
    a, b = p2(a), p2(b)
    out = np.zeros((a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i, j]:
                out[i: i + b.shape[0], j: j + b.shape[1]] += a[i, j] * b
    return out


def p2_scale(a, s):
    return p2(a) * s


def p2_eval(c, u, w):
    c = p2(c)
    u = np.asarray(u, float)
    w = np.asarray(w, float)
    out = np.zeros(np.broadcast(u, w).shape)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c[i, j]:
                out = out + c[i, j] * u**i * w**j
    return out


def p2_trim(c):
    c = p2(c)
    nz = np.argwhere(np.abs(c) > 0)
    if len(nz) == 0:
        return np.zeros((1, 1))
    return c[: nz[:, 0].max() + 1, : nz[:, 1].max() + 1]


def p2_degree(c):
    c = p2_trim(c)
    return c.shape[0] - 1, c.shape[1] - 1


def p2_compose_shear(H, p):
    """H(u, w) with w = y - p(x), u = x  ->  coefficients in (x, y).
    p: 1D power coefficients of p(x)."""
    H = p2_trim(H)
    base = p2_add([[0, 1]], -p2(np.asarray(p, float)[:, None]))  # y - p(x)
    out = np.zeros((1, 1))
    powj = np.ones((1, 1))
    for j in range(H.shape[1]):
        col = H[:, j]
        if np.any(col):
            out = p2_add(out, p2_mul(p2(col[:, None]), powj))
        powj = p2_mul(powj, base)
    return p2_trim(out)


def bernstein_from_power_2d(a, p, q):
    """Power coefficients a[i][j] in (s,t) on [0,1]^2 -> Bernstein coefficients of degree (p,q)."""
    a = p2(a)
    A = np.zeros((p + 1, q + 1))
    A[: a.shape[0], : a.shape[1]] = a[: p + 1, : q + 1]
    b = np.zeros((p + 1, q + 1))
    for k in range(p + 1):
        for l in range(q + 1):
            s = 0.0
            for i in range(k + 1):
                for j in range(l + 1):
                    s += comb(k, i) / comb(p, i) * comb(l, j) / comb(q, j) * A[i, j]
            b[k, l] = s
    return b


def p2_reparam(c, u0, u1, w0, w1):
    """c(u,w) -> coefficients in (s,t) with u = u0 + (u1-u0)s, w = w0 + (w1-w0)t."""
    c = p2_trim(c)
    out = np.zeros((1, 1))
    us = p2([[u0], [u1 - u0]])  # polynomial in s (rows)
    wt = p2([[w0, w1 - w0]])    # polynomial in t (cols)
    upow = [np.ones((1, 1))]
    for _ in range(c.shape[0]):
        upow.append(p2_mul(upow[-1], us))
    wpow = [np.ones((1, 1))]
    for _ in range(c.shape[1]):
        wpow.append(p2_mul(wpow[-1], wt))
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c[i, j]:
                out = p2_add(out, c[i, j] * p2_mul(upow[i], wpow[j]))
    return out


def sheared_heightfield_nurbs(H, shear, ur, wr, label, degree=None):
    """Polynomial surface S(u,w) = (u, w + p(u), H(u,w)) on [u0,u1]x[w0,w1] as an exact
    Bezier patch. Its implicit form is z = H(x, y - p(x)) (a polynomial height field).
    degree: optional (p,q) >= the polynomial degrees (Bezier degree elevation is exact)."""
    H = p2_trim(H)
    pc = np.asarray(shear if shear is not None else [0.0], float)
    X = p2([[0], [1]])
    Y = p2_add([[0, 1]], p2(pc[:, None]))
    Z = H
    dp = max(p2_degree(X)[0], p2_degree(Y)[0], p2_degree(Z)[0], 1)
    dq = max(p2_degree(X)[1], p2_degree(Y)[1], p2_degree(Z)[1], 1)
    if degree is not None:
        assert degree[0] >= dp and degree[1] >= dq, (label, degree, dp, dq)
        dp, dq = degree
    net = np.zeros((dp + 1, dq + 1, 4))
    for k, C in enumerate((X, Y, Z)):
        net[:, :, k] = bernstein_from_power_2d(p2_reparam(C, ur[0], ur[1], wr[0], wr[1]), dp, dq)
    net[:, :, 3] = 1.0
    U = [0.0] * (dp + 1) + [1.0] * (dp + 1)
    V = [0.0] * (dq + 1) + [1.0] * (dq + 1)
    truth = {"kind": "heightfield", "coeffs": p2_compose_shear(H, pc).tolist()}

    def inside(P, tol):
        u = P[:, 0]
        w = P[:, 1] - np.polyval(pc[::-1], u)
        return (u >= ur[0] - tol) & (u <= ur[1] + tol) & (w >= wr[0] - tol) & (w <= wr[1] + tol)

    s = Nurbs(dp, dq, U, V, net, truth, inside, label)
    s.H, s.shear, s.ur, s.wr = H, pc, ur, wr
    return s


def sheared_point(H, shear, u, w):
    pc = np.asarray(shear if shear is not None else [0.0], float)
    u = np.asarray(u, float)
    w = np.asarray(w, float)
    return np.column_stack([u, w + np.polyval(pc[::-1], u), p2_eval(H, u, w)])


# ----------------------------------------------------------------------------
# ellipsoid (affine image of the rational sphere: exact NURBS)


def ellipsoid_truth(frame, a, b, c):
    return {"kind": "ellipsoid", "center": list(frame.o), "axes": [list(frame.x), list(frame.y), list(frame.z)],
            "abc": [a, b, c]}


def nurbs_ellipsoid(frame, a, b, c, label):
    sph = nurbs_sphere(Frame(), 1.0, (0, 360), (-90, 90), label)
    net = sph.net.copy()
    w = net[:, :, 3:4]
    P = net[:, :, :3] / w
    P = P * np.array([a, b, c])
    Pw = frame.to_world(P.reshape(-1, 3)).reshape(P.shape)
    net[:, :, :3] = Pw * w
    truth = ellipsoid_truth(frame, a, b, c)
    return Nurbs(sph.p, sph.q, sph.U, sph.V, net, truth, lambda P_, tol: np.ones(len(P_), bool), label)


# ----------------------------------------------------------------------------
# general polynomial patches, multi-span degree elevation, more exact shapes


def poly_patch(X, Y, Z, ur, vr, label, truth, inside, degree=None):
    """S(u,v) = (X(u,v), Y(u,v), Z(u,v)) with X, Y, Z 2D power-coefficient arrays,
    on [u0,u1]x[v0,v1], as one exact Bezier patch (optionally degree-elevated)."""
    X, Y, Z = p2_trim(X), p2_trim(Y), p2_trim(Z)
    dp = max(p2_degree(C)[0] for C in (X, Y, Z))
    dq = max(p2_degree(C)[1] for C in (X, Y, Z))
    dp, dq = max(dp, 1), max(dq, 1)
    if degree is not None:
        assert degree[0] >= dp and degree[1] >= dq, (label, degree, dp, dq)
        dp, dq = degree
    net = np.zeros((dp + 1, dq + 1, 4))
    for k, C in enumerate((X, Y, Z)):
        net[:, :, k] = bernstein_from_power_2d(p2_reparam(C, ur[0], ur[1], vr[0], vr[1]), dp, dq)
    net[:, :, 3] = 1.0
    U = [0.0] * (dp + 1) + [1.0] * (dp + 1)
    V = [0.0] * (dq + 1) + [1.0] * (dq + 1)
    s = Nurbs(dp, dq, U, V, net, truth, inside, label)
    s.X, s.Y, s.Z, s.ur, s.vr = X, Y, Z, ur, vr
    return s


def poly_eval(s, u, v):
    return np.column_stack([p2_eval(s.X, u, v), p2_eval(s.Y, u, v), p2_eval(s.Z, u, v)])


def _elevate_dir(net, knots, p, t):
    """Elevate a B-spline whose interior knots all have multiplicity p (C0 Bezier
    pieces) by t, along axis 0 of net. Returns (net, knots)."""
    n = net.shape[0]
    segs = (n - 1) // p
    assert segs * p == n - 1, "elevate_dir needs C0 Bezier pieces"
    pieces = []
    for k in range(segs):
        blk = net[k * p: k * p + p + 1]
        el = np.stack([bezier_elevate(blk[:, j, :], t) for j in range(blk.shape[1])], axis=1)
        pieces.append(el if k == 0 else el[1:])
    new = np.concatenate(pieces, axis=0)
    uniq = sorted(set(np.round(knots, 14)))
    q = p + t
    K = [uniq[0]] * (q + 1)
    for x in uniq[1:-1]:
        K += [x] * q
    K += [uniq[-1]] * (q + 1)
    return new, np.array(K, float)


def elevate_any(s, tu, tv):
    """Exact degree elevation of a NURBS whose interior knots have full multiplicity
    (as built by arc_ctrl / revolve_nurbs / Bezier patches)."""
    if tu:
        s.net, s.U = _elevate_dir(s.net, s.U, s.p, tu)
        s.p += tu
    if tv:
        net, V = _elevate_dir(np.transpose(s.net, (1, 0, 2)), s.V, s.q, tv)
        s.net = np.transpose(net, (1, 0, 2))
        s.V = V
        s.q += tv
    return s


def quadric_truth(frame, a, b, c, signs):
    """sum_i signs[i] * (L_i / abc_i)^2 = 1 in the frame's local coordinates."""
    return {"kind": "quadric", "center": list(frame.o), "axes": [list(frame.x), list(frame.y), list(frame.z)],
            "abc": [a, b, c], "signs": list(signs)}


def nurbs_hyperboloid(frame, a, c, Zr, angle, label):
    """One-sheet hyperboloid of revolution rho^2/a^2 - z^2/c^2 = 1 about frame.z,
    z in [-Zr, Zr] (symmetric), as an exact rational biquadratic surface of revolution."""
    rho0 = a * math.sqrt(1 + (Zr / c) ** 2)
    # meridian hyperbola arc: (rho0,-Zr) w=1, (a^2/rho0, 0) w=rho0/a, (rho0, Zr) w=1
    mer = np.array([[rho0, -Zr], [a * a / rho0, 0.0], [rho0, Zr]])
    mw = np.array([1.0, rho0 / a, 1.0])
    truth = quadric_truth(frame, a, a, c, (1, 1, -1))

    def inside(P, tol):
        L = frame.to_local(P)
        th = np.degrees(np.arctan2(L[:, 1], L[:, 0]))
        return angle_in(th, *angle, tol=1e-7) & (np.abs(L[:, 2]) <= Zr + tol)
    return revolve_nurbs(frame, mer, mw, np.array([0, 0, 0, 1, 1, 1.0]), 2, angle, truth, inside, label)


def lobed_cylinder(frame, r, half_lengths, label):
    """Full rational cylinder of radius r about frame.z whose axial extent varies
    with the angle: the j-th control point column of the 4-arc circle gets axial
    half-length half_lengths[i] (9 values, homogeneous blend), so the patch has
    lobes. Every point lies exactly on the cylinder; v in [-1, 1] maps to
    z = v * l(theta)."""
    cp, cw, ck = arc_ctrl(0, 360)
    L = np.asarray(half_lengths, float)
    assert len(L) == len(cp)
    net = np.zeros((len(cp), 2, 4))
    for i in range(len(cp)):
        for j, v in enumerate((-1.0, 1.0)):
            loc = np.array([[r * cp[i, 0], r * cp[i, 1], v * L[i]]])
            P = frame.to_world(loc)[0]
            net[i, j] = [*(P * cw[i]), cw[i]]
    truth = {"kind": "cylinder", "point": list(frame.o), "axis": list(frame.z), "radius": r}
    s = Nurbs(2, 1, ck, [0, 0, 1, 1.0], net, truth, None, label)

    def half_length_at(theta):
        # l(theta): blend of L along the circle at the parameter that reaches angle theta
        th = np.mod(np.asarray(theta, float), 2 * math.pi)
        out = np.empty_like(th)
        for k, a in enumerate(np.atleast_1d(th)):
            lo, hi = 0.0, 1.0
            for _ in range(60):
                m = 0.5 * (lo + hi)
                su, Nu = bspline_basis(s.U, 2, m)
                W = Nu @ cw[su - 2: su + 1]
                P2 = (Nu * cw[su - 2: su + 1]) @ cp[su - 2: su + 1] / W
                ang = math.atan2(P2[1], P2[0]) % (2 * math.pi)
                if m > 0.999999 and ang < 1e-9:
                    ang = 2 * math.pi
                if ang < a:
                    lo = m
                else:
                    hi = m
            su, Nu = bspline_basis(s.U, 2, 0.5 * (lo + hi))
            out.flat[k] = (Nu * cw[su - 2: su + 1]) @ L[su - 2: su + 1] / (Nu @ cw[su - 2: su + 1])
        return out

    def inside(P, tol):
        Lc = frame.to_local(P)
        th = np.arctan2(Lc[:, 1], Lc[:, 0])
        return np.abs(Lc[:, 2]) <= half_length_at(th) + tol
    s._inside = inside
    s.half_length_at = half_length_at
    return s


# ----------------------------------------------------------------------------
# distance to a rational polynomial patch given by homogeneous power coefficients


def _p2_du(c):
    c = p2(c)
    if c.shape[0] == 1:
        return np.zeros((1, c.shape[1]))
    return c[1:] * np.arange(1, c.shape[0])[:, None]


def _p2_dv(c):
    c = p2(c)
    if c.shape[1] == 1:
        return np.zeros((c.shape[0], 1))
    return c[:, 1:] * np.arange(1, c.shape[1])[None, :]


def param_eval(truth, u, v):
    H = [p2(c) for c in truth["hom"]]
    h = [p2_eval(c, u, v) for c in H]
    return np.stack([h[0] / h[3], h[1] / h[3], h[2] / h[3]], axis=-1)


def param_derivs(truth, u, v):
    H = [p2(c) for c in truth["hom"]]
    h = np.array([p2_eval(c, u, v) for c in H])
    hu = np.array([p2_eval(_p2_du(c), u, v) for c in H])
    hv = np.array([p2_eval(_p2_dv(c), u, v) for c in H])
    S = h[:3] / h[3]
    Su = (hu[:3] - S * hu[3]) / h[3]
    Sv = (hv[:3] - S * hv[3]) / h[3]
    return S, Su, Sv


def param_distance(truth, P, n_seed=41):
    """Distance from points to the patch: vectorised Gauss-Newton closest point
    from the nearest grid seed (clamped to the domain); points that do not
    converge to a tiny distance are retried from further seeds (the patch may
    overlap itself)."""
    P = np.atleast_2d(np.asarray(P, float))
    d0 = truth["domain"]
    us = np.linspace(d0[0], d0[1], n_seed)
    vs = np.linspace(d0[2], d0[3], n_seed)
    UU, VV = np.meshgrid(us, vs, indexing="ij")
    Gr = param_eval(truth, UU.ravel(), VV.ravel())
    D2 = np.sum((P[:, None, :] - Gr[None, :, :]) ** 2, axis=2) if len(P) * len(Gr) < 4e7 else None
    order = np.argsort(D2, axis=1)[:, :12] if D2 is not None else None
    out = np.full(len(P), np.inf)
    for rank in range(12):
        todo = np.nonzero(out > 1e-12)[0]
        if not len(todo):
            break
        if order is not None:
            idx = order[todo, rank]
        else:
            idx = np.array([np.argsort(np.sum((Gr - P[i]) ** 2, axis=1))[rank] for i in todo])
        u = UU.ravel()[idx].copy()
        v = VV.ravel()[idx].copy()
        X = P[todo]
        for _ in range(40):
            S, Su, Sv = param_derivs(truth, u, v)
            S, Su, Sv = S.T if S.ndim > 1 else S, Su.T if Su.ndim > 1 else Su, Sv.T if Sv.ndim > 1 else Sv
            r = S - X
            a11 = np.sum(Su * Su, 1)
            a12 = np.sum(Su * Sv, 1)
            a22 = np.sum(Sv * Sv, 1)
            b1 = -np.sum(Su * r, 1)
            b2 = -np.sum(Sv * r, 1)
            det = a11 * a22 - a12 * a12
            det = np.where(np.abs(det) < 1e-300, 1e-300, det)
            du = (a22 * b1 - a12 * b2) / det
            dv = (a11 * b2 - a12 * b1) / det
            u = np.clip(u + du, d0[0], d0[1])
            v = np.clip(v + dv, d0[2], d0[3])
        dist = np.linalg.norm(param_eval(truth, u, v) - X, axis=1)
        out[todo] = np.minimum(out[todo], dist)
    return out


def _gn_project(truth, X, u, v, d0):
    if True:
        for _ in range(60):
            S, Su, Sv = param_derivs(truth, u, v)
            r = S - X
            J = np.column_stack([Su, Sv])
            try:
                du, dv = np.linalg.lstsq(J, -r, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            un = min(max(u + du, d0[0]), d0[1])
            vn = min(max(v + dv, d0[2]), d0[3])
            if abs(un - u) + abs(vn - v) < 1e-16:
                u, v = un, vn
                break
            u, v = un, vn
        return float(np.linalg.norm(param_eval(truth, u, v) - X))


def rational_poly_patch(Hx, Hy, Hz, Hw, ur, vr, label, inside=None, degree=None):
    """Patch with homogeneous coordinates (Hx, Hy, Hz, Hw) polynomial in (u,v)
    (power coefficients) on [u0,u1]x[v0,v1], as one exact rational Bezier patch.
    Its truth is the parametric patch itself (distance by closest point)."""
    Hs = [p2_trim(c) for c in (Hx, Hy, Hz, Hw)]
    if Hs[3].shape == (1, 1):  # constant weight: plain polynomial patch
        w0 = float(Hs[3][0, 0])
        Hs = [c / w0 for c in Hs]
    dp = max(max(p2_degree(C)[0] for C in Hs), 1)
    dq = max(max(p2_degree(C)[1] for C in Hs), 1)
    if degree is not None:
        assert degree[0] >= dp and degree[1] >= dq
        dp, dq = degree
    net = np.zeros((dp + 1, dq + 1, 4))
    for k, C in enumerate(Hs):
        net[:, :, k] = bernstein_from_power_2d(p2_reparam(C, ur[0], ur[1], vr[0], vr[1]), dp, dq)
    assert (net[:, :, 3] > 0).all(), "non-positive weight"
    U = [0.0] * (dp + 1) + [1.0] * (dp + 1)
    V = [0.0] * (dq + 1) + [1.0] * (dq + 1)
    truth = {"kind": "param", "hom": [c.tolist() for c in Hs], "domain": [ur[0], ur[1], vr[0], vr[1]]}
    s = Nurbs(dp, dq, U, V, net, truth, inside or (lambda P_, tol: np.ones(len(P_), bool)), label)
    s.hom, s.ur, s.vr = Hs, ur, vr
    return s
