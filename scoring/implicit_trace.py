"""Ground truth for torus x implicit-surface intersections (own algebra, numpy only).

The torus is parameterised by (theta, phi) (major, minor angle; phi = 0 on the
outer equator). Substituting it into the other surface's implicit equation E
gives F(theta, phi) = E(T(theta, phi)); the intersection is the zero set of F on
the periodic square. We extract it with marching squares on a fine grid, refine
every vertex to F = 0 by bisection along its grid edge (machine precision),
and locate tangency points (F = dF/dtheta = dF/dphi = 0) with Newton's method.
Crossing tangency points are spliced into the traced chains; tangency points
with no curve through them are isolated contact points.
"""
import math

import numpy as np


class TorusImplicit:
    def __init__(self, frame, R, r, E, gradE):
        self.f, self.R, self.r, self.E, self.gradE = frame, R, r, E, gradE

    def T(self, th, ph):
        th = np.asarray(th, float)
        ph = np.asarray(ph, float)
        rho = self.R + self.r * np.cos(ph)
        L = np.stack([rho * np.cos(th), rho * np.sin(th), self.r * np.sin(ph)], axis=-1)
        return self.f.o + L[..., :1] * self.f.x + L[..., 1:2] * self.f.y + L[..., 2:3] * self.f.z

    def dT(self, th, ph):
        rho = self.R + self.r * np.cos(ph)
        Tt = (-rho * np.sin(th))[..., None] * self.f.x + (rho * np.cos(th))[..., None] * self.f.y
        Tp = (-self.r * np.sin(ph) * np.cos(th))[..., None] * self.f.x + \
             (-self.r * np.sin(ph) * np.sin(th))[..., None] * self.f.y + (self.r * np.cos(ph))[..., None] * self.f.z
        return Tt, Tp

    def F(self, th, ph):
        return self.E(self.T(th, ph))

    def gradF(self, th, ph):
        P = self.T(th, ph)
        g = self.gradE(P)
        Tt, Tp = self.dT(np.asarray(th, float), np.asarray(ph, float))
        return np.sum(g * Tt, axis=-1), np.sum(g * Tp, axis=-1)


def _bisect_edge(F, a, b, fa, fb):
    """a, b: (th, ph) with F of opposite sign; returns zero point."""
    for _ in range(60):
        m = 0.5 * (a + b)
        fm = F(m[0], m[1])
        if fm == 0:
            return m
        if (fm > 0) == (fa > 0):
            a, fa = m, fm
        else:
            b, fb = m, fm
    return 0.5 * (a + b)


def marching_squares(ti, N):
    two_pi = 2 * math.pi
    h = two_pi / N
    g = np.arange(N) * h
    TH, PH = np.meshgrid(g, g, indexing="ij")
    Fv = ti.F(TH, PH)
    Fv = np.where(Fv == 0, 1e-300, Fv)
    pos = Fv > 0
    # edges with sign change: horizontal (i,j)-(i+1,j), vertical (i,j)-(i,j+1), periodic
    hx = pos != np.roll(pos, -1, axis=0)
    vx = pos != np.roll(pos, -1, axis=1)
    pts = {}

    def edge_point(key):
        if key in pts:
            return pts[key]
        kind, i, j = key
        a = np.array([i * h, j * h])
        b = a + (np.array([h, 0]) if kind == "h" else np.array([0, h]))
        fa = Fv[i, j]
        fb = Fv[(i + 1) % N, j] if kind == "h" else Fv[i, (j + 1) % N]
        p = _bisect_edge(ti.F, a, b, fa, fb)
        pts[key] = p
        return p

    adj = {}

    def link(k1, k2):
        adj.setdefault(k1, []).append(k2)
        adj.setdefault(k2, []).append(k1)

    I, J = np.nonzero(hx[:, :] | np.roll(hx, -1, axis=1) | vx | np.roll(vx, -1, axis=0))
    for i, j in zip(I, J):
        i1, j1 = (i + 1) % N, (j + 1) % N
        e = []
        if hx[i, j]:
            e.append(("h", i, j))      # bottom
        if vx[i1, j]:
            e.append(("v", i1, j))     # right
        if hx[i, j1]:
            e.append(("h", i, j1))     # top
        if vx[i, j]:
            e.append(("v", i, j))      # left
        if len(e) == 2:
            link(e[0], e[1])
        elif len(e) == 4:
            centre = ti.F(i * h + h / 2, j * h + h / 2)
            # pair edges so that the centre's sign region is connected
            if (centre > 0) == pos[i, j]:
                link(e[0], e[1])
                link(e[2], e[3])
            else:
                link(e[0], e[3])
                link(e[1], e[2])
    # walk chains
    seen = set()
    chains = []
    for start in adj:
        if start in seen:
            continue
        chain = [start]
        seen.add(start)
        prev, cur = None, start
        closed = False
        while True:
            nxt = [k for k in adj[cur] if k != prev]
            if not nxt:
                break
            k = nxt[0]
            if k == start:
                closed = True
                break
            if k in seen:
                break
            chain.append(k)
            seen.add(k)
            prev, cur = cur, k
        if not closed and len(adj[start]) == 2:
            # extend backwards from start
            prev, cur = chain[1] if len(chain) > 1 else None, start
            back = []
            while True:
                nxt = [k for k in adj[cur] if k != prev and k not in seen]
                if not nxt:
                    break
                k = nxt[0]
                back.append(k)
                seen.add(k)
                prev, cur = cur, k
            chain = back[::-1] + chain
        P = np.array([edge_point(k) for k in chain])
        chains.append((P, closed))
    return chains, Fv, h


def unwrap(P):
    """Unwrap a periodic (theta, phi) chain so that consecutive points are close."""
    Q = P.copy()
    d = np.diff(Q, axis=0)
    d = (d + math.pi) % (2 * math.pi) - math.pi
    Q[1:] = Q[0] + np.cumsum(d, axis=0)
    return Q


def singular_points(ti, Fv, h, N, rel=2e-2):
    """Newton on (F_theta, F_phi) = 0 from grid nodes where |F| is a local
    minimum (catches crossings and isolated contacts)."""
    A = np.abs(Fv)
    scale = A.max()
    cand = (A <= np.roll(A, 1, 0)) & (A <= np.roll(A, -1, 0)) & (A <= np.roll(A, 1, 1)) & \
           (A <= np.roll(A, -1, 1)) & (A < rel * scale)
    gt, gp = ti.gradF(*np.meshgrid(np.arange(N) * h, np.arange(N) * h, indexing="ij"))
    G = np.hypot(gt, gp)
    cand2 = (G <= np.roll(G, 1, 0)) & (G <= np.roll(G, -1, 0)) & (G <= np.roll(G, 1, 1)) & \
            (G <= np.roll(G, -1, 1)) & (A < rel * scale)
    I, J = np.nonzero(cand | cand2)
    found = []
    for i, j in zip(I, J):
        x = np.array([i * h, j * h])
        ok = False
        for _ in range(60):
            g = np.array(ti.gradF(x[0], x[1]))
            e = 1e-6
            H = np.zeros((2, 2))
            for k in range(2):
                dx = np.zeros(2)
                dx[k] = e
                H[:, k] = (np.array(ti.gradF(*(x + dx))) - np.array(ti.gradF(*(x - dx)))) / (2 * e)
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            x = x - step
            if np.linalg.norm(step) < 1e-14:
                ok = True
                break
        if not ok:
            continue
        Fx = float(ti.F(x[0], x[1]))
        if abs(Fx) < 1e-9 * max(1.0, scale):
            x = np.mod(x, 2 * math.pi)
            if all(np.linalg.norm(((x - y) + math.pi) % (2 * math.pi) - math.pi) > 1e-6 for y in found):
                found.append(x)
    return found


def trace(ti, N=1600):
    chains, Fv, h = marching_squares(ti, N)
    sing = singular_points(ti, Fv, h, N)
    branches = []
    for P, closed in chains:
        Q = unwrap(P)
        branches.append([Q, closed])
    crossing, isolated = [], []
    for s in sing:
        hit = False
        for b in branches:
            Q = b[0]
            d = (Q - s + math.pi) % (2 * math.pi) - math.pi
            dist = np.hypot(d[:, 0], d[:, 1])
            k = int(np.argmin(dist))
            if dist[k] < 3 * h:
                hit = True
                near = dist < 1.5 * h
                near[k] = True
                idx = np.nonzero(near)[0]
                # replace the run of vertices next to the singular point by the point itself
                lo, hi = idx.min(), idx.max()
                if hi - lo < 10:
                    ins = Q[k] - d[k]  # s, unwrapped next to the chain
                    b[0] = np.vstack([Q[:lo], ins[None, :], Q[hi + 1:]])
        (crossing if hit else isolated).append(s)
    out = []
    for Q, closed in branches:
        P = ti.T(Q[:, 0], Q[:, 1])
        if closed:
            P = np.vstack([P, P[:1]])
        out.append((P, closed))
    return out, [ti.T(*s) for s in crossing], [ti.T(*s) for s in isolated]


# ----------------------------------------------------------------------------
# non-periodic version: zero set of F(u, v) on a rectangle (NaN = undefined)


def marching_rect(F, ur, vr, N, M=None):
    """Chains of the zero set of F on [u0,u1]x[v0,v1] (N x M cells). F is
    vectorised; NaN marks points where F is undefined (cells touching them are
    skipped). Vertices are bisection-refined on grid edges. Returns a list of
    (points (k,2), closed)."""
    M = M or N
    us = np.linspace(ur[0], ur[1], N + 1)
    vs = np.linspace(vr[0], vr[1], M + 1)
    UU, VV = np.meshgrid(us, vs, indexing="ij")
    Fv = F(UU, VV)
    ok = np.isfinite(Fv)
    Fv = np.where(Fv == 0, 1e-300, Fv)
    pos = Fv > 0
    pts = {}

    def edge_point(key):
        if key in pts:
            return pts[key]
        kind, i, j = key
        a = np.array([us[i], vs[j]])
        b = np.array([us[i + 1], vs[j]]) if kind == "h" else np.array([us[i], vs[j + 1]])
        fa = Fv[i, j]
        fb = Fv[i + 1, j] if kind == "h" else Fv[i, j + 1]
        Fs = lambda x, y: float(F(np.array(x), np.array(y)))
        p = _bisect_edge(Fs, a, b, fa, fb)
        pts[key] = p
        return p

    adj = {}

    def link(k1, k2):
        adj.setdefault(k1, []).append(k2)
        adj.setdefault(k2, []).append(k1)

    for i in range(N):
        for j in range(M):
            if not (ok[i, j] and ok[i + 1, j] and ok[i, j + 1] and ok[i + 1, j + 1]):
                continue
            e = []
            if pos[i, j] != pos[i + 1, j]:
                e.append(("h", i, j))
            if pos[i + 1, j] != pos[i + 1, j + 1]:
                e.append(("v", i + 1, j))
            if pos[i, j + 1] != pos[i + 1, j + 1]:
                e.append(("h", i, j + 1))
            if pos[i, j] != pos[i, j + 1]:
                e.append(("v", i, j))
            if len(e) == 2:
                link(e[0], e[1])
            elif len(e) == 4:
                centre = float(F(np.array(0.5 * (us[i] + us[i + 1])), np.array(0.5 * (vs[j] + vs[j + 1]))))
                if (centre > 0) == pos[i, j]:
                    link(e[0], e[1])
                    link(e[2], e[3])
                else:
                    link(e[0], e[3])
                    link(e[1], e[2])
    seen = set()
    chains = []
    # start open chains at their ends first
    starts = [k for k in adj if len(adj[k]) == 1] + [k for k in adj if len(adj[k]) != 1]
    for start in starts:
        if start in seen:
            continue
        chain = [start]
        seen.add(start)
        prev, cur, closed = None, start, False
        while True:
            nxt = [k for k in adj[cur] if k != prev]
            if not nxt:
                break
            k = nxt[0]
            if k == start:
                closed = True
                break
            if k in seen:
                break
            chain.append(k)
            seen.add(k)
            prev, cur = cur, k
        chains.append((np.array([edge_point(k) for k in chain]), closed))
    return chains
