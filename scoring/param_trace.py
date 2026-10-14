"""Intersection of two parametric patches P(u,v) and Q(s,t) by numerical
continuation in (u,v,s,t) (own code, numpy only).

The curve is the solution set of G(u,v,s,t) = P(u,v) - Q(s,t) = 0 (3 equations,
4 unknowns) inside the box [u0,u1]x[v0,v1]x[s0,s1]x[t0,t1]. Starting from seed
points, every point is corrected with Gauss-Newton (minimum-norm steps, to
machine precision), the curve is followed along the null vector of the
Jacobian, and where it leaves the box the exit point is solved exactly with
the active bound as a fourth equation. Derivatives use the complex step, so P
and Q must be written with complex-safe numpy operations.
"""
import numpy as np

H_CS = 1e-30


def jac(G, X):
    J = np.zeros((3, 4))
    for k in range(4):
        dx = np.zeros(4, complex)
        dx[k] = 1j * H_CS
        J[:, k] = np.imag(G(X + dx)) / H_CS
    return J


def correct(G, X, iters=50, fix=None):
    """Newton to G=0 (and X[fix[0]] = fix[1] if given)."""
    X = X.astype(float).copy()
    for _ in range(iters):
        g = np.real(G(X.astype(complex)))
        J = jac(G, X)
        if fix is not None:
            k, val = fix
            g = np.append(g, X[k] - val)
            row = np.zeros(4)
            row[k] = 1.0
            J = np.vstack([J, row])
            try:
                step = np.linalg.solve(J, g)
            except np.linalg.LinAlgError:
                return X, False
        else:
            step = np.linalg.lstsq(J, g, rcond=None)[0]
        X = X - step
        if np.linalg.norm(step) < 1e-15 * max(1.0, np.linalg.norm(X)):
            break
    res = np.linalg.norm(np.real(G(X.astype(complex))))
    return X, res < 1e-12


def tangent(G, X, prev=None):
    J = jac(G, X)
    _, sv, Vt = np.linalg.svd(J)
    t = Vt[-1]
    if prev is not None and t @ prev < 0:
        t = -t
    return t / np.linalg.norm(t), sv


def inbox(X, box, eps=0.0):
    return all(box[k][0] - eps <= X[k] <= box[k][1] + eps for k in range(4))


def exit_point(G, Xin, Xout, box):
    """Exact point on the box boundary between Xin (inside) and Xout (outside)."""
    # which bound was crossed first along the segment
    best = None
    for k in range(4):
        for side in (0, 1):
            b = box[k][side]
            a0, a1 = Xin[k] - b, Xout[k] - b
            if (a0 >= 0) != (a1 >= 0) or a1 == 0:
                lam = a0 / (a0 - a1) if a0 != a1 else 0.0
                if best is None or lam < best[0]:
                    best = (lam, k, b)
    if best is None:
        return None
    lam, k, b = best
    X0 = Xin + lam * (Xout - Xin)
    X0[k] = b
    X, ok = correct(G, X0, fix=(k, b))
    if not ok or not inbox(X, box, 1e-9):
        return None
    X = np.array([min(max(X[i], box[i][0]), box[i][1]) if i != k else b for i in range(4)])
    return X


def follow(G, X0, box, h=2e-3, max_steps=200000, direction=1.0):
    pts = [X0]
    t0, _ = tangent(G, X0)
    t = direction * t0
    X = X0
    closed = False
    for n in range(max_steps):
        hh = h
        for _ in range(30):
            Xp = X + hh * t
            Xc, ok = correct(G, Xp, iters=30)
            if ok and np.linalg.norm(Xc - Xp) < 0.3 * hh and np.linalg.norm(Xc - X) < 2 * hh:
                tn, _ = tangent(G, Xc, t)
                if tn @ t > 0.9:
                    break
            hh *= 0.5
        else:
            return np.array(pts), False, "stalled"
        if not inbox(Xc, box):
            E = exit_point(G, X, Xc, box)
            if E is not None:
                pts.append(E)
            return np.array(pts), False, "boundary"
        X, t = Xc, tn
        pts.append(X)
        if n > 20 and np.linalg.norm(X - X0) < 1.5 * h:
            closed = True
            return np.array(pts), closed, "closed"
    return np.array(pts), False, "max_steps"


def trace_from_seeds(G, seeds, box, h=2e-3, cover_tol=5e-3):
    """Trace every component reached from the seeds; returns list of (X (k,4), closed)."""
    comps = []
    for s in seeds:
        X0, ok = correct(G, np.asarray(s, float))
        if not ok or not inbox(X0, box):
            continue
        if any(np.min(np.linalg.norm(c - X0, axis=1)) < cover_tol for c, _ in comps):
            continue
        a, closed, why_a = follow(G, X0, box, h, direction=1.0)
        if closed:
            curve = a
        else:
            b, _, why_b = follow(G, X0, box, h, direction=-1.0)
            curve = np.vstack([b[::-1], a[1:]])
        comps.append((curve, closed))
    return comps
