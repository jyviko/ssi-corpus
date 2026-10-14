"""Small derivative-free optimiser used by fit_t2_ex4/5/6.py: cyclic coordinate search with golden-section
line searches and step adaptation. Robust to the piecewise-constant objectives of mask IoU."""
import numpy as np


def coord_search(f, x0, steps, cycles=6, shrink=0.5, verbose=True, names=None):
    x = np.array(x0, float)
    st = np.array(steps, float)
    fx = f(x)
    for c in range(cycles):
        improved = 0
        for i in range(len(x)):
            best = (fx, x[i])
            for mult in (1.0, -1.0, 2.0, -2.0, 0.5, -0.5):
                xi = x.copy()
                xi[i] = x[i] + mult * st[i]
                fi = f(xi)
                if fi < best[0] - 1e-9:
                    best = (fi, xi[i])
            if best[0] < fx:
                # continue in the improving direction while it pays
                d = best[1] - x[i]
                x[i] = best[1]
                fx = best[0]
                for _ in range(6):
                    xi = x.copy()
                    xi[i] += d
                    fi = f(xi)
                    if fi < fx - 1e-9:
                        x, fx = xi, fi
                    else:
                        break
                improved += 1
        if verbose:
            print("cycle %d  f=%.5f  improved %d/%d" % (c, fx, improved, len(x)), flush=True)
        if improved == 0:
            st *= shrink
        else:
            st *= 0.8
    return x, fx


def independent_check(Af, Bf, domA, domB, truth, n=160, iters=40, h=1e-7):
    """Independent intersection search, used only as a check of a closed-form truth.

    Seeds: pairs of nearby samples of the two patches (n x n grids, KD-tree), then Gauss-Newton on
    |A(s,t) - B(u,v)|^2 over the 4 parameters (clamped to the domains). Converged points (|A - B| < 1e-9)
    are compared with the truth polyline set: returns (number of converged points, max distance of a
    converged point to the truth, max distance of a truth point to the converged cloud)."""
    from scipy.spatial import cKDTree
    sa = np.linspace(*domA[0], n)
    ta = np.linspace(*domA[1], n)
    sb = np.linspace(*domB[0], n)
    tb = np.linspace(*domB[1], n)
    SA, TA = np.meshgrid(sa, ta, indexing="ij")
    SB, TB = np.meshgrid(sb, tb, indexing="ij")
    PA = Af(SA.ravel(), TA.ravel()).reshape(-1, 3)
    PB = Bf(SB.ravel(), TB.ravel()).reshape(-1, 3)
    ext = np.ptp(np.vstack([PA, PB]), axis=0).max()
    step = 3.0 * ext / n
    kd = cKDTree(PB)
    d, j = kd.query(PA)
    sel = np.nonzero(d < step)[0]
    X = np.stack([SA.ravel()[sel], TA.ravel()[sel], SB.ravel()[j[sel]], TB.ravel()[j[sel]]], 1)
    lo = np.array([domA[0][0], domA[1][0], domB[0][0], domB[1][0]])
    hi = np.array([domA[0][1], domA[1][1], domB[0][1], domB[1][1]])
    for _ in range(iters):
        r = Af(X[:, 0], X[:, 1]) - Bf(X[:, 2], X[:, 3])
        J = np.zeros((len(X), 3, 4))
        for k in range(4):
            e = np.zeros(4)
            e[k] = h
            Xp, Xm = X + e, X - e
            rp = Af(Xp[:, 0], Xp[:, 1]) - Bf(Xp[:, 2], Xp[:, 3])
            rm_ = Af(Xm[:, 0], Xm[:, 1]) - Bf(Xm[:, 2], Xm[:, 3])
            J[:, :, k] = (rp - rm_) / (2 * h)
        # minimum-norm Gauss-Newton step
        dx = np.einsum("nki,nk->ni", J, r)
        JJt = np.einsum("nki,nli->nkl", J, J) + 1e-14 * np.eye(3)
        y = np.linalg.solve(JJt, r[..., None])[..., 0]
        dx = -np.einsum("nki,nk->ni", J, y)
        X = np.clip(X + dx, lo, hi)
    res = np.linalg.norm(Af(X[:, 0], X[:, 1]) - Bf(X[:, 2], X[:, 3]), axis=1)
    ok = res < 1e-9
    P = Af(X[ok, 0], X[ok, 1]).reshape(-1, 3)
    T = np.vstack(truth)
    kt = cKDTree(T)
    d1 = kt.query(P)[0] if len(P) else np.zeros(1)
    d2 = cKDTree(P).query(T)[0] if len(P) else np.full(len(T), np.inf)
    return int(ok.sum()), float(d1.max()), float(d2.max()), P


def paper(stem, panel):
    import os
    from PIL import Image
    here = os.path.dirname(os.path.abspath(__file__))
    return np.asarray(Image.open(os.path.join(here, "paper", "%s_%s.png" % (stem, panel))).convert("RGB"))


def fit_framing(rm, grids, cam, P, alphas=(1.0, 1.0), n=41):
    """Refit only scale / cx / cy of cam to the union silhouette of panel P (same view direction)."""
    from scipy import optimize
    H, W = P.shape[:2]
    fg = rm.paper_mask(P)

    def f(v):
        c = dict(cam)
        c.update(scale=v[0], cx=v[1], cy=v[2])
        _, m = rm.render(grids, c, W, H, ss=1, n=n, alphas=alphas)
        return 1 - rm.mask_iou(fg, m[0] | m[1])
    r = optimize.minimize(f, [cam["scale"], W / 2, H / 2], method="Nelder-Mead", options=dict(maxiter=300))
    c = dict(cam)
    c.update(scale=float(r.x[0]), cx=float(r.x[1]), cy=float(r.x[2]))
    return c, 1 - float(r.fun)


def ious(rm, P, rgb, masks):
    fg = rm.paper_mask(P)
    pb, po, _ = rm.color_masks(P)
    ob, oo, _ = rm.color_masks(rgb)
    return (round(rm.mask_iou(fg, masks[0] | masks[1]), 4), round(rm.mask_iou(pb, ob), 4),
            round(rm.mask_iou(po, oo), 4))


def fit_camera(rm, grids, P, alphas, starts=((20, -60), (20, -120), (20, 60), (20, 120), (60, -90), (-20, -90),
                                             (-20, 90), (0, 0), (0, 180), (80, -90)), iters=250, n=41):
    """Best orthographic camera (el, az, roll, scale, cx, cy) for given surfaces against panel P, by mean IoU
    (union, blue-dominant, orange-dominant); used for the IoU of the previous reconstruction ("before")."""
    from scipy import optimize
    H, W = P.shape[:2]
    G = [g if g.shape[0] == n else g[:: max(1, g.shape[0] // n), :: max(1, g.shape[1] // n)] for g in grids]
    allp = np.vstack([g.reshape(-1, 3) for g in G])
    c0 = 0.5 * (allp.min(0) + allp.max(0))
    ext = np.ptp(allp, axis=0).max()

    def mk(v):
        return dict(proj="ortho", el=v[0], az=v[1], roll=v[2], target=c0.tolist(), scale=v[3], cx=v[4], cy=v[5])

    def f(v):
        rgb, m = rm.render(G, mk(v), W, H, ss=1, n=n, alphas=alphas)
        return 1 - float(np.mean(ious(rm, P, rgb, m)))
    best = None
    for el, az in starts:
        for roll in (0.0, 90.0, -90.0):
            v0 = [el, az, roll, 0.8 * min(W, H) / ext, W / 2, H / 2]
            r = optimize.minimize(f, v0, method="Nelder-Mead", options=dict(maxiter=iters, adaptive=True))
            if best is None or r.fun < best.fun:
                best = r
    return mk(best.x), 1 - float(best.fun)


def curves_only(rm, curves, cam, W, H, points=(), width=5, radius=4, ss=3):
    """The paper's result-row picture: red curves (and dots) on white, no surfaces."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (W * ss, H * ss), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    for C in curves:
        x, y, _ = rm.project(np.asarray(C, float), cam, W, H)
        dr.line(list(zip(x * ss, y * ss)), fill=(235, 0, 0), width=int(width * ss), joint="curve")
    for p in points:
        x, y, _ = rm.project(np.asarray(p, float), cam, W, H)
        R = radius * ss
        dr.ellipse([x * ss - R, y * ss - R, x * ss + R, y * ss + R], fill=(235, 0, 0))
    return np.asarray(im.resize((W, H), Image.LANCZOS))
