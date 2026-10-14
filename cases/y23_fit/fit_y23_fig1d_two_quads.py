"""Fit the Fig. 1(d) ribbon (a (2, 5) Bezier net, w in [0, 2]) to the paper panels (600 dpi crops).

Pipeline (orthographic cameras; the Ours camera is a fixed gauge: el 20, az -90, 100 px per unit):
  1. Ours-row network only: chamfer of the 7 network edges against the red pixels, both ways, plus
     control-net fairness and a depth gauge, from a net through the six corners read off the panel.
  2. Two-view stage: a second, free camera for the Surfaces row (network one way + silhouette outline).
     The second camera came out close to a shifted copy of the first; the panels are the same view
     framed 16.1 px / 2.6 px apart (their silhouettes agree to IoU 0.89 after the shift), so
  3. single camera: Ours network both ways + Surfaces outline under the shifted camera, fairness 0.3.
     The net's depth is not measured by one view; this keeps the fairest net consistent with the image.
Run: .venv/bin/python -I cases/y23_fit/fit_y23_fig1d_two_quads.py  -> prints the net as JSON
(the generator y23_fig1d_two_quads.py holds the rounded result).
"""
import json
import math
import os
import sys
import numpy as np
from PIL import Image
from scipy import ndimage, optimize
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_match as rm

STEM = 'y23_fig1d_two_quads'
OURS = np.asarray(Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper', STEM + '_ours.png')).convert('RGB'))
H, W = OURS.shape[:2]
_, _, RED = rm.color_masks(OURS)
DT = ndimage.distance_transform_edt(~RED)
ry, rx = np.nonzero(RED)
RP = np.stack([rx, ry], 1)[::6].astype(float)
CAM = dict(proj='ortho', el=20.0, az=-90.0, roll=0.0, scale=100.0, dist=20.0, cx=216.0, cy=190.0, target=[0, 0, 0])
PU, PW = 2, 5


def bern(n, t):
    t = np.asarray(t, float)[:, None]
    k = np.arange(n + 1)[None]
    return np.array([math.comb(n, i) for i in range(n + 1)])[None] * t ** k * (1 - t) ** (n - k)


def ev(net, u, w):
    """net (PU+1, PW+1, 3); u in [0,1], w in [0,2] arrays of equal length -> points."""
    Bu, Bw = bern(PU, u), bern(PW, np.asarray(w) / 2)
    return np.einsum('ni,nj,ijk->nk', Bu, Bw, net)


def network(net, n=60):
    t = np.linspace(0, 1, n)
    E = []
    for u0 in (0.0, 1.0):
        E += [ev(net, u0 + 0 * t, t), ev(net, u0 + 0 * t, 1 + t)]
    for w0 in (0.0, 1.0, 2.0):
        E.append(ev(net, t, w0 + 0 * t))
    return E


def bil(img, x, y):
    return ndimage.map_coordinates(img, [y, x], order=1, mode='nearest')


def resid(x, cam=CAM, wreg=0.3):
    net = x.reshape(PU + 1, PW + 1, 3)
    E = network(net, 40)
    P = np.vstack(E)
    px, py, _ = rm.project(P, cam, W, H)
    r1 = bil(DT, px, py)
    Pd = np.vstack(network(net, 120))
    qx, qy, _ = rm.project(Pd, cam, W, H)
    d, _ = cKDTree(np.stack([qx, qy], 1)).query(RP)
    r2 = 0.7 * d
    # fairness: second differences of the control net
    d2w = (net[:, 2:] - 2 * net[:, 1:-1] + net[:, :-2]).ravel()
    d2u = (net[2:] - 2 * net[1:-1] + net[:-2]).ravel()
    return np.concatenate([r1, r2, wreg * 30 * d2w, wreg * 30 * d2u])


def init_net():
    px2w = lambda p: np.array([(p[0] - 216) / 100.0, 0.0, -(p[1] - 190) / 100.0])
    C = {(0, 0): (37, 98), (1, 0): (115, 60), (0, 1): (163, 228), (1, 1): (262, 212), (0, 2): (282, 355),
         (1, 2): (400, 363)}
    net = np.zeros((PU + 1, PW + 1, 3))
    for i in range(PU + 1):
        for j in range(PW + 1):
            u, w = i / PU, 2 * j / PW
            if w <= 1:
                a = (1 - w) * np.array(C[(0, 0)]) + w * np.array(C[(0, 1)])
                b = (1 - w) * np.array(C[(1, 0)]) + w * np.array(C[(1, 1)])
            else:
                a = (2 - w) * np.array(C[(0, 1)]) + (w - 1) * np.array(C[(0, 2)])
                b = (2 - w) * np.array(C[(1, 1)]) + (w - 1) * np.array(C[(1, 2)])
            net[i, j] = px2w((1 - u) * a + u * b) + (np.array([0, 0, 0.6]) if i == 1 else 0)
    return net


# ---- two-view fit: ours camera fixed (gauge), surfaces camera free; depth from the second view
SURF = np.asarray(Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper', STEM + '_surfaces.png')).convert('RGB'))
_, _, RED_S = rm.color_masks(SURF)
DT_S = ndimage.distance_transform_edt(~RED_S)
rsy, rsx = np.nonzero(RED_S)
RPS = np.stack([rsx, rsy], 1)[::6].astype(float)
FG_S = rm.paper_mask(SURF)
FGE = FG_S & ~ndimage.binary_erosion(FG_S)
DT_FGE = ndimage.distance_transform_edt(~FGE)
FGE_P = np.stack(np.nonzero(FGE)[::-1], 1)[::4].astype(float)


def cam_s(c):
    return dict(proj='ortho', el=c[0], az=c[1], roll=c[2], scale=c[3], dist=20.0, cx=c[5], cy=c[6], target=[0, 0, 0])


def outline_pts(net, cam, n=90):
    u = np.linspace(0, 1, n); w = np.linspace(0, 2, int(1.6 * n))
    U, Wg = np.meshgrid(u, w, indexing='ij')
    G = ev(net, U.ravel(), Wg.ravel())
    x, y, _ = rm.project(G, cam, W, H)
    m = np.zeros((H, W), bool)
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    m[y[ok].astype(int), x[ok].astype(int)] = True
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=2))
    e = m & ~ndimage.binary_erosion(m)
    return m, e


def resid2(x, wreg=0.1, wout=0.5):
    net = x[:-7].reshape(PU + 1, PW + 1, 3)
    cs = cam_s(x[-7:])
    r = [resid(x[:-7], CAM, wreg)]
    P = np.vstack(network(net, 40))
    px, py, _ = rm.project(P, cs, W, H)
    Pd = np.vstack(network(net, 120))
    qx, qy, _ = rm.project(Pd, cs, W, H)
    d, _ = cKDTree(np.stack([qx, qy], 1)).query(RPS)
    r.append(0.7 * d)                                      # every red pixel of the Surfaces panel on some edge
    m, e = outline_pts(net, cs)
    ey, ex = np.nonzero(e)
    if len(ex):
        d2, _ = cKDTree(np.stack([ex, ey], 1)).query(FGE_P)
        r.append(wout * d2)
        de = DT_FGE[ey, ex]
        r.append(wout * de[np.linspace(0, len(de) - 1, 300).astype(int)])
    else:
        r.append(np.full(len(FGE_P) + 300, 100.0))
    # depth gauge: control-net centroid on the ours-camera target plane
    r.append([10 * net.reshape(-1, 3).mean(0)[1]])
    return np.concatenate([np.ravel(a) for a in r])


def stage3(z, wreg=0.3):
    CS = [CAM['el'], CAM['az'], CAM['roll'], CAM['scale'], 20.0, CAM['cx'] - 16.13, CAM['cy'] - 2.55]
    r = optimize.least_squares(lambda q: resid2(np.concatenate([q, CS]), wreg, 1.0), z, loss='soft_l1', f_scale=3.0,
                               max_nfev=500)
    return r.x


if __name__ == '__main__':
    def r_ours(z):                                      # stage 1 (with a depth gauge: mean depth 0)
        return np.concatenate([resid(z, CAM, 0.3), [10 * z.reshape(-1, 3).mean(0)[1]]])
    x = optimize.least_squares(r_ours, init_net().ravel(), loss='soft_l1', f_scale=3.0, max_nfev=3000).x
    best = None                                         # stage 2
    for dels in (-20, -10, 0, 10, 20):
        for daz in (-30, -15, 0, 15, 30):
            y0 = np.concatenate([x, [20 + dels, -90 + daz, 0, 100, 20, 216, 190]])
            rr = optimize.least_squares(lambda z: resid2(z), y0, loss='soft_l1', f_scale=3.0, max_nfev=40)
            c = float(np.sum(rr.fun ** 2))
            if best is None or c < best[0]:
                best = (c, rr.x)
    y = optimize.least_squares(lambda z: resid2(z), best[1], loss='soft_l1', f_scale=3.0, max_nfev=1500).x
    z = stage3(y[:-7])                                  # stage 3
    rr = resid(z, CAM, 0)
    print('Ours network: model->red %.2f px, red->model %.2f px' % (rr[:280].mean(), (rr[280:280 + len(RP)] / 0.7).mean()))
    print(json.dumps({'net': np.round(z, 4).tolist()}))
