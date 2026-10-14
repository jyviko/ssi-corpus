"""Run the SMLib SSI runner over the cases and score each result against the
closed-form ground truth.

usage: python3 -I scoring/score.py <mode> <approx_tol|default> [case-glob]
writes results/raw/<mode>_<tol>/<id>.json (runner output, gitignored) and
results/score_<mode>_<tol>.json (scores, committed).

Engine (environment): SSI_ENGINE names the engine (default smlib) and SSI_RUNNER the runner
binary (default build/ssi_runner); for another engine the raw output goes to
results/raw/<engine>_<mode>_<tol>/ and the scores to results/score_<engine>_<mode>_<tol>.json, or
results/score_<SSI_LABEL>.json when SSI_LABEL is set. A runner may report its own status name
("status_name" in its output).

Verdicts (most severe wins; all flags are kept):
  CRASH           runner died (signal) or exceeded the time limit
  ERROR           non-success status from SMLib
  INCOMPLETE      an expected branch / part of a branch / isolated point /
                  coincident region boundary is not covered by the output
  WRONG TOPOLOGY  spurious output, wrong number of connected components,
                  wrong open/closed structure (free ends), or a singular point
                  where the output curves are not split
  INACCURATE      output points (3D curves, or the surfaces evaluated on the
                  output UV curves) farther than acc_tol from a true surface
  PASS
"""
import glob
import json
import math
import multiprocessing
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ssi_geom as g  # noqa: E402

ROOT = os.path.dirname(HERE)
ENGINE = os.environ.get("SSI_ENGINE", "smlib")
RUNNER = os.environ.get("SSI_RUNNER") or os.path.join(ROOT, "build", "ssi_runner")
TIMEOUT_S = 300
SM_SUCCESS = 1000


def status_names():
    names = {}
    path = os.path.join(ROOT, "upstream", "source", "SMLib", "inc", "SmMessages.h")
    try:
        for line in open(path, errors="replace"):
            m = re.match(r"#define\s+(SM_\w+)\s+(\d+)L", line)
            if m and int(m.group(2)) not in names:
                names[int(m.group(2))] = m.group(1)
    except OSError:
        pass
    return names


STATUS = status_names()

# ----------------------------------------------------------------------------
# distances from points to a set of polylines (segment hash, exact below the
# cell size, approximate (nearest decimated vertex) above it)


class PolySet:
    def __init__(self, polylines, points=(), cell=None):
        segs_a, segs_b, owner = [], [], []
        for k, P in enumerate(polylines):
            P = np.asarray(P, float)
            if len(P) == 1:
                P = np.vstack([P, P])
            segs_a.append(P[:-1])
            segs_b.append(P[1:])
            owner.append(np.full(len(P) - 1, k))
        for p in points:
            p = np.asarray(p, float)[None, :]
            segs_a.append(p)
            segs_b.append(p)
            owner.append(np.full(1, -1))
        self.empty = not segs_a
        if self.empty:
            return
        self.a = np.vstack(segs_a)
        self.b = np.vstack(segs_b)
        self.owner = np.concatenate(owner)
        L = np.linalg.norm(self.b - self.a, axis=1)
        self.cell = cell if cell else max(float(np.percentile(L, 95)) * 2, 1e-300)
        self.cell = max(self.cell, float(L.max()) / 50 if L.max() > 0 else self.cell)
        self.table = {}
        for i in range(len(self.a)):
            n = max(1, int(math.ceil(L[i] / self.cell)) * 2)
            for t in np.linspace(0, 1, n + 1):
                key = tuple(np.floor((self.a[i] + t * (self.b[i] - self.a[i])) / self.cell).astype(np.int64))
                self.table.setdefault(key, set()).add(i)
        self.table = {k: np.fromiter(v, dtype=np.int64) for k, v in self.table.items()}
        step = max(1, len(self.a) // 4000)
        self.coarse = self.a[::step]

    def seg_dist(self, q, idx):
        a, b = self.a[idx], self.b[idx]
        d = b - a
        dd = np.einsum("ij,ij->i", d, d)
        t = np.where(dd > 0, np.einsum("ij,ij->i", q - a, d) / np.where(dd > 0, dd, 1), 0)
        t = np.clip(t, 0, 1)
        c = a + t[:, None] * d
        return np.linalg.norm(q - c, axis=1)

    def query(self, Q):
        Q = np.atleast_2d(np.asarray(Q, float))
        if self.empty:
            return np.full(len(Q), np.inf)
        out = np.empty(len(Q))
        offs = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
        keys = np.floor(Q / self.cell).astype(np.int64)
        for n, q in enumerate(Q):
            kq = keys[n]
            cand = [self.table.get((kq[0] + i, kq[1] + j, kq[2] + k)) for i, j, k in offs]
            cand = [c for c in cand if c is not None]
            best = np.inf
            if cand:
                idx = np.unique(np.concatenate(cand))
                best = float(self.seg_dist(q, idx).min())
            if best > self.cell:
                best = min(best, float(np.linalg.norm(self.coarse - q, axis=1).min()))
            out[n] = best
        return out


# ----------------------------------------------------------------------------


def diameter(P):
    return float(np.ptp(P, axis=0).max()) if len(P) else 0.0


def seg_seg(p1, q1, p2, q2):
    """Exact distance between segments p1q1 and p2q2 (clamped closest points)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a <= 1e-300 and e <= 1e-300:
        return float(np.linalg.norm(r))
    if a <= 1e-300:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = d1 @ r
        if e <= 1e-300:
            t, s = 0.0, min(max(-c / a, 0.0), 1.0)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = min(max((b * f - c * e) / den, 0.0), 1.0) if den > 1e-300 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, min(max(-c / a, 0.0), 1.0)
            elif t > 1:
                t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def polyline_distance(A, B, setB, tol):
    """Min distance between polylines A and B: vertex query first, exact
    segment-segment distance around the closest approach when it matters."""
    dv = setB.query(A)
    k = int(np.argmin(dv))
    d = float(dv[k])
    if d <= tol:
        return d
    segA = np.linalg.norm(np.diff(A, axis=0), axis=1)
    segB = np.linalg.norm(np.diff(B, axis=0), axis=1)
    if d > tol + max(segA.max(initial=0), segB.max(initial=0)):
        return d
    best = d
    for i in np.nonzero(dv <= tol + segA.max(initial=0) + segB.max(initial=0))[0]:
        for ia in (i - 1, i):
            if ia < 0 or ia + 1 >= len(A):
                continue
            jb = int(np.argmin(np.linalg.norm(B - A[ia], axis=1)))
            for j in range(max(0, jb - 3), min(len(B) - 1, jb + 3)):
                best = min(best, seg_seg(A[ia], A[ia + 1], B[j], B[j + 1]))
    return best


def components(curves, points, join_tol):
    """Connected components of curves (polylines) and points; returns list of
    lists of ('c', i) / ('p', i)."""
    items = [("c", i) for i in range(len(curves))] + [("p", i) for i in range(len(points))]
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    sets = [PolySet([c]) for c in curves]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            ti, ki = items[i]
            tj, kj = items[j]
            if ti == "p" and tj == "p":
                d = np.linalg.norm(np.asarray(points[ki]) - points[kj])
            elif ti == "c" and tj == "c":
                d = polyline_distance(curves[ki], curves[kj], sets[kj], join_tol)
            else:
                c, p = (ki, points[kj]) if ti == "c" else (kj, points[ki])
                d = sets[c].query([p]).min()
            if d <= join_tol:
                parent[find(i)] = find(j)
    groups = {}
    for i, it in enumerate(items):
        groups.setdefault(find(i), []).append(it)
    return list(groups.values())


def free_ends(curves, join_tol):
    """Number of free endpoints of each curve."""
    res = []
    sets = [PolySet([c]) for c in curves]
    for i, C in enumerate(curves):
        if np.linalg.norm(C[0] - C[-1]) <= join_tol and diameter(C) > 2 * join_tol:
            res.append(0)
            continue
        seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        n = 0
        for end, si in ((C[0], 0.0), (C[-1], s[-1])):
            glued = False
            for j in range(len(curves)):
                if j != i and sets[j].query([end])[0] <= join_tol:
                    glued = True
                    break
            if not glued:
                far = np.abs(s - si) > 3 * join_tol
                if far.any() and np.linalg.norm(C[far] - end, axis=1).min() <= join_tol:
                    glued = True
            n += 0 if glued else 1
        res.append(n)
    return res


def signature(curves, points, join_tol):
    comps = components(curves, points, join_tol)
    fe = free_ends(curves, join_tol) if curves else []
    sig = []
    for comp in comps:
        cs = [k for t, k in comp if t == "c"]
        if not cs:
            sig.append("point")
        else:
            sig.append("curve(free_ends=%d)" % sum(fe[k] for k in cs))
    return sorted(sig)


def score_case(case, out, acc_tol, crash=None, cov_floor=0.0):
    exp = case["expected"]
    truths = case["truth"]
    cov_tol = case.get("cov_tol", 1e-3)
    cov_eff = cov_tol
    if case.get("ill_cond") and acc_tol:
        cov_eff = max(cov_tol, min(0.5, 2 * acc_tol * case["ill_cond"]))
    cov_eff = max(cov_eff, cov_floor)
    join_tol = max(cov_tol, cov_floor)
    r = {"id": case["id"], "class": case["class"], "cov_tol": cov_eff, "flags": []}
    if crash:
        r.update(verdict="CRASH", detail=crash)
        return r
    r["status"] = out["status"]
    r["status_name"] = out.get("status_name") or STATUS.get(out["status"], str(out["status"]))
    r["time_ms"] = out["time_ms"]
    curves_raw = out["curves"]
    out_curves, out_points, types = [], [], []
    nonfinite = 0
    for c in curves_raw:
        P = np.asarray([q if q is not None else [np.nan] * 3 for q in c["xyz"]], float).reshape(-1, 3)
        bad = ~np.isfinite(P).all(axis=1)
        if bad.any():
            nonfinite += int(bad.sum())
            P = P[~bad]
        if len(P) == 0:
            continue
        if c["type"] == "TOUCHING" or diameter(P) <= cov_tol:
            out_points.append(P.mean(axis=0))
        else:
            out_curves.append(P)
        types.append(c["type"])
    for p in out.get("points", []):
        out_points.append(np.asarray(p, float))
    r["n_curves"] = len(curves_raw)
    r["n_out_points"] = len(out_points)
    r["types"] = types

    # accuracy against the exact surfaces
    acc3d, accuv = 0.0, 0.0
    for c in curves_raw:
        P = np.asarray([q if q is not None else [np.nan] * 3 for q in c["xyz"]], float).reshape(-1, 3)
        P = P[np.isfinite(P).all(axis=1)]
        if len(P):
            acc3d = max(acc3d, max(float(g.truth_distance(t, P).max()) for t in truths))
        for k, key in ((1, "s1_at_uv"), (0, "s2_at_uv")):  # S1(uv1) must lie on surface 2, etc.
            Q = np.asarray([q if q is not None else [np.nan] * 3 for q in c.get(key, [])], float).reshape(-1, 3)
            Q = Q[np.isfinite(Q).all(axis=1)]
            if len(Q):
                accuv = max(accuv, float(g.truth_distance(truths[k], Q).max()))
    if out_points:
        acc3d = max(acc3d, max(float(g.truth_distance(t, np.array(out_points)).max()) for t in truths))
    r["max_dist_3d"] = acc3d
    r["max_dist_uv"] = accuv

    truth_polys = [np.asarray(b["points"], float) for b in exp["branches"]] + \
                  [np.asarray(x["boundary"], float) for x in exp["coincident_regions"]]
    truth_pts = [np.asarray(p, float) for p in exp["isolated_points"]]
    expected_any = bool(truth_polys or truth_pts)

    ok_status = out["status"] == SM_SUCCESS
    if not ok_status:
        if not expected_any and not curves_raw and not out_points:
            r["flags"].append("status %s with empty output (expected empty)" % r["status_name"])
        else:
            r["flags"].append("ERROR status %s" % r["status_name"])

    # coverage truth -> output
    outset = PolySet(out_curves, out_points)
    cover = []
    missed = []
    for k, P in enumerate(truth_polys):
        d = outset.query(P[:: max(1, len(P) // 1000)])
        frac = float((d <= cov_eff).mean())
        cover.append({"max": float(d.max()), "covered_frac": frac})
        if d.max() > cov_eff:
            kind = "region boundary" if k >= len(exp["branches"]) else "branch %d" % k
            missed.append("%s covered %.0f%% (max gap %.2e)" % (kind, 100 * frac, d.max()))
    for k, p in enumerate(truth_pts):
        d = float(outset.query([p])[0])
        cover.append({"point": k, "dist": d})
        if d > cov_eff:
            missed.append("isolated point %d not found (nearest output %.2e)" % (k, d))
    r["coverage"] = cover
    r["hausdorff_truth_to_out"] = max([c.get("max", c.get("dist", 0)) for c in cover], default=0.0)

    # output -> truth (spurious output)
    tset = PolySet(truth_polys, truth_pts)
    spurious = []
    h_out = 0.0
    for k, P in enumerate(out_curves):
        d = tset.query(P[:: max(1, len(P) // 1000)])
        h_out = max(h_out, float(d.max()))
        if d.max() > cov_eff:
            spurious.append("output curve %d off the true intersection by up to %.2e (%.0f%% of it)"
                            % (k, d.max(), 100 * float((d > cov_eff).mean())))
    for k, p in enumerate(out_points):
        d = float(tset.query([p])[0])
        h_out = max(h_out, d)
        if d > cov_eff:
            spurious.append("output point %d off the true intersection by %.2e" % (k, d))
    r["hausdorff_out_to_truth"] = h_out

    # topology signature
    sig_exp = signature(truth_polys, truth_pts, join_tol)
    sig_out = signature(out_curves, out_points, join_tol)
    r["expected_topology"] = sig_exp
    r["output_topology"] = sig_out
    topo = []
    if nonfinite:
        topo.append("%d non-finite output sample(s)" % nonfinite)
    # duplicated / overlapping output curves
    osets = [PolySet([C]) for C in out_curves]
    for i in range(len(out_curves)):
        for j in range(i + 1, len(out_curves)):
            Ci, Cj = out_curves[i], out_curves[j]
            di = osets[j].query(Ci[:: max(1, len(Ci) // 500)])
            dj = osets[i].query(Cj[:: max(1, len(Cj) // 500)])
            frac = max(float((di <= join_tol).mean()), float((dj <= join_tol).mean()))
            if frac > 0.2:
                topo.append("output curves %d and %d overlap (%.0f%% of one lies on the other)" % (i, j, 100 * frac))
    if sig_exp != sig_out:
        topo.append("topology %s, expected %s" % (sig_out, sig_exp))
    for k, s in enumerate(exp["singular_points"]):
        s = np.asarray(s, float)
        ends = [C[0] for C in out_curves] + [C[-1] for C in out_curves] + out_points
        d = min([float(np.linalg.norm(e - s)) for e in ends], default=np.inf)
        if d > cov_eff:
            topo.append("singular point %d not identified (no curve end/point within %.1e; nearest %.2e)"
                        % (k, cov_eff, d))
    inacc = []
    if acc_tol is not None:
        if acc3d > acc_tol:
            inacc.append("3D output up to %.2e from a surface (acc_tol %.0e)" % (acc3d, acc_tol))
        if accuv > acc_tol:
            inacc.append("surface at output UV up to %.2e from the other surface" % accuv)

    r["flags"] += missed + spurious + topo + inacc
    if not ok_status and expected_any or (not ok_status and (curves_raw or out_points)):
        v = "ERROR"
    elif missed:
        v = "INCOMPLETE"
    elif spurious or topo:
        v = "WRONG TOPOLOGY"
    elif inacc:
        v = "INACCURATE"
    else:
        v = "PASS"
    if cov_floor and case.get("feature_size") is not None and case["feature_size"] < cov_floor:
        r["flags"].insert(0, "feature size %.1e below the %.0e judging floor of default-tolerance runs" %
                          (case["feature_size"], cov_floor))
        v = "N/A"
    r["verdict"] = v
    return r


def run_case(path, mode, tol, rawdir):
    cid = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(rawdir, cid + ".json")
    log = os.path.join(rawdir, cid + ".log")
    t0 = time.time()
    try:
        with open(log, "w") as lf:
            p = subprocess.run([RUNNER, path, mode, tol, out], stdout=lf, stderr=subprocess.STDOUT,
                               timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None, "timeout after %d s" % TIMEOUT_S
    if p.returncode != 0:
        sig = -p.returncode if p.returncode < 0 else None
        return None, ("killed by signal %d" % sig) if sig else ("runner exit code %d" % p.returncode)
    try:
        return json.load(open(out)), None
    except (OSError, ValueError) as e:
        return None, "unreadable output: %s" % e


def _one(job):
    f, mode, tol, rawdir, acc_tol = job
    case = json.load(open(f))
    out, crash = run_case(f, mode, tol, rawdir)
    # with SMLib's default approximation tolerance (1/1000 of the surface size) the
    # curves are only that accurate: judge presence/topology with a 5e-3 floor
    r = score_case(case, out, acc_tol, crash, cov_floor=5e-3 if tol == "default" else 0.0)
    r["mode"], r["approx_tol"], r["acc_tol"] = mode, tol, acc_tol
    return r


def main():
    mode, tol = sys.argv[1], sys.argv[2]
    pattern = sys.argv[3] if len(sys.argv) > 3 else "*"
    acc_tol = None if tol == "default" else 10 * float(tol)
    run = "%s_%s" % (mode, tol) if ENGINE == "smlib" else "%s_%s_%s" % (ENGINE, mode, tol)
    rawdir = os.path.join(ROOT, "results", "raw", run)
    os.makedirs(rawdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ROOT, "cases", "data", pattern + ".json")))
    jobs = [(f, mode, tol, rawdir, acc_tol) for f in files]
    with multiprocessing.Pool(int(os.environ.get("SSI_JOBS", "6"))) as pool:
        results = []
        for r in pool.imap(_one, jobs):
            results.append(r)
            print("%-36s %-15s %s" % (r["id"], r["verdict"], "; ".join(r.get("flags", []))[:220]), flush=True)
    name = "score_%s%s.json" % (os.environ.get("SSI_LABEL") or run, "" if pattern == "*" else "_" + re.sub(r"\W", "", pattern))
    with open(os.path.join(ROOT, "results", name), "w") as fh:
        json.dump(results, fh, indent=1)
    counts = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print(counts)


if __name__ == "__main__":
    main()
