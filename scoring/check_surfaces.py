"""Verify that the runner builds every case surface exactly as specified:
SMLib evaluates each surface on a grid of its natural domain ("probe" mode);
NURBS grids are compared point-by-point with our own evaluator, analytic grids
are checked against the implicit truth, the patch membership test and the
patch bounding box.  Run: python3 -I scoring/check_surfaces.py
Another engine's runner (SSI_RUNNER) may also report checks of the faces it builds from each
surface: boundary edges on the surface (edge_dev), every face meshed (meshed), and the mesh's
area against the patch's (area_mesh / area_patch); these are checked when present."""
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ssi_geom as g  # noqa: E402

ROOT = os.path.dirname(HERE)
RUNNER = os.environ.get("SSI_RUNNER") or os.path.join(ROOT, "build", "ssi_runner")
N = 21


def main(pattern="*"):
    bad = 0
    files = sorted(glob.glob(os.path.join(ROOT, "cases", "data", pattern + ".json")))
    with tempfile.TemporaryDirectory() as td:
        for f in files:
            case = json.load(open(f))
            out = os.path.join(td, "p.json")
            r = subprocess.run([RUNNER, f, "probe", str(N), out], capture_output=True, timeout=120)
            if r.returncode != 0:
                print("PROBE FAIL", case["id"], r.returncode, r.stderr[-300:])
                bad += 1
                continue
            pr = json.load(open(out))
            msgs = []
            for k in range(2):
                spec, truth = case["surfaces"][k], case["truth"][k]
                G = np.asarray(pr["grid%d" % (k + 1)], float)
                if not np.isfinite(G).all():
                    msgs.append("s%d non-finite eval" % (k + 1))
                    continue
                s = g.from_spec(spec)
                res = g.truth_distance(truth, G).max()
                if spec["type"] == "nurbs":
                    d = pr["domain%d" % (k + 1)]
                    want = spec.get("domain") or [spec["knots_u"][0], spec["knots_u"][-1], spec["knots_v"][0],
                                                  spec["knots_v"][-1]]
                    if max(abs(a - b) for a, b in zip(d, want)) > 1e-12:
                        msgs.append("s%d natural domain %s != knot range %s BAD" % (k + 1, d, want))
                        bad += 1
                        continue
                    mine = np.array([s.eval(u, v) for u in np.linspace(d[0], d[1], N) for v in np.linspace(d[2], d[3], N)])
                    diff = np.abs(mine - G).max()
                    ok = diff < 1e-12 and res < 1e-12
                    msgs.append("s%d nurbs diff=%.1e res=%.1e%s" % (k + 1, diff, res, "" if ok else " BAD"))
                else:
                    ins = s.inside(G, 1e-7).all()
                    mine = s.grid(4 * N)
                    bb = max(np.abs(mine.min(0) - G.min(0)).max(), np.abs(mine.max(0) - G.max(0)).max())
                    ok = res < 1e-12 and ins and bb < 0.02 * s.size()  # coarse 21x21 grid vs fine grid: chordal slack
                    msgs.append("s%d %s res=%.1e inside=%s bbox=%.1e%s" % (k + 1, pr["class%d" % (k + 1)], res, ins, bb,
                                                                        "" if ok else " BAD"))
                if "area_mesh%d" % (k + 1) in pr:
                    am, ap = pr["area_mesh%d" % (k + 1)], pr["area_patch%d" % (k + 1)]
                    ed, me = pr["edge_dev%d" % (k + 1)], pr["meshed%d" % (k + 1)]
                    fok = me and ed is not None and ed < 1e-9 and abs(am / ap - 1) < 0.03
                    msgs.append("faces=%d edge_dev=%.1e area mesh/patch=%.4f%s" % (
                        pr["nfaces%d" % (k + 1)], ed if ed is not None else float("nan"), am / ap,
                        "" if fok else " BAD"))
                    ok = ok and fok
                bad += 0 if ok else 1
            print("%-34s %s" % (case["id"], " | ".join(msgs)))
    print("problems:", bad)
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(*(sys.argv[1:2])) else 0)
