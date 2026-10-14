# SSI corpus

119 surface/surface intersection cases with exact ground truth, and a scorer that grades any engine's output against it. Built for the SolidCraft and solidcore SSI work (bherbruck/solidcore#17).

## Contents

- `cases/data/<id>.json`: one case per file. Each case holds:
  - the two surfaces (plane, quadrics, torus, B-spline and NURBS, as typed parameters or control nets);
  - the exact intersection, sampled densely (`truth`);
  - the expected topology (`expected`): branches, isolated points, singular points and coincident regions;
  - the coverage tolerance (`cov_tol`) and a note on how the truth was obtained.
- `cases/gen_cases.py`: generator for the 94 own cases (classes 1–9: transversal, tangent curves, isolated points, singular points, boundaries, coincident regions, near-degenerate, seams and poles, high degree and rational).
- `cases/gen_yang2023.py`: the 6 cases with the paper's published data (Table 3/4 torus/ellipsoid pairs, appendix Example 2).
- `cases/y23_fit/<id>.py`: the 19 reconstructions of the paper's Fig. 1, Table 2 and Table 6 examples. The paper publishes no geometry for these. Each surface pair was fitted to the paper's figure through a camera recovered from the figure, keeping the stated degree, topology and contact type, with the exact intersection in closed form. `fit_*.py` and `render_match.py` are the fitting tools.
- `scoring/score.py`: runs an engine on every case and scores the result. `ssi_geom.py`, `implicit_trace.py`, `param_trace.py` and `check_surfaces.py` are its geometry helpers.
- `runner/ssi_runner.cpp` + `CMakeLists.txt`: the runner used for SMLib (commit `c9c5979`), as an example of the runner contract.

## Scoring an engine

    SSI_ENGINE=<name> SSI_RUNNER=<binary> python3 -I scoring/score.py <mode> <tol> [case-glob]

The runner is called as `<binary> <case.json> <mode> <tol> <out.json>` and writes JSON with `status`, `time_ms`, and `curves`: a list of `{xyz, uv1, uv2, s1_at_uv, s2_at_uv, type}` samples, plus optional `points`. See `runner/ssi_runner.cpp` for the exact format.

Verdicts, most severe first: CRASH, ERROR, INCOMPLETE (an expected branch, point or region is not covered), WRONG TOPOLOGY (spurious output, wrong component count or open/closed structure, duplicated curves, or a curve running through a singular point instead of ending there), INACCURATE, PASS.

Python 3 with numpy and scipy; the y23 fitting tools also need Pillow.

## Reference results (approximation tolerance 1e-6)

| engine | pass |
|---|---|
| SMLib `GlobalSurfaceIntersect` | 81 / 119 |
| SolidCraft face/face intersection | 47 / 119 |

## Source

The Yang cases come from Yang, Jia and Yan, "Topology Guaranteed B-Spline Surface/Surface Intersection", ACM Transactions on Graphics 42(6), Article 211, 2023 (CC-BY). Every such case cites the paper in its `source` field.

## License

MIT OR Apache-2.0, at your option.
