// SSI evidence runner: builds two SMLib surfaces from a case JSON, runs one of
// SMLib/SM_API's surface/surface intersection entry points, and writes the raw
// result as JSON (3D curve samples, UV curve samples on both surfaces with the
// surfaces evaluated there, curve types, status, time).
//
// usage: ssi_runner <case.json> <mode> <approx_tol|default> <out.json>
//        ssi_runner <case.json> probe <n> <out.json>
// modes:
//   gsi      SmSurface::GlobalSurfaceIntersect (virtual: analytic special cases,
//            otherwise SmAdvSurfaceIntersector) - what SmApiIntersectSurfaces calls
//   adv      SmAdvSurfaceIntersector::DoIntersection directly (general NURBS path)
//   basic    SmSurfaceIntersector::DoIntersection (basic intersector)
//   face     SmFace::FaceIntersect on two single-face breps (also returns points)
//   api      SmApiIntersectSurfaces (SM_API defaults; no tolerance argument)
//   apiface  SmApiIntersectFaces (SM_API defaults, face tolerance)
//   probe    no intersection: evaluate each surface on an n x n grid of its
//            natural UV domain (used to verify surface construction)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <SmApiGeneral.h>
#include <SmApiIntersectors.h>
#include <SmApiTypes.h>
#include <SmAdvSurfaceIntersector.h>
#include <SmBSplineCurve.h>
#include <SmBSplineSurface.h>
#include <SmBrep.h>
#include <SmCone.h>
#include <SmContext.h>
#include <SmCurve.h>
#include <SmCylinder.h>
#include <SmFace.h>
#include <SmNurbs.h>
#include <SmNurbsSrf.h>
#include <NL_Geometry.h>
#include <NL_Nurbsdef.h>
#include <SmAxis2Placement.h>
#include <SmPlane.h>
#include <SmSphere.h>
#include <SmSurface.h>
#include <SmSurfaceIntersector.h>
#include <SmTorus.h>

#include "json.h"

namespace {

SmVector3d vec(const mj::Value& v) { return SmVector3d(v[0].num(), v[1].num(), v[2].num()); }

SmVector3d unit_cross(const SmVector3d& a, const SmVector3d& b) {
  SmVector3d c(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
  double l = std::sqrt(c.x * c.x + c.y * c.y + c.z * c.z);
  return SmVector3d(c.x / l, c.y / l, c.z / l);
}

struct Built {
  SmSurface* surf = nullptr;
  SmExtent2d domain;
};

// Build one SMLib surface from its case description.
Built build_surface(const SmContext& ctx, const mj::Value& s) {
  const std::string& type = s["type"].str();
  Built b;
  if (type == "plane") {
    SmPoint3d o = vec(s["origin"]);
    SmVector3d x = vec(s["xaxis"]), y = vec(s["yaxis"]);
    SmExtent2d dom(s["u"][0].num(), s["v"][0].num(), s["u"][1].num(), s["v"][1].num());
    b.surf = new (ctx) SmPlane(o, x, y, SmVector2d(1.0, 1.0), dom, &ctx);
  } else if (type == "cylinder") {
    SmVector3d x = vec(s["xaxis"]), y = vec(s["yaxis"]);
    SmVector3d z = unit_cross(x, y);
    double h0 = s["height"][0].num(), h1 = s["height"][1].num();
    SmPoint3d o = vec(s["origin"]) + z * h0;
    b.surf = new (ctx) SmCylinder(o, x, y, s["radius"].num(), s["angle"][0].num(),
                                  s["angle"][1].num(), h1 - h0, FALSE, FALSE, FALSE, &ctx);
  } else if (type == "cone") {
    // apex-based description: radius(h) = h * tan(half_angle), h in [h0, h1] along z,
    // full 360 degree rotation
    SmVector3d x = vec(s["xaxis"]), y = vec(s["yaxis"]);
    SmVector3d z = unit_cross(x, y);
    double h0 = s["height"][0].num(), h1 = s["height"][1].num();
    double t = std::tan(s["half_angle_deg"].num() * M_PI / 180.0);
    SmPoint3d o = vec(s["apex"]) + z * h0;
    SmAxis2Placement pl(o, x, y);
    SmCone* cone = nullptr;
    if (SmCone::CreateCanonical(ctx, pl, h0 * t, h1 * t, h1 - h0, cone) != SM_SUCCESS || !cone)
      throw std::runtime_error("cone CreateCanonical failed");
    b.surf = cone;
  } else if (type == "sphere") {
    SmExtent2d anal(s["lon"][0].num(), s["lat"][0].num(), s["lon"][1].num(), s["lat"][1].num());
    b.surf = new (ctx) SmSphere(vec(s["center"]), vec(s["xaxis"]), vec(s["yaxis"]), anal,
                                s["radius"].num(), FALSE, FALSE, NULL, &ctx);
  } else if (type == "torus") {
    SmExtent2d anal(s["major"][0].num(), s["minor"][0].num(), s["major"][1].num(),
                    s["minor"][1].num());
    b.surf = new (ctx) SmTorus(vec(s["center"]), vec(s["xaxis"]), vec(s["yaxis"]), anal,
                               s["R"].num(), s["r"].num(), FALSE, FALSE, NULL, &ctx);
  } else if (type == "nurbs") {
    int p = static_cast<int>(s["degree"][0].num()), q = static_cast<int>(s["degree"][1].num());
    const mj::Value& ku = s["knots_u"];
    const mj::Value& kv = s["knots_v"];
    ULONG nu = static_cast<ULONG>(s["nu"].num()), nv = static_cast<ULONG>(s["nv"].num());
    const mj::Value& pts = s["points"];
    const mj::Value& wts = s["weights"];
    if (pts.size() != nu * nv || wts.size() != nu * nv) throw std::runtime_error("nurbs: bad sizes");
    if (ku.size() != nu + p + 1 || kv.size() != nv + q + 1) throw std::runtime_error("nurbs: bad knots");
    std::vector<double> U(ku.size()), V(kv.size());
    for (size_t i = 0; i < ku.size(); ++i) U[i] = ku[i].num();
    for (size_t i = 0; i < kv.size(); ++i) V[i] = kv[i].num();
    // Allocate the NLib surface and fill it directly. (The SmBSplineSurface
    // component constructor passes the U knot count as the destination size when
    // copying the V knots, so V knots are silently not copied whenever the V knot
    // vector is longer than the U knot vector; see report.)
    gw_SURFACE* gs = sm_AllocateNurbSurface(nu - 1, nv - 1, static_cast<gw_DEGREE>(p),
                                            static_cast<gw_DEGREE>(q), U.size() - 1, V.size() - 1);
    if (!gs) throw std::runtime_error("nurbs: allocation failed");
    for (size_t i = 0; i < U.size(); ++i) gs->knu->U[i] = U[i];
    for (size_t i = 0; i < V.size(); ++i) gs->knv->U[i] = V[i];
    for (ULONG i = 0; i < nu; ++i) {
      for (ULONG j = 0; j < nv; ++j) {
        const mj::Value& P = pts[i * nv + j];
        double w = wts[i * nv + j].num();
        // NLib control points are homogeneous: Cartesian = (x/w, y/w, z/w)
        gs->net->Pw[i][j].x = P[0].num() * w;
        gs->net->Pw[i][j].y = P[1].num() * w;
        gs->net->Pw[i][j].z = P[2].num() * w;
        gs->net->Pw[i][j].w = w;
      }
    }
    SmBSplineSurface* bs = new (ctx) SmBSplineSurface(gs, FALSE, &ctx);
    b.surf = bs;
  } else {
    throw std::runtime_error("unknown surface type: " + type);
  }
  b.domain = b.surf->GetNaturalUVDomain();
  if (s.has("domain")) {  // optional trim box in natural parameters [u0,u1,v0,v1]
    const mj::Value& d = s["domain"];
    b.domain = SmExtent2d(d[0].num(), d[2].num(), d[1].num(), d[3].num());
  }
  return b;
}

const char* type_name(int t) {
  switch (t) {
    case SM_TC_UNKNOWN: return "UNKNOWN";
    case SM_TC_TOUCHING: return "TOUCHING";
    case SM_TC_CROSSING: return "CROSSING";
    case SM_TC_TANGENT: return "TANGENT";
    case SM_TC_COINCIDENT: return "COINCIDENT";
    case SM_TC_NEAR_TANGENT: return "NEAR_TANGENT";
    case SM_TC_REGION_BOUNDARY: return "REGION_BOUNDARY";
    default: return "?";
  }
}

std::string num(double d) {
  if (!std::isfinite(d)) return "null";
  char buf[40];
  std::snprintf(buf, sizeof buf, "%.17g", d);
  return buf;
}
std::string p3(const SmVector3d& p) { return "[" + num(p.x) + "," + num(p.y) + "," + num(p.z) + "]"; }

// Sample a curve uniformly in its natural parameter.
std::vector<SmPoint3d> sample_curve(const SmCurve* c, int n, SmExtent1d* ivl_out) {
  std::vector<SmPoint3d> out;
  if (!c) return out;
  SmExtent1d ivl = c->GetNaturalInterval();
  if (ivl_out) *ivl_out = ivl;
  for (int i = 0; i < n; ++i) {
    double t = ivl.GetMin() + (ivl.GetMax() - ivl.GetMin()) * (static_cast<double>(i) / (n - 1));
    SmPoint3d p;
    if (c->EvaluatePoint(t, p) == SM_SUCCESS) out.push_back(p);
  }
  return out;
}

std::string curve_json(const SmCurve* c3, const SmCurve* uv1, const SmCurve* uv2, const SmSurface* s1,
                       const SmSurface* s2, int type, double dev, int n) {
  std::ostringstream o;
  o << "{\"type\":\"" << (type >= 0 ? type_name(type) : "n/a") << "\"";
  o << ",\"deviation\":" << num(dev);
  SmExtent1d ivl;
  std::vector<SmPoint3d> pts = sample_curve(c3, n, &ivl);
  if (c3) {
    o << ",\"interval\":[" << num(ivl.GetMin()) << "," << num(ivl.GetMax()) << "]";
    o << ",\"class\":\"" << c3->GetTypeString() << "\"";
    const SmBSplineCurve* bc = dynamic_cast<const SmBSplineCurve*>(c3);
    if (bc) o << ",\"degree\":" << bc->GetDegree() << ",\"ncp\":" << bc->GetNumberControlPoints();
  }
  o << ",\"xyz\":[";
  for (size_t i = 0; i < pts.size(); ++i) o << (i ? "," : "") << p3(pts[i]);
  o << "]";
  const SmCurve* uvs[2] = {uv1, uv2};
  const SmSurface* ss[2] = {s1, s2};
  for (int k = 0; k < 2; ++k) {
    o << ",\"uv" << (k + 1) << "\":[";
    std::vector<SmPoint3d> uv = sample_curve(uvs[k], n, nullptr);
    std::ostringstream on;
    for (size_t i = 0; i < uv.size(); ++i) {
      o << (i ? "," : "") << "[" << num(uv[i].x) << "," << num(uv[i].y) << "]";
      SmPoint3d q;
      if (ss[k]->EvaluatePoint(SmPoint2d(uv[i].x, uv[i].y), q) != SM_SUCCESS) q = SmPoint3d(NAN, NAN, NAN);
      on << (i ? "," : "") << p3(q);
    }
    o << "],\"s" << (k + 1) << "_at_uv\":[" << on.str() << "]";
  }
  o << "}";
  return o.str();
}

std::string read_file(const char* path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: ssi_runner case.json mode approx_tol|default out.json\n");
    return 2;
  }
  std::string mode = argv[2];
  std::string tolArg = argv[3];
  mj::Value cs;
  try {
    cs = mj::parse(read_file(argv[1]));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "case parse error: %s\n", e.what());
    return 2;
  }

  SmContext* pctx = SmApiGetOrCreateContext();
  const SmContext& ctx = *pctx;
  Built a, b;
  try {
    a = build_surface(ctx, cs["surfaces"][0]);
    b = build_surface(ctx, cs["surfaces"][1]);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "surface build error: %s\n", e.what());
    return 3;
  }

  std::ostringstream out;
  out << "{\"case\":\"" << cs["id"].str() << "\",\"mode\":\"" << mode << "\"";
  out << ",\"domain1\":[" << num(a.domain.GetMin().x) << "," << num(a.domain.GetMax().x) << ","
      << num(a.domain.GetMin().y) << "," << num(a.domain.GetMax().y) << "]";
  out << ",\"domain2\":[" << num(b.domain.GetMin().x) << "," << num(b.domain.GetMax().x) << ","
      << num(b.domain.GetMin().y) << "," << num(b.domain.GetMax().y) << "]";

  if (mode == "probe") {
    int n = std::atoi(argv[3]);
    if (n < 2) n = 2;
    const Built* bs[2] = {&a, &b};
    for (int k = 0; k < 2; ++k) {
      out << ",\"grid" << (k + 1) << "\":[";
      const SmExtent2d& d = bs[k]->domain;
      bool first = true;
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
          double u = d.GetMin().x + (d.GetMax().x - d.GetMin().x) * i / (n - 1.0);
          double v = d.GetMin().y + (d.GetMax().y - d.GetMin().y) * j / (n - 1.0);
          SmPoint3d q;
          if (bs[k]->surf->EvaluatePoint(SmPoint2d(u, v), q) != SM_SUCCESS) q = SmPoint3d(NAN, NAN, NAN);
          out << (first ? "" : ",") << p3(q);
          first = false;
        }
      out << "]";
      out << ",\"class" << (k + 1) << "\":\"" << bs[k]->surf->GetTypeString() << "\"";
    }
    out << "}\n";
    std::ofstream(argv[4]) << out.str();
    return 0;
  }

  bool useDefaultTol = (tolArg == "default");
  SmApproxTol3d tol(useDefaultTol ? 0.0 : std::atof(tolArg.c_str()));
  const SmApproxTol3d* ptol = useDefaultTol ? nullptr : &tol;
  out << ",\"approx_tol\":" << (useDefaultTol ? std::string("\"default\"") : num(tol.val));

  SmTArray<SmCurve*> c3, uv1, uv2;
  SmTArray<SmTsectCurveType> types;
  SmTArray<double> devs;
  SmTArray<SmPoint3d> points;
  SmBoolean edges[2] = {TRUE, TRUE};
  SmStatus stat = SM_SUCCESS;
  bool haveUV = true, haveTypes = true;

  auto t0 = std::chrono::steady_clock::now();
  if (mode == "gsi") {
    stat = a.surf->GlobalSurfaceIntersect(ctx, a.domain, *b.surf, b.domain, edges, ptol, NULL, &c3, &uv1,
                                          &uv2, &types, &devs);
  } else if (mode == "adv") {
    SmAdvSurfaceIntersector ssi(*a.surf, a.domain, *b.surf, b.domain);
    ssi.SetUseSurfaceEdges(TRUE, TRUE);
    stat = ssi.DoIntersection(ctx, ptol, NULL, &c3, &uv1, &uv2, &types, &devs);
  } else if (mode == "basic") {
    SmSurfaceIntersector ssi(*a.surf, a.domain, *b.surf, b.domain);
    ssi.SetUseSurfaceEdges(TRUE, TRUE);
    stat = ssi.DoIntersection(ctx, ptol, NULL, &c3, &uv1, &uv2, &types, &devs);
  } else if (mode == "face" || mode == "apiface") {
    SmBrep* br1 = new (ctx) SmBrep();
    SmBrep* br2 = new (ctx) SmBrep();
    SmFace *f1 = nullptr, *f2 = nullptr;
    SmStatus s1 = br1->CreateFaceFromSurface(a.surf, a.domain, f1);
    SmStatus s2 = br2->CreateFaceFromSurface(b.surf, b.domain, f2);
    if (s1 != SM_SUCCESS || s2 != SM_SUCCESS || !f1 || !f2) {
      stat = (s1 != SM_SUCCESS) ? s1 : s2;
      if (stat == SM_SUCCESS) stat = SM_ERR;
    } else if (mode == "face") {
      double angTolDeg = 20.0;  // same value SmApiIntersectFaces uses
      SmApproxTol3d ftol = useDefaultTol ? SmApproxTol3d(f1->GetTolerance()) : tol;
      stat = f1->FaceIntersect(ctx, f2, ftol, angTolDeg, c3, &uv1, &uv2, &points, NULL, NULL);
      haveTypes = false;
    } else {
      stat = SmApiIntersectFaces(f1, f2, c3, &points);
      haveTypes = false;
      haveUV = false;
    }
  } else if (mode == "api") {
    stat = SmApiIntersectSurfaces(a.surf, b.surf, &c3);
    haveTypes = false;
    haveUV = false;
  } else {
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
  }
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  out << ",\"status\":" << static_cast<long>(stat) << ",\"time_ms\":" << num(ms);
  out << ",\"curves\":[";
  const int N = 4001;
  for (ULONG i = 0; i < c3.GetSize(); ++i) {
    const SmCurve* u1 = (haveUV && i < uv1.GetSize()) ? uv1[i] : nullptr;
    const SmCurve* u2 = (haveUV && i < uv2.GetSize()) ? uv2[i] : nullptr;
    int ty = (haveTypes && i < types.GetSize()) ? static_cast<int>(types[i]) : -1;
    double dv = (i < devs.GetSize()) ? devs[i] : NAN;
    out << (i ? "," : "") << curve_json(c3[i], u1, u2, a.surf, b.surf, ty, dv, N);
  }
  out << "],\"points\":[";
  for (ULONG i = 0; i < points.GetSize(); ++i) out << (i ? "," : "") << p3(points[i]);
  out << "]}\n";
  std::ofstream(argv[4]) << out.str();
  return 0;
}
