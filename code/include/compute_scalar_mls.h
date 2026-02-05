#ifndef COMPUTE_SCALAR_MLS_H
#define COMPUTE_SCALAR_MLS_H

#include <cmath>
#include <limits>
#include <vector>
#include <array>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/SVD>

static inline double NaN() { return std::numeric_limits<double>::quiet_NaN(); }

// Wendland C2: (1-q)^4 (4q+1), q in [0,1]
static inline double wendlandC2(double q) {
  if (q >= 1.0) return 0.0;
  if (q < 0.0) q = 0.0;
  const double t = 1.0 - q;
  return (t*t*t*t) * (4.0*q + 1.0);
}

static inline Eigen::RowVector3d safeUnit(const Eigen::RowVector3d& n) {
  const double nn = n.norm();
  if (nn > 1e-12 && std::isfinite(nn)) return n / nn;
  return Eigen::RowVector3d(0.0, 0.0, 0.0);
}

Eigen::VectorXd compute_scalar_mls(const Eigen::MatrixXd& gridLocations,
                                   const Eigen::MatrixXd& pointCloud,
                                   const Eigen::MatrixXd& pointNormals,
                                   const int N,
                                   const double h,
                                   const double epsNormal) {
  using namespace Eigen;

  const int G = (int)gridLocations.rows();
  const int P = (int)pointCloud.rows();

  VectorXd out(G);
  out.setConstant(NaN());

  if (G == 0 || P == 0) return out;
  if (N < 0 || !(h > 0.0) || !std::isfinite(h)) return out;

  const bool useOffsets = (std::abs(epsNormal) > 0.0);
  const double h2 = h * h;

  // monomials: i+j+k <= N
  std::vector<std::array<int,3>> exps;
  exps.reserve((N+1)*(N+2)*(N+3)/6);
  for (int i = 0; i <= N; ++i)
    for (int j = 0; j <= N - i; ++j)
      for (int k = 0; k <= N - i - j; ++k)
        exps.push_back({i, j, k});
  const int M = (int)exps.size();

  // local basis at u = (y-x0)/h
  auto eval_phi_local = [&](const RowVector3d& u, VectorXd& phi) {
    phi.resize(M);
    std::vector<double> xp(N+1, 1.0), yp(N+1, 1.0), zp(N+1, 1.0);
    for (int d = 1; d <= N; ++d) {
      xp[d] = xp[d-1] * u(0);
      yp[d] = yp[d-1] * u(1);
      zp[d] = zp[d-1] * u(2);
    }
    for (int m = 0; m < M; ++m) {
      const int i = exps[m][0];
      const int j = exps[m][1];
      const int k = exps[m][2];
      phi(m) = xp[i] * yp[j] * zp[k];
    }
  };

  // weight with tolerant support check (important for boundary points)
  auto weight_at = [&](const RowVector3d& y, const RowVector3d& x0) -> double {
    const RowVector3d d = y - x0;
    const double r2 = d.squaredNorm();
    // tolerate tiny floating overshoot
    if (r2 > h2 * (1.0 + 1e-12)) return 0.0;

    const double r = std::sqrt(std::max(0.0, r2));
    double q = r / h;
    if (q >= 1.0) q = std::nextafter(1.0, 0.0); // keep inside so Wendland is >0-ish near boundary

    const double w = wendlandC2(q);
    return (w > 0.0 && std::isfinite(w)) ? w : 0.0;
  };

  MatrixXd C;
  VectorXd b;
  VectorXd phi;

  // add one constraint row if inside support
  auto add_row = [&](const RowVector3d& x0,
                     const RowVector3d& y,
                     const double rhs,
                     int& rows) {
    const double w = weight_at(y, x0);
    if (w <= 0.0) return;

    const double sw = std::sqrt(std::max(w, 1e-30));
    const RowVector3d u = (y - x0) / h;   // local coords
    eval_phi_local(u, phi);

    C.row(rows) = (sw * phi).transpose();
    b(rows) = sw * rhs;
    ++rows;
  };

  for (int gi = 0; gi < G; ++gi) {
    const RowVector3d x0 = gridLocations.row(gi);

    // worst case: each point adds 1 or 3 rows
    const int rowsMax = useOffsets ? (3 * P) : P;
    C.resize(rowsMax, M);
    b.resize(rowsMax);

    int rows = 0;

    for (int pi = 0; pi < P; ++pi) {
      const RowVector3d p = pointCloud.row(pi);

      // on-surface
      add_row(x0, p, 0.0, rows);

      if (useOffsets) {
        const RowVector3d n = safeUnit(pointNormals.row(pi));
        if (n.squaredNorm() > 0.0) {
          add_row(x0, p + epsNormal * n, +epsNormal, rows);
          add_row(x0, p - epsNormal * n, -epsNormal, rows);
        }
      }
    }

    // ✅ 关键：GT 更像用 rows<M 才 NaN（而不是点数<M）
    if (rows < M) {
      out(gi) = NaN();
      continue;
    }

    C.conservativeResize(rows, NoChange);
    b.conservativeResize(rows);

    // mild column scaling for stability
    VectorXd invCol(M);
    for (int j = 0; j < M; ++j) {
      const double cn = C.col(j).norm();
      invCol(j) = (cn > 1e-12 && std::isfinite(cn)) ? (1.0 / cn) : 1.0;
      C.col(j) *= invCol(j);
    }

    JacobiSVD<MatrixXd> svd(C, ComputeThinU | ComputeThinV);
    VectorXd a_scaled = svd.solve(b);
    if (!a_scaled.allFinite()) {
      out(gi) = NaN();
      continue;
    }
    VectorXd a = a_scaled.cwiseProduct(invCol);

    // local basis at u=0 => phi=[1,0,...] so F(x0)=a0
    const double val = a(0);
    out(gi) = std::isfinite(val) ? val : NaN();
  }

  return out;
}

#endif
