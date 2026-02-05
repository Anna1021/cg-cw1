#ifndef COMPUTE_FUNCTION_MLS_H
#define COMPUTE_FUNCTION_MLS_H

#include <cmath>
#include <limits>
#include <Eigen/Core>

double nanValue = std::numeric_limits<double>::quiet_NaN();

// Wendland C2 kernel (compact support on [0, h])
static inline double wendlandC2(double r, double h) {
  if (r >= h) return 0.0;
  double t = 1.0 - (r / h);               // t in (0,1]
  double t2 = t * t;
  double t4 = t2 * t2;
  double s  = 4.0 * (r / h) + 1.0;
  return t4 * s;
}

Eigen::VectorXd compute_function_mls(const Eigen::MatrixXd& gridLocations,
                                     const Eigen::MatrixXd& pointCloud,
                                     const Eigen::MatrixXd& pointNormals,
                                     const double h) {
  using namespace Eigen;

  VectorXd MLSValues(gridLocations.rows());
  const double h2 = h * h;

  for (int gi = 0; gi < gridLocations.rows(); ++gi) {
    const RowVector3d x = gridLocations.row(gi);

    double numerator = 0.0;
    double denom     = 0.0;

    for (int pi = 0; pi < pointCloud.rows(); ++pi) {
      const RowVector3d p = pointCloud.row(pi);
      const RowVector3d n = pointNormals.row(pi);

      const RowVector3d d = x - p;
      const double r2 = d.squaredNorm();

      // Only points within support
      if (r2 > h2) continue;

      const double r = std::sqrt(r2);
      const double w = wendlandC2(r, h);
      if (w == 0.0) continue;

      // local signed distance to tangent plane at p_i
      const double planeDist = n.dot(d);

      numerator += w * planeDist;
      denom     += w;
    }

    if (denom == 0.0) {
      MLSValues(gi) = nanValue;   // required: no neighbors -> NaN
    } else {
      MLSValues(gi) = numerator / denom;
    }
  }

  return MLSValues;
}

#endif
