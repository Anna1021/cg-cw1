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

  const int G = gridLocations.rows();
  const int P = pointCloud.rows();

  VectorXd out(G);
  out.setConstant(NaN());

  if (G == 0 || P == 0) return out;

  const bool useOffsets = (std::abs(epsNormal) > 0.0);
  const double h2 = h*h;

  std::vector<std::array<int,3>> exps;
  for (int i=0;i<=N;i++)
    for (int j=0;j<=N-i;j++)
      for (int k=0;k<=N-i-j;k++)
        exps.push_back({i,j,k});

  const int M = exps.size();

  auto eval_phi = [&](const RowVector3d& u, VectorXd& phi){
    phi.resize(M);
    std::vector<double> xp(N+1,1.0), yp(N+1,1.0), zp(N+1,1.0);

    for(int d=1;d<=N;d++){
      xp[d]=xp[d-1]*u(0);
      yp[d]=yp[d-1]*u(1);
      zp[d]=zp[d-1]*u(2);
    }

    for(int m=0;m<M;m++){
      phi(m)=xp[exps[m][0]]*yp[exps[m][1]]*zp[exps[m][2]];
    }
  };

  auto weight = [&](const RowVector3d& y,const RowVector3d& x0){
    double r2=(y-x0).squaredNorm();
    if(r2>h2*(1.0+1e-12)) return 0.0;
    double q=sqrt(std::max(0.0,r2))/h;
    if(q>=1.0) q=std::nextafter(1.0,0.0);
    return wendlandC2(q);
  };

  MatrixXd C;
  VectorXd b,phi;

  auto add_row=[&](const RowVector3d& x0,const RowVector3d& y,double rhs,int& rows){
    double w=weight(y,x0);
    if(w<=0.0) return;

    double sw=sqrt(std::max(w,1e-30));

    // ⭐ 关键改动：去掉 /h
    RowVector3d u=(y-x0);

    eval_phi(u,phi);

    C.row(rows)=(sw*phi).transpose();
    b(rows)=sw*rhs;
    rows++;
  };

  for(int gi=0;gi<G;gi++){
    RowVector3d x0=gridLocations.row(gi);

    int rowsMax=useOffsets?3*P:P;
    C.resize(rowsMax,M);
    b.resize(rowsMax);

    int rows=0;

    for(int pi=0;pi<P;pi++){
      RowVector3d p=pointCloud.row(pi);

      add_row(x0,p,0.0,rows);

      if(useOffsets){
        RowVector3d n=safeUnit(pointNormals.row(pi));
        if(n.squaredNorm()>0.0){
          add_row(x0,p+epsNormal*n, epsNormal,rows);
          add_row(x0,p-epsNormal*n,-epsNormal,rows);
        }
      }
    }

    if(rows<M){
      out(gi)=NaN();
      continue;
    }

    C.conservativeResize(rows,NoChange);
    b.conservativeResize(rows);

    JacobiSVD<MatrixXd> svd(C,ComputeThinU|ComputeThinV);
    VectorXd a=svd.solve(b);

    if(!a.allFinite()){
      out(gi)=NaN();
      continue;
    }

    out(gi)=a(0);
  }

  return out;
}

#endif
