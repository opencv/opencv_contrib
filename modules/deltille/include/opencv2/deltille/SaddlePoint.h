#ifndef SADDLE_POINT
#define SADDLE_POINT

#include <opencv2/deltille/SaddleCluster.h>
#include <opencv2/core.hpp>

#include <cmath>

struct SaddlePoint {
  typedef SaddleClusterDesc ClusterDescType;
  typedef double PolarityStorageType;
  const static int polarityStorage = CV_64FC1;
  const static bool isTriangular = false;
  enum {
    PolaritySine1 = 0,
    PolarityCosine1 = 1,
    PolaritySine2 = 2,
    PolarityCosine2 = 3,
    NumPolarities // keep this one at the end...
  };

public:
  SaddlePoint() {}
  SaddlePoint(double x, double y) : x(x), y(y) {}
  SaddlePoint(const ClusterDescType &other)
      : x(other.cx), y(other.cy), a1(other.a1), a2(other.a2) {}

  void computePolarities(double *p) const {
    double a_1 = a1 + a2;
    double a_2 = a1 - a2;
    p[PolaritySine1] = sin(a_1);
    p[PolarityCosine1] = cos(a_1);
    p[PolaritySine2] = sin(a_2);
    p[PolarityCosine2] = cos(a_2);
  }

public:
  double x, y;
  PolarityStorageType a1, a2;
  double s, det;
};

#endif