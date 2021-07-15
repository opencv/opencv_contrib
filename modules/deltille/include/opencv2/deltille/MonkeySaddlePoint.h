#ifndef MONKEY_SADDLE_POINT
#define MONKEY_SADDLE_POINT

#include <opencv2/deltille/SaddleCluster.h>
#include <opencv2/core.hpp>

#include <complex>
#include <cmath>


struct MonkeySaddlePoint {
  typedef MonkeySaddleClusterDesc ClusterDescType;
  typedef std::complex<double> PolarityStorageType;
  const static int polarityStorage = CV_64FC2;
  const static bool isTriangular = true;
  enum {
    PolaritySine1 = 0,
    PolarityCosine1 = 1,
    PolaritySine2 = 2,
    PolarityCosine2 = 3,
    PolaritySine3 = 4,
    PolarityCosine3 = 5,
    NumPolarities // keep this one at the end...
  };

public:
  MonkeySaddlePoint() {}
  MonkeySaddlePoint(double x, double y) : x(x), y(y) {}
  MonkeySaddlePoint(const ClusterDescType &other)
      : x(other.cx), y(other.cy), a1(other.a1), a2(other.a2), a3(other.a3) {}

  void computePolarities(PolarityStorageType *p) const {
    p[PolaritySine1] = sin(a1);
    p[PolarityCosine1] = cos(a1);
    p[PolaritySine2] = sin(a2);
    p[PolarityCosine2] = cos(a2);
    p[PolaritySine3] = sin(a3);
    p[PolarityCosine3] = cos(a3);
  }

public:
  double x, y;
  PolarityStorageType a1, a2, a3;
  double s, det;
};

#endif