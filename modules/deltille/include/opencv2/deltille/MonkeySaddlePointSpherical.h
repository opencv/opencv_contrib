#ifndef INCLUDE_MONKEY_SADDLE
#define INCLUDE_MONKEY_SADDLE

#include <opencv2/opencv.hpp>
#include <opencv2/deltille/SaddleCluster.h>

/*
MONKEY SADDLE

Type class for a monkey saddle object.
*/

namespace orp {
namespace calibration {

struct MonkeySaddlePointSpherical {
  typedef MonkeySaddleClusterDesc ClusterDescType;
  typedef cv::Vec3d PolarityStorageType;
  const static int polarityStorage = CV_64FC3;
  const static bool isTriangular = true;
  enum {
    Polarity1 = 0,
    Polarity2 = 1,
    Polarity3 = 2,
    NumPolarities // keep this one at the end...
  };

public:
  MonkeySaddlePointSpherical() {}
  MonkeySaddlePointSpherical(double x, double y) : x(x), y(y) {}
  MonkeySaddlePointSpherical(const ClusterDescType &other)
      : x(other.cx), y(other.cy), a1(other.a1), a2(other.a2), a3(other.a3) {}

  void complexToSpherical(const std::complex<double> &angle,
                          PolarityStorageType &vec) const {
    const double ci = cos(angle.imag());
    vec[0] = ci * cos(angle.real());
    vec[1] = ci * sin(angle.real());
    vec[2] = sin(angle.imag());
  }

  void computePolarities(PolarityStorageType *p) const {
    // convert complex angles to vectors on unit sphere
    complexToSpherical(a1, p[0]);
    complexToSpherical(a2, p[1]);
    complexToSpherical(a3, p[2]);
  }

  static bool comparePolaritiesUnderRotation(const PolarityStorageType *p1,
                                             const PolarityStorageType *p2,
                                             int rotation) {
    const double triangle_polarity_threshold =
        cos(10.0 / 180 * M_PI); // cos(detector_params.triangle_polarity_angle);
    return std::abs(p1[0].dot(p2[rotation])) > triangle_polarity_threshold &&
           std::abs(p1[1].dot(p2[(rotation + 1) % 3])) >
               triangle_polarity_threshold &&
           std::abs(p1[2].dot(p2[(rotation + 2) % 3])) >
               triangle_polarity_threshold;
  }



public:
  double x, y;
  std::complex<double> a1, a2, a3; // these can be possibly complex numbers ...
  double s, det;
};
}
}
#endif