#ifndef SADDLE_CLUSTER
#define SADDLE_CLUSTER

#include <vector>


struct SaddleClusterDesc {
  typedef struct SaddlePoint PointType;

  int id;
  std::vector<int> idxs;
  double cx, cy;
  double a1, a2;
  SaddleClusterDesc() : id(-1), cx(0.0), cy(0.0) {}

  static bool sortByClusterSize(const SaddleClusterDesc &a,
                                const SaddleClusterDesc &b) {
    return a.idxs.size() < b.idxs.size();
  }

  static bool clusterUnused(const SaddleClusterDesc &c) { return c.id < 0; }

  static bool isTriangular() { return false; }

  template <typename PointType>
  void computeClusterMeans(const std::vector<PointType> &pts) {
    int sigp = 0, sigm = 0;
    std::vector<int> sgn(idxs.size());
    for (size_t j = 0; j < idxs.size(); j++) {
      double cost = (cos(pts[idxs[j]].a1) * cos(pts[idxs[0]].a1) +
                     sin(pts[idxs[j]].a1) * sin(pts[idxs[0]].a1));
      if (cost < 0) {
        sgn[j] = -1;
        sigm++;
      } else {
        sgn[j] = 1;
        sigp++;
      }
    }
    double sign = sigp < sigm ? -1 : 1;
    // reset means
    a1 = 0, a2 = 0;
    cx = 0;
    cy = 0;
    double c1 = 0, c2 = 0;

    for (size_t j = 0; j < idxs.size(); j++) {
      const PointType &pt = pts[idxs[j]];
      cx += pt.x;
      cy += pt.y;

      a1 += sign * sgn[j] * sin(pt.a1);
      c1 += sign * sgn[j] * cos(pt.a1);

      a2 += sin(pt.a2);
      c2 += cos(pt.a2);
    }
    a1 = atan(a1 / c1);
    a2 = atan(a2 / c2);

    cx /= idxs.size();
    cy /= idxs.size();
  }
};


struct MonkeySaddleClusterDesc {
  typedef struct MonkeySaddlePoint PointType;
  int id;
  std::vector<int> idxs;
  double cx, cy;
  std::complex<double> a1, a2, a3;

  MonkeySaddleClusterDesc() : id(-1), cx(0.0), cy(0.0) {}

  static bool sortByClusterSize(const MonkeySaddleClusterDesc &a,
                                const MonkeySaddleClusterDesc &b) {
    return a.idxs.size() < b.idxs.size();
  }

  static bool clusterUnused(const MonkeySaddleClusterDesc &c) {
    return c.id < 0;
  }

  static bool isTriangular() { return true; }

  template <typename PointType>
  void computeClusterMeans(const std::vector<PointType> &pts) {
    std::complex<double> c1 = 0, c2 = 0, c3 = 0;
    a1 = 0;
    a2 = 0;
    a3 = 0;
    cx = 0;
    cy = 0;
    for (size_t j = 0; j < idxs.size(); j++) {
      const PointType &pt = pts[idxs[j]];
      cx += pt.x;
      cy += pt.y;

      a1 += sin(pt.a1);
      c1 += cos(pt.a1);

      a2 += sin(pt.a2);
      c2 += cos(pt.a2);

      a3 += sin(pt.a3);
      c3 += cos(pt.a3);
    }
    a1 = atan(a1 / c1);
    a2 = atan(a2 / c2);
    a3 = atan(a3 / c3);

    cx /= idxs.size();
    cy /= idxs.size();
  }
};


#endif