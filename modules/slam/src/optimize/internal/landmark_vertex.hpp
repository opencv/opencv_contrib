#ifndef SLAM_OPTIMIZER_G2O_LANDMARK_VERTEX_H
#define SLAM_OPTIMIZER_G2O_LANDMARK_VERTEX_H

#include "type.hpp"

#include <g2o/core/base_vertex.h>

namespace cv::slam {
namespace optimize {
namespace internal {

class landmark_vertex final : public g2o::BaseVertex<3, Vec3_t> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    landmark_vertex();

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void setToOriginImpl() override;

    void oplusImpl(const double* update) override;
};

inline landmark_vertex::landmark_vertex()
    : g2o::BaseVertex<3, Vec3_t>() {}

inline bool landmark_vertex::read(std::istream& is) {
    Vec3_t lv;
    for (unsigned int i = 0; i < 3; ++i) {
        is >> _estimate(i);
    }
    return true;
}

inline bool landmark_vertex::write(std::ostream& os) const {
    const Vec3_t pos_w = estimate();
    for (unsigned int i = 0; i < 3; ++i) {
        os << pos_w(i) << " ";
    }
    return os.good();
}

inline void landmark_vertex::setToOriginImpl() {
    _estimate.fill(0);
}

inline void landmark_vertex::oplusImpl(const double* update) {
    Eigen::Map<const Vec3_t> v(update);
    _estimate += v;
}

} // namespace internal
} // namespace optimize
} // namespace cv::slam

#endif // SLAM_OPTIMIZER_G2O_LANDMARK_VERTEX_H
