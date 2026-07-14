#ifndef SLAM_OPTIMIZER_G2O_SIM3_SHOT_VERTEX_H
#define SLAM_OPTIMIZER_G2O_SIM3_SHOT_VERTEX_H

#include "type.hpp"

#include <g2o/core/base_vertex.h>
#include <g2o/types/sim3/sim3.h>

namespace cv::slam {
namespace optimize {
namespace internal {
namespace sim3 {

class shot_vertex final : public g2o::BaseVertex<7, g2o::Sim3> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    shot_vertex();

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void setToOriginImpl() override;

    void oplusImpl(const double* update_) override;

    bool fix_scale_;
};

inline shot_vertex::shot_vertex()
    : g2o::BaseVertex<7, g2o::Sim3>() {}

inline bool shot_vertex::read(std::istream& is) {
    Vec7_t g2o_sim3_wc;
    for (int i = 0; i < 7; ++i) {
        is >> g2o_sim3_wc(i);
    }
    setEstimate(g2o::Sim3(g2o_sim3_wc).inverse());
    return true;
}

inline bool shot_vertex::write(std::ostream& os) const {
    g2o::Sim3 g2o_Sim3_wc(estimate().inverse());
    const Vec7_t g2o_sim3_wc = g2o_Sim3_wc.log();
    for (int i = 0; i < 7; ++i) {
        os << g2o_sim3_wc(i) << " ";
    }
    return os.good();
}

inline void shot_vertex::setToOriginImpl() {
    _estimate = g2o::Sim3();
}

inline void shot_vertex::oplusImpl(const double* update_) {
    Eigen::Map<Vec7_t> update(const_cast<double*>(update_));

    if (fix_scale_) {
        update(6) = 0;
    }

    const g2o::Sim3 s(update);
    setEstimate(s * estimate());
}

} // namespace sim3
} // namespace internal
} // namespace optimize
} // namespace cv::slam

#endif // SLAM_OPTIMIZER_G2O_SIM3_SHOT_VERTEX_H
