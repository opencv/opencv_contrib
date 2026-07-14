#ifndef SLAM_OPTIMIZER_G2O_SIM3_GRAPH_OPT_EDGE_H
#define SLAM_OPTIMIZER_G2O_SIM3_GRAPH_OPT_EDGE_H

#include "type.hpp"
#include "optimize/internal/sim3/shot_vertex.hpp"

#include <g2o/core/base_binary_edge.h>

namespace cv::slam {
namespace optimize {
namespace internal {
namespace sim3 {

class graph_opt_edge final : public g2o::BaseBinaryEdge<7, g2o::Sim3, shot_vertex, shot_vertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    graph_opt_edge();

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void computeError() override;

    double initialEstimatePossible(const g2o::OptimizableGraph::VertexSet&, g2o::OptimizableGraph::Vertex*) override;

    void initialEstimate(const g2o::OptimizableGraph::VertexSet& from, g2o::OptimizableGraph::Vertex*) override;
};

inline graph_opt_edge::graph_opt_edge()
    : g2o::BaseBinaryEdge<7, g2o::Sim3, shot_vertex, shot_vertex>() {}

inline bool graph_opt_edge::read(std::istream& is) {
    Vec7_t sim3_wc;
    for (unsigned int i = 0; i < 7; ++i) {
        is >> sim3_wc(i);
    }
    const g2o::Sim3 Sim3_wc(sim3_wc);
    setMeasurement(Sim3_wc.inverse());
    for (int i = 0; i < information().rows(); ++i) {
        for (int j = i; j < information().cols(); ++j) {
            is >> information()(i, j);
            if (i != j) {
                information()(j, i) = information()(i, j);
            }
        }
    }
    return true;
}

inline bool graph_opt_edge::write(std::ostream& os) const {
    g2o::Sim3 Sim3_wc(measurement().inverse());
    const auto sim3_wc = Sim3_wc.log();
    for (unsigned int i = 0; i < 7; ++i) {
        os << sim3_wc(i) << " ";
    }
    for (int i = 0; i < information().rows(); ++i) {
        for (int j = i; j < information().cols(); ++j) {
            os << " " << information()(i, j);
        }
    }
    return os.good();
}

inline void graph_opt_edge::computeError() {
    const auto v1 = static_cast<const shot_vertex*>(_vertices.at(0));
    const auto v2 = static_cast<const shot_vertex*>(_vertices.at(1));

    const g2o::Sim3 C(_measurement);
    const g2o::Sim3 error_ = C * v1->estimate() * v2->estimate().inverse();
    _error = error_.log();
}

inline double graph_opt_edge::initialEstimatePossible(const g2o::OptimizableGraph::VertexSet&, g2o::OptimizableGraph::Vertex*) {
    return 1.0;
}

inline void graph_opt_edge::initialEstimate(const g2o::OptimizableGraph::VertexSet& from, g2o::OptimizableGraph::Vertex*) {
    auto v1 = static_cast<shot_vertex*>(_vertices[0]);
    auto v2 = static_cast<shot_vertex*>(_vertices[1]);
    if (0 < from.count(v1)) {
        v2->setEstimate(measurement() * v1->estimate());
    }
    else {
        v1->setEstimate(measurement().inverse() * v2->estimate());
    }
}

} // namespace sim3
} // namespace internal
} // namespace optimize
} // namespace cv::slam

#endif // SLAM_OPTIMIZER_G2O_SIM3_GRAPH_OPT_EDGE_H
