#ifndef SLAM_OPTIMIZE_G2O_MARKER_VERTEX_CONTAINER_H
#define SLAM_OPTIMIZE_G2O_MARKER_VERTEX_CONTAINER_H

#include "type.hpp"
#include "data/marker.hpp"
#include "optimize/internal/landmark_vertex.hpp"

#include <unordered_map>
#include <memory>

namespace cv::slam {

namespace data {
class marker;
} // namespace data

namespace optimize {
namespace internal {

class marker_vertex_container {
public:
    //! Constructor
    explicit marker_vertex_container(const std::shared_ptr<unsigned int> offset, const unsigned int num_reserve = 200);

    //! Destructor
    virtual ~marker_vertex_container() = default;

    //! Create and return the g2o vertex created from the specified marker
    std::vector<landmark_vertex*> create_vertices(const std::shared_ptr<data::marker>& mkr, const bool is_constant);

    //! Get vertex corresponding with the specified marker
    landmark_vertex* get_vertex(const std::shared_ptr<data::marker>& mkr, const unsigned int corner_id) const;

    //! Get vertex corresponding with the specified marker ID
    landmark_vertex* get_vertex(const unsigned int marker_id, const unsigned int corner_id) const;

private:
    landmark_vertex* create_vertex(const unsigned int marker_id, const unsigned int corner_id,
                                   const Vec3_t& pos_w, const bool is_constant);

    //! vertex ID = offset + marker ID * 4 + corner ID
    const std::shared_ptr<unsigned int> offset_ = nullptr;

    //! key: marker ID, value: vertexs
    std::unordered_map<unsigned int, std::vector<landmark_vertex*>> vtx_container_;

    //! key: marker ID, value: vertex IDs
    std::unordered_map<unsigned int, std::vector<unsigned int>> vtx_id_container_;

    //! key: vertex ID, value: marker ID and corner ID
    std::unordered_map<unsigned int, std::pair<unsigned int, unsigned int>> id_container_;
};

inline marker_vertex_container::marker_vertex_container(const std::shared_ptr<unsigned int> offset, const unsigned int num_reserve)
    : offset_(offset) {
    vtx_container_.reserve(num_reserve);
    vtx_id_container_.reserve(num_reserve);
    id_container_.reserve(num_reserve);
}

inline std::vector<landmark_vertex*> marker_vertex_container::create_vertices(const std::shared_ptr<data::marker>& mkr, const bool is_constant) {
    std::vector<landmark_vertex*> vertices;
    std::vector<unsigned int> vtx_ids;
    for (unsigned int i = 0; i < mkr->corners_pos_w_.size(); ++i) {
        vtx_ids.push_back(*offset_);
        vertices.push_back(create_vertex(mkr->id_, i, mkr->corners_pos_w_[i], is_constant));
    }
    vtx_id_container_[mkr->id_] = vtx_ids;
    vtx_container_[mkr->id_] = vertices;
    return vertices;
}

inline landmark_vertex* marker_vertex_container::create_vertex(const unsigned int marker_id, const unsigned int corner_id,
                                                               const Vec3_t& pos_w, const bool is_constant) {
    // Create vertex
    const auto vtx_id = *offset_;
    (*offset_)++;
    auto vtx = new landmark_vertex();
    vtx->setId(vtx_id);
    vtx->setEstimate(pos_w);
    vtx->setFixed(is_constant);
    vtx->setMarginalized(true);
    // Set to id database
    id_container_[vtx_id] = std::make_pair(marker_id, corner_id);
    return vtx;
}

inline landmark_vertex* marker_vertex_container::get_vertex(const std::shared_ptr<data::marker>& mkr, const unsigned int corner_id) const {
    return get_vertex(mkr->id_, corner_id);
}

inline landmark_vertex* marker_vertex_container::get_vertex(const unsigned int marker_id, const unsigned int corner_id) const {
    return vtx_container_.at(marker_id)[corner_id];
}

} // namespace internal
} // namespace optimize
} // namespace cv::slam

#endif // SLAM_OPTIMIZE_G2O_MARKER_VERTEX_CONTAINER_H
