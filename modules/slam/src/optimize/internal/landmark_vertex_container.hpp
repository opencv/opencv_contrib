#ifndef SLAM_OPTIMIZE_G2O_LANDMARK_VERTEX_CONTAINER_H
#define SLAM_OPTIMIZE_G2O_LANDMARK_VERTEX_CONTAINER_H

#include "type.hpp"
#include "data/landmark.hpp"
#include "optimize/internal/landmark_vertex.hpp"

#include <unordered_map>
#include <memory>

namespace cv::slam {

namespace data {
class landmark;
} // namespace data

namespace optimize {
namespace internal {

class landmark_vertex_container {
public:
    //! Constructor
    explicit landmark_vertex_container(const std::shared_ptr<unsigned int> offset, const unsigned int num_reserve = 200);

    //! Destructor
    virtual ~landmark_vertex_container() = default;

    //! Create and return the g2o vertex created from the specified landmark
    landmark_vertex* create_vertex(const std::shared_ptr<data::landmark>& lm, const bool is_constant);

    //! Create and return the g2o vertex created from the specified landmark
    landmark_vertex* create_vertex(const unsigned int id, const Vec3_t& pos_w, const bool is_constant);

    //! Get vertex corresponding with the specified landmark
    landmark_vertex* get_vertex(const std::shared_ptr<data::landmark>& lm) const;

    //! Get vertex corresponding with the specified landmark ID
    landmark_vertex* get_vertex(const unsigned int id) const;

    //! Convert landmark to vertex ID
    unsigned int get_vertex_id(const std::shared_ptr<data::landmark>& lm) const;

    //! Convert landmark ID to vertex ID
    unsigned int get_vertex_id(const unsigned int id) const;

    //! Convert vertex to landmark ID
    unsigned int get_id(landmark_vertex* vtx) const;

    //! Convert vertex ID to landmark ID
    unsigned int get_id(const unsigned int vtx_id) const;

    //! Contains the specified landmark or not
    bool contain(const std::shared_ptr<data::landmark>& lm) const;

    // iterators to sweep landmark vertices
    using iterator = std::unordered_map<unsigned int, landmark_vertex*>::iterator;
    using const_iterator = std::unordered_map<unsigned int, landmark_vertex*>::const_iterator;
    iterator begin();
    const_iterator begin() const;
    iterator end();
    const_iterator end() const;

private:
    //! vertex ID = offset + landmark ID
    const std::shared_ptr<unsigned int> offset_ = nullptr;

    //! key: landmark ID, value: vertex
    std::unordered_map<unsigned int, landmark_vertex*> vtx_container_;

    //! key: landmark ID, value: vertex ID
    std::unordered_map<unsigned int, unsigned int> vtx_id_container_;

    //! key: vertex ID, value: frame/keyframe ID
    std::unordered_map<unsigned int, unsigned int> id_container_;
};

inline landmark_vertex_container::landmark_vertex_container(const std::shared_ptr<unsigned int> offset, const unsigned int num_reserve)
    : offset_(offset) {
    vtx_container_.reserve(num_reserve);
    vtx_id_container_.reserve(num_reserve);
    id_container_.reserve(num_reserve);
}

inline landmark_vertex* landmark_vertex_container::create_vertex(const std::shared_ptr<data::landmark>& lm, const bool is_constant) {
    return create_vertex(lm->id_, lm->get_pos_in_world(), is_constant);
}

inline landmark_vertex* landmark_vertex_container::create_vertex(const unsigned int id, const Vec3_t& pos_w, const bool is_constant) {

    const auto vtx_id = *offset_;
    (*offset_)++;
    auto vtx = new landmark_vertex();
    vtx->setId(vtx_id);
    vtx->setEstimate(pos_w);
    vtx->setFixed(is_constant);
    vtx->setMarginalized(true);

    id_container_[vtx_id] = id;
    vtx_id_container_[id] = vtx_id;
    vtx_container_[id] = vtx;

    return vtx;
}

inline landmark_vertex* landmark_vertex_container::get_vertex(const std::shared_ptr<data::landmark>& lm) const {
    return get_vertex(lm->id_);
}

inline landmark_vertex* landmark_vertex_container::get_vertex(const unsigned int id) const {
    return vtx_container_.at(id);
}

inline unsigned int landmark_vertex_container::get_vertex_id(const std::shared_ptr<data::landmark>& lm) const {
    return get_vertex_id(lm->id_);
}

inline unsigned int landmark_vertex_container::get_vertex_id(const unsigned int id) const {
    return vtx_id_container_.at(id);
}

inline unsigned int landmark_vertex_container::get_id(landmark_vertex* vtx) const {
    return get_id(vtx->id());
}

inline unsigned int landmark_vertex_container::get_id(const unsigned int vtx_id) const {
    return id_container_.at(vtx_id);
}

inline bool landmark_vertex_container::contain(const std::shared_ptr<data::landmark>& lm) const {
    return 0 != vtx_container_.count(lm->id_);
}

inline landmark_vertex_container::iterator landmark_vertex_container::begin() {
    return vtx_container_.begin();
}

inline landmark_vertex_container::const_iterator landmark_vertex_container::begin() const {
    return vtx_container_.begin();
}

inline landmark_vertex_container::iterator landmark_vertex_container::end() {
    return vtx_container_.end();
}

inline landmark_vertex_container::const_iterator landmark_vertex_container::end() const {
    return vtx_container_.end();
}

} // namespace internal
} // namespace optimize
} // namespace cv::slam

#endif // SLAM_OPTIMIZE_G2O_LANDMARK_VERTEX_CONTAINER_H
