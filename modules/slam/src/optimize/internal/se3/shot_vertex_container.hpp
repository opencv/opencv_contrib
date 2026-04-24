#ifndef SLAM_OPTIMIZE_G2O_SE3_SHOT_VERTEX_CONTAINER_H
#define SLAM_OPTIMIZE_G2O_SE3_SHOT_VERTEX_CONTAINER_H

#include "type.hpp"
#include "data/frame.hpp"
#include "data/keyframe.hpp"
#include "optimize/internal/se3/shot_vertex.hpp"

#include <unordered_map>
#include <memory>

namespace cv::slam {

namespace data {
class frame;
class keyframe;
} // namespace data

namespace optimize {
namespace internal {
namespace se3 {

class shot_vertex_container {
public:
    //! Constructor
    explicit shot_vertex_container(const std::shared_ptr<unsigned int> offset, const unsigned int num_reserve = 50);

    //! Destructor
    virtual ~shot_vertex_container() = default;

    //! Create and return the g2o vertex created from the specified frame
    shot_vertex* create_vertex(data::frame* frm, const bool is_constant);

    //! Create and return the g2o vertex created from the specified keyframe
    shot_vertex* create_vertex(const std::shared_ptr<data::keyframe>& keyfrm, const bool is_constant);

    //! Create and return the g2o vertex created from shot ID and camera pose
    shot_vertex* create_vertex(const unsigned int id, const Mat44_t& pose_cw, const bool is_constant);

    //! Get vertex corresponding with the specified frame
    shot_vertex* get_vertex(data::frame* frm) const;

    //! Get vertex corresponding with the specified keyframe
    shot_vertex* get_vertex(const std::shared_ptr<data::keyframe>& keyfrm) const;

    //! Get vertex corresponding with the specified shot (frame/keyframe) ID
    shot_vertex* get_vertex(const unsigned int id) const;

    //! Convert frame ID to vertex ID
    unsigned int get_vertex_id(data::frame* frm) const;

    //! Convert keyframe ID to vertex ID
    unsigned int get_vertex_id(const std::shared_ptr<data::keyframe>& keyfrm) const;

    //! Convert shot (frame/keyframe) ID to vertex ID
    unsigned int get_vertex_id(unsigned int id) const;

    //! Convert vertex ID to shot (frame/keyframe) ID
    unsigned int get_id(shot_vertex* vtx);

    //! Convert vertex ID to shot (frame/keyframe) ID
    unsigned int get_id(unsigned int vtx_id) const;

    //! Contains the specified keyframe or not
    bool contain(const std::shared_ptr<data::keyframe>& keyfrm) const;

    // iterators to sweep shot vertices
    using iterator = std::unordered_map<unsigned int, shot_vertex*>::iterator;
    using const_iterator = std::unordered_map<unsigned int, shot_vertex*>::const_iterator;
    iterator begin();
    const_iterator begin() const;
    iterator end();
    const_iterator end() const;

private:
    const std::shared_ptr<unsigned int> offset_ = nullptr;

    //! key: vertex ID, value: vertex
    std::unordered_map<unsigned int, shot_vertex*> vtx_container_;

    //! key: frame/keyframe ID, value: vertex ID
    std::unordered_map<unsigned int, unsigned int> vtx_id_container_;

    //! key: vertex ID, value: frame/keyframe ID
    std::unordered_map<unsigned int, unsigned int> id_container_;
};

inline shot_vertex_container::shot_vertex_container(const std::shared_ptr<unsigned int> offset, const unsigned int num_reserve)
    : offset_(offset) {
    vtx_container_.reserve(num_reserve);
    vtx_id_container_.reserve(num_reserve);
    id_container_.reserve(num_reserve);
}

inline shot_vertex* shot_vertex_container::create_vertex(data::frame* frm, const bool is_constant) {
    return create_vertex(frm->id_, frm->get_pose_cw(), is_constant);
}

inline shot_vertex* shot_vertex_container::create_vertex(const std::shared_ptr<data::keyframe>& keyfrm, const bool is_constant) {
    return create_vertex(keyfrm->id_, keyfrm->get_pose_cw(), is_constant);
}

inline shot_vertex* shot_vertex_container::create_vertex(const unsigned int id, const Mat44_t& pose_cw, const bool is_constant) {
    
    const auto vtx_id = *offset_;
    (*offset_)++;
    auto vtx = new shot_vertex();
    vtx->setId(vtx_id);
    vtx->setEstimate(util::converter::to_g2o_SE3(pose_cw));
    vtx->setFixed(is_constant);
    
    id_container_[vtx_id] = id;
    vtx_id_container_[id] = vtx_id;
    vtx_container_[id] = vtx;
    
    return vtx;
}

inline shot_vertex* shot_vertex_container::get_vertex(data::frame* frm) const {
    return get_vertex(frm->id_);
}

inline shot_vertex* shot_vertex_container::get_vertex(const std::shared_ptr<data::keyframe>& keyfrm) const {
    return get_vertex(keyfrm->id_);
}

inline shot_vertex* shot_vertex_container::get_vertex(const unsigned int id) const {
    return vtx_container_.at(id);
}

inline unsigned int shot_vertex_container::get_vertex_id(data::frame* frm) const {
    return get_vertex_id(frm->id_);
}

inline unsigned int shot_vertex_container::get_vertex_id(const std::shared_ptr<data::keyframe>& keyfrm) const {
    return get_vertex_id(keyfrm->id_);
}

inline unsigned int shot_vertex_container::get_vertex_id(unsigned int id) const {
    return vtx_id_container_.at(id);
}

inline unsigned int shot_vertex_container::get_id(shot_vertex* vtx) {
    return get_id(vtx->id());
}

inline unsigned int shot_vertex_container::get_id(unsigned int vtx_id) const {
    return id_container_.at(vtx_id);
}

inline bool shot_vertex_container::contain(const std::shared_ptr<data::keyframe>& keyfrm) const {
    return 0 != vtx_container_.count(keyfrm->id_);
}

inline shot_vertex_container::iterator shot_vertex_container::begin() {
    return vtx_container_.begin();
}

inline shot_vertex_container::const_iterator shot_vertex_container::begin() const {
    return vtx_container_.begin();
}

inline shot_vertex_container::iterator shot_vertex_container::end() {
    return vtx_container_.end();
}

inline shot_vertex_container::const_iterator shot_vertex_container::end() const {
    return vtx_container_.end();
}

} // namespace se3
} // namespace internal
} // namespace optimize
} // namespace cv::slam

#endif // SLAM_OPTIMIZE_G2O_SE3_SHOT_VERTEX_CONTAINER_H
