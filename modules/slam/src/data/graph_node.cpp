#include "data/keyframe.hpp"
#include "data/graph_node.hpp"
#include "data/landmark.hpp"

namespace cv::slam {
namespace data {

graph_node::graph_node(std::shared_ptr<keyframe>& keyfrm)
    : owner_keyfrm_(keyfrm) {}

void graph_node::add_connection(const std::shared_ptr<keyframe>& keyfrm, const unsigned int num_shared_lms) {
    std::lock_guard<std::mutex> lock(mtx_);
    bool need_update = false;
    if (!connected_keyfrms_and_num_shared_lms_.count(keyfrm)) {
        // if `keyfrm` not exists
        connected_keyfrms_and_num_shared_lms_[keyfrm] = num_shared_lms;
        need_update = true;
    }
    else if (connected_keyfrms_and_num_shared_lms_.at(keyfrm) != num_shared_lms) {
        // if the number of shared landmarks is updated
        connected_keyfrms_and_num_shared_lms_.at(keyfrm) = num_shared_lms;
        need_update = true;
    }

    if (need_update) {
        update_covisibility_orders_impl();
    }
}

void graph_node::erase_connection(const std::shared_ptr<keyframe>& keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_);
    bool need_update = false;
    if (connected_keyfrms_and_num_shared_lms_.count(keyfrm)) {
        connected_keyfrms_and_num_shared_lms_.erase(keyfrm);
        need_update = true;
    }

    if (need_update) {
        update_covisibility_orders_impl();
    }
}

void graph_node::erase_all_connections() {
    // remote myself from the connected keyframes
    for (const auto& keyfrm_and_num_shared_lms : connected_keyfrms_and_num_shared_lms_) {
        if (keyfrm_and_num_shared_lms.first.expired()) {
            continue;
        }
        keyfrm_and_num_shared_lms.first.lock()->graph_node_->erase_connection(owner_keyfrm_.lock());
    }
    // remove the buffers
    connected_keyfrms_and_num_shared_lms_.clear();
    ordered_covisibilities_.clear();
    ordered_num_shared_lms_.clear();
}

void graph_node::update_connections(unsigned int min_num_shared_lms) {
    const auto owner_keyfrm = owner_keyfrm_.lock();
    const auto landmarks = owner_keyfrm->get_landmarks();

    id_ordered_map<std::weak_ptr<keyframe>, unsigned int> keyfrm_to_num_shared_lms;

    for (const auto& lm : landmarks) {
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        const auto observations = lm->get_observations();

        for (const auto& obs : observations) {
            auto keyfrm = obs.first;
            auto locked_keyfrm = keyfrm.lock();

            if (locked_keyfrm->graph_node_->spanning_parent_.expired() && !locked_keyfrm->graph_node_->is_spanning_root()) {
                continue;
            }
            if (locked_keyfrm->id_ == owner_keyfrm->id_) {
                continue;
            }
            // count up number of shared landmarks of `keyfrm`
            keyfrm_to_num_shared_lms[keyfrm]++;
        }
    }

    if (keyfrm_to_num_shared_lms.empty()) {
        return;
    }

    unsigned int max_num_shared_lms = 0;
    std::shared_ptr<keyframe> nearest_covisibility = nullptr;

    // vector for sorting
    std::vector<std::pair<unsigned int, std::shared_ptr<keyframe>>> num_shared_lms_and_covisibility_pairs;
    num_shared_lms_and_covisibility_pairs.reserve(keyfrm_to_num_shared_lms.size());
    for (const auto& keyfrm_and_num_shared_lms : keyfrm_to_num_shared_lms) {
        auto keyfrm = keyfrm_and_num_shared_lms.first.lock();
        const auto num_shared_lms = keyfrm_and_num_shared_lms.second;

        // nearest_covisibility with greatest id_ will be selected if number of shared landmarks are the same due to ordering of keyfrm_and_num_shared_lms.
        if (max_num_shared_lms <= num_shared_lms) {
            max_num_shared_lms = num_shared_lms;
            nearest_covisibility = keyfrm;
        }

        if (min_num_shared_lms < num_shared_lms) {
            num_shared_lms_and_covisibility_pairs.emplace_back(std::make_pair(num_shared_lms, keyfrm));
        }
    }
    // add ONE node at least
    if (num_shared_lms_and_covisibility_pairs.empty()) {
        num_shared_lms_and_covisibility_pairs.emplace_back(std::make_pair(max_num_shared_lms, nearest_covisibility));
    }

    // add connection from the covisibility to myself
    for (const auto& num_shared_lms_and_covisibility : num_shared_lms_and_covisibility_pairs) {
        auto covisibility = num_shared_lms_and_covisibility.second;
        const auto num_shared_lms = num_shared_lms_and_covisibility.first;
        covisibility->graph_node_->add_connection(owner_keyfrm, num_shared_lms);
    }

    // sort with number of shared landmarks and keyframe IDs for consistency; IDs are also in reverse order
    // to match selection of nearest_covisibility.
    std::sort(num_shared_lms_and_covisibility_pairs.rbegin(), num_shared_lms_and_covisibility_pairs.rend(),
              less_number_and_id_object_pairs<unsigned int, data::keyframe>());

    decltype(ordered_covisibilities_) ordered_covisibilities;
    ordered_covisibilities.reserve(num_shared_lms_and_covisibility_pairs.size());
    decltype(ordered_num_shared_lms_) ordered_num_shared_lms;
    ordered_num_shared_lms.reserve(num_shared_lms_and_covisibility_pairs.size());
    for (const auto& num_shared_lms_and_keyfrm_pair : num_shared_lms_and_covisibility_pairs) {
        ordered_covisibilities.push_back(num_shared_lms_and_keyfrm_pair.second);
        ordered_num_shared_lms.push_back(num_shared_lms_and_keyfrm_pair.first);
    }

    {
        std::lock_guard<std::mutex> lock(mtx_);

        connected_keyfrms_and_num_shared_lms_ = decltype(connected_keyfrms_and_num_shared_lms_)(keyfrm_to_num_shared_lms.begin(), keyfrm_to_num_shared_lms.end());

        ordered_covisibilities_ = ordered_covisibilities;
        ordered_num_shared_lms_ = ordered_num_shared_lms;

        if (spanning_parent_.expired() && !is_spanning_root_impl()) {
            // set the parent of spanning tree
            assert(nearest_covisibility->id_ == ordered_covisibilities.front().lock()->id_);
            spanning_parent_ = nearest_covisibility;
            spanning_root_ = spanning_parent_.lock()->graph_node_->get_spanning_root_impl();
            nearest_covisibility->graph_node_->add_spanning_child(owner_keyfrm);
        }
    }
}

void graph_node::update_covisibility_orders() {
    std::lock_guard<std::mutex> lock(mtx_);
    update_covisibility_orders_impl();
}

void graph_node::update_covisibility_orders_impl() {
    std::vector<std::pair<unsigned int, std::shared_ptr<keyframe>>> num_shared_lms_and_keyfrm_pairs;
    num_shared_lms_and_keyfrm_pairs.reserve(connected_keyfrms_and_num_shared_lms_.size());

    for (const auto& keyfrm_and_num_shared_lms : connected_keyfrms_and_num_shared_lms_) {
        num_shared_lms_and_keyfrm_pairs.emplace_back(std::make_pair(keyfrm_and_num_shared_lms.second, keyfrm_and_num_shared_lms.first.lock()));
    }

    // sort with number of shared landmarks and keyframe IDs for consistency
    std::sort(num_shared_lms_and_keyfrm_pairs.rbegin(), num_shared_lms_and_keyfrm_pairs.rend(),
              less_number_and_id_object_pairs<unsigned int, data::keyframe>());

    ordered_covisibilities_.clear();
    ordered_covisibilities_.reserve(num_shared_lms_and_keyfrm_pairs.size());
    ordered_num_shared_lms_.clear();
    ordered_num_shared_lms_.reserve(num_shared_lms_and_keyfrm_pairs.size());
    for (const auto& num_shared_lms_and_keyfrm_pair : num_shared_lms_and_keyfrm_pairs) {
        ordered_covisibilities_.push_back(num_shared_lms_and_keyfrm_pair.second);
        ordered_num_shared_lms_.push_back(num_shared_lms_and_keyfrm_pair.first);
    }
}

std::set<std::shared_ptr<keyframe>> graph_node::get_connected_keyframes() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::set<std::shared_ptr<keyframe>> keyfrms;

    for (const auto& keyfrm_and_num_shared_lms : connected_keyfrms_and_num_shared_lms_) {
        keyfrms.insert(keyfrm_and_num_shared_lms.first.lock());
    }

    return keyfrms;
}

std::vector<std::shared_ptr<keyframe>> graph_node::get_covisibilities() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<std::shared_ptr<keyframe>> covisibilities;

    for (const auto& covisibility : ordered_covisibilities_) {
        if (covisibility.expired()) {
            continue;
        }
        covisibilities.push_back(covisibility.lock());
    }
    return covisibilities;
}

std::vector<std::shared_ptr<keyframe>> graph_node::get_top_n_covisibilities(const unsigned int num_covisibilities) const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<std::shared_ptr<keyframe>> covisibilities;
    unsigned int i = 0;
    for (const auto& covisibility : ordered_covisibilities_) {
        if (i == num_covisibilities) {
            break;
        }
        if (covisibility.expired()) {
            continue;
        }
        covisibilities.push_back(covisibility.lock());
        i++;
    }
    return covisibilities;
}

std::vector<std::shared_ptr<keyframe>> graph_node::get_covisibilities_over_min_num_shared_lms(const unsigned int min_num_shared_lms) const {
    std::lock_guard<std::mutex> lock(mtx_);

    if (ordered_covisibilities_.empty()) {
        return std::vector<std::shared_ptr<keyframe>>();
    }

    auto itr = std::upper_bound(ordered_num_shared_lms_.begin(), ordered_num_shared_lms_.end(), min_num_shared_lms, std::greater<unsigned int>());
    if (itr == ordered_num_shared_lms_.end()) {
        std::vector<std::shared_ptr<keyframe>> covisibilities;
        for (const auto& covisibility : ordered_covisibilities_) {
            if (covisibility.expired()) {
                continue;
            }
            covisibilities.push_back(covisibility.lock());
        }
        return covisibilities;
    }
    else {
        const auto upper_bound_idx = static_cast<unsigned int>(itr - ordered_num_shared_lms_.begin());
        std::vector<std::shared_ptr<keyframe>> covisibilities;
        unsigned int idx = 0;
        for (const auto& covisibility : ordered_covisibilities_) {
            if (idx == upper_bound_idx) {
                break;
            }
            idx++;
            if (covisibility.expired()) {
                continue;
            }
            covisibilities.push_back(covisibility.lock());
        }
        return covisibilities;
    }
}

unsigned int graph_node::get_num_shared_landmarks(const std::shared_ptr<keyframe>& keyfrm) const {
    std::lock_guard<std::mutex> lock(mtx_);
    if (connected_keyfrms_and_num_shared_lms_.count(keyfrm)) {
        return connected_keyfrms_and_num_shared_lms_.at(keyfrm);
    }
    else {
        return 0;
    }
}

void graph_node::set_spanning_parent(const std::shared_ptr<keyframe>& keyfrm) {
    // NOTE: keyfrm can be nullptr
    std::lock_guard<std::mutex> lock(mtx_);
    assert(spanning_parent_.expired());
    spanning_parent_ = keyfrm;
}

std::shared_ptr<keyframe> graph_node::get_spanning_parent() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return spanning_parent_.lock();
}

void graph_node::change_spanning_parent(const std::shared_ptr<keyframe>& keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_);
    spanning_parent_ = keyfrm;
    keyfrm->graph_node_->add_spanning_child(owner_keyfrm_.lock());
}

void graph_node::add_spanning_child(const std::shared_ptr<keyframe>& keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_);
    spanning_children_.insert(keyfrm);
}

void graph_node::erase_spanning_child(const std::shared_ptr<keyframe>& keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_);
    spanning_children_.erase(keyfrm);
}

void graph_node::recover_spanning_connections() {
    std::lock_guard<std::mutex> lock(mtx_);

    // 1. find new parents for my children

    std::set<std::shared_ptr<keyframe>> new_parent_candidates;
    new_parent_candidates.insert(spanning_parent_.lock());

    while (!spanning_children_.empty()) {
        bool max_is_found = false;

        unsigned int max_num_shared_lms = 0;
        std::shared_ptr<keyframe> max_num_shared_lms_parent = nullptr;
        std::shared_ptr<keyframe> max_num_shared_lms_child = nullptr;

        for (const auto& spanning_child : spanning_children_) {
            auto locked_spanning_child = spanning_child.lock();
            if (locked_spanning_child->will_be_erased()) {
                continue;
            }

            // get intersection between the parent candidates and the spanning-child's covisibilities
            const auto child_covisibilities = locked_spanning_child->graph_node_->get_covisibilities();
            const auto intersection = extract_intersection(new_parent_candidates, child_covisibilities);

            // find the new parent (which has the maximum number of shared landmarks with the spanning child) from the intersection
            for (const auto& parent_candidate : intersection) {
                const auto num_shared_lms = locked_spanning_child->graph_node_->get_num_shared_landmarks(parent_candidate);
                if (max_num_shared_lms < num_shared_lms) {
                    max_num_shared_lms = num_shared_lms;
                    max_num_shared_lms_parent = parent_candidate;
                    max_num_shared_lms_child = locked_spanning_child;
                    max_is_found = true;
                }
            }
        }

        if (max_is_found) {
            // update spanning tree
            max_num_shared_lms_child->graph_node_->change_spanning_parent(max_num_shared_lms_parent);
            spanning_children_.erase(max_num_shared_lms_child);
            new_parent_candidates.insert(max_num_shared_lms_child);
        }
        else {
            // cannot update anymore
            break;
        }
    }

    // set my parent as the new parent
    for (const auto& spanning_child : spanning_children_) {
        const auto child = spanning_child.lock();
        const auto parent = spanning_parent_.lock();
        child->graph_node_->change_spanning_parent(parent);
    }

    spanning_children_.clear();

    // 2. remove myself from my parent's children list

    spanning_parent_.lock()->graph_node_->erase_spanning_child(owner_keyfrm_.lock());
}

id_ordered_set<std::shared_ptr<keyframe>> graph_node::get_spanning_children() const {
    std::lock_guard<std::mutex> lock(mtx_);
    id_ordered_set<std::shared_ptr<keyframe>> locked_spanning_children;
    for (const auto& keyfrm : spanning_children_) {
        locked_spanning_children.insert(keyfrm.lock());
    }
    return locked_spanning_children;
}

bool graph_node::has_spanning_child(const std::shared_ptr<keyframe>& keyfrm) const {
    std::lock_guard<std::mutex> lock(mtx_);
    return static_cast<bool>(spanning_children_.count(keyfrm));
}

void graph_node::add_loop_edge(const std::shared_ptr<keyframe>& keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_);
    loop_edges_.insert(keyfrm);
    // cannot erase loop edges
    owner_keyfrm_.lock()->set_not_to_be_erased();
}

std::set<std::shared_ptr<keyframe>> graph_node::get_loop_edges() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::set<std::shared_ptr<keyframe>> locked_loop_edges;
    for (const auto& keyfrm : loop_edges_) {
        locked_loop_edges.insert(keyfrm.lock());
    }
    return locked_loop_edges;
}

bool graph_node::has_loop_edge() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return !loop_edges_.empty();
}

std::shared_ptr<keyframe> graph_node::get_spanning_root() {
    std::lock_guard<std::mutex> lock(mtx_);
    return get_spanning_root_impl();
}

std::shared_ptr<keyframe> graph_node::get_spanning_root_impl() {
    auto spanning_root = spanning_root_.lock();
    if (spanning_root) {
        return spanning_root;
    }
    else {
        auto spanning_parent = spanning_parent_.lock();
        if (spanning_parent) {
            spanning_root_ = spanning_parent->graph_node_->get_spanning_root_impl();
        }
        else {
            spanning_root_ = owner_keyfrm_.lock();
        }
        return spanning_root_.lock();
    }
}

bool graph_node::is_spanning_root() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return is_spanning_root_impl();
}

bool graph_node::is_spanning_root_impl() const {
    auto spanning_root = spanning_root_.lock();
    auto owner_keyfrm = owner_keyfrm_.lock();
    assert(owner_keyfrm);
    return spanning_root && spanning_root->id_ == owner_keyfrm->id_;
}

void graph_node::set_spanning_root(std::shared_ptr<keyframe>& keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_);
    spanning_root_ = keyfrm;
}

std::vector<std::shared_ptr<keyframe>> graph_node::get_keyframes_from_root() {
    std::vector<std::shared_ptr<keyframe>> keyfrms;
    std::list<std::shared_ptr<data::keyframe>> keyfrms_to_check;
    keyfrms_to_check.push_back(get_spanning_root());
    while (!keyfrms_to_check.empty()) {
        auto parent = keyfrms_to_check.front();
        keyfrms.push_back(parent);
        const auto children = parent->graph_node_->get_spanning_children();
        for (auto child : children) {
            keyfrms_to_check.push_back(child);
        }
        keyfrms_to_check.pop_front();
    }
    return keyfrms;
}

template<typename T, typename U>
std::vector<std::shared_ptr<keyframe>> graph_node::extract_intersection(const T& keyfrms_1, const U& keyfrms_2) {
    std::vector<std::shared_ptr<keyframe>> intersection;
    intersection.reserve(std::min(keyfrms_1.size(), keyfrms_2.size()));
    for (const auto& keyfrm_1 : keyfrms_1) {
        for (const auto& keyfrm_2 : keyfrms_2) {
            if (*keyfrm_1 == *keyfrm_2) {
                intersection.push_back(keyfrm_1);
            }
        }
    }
    return intersection;
}

} // namespace data
} // namespace cv::slam
