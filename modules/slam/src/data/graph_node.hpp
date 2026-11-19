#ifndef SLAM_DATA_GRAPH_NODE_H
#define SLAM_DATA_GRAPH_NODE_H

#include <atomic>
#include <mutex>
#include <vector>
#include <map>
#include <set>
#include <memory>

namespace cv::slam {
namespace data {

class keyframe;

class graph_node {
public:
    /**
     * Constructor
     */
    explicit graph_node(std::shared_ptr<keyframe>& keyfrm);

    /**
     * Destructor
     */
    ~graph_node() = default;

    //-----------------------------------------
    // covisibility graph

    /**
     * Add connection between myself and specified keyframes with the number of shared landmarks
     */
    void add_connection(const std::shared_ptr<keyframe>& keyfrm, const unsigned int num_shared_lms);

    /**
     * Erase connection between myself and specified keyframes
     */
    void erase_connection(const std::shared_ptr<keyframe>& keyfrm);

    /**
     * Erase all connections
     */
    void erase_all_connections();

    /**
     * Update the connections and the covisibilities by referring landmark observations
     */
    void update_connections(unsigned int min_num_shared_lms);

    /**
     * Update the order of the covisibilities
     * (NOTE: the new keyframe won't inserted)
     */
    void update_covisibility_orders();

    /**
     * Get the connected keyframes
     */
    std::set<std::shared_ptr<keyframe>> get_connected_keyframes() const;

    /**
     * Get the covisibility keyframes
     */
    std::vector<std::shared_ptr<keyframe>> get_covisibilities() const;

    /**
     * Get the top-n covisibility keyframes
     */
    std::vector<std::shared_ptr<keyframe>> get_top_n_covisibilities(const unsigned int num_covisibilities) const;

    /**
     * Get the covisibility keyframes which have shared landmarks over the threshold
     */
    std::vector<std::shared_ptr<keyframe>> get_covisibilities_over_min_num_shared_lms(const unsigned int min_num_shared_lms) const;

    /**
     * Get the number of shared landmarks between this and specified keyframe
     */
    unsigned int get_num_shared_landmarks(const std::shared_ptr<keyframe>& keyfrm) const;

    //-----------------------------------------
    // spanning tree

    /**
     * Set the parent node of spanning tree
     * (NOTE: this functions will be only used for map loading or initialization)
     */
    void set_spanning_parent(const std::shared_ptr<keyframe>& keyfrm);

    /**
     * Get the parent of spanning tree
     */
    std::shared_ptr<keyframe> get_spanning_parent() const;

    /**
     * Change the parent node of spanning tree
     */
    void change_spanning_parent(const std::shared_ptr<keyframe>& keyfrm);

    /**
     * Add the child note of spanning tree
     */
    void add_spanning_child(const std::shared_ptr<keyframe>& keyfrm);

    /**
     * Erase the child node of spanning tree
     */
    void erase_spanning_child(const std::shared_ptr<keyframe>& keyfrm);

    /**
     * Recover the spanning connections of the connected keyframes
     */
    void recover_spanning_connections();

    /**
     * Get the children of spanning tree
     */
    id_ordered_set<std::shared_ptr<keyframe>> get_spanning_children() const;

    /**
     * Whether this node has the specified child or not
     */
    bool has_spanning_child(const std::shared_ptr<keyframe>& keyfrm) const;

    //-----------------------------------------
    // loop edge

    /**
     * Add the loop edge
     */
    void add_loop_edge(const std::shared_ptr<keyframe>& keyfrm);

    /**
     * Get the loop edges
     */
    std::set<std::shared_ptr<keyframe>> get_loop_edges() const;

    /**
     * Whether this node has any loop edges or not
     */
    bool has_loop_edge() const;

    //-----------------------------------------
    // root

    std::shared_ptr<keyframe> get_spanning_root();

    bool is_spanning_root() const;

    void set_spanning_root(std::shared_ptr<keyframe>& keyfrm);

    std::vector<std::shared_ptr<keyframe>> get_keyframes_from_root();

private:
    /**
     * Update the order of the covisibilities (without mutex)
     * (NOTE: the new keyframe won't inserted)
     */
    void update_covisibility_orders_impl();

    /**
     * Extract intersection from the two lists of keyframes
     */
    template<typename T, typename U>
    static std::vector<std::shared_ptr<keyframe>> extract_intersection(const T& keyfrms_1, const U& keyfrms_2);

    //-----------------------------------------
    // implementation

    std::shared_ptr<keyframe> get_spanning_root_impl();

    bool is_spanning_root_impl() const;

    //! keyframe of this node
    std::weak_ptr<keyframe> const owner_keyfrm_;

    //! all connected keyframes and the number of shared landmarks between the keyframes
    id_ordered_map<std::weak_ptr<keyframe>, unsigned int> connected_keyfrms_and_num_shared_lms_;

    //! covisibility keyframe in descending order of the number of shared landmarks
    std::vector<std::weak_ptr<keyframe>> ordered_covisibilities_;
    //! number of shared landmarks in descending order
    std::vector<unsigned int> ordered_num_shared_lms_;

    //! parent of spanning tree
    std::weak_ptr<keyframe> spanning_parent_;
    //! children of spanning tree
    id_ordered_set<std::weak_ptr<keyframe>> spanning_children_;
    //! root keyframe of spanning tree
    std::weak_ptr<keyframe> spanning_root_;

    //! loop edges
    id_ordered_set<std::weak_ptr<keyframe>> loop_edges_;

    //! need mutex for access to connections
    mutable std::mutex mtx_;
};

} // namespace data
} // namespace cv::slam

#endif // SLAM_DATA_GRAPH_NODE_H
