// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef __OPENCV_RGBD_SUBMAP_HPP__
#define __OPENCV_RGBD_SUBMAP_HPP__

#include <opencv2/core/cvdef.h>

#include <opencv2/core/affine.hpp>
#include <unordered_map>

#include "hash_tsdf.hpp"
#include "pose_graph.hpp"

namespace cv
{
namespace kinfu
{
class Submap
{
   public:
    Submap(int _submapId, const VolumeParams& volumeParams, const cv::Affine3f& _pose = cv::Affine3f::Identity())
        : submapId(_submapId), pose(_pose), volume(volumeParams)
    {
        std::cout << "Created volume\n";
    }

    virtual ~Submap() = default;

    virtual size_t getTotalAllocatedBlocks() const { return volume.getTotalVolumeUnits(); };
    virtual size_t getVisibleBlocks(int currFrameId) const
    {
        return volume.getVisibleBlocks(currFrameId, FRAME_VISIBILITY_THRESHOLD);
    };

    //! TODO: Possibly useless
    virtual void setStartFrameId(int _startFrameId) { startFrameId = _startFrameId; };
    virtual void setStopFrameId(int _stopFrameId) { stopFrameId = _stopFrameId; };

    virtual int getId() const { return submapId; }
    virtual cv::Affine3f getPose() const { return pose; }

   public:
    //! TODO: Should we support submaps for regular volumes?
    static constexpr int FRAME_VISIBILITY_THRESHOLD = 5;
    HashTSDFVolumeCPU volume;

   private:
    const int submapId;
    cv::Affine3f pose;

    int startFrameId;
    int stopFrameId;
};

struct Constraint
{
    enum Type
    {
        CURRENT_ACTIVE,
        ACTIVE,
    };
    int idx;
    Type type;
};

/**
 * @brief: Manages all the created submaps for a particular scene
 */
class SubmapManager
{
   public:
    SubmapManager(const VolumeParams& _volumeParams);
    virtual ~SubmapManager() = default;

    void reset() { submapList.clear(); };

    //! Adds a new submap/volume into the current list of managed/Active submaps
    int createNewSubmap(bool isCurrentActiveMap, const Affine3f& pose = cv::Affine3f::Identity());
    void removeSubmap(int _submapId);
    size_t getTotalSubmaps(void) const { return submapList.size(); };

    Submap getSubmap(int _submapId) const;
    Submap getCurrentSubmap(void) const;

    bool shouldCreateSubmap(int frameId);

    void setPose(int _submapId);

   protected:
    void addCameraCameraConstraint(int prevId, int currId, const Affine3f& prevPose, const Affine3f& currPose);

    VolumeParams volumeParams;
    std::vector<cv::Ptr<Submap>> submapList;
    std::vector<Constraint> constraints;

    PoseGraph poseGraph;
};

}  // namespace kinfu
}  // namespace cv
#endif /* ifndef __OPENCV_RGBD_SUBMAP_HPP__ */
