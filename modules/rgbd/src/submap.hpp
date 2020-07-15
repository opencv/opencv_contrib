// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef __OPENCV_RGBD_SUBMAP_H__
#define __OPENCV_RGBD_SUBMAP_H__

#include <opencv2/core/cvdef.h>

#include <opencv2/core/affine.hpp>
#include <unordered_map>

#include "hash_tsdf.hpp"
#include "pose_graph.h"

namespace cv
{
namespace kinfu
{
typedef uint16_t SubmapId;
typedef uint16_t FrameId;

//! T is either HashTSDFVolumeCPU or HashTSDFVolumeGPU
class Submap
{
   public:
    Submap(SubmapId _submapId, const VolumeParams& volumeParams, const cv::Affine3f& _pose = cv::Affine3f::Identity());
    virtual ~Submap() = default;

    virtual size_t getTotalAllocatedBlocks() const { return volume->getTotalVolumeUnits(); };
    virtual size_t getVisibleBlocks(int currFrameId) const { return volume->getVisibleBlocks(currFrameId, FRAME_VISIBILITY_THRESHOLD); };

    virtual void setStartFrameId(FrameId _startFrameId) { startFrameId = _startFrameId; };
    virtual void setStopFrameId(FrameId _stopFrameId) { stopFrameId = _stopFrameId; };

    virtual SubmapId getId() const { return submapId; }
    virtual cv::Affine3f getPose() const { return pose; }

   public:
    //! TODO: Should we support submaps for regular volumes?
    static constexpr int FRAME_VISIBILITY_THRESHOLD = 5;
    cv::Ptr<HashTSDFVolume<HashTSDFVolumeCPU>> volume;
   private:
    const SubmapId submapId;
    cv::Affine3f pose;

    FrameId startFrameId;
    FrameId stopFrameId;
};

struct Constraint
{
    enum Type
    {
        CURRENT_ACTIVE,
        ACTIVE,
    };
    SubmapId idx;
    Type type;
};

class SubmapManager
{
   public:
    SubmapManager(const VolumeParams& _volumeParams);
    virtual ~SubmapManager() = default;

    void reset();
    SubmapId createNewSubmap(bool isCurrentActiveMap, const Affine3f& pose = cv::Affine3f::Identity());
    void removeSubmap(SubmapId _submapId);
    size_t getTotalSubmaps(void) const { return submaps.size(); };
    Submap getSubmap(SubmapId _submapId) const;
    cv::Ptr<Submap> getCurrentSubmap(void);

    bool shouldCreateSubmap(int frameId);

    void setPose(SubmapId _submapId);

   protected:

    void addCameraCameraConstraint(SubmapId prevId, SubmapId currId, const Affine3f& prevPose, const Affine3f& currPose);
    VolumeParams volumeParams;
    std::vector<cv::Ptr<Submap>> submaps;
    std::vector<Constraint> constraints;

    PoseGraph poseGraph;
};

}  // namespace kinfu
}  // namespace cv
#endif /* ifndef __OPENCV_RGBD_SUBMAP_H__ */
