// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef __OPENCV_RGBD_SUBMAP_HPP__
#define __OPENCV_RGBD_SUBMAP_HPP__

#include <opencv2/core/cvdef.h>

#include <opencv2/core/affine.hpp>

#include "hash_tsdf.hpp"
#include "pose_graph.hpp"

namespace cv
{
namespace kinfu
{
template<typename MatType>
class Submap
{
   public:
    enum class Type
    {
        ACTIVE         = 0,
        CURRENT_ACTIVE = 1,
        LOOP_CLOSURE   = 2,
        LOST           = 3
    };

    Submap(int _id, Type _type, const VolumeParams& volumeParams, const cv::Affine3f& _pose = cv::Affine3f::Identity(), int _startFrameId = 0)
        : id(_id), type(_type), pose(_pose), volume(volumeParams), startFrameId(_startFrameId)
    {
        //! First camera pose is identity w.r.t submap pose
        cameraTraj.emplace_back(Matx44f::eye());
        std::cout << "Created volume\n";
    }

    virtual ~Submap() = default;

    virtual void updateCameraPose(const cv::Affine3f& cameraPose)
    {
        if (cameraTraj.size() == 0)
        {
            cameraTraj.push_back(cameraPose);
            return;
        }
        cv::Affine3f currPose = cameraTraj.back() * cameraPose;
        cameraTraj.push_back(currPose);
    }
    virtual void integrate(InputArray _depth, float depthFactor, const cv::kinfu::Intr& intrinsics, const int currframeId);
    virtual void raycast(const cv::Affine3f& cameraPose, const cv::kinfu::Intr& intrinsics, cv::Size frameSize,
                         OutputArray points, OutputArray normals);
    virtual void updatePyrPointsNormals(const int pyramidLevels);

    virtual size_t getTotalAllocatedBlocks() const { return volume.getTotalVolumeUnits(); };
    virtual size_t getVisibleBlocks(int currFrameId) const
    {
        return volume.getVisibleBlocks(currFrameId, FRAME_VISIBILITY_THRESHOLD);
    };

    //! TODO: Possibly useless
    virtual void setStartFrameId(int _startFrameId) { startFrameId = _startFrameId; };
    virtual void setStopFrameId(int _stopFrameId) { stopFrameId = _stopFrameId; };

    virtual Type getType() const { return type; }
    virtual int getId() const { return id; }
    virtual cv::Affine3f getPose() const { return pose; }
    virtual cv::Affine3f getCurrentCameraPose() const
    {
        return cameraTraj.size() > 0 ? cameraTraj.back() : cv::Affine3f::Identity();
    }

   public:
    //! TODO: Should we support submaps for regular volumes?
    static constexpr int FRAME_VISIBILITY_THRESHOLD = 5;
    //! TODO: Add support for GPU arrays (UMat)
    std::vector<MatType> pyrPoints;
    std::vector<MatType> pyrNormals;
    HashTSDFVolumeCPU volume;

   private:
    const int id;
    Type type;
    cv::Affine3f pose;
    //! Accumulates the camera to submap pose transformations
    std::vector<cv::Affine3f> cameraTraj;

    int startFrameId;
    int stopFrameId;
};

template<typename MatType>
void Submap<MatType>::integrate(InputArray _depth, float depthFactor, const cv::kinfu::Intr& intrinsics,
                                const int currFrameId)
{
    int index = currFrameId - startFrameId;
    std::cout << "Current frame ID: " <<  currFrameId << " startFrameId: " << startFrameId << "\n";
    std::cout << " Index: " << index << " Camera trajectory size: " << cameraTraj.size() << std::endl;
    CV_Assert(currFrameId >= startFrameId);
    CV_Assert(index >= 0 && index < int(cameraTraj.size()));

    const cv::Affine3f& currPose = cameraTraj.at(index);
    volume.integrate(_depth, depthFactor, currPose, intrinsics, currFrameId);
}

template<typename MatType>
void Submap<MatType>::raycast(const cv::Affine3f& cameraPose, const cv::kinfu::Intr& intrinsics, cv::Size frameSize,
                              OutputArray points, OutputArray normals)
{
    volume.raycast(cameraPose, intrinsics, frameSize, points, normals);
}

template<typename MatType>
void Submap<MatType>::updatePyrPointsNormals(const int pyramidLevels)
{
    MatType& points  = pyrPoints[0];
    MatType& normals = pyrNormals[0];

    buildPyramidPointsNormals(points, normals, pyrPoints, pyrNormals, pyramidLevels);
}

/**
 * @brief: Manages all the created submaps for a particular scene
 */
template<typename MatType>
class SubmapManager
{
   public:
    SubmapManager(const VolumeParams& _volumeParams)
        : volumeParams(_volumeParams)
    {}
    virtual ~SubmapManager() = default;

    void reset() { submapList.clear(); };

    bool shouldCreateSubmap(int frameId);
    //! Adds a new submap/volume into the current list of managed/Active submaps
    int createNewSubmap(bool isCurrentActiveMap, const int currFrameId = 0, const Affine3f& pose = cv::Affine3f::Identity());

    void removeSubmap(int _id);
    size_t getTotalSubmaps(void) const { return submapList.size(); };

    cv::Ptr<Submap<MatType>> getSubmap(int _id) const;
    cv::Ptr<Submap<MatType>> getCurrentSubmap(void) const;

    void setPose(int _id);

   protected:
    /* void addCameraCameraConstraint(int prevId, int currId, const Affine3f& prevPose, const Affine3f& currPose); */

    VolumeParams volumeParams;
    std::vector<cv::Ptr<Submap<MatType>>> submapList;

    PoseGraph poseGraph;
};

template<typename MatType>
int SubmapManager<MatType>::createNewSubmap(bool isCurrentActiveMap, int currFrameId, const Affine3f& pose)
{
    int newId = int(submapList.size());
    typename Submap<MatType>::Type type =
        isCurrentActiveMap ? Submap<MatType>::Type::CURRENT_ACTIVE : Submap<MatType>::Type::ACTIVE;
    cv::Ptr<Submap<MatType>> newSubmap = cv::makePtr<Submap<MatType>>(newId, type, volumeParams, pose, currFrameId);

    submapList.push_back(newSubmap);
    std::cout << "Created new submap\n";
    return newId;
}

template<typename MatType>
cv::Ptr<Submap<MatType>> SubmapManager<MatType>::getSubmap(int _id) const
{
    CV_Assert(_id >= 0 && _id < int(submapList.size()));
    if (submapList.size() > 0)
        return submapList.at(_id);
    return nullptr;
}

template<typename MatType>
cv::Ptr<Submap<MatType>> SubmapManager<MatType>::getCurrentSubmap(void) const
{
    if (submapList.size() > 0)
        return submapList.back();
    return nullptr;
}

template<typename MatType>
bool SubmapManager<MatType>::shouldCreateSubmap(int currFrameId)
{
    cv::Ptr<Submap<MatType>> curr_submap = getCurrentSubmap();
    int allocate_blocks                  = curr_submap->getTotalAllocatedBlocks();
    int visible_blocks                   = curr_submap->getVisibleBlocks(currFrameId);
    float ratio                          = float(visible_blocks) / float(allocate_blocks);
    std::cout << "Ratio: " << ratio << "\n";

    if (ratio < 0.2f)
        return true;
    return false;
}

}  // namespace kinfu
}  // namespace cv
#endif /* ifndef __OPENCV_RGBD_SUBMAP_HPP__ */
