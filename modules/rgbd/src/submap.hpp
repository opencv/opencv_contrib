// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef __OPENCV_RGBD_SUBMAP_HPP__
#define __OPENCV_RGBD_SUBMAP_HPP__

#include <opencv2/core/cvdef.h>

#include <opencv2/core/affine.hpp>
#include <vector>

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
        NEW            = 0,
        CURRENT        = 1,
        RELOCALISATION = 2,
        LOOP_CLOSURE   = 3,
        LOST           = 4
    };
    struct PoseConstraint
    {
        Affine3f estimatedPose;
        std::vector<Affine3f> observations;
        int weight;

        PoseConstraint() : weight(0){};

        void accumulatePose(const Affine3f& _pose, int _weight = 1)
        {
            Matx44f accPose = estimatedPose.matrix * weight + _pose.matrix * _weight;
            weight += _weight;
            estimatedPose = Affine3f(accPose * 1 / float(weight));
        }
        void addObservation(const Affine3f& _pose) { observations.push_back(_pose); }
    };

    typedef std::map<int, PoseConstraint> Constraints;

    Submap(int _id, Type _type, const VolumeParams& volumeParams, const cv::Affine3f& _pose = cv::Affine3f::Identity(),
           int _startFrameId = 0)
        : id(_id),
          type(_type),
          pose(_pose),
          cameraPose(Affine3f::Identity()),
          startFrameId(_startFrameId),
          volume(volumeParams)
    {
        std::cout << "Created volume\n";
    }

    virtual ~Submap() = default;

    virtual void integrate(InputArray _depth, float depthFactor, const cv::kinfu::Intr& intrinsics, const int currframeId);
    virtual void raycast(const cv::Affine3f& cameraPose, const cv::kinfu::Intr& intrinsics, cv::Size frameSize,
                         OutputArray points, OutputArray normals);
    virtual void updatePyrPointsNormals(const int pyramidLevels);

    virtual size_t getTotalAllocatedBlocks() const { return volume.getTotalVolumeUnits(); };
    virtual size_t getVisibleBlocks(int currFrameId) const
    {
        return volume.getVisibleBlocks(currFrameId, FRAME_VISIBILITY_THRESHOLD);
    }

    //! TODO: Possibly useless
    virtual void setStartFrameId(int _startFrameId) { startFrameId = _startFrameId; };
    virtual void setStopFrameId(int _stopFrameId) { stopFrameId = _stopFrameId; };

    void composeCameraPose(const cv::Affine3f& _relativePose) { cameraPose = cameraPose * _relativePose; }
    PoseConstraint& getConstraint(const int _id)
    {
        //! Creates constraints if doesn't exist yet
        return constraints[_id];
    }

   public:
    const int id;
    Type type;
    cv::Affine3f pose;
    cv::Affine3f cameraPose;
    Constraints constraints;
    int trackingAttempts = 0;

    int startFrameId;
    int stopFrameId;
    //! TODO: Should we support submaps for regular volumes?
    static constexpr int FRAME_VISIBILITY_THRESHOLD = 5;

    //! TODO: Add support for GPU arrays (UMat)
    std::vector<MatType> pyrPoints;
    std::vector<MatType> pyrNormals;
    HashTSDFVolumeCPU volume;
};

template<typename MatType>
void Submap<MatType>::integrate(InputArray _depth, float depthFactor, const cv::kinfu::Intr& intrinsics,
                                const int currFrameId)
{
    CV_Assert(currFrameId >= startFrameId);
    volume.integrate(_depth, depthFactor, cameraPose, intrinsics, currFrameId);
}

template<typename MatType>
void Submap<MatType>::raycast(const cv::Affine3f& _cameraPose, const cv::kinfu::Intr& intrinsics, cv::Size frameSize,
                              OutputArray points, OutputArray normals)
{
    volume.raycast(_cameraPose, intrinsics, frameSize, points, normals);
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
    typedef Submap<MatType> SubmapT;
    typedef std::map<int, Ptr<SubmapT>> IdToSubmapPtr;

    SubmapManager(const VolumeParams& _volumeParams) : volumeParams(_volumeParams) {}
    virtual ~SubmapManager() = default;

    void reset() { submapList.clear(); };

    bool shouldCreateSubmap(int frameId);

    //! Adds a new submap/volume into the current list of managed/Active submaps
    int createNewSubmap(bool isCurrentActiveMap, const int currFrameId = 0, const Affine3f& pose = cv::Affine3f::Identity());

    void removeSubmap(int _id);
    size_t numOfSubmaps(void) const { return submapList.size(); };
    size_t numOfActiveSubmaps(void) const { return activeSubmapList.size(); };

    Ptr<SubmapT> getSubmap(int _id) const;
    Ptr<SubmapT> getActiveSubmap(int _id) const;
    Ptr<SubmapT> getCurrentSubmap(void) const;

    bool updateConstraint(Ptr<SubmapT> submap1, Ptr<SubmapT> submap2);
    void updateMap(int _frameId, std::vector<MatType> _framePoints, std::vector<MatType> _frameNormals);

    VolumeParams volumeParams;
    std::vector<Ptr<SubmapT>> submapList;
    //! Maintains a pointer to active submaps in the entire submapList
    IdToSubmapPtr activeSubmapList;

    PoseGraph poseGraph;
};

template<typename MatType>
int SubmapManager<MatType>::createNewSubmap(bool isCurrentMap, int currFrameId, const Affine3f& pose)
{
    int newId = int(submapList.size());

    typename SubmapT::Type type = isCurrentMap ? SubmapT::Type::CURRENT : SubmapT::Type::NEW;
    Ptr<SubmapT> newSubmap      = cv::makePtr<SubmapT>(newId, type, volumeParams, pose, currFrameId);

    submapList.push_back(newSubmap);
    activeSubmapList[newId] = newSubmap;

    std::cout << "Created new submap\n";

    return newId;
}

template<typename MatType>
Ptr<Submap<MatType>> SubmapManager<MatType>::getSubmap(int _id) const
{
    CV_Assert(submapList.size() > 0);
    CV_Assert(_id >= 0 && _id < int(submapList.size()));
    return submapList.at(_id);
}

template<typename MatType>
Ptr<Submap<MatType>> SubmapManager<MatType>::getActiveSubmap(int _id) const
{
    CV_Assert(submapList.size() > 0);
    CV_Assert(_id >= 0);
    return activeSubmapList.at(_id);
}

template<typename MatType>
Ptr<Submap<MatType>> SubmapManager<MatType>::getCurrentSubmap(void) const
{
    for (const auto& pSubmapPair : activeSubmapList)
    {
        if (pSubmapPair.second->type == SubmapT::Type::CURRENT)
            return pSubmapPair.second;
    }
    return nullptr;
}

template<typename MatType>
bool SubmapManager<MatType>::shouldCreateSubmap(int currFrameId)
{
    Ptr<SubmapT> currSubmap = nullptr;
    for(const auto& pActiveSubmapPair : activeSubmapList)
    {
        auto submap = pActiveSubmapPair.second;
        //! If there are already new submaps created, don't create more
        if(submap->type == SubmapT::Type::NEW)
        {
            return false;
        }
        if(submap->type == SubmapT::Type::CURRENT)
        {
            currSubmap = submap;
        }
    }
    //!TODO: This shouldn't be happening? since there should always be one active current submap
    if(!currSubmap)
    {
        return false;
    }
    int allocate_blocks     = currSubmap->getTotalAllocatedBlocks();
    int visible_blocks      = currSubmap->getVisibleBlocks(currFrameId);
    float ratio             = float(visible_blocks) / float(allocate_blocks);

    std::cout << "Ratio: " << ratio << "\n";

    if (ratio < 0.2f)
        return true;
    return false;
}

template<typename MatType>
bool SubmapManager<MatType>::updateConstraint(Ptr<SubmapT> submap, Ptr<SubmapT> currActiveSubmap)
{
    static constexpr int MAX_ITER                    = 10;
    static constexpr float CONVERGE_WEIGHT_THRESHOLD = 0.01f;
    static constexpr float INLIER_WEIGHT_THRESH      = 0.75f;
    static constexpr int MIN_INLIERS                 = 10;

    //! thresh = HUBER_THRESH
    auto huberWeight = [](float residual, float thresh = 0.1f) -> float {
        float rAbs = abs(residual);
        if (rAbs < thresh)
            return 1.0;
        float numerator = sqrt(2 * thresh * rAbs - thresh * thresh);
        return numerator / rAbs;
    };

    Affine3f TcameraToActiveSubmap = currActiveSubmap->cameraPose;
    Affine3f TcameraToSubmap       = submap->cameraPose;

    // ActiveSubmap -> submap transform
    Affine3f candidateConstraint                     = TcameraToSubmap * TcameraToActiveSubmap.inv();
    std::cout << "Candidate constraint: " << candidateConstraint.matrix << "\n";
    typename SubmapT::PoseConstraint& currConstraint = currActiveSubmap->getConstraint(submap->id);
    currConstraint.addObservation(candidateConstraint);
    std::vector<float> weights(currConstraint.observations.size() + 1, 1.0f);

    Affine3f prevConstraint = currActiveSubmap->getConstraint(submap->id).estimatedPose;
    int prevWeight          = currActiveSubmap->getConstraint(submap->id).weight;

    std::cout << "Previous constraint pose: " << prevConstraint.matrix << "\n previous Weight: " << prevWeight << "\n";

    Vec6f meanConstraint;
    float sumWeight = 0.0f;
    for (int i = 0; i < MAX_ITER; i++)
    {
        Vec6f constraintVec;
        for (int j = 0; j < int(weights.size() - 1); j++)
        {
            Affine3f currObservation = currConstraint.observations[j];
            cv::vconcat(currObservation.rvec(), currObservation.translation(), constraintVec);
            meanConstraint += weights[j] * constraintVec;
            sumWeight += weights[j];
        }
        // Heavier weight given to the estimatedPose
        cv::vconcat(prevConstraint.rvec(), prevConstraint.translation(), constraintVec);
        meanConstraint += weights.back() * prevWeight * constraintVec;
        sumWeight += prevWeight;
        meanConstraint = meanConstraint * (1 / sumWeight);

        float residual = 0.0f;
        float diff     = 0;
        for (int j = 0; j < int(weights.size()); j++)
        {
            float w;
            if (j == int(weights.size() - 1))
            {
                cv::vconcat(prevConstraint.rvec(), prevConstraint.translation(), constraintVec);
                w = prevWeight;
            }
            else
            {
                Affine3f currObservation = currConstraint.observations[j];
                cv::vconcat(currObservation.rvec(), currObservation.translation(), constraintVec);
                w = 1.0f;
            }

            residual         = norm(constraintVec - meanConstraint);
            double newWeight = huberWeight(residual);
            std::cout << "iteration: " << i << " residual: "<< residual << " " << j << "th weight before update: " << weights[j] << " after update: " << newWeight << "\n";
            diff += w * abs(newWeight - weights[j]);
            weights[j] = newWeight;
        }

        if (diff / (prevWeight + weights.size() - 1) < CONVERGE_WEIGHT_THRESHOLD)
            break;
    }

    int inliers = 0;
    for (int i = 0; i < int(weights.size() - 1); i++)
    {
        if (weights[i] > INLIER_WEIGHT_THRESH)
            inliers++;
    }
    std::cout << " inliers: " << inliers << "\n";
    if (inliers >= MIN_INLIERS)
    {
        currConstraint.accumulatePose(candidateConstraint);
        Affine3f updatedPose = currConstraint.estimatedPose;
        std::cout << "Current updated constraint pose : " << updatedPose.matrix << "\n";
        //! TODO: Should clear observations? not sure
        currConstraint.observations.clear();
        return true;
    }
    return false;
}
template<typename MatType>
void SubmapManager<MatType>::updateMap(int _frameId, std::vector<MatType> _framePoints, std::vector<MatType> _frameNormals)
{
    Ptr<SubmapT> currActiveSubmap = getCurrentSubmap();
    for (const auto& pSubmapPair : activeSubmapList)
    {
        auto submap = pSubmapPair.second;

        if (submap->type == SubmapT::Type::NEW)
        {
            // Check with previous estimate
            bool success = updateConstraint(submap, currActiveSubmap);
            if (success)
            {
                //! TODO: Check for visibility and change currentActiveMap
            }
        }
    }

    if (shouldCreateSubmap(_frameId))
    {
        Affine3f newSubmapPose = currActiveSubmap->pose * currActiveSubmap->cameraPose;
        int submapId = createNewSubmap(false, _frameId, newSubmapPose);
        auto newSubmap = getSubmap(submapId);
        newSubmap->pyrPoints = _framePoints;
        newSubmap->pyrNormals = _frameNormals;

    }
}

}  // namespace kinfu
}  // namespace cv
#endif /* ifndef __OPENCV_RGBD_SUBMAP_HPP__ */
