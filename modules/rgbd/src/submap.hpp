// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef __OPENCV_RGBD_SUBMAP_HPP__
#define __OPENCV_RGBD_SUBMAP_HPP__

#include <opencv2/core/cvdef.h>

#include <opencv2/core/affine.hpp>
#include <type_traits>
#include <vector>

#include "hash_tsdf.hpp"
#include "opencv2/core/mat.inl.hpp"
#include "pose_graph.hpp"

namespace cv
{
namespace kinfu
{
template<typename MatType>
class Submap
{
   public:
    struct PoseConstraint
    {
        Affine3f estimatedPose;
        int weight;

        PoseConstraint() : weight(0){};

        void accumulatePose(const Affine3f& _pose, int _weight = 1)
        {
            Matx44f accPose = estimatedPose.matrix * weight + _pose.matrix * _weight;
            weight         += _weight;
            accPose        /= float(weight);
            estimatedPose   = Affine3f(accPose);
        }
    };
    typedef std::map<int, PoseConstraint> Constraints;

    Submap(int _id, const VolumeParams& volumeParams, const cv::Affine3f& _pose = cv::Affine3f::Identity(),
           int _startFrameId = 0)
        : id(_id), pose(_pose), cameraPose(Affine3f::Identity()), startFrameId(_startFrameId), volume(volumeParams)
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

    float calcVisibilityRatio(int currFrameId) const
    {
        int allocate_blocks = getTotalAllocatedBlocks();
        int visible_blocks  = getVisibleBlocks(currFrameId);
        return float(visible_blocks) / float(allocate_blocks);
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
    cv::Affine3f pose;
    cv::Affine3f cameraPose;
    Constraints constraints;

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
    enum class Type
    {
        NEW            = 0,
        CURRENT        = 1,
        RELOCALISATION = 2,
        LOOP_CLOSURE   = 3,
        LOST           = 4
    };

    struct ActiveSubmapData
    {
        Type type;
        std::vector<Affine3f> constraints;
        int trackingAttempts;
    };
    typedef Submap<MatType> SubmapT;
    typedef std::map<int, Ptr<SubmapT>> IdToSubmapPtr;
    typedef std::unordered_map<int, ActiveSubmapData> IdToActiveSubmaps;

    SubmapManager(const VolumeParams& _volumeParams) : volumeParams(_volumeParams) {}
    virtual ~SubmapManager() = default;

    void reset() { submapList.clear(); };

    bool shouldCreateSubmap(int frameId);
    bool shouldChangeCurrSubmap(int _frameId, int toSubmapId);

    //! Adds a new submap/volume into the current list of managed/Active submaps
    int createNewSubmap(bool isCurrentActiveMap, const int currFrameId = 0, const Affine3f& pose = cv::Affine3f::Identity());

    void removeSubmap(int _id);
    size_t numOfSubmaps(void) const { return submapList.size(); };
    size_t numOfActiveSubmaps(void) const { return activeSubmaps.size(); };

    Ptr<SubmapT> getSubmap(int _id) const;
    Ptr<SubmapT> getCurrentSubmap(void) const;

    bool estimateConstraint(int fromSubmapId, int toSubmapId, int& inliers, Affine3f& inlierPose);
    void updateMap(int _frameId, std::vector<MatType> _framePoints, std::vector<MatType> _frameNormals);

    VolumeParams volumeParams;

    std::vector<Ptr<SubmapT>> submapList;
    IdToActiveSubmaps activeSubmaps;

    PoseGraph poseGraph;
};

template<typename MatType>
int SubmapManager<MatType>::createNewSubmap(bool isCurrentMap, int currFrameId, const Affine3f& pose)
{
    int newId = int(submapList.size());

    Ptr<SubmapT> newSubmap = cv::makePtr<SubmapT>(newId, volumeParams, pose, currFrameId);
    submapList.push_back(newSubmap);

    ActiveSubmapData newSubmapData;
    newSubmapData.trackingAttempts = 0;
    newSubmapData.type             = isCurrentMap ? Type::CURRENT : Type::NEW;
    activeSubmaps[newId]           = newSubmapData;

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
Ptr<Submap<MatType>> SubmapManager<MatType>::getCurrentSubmap(void) const
{
    for (const auto& it : activeSubmaps)
    {
        if (it.second.type == Type::CURRENT)
            return getSubmap(it.first);
    }
    return nullptr;
}

template<typename MatType>
bool SubmapManager<MatType>::shouldCreateSubmap(int currFrameId)
{
    int currSubmapId = -1;
    for (const auto& it : activeSubmaps)
    {
        auto submapData = it.second;
        // No more than 1 new submap at a time!
        if (submapData.type == Type::NEW)
        {
            return false;
        }
        if (submapData.type == Type::CURRENT)
        {
            currSubmapId = it.first;
        }
    }
    //! TODO: This shouldn't be happening? since there should always be one active current submap
    if (currSubmapId < 0)
    {
        return false;
    }

    Ptr<SubmapT> currSubmap = getSubmap(currSubmapId);
    float ratio             = currSubmap->calcVisibilityRatio(currFrameId);

    std::cout << "Ratio: " << ratio << "\n";

    if (ratio < 0.2f)
        return true;
    return false;
}

template<typename MatType>
bool SubmapManager<MatType>::estimateConstraint(int fromSubmapId, int toSubmapId, int& inliers, Affine3f& inlierPose)
{
    static constexpr int MAX_ITER                    = 10;
    static constexpr float CONVERGE_WEIGHT_THRESHOLD = 0.01f;
    static constexpr float INLIER_WEIGHT_THRESH      = 0.8f;
    static constexpr int MIN_INLIERS                 = 10;

    //! thresh = HUBER_THRESH
    auto huberWeight = [](float residual, float thresh = 0.05f) -> float {
        float rAbs = abs(residual);
        if (rAbs < thresh)
            return 1.0;
        float numerator = sqrt(2 * thresh * rAbs - thresh * thresh);
        return numerator / rAbs;
    };

    Ptr<SubmapT> fromSubmap          = getSubmap(fromSubmapId);
    Ptr<SubmapT> toSubmap            = getSubmap(toSubmapId);
    ActiveSubmapData& fromSubmapData = activeSubmaps.at(fromSubmapId);

    Affine3f TcameraToFromSubmap = fromSubmap->cameraPose;
    Affine3f TcameraToToSubmap   = toSubmap->cameraPose;

    // FromSubmap -> ToSubmap transform
    Affine3f candidateConstraint = TcameraToToSubmap * TcameraToFromSubmap.inv();
    fromSubmapData.trackingAttempts++;
    fromSubmapData.constraints.push_back(candidateConstraint);

    std::cout << "Candidate constraint from: " << fromSubmap->id << " to " << toSubmap->id << "\n"
              << candidateConstraint.matrix << "\n";
    std::cout << "Constraint observations size: " << fromSubmapData.constraints.size() << "\n";

    std::vector<float> weights(fromSubmapData.constraints.size() + 1, 1.0f);

    Affine3f prevConstraint = fromSubmap->getConstraint(toSubmap->id).estimatedPose;
    int prevWeight          = fromSubmap->getConstraint(toSubmap->id).weight;

    std::cout << "Previous constraint pose: \n" << prevConstraint.matrix << "\n previous Weight: " << prevWeight << "\n";

    // Iterative reweighted least squares with huber threshold to find the inliers in the past observations
    Vec6f meanConstraint;
    float sumWeight = 0.0f;
    for (int i = 0; i < MAX_ITER; i++)
    {
        Vec6f constraintVec;
        for (int j = 0; j < int(weights.size() - 1); j++)
        {
            Affine3f currObservation = fromSubmapData.constraints[j];
            cv::vconcat(currObservation.rvec(), currObservation.translation(), constraintVec);
            meanConstraint += weights[j] * constraintVec;
            sumWeight += weights[j];
        }
        // Heavier weight given to the estimatedPose
        cv::vconcat(prevConstraint.rvec(), prevConstraint.translation(), constraintVec);
        meanConstraint += weights.back() * prevWeight * constraintVec;
        sumWeight += prevWeight;
        /* std::cout << "meanConstraint before average: " << meanConstraint << " sumWeight: " << sumWeight << "\n"; */
        meanConstraint /= float(sumWeight);

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
                Affine3f currObservation = fromSubmapData.constraints[j];
                cv::vconcat(currObservation.rvec(), currObservation.translation(), constraintVec);
                w = 1.0f;
            }

            std::cout << "meanConstraint: " << meanConstraint << " ConstraintVec: " << constraintVec << "\n";
            cv::Vec6f residualVec = (constraintVec - meanConstraint);
            std::cout << "Residual Vec: " << residualVec << "\n";
            residual         = norm(residualVec);
            double newWeight = huberWeight(residual);
            std::cout << "iteration: " << i << " residual: " << residual << " " << j
                      << "th weight before update: " << weights[j] << " after update: " << newWeight << "\n";
            diff += w * abs(newWeight - weights[j]);
            weights[j] = newWeight;
        }

        if (diff / (prevWeight + weights.size() - 1) < CONVERGE_WEIGHT_THRESHOLD)
            break;
    }

    int localInliers = 0;
    Matx44f inlierConstraint;
    for (int i = 0; i < int(weights.size()); i++)
    {
        if (weights[i] > INLIER_WEIGHT_THRESH)
        {
            localInliers++;
            if (i == int(weights.size() - 1))
                inlierConstraint += prevConstraint.matrix;
            else
                inlierConstraint += fromSubmapData.constraints[i].matrix;
        }
    }
    inlierConstraint /= float(max(localInliers, 1));
    inlierPose = Affine3f(inlierConstraint);
    inliers    = localInliers;

    /* std::cout << "Rvec: " << rvec << " tvec: " << tvec << "\n"; */
    std::cout << inlierPose.matrix << "\n";
    std::cout << " inliers: " << inliers << "\n";

    if (inliers >= MIN_INLIERS)
    {
        return true;
    }
    return false;
}

template<typename MatType>
bool SubmapManager<MatType>::shouldChangeCurrSubmap(int _frameId, int toSubmapId)
{
    auto toSubmap         = getSubmap(toSubmapId);
    auto toSubmapData     = activeSubmaps.at(toSubmapId);
    auto currActiveSubmap = getCurrentSubmap();

    int blocksInNewMap = toSubmap->getTotalAllocatedBlocks();
    float newRatio     = toSubmap->calcVisibilityRatio(_frameId);

    float currRatio = currActiveSubmap->calcVisibilityRatio(_frameId);

    //! TODO: Check for a specific threshold?
    if (blocksInNewMap <= 0)
        return false;
    if ((newRatio > currRatio) && (toSubmapData.type == Type::NEW))
        return true;

    return false;
}

template<typename MatType>
void SubmapManager<MatType>::updateMap(int _frameId, std::vector<MatType> _framePoints, std::vector<MatType> _frameNormals)
{
    const int currSubmapId  = getCurrentSubmap()->id;
    int changedCurrentMapId = -1;
    for (auto& it : activeSubmaps)
    {
        int submapId     = it.first;
        auto& submapData = it.second;
        if (submapData.type == Type::NEW || submapData.type == Type::LOOP_CLOSURE)
        {
            // Check with previous estimate
            int inliers;
            Affine3f inlierPose;
            bool success = estimateConstraint(submapId, currSubmapId, inliers, inlierPose);
            std::cout << "SubmapId: " << submapId << " Tracking attempts: " << submapData.trackingAttempts << "\n";
            if (success)
            {
                typename SubmapT::PoseConstraint& submapConstraint = getSubmap(submapId)->getConstraint(currSubmapId);
                std::cout << "Updated fixed constraint with inlierPose: \n" << inlierPose.matrix << "\n";
                submapConstraint.accumulatePose(inlierPose, inliers);
                std::cout << "Submap constraint estimated pose: \n" << submapConstraint.estimatedPose.matrix << "\n";
                submapData.constraints.clear();
                submapData.trackingAttempts = 0;

                //! TODO: Check for visibility and change currentActiveMap
                if (shouldChangeCurrSubmap(_frameId, submapId))
                {
                    //! TODO: Change submap and update constraints accordingly
                    std::cout << "Should change current map to the new map\n";
                    changedCurrentMapId = submapId;
                }
            }
            else
            {
                //! If tried tracking for threshold number of times, mark the data as lost
            }
        }
    }

    std::vector<int> createNewConstraintsList;
    for (auto& it : activeSubmaps)
    {
        int submapId     = it.first;
        auto& submapData = it.second;

        if (submapId == changedCurrentMapId)
        {
            submapData.type = Type::CURRENT;
        }
        if ((submapData.type == Type::CURRENT) && (changedCurrentMapId >= 0) && (submapId != changedCurrentMapId))
        {
            submapData.type = Type::LOST;
            createNewConstraintsList.push_back(submapId);
        }
        if ((submapData.type == Type::NEW || submapData.type == Type::LOOP_CLOSURE) && (changedCurrentMapId >= 0))
        {
            //! TODO: Add a new type called NEW_LOST?
            submapData.type = Type::LOST;
            createNewConstraintsList.push_back(submapId);
        }
    }

    for (typename IdToActiveSubmaps::iterator it = activeSubmaps.begin(); it != activeSubmaps.end();)
    {
        auto& submapData = it->second;
        if (submapData.type == Type::LOST)
            it = activeSubmaps.erase(it);
        else
            it++;
    }

    for (std::vector<int>::const_iterator it = createNewConstraintsList.begin(); it != createNewConstraintsList.end(); ++it)
    {
        int dataId = *it;
        ActiveSubmapData newSubmapData;
        newSubmapData.trackingAttempts = 0;
        newSubmapData.type             = Type::LOOP_CLOSURE;
        activeSubmaps[dataId]          = newSubmapData;
    }

    if (shouldCreateSubmap(_frameId))
    {
        Ptr<SubmapT> currActiveSubmap = getCurrentSubmap();
        Affine3f newSubmapPose        = currActiveSubmap->pose * currActiveSubmap->cameraPose;
        int submapId                  = createNewSubmap(false, _frameId, newSubmapPose);
        auto newSubmap                = getSubmap(submapId);
        newSubmap->pyrPoints          = _framePoints;
        newSubmap->pyrNormals         = _frameNormals;
    }
}

}  // namespace kinfu
}  // namespace cv
#endif /* ifndef __OPENCV_RGBD_SUBMAP_HPP__ */
