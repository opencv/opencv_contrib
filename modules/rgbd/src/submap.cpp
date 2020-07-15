// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "submap.hpp"

#include "hash_tsdf.hpp"
#include "opencv2/core/cvstd_wrapper.hpp"
#include "pose_graph.h"
#include "precomp.hpp"
namespace cv
{
namespace kinfu
{
Submap::Submap(SubmapId _submapId, const VolumeParams& volumeParams, const cv::Affine3f& _pose)
    : submapId(_submapId), pose(_pose)
{
    volume = cv::makePtr<HashTSDFVolumeCPU>(volumeParams);
    std::cout << "Created volume\n";
}

SubmapManager::SubmapManager(const VolumeParams& _volumeParams) : volumeParams(_volumeParams) {}

SubmapId SubmapManager::createNewSubmap(bool isCurrentActiveMap, const Affine3f& pose)
{
    SubmapId newSubmapId       = submaps.size();
    cv::Ptr<Submap> prevSubmap = getCurrentSubmap();

    cv::Ptr<Submap> newSubmap = cv::makePtr<Submap>(newSubmapId, volumeParams, pose);

    //! If first submap being added. Add prior constraint to PoseGraph
    if (submaps.size() == 0 || !prevSubmap)
    {
        poseGraph.addNode(newSubmapId, cv::makePtr<PoseGraphNode>(pose));
    }
    else
    {
        addCameraCameraConstraint(prevSubmap->getId(), newSubmap->getId(), prevSubmap->getPose(), newSubmap->getPose());
    }
    submaps.push_back(newSubmap);
    std::cout << "Created new submap\n";

    return idx;
}

void SubmapManager::addPriorConstraint(SubmapId currId, const Affine3f& currPose)
{
    poseGraph.addNode(currId, cv::makePtr<PoseGraphNode>(currPose));
    poseGraph.addEdge
}

void SubmapManager::addCameraCameraConstraint(SubmapId prevId, SubmapId currId, const Affine3f& prevPose,
                                              const Affine3f& currPose)
{
    //! 1. Add new posegraph node
    //! 2. Add new posegraph constraint

    //! TODO: Attempt registration between submaps
    Affine3f Tprev2curr = currPose.inv() * currPose;
    //! Constant information matrix for all odometry constraints
    Matx66f information = Matx66f::eye() * 1000;

    //! Inserts only if the element does not exist already
    poseGraph.addNode(prevId, cv::makePtr<PoseGraphNode>(prevPose));
    poseGraph.addNode(currId, cv::makePtr<PoseGraphNode>(currPose));

    poseGraph.addEdge(cv::makePtr<PoseGraphEdge>(prevId, currId, Tprev2curr, information));
}

cv::Ptr<Submap> SubmapManager::getCurrentSubmap(void)
{
    if (submaps.size() > 0)
        return submaps.at(submaps.size() - 1);
    else
        return nullptr;
}

void SubmapManager::reset()
{
    //! Technically should delete all the submaps;
    for (const auto& submap : submaps)
    {
        submap->volume->reset();
    }
    submaps.clear();
}

bool SubmapManager::shouldCreateSubmap(int currFrameId)
{
    cv::Ptr<Submap> curr_submap = getCurrentSubmap();
    int allocate_blocks         = curr_submap->getTotalAllocatedBlocks();
    int visible_blocks          = curr_submap->getVisibleBlocks(currFrameId);
    float ratio                 = float(visible_blocks) / float(allocate_blocks);
    std::cout << "Ratio: " << ratio << "\n";

    if (ratio < 0.2f)
        return true;
    return false;
}

}  // namespace kinfu
}  // namespace cv
