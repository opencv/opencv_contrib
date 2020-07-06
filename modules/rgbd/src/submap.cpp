// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "submap.hpp"

#include "hash_tsdf.hpp"
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

SubmapManager::SubmapManager(const VolumeParams& _volumeParams)
    : volumeParams(_volumeParams)
{}

SubmapId SubmapManager::createNewSubmap(bool isCurrentActiveMap, const Affine3f& pose)
{
    size_t idx = submaps.size();
    submaps.push_back(cv::makePtr<Submap>(idx, volumeParams, pose));
    /* Constraint newConstraint; */
    /* newConstraint.idx = idx; */
    /* newConstraint.type = isCurrentActiveMap ? CURRENT_ACTIVE : ACTIVE; */
    std::cout << "Created new submap\n";
    return idx;
}

cv::Ptr<Submap> SubmapManager::getCurrentSubmap(void)
{
    if(submaps.size() > 0)
        return submaps.at(submaps.size() - 1);
    else
        return nullptr;
}

void SubmapManager::reset()
{
    //! Technically should delete all the submaps;
    for(const auto& submap : submaps)
    {
        submap->volume->reset();
    }
    submaps.clear();
}

bool SubmapManager::shouldCreateSubmap(int currFrameId)
{
    cv::Ptr<Submap> curr_submap = getCurrentSubmap();
    int allocate_blocks = curr_submap->getTotalAllocatedBlocks();
    int visible_blocks = curr_submap->getVisibleBlocks(currFrameId);
    float ratio = float(visible_blocks) / float(allocate_blocks);
    std::cout << "Ratio: " << ratio << "\n";

    if(ratio < 0.2f)
        return true;
    return false;
}



}  // namespace rgbd
}  // namespace cv
