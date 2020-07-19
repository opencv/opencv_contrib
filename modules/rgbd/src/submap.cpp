// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "submap.hpp"

#include "hash_tsdf.hpp"
#include "kinfu_frame.hpp"
#include "precomp.hpp"

namespace cv
{
namespace kinfu
{

/* void SubmapManager::addCameraCameraConstraint(int prevId, int currId, const Affine3f& prevPose, const Affine3f& currPose)
 */
/* { */
/*     //! 1. Add new posegraph node */
/*     //! 2. Add new posegraph constraint */

/*     //! TODO: Attempt registration between submaps */
/*     Affine3f Tprev2curr = currPose.inv() * currPose; */
/*     //! Constant information matrix for all odometry constraints */
/*     Matx66f information = Matx66f::eye() * 1000; */

/*     //! Inserts only if the element does not exist already */
/*     poseGraph.addNode(prevId, cv::makePtr<PoseGraphNode>(prevPose)); */
/*     poseGraph.addNode(currId, cv::makePtr<PoseGraphNode>(currPose)); */

/*     poseGraph.addEdge(cv::makePtr<PoseGraphEdge>(prevId, currId, Tprev2curr, information)); */
/* } */


}  // namespace kinfu
}  // namespace cv
