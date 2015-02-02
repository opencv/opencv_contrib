#ifndef CVVISUAL_DEBUG_DMATCH_HPP
#define CVVISUAL_DEBUG_DMATCH_HPP

#include <string>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"

#include "call_meta_data.hpp"
#include "debug_mode.hpp"

#ifdef CV_DOXYGEN
#define CVVISUAL_DEBUGMODE
#endif

namespace cvv
{

//! @addtogroup cvv
//! @{

namespace impl
{
CV_EXPORTS void debugDMatch(cv::InputArray img1, std::vector<cv::KeyPoint> keypoints1,
                 cv::InputArray img2, std::vector<cv::KeyPoint> keypoints2,
                 std::vector<cv::DMatch> matches, const CallMetaData &data,
                 const char *description, const char *view,
                 bool useTrainDescriptor);
} // namespace impl

#ifdef CVVISUAL_DEBUGMODE
/** @brief Add a filled in DMatch \<dmatch\> to debug GUI.

The matches can are visualized for interactive inspection in different GUI views (one similar to an
interactive :draw_matches:drawMatches\<\>).

@param img1 First image used in DMatch \<dmatch\>.
@param keypoints1 Keypoints of first image.
@param img2 Second image used in DMatch.
@param keypoints2 Keypoints of second image.
@param matches
@param data See showImage
@param description See showImage
@param view See showImage
@param useTrainDescriptor Use DMatch \<dmatch\>'s train descriptor index instead of query
descriptor index.
 */
static inline void
debugDMatch(cv::InputArray img1, std::vector<cv::KeyPoint> keypoints1,
            cv::InputArray img2, std::vector<cv::KeyPoint> keypoints2,
            std::vector<cv::DMatch> matches, const impl::CallMetaData &data,
            const char *description = nullptr, const char *view = nullptr,
            bool useTrainDescriptor = true)
{
	if (debugMode())
	{
		impl::debugDMatch(img1, std::move(keypoints1), img2,
		                  std::move(keypoints2), std::move(matches),
		                  data, description, view, useTrainDescriptor);
	}
}
/** @overload */
static inline void
debugDMatch(cv::InputArray img1, std::vector<cv::KeyPoint> keypoints1,
            cv::InputArray img2, std::vector<cv::KeyPoint> keypoints2,
            std::vector<cv::DMatch> matches, const impl::CallMetaData &data,
            const std::string &description, const std::string &view,
            bool useTrainDescriptor = true)
{
	if (debugMode())
	{
		impl::debugDMatch(img1, std::move(keypoints1), img2,
		                  std::move(keypoints2), std::move(matches),
		                  data, description.c_str(), view.c_str(),
		                  useTrainDescriptor);
	}
}
#else
static inline void debugDMatch(cv::InputArray, std::vector<cv::KeyPoint>,
                               cv::InputArray, std::vector<cv::KeyPoint>,
                               std::vector<cv::DMatch>,
                               const impl::CallMetaData &,
                               const char * = nullptr, const char * = nullptr,
                               bool = true)
{
}
static inline void debugDMatch(cv::InputArray, std::vector<cv::KeyPoint>,
                               cv::InputArray, std::vector<cv::KeyPoint>,
                               std::vector<cv::DMatch>,
                               const impl::CallMetaData &, const std::string &,
                               const std::string &, bool = true)
{
}
#endif

//! @}

} // namespace cvv

#endif
