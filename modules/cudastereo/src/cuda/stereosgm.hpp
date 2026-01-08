// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: The "adaskit Team" at Fixstars Corporation

#ifndef OPENCV_CUDASTEREO_SGM_HPP
#define OPENCV_CUDASTEREO_SGM_HPP

#include "opencv2/core/cuda.hpp"

namespace cv { namespace cuda { namespace device {
namespace stereosgm
{

namespace census_transform
{
CV_EXPORTS void censusTransform(const GpuMat& src, GpuMat& dest, cv::cuda::Stream& stream);
}

namespace path_aggregation
{
class PathAggregation
{
private:
    static constexpr unsigned int MAX_NUM_PATHS = 8;

    std::array<Stream, MAX_NUM_PATHS> streams;
    std::array<Event, MAX_NUM_PATHS> events;
    std::array<GpuMat, MAX_NUM_PATHS> subs;
public:
    template <size_t MAX_DISPARITY>
    void operator() (const GpuMat& left, const GpuMat& right, GpuMat& dest, int mode, int p1, int p2, int min_disp, Stream& stream);
};
}

namespace winner_takes_all
{
template <size_t MAX_DISPARITY>
void winnerTakesAll(const GpuMat& src, GpuMat& left, GpuMat& right, float uniqueness, bool subpixel, int mode, cv::cuda::Stream& stream);
}

namespace median_filter
{
void medianFilter(const GpuMat& src, GpuMat& dst, Stream& stream);
}

namespace check_consistency
{
void checkConsistency(GpuMat& left_disp, const GpuMat& right_disp, const GpuMat& src_left, bool subpixel, Stream& stream);
}

namespace correct_disparity_range
{
void correctDisparityRange(GpuMat& disp, bool subpixel, int min_disp, Stream& stream);
}

} // namespace stereosgm
}}} // namespace cv { namespace cuda { namespace device {

#endif /* OPENCV_CUDASTEREO_SGM_HPP */
