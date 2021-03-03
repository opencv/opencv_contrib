// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: The "adaskit Team" at Fixstars Corporation

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

Ptr<cuda::StereoSGM> cv::cuda::createStereoSGM(int, int, int, int, int, int) { throw_no_cuda(); return Ptr<cuda::StereoSGM>(); }

#else /* !defined (HAVE_CUDA) */

#include "cuda/stereosgm.hpp"

namespace
{
struct StereoSGMParams
{
    int minDisparity;
    int numDisparities;
    int P1;
    int P2;
    int uniquenessRatio;
    int mode;
    StereoSGMParams(int minDisparity = 0, int numDisparities = 128, int P1 = 10, int P2 = 120, int uniquenessRatio = 5, int mode = StereoSGM::MODE_HH4) : minDisparity(minDisparity), numDisparities(numDisparities), P1(P1), P2(P2), uniquenessRatio(uniquenessRatio), mode(mode) {}
};

class StereoSGMImpl CV_FINAL : public StereoSGM
{
public:
    StereoSGMImpl(int minDisparity, int numDisparities, int P1, int P2, int uniquenessRatio, int mode);

    void compute(InputArray left, InputArray right, OutputArray disparity) CV_OVERRIDE;
    void compute(InputArray left, InputArray right, OutputArray disparity, Stream& stream) CV_OVERRIDE;

    int getBlockSize() const CV_OVERRIDE { return -1; }
    void setBlockSize(int /*blockSize*/) CV_OVERRIDE {}

    int getDisp12MaxDiff() const CV_OVERRIDE { return 1; }
    void setDisp12MaxDiff(int /*disp12MaxDiff*/) CV_OVERRIDE {}

    int getMinDisparity() const CV_OVERRIDE { return params.minDisparity; }
    void setMinDisparity(int minDisparity) CV_OVERRIDE { params.minDisparity = minDisparity; }

    int getNumDisparities() const CV_OVERRIDE { return params.numDisparities; }
    void setNumDisparities(int numDisparities) CV_OVERRIDE { params.numDisparities = numDisparities; }

    int getSpeckleWindowSize() const CV_OVERRIDE { return 0; }
    void setSpeckleWindowSize(int /*speckleWindowSize*/) CV_OVERRIDE {}

    int getSpeckleRange() const CV_OVERRIDE { return 0; }
    void setSpeckleRange(int /*speckleRange*/) CV_OVERRIDE {}

    int getP1() const CV_OVERRIDE { return params.P1; }
    void setP1(int P1) CV_OVERRIDE { params.P1 = P1; }

    int getP2() const CV_OVERRIDE { return params.P2; }
    void setP2(int P2) CV_OVERRIDE { params.P2 = P2; }

    int getUniquenessRatio() const CV_OVERRIDE { return params.uniquenessRatio; }
    void setUniquenessRatio(int uniquenessRatio) CV_OVERRIDE { params.uniquenessRatio = uniquenessRatio; }

    int getMode() const CV_OVERRIDE { return params.mode; }
    void setMode(int mode) CV_OVERRIDE { params.mode = mode; }

    int getPreFilterCap() const CV_OVERRIDE { return -1; }
    void setPreFilterCap(int /*preFilterCap*/) CV_OVERRIDE {}

private:
    StereoSGMParams params;
    device::stereosgm::path_aggregation::PathAggregation pathAggregation;
    GpuMat censused_left, censused_right;
    GpuMat aggregated;
    GpuMat left_disp_tmp, right_disp_tmp;
    GpuMat right_disp;
};

StereoSGMImpl::StereoSGMImpl(int minDisparity, int numDisparities, int P1, int P2, int uniquenessRatio, int mode)
    : params(minDisparity, numDisparities, P1, P2, uniquenessRatio, mode)
{
}

void StereoSGMImpl::compute(InputArray left, InputArray right, OutputArray disparity)
{
    compute(left, right, disparity, Stream::Null());
}

void StereoSGMImpl::compute(InputArray _left, InputArray _right, OutputArray _disparity, Stream& _stream)
{
    using namespace device::stereosgm;

    GpuMat left = _left.getGpuMat();
    GpuMat right = _right.getGpuMat();
    const Size size = left.size();

    if (params.mode != MODE_HH && params.mode != MODE_HH4)
    {
        CV_Error(Error::StsBadArg, "Unsupported mode");
    }
    const unsigned int num_paths = params.mode == MODE_HH4 ? 4 : 8;

    CV_Assert(left.type() == CV_8UC1 || left.type() == CV_16UC1);
    CV_Assert(size == right.size() && left.type() == right.type());

    _disparity.create(size, CV_16SC1);
    ensureSizeIsEnough(size, CV_16SC1, right_disp);
    GpuMat left_disp = _disparity.getGpuMat();

    ensureSizeIsEnough(size, CV_32SC1, censused_left);
    ensureSizeIsEnough(size, CV_32SC1, censused_right);
    census_transform::censusTransform(left, censused_left, _stream);
    census_transform::censusTransform(right, censused_right, _stream);

    ensureSizeIsEnough(1, size.width * size.height * params.numDisparities * num_paths, CV_8UC1, aggregated);
    ensureSizeIsEnough(size, CV_16SC1, left_disp_tmp);
    ensureSizeIsEnough(size, CV_16SC1, right_disp_tmp);

    switch (params.numDisparities)
    {
    case 64:
        pathAggregation.operator()<64>(censused_left, censused_right, aggregated, params.mode, params.P1, params.P2, params.minDisparity, _stream);
        winner_takes_all::winnerTakesAll<64>(aggregated, left_disp_tmp, right_disp_tmp, (float)(100 - params.uniquenessRatio) / 100, true, params.mode, _stream);
        break;
    case 128:
        pathAggregation.operator()<128>(censused_left, censused_right, aggregated, params.mode, params.P1, params.P2, params.minDisparity, _stream);
        winner_takes_all::winnerTakesAll<128>(aggregated, left_disp_tmp, right_disp_tmp, (float)(100 - params.uniquenessRatio) / 100, true, params.mode, _stream);
        break;
    case 256:
        pathAggregation.operator()<256>(censused_left, censused_right, aggregated, params.mode, params.P1, params.P2, params.minDisparity, _stream);
        winner_takes_all::winnerTakesAll<256>(aggregated, left_disp_tmp, right_disp_tmp, (float)(100 - params.uniquenessRatio) / 100, true, params.mode, _stream);
        break;
    default:
        CV_Error(Error::StsBadArg, "Unsupported num of disparities");
    }

    median_filter::medianFilter(left_disp_tmp, left_disp, _stream);
    median_filter::medianFilter(right_disp_tmp, right_disp, _stream);
    check_consistency::checkConsistency(left_disp, right_disp, left, true, _stream);
    correct_disparity_range::correctDisparityRange(left_disp, true, params.minDisparity, _stream);
}
} // anonymous namespace

Ptr<cuda::StereoSGM> cv::cuda::createStereoSGM(int minDisparity, int numDisparities, int P1, int P2, int uniquenessRatio, int mode)
{
    return makePtr<StereoSGMImpl>(minDisparity, numDisparities, P1, P2, uniquenessRatio, mode);
}

#endif /* !defined (HAVE_CUDA) */
