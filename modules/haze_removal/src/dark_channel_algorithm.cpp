// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc/edge_filter.hpp"
#include <vector>

using namespace cv;
using namespace cv::haze_removal;

namespace {

int intensityOfNthBrightestPixel(cv::InputArray _src, int n)
{
    const Mat src = _src.getMat();

    CV_Assert(src.channels() == 1);
    CV_Assert(src.type() == CV_8UC1);

    // simple histogram
    std::vector<int> histogram(256, 0);
    for (int i = 0; i < int(src.rows * src.cols); ++i)
        histogram[src.at<uchar>(i)]++;

    // find max threshold value (pixels from [0-max_threshold] will be removed)
    int max_threshold = (int)histogram.size() - 1;
    for (; max_threshold >= 0 && n > 0; --max_threshold)
    {
        n -= histogram[max_threshold];
    }

    max_threshold = max(max_threshold, 0);
    return max_threshold;
}

class HazeRemovalHeImpl CV_FINAL : public HazeRemovalBase::HazeRemovalImpl
{
private:
    int erosionSize; //Size of kernel to use for computing Dark Channeal
    int erosionType; //Type of kernel to use for computing Dark Channel

    float percentageBrightestPixelsForAtmoLight; //See the papers "Estimating the Atmospheric Light" section for details
    float omega;                                 // controls how much fog to remove :range[0,1]: 1 means remove all fog, 0 means no fog
    float guidedFilterRadius;
    float guidedFilterEps;
    float transmissionLowerBound; //each element of transmission matrix must be atleast this

    Mat kernelForEroding; // The kernel used for eroding

public:
    HazeRemovalHeImpl()
    {
        erosionSize = 15;
        erosionType = MORPH_RECT;
        percentageBrightestPixelsForAtmoLight = 0.001;
        omega = 0.95;
        guidedFilterRadius = 60;
        guidedFilterEps = 0.0001;
        transmissionLowerBound = 0.1;

        kernelForEroding = getStructuringElement(erosionType, Size(erosionSize, erosionSize));
    }

    //setter functions
    void setKernel(int _erosionSize, int _erosionType)
    {
        erosionSize = _erosionSize;
        erosionType = _erosionType;
        kernelForEroding = getStructuringElement(erosionType, Size(erosionSize, erosionSize));
    }
    void setKernel(cv::InputArray _kernelForEroding)
    {
        kernelForEroding = _kernelForEroding.getMat();
    }
    void setPercentageBrightestPixelsForAtmoLight(float _percentageBrightestPixelsForAtmoLight)
    {
        percentageBrightestPixelsForAtmoLight = _percentageBrightestPixelsForAtmoLight;
    }
    void setOmega(float _omega)
    {
        omega = _omega;
    }
    void setGuidedFilterRadius(float _guidedFilterRadius)
    {
        guidedFilterRadius = _guidedFilterRadius;
    }
    void setGuidedFilterEps(float _guidedFilterEps)
    {
        guidedFilterEps = _guidedFilterEps;
    }
    void setTransmissionLowerBound(float _transmissionLowerBound)
    {
        transmissionLowerBound = _transmissionLowerBound;
    }

    void getDarkChannel(cv::InputArray _src, cv::OutputArray _dst)
    {
        const Mat src = _src.getMat();
        if (src.channels() != 3)
            CV_Error(Error::BadNumChannels,
                     "Unsupported channels count: Only 3 channel BGR images supported");
        erode(src, _dst, kernelForEroding);
        std::vector<Mat> planes(3);
        split(_dst, planes);
        _dst.assign(min(planes[2], min(planes[1], planes[0])));
        return;
    }

    Scalar getAtmoLight(cv::InputArray _src)
    {

        const Mat src = _src.getMat();
        Mat darkChannel;
        getDarkChannel(src, darkChannel);

        int pixelsToKeep = src.rows * src.cols * percentageBrightestPixelsForAtmoLight;
        int thresholdAtmo = intensityOfNthBrightestPixel(darkChannel, pixelsToKeep);

        // Apply a threshold to get a mask of the $percentAtmo brightest pixels.
        Mat dst;
        threshold(darkChannel, dst, thresholdAtmo, 255, cv::THRESH_BINARY);

        Scalar atmoLight = mean(src, dst);

        return atmoLight;
    }

    void getTransmission(cv::InputArray _src, cv::OutputArray _dst, Scalar atmoLight)
    {
        const Mat src = _src.getMat();
        Mat dst;
        Mat src_float;
        src.convertTo(src_float, CV_32FC3);
        Mat src_normalized = src_float / atmoLight;
        getDarkChannel(src_normalized, dst);
        dst = 1 - dst * omega;
        _dst.assign(dst);
    }

    void refineTransmission(cv::InputArray _src, cv::InputArray _unrefinedTransmission, cv::OutputArray _refinedTransmission)
    {
        Mat gray;
        cvtColor(_src, gray, COLOR_BGR2GRAY);
        ximgproc::guidedFilter(gray, _unrefinedTransmission, _refinedTransmission, guidedFilterRadius, guidedFilterEps);
    }

    virtual void dehaze(cv::InputArray _src, cv::OutputArray _dst) CV_OVERRIDE
    {
        const cv::Mat src = _src.getMat();
        CV_Assert(src.channels() == 3);
        CV_Assert(src.type() == CV_8UC3);

        Scalar atmoLight = getAtmoLight(src);

        Mat unrefinedTransmission;
        getTransmission(src, unrefinedTransmission, atmoLight);

        Mat refinedTransmission;

        refineTransmission(src, unrefinedTransmission, refinedTransmission);
        refinedTransmission = max(refinedTransmission, transmissionLowerBound);

        std::vector<Mat> planes;

        for (int i = 0; i < 3; i++)
            planes.push_back(refinedTransmission);

        merge(planes, refinedTransmission);

        Mat src_32FC3;
        src.convertTo(src_32FC3, CV_32F);

        Mat radiance_32FC3;
        Mat radiance;

        radiance_32FC3 = (src_32FC3 - atmoLight) / refinedTransmission + atmoLight;
        radiance_32FC3.convertTo(radiance, CV_8U);

        _dst.assign(radiance);

        return;
    }
};

inline HazeRemovalHeImpl *getLocalImpl(HazeRemovalBase::HazeRemovalImpl *ptr)
{
    HazeRemovalHeImpl *impl = static_cast<HazeRemovalHeImpl *>(ptr);
    CV_Assert(impl);
    return impl;
}

} // namespace

namespace cv {
namespace haze_removal {

#ifdef OPENCV_ENABLE_NONFREE

Ptr<DarkChannelPriorHazeRemoval> DarkChannelPriorHazeRemoval::create()
{

    Ptr<DarkChannelPriorHazeRemoval> res(new DarkChannelPriorHazeRemoval());
    res->pImpl = makePtr<HazeRemovalHeImpl>();
    return res;
}

#else // ! #ifdef OPENCV_ENABLE_NONFREE

Ptr<DarkChannelPriorHazeRemoval> DarkChannelPriorHazeRemoval::create()
{

    CV_Error(Error::StsNotImplemented,
             "This algorithm is patented and is excluded in this configuration; "
             "Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library");
}

#endif

void DarkChannelPriorHazeRemoval::setKernel(int _erosionSize, int _erosionType)
{
    getLocalImpl(pImpl)->setKernel(_erosionSize, _erosionType);
}
void DarkChannelPriorHazeRemoval::setKernel(cv::InputArray _kernelForEroding)
{
    getLocalImpl(pImpl)->setKernel(_kernelForEroding);
}
void DarkChannelPriorHazeRemoval::setPercentageBrightestPixelsForAtmoLight(float _percentageBrightestPixelsForAtmoLight)
{
    getLocalImpl(pImpl)->setPercentageBrightestPixelsForAtmoLight(_percentageBrightestPixelsForAtmoLight);
}
void DarkChannelPriorHazeRemoval::setOmega(float _omega)
{
    getLocalImpl(pImpl)->setOmega(_omega);
}
void DarkChannelPriorHazeRemoval::setGuidedFilterRadius(float _guidedFilterRadius)
{
    getLocalImpl(pImpl)->setGuidedFilterRadius(_guidedFilterRadius);
}
void DarkChannelPriorHazeRemoval::setGuidedFilterEps(float _guidedFilterEps)
{
    getLocalImpl(pImpl)->setGuidedFilterEps(_guidedFilterEps);
}
void DarkChannelPriorHazeRemoval::setTransmissionLowerBound(float _transmissionLowerBoundsetPlotLineColor)
{
    getLocalImpl(pImpl)->setTransmissionLowerBound(_transmissionLowerBoundsetPlotLineColor);
}

void darkChannelPriorHazeRemoval(cv::InputArray _src, cv::OutputArray _dst)
{
    HazeRemovalHeImpl().dehaze(_src, _dst);
}

}} // cv::haze_removal::
