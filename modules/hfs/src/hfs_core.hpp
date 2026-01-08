// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_HFS_CORE_HPP_
#define _OPENCV_HFS_CORE_HPP_

#include "opencv2/core.hpp"

#include "magnitude/magnitude.hpp"
#include "merge/merge.hpp"

#include "or_utils/or_types.hpp"
#include "slic/slic.hpp"

#define DOUBLE_EPS 1E-6

namespace cv { namespace hfs {


struct HfsSettings
{
    float egbThresholdI;
    int minRegionSizeI;
    float egbThresholdII;
    int minRegionSizeII;
    cv::hfs::slic::slicSettings slicSettings;
};

class HfsCore
{
public:
    HfsCore(int height, int width,
        float segThresholdI, int minRegionSizeI,
        float segThresholdII, int minRegionSizeII,
        float spatialWeight, int spixelSize, int numIter);
    ~HfsCore();

    void loadImage( const cv::Mat& inimg, Ptr<UChar4Image> outimg );
    inline float getEulerDistance( cv::Vec3f in1, cv::Vec3f in2 )
    {
        cv::Vec3f diff = in1 - in2;
        return sqrt(diff.dot(diff));
    }

    cv::Vec4f getColorFeature( const cv::Vec3f& in1, const cv::Vec3f& in2 );
    int getAvgGradientBdry( const cv::Mat& idx_mat,
        const std::vector<cv::Mat> &mag1u, int num_css, cv::Mat &bd_num,
        std::vector<cv::Mat> &gradients );

    void getSegmentationI( const cv::Mat& lab3u,
        const cv::Mat& mag1u, const cv::Mat& idx_mat,
        float c, int min_size, cv::Mat& seg, int& num_css);
    void getSegmentationII(
        const cv::Mat& lab3u, const cv::Mat& mag1u, const cv::Mat& idx_mat,
        float c, int min_size, cv::Mat& seg, int &num_css );
    void drawSegmentationRes( const cv::Mat& seg, const cv::Mat& img3u,
                              int num_css, cv::Mat& show );

    cv::Mat getSLICIdxCpu(const cv::Mat& img3u, int &num_css);
    int processImageCpu( const cv::Mat& img3u, cv::Mat& seg );
    int processImageGpu(const cv::Mat& img3u, cv::Mat& seg);

    void constructEngine();
    void reconstructEngine();

public:
    HfsSettings hfsSettings;

private:
    std::vector<float> w1, w2;
    Ptr<Magnitude> mag_engine;

#ifdef _HFS_CUDA_ON_
public:
    cv::Mat getSLICIdxGpu(const cv::Mat& img3u, int &num_css);
private:
    cv::Ptr<UChar4Image> in_img, out_img;
    cv::Ptr<slic::engines::CoreEngine> gslic_engine;
#endif

};

}}

#endif
