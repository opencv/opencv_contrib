/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "precomp.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>

namespace cv
{
namespace optflow
{

    class SiftImageExtractorImpl : public SiftImageExtractor
    {
    public:
        //!	Default Constructor.
        SiftImageExtractorImpl(int scale_format = SiftImageExtractor::SCALE_UNIFORM);

        void compute(const Mat& img, Mat& siftImg);

        void compute(const Mat& img0, const Mat& img1, Mat& siftImg0, Mat& siftImg1);

    private:
        void compute(const Mat& img, Mat& siftImg, const Mat& scalemap);

    private:
        Ptr<ScaleMap> mScaleMap;
    };

    Ptr<SiftImageExtractor> SiftImageExtractor::create(int scale_format)
    {
        return makePtr<SiftImageExtractorImpl>(scale_format);
    }

    SiftImageExtractorImpl::SiftImageExtractorImpl(int scale_format)
    {
        if (scale_format == SCALE_UNIFORM) return;

        mScaleMap = ScaleMap::create(scale_format == SCALE_EXP);
    }

    void SiftImageExtractorImpl::compute(const Mat& img, Mat& siftImg)
    {
    }

    void SiftImageExtractorImpl::compute(const Mat& img0, const Mat& img1,
        Mat& siftImg0, Mat& siftImg1)
    {

    }

    void SiftImageExtractorImpl::compute(const Mat& img, Mat& siftImg, const Mat& scalemap)
    {
        cv::Ptr<xfeatures2d::SIFT> siftExtractor = xfeatures2d::SIFT::create();
        int r, c, index = 0;

        std::vector<cv::KeyPoint> keypoints(img.rows*img.cols);

        if (scalemap.empty())
        {
            for (r = 0; r < img.rows; r += 1)
            {
                for (c = 0; c < img.cols; c += 1)
                {
                    cv::KeyPoint& key = keypoints[index++];
                    key.pt.x = c;
                    key.pt.y = r;
                    key.size = 3.0f;
                }
            }
        }
        else
        {
            float* scalemap_data = (float*)scalemap.data;
            for (r = 0; r < img.rows; r += 1)
            {
                for (c = 0; c < img.cols; c += 1)
                {
                    cv::KeyPoint& key = keypoints[index++];
                    key.pt.x = c;
                    key.pt.y = r;
                    key.size = *scalemap_data++;
                }
            }
        }

        cv::Mat desc;

        cv::Mat bimg;
        img.convertTo(bimg, CV_8U);
        siftExtractor->compute(bimg, keypoints, desc);

        // Output sift image
        siftImg.create(img.rows, img.cols, CV_32FC(128));
        memcpy(siftImg.data, desc.data, desc.total()*sizeof(float));
    }

}//optflow
}//cv
