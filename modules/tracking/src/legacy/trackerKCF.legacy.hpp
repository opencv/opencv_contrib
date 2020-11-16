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
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#include "opencv2/tracking/tracking_legacy.hpp"

namespace cv {
namespace legacy {
inline namespace tracking {
namespace impl {

/*---------------------------
|  TrackerKCF
|---------------------------*/
class TrackerKCFImpl CV_FINAL : public legacy::TrackerKCF
{
public:
    cv::tracking::impl::TrackerKCFImpl impl;

    TrackerKCFImpl(const legacy::TrackerKCF::Params &parameters)
        : impl(parameters)
    {
        isInit = false;
    }
    void read(const FileNode& fn) CV_OVERRIDE
    {
        static_cast<legacy::TrackerKCF::Params&>(impl.params).read(fn);
    }
    void write(FileStorage& fs) const CV_OVERRIDE
    {
        static_cast<const legacy::TrackerKCF::Params&>(impl.params).write(fs);
    }

    bool initImpl(const Mat& image, const Rect2d& boundingBox) CV_OVERRIDE
    {
        impl.init(image, boundingBox);
        model = impl.model;
        sampler = makePtr<TrackerContribSampler>();
        featureSet = makePtr<TrackerContribFeatureSet>();
        isInit = true;
        return true;
    }
    bool updateImpl(const Mat& image, Rect2d& boundingBox) CV_OVERRIDE
    {
        Rect bb;
        bool res = impl.update(image, bb);
        boundingBox = bb;
        return res;
    }
    void setFeatureExtractor(void (*f)(const Mat, const Rect, Mat&), bool pca_func = false) CV_OVERRIDE
    {
        impl.setFeatureExtractor(f, pca_func);
    }
};

}  // namespace

void legacy::TrackerKCF::Params::read(const cv::FileNode& fn)
{
      *this = TrackerKCF::Params();

      if (!fn["detect_thresh"].empty())
          fn["detect_thresh"] >> detect_thresh;

      if (!fn["sigma"].empty())
          fn["sigma"] >> sigma;

      if (!fn["lambda"].empty())
          fn["lambda"] >> lambda;

      if (!fn["interp_factor"].empty())
          fn["interp_factor"] >> interp_factor;

      if (!fn["output_sigma_factor"].empty())
          fn["output_sigma_factor"] >> output_sigma_factor;

      if (!fn["resize"].empty())
          fn["resize"] >> resize;

      if (!fn["max_patch_size"].empty())
          fn["max_patch_size"] >> max_patch_size;

      if (!fn["split_coeff"].empty())
          fn["split_coeff"] >> split_coeff;

      if (!fn["wrap_kernel"].empty())
          fn["wrap_kernel"] >> wrap_kernel;


      if (!fn["desc_npca"].empty())
          fn["desc_npca"] >> desc_npca;

      if (!fn["desc_pca"].empty())
          fn["desc_pca"] >> desc_pca;

      if (!fn["compress_feature"].empty())
          fn["compress_feature"] >> compress_feature;

      if (!fn["compressed_size"].empty())
          fn["compressed_size"] >> compressed_size;

      if (!fn["pca_learning_rate"].empty())
          fn["pca_learning_rate"] >> pca_learning_rate;
}

void legacy::TrackerKCF::Params::write(cv::FileStorage& fs) const
{
    fs << "detect_thresh" << detect_thresh;
    fs << "sigma" << sigma;
    fs << "lambda" << lambda;
    fs << "interp_factor" << interp_factor;
    fs << "output_sigma_factor" << output_sigma_factor;
    fs << "resize" << resize;
    fs << "max_patch_size" << max_patch_size;
    fs << "split_coeff" << split_coeff;
    fs << "wrap_kernel" << wrap_kernel;
    fs << "desc_npca" << desc_npca;
    fs << "desc_pca" << desc_pca;
    fs << "compress_feature" << compress_feature;
    fs << "compressed_size" << compressed_size;
    fs << "pca_learning_rate" << pca_learning_rate;
}


}}  // namespace legacy::tracking

Ptr<legacy::TrackerKCF> legacy::TrackerKCF::create(const legacy::TrackerKCF::Params &parameters)
{
    return makePtr<legacy::tracking::impl::TrackerKCFImpl>(parameters);
}
Ptr<legacy::TrackerKCF> legacy::TrackerKCF::create()
{
    return create(legacy::TrackerKCF::Params());
}

}
