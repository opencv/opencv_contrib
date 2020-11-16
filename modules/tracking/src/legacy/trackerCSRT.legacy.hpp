// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/tracking/tracking_legacy.hpp"

namespace cv {
namespace legacy {
inline namespace tracking {
namespace impl {

class TrackerCSRTImpl CV_FINAL : public legacy::TrackerCSRT
{
public:
    cv::tracking::impl::TrackerCSRTImpl impl;

    TrackerCSRTImpl(const legacy::TrackerCSRT::Params &parameters)
        : impl(parameters)
    {
        isInit = false;
    }

    void read(const FileNode& fn) CV_OVERRIDE
    {
        static_cast<legacy::TrackerCSRT::Params&>(impl.params).read(fn);
    }
    void write(FileStorage& fs) const CV_OVERRIDE
    {
        static_cast<const legacy::TrackerCSRT::Params&>(impl.params).write(fs);
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

    virtual void setInitialMask(InputArray mask) CV_OVERRIDE
    {
        impl.setInitialMask(mask);
    }
};

}  // namespace

void legacy::TrackerCSRT::Params::read(const FileNode& fn)
{
    *this = TrackerCSRT::Params();
    if(!fn["padding"].empty())
        fn["padding"] >> padding;
    if(!fn["template_size"].empty())
        fn["template_size"] >> template_size;
    if(!fn["gsl_sigma"].empty())
        fn["gsl_sigma"] >> gsl_sigma;
    if(!fn["hog_orientations"].empty())
        fn["hog_orientations"] >> hog_orientations;
    if(!fn["num_hog_channels_used"].empty())
        fn["num_hog_channels_used"] >> num_hog_channels_used;
    if(!fn["hog_clip"].empty())
        fn["hog_clip"] >> hog_clip;
    if(!fn["use_hog"].empty())
        fn["use_hog"] >> use_hog;
    if(!fn["use_color_names"].empty())
        fn["use_color_names"] >> use_color_names;
    if(!fn["use_gray"].empty())
        fn["use_gray"] >> use_gray;
    if(!fn["use_rgb"].empty())
        fn["use_rgb"] >> use_rgb;
    if(!fn["window_function"].empty())
        fn["window_function"] >> window_function;
    if(!fn["kaiser_alpha"].empty())
        fn["kaiser_alpha"] >> kaiser_alpha;
    if(!fn["cheb_attenuation"].empty())
        fn["cheb_attenuation"] >> cheb_attenuation;
    if(!fn["filter_lr"].empty())
        fn["filter_lr"] >> filter_lr;
    if(!fn["admm_iterations"].empty())
        fn["admm_iterations"] >> admm_iterations;
    if(!fn["number_of_scales"].empty())
        fn["number_of_scales"] >> number_of_scales;
    if(!fn["scale_sigma_factor"].empty())
        fn["scale_sigma_factor"] >> scale_sigma_factor;
    if(!fn["scale_model_max_area"].empty())
        fn["scale_model_max_area"] >> scale_model_max_area;
    if(!fn["scale_lr"].empty())
        fn["scale_lr"] >> scale_lr;
    if(!fn["scale_step"].empty())
        fn["scale_step"] >> scale_step;
    if(!fn["use_channel_weights"].empty())
        fn["use_channel_weights"] >> use_channel_weights;
    if(!fn["weights_lr"].empty())
        fn["weights_lr"] >> weights_lr;
    if(!fn["use_segmentation"].empty())
        fn["use_segmentation"] >> use_segmentation;
    if(!fn["histogram_bins"].empty())
        fn["histogram_bins"] >> histogram_bins;
    if(!fn["background_ratio"].empty())
        fn["background_ratio"] >> background_ratio;
    if(!fn["histogram_lr"].empty())
        fn["histogram_lr"] >> histogram_lr;
    if(!fn["psr_threshold"].empty())
        fn["psr_threshold"] >> psr_threshold;
    CV_Assert(number_of_scales % 2 == 1);
    CV_Assert(use_gray || use_color_names || use_hog || use_rgb);
}
void legacy::TrackerCSRT::Params::write(FileStorage& fs) const
{
    fs << "padding" << padding;
    fs << "template_size" << template_size;
    fs << "gsl_sigma" << gsl_sigma;
    fs << "hog_orientations" << hog_orientations;
    fs << "num_hog_channels_used" << num_hog_channels_used;
    fs << "hog_clip" << hog_clip;
    fs << "use_hog" << use_hog;
    fs << "use_color_names" << use_color_names;
    fs << "use_gray" << use_gray;
    fs << "use_rgb" << use_rgb;
    fs << "window_function" << window_function;
    fs << "kaiser_alpha" << kaiser_alpha;
    fs << "cheb_attenuation" << cheb_attenuation;
    fs << "filter_lr" << filter_lr;
    fs << "admm_iterations" << admm_iterations;
    fs << "number_of_scales" << number_of_scales;
    fs << "scale_sigma_factor" << scale_sigma_factor;
    fs << "scale_model_max_area" << scale_model_max_area;
    fs << "scale_lr" << scale_lr;
    fs << "scale_step" << scale_step;
    fs << "use_channel_weights" << use_channel_weights;
    fs << "weights_lr" << weights_lr;
    fs << "use_segmentation" << use_segmentation;
    fs << "histogram_bins" << histogram_bins;
    fs << "background_ratio" << background_ratio;
    fs << "histogram_lr" << histogram_lr;
    fs << "psr_threshold" << psr_threshold;
}

}}  // namespace

Ptr<legacy::TrackerCSRT> legacy::TrackerCSRT::create(const legacy::TrackerCSRT::Params &parameters)
{
    return makePtr<legacy::tracking::impl::TrackerCSRTImpl>(parameters);
}
Ptr<legacy::TrackerCSRT> legacy::TrackerCSRT::create()
{
    return create(legacy::TrackerCSRT::Params());
}

}  // namespace
