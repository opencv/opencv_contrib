// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/core.hpp>
#include <opencv2/core/utils/trace.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/xphoto.hpp"

namespace cv { namespace xphoto {

#ifdef OPENCV_ENABLE_NONFREE
static inline
void mapLuminance(Mat src, Mat dst, Mat lum, Mat new_lum, float saturation)
{
    std::vector<Mat> channels(3);
    split(src, channels);
    for(int i = 0; i < 3; i++) {
        channels[i] = channels[i].mul(1.0f / lum);
        pow(channels[i], saturation, channels[i]);
        channels[i] = channels[i].mul(new_lum);
    }
    merge(channels, dst);
}

static inline
void log_(const Mat& src, Mat& dst)
{
    max(src, Scalar::all(1e-4), dst);
    log(dst, dst);
}

class TonemapDurandImpl CV_FINAL : public TonemapDurand
{
public:
    TonemapDurandImpl(float _gamma, float _contrast, float _saturation, float _sigma_color, float _sigma_space) :
        name("TonemapDurand"),
        gamma(_gamma),
        contrast(_contrast),
        saturation(_saturation),
        sigma_color(_sigma_color),
        sigma_space(_sigma_space)
    {
    }

    void process(InputArray _src, OutputArray _dst) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();

        Mat src = _src.getMat();
        CV_Assert(!src.empty());
        _dst.create(src.size(), CV_32FC3);
        Mat img = _dst.getMat();
        Ptr<Tonemap> linear = createTonemap(1.0f);
        linear->process(src, img);

        Mat gray_img;
        cvtColor(img, gray_img, COLOR_RGB2GRAY);
        Mat log_img;
        log_(gray_img, log_img);
        Mat map_img;
        bilateralFilter(log_img, map_img, -1, sigma_color, sigma_space);

        double min, max;
        minMaxLoc(map_img, &min, &max);
        float scale = contrast / static_cast<float>(max - min);
        exp(map_img * (scale - 1.0f) + log_img, map_img);
        log_img.release();

        mapLuminance(img, img, gray_img, map_img, saturation);
        pow(img, 1.0f / gamma, img);
    }

    float getGamma() const CV_OVERRIDE { return gamma; }
    void setGamma(float val) CV_OVERRIDE { gamma = val; }

    float getSaturation() const CV_OVERRIDE { return saturation; }
    void setSaturation(float val) CV_OVERRIDE { saturation = val; }

    float getContrast() const CV_OVERRIDE { return contrast; }
    void setContrast(float val) CV_OVERRIDE { contrast = val; }

    float getSigmaColor() const CV_OVERRIDE { return sigma_color; }
    void setSigmaColor(float val) CV_OVERRIDE { sigma_color = val; }

    float getSigmaSpace() const CV_OVERRIDE { return sigma_space; }
    void setSigmaSpace(float val) CV_OVERRIDE { sigma_space = val; }

    void write(FileStorage& fs) const CV_OVERRIDE
    {
        writeFormat(fs);
        fs << "name" << name
           << "gamma" << gamma
           << "contrast" << contrast
           << "sigma_color" << sigma_color
           << "sigma_space" << sigma_space
           << "saturation" << saturation;
    }

    void read(const FileNode& fn) CV_OVERRIDE
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        gamma = fn["gamma"];
        contrast = fn["contrast"];
        sigma_color = fn["sigma_color"];
        sigma_space = fn["sigma_space"];
        saturation = fn["saturation"];
    }

protected:
    String name;
    float gamma, contrast, saturation, sigma_color, sigma_space;
};

Ptr<TonemapDurand> createTonemapDurand(float gamma, float contrast, float saturation, float sigma_color, float sigma_space)
{
    return makePtr<TonemapDurandImpl>(gamma, contrast, saturation, sigma_color, sigma_space);
}
#else
Ptr<TonemapDurand> createTonemapDurand(float /*gamma*/, float /*contrast*/, float /*saturation*/, float /*sigma_color*/, float /*sigma_space*/)
{
    CV_Error(Error::StsNotImplemented,
        "This algorithm is patented and is excluded in this configuration; "
        "Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library");
}
#endif  // OPENCV_ENABLE_NONFREE

}}  // namespace
