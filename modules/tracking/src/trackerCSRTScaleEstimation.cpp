// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "trackerCSRTScaleEstimation.hpp"
#include "trackerCSRTUtils.hpp"

//Discriminative Scale Space Tracking
namespace cv
{

class ParallelGetScaleFeatures : public ParallelLoopBody
{
public:
    ParallelGetScaleFeatures(
        Mat img,
        Point2f pos,
        Size2f base_target_sz,
        float current_scale,
        std::vector<float> &scale_factors,
        Mat scale_window,
        Size scale_model_sz,
        int col_len,
        Mat &result)
    {
        this->img = img;
        this->pos = pos;
        this->base_target_sz = base_target_sz;
        this->current_scale = current_scale;
        this->scale_factors = scale_factors;
        this->scale_window = scale_window;
        this->scale_model_sz = scale_model_sz;
        this->col_len = col_len;
        this->result = result;
    }
    virtual void operator ()(const Range& range) const CV_OVERRIDE
    {
        Mat img_patch_f32, resized_patch, im_64f, hogmatrix, histM, normM;

        for (int s = range.start; s < range.end; s++) {
            Size patch_sz = Size(static_cast<int>(current_scale * scale_factors[s] * base_target_sz.width),
                    static_cast<int>(current_scale * scale_factors[s] * base_target_sz.height));
            Mat img_patch = get_subwindow(img, pos, patch_sz.width, patch_sz.height);
            img_patch.convertTo(img_patch_f32, CV_32FC3);
            resize(img_patch_f32, resized_patch, Size(scale_model_sz.width, scale_model_sz.height), 0, 0, INTER_LINEAR);
            resized_patch.convertTo(im_64f, CV_64FC3, 1.0/255.0);
            computeHOG32D(im_64f, hogmatrix, 4, 1, 1, histM, normM);

            const float win = scale_window.at<float>(0, s);
            const int hog_h = hogmatrix.rows;
            const int hog_w = hogmatrix.cols / 32;
            for (int y = 0; y < hog_h; ++y) {
                const double* hrow = hogmatrix.ptr<double>(y);
                for (int x = 0; x < hog_w; ++x) {
                    const double* pix = hrow + x * 32;
                    const int base_row = x * hog_h + y;
                    for (int i = 0; i < 32; ++i) {
                        result.ptr<float>(i * col_len + base_row)[s] = static_cast<float>(pix[i]) * win;
                    }
                }
            }
        }
    }

    ParallelGetScaleFeatures& operator=(const ParallelGetScaleFeatures &) {
        return *this;
    }

private:
    Mat img;
    Point2f pos;
    Size2f base_target_sz;
    float current_scale;
    std::vector<float> scale_factors;
    Mat scale_window;
    Size scale_model_sz;
    int col_len;
    mutable Mat result;
};


DSST::DSST(const Mat &image,
        Rect2f bounding_box,
        Size2f template_size,
        int numberOfScales,
        float scaleStep,
        float maxModelArea,
        float sigmaFactor,
        float scaleLearnRate):
    scales_count(numberOfScales), scale_step(scaleStep), max_model_area(maxModelArea),
    sigma_factor(sigmaFactor), learn_rate(scaleLearnRate)
{
    original_targ_sz = bounding_box.size();
    Point2f object_center = Point2f(bounding_box.x + original_targ_sz.width / 2,
            bounding_box.y + original_targ_sz.height / 2);

    current_scale_factor = 1.0;
    if(scales_count % 2 == 0)
        scales_count++;

    scale_sigma = static_cast<float>(sqrt(scales_count) * sigma_factor);

    min_scale_factor = static_cast<float>(pow(scale_step,
            cvCeil(log(max(5.0 / template_size.width, 5.0 / template_size.height)) / log(scale_step))));
    max_scale_factor = static_cast<float>(pow(scale_step,
            cvFloor(log(min((float)image.rows / (float)bounding_box.width,
            (float)image.cols / (float)bounding_box.height)) / log(scale_step))));
    ys = Mat(1, scales_count, CV_32FC1);
    float ss, sf;
    for(int i = 0; i < ys.cols; ++i) {
        ss = (float)(i+1) - cvCeil((float)scales_count / 2.0f);
        ys.at<float>(0,i) = static_cast<float>(exp(-0.5 * pow(ss,2) / pow(scale_sigma,2)));
        sf = static_cast<float>(i + 1);
        scale_factors.push_back(pow(scale_step, cvCeil((float)scales_count / 2.0f) - sf));
    }

    scale_window = get_hann_win(Size(scales_count, 1));

    float scale_model_factor = 1.0;
    if(template_size.width * template_size.height * pow(scale_model_factor, 2) > max_model_area)
    {
        scale_model_factor = sqrt(max_model_area /
                (template_size.width * template_size.height));
    }
    scale_model_sz = Size(cvFloor(template_size.width * scale_model_factor),
            cvFloor(template_size.height * scale_model_factor));

    Mat scale_resp = get_scale_features(image, object_center, original_targ_sz,
            current_scale_factor, scale_factors, scale_window, scale_model_sz);
    dft_rows_fwd_ = hal::DFT2D::create(scale_resp.cols, scale_resp.rows, CV_32F, 1, 2,
            CV_HAL_DFT_IS_CONTINUOUS | CV_HAL_DFT_ROWS);

    Mat ysf_row = Mat(ys.size(), CV_32FC2);
    dft(ys, ysf_row, DFT_ROWS | DFT_COMPLEX_OUTPUT, 0);
    ysf = repeat(ysf_row, scale_resp.rows, 1);
    Mat Fscale_resp(scale_resp.size(), CV_32FC2);
    if (!dft_rows_fwd_.empty() && scale_resp.isContinuous() && Fscale_resp.isContinuous())
        dft_rows_fwd_->apply(scale_resp.data, scale_resp.step, Fscale_resp.data, Fscale_resp.step);
    else
        dft(scale_resp, Fscale_resp, DFT_ROWS | DFT_COMPLEX_OUTPUT);
    mulSpectrums(ysf, Fscale_resp, sf_num, 0 , true);
    Mat sf_den_all;
    mulSpectrums(Fscale_resp, Fscale_resp, sf_den_all, 0, true);
    reduce(sf_den_all, sf_den, 0, REDUCE_SUM, -1);
}

DSST::~DSST()
{
}

Mat DSST::get_scale_features(
        Mat img,
        Point2f pos,
        Size2f base_target_sz,
        float current_scale,
        std::vector<float> &scale_factors,
        Mat scale_window,
        Size scale_model_sz)
{
    const int hog_w = scale_model_sz.width / 4;
    const int hog_h = scale_model_sz.height / 4;
    const int col_len = hog_w * hog_h;
    Mat result(Size((int)scale_factors.size(), col_len * 32), CV_32F);

    ParallelGetScaleFeatures parallelGetScaleFeatures(img, pos, base_target_sz,
            current_scale, scale_factors, scale_window, scale_model_sz, col_len, result);
    const double nstripes = std::min(4, std::max(1, cv::getNumThreads()));
    parallel_for_(Range(0, static_cast<int>(scale_factors.size())), parallelGetScaleFeatures, nstripes);
    return result;
}

void DSST::update(const Mat &image, const Point2f object_center)
{
    Mat scale_features = get_scale_features(image, object_center, original_targ_sz,
            current_scale_factor, scale_factors, scale_window, scale_model_sz);
    Mat Fscale_features(scale_features.size(), CV_32FC2);
    if (!dft_rows_fwd_.empty() && scale_features.isContinuous() && Fscale_features.isContinuous())
        dft_rows_fwd_->apply(scale_features.data, scale_features.step, Fscale_features.data, Fscale_features.step);
    else
        dft(scale_features, Fscale_features, DFT_ROWS | DFT_COMPLEX_OUTPUT);
    Mat new_sf_num;
    Mat new_sf_den;
    Mat new_sf_den_all;
    mulSpectrums(ysf, Fscale_features, new_sf_num, DFT_ROWS, true);
    mulSpectrums(Fscale_features, Fscale_features, new_sf_den_all, DFT_ROWS, true);
    reduce(new_sf_den_all, new_sf_den, 0, REDUCE_SUM, -1);

    addWeighted(sf_num, 1.0 - learn_rate, new_sf_num, learn_rate, 0.0, sf_num);
    addWeighted(sf_den, 1.0 - learn_rate, new_sf_den, learn_rate, 0.0, sf_den);
}

float DSST::getScale(const Mat &image, const Point2f object_center)
{
    Mat scale_features = get_scale_features(image, object_center, original_targ_sz,
            current_scale_factor, scale_factors, scale_window, scale_model_sz);

    Mat Fscale_features(scale_features.size(), CV_32FC2);
    if (!dft_rows_fwd_.empty() && scale_features.isContinuous() && Fscale_features.isContinuous())
        dft_rows_fwd_->apply(scale_features.data, scale_features.step, Fscale_features.data, Fscale_features.step);
    else
        dft(scale_features, Fscale_features, DFT_ROWS | DFT_COMPLEX_OUTPUT);

    mulSpectrums(Fscale_features, sf_num, Fscale_features, 0, false);
    Mat scale_resp;
    reduce(Fscale_features, scale_resp, 0, REDUCE_SUM, -1);
    divide_complex_add_scalar(scale_resp, sf_den, 0.01f, scale_resp);
    idft(scale_resp, scale_resp, DFT_REAL_OUTPUT|DFT_SCALE);
    Point max_loc;
    minMaxLoc(scale_resp, NULL, NULL, NULL, &max_loc);

    current_scale_factor *= scale_factors[max_loc.x];
    if(current_scale_factor < min_scale_factor)
        current_scale_factor = min_scale_factor;
    else if(current_scale_factor > max_scale_factor)
        current_scale_factor = max_scale_factor;

    return current_scale_factor;
}
} /* namespace cv */
