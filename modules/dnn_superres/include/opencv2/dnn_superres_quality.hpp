// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CONTRIB_DNN_SUPERRES_QUALITY_HPP
#define OPENCV_CONTRIB_DNN_SUPERRES_QUALITY_HPP

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdarg>
#include <sstream>
#include <ctime>

#ifdef _WIN32
#include <Windows.h>
#endif

#include <opencv2/dnn_superres.hpp>
#include "opencv2/dnn.hpp"

namespace cv
{
namespace dnn_superres
{

//! @addtogroup dnn_superres
//! @{

/** @brief A class benchmarking the dnn based super resolution algorithms.
 */
class CV_EXPORTS DnnSuperResQuality
{
private:

    static int fontFace;

    static double fontScale;

    static cv::Scalar fontColor;

public:

    /** @brief Sets the font face used for drawing measures on images drawn by showBenchmark().
    @param fontface font face
    */
    static void setFontFace(int fontface);

    /** @brief Sets the font scale used for drawing measures on images drawn by showBenchmark().
    @param fontscale font scale
    */
    static void setFontScale(double fontscale);

    /** @brief Sets the font color used for drawing measures on images drawn by showBenchmark().
    @param fontcolor font color
    */
    static void setFontColor(cv::Scalar fontcolor);

    /** @brief Returns the PSNR of two given image.
    @param img Upscaled image
    @param orig Original image
    @return PSNR value.
    */
    static double psnr(Mat img, Mat orig);

    /** @brief Returns the SSIM of two given image.
    @param img Upscaled image
    @param orig Original image
    @return SSIM value.
    */
    static double ssim(Mat img, Mat orig);

    /** @brief Gives a benchmark for the given super resolution algorithm. It compares it to
    * bicubic, nearest neighbor, and lanczos interpolation methods.
    @param sr DnnSuperRes object
    @param img Image to upscale
    @param showImg Displays the images if set to true
    */
    static void benchmark(DnnSuperResImpl sr, Mat img, bool showImg = false);

    /** @brief Gives a benchmark for the given super resolution algorithm. It compares it to
     * bicubic, nearest neighbor, and lanczos interpolation methods.
    @param sr DnnSuperRes object
    @param img Image to upscale
    @param psnrValues Output container of the PSNR values
    @param ssimValues Output container of the SSIM values
    @param perfValues Output container of the Performance values
    @param showImg Displays the images if set to true
    @param showOutput Writes benchmarking to standard output if set to true
    */
    static void benchmark(DnnSuperResImpl sr, Mat img,
                            std::vector<double>& psnrValues,
                            std::vector<double>& ssimValues,
                            std::vector<double>& perfValues,
                            bool showImg = false,
                            bool showOutput = false);

    /** @brief Displays benchmarking for given images.
    @param orig Original image
    @param images Upscaled images
    @param title Title to window
    @param imageSize Display size of images
    @param imageTitles Titles of images values to display (optional)
    @param psnrValues PSNR values to display (optional)
    @param ssimValues SSIM values to display (optional)
    @param perfValues Speed values to display (optional)
    */
    static void showBenchmark(Mat orig, std::vector<Mat> images, std::string title, Size imageSize,
                            const std::vector<String> imageTitles = std::vector<String>(),
                            const std::vector<double> psnrValues = std::vector<double>(),
                            const std::vector<double> ssimValues = std::vector<double>(),
                            const std::vector<double> perfValues = std::vector<double>());
};

//! @}

}} // cv::dnn_superres::
#endif //OPENCV_CONTRIB_DNN_SUPERRES_QUALITY_HPP
