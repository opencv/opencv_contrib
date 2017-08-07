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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_STRUCTURED_EDGE_DETECTION_HPP__
#define __OPENCV_STRUCTURED_EDGE_DETECTION_HPP__
#ifdef __cplusplus

/** @file
@date Jun 17, 2014
@author Yury Gitman
*/

#include <opencv2/core.hpp>

namespace cv
{
namespace ximgproc
{

//! @addtogroup ximgproc_edge
//! @{

/*!
  Helper class for training part of @cite Dollar2013 .
 */
class CV_EXPORTS_W RFFeatureGetter : public Algorithm
{
public:

    /*!
     * This function extracts feature channels from src.
     * Then StructureEdgeDetection uses this feature space
     * to detect edges.
     *
     * \param src : source image to extract features
     * \param features : output n-channel floating point feature matrix.
     *
     * \param gnrmRad : __rf.options.gradientNormalizationRadius
     * \param gsmthRad : __rf.options.gradientSmoothingRadius
     * \param shrink : __rf.options.shrinkNumber
     * \param outNum : __rf.options.numberOfOutputChannels
     * \param gradNum : __rf.options.numberOfGradientOrientations
     */
    CV_WRAP virtual void getFeatures(const Mat &src, Mat &features,
                                     const int gnrmRad,
                                     const int gsmthRad,
                                     const int shrink,
                                     const int outNum,
                                     const int gradNum) const = 0;
};

CV_EXPORTS_W Ptr<RFFeatureGetter> createRFFeatureGetter();

/** @brief It implements the edge detection algorithm presented by @cite Dollar2013 .

   The class takes the following inputs:
    - an image of type CV_32FC3 with value in the range [0,1]
    - a model filename
   and generates an edge image of type CV_32FC1 with value in the range [0,1].

   A pre-trained model can be found at <https://github.com/opencv/opencv_extra/blob/master/testdata/cv/ximgproc/model.yml.bin>.
   Refer to the tutorial <http://docs.opencv.org/trunk/d2/d59/tutorial_ximgproc_training.html> for how to train you own models.

   The following is an example of how to use this class.
   @code
   cv::String model_filename = "SED_model/model.yml.bin";
   cv::String image_filename = "images/teddy/im2.ppm";

   cv::Mat image, edge;

   image = cv::imread(image_filename, cv::IMREAD_COLOR);
   CV_Assert(!image.empty());
   image.convertTo(image, CV_32F, 1.0/255);

   cv::Ptr<cv::ximgproc::StructuredEdgeDetection> pDollar = cv::ximgproc::StructuredEdgeDetection::create(model_filename);
   pDollar->detectEdges(image, edge);

   cv::Mat orientation_map;
   pDollar->computeOrientation(edge, orientation_map);

   // edge_nms is thinner than edges
   cv::Mat edge_nms;
   pDollar->edgesNms(edge, orientation_map, edge_nms, 2, 0, 1, true);

   cv::namedWindow("edges", cv::WINDOW_NORMAL);
   cv::imshow("edges", edge);

   cv::namedWindow("edges nms", cv::WINDOW_NORMAL);
   cv::imshow("edges nms", edge_nms);

   cv::waitKey(0);
   @endcode
*/
class CV_EXPORTS_W StructuredEdgeDetection : public Algorithm
{
public:

    /** @brief The function detects edges in src and draws them to dst.

    The algorithm underlying this function is much more robust to texture presence than common
    approaches, e.g. Sobel
    @param _src source image (CV_32FC3 with values in [0,1]) to detect edges
    @param _dst destination image (grayscale, CV_32FC1 with values in [0,1]) where edges are drawn
    @sa Sobel, Canny
     */
    CV_WRAP virtual void detectEdges(cv::InputArray _src, cv::OutputArray _dst) const = 0;

    /** @brief The function computes orientation from edge image.

    @param _src edge image.
    @param _dst orientation image.
     */
    CV_WRAP virtual void computeOrientation(cv::InputArray _src, cv::OutputArray _dst) const = 0;

    /** @brief The function edgenms in edge image and suppresses edges that are stronger in orthogonal direction.

    @param edge_image edge image from detectEdges function.
    @param orientation_image orientation image from computeOrientation function.
    @param _dst suppressed image (grayscale, float, in [0;1])
    @param r radius for NMS suppression.
    @param s radius for boundary suppression.
    @param m multiplier for conservative suppression.
    @param isParallel enables/disables parallel computing.
     */
    CV_WRAP virtual void edgesNms(cv::InputArray edge_image, cv::InputArray orientation_image, cv::OutputArray _dst, int r = 2, int s = 0, float m = 1, bool isParallel = true) const = 0;

    /*! @brief Create an instance of this class.

        It needs to specify the model filename.

        The format of the model file can be either a `.yml` text file or a binary file.
        A pre-trained model file in `.yml` format can be found at <https://github.com/opencv/opencv_extra/blob/master/testdata/cv/ximgproc/model.yml.gz>
        and a binary model file can be obtained at <https://github.com/opencv/opencv_extra/blob/master/testdata/cv/ximgproc/model.yml.bin>

        The binary format can be parsed much **faster** than the text format.

        If you want to train you own models, please refer to the tutorial <http://docs.opencv.org/trunk/d2/d59/tutorial_ximgproc_training.html>
        and use the script at <https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/tutorials/scripts/modelConvert.m>
        to generate a model in `.yml` format and the script
        <https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/tutorials/scripts/modelConvertToBin.m> for
        binary format.

        @param model_filename   The name of the file that saves the model parameters. Notice that it MUST have one of the following
                                extensions: `.yml`, `.yml.gz` or `.bin`.
        @param howToGetFeatures Optional object which inherits from RFFeatureGetter. It is needed only when you want to train your own model.
                                Otherwise, the default value is good enough. Refer to the tutorial
                                <http://docs.opencv.org/trunk/d2/d59/tutorial_ximgproc_training.html>
                                to implement you own feature extraction class.
    */
    CV_WRAP static Ptr<StructuredEdgeDetection> create(const String &model_filename, Ptr<const RFFeatureGetter> howToGetFeatures = Ptr<RFFeatureGetter>());
};

} // end namespace ximgproc
} // end namespace cv
#endif // #ifdef __cplusplus
#endif /* __OPENCV_STRUCTURED_EDGE_DETECTION_HPP__ */
