// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_EDGE_DRAWING_HPP__
#define __OPENCV_EDGE_DRAWING_HPP__

#include <opencv2/core.hpp>

namespace cv
{
namespace ximgproc
{

//! @addtogroup ximgproc_edge_drawing
//! @{

/** @brief Class implementing the ED (EdgeDrawing) @cite topal2012edge, EDLines @cite akinlar2011edlines, EDPF @cite akinlar2012edpf and EDCircles @cite akinlar2013edcircles algorithms
*/

class CV_EXPORTS_W EdgeDrawing : public Algorithm
{
public:

    enum GradientOperator
    {
        PREWITT = 0,
        SOBEL   = 1,
        SCHARR  = 2,
        LSD     = 3
    };

    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_WRAP Params();
        //! Parameter Free mode will be activated when this value is set as true. Default value is false.
        CV_PROP_RW bool PFmode;
        /** @brief indicates the operator used for gradient calculation.

        one of the flags cv::ximgproc::EdgeDrawing::GradientOperator. Default value is PREWITT
        */
        CV_PROP_RW int EdgeDetectionOperator;
        //! threshold value of gradiential difference between pixels. Used to create gradient image. Default value is 20
        CV_PROP_RW int GradientThresholdValue;
        //! threshold value used to select anchor points. Default value is 0
        CV_PROP_RW int AnchorThresholdValue;
        //! Default value is 1
        CV_PROP_RW int ScanInterval;
        /** @brief minimun connected pixels length processed to create an edge segment.

        in gradient image, minimum connected pixels length processed to create an edge segment. pixels having upper value than GradientThresholdValue
        will be processed. Default value is 10
        */
        CV_PROP_RW int MinPathLength;
        //! sigma value for internal GaussianBlur() function. Default value is 1.0
        CV_PROP_RW float Sigma;
        CV_PROP_RW bool SumFlag;
        //! Default value is true. indicates if NFA (Number of False Alarms) algorithm will be used for line and ellipse validation.
        CV_PROP_RW bool NFAValidation;
        //! minimun line length to detect.
        CV_PROP_RW int MinLineLength;
        //! Default value is 6.0
        CV_PROP_RW double MaxDistanceBetweenTwoLines;
        //! Default value is 1.0
        CV_PROP_RW double LineFitErrorThreshold;
        //! Default value is 1.3
        CV_PROP_RW double MaxErrorThreshold;

        void read(const FileNode& fn);
        void write(FileStorage& fs) const;
    };

    /** @brief Detects edges in a grayscale image and prepares them to detect lines and ellipses.

    @param src 8-bit, single-channel, grayscale input image.
    */
    CV_WRAP virtual void detectEdges(InputArray src) = 0;

    /** @brief returns Edge Image prepared by detectEdges() function.

    @param dst returns 8-bit, single-channel output image.
    */
    CV_WRAP virtual void getEdgeImage(OutputArray dst) = 0;

    /** @brief returns Gradient Image prepared by detectEdges() function.

    @param dst returns 16-bit, single-channel output image.
    */
    CV_WRAP virtual void getGradientImage(OutputArray dst) = 0;

    /** @brief Returns std::vector<std::vector<Point>> of detected edge segments, see detectEdges()
    */
    CV_WRAP virtual std::vector<std::vector<Point> > getSegments() = 0;

    /** @brief Returns for each line found in detectLines() its edge segment index in getSegments()
     */
    CV_WRAP virtual std::vector<int> getSegmentIndicesOfLines() const = 0;

    /** @brief Detects lines.

    @param lines  output Vec<4f> contains the start point and the end point of detected lines.
    @note you should call detectEdges() before calling this function.
    */
    CV_WRAP virtual void detectLines(OutputArray lines) = 0;

    /** @brief Detects circles and ellipses.

    @param ellipses  output Vec<6d> contains center point and perimeter for circles, center point, axes and angle for ellipses.
    @note you should call detectEdges() before calling this function.
    */
    CV_WRAP virtual void detectEllipses(OutputArray ellipses) = 0;

    CV_WRAP Params params;

    /** @brief sets parameters.

    this function is meant to be used for parameter setting in other languages than c++ like python.
    @param parameters
    */
    CV_WRAP void setParams(const EdgeDrawing::Params& parameters);
    virtual ~EdgeDrawing() { }
};

/** @brief Creates a smart pointer to a EdgeDrawing object and initializes it
*/
CV_EXPORTS_W Ptr<EdgeDrawing> createEdgeDrawing();
//! @}

}
}

#endif /* __OPENCV_EDGE_DRAWING_HPP__ */
