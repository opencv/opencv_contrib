// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
 * MIT License
 *
 * Copyright (c) 2018 Pedro Diamel Marrero Fernández
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _MCC_CHECKER_MODEL_HPP
#define _MCC_CHECKER_MODEL_HPP

namespace cv
{
namespace mcc
{

//! @addtogroup mcc
//! @{

/** CChartModel
      *
      * @brief Class for handing the different chart models.
      *
      *  This class contains the variables and functions which are specific
      *  to a given chart type and will be used for it detections.

      */

class CChartModel
{

public:
    /** SUBCCMModel
        *
        * @brief Information about a continuous subregion of the chart.
        *
        * Usually not all the cells of the chart can be detected by the
        * detector. This submodel contains the information about the
        * detected squares.
        */

    typedef struct
        _SUBCCMModel
    {

        cv::Mat sub_chart;
        cv::Size2i color_size;
        std::vector<cv::Point2f> centers;

    } SUBCCMModel;

public:
    CChartModel(const TYPECHART chartType);
    ~CChartModel();

    /** @brief evaluate submodel in this checker type*/
    bool evaluate(const SUBCCMModel &subModel, int &offset, int &iTheta, float &error);

    // function utils

    void copyToColorMat(OutputArray lab, int cs = 0);
    void rotate90();
    void flip();

public:
    // Cie L*a*b* values use illuminant D50 2 degree observer sRGB values for
    // for iluminante D65.

    cv::Size2i size;
    cv::Size2f boxsize;
    std::vector<cv::Point2f> box;
    std::vector<cv::Point2f> cellchart;
    std::vector<cv::Point2f> center;
    std::vector<std::vector<float>> chart;

protected:
    /** \brief match checker color
          * \param[in] subModel sub-checker
          * \param[in] iTheta angle
          * \param[out] error
          * \param[out] ierror
          * \return state
          */
    bool match(const SUBCCMModel &subModel, int iTheta, float &error, int &ierror);

    /** \brief euclidian dist L2 for Lab space
          * \note
          * \f$ \sum_i \sqrt (\sum_k (ab1-ab2)_k.^2) \f$
          * \param[in] lab1
          * \param[in] lab2
          * \return distance
          */
    float dist_color_lab(InputArray lab1, InputArray lab2);

    /** \brief rotate matrix 90 degree */
    void rot90(InputOutputArray mat, int itheta);
};
/** CChecker
      *
      * \brief checker model
      * \author Pedro Marrero Fernandez
      *
      */
class CCheckerImpl : public CChecker
{
public:
public:
    CCheckerImpl() {}
    ~CCheckerImpl() {}

    void setTarget(TYPECHART _target)  CV_OVERRIDE;
    void setBox(std::vector<Point2f> _box) CV_OVERRIDE;
    void setPerPatchCosts(std::vector<Point2f> cost) CV_OVERRIDE;

    // Detected directly by the detector or found using geometry, useful to determine if a patch is occluded
    void setBoxDetectionType(std::vector<std::vector<Point2f>> detectionType) CV_OVERRIDE;
    void setChartsRGB(Mat _chartsRGB) CV_OVERRIDE;
    void setChartsYCbCr(Mat _chartsYCbCr) CV_OVERRIDE;
    void setCost(float _cost) CV_OVERRIDE;
    void setCenter(Point2f _center) CV_OVERRIDE;
    void setPatchCenters(std::vector<Point2f> _patchCenters) CV_OVERRIDE;
    void setPatchBoundingBoxes(std::vector<Point2f> _patchBoundingBoxes) CV_OVERRIDE;

    TYPECHART getTarget() CV_OVERRIDE;
    std::vector<Point2f> getBox() CV_OVERRIDE;
    std::vector<Point2f> getPerPatchCosts() CV_OVERRIDE;
    std::vector<std::vector<Point2f>> getBoxDetectionType() CV_OVERRIDE;
    Mat getChartsRGB() CV_OVERRIDE;
    Mat getChartsYCbCr() CV_OVERRIDE;
    float getCost() CV_OVERRIDE;
    Point2f getCenter() CV_OVERRIDE;
    std::vector<Point2f> getPatchCenters() CV_OVERRIDE;
    std::vector<Point2f> getPatchBoundingBoxes(float sideRatio=0.5) CV_OVERRIDE;

    bool calculate() CV_OVERRIDE;//basic precomp

private:
    TYPECHART target;             ///< type of checkercolor
    std::vector<cv::Point2f> box; ///< positions of the corners
    std::vector<cv::Point2f> perPatchCost; ///< Cost of each patch in chart
    std::vector<cv::Point2f> patchBoundingBoxes; ///< Bounding box of the patches
    std::vector<cv::Point2f> patchCenters; ///< Centers of all the patches
    std::vector<std::vector<cv::Point2f>> boxDetectionType; ///< contours of patches directly dectected, not using geometry
    cv::Mat chartsRGB;             ///< charts profile in rgb color space
    cv::Mat chartsYCbCr;         ///< charts profile in YCbCr color space
    float cost;                     ///< cost to aproximate
    cv::Point2f center;             ///< center of the chart.
    float defaultSideRatio = 0.5;   ///< ratio of side of patchBoundingBox and actual patch size
};

//////////////////////////////////////////////////////////////////////////////////////////////
// CheckerDraw

/** \brief checker draw
      * \author Pedro Marrero Fernandez
      */
class CCheckerDrawImpl : public CCheckerDraw
{

public:
    CCheckerDrawImpl(Ptr<CChecker> pChecker, cv::Scalar color = CV_RGB(0, 250, 0), int thickness = 2)
        : m_pChecker(pChecker), m_color(color), m_thickness(thickness)
    {
        CV_Assert(pChecker);
    }

    void draw(InputOutputArray img, float sideRatio =0.5,bool drawActualDetection=false) CV_OVERRIDE;

private:
    Ptr<CChecker> m_pChecker;
    cv::Scalar m_color;
    int m_thickness;

};
// @}

} // namespace mcc
} // namespace cv

#endif //_MCC_CHECKER_MODEL_HPP
