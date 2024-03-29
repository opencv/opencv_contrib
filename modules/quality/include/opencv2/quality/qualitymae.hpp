// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_QUALITY_MAE_HPP
#define OPENCV_QUALITY_MAE_HPP

#include "qualitybase.hpp"


namespace cv
{

namespace quality
{

/** @brief Flags to choose which algorithm MAE should use.
 */
enum MAEStatsFlags
{
    MAE_MAX,
    MAE_MEAN
};

/** @brief This class implement two algorithm which commonly refered as MAE in the litterature.
Both definition shares the absolute error, which can be defined as: \f[ absolute\_error(x,y) = |I_{ref}(x,y) - I_{cmp}(x,y)|\f].
The two algorithms follows the mathematic:
-   **MAE_MAX**
    \f[score =  \fork{\texttt{absolute\_error(x,y)}}{if \(src(x,y) > score\)}{score}{otherwise}\f]
-   **MAE_MEAN**
    \f[score = \frac{\sum_{r=0}^{nb\_rows}\sum_{c=0}^{nb\_cols} \texttt{absolute\_error(r,c)}}{nb\_rows \times \nb\_cols}\f]
More informations about the the Mean of Absolute Error can be found here: https://en.wikipedia.org/wiki/Mean_absolute_error
*/
class CV_EXPORTS_W QualityMAE : public QualityBase
{
public:
    /** @brief Computes MAE for reference images supplied in class constructor and provided comparison images
    @param cmpImgs Comparison image(s)
    @returns cv::Scalar with per-channel quality values.  Values range from 0 (best) to potentially max float (worst)
    */
    CV_WRAP Scalar compute( InputArray cmpImgs ) CV_OVERRIDE;

    /** @brief Implements Algorithm::empty()  */
    CV_WRAP bool empty() const CV_OVERRIDE { return _ref.empty() && QualityBase::empty(); }

    /** @brief Implements Algorithm::clear()  */
    CV_WRAP void clear() CV_OVERRIDE { _ref = _mat_type(); QualityBase::clear(); }

    /**
    @brief Create an object which calculates quality
    @param ref input image to use as the reference for comparison
    @param statsProc statistical method to apply on the error
    */
    CV_WRAP static Ptr<QualityMAE> create(InputArray ref, int statsProc = MAE_MEAN);

    /**
    @brief static method for computing quality
    @param ref reference image
    @param cmp comparison image=
    @param qualityMap output quality map, or cv::noArray()
    @param statsProc which statistical method should be apply on the absolute error
    @returns cv::Scalar with per-channel quality values.  Values range from 0 (best) to max float (worst)
    */
    CV_WRAP static Scalar compute( InputArray ref, InputArray cmp, OutputArray qualityMap, int statsProc = MAE_MEAN );


protected:

    /** @brief Reference image, converted to internal mat type */
    QualityBase::_mat_type _ref;

    /** @brief What statistics analysis to apply on the absolute error */
    int _flag;

    /**
    @brief Constructor
    @param ref reference image, converted to internal type
    @param statsProc statistical method to apply on the error
    */
    QualityMAE(QualityBase::_mat_type ref, int statsProc);

};

} // quality

} // cv

#endif // OPENCV_QUALITY_MAE_HPP
