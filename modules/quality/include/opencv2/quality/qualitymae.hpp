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
    MAE_Max,
    MAE_Mean
};

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
    CV_WRAP static Ptr<QualityMAE> create(InputArray ref, int statsProc = MAE_Mean);

    /**
    @brief static method for computing quality
    @param ref reference image
    @param cmp comparison image=
    @param qualityMap output quality map, or cv::noArray()
    @param statsProc which statistical method should be apply on the absolute error
    @returns cv::Scalar with per-channel quality values.  Values range from 0 (best) to max float (worst)
    */
    CV_WRAP static Scalar compute( InputArray ref, InputArray cmp, OutputArray qualityMap, int statsProc = MAE_Mean );


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
    QualityMAE(QualityBase::_mat_type ref, int statsProc)
        : _ref(std::move(ref)),
          _flag(statsProc)
    {}

};

} // quality

} // cv

#endif // OPENCV_QUALITY_MAE_HPP
