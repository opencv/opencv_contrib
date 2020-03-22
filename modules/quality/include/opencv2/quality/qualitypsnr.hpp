// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_QUALITY_QUALITYPSNR_HPP
#define OPENCV_QUALITY_QUALITYPSNR_HPP

#include <limits>   // numeric_limits
#include "qualitybase.hpp"
#include "qualitymse.hpp"

namespace cv
{
namespace quality
{

/**
@brief Full reference peak signal to noise ratio (PSNR) algorithm  https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
*/
class CV_EXPORTS_W QualityPSNR
    : public QualityBase {

public:

    /** @brief Default maximum pixel value */
#if __cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1900/*MSVS 2015*/)
    static constexpr double MAX_PIXEL_VALUE_DEFAULT = 255.;
#else
    // support MSVS 2013
    static const int MAX_PIXEL_VALUE_DEFAULT = 255;
#endif

    /**
    @brief Create an object which calculates quality
    @param ref input image to use as the source for comparison
    @param maxPixelValue maximum per-channel value for any individual pixel; eg 255 for uint8 image
    */
    CV_WRAP static Ptr<QualityPSNR> create( InputArray ref, double maxPixelValue = QualityPSNR::MAX_PIXEL_VALUE_DEFAULT )
    {
        return Ptr<QualityPSNR>(new QualityPSNR(QualityMSE::create(ref), maxPixelValue));
    }

    /**
    @brief Compute the PSNR
    @param cmp Comparison image
    @returns Per-channel PSNR value, or std::numeric_limits<double>::infinity() if the MSE between the two images == 0
    */
    CV_WRAP cv::Scalar compute( InputArray cmp ) CV_OVERRIDE
    {
        auto result = _qualityMSE->compute( cmp );
        _qualityMSE->getQualityMap(_qualityMap);  // copy from internal obj to this obj
        return _mse_to_psnr(
            result
            , _maxPixelValue
        );
    }

    /** @brief Implements Algorithm::empty()  */
    CV_WRAP bool empty() const CV_OVERRIDE { return _qualityMSE->empty() && QualityBase::empty(); }

    /** @brief Implements Algorithm::clear()  */
    CV_WRAP void clear() CV_OVERRIDE { _qualityMSE->clear(); QualityBase::clear(); }

    /**
    @brief static method for computing quality
    @param ref reference image
    @param cmp comparison image
    @param qualityMap output quality map, or cv::noArray()
    @param maxPixelValue maximum per-channel value for any individual pixel; eg 255 for uint8 image
    @returns PSNR value, or std::numeric_limits<double>::infinity() if the MSE between the two images == 0
    */
    CV_WRAP static cv::Scalar compute( InputArray ref, InputArray cmp, OutputArray qualityMap, double maxPixelValue = QualityPSNR::MAX_PIXEL_VALUE_DEFAULT)
    {
        return _mse_to_psnr(
            QualityMSE::compute(ref, cmp, qualityMap)
            , maxPixelValue
        );
    }

    /** @brief return the maximum pixel value used for PSNR computation */
    CV_WRAP double getMaxPixelValue() const { return _maxPixelValue; }

    /**
    @brief sets the maximum pixel value used for PSNR computation
    @param val Maximum pixel value
    */
    CV_WRAP void setMaxPixelValue(double val) { this->_maxPixelValue = val; }

protected:

    Ptr<QualityMSE> _qualityMSE;
    double _maxPixelValue = QualityPSNR::MAX_PIXEL_VALUE_DEFAULT;

    /** @brief Constructor */
    QualityPSNR( Ptr<QualityMSE> qualityMSE, double maxPixelValue )
        : _qualityMSE(std::move(qualityMSE))
        , _maxPixelValue(maxPixelValue)
    {}

    // convert mse to psnr
    static double _mse_to_psnr(double mse, double max_pixel_value)
    {
        return (mse == 0.)
            ? std::numeric_limits<double>::infinity()
            : 10. * std::log10((max_pixel_value * max_pixel_value) / mse)
            ;
    }

    // convert scalar of mses to psnrs
    static cv::Scalar _mse_to_psnr(cv::Scalar mse, double max_pixel_value)
    {
        for (int i = 0; i < mse.rows; ++i)
            mse(i) = _mse_to_psnr(mse(i), max_pixel_value);
        return mse;
    }

};    // QualityPSNR
}   // quality
}   // cv
#endif