// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_QUALITY_QUALITYPSNR_HPP
#define OPENCV_QUALITY_QUALITYPSNR_HPP

#include <limits>   // numeric_limits
#include "QualityBase.hpp"
#include "QualityMSE.hpp"

namespace cv {
	namespace quality {

        namespace detail {

            // convert mse to psnr
            inline CV_CONSTEXPR double mse_to_psnr( double mse, double max_pixel_value )
            { 
                return (mse == 0.) 
                    ? std::numeric_limits<double>::infinity() 
                    : 10. * std::log10((max_pixel_value * max_pixel_value ) / mse)
                    ; 
            }

            // convert scalar of mses to psnrs
            inline cv::Scalar mse_to_psnr(cv::Scalar mse, double max_pixel_value)
            {
                for (int i = 0; i < mse.rows; ++i)
                    mse(i) = mse_to_psnr(mse(i), max_pixel_value);
                return std::move(mse);
            }

        }

        /** @brief Default maximum pixel value */
        #define QUALITY_PSNR_MAX_PIXEL_VALUE_DEFAULT 255.

		/** 
        @brief Full reference peak signal to noise ratio (PSNR) algorithm  https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        */
		class CV_EXPORTS QualityPSNR 
            : public QualityBase {

		public:
            
            /** 
            @brief Create an object which calculates quality via mean square error
	        @param refImgs input image(s) to use as the source for comparison
            */
            CV_WRAP static Ptr<QualityPSNR> create(InputArrayOfArrays refImgs, double maxPixelValue = QUALITY_PSNR_MAX_PIXEL_VALUE_DEFAULT)
            {
                return Ptr<QualityPSNR>(new QualityPSNR(QualityMSE::create(refImgs), maxPixelValue));
            }
            
            /** @brief compute the PSNR
            @returns cv::Scalar with per-channel PSNR.  Values range from 0 (worst) to +infinity (best)
            */
            CV_WRAP cv::Scalar compute(InputArrayOfArrays cmpImgs) CV_OVERRIDE
            {
                return detail::mse_to_psnr(
                    _qualityMSE->compute(cmpImgs)
                    , _maxPixelValue
                );
            }

            /** @brief Implements Algorithm::empty()  */
            CV_WRAP bool empty() const CV_OVERRIDE { return _qualityMSE->empty(); }

            /** @brief Implements Algorithm::clear()  */
            CV_WRAP void clear() CV_OVERRIDE { _qualityMSE->clear(); Algorithm::clear(); }

            /**
            @brief static method for computing quality
            @param refImgs reference image(s)
            @param cmpImgs comparison image(s)
            @param output qualityMaps quality map(s)
            @param maxPixelValue maximum value for a pixel; eg 255 for uint8 images
            @returns PSNR value, or std::numeric_limits<double>::infinity() if the MSE between the two images == 0
            The PSNR for multi-frame images is computed by calculating the average MSE of all frames and then generating the PSNR from that value
            */
            CV_WRAP static cv::Scalar compute(InputArrayOfArrays refImgs, InputArrayOfArrays cmpImgs, OutputArrayOfArrays qualityMaps, double maxPixelValue = QUALITY_PSNR_MAX_PIXEL_VALUE_DEFAULT)
            {
                return detail::mse_to_psnr(
                    QualityMSE::compute(refImgs, cmpImgs, qualityMaps)
                    , maxPixelValue
                );
            }

            /** @brief return the maximum pixel value used for PSNR computation */
            CV_WRAP double getMaxPixelValue() const { return _maxPixelValue; }

            /** @brief sets the maximum pixel value used for PSNR computation */
            CV_WRAP void setMaxPixelValue(double val) { this->_maxPixelValue = val; }

            /** @brief Returns pointer to output quality maps images that were generated during computation, if supported by the algorithm.  */
            CV_WRAP const std::vector<quality_map_type>& getQualityMaps() const CV_OVERRIDE { return _qualityMSE->getQualityMaps(); }

        protected:

            Ptr<QualityMSE> _qualityMSE;
            double _maxPixelValue = QUALITY_PSNR_MAX_PIXEL_VALUE_DEFAULT;

            /** @brief Constructor */
            QualityPSNR( Ptr<QualityMSE> qualityMSE, double maxPixelValue )
                : _qualityMSE(std::move(qualityMSE))
                , _maxPixelValue(maxPixelValue)
            {}

		};	// QualityPSNR
    } 
}
#endif

