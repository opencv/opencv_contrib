/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2016, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

/*
Contributed by Gregor Kovalcik <gregor dot kovalcik at gmail dot com>
    based on code provided by Martin Krulis, Jakub Lokoc and Tomas Skopal.

References:
    Martin Krulis, Jakub Lokoc, Tomas Skopal.
    Efficient Extraction of Clustering-Based Feature Signatures Using GPU Architectures.
    Multimedia tools and applications, 75(13), pp.: 8071–8103, Springer, ISSN: 1380-7501, 2016

    Christian Beecks, Merih Seran Uysal, Thomas Seidl.
    Signature quadratic form distance.
    In Proceedings of the ACM International Conference on Image and Video Retrieval, pages 438-445.
    ACM, 2010.
*/


#ifndef _OPENCV_XFEATURES_2D_PCT_SIGNATURES_SAMPLER_HPP_
#define _OPENCV_XFEATURES_2D_PCT_SIGNATURES_SAMPLER_HPP_

#ifdef __cplusplus


#include "constants.hpp"
#include "grayscale_bitmap.hpp"


namespace cv
{
    namespace xfeatures2d
    {
        namespace pct_signatures
        {
            class PCTSampler : public Algorithm
            {
            public:

                static Ptr<PCTSampler> create(
                    const std::vector<Point2f>& initSamplingPoints,
                    int                     grayscaleBits = 4,
                    int                     windowRadius = 3);


                /**
                * @brief Sampling algorithm that produces samples of the input image
                *       using the stored sampling point locations.
                * @param image Input image.
                * @param signature Output list of computed image samples.
                */
                virtual void sample(InputArray image, OutputArray samples) const = 0;


                /**** accessors ****/

                virtual int getSampleCount() const = 0;
                virtual int getGrayscaleBits() const = 0;
                virtual int getWindowRadius() const = 0;
                virtual float getWeightX() const = 0;
                virtual float getWeightY() const = 0;
                virtual float getWeightL() const = 0;
                virtual float getWeightA() const = 0;
                virtual float getWeightB() const = 0;
                virtual float getWeightConstrast() const = 0;
                virtual float getWeightEntropy() const = 0;

                virtual std::vector<Point2f> getSamplingPoints() const = 0;


                virtual void setGrayscaleBits(int grayscaleBits) = 0;
                virtual void setWindowRadius(int radius) = 0;
                virtual void setWeightX(float weight) = 0;
                virtual void setWeightY(float weight) = 0;
                virtual void setWeightL(float weight) = 0;
                virtual void setWeightA(float weight) = 0;
                virtual void setWeightB(float weight) = 0;
                virtual void setWeightContrast(float weight) = 0;
                virtual void setWeightEntropy(float weight) = 0;

                virtual void setWeight(int idx, float value) = 0;
                virtual void setWeights(const std::vector<float>& weights) = 0;
                virtual void setTranslation(int idx, float value) = 0;
                virtual void setTranslations(const std::vector<float>& translations) = 0;

                virtual void setSamplingPoints(std::vector<Point2f> samplingPoints) = 0;

            };

        }
    }
}


#endif

#endif
