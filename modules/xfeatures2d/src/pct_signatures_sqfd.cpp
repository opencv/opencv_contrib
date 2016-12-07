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
#include "precomp.hpp"

#include "pct_signatures/constants.hpp"
#include "pct_signatures/similarity.hpp"

namespace cv
{
    namespace xfeatures2d
    {
        namespace pct_signatures
        {
            class PCTSignaturesSQFD_Impl : public PCTSignaturesSQFD
            {
            public:
                PCTSignaturesSQFD_Impl(
                    const int distanceFunction,
                    const int similarityFunction,
                    const float similarityParameter)
                    : mDistanceFunction(distanceFunction),
                    mSimilarityFunction(similarityFunction),
                    mSimilarityParameter(similarityParameter)
                {

                }


                float computeQuadraticFormDistance(
                    InputArray _signature0,
                    InputArray _signature1) const;

                void computeQuadraticFormDistances(
                    const Mat& sourceSignature,
                    const std::vector<Mat>& imageSignatures,
                    std::vector<float>& distances) const;


            private:
                int mDistanceFunction;
                int mSimilarityFunction;
                float mSimilarityParameter;

                float computePartialSQFD(
                    const Mat& signature0,
                    const Mat& signature1) const;

            };


            /**
            * @brief Class implementing parallel computing of SQFD distance for multiple images.
            */
            class Parallel_computeSQFDs : public ParallelLoopBody
            {
            private:
                const PCTSignaturesSQFD* mPctSignaturesSQFDAlgorithm;
                const Mat* mSourceSignature;
                const std::vector<Mat>* mImageSignatures;
                std::vector<float>* mDistances;

            public:
                Parallel_computeSQFDs(
                    const PCTSignaturesSQFD* pctSignaturesSQFDAlgorithm,
                    const Mat* sourceSignature,
                    const std::vector<Mat>* imageSignatures,
                    std::vector<float>* distances)
                    : mPctSignaturesSQFDAlgorithm(pctSignaturesSQFDAlgorithm),
                    mSourceSignature(sourceSignature),
                    mImageSignatures(imageSignatures),
                    mDistances(distances)
                {
                    mDistances->resize(imageSignatures->size());
                }

                void operator()(const Range& range) const
                {
                    if (mSourceSignature->empty())
                    {
                        CV_Error(Error::StsBadArg, "Source signature is empty!");
                    }

                    for (int i = range.start; i < range.end; i++)
                    {
                        if (mImageSignatures[i].empty())
                        {
                            CV_Error_(Error::StsBadArg, ("Signature ID: %d is empty!", i));
                        }

                        (*mDistances)[i] = mPctSignaturesSQFDAlgorithm->computeQuadraticFormDistance(
                            *mSourceSignature, (*mImageSignatures)[i]);
                    }
                }
            };


            float PCTSignaturesSQFD_Impl::computeQuadraticFormDistance(
                      InputArray _signature0,
                      InputArray _signature1) const
            {
                // check input
                if (_signature0.empty() || _signature1.empty())
                {
                    CV_Error(Error::StsBadArg, "Empty signature!");
                }

                Mat signature0 = _signature0.getMat();
                Mat signature1 = _signature1.getMat();

                if (signature0.cols != SIGNATURE_DIMENSION || signature1.cols != SIGNATURE_DIMENSION)
                {
                    CV_Error_(Error::StsBadArg, ("Signature dimension must be %d!", SIGNATURE_DIMENSION));
                }

                if (signature0.rows <= 0 || signature1.rows <= 0)
                {
                    CV_Error(Error::StsBadArg, "Signature count must be greater than 0!");
                }

                // compute sqfd
                float result = 0;
                result += computePartialSQFD(signature0, signature0);
                result += computePartialSQFD(signature1, signature1);
                result -= computePartialSQFD(signature0, signature1) * 2;

                return sqrt(result);
            }

            void PCTSignaturesSQFD_Impl::computeQuadraticFormDistances(
                      const Mat& sourceSignature,
                      const std::vector<Mat>& imageSignatures,
                      std::vector<float>& distances) const
            {
                parallel_for_(Range(0, (int)imageSignatures.size()),
                    Parallel_computeSQFDs(this, &sourceSignature, &imageSignatures, &distances));
            }

            float PCTSignaturesSQFD_Impl::computePartialSQFD(
                      const Mat& signature0,
                      const Mat& signature1) const
            {
                float result = 0;
                for (int i = 0; i < signature0.rows; i++)
                {
                    for (int j = 0; j < signature1.rows; j++)
                    {
                        result += signature0.at<float>(i, WEIGHT_IDX) * signature1.at<float>(j, WEIGHT_IDX)
                            * computeSimilarity(mDistanceFunction, mSimilarityFunction, mSimilarityParameter, signature0, i, signature1, j);
                    }
                }
                return result;
            }



        }// end of namespace pct_signatures


        Ptr<PCTSignaturesSQFD> PCTSignaturesSQFD::create(
            const int distanceFunction,
            const int similarityFunction,
            const float similarityParameter)
        {
            return makePtr<pct_signatures::PCTSignaturesSQFD_Impl>(distanceFunction, similarityFunction, similarityParameter);
        }

    }
}
