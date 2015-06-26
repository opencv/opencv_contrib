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
 // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

/************************************************************************************
*									   Includes										*
************************************************************************************/
#include "precomp.hpp"
#include <numeric>
#include <algorithm>

// Eigen (for solving the sparse linear system)
#include <Eigen/SparseCore>
#include <Eigen/Sparse>

/************************************************************************************
*									  Namespaces									*
************************************************************************************/
using std::vector;

/************************************************************************************
*									 Declarations									*
************************************************************************************/
const float EPS = 1e-7;

/************************************************************************************
*									Implementation									*
************************************************************************************/

namespace cv
{
namespace optflow
{
    // Values must be of size 2 or greater
    template <typename T>
    void getStat(const vector<T>& values, size_t size, T& mean, T& var)
    {
        T sum = std::accumulate(values.begin(), values.begin() + size, 0.0);
        mean = sum / size;

        T accum = 0.0;
        std::for_each(values.begin(), values.begin() + size, [&](const T d) {
            accum += (d - mean) * (d - mean);
        });

        var = accum / (size - 1);
    }

    // weights(1:n) = 1 + (1 / var) * (weights(1:n) - mean) * (image(i, j) - mean)
    void weight_linear(vector<float>& weights, size_t weightSize, float center, float mean, float var)
    {
        float ivar = (1 / var);
        for (size_t i = 0; i < weightSize; ++i)
        {
            weights[i] = 1 + ivar * (weights[i] - mean) * (center - mean);
        }
    }

    // weights(1:n) = exp(-(weights(1:n) - image(i, j)) .^ 2 / (0.6 * var))
    void weight_exp(vector<float>& weights, size_t weightSize, float center, float mean, float var)
    {
        float diff;
        for (size_t i = 0; i < weightSize; ++i)
        {
            diff = weights[i] - center;
            weights[i] = exp(-diff*diff / (0.6f*var));
        }
    }

    class ScaleMapImpl : public ScaleMap
    {
    public:
        //!	Default Constructor.
        ScaleMapImpl(bool exponential);

        void compute(InputArray image,
            const std::vector<KeyPoint>& keypoints, Mat& scalemap);

    private:

        //! Function pointer to the selected weight function
        void(*mWeightFunc)(std::vector<float>& weights, size_t weightSize,
            float center, float mean, float var);
    };

    Ptr<ScaleMap> ScaleMap::create(bool exponential)
    {
        return makePtr<ScaleMapImpl>(exponential);
    }

    ScaleMapImpl::ScaleMapImpl(bool exponential) : 
        mWeightFunc(exponential ? &weight_exp : &weight_linear)
    {
    }

    void ScaleMapImpl::compute(InputArray image, const std::vector<KeyPoint>& keypoints,
        Mat& scalemap)
    {
        size_t r, c, i, j, nr, nc, min_row, max_row, min_col, max_col;
        size_t neighborCount, coefficientCount = 0;
        std::vector<Eigen::Triplet<float>> coefficients;	// list of non-zeros coefficients
        vector<float> weights(9);
        float m, v, sum;
        Mat I = image.getMat();
        size_t pixels = I.total();
        int index = -1, nindex;

        // Initialization
        vector<bool> scale(pixels, false);
        for (i = 0; i < keypoints.size(); ++i)
        {
            const KeyPoint& feat = keypoints[i];
            scale[((int)feat.pt.y)*I.cols + (int)feat.pt.x] = true;
        }

        // Adjust image values
        cv::Mat scaledImg = I.clone();
        scaledImg += 1;
        scaledImg *= (1 / 32.0f);

        // For each pixel in the image
        for (r = 0; r < scaledImg.rows; ++r)
        {
            min_row = (size_t)std::max(int(r - 1), 0);
            max_row = (size_t)std::min(scaledImg.rows - 1, int(r + 1));
            for (c = 0; c < scaledImg.cols; ++c)
            {
                // Increment pixel index
                ++index;

                // If this is not a feature point
                if (!scale[index])
                {
                    min_col = (size_t)std::max(int(c - 1), 0);
                    max_col = (size_t)std::min(scaledImg.cols - 1, int(c + 1));
                    neighborCount = 0;

                    // Loop over 3x3 neighborhoods matrix
                    // and calculate the variance of the intensities
                    for (nr = min_row; nr <= max_row; ++nr)
                    {
                        for (nc = min_col; nc <= max_col; ++nc)
                        {
                            if (nr == r && nc == c) continue;
                            weights[neighborCount++] = scaledImg.at<float>(nr, nc);
                        }
                    }
                    weights[neighborCount] = scaledImg.at<float>(r, c);

                    // Calculate the weights statistics
                    getStat(weights, neighborCount + 1, m, v);
                    m *= 0.6;
                    if (v < EPS) v = EPS;	// Avoid division by 0

                    // Apply weight function
                    mWeightFunc(weights, neighborCount, scaledImg.at<float>(r, c), m, v);

                    // Normalize the weights and set to coefficients
                    sum = std::accumulate(weights.begin(), weights.begin() + neighborCount, 0.0f);
                    i = 0;
                    for (nr = min_row; nr <= max_row; ++nr)
                    {
                        for (nc = min_col; nc <= max_col; ++nc)
                        {
                            if (nr == r && nc == c) continue;
                            nindex = nr*scaledImg.cols + nc;
                            coefficients.push_back(Eigen::Triplet<float>(
                                index, nindex, -weights[i++] / sum));
                        }
                    }
                }

                // Add center coefficient
                coefficients.push_back(Eigen::Triplet<float>(index, index, 1));
            }
        }

        // Build right side equation vector
        Eigen::VectorXf b = Eigen::VectorXf::Zero(pixels);
        for (i = 0; i < keypoints.size(); ++i)
        {
            const KeyPoint& feat = keypoints[i];
            b[((int)feat.pt.y)*scaledImg.cols + (int)feat.pt.x] = feat.size;
        }

        // Build left side equation matrix
        Eigen::SparseMatrix<float> A(pixels, pixels);
        A.setFromTriplets(coefficients.begin(), coefficients.end());

        // Solving
        Eigen::SparseLU<Eigen::SparseMatrix<float>> slu(A);
        Eigen::VectorXf x = slu.solve(b);

        // Copy to output
        scalemap.create(I.rows, I.cols, CV_32F);
        memcpy(scalemap.data, x.data(), pixels*sizeof(float));
    }


}//optflow
}//cv
