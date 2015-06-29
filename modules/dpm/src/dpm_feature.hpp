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
// Copyright (C) 2015, Itseez Inc, all rights reserved.
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
// In no event shall the Itseez Inc or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __DPM_FEATURE__
#define __DPM_FEATURE__

#include "opencv2/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc.hpp"

#include <string>
#include <vector>

namespace cv
{
namespace dpm
{
// parameters of the feature pyramid
class PyramidParameter
{
    public:
        // number of levels per octave in feature pyramid
        int interval;
        // HOG cell size
        int binSize;
        // horizontal padding (in cells)
        int padx;
        // vertical padding (in cells)
        int pady;
        // scale factor
        double sfactor;
        // maximum number of scales in the pyramid
        int maxScale;
        // scale of each level
        std::vector< double > scales;

    public:
        PyramidParameter()
        {
            // default parameters
            interval = 10;
            binSize = 8;
            padx = 0;
            pady = 0;
            sfactor = 1.0;
            maxScale = 0;
        }

        ~PyramidParameter() {}
};

/** @brief This class contains DPM model parameters
 */
class Feature
{
    public:
        // dimension of the HOG features in a sigle cell
        static const int dimHOG = 32;
        // top dimPCA PCA eigenvectors
        int dimPCA;

        // set pyramid parameter
        void setPyramidParameters(PyramidParameter val)
        {
            params = val;
        }

        // returns pyramid parameters
        PyramidParameter getPyramidParameters()
        {
            return params;
        }

        // constructor
        Feature ();

        // constructor with parameters
        Feature (PyramidParameter p);

        // destrcutor
        ~Feature () {}

        // compute feature pyramid
        void computeFeaturePyramid(const Mat &imageM, std::vector< Mat > &pyramid);

        // project the feature pyramid with PCA coefficient matrix
        void projectFeaturePyramid(const Mat &pcaCoeff, const std::vector< Mat > &pyramid, std::vector< Mat > &projPyramid);

        // compute 32 dimension HOG as described in
        // "Object Detection with Discriminatively Trained Part-based Models"
        // by Felzenszwalb, Girshick, McAllester and Ramanan, PAMI 2010
        static void computeHOG32D(const Mat &imageM, Mat &featM, const int sbin, const int padx, const int pady);

        // compute location features
        void computeLocationFeatures(const int numLevels, Mat &locFeature);

    private:
        PyramidParameter params;

};

#ifdef HAVE_TBB
/** @brief This class computes feature pyramid in parallel
 * using Intel Threading Building Blocks (TBB)
 */
class ParalComputePyramid : public ParallelLoopBody
{
    public:
        // constructor
        ParalComputePyramid(const Mat &inputImage, \
                std::vector< Mat > &outputPyramid,\
                PyramidParameter &p);

        // initializate parameters
        void initialize();

        // parallel loop body
        void operator() (const Range &range) const;

    private:
        // image to compute feature pyramid
        const Mat &imageM;
        // image size
        Size_<double> imSize;
        // output feature pyramid
        std::vector< Mat > &pyramid;
        // pyramid parameters
        PyramidParameter &params;
};
#endif

} // namespace dpm
} // namespace cv

#endif // __DPM_FEATURE_
