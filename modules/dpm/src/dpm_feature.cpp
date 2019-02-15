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

#include "dpm_feature.hpp"

using namespace std;

namespace cv
{
namespace dpm
{

Feature::Feature()
{
}

Feature::Feature (PyramidParameter p):params(p)
{
}

void Feature::computeFeaturePyramid(const Mat &imageM, vector< Mat > &pyramid)
{
    ParalComputePyramid paralTask(imageM, pyramid, params);
    paralTask.initialize();
    parallel_for_(Range(0, params.interval), paralTask);
}

ParalComputePyramid::ParalComputePyramid(const Mat &inputImage, \
        vector< Mat > &outputPyramid,\
        PyramidParameter &p):
    imageM(inputImage), pyramid(outputPyramid), params(p)
{
}

void ParalComputePyramid::initialize()
{
    CV_Assert(params.interval > 0);

    // scale factor between two levels
    params.sfactor = pow(2.0, 1.0/params.interval);
    imSize = imageM.size();
    params.maxScale = 1 + (int)floor(log(min(imSize.width, imSize.height)/(float)(params.binSize*5.0))/log(params.sfactor));

    if (params.maxScale < params.interval)
    {
        CV_Error(CV_StsBadArg, "The image is too small to create a pyramid");
        return;
    }

    pyramid.resize(params.maxScale + params.interval);
    params.scales.resize(params.maxScale + params.interval);
}

void ParalComputePyramid::operator() (const Range &range) const
{
    for (int i = range.start; i != range.end; i++)
    {
        const double scale = (double)(1.0f/pow(params.sfactor, i));
        Mat imScaled;
        resize(imageM, imScaled, imSize * scale);

        params.scales[i] = 2*scale;

        // First octave at twice the image resolution
        Feature::computeHOG32D(imScaled, pyramid[i],
                params.binSize/2, params.padx + 1, params.pady + 1);

        // Second octave at the original resolution
        if (i + params.interval <= params.maxScale)
            Feature::computeHOG32D(imScaled, pyramid[i+params.interval],
                    params.binSize, params.padx + 1, params.pady + 1);

        params.scales[i+params.interval] = scale;

        // Remaining octaves
        for ( int j = i + params.interval; j < params.maxScale; j += params.interval)
        {
            Mat imScaled2;
            Size_<double> imScaledSize = imScaled.size();
            resize(imScaled, imScaled2, imScaledSize*0.5);
            imScaled = imScaled2;
            Feature::computeHOG32D(imScaled2, pyramid[j+params.interval],
                    params.binSize, params.padx + 1, params.pady + 1);
            params.scales[j+params.interval] = params.scales[j]*0.5;
        }
    }
}

void Feature::computeHOG32D(const Mat &imageM, Mat &featM, const int sbin, const int pad_x, const int pad_y)
{
    CV_Assert(pad_x >= 0);
    CV_Assert(pad_y >= 0);
    CV_Assert(imageM.channels() == 3);
    CV_Assert(imageM.depth() == CV_64F);

    // epsilon to avoid division by zero
    const double eps = 0.0001;
    // number of orientations
    const int numOrient = 18;
    // unit vectors to compute gradient orientation
    const double uu[9] = {1.000, 0.9397, 0.7660, 0.5000, 0.1736, -0.1736, -0.5000, -0.7660, -0.9397};
    const double vv[9] = {0.000, 0.3420, 0.6428, 0.8660, 0.9848,  0.9848,  0.8660,  0.6428,  0.3420};

    // image size
    const Size imageSize = imageM.size();
    // block size
    int bW = cvRound((double)imageSize.width/(double)sbin);
    int bH = cvRound((double)imageSize.height/(double)sbin);
    const Size blockSize(bW, bH);
    // size of HOG features
    int oW = max(blockSize.width-2, 0) + 2*pad_x;
    int oH = max(blockSize.height-2, 0) + 2*pad_y;
    Size outSize = Size(oW, oH);
    // size of visible
    const Size visible = blockSize*sbin;

    // initialize historgram, norm, output feature matrices
    Mat histM = Mat::zeros(Size(blockSize.width*numOrient, blockSize.height), CV_64F);
    Mat normM = Mat::zeros(Size(blockSize.width, blockSize.height), CV_64F);
    featM = Mat::zeros(Size(outSize.width*dimHOG, outSize.height), CV_64F);

    // get the stride of each matrix
    const size_t imStride = imageM.step1();
    const size_t histStride = histM.step1();
    const size_t normStride = normM.step1();
    const size_t featStride = featM.step1();

    // calculate the zero offset
    const double* im = imageM.ptr<double>(0);
    double* const hist = histM.ptr<double>(0);
    double* const norm = normM.ptr<double>(0);
    double* const feat = featM.ptr<double>(0);

    for (int y = 1; y < visible.height - 1; y++)
    {
        for (int x = 1; x < visible.width - 1; x++)
        {
            // OpenCV uses an interleaved format: BGR-BGR-BGR
            const double* s = im + 3*min(x, imageM.cols-2) + min(y, imageM.rows-2)*imStride;

            // blue image channel
            double dyb = *(s+imStride) - *(s-imStride);
            double dxb = *(s+3) - *(s-3);
            double vb = dxb*dxb + dyb*dyb;

            // green image channel
            s += 1;
            double dyg = *(s+imStride) - *(s-imStride);
            double dxg = *(s+3) - *(s-3);
            double vg = dxg*dxg + dyg*dyg;

            // red image channel
            s += 1;
            double dy = *(s+imStride) - *(s-imStride);
            double dx = *(s+3) - *(s-3);
            double v = dx*dx + dy*dy;

            // pick the channel with the strongest gradient
            if (vg > v) { v = vg; dx = dxg; dy = dyg; }
            if (vb > v) { v = vb; dx = dxb; dy = dyb; }

            // snap to one of the 18 orientations
            double best_dot = 0;
            int best_o = 0;
            for (int o = 0; o < (int)numOrient/2; o++)
            {
                double dot =  uu[o]*dx + vv[o]*dy;
                if (dot > best_dot)
                {
                    best_dot = dot;
                    best_o = o;
                }
                else if (-dot > best_dot)
                {
                    best_dot = -dot;
                    best_o = o + (int)(numOrient/2);
                }
            }

            // add to 4 historgrams around pixel using bilinear interpolation
            double yp =  ((double)y+0.5)/(double)sbin - 0.5;
            double xp =  ((double)x+0.5)/(double)sbin - 0.5;
            int iyp = (int)floor(yp);
            int ixp = (int)floor(xp);
            double vy0 = yp - iyp;
            double vx0 = xp - ixp;
            double vy1 = 1.0 - vy0;
            double vx1 = 1.0 - vx0;
            v = sqrt(v);

            // fill the value into the 4 neighborhood cells
            if (iyp >= 0 && ixp >= 0)
                *(hist + iyp*histStride + ixp*numOrient + best_o) += vy1*vx1*v;

            if (iyp >= 0 && ixp+1 < blockSize.width)
                *(hist + iyp*histStride + (ixp+1)*numOrient + best_o) += vx0*vy1*v;

            if (iyp+1 < blockSize.height && ixp >= 0)
                *(hist + (iyp+1)*histStride + ixp*numOrient + best_o) += vy0*vx1*v;

            if (iyp+1 < blockSize.height && ixp+1 < blockSize.width)
                *(hist + (iyp+1)*histStride + (ixp+1)*numOrient + best_o) += vy0*vx0*v;

        } // for y
    } // for x

    // compute the energy in each block by summing over orientation
    for (int y = 0; y < blockSize.height; y++)
    {
        const double* src = hist + y*histStride;
        double* dst = norm + y*normStride;
        double const* const dst_end = dst + blockSize.width;
        // for each cell
        while (dst < dst_end)
        {
            *dst = 0;
            for (int o = 0; o < (int)(numOrient/2); o++)
            {
                *dst += (*src + *(src + numOrient/2))*
                    (*src + *(src + numOrient/2));
                src++;
            }
            dst++;
            src += numOrient/2;
        }
    }

    // compute the features
    for (int y = pad_y; y < outSize.height - pad_y; y++)
    {
        for (int x = pad_x; x < outSize.width - pad_x; x++)
        {
            double* dst = feat + y*featStride + x*dimHOG;
            double* p, n1, n2, n3, n4;
            const double* src;

            p = norm + (y - pad_y + 1)*normStride + (x - pad_x + 1);
            n1 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);
            p = norm + (y - pad_y)*normStride + (x - pad_x + 1);
            n2 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);
            p = norm + (y- pad_y + 1)*normStride + x - pad_x;
            n3 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);
            p = norm + (y - pad_y)*normStride + x - pad_x;
            n4 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);

            double t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;

            // contrast-sesitive features
            src = hist + (y - pad_y + 1)*histStride + (x - pad_x + 1)*numOrient;
            for (int o = 0; o < numOrient; o++)
            {
                double val = *src;
                double h1 = min(val*n1, 0.2);
                double h2 = min(val*n2, 0.2);
                double h3 = min(val*n3, 0.2);
                double h4 = min(val*n4, 0.2);
                *(dst++) = 0.5 * (h1 + h2 + h3 + h4);
                src++;
                t1 += h1;
                t2 += h2;
                t3 += h3;
                t4 += h4;
            }

            // contrast-insensitive features
            src =  hist + (y - pad_y + 1)*histStride + (x - pad_x + 1)*numOrient;
            for (int o = 0; o < numOrient/2; o++)
            {
                double sum = *src + *(src + numOrient/2);
                double h1 = min(sum * n1, 0.2);
                double h2 = min(sum * n2, 0.2);
                double h3 = min(sum * n3, 0.2);
                double h4 = min(sum * n4, 0.2);
                *(dst++) = 0.5 * (h1 + h2 + h3 + h4);
                src++;
            }

            // texture features
            *(dst++) = 0.2357 * t1;
            *(dst++) = 0.2357 * t2;
            *(dst++) = 0.2357 * t3;
            *(dst++) = 0.2357 * t4;

            // truncation feature
            *dst = 0;
        }// for x
    }// for y

    // Truncation features
    for (int m = 0; m < featM.rows; m++)
    {
        for (int n = 0; n < featM.cols; n += dimHOG)
        {
            if (m > pad_y - 1 && m < featM.rows - pad_y && n > pad_x*dimHOG - 1 && n < featM.cols - pad_x*dimHOG)
                continue;

            featM.at<double>(m, n + dimHOG - 1) = 1;
        } // for x
    }// for y
}

void Feature::projectFeaturePyramid(const Mat &pcaCoeff, const std::vector< Mat > &pyramid, std::vector< Mat > &projPyramid)
{
    CV_Assert(dimHOG == pcaCoeff.rows);
    dimPCA = pcaCoeff.cols;

    projPyramid.resize(pyramid.size());

    // loop for each level of the pyramid
    for (unsigned int i = 0; i < pyramid.size(); i++)
    {
        Mat orgM = pyramid[i];
        // note that the features are stored in 32-32-32
        int width = orgM.cols/dimHOG;
        int height = orgM.rows;
        // initialize the project feature matrix
        Mat projM = Mat::zeros(height, width*dimPCA, CV_64F);
        //get the pointer of the matrix
        double* const featOrg = orgM.ptr<double>(0);
        double* const featProj = projM.ptr<double>(0);

        // get the stride of each matrix
        const size_t orgStride = orgM.step1();
        const size_t projStride = projM.step1();

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double* proj = featProj + y*projStride + x*dimPCA;
                // for each pca dimension
                for (int c = 0; c < dimPCA; c++)
                {
                    double* org = featOrg + y*orgStride + x*dimHOG;
                    // dot product 32d HOG feature with the coefficient vector
                    for (int r = 0; r < dimHOG; r++)
                    {
                        *proj += *org * pcaCoeff.at<double>(r, c);
                        org++;
                    }
                    proj++;
                }

            } // for x
        } // for y
        projPyramid[i] = projM;
    } // for each level of the pyramid
}

void Feature::computeLocationFeatures(const int numLevels, Mat &locFeature)
{
    locFeature = Mat::zeros(Size(numLevels, 3), CV_64F);

    int b = 0;
    int e = min(numLevels, params.interval);

    for (int x = b; x < e; x++)
        locFeature.at<double>(0, x) = 1;

    b = e;
    e = min(numLevels, 2*e);

    for (int x = b; x < e; x++)
        locFeature.at<double>(1, x) = 1;

    b = e;
    e = min(numLevels, 3*e);

    for (int x = b; x < e; x++)
        locFeature.at<double>(2, x) = 1;
}
} // namespace dpm
} // namespace cv
