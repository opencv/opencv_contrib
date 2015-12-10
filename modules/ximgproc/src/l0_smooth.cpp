/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  *Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

#include "precomp.hpp"
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

namespace
{

    class ParallelDft : public ParallelLoopBody
    {
    private:
        vector<Mat> src_;
    public:
        ParallelDft(vector<Mat> &s)
        {
            src_ = s;
        }
        void operator() (const Range& range) const
        {
            for (int i = range.start; i != range.end; i++)
            {
                dft(src_[i], src_[i]);
            }
        }
    };

    class ParallelIdft : public ParallelLoopBody
    {
    private:
        vector<Mat> src_;
    public:
        ParallelIdft(vector<Mat> &s)
        {
            src_ = s;
        }
        void operator() (const Range& range) const
        {
            for (int i = range.start; i != range.end; i++)
            {
                idft(src_[i], src_[i],DFT_SCALE);
            }
        }
    };

    class ParallelDivComplexByReal : public ParallelLoopBody
    {
    private:
        vector<Mat> numer_;
        vector<Mat> denom_;
        vector<Mat> dst_;

    public:
        ParallelDivComplexByReal(vector<Mat> &numer, vector<Mat> &denom, vector<Mat> &dst)
        {
            numer_ = numer;
            denom_ = denom;
            dst_ = dst;
        }
        void operator() (const Range& range) const
        {
            for (int i = range.start; i != range.end; i++)
            {
                Mat aPanels[2];
                Mat bPanels[2];
                split(numer_[i], aPanels);
                split(denom_[i], bPanels);

                Mat realPart;
                Mat imaginaryPart;

                divide(aPanels[0], denom_[i], realPart);
                divide(aPanels[1], denom_[i], imaginaryPart);

                aPanels[0] = realPart;
                aPanels[1] = imaginaryPart;

                merge(aPanels, 2, dst_[i]);
            }
        }
    };


    void shift(InputArray src, OutputArray dst, int shift_x, int shift_y)
    {
        Mat S = src.getMat();
        Mat D = dst.getMat();

        if(S.data == D.data)
        {
            S = S.clone();
        }

        D.create(S.size(), S.type());

        Mat s0(S, Rect(0, 0, S.cols - shift_x, S.rows - shift_y));
        Mat s1(S, Rect(S.cols - shift_x, 0, shift_x, S.rows - shift_y));
        Mat s2(S, Rect(0, S.rows - shift_y, S.cols-shift_x, shift_y));
        Mat s3(S, Rect(S.cols - shift_x, S.rows- shift_y, shift_x, shift_y));

        Mat d0(D, Rect(shift_x, shift_y, S.cols - shift_x, S.rows - shift_y));
        Mat d1(D, Rect(0, shift_y, shift_x, S.rows - shift_y));
        Mat d2(D, Rect(shift_x, 0, S.cols-shift_x, shift_y));
        Mat d3(D, Rect(0,0,shift_x, shift_y));

        s0.copyTo(d0);
        s1.copyTo(d1);
        s2.copyTo(d2);
        s3.copyTo(d3);
    }

    // dft after padding imaginary
    void fft(InputArray src, OutputArray dst)
    {
        Mat S = src.getMat();
        Mat planes[] = {S.clone(), Mat::zeros(S.size(), S.type())};
        Mat x;
        merge(planes, 2, dst);

        // compute the result
        dft(dst, dst);
    }

    void psf2otf(InputArray src, OutputArray dst, int height, int width)
    {
        Mat S = src.getMat();
        Mat D = dst.getMat();

        Mat padded;

        if(S.data == D.data){
            S = S.clone();
        }

        // add padding
        copyMakeBorder(S, padded, 0, height - S.rows, 0, width - S.cols,
            BORDER_CONSTANT, Scalar::all(0));

        shift(padded, padded, width - S.cols / 2, height - S.rows / 2);

        // convert to frequency domain
        fft(padded, dst);
    }

    void dftMultiChannel(InputArray src, vector<Mat> &dst)
    {
        Mat S = src.getMat();

        split(S, dst);

        for(int i = 0; i < S.channels(); i++){
            Mat planes[] = {dst[i].clone(), Mat::zeros(dst[i].size(), dst[i].type())};
            merge(planes, 2, dst[i]);
        }

        parallel_for_(cv::Range(0,S.channels()), ParallelDft(dst));

    }

    void idftMultiChannel(const vector<Mat> &src, OutputArray dst)
    {
        vector<Mat> channels(src);

        parallel_for_(Range(0, int(src.size())), ParallelIdft(channels));

        for(int i = 0; unsigned(i) < src.size(); i++){
            Mat panels[2];
            split(channels[i], panels);
            channels[i] = panels[0];
        }

        Mat D;
        merge(channels, D);
        D.copyTo(dst);
    }

    void addComplex(InputArray aSrc, int bSrc, OutputArray dst)
    {
        Mat panels[2];
        split(aSrc.getMat(), panels);
        panels[0] = panels[0] + bSrc;
        merge(panels, 2, dst);
    }

    void divComplexByRealMultiChannel(vector<Mat> &numer,
        vector<Mat> &denom, vector<Mat> &dst)
    {

        for(int i = 0; unsigned(i) < numer.size(); i++)
        {
            dst[i].create(numer[i].size(), numer[i].type());
        }
        parallel_for_(Range(0, int(numer.size())), ParallelDivComplexByReal(numer, denom, dst));

    }

        // power of 2 of the absolute value of the complex
    Mat pow2absComplex(InputArray src)
    {
        Mat S = src.getMat();

        Mat sPanels[2];
        split(S, sPanels);

        Mat mag;
        magnitude(sPanels[0], sPanels[1], mag);
        pow(mag, 2, mag);

        return mag;
    }
}

namespace cv
{
    namespace ximgproc
    {

        void l0Smooth(InputArray src, OutputArray dst, double lambda, double kappa)
        {
            Mat S = src.getMat();

            CV_Assert(!S.empty());
            CV_Assert(S.depth() == CV_8U || S.depth() == CV_16U
            || S.depth() == CV_32F || S.depth() == CV_64F);

            dst.create(src.size(), src.type());

            if(S.data == dst.getMat().data)
            {
                S = S.clone();
            }

            if(S.depth() == CV_8U)
            {
                S.convertTo(S, CV_32F, 1/255.0f);
            }
            else if(S.depth() == CV_16U)
            {
                S.convertTo(S, CV_32F, 1/65535.0f);
            }
            else if(S.depth() == CV_64F)
            {
                S.convertTo(S, CV_32F);
            }

            const double betaMax = 100000;

            // gradient operators in frequency domain
            Mat otfFx, otfFy;
            float kernel[2] = {-1, 1};
            float kernel_inv[2] = {1,-1};
            psf2otf(Mat(1,2,CV_32FC1, kernel_inv), otfFx, S.rows, S.cols);
            psf2otf(Mat(2,1,CV_32FC1, kernel_inv), otfFy, S.rows, S.cols);

            vector<Mat> denomConst;
            Mat tmp = pow2absComplex(otfFx) + pow2absComplex(otfFy);

            for(int i = 0; i < S.channels(); i++)
            {
                denomConst.push_back(tmp);
            }

            // input image in frequency domain
            vector<Mat> numerConst;
            dftMultiChannel(S, numerConst);
            /*********************************
            * solver
            *********************************/
            double beta = 2 * lambda;
            while(beta < betaMax){
                // h, v subproblem
                Mat h, v;

                filter2D(S, h, -1, Mat(1, 2, CV_32FC1, kernel), Point(0, 0),
                0, BORDER_REPLICATE);
                filter2D(S, v, -1, Mat(2, 1, CV_32FC1, kernel), Point(0, 0),
                0, BORDER_REPLICATE);

                Mat hvMag = h.mul(h) + v.mul(v);

                Mat mask;
                if(S.channels() == 1)
                {
                    threshold(hvMag, mask, lambda/beta, 1, THRESH_BINARY);
                }
                else if(S.channels() > 1)
                {
                    vector<Mat> channels(S.channels());
                    split(hvMag, channels);
                    hvMag = channels[0];

                    for(int i = 1; i < S.channels(); i++)
                    {
                        hvMag = hvMag + channels[i];
                    }

                    threshold(hvMag, mask, lambda/beta, 1, THRESH_BINARY);

                    Mat in[] = {mask, mask, mask};
                    merge(in, 3, mask);
                }

                h = h.mul(mask);
                v = v.mul(mask);

                // S subproblem
                vector<Mat> denom(S.channels());
                for(int i = 0; i < S.channels(); i++)
                {
                    denom[i] = beta * denomConst[i] + 1;
                }

                Mat hGrad, vGrad;
                filter2D(h, hGrad, -1, Mat(1, 2, CV_32FC1, kernel_inv));
                filter2D(v, vGrad, -1, Mat(2, 1, CV_32FC1, kernel_inv));

                vector<Mat> hvGradFreq;
                dftMultiChannel(hGrad+vGrad, hvGradFreq);

                vector<Mat> numer(S.channels());
                for(int i = 0; i < S.channels(); i++)
                {
                    numer[i] = numerConst[i] + hvGradFreq[i] * beta;
                }

                vector<Mat> sFreq(S.channels());
                divComplexByRealMultiChannel(numer, denom, sFreq);

                idftMultiChannel(sFreq, S);

                beta = beta * kappa;
            }

            Mat D = dst.getMat();
            if(D.depth() == CV_8U)
            {
                S.convertTo(D, CV_8U, 255);
            }
            else if(D.depth() == CV_16U)
            {
                S.convertTo(D, CV_16U, 65535);
            }
            else if(D.depth() == CV_64F)
            {
                S.convertTo(D, CV_64F);
            }
            else
            {
                S.copyTo(D);
            }
        }
    }
}
