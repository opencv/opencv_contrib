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
 *  * Redistributions of source code must retain the above copyright notice,
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
        vector<Mat>& src_;
        ParallelDft& operator=(const ParallelDft&);
    public:
        ParallelDft(vector<Mat>& s) : src_(s) {}
        void operator()(const Range& range) const CV_OVERRIDE
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
        vector<Mat>& src_;
        ParallelIdft& operator=(const ParallelIdft&);
    public:
        ParallelIdft(vector<Mat>& s) : src_(s) {}
        void operator()(const Range& range) const CV_OVERRIDE
        {
            for (int i = range.start; i != range.end; i++)
            {
                idft(src_[i], src_[i], DFT_SCALE);
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
        dst.create(S.size(), CV_32FC2);
        Mat D = dst.getMat();
        for (int r = 0; r < S.rows; ++r)
        {
            const float* sRow = S.ptr<float>(r);
            Vec2f* dRow = D.ptr<Vec2f>(r);
            for (int c = 0; c < S.cols; ++c)
                dRow[c] = Vec2f(sRow[c], 0.0f);
        }
        dft(D, D);
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

    void packRealToComplexMultiChannel(const Mat& S, vector<Mat>& dst)
    {
        const int cn = S.channels();
        dst.resize(cn);
        for (int c = 0; c < cn; ++c)
            dst[c].create(S.size(), CV_32FC2);

        for (int r = 0; r < S.rows; ++r)
        {
            const float* sRow = S.ptr<float>(r);
            if (cn == 1)
            {
                Vec2f* d0 = dst[0].ptr<Vec2f>(r);
                for (int x = 0; x < S.cols; ++x)
                    d0[x] = Vec2f(sRow[x], 0.0f);
            }
            else if (cn == 3)
            {
                Vec2f* d0 = dst[0].ptr<Vec2f>(r);
                Vec2f* d1 = dst[1].ptr<Vec2f>(r);
                Vec2f* d2 = dst[2].ptr<Vec2f>(r);
                for (int x = 0; x < S.cols; ++x)
                {
                    d0[x] = Vec2f(sRow[3 * x], 0.0f);
                    d1[x] = Vec2f(sRow[3 * x + 1], 0.0f);
                    d2[x] = Vec2f(sRow[3 * x + 2], 0.0f);
                }
            }
            else
            {
                for (int c = 0; c < cn; ++c)
                {
                    Vec2f* dc = dst[c].ptr<Vec2f>(r);
                    for (int x = 0; x < S.cols; ++x)
                        dc[x] = Vec2f(sRow[cn * x + c], 0.0f);
                }
            }
        }
    }

    void addGradAndPackComplexMultiChannel(const Mat& hGrad, const Mat& vGrad, vector<Mat>& dst)
    {
        const int cn = hGrad.channels();
        for (int r = 0; r < hGrad.rows; ++r)
        {
            const float* hgRow = hGrad.ptr<float>(r);
            const float* vgRow = vGrad.ptr<float>(r);
            if (cn == 1)
            {
                Vec2f* d0 = dst[0].ptr<Vec2f>(r);
                for (int x = 0; x < hGrad.cols; ++x)
                    d0[x] = Vec2f(hgRow[x] + vgRow[x], 0.0f);
            }
            else if (cn == 3)
            {
                Vec2f* d0 = dst[0].ptr<Vec2f>(r);
                Vec2f* d1 = dst[1].ptr<Vec2f>(r);
                Vec2f* d2 = dst[2].ptr<Vec2f>(r);
                for (int x = 0; x < hGrad.cols; ++x)
                {
                    d0[x] = Vec2f(hgRow[3 * x] + vgRow[3 * x], 0.0f);
                    d1[x] = Vec2f(hgRow[3 * x + 1] + vgRow[3 * x + 1], 0.0f);
                    d2[x] = Vec2f(hgRow[3 * x + 2] + vgRow[3 * x + 2], 0.0f);
                }
            }
            else
            {
                for (int c = 0; c < cn; ++c)
                {
                    Vec2f* dc = dst[c].ptr<Vec2f>(r);
                    for (int x = 0; x < hGrad.cols; ++x)
                        dc[x] = Vec2f(hgRow[cn * x + c] + vgRow[cn * x + c], 0.0f);
                }
            }
        }
    }

    void unpackRealFromComplexMultiChannel(const vector<Mat>& src, Mat& S)
    {
        const int cn = S.channels();
        for (int r = 0; r < S.rows; ++r)
        {
            float* sRow = S.ptr<float>(r);
            if (cn == 1)
            {
                const Vec2f* c0 = src[0].ptr<Vec2f>(r);
                for (int x = 0; x < S.cols; ++x)
                    sRow[x] = c0[x][0];
            }
            else if (cn == 3)
            {
                const Vec2f* c0 = src[0].ptr<Vec2f>(r);
                const Vec2f* c1 = src[1].ptr<Vec2f>(r);
                const Vec2f* c2 = src[2].ptr<Vec2f>(r);
                for (int x = 0; x < S.cols; ++x)
                {
                    sRow[3 * x] = c0[x][0];
                    sRow[3 * x + 1] = c1[x][0];
                    sRow[3 * x + 2] = c2[x][0];
                }
            }
            else
            {
                for (int c = 0; c < cn; ++c)
                {
                    const Vec2f* cc = src[c].ptr<Vec2f>(r);
                    for (int x = 0; x < S.cols; ++x)
                        sRow[cn * x + c] = cc[x][0];
                }
            }
        }
    }

    void accumPow2AbsComplex(const Mat& S, Mat& acc, bool initZero)
    {
        acc.create(S.size(), CV_32FC1);
        for (int r = 0; r < S.rows; ++r)
        {
            const Vec2f* sRow = S.ptr<Vec2f>(r);
            float* aRow = acc.ptr<float>(r);
            if (initZero)
            {
                for (int c = 0; c < S.cols; ++c)
                    aRow[c] = sRow[c][0] * sRow[c][0] + sRow[c][1] * sRow[c][1];
            }
            else
            {
                for (int c = 0; c < S.cols; ++c)
                    aRow[c] += sRow[c][0] * sRow[c][0] + sRow[c][1] * sRow[c][1];
            }
        }
    }

    void hardThresholdGradientsInPlace(Mat& h, Mat& v, float thresh)
    {
        const int cn = h.channels();
        const int cols = h.cols;
        for (int r = 0; r < h.rows; ++r)
        {
            float* hRow = h.ptr<float>(r);
            float* vRow = v.ptr<float>(r);
            if (cn == 1)
            {
                for (int x = 0; x < cols; ++x)
                {
                    const float mag = hRow[x] * hRow[x] + vRow[x] * vRow[x];
                    if (!(mag > thresh))
                    {
                        hRow[x] = 0.0f;
                        vRow[x] = 0.0f;
                    }
                }
            }
            else if (cn == 3)
            {
                for (int x = 0; x < cols; ++x)
                {
                    const int idx = 3 * x;
                    const float mag =
                        (hRow[idx] * hRow[idx] + vRow[idx] * vRow[idx]) +
                        (hRow[idx + 1] * hRow[idx + 1] + vRow[idx + 1] * vRow[idx + 1]) +
                        (hRow[idx + 2] * hRow[idx + 2] + vRow[idx + 2] * vRow[idx + 2]);
                    if (!(mag > thresh))
                    {
                        hRow[idx] = hRow[idx + 1] = hRow[idx + 2] = 0.0f;
                        vRow[idx] = vRow[idx + 1] = vRow[idx + 2] = 0.0f;
                    }
                }
            }
            else
            {
                for (int x = 0; x < cols; ++x)
                {
                    const int base = cn * x;
                    float mag = 0.0f;
                    for (int c = 0; c < cn; ++c)
                        mag += hRow[base + c] * hRow[base + c] + vRow[base + c] * vRow[base + c];
                    if (!(mag > thresh))
                    {
                        for (int c = 0; c < cn; ++c)
                        {
                            hRow[base + c] = 0.0f;
                            vRow[base + c] = 0.0f;
                        }
                    }
                }
            }
        }
    }

    struct ParallelSpectrumSolve : public ParallelLoopBody
    {
        const vector<Mat>& numerConst;
        const Mat& denomConst;
        vector<Mat>& workFreq;
        float beta;

        ParallelSpectrumSolve(const vector<Mat>& _nc, const Mat& _dc, vector<Mat>& _wf, float _beta)
            : numerConst(_nc), denomConst(_dc), workFreq(_wf), beta(_beta) {}

        void operator()(const Range& range) const CV_OVERRIDE
        {
            const int cn = (int)workFreq.size();
            const int cols = denomConst.cols;
            for (int r = range.start; r < range.end; ++r)
            {
                const float* dcRow = denomConst.ptr<float>(r);
                for (int c = 0; c < cn; ++c)
                {
                    const Vec2f* ncRow = numerConst[c].ptr<Vec2f>(r);
                    Vec2f* wfRow = workFreq[c].ptr<Vec2f>(r);
                    for (int x = 0; x < cols; ++x)
                    {
                        const float denom = beta * dcRow[x] + 1.0f;
                        wfRow[x][0] = (ncRow[x][0] + wfRow[x][0] * beta) / denom;
                        wfRow[x][1] = (ncRow[x][1] + wfRow[x][1] * beta) / denom;
                    }
                }
            }
        }

    private:
        ParallelSpectrumSolve& operator=(const ParallelSpectrumSolve&);
    };
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
            CV_Assert(lambda > 0.0);
            CV_Assert(kappa > 1.0);

            dst.create(src.size(), src.type());

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
            else
            {
                S = S.clone();
            }

            const double betaMax = 100000;
            const int cn = S.channels();
            const int nstripes = std::max(1, getNumThreads()) * 4;

            // gradient operators in frequency domain (transient: released immediately after denomConst)
            float kernel[2] = {-1, 1};
            float kernel_inv[2] = {1, -1};
            Mat denomConst;
            {
                Mat otf;
                psf2otf(Mat(1, 2, CV_32FC1, kernel_inv), otf, S.rows, S.cols);
                accumPow2AbsComplex(otf, denomConst, true);
                psf2otf(Mat(2, 1, CV_32FC1, kernel_inv), otf, S.rows, S.cols);
                accumPow2AbsComplex(otf, denomConst, false);
            }

            // input image in frequency domain
            vector<Mat> numerConst;
            packRealToComplexMultiChannel(S, numerConst);
            parallel_for_(cv::Range(0, cn), ParallelDft(numerConst));

            // preallocate reusable solver workspace once outside beta loop
            Mat h(S.size(), S.type());
            Mat v(S.size(), S.type());
            vector<Mat> workFreq(cn);
            for (int c = 0; c < cn; ++c)
                workFreq[c].create(S.size(), CV_32FC2);

            const Mat kFwdH(1, 2, CV_32FC1, kernel);
            const Mat kFwdV(2, 1, CV_32FC1, kernel);
            const Mat kInvH(1, 2, CV_32FC1, kernel_inv);
            const Mat kInvV(2, 1, CV_32FC1, kernel_inv);

            /*********************************
            * solver
            *********************************/
            double beta = 2 * lambda;
            while(beta < betaMax)
            {
                // h, v subproblem
                filter2D(S, h, -1, kFwdH, Point(0, 0), 0, BORDER_REPLICATE);
                filter2D(S, v, -1, kFwdV, Point(0, 0), 0, BORDER_REPLICATE);

                hardThresholdGradientsInPlace(h, v, static_cast<float>(lambda / beta));

                // S subproblem: S is dead until IDFT unpack, so reuse S for hGrad and h for vGrad
                filter2D(h, S, -1, kInvH);
                filter2D(v, h, -1, kInvV);

                addGradAndPackComplexMultiChannel(S, h, workFreq);
                parallel_for_(cv::Range(0, cn), ParallelDft(workFreq));
                parallel_for_(cv::Range(0, S.rows),
                              ParallelSpectrumSolve(numerConst, denomConst, workFreq, static_cast<float>(beta)),
                              nstripes);
                parallel_for_(cv::Range(0, cn), ParallelIdft(workFreq));
                unpackRealFromComplexMultiChannel(workFreq, S);

                beta = beta * kappa;
            }

            h.release();
            v.release();
            workFreq.clear();
            numerConst.clear();
            denomConst.release();

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
