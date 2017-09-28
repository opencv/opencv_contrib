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

/*
 * The MIT License(MIT)
 * 
 * Copyright(c) 2013 Vladislav Vinogradov
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files(the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and / or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions :
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "test_precomp.hpp"
#include <opencv2/core/private.hpp>
#include <cmath>

namespace
{

    using std::numeric_limits;
    using namespace cv;
    using namespace cv::ximgproc;

    struct Buf
    {
        Mat_<Point3f> eta_1;
        Mat_<uchar> cluster_1;

        Mat_<Point3f> tilde_dst;
        Mat_<float> alpha;
        Mat_<Point3f> diff;
        Mat_<Point3f> dst;

        Mat_<float> V;

        Mat_<Point3f> dIcdx;
        Mat_<Point3f> dIcdy;
        Mat_<float> dIdx;
        Mat_<float> dIdy;
        Mat_<float> dHdx;
        Mat_<float> dVdy;

        Mat_<float> t;

        Mat_<float> theta_masked;
        Mat_<Point3f> mul;
        Mat_<Point3f> numerator;
        Mat_<float> denominator;
        Mat_<Point3f> numerator_filtered;
        Mat_<float> denominator_filtered;

        Mat_<Point3f> X;
        Mat_<Point3f> eta_k_small;
        Mat_<Point3f> eta_k_big;
        Mat_<Point3f> X_squared;
        Mat_<float> pixel_dist_to_manifold_squared;
        Mat_<float> gaussian_distance_weights;
        Mat_<Point3f> Psi_splat;
        Mat_<Vec4f> Psi_splat_joined;
        Mat_<Vec4f> Psi_splat_joined_resized;
        Mat_<Vec4f> blurred_projected_values;
        Mat_<Point3f> w_ki_Psi_blur;
        Mat_<float> w_ki_Psi_blur_0;
        Mat_<Point3f> w_ki_Psi_blur_resized;
        Mat_<float> w_ki_Psi_blur_0_resized;
        Mat_<float> rand_vec;
        Mat_<float> v1;
        Mat_<float> Nx_v1_mult;
        Mat_<float> theta;

        std::vector<Mat_<Point3f> > eta_minus;
        std::vector<Mat_<uchar> > cluster_minus;
        std::vector<Mat_<Point3f> > eta_plus;
        std::vector<Mat_<uchar> > cluster_plus;

        void release();
    };

    void Buf::release()
    {
        eta_1.release();
        cluster_1.release();

        tilde_dst.release();
        alpha.release();
        diff.release();
        dst.release();

        V.release();

        dIcdx.release();
        dIcdy.release();
        dIdx.release();
        dIdy.release();
        dHdx.release();
        dVdy.release();

        t.release();

        theta_masked.release();
        mul.release();
        numerator.release();
        denominator.release();
        numerator_filtered.release();
        denominator_filtered.release();

        X.release();
        eta_k_small.release();
        eta_k_big.release();
        X_squared.release();
        pixel_dist_to_manifold_squared.release();
        gaussian_distance_weights.release();
         Psi_splat.release();
        Psi_splat_joined.release();
        Psi_splat_joined_resized.release();
        blurred_projected_values.release();
        w_ki_Psi_blur.release();
        w_ki_Psi_blur_0.release();
        w_ki_Psi_blur_resized.release();
        w_ki_Psi_blur_0_resized.release();
        rand_vec.release();
        v1.release();
        Nx_v1_mult.release();
        theta.release();

        eta_minus.clear();
        cluster_minus.clear();
        eta_plus.clear();
        cluster_plus.clear();
    }

    class AdaptiveManifoldFilterRefImpl : public AdaptiveManifoldFilter
    {
    public:
        AdaptiveManifoldFilterRefImpl();

        void filter(InputArray src, OutputArray dst, InputArray joint);

        void collectGarbage();

        CV_IMPL_PROPERTY(double, SigmaS, sigma_s_)
        CV_IMPL_PROPERTY(double, SigmaR, sigma_r_)
        CV_IMPL_PROPERTY(int, TreeHeight, tree_height_)
        CV_IMPL_PROPERTY(int, PCAIterations, num_pca_iterations_)
        CV_IMPL_PROPERTY(bool, AdjustOutliers, adjust_outliers_)
        CV_IMPL_PROPERTY(bool, UseRNG, useRNG)

    protected:
        bool adjust_outliers_;
        double sigma_s_;
        double sigma_r_;
        int tree_height_;
        int num_pca_iterations_;
        bool useRNG;

    private:
        void buildManifoldsAndPerformFiltering(const Mat_<Point3f>& eta_k, const Mat_<uchar>& cluster_k, int current_tree_level);

        Buf buf_;

        Mat_<Point3f> src_f_;
        Mat_<Point3f> src_joint_f_;

        Mat_<Point3f> sum_w_ki_Psi_blur_;
        Mat_<float> sum_w_ki_Psi_blur_0_;

        Mat_<float> min_pixel_dist_to_manifold_squared_;

        RNG rng_;

        int cur_tree_height_;
        float sigma_r_over_sqrt_2_;
    };

    AdaptiveManifoldFilterRefImpl::AdaptiveManifoldFilterRefImpl()
    {
        sigma_s_ = 16.0;
        sigma_r_ = 0.2;
        tree_height_ = -1;
        num_pca_iterations_ = 1;
        adjust_outliers_ = false;
        useRNG = true;
    }

    void AdaptiveManifoldFilterRefImpl::collectGarbage()
    {
        buf_.release();

        src_f_.release();
        src_joint_f_.release();

        sum_w_ki_Psi_blur_.release();
        sum_w_ki_Psi_blur_0_.release();

        min_pixel_dist_to_manifold_squared_.release();
    }

    inline double Log2(double n)
    {
        return log(n) / log(2.0);
    }

    inline int computeManifoldTreeHeight(double sigma_s, double sigma_r)
    {
        const double Hs = floor(Log2(sigma_s)) - 1.0;
        const double Lr = 1.0 - sigma_r;
        return max(2, static_cast<int>(ceil(Hs * Lr)));
    }
/*
    void ensureSizeIsEnough(int rows, int cols, int type, Mat& m)
    {
        if (m.empty() || m.type() != type || m.data != m.datastart)
            m.create(rows, cols, type);
        else
        {
            const size_t esz = m.elemSize();
            const ptrdiff_t delta2 = m.dataend - m.datastart;

            const size_t minstep = m.cols * esz;

            Size wholeSize;
            wholeSize.height = std::max(static_cast<int>((delta2 - minstep) / m.step + 1), m.rows);
            wholeSize.width = std::max(static_cast<int>((delta2 - m.step * (wholeSize.height - 1)) / esz), m.cols);

            if (wholeSize.height < rows || wholeSize.width < cols)
                m.create(rows, cols, type);
            else
            {
                m.cols = cols;
                m.rows = rows;
            }
        }
    }

    inline void ensureSizeIsEnough(Size size, int type, Mat& m)
    {
        ensureSizeIsEnough(size.height, size.width, type, m);
    }
*/
    template <typename T>
    inline void ensureSizeIsEnough(int rows, int cols, Mat_<T>& m)
    {
        if (m.empty() || m.data != m.datastart)
            m.create(rows, cols);
        else
        {
            const size_t esz = m.elemSize();
            const ptrdiff_t delta2 = m.dataend - m.datastart;

            const size_t minstep = m.cols * esz;

            Size wholeSize;
            wholeSize.height = std::max(static_cast<int>((delta2 - minstep) / m.step + 1), m.rows);
            wholeSize.width = std::max(static_cast<int>((delta2 - m.step * (wholeSize.height - 1)) / esz), m.cols);

            if (wholeSize.height < rows || wholeSize.width < cols)
                m.create(rows, cols);
            else
            {
                m.cols = cols;
                m.rows = rows;
            }
        }
    }

    template <typename T>
    inline void ensureSizeIsEnough(Size size, Mat_<T>& m)
    {
        ensureSizeIsEnough(size.height, size.width, m);
    }

    template <typename T>
    void h_filter(const Mat_<T>& src, Mat_<T>& dst, float sigma)
    {
        CV_DbgAssert( src.depth() == CV_32F );

        const float a = exp(-sqrt(2.0f) / sigma);

        ensureSizeIsEnough(src.size(), dst);

        for (int y = 0; y < src.rows; ++y)
        {
            const T* src_row = src[y];
            T* dst_row = dst[y];

            dst_row[0] = src_row[0];
            for (int x = 1; x < src.cols; ++x)
            {
                //dst_row[x] = src_row[x] + a * (src_row[x - 1] - src_row[x]);
                dst_row[x] = src_row[x] + a * (dst_row[x - 1] - src_row[x]); //!!!
            }
            for (int x = src.cols - 2; x >= 0; --x)
            {
                dst_row[x] = dst_row[x] + a * (dst_row[x + 1] - dst_row[x]);
            }
        }

        for (int y = 1; y < src.rows; ++y)
        {
            T* dst_cur_row = dst[y];
            T* dst_prev_row = dst[y - 1];

            for (int x = 0; x < src.cols; ++x)
            {
                dst_cur_row[x] = dst_cur_row[x] + a * (dst_prev_row[x] - dst_cur_row[x]);
            }
        }
        for (int y = src.rows - 2; y >= 0; --y)
        {
            T* dst_cur_row = dst[y];
            T* dst_prev_row = dst[y + 1];

            for (int x = 0; x < src.cols; ++x)
            {
                dst_cur_row[x] = dst_cur_row[x] + a * (dst_prev_row[x] - dst_cur_row[x]);
            }
        }
    }

    template <typename T>
    void rdivide(const Mat_<T>& a, const Mat_<float>& b, Mat_<T>& dst)
    {
        CV_DbgAssert( a.depth() == CV_32F );
        CV_DbgAssert( a.size() == b.size() );

        ensureSizeIsEnough(a.size(), dst);
        dst.setTo(0);

        for (int y = 0; y < a.rows; ++y)
        {
            const T* a_row = a[y];
            const float* b_row = b[y];
            T* dst_row = dst[y];

            for (int x = 0; x < a.cols; ++x)
            {
                //if (b_row[x] > numeric_limits<float>::epsilon())
                    dst_row[x] = a_row[x] * (1.0f / b_row[x]);
            }
        }
    }

    template <typename T>
    void times(const Mat_<T>& a, const Mat_<float>& b, Mat_<T>& dst)
    {
        CV_DbgAssert( a.depth() == CV_32F );
        CV_DbgAssert( a.size() == b.size() );

        ensureSizeIsEnough(a.size(), dst);

        for (int y = 0; y < a.rows; ++y)
        {
            const T* a_row = a[y];
            const float* b_row = b[y];
            T* dst_row = dst[y];

            for (int x = 0; x < a.cols; ++x)
            {
                dst_row[x] = a_row[x] * b_row[x];
            }
        }
    }

    void AdaptiveManifoldFilterRefImpl::filter(InputArray _src, OutputArray _dst, InputArray _joint)
    {
        const Mat src = _src.getMat();
        const Mat src_joint = _joint.getMat();

        const Size srcSize = src.size();

        CV_Assert( src.type() == CV_8UC3 );
        CV_Assert( src_joint.empty() || (src_joint.type() == src.type() && src_joint.size() == srcSize) );

        ensureSizeIsEnough(srcSize, src_f_);
        src.convertTo(src_f_, src_f_.type(), 1.0 / 255.0);

        ensureSizeIsEnough(srcSize, sum_w_ki_Psi_blur_);
        sum_w_ki_Psi_blur_.setTo(Scalar::all(0));

        ensureSizeIsEnough(srcSize, sum_w_ki_Psi_blur_0_);
        sum_w_ki_Psi_blur_0_.setTo(Scalar::all(0));

        ensureSizeIsEnough(srcSize, min_pixel_dist_to_manifold_squared_);
        min_pixel_dist_to_manifold_squared_.setTo(Scalar::all(numeric_limits<float>::max()));

        // If the tree_height was not specified, compute it using Eq. (10) of our paper.
        cur_tree_height_ = tree_height_ > 0 ? tree_height_ : computeManifoldTreeHeight(sigma_s_, sigma_r_);

        // If no joint signal was specified, use the original signal
        ensureSizeIsEnough(srcSize, src_joint_f_);
        if (src_joint.empty())
            src_f_.copyTo(src_joint_f_);
        else
            src_joint.convertTo(src_joint_f_, src_joint_f_.type(), 1.0 / 255.0);

        // Use the center pixel as seed to random number generation.
        const double seedCoef = src_joint_f_(src_joint_f_.rows / 2, src_joint_f_.cols / 2).x;
        const uint64 baseCoef = numeric_limits<uint64>::max() / 0xFFFF;
        rng_.state = static_cast<uint64>(baseCoef*seedCoef);

        // Dividing the covariance matrix by 2 is equivalent to dividing the standard deviations by sqrt(2).
        sigma_r_over_sqrt_2_ = static_cast<float>(sigma_r_ / sqrt(2.0));

        // Algorithm 1, Step 1: compute the first manifold by low-pass filtering.
        h_filter(src_joint_f_, buf_.eta_1, static_cast<float>(sigma_s_));

        ensureSizeIsEnough(srcSize, buf_.cluster_1);
        buf_.cluster_1.setTo(Scalar::all(1));

        buf_.eta_minus.resize(cur_tree_height_);
        buf_.cluster_minus.resize(cur_tree_height_);
        buf_.eta_plus.resize(cur_tree_height_);
        buf_.cluster_plus.resize(cur_tree_height_);
        buildManifoldsAndPerformFiltering(buf_.eta_1, buf_.cluster_1, 1);

        // Compute the filter response by normalized convolution -- Eq. (4)
        rdivide(sum_w_ki_Psi_blur_, sum_w_ki_Psi_blur_0_, buf_.tilde_dst);

        if (!adjust_outliers_)
        {
            buf_.tilde_dst.convertTo(_dst, CV_8U, 255.0);
        }
        else
        {
            // Adjust the filter response for outlier pixels -- Eq. (10)
            ensureSizeIsEnough(srcSize, buf_.alpha);
            exp(min_pixel_dist_to_manifold_squared_ * (-0.5 / sigma_r_ / sigma_r_), buf_.alpha);

            ensureSizeIsEnough(srcSize, buf_.diff);
            subtract(buf_.tilde_dst, src_f_, buf_.diff);
            times(buf_.diff, buf_.alpha, buf_.diff);

            ensureSizeIsEnough(srcSize, buf_.dst);
            add(src_f_, buf_.diff, buf_.dst);

            buf_.dst.convertTo(_dst, CV_8U, 255.0);
        }
    }

    inline double floor_to_power_of_two(double r)
    {
        return pow(2.0, floor(Log2(r)));
    }

    void channelsSum(const Mat_<Point3f>& src, Mat_<float>& dst)
    {
        ensureSizeIsEnough(src.size(), dst);

        for (int y = 0; y < src.rows; ++y)
        {
            const Point3f* src_row = src[y];
            float* dst_row = dst[y];

            for (int x = 0; x < src.cols; ++x)
            {
                const Point3f src_val = src_row[x];
                dst_row[x] = src_val.x + src_val.y + src_val.z;
            }
        }
    }

    void phi(const Mat_<float>& src, Mat_<float>& dst, float sigma)
    {
        ensureSizeIsEnough(src.size(), dst);

        for (int y = 0; y < dst.rows; ++y)
        {
            const float* src_row = src[y];
            float* dst_row = dst[y];

            for (int x = 0; x < dst.cols; ++x)
            {
                dst_row[x] = exp(-0.5f * src_row[x] / sigma / sigma);
            }
        }
    }

    void catCn(const Mat_<Point3f>& a, const Mat_<float>& b, Mat_<Vec4f>& dst)
    {
        ensureSizeIsEnough(a.size(), dst);

        for (int y = 0; y < a.rows; ++y)
        {
            const Point3f* a_row = a[y];
            const float* b_row = b[y];
            Vec4f* dst_row = dst[y];

            for (int x = 0; x < a.cols; ++x)
            {
                const Point3f a_val = a_row[x];
                const float b_val = b_row[x];
                dst_row[x] = Vec4f(a_val.x, a_val.y, a_val.z, b_val);
            }
        }
    }

    void diffY(const Mat_<Point3f>& src, Mat_<Point3f>& dst)
    {
        ensureSizeIsEnough(src.rows - 1, src.cols, dst);

        for (int y = 0; y < src.rows - 1; ++y)
        {
            const Point3f* src_cur_row = src[y];
            const Point3f* src_next_row = src[y + 1];
            Point3f* dst_row = dst[y];

            for (int x = 0; x < src.cols; ++x)
            {
                dst_row[x] = src_next_row[x] - src_cur_row[x];
            }
        }
    }

    void diffX(const Mat_<Point3f>& src, Mat_<Point3f>& dst)
    {
        ensureSizeIsEnough(src.rows, src.cols - 1, dst);

        for (int y = 0; y < src.rows; ++y)
        {
            const Point3f* src_row = src[y];
            Point3f* dst_row = dst[y];

            for (int x = 0; x < src.cols - 1; ++x)
            {
                dst_row[x] = src_row[x + 1] - src_row[x];
            }
        }
    }

    void TransformedDomainRecursiveFilter(const Mat_<Vec4f>& I, const Mat_<float>& DH, const Mat_<float>& DV, Mat_<Vec4f>& dst, float sigma, Buf& buf)
    {
        CV_DbgAssert( I.size() == DH.size() );

        const float a = exp(-sqrt(2.0f) / sigma);

        ensureSizeIsEnough(I.size(), dst);
        I.copyTo(dst);

        ensureSizeIsEnough(DH.size(), buf.V);

        for (int y = 0; y < DH.rows; ++y)
        {
            const float* D_row = DH[y];
            float* V_row = buf.V[y];

            for (int x = 0; x < DH.cols; ++x)
            {
                V_row[x] = pow(a, D_row[x]);
            }
        }
        for (int y = 0; y < I.rows; ++y)
        {
            const float* V_row = buf.V[y];
            Vec4f* dst_row = dst[y];

            for (int x = 1; x < I.cols; ++x)
            {
                Vec4f dst_cur_val = dst_row[x];
                const Vec4f dst_prev_val = dst_row[x - 1];
                const float V_val = V_row[x];

                dst_cur_val[0] += V_val * (dst_prev_val[0] - dst_cur_val[0]);
                dst_cur_val[1] += V_val * (dst_prev_val[1] - dst_cur_val[1]);
                dst_cur_val[2] += V_val * (dst_prev_val[2] - dst_cur_val[2]);
                dst_cur_val[3] += V_val * (dst_prev_val[3] - dst_cur_val[3]);

                dst_row[x] = dst_cur_val;
            }
            for (int x = I.cols - 2; x >= 0; --x)
            {
                Vec4f dst_cur_val = dst_row[x];
                const Vec4f dst_prev_val = dst_row[x + 1];
                //const float V_val = V_row[x];
                const float V_val = V_row[x+1];

                dst_cur_val[0] += V_val * (dst_prev_val[0] - dst_cur_val[0]);
                dst_cur_val[1] += V_val * (dst_prev_val[1] - dst_cur_val[1]);
                dst_cur_val[2] += V_val * (dst_prev_val[2] - dst_cur_val[2]);
                dst_cur_val[3] += V_val * (dst_prev_val[3] - dst_cur_val[3]);

                dst_row[x] = dst_cur_val;
            }
        }

        for (int y = 0; y < DV.rows; ++y)
        {
            const float* D_row = DV[y];
            float* V_row = buf.V[y];

            for (int x = 0; x < DV.cols; ++x)
            {
                V_row[x] = pow(a, D_row[x]);
            }
        }
        for (int y = 1; y < I.rows; ++y)
        {
            const float* V_row = buf.V[y];
            Vec4f* dst_cur_row = dst[y];
            Vec4f* dst_prev_row = dst[y - 1];

            for (int x = 0; x < I.cols; ++x)
            {
                Vec4f dst_cur_val = dst_cur_row[x];
                const Vec4f dst_prev_val = dst_prev_row[x];
                const float V_val = V_row[x];

                dst_cur_val[0] += V_val * (dst_prev_val[0] - dst_cur_val[0]);
                dst_cur_val[1] += V_val * (dst_prev_val[1] - dst_cur_val[1]);
                dst_cur_val[2] += V_val * (dst_prev_val[2] - dst_cur_val[2]);
                dst_cur_val[3] += V_val * (dst_prev_val[3] - dst_cur_val[3]);

                dst_cur_row[x] = dst_cur_val;
            }
        }
        for (int y = I.rows - 2; y >= 0; --y)
        {
            //const float* V_row = buf.V[y];
            const float* V_row = buf.V[y + 1];
            Vec4f* dst_cur_row = dst[y];
            Vec4f* dst_prev_row = dst[y + 1];

            for (int x = 0; x < I.cols; ++x)
            {
                Vec4f dst_cur_val = dst_cur_row[x];
                const Vec4f dst_prev_val = dst_prev_row[x];
                const float V_val = V_row[x];

                dst_cur_val[0] += V_val * (dst_prev_val[0] - dst_cur_val[0]);
                dst_cur_val[1] += V_val * (dst_prev_val[1] - dst_cur_val[1]);
                dst_cur_val[2] += V_val * (dst_prev_val[2] - dst_cur_val[2]);
                dst_cur_val[3] += V_val * (dst_prev_val[3] - dst_cur_val[3]);

                dst_cur_row[x] = dst_cur_val;
            }
        }
    }

    void RF_filter(const Mat_<Vec4f>& src, const Mat_<Point3f>& src_joint, Mat_<Vec4f>& dst, float sigma_s, float sigma_r, Buf& buf)
    {
        CV_DbgAssert( src_joint.size() == src.size() );

        diffX(src_joint, buf.dIcdx);
        diffY(src_joint, buf.dIcdy);

        ensureSizeIsEnough(src.size(), buf.dIdx);
        buf.dIdx.setTo(Scalar::all(0));
        for (int y = 0; y < src.rows; ++y)
        {
            const Point3f* dIcdx_row = buf.dIcdx[y];
            float* dIdx_row = buf.dIdx[y];

            for (int x = 1; x < src.cols; ++x)
            {
                const Point3f val = dIcdx_row[x - 1];
                dIdx_row[x] = val.dot(val);
            }
        }

        ensureSizeIsEnough(src.size(), buf.dIdy);
        buf.dIdy.setTo(Scalar::all(0));
        for (int y = 1; y < src.rows; ++y)
        {
            const Point3f* dIcdy_row = buf.dIcdy[y - 1];
            float* dIdy_row = buf.dIdy[y];

            for (int x = 0; x < src.cols; ++x)
            {
                const Point3f val = dIcdy_row[x];
                dIdy_row[x] = val.dot(val);
            }
        }

        ensureSizeIsEnough(buf.dIdx.size(), buf.dHdx);
        buf.dIdx.convertTo(buf.dHdx, buf.dHdx.type(), (sigma_s / sigma_r) * (sigma_s / sigma_r), (sigma_s / sigma_s) * (sigma_s / sigma_s));
        sqrt(buf.dHdx, buf.dHdx);

        ensureSizeIsEnough(buf.dIdy.size(), buf.dVdy);
        buf.dIdy.convertTo(buf.dVdy, buf.dVdy.type(), (sigma_s / sigma_r) * (sigma_s / sigma_r), (sigma_s / sigma_s) * (sigma_s / sigma_s));
        sqrt(buf.dVdy, buf.dVdy);

        ensureSizeIsEnough(src.size(), dst);
        src.copyTo(dst);
        TransformedDomainRecursiveFilter(src, buf.dHdx, buf.dVdy, dst, sigma_s, buf);
    }

    void split_3_1(const Mat_<Vec4f>& src, Mat_<Point3f>& dst1, Mat_<float>& dst2)
    {
        ensureSizeIsEnough(src.size(), dst1);
        ensureSizeIsEnough(src.size(), dst2);

        for (int y = 0; y < src.rows; ++y)
        {
            const Vec4f* src_row = src[y];
            Point3f* dst1_row = dst1[y];
            float* dst2_row = dst2[y];

            for (int x = 0; x < src.cols; ++x)
            {
                Vec4f val = src_row[x];
                dst1_row[x] = Point3f(val[0], val[1], val[2]);
                dst2_row[x] = val[3];
            }
        }
    }

    void computeEigenVector(const Mat_<float>& X, const Mat_<uchar>& mask, Mat_<float>& dst, int num_pca_iterations, const Mat_<float>& rand_vec, Buf& buf)
    {
        CV_DbgAssert( X.cols == rand_vec.cols );
        CV_DbgAssert( X.rows == mask.size().area() );
        CV_DbgAssert( rand_vec.rows == 1 );

        ensureSizeIsEnough(rand_vec.size(), dst);
        rand_vec.copyTo(dst);

        ensureSizeIsEnough(X.size(), buf.t);

        float* dst_row = dst[0];

        for (int i = 0; i < num_pca_iterations; ++i)
        {
            buf.t.setTo(Scalar::all(0));

            for (int y = 0, ind = 0; y < mask.rows; ++y)
            {
                const uchar* mask_row = mask[y];

                for (int x = 0; x < mask.cols; ++x, ++ind)
                {
                    if (mask_row[x])
                    {
                        const float* X_row = X[ind];
                        float* t_row = buf.t[ind];

                        float dots = 0.0;
                        for (int c = 0; c < X.cols; ++c)
                            dots += dst_row[c] * X_row[c];

                        for (int c = 0; c < X.cols; ++c)
                            t_row[c] = dots * X_row[c];
                    }
                }
            }

            dst.setTo(0.0);
            for (int k = 0; k < X.rows; ++k)
            {
                const float* t_row = buf.t[k];

                for (int c = 0; c < X.cols; ++c)
                {
                    dst_row[c] += t_row[c];
                }
            }
        }

        double n = norm(dst);
        divide(dst, n, dst);
    }

    void calcEta(const Mat_<Point3f>& src_joint_f, const Mat_<float>& theta, const Mat_<uchar>& cluster, Mat_<Point3f>& dst, float sigma_s, float df, Buf& buf)
    {
        ensureSizeIsEnough(theta.size(), buf.theta_masked);
        buf.theta_masked.setTo(Scalar::all(0));
        theta.copyTo(buf.theta_masked, cluster);

        times(src_joint_f, buf.theta_masked, buf.mul);

        const Size nsz = Size(saturate_cast<int>(buf.mul.cols * (1.0 / df)), saturate_cast<int>(buf.mul.rows * (1.0 / df)));

        ensureSizeIsEnough(nsz, buf.numerator);
        resize(buf.mul, buf.numerator, Size(), 1.0 / df, 1.0 / df);

        ensureSizeIsEnough(nsz, buf.denominator);
        resize(buf.theta_masked, buf.denominator, Size(), 1.0 / df, 1.0 / df);        
        h_filter(buf.numerator, buf.numerator_filtered, sigma_s / df);
        h_filter(buf.denominator, buf.denominator_filtered, sigma_s / df);


        rdivide(buf.numerator_filtered, buf.denominator_filtered, dst);
    }

    void AdaptiveManifoldFilterRefImpl::buildManifoldsAndPerformFiltering(const Mat_<Point3f>& eta_k, const Mat_<uchar>& cluster_k, int current_tree_level)
    {
        // Compute downsampling factor

        double df = min(sigma_s_ / 4.0, 256.0 * sigma_r_);
        df = floor_to_power_of_two(df);
        df = max(1.0, df);

        // Splatting: project the pixel values onto the current manifold eta_k

        if (eta_k.rows == src_joint_f_.rows)
        {
            ensureSizeIsEnough(src_joint_f_.size(), buf_.X);
            subtract(src_joint_f_, eta_k, buf_.X);

            const Size nsz = Size(saturate_cast<int>(eta_k.cols * (1.0 / df)), saturate_cast<int>(eta_k.rows * (1.0 / df)));
            ensureSizeIsEnough(nsz, buf_.eta_k_small);
            resize(eta_k, buf_.eta_k_small, Size(), 1.0 / df, 1.0 / df);
        }
        else
        {
            ensureSizeIsEnough(eta_k.size(), buf_.eta_k_small);
            eta_k.copyTo(buf_.eta_k_small);

            ensureSizeIsEnough(src_joint_f_.size(), buf_.eta_k_big);
            resize(eta_k, buf_.eta_k_big, src_joint_f_.size());

            ensureSizeIsEnough(src_joint_f_.size(), buf_.X);
            subtract(src_joint_f_, buf_.eta_k_big, buf_.X);
        }

        // Project pixel colors onto the manifold -- Eq. (3), Eq. (5)

        ensureSizeIsEnough(buf_.X.size(), buf_.X_squared);
        multiply(buf_.X, buf_.X, buf_.X_squared);

        channelsSum(buf_.X_squared, buf_.pixel_dist_to_manifold_squared);

        phi(buf_.pixel_dist_to_manifold_squared, buf_.gaussian_distance_weights, sigma_r_over_sqrt_2_);

        times(src_f_, buf_.gaussian_distance_weights, buf_.Psi_splat);

        const Mat_<float>& Psi_splat_0 = buf_.gaussian_distance_weights;

        // Save min distance to later perform adjustment of outliers -- Eq. (10)

        if (adjust_outliers_)
        {
            cv::min(_InputArray(min_pixel_dist_to_manifold_squared_), _InputArray(buf_.pixel_dist_to_manifold_squared), _OutputArray(min_pixel_dist_to_manifold_squared_));
        }

        // Blurring: perform filtering over the current manifold eta_k

        catCn(buf_.Psi_splat, Psi_splat_0, buf_.Psi_splat_joined);

        ensureSizeIsEnough(buf_.eta_k_small.size(), buf_.Psi_splat_joined_resized);
        resize(buf_.Psi_splat_joined, buf_.Psi_splat_joined_resized, buf_.eta_k_small.size());

        RF_filter(buf_.Psi_splat_joined_resized, buf_.eta_k_small, buf_.blurred_projected_values, static_cast<float>(sigma_s_ / df), sigma_r_over_sqrt_2_, buf_);

        split_3_1(buf_.blurred_projected_values, buf_.w_ki_Psi_blur, buf_.w_ki_Psi_blur_0);

        // Slicing: gather blurred values from the manifold

        // Since we perform splatting and slicing at the same points over the manifolds,
        // the interpolation weights are equal to the gaussian weights used for splatting.
        const Mat_<float>& w_ki = buf_.gaussian_distance_weights;

        ensureSizeIsEnough(src_f_.size(), buf_.w_ki_Psi_blur_resized);
        resize(buf_.w_ki_Psi_blur, buf_.w_ki_Psi_blur_resized, src_f_.size());
        times(buf_.w_ki_Psi_blur_resized, w_ki, buf_.w_ki_Psi_blur_resized);
        add(sum_w_ki_Psi_blur_, buf_.w_ki_Psi_blur_resized, sum_w_ki_Psi_blur_);

        ensureSizeIsEnough(src_f_.size(), buf_.w_ki_Psi_blur_0_resized);
        resize(buf_.w_ki_Psi_blur_0, buf_.w_ki_Psi_blur_0_resized, src_f_.size());
        times(buf_.w_ki_Psi_blur_0_resized, w_ki, buf_.w_ki_Psi_blur_0_resized);
        add(sum_w_ki_Psi_blur_0_, buf_.w_ki_Psi_blur_0_resized, sum_w_ki_Psi_blur_0_);

        // Compute two new manifolds eta_minus and eta_plus

        if (current_tree_level < cur_tree_height_)
        {
            // Algorithm 1, Step 2: compute the eigenvector v1
            const Mat_<float> nX(src_joint_f_.size().area(), 3, (float*) buf_.X.data);

            ensureSizeIsEnough(1, nX.cols, buf_.rand_vec);
            if (useRNG)
            {
                rng_.fill(buf_.rand_vec, RNG::UNIFORM, -0.5, 0.5);
            }
            else
            {
                for (int i = 0; i < (int)buf_.rand_vec.total(); i++)
                    buf_.rand_vec(0, i) = (i % 2 == 0) ? 0.5f : -0.5f;
            }

            computeEigenVector(nX, cluster_k, buf_.v1, num_pca_iterations_, buf_.rand_vec, buf_);

            // Algorithm 1, Step 3: Segment pixels into two clusters -- Eq. (6)

            ensureSizeIsEnough(nX.rows, buf_.v1.rows, buf_.Nx_v1_mult);
            gemm(nX, buf_.v1, 1.0, noArray(), 0.0, buf_.Nx_v1_mult, GEMM_2_T);

            const Mat_<float> dot(src_joint_f_.rows, src_joint_f_.cols, (float*) buf_.Nx_v1_mult.data);

            Mat_<uchar>& cluster_minus = buf_.cluster_minus[current_tree_level];
            ensureSizeIsEnough(dot.size(), cluster_minus);
            compare(dot, 0, cluster_minus, CMP_LT);
            bitwise_and(cluster_minus, cluster_k, cluster_minus);

            Mat_<uchar>& cluster_plus = buf_.cluster_plus[current_tree_level];
            ensureSizeIsEnough(dot.size(), cluster_plus);
            //compare(dot, 0, cluster_plus, CMP_GT);
            compare(dot, 0, cluster_plus, CMP_GE);
            bitwise_and(cluster_plus, cluster_k, cluster_plus);

            // Algorithm 1, Step 4: Compute new manifolds by weighted low-pass filtering -- Eq. (7-8)

            ensureSizeIsEnough(w_ki.size(), buf_.theta);
            buf_.theta.setTo(Scalar::all(1.0));
            subtract(buf_.theta, w_ki, buf_.theta);

            Mat_<Point3f>& eta_minus = buf_.eta_minus[current_tree_level];
            calcEta(src_joint_f_, buf_.theta, cluster_minus, eta_minus, (float)sigma_s_, (float)df, buf_);

            Mat_<Point3f>& eta_plus = buf_.eta_plus[current_tree_level];
            calcEta(src_joint_f_, buf_.theta, cluster_plus, eta_plus, (float)sigma_s_, (float)df, buf_);

            // Algorithm 1, Step 5: recursively build more manifolds.

            buildManifoldsAndPerformFiltering(eta_minus, cluster_minus, current_tree_level + 1);
            buildManifoldsAndPerformFiltering(eta_plus, cluster_plus, current_tree_level + 1);
        }
    }
}

namespace cvtest
{

using namespace cv::ximgproc;

Ptr<AdaptiveManifoldFilter> createAMFilterRefImpl(double sigma_s, double sigma_r, bool adjust_outliers)
{
    Ptr<AdaptiveManifoldFilter> amf(new AdaptiveManifoldFilterRefImpl());
    
    amf->setSigmaS(sigma_s);
    amf->setSigmaR(sigma_r);
    amf->setAdjustOutliers(adjust_outliers);

    return amf;
}

}
