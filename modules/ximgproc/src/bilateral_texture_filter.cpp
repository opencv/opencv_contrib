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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "precomp.hpp"
#include <opencv2/ximgproc.hpp>
#include <vector>

namespace cv
{
namespace ximgproc
{
  void compute_mRTV(const Mat& L, Mat& mRTV, int fr);
  void compute_G(const Mat& B, const Mat& mRTV, Mat& G, Mat& alpha, int fr);
  void joint_bilateral_filter(const Mat& img, const Mat& G, Mat& r_img, int fr2, double sigma_avg);
  void joint_bilateral_filter3(const Mat& img, const Mat& G, Mat& r_img, int fr2, double sigma_avg);

  void bilateralTextureFilter(InputArray src_, OutputArray dst_, int fr,
                              int numIter, double sigmaAlpha, double sigmaAvg)
  {
    CV_Assert(!src_.empty());

    Mat src = src_.getMat();
    CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);

    CV_Assert(fr > 0 && numIter > 0);

    if (sigmaAlpha < 0)
      sigmaAlpha = 5. * fr;
    if (sigmaAvg < 0)
      sigmaAvg = 0.05 * sqrt(src.channels());

    Mat I;
    src.copyTo(I);
    if (src.type() == CV_8UC1) {
      I.convertTo(I, CV_32FC1, 1.0 / 255.0);
    }
    else if (src.type() == CV_8UC3) {
      I.convertTo(I, CV_32FC3, 1.0 / 255.0);
    }

    for (int iter = 0; iter < numIter; iter++)
    {
      Mat B;
      blur(I, B, Size(2 * fr + 1, 2 * fr + 1), Point(-1, -1), BORDER_REFLECT);

      Mat mRTV;
      compute_mRTV(I, mRTV, fr);

      Mat G, minmRTV;
      compute_G(B, mRTV, G, minmRTV, fr);

      // alpha blending
      Mat Gtilde;
      Mat diff = mRTV - minmRTV;
      Mat alpha = -diff.mul(sigmaAlpha);
      exp(alpha, alpha);
      alpha = alpha + 1.;
      pow(alpha, -1, alpha);
      alpha = (alpha - 0.5) * 2;
      Mat alphainv = -(alpha - 1);

      std::vector<Mat> Gi, Bi;
      Gi.resize(I.channels());
      Bi.resize(I.channels());
      if (I.channels() == 3) {
        split(G, &Gi[0]);
        split(B, &Bi[0]);
      }
      else {
        G.copyTo(Gi[0]);
        B.copyTo(Bi[0]);
      }

      std::vector<Mat> Gtildei;
      Gtildei.resize(I.channels());
      for (int i = 0; i < B.channels(); i++)
        Gtildei[i] = Gi[i].mul(alpha) + Bi[i].mul(alphainv);
      merge(&Gtildei[0], B.channels(), Gtilde);

      // joint bilateral filter
      cv::Mat J;
      if (I.channels() == 1)
        joint_bilateral_filter(I, Gtilde, J, fr * 2, sigmaAvg);
      else if (I.channels() == 3)
        joint_bilateral_filter3(I, Gtilde, J, fr * 2, sigmaAvg);
      I = J;
    }
    if (src.type() == CV_8UC1) {
      I.convertTo(I, CV_8UC1, 255.0);
    }
    else if (src.type() == CV_8UC3) {
      I.convertTo(I, CV_8UC3, 255.0);
    }

    I.copyTo(dst_);
  }

  void compute_mRTV(const Mat& L, Mat& mRTV, int fr)
  {
    mRTV = Mat::zeros(L.size(), CV_32FC1);

    const float eps = 0.00001f;

    // Calculate image derivative(gradient)
    Mat G;
    Mat Gx, Gy, kernelx, kernely;
    kernelx = Mat::zeros(1, 3, CV_32F);
    kernelx.at<float>(0, 1) = -1.0;
    kernelx.at<float>(0, 2) = 1.0;
    filter2D(L, Gx, -1, kernelx, Point(-1, -1), 0, BORDER_REFLECT);
    kernely = Mat::zeros(3, 1, CV_32F);
    kernely.at<float>(1, 0) = -1.0;
    kernely.at<float>(2, 0) = 1.0;
    filter2D(L, Gy, -1, kernely, Point(-1, -1), 0, BORDER_REFLECT);

    Gx = Gx.mul(Gx);
    Gy = Gy.mul(Gy);
    sqrt(Gx + Gy, G);

    // Pad image L and G
    Mat padL;
    Mat padG;
    copyMakeBorder(L, padL, fr, fr, fr, fr, BORDER_REFLECT);
    copyMakeBorder(G, padG, fr, fr, fr, fr, BORDER_REFLECT);

    // Calculate maxL, minL, maxG, sumG
    int pu = fr;
    int pb = pu + L.rows;
    int pl = fr;
    int pr = pl + L.cols;

    std::vector<Mat> Li, Gi;
    Li.resize(L.channels());
    Gi.resize(L.channels());
    if (L.channels() == 3) {
      split(padL, &Li[0]);
      split(padG, &Gi[0]);
    }
    else {
      padL.copyTo(Li[0]);
      padG.copyTo(Gi[0]);
    }

    for (int i = 0; i < L.channels(); i++)
    {
      Mat maxL = Mat::zeros(L.size(), CV_32FC1);
      Mat minL = Mat::ones(L.size(), CV_32FC1);
      Mat maxG = Mat::zeros(L.size(), CV_32FC1);
      Mat sumG = Mat::zeros(L.size(), CV_32FC1);
      for (int y = -fr; y <= fr; y++)
      {
        for (int x = -fr; x <= fr; x++)
        {
          Mat temp = Li[i](
            Range(pu + y, pb + y),
            Range(pl + x, pr + x)
          );
          maxL = max(maxL, temp);
          minL = min(minL, temp);

          temp = Gi[i](
            Range(pu + y, pb + y),
            Range(pl + x, pr + x)
          );
          maxG = max(maxG, temp);
          sumG = sumG + temp;
        }
      }
      Mat deltai = maxL - minL;
      sumG = max(sumG, eps);
      Mat mRTVi = maxG / sumG * (2 * fr + 1);
      mRTV = mRTV + mRTVi.mul(deltai);
    }
    if (L.channels() == 3)
      mRTV = mRTV / 3;
  }

  void compute_G(const Mat& B, const Mat& mRTV, Mat& G, Mat& alpha, int fr)
  {
    B.copyTo(G);
    alpha = Mat::ones(B.size(), CV_32FC1);
    for (int y = -fr; y <= fr; y++)
    {
      for (int x = -fr; x <= fr; x++)
      {
        Point pb;
        Point pt;
        for (pb.y = 0; pb.y < B.rows; pb.y++)
        {
          for (pb.x = 0; pb.x < B.cols; pb.x++)
          {
            pt.x = min(max(pb.x + x, 0), B.cols - 1);
            pt.y = min(max(pb.y + y, 0), B.rows - 1);
            if (alpha.at<float>(pb) > mRTV.at<float>(pt))
            {
              alpha.at<float>(pb) = mRTV.at<float>(pt);
              if (B.channels() == 3)
                G.at<Vec3f>(pb) = B.at<Vec3f>(pt);
              else if (B.channels() == 1)
                G.at<float>(pb) = B.at<float>(pt);
            }
          }
        }
      }
    }
  }

  void joint_bilateral_filter(const Mat& img, const Mat& G, Mat& r_img, int fr2, double sigma_avg)
  {
    Mat p_G;
    copyMakeBorder(G, p_G, fr2, fr2, fr2, fr2, BORDER_REFLECT);

    Mat p_img;
    copyMakeBorder(img, p_img, fr2, fr2, fr2, fr2, BORDER_REFLECT);

    Mat SW;
    if (SW.empty()) {
      SW = Mat(2*fr2+1, 2*fr2+1, CV_32FC1);
      int r, c;
      float y, x;
      for (r = 0, y = (float)-fr2; r < SW.rows; r++, y += 1.0) {
        for(c = 0, x = (float)-fr2; c < SW.cols; c++, x += 1.0) {
          SW.at<float>(r,c) = exp(-(x*x + y*y) / (2*fr2*fr2));
        }
      }
    }

    r_img = Mat::zeros(img.size(), CV_32FC1);
    {
      Mat sum_d_W = Mat::zeros(img.size(), CV_32FC1);
      Mat d_W = Mat::zeros(G.size(), CV_32FC1);

      for (int x = -fr2; x <= fr2; x++) {
        for (int y = -fr2; y <= fr2; y++) {
          d_W = p_G(Rect(fr2+x, fr2+y, img.cols, img.rows)) - G;
          multiply(d_W, d_W, d_W);
          exp(-0.5 * d_W / (sigma_avg*sigma_avg), d_W);

          d_W = d_W * SW.at<float>(fr2+y, fr2+x); //Gaussian weight

          sum_d_W = sum_d_W + d_W;
          multiply(d_W, p_img(Rect(fr2+x, fr2+y, img.cols, img.rows)), d_W);
          r_img = r_img + d_W;
        }
      }
      max(1e-5f, sum_d_W, sum_d_W);
      divide(r_img, sum_d_W, r_img);
    }
  }

  void joint_bilateral_filter3(const Mat& img, const Mat& G, Mat& r_img, int fr2, double sigma_avg)
  {
    Mat p_G;
    copyMakeBorder(G, p_G, fr2, fr2, fr2, fr2, BORDER_REFLECT);

    Mat p_img;
    copyMakeBorder(img, p_img, fr2, fr2, fr2, fr2, BORDER_REFLECT);

    Mat SW;
    if (SW.empty()) {
      SW = Mat(2*fr2+1, 2*fr2+1, CV_32FC1);
      int r, c;
      float y, x;
      for (r = 0, y = (float)-fr2; r < SW.rows; r++, y += 1.0) {
        for(c = 0, x = (float)-fr2; c < SW.cols; c++, x += 1.0) {
          SW.at<float>(r,c) = exp(-(x*x + y*y) / (2*fr2*fr2));
        }
      }
    }

    std::vector<Mat> G_channels(3);
    split(G, G_channels);
    std::vector<Mat> p_G_channels(3);
    split(p_G, p_G_channels);
    std::vector<Mat> p_img_channels(3);
    split(p_img, p_img_channels);

    Mat sum_d_W = Mat::zeros(img.size(), CV_32FC1);
    std::vector<Mat> d_W_channels(3);
    for (int ch = 0; ch < 3; ch++) {
      d_W_channels[ch] = Mat::zeros(G.size(), CV_32FC1);
    }
    Mat d_W = Mat::zeros(G.size(), CV_32FC1);

    std::vector<Mat> r_img_channels(3);
    for (int ch = 0; ch < 3; ch++) {
      r_img_channels[ch] = Mat::zeros(G.size(), CV_32FC1);
    }

    for (int x = -fr2; x <= fr2; x++) {
      for (int y = -fr2; y <= fr2; y++) {
        d_W.setTo(0);
        for (int ch = 0; ch < 3; ch++) {
          subtract(p_G_channels[ch](Rect(fr2+x, fr2+y, img.cols, img.rows)), G_channels[ch], d_W_channels[ch]);
          multiply(d_W_channels[ch], d_W_channels[ch], d_W_channels[ch]);
        }
        for (int ch = 0; ch < 3; ch++) {
          add(d_W, d_W_channels[ch], d_W);
        }
        exp(-0.5 * d_W / (sigma_avg*sigma_avg), d_W);

        d_W = d_W * SW.at<float>(fr2+y, fr2+x); //Gaussian weight

        add(sum_d_W, d_W, sum_d_W);

        for (int ch = 0; ch < 3; ch++) {
          Mat n_p_img = p_img_channels[ch](Rect(fr2+x, fr2+y, img.cols, img.rows));
          accumulateProduct(d_W, n_p_img, r_img_channels[ch]);
        }
      }
    }

    max(1e-5f, sum_d_W, sum_d_W);
    for (int ch = 0; ch < 3; ch++) {
      divide(r_img_channels[ch], sum_d_W, r_img_channels[ch]);
    }
    merge(r_img_channels, r_img);
  }

}
}
