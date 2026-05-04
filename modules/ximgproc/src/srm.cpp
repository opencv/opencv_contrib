/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.
                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)
Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/


/******************************************************************************\
 *                        Graph based segmentation                             *
 * This code implements the segmentation method described in:                  *
 * R. Nock and F. Nielsen, "Statistical region merging," in IEEE Transactions  *
 * on  Pattern Analysis and Machine Intelligence, vol. 26, no. 11,
 * pp. 1452-1458, Nov. 2004, doi: 10.1109/TPAMI.2004.110. * Author: Dheer Prasad / IIT Bombay / 2026
 **
 ************************************************************************    *******/


#include "precomp.hpp"
#include "opencv2/ximgproc/segmentation.hpp"

#include <iostream>
#include <unordered_set>

using namespace std;
using namespace cv;

namespace cv { namespace ximgproc { namespace segmentation {
class region
{
    // defines what region a point is part of;
public:
    int parent;
    int gray;
    int size;
    bool is_boundary;
};
class p_error
{
public:
    int p1;
    int p2;
    float error;
};


class srm_segment_impl : public srm_segment
{
public:
    srm_segment_impl()
    {
        Q = 32;  // default value also used in the paper;
        sigma = 0.5;  // for gaussian blur
        name_ = "SRMSegmentation";
    }
    ~srm_segment_impl() CV_OVERRIDE {};

    virtual void ProcessImage(InputArray src, OutputArray out) CV_OVERRIDE;
    virtual void set_Q(int Q_) CV_OVERRIDE
    {
        if (Q_ <= 0)
        {
            Q_ = 1;
        }
        Q = Q_;
    }
    virtual void set_sigma(double sigma_) CV_OVERRIDE
    {
        if (sigma_ <= 0)
        {
            sigma_ = 0.01;
        }
        sigma = sigma_;
    }

private:
    int Q;
    double sigma;
    String name_;
    void uniter(vector<region>& regions, int r1, int r2);
    void filter(const Mat& img, Mat& filtered_img);
    void build_diff(const Mat& in_img, Mat& out_img_x, Mat& out_img_y);
    void build_regions(Mat& in_img, vector<region>& regions);
    void compare_region(vector<region>& regions, int r1, int r2, int out_h, int out_w);
    int finder(vector<region>& regions, int x);
    float br(int out_h, int out_w, region r);
    float delta_log(int height, int width);
    void build_error_graph(Mat& img, Mat& out_img_x, vector<p_error>& error_graph);
};
float srm_segment_impl::delta_log(int height, int width)
{
    float i = (float)(height) * (float)width;
    return -(logf(6.0f) + 2.0f * logf(i));
}
float srm_segment_impl::br(int out_h, int out_w, region r)
{
    float log_term = log((float)r.size) - delta_log(out_h, out_w);
    float denom = 2.0f * (float)Q * (float)r.size;
    return 256.0f * sqrtf(log_term / denom);
}
void srm_segment_impl::uniter(vector<region>& regions, int r1, int r2)
{
    r1 = finder(regions, r1);
    r2 = finder(regions, r2);
    if (r1 == r2)
    {
        return;
    }
    if (regions[r1].size < regions[r2].size)
    {
        int temp = r1;
        r1 = r2;
        r2 = temp;
    }
    regions[r2].parent = r1;
    int size1 = regions[r1].size;
    int size2 = regions[r2].size;
    regions[r1].gray = (regions[r1].gray * size1 + regions[r2].gray * size2) / (size1 + size2);
    regions[r1].size += regions[r2].size;
    return;
}
void srm_segment_impl::compare_region(vector<region>& regions, int r1, int r2, int out_h, int out_w)
{
    float br1 = br(out_h, out_w, regions[r1]);
    float br2 = br(out_h, out_w, regions[r2]);
    float threshold = sqrtl((br1 * br1) + (br2 * br2));

    if (abs(regions[r1].gray - regions[r2].gray) <= threshold)
    {
        uniter(regions, r1, r2);
    }
}
void srm_segment_impl::filter(const Mat& img_converted, Mat& filtered_img)
{
    GaussianBlur(img_converted, filtered_img, Size(0, 0), sigma, sigma);
}
void srm_segment_impl::build_diff(const Mat& in_img, Mat& out_img_x, Mat& out_img_y)
{
    // Here I am applying [-2,0,2] kernel for out_x;
    // It's transpose for y;

    Mat kernel_x = (Mat_<float>(1, 3) << -2, 0, 2);
    Mat kernel_y = (Mat_<float>(3, 1) << -2, 0, 2);
    filter2D(in_img, out_img_x, CV_32F, kernel_x);
    filter2D(in_img, out_img_y, CV_32F, kernel_y);
}
void srm_segment_impl::build_error_graph(Mat& img, Mat& out_img_x, vector<p_error>& error_graph)
{
    int counter = 0;
    for (int y = 0; y < out_img_x.rows; y++)
    {
        for (int x = 0; x < out_img_x.cols; x++)
        {
            if (x < out_img_x.cols - 1)
            {
                p_error pe;
                pe.p1 = counter;
                pe.p2 = counter + 1;
                // pe.error=abs(out_img_x.at<float>(y, x));
                pe.error = abs(img.at<float>(y, x) - img.at<float>(y, x + 1));
                error_graph.push_back(pe);
            }
            if (y < out_img_x.rows - 1)
            {
                p_error pe;
                pe.p1 = counter;
                pe.p2 = counter + out_img_x.cols;
                // pe.error=abs(out_img_y.at<float>(y,x));
                pe.error = abs(img.at<float>(y, x) - img.at<float>(y + 1, x));
                error_graph.push_back(pe);
            }
            counter++;
        }
    }

    sort(error_graph.begin(), error_graph.end(), [](const p_error& a, const p_error& b) {
        return a.error < b.error;
    });
}
void srm_segment_impl::build_regions(Mat& in_img, vector<region>& regions)
{
    int counter = 0;
    for (int y = 0; y < in_img.rows; y++)
    {
        for (int x = 0; x < in_img.cols; x++)
        {
            region r1;
            r1.parent = counter;
            r1.gray = in_img.at<float>(y, x);
            r1.size = 1;
            r1.is_boundary = false;
            regions[counter] = r1;
            counter++;
        }
    }
}
int srm_segment_impl::finder(vector<region>& regions, int x)
{
    // We can try amortized O(1) here but
    // that will lead to a stack overflow for larger images
    int root = x;
    while (regions[root].parent != root)
    {
        root = regions[root].parent;
    }
    return root;
}
void srm_segment_impl::ProcessImage(InputArray src, OutputArray dst)
{
    Mat img_color = src.getMat();
    dst.create(img_color.rows, img_color.cols, CV_32F);
    Mat img;
    if (img_color.channels() == 3)
    {
        cvtColor(img_color, img, COLOR_BGR2GRAY);
    }
    else
    {
        img = img_color;
    }

    img.convertTo(img, CV_32F);
    Mat output = dst.getMat();
    output.setTo(0);
    Mat img_filter;
    filter(img, img_filter);
    Mat out_img_x(img.rows, img.cols, CV_32F);
    out_img_x.setTo(0);
    Mat out_img_y(img.rows, img.cols, CV_32F);
    out_img_y.setTo(0);
    build_diff(img_filter, out_img_x, out_img_y);
    vector<p_error> error_graph;
    build_error_graph(img_filter, out_img_x, error_graph);
    vector<region> regions(img.rows * img.cols);
    build_regions(img_filter, regions);
    for (int i = 0; i < int(error_graph.size()); i++)
    {
        int i1 = error_graph[i].p1;
        int i2 = error_graph[i].p2;
        int r1 = finder(regions, i1);
        int r2 = finder(regions, i2);
        if (r1 == r2)
        {
            continue;
        }
        else
        {
            compare_region(regions, r1, r2, img.rows, img.cols);
        }
    }
    int counter = 0;
    unordered_set<int> unique_regions;
    for (int y = 0; y < img.rows; y++)
    {
        for (int x = 0; x < img.cols; x++)
        {
            int re1 = finder(regions, counter);
            unique_regions.insert(re1);
            if (x < img.cols - 1)
            {
                int re2 = finder(regions, counter + 1);
                if (re1 != re2)
                {
                    output.at<float>(y, x) = 1;
                    regions[counter].is_boundary = true;
                }
                else
                {
                    output.at<float>(y, x) = regions[re1].gray;
                }
            }
            if (y < img.rows - 1)
            {
                int re2 = finder(regions, counter + img.cols);
                if (re1 != re2)
                {
                    output.at<float>(y, x) = 1;
                    regions[counter].is_boundary = true;
                }
                else
                {
                    output.at<float>(y, x) = regions[re1].gray;
                }
            }
            counter++;
        }
    }
}
Ptr<srm_segment> createSRMSegmentation()
{
    return makePtr<srm_segment_impl>();
}
void SRMSegmentation(InputArray src, OutputArray dst, int Q, double sigma)
{
    Ptr<srm_segment> srm = createSRMSegmentation();
    srm->set_Q(Q);
    srm->set_sigma(sigma);
    srm->ProcessImage(src, dst);
}
}}}  // namespace cv::ximgproc::segmentation
