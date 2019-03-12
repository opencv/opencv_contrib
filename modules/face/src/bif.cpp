/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015, OpenCV Foundation, all rights reserved.
Copyright (C) 2015, Itseez Inc., all rights reserved.
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
This file contains implementation of the bio-inspired features (BIF) approach
for computing image descriptors, applicable for human age estimation. For more
details we refer to [1,2].

REFERENCES
  [1] Guo, Guodong, et al. "Human age estimation using bio-inspired features."
      Computer Vision and Pattern Recognition, 2009. CVPR 2009.
  [2] Spizhevoi, A. S., and A. V. Bovyrin. "Estimating human age using
      bio-inspired features and the ranking method." Pattern Recognition and
      Image Analysis 25.3 (2015): 547-552.
*/

#include "precomp.hpp"
#include "opencv2/face/bif.hpp"
#include <iostream>
#include <vector>

namespace {

// The constants below are taken from paper [1].

const int kNumBandsMax = 8;

const cv::Size kCellSizes[kNumBandsMax] = {
    cv::Size(6,6), cv::Size(8,8), cv::Size(10,10), cv::Size(12,12),
    cv::Size(14,14), cv::Size(16,16), cv::Size(18,18), cv::Size(20,20)
};

const cv::Size kGaborSize[kNumBandsMax][2] = {
    {cv::Size(5,5), cv::Size(7,7)}, {cv::Size(9,9), cv::Size(11,11)},
    {cv::Size(13,13), cv::Size(15,15)}, {cv::Size(17,17), cv::Size(19,19)},
    {cv::Size(21,21), cv::Size(23,23)}, {cv::Size(25,25), cv::Size(27,27)},
    {cv::Size(29,29), cv::Size(31,31)}, {cv::Size(33,33), cv::Size(35,35)}
};

const double kGaborGamma = 0.3;

const double kGaborSigmas[kNumBandsMax][2] = {
    {2.0, 2.8}, {3.6, 4.5}, {5.4, 6.3}, {7.3, 8.2},
    {9.2, 10.2}, {11.3, 12.3}, {13.4, 14.6}, {15.8, 17.0}
};

const double kGaborWavelens[kNumBandsMax][2] = {
    {2.5, 3.5}, {4.6, 5.6}, {6.8, 7.9}, {9.1, 10.3},
    {11.5, 12.7}, {14.1, 15.4}, {16.8, 18.2}, {19.7, 21.2}
};

class BIFImpl CV_FINAL : public cv::face::BIF {
public:
    BIFImpl(int num_bands, int num_rotations) {
        initUnits(num_bands, num_rotations);
    }

    virtual int getNumBands() const CV_OVERRIDE { return num_bands_; }

    virtual int getNumRotations() const CV_OVERRIDE { return num_rotations_; }

    virtual void compute(cv::InputArray image,
                         cv::OutputArray features) const CV_OVERRIDE;

private:
    struct UnitParams {
        cv::Size cell_size;
        cv::Mat filter1, filter2;
    };

    void initUnits(int num_bands, int num_rotations);
    void computeUnit(int unit_idx, const cv::Mat &img, cv::Mat &dst) const;

    int num_bands_;
    int num_rotations_;
    std::vector<UnitParams> units_;
};

void BIFImpl::compute(cv::InputArray _image,
                      cv::OutputArray _features) const {
    cv::Mat image = _image.getMat();
    CV_Assert(image.type() == CV_32F);

    std::vector<cv::Mat> fea_units(units_.size());
    int fea_dim = 0;

    for (size_t i = 0; i < units_.size(); ++i) {
        computeUnit(static_cast<int>(i), image, fea_units[i]);
        fea_dim += fea_units[i].rows;
    }

    _features.create(fea_dim, 1, CV_32F);
    cv::Mat fea = _features.getMat();

    int offset = 0;
    for (size_t i = 0; i < fea_units.size(); ++i) {
        cv::Mat roi = fea.rowRange(offset, offset + fea_units[i].rows);
        fea_units[i].copyTo(roi);
        offset += fea_units[i].rows;
    }
    CV_Assert(offset == fea_dim);
}

void BIFImpl::initUnits(int num_bands, int num_rotations) {
    CV_Assert(num_bands > 0 && num_bands <= kNumBandsMax);
    CV_Assert(num_rotations > 0);

    num_bands_ = num_bands;
    num_rotations_ = num_rotations;

    for (int ri = 0; ri < num_rotations; ++ri) {
        double angle = CV_PI / num_rotations * ri;

        for (int bi = 0; bi < num_bands; ++bi) {
            cv::Mat kernel[2];
            for (int i = 0; i < 2; ++i) {
                kernel[i] = cv::getGaborKernel(
                    kGaborSize[bi][i], kGaborSigmas[bi][i], angle,
                    kGaborWavelens[bi][i], kGaborGamma, 0, CV_32F);

                // Make variance for the Gaussian part of the Gabor filter
                // the same across all filters.
                kernel[i] /= 2 * kGaborSigmas[bi][i] * kGaborSigmas[bi][i]
                             / kGaborGamma;
            }

            UnitParams unit;
            unit.cell_size = kCellSizes[bi];
            unit.filter1 = kernel[0];
            unit.filter2 = kernel[1];
            units_.push_back(unit);
        }
    }
}

void BIFImpl::computeUnit(int unit_idx, const cv::Mat &img,
                          cv::Mat &dst) const {
    cv::Mat resp1, resp2;
    cv::filter2D(img, resp1, CV_32F, units_[unit_idx].filter1);
    cv::filter2D(img, resp2, CV_32F, units_[unit_idx].filter2);

    cv::Mat resp, sum, sumsq;
    cv::max(resp1, resp2, resp);
    cv::integral(resp, sum, sumsq);

    int Hhalf = units_[unit_idx].cell_size.height / 2;
    int Whalf = units_[unit_idx].cell_size.width / 2;

    int nrows = (resp.rows + Hhalf - 1) / Hhalf;
    int ncols = (resp.cols + Whalf - 1) / Whalf;
    dst.create(nrows*ncols, 1, CV_32F);

    for (int pos = 0, yc = 0; yc < resp.rows; yc += Hhalf) {
        int y0 = std::max(0, yc - Hhalf);
        int y1 = std::min(resp.rows, yc + Hhalf);

        for (int xc = 0; xc < resp.cols; xc += Whalf, ++pos) {
            int x0 = std::max(0, xc - Whalf);
            int x1 = std::min(resp.cols, xc + Whalf);
            int area = (y1-y0) * (x1-x0);

            double mean = sum.at<double>(y1,x1) - sum.at<double>(y1,x0)
                         - sum.at<double>(y0,x1) + sum.at<double>(y0,x0);
            mean /= area;

            double sd = sumsq.at<double>(y1,x1) - sumsq.at<double>(y1,x0)
                        - sumsq.at<double>(y0,x1) + sumsq.at<double>(y0,x0);
            sd = sqrt(std::max(0.0, sd / area - mean * mean));

            dst.at<float>(pos) = static_cast<float>(sd);
        }
    }
}

}  // namespace

cv::Ptr<cv::face::BIF> cv::face::BIF::create(int num_bands, int num_rotations) {
    return cv::Ptr<cv::face::BIF>(new BIFImpl(num_bands, num_rotations));
}
