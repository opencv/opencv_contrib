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
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#pragma once
#include "kyheader.h"
class FilterTIG
{
public:
    void update(CMat &w);

    // For a W by H gradient magnitude map, find a W-7 by H-7 CV_32F matching score map
    cv::Mat matchTemplate(const cv::Mat &mag1u);

    inline float dot(const int64_t tig1, const int64_t tig2, const int64_t tig4, const int64_t tig8);

public:
    void reconstruct(cv::Mat &w); // For illustration purpose

private:
    static const int NUM_COMP = 2; // Number of components
    static const int D = 64; // Dimension of TIG
    int64_t _bTIGs[NUM_COMP]; // Binary TIG features
    float _coeffs1[NUM_COMP]; // Coefficients of binary TIG features

    // For efficiently deals with different bits in CV_8U gradient map
    float _coeffs2[NUM_COMP], _coeffs4[NUM_COMP], _coeffs8[NUM_COMP];
};


inline float FilterTIG::dot(const int64_t tig1, const int64_t tig2, const int64_t tig4, const int64_t tig8)
{
  int64_t bcT1 = (int64_t)POPCNT64(tig1);
  int64_t bcT2 = (int64_t)POPCNT64(tig2);
  int64_t bcT4 = (int64_t)POPCNT64(tig4);
  int64_t bcT8 = (int64_t)POPCNT64(tig8);

  int64_t bc01 = (int64_t)(POPCNT64(_bTIGs[0] & tig1) << 1) - bcT1;
  int64_t bc02 = (int64_t)((POPCNT64(_bTIGs[0] & tig2) << 1) - bcT2) << 1;
  int64_t bc04 = (int64_t)((POPCNT64(_bTIGs[0] & tig4) << 1) - bcT4) << 2;
  int64_t bc08 = (int64_t)((POPCNT64(_bTIGs[0] & tig8) << 1) - bcT8) << 3;

  int64_t bc11 = (int64_t)(POPCNT64(_bTIGs[1] & tig1) << 1) - bcT1;
  int64_t bc12 = (int64_t)((POPCNT64(_bTIGs[1] & tig2) << 1) - bcT2) << 1;
  int64_t bc14 = (int64_t)((POPCNT64(_bTIGs[1] & tig4) << 1) - bcT4) << 2;
  int64_t bc18 = (int64_t)((POPCNT64(_bTIGs[1] & tig8) << 1) - bcT8) << 3;

    return _coeffs1[0] * (bc01 + bc02 + bc04 + bc08) + _coeffs1[1] * (bc11 + bc12 + bc14 + bc18);
}
