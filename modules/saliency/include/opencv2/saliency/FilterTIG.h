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

    inline float dot(const INT64 tig1, const INT64 tig2, const INT64 tig4, const INT64 tig8);

public:
    void reconstruct(cv::Mat &w); // For illustration purpose

private:
    static const int NUM_COMP = 2; // Number of components
    static const int D = 64; // Dimension of TIG
    INT64 _bTIGs[NUM_COMP]; // Binary TIG features
    float _coeffs1[NUM_COMP]; // Coefficients of binary TIG features

    // For efficiently deals with different bits in CV_8U gradient map
    float _coeffs2[NUM_COMP], _coeffs4[NUM_COMP], _coeffs8[NUM_COMP];
};

inline int __builtin_popcount(unsigned int x) {
static const unsigned int m1 = 0x55555555; //binary: 0101...
static const unsigned int m2 = 0x33333333; //binary: 00110011..
static const unsigned int m4 = 0x0f0f0f0f; //binary: 4 zeros, 4 ones ...
static const unsigned int h01= 0x01010101; //the sum of 256 to the power of 0,1,2,3...
x -= (x >> 1) & m1; //put count of each 2 bits into those 2 bits
x = (x & m2) + ((x >> 2) & m2); //put count of each 4 bits into those 4 bits
x = (x + (x >> 4)) & m4; //put count of each 8 bits into those 8 bits
return (x * h01) >> 24; //returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24)
}

inline int __builtin_popcountl(unsigned long x) {
return __builtin_popcount(static_cast<int>(x));
}

inline int __builtin_popcountll(unsigned long long x) {
static const unsigned long long m1 = 0x5555555555555555; //binary: 0101...
static const unsigned long long m2 = 0x3333333333333333; //binary: 00110011..
static const unsigned long long m4 = 0x0f0f0f0f0f0f0f0f; //binary: 4 zeros, 4 ones ...
static const unsigned long long h01 = 0x0101010101010101; //the sum of 256 to the power of 0,1,2,3...
x -= (x >> 1) & m1; //put count of each 2 bits into those 2 bits
x = (x & m2) + ((x >> 2) & m2); //put count of each 4 bits into those 4 bits
x = (x + (x >> 4)) & m4; //put count of each 8 bits into those 8 bits
return static_cast<int>((x * h01)>>56); //returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ...
}


inline float FilterTIG::dot(const INT64 tig1, const INT64 tig2, const INT64 tig4, const INT64 tig8)
{
    INT64 bcT1 = __builtin_popcountll(tig1);
    INT64 bcT2 = __builtin_popcountll(tig2);
    INT64 bcT4 = __builtin_popcountll(tig4);
    INT64 bcT8 = __builtin_popcountll(tig8);

    INT64 bc01 = (__builtin_popcountll(_bTIGs[0] & tig1) << 1) - bcT1;
    INT64 bc02 = ((__builtin_popcountll(_bTIGs[0] & tig2) << 1) - bcT2) << 1;
    INT64 bc04 = ((__builtin_popcountll(_bTIGs[0] & tig4) << 1) - bcT4) << 2;
    INT64 bc08 = ((__builtin_popcountll(_bTIGs[0] & tig8) << 1) - bcT8) << 3;

    INT64 bc11 = (__builtin_popcountll(_bTIGs[1] & tig1) << 1) - bcT1;
    INT64 bc12 = ((__builtin_popcountll(_bTIGs[1] & tig2) << 1) - bcT2) << 1;
    INT64 bc14 = ((__builtin_popcountll(_bTIGs[1] & tig4) << 1) - bcT4) << 2;
    INT64 bc18 = ((__builtin_popcountll(_bTIGs[1] & tig8) << 1) - bcT8) << 3;

    return _coeffs1[0] * (bc01 + bc02 + bc04 + bc08) + _coeffs1[1] * (bc11 + bc12 + bc14 + bc18);

}
