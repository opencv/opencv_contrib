/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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

/**
 * @file   synthetic_seq.cpp
 * @author Vladislav Samsonov <vvladxx@gmail.com>
 * @brief  Synthetic frame sequence generator for testing background subtraction algorithms.
 *
*/

#include "precomp.hpp"

namespace cv
{
namespace bgsegm
{
namespace
{

inline int clamp(int x, int l, int u) {
    return ((x) < (l)) ? (l) : (((x) > (u)) ? (u) : (x));
}

inline int within(int a, int b, int c) {
    return (((a) <= (b)) && ((b) <= (c))) ? 1 : 0;
}

void bilinearInterp(uchar* dest, double x, double y, unsigned bpp, const uchar** values) {
    x = std::fmod(x, 1.0);
    y = std::fmod(y, 1.0);

    if (x < 0.0)
        x += 1.0;
    if (y < 0.0)
        y += 1.0;

    for (unsigned i = 0; i < bpp; i++) {
        double m0 = (1.0 - x) * values[0][i] + x * values[1][i];
        double m1 = (1.0 - x) * values[2][i] + x * values[3][i];
        dest[i] = (uchar) ((1.0 - y) * m0 + y * m1);
    }
}

// Static background is a way too easy test. We will add distortion to it.
void waveDistortion(const uchar* src, uchar* dst, int width, int height, int bypp, double amplitude, double wavelength, double phase) {
    const uchar zeroes[4] = {0, 0, 0, 0};
    const long rowsiz = width * bypp;
    const double xhsiz = (double) width / 2.0;
    const double yhsiz = (double) height / 2.0;
    double xscale, yscale;

    if (xhsiz < yhsiz) {
        xscale = yhsiz / xhsiz;
        yscale = 1.0;
    }
    else if (xhsiz > yhsiz) {
        xscale = 1.0;
        yscale = xhsiz / yhsiz;
    }
    else {
        xscale = 1.0;
        yscale = 1.0;
    }

    wavelength *= 2;

    for (int y = 0; y < height; y++) {
        uchar* dest = dst;

        for (int x = 0; x < width; x++) {
            const double dx = x * xscale;
            const double dy = y * yscale;
            const double d = sqrt (dx * dx + dy * dy);
            const double amnt = amplitude * sin(((d / wavelength) * (2.0 * M_PI) + phase));
            const double needx = (amnt + dx) / xscale;
            const double needy = (amnt + dy) / yscale;
            const int xi = clamp(int(needx), 0, width - 2);
            const int yi = clamp(int(needy), 0, height - 2);

            const uchar* p = src + rowsiz * yi + xi * bypp;

            const int x1_in = within(0, xi, width - 1);
            const int y1_in = within(0, yi, height - 1);
            const int x2_in = within(0, xi + 1, width - 1);
            const int y2_in = within(0, yi + 1, height - 1);
            const uchar* values[4];

            if (x1_in && y1_in)
                values[0] = p;
            else
                values[0] = zeroes;

            if (x2_in && y1_in)
                values[1] = p + bypp;
            else
                values[1] = zeroes;

            if (x1_in && y2_in)
                values[2] = p + rowsiz;
            else
                values[2] = zeroes;

            if (x2_in && y2_in)
                values[3] = p + bypp + rowsiz;
            else
                values[3] = zeroes;

            bilinearInterp(dest, needx, needy, bypp, values);
            dest += bypp;
        }

        dst += rowsiz;
    }
}

}

SyntheticSequenceGenerator::SyntheticSequenceGenerator(InputArray _background, InputArray _object, double _amplitude, double _wavelength, double _wavespeed, double _objspeed)
: amplitude(_amplitude), wavelength(_wavelength), wavespeed(_wavespeed), objspeed(_objspeed), timeStep(0) {
    _background.getMat().copyTo(background);
    _object.getMat().copyTo(object);

    if (background.channels() == 1) {
        cvtColor(background, background, COLOR_GRAY2BGR);
    }

    if (object.channels() == 1) {
        cvtColor(object, object, COLOR_GRAY2BGR);
    }

    CV_Assert(background.channels() == 3);
    CV_Assert(object.channels() == 3);
    CV_Assert(background.size().width > object.size().width);
    CV_Assert(background.size().height > object.size().height);

    background.convertTo(background, CV_8U);
    object.convertTo(object, CV_8U);

    pos.x = (background.size().width - object.size().width) / 2;
    pos.y = (background.size().height - object.size().height) / 2;

    const double phi = rng.uniform(0.0, CV_2PI);
    dir.x = std::cos(phi);
    dir.y = std::sin(phi);
}

void SyntheticSequenceGenerator::getNextFrame(OutputArray _frame, OutputArray _gtMask) {
    CV_Assert(!background.empty() && !object.empty());
    const Size sz = background.size();

    _frame.create(sz, CV_8UC3);
    Mat frame = _frame.getMat();

    CV_Assert(background.isContinuous() && frame.isContinuous());

    waveDistortion(background.ptr(), frame.ptr(), sz.width, sz.height, 3, amplitude, wavelength, double(timeStep) * wavespeed);

    const Size objSz = object.size();

    object.copyTo(frame(Rect(Point2i(pos), objSz)));

    while (pos.x + dir.x * objspeed < 0 || pos.x + dir.x * objspeed >= sz.width - objSz.width || pos.y + dir.y * objspeed < 0 || pos.y + dir.y * objspeed >= sz.height - objSz.height) {
        const double phi = rng.uniform(0.0, CV_2PI);
        dir.x = std::cos(phi);
        dir.y = std::sin(phi);
    }

    _gtMask.create(sz, CV_8U);
    Mat gtMask = _gtMask.getMat();
    gtMask.setTo(cv::Scalar::all(0));
    gtMask(Rect(Point2i(pos), objSz)) = 255;

    pos += dir * objspeed;
    ++timeStep;
}

Ptr<SyntheticSequenceGenerator> createSyntheticSequenceGenerator(InputArray background, InputArray object, double amplitude, double wavelength, double wavespeed, double objspeed) {
    return makePtr<SyntheticSequenceGenerator>(background, object, amplitude, wavelength, wavespeed, objspeed);
}

}
}
