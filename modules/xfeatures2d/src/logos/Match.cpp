// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
/*
 * MIT License
 *
 * Copyright (c) 2018 Stephanie Lowry
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <algorithm>    // std::min
#include <cmath>        // log, acos
#include <iostream>
#include <opencv2/core.hpp>
#include "Match.hpp"

namespace logos
{
Match::Match(PointPair* r_, PointPair* s_) :
    r(r_), s(s_)
{
    calculateInternalVariables();
    setRelOrientation();
    setRelScale();
    interOrientationAndScale();
}

void Match::calculateInternalVariables()
{
    vijx = r->getx1() - s->getx1();
    vijy = r->gety1() - s->gety1();
    vmnx = r->getx2() - s->getx2();
    vmny = r->gety2() - s->gety2();
}

void Match::setRelOrientation()
{
    relOrientation = angleAbsDiff(r->getRelOri(), s->getRelOri());
}

void Match::setRelScale()
{
    relScale = std::fabs(r->getRelScale() - s->getRelScale());
}

float Match::angleAbsDiff(float a1, float a2)
{
    float ad = std::fabs(a1-a2);
    while (ad > 2*CV_PI)
    {
        ad = static_cast<float>(ad-2*CV_PI);
    }
    ad = std::min(std::fabs(ad), std::fabs(static_cast<float>(2*CV_PI-std::fabs(ad))));

    return ad;
}

void Match::interOrientationAndScale()
{
    float cp =  vijx*vmny - vijy*vmnx; // analogous to 2D cross product
    float nmij = std::sqrt(vijx*vijx + vijy*vijy);
    float nmnm = std::sqrt(vmnx*vmnx + vmny*vmny);

    float fr = (vijx*vmnx+vijy*vmny) / (nmij*nmnm);  // numerator equivalent to dot product
    fr = std::min(std::max(fr, -1.0f), 1.0f);
    ro3 = std::acos(fr)*sign(cp);

    rs3 = std::log(nmij) - std::log(nmnm);

    interOrientation = angleAbsDiff(r->getRelOri(), ro3);
    interScale = std::fabs(r->getRelScale() - rs3);
}

void Match::printMatch() const
{
    std::cout << "Relative Orientation: " << relOrientation << std::endl;
    std::cout << "Relative Scale: " << relScale << std::endl;
    std::cout << "Inter Orientation: " << interOrientation << std::endl;
    std::cout << "Inter Scale: " << interScale << std::endl;
    std::cout << "Global Relative Orientation: " << r->getRelOri() << std::endl;
}

int Match::sign(float x)
{
    return (x > 0) - (x < 0);
}
}
