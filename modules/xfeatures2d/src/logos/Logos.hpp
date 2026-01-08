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
#ifndef LOGOS_HPP
#define LOGOS_HPP

#include "Point.hpp"
#include "Match.hpp"
#include "PointPair.hpp"

namespace logos
{
struct LogosParameters
{
    LogosParameters() :
        INTRAORILIMIT(0.1f), INTRASCALELIMIT(0.1f), INTERORILIMIT(0.1f), INTERSCALELIMIT(0.1f), GLOBALORILIMIT(0.1f),
        NUM1(5), NUM2(5) {}

    float INTRAORILIMIT;
    float INTRASCALELIMIT;
    float INTERORILIMIT;
    float INTERSCALELIMIT;
    float GLOBALORILIMIT;
    int NUM1;
    int NUM2;
};

class Logos
{
private:
    std::vector<PointPair*> pp;
    std::vector<PointPair*> matches;

    LogosParameters logosParams;
    float LB;
    float BINSIZE;
    unsigned int BINNUMBER;
    std::vector<int> bins;

public:
    Logos();
    Logos(const LogosParameters& p);

    void init(const LogosParameters& p);

    int estimateMatches(std::vector<Point*> vP1, std::vector<Point*> vP2, std::vector<PointPair*>& globalmatches);
    bool evaluateMatch(const Match& m) const;

    inline float getIntraOriLimit() const { return logosParams.INTRAORILIMIT; }
    inline float getIntraScaleLimit() const { return logosParams.INTRASCALELIMIT; }
    inline float getInterOriLimit() const { return logosParams.INTERORILIMIT; }
    inline float getInterScaleLimit() const { return logosParams.INTERSCALELIMIT; }
    inline float getGlobalOriLimit() const { return logosParams.GLOBALORILIMIT; }
    inline int getNum1() const { return logosParams.NUM1; }
    inline int getNum2() const { return logosParams.NUM2; }

    void updateBin(float input);
    float calcGlobalOrientation();

    inline void setParams(const LogosParameters& p) { logosParams = p; }
};
}

#endif
