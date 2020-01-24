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
#ifndef MATCH_HPP
#define MATCH_HPP

#include "PointPair.hpp"

namespace logos
{
class Match
{
private:
    PointPair* r;
    PointPair* s;
    float relOrientation;
    float relScale;
    float interOrientation;
    float interScale;

    // Internal variables
    float ro3;
    float rs3;
    float vijx;
    float vijy;
    float vmnx;
    float vmny;

    // Internal functions
    void calculateInternalVariables();
    void setRelOrientation();
    void setRelScale();
    float angleDiff(float a1,float a2);
    float angleAbsDiff(float a1, float a2);
    void interOrientationAndScale();
    int sign(float x);

public:
    Match(PointPair* r, PointPair* s);

    inline float getRelOrientation() const { return relOrientation; }
    inline float getRelScale() const { return relScale; }
    inline float getInterOrientation() const { return interOrientation; }
    inline float getInterScale() const { return interScale; }

    void printMatch() const;
};
}

#endif
