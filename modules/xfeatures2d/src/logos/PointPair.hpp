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
#ifndef POINTPAIR_HPP
#define POINTPAIR_HPP

#include "Point.hpp"

namespace logos
{
class PointPair
{
private:
    Point* p;
    Point* q;
    int support;
    float relOri;
    float relScale;
    int pos1;
    int pos2;

    float angleDiff(float a1, float a2);

public:
    PointPair(Point* p_, Point* q_);

    void computeLocalSupport(std::vector<PointPair*>& pp, int N);

    void calculateInternalVariables();

    float getRelOri() const { return relOri; }
    float getRelScale() const { return relScale; }

    float getx1() const { return p->getx(); }
    float getx2() const { return q->getx(); }
    float gety1() const { return p->gety(); }
    float gety2() const { return q->gety(); }

    void addPositions(int pos1_, int pos2_) { pos1 = pos1_; pos2 = pos2_; }
    int getPos1() const { return pos1; }
    int getPos2() const { return pos2; }

    int getSupport() const { return support; }
    void setSupport(int support_) { support = support_; }
};
}

#endif
