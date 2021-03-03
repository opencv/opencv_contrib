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
#ifndef POINT_HPP
#define POINT_HPP

#include <vector>

namespace logos
{
struct MatchPoint
{
    float sd;
    int index;

    MatchPoint(float sd_, int idx) :
        sd(sd_), index(idx) {}
};

class Point
{
private:
    float x;
    float y;
    float orientation;
    float scale;
    std::vector<Point*> nnVector;
    bool nnFound;
    int label;

public:
    Point();
    Point(float x_, float y_, float orientation_, float scale_, int label_ = 0);

    inline float getx() const { return x; }
    inline float gety() const { return y; }
    inline float getOrientation() const { return orientation; }
    inline float getScale() const { return scale; }

    inline int getLabel() const { return label; }
    inline void setLabel(int label_) { label = label_; }

    inline void getNNVector(std::vector<Point*>& nnv) const { nnv = nnVector; }
    void matchLabel(int label, std::vector<Point*>& mNN);

    void nearestNeighbours(const std::vector<Point*>& vP, int index, int N);
    void nearestNeighboursNaive(const std::vector<Point*>& vP, int index, int N);

    void printPoint() const;
    void printNN() const;

    float squareDist(float x1, float y1, float x2, float y2);
};
}

#endif
