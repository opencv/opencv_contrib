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
#include <iostream>
#include <algorithm> // std::sort
#include "Point.hpp"

namespace logos
{
static bool cMP(const MatchPoint& m, const MatchPoint& n)
{
    return (m.sd < n.sd);
}

Point::Point() :
    x(0), y(0), orientation(0), scale(1), nnVector(), nnFound(false), label(0)
{
}

Point::Point(float x_, float y_, float orientation_, float scale_, int label_) :
    x(x_), y(y_), orientation(orientation_), scale(scale_), nnVector(), nnFound(false), label(label_)
{
}

void Point::nearestNeighbours(const std::vector<Point*>& vP, int index, int N)
{
    nearestNeighboursNaive(vP, index, N);
}

void Point::nearestNeighboursNaive(const std::vector<Point*>& vP, int index, int N)
{
    // only want to calculate once.
    if (nnFound)
    {
        return;
    }

    std::vector<MatchPoint> minMatch;
    minMatch.reserve(vP.size());

    int i = 0;
    for (std::vector<Point*>::const_iterator it = vP.begin(); it != vP.end(); ++it, i++)
    {
        // A point is not it's own neighbour
        if (i == index)
        {
            continue;
        }
        float sd = squareDist(getx(), gety(), (*it)->getx(), (*it)->gety());
        MatchPoint mP(sd, i);
        minMatch.push_back(mP);
    }

    std::sort(minMatch.begin(), minMatch.end(), cMP);
    nnVector.resize(static_cast<size_t>(N));
    int count = 0;
    for (std::vector<MatchPoint>::const_iterator mmit = minMatch.begin(); count < N; ++mmit, count++)
    {
        nnVector[static_cast<size_t>(count)] = vP[static_cast<size_t>(mmit->index)];
    }

    nnFound = true;
}

void Point::matchLabel(int label_, std::vector<Point*>& matchNN)
{
    for (std::vector<Point*>::const_iterator nnIterator = nnVector.begin();
         nnIterator != nnVector.end(); ++nnIterator)
    {
        if ((*nnIterator)->label == label_)
        {
            matchNN.push_back(*nnIterator);
        }
    }
}

void Point::printPoint() const
{
    std::cout << getx() << " "
              << gety() << " " << getOrientation() << " "
              << getScale() << " " << getLabel() << std::endl;
}

void Point::printNN() const
{
    for(std::vector<Point*>::const_iterator nnIterator = nnVector.begin();
        nnIterator != nnVector.end(); ++nnIterator)
    {
        (*nnIterator)->printPoint();
    }
}

float Point::squareDist(float x1, float y1, float x2, float y2)
{
    return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
}
}
