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
#include <opencv2/core.hpp>
#include "PointPair.hpp"

namespace logos
{
PointPair::PointPair(Point* p_, Point* q_) :
    p(p_), q(q_), support(0), pos1(0), pos2(0)
{
    calculateInternalVariables();
}

void PointPair::computeLocalSupport(std::vector<PointPair*>& pp, int N)
{
    std::vector<Point*> nnVector;
    p->getNNVector(nnVector); // Exposes the nearest neighbours

    // for each nearest neighbour
    for (std::vector<Point*>::iterator nnIterator = nnVector.begin(); nnIterator != nnVector.end(); ++nnIterator)
    {
        // is there a matching nearestNeighbour?
        std::vector<Point*> matchNN;
        matchNN.reserve(static_cast<size_t>(N));
        q->matchLabel((*nnIterator)->getLabel(), matchNN);
        for (std::vector<Point*>::const_iterator mit = matchNN.begin(); mit != matchNN.end(); ++mit)
        {
            PointPair* m = new PointPair(*nnIterator, *mit);
            pp.push_back(m);
        }
    }
}

void PointPair::calculateInternalVariables()
{
    relOri = angleDiff(p->getOrientation(), q->getOrientation());
    relScale = std::log(p->getScale()) - std::log(q->getScale());
}

float PointPair::angleDiff(float a1, float a2)
{
    float ad = a1 - a2;
    while (ad > CV_PI)
    {
        ad = static_cast<float>(ad-2*CV_PI);
    }
    while (ad < -CV_PI)
    {
        ad = static_cast<float>(ad + 2*CV_PI);
    }
    return ad;
}
}
