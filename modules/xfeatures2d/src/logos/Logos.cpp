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
#include <cmath>
#include "Logos.hpp"
#include <opencv2/core.hpp>

namespace logos
{
Logos::Logos()
{
    LogosParameters defaultParams;
    init(defaultParams);
}

Logos::Logos(const LogosParameters& p)
{
    init(p);
}

void Logos::init(const LogosParameters& p)
{
    setParams(p);
    LB = static_cast<float>(-CV_PI);
    BINSIZE = logosParams.GLOBALORILIMIT/3;
    BINNUMBER = static_cast<unsigned int>(ceil(2*CV_PI/BINSIZE));
    bins.resize(BINNUMBER);
    std::fill(bins.begin(), bins.end(), 0);
}

int Logos::estimateMatches(std::vector<Point*> vP1, std::vector<Point*> vP2, std::vector<PointPair*>& globalmatches)
{
    matches.clear();

    // for each point
    int count1 = 0;

    for (std::vector<Point*>::iterator pit1 = vP1.begin(); pit1 != vP1.end(); ++pit1, count1++)
    {
        (*pit1)->nearestNeighbours(vP1, count1, getNum1());
        int count2 = 0;

        // find possible matches
        for (std::vector<Point*>::iterator pit2 = vP2.begin(); pit2 != vP2.end(); ++pit2, count2++)
        {
            if ((*pit1)->getLabel() != (*pit2)->getLabel())
            {
                continue;
            }
            // this is a possible match in Image 2
            // get nearest neighbours
            (*pit2)->nearestNeighbours(vP2, count2, getNum2());

            PointPair* ptpr = new PointPair(*pit1, *pit2);
            ptpr->addPositions(count1, count2);
            ptpr->computeLocalSupport(pp, getNum2());

            // calc matches
            int support = 0;
            for (std::vector<PointPair*>::const_iterator it = pp.begin(); it < pp.end(); ++it)
            {
                Match m(ptpr, *it);
                if (evaluateMatch(m))
                {
                    support++;
                }
            }
            for (size_t i = 0; i < pp.size(); i++)
            {
                delete pp[i];
            }
            pp.clear();
            if (support > 0)
            {
                ptpr->setSupport(support);
                matches.push_back(ptpr);
                updateBin(ptpr->getRelOri());
            }
            else
            {
                delete ptpr;
                ptpr = NULL;
            }
        }
    }

    // do global orientation
    double maxang = calcGlobalOrientation();

    // find which matches are within global orientation limit
    int numinliers = 0;
    globalmatches.clear();
    for (std::vector<PointPair*>::iterator it = matches.begin(); it != matches.end(); ++it)
    {
        if (std::fabs((*it)->getRelOri() - maxang) < logosParams.GLOBALORILIMIT)
        {
            numinliers++;
            globalmatches.push_back(*it);
        }
        else
        {
            delete *it;
            *it = NULL;
        }
    }

    return numinliers;
}

bool Logos::evaluateMatch(const Match& m) const
{
    return ((m.getRelOrientation() < getIntraOriLimit()) &&
            (m.getRelScale() < getIntraScaleLimit()) &&
            (m.getInterOrientation() < getInterOriLimit()) &&
            (m.getInterScale() < getInterScaleLimit()));
}

void Logos::updateBin(float input)
{
    unsigned int binnumber = static_cast<unsigned int>(cvFloor((input-LB) / BINSIZE));
    // compare binnumber to BINNUMBER
    if (binnumber < BINNUMBER)
    {
        bins[binnumber]++;
    }
    else
    {
        bins[BINNUMBER-1]++;
    }
}

float Logos::calcGlobalOrientation()
{
    // find max bin
    // check BINNUMBER is big enough
    if (BINNUMBER < 3)
    {
        return 0;
    }

    std::vector<int> bins2(BINNUMBER);
    int maxval = 0;
    unsigned int maxix = 0;
    bins2[0] = bins[0] + bins[1] + bins[BINNUMBER-1];
    maxval = bins2[0];
    for (unsigned int i = 1; i < BINNUMBER; i++)
    {
        if (i == BINNUMBER-1)
        {
            bins2[i] = bins[i]+bins[i-1]+bins[0];
        }
        else
        {
            bins2[i] = bins[i]+bins[i-1]+bins[i+1];
        }
        if (bins2[i] > maxval)
        {
            maxval = bins2[i];
            maxix = i;
        }
    }

    // convert to an angle
    return LB + maxix*BINSIZE + BINSIZE/2;
}
}
