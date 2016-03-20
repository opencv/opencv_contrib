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
// Copyright (C) 2015, Itseez Inc, all rights reserved.
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
// In no event shall the Itseez Inc or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "dpm_nms.hpp"
#include <algorithm>

using namespace std;

namespace cv
{
namespace dpm
{

void NonMaximumSuppression::sort(const vector< double > x, vector< int > &indices)
{
    for (unsigned int i = 0; i < x.size(); i++)
    {
        for (unsigned int j = i + 1; j < x.size(); j++)
        {
            if (x[indices[j]] < x[indices[i]])
            {
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }
    }
}

void NonMaximumSuppression::process(vector< vector< double > > &detections, double overlapThreshold)
{
    int numBoxes = (int) detections.size();

    if (numBoxes <= 0)
        return;

    vector< double > area(numBoxes);
    vector< double > score(numBoxes);
    vector< int > indices(numBoxes);

    for (int i = 0; i < numBoxes; i++)
    {
        indices[i] = i;
        int s = (int)detections[i].size();
        double x1 = detections[i][0];
        double y1 = detections[i][1];
        double x2 = detections[i][2];
        double y2 = detections[i][3];
        double sc = detections[i][s-1];
        score[i] = sc;
        area[i] = (x2 - x1 + 1) * ( y2 - y1 + 1);
    }

    // sort boxes by score
    sort(score, indices);
    vector< int > pick;
    vector< int > suppress;

    while (indices.size() > 0)
    {
        int last = (int) indices.size() - 1;
        int i = indices[last];
        pick.push_back(i);
        suppress.clear();
        suppress.push_back(last);

        for (int k = 0; k <= last - 1; k++)
        {
            int j = indices[k];
            double xx1 = max(detections[i][0], detections[j][0]);
            double yy1 = max(detections[i][1], detections[j][1]);
            double xx2 = min(detections[i][2], detections[j][2]);
            double yy2 = min(detections[i][3], detections[j][3]);

            double w = xx2 - xx1 + 1;
            double h = yy2 - yy1 + 1;

            if (w > 0 && h > 0)
            {
                // compute overlap
                double o = w*h / area[j];
                if (o > overlapThreshold)
                    suppress.push_back(k);
            }
        } // k

        // remove suppressed indices
        vector< int > newIndices;
        for (unsigned int n = 0; n < indices.size(); n++)
        {
            bool isSuppressed = false;
            for (unsigned int r = 0; r < suppress.size(); r++)
            {
                if (n == (unsigned int)suppress[r])
                {
                    isSuppressed = true;
                    break;
                }
            }

            if (!isSuppressed)
                newIndices.push_back(indices[n]);
        }
        indices = newIndices;
    } // while

    vector< vector< double > > newDetections(pick.size());
    for (unsigned int i = 0; i < pick.size(); i++)
        newDetections[i] = detections[pick[i]];

    detections = newDetections;
}

} // dpm
} // cv
