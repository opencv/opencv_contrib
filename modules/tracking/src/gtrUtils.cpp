/*///////////////////////////////////////////////////////////////////////////////////////
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

#include "gtrUtils.hpp"


namespace cv
{
namespace gtr
{

double generateRandomLaplacian(double b, double m)
{
    double t = (double)rand() / (RAND_MAX);
    double n = (double)rand() / (RAND_MAX);

    if (t > 0.5)
        return m + b*log(n);
    else
        return m - b*log(n);
}

Rect2f anno2rect(vector<Point2f> annoBB)
{
    Rect2f rectBB;
    rectBB.x = min(annoBB[0].x, annoBB[1].x);
    rectBB.y = min(annoBB[0].y, annoBB[2].y);
    rectBB.width = fabs(annoBB[0].x - annoBB[1].x);
    rectBB.height = fabs(annoBB[0].y - annoBB[2].y);

    return rectBB;
}

vector <TrainingSample> gatherFrameSamples(Mat prevFrame, Mat currFrame, Rect2f prevBB, Rect2f currBB)
{
    vector <TrainingSample> trainingSamples;
    Point2f currCenter, prevCenter;
    Rect2f targetPatchRect, searchPatchRect;
    Mat targetPatch, searchPatch;
    Mat prevFramePadded, currFramePadded;

    //Crop Target Patch

    //Padding

    //Previous frame GTBBs center
    prevCenter.x = prevBB.x + prevBB.width / 2;
    prevCenter.y = prevBB.y + prevBB.height / 2;

    targetPatchRect.width = (float)(prevBB.width*padTarget);
    targetPatchRect.height = (float)(prevBB.height*padTarget);
    targetPatchRect.x = (float)(prevCenter.x - prevBB.width*padTarget / 2.0 + targetPatchRect.width);
    targetPatchRect.y = (float)(prevCenter.y - prevBB.height*padTarget / 2.0 + targetPatchRect.height);

    copyMakeBorder(prevFrame, prevFramePadded, (int)targetPatchRect.height, (int)targetPatchRect.height, (int)targetPatchRect.width, (int)targetPatchRect.width, BORDER_REPLICATE);

    targetPatch = prevFramePadded(targetPatchRect);


    for (int i = 0; i < samplesInFrame; i++)
    {
        TrainingSample sample;

        //Current frame GTBBs center
        currCenter.x = (float)(currBB.x + currBB.width / 2.0);
        currCenter.y = (float)(currBB.y + currBB.height / 2.0);

        //Generate and add random Laplacian distribution (Scaling from target size)
        double dx, dy, ds;
        dx = generateRandomLaplacian(bX, 0)*prevBB.width;
        dy = generateRandomLaplacian(bY, 0)*prevBB.height;
        ds = generateRandomLaplacian(bS, 1);

        //Limit coefficients
        dx = min(dx, (double)prevBB.width);
        dx = max(dx, (double)-prevBB.width);
        dy = min(dy, (double)prevBB.height);
        dy = max(dy, (double)-prevBB.height);
        ds = min(ds, Ymax);
        ds = max(ds, Ymin);

        searchPatchRect.width = (float)(prevBB.width*padSearch*ds);
        searchPatchRect.height =(float)(prevBB.height*padSearch*ds);
        searchPatchRect.x = (float)(currCenter.x + dx - searchPatchRect.width / 2.0 + searchPatchRect.width);
        searchPatchRect.y = (float)(currCenter.y + dy - searchPatchRect.height / 2.0 + searchPatchRect.height);
        copyMakeBorder(currFrame, currFramePadded, (int)searchPatchRect.height, (int)searchPatchRect.height, (int)searchPatchRect.width, (int)searchPatchRect.width, BORDER_REPLICATE);
        searchPatch = currFramePadded(searchPatchRect);

        //Calculate Relative GTBB in search patch
        Rect2f relGTBB;
        relGTBB.width = currBB.width;
        relGTBB.height = currBB.height;
        relGTBB.x = currBB.x - searchPatchRect.x + searchPatchRect.width;
        relGTBB.y = currBB.y - searchPatchRect.y + searchPatchRect.height;

        //Link to the sample struct
        sample.targetPatch = targetPatch.clone();
        sample.searchPatch = searchPatch.clone();
        sample.targetBB = relGTBB;

        trainingSamples.push_back(sample);
    }

    return trainingSamples;
}

}
}
