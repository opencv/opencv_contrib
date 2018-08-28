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
 // Copyright (C) 2014, OpenCV Foundation, all rights reserved.
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

#include "precomp.hpp"

namespace cv
{
namespace saliency
{

/**
 * Fine Grained Saliency
 */


StaticSaliencyFineGrained::StaticSaliencyFineGrained()
{
    className = "FINE_GRAINED";
}

StaticSaliencyFineGrained::~StaticSaliencyFineGrained()
{

}


bool StaticSaliencyFineGrained::computeSaliencyImpl(InputArray image, OutputArray saliencyMap )
{
    Mat dst(Size(image.getMat().cols, image.getMat().rows), CV_8UC1);
    calcIntensityChannel(image.getMat(), dst);
    dst.convertTo(saliencyMap, CV_32F, 1.0f/255.0f); // values are in range [0; 1]

    #ifdef SALIENCY_DEBUG
    // visualize saliency map
    imshow( "Saliency Map Interna", saliencyMap );
    #endif

    return true;
}

void StaticSaliencyFineGrained::copyImage(Mat srcArg, Mat dstArg)
{
    srcArg.copyTo(dstArg);
}

void StaticSaliencyFineGrained::calcIntensityChannel(Mat srcArg, Mat dstArg)
{
    if(dstArg.channels() > 1)
    {
        //("Error: Destiny image must have only one channel.\n");
        return;
    }
    const int numScales = 6;
    Mat intensityScaledOn[numScales];
    Mat intensityScaledOff[numScales];
    Mat gray = Mat::zeros(Size(srcArg.cols, srcArg.rows), CV_8UC1);
    Mat integralImage(Size(srcArg.cols + 1, srcArg.rows + 1), CV_32FC1);
    Mat intensity(Size(srcArg.cols, srcArg.rows), CV_8UC1);
    Mat intensityOn(Size(srcArg.cols, srcArg.rows), CV_8UC1);
    Mat intensityOff(Size(srcArg.cols, srcArg.rows), CV_8UC1);

    int i;
    int neighborhood;
    int neighborhoods[] = {3*4, 3*4*2, 3*4*2*2, 7*4, 7*4*2, 7*4*2*2};

    for(i=0; i<numScales; i++)
    {
        intensityScaledOn[i] = Mat(Size(srcArg.cols, srcArg.rows), CV_8UC1);
        intensityScaledOff[i] = Mat(Size(srcArg.cols, srcArg.rows), CV_8UC1);
    }

    // Prepare the input image: put it into a grayscale image.
    if(srcArg.channels()==3)
    {
        cvtColor(srcArg, gray, COLOR_BGR2GRAY);
    }
    else
    {
        srcArg.copyTo(gray);
    }

    // smooth pixels at least twice, as done by Frintrop and Itti
    GaussianBlur( gray, gray, Size( 3, 3 ), 0, 0 );
    GaussianBlur( gray, gray, Size( 3, 3 ), 0, 0 );


    // Calculate integral image, only once.
    integral(gray, integralImage, CV_32F);


    for(i=0; i< numScales; i++)
    {
        neighborhood = neighborhoods[i] ;
        getIntensityScaled(integralImage, gray, intensityScaledOn[i], intensityScaledOff[i], neighborhood);
    }

    mixScales(intensityScaledOn, intensityOn, intensityScaledOff, intensityOff, numScales);

    mixOnOff(intensityOn, intensityOff, intensity);

    intensity.copyTo(dstArg);
}

void StaticSaliencyFineGrained::getIntensityScaled(Mat integralImage, Mat gray, Mat intensityScaledOn, Mat intensityScaledOff, int neighborhood)
{
    float value, meanOn, meanOff;
    Point2i point;
    int x,y;
    intensityScaledOn.setTo(Scalar::all(0));
    intensityScaledOff.setTo(Scalar::all(0));


    for(y = 0; y < gray.rows; y++)
    {
        for(x = 0; x < gray.cols; x++)
        {
            point.x = x;
            point.y = y;
            value = getMean(integralImage, point, neighborhood, gray.at<uchar>(y, x));

            meanOn = gray.at<uchar>(y, x) - value;
            meanOff = value - gray.at<uchar>(y, x);

            if(meanOn > 0)
                intensityScaledOn.at<uchar>(y, x) = (uchar)meanOn;
            else
                intensityScaledOn.at<uchar>(y, x) = 0;

            if(meanOff > 0)
                intensityScaledOff.at<uchar>(y, x) = (uchar)meanOff;
            else
                intensityScaledOff.at<uchar>(y, x) = 0;
        }
    }
}

float StaticSaliencyFineGrained::getMean(Mat srcArg, Point2i PixArg, int neighbourhood, int centerVal)
{
    Point2i P1, P2;
    float value;

    P1.x = PixArg.x - neighbourhood + 1;
    P1.y = PixArg.y - neighbourhood + 1;
    P2.x = PixArg.x + neighbourhood + 1;
    P2.y = PixArg.y + neighbourhood + 1;

    if(P1.x < 0)
        P1.x = 0;
    else if(P1.x > srcArg.cols - 1)
        P1.x = srcArg.cols - 1;
    if(P2.x < 0)
        P2.x = 0;
    else if(P2.x > srcArg.cols - 1)
        P2.x = srcArg.cols - 1;
    if(P1.y < 0)
        P1.y = 0;
    else if(P1.y > srcArg.rows - 1)
        P1.y = srcArg.rows - 1;
    if(P2.y < 0)
        P2.y = 0;
    else if(P2.y > srcArg.rows - 1)
        P2.y = srcArg.rows - 1;

    // we use the integral image to compute fast features
    value = (float) (
            (srcArg.at<float>(P2.y, P2.x)) +
            (srcArg.at<float>(P1.y, P1.x)) -
            (srcArg.at<float>(P2.y, P1.x)) -
            (srcArg.at<float>(P1.y, P2.x))
    );
    value = (value - centerVal)/  (( (P2.x - P1.x) * (P2.y - P1.y))-1)  ;
    return value;
}

void StaticSaliencyFineGrained::mixScales(Mat *intensityScaledOn, Mat intensityOn, Mat *intensityScaledOff, Mat intensityOff, const int numScales)
{
    int i=0, x, y;
    int width = intensityScaledOn[0].cols;
    int height = intensityScaledOn[0].rows;
    short int maxValOn = 0, currValOn=0;
    short int maxValOff = 0, currValOff=0;
    int maxValSumOff = 0, maxValSumOn=0;
    Mat mixedValuesOn(Size(width, height), CV_16UC1);
    Mat mixedValuesOff(Size(width, height), CV_16UC1);

    mixedValuesOn.setTo(Scalar::all(0));
    mixedValuesOff.setTo(Scalar::all(0));

    for(i=0;i<numScales;i++)
    {
        for(y=0;y<height;y++)
            for(x=0;x<width;x++)
            {
                      currValOn = intensityScaledOn[i].at<uchar>(y, x);
                      if(currValOn > maxValOn)
                          maxValOn = currValOn;

                      currValOff = intensityScaledOff[i].at<uchar>(y, x);
                      if(currValOff > maxValOff)
                          maxValOff = currValOff;

                      mixedValuesOn.at<unsigned short>(y, x) += currValOn;
                      mixedValuesOff.at<unsigned short>(y, x) += currValOff;
            }
    }

    for(y=0;y<height;y++)
        for(x=0;x<width;x++)
        {
            currValOn = mixedValuesOn.at<unsigned short>(y, x);
            currValOff = mixedValuesOff.at<unsigned short>(y, x);
                  if(currValOff > maxValSumOff)
                      maxValSumOff = currValOff;
                  if(currValOn > maxValSumOn)
                      maxValSumOn = currValOn;
        }


    for(y=0;y<height;y++)
        for(x=0;x<width;x++)
        {
            intensityOn.at<uchar>(y, x) = (uchar)(255.*((float)(mixedValuesOn.at<unsigned short>(y, x) / (float)maxValSumOn)));
            intensityOff.at<uchar>(y, x) = (uchar)(255.*((float)(mixedValuesOff.at<unsigned short>(y, x) / (float)maxValSumOff)));
        }

}

void StaticSaliencyFineGrained::mixOnOff(Mat intensityOn, Mat intensityOff, Mat intensityArg)
{
    int x,y;
    int width = intensityOn.cols;
    int height= intensityOn.rows;
    int maxVal=0;

    int currValOn, currValOff, maxValSumOff, maxValSumOn;

    Mat intensity(Size(width, height), CV_8UC1);


    maxValSumOff = 0;
    maxValSumOn = 0;

    for(y=0;y<height;y++)
    for(x=0;x<width;x++)
    {
        currValOn = intensityOn.at<uchar>(y, x);
        currValOff = intensityOff.at<uchar>(y, x);
              if(currValOff > maxValSumOff)
                  maxValSumOff = currValOff;
              if(currValOn > maxValSumOn)
                  maxValSumOn = currValOn;
    }

    if(maxValSumOn > maxValSumOff)
        maxVal = maxValSumOn;
    else
        maxVal = maxValSumOff;



    for(y=0;y<height;y++)
        for(x=0;x<width;x++)
        {
            intensity.at<uchar>(y, x) = (uchar) (255. * (float) (intensityOn.at<uchar>(y, x) + intensityOff.at<uchar>(y, x)) / (float)maxVal);
        }

    intensity.copyTo(intensityArg);
}


} /* namespace saliency */
}/* namespace cv */
