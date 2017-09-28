/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *  
 *  
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *  
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *  
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *  
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *  
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *  
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
using namespace cv;
using namespace cv::ximgproc;

#include <iostream>
using namespace std;

typedef void(*FilteringOperation)(const Mat& src, Mat& dst);
//current mode (filtering operation example)
FilteringOperation g_filterOp = NULL;

//list of filtering operations
void filterDoNothing(const Mat& frame, Mat& dst);
void filterBlurring(const Mat& frame, Mat& dst);
void filterStylize(const Mat& frame, Mat& dst);
void filterDetailEnhancement(const Mat& frame8u, Mat& dst);

//common sliders for every mode
int g_sigmaColor = 25;
int g_sigmaSpatial = 10;

//for Stylizing mode
int g_edgesGamma = 100;

//for Details Enhancement mode
int g_contrastBase = 100;
int g_detailsLevel = 100;

int g_numberOfCPUs = cv::getNumberOfCPUs();

//We will use two callbacks to change parameters
void changeModeCallback(int state, void *filter);
void changeNumberOfCpuCallback(int count, void*);

void splitScreen(const Mat& rawFrame, Mat& outputFrame, Mat& srcFrame, Mat& processedFrame);

//trivial filter
void filterDoNothing(const Mat& frame, Mat& dst)
{
    frame.copyTo(dst);
}

//simple edge-aware blurring
void filterBlurring(const Mat& frame, Mat& dst)
{
    dtFilter(frame, frame, dst, g_sigmaSpatial, g_sigmaColor, DTF_RF);
}

//stylizing filter
void filterStylize(const Mat& frame, Mat& dst)
{
    //blur frame
    Mat filtered;
    dtFilter(frame, frame, filtered, g_sigmaSpatial, g_sigmaColor, DTF_NC);

    //compute grayscale blurred frame
    Mat filteredGray;
    cvtColor(filtered, filteredGray, COLOR_BGR2GRAY);

    //find gradients of blurred image
    Mat gradX, gradY;
    Sobel(filteredGray, gradX, CV_32F, 1, 0, 3, 1.0/255);
    Sobel(filteredGray, gradY, CV_32F, 0, 1, 3, 1.0/255);

    //compute magnitude of gradient and fit it accordingly the gamma parameter
    Mat gradMagnitude;
    magnitude(gradX, gradY, gradMagnitude);
    cv::pow(gradMagnitude, g_edgesGamma/100.0, gradMagnitude);

    //multiply a blurred frame to the value inversely proportional to the magnitude
    Mat multiplier = 1.0/(1.0 + gradMagnitude);
    cvtColor(multiplier, multiplier, COLOR_GRAY2BGR);
    multiply(filtered, multiplier, dst, 1, dst.type());
}

void filterDetailEnhancement(const Mat& frame8u, Mat& dst)
{
    Mat frame;
    frame8u.convertTo(frame, CV_32F, 1.0/255);

    //Decompose image to 3 Lab channels
    Mat frameLab, frameLabCn[3];
    cvtColor(frame, frameLab, COLOR_BGR2Lab);
    split(frameLab, frameLabCn);

    //Generate progressively smoother versions of the lightness channel
    Mat layer0 = frameLabCn[0]; //first channel is original lightness
    Mat layer1, layer2;
    dtFilter(layer0, layer0, layer1, g_sigmaSpatial, g_sigmaColor, DTF_IC);
    dtFilter(layer1, layer1, layer2, 2*g_sigmaSpatial, g_sigmaColor, DTF_IC);

    //Compute detail layers
    Mat detailLayer1 = layer0 - layer1;
    Mat detailLayer2 = layer1 - layer2;

    double cBase = g_contrastBase / 100.0;
    double cDetails1 = g_detailsLevel / 100.0;
    double cDetails2 = 2.0 - g_detailsLevel / 100.0;

    //Generate lightness
    double meanLigtness = mean(frameLabCn[0])[0];
    frameLabCn[0]  = cBase*(layer2 - meanLigtness) + meanLigtness; //fit contrast of base (most blurred) layer
    frameLabCn[0] += cDetails1*detailLayer1; //add weighted sum of detail layers to new lightness
    frameLabCn[0] += cDetails2*detailLayer2; //

    //Update new lightness
    merge(frameLabCn, 3, frameLab);
    cvtColor(frameLab, frame, COLOR_Lab2BGR);
    frame.convertTo(dst, CV_8U, 255);
}

void changeModeCallback(int state, void *filter)
{
    if (state == 1)
        g_filterOp = (FilteringOperation) filter;
}

void changeNumberOfCpuCallback(int count, void*)
{
    count = std::max(1, count);
    cv::setNumThreads(count);
    g_numberOfCPUs = count;
}

//divide screen on two parts: srcFrame and processed Frame
void splitScreen(const Mat& rawFrame, Mat& outputFrame, Mat& srcFrame, Mat& processedFrame)
{
    int h = rawFrame.rows;
    int w = rawFrame.cols;
    int cn = rawFrame.channels();

    outputFrame.create(h, 2 * w, CV_MAKE_TYPE(CV_8U, cn));
    srcFrame = outputFrame(Range::all(), Range(0, w));
    processedFrame = outputFrame(Range::all(), Range(w, 2 * w));
    rawFrame.convertTo(srcFrame, srcFrame.type());
}

int main()
{
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cerr << "Capture device was not found" << endl;
        return -1;
    }

    namedWindow("Demo");
    displayOverlay("Demo", "Press Ctrl+P to show property window", 5000);
    
    //Thread trackbar
    cv::setNumThreads(g_numberOfCPUs); //speedup filtering
    createTrackbar("Threads", String(), &g_numberOfCPUs, cv::getNumberOfCPUs(), changeNumberOfCpuCallback);

    //Buttons to choose different modes
    createButton("Mode Details Enhancement", changeModeCallback, (void*)filterDetailEnhancement, QT_RADIOBOX, true);
    createButton("Mode Stylizing",           changeModeCallback, (void*)filterStylize,           QT_RADIOBOX, false);
    createButton("Mode Blurring",            changeModeCallback, (void*)filterBlurring,          QT_RADIOBOX, false);
    createButton("Mode DoNothing",           changeModeCallback, (void*)filterDoNothing,         QT_RADIOBOX, false);

    //sliders for Details Enhancement mode
    g_filterOp = filterDetailEnhancement; //set Details Enhancement as default filter
    createTrackbar("Detail contrast", String(), &g_contrastBase, 200);
    createTrackbar("Detail level" , String(), &g_detailsLevel, 200);
    
    //sliders for Stylizing mode
    createTrackbar("Style gamma", String(), &g_edgesGamma, 300);

    //sliders for every mode
    createTrackbar("Sigma Spatial", String(), &g_sigmaSpatial, 200);
    createTrackbar("Sigma Color"  , String(), &g_sigmaColor, 200);

    Mat rawFrame, outputFrame;
    Mat srcFrame, processedFrame;

    for (;;)
    {
        do
        {
            cap >> rawFrame;
        } while (rawFrame.empty());

        splitScreen(rawFrame, outputFrame, srcFrame, processedFrame);
        g_filterOp(srcFrame, processedFrame);

        imshow("Demo", outputFrame);

        if (waitKey(1) == 27) break;
    }

    return 0;
}
