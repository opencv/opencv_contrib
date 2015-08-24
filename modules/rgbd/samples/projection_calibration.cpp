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
// Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2015, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
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

#include <opencv2/rgbd.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/videoio.hpp>
#include <opencv2/structured_light.hpp>
#include <opencv2/core/utility.hpp>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace cv;
using namespace cv::rgbd;
using namespace cv::structured_light;
using namespace std;

void captureImageMultipleTimes(Ptr<VideoCapture>& capture, Mat& image, Mat& gray, Mat& depth, int wait = 0, int n = 1);

void captureImageMultipleTimes(Ptr<VideoCapture>& capture, Mat& image, Mat& gray, Mat& depth, int wait, int n)
{
    // sometimes need to grab n times to get good images
    for (int i = 0; i < n; i++)
    {
        if (wait > 0)
        {
            waitKey(wait);
        }

        capture->grab();

        capture->retrieve(depth, CAP_OPENNI_DEPTH_MAP);
        capture->retrieve(image, CAP_OPENNI_BGR_IMAGE);
    }
    flip(depth, depth, 1);
    flip(image, image, 1);
    cvtColor(image, gray, COLOR_BGR2GRAY);
}

int main(int argc, char** argv)
{
    int devId;
    int lightThreshold;
    int lightIntensity;
    int monitorWidth;
    GrayCodePattern::Params params;
    bool useOpenni = false;

    const String keys =
        "{help h usage ? |       | print this message   }"
        "{id             |   0   | device ID            }"
        "{mw             |  -1   | monitor width        }"
        "{pw             |  -1   | projector width      }"
        "{ph             |  -1   | projector height     }"
        "{openni         |   0   | use OpenNI device    }"
        "{threshold      |   5   | noticable difference of inverted structured lights }"
        "{intensity      |  70   | intensity of structured lights }"
        ;
    CommandLineParser parser(argc, argv, keys);

    devId = parser.get<int>("id");
    monitorWidth = parser.get<int>("mw");
    params.width = parser.get<int>("pw");
    params.height = parser.get<int>("ph");
    useOpenni = parser.get<int>("openni") > 0;
    lightThreshold = parser.get<int>("threshold");
    lightIntensity = parser.get<int>("intensity");

    if (params.width > 0 && params.height > 0)
    {
        // fine; use command line arguments
    }
    else
    {
        // read from yml file
        FileStorage fs("capturer_parameters.yml", FileStorage::READ);
        fs["deviceId"] >> devId;
        fs["lightThreshold"] >> lightThreshold;
        fs["lightIntensity"] >> lightIntensity;
        fs["monitorWidth"] >> monitorWidth;
        fs["projectorWidth"] >> params.width;
        fs["projectorHeight"] >> params.height;
        fs["useOpenni"] >> useOpenni;
    }

    Ptr<VideoCapture> capture;
    if (useOpenni)
    {
        capture = makePtr<VideoCapture>(static_cast<int>(CAP_OPENNI2));
    }
    else
    {
        capture = makePtr<VideoCapture>(devId);
    }

    // set registeration on
    capture->set(CAP_PROP_OPENNI_REGISTRATION, 0.0);

    if (!capture->isOpened())
    {
        cout << "Camera unavailable" << endl;
        return -1;
    }

    Mat image, gray, depth;

    // initialize gray coding
    params.width /= 2;
    params.height /= 2;
    Ptr<GrayCodePattern> pattern = GrayCodePattern::create(params);

    vector<Mat> patternImages;
    pattern->generate(patternImages, Scalar(0, 0, 0), Scalar(1, 1, 1) * lightIntensity);

    string window = "pattern";
    namedWindow(window, WINDOW_NORMAL);
    imshow(window, patternImages.at(patternImages.size() / 2 - 1));
    moveWindow(window, monitorWidth, 0);
    setWindowProperty(window, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

    // window placement; wait for user
    for (;;)
    {
        captureImageMultipleTimes(capture, image, gray, depth, 0, 1);
        imshow("depth", depth * 10);
        imshow("camera", gray);

        int key = waitKey(30);
        if (key == 'f')
        {
            // TODO: 1px border when fullscreen on Windows (Surface?)
            setWindowProperty(window, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
        }
        else if (key == 'w')
        {
            setWindowProperty(window, WND_PROP_FULLSCREEN, WINDOW_NORMAL);
        }
        else if (key == ' ')
        {
            break;
        }
    }

    vector<Mat> cameraImages;
    // start structured lighting
    for (size_t i = 0; i < patternImages.size(); i++)
    {
        waitKey(50);

        imshow(window, patternImages.at(i));

        captureImageMultipleTimes(capture, image, gray, depth, 50, 5);
        imshow("camera", gray);

        Mat grayTemp;
        gray.copyTo(grayTemp);
        cameraImages.push_back(grayTemp);

        waitKey(50);
    }

    // capture color image for warping
    {
        imshow(window, Mat::zeros(params.height, params.width, CV_8U));
        waitKey(50);

        captureImageMultipleTimes(capture, image, gray, depth, 50, 5);
    }

    // decode
    pattern->setLightThreshold(lightThreshold);
    Mat correspondenceMapX = Mat(image.size(), CV_8UC3, Scalar(255, 255, 255));
    Mat correspondenceMapY = Mat(image.size(), CV_8UC3, Scalar(255, 255, 255));
    Size projectorSize(params.width * 2, params.height * 2);
    Mat projectorImage = Mat::zeros(projectorSize, CV_8UC3);
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            Point point;

            const bool error = true;
            if (pattern->getProjPixel(cameraImages, x, y, point) == error)
            {
                continue;
            }

            if (point.x >= params.width - 1 || point.y >= params.height - 1 || point.x < 1 || point.y < 1)
            {
                continue;
            }

            point *= 2;

            Range xr(x, x + 1);
            Range yr(y, y + 1);
            // weird encoding to reliably recover the point...
            correspondenceMapX(yr, xr) = Scalar(0, (point.x & 0xFF00) / 256, (point.x & 0xFF));
            correspondenceMapY(yr, xr) = Scalar(0, (point.y & 0xFF00) / 256, (point.y & 0xFF));

            Range xp(point.x, point.x + 1);
            Range yp(point.y, point.y + 1);
            Vec3b color = image.at<Vec3b>(y, x);
            projectorImage(yp, xp) = Scalar(color);
        }
    }

    imshow("correspondence X", correspondenceMapX);
    imshow("correspondence Y", correspondenceMapY);

    imshow("Projector Image", projectorImage);

    imwrite("correspondenceX.png", correspondenceMapX);
    imwrite("correspondenceY.png", correspondenceMapY);

    imwrite("projectorImage.png", projectorImage);

    waitKey(0);
    return 0;
}
