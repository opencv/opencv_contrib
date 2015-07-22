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

// Camera-projector calibration using checkerboard and structured light based on
// Daniel Moreno and Gabriel Taubin. Simple, Accurate, and Robust Projector-Camera Calibration.
// 3D Imaging, Modeling, Processing, 2012 Second International Conference on Visualization and Transmission (3DIMPVT). IEEE, 2012.

//#define USE_OPENNI2

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

int main(int argc, char** argv)
{
#ifdef USE_OPENNI2
    VideoCapture capture(CAP_OPENNI2);
    // set registeration on
    capture.set(CAP_PROP_OPENNI_REGISTRATION, 0.0);
    VideoCapture capture(CAP_OPENNI2);
#else
    VideoCapture capture(2);
    capture.set(CAP_PROP_FRAME_WIDTH, 1280); 
    capture.set(CAP_PROP_FRAME_HEIGHT, 720);
#endif

    if (!capture.isOpened())
    {
        cout << "Camera unavailable" << endl;
        return -1;
    }

    // storage for calibration input/output
    struct Calibration {
        Mat cameraMatrix;
        Mat distCoeffs;
        vector<vector<Point2f> > imagePoints;
    };
    Calibration camera, projector;

    // number of checkerboard poses to capture
    const int numSequences = 3;

    Mat image;

    // initialize checkerboard parameters
    vector<vector<Point3f> > objectPoints;
    cv::Size chessSize(9, 6);
    float chessDimension = 20; // [mm]
    {
        vector<Point3f> chessPoints;
        for (int i = 0; i < chessSize.height; i++)
        {
            for (int j = 0; j < chessSize.width; j++)
            {
                chessPoints.push_back(Point3f(j, i, 0) * chessDimension);
            }
        }
        for (int sequence = 0; sequence < numSequences; sequence++)
        {
            objectPoints.push_back(chessPoints);
        }
    }

    // initialize gray coding
    GrayCodePattern::Params params;
    params.width = 1024;
    params.height = 768;
    Ptr<GrayCodePattern> pattern = GrayCodePattern::create(params);

    vector<Mat> patternImages;
    pattern->generate(patternImages, Scalar(0, 0, 0), Scalar(100, 100, 100));

    string window = "pattern";
    namedWindow(window, WINDOW_NORMAL);
    moveWindow(window, 0, 0);
    imshow(window, Mat::zeros(params.height, params.width, CV_8U));

    // window placement; wait for user
    while (true)
    {
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

    // run structured light sequences for different checkerboard poses
    for (int sequence = 0; sequence < numSequences;)
    {
        vector<Mat> cameraImages;
        vector<Point2f> imagePointsCamera, imagePointsProjector;

        // detect checkerboard corners
        while (true)
        {
            capture.grab();
#ifdef USE_OPENNI2
            capture.retrieve(image, CAP_OPENNI_BGR_IMAGE);
            flip(image, image, 1);
#else
            capture.retrieve(image);
#endif
            bool patternWasFound;
            patternWasFound = findChessboardCorners(image, chessSize, imagePointsCamera);
            drawChessboardCorners(image, chessSize, imagePointsCamera, patternWasFound);

            imshow("camera", image);

            // if space key is pressed, proceed
            if (waitKey(30) == ' ')
            {
                break;
            }
        }

        // start structured lighting
        for (size_t i = 0; i < patternImages.size(); i++)
        {
            waitKey(50);

            imshow(window, patternImages.at(i));

            waitKey(50);

            for (int t = 0; t < 5; t++)
            {
                waitKey(50);
                capture.grab();

#ifdef USE_OPENNI2
                capture.retrieve(image, CAP_OPENNI_BGR_IMAGE);
                flip(image, image, 1);
#else
                capture.retrieve(image);
#endif
            }

            Mat gray;
            cvtColor(image, gray, COLOR_BGR2GRAY);
            imshow("camera", gray);
            cameraImages.push_back(gray);

            waitKey(50);
        }

        // decode
        // we care only the points within checkerboard
        Mat correspondenceMap = Mat::zeros(image.size(), CV_8UC3);
        vector<Point2d> homographyPointsCamera, homographyPointsProjector;
        for (size_t i = 0; i < imagePointsCamera.size(); i++)
        {
            Point point;

            int x = static_cast<int>(imagePointsCamera.at(i).x);
            int y = static_cast<int>(imagePointsCamera.at(i).y);

            const bool error = true;
            if (pattern->getProjPixel(cameraImages, x, y, point) == error)
            {
                //continue;
            }

            Range xr(x, x + 1);
            Range yr(y, y + 1);
            correspondenceMap(yr, xr) = Scalar(point.x * 255 / params.width, point.y * 255 / params.height, 0);

            Point2d p(x, y);
            homographyPointsCamera.push_back(p);
            homographyPointsProjector.push_back(point);
        }

        imshow("correspondence", correspondenceMap);

        // warp checkerboard corners to projector image plane
        Mat H = findHomography(homographyPointsCamera, homographyPointsProjector, RANSAC);
        for (int i = 0; i < imagePointsCamera.size(); i++)
        {
            Mat p = Mat(3, 1, CV_64F);
            p.at<double>(0) = imagePointsCamera.at(i).x;
            p.at<double>(1) = imagePointsCamera.at(i).y;
            p.at<double>(2) = 1;
            p = H * p;
            p *= (1.0f / p.at<double>(2));
            imagePointsProjector.push_back(Point2d(p.at<double>(0), p.at<double>(1)));
        }

        camera.imagePoints.push_back(imagePointsCamera);
        projector.imagePoints.push_back(imagePointsProjector);

        Mat warpedImagePoints = Mat::zeros(params.height, params.width, CV_8UC3);
        drawChessboardCorners(warpedImagePoints, chessSize, imagePointsProjector, true);
        imshow(window, warpedImagePoints);

        while (true)
        {
            int key = waitKey(30);
            if (key == ' ')
            {
                // proceed to next pose
                sequence++;
                break;
            }
            else if (key == 'c')
            {
                // discard and recapture
                break;
            }
        }

        imshow(window, Mat::zeros(params.height, params.width, CV_8U));
    }

/*    FileStorage tempfs("calibrationTemp.yml", FileStorage::WRITE);
    tempfs << "objectPoints" << Mat(objectPoints);
    tempfs << "imagePointsCamera" << Mat(camera.imagePoints);
    tempfs << "imagePointsProjector" << Mat(projector.imagePoints);
*/

    int flags = 0;
    Size imageSize = Size(params.width, params.height);
    Mat R, T, E, F;
    stereoCalibrate(objectPoints, projector.imagePoints, camera.imagePoints, projector.cameraMatrix, projector.distCoeffs,
        camera.cameraMatrix, camera.distCoeffs, imageSize, R, T, E, F, flags);

    cout << projector.cameraMatrix << endl;
    cout << projector.distCoeffs << endl;
    cout << R << endl;
    cout << T << endl;
    cout << objectPoints.size() << endl;
    Mat extrinsics = Mat::eye(4, 4, CV_64FC1);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            extrinsics.at<double>(i, j) = R.at<double>(i, j);
        }
        extrinsics.at<double>(i, 3) = T.at<double>(i);
    }

    // file output
    cout << extrinsics << endl;

    FileStorage fs("calibration.xml", FileStorage::WRITE);
    fs << "ProjectorCameraEnsemble";
    fs << "{";
    {
        fs << "name" << "OpenCV calibration";
        fs << "cameras";
        fs << "{";
        {
            fs << "Camera";
            fs << "{";
            {
                fs << "name" << 0;
                fs << "hostNameOrAddress" << "localhost";
                fs << "width" << static_cast<int>(capture.get(CAP_PROP_FRAME_WIDTH));
                fs << "height" << static_cast<int>(capture.get(CAP_PROP_FRAME_HEIGHT));
                fs << "cameraMatrix";
                fs << "{";
                {
                    fs << "ValuesByColumn";
                    fs << "{";
                    for (int j = 0; j < 3; j++)
                    {
                        fs << "ArrayOfDouble";
                        fs << "{";
                        for (int i = 0; i < 3; i++)
                        {
                            fs << "double" << camera.cameraMatrix.at<double>(i, j);
                        }
                        fs << "}";
                    }
                    fs << "}";
                }
                fs << "}";
                fs << "lensDistortion";
                fs << "{";
                {
                    fs << "ValuesByColumn";
                    fs << "{";
                    {
                        fs << "ArrayOfDouble";
                        fs << "{";
                        for (int i = 0; i < 2; i++)
                        {
                            fs << "double" << camera.distCoeffs.at<double>(i);
                        }
                        fs << "}";
                    }
                    fs << "}";
                }
                fs << "}";
            }
            fs << "}";
        }
        fs << "}";
        fs << "projectors";
        fs << "{";
        {
            fs << "Projector";
            fs << "{";
            {
                fs << "name" << 0;
                fs << "hostNameOrAddress" << "localhost";
                fs << "displayIndex" << 0;
                fs << "width" << params.width;
                fs << "height" << params.height;
                fs << "cameraMatrix";
                fs << "{";
                {
                    fs << "ValuesByColumn";
                    fs << "{";
                    for (int j = 0; j < 3; j++)
                    {
                        fs << "ArrayOfDouble";
                        fs << "{";
                        for (int i = 0; i < 3; i++)
                        {
                            fs << "double" << projector.cameraMatrix.at<double>(i, j);
                        }
                        fs << "}";
                    }
                    fs << "}";
                }
                fs << "}";
                fs << "lensDistortion";
                fs << "{";
                {
                    fs << "ValuesByColumn";
                    fs << "{";
                    {
                        fs << "ArrayOfDouble";
                        fs << "{";
                        for (int i = 0; i < 2; i++)
                        {
                            fs << "double" << projector.distCoeffs.at<double>(i);
                        }
                        fs << "}";
                    }
                    fs << "}";
                }
                fs << "}";
                fs << "pose";
                fs << "{";
                {
                    fs << "ValuesByColumn";
                    fs << "{";
                    for (int j = 0; j < 4; j++)
                    {
                        fs << "ArrayOfDouble";
                        fs << "{";
                        for (int i = 0; i < 4; i++)
                        {
                            fs << "double" << extrinsics.at<double>(i, j);
                        }
                        fs << "}";
                    }
                    fs << "}";
                }
                fs << "}";
            }
            fs << "}";
        }
        fs << "}";
    }
    fs << "}";

    waitKey(0);
    return 0;
}
