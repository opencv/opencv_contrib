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
        Size size;
        Mat correspondenceMap;
        Mat pointcloud;
        Mat depth;
    };
    Calibration camera, projector;
    camera.size = Size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT));
    projector.size = Size(1024, 768);
    Mat extrinsics;

    FileStorage yfs("calibration.yml", FileStorage::READ);
    yfs["cameraIntrinsics"] >> camera.cameraMatrix;
    yfs["projectorIntrinsics"] >> projector.cameraMatrix;
    yfs["cameraDistCoeffs"] >> camera.distCoeffs;
    yfs["projectorDistCoeffs"] >> projector.distCoeffs;
    yfs["projectorExtrinsics"] >> extrinsics;

    Mat image;

    // initialize gray coding
    GrayCodePattern::Params params;
    params.width = projector.size.width;
    params.height = projector.size.height;
    Ptr<GrayCodePattern> pattern = GrayCodePattern::create(params);

    vector<Mat> patternImages;
    pattern->generate(patternImages, Scalar(0, 0, 0), Scalar(100, 100, 100));

    string window = "pattern";
    namedWindow(window, WINDOW_NORMAL);
    moveWindow(window, 0, 0);
    imshow(window, Mat::zeros(projector.size, CV_8U));

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

    vector<Mat> cameraImages;

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
    camera.correspondenceMap = Mat::zeros(camera.size, CV_8UC3);
    projector.correspondenceMap = Mat::zeros(projector.size, CV_8UC3);
    camera.pointcloud = Mat::zeros(camera.size, CV_64FC3);
    camera.depth = Mat::zeros(camera.size, CV_64F);
    projector.depth = Mat::zeros(projector.size, CV_64F);
    Mat udepthP = Mat::zeros(projector.size, CV_8U);
    Mat udepthC = Mat::zeros(camera.size, CV_8U);
    Mat kCam = camera.cameraMatrix * Mat_<double>::eye(3, 4);
    Mat kPro = projector.cameraMatrix * extrinsics(Rect(0, 0, 4, 3));
    cout << kCam << endl;
    cout << kPro << endl;
    for (int y = 0; y < image.rows; y+=4) {
        for (int x = 0; x < image.cols; x+=4) {
            Point point;

            const bool error = true;
            pattern->getProjPixel(cameraImages, x, y, point); 
            // error detection in getProjPixel is too strict
            //if (pattern->getProjPixel(cameraImages, x, y, point) == error)
            if (0 > point.x || point.x >= projector.size.width || 0 > point.y || point.y >= projector.size.height)
            {
                continue;
            }

            Range xr(x, x + 1);
            Range yr(y, y + 1);
            camera.correspondenceMap(yr, xr) = Scalar(point.x * 255 / projector.size.width, point.y * 255 / projector.size.height, 0);

            xr = Range(point.x, point.x + 1);
            yr = Range(point.y, point.y + 1);
            projector.correspondenceMap(yr, xr) = Scalar(x * 255 / camera.size.width, y * 255 / camera.size.height, 0);

            Point3d p;
            Mat m = Mat(4, 4, CV_64F);
            Mat uCam = (Mat_<double>(3, 1) << x, y, 1);
            Mat uPro = (Mat_<double>(3, 1) << point.x, point.y, 1);
            uCam *= sqrt(2.0) / norm(uCam, NORM_L2);
            uPro *= sqrt(2.0) / norm(uPro, NORM_L2);

            m.at<double>(0, 0) = uCam.at<double>(0) * kCam.at<double>(2, 0) - uCam.at<double>(2) * kCam.at<double>(0, 0);
            m.at<double>(0, 1) = uCam.at<double>(0) * kCam.at<double>(2, 1) - uCam.at<double>(2) * kCam.at<double>(0, 1);
            m.at<double>(0, 2) = uCam.at<double>(0) * kCam.at<double>(2, 2) - uCam.at<double>(2) * kCam.at<double>(0, 2);
            m.at<double>(0, 3) = uCam.at<double>(0) * kCam.at<double>(2, 3) - uCam.at<double>(2) * kCam.at<double>(0, 3);

            m.at<double>(1, 0) = uCam.at<double>(1) * kCam.at<double>(2, 0) - uCam.at<double>(2) * kCam.at<double>(1, 0);
            m.at<double>(1, 1) = uCam.at<double>(1) * kCam.at<double>(2, 1) - uCam.at<double>(2) * kCam.at<double>(1, 1);
            m.at<double>(1, 2) = uCam.at<double>(1) * kCam.at<double>(2, 2) - uCam.at<double>(2) * kCam.at<double>(1, 2);
            m.at<double>(1, 3) = uCam.at<double>(1) * kCam.at<double>(2, 3) - uCam.at<double>(2) * kCam.at<double>(1, 3);

            m.at<double>(2, 0) = uPro.at<double>(0) * kPro.at<double>(2, 0) - uPro.at<double>(2) * kPro.at<double>(0, 0);
            m.at<double>(2, 1) = uPro.at<double>(0) * kPro.at<double>(2, 1) - uPro.at<double>(2) * kPro.at<double>(0, 1);
            m.at<double>(2, 2) = uPro.at<double>(0) * kPro.at<double>(2, 2) - uPro.at<double>(2) * kPro.at<double>(0, 2);
            m.at<double>(2, 3) = uPro.at<double>(0) * kPro.at<double>(2, 3) - uPro.at<double>(2) * kPro.at<double>(0, 3);

            m.at<double>(3, 0) = uPro.at<double>(1) * kPro.at<double>(2, 0) - uPro.at<double>(2) * kPro.at<double>(1, 0);
            m.at<double>(3, 1) = uPro.at<double>(1) * kPro.at<double>(2, 1) - uPro.at<double>(2) * kPro.at<double>(1, 1);
            m.at<double>(3, 2) = uPro.at<double>(1) * kPro.at<double>(2, 2) - uPro.at<double>(2) * kPro.at<double>(1, 2);
            m.at<double>(3, 3) = uPro.at<double>(1) * kPro.at<double>(2, 3) - uPro.at<double>(2) * kPro.at<double>(1, 3);

            SVD svd;
            Mat w, u, vt;
            svd.compute(m, w, u, vt, SVD::Flags::FULL_UV);
            Mat nullVector = Mat_<double>::zeros(4, 1);
            nullVector.at<double>(0) = vt.at<double>(3, 0);
            nullVector.at<double>(1) = vt.at<double>(3, 1);
            nullVector.at<double>(2) = vt.at<double>(3, 2);
            nullVector.at<double>(3) = vt.at<double>(3, 3);
            p.x = nullVector.at<double>(0) / nullVector.at<double>(3);
            p.y = nullVector.at<double>(1) / nullVector.at<double>(3);
            p.z = nullVector.at<double>(2) / nullVector.at<double>(3);

            if (p.z < 0) p *= -1;

            camera.pointcloud.at<Point3d>(y, x) = p * 0.001;
            camera.depth.at<double>(y, x) = p.z * 0.001;
            udepthC.at<uchar>(y, x) = p.z * 0.1;

            Mat pPro = (Mat_<double>(4, 1) << p.x, p.y, p.z, 1);
            pPro = extrinsics * pPro;
            pPro *= 1.0 / pPro.at<double>(3);

            if (pPro.at<double>(2) < 0) pPro *= -1;

            projector.depth.at<double>(point.y, point.x) = pPro.at<double>(2) * 0.001;
            udepthP.at<uchar>(point.y, point.x) = pPro.at<double>(2) * 0.1;
        }
    }
    Mat depthMapC, depthMapP;
    applyColorMap(udepthC, depthMapC, COLORMAP_HOT);
    applyColorMap(udepthP, depthMapP, COLORMAP_HOT);
    imshow("correspondence_camera", camera.correspondenceMap);
    imshow("correspondence_projector", projector.correspondenceMap);
    imshow("pointcloud", camera.pointcloud);
    imshow(window, depthMapP);
    imshow("depth_camera", depthMapC);
    waitKey(0);
    return 0;
}
