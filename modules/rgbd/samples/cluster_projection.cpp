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
#include <opencv2/core/utility.hpp>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace cv;
using namespace cv::rgbd;
using namespace std;

int main( int argc, char** argv )
{
    Mat projectorPixels;
    Mat correspondenceMapX, correspondenceMapY;
    correspondenceMapX = imread("correspondenceX.png");
    correspondenceMapY = imread("correspondenceY.png");

    Mat image, depth;
    float focalLength;
    Mat cameraMatrix = Mat::eye(3, 3, CV_32FC1);

    if (argc != 2) {
        // bad arguments
        argv[1] = "OPENNI";
        //exit(1);
    }

    if (string(argv[1]) == "OPENNI")
    {
        VideoCapture capture(CAP_OPENNI2);

        if (!capture.isOpened())
        {
            cout << "OpenNI2 device unavailable" << endl;
            return -1;
        }

        // set registeration on
        capture.set(CAP_PROP_OPENNI_REGISTRATION, 0.0);

        // disable mirroring not working??
        // capture.set(CAP_PROP_OPENNI2_MIRROR, 0.0);

        focalLength = static_cast<float>(capture.get(CAP_PROP_OPENNI_FOCAL_LENGTH));

        for (int i = 0; i < 5; i++)
        {
            capture.grab();

            capture.retrieve(image, CAP_OPENNI_BGR_IMAGE);
            imshow("Color", image);

            capture.retrieve(depth, CAP_OPENNI_DEPTH_MAP);
            imshow("Depth", depth * 8);

            waitKey(30);
        }

        flip(image, image, 1);
        flip(depth, depth, 1);

        depth.convertTo(depth, CV_32F);
        depth = depth * 0.001f; // [mm] to [m]

        // kinect parameter to focal length
        // https://github.com/OpenKinect/libfreenect/blob/master/src/registration.c#L317
        float fx = focalLength,
            fy = focalLength,
            cx = 319.5f,
            cy = 239.5f;

        cameraMatrix.at<float>(0, 0) = fx;
        cameraMatrix.at<float>(1, 1) = fy;
        cameraMatrix.at<float>(0, 2) = cx;
        cameraMatrix.at<float>(1, 2) = cy;

        cout << cameraMatrix << endl;
    }
    else
    {
        // read depth data from libfreenect
        cv::FileStorage file(string(argv[1]), cv::FileStorage::READ);

        float pixelSize, refDistance;
        file["depth"] >> depth;
        file["cameraMatrix"] >> cameraMatrix;
        depth.convertTo(depth, CV_32F);
    }

    Ptr<RgbdFrame> frame = makePtr<RgbdFrame>(image, depth);
    frame->cameraMatrix = cameraMatrix;

    Ptr<DepthCleaner> cleaner = makePtr<DepthCleaner>(CV_32F, 5);
    Mat tmp;
    (*cleaner)(frame->depth, tmp);
    frame->depth = tmp;

    depthTo3d(*frame);

    // generate valid point mask for clusters
    compare(frame->depth, 0, frame->mask, CMP_GT);
    projectorPixels = Mat::zeros(correspondenceMapX.size(), CV_32SC2);
    for (int i = 0; i < frame->mask.rows; i++)
    {
        for (int j = 0; j < frame->mask.cols; j++)
        {
            if (frame->mask.at<uchar>(i, j) == 0)
                continue;

            Vec3b sx = correspondenceMapX.at<Vec3b>(i, j);
            Vec3b sy = correspondenceMapY.at<Vec3b>(i, j);
            if (sx[0] == 255
                && sx[1] == 255
                && sx[2] == 255)
            {
                projectorPixels.at<Point2i>(i, j) = Point2i(-1, -1);
            }
            else
            {
                projectorPixels.at<Point2i>(i, j) = Point2i((int)sx[1] * 256 + (int)sx[2], (int)sy[1] * 256 + (int)sy[2]);
            }
        }
    }

    imshow("depth", frame->points3d);
    waitKey(30);

    RgbdClusterMesh mainCluster(frame);
    vector<RgbdClusterMesh> clusters;
    planarSegmentation(mainCluster, clusters, 2);
    deleteEmptyClusters(clusters);

    for (std::size_t i = 0; i < clusters.size(); i++) {
        Mat labels;
        Mat stats;
        Mat centroids;

        if (clusters.at(i).bPlane) {
            stringstream ss;
            ss << "cluster" << i;
            // downsample by 2x
            clusters.at(i).increment_step = 2;
            clusters.at(i).projectorPixels = projectorPixels;
            clusters.at(i).calculatePoints(true);
            clusters.at(i).unwrapTexCoord();
            clusters.at(i).save(ss.str() + ".obj");
            imshow(ss.str(), clusters.at(i).silhouette * 255);
            continue;
        }

        vector<RgbdClusterMesh> smallClusters;
        euclideanClustering(clusters.at(i), smallClusters);
        //deleteEmptyClusters(smallClusters);
        for (std::size_t j = 0; j < smallClusters.size(); j++) {
            stringstream ss;
            ss << "mesh_" << i << "_" << j;
            imshow(ss.str(), smallClusters.at(j).silhouette * 255);

            // no downsample
            smallClusters.at(j).increment_step = 1;
            smallClusters.at(j).projectorPixels = projectorPixels;
            smallClusters.at(j).calculatePoints(true);
            if (smallClusters.at(j).getNumPoints() == 0)
            {
                continue;
            }
            smallClusters.at(j).unwrapTexCoord();
            smallClusters.at(j).save(ss.str() + ".obj");
        }
    }

    waitKey(0);
    return 0;
}
