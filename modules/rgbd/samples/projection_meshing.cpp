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
#include <opencv2/videoio.hpp>
#include <opencv2/core/utility.hpp>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace cv;
using namespace cv::rgbd;
using namespace std;

int main(int argc, char** argv)
{
    int devId;
    //int lightThreshold;
    //int lightIntensity;
    Size camSize, projSize;
    bool useOpenni;

    const String keys =
        "{help h usage ? |       | print this message   }"
        "{id             |   0   | device ID            }"
        "{w              |  -1   | projector width      }"
        "{h              |  -1   | projector height     }"
        "{openni         |   0   | use OpenNI device    }"
        ;
    CommandLineParser parser(argc, argv, keys);

    devId = parser.get<int>("id");
    projSize.width = parser.get<int>("w");
    projSize.height = parser.get<int>("h");
    useOpenni = parser.get<int>("openni") > 0;

    if (projSize.width > 0 && projSize.height > 0)
    {
        // fine; use command line arguments
    }
    else
    {
        // read from yml file
        FileStorage fs("capturer_parameters.yml", FileStorage::Mode::READ);
        fs["deviceId"] >> devId;
        //fs["lightThreshold"] >> lightThreshold;
        //fs["lightIntensity"] >> lightIntensity;
        fs["projectorWidth"] >> projSize.width;
        fs["projectorHeight"] >> projSize.height;
        fs["useOpenni"] >> useOpenni;
    }

    Mat projectorPixels, cameraPixels;
    // load pixel correspondence maps
    {
        Mat correspondenceMapX, correspondenceMapY;
        correspondenceMapX = imread("correspondenceX.png");
        correspondenceMapY = imread("correspondenceY.png");

        camSize = correspondenceMapX.size();
        projectorPixels = Mat::zeros(correspondenceMapX.size(), CV_32SC2);
        cameraPixels = Mat::zeros(projSize, CV_32SC2);
        for (int i = 0; i < camSize.height; i++)
        {
            for (int j = 0; j < camSize.width; j++)
            {
                Vec3b sx = correspondenceMapX.at<Vec3b>(i, j);
                Vec3b sy = correspondenceMapY.at<Vec3b>(i, j);
                int x = (int)sx[1] * 256 + (int)sx[2];
                int y = (int)sy[1] * 256 + (int)sy[2];
                if (x < 0 || projSize.width <= x || y < 0 || projSize.height <= y)
                {
                    projectorPixels.at<Point2i>(i, j) = Point2i(-1, -1);
                }
                else
                {
                    projectorPixels.at<Point2i>(i, j) = Point2i(x, y);
                    cameraPixels.at<Point2i>(y, x) = Point2i(j, i);
                }
            }
        }

        // eliminate non smooth points
        for (int i = 2; i < camSize.height - 2; i++)
        {
            for (int j = 2; j < camSize.width - 2; j++)
            {
                Point2i & p = projectorPixels.at<Point2i>(i, j);
                if (p.x < 0 || p.y < 0)
                {
                    continue;
                }
                // look for 8 neighbors
                int count = 0;
                float avgx = 0;
                float avgy = 0;
                for (int dy = -2; dy <= 2; dy++)
                {
                    for (int dx = -2; dx <= 2; dx++)
                    {
                        Point2i & pt = projectorPixels.at<Point2i>(i + dy, j + dx);
                        if (pt.x < 0 || pt.y < 0)
                        {
                            continue;
                        }
                        avgx += pt.x;
                        avgy += pt.y;
                        count++;
                    }
                }

                if (count < 10)
                {
                    // not enough samples, skip
                    p.x = -1; p.y = -1;
                }
                else
                {
                    avgx /= count;
                    avgy /= count;
                    if (norm(Point2f(p.x - avgx, p.y - avgy)) > 3)
                    {
                        // too far, skip
                        p.x = -1; p.y = -1;
                    }
                }
            }
        }
    }

    Mat image, depth;
    float focalLength;
    Mat cameraMatrix = Mat::eye(3, 3, CV_32FC1);

    if (!useOpenni)
    {
        cout << "Meshing example requires an OpenNI2-compatible depth camera!" << endl;
        return 1;
    }

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

    Ptr<RgbdFrame> frame = makePtr<RgbdFrame>(image, depth);
    frame->cameraMatrix = cameraMatrix;

    Ptr<DepthCleaner> cleaner = makePtr<DepthCleaner>(CV_32F, 5);
    Mat tmp;
    (*cleaner)(frame->depth, tmp);
    frame->depth = tmp;

    depthTo3d(*frame);

    // generate valid point mask for clusters
    compare(frame->depth, 0, frame->mask, CMP_GT);

    // eliminate depth discontinuity which may cause problems when meshing
    for (int i = 1; i < camSize.height - 1; i++)
    {
        for (int j = 1; j < camSize.width - 1; j++)
        {
            float p = frame->depth.at<float>(i, j);
            if (frame->mask.at<uchar>(i, j) == 0)
            {
                continue;
            }
            // look for 8 neighbors
            int count = 0;
            float avg = 0;
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    float pt = frame->depth.at<float>(i + dy, j + dx);
                    if (pt == 0)
                    {
                        continue;
                    }
                    avg += pt;
                    count++;
                }
            }

            if (count < 5)
            {
                // not enough samples
                frame->mask.at<uchar>(i, j) = 0;
                frame->depth.at<float>(i, j) = 0;
                frame->points3d.at<Point3f>(i, j) = Point3f(0, 0, 0);
            }
            else
            {
                avg /= count;
                if (abs(p - avg) > 0.005f) // [m]
                {
                    // too far
                    frame->mask.at<uchar>(i, j) = 0;
                    frame->depth.at<float>(i, j) = 0;
                    frame->points3d.at<Point3f>(i, j) = Point3f(0, 0, 0);
                }
            }
        }
    }

    imshow("depth", frame->points3d);
    waitKey(30);

    RgbdMesh mainCluster(frame);
    vector<RgbdMesh> clusters;
    planarSegmentation(mainCluster, clusters, 1, 100000);
    deleteEmptyClusters(clusters);

    for (std::size_t i = 0; i < clusters.size(); i++) {
        if (clusters.at(i).bPlane) {
            stringstream ss;
            ss << "cluster" << i;
            // downsample by 2x
            //clusters.at(i).increment_step = 2;
            //clusters.at(i).projectorPixels = projectorPixels;
            //clusters.at(i).calculatePoints(true);
            //clusters.at(i).unwrapTexCoord();
            //clusters.at(i).save(ss.str() + ".obj");
            imshow(ss.str(), clusters.at(i).silhouette * 255);
            continue;
        }

        vector<RgbdMesh> smallClusters;

        // euclidianClustering is made for segmentation.
        // however, it is hard to maintain 3D mesh topology in camera and projector space
        // which is important for UV unwrapping.
        // it seems computing a mask in projector image,
        // segment the mask, and return to the camera space is the best way to maintain the topology.

        //euclideanClustering(clusters.at(i), smallClusters);
        
        // generate a silhouette in projector space
        Mat projectorLabels, stats, centroids;
        int minArea = 1000;
        // convert camera space silhouette to projector space
        Mat projectorSilhouette = Mat::zeros(cameraPixels.size(), CV_8U);
        Mat & cameraSilhouette = clusters.at(i).silhouette;
        for (int y = 0; y < cameraSilhouette.rows; y++)
        {
            for (int x = 0; x < cameraSilhouette.cols; x++)
            {
                Point2i & p = projectorPixels.at<Point2i>(y, x);
                if (cameraSilhouette.at<uchar>(y, x) > 0 && p.x >= 0 && p.y >= 0)
                {
                    projectorSilhouette.at<uchar>(p) = 255;
                }
            }
        }

        // since projector space silhouette is sparse,
        // dilate before computing connectedComponentsWithStats
        Mat kernel = Mat::ones(7, 7, CV_32S);
        dilate(projectorSilhouette, projectorSilhouette, kernel);
        connectedComponentsWithStats(projectorSilhouette, projectorLabels, stats, centroids, 8);
        Mat labels = Mat::zeros(cameraSilhouette.size(), CV_32S);
        // return to camera space
        for (int y = 0; y < projectorSilhouette.rows; y++)
        {
            for (int x = 0; x < projectorSilhouette.cols; x++)
            {
                int label = projectorLabels.at<int>(y, x);
                if (label > 0)
                {
                    labels.at<int>(cameraPixels.at<Point2i>(y, x)) = label;
                }
            }
        }
        imshow("silhouette", projectorSilhouette);
        for (int label = 1; label < stats.rows; label++)
        { // 0: background label
            if (stats.at<int>(label, CC_STAT_AREA) >= minArea)
            {
                smallClusters.push_back(RgbdMesh(clusters.at(i).rgbdFrame));
                RgbdMesh& cluster = smallClusters.back();
                compare(labels, label, cluster.silhouette, CMP_EQ);
                cluster.roi = Rect(0, 0, cameraSilhouette.cols, cameraSilhouette.rows);
                cluster.calculatePoints();
            }
        }

        for (std::size_t j = 0; j < smallClusters.size(); j++) {
            stringstream ss;
            ss << "mesh_" << i << "_" << j;

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
            imshow(ss.str(), smallClusters.at(j).silhouette * 255);
        }
    }

    waitKey(0);
    return 0;
}
