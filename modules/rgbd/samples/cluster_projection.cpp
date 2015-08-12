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
    int devId;
    int lightThreshold;
    int lightIntensity;
    Size camSize, projSize;
    bool useOpenni;
    FileStorage fs("capturer_parameters.yml", FileStorage::Mode::READ);
    fs["deviceId"] >> devId;
    fs["lightThreshold"] >> lightThreshold;
    fs["lightIntensity"] >> lightIntensity;
    fs["projectorWidth"] >> projSize.width;
    fs["projectorHeight"] >> projSize.height;
    fs["useOpenni"] >> useOpenni;

    Mat projectorPixels;
    {
        Mat correspondenceMapX, correspondenceMapY;
        correspondenceMapX = imread("correspondenceX.png");
        correspondenceMapY = imread("correspondenceY.png");
        camSize = correspondenceMapX.size();

        projectorPixels = Mat::zeros(correspondenceMapX.size(), CV_32SC2);
        for (int i = 0; i < camSize.height; i++)
        {
            for (int j = 0; j < camSize.width; j++)
            {
//                if (frame->mask.at<uchar>(i, j) == 0)
//                    continue;

                Vec3b sx = correspondenceMapX.at<Vec3b>(i, j);
                Vec3b sy = correspondenceMapY.at<Vec3b>(i, j);
                int x = (int)sx[1] * 256 + (int)sx[2];
                int y = (int)sy[1] * 256 + (int)sy[2];
                if (/*sx[0] == 255
                    && sx[1] == 255
                    && sx[2] == 255*/
                    x < 0 || projSize.width <= x || y < 0 || projSize.height <= y)
                {
                    projectorPixels.at<Point2i>(i, j) = Point2i(-1, -1);
                }
                else
                {
                    projectorPixels.at<Point2i>(i, j) = Point2i(x, y);
                }
            }
        }

    }

    Mat image, depth;
    float focalLength;
    Mat cameraMatrix = Mat::eye(3, 3, CV_32FC1);

    if (useOpenni)
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

        Ptr<RgbdFrame> frame = makePtr<RgbdFrame>(image, depth);
        frame->cameraMatrix = cameraMatrix;

        Ptr<DepthCleaner> cleaner = makePtr<DepthCleaner>(CV_32F, 5);
        Mat tmp;
        (*cleaner)(frame->depth, tmp);
        frame->depth = tmp;

        depthTo3d(*frame);

        // generate valid point mask for clusters
        compare(frame->depth, 0, frame->mask, CMP_GT);

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

    }
    else
    {
        Subdiv2D subdiv(Rect(0, 0, projSize.width, projSize.height));
        Mat correspondenceMapPro = Mat(projSize, CV_32S, -1);

        int xMin = 1e5, xMax = -1e5, yMin = 1e5, yMax = -1e5;
        int count = 0;
        vector<Point2i> points;
        for (int i = 0; i < camSize.height; i++)
        {
            for (int j = 0; j < camSize.width; j++)
            {
                Point2i & p = projectorPixels.at<Point2i>(i, j);
                if (p.x < 0 || p.y < 0)
                {
                    continue;
                }
                points.push_back(p);
                subdiv.insert(p);
                correspondenceMapPro.at<int>(p) = count;
                count++;

                xMin = min(xMin, p.x);
                xMax = max(xMax, p.x);
                yMin = min(yMin, p.y);
                yMax = max(yMax, p.y);
            }
        }
        Rect projRoi(xMin, yMin, xMax - xMin + 1, yMax - yMin + 1);

        std::vector<Vec6f> triangleList;
        subdiv.getTriangleList(triangleList);

        vector<int> indices;

        for (std::size_t i = 0; i < triangleList.size(); i++)
        {
            Point2i p0(triangleList.at(i)[0], triangleList.at(i)[1]);
            Point2i p1(triangleList.at(i)[2], triangleList.at(i)[3]);
            Point2i p2(triangleList.at(i)[4], triangleList.at(i)[5]);
            if (!projRoi.contains(p0) || !projRoi.contains(p1) || !projRoi.contains(p2))
                continue;

            int v0 = correspondenceMapPro.at<int>(cvRound(triangleList.at(i)[1]), cvRound(triangleList.at(i)[0]));
            int v1 = correspondenceMapPro.at<int>(cvRound(triangleList.at(i)[3]), cvRound(triangleList.at(i)[2]));
            int v2 = correspondenceMapPro.at<int>(cvRound(triangleList.at(i)[5]), cvRound(triangleList.at(i)[4]));

            indices.push_back(v0);
            indices.push_back(v1);
            indices.push_back(v2);
        }

        string path = "mesh.obj";
        std::ofstream ofs(path.c_str(), std::ofstream::out);
        for (std::size_t i = 0; i < points.size(); i++)
        {
            Point2i & x = points.at(i);
            // negate xy for Unity compatibility
            ofs << "v " << -x.x << " " << -x.y << " " << 0 << std::endl;
        }

        for (std::size_t i = 0; i < points.size(); i++)
        {
            Point2i & vt = points.at(i);
            ofs << "vt " << vt.x << " " << vt.y << std::endl;
        }

        for (std::size_t i = 0; i < indices.size(); i += 3)
        {
            int i0 = indices.at(i);
            int i2 = indices.at(i + 1);
            int i1 = indices.at(i + 2);

            float distanceThreshold = 10; // [px]
            if (norm(points.at(i0) - points.at(i1)) > distanceThreshold
                || norm(points.at(i1) - points.at(i2)) > distanceThreshold
                || norm(points.at(i2) - points.at(i0)) > distanceThreshold)
            {
                continue;
            }

            ofs << "f " << i0 + 1 << "/" << i0 + 1
                << "/ " << i1 + 1 << "/" << i1 + 1
                << "/ " << i2 + 1 << "/" << i2 + 1
                << "/" << std::endl;
        }
        ofs.close();

    }


    waitKey(0);
    return 0;
}
