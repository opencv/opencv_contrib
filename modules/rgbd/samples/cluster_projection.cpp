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
    Mat image, depth;
    float focalLength;

    if(argc != 2) {
        // bad arguments
        exit(1);
    }

    if(string(argv[1]) == "OPENNI")
    {
        VideoCapture capture(CAP_OPENNI2);

        if (!capture.isOpened())
        {
            cout << "OpenNI2 device unavailable" << endl;
            return -1;
        }

        // set registeration on
        capture.set(CAP_PROP_OPENNI_REGISTRATION, 0.0);

        focalLength = static_cast<float>(capture.get(CAP_PROP_OPENNI_FOCAL_LENGTH));

        while(true)
        {
            capture.grab();

            capture.retrieve(image, CAP_OPENNI_BGR_IMAGE);
            imshow("Color", image);

            capture.retrieve(depth, CAP_OPENNI_DEPTH_MAP);
            imshow("Depth", depth * 8);

            if (waitKey(30) >= 0)
                break;
        }

        depth.convertTo(depth, CV_32F);
    }
    else
    {
        // read depth data from libfreenect
        cv::FileStorage file(string(argv[1]), cv::FileStorage::READ);

        float pixelSize, refDistance;
        file["depth"] >> depth;
        file["zeroPlanePixelSize"] >> pixelSize;
        file["zeroPlaneDistance"] >> refDistance;

        focalLength = refDistance * 0.5f / pixelSize;
    }

    depth = depth * 0.001f; // libfreenect is in [mm]

    // kinect parameter to focal length
    // https://github.com/OpenKinect/libfreenect/blob/master/src/registration.c#L317
    float fx = focalLength,
        fy = focalLength,
        cx = 319.5f,
        cy = 239.5f;

    Ptr<RgbdFrame> frame = makePtr<RgbdFrame>(image, depth);

    frame->cameraMatrix = Mat::eye(3, 3, CV_32FC1);
    {
        frame->cameraMatrix.at<float>(0, 0) = fx;
        frame->cameraMatrix.at<float>(1, 1) = fy;
        frame->cameraMatrix.at<float>(0, 2) = cx;
        frame->cameraMatrix.at<float>(1, 2) = cy;
    }

    Ptr<DepthCleaner> cleaner = makePtr<DepthCleaner>(CV_32F, 5);
    Mat tmp;
    (*cleaner)(frame->depth, tmp);
    frame->depth = tmp;

    depthTo3d(*frame);

    // generate valid point mask for clusters
    compare(frame->depth, 0, frame->mask, CMP_GT);

    RgbdClusterMesh mainCluster(frame);
    vector<RgbdClusterMesh> clusters;
    planarSegmentation(mainCluster, clusters);
    deleteEmptyClusters(clusters);

    for(std::size_t i = 0; i < clusters.size(); i++) {
        {
            stringstream ss;
            ss << "cluster " << i;
            imshow(ss.str(), clusters.at(i).silhouette * 255);
        }

        Mat labels;
        Mat stats;
        Mat centroids;

        if(clusters.at(i).bPlane) {
            stringstream ss;
            ss << "cluster" << i;
            clusters.at(i).increment_step = 2;
            clusters.at(i).calculatePoints();
            clusters.at(i).unwrapTexCoord();
            clusters.at(i).save(ss.str() + ".obj");
            clusters.at(i).save(ss.str() + ".ply");
            continue;
        }

        vector<RgbdClusterMesh> smallClusters;
        euclideanClustering(clusters.at(i), smallClusters);
        //deleteEmptyClusters(smallClusters);
        for(std::size_t j = 0; j < smallClusters.size(); j++) {
            stringstream ss;
            ss << "mesh_" << i << "_" << j;
            imshow(ss.str(), smallClusters.at(j).silhouette * 255);

            // downsample by 0.5x
            smallClusters.at(j).increment_step = 2;
            smallClusters.at(j).calculatePoints();
            smallClusters.at(j).unwrapTexCoord();
            smallClusters.at(j).save(ss.str() + ".obj");
            smallClusters.at(j).save(ss.str() + ".ply");
        }
    }

    waitKey(0);
    return 0;
}
