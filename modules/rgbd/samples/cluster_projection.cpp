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
    float pixelSize, refDistance;

    if(argc != 2) {
        // bad arguments
        exit(1);
    }

    // read depth data from libfreenect
    cv::FileStorage file(string(argv[1]), cv::FileStorage::READ);

    file["depth"] >> depth;
    file["zeroPlanePixelSize"] >> pixelSize;
    file["zeroPlaneDistance"] >> refDistance;
    depth = depth * 0.001f; // libfreenect is in [mm]

    // kinect parameter to focal length
    // https://github.com/OpenKinect/libfreenect/blob/master/src/registration.c#L317
    float fx = refDistance * 0.5f / pixelSize,
        fy = refDistance * 0.5f / pixelSize,
        cx = 319.5f,
        cy = 239.5f;

    Mat cameraMatrix = Mat::eye(3,3,CV_32FC1);
    {
        cameraMatrix.at<float>(0,0) = fx;
        cameraMatrix.at<float>(1,1) = fy;
        cameraMatrix.at<float>(0,2) = cx;
        cameraMatrix.at<float>(1,2) = cy;
    }

    Ptr<RgbdFrame> frame = makePtr<RgbdFrame>(image, depth);
    Ptr<DepthCleaner> cleaner = makePtr<DepthCleaner>(CV_32F, 5);
    Mat tmp;
    (*cleaner)(frame->depth, tmp);
    frame->depth = tmp;

    //auto normals = makePtr<RgbdNormals>(frame->depth.rows, frame->depth.cols, frame->depth.depth(), cameraMatrix, 5, RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD);
    Mat points3d;
    depthTo3d(frame->depth, cameraMatrix, points3d);
    //(*normals)(frame->depth, frame->normals);

    RgbdClusterMesh mainCluster;
    mainCluster.points3d = points3d;
    mainCluster.depth = frame->depth;
    vector<RgbdClusterMesh> clusters;
    planarSegmentation(mainCluster, clusters);
    deleteEmptyClusters(clusters);

    for(std::size_t i = 0; i < clusters.size(); i++) {
        {
            stringstream ss;
            ss << "cluster " << i;
            imshow(ss.str(), clusters.at(i).mask * 255);
        }

        Mat labels;
        Mat stats;
        Mat centroids;

        if(clusters.at(i).bPlane) {
            continue;
        }

        vector<RgbdClusterMesh> smallClusters;
        euclideanClustering(clusters.at(i), smallClusters);
        //deleteEmptyClusters(smallClusters);
        for(std::size_t j = 0; j < smallClusters.size(); j++) {
            stringstream ss;
            ss << "mesh_" << i << "_" << j;
            imshow(ss.str(), smallClusters.at(j).mask * 255);
            smallClusters.at(j).unwrapTexCoord();
            smallClusters.at(j).save(ss.str() + ".obj");
        }
    }

    waitKey(0);
    return 0;
}
