/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2015, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

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

    RgbdCluster mainCluster;
    mainCluster.points3d = points3d;
    mainCluster.depth = frame->depth;
    vector<RgbdCluster> clusters;
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

        vector<RgbdCluster> smallClusters;
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
