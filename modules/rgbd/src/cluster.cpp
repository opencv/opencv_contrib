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

#include "precomp.hpp"
#include <fstream>

namespace cv
{
namespace rgbd
{
    RgbdCluster::RgbdCluster(Ptr<RgbdFrame> _rgbdFrame) : bPlane(false), bVectorPointsUpdated(false), increment_step(1), rgbdFrame(_rgbdFrame)
    {
        CV_Assert(!rgbdFrame->depth.empty());
        CV_Assert(!rgbdFrame->mask.empty());

        (rgbdFrame->mask).copyTo(silhouette);

        roi.x = 0;
        roi.y = 0;
        roi.width = rgbdFrame->depth.cols;
        roi.height = rgbdFrame->depth.rows;
    }

    int RgbdCluster::getNumPoints()
    {
        if(bVectorPointsUpdated)
            return static_cast<int>(points.size());
        else
        {
            // assuming elements of silhouette are 0 or 1
            return static_cast<int>(sum(silhouette)[0]);
        }
    }

    void RgbdCluster::calculatePoints(bool skipNoCorrespondencePoints)
    {
        pointsIndex = Mat_<int>::eye(silhouette.rows, silhouette.cols) * -1;
        points.clear();
        for(int i = roi.y; i < silhouette.rows && i < roi.y + roi.height; i += increment_step)
        {
            for(int j = roi.x; j < silhouette.cols && j < roi.x + roi.width; j += increment_step)
            {
                if(silhouette.at<uchar>(i, j) > 0)
                {
                    if (skipNoCorrespondencePoints)
                    {
                        Point2i & projectorPoint = projectorPixels.at<Point2i>(i, j);
                        if(projectorPoint.x < 0 || projectorPoint.y < 0)
                            continue;
                    }

                    if(rgbdFrame->depth.at<float>(i, j) > 0)
                    {
                        RgbdPoint point;
                        point.world_xyz = rgbdFrame->points3d.at<Point3f>(i, j);
                        point.image_xy = Point2i(j, i);
                        if (skipNoCorrespondencePoints)
                        {
                            point.projector_xy = projectorPixels.at<Point2i>(i, j);
                        }

                        pointsIndex.at<int>(i, j) = static_cast<int>(points.size());
                        points.push_back(point);
                    }
                    else
                    {
                        silhouette.at<uchar>(i, j) = 0;
                    }
                }
            }
        }
        bVectorPointsUpdated = true;
    }

    template<typename T> void eliminateSmallClusters(std::vector<T>& clusters, int minPoints)
    {
        for(std::size_t i = 0; i < clusters.size(); )
        {
            if(clusters.at(i).getNumPoints() >= 0 && clusters.at(i).getNumPoints() <= minPoints)
            {
                clusters.erase(clusters.begin() + i);
            }
            else
            {
                i++;
            }
        }
    }
    template CV_EXPORTS void eliminateSmallClusters<RgbdCluster>(std::vector<RgbdCluster>& clusters, int minPoints);
    template CV_EXPORTS void eliminateSmallClusters<RgbdClusterMesh>(std::vector<RgbdClusterMesh>& clusters, int minPoints);

    template<typename T> void deleteEmptyClusters(std::vector<T>& clusters)
    {
        eliminateSmallClusters(clusters, 0);
    }
    template CV_EXPORTS void deleteEmptyClusters<RgbdCluster>(std::vector<RgbdCluster>& clusters);
    template CV_EXPORTS void deleteEmptyClusters<RgbdClusterMesh>(std::vector<RgbdClusterMesh>& clusters);

    template<typename T1, typename T2> void planarSegmentation(T1& mainCluster, std::vector<T2>& clusters, int maxPlaneNum, int minArea)
    {
        // assert frame size == points3d size

        Ptr<RgbdPlane> plane = makePtr<RgbdPlane>();
        plane->setThreshold(0.025f);
        Mat mask;
        std::vector<Vec4f> coeffs;
        (*plane)(*(mainCluster.rgbdFrame), mask, coeffs);

        if(static_cast<int>(coeffs.size()) < maxPlaneNum)
        {
            maxPlaneNum = static_cast<int>(coeffs.size());
        }

        for(int label = 0; label < maxPlaneNum + 1; label++)
        {
            clusters.push_back(T2(mainCluster.rgbdFrame));
            T2& cluster = clusters.back();
            if(label < maxPlaneNum)
            {
                compare(mask, label, cluster.silhouette, CMP_EQ);
                cluster.bPlane = true;
                cluster.calculatePoints();
                cluster.plane_coefficients = coeffs.at(label);
                if (cluster.getNumPoints() < minArea) {
                    // TODO: isn't this should be added to the residual cluster?
                    clusters.pop_back();
                }
            }
            else
            {
                compare(mask, label, cluster.silhouette, CMP_GE); // residual
            }
        }
    }
    template CV_EXPORTS void planarSegmentation<RgbdCluster, RgbdCluster>(RgbdCluster& mainCluster, std::vector<RgbdCluster>& clusters, int maxPlaneNum, int minArea);
    template CV_EXPORTS void planarSegmentation<RgbdClusterMesh, RgbdClusterMesh>(RgbdClusterMesh& mainCluster, std::vector<RgbdClusterMesh>& clusters, int maxPlaneNum, int minArea);
    template CV_EXPORTS void planarSegmentation<RgbdCluster, RgbdClusterMesh>(RgbdCluster& mainCluster, std::vector<RgbdClusterMesh>& clusters, int maxPlaneNum, int minArea);

    template<typename T1, typename T2> void euclideanClustering(T1& mainCluster, std::vector<T2>& clusters, int minArea)
    {
        Mat labels, stats, centroids;
        connectedComponentsWithStats(mainCluster.silhouette, labels, stats, centroids, 8);
        for(int label = 1; label < stats.rows; label++)
        { // 0: background label
            if(stats.at<int>(label, CC_STAT_AREA) >= minArea)
            {
                clusters.push_back(T2(mainCluster.rgbdFrame));
                T2& cluster = clusters.back();
                compare(labels, label, cluster.silhouette, CMP_EQ);
                cluster.roi = Rect(stats.at<int>(label, CC_STAT_LEFT), stats.at<int>(label, CC_STAT_TOP),
                    stats.at<int>(label, CC_STAT_WIDTH), stats.at<int>(label, CC_STAT_HEIGHT));
                cluster.calculatePoints();
            }
        }
    }
    template CV_EXPORTS void euclideanClustering<RgbdCluster, RgbdCluster>(RgbdCluster& mainCluster, std::vector<RgbdCluster>& clusters, int minArea);
    template CV_EXPORTS void euclideanClustering<RgbdClusterMesh, RgbdClusterMesh>(RgbdClusterMesh& mainCluster, std::vector<RgbdClusterMesh>& clusters, int minArea);
    template CV_EXPORTS void euclideanClustering<RgbdCluster, RgbdClusterMesh>(RgbdCluster& mainCluster, std::vector<RgbdClusterMesh>& clusters, int minArea);
}
}
