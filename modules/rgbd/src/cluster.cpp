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

#include "precomp.hpp"
#include <fstream>

namespace cv
{
namespace rgbd
{
    int RgbdCluster::getNumPoints()
    {
        if(bPointsUpdated)
            return static_cast<int>(points.size());
        else return -1;
    }

    void RgbdCluster::calculatePoints()
    {
        pointsIndex = Mat_<int>::eye(mask.rows, mask.cols) * -1;
        points.clear();
        for(int i = 0; i < mask.rows; i++)
        {
            for(int j = 0; j < mask.cols; j++)
            {
                if(mask.at<uchar>(i, j) > 0)
                {
                    if(depth.at<float>(i, j) > 0)
                    {
                        RgbdPoint point;
                        point.world_xyz = points3d.at<Point3f>(i, j);
                        point.image_xy = Point2i(j, i);

                        pointsIndex.at<int>(i, j) = static_cast<int>(points.size());
                        points.push_back(point);
                    }
                    else
                    {
                        mask.at<uchar>(i, j) = 0;
                    }
                }
            }
        }
        bPointsUpdated = true;
    }

    void RgbdCluster::calculateFaceIndices(float depthDiff)
    {
        if(!bPointsUpdated)
        {
            calculatePoints();
        }
        for(int i = 0; i < mask.rows; i++)
        {
            for(int j = 0; j < mask.cols; j++)
            {
                if(mask.at<uchar>(i, j) == 0)
                {
                    continue;
                }
                if(i + 1 == mask.rows || j + 1 == mask.cols)
                {
                    continue;
                }
                if(mask.at<uchar>(i + 1, j) > 0 &&
                    mask.at<uchar>(i, j + 1) > 0 &&
                    mask.at<uchar>(i + 1, j + 1) > 0)
                {
                    //depth comparison not working?
                    if(abs(depth.at<float>(i, j) - depth.at<float>(i + 1, j)) > depthDiff)
                    {
                        continue;
                    }
                    if(abs(depth.at<float>(i, j) - depth.at<float>(i, j + 1)) > depthDiff)
                    {
                        continue;
                    }
                    if(abs(depth.at<float>(i, j) - depth.at<float>(i + 1, j + 1)) > depthDiff)
                    {
                        continue;
                    }
                    faceIndices.push_back(pointsIndex.at<int>(i, j));
                    faceIndices.push_back(pointsIndex.at<int>(i+1, j));
                    faceIndices.push_back(pointsIndex.at<int>(i, j+1));
                    faceIndices.push_back(pointsIndex.at<int>(i, j+1));
                    faceIndices.push_back(pointsIndex.at<int>(i+1, j));
                    faceIndices.push_back(pointsIndex.at<int>(i+1, j+1));
                }
            }
        }

    }

    void RgbdCluster::unwrapTexCoord()
    {
        if(!bPointsUpdated)
        {
            calculatePoints();
        }
        // TODO: implement LSCM
        for(std::size_t i = 0; i < points.size(); i++) {
            RgbdPoint & point = points.at(i);
            point.texture_uv = Point2f((float)point.image_xy.x / mask.cols, (float)point.image_xy.y / mask.rows);
        }

        return;
    }

    void RgbdCluster::save(const std::string &path)
    {
        if(!bFaceIndicesUpdated)
        {
            calculateFaceIndices();
        }
        std::ofstream fs(path.c_str(), std::ofstream::out);
        for(std::size_t i = 0; i < points.size(); i++)
        {
            Point3f & v = points.at(i).world_xyz;
            Point2f & vt = points.at(i).texture_uv;
            // negate xy for Unity compatibility
            std::stringstream ss;
            fs << "v " << -v.x << " " << -v.y << " " << v.z << std::endl;
            fs << "vt " << vt.x << " " << vt.y << std::endl;
        }
        for(std::size_t i = 0; i < faceIndices.size(); i+=3)
        {
            fs << "f " << faceIndices.at(i)+1 << "/" << faceIndices.at(i)+1
                << "/ " << faceIndices.at(i+1)+1 << "/" << faceIndices.at(i+1)+1
                << "/ " << faceIndices.at(i+2)+1 << "/" << faceIndices.at(i+2)+1
                << "/" << std::endl;
        }
        fs.close();
    }


    void eliminateSmallClusters(std::vector<RgbdCluster>& clusters, int minPoints)
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

    void deleteEmptyClusters(std::vector<RgbdCluster>& clusters)
    {
        eliminateSmallClusters(clusters, 0);
    }

    void planarSegmentation(RgbdCluster& mainCluster, std::vector<RgbdCluster>& clusters, int maxPlaneNum, int minArea)
    {
        // assert frame size == points3d size

        Ptr<RgbdPlane> plane = makePtr<RgbdPlane>();
        plane->setThreshold(0.025f);
        Mat mask;
        std::vector<Vec4f> coeffs;
        //(*plane)(points3d, frame->normals, mask, coeffs);
        (*plane)(mainCluster.points3d, mask, coeffs);

        Mat colorLabels = Mat_<Vec3f>(mask.rows, mask.cols);
        for(int label = 0; label < maxPlaneNum + 1; label++)
        {
            clusters.push_back(RgbdCluster());
            RgbdCluster& cluster = clusters.back();
            mainCluster.depth.copyTo(cluster.depth);
            mainCluster.points3d.copyTo(cluster.points3d);
            if(label < maxPlaneNum)
            {
                compare(mask, label, cluster.mask, CMP_EQ);
                cluster.bPlane = true;
            }
            else
            {
                compare(mask, label, cluster.mask, CMP_GE); // residual
            }
            cluster.calculatePoints();
            if(cluster.getNumPoints() < minArea) {
                // discard;
            }
        }
    }

    void euclideanClustering(RgbdCluster& mainCluster, std::vector<RgbdCluster>& clusters, int minArea)
    {
        Mat labels, stats, centroids;
        connectedComponentsWithStats(mainCluster.mask, labels, stats, centroids, 8);
        for(int label = 1; label < stats.rows; label++)
        { // 0: background label
            if(stats.at<int>(label, CC_STAT_AREA) >= minArea)
            {
                clusters.push_back(RgbdCluster());
                RgbdCluster& cluster = clusters.back();
                mainCluster.depth.copyTo(cluster.depth);
                mainCluster.points3d.copyTo(cluster.points3d);
                compare(labels, label, cluster.mask, CMP_EQ);
                cluster.calculatePoints();
            }
        }
    }
}
}
