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
    int RgbdClusterMesh::getNumPoints()
    {
        if(bPointsUpdated)
            return static_cast<int>(points.size());
        else return -1;
    }

    void RgbdClusterMesh::calculatePoints()
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

    void RgbdClusterMesh::calculateFaceIndices(float depthDiff)
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

    void RgbdClusterMesh::unwrapTexCoord()
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

    void RgbdClusterMesh::save(const std::string &path)
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

}
}
