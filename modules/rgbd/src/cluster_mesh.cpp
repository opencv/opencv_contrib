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
    RgbdClusterMesh::RgbdClusterMesh(Ptr<RgbdFrame> _rgbdFrame) : RgbdCluster(_rgbdFrame), bFaceIndicesUpdated(false)
    {
    }

    void RgbdClusterMesh::calculateFaceIndices(float depthDiff)
    {
        if(!bVectorPointsUpdated)
        {
            calculatePoints();
        }
        for(int i = roi.y; i < silhouette.rows && i < roi.y + roi.height; i += increment_step)
        {
            for(int j = roi.x; j < silhouette.cols && j < roi.x + roi.width; j += increment_step)
            {
                if(silhouette.at<uchar>(i, j) == 0)
                {
                    continue;
                }
                if(i + static_cast<int>(increment_step) == silhouette.rows || j + static_cast<int>(increment_step) == silhouette.cols)
                {
                    continue;
                }
                if(silhouette.at<uchar>(i + increment_step, j) > 0 &&
                    silhouette.at<uchar>(i, j + increment_step) > 0 &&
                    silhouette.at<uchar>(i + increment_step, j + increment_step) > 0)
                {
                    //depth comparison not working?
                    if(abs(rgbdFrame->depth.at<float>(i, j) - rgbdFrame->depth.at<float>(i + increment_step, j)) > depthDiff)
                    {
                        continue;
                    }
                    if(abs(rgbdFrame->depth.at<float>(i, j) - rgbdFrame->depth.at<float>(i, j + increment_step)) > depthDiff)
                    {
                        continue;
                    }
                    if(abs(rgbdFrame->depth.at<float>(i, j) - rgbdFrame->depth.at<float>(i + increment_step, j + increment_step)) > depthDiff)
                    {
                        continue;
                    }
                    faceIndices.push_back(pointsIndex.at<int>(i, j));
                    faceIndices.push_back(pointsIndex.at<int>(i+increment_step, j));
                    faceIndices.push_back(pointsIndex.at<int>(i, j+increment_step));
                    faceIndices.push_back(pointsIndex.at<int>(i, j+increment_step));
                    faceIndices.push_back(pointsIndex.at<int>(i+increment_step, j));
                    faceIndices.push_back(pointsIndex.at<int>(i+increment_step, j+increment_step));
                }
            }
        }

    }

    inline void project_triangle(Point3f& p0, Point3f& p1, Point3f& p2,
        Point2f& z0, Point2f& z1, Point2f& z2)
    {
        Point3f X = p1 - p0;
        float x1 = static_cast<float>(norm(X));
        X *= 1.0f / x1;
        Point3f p02 = p2 - p0;
        Point3f Z = X.cross(p02);
        Z *= 1.0f / norm(Z);
        Point3f Y = Z.cross(X);

        float x2 = X.dot(p02);
        float y2 = Y.dot(p02);

        z0 = Point2f(0, 0);
        z1 = Point2f(x1, 0);
        z2 = Point2f(x2, y2);
    }

    void RgbdClusterMesh::unwrapTexCoord()
    {
        if(!bVectorPointsUpdated)
        {
            calculatePoints();
        }

        CV_Assert(points.size() > 0);

        if (!bFaceIndicesUpdated)
        {
            calculateFaceIndices();
        }

        if (bPlane)
        {
            Vec3f center = points.at(0).world_xyz;
            Vec3f planeNormal(plane_coefficients[0], plane_coefficients[1], plane_coefficients[2]);

            Vec3f tangent, bitangent;
            Vec3f arb(0, 1, 0);
            tangent = arb.cross(planeNormal);
            normalize(tangent);
            bitangent = planeNormal.cross(tangent);
            normalize(bitangent);

            for (int i = 0; i < getNumPoints(); i++) {
                RgbdPoint & point = points.at(i);

                float x = point.world_xyz.dot(tangent) * 0.001f;
                // normalize to 0-1
                x = x - (long)x;
                if (x < 0) x += 1;
                float y = point.world_xyz.dot(bitangent) * 0.001f;
                // normalize to 0-1
                y = y - (long)y;
                if (y < 0) y += 1;

                point.texture_uv = Point2f(x, y);
            }

            return;
        }
        else
        {
            // TODO: implement LSCM
            for (std::size_t i = 0; i < points.size(); i++) {
                RgbdPoint & point = points.at(i);
                point.texture_uv = Point2f((float)point.image_xy.x / silhouette.cols, (float)point.image_xy.y / silhouette.rows);
            }
        }

        return;

#if 0
        nlNewContext();
        nlSolverParameteri(NL_SOLVER, NL_CG);
        nlSolverParameteri(NL_PRECONDITIONER, NL_PRECOND_JACOBI);
        nlSolverParameteri(NL_NB_VARIABLES, 2 * getNumPoints());
        nlSolverParameteri(NL_LEAST_SQUARES, NL_TRUE);
        nlSolverParameteri(NL_MAX_ITERATIONS, 5 * getNumPoints());
        nlSolverParameterd(NL_THRESHOLD, 1e-10);

        nlBegin(NL_SYSTEM);

        // original implementation finds which axes (XYZ) to
        // assign initial UV, but here we assume X->U Y->V
        // since we use a depth map

        // find min/max in U direction
        float uMax = -1000000000;
        float uMin = 1000000000;
        int uMaxInd = 0;
        int uMinInd = 0;
        for(int i = 0; i < getNumPoints(); i++) {
            Point3f & point = points.at(i).world_xyz;
            nlSetVariable(2 * i, point.x);
            nlSetVariable(2 * i + 1, point.y);

            if (point.x > uMax){
                uMax = point.x;
                uMaxInd = i;
            }
            if (point.x < uMin){
                uMin = point.x;
                uMinInd = i;
            }

        }
        // "pin" leftmost and rightmost two points
        nlLockVariable(2 * uMaxInd);
        nlLockVariable(2 * uMaxInd + 1);
        nlLockVariable(2 * uMinInd);
        nlLockVariable(2 * uMinInd + 1);

        nlBegin(NL_MATRIX);

        size_t numTriangles = faceIndices.size() / 3;
        for(int i = 0; i < numTriangles; i++) {
            int idx0, idx1, idx2;
            idx0 = faceIndices.at(3 * i);
            idx1 = faceIndices.at(3 * i + 1);
            idx2 = faceIndices.at(3 * i + 2);

            Point2f z0, z1, z2;
            Point3f p0, p1, p2;
            p0 = points.at(idx0).world_xyz;
            p1 = points.at(idx1).world_xyz;
            p2 = points.at(idx2).world_xyz;
            project_triangle(p0, p1, p2, z0, z1, z2);

            Point2f z01 = z1 - z0;
            Point2f z02 = z2 - z0;
            float a = z01.x;
            float b = z01.y; // 0
            float c = z02.x;
            float d = z02.y;

            // Real part
            nlBegin(NL_ROW);
            nlCoefficient(2 * idx0, -a + c);
            nlCoefficient(2 * idx0 + 1, b - d);
            nlCoefficient(2 * idx1, -c);
            nlCoefficient(2 * idx1 + 1, d);
            nlCoefficient(2 * idx2, a);
            nlEnd(NL_ROW);

            // Imaginary part
            nlBegin(NL_ROW);
            nlCoefficient(2 * idx0, -b + d);
            nlCoefficient(2 * idx0 + 1, -a + c);
            nlCoefficient(2 * idx1, -d);
            nlCoefficient(2 * idx1 + 1, -c);
            nlCoefficient(2 * idx2 + 1, a);
            nlEnd(NL_ROW);
        }

        nlEnd(NL_MATRIX);
        nlEnd(NL_SYSTEM);
        nlSolve();

        for (std::size_t i = 0; i < points.size(); i++) {
            RgbdPoint & point = points.at(i);
            float u = (float)nlGetVariable(static_cast<NLuint>(i) * 2    ) / (uMax - uMin);
            float v = (float)nlGetVariable(static_cast<NLuint>(i) * 2 + 1) / (uMax - uMin);
            point.texture_uv = Point2f(u, v);
        }

        nlDeleteContext(nlGetCurrent());

        return;
#endif
    }

    void RgbdClusterMesh::save(const std::string &path)
    {
        // has extension
        CV_Assert(path.length() >= 3);

        // ply or obj
        std::string extension = path.substr(path.length() - 4, 4);
        CV_Assert(extension.compare(".ply") == 0 || extension.compare(".obj") == 0);

        if(!bFaceIndicesUpdated)
        {
            calculateFaceIndices();
        }
        
        std::ofstream fs(path.c_str(), std::ofstream::out);
        
        if(extension.compare(".ply") == 0)
        {
            fs << "ply" << std::endl;
            fs << "format ascii 1.0" << std::endl;
            fs << "element vertex " << points.size() << std::endl;
            fs << "property float x" << std::endl;
            fs << "property float y" << std::endl;
            fs << "property float z" << std::endl;
            fs << "property float u" << std::endl;
            fs << "property float v" << std::endl;
            fs << "element face " << faceIndices.size() / 3 << std::endl;
            fs << "property list uchar uint vertex_indices" << std::endl;
            fs << "end_header" << std::endl;
            for(std::size_t i = 0; i < points.size(); i++)
            {
                Point3f & v = points.at(i).world_xyz;
                Point2f & vt = points.at(i).texture_uv;
                // negate xy for Unity compatibility
                std::stringstream ss;
                fs << -v.x << " " << -v.y << " " << v.z << " " << vt.x << " " << vt.y << std::endl;
            }
            for(std::size_t i = 0; i < faceIndices.size(); i += 3)
            {
                fs << "3 " << faceIndices.at(i) << " "
                    << faceIndices.at(i + 1) << " "
                    << faceIndices.at(i + 2) << std::endl;
            }
        }
        else if(extension.compare(".obj") == 0)
        {
            for(std::size_t i = 0; i < points.size(); i++)
            {
                Point3f & v = points.at(i).world_xyz;
                // negate xy for Unity compatibility
                std::stringstream ss;
                fs << "v " << -v.x << " " << -v.y << " " << v.z << std::endl;
            }
            for(std::size_t i = 0; i < points.size(); i++)
            {
                Point2f & vt = points.at(i).texture_uv;
                std::stringstream ss;
                fs << "vt " << vt.x << " " << vt.y << std::endl;
            }
            for(std::size_t i = 0; i < faceIndices.size(); i += 3)
            {
                fs << "f " << faceIndices.at(i) + 1 << "/" << faceIndices.at(i) + 1
                    << "/ " << faceIndices.at(i + 1) + 1 << "/" << faceIndices.at(i + 1) + 1
                    << "/ " << faceIndices.at(i + 2) + 1 << "/" << faceIndices.at(i + 2) + 1
                    << "/" << std::endl;
            }
        }
        fs.close();
    }

}
}
