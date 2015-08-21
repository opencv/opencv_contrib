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
    RgbdMesh::RgbdMesh(Ptr<RgbdFrame> _rgbdFrame) :
        bPlane(false), bVectorPointsUpdated(false), increment_step(1), rgbdFrame(_rgbdFrame), bFaceIndicesUpdated(false)
    {
        CV_Assert(!rgbdFrame->depth.empty());
        CV_Assert(!rgbdFrame->mask.empty());

        (rgbdFrame->mask).copyTo(silhouette);

        roi.x = 0;
        roi.y = 0;
        roi.width = rgbdFrame->depth.cols;
        roi.height = rgbdFrame->depth.rows;
    }

    int RgbdMesh::getNumPoints()
    {
        if (bVectorPointsUpdated)
            return static_cast<int>(points.size());
        else
        {
            // assuming elements of silhouette are 0 or 1
            return static_cast<int>(sum(silhouette)[0]);
        }
    }

    void RgbdMesh::calculatePoints(bool skipNoCorrespondencePoints)
    {
        pointsIndex = Mat_<int>::eye(silhouette.rows, silhouette.cols) * -1;
        points.clear();
        for (int i = roi.y; i < silhouette.rows && i < roi.y + roi.height; i += increment_step)
        {
            for (int j = roi.x; j < silhouette.cols && j < roi.x + roi.width; j += increment_step)
            {
                if (silhouette.at<uchar>(i, j) > 0)
                {
                    if (skipNoCorrespondencePoints)
                    {
                        Point2i & projectorPoint = projectorPixels.at<Point2i>(i, j);
                        if (projectorPoint.x < 0 || projectorPoint.y < 0)
                            continue;
                    }

                    if (rgbdFrame->depth.at<float>(i, j) > 0)
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

    void RgbdMesh::calculateFaceIndices()
    {
        if(!bVectorPointsUpdated)
        {
            calculatePoints();
        }
        // TODO: optimize projector space ROI
        Rect faceRoi(1, 1, 1023, 767);
        Subdiv2D subdiv(faceRoi);
        Mat correspondenceMapPro = Mat::zeros(768, 1024, CV_32S);

        int xMin = 100000, xMax = -100000, yMin = 100000, yMax = -100000;
        for (int i = 0; i < getNumPoints(); i++)
        {
            Point2i & p = points.at(i).projector_xy;
            subdiv.insert(p);
            correspondenceMapPro.at<int>(p) = i;

            xMin = min(xMin, p.x);
            xMax = max(xMax, p.x);
            yMin = min(yMin, p.y);
            yMax = max(yMax, p.y);
        }
        faceRoi = Rect(xMin, yMin, xMax - xMin + 1, yMax - yMin + 1);

        std::vector<Vec6f> triangleList;
        subdiv.getTriangleList(triangleList);

        for (std::size_t i = 0; i < triangleList.size(); i++)
        {
            Point2i p0(cvRound(triangleList.at(i)[0]), cvRound(triangleList.at(i)[1]));
            Point2i p1(cvRound(triangleList.at(i)[2]), cvRound(triangleList.at(i)[3]));
            Point2i p2(cvRound(triangleList.at(i)[4]), cvRound(triangleList.at(i)[5]));
            if (!faceRoi.contains(p0) || !faceRoi.contains(p1) || !faceRoi.contains(p2))
                continue;

            int v0 = correspondenceMapPro.at<int>(p0.y, p0.x);
            int v1 = correspondenceMapPro.at<int>(p1.y, p1.x);
            int v2 = correspondenceMapPro.at<int>(p2.y, p2.x);

#if 0
            // eliminate too big triangles
            float distanceThreshold = 50; // [mm]
            if (norm(points.at(v0).world_xyz - points.at(v1).world_xyz) > distanceThreshold
                || norm(points.at(v1).world_xyz - points.at(v2).world_xyz) > distanceThreshold
                || norm(points.at(v2).world_xyz - points.at(v0).world_xyz) > distanceThreshold)
            {
                continue;
            }
#endif

            faceIndices.push_back(v0);
            faceIndices.push_back(v1);
            faceIndices.push_back(v2);
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

    void RgbdMesh::unwrapTexCoord()
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

#if 0
        if (bPlane)
        {
            // TODO: seems not working
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
#endif

#ifdef HAVE_OPENNL
        nlNewContext();
        nlSolverParameteri(NL_SOLVER, NL_CG);
        nlSolverParameteri(NL_PRECONDITIONER, NL_PRECOND_JACOBI);
        nlSolverParameteri(NL_NB_VARIABLES, 2 * getNumPoints());
        nlSolverParameteri(NL_LEAST_SQUARES, NL_TRUE);
        nlSolverParameteri(NL_MAX_ITERATIONS, 500);
        nlSolverParameterd(NL_THRESHOLD, 1e-10);

        nlBegin(NL_SYSTEM);

        // original implementation finds which axes (XYZ) to
        // assign initial UV, but here we assume X->U Y->V
        // since we use a depth map

        {
            // find min/max in U direction
            float uMax = -1e5f;
            float uMin = 1e5f;
            int uMaxInd = 0;
            int uMinInd = 0;
            for (int i = 0; i < getNumPoints(); i++)
            {
                Point2i & point = points.at(i).image_xy;
                nlSetVariable(2 * i, point.x);
                nlSetVariable(2 * i + 1, point.y);

                if (point.x > uMax) {
                    uMax = static_cast<float>(point.x);
                    uMaxInd = i;
                }
                if (point.x < uMin) {
                    uMin = static_cast<float>(point.x);
                    uMinInd = i;
                }

            }
            // "pin" leftmost and rightmost two points
            nlLockVariable(2 * uMaxInd);
            nlLockVariable(2 * uMaxInd + 1);
            nlLockVariable(2 * uMinInd);
            nlLockVariable(2 * uMinInd + 1);
        }

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

        {
            // store in a temporary vector for scaling
            std::vector<Point2f> uvs;
            float uMin = 1e5, uMax = -1e5, vMin = 1e5, vMax = -1e5;
            for (int i = 0; i < getNumPoints(); i++)
            {
                float u = (float)nlGetVariable(static_cast<NLuint>(i) * 2);
                float v = (float)nlGetVariable(static_cast<NLuint>(i) * 2 + 1);
                uvs.push_back(Point2f(u, v));
                uMin = std::min(uMin, u);
                uMax = std::max(uMax, u);
                vMin = std::min(vMin, v);
                vMax = std::max(vMax, v);
            }

            for (int i = 0; i < getNumPoints(); i++)
            {
                RgbdPoint & point = points.at(i);
                float u = uvs.at(i).x;
                float v = uvs.at(i).y;
                point.texture_uv = Point2f((u - uMin) / (uMax - uMin), (v - vMin) / (vMax - vMin) - 1);
            }
        }

        nlDeleteContext(nlGetCurrent());

        return;
#else
        std::cout << "UV mapping requires build with OpenNL" << std::endl;
        return;
#endif
    }

    void RgbdMesh::save(const std::string &path, bool replaceXyByTexCoord)
    {
        // has extension
        CV_Assert(path.length() >= 3);

        // ply or obj
        std::string extension = path.substr(path.length() - 4, 4);
        CV_Assert(extension.compare(".ply") == 0 || extension.compare(".obj") == 0);

        if (!bFaceIndicesUpdated)
        {
            calculateFaceIndices();
        }

        std::ofstream fs(path.c_str(), std::ofstream::out);

        if (extension.compare(".ply") == 0)
        {
            std::cout << "ply not supported" << std::endl;
#if 0
            // needs update
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
            for (std::size_t i = 0; i < points.size(); i++)
            {
                Point3f & v = points.at(i).world_xyz;
                Point2f & vt = points.at(i).texture_uv;
                // negate xy for Unity compatibility
                std::stringstream ss;
                fs << -v.x << " " << -v.y << " " << v.z << " " << vt.x << " " << vt.y << std::endl;
            }
            for (std::size_t i = 0; i < faceIndices.size(); i += 3)
            {
                fs << "3 " << faceIndices.at(i) << " "
                    << faceIndices.at(i + 1) << " "
                    << faceIndices.at(i + 2) << std::endl;
            }
#endif
        }
        else if (extension.compare(".obj") == 0)
        {
            for (std::size_t i = 0; i < points.size(); i++)
            {
                Point3f v;
                if (replaceXyByTexCoord == false)
                {
                    v.x = static_cast<float>(points.at(i).projector_xy.x);
                    v.y = static_cast<float>(points.at(i).projector_xy.y);
                    v.z = 1000 * points.at(i).world_xyz.z;
                }
                else
                {
                    v.x = points.at(i).texture_uv.x * 100;
                    v.y = points.at(i).texture_uv.y * 100;
                    v.z = 0;
                }

                // negate xy for Unity compatibility
                fs << "v " << -v.x << " " << -v.y << " " << v.z << std::endl;
            }

            for (std::size_t i = 0; i < points.size(); i++)
            {
                Point2f & vt = points.at(i).texture_uv;
                fs << "vt " << vt.x << " " << -vt.y << std::endl;
            }

            for (std::size_t i = 0; i < faceIndices.size(); i += 3)
            {
                int i0 = faceIndices.at(i);
                int i2 = faceIndices.at(i + 1);
                int i1 = faceIndices.at(i + 2);

                float distanceThreshold = 10; // [px]
                if (norm(points.at(i0).projector_xy - points.at(i1).projector_xy) > distanceThreshold
                    || norm(points.at(i1).projector_xy - points.at(i2).projector_xy) > distanceThreshold
                    || norm(points.at(i2).projector_xy - points.at(i0).projector_xy) > distanceThreshold)
                {
                    //continue;
                }

                fs << "f " << i0 + 1 << "/" << i0 + 1
                    << "/ " << i1 + 1 << "/" << i1 + 1
                    << "/ " << i2 + 1 << "/" << i2 + 1
                    << "/" << std::endl;
            }
        }
        fs.close();
    }

    void eliminateSmallClusters(std::vector<RgbdMesh>& clusters, int minPoints)
    {
        for (std::size_t i = 0; i < clusters.size(); )
        {
            if (clusters.at(i).getNumPoints() >= 0 && clusters.at(i).getNumPoints() <= minPoints)
            {
                clusters.erase(clusters.begin() + i);
            }
            else
            {
                i++;
            }
        }
    }

    void deleteEmptyClusters(std::vector<RgbdMesh>& clusters)
    {
        eliminateSmallClusters(clusters, 0);
    }

    void planarSegmentation(RgbdMesh& mainCluster, std::vector<RgbdMesh>& clusters, int maxPlaneNum, int minArea)
    {
        // assert frame size == points3d size

        Ptr<RgbdPlane> plane = makePtr<RgbdPlane>();
        plane->setThreshold(0.025f);
        Mat mask;
        std::vector<Vec4f> coeffs;
        (*plane)(*(mainCluster.rgbdFrame), mask, coeffs);

        int numExtractedPlanes = static_cast<int>(coeffs.size());

        int numValidPlanes = 0;
        int residualLabel = 0;
        for (int label = 0; label < numExtractedPlanes; label++)
        {
            clusters.push_back(RgbdMesh(mainCluster.rgbdFrame));
            RgbdMesh& cluster = clusters.back();

            compare(mask, label, cluster.silhouette, CMP_EQ);
            cluster.bPlane = true;
            cluster.calculatePoints();
            cluster.plane_coefficients = coeffs.at(label);
            if (cluster.getNumPoints() < minArea) {
                // rollback
                clusters.pop_back();
            }
            else
            {
                // valid plane
                numValidPlanes++;
                if (numValidPlanes >= maxPlaneNum)
                {
                    residualLabel = label + 1;
                    break;
                }
            }
        }

        // residual
        clusters.push_back(RgbdMesh(mainCluster.rgbdFrame));
        RgbdMesh& cluster = clusters.back();
        compare(mask, residualLabel, cluster.silhouette, CMP_GE);
        cluster.calculatePoints();
    }

    void euclideanClustering(RgbdMesh& mainCluster, std::vector<RgbdMesh>& clusters, int minArea)
    {
        Mat labels, stats, centroids;
        connectedComponentsWithStats(mainCluster.silhouette, labels, stats, centroids, 8);
        for (int label = 1; label < stats.rows; label++)
        { // 0: background label
            if (stats.at<int>(label, CC_STAT_AREA) >= minArea)
            {
                clusters.push_back(RgbdMesh(mainCluster.rgbdFrame));
                RgbdMesh& cluster = clusters.back();
                compare(labels, label, cluster.silhouette, CMP_EQ);
                cluster.roi = Rect(stats.at<int>(label, CC_STAT_LEFT), stats.at<int>(label, CC_STAT_TOP),
                    stats.at<int>(label, CC_STAT_WIDTH), stats.at<int>(label, CC_STAT_HEIGHT));
                cluster.calculatePoints();
            }
        }
    }
}
}
