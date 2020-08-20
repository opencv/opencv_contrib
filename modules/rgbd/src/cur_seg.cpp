//
// Created by YIMIN TANG on 8/2/20.
//

#include "precomp.hpp"
#include "cur_seg.hpp"

namespace cv
{
namespace pcseg
{
    float angleBetween(const Point3f &v1, const Point3f &v2)
    {
        float len1 = sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
        float len2 = sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);
        float dot = v1.dot(v2);
        float a = dot / (len1 * len2);
        if (a >= 1.0)
            return 0.0;
        else if (a <= -1.0)
            return M_PI;
        else
            return acos(a); // 0..PI
    }

    std::vector<float> calCurvatures(Mat& pointsWithNormal, int k)
    {
        std::vector<Point3f> points;
        std::vector<Point3f> normal;
        int len = pointsWithNormal.size().height;
        int channel = pointsWithNormal.size().width;
        printf("%d\n",len);
        if (channel < 6)
        {
            Mat pointsWithNormal2;
            ppf_match_3d::computeNormalsPC3d(pointsWithNormal, pointsWithNormal2, k, 0, Vec3f(0,0,0));
            pointsWithNormal = pointsWithNormal2;
            channel = pointsWithNormal.size().width;
        }
        for (int i=0;i<len;i++)
        {
            Point3f p(pointsWithNormal.at<float>(i,0), pointsWithNormal.at<float>(i,1), pointsWithNormal.at<float>(i,2));
            points.push_back(p);
            Point3f q(pointsWithNormal.at<float>(i,3), pointsWithNormal.at<float>(i,4), pointsWithNormal.at<float>(i,5));
            normal.push_back(q);
        }
        printf("Build KDTree\n");fflush(stdout);
        std::vector<float> curvatures;
        flann::KDTreeIndexParams indexParams;
        flann::Index kdtree(Mat(points).reshape(1), indexParams);
        printf("Build KDTree Done\n");fflush(stdout);

        Mat querys = Mat(points).reshape(1);
        Mat indices;
        Mat dists;
        flann::SearchParams params(32);
        printf("kd tree search\n");fflush(stdout);
        kdtree.knnSearch(querys, indices, dists, k, params);
        std::cout<<indices.size()<<std::endl;
        printf("kd tree search end\n");fflush(stdout);
        for (int i=0; i<len;i++)
        {
            // printf("%d\n",i);fflush(stdout);
            std::vector<Point3f> nearPoints;
            // nearPoints.push_back(normal[i]);
            for (int j=0;j<k;j++) nearPoints.push_back(normal[indices.at<int>(i,j)]);
            Mat pointMat = Mat(nearPoints).reshape(1);
            PCA pca(pointMat, Mat(), 0);
            int size = pca.eigenvalues.size().height;
            float a = pca.eigenvalues.at<float>(0);
            float b = pca.eigenvalues.at<float>(size/2);
            float c = pca.eigenvalues.at<float>(size-1);
            curvatures.push_back(c/(a+b+c));
        }
        return curvatures;
    }


    std::vector<int> planarSegments(
            Mat& pointsWithNormal,
            std::vector<float>& curvatures,
            int k,
            float thetaThreshold,
            float curvatureThreshold)
    {

        std::vector<Point3f> points;
        std::vector<Point3f> normal;
        int len = pointsWithNormal.size().height;
        int channel = pointsWithNormal.size().width;
        if (channel != 6) CV_Error(Error::StsBadArg, String("No Normal Channel!"));
        for (int i=0;i<len;i++)
        {
            Point3f p(pointsWithNormal.at<float>(i,0), pointsWithNormal.at<float>(i,1), pointsWithNormal.at<float>(i,2));
            points.push_back(p);
            Point3f q(pointsWithNormal.at<float>(i,3), pointsWithNormal.at<float>(i,4), pointsWithNormal.at<float>(i,5));
            normal.push_back(q);
        }


        flann::KDTreeIndexParams indexParams;
        flann::Index kdtree(Mat(points).reshape(1), indexParams);


        int isSegCount = 0;
        std::queue<int> q;
        std::vector<bool> isSeg(len, 0 );
        std::vector<int> idSeg(len);
        for (int i=0;i<len;i++) idSeg[i] = i;

        while (isSegCount < len || !q.empty()) {
            int seedPointId = -1;
            if (q.empty())
            {
                float cur = 1e6;
                for (int i=0;i<len;i++)
                    if (cur > curvatures[i] && isSeg[i] == 0) {seedPointId = i; cur = curvatures[i];}
                isSeg[seedPointId] = 1;
                isSegCount++;
            }
            else {
                seedPointId = q.front();
                q.pop();
            }

            std::vector<float> query;
            query.push_back(points[seedPointId].x);
            query.push_back(points[seedPointId].y);
            query.push_back(points[seedPointId].z);
            std::vector<int> indices;
            std::vector<float> dists;
            kdtree.knnSearch(query, indices, dists, k);
            for (int i=0;i<indices.size();i++)
            {
                if (angleBetween(normal[seedPointId], normal[indices[i]]) < thetaThreshold)
                {
                    idSeg[indices[i]] = idSeg[seedPointId];
                    if (curvatures[indices[i]] < curvatureThreshold && isSeg[indices[i]] == 0)
                    {
                        isSeg[indices[i]] = 1;
                        isSegCount++;
                        q.push(indices[i]);
                    }
                }
            }
        }
        return idSeg;
    }

    bool planarMerge(Mat& pointsWithNormal ,
                     std::vector<int>& idA,
                     Point3f& normalA,
                     int idACenter,
                     double& timestepA,
                     std::vector<int>& idB,
                     Point3f& normalB,
                     int idBCenter,
                     float disThreshold)
    {
        std::vector<Point3f> points;
        std::vector<Point3f> normal;
        int len = pointsWithNormal.size().height;
        int channel = pointsWithNormal.size().width;
        if (channel != 6) CV_Error(Error::StsBadArg, String("No Normal Channel!"));
        for (int i=0;i<len;i++)
        {
            Point3f p(pointsWithNormal.at<float>(i,0), pointsWithNormal.at<float>(i,1), pointsWithNormal.at<float>(i,2));
            points.push_back(p);
            Point3f q(pointsWithNormal.at<float>(i,3), pointsWithNormal.at<float>(i,4), pointsWithNormal.at<float>(i,5));
            normal.push_back(q);
        }
        Point3f aCenter = points[idACenter];
        Point3f bCenter = points[idBCenter];
        for (int i=0;i<idB.size();i++)
        {
            Point3f h = points[idB[i]];
            Point3f dis = aCenter - h;
            if (dis.dot(dis) < disThreshold*disThreshold)
            {
                normalA = normalA*sqrt(aCenter.dot(aCenter)) + normalB*sqrt(bCenter.dot(bCenter));
                normalA /= (sqrt(aCenter.dot(aCenter)) + sqrt(bCenter.dot(bCenter)));
                for (int j=0;j<idB.size();j++) idA.push_back(idB[j]);
                timestepA = 0;
                idB.clear();
                return true;
            }
        }
        return false;
    }

    void growingPlanar(Mat& newPointsWithNormal,
                       std::vector<std::vector<int> >& idNs,
                       std::vector<Point3f>& normalNs,
                       std::vector<int>& idCenterNs,
                       std::vector<float>& timestepNs,
                       Mat& oldPointsWithNormal,
                       std::vector<std::vector<int> >& idQs,
                       std::vector<Point3f>& normalQs,
                       std::vector<int>& idCenterQs,
                       std::vector<float>& timestepQs,
                       Point6f& curCameraPos,
                       float thetaThreshold,
                       float timestepThreshold,
                       float timestepDisThreshold
    )
    {

    }
}
}