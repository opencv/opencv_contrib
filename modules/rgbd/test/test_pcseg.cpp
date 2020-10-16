// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
//
// Created by YIMIN TANG on 10/11/20.
//


#include "test_precomp.hpp"
namespace opencv_test { namespace {
using namespace cv;
class CV_RgbdDepthPcsegTest: public cvtest::BaseTest
{
public:
    CV_RgbdDepthPcsegTest()
    {
    }
    ~CV_RgbdDepthPcsegTest()
    {
    }
protected:

    bool test_angleBetween()
    {
        printf("\ntest_angleBetween\n");
        Point3f p1(0,1,0);
        Point3f p2(0,0,1);
        printf("%f, Should be %f\n", pcseg::angleBetween(p1,p2), M_PI/2);
        p1 = Point3f(0,1,0);
        p2 = Point3f(0,1,0);
        printf("%f, Should be %f\n", pcseg::angleBetween(p1,p2), 0.);
        p1 = Point3f(1,1,0);
        p2 = Point3f(-1,1,0);
        printf("%f, Should be %f\n", pcseg::angleBetween(p1,p2), M_PI/2);
        p1 = Point3f(-1,0,0);
        p2 = Point3f(1,1,0);
        printf("%f, Should be %f\n", pcseg::angleBetween(p1,p2), M_PI - M_PI/4);
        return 1;
    }

    bool test_calCurvatures()
    {
        printf("\ntest_calCurvatures\n");
        float data[3][6] = {
                {0,0,1.1, 1.1,0,0},
                {0,1.2,0, 0,1.2,0},
                {1.3,0,0, 0,0,1.3}};
        Mat A = Mat(3, 6, CV_32FC1, &data);
        std::vector<Point3f> points;
        std::vector<Point3f> normal;
        std::vector<float> curvatures;
        pcseg::calCurvatures(A,3,points,normal,curvatures);
        for (int i=0;i< (int) curvatures.size();i++)
            printf("%f ",curvatures[i]);
        printf("\n");
        return 1;
    }

    bool test_planarSegments()
    {
        printf("\ntest_planarSegments\n");
        float data[6][6] = {
                {  0,   0,   1,   0,   0,   1},
                {  0,   0,   2,   0,   0,   1},
                {  0,   0,   3,   0,   0,   1},
                {100,   0,   1,   0,   0,   1},
                {100,   0,   2,   0,   0,   1},
                {100,   0,   3,   0,   0,   1}};
        Mat A = Mat(3, 6, CV_32FC1, &data);
        int k = 3;
        std::vector<Point3f> points;
        std::vector<Point3f> normal;
        std::vector<float> curvatures;
        pcseg::calCurvatures(A,k,points,normal,curvatures);
        std::vector<std::vector<Point3f> > vecRetPoints;
        std::vector<std::vector<Point3f> > vecRetNormals;
        pcseg::planarSegments(points, normal, curvatures, k, 10.0/360*2*M_PI, 0.1, vecRetPoints, vecRetNormals);
        return 1;
    }

    bool test_planarMerge()
    {
        printf("\ntest_planarMerge\n");
        std::vector<Point3f> pointsA = {
                {  0,   0,   1},
                {  0,   0,   1},
                {  0,   0,   2},
                {  0,   0,   3}};
        std::vector<Point3f> normalsA = {
                {  0,   0,   1},
                {  0,   0,   1},
                {  0,   0,   1},
                {  0,   0,   1}};

        std::vector<Point3f> pointsB = {
                {  0,   0,   4},
                {  0,   0,   4},
                {  0,   0,   5},
                {  0,   0,   6}};
        std::vector<Point3f> normalsB = {
                {  0,   0,   1},
                {  0,   0,   1},
                {  0,   0,   1},
                {  0,   0,   1}};
        int timesteps = 0;
        bool result = pcseg::planarMerge(pointsA, normalsA, timesteps, pointsB, normalsB, 10);
        printf("%d\n", result);
        for (int i=0;i< (int) pointsA.size();i++) std::cout<<pointsA[i]<<' ';
        return 1;
    }

    // algo 1
    bool test_showable_planarSegments_oneFrame()
    {
        printf("\ntest_showable_planarSegments\n");
        String file1 = "/Users/yimintang/OneDrive/Documents/seg/rgbd_dataset_freiburg1_room/plyfiles/plyfiles-1305031910.765238.ply";
        Mat pointsWithNormal1 = ppf_match_3d::loadPLYSimple(file1.c_str(), 1);
        int k = 15;
        std::vector<Point3f> points, normals;
        std::vector<float> curvatures;
        pcseg::calCurvatures(pointsWithNormal1, k, points, normals, curvatures);
        std::vector<std::vector<Point3f> > retPointsA, retNormalsA;
        pcseg::planarSegments(points, normals, curvatures, k, 10.0/360*M_PI*2, 0.1, retPointsA, retNormalsA);
        printf("retPoints size: %lu\n", retPointsA.size());
        for (int i=0;i<retPointsA.size();i++)
        {
            for (int j=0;j<retPointsA[i].size();j++)
                printf("%f %f %f %d\n", retPointsA[i][j].x,
                       retPointsA[i][j].y,
                       retPointsA[i][j].z, i);
        }
        return 1;
    }

    // algo 1
    bool test_showable_planarSegments_twoFrame()
    {
        printf("\ntest_showable_planarSegments\n");
        String file1 = "/Users/yimintang/OneDrive/Documents/seg/rgbd_dataset_freiburg1_room/plyfiles/plyfiles-1305031910.765238.ply";
        String file2 = "/Users/yimintang/OneDrive/Documents/seg/rgbd_dataset_freiburg1_room/plyfiles/plyfiles-1305031911.097196.ply";
        Mat pointsWithNormal1 = ppf_match_3d::loadPLYSimple(file1.c_str(), 1);
        Mat pointsWithNormal2 = ppf_match_3d::loadPLYSimple(file2.c_str(), 1);
        int k = 15;
        std::vector<Point3f> points, normals;
        std::vector<float> curvatures;
        pcseg::calCurvatures(pointsWithNormal1, k, points, normals, curvatures);
        std::vector<std::vector<Point3f> > retPointsA, retNormalsA;
        pcseg::planarSegments(points, normals, curvatures, k, 10.0/360*M_PI*2, 0.1, retPointsA, retNormalsA);
        printf("retPoints size: %lu\n", retPointsA.size());

        pcseg::calCurvatures(pointsWithNormal2, k, points, normals, curvatures);
        pcseg::planarSegments(points, normals, curvatures, k, 10.0/360*M_PI*2, 0.1, retPointsA, retNormalsA);
        printf("retPoints size: %lu\n", retPointsA.size());
        for (int i=0;i<retPointsA.size();i++)
        {
            for (int j=0;j<retPointsA[i].size();j++)
                printf("%f %f %f %d\n", retPointsA[i][j].x,
                       retPointsA[i][j].y,
                       retPointsA[i][j].z, i);
        }
        return 1;
    }


    // algo 1
    // algo 3
    //      algo 2
    // algo 4
    //      algo 2
    bool test_showable_planarSegments_twoFrame_growingPlanar_mergeCloseSegments()
    {
        printf("\ntest_growingPlanar\n");
        String file1 = "/Users/yimintang/OneDrive/Documents/seg/rgbd_dataset_freiburg1_room/plyfiles/plyfiles-1305031910.765238.ply";
        String file2 = "/Users/yimintang/OneDrive/Documents/seg/rgbd_dataset_freiburg1_room/plyfiles/plyfiles-1305031911.097196.ply";
        Mat pointsWithNormal1 = ppf_match_3d::loadPLYSimple(file1.c_str(), 1);
        Mat pointsWithNormal2 = ppf_match_3d::loadPLYSimple(file2.c_str(), 1);
        int k = 20;
        std::vector<Point3f> points, normals;
        std::vector<float> curvatures;
        pcseg::calCurvatures(pointsWithNormal1, k, points, normals, curvatures);
        std::vector<std::vector<Point3f> > retPointsA, retNormalsA;
        pcseg::planarSegments(points, normals, curvatures, k, 10.0/360*M_PI*2, 0.1, retPointsA, retNormalsA);
        printf("retPointsA size: %lu\n", retPointsA.size());

        std::vector<std::vector<Point3f> > retPointsB, retNormalsB;
        pcseg::calCurvatures(pointsWithNormal2, k, points, normals, curvatures);
        pcseg::planarSegments(points, normals, curvatures, k, 10.0/360*M_PI*2, 0.1, retPointsB, retNormalsB);
        printf("retPointsB size: %lu\n", retPointsB.size());

        std::vector<int> timestepsA(retPointsA.size(),0);
        std::vector<int> timestepsB(retPointsB.size(),0);

        Point3f curCameraPos(-0.8739,0.6915,1.5594);
        std::vector< std::pair<int,int> > retS;
        pcseg::growingPlanar(
                retPointsA, retNormalsA, timestepsA,
                retPointsB, retNormalsB, timestepsB,
                curCameraPos, retS
        );

        std::vector<int> alphaS(retS.size(),0);
        pcseg::mergeCloseSegments(retS, alphaS, retPointsA, retNormalsA, timestepsA);
        for (int i=0;i<retPointsA.size();i++)
        {
            for (int j=0;j<retPointsA[i].size();j++)
                printf("%f %f %f %d\n", retPointsA[i][j].x,
                       retPointsA[i][j].y,
                       retPointsA[i][j].z, i);
        }
        return 1;
    }

    void
    run(int)
    {
        int tests_pass = 0;
        try {
//            tests_pass += test_showable_planarSegments_oneFrame();
//            tests_pass += test_showable_planarSegments_twoFrame();
            tests_pass += test_showable_planarSegments_twoFrame_growingPlanar_mergeCloseSegments();

        } catch (...)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_ERROR_IN_CALLED_FUNC);
        }
        ts->set_failed_test_info(cvtest::TS::OK);
    }


};

TEST(Rgbd_DepthPcseg, compute)
{
    CV_RgbdDepthPcsegTest test;
    test.safe_run();
}


}
}


