// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {


TEST(ximgproc_matchcolortemplate,test_QFFT)
{
    String openCVExtraDir = cvtest::TS::ptr()->get_data_path();
    String dataPath = openCVExtraDir;
#ifdef GENERATE_TESTDATA
    FileStorage fs;
    dataPath += "cv/ximgproc/sources/07.png";
    Mat imgTest = imread(dataPath, IMREAD_COLOR);
    resize(imgTest, imgTest, Size(), 0.0625, 0.0625);
    Mat qimgTest, qdftimgTest;
    ximgproc::createQuaternionImage(imgTest, qimgTest);
    ximgproc::qdft(qimgTest, qdftimgTest, 0, true);
    fs.open(openCVExtraDir + "cv/ximgproc/qdftData.yml.gz", FileStorage::WRITE);
    fs << "image" << imgTest;
    fs << "qdftleft" << qdftimgTest;
    ximgproc::qdft(qimgTest, qdftimgTest, 0, false);
    fs << "qdftright" << qdftimgTest;
    ximgproc::qdft(qimgTest, qdftimgTest, DFT_INVERSE, true);
    fs << "qidftleft" << qdftimgTest;
    ximgproc::qdft(qimgTest, qdftimgTest, DFT_INVERSE, false);
    fs << "qidftright" << qdftimgTest;
    fs.release();
#endif
    dataPath = openCVExtraDir + "cv/ximgproc/qdftData.yml.gz";
    FileStorage f;
    f.open(dataPath, FileStorage::READ);
    Mat img;
    f["image"] >> img;
    Mat qTest;
    vector<String> nodeName = { "qdftleft","qdftright","qidftleft","qidftright" };
    vector<int> flag = { 0,0,DFT_INVERSE,DFT_INVERSE };
    vector<bool> leftSize = {true,false,true,false};
    ximgproc::createQuaternionImage(img, img);
    for (int i=0;i<static_cast<int>(nodeName.size());i++)
    {
        Mat test, dd;
        f[nodeName[i]] >> qTest;
        ximgproc::qdft(img, test, flag[i], leftSize[i]);
        absdiff(test, qTest, dd);
        vector<Mat> plane;
        split(dd, plane);
        for (auto p : plane)
        {
            double maxVal;
            Point pIdx;
            minMaxLoc(p, NULL, &maxVal, NULL, &pIdx);
            ASSERT_LE(p.at<double>(pIdx), 1e-5);
        }
    }
}

TEST(ximgproc_matchcolortemplate, test_COLORMATCHTEMPLATE)
{
    String openCVExtraDir = cvtest::TS::ptr()->get_data_path();
    String dataPath = openCVExtraDir + "cv/ximgproc/corr.yml.gz";
    Mat img, logo;
    Mat corrRef,corr;
    img = imread(openCVExtraDir + "cv/ximgproc/image.png", IMREAD_COLOR);
    logo = imread(openCVExtraDir + "cv/ximgproc/opencv_logo.png", IMREAD_COLOR);
    ximgproc::colorMatchTemplate(img, logo, corr);
#ifdef GENERATE_TESTDATA
    FileStorage fs;
    fs.open(dataPath, FileStorage::WRITE);
    fs << "corr" << imgcorr;
    fs.release();
#endif
    FileStorage f;
    f.open(dataPath, FileStorage::READ);
    f["corr"] >> corrRef;
    EXPECT_LE(cv::norm(corr, corrRef, NORM_INF), 1e-5);
}
}} // namespace
