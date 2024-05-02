// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#include <vector>

namespace opencv_test
{
namespace
{

using namespace std;
/****************************************************************************************\
 *                Test drawing works properly
\****************************************************************************************/

void runCCheckerDraw(Ptr<CChecker> pChecker, int rows, int cols, unsigned int number_of_cells_in_colorchecker)
{
    cv::Mat img(rows, cols, CV_8UC3, {0, 0, 0});

    Ptr<CCheckerDraw> cdraw = CCheckerDraw::create(pChecker);

    cdraw->draw(img);

    //make sure this contains extacly as many rectangles as in the pChecker
    vector<vector<Point>> contours;
    cv::cvtColor(img, img, COLOR_BGR2GRAY);
    findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    ASSERT_EQ(contours.size(), number_of_cells_in_colorchecker);
}

TEST(CV_mccRunCCheckerDrawTest, accuracy_MCC24)
{
    Ptr<CChecker> pChecker = CChecker::create();
    pChecker->setTarget(MCC24);
    pChecker->setBox({{0, 0}, {480, 0}, {480, 640}, {0, 640}});
    runCCheckerDraw(pChecker, 640, 480, 24);
}
TEST(CV_mccRunCCheckerDrawTest, accuracy_SG140)
{
    Ptr<CChecker> pChecker = CChecker::create();
    pChecker->setTarget(SG140);
    pChecker->setBox({{0, 0}, {480, 0}, {480, 640}, {0, 640}});
    runCCheckerDraw(pChecker, 640, 480, 140);
}
TEST(CV_mccRunCCheckerDrawTest, accuracy_VINYL18)
{
    Ptr<CChecker> pChecker = CChecker::create();
    pChecker->setTarget(VINYL18);
    pChecker->setBox({{0, 0}, {480, 0}, {480, 640}, {0, 640}});
    runCCheckerDraw(pChecker, 640, 480, 18);
}

/****************************************************************************************\
 *                Test detection works properly on the simplest images
\****************************************************************************************/

void runCCheckerDetectorBasic(std::string image_name, TYPECHART chartType)
{
    Ptr<CCheckerDetector> detector = CCheckerDetector::create();
    std::string path = cvtest::findDataFile("mcc/" + image_name);
    cv::Mat img = imread(path);
    ASSERT_FALSE(img.empty()) << "Test image can't be loaded: " << path;

    ASSERT_TRUE(detector->process(img, chartType));
}
TEST(CV_mccRunCCheckerDetectorBasic, accuracy_SG140)
{
    runCCheckerDetectorBasic("SG140.png", SG140);
}
TEST(CV_mccRunCCheckerDetectorBasic, accuracy_MCC24)
{
    runCCheckerDetectorBasic("MCC24.png", MCC24);
}

TEST(CV_mccRunCCheckerDetectorBasic, accuracy_VINYL18)
{
    runCCheckerDetectorBasic("VINYL18.png", VINYL18);
}

TEST(CV_mcc_ccm_test, detect_Macbeth)
{
    string path = cvtest::findDataFile("mcc/mcc_ccm_test.jpg");
    Mat img = imread(path, IMREAD_COLOR);
    Ptr<CCheckerDetector> detector = CCheckerDetector::create();

    // detect MCC24 board
    ASSERT_TRUE(detector->process(img, MCC24, 1, false));

    // read gold Macbeth corners
    path = cvtest::findDataFile("mcc/mcc_ccm_test.yml");
    FileStorage fs(path, FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());
    FileNode node = fs["Macbeth_corners"];
    ASSERT_FALSE(node.empty());
    vector<Point2f> gold_corners;
    node >> gold_corners;
    Ptr<CChecker> checker = detector->getBestColorChecker();

    // check Macbeth corners
    vector<Point2f> corners = checker->getBox();
    EXPECT_MAT_NEAR(gold_corners, corners, 3.6); // diff 3.57385 in ARM only

    // read gold chartsRGB
    node = fs["chartsRGB"];
    Mat goldChartsRGB;
    node >> goldChartsRGB;
    fs.release();

    // check chartsRGB
    Mat chartsRGB = checker->getChartsRGB();
    EXPECT_MAT_NEAR(goldChartsRGB.col(1), chartsRGB.col(1), 0.25); // diff 0.240634 in ARM only
}

TEST(CV_mcc_ccm_test, compute_ccm)
{
    // read gold chartsRGB
    string path = cvtest::findDataFile("mcc/mcc_ccm_test.yml");
    FileStorage fs(path, FileStorage::READ);
    Mat chartsRGB;
    FileNode node = fs["chartsRGB"];
    node >> chartsRGB;

    // compute CCM
    ColorCorrectionModel model(chartsRGB.col(1).clone().reshape(3, chartsRGB.rows/3) / 255., COLORCHECKER_Macbeth);
    model.run();

    // read gold CCM
    node = fs["ccm"];
    ASSERT_FALSE(node.empty());
    Mat gold_ccm;
    node >> gold_ccm;
    fs.release();

    // check CCM
    Mat ccm = model.getCCM();
    EXPECT_MAT_NEAR(gold_ccm, ccm, 1e-8);

    const double gold_loss = 4.6386569120323129;
    // check loss
    const double loss = model.getLoss();
    EXPECT_NEAR(gold_loss, loss, 1e-8);
}

TEST(CV_mcc_ccm_test, infer)
{
    string path = cvtest::findDataFile("mcc/mcc_ccm_test.jpg");
    Mat img = imread(path, IMREAD_COLOR);
    // read gold calibrate img
    path = cvtest::findDataFile("mcc/mcc_ccm_test_res.png");
    Mat gold_img = imread(path);

    // read gold chartsRGB
    path = cvtest::findDataFile("mcc/mcc_ccm_test.yml");
    FileStorage fs(path, FileStorage::READ);
    Mat chartsRGB;
    FileNode node = fs["chartsRGB"];
    node >> chartsRGB;
    fs.release();

    // compute CCM
    ColorCorrectionModel model(chartsRGB.col(1).clone().reshape(3, chartsRGB.rows/3) / 255., COLORCHECKER_Macbeth);
    model.run();

    // compute calibrate image
    Mat calibratedImage;
    cvtColor(img, calibratedImage, COLOR_BGR2RGB);
    calibratedImage.convertTo(calibratedImage, CV_64F, 1. / 255.);
    calibratedImage = model.infer(calibratedImage);
    calibratedImage.convertTo(calibratedImage, CV_8UC3, 255.);
    cvtColor(calibratedImage, calibratedImage, COLOR_RGB2BGR);
    // check calibrated image
    EXPECT_MAT_NEAR(gold_img, calibratedImage, 0.1);
}


} // namespace
} // namespace opencv_test
