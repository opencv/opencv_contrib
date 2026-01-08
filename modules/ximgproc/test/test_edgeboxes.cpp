// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(ximgproc_Edgeboxes, regression)
{
    //Testing Edgeboxes implementation by asking for one proposal
    //on a simple test image from the PASCAL VOC 2012 dataset.
    std::vector<Rect> boxes;
    std::vector<float> scores;
    float expectedScore = 0.48742563f;
    Rect expectedProposal(158, 69, 125, 154);

    //Using sample model file, compute orientations map for use with edge detection.
    cv::String testImagePath = cvtest::TS::ptr()->get_data_path() + "cv/ximgproc/" + "pascal_voc_bird.png";
    Mat testImg = imread(testImagePath);
    ASSERT_FALSE(testImg.empty()) << "Could not load input image " << testImagePath;
    cvtColor(testImg, testImg, COLOR_BGR2RGB);
    testImg.convertTo(testImg, CV_32F, 1.0 / 255.0f);

    //Use the model for structured edge detection that is already provided in opencv_extra.
    cv::String model_path = cvtest::TS::ptr()->get_data_path() + "cv/ximgproc/" + "model.yml.gz";
    Ptr<StructuredEdgeDetection> sed = createStructuredEdgeDetection(model_path);
    Mat edgeImage, edgeOrientations;
    sed->detectEdges(testImg, edgeImage);
    sed->computeOrientation(edgeImage, edgeOrientations);

    //Obtain one proposal and its score from Edgeboxes.
    Ptr<EdgeBoxes> edgeboxes = createEdgeBoxes();
    edgeboxes->setMaxBoxes(1);
    edgeboxes->getBoundingBoxes(edgeImage, edgeOrientations, boxes, scores);

    //We asked for one proposal and thus one score, we better get one back only.
    ASSERT_TRUE(boxes.size() == 1);
    ASSERT_TRUE(scores.size() == 1);

    //Check the proposal and its score.
    EXPECT_NEAR(scores[0], expectedScore, 1e-8);
    EXPECT_EQ(expectedProposal.x, boxes[0].x);
    EXPECT_EQ(expectedProposal.y, boxes[0].y);
    EXPECT_EQ(expectedProposal.height, boxes[0].height);
    EXPECT_EQ(expectedProposal.width, boxes[0].width);
}

}} // namespace
