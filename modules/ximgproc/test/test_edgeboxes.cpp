// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(ximpgroc_Edgeboxes, regression)
{
	//Testing Edgeboxes implementation by asking for one proposal
	//on a simple test image from the PASCAL VOC 2012 dataset.
    std::vector<Rect> boxes;
    std::vector<float> scores;
    float expectedScore = 0.48742563;
    Rect expectedProposal(158, 69, 125, 154);

    //Using sample model file, compute orientations map for use with edge detection.
    cv::String testImagePath = cvtest::TS::ptr()->get_data_path() + "cv/ximgproc/" + "pascal_voc_bird.jpg";
    Mat testImg = imread(testImagePath);
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
    EXPECT_FLOAT_EQ(scores[0], expectedScore);
    EXPECT_TRUE(expectedProposal.x == boxes[0].x && expectedProposal.y == boxes[0].y && expectedProposal.height == boxes[0].height && expectedProposal.width == boxes[0].width);
}

}} // namespace
