// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/imgcodecs.hpp"

namespace opencv_test { namespace {

// Just skip test in case of missed testdata
static cv::String findDataFile(const String& path)
{
    return cvtest::findDataFile(path, false);
}


PARAM_TEST_CASE(Detection, std::string, bool)
{
    Ptr<ERFilter> er_filter1;
    Ptr<ERFilter> er_filter2;

    // SetUp doesn't handle SkipTestException
    void InitERFilter()
    {
        String nm1_file = findDataFile("trained_classifierNM1.xml");
        String nm2_file = findDataFile("trained_classifierNM2.xml");

        // Create ERFilter objects with the 1st and 2nd stage default classifiers
        er_filter1 = createERFilterNM1(loadClassifierNM1(nm1_file),16,0.00015f,0.13f,0.2f,true,0.1f);
        er_filter2 = createERFilterNM2(loadClassifierNM2(nm2_file),0.5);
    }
};

TEST_P(Detection, sample)
{
    InitERFilter();

    std::string imageName = GET_PARAM(0);
    bool anyDirection = GET_PARAM(1);
    std::cout << "Image: " << imageName << std::endl;
    std::cout << "Orientation: " << (anyDirection ? "any" : "horiz") << std::endl;
    Mat src = cv::imread(findDataFile(imageName));
    ASSERT_FALSE(src.empty());

    // Extract channels to be processed individually
    std::vector<Mat> channels;
    computeNMChannels(src, channels);

    // Append negative channels to detect ER- (bright regions over dark background)
    for (size_t c = channels.size(); c > 0; c--)
        channels.push_back(255 - channels[c - 1]);

    std::vector<std::vector<ERStat> > regions(channels.size());
    // Apply the default cascade classifier to each independent channel (could be done in parallel)
    for (size_t c = 0; c < channels.size(); c++)
    {
        er_filter1->run(channels[c], regions[c]);
        er_filter2->run(channels[c], regions[c]);
    }

    // Detect character groups
    std::vector< std::vector<Vec2i> > region_groups;
    std::vector<Rect> groups_boxes;
    if (!anyDirection)
        erGrouping(src, channels, regions, region_groups, groups_boxes, ERGROUPING_ORIENTATION_HORIZ);
    else
        erGrouping(src, channels, regions, region_groups, groups_boxes, ERGROUPING_ORIENTATION_ANY,
                   findDataFile("trained_classifier_erGrouping.xml"), 0.5);

    std::cout << "Found groups: " << groups_boxes.size() << std::endl;

    EXPECT_GT(groups_boxes.size(), 3u);
}

INSTANTIATE_TEST_CASE_P(Text, Detection,
    testing::Combine(
        testing::Values(
            "text/scenetext01.jpg",
            "text/scenetext02.jpg",
            "text/scenetext03.jpg",
            "text/scenetext04.jpg",
            "text/scenetext05.jpg",
            "text/scenetext06.jpg"
        ),
        testing::Bool()
    ));


}} // namespace
