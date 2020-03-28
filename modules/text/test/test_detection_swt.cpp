// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace cv;
using namespace std;
namespace opencv_test { namespace {

TEST (TextDetectionSWT, accuracy_light_on_dark) {
    std::vector<cv::Rect> letters;
    String dataPath = cvtest::TS::ptr()->get_data_path() + "cv/mser/mser_test.png";
    Mat image = imread(dataPath, IMREAD_COLOR);
    vector<cv::Rect> components;
    cv::text::detectTextSWT(image, components, false);
    cout << components.size();
    /* all 5 letter candidates should be identified (R9888) */
    EXPECT_EQ(components.size(), (unsigned) 5);
}

TEST (TextDetectionSWT, accuracy_dark_on_light) {
    std::vector<cv::Rect> letters;
    String dataPath = cvtest::TS::ptr()->get_data_path() + "cv/mser/mser_test2.png";
    Mat image = imread(dataPath, IMREAD_COLOR);
    vector<cv::Rect> components;
    cv::text::detectTextSWT(image, components, true);
    cout << components.size();
    /* all 3 letter candidates should be identified 2, 5, 8 */
    EXPECT_EQ(components.size(), (unsigned) 3);
}

TEST (TextDetectionSWT, accuracy_handwriting) {
    std::vector<cv::Rect> letters;
    String dataPath = cvtest::TS::ptr()->get_data_path() + "cv/cloning/Mixed_Cloning/source1.png";
    Mat image = imread(dataPath, IMREAD_COLOR);
    vector<cv::Rect> components;
    cv::text::detectTextSWT(image, components, true);
    cout << components.size();
    /* Handwritten Text is generally more difficult to detect using SWT algorithm due to high variation in stroke width. */
    EXPECT_GT(components.size(), (unsigned) 11);
    /* Although the text contains 15 characters, the current implementation of algorithm outputs 14, including three wrong guesses. So, we check at least 11 (14 - 3) letters are detected.*/
}

TEST (TextDetectionSWT, regression_natural_scene) {
    std::vector<cv::Rect> letters;
    String dataPath = cvtest::TS::ptr()->get_data_path() + "cv/shared/box_in_scene.png";
    Mat image = imread(dataPath, IMREAD_COLOR);
    vector<cv::Rect> light_components;
    cv::text::detectTextSWT(image, light_components, false);
    cout << light_components.size();
    EXPECT_EQ(light_components.size(), (unsigned) 68);

    vector<cv::Rect> dark_components;
    cv::text::detectTextSWT(image, dark_components, true);
    cout << dark_components.size();
    EXPECT_EQ(dark_components.size(), (unsigned) 14);
    /* Verifies that both modes of algorithm run on natural scenes */
}

TEST (TextDetectionSWT, accuracy_chaining) {
    std::vector<cv::Rect> letters;
    String dataPath = cvtest::TS::ptr()->get_data_path() + "cv/mser/mser_test.png";
    Mat image = imread(dataPath, IMREAD_COLOR);
    vector<cv::Rect> components;
    Mat out( image.size(), CV_8UC3 );
    vector<cv::Rect> chains;
    cv::text::detectTextSWT(image, components, false, out, chains);
    cv::Rect chain = chains[0];
    /* Since the word is already segmented and cropped, most of the area is covered by text. It confirms that chaining works. */
    EXPECT_GT(chain.area(), 0.95 * image.rows * image.cols);
}

}} // namespace
