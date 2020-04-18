// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST (TextDetectionSWT, accuracy_light_on_dark) {
    const string dataPath = cvtest::findDataFile("cv/mser/mser_test.png");
    Mat image = imread(dataPath, IMREAD_COLOR);
    vector<Rect> components;
    detectTextSWT(image, components, false);
    /* all 5 letter candidates should be identified (R9888) */
    EXPECT_EQ(5u, components.size());
}

TEST (TextDetectionSWT, accuracy_dark_on_light) {
    const string dataPath = cvtest::findDataFile("cv/mser/mser_test2.png");
    Mat image = imread(dataPath, IMREAD_COLOR);
    vector<Rect> components;
    detectTextSWT(image, components, true);
    /* all 3 letter candidates should be identified 2, 5, 8 */
    EXPECT_EQ(3u, components.size());
}

TEST (TextDetectionSWT, accuracy_handwriting) {
    const string dataPath = cvtest::findDataFile("cv/cloning/Mixed_Cloning/source1.png");
    Mat image = imread(dataPath, IMREAD_COLOR);
    vector<Rect> components;
    detectTextSWT(image, components, true);
    /* Handwritten Text is generally more difficult to detect using SWT algorithm due to high variation in stroke width. */
    EXPECT_LT(11u, components.size());
    /* Although the text contains 15 characters, the current implementation of algorithm outputs 14, including three wrong guesses. So, we check at least 11 (14 - 3) letters are detected.*/
}

TEST (TextDetectionSWT, accuracy_chaining) {
    const string dataPath = cvtest::findDataFile("cv/mser/mser_test.png");
    Mat image = imread(dataPath, IMREAD_COLOR);
    vector<Rect> components;
    Mat out(image.size(), CV_8UC3);
    vector<Rect> chains;
    detectTextSWT(image, components, false, out, chains);
    Rect chain = chains[0];
    /* Since the word is already segmented and cropped, most of the area is covered by text. It confirms that chaining works. */
    EXPECT_LT(0.95 * image.total(), (double)chain.area());
}

}} // namespace
