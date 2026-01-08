#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/intensity_transform.hpp"

#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::intensity_transform;

namespace
{
static std::string keys =
    "{ help  h     | | Print help message. }"
    "{ input i     | | Path to the input image. }";

// global variables
Mat g_image;

int g_gamma = 40;
const int g_gammaMax = 500;
Mat g_imgGamma;
const std::string g_gammaWinName = "Gamma Correction";

Mat g_contrastStretch;
int g_r1 = 70;
int g_s1 = 15;
int g_r2 = 120;
int g_s2 = 240;
const std::string g_contrastWinName = "Contrast Stretching";

Mat g_imgBIMEF;
int g_mu = 50;
const int g_muMax = 100;
const std::string g_BIMEFWinName = "BIMEF";

static void onTrackbarGamma(int, void*)
{
    float gamma = g_gamma / 100.0f;
    gammaCorrection(g_image, g_imgGamma, gamma);
    imshow(g_gammaWinName, g_imgGamma);
}

static void onTrackbarContrastR1(int, void*)
{
    contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);
    imshow("Contrast Stretching", g_contrastStretch);
}

static void onTrackbarContrastS1(int, void*)
{
    contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);
    imshow("Contrast Stretching", g_contrastStretch);
}

static void onTrackbarContrastR2(int, void*)
{
    contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);
    imshow("Contrast Stretching", g_contrastStretch);
}

static void onTrackbarContrastS2(int, void*)
{
    contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);
    imshow("Contrast Stretching", g_contrastStretch);
}

static void onTrackbarBIMEF(int, void*)
{
    float mu = g_mu / 100.0f;
    BIMEF(g_image, g_imgBIMEF, mu);
    imshow(g_BIMEFWinName, g_imgBIMEF);
}
}

int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, keys);

    const std::string inputFilename = parser.get<String>("input");
    parser.about("Use this script to apply intensity transformation on an input image.");
    if (parser.has("help") || inputFilename.empty())
    {
        parser.printMessage();
        return 0;
    }

    // Read input image
    g_image = imread(inputFilename);

    // Create trackbars
    namedWindow(g_gammaWinName);
    createTrackbar("Gamma value", g_gammaWinName, &g_gamma, g_gammaMax, onTrackbarGamma);

    namedWindow(g_contrastWinName);
    createTrackbar("Contrast R1", g_contrastWinName, &g_r1, 256, onTrackbarContrastR1);
    createTrackbar("Contrast S1", g_contrastWinName, &g_s1, 256, onTrackbarContrastS1);
    createTrackbar("Contrast R2", g_contrastWinName, &g_r2, 256, onTrackbarContrastR2);
    createTrackbar("Contrast S2", g_contrastWinName, &g_s2, 256, onTrackbarContrastS2);

    namedWindow(g_BIMEFWinName);
    createTrackbar("Enhancement ratio mu", g_BIMEFWinName, &g_mu, g_muMax, onTrackbarBIMEF);

    // Apply intensity transformations
    Mat imgAutoscaled, imgLog;
    autoscaling(g_image, imgAutoscaled);
    gammaCorrection(g_image, g_imgGamma, g_gamma/100.0f);
    logTransform(g_image, imgLog);
    contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);
    BIMEF(g_image, g_imgBIMEF, g_mu / 100.0f);

    // Display intensity transformation results
    imshow("Original Image", g_image);
    imshow("Autoscale", imgAutoscaled);
    imshow(g_gammaWinName, g_imgGamma);
    imshow("Log Transformation", imgLog);
    imshow(g_contrastWinName, g_contrastStretch);
    imshow(g_BIMEFWinName, g_imgBIMEF);

    waitKey(0);
    return 0;
}
