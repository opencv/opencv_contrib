#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/structured_light.hpp>

using namespace std;
using namespace cv;

// 1. Calibrate camera=projector
// 2. Save calibration params
// 3. Generate projected pattern
// 4. Distort projected pattern
// 5. Capture patterns reference pattern - optional
// 6. Capture patterns with object
// 7. Image preprocessing
// 7. Load refs and images to unwrap algorithm
// 8. Save points to file

int main()
{
    cv::Size projector_size = cv::Size(512, 512);
    string alg_type = "PCG";
    vector<cv::Mat> patterns, refs, imgs;

    structured_light::StructuredLightMono sl(projector_size, imgNum, 37, alg_type);

    sl.generatePatterns(patterns, 0.3);

    sl.captureImages(patterns, refs, imgs);

    string filename = "../images/calibration_result.xml";
    Mat cameraMatrix, projectorMatrix, cameraDistortion, projectorDistortion, rotation, translation;
    structured_light::loadCalibrationData(filename, cameraMatrix, projectorMatrix, cameraDistortion, projectorDistortion, rotation, translation);


    for (unsigned i = 0; i < refs.size(); i++)
    {
        Mat undistored;
        undistort(refs[i], undistored, cameraMatrix, cameraDistortion);
        GaussianBlur(undistored, refs[i], cv::Size(5, 5), 0);

        undistort(imgs[i], undistored, cameraMatrix, cameraDistortion);
        GaussianBlur(undistored, imgs[i], cv::Size(5, 5), 0);
    }

    Mat phase;

    sl.unwrapPhase(refs, imgs, phase);

    double min, max;
    minMaxLoc(phase, &min, &max);
    phase -= min;
    phase.convertTo(phase, CV_32FC1, 1.f/(max-min));

    namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", phase );                   // Show our image inside it.

    cv::waitKey();

    return 0;
}