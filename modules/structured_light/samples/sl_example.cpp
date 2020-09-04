#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/structured_light.hpp>

using namespace std;

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
    int imgNum = 4;
    string alg_type = "PCG";
    vector<cv::Mat> patterns, refs, imgs;

    cv::structured_light::StructuredLightMono sl(projector_size, imgNum, 37, alg_type);

    sl.generatePatterns(patterns, 0.3);

    sl.captureImages(patterns, refs, imgs);

    string filename = "../images/calibration_result.xml";
    cv::Mat cameraMatrix, projectorMatrix, cameraDistortion, projectorDistortion, rotation, translation;
    cv::structured_light::loadCalibrationData(filename, cameraMatrix, projectorMatrix, cameraDistortion, projectorDistortion, rotation, translation);


    for (auto i = 0; i < refs.size(); i++)
    {
        cv::Mat undistored;
        cv::undistort(refs[i], undistored, cameraMatrix, cameraDistortion);
        cv::GaussianBlur(undistored, refs[i], cv::Size(5, 5), 0);

        cv::undistort(imgs[i], undistored, cameraMatrix, cameraDistortion);
        cv::GaussianBlur(undistored, imgs[i], cv::Size(5, 5), 0);
    }

    /*
    vector<string> refs_files{"../images/hf_ref0.png", "../images/hf_ref1.png", "../images/hf_ref2.png", "../images/hf_ref3.png",
                              "../images/lf_ref0.png", "../images/lf_ref1.png", "../images/lf_ref2.png", "../images/lf_ref3.png"};

    vector<string> imgs_files{"../images/hf_phase0.png", "../images/hf_phase1.png", "../images/hf_phase2.png", "../images/hf_phase3.png",
                              "../images/lf_phase0.png", "../images/lf_phase1.png", "../images/lf_phase2.png", "../images/lf_phase3.png"};

    sl.readImages(refs_files, imgs_files, refs, imgs);
    */

    cv::Mat phase;

    sl.unwrapPhase(refs, imgs, phase);

    double min, max;
    cv::minMaxLoc(phase, &min, &max);
    phase -= min;
    phase.convertTo(phase, CV_32FC1, 1.f/(max-min));

    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Display window", phase );                   // Show our image inside it.

    cv::waitKey();

    return 0;
}