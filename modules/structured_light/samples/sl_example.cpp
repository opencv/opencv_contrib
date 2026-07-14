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

void readImages(vector<string> refs_files, vector<string> imgs_files, OutputArrayOfArrays refs, OutputArrayOfArrays imgs)
{
    vector<Mat>& refs_ = *(vector<Mat>*) refs.getObj();
    vector<Mat>& imgs_ = *(vector<Mat>*) imgs.getObj();

    for(uint i = 0; i < refs_files.size(); i++)
    {
        auto img = imread(refs_files[i], IMREAD_COLOR);
        cvtColor(img, img, COLOR_RGBA2GRAY);
        img.convertTo(img, CV_32FC1, 1.f/255);
        refs_.push_back(img);

        img = imread(imgs_files[i], IMREAD_COLOR);
        cvtColor(img, img, COLOR_RGBA2GRAY);
        img.convertTo(img, CV_32FC1, 1.f/255);
        imgs_.push_back(img);
    }
}

void captureImages(InputArrayOfArrays patterns, OutputArrayOfArrays refs, OutputArrayOfArrays imgs, cv::Size projector_size, bool isCaptureRefs)
{
    vector<Mat>& patterns_ = *(vector<Mat>*)patterns.getObj();
    vector<Mat>& refs_ = *(vector<Mat>*)refs.getObj();
    vector<Mat>& imgs_ = *(vector<Mat>*)imgs.getObj();

    VideoCapture cap;
    if(cap.open(0))
    {
        Mat pause(projector_size, CV_64FC3, Scalar(0));
        putText(pause, "Place the object", Point(projector_size.width/4, projector_size.height/4), FONT_HERSHEY_COMPLEX_SMALL, projector_size.width/400, Scalar(255,255,255), 2);
        putText(pause, "Press any key when ready", Point(projector_size.width/4, projector_size.height/4+projector_size.height/15), FONT_HERSHEY_COMPLEX_SMALL, projector_size.width/400, Scalar(255,255,255), 2);
        namedWindow("Display pattern", WINDOW_NORMAL);// Create a window for display.
        setWindowProperty("Display pattern", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
        imshow("Display pattern", pause);
        waitKey();

        if (isCaptureRefs)
        {
            for(uint i = 0; i < patterns_.size(); i++)
            {
                Mat frame;
                cap >> frame;
                if(frame.empty()) break; // end of video stream

                namedWindow("Display pattern", WINDOW_NORMAL);// Create a window for display.
                setWindowProperty("Display pattern", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
                imshow("Display pattern", patterns_[i]);
                waitKey();

                Mat grayFrame;
                cv::cvtColor(frame, grayFrame, COLOR_RGB2GRAY);
                grayFrame.convertTo(grayFrame, CV_32FC1, 1.f/255);
                refs_.push_back(grayFrame); //ADD ADDITIONAL SWITCH TO SELECT WHERE to SAVE

            }
        }

        pause = Mat(projector_size, CV_64FC3, Scalar(0));
        putText(pause, "Place the object", Point(projector_size.width/4, projector_size.height/4), FONT_HERSHEY_COMPLEX_SMALL, projector_size.width/400, Scalar(255,255,255), 2);
        putText(pause, "Press any key when ready", Point(projector_size.width/4, projector_size.height/4+projector_size.height/15), FONT_HERSHEY_COMPLEX_SMALL, projector_size.width/400, Scalar(255,255,255), 2);
        namedWindow("Display pattern", WINDOW_NORMAL);// Create a window for display.
        setWindowProperty("Display pattern", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
        imshow( "Display pattern", pause);
        waitKey();

        for(uint i = 0; i < patterns_.size(); i++)
        {
            Mat frame;
            cap >> frame;
            if( frame.empty() ) break; // end of video stream

            namedWindow("Display pattern", WINDOW_NORMAL);// Create a window for display.
            setWindowProperty("Display pattern", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
            imshow( "Display pattern", patterns_[i]);
            waitKey();

            Mat grayFrame;
            cv::cvtColor(frame, grayFrame, COLOR_RGB2GRAY);
            grayFrame.convertTo(grayFrame, CV_32FC1, 1.f/255);
            imgs_.push_back(grayFrame); //ADD ADDITIONAL SWITCH TO SELECT WHERE to SAVE

        }

        cap.release();
    }
}

int main( int argc, char **argv )
{
    int imgNum = 4;
    cv::Size projector_size = cv::Size(512, 512);
    string alg_type = "PCG";
    vector<cv::Mat> patterns, refs, imgs;

    structured_light::StructuredLightMono sl(projector_size, imgNum, 37, alg_type);

    sl.generatePatterns(patterns, 0.3f);

    captureImages(patterns, refs, imgs, projector_size, true);

    string filename = "../images/calibration_result.xml";
    Mat cameraMatrix, projectorMatrix, cameraDistortion, projectorDistortion, rotation, translation;
    structured_light::loadCalibrationData(filename, cameraMatrix, projectorMatrix, cameraDistortion, projectorDistortion, rotation, translation);


    for (uint i = 0; i < refs.size(); i++)
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