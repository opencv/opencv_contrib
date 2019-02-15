#include "opencv2/bgsegm.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/utility.hpp>
#include <iostream>

using namespace cv;
using namespace cv::bgsegm;

const String about =
    "\nA program demonstrating the use and capabilities of different background subtraction algrorithms\n"
    "Using OpenCV version " + String(CV_VERSION) +
    "\nPress q or ESC to exit\n";

const String keys =
        "{help h usage ? |      | print this message   }"
        "{vid            |      | path to a video file }"
        "{algo           | GMG  | name of the algorithm (GMG, CNT, KNN, MOG, MOG2) }"
        ;

static Ptr<BackgroundSubtractor> createBGSubtractorByName(const String& algoName)
{
    Ptr<BackgroundSubtractor> algo;
    if(algoName == String("GMG"))
        algo = createBackgroundSubtractorGMG(20, 0.7);
    else if(algoName == String("CNT"))
        algo = createBackgroundSubtractorCNT();
    else if(algoName == String("KNN"))
        algo = createBackgroundSubtractorKNN();
    else if(algoName == String("MOG"))
        algo = createBackgroundSubtractorMOG();
    else if(algoName == String("MOG2"))
        algo = createBackgroundSubtractorMOG2();

    return algo;
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);
    parser.printMessage();
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    String videoPath = parser.get<String>("vid");
    String algoName = parser.get<String>("algo");

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    Ptr<BackgroundSubtractor> bgfs = createBGSubtractorByName(algoName);
    if (!bgfs)
    {
        std::cerr << "Failed to create " << algoName << " background subtractor" << std::endl;
        return -1;
    }

    VideoCapture cap;
    if (argc > 1)
        cap.open(videoPath);
    else
        cap.open(0);

    if (!cap.isOpened())
    {
        std::cerr << "Cannot read video. Try moving video file to sample directory." << std::endl;
        return -1;
    }

    Mat frame, fgmask, segm;

    namedWindow("FG Segmentation", WINDOW_NORMAL);

    for (;;)
    {
        cap >> frame;

        if (frame.empty())
            break;

        bgfs->apply(frame, fgmask);

        frame.convertTo(segm, CV_8U, 0.5);
        add(frame, Scalar(100, 100, 0), segm, fgmask);

        imshow("FG Segmentation", segm);

        int c = waitKey(30);
        if (c == 'q' || c == 'Q' || (c & 255) == 27)
            break;
    }

    return 0;
}
