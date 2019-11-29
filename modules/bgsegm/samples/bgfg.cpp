#include "opencv2/bgsegm.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/utility.hpp>
#include <iostream>

using namespace cv;
using namespace cv::bgsegm;

const String about =
    "\nA program demonstrating the use and capabilities of different background subtraction algorithms\n"
    "Using OpenCV version " + String(CV_VERSION) +
    "\n\nPress 'c' to change the algorithm"
    "\nPress 'm' to toggle showing only foreground mask or ghost effect"
    "\nPress 'n' to change number of threads"
    "\nPress SPACE to toggle wait delay of imshow"
    "\nPress 'q' or ESC to exit\n";

const String algos[7] = { "GMG", "CNT", "KNN", "MOG", "MOG2", "GSOC", "LSBP" };

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
    else if(algoName == String("GSOC"))
        algo = createBackgroundSubtractorGSOC();
    else if(algoName == String("LSBP"))
        algo = createBackgroundSubtractorLSBP();

    return algo;
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, "{@video | vtest.avi | path to a video file}");
    parser.about(about);
    parser.printMessage();

    String videoPath = samples::findFile(parser.get<String>(0),false);

    Ptr<BackgroundSubtractor> bgfs = createBGSubtractorByName(algos[0]);

    VideoCapture cap;
    cap.open(videoPath);

    if (!cap.isOpened())
    {
        std::cerr << "Cannot read video. Try moving video file to sample directory." << std::endl;
        return -1;
    }

    Mat frame, fgmask, segm;

    int delay = 30;
    int algo_index = 0;
    int nthreads = getNumberOfCPUs();
    bool show_fgmask = false;

    for (;;)
    {
        cap >> frame;

        if (frame.empty())
        {
            cap.set(CAP_PROP_POS_FRAMES, 0);
            cap >> frame;
        }

        bgfs->apply(frame, fgmask);

        if (show_fgmask)
            segm = fgmask;
        else
        {
            frame.convertTo(segm, CV_8U, 0.5);
            add(frame, Scalar(100, 100, 0), segm, fgmask);
        }

        putText(segm, algos[algo_index], Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 0, 255), 2, LINE_AA);
        putText(segm, format("%d threads", nthreads), Point(10, 60), FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 0, 255), 2, LINE_AA);

        imshow("FG Segmentation", segm);

        int c = waitKey(delay);

        if (c == ' ')
            delay = delay == 30 ? 1 : 30;

        if (c == 'c' || c == 'C')
        {
            algo_index++;
            if ( algo_index > 6 )
                 algo_index = 0;

            bgfs = createBGSubtractorByName(algos[algo_index]);
        }

        if (c == 'n' || c == 'N')
        {
            nthreads++;
            if ( nthreads > 8 )
                 nthreads = 1;

            setNumThreads(nthreads);
        }

        if (c == 'm' || c == 'M')
            show_fgmask = !show_fgmask;

        if (c == 'q' || c == 'Q' || c  == 27)
            break;
    }

    return 0;
}
