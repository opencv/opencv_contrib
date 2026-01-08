#include <iostream>

#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

// Adapted from cv_timer in cv_utilities
class Timer
{
public:
  Timer() : start_(0), time_(0) {}

  void start()
  {
    start_ = cv::getTickCount();
  }

  void stop()
  {
    CV_Assert(start_ != 0);
    int64 end = cv::getTickCount();
    time_ += end - start_;
    start_ = 0;
  }

  double time()
  {
    double ret = time_ / cv::getTickFrequency();
    time_ = 0;
    return ret;
  }

private:
  int64 start_, time_;
};

static void help()
{

printf("\nAllows to estimate the efficiency of the morphology operations implemented\n"
    "in ximgproc/run_length_morphology.cpp\n"
    "Call:\n  example_ximgproc_run_length_morphology_demo [image] -u=factor_upscaling image\n"
    "Similar to the morphology2 sample of the main opencv library it shows the use\n"
    "of rect, ellipse and cross kernels\n\n"
    "As rectangular and cross-shaped structuring elements are highly optimized in opencv_imgproc module,\n"
    "only with elliptical structuring elements a speedup is possible (e.g. for larger circles).\n"
    "Run-length morphology has advantages for larger images.\n"
    "You can verify this by upscaling your input with e.g. -u=2\n");
printf( "Hot keys: \n"
    "\tESC - quit the program\n"
    "\tr - use rectangle structuring element\n"
    "\te - use elliptic structuring element\n"
    "\tc - use cross-shaped structuring element\n"
    "\tSPACE - loop through all the options\n" );
}

static void print_introduction()
{
    printf("\nFirst select a threshold for binarization.\n"
        "Then move the sliders for erosion/dilation or open/close operation\n\n"
        "The ratio between the time of the execution from opencv_imgproc\n"
        "and the code using run-length encoding will be displayed in the console\n\n");
}

Mat src, dst;

int element_shape = MORPH_ELLIPSE;

//the address of variable which receives trackbar position update
int max_size = 40;
int open_close_pos = 0;
int erode_dilate_pos = 0;
int nThreshold = 100;
cv::Mat binaryImage;
cv::Mat binaryRLE, dstRLE;
cv::Mat rlePainted;

static void PaintRLEToImage(cv::Mat& rleImage, cv::Mat& res, unsigned char uValue)
{
    res = cv::Scalar(0);
    rl::paint(res, rleImage, Scalar((double) uValue));
}


static bool AreImagesIdentical(cv::Mat& image1, cv::Mat& image2)
{
    cv::Mat diff;
    cv::absdiff(image1, image2, diff);
    int nDiff = cv::countNonZero(diff);
    return (nDiff == 0);
}

// callback function for open/close trackbar
static void OpenClose(int, void*)
{
    int n = open_close_pos - max_size;
    int an = n > 0 ? n : -n;
    Mat element = getStructuringElement(element_shape, Size(an*2+1, an*2+1), Point(an, an) );
    Timer timer;
    timer.start();
    if( n < 0 )
        morphologyEx(binaryImage, dst, MORPH_OPEN, element);
    else
        morphologyEx(binaryImage, dst, MORPH_CLOSE, element);
    timer.stop();
    double imgproc_duration = timer.time();

    element = rl::getStructuringElement(element_shape, Size(an * 2 + 1, an * 2 + 1));

    Timer timer2;
    timer2.start();
    if (n < 0)
        rl::morphologyEx(binaryRLE, dstRLE, MORPH_OPEN, element, true);
    else
        rl::morphologyEx(binaryRLE, dstRLE, MORPH_CLOSE, element, true);

    timer2.stop();
    double rl_duration = timer2.time();
    cout << "ratio open/close duration: " << rl_duration / imgproc_duration << " (run-length: "
        << rl_duration << ", pixelwise: " << imgproc_duration << " )" << std::endl;

    PaintRLEToImage(dstRLE, rlePainted, (unsigned char)255);
    if (!AreImagesIdentical(dst, rlePainted))
    {
        cout << "error result image are not identical" << endl;
    }

    imshow("Open/Close", rlePainted);
}

// callback function for erode/dilate trackbar
static void ErodeDilate(int, void*)
{
    int n = erode_dilate_pos - max_size;
    int an = n > 0 ? n : -n;
    Mat element = getStructuringElement(element_shape, Size(an*2+1, an*2+1), Point(an, an) );
    Timer timer;
    timer.start();
    if( n < 0 )
        erode(binaryImage, dst, element);
    else
        dilate(binaryImage, dst, element);
    timer.stop();
    double imgproc_duration = timer.time();

    element = rl::getStructuringElement(element_shape, Size(an*2+1, an*2+1));

    Timer timer2;
    timer2.start();
    if( n < 0 )
        rl::erode(binaryRLE, dstRLE, element, true);
    else
        rl::dilate(binaryRLE, dstRLE, element);
    timer2.stop();
    double rl_duration = timer2.time();

    PaintRLEToImage(dstRLE, rlePainted, (unsigned char)255);
    cout << "ratio erode/dilate duration: " << rl_duration / imgproc_duration <<
        " (run-length: " << rl_duration << ", pixelwise: " << imgproc_duration << " )" << std::endl;

    if (!AreImagesIdentical(dst, rlePainted))
    {
        cout << "error result image are not identical" << endl;
    }

    imshow("Erode/Dilate", rlePainted);
}

static void OnChangeThreshold(int, void*)
{
  threshold(src, binaryImage, (double) nThreshold, 255.0, THRESH_BINARY );
  rl::threshold(src, binaryRLE, (double) nThreshold, THRESH_BINARY);
  imshow("Threshold", binaryImage);
}


int main( int argc, char** argv )
{
    cv::CommandLineParser parser(argc, argv, "{help h||}{ @image | ../data/aloeL.jpg | }{u| |}");
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    std::string filename = parser.get<std::string>("@image");

    cv::Mat srcIn;
    if( (srcIn = imread(filename,IMREAD_GRAYSCALE)).empty() )
    {
        help();
        return -1;
    }
    int nScale = 1;
    if (parser.has("u"))
    {
        int theScale = parser.get<int>("u");
        if (theScale > 1)
            nScale = theScale;
    }

    if (nScale == 1)
        src = srcIn;
    else
        cv::resize(srcIn, src, cv::Size(srcIn.rows * nScale, srcIn.cols * nScale));

    cout << "scale factor read " << nScale << endl;

    print_introduction();

    //create windows for output images
    namedWindow("Open/Close",1);
    namedWindow("Erode/Dilate",1);
    namedWindow("Threshold",1);

    open_close_pos = erode_dilate_pos = max_size - 10;
    createTrackbar("size s.e.", "Open/Close",&open_close_pos,max_size*2+1,OpenClose);
    createTrackbar("size s.e.", "Erode/Dilate",&erode_dilate_pos,max_size*2+1,ErodeDilate);
    createTrackbar("threshold", "Threshold",&nThreshold,255, OnChangeThreshold);
    OnChangeThreshold(0, 0);
    rlePainted.create(cv::Size(src.cols, src.rows), CV_8UC1);

    for(;;)
    {
        OpenClose(open_close_pos, 0);
        ErodeDilate(erode_dilate_pos, 0);
        char c = (char)waitKey(0);

        if( c == 27 )
            break;
        if( c == 'e' )
            element_shape = MORPH_ELLIPSE;
        else if( c == 'r' )
            element_shape = MORPH_RECT;
        else if( c == 'c' )
            element_shape = MORPH_CROSS;
        else if( c == ' ' )
            element_shape = (element_shape + 1) % 3;
    }

    return 0;
}