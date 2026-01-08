#include "opencv2/optflow.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <time.h>
#include <stdio.h>
#include <ctype.h>

using namespace cv;
using namespace std;
using namespace cv::motempl;

static void help(void)
{
    printf(
        "\nThis program demonstrated the use of motion templates -- basically using the gradients\n"
        "of thresholded layers of decaying frame differencing. New movements are stamped on top with floating system\n"
        "time code and motions too old are thresholded away. This is the 'motion history file'. The program reads from the camera of your choice or from\n"
        "a file. Gradients of motion history are used to detect direction of motion etc\n"
        "Usage :\n"
        "./motempl [camera number 0-n or file name, default is camera 0]\n"
        );
}
// various tracking parameters (in seconds)
const double MHI_DURATION = 5;
const double MAX_TIME_DELTA = 0.5;
const double MIN_TIME_DELTA = 0.05;
// number of cyclic frame buffer used for motion detection
// (should, probably, depend on FPS)

// ring image buffer
vector<Mat> buf;
int last = 0;

// temporary images
Mat mhi, orient, mask, segmask, zplane;
vector<Rect> regions;

// parameters:
//  img - input video frame
//  dst - resultant motion picture
//  args - optional parameters
static void  update_mhi(const Mat& img, Mat& dst, int diff_threshold)
{
    double timestamp = (double)clock() / CLOCKS_PER_SEC; // get current time in seconds
    Size size = img.size();
    int i, idx1 = last;
    Rect comp_rect;
    double count;
    double angle;
    Point center;
    double magnitude;
    Scalar color;

    // allocate images at the beginning or
    // reallocate them if the frame size is changed
    if (mhi.size() != size)
    {
        mhi = Mat::zeros(size, CV_32F);
        zplane = Mat::zeros(size, CV_8U);

        buf[0] = Mat::zeros(size, CV_8U);
        buf[1] = Mat::zeros(size, CV_8U);
    }

    cvtColor(img, buf[last], COLOR_BGR2GRAY); // convert frame to grayscale

    int idx2 = (last + 1) % 2; // index of (last - (N-1))th frame
    last = idx2;

    Mat silh = buf[idx2];
    absdiff(buf[idx1], buf[idx2], silh); // get difference between frames

    threshold(silh, silh, diff_threshold, 1, THRESH_BINARY); // and threshold it
    updateMotionHistory(silh, mhi, timestamp, MHI_DURATION); // update MHI

    // convert MHI to blue 8u image
    mhi.convertTo(mask, CV_8U, 255. / MHI_DURATION, (MHI_DURATION - timestamp)*255. / MHI_DURATION);

    Mat planes[] = { mask, zplane, zplane };
    merge(planes, 3, dst);

    // calculate motion gradient orientation and valid orientation mask
    calcMotionGradient(mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3);

    // segment motion: get sequence of motion components
    // segmask is marked motion components map. It is not used further
    regions.clear();
    segmentMotion(mhi, segmask, regions, timestamp, MAX_TIME_DELTA);

    // iterate through the motion components,
    // One more iteration (i == -1) corresponds to the whole image (global motion)
    for (i = -1; i < (int)regions.size(); i++) {

        if (i < 0) { // case of the whole image
            comp_rect = Rect(0, 0, size.width, size.height);
            color = Scalar(255, 255, 255);
            magnitude = 100;
        }
        else { // i-th motion component
            comp_rect = regions[i];
            if (comp_rect.width + comp_rect.height < 100) // reject very small components
                continue;
            color = Scalar(0, 0, 255);
            magnitude = 30;
        }

        // select component ROI
        Mat silh_roi = silh(comp_rect);
        Mat mhi_roi = mhi(comp_rect);
        Mat orient_roi = orient(comp_rect);
        Mat mask_roi = mask(comp_rect);

        // calculate orientation
        angle = calcGlobalOrientation(orient_roi, mask_roi, mhi_roi, timestamp, MHI_DURATION);
        angle = 360.0 - angle;  // adjust for images with top-left origin

        count = norm(silh_roi, NORM_L1);; // calculate number of points within silhouette ROI

        // check for the case of little motion
        if (count < comp_rect.width*comp_rect.height * 0.05)
            continue;

        // draw a clock with arrow indicating the direction
        center = Point((comp_rect.x + comp_rect.width / 2),
            (comp_rect.y + comp_rect.height / 2));

        circle(img, center, cvRound(magnitude*1.2), color, 3, 16, 0);
        line(img, center, Point(cvRound(center.x + magnitude*cos(angle*CV_PI / 180)),
            cvRound(center.y - magnitude*sin(angle*CV_PI / 180))), color, 3, 16, 0);
    }
}


int main(int argc, char** argv)
{
    VideoCapture cap;

    help();

    if (argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
        cap.open(argc == 2 ? argv[1][0] - '0' : 0);
    else if (argc == 2)
        cap.open(argv[1]);

    if (!cap.isOpened())
    {
        printf("Could not initialize video capture\n");
        return 0;
    }
    buf.resize(2);
    Mat image, motion;
    for (;;)
    {
        cap >> image;
        if (image.empty())
            break;

        update_mhi(image, motion, 30);
        imshow("Image", image);
        imshow("Motion", motion);

        if (waitKey(10) >= 0)
            break;
    }

    return 0;
}
