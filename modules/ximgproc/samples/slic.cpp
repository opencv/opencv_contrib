#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>

#include <opencv2/ximgproc.hpp>

#include <ctype.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace cv::ximgproc;
using namespace std;

static const char* window_name = "SLIC Superpixels";

static const char* keys =
    "{h help      | | help menu}"
    "{c camera    |0| camera id}"
    "{i image     | | image file}"
    "{a algorithm |1| SLIC(0),SLICO(1)}"
    ;

int main(int argc, char** argv)
{
    CommandLineParser cmd(argc,argv,keys);
    if (cmd.has("help")) {
        cmd.about("This program demonstrates SLIC superpixels using OpenCV class SuperpixelSLIC.\n"
            "If no image file is supplied, try to open a webcam.\n"
            "Use [space] to toggle output mode, ['q' or 'Q' or 'esc'] to exit.\n");
        cmd.printMessage();
        return 0;
    }
    int capture = cmd.get<int>("camera");
    String img_file = cmd.get<String>("image");
    int algorithm = cmd.get<int>("algorithm");
    int region_size = 50;
    int ruler = 30;
    int min_element_size = 50;
    int num_iterations = 3;
    bool use_video_capture = img_file.empty();

    VideoCapture cap;
    Mat input_image;

    if( use_video_capture )
    {
        if( !cap.open(capture) )
        {
            cout << "Could not initialize capturing..."<<capture<<"\n";
            return -1;
        }
    }
    else
    {
        input_image = imread(img_file);
        if( input_image.empty() )
        {
            cout << "Could not open image..."<<img_file<<"\n";
            return -1;
        }
    }

    namedWindow(window_name, 0);
    createTrackbar("Algorithm", window_name, &algorithm, 1, 0);
    createTrackbar("Region size", window_name, &region_size, 200, 0);
    createTrackbar("Ruler", window_name, &ruler, 100, 0);
    createTrackbar("Connectivity", window_name, &min_element_size, 100, 0);
    createTrackbar("Iterations", window_name, &num_iterations, 12, 0);

    Mat result, mask;
    int display_mode = 0;

    for (;;)
    {
        Mat frame;
        if( use_video_capture )
            cap >> frame;
        else
            input_image.copyTo(frame);

        if( frame.empty() )
            break;

        result = frame;
        Mat converted;
        cvtColor(frame, converted, COLOR_BGR2HSV);

        double t = (double) getTickCount();

        Ptr<SuperpixelSLIC> slic = createSuperpixelSLIC(converted,algorithm+SLIC,region_size,float(ruler));
        slic->iterate(num_iterations);
        if (min_element_size>0)
            slic->enforceLabelConnectivity(min_element_size);

        t = ((double) getTickCount() - t) / getTickFrequency();
        cout << "SLIC" << (algorithm?'O':' ')
             << " segmentation took " << (int) (t * 1000)
             << " ms with " << slic->getNumberOfSuperpixels() << " superpixels" << endl;

        // get the contours for displaying
        slic->getLabelContourMask(mask, true);
        result.setTo(Scalar(0, 0, 255), mask);

        // display output
        switch (display_mode)
        {
        case 0: //superpixel contours
            imshow(window_name, result);
            break;
        case 1: //mask
            imshow(window_name, mask);
            break;
        case 2: //labels array
        {
            // use the last x bit to determine the color. Note that this does not
            // guarantee that 2 neighboring superpixels have different colors.
            // retrieve the segmentation result
            Mat labels;
            slic->getLabels(labels);
            const int num_label_bits = 2;
            labels &= (1 << num_label_bits) - 1;
            labels *= 1 << (16 - num_label_bits);
            imshow(window_name, labels);
            break;
        }
        }

        int c = waitKey(1) & 0xff;
        if( c == 'q' || c == 'Q' || c == 27 )
            break;
        else if( c == ' ' )
            display_mode = (display_mode + 1) % 3;
    }

    return 0;
}
