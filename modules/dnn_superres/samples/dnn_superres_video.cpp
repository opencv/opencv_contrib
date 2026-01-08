// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <iostream>

#include <opencv2/dnn_superres.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dnn_superres;

int main(int argc, char *argv[])
{
    // Check for valid command line arguments, print usage
    // if insufficient arguments were given.
    if (argc < 4) {
        cout << "usage:   Arg 1: input video path" << endl;
        cout << "\t Arg 2: output video path" << endl;
        cout << "\t Arg 3: algorithm | edsr, espcn, fsrcnn or lapsrn" << endl;
        cout << "\t Arg 4: scale     | 2, 3, 4 or 8 \n";
        cout << "\t Arg 5: path to model file \n";
        return -1;
    }

    string input_path = string(argv[1]);
    string output_path = string(argv[2]);
    string algorithm = string(argv[3]);
    int scale = atoi(argv[4]);
    string path = string(argv[5]);

    VideoCapture input_video(input_path);
    int ex = static_cast<int>(input_video.get(CAP_PROP_FOURCC));
    Size S = Size((int) input_video.get(CAP_PROP_FRAME_WIDTH) * scale,
                  (int) input_video.get(CAP_PROP_FRAME_HEIGHT) * scale);

    VideoWriter output_video;
    output_video.open(output_path, ex, input_video.get(CAP_PROP_FPS), S, true);

    if (!input_video.isOpened())
    {
        std::cerr << "Could not open the video." << std::endl;
        return -1;
    }

    DnnSuperResImpl sr;
    sr.readModel(path);
    sr.setModel(algorithm, scale);

    for(;;)
    {
        Mat frame, output_frame;
        input_video >> frame;

        if ( frame.empty() )
            break;

        sr.upsample(frame, output_frame);
        output_video << output_frame;

        namedWindow("Upsampled video", WINDOW_AUTOSIZE);
        imshow("Upsampled video", output_frame);

        namedWindow("Original video", WINDOW_AUTOSIZE);
        imshow("Original video", frame);

        char c=(char)waitKey(25);
        if(c==27)
            break;
    }

    input_video.release();
    output_video.release();

    return 0;
}