#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/intensity_transform.hpp"

#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::intensity_transform;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cerr << "Must input the path of the input image. Ex: intensity_transform image.jpg" << endl;
        return -1;
    }

    // Read input image
    Mat image = imread(argv[1]);

    // Apply intensity transformations
    Mat imgGamma, imgAutoscaled, imgLog, contrastStretch;
    gammaCorrection(image, imgGamma, (float)(0.4));
    autoscaling(image, imgAutoscaled);
    logTransform(image, imgLog);
    contrastStretching(image, contrastStretch, 70, 15, 120, 240);

    // Display intensity transformation results
    imshow("Original Image", image);
    imshow("Autoscale", imgAutoscaled);
    imshow("Gamma Correction", imgGamma);
    imshow("Log Transformation", imgLog);
    imshow("Contrast Stretching", contrastStretch);
    waitKey(0);

    return 0;
}