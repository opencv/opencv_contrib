#include <iostream>

#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudaimgproc.hpp"

using namespace cv;
using namespace std;
using namespace cv::cuda;


void colorLabels(const Mat1i& labels, Mat3b& colors) {
    colors.create(labels.size());
    for (int r = 0; r < labels.rows; ++r) {
        int const* labels_row = labels.ptr<int>(r);
        Vec3b* colors_row = colors.ptr<Vec3b>(r);
        for (int c = 0; c < labels.cols; ++c) {
            colors_row[c] = Vec3b(labels_row[c] * 131 % 255, labels_row[c] * 241 % 255, labels_row[c] * 251 % 255);
        }
    }
}


int main(int argc, const char** argv)
{
    CommandLineParser parser(argc, argv, "{@image|stuff.jpg|image for converting to a grayscale}");
    parser.about("This program finds connected components in a binary image and assign each of them a different color.\n"
        "The connected components labeling is performed in GPU.\n");
    parser.printMessage();

    String inputImage = parser.get<string>(0);
    Mat1b img = imread(samples::findFile(inputImage), IMREAD_GRAYSCALE);
    Mat1i labels;

    if (img.empty())
    {
        cout << "Could not read input image file: " << inputImage << endl;
        return EXIT_FAILURE;
    }


    GpuMat d_img, d_labels;
    d_img.upload(img);

    cuda::connectedComponents(d_img, d_labels, 8, CV_32S);

    d_labels.download(labels);

    Mat3b colors;
    colorLabels(labels, colors);

    imshow("Labels", colors);
    waitKey(0);

    return EXIT_SUCCESS;
}
