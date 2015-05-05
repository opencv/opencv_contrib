#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/fuzzy.hpp"

using namespace std;
using namespace cv;

int main(void)
{
    // Input image
    Mat I = imread("input.png");

    // Kernel cretion
    Mat kernel1, kernel2;

    ft::createKernel(ft::LINEAR, 3, kernel1, 3);
    ft::createKernel(ft::LINEAR, 100, kernel2, 3);

    // Filtering
    Mat output1, output2;

    ft::filter(I, kernel1, output1);
    ft::filter(I, kernel2, output2);

    // Save output

    imwrite("output1_filter.png", output1);
    imwrite("output2_filter.png", output2);

    return 0;
}
