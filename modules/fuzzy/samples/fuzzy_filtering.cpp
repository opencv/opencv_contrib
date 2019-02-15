/* Sample - Filtering
 * Target is to apply filtering using F-transform
 * on the image "input.png". Two different kernels
 * are used, where bigger radius (100 in this case)
 * means higher level of blurriness.
 *
 * Image "output1_filter.png" is created from "input.png"
 * using "kernel1" with radius 3.
 *
 * Image "output2_filter.png" is created from "input.png"
 * using "kernel2" with radius 100.
 *
 * Both kernels are created from linear function, using
 * linear interpolation (parameter ft:LINEAR).
 */

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/fuzzy.hpp"

using namespace std;
using namespace cv;

int main(void)
{
    // Input image
    Mat I = imread("input.png");

    // Kernel creation
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
