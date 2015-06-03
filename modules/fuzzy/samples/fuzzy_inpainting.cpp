/* Sample - Inpainting
 * Target is to apply inpainting using F-transform
 * on the image "input.png". The image is damaged
 * by various types of corruption:
 *
 * input1 = image & mask1
 * input2 = image & mask2
 * input3 = image & mask3
 *
 * Three algorithms "ft::ONE_STEP", "ft::MULTI_STEP"
 * and "ft::ITERATIVE" are demonstrated on the
 * appropriate type of damage.
 *
 * ft::ONE_STEP
 * "output1_inpaint.png": input1, mask1
 *
 * ft::MULTI_STEP
 * "output2_inpaint.png": input2, mask2
 * "output3_inpaint.png": input3, mask3
 *
 * ft::ITERATIVE
 * "output4_inpaint.png": input3, mask3
 *
 * Linear kernel with radius 2 is used for all
 * samples.
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

    // Various masks
    Mat mask1 = imread("mask1.png");
    Mat mask2 = imread("mask2.png");
    Mat mask3 = imread("mask3.png");

    // Apply the damage
    Mat input1, input2, input3;

    I.copyTo(input1, mask1);
    I.copyTo(input2, mask2);
    I.copyTo(input3, mask3);

    // Inpaint with various algorithm
    Mat output1, output2, output3, output4;

    ft::inpaint(input1, mask1, output1, 2, ft::LINEAR, ft::ONE_STEP);
    ft::inpaint(input2, mask2, output2, 2, ft::LINEAR, ft::MULTI_STEP);
    ft::inpaint(input3, mask3, output3, 2, ft::LINEAR, ft::MULTI_STEP);
    ft::inpaint(input3, mask3, output4, 2, ft::LINEAR, ft::ITERATIVE);

    // Save output
    imwrite("output1_inpaint.png", output1);
    imwrite("output2_inpaint.png", output2);
    imwrite("output3_inpaint.png", output3);
    imwrite("output4_inpaint.png", output4);

    // Save damaged input for comparison
    imwrite("input1.png", input1);
    imwrite("input2.png", input2);
    imwrite("input3.png", input3);

    return 0;
}
