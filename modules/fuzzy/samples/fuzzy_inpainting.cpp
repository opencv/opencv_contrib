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
