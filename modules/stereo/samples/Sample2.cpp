#include <iostream>
#include "opencv2/stereo.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace stereo;
using namespace std;


int main(int, char**)
{
    //begin the program
    cout << " Running Main function \n";
    //declare 2 images
    Mat image1, image2;

    //  -- 1. Call the constructor for StereoBinaryBM
    int ndisparities = 32;   /**< Range of disparity */
    int kernelSize = 9; /**< Size of the block window. Must be odd */

    Ptr<StereoBinaryBM> sbm = StereoBinaryBM::create(ndisparities, kernelSize);

    // -- 2. Set parameters
    sbm->setPreFilterCap(31);
    sbm->setMinDisparity(0);
    sbm->setTextureThreshold(10);
    sbm->setUniquenessRatio(0);
    sbm->setSpeckleWindowSize(400);//speckle size
    sbm->setSpeckleRange(200);
    sbm->setDisp12MaxDiff(0);
    sbm->setScalleFactor(4);//the scalling factor
    sbm->setBinaryKernelType(CV_MEAN_VARIATION);//binary descriptor kernel
    sbm->setAgregationWindowSize(9);
    sbm->setSpekleRemovalTechnique(CV_SPECKLE_REMOVAL_AVG_ALGORITHM);//speckle removal algorithm
    sbm->setUsePrefilter(false);//prefilter or not the images prior to making the transformations

    //load 2 images from disc
    image1 = imread("D:\\rezult0l.bmp", CV_8UC1);
    image2 = imread("D:\\rezult0.bmp", CV_8UC1);
    //set a certain region of interest
    Rect region_of_interest = Rect(0, 20, image1.cols, (image1.rows - 20 - 110));

    Mat imgLeft = image1(region_of_interest);
    Mat imgRight = image2(region_of_interest);

    Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);
    if (imgLeft.empty() || imgRight.empty())
    {
        std::cout << " --(!) Error reading images \n" ; return -1;
    }
    ////-- 3. Calculate the disparity image
    sbm->compute(imgLeft, imgRight, imgDisparity8U);

    imshow("RealImage", image1);
    imshow("Disparity", imgDisparity8U);

    waitKey(0);
    return 0;
}
