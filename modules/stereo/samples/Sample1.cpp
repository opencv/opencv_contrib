#include <iostream>
#include "opencv2/stereo.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace stereo;
using namespace std;

//in this example we will load a sequence of images from a file process them and display the result on the screen
//the descriptor used is the modified_census transform

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
    sbm->setBinaryKernelType(CV_MODIFIED_CENSUS_TRANSFORM);//binary descriptor kernel
    sbm->setAgregationWindowSize(9);
    sbm->setSpekleRemovalTechnique(CV_SPECKLE_REMOVAL_AVG_ALGORITHM);//speckle removal algorithm
    sbm->setUsePrefilter(false);//prefilter or not the images prior to making the transformations

    for(int i = 0 ; i < 200; i++)
    {
        string path = "D:\\WorkingSec";
        string left = "l.bmp";
        string right = ".bmp";

        std::string s;
        std::stringstream out;
        out << i;
        s = out.str();

        string finLeft = path + "\\rezult" + s + left;
        string finRigth = path + "\\rezult" + s + right;

        image1 = imread(finLeft, CV_8UC1);
        image2 = imread(finRigth, CV_8UC1);
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
        waitKey(1);
    }

    waitKey(0);
    return 0;
}
