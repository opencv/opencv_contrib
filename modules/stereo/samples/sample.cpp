#include "opencv2/stereo.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <string.h>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::stereo;

enum { STEREO_BINARY_BM, STEREO_BINARY_SGM };
static cv::CommandLineParser parse_argument_values(int argc, char **argv, string &left, string &right, int &kernel_size, int &number_of_disparities,
                                                   int &aggregation_window, int &P1, int &P2, float &scale, int &algo, int &binary_descriptor_type, int &success);
int main(int argc, char** argv)
{
    string left, right;
    int kernel_size = 0, number_of_disparities = 0, aggregation_window = 0, P1 = 0, P2 = 0;
    float scale = 4;
    int algo = STEREO_BINARY_BM;
    int binary_descriptor_type = 0;
    int success;
    // here we extract the values that were added as arguments
    // we also test to see if they are provided correcly
    cv::CommandLineParser parser =
        parse_argument_values(argc, argv, left, right,
        kernel_size,
        number_of_disparities,
        aggregation_window,
        P1, P2,
        scale,
        algo, binary_descriptor_type,success);
    if (!parser.check() || !success)
    {
        parser.printMessage();
        return 1;
    }
    // verify if the user inputs the correct number of parameters
    Mat image1, image2;
    // we read  a pair of images from the disk
    image1 = imread(left, CV_8UC1);
    image2 = imread(right, CV_8UC1);
    // verify if they are loaded correctly
    if (image1.empty() || image2.empty())
    {
        cout << " --(!) Error reading images \n";

        parser.printMessage();
        return 1;
    }
    // we display the parsed parameters
    const char *b[7] = { "CV_DENSE_CENSUS", "CV_SPARSE_CENSUS", "CV_CS_CENSUS", "CV_MODIFIED_CS_CENSUS",
        "CV_MODIFIED_CENSUS_TRANSFORM", "CV_MEAN_VARIATION", "CV_STAR_KERNEL" };
    cout << "Program Name: " << argv[0];
    cout << "\nPath to left image " << left << " \n" << "Path to right image " << right << "\n";
    cout << "\nkernel size " << kernel_size << "\n"
        << "numberOfDisparities " << number_of_disparities << "\n"
        << "aggregationWindow " << aggregation_window << "\n"
        << "scallingFactor " << scale << "\n" << "Descriptor name : " << b[binary_descriptor_type] << "\n";

    Mat imgDisparity16S2 = Mat(image1.rows, image1.cols, CV_16S);
    Mat imgDisparity8U2 = Mat(image1.rows, image1.cols, CV_8UC1);
    imshow("Original Left image", image1);

    if (algo == STEREO_BINARY_BM)
    {
        Ptr<StereoBinaryBM> sbm = StereoBinaryBM::create(number_of_disparities, kernel_size);
        // we set the corresponding parameters
        sbm->setPreFilterCap(31);
        sbm->setMinDisparity(0);
        sbm->setTextureThreshold(10);
        sbm->setUniquenessRatio(0);
        sbm->setSpeckleWindowSize(400); // speckle size
        sbm->setSpeckleRange(200);
        sbm->setDisp12MaxDiff(0);
        sbm->setScalleFactor((int)scale); // the scaling factor
        sbm->setBinaryKernelType(binary_descriptor_type); // binary descriptor kernel
        sbm->setAgregationWindowSize(aggregation_window);
        // the user can choose between the average speckle removal algorithm or
        // the classical version that was implemented in OpenCV
        sbm->setSpekleRemovalTechnique(CV_SPECKLE_REMOVAL_AVG_ALGORITHM);
        sbm->setUsePrefilter(false);
        //-- calculate the disparity image
        sbm->compute(image1, image2, imgDisparity8U2);
        imshow("Disparity", imgDisparity8U2);
    }
    else if (algo == STEREO_BINARY_SGM)
    {
        // we set the corresponding parameters
        Ptr<StereoBinarySGBM> sgbm = StereoBinarySGBM::create(0, number_of_disparities, kernel_size);
        // setting the penalties for sgbm
        sgbm->setP1(P1);
        sgbm->setP2(P2);
        sgbm->setMinDisparity(0);
        sgbm->setUniquenessRatio(5);
        sgbm->setSpeckleWindowSize(400);
        sgbm->setSpeckleRange(0);
        sgbm->setDisp12MaxDiff(1);
        sgbm->setBinaryKernelType(binary_descriptor_type);
        sgbm->setSpekleRemovalTechnique(CV_SPECKLE_REMOVAL_AVG_ALGORITHM);
        sgbm->setSubPixelInterpolationMethod(CV_SIMETRICV_INTERPOLATION);
        sgbm->compute(image1, image2, imgDisparity16S2);
        /*Alternative for scalling
        imgDisparity16S2.convertTo(imgDisparity8U2, CV_8UC1, scale);
        */
        double minVal; double maxVal;
        minMaxLoc(imgDisparity16S2, &minVal, &maxVal);
        imgDisparity16S2.convertTo(imgDisparity8U2, CV_8UC1, 255 / (maxVal - minVal));
        //show the disparity image
        imshow("Windowsgm", imgDisparity8U2);
    }
    waitKey(0);
    return 0;
}
static cv::CommandLineParser parse_argument_values(int argc, char **argv, string &left, string &right, int &kernel_size, int &number_of_disparities,
                                                   int &aggregation_window, int &P1, int &P2, float &scale, int &algo, int &binary_descriptor_type, int &success)
{
    static const char* keys =
        "{ @left                |         | }"
        "{ @right               |         | }"
        "{ k kernel_size        |    9    | }"
        "{ d disparity          |    128   | }"
        "{ w aggregation_window |    9     | }"
        "{ P1                   |   100    | }"
        "{ P2                   |   1000   | }"
        "{ b binary_descriptor  |     4    | Index of the descriptor type:\n 0 - CV_DENSE_CENSUS,\n 1 - CV_SPARSE_CENSUS,\n 2 - CV_CS_CENSUS,\n 3 - CV_MODIFIED_CS_CENSUS,\n 4 - CV_MODIFIED_CENSUS_TRANSFORM,\n 5 - CV_MEAN_VARIATION,\n 6 - CV_STAR_KERNEL}"
        "{ s scale              |    1.01593    | }"
        "{ a algorithm          | sgm     | }"
        ;
    cv::CommandLineParser parser( argc, argv, keys );

    left = parser.get<string>(0);
    right = parser.get<string>(1);
    kernel_size = parser.get<int>("kernel_size");
    number_of_disparities = parser.get<int>("disparity");
    aggregation_window = parser.get<int>("aggregation_window");
    P1 = parser.get<int>("P1");
    P2 = parser.get<int>("P2");
    binary_descriptor_type = parser.get<int>("binary_descriptor");
    scale = parser.get<float>("scale");
    algo = parser.get<string>("algorithm") == "sgm" ? STEREO_BINARY_SGM : STEREO_BINARY_BM;

    parser.about("\nDemo stereo matching converting L and R images into disparity images using BM and SGBM\n");

    success = 1;
    //TEST if the provided parameters are correct
    if(binary_descriptor_type == CV_DENSE_CENSUS && kernel_size > 5)
    {
        cout << "For the dense census transform the maximum kernel size should be 5\n";
        success = 0;
    }
    if((binary_descriptor_type == CV_MEAN_VARIATION || binary_descriptor_type == CV_MODIFIED_CENSUS_TRANSFORM || binary_descriptor_type == CV_STAR_KERNEL) && kernel_size != 9)
    {
        cout <<" For Mean variation and the modified census transform the kernel size should be equal to 9\n";
        success = 0;
    }
    if((binary_descriptor_type == CV_CS_CENSUS || binary_descriptor_type == CV_MODIFIED_CS_CENSUS) && kernel_size > 7)
    {
        cout << " The kernel size should be smaller or equal to 7 for the CS census and modified center symetric census\n";
        success = 0;
    }
    if(binary_descriptor_type == CV_SPARSE_CENSUS && kernel_size > 11)
    {
        cout << "The kernel size for the sparse census must be smaller or equal to 11\n";
        success = 0;
    }
    if(number_of_disparities < 10)
    {
        cout << "Number of disparities should be greater than 10\n";
        success = 0;
    }
    if(aggregation_window < 3)
    {
        cout << "Aggregation window should be > 3";
        success = 0;
    }
    if(scale < 1)
    {
        cout << "The scale should be a positive number \n";
        success = 0;
    }
    if(P1 != 0)
    {
        if(P2 / P1 < 2)
        {
            cout << "You should probably choose a greater P2 penalty\n";
            success = 0;
        }
    }
    else
    {
        cout << " Penalties should be greater than 0\n";
        success = 0;
    }
    return parser;
}
