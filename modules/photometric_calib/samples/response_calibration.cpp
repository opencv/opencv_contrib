#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/photometric_calib.hpp"

using namespace std;
using namespace cv;

int main()
{
    // Please down load the sample dataset from:
    // https://www.dropbox.com/s/5x48uhc7k2bgjcj/GSoC2017_PhotometricCalib_Sample_Data.zip?dl=0
    // By unzipping the file, you would get a folder named /GSoC2017_PhotometricCalib_Sample_Data which contains 4 subfolders:
    // response_calib, response remover, vignette_calib, vignette_remover
    // in this sample, we will use the data in the folder response_calib and response_remover

    // Prefix for the data, e.g. /Users/Yelen/GSoC2017_PhotometricCalib_Sample
    string userPrefix = "/Users/Yelen/GSoC2017_PhotometricCalib_Sample_Data/";
    // The path for the images used for response calibration
    string imageFolderPath = userPrefix + "response_calib/images";
    // The yaml file which contains the timestamps and exposure times for each image used for camera response calibration
    string timePath = userPrefix + "response_calib/times.yaml";

    // Construct a photometric_calib::ResponseCalib object by giving path of image, path of time file and specify the format of images
    photometric_calib::ResponseCalib resCal(imageFolderPath, timePath, "jpg");

    // Debug mode will generate some temporary data
    bool debug = true;
    // Calibration of camera response function begins
    resCal.calib(debug);

    // The result and some intermediate data are stored in the folder ./photoCalibResult in which
    // pcalib.yaml is the camera response function file
    // Since we are using debug mode, we can visualize the response function:
    Mat invRes = imread("./photoCalibResult/G-10.png", IMREAD_UNCHANGED);
    // As shown as Fig.3 in the paper from J.Engel, et al. in the paper A Photometrically Calibrated Benchmark For Monocular Visual Odometry
    namedWindow( "Inverse Response Function", WINDOW_AUTOSIZE );
    imshow("Inverse Response Function", invRes);

    // To see the response-calibrated image, we can use GammaRemover
    Mat oriImg = imread(imageFolderPath + "/00480.jpg", IMREAD_UNCHANGED);
    photometric_calib::GammaRemover gammaRemover("./photoCalibResult/pcalib.yaml", oriImg.cols, oriImg.rows);
    Mat caliImg = gammaRemover.getUnGammaImageMat(oriImg);

    // Visualization
    namedWindow( "Original Image", WINDOW_AUTOSIZE );
    imshow("Original Image", oriImg);
    namedWindow( "Gamma Removed Image", WINDOW_AUTOSIZE );
    imshow("Gamma Removed Image", caliImg);
    waitKey(0);

    return 0;
}