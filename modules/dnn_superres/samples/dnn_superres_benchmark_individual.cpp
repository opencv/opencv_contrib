// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <iostream>

#include <opencv2/dnn_superres.hpp>
#include <opencv2/dnn_superres_quality.hpp>

using namespace std;
using namespace cv;
using namespace dnn_superres;

int main(int argc, char *argv[])
{
    // Check for valid command line arguments, print usage
    // if insufficient arguments were given.
    if (argc < 4) {
        cout << "usage:   Arg 1: image path  | Path to image" << endl;
        cout << "\t Arg 2: algorithm | edsr, espcn, fsrcnn or lapsrn" << endl;
        cout << "\t Arg 3: path to model file 2 \n";
        cout << "\t Arg 4: scale  | 2, 3, 4 or 8 \n";
        return -1;
    }

    string path = string(argv[1]);
    string algorithm = string(argv[2]);
    string model = string(argv[3]);
    int scale = atoi(argv[4]);

    Mat img = cv::imread(path);
    if (img.empty())
    {
        std::cerr << "Couldn't load image: " << img << "\n";
        return -2;
    }

    std::cout << "Image loaded." << std::endl;

    //Crop the image so the images will be aligned
    int width = img.cols - (img.cols % scale);
    int height = img.rows - (img.rows % scale);
    Mat cropped = img(Rect(0, 0, width, height));

    //Downscale the image for benchmarking
    Mat img_downscaled;
    cv::resize(cropped, img_downscaled, cv::Size(), 1.0/scale, 1.0/scale);

    //Make dnn super resolution instance
    DnnSuperResImpl sr;

    std::vector<Mat> allImages;
    Mat img_new;

    //Read and set the dnn model
    sr.readModel(model);
    sr.setModel(algorithm, scale);
    sr.upsample(img_downscaled, img_new);

    double ps = DnnSuperResQuality::psnr(img_new, cropped);
    double ssim = DnnSuperResQuality::ssim(img_new, cropped);

    if ( img_new.empty() )
    {
        std::cerr << "Upsampling failed. \n";
        return -3;
    }
    cout << "Upsampling succeeded. \n";

    std::cout << algorithm << " | PSNR: " << ps << " SSIM: " << ssim << std::endl;

    Mat bicubic;
    cv::resize(img_downscaled, bicubic, cv::Size(), scale, scale, cv::INTER_CUBIC );
    double ps_bicubic = DnnSuperResQuality::psnr(bicubic, cropped);
    double ssim_bicubic = DnnSuperResQuality::ssim(bicubic, cropped);
    std::cout << "Bicubic" << " | PSNR: " << ps_bicubic << " SSIM: " << ssim_bicubic << std::endl;

    Mat nearest;
    cv::resize(img_downscaled, nearest, cv::Size(), scale, scale, cv::INTER_NEAREST );
    double ps_nearest = DnnSuperResQuality::psnr(nearest, cropped);
    double ssim_nearest = DnnSuperResQuality::ssim(nearest, cropped);
    std::cout << "Nearest neighbor" << " | PSNR: " << ps_nearest << " SSIM: " << ssim_nearest << std::endl;

    Mat lanczos;
    cv::resize(img_downscaled, lanczos, cv::Size(), scale, scale, cv::INTER_LANCZOS4 );
    double ps_lanczos = DnnSuperResQuality::psnr(lanczos, cropped);
    double ssim_lanczos = DnnSuperResQuality::ssim(lanczos, cropped);
    std::cout << "Lanczos" << " | PSNR: " << ps_lanczos << " SSIM: " << ssim_lanczos << std::endl;

    allImages.push_back(img_new);
    allImages.push_back(bicubic);
    allImages.push_back(nearest);
    allImages.push_back(lanczos);

    std::vector<String> titles{algorithm, "bicubic", "nearest", "lanczos"};
    DnnSuperResQuality::showBenchmark(cropped, allImages, "Image", cv::Size(bicubic.cols, bicubic.rows), titles);

    cv::waitKey(0);

    return 0;
}