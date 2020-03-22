// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <iostream>
#include <opencv2/opencv_modules.hpp>

#ifdef HAVE_OPENCV_QUALITY
#include <opencv2/dnn_superres.hpp>
#include <opencv2/quality.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dnn_superres;

static void showBenchmark(vector<Mat> images, string title, Size imageSize,
                          const vector<String> imageTitles,
                          const vector<double> psnrValues,
                          const vector<double> ssimValues)
{
    int fontFace = FONT_HERSHEY_COMPLEX_SMALL;
    int fontScale = 1;
    Scalar fontColor = Scalar(255, 255, 255);

    int len = static_cast<int>(images.size());

    int cols = 2, rows = 2;

    Mat fullImage = Mat::zeros(Size((cols * 10) + imageSize.width * cols, (rows * 10) + imageSize.height * rows),
                               images[0].type());

    stringstream ss;
    int h_ = -1;
    for (int i = 0; i < len; i++) {

        int fontStart = 15;
        int w_ = i % cols;
        if (i % cols == 0)
            h_++;

        Rect ROI((w_ * (10 + imageSize.width)), (h_ * (10 + imageSize.height)), imageSize.width, imageSize.height);
        Mat tmp;
        resize(images[i], tmp, Size(ROI.width, ROI.height));

        ss << imageTitles[i];
        putText(tmp,
                ss.str(),
                Point(5, fontStart),
                fontFace,
                fontScale,
                fontColor,
                1,
                16);

        ss.str("");
        fontStart += 20;

        ss << "PSNR: " << psnrValues[i];
        putText(tmp,
                ss.str(),
                Point(5, fontStart),
                fontFace,
                fontScale,
                fontColor,
                1,
                16);

        ss.str("");
        fontStart += 20;

        ss << "SSIM: " << ssimValues[i];
        putText(tmp,
                ss.str(),
                Point(5, fontStart),
                fontFace,
                fontScale,
                fontColor,
                1,
                16);

        ss.str("");
        fontStart += 20;

        tmp.copyTo(fullImage(ROI));
    }

    namedWindow(title, 1);
    imshow(title, fullImage);
    waitKey();
}

static Vec2d getQualityValues(Mat orig, Mat upsampled)
{
    double psnr = PSNR(upsampled, orig);
    Scalar q = quality::QualitySSIM::compute(upsampled, orig, noArray());
    double ssim = mean(Vec3d((q[0]), q[1], q[2]))[0];
    return Vec2d(psnr, ssim);
}

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

    Mat img = imread(path);
    if (img.empty()) {
        cerr << "Couldn't load image: " << img << "\n";
        return -2;
    }

    //Crop the image so the images will be aligned
    int width = img.cols - (img.cols % scale);
    int height = img.rows - (img.rows % scale);
    Mat cropped = img(Rect(0, 0, width, height));

    //Downscale the image for benchmarking
    Mat img_downscaled;
    resize(cropped, img_downscaled, Size(), 1.0 / scale, 1.0 / scale);

    //Make dnn super resolution instance
    DnnSuperResImpl sr;

    vector <Mat> allImages;
    Mat img_new;

    //Read and set the dnn model
    sr.readModel(model);
    sr.setModel(algorithm, scale);
    sr.upsample(img_downscaled, img_new);

    vector<double> psnrValues = vector<double>();
    vector<double> ssimValues = vector<double>();

    //DL MODEL
    Vec2f quality = getQualityValues(cropped, img_new);

    psnrValues.push_back(quality[0]);
    ssimValues.push_back(quality[1]);

    cout << sr.getAlgorithm() << ":" << endl;
    cout << "PSNR: " << quality[0] << " SSIM: " << quality[1] << endl;
    cout << "----------------------" << endl;

    //BICUBIC
    Mat bicubic;
    resize(img_downscaled, bicubic, Size(), scale, scale, INTER_CUBIC);
    quality = getQualityValues(cropped, bicubic);

    psnrValues.push_back(quality[0]);
    ssimValues.push_back(quality[1]);

    cout << "Bicubic " << endl;
    cout << "PSNR: " << quality[0] << " SSIM: " << quality[1] << endl;
    cout << "----------------------" << endl;

    //NEAREST NEIGHBOR
    Mat nearest;
    resize(img_downscaled, nearest, Size(), scale, scale, INTER_NEAREST);
    quality = getQualityValues(cropped, nearest);

    psnrValues.push_back(quality[0]);
    ssimValues.push_back(quality[1]);

    cout << "Nearest neighbor" << endl;
    cout << "PSNR: " << quality[0] << " SSIM: " << quality[1] << endl;
    cout << "----------------------" << endl;

    //LANCZOS
    Mat lanczos;
    resize(img_downscaled, lanczos, Size(), scale, scale, INTER_LANCZOS4);
    quality = getQualityValues(cropped, lanczos);

    psnrValues.push_back(quality[0]);
    ssimValues.push_back(quality[1]);

    cout << "Lanczos" << endl;
    cout << "PSNR: " << quality[0] << " SSIM: " << quality[1] << endl;
    cout << "-----------------------------------------------" << endl;

    vector <Mat> imgs{img_new, bicubic, nearest, lanczos};
    vector <String> titles{sr.getAlgorithm(), "Bicubic", "Nearest neighbor", "Lanczos"};
    showBenchmark(imgs, "Quality benchmark", Size(bicubic.cols, bicubic.rows), titles, psnrValues, ssimValues);

    waitKey(0);

    return 0;
}
#else
int main()
{
    std::cout << "This sample requires the OpenCV Quality module." << std::endl;
    return 0;
}
#endif
