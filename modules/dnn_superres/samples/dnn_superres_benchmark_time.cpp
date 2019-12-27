// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <iostream>

#include <opencv2/dnn_superres.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dnn_superres;

static void showBenchmark(vector<Mat> images, string title, Size imageSize,
                          const vector<String> imageTitles,
                          const vector<double> perfValues)
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

        ss << perfValues[i];
        putText(tmp,
                ss.str(),
                Point(5, fontStart),
                fontFace,
                fontScale,
                fontColor,
                1,
                16);

        tmp.copyTo(fullImage(ROI));
    }

    namedWindow(title, 1);
    imshow(title, fullImage);
    waitKey();
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
    Mat img_new;

    //Read and set the dnn model
    sr.readModel(model);
    sr.setModel(algorithm, scale);

    double elapsed = 0.0;
    vector<double> perf;

    TickMeter tm;

    //DL MODEL
    tm.start();
    sr.upsample(img_downscaled, img_new);
    tm.stop();
    elapsed = tm.getTimeSec() / tm.getCounter();
    perf.push_back(elapsed);

    cout << sr.getAlgorithm() << " : " << elapsed << endl;

    //BICUBIC
    Mat bicubic;
    tm.start();
    resize(img_downscaled, bicubic, Size(), scale, scale, INTER_CUBIC);
    tm.stop();
    elapsed = tm.getTimeSec() / tm.getCounter();
    perf.push_back(elapsed);

    cout << "Bicubic" << " : " << elapsed << endl;

    //NEAREST NEIGHBOR
    Mat nearest;
    tm.start();
    resize(img_downscaled, nearest, Size(), scale, scale, INTER_NEAREST);
    tm.stop();
    elapsed = tm.getTimeSec() / tm.getCounter();
    perf.push_back(elapsed);

    cout << "Nearest" << " : " << elapsed << endl;

    //LANCZOS
    Mat lanczos;
    tm.start();
    resize(img_downscaled, lanczos, Size(), scale, scale, INTER_LANCZOS4);
    tm.stop();
    elapsed = tm.getTimeSec() / tm.getCounter();
    perf.push_back(elapsed);

    cout << "Lanczos" << " : " << elapsed << endl;

    vector <Mat> imgs{img_new, bicubic, nearest, lanczos};
    vector <String> titles{sr.getAlgorithm(), "Bicubic", "Nearest neighbor", "Lanczos"};
    showBenchmark(imgs, "Time benchmark", Size(bicubic.cols, bicubic.rows), titles, perf);

    waitKey(0);

    return 0;
}