#include <opencv2/xobjdetect.hpp>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace std;
using namespace cv;
using namespace cv::xobjdetect;

int main(int argc, char **argv)
{
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " train <model_filename> <pos_path> <neg_path>" << endl;
        cerr << "       " << argv[0] << " detect <model_filename> <img_filename> <out_filename>" << endl;
        return 0;
    }

    string mode = argv[1];
    WBDetector detector(argv[2]);
    if (mode == "train") {
        detector.train(argv[3], argv[4]);
    } else if (mode == "detect") {
        cerr << "detect" << endl;
        vector<Rect> bboxes;
        vector<double> confidences;
        Mat img = imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE);
        detector.detect(img, bboxes, confidences);
        for (size_t i = 0; i < bboxes.size(); ++i) {
            rectangle(img, bboxes[i], Scalar(255, 0, 0));
        }
        imwrite(argv[4], img);
    }
}
