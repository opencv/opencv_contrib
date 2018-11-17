#include "opencv2/xobjdetect.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <cstdio>
using namespace std;
using namespace cv;
using namespace cv::xobjdetect;

int main(int argc, char **argv)
{
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " train <model_filename> <pos_path> <neg_path>" << endl;
        cerr << "       " << argv[0] << " detect <model_filename> <img_filename> <out_filename> <labelling_filename>" << endl;
        return 0;
    }

    string mode = argv[1];
    Ptr<WBDetector> detector = WBDetector::create();
    if (mode == "train") {
        assert(argc == 5);
        detector->train(argv[3], argv[4]);
        FileStorage fs(argv[2], FileStorage::WRITE);
        fs << "waldboost";
        detector->write(fs);
    } else if (mode == "detect") {
        assert(argc == 6);
        vector<Rect> bboxes;
        vector<double> confidences;
        Mat img = imread(argv[3], IMREAD_GRAYSCALE);
        FileStorage fs(argv[2], FileStorage::READ);
        detector->read(fs.getFirstTopLevelNode());
        detector->detect(img, bboxes, confidences);

        FILE *fhandle = fopen(argv[5], "a");
        for (size_t i = 0; i < bboxes.size(); ++i) {
            Rect o = bboxes[i];
            fprintf(fhandle, "%s;%u;%u;%u;%u;%lf\n",
                argv[3], o.x, o.y, o.width, o.height, confidences[i]);
        }
        for (size_t i = 0; i < bboxes.size(); ++i) {
            rectangle(img, bboxes[i], Scalar(255, 0, 0));
        }
        imwrite(argv[4], img);
    }
}
