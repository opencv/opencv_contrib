#include <string>
using std::string;

#include <vector>
using std::vector;

#include <iostream>
using std::cerr;
using std::endl;

#include <opencv2/core.hpp>
using cv::Rect;
using cv::Size;
using cv::Mat;
using cv::Mat_;
using cv::Vec3b;

#include <opencv2/highgui.hpp>
using cv::imread;
using cv::imwrite;

#include <opencv2/core/utility.hpp>
using cv::CommandLineParser;
using cv::FileStorage;

#include <opencv2/xobjdetect.hpp>
using cv::xobjdetect::ICFDetector;

static Mat visualize(const Mat &image, const vector<Rect> &objects)
{
    CV_Assert(image.type() == CV_8UC3);
    Mat_<Vec3b> img = image.clone();
    for( size_t j = 0; j < objects.size(); ++j )
    {
        Rect obj = objects[j];
        int x = obj.x;
        int y = obj.y;
        int width = obj.width;
        int height = obj.height;
        for( int i = y; i <= y + height; ++i ) {
            img(i, x) = Vec3b(255, 0, 0);
            img(i, x + width) = Vec3b(255, 0, 0);
        }
        for( int i = x; i <= x + width; ++i) {
            img(y, i) = Vec3b(255, 0, 0);
            img(y + height, i) = Vec3b(255, 0, 0);
        }
    }
    return img;
}

int main(int argc, char *argv[])
{
    const string keys =
        "{help           |           | print this message}"
        "{model_filename | model.xml | filename for reading model}"
        "{image_path     |  test.png | path to image for detection}"
        "{out_image_path |   out.png | path to image for output}"
        "{threshold      |       0.0 | threshold for cascade}"
        ;

    CommandLineParser parser(argc, argv, keys);
    parser.about("FCW detection");

    if( parser.has("help") || argc == 1)
    {
        parser.printMessage();
        return 0;
    }

    string model_filename = parser.get<string>("model_filename");
    string image_path = parser.get<string>("image_path");
    string out_image_path = parser.get<string>("out_image_path");
    float threshold = parser.get<float>("threshold");


    if( !parser.check() )
    {
        parser.printErrors();
        return 1;
    }

    ICFDetector detector;
    FileStorage fs(model_filename, FileStorage::READ);
    detector.read(fs["icfdetector"]);
    fs.release();
    vector<Rect> objects;
    Mat img = imread(image_path);
    detector.detect(img, objects, 1.1f, Size(40, 40),
        Size(300, 300), threshold);
    imwrite(out_image_path, visualize(img, objects));
}
