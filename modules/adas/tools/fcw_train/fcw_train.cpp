#include <cstdio>
#include <cstring>

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <fstream>
using std::ifstream;
using std::getline;

#include <sstream>
using std::stringstream;

#include <iostream>
using std::cerr;
using std::endl;

#include <opencv2/core.hpp>
using cv::Rect;
using cv::Size;
#include <opencv2/highgui.hpp>
using cv::imread;
#include <opencv2/core/utility.hpp>
using cv::CommandLineParser;
using cv::FileStorage;

#include <opencv2/xobjdetect.hpp>

using cv::xobjdetect::ICFDetectorParams;
using cv::xobjdetect::ICFDetector;
using cv::xobjdetect::WaldBoost;
using cv::xobjdetect::WaldBoostParams;
using cv::Mat;

static bool read_model_size(const char *str, int *rows, int *cols)
{
    int pos = 0;
    if( sscanf(str, "%dx%d%n", rows, cols, &pos) != 2 || str[pos] != '\0' ||
        *rows <= 0 || *cols <= 0)
    {
        return false;
    }
    return true;
}

int main(int argc, char *argv[])
{
    const string keys =
        "{help           |           | print this message}"
        "{pos_path       |       pos | path to training object samples}"
        "{bg_path        |        bg | path to background images}"
        "{bg_per_image   |         5 | number of windows to sample per bg image}"
        "{feature_count  |     10000 | number of features to generate}"
        "{weak_count     |       100 | number of weak classifiers in cascade}"
        "{model_size     |     40x40 | model size in pixels}"
        "{model_filename | model.xml | filename for saving model}"
        ;

    CommandLineParser parser(argc, argv, keys);
    parser.about("FCW trainer");

    if( parser.has("help") || argc == 1)
    {
        parser.printMessage();
        return 0;
    }

    string pos_path = parser.get<string>("pos_path");
    string bg_path = parser.get<string>("bg_path");
    string model_filename = parser.get<string>("model_filename");

    ICFDetectorParams params;
    params.feature_count = parser.get<int>("feature_count");
    params.weak_count = parser.get<int>("weak_count");
    params.bg_per_image = parser.get<int>("bg_per_image");

    string model_size = parser.get<string>("model_size");
    if( !read_model_size(model_size.c_str(), &params.model_n_rows,
        &params.model_n_cols) )
    {
        cerr << "Error reading model size from `" << model_size << "`" << endl;
        return 1;
    }

    if( params.feature_count <= 0 )
    {
        cerr << "feature_count must be positive number" << endl;
        return 1;
    }

    if( params.weak_count <= 0 )
    {
        cerr << "weak_count must be positive number" << endl;
        return 1;
    }

    if( params.bg_per_image <= 0 )
    {
        cerr << "bg_per_image must be positive number" << endl;
        return 1;
    }

    if( !parser.check() )
    {
        parser.printErrors();
        return 1;
    }

    ICFDetector detector;
    detector.train(pos_path, bg_path, params);
    FileStorage fs(model_filename, FileStorage::WRITE);
    fs << "icfdetector";
    detector.write(fs);
    fs.release();
}
