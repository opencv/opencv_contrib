#include "icfdetector.hpp"

#include <iostream>

using std::vector;
using std::string;

#include <opencv2/core.hpp>
using cv::Rect;

void ICFDetector::train(const vector<string>& image_filenames,
                        const vector< vector<Rect> >& labelling,
                        ICFDetectorParams params)
{
    std::cout << "train" << std::endl;

}

bool ICFDetector::save(const string& filename)
{
    return true;
}
