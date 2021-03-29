// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/datasets/sr_general100.hpp"

#include <opencv2/core.hpp>

#include <cstdio>

#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::datasets;

int main(int argc, char *argv[])
{
    const char *keys =
            "{ help h usage ? |    | show this message }"
            "{ path p         |true| path to dataset (General-100 dataset folder) }";
    CommandLineParser parser(argc, argv, keys);
    string path(parser.get<string>("path"));
    if (parser.has("help") || path=="true")
    {
        parser.printMessage();
        return -1;
    }

    Ptr<SR_general100> dataset = SR_general100::create();
    dataset->load(path);

    // ***************
    // Dataset contains all images.
    // For example, let's output dataset size; first image name; and second image full path.
    printf("dataset size: %u\n", (unsigned int)dataset->getTrain().size());

    SR_general100Obj *example = static_cast<SR_general100Obj *>(dataset->getTrain()[0].get());
    printf("first image name: %s\n", example->imageName.c_str());

    SR_general100Obj *example2 = static_cast<SR_general100Obj *>(dataset->getTrain()[1].get());
    string fullPath = path + "/" + example2->imageName.c_str();
    printf("second image full path: %s\n", fullPath.c_str());

    return 0;
}