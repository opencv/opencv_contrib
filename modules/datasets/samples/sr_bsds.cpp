// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/datasets/sr_bsds.hpp"

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
            "{ path p         |true| path to dataset (images, iids_train.txt, iids_test.txt) }";
    CommandLineParser parser(argc, argv, keys);
    string path(parser.get<string>("path"));
    if (parser.has("help") || path=="true")
    {
        parser.printMessage();
        return -1;
    }

    Ptr<SR_bsds> dataset = SR_bsds::create();
    dataset->load(path);

    // ***************
    // Dataset train & test contain names of appropriate images.
    // For example, let's output full path & name for first train and test images.
    // And dataset sizes.
    printf("train size: %u\n", (unsigned int)dataset->getTrain().size());
    printf("test size: %u\n", (unsigned int)dataset->getTest().size());

    SR_bsdsObj *example1 = static_cast<SR_bsdsObj *>(dataset->getTrain()[0].get());
    string fullPath(path + "images/train/" + example1->imageName + ".jpg");
    printf("first train image: %s\n", fullPath.c_str());

    SR_bsdsObj *example2 = static_cast<SR_bsdsObj *>(dataset->getTest()[0].get());
    fullPath = path + "images/test/" + example2->imageName + ".jpg";
    printf("first test image: %s\n", fullPath.c_str());

    return 0;
}