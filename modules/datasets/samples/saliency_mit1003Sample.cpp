// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#include <opencv2/highgui.hpp>
#include "opencv2/datasets/saliency_mit1003.hpp"
#include <vector>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::datasets;

int main(int argc, char** argv)
{
    if (argc < 2) return 0;

    Ptr<SALIENCY_mit1003> datasetConnector = SALIENCY_mit1003::create();
    datasetConnector->load(argv[1]);
    vector<vector<Mat> > dataset(datasetConnector->getDataset()); //dataset[0] is original img, dataset[1] is fixMap, dataset[2] is fixPts
    //You can use mit1003 dataset do what ever you want
    for ( unsigned i = 0; i < dataset[0].size(); i++)
    {
        imshow("img", dataset[0][i]);
        waitKey(0);
    }
    return 0;
}