// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/saliency.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace saliency;


int main(int argc, char* argv[])
{
    const char *keys =
            "{ help h usage ? |     | show this message }"
            "{ start_frame    |0    | start frame index }"
            "{ length         |12   | # of frames video contain   }"
            "{ default        |1    | use default deep net(AlexNet) and default weights }"
            "{ video_name     |skiing| the name of video in UCSD background subtraction }"
            "{ img_folder_path|JPEGS| path to folder with frames }"
            "{ res_level      |  3  | resolution level of output saliency map. Suggested Range [0, 4]. The higher the level is, the fast the processing is, the lower the resolution is }";

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    vector<Mat> img_sq;
    DiscriminantSaliency t;
    if ( parser.get<bool>( "default" ) )
    {
        t = DiscriminantSaliency();
    }
    else
    {
        t = DiscriminantSaliency(parser.get<int>( "res_level" ));
    }
    for ( unsigned i = 1; i < parser.get<unsigned>( "length" ); i++ )
    {
        char index[256] = {0};
        sprintf(index, "%d", i + parser.get<int>( "start_frame" ));
        Mat temp = imread(parser.get<string>("img_folder_path") + "/" + parser.get<string>("video_name") + "/frame_" + index + ".jpg", 0);
        //Mat temp = imread(string("JPEGS/traffic/frame_") + index + ".jpg", 0);
        //resize(temp, temp, Size(127, 127));
        img_sq.push_back(temp);
    }
    vector<Mat> saliency_sq;
    t.computeSaliency(img_sq, saliency_sq);
    for ( unsigned i = 0; i < saliency_sq.size(); i++ )
    {
       resize(saliency_sq[i], saliency_sq[i], Size(1024, 768));
       t.saliencyMapVisualize(saliency_sq[i]);
    }
    return 0;
} //main
