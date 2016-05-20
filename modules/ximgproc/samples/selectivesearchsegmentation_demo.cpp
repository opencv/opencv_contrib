/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.
                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)
Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.
This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/


#include "opencv2/ximgproc/segmentation.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <ctime>

using namespace cv;
using namespace cv::ximgproc::segmentation;

static void help() {
    std::cout << std::endl <<
    "A program demonstrating the use and capabilities of a particular image segmentation algorithm described" << std::endl <<
    " in Jasper R. R. Uijlings, Koen E. A. van de Sande, Theo Gevers, Arnold W. M. Smeulders: " << std::endl <<
    "                       \"Selective Search for Object Recognition\"" << std::endl <<
    "International Journal of Computer Vision, Volume 104 (2), page 154-171, 2013" << std::endl << std::endl <<
    "Usage:" << std::endl <<
    "./selectivesearchsegmentation_demo input_image (single|fast|quality)" << std::endl <<
    "Use a to display less rects, d to display more rects, q to quit" << std::endl;
}


int main(int argc, char** argv) {

    if (argc < 3) {
        help();
        return -1;
    }

    setUseOptimized(true);
    setNumThreads(8);

    std::srand((int)std::time(0));

    Mat img = imread(argv[1]);

    Ptr<SelectiveSearchSegmentation> gs = createSelectiveSearchSegmentation();
    gs->setBaseImage(img);

    if (argv[2][0] == 's') {
        gs->switchToSingleStrategy();
    } else if (argv[2][0] == 'f') {
        gs->switchToSelectiveSearchFast();
    } else if (argv[2][0] == 'q') {
        gs->switchToSelectiveSearchQuality();
    } else {
        help();
        return -2;
    }

    std::vector<Rect> rects;
    gs->process(rects);

    int nb_rects = 10;

    char c = (char)waitKey();

    while(c != 'q') {

        Mat wimg = img.clone();

        int i = 0;

        for(std::vector<Rect>::iterator it = rects.begin(); it != rects.end(); ++it) {
            if (i++ < nb_rects) {
                rectangle(wimg, *it, Scalar(0, 0, 255));
            }
        }

        imshow("Output", wimg);
        c = (char)waitKey();

        if (c == 'd') {
            nb_rects += 10;
        }

        if (c == 'a' && nb_rects > 10) {
            nb_rects -= 10;
        }
    }

    return 0;
}
