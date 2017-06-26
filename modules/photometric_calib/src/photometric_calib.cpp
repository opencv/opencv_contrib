// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/photometric_calib.hpp"

namespace cv{ namespace photometric_calib{

using namespace std;

bool PhotometricCalibrator::validImgs(const std::vector <Mat> &inputImgs, const std::vector<double> &exposureTime)
{
    if(inputImgs.empty() || exposureTime.empty() || inputImgs.size() != exposureTime.size())
        return false;

    int width = 0, height = 0;
    for(size_t i = 0; i < inputImgs.size(); ++ i)
    {
        Mat img;
        img = inputImgs[i];
        if(img.type() != CV_8U)
        {
            cout<<"The type of the image should be CV_8U!"<<endl;
            return false;
        }
        if((width!=0 && width != img.cols) || img.cols==0)
        {
            cout<<"Width mismatch!"<<endl;
            return false;
        };
        if((height!=0 && height != img.rows) || img.rows==0)
        {
            cout<<"Height mismatch!"<<endl;
            return false;
        };
        width = img.cols;
        height = img.rows;
    }
    return true;
}

}} // namespace photometric_calib, cv