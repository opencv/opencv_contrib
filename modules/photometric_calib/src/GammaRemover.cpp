// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/photometric_calib/GammaRemover.hpp"

namespace cv {
namespace photometric_calib {

GammaRemover::GammaRemover(const std::string &gammaPath, int w_, int h_)
{
    validGamma = false;
    w = w_;
    h = h_;

    // check the extension of the time file.
    CV_Assert(gammaPath.substr(gammaPath.find_last_of(".") + 1) == "yaml" ||
              gammaPath.substr(gammaPath.find_last_of(".") + 1) == "yml");

    FileStorage gammaFile;
    gammaFile.open(gammaPath, FileStorage::READ);
    CV_Assert(gammaFile.isOpened());

    FileNode gammaNode = gammaFile["gamma"];
    CV_Assert(gammaNode.type() == FileNode::SEQ);
    FileNodeIterator itS = gammaNode.begin(), itE = gammaNode.end();
    std::vector<float> GInvVec;
    for (; itS != itE; ++itS)
    {
        GInvVec.push_back((float) *itS);
    }
    CV_Assert(GInvVec.size() == 256);

    for (int i = 0; i < 256; i++) GInv[i] = GInvVec[i];
    for (int i = 0; i < 255; i++)
    {
        CV_Assert(GInv[i + 1] > GInv[i]);
    }
    float min = GInv[0];
    float max = GInv[255];
    for (int i = 0; i < 256; i++) GInv[i] = (float) (255.0 * (GInv[i] - min) / (max - min));
    for (int i = 1; i < 255; i++)
    {
        for (int s = 1; s < 255; s++)
        {
            if (GInv[s] <= i && GInv[s + 1] >= i)
            {
                G[i] = s + (i - GInv[s]) / (GInv[s + 1] - GInv[s]);
                break;
            }
        }
    }
    G[0] = 0;
    G[255] = 255;
    gammaFile.release();
    validGamma = true;
}

Mat GammaRemover::getUnGammaImageMat(Mat inputIm)
{
    CV_Assert(validGamma);
    uchar *inputImArr = inputIm.data;
    float *outImArr = new float[w * h];
    for (int i = 0; i < w * h; ++i)
    {
        outImArr[i] = GInv[inputImArr[i]];
    }
    Mat _outIm(h, w, CV_32F, outImArr);
    Mat outIm = _outIm * (1 / 255.0f);
    delete[] outImArr;
    return outIm;
}

void GammaRemover::getUnGammaImageVec(Mat inputIm, std::vector<float> &outImVec)
{
    CV_Assert(validGamma);
    uchar *inputImArr = inputIm.data;
    CV_Assert(outImVec.size() == (unsigned long) w * h);
    for (int i = 0; i < w * h; i++) outImVec[i] = GInv[inputImArr[i]];
}

} // namespace photometric_calib
} // namespace cv
