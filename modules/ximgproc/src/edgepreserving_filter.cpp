// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
//
//                    Created by Simon Reich
//

#include "precomp.hpp"

namespace cv
{
namespace ximgproc
{
using namespace std;

void edgePreservingFilter(InputArray _src, OutputArray _dst, int d,
                          double threshold)
{
    CV_Assert(_src.type() == CV_8UC3);

    Mat src = _src.getMat();

    // [re]create the output array so that it has the proper size and type.
    _dst.create(src.size(), src.type());
    Mat dst = _dst.getMat();
    src.copyTo(dst);

    if (d < 3)
        d = 3;
    int subwindowX = d, subwindowY = d;

    if (threshold < 0)
        threshold = 0;

    // number of image channels
    int nChannel = src.channels();

    vector<double> pixel(nChannel, 0);
    vector<vector<double>> line1(src.rows, pixel);
    vector<vector<vector<double>>> weight(src.cols,
                                          line1); // global weights
    vector<vector<vector<double>>> imageResult(
        src.cols, line1); // global normalized image

    // do algorithm
    cv::Mat subwindow, subwindow1;
    for (int posX = 0; posX < src.cols - subwindowX; posX++)
    {
        for (int posY = 0; posY < src.rows - subwindowY; posY++)
        {
            cv::Rect roi =
                cv::Rect(posX, posY, subwindowX, subwindowY);
            subwindow1 = src(roi);
            cv::GaussianBlur(subwindow1, subwindow, cv::Size(5, 5),
                             0.3, 0.3);

            // compute arithmetic mean of subwindow
            cv::Scalar ArithmeticMean = cv::mean(subwindow);

            // compute pixelwise distance
            vector<vector<double>> pixelwiseDist;

            for (int subPosX = 0; subPosX < subwindow.cols;
                 subPosX++)
            {
                vector<double> line;
                for (int subPosY = 0; subPosY < subwindow.rows;
                     subPosY++)
                {
                    cv::Vec3b intensity =
                        subwindow.at<cv::Vec3b>(subPosY,
                                                subPosX);
                    double distance =
                        ((double)intensity.val[0] -
                         ArithmeticMean[0]) *
                            ((double)intensity.val[0] -
                             ArithmeticMean[0]) +
                        ((double)intensity.val[1] -
                         ArithmeticMean[1]) *
                            ((double)intensity.val[1] -
                             ArithmeticMean[1]) +
                        ((double)intensity.val[2] -
                         ArithmeticMean[2]) *
                            ((double)intensity.val[2] -
                             ArithmeticMean[2]);
                    distance = sqrt(distance);

                    line.push_back(distance);
                };

                pixelwiseDist.push_back(line);
            };

            // compute mean pixelwise distance
            double meanPixelwiseDist = 0;

            for (int i = 0; i < (int)pixelwiseDist.size(); i++)
                for (int j = 0;
                     j < (int)pixelwiseDist[i].size(); j++)
                    meanPixelwiseDist +=
                        pixelwiseDist[i][j];

            meanPixelwiseDist /= ((int)pixelwiseDist.size() *
                                  (int)pixelwiseDist[0].size());

            // detect edge
            for (int subPosX = 0; subPosX < subwindow.cols;
                 subPosX++)
            {
                for (int subPosY = 0; subPosY < subwindow.rows;
                     subPosY++)
                {
                    if ((meanPixelwiseDist <= threshold &&
                         pixelwiseDist[subPosX][subPosY] <=
                             threshold) ||
                        (meanPixelwiseDist <= threshold &&
                         pixelwiseDist[subPosX][subPosY] >
                             threshold))
                    {
                        // global Position
                        int globalPosX = posX + subPosX;
                        int globalPosY = posY + subPosY;

                        // compute global weight
                        cv::Vec3b intensity =
                            subwindow.at<cv::Vec3b>(
                                subPosY, subPosX);
                        weight[globalPosX][globalPosY]
                              [0] +=
                            intensity.val[0] *
                            (threshold -
                             pixelwiseDist[subPosX]
                                          [subPosY]) *
                            (threshold -
                             pixelwiseDist[subPosX]
                                          [subPosY]);
                        weight[globalPosX][globalPosY]
                              [1] +=
                            intensity.val[1] *
                            (threshold -
                             pixelwiseDist[subPosX]
                                          [subPosY]) *
                            (threshold -
                             pixelwiseDist[subPosX]
                                          [subPosY]);
                        weight[globalPosX][globalPosY]
                              [2] +=
                            intensity.val[2] *
                            (threshold -
                             pixelwiseDist[subPosX]
                                          [subPosY]) *
                            (threshold -
                             pixelwiseDist[subPosX]
                                          [subPosY]);

                        // compute final image
                        imageResult[globalPosX]
                                   [globalPosY][0] +=
                            intensity.val[0] *
                            (threshold -
                             pixelwiseDist[subPosX]
                                          [subPosY]) *
                            (threshold -
                             pixelwiseDist[subPosX]
                                          [subPosY]) *
                            ArithmeticMean[0];
                        imageResult[globalPosX]
                                   [globalPosY][1] +=
                            intensity.val[1] *
                            (threshold -
                             pixelwiseDist[subPosX]
                                          [subPosY]) *
                            (threshold -
                             pixelwiseDist[subPosX]
                                          [subPosY]) *
                            ArithmeticMean[1];
                        imageResult[globalPosX]
                                   [globalPosY][2] +=
                            intensity.val[2] *
                            (threshold -
                             pixelwiseDist[subPosX]
                                          [subPosY]) *
                            (threshold -
                             pixelwiseDist[subPosX]
                                          [subPosY]) *
                            ArithmeticMean[2];
                    };
                };
            };
        };
    };

    // compute final image
    for (int globalPosX = 0; globalPosX < (int)imageResult.size();
         globalPosX++)
    {
        for (int globalPosY = 0;
             globalPosY < (int)imageResult[globalPosX].size();
             globalPosY++)
        {
            // cout << "globalPosX: " << globalPosX << "/"
            // << dst.cols << "," << imageResult.size () <<
            // "\tglobalPosY: " << globalPosY << "/" <<
            // dst.rows << "," <<imageResult.at
            // (globalPosX).size () << endl;

            // add image to result
            cv::Vec3b intensity =
                src.at<cv::Vec3b>(globalPosY, globalPosX);
            imageResult[globalPosX][globalPosY][0] +=
                (double)intensity.val[0];
            imageResult[globalPosX][globalPosY][1] +=
                (double)intensity.val[1];
            imageResult[globalPosX][globalPosY][2] +=
                (double)intensity.val[2];

            // normalize using weight
            imageResult[globalPosX][globalPosY][0] /=
                (weight[globalPosX][globalPosY][0] + 1);
            imageResult[globalPosX][globalPosY][1] /=
                (weight[globalPosX][globalPosY][1] + 1);
            imageResult[globalPosX][globalPosY][2] /=
                (weight[globalPosX][globalPosY][2] + 1);

            // copy to output image frame
            dst.at<cv::Vec3b>(globalPosY, globalPosX)[0] =
                (uchar)imageResult[globalPosX][globalPosY][0];
            dst.at<cv::Vec3b>(globalPosY, globalPosX)[1] =
                (uchar)imageResult[globalPosX][globalPosY][1];
            dst.at<cv::Vec3b>(globalPosY, globalPosX)[2] =
                (uchar)imageResult[globalPosX][globalPosY][2];
        };
    };
}
} // namespace ximgproc
} // namespace cv
