// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/photometric_calib/ResponseCalib.hpp"

#include <fstream>
#include <iostream>
#include <math.h>

namespace cv {
namespace photometric_calib {

ResponseCalib::ResponseCalib(std::string folderPath, std::string timePath, std::string imageFormat) : _leakPadding(2),
                                                                                                      _nIts(10),
                                                                                                      _skipFrames(1)
{
    imageReader = new Reader(folderPath, imageFormat, timePath);
}

ResponseCalib::ResponseCalib(std::string folderPath, std::string timePath, int leakPadding, int nIts, int skipFrames,
                             std::string imageFormat) :
        _leakPadding(leakPadding), _nIts(nIts), _skipFrames(skipFrames)
{
    imageReader = new Reader(folderPath, imageFormat, timePath);
}

Vec2d ResponseCalib::rmse(const double *G, const double *E, const std::vector<double> &exposureVec,
                          const std::vector<uchar *> &dataVec,
                          int wh)
{
    long double e = 0;        // yeah - these will be sums of a LOT of values, so we need super high precision.
    long double num = 0;

    size_t n = dataVec.size();
    for (size_t i = 0; i < n; i++)
    {
        for (int k = 0; k < wh; k++)
        {
            if (dataVec[i][k] == 255) continue;
            double r = G[dataVec[i][k]] - exposureVec[i] * E[k];
            if (!std::isfinite(r)) continue;
            e += r * r * 1e-10;
            num++;
        }
    }

    //return Eigen::Vector2d(1e5*sqrtl((e/num)), (double)num);
    return Vec2d((double) (1e5 * sqrt((e / num))), (double) num);
}

void ResponseCalib::plotE(const double *E, int w, int h, const std::string &saveTo)
{

    // try to find some good color scaling for plotting.
    double offset = 20;
    double min = 1e10, max = -1e10;

    double Emin = 1e10, Emax = -1e10;

    for (int i = 0; i < w * h; i++)
    {
        double le = log(E[i] + offset);
        if (le < min) min = le;
        if (le > max) max = le;

        if (E[i] < Emin) Emin = E[i];
        if (E[i] > Emax) Emax = E[i];
    }

    cv::Mat EImg = cv::Mat(h, w, CV_8UC3);
    cv::Mat EImg16 = cv::Mat(h, w, CV_16U);

    for (int i = 0; i < w * h; i++)
    {
        float val = (float) (3 * (exp((log(E[i] + offset) - min) / (max - min)) - 1) / 1.7183);

        int icP = (int) val;
        float ifP = val - icP;
        icP = icP % 3;

        Vec3b color;
        if (icP == 0) color = cv::Vec3b(0, 0, (uchar) (255 * ifP));
        if (icP == 1) color = cv::Vec3b(0, (uchar) (255 * ifP), 255);
        if (icP == 2) color = cv::Vec3b((uchar) (255 * ifP), 255, 255);

        EImg.at<cv::Vec3b>(i) = color;
        EImg16.at<ushort>(i) = (ushort) (255 * 255 * (E[i] - Emin) / (Emax - Emin));
    }

    std::cout << "Irradiance " << Emin << " - " << Emax << std::endl;
    cv::imshow("lnE", EImg);

    if (!saveTo.empty())
    {
        imwrite(saveTo + ".png", EImg);
        std::cout << "Saved: " << saveTo + ".png" << std::endl;
        imwrite(saveTo + "-16.png", EImg16);
        std::cout << "Saved: " << saveTo + "-16.png" << std::endl;
    }
}

void ResponseCalib::plotG(const double *G, const std::string &saveTo)
{
    cv::Mat GImg = cv::Mat(256, 256, CV_32FC1);
    GImg.setTo(0);

    double min = 1e10, max = -1e10;

    for (int i = 0; i < 256; i++)
    {
        if (G[i] < min) min = G[i];
        if (G[i] > max) max = G[i];
    }

    for (int i = 0; i < 256; i++)
    {
        double val = 256 * (G[i] - min) / (max - min);
        for (int k = 0; k < 256; k++)
        {
            if (val < k)
            {
                GImg.at<float>(255 - k, i) = (float) (k - val);
            }
        }
    }

    std::cout << "Inv. Response " << min << " - " << max << std::endl;
    cv::imshow("G", GImg);
    if (!saveTo.empty()) cv::imwrite(saveTo, GImg * 255);
    std::cout << "Saved: " << saveTo << std::endl;
}

void ResponseCalib::calib(bool debug)
{
    int w = 0, h = 0;
    size_t n = 0;

    std::vector<double> exposureDurationVec;
    std::vector<uchar *> dataVec;

    std::cout << "Preprocessing for response calibration... " << std::endl;
    for (unsigned long i = 0; i < imageReader->getNumImages(); i += _skipFrames)
    {
        cv::Mat img = imageReader->getImage(i);
        if (img.rows == 0 || img.cols == 0) continue;
        CV_Assert(img.type() == CV_8U);

        if ((w != 0 && w != img.cols) || img.cols == 0)
        {
            std::cout << "Width mismatch!" << std::endl;
            exit(1);
        }
        if ((h != 0 && h != img.rows) || img.rows == 0)
        {
            std::cout << "Height mismatch!" << std::endl;
            exit(1);
        }
        w = img.cols;
        h = img.rows;

        uchar *data = new uchar[w * h];
        memcpy(data, img.data, w * h);
        dataVec.push_back(data);
        exposureDurationVec.push_back((double) (imageReader->getExposureDuration(i)));

        unsigned char *data2 = new unsigned char[w * h];
        for (int j = 0; j < _leakPadding; ++j)
        {
            memcpy(data2, data, w * h);
            for (int y = 1; y < h - 1; ++y)
            {
                for (int x = 1; x < w - 1; ++x)
                {
                    if (data[x + y * w] == 255)
                    {
                        data2[x + 1 + w * (y + 1)] = 255;
                        data2[x + 1 + w * (y)] = 255;
                        data2[x + 1 + w * (y - 1)] = 255;

                        data2[x + w * (y + 1)] = 255;
                        data2[x + w * (y)] = 255;
                        data2[x + w * (y - 1)] = 255;

                        data2[x - 1 + w * (y + 1)] = 255;
                        data2[x - 1 + w * (y)] = 255;
                        data2[x - 1 + w * (y - 1)] = 255;
                    }
                }
            }
            memcpy(data, data2, w * h);
        }
        delete[] data2;
    }
    n = dataVec.size();
    std::cout << "Loaded " << n << " images!" << std::endl;
    std::cout << "Response calibration begin!" << std::endl;

    double *E = new double[w * h];        // scene irradiance
    double *En = new double[w * h];        // scene irradiance
    double *G = new double[256];        // inverse response function

    // set starting scene irradiance to mean of all images.
    memset(E, 0, sizeof(double) * w * h);
    memset(En, 0, sizeof(double) * w * h);
    memset(G, 0, sizeof(double) * 256);

    for (size_t i = 0; i < n; i++)
    {
        for (int k = 0; k < w * h; k++)
        {
            //if(dataVec[i][k]==255) continue;
            E[k] += dataVec[i][k];
            En[k]++;
        }
    }
    for (int k = 0; k < w * h; k++)
    {
        E[k] = E[k] / En[k];
    }

    // TODO: System independent folder creating
    // Only on Linux for now.
    if (-1 == system("rm -rf photoCalibResult"))
    {
        std::cout << "could not delete old photoCalibResult folder!" << std::endl;
    }
    if (-1 == system("mkdir photoCalibResult"))
    {
        std::cout << "could not create photoCalibResult folder!" << std::endl;
    }

    std::ofstream logFile;
    logFile.open("photoCalibResult/log.txt", std::ios::trunc | std::ios::out);
    logFile.precision(15);

    std::cout << "Initial RMSE = " << rmse(G, E, exposureDurationVec, dataVec, w * h)[0] << "!" << std::endl;
    if (debug)
    {
        plotE(E, w, h, "photoCalibResult/E-0");
        cv::waitKey(100);
    }

    bool optE = true;
    bool optG = true;

    for (int it = 0; it < _nIts; it++)
    {
        std::cout << "Iteration " << it + 1 << "..." << std::endl;
        if (optG)
        {
            // optimize log inverse response function.
            double *GSum = new double[256];
            double *GNum = new double[256];
            memset(GSum, 0, 256 * sizeof(double));
            memset(GNum, 0, 256 * sizeof(double));
            for (size_t i = 0; i < n; i++)
            {
                for (int k = 0; k < w * h; k++)
                {
                    int b = dataVec[i][k];
                    if (b == 255) continue;
                    GNum[b]++;
                    GSum[b] += E[k] * exposureDurationVec[i];
                }
            }
            for (int i = 0; i < 256; i++)
            {
                G[i] = GSum[i] / GNum[i];
                if (!std::isfinite(G[i]) && i > 1) G[i] = G[i - 1] + (G[i - 1] - G[i - 2]);
            }
            delete[] GSum;
            delete[] GNum;
            printf("optG RMSE = %f! \t", rmse(G, E, exposureDurationVec, dataVec, w * h)[0]);

            if (debug)
            {
                char buf[1000];
                snprintf(buf, 1000, "photoCalibResult/G-%02d.png", it + 1);
                plotG(G, buf);
            }
        }

        if (optE)
        {
            // optimize scene irradiance function.
            double *ESum = new double[w * h];
            double *ENum = new double[w * h];
            memset(ESum, 0, w * h * sizeof(double));
            memset(ENum, 0, w * h * sizeof(double));
            for (size_t i = 0; i < n; i++)
            {
                for (int k = 0; k < w * h; k++)
                {
                    int b = dataVec[i][k];
                    if (b == 255) continue;
                    ENum[k] += exposureDurationVec[i] * exposureDurationVec[i];
                    ESum[k] += (G[b]) * exposureDurationVec[i];
                }
            }
            for (int i = 0; i < w * h; i++)
            {
                E[i] = ESum[i] / ENum[i];
                if (E[i] < 0) E[i] = 0;
            }

            delete[] ENum;
            delete[] ESum;
            printf("OptE RMSE = %f!  \t", rmse(G, E, exposureDurationVec, dataVec, w * h)[0]);

            if (debug)
            {
                char buf[1000];
                snprintf(buf, 1000, "photoCalibResult/E-%02d", it + 1);
                plotE(E, w, h, buf);
            }
        }

        // rescale such that maximum response is 255 (fairly arbitrary choice).
        double rescaleFactor = 255.0 / G[255];
        for (int i = 0; i < w * h; i++)
        {
            E[i] *= rescaleFactor;
            if (i < 256) G[i] *= rescaleFactor;
        }

        Vec2d err = rmse(G, E, exposureDurationVec, dataVec, w * h);
        printf("Rescaled RMSE = %f!  \trescale with %f!\n\n", err[0], rescaleFactor);

        logFile << it << " " << n << " " << err[1] << " " << err[0] << "\n";

        cv::waitKey(100);
    }

    logFile.flush();
    logFile.close();

    std::ofstream lg;
    lg.open("photoCalibResult/pcalib.yaml", std::ios::trunc | std::ios::out);
    lg << "%YAML:1.0\ngamma: [";
    lg.precision(15);
    for (int i = 0; i < 255; i++)
    {
        lg << G[i] << ", ";
    }
    lg << G[255] << ']';
    lg << "\n";

    lg.flush();
    lg.close();

    std::cout << "pcalib file has been saved to: photoCalibResult/pcalib.yaml" << std::endl;

    delete[] E;
    delete[] En;
    delete[] G;
    for (size_t i = 0; i < n; i++)
    {
        delete[] dataVec[i];
    }

    std::cout << "Camera response function calibration finished!" << std::endl;
}

} // namespace photometric_calib
} // namespace cv