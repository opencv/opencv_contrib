// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/photometric_calib/VignetteCalib.hpp"

#include <fstream>
#include <iostream>

namespace cv {
namespace photometric_calib {


VignetteCalib::VignetteCalib(std::string folderPath, std::string timePath, std::string cameraFile,
                             std::string gammaFile, std::string imageFormat) :
        _imageSkip(1), _maxIterations(20), _outlierTh(15), _gridWidth(1000), _gridHeight(1000), _facW(5), _facH(5),
        _maxAbsGrad(255)
{
    imageReader = new Reader(folderPath, imageFormat, timePath);
    // check the extension of the camera file.
    CV_Assert(cameraFile.substr(cameraFile.find_last_of(".") + 1) == "yaml" ||
              cameraFile.substr(cameraFile.find_last_of(".") + 1) == "yml");
    FileStorage cameraStorage(cameraFile, FileStorage::READ);
    cameraStorage["cameraMatrix"] >> _cameraMatrix;
    cameraStorage["distCoeffs"] >> _distCoeffs;
    gammaRemover = new GammaRemover(gammaFile, imageReader->getWidth(), imageReader->getHeight());

    // affine map from plane cordinates to grid coordinates.
    _K_p2idx = Matx33f::eye();
    _K_p2idx(0, 0) = _gridWidth / _facW;
    _K_p2idx(1, 1) = _gridHeight / _facH;
    _K_p2idx(0, 2) = _gridWidth / 2.f;
    _K_p2idx(1, 2) = _gridHeight / 2.f;
    _K_p2idx_inverse = _K_p2idx.inv();

    _meanExposure = calMeanExposureTime();
}

VignetteCalib::VignetteCalib(std::string folderPath, std::string timePath, std::string cameraFile,
                             std::string gammaFile, std::string imageFormat, int imageSkip,
                             int maxIterations,
                             int outlierTh, int gridWidth, int gridHeight, float facW, float facH, int maxAbsGrad) :
        _imageSkip(imageSkip), _maxIterations(maxIterations), _outlierTh(outlierTh), _gridWidth(gridWidth),
        _gridHeight(gridHeight),
        _facW(facW), _facH(facH), _maxAbsGrad(maxAbsGrad)
{
    imageReader = new Reader(folderPath, imageFormat, timePath);
    // check the extension of the camera file.
    CV_Assert(cameraFile.substr(cameraFile.find_last_of(".") + 1) == "yaml" ||
              cameraFile.substr(cameraFile.find_last_of(".") + 1) == "yml");
    FileStorage cameraStorage(cameraFile, FileStorage::READ);
    cameraStorage["cameraMatrix"] >> _cameraMatrix;
    cameraStorage["distCoeffs"] >> _distCoeffs;
    gammaRemover = new GammaRemover(gammaFile, imageReader->getWidth(), imageReader->getHeight());

    // affine map from plane cordinates to grid coordinates.
    _K_p2idx = Matx33f::eye();
    _K_p2idx(0, 0) = _gridWidth / _facW;
    _K_p2idx(1, 1) = _gridHeight / _facH;
    _K_p2idx(0, 2) = _gridWidth / 2.f;
    _K_p2idx(1, 2) = _gridHeight / 2.f;
    _K_p2idx_inverse = _K_p2idx.inv();

    _meanExposure = calMeanExposureTime();
}

float VignetteCalib::getInterpolatedElement(const float *const mat, const float x, const float y, const int width)
{
    int ix = (int) x;
    int iy = (int) y;
    float dx = x - ix;
    float dy = y - iy;
    float dxdy = dx * dy;
    const float *bp = mat + ix + iy * width;

    float res = dxdy * bp[1 + width]
                + (dy - dxdy) * bp[width]
                + (dx - dxdy) * bp[1]
                + (1 - dx - dy + dxdy) * bp[0];

    return res;
}

void VignetteCalib::displayImage(float *I, int w, int h, std::string name)
{
    float vmin = 1e10;
    float vmax = -1e10;

    for (int i = 0; i < w * h; i++)
    {
        if (vmin > I[i]) vmin = I[i];
        if (vmax < I[i]) vmax = I[i];
    }

    Mat img = Mat(h, w, CV_8UC3);

    for (int i = 0; i < w * h; i++)
    {
        //isnanf? isnan?
        if (cvIsNaN(I[i]) == 1)
        { img.at<Vec3b>(i) = Vec3b(0, 0, 255); }
        else
        {
            img.at<Vec3b>(i) = Vec3b((uchar) (255 * (I[i] - vmin) / (vmax - vmin)),
                                     (uchar) (255 * (I[i] - vmin) / (vmax - vmin)),
                                     (uchar) (255 * (I[i] - vmin) / (vmax - vmin)));
        }
    }

    std::cout << "plane image values " << vmin << " - " << vmax << "!" << std::endl;

    imshow(name, img);
    imwrite("vignetteCalibResult/plane.png", img);
}

void VignetteCalib::displayImageV(float *I, int w, int h, std::string name)
{
    Mat img = Mat(h, w, CV_8UC3);
    for (int i = 0; i < w * h; i++)
    {
        if (cvIsNaN(I[i]) == 1)
        {
            img.at<Vec3b>(i) = Vec3b(0, 0, 255);
        }
        else
        {
            float c = 254 * I[i];
            img.at<Vec3b>(i) = Vec3b((uchar) c, (uchar) c, (uchar) c);
        }

    }
    imshow(name, img);
}

float VignetteCalib::calMeanExposureTime()
{
    float meanExposure = 0.f;
    for (unsigned long i = 0; i < imageReader->getNumImages(); i += _imageSkip)
    {
        meanExposure += imageReader->getExposureDuration(i);
    }
    meanExposure = meanExposure / imageReader->getNumImages();
    if (meanExposure == 0) meanExposure = 1;
    return meanExposure;
}

bool VignetteCalib::preCalib(unsigned long id, float *&image, float *&plane2imgX, float *&plane2imgY, bool debug)
{
    int wI = imageReader->getWidth();
    int hI = imageReader->getHeight();

    std::vector<int> markerIds;
    std::vector<std::vector<Point2f> > markerCorners, rejectedCandidates;

    Mat oriImg = imageReader->getImage(id);
    Mat undisImg;
    undistort(oriImg, undisImg, _cameraMatrix, _distCoeffs);

    Ptr<aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_ARUCO_ORIGINAL);

    aruco::detectMarkers(undisImg, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
    if (markerCorners.size() != 1)
    {
        return false;
    }

    std::vector<Point2f> ptsP;
    std::vector<Point2f> ptsI;
    ptsI.push_back(Point2f(markerCorners[0][0].x, markerCorners[0][0].y));
    ptsI.push_back(Point2f(markerCorners[0][1].x, markerCorners[0][1].y));
    ptsI.push_back(Point2f(markerCorners[0][2].x, markerCorners[0][2].y));
    ptsI.push_back(Point2f(markerCorners[0][3].x, markerCorners[0][3].y));
    ptsP.push_back(Point2f(-0.5, 0.5));
    ptsP.push_back(Point2f(0.5, 0.5));
    ptsP.push_back(Point2f(0.5, -0.5));
    ptsP.push_back(Point2f(-0.5, -0.5));

    Matx33f H(findHomography(ptsP, ptsI));

    std::vector<float> imgRaw(wI * hI);
    gammaRemover->getUnGammaImageVec(imageReader->getImage(id), imgRaw);

    plane2imgX = new float[_gridWidth * _gridHeight];
    plane2imgY = new float[_gridWidth * _gridHeight];

    Matx33f HK = H * _K_p2idx_inverse;

    int idx = 0;
    for (int y = 0; y < _gridHeight; y++)
    {
        for (int x = 0; x < _gridWidth; x++)
        {
            Vec3f pp = HK * Vec3f((float) x, (float) y, 1);
            plane2imgX[idx] = pp[0] / pp[2];
            plane2imgY[idx] = pp[1] / pp[2];
            idx++;
        }
    }

    float expDuration;
    expDuration = (imageReader->getExposureDuration(id) == 0 ? 1 : imageReader->getExposureDuration(id));
    image = new float[wI * hI];
    for (int y = 0; y < hI; y++)
    {
        for (int x = 0; x < wI; x++)
        {
            image[x + y * wI] = _meanExposure * imgRaw[x + y * wI] / expDuration;
        }
    }

    for (int y = 2; y < hI - 2; y++)
    {
        for (int x = 2; x < wI - 2; x++)
        {
            for (int deltax = -2; deltax < 3; deltax++)
            {
                for (int deltay = -2; deltay < 3; deltay++)
                {
                    if (fabsf(image[x + y * wI] - image[x + deltax + (y + deltay) * wI]) > _maxAbsGrad)
                    {
                        image[x + y * wI] = NAN;
                        image[x + deltax + (y + deltay) * wI] = NAN;
                    }
                }
            }
        }
    }

    if (debug)
    {
        // debug-plot.
        Mat dbgImg(hI, wI, CV_8UC3);
        for (int j = 0; j < wI * hI; j++)
        {
            dbgImg.at<Vec3b>(j) = Vec3b((uchar) imgRaw[j], (uchar) imgRaw[j], (uchar) imgRaw[j]);
        }

        for (int x = 0; x <= _gridWidth; x += 200)
        {
            for (int y = 0; y <= _gridHeight; y += 10)
            {
                int idxS = (x < _gridWidth ? x : _gridWidth - 1) + (y < _gridHeight ? y : _gridHeight - 1) * _gridWidth;
                int idxT = (x < _gridWidth ? x : _gridWidth - 1) +
                           ((y + 10) < _gridHeight ? (y + 10) : _gridHeight - 1) * _gridWidth;

                int u_dS = (int) lround((plane2imgX[idxS] + 0.5));
                int v_dS = (int) lround((plane2imgY[idxS] + 0.5));

                int u_dT = (int) lround((plane2imgX[idxT] + 0.5));
                int v_dT = (int) lround((plane2imgY[idxT] + 0.5));

                if (u_dS >= 0 && v_dS >= 0 && u_dS < wI && v_dS < hI && u_dT >= 0 && v_dT >= 0 && u_dT < wI &&
                    v_dT < hI)
                {
                    line(dbgImg, Point(u_dS, v_dS), Point(u_dT, v_dT), Scalar(0, 0, 255), 10, LINE_AA);
                }
            }
        }

        for (int x = 0; x <= _gridWidth; x += 10)
        {
            for (int y = 0; y <= _gridHeight; y += 200)
            {
                int idxS = (x < _gridWidth ? x : _gridWidth - 1) + (y < _gridHeight ? y : _gridHeight - 1) * _gridWidth;
                int idxT = ((x + 10) < _gridWidth ? (x + 10) : _gridWidth - 1) +
                           (y < _gridHeight ? y : _gridHeight - 1) * _gridWidth;

                int u_dS = (int) lround(plane2imgX[idxS] + 0.5);
                int v_dS = (int) lround(plane2imgY[idxS] + 0.5);

                int u_dT = (int) lround(plane2imgX[idxT] + 0.5);
                int v_dT = (int) lround(plane2imgY[idxT] + 0.5);

                if (u_dS >= 0 && v_dS >= 0 && u_dS < wI && v_dS < hI && u_dT >= 0 && v_dT >= 0 && u_dT < wI &&
                    v_dT < hI)
                {
                    line(dbgImg, Point(u_dS, v_dS), Point(u_dT, v_dT), Scalar(0, 0, 255), 10, LINE_AA);
                }
            }
        }

        for (int x = 0; x < _gridWidth; x++)
        {
            for (int y = 0; y < _gridHeight; y++)
            {
                int u_d = (int) lround(plane2imgX[x + y * _gridWidth] + 0.5);
                int v_d = (int) lround(plane2imgY[x + y * _gridWidth] + 0.5);
                if (!(u_d > 1 && v_d > 1 && u_d < wI - 2 && v_d < hI - 2))
                {
                    plane2imgX[x + y * _gridWidth] = NAN;
                    plane2imgY[x + y * _gridWidth] = NAN;
                }
            }
        }

        imshow("inRaw", dbgImg);
        waitKey(1);

        if (rand() % 40 == 0)
        {
            char buf[1000];
            snprintf(buf, 1000, "vignetteCalibResult/img%u.png", (unsigned) id);
            imwrite(buf, dbgImg);
        }
    }

    else
    {
        for (int x = 0; x < _gridWidth; x++)
        {
            for (int y = 0; y < _gridHeight; y++)
            {
                int u_d = (int) lround(plane2imgX[x + y * _gridWidth] + 0.5);
                int v_d = (int) lround(plane2imgY[x + y * _gridWidth] + 0.5);
                if (!(u_d > 1 && v_d > 1 && u_d < wI - 2 && v_d < hI - 2))
                {
                    plane2imgX[x + y * _gridWidth] = NAN;
                    plane2imgY[x + y * _gridWidth] = NAN;
                }
            }
        }
    }

    return true;
}

void VignetteCalib::calib(bool debug)
{
    // Create folder for vignette calibration
    if (-1 == system("rm -rf vignetteCalibResult"))
    {
        std::cout << "could not delete old vignetteCalibResult folder!" << std::endl;
    }
    if (-1 == system("mkdir vignetteCalibResult"))
    {
        std::cout << "could not delete old vignetteCalibResult folder" << std::endl;
    }

    int wO, hO;
    wO = imageReader->getWidth();
    hO = imageReader->getHeight();
    int wI = wO, hI = hO;

    // log file
    std::ofstream logFile;
    logFile.open("vignetteCalibResult/log.txt", std::ios::trunc | std::ios::out);
    logFile.precision(15);

    // number of images in total
    unsigned long n = imageReader->getNumImages();

    float *planeColor = new float[_gridWidth * _gridHeight];
    float *planeColorFF = new float[_gridWidth * _gridHeight];
    float *planeColorFC = new float[_gridWidth * _gridHeight];
    float *vignetteFactor = new float[hI * wI];
    float *vignetteFactorTT = new float[hI * wI];
    float *vignetteFactorCT = new float[hI * wI];

    // initialize vignette factors to 1.
    for (int i = 0; i < hI * wI; i++) vignetteFactor[i] = 1;

    double E = 0;
    double R = 0;

    // optimization begins
    for (int it = 0; it < _maxIterations; it++)
    {
        std::cout << "Iteration " << it << "..." << std::endl;

        int oth2 = _outlierTh * _outlierTh;
        if (it < _maxIterations / 2) oth2 = 10000 * 10000;

        // ============================ optimize planeColor ================================
        std::cout << "Optimize planeColor..." << std::endl;
        memset(planeColorFF, 0, _gridWidth * _gridHeight * sizeof(float));
        memset(planeColorFC, 0, _gridWidth * _gridHeight * sizeof(float));
        E = 0;
        R = 0;

        // for each plane pixel, it's optimum is at sum(CF)/sum(FF)
        for (unsigned long img = 0; img < n; img++)    // for all images
        {
            float *plane2imgX = NULL;
            float *plane2imgY = NULL;
            float *image = NULL;
            if (!preCalib(img, image, plane2imgX, plane2imgY, debug))
            {
                continue;
            }
            for (int pi = 0; pi < _gridWidth * _gridHeight; pi++)        // for all plane points
            {
                if (cvIsNaN(plane2imgX[pi]) == 1) continue;
                // get vignetted color at that point, and add to build average.
                float color = getInterpolatedElement(image, plane2imgX[pi], plane2imgY[pi], wI);
                float fac = getInterpolatedElement(vignetteFactor, plane2imgX[pi], plane2imgY[pi], wI);
                if (cvIsNaN(fac) == 1) continue;
                if (cvIsNaN(color) == 1) continue;
                double residual = (double) ((color - planeColor[pi] * fac) * (color - planeColor[pi] * fac));
                if (abs(residual) > oth2)
                {
                    E += oth2;
                    R++;
                    continue;
                }
                planeColorFF[pi] += fac * fac;
                planeColorFC[pi] += color * fac;
                if (cvIsNaN(planeColor[pi]) == 1) continue;
                E += residual;
                R++;
            }
            delete[] plane2imgX;
            delete[] plane2imgY;
            delete[] image;
        }

        for (int pi = 0; pi < _gridWidth * _gridWidth; pi++)        // for all plane points
        {
            if (planeColorFF[pi] < 1)
            {
                planeColor[pi] = NAN;
            }
            else
            {
                planeColor[pi] = planeColorFC[pi] / planeColorFF[pi];
            }
        }
        if (debug)
        {
            displayImage(planeColor, _gridWidth, _gridWidth, "Plane");
        }
        std::cout << R << " residual terms => " << sqrt(E / R) << std::endl;

        // ================================ optimize vignette =======================================
        std::cout << "Optimize Vignette..." << std::endl;
        memset(vignetteFactorTT, 0, hI * wI * sizeof(float));
        memset(vignetteFactorCT, 0, hI * wI * sizeof(float));
        E = 0;
        R = 0;
        for (unsigned long img = 0; img < n; img++)    // for all images
        {
            float *plane2imgX = NULL;
            float *plane2imgY = NULL;
            float *image = NULL;
            if (!preCalib(img, image, plane2imgX, plane2imgY, debug))
            {
                continue;
            }
            for (int pi = 0; pi < _gridWidth * _gridWidth; pi++)        // for all plane points
            {
                if (cvIsNaN(plane2imgX[pi]) == 1) continue;
                float x = plane2imgX[pi];
                float y = plane2imgY[pi];
                float colorImage = getInterpolatedElement(image, x, y, wI);
                float fac = getInterpolatedElement(vignetteFactor, x, y, wI);
                float colorPlane = planeColor[pi];
                if (cvIsNaN(colorPlane) == 1) continue;
                if (cvIsNaN(colorImage) == 1) continue;
                double residual = (double) ((colorImage - colorPlane * fac) * (colorImage - colorPlane * fac));
                if (abs(residual) > oth2)
                {
                    E += oth2;
                    R++;
                    continue;
                }
                int ix = (int) x;
                int iy = (int) y;
                float dx = x - ix;
                float dy = y - iy;
                float dxdy = dx * dy;
                vignetteFactorTT[ix + iy * wI + 0] += (1 - dx - dy + dxdy) * colorPlane * colorPlane;
                vignetteFactorTT[ix + iy * wI + 1] += (dx - dxdy) * colorPlane * colorPlane;
                vignetteFactorTT[ix + iy * wI + wI] += (dy - dxdy) * colorPlane * colorPlane;
                vignetteFactorTT[ix + iy * wI + 1 + wI] += dxdy * colorPlane * colorPlane;
                vignetteFactorCT[ix + iy * wI + 0] += (1 - dx - dy + dxdy) * colorImage * colorPlane;
                vignetteFactorCT[ix + iy * wI + 1] += (dx - dxdy) * colorImage * colorPlane;
                vignetteFactorCT[ix + iy * wI + wI] += (dy - dxdy) * colorImage * colorPlane;
                vignetteFactorCT[ix + iy * wI + 1 + wI] += dxdy * colorImage * colorPlane;
                if (cvIsNaN(fac) == 1) continue;
                E += residual;
                R++;
            }
            delete[] plane2imgX;
            delete[] plane2imgY;
            delete[] image;
        }
        float maxFac = 0;
        for (int pi = 0; pi < hI * wI; pi++)        // for all plane points
        {
            if (vignetteFactorTT[pi] < 1)
            {
                vignetteFactor[pi] = NAN;
            }
            else
            {
                vignetteFactor[pi] = vignetteFactorCT[pi] / vignetteFactorTT[pi];
                if (vignetteFactor[pi] > maxFac) maxFac = vignetteFactor[pi];
            }
        }
        std::cout << R << " residual terms => " << sqrt(E / R) << std::endl;

        // normalize to vignette max. factor 1.
        for (int pi = 0; pi < hI * wI; pi++)
        {
            vignetteFactor[pi] /= maxFac;
        }
        logFile << it << " " << n << " " << R << " " << sqrt(E / R) << "\n";

        // dilate & smoothe vignette by 4 pixel for output.
        // does not change anything in the optimization; uses vignetteFactorTT and vignetteFactorCT for temporary storing
        memcpy(vignetteFactorTT, vignetteFactor, sizeof(float) * hI * wI);
        for (int dilit = 0; dilit < 4; dilit++)
        {
            memcpy(vignetteFactorCT, vignetteFactorTT, sizeof(float) * hI * wI);
            for (int y = 0; y < hI; y++)
            {
                for (int x = 0; x < wI; x++)
                {
                    int idx = x + y * wI;
                    {
                        float sum = 0, num = 0;
                        if (x < wI - 1 && y < hI - 1 && cvIsNaN(vignetteFactorCT[idx + 1 + wI]) != 1)
                        {
                            sum += vignetteFactorCT[idx + 1 + wI];
                            num++;
                        }
                        if (x < wI - 1 && cvIsNaN(vignetteFactorCT[idx + 1]) != 1)
                        {
                            sum += vignetteFactorCT[idx + 1];
                            num++;
                        }
                        if (x < wI - 1 && y > 0 && cvIsNaN(vignetteFactorCT[idx + 1 - wI]) != 1)
                        {
                            sum += vignetteFactorCT[idx + 1 - wI];
                            num++;
                        }

                        if (y < hI - 1 && cvIsNaN(vignetteFactorCT[idx + wI]) != 1)
                        {
                            sum += vignetteFactorCT[idx + wI];
                            num++;
                        }
                        if (cvIsNaN(vignetteFactorCT[idx]) != 1)
                        {
                            sum += vignetteFactorCT[idx];
                            num++;
                        }
                        if (y > 0 && cvIsNaN(vignetteFactorCT[idx - wI]) != 1)
                        {
                            sum += vignetteFactorCT[idx - wI];
                            num++;
                        }

                        if (y < hI - 1 && x > 0 && cvIsNaN(vignetteFactorCT[idx - 1 + wI]) != 1)
                        {
                            sum += vignetteFactorCT[idx - 1 + wI];
                            num++;
                        }
                        if (x > 0 && cvIsNaN(vignetteFactorCT[idx - 1]) != 1)
                        {
                            sum += vignetteFactorCT[idx - 1];
                            num++;
                        }
                        if (y > 0 && x > 0 && cvIsNaN(vignetteFactorCT[idx - 1 - wI]) != 1)
                        {
                            sum += vignetteFactorCT[idx - 1 - wI];
                            num++;
                        }

                        if (num > 0) vignetteFactorTT[idx] = sum / num;
                    }
                }
            }
        }

        // ================================ Store Vignette Image =======================================
        if (debug)
        {
            displayImageV(vignetteFactorTT, wI, hI, "VignetteSmoothed");
        }
        Mat wrapSmoothed = Mat(hI, wI, CV_32F, vignetteFactorTT) * 254.9 * 254.9;
        Mat wrapSmoothed16;
        wrapSmoothed.convertTo(wrapSmoothed16, CV_16U, 1, 0);
        imwrite("vignetteCalibResult/vignetteSmoothed.png", wrapSmoothed16);
        waitKey(50);

        if (debug)
        {
            displayImageV(vignetteFactor, wI, hI, "VignetteOrg");
        }
        Mat wrap = Mat(hI, wI, CV_32F, vignetteFactor) * 254.9 * 254.9;
        Mat wrap16;
        wrap.convertTo(wrap16, CV_16U, 1, 0);
        imwrite("vignetteCalibResult/vignette.png", wrap16);
        waitKey(50);
    }

    logFile.flush();
    logFile.close();

    delete[] planeColor;
    delete[] planeColorFF;
    delete[] planeColorFC;
    delete[] vignetteFactor;
    delete[] vignetteFactorTT;
    delete[] vignetteFactorCT;
}

void VignetteCalib::calibFast(bool debug)
{
    std::cout<<"Fast mode! This requires large memory (10GB+)!"<<std::endl;

    if (-1 == system("rm -rf vignetteCalibResult"))
    {
        std::cout << "could not delete old vignetteCalibResult folder!" << std::endl;
    }
    if (-1 == system("mkdir vignetteCalibResult"))
    {
        std::cout << "could not delete old vignetteCalibResult folder" << std::endl;
    }

    int wO, hO;
    wO = imageReader->getWidth();
    hO = imageReader->getHeight();
    int wI = wO, hI = hO;

    Ptr<aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_ARUCO_ORIGINAL);

    std::vector<float *> images;
    std::vector<float *> p2imgX;
    std::vector<float *> p2imgY;

    std::cout<<"Preprocessing images..."<<std::endl;
    for (unsigned long i = 0; i < imageReader->getNumImages(); ++i)
    {
        std::vector<int> markerIds;
        std::vector<std::vector<Point2f> > markerCorners, rejectedCandidates;

        Mat oriImg = imageReader->getImage(i);
        Mat undisImg;
        undistort(oriImg, undisImg, _cameraMatrix, _distCoeffs);

        aruco::detectMarkers(undisImg, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

        if (markerCorners.size() != 1) continue;

        std::vector<Point2f> ptsP;
        std::vector<Point2f> ptsI;
        ptsI.push_back(Point2f(markerCorners[0][0].x, markerCorners[0][0].y));
        ptsI.push_back(Point2f(markerCorners[0][1].x, markerCorners[0][1].y));
        ptsI.push_back(Point2f(markerCorners[0][2].x, markerCorners[0][2].y));
        ptsI.push_back(Point2f(markerCorners[0][3].x, markerCorners[0][3].y));
        ptsP.push_back(Point2f(-0.5, 0.5));
        ptsP.push_back(Point2f(0.5, 0.5));
        ptsP.push_back(Point2f(0.5, -0.5));
        ptsP.push_back(Point2f(-0.5, -0.5));

        Matx33f H(findHomography(ptsP, ptsI));

        std::vector<float> imgRaw(wI * hI);
        gammaRemover->getUnGammaImageVec(imageReader->getImage(i), imgRaw);

        float *plane2imgX = new float[_gridWidth * _gridHeight];
        float *plane2imgY = new float[_gridWidth * _gridHeight];

        Matx33f HK = H * _K_p2idx_inverse;

        int idx = 0;
        for (int y = 0; y < _gridHeight; y++)
        {
            for (int x = 0; x < _gridWidth; x++)
            {
                Vec3f pp = HK * Vec3f((float) x, (float) y, 1);
                plane2imgX[idx] = pp[0] / pp[2];
                plane2imgY[idx] = pp[1] / pp[2];
                idx++;
            }
        }

        float expDuration;
        expDuration = (imageReader->getExposureDuration(i) == 0 ? 1 : imageReader->getExposureDuration(i));
        float *image = new float[wI * hI];
        for (int y = 0; y < hI; y++)
        {
            for (int x = 0; x < wI; x++)
            {
                image[x + y * wI] = _meanExposure * imgRaw[x + y * wI] / expDuration;
            }
        }

        for (int y = 2; y < hI - 2; y++)
        {
            for (int x = 2; x < wI - 2; x++)
            {
                for (int deltax = -2; deltax < 3; deltax++)
                {
                    for (int deltay = -2; deltay < 3; deltay++)
                    {
                        if (fabsf(image[x + y * wI] - image[x + deltax + (y + deltay) * wI]) > _maxAbsGrad)
                        {
                            image[x + y * wI] = NAN;
                            image[x + deltax + (y + deltay) * wI] = NAN;
                        }
                    }
                }
            }
        }

        images.push_back(image);

        if (debug)
        {
            // debug-plot.
            Mat dbgImg(hI, wI, CV_8UC3);
            for (int j = 0; j < wI * hI; j++)
            {
                dbgImg.at<Vec3b>(j) = Vec3b((uchar) imgRaw[j], (uchar) imgRaw[j], (uchar) imgRaw[j]);
            }

            for (int x = 0; x <= _gridWidth; x += 200)
            {
                for (int y = 0; y <= _gridHeight; y += 10)
                {
                    int idxS = (x < _gridWidth ? x : _gridWidth - 1) +
                               (y < _gridHeight ? y : _gridHeight - 1) * _gridWidth;
                    int idxT = (x < _gridWidth ? x : _gridWidth - 1) +
                               ((y + 10) < _gridHeight ? (y + 10) : _gridHeight - 1) * _gridWidth;

                    int u_dS = (int) lround((plane2imgX[idxS] + 0.5));
                    int v_dS = (int) lround((plane2imgY[idxS] + 0.5));

                    int u_dT = (int) lround((plane2imgX[idxT] + 0.5));
                    int v_dT = (int) lround((plane2imgY[idxT] + 0.5));

                    if (u_dS >= 0 && v_dS >= 0 && u_dS < wI && v_dS < hI && u_dT >= 0 && v_dT >= 0 && u_dT < wI &&
                        v_dT < hI)
                    {
                        line(dbgImg, Point(u_dS, v_dS), Point(u_dT, v_dT), Scalar(0, 0, 255), 10, LINE_AA);
                    }
                }
            }

            for (int x = 0; x <= _gridWidth; x += 10)
            {
                for (int y = 0; y <= _gridHeight; y += 200)
                {
                    int idxS = (x < _gridWidth ? x : _gridWidth - 1) +
                               (y < _gridHeight ? y : _gridHeight - 1) * _gridWidth;
                    int idxT = ((x + 10) < _gridWidth ? (x + 10) : _gridWidth - 1) +
                               (y < _gridHeight ? y : _gridHeight - 1) * _gridWidth;

                    int u_dS = (int) lround(plane2imgX[idxS] + 0.5);
                    int v_dS = (int) lround(plane2imgY[idxS] + 0.5);

                    int u_dT = (int) lround(plane2imgX[idxT] + 0.5);
                    int v_dT = (int) lround(plane2imgY[idxT] + 0.5);

                    if (u_dS >= 0 && v_dS >= 0 && u_dS < wI && v_dS < hI && u_dT >= 0 && v_dT >= 0 && u_dT < wI &&
                        v_dT < hI)
                    {
                        line(dbgImg, Point(u_dS, v_dS), Point(u_dT, v_dT), Scalar(0, 0, 255), 10, LINE_AA);
                    }
                }
            }

            for (int x = 0; x < _gridWidth; x++)
            {
                for (int y = 0; y < _gridHeight; y++)
                {
                    int u_d = (int) lround(plane2imgX[x + y * _gridWidth] + 0.5);
                    int v_d = (int) lround(plane2imgY[x + y * _gridWidth] + 0.5);

                    if (!(u_d > 1 && v_d > 1 && u_d < wI - 2 && v_d < hI - 2))
                    {
                        plane2imgX[x + y * _gridWidth] = NAN;
                        plane2imgY[x + y * _gridWidth] = NAN;
                    }
                }
            }

            imshow("inRaw", dbgImg);

            if (rand() % 40 == 0)
            {
                char buf[1000];
                snprintf(buf, 1000, "vignetteCalibResult/img%u.png", (unsigned) i);
                imwrite(buf, dbgImg);
            }

            waitKey(1);
        }

        else
        {
            for (int x = 0; x < _gridWidth; x++)
            {
                for (int y = 0; y < _gridHeight; y++)
                {
                    int u_d = (int) lround(plane2imgX[x + y * _gridWidth] + 0.5);
                    int v_d = (int) lround(plane2imgY[x + y * _gridWidth] + 0.5);

                    if (!(u_d > 1 && v_d > 1 && u_d < wI - 2 && v_d < hI - 2))
                    {
                        plane2imgX[x + y * _gridWidth] = NAN;
                        plane2imgY[x + y * _gridWidth] = NAN;
                    }
                }
            }
        }

        p2imgX.push_back(plane2imgX);
        p2imgY.push_back(plane2imgY);

    }

    std::ofstream logFile;
    logFile.open("vignetteCalibResult/log.txt", std::ios::trunc | std::ios::out);
    logFile.precision(15);

    unsigned long n = imageReader->getNumImages();

    float *planeColor = new float[_gridWidth * _gridHeight];
    float *planeColorFF = new float[_gridWidth * _gridHeight];
    float *planeColorFC = new float[_gridWidth * _gridHeight];
    float *vignetteFactor = new float[hI * wI];
    float *vignetteFactorTT = new float[hI * wI];
    float *vignetteFactorCT = new float[hI * wI];

    // initialize vignette factors to 1.
    for (int i = 0; i < hI * wI; i++) vignetteFactor[i] = 1;

    double E = 0;
    double R = 0;

    for (int it = 0; it < _maxIterations; it++)
    {
        int oth2 = _outlierTh * _outlierTh;
        if (it < _maxIterations / 2) oth2 = 10000 * 10000;

        // ============================ optimize planeColor ================================
        memset(planeColorFF, 0, _gridWidth * _gridHeight * sizeof(float));
        memset(planeColorFC, 0, _gridWidth * _gridHeight * sizeof(float));
        E = 0;
        R = 0;

        // for each plane pixel, it's optimum is at sum(CF)/sum(FF)
        for (unsigned long img = 0; img < n; img++)    // for all images
        {
            float *plane2imgX = p2imgX[img];
            float *plane2imgY = p2imgY[img];
            float *image = images[img];

            for (int pi = 0; pi < _gridWidth * _gridHeight; pi++)        // for all plane points
            {
                if (cvIsNaN(plane2imgX[pi]) == 1) continue;

                // get vignetted color at that point, and add to build average.
                float color = getInterpolatedElement(image, plane2imgX[pi], plane2imgY[pi], wI);
                float fac = getInterpolatedElement(vignetteFactor, plane2imgX[pi], plane2imgY[pi], wI);

                if (cvIsNaN(fac) == 1) continue;
                if (cvIsNaN(color) == 1) continue;

                double residual = (double) ((color - planeColor[pi] * fac) * (color - planeColor[pi] * fac));
                if (abs(residual) > oth2)
                {
                    E += oth2;
                    R++;
                    continue;
                }


                planeColorFF[pi] += fac * fac;
                planeColorFC[pi] += color * fac;

                if (cvIsNaN(planeColor[pi]) == 1) continue;
                E += residual;
                R++;
            }
        }

        for (int pi = 0; pi < _gridWidth * _gridWidth; pi++)        // for all plane points
        {
            if (planeColorFF[pi] < 1)
            {
                planeColor[pi] = NAN;
            }
            else
            {
                planeColor[pi] = planeColorFC[pi] / planeColorFF[pi];
            }
        }
        if (debug)
        {
            displayImage(planeColor, _gridWidth, _gridWidth, "Plane");
        }

        std::cout << R << " residual terms => " << sqrt(E / R) << std::endl;

        // ================================ optimize vignette =======================================
        memset(vignetteFactorTT, 0, hI * wI * sizeof(float));
        memset(vignetteFactorCT, 0, hI * wI * sizeof(float));
        E = 0;
        R = 0;

        for (unsigned long img = 0; img < n; img++)    // for all images
        {
            float *plane2imgX = p2imgX[img];
            float *plane2imgY = p2imgY[img];
            float *image = images[img];

            for (int pi = 0; pi < _gridWidth * _gridWidth; pi++)        // for all plane points
            {
                if (cvIsNaN(plane2imgX[pi]) == 1) continue;
                float x = plane2imgX[pi];
                float y = plane2imgY[pi];

                float colorImage = getInterpolatedElement(image, x, y, wI);
                float fac = getInterpolatedElement(vignetteFactor, x, y, wI);
                float colorPlane = planeColor[pi];

                if (cvIsNaN(colorPlane) == 1) continue;
                if (cvIsNaN(colorImage) == 1) continue;

                double residual = (double) ((colorImage - colorPlane * fac) * (colorImage - colorPlane * fac));
                if (abs(residual) > oth2)
                {
                    E += oth2;
                    R++;
                    continue;
                }


                int ix = (int) x;
                int iy = (int) y;
                float dx = x - ix;
                float dy = y - iy;
                float dxdy = dx * dy;

                vignetteFactorTT[ix + iy * wI + 0] += (1 - dx - dy + dxdy) * colorPlane * colorPlane;
                vignetteFactorTT[ix + iy * wI + 1] += (dx - dxdy) * colorPlane * colorPlane;
                vignetteFactorTT[ix + iy * wI + wI] += (dy - dxdy) * colorPlane * colorPlane;
                vignetteFactorTT[ix + iy * wI + 1 + wI] += dxdy * colorPlane * colorPlane;

                vignetteFactorCT[ix + iy * wI + 0] += (1 - dx - dy + dxdy) * colorImage * colorPlane;
                vignetteFactorCT[ix + iy * wI + 1] += (dx - dxdy) * colorImage * colorPlane;
                vignetteFactorCT[ix + iy * wI + wI] += (dy - dxdy) * colorImage * colorPlane;
                vignetteFactorCT[ix + iy * wI + 1 + wI] += dxdy * colorImage * colorPlane;

                if (cvIsNaN(fac) == 1) continue;
                E += residual;
                R++;
            }
        }

        float maxFac = 0;
        for (int pi = 0; pi < hI * wI; pi++)        // for all plane points
        {
            if (vignetteFactorTT[pi] < 1)
            {
                vignetteFactor[pi] = NAN;
            }
            else
            {
                vignetteFactor[pi] = vignetteFactorCT[pi] / vignetteFactorTT[pi];
                if (vignetteFactor[pi] > maxFac) maxFac = vignetteFactor[pi];
            }
        }

        std::cout << R << " residual terms => " << sqrt(E / R) << std::endl;

        // normalize to vignette max. factor 1.
        for (int pi = 0; pi < hI * wI; pi++)
        {
            vignetteFactor[pi] /= maxFac;
        }

        logFile << it << " " << n << " " << R << " " << sqrt(E / R) << "\n";

        // dilate & smoothe vignette by 4 pixel for output.
        // does not change anything in the optimization; uses vignetteFactorTT and vignetteFactorCT for temporary storing
        memcpy(vignetteFactorTT, vignetteFactor, sizeof(float) * hI * wI);
        for (int dilit = 0; dilit < 4; dilit++)
        {
            memcpy(vignetteFactorCT, vignetteFactorTT, sizeof(float) * hI * wI);
            for (int y = 0; y < hI; y++)
            {
                for (int x = 0; x < wI; x++)
                {
                    int idx = x + y * wI;
                    {
                        float sum = 0, num = 0;
                        if (x < wI - 1 && y < hI - 1 && cvIsNaN(vignetteFactorCT[idx + 1 + wI]) != 1)
                        {
                            sum += vignetteFactorCT[idx + 1 + wI];
                            num++;
                        }
                        if (x < wI - 1 && cvIsNaN(vignetteFactorCT[idx + 1]) != 1)
                        {
                            sum += vignetteFactorCT[idx + 1];
                            num++;
                        }
                        if (x < wI - 1 && y > 0 && cvIsNaN(vignetteFactorCT[idx + 1 - wI]) != 1)
                        {
                            sum += vignetteFactorCT[idx + 1 - wI];
                            num++;
                        }

                        if (y < hI - 1 && cvIsNaN(vignetteFactorCT[idx + wI]) != 1)
                        {
                            sum += vignetteFactorCT[idx + wI];
                            num++;
                        }
                        if (cvIsNaN(vignetteFactorCT[idx]) != 1)
                        {
                            sum += vignetteFactorCT[idx];
                            num++;
                        }
                        if (y > 0 && cvIsNaN(vignetteFactorCT[idx - wI]) != 1)
                        {
                            sum += vignetteFactorCT[idx - wI];
                            num++;
                        }

                        if (y < hI - 1 && x > 0 && cvIsNaN(vignetteFactorCT[idx - 1 + wI]) != 1)
                        {
                            sum += vignetteFactorCT[idx - 1 + wI];
                            num++;
                        }
                        if (x > 0 && cvIsNaN(vignetteFactorCT[idx - 1]) != 1)
                        {
                            sum += vignetteFactorCT[idx - 1];
                            num++;
                        }
                        if (y > 0 && x > 0 && cvIsNaN(vignetteFactorCT[idx - 1 - wI]) != 1)
                        {
                            sum += vignetteFactorCT[idx - 1 - wI];
                            num++;
                        }

                        if (num > 0) vignetteFactorTT[idx] = sum / num;
                    }
                }
            }
        }

        // ================================ Store Vignette Image =======================================
        if (debug)
        {
            displayImageV(vignetteFactorTT, wI, hI, "VignetteSmoothed");
        }
        Mat wrapSmoothed = Mat(hI, wI, CV_32F, vignetteFactorTT) * 254.9 * 254.9;
        Mat wrapSmoothed16;
        wrapSmoothed.convertTo(wrapSmoothed16, CV_16U, 1, 0);
        imwrite("vignetteCalibResult/vignetteSmoothed.png", wrapSmoothed16);
        waitKey(50);

        if (debug)
        {
            displayImageV(vignetteFactor, wI, hI, "VignetteOrg");
        }
        Mat wrap = Mat(hI, wI, CV_32F, vignetteFactor) * 254.9 * 254.9;
        Mat wrap16;
        wrap.convertTo(wrap16, CV_16U, 1, 0);
        imwrite("vignetteCalibResult/vignette.png", wrap16);
        waitKey(50);
    }

    logFile.flush();
    logFile.close();

    delete[] planeColor;
    delete[] planeColorFF;
    delete[] planeColorFC;
    delete[] vignetteFactor;
    delete[] vignetteFactorTT;
    delete[] vignetteFactorCT;

    for (unsigned long i = 0; i < n; i++)
    {
        delete[] images[i];
        delete[] p2imgX[i];
        delete[] p2imgY[i];
    }
}


VignetteCalib::~VignetteCalib()
{
    delete imageReader;
    delete gammaRemover;
}

} // namespace photometric_calib
} // namespace cv