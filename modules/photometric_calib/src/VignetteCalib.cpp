// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/photometric_calib/VignetteCalib.hpp"

#include <fstream>
#include <iostream>
#include <algorithm>

namespace cv { namespace photometric_calib{


VignetteCalib::VignetteCalib(std::string folderPath, std::string timePath, std::string cameraFile, std::string gammaFile) :
        _imageSkip(1), _maxIterations(20), _outlierTh(15), _gridWidth(1000), _gridHeight(1000), _facW(5), _facH(5),
        _maxAbsGrad(255)
{
    imageReader = new Reader(folderPath, "png", timePath);
    // check the extension of the camera file.
    CV_Assert(cameraFile.substr(cameraFile.find_last_of(".") + 1) == "yaml" || cameraFile.substr(cameraFile.find_last_of(".") + 1) == "yml");
    FileStorage cameraStorage(cameraFile, FileStorage::READ);
    cameraStorage["cameraMatrix"] >> _cameraMatrix;
    cameraStorage["distCoeffs"] >> _distCoeffs;
    gammaRemover = new GammaRemover(gammaFile, imageReader->getWidth(), imageReader->getHeight());
}

VignetteCalib::VignetteCalib(std::string folderPath, std::string timePath, std::string cameraFile, std::string gammaFile, int imageSkip, int maxIterations,
                             int outlierTh, int gridWidth, int gridHeight, float facW, float facH, int maxAbsGrad) :
        _imageSkip(imageSkip), _maxIterations(maxIterations), _outlierTh(outlierTh), _gridWidth(gridWidth), _gridHeight(gridHeight),
        _facW(facW), _facH(facH), _maxAbsGrad(maxAbsGrad)
{
    imageReader = new Reader(folderPath, "png", timePath);
    // check the extension of the camera file.
    CV_Assert(cameraFile.substr(cameraFile.find_last_of(".") + 1) == "yaml" || cameraFile.substr(cameraFile.find_last_of(".") + 1) == "yml");
    FileStorage cameraStorage(cameraFile, FileStorage::READ);
    cameraStorage["cameraMatrix"] >> _cameraMatrix;
    cameraStorage["distCoeffs"] >> _distCoeffs;
    gammaRemover = new GammaRemover(gammaFile, imageReader->getWidth(), imageReader->getHeight());
}

float VignetteCalib::getInterpolatedElement(const float *const mat, const float x, const float y, const int width)
{
    int ix = (int)x;
    int iy = (int)y;
    float dx = x - ix;
    float dy = y - iy;
    float dxdy = dx*dy;
    const float* bp = mat +ix+iy*width;

    float res =   dxdy * bp[1+width]
                  + (dy-dxdy) * bp[width]
                  + (dx-dxdy) * bp[1]
                  + (1-dx-dy+dxdy) * bp[0];

    return res;
}

void VignetteCalib::displayImage(float *I, int w, int h, std::string name)
{
    float vmin=1e10;
    float vmax=-1e10;

    for(int i=0;i<w*h;i++)
    {
        if(vmin > I[i]) vmin = I[i];
        if(vmax < I[i]) vmax = I[i];
    }

    Mat img = Mat(h,w,CV_8UC3);

    for(int i=0;i<w*h;i++)
    {
        //isnanf? isnan?
        if(cvIsNaN(I[i]) == 1) img.at<Vec3b>(i) = Vec3b(0,0,255);
        else img.at<Vec3b>(i) = Vec3b(255*(I[i]-vmin) / (vmax-vmin),255*(I[i]-vmin) / (vmax-vmin),255*(I[i]-vmin) / (vmax-vmin));
    }

    printf("plane image values %f - %f!\n", vmin, vmax);
    imshow(name, img);
    imwrite("vignetteCalibResult/plane.png", img);
}

void VignetteCalib::displayImageV(float *I, int w, int h, std::string name)
{
    Mat img = Mat(h,w,CV_8UC3);
    for(int i=0;i<w*h;i++)
    {
        if(cvIsNaN(I[i]) == 1)
            img.at<Vec3b>(i) = Vec3b(0,0,255);
        else
        {
            float c = 254*I[i];
            img.at<Vec3b>(i) = Vec3b(c,c,c);
        }

    }
    imshow(name, img);
}

void VignetteCalib::calib()
{
    if(-1 == system("rm -rf vignetteCalibResult")) printf("could not delete old vignetteCalibResult folder!\n");
    if(-1 == system("mkdir vignetteCalibResult")) printf("could not delete old vignetteCalibResult folder!\n");

    // affine map from plane cordinates to grid coordinates.
    Matx33f K_p2idx = Matx33f::eye();
    Mat1f test = Mat1f::eye(3,3);
    K_p2idx(0,0) = _gridWidth / _facW;
    K_p2idx(1,1) = _gridHeight / _facH;
    K_p2idx(0,2) = _gridWidth / 2;
    K_p2idx(1,2) = _gridHeight / 2;
    Matx33f K_p2idx_inverse = K_p2idx.inv();

    int wO, hO;
    wO = imageReader->getWidth();
    hO = imageReader->getHeight();
    int wI = wO, hI = hO;

    Ptr<aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

    std::vector<float*> images;
    std::vector<float*> p2imgX;
    std::vector<float*> p2imgY;

    float meanExposure = 0.f;
    for(size_t i=0;i<imageReader->getNumImages();i+=_imageSkip)
        meanExposure+=imageReader->getExposureDuration(i);
    meanExposure = meanExposure/imageReader->getNumImages();

    if(meanExposure==0) meanExposure = 1;

    for(size_t i=0; i < imageReader->getNumImages(); ++i)
    {
        std::vector<int> markerIds;
        std::vector< std::vector<Point2f> > markerCorners, rejectedCandidates;

        Mat oriImg = imageReader->getImage(i);
        Mat undisImg;
        undistort(oriImg, undisImg, _cameraMatrix, _distCoeffs);

        aruco::detectMarkers(undisImg, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

        if(markerCorners.size() != 1) continue;

        std::vector<Point2f> ptsP;
        std::vector<Point2f> ptsI;
        ptsI.push_back(Point2f(markerCorners[0][0].x, markerCorners[0][0].y));
        ptsI.push_back(Point2f(markerCorners[0][1].x, markerCorners[0][1].y));
        ptsI.push_back(Point2f(markerCorners[0][2].x, markerCorners[0][2].y));
        ptsI.push_back(Point2f(markerCorners[0][3].x, markerCorners[0][3].y));
        ptsP.push_back(Point2f(-0.5,0.5));
        ptsP.push_back(Point2f(0.5,0.5));
        ptsP.push_back(Point2f(0.5,-0.5));
        ptsP.push_back(Point2f(-0.5,-0.5));

        Mat Hcv = findHomography(ptsP, ptsI);
//        Eigen::Matrix3f H;
//        H(0,0) = Hcv.at<double>(0,0);
//        H(0,1) = Hcv.at<double>(0,1);
//        H(0,2) = Hcv.at<double>(0,2);
//        H(1,0) = Hcv.at<double>(1,0);
//        H(1,1) = Hcv.at<double>(1,1);
//        H(1,2) = Hcv.at<double>(1,2);
//        H(2,0) = Hcv.at<double>(2,0);
//        H(2,1) = Hcv.at<double>(2,1);
//        H(2,2) = Hcv.at<double>(2,2);
        Matx33f H(Hcv);

        std::vector<float> imgRaw(wI * hI);
        gammaRemover->getUnGammaImageVec(imageReader->getImage(i), imgRaw);

        float* plane2imgX = new float[_gridWidth*_gridHeight];
        float* plane2imgY = new float[_gridWidth*_gridHeight];

        Matx33f HK = H * K_p2idx_inverse;

        int idx=0;
        for(int y=0;y<_gridHeight;y++)
            for(int x=0;x<_gridWidth;x++)
            {
                //Eigen::Vector3f pp = HK*Eigen::Vector3f(x,y,1);
                Vec3f pp = HK * Vec3f(x,y,1);
                plane2imgX[idx] = pp[0] / pp[2];
                plane2imgY[idx] = pp[1] / pp[2];
                idx++;
            }

        //reader->getUndistorter()->distortCoordinates(plane2imgX, plane2imgY, gw*gh);

        float expDuration;
        expDuration = (imageReader->getExposureDuration(i) == 0 ? 1 : imageReader->getExposureDuration(i));
        float* image = new float[wI*hI];
        for(int y=0; y<hI;y++)
            for(int x=0; x<wI;x++)
                image[x+y*wI] = meanExposure*imgRaw[x+y*wI] / expDuration;

        for(int y=2; y<hI-2;y++)
            for(int x=2; x<wI-2;x++)
            {
                for(int deltax=-2; deltax<3;deltax++)
                    for(int deltay=-2; deltay<3;deltay++)
                    {
                        if(fabsf(image[x+y*wI] - image[x+deltax+(y+deltay)*wI]) > _maxAbsGrad) { image[x+y*wI] = NAN; image[x+deltax+(y+deltay)*wI]=NAN; }
                    }
            }

        images.push_back(image);

        // debug-plot.
        Mat dbgImg(hI, wI, CV_8UC3);
        for(int j=0;j<wI*hI;j++)
            dbgImg.at<Vec3b>(j) = Vec3b(imgRaw[j], imgRaw[j], imgRaw[j]);

        for(int x=0; x<=_gridWidth;x+=200)
            for(int y=0; y<=_gridHeight;y+=10)
            {
                int idxS = (x<_gridWidth ? x : _gridWidth-1)+(y<_gridHeight ? y : _gridHeight-1)*_gridWidth;
                int idxT = (x<_gridWidth ? x : _gridWidth-1)+((y+10)<_gridHeight ? (y+10) : _gridHeight-1)*_gridWidth;

                int u_dS = plane2imgX[idxS]+0.5;
                int v_dS = plane2imgY[idxS]+0.5;

                int u_dT = plane2imgX[idxT]+0.5;
                int v_dT = plane2imgY[idxT]+0.5;

                if(u_dS>=0 && v_dS >=0 && u_dS<wI && v_dS<hI && u_dT>=0 && v_dT >=0 && u_dT<wI && v_dT<hI)
                    line(dbgImg, Point(u_dS, v_dS), Point(u_dT, v_dT), Scalar(0,0,255), 10, LINE_AA);
                    //line(dbgImg, Point(u_dS, v_dS), Point(u_dT, v_dT), Scalar(0,0,255), 10, CV_AA);
            }

        for(int x=0; x<=_gridWidth;x+=10)
            for(int y=0; y<=_gridHeight;y+=200)
            {
                int idxS = (x<_gridWidth ? x : _gridWidth-1)+(y<_gridHeight ? y : _gridHeight-1)*_gridWidth;
                int idxT = ((x+10)<_gridWidth ? (x+10) : _gridWidth-1)+(y<_gridHeight ? y : _gridHeight-1)*_gridWidth;

                int u_dS = plane2imgX[idxS]+0.5;
                int v_dS = plane2imgY[idxS]+0.5;

                int u_dT = plane2imgX[idxT]+0.5;
                int v_dT = plane2imgY[idxT]+0.5;

                if(u_dS>=0 && v_dS >=0 && u_dS<wI && v_dS<hI && u_dT>=0 && v_dT >=0 && u_dT<wI && v_dT<hI)
                    line(dbgImg, Point(u_dS, v_dS), Point(u_dT, v_dT), Scalar(0,0,255), 10, LINE_AA);
            }

        for(int x=0; x<_gridWidth;x++)
            for(int y=0; y<_gridHeight;y++)
            {
                int u_d = plane2imgX[x+y*_gridWidth]+0.5;
                int v_d = plane2imgY[x+y*_gridWidth]+0.5;

                if(!(u_d>1 && v_d >1 && u_d<wI-2 && v_d<hI-2))
                {
                    plane2imgX[x+y*_gridWidth] = NAN;
                    plane2imgY[x+y*_gridWidth] = NAN;
                }
            }

        imshow("inRaw",dbgImg);

        if(rand()%40==0)
        {
            char buf[1000];
            snprintf(buf,1000,"vignetteCalibResult/img%lu.png",i);
            imwrite(buf, dbgImg);
        }

        waitKey(1);

        p2imgX.push_back(plane2imgX);
        p2imgY.push_back(plane2imgY);

    }

    std::ofstream logFile;
    logFile.open("vignetteCalibResult/log.txt", std::ios::trunc | std::ios::out);
    logFile.precision(15);

    int n = imageReader->getNumImages();

    float* planeColor = new float[_gridWidth*_gridHeight];
    float* planeColorFF = new float[_gridWidth*_gridHeight];
    float* planeColorFC = new float[_gridWidth*_gridHeight];
    float* vignetteFactor = new float[hI*wI];
    float* vignetteFactorTT = new float[hI*wI];
    float* vignetteFactorCT = new float[hI*wI];

    // initialize vignette factors to 1.
    for(int i=0;i<hI*wI;i++) vignetteFactor[i] = 1;

    double E=0;
    double R=0;

    for(int it=0;it<_maxIterations;it++)
    {
        int oth2 = _outlierTh*_outlierTh;
        if(it < _maxIterations/2) oth2=10000*10000;

        // ============================ optimize planeColor ================================
        memset(planeColorFF,0,_gridWidth*_gridHeight*sizeof(float));
        memset(planeColorFC,0,_gridWidth*_gridHeight*sizeof(float));
        E=0;R=0;

        // for each plane pixel, it's optimum is at sum(CF)/sum(FF)
        for(int img=0;img<n;img++)	// for all images
        {
            float* plane2imgX = p2imgX[img];
            float* plane2imgY = p2imgY[img];
            float* image = images[img];

            for(int pi=0;pi<_gridWidth*_gridHeight;pi++)		// for all plane points
            {
                if(cvIsNaN(plane2imgX[pi])) continue;

                // get vignetted color at that point, and add to build average.
                float color = getInterpolatedElement(image, plane2imgX[pi], plane2imgY[pi], wI);
                float fac = getInterpolatedElement(vignetteFactor, plane2imgX[pi], plane2imgY[pi], wI);

                if(cvIsNaN(fac)) continue;
                if(cvIsNaN(color)) continue;

                double residual = (double)((color - planeColor[pi]*fac)*(color - planeColor[pi]*fac));
                if(abs(residual) > oth2)
                {
                    E += oth2;
                    R ++;
                    continue;
                }


                planeColorFF[pi] += fac*fac;
                planeColorFC[pi] += color*fac;

                if(cvIsNaN(planeColor[pi])) continue;
                E += residual;
                R ++;
            }
        }

        for(int pi=0;pi<_gridWidth*_gridWidth;pi++)		// for all plane points
        {
            if(planeColorFF[pi] < 1)
                planeColor[pi]=NAN;
            else
                planeColor[pi] = planeColorFC[pi] / planeColorFF[pi];
        }
        displayImage(planeColor, _gridWidth, _gridWidth, "Plane");

        printf("%f residual terms => %f\n", R, sqrtf(E/R));

        // ================================ optimize vignette =======================================
        memset(vignetteFactorTT,0,hI*wI*sizeof(float));
        memset(vignetteFactorCT,0,hI*wI*sizeof(float));
        E=0;R=0;

        for(int img=0;img<n;img++)	// for all images
        {
            float* plane2imgX = p2imgX[img];
            float* plane2imgY = p2imgY[img];
            float* image = images[img];

            for(int pi=0;pi<_gridWidth*_gridWidth;pi++)		// for all plane points
            {
                if(cvIsNaN(plane2imgX[pi])) continue;
                float x = plane2imgX[pi];
                float y = plane2imgY[pi];

                float colorImage = getInterpolatedElement(image, x, y, wI);
                float fac = getInterpolatedElement(vignetteFactor, x, y, wI);
                float colorPlane = planeColor[pi];

                if(cvIsNaN(colorPlane)) continue;
                if(cvIsNaN(colorImage)) continue;

                double residual = (double)((colorImage - colorPlane*fac)*(colorImage - colorPlane*fac));
                if(abs(residual) > oth2)
                {
                    E += oth2;
                    R ++;
                    continue;
                }


                int ix = (int)x;
                int iy = (int)y;
                float dx = x - ix;
                float dy = y - iy;
                float dxdy = dx*dy;

                vignetteFactorTT[ix+iy*wI + 0] += (1-dx-dy+dxdy) * 	colorPlane*colorPlane;
                vignetteFactorTT[ix+iy*wI + 1] += (dx-dxdy) * 		colorPlane*colorPlane;
                vignetteFactorTT[ix+iy*wI + wI] += (dy-dxdy) * 		colorPlane*colorPlane;
                vignetteFactorTT[ix+iy*wI + 1+wI] += dxdy * 		colorPlane*colorPlane;

                vignetteFactorCT[ix+iy*wI + 0] += (1-dx-dy+dxdy) * 	colorImage*colorPlane;
                vignetteFactorCT[ix+iy*wI + 1] += (dx-dxdy) * 		colorImage*colorPlane;
                vignetteFactorCT[ix+iy*wI + wI] += (dy-dxdy) * 		colorImage*colorPlane;
                vignetteFactorCT[ix+iy*wI + 1+wI] += dxdy * 		colorImage*colorPlane;

                if(cvIsNaN(fac)) continue;
                E += residual;
                R ++;
            }
        }

        float maxFac=0;
        for(int pi=0;pi<hI*wI;pi++)		// for all plane points
        {
            if(vignetteFactorTT[pi] < 1)
                vignetteFactor[pi]=NAN;
            else
            {
                vignetteFactor[pi] = vignetteFactorCT[pi] / vignetteFactorTT[pi];
                if(vignetteFactor[pi]>maxFac) maxFac=vignetteFactor[pi];
            }
        }

        printf("%f residual terms => %f\n", R, sqrtf(E/R));

        // normalize to vignette max. factor 1.
        for(int pi=0;pi<hI*wI;pi++)
            vignetteFactor[pi] /= maxFac;



        logFile << it << " " << n << " " << R << " " << sqrtf(E/R) << "\n";






        // dilate & smoothe vignette by 4 pixel for output.
        // does not change anything in the optimization; uses vignetteFactorTT and vignetteFactorCT for temporary storing
        {
            memcpy(vignetteFactorTT, vignetteFactor, sizeof(float)*hI*wI);
            for(int dilit=0; dilit<4;dilit++)
            {
                memcpy(vignetteFactorCT, vignetteFactorTT, sizeof(float)*hI*wI);
                for(int y=0; y<hI;y++)
                    for(int x=0; x<wI;x++)
                    {
                        int idx = x+y*wI;
                        {
                            float sum=0, num=0;
                            if(x<wI-1 && y<hI-1 && !cvIsNaN(vignetteFactorCT[idx+1+wI])) {sum += vignetteFactorCT[idx+1+wI]; num++;}
                            if(x<wI-1 &&           !cvIsNaN(vignetteFactorCT[idx+1])) {sum += vignetteFactorCT[idx+1]; num++;}
                            if(x<wI-1 && y>0 &&    !cvIsNaN(vignetteFactorCT[idx+1-wI])) {sum += vignetteFactorCT[idx+1-wI]; num++;}

                            if(y<hI-1 &&           !cvIsNaN(vignetteFactorCT[idx+wI])) {sum += vignetteFactorCT[idx+wI]; num++;}
                            if(			           !cvIsNaN(vignetteFactorCT[idx])) {sum += vignetteFactorCT[idx]; num++;}
                            if(y>0 &&              !cvIsNaN(vignetteFactorCT[idx-wI])) {sum += vignetteFactorCT[idx-wI]; num++;}

                            if(y<hI-1 && x>0 &&    !cvIsNaN(vignetteFactorCT[idx-1+wI])) {sum += vignetteFactorCT[idx-1+wI]; num++;}
                            if(x>0 &&              !cvIsNaN(vignetteFactorCT[idx-1])) {sum += vignetteFactorCT[idx-1]; num++;}
                            if(y>0 && x>0 &&       !cvIsNaN(vignetteFactorCT[idx-1-wI])) {sum += vignetteFactorCT[idx-1-wI]; num++;}

                            if(num>0) vignetteFactorTT[idx] = sum/num;
                        }
                    }
            }

            {
                displayImageV(vignetteFactorTT, wI, hI, "VignetteSmoothed");
                Mat wrap = Mat(hI, wI, CV_32F, vignetteFactorTT)*254.9*254.9;
                Mat wrap16;
                wrap.convertTo(wrap16, CV_16U,1,0);
                imwrite("vignetteCalibResult/vignetteSmoothed.png", wrap16);
                waitKey(50);
            }
            {
                displayImageV(vignetteFactor, wI, hI, "VignetteOrg");
                Mat wrap = Mat(hI, wI, CV_32F, vignetteFactor)*254.9*254.9;
                Mat wrap16;
                wrap.convertTo(wrap16, CV_16U,1,0);
                imwrite("vignetteCalibResult/vignette.png", wrap16);
                waitKey(50);
            }
        }
    }

    logFile.flush();
    logFile.close();

    delete[] planeColor;
    delete[] planeColorFF;
    delete[] planeColorFC;
    delete[] vignetteFactor;
    delete[] vignetteFactorTT;
    delete[] vignetteFactorCT;

    for(int i=0;i<n;i++)
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
}}// namespace photometric_calib, cv