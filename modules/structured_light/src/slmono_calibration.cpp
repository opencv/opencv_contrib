#include "precomp.hpp"
#include <iostream>

#include <opencv2/structured_light/slmono_calibration.hpp>

namespace cv{
namespace structured_light{

//Deafault setting for calibration
Settings::Settings(){
    patternType = CHESSBOARD;
    patternSize = Size(13, 9);
    subpixelSize = Size(11, 11);
    squareSize = 50;
    nbrOfFrames = 25;
}

void loadSettings( String path, Settings &sttngs )
{
    FileStorage fsInput(path, FileStorage::READ);

    fsInput["PatternWidth"] >> sttngs.patternSize.width;
    fsInput["PatternHeight"] >> sttngs.patternSize.height;
    fsInput["SubPixelWidth"] >> sttngs.subpixelSize.width;
    fsInput["SubPixelHeight"] >> sttngs.subpixelSize.height;
    fsInput["SquareSize"] >> sttngs.squareSize;
    fsInput["NbrOfFrames"] >> sttngs.nbrOfFrames;
    fsInput["PatternType"] >> sttngs.patternType;
    fsInput.release();
}

double calibrate(InputArrayOfArrays objPoints, InputArrayOfArrays imgPoints,
               InputOutputArray cameraMatrix, InputOutputArray distCoeffs, OutputArrayOfArrays r, OutputArrayOfArrays t, Size imgSize )
{
    int calibFlags = 0;

    double rms = calibrateCamera(objPoints, imgPoints, imgSize, cameraMatrix,
                                distCoeffs, r, t, calibFlags);

    return rms;
}

void createObjectPoints(InputArrayOfArrays patternCorners, Size patternSize, float squareSize, int patternType)
{
    std::vector<Point3f>& patternCorners_ = *( std::vector<Point3f>* ) patternCorners.getObj();
    switch( patternType )
    {
        case CHESSBOARD:
        case CIRCLES_GRID:
            for( int i = 0; i < patternSize.height; ++i )
            {
                for( int j = 0; j < patternSize.width; ++j )
                {
                    patternCorners_.push_back(Point3f(float(i*squareSize), float(j*squareSize), 0));
                }
            }
            break;
        case ASYMETRIC_CIRCLES_GRID:
            break;
    }
}

void createProjectorObjectPoints(InputArrayOfArrays patternCorners, Size patternSize, float squareSize,
                        int patternType )
{
    std::vector<Point2f>& patternCorners_ = *( std::vector<Point2f>* ) patternCorners.getObj();

    switch( patternType )
    {
        case CHESSBOARD:
        case CIRCLES_GRID:
            for( int i = 1; i <= patternSize.height; ++i )
            {
                for( int j = 1; j <= patternSize.width; ++j )
                {
                    patternCorners_.push_back(Point2f(float(j*squareSize), float(i*squareSize)));
                }
            }
            break;
        case ASYMETRIC_CIRCLES_GRID:
            break;
    }
}

void fromCamToWorld(InputArray cameraMatrix, InputArrayOfArrays rV, InputArrayOfArrays tV,
                    InputArrayOfArrays imgPoints, OutputArrayOfArrays worldPoints)
{
    std::vector<std::vector<Point2f>>& imgPoints_ = *( std::vector< std::vector<Point2f> >* ) imgPoints.getObj();
    std::vector<std::vector<Point3f>>& worldPoints_ = *( std::vector< std::vector<Point3f> >* ) worldPoints.getObj();
    std::vector<Mat>& rV_ = *( std::vector<Mat>* ) rV.getObj();
    std::vector<Mat>& tV_ = *( std::vector<Mat>* ) tV.getObj();

    int s = (int) rV_.size();
    Mat invK64, invK;
    //invK64 = cameraMatrix.inv();
    invert(cameraMatrix, invK64);
    invK64.convertTo(invK, CV_32F);

    for(int i = 0; i < s; ++i)
    {
        Mat r, t, rMat;
        rV_[i].convertTo(r, CV_32F);
        tV_[i].convertTo(t, CV_32F);

        Rodrigues(r, rMat);
        Mat transPlaneToCam = rMat.inv()*t;

        vector<Point3f> wpTemp;
        int s2 = (int) imgPoints_[i].size();
        for(int j = 0; j < s2; ++j){
            Mat coords(3, 1, CV_32F);
            coords.at<float>(0, 0) = imgPoints_[i][j].x;
            coords.at<float>(1, 0) = imgPoints_[i][j].y;
            coords.at<float>(2, 0) = 1.0f;

            Mat worldPtCam = invK*coords;
            Mat worldPtPlane = rMat.inv()*worldPtCam;

            float scale = transPlaneToCam.at<float>(2)/worldPtPlane.at<float>(2);
            Mat worldPtPlaneReproject = scale*worldPtPlane - transPlaneToCam;

            Point3f pt;
            pt.x = worldPtPlaneReproject.at<float>(0);
            pt.y = worldPtPlaneReproject.at<float>(1);
            pt.z = 0;
            wpTemp.push_back(pt);
        }
        worldPoints_.push_back(wpTemp);
    }
}

void saveCalibrationResults( String path, InputArray camK, InputArray camDistCoeffs, InputArray projK, InputArray projDistCoeffs,
                      InputArray fundamental )
{
    Mat& camK_ = *(Mat*) camK.getObj();
    Mat& camDistCoeffs_ = *(Mat*) camDistCoeffs.getObj();
    Mat& projK_ = *(Mat*) projK.getObj();
    Mat& projDistCoeffs_ = *(Mat*) projDistCoeffs.getObj();
    Mat& fundamental_ = *(Mat*) fundamental.getObj();

    FileStorage fs(path + ".yml", FileStorage::WRITE);
    fs << "camIntrinsics" << camK_;
    fs << "camDistCoeffs" << camDistCoeffs_;
    fs << "projIntrinsics" << projK_;
    fs << "projDistCoeffs" << projDistCoeffs_;
    fs << "fundamental" << fundamental_;
    fs.release();
}

void saveCalibrationData(String path, InputArrayOfArrays T1, InputArrayOfArrays T2, InputArrayOfArrays ptsProjCam,
                        InputArrayOfArrays ptsProjProj, InputArrayOfArrays ptsProjCamN, InputArrayOfArrays ptsProjProjN)
{
    std::vector<Mat>& T1_ = *( std::vector<Mat>* ) T1.getObj();
    std::vector<Mat>& T2_ = *( std::vector<Mat>* ) T2.getObj();
    std::vector<Mat>& ptsProjCam_ = *( std::vector<Mat>* ) ptsProjCam.getObj();
    std::vector<Mat>& ptsProjProj_ = *( std::vector<Mat>* ) ptsProjProj.getObj();
    std::vector<Mat>& ptsProjCamN_ = *( std::vector<Mat>* ) ptsProjCamN.getObj();
    std::vector<Mat>& ptsProjProjN_ = *( std::vector<Mat>* ) ptsProjProjN.getObj();

    FileStorage fs(path + ".yml", FileStorage::WRITE);

    int size = (int) T1_.size();
    fs << "size" << size;
    for( int i = 0; i < (int)T1_.size(); ++i )
    {
        ostringstream nbr;
        nbr << i;
        fs << "TprojCam" + nbr.str() << T1_[i];
        fs << "TProjProj" + nbr.str() << T2_[i];
        fs << "ptsProjCam" + nbr.str() << ptsProjCam_[i];
        fs << "ptsProjProj" + nbr.str() << ptsProjProj_[i];
        fs << "ptsProjCamN" + nbr.str() << ptsProjCamN_[i];
        fs << "ptsProjProjN" + nbr.str() << ptsProjProjN_[i];
    }
    fs.release();

}

void loadCalibrationData(string filename, OutputArray cameraIntrinsic, OutputArray projectorIntrinsic,
                        OutputArray cameraDistortion, OutputArray projectorDistortion, OutputArray rotation, OutputArray translation)
{
    Mat& cameraIntrinsic_ = *(Mat*) cameraIntrinsic.getObj();
    Mat& projectorIntrinsic_ = *(Mat*) projectorIntrinsic.getObj();
    Mat& cameraDistortion_ = *(Mat*) cameraDistortion.getObj();
    Mat& projectorDistortion_ = *(Mat*) projectorDistortion.getObj();
    Mat& rotation_ = *(Mat*) rotation.getObj();
    Mat& translation_ = *(Mat*) translation.getObj();

    FileStorage fs;
    fs.open(filename, FileStorage::READ);

    if (!fs.isOpened())
    {
        cerr << "Failed to open " << filename << endl;
    }

    fs["cam_int"] >> cameraIntrinsic_;
    fs["cam_dist"] >> cameraDistortion_;
    fs["proj_int"] >> projectorIntrinsic_;
    fs["proj_dist"] >> projectorDistortion_;
    fs["roration"] >> rotation_;
    fs["translation"] >> translation_;

    fs.release();

}

void normalize( InputArray pts, const int& dim, InputOutputArray normpts, OutputArray T)
{
    Mat& pts_ = *(Mat*) pts.getObj();
    Mat& normpts_ = *(Mat*) normpts.getObj();
    Mat& T_ = *(Mat*) T.getObj();

    float averagedist = 0;
    float scale = 0;

    //centroid

    Mat centroid(dim,1,CV_32F);
    Scalar tmp;

    if( normpts_.empty() )
    {
        normpts_ = Mat(pts_.rows,pts_.cols,CV_32F);
    }

    for( int i = 0 ; i < dim ; ++i )
    {
        tmp = mean(pts_.row(i));
        centroid.at<float>(i,0) = (float)tmp[0];
        subtract(pts_.row(i), centroid.at<float>(i, 0), normpts_.row(i));
    }

    //average distance

    Mat ptstmp;
    for( int i = 0 ; i < normpts_.cols; ++i )
    {
        ptstmp = normpts_.col(i);
        averagedist = averagedist+(float)norm(ptstmp);
    }
    averagedist = averagedist / normpts_.cols;
    scale = (float)(sqrt(static_cast<float>(dim)) / averagedist);

    normpts_ = normpts_ * scale;

    T_=cv::Mat::eye(dim+1,dim+1,CV_32F);
    for( int i = 0; i < dim; ++i )
    {
        T_.at<float>(i, i) = scale;
        T_.at<float>(i, dim) = -scale*centroid.at<float>(i, 0);
    }
}

void fromVectorToMat(InputArrayOfArrays v, OutputArray pts)
{
    Mat& pts_ = *(Mat*) pts.getObj();
    std::vector<Point2f>& v_ = *( std::vector<Point2f>* ) v.getObj();

    int nbrOfPoints = (int) v_.size();

    if( pts_.empty() )
        pts_.create(2, nbrOfPoints, CV_32F);

    for( int i = 0; i < nbrOfPoints; ++i )
    {
        pts_.at<float>(0, i) = v_[i].x;
        pts_.at<float>(1, i) = v_[i].y;
    }
}

void fromMatToVector(InputArray pts,  OutputArrayOfArrays v)
{
    Mat& pts_ = *(Mat*) pts.getObj();
    std::vector<Point2f>& v_ = *(std::vector<Point2f>* ) v.getObj();

    int nbrOfPoints = pts_.cols;

    for( int i = 0; i < nbrOfPoints; ++i )
    {
        Point2f temp;
        temp.x = pts_.at<float>(0, i);
        temp.y = pts_.at<float>(1, i);
        v_.push_back(temp);
    }
}

Point2f back(Point2f point, double fx, double fy, double ux, double uy)
{
    double x = point.x * fx + ux;
    double y = point.y * fy + uy;

    return Point2f((float)x, (float)y);
}

void distortImage(InputArray input, InputArray camMat, InputArray distCoef, OutputArray output)
{
    Mat& camMat_ = *(Mat*) camMat.getObj();
    Mat& input_ = *(Mat*) input.getObj();

    double fx = camMat_.at<double>(0,0);
    double fy = camMat_.at<double>(1,1);
    double ux = camMat_.at<double>(0,2);
    double uy = camMat_.at<double>(1,2);

    vector<Point2f> undistortedPoints, distortedPoints;
    for (int i = 0; i < input_.rows; i++)
    {
        for (int j = 0; j < input_.cols; j++)
        {
            undistortedPoints.push_back(Point2f((float)i, (float)j));
        }
    }
    undistortPoints(undistortedPoints, distortedPoints, camMat, distCoef, Mat(), Mat());

    Mat mapx(cv::Size(input_.rows, input_.cols), CV_32FC1);
    Mat mapy(cv::Size(input_.rows, input_.cols), CV_32FC1);

    for (int i = 0; i < input_.rows; i++)
    {
        for (int j = 0; j < input_.cols; j++)
        {
            distortedPoints[i*input_.cols+j] = back(distortedPoints[i*input_.cols+j], fx, fy, ux, uy);
            mapx.at<float>(j, i) = distortedPoints[i*input_.cols+j].x;
            mapy.at<float>(j, i) = distortedPoints[i*input_.cols+j].y;
        }
    }

    remap(input, output, mapx, mapy, INTER_CUBIC);
}

}
}