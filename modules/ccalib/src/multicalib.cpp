/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2015, Baisheng Lai (laibaisheng@gmail.com), Zhejiang University,
// all rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/**
 * This module was accepted as a GSoC 2015 project for OpenCV, authored by
 * Baisheng Lai, mentored by Bo Li.
 *
 * The omnidirectional camera in this module is denoted by the catadioptric
 * model. Please refer to Mei's paper for details of the camera model:
 *
 *      C. Mei and P. Rives, "Single view point omnidirectional camera
 *      calibration from planar grids", in ICRA 2007.
 *
 * The implementation of the calibration part is based on Li's calibration
 * toolbox:
 *
 *     B. Li, L. Heng, K. Kevin  and M. Pollefeys, "A Multiple-Camera System
 *     Calibration Toolbox Using A Feature Descriptor-Based Calibration
 *     Pattern", in IROS 2013.
 */

#include "precomp.hpp"
#include "opencv2/ccalib/multicalib.hpp"
#include "opencv2/core.hpp"
#include <string>
#include <vector>
#include <queue>
#include <iostream>

namespace cv { namespace multicalib {

MultiCameraCalibration::MultiCameraCalibration(int cameraType, int nCameras, const std::string& fileName,
    float patternWidth, float patternHeight, int verbose, int showExtration, int nMiniMatches, int flags, TermCriteria criteria,
    Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> descriptor,
    Ptr<DescriptorMatcher> matcher)
{
    _camType = cameraType;
    _nCamera = nCameras;
    _flags = flags;
    _nMiniMatches = nMiniMatches;
    _filename = fileName;
    _patternWidth = patternWidth;
    _patternHeight = patternHeight;
    _criteria = criteria;
    _showExtraction = showExtration;
    _objectPointsForEachCamera.resize(_nCamera);
    _imagePointsForEachCamera.resize(_nCamera);
    _cameraMatrix.resize(_nCamera);
    _distortCoeffs.resize(_nCamera);
    _xi.resize(_nCamera);
    _omEachCamera.resize(_nCamera);
    _tEachCamera.resize(_nCamera);
    _detector = detector;
    _descriptor = descriptor;
    _matcher = matcher;
	_verbose = verbose;
    for (int i = 0; i < _nCamera; ++i)
    {
        _vertexList.push_back(vertex());
    }
}

double MultiCameraCalibration::run()
{
    loadImages();
    initialize();
    double error = optimizeExtrinsics();
    return error;
}

std::vector<std::string> MultiCameraCalibration::readStringList()
{
    std::vector<std::string> l;
    l.resize(0);
    FileStorage fs(_filename, FileStorage::READ);

    FileNode n = fs.getFirstTopLevelNode();

    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it )
        l.push_back((std::string)*it);

    return l;
}

void MultiCameraCalibration::loadImages()
{
    std::vector<std::string> file_list;
    file_list = readStringList();

    Ptr<FeatureDetector> detector = _detector;
    Ptr<DescriptorExtractor> descriptor = _descriptor;
    Ptr<DescriptorMatcher> matcher = _matcher;

    randpattern::RandomPatternCornerFinder finder(_patternWidth, _patternHeight, _nMiniMatches, CV_32F, _verbose, this->_showExtraction, detector, descriptor, matcher);
    Mat pattern = cv::imread(file_list[0]);
    finder.loadPattern(pattern);

    std::vector<std::vector<std::string> > filesEachCameraFull(_nCamera);
    std::vector<std::vector<int> > timestampFull(_nCamera);
	std::vector<std::vector<int> > timestampAvailable(_nCamera);

    for (int i = 1; i < (int)file_list.size(); ++i)
    {
        int cameraVertex, timestamp;
        std::string filename = file_list[i].substr(0, file_list[i].find('.'));
        size_t spritPosition1 = filename.rfind('/');
        size_t spritPosition2 = filename.rfind('\\');
        if (spritPosition1!=std::string::npos)
        {
            filename = filename.substr(spritPosition1+1, filename.size() - 1);
        }
        else if(spritPosition2!= std::string::npos)
        {
            filename = filename.substr(spritPosition2+1, filename.size() - 1);
        }
        sscanf(filename.c_str(), "%d-%d", &cameraVertex, &timestamp);
        filesEachCameraFull[cameraVertex].push_back(file_list[i]);
        timestampFull[cameraVertex].push_back(timestamp);
    }


    // calibrate each camera individually
    for (int camera = 0; camera < _nCamera; ++camera)
    {
        Mat image, cameraMatrix, distortCoeffs;

        // find image and object points
        for (int imgIdx = 0; imgIdx < (int)filesEachCameraFull[camera].size(); ++imgIdx)
        {
            image = imread(filesEachCameraFull[camera][imgIdx], IMREAD_GRAYSCALE);
			if (!image.empty() && _verbose)
			{
				std::cout << "open image " << filesEachCameraFull[camera][imgIdx] << " successfully" << std::endl;
			}
			else if (image.empty() && _verbose)
			{
				std::cout << "open image" << filesEachCameraFull[camera][imgIdx] << " failed" << std::endl;
			}
            std::vector<Mat> imgObj = finder.computeObjectImagePointsForSingle(image);
			if ((int)imgObj[0].total() > _nMiniMatches)
			{
				_imagePointsForEachCamera[camera].push_back(imgObj[0]);
				_objectPointsForEachCamera[camera].push_back(imgObj[1]);
				timestampAvailable[camera].push_back(timestampFull[camera][imgIdx]);
			}
			else if ((int)imgObj[0].total() <= _nMiniMatches && _verbose)
			{
				std::cout << "image " << filesEachCameraFull[camera][imgIdx] <<" has too few matched points "<< std::endl;
			}
        }

        // calibrate
        Mat idx;
        double rms = 0.0;
        if (_camType == PINHOLE)
        {
            rms = cv::calibrateCamera(_objectPointsForEachCamera[camera], _imagePointsForEachCamera[camera],
                image.size(), _cameraMatrix[camera], _distortCoeffs[camera], _omEachCamera[camera],
                _tEachCamera[camera],_flags);
            idx = Mat(1, (int)_omEachCamera[camera].size(), CV_32S);
            for (int i = 0; i < (int)idx.total(); ++i)
            {
                idx.at<int>(i) = i;
            }
        }
        //else if (_camType == FISHEYE)
        //{
        //    rms = cv::fisheye::calibrate(_objectPointsForEachCamera[camera], _imagePointsForEachCamera[camera],
        //        image.size(), _cameraMatrix[camera], _distortCoeffs[camera], _omEachCamera[camera],
        //        _tEachCamera[camera], _flags);
        //    idx = Mat(1, (int)_omEachCamera[camera].size(), CV_32S);
        //    for (int i = 0; i < (int)idx.total(); ++i)
        //    {
        //        idx.at<int>(i) = i;
        //    }
        //}
        else if (_camType == OMNIDIRECTIONAL)
        {
            rms = cv::omnidir::calibrate(_objectPointsForEachCamera[camera], _imagePointsForEachCamera[camera],
                image.size(), _cameraMatrix[camera], _xi[camera], _distortCoeffs[camera], _omEachCamera[camera],
                _tEachCamera[camera], _flags, TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 300, 1e-7),
                idx);
        }
        _cameraMatrix[camera].convertTo(_cameraMatrix[camera], CV_32F);
        _distortCoeffs[camera].convertTo(_distortCoeffs[camera], CV_32F);
        _xi[camera].convertTo(_xi[camera], CV_32F);
        //else
        //{
        //    CV_Error_(CV_StsOutOfRange, "Unknown camera type, use PINHOLE or OMNIDIRECTIONAL");
        //}

        for (int i = 0; i < (int)_omEachCamera[camera].size(); ++i)
        {
			int cameraVertex, timestamp, photoVertex;
			cameraVertex = camera;
			timestamp = timestampAvailable[camera][idx.at<int>(i)];

			photoVertex = this->getPhotoVertex(timestamp);

			if (_omEachCamera[camera][i].type()!=CV_32F)
			{
				_omEachCamera[camera][i].convertTo(_omEachCamera[camera][i], CV_32F);
			}
			if (_tEachCamera[camera][i].type()!=CV_32F)
			{
				_tEachCamera[camera][i].convertTo(_tEachCamera[camera][i], CV_32F);
			}

			Mat transform = Mat::eye(4, 4, CV_32F);
			Mat R, T;
			Rodrigues(_omEachCamera[camera][i], R);
			T = (_tEachCamera[camera][i]).reshape(1, 3);
			R.copyTo(transform.rowRange(0, 3).colRange(0, 3));
			T.copyTo(transform.rowRange(0, 3).col(3));

			this->_edgeList.push_back(edge(cameraVertex, photoVertex, idx.at<int>(i), transform));
        }
		std::cout << "initialized for camera " << camera << " rms = " << rms << std::endl;
		std::cout << "initialized camera matrix for camera " << camera << " is" << std::endl;
		std::cout << _cameraMatrix[camera] << std::endl;
		std::cout << "xi for camera " << camera << " is " << _xi[camera] << std::endl;
    }

}

int MultiCameraCalibration::getPhotoVertex(int timestamp)
{
    int photoVertex = INVALID;

    // find in existing photo vertex
    for (int i = 0; i < (int)_vertexList.size(); ++i)
    {
        if (_vertexList[i].timestamp == timestamp)
        {
            photoVertex = i;
            break;
        }
    }

    // add a new photo vertex
    if (photoVertex == INVALID)
    {
        _vertexList.push_back(vertex(Mat::eye(4, 4, CV_32F), timestamp));
        photoVertex = (int)_vertexList.size() - 1;
    }

    return photoVertex;
}

void MultiCameraCalibration::initialize()
{
    int nVertices = (int)_vertexList.size();
    int nEdges = (int) _edgeList.size();

    // build graph
    Mat G = Mat::zeros(nVertices, nVertices, CV_32S);
    for (int edgeIdx = 0; edgeIdx < nEdges; ++edgeIdx)
    {
        G.at<int>(this->_edgeList[edgeIdx].cameraVertex, this->_edgeList[edgeIdx].photoVertex) = edgeIdx + 1;
    }
    G = G + G.t();

    // traverse the graph
    std::vector<int> pre, order;
    graphTraverse(G, 0, order, pre);

    for (int i = 0; i < _nCamera; ++i)
    {
        if (pre[i] == INVALID)
        {
            std::cout << "camera" << i << "is not connected" << std::endl;
        }
    }

    for (int i = 1; i < (int)order.size(); ++i)
    {
        int vertexIdx = order[i];
        Mat prePose = this->_vertexList[pre[vertexIdx]].pose;
        int edgeIdx = G.at<int>(vertexIdx, pre[vertexIdx]) - 1;
        Mat transform = this->_edgeList[edgeIdx].transform;

        if (vertexIdx < _nCamera)
        {
            this->_vertexList[vertexIdx].pose = transform * prePose.inv();
            this->_vertexList[vertexIdx].pose.convertTo(this->_vertexList[vertexIdx].pose, CV_32F);
			if (_verbose)
			{
				std::cout << "initial pose for camera " << vertexIdx << " is " << std::endl;
				std::cout << this->_vertexList[vertexIdx].pose << std::endl;
			}
        }
        else
        {
            this->_vertexList[vertexIdx].pose = prePose.inv() * transform;
            this->_vertexList[vertexIdx].pose.convertTo(this->_vertexList[vertexIdx].pose, CV_32F);
        }
    }
}

double MultiCameraCalibration::optimizeExtrinsics()
{
    // get om, t vector
    int nVertex = (int)this->_vertexList.size();

    Mat extrinParam(1, (nVertex-1)*6, CV_32F);
    int offset = 0;
    // the pose of the vertex[0] is eye
    for (int i = 1; i < nVertex; ++i)
    {
        Mat rvec, tvec;
        cv::Rodrigues(this->_vertexList[i].pose.rowRange(0,3).colRange(0, 3), rvec);
        this->_vertexList[i].pose.rowRange(0,3).col(3).copyTo(tvec);

        rvec.reshape(1, 1).copyTo(extrinParam.colRange(offset, offset + 3));
        tvec.reshape(1, 1).copyTo(extrinParam.colRange(offset+3, offset +6));
        offset += 6;
    }
    //double error_pre = computeProjectError(extrinParam);
    // optimization
    const double alpha_smooth = 0.01;
    double change = 1;
    for(int iter = 0; ; ++iter)
    {
        if ((_criteria.type == 1 && iter >= _criteria.maxCount)  ||
            (_criteria.type == 2 && change <= _criteria.epsilon) ||
            (_criteria.type == 3 && (change <= _criteria.epsilon || iter >= _criteria.maxCount)))
            break;
        double alpha_smooth2 = 1 - std::pow(1 - alpha_smooth, (double)iter + 1.0);
        Mat JTJ_inv, JTError;
        this->computeJacobianExtrinsic(extrinParam, JTJ_inv, JTError);
        Mat G = alpha_smooth2*JTJ_inv * JTError;
        if (G.depth() == CV_64F)
        {
            G.convertTo(G, CV_32F);
        }

        extrinParam = extrinParam + G.reshape(1, 1);

        change = norm(G) / norm(extrinParam);
        //double error = computeProjectError(extrinParam);
    }

    double error = computeProjectError(extrinParam);

    std::vector<Vec3f> rvecVertex, tvecVertex;
    vector2parameters(extrinParam, rvecVertex, tvecVertex);
    for (int verIdx = 1; verIdx < (int)_vertexList.size(); ++verIdx)
    {
        Mat R;
        Mat pose = Mat::eye(4, 4, CV_32F);
        Rodrigues(rvecVertex[verIdx-1], R);
        R.copyTo(pose.colRange(0, 3).rowRange(0, 3));
        Mat(tvecVertex[verIdx-1]).reshape(1, 3).copyTo(pose.rowRange(0, 3).col(3));
        _vertexList[verIdx].pose = pose;
		if (_verbose && verIdx < _nCamera)
		{
			std::cout << "final camera pose of camera " << verIdx << " is" << std::endl;
			std::cout << pose << std::endl;
		}
    }
    return error;
}

void MultiCameraCalibration::computeJacobianExtrinsic(const Mat& extrinsicParams, Mat& JTJ_inv, Mat& JTE)
{
    int nParam = (int)extrinsicParams.total();
    int nEdge = (int)_edgeList.size();
    std::vector<int> pointsLocation(nEdge+1, 0);

    for (int edgeIdx = 0; edgeIdx < nEdge; ++edgeIdx)
    {
        int nPoints = (int)_objectPointsForEachCamera[_edgeList[edgeIdx].cameraVertex][_edgeList[edgeIdx].photoIndex].total();
        pointsLocation[edgeIdx+1] = pointsLocation[edgeIdx] + nPoints*2;
    }

    JTJ_inv = Mat(nParam, nParam, CV_64F);
    JTE = Mat(nParam, 1, CV_64F);

    Mat J = Mat::zeros(pointsLocation[nEdge], nParam, CV_64F);
    Mat E = Mat::zeros(pointsLocation[nEdge], 1, CV_64F);

    for (int edgeIdx = 0; edgeIdx < nEdge; ++edgeIdx)
    {
        int photoVertex = _edgeList[edgeIdx].photoVertex;
        int photoIndex = _edgeList[edgeIdx].photoIndex;
        int cameraVertex = _edgeList[edgeIdx].cameraVertex;

        Mat objectPoints = _objectPointsForEachCamera[cameraVertex][photoIndex];
        Mat imagePoints = _imagePointsForEachCamera[cameraVertex][photoIndex];

        Mat rvecTran, tvecTran;
        Mat R = _edgeList[edgeIdx].transform.rowRange(0, 3).colRange(0, 3);
        tvecTran = _edgeList[edgeIdx].transform.rowRange(0, 3).col(3);
        cv::Rodrigues(R, rvecTran);

        Mat rvecPhoto = extrinsicParams.colRange((photoVertex-1)*6, (photoVertex-1)*6 + 3);
        Mat tvecPhoto = extrinsicParams.colRange((photoVertex-1)*6 + 3, (photoVertex-1)*6 + 6);

        Mat rvecCamera, tvecCamera;
        if (cameraVertex > 0)
        {
            rvecCamera = extrinsicParams.colRange((cameraVertex-1)*6, (cameraVertex-1)*6 + 3);
            tvecCamera = extrinsicParams.colRange((cameraVertex-1)*6 + 3, (cameraVertex-1)*6 + 6);
        }
        else
        {
            rvecCamera = Mat::zeros(3, 1, CV_32F);
            tvecCamera = Mat::zeros(3, 1, CV_32F);
        }

        Mat jacobianPhoto, jacobianCamera, error;
        computePhotoCameraJacobian(rvecPhoto, tvecPhoto, rvecCamera, tvecCamera, rvecTran, tvecTran,
            objectPoints, imagePoints, this->_cameraMatrix[cameraVertex], this->_distortCoeffs[cameraVertex],
            this->_xi[cameraVertex], jacobianPhoto, jacobianCamera, error);
        if (cameraVertex > 0)
        {
            jacobianCamera.copyTo(J.rowRange(pointsLocation[edgeIdx], pointsLocation[edgeIdx+1]).
                colRange((cameraVertex-1)*6, cameraVertex*6));
        }
        jacobianPhoto.copyTo(J.rowRange(pointsLocation[edgeIdx], pointsLocation[edgeIdx+1]).
            colRange((photoVertex-1)*6, photoVertex*6));
        error.copyTo(E.rowRange(pointsLocation[edgeIdx], pointsLocation[edgeIdx+1]));
    }
    //std::cout << J.t() * J << std::endl;
    JTJ_inv = (J.t() * J + 1e-10).inv();
    JTE = J.t() * E;

}
void MultiCameraCalibration::computePhotoCameraJacobian(const Mat& rvecPhoto, const Mat& tvecPhoto, const Mat& rvecCamera,
    const Mat& tvecCamera, Mat& rvecTran, Mat& tvecTran, const Mat& objectPoints, const Mat& imagePoints, const Mat& K,
    const Mat& distort, const Mat& xi, Mat& jacobianPhoto, Mat& jacobianCamera, Mat& E)
{
    Mat drvecTran_drecvPhoto, drvecTran_dtvecPhoto,
        drvecTran_drvecCamera, drvecTran_dtvecCamera,
        dtvecTran_drvecPhoto, dtvecTran_dtvecPhoto,
        dtvecTran_drvecCamera, dtvecTran_dtvecCamera;

    MultiCameraCalibration::compose_motion(rvecPhoto, tvecPhoto, rvecCamera, tvecCamera, rvecTran, tvecTran,
        drvecTran_drecvPhoto, drvecTran_dtvecPhoto, drvecTran_drvecCamera, drvecTran_dtvecCamera,
        dtvecTran_drvecPhoto, dtvecTran_dtvecPhoto, dtvecTran_drvecCamera, dtvecTran_dtvecCamera);

    if (rvecTran.depth() == CV_64F)
    {
        rvecTran.convertTo(rvecTran, CV_32F);
    }
    if (tvecTran.depth() == CV_64F)
    {
        tvecTran.convertTo(tvecTran, CV_32F);
    }
    float xif = 0.0f;
    if (_camType == OMNIDIRECTIONAL)
    {
        xif= xi.at<float>(0);
    }

    Mat imagePoints2, jacobian, dx_drvecCamera, dx_dtvecCamera, dx_drvecPhoto, dx_dtvecPhoto;
    if (_camType == PINHOLE)
    {
        cv::projectPoints(objectPoints, rvecTran, tvecTran, K, distort, imagePoints2, jacobian);
    }
    //else if (_camType == FISHEYE)
    //{
    //    cv::fisheye::projectPoints(objectPoints, imagePoints2, rvecTran, tvecTran, K, distort, 0, jacobian);
    //}
    else if (_camType == OMNIDIRECTIONAL)
    {
        cv::omnidir::projectPoints(objectPoints, imagePoints2, rvecTran, tvecTran, K, xif, distort, jacobian);
    }
    if (objectPoints.depth() == CV_32F)
    {
        Mat(imagePoints - imagePoints2).convertTo(E, CV_64FC2);
    }
    else
    {
        E = imagePoints - imagePoints2;
    }
    E = E.reshape(1, (int)imagePoints.total()*2);

    dx_drvecCamera = jacobian.colRange(0, 3) * drvecTran_drvecCamera + jacobian.colRange(3, 6) * dtvecTran_drvecCamera;
    dx_dtvecCamera = jacobian.colRange(0, 3) * drvecTran_dtvecCamera + jacobian.colRange(3, 6) * dtvecTran_dtvecCamera;
    dx_drvecPhoto = jacobian.colRange(0, 3) * drvecTran_drecvPhoto + jacobian.colRange(3, 6) * dtvecTran_drvecPhoto;
    dx_dtvecPhoto = jacobian.colRange(0, 3) * drvecTran_dtvecPhoto + jacobian.colRange(3, 6) * dtvecTran_dtvecPhoto;

    jacobianCamera = cv::Mat(dx_drvecCamera.rows, 6, CV_64F);
    jacobianPhoto = cv::Mat(dx_drvecPhoto.rows, 6, CV_64F);

    dx_drvecCamera.copyTo(jacobianCamera.colRange(0, 3));
    dx_dtvecCamera.copyTo(jacobianCamera.colRange(3, 6));
    dx_drvecPhoto.copyTo(jacobianPhoto.colRange(0, 3));
    dx_dtvecPhoto.copyTo(jacobianPhoto.colRange(3, 6));

}
void MultiCameraCalibration::graphTraverse(const Mat& G, int begin, std::vector<int>& order, std::vector<int>& pre)
{
    CV_Assert(!G.empty() && G.rows == G.cols);
    int nVertex = G.rows;
    order.resize(0);
    pre.resize(nVertex, INVALID);
    pre[begin] = -1;
    std::vector<bool> visited(nVertex, false);
    std::queue<int> q;
    visited[begin] = true;
    q.push(begin);
    order.push_back(begin);

    while(!q.empty())
    {
        int v = q.front();
        q.pop();
        Mat idx;
        // use my findNonZero maybe
        findRowNonZero(G.row(v), idx);
        for(int i = 0; i < (int)idx.total(); ++i)
        {
            int neighbor = idx.at<int>(i);
            if (!visited[neighbor])
            {
                visited[neighbor] = true;
                q.push(neighbor);
                order.push_back(neighbor);
                pre[neighbor] = v;
            }
        }
    }
}

void MultiCameraCalibration::findRowNonZero(const Mat& row, Mat& idx)
{
    CV_Assert(!row.empty() && row.rows == 1 && row.channels() == 1);
    Mat _row;
    std::vector<int> _idx;
    row.convertTo(_row, CV_32F);
    for (int i = 0; i < (int)row.total(); ++i)
    {
        if (_row.at<float>(i) != 0)
        {
            _idx.push_back(i);
        }
    }
    idx.release();
    idx.create(1, (int)_idx.size(), CV_32S);
    for (int i = 0; i < (int)_idx.size(); ++i)
    {
        idx.at<int>(i) = _idx[i];
    }
}

double MultiCameraCalibration::computeProjectError(Mat& parameters)
{
    int nVertex = (int)_vertexList.size();
    CV_Assert((int)parameters.total() == (nVertex-1) * 6 && parameters.depth() == CV_32F);
    int nEdge = (int)_edgeList.size();

    // recompute the transform between photos and cameras

    std::vector<edge> edgeList = this->_edgeList;
    std::vector<Vec3f> rvecVertex, tvecVertex;
    vector2parameters(parameters, rvecVertex, tvecVertex);

    float totalError = 0;
    int totalNPoints = 0;
    for (int edgeIdx = 0; edgeIdx < nEdge; ++edgeIdx)
    {
        Mat RPhoto, RCamera, TPhoto, TCamera, transform;
        int cameraVertex = edgeList[edgeIdx].cameraVertex;
        int photoVertex = edgeList[edgeIdx].photoVertex;
        int PhotoIndex = edgeList[edgeIdx].photoIndex;
        TPhoto = Mat(tvecVertex[photoVertex - 1]).reshape(1, 3);

        //edgeList[edgeIdx].transform = Mat::ones(4, 4, CV_32F);
        transform = Mat::eye(4, 4, CV_32F);
        cv::Rodrigues(rvecVertex[photoVertex-1], RPhoto);
        if (cameraVertex == 0)
        {
            RPhoto.copyTo(transform.rowRange(0, 3).colRange(0, 3));
            TPhoto.copyTo(transform.rowRange(0, 3).col(3));
        }
        else
        {
            TCamera = Mat(tvecVertex[cameraVertex - 1]).reshape(1, 3);
            cv::Rodrigues(rvecVertex[cameraVertex - 1], RCamera);
            Mat(RCamera*RPhoto).copyTo(transform.rowRange(0, 3).colRange(0, 3));
            Mat(RCamera * TPhoto + TCamera).copyTo(transform.rowRange(0, 3).col(3));
        }

        transform.copyTo(edgeList[edgeIdx].transform);
        Mat rvec, tvec;
        cv::Rodrigues(transform.rowRange(0, 3).colRange(0, 3), rvec);
        transform.rowRange(0, 3).col(3).copyTo(tvec);

        Mat objectPoints, imagePoints, proImagePoints;
        objectPoints = this->_objectPointsForEachCamera[cameraVertex][PhotoIndex];
        imagePoints = this->_imagePointsForEachCamera[cameraVertex][PhotoIndex];

        if (this->_camType == PINHOLE)
        {
            cv::projectPoints(objectPoints, rvec, tvec, _cameraMatrix[cameraVertex], _distortCoeffs[cameraVertex],
                proImagePoints);
        }
        //else if (this->_camType == FISHEYE)
        //{
        //    cv::fisheye::projectPoints(objectPoints, proImagePoints, rvec, tvec, _cameraMatrix[cameraVertex],
        //        _distortCoeffs[cameraVertex]);
        //}
        else if (this->_camType == OMNIDIRECTIONAL)
        {
            float xi = _xi[cameraVertex].at<float>(0);

            cv::omnidir::projectPoints(objectPoints, proImagePoints, rvec, tvec, _cameraMatrix[cameraVertex],
                xi, _distortCoeffs[cameraVertex]);
        }
        Mat error = imagePoints - proImagePoints;
        Vec2f* ptr_err = error.ptr<Vec2f>();
        for (int i = 0; i < (int)error.total(); ++i)
        {
            totalError += sqrt(ptr_err[i][0]*ptr_err[i][0] + ptr_err[i][1]*ptr_err[i][1]);
        }
        totalNPoints += (int)error.total();
    }
    double meanReProjError = totalError / totalNPoints;
    _error = meanReProjError;
    return meanReProjError;
}

void MultiCameraCalibration::compose_motion(InputArray _om1, InputArray _T1, InputArray _om2, InputArray _T2, Mat& om3, Mat& T3, Mat& dom3dom1,
    Mat& dom3dT1, Mat& dom3dom2, Mat& dom3dT2, Mat& dT3dom1, Mat& dT3dT1, Mat& dT3dom2, Mat& dT3dT2)
{
    Mat om1, om2, T1, T2;
    _om1.getMat().convertTo(om1, CV_64F);
    _om2.getMat().convertTo(om2, CV_64F);
    _T1.getMat().reshape(1, 3).convertTo(T1, CV_64F);
    _T2.getMat().reshape(1, 3).convertTo(T2, CV_64F);
    /*Mat om2 = _om2.getMat();
    Mat T1 = _T1.getMat().reshape(1, 3);
    Mat T2 = _T2.getMat().reshape(1, 3);*/

    //% Rotations:
    Mat R1, R2, R3, dR1dom1(9, 3, CV_64FC1), dR2dom2;
    cv::Rodrigues(om1, R1, dR1dom1);
    cv::Rodrigues(om2, R2, dR2dom2);
    /*JRodriguesMatlab(dR1dom1, dR1dom1);
    JRodriguesMatlab(dR2dom2, dR2dom2);*/
    dR1dom1 = dR1dom1.t();
    dR2dom2 = dR2dom2.t();

    R3 = R2 * R1;
    Mat dR3dR2, dR3dR1;
    //dAB(R2, R1, dR3dR2, dR3dR1);
    matMulDeriv(R2, R1, dR3dR2, dR3dR1);
    Mat dom3dR3;
    cv::Rodrigues(R3, om3, dom3dR3);
    //JRodriguesMatlab(dom3dR3, dom3dR3);
    dom3dR3 = dom3dR3.t();

    dom3dom1 = dom3dR3 * dR3dR1 * dR1dom1;
    dom3dom2 = dom3dR3 * dR3dR2 * dR2dom2;
    dom3dT1 = Mat::zeros(3, 3, CV_64FC1);
    dom3dT2 = Mat::zeros(3, 3, CV_64FC1);

    //% Translations:
    Mat T3t = R2 * T1;
    Mat dT3tdR2, dT3tdT1;
    //dAB(R2, T1, dT3tdR2, dT3tdT1);
    matMulDeriv(R2, T1, dT3tdR2, dT3tdT1);

    Mat dT3tdom2 = dT3tdR2 * dR2dom2;
    T3 = T3t + T2;
    dT3dT1 = dT3tdT1;
    dT3dT2 = Mat::eye(3, 3, CV_64FC1);
    dT3dom2 = dT3tdom2;
    dT3dom1 = Mat::zeros(3, 3, CV_64FC1);
}

void MultiCameraCalibration::vector2parameters(const Mat& parameters, std::vector<Vec3f>& rvecVertex, std::vector<Vec3f>& tvecVertexs)
{
    int nVertex = (int)_vertexList.size();
    CV_Assert((int)parameters.channels() == 1 && (int)parameters.total() == 6*(nVertex - 1));
    CV_Assert(parameters.depth() == CV_32F);
    parameters.reshape(1, 1);

    rvecVertex.reserve(0);
    tvecVertexs.resize(0);

    for (int i = 0; i < nVertex - 1; ++i)
    {
        rvecVertex.push_back(Vec3f(parameters.colRange(i*6, i*6 + 3)));
        tvecVertexs.push_back(Vec3f(parameters.colRange(i*6 + 3, i*6 + 6)));
    }
}

void MultiCameraCalibration::parameters2vector(const std::vector<Vec3f>& rvecVertex, const std::vector<Vec3f>& tvecVertex, Mat& parameters)
{
    CV_Assert(rvecVertex.size() == tvecVertex.size());
    int nVertex = (int)rvecVertex.size();
    // the pose of the first camera is known
    parameters.create(1, 6*(nVertex-1), CV_32F);

    for (int i = 0; i < nVertex-1; ++i)
    {
        Mat(rvecVertex[i]).reshape(1, 1).copyTo(parameters.colRange(i*6, i*6 + 3));
        Mat(tvecVertex[i]).reshape(1, 1).copyTo(parameters.colRange(i*6 + 3, i*6 + 6));
    }
}

void MultiCameraCalibration::writeParameters(const std::string& filename)
{
    FileStorage fs( filename, FileStorage::WRITE );

    fs << "nCameras" << _nCamera;

    for (int camIdx = 0; camIdx < _nCamera; ++camIdx)
    {
        char num[10];
        sprintf(num, "%d", camIdx);
        std::string cameraMatrix = "camera_matrix_" + std::string(num);
        std::string cameraPose = "camera_pose_" + std::string(num);
        std::string cameraDistortion = "camera_distortion_" + std::string(num);
        std::string cameraXi = "xi_" + std::string(num);

        fs << cameraMatrix << _cameraMatrix[camIdx];
        fs << cameraDistortion << _distortCoeffs[camIdx];
        if (_camType == OMNIDIRECTIONAL)
        {
            fs << cameraXi << _xi[camIdx].at<float>(0);
        }

        fs << cameraPose << _vertexList[camIdx].pose;
    }

    fs << "meanReprojectError" <<_error;

    for (int photoIdx = _nCamera; photoIdx < (int)_vertexList.size(); ++photoIdx)
    {
        char timestamp[100];
        sprintf(timestamp, "%d", _vertexList[photoIdx].timestamp);
        std::string photoTimestamp = "pose_timestamp_" + std::string(timestamp);

        fs << photoTimestamp << _vertexList[photoIdx].pose;
    }
}
}} // namespace multicalib, cv