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

#ifndef __OPENCV_ARUCO_CPP__
#define __OPENCV_ARUCO_CPP__
#ifdef __cplusplus

#include "precomp.hpp"
#include "opencv2/aruco.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "predefined_dictionaries.cpp"

#include <iostream>


namespace cv{ namespace aruco{

using namespace std;


/**
 * @brief detect marker candidates
 * 
 * @param _image input image
 * @param _candidates return candidate corners positions
 * @param _threshParam window size for adaptative thresholding
 * @param _minLenght minimum size of candidates contour lenght. It is indicated as a ratio
 *                  respect to the largest image dimension
 * @param _thresholdedImage if set, returns the thresholded image for debugging purposes.
 * @return void
 */
void _detectCandidates(InputArray _image, OutputArrayOfArrays _candidates, int _threshParam,
                           float _minLenght, OutputArray _thresholdedImage=noArray()) {

    cv::Mat image = _image.getMat();


    /// 1. CONVERT TO GRAY
    cv::Mat grey;
    if ( image.type() ==CV_8UC3 )   cv::cvtColor ( image,grey,cv::COLOR_BGR2GRAY );
    else grey=image;


    /// 2. THRESHOLD
    CV_Assert(_threshParam >= 3);
    cv::Mat thresh;
    cv::adaptiveThreshold ( grey,thresh,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY_INV,_threshParam,7 );
    if(_thresholdedImage.needed()) thresh.copyTo(_thresholdedImage);


    /// 3. DETECT RECTANGLES
    int minSize=_minLenght*std::max(thresh.cols,thresh.rows);
    cv::Mat contoursImg;
    thresh.copyTo ( contoursImg );
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours ( contoursImg , contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE );
    std::vector< std::vector<cv::Point2f> > candidates;
    for ( unsigned int i=0;i<contours.size();i++ )
    {
        if(contours[i].size() < minSize) continue;
        vector<Point>  approxCurve;
        cv::approxPolyDP (  contours[i], approxCurve, double ( contours[i].size() ) *0.05 , true );
        if(approxCurve.size() != 4 || !cv::isContourConvex(approxCurve) ) continue;
        float minDistSq = 1e10;
        for ( int j=0;j<4;j++ ) {
            float d= ( float ) ( approxCurve[j].x-approxCurve[ ( j+1 ) %4].x ) * ( approxCurve[j].x-approxCurve[ ( j+1 ) %4].x ) +
                                 ( approxCurve[j].y-approxCurve[ ( j+1 ) %4].y ) * ( approxCurve[j].y-approxCurve[ ( j+1 ) %4].y ) ;
            minDistSq = std::min(minDistSq,d);
        }
        if(minDistSq<100) continue;
        std::vector<cv::Point2f> currentCandidate;
        currentCandidate.resize(4);
        for ( int j=0;j<4;j++ ) {
            currentCandidate[j] = cv::Point2f ( approxCurve[j].x,approxCurve[j].y );
        }
        candidates.push_back(currentCandidate);
    }


    /// 4. SORT CORNERS
    for ( unsigned int i=0;i<candidates.size(); i++ ) {
        double dx1 = candidates[i][1].x - candidates[i][0].x;
        double dy1 = candidates[i][1].y - candidates[i][0].y;
        double dx2 = candidates[i][2].x - candidates[i][0].x;
        double dy2 = candidates[i][2].y - candidates[i][0].y;
        double o = ( dx1*dy2 )- ( dy1*dx2 );

        if ( o  < 0.0 )	swap ( candidates[i][1],candidates[i][3] );
    }


    /// 5. FILTER OUT NEAR CANDIDATE PAIRS
    std::vector< std::pair<int,int>  > nearCandidates;
    for ( unsigned int i=0;i<candidates.size();i++ ) {
        for ( unsigned int j=i+1;j<candidates.size(); j++ ) {
            float distSq=0;
            for ( int c=0;c<4;c++ )
                distSq += ( candidates[i][c].x-candidates[j][c].x ) * ( candidates[i][c].x-candidates[j][c].x )
                        + ( candidates[i][c].y-candidates[j][c].y ) * ( candidates[i][c].y-candidates[j][c].y ) ;
            distSq/=4.;
            if(distSq < 100) nearCandidates.push_back( std::pair<int,int> ( i,j ) );
        }
    }


    /// 6. MARK SMALLER CANDIDATES IN NEAR PAIRS TO REMOVE
    std::vector<bool> toRemove(candidates.size(), false);
    for(unsigned int i=0; i<nearCandidates.size(); i++) {
        float perimeterSq1=0, perimeterSq2=0;
        for(unsigned int c=0; c<4; c++) {
            perimeterSq1 += (candidates[nearCandidates[i].first][c].x-candidates[nearCandidates[i].first][(c+1)%4].x) *
                            (candidates[nearCandidates[i].first][c].x-candidates[nearCandidates[i].first][(c+1)%4].x) +
                            (candidates[nearCandidates[i].first][c].y-candidates[nearCandidates[i].first][(c+1)%4].y) *
                            (candidates[nearCandidates[i].first][c].y-candidates[nearCandidates[i].first][(c+1)%4].y);
            perimeterSq2 += (candidates[nearCandidates[i].second][c].x-candidates[nearCandidates[i].second][(c+1)%4].x) *
                            (candidates[nearCandidates[i].second][c].x-candidates[nearCandidates[i].second][(c+1)%4].x) +
                            (candidates[nearCandidates[i].second][c].y-candidates[nearCandidates[i].second][(c+1)%4].y) *
                            (candidates[nearCandidates[i].second][c].y-candidates[nearCandidates[i].second][(c+1)%4].y);
            if(perimeterSq1 > perimeterSq2) toRemove[nearCandidates[i].second]=true;
            else toRemove[nearCandidates[i].first]=true;
        }
    }



    /// 7. REMOVE EXTRA CANDIDATES
    int totalRemaining=0;
    for(unsigned int i=0; i<toRemove.size(); i++) if(!toRemove[i]) totalRemaining++;
    _candidates.create(totalRemaining, 1, CV_32FC2);
    for(unsigned int i=0, currIdx=0; i<candidates.size(); i++) {
        if(toRemove[i]) continue;
        _candidates.create(4,1,CV_32FC2, currIdx, true);
        Mat m = _candidates.getMat(currIdx);
        for(int j=0; j<4; j++) m.ptr<cv::Vec2f>(0)[j] = candidates[i][j];
        currIdx++;
    }





//    //sort by id
//    std::sort ( detectedMarkers.begin(),detectedMarkers.end() );
//    //there might be still the case that a marker is detected twice because of the double border indicated earlier,
//    //detect and remove these cases
//    int borderDistThresX=_borderDistThres*float(input.cols);
//    int borderDistThresY=_borderDistThres*float(input.rows);
//    vector<bool> toRemove ( detectedMarkers.size(),false );
//    for ( int i=0;i<int ( detectedMarkers.size() )-1;i++ )
//    {
//        if ( detectedMarkers[i].id==detectedMarkers[i+1].id && !toRemove[i+1] )
//        {
//            //deletes the one with smaller perimeter
//            if ( perimeter ( detectedMarkers[i] ) >perimeter ( detectedMarkers[i+1] ) ) toRemove[i+1]=true;
//            else toRemove[i]=true;
//        }
//        //delete if any of the corners is too near image border
//        for(size_t c=0;c<detectedMarkers[i].size();c++){
//        if ( detectedMarkers[i][c].x<borderDistThresX ||
//          detectedMarkers[i][c].y<borderDistThresY ||
//          detectedMarkers[i][c].x>input.cols-borderDistThresX ||
//          detectedMarkers[i][c].y>input.rows-borderDistThresY ) toRemove[i]=true;

//    }


//    }
//    //remove the markers marker
//    removeElements ( detectedMarkers, toRemove );


}




/**
 * @brief identify a vector of marker candidates based on the dictionary codification
 * 
 * @param image input image
 * @param candidates candidate corners positions
 * @param dictionary
 * @param accepted returns vector of accepted marker corners
 * @param ids returns vector of accepted markers ids
 * @param rejected ... if set, return vector of rejected markers
 * @return void
 */
void _identifyCandidates(InputArray image, InputArrayOfArrays _candidates,
                             Dictionary dictionary, OutputArrayOfArrays _accepted, OutputArray _ids,
                             OutputArrayOfArrays _rejected=noArray()) {


    int ncandidates = _candidates.total();
    CV_Assert(ncandidates > 0);

    std::vector< cv::Mat > accepted;
    std::vector< cv::Mat > rejected;
    std::vector< int > ids;

    cv::Mat grey;
    if ( image.getMat().type() ==CV_8UC3 )   cv::cvtColor ( image.getMat(),grey,cv::COLOR_BGR2GRAY );
    else grey=image.getMat();

    for(int i=0; i<ncandidates; i++) {
        int currId;
        cv::Mat currentCandidate = _candidates.getMat(i);
        if( dictionary.identify(grey,currentCandidate,currId) ) {
            accepted.push_back(currentCandidate);
            ids.push_back(currId);
        }
        else rejected.push_back(_candidates.getMat(i));
    }

    _accepted.create((int)accepted.size(), 1, CV_32FC2);
    for(unsigned int i=0; i<accepted.size(); i++) {
        _accepted.create(4,1,CV_32FC2, i, true);
        Mat m = _accepted.getMat(i);
        accepted[i].copyTo(m);
    }

    _ids.create((int)ids.size(), 1, CV_32SC1);
    for(unsigned int i=0; i<ids.size(); i++) _ids.getMat().ptr<int>(0)[i] = ids[i];

    if(_rejected.needed()) {
        _rejected.create((int)rejected.size(), 1, CV_32FC2);
        for(unsigned int i=0; i<rejected.size(); i++) {
            _rejected.create(4,1,CV_32FC2, i, true);
            Mat m = _rejected.getMat(i);
            rejected[i].copyTo(m);
        }
    }

}



/**
 * @brief Given the marker size, it returns the vector of object points for pose estimation
 * 
 * @param markerSize size of marker in meters
 * @param objPnts vector of 4 3d points
 * @return void
 */
void getSingleMarkerObjectPoints(float markerSize, OutputArray _objPnts) {

    CV_Assert(markerSize > 0);

     _objPnts.create(4, 1, CV_32FC3);
    cv::Mat objPnts = _objPnts.getMat();
    objPnts.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerSize/2., markerSize/2., 0);
    objPnts.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerSize/2., markerSize/2., 0);
    objPnts.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerSize/2., -markerSize/2., 0);
    objPnts.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerSize/2., -markerSize/2., 0);

}


/**
  */
void detectMarkers(InputArray image, Dictionary dictionary, OutputArrayOfArrays imgPoints,
                       OutputArray ids, int threshParam,float minLenght) {

    cv::Mat grey;
    if ( image.getMat().type() ==CV_8UC3 )   cv::cvtColor ( image.getMat(),grey,cv::COLOR_BGR2GRAY );
    else grey=image.getMat();

    /// STEP 1: Detect marker candidates
    std::vector<std::vector<cv::Point2f> > candidates;
    _detectCandidates(grey,candidates,threshParam,minLenght);

    /// STEP 2: Check candidate codification (identify markers)
    _identifyCandidates(grey, candidates, dictionary, imgPoints, ids);

    /// STEP 3: Clean candidates

    for(int i=0; i<imgPoints.total(); i++) {
        /// STEP 4: Corner refinement
        cv::cornerSubPix ( grey, imgPoints.getMat(i), cvSize ( 5,5 ), cvSize ( -1,-1 ),cvTermCriteria ( CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,30,0.1 ) );
    }

}


/**
  */
const Dictionary& getPredefinedDictionary(PREDEFINED_DICTIONARIES name) {
    switch(name) {
        case DICT_ARUCO:
        return _dict_aruco;
    }
    return Dictionary();
}


/**
  */
void detectMarkers(InputArray image, PREDEFINED_DICTIONARIES dictionary, OutputArrayOfArrays imgPoints,
                       OutputArray ids, int threshParam,float minLenght) {

    detectMarkers(image, getPredefinedDictionary(dictionary), imgPoints, ids, threshParam, minLenght);
}





/**
  */
void estimatePoseSingleMarkers(InputArrayOfArrays imgPoints, float markersize, InputArray cameraMatrix,
                                          InputArray distCoeffs, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs) {

    cv::Mat markerObjPoints;
    getSingleMarkerObjectPoints(markersize, markerObjPoints);
    rvecs.create( (int)imgPoints.total(), 1, CV_32FC1);
    tvecs.create( (int)imgPoints.total(), 1, CV_32FC1);

    for(int i=0; i<imgPoints.total(); i++) {
        rvecs.create(3,1,CV_64FC1, i, true);
        tvecs.create(3,1,CV_64FC1, i, true);
        cv::solvePnP(markerObjPoints, imgPoints.getMat(i), cameraMatrix, distCoeffs, rvecs.getMat(i), tvecs.getMat(i));
    }

}


/**
  */
void estimatePoseBoard(InputArrayOfArrays imgPoints, InputArray ids, Board board, InputArray cameraMatrix,
                                          InputArray distCoeffs, OutputArray rvec, OutputArray tvec) {

    cv::Mat objectPointsConcatenation,imagePointsConcatenation;
    board.getObjectAndImagePoints(ids, imgPoints, imagePointsConcatenation, objectPointsConcatenation);

    rvec.create(3,1,CV_64FC1);
    tvec.create(3,1,CV_64FC1);
    cv::solvePnP(objectPointsConcatenation, imagePointsConcatenation, cameraMatrix, distCoeffs, rvec, tvec);
}




/**
 */
void drawDetectedMarkers(InputArray in, OutputArray out, InputArrayOfArrays markers, InputArray ids) {

    out.create(in.size(), in.type());
    cv::Mat outImg = out.getMat();
    in.getMat().copyTo(outImg);
  
    for(int i=0; i<markers.total(); i++) {
        cv::Mat currentMarker = markers.getMat(i);
        for(int j=0; j<4; j++) {
            cv::Point2f p0, p1;
            p0 = currentMarker.ptr<cv::Point2f>(0)[j];
            p1 = currentMarker.ptr<cv::Point2f>(0)[(j+1)%4];
            cv::line(outImg, p0, p1, cv::Scalar(0,255,0),2);
        }
        cv::rectangle( outImg, currentMarker.ptr<cv::Point2f>(0)[0]-Point2f(3,3),currentMarker.ptr<cv::Point2f>(0)[0]+Point2f(3,3),Scalar(255,0,0),2,cv::LINE_AA);
        if(ids.total()!=0) {
            Point2f cent(0,0);
            for(int p=0; p<4; p++) cent += currentMarker.ptr<cv::Point2f>(0)[p];
            cent = cent/4.;
            stringstream s;
            s << "id=" << ids.getMat().ptr<int>(0)[i] ;
            putText(outImg, s.str(), cent, cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 2);
        }
    }



}


/**
 */
void drawAxis(InputOutputArray image, InputArray cameraMatrix, InputArray distCoeffs, InputArray rvec, InputArray tvec, float lenght) {
    std::vector<cv::Point3f> axisPoints;
    axisPoints.push_back(cv::Point3f(0,0,0));
    axisPoints.push_back(cv::Point3f(lenght,0,0));
    axisPoints.push_back(cv::Point3f(0,lenght,0));
    axisPoints.push_back(cv::Point3f(0,0,lenght));
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

    cv::line(image, imagePoints[0], imagePoints[1], cv::Scalar(0,0,255), 3);
    cv::line(image, imagePoints[0], imagePoints[2], cv::Scalar(0,255,0), 3);
    cv::line(image, imagePoints[0], imagePoints[3], cv::Scalar(255,0,0), 3);

}




/**
 */
void drawMarker(Dictionary dict, int id, int sidePixels, OutputArray img) {
    dict.drawMarker(id,sidePixels,img);
}


/**
 */
void drawMarker(PREDEFINED_DICTIONARIES dict, int id, int sidePixels, OutputArray img) {
    drawMarker(getPredefinedDictionary(dict), id, sidePixels, img);
}


/**
 */
void drawPlanarBoard(Board board, Dictionary dict, cv::Size outSize, OutputArray img) {
    board.drawBoard(dict, outSize, img);
}

/**
 */
void drawPlanarBoard(Board board, PREDEFINED_DICTIONARIES dict, cv::Size outSize, OutputArray img) {
    drawPlanarBoard(board, getPredefinedDictionary(dict), outSize, img);
}




}}

#endif // cplusplus
#endif // __OPENCV_ARUCO_CPP__

