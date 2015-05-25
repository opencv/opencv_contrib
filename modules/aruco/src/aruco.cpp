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

#include <vector>

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
        if(contours[i].size() > minSize) continue;
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
        for ( int j=0;j<4;j++ ) {
            currentCandidate.push_back( Point2f ( approxCurve[j].x,approxCurve[j].y ) );
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


//    //find all rectangles in the thresholdes image
//    vector<MarkerCandidate > MarkerCanditates;
//    detectRectangles ( thres,MarkerCanditates );
//    //if the image has been downsampled, then calcualte the location of the corners in the original image
//    if ( pyrdown_level!=0 )
//    {
//        float red_den=pow ( 2.0f,pyrdown_level );
//        float offInc= ( ( pyrdown_level/2. )-0.5 );
//        for ( unsigned int i=0;i<MarkerCanditates.size();i++ ) {
//            for ( int c=0;c<4;c++ )
//            {
//                MarkerCanditates[i][c].x=MarkerCanditates[i][c].x*red_den+offInc;
//                MarkerCanditates[i][c].y=MarkerCanditates[i][c].y*red_den+offInc;
//            }
//            //do the same with the the contour points
//            for ( int c=0;c<MarkerCanditates[i].contour.size();c++ )
//            {
//                MarkerCanditates[i].contour[c].x=MarkerCanditates[i].contour[c].x*red_den+offInc;
//                MarkerCanditates[i].contour[c].y=MarkerCanditates[i].contour[c].y*red_den+offInc;
//            }
//        }
//    }


//    ///identify the markers
//    vector<vector<Marker> >markers_omp(omp_get_max_threads());
//    vector<vector < std::vector<cv::Point2f> > >candidates_omp(omp_get_max_threads());
//    #pragma omp parallel for
//    for ( unsigned int i=0;i<MarkerCanditates.size();i++ )
//    {
//        //Find proyective homography
//        Mat canonicalMarker;
//        bool resW=false;
//        resW=warp ( grey,canonicalMarker,Size ( _markerWarpSize,_markerWarpSize ),MarkerCanditates[i] );
//        if (resW) {
//             int nRotations;
//            int id= ( *markerIdDetector_ptrfunc ) ( canonicalMarker,nRotations );
//            if ( id!=-1 )
//            {
//        if(_cornerMethod==LINES) // make LINES refinement before lose contour points
//          refineCandidateLines( MarkerCanditates[i], camMatrix, distCoeff );
//                markers_omp[omp_get_thread_num()].push_back ( MarkerCanditates[i] );
//                markers_omp[omp_get_thread_num()].back().id=id;
//                //sort the points so that they are always in the same order no matter the camera orientation
//                std::rotate ( markers_omp[omp_get_thread_num()].back().begin(),markers_omp[omp_get_thread_num()].back().begin() +4-nRotations,markers_omp[omp_get_thread_num()].back().end() );
//            }
//            else candidates_omp[omp_get_thread_num()].push_back ( MarkerCanditates[i] );
//        }

//    }
//    //unify parallel data
//    joinVectors(markers_omp,detectedMarkers,true);
//    joinVectors(candidates_omp,_candidates,true);



//    ///refine the corner location if desired
//    if ( detectedMarkers.size() >0 && _cornerMethod!=NONE && _cornerMethod!=LINES )
//    {
//        vector<Point2f> Corners;
//        for ( unsigned int i=0;i<detectedMarkers.size();i++ )
//            for ( int c=0;c<4;c++ )
//                Corners.push_back ( detectedMarkers[i][c] );

//        if ( _cornerMethod==HARRIS )
//            findBestCornerInRegion_harris ( grey, Corners,7 );
//        else if ( _cornerMethod==SUBPIX )
//            cornerSubPix ( grey, Corners,cvSize ( 5,5 ), cvSize ( -1,-1 )   ,cvTermCriteria ( CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,3,0.05 ) );

//        //copy back
//        for ( unsigned int i=0;i<detectedMarkers.size();i++ )
//            for ( int c=0;c<4;c++ )     detectedMarkers[i][c]=Corners[i*4+c];
//    }
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

    for(int i=0; i<ncandidates; i++) {
        int currId;        
        if( dictionary.identify(image,_candidates.getMat(i),currId) ) {
            accepted.push_back(_candidates.getMat(i));
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
void detectSingleMarkers(InputArray image, InputArray cameraMatrix, InputArray distCoeffs,
                        float markersize, Dictionary dictionary, OutputArrayOfArrays imgPoints,
                        OutputArray ids, OutputArray Rvec, OutputArray Tvec,
                        int threshParam, float minLenght) {

    // STEP 1: Detect marker candidates
    std::vector<std::vector<cv::Point2f> > candidates;
    _detectCandidates(image,candidates,threshParam,minLenght);

    // STEP 2: Check candidate codification (identify markers)
    std::vector< std::vector<cv::Point2f> > accepted;
    _identifyCandidates(image, candidates, dictionary, accepted, ids);

    // STEP 3: Clean candidates
	

    for(int i=0; i<imgPoints.total(); i++) {
        // STEP 4: Corner refinement
        //cv::cornerSubpix...

        // STEP 5: Pose Estimation
        
    }

}



/**
 */
void detectBoardMarkers(InputArray image, InputArray cameraMatrix, InputArray distCoeffs,
                       Board board, OutputArrayOfArrays imgPoints, OutputArray ids,
                       OutputArray rvec, OutputArray tvec, int threshParam, float minLenght) {

    // STEP 1: Detect marker candidates
 /*   std::vector<std::vector<cv::Point2f> > candidates;
    _detectArucoCandidates(image,candidates,threshParam1,threshParam2,minLenght,maxLenght);

    // STEP 2: Check candidate codification (identify markers)
    std::vector<std::vector<cv::Point2f> > accepted;
    std::vector< int > ids;
    _identifyArucoCandidates(candidates, dictionary, accepted, ids);

    // STEP 3: Corner refinement
    for(int i=0; i<accepted.size; i++) {
        
        cv::cornerSubpix...

    }

    // STEP 4: Pose Estimation
*/
}


/**
 */
void drawDetectedMarkers(InputArray image, InputArrayOfArrays markers, InputArray ids) {
    /// TODO
}


/**
 */
void drawAxis(InputArray image, InputArray cameraMatrix, InputArray distCoeffs, InputArray rvec, InputArray tvec, float lenght) {
    /// TODO
}




/**
 */
bool Dictionary::identify(InputArray image, InputArray imgPoints, int &idx) {
    /// TODO
    // get canonical image
    // get code
    cv::Mat candidateBits;
    // get quartets
    cv::Mat candidateQuartets = _getQuartet(candidateBits);

    // search closest marker in dict
    int closestId=-1;
    unsigned int closestDistance=markerSize*markerSize+1;
    cv::Mat candidateDistances = _getDistances(candidateQuartets);
    for(int i=0; i<codes.rows; i++) {
        if(candidateDistances.ptr<unsigned int>(0)[i] > closestDistance) {
            closestDistance = candidateDistances.ptr<unsigned int>(0)[i];
            closestId = i;
        }
    }
    // return closest id
    if(closestId!=-1 && closestDistance<=maxCorrectionBits) {
        idx = closestId;
        return true;
    }
    else {
        idx = -1;
        return false;
    }
}



/**
 */
void Dictionary::drawMarker(InputOutputArray img, int id) {
    /// TODO
}


/**
  */
cv::Mat Dictionary::_getQuartet(cv::Mat bits) {

    int nquartets = (markerSize*markerSize)/4 + (markerSize*markerSize)%4;
    cv::Mat candidateQuartets(1, nquartets, CV_8UC1);
    int currentQuartet=0;
    for(int row=0; row<markerSize/2; row++)
    {
        for(int col=row; col<markerSize-row-1; col++) {
            unsigned char bit3 = bits.at<unsigned char>(row,col);
            unsigned char bit2 = bits.at<unsigned char>(col,markerSize-row);
            unsigned char bit1 = bits.at<unsigned char>(markerSize-row,markerSize-col);
            unsigned char bit0 = bits.at<unsigned char>(markerSize-col,row);
            unsigned char quartet = 8*bit3 + 4*bit2 + 2*bit1 + bit0;
            candidateQuartets.ptr<unsigned char>()[currentQuartet] = quartet;
            currentQuartet++;
        }
    }
    if((markerSize*markerSize)%4 == 1) { // middle bit
        unsigned char middleBit = bits.at<unsigned char>(markerSize/2+1,markerSize/2+1);
        candidateQuartets.ptr<unsigned char>()[currentQuartet] = middleBit;
    }
    return candidateQuartets;

}


/**
  */
cv::Mat Dictionary::_getDistances(cv::Mat quartets) {
    //cv::Mat res(codes.size(), 1, CV_32UC1);
    /// TODO
    return cv::Mat();
}



/**
 */
void Board::drawBoard(InputOutputArray img) {
    /// TODO
}


/**
 */
Board Board::createPlanarBoard(int width, int height, float markerSize, 
				float markerSeparation, Dictionary dictionary) {
    /// TODO
    return Board();
}


}}

#endif // cplusplus
#endif // __OPENCV_ARUCO_CPP__

