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
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
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

#include "precomp.hpp"

namespace cv
{

/*
 *  TrackerMedianFlow
 */

class TrackerMedianFlowModel : public TrackerModel
{
 public:
  TrackerMedianFlowModel(){}
  Rect getBoundingBox(){return boundingBox_;}
  void setBoudingBox(Rect boundingBox){boundingBox_=boundingBox;}
  Mat getImage(){return image_;}
  void setImage(Mat image){image_=image;}
  bool setTrackerStateEstimator( Ptr<TrackerStateEstimator> /*trackerStateEstimator*/ ){return false;}
  void modelEstimation( const std::vector<Mat>& /*responses*/ ){}
  void modelUpdate(){}
  bool runStateEstimator(){return false;}
  void setLastTargetState( const Ptr<TrackerTargetState>& /*lastTargetState*/ ){}
  Ptr<TrackerTargetState> getLastTargetState() const{return trajectory.back();}
  const std::vector<ConfidenceMap>& getConfidenceMaps() const{return confidenceMaps;}
  const ConfidenceMap& getLastConfidenceMap() const{return currentConfidenceMap;}
  Ptr<TrackerStateEstimator> getTrackerStateEstimator() const{return stateEstimator;}
 private:
  void clearCurrentConfidenceMap(){}
 protected:
  Rect boundingBox_;
  Mat image_;
  std::vector<ConfidenceMap> confidenceMaps;
  Ptr<TrackerStateEstimator> stateEstimator;
  ConfidenceMap currentConfidenceMap;
  Trajectory trajectory;
  int maxCMLength;
  void modelEstimationImpl( const std::vector<Mat>& responses ){}
  void modelUpdateImpl(){}
};
Rect medianFlowImpl(Mat oldImage,Mat newImage,Rect oldBox);

/*
 * Parameters
 */
TrackerMedianFlow::Params::Params()
{
    printf("tesi me %d %s\n",__LINE__,__FILE__);
}

void TrackerMedianFlow::Params::read( const cv::FileNode& /*fn*/ )
{
  //numClassifiers = fn["numClassifiers"];
}

void TrackerMedianFlow::Params::write( cv::FileStorage& /*fs*/ ) const
{
  //fs << "numClassifiers" << numClassifiers;
}

/*
 * Constructor
 */
TrackerMedianFlow::TrackerMedianFlow( const TrackerMedianFlow::Params &parameters) :
    params( parameters )
{
  isInit = false;
}

/*
 * Destructor
 */
TrackerMedianFlow::~TrackerMedianFlow()
{
}

void TrackerMedianFlow::read( const cv::FileNode& fn )
{
  params.read( fn );
}

void TrackerMedianFlow::write( cv::FileStorage& fs ) const
{
  params.write( fs );
}

bool TrackerMedianFlow::initImpl( const Mat& image, const Rect& boundingBox )
{
    model=Ptr<TrackerMedianFlowModel>(new TrackerMedianFlowModel());
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setImage(image);
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setBoudingBox(boundingBox);
    return true;
}

bool TrackerMedianFlow::updateImpl( const Mat& image, Rect& boundingBox )
{
    Mat oldImage=((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->getImage();
    Rect oldBox=((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->getBoundingBox();
    boundingBox=medianFlowImpl(oldImage,image,oldBox);
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setImage(image);
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setBoudingBox(boundingBox);
    return true;
}

Rect medianFlowImpl(Mat oldImage,Mat newImage,Rect oldBox){
    return oldBox;
    //make grid TODO: make rectangle colored with same num of rects in every dim
    //compute opt flow for every point in grid
    //for every point:
    //      compute FB error
    //      compute NCC
    // filter
    // vote
}

} /* namespace cv */
