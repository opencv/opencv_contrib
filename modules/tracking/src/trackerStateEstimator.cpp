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
 *  TrackerStateEstimator
 */

TrackerStateEstimator::~TrackerStateEstimator()
{

}

Ptr<TrackerTargetState> TrackerStateEstimator::estimate( const std::vector<ConfidenceMap>& confidenceMaps )
{
  if( confidenceMaps.empty() )
    return Ptr<TrackerTargetState>();

  return estimateImpl( confidenceMaps );

}

void TrackerStateEstimator::update( std::vector<ConfidenceMap>& confidenceMaps )
{
  if( confidenceMaps.empty() )
    return;

  return updateImpl( confidenceMaps );

}

Ptr<TrackerStateEstimator> TrackerStateEstimator::create( const String& trackeStateEstimatorType )
{

  if( trackeStateEstimatorType.find( "SVM" ) == 0 )
  {
    return Ptr<TrackerStateEstimatorSVM>( new TrackerStateEstimatorSVM() );
  }

  if( trackeStateEstimatorType.find( "BOOSTING" ) == 0 )
  {
    return Ptr<TrackerStateEstimatorMILBoosting>( new TrackerStateEstimatorMILBoosting() );
  }

  CV_Error( -1, "Tracker state estimator type not supported" );
  return Ptr<TrackerStateEstimator>();
}

String TrackerStateEstimator::getClassName() const
{
  return className;
}

/**
 * TrackerStateEstimatorMILBoosting::TrackerMILTargetState
 */
TrackerStateEstimatorMILBoosting::TrackerMILTargetState::TrackerMILTargetState( const Point2f& position, int width, int height, bool foreground,
                                                                                const Mat& features )
{
  setTargetPosition( position );
  setTargetWidth( width );
  setTargetHeight( height );
  setTargetFg( foreground );
  setFeatures( features );
}

void TrackerStateEstimatorMILBoosting::TrackerMILTargetState::setTargetFg( bool foreground )
{
  isTarget = foreground;
}

void TrackerStateEstimatorMILBoosting::TrackerMILTargetState::setFeatures( const Mat& features )
{
  targetFeatures = features;
}

bool TrackerStateEstimatorMILBoosting::TrackerMILTargetState::isTargetFg() const
{
  return isTarget;
}

Mat TrackerStateEstimatorMILBoosting::TrackerMILTargetState::getFeatures() const
{
  return targetFeatures;
}

TrackerStateEstimatorMILBoosting::TrackerStateEstimatorMILBoosting( int nFeatures )
{
  className = "BOOSTING";
  trained = false;
  numFeatures = nFeatures;
}

TrackerStateEstimatorMILBoosting::~TrackerStateEstimatorMILBoosting()
{

}

void TrackerStateEstimatorMILBoosting::setCurrentConfidenceMap( ConfidenceMap& confidenceMap )
{
  currentConfidenceMap.clear();
  currentConfidenceMap = confidenceMap;
}

uint TrackerStateEstimatorMILBoosting::max_idx( const std::vector<float> &v )
{
  const float* findPtr = & ( *std::max_element( v.begin(), v.end() ) );
  const float* beginPtr = & ( *v.begin() );
  return (uint) ( findPtr - beginPtr );
}

Ptr<TrackerTargetState> TrackerStateEstimatorMILBoosting::estimateImpl( const std::vector<ConfidenceMap>& /*confidenceMaps*/)
{
  //run ClfMilBoost classify in order to compute next location
  if( currentConfidenceMap.empty() )
    return Ptr<TrackerTargetState>();

  Mat positiveStates;
  Mat negativeStates;

  prepareData( currentConfidenceMap, positiveStates, negativeStates );

  std::vector<float> prob = boostMILModel.classify( positiveStates );

  int bestind = max_idx( prob );
  //float resp = prob[bestind];

  return currentConfidenceMap.at( bestind ).first;
}

void TrackerStateEstimatorMILBoosting::prepareData( const ConfidenceMap& confidenceMap, Mat& positive, Mat& negative )
{

  int posCounter = 0;
  int negCounter = 0;

  for ( size_t i = 0; i < confidenceMap.size(); i++ )
  {
    Ptr<TrackerMILTargetState> currentTargetState = confidenceMap.at( i ).first.staticCast<TrackerMILTargetState>();
    if( currentTargetState->isTargetFg() )
      posCounter++;
    else
      negCounter++;
  }

  positive.create( posCounter, numFeatures, CV_32FC1 );
  negative.create( negCounter, numFeatures, CV_32FC1 );

  //TODO change with mat fast access
  //initialize trainData (positive and negative)

  int pc = 0;
  int nc = 0;
  for ( size_t i = 0; i < confidenceMap.size(); i++ )
  {
    Ptr<TrackerMILTargetState> currentTargetState = confidenceMap.at( i ).first.staticCast<TrackerMILTargetState>();
    Mat stateFeatures = currentTargetState->getFeatures();

    if( currentTargetState->isTargetFg() )
    {
      for ( int j = 0; j < stateFeatures.rows; j++ )
      {
        //fill the positive trainData with the value of the feature j for sample i
        positive.at<float>( pc, j ) = stateFeatures.at<float>( j, 0 );
      }
      pc++;
    }
    else
    {
      for ( int j = 0; j < stateFeatures.rows; j++ )
      {
        //fill the negative trainData with the value of the feature j for sample i
        negative.at<float>( nc, j ) = stateFeatures.at<float>( j, 0 );
      }
      nc++;
    }

  }
}

void TrackerStateEstimatorMILBoosting::updateImpl( std::vector<ConfidenceMap>& confidenceMaps )
{

  if( !trained )
  {
    //this is the first time that the classifier is built
    //init MIL
    boostMILModel.init();
    trained = true;
  }

  ConfidenceMap lastConfidenceMap = confidenceMaps.back();
  Mat positiveStates;
  Mat negativeStates;

  prepareData( lastConfidenceMap, positiveStates, negativeStates );
  //update MIL
  boostMILModel.update( positiveStates, negativeStates );

}

/**
 * TrackerStateEstimatorAdaBoosting
 */
TrackerStateEstimatorAdaBoosting::TrackerStateEstimatorAdaBoosting( int numClassifer, int initIterations, int nFeatures, Size patchSize, const Rect& ROI )
{
  className = "ADABOOSTING";
  numBaseClassifier = numClassifer;
  numFeatures = nFeatures;
  iterationInit = initIterations;
  initPatchSize = patchSize;
  trained = false;
  sampleROI = ROI;

}

Rect TrackerStateEstimatorAdaBoosting::getSampleROI() const
{
  return sampleROI;
}

void TrackerStateEstimatorAdaBoosting::setSampleROI( const Rect& ROI )
{
  sampleROI = ROI;
}

/**
 * TrackerAdaBoostingTargetState::TrackerAdaBoostingTargetState
 */
TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState::TrackerAdaBoostingTargetState( const Point2f& position, int width, int height,
                                                                                                bool foreground, const Mat& responses )
{
  setTargetPosition( position );
  setTargetWidth( width );
  setTargetHeight( height );

  setTargetFg( foreground );
  setTargetResponses( responses );
}

void TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState::setTargetFg( bool foreground )
{
  isTarget = foreground;
}

bool TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState::isTargetFg() const
{
  return isTarget;
}

void TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState::setTargetResponses( const Mat& responses )
{
  targetResponses = responses;
}

Mat TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState::getTargetResponses() const
{
  return targetResponses;
}

TrackerStateEstimatorAdaBoosting::~TrackerStateEstimatorAdaBoosting()
{

}
void TrackerStateEstimatorAdaBoosting::setCurrentConfidenceMap( ConfidenceMap& confidenceMap )
{
  currentConfidenceMap.clear();
  currentConfidenceMap = confidenceMap;
}

std::vector<int> TrackerStateEstimatorAdaBoosting::computeReplacedClassifier()
{
  return replacedClassifier;
}

std::vector<int> TrackerStateEstimatorAdaBoosting::computeSwappedClassifier()
{
  return swappedClassifier;
}

std::vector<int> TrackerStateEstimatorAdaBoosting::computeSelectedWeakClassifier()
{
  return boostClassifier->getSelectedWeakClassifier();
}

Ptr<TrackerTargetState> TrackerStateEstimatorAdaBoosting::estimateImpl( const std::vector<ConfidenceMap>& /*confidenceMaps*/ )
{
  //run classify in order to compute next location
  if( currentConfidenceMap.empty() )
    return Ptr<TrackerTargetState>();

  std::vector<Mat> images;

  for ( size_t i = 0; i < currentConfidenceMap.size(); i++ )
  {
    Ptr<TrackerAdaBoostingTargetState> currentTargetState = currentConfidenceMap.at( i ).first.staticCast<TrackerAdaBoostingTargetState>();
    images.push_back( currentTargetState->getTargetResponses() );
  }

  int bestIndex;
  boostClassifier->classifySmooth( images, sampleROI, bestIndex );

  // get bestIndex from classifySmooth
  return currentConfidenceMap.at( bestIndex ).first;

}

void TrackerStateEstimatorAdaBoosting::updateImpl( std::vector<ConfidenceMap>& confidenceMaps )
{
  if( !trained )
  {
    //this is the first time that the classifier is built
    int numWeakClassifier = numBaseClassifier * 10;

    bool useFeatureExchange = true;
    boostClassifier = Ptr<StrongClassifierDirectSelection>(
        new StrongClassifierDirectSelection( numBaseClassifier, numWeakClassifier, initPatchSize, sampleROI, useFeatureExchange, iterationInit ) );
    //init base classifiers
    boostClassifier->initBaseClassifier();

    trained = true;
  }

  ConfidenceMap lastConfidenceMap = confidenceMaps.back();
  bool featureEx = boostClassifier->getUseFeatureExchange();

  replacedClassifier.clear();
  replacedClassifier.resize( lastConfidenceMap.size(), -1 );
  swappedClassifier.clear();
  swappedClassifier.resize( lastConfidenceMap.size(), -1 );

  for ( size_t i = 0; i < lastConfidenceMap.size() / 2; i++ )
  {
    Ptr<TrackerAdaBoostingTargetState> currentTargetState = lastConfidenceMap.at( i ).first.staticCast<TrackerAdaBoostingTargetState>();

    int currentFg = 1;
    if( !currentTargetState->isTargetFg() )
      currentFg = -1;
    Mat res = currentTargetState->getTargetResponses();

    boostClassifier->update( res, currentFg );
    if( featureEx )
    {
      replacedClassifier[i] = boostClassifier->getReplacedClassifier();
      swappedClassifier[i] = boostClassifier->getSwappedClassifier();
      if( replacedClassifier[i] >= 0 && swappedClassifier[i] >= 0 )
        boostClassifier->replaceWeakClassifier( replacedClassifier[i] );
    }
    else
    {
      replacedClassifier[i] = -1;
      swappedClassifier[i] = -1;
    }

    int mapPosition = (int)(i + lastConfidenceMap.size() / 2);
    Ptr<TrackerAdaBoostingTargetState> currentTargetState2 = lastConfidenceMap.at( mapPosition ).first.staticCast<TrackerAdaBoostingTargetState>();

    currentFg = 1;
    if( !currentTargetState2->isTargetFg() )
      currentFg = -1;
    const Mat res2 = currentTargetState2->getTargetResponses();

    boostClassifier->update( res2, currentFg );
    if( featureEx )
    {
      replacedClassifier[mapPosition] = boostClassifier->getReplacedClassifier();
      swappedClassifier[mapPosition] = boostClassifier->getSwappedClassifier();
      if( replacedClassifier[mapPosition] >= 0 && swappedClassifier[mapPosition] >= 0 )
        boostClassifier->replaceWeakClassifier( replacedClassifier[mapPosition] );
    }
    else
    {
      replacedClassifier[mapPosition] = -1;
      swappedClassifier[mapPosition] = -1;
    }
  }

}

/**
 * TrackerStateEstimatorSVM
 */
TrackerStateEstimatorSVM::TrackerStateEstimatorSVM()
{
  className = "SVM";
}

TrackerStateEstimatorSVM::~TrackerStateEstimatorSVM()
{

}

Ptr<TrackerTargetState> TrackerStateEstimatorSVM::estimateImpl( const std::vector<ConfidenceMap>& confidenceMaps )
{
  return confidenceMaps.back().back().first;
}

void TrackerStateEstimatorSVM::updateImpl( std::vector<ConfidenceMap>& /*confidenceMaps*/)
{

}

/**
* TrackerStateEstimatorStruckSVM::SVMSupportPattern
*/

Rect2f cv::TrackerStateEstimatorStruckSVM::SVMSupportPattern::rectY()
{
	return rects[y];
}

/**
* TrackerStateEstimatorStruckSVM::SVMSupportVector
*/

Mat cv::TrackerStateEstimatorStruckSVM::SVMSupportVector::xy()
{
	return x->x.at(y);
}

Rect2f cv::TrackerStateEstimatorStruckSVM::SVMSupportVector::rectY()
{
	return x->rects[y];
}

/**
* TrackerStateEstimatorStruckSVM::TrackerStruckTargetState
*/

cv::TrackerStateEstimatorStruckSVM::TrackerStruckTargetState::TrackerStruckTargetState(const Point2f & position, int width, int height, Mat responses)
{
	setTargetPosition(position);
	setTargetWidth(width);
	setTargetHeight(height);

	y = Rect2d(position.x, position.y, width, height);

	setResp(responses);
	_centre = false;
	_updateOnly = false;
}

Mat cv::TrackerStateEstimatorStruckSVM::TrackerStruckTargetState::getResp()
{
	return x;
}

void cv::TrackerStateEstimatorStruckSVM::TrackerStruckTargetState::setResp(Mat value)
{
	x = value;
}

Rect2d cv::TrackerStateEstimatorStruckSVM::TrackerStruckTargetState::getBoundingBox()
{
	return y;
}

bool cv::TrackerStateEstimatorStruckSVM::TrackerStruckTargetState::isCentre()
{
	return _centre;
}

bool cv::TrackerStateEstimatorStruckSVM::TrackerStruckTargetState::isCentre(bool value)
{
	_centre = value;
	return _centre;
}

bool TrackerStateEstimatorStruckSVM::TrackerStruckTargetState::isUpdateOnly()
{
	return _updateOnly;
}

bool TrackerStateEstimatorStruckSVM::TrackerStruckTargetState::isUpdateOnly(bool value)
{
	return _updateOnly = value;
}

/**
* TrackerStateEstimatorStruckSVM
*/

cv::TrackerStateEstimatorStruckSVM::Params::Params() {
	svmBudgetSize = 100;
	svmC = 100.0;
}

cv::TrackerStateEstimatorStruckSVM::TrackerStateEstimatorStruckSVM()
{
	className = "StruckSVM";
	svmBudgetSize = 100;
	svmC = 100.0;
	currentBestIndex = -1;
	searchRadius = 30;
}

cv::TrackerStateEstimatorStruckSVM::TrackerStateEstimatorStruckSVM(Params params) :
	TrackerStateEstimatorStruckSVM()
{
	svmBudgetSize = params.svmBudgetSize;
	svmC = params.svmC;
}

void cv::TrackerStateEstimatorStruckSVM::setCurrentConfidenceMap(ConfidenceMap & confidenceMap)
{
	currentConfidenceMap.clear();
	currentConfidenceMap = confidenceMap;
}

void cv::TrackerStateEstimatorStruckSVM::setCurrentCentre(Rect2d rect)
{
	centre = rect;
}

void TrackerStateEstimatorStruckSVM::setSearchRadius(int radius)
{
	searchRadius = radius;
}

double LossFunc(Rect2d y1, Rect2d y2) {
	Rect2d overlap = y1 & y2;
	//double coeff = overlap.area() / (y1.area() + y2.area() - overlap.area());
	double coeff = 2 * overlap.area() / (y1.area() + y2.area() - overlap.area());
	return 1 - coeff;
}

double GaussianKernelEval(const Mat x1, const Mat x2)
{
	double normSub = norm(x1 - x2);
	return exp(-0.2 * (normSub*normSub));
}

int delta(int y1, int y2) {
	return (int)(y1 == y2);
}

double TrackerStateEstimatorStruckSVM::F(Mat x, Rect2d /*y*/) {

	double f = 0.0;

//#pragma omp parallel for reduction (+:f)
	for (int i = 0; i < (int)supportVectors.size(); ++i) {
		Ptr<SVMSupportVector> sv = supportVectors.at(i);		
		f += sv->beta * GaussianKernelEval(x, sv->x->x[sv->y]);
	}

	return f;
}

void TrackerStateEstimatorStruckSVM::Evaluate(ConfidenceMap& map) {
	const int size = map.size();

#pragma omp parallel for
	for (int i = 0; i < (int)size; i++) {

		Ptr<TrackerStruckTargetState> targetState = 
			map[i].first.staticCast<TrackerStateEstimatorStruckSVM::TrackerStruckTargetState>();

		Rect2d r(targetState->getBoundingBox());
		r.x -= centre.x;
		r.y -= centre.y;

		// 2 : yt = arg max F ( xt^pt-1, y)
		map[i].second = (float) F(targetState->getResp(), r);
	}
}

Ptr<TrackerTargetState> TrackerStateEstimatorStruckSVM::estimateImpl(const std::vector<ConfidenceMap>& /*confidenceMaps*/)
{
	/*ConfidenceMap currentConfidenceMap = confidenceMaps.back();*/

	// 1 : Estimate change in object location
	Evaluate(currentConfidenceMap);

	// finding arg max y
	double bestScore = -DBL_MAX;
	int bestIndex = -1;

	for (int i = 0; i < (int)currentConfidenceMap.size(); ++i)
	{
		if (currentConfidenceMap.at(i).second > bestScore)
		{
			bestScore = currentConfidenceMap.at(i).second;
			bestIndex = i;
		}
	}
	
	currentBestIndex = bestIndex;
	return currentConfidenceMap.at(bestIndex).first;
}

void cv::TrackerStateEstimatorStruckSVM::RemoveSupportVector(Ptr<SVMSupportVector> sv)
{
	Ptr<SVMSupportPattern> pattern = sv->x;

	pattern->refCount--;
	if (pattern->refCount == 0)
	{
		// remove the support pattern
		supportPatterns.erase(std::remove(supportPatterns.begin(), supportPatterns.end(), pattern));
		pattern.release();
	}

	// remove the support vector
	supportVectors.erase(std::remove(supportVectors.begin(), supportVectors.end(), sv));
	sv.release();
}

Ptr<TrackerStateEstimatorStruckSVM::SVMSupportVector> cv::TrackerStateEstimatorStruckSVM::AddSupportVector(Ptr<SVMSupportPattern> sp, int y, double g)
{
	if (!sp) return Ptr<SVMSupportVector>();

	Ptr<SVMSupportVector> new_sv = Ptr<SVMSupportVector>(new SVMSupportVector());

	new_sv->beta = 0.0;
	new_sv->x = sp;
	new_sv->x->refCount++;
	
	assert(y >= 0);
	new_sv->y = y;
	new_sv->g = g;

	supportVectors.push_back(new_sv);

	return new_sv;
}

void cv::TrackerStateEstimatorStruckSVM::MinGrad(Ptr<SVMSupportPattern> sp, int &y, double &g)
{
	y = -1;
	g = DBL_MAX;

//#pragma omp parallel for
	for (int i = 0; i < (int)sp->rects.size(); i++) {
		double gradient = -LossFunc(sp->rects[i], sp->rects[sp->y]) - F(sp->x[i], sp->rects[i]);
		
		//#pragma omp critical
		{
			if (gradient < g) {
				y = i;
				g = gradient;
			}
		}
	}
}

void cv::TrackerStateEstimatorStruckSVM::ProcessNew(Ptr<SVMSupportPattern> sp, Ptr<SVMSupportVector> & ypos, Ptr<SVMSupportVector> & yneg)
{
	// for positive support vector, loss is zero
	ypos = AddSupportVector(sp, sp->y, -F(sp->x[sp->y], sp->rects[sp->y]));

	int ymin;
	double gmin;
	MinGrad(sp, ymin, gmin);

	yneg = AddSupportVector(sp, ymin, gmin);

	// link between support vectors
	ypos->relative = Ptr<SVMSupportVector>(yneg);
	yneg->relative = Ptr<SVMSupportVector>(ypos);
}

void cv::TrackerStateEstimatorStruckSVM::ProcessOld(Ptr<SVMSupportVector> & ypos, Ptr<SVMSupportVector> & yneg)
{
	if (supportPatterns.size() == 0) return;

	// choose ramdomically
	int ind = rand() % supportPatterns.size();
	Ptr<SVMSupportPattern> pattern = supportPatterns.at(ind);

	double maxGrad = -DBL_MAX;
	ypos = Ptr<SVMSupportVector>();

	for (size_t i = 0; i < supportVectors.size(); i++) {		
		Ptr<SVMSupportVector> sv = supportVectors.at(i);

		if (supportVectors[i]->x != pattern) continue;
		
		if ((sv->g > maxGrad) && (sv->beta < (delta(sv->y, pattern->y) * svmC))) {
			maxGrad = sv->g;
			ypos = sv;
		}
	}

	if (!ypos) return;

	int ymin;
	double gmin;
	MinGrad(pattern, ymin, gmin);

	yneg = Ptr<SVMSupportVector>();
	for (size_t i = 0; i < supportVectors.size(); i++) {
		Ptr<SVMSupportVector> sv = supportVectors.at(i);

		if (supportVectors.at(i)->x != pattern) continue;

		if (sv->y == ymin) {
			yneg = sv;
			break;
		}
	}

	if (!yneg)
		yneg = AddSupportVector(pattern, ymin, gmin);
}

void cv::TrackerStateEstimatorStruckSVM::Optimize(Ptr<SVMSupportVector> & ypos, Ptr<SVMSupportVector> & yneg)
{
	if (supportPatterns.size() == 0) return;

	// choose ramdomically
	int ind = rand() % supportPatterns.size();
	Ptr<SVMSupportPattern> pattern = supportPatterns[ind];

	double maxGrad = -DBL_MAX;
	double minGrad = DBL_MAX;

	ypos = Ptr<SVMSupportVector>();
	yneg = Ptr<SVMSupportVector>();

	for (int i = 0; i < (int)supportVectors.size(); i++) {
		const Ptr<SVMSupportVector> sv = supportVectors[i];

		if (sv->x != pattern) continue;

		bool restr = sv->beta < (delta(sv->y, pattern->y) * svmC);
		
		if (sv->g > maxGrad && restr) {
			maxGrad = sv->g;
			ypos = sv;
		}

		if (sv->g < minGrad) {
			minGrad = sv->g;
			yneg = sv;
		}

	}

	assert(yneg);
	assert(ypos);
}

void cv::TrackerStateEstimatorStruckSVM::SMOStep(Ptr<SVMSupportVector> svPos, Ptr<SVMSupportVector> svNeg)
{
	if (svPos == svNeg) return;

	if ((svPos->g - svNeg->g) >= 1e-5)
	{
		// 1: k00 = <fi(xi, y+), fi(xi, y+)>
		double k00 = GaussianKernelEval(svPos->x->x[svPos->y], svPos->x->x[svPos->y]);

		// 2: k11 = <fi(xi, y-), fi(xi, y-)>
		double k11 = GaussianKernelEval(svNeg->x->x[svNeg->y], svNeg->x->x[svNeg->y]);

		// 3: k01 = <fi(xi, y+), fi(xi, y-)>
		double k01 = GaussianKernelEval(svPos->x->x[svPos->y], svNeg->x->x[svNeg->y]);

		// 4: lu = gi(y+) - gi(y-)/k00+k11-2*k01
		double lu = (svPos->g - svNeg->g) / (k00 + k11 - 2.0 * k01);

		// 5: l = max(0, min(lu, C*delta(y+, yi) - beta-y+-i))
		double l = max(0.0, min(lu, svmC * delta(svPos->y, svPos->x->y) - svPos->beta));

		// 6: *** Update coefficients
		// 7: betai-ypos = betai-ypos + lambda
		svPos->beta += l;

		// 8: betai-ypos = betai-yneg + lambda
		svNeg->beta -= l;

		// 9: *** Update gradients
		// 10: for (xj,y) in S do
//#pragma omp parallel for
		for (int j = 0; j < (int)supportVectors.size(); j++) {
			Ptr<SVMSupportVector> sv = supportVectors.at(j);

			// 11: k0 = <fi(xj, y), fi(xi, y++)>
			double k0 = GaussianKernelEval(sv->x->x[sv->y], svPos->x->x[svPos->y]);

			// 12: k1 = <fi(xj, y), fi(xi, y++)> 
			double k1 = GaussianKernelEval(sv->x->x[sv->y], svNeg->x->x[svNeg->y]);

			// 13: gi(y) = gj(y) - l * (k0 - k1)
			sv->g -= l * (k0 - k1);
		}
	}

	// so, maybe we should remove it
	if (fabs(svPos->beta) < 1e-8) {
		RemoveSupportVector(svPos);
	}
	if (fabs(svNeg->beta) < 1e-8) {
		RemoveSupportVector(svNeg);
	}
}

void cv::TrackerStateEstimatorStruckSVM::BudgetMaintenance()
{
	// if budget exceeded
	if (supportVectors.size() > (size_t)svmBudgetSize) {

		// remove the excess
		while (supportVectors.size() > (size_t)svmBudgetSize) {

			double minGrad = DBL_MAX;
			Ptr<SVMSupportVector> svPos, svNeg;

			// search min ||delta w|| ^ 2 (grad)
			for (int i = 0; i < (int)supportVectors.size(); ++i)
			{
				// find negative support vectors
				Ptr<SVMSupportVector> sv = supportVectors[i];
				if (sv->beta < 0.0)
				{
					// get the relative (positive support vector)
					Ptr<SVMSupportVector> svr = sv->relative;
					//Ptr<SVMSupportVector> svr = Ptr<SVMSupportVector>();

					if (!svr) {
						for (int k = 0; k < (int)supportVectors.size(); ++k)
						{
							if (supportVectors[k]->beta > 0.0 && supportVectors[k]->x == sv->x)
							{
								svr = supportVectors[k];
								break;
							}
						}
					}

					//if (svr) 
					//{
						assert(svr);

						double grad = (sv->beta * sv->beta) * (
							GaussianKernelEval(sv->xy(), sv->xy())
							+ GaussianKernelEval(svr->xy(), svr->xy())
							- 2.0 * GaussianKernelEval(sv->xy(), svr->xy()));
						
						if (grad < minGrad) {

							minGrad = grad;
							svPos = svr;
							svNeg = sv;

						}
					//}
				}
			} // end for

			assert(svPos);
			assert(svNeg);

			svPos->beta += svNeg->beta;

			// remove vectors
			RemoveSupportVector(svNeg);
			if (svPos->beta < 1e-8)
			{
				RemoveSupportVector(svPos);
			}

			// update g of remaining vectors
			for (int i = 0; i < (int)supportVectors.size(); ++i)
			{
				Ptr<SVMSupportVector> sv = supportVectors.at(i);
				sv->g = -LossFunc(sv->rectY(), sv->x->rectY()) -F(sv->xy(), sv->x->rects[sv->y]);
			}

		} // end while
	}

}

void TrackerStateEstimatorStruckSVM::updateImpl(std::vector<ConfidenceMap>& /*confidenceMaps*/)
{
	ConfidenceMap lastConfidenceMap = currentConfidenceMap; /*confidenceMaps.back();*/
	
	// create the support pattern
	Ptr<SVMSupportPattern> sp = Ptr<SVMSupportPattern>(new SVMSupportPattern());
	sp->refCount = 0;	

//#pragma omp parallel for
	for (int i = 0; i < (int)lastConfidenceMap.size(); ++i)
	{
		Ptr<TrackerStruckTargetState> targetState =
			lastConfidenceMap[i].first.staticCast<TrackerStateEstimatorStruckSVM::TrackerStruckTargetState>();		
	
		Rect2d r(targetState->getBoundingBox());
		r.x -= centre.x;
		r.y -= centre.y;

	//#pragma omp critical
	//{
		sp->x.push_back(targetState->getResp());
		sp->rects.push_back(r);

		if (targetState->isCentre())
			sp->y = (sp->x.size() - 1);				
	//}
	}
	
	supportPatterns.push_back(sp);	

	// 4: *** Update discriminant function
	Ptr<SVMSupportVector> ypos, yneg;

	ProcessNew(sp, ypos, yneg);
	SMOStep(ypos, yneg);

	BudgetMaintenance();

	for (int j = 0; j < 10; j++) {

		ProcessOld(ypos, yneg);
		SMOStep(ypos, yneg);

		BudgetMaintenance();

		for (int k = 0; k < 10; k++) {

			Optimize(ypos, yneg);
			SMOStep(ypos, yneg);
		} // end for

	} // end for
}

} /* namespace cv */
