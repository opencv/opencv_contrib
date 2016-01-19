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
#include "TrackerStruckModel.hpp" 


/*---------------------------
|  TrackerStruck
|---------------------------*/ 
namespace cv
{
    /*
    * Prototype
    */
    class TrackerStruckImpl : public TrackerStruck
    {
    public:
        TrackerStruckImpl(const TrackerStruck::Params &parameters = TrackerStruck::Params());
        void read( const FileNode& fn );
        void write( FileStorage& fs ) const;

    protected:
		void normalizeResps(std::vector<Mat>& resps);
		
		/*
        * basic functions and vars
        */
        bool initImpl( const Mat& /*image*/, const Rect2d& boundingBox );
        bool updateImpl( const Mat& image, Rect2d& boundingBox );

        TrackerStruck::Params params;
	};
    
    /*
    * Constructor
    */
    Ptr<TrackerStruck> TrackerStruck::createTracker(const TrackerStruck::Params &parameters){
        return Ptr<TrackerStruckImpl>(new TrackerStruckImpl(parameters));
    } 
    TrackerStruckImpl::TrackerStruckImpl( const TrackerStruck::Params &parameters ) :
        params( parameters )
    {
        isInit = false;
    }

    void TrackerStruckImpl::read( const cv::FileNode& fn ){
        params.read( fn );
    }

    void TrackerStruckImpl::write( cv::FileStorage& fs ) const{
        params.write( fs );
    }

	void TrackerStruckImpl::normalizeResps(std::vector<Mat>& resps)
	{
#pragma omp parallel for
		for (int i = 0; i < resps[0].rows; i++) {
			const float n = (float)norm(resps[0].row(i));

			for (int j = 0; j < resps[0].cols; j++) {
				Mat_<float>(resps[0])(i, j) /= n;
			}
		}
	}	

	/* ----- INIT Implementation -------------------------------------------------------------*/
	bool TrackerStruckImpl::initImpl(const Mat& image, const Rect2d& boundingBox)
	{
		// initialize seed
		if (params.customSeed != -1)
			srand(params.customSeed);
		else
			srand(time(NULL));

		// Struck Sampler
		TrackerSamplerCircular::Params samplerParams;
		samplerParams.radius = params.searchRadius;
		Ptr<TrackerSamplerAlgorithm> circularSampler = Ptr<TrackerSamplerCircular>(new TrackerSamplerCircular(samplerParams));
		if (!sampler->addTrackerSamplerAlgorithm(circularSampler))
			return false;

		// HAAR features
		TrackerFeatureHAAR::Params HAARParams;
		HAARParams.numFeatures = 192;
		HAARParams.isIntegral = true;
		HAARParams.rectSize = Size(static_cast<int>(boundingBox.width), static_cast<int>(boundingBox.height));
		Ptr<TrackerFeature> trackerFeature = Ptr<TrackerFeatureHAAR>(new TrackerFeatureHAAR(HAARParams));
		if (!featureSet->addTrackerFeature(trackerFeature))
			return false;

		// Model
		model = Ptr<TrackerStruckModel>(new TrackerStruckModel(boundingBox));
		model.staticCast<TrackerStruckModel>()->setSearchRadius(params.searchRadius);

		// Estimator
		TrackerStateEstimatorStruckSVM::Params stateEstimatorParams;
		stateEstimatorParams.svmBudgetSize = params.svmBudgetSize;
		stateEstimatorParams.svmC = params.svmC;

		Ptr<TrackerStateEstimatorStruckSVM> stateEstimator =
			Ptr<TrackerStateEstimatorStruckSVM>(new TrackerStateEstimatorStruckSVM(stateEstimatorParams));
		stateEstimator->setSearchRadius(params.searchRadius);
		model->setTrackerStateEstimator(stateEstimator);

		Mat frame;
		image.copyTo(frame);
		cvtColor(frame, frame, CV_BGR2GRAY);
		Mat tmp(frame.rows, frame.cols, CV_8UC1);
		frame.copyTo(tmp);
		Mat intImage(tmp.rows + 1, tmp.cols + 1, CV_32SC1);
		integral(tmp, intImage);

		circularSampler.staticCast<TrackerSamplerCircular>()->setMode(TrackerSamplerCircular::MODE_RADIAL);
		sampler->sampling(intImage, boundingBox);
		std::vector<Mat> samples = sampler->getSamples();

		featureSet->extraction(samples);
		std::vector<Mat> resps = featureSet->getResponses();
		normalizeResps(resps);

		model.staticCast<TrackerStruckModel>()->setCurrentSamples(samples);
		model.staticCast<TrackerStruckModel>()->setCurrentBoundingBox(boundingBox);
		model->modelEstimation(resps);
		ConfidenceMap currentMap = model.staticCast<TrackerStruckModel>()->getCurrentConfidenceMap();

		model->getTrackerStateEstimator().staticCast<TrackerStateEstimatorStruckSVM>()->setCurrentConfidenceMap(currentMap);
		model->getTrackerStateEstimator().staticCast<TrackerStateEstimatorStruckSVM>()->setCurrentCentre(boundingBox);

		model->modelUpdate();
        
        return true;
    }
    
    /* ----- UPDATE Implementation -----------------------------------------------------------*/    
    bool TrackerStruckImpl::updateImpl( const Mat& image, Rect2d& boundingBox ) 
    {       
		// tracking phase
		Ptr<TrackerTargetState> lastTargetState = model->getLastTargetState();
		Rect2d lastbb(lastTargetState->getTargetPosition().x, lastTargetState->getTargetPosition().y, lastTargetState->getTargetWidth(), lastTargetState->getTargetHeight());

		Mat frame;
		image.copyTo(frame);
		cvtColor(frame, frame, CV_BGR2GRAY);
		Mat tmp(frame.rows, frame.cols, CV_8UC1);
		frame.copyTo(tmp);
		Mat intImage(tmp.rows + 1, tmp.cols + 1, CV_32SC1);
		integral(tmp, intImage);

		(sampler->getSamplers().at(0).second).staticCast<TrackerSamplerCircular>()->setMode(TrackerSamplerCircular::MODE_PIXELS);
		sampler->sampling(intImage, lastbb);
		std::vector<Mat> samples = sampler->getSamples();

		featureSet->extraction(samples);
		std::vector<Mat> resps = featureSet->getResponses();
		normalizeResps(resps);

		model.staticCast<TrackerStruckModel>()->setCurrentSamples(samples);
		model.staticCast<TrackerStruckModel>()->setCurrentBoundingBox(lastbb);
		model->modelEstimation(resps);

		ConfidenceMap currentMap = model.staticCast<TrackerStruckModel>()->getCurrentConfidenceMap();
		model->getTrackerStateEstimator().staticCast<TrackerStateEstimatorStruckSVM>()->setCurrentConfidenceMap(currentMap);
		model->getTrackerStateEstimator().staticCast<TrackerStateEstimatorStruckSVM>()->setCurrentCentre(boundingBox);
		if (!model->runStateEstimator()) {
			return false;
		}

		// update model phase
		samples.clear();
		resps.clear();
		currentMap.clear();

		Ptr<TrackerTargetState> currentState = model->getLastTargetState();
		boundingBox = Rect((int)currentState->getTargetPosition().x, (int)currentState->getTargetPosition().y, 
			currentState->getTargetWidth(),
			currentState->getTargetHeight());

		(sampler->getSamplers().at(0).second).staticCast<TrackerSamplerCircular>()->setMode(TrackerSamplerCircular::MODE_RADIAL);
		sampler->sampling(intImage, boundingBox);
		samples = sampler->getSamples();

		featureSet->extraction(samples);
		resps = featureSet->getResponses();
		normalizeResps(resps);

		model.staticCast<TrackerStruckModel>()->setCurrentSamples(samples);
		model.staticCast<TrackerStruckModel>()->setCurrentBoundingBox(boundingBox);
		model.staticCast<TrackerStruckModel>()->responseToConfidenceMap(resps, currentMap);
		model->getTrackerStateEstimator().staticCast<TrackerStateEstimatorStruckSVM>()->setCurrentConfidenceMap(currentMap);

		model->getTrackerStateEstimator().staticCast<TrackerStateEstimatorStruckSVM>()->setCurrentCentre(boundingBox);

        model->modelUpdate();
        
        return true;
    }
    
    /*
    * Parameters
    */
    TrackerStruck::Params::Params(){
        // default values of the params
        searchRadius = 30; // px
        svmC = 100.0;
        svmBudgetSize = 100;
		customSeed = -1;
    }

    void TrackerStruck::Params::read( const cv::FileNode& fn ){

    }

    void TrackerStruck::Params::write( cv::FileStorage& fs ) const{

    }

} /* namespace cv */