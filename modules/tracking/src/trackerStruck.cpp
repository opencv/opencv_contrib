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
/*		
		class HaarFeature {
		private:
			std::vector<Rect2f> m_rects;
			std::vector<float> m_weights;
			float m_factor;
			Rect2d m_bb;
			//int Sum(const Mat & s, const Size & imageSize, Rect2i& rect, int channel = 0) const;
			int Sum(const Mat & s, const Size & imageSize, int x, int y, int width, int height) const;
		public:
			HaarFeature(const Rect2d& bb, int type);
			~HaarFeature();
			float HaarFeature::Eval(const Mat& intImage, const Mat& s) const;
		};
*/

		void normalizeResps(std::vector<Mat>& resps);

/*
		std::vector<HaarFeature> m_features;
		void GenerateSystematic();
		void ExtractFeatures(const Mat & intImage, const std::vector<Mat>& images, Mat& response);
*/
		
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

	/*
	void TrackerStruckImpl::GenerateSystematic()
	{
		float x[] = { 0.2f, 0.4f, 0.6f, 0.8f };
		float y[] = { 0.2f, 0.4f, 0.6f, 0.8f };
		float s[] = { 0.2f, 0.4f };
		for (int iy = 0; iy < 4; ++iy)
		{
			for (int ix = 0; ix < 4; ++ix)
			{
				for (int is = 0; is < 2; ++is)
				{
					Rect2d r(x[ix] - s[is] / 2, y[iy] - s[is] / 2, s[is], s[is]);
					for (int it = 0; it < 6; ++it)
					{
						m_features.push_back(HaarFeature(r, it));
					}
				}
			}
		}
	}

	void TrackerStruckImpl::ExtractFeatures(const Mat & intImage, const std::vector<Mat>& images, Mat & response)
	{
		response = Mat_<double>(Size((int)images.size(), m_features.size()));

//#pragma omp parallel for
		for (int i = 0; i < (int)images.size(); i++) {

			for (int j = 0; j < (int)m_features.size(); j++) {

				//response.at<double>(j, i) =
				double f = m_features[j].Eval(intImage, images[i]);
			}
		}
	}
*/

	/* ----- INIT Implementation -------------------------------------------------------------*/
	bool TrackerStruckImpl::initImpl(const Mat& image, const Rect2d& boundingBox)
	{
		// initialize seed
		if (params.customSeed != -1)
			srand(params.customSeed);
		else
			srand(time(NULL));

		// CSC sampler
		TrackerSamplerCSC::Params CSCparameters;
		CSCparameters.initInRad = 2*params.searchRadius;
		Ptr<TrackerSamplerAlgorithm> CSCSampler = Ptr<TrackerSamplerCSC>(new TrackerSamplerCSC(CSCparameters));
		sampler->addTrackerSamplerAlgorithm(CSCSampler);

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

		Mat tmp(image.rows, image.cols, CV_8UC1);
		image.copyTo(tmp);
		Mat intImage(tmp.rows + 1, tmp.cols + 1, CV_32SC1);
		integral(tmp, intImage);

		sampler->sampling(intImage, boundingBox);
		std::vector<Mat> samples = sampler->getSamples();

		featureSet->extraction(samples);
		std::vector<Mat> resps = featureSet->getResponses();

		normalizeResps(resps);
/*		
		GenerateSystematic();

		std::vector<Mat> resps;
		Mat r;
		ExtractFeatures(intImage, samples, r);
		resps.push_back(r);
		*/
/*
		for (int i = 0; i < resps[0].rows; i++) {
			float result2 = (float)norm(resps[0].row(i));

			for (int j = 0; j < resps[0].cols; j++) {

				float f = resps[0].at<float>(i, j);
				f = f / result2;
				Mat_<float>(resps[0])(i, j) = f;
				//float result1 = f / (float) norm(resps.at(0).col(j));				

				//std::vector<Rect> areas = trackerFeature.staticCast<TrackerFeatureHAAR>()->getFeatureAt(i).getAreas();
				//float mean = trackerFeature.staticCast<TrackerFeatureHAAR>()->getFeatureAt(i).getInitMean();
				//float sigma = trackerFeature.staticCast<TrackerFeatureHAAR>()->getFeatureAt(i).getInitSigma();
				//int numAreas = trackerFeature.staticCast<TrackerFeatureHAAR>()->getFeatureAt(i).getNumAreas();
				//std::vector<float> weights = trackerFeature.staticCast<TrackerFeatureHAAR>()->getFeatureAt(i).getWeights();

			}
		}
*/

		model.staticCast<TrackerStruckModel>()->setCurrentSamples(samples);
		model.staticCast<TrackerStruckModel>()->setCurrentBoundingBox(boundingBox);
		model->getTrackerStateEstimator().staticCast<TrackerStateEstimatorStruckSVM>()->setCurrentCentre(boundingBox);
		model->modelEstimation(resps);

		model->modelUpdate();
        
        return true;
    }
    
    /* ----- UPDATE Implementation -----------------------------------------------------------*/    
    bool TrackerStruckImpl::updateImpl( const Mat& image, Rect2d& boundingBox ) 
    {
        //TrackerStruckModel* struckModel = ((TrackerStruckModel*)static_cast<TrackerModel*>(model));
        
		Ptr<TrackerTargetState> lastTargetState = model->getLastTargetState();
		Rect2d lastbb(lastTargetState->getTargetPosition().x, lastTargetState->getTargetPosition().y, lastTargetState->getTargetWidth(), lastTargetState->getTargetHeight());

		Mat tmp(image.rows, image.cols, CV_8UC1);
		image.copyTo(tmp);
		Mat intImage(tmp.rows + 1, tmp.cols + 1, CV_32SC1);
		integral(tmp, intImage);

		sampler->sampling(intImage, lastbb);
		std::vector<Mat> samples = sampler->getSamples();

		featureSet->extraction(samples);
		std::vector<Mat> resps = featureSet->getResponses();

		normalizeResps(resps);

/*
		std::vector<Mat> resps;
		Mat r;
		ExtractFeatures(intImage, samples, r);
		resps.push_back(r);
*/
		model.staticCast<TrackerStruckModel>()->setCurrentSamples(samples);
		model.staticCast<TrackerStruckModel>()->setCurrentBoundingBox(lastbb);
		model->modelEstimation(resps);

		ConfidenceMap currentMap = model.staticCast<TrackerStruckModel>()->getCurrentConfidenceMap();
		model->getTrackerStateEstimator().staticCast<TrackerStateEstimatorStruckSVM>()->setCurrentConfidenceMap(currentMap);
		model->getTrackerStateEstimator().staticCast<TrackerStateEstimatorStruckSVM>()->setCurrentCentre(boundingBox);
		if (!model->runStateEstimator()) {
			return false;
		}

        model->modelUpdate();

		Ptr<TrackerTargetState> currentState = model->getLastTargetState();
		boundingBox = Rect((int)currentState->getTargetPosition().x, (int)currentState->getTargetPosition().y, currentState->getTargetWidth(),
			currentState->getTargetHeight());
        
        return true;
    }
    
    /*
    * Parameters
    */
    TrackerStruck::Params::Params(){
        // default values of the params
        searchRadius = 20; // px
        svmC = 100.0;
        svmBudgetSize = 100;
		customSeed = -1;
    }

    void TrackerStruck::Params::read( const cv::FileNode& fn ){

    }

    void TrackerStruck::Params::write( cv::FileStorage& fs ) const{

    }

	/*
	//int TrackerStruckImpl::HaarFeature::Sum(const Mat & s, const Size & imageSize, Rect2i& rect, int channel) const
	int TrackerStruckImpl::HaarFeature::Sum(const Mat & s, const Size & imageSize, int x, int y, int width, int height) const
	{
		//assert(rect.x >= 0 && rect.y >= 0 && (rect.x + rect.width) <= imageSize.width && (rect.y + rect.height) <= imageSize.height);
		//return s.at<int>(rect.y, rect.x) +
		//	s.at<int>((rect.y + rect.height), (rect.x + rect.width)) -
		//	s.at<int>((rect.y + rect.height), rect.x) -
		//	s.at<int>(rect.y, (rect.x + rect.width));

		assert(x >= 0 && y >= 0 && (x + width) <= imageSize.width && (y + height) <= imageSize.height);
		return s.at<int>(y, x) +
			s.at<int>((y + height), (x + width)) -
			s.at<int>((y + height), x) -
			s.at<int>(y, (x + width));
	}

	TrackerStruckImpl::HaarFeature::HaarFeature(const Rect2d & bb, int type) :
		m_bb(bb)
	{
		assert(type < 6);

		switch (type)
		{
			case 0:
			{
				m_rects.push_back(Rect2f(bb.x, bb.y, bb.width, bb.height / 2));
				m_rects.push_back(Rect2f(bb.x, bb.y + bb.height / 2, bb.width, bb.height / 2));
				m_weights.push_back(1.f);
				m_weights.push_back(-1.f);
				m_factor = 255 * 1.f / 2;
				break;
			}
			case 1:
			{
				m_rects.push_back(Rect2f(bb.x, bb.y, bb.width / 2, bb.height));
				m_rects.push_back(Rect2f(bb.x + bb.width / 2, bb.y, bb.width / 2, bb.height));
				m_weights.push_back(1.f);
				m_weights.push_back(-1.f);
				m_factor = 255 * 1.f / 2;
				break;
			}
			case 2:
			{
				m_rects.push_back(Rect2f(bb.x, bb.y, bb.width / 3, bb.height));
				m_rects.push_back(Rect2f(bb.x + bb.width / 3, bb.y, bb.width / 3, bb.height));
				m_rects.push_back(Rect2f(bb.x + 2 * bb.width / 3, bb.y, bb.width / 3, bb.height));
				m_weights.push_back(1.f);
				m_weights.push_back(-2.f);
				m_weights.push_back(1.f);
				m_factor = 255 * 2.f / 3;
				break;
			}
			case 3:
			{
				m_rects.push_back(Rect2f(bb.x, bb.y, bb.width, bb.height / 3));
				m_rects.push_back(Rect2f(bb.x, bb.y + bb.height / 3, bb.width, bb.height / 3));
				m_rects.push_back(Rect2f(bb.x, bb.y + 2 * bb.height / 3, bb.width, bb.height / 3));
				m_weights.push_back(1.f);
				m_weights.push_back(-2.f);
				m_weights.push_back(1.f);
				m_factor = 255 * 2.f / 3;
				break;
			}
			case 4:
			{
				m_rects.push_back(Rect2f(bb.x, bb.y, bb.width / 2, bb.height / 2));
				m_rects.push_back(Rect2f(bb.x + bb.width / 2, bb.y + bb.height / 2, bb.width / 2, bb.height / 2));
				m_rects.push_back(Rect2f(bb.x, bb.y + bb.height / 2, bb.width / 2, bb.height / 2));
				m_rects.push_back(Rect2f(bb.x + bb.width / 2, bb.y, bb.width / 2, bb.height / 2));
				m_weights.push_back(1.f);
				m_weights.push_back(1.f);
				m_weights.push_back(-1.f);
				m_weights.push_back(-1.f);
				m_factor = 255 * 1.f / 2;
				break;
			}
			case 5:
			{
				m_rects.push_back(Rect2f(bb.x, bb.y, bb.width, bb.height));
				m_rects.push_back(Rect2f(bb.x + bb.width / 4, bb.y + bb.height / 4, bb.width / 2, bb.height / 2));
				m_weights.push_back(1.f);
				m_weights.push_back(-4.f);
				m_factor = 255 * 3.f / 4;
				break;
			}
		}
	}

	TrackerStruckImpl::HaarFeature::~HaarFeature()
	{
	}

	float TrackerStruckImpl::HaarFeature::Eval(const Mat& intImage, const Mat & s) const
	{
		Size currentSize;
		Point currentOfs;
		s.locateROI(currentSize, currentOfs);
		const Rect2d roi(currentOfs.x, currentOfs.y, s.cols, s.rows);

		float value = 0.f;
		for (int i = 0; i < (int)m_rects.size(); ++i)
		{
			const Rect2f& r = m_rects[i];
			
			//Rect2i sampleRect(
			//	(int)(roi.x + r.x*roi.width + 0.5f),
			//	(int)(roi.y + r.y*roi.height + 0.5f),
			//	(int)(r.width*roi.width), 
			//	(int)(r.height*roi.height));

			value += m_weights[i] * Sum(intImage, currentSize, 
				(int)(roi.x + r.x*roi.width + 0.5f),
				(int)(roi.y + r.y*roi.height + 0.5f),
				(int)(r.width*roi.width),
				(int)(r.height*roi.height));
		}
		return value / (m_factor*roi.area()*m_bb.area());
	}
*/
} /* namespace cv */