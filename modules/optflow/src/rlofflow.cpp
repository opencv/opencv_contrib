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
 // Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
#include "rlof/rlof_localflow.h"
#include "rlof/geo_interpolation.hpp"
#include "opencv2/ximgproc.hpp"
namespace cv
{ 
	namespace optflow 
	{
				
		class DenseOpticalFlowRLOFImpl : public DenseRLOFOpticalFlow
		{
		public:
			DenseOpticalFlowRLOFImpl()
				: forwardBackwardThreshold(1.f)
				, gridStep(6, 6)
				, interp_type(InterpolationType::INTERP_GEO)
				, k(128)
				, sigma(0.05f)
				, lambda(999.f)
				, fgs_lambda(500.0f)
				, fgs_sigma(1.5f)
				, use_post_proc(true)

			{
				prevPyramid[0] = cv::Ptr< CImageBuffer>(new CImageBuffer);
				prevPyramid[1] = cv::Ptr< CImageBuffer>(new CImageBuffer);
				currPyramid[0] = cv::Ptr< CImageBuffer>(new CImageBuffer);
				currPyramid[1] = cv::Ptr< CImageBuffer>(new CImageBuffer);
			}
			virtual void setRLOFOpticalFlowParameter(RLOFOpticalFlowParameter val) CV_OVERRIDE { param = val; }
			virtual RLOFOpticalFlowParameter getRLOFOpticalFlowParameter() const CV_OVERRIDE { return param; }

			virtual float getForwardBackward() const CV_OVERRIDE { return forwardBackwardThreshold; }
			virtual void setForwardBackward(float val) CV_OVERRIDE { forwardBackwardThreshold = val; }

			virtual void setInterpolation(InterpolationType val) CV_OVERRIDE { interp_type = val; }
			virtual InterpolationType getInterpolation() const CV_OVERRIDE { return interp_type; }

			virtual Size getGridStep() const CV_OVERRIDE { return gridStep; }
			virtual void setGridStep(Size val) CV_OVERRIDE { gridStep = val; }

			virtual int getEPICK() const CV_OVERRIDE { return k; }
			virtual void setEPICK(int val) CV_OVERRIDE { k = val; }

			virtual float getEPICSigma() const CV_OVERRIDE { return sigma; }
			virtual void setEPICSigma(float val) CV_OVERRIDE { sigma = val; }

			virtual float getEPICLambda() const CV_OVERRIDE { return lambda; }
			virtual void setEPICLambda(float val)  CV_OVERRIDE { lambda = val; }

			virtual float getFgsLambda() const CV_OVERRIDE { return fgs_lambda; }
			virtual void setFgsLambda(float val) CV_OVERRIDE { fgs_lambda = val; }

			virtual float getFgsSigma() const CV_OVERRIDE { return fgs_sigma; }
			virtual void setFgsSigma(float val) CV_OVERRIDE { fgs_sigma = val; }

			virtual bool getUsePostProc() const CV_OVERRIDE { return use_post_proc; }
			virtual void setUsePostProc(bool val) CV_OVERRIDE { use_post_proc = val; }

			virtual void calc(InputArray I0, InputArray I1, InputOutputArray flow) CV_OVERRIDE 
			{
				CV_Assert(!I0.empty() && I0.depth() == CV_8U && (I0.channels() == 3 || I0.channels() == 1));
				CV_Assert(!I1.empty() && I1.depth() == CV_8U && (I1.channels() == 3 || I1.channels() == 1));
				CV_Assert(I0.sameSize(I1));
				if (param.supportRegionType == RLOFOpticalFlowParameter::SR_CROSS)
					CV_Assert( I0.channels() == 3 && I1.channels() == 3);
				CV_Assert(interp_type == InterpolationType::INTERP_EPIC || interp_type == InterpolationType::INTERP_GEO);

				Mat prevImage = I0.getMat();
				Mat currImage = I1.getMat();
				int noPoints = prevImage.cols * prevImage.rows;
				std::vector<cv::Point2f> prevPoints(noPoints);
				std::vector<cv::Point2f> currPoints, refPoints;
				noPoints = 0;
				cv::Size grid_h = gridStep / 2;
				for (int r = grid_h.height; r < prevImage.rows; r += gridStep.height)
				{
					for (int c = grid_h.width; c < prevImage.cols; c += gridStep.width)
					{
						prevPoints[noPoints++] = cv::Point2f(static_cast<float>(c), static_cast<float>(r));
					}
				}
				prevPoints.erase(prevPoints.begin() + noPoints, prevPoints.end());
				currPoints.resize(prevPoints.size());

				calcLocalOpticalFlow(prevImage, currImage, prevPyramid, currPyramid, prevPoints, currPoints, param);

				flow.create(prevImage.size(), CV_32FC2);
				Mat dense_flow = flow.getMat();

				std::vector<Point2f> filtered_prevPoints;
				std::vector<Point2f> filtered_currPoints;
				if (gridStep == cv::Size(1, 1) && forwardBackwardThreshold <= 0)
				{
					for (unsigned int n = 0; n < prevPoints.size(); n++)
					{
						dense_flow.at<Point2f>(prevPoints[n]) = currPoints[n] - prevPoints[n];
					}
					return;
				}
				if (forwardBackwardThreshold > 0)
				{
					// reuse image pyramids
					calcLocalOpticalFlow(currImage, prevImage, currPyramid, prevPyramid, currPoints, refPoints, param);

					filtered_prevPoints.resize(prevPoints.size());
					filtered_currPoints.resize(prevPoints.size());
					float sqrForwardBackwardThreshold = forwardBackwardThreshold * forwardBackwardThreshold;
					noPoints = 0;
					for (unsigned int r = 0; r < refPoints.size(); r++)
					{
						Point2f diff = refPoints[r] - prevPoints[r];
						if (diff.x * diff.x + diff.y * diff.y < sqrForwardBackwardThreshold)
						{
							filtered_prevPoints[noPoints] = prevPoints[r];
							filtered_currPoints[noPoints++] = currPoints[r];
						}
					}

					filtered_prevPoints.erase(filtered_prevPoints.begin() + noPoints, filtered_prevPoints.end());
					filtered_currPoints.erase(filtered_currPoints.begin() + noPoints, filtered_currPoints.end());

				}
				else
				{
					filtered_prevPoints = prevPoints;
					filtered_currPoints = currPoints;
				}

				if (interp_type == InterpolationType::INTERP_EPIC)
				{
					Ptr<ximgproc::EdgeAwareInterpolator> gd = ximgproc::createEdgeAwareInterpolator();
					gd->setK(k);
					gd->setSigma(sigma);
					gd->setLambda(lambda);
					gd->setUsePostProcessing(false);
					gd->interpolate(prevImage, filtered_prevPoints, currImage, filtered_currPoints, dense_flow);
				}
				else
				{
					Mat blurredPrevImage, blurredCurrImage;
					GaussianBlur(prevImage, blurredPrevImage, cv::Size(5, 5), -1);
					std::vector<uchar> status(filtered_currPoints.size(), 1);
					interpolate_irregular_nn_raster(filtered_prevPoints, filtered_currPoints, status, blurredPrevImage).copyTo(dense_flow);
					std::vector<Mat> vecMats;
					std::vector<Mat> vecMats2(2);
					cv::split(dense_flow, vecMats);
					cv::bilateralFilter(vecMats[0], vecMats2[0], 5, 2, 20);
					cv::bilateralFilter(vecMats[1], vecMats2[1], 5, 2, 20);
					cv::merge(vecMats2, dense_flow);
				}
				if (use_post_proc)
				{
					ximgproc::fastGlobalSmootherFilter(prevImage, flow, flow, fgs_lambda, fgs_sigma);
				}
			}
			
			virtual void collectGarbage() CV_OVERRIDE
			{
				prevPyramid[0].release();
				prevPyramid[1].release();
				currPyramid[0].release();
				currPyramid[1].release();
			}

		protected:
			RLOFOpticalFlowParameter		param;
			float				forwardBackwardThreshold;
			Ptr<CImageBuffer>	prevPyramid[2];
			Ptr<CImageBuffer>	currPyramid[2];
			cv::Size			gridStep;
			InterpolationType   interp_type;
			int					k;
			float				sigma;
			float				lambda;
			float				fgs_lambda;
			float				fgs_sigma;
			bool				use_post_proc;
		};
	
		Ptr<DenseRLOFOpticalFlow> DenseRLOFOpticalFlow::create(
			const RLOFOpticalFlowParameter & rlofParam,
			float forwardBackwardThreshold,
			cv::Size gridStep,
			InterpolationType interp_type,
			int epicK,
			float epicSigma,
			float epicLambda,
			float fgs_lambda,
			float fgs_sigma)
		{
			Ptr<DenseRLOFOpticalFlow> algo = makePtr<DenseOpticalFlowRLOFImpl>();
			algo->setRLOFOpticalFlowParameter(rlofParam);
			algo->setForwardBackward(forwardBackwardThreshold);
			algo->setGridStep(gridStep);
			algo->setInterpolation(interp_type);
			algo->setEPICK(epicK);
			algo->setEPICSigma(epicSigma);
			algo->setEPICLambda(epicLambda);
			algo->setFgsLambda(fgs_lambda);
			algo->setFgsSigma(fgs_sigma);
			return algo;
		}

		class SparseRLOFOpticalFlowImpl : public SparseRLOFOpticalFlow
		{
			public:
			SparseRLOFOpticalFlowImpl()
				: forwardBackwardThreshold(1.f)
			{
				prevPyramid[0] = cv::Ptr< CImageBuffer>(new CImageBuffer);
				prevPyramid[1] = cv::Ptr< CImageBuffer>(new CImageBuffer);
				currPyramid[0] = cv::Ptr< CImageBuffer>(new CImageBuffer);
				currPyramid[1] = cv::Ptr< CImageBuffer>(new CImageBuffer);
			}
			virtual void setRLOFOpticalFlowParameter(RLOFOpticalFlowParameter val) CV_OVERRIDE  { param = val; }
			virtual RLOFOpticalFlowParameter getRLOFOpticalFlowParameter() const CV_OVERRIDE { return param; }

			virtual float getForwardBackward()  const CV_OVERRIDE { return forwardBackwardThreshold; }
			virtual void setForwardBackward(float val) CV_OVERRIDE { forwardBackwardThreshold = val; }

			virtual void calc(InputArray prevImg, InputArray nextImg,
				InputArray prevPts, InputOutputArray nextPts,
				OutputArray status,
				OutputArray err) CV_OVERRIDE
			{
				CV_Assert(!prevImg.empty() && prevImg.depth() == CV_8U && (prevImg.channels() == 3 || prevImg.channels() == 1));
				CV_Assert(!nextImg.empty() && nextImg.depth() == CV_8U && (nextImg.channels() == 3 || nextImg.channels() == 1));
				CV_Assert(prevImg.sameSize(nextImg));

				//CV_Assert(param.supportRegionType == RLOFOpticalFlowParameter::SR_CROSS && I0.channels() == 3 && I1.channels() == 3);

				Mat prevImage = prevImg.getMat();
				Mat nextImage = nextImg.getMat();
				Mat prevPtsMat = prevPts.getMat();

				if (param.useInitialFlow == false)
					nextPts.create(prevPtsMat.size(), prevPtsMat.type(), -1, true);

				int npoints = 0;
				CV_Assert((npoints = prevPtsMat.checkVector(2, CV_32F, true)) >= 0);

				if (npoints == 0)
				{
					nextPts.release();
					status.release();
					err.release();
					return;
				}

				Mat nextPtsMat = nextPts.getMat();
				CV_Assert(nextPtsMat.checkVector(2, CV_32F, true) == npoints);
				std::vector<cv::Point2f> prevPoints(npoints), nextPoints(npoints), refPoints;
				prevPtsMat.copyTo(cv::Mat(1, npoints, CV_32FC2, &prevPoints[0]));
				if (param.useInitialFlow )
					nextPtsMat.copyTo(cv::Mat(1, nextPtsMat.cols, CV_32FC2, &nextPoints[0]));

				cv::Mat statusMat;
				cv::Mat errorMat;
				if (status.needed() || forwardBackwardThreshold > 0)
				{
					status.create((int)npoints, 1, CV_8U, -1, true);
					statusMat = status.getMat();
					statusMat.setTo(1);
				}

				if (err.needed() || forwardBackwardThreshold > 0)
				{
					err.create((int)npoints, 1, CV_32F, -1, true);
					errorMat = err.getMat();
					errorMat.setTo(0);
				}

				calcLocalOpticalFlow(prevImage, nextImage, prevPyramid, currPyramid, prevPoints, nextPoints, param);
				cv::Mat(1,npoints , CV_32FC2, &nextPoints[0]).copyTo(nextPtsMat);
				if (forwardBackwardThreshold > 0)
				{
					// reuse image pyramids
					calcLocalOpticalFlow(nextImage, prevImage, currPyramid, prevPyramid, nextPoints, refPoints, param);
				}
				for (unsigned int r = 0; r < refPoints.size(); r++)
				{
					Point2f diff = refPoints[r] - prevPoints[r];
					errorMat.at<float>(r) = sqrt(diff.x * diff.x + diff.y * diff.y);
					if (errorMat.at<float>(r) <= forwardBackwardThreshold)
						statusMat.at<uchar>(r) = 0;
				}

			}

		protected:
			RLOFOpticalFlowParameter	param;
			float				forwardBackwardThreshold;
			Ptr<CImageBuffer>	prevPyramid[2];
			Ptr<CImageBuffer>	currPyramid[2];
		};
		
		Ptr<SparseRLOFOpticalFlow> SparseRLOFOpticalFlow::create(
			const RLOFOpticalFlowParameter & rlofParam,
			float forwardBackwardThreshold)
		{
			Ptr<SparseRLOFOpticalFlow> algo = makePtr<SparseRLOFOpticalFlowImpl>();
			algo->setRLOFOpticalFlowParameter(rlofParam);
			algo->setForwardBackward(forwardBackwardThreshold);
			return algo;
		}
		
		void calcOpticalFlowDenseRLOF(InputArray I0, InputArray I1, InputOutputArray flow,
			RLOFOpticalFlowParameter rlofParam ,
			float forewardBackwardThreshold, Size gridStep, 
			DenseRLOFOpticalFlow::InterpolationType interp_type,
			int epicK, float epicSigma, 
			bool use_post_proc, float fgsLambda, float fgsSigma)
		{
			Ptr<DenseRLOFOpticalFlow> algo = DenseRLOFOpticalFlow::create(
				rlofParam, forewardBackwardThreshold, gridStep, interp_type,
				epicK, epicSigma, use_post_proc, fgsLambda, fgsSigma);
			algo->calc(I0, I1, flow);
			algo->collectGarbage();
		}

		void calcOpticalFlowSparseRLOF(InputArray prevImg, InputArray nextImg,
			InputArray prevPts, InputOutputArray nextPts,
			OutputArray status, OutputArray err,
			RLOFOpticalFlowParameter rlofParam,
			float forewardBackwardThreshold)
		{
			Ptr<SparseRLOFOpticalFlow> algo = SparseRLOFOpticalFlow::create(
				rlofParam, forewardBackwardThreshold);
			algo->calc(prevImg, nextImg, prevPts, nextPts, status, err);
		}
		Ptr<DenseOpticalFlow> createOptFlow_DenseRLOF()
		{
			return DenseRLOFOpticalFlow::create();
		}

		Ptr<SparseOpticalFlow> createOptFlow_SparseRLOF()
		{
			return SparseRLOFOpticalFlow::create();
		}
	}


}