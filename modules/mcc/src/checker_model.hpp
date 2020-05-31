#ifndef _MCC_CHECKER_MODEL_H
#define _MCC_CHECKER_MODEL_H

#include "core.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>


namespace cv{
namespace mcc{


	/** CChartClassicModel
	  *
	  * @brief ColorChecker� Classic
	  * @note

	  * @author Pedro Marrero Fern�ndez
	  */

    class CChartModel
	{

	public:

		typedef struct
		_SUBCCMModel {

			cv::Mat sub_chart;
			cv::Size2i color_size;
			std::vector<cv::Point2f> centers;

		}SUBCCMModel;

	public:

		CChartModel(const int chartType);
		~CChartModel();

		/** @brief evaluate submodel in this checker type*/
		bool evaluate(const SUBCCMModel &subModel, int &offset, int &iTheta, float &error);

		// function utils

		void copyToColorMat(cv::Mat &lab, int cs = 0);
		void rotate90();
		void flip();

	public:

		// Cie L*a*b* values use illuminant D50 2 degree observer sRGB values for
		// for iluminante D65.

		cv::Size2i size;
		cv::Size2f boxsize;
		std::vector<cv::Point2f> box;
		std::vector<cv::Point2f> cellchart;
		std::vector<cv::Point2f> center;
		std::vector<std::vector<float> > chart;

	protected:

		/** \brief match checker color
		  * \param[in] subModel sub-checker
		  * \param[in] iTheta angle
		  * \param[out] error
		  * \param[out] ierror
		  * \return state
		  */
		bool match(const SUBCCMModel &subModel, int iTheta, float &error, int &ierror);

		/** \brief euclidian dist L2 for Lab space
		  * \note
		  * \f$ \sum_i \sqrt (\sum_k (ab1-ab2)_k.^2) \f$
		  * \param[in] lab1
		  * \param[in] lab2
		  * \return distance
		  */
		float dist_color_lab(const cv::Mat &lab1, const cv::Mat &lab2);

		/** \brief rotate matrix 90 grado */
		void rot90(cv::Mat &mat, int itheta);
	};
	/** CChecker
	  *
	  * \brief checker model
	  * \author Pedro Marrero Fernandez
	  *
	  */
	class CCheckerImpl:public CChecker
	{	public:


		typedef
		enum _TYPECHAR
		{
			MCC24 = 0,
			SG140,
			PASSPORT

		}TYPECHRT;

	public:

		CCheckerImpl(): target(MCC24), N(24) {}
		~CCheckerImpl() {}



	public:
		TYPECHRT target;				// o tipo de checkercolor
		int N;							// number of charts
		std::vector< cv::Point2f > box;	// corner box
		cv::Mat charts_rgb;				// charts profile rgb color space
		cv::Mat charts_ycbcr;			// charts profile YCbCr color space
		float cost;						// cost to aproximate
		cv::Point2f center;


	};

	//////////////////////////////////////////////////////////////////////////////////////////////
	// CheckerDraw

	/** \brief checker draw
	  * \author Pedro Marrero Fernandez
	  */
	class CCheckerDrawImpl : public CCheckerDraw
	{

	public:
		CCheckerDrawImpl(Ptr<CChecker> pChecker, cv::Scalar color = CV_RGB(0, 250, 0), int thickness = 2)
			: m_pChecker(pChecker), m_color(color), m_thickness(thickness)
		{
			assert(pChecker);
		}

		void draw(cv::Mat &img, const int chartType) CV_OVERRIDE;

	private:
		Ptr<CChecker> m_pChecker;
		cv::Scalar m_color;
		int m_thickness;

	private:
		/** \brief transformation perspetive*/
		void transform_points_forward(
			const cv::Matx33f &T,
			const std::vector<cv::Point2f> &X,
			std::vector<cv::Point2f> &Xt);
	};


	} // namespace mcc
	} // namespace cv

#endif //_MCC_CHECKER_MODEL_H
