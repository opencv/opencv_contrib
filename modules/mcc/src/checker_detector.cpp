#include "opencv2/mcc/checker_detector.hpp"
#include "opencv2/mcc/checker_model.hpp"

#include "opencv2/dnn.hpp"

#include "checker_detector.hpp"

#include "graph_cluster.hpp"
#include "bound_min.hpp"
#include "wiener_filter.hpp"

#include "checker_model.hpp"

#include <iostream>
#include <mutex>          // std::mutex


namespace cv
{
namespace mcc
{

std::mutex mtx;           // mutex for critical section

Ptr<CCheckerDetector> CCheckerDetector::create()
{
	return makePtr<CCheckerDetectorImpl>();
}

CCheckerDetectorImpl::
	CCheckerDetectorImpl()

{
}

CCheckerDetectorImpl::~CCheckerDetectorImpl()
{
}

bool CCheckerDetectorImpl::
	setNet(cv::dnn::Net _net)
{
	net = _net;
	return !net.empty();
}

bool CCheckerDetectorImpl::
	_no_net_process(const cv::Mat &image, const TYPECHART chartType, const int nc, const Ptr<DetectorParameters> &params, std::vector<cv::Rect> regionsOfInterest)
{
	m_checkers.clear();
	this->net_used=false;
	for (cv::Rect region : regionsOfInterest)
	{
		//-------------------------------------------------------------------
		// Run the model to find good regions
		//-------------------------------------------------------------------
		cv::Mat croppedImage = image(region);
#ifdef SHOW_DEBUG_IMAGES
				string pathOut = "./";
#endif
		//-------------------------------------------------------------------
		// prepare image
		//-------------------------------------------------------------------

		cv::Mat img_bgr, img_gray;
		float asp;
		prepareImage(croppedImage, img_gray, img_bgr, asp, params);

		//-------------------------------------------------------------------
		// thresholding
		//-------------------------------------------------------------------
		std::vector<cv::Mat> img_bw;
		performThreshold(img_gray, img_bw, params);
		parallel_for_(Range(0, (int)img_bw.size()), [&](const Range &range) {
			const int begin = range.start;
			const int end = range.end;
			for (int i = begin; i < end; i++)
			{

#ifdef SHOW_DEBUG_IMAGES
				showAndSave("threshold_image", img_bw[i], pathOut);
#endif
				// find contour
				//-------------------------------------------------------------------
				ContoursVector contours;
				findContours(img_bw[i], contours, params);

				if (contours.empty())
					continue;
#ifdef SHOW_DEBUG_IMAGES
						cv::Mat im_contour(img_bgr.size(), CV_8UC1);
						im_contour = cv::Scalar(0);
						cv::drawContours(im_contour, contours, -1, cv::Scalar(255), 2, LINE_AA);
						showAndSave("find_contour", im_contour, pathOut);
#endif
				//-------------------------------------------------------------------
				// find candidate
				//-------------------------------------------------------------------

				std::vector<CChart> detectedCharts;
				findCandidates(contours, detectedCharts, params);


				if (detectedCharts.empty())
					continue;

#ifdef SHOW_DEBUG_IMAGES
						std::printf(">> Number of detected Charts Candidates :%lu", detectedCharts.size());
						cv::Mat img_chart;
						img_bgr.copyTo(img_chart);

						for (size_t ind = 0; ind < detectedCharts.size(); ind++)
						{

							CChartDraw chrtdrw(&(detectedCharts[ind]), &img_chart);
							chrtdrw.drawCenter();
							chrtdrw.drawContour();
						}
						showAndSave("find_candidate", img_chart, pathOut);
#endif
				//-------------------------------------------------------------------
				// clusters analysis
				//-------------------------------------------------------------------

				std::vector<int> G;
				clustersAnalysis(detectedCharts, G, params);

				if (G.empty())
					continue;

#ifdef SHOW_DEBUG_IMAGES
						cv::Mat im_gru;
						img_bgr.copyTo(im_gru);
						RNG rng(0xFFFFFFFF);
						int radius = 10, thickness = -1;

						std::vector<int> g;
						unique(G, g);
						size_t Nc = g.size();
						std::vector<cv::Scalar> colors(Nc);
						for (size_t ind = 0; ind < Nc; ind++)
							colors[ind] = randomcolor(rng);

						for (size_t ind = 0; ind < detectedCharts.size(); ind++)
							cv::circle(im_gru, detectedCharts[ind].center, radius, colors[G[ind]], thickness);
						showAndSave("clusters_analysis", im_gru, pathOut);
#endif
				//-------------------------------------------------------------------
				// checker color recognize
				//-------------------------------------------------------------------

				std::vector<std::vector<cv::Point2f>> colorCharts;
				checkerRecognize(img_bgr, detectedCharts, G, chartType, colorCharts, params);

				if (colorCharts.empty())
					continue;

#ifdef SHOW_DEBUG_IMAGES
						cv::Mat image_box;
						img_bgr.copyTo(image_box);
						for (size_t ind = 0; ind < colorCharts.size(); ind++)
						{
							std::vector<cv::Point2f> ibox = colorCharts[ind];
							cv::Scalar color_box = CV_RGB(0, 0, 255);
							int thickness_box = 2;
							cv::line(image_box, ibox[0], ibox[1], color_box, thickness_box, LINE_AA);
							cv::line(image_box, ibox[1], ibox[2], color_box, thickness_box, LINE_AA);
							cv::line(image_box, ibox[2], ibox[3], color_box, thickness_box, LINE_AA);
							cv::line(image_box, ibox[3], ibox[0], color_box, thickness_box, LINE_AA);
							//cv::circle(image_box, ibox[0], 10, cv::Scalar(0, 0, 255), 3);
							//cv::circle(image_box, ibox[1], 10, cv::Scalar(0, 255, 0), 3);
						}
						showAndSave("checker_recognition", image_box, pathOut);
#endif
				//-------------------------------------------------------------------
				// checker color analysis
				//-------------------------------------------------------------------
				std::vector<Ptr<CChecker>> checkers;
				checkerAnalysis(img_bgr, image, chartType, nc, colorCharts, checkers, asp, params);

#ifdef SHOW_DEBUG_IMAGES
						cv::Mat image_checker;
						croppedImage.copyTo(image_checker);
						for (size_t ck = 0; ck < checkers.size(); ck++)
						{
							Ptr<CCheckerDraw> cdraw = CCheckerDraw::create((checkers[ck]));
							cdraw->draw(image_checker);
						}
						showAndSave("checker_analysis", image_checker, pathOut);
#endif
				for (Ptr<CChecker> checker : checkers)
				{
					for (cv::Point2f &corner : checker->box)
						corner += static_cast<cv::Point2f>(region.tl());

					mtx.lock(); // push_back is not thread safe
					m_checkers.push_back(checker);
					mtx.unlock();
				}
			}

		});
	}
	//remove too close detections
	removeTooCloseDetections(params);

	return !m_checkers.empty();
}

bool CCheckerDetectorImpl::
	process(const cv::Mat &image, const TYPECHART chartType, const int nc /*= 1*/,bool useNet/*=false*/, const Ptr<DetectorParameters> &params /* = DetectorParameters::create()*/, std::vector<cv::Rect> regionsOfInterest /*= std::vector<cv::Rect>()*/)
{

	m_checkers.clear();

	if (regionsOfInterest.empty())
	{
		regionsOfInterest.push_back(cv::Rect(0, 0, image.size[1], image.size[0]));
	}
	if (this->net.empty() || !useNet)
	{
		return _no_net_process(image, chartType, nc, params, regionsOfInterest);
	}
	this->net_used=true;
	for (cv::Rect region : regionsOfInterest)
	{
		//-------------------------------------------------------------------
		// Run the model to find good regions
		//-------------------------------------------------------------------

		cv::Mat croppedImage = image(region);

		int rows = croppedImage.size[0];
		int cols = croppedImage.size[1];
		net.setInput(cv::dnn::blobFromImage(croppedImage, 1.0, cv::Size(), cv::Scalar(), true));
		cv::Mat output = net.forward();

		Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

		for (int i = 0; i < detectionMat.rows; i++)
		{
			float confidence = detectionMat.at<float>(i, 2);
			if (confidence > params->confidenceThreshold)
			{
				float xTopLeft = max(0.0f, detectionMat.at<float>(i, 3) * cols - params->borderWidth);
				float yTopLeft = max(0.0f, detectionMat.at<float>(i, 4) * rows - params->borderWidth);
				float xBottomRight = min((float)cols - 1, detectionMat.at<float>(i, 5) * cols + params->borderWidth);
				float yBottomRight = min((float)rows - 1, detectionMat.at<float>(i, 6) * rows + params->borderWidth);

				cv::Point2f topLeft = {xTopLeft, yTopLeft};
				cv::Point2f bottomRight = {xBottomRight, yBottomRight};

				cv::Rect innerRegion(topLeft, bottomRight);
				cv::Mat innerCroppedImage = croppedImage(innerRegion);

#ifdef SHOW_DEBUG_IMAGES
				string pathOut = "./";
#endif
				//-------------------------------------------------------------------
				// prepare image
				//-------------------------------------------------------------------

				cv::Mat img_bgr, img_gray;
				float asp;
				prepareImage(innerCroppedImage, img_gray, img_bgr, asp, params);

				//-------------------------------------------------------------------
				// thresholding
				//-------------------------------------------------------------------

				std::vector<cv::Mat> img_bw;
				performThreshold(img_gray, img_bw, params);
				parallel_for_(Range(0, (int)img_bw.size()), [&](const Range &range) {
					const int begin = range.start;
					const int end = range.end;

					for (int ind = begin; ind < end; ind++)
					{

#ifdef SHOW_DEBUG_IMAGES
				showAndSave("threshold_image", img_bw[ind], pathOut);
#endif
						//------------------------------------------------------------------
						// find contour
						//-------------------------------------------------------------------
						ContoursVector contours;
						findContours(img_bw[ind], contours, params);

						if (contours.empty())
							continue;
#ifdef SHOW_DEBUG_IMAGES
						cv::Mat im_contour(img_bgr.size(), CV_8UC1);
						im_contour = cv::Scalar(0);
						cv::drawContours(im_contour, contours, -1, cv::Scalar(255), 2, LINE_AA);
						showAndSave("find_contour", im_contour, pathOut);
#endif
						//-------------------------------------------------------------------
						// find candidate
						//-------------------------------------------------------------------

						std::vector<CChart> detectedCharts;
						findCandidates(contours, detectedCharts, params);

						if (detectedCharts.empty())
							continue;

#ifdef SHOW_DEBUG_IMAGES
						std::printf(">> Number of detected Charts Candidates :%lu", detectedCharts.size());
						cv::Mat img_chart;
						img_bgr.copyTo(img_chart);

						for (size_t index = 0; index < detectedCharts.size(); index++)
						{

							CChartDraw chrtdrw(&(detectedCharts[index]), &img_chart);
							chrtdrw.drawCenter();
							chrtdrw.drawContour();
						}
						showAndSave("find_candidate", img_chart, pathOut);
#endif
						//-------------------------------------------------------------------
						// clusters analysis
						//-------------------------------------------------------------------

						std::vector<int> G;
						clustersAnalysis(detectedCharts, G, params);

						if (G.empty())
							continue;
#ifdef SHOW_DEBUG_IMAGES
						cv::Mat im_gru;
						img_bgr.copyTo(im_gru);
						RNG rng(0xFFFFFFFF);
						int radius = 10, thickness = -1;

						std::vector<int> g;
						unique(G, g);
						size_t Nc = g.size();
						std::vector<cv::Scalar> colors(Nc);
						for (size_t index = 0; index < Nc; index++)
							colors[index] = randomcolor(rng);

						for (size_t index = 0; index < detectedCharts.size(); index++)
							cv::circle(im_gru, detectedCharts[index].center, radius, colors[G[index]], thickness);
						showAndSave("clusters_analysis", im_gru, pathOut);
#endif

						//-------------------------------------------------------------------
						// checker color recognize
						//-------------------------------------------------------------------

						std::vector<std::vector<cv::Point2f>> colorCharts;
						checkerRecognize(img_bgr, detectedCharts, G, chartType, colorCharts, params);

						if (colorCharts.empty())
							continue;

#ifdef SHOW_DEBUG_IMAGES
						cv::Mat image_box;
						img_bgr.copyTo(image_box);
						for (size_t index = 0; index < colorCharts.size(); index++)
						{
							std::vector<cv::Point2f> ibox = colorCharts[index];
							cv::Scalar color_box = CV_RGB(0, 0, 255);
							int thickness_box = 2;
							cv::line(image_box, ibox[0], ibox[1], color_box, thickness_box, LINE_AA);
							cv::line(image_box, ibox[1], ibox[2], color_box, thickness_box, LINE_AA);
							cv::line(image_box, ibox[2], ibox[3], color_box, thickness_box, LINE_AA);
							cv::line(image_box, ibox[3], ibox[0], color_box, thickness_box, LINE_AA);
							//cv::circle(image_box, ibox[0], 10, cv::Scalar(0, 0, 255), 3);
							//cv::circle(image_box, ibox[1], 10, cv::Scalar(0, 255, 0), 3);
						}
						showAndSave("checker_recognition", image_box, pathOut);
#endif
						//-------------------------------------------------------------------
						// checker color analysis
						//-------------------------------------------------------------------
						std::vector<Ptr<CChecker>> checkers;
						checkerAnalysis(img_bgr, image, chartType, nc, colorCharts, checkers, asp, params);
#ifdef SHOW_DEBUG_IMAGES
						cv::Mat image_checker;
						innerCroppedImage.copyTo(image_checker);
						for (size_t ck = 0; ck < checkers.size(); ck++)
						{
							Ptr<CCheckerDraw> cdraw = CCheckerDraw::create((checkers[ck]));
							cdraw->draw(image_checker);
						}
						showAndSave("checker_analysis", image_checker, pathOut);
#endif
						for (Ptr<CChecker> checker : checkers)
						{
							for (cv::Point2f &corner : checker->box)
								corner += static_cast<cv::Point2f>(region.tl() + innerRegion.tl());
							mtx.lock(); // push_back is not thread safe
							m_checkers.push_back(checker);
							mtx.unlock();
						}
					}
				});
			}
		}
	}
	// As a failsafe try the classical method
	if (m_checkers.empty())
	{
		return _no_net_process(image, chartType, nc, params, regionsOfInterest);
	}
	//remove too close detections
	removeTooCloseDetections(params);
	return !m_checkers.empty();
}

void CCheckerDetectorImpl::
	prepareImage(const cv::Mat &bgr, cv::Mat &grayOut,
				 cv::Mat &bgrOut, float &aspOut,
				 const Ptr<DetectorParameters> &params) const
{

	int min_size;
	cv::Size size = bgr.size();
	aspOut = 1;
	bgr.copyTo(bgrOut);

	// Resize image
	min_size = std::min(size.width, size.height);
	if (params->minImageSize > min_size)
	{
		aspOut = (float)params->minImageSize / min_size;
		cv::resize(bgr, bgrOut, cv::Size(int(size.width * aspOut), int(size.height * aspOut)));
	}

	// Convert to grayscale
	cv::cvtColor(bgrOut, grayOut, COLOR_BGR2GRAY);

	// PDiamel: wiener adaptative methods to minimize the noise effets
	// by illumination

	CWienerFilter filter;
	filter.wiener2(grayOut, grayOut, 5, 5);

	//JLeandro: perform morphological open on the equalized image
	//to minimize the noise effects by CLAHE and to even intensities
	//inside the MCC patches (regions)

	cv::Mat strelbox = cv::getStructuringElement(cv::MORPH_RECT, Size(5, 5));
	cv::morphologyEx(grayOut, grayOut, MORPH_OPEN, strelbox);
}

void CCheckerDetectorImpl::
	performThreshold(const cv::Mat &grayscaleImg,
					 std::vector<cv::Mat> &thresholdImgs,
					 const Ptr<DetectorParameters> &params) const
{
	// number of window sizes (scales) to apply adaptive thresholding
	int nScales = (params->adaptiveThreshWinSizeMax - params->adaptiveThreshWinSizeMin) /
					  params->adaptiveThreshWinSizeStep + 1;
		for (int i = 0; i < nScales; i++)
		{
			int currScale = params->adaptiveThreshWinSizeMin + i * params->adaptiveThreshWinSizeStep;

			cv::Mat tempThresholdImg;
			cv::adaptiveThreshold(grayscaleImg, tempThresholdImg, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, currScale, params->adaptiveThreshConstant);
			thresholdImgs.push_back(tempThresholdImg);
		}
}

void CCheckerDetectorImpl::
	findContours(
		const cv::Mat &srcImg,
		ContoursVector &contours,
		const Ptr<DetectorParameters> &params) const
{
	// contour detected
	// [Suzuki85] Suzuki, S. and Abe, K., Topological Structural Analysis of Digitized
	// Binary Images by Border Following. CVGIP 30 1, pp 32-46 (1985)
	ContoursVector allContours;
	cv::findContours(srcImg, allContours, RETR_LIST, CHAIN_APPROX_NONE);

	//select contours
	contours.clear();

	const long long int srcImgArea = srcImg.size[0] * srcImg.size[1];
	for (size_t i = 0; i < allContours.size(); i++)
	{

		PointsVector contour;
		contour = allContours[i];

		int contourSize = (int)contour.size();
		if (contourSize <= params->minContourPointsAllowed)
			continue;

		double area = cv::contourArea(contour);
		// double perm = cv::arcLength(contour, true);

		if (this->net_used && area/srcImgArea < params->minContoursAreaRate)
			continue;

		if (!this->net_used && area < params->minContoursArea)
			continue;
		// Circularity factor condition
		// KORDECKI, A., & PALUS, H. (2014). Automatic detection of colour charts in images.
		// Przegl?d Elektrotechniczny, 90(9), 197-202.
		// 0.65 < \frac{4*pi*A}{P^2} < 0.97
		// double Cf = 4 * CV_PI * area / (perm * perm);
		// if (Cf < 0.5 || Cf > 0.97) continue;

		// Soliditys
		// This measure is proposed in this work.
		PointsVector hull;
		cv::convexHull(contour, hull);
		double area_hull = cv::contourArea(hull);
		double S = area / area_hull;
		if (S < params->minContourSolidity)
			continue;

		// Texture analysis
		// ...

		contours.push_back(allContours[i]);
	}
}

void CCheckerDetectorImpl::
	findCandidates(
		const ContoursVector &contours,
		std::vector<CChart> &detectedCharts,
		const Ptr<DetectorParameters> &params)
{
	std::vector<cv::Point> approxCurve;
	std::vector<CChart> possibleCharts;

	// For each contour, analyze if it is a parallelepiped likely to be the chart
	for (size_t i = 0; i < contours.size(); i++)
	{
		// Approximate to a polygon
		//  It uses the Douglas-Peucker algorithm
		// http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
		double eps = contours[i].size() * params->findCandidatesApproxPolyDPEpsMultiplier;
		cv::approxPolyDP(contours[i], approxCurve, eps, true);

		// We interested only in polygons that contains only four points
		if (approxCurve.size() != 4)
			continue;

		// And they have to be convex
		if (!cv::isContourConvex(approxCurve))
			continue;

		// Ensure that the distance between consecutive points is large enough
		float minDist = INFINITY;

		for (size_t j = 0; j < 4; j++)
		{
			cv::Point side = approxCurve[j] - approxCurve[(j + 1) % 4];
			float squaredSideLength = (float)side.dot(side);
			minDist = std::min(minDist, squaredSideLength);
		}

		// Check that distance is not very small
		if (minDist < params->minContourLengthAllowed)
			continue;

		// All tests are passed. Save chart candidate:
		CChart chart;

		std::vector<cv::Point2f> corners(4);
		for (int j = 0; j < 4; j++)
			corners[j] = cv::Point2f((float)approxCurve[j].x, (float)approxCurve[j].y);
		chart.setCorners(corners);

		possibleCharts.push_back(chart);
	}

	// Remove these elements which corners are too close to each other.
	// Eliminate overlaps!!!
	// First detect candidates for removal:
	std::vector<std::pair<int, int>> tooNearCandidates;
	for (int i = 0; i < (int) possibleCharts.size(); i++)
	{
		const CChart &m1 = possibleCharts[i];

		//calculate the average distance of each corner to the nearest corner of the other chart candidate
		for (int j = i + 1; j < (int) possibleCharts.size(); j++)
		{
			const CChart &m2 = possibleCharts[j];

			float distSquared = 0;

			for (int c = 0; c < 4; c++)
			{
				cv::Point v = m1.corners[c] - m2.corners[c];
				distSquared += v.dot(v);
			}

			distSquared /= 4;

			if (distSquared < params->minInterContourDistance)
			{
				tooNearCandidates.push_back(std::pair<int, int>(i, j));
			}
		}
	}

	// Mark for removal the element of the pair with smaller perimeter
	std::vector<bool> removalMask(possibleCharts.size(), false);

	for (size_t i = 0; i < tooNearCandidates.size(); i++)
	{
		float p1 = perimeter(possibleCharts[tooNearCandidates[i].first].corners);
		float p2 = perimeter(possibleCharts[tooNearCandidates[i].second].corners);

		size_t removalIndex;
		if (p1 > p2)
			removalIndex = tooNearCandidates[i].second;
		else
			removalIndex = tooNearCandidates[i].first;

		removalMask[removalIndex] = true;
	}

	// Return candidates
	detectedCharts.clear();
	for (size_t i = 0; i < possibleCharts.size(); i++)
	{
		if (removalMask[i])
			continue;
		detectedCharts.push_back(possibleCharts[i]);
	}
}

void CCheckerDetectorImpl::
	clustersAnalysis(
		const std::vector<CChart> &detectedCharts,
		std::vector<int> &groups,
		const Ptr<DetectorParameters> &params)
{
	size_t N = detectedCharts.size();
	std::vector<cv::Point> X(N);
	std::vector<double> B0(N), W(N);
	std::vector<int> G;

	CChart chart;
	double b0;
	for (size_t i = 0; i < N; i++)
	{
		chart = detectedCharts[i];
		b0 = chart.large_side * params->B0factor;
		X[i] = chart.center;
		W[i] = chart.area;
		B0[i] = b0;
	}

	CB0cluster bocluster;
	bocluster.setVertex(X);
	bocluster.setWeight(W);
	bocluster.setB0(B0);
	bocluster.group();
	bocluster.getGroup(G);
	groups = G;
}

void CCheckerDetectorImpl::
	checkerRecognize(
		const Mat &img,
		const std::vector<CChart> &detectedCharts,
		const std::vector<int> &G,
		const TYPECHART chartType,
		std::vector<std::vector<cv::Point2f>> &colorChartsOut,
		const Ptr<DetectorParameters> &params)
{
	std::vector<int> gU;
	unique(G, gU);
	size_t Nc = gU.size();				//numero de grupos
	size_t Ncc = detectedCharts.size(); //numero de charts

	std::vector<std::vector<cv::Point2f>> colorCharts;

	for (size_t g = 0; g < Nc; g++)
	{

		///-------------------------------------------------
		/// selecionar grupo i-esimo

		std::vector<CChart> chartSub;
		for (size_t i = 0; i < Ncc; i++)
			if (G[i] == (int)g)
				chartSub.push_back(detectedCharts[i]);

		size_t Nsc = chartSub.size();
		if (Nsc < params->minGroupSize)
			continue;

		///-------------------------------------------------
		/// min box estimation

		CBoundMin bm;
		std::vector<cv::Point2f> points;

		bm.setCharts(chartSub);
		bm.calculate();
		bm.getCorners(points);

		// boundary condition
		if (points.size() == 0)
			continue;

		// sort the points in anti-clockwise order
		polyanticlockwise(points);

		///-------------------------------------------------
		/// box projective transformation

		// get physical char box model
		std::vector<cv::Point2f> chartPhy;
		cv::Size size_box_phy;
		get_subbox_chart_physical(points, chartPhy, size_box_phy);

		// Find the perspective transformation that brings current chart to rectangular form
		Matx33f ccT = cv::getPerspectiveTransform(points, chartPhy);

		// transformer
		std::vector<cv::Point2f> c(Nsc), ct;
		std::vector<cv::Point2f> ch(4 * Nsc), cht;

		for (size_t i = 0; i < Nsc; i++)
		{

			CChart cc = chartSub[i];
			for (size_t j = 0; j < 4; j++)
				ch[i * 4 + j] = cc.corners[j];
			c[i] = chartSub[i].center;
		}

		transform_points_forward(ccT, c, ct);
		transform_points_forward(ccT, ch, cht);

		float wchart = 0, hchart = 0;
		std::vector<float> cx(Nsc), cy(Nsc);
		for (size_t i = 0, k = 0; i < Nsc; i++)
		{
			k = i * 4;
			cv::Point2f v1 = cht[k + 1] - cht[k + 0];
			cv::Point2f v2 = cht[k + 3] - cht[k + 0];
			wchart += (float)norm(v1);
			hchart += (float)norm(v2);
			cx[i] = ct[i].x;
			cy[i] = ct[i].y;
		}

		wchart /= Nsc;
		hchart /= Nsc;

		///-------------------------------------------------
		/// centers and color estimate

		float tolx = wchart / 2, toly = hchart / 2;
		std::vector<float> cxr, cyr;
		reduce_array(cx, cxr, tolx);
		reduce_array(cy, cyr, toly);

		if (cxr.size() == 1 || cyr.size() == 1) //no information can be extracted if only one row or columns in present
			continue;
		// color and center rectificate
		cv::Size2i colorSize = cv::Size2i((int)cxr.size(), (int)cyr.size());
		cv::Mat colorMat(colorSize, CV_32FC3);
		std::vector<cv::Point2f> cte(colorSize.area());

		int k = 0;

		for (int i = 0; i < colorSize.height; i++)
		{
			for (int j = 0; j < colorSize.width; j++)
			{
				cv::Point2f vc = cv::Point2f(cxr[j], cyr[i]);
				cte[k] = vc;

				// recovery color
				cv::Point2f cti;
				cv::Matx31f p, xt;

				p(0, 0) = vc.x;
				p(1, 0) = vc.y;
				p(2, 0) = 1;
				xt = ccT.inv() * p;
				cti.x = xt(0, 0) / xt(2, 0);
				cti.y = xt(1, 0) / xt(2, 0);

				// color
				int x, y;
				x = (int)cti.x;
				y = (int)cti.y;
				Vec3f &srgb = colorMat.at<Vec3f>(i, j);
				Vec3b rgb;
				if (0 <= y && y < img.rows && 0 <= x && x < img.cols)
					rgb = img.at<Vec3b>(y, x);

				srgb[0] = (float)rgb[0] / 255;
				srgb[1] = (float)rgb[1] / 255;
				srgb[2] = (float)rgb[2] / 255;

				k++;
			}
		}

		CChartModel::SUBCCMModel scm;
		scm.centers = cte;
		scm.color_size = colorSize;
		colorMat = colorMat.t();
		scm.sub_chart = colorMat.reshape(3, colorSize.area());

		///-------------------------------------------------

		// color chart model
		CChartModel cccm(chartType);

		int iTheta;	 // rotation angle of chart
		int offset;	 // offset
		float error; // min error
		if (!cccm.evaluate(scm, offset, iTheta, error))
			continue;
		if (iTheta >= 4)
			cccm.flip();

		for (int i = 0; i < iTheta % 4; i++)
			cccm.rotate90();

		///-------------------------------------------------
		/// calculate coordanate

		cv::Size2i dim = cccm.size;
		std::vector<cv::Point2f> center = cccm.center;
		std::vector<cv::Point2f> box = cccm.box;
		int cols = dim.height - colorSize.width + 1;

		int x = (offset) / cols;
		int y = (offset) % cols;

		// seleccionar sub grid centers of model
		std::vector<cv::Point2f> ctss(colorSize.area());
		cv::Point2f point_ac = cv::Point2f(0, 0);
		int p = 0;

		for (int i = x; i < (x + colorSize.height); i++)
		{
			for (int j = y; j < (y + colorSize.width); j++)
			{
				int iter = i * dim.height + j;
				ctss[p] = center[iter];
				point_ac += ctss[p];
				p++;
			}
		}
		// is colineal point
		if (point_ac.x == ctss[0].x * p || point_ac.y == ctss[0].y * p)
			continue;
		// Find the perspective transformation
		cv::Matx33f ccTe = cv::findHomography(ctss, cte);

		std::vector<cv::Point2f> tbox, ibox;
		transform_points_forward(ccTe, box, tbox);
		transform_points_inverse(ccT, tbox, ibox);

		// sort the points in anti-clockwise order
		if (iTheta < 4)
			mcc::polyanticlockwise(ibox);
		else
			mcc::polyclockwise(ibox);
		// circshift(ibox, 4 - iTheta);
		colorCharts.push_back(ibox);
	}

	// return
	colorChartsOut = colorCharts;
}

void CCheckerDetectorImpl::
	checkerAnalysis(
		const cv::Mat &img,
		const cv::Mat &img_org,
		const TYPECHART chartType,
		const unsigned int nc,
		std::vector<std::vector<cv::Point2f>> colorCharts,
		std::vector<Ptr<CChecker>> &checkers,
		float asp,
		const Ptr<DetectorParameters> &params)
{
	size_t N;
	std::vector<cv::Point2f> ibox;

	N = colorCharts.size();
	std::vector<float> J(N);
	for (size_t i = 0; i < N; i++)
	{
		ibox = colorCharts[i];
		J[i] = cost_function(img, ibox, chartType);
	}

	std::vector<int> idx;
	sort(J, idx);
	float invAsp = 1 / asp;
	size_t n = cv::min(nc, (unsigned)N);
	checkers.clear();

	for (size_t i = 0; i < n; i++)
	{
		ibox = colorCharts[idx[i]];

		if (J[i] > params->maxError)
			continue;

		// redimention box
		for (size_t j = 0; j < 4; j++)
			ibox[j] = invAsp * ibox[j];

		cv::Mat charts_rgb, charts_ycbcr;
		get_profile(img_org, ibox, chartType, charts_rgb, charts_ycbcr);

		// result
		Ptr<CChecker> checker = CChecker::create();
		checker->box = ibox;
		checker->target = chartType;
		checker->charts_rgb = charts_rgb;
		checker->charts_ycbcr = charts_ycbcr;
		checker->center = mace_center(ibox);
		checker->cost = J[i];

		checkers.push_back(checker);
	}
}

void CCheckerDetectorImpl::
	removeTooCloseDetections(const Ptr<DetectorParameters> &params)
{
	// Remove these elements which corners are too close to each other.
	// Eliminate overlaps!!!
	// First detect candidates for removal:
	std::vector<std::pair<int, int>> tooNearCandidates;
	for (int i = 0; i < (int) m_checkers.size(); i++)
	{
		const Ptr<CChecker> &m1 = m_checkers[i];

		//calculate the average distance of each corner to the nearest corner of the other chart candidate
		for (int j = i + 1; j < (int) m_checkers.size(); j++)
		{
			const Ptr<CChecker> &m2 = m_checkers[j];

			float distSquared = 0;

			for (int c = 0; c < 4; c++)
			{
				cv::Point v = m1->box[c] - m2->box[c];
				distSquared += v.dot(v);
			}

			distSquared /= 4;

			if (distSquared < params->minInterCheckerDistance)
			{
				tooNearCandidates.push_back(std::pair<int, int>(i, j));
			}
		}
	}

	// Mark for removal the element of the pair with smaller cost
	std::vector<bool> removalMask(m_checkers.size(), false);

	for (size_t i = 0; i < tooNearCandidates.size(); i++)
	{
		float p1 = m_checkers[tooNearCandidates[i].first]->cost;
		float p2 = m_checkers[tooNearCandidates[i].second]->cost;

		size_t removalIndex;
		if (p1 < p2 )
			removalIndex = tooNearCandidates[i].second;
		else
			removalIndex = tooNearCandidates[i].first;

		removalMask[removalIndex] = true;
	}

	std::vector<Ptr<CChecker>> copy_m_checkers = m_checkers;
	m_checkers.clear();

	for (size_t i = 0; i < copy_m_checkers.size(); i++)
	{
		if (removalMask[i])
			continue;
		m_checkers.push_back(copy_m_checkers[i]);
	}
}

void CCheckerDetectorImpl::
	get_subbox_chart_physical(const std::vector<cv::Point2f> &points, std::vector<cv::Point2f> &chartPhy, cv::Size &size)
{
	float w, h;
	cv::Point2f v1 = points[1] - points[0];
	cv::Point2f v2 = points[3] - points[0];
	float asp = (float) (norm(v2) / norm(v1));

	w = 100;
	h = (float)floor(100 * asp + 0.5);

	chartPhy.clear();
	chartPhy.resize(4);
	chartPhy[0] = cv::Point2f(0, 0);
	chartPhy[1] = cv::Point2f(w, 0);
	chartPhy[2] = cv::Point2f(w, h);
	chartPhy[3] = cv::Point2f(0, h);

	size = cv::Size((int)w, (int)h);
}

void CCheckerDetectorImpl::
	reduce_array(const std::vector<float> &x, std::vector<float> &x_new, float tol)
{
	size_t n = x.size(), nn;
	std::vector<float> xx = x;
	x_new.clear();

	// sort array
	std::sort(xx.begin(), xx.end());

	// label array
	std::vector<int> label(n);
	for (size_t i = 0; i < n; i++)
		label[i] = abs(xx[(n + i - 1) % n] - xx[i]) > tol;

	// diff array
	for (size_t i = 1; i < n; i++)
		label[i] += label[i - 1];

	// unique array
	std::vector<int> ulabel;
	unique(label, ulabel);

	// mean for group
	nn = ulabel.size();
	x_new.resize(nn);
	for (size_t i = 0; i < nn; i++)
	{
		float mu = 0, s = 0;
		for (size_t j = 0; j < n; j++)
		{
			mu += (label[j] == ulabel[i]) * xx[j];
			s += (label[j] == ulabel[i]);
		}
		x_new[i] = mu / s;
	}

	// diff array
	std::vector<float> dif(nn - 1);
	for (size_t i = 0; i < nn - 1; i++)
		dif[i] = (x_new[(i + 1) % nn] - x_new[i]);

	// max and idx
	float fmax = 0;
	size_t idx = 0;
	for (size_t i = 0; i < nn - 1; i++)
		if (fmax < dif[i])
		{
			fmax = dif[i];
			idx = i;
		}

	// add ... X[i] MAX X[i+] ...
	if (fmax > 4 * tol)
		x_new.insert(x_new.begin() + idx + 1, (x_new[idx] + x_new[idx + 1]) / 2);
}

void CCheckerDetectorImpl::
	transform_points_forward(const cv::Matx33f &T, const std::vector<cv::Point2f> &X, std::vector<cv::Point2f> &Xt)
{
	size_t N = X.size();
	if (N == 0)
		return;

	Xt.clear();
	Xt.resize(N);
	cv::Matx31f p, xt;
	cv::Point2f pt;

	for (int i = 0; i < (int)N; i++)
	{
		p(0, 0) = X[i].x;
		p(1, 0) = X[i].y;
		p(2, 0) = 1;
		xt = T * p;
		pt.x = xt(0, 0) / xt(2, 0);
		pt.y = xt(1, 0) / xt(2, 0);
		Xt[i] = pt;
	}
}

void CCheckerDetectorImpl::
	transform_points_inverse(const cv::Matx33f &T, const std::vector<cv::Point2f> &X, std::vector<cv::Point2f> &Xt)
{
	cv::Matx33f Tinv = T.inv();
	transform_points_forward(Tinv, X, Xt);
}
void CCheckerDetectorImpl::
	get_profile(
		const cv::Mat &img,
		const std::vector<cv::Point2f> &ibox,
		const TYPECHART chartType,
		cv::Mat &charts_rgb,
		cv::Mat &charts_ycbcr)
{
	// color chart classic model
	CChartModel cccm(chartType);
	cv::Mat lab;
	size_t N;
	std::vector<cv::Point2f> fbox = cccm.box;
	std::vector<cv::Point2f> cellchart = cccm.cellchart;
	cv::Mat3f im_rgb, im_ycbcr, im_bgr(img);
	cv::Mat rgb[3], ycbcr[3];

	// Convert to RGB and YCbCr space
	cv::cvtColor(im_bgr, im_rgb, COLOR_BGR2RGB);
	cv::cvtColor(im_bgr, im_ycbcr, COLOR_BGR2YCrCb);

	// Get chanels
	split(im_rgb, rgb);
	split(im_ycbcr, ycbcr);

	// tranformation
	Matx33f ccT = cv::getPerspectiveTransform(fbox, ibox);

	cv::Mat mask;
	std::vector<cv::Point2f> bch(4), bcht(4);
	N = cellchart.size() / 4;

	// Create table charts information
	//		  |p_size|average|stddev|max|min|
	//	RGB   |      |       |      |   |   |
	//  YCbCr |

	charts_rgb = cv::Mat(cv::Size(5, 3 * (int)N), CV_64F);
	charts_ycbcr = cv::Mat(cv::Size(5, 3 * (int)N), CV_64F);

	cv::Scalar mu_rgb, st_rgb, mu_ycb, st_ycb, p_size;
	double max_rgb[3], min_rgb[3], max_ycb[3], min_ycb[3];

	for (int i = 0, k; i < (int)N; i++)
	{
		k = 4 * i;
		bch[0] = cellchart[k + 0];
		bch[1] = cellchart[k + 1];
		bch[2] = cellchart[k + 2];
		bch[3] = cellchart[k + 3];
		polyanticlockwise(bch);
		transform_points_forward(ccT, bch, bcht);

		cv::Point2f c(0, 0);
		for (int j = 0; j < 4; j++)
			c += bcht[j];
		c /= 4;
		for (size_t j = 0; j < 4; j++)
			bcht[j] = ((bcht[j] - c) * 0.50) + c;

		mask = poly2mask(bcht, img.size());
		p_size = cv::sum(mask);

		// rgb space
		cv::meanStdDev(im_rgb, mu_rgb, st_rgb, mask);
		cv::minMaxLoc(rgb[0], &min_rgb[0], &max_rgb[0], NULL, NULL, mask);
		cv::minMaxLoc(rgb[1], &min_rgb[1], &max_rgb[1], NULL, NULL, mask);
		cv::minMaxLoc(rgb[2], &min_rgb[2], &max_rgb[2], NULL, NULL, mask);

		// create tabla
		//|p_size|average|stddev|max|min|
		// raw_r
		charts_rgb.at<double>(3 * i + 0, 0) = p_size(0);
		charts_rgb.at<double>(3 * i + 0, 1) = mu_rgb(0);
		charts_rgb.at<double>(3 * i + 0, 2) = st_rgb(0);
		charts_rgb.at<double>(3 * i + 0, 3) = min_rgb[0];
		charts_rgb.at<double>(3 * i + 0, 4) = max_rgb[0];
		// raw_g
		charts_rgb.at<double>(3 * i + 1, 0) = p_size(0);
		charts_rgb.at<double>(3 * i + 1, 1) = mu_rgb(1);
		charts_rgb.at<double>(3 * i + 1, 2) = st_rgb(1);
		charts_rgb.at<double>(3 * i + 1, 3) = min_rgb[1];
		charts_rgb.at<double>(3 * i + 1, 4) = max_rgb[1];
		// raw_b
		charts_rgb.at<double>(3 * i + 2, 0) = p_size(0);
		charts_rgb.at<double>(3 * i + 2, 1) = mu_rgb(2);
		charts_rgb.at<double>(3 * i + 2, 2) = st_rgb(2);
		charts_rgb.at<double>(3 * i + 2, 3) = min_rgb[2];
		charts_rgb.at<double>(3 * i + 2, 4) = max_rgb[2];

		// YCbCr space
		cv::meanStdDev(im_ycbcr, mu_ycb, st_ycb, mask);
		cv::minMaxLoc(ycbcr[0], &min_ycb[0], &max_ycb[0], NULL, NULL, mask);
		cv::minMaxLoc(ycbcr[1], &min_ycb[1], &max_ycb[1], NULL, NULL, mask);
		cv::minMaxLoc(ycbcr[2], &min_ycb[2], &max_ycb[2], NULL, NULL, mask);

		// create tabla
		//|p_size|average|stddev|max|min|
		// raw_Y
		charts_ycbcr.at<double>(3 * i + 0, 0) = p_size(0);
		charts_ycbcr.at<double>(3 * i + 0, 1) = mu_ycb(0);
		charts_ycbcr.at<double>(3 * i + 0, 2) = st_ycb(0);
		charts_ycbcr.at<double>(3 * i + 0, 3) = min_ycb[0];
		charts_ycbcr.at<double>(3 * i + 0, 4) = max_ycb[0];
		// raw_Cb
		charts_ycbcr.at<double>(3 * i + 1, 0) = p_size(0);
		charts_ycbcr.at<double>(3 * i + 1, 1) = mu_ycb(1);
		charts_ycbcr.at<double>(3 * i + 1, 2) = st_ycb(1);
		charts_ycbcr.at<double>(3 * i + 1, 3) = min_ycb[1];
		charts_ycbcr.at<double>(3 * i + 1, 4) = max_ycb[1];
		// raw_Cr
		charts_ycbcr.at<double>(3 * i + 2, 0) = p_size(0);
		charts_ycbcr.at<double>(3 * i + 2, 1) = mu_ycb(2);
		charts_ycbcr.at<double>(3 * i + 2, 2) = st_ycb(2);
		charts_ycbcr.at<double>(3 * i + 2, 3) = min_ycb[2];
		charts_ycbcr.at<double>(3 * i + 2, 4) = max_ycb[2];
	}
}
float CCheckerDetectorImpl::
	cost_function(const cv::Mat &img, const std::vector<cv::Point2f> &ibox, const TYPECHART chartType)
{
	// color chart classic model
	CChartModel cccm(chartType);
	cv::Mat lab;
	float J = 0;
	size_t N;
	std::vector<cv::Point2f> fbox = cccm.box;
	std::vector<cv::Point2f> cellchart = cccm.cellchart;

	cccm.copyToColorMat(lab, 0);
	lab = lab.reshape(3, lab.size().area());

	cv::Mat3f im_lab, im_rgb(img);
	//cv::cvtColor(im_rgb, im_lab, COLOR_BGR2Lab);
	cv::cvtColor(im_rgb, im_lab, COLOR_BGR2RGB);

	lab /= 255;
	im_lab /= 255;

	// tranformation
	Matx33f ccT = cv::getPerspectiveTransform(fbox, ibox);

	cv::Mat mask;
	std::vector<cv::Point2f> bch(4), bcht(4);
	N = cellchart.size() / 4;

	float ec = 0, es = 0;
	for (int i = 0, k; i < (int)N; i++)
	{

		cv::Vec3f r = lab.at<cv::Vec3f>(i);

		k = 4 * i;
		bch[0] = cellchart[k + 0];
		bch[1] = cellchart[k + 1];
		bch[2] = cellchart[k + 2];
		bch[3] = cellchart[k + 3];
		polyanticlockwise(bch);
		transform_points_forward(ccT, bch, bcht);

		cv::Point2f c(0, 0);
		for (int j = 0; j < 4; j++)
			c += bcht[j];
		c /= 4;
		for (int j = 0; j < 4; j++)
			bcht[j] = ((bcht[j] - c) * 0.75) + c;

		cv::Scalar mu, st;
		mask = poly2mask(bcht, img.size());
		cv::meanStdDev(im_lab, mu, st, mask);

		// cos error
		float costh;
		costh = (float)(mu.dot(cv::Scalar(r)) / (norm(mu) * norm(r) + FLT_EPSILON));
		ec += (1 - (1 + costh) / 2);

		// standar desviation
		es += (float)st.dot(st);
	}

	// J = arg min ec + es
	J = ec + es;
	return J/N;
}

} // namespace cv
} // namespace cv
