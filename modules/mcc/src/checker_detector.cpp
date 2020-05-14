#include "opencv2/mcc/checker_detector.hpp"
#include "opencv2/mcc/graph_cluster.hpp"
#include "opencv2/mcc/bound_min.hpp"
#include "opencv2/mcc/wiener_filter.hpp"



namespace cv{
namespace mcc{
CCheckerDetector::
CCheckerDetector(float minerr /*= 2.0*/, int nc /*= 1*/, int fsize /*= 1000*/)
	: m_fact_size( fsize )
	, m_num_ch(nc)
	, m_min_error(minerr)
{
}


CCheckerDetector::~
CCheckerDetector()
{
}


bool CCheckerDetector::
process(const std::string& pathImage)
{

	cv::Mat image;

	//-----------------------------------------------------------
	// read image
	//-----------------------------------------------------------

	// load image color
	image = cv::imread(pathImage, IMREAD_COLOR);
	if (image.empty()) return false;

	// ----------------------------------------------------------
	// process
	// ----------------------------------------------------------

	return process(image, "./out/");

}



bool CCheckerDetector::
process(const cv::Mat & image)
{

	//-------------------------------------------------------------------
	// prepare image
	//-------------------------------------------------------------------

	cv::Mat img_bgr, img_gray;  float asp;
	prepareImage(image, img_gray, img_bgr, asp, m_fact_size);

	//-------------------------------------------------------------------
	// thresholding
	//-------------------------------------------------------------------

	cv::Mat img_bw;
	int step, wndx, wndy; step = 10; wndx = 57; wndy = 57;
	performThreshold(img_gray, img_bw, wndx, wndy, step);

	//-------------------------------------------------------------------
	// find contour
	//-------------------------------------------------------------------

	ContoursVector contours;
	int minContourPointsAllowed = 1;
	findContours(img_bw, contours, minContourPointsAllowed);

	if (contours.empty())
		return false;

	//-------------------------------------------------------------------
	// find candidate
	//-------------------------------------------------------------------

	std::vector<CChart> detectedCharts;
	int minContourLengthAllowed = 100;
	findCandidates(contours, detectedCharts, minContourLengthAllowed);

	if (detectedCharts.empty())
		return false;

	//-------------------------------------------------------------------
	// clusters analysis
	//-------------------------------------------------------------------

	std::vector<int> G;
	clustersAnalysis(detectedCharts, G );

	if (G.empty())
		return false;

	//-------------------------------------------------------------------
	// checker color recognize
	//-------------------------------------------------------------------

	std::vector< std::vector<cv::Point2f > > colorCharts;
	checkerRecognize( img_bgr, detectedCharts, G, colorCharts);

	if (colorCharts.empty())
		return false;

	//-------------------------------------------------------------------
	// checker color analysis
	//-------------------------------------------------------------------

	std::vector<CChecker> checkers;
	checkerAnalysis(img_bgr, image, colorCharts, checkers, asp);
	m_checkers = checkers;

	return !m_checkers.empty();


}


#ifdef _DEBUG
bool CCheckerDetector::
process(const cv::Mat & image, const std::string &pathOut)
{

	double tic, toc, dt;


#ifdef SHOW_DEBUG_IMAGES
	printf(">> Original Image Time:	%.4f sec \n", 0);
	showAndSave("original_image", image, pathOut);
#endif


	//-------------------------------------------------------------------
	// prepare image
	//-------------------------------------------------------------------

	cv::Mat img_bgr, img_gray; float asp;

	tic = getcputime();
	prepareImage(image, img_gray, img_bgr, asp, m_fact_size);
	toc = getcputime();
	dt = toc - tic;


#ifdef SHOW_DEBUG_IMAGES
	printf(">> Prepare Image Time:	%.4f sec \n", mili2sectime(dt));
	showAndSave("prepare_image", img_gray, pathOut);
#endif


	//-------------------------------------------------------------------
	// thresholding
	//-------------------------------------------------------------------

	cv::Mat img_bw;
	int step, wndx, wndy; step = 10; wndx = 57; wndy = 57;

	tic = getcputime();
	performThreshold(img_gray, img_bw, wndx, wndy, step);
	toc = getcputime();
	dt = toc - tic;


#ifdef SHOW_DEBUG_IMAGES
	printf(">> Thresholding Time:	%.4f sec \n", mili2sectime(dt));
	showAndSave("threshold_image", img_bw, pathOut);
#endif


	//-------------------------------------------------------------------
	// find contour
	//-------------------------------------------------------------------

	ContoursVector contours;
	int minContourPointsAllowed = 1;

	tic = getcputime();
	findContours(img_bw, contours, minContourPointsAllowed);
	toc = getcputime();
	dt = toc - tic;

	if (contours.empty())
		return false;


#ifdef SHOW_DEBUG_IMAGES
	printf(">> Find Contour Time:	%.4f sec \n", mili2sectime(dt));
	cv::Mat im_contour(img_bgr.size(), CV_8UC1);
	im_contour = cv::Scalar(0);
	cv::drawContours(im_contour, contours, -1, cv::Scalar(255), 2, LINE_AA);
	showAndSave("find_contour", im_contour, pathOut);
#endif


	//-------------------------------------------------------------------
	// find candidate
	//-------------------------------------------------------------------

	std::vector<CChart> detectedCharts;
	int minContourLengthAllowed = 100;

	tic = getcputime();
	findCandidates(contours, detectedCharts, minContourLengthAllowed);
	toc = getcputime();
	dt = toc - tic;

	if (detectedCharts.empty())
		return false;

#ifdef SHOW_DEBUG_IMAGES
	std::printf(">> Find Charts Candidates Time:	%.4f sec \n", mili2sectime(dt));
	cv::Mat img_chart; img_bgr.copyTo(img_chart);


	for (size_t i = 0; i < detectedCharts.size(); i++)
	{

		CChartDraw chrtdrw( &(detectedCharts[i]), &img_chart );
		chrtdrw.drawCenter();
		chrtdrw.drawContour();

	}
	showAndSave("find_candidate", img_chart, pathOut);
#endif



	//-------------------------------------------------------------------
	// clusters analysis
	//-------------------------------------------------------------------
	std::vector<int> G;

	tic = getcputime();
	clustersAnalysis(detectedCharts, G);
	toc = getcputime();
	dt = toc - tic;

	if (G.empty())
		return false;


#ifdef SHOW_DEBUG_IMAGES
	printf(">> Clusters analysis Time:	%.4f sec \n", mili2sectime(dt));
	cv::Mat im_gru;
	img_bgr.copyTo(im_gru);
	RNG rng(0xFFFFFFFF);
	int radius = 10, thickness = -1;

	std::vector<int> g;
	unique(G, g);
	int Nc = g.size();
	std::vector<cv::Scalar> colors(Nc);
	for (size_t i = 0; i < Nc; i++) colors[i] = randomcolor(rng);

	for (size_t i = 0; i < detectedCharts.size(); i++)
	cv::circle(im_gru, detectedCharts[i].center, radius, colors[G[i]], thickness);
	showAndSave("clusters_analysis", im_gru, pathOut);
#endif

	//-------------------------------------------------------------------
	// checker color recognize
	//-------------------------------------------------------------------

	std::vector< std::vector<cv::Point2f > > colorCharts;

	tic = getcputime();
	checkerRecognize(img_bgr, detectedCharts, G, colorCharts);
	toc = getcputime();
	dt = toc - tic;

	if (colorCharts.empty()) return false;


#ifdef SHOW_DEBUG_IMAGES
	printf(">> Checker recognition Time:	%.4f sec \n", mili2sectime(dt));
	cv::Mat image_box; img_bgr.copyTo(image_box);
	for (size_t i = 0; i < colorCharts.size(); i++)
	{
		std::vector<cv::Point2f> ibox = colorCharts[i];
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

	std::vector<CChecker> checkers;

	tic = getcputime();
	checkerAnalysis(img_bgr, image, colorCharts, checkers, asp);
	toc = getcputime();
	dt = toc - tic;

#ifdef SHOW_DEBUG_IMAGES
	printf(">> Checker analysis Time: %.4f sec \n", mili2sectime(dt));
	cv::Mat image_checker;
	image.copyTo(image_checker);
	for (int ck = 0; ck < checkers.size(); ck++)
	{
		CCheckerDraw cdraw(&(checkers[ck]));
		cdraw.draw(image_checker);
	}
	showAndSave("checker_analysis", image_checker, pathOut);
#endif


	m_checkers = checkers;
	return !m_checkers.empty();

}
#endif


bool CCheckerDetector::
startTracking(
	const cv::Mat & image,
	std::vector< std::vector<cv::Point2f> >& colorCharts
	)
{

	//-------------------------------------------------------------------
	// prepare image
	//-------------------------------------------------------------------

	cv::Mat img_bgr, img_gray;  float asp;
	prepareImage(image, img_gray, img_bgr, asp, m_fact_size);

	//-------------------------------------------------------------------
	// thresholding
	//-------------------------------------------------------------------

	cv::Mat img_bw;
	int step, wndx, wndy; step = 10; wndx = 57; wndy = 57;
	performThreshold(img_gray, img_bw, wndx, wndy, step);

	//-------------------------------------------------------------------
	// find contour
	//-------------------------------------------------------------------

	ContoursVector contours;
	int minContourPointsAllowed = 1;
	findContours(img_bw, contours, minContourPointsAllowed);

	if (contours.empty())
		return false;

	//-------------------------------------------------------------------
	// find candidate
	//-------------------------------------------------------------------

	std::vector<CChart> detectedCharts;
	int minContourLengthAllowed = 100;
	findCandidates(contours, detectedCharts, minContourLengthAllowed);

	if (detectedCharts.empty())
		return false;

	//-------------------------------------------------------------------
	// clusters analysis
	//-------------------------------------------------------------------

	std::vector<int> G;
	clustersAnalysis(detectedCharts, G);

	if (G.empty())
		return false;

	//-------------------------------------------------------------------
	// checker color recognize
	//-------------------------------------------------------------------

	colorCharts.clear();
	checkerRecognize(img_bgr, detectedCharts, G, colorCharts);

	if (colorCharts.empty())
		return false;

	//-------------------------------------------------------------------
	// checker color analysis
	//-------------------------------------------------------------------

	std::vector<CChecker> checkers;
	checkerAnalysis(img_bgr, image, colorCharts, checkers, asp);
	m_checkers = checkers;

	return !m_checkers.empty();

}

bool CCheckerDetector::
continueTracking(
	const cv::Mat & image,
	const std::vector< std::vector<cv::Point2f> >& colorCharts
	)
{

	//-------------------------------------------------------------------
	// prepare image
	//-------------------------------------------------------------------

	cv::Mat img_bgr, img_gray;  float asp;
	prepareImage(image, img_gray, img_bgr, asp, m_fact_size);

	//-------------------------------------------------------------------
	// checker color analysis
	//-------------------------------------------------------------------

	std::vector<CChecker> checkers;
	checkerAnalysis(img_bgr, image, colorCharts, checkers, asp);
	m_checkers = checkers;

	return !m_checkers.empty();

}



void  CCheckerDetector::
prepareImage( const cv::Mat& bgr, cv::Mat& grayOut,
	cv::Mat &bgrOut, float &aspOut,
	int fsize
	) const
{

	int min_size;
	cv::Size size = bgr.size();
	aspOut = 1;
	bgr.copyTo(bgrOut);

	// Resize image
	min_size = std::min(size.width, size.height);
	if( fsize < min_size )
	{
		aspOut = (float)fsize / min_size;
		cv::resize(bgr, bgrOut, cv::Size(size.width*aspOut, size.height*aspOut));
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

	cv::Mat strelbox = cv::getStructuringElement(cv::MORPH_RECT, Size(5, 5), Point(-1, -1));
	cv::morphologyEx(grayOut, grayOut, MORPH_OPEN, strelbox);


}


void  CCheckerDetector::
performThreshold( const cv::Mat& grayscaleImg,
	cv::Mat& thresholdImg,
	int wndx, int wndy, int step
	)
{
	int n, m, w, h, dx, dy, indx;
	n = grayscaleImg.rows; m = grayscaleImg.cols;
	thresholdImg = cv::Mat::zeros(n, m, CV_8UC1);

	for (int y = 0; y < n / step - 1; y++)
	{
		for (int x = 0; x < m / step - 1; x++)
		{
			// get size
			w = x*step + wndx; h = y*step + wndy;
			dx = 0; dy = 0;

			//  boundary condition
			if (w >= m) dx = w - m; if (h >= n) dy = h - n;

			// get subimage
			cv::Mat Iwnd = grayscaleImg(cv::Rect(x*step, y*step, wndx - dx, wndy - dy));

			// threshold otsu
			cv::Mat It;
			cv::threshold(Iwnd, It, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

			// update
			// I_t = I_t or I_tresh
			thresholdImg(cv::Rect(x*step, y*step, wndx - dx, wndy - dy)) |= It;

		}
	}
}


void  CCheckerDetector::
findContours(
	const cv::Mat & srcImg,
	ContoursVector & contours,
	int minContourPointsAllowed
	) const
{

	// contour detected
	// [Suzuki85] Suzuki, S. and Abe, K., Topological Structural Analysis of Digitized
	// Binary Images by Border Following. CVGIP 30 1, pp 32-46 (1985)
	ContoursVector allContours;
	cv::findContours(srcImg, allContours, RETR_LIST, CHAIN_APPROX_NONE);
	static const double pi = 3.14159265358979323846;

	//select contours
	contours.clear();
	for (size_t i = 0; i < allContours.size(); i++)
	{

		PointsVector contour;
		contour = allContours[i];

		int contourSize = contour.size();
		if (contourSize <= minContourPointsAllowed)
			continue;

		double perm, area;
		perm = cv::arcLength(contour, true);
		area = cv::contourArea(contour);


		// Circularity factor condition
		// KORDECKI, A., & PALUS, H. (2014). Automatic detection of colour charts in images.
		// Przegl?d Elektrotechniczny, 90(9), 197-202.
		// 0.65 < \frac{4*pi*A}{P^2} < 0.97
		double Cf = 4 * pi*area / (perm*perm);
		if (Cf < 0.65 || Cf > 0.97) continue;


		// Soliditys
		// This measure is proposed in this work.
		PointsVector hull;
		cv::convexHull(contour, hull);
		double area_hull = cv::contourArea(hull);
		double S = area / area_hull;
		if (S < 0.90) continue;


		// Texture analysis
		// ...


		contours.push_back(allContours[i]);

	}
}


void CCheckerDetector::
findCandidates(
	const ContoursVector & contours,
	std::vector< CChart >& detectedCharts,
	int minContourLengthAllowed
	)
{
	std::vector<cv::Point>  approxCurve;
	std::vector<CChart>   possibleCharts;

	// For each contour, analyze if it is a parallelepiped likely to be the chart
	for (size_t i = 0; i < contours.size(); i++)
	{
		// Approximate to a polygon
		//  It uses the Douglas-Peucker algorithm
		// http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
		double eps = contours[i].size() * 0.05;
		cv::approxPolyDP(contours[i], approxCurve, eps, true);

		// We interested only in polygons that contains only four points
		if (approxCurve.size() != 4)
			continue;

		// And they have to be convex
		if (!cv::isContourConvex(approxCurve))
			continue;

		// Ensure that the distance between consecutive points is large enough
		float minDist = INFINITY;

		for (int i = 0; i < 4; i++)
		{
			cv::Point side = approxCurve[i] - approxCurve[(i + 1) % 4];
			float squaredSideLength = side.dot(side);
			minDist = std::min(minDist, squaredSideLength);
		}

		// Check that distance is not very small
		if (minDist < minContourLengthAllowed)
			continue;

		// All tests are passed. Save chart candidate:
		CChart chart;

		std::vector<cv::Point2f> corners(4);
		for (int i = 0; i<4; i++)
		corners[i] = cv::Point2f(approxCurve[i].x, approxCurve[i].y);
		chart.setCorners(corners);


		possibleCharts.push_back(chart);

	}

	// Remove these elements which corners are too close to each other.
	// Eliminate overlaps!!!
	// First detect candidates for removal:
	std::vector< std::pair<int, int> > tooNearCandidates;
	for (size_t i = 0;i<possibleCharts.size();i++)
	{
		const CChart& m1 = possibleCharts[i];

		//calculate the average distance of each corner to the nearest corner of the other chart candidate
		for (size_t j = i + 1;j<possibleCharts.size();j++)
		{
			const CChart& m2 = possibleCharts[j];

			float distSquared = 0;

			for (int c = 0; c < 4; c++)
			{
				cv::Point v = m1.corners[c] - m2.corners[c];
				distSquared += v.dot(v);
			}

			distSquared /= 4;

			if (distSquared < 100)
			{
				tooNearCandidates.push_back(std::pair<int, int>(i, j));
			}
		}
	}

	// Mark for removal the element of the pair with smaller perimeter
	std::vector<bool> removalMask(possibleCharts.size(), false);

	for (size_t i = 0; i<tooNearCandidates.size(); i++)
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
	for (size_t i = 0;i<possibleCharts.size();i++)
	{
		if (removalMask[i])	continue;
		detectedCharts.push_back( possibleCharts[i] );
	}


}


void CCheckerDetector::
clustersAnalysis(
	const std::vector<CChart>& detectedCharts,
	std::vector<int>& groups)
{

	int N = detectedCharts.size();
	std::vector<cv::Point> X(N);
	std::vector<float> B0(N), W(N);
	std::vector<int> G;

	CChart chart; float b0;
	for (size_t i = 0; i < N; i++)
	{
		chart = detectedCharts[i];
		b0 = chart.large_side + chart.large_side * 0.45;
		X[i] = chart.center; W[i] = chart.area; B0[i] = b0;

	}

	CB0cluster bocluster;
	bocluster.setVertex(X); bocluster.setWeight(W); bocluster.setB0(B0);
	bocluster.group();
	bocluster.getGroup( G );
	groups = G;

}


void CCheckerDetector::
checkerRecognize(
	const Mat &img,
	const std::vector<CChart>& detectedCharts,
	const std::vector<int> &G,
	std::vector< std::vector<cv::Point2f> > &colorChartsOut)
{

	std::vector<int> gU;
	unique(G, gU);
	int Nc = gU.size(); //numero de grupos
	int Ncc = detectedCharts.size(); //numero de charts

	std::vector< std::vector<cv::Point2f> > colorCharts;

	for (size_t g = 0; g < Nc; g++)
	{

		///-------------------------------------------------
		/// selecionar grupo i-esimo

		std::vector<CChart> chartSub;
		for (size_t i = 0; i < Ncc; i++)
			if (G[i] == g) chartSub.push_back(detectedCharts[i]);

		int Nsc = chartSub.size();
		if (Nsc < 4) continue;

		///-------------------------------------------------
		/// min box estimation

		CBoundMin bm;
		std::vector<cv::Point2f> points;

		bm.setCharts(chartSub);
		bm.calculate();
		bm.getCorners(points);

		// boundary condition
		if (points.size() == 0) continue;


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
			wchart += norm(v1); hchart += norm(v2);
			cx[i] = ct[i].x; cy[i] = ct[i].y;

		}

		wchart /= Nsc; hchart /= Nsc;

		///-------------------------------------------------
		/// centers and color estimate

		float tolx = wchart / 2, toly = hchart / 2;
		std::vector<float> cxr, cyr;
		reduce_array(cx, cxr, tolx); reduce_array(cy, cyr, toly);


		// color and center rectificate
		cv::Size2i colorSize = cv::Size2i(cxr.size(), cyr.size());
		cv::Mat colorMat(colorSize, CV_32FC3);
		std::vector<cv::Point2f> cte(colorSize.area());


		int k = 0;
		for (size_t i = 0; i < colorSize.height; i++)
		{
			for (size_t j = 0; j < colorSize.width; j++)
			{
				cv::Point2f vc = cv::Point2f(cxr[j], cyr[i]);
				cte[k] = vc;

				// recovery color
				cv::Point2f cti;
				cv::Matx31f p, xt;

				p(0, 0) = vc.x; p(1, 0) = vc.y; p(2, 0) = 1;
				xt = ccT.inv()*p;
				cti.x = xt(0, 0) / xt(2, 0); cti.y = xt(1, 0) / xt(2, 0);

				// color
				int x, y;
				x = cti.x; y = cti.y;
				Vec3f &srgb = colorMat.at<Vec3f>(i, j);
				Vec3b rgb = img.at<Vec3b>(y, x);

				srgb[0] = (float)rgb[0]/255;
				srgb[1] = (float)rgb[1]/255;
				srgb[2] = (float)rgb[2]/255;

				k++;
			}
		}

		CChartClassicModel::SUBCCMModel scm;
		scm.centers = cte;
		scm.color_size = colorSize;
		colorMat = colorMat.t();
		scm.sub_chart = colorMat.reshape(3, colorSize.area());


		///-------------------------------------------------
		/// Estimate model chart ColorChecker� Classic

		if (cte.size() > 24) continue;

		// color chart classic model
		CChartClassicModel cccm;

		int iTheta;  // rotation angle of chart
		int offset;  // offset
		float error; // min error
		if (!cccm.evaluate(scm, offset, iTheta, error))
			continue;

		if (iTheta == 1 || iTheta == 3) // 90 | 270
			cccm.rotate90();

		///-------------------------------------------------
		/// calculate coordanate

		cv::Size2i dim = cccm.size;
		std::vector<cv::Point2f> center = cccm.center;
		std::vector<cv::Point2f>  box = cccm.box;
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
				int iter = i*dim.height + j;
				ctss[p] = center[iter];
				point_ac += ctss[p];
				p++;
			}
		}


		// is colineal point
		if (point_ac.x == ctss[0].x*p || point_ac.y == ctss[0].y*p)
			continue;

		// Find the perspective transformation
		cv::Matx33f ccTe = cv::findHomography(ctss, cte);

		std::vector<cv::Point2f> tbox, ibox;
		transform_points_forward(ccTe, box, tbox);
		transform_points_inverse(ccT, tbox, ibox);

		// sort the points in anti-clockwise order
		polyanticlockwise(ibox);
		circshift(ibox, 4 - iTheta);
		colorCharts.push_back(ibox);


	}

	// return
	colorChartsOut = colorCharts;


}


void CCheckerDetector::
checkerAnalysis(
	const cv::Mat &img,
	const cv::Mat &img_org,
	std::vector< std::vector< cv::Point2f > > colorCharts,
	std::vector<CChecker> &checkers,
	float asp
	)
{

	int N, sl;
	std::vector< cv::Point2f > ibox;

	N = colorCharts.size();
	std::vector< float > J(N);
	for (size_t i = 0; i < N; i++)
	{
		ibox = colorCharts[i];
		J[i] = cost_function(img, ibox);
	}
    std::vector<int> idx;
	sort(J,idx);

	float invAsp = 1 / asp;
	int n = cv::min(m_num_ch, N);
	CChecker checker;
	checkers.clear();

	for (size_t i = 0; i < n ; i++)
	{
		ibox = colorCharts[idx[i]];
		if (J[i] > m_min_error)
			continue;

		// redimention box
		for (size_t i = 0; i < 4; i++)
			ibox[i] = invAsp*ibox[i];

		cv::Mat charts_rgb, charts_ycbcr;
		get_profile(img_org, ibox, charts_rgb, charts_ycbcr);

		// result
		checker.target = CChecker::MCC24;
		checker.N = 24;
		checker.box = ibox;
		checker.charts_rgb = charts_rgb;
		checker.charts_ycbcr = charts_ycbcr;
		checker.center = mace_center(ibox);
		checker.cost = J[i];

		checkers.push_back(checker);
	}


}

void CCheckerDetector::
get_subbox_chart_physical(const std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& chartPhy, cv::Size & size)
{

	float w, h;
	cv::Point2f v1 = points[1] - points[0];
	cv::Point2f v2 = points[3] - points[0];
	float asp = norm(v2) / norm(v1);

	w = 100;
	h = floor(100 * asp + 0.5);

	chartPhy.clear(); chartPhy.resize(4);
	chartPhy[0] = cv::Point2f(0, 0);
	chartPhy[1] = cv::Point2f(w, 0);
	chartPhy[2] = cv::Point2f(w, h);
	chartPhy[3] = cv::Point2f(0, h);

	size = cv::Size(w, h);

}

void CCheckerDetector::
reduce_array(const std::vector<float>& x, std::vector<float>& x_new, float tol)
{

	int n = x.size(), nn;
	std::vector<float> xx = x;
	x_new.clear();

	// sort array
	std::sort(xx.begin(), xx.end());

	// label array
	std::vector<int> label(n);
	for (int i = 0; i < n; i++)
		label[i] = abs(xx[(n + i - 1) % n] - xx[i]) > tol;

	// diff array
	for (int i = 1, j; i < n; i++)  label[i] += label[i - 1];

	// unique array
	std::vector<int> ulabel;
	unique(label, ulabel);

	// mean for group
	nn = ulabel.size(); x_new.resize(nn);
	for (int i = 0; i < nn; i++)
	{
		float mu = 0, s = 0;
		for (int j = 0; j < n; j++)
		{
			mu += (label[j] == ulabel[i])*xx[j];
			s += (label[j] == ulabel[i]);
		}
		x_new[i] = mu / s;
	}

	// diff array
	std::vector<float> dif(nn - 1);
	for (int i = 0, j; i < nn - 1; i++)
		dif[i] = (x_new[(i + 1) % nn] - x_new[i]);

	// max and idx
	float fmax = 0; int idx = 0;
	for (int i = 0; i < nn - 1; i++)
		if (fmax < dif[i]) { fmax = dif[i];  idx = i; }

	// add ... X[i] MAX X[i+] ...
	if (fmax > 4 * tol)
		x_new.insert(x_new.begin() + idx + 1, (x_new[idx] + x_new[idx + 1]) / 2);


}

void CCheckerDetector::
transform_points_forward(const cv::Matx33f & T, const std::vector<cv::Point2f>& X, std::vector<cv::Point2f>& Xt)
{

	int N = X.size();
	if (N == 0) return;

	Xt.clear(); Xt.resize(N);
	cv::Matx31f p, xt;
	cv::Point2f pt;

	for (int i = 0; i < N; i++)
	{
		p(0, 0) = X[i].x; p(1, 0) = X[i].y; p(2, 0) = 1;
		xt = T*p;
		pt.x = xt(0, 0) / xt(2, 0); pt.y = xt(1, 0) / xt(2, 0);
		Xt[i] = pt;
	}
}

void CCheckerDetector::
transform_points_inverse(const cv::Matx33f & T, const std::vector<cv::Point2f>& X, std::vector<cv::Point2f>& Xt)
{
	cv::Matx33f Tinv = T.inv();
	transform_points_forward(Tinv, X, Xt);
}

float CCheckerDetector::
cost_function(const cv::Mat & img, const std::vector<cv::Point2f>& ibox)
{

	// color chart classic model
	CChartClassicModel cccm;
	cv::Mat lab; float J = 0; int N;
	std::vector<cv::Point2f>  fbox = cccm.box;
	std::vector<cv::Point2f> cellchart = cccm.cellchart;

	cccm.copyToColorMat(lab, 0);
	lab = lab.reshape(3, lab.size().area());

	cv::Mat3f im_lab, im_rgb(img);
	//cv::cvtColor(im_rgb, im_lab, COLOR_BGR2Lab);
	cv::cvtColor(im_rgb, im_lab, COLOR_BGR2RGB);

	lab /= 255; im_lab /= 255;

	// tranformation
	Matx33f ccT = cv::getPerspectiveTransform(fbox, ibox);

	cv::Mat mask;
	std::vector<cv::Point2f> bch(4), bcht(4);
	N = cellchart.size() / 4;
	cv::Point3f r;

	float ec = 0, es = 0;
	for (size_t i = 0, k; i < N; i++)
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
		for (size_t j = 0; j < 4; j++) 	c += bcht[j];  c /= 4;
		for (size_t j = 0; j < 4; j++) 	bcht[j] = ((bcht[j] - c)*0.75) + c;

		cv::Scalar mu, st;
		mask = poly2mask(bcht, img.size());
		cv::meanStdDev(im_lab, mu, st, mask);

		// cos error
		float costh;
		costh = mu.dot(cv::Scalar(r)) / (norm(mu)*norm(r) + FLT_EPSILON );
		ec += (1 - (1 + costh) / 2);

		// standar desviation
		es += st.dot(st);

	}

	// J = arg min ec + es
	J = ec + es;
	return J;
}

}
}
