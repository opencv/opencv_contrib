#include "common.hpp"

namespace cv{
namespace mcc{
cv::Mat poly2mask(const std::vector<cv::Point2f> &poly, cv::Size size)
{

	/* ROI by creating mask for the parallelogram */
	cv::Mat mask(size, CV_8UC1);

	// Create black image with the same size as the original
	for (int i = 0; i < mask.cols; i++)
		for (int j = 0; j < mask.rows; j++)
			mask.at<uchar>(Point(i, j)) = 0;

	// Create Polygon from vertices
	vector<Point> roi_poly;
	approxPolyDP(poly, roi_poly, 1.0, true);

	// Fill polygon white
	fillConvexPoly(mask, &roi_poly[0], (int)roi_poly.size(), 1, 8, 0);

	// return the mask
	return mask;

}


float perimeter(const std::vector<cv::Point2f> &ps)
{
	float sum = 0, dx, dy;

	for (size_t i = 0;i<ps.size();i++)
	{
		int i2 = (i + 1) % (int)ps.size();

		dx = ps[i].x - ps[i2].x;
		dy = ps[i].y - ps[i2].y;

		sum += sqrt(dx*dx + dy*dy);
	}

	return sum;
}


cv::Point2f
mace_center(const std::vector<cv::Point2f> &ps)
{
	cv::Point2f center;
	int n;

	center = cv::Point2f(0);
	n = (int)ps.size();
	for (int i = 0; i < n; i++)
		center += ps[i];
	center /= n;

	return center;
}

}
}
