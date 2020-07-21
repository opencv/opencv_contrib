#include "common.hpp"

namespace cv{
namespace mcc{
Rect poly2mask(const std::vector<cv::Point2f> &poly, cv::Size size, cv::Mat& mask)
{
    // Create black image with the same size as the original
    //mask.create(size, CV_8UC1);
    //mask.setTo(Scalar::all(0));

	// Create Polygon from vertices
	vector<Point> roi_poly;
	approxPolyDP(poly, roi_poly, 1.0, true);

    Rect roi = boundingRect(roi_poly);

	// Fill polygon white
	fillConvexPoly(mask, &roi_poly[0], (int)roi_poly.size(), 1, 8, 0);

    roi &= Rect(0, 0, size.width, size.height);
    if(roi.empty())
        roi = Rect(0, 0, 1, 1);
    return roi;
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
