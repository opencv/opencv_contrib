#include "opencv2/mcc/charts.hpp"


namespace cv{
namespace mcc{
CChart::CChart()
	: perimetro(0)
	, area(0)
	, large_side(0)
{
}


CChart::~CChart()
{
}


void CChart::
setCorners( std::vector<cv::Point2f> p )
{
	cv::Point v1, v2;
	if ( p.empty() ) return;

	// copy
	corners = p;

	// Sort the corners in anti-clockwise order
	polyanticlockwise(corners);

	// Properties
	area = cv::contourArea(corners);
	perimetro = perimeter(corners);
	center = mace_center(corners);

	v1 = corners[2] - corners[0];
	v2 = corners[3] - corners[1];
	large_side = std::max(cv::norm(v1), cv::norm(v2));

}



//////////////////////////////////////////////////////////////////////////////////////////////

CChartDraw::
CChartDraw(CChart *pChart, cv::Mat *image )
	: m_pChart(pChart)
	, m_image(image)
{
	assert(image);
	assert(pChart);

}

void CChartDraw::
drawContour(cv::Scalar color /*= CV_RGB(0, 250, 0)*/) const
{


	//Draw lines
	float thickness = 2;
	cv::line(*m_image, (*m_pChart).corners[0], (*m_pChart).corners[1], color, thickness, LINE_AA);
	cv::line(*m_image, (*m_pChart).corners[1], (*m_pChart).corners[2], color, thickness, LINE_AA);
	cv::line(*m_image, (*m_pChart).corners[2], (*m_pChart).corners[3], color, thickness, LINE_AA);
	cv::line(*m_image, (*m_pChart).corners[3], (*m_pChart).corners[0], color, thickness, LINE_AA);

}

void CChartDraw::
drawCenter(cv::Scalar color /*= CV_RGB(0, 0, 255)*/) const
{
	int radius = 3;	int thickness = 2;
	cv::circle(*m_image, (*m_pChart).center, radius, color, thickness);
}
}
}
