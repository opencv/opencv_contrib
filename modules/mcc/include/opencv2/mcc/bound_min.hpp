#ifndef _MCC_BOUND_MIN_H
#define _MCC_BOUND_MIN_H

#include "core.hpp"
#include "charts.hpp"

#include <opencv2/core.hpp>
#include <vector>

namespace cv{

namespace mcc{

class CBoundMin
{

public:

	CBoundMin();
	~CBoundMin();


	void setCharts(const std::vector<CChart> &chartIn) { chart = chartIn;  }
	void getCorners(std::vector<cv::Point2f> &cornersOut) { cornersOut = corners; }
	void calculate();


private:

	std::vector<CChart> chart;
	std::vector<cv::Point2f> corners;


private:




	bool validateLine(const std::vector<cv::Point3f> &Lc, cv::Point3f ln, int k, int &j)
	{

		float theta;
		cv::Point2d v0, v1;

		for (j = 0; j < k; j++)
		{
			v0.x = Lc[j].x; v0.y = Lc[j].y;
			v1.x = ln.x; v1.y = ln.y;
			theta = v0.dot(v1) / (norm(v0)*norm(v1));
			theta = acos(theta);

			if (theta < 0.5)  return true;

		}

		return false;

	}




};


}

}
#endif //_MCC_BOUND_MIN_H
