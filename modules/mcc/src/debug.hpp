#ifndef _MCC_DEBUG_HPP
#define _MCC_DEBUG_HPP

#include "precomp.hpp"

namespace cv{
namespace mcc{

cv::Scalar randomcolor(RNG& rng);

void imshow_250xN(const std::string& name_, const cv::Mat& patch);

double getcputime(void);

float mili2sectime(double t);

inline void showAndSave(std::string name, const cv::Mat& m, std::string path = ".")
{
	imshow_250xN(name, m);
	cv::imwrite(path + "/" + name + ".png", m);
	if (waitKey(0) == 'q')return;
}

}
}


#endif //_MCC_DEBUG_HPP
