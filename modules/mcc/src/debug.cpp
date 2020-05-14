#include "opencv2/mcc/debug.hpp"
#include <time.h>


namespace cv{
namespace mcc{
cv::Scalar randomcolor(RNG& rng)
{
	int icolor = (unsigned)rng;
	return Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

void imshow_250xN(const string& name_, const Mat& patch) {

	cv::Mat bigpatch; cv::Size size = patch.size();
	float asp = (float)size.height / size.width;
	float new_size = 550;
	cv::resize(patch, bigpatch, cv::Size(new_size, new_size * asp));
	imshow(name_, bigpatch);

}

double getcputime(void) { return (double)clock(); } /// (double)CLOCKS_PER_SEC
float  mili2sectime(double t) { return ((float)t) / (double)CLOCKS_PER_SEC; }
}
}
