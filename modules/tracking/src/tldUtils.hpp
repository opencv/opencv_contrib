#ifndef OPENCV_TLD_UTILS
#define OPENCV_TLD_UTILS

namespace cv {
inline namespace tracking {
namespace impl {
namespace tld {

		//debug functions and variables
		#define ALEX_DEBUG
		#ifdef ALEX_DEBUG
		#define dfprintf(x) fprintf x
		#define dprintf(x) printf x
		#else
		#define dfprintf(x)
		#define dprintf(x)
		#endif
		#define MEASURE_TIME(a)\
			{\
			clock_t start; float milisec = 0.0; \
			start = clock(); {a} milisec = 1000.0 * (clock() - start) / CLOCKS_PER_SEC; \
			dprintf(("%-90s took %f milis\n", #a, milisec));\
			}
		#define HERE dprintf(("line %d\n", __LINE__)); fflush(stderr);
		#define START_TICK(name)\
			{ \
			clock_t start; double milisec = 0.0; start = clock();
		#define END_TICK(name) milisec = 1000.0 * (clock() - start) / CLOCKS_PER_SEC; \
			dprintf(("%s took %f milis\n", name, milisec)); \
			}
		extern Rect2d etalon;

		void myassert(const Mat& img);
		void printPatch(const Mat_<uchar>& standardPatch);
		std::string type2str(const Mat& mat);

		//aux functions and variables
		template<typename T> inline T CLIP(T x, T a, T b){ return std::min(std::max(x, a), b); }
		/** Computes overlap between the two given rectangles. Overlap is computed as ratio of rectangles' intersection to that
		* of their union.*/
		double overlap(const Rect2d& r1, const Rect2d& r2);
		/** Resamples the area surrounded by r2 in img so it matches the size of samples, where it is written.*/
		void resample(const Mat& img, const RotatedRect& r2, Mat_<uchar>& samples);
		/** Specialization of resample() for rectangles without retation for better performance and simplicity.*/
		void resample(const Mat& img, const Rect2d& r2, Mat_<uchar>& samples);
		/** Computes the variance of single given image.*/
		double variance(const Mat& img);
		void getClosestN(std::vector<Rect2d>& scanGrid, Rect2d bBox, int n, std::vector<Rect2d>& res);
		double scaleAndBlur(const Mat& originalImg, int scale, Mat& scaledImg, Mat& blurredImg, Size GaussBlurKernelSize, double scaleStep);

}}}}  // namespace

#endif
