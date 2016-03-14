///////////// see LICENSE.txt in the OpenCV root directory //////////////

#ifndef __OPENCV_PYRAMID_HPP__
#define __OPENCV_PYRAMID_HPP__
#include "opencv2/core.hpp"
#include <vector>

namespace cv
{
namespace pyramid
{


    enum {
    GAUSSIAN_PYRAMID,
    BURT_ADELSON_PYRAMID,
    LAPLACIAN_LIKE_PYRAMID,
    RIESZ_PYRAMID
};

class CV_EXPORTS_W  Pyramid {
protected:
    int type; // GAUSSIAN_PYRAMID  BURT_ADELSON_PYRAMID  LAPLACIAN_LIKE_PYRAMID  (ref 1),3 RIESZ_PYRAMID
	int nbBand;
    std::vector<std::vector<Mat > > pyr;
public :
	Pyramid() {nbBand=0;type=GAUSSIAN_PYRAMID;};
	Pyramid(Mat m,int level=-1);
	Pyramid(Pyramid &p, bool zero,int idxBand=-1);
    Pyramid(Pyramid const &p);
    std::vector <std::vector<Mat> > &get(){return pyr;};

    void push_back(Mat m){ pyr.push_back(m);return; };
	size_t size() { return static_cast<int> (pyr.size()); };
	int NbBand() { if (pyr.size() == 0) return 0; return static_cast<int> (pyr[0].size()); };// A REVOIR

    std::vector<Mat> & operator [](int i) {return pyr[i];}
	Pyramid& operator=(Pyramid &x);
	Pyramid operator+=(Pyramid &a);
    Mat collapse();
	void reduce();
    ~Pyramid(){}
};

/* Not ready */
class CV_EXPORTS_W GaussianPyramid:public Pyramid {
public :
    GaussianPyramid(Mat m);
    ~GaussianPyramid(){}

};


class CV_EXPORTS_W LaplacianPyramid :public Pyramid {
    Mat lowPassFilter;
    Mat highPassFilter;

    void InitFilters();
public:
	LaplacianPyramid(Mat &,int typeLaplacian=BURT_ADELSON_PYRAMID);
    LaplacianPyramid(LaplacianPyramid &p);
    LaplacianPyramid(LaplacianPyramid &p, bool zero, int idxBand=-1);
	Mat Collapse();
    void Reduce(){return;};
    ~LaplacianPyramid(){}

};



class CV_EXPORTS_W PyramidRiesz:public Pyramid {

public :
    PyramidRiesz(LaplacianPyramid &p); // construct Riesz pyramid using laplacian pyramid
	Mat Collapse(){return Mat();};
    void Reduce(){return;};

};


}
}

#endif
