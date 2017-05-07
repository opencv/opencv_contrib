/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#ifndef __OPENCV_PYRAMID_HPP__
#define __OPENCV_PYRAMID_HPP__
#include "opencv2/core.hpp"
#include <vector>

/**
@defgroup pyramid hierarchical pyramid
@ref pyramid
*/

namespace cv
{
namespace pyramid
{
//! @addtogroup pyramid
//! @{

/** @brief virtual class pyramid Pyramid  .
 */


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

/** @brief  class GaussianPyramid TO DO .
 */
class CV_EXPORTS_W GaussianPyramid:public Pyramid {
public :
    GaussianPyramid(Mat m);
    ~GaussianPyramid(){}

};

/** @brief  class LaplacianPyramid  BURT ADELSON pyramid and Laplacian like pyramid.
 */

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


/** @brief  class PyramidRiesz  .
 */

class CV_EXPORTS_W PyramidRiesz:public Pyramid {

public :
    PyramidRiesz(LaplacianPyramid &p); // construct Riesz pyramid using laplacian pyramid
	Mat Collapse(){return Mat();};
    void Reduce(){return;};

};


}
}

#endif
