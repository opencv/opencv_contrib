// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "precomp.hpp"
#include <vector>
#include <functional>
#include <cmath>
#include <string>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <utility>
#include <numeric>
#include <fstream>



using namespace std;
using namespace cv;
using namespace cv::saliency;

namespace cv
{
namespace saliency
{
DiscriminantSaliency::DiscriminantSaliency()
{
    imgProcessingSize = Size(127, 127); 
    hiddenSpaceDimension = 10;
    centerSize = 16;
    windowSize = 96;
    patchSize = 8;
    temporalSize = 11;
    CV_Assert( hiddenSpaceDimension <= temporalSize && temporalSize <= (unsigned)imgProcessingSize.width * imgProcessingSize.height );
}

DiscriminantSaliency::~DiscriminantSaliency(){}

bool DiscriminantSaliency::computeSaliencyImpl( InputArray image, OutputArray saliencyMap )
{
    return true;	
}

vector<Mat> DiscriminantSaliency::saliencyMapGenerator( const std::vector<Mat> img_sq)
{
	
    return img_sq;
}

double DiscriminantSaliency::KLdivDT( const Mat img_sq, DT& para_c, DT& para_w )
{
	Mat MU_c = para_c.MU.clone();//1
	Mat MU_w = para_w.MU.clone();//1
	Mat S_c = para_c.S.clone();//1
	Mat S_w = para_w.S.clone();//1
	Mat Beta = ((para_w.S.inv()) + Mat::ones(para_w.S.size(), CV_64F) / (para_w.VAR)).inv();//1
	Mat Omega = -1 * para_w.Q.inv() * para_w.A;//const
	Mat Theta = (para_w.S.inv()) + (para_w.A.t()) * (para_w.Q.inv()) * (para_w.A);//const
	Mat Vc = para_w.C.t() * para_c.C * MU_c - MU_w;//1
	
	Mat U_w = para_w.A * para_w.S;//2
	Mat H = Theta + Mat::ones(Theta.size(), CV_64F) / (para_w.VAR) - Omega.t() * Beta * Omega;//2
	Mat G = -1 * Beta * Omega;//2
	MU_c *= para_c.A;//2
	MU_w *= para_w.A;//2
	S_c = para_c.A * S_c * para_c.A.t() + para_c.Q;
	S_w = para_w.A * S_w * para_w.A.t() + para_w.Q;
	/*Beta = 
	
	
	for ( unsigned i = 1; i < img_sq.size[1]; i++)
	{
		
	}
	Mat Zc
	
	Mat Gama*/
	return 0;
}

void DiscriminantSaliency::dynamicTextureEstimator( const Mat img_sq, DT& para )
{
	
    unsigned tau = img_sq.size[1]; //row represents pixel location and column represents temporal location
    Mat me( img_sq.size[0], 1, CV_64F, Scalar::all(0.0) );
    Mat temp( img_sq.size[0], img_sq.size[1], CV_64F, Scalar::all(0.0) );
    Mat X, V, B, W, Y;
    
    for ( unsigned i = 0; i < tau; i++ )
    {
        me += img_sq.col(i) / tau; 
    }
    temp += me * Mat::ones(1, tau, CV_64F) * (-1);
    temp += img_sq;
    Y = temp.clone();
    
    SVD s = SVD(temp, SVD::MODIFY_A);
    para.C = s.u.colRange(0, hiddenSpaceDimension);
    para.C = para.C.clone();
    X = Mat::diag(s.w.rowRange(0, hiddenSpaceDimension)) * s.vt.rowRange(0, hiddenSpaceDimension);
    para.A = (X.colRange(1, tau) * (X.colRange(0, tau - 1)).inv());
    para.A = para.A.clone();
    
    V = (X.colRange(1, tau) - para.A * X.colRange(0, tau - 1));
    V = V.clone();
    SVD sv = SVD(V, SVD::MODIFY_A);
    B = sv.u * Mat::diag(sv.w) / sqrt(tau - 1);
    para.Q = (B * B.t());
    para.Q = para.Q.clone();
    
    W = (Y - para.C * X);
    W = W.clone();
    SVD sw = SVD(W, SVD::MODIFY_A);
    B = sw.u * Mat::diag(sw.w) / sqrt(tau - 1);
    para.R = (B * B.t());
    para.R = para.R.clone();
    para.VAR = (trace(para.R) / para.R.size[0])[0];
    
    para.MU = Mat( hiddenSpaceDimension, 1, CV_64F, Scalar::all(0.0) );
    for (unsigned i = 0; i < tau; i++ )
    {
    	para.MU += X.col(i) / tau;
    }
    para.S = (X - para.MU * Mat::ones(1, tau, CV_64F)) * ((X - para.MU * Mat::ones(1, tau, CV_64F)).t()) / (tau - 1);
    para.S = para.S.clone();
}

}
}