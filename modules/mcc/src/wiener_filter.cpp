#include "wiener_filter.hpp"

namespace cv{
namespace mcc{
CWienerFilter::CWienerFilter()
{
}

CWienerFilter::~CWienerFilter()
{
}


void CWienerFilter::
wiener2(const Mat& src, Mat& dst, int szWindowX, int szWindowY)
{

	CV_Assert( szWindowX > 0 && szWindowY > 0);
	int nRows;
	int nCols;
	Scalar v = 0;
	Mat p_kernel ;
	Mat srcStub ;
	//Now create a temporary holding matrix
	Mat p_tmpMat1, p_tmpMat2, p_tmpMat3, p_tmpMat4;
	double noise_power;

	nRows = szWindowY;
	nCols = szWindowX;


	p_kernel = Mat(nRows, nCols, CV_32F);
	p_kernel = Scalar(1.0 / (double)(nRows * nCols));


	//Local mean of input
	filter2D(src, p_tmpMat1, -1 , p_kernel, Point(nCols / 2, nRows / 2)); //localMean

    //Local variance of input
    p_tmpMat2 = src.mul(  src);
	filter2D(p_tmpMat2, p_tmpMat3, -1 , p_kernel, Point(nCols / 2, nRows / 2));

	//Subtract off local_mean^2 from local variance
	p_tmpMat4 = p_tmpMat1.mul( p_tmpMat1);//localMean^2
	p_tmpMat3 = p_tmpMat3 - p_tmpMat4;
	// Sub(p_tmpMat3, p_tmpMat4, p_tmpMat3); //filter(in^2) - localMean^2 ==> localVariance

											//Estimate noise power
	v = mean(p_tmpMat3);
	noise_power = v.val[0];
	// result = local_mean  + ( max(0, localVar - noise) ./ max(localVar, noise)) .* (in - local_mean)

    dst = src - p_tmpMat1;//in - local_mean

	p_tmpMat2 = max(p_tmpMat3, noise_power); //max(localVar, noise)

    add(p_tmpMat3, Scalar(-noise_power), p_tmpMat3); //localVar - noise
    p_tmpMat3 = max(p_tmpMat3, 0);// max(0, localVar - noise)

	p_tmpMat3 = (p_tmpMat3/ p_tmpMat2); //max(0, localVar-noise) / max(localVar, noise)

	dst = p_tmpMat3.mul(dst);
	dst = dst + p_tmpMat1;
}

} // namespace mcc
} // namespace cv
