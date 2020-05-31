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
cvWiener2(const Mat& srcMat, Mat& dstMat, int szWindowX, int szWindowY)
{

	szWindowX = cv::max(szWindowX, 1);
	szWindowY = cv::max(szWindowY, 1);

	int nRows;
	int nCols;
	Scalar v = 0;
	Mat p_kernel ;
	Mat srcStub ;
	Mat p_tmpMat1, p_tmpMat2, p_tmpMat3, p_tmpMat4;
	double noise_power;



	nRows = szWindowY;
	nCols = szWindowX;


	p_kernel = Mat(nRows, nCols, CV_32F);
	p_kernel = Scalar(1.0 / (double)(nRows * nCols));

	//Now create a temporary holding matrix
	p_tmpMat1 = Mat(srcMat.size(), (srcMat.type()));
	p_tmpMat2 = Mat(srcMat.size(), (srcMat.type()));
	p_tmpMat3 = Mat(srcMat.size(), (srcMat.type()));
	p_tmpMat4 = Mat(srcMat.size(), (srcMat.type()));

	//Local mean of input
	filter2D(srcMat, p_tmpMat1, -1 , p_kernel, Point(nCols / 2, nRows / 2)); //localMean

    //Local variance of input
    p_tmpMat2 = srcMat.mul(  srcMat);
	filter2D(p_tmpMat2, p_tmpMat3, -1 , p_kernel, Point(nCols / 2, nRows / 2));

	//Subtract off local_mean^2 from local variance
	p_tmpMat4 = p_tmpMat1.mul( p_tmpMat1);
	// Mul(p_tmpMat1, p_tmpMat1, p_tmpMat4); //localMean^2
	p_tmpMat3 = p_tmpMat3.mul(p_tmpMat4);
	// Sub(p_tmpMat3, p_tmpMat4, p_tmpMat3); //filter(in^2) - localMean^2 ==> localVariance

											//Estimate noise power
	v = mean(p_tmpMat3);
	noise_power = v.val[0];
	// result = local_mean  + ( max(0, localVar - noise) ./ max(localVar, noise)) .* (in - local_mean)

    dstMat = srcMat - p_tmpMat1;//in - local_mean

	p_tmpMat2 = max(p_tmpMat3, noise_power); //max(localVar, noise)

    add(p_tmpMat3, Scalar(-noise_power), p_tmpMat3); //localVar - noise
    p_tmpMat3 = max(p_tmpMat3, 0);// max(0, localVar - noise)

	p_tmpMat3 = (p_tmpMat3/ p_tmpMat2); //max(0, localVar-noise) / max(localVar, noise)

	dstMat = p_tmpMat3.mul(dstMat);
	dstMat = dstMat + p_tmpMat1;


}

void CWienerFilter::
wiener2(cv::Mat & src, cv::Mat & dest, int szWindowX, int szWindowY)
{
	if (szWindowY < 0)szWindowY = szWindowX;
	if (dest.empty())dest.create(src.size(), src.type());

	cvWiener2( src, dest, szWindowX, szWindowY);
}

}
}
