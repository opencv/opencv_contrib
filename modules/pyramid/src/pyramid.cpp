#include "precomp.hpp"

using namespace std;
namespace cv
{
namespace pyramid
{

/* Reference
This code is patented http://www.google.com/patents/US20140072229
1 ICCP 2014 Riesz Pyramids for Fast Phase-Based Video Magnification  http://people.csail.mit.edu/nwadhwa/riesz-pyramid/
2 In "Supplemental for Riesz Pyramids for Fast Phase-Based Video Magnification" supplemental material : http://people.csail.mit.edu/nwadhwa/riesz-pyramid/RieszPyrSupplemental.zip


*/

Pyramid::Pyramid(Pyramid const &p)
{
    type=p.type;
    nbBand = p.nbBand;
	pyr.resize(p.pyr.size());
	for (int i = 0; i < static_cast<int>(p.pyr.size()); i++)
	{
	    pyr[i].resize(p.pyr[i].size());
	}

}


#define MIN_ROW_COL_PYRAMID 32
Mat Pyramid::collapse()
{
    cv::Exception e;
	e.code = -2;
	e.msg = "Not implemented for base class";
	throw e;
}
void Pyramid::reduce()
{
    cv::Exception e;
	e.code = -2;
	e.msg = "Not implemented for base class";
	throw e;
}


Pyramid& Pyramid::operator=(Pyramid &x)
{
    nbBand = x.NbBand();
    type= x.type;

    for (int i = 0; i < static_cast<int>(x.size()); i++)
    {
        pyr.push_back(x[i]);
    }
	return *this;
}

Pyramid Pyramid::operator+=(Pyramid& a)
{
	if (pyr.size() != a.size() )
	{
		cv::Exception e;
		e.code = -2;
		e.msg = "Pyramid size must be equal";
		throw e;
	}
	if (pyr[0].size() != a.get()[0].size())
	{
		cv::Exception e;
		e.code = -3;
		e.msg = "level 0 size  size must be equal";
		throw e;
	}
	Pyramid p(a, false);
	for (int i = 0; i < static_cast<int>(pyr.size()); i++)
	{
		for (int j=0; j<static_cast<int>(a.get()[i].size());j++)
			p.get()[i][j] = pyr[i][j] + a.get()[i][j];
	}
    p.type=a.type;
	return p;
}



Pyramid::Pyramid(Pyramid &p, bool zero, int idxBand)
{
    type=p.type;
    nbBand = p.NbBand();

    if (idxBand >= 0 && idxBand >= p.NbBand())
    {
		cv::Exception e;
        e.code = -1;
        e.msg = "invalid index band";
        throw e;
    }
	pyr.resize(p.size());
	for (int i = 0; i < static_cast<int>(p.size()); i++)
	{
        if (idxBand==-1)
            for (int j = 0; j < static_cast<int>(p.get()[i].size()); j++)
		    {
			    Mat m;
			    if (zero)
				    m = Mat::zeros(p.get()[i][j].size(), p.get()[i][j].type());
			    else
				    m = p.get()[i][j].clone();
			    pyr[i].push_back(m);
		    }
        else
        {
			Mat m;
			if (zero)
				m = Mat::zeros(p.get()[i][idxBand].size(), p.get()[i][idxBand].type());
			else
				m = p.get()[i][idxBand].clone();
			pyr[i].push_back(m);

        }
	}
}


GaussianPyramid::GaussianPyramid(Mat m):Pyramid()
{
    Mat x=m;
    Mat y;
    pyr.push_back(x);
    nbBand=1;
    while (x.rows >= MIN_ROW_COL_PYRAMID && x.cols > MIN_ROW_COL_PYRAMID)
    {
		vector<Mat> v;
		pyrDown(x,y);
		v.push_back(y);
        pyr.push_back(v);
        x=y;
    }

}

LaplacianPyramid::LaplacianPyramid(LaplacianPyramid &p) :Pyramid(p)
{
    type=p.type;
    nbBand = p.NbBand();
	pyr.resize(p.size());
	for (int i = 0; i < static_cast<int>(p.size()); i++)
	{
	    pyr[i].resize(p[i].size());
	}
    if (type==LAPLACIAN_LIKE_PYRAMID)
        InitFilters();
}

LaplacianPyramid::LaplacianPyramid(LaplacianPyramid &p, bool zero, int idxBand) :Pyramid(p,zero,idxBand)
{
    type=p.type;
    nbBand = p.NbBand();

    if (idxBand >= 0 && idxBand >= p.NbBand())
    {
		cv::Exception e;
        e.code = -1;
        e.msg = "invalid index band";
        throw e;
    }
	pyr.resize(p.size());
	for (int i = 0; i < static_cast<int>(p.size()); i++)
	{
        if (idxBand==-1)
            for (int j = 0; j < static_cast<int>(p.get()[i].size()); j++)
		    {
			    Mat m;
			    if (zero)
				    m = Mat::zeros(p.get()[i][j].size(), p.get()[i][j].type());
			    else
				    m = p.get()[i][j].clone();
			    pyr[i].push_back(m);
		    }
        else
        {
			Mat m;
			if (zero)
				m = Mat::zeros(p.get()[i][idxBand].size(), p.get()[i][idxBand].type());
			else
				m = p.get()[i][idxBand].clone();
			pyr[i].push_back(m);

        }
	}
}



LaplacianPyramid::LaplacianPyramid(Mat &m,int laplacianType)
{
    if (laplacianType != LAPLACIAN_LIKE_PYRAMID && laplacianType != BURT_ADELSON_PYRAMID)
    {
		cv::Exception e;
        e.code = -1;
        e.msg = "Unknown laplacian type";
        throw e;

    }
    if (laplacianType==LAPLACIAN_LIKE_PYRAMID && lowPassFilter.empty())
        InitFilters();

    type=laplacianType;
    if (laplacianType == LAPLACIAN_LIKE_PYRAMID)
    {
        Mat tmp=m;
        while (tmp.rows >= MIN_ROW_COL_PYRAMID && tmp.cols > MIN_ROW_COL_PYRAMID)
        {
	        Mat tmpLow,tmpHigh;
            filter2D(tmp,tmpHigh,CV_32F,highPassFilter);
            vector<Mat> v;
            v.push_back(tmpHigh);
            pyr.push_back(v);
            filter2D(tmp,tmpLow,CV_32F,lowPassFilter);
    //		resize(tmpLow,tmp,Size(),0.5,0.5);
		    pyrDown(tmpLow,tmp);
        }
        vector<Mat> v;
        v.push_back(tmp);
        pyr.push_back(v);
    }
    if (laplacianType == BURT_ADELSON_PYRAMID)
    {
        Mat tmp=m;
        while (tmp.rows >= MIN_ROW_COL_PYRAMID && tmp.cols > MIN_ROW_COL_PYRAMID)
        {
            Mat tmp1,tmp2;
            pyrDown(tmp,tmp1);
            pyrUp(tmp1,tmp2);
            subtract(tmp,tmp2,tmp1,noArray(),CV_32F);
            vector<Mat> v;
            v.push_back(tmp1);
            pyr.push_back(v);
            tmp=tmp1;
        }
        vector<Mat> v;
        v.push_back(tmp);
        pyr.push_back(v);
    }
}


Mat LaplacianPyramid::Collapse()
{
	Mat x, y;

	y = pyr[pyr.size() - 1][0];
    switch (type){
    case  BURT_ADELSON_PYRAMID :
	    for (int i = static_cast<int>(pyr.size()) - 2; i >= 0; i--)
	    {
		    pyrUp(y, x);
		    add(x, pyr[i][0], y);
	    }
        break;
    case  LAPLACIAN_LIKE_PYRAMID :
        for (int i = static_cast<int>(pyr.size())-2; i>=0;i--)
        {
            Mat tmp1,tmp2;
            resize(y,x,pyr[i][0].size(),-1,-1,0);
            filter2D(x,tmp1,CV_32F,lowPassFilter);
            filter2D(pyr[i][0],tmp2,CV_32F,highPassFilter);
            add(tmp1,tmp2,y);
        }
        break;
    }
	return y;

}


void LaplacianPyramid::InitFilters()
{
    // Ref 2
    lowPassFilter = (Mat_<float>(9,9)<< -0.0001,   -0.0007,  -0.0023,  -0.0046,  -0.0057,  -0.0046,  -0.0023,  -0.0007,  -0.0001,
	                                    -0.0007,   -0.0030,  -0.0047,  -0.0025,  -0.0003,  -0.0025,  -0.0047,  -0.0030,  -0.0007,
	                                    -0.0023,   -0.0047,   0.0054,   0.0272,   0.0387,   0.0272,   0.0054,  -0.0047,  -0.0023,
	                                    -0.0046,   -0.0025,   0.0272,   0.0706,   0.0910,   0.0706,   0.0272,  -0.0025,  -0.0046,
	                                    -0.0057,   -0.0003,   0.0387,   0.0910,   0.1138,   0.0910,   0.0387,  -0.0003,  -0.0057,
	                                    -0.0046,   -0.0025,   0.0272,   0.0706,   0.0910,   0.0706,   0.0272,  -0.0025,  -0.0046,
	                                    -0.0023,   -0.0047,   0.0054,   0.0272,   0.0387,   0.0272,   0.0054,  -0.0047,  -0.0023,
	                                    -0.0007,   -0.0030,  -0.0047,  -0.0025,  -0.0003,  -0.0025,  -0.0047,  -0.0030,  -0.0007,
	                                    -0.0001,   -0.0007,  -0.0023,  -0.0046,  -0.0057,  -0.0046,  -0.0023,  -0.0007,  -0.0001);
    highPassFilter=(Mat_<float>(9,9)<<   0.0000,   0.0003,   0.0011,   0.0022,   0.0027,   0.0022,   0.0011,   0.0003,   0.0000,
	                                     0.0003,   0.0020,   0.0059,   0.0103,   0.0123,   0.0103,   0.0059,   0.0020,   0.0003,
	                                     0.0011,   0.0059,   0.0151,   0.0249,   0.0292,   0.0249,   0.0151,   0.0059,   0.0011,
	                                     0.0022,   0.0103,   0.0249,   0.0402,   0.0469,   0.0402,   0.0249,   0.0103,   0.0022,
	                                     0.0027,   0.0123,   0.0292,   0.0469,  -0.9455,   0.0469,   0.0292,   0.0123,   0.0027,
	                                     0.0022,   0.0103,   0.0249,   0.0402,   0.0469,   0.0402,   0.0249,   0.0103,   0.0022,
	                                     0.0011,   0.0059,   0.0151,   0.0249,   0.0292,   0.0249,   0.0151,   0.0059,   0.0011,
	                                     0.0003,   0.0020,   0.0059,   0.0103,   0.0123,   0.0103,   0.0059,   0.0020,   0.0003,
	                                     0.0000,   0.0003,   0.0011,   0.0022,   0.0027,   0.0022,   0.0011,   0.0003,   0.0000);
}







PyramidRiesz::PyramidRiesz(LaplacianPyramid &p)
{
    type=RIESZ_PYRAMID;
	    Mat xKernel=(Mat_<float>(3,3) << 0, 0, 0, 0.5, 0, -0.5, 0, 0, 0);
	    Mat yKernel=(Mat_<float>(3,3) << 0, .5, 0, 0, 0, 0, 0, -0.5, 0);
	//    Mat yKernel=(Mat_<float>(3,3) << -0.12, -0.34, -0.12, 0,0,0, 0.12, 0.34, 0.12);
	//    Mat xKernel=yKernel.t();

	for (int i = 0; i<static_cast<int>(p.size())-1;i++)
    {
		vector<Mat> v;
        Mat tmp1,tmp2;
        filter2D(p[i][0],tmp1,CV_32F,xKernel);
        v.push_back(tmp1);
		filter2D(p[i][0],tmp2,CV_32F,yKernel);
        v.push_back(tmp2);
		pyr.push_back(v);
    }

}
}
}
