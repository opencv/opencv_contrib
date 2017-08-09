// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
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
#include <set>



using namespace std;
using namespace cv;
using namespace cv::saliency;
using namespace cv::ximgproc;

namespace cv
{
namespace saliency
{

BackgroundContrast::BackgroundContrast()
{

}
BackgroundContrast::~BackgroundContrast(){};

Mat BackgroundContrast::saliencyMapGenerator( const Mat img )
{
    Mat idxImg, adjcMatrix, colDistM, posDistM, bdProb, wCtr, saliency;
    superpixelSplit(img, idxImg, adjcMatrix);
    vector<unsigned> bdIds = getBndPatchIds(idxImg);
    for (unsigned i = 0; i < bdIds.size(); i++)
    {
    	for (unsigned j = 0; j < bdIds.size(); j++)
    	{
    		adjcMatrix.at<uchar>(i, j) = 1;
    	}
    }
    getColorPosDis(img, idxImg, colDistM, posDistM, adjcMatrix.size[0]);
    boundaryConnectivity(adjcMatrix, colDistM, bdProb, bdIds);
    getWeightedContrast(colDistM, posDistM, bdProb, wCtr);
    saliency = Mat(img.size[0], img.size[1], CV_64F, Scalar::all(0.0));
    for (int i = 0; i < img.size[0]; i++)
    {
    	for (int j = 0; j < img.size[1]; j++)
    	{
    		saliency.at<double>(i, j) = wCtr.at<double>(idxImg.at<unsigned>(i, j), 0);
    	}
    }
    return saliency;
}

bool BackgroundContrast::computeSaliencyImpl( InputArray image, OutputArray saliencyMap )
{
    return true;
}

void BackgroundContrast::superpixelSplit( const Mat img, Mat& idxImg, Mat& adjcMatrix)
{
    Ptr<SuperpixelSEEDS> seeds;
    seeds = createSuperpixelSEEDS( img.size().width, img.size().height, img.channels(), min(img.size().width  * img.size().height / 600, 300), 4, 2, 5, false);
    seeds->iterate( img, 4 );
    Mat mask;
    adjcMatrix = Mat::eye( seeds->getNumberOfSuperpixels(), seeds->getNumberOfSuperpixels(), CV_8U );
    seeds->getLabels(idxImg);
    seeds->getLabelContourMask(mask, true);
    for ( int i = 0; i < mask.size[0]; i++ )
    {
        for (int j = 0; j < mask.size[1]; j++ )
        {
            if (mask.at<uchar>(i, j) != 0)
            {
                if ( idxImg.at<unsigned>(i, j) != idxImg.at<unsigned>(max(i - 1, 0), j) )
                {
                    adjcMatrix.at<uchar>(idxImg.at<unsigned>(i, j), idxImg.at<unsigned>(i - 1, j)) = 2;
                    adjcMatrix.at<uchar>(idxImg.at<unsigned>(i - 1, j), idxImg.at<unsigned>(i, j)) = 2;
                }
                if ( idxImg.at<unsigned>(i, j) != idxImg.at<unsigned>(min(i + 1, mask.size[0] - 1), j) )
                {
                    adjcMatrix.at<uchar>(idxImg.at<unsigned>(i, j), idxImg.at<unsigned>(i + 1, j)) = 2;
                    adjcMatrix.at<uchar>(idxImg.at<unsigned>(i + 1, j), idxImg.at<unsigned>(i, j)) = 2;
                }
                if ( idxImg.at<unsigned>(i, j) != idxImg.at<unsigned>(i, max(j - 1, 0)) )
                {
                    adjcMatrix.at<uchar>(idxImg.at<unsigned>(i, j), idxImg.at<unsigned>(i, j - 1)) = 2;
                    adjcMatrix.at<uchar>(idxImg.at<unsigned>(i, j - 1), idxImg.at<unsigned>(i, j)) = 2;
                }
                if ( idxImg.at<unsigned>(i, j) != idxImg.at<unsigned>(i, min(j + 1, mask.size[1] - 1)) )
                {
                    adjcMatrix.at<uchar>(idxImg.at<unsigned>(i, j), idxImg.at<unsigned>(i, j + 1)) = 2;
                    adjcMatrix.at<uchar>(idxImg.at<unsigned>(i, j + 1), idxImg.at<unsigned>(i, j)) = 2;
                }
            }
        }
    }
}

vector<unsigned> BackgroundContrast::getBndPatchIds( const Mat idxImg, int thickness )
{
    set<unsigned> BndPatchIds;
    CV_Assert(idxImg.size[0] > 2 * thickness && idxImg.size[1] > 2 * thickness);
    for ( int i = 0; i < idxImg.size[0]; i++)
    {
        for (int j = 0; j < idxImg.size[1]; j++)
        {
        	if ( ((i >= 0 && i < thickness) || (i >= idxImg.size[0] - thickness && i < idxImg.size[0])) || ((j >= 0 && j < thickness) || (j >= idxImg.size[1] - thickness && j < idxImg.size[1])))
        	{
        	    BndPatchIds.insert(idxImg.at<unsigned>(i, j));
        	}
        }
    }
    vector<unsigned> res(BndPatchIds.begin(), BndPatchIds.end());
    return res;
}

void BackgroundContrast::getColorPosDis( const Mat img, const Mat idxImg, Mat& colDistM, Mat& posDistM, int nOfSP )
{
    vector<int> szOfSP = vector<int>(nOfSP, 0);
    Mat meanCol = Mat(nOfSP, img.channels(), CV_64F, Scalar::all(0.0));
    Mat meanPos = Mat(nOfSP, 2, CV_64F, Scalar::all(0.0));
    colDistM = Mat(nOfSP, nOfSP, CV_64F, Scalar::all(0.0));
    posDistM = Mat(nOfSP, nOfSP, CV_64F, Scalar::all(0.0));
    for (int i = 0; i < img.size[0]; i++ )
    {
        for (int j = 0; j < img.size[1]; j++ )
        {
            szOfSP[idxImg.at<unsigned>(i, j)] += 1;
            for (int k = 0; k < img.channels(); k++)
            {
            	meanCol.at<double>(idxImg.at<unsigned>(i, j), k) += (double)img.at<Vec3b>(i, j)[k];
            }
            meanPos.at<double>(idxImg.at<unsigned>(i, j), 0) += (double)i;
            meanPos.at<double>(idxImg.at<unsigned>(i, j), 1) += (double)j;
        }
    }
    for (int i = 0; i < nOfSP; i++)
    {
    	meanCol.row(i) /= szOfSP[i];
    	meanPos.row(i) /= szOfSP[i];
    }
    meanPos.col(0) /= img.size[0];
    meanPos.col(1) /= img.size[1];

    for ( int i = 0; i < meanCol.size[1]; i++)
    {
    	Mat temp = ( repeat(meanCol.col(i), 1, nOfSP) - repeat(meanCol.col(i).t(), nOfSP, 1) );
    	pow(temp, 2, temp);
    	colDistM += temp;
    }
    sqrt(colDistM, colDistM);

    for ( int i = 0; i < meanPos.size[1]; i++)
    {
        Mat temp = ( repeat(meanPos.col(i), 1, nOfSP) - repeat(meanPos.col(i).t(), nOfSP, 1) );
        pow(temp, 2, temp);
        posDistM += temp;
    }
    sqrt(posDistM, posDistM);
}

void BackgroundContrast::boundaryConnectivity(const Mat adjcMatrix, const Mat colDistM, Mat& bdProb, vector<unsigned> bdIds, double clipVal, double geoSigma)
{
	Mat geoDistMatrix = Mat(adjcMatrix.size[0], adjcMatrix.size[1], CV_64F, Scalar::all(0.0));
	double ma = 0;
	minMaxLoc( colDistM, NULL, &ma );
	for ( int i = 0; i < adjcMatrix.size[0]; i++ )
	{
	    for (int j = 0; j < adjcMatrix.size[1]; j++ )
	    {
	    	if ( adjcMatrix.at<uchar>(i, j) == 0 )
	    	{
	    		geoDistMatrix.at<double>(i, j) = ma * adjcMatrix.size[0]; //fake inf
	    	}
	    	else
	    	{
	    	    geoDistMatrix.at<double>(i, j) = colDistM.at<double>(i, j);
	    	}
	    }
	}
	for ( int k = 0; k < adjcMatrix.size[0]; k++ )
	{
	    for ( int i = 0; i < adjcMatrix.size[0]; i++ )
	    {
	    	for (int j = 0; j < adjcMatrix.size[1]; j++ )
	    	{
	    		geoDistMatrix.at<double>(i, j) = min(geoDistMatrix.at<double>(i, j), geoDistMatrix.at<double>(i, k) + geoDistMatrix.at<double>(k, j));
	    		geoDistMatrix.at<double>(j, i) = geoDistMatrix.at<double>(i, j);
	    	}
	    }
	}
	pow(geoDistMatrix, 2, geoDistMatrix);
	geoDistMatrix /= (-1 * 2 * geoSigma * geoSigma);
	exp(geoDistMatrix, geoDistMatrix);
	bdProb = Mat(adjcMatrix.size[0], 1, CV_64F, Scalar::all(0.0));
	for ( int i = 0; i < adjcMatrix.size[0]; i++ )
	{
	    for ( unsigned j = 0; j < bdIds.size(); j++ )
	    {
	        bdProb.at<double>(i, 0) += geoDistMatrix.at<double>(i, bdIds[j]);
	    }
	    bdProb.at<double>(i, 0) /= sqrt(sum(geoDistMatrix.row(i)).val[0]);
	}

	pow(bdProb, 2, bdProb);
	bdProb /= (-1 * 2);
	exp(bdProb, bdProb);
	bdProb = 1 - bdProb;
}

void BackgroundContrast::getWeightedContrast( const Mat colDistM, const Mat posDistM, const Mat bgProb, Mat& wCtr )
{
	wCtr = posDistM.clone();
	pow(wCtr, 2, wCtr);
	wCtr /= (-1 * 2 * 0.16);
	exp(wCtr, wCtr);
	multiply(colDistM, wCtr, wCtr);
	wCtr *= bgProb;
}


void BackgroundContrast::saliencyMapVisualize( InputArray _saliencyMap )
{
    Mat saliency = _saliencyMap.getMat().clone();
    double mi = 0, ma = 0;
    minMaxLoc( saliency, &mi, &ma );
    saliency -= mi;
    saliency /= ( ma - mi );
    imshow( "saliencyVisual", saliency );
    waitKey( 0 );
}

}
}
