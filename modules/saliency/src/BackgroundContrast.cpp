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
    bgWei = 5;
    limitOfSP = 600;
    nOfLevel = 4;
    usePrior = 2;
    histBin = 5;
}
BackgroundContrast::BackgroundContrast( double _bgWei, int _limitOfSP, int _nOfLevel, int _usePrior, int _histBin ): limitOfSP(_limitOfSP), nOfLevel(_nOfLevel), usePrior(_usePrior), histBin(_histBin), bgWei(_bgWei) {}
BackgroundContrast::~BackgroundContrast(){}

Mat BackgroundContrast::saliencyMapGenerator( const Mat img, const Mat fgImg, int option )
{
    CV_Assert( !(img.empty()) );
    CV_Assert( !(option == 1 && fgImg.empty()) );
    Mat idxImg, adjcMatrix, colDistM, posDistM, bdProb, wCtr, saliency;
    superpixelSplit(img, idxImg, adjcMatrix);
    vector<unsigned> bdIds = getBndPatchIds(idxImg);
    for (unsigned i = 0; i < bdIds.size(); i++)
    {
        for (unsigned j = 0; j < bdIds.size(); j++)
        {
            adjcMatrix.at<uchar>(bdIds[i], bdIds[j]) = 1;
            adjcMatrix.at<uchar>(bdIds[j], bdIds[i]) = 1;
        }
    }
    getColorPosDis(img, idxImg, colDistM, posDistM, adjcMatrix.size[0]);
    boundaryConnectivity(adjcMatrix, colDistM, bdProb, bdIds);

    if ( option == 0 )
    {
        getWeightedContrast(colDistM, posDistM, bdProb, wCtr);

    }
    else
    {
        Mat temp = fgImg.clone();
        if (temp.channels() == 3) cvtColor(temp, temp, COLOR_BGR2GRAY);
        wCtr = Mat(adjcMatrix.size[0], 1, CV_64F, Scalar::all(0.0));
        resize(temp, temp, img.size());
        vector<int> szOfSP = vector<int>(adjcMatrix.size[0], 0);
        for ( int i = 0; i < img.size[0]; i++ )
        {
            for ( int j = 0; j < img.size[1]; j++ )
            {
                szOfSP[idxImg.at<unsigned>(i, j)] += 1;
                wCtr.at<double>(idxImg.at<unsigned>(i, j), 0) += temp.at<double>(i, j);
            }
        }
        for ( unsigned i = 0; i < szOfSP.size(); i++ )
        {
            wCtr.at<double>(i, 0) /= szOfSP[i];
        }
    }
    saliencyOptimize(adjcMatrix, colDistM, bdProb, wCtr, wCtr, bgWei);
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

void BackgroundContrast::saliencyOptimize( const Mat adjcMatrix, const Mat colDistM, const Mat bgWeight, const Mat fgWeight, Mat& saliencyOptimized, double bgLambda, double neiSigma )
{

    Mat smoothWeight = colDistM.clone();
    Mat smoothDeri = Mat(smoothWeight.size[0], smoothWeight.size[1], CV_64F, Scalar::all(0.0));
    Mat bgWeightDig = Mat(smoothWeight.size[0], smoothWeight.size[1], CV_64F, Scalar::all(0.0));
    Mat fgWeightDig = Mat(smoothWeight.size[0], smoothWeight.size[1], CV_64F, Scalar::all(0.0));
    Mat temp;

    double mi = 0, ma = 0;
    minMaxLoc( fgWeight, &mi, &ma );
    Mat fg = fgWeight.clone();
    fg -= mi;
    fg /= ( ma - mi + 0.000001 );
    fg *= 255;
    fg.convertTo(fg, CV_8U);
    threshold(fg, fg, 0, 255, THRESH_TOZERO | THRESH_OTSU);
    fg.convertTo(fg, CV_64F);
    fg /= 255; // clean fore ground cue

    minMaxLoc( smoothWeight, NULL, &ma );
    for ( int i = 0; i < smoothWeight.size[0]; i++ )
    {
        for ( int j = 0; j < smoothWeight.size[1]; j++ )
        {
            if ( adjcMatrix.at<uchar>(i, j) == 0 )
            {
                smoothWeight.at<double>(i, j) = ma * adjcMatrix.size[0];
            }
        }
    }

    dist2WeightMatrix(smoothWeight, smoothWeight, neiSigma);
    adjcMatrix.convertTo(temp, CV_64F);
    smoothWeight += temp * 0.1;//add small coefficients for regularization term
    reduce(smoothWeight, temp, 0, REDUCE_SUM);
    for ( int i = 0; i < smoothDeri.size[0]; i++ )
    {
        smoothDeri.at<double>(i, i) = temp.at<double>(0, i);
    }
    for ( int i = 0; i < bgWeightDig.size[0]; i++ )
    {
        bgWeightDig.at<double>(i, i) = bgWeight.at<double>(i, 0) * bgLambda;
    }
    for ( int i = 0; i < fgWeightDig.size[0]; i++ )
    {
        fgWeightDig.at<double>(i, i) = fg.at<double>(i, 0);
    }
    //temp = (smoothDeri - smoothWeight + bgWeightDig + fgWeightDig);
    //saliencyOptimized = temp.inv() * fgWeight;
    solve((smoothDeri - smoothWeight + bgWeightDig + fgWeightDig), fg, saliencyOptimized, DECOMP_NORMAL);
}

bool BackgroundContrast::computeSaliencyImpl( InputArray image, OutputArray saliencyMap )
{
    CV_Assert( !(image.getMat().empty()) );
    saliencyMap.assign(saliencyMapGenerator(image.getMat()));
    return true;
}

void BackgroundContrast::superpixelSplit( const Mat img, Mat& idxImg, Mat& adjcMatrix)
{
    Ptr<SuperpixelSEEDS> seeds;
    seeds = createSuperpixelSEEDS( img.size().width, img.size().height, img.channels(), max(min(img.size().width  * img.size().height / 600, limitOfSP), 10), nOfLevel, usePrior, histBin, false);
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
    rgb2lab(meanCol, meanCol);
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
                geoDistMatrix.at<double>(i, j) = max(0.0, colDistM.at<double>(i, j) - clipVal);
            }
        }
    }
    for ( int k = 0; k < adjcMatrix.size[0]; k++ ) // floyd algorithm, you can replace it with johnson algorithm but it's too long
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
    dist2WeightMatrix(geoDistMatrix, geoDistMatrix, geoSigma);
    bdProb = Mat(adjcMatrix.size[0], 1, CV_64F, Scalar::all(0.0));
    for ( int i = 0; i < adjcMatrix.size[0]; i++ )
    {
        for ( unsigned j = 0; j < bdIds.size(); j++ )
        {
            bdProb.at<double>(i, 0) += geoDistMatrix.at<double>(i, bdIds[j]);
        }
        bdProb.at<double>(i, 0) /= sqrt(sum(geoDistMatrix.row(i)).val[0]);
    }
    dist2WeightMatrix(bdProb, bdProb, 1);
    bdProb = 1 - bdProb;
}

void BackgroundContrast::getWeightedContrast( const Mat colDistM, const Mat posDistM, const Mat bgProb, Mat& wCtr )
{
    wCtr = posDistM.clone();
    dist2WeightMatrix(wCtr,wCtr, 0.4);
    multiply(colDistM, wCtr, wCtr);
    wCtr *= bgProb;
}

void BackgroundContrast::dist2WeightMatrix( Mat& input, Mat& output, double sigma )
{
    Mat temp = input.clone();
    output = input.clone();
    for ( int i = 0; i < output.size[0]; i++ )
    {
        for ( int j = 0; j < output.size[1]; j++ )
        {
            //if (temp.at<double>(i, j) > 3 * sigma) output.at<double>(i, j) = 0;
            //else
            //{
                output.at<double>(i, j) = exp(-1 * temp.at<double>(i, j) * temp.at<double>(i, j) / 2 / sigma / sigma);
            //}
        }
    }
}

void BackgroundContrast::rgb2lab( Mat& inputMeanCol, Mat& outputMeanCol )
{
    Mat temp = Mat(inputMeanCol.size[0], 1, CV_32FC3, Scalar::all(0));
    for ( int i = 0; i < inputMeanCol.size[0]; i++ )
    {
        temp.at<Vec3f>(i, 0)[0] = (float)inputMeanCol.at<double>(i, 0) / 255;
        temp.at<Vec3f>(i, 0)[1] = (float)inputMeanCol.at<double>(i, 1) / 255;
        temp.at<Vec3f>(i, 0)[2] = (float)inputMeanCol.at<double>(i, 2) / 255;
    }
    cvtColor(temp, temp, COLOR_BGR2Lab);
    outputMeanCol = inputMeanCol.clone();
    for ( int i = 0; i < outputMeanCol.size[0]; i++ )
    {
        outputMeanCol.at<double>(i, 0) = temp.at<Vec3f>(i, 0)[0];
        outputMeanCol.at<double>(i, 1) = temp.at<Vec3f>(i, 0)[1];
        outputMeanCol.at<double>(i, 2) = temp.at<Vec3f>(i, 0)[2];
    }
}

Mat BackgroundContrast::saliencyMapVisualize( InputArray _saliencyMap, int option )
{
    Mat saliency = _saliencyMap.getMat().clone();

    double mi = 0, ma = 0;
    minMaxLoc( saliency, &mi, &ma );
    saliency -= mi;
    saliency /= ( ma - mi + 0.000001 );

    if (option != 0 )
    {
        saliency *= 255;
        saliency.convertTo(saliency, CV_8U);
        if (option == 1) threshold(saliency, saliency, 0, 255, THRESH_BINARY | THRESH_OTSU);
        else if (option == 2) threshold(saliency, saliency, 0, 255, THRESH_TOZERO | THRESH_OTSU);
        //threshold(saliency, saliency, 0, 255, THRESH_TOZERO | THRESH_OTSU);
    }
    imshow( "saliencyVisual", saliency );
    waitKey( 0 );
    return saliency;
}

}
}
