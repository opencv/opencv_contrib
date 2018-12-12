/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2015-2018, OpenCV Foundation, all rights reserved.
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
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
 */

#include "opencv2/qds/quasiDenseStereo.hpp"

namespace cv {
namespace qds {

QuasiDenseStereo::QuasiDenseStereo(cv::Size monoImgSize, cv::String paramFilepath)
{
    loadParameters(paramFilepath);
    width = monoImgSize.width;
    height = monoImgSize.height;
    refMap = cv::Mat_<cv::Point2i>(monoImgSize);
    mtcMap = cv::Mat_<cv::Point2i>(monoImgSize);

    cv::Size integralSize = cv::Size(monoImgSize.width+1, monoImgSize.height+1);
    sum0 = cv::Mat_<int32_t>(integralSize);
    sum1 = cv::Mat_<int32_t>(integralSize);
    ssum0 = cv::Mat_<double>(integralSize);
    ssum1 = cv::Mat_<double>(integralSize);
    // the disparity image.
    disparity = cv::Mat_<float>(monoImgSize);
    disparityImg = cv::Mat_<uchar>(monoImgSize);
    // texture images.
    textureDescLeft = cv::Mat_<int> (monoImgSize);
    textureDescRight = cv::Mat_<int> (monoImgSize);
}

QuasiDenseStereo::~QuasiDenseStereo()
{

    rightFeatures.clear();
    leftFeatures.clear();

    refMap.release();
    mtcMap.release();

    sum0.release();
    sum1.release();
    ssum0.release();
    ssum1.release();
    // the disparity image.
    disparity.release();
    disparityImg.release();
    // texture images.
    textureDescLeft.release();
    textureDescRight.release();
}


int QuasiDenseStereo::loadParameters(cv::String filepath)
{
    cv::FileStorage fs;
    //if user specified a pathfile, try to use it.
    if (!filepath.empty())
    {
        fs.open(filepath, cv::FileStorage::READ);
    }
    // If the file opened, read the parameters.
    if (fs.isOpened())
    {
        fs["borderX"] >> Param.borderX;
        fs["borderY"] >> Param.borderY;
        fs["corrWinSizeX"] >> Param.corrWinSizeX;
        fs["corrWinSizeY"] >> Param.corrWinSizeY;
        fs["correlationThreshold"] >> Param.correlationThreshold;
        fs["textrureThreshold"] >> Param.textrureThreshold;

        fs["neighborhoodSize"] >> Param.neighborhoodSize;
        fs["disparityGradient"] >> Param.disparityGradient;

        fs["lkTemplateSize"] >> Param.lkTemplateSize;
        fs["lkPyrLvl"] >> Param.lkPyrLvl;
        fs["lkTermParam1"] >> Param.lkTermParam1;
        fs["lkTermParam2"] >> Param.lkTermParam2;

        fs["gftQualityThres"] >> Param.gftQualityThres;
        fs["gftMinSeperationDist"] >> Param.gftMinSeperationDist;
        fs["gftMaxNumFeatures"] >> Param.gftMaxNumFeatures;
        fs.release();
        return 1;
    }
    // If the filepath was incorrect or non existent, load the defaults.
    Param.borderX = BORDER_X;
    Param.borderY = BORDER_Y;
    Param.corrWinSizeX = CORR_WIN_SIZE_X;
    Param.corrWinSizeY = CORR_WIN_SIZE_Y;
    Param.correlationThreshold = CORR_THRESHOLD;
    Param.textrureThreshold = TEXTURE_THRESHOLD;

    Param.neighborhoodSize = NEIGHBORHOOD_SIZE;
    Param.disparityGradient = DISPARITY_GRADIENT;

    Param.lkTemplateSize = LK_FLOW_TEMPLAETE_SIZE;
    Param.lkPyrLvl = LK_FLOW_PYR_LVL;
    Param.lkTermParam1 = LK_FLOW_TERM_1;
    Param.lkTermParam2 = LK_FLOW_TERM_2;

    Param.gftQualityThres = GFT_QUALITY_THRESHOLD;
    Param.gftMinSeperationDist = GFT_MIN_SEPERATION_DIST;
    Param.gftMaxNumFeatures = GFT_MAX_NUM_FEATURES;
    // Return 0 if there was no filepath provides.
    // Return -1 if there was a problem opening the filepath provided.
    if(filepath.empty())
    {
        return 0;
    }
    return -1;
}

int QuasiDenseStereo::saveParameters(cv::String filepath)
{
    cv::FileStorage fs(filepath, cv::FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "borderX" << Param.borderX;
        fs << "borderY" << Param.borderY;
        fs << "corrWinSizeX" << Param.corrWinSizeX;
        fs << "corrWinSizeY" << Param.corrWinSizeY;
        fs << "correlationThreshold" << Param.correlationThreshold;
        fs << "textrureThreshold" << Param.textrureThreshold;

        fs << "neighborhoodSize" << Param.neighborhoodSize;
        fs << "disparityGradient" << Param.disparityGradient;

        fs << "lkTemplateSize" << Param.lkTemplateSize;
        fs << "lkPyrLvl" << Param.lkPyrLvl;
        fs << "lkTermParam1" << Param.lkTermParam1;
        fs << "lkTermParam2" << Param.lkTermParam2;

        fs << "gftQualityThres" << Param.gftQualityThres;
        fs << "gftMinSeperationDist" << Param.gftMinSeperationDist;
        fs << "gftMaxNumFeatures" << Param.gftMaxNumFeatures;
        fs.release();
    }
    return -1;
}

void QuasiDenseStereo::getSparseMatches(std::vector<qds::Match> &sMatches)
{
    Match tmpMatch;
    sMatches.clear();
    sMatches.resize(leftFeatures.size());
    for (uint i=0; i<leftFeatures.size(); i++)
    {
        tmpMatch.p0 = leftFeatures[i];
        tmpMatch.p1 = rightFeatures[i];
        sMatches.push_back(tmpMatch);
    }
}
void QuasiDenseStereo::getDenseMatches(std::vector<qds::Match> &dMatches)
{
    Match tmpMatch;
    dMatches.clear();
//	dMatches.resize(dMatchesLen);
    for (int row=0; row<height; row++)
    {
        for(int col=0; col<width; col++)
        {
            tmpMatch.p0 = cv::Point(col, row);
            tmpMatch.p1 = refMap.at<Point2i>(row, col);
            if (tmpMatch.p1 == NO_MATCH)
            {
                continue;
            }
            dMatches.push_back(tmpMatch);
        }
    }
}

void QuasiDenseStereo::process(const cv::Mat &imgLeft , const cv::Mat &imgRight)
{
    if (imgLeft.channels()>1)
    {
        cv::cvtColor(imgLeft, grayLeft, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imgRight, grayRight, cv::COLOR_BGR2GRAY);
    }
    else
    {
        grayLeft = imgLeft.clone();
        grayRight = imgRight.clone();
    }
    sparseMatching(grayLeft, grayRight, leftFeatures, rightFeatures);
    quasiDenseMatching(leftFeatures, rightFeatures);
}


cv::Point2f QuasiDenseStereo::getMatch(const int x, const int y)
{
    return refMap.at<cv::Point2i>(y, x);
}


void QuasiDenseStereo::sparseMatching(const cv::Mat &imgLeft ,const cv::Mat &imgRight,
                                        std::vector< cv::Point2f > &featuresLeft,
                                        std::vector< cv::Point2f > &featuresRight)
{
    std::vector< uchar > featureStatus;
    std::vector< float > error;
    featuresLeft.clear();
    featuresRight.clear();

    cv::goodFeaturesToTrack(imgLeft, featuresLeft, Param.gftMaxNumFeatures,
    Param.gftQualityThres, Param.gftMinSeperationDist);

    cv::Size templateSize(Param.lkTemplateSize,Param.lkTemplateSize);
    cv::TermCriteria termination(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,
    Param.lkTermParam1, Param.lkTermParam2);
    cv::calcOpticalFlowPyrLK(imgLeft, imgRight, featuresLeft, featuresRight,
    featureStatus, error,
    templateSize, Param.lkPyrLvl, termination);
    //discard bad features.
    for(size_t i=0; i<featuresLeft.size();)
    {
        if( featureStatus[i]==0 )
        {
            std::swap(featuresLeft[i], featuresLeft.back());
            featuresLeft.pop_back();
            std::swap(featureStatus[i], featureStatus.back());
            featureStatus.pop_back();
            std::swap(featuresRight[i], featuresRight.back());
            featuresRight.pop_back();
        }
        else
            ++i;
    }
}

float QuasiDenseStereo::iZNCC_c1(const cv::Point2i p0, const cv::Point2i p1, const int wx, const int wy)
{
    float m0=0.0 ,m1=0.0 ,s0=0.0 ,s1=0.0;
    float wa = (float)(2*wy+1)*(2*wx+1);
    float zncc=0.0;


    patchSumSum2(p0, sum0, ssum0, m0, s0, wx, wy);
    patchSumSum2(p1, sum1, ssum1, m1, s1, wx, wy);

    m0 /= wa;
    m1 /= wa;

    // standard deviations
    s0 = sqrt(s0-wa*m0*m0);
    s1 = sqrt(s1-wa*m1*m1);


    for (int col=-wy; col<=wy; col++)
    {
        for (int row=-wx; row<=wx; row++)
        {
            zncc += (float)grayLeft.at<uchar>(p0.y+row, p0.x+col) *
            (float)grayRight.at<uchar>(p1.y+row, p1.x+col);
        }
    }
    zncc = (zncc-wa*m0*m1)/(s0*s1);
    return zncc;
}

void QuasiDenseStereo::patchSumSum2(const cv::Point2i p, const cv::Mat &sum, const cv::Mat &ssum,
    float &s, float &ss, const int xWindow, const int yWindow)
{
    cv::Point2i otl(p.x-xWindow, p.y-yWindow);
    //outer top right
    cv::Point2i otr(p.x+xWindow+1, p.y-yWindow);
    //outer bottom left
    cv::Point2i obl(p.x-xWindow, p.y+yWindow+1);
    //outer bottom right
    cv::Point2i obr(p.x+xWindow+1, p.y+yWindow+1);

    // sum and squared sum for right window
    s = sum.at<int>(otl) - sum.at<int>(otr)
    -sum.at<int>(obl) + sum.at<int>(obr);

    ss = ssum.at<double>(otl) - ssum.at<double>(otr)
    -ssum.at<double>(obl) + ssum.at<double>(obr);
}



void QuasiDenseStereo::buildTextureDescriptor(cv::Mat &src,cv::Mat &descriptor)
{

    float a, b, c, d;

    //for each pixel in the input image. The boundaries are this way not to raise errors when computing a,b,c,d?

    uint8_t center, top, bottom, right, left;
    //reset descriptors

    // traverse every pixel.
    for(int row=1; row<height-1; row++)
    {
        for(int col=1; col<width-1; col++)
        {
            // the values of the current pixel.
            center = src.at<uchar>(row,col);
            top = src.at<uchar>(row-1,col);
            bottom = src.at<uchar>(row+1,col);
            left = src.at<uchar>(row,col-1);
            right = src.at<uchar>(row,col+1);

            a = (float)abs(center - top);
            b = (float)abs(center - bottom);
            c = (float)abs(center - left);
            d = (float)abs(center - right);
            //choose the biggest of them.
            int val = std::max(a, std::max(b, std::max(c, d)));
            descriptor.at<int>(row, col) = val;
        }
    }
}

bool QuasiDenseStereo::CheckBorder(Match m, int bx, int by, int w, int h)
{
    if(m.p0.x<bx || m.p0.x>w-bx || m.p0.y<by || m.p0.y>h-by ||
    m.p1.x<bx || m.p1.x>w-bx || m.p1.y<by || m.p1.y>h-by)
    {
        return false;
    }

    return true;
}

bool QuasiDenseStereo::MatchCompare(const Match a, const Match b)
{
    if(a.corr<=b.corr)return true;
    return false;
}

t_matchPriorityQueue QuasiDenseStereo::extractSparseSeeds(const std::vector< cv::Point2f > &featuesLeft,
                                                          const std::vector< cv::Point2f > &featuresRight,
                                                          cv::Mat_<cv::Point2i> &leftMap,
                                                          cv::Mat_<cv::Point2i> &rightMap)
{
    t_matchPriorityQueue seeds;
    for(uint i=0; i < featuesLeft.size(); i++)
    {
        // Calculate correlation and store match in Seeds.
        Match m;
        m.p0 = cv::Point2i(featuesLeft[i]);
        m.p1 = cv::Point2i(featuresRight[i]);
        m.corr = 0;

        // Check if too close to boundary.
        if(!CheckBorder(m,Param.borderX,Param.borderY, width, height))
        continue;

        m.corr = iZNCC_c1(m.p0, m.p1, Param.corrWinSizeX, Param.corrWinSizeY);
        // Can we add it to the list
        if( m.corr > Param.correlationThreshold )
        {
            seeds.push(m);
            leftMap.at<cv::Point2i>(m.p0.y, m.p0.x) = m.p1;
            rightMap.at<cv::Point2i>(m.p1.y, m.p1.x) = m.p0;
        }
    }
    return seeds;
}

void QuasiDenseStereo::quasiDenseMatching(const std::vector< cv::Point2f > &featuresLeft,
                                            const std::vector< cv::Point2f > &featuresRight)
{
    refMap = cv::Mat_<cv::Point2i>(cv::Size(width, height), cv::Point2i(0, 0));
    mtcMap = cv::Point2i(0, 0);

    // build texture homogeneity reference maps.
    buildTextureDescriptor(grayLeft, textureDescLeft);
    buildTextureDescriptor(grayRight, textureDescRight);

    // generate the intergal images for fast variable window correlation calculations
    cv::integral(grayLeft, sum0, ssum0);
    cv::integral(grayRight, sum1, ssum1);

    // Seed priority queue. The algorithm wants to pop the best seed available in order to densify the mess. // Evs mess ??? the sparse set maybe better ?? !!!!!!!
    t_matchPriorityQueue seeds = extractSparseSeeds(featuresLeft, featuresRight,
    refMap, mtcMap);


    // Do the propagation part
    while(!seeds.empty())
    {
        t_matchPriorityQueue Local;

        // Get the best seed at the moment
        Match m = seeds.top();
        seeds.pop();

        // Ignore the border
        if(!CheckBorder(m, Param.borderX, Param.borderY, width, height))
            continue;

        // For all neighbours of the seed in image 1
        //the neighborghoud is defined with Param.N*2 dimentrion
        for(int y=-Param.neighborhoodSize;y<=Param.neighborhoodSize;y++)
        {
            for(int x=-Param.neighborhoodSize;x<=Param.neighborhoodSize;x++)
            {
                CvPoint p0 = cvPoint(m.p0.x+x,m.p0.y+y);

                // Check if its unique in ref
                if(refMap.at<cv::Point2i>(p0.y,p0.x) != NO_MATCH)
                    continue;

                // Check the texture descriptor for a boundary
                if(textureDescLeft.at<int>(p0.y, p0.x) > Param.textrureThreshold)
                    continue;

                // For all candidate matches.
                for(int wy=-Param.disparityGradient; wy<=Param.disparityGradient; wy++)
                {
                    for(int wx=-Param.disparityGradient; wx<=Param.disparityGradient; wx++)
                    {
                        cv::Point p1 = cv::Point(m.p1.x+x+wx,m.p1.y+y+wy);

                        // Check if its unique in ref
                        if(mtcMap.at<cv::Point2i>(p1.y, p1.x) != NO_MATCH)
                            continue;

                        // Check the texture descriptor for a boundary
                        if(textureDescRight.at<int>(p1.y, p1.x) > Param.textrureThreshold)
                            continue;

                        // Calculate ZNCC and store local match.
                        float corr = iZNCC_c1(p0,p1,Param.corrWinSizeX,Param.corrWinSizeY);

                        // push back if this is valid match
                        if( corr > Param.correlationThreshold )
                        {
                            Match nm;
                            nm.p0 = p0;
                            nm.p1 = p1;
                            nm.corr = corr;
                            Local.push(nm);
                        }
                    }
                }
            }
        }

        // Get seeds from the local
        while( !Local.empty() )
        {
            Match lm = Local.top();
            Local.pop();
            // Check if its unique in both ref and dst.
            if(refMap.at<cv::Point2i>(lm.p0.y, lm.p0.x) != NO_MATCH)
                continue;
            if(mtcMap.at<cv::Point2i>(lm.p1.y, lm.p1.x) != NO_MATCH)
                continue;


            // Unique match
            refMap.at<cv::Point2i>(lm.p0.y, lm.p0.x) = lm.p1;
            mtcMap.at<cv::Point2i>(lm.p1.y, lm.p1.x) = lm.p0;
            // Add to the seed list
            seeds.push(lm);
        }
    }
}

void QuasiDenseStereo::computeDisparity(const cv::Mat_<cv::Point2i> &matchMap,
                                        cv::Mat_<float> &dispMat)
{
    for(int row=0; row< height; row++)
    {
        for(int col=0; col<width; col++)
        {
            cv::Point2d tmpPoint(col, row);

            if (matchMap.at<cv::Point2i>(tmpPoint) == NO_MATCH)
            {
                dispMat.at<float>(tmpPoint) = 200;
                continue;
            }
            //if a match is found, compute the difference in location of the match and current pixel.
            float dx = col-matchMap.at<cv::Point2i>(tmpPoint).x;
            float dy = row-matchMap.at<cv::Point2i>(tmpPoint).y;
            //calculate disparity of current pixel.
            dispMat.at<float>(tmpPoint) = sqrt(float(dx*dx+dy*dy));
        }
    }
}

cv::Mat QuasiDenseStereo::quantiseDisparity(const cv::Mat_<float> &dispMat, const int lvls)
{
    float tmpPixelVal ;
    double min, max;
//	cv::minMaxLoc(disparity, &min, &max);
    min = 0;
    max = lvls;
    for(int row=0; row<height; row++)
    {
        for(int col=0; col<width; col++)
        {
            tmpPixelVal = dispMat.at<float>(row, col);
            tmpPixelVal = 255. - 255.0*(tmpPixelVal-min)/(max-min);

            disparityImg.at<uchar>(row, col) =  (uint8_t) tmpPixelVal;
        }
    }
    return disparityImg;
}

cv::Mat QuasiDenseStereo::getDisparity(uint8_t disparityLvls)
{
    computeDisparity(refMap, disparity);
    return quantiseDisparity(disparity, disparityLvls);
}

}
}
