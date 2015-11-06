/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015, OpenCV Foundation, all rights reserved.
Copyright (C) 2015, Itseez Inc., all rights reserved.
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

Algorithmic details of this algorithm can be found at:
 * O. Green, Y. Birk, "A Computationally Efficient Algorithm for the 2D Covariance Method", ACM/IEEE International Conference on High Performance Computing, Networking, Storage and Analysis, Denver, Colorado, 2013
A previous and less efficient version of the algorithm can be found:
 * O. Green, L. David, A. Galperin, Y. Birk, "Efficient parallel computation of the estimated covariance matrix", arXiv, 2013


*/

#include "precomp.hpp"

using namespace cv;
using namespace std;


namespace cv{
namespace ximgproc{

class EstimateCovariance{

public:
    EstimateCovariance(int pr_, int pc_);
    ~EstimateCovariance();

    void computeEstimateCovariance(Mat inputData,Mat outputData);
    int combinationCount();

private:
    typedef struct {
        int mult1r;
        int mult1c;
        int mult2r;
        int mult2c;
        int type2; // 0 - for the first P*P. 1 - for the next (P-1)(P-1)
        int id;
    } Combination;

    void initInternalDataStructures();
    void buildCombinationsTable();

    void iterateCombinations(Mat inputData,Mat outputData);
    void computeOneCombination(int comb_id, Mat inputData , Mat outputData,
        Mat outputVector,std::vector<int> finalMatPosR, std::vector<int> finalMatPosC);

    inline void complexSubtract(std::complex<float>& src, std::complex<float>& dst){dst-=src;}
    inline void complexAdd(std::complex<float>& src, std::complex<float>& dst){dst+=src;}
    inline void complexConjMulAndAdd(std::complex<float>& a, std::complex<float>& b,
            std::complex<float>& dst){dst += a*b;}
    inline void complexConjMul(std::complex<float>& a, std::complex<float>& b,
            std::complex<float>& dst){dst = a*b;}

private:
    int nr;
    int nc;
    int pr;
    int pc;

    std::vector<Combination> combinationsTable;
};



EstimateCovariance::EstimateCovariance(int pr_, int pc_){
    pr=pr_; pc=pc_;
}

EstimateCovariance::~EstimateCovariance(){

}

void EstimateCovariance::initInternalDataStructures(){
    int combCount = combinationCount();
    combinationsTable.resize(combCount);
    buildCombinationsTable();
}

int EstimateCovariance::combinationCount(){
    return  (pr*pc+(pr-1)*(pc-1));
}

void EstimateCovariance::buildCombinationsTable()
{
    int idx_row,idx_col;
    int comb_idx = 0;
    Combination comb;

    // The first element of the product is [0,0] and the second is to down and to the right of it.
    for (idx_col=0; idx_col<pc; ++idx_col)  {
        for (idx_row=0; idx_row<pr; ++idx_row)      {
            comb.mult1r=0;
            comb.mult1c=0;
            comb.mult2r=idx_row;
            comb.mult2c=idx_col;
            comb.type2 = 0;
            comb.id = comb_idx;
            memcpy(&combinationsTable[comb_idx++], &comb, sizeof(Combination));
        }
    }

    // The first element is on the top right, and the second element is to the left and down of it.
    for (idx_row=1; idx_row<pr; ++idx_row)  {
        for (idx_col=1; idx_col<pc; ++idx_col)      {
            comb.mult1r=idx_row;
            comb.mult1c=0;
            comb.mult2r=0;
            comb.mult2c=idx_col;

            comb.type2 = 1;
            comb.id = comb_idx;
            memcpy(&combinationsTable[comb_idx++], &comb, sizeof(Combination));
        }
    }
}

void EstimateCovariance::computeEstimateCovariance(Mat inputData,Mat outputData){
    initInternalDataStructures();
    nr=inputData.rows;
    nc=inputData.cols;

    iterateCombinations(inputData,outputData);
}


void EstimateCovariance::iterateCombinations(Mat inputData,Mat outputData)
{
    Mat outputVector(pr*pc,1,  DataType<std::complex<float> >::type);

    std::vector<int> finalMatPosR(pr*pc,0);
    std::vector<int> finalMatPosC(pr*pc,0);
    int combs=combinationCount();
    for (int idx=0; idx<combs; idx++){
        outputVector.setTo(Scalar(0,0));
        for (unsigned x=0; x<finalMatPosR.size(); x++)
            finalMatPosR[x]=0;
        for (unsigned x=0; x<finalMatPosC.size(); x++)
            finalMatPosC[x]=0;
        computeOneCombination(idx++, inputData, outputData,
                outputVector,finalMatPosR, finalMatPosC);
    }
}

void EstimateCovariance::computeOneCombination(int comb_id,Mat inputData, Mat outputData,
            Mat outputVector,std::vector<int> finalMatPosR, std::vector<int> finalMatPosC)
{
    Combination* comb = &combinationsTable[comb_id];
    int type2 = comb->type2;
    int deltaR = (int)abs((int)(comb->mult1r-comb->mult2r));
    int deltaC = (int)abs((int)(comb->mult1c-comb->mult2c));
    int numElementsInBlock = pr-abs(deltaR);
    int numBlocks = pc-deltaC;
    const int DR= nr-pr;
    const int DC= nc-pc;

    int elementC=0;
    std::complex<float> temp_res = std::complex<float>(0,0);
    int i,j,r,c;

    if (!type2) {
        // Computing the first index of the combination.
        // This index is made up
        for(i=0; i<=( DR); i++) {
            int iPdr=i+deltaR;
            for(j=0; j<= (DC); j++) {
                int jPdc = j+deltaC;
                complexConjMulAndAdd(inputData.at<std::complex<float> >(i,j), inputData.at<std::complex<float> >(iPdr,jPdc), temp_res);
            }
        }
    }else{
        // Computing the first index of the combination.
        for(i=0; i<=( DR); i++) {
            int iPdr=i+deltaR;
            for(j=0; j<= (DC); j++) {
                int jPdc = j+deltaC;
                complexConjMulAndAdd(inputData.at<std::complex<float> >(iPdr,j), inputData.at<std::complex<float> >(i,jPdc), temp_res);
            }
        }
    }
    outputVector.at<std::complex<float> >(0,0) = temp_res;

    // Checking if the first element belongs to the first set of combinatons.
    // The combination that the first element is above the second.
    if (!type2) {
        finalMatPosR[0]=0;
        finalMatPosC[0]=pr*deltaC+deltaR;
        elementC++;
    }else{
        finalMatPosR[0]=deltaR;
        finalMatPosC[0]=pr*deltaC;
        elementC++;
    }

    for(r=1;r<(pr-deltaR);r++){
        std::complex<float> newRowSum = std::complex<float>(0,0),oldRowSum = std::complex<float>(0,0);
        std::complex<float> addRows  = std::complex<float>(0,0);

        int rM1 = r-1;
        int k = DR+1 + rM1;
        int kPdr = k+deltaR;
        int rM1Pdr = rM1 + deltaR;
        int cPdc;

        if (!type2) {
            for(c=0;c<=(DC); c++){
                cPdc = c+deltaC;
                complexConjMulAndAdd(inputData.at<std::complex<float> >(k,c),inputData.at<std::complex<float> >(kPdr,cPdc),newRowSum);
                complexConjMulAndAdd(inputData.at<std::complex<float> >(rM1,c),inputData.at<std::complex<float> >(rM1Pdr,cPdc),oldRowSum);
            }
        }else{
            for(c=0;c<=(DC); c++){
                cPdc = c+deltaC;
                complexConjMulAndAdd(inputData.at<std::complex<float> >(kPdr,c),inputData.at<std::complex<float> >(k,cPdc),newRowSum);
                complexConjMulAndAdd(inputData.at<std::complex<float> >(rM1Pdr,c),inputData.at<std::complex<float> >(rM1,cPdc),oldRowSum);
            }
        }
        complexAdd(newRowSum,addRows);
        complexSubtract(oldRowSum,addRows);
        complexAdd(outputVector.at<std::complex<float> >(rM1,0),outputVector.at<std::complex<float> >(r,0));
        complexAdd(addRows,outputVector.at<std::complex<float> >(r,0));

        finalMatPosR[elementC]=finalMatPosR[elementC-1]+1;;
        finalMatPosC[elementC]=finalMatPosC[elementC-1]+1;
        elementC++;
    }

    for(c=1; c<numBlocks; c++)  {
        std::complex<float> newColSum = std::complex<float>(0,0),oldColSum = std::complex<float>(0,0);
        std::complex<float> addCols  = std::complex<float>(0,0);

        // Index arithmetic
        int cM1 = c-1;
        int dcPc = DC+c;
        int q = DC+1 + cM1;
        int cM1PdeltaC = cM1 + deltaC;
        int dcPcPdeltaC = dcPc+deltaC;

        if (!type2) {
            for(r=0;r<=(DR); r++){
                int rPdr = r+deltaR;
                complexConjMulAndAdd(inputData.at<std::complex<float> >(r,q),inputData.at<std::complex<float> >(rPdr,q+deltaC),newColSum);
                complexConjMulAndAdd(inputData.at<std::complex<float> >(r,cM1),inputData.at<std::complex<float> >(rPdr,cM1+deltaC),oldColSum);
            }
        }else{
            for(r=0;r<=(DR); r++){
                int rPdr = r+deltaR;
                complexConjMulAndAdd(inputData.at<std::complex<float> >(rPdr,q),inputData.at<std::complex<float> >(r,q+deltaC),newColSum);
                complexConjMulAndAdd(inputData.at<std::complex<float> >(rPdr,cM1),inputData.at<std::complex<float> >(r,cM1+deltaC),oldColSum);
            }
        }
        complexAdd(newColSum,addCols);
        complexSubtract(oldColSum,addCols);
        complexAdd(outputVector.at<std::complex<float> >((c-1)*numElementsInBlock,0),outputVector.at<std::complex<float> >(c*numElementsInBlock,0));
        complexAdd(addCols,outputVector.at<std::complex<float> >(c*numElementsInBlock,0));

        finalMatPosR[elementC]=finalMatPosR[elementC-numElementsInBlock]+pr;
        finalMatPosC[elementC]=finalMatPosC[elementC-numElementsInBlock]+pr;
        elementC++;

        for(r=1; r<numElementsInBlock; r++) {
            std::complex<float> w = std::complex<float>(0,0),x = std::complex<float>(0,0),
                y = std::complex<float>(0,0),z = std::complex<float>(0,0),deltaRowSum = std::complex<float>(0,0);
            std::complex<float> tempRes = std::complex<float>(0,0);
            // Index arithmetic
            int rM1 = r-1;
            int drPr = DR+r;
            int rM1PdeltaR = rM1 + deltaR;
            int drPrPdeltaR = drPr+deltaR;

            if (!type2) {
                complexConjMul(inputData.at<std::complex<float> >(rM1,cM1),inputData.at<std::complex<float> >(rM1PdeltaR,cM1PdeltaC),w);
                complexConjMul(inputData.at<std::complex<float> >(rM1,dcPc),inputData.at<std::complex<float> >(rM1PdeltaR,dcPcPdeltaC),x);
                complexConjMul(inputData.at<std::complex<float> >(drPr,cM1),inputData.at<std::complex<float> >(drPrPdeltaR,cM1PdeltaC),y);
                complexConjMul(inputData.at<std::complex<float> >(drPr,dcPc),inputData.at<std::complex<float> >(drPrPdeltaR,dcPcPdeltaC),z);
            }else{
                complexConjMul(inputData.at<std::complex<float> >(rM1PdeltaR,cM1),inputData.at<std::complex<float> >(rM1,cM1PdeltaC),w);
                complexConjMul(inputData.at<std::complex<float> >(rM1PdeltaR,dcPc),inputData.at<std::complex<float> >(rM1,dcPcPdeltaC),x);
                complexConjMul(inputData.at<std::complex<float> >(drPrPdeltaR,cM1),inputData.at<std::complex<float> >(drPr,cM1PdeltaC),y);
                complexConjMul(inputData.at<std::complex<float> >(drPrPdeltaR,dcPc),inputData.at<std::complex<float> >(drPr,dcPcPdeltaC),z);
            }
            complexAdd(w,tempRes);
            complexSubtract(x,tempRes);
            complexSubtract(y,tempRes);
            complexAdd(z,tempRes);

            complexAdd(outputVector.at<std::complex<float> >((c-1)*numElementsInBlock + r,0),deltaRowSum);
            complexSubtract(outputVector.at<std::complex<float> >((c-1)*numElementsInBlock+ rM1,0),deltaRowSum);

            complexAdd(deltaRowSum,tempRes);
            complexAdd(outputVector.at<std::complex<float> >(c*numElementsInBlock+rM1,0),tempRes);
            complexAdd(tempRes,outputVector.at<std::complex<float> >(c*numElementsInBlock+r,0));

            finalMatPosR[elementC]=finalMatPosR[elementC-1]+1;
            finalMatPosC[elementC]=finalMatPosC[elementC-1]+1;
            elementC++;
        }
    }

    for(i=0; i<numElementsInBlock*numBlocks; i++){
        outputData.at<std::complex<float> >(finalMatPosR[i],finalMatPosC[i])=outputVector.at<std::complex<float> >(i,0);
    }
}



void covarianceEstimation(InputArray input_, OutputArray output_,int windowRows, int windowCols){

    CV_Assert( input_.channels() <= 2);   // Does not take color images.

    Mat input;

    Mat temp=input_.getMat();
    if(temp.channels() == 1){
        temp.convertTo(temp,CV_32FC2);
        Mat zmat = Mat::zeros(temp.size(), CV_32F);
        Mat twoChannelsbefore[] = {temp,zmat};
        cv::merge(twoChannelsbefore,2,input);
    }else{
        temp.convertTo(input, CV_32FC2);

    }

    EstimateCovariance estCov(windowRows,windowCols);

    //input_.getMat().convertTo(input, CV_32FC2);

    output_.create(windowRows*windowCols,windowRows*windowCols,  DataType<std::complex<float> >::type);

    Mat output = output_.getMat();

    estCov.computeEstimateCovariance(input,output);
}

} // namespace ximgproc
} // namespace cv
