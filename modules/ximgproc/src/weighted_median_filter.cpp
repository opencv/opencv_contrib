/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

#include "precomp.hpp"
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

namespace{

using namespace cv::ximgproc;

/***************************************************************/
/* Function: from32FTo32S
 * Description: adaptive quantization for changing a floating-point 1D image to integer image.
 *                The adaptive quantization strategy is based on binary search, which searches an
 *                upper bound of quantization error.
 *                The function also return a mapping between quantized value (32F) and quantized index (32S).
 *                The mapping is used to convert integer image back to floating-point image after filtering.
 ***************************************************************/
void from32FTo32S(Mat &img, Mat &outImg, int nI, float *mapping)
{
    int rows = img.rows, cols = img.cols;
    int alls = rows * cols;

    float *imgPtr = img.ptr<float>();
    typedef pair<float,int> pairFI;
    pairFI *data = (pairFI *)malloc(alls*sizeof(pairFI));

    // Sort all pixels of the image by ascending order of pixel value
    for(int i=0;i<alls;i++){
        data[i].second = i;
        data[i].first = imgPtr[i];
    }
    sort(data,data+alls);

    // Find lower bound and upper bound of the pixel values
    double maxVal,minVal;
    minMaxLoc(img,&minVal,&maxVal);
    float maxRange = (float)(maxVal - minVal);
    float th = 1e-5f;

    float l = 0, r = maxRange*2.0f/nI;
    // Perform binary search on error bound
    while(r-l > th)
    {
        float m = (r+l)*0.5f;
        bool suc = true;
        float base = (float)minVal;
        int cnt=0;
        for(int i=0;i<alls;i++)
        {
            if(data[i].first>base+m)
            {
                cnt++;
                base = data[i].first;
                if(cnt==nI)
                {
                    suc = false;
                    break;
                }
            }
        }
        if(suc)r=m;
        else l=m;
    }

    Mat retImg(img.size(),CV_32SC1);
    int *retImgPtr = retImg.ptr<int>();

    // In the sorted list, divide pixel values into clusters according to the minimum error bound
    // Quantize each value to the median of its cluster
    // Also record the mapping of quantized value and quantized index.
    float base = (float)minVal;
    int baseI = 0;
    int cnt = 0;
    for(int i=0;i<=alls;i++)
    {
        if(i==alls || data[i].first>base+r)
        {
            mapping[cnt] = data[(baseI+i-1)>>1].first; //median
            if(i==alls)break;
            cnt++;
            base = data[i].first;
            baseI = i;
        }
        retImgPtr[data[i].second] = cnt;
    }

    free(data);

    //end of the function
    outImg = retImg;
}

/***************************************************************/
/* Function: from32STo32F
 * Description: convert the quantization index image back to the floating-point image accroding to the mapping
***************************************************************/
void from32STo32F(Mat &img, Mat &outImg, float *mapping)
{
    Mat retImg(img.size(),CV_32F);
    int rows = img.rows, cols = img.cols, alls = rows*cols;
    float *retImgPtr = retImg.ptr<float>();
    int *imgPtr = img.ptr<int>();

    // convert 32S index to 32F real value
    for(int i=0;i<alls;i++)
    {
        retImgPtr[i] = mapping[imgPtr[i]];
    }

    // end of the function
    outImg = retImg;
}


/***************************************************************
 * Function: float2D
 * Description: allocate a 2D float array with dimension "dim1 x dim2"
 ***************************************************************/
float** float2D(int dim1, int dim2)
{
    float **ret = new float*[dim1];
    ret[0] = new float[dim1*dim2];
    for(int i=1;i<dim1;i++)ret[i] = ret[i-1]+dim2;

    return ret;
}

/***************************************************************
 * Function: float2D_release
 * Description: deallocate the 2D array created by float2D()
 ***************************************************************/
void float2D_release(float **p)
{
    delete []p[0];
    delete []p;
}

/***************************************************************
 * Function: int2D
 * Description: allocate a 2D integer array with dimension "dim1 x dim2"
 ***************************************************************/
int** int2D(int dim1, int dim2)
{
    int **ret = new int*[dim1];
    ret[0] = new int[dim1*dim2];
    for(int i=1;i<dim1;i++)ret[i] = ret[i-1]+dim2;

    return ret;
}

/***************************************************************
 * Function: int2D_release
 * Description: deallocate the 2D array created by int2D()
 ***************************************************************/
void int2D_release(int **p)
{
    delete []p[0];
    delete []p;
}

/***************************************************************
 * Function: updateBCB
 * Description: maintain the necklace table of BCB
 ***************************************************************/
inline void updateBCB(int &num,int *f,int *b,int i,int v)
{
    static int p1,p2;

    if(i)
    {
        if(!num)
        { // cell is becoming non-empty
            p2=f[0];
            f[0]=i;
            f[i]=p2;
            b[p2]=i;
            b[i]=0;
        }
        else if(!(num+v))
        {// cell is becoming empty
            p1=b[i],p2=f[i];
            f[p1]=p2;
            b[p2]=p1;
        }
    }

    // update the cell count
    num += v;
}

/***************************************************************
 * Function: featureIndexing
 * Description: convert uchar feature image "F" to CV_32SC1 type.
 *                If F is 3-channel, perform k-means clustering
 *                If F is 1-channel, only perform type-casting
 ***************************************************************/
void featureIndexing(Mat &F, float **&wMap, int &nF, float sigmaI, WMFWeightType weightType){
    // Configuration and Declaration
    Mat FNew;
    int cols = F.cols, rows = F.rows;
    int alls = cols * rows;
    int KmeansAttempts=1;

    /* For 1 channel feature image (uchar)*/
    if(F.channels() == 1)
    {
        nF = 256;

        // Type-casting
        F.convertTo(FNew, CV_32S);

        // Compute weight map (weight between each pair of feature index)
        wMap = float2D(nF,nF);
        float nSigmaI = sigmaI;
        float divider = (1.0f/(2*nSigmaI*nSigmaI));

        for(int i=0;i<nF;i++)
        {
            for(int j=i;j<nF;j++)
            {
                float diff = fabs((float)(i-j));
                float val;

                switch(weightType)
                {
                    case WMF_EXP: val = exp(-(diff*diff)*divider); break;
                    case WMF_IV1: val = 1.0f/(diff+nSigmaI); break;
                    case WMF_IV2: val = 1.0f / (diff*diff+nSigmaI*nSigmaI); break;
                    case WMF_COS: val = 1.0f; break;
                    case WMF_JAC: val = (float)(min(i,j)*1.0/max(i,j)); break;
                    case WMF_OFF: val = 1.0f; break;
                    default: val = exp(-(diff*diff)*divider);
                }

                wMap[i][j] = wMap[j][i] = val;
            }
        }
    }

    /* For 3 channel feature image (uchar)*/
    else if(F.channels() == 3)
    {
        const int shift = 2; // 256(8-bit)->64(6-bit)
        const int LOW_NUM = 256>>shift;
        static int hash[LOW_NUM][LOW_NUM][LOW_NUM]={{{0}}};

        memset(hash,0,sizeof(hash));

        // throw pixels into a 2D histogram
        int candCnt = 0;
        {
            int lowR,lowG,lowB;
            uchar *FPtr = F.ptr<uchar>();
            for(int i=0,i3=0;i<alls;i++,i3+=3)
            {
                lowB = FPtr[i3]>>shift;
                lowG = FPtr[i3+1]>>shift;
                lowR = FPtr[i3+2]>>shift;

                if(hash[lowB][lowG][lowR]==0)
                {
                    candCnt++;
                    hash[lowB][lowG][lowR]=1;
                }
            }
        }

        nF = min(nF, candCnt);
        Mat samples(candCnt,3,CV_32F);

        //prepare for K-means
        int top=0;
        for(int i=0;i<LOW_NUM;i++)for(int j=0;j<LOW_NUM;j++)for(int k=0;k<LOW_NUM;k++){
            if(hash[i][j][k]){
                samples.ptr<float>(top)[0] = (float)i;
                samples.ptr<float>(top)[1] = (float)j;
                samples.ptr<float>(top)[2] = (float)k;
                top++;
            }
        }

        //do K-means
        Mat labels;
        Mat centers;
        kmeans(samples, nF, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 0, 10000), KmeansAttempts, KMEANS_PP_CENTERS, centers );

        //make connection (i,j,k) <-> index
        top = 0;
        for(int i=0;i<LOW_NUM;i++)for(int j=0;j<LOW_NUM;j++)for(int k=0;k<LOW_NUM;k++)
        {
            if(hash[i][j][k])
            {
                hash[i][j][k] = labels.ptr<int>(top)[0];
                top++;
            }
        }

        // generate index map
        FNew = Mat(F.size(),CV_32SC1);

        int lowR,lowG,lowB;
        uchar *FPtr = F.ptr<uchar>();
        for(int i=0,i3=0;i<alls;i++,i3+=3)
        {
            lowB = FPtr[i3]>>shift;
            lowG = FPtr[i3+1]>>shift;
            lowR = FPtr[i3+2]>>shift;

            FNew.ptr<int>()[i] = hash[lowB][lowG][lowR];
        }

        // Compute weight map (weight between each pair of feature index)
        wMap = float2D(nF,nF);
        float nSigmaI = sigmaI/256.0f*LOW_NUM;
        float divider = (1.0f/(2*nSigmaI*nSigmaI));

        float *length = new float[nF];
        for(int i=0;i<nF;i++)
        {
            float a0 = centers.ptr<float>(i)[0];
            float a1 = centers.ptr<float>(i)[1];
            float a2 = centers.ptr<float>(i)[2];
            length[i] = sqrt(a0*a0+a1*a1+a2*a2);
        }

        for(int i=0;i<nF;i++)
        {
            for(int j=i;j<nF;j++)
            {
                float a0 = centers.ptr<float>(i)[0], b0 = centers.ptr<float>(j)[0];
                float a1 = centers.ptr<float>(i)[1], b1 = centers.ptr<float>(j)[1];
                float a2 = centers.ptr<float>(i)[2], b2 = centers.ptr<float>(j)[2];
                float diff0 = a0-b0;
                float diff1 = a1-b1;
                float diff2 = a2-b2;

                float val;

                switch(weightType)
                {
                    case WMF_EXP: val = exp(-(diff0*diff0+diff1*diff1+diff2*diff2)*divider); break;
                    case WMF_IV1: val = 1.0f/(fabs(diff0)+fabs(diff1)+fabs(diff2)+nSigmaI); break;
                    case WMF_IV2: val = 1.0f / (diff0*diff0+diff1*diff1+diff2*diff2+nSigmaI*nSigmaI); break;
                    case WMF_COS: val = (a0*b0+a1*b1+a2*b2)/(length[i]*length[j]); break;
                    case WMF_JAC: val = (min(a0,b0)+min(a1,b1)+min(a2,b2))/(max(a0,b0)+max(a1,b1)+max(a2,b2)); break;
                    case WMF_OFF: val = 1.0f; break;
                    default: val = exp(-(diff0*diff0+diff1*diff1+diff2*diff2)*divider);
                }

                wMap[i][j] = wMap[j][i] = val;
            }
        }

        delete []length;
    }
    //end of the function
    F = FNew;
}

Mat filterCore(Mat &I, Mat &F, float **wMap, int r=20, int nF=256, int nI=256, Mat mask=Mat())
{
    // Check validation
    assert(I.depth() == CV_32S && I.channels()==1);//input image: 32SC1
    assert(F.depth() == CV_32S && F.channels()==1);//feature image: 32SC1

    // Configuration and declaration
    int rows = I.rows, cols = I.cols;
    Mat outImg = I.clone();

    // Handle Mask
    if(mask.empty())
    {
        mask = Mat(I.size(),CV_8U);
        mask = Scalar(1);
    }

    // Allocate memory for joint-histogram and BCB
    int **H = int2D(nI,nF);
    int *BCB = new int[nF];

    // Allocate links for necklace table
    int **Hf = int2D(nI,nF);//forward link
    int **Hb = int2D(nI,nF);//backward link
    int *BCBf = new int[nF];//forward link
    int *BCBb = new int[nF];//backward link

    // Column Scanning
    for(int x=0;x<cols;x++)
    {
        // Reset histogram and BCB for each column
        memset(BCB, 0, sizeof(int)*nF);
        memset(H[0], 0, sizeof(int)*nF*nI);
        for(int i=0;i<nI;i++)Hf[i][0]=Hb[i][0]=0;
        BCBf[0]=BCBb[0]=0;

        // Reset cut-point
        int medianVal = -1;

        // Precompute "x" range and checks boundary
        int downX = max(0,x-r);
        int upX = min(cols-1,x+r);

        // Initialize joint-histogram and BCB for the first window
        int upY = min(rows-1,r);
        for(int i=0;i<=upY;i++)
        {
            int *IPtr = I.ptr<int>(i);
            int *FPtr = F.ptr<int>(i);
            uchar *maskPtr = mask.ptr<uchar>(i);

            for(int j=downX;j<=upX;j++)
            {
                if(!maskPtr[j])continue;

                int fval = IPtr[j];
                int *curHist = H[fval];
                int gval = FPtr[j];

                // Maintain necklace table of joint-histogram
                if(!curHist[gval] && gval)
                {
                    int *curHf = Hf[fval];
                    int *curHb = Hb[fval];

                    int p1=0,p2=curHf[0];
                    curHf[p1]=gval;
                    curHf[gval]=p2;
                    curHb[p2]=gval;
                    curHb[gval]=p1;
                }

                curHist[gval]++;
                // Maintain necklace table of BCB
                updateBCB(BCB[gval],BCBf,BCBb,gval,-1);
            }
        }

        for(int y=0;y<rows;y++)
        {
            // Find weighted median with help of BCB and joint-histogram
            float balanceWeight = 0;
            int curIndex = F.ptr<int>(y,x)[0];
            float *fPtr = wMap[curIndex];
            int &curMedianVal = medianVal;

            // Compute current balance
            {
                int i=0;
                do
                {
                    balanceWeight += BCB[i]*fPtr[i];
                    i=BCBf[i];
                }while(i);
            }

            // Move cut-point to the left
            if(balanceWeight >= 0)
            {
                for(;balanceWeight >= 0 && curMedianVal; curMedianVal--)
                {
                    float curWeight = 0;
                    int *nextHist = H[curMedianVal];
                    int *nextHf = Hf[curMedianVal];

                    // Compute weight change by shift cut-point
                    int i=0;
                    do
                    {
                        curWeight += (nextHist[i]<<1)*fPtr[i];

                        // Update BCB and maintain the necklace table of BCB
                        updateBCB(BCB[i],BCBf,BCBb,i,-(nextHist[i]<<1));

                        i=nextHf[i];
                    }while(i);

                    balanceWeight -= curWeight;
                }
            }
            // Move cut-point to the right
            else if(balanceWeight < 0)
            {
                for(;balanceWeight < 0 && curMedianVal != nI-1; curMedianVal++)
                {
                    float curWeight = 0;
                    int *nextHist = H[curMedianVal+1];
                    int *nextHf = Hf[curMedianVal+1];

                    // Compute weight change by shift cut-point
                    int i=0;
                    do
                    {
                        curWeight += (nextHist[i]<<1)*fPtr[i];

                        // Update BCB and maintain the necklace table of BCB
                        updateBCB(BCB[i],BCBf,BCBb,i,nextHist[i]<<1);

                        i=nextHf[i];
                    }while(i);
                    balanceWeight += curWeight;
                }
            }

            // Weighted median is found and written to the output image
            if(balanceWeight<0)outImg.ptr<int>(y,x)[0] = curMedianVal+1;
            else outImg.ptr<int>(y,x)[0] = curMedianVal;

            // Update joint-histogram and BCB when local window is shifted.
            int fval,gval,*curHist;

            // Add entering pixels into joint-histogram and BCB
            int rownum = y + r + 1;
            if(rownum < rows)
            {
                    int *inputImgPtr = I.ptr<int>(rownum);
                    int *guideImgPtr = F.ptr<int>(rownum);
                    uchar *maskPtr = mask.ptr<uchar>(rownum);

                    for(int j=downX;j<=upX;j++)
                    {
                        if(!maskPtr[j])continue;

                        fval = inputImgPtr[j];
                        curHist = H[fval];
                        gval = guideImgPtr[j];

                        // Maintain necklace table of joint-histogram
                        if(!curHist[gval] && gval)
                        {
                            int *curHf = Hf[fval];
                            int *curHb = Hb[fval];

                            int p1=0,p2=curHf[0];
                            curHf[gval]=p2;
                            curHb[gval]=p1;
                            curHf[p1]=curHb[p2]=gval;
                        }

                        curHist[gval]++;

                        // Maintain necklace table of BCB
                        updateBCB(BCB[gval],BCBf,BCBb,gval,((fval <= medianVal)<<1)-1);
                    }
                }


                // Delete leaving pixels into joint-histogram and BCB
                rownum = y - r;
                if(rownum >= 0)
                {
                    int *inputImgPtr = I.ptr<int>(rownum);
                    int *guideImgPtr = F.ptr<int>(rownum);
                    uchar *maskPtr = mask.ptr<uchar>(rownum);

                    for(int j=downX;j<=upX;j++)
                    {
                        if(!maskPtr[j])continue;

                        fval = inputImgPtr[j];
                        curHist = H[fval];
                        gval = guideImgPtr[j];

                        curHist[gval]--;

                        // Maintain necklace table of joint-histogram
                        if(!curHist[gval] && gval)
                        {
                            int *curHf = Hf[fval];
                            int *curHb = Hb[fval];

                            int p1=curHb[gval],p2=curHf[gval];
                            curHf[p1]=p2;
                            curHb[p2]=p1;
                        }

                        // Maintain necklace table of BCB
                        updateBCB(BCB[gval],BCBf,BCBb,gval,-((fval <= medianVal)<<1)+1);
                    }
                }
        }
    }

    // Deallocate the memory
    {
        delete []BCB;
        delete []BCBf;
        delete []BCBb;
        int2D_release(H);
        int2D_release(Hf);
        int2D_release(Hb);
    }

    // end of the function
    return outImg;
}
}

namespace cv
{
namespace ximgproc
{
void weightedMedianFilter(InputArray joint, InputArray src, OutputArray dst, int r, double sigma, WMFWeightType weightType, Mat mask)
{
    CV_Assert(!src.empty());
    CV_Assert(r > 0 && sigma > 0);

    int nI = 256;
    int nF = 256;

    Mat I = src.getMat();
    Mat F = joint.getMat();
    if(joint.empty())
    {
        medianBlur(src, dst, r);
        return;
    }

    CV_Assert(I.depth() == CV_32F || I.depth() == CV_8U);
    CV_Assert(F.depth() == CV_8U && (F.channels() == 1 || F.channels() == 3));

    dst.create(src.size(), src.type());
    Mat D = dst.getMat();

    if(D.data == F.data)
        F = F.clone();
    if(D.data == I.data)
        I = I.clone();

    //Preprocess I
    //OUTPUT OF THIS STEP: Is, iMap
    //If I is floating point image, "adaptive quantization" is done in from32FTo32S.
    //The mapping of floating value to integer value is stored in iMap (for each channel).
    //"Is" stores each channel of "I". The channels are converted to CV_32S type after this step.
    vector<float *> iMap(I.channels());
    vector<Mat> Is;
    split(I,Is);
    for(int i=0;i<(int)Is.size();i++)
    {
        if(I.depth() == CV_32F)
        {
            iMap[i] = new float[nI];
            from32FTo32S(Is[i],Is[i],nI,iMap[i]);
        }
        else if(I.depth() == CV_8U)
        {
            Is[i].convertTo(Is[i],CV_32S);
        }
    }

    //Preprocess F
    //OUTPUT OF THIS STEP: F(new), wMap
    //If "F" is 3-channel image, "clustering feature image" is done in featureIndexing.
    //If "F" is 1-channel image, featureIndexing only does a type-casting on "F".
    //The output "F" is CV_32S type, containing indexes of feature values.
    //"wMap" is a 2D array that defines the distance between each pair of feature indexes.
    // wMap[i][j] is the weight between feature index "i" and "j".
    float **wMap = NULL;
    featureIndexing(F, wMap, nF, float(sigma), weightType);

    //Filtering - Joint-Histogram Framework
    for(int i=0; i<(int)Is.size(); i++)
    {
        Is[i] = filterCore(Is[i], F, wMap, r, nF,nI,mask);
    }
    float2D_release(wMap);

    //Postprocess F
    //Convert input image back to the original type.
    for(int i = 0; i < (int)Is.size(); i++)
    {
        if(I.depth()==CV_32F)
        {
            from32STo32F(Is[i],Is[i],iMap[i]);
            delete []iMap[i];
        }
        else if(I.depth()==CV_8U)
        {
            Is[i].convertTo(Is[i],CV_8U);
        }
    }

    //merge the channels
    merge(Is, D);
}
}
}
