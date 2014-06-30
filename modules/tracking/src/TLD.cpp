/*///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "precomp.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "time.h"
#include <algorithm>
#include <limits.h>
#include <math.h>
#include <opencv2/highgui.hpp>
#include "TLD.hpp"

#define PI 3.14159265

using namespace cv;

namespace cv {namespace tld
{

//debug functions and variables
Rect2d etalon(14.0,110.0,20.0,20.0);
void drawWithRects(const Mat& img,std::vector<Rect2d>& blackOnes,Rect2d whiteOne){
    Mat image;
    img.copyTo(image);
    if(whiteOne.width>=0){
        rectangle( image,whiteOne, 255, 1, 1 );
    }
    for(int i=0;i<(int)blackOnes.size();i++){
        rectangle( image,blackOnes[i], 0, 1, 1 );
    }
    imshow("img",image);
}
void drawWithRects(const Mat& img,std::vector<Rect2d>& blackOnes,std::vector<Rect2d>& whiteOnes){
    Mat image;
    img.copyTo(image);
    for(int i=0;i<(int)whiteOnes.size();i++){
        rectangle( image,whiteOnes[i], 255, 1, 1 );
    }
    for(int i=0;i<(int)blackOnes.size();i++){
        rectangle( image,blackOnes[i], 0, 1, 1 );
    }
    imshow("img",image);
}
void myassert(const Mat& img){
    int count=0;
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            if(img.at<uchar>(i,j)==0){
                count++;
            }
        }
    }
    dprintf(("black: %d out of %d (%f)\n",count,img.rows*img.cols,1.0*count/img.rows/img.cols));
}

void printPatch(const Mat_<uchar>& standardPatch){
    for(int i=0;i<standardPatch.rows;i++){
        for(int j=0;j<standardPatch.cols;j++){
            dprintf(("%5.2f, ",(double)standardPatch(i,j)));
        }
        dprintf(("\n"));
    }
}

std::string type2str(const Mat& mat){
  int type=mat.type();
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans =(uchar)( 1 + (type >> CV_CN_SHIFT));

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

//generic functions
double scaleAndBlur(const Mat& originalImg,int scale,Mat& scaledImg,Mat& blurredImg,Size GaussBlurKernelSize){
    double dScale=1.0;
    for(int i=0;i<scale;i++,dScale*=1.2);
    Size2d size=originalImg.size();
    size.height/=dScale;size.width/=dScale;
    resize(originalImg,scaledImg,size);
    GaussianBlur(scaledImg,blurredImg,GaussBlurKernelSize,0.0);
    return dScale;
}
void getClosestN(std::vector<Rect2d>& scanGrid,Rect2d bBox,int n,std::vector<Rect2d>& res){
    if(n>=(int)scanGrid.size()){
        res.assign(scanGrid.begin(),scanGrid.end());
        return;
    }
    std::vector<double> overlaps(n,0.0);
    res.assign(scanGrid.begin(),scanGrid.begin()+n);
    for(int i=0;i<n;i++){
        overlaps[i]=overlap(res[i],bBox);
    }
    double otmp;
    Rect2d rtmp;
    for (int i = 1; i < n; i++){
        int j = i;
        while (j > 0 && overlaps[j - 1] > overlaps[j]) {
            otmp = overlaps[j];overlaps[j] = overlaps[j - 1];overlaps[j - 1] = otmp;
            rtmp = res[j];res[j] = res[j - 1];res[j - 1] = rtmp;
            j--;
        }
    }

    double o=0.0;
    for(int i=n;i<(int)scanGrid.size();i++){
        if((o=overlap(scanGrid[i],bBox))<=overlaps[0]){
            continue;
        }
        int j=0;
        for(j=0;j<n && overlaps[j]<o;j++);
        j--;
        for(int k=0;k<j;overlaps[k]=overlaps[k+1],res[k]=res[k+1],k++);
        overlaps[j]=o;res[j]=scanGrid[i];
    }
}

double variance(const Mat& img){
    double p=0,p2=0;
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            p+=img.at<uchar>(i,j);
            p2+=img.at<uchar>(i,j)*img.at<uchar>(i,j);
        }
    }
    p/=(img.cols*img.rows);
    p2/=(img.cols*img.rows);
    return p2-p*p;
}
double variance(Mat_<unsigned int>& intImgP,Mat_<unsigned int>& intImgP2,Rect box){
    int x=(box.x),y=(box.y),width=(box.width),height=(box.height);
    CV_Assert(0<=x && (x+width)<=intImgP.cols && (x+width)<=intImgP2.cols);
    CV_Assert(0<=y && (y+height)<=intImgP.rows && (y+height)<=intImgP2.rows);
    double p=0,p2=0;
    unsigned int A,B,C,D;

    A=((y>0&&x>0)?intImgP(y-1,x-1):0);
    B=((y>0)?intImgP(y-1,x+width-1):0);
    C=((x>0)?intImgP(y+height-1,x-1):0);
    D=intImgP(y+height-1,x+width-1);
    p=(0.0+A+D-B-C)/(width*height);

    A=((y>0&&x>0)?intImgP2(y-1,x-1):0);
    B=((y>0)?intImgP2(y-1,x+width-1):0);
    C=((x>0)?intImgP2(y+height-1,x-1):0);
    D=intImgP2(y+height-1,x+width-1);
    p2=(0.0+(D-B)-(C-A))/(width*height);

    return p2-p*p;
}

double NCC(Mat_<uchar> patch1,Mat_<uchar> patch2){
    CV_Assert(patch1.rows==patch2.rows);
    CV_Assert(patch1.cols==patch2.cols);

    int N=patch1.rows*patch1.cols;
    double s1=sum(patch1)(0),s2=sum(patch2)(0);
    double n1=norm(patch1),n2=norm(patch2);
    double prod=patch1.dot(patch2);
    double sq1=sqrt(MAX(0.0,n1*n1-s1*s1/N)),sq2=sqrt(MAX(0.0,n2*n2-s2*s2/N));
    double ares=(sq2==0)?sq1/abs(sq1):(prod-s1*s2/N)/sq1/sq2;
    return ares;
}
unsigned int getMedian(const std::vector<unsigned int>& values, int size){
    if(size==-1){
        size=(int)values.size();
    }
    std::vector<int> copy(values.begin(),values.begin()+size);
    std::sort(copy.begin(),copy.end());
    if(size%2==0){
        return (copy[size/2-1]+copy[size/2])/2;
    }else{
        return copy[(size-1)/2];
    }
}

double overlap(const Rect2d& r1,const Rect2d& r2){
    double a1=r1.area(), a2=r2.area(), a0=(r1&r2).area();
    return a0/(a1+a2-a0);
}

void resample(const Mat& img,const RotatedRect& r2,Mat_<uchar>& samples){
    Mat_<float> M(2,3),R(2,2),Si(2,2),s(2,1),o(2,1);
    R(0,0)=(float)cos(r2.angle*PI/180);R(0,1)=(float)(-sin(r2.angle*PI/180));
    R(1,0)=(float)sin(r2.angle*PI/180);R(1,1)=(float)cos(r2.angle*PI/180);
    Si(0,0)=(float)(samples.cols/r2.size.width); Si(0,1)=0.0f;
    Si(1,0)=0.0f; Si(1,1)=(float)(samples.rows/r2.size.height);
    s(0,0)=(float)samples.cols; s(1,0)=(float)samples.rows;
    o(0,0)=r2.center.x;o(1,0)=r2.center.y;
    Mat_<float> A(2,2),b(2,1);
    A=Si*R;
    b=s/2.0-Si*R*o;
    A.copyTo(M.colRange(Range(0,2)));
    b.copyTo(M.colRange(Range(2,3)));
    warpAffine(img,samples,M,samples.size());
}
void resample(const Mat& img,const Rect2d& r2,Mat_<uchar>& samples){
    Mat_<float> M(2,3);
    M(0,0)=(float)(samples.cols/r2.width); M(0,1)=0.0f; M(0,2)=(float)(-r2.x*samples.cols/r2.width);
    M(1,0)=0.0f; M(1,1)=(float)(samples.rows/r2.height); M(1,2)=(float)(-r2.y*samples.rows/r2.height);
    warpAffine(img,samples,M,samples.size());

}

//other stuff
void TLDEnsembleClassifier::stepPrefSuff(uchar* arr,int len){
    int gridSize=getGridSize();
#if 0
        int step=len/(gridSize-1), pref=(len-step*(gridSize-1))/2;
        for(int i=0;i<(int)(sizeof(x1)/sizeof(x1[0]));i++){
            arr[i]=pref+arr[i]*step;
        }
#else
        int total=len-gridSize;
        int quo=total/(gridSize-1),rem=total%(gridSize-1);
        int smallStep=quo,bigStep=quo+1;
        int bigOnes=rem,smallOnes=gridSize-bigOnes-1;
        int bigOnes_front=bigOnes/2,bigOnes_back=bigOnes-bigOnes_front;
        for(int i=0;i<(int)(sizeof(x1)/sizeof(x1[0]));i++){
            if(arr[i]<bigOnes_back){
                arr[i]=(uchar)(arr[i]*bigStep+arr[i]);
                continue;
            }
            if(arr[i]<(bigOnes_front+smallOnes)){
                arr[i]=(uchar)(bigOnes_front*bigStep+(arr[i]-bigOnes_front)*smallStep+arr[i]);
                continue;
            }
            if(arr[i]<(bigOnes_front+smallOnes+bigOnes_back)){
                arr[i]=(uchar)(bigOnes_front*bigStep+smallOnes*smallStep+(arr[i]-(bigOnes_front+smallOnes))*bigStep+arr[i]);
                continue;
            }
            arr[i]=(uchar)(len-1);
        }
#endif
}
TLDEnsembleClassifier::TLDEnsembleClassifier(int ordinal,Size size){
    preinit(ordinal);
    stepPrefSuff(x1,size.width);
    stepPrefSuff(x2,size.width);
    stepPrefSuff(y1,size.height);
    stepPrefSuff(y2,size.height);
}
void TLDEnsembleClassifier::integrate(Mat_<uchar> patch,bool isPositive){
    unsigned short int position=code(patch.data,(int)patch.step[0]);
    if(isPositive){
        pos[position]++;
    }else{
        neg[position]++;
    }
}
double TLDEnsembleClassifier::posteriorProbability(const uchar* data,int rowstep)const{
    unsigned short int position=code(data,rowstep);
    double posNum=(double)pos[position], negNum=(double)neg[position];
    if(posNum==0.0 && negNum==0.0){
        return 0.0;
    }else{
        return posNum/(posNum+negNum);
    }
}
unsigned short int TLDEnsembleClassifier::code(const uchar* data,int rowstep)const{
    unsigned short int position=0;
    //char codeS[20];
    for(int i=0;i<(int)(sizeof(x1)/sizeof(x1[0]));i++){
        position=position<<1;
        if(*(data+rowstep*y1[i]+x1[i])<*(data+rowstep*y2[i]+x2[i])){
            position++;
            //codeS[i]='o';
        }else{
            //codeS[i]='x';
        }
    }
    //codeS[13]='\0';
    //dprintf(("integrate with code %s\n",codeS));
    return position;
}

}}
