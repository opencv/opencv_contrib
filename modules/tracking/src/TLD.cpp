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
#include <opencv2/highgui.hpp>
#include "TLD.hpp"

using namespace cv;

namespace cv
{

//debug functions and variables
Rect2d etalon(14.0,110.0,20.0,20.0);
void drawWithRects(const Mat& img,std::vector<Rect2d>& blackOnes,Rect2d whiteOne){
    Mat image;
    img.copyTo(image);
    if(whiteOne.width>=0){
        rectangle( image,whiteOne, 255, 1, 1 );
    }
    for(int i=0;i<blackOnes.size();i++){
        rectangle( image,blackOnes[i], 0, 1, 1 );
    }
    imshow("img",image);
}
void drawWithRects(const Mat& img,std::vector<Rect2d>& blackOnes,std::vector<Rect2d>& whiteOnes){
    Mat image;
    img.copyTo(image);
    for(int i=0;i<whiteOnes.size();i++){
        rectangle( image,whiteOnes[i], 255, 1, 1 );
    }
    for(int i=0;i<blackOnes.size();i++){
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
    printf("black: %d out of %d (%f)\n",count,img.rows*img.cols,1.0*count/img.rows/img.cols);
}

void printPatch(const Mat_<uchar>& standardPatch){
    for(int i=0;i<standardPatch.rows;i++){
        for(int j=0;j<standardPatch.cols;j++){
            printf("%5.2f, ",standardPatch(i,j));
        }
        printf("\n");
    }
}

std::string type2str(const Mat& mat){
  int type=mat.type();
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

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
    if(n>=scanGrid.size()){
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
    for(int i=n;i<scanGrid.size();i++){
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
    double p=0,p2=0;
    unsigned int A,B,C,D;

    A=((y>0&&x>0)?intImgP(y-1,x-1):0.0);
    B=((y>0)?intImgP(y-1,x+width-1):0.0);
    C=((x>0)?intImgP(y+height-1,x-1):0.0);
    D=intImgP(y+height-1,x+width-1);
    p=(0.0+A+D-B-C)/(width*height);

    A=((y>0&&x>0)?intImgP2(y-1,x-1):0.0);
    B=((y>0)?intImgP2(y-1,x+width-1):0.0);
    C=((x>0)?intImgP2(y+height-1,x-1):0.0);
    D=intImgP2(y+height-1,x+width-1);
    p2=(0.0+(D-B)-(C-A))/(width*height);

    return p2-p*p;
}

double NCC(Mat_<uchar> patch1,Mat_<uchar> patch2){
    CV_Assert(patch1.rows=patch2.rows);
    CV_Assert(patch1.cols=patch2.cols);

    int N=patch1.rows*patch1.cols;
    double s1=sum(patch1)(0),s2=sum(patch2)(0);
    double n1=norm(patch1),n2=norm(patch2);
    double prod=patch1.dot(patch2);
    double sq1=sqrt(n1*n1-s1*s1/N),sq2=sqrt(n2*n2-s2*s2/N);
    double ares=(sq2==0)?sq1/abs(sq1):(prod-s1*s2/N)/sq1/sq2;
    return ares;

    /*Mat_<uchar> p1(80,80),p2(80,80);
    printf("NCC\n");
    resample(patch1,Rect2d(Point2d(0,0),patch1.size()),p1);
    resample(patch2,Rect2d(Point2d(0,0),patch2.size()),p2);
    imshow("patch1",p1);
    imshow("patch2",p2);
    printf("NCC=%f\n",ncc);
    waitKey();*/
}
unsigned int getMedian(const std::vector<unsigned int>& values, int size){
    if(size==-1){
        size=values.size();
    }
    std::vector<int> copy(values.begin(),values.begin()+size);
    std::sort(copy.begin(),copy.end());
    if(size%2==0){
        return (copy[size/2-1]+copy[size/2])/2.0;
    }else{
        return copy[(size-1)/2];
    }
}

inline double overlap(const Rect2d& r1,const Rect2d& r2){
    double a1=r1.area(), a2=r2.area(), a0=(r1&r2).area();
    return a0/(a1+a2-a0);
}

void resample(const Mat& img,const RotatedRect& r2,Mat_<uchar>& samples){
    Point2f vertices[4];
    r2.points(vertices);

    int ref=0;
    float minx=vertices[0].x,miny=vertices[0].y;
    for(int i=1;i<4;i++){
        if(vertices[i].x<minx || (vertices[i].x==minx && vertices[i].y<miny)){
            minx=vertices[i].x;
            miny=vertices[i].y;
            ref=i;
        }
    }

    float dx1=vertices[(ref+1)%4].x-vertices[ref].x,
          dy1=vertices[(ref+1)%4].y-vertices[ref].y,
          dx2=vertices[(ref+3)%4].x-vertices[ref].x,
          dy2=vertices[(ref+3)%4].y-vertices[ref].y;
    for(int i=0;i<samples.rows;i++){
        for(int j=0;j<samples.cols;j++){
            float x=vertices[ref].x+dx1*j/samples.cols+dx2*i/samples.rows,
                  y=vertices[ref].y+dy1*j/samples.cols+dy2*i/samples.rows;
            int ix=cvFloor(x),iy=cvFloor(y);
            float tx=x-ix,ty=y-iy;
            float a=img.at<uchar>(CLIP(iy,0,img.cols-1),CLIP(ix,0,img.rows-1))*(1.0-tx)+
                img.at<uchar>(CLIP(iy,0,img.cols-1),CLIP(ix+1,0,img.rows-1))* tx;
            float b=img.at<uchar>(CLIP(iy+1,0,img.cols-1),CLIP(ix,0,img.rows-1))*(1.0-tx)+
                img.at<uchar>(CLIP(iy+1,0,img.cols-1),CLIP(ix+1,0,img.rows-1))* tx;
            samples(i,j)=(uchar)(a * (1.0 - ty) + b * ty);
        }
    }
}
void resample(const Mat& img,const Rect2d& r2,Mat_<uchar>& samples){
    if(true){
        float x,y,a,b,tx,ty;int ix,iy;
        for(int i=0;i<samples.rows;i++){
            y=r2.y+i*r2.height/samples.rows;
            iy=cvFloor(y);ty=y-iy;
            for(int j=0;j<samples.cols;j++){
                x=r2.x+j*r2.width/samples.cols;
                ix=cvFloor(x);tx=x-ix;
                a=img.at<uchar>(CLIP(iy,0,img.cols-1),CLIP(ix,0,img.rows-1))*(1.0-tx)+
                    img.at<uchar>(CLIP(iy,0,img.cols-1),CLIP(ix+1,0,img.rows-1))* tx;
                b=img.at<uchar>(CLIP(iy+1,0,img.cols-1),CLIP(ix,0,img.rows-1))*(1.0-tx)+
                    img.at<uchar>(CLIP(iy+1,0,img.cols-1),CLIP(ix+1,0,img.rows-1))* tx;
                samples(i,j)=(uchar)(a * (1.0 - ty) + b * ty);
            }
        }
    }else{
        Point2f center(r2.x+r2.width/2,r2.y+r2.height/2);
        return resample(img,RotatedRect(center,Size2f(r2.width,r2.height),0.0),samples);
    }
}

//other stuff
void TLDEnsembleClassifier::stepPrefSuff(uchar* arr,int len){
    int gridSize=getGridSize();
    if(false){
        int step=len/(gridSize-1), pref=(len-step*(gridSize-1))/2;
        for(int i=0;i<(sizeof(x1)/sizeof(x1[0]));i++){
            arr[i]=pref+arr[i]*step;
        }
    }else{
        int total=len-gridSize;
        int quo=total/(gridSize-1),rem=total%(gridSize-1);
        int smallStep=quo,bigStep=quo+1;
        int bigOnes=rem,smallOnes=gridSize-bigOnes-1;
        int bigOnes_front=bigOnes/2,bigOnes_back=bigOnes-bigOnes_front;
        for(int i=0;i<(sizeof(x1)/sizeof(x1[0]));i++){
            if(arr[i]<bigOnes_back){
                arr[i]=arr[i]*bigStep+arr[i];
                continue;
            }
            if(arr[i]<(bigOnes_front+smallOnes)){
                arr[i]=bigOnes_front*bigStep+(arr[i]-bigOnes_front)*smallStep+arr[i];
                continue;
            }
            if(arr[i]<(bigOnes_front+smallOnes+bigOnes_back)){
                arr[i]=bigOnes_front*bigStep+smallOnes*smallStep+(arr[i]-(bigOnes_front+smallOnes))*bigStep+arr[i];
                continue;
            }
            arr[i]=len-1;
        }
    }
}
TLDEnsembleClassifier::TLDEnsembleClassifier(int ordinal,Size size){
    preinit(ordinal);
    stepPrefSuff(x1,size.width);
    stepPrefSuff(x2,size.width);
    stepPrefSuff(y1,size.height);
    stepPrefSuff(y2,size.height);
}
void TLDEnsembleClassifier::integrate(Mat_<uchar> patch,bool isPositive){
    unsigned short int position=code(patch.data,patch.step[0]);
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
    char codeS[20];
    for(int i=0;i<(sizeof(x1)/sizeof(x1[0]));i++,position<<1){
        if(*(data+rowstep*y1[i]+x1[i])<*(data+rowstep*y2[i]+x2[i])){
            position++;
            codeS[i]='o';
        }else{
            codeS[i]='x';
        }
    }
    codeS[13]='\0';
    //printf("integrate with code %s\n",codeS);
    return position;
}

}
