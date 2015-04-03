/*M///////////////////////////////////////////////////////////////////////////////////////
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
// Copyright (C) 2010-2013, University of Nizhny Novgorod, all rights reserved.
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
#include <stdio.h>
#include "string.h"
#include "_lsvmc_parser.h"
#include "_lsvmc_error.h"

namespace cv
{
namespace lsvm
{

int isMODEL        (char *str);
int isP            (char *str);
int isSCORE        (char *str);
int isCOMP         (char *str);
int isRFILTER      (char *str);
int isPFILTERs     (char *str);
int isPFILTER      (char *str);
int isSIZEX        (char *str);
int isSIZEY        (char *str);
int isWEIGHTS      (char *str);
int isV            (char *str);
int isVx           (char *str);
int isVy           (char *str);
int isD            (char *str);
int isDx           (char *str);
int isDy           (char *str);
int isDxx          (char *str);
int isDyy          (char *str);
int isB            (char *str);
int isWEIGHTS_PCA  (char *str);
int isPCA          (char *str);
int isPCAcoeff     (char *str);
int isCASCADE_Th   (char *str);
int isHYPOTHES_PCA (char *str);
int isDEFORM_PCA   (char *str);
int isHYPOTHES     (char *str);
int isDEFORM       (char *str);
int getTeg         (char *str);

void addFilter(CvLSVMFilterObjectCascade *** model, int *last, int *max);

void parserCascadeThresholds  (FILE * xmlf, CvLSVMFilterObjectCascade * model);

void parserRFilter  (FILE * xmlf, int p, int pca, CvLSVMFilterObjectCascade * model, float *b);

void parserV  (FILE * xmlf, int /*p*/, CvLSVMFilterObjectCascade * model);

void parserD  (FILE * xmlf, int /*p*/, CvLSVMFilterObjectCascade * model);

void parserPFilter  (FILE * xmlf, int p, int pca, int /*N_path*/, CvLSVMFilterObjectCascade * model);

void parserPFilterS (FILE * xmlf, int p, int pca, CvLSVMFilterObjectCascade *** model, int *last, int *max);

void parserComp (FILE * xmlf, int p, int pca, int *N_comp, CvLSVMFilterObjectCascade *** model, float *b, int *last, int *max);

void parserModel(FILE * xmlf, CvLSVMFilterObjectCascade *** model, int *last, int *max, int **comp, float **b, int *count, float * score, float** PCAcoeff);

void LSVMparser(const char * filename,
                CvLSVMFilterObjectCascade *** model,
                int *last,
                int *max,
                int **comp,
                float **b,
                int *count,
                float * score,
                float** PCAcoeff);

int isMODEL    (char *str){
    char stag [] = "<Model>";
    char etag [] = "</Model>";
    if(strcmp(stag, str) == 0)return  MODEL;
    if(strcmp(etag, str) == 0)return EMODEL;
    return 0;
}
int isP        (char *str){
    char stag [] = "<P>";
    char etag [] = "</P>";
    if(strcmp(stag, str) == 0)return  P;
    if(strcmp(etag, str) == 0)return EP;
    return 0;
}
int isSCORE        (char *str){
    char stag [] = "<ScoreThreshold>";
    char etag [] = "</ScoreThreshold>";
    if(strcmp(stag, str) == 0)return  SCORE;
    if(strcmp(etag, str) == 0)return ESCORE;
    return 0;
}
int isCOMP     (char *str){
    char stag [] = "<Component>";
    char etag [] = "</Component>";
    if(strcmp(stag, str) == 0)return  COMP;
    if(strcmp(etag, str) == 0)return ECOMP;
    return 0;
}
int isRFILTER  (char *str){
    char stag [] = "<RootFilter>";
    char etag [] = "</RootFilter>";
    if(strcmp(stag, str) == 0)return  RFILTER;
    if(strcmp(etag, str) == 0)return ERFILTER;
    return 0;
}
int isPFILTERs (char *str){
    char stag [] = "<PartFilters>";
    char etag [] = "</PartFilters>";
    if(strcmp(stag, str) == 0)return  PFILTERs;
    if(strcmp(etag, str) == 0)return EPFILTERs;
    return 0;
}
int isPFILTER  (char *str){
    char stag [] = "<PartFilter>";
    char etag [] = "</PartFilter>";
    if(strcmp(stag, str) == 0)return  PFILTER;
    if(strcmp(etag, str) == 0)return EPFILTER;
    return 0;
}
int isSIZEX    (char *str){
    char stag [] = "<sizeX>";
    char etag [] = "</sizeX>";
    if(strcmp(stag, str) == 0)return  SIZEX;
    if(strcmp(etag, str) == 0)return ESIZEX;
    return 0;
}
int isSIZEY    (char *str){
    char stag [] = "<sizeY>";
    char etag [] = "</sizeY>";
    if(strcmp(stag, str) == 0)return  SIZEY;
    if(strcmp(etag, str) == 0)return ESIZEY;
    return 0;
}
int isWEIGHTS  (char *str){
    char stag [] = "<Weights>";
    char etag [] = "</Weights>";
    if(strcmp(stag, str) == 0)return  WEIGHTS;
    if(strcmp(etag, str) == 0)return EWEIGHTS;
    return 0;
}
int isV        (char *str){
    char stag [] = "<V>";
    char etag [] = "</V>";
    if(strcmp(stag, str) == 0)return  TAGV;
    if(strcmp(etag, str) == 0)return ETAGV;
    return 0;
}
int isVx       (char *str){
    char stag [] = "<Vx>";
    char etag [] = "</Vx>";
    if(strcmp(stag, str) == 0)return  Vx;
    if(strcmp(etag, str) == 0)return EVx;
    return 0;
}
int isVy       (char *str){
    char stag [] = "<Vy>";
    char etag [] = "</Vy>";
    if(strcmp(stag, str) == 0)return  Vy;
    if(strcmp(etag, str) == 0)return EVy;
    return 0;
}
int isD        (char *str){
    char stag [] = "<Penalty>";
    char etag [] = "</Penalty>";
    if(strcmp(stag, str) == 0)return  TAGD;
    if(strcmp(etag, str) == 0)return ETAGD;
    return 0;
}
int isDx       (char *str){
    char stag [] = "<dx>";
    char etag [] = "</dx>";
    if(strcmp(stag, str) == 0)return  Dx;
    if(strcmp(etag, str) == 0)return EDx;
    return 0;
}
int isDy       (char *str){
    char stag [] = "<dy>";
    char etag [] = "</dy>";
    if(strcmp(stag, str) == 0)return  Dy;
    if(strcmp(etag, str) == 0)return EDy;
    return 0;
}
int isDxx      (char *str){
    char stag [] = "<dxx>";
    char etag [] = "</dxx>";
    if(strcmp(stag, str) == 0)return  Dxx;
    if(strcmp(etag, str) == 0)return EDxx;
    return 0;
}
int isDyy      (char *str){
    char stag [] = "<dyy>";
    char etag [] = "</dyy>";
    if(strcmp(stag, str) == 0)return  Dyy;
    if(strcmp(etag, str) == 0)return EDyy;
    return 0;
}
int isB      (char *str){
    char stag [] = "<LinearTerm>";
    char etag [] = "</LinearTerm>";
    if(strcmp(stag, str) == 0)return  BTAG;
    if(strcmp(etag, str) == 0)return EBTAG;
    return 0;
}

int isWEIGHTS_PCA  (char *str){
    char stag [] = "<WeightsPCA>";
    char etag [] = "</WeightsPCA>";
    if(strcmp(stag, str) == 0)return  WEIGHTSPCA;
    if(strcmp(etag, str) == 0)return EWEIGHTSPCA;
    return 0;
}

int isPCA  (char *str){
    char stag [] = "<PCA>";
    char etag [] = "</PCA>";
    if(strcmp(stag, str) == 0)return  PCA;
    if(strcmp(etag, str) == 0)return EPCA;
    return 0;
}

int isPCAcoeff  (char *str){
    char stag [] = "<PCAcoeff>";
    char etag [] = "</PCAcoeff>";
    if(strcmp(stag, str) == 0)return  PCACOEFF;
    if(strcmp(etag, str) == 0)return EPCACOEFF;
    return 0;
}

int isCASCADE_Th  (char *str){
    char stag [] = "<CascadeThresholds>";
    char etag [] = "</CascadeThresholds>";
    if(strcmp(stag, str) == 0)return  CASCADE_Th;
    if(strcmp(etag, str) == 0)return ECASCADE_Th;
    return 0;
}

int isHYPOTHES_PCA  (char *str){
    char stag [] = "<HypothesisThresholdPCA>";
    char etag [] = "</HypothesisThresholdPCA>";
    if(strcmp(stag, str) == 0)return  HYPOTHES_PCA;
    if(strcmp(etag, str) == 0)return EHYPOTHES_PCA;
    return 0;
}
int isDEFORM_PCA  (char *str){
    char stag [] = "<DeformationThresholdPCA>";
    char etag [] = "</DeformationThresholdPCA>";
    if(strcmp(stag, str) == 0)return  DEFORM_PCA;
    if(strcmp(etag, str) == 0)return EDEFORM_PCA;
    return 0;
}
int isHYPOTHES  (char *str){
    char stag [] = "<HypothesisThreshold>";
    char etag [] = "</HypothesisThreshold>";
    if(strcmp(stag, str) == 0)return  HYPOTHES;
    if(strcmp(etag, str) == 0)return EHYPOTHES;
    return 0;
}
int isDEFORM  (char *str){
    char stag [] = "<DeformationThreshold>";
    char etag [] = "</DeformationThreshold>";
    if(strcmp(stag, str) == 0)return  DEFORM;
    if(strcmp(etag, str) == 0)return EDEFORM;
    return 0;
}

int getTeg(char *str){
    int sum = 0;
    sum = isMODEL (str)+
    isP        (str)+
    isSCORE    (str)+
    isCOMP     (str)+
    isRFILTER  (str)+
    isPFILTERs (str)+
    isPFILTER  (str)+
    isSIZEX    (str)+
    isSIZEY    (str)+
    isWEIGHTS  (str)+
    isV        (str)+
    isVx       (str)+
    isVy       (str)+
    isD        (str)+
    isDx       (str)+
    isDy       (str)+
    isDxx      (str)+
    isDyy      (str)+
    isB        (str)+
    isPCA         (str)+
    isCASCADE_Th  (str)+
    isHYPOTHES_PCA(str)+
    isDEFORM_PCA  (str)+
    isHYPOTHES    (str)+
    isDEFORM      (str)+
    isWEIGHTS_PCA (str)+
    isPCAcoeff    (str)
    ;

    return sum;
}

void addFilter(CvLSVMFilterObjectCascade *** model, int *last, int *max)
{
    CvLSVMFilterObjectCascade ** nmodel;
    int i;
    (*last) ++;
    if((*last) >= (*max)){
        (*max) += 10;
        nmodel = (CvLSVMFilterObjectCascade **)malloc(sizeof(CvLSVMFilterObjectCascade *) * (*max));
        for(i = 0; i < *last; i++){
            nmodel[i] = (* model)[i];
        }
        free(* model);
        (*model) = nmodel;
    }
    (*model) [(*last)] = (CvLSVMFilterObjectCascade *)malloc(sizeof(CvLSVMFilterObjectCascade));
    (*model) [(*last)]->Hypothesis      = 0.0f;
    (*model) [(*last)]->Deformation     = 0.0f;
    (*model) [(*last)]->Hypothesis_PCA  = 0.0f;
    (*model) [(*last)]->Deformation_PCA = 0.0f;

}

//##############################################
void parserCascadeThresholds  (FILE * xmlf, CvLSVMFilterObjectCascade * model){
    int st = 0;
    int tag;
    int tagVal;
    char ch;
    int i,j;
    char buf[1024];
    char tagBuf[1024];
    
    i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = (char) fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';
                
                tagVal = getTeg(tagBuf);
               
                if(tagVal == ECASCADE_Th){
                    return;
                }
                if(tagVal == HYPOTHES_PCA){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EHYPOTHES_PCA){
                    st = 0;
                    buf[i] = '\0';
                    model->Hypothesis_PCA =(float) atof(buf);
                }
                if(tagVal == DEFORM_PCA){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EDEFORM_PCA){
                    st = 0;
                    buf[i] = '\0';
                    model->Deformation_PCA =(float) atof(buf);
                }
                if(tagVal == HYPOTHES){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EHYPOTHES){
                    st = 0;
                    buf[i] = '\0';
                    model->Hypothesis = (float)atof(buf);
                }
                if(tagVal == DEFORM){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EDEFORM){
                    st = 0;
                    buf[i] = '\0';
                    model->Deformation = (float)atof(buf);
                }
                tag = 0;
                i   = 0;                
            }else{
                if((tag == 0)&& (st == 1)){
                    buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }        
    }
}
//##############################################

void parserRFilter  (FILE * xmlf, int p, int pca, CvLSVMFilterObjectCascade * model, float *b){
    int st = 0;
    int sizeX = 0, sizeY = 0;
    int tag;
    int tagVal;
    char ch;
    int i,j,ii;
    char buf[1024];
    char tagBuf[1024];
    double *data;
    //printf("<RootFilter>\n");

    model->V.x = 0;
    model->V.y = 0;
    model->V.l = 0;
    model->fineFunction[0] = 0.0;
    model->fineFunction[1] = 0.0;
    model->fineFunction[2] = 0.0;
    model->fineFunction[3] = 0.0;

    i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = (char)fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';
                
                tagVal = getTeg(tagBuf);
               
                if(tagVal == ERFILTER){
                    //printf("</RootFilter>\n");
                    return;
                }
                if(tagVal == SIZEX){
                    st = 1;
                    i = 0;
                }
                if(tagVal == ESIZEX){
                    st = 0;
                    buf[i] = '\0';
                    sizeX = atoi(buf);
                    model->sizeX = sizeX;
                    //printf("<sizeX>%d</sizeX>\n", sizeX);
                }
                if(tagVal == SIZEY){
                    st = 1;
                    i = 0;
                }
                if(tagVal == ESIZEY){
                    st = 0;
                    buf[i] = '\0';
                    sizeY = atoi(buf);
                    model->sizeY = sizeY;
                    //printf("<sizeY>%d</sizeY>\n", sizeY);
                }
                if(tagVal == WEIGHTS){
                    data = (double *)malloc( sizeof(double) * p * sizeX * sizeY);
                    CV_Assert(fread(data, sizeof(double), p * sizeX * sizeY, xmlf));
                    model->H = (float *)malloc(sizeof(float)* p * sizeX * sizeY);
                    for(ii = 0; ii < p * sizeX * sizeY; ii++){
                        model->H[ii] = (float)data[ii];
                    }
                    free(data);
                }
                if(tagVal == WEIGHTSPCA){
                    data = (double *)malloc( sizeof(double) * pca * sizeX * sizeY);
                    CV_Assert(fread(data, sizeof(double), pca * sizeX * sizeY, xmlf));
                    model->H_PCA = (float *)malloc(sizeof(float)* pca * sizeX * sizeY);
                    for(ii = 0; ii < pca * sizeX * sizeY; ii++){
                        model->H_PCA[ii] = (float)data[ii];
                    }
                    free(data);
                }

                if(tagVal == CASCADE_Th){
                    parserCascadeThresholds  (xmlf, model);
                }

                if(tagVal == BTAG){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EBTAG){
                    st = 0;
                    buf[i] = '\0';
                    *b =(float) atof(buf);
                    //printf("<B>%f</B>\n", *b);
                }

                tag = 0;
                i   = 0;                
            }else{
                if((tag == 0)&& (st == 1)){
                    buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }        
    }
}

void parserV  (FILE * xmlf, int /*p*/, CvLSVMFilterObjectCascade * model){
    int st = 0;
    int tag;
    int tagVal;
    char ch;
    int i,j;
    char buf[1024];
    char tagBuf[1024];
    //printf("    <V>\n");

    i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = (char) fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';
                
                tagVal = getTeg(tagBuf);
               
                if(tagVal == ETAGV){
                    //printf("    </V>\n");
                    return;
                }
                if(tagVal == Vx){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EVx){
                    st = 0;
                    buf[i] = '\0';
                    model->V.x = atoi(buf);
                    //printf("        <Vx>%d</Vx>\n", model->V.x);
                }
                if(tagVal == Vy){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EVy){
                    st = 0;
                    buf[i] = '\0';
                    model->V.y = atoi(buf);
                    //printf("        <Vy>%d</Vy>\n", model->V.y);
                }
                tag = 0;
                i   = 0;                
            }else{
                if((tag == 0)&& (st == 1)){
                    buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }        
    }
}
void parserD  (FILE * xmlf, int /*p*/, CvLSVMFilterObjectCascade * model){
    int st = 0;
    int tag;
    int tagVal;
    char ch;
    int i,j;
    char buf[1024];
    char tagBuf[1024];
    //printf("    <D>\n");

    i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = (char)fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';
                
                tagVal = getTeg(tagBuf);
               
                if(tagVal == ETAGD){
                    //printf("    </D>\n");
                    return;
                }
                if(tagVal == Dx){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EDx){
                    st = 0;
                    buf[i] = '\0';
                    
                    model->fineFunction[0] = (float)atof(buf);
                    //printf("        <Dx>%f</Dx>\n", model->fineFunction[0]);
                }
                if(tagVal == Dy){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EDy){
                    st = 0;
                    buf[i] = '\0';
                    
                    model->fineFunction[1] = (float)atof(buf);
                    //printf("        <Dy>%f</Dy>\n", model->fineFunction[1]);
                }
                if(tagVal == Dxx){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EDxx){
                    st = 0;
                    buf[i] = '\0';
                    
                    model->fineFunction[2] = (float)atof(buf);
                    //printf("        <Dxx>%f</Dxx>\n", model->fineFunction[2]);
                }
                if(tagVal == Dyy){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EDyy){
                    st = 0;
                    buf[i] = '\0';
                    
                    model->fineFunction[3] = (float)atof(buf);
                    //printf("        <Dyy>%f</Dyy>\n", model->fineFunction[3]);
                }

                tag = 0;
                i   = 0;                
            }else{
                if((tag == 0)&& (st == 1)){
                    buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }        
    }
}

void parserPFilter  (FILE * xmlf, int p, int pca, int /*N_path*/, CvLSVMFilterObjectCascade * model){
    int st = 0;
    int sizeX = 0, sizeY = 0;
    int tag;
    int tagVal;
    char ch;
    int i,j, ii;
    char buf[1024];
    char tagBuf[1024];
    double *data;
    //printf("<PathFilter> (%d)\n", N_path);

    model->V.x = 0;
    model->V.y = 0;
    model->V.l = 0;
    model->fineFunction[0] = 0.0f;
    model->fineFunction[1] = 0.0f;
    model->fineFunction[2] = 0.0f;
    model->fineFunction[3] = 0.0f;

    i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = (char)fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';
                
                tagVal = getTeg(tagBuf);
               
                if(tagVal == EPFILTER){
                    //printf("</PathFilter>\n");
                    return;
                }

                if(tagVal == TAGV){
                    parserV(xmlf, p, model);
                }
                if(tagVal == TAGD){
                    parserD(xmlf, p, model);
                }
                if(tagVal == SIZEX){
                    st = 1;
                    i = 0;
                }
                if(tagVal == ESIZEX){
                    st = 0;
                    buf[i] = '\0';
                    sizeX = atoi(buf);
                    model->sizeX = sizeX;
                    //printf("<sizeX>%d</sizeX>\n", sizeX);
                }
                if(tagVal == SIZEY){
                    st = 1;
                    i = 0;
                }
                if(tagVal == ESIZEY){
                    st = 0;
                    buf[i] = '\0';
                    sizeY = atoi(buf);
                    model->sizeY = sizeY;
                    //printf("<sizeY>%d</sizeY>\n", sizeY);
                }
                if(tagVal == WEIGHTS){
                    data = (double *)malloc( sizeof(double) * p * sizeX * sizeY);
                    CV_Assert(fread(data, sizeof(double), p * sizeX * sizeY, xmlf));
                    model->H = (float *)malloc(sizeof(float)* p * sizeX * sizeY);
                    for(ii = 0; ii < p * sizeX * sizeY; ii++){
                        model->H[ii] = (float)data[ii];
                    }
                    free(data);
                }
                if(tagVal == WEIGHTSPCA){
                    data = (double *)malloc( sizeof(double) * pca * sizeX * sizeY);
                    CV_Assert(fread(data, sizeof(double), pca * sizeX * sizeY, xmlf));
                    model->H_PCA = (float *)malloc(sizeof(float)* pca * sizeX * sizeY);
                    for(ii = 0; ii < pca * sizeX * sizeY; ii++){
                        model->H_PCA[ii] = (float)data[ii];
                    }
                    free(data);
                }
                if(tagVal == CASCADE_Th){
                    parserCascadeThresholds  (xmlf, model);
                }
                if(tagVal == EWEIGHTS){
                    //printf("WEIGHTS OK\n");
                }
                tag = 0;
                i   = 0;                
            }else{
                if((tag == 0)&& (st == 1)){
                    buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }        
    }
}
void parserPFilterS (FILE * xmlf, int p, int pca, CvLSVMFilterObjectCascade *** model, int *last, int *max){
    int st = 0;
    int N_path = 0;
    int tag;
    int tagVal;
    char ch;
    int j;
    char tagBuf[1024];
    //printf("<PartFilters>\n");

    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = (char)fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';
                
                tagVal = getTeg(tagBuf);
               
                if(tagVal == EPFILTERs){
                    //printf("</PartFilters>\n");
                    return;
                }
                if(tagVal == PFILTER){
                    addFilter(model, last, max);
                    parserPFilter  (xmlf, p, pca, N_path, (*model)[*last]);
                    N_path++;
                }
                tag = 0;              
            }else{
                if((tag != 0) || (st != 1)){
                    tagBuf[j] = ch; j++;
                }
            }
        }        
    }
}
void parserComp (FILE * xmlf, int p, int pca, int *N_comp, CvLSVMFilterObjectCascade *** model, float *b, int *last, int *max){
    int st = 0;
    int tag;
    int tagVal;
    char ch;
    int j;
    char tagBuf[1024];
    //printf("<Component> %d\n", *N_comp);

    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = (char)fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';
                
                tagVal = getTeg(tagBuf);
               
                if(tagVal == ECOMP){
                    (*N_comp) ++;
                    return;
                }
                if(tagVal == RFILTER){
                    addFilter(model, last, max);
                    parserRFilter   (xmlf, p, pca, (*model)[*last],b);
                }
                if(tagVal == PFILTERs){
                    parserPFilterS  (xmlf, p, pca, model, last, max);
                }
                tag = 0;              
            }else{
                if((tag != 0) || (st != 1)){
                    tagBuf[j] = ch; j++;
                }
            }
        }        
    }
}
void parserModel(FILE * xmlf, CvLSVMFilterObjectCascade *** model, int *last, int *max, int **comp, float **b, int *count, float * score, float** PCAcoeff){
    int p = 0, pca = 0;
    int N_comp = 0;
    int * cmp;
    float *bb;
    int st = 0;
    int tag;
    int tagVal;
    char ch;
    int i,j, ii = 0, jj;
    char buf[1024];
    char tagBuf[1024];
    double *data;
    //printf("<Model>\n");
    
    i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = (char)fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';
                
                tagVal = getTeg(tagBuf);
               
                if(tagVal == EMODEL){
                    //printf("</Model>\n");
                    for(ii = 0; ii <= *last; ii++){
                        (*model)[ii]->numFeatures = p;
                    }
                    * count = N_comp;
                    return;
                }
                if(tagVal == COMP){
                    if(N_comp == 0){
                        cmp = (int   *)malloc(sizeof(int));
                        bb  = (float *)malloc(sizeof(float));
                        * comp = cmp;
                        * b    = bb;
                        * count = N_comp + 1; 
                    } else {
                        cmp = (int   *)malloc(sizeof(int)   * (N_comp + 1));
                        bb  = (float *)malloc(sizeof(float) * (N_comp + 1));
                        for(ii = 0; ii < N_comp; ii++){
                            cmp[ii] = (* comp)[ii];
                            bb [ii] = (* b   )[ii];
                        }
                        free(* comp);
                        free(* b   );
                        * comp = cmp;
                        * b    = bb;
                        * count = N_comp + 1; 
                    }
                    parserComp(xmlf, p, pca, &N_comp, model, &((*b)[N_comp]), last, max);
                    cmp[N_comp - 1] = *last;
                }
                if(tagVal == P){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EP){
                    st = 0;
                    buf[i] = '\0';
                    p = atoi(buf);
                    //printf("<P>%d</P>\n", p);
                }
                if(tagVal == PCA){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EPCA){
                    st = 0;
                    buf[i] = '\0';
                    pca = atoi(buf);
                    //printf("<PCA>%d</PCA>\n", p);
                }
                if(tagVal == SCORE){
                    st = 1;
                    i = 0;
                }
                if(tagVal == PCACOEFF){
                    st = 0;
                    i = 0;
                    p--;
                    data = (double *)malloc( sizeof(double) * p * p);
                    (*PCAcoeff) = (float *)malloc( sizeof(float) * p * p);
                    CV_Assert(fread(data, sizeof(double), p * p, xmlf));
                    for(jj = 0; jj < p * p; jj++){
                      (*PCAcoeff)[jj] = (float)data[jj];
                    }
                    free(data);
                    p++;
                }
                if(tagVal == EPCACOEFF){
                    st = 0;
                    //printf("<PCA>%d</PCA>\n", p);
                }
                if(tagVal == SCORE){
                    st = 1;
                    i = 0;
                }
                if(tagVal == ESCORE){
                    st = 0;
                    buf[i] = '\0';
                    *score = (float)atof(buf);
                    //printf("<ScoreThreshold>%f</ScoreThreshold>\n", score);
                }
                tag = 0;
                i   = 0;                
            }else{
                if((tag == 0)&& (st == 1)){
                    buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }        
    }
}

void LSVMparser(const char * filename, 
                CvLSVMFilterObjectCascade *** model, 
                int *last, 
                int *max, 
                int **comp, 
                float **b, 
                int *count, 
                float * score,
                float** PCAcoeff)
{
    int tag;
    char ch;
    int j;
    FILE *xmlf;
    char tagBuf[1024];

    (*max) = 10;
    (*last) = -1;
    (*model) = (CvLSVMFilterObjectCascade ** )malloc((sizeof(CvLSVMFilterObjectCascade * )) * (*max));

    //printf("parse : %s\n", filename);
    xmlf = fopen(filename, "rb");
    
    j   = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = (char) fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tag = 0;
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';
                if(getTeg(tagBuf) == MODEL){
                    parserModel(xmlf, model, last, max, comp, b, count, score, PCAcoeff);
                }
            }else{
                if(tag != 0)
                {
                    tagBuf[j] = ch; j++;
                }
            }
        }        
    }
    fclose(xmlf);
}

int loadModel(
              const char *modelPath,
             
              CvLSVMFilterObjectCascade ***filters,
              int *kFilters, 
              int *kComponents, 
              int **kPartFilters, 
              float **b, 
              float *scoreThreshold,
              float ** PCAcoeff){ 
    int last;
    int max;
    int *comp;
    int count;
    int i;
    float score;

    LSVMparser(modelPath, filters, &last, &max, &comp, b, &count, &score, PCAcoeff);
    (*kFilters)       = last + 1;
    (*kComponents)    = count;
    (*scoreThreshold) = (float) score;

    (*kPartFilters) = (int *)malloc(sizeof(int) * count);

    for(i = 1; i < count;i++){
        (*kPartFilters)[i] = (comp[i] - comp[i - 1]) - 1;
    }
    (*kPartFilters)[0] = comp[0];


    for(i = 0; i < (*kFilters);i++){
        (*(filters))[i]->deltaX = 5;// maxX;
        (*(filters))[i]->deltaY = 5;// maxY;
    }

    return 0;
}
}
}
