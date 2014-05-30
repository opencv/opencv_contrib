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
//#include "opencv2/line_descriptor/descriptor.hpp"

using namespace cv;

/* return default parameters */
BinaryDescriptor::Params::Params()
{
    LowestThreshold = 0.35;
    NNDRThreshold = 0.6;
    ksize_ = 5;
    numOfOctave_ = 5;
    numOfBand_ = 9;
    widthOfBand_ = 7;
    reductionRatio = 2;

}

/* read parameters from a FileNode object and store them (struct function) */
void BinaryDescriptor::Params::read(const cv::FileNode& fn )
{
    LowestThreshold = fn["LowestThreshold"];
    NNDRThreshold = fn["NNDRThreshold"];
    ksize_ = fn["ksize_"];
    numOfOctave_ = fn["numOfOctave_"];
    numOfBand_ = fn["numOfBand_"];
    widthOfBand_ = fn["widthOfBand_"];
    reductionRatio = fn["reductionRatio"];
}

/* store parameters to a FileStorage object (struct function) */
void BinaryDescriptor::Params::write(cv::FileStorage& fs) const
{
    fs << "LowestThreshold" << LowestThreshold;
    fs << "NNDRThreshold" << NNDRThreshold;
    fs << "ksize_" <<  ksize_;
    fs << "numOfOctave_" << numOfOctave_;
    fs << "numOfBand_" <<  numOfBand_;
    fs << "widthOfBand_" << widthOfBand_;
    fs << "reductionRatio" << reductionRatio;
}

/* construct a BinaryDescrptor object and compute external private parameters */
BinaryDescriptor::BinaryDescriptor(const BinaryDescriptor::Params &parameters) : params(parameters)
{

    /* prepare a vector to host local weights F_l*/
    gaussCoefL_.resize(params.widthOfBand_*3);

    /* compute center of central band (every computation involves 2-3 bands) */
    double u = (params.widthOfBand_*3-1)/2;

    /* compute exponential part of F_l */
    double sigma = (params.widthOfBand_*2+1)/2;// (widthOfBand_*2+1)/2;
    double invsigma2 = -1/(2*sigma*sigma);

    /* compute all local weights */
    double dis;
    for(int i=0; i<params.widthOfBand_*3; i++)
    {
      dis = i-u;
      gaussCoefL_[i] = exp(dis*dis*invsigma2);
    }

    /* prepare a vector for global weights F_g*/
    gaussCoefG_.resize(params.numOfBand_*params.widthOfBand_);

    /* compute center of LSR */
    u = (params.numOfBand_*params.widthOfBand_-1)/2;

    /* compute exponential part of F_g */
    sigma = u;
    invsigma2 = -1/(2*sigma*sigma);
    for(int i=0; i<params.numOfBand_*params.widthOfBand_; i++)
    {
     dis = i-u;
     gaussCoefG_[i] = exp(dis*dis*invsigma2);
    }
}

/* read parameters from a FileNode object and store them (class function ) */
void BinaryDescriptor::read( const cv::FileNode& fn )
{
    params.read(fn);
}

/* store parameters to a FileStorage object (class function) */
void BinaryDescriptor::write( cv::FileStorage& fs ) const
{
    params.write(fs);
}

/* extract lines from an image and compute their descriptors */
void BinaryDescriptor::getLineBinaryDescriptor(cv::Mat & oct_binaryDescMat, ScaleLines &keyLines)
{
    /* prepare a matrix to store Gaussian pyramid of input matrix */
    std::vector<cv::Mat> matVec(params.numOfOctave_);

    /* reinitialize structures for hosting images' derivatives and sizes
    (they may have been used in the past) */
    dxImg_vector.clear();
    dyImg_vector.clear();
    images_sizes.clear();

    dxImg_vector.resize(params.numOfOctave_);
    dyImg_vector.resize(params.numOfOctave_);
    images_sizes.resize(params.numOfOctave_);

    /* insert input image into pyramid */
    cv::Mat currentMat = oct_binaryDescMat.clone();
    matVec.push_back(currentMat);
    images_sizes.push_back(currentMat.size());

    /* compute and store derivatives of input image */
    cv:Mat currentDx, currentDy;
    cv::Sobel( currentMat, currentDx, CV_16SC1, 1, 0, 3);
    cv::Sobel( currentMat, currentDy, CV_16SC1, 0, 1, 3);

    dxImg_vector.push_back(currentDx);
    dyImg_vector.push_back(currentDy);

    /* fill Gaussian pyramid */
    for(int i = 1; i<params.numOfOctave_; i++)
    {
        /* compute and store next image in pyramid and its size */
        pyrDown( currentMat, currentMat, Size( currentMat.cols/params.reductionRatio, currentMat.rows/params.reductionRatio ));
        matVec.push_back(currentMat);
        images_sizes.push_back(currentMat.size());

        /* compute and store derivatives of new image */
        cv::Sobel( currentMat, currentDx, CV_16SC1, 1, 0, 3);
        cv::Sobel( currentMat, currentDy, CV_16SC1, 0, 1, 3);

        dxImg_vector.push_back(currentDx);
        dyImg_vector.push_back(currentDy);
    }

    /* compute descriptors */
    getLineBinaryDescriptorImpl(matVec, keyLines);

}

/* compute descriptors */
void BinaryDescriptor::getLineBinaryDescriptorImpl(std::vector<cv::Mat> & oct_binaryDescMat, ScaleLines &keyLines)
{
    for(size_t scaleCounter = 0; scaleCounter<oct_binaryDescMat.size(); scaleCounter++)
    {
        /* get current scaled image */
        cv::Mat currentScaledImage = oct_binaryDescMat[scaleCounter];

        /* create an LSD detector and store a pointer to it */
        cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);

        /* prepare a vectore to host extracted segments */
        std::vector<cv::Vec4i> lines_std;

        /* use detector to extract segments */
        ls->detect(currentScaledImage, lines_std);

        /* store information for every extracted segment */
        for (size_t  lineCounter = 0; lineCounter<lines_std.size(); lineCounter++)
        {
             /* get current segment and store its extremes */
             const cv::Vec4i v = lines_std[lineCounter];
             cv::Point b(v[0], v[1]);
             cv::Point e(v[2], v[3]);

            /* create an object to store line information */
            OctaveSingleLine osl;
            osl.startPointX = b.x;
            osl.startPointY = b.y;
            osl.endPointX = e.x;
            osl.endPointY = e.y;
            osl.sPointInOctaveX = b.x;
            osl.sPointInOctaveY = b.y;
            osl.ePointInOctaveX = e.x;
            osl.ePointInOctaveY = e.y;
            osl.direction = 0;
            osl.salience = 0;
            osl.lineLength = 0;
            osl.numOfPixels = std::sqrt((b.x-e.x)*(b.x-e.x) + (b.y-e.y)*(b.y-e.y));
            osl.octaveCount = scaleCounter;

            /* create a new LineVec and add new line to it */
            LinesVec lineScaleLs;
            lineScaleLs.push_back(osl);

            /* add current LineVec to other ones in the output list */
            keyLines.push_back(lineScaleLs);

       }

    }

    /* compute LBD descriptors */
    ComputeLBD_(keyLines);

}

/* power function with error management */
static inline int get2Pow(int i) {
    if(i>=0 && i<=7)
        return pow(2,i);

    else
    {
        CV_Assert(false);
        return -1;
    }
}

/* conversion of an LBD descriptor to the decimal equivalent of its binary representation */
unsigned char BinaryDescriptor::binaryTest(float* f1, float* f2)
{
    uchar result = 0;
     for(int i = 0; i<8; i++)
     {
         if(f1[i]>f2[i])
             result+=get2Pow(i);
     }

     return result;

}

/* compute LBD descriptors */
int BinaryDescriptor::ComputeLBD_(ScaleLines &keyLines)
{
    //the default length of the band is the line length.
    short numOfFinalLine = keyLines.size();
    float *dL = new float[2];//line direction cos(dir), sin(dir)
    float *dO = new float[2];//the clockwise orthogonal vector of line direction.
    short heightOfLSP = params.widthOfBand_*params.numOfBand_;//the height of line support region;
    short descriptorSize = params.numOfBand_ * 8;//each band, we compute the m( pgdL, ngdL,  pgdO, ngdO) and std( pgdL, ngdL,  pgdO, ngdO);
    float pgdLRowSum;//the summation of {g_dL |g_dL>0 } for each row of the region;
    float ngdLRowSum;//the summation of {g_dL |g_dL<0 } for each row of the region;
    float pgdL2RowSum;//the summation of {g_dL^2 |g_dL>0 } for each row of the region;
    float ngdL2RowSum;//the summation of {g_dL^2 |g_dL<0 } for each row of the region;
    float pgdORowSum;//the summation of {g_dO |g_dO>0 } for each row of the region;
    float ngdORowSum;//the summation of {g_dO |g_dO<0 } for each row of the region;
    float pgdO2RowSum;//the summation of {g_dO^2 |g_dO>0 } for each row of the region;
    float ngdO2RowSum;//the summation of {g_dO^2 |g_dO<0 } for each row of the region;

    float *pgdLBandSum  = new float[params.numOfBand_];//the summation of {g_dL |g_dL>0 } for each band of the region;
    float *ngdLBandSum  = new float[params.numOfBand_];//the summation of {g_dL |g_dL<0 } for each band of the region;
    float *pgdL2BandSum = new float[params.numOfBand_];//the summation of {g_dL^2 |g_dL>0 } for each band of the region;
    float *ngdL2BandSum = new float[params.numOfBand_];//the summation of {g_dL^2 |g_dL<0 } for each band of the region;
    float *pgdOBandSum  = new float[params.numOfBand_];//the summation of {g_dO |g_dO>0 } for each band of the region;
    float *ngdOBandSum  = new float[params.numOfBand_];//the summation of {g_dO |g_dO<0 } for each band of the region;
    float *pgdO2BandSum = new float[params.numOfBand_];//the summation of {g_dO^2 |g_dO>0 } for each band of the region;
    float *ngdO2BandSum = new float[params.numOfBand_];//the summation of {g_dO^2 |g_dO<0 } for each band of the region;

    short numOfBitsBand = params.numOfBand_*sizeof(float);
    short lengthOfLSP; //the length of line support region, varies with lines
    short halfHeight = (heightOfLSP-1)/2;
    short halfWidth;
    short bandID;
    float coefInGaussion;
    float lineMiddlePointX, lineMiddlePointY;
    float sCorX, sCorY,sCorX0, sCorY0;
    short tempCor, xCor, yCor;//pixel coordinates in image plane
    short dx, dy;
    float gDL;//store the gradient projection of pixels in support region along dL vector
    float gDO;//store the gradient projection of pixels in support region along dO vector
    short imageWidth, imageHeight, realWidth;
    short *pdxImg, *pdyImg;
    float *desVec;

    short sameLineSize;
    short octaveCount;
    OctaveSingleLine *pSingleLine;
    /* loop over list of LineVec */
    for(short lineIDInScaleVec = 0; lineIDInScaleVec<numOfFinalLine; lineIDInScaleVec++){
        sameLineSize = keyLines[lineIDInScaleVec].size();
        /* loop over current LineVec's lines */
        for(short lineIDInSameLine = 0; lineIDInSameLine<sameLineSize; lineIDInSameLine++){
            /* get a line in current LineVec and its original ID in its octave */
            pSingleLine = &(keyLines[lineIDInScaleVec][lineIDInSameLine]);
            octaveCount = pSingleLine->octaveCount;

            /* retrieve associated dxImg and dyImg
            pdxImg = edLineVec_[octaveCount]->dxImg_.ptr<short>();
            pdyImg = edLineVec_[octaveCount]->dyImg_.ptr<short>(); */
            pdxImg = dxImg_vector[octaveCount].ptr<short>();
            pdyImg = dyImg_vector[octaveCount].ptr<short>();

            /* get image size to work on from real one
            realWidth = edLineVec_[octaveCount]->imageWidth;
            imageWidth  = realWidth -1;
            imageHeight = edLineVec_[octaveCount]->imageHeight-1; */
            realWidth = images_sizes[octaveCount].width;
            imageWidth = realWidth - 1;
            imageHeight = images_sizes[octaveCount].height - 1;


            /* initialize memory areas */
            memset(pgdLBandSum,  0, numOfBitsBand);
            memset(ngdLBandSum, 0, numOfBitsBand);
            memset(pgdL2BandSum,  0, numOfBitsBand);
            memset(ngdL2BandSum, 0, numOfBitsBand);
            memset(pgdOBandSum,  0, numOfBitsBand);
            memset(ngdOBandSum, 0, numOfBitsBand);
            memset(pgdO2BandSum,  0, numOfBitsBand);
            memset(ngdO2BandSum, 0, numOfBitsBand);

            /* get length of line and its half */
            lengthOfLSP = keyLines[lineIDInScaleVec][lineIDInSameLine].numOfPixels;
            halfWidth   = (lengthOfLSP-1)/2;

            /* get middlepoint of line */
            lineMiddlePointX = 0.5 * (pSingleLine->sPointInOctaveX +  pSingleLine->ePointInOctaveX);
            lineMiddlePointY = 0.5 * (pSingleLine->sPointInOctaveY +  pSingleLine->ePointInOctaveY);

            /*1.rotate the local coordinate system to the line direction (direction is the angle
                between positive line direction and positive X axis)
             *2.compute the gradient projection of pixels in line support region*/

            /* get the vector representing original image reference system after rotation to aligh with
               line's direction */
            dL[0] = cos(pSingleLine->direction);
            dL[1] = sin(pSingleLine->direction);

            /* set the clockwise orthogonal vector of line direction */
            dO[0] = -dL[1];
            dO[1] = dL[0];

            /* get rotated reference frame */
            sCorX0= -dL[0]*halfWidth + dL[1]*halfHeight + lineMiddlePointX;//hID =0; wID = 0;
            sCorY0= -dL[1]*halfWidth - dL[0]*halfHeight + lineMiddlePointY;


            /* BIAS::Matrix<float> gDLMat(heightOfLSP,lengthOfLSP) */
            for(short hID = 0; hID <heightOfLSP; hID++){
                /*initialization */
                sCorX = sCorX0;
                sCorY = sCorY0;

                pgdLRowSum = 0;
                ngdLRowSum = 0;
                pgdORowSum = 0;
                ngdORowSum = 0;

                for(short wID = 0; wID <lengthOfLSP; wID++){
                    tempCor = round(sCorX);
                    xCor = (tempCor<0)?0:(tempCor>imageWidth)?imageWidth:tempCor;
                    tempCor = round(sCorY);
                    yCor = (tempCor<0)?0:(tempCor>imageHeight)?imageHeight:tempCor;

                    /* To achieve rotation invariance, each simple gradient is rotated aligned with
                     * the line direction and clockwise orthogonal direction.*/
                    dx = pdxImg[yCor*realWidth+xCor];
                    dy = pdyImg[yCor*realWidth+xCor];
                    gDL = dx * dL[0] + dy * dL[1];
                    gDO = dx * dO[0] + dy * dO[1];
                    if(gDL>0){
                        pgdLRowSum  += gDL;
                    }else{
                        ngdLRowSum  -= gDL;
                    }
                    if(gDO>0){
                        pgdORowSum  += gDO;
                    }else{
                        ngdORowSum  -= gDO;
                    }
                    sCorX +=dL[0];
                    sCorY +=dL[1];
                    /* gDLMat[hID][wID] = gDL; */
                }
                sCorX0 -=dL[1];
                sCorY0 +=dL[0];
                coefInGaussion = gaussCoefG_[hID];
                pgdLRowSum = coefInGaussion * pgdLRowSum;
                ngdLRowSum = coefInGaussion * ngdLRowSum;
                pgdL2RowSum = pgdLRowSum * pgdLRowSum;
                ngdL2RowSum = ngdLRowSum * ngdLRowSum;
                pgdORowSum = coefInGaussion * pgdORowSum;
                ngdORowSum = coefInGaussion * ngdORowSum;
                pgdO2RowSum = pgdORowSum * pgdORowSum;
                ngdO2RowSum = ngdORowSum * ngdORowSum;

                /* compute {g_dL |g_dL>0 }, {g_dL |g_dL<0 },
                {g_dO |g_dO>0 }, {g_dO |g_dO<0 } of each band in the line support region
                first, current row belong to current band */
                bandID = hID/params.widthOfBand_;
                coefInGaussion = gaussCoefL_[hID%params.widthOfBand_+params.widthOfBand_];
                pgdLBandSum[bandID] +=  coefInGaussion * pgdLRowSum;
                ngdLBandSum[bandID] +=  coefInGaussion * ngdLRowSum;
                pgdL2BandSum[bandID] +=  coefInGaussion * coefInGaussion * pgdL2RowSum;
                ngdL2BandSum[bandID] +=  coefInGaussion * coefInGaussion * ngdL2RowSum;
                pgdOBandSum[bandID] +=  coefInGaussion * pgdORowSum;
                ngdOBandSum[bandID] +=  coefInGaussion * ngdORowSum;
                pgdO2BandSum[bandID] +=  coefInGaussion * coefInGaussion * pgdO2RowSum;
                ngdO2BandSum[bandID] +=  coefInGaussion * coefInGaussion * ngdO2RowSum;

                /* In order to reduce boundary effect along the line gradient direction,
                 * a row's gradient will contribute not only to its current band, but also
                 * to its nearest upper and down band with gaussCoefL_.*/
                bandID--;
                if(bandID>=0){/* the band above the current band */
                    coefInGaussion = gaussCoefL_[hID%params.widthOfBand_ + 2*params.widthOfBand_];
                    pgdLBandSum[bandID] +=  coefInGaussion * pgdLRowSum;
                    ngdLBandSum[bandID] +=  coefInGaussion * ngdLRowSum;
                    pgdL2BandSum[bandID] +=  coefInGaussion * coefInGaussion * pgdL2RowSum;
                    ngdL2BandSum[bandID] +=  coefInGaussion * coefInGaussion * ngdL2RowSum;
                    pgdOBandSum[bandID] +=  coefInGaussion * pgdORowSum;
                    ngdOBandSum[bandID] +=  coefInGaussion * ngdORowSum;
                    pgdO2BandSum[bandID] +=  coefInGaussion * coefInGaussion * pgdO2RowSum;
                    ngdO2BandSum[bandID] +=  coefInGaussion * coefInGaussion * ngdO2RowSum;
                }
                bandID = bandID+2;
                if(bandID<params.numOfBand_){/*the band below the current band */
                    coefInGaussion = gaussCoefL_[hID%params.widthOfBand_];
                    pgdLBandSum[bandID] +=  coefInGaussion * pgdLRowSum;
                    ngdLBandSum[bandID] +=  coefInGaussion * ngdLRowSum;
                    pgdL2BandSum[bandID] +=  coefInGaussion * coefInGaussion * pgdL2RowSum;
                    ngdL2BandSum[bandID] +=  coefInGaussion * coefInGaussion * ngdL2RowSum;
                    pgdOBandSum[bandID] +=  coefInGaussion * pgdORowSum;
                    ngdOBandSum[bandID] +=  coefInGaussion * ngdORowSum;
                    pgdO2BandSum[bandID] +=  coefInGaussion * coefInGaussion * pgdO2RowSum;
                    ngdO2BandSum[bandID] +=  coefInGaussion * coefInGaussion * ngdO2RowSum;
                }
            }
            /* gDLMat.Save("gDLMat.txt");
            return 0; */

            /* construct line descriptor */
            pSingleLine->descriptor.resize(descriptorSize);
            desVec = pSingleLine->descriptor.data();

            short desID;

            /*Note that the first and last bands only have (lengthOfLSP * widthOfBand_ * 2.0) pixels
             * which are counted. */
            float invN2 = 1.0/(params.widthOfBand_ * 2.0);
            float invN3 = 1.0/(params.widthOfBand_ * 3.0);
            float invN, temp;
            for(bandID = 0; bandID<params.numOfBand_; bandID++){
                if(bandID==0||bandID==params.numOfBand_-1){	invN = invN2;
                }else{ invN = invN3;}
                desID = bandID * 8;
                temp = pgdLBandSum[bandID] * invN;
                desVec[desID]   = temp;/* mean value of pgdL; */
                desVec[desID+4] = sqrt(pgdL2BandSum[bandID] * invN - temp*temp);//std value of pgdL;
                temp = ngdLBandSum[bandID] * invN;
                desVec[desID+1] = temp;//mean value of ngdL;
                desVec[desID+5] = sqrt(ngdL2BandSum[bandID] * invN - temp*temp);//std value of ngdL;

                temp = pgdOBandSum[bandID] * invN;
                desVec[desID+2] = temp;//mean value of pgdO;
                desVec[desID+6] = sqrt(pgdO2BandSum[bandID] * invN - temp*temp);//std value of pgdO;
                temp = ngdOBandSum[bandID] * invN;
                desVec[desID+3] = temp;//mean value of ngdO;
                desVec[desID+7] = sqrt(ngdO2BandSum[bandID] * invN - temp*temp);//std value of ngdO;
            }

            // normalize;
            float tempM, tempS;
            tempM = 0;
            tempS = 0;
            desVec = pSingleLine->descriptor.data();

            int base = 0;
            for(short i=0; i<params.numOfBand_*8; ++base, i=base*8){
                tempM += *(desVec+i) * *(desVec+i);//desVec[8*i+0] * desVec[8*i+0];
                tempM += *(desVec+i+1) * *(desVec+i+1);//desVec[8*i+1] * desVec[8*i+1];
                tempM += *(desVec+i+2) * *(desVec+i+2);//desVec[8*i+2] * desVec[8*i+2];
                tempM += *(desVec+i+3) * *(desVec+i+3);//desVec[8*i+3] * desVec[8*i+3];
                tempS += *(desVec+i+4) * *(desVec+i+4);//desVec[8*i+4] * desVec[8*i+4];
                tempS += *(desVec+i+5) * *(desVec+i+5);//desVec[8*i+5] * desVec[8*i+5];
                tempS += *(desVec+i+6) * *(desVec+i+6);//desVec[8*i+6] * desVec[8*i+6];
                tempS += *(desVec+i+7) * *(desVec+i+7);//desVec[8*i+7] * desVec[8*i+7];
            }

            tempM = 1/sqrt(tempM);
            tempS = 1/sqrt(tempS);
            desVec = pSingleLine->descriptor.data();
            base = 0;
            for(short i=0; i<params.numOfBand_*8; ++base, i=base*8){
                *(desVec+i) = *(desVec+i) * tempM;//desVec[8*i] =  desVec[8*i] * tempM;
                *(desVec+1+i) = *(desVec+1+i) * tempM;//desVec[8*i+1] =  desVec[8*i+1] * tempM;
                *(desVec+2+i) = *(desVec+2+i) * tempM;//desVec[8*i+2] =  desVec[8*i+2] * tempM;
                *(desVec+3+i) = *(desVec+3+i) * tempM;//desVec[8*i+3] =  desVec[8*i+3] * tempM;
                *(desVec+4+i) = *(desVec+4+i) * tempS;//desVec[8*i+4] =  desVec[8*i+4] * tempS;
                *(desVec+5+i) = *(desVec+5+i) * tempS;//desVec[8*i+5] =  desVec[8*i+5] * tempS;
                *(desVec+6+i) = *(desVec+6+i) * tempS;//desVec[8*i+6] =  desVec[8*i+6] * tempS;
                *(desVec+7+i) = *(desVec+7+i) * tempS;//desVec[8*i+7] =  desVec[8*i+7] * tempS;
            }

            /* In order to reduce the influence of non-linear illumination,
             * a threshold is used to limit the value of element in the unit feature
             * vector no larger than this threshold. In Z.Wang's work, a value of 0.4 is found
             * empirically to be a proper threshold.*/
            desVec = pSingleLine->descriptor.data();
            for(short i=0; i<descriptorSize; i++ ){
                if(desVec[i]>0.4){
                    desVec[i]=0.4;
                }
            }

            //re-normalize desVec;
            temp = 0;
            for(short i=0; i<descriptorSize; i++){
                temp += desVec[i] * desVec[i];
            }

            temp = 1/sqrt(temp);
            for(short i=0; i<descriptorSize; i++){
                desVec[i] =  desVec[i] * temp;
            }
        }/* end for(short lineIDInSameLine = 0; lineIDInSameLine<sameLineSize;
            lineIDInSameLine++) */

    }/* end for(short lineIDInScaleVec = 0;
        lineIDInScaleVec<numOfFinalLine; lineIDInScaleVec++) */

    delete [] dL;
    delete [] dO;
    delete [] pgdLBandSum;
    delete [] ngdLBandSum;
    delete [] pgdL2BandSum;
    delete [] ngdL2BandSum;
    delete [] pgdOBandSum;
    delete [] ngdOBandSum;
    delete [] pgdO2BandSum;
    delete [] ngdO2BandSum;

    return 1;
}


