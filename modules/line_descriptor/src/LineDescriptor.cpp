/*IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

 By downloading, copying, installing or using the software you agree to this license.
 If you do not agree to this license, do not download, install,
 copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library

Copyright (C) 2011-2012, Lilian Zhang, all rights reserved.
Copyright (C) 2013, Manuele Tamburrano, Stefano Fabri, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * The name of the copyright holders may not be used to endorse or promote products
    derived from this software without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall the Intel Corporation or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

//#include "/include/opencv2/line_descriptor/LineDescriptor.hpp"
//#include "opencv2/LineDescriptor.hpp"

#include "precomp.hpp"

#define SalienceScale 0.9//0.9

//test pairs (0,7),(0,8),(1,7),(1,8) are excluded to get a descriptor size of 256 bit
static const int combinations [32][2] = {{0,1},{0,2},{0,3},{0,4},{0,5},{0,6},/*{0,7},{0,8},*/{1,2},{1,3},{1,4},{1,5},{1,6},/*{1,7},{1,8},*/{2,3},{2,4},{2,5},{2,6},{2,7},{2,8},{3,4},{3,5},{3,6},{3,7},{3,8},{4,5},{4,6},{4,7},{4,8},{5,6},{5,7},{5,8},{6,7},{6,8},{7,8}};

static inline int get2Pow(int i) {
    switch (i) {
    case 0:
        return 1;
    case 1:
        return 2;
    case 2:
        return 4;
    case 3:
        return 8;
    case 4:
        return 16;
    case 5:
        return 32;
    case 6:
        return 64;
    case 7:
        return 128;
    default:
        std::cout<<"Invalid pow"<<std::endl;
        CV_Assert(false);
        return -1; //avoid warning
    }
}


//#define DEBUGLinesInOctaveImages

using namespace std;

LineDescriptor::LineDescriptor(int n_octave)
{
    /* generate a random number by using time as a seed */
	srand ( time(NULL) );

    /* set gaussian kernel's size */
    ksize_  = 5;

    /* set number of octaves (default = 5) */
	numOfOctave_ = n_octave;//5

    /* create as many LineVecs as the number of octaves */
    edLineVec_.resize(numOfOctave_);
    for(unsigned int i=0; i<numOfOctave_; i++){
        edLineVec_[i] = new EDLineDetector;
    }

    /* set number of bands */
    numOfBand_   = 9; // suggested by authors

    /* set bands' width */
    widthOfBand_ = 7; //widthOfBand_%3 must equal to 0; 7 is a good value.

    /* prepare a vector to host local weights F_l*/
	gaussCoefL_.resize(widthOfBand_*3);

    /* compute center of central band (every computation involves 2-3 bands) */
	double u = (widthOfBand_*3-1)/2;

    /* compute exponential part of F_l */
    double sigma = (widthOfBand_*2+1)/2;// (widthOfBand_*2+1)/2;
	double invsigma2 = -1/(2*sigma*sigma);

    /* compute all local weights */
	double dis;
	for(int i=0; i<widthOfBand_*3; i++){
		dis = i-u;
		gaussCoefL_[i] = exp(dis*dis*invsigma2);
	}

    /* prepare a vector for global weights F_g*/
	gaussCoefG_.resize(numOfBand_*widthOfBand_);

    /* compute center of LSR */
	u = (numOfBand_*widthOfBand_-1)/2;

    /* compute exponential part of F_g */
	sigma = u;
	invsigma2 = -1/(2*sigma*sigma);
	for(int i=0; i<numOfBand_*widthOfBand_; i++){
		dis = i-u;
		gaussCoefG_[i] = exp(dis*dis*invsigma2);
	}

    /* THRESHOLDS
    2 is used to show recall ratio;
    0.2 is used to show scale space results;
    0.35 is used when verify geometric constraints */
    LowestThreshold = 0.3;
	NNDRThreshold   = 0.6;
}

LineDescriptor::LineDescriptor(unsigned int numOfBand, unsigned int widthOfBand, int n_octave){

    /* generate a random number by using time as a seed */
    srand ( time(NULL) );

    /* set gaussian kernel's size */
    ksize_  = 5;

    /* set number of octaves (default = 5) */
    numOfOctave_ = n_octave;

     /* create as many LineVecs as the number of octaves */
    edLineVec_.resize(numOfOctave_);
    for(unsigned int i=0; i<numOfOctave_; i++){
        edLineVec_[i] = new EDLineDetector;
    }

    /* set number of bands */
	numOfBand_   = numOfBand;

    /* set bands' width */
    widthOfBand_ = widthOfBand;

    /* prepare a vector to host local weights F_l*/
	gaussCoefL_.resize(widthOfBand_*3);

    /* compute center of central band (every computation involves 2-3 bands) */
	double u = (widthOfBand_*3-1)/2;

    /* compute exponential part of F_l */
	double sigma = (widthOfBand_*2+1)/2;// (widthOfBand_*2+1)/2;
	double invsigma2 = -1/(2*sigma*sigma);

    /* compute all local weights */
	double dis;
	for(int i=0; i<widthOfBand_*3; i++){
		dis = i-u;
		gaussCoefL_[i] = exp(dis*dis*invsigma2);
	}

     /* prepare a vector for global weights F_g */
	gaussCoefG_.resize(numOfBand_*widthOfBand_);

    /* compute center of LSR */
	u = (numOfBand_*widthOfBand_-1)/2;

    /* compute exponential part of F_g */
	sigma = u;
	invsigma2 = -1/(2*sigma*sigma);
	for(int i=0; i<numOfBand_*widthOfBand_; i++){
		dis = i-u;
		gaussCoefG_[i] = exp(dis*dis*invsigma2);
	}

    /* THRESHOLDS
    2 is used to show recall ratio;
    0.2 is used to show scale space results;
    0.35 is used when verify geometric constraints */
	LowestThreshold = 0.35;//0.35;
	NNDRThreshold   = 0.2;//0.6
}

/* destructor */
LineDescriptor::~LineDescriptor(){
//	for(unsigned int i=0; i<numOfOctave_; i++){
//		if(edLineVec_[i] !=NULL){
//			delete edLineVec_[i];
//		}
//	}
}

/*Line detection method: element in keyLines[i] includes a set of lines which is the same line
 * detected in different octave images.
 */
int LineDescriptor::OctaveKeyLines(cv::Mat & image, ScaleLines &keyLines)
{

    /* final number of extracted lines */
    unsigned int numOfFinalLine = 0;
	
    /* sigma values and reduction factor used in Gaussian pyramids */
    float preSigma2 = 0; //orignal image is not blurred, has zero sigma;
    float curSigma2 = 1.0; //[sqrt(2)]^0=1;
    float factor = sqrt(2); //the down sample factor between connective two octave images
	
    /* loop over number of octaves */
    for(unsigned int octaveCount = 0; octaveCount<numOfOctave_; octaveCount++){
	    
        /* matrix storing results from blurring processes */
        cv::Mat blur;

        /* Compute sigma value for Gaussian filter for each level by adding incremental blur from previous level.
         * curSigma = [sqrt(2)]^octaveCount;
         * increaseSigma^2 = curSigma^2 - preSigma^2 */
        float increaseSigma = sqrt(curSigma2-preSigma2);

        /* apply Gaussian blur */
        cv::GaussianBlur(image, blur, cv::Size(ksize_,ksize_), increaseSigma);
		
        /* for current octave, extract lines */
        if((edLineVec_[octaveCount]->EDline(blur,true))!= true){
            return -1;
        }

        /* update number of total extracted lines */
        numOfFinalLine += edLineVec_[octaveCount]->lines_.numOfLines;

        /* resize image for next level of pyramid */
        cv::resize(blur, image, cv::Size(), (1.f/factor), (1.f/factor));

        /* update sigma values */
        preSigma2 = curSigma2;
        curSigma2 = curSigma2*2;

    } /* end of loop over number of octaves */


    /*lines which correspond to the same line in the octave images will be stored
    in the same element of ScaleLines.*/

    /* prepare a vector to store octave information associated to extracted lines */
    std::vector<OctaveLine> octaveLines(numOfFinalLine);

    /* set lines' counter to 0 for reuse */
    numOfFinalLine = 0;

    /* counter to give a unique ID to lines in LineVecs */
    unsigned int lineIDInScaleLineVec = 0;

    /* floats to compute lines' lengths */
    float dx, dy;

    /* loop over lines extracted from scale 0 (original image) */
    for(unsigned int lineCurId=0;lineCurId<edLineVec_[0]->lines_.numOfLines;lineCurId++){
        /* FOR CURRENT LINE: */

        /* set octave from which it was extracted */
        octaveLines[numOfFinalLine].octaveCount = 0;
        /* set ID within its octave */
        octaveLines[numOfFinalLine].lineIDInOctave = lineCurId;
        /* set a unique ID among all lines extracted in all octaves */
        octaveLines[numOfFinalLine].lineIDInScaleLineVec = lineIDInScaleLineVec;

        /* compute absolute value of difference between X coordinates of line's extreme points */
        dx = fabs(edLineVec_[0]->lineEndpoints_[lineCurId][0]-edLineVec_[0]->lineEndpoints_[lineCurId][2]);
        /* compute absolute value of difference between Y coordinates of line's extreme points */
        dy = fabs(edLineVec_[0]->lineEndpoints_[lineCurId][1]-edLineVec_[0]->lineEndpoints_[lineCurId][3]);
        /* compute line's length */
        octaveLines[numOfFinalLine].lineLength = sqrt(dx*dx+dy*dy);

        /* update counters */
        numOfFinalLine++;
        lineIDInScaleLineVec++;
    }

    /* create and fill an array to store scale factors */
    float *scale = new float[numOfOctave_];
    scale[0] = 1;
    for(unsigned int octaveCount = 1; octaveCount<numOfOctave_; octaveCount++ ){
        scale[octaveCount] = factor * scale[octaveCount-1];
    }

    /* some variables' declarations */
    float rho1, rho2, tempValue;
    float direction, near, length;
    unsigned int octaveID, lineIDInOctave;

    /*more than one octave image, organize lines in scale space.
     *lines corresponding to the same line in octave images should have the same index in the ScaleLineVec */
    if(numOfOctave_>1){
        /* some other variables' declarations */
        float twoPI = 2*M_PI;
        unsigned int closeLineID;
        float endPointDis,minEndPointDis,minLocalDis,maxLocalDis;
        float lp0,lp1, lp2, lp3, np0,np1, np2, np3;

        /* loop over list of octaves */
        for(unsigned int octaveCount = 1; octaveCount<numOfOctave_; octaveCount++){
            /*for each line in current octave image, find their corresponding lines in the octaveLines,
             *give them the same value of lineIDInScaleLineVec*/

            /* loop over list of lines extracted from current octave */
            for(unsigned int lineCurId=0;lineCurId<edLineVec_[octaveCount]->lines_.numOfLines;lineCurId++){
                /* get (scaled) known term from equation of current line */
                rho1 = scale[octaveCount] *  fabs(edLineVec_[octaveCount]->lineEquations_[lineCurId][2]);

                /*nearThreshold depends on the distance of the image coordinate origin to current line.
                 *so nearThreshold = rho1 * nearThresholdRatio, where nearThresholdRatio = 1-cos(10*pi/180) = 0.0152*/
                tempValue = rho1 * 0.0152;
                float nearThreshold = (tempValue>6)?(tempValue):6;
                nearThreshold = (nearThreshold<12)?nearThreshold:12;

                /* compute scaled lenght of current line */
                dx = fabs(edLineVec_[octaveCount]->lineEndpoints_[lineCurId][0]-edLineVec_[octaveCount]->lineEndpoints_[lineCurId][2]);//x1-x2
                dy = fabs(edLineVec_[octaveCount]->lineEndpoints_[lineCurId][1]-edLineVec_[octaveCount]->lineEndpoints_[lineCurId][3]);//y1-y2
                length = scale[octaveCount] * sqrt(dx*dx+dy*dy);

                minEndPointDis = 12;
                /* loop over the octave representations of all lines */
                for(unsigned int lineNextId=0; lineNextId<numOfFinalLine;lineNextId++){
                    /* if a line from same octave is encountered,
                    a comparison with it shouldn't be considered */
                    octaveID = octaveLines[lineNextId].octaveCount;
                    if(octaveID==octaveCount){//lines in the same layer of octave image should not be compared.
                        break;
                    }

                    /* take ID in octave of line to be compared */
                    lineIDInOctave = octaveLines[lineNextId].lineIDInOctave;

                    /*first check whether current line and next line are parallel.
                     *If line1:a1*x+b1*y+c1=0 and line2:a2*x+b2*y+c2=0 are parallel, then
                     *-a1/b1=-a2/b2, i.e., a1b2=b1a2.
                     *we define parallel=fabs(a1b2-b1a2)
                     *note that, in EDLine class, we have normalized the line equations
                     *to make a1^2+ b1^2 = a2^2+ b2^2 = 1*/
                    direction = fabs(edLineVec_[octaveCount]->lineDirection_[lineCurId] -
                            edLineVec_[octaveID]->lineDirection_[lineIDInOctave]);

                    /* the angle between two lines are larger than 10degrees
                    (i.e. 10*pi/180=0.1745), they are not close to parallel */
                    if(direction>0.1745 && (twoPI - direction>0.1745)){
                        continue;
                    }
                    /*now check whether current line and next line are near to each other.
                     *If line1:a1*x+b1*y+c1=0 and line2:a2*x+b2*y+c2=0 are near in image, then
                     *rho1 = |a1*0+b1*0+c1|/sqrt(a1^2+b1^2) and rho2 = |a2*0+b2*0+c2|/sqrt(a2^2+b2^2) should close.
                     *In our case, rho1 = |c1| and rho2 = |c2|, because sqrt(a1^2+b1^2) = sqrt(a2^2+b2^2) = 1;
                     *note that, lines are in different octave images, so we define near =  fabs(scale*rho1 - rho2) or
                     *where scale is the scale factor between to octave images*/

                    /* get known term from equation to be compared */
                    rho2 = scale[octaveID] * fabs(edLineVec_[octaveID]->lineEquations_[lineIDInOctave][2]);
                    /* compute difference between known ters */
                    near = fabs(rho1 - rho2);

                    /* two lines are not close in the image */
                    if(near>nearThreshold){
                        continue;
                    }

                    /*now check the end points distance between two lines, the scale of  distance is in the original image size.
                     * find the minimal and maximal end points distance*/

                    /* get the extreme points of the two lines */
                    lp0 = scale[octaveCount] *edLineVec_[octaveCount]->lineEndpoints_[lineCurId][0];
                    lp1 = scale[octaveCount] *edLineVec_[octaveCount]->lineEndpoints_[lineCurId][1];
                    lp2 = scale[octaveCount] *edLineVec_[octaveCount]->lineEndpoints_[lineCurId][2];
                    lp3 = scale[octaveCount] *edLineVec_[octaveCount]->lineEndpoints_[lineCurId][3];
                    np0 = scale[octaveID] * edLineVec_[octaveID]->lineEndpoints_[lineIDInOctave][0];
                    np1 = scale[octaveID] * edLineVec_[octaveID]->lineEndpoints_[lineIDInOctave][1];
                    np2 = scale[octaveID] * edLineVec_[octaveID]->lineEndpoints_[lineIDInOctave][2];
                    np3 = scale[octaveID] * edLineVec_[octaveID]->lineEndpoints_[lineIDInOctave][3];

                    /* get the distance between the two leftmost extremes of lines
                    L1(0,1)<->L2(0,1) */
                    dx = lp0 - np0;
                    dy = lp1 - np1;
                    endPointDis = sqrt(dx*dx + dy*dy);

                    /* set momentaneously min and max distance between lines to
                    the one between left extremes */
                    minLocalDis = endPointDis;
                    maxLocalDis = endPointDis;

                    /* compute distance between right extremes
                    L1(2,3)<->L2(2,3) */
                    dx = lp2 - np2;
                    dy = lp3 - np3;
                    endPointDis = sqrt(dx*dx + dy*dy);

                    /* update (if necessary) min and max distance between lines */
                    minLocalDis = (endPointDis<minLocalDis)?endPointDis:minLocalDis;
                    maxLocalDis = (endPointDis>maxLocalDis)?endPointDis:maxLocalDis;


                    /* compute distance between left extreme of current line and
                    right extreme of line to be compared
                    L1(0,1)<->L2(2,3) */
                    dx = lp0 - np2;
                    dy = lp1 - np3;
                    endPointDis = sqrt(dx*dx + dy*dy);

                    /* update (if necessary) min and max distance between lines */
                    minLocalDis = (endPointDis<minLocalDis)?endPointDis:minLocalDis;
                    maxLocalDis = (endPointDis>maxLocalDis)?endPointDis:maxLocalDis;

                    /* compute distance between right extreme of current line and
                    left extreme of line to be compared
                    L1(2,3)<->L2(0,1) */
                    dx = lp2 - np0;
                    dy = lp3 - np1;
                    endPointDis = sqrt(dx*dx + dy*dy);

                    /* update (if necessary) min and max distance between lines */
                    minLocalDis = (endPointDis<minLocalDis)?endPointDis:minLocalDis;
                    maxLocalDis = (endPointDis>maxLocalDis)?endPointDis:maxLocalDis;

                    /* check whether conditions for considering line to be compared
                    worth to be inserted in the same LineVec are satisfied */
                    if((maxLocalDis<0.8*(length+octaveLines[lineNextId].lineLength))
                        &&(minLocalDis<minEndPointDis)){//keep the closest line
                        minEndPointDis = minLocalDis;
                        closeLineID = lineNextId;
                    }
                }


                /* add current line into octaveLines */
                if(minEndPointDis<12){
                    octaveLines[numOfFinalLine].lineIDInScaleLineVec = octaveLines[closeLineID].lineIDInScaleLineVec;
                }else{
                    octaveLines[numOfFinalLine].lineIDInScaleLineVec = lineIDInScaleLineVec;
                    lineIDInScaleLineVec++;
                }
                octaveLines[numOfFinalLine].octaveCount    = octaveCount;
                octaveLines[numOfFinalLine].lineIDInOctave = lineCurId;
                octaveLines[numOfFinalLine].lineLength     = length;
                numOfFinalLine++;
            }
        }//end for(unsigned int octaveCount = 1; octaveCount<numOfOctave_; octaveCount++)
    }//end if(numOfOctave_>1)

    ////////////////////////////////////
    //Reorganize the detected lines into keyLines
    keyLines.clear();
    keyLines.resize(lineIDInScaleLineVec);
  unsigned int tempID;
    float s1,e1,s2,e2;
    bool shouldChange;
    OctaveSingleLine singleLine;
    for(unsigned int  lineID = 0;lineID < numOfFinalLine; lineID++){
        lineIDInOctave = octaveLines[lineID].lineIDInOctave;
        octaveID       = octaveLines[lineID].octaveCount;
        direction      = edLineVec_[octaveID]->lineDirection_[lineIDInOctave];
        singleLine.octaveCount = octaveID;
        singleLine.direction = direction;
        singleLine.lineLength = octaveLines[lineID].lineLength;
        singleLine.salience  = edLineVec_[octaveID]->lineSalience_[lineIDInOctave];
        singleLine.numOfPixels = edLineVec_[octaveID]->lines_.sId[lineIDInOctave+1]-
                                 edLineVec_[octaveID]->lines_.sId[lineIDInOctave];
        //decide the start point and end point
        shouldChange = false;
        s1 = edLineVec_[octaveID]->lineEndpoints_[lineIDInOctave][0];//sx
        s2 = edLineVec_[octaveID]->lineEndpoints_[lineIDInOctave][1];//sy
        e1 = edLineVec_[octaveID]->lineEndpoints_[lineIDInOctave][2];//ex
        e2 = edLineVec_[octaveID]->lineEndpoints_[lineIDInOctave][3];//ey
        dx = e1 - s1;//ex-sx
        dy = e2 - s2;//ey-sy
        if(direction>=-0.75*M_PI&&direction<-0.25*M_PI){
            if(dy>0){shouldChange = true;}
        }
        if(direction>=-0.25*M_PI&&direction<0.25*M_PI){
            if(dx<0){shouldChange = true;}
        }
        if(direction>=0.25*M_PI&&direction<0.75*M_PI){
            if(dy<0){shouldChange = true;}
        }
        if((direction>=0.75*M_PI&&direction<M_PI)||(direction>=-M_PI&&direction<-0.75*M_PI)){
            if(dx>0){shouldChange = true;}
        }
        tempValue = scale[octaveID];
        if(shouldChange){
            singleLine.sPointInOctaveX = e1;
            singleLine.sPointInOctaveY = e2;
            singleLine.ePointInOctaveX = s1;
            singleLine.ePointInOctaveY = s2;
            singleLine.startPointX = tempValue * e1;
            singleLine.startPointY = tempValue * e2;
            singleLine.endPointX   = tempValue * s1;
            singleLine.endPointY   = tempValue * s2;
        }else{
            singleLine.sPointInOctaveX = s1;
            singleLine.sPointInOctaveY = s2;
            singleLine.ePointInOctaveX = e1;
            singleLine.ePointInOctaveY = e2;
            singleLine.startPointX = tempValue * s1;
            singleLine.startPointY = tempValue * s2;
            singleLine.endPointX   = tempValue * e1;
            singleLine.endPointY   = tempValue * e2;
        }
        tempID = octaveLines[lineID].lineIDInScaleLineVec;
        keyLines[tempID].push_back(singleLine);
    }

    ////////////////////////////////////

  delete [] scale;
  return 1;
}

/*The definitions of line descriptor,mean values of {g_dL>0},{g_dL<0},{g_dO>0},{g_dO<0} of each row in band
 *and std values of sum{g_dL>0},sum{g_dL<0},sum{g_dO>0},sum{g_dO<0} of each row in band.
 * With overlap region. */
int LineDescriptor::ComputeLBD_(ScaleLines &keyLines)
{
	//the default length of the band is the line length.
	short numOfFinalLine = keyLines.size();
	float *dL = new float[2];//line direction cos(dir), sin(dir)
	float *dO = new float[2];//the clockwise orthogonal vector of line direction.
	short heightOfLSP = widthOfBand_*numOfBand_;//the height of line support region;
	short descriptorSize = numOfBand_ * 8;//each band, we compute the m( pgdL, ngdL,  pgdO, ngdO) and std( pgdL, ngdL,  pgdO, ngdO);
	float pgdLRowSum;//the summation of {g_dL |g_dL>0 } for each row of the region;
	float ngdLRowSum;//the summation of {g_dL |g_dL<0 } for each row of the region;
	float pgdL2RowSum;//the summation of {g_dL^2 |g_dL>0 } for each row of the region;
	float ngdL2RowSum;//the summation of {g_dL^2 |g_dL<0 } for each row of the region;
	float pgdORowSum;//the summation of {g_dO |g_dO>0 } for each row of the region;
	float ngdORowSum;//the summation of {g_dO |g_dO<0 } for each row of the region;
	float pgdO2RowSum;//the summation of {g_dO^2 |g_dO>0 } for each row of the region;
	float ngdO2RowSum;//the summation of {g_dO^2 |g_dO<0 } for each row of the region;

	float *pgdLBandSum  = new float[numOfBand_];//the summation of {g_dL |g_dL>0 } for each band of the region;
	float *ngdLBandSum  = new float[numOfBand_];//the summation of {g_dL |g_dL<0 } for each band of the region;
	float *pgdL2BandSum = new float[numOfBand_];//the summation of {g_dL^2 |g_dL>0 } for each band of the region;
	float *ngdL2BandSum = new float[numOfBand_];//the summation of {g_dL^2 |g_dL<0 } for each band of the region;
	float *pgdOBandSum  = new float[numOfBand_];//the summation of {g_dO |g_dO>0 } for each band of the region;
	float *ngdOBandSum  = new float[numOfBand_];//the summation of {g_dO |g_dO<0 } for each band of the region;
	float *pgdO2BandSum = new float[numOfBand_];//the summation of {g_dO^2 |g_dO>0 } for each band of the region;
	float *ngdO2BandSum = new float[numOfBand_];//the summation of {g_dO^2 |g_dO<0 } for each band of the region;

	short numOfBitsBand = numOfBand_*sizeof(float);
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

            /* retrieve associated dxImg and dyImg */
			pdxImg = edLineVec_[octaveCount]->dxImg_.ptr<short>();
			pdyImg = edLineVec_[octaveCount]->dyImg_.ptr<short>();

            /* get image size to work on from real one */
			realWidth = edLineVec_[octaveCount]->imageWidth;
			imageWidth  = realWidth -1;
			imageHeight = edLineVec_[octaveCount]->imageHeight-1;


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
				bandID = hID/widthOfBand_;
				coefInGaussion = gaussCoefL_[hID%widthOfBand_+widthOfBand_];
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
					coefInGaussion = gaussCoefL_[hID%widthOfBand_ + 2*widthOfBand_];
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
                if(bandID<numOfBand_){/*the band below the current band */
					coefInGaussion = gaussCoefL_[hID%widthOfBand_];
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
			float invN2 = 1.0/(widthOfBand_ * 2.0);
			float invN3 = 1.0/(widthOfBand_ * 3.0);
			float invN, temp;
			for(bandID = 0; bandID<numOfBand_; bandID++){
				if(bandID==0||bandID==numOfBand_-1){	invN = invN2;
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
			for(short i=0; i<numOfBand_*8; ++base, i=base*8){
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
			for(short i=0; i<numOfBand_*8; ++base, i=base*8){
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
}


unsigned char binaryTest(float* f1, float*f2)
{
    uchar result=0;
    for(int i = 0; i<8; i++)
    {
//        std::cout<<"f1[: "<<i<<"]: "<<f1[i];
//        std::cout<<" -- f2[: "<<i<<"]: "<<f2[i]<<std::endl;
        if(f1[i]>f2[i])
        {
//            std::cout<< " ------  1  ------- "<<std::endl;
            result+=get2Pow(i);
        }
        else
        {
//            std::cout<< " ------  0  ------- "<<std::endl;
        }
    }
    
    return result;
        
}

//unsigned char binaryTest2(float* f1, float*f2)
//{
//    uchar result=0;
//    for(int i = 0; i<8; i++)
//    {
////        std::cout<<"f1[: "<<i<<"]: "<<f1[i];
////        std::cout<<" -- f2[: "<<i<<"]: "<<f2[i]<<std::endl;
//        if(f1[i]>f2[i])
//        {
////            std::cout<< " ------  1  ------- "<<std::endl;
//            result+=get2Pow(i);
//        }
//        else
//        {
////            std::cout<< " ------  0  ------- "<<std::endl;
//        }
//    }
//    
//    return result;
//        
//}

unsigned char binaryIndexTest(int f1, int f2, std::vector<float>& desc)
{
    uchar result=0;
    for(int i = 0; i<8; i++)
    {
//        std::cout<<"f1[: "<<i<<"]: "<<f1[i];
//        std::cout<<" -- f2[: "<<i<<"]: "<<f2[i]<<std::endl;
        if(desc[i+f1]>desc[i+f2])
        {
//            std::cout<< " ------  1  ------- "<<std::endl;
            result+=get2Pow(i);
        }
        else
        {
//            std::cout<< " ------  0  ------- "<<std::endl;
        }
    }
    
    return result;
        
}

/* create a vector of matrices: every matrix represents i-th octave and it has a (binary) descriptor
  of one of lines extracted from i-th octave on each row */
void LineDescriptor::GetLineBinaryDescriptor(std::vector<cv::Mat> & oct_binaryDescMat, ScaleLines & keyLines)
{

    /* std::cout<<"numOfOctave: "<<numOfOctave_<<std::endl; */
    
    /* create an int vector, whose i-th position stores how many lines where extracted from
    i-th octave.
    At the beginning, initialize vector as a sequence of counters set to 0 */
    std::vector<int> rows_size;
    for(int i = 0; i<numOfOctave_; i++)
        rows_size.push_back(0);

    /* modify counters in rows_size in order to reflect
    the number of lines extracted from each octave */
    for(int offsetKeyLines = 0; offsetKeyLines<keyLines.size(); offsetKeyLines++)
    {
            for(int offsetOctave = 0; offsetOctave<keyLines[offsetKeyLines].size(); offsetOctave++)
            {
                rows_size[keyLines[offsetKeyLines][offsetOctave].octaveCount]++;
            }
    }

    /* prepare a vector of pointers to the first rows of matrices
    that are going to be created in next loop */
    std::vector<uchar *> vec_binaryMat_p;
    
    /* loop on the number of the octaves */
    for(int i = 0; i<numOfOctave_; i++)
    {
        /* create a matrix having as many rows as the number of lines
        extracted from i-th octave and 32 columns (each row stores
        a 32 bit string) */
        cv::Mat mat_binary(rows_size[i], 32, CV_8UC1);
        
        /* add created matrix to list passed as an argument */
        oct_binaryDescMat.push_back(mat_binary);
        
        /* store a pointer to the first row of the matrix that has been
        just created */
        vec_binaryMat_p.push_back(oct_binaryDescMat.at(i).ptr());
        
    }
    
    /* loop over the number of LineVecs */
    for(int offsetKeyLines = 0; offsetKeyLines<keyLines.size(); offsetKeyLines++)
    {
        /* loop over the number of lines inside each LineVec */
        for(int offsetOctave = 0; offsetOctave<keyLines[offsetKeyLines].size(); offsetOctave++)
        {   
            /* binaryMat_p is a pointer to a pointer,
            because we must increment the content of a vector,
            in such a way that pointers are updated with the right row of matrix */
            uchar ** binaryMat_p = &vec_binaryMat_p.at(keyLines[offsetKeyLines][offsetOctave].octaveCount);

            /* get descriptor associated to i-th line (as a sequence of floats) */
            float * desVec = keyLines[offsetKeyLines][offsetOctave].descriptor.data();


            for(int comb = 0; comb < 32; comb++)
            {
                *(*binaryMat_p) =  binaryTest(&desVec[8*combinations[comb][0]], &desVec[8*combinations[comb][1]]);
                (*binaryMat_p)++; 
                
            }
            
        }
    }


    //writeMat(oct_binaryDescMat[0], "binaryMat_deb",0);
            
}

/* compute LBD descriptors */
int LineDescriptor::GetLineDescriptor(cv::Mat & image, ScaleLines & keyLines)
{

    /*check whether image depth is different from 0 */
    if(image.depth() != 0)
    {
        std::cout << "Warning, depth image!= 0" << std::endl;
        CV_Assert(false);
    }
    
    /* get clock's TIC */
    double t = (double)cv::getTickCount();

    /* compute LineVecs extraction and in case of failure, return an error code */
    if((OctaveKeyLines(image,keyLines))!=true){
        cout << "OctaveKeyLines failed" << endl;
        return -1;
    }

    /* get time lapse */
    t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
    std::cout << "Time of line extraction: "<< t << "s" <<std::endl;
    
//    for(int j = 0; j<keyLines.size(); j++)
//    {
//        for(int k = 0; k<keyLines[j].size(); k++)
//        {
//            OctaveSingleLine singleLine = keyLines[j][k];
//            std::cout<<"-----------["<<j<<"]["<<k<<"]--------------"<<std::endl;
//            std::cout<<"singleLine.octaveCount :"<<singleLine.octaveCount<<std::endl;
//            std::cout<<"singleLine.direction :"<<singleLine.direction<<std::endl;
//            std::cout<<"singleLine.lineLength :"<<singleLine.lineLength<<std::endl;
//            std::cout<<"singleLine.salience :"<<singleLine.salience<<std::endl;
//            std::cout<<"singleLine.numOfPixels :"<<singleLine.numOfPixels<<std::endl;
//            std::cout<<"singleLine.sPointInOctaveX :"<<singleLine.sPointInOctaveX<<std::endl;
//            std::cout<<"singleLine.sPointInOctaveY :"<<singleLine.sPointInOctaveY<<std::endl;
//            std::cout<<"singleLine.ePointInOctaveX :"<<singleLine.ePointInOctaveX<<std::endl;
//            std::cout<<"singleLine.ePointInOctaveY :"<<singleLine.ePointInOctaveY<<std::endl;
//            std::cout<<"singleLine.startPointX :"<<singleLine.startPointX<<std::endl;
//            std::cout<<"singleLine.startPointY :"<<singleLine.startPointY<<std::endl;
//            std::cout<<"singleLine.endPointX :"<<singleLine.endPointX<<std::endl;
//            std::cout<<"singleLine.endPointY :"<<singleLine.endPointY<<std::endl;
//            std::cout<<"--------------------------------"<<std::endl;
//         }
//    }
    
//    t = (double)cv::getTickCount();

    /* compute LBD descriptors */
    ComputeLBD_(keyLines);


//    t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
//    std::cout<<"time descriptor extraction: "<<t<<"s"<<std::endl;
//    
    
//    for(int j = 0; j<keyLines.size(); j++)
//    {
//        for(int k = 0; k<keyLines[j].size(); k++)
//        {
//            for(int i = 0; i<keyLines[j][k].descriptor.size(); i++)
//                std::cout<<"keylines["<<j<<"]["<<k<<"].descriptor["<<i<<"]: "<<keyLines[j][k].descriptor[i]<<std::endl;
//        }
//    }
    
//    srand((unsigned)time(0));
//   int lowest=100, highest=25555;
//   int range=(highest-lowest)+1;
//   int r;
//   r = lowest+int(rand()%range);
//   string folder_debug = "/home/manuele/src/imgretrieval/mat_debug/";
//   std::cout<<"LINDESCRIPTOR INIZIO"<<std::endl;
//   cv::Mat desc_float(keyLines.size(), keyLines[0][0].descriptor.size(), CV_32FC1);
//   for(int i = 0; i<keyLines.size(); i++)
//   {
//       for(int j = 0; j<keyLines[i][0].descriptor.size(); j++)
//       {
//           //std::cout<<keyLines[i][0].descriptor[j];
//           desc_float.at<float>(i, j) = (float)keyLines[i][0].descriptor[j];
//       }
//       std::cout<<std::endl;
//   }
////   std::cout<<desc_float<<std::endl;
//   std::cout<<"LINDESCRIPTOR FINE"<<std::endl<<std::endl;
//   writeMat(desc_float, folder_debug+"linedesc", r);
    
    return 1;
}

/* check whether two lines are parallel, using their directions */
bool areParallels(float direction1, float direction2)
{
	if(abs(abs(direction1) - abs(direction2)) <= 0.02)
		return true;

	if(abs(direction1) + abs(direction2) >= 3.12)
		return true;

	return false;
}


void LineDescriptor::findNearestParallelLines(ScaleLines & keyLines)
{

	std::cout<<"PARALLELLINES: size: "<<keyLines.size()<<std::endl;
	std::map<float, OctaveSingleLine> parallels;

    /* loop over LineVecs */
    for(int j = 0; j<keyLines.size(); j++)
    {
        /* loop over lines in current LineVec */
        for(int k = 0; k<keyLines[j].size(); k++)
        {
            /* get current line */
            OctaveSingleLine singleLine = keyLines[j][k];

            /* get an iterator to map of lines */
            std::map<float, OctaveSingleLine>::iterator it;

            /* scan map to searck for a line parallel to current one */
            bool foundParallel = false;
            for(it = parallels.begin(); it != parallels.end(); it++) {
                if(!areParallels(it->first, singleLine.direction))
                {
                	foundParallel = true;
                	break;
                }

            }

            /* if a parallel line has not been found, add current line
               to map, using its direction as a key */
            if(!foundParallel)
            	parallels[singleLine.direction] = singleLine;

        }
    }

    /* create a vector of LineVecs, each one containing a line that is
        not parallel to any other one inside a different LineVec */
    ScaleLines newKeyLines;
    std::map<float, OctaveSingleLine>::iterator it;
    for(it = parallels.begin(); it != parallels.end(); it++) {
		LinesVec   lineScaleLs;
		lineScaleLs.push_back(it->second);
		newKeyLines.push_back(lineScaleLs);
    }

    keyLines = newKeyLines;

}

int LineDescriptor::GetLineDescriptor(std::vector<cv::Mat> &scale_images, ScaleLines &keyLines)
{

    /*check whether image depth is different from 0
    if(image.depth() != 0)
    {
        std::cout << "Warning, depth image != 0" << std::endl;
        CV_Assert(false);
    }*/

    for(size_t scaleCounter = 0; scaleCounter<scale_images.size(); scaleCounter++)
    {
        /* get current scaled image */
        cv::Mat currentScaledImage = scale_images[scaleCounter];

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

    /* compute line descriptors */
    ComputeLBD_(keyLines);

    return 1;
}

/*Match lines by their descriptors.
 *The function will use opencv FlannBasedMatcher to mathc lines. */
int LineDescriptor::MatchLineByDescriptor(ScaleLines &keyLinesLeft, ScaleLines &keyLinesRight,
		std::vector<short> &matchLeft, std::vector<short> &matchRight,
		int criteria)
{

    /* check whether any input is void */
	int leftSize = keyLinesLeft.size();
	int rightSize = keyLinesRight.size();
	if(leftSize<1||rightSize<1){
		return -1;
	}

    /* void vectors */
	matchLeft.clear();
	matchRight.clear();


	int desDim = keyLinesLeft[0][0].descriptor.size();
	float *desL, *desR, *desMax, *desOld;

    if(criteria==NearestNeighbor)
    {
		float minDis,dis,temp;
		int corresId;

        /* loop over left list of LineVecs */
        for(int idL=0; idL<leftSize; idL++)
        {
            /* get size of current left LineVec */
			short sameLineSize = keyLinesLeft[idL].size();

            /* initialize a distance threshold */
			minDis = 100;

            /* loop over lines inside current left LineVec */
            for(short lineIDInSameLines = 0; lineIDInSameLines<sameLineSize; lineIDInSameLines++)
            {
                /* get current left line */
				desOld = keyLinesLeft[idL][lineIDInSameLines].descriptor.data();

                /* loop over right list of LineVecs */
                for(int idR=0; idR<rightSize; idR++)
                {
                    /* get size of current right LineVec */
					short sameLineSizeR = keyLinesRight[idR].size();

                    /* loop over lines inside current right LineVec */
                    for(short lineIDInSameLinesR = 0; lineIDInSameLinesR<sameLineSizeR; lineIDInSameLinesR++)
                    {
                        /* store temporay descriptor of left line */
						desL = desOld;

                        /* get descriptor of right line */
						desR = keyLinesRight[idR][lineIDInSameLinesR].descriptor.data();

						desMax = desR+desDim;
						dis = 0;

                        /* compute euclidean distance between the two descriptors */
                        while(desR<desMax)
                        {
							temp = *(desL++) - *(desR++);
							dis += temp*temp;
						}

						dis = sqrt(dis);

                        /* if euclidean distance is below current threshold,
                            keep track of current match */
                        if(dis<minDis)
                        {
							minDis = dis;
							corresId = idR;
						}
					}
				}//end for(int idR=0; idR<rightSize; idR++)
			}//end for(short lineIDInSameLines = 0; lineIDInSameLines<sameLineSize; lineIDInSameLines++)

            /* if minimum found distance is below fixed threshold, cnfirm match
                by storing corresponding matcjed lines */
            if(minDis<LowestThreshold)
            {
				matchLeft.push_back(idL);
				matchRight.push_back(corresId);
			}
		}// end for(int idL=0; idL<leftSize; idL++)
	}
}

