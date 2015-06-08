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
 *  *Redistributions of source code must retain the above copyright notice,
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

namespace cv {
namespace ximgproc {

class WeightedLeastSquaresFilterImpl : public WeightedLeastSquaresFilter
{
public:
    static Ptr<WeightedLeastSquaresFilterImpl> create(InputArray guide, double lambda, double sigma_color, int num_iter);
    void filter(InputArray src, OutputArray dst);
    ~WeightedLeastSquaresFilterImpl();
protected:
    int w,h;
    double sigmaColor,lambda;
    int num_iter;
    double *weights_LUT;
    double *Ahor,  *Bhor,  *Chor, 
            *Avert, *Bvert, *Cvert;
    double *interD,*interE,*cur_res;
    void init(InputArray guide,double _lambda,double _sigmaColor,int _num_iter);
    void buildCoefMatrices(Mat& guide);
    void horizontalPass(double* cur);
    void verticalPass(double* cur);
};

void WeightedLeastSquaresFilterImpl::init(InputArray guide,double _lambda,double _sigmaColor,int _num_iter)
{
    //currently support only 3 channel 8bit images as guides
    CV_Assert( !guide.empty() && _lambda >= 0 && _sigmaColor >= 0 && _num_iter >=1 );
    CV_Assert( guide.depth() == CV_8U && guide.channels() == 3 );
    sigmaColor = _sigmaColor;
    lambda = _lambda;
    num_iter = _num_iter;
    int num_levels = 3*256*256;
    weights_LUT = new double[num_levels];
    for(int i=0;i<num_levels;i++)
        weights_LUT[i] = exp(-sqrt((double)i)/sigmaColor);
    w = guide.cols();
    h = guide.rows();
    int sz = w*h;
    Ahor  = new double[sz];Bhor  = new double[sz];Chor  = new double[sz];
    Avert = new double[sz];Bvert = new double[sz];Cvert = new double[sz];
    interD = new double[sz];interE = new double[sz];cur_res = new double[sz];
    Mat guideMat = guide.getMat();
    buildCoefMatrices(guideMat);
}

Ptr<WeightedLeastSquaresFilterImpl> WeightedLeastSquaresFilterImpl::create(InputArray guide, double lambda, double sigma_color, int num_iter)
{
    WeightedLeastSquaresFilterImpl *wls = new WeightedLeastSquaresFilterImpl();
    wls->init(guide,lambda,sigma_color,num_iter);
    return Ptr<WeightedLeastSquaresFilterImpl>(wls);
}

WeightedLeastSquaresFilter::~WeightedLeastSquaresFilter(){}
WeightedLeastSquaresFilterImpl::~WeightedLeastSquaresFilterImpl()
{
    delete[] weights_LUT;
    delete[] Ahor;  delete[] Bhor;  delete[] Chor;
    delete[] Avert; delete[] Bvert; delete[] Cvert;
    delete[] interD;delete[] interE;delete[] cur_res;
}

void WeightedLeastSquaresFilterImpl::buildCoefMatrices(Mat& guide)
{
    double hor_weight;
    const unsigned char *row,*row_prev,*row_next;
    for(int i=0;i<h;i++)
    {
        //compute horizontal coefs:
        row = guide.ptr(i);
        Ahor[i*w] = 0;
        hor_weight = weights_LUT[ (row[0]-row[3])*(row[0]-row[3])+
                                  (row[1]-row[4])*(row[1]-row[4])+
                                  (row[2]-row[5])*(row[2]-row[5]) ];
        Chor[i*w] = -lambda*hor_weight;
        Bhor[i*w] = 1 - Ahor[i*w] - Chor[i*w];
        row+=3;
        for(int j=1;j<w-1;j++)
        {
            Ahor[i*w+j] = -lambda*hor_weight;
            hor_weight = weights_LUT[ (row[0]-row[3])*(row[0]-row[3])+
                                      (row[1]-row[4])*(row[1]-row[4])+
                                      (row[2]-row[5])*(row[2]-row[5]) ];
            Chor[i*w+j] = -lambda*hor_weight;
            Bhor[i*w+j] = 1 - Ahor[i*w+j] - Chor[i*w+j];
            row+=3;
        }
        Ahor[i*w+w-1] = -lambda*hor_weight;
        Chor[i*w+w-1] = 0;
        Bhor[i*w+w-1] = 1 - Ahor[i*w+w-1] - Chor[i*w+w-1];

        //compute vertical coefs:
        row = guide.ptr(i);
        if(i==0)
        {
            row_next = guide.ptr(i+1);
            for(int j=0;j<w;j++)
            {
                Avert[i*w+j] = 0;
                Cvert[i*w+j] = -lambda*weights_LUT[ (row[0]-row_next[0])*(row[0]-row_next[0])+
                                                    (row[1]-row_next[1])*(row[1]-row_next[1])+
                                                    (row[2]-row_next[2])*(row[2]-row_next[2]) ];
                Bvert[i*w+j] = 1 - Avert[i*w+j] - Cvert[i*w+j];
                row+=3;
                row_next+=3;
            }
        }
        else if(i==h-1)
        {
            row_prev = guide.ptr(i-1);
            for(int j=0;j<w;j++)
            {
                Avert[i*w+j] = -lambda*weights_LUT[ (row[0]-row_prev[0])*(row[0]-row_prev[0])+
                                                    (row[1]-row_prev[1])*(row[1]-row_prev[1])+
                                                    (row[2]-row_prev[2])*(row[2]-row_prev[2]) ];
                Cvert[i*w+j] = 0;
                Bvert[i*w+j] = 1 - Avert[i*w+j] - Cvert[i*w+j];
                row+=3;
                row_prev+=3;
            }
        }
        else
        {
            row_prev = guide.ptr(i-1);
            row_next = guide.ptr(i+1);
            for(int j=0;j<w;j++)
            {
                Avert[i*w+j] = -lambda*weights_LUT[ (row[0]-row_prev[0])*(row[0]-row_prev[0])+
                                                    (row[1]-row_prev[1])*(row[1]-row_prev[1])+
                                                    (row[2]-row_prev[2])*(row[2]-row_prev[2]) ];
                Cvert[i*w+j] = -lambda*weights_LUT[ (row[0]-row_next[0])*(row[0]-row_next[0])+
                    (row[1]-row_next[1])*(row[1]-row_next[1])+
                    (row[2]-row_next[2])*(row[2]-row_next[2]) ];
                Bvert[i*w+j] = 1 - Avert[i*w+j] - Cvert[i*w+j];
                row+=3;
                row_prev+=3;
                row_next+=3;
            }
        }
    }
}

void WeightedLeastSquaresFilterImpl::filter(InputArray src, OutputArray dst)
{
    //temporarily support only one-channel CV_16S src type (for disparity map filtering)
    CV_Assert(!src.empty() && (src.depth() == CV_16S) && src.channels()==1);
    if (src.rows() != h || src.cols() != w)
    {
        CV_Error(Error::StsBadSize, "Size of filtering image must be equal to size of guide image");
        return;
    }

    Mat srcMat = src.getMat(); 
    Mat& dstMat = dst.getMatRef();
    short* row;
    for(int i=0;i<h;i++)
    {
        row = (short*)srcMat.ptr(i);
        for(int j=0;j<w;j++)
        {
            cur_res[i*w+j] = (double)(*row);
            row++;
        }
    }

    for(int n=0;n<num_iter;n++)
    {
        horizontalPass(cur_res);
        verticalPass(cur_res);
    }

    for(int i=0;i<h;i++)
    {
        row = (short*)dstMat.ptr(i);
        for(int j=0;j<w;j++)
        {
            *row = saturate_cast<short>(cur_res[i*w+j]);
            row++;
        }
    }
}

void WeightedLeastSquaresFilterImpl::horizontalPass(double* cur)
{
    double denom;
    for(int i=0;i<h;i++)
    {
        //forward pass:
        interD[i*w] = Chor[i*w]/Bhor[i*w];
        interE[i*w] = cur[i*w] /Bhor[i*w];
        for(int j=1;j<w;j++)
        {
            denom = Bhor[i*w+j]-interD[i*w+j-1]*Ahor[i*w+j];
            interD[i*w+j] = Chor[i*w+j]/denom;
            interE[i*w+j] = (cur[i*w+j]-interE[i*w+j-1]*Ahor[i*w+j])/denom;
        }

        //backward pass:
        cur[i*w+w-1] = interE[i*w+w-1];
        for(int j=w-2;j>=0;j--)
            cur[i*w+j] = interE[i*w+j]-interD[i*w+j]*cur[i*w+j+1];
    }
}

void WeightedLeastSquaresFilterImpl::verticalPass(double* cur)
{
    double denom;
    //forward pass:
    for(int j=0;j<w;j++)
    {
        interD[j] = Cvert[j]/Bvert[j];
        interE[j] = cur[j]/Bvert[j];
    }
    for(int i=1;i<h;i++)
    {
        for(int j=0;j<w;j++)
        {
            denom = Bvert[i*w+j]-interD[(i-1)*w+j]*Avert[i*w+j];
            interD[i*w+j] = Cvert[i*w+j]/denom;
            interE[i*w+j] = (cur[i*w+j]-interE[(i-1)*w+j]*Avert[i*w+j])/denom;
        }
    }
    //backward pass:
    for(int j=0;j<w;j++)
        cur[(h-1)*w+j] = interE[(h-1)*w+j];
    for(int i=h-2;i>=0;i--)
        for(int j=w-1;j>=0;j--)
            cur[i*w+j] = interE[i*w+j]-interD[i*w+j]*cur[(i+1)*w+j];
}

////////////////////////////////////////////////////////////////////////////////////////////////

CV_EXPORTS_W
Ptr<WeightedLeastSquaresFilter> createWeightedLeastSquaresFilter(InputArray guide, double lambda, double sigma_color, int num_iter)
{
    return Ptr<WeightedLeastSquaresFilter>(WeightedLeastSquaresFilterImpl::create(guide, lambda, sigma_color, num_iter));
}

CV_EXPORTS_W
void weightedLeastSquaresFilter(InputArray guide, InputArray src, OutputArray dst, double lambda, double sigma_color, int num_iter)
{
    Ptr<WeightedLeastSquaresFilter> wls = createWeightedLeastSquaresFilter(guide, lambda, sigma_color, num_iter);
    wls->filter(src, dst);
}

}
}
