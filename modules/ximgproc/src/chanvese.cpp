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


/***************************************************************/
/* Function: dheer-prog
 * Description: This Chan vese Implementation is based on the 
 following paper 
 * Chan, Tony F., and Luminita A. Vese. "Active contours without edges." IEEE Transactions on Image Processing 10.2 (2001): 266-277.
 * This implementation is based on the level-set method and takes 
 * grayscale image as input and return a binary image as output.
 
 ***************************************************************/

#include "precomp.hpp"

using namespace std;
using namespace cv;

namespace cv { namespace ximgproc { namespace segmentation {

struct avg_vals
{
    float c1;
    int pos_counter;
    float c2;
    int neg_counter;
    avg_vals(){
        c1=0.0f; 
        pos_counter=0; 
        c2=0.0f; 
        neg_counter=0;
    }
};
class ChanVeseImpl : public ChanVese
{
public:
    ChanVeseImpl() {
        l1=1.0f; 
        l2=1.0f; 
        mu=0.1f;
        v=0.0f; 
        iter=40; 
        tol=1e-3; 
        dt=0.5f;
    }
    ~ChanVeseImpl() CV_OVERRIDE {};
    virtual void ProcessImage(InputArray src, OutputArray out) CV_OVERRIDE;
    virtual void set_Lambda(float l) CV_OVERRIDE
    {
        if (l <= 0)
        {
            l = 1;
        }
        l1 = l;
        l2 = l;
    }
    virtual void set_mu(float _mu) CV_OVERRIDE
    {
        if (_mu < 0)
        {
            _mu = 0.1;
        }
        mu = _mu;
    }
    virtual void set_v(float _v) CV_OVERRIDE
    {
        if (_v < 0)
        {
            _v = 0;
        }
        v = _v;
    }
    virtual void set_iterations(int _iter) CV_OVERRIDE
    {
        if (_iter <= 0)
        {
            _iter = 40;
        }
        iter = _iter;
    }
    virtual void set_dt (float _dt) CV_OVERRIDE
    {
        if (_dt <= 0)
        {
            _dt = 1e-3;
        }
        dt = _dt;
    }

private:
    float mu;
    float v;
    float l1, l2;
    float tol;
    int iter;
    float dt;
    static constexpr double PI = 3.14159265358979323846;
    Mat update_phi(Mat& in_img,Mat& phi,avg_vals& avg,avg_vals& new_avg,bool& continue_flag);
    
    void initialize_phi(Mat& init,Mat& in_img, avg_vals& avg);
    float dirac_t(float phi_val);
    float compute_div(Mat& in_img,Mat& phi, int& y, int& x, avg_vals& avg);

    
};
void ChanVeseImpl::initialize_phi(Mat& init,Mat& in_img, avg_vals& avg)
{
    for (int y = 0; y < init.rows; y++)
    {
        for (int x = 0; x < init.cols; x++)
        {
            float num1 = (PI / 5.0f) * (static_cast<float>(y));
            float num2 = (PI / 5.0f) * (static_cast<float>(x));
            init.at<float>(y, x) = sin(num1) * sin(num2);
            if (init.at<float>(y, x) > 0)
            {
                avg.c1 = avg.c1 + in_img.at<float>(y, x);
                avg.pos_counter++;
            }
            else
            {
                avg.c2 = avg.c2 + in_img.at<float>(y, x);
                avg.neg_counter++;
            }
        }
    }
}
float ChanVeseImpl::dirac_t(float phi_val)
{
    float denum = PI * (1 + (phi_val * phi_val));
    return (1 / denum);
}
float ChanVeseImpl::compute_div(Mat& in_img,Mat& phi, int& y, int& x, avg_vals& avg)
{
    float pos_x;
    if (x < phi.cols - 1)
    {
        pos_x = phi.at<float>(y, x + 1);
    }
    else
    {
        pos_x = phi.at<float>(y, x);
    }
    float neg_x;
    if (x > 0)
    {
        neg_x = phi.at<float>(y, x - 1);
    }
    else
    {
        neg_x = phi.at<float>(y, x);
    }
    float pos_y;
    if (y < phi.rows - 1)
    {
        pos_y = phi.at<float>(y + 1, x);
    }
    else
    {
        pos_y = phi.at<float>(y, x);
    }
    float neg_y;
    if (y > 0)
    {
        neg_y = phi.at<float>(y - 1, x);
    }
    else
    {
        neg_y = phi.at<float>(y, x);
    }
    float pos_diff_x = pos_x - phi.at<float>(y, x);
    float neg_diff_x = phi.at<float>(y, x) - neg_x;
    float pos_diff_y = pos_y - phi.at<float>(y, x);
    float neg_diff_y = phi.at<float>(y, x) - neg_y;
    float diff_x = (pos_diff_x - neg_diff_x) / 2;
    float diff_y = (pos_diff_y - neg_diff_y) / 2;
    float derac_term = dirac_t(phi.at<float>(y, x));

    float t1 =
        (neg_diff_x * neg_diff_x) / (sqrt(pow(1e-3, 2) + pow(pos_diff_x, 2) + pow(diff_y, 2)));
    float t2 =
        (neg_diff_y * pos_diff_y) / (sqrt(pow(1e-3, 2) + pow(pos_diff_y, 2) + pow(diff_x, 2)));
    float curvature_term = mu * (t1 + t2);
    float avg_c1 = avg.c1 / static_cast<float>(avg.pos_counter);
    if(avg.pos_counter == 0)
    {
        avg_c1 = 0.0f;
    }
    float avg_c2 = avg.c2 / static_cast<float>(avg.neg_counter);
    if(avg.neg_counter == 0)
    {
        avg_c2 = 0.0f;
    }
    float lam_term =
        l2 * (pow(in_img.at<float>(y, x) - avg_c2, 2)) - l1 * (pow(in_img.at<float>(y, x) - avg_c1, 2));
    return derac_term * (curvature_term + lam_term - v);
}

Mat ChanVeseImpl::update_phi(
    Mat& in_img,
    Mat& phi,
    avg_vals& avg,
    avg_vals& new_avg,
    bool& continue_flag)
{
    Mat new_phi;
    new_phi.create(phi.rows, phi.cols, CV_32F);
    float l2_err = 0.0f;
    for (int y = 0; y < phi.rows; y++)
    {
        for (int x = 0; x < phi.cols; x++)
        {
            float div = compute_div(in_img,phi, y, x, avg);
            new_phi.at<float>(y, x) = phi.at<float>(y, x) + (div * dt);
            l2_err = l2_err + sqrtf(pow(new_phi.at<float>(y, x) - phi.at<float>(y, x), 2));
        
             
            if (new_phi.at<float>(y, x) > 0)
            {
                new_avg.c1 = new_avg.c1 + in_img.at<float>(y, x);
                new_avg.pos_counter++;
            }
            else
            {
                new_avg.c2 = new_avg.c2 + in_img.at<float>(y, x);
                new_avg.neg_counter++;
            }
        }
    }
    continue_flag = false;
    if (l2_err > tol)
    {
        continue_flag = true;
    }
    return new_phi;
}
void ChanVeseImpl::ProcessImage(InputArray src, OutputArray out)
{
    Mat in_img = src.getMat();
    if(in_img.channels()>1)
    {
        cvtColor(in_img, in_img, COLOR_BGR2GRAY);
    }
    out.create(in_img.rows, in_img.cols, CV_32F);
    Mat output = out.getMat();
    in_img.convertTo(in_img, CV_32F);
    Mat phi;
    phi.create(in_img.rows, in_img.cols, CV_32F);
    avg_vals avg;
    avg.c1 = 0.0f;
    avg.c2 = 0.0f;
    avg.pos_counter = 0;
    avg.neg_counter = 0;
    initialize_phi(phi,in_img, avg);
    bool continue_flag = true;
    for (int i = 0; i < iter; i++)
    {
        float new_c1 = 0.0f;
        float new_c2 = 0.0f;
        int new_pos_counter = 0;
        int new_neg_counter = 0;
        struct avg_vals new_avg;
        new_avg.c1 = new_c1;
        new_avg.c2 = new_c2;
        new_avg.pos_counter = new_pos_counter;
        new_avg.neg_counter = new_neg_counter;
        phi = update_phi(in_img, phi, avg, new_avg, continue_flag);
        avg = new_avg;
        if (!continue_flag)
        {
            break;
        }
    }
    in_img.copyTo(output);
    for (int y = 0; y < phi.rows; y++)
    {
        for (int x = 0; x < phi.cols; x++)
        {
            if (phi.at<float>(y, x)<=0)
            {
                output.at<float>(y, x)=0.0f; 
            }
            else{
                output.at<float>(y, x)=255.0f; 
            }
        }
    }
}
Ptr<ChanVese> createChanVese()
{
    return makePtr<ChanVeseImpl>();
}
void ChanVeseInit(InputArray src, OutputArray dst, float Lambda, float v, float mu, int iter, float dt)
{
    Ptr<ChanVese> chanvese = createChanVese();
    chanvese->set_Lambda(Lambda);
    chanvese->set_v(v);
    chanvese->set_mu(mu);
    chanvese->set_iterations(iter);
    chanvese->set_dt(dt);
    chanvese->ProcessImage(src, dst);
}
}}}  // namespace cv::ximgproc::segmentation
