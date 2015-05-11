/*********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2009
 * Engin Tola
 * web : http://www.engintola.com
 * email : engin.tola+libdaisy@gmail.com
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/*
 "DAISY: An Efficient Dense Descriptor Applied to Wide Baseline Stereo"
 by Engin Tola, Vincent Lepetit and Pascal Fua. IEEE Transactions on
 Pattern Analysis and achine Intelligence, 31 Mar. 2009.
 IEEE computer Society Digital Library. IEEE Computer Society,
 http:doi.ieeecomputersociety.org/10.1109/TPAMI.2009.77

 "A fast local descriptor for dense matching" by Engin Tola, Vincent
 Lepetit, and Pascal Fua. Intl. Conf. on Computer Vision and Pattern
 Recognition, Alaska, USA, June 2008

 OpenCV port by: Cristian Balint <cristian dot balint at gmail dot com>
 */

#include "precomp.hpp"
#include "opencv2/imgproc/imgproc_c.h"

#include <fstream>
#include <stdlib.h>

namespace cv
{
namespace xfeatures2d
{

// constants
const double g_sigma_0 = 1;
const double g_sigma_1 = sqrt(2.0);
const double g_sigma_2 = 8;
const double g_sigma_step = std::pow(2,1.0/2);
const int g_scale_st = int( (log(g_sigma_1/g_sigma_0)) / log(g_sigma_step) );
static int g_scale_en = 1;

const double g_sigma_init = 1.6;
const static int g_grid_orientation_resolution = 360;

static const int MAX_CUBE_NO = 64;
static const int MAX_NORMALIZATION_ITER = 5;

int g_cube_number;
int g_selected_cubes[MAX_CUBE_NO]; // m_rad_q_no < MAX_CUBE_NO

/*
 !DAISY implementation
 */
class DAISY_Impl : public DAISY
{

public:
    /** Constructor
     @param radius radius of the descriptor at the initial scale
     @param q_radius amount of radial range divisions
     @param q_theta amount of angular range divisions
     @param q_hist amount of gradient orientations range divisions
     @param mode computation of descriptors
     @param norm normalization type
     @param H optional 3x3 homography matrix used to warp the grid of daisy but sampling keypoints remains unwarped on image
     @param interpolation switch to disable interpolation at minor costs of quality (default is true)
     */
    explicit DAISY_Impl(float radius=15, int q_radius=3, int q_theta=8, int q_hist=8,
        int mode = DAISY::ONLY_KEYS, int norm = DAISY::NRM_NONE, InputArray H = noArray(),
        bool interpolation = true, bool use_orientation = false);

    virtual ~DAISY_Impl();

    /** returns the descriptor length in bytes */
    virtual int descriptorSize() const {
        // +1 is for center pixel
        return ( (m_rad_q_no * m_th_q_no + 1) * m_hist_th_q_no );
    };
    /** returns the descriptor type */
    virtual int descriptorType() const { return CV_32F; }
    /** returns the default norm type */
    virtual int defaultNorm() const { return NORM_L2; }

    // main compute routine
    virtual void compute( InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors );

protected:

    /*
     * DAISY parameters
     */

    // operation mode
    int m_mode;

    // maximum radius of the descriptor region.
    float m_rad;

    // the number of quantizations of the radius.
    int m_rad_q_no;

    // the number of quantizations of the angle.
    int m_th_q_no;

    // the number of quantizations of the gradient orientations.
    int m_hist_th_q_no;

    // holds the type of the normalization to apply; equals to NRM_PARTIAL by
    // default. change the value using set_normalization() function.
    int m_nrm_type;

    // holds optional H matrix
    InputArray m_h_matrix;

    // input image.
    float* m_image;

    // image height
    int m_h;

    // image width
    int m_w;

    // stores the descriptors : its size is [ m_w * m_h * m_descriptor_size ].
    float* m_dense_descriptors;

    // stores the layered gradients in successively smoothed form: layer[n] =
    // m_gradient_layers * gaussian( sigma_n ); n>= 1; layer[0] is the layered_gradient
    float* m_smoothed_gradient_layers;

    // if set to true, descriptors are scale invariant
    bool m_scale_invariant;

    // if set to true, descriptors are rotation invariant
    bool m_rotation_invariant;

    // number of bins in the histograms while computing orientation
    int m_orientation_resolution;

    // hold the scales of the pixels
    float* m_scale_map;

    // holds the orientaitons of the pixels
    int* m_orientation_map;

    // Holds the oriented coordinates (y,x) of the grid points of the region.
    double** m_oriented_grid_points;

    // holds the gaussian sigmas for radius quantizations for an incremental
    // application
    double* m_cube_sigmas;

    bool m_descriptor_memory;
    bool m_workspace_memory;

    // the number of grid locations
    int m_grid_point_number;

    // the size of the descriptor vector
    int m_descriptor_size;

    // holds the amount of shift that's required for histogram computation
    double m_orientation_shift_table[360];

    // if enabled, descriptors are computed with casting non-integer locations
    // to integer positions otherwise we use interpolation.
    bool m_disable_interpolation;

    // switch to enable sample by keypoints orientation
    bool m_use_orientation;

    // size of m_hsz layers at a single sigma: m_hsz * m_layer_size
    int m_cube_size;

    // size of the layer : m_h*m_w
    int m_layer_size;

    /*
     * DAISY functions
     */

    // computes the histogram at yx; the size of histogram is m_hist_th_q_no
    void compute_histogram( float* hcube, int y, int x, float* histogram );

    // reorganizes the cube data so that histograms are sequential in memory.
    void compute_histograms();

    // emulates the way sift is normalized.
    void normalize_sift_way( float* desc );

    // normalizes the descriptor histogram by histogram
    void normalize_partial( float* desc );

    // normalizes the full descriptor.
    void normalize_full( float* desc );

    // initializes the class: computes gradient and structure-points
    void initialize();

    void update_selected_cubes();

    int quantize_radius( float rad );

    int filter_size( double sigma );

    // computes scales for every pixel and scales the structure grid so that the
    // resulting descriptors are scale invariant.  you must set
    // m_scale_invariant flag to 1 for the program to call this function
    void compute_scales();

    // Return a number in the range [-0.5, 0.5] that represents the location of
    // the peak of a parabola passing through the 3 evenly spaced samples.  The
    // center value is assumed to be greater than or equal to the other values
    // if positive, or less than if negative.
    float interpolate_peak( float left, float center, float right );

    // Smooth a histogram by using a [1/3 1/3 1/3] kernel.  Assume the histogram
    // is connected in a circular buffer.
    void smooth_histogram( float *hist, int bins );

    // computes pixel orientations and rotates the structure grid so that
    // resulting descriptors are rotation invariant. If the scales is also
    // detected, then orientations are computed at the computed scales. you must
    // set m_rotation_invariant flag to 1 for the program to call this function
    void compute_orientations();

    // the clipping threshold to use in normalization: values above this value
    // are clipped to this value for normalize_sift_way() function
    float m_descriptor_normalization_threshold;

    // computes the sigma's of layers from descriptor parameters if the user did
    // not sets it. these define the size of the petals of the descriptor.
    void compute_cube_sigmas();

    // Computes the locations of the unscaled unrotated points where the
    // histograms are going to be computed according to the given parameters.
    void compute_grid_points();

    // Computes the locations of the unscaled rotated points where the
    // histograms are going to be computed according to the given parameters.
    void compute_oriented_grid_points();

    // smooths each of the layers by a Gaussian having "sigma" standart
    // deviation.
    void smooth_layers( float*layers, int h, int w, int layer_number, float sigma );

    // Holds the coordinates (y,x) of the grid points of the region.
    double** m_grid_points;

    int get_hq() { return m_hist_th_q_no; }
    int get_thq() { return m_th_q_no; }
    int get_rq() { return m_rad_q_no; }
    float get_rad() { return m_rad; }

    // sets the type of the normalization to apply out of {NRM_PARTIAL,
    // NRM_FULL, NRM_SIFT}. Call before using get_descriptor() if you want to
    // change the default normalization type.
    void set_normalization( int nrm_type ) { m_nrm_type = nrm_type; }

    // applies one of the normalizations (partial,full,sift) to the desciptors.
    void normalize_descriptors( int nrm_type = DAISY::NRM_NONE );

    // normalizes histograms individually
    void normalize_histograms();

    // gets the histogram at y,x with 'orientation' from the r'th cube
    float* get_histogram( int y, int x, int r );

    // if called, I don't use interpolation in the computation of
    // descriptors.
    void disable_interpolation() { m_disable_interpolation = true; }

    // returns the region number.
    int grid_point_number() { return m_grid_point_number; }

    // releases all the used memory; call this if you want to process
    // multiple images within a loop.
    void reset();

    // releases unused memory after descriptor computation is completed.
    void release_auxilary();

    // computes the descriptors for every pixel in the image.
    void compute_descriptors();

    // returns all the descriptors.
    float* get_dense_descriptors();

    // returns oriented grid points. default is 0 orientation.
    double* get_grid(int o=0);

    // EXPERIMENTAL: DO NOT USE IF YOU ARE NOT ENGIN TOLA: tells to compute the
    // scales for every pixel so that the resulting descriptors are scale
    // invariant.
    void scale_invariant( bool state = true )
    {
         g_scale_en = (int)( (log(g_sigma_2/g_sigma_0)) / log(g_sigma_step) ) - g_scale_st;
         m_scale_invariant = state;
    }

    // EXPERIMENTAL: DO NOT USE IF YOU ARE NOT ENGIN TOLA: tells to compute the
    // orientations for every pixel so that the resulting descriptors are
    // rotation invariant. orientation steps are 360/ori_resolution
    void rotation_invariant( int ori_resolution = 36, bool state = true )
    {
         m_rotation_invariant = state;
         m_orientation_resolution = ori_resolution;
    }

    // sets the gaussian variances manually. must be called before
    // initialize() to be considered. must be exact sigma values -> f
    // converts to incremental format.
    void set_cube_gaussians( double* sigma_array, int sz );

    int* get_orientation_map() { return m_orientation_map; }

    // call compute_descriptor_memory to find the amount of memory to allocate
    void set_descriptor_memory( float* descriptor, long int d_size );

    // call compute_workspace_memory to find the amount of memory to allocate
    void set_workspace_memory( float* workspace, long int w_size );

    // returns the amount of memory needed for the compute_descriptors()
    // function. it is basically equal to imagesize x descriptor_size
    int compute_descriptor_memory() {
      if( m_h == 0 || m_descriptor_size == 0 ) {
          CV_Error( Error::StsInternal, "Image and descriptor size is zero" );
      }
      return m_w * m_h * m_descriptor_size;
    }

    // returns the amount of memory needed for workspace. call before initialize()
    int compute_workspace_memory() {
      if( m_cube_size == 0 ) {
          CV_Error( Error::StsInternal, "Cube size is zero" );
      }
      return (g_cube_number+1)* m_cube_size;
    }

    void normalize_descriptor( float* desc, int nrm_type = DAISY::NRM_NONE )
    {
      if( nrm_type == DAISY::NRM_NONE )      nrm_type = m_nrm_type;
      else if( nrm_type == DAISY::NRM_PARTIAL ) normalize_partial(desc);
      else if( nrm_type == DAISY::NRM_FULL    ) normalize_full(desc);
      else if( nrm_type == DAISY::NRM_SIFT    ) normalize_sift_way(desc);
      else
          CV_Error( Error::StsInternal, "No such normalization" );
    }

    // transform a point via the homography
    void point_transform_via_homography( double* H, double x, double y, double &u, double &v )
    {
      double kxp = H[0]*x + H[1]*y + H[2];
      double kyp = H[3]*x + H[4]*y + H[5];
      double kp  = H[6]*x + H[7]*y + H[8];
      u = kxp / kp;
      v = kyp / kp;
    }

private:

    // returns the descriptor vector for the point (y, x) !!! use this for
    // precomputed operations meaning that you must call compute_descriptors()
    // before calling this function. if you want normalized descriptors, call
    // normalize_descriptors() before calling compute_descriptors()
    inline void get_descriptor( int y, int x, float* &descriptor );

    // computes the descriptor and returns the result in 'descriptor' ( allocate
    // 'descriptor' memory first ie: float descriptor = new
    // float[m_descriptor_size]; -> the descriptor is normalized.
    inline void get_descriptor( double y, double x, int orientation, float* descriptor );

    // computes the descriptor and returns the result in 'descriptor' ( allocate
    // 'descriptor' memory first ie: float descriptor = new
    // float[m_descriptor_size]; -> the descriptor is NOT normalized.
    inline void get_unnormalized_descriptor( double y, double x, int orientation, float* descriptor );

    // computes the descriptor at homography-warped grid. (y,x) is not the
    // coordinates of this image but the coordinates of the original grid where
    // the homography will be applied. Meaning that the grid is somewhere else
    // and we warp this grid with H and compute the descriptor on this warped
    // grid; returns null/false if centers falls outside the image; allocate
    // 'descriptor' memory first. descriptor is normalized.
    inline bool get_descriptor( double y, double x, int orientation, double* H, float* descriptor);

    // computes the descriptor at homography-warped grid. (y,x) is not the
    // coordinates of this image but the coordinates of the original grid where
    // the homography will be applied. Meaning that the grid is somewhere else
    // and we warp this grid with H and compute the descriptor on this warped
    // grid; returns null/false if centers falls outside the image; allocate
    // 'descriptor' memory first. descriptor is NOT normalized.
    inline bool get_unnormalized_descriptor( double y, double x, int orientation, double* H, float* descriptor );

    // compute the smoothed gradient layers.
    inline void compute_smoothed_gradient_layers();

    // does not use interpolation while computing the histogram.
    inline void ni_get_histogram( float* histogram, int y, int x, int shift, float* hcube );

    // returns the interpolated histogram: picks either bi_get_histogram or
    // ti_get_histogram depending on 'shift'
    inline void i_get_histogram( float* histogram, double y, double x, double shift, float* cube );

    // records the histogram that is computed by bilinear interpolation
    // regarding the shift in the spatial coordinates. hcube is the
    // histogram cube for a constant smoothness level.
    inline void bi_get_histogram( float* descriptor, double y, double x, int shift, float* hcube );

    // records the histogram that is computed by trilinear interpolation
    // regarding the shift in layers and spatial coordinates. hcube is the
    // histogram cube for a constant smoothness level.
    inline void ti_get_histogram( float* descriptor, double y, double x, double shift, float* hcube );

    // uses interpolation, for no interpolation call ni_get_descriptor. see also get_descriptor
    inline void i_get_descriptor( double y, double x, int orientation, float* descriptor );

    // does not use interpolation. for w/interpolation, call i_get_descriptor. see also get_descriptor
    inline void ni_get_descriptor( double y, double x, int orientation, float* descriptor );

    // uses interpolation for no interpolation call ni_get_descriptor. see also get_descriptor
    inline bool i_get_descriptor( double y, double x, int orientation, double* H, float* descriptor );

    // does not use interpolation. for w/interpolation, call i_get_descriptor. see also get_descriptor
    inline bool ni_get_descriptor( double y, double x, int orientation, double* H, float* descriptor );

    // creates a 1D gaussian filter with N(mean,sigma).
    inline void gaussian_1d( float* fltr, int fsz, float sigma, float mean )
    {
      CV_Assert(fltr != NULL);
      int sz = (fsz-1)/2;
      int counter=-1;
      float sum = 0.0;
      float v = 2*sigma*sigma;
      for( int x=-sz; x<=sz; x++ )
      {
         counter++;
         fltr[counter] = exp((-(x-mean)*(x-mean))/v);
         sum += fltr[counter];
      }

      if( sum != 0 )
      {
         for( int x=0; x<fsz; x++ )
            fltr[x] /= sum;
      }
    }

    inline void conv_horizontal( float* image, int h, int w, float* kernel, int ksize )
    {
      CvMat cvI; cvInitMatHeader(&cvI, h, w, CV_32FC1, (float*)image);
      CvMat cvK; cvInitMatHeader(&cvK, 1, ksize, CV_32FC1, (float*)kernel);
      cvFilter2D( &cvI, &cvI, &cvK );
    }
    inline void conv_horizontal( double* image, int h, int w, double* kernel, int ksize )
    {
      CvMat cvI; cvInitMatHeader(&cvI, h, w, CV_64FC1, (double*)image);
      CvMat cvK; cvInitMatHeader(&cvK, 1, ksize, CV_64FC1, (double*)kernel);
      cvFilter2D( &cvI, &cvI, &cvK );
    }

    inline void conv_vertical( float* image, int h, int w, float* kernel, int ksize )
    {
      CvMat cvI; cvInitMatHeader(&cvI, h, w, CV_32FC1, (float*)image);
      CvMat cvK; cvInitMatHeader(&cvK, ksize, 1, CV_32FC1, (float*)kernel);
      cvFilter2D( &cvI, &cvI, &cvK );
    }

    inline void conv_vertical( double* image, int h, int w, double* kernel, int ksize )
    {
      CvMat cvI; cvInitMatHeader(&cvI, h, w, CV_64FC1, (double*)image);
      CvMat cvK; cvInitMatHeader(&cvK, ksize, 1, CV_64FC1, (double*)kernel);
      cvFilter2D( &cvI, &cvI, &cvK );
    }

    /*
     * DAISY utilities
     */

    template<class T>
    class rectangle
    {
    public:
      T lx, ux, ly, uy;
      T dx, dy;
      rectangle(T xl, T xu, T yl, T yu) { lx=xl; ux=xu; ly=yl; uy=yu; dx=ux-lx; dy=uy-ly; };
      rectangle()                       { lx = ux = ly = uy = dx = dy = 0; };
    };

    // checks if the number x is between lx - ux interval.
    // the equality is checked depending on the value of le and ue parameters.
    // if le=1 => lx<=x is checked else lx<x is checked
    // if ue=1 => x<=ux is checked else x<ux is checked
    // by default x is searched inside of [lx,ux)
    template<class T1, class T2, class T3> inline
    bool is_inside(T1 x, T2 lx, T3 ux, bool le=true, bool ue=false)
    {
      if( ( ((lx<x)&&(!le)) || ((lx<=x)&&le) ) && ( ((x<ux)&&(!ue)) || ((x<=ux)&&ue) )    )
      {
         return true;
      }
      else
      {
         return false;
      }
    }

    // checks if the number x is between lx - ux and/or y is between ly - uy interval.
    // If the number is inside, then function returns true, else it returns false.
    // the equality is checked depending on the value of le and ue parameters.
    // if le=1 => lx<=x is checked else lx<x is checked
    // if ue=1 => x<=ux is checked else x<ux is checked
    // by default x is searched inside of [lx,ux).
    // the same equality check is applied to the y variable as well.
    // If the 'oper' is set '&' both of the numbers must be within the interval to return true
    // But if the 'oper' is set to '|' then only one of them being true is sufficient.
    template<class T1, class T2, class T3> inline
    bool is_inside(T1 x, T2 lx, T3 ux, T1 y, T2 ly, T3 uy, bool le=true, bool ue=false, char oper='&')
    {
      switch( oper )
      {
      case '|':
         if( is_inside(x,lx,ux,le,ue) || is_inside(y,ly,uy,le,ue) )
            return true;
         return false;

      default:
         if( is_inside(x,lx,ux,le,ue) && is_inside(y,ly,uy,le,ue) )
            return true;
         return false;
      }
    }

    // checks if the number x is between lx - ux and/or y is between ly - uy interval.
    // If the number is inside, then function returns true, else it returns false.
    // the equality is checked depending on the value of le and ue parameters.
    // if le=1 => lx<=x is checked else lx<x is checked
    // if ue=1 => x<=ux is checked else x<ux is checked
    // by default x is searched inside of [lx,ux).
    // the same equality check is applied to the y variable as well.
    // If the 'oper' is set '&' both of the numbers must be within the interval to return true
    // But if the 'oper' is set to '|' then only one of them being true is sufficient.
    template<class T1, class T2> inline
    bool is_inside(T1 x, T1 y, rectangle<T2> roi, bool le=true, bool ue=false, char oper='&')
    {
      switch( oper )
      {
      case '|':
         if( is_inside(x,roi.lx,roi.ux,le,ue) || is_inside(y,roi.ly,roi.uy,le,ue) )
            return true;
         return false;

      default:
         if( is_inside(x,roi.lx,roi.ux,le,ue) && is_inside(y,roi.ly,roi.uy,le,ue) )
            return true;
         return false;
      }
    }

    // checks if the number x is outside lx - ux interval
    // the equality is checked depending on the value of le and ue parameters.
    // if le=1 => lx>x is checked else lx>=x is checked
    // if ue=1 => x>ux is checked else x>=ux is checked
    // by default is x is searched outside of [lx,ux)
    template<class T1, class T2, class T3> inline
    bool is_outside(T1 x, T2 lx, T3 ux, bool le=true, bool ue=false)
    {
      return !(is_inside(x,lx,ux,le,ue));
    }

    // checks if the numbers x and y is outside their intervals.
    // The equality is checked depending on the value of le and ue parameters.
    // If le=1 => lx>x is checked else lx>=x is checked
    // If ue=1 => x>ux is checked else x>=ux is checked
    // By default is x is searched outside of [lx,ux) (Similarly for y)
    // By default, 'oper' is set to OR. If one of them is outside it returns
    // true otherwise false.
    template<class T1, class T2, class T3> inline
    bool is_outside(T1 x, T2 lx, T3 ux, T1 y, T2 ly, T3 uy, bool le=true, bool ue=false, char oper='|')
    {
      switch( oper )
      {
      case '&':
         if( is_outside(x,lx,ux,le,ue) && is_outside(y,ly,uy,le,ue) )
            return true;
         return false;
      default:
         if( is_outside(x,lx,ux,le,ue) || is_outside(y,ly,uy,le,ue) )
            return true;
         return false;
      }
    }

    // checks if the numbers x and y is outside their intervals.
    // The equality is checked depending on the value of le and ue parameters.
    // If le=1 => lx>x is checked else lx>=x is checked
    // If ue=1 => x>ux is checked else x>=ux is checked
    // By default is x is searched outside of [lx,ux) (Similarly for y)
    // By default, 'oper' is set to OR. If one of them is outside it returns
    // true otherwise false.
    template<class T1, class T2> inline
    bool is_outside(T1 x, T1 y, rectangle<T2> roi, bool le=true, bool ue=false, char oper='|')
    {
      switch( oper )
      {
      case '&':
         if( is_outside(x,roi.lx,roi.ux,le,ue) && is_outside(y,roi.ly,roi.uy,le,ue) )
            return true;
         return false;
      default:
         if( is_outside(x,roi.lx,roi.ux,le,ue) || is_outside(y,roi.ly,roi.uy,le,ue) )
            return true;
         return false;
      }
    }

    // computes the square of a number and returns it.
    template<class T> inline
    T square(T a)
    {
      return a*a;
    }

    // computes the square of an array. if in_place is enabled, the
    // result is returned in the array arr.
    template<class T> inline
    T* square(T* arr, int sz, bool in_place=false)
    {
      T* out;
      if( in_place ) out = arr;
      else           out = allocate<T>(sz);

      for( int i=0; i<sz; i++ )
         out[i] = arr[i]*arr[i];

      return out;
    }

    // computes the l2norm of an array: [ sum_i( [a(i)]^2 ) ]^0.5
    template<class T> inline
    float l2norm( T* a, int sz)
    {
      float norm=0;
      for( int k=0; k<sz; k++ )
         norm += a[k]*a[k];
      return sqrt(norm);
    }

    // computes the l2norm of the difference of two arrays: [ sum_i( [a(i)-b(i)]^2 ) ]^0.5
    template<class T1, class T2> inline
    float l2norm( T1* a, T2* b, int sz)
    {
      float norm=0;
      for( int i=0; i<sz; i++ )
      {
         norm += square( (float)a[i] - (float)b[i] );
      }
      norm = sqrt( norm );

      return norm;
    }

    template<class T> inline
    float l2norm( T y0, T x0, T y1, T x1 )
    {
      float d0 = x0 - x1;
      float d1 = y0 - y1;

      return sqrt( d0*d0 + d1*d1 );
    }

    // allocates a memory of size sz and returns a pointer to the array
    template<class T> inline
    T* allocate(const int sz)
    {
      T* array = new T[sz];
      return array;
    }

    // allocates a memory of size ysz x xsz and returns a double pointer to it
    template<class T> inline
    T** allocate(const int ysz, const int xsz)
    {
      T** mat = new T*[ysz];
      int i;

      for(i=0; i<ysz; i++ )
         mat[i] = new T[xsz];
      // allocate<T>(xsz);

      return mat;
    }

    // deallocates the memory and sets the pointer to null.
    template<class T> inline
    void deallocate(T* &array)
    {
      delete[] array;
      array = NULL;
    }

    // deallocates the memory and sets the pointer to null.
    template<class T> inline
    void deallocate(T** &mat, int ysz)
    {
      if( mat == NULL ) return;

      for(int i=0; i<ysz; i++)
         deallocate(mat[i]);

      delete[] mat;
      mat = NULL;
    }

    // Converts the given polar coordinates of a point to cartesian
    // ones.
    template<class T1, class T2> inline
    void polar2cartesian(T1 r, T1 t, T2 &y, T2 &x)
    {
      x = (T2)( r * cos( t ) );
      y = (T2)( r * sin( t ) );
    }


    template<typename T> inline
    void convolve_sym_( T* image, int h, int w, T* kernel, int ksize )
    {
      conv_horizontal( image, h, w, kernel, ksize );
      conv_vertical  ( image, h, w, kernel, ksize );
    }

    template<class T>
    inline void convolve_sym( T* image, int h, int w, T* kernel, int ksize, T* out=NULL )
    {
      if( out == NULL ) out = image;
      else memcpy( out, image, sizeof(T)*h*w );
      convolve_sym_(out, h, w, kernel, ksize);
    }

    // divides the elements of the array with num
    template<class T1, class T2> inline
    void divide(T1* arr, int sz, T2 num )
    {
      float inv_num = 1.0 / num;

      for( int i=0; i<sz; i++ )
      {
         arr[i] = (T1)(arr[i]*inv_num);
      }
    }

    // returns an array filled with zeroes.
    template<class T> inline
    T* zeros(int r)
    {
      T* data = allocate<T>(r);
      memset( data, 0, sizeof(T)*r );
      return data;
    }

    template<class T> inline
    T* layered_gradient( T* data, int h, int w, int layer_no=8 )
    {
      int data_size = h * w;
      T* layers = zeros<T>(layer_no * data_size);

      // smooth the data matrix
      T* bdata = blur_gaussian_2d<T,T>( data, h, w, 0.5, 5, false );

      T *dx = new T[data_size];
      T *dy = new T[data_size];
      gradient(bdata, h, w, dy, dx);
      deallocate( bdata );

#if defined _OPENMP
#pragma omp parallel for
#endif
      for( int l=0; l<layer_no; l++ )
      {
         float angle = 2*l*CV_PI/layer_no;
         float kos = cos( angle );
         float zin = sin( angle );

         T* layer_l = layers + l*data_size;

         for( int index=0; index<data_size; index++ )
         {
            float value = kos * dx[ index ] + zin * dy[ index ];
            if( value > 0 ) layer_l[index] = value;
            else            layer_l[index] = 0;
         }
      }
      deallocate(dy);
      deallocate(dx);

      return layers;
    }

    // computes the gradient of an image and returns the result in
    // pointers to REAL.
    template <class T> inline
    void gradient(T* im, int h, int w, T* dy, T* dx)
    {
      CV_Assert( dx != NULL );
      CV_Assert( dy != NULL );

      for( int y=0; y<h; y++ )
      {
         int yw = y*w;
         for( int x=0; x<w; x++ )
         {
            int ind = yw+x;
            // dx
            if( x>0 && x<w-1 ) dx[ind] = ((T)im[ind+1]-(T)im[ind-1])/2.0;
            if( x==0         ) dx[ind] = ((T)im[ind+1]-(T)im[ind]);
            if( x==w-1       ) dx[ind] = ((T)im[ind  ]-(T)im[ind-1]);

            //dy
            if( y>0 && y<h-1 ) dy[ind] = ((T)im[ind+w]-(T)im[ind-w])/2.0;
            if( y==0         ) dy[ind] = ((T)im[ind+w]-(T)im[ind]);
            if( y==h-1       ) dy[ind] = ((T)im[ind]  -(T)im[ind-w]);
         }
      }
    }

    // be careful, 'data' is destroyed afterwards
    template<class T> inline
    //  original T* workspace=0 was removed
    void layered_gradient( T* data, int h, int w, int layer_no, T* layers, int lwork = 0 )
    {
      int data_size = h * w;
      CV_Assert(layers!=NULL);
      memset(layers,0,sizeof(T)*data_size*layer_no);

      bool was_empty = false;
      T* work=NULL;
      if( lwork < 3*data_size ) {
         work = new T[3*data_size];
         was_empty = true;
      }

      // // smooth the data matrix
      // T* bdata = blur_gaussian_2d<T,T>( data, h, w, 0.5, 5, false);
      float kernel[5]; gaussian_1d(kernel, 5, 0.5, 0);
      memcpy( work, data, sizeof(T)*data_size );
      convolve_sym( work, h, w, kernel, 5 );

      T *dx = work+data_size;
      T *dy = work+2*data_size;
      gradient( work, h, w, dy, dx );

#if defined _OPENMP
#pragma omp parallel for
#endif
      for( int l=0; l<layer_no; l++ )
      {
         float angle = 2*l*CV_PI/layer_no;
         float kos = cos( angle );
         float zin = sin( angle );

         T* layer_l = layers + l*data_size;

         for( int index=0; index<data_size; index++ )
         {
            float value = kos * dx[ index ] + zin * dy[ index ];
            if( value > 0 ) layer_l[index] = value;
            else            layer_l[index] = 0;
         }
      }
      if( was_empty ) delete []work;
    }

    // casts a type T2 array into a type T1 array.
    template<class T1, class T2> inline
    T1* type_cast(T2* data, int sz)
    {
      T1* out = new T1[sz];

      for( int i=0; i<sz; i++ )
         out[i] = (T1)data[i];

      return out;
    }

    // Applies a 2d gaussian blur of sigma std to the input array.  if
    // kernel_size is not set or it is set to 0, then it is taken as
    // 3*sigma and if it is set to an even number, it is incremented
    // to be an odd number.  if in_place=true, then T1 must be equal
    // to T2 naturally.
    template<class T1, class T2> inline
    T1* blur_gaussian_2d( T2* array, int rn, int cn, float sigma, int kernel_size = 0, bool in_place = false )
    {
      T1* out = NULL;

      if( in_place )
         out = (T1*)array;
      else
         out = type_cast<T1,T2>(array,rn*cn);

      if( kernel_size == 0 )
         kernel_size = (int)(3*sigma);

      if( kernel_size%2 == 0 ) kernel_size++; // kernel size must be odd
      if( kernel_size < 3 ) kernel_size= 3;  // kernel size cannot be smaller than 3

      float* kernel = new float[kernel_size];
      gaussian_1d(kernel, kernel_size, sigma, 0);

      // !! apply the filter separately
      convolve_sym( out, rn, cn, kernel, kernel_size );
      // conv_horizontal( out, rn, cn, kernel, kernel_size);
      // conv_vertical  ( out, rn, cn, kernel, kernel_size);

      deallocate(kernel);
      return out;
    }

}; // END DAISY_Impl CLASS


// -------------------------------------------------
/* DAISY computation routines */

float* DAISY_Impl::get_histogram( int y, int x, int r )
{
    CV_Assert( y >= 0 && y < m_h );
    CV_Assert( x >= 0 && x < m_w );
    CV_Assert( m_smoothed_gradient_layers );
    CV_Assert( m_oriented_grid_points );
    return m_smoothed_gradient_layers+g_selected_cubes[r]*m_cube_size + (y*m_w+x)*m_hist_th_q_no;
    // i_get_histogram( histogram, y, x, 0, m_smoothed_gradient_layers+g_selected_cubes[r]*m_cube_size );
}


float* DAISY_Impl::get_dense_descriptors()
{
    return m_dense_descriptors;
}

double* DAISY_Impl::get_grid(int o)
{
    CV_Assert( o >= 0 && o < 360 );
    return m_oriented_grid_points[o];
}

void DAISY_Impl::reset()
{
    deallocate( m_image );
    // deallocate( m_grid_points, m_grid_point_number );
    // deallocate( m_oriented_grid_points, g_grid_orientation_resolution );
    // deallocate( m_cube_sigmas );
    deallocate( m_orientation_map );
    deallocate( m_scale_map );
    if( !m_descriptor_memory ) deallocate( m_dense_descriptors );
    if( !m_workspace_memory ) deallocate(m_smoothed_gradient_layers);
}

void DAISY_Impl::release_auxilary()
{
    deallocate( m_image );
    deallocate( m_orientation_map );
    deallocate( m_scale_map );

    if( !m_workspace_memory ) deallocate(m_smoothed_gradient_layers);

    deallocate( m_grid_points, m_grid_point_number );
    deallocate( m_oriented_grid_points, g_grid_orientation_resolution );
    deallocate( m_cube_sigmas );
}

void DAISY_Impl::compute_grid_points()
{
    double r_step = m_rad / m_rad_q_no;
    double t_step = 2*CV_PI/ m_th_q_no;

    if( m_grid_points )
      deallocate( m_grid_points, m_grid_point_number );

    m_grid_points = allocate<double>(m_grid_point_number, 2);
    for( int y=0; y<m_grid_point_number; y++ )
    {
      m_grid_points[y][0] = 0;
      m_grid_points[y][1] = 0;
    }

    for( int r=0; r<m_rad_q_no; r++ )
    {
      int region = r*m_th_q_no+1;
      for( int t=0; t<m_th_q_no; t++ )
      {
         double y, x;
         polar2cartesian( (r+1)*r_step, t*t_step, y, x );
         m_grid_points[region+t][0] = y;
         m_grid_points[region+t][1] = x;
      }
    }

    compute_oriented_grid_points();
}

/// Computes the descriptor by sampling convoluted orientation maps.
void DAISY_Impl::compute_descriptors()
{
    if( m_scale_invariant    ) compute_scales();
    if( m_rotation_invariant ) compute_orientations();
    if( !m_descriptor_memory ) m_dense_descriptors = allocate <float>(m_h*m_w*m_descriptor_size);

    memset(m_dense_descriptors, 0, sizeof(float)*m_h*m_w*m_descriptor_size);

    int y, x, index, orientation;
#if defined _OPENMP
#pragma omp parallel for private(y,x,index,orientation)
#endif
    for( y=0; y<m_h; y++ )
    {
      for( x=0; x<m_w; x++ )
      {
         index=y*m_w+x;
         orientation=0;
         if( m_orientation_map ) orientation = m_orientation_map[index];
         if( !( orientation >= 0 && orientation < g_grid_orientation_resolution ) ) orientation = 0;
         get_unnormalized_descriptor( y, x, orientation, &(m_dense_descriptors[index*m_descriptor_size]) );
      }
    }
}

void DAISY_Impl::smooth_layers( float* layers, int h, int w, int layer_number, float sigma )
{
    int fsz = filter_size(sigma);
    float* filter = new float[fsz];
    gaussian_1d(filter, fsz, sigma, 0);
    int i;
    float* layer=0;
#if defined _OPENMP
#pragma omp parallel for private(i, layer)
#endif
    for( i=0; i<layer_number; i++ )
    {
      layer = layers + i*h*w;
      convolve_sym( layer, h, w, filter, fsz );
    }
    deallocate(filter);
}

void DAISY_Impl::normalize_partial( float* desc )
{
    float norm;
    for( int h=0; h<m_grid_point_number; h++ )
    {
      norm =  l2norm( &(desc[h*m_hist_th_q_no]), m_hist_th_q_no );
      if( norm != 0.0 ) divide( desc+h*m_hist_th_q_no, m_hist_th_q_no, norm);
    }
}

void DAISY_Impl::normalize_full( float* desc )
{
    float norm =  l2norm( desc, m_descriptor_size );
    if( norm != 0.0 ) divide(desc, m_descriptor_size, norm);
}

void DAISY_Impl::normalize_sift_way( float* desc )
{
    bool changed = true;
    int iter = 0;
    float norm;
    int h;
    while( changed && iter < MAX_NORMALIZATION_ITER )
    {
      iter++;
      changed = false;

      norm = l2norm( desc, m_descriptor_size );
      if( norm > 1e-5 )
         divide( desc, m_descriptor_size, norm);

      for( h=0; h<m_descriptor_size; h++ )
      {
         if( desc[ h ] > m_descriptor_normalization_threshold )
         {
            desc[ h ] = m_descriptor_normalization_threshold;
            changed = true;
         }
      }
    }
}

void DAISY_Impl::normalize_descriptors( int nrm_type )
{
    int number_of_descriptors =  m_h * m_w;
    int d;

#if defined _OPENMP
#pragma omp parallel for private(d)
#endif
    for( d=0; d<number_of_descriptors; d++ )
      normalize_descriptor( m_dense_descriptors+d*m_descriptor_size, nrm_type );
}

void DAISY_Impl::initialize()
{
    CV_Assert(m_h != 0); // no image ?
    CV_Assert(m_w != 0);

    if( m_layer_size==0 ) {
      m_layer_size = m_h*m_w;
      m_cube_size = m_layer_size*m_hist_th_q_no;
    }

    int glsz = compute_workspace_memory();
    if( !m_workspace_memory ) m_smoothed_gradient_layers = new float[glsz];

    float* gradient_layers = m_smoothed_gradient_layers;

    layered_gradient( m_image, m_h, m_w, m_hist_th_q_no, gradient_layers );

    // assuming a 0.5 image smoothness, we pull this to 1.6 as in sift
    smooth_layers( gradient_layers, m_h, m_w, m_hist_th_q_no, sqrt(g_sigma_init*g_sigma_init-0.25) );

}

void DAISY_Impl::compute_cube_sigmas()
{
    if( m_cube_sigmas == NULL )
    {
      // user didn't set the sigma's; set them from the descriptor parameters
      g_cube_number = m_rad_q_no;
      m_cube_sigmas = allocate<double>(g_cube_number);

      double r_step = double(m_rad)/m_rad_q_no;
      for( int r=0; r< m_rad_q_no; r++ )
      {
         m_cube_sigmas[r] = (r+1)*r_step/2;
      }
    }
    update_selected_cubes();
}

void DAISY_Impl::set_cube_gaussians( double* sigma_array, int sz )
{
    g_cube_number = sz;

    if( m_cube_sigmas ) deallocate( m_cube_sigmas );
    m_cube_sigmas = allocate<double>( g_cube_number );

    for( int r=0; r<g_cube_number; r++ )
    {
      m_cube_sigmas[r] = sigma_array[r];
    }
    update_selected_cubes();
}

void DAISY_Impl::update_selected_cubes()
{
    for( int r=0; r<m_rad_q_no; r++ )
    {
      double seed_sigma = (r+1)*m_rad/m_rad_q_no/2.0;
      g_selected_cubes[r] = quantize_radius( seed_sigma );
    }
}

int DAISY_Impl::quantize_radius( float rad )
{
    if( rad <= m_cube_sigmas[0              ] ) return 0;
    if( rad >= m_cube_sigmas[g_cube_number-1] ) return g_cube_number-1;

    float dist;
    float mindist=FLT_MAX;
    int mini=0;
    for( int c=0; c<g_cube_number; c++ ) {
      dist = fabs( m_cube_sigmas[c]-rad );
      if( dist < mindist ) {
         mindist = dist;
         mini=c;
      }
    }
    return mini;
}

void DAISY_Impl::compute_histograms()
{
    int r, y, x, ind;
    float* hist=0;

    for( r=0; r<g_cube_number; r++ )
    {
      float* dst = m_smoothed_gradient_layers+r*m_cube_size;
      float* src = m_smoothed_gradient_layers+(r+1)*m_cube_size;

#if defined _OPENMP
#pragma omp parallel for private(y,x,ind,hist)
#endif
      for( y=0; y<m_h; y++ )
      {
         for( x=0; x<m_w; x++ )
         {
            ind = y*m_w+x;
            hist = dst+ind*m_hist_th_q_no;
            compute_histogram( src, y, x, hist );
         }
      }
    }
}

void DAISY_Impl::normalize_histograms()
{
    for( int r=0; r<g_cube_number; r++ )
    {
      float* dst = m_smoothed_gradient_layers+r*m_cube_size;

#if defined _OPENMP
#pragma omp parallel for
#endif
      for( int y=0; y<m_h; y++ )
      {
         for( int x=0; x<m_w; x++ )
         {
            float* hist = dst + (y*m_w+x)*m_hist_th_q_no;
            float norm =  l2norm( hist, m_hist_th_q_no );
            if( norm != 0.0 ) divide( hist, m_hist_th_q_no, norm );
         }
      }
    }
}

void DAISY_Impl::compute_smoothed_gradient_layers()
{

    float* prev_cube = m_smoothed_gradient_layers;
    float* cube = NULL;

    double sigma;
    for( int r=0; r<g_cube_number; r++ )
    {
      cube = m_smoothed_gradient_layers + (r+1)*m_cube_size;

      // incremental smoothing
      if( r == 0 ) sigma = m_cube_sigmas[0];
      else         sigma = sqrt( m_cube_sigmas[r]*m_cube_sigmas[r] - m_cube_sigmas[r-1]*m_cube_sigmas[r-1] );

      int fsz = filter_size(sigma);
      float* filter = new float[fsz];
      gaussian_1d(filter, fsz, sigma, 0);

#if defined _OPENMP
#pragma omp parallel for
#endif
      for( int th=0; th<m_hist_th_q_no; th++ )
      {
         convolve_sym( prev_cube+th*m_layer_size, m_h, m_w, filter, fsz, cube+th*m_layer_size );
      }
      deallocate(filter);
      prev_cube = cube;
    }

    compute_histograms();
}

void DAISY_Impl::compute_oriented_grid_points()
{
    m_oriented_grid_points = allocate<double>( g_grid_orientation_resolution, m_grid_point_number*2 );

    for( int i=0; i<g_grid_orientation_resolution; i++ )
    {
      double angle = -i*2.0*CV_PI/g_grid_orientation_resolution;

      double kos = cos( angle );
      double zin = sin( angle );

      double* point_list = m_oriented_grid_points[ i ];

      for( int k=0; k<m_grid_point_number; k++ )
      {
         double y = m_grid_points[k][0];
         double x = m_grid_points[k][1];

         point_list[2*k+1] =  x*kos + y*zin; // x
         point_list[2*k  ] = -x*zin + y*kos; // y
      }
    }
}

void DAISY_Impl::smooth_histogram(float *hist, int hsz)
{
    int i;
    float prev, temp;

    prev = hist[hsz - 1];
    for (i = 0; i < hsz; i++)
    {
      temp = hist[i];
      hist[i] = (prev + hist[i] + hist[(i + 1 == hsz) ? 0 : i + 1]) / 3.0;
      prev = temp;
    }
}

float DAISY_Impl::interpolate_peak(float left, float center, float right)
{
    if( center < 0.0 )
    {
      left = -left;
      center = -center;
      right = -right;
    }
    CV_Assert(center >= left  &&  center >= right);

    float den = (left - 2.0 * center + right);

    if( den == 0 ) return 0;
    else           return 0.5*(left -right)/den;
}

int DAISY_Impl::filter_size( double sigma )
{
    int fsz = (int)(5*sigma);

    // kernel size must be odd
    if( fsz%2 == 0 ) fsz++;

    // kernel size cannot be smaller than 3
    if( fsz < 3 ) fsz = 3;

    return fsz;
}

void DAISY_Impl::compute_scales()
{
    //###############################################################################
    //# scale detection is work-in-progress! do not use it if you're not Engin Tola #
    //###############################################################################

    int imsz = m_w * m_h;

    float sigma = pow( g_sigma_step, g_scale_st)*g_sigma_0;

    float* sim = blur_gaussian_2d<float,float>( m_image, m_h, m_w, sigma, filter_size(sigma), false);

    float* next_sim = NULL;

    float* max_dog = allocate<float>(imsz);

    m_scale_map = allocate<float>(imsz);

    memset( max_dog, 0, imsz*sizeof(float) );
    memset( m_scale_map, 0, imsz*sizeof(float) );

    int i;
    float sigma_prev;
    float sigma_new;
    float sigma_inc;

    sigma_prev = g_sigma_0;
    for( i=0; i<g_scale_en; i++ )
    {
      sigma_new  = pow( g_sigma_step, g_scale_st+i  ) * g_sigma_0;
      sigma_inc  = sqrt( sigma_new*sigma_new - sigma_prev*sigma_prev );
      sigma_prev = sigma_new;

      next_sim = blur_gaussian_2d<float,float>( sim, m_h, m_w, sigma_inc, filter_size( sigma_inc ) , false);

#if defined _OPENMP
#pragma omp parallel for
#endif
      for( int p=0; p<imsz; p++ )
      {
         float dog = fabs( next_sim[p] - sim[p] );
         if( dog > max_dog[p] )
         {
            max_dog[p] = dog;
            m_scale_map[p] = i;
         }
      }
      deallocate( sim );

      sim = next_sim;
    }

    blur_gaussian_2d<float,float>( m_scale_map, m_h, m_w, 10.0, filter_size(10), true);

#if defined _OPENMP
#pragma omp parallel for
#endif
    for( int q=0; q<imsz; q++ )
    {
      m_scale_map[q] = round( m_scale_map[q] );
    }

//    save( m_scale_map, m_h, m_w, "scales.dat");

    deallocate( sim );
    deallocate( max_dog );
}

void DAISY_Impl::compute_orientations()
{
    //#####################################################################################
    //# orientation detection is work-in-progress! do not use it if you're not Engin Tola #
    //#####################################################################################

    CV_Assert( m_image != NULL );

    int data_size = m_w*m_h;
    float* rotation_layers = layered_gradient( m_image, m_h, m_w, m_orientation_resolution );

    m_orientation_map = new int[data_size];
    memset( m_orientation_map, 0, sizeof(int)*data_size );

    int ori, max_ind;
    int ind;
    float max_val;

    int next, prev;
    float peak, angle;

    int x, y, kk;

    float* hist=NULL;

    float sigma_inc;
    float sigma_prev = 0;
    float sigma_new;

    for( int scale=0; scale<g_scale_en; scale++ )
    {
      sigma_new  = pow( g_sigma_step, scale  ) * m_rad/3.0;
      sigma_inc  = sqrt( sigma_new*sigma_new - sigma_prev*sigma_prev );
      sigma_prev = sigma_new;

      smooth_layers( rotation_layers, m_h, m_w, m_orientation_resolution, sigma_inc);

      for( y=0; y<m_h; y ++ )
      {
         hist = allocate<float>(m_orientation_resolution);

         for( x=0; x<m_w; x++ )
         {
            ind = y*m_w+x;

            if( m_scale_invariant && m_scale_map[ ind ] != scale ) continue;

            for( ori=0; ori<m_orientation_resolution; ori++ )
            {
               hist[ ori ] = rotation_layers[ori*data_size+ind];
            }

            for( kk=0; kk<6; kk++ )
               smooth_histogram( hist, m_orientation_resolution );

            max_val = -1;
            max_ind =  0;
            for( ori=0; ori<m_orientation_resolution; ori++ )
            {
               if( hist[ori] > max_val )
               {
                  max_val = hist[ori];
                  max_ind = ori;
               }
            }

            prev = max_ind-1;
            if( prev < 0 )
               prev += m_orientation_resolution;

            next = max_ind+1;
            if( next >= m_orientation_resolution )
               next -= m_orientation_resolution;

            peak = interpolate_peak(hist[prev], hist[max_ind], hist[next]);
            angle = (max_ind + peak)*360.0/m_orientation_resolution;

            int iangle = int(angle);

            if( iangle <    0 ) iangle += 360;
            if( iangle >= 360 ) iangle -= 360;


            if( !(iangle >= 0.0 && iangle < 360.0) )
            {
               angle = 0;
            }

            m_orientation_map[ ind ] = iangle;
         }
         deallocate(hist);
      }
    }

    deallocate( rotation_layers );

    compute_oriented_grid_points();
}

void DAISY_Impl::set_descriptor_memory( float* descriptor, long int d_size )
{
    CV_Assert( m_descriptor_memory == false );
    CV_Assert( m_h*m_w != 0 );
    CV_Assert( d_size >= compute_descriptor_memory() );

    m_dense_descriptors = descriptor;
    m_descriptor_memory = true;
}

void DAISY_Impl::set_workspace_memory( float* workspace, long int w_size )
{
    CV_Assert( m_workspace_memory == false );
    CV_Assert( m_h*m_w != 0 );
    CV_Assert( w_size >= compute_workspace_memory() );

    m_smoothed_gradient_layers = workspace;
    m_workspace_memory = true;
}

// -------------------------------------------------
/* DAISY helper routines */

inline void DAISY_Impl::compute_histogram( float* hcube, int y, int x, float* histogram )
{
    if( is_outside(x, 0, m_w-1, y, 0, m_h-1) ) return;

    float* spatial_shift = hcube + y * m_w + x;
    int data_size =  m_w * m_h;

    for( int h=0; h<m_hist_th_q_no; h++ )
      histogram[h] = *(spatial_shift + h*data_size);
}
inline void DAISY_Impl::i_get_histogram( float* histogram, double y, double x, double shift, float* cube )
{
    int ishift=(int)shift;
    double fshift=shift-ishift;
    if     ( fshift < 0.01 ) bi_get_histogram( histogram, y, x, ishift  , cube );
    else if( fshift > 0.99 ) bi_get_histogram( histogram, y, x, ishift+1, cube );
    else                     ti_get_histogram( histogram, y, x,  shift  , cube );
}

inline void DAISY_Impl::bi_get_histogram( float* histogram, double y, double x, int shift, float* hcube )
{
    int mnx = int( x );
    int mny = int( y );

    if( mnx >= m_w-2  || mny >= m_h-2 )
    {
      memset(histogram, 0, sizeof(float)*m_hist_th_q_no);
      return;
    }

    int ind =  mny*m_w+mnx;
    // A C --> pixel positions
    // B D
    float* A = hcube+ind*m_hist_th_q_no;
    float* B = A+m_w*m_hist_th_q_no;
    float* C = A+m_hist_th_q_no;
    float* D = A+(m_w+1)*m_hist_th_q_no;

    double alpha = mnx+1-x;
    double beta  = mny+1-y;

    float w0 = alpha*beta;
    float w1 = beta-w0; // (1-alpha)*beta;
    float w2 = alpha-w0; // (1-beta)*alpha;
    float w3 = 1+w0-alpha-beta; // (1-beta)*(1-alpha);

    int h;

    for( h=0; h<m_hist_th_q_no; h++ ) {
      if( h+shift < m_hist_th_q_no ) histogram[h] = w0*A[h+shift];
      else                           histogram[h] = w0*A[h+shift-m_hist_th_q_no];
    }
    for( h=0; h<m_hist_th_q_no; h++ ) {
      if( h+shift < m_hist_th_q_no ) histogram[h] += w1*C[h+shift];
      else                           histogram[h] += w1*C[h+shift-m_hist_th_q_no];
    }
    for( h=0; h<m_hist_th_q_no; h++ ) {
      if( h+shift < m_hist_th_q_no ) histogram[h] += w2*B[h+shift];
      else                           histogram[h] += w2*B[h+shift-m_hist_th_q_no];
    }
    for( h=0; h<m_hist_th_q_no; h++ ) {
      if( h+shift < m_hist_th_q_no ) histogram[h] += w3*D[h+shift];
      else                           histogram[h] += w3*D[h+shift-m_hist_th_q_no];
    }
}

inline void DAISY_Impl::ti_get_histogram( float* histogram, double y, double x, double shift, float* hcube )
{
    int ishift = int( shift );
    double layer_alpha  = shift - ishift;

    float thist[MAX_CUBE_NO];
    bi_get_histogram( thist, y, x, ishift, hcube );

    for( int h=0; h<m_hist_th_q_no-1; h++ )
      histogram[h] = (1-layer_alpha)*thist[h]+layer_alpha*thist[h+1];
    histogram[m_hist_th_q_no-1] = (1-layer_alpha)*thist[m_hist_th_q_no-1]+layer_alpha*thist[0];
}

inline void DAISY_Impl::ni_get_histogram( float* histogram, int y, int x, int shift, float* hcube )
{
    if( is_outside(x, 0, m_w-1, y, 0, m_h-1) ) return;
    float* hptr = hcube + (y*m_w+x)*m_hist_th_q_no;

    for( int h=0; h<m_hist_th_q_no; h++ )
    {
      int hi = h+shift;
      if( hi >= m_hist_th_q_no ) hi -= m_hist_th_q_no;
      histogram[h] = hptr[hi];
    }
}

inline void DAISY_Impl::get_descriptor( int y, int x, float* &descriptor )
{
    CV_Assert( m_dense_descriptors != NULL );
    CV_Assert( y<m_h && x<m_w && y>=0 && x>=0 );
    descriptor = &(m_dense_descriptors[(y*m_w+x)*m_descriptor_size]);
}

inline void DAISY_Impl::get_descriptor( double y, double x, int orientation, float* descriptor )
{
    get_unnormalized_descriptor(y, x, orientation, descriptor );
    normalize_descriptor(descriptor, m_nrm_type);
}

inline void DAISY_Impl::get_unnormalized_descriptor( double y, double x, int orientation, float* descriptor )
{
    if( m_disable_interpolation ) ni_get_descriptor(y,x,orientation,descriptor);
    else                           i_get_descriptor(y,x,orientation,descriptor);
}

inline void DAISY_Impl::i_get_descriptor( double y, double x, int orientation, float* descriptor )
{
    // memset( descriptor, 0, sizeof(float)*m_descriptor_size );
    //
    // i'm not changing the descriptor[] values if the gridpoint is outside
    // the image. you should memset the descriptor array to 0 if you don't
    // want to have stupid values there.
    //
    CV_Assert( y >= 0 && y < m_h );
    CV_Assert( x >= 0 && x < m_w );
    CV_Assert( orientation >= 0 && orientation < 360 );
    CV_Assert( m_smoothed_gradient_layers );
    CV_Assert( m_oriented_grid_points );
    CV_Assert( descriptor != NULL );

    double shift = m_orientation_shift_table[orientation];

    i_get_histogram( descriptor, y, x, shift, m_smoothed_gradient_layers+g_selected_cubes[0]*m_cube_size );

    int r, rdt, region;
    double yy, xx;
    float* histogram = 0;
    double* grid = m_oriented_grid_points[orientation];

    // petals of the flower
    for( r=0; r<m_rad_q_no; r++ )
    {
      rdt  = r*m_th_q_no+1;
      for( region=rdt; region<rdt+m_th_q_no; region++ )
      {
         yy = y + grid[2*region  ];
         xx = x + grid[2*region+1];
         if( is_outside(xx, 0, m_w-1, yy, 0, m_h-1) ) continue;
         histogram = descriptor+region*m_hist_th_q_no;
         i_get_histogram( histogram, yy, xx, shift, m_smoothed_gradient_layers+g_selected_cubes[r]*m_cube_size );
      }
    }
}

inline void DAISY_Impl::ni_get_descriptor( double y, double x, int orientation, float* descriptor )
{
    // memset( descriptor, 0, sizeof(float)*m_descriptor_size );
    //
    // i'm not changing the descriptor[] values if the gridpoint is outside
    // the image. you should memset the descriptor array to 0 if you don't
    // want to have stupid values there.
    //
    CV_Assert( y >= 0 && y < m_h );
    CV_Assert( x >= 0 && x < m_w );
    CV_Assert( orientation >= 0 && orientation < 360 );
    CV_Assert( m_smoothed_gradient_layers );
    CV_Assert( m_oriented_grid_points );
    CV_Assert( descriptor != NULL );

    double shift = m_orientation_shift_table[orientation];
    int ishift = (int)shift;
    if( shift - ishift > 0.5  ) ishift++;

    int iy = (int)y; if( y - iy > 0.5 ) iy++;
    int ix = (int)x; if( x - ix > 0.5 ) ix++;

    // center
    ni_get_histogram( descriptor, iy, ix, ishift, m_smoothed_gradient_layers+g_selected_cubes[0]*m_cube_size );

    double yy, xx;
    float* histogram=0;
    // petals of the flower
    int r, rdt, region;
    double* grid = m_oriented_grid_points[orientation];
    for( r=0; r<m_rad_q_no; r++ )
    {
      rdt = r*m_th_q_no+1;
      for( region=rdt; region<rdt+m_th_q_no; region++ )
      {
         yy = y + grid[2*region  ];
         xx = x + grid[2*region+1];
         iy = (int)yy; if( yy - iy > 0.5 ) iy++;
         ix = (int)xx; if( xx - ix > 0.5 ) ix++;

         if( is_outside(ix, 0, m_w-1, iy, 0, m_h-1) ) continue;

         histogram = descriptor+region*m_hist_th_q_no;
         ni_get_histogram( histogram, iy, ix, ishift, m_smoothed_gradient_layers+g_selected_cubes[r]*m_cube_size );
      }
    }
}

// Warped get_descriptor's
inline bool DAISY_Impl::get_descriptor( double y, double x, int orientation, double* H, float* descriptor )
{
    bool rval = get_unnormalized_descriptor(y,x,orientation, H, descriptor);
    if( rval ) normalize_descriptor(descriptor, m_nrm_type);
    return rval;
}

inline bool DAISY_Impl::get_unnormalized_descriptor( double y, double x, int orientation, double* H, float* descriptor )
{
    if( m_disable_interpolation ) return ni_get_descriptor(y,x,orientation,H,descriptor);
    else                          return   i_get_descriptor(y,x,orientation,H,descriptor);
}

inline bool DAISY_Impl::i_get_descriptor( double y, double x, int orientation, double* H, float* descriptor )
{
    // memset( descriptor, 0, sizeof(float)*m_descriptor_size );
    //
    // i'm not changing the descriptor[] values if the gridpoint is outside
    // the image. you should memset the descriptor array to 0 if you don't
    // want to have stupid values there.
    //
    CV_Assert( orientation >= 0 && orientation < 360 );
    CV_Assert( m_smoothed_gradient_layers );
    CV_Assert( descriptor != NULL );

    int hradius[MAX_CUBE_NO];

    double hy, hx, ry, rx;
    point_transform_via_homography( H, x, y, hx, hy );
    if( is_outside( hx, 0, m_w, hy, 0, m_h ) ) return false;

    point_transform_via_homography( H, x+m_cube_sigmas[g_selected_cubes[0]], y, rx, ry);
    double radius =  l2norm( ry, rx, hy, hx );
    hradius[0] = quantize_radius( radius );

    double shift = m_orientation_shift_table[orientation];
    i_get_histogram( descriptor, hy, hx, shift, m_smoothed_gradient_layers+hradius[0]*m_cube_size );

    double gy, gx;
    int r, rdt, th, region;
    float* histogram=0;
    for( r=0; r<m_rad_q_no; r++)
    {
      rdt = r*m_th_q_no + 1;
      for( th=0; th<m_th_q_no; th++ )
      {
         region = rdt + th;

         gy = y + m_grid_points[region][0];
         gx = x + m_grid_points[region][1];

         point_transform_via_homography(H, gx, gy, hx, hy);
         if( th == 0 )
         {
            point_transform_via_homography(H, gx+m_cube_sigmas[g_selected_cubes[r]], gy, rx, ry);
            radius = l2norm( ry, rx, hy, hx );
            hradius[r] = quantize_radius( radius );
         }

         if( is_outside(hx, 0, m_w-1, hy, 0, m_h-1) ) continue;

         histogram = descriptor+region*m_hist_th_q_no;
         i_get_histogram( histogram, hy, hx, shift, m_smoothed_gradient_layers+hradius[r]*m_cube_size );
      }
    }
    return true;
}

inline bool DAISY_Impl::ni_get_descriptor( double y, double x, int orientation, double* H, float* descriptor )
{
    // memset( descriptor, 0, sizeof(float)*m_descriptor_size );
    //
    // i'm not changing the descriptor[] values if the gridpoint is outside
    // the image. you should memset the descriptor array to 0 if you don't
    // want to have stupid values there.
    //
    CV_Assert( orientation >= 0 && orientation < 360 );
    CV_Assert( m_smoothed_gradient_layers );
    CV_Assert( descriptor != NULL );

    int hradius[MAX_CUBE_NO];
    double radius;

    double hy, hx, ry, rx;

    point_transform_via_homography(H, x, y, hx, hy );
    if( is_outside( hx, 0, m_w, hy, 0, m_h ) ) return false;

    double shift = m_orientation_shift_table[orientation];
    int  ishift = (int)shift; if( shift - ishift > 0.5  ) ishift++;

    point_transform_via_homography(H, x+m_cube_sigmas[g_selected_cubes[0]], y, rx, ry);
    radius =  l2norm( ry, rx, hy, hx );
    hradius[0] = quantize_radius( radius );

    int ihx = (int)hx; if( hx - ihx > 0.5 ) ihx++;
    int ihy = (int)hy; if( hy - ihy > 0.5 ) ihy++;

    int r, rdt, th, region;
    double gy, gx;
    float* histogram=0;
    ni_get_histogram( descriptor, ihy, ihx, ishift, m_smoothed_gradient_layers+hradius[0]*m_cube_size );
    for( r=0; r<m_rad_q_no; r++)
    {
      rdt = r*m_th_q_no + 1;
      for( th=0; th<m_th_q_no; th++ )
      {
         region = rdt + th;

         gy = y + m_grid_points[region][0];
         gx = x + m_grid_points[region][1];

         point_transform_via_homography(H, gx, gy, hx, hy);
         if( th == 0 )
         {
            point_transform_via_homography(H, gx+m_cube_sigmas[g_selected_cubes[r]], gy, rx, ry);
            radius = l2norm( ry, rx, hy, hx );
            hradius[r] = quantize_radius( radius );
         }

         ihx = (int)hx; if( hx - ihx > 0.5 ) ihx++;
         ihy = (int)hy; if( hy - ihy > 0.5 ) ihy++;

         if( is_outside(ihx, 0, m_w-1, ihy, 0, m_h-1) ) continue;
         histogram = descriptor+region*m_hist_th_q_no;
         ni_get_histogram( histogram, ihy, ihx, ishift, m_smoothed_gradient_layers+hradius[r]*m_cube_size );
      }
    }
    return true;
}

// -------------------------------------------------
/* DAISY interface implementation */

void DAISY_Impl::compute( InputArray _image, std::vector<KeyPoint>& keypoints, OutputArray _descriptors )
{
    // do nothing if no image
    Mat image = _image.getMat();
    if( image.empty() )
        return;

    // get homography if supplied
    Mat H = m_h_matrix.getMat();

    // convert to float if case
    if ( image.depth() != CV_64F )
        H.convertTo( H, CV_64F );
    /*
     * daisy set_image()
     */

    // base size
    m_h = image.rows;
    m_w = image.cols;

    // clone image for conversion
    if ( image.depth() != CV_32F ) {

      Mat work_image = image.clone();

      // convert to gray inplace
      if( work_image.channels() > 1 )
          cvtColor( work_image, work_image, COLOR_BGR2GRAY );

      // convert to float if it is necessary
      if ( work_image.depth() != CV_32F )
      {
          // convert and normalize
          work_image.convertTo( work_image, CV_32F );
          work_image /= 255.0f;
      } else
          CV_Error( Error::StsUnsupportedFormat, "" );

      // use cloned work image
      m_image = work_image.ptr<float>(0);

    } else
      // use original CV_32F image
      m_image = image.ptr<float>(0);

    // full mode if noArray()
    // was passed to _descriptors
    if ( _descriptors.needed() == false )
        m_mode = DAISY::COMP_FULL;

    /*
     * daisy set_parameters()
     */

    m_grid_point_number = m_rad_q_no * m_th_q_no + 1; // +1 is for center pixel
    m_descriptor_size = m_grid_point_number * m_hist_th_q_no;

    for( int i=0; i<360; i++ )
    {
      m_orientation_shift_table[i] = i/360.0 * m_hist_th_q_no;
    }
    m_layer_size = m_h*m_w;
    m_cube_size = m_layer_size*m_hist_th_q_no;

    compute_cube_sigmas();
    compute_grid_points();


    /*
     * daisy initialize_single_descriptor_mode();
     */

    // initializes for get_descriptor(double, double, int) mode: pre-computes
    // convolutions of gradient layers in m_smoothed_gradient_layers

    initialize();
    compute_smoothed_gradient_layers();

    /*
     * daisy compute descriptors given operating mode
     */

    if ( m_mode == COMP_FULL )
    {
        CV_Assert( H.empty() );
        CV_Assert( keypoints.empty() );
        CV_Assert( ! m_use_orientation );

        compute_descriptors();
        normalize_descriptors();

        cv::Mat descriptors;
        descriptors = _descriptors.getMat();
        descriptors = Mat( m_h * m_w, m_descriptor_size,
                           CV_32F, &m_dense_descriptors[0] );
    } else
    if ( m_mode == ONLY_KEYS )
    {
        cv::Mat descriptors;
        _descriptors.create( keypoints.size(), m_descriptor_size, CV_32F );
        descriptors = _descriptors.getMat();

        if ( H.empty() )
          for (size_t k = 0; k < keypoints.size(); k++)
          {
            get_descriptor( keypoints[k].pt.y, keypoints[k].pt.x,
                            m_use_orientation ? keypoints[k].angle : 0,
                            &descriptors.at<float>( k, 0 ) );
          }
        else
          for (size_t k = 0; k < keypoints.size(); k++)
          {
            get_descriptor( keypoints[k].pt.y, keypoints[k].pt.x,
                            m_use_orientation ? keypoints[k].angle : 0,
                            &H.at<double>( 0 ), &descriptors.at<float>( k, 0 ) );
          }

    } else
        CV_Error( Error::StsInternal, "Unknown computation mode" );
}

// constructor
DAISY_Impl::DAISY_Impl( float _radius, int _q_radius, int _q_theta, int _q_hist,
             int _mode, int _norm, InputArray _H, bool _interpolation, bool _use_orientation )
           : m_mode(_mode), m_rad(_radius), m_rad_q_no(_q_radius), m_th_q_no(_q_theta), m_hist_th_q_no(_q_hist),
             m_nrm_type(_norm), m_h_matrix(_H), m_disable_interpolation(_interpolation), m_use_orientation(_use_orientation)
{
    m_w = 0;
    m_h = 0;

    m_image = 0;

    m_grid_point_number = 0;
    m_descriptor_size = 0;

    m_smoothed_gradient_layers = NULL;
    m_dense_descriptors = NULL;
    m_grid_points = NULL;
    m_oriented_grid_points = NULL;

    m_scale_invariant = false;
    m_rotation_invariant = false;

    m_scale_map = NULL;
    m_orientation_map = NULL;
    m_orientation_resolution = 36;
    m_scale_map = NULL;

    m_cube_sigmas = NULL;

    m_descriptor_memory = false;
    m_workspace_memory = false;
    m_descriptor_normalization_threshold = 0.154; // sift magical number

    m_cube_size = 0;
    m_layer_size = 0;

}

// destructor
DAISY_Impl::~DAISY_Impl()
{
    if( !m_workspace_memory ) deallocate( m_smoothed_gradient_layers );
    deallocate( m_grid_points, m_grid_point_number );
    deallocate( m_oriented_grid_points, g_grid_orientation_resolution );
    deallocate( m_orientation_map );
    deallocate( m_scale_map );
    deallocate( m_cube_sigmas );
}

Ptr<DAISY> DAISY::create( float radius, int q_radius, int q_theta, int q_hist,
             int mode, int norm, InputArray H, bool interpolation, bool use_orientation)
{
    return makePtr<DAISY_Impl>(radius, q_radius, q_theta, q_hist, mode, norm, H, interpolation, use_orientation);
}


} // END NAMESPACE XFEATURES2D
} // END NAMESPACE CV
