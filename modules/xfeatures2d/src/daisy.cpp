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
     * @param radius radius of the descriptor at the initial scale
     * @param q_radius amount of radial range divisions
     * @param q_theta amount of angular range divisions
     * @param q_hist amount of gradient orientations range divisions
     * @param norm normalization type
     * @param H optional 3x3 homography matrix used to warp the grid of daisy but sampling keypoints remains unwarped on image
     * @param interpolation switch to disable interpolation at minor costs of quality (default is true)
     * @param use_orientation sample patterns using keypoints orientation, disabled by default.
     */
    explicit DAISY_Impl(float radius=15, int q_radius=3, int q_theta=8, int q_hist=8,
                        int norm = DAISY::NRM_NONE, InputArray H = noArray(),
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

    /**
     * @param image image to extract descriptors
     * @param keypoints of interest within image
     * @param descriptors resulted descriptors array
     */
    virtual void compute( InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors );

    /** @overload
     * @param image image to extract descriptors
     * @param roi region of interest within image
     * @param descriptors resulted descriptors array
     */
    virtual void compute( InputArray image, Rect roi, OutputArray descriptors );

    /** @overload
     * @param image image to extract descriptors
     * @param descriptors resulted descriptors array
     */
    virtual void compute( InputArray image, OutputArray descriptors );

    /**
     * @param y position y on image
     * @param x position x on image
     * @param ori orientation on image (0->360)
     * @param descriptor supplied array for descriptor storage
     */
    virtual void get_descriptor( double y, double x, int orientation, float* descriptor ) const;

    /**
     * @param y position y on image
     * @param x position x on image
     * @param ori orientation on image (0->360)
     * @param H homography matrix for warped grid
     * @param descriptor supplied array for descriptor storage
     * @param get_descriptor true if descriptor was computed
     */
    virtual bool get_descriptor( double y, double x, int orientation, double* H, float* descriptor ) const;

    /**
     * @param y position y on image
     * @param x position x on image
     * @param ori orientation on image (0->360)
     * @param descriptor supplied array for descriptor storage
     */
    virtual void get_unnormalized_descriptor( double y, double x, int orientation, float* descriptor ) const;

    /**
     * @param y position y on image
     * @param x position x on image
     * @param ori orientation on image (0->360)
     * @param H homography matrix for warped grid
     * @param descriptor supplied array for descriptor storage
     * @param get_unnormalized_descriptor true if descriptor was computed
     */
    virtual bool get_unnormalized_descriptor( double y, double x, int orientation, double* H, float* descriptor ) const;


protected:

    /*
     * DAISY parameters
     */

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

    // number of bins in the histograms while computing orientation
    int m_orientation_resolution;

    // the number of grid locations
    int m_grid_point_number;

    // the size of the descriptor vector
    int m_descriptor_size;

    // size of m_hsz layers at a single sigma: m_hsz * m_layer_size
    int m_cube_size;

    // size of the layer :
    // m_roi.width*m_roi.height
    int m_layer_size;

    // the clipping threshold to use in normalization: values above this value
    // are clipped to this value for normalize_sift_way() function
    float m_descriptor_normalization_threshold;

    /*
     * DAISY switches
     */

    // if set to true, descriptors are scale invariant
    bool m_scale_invariant;

    // if set to true, descriptors are rotation invariant
    bool m_rotation_invariant;

    // if enabled, descriptors are computed with casting non-integer locations
    // to integer positions otherwise we use interpolation.
    bool m_disable_interpolation;

    // switch to enable sample by keypoints orientation
    bool m_use_orientation;

    /*
     * DAISY arrays
     */

    // holds optional H matrix
    Mat m_h_matrix;

    // input image.
    Mat m_image;

    // image roi
    Rect m_roi;

    // stores the descriptors :
    // its size is [ m_roi.width*m_roi.height*m_descriptor_size ].
    Mat m_dense_descriptors;

    // stores the layered gradients in successively smoothed form :
    // layer[n] = m_gradient_layers * gaussian( sigma_n );
    // n>= 1; layer[0] is the layered_gradient
    Mat m_smoothed_gradient_layers;

    // hold the scales of the pixels
    Mat m_scale_map;

    // holds the orientaitons of the pixels
    Mat m_orientation_map;

    // Holds the oriented coordinates (y,x) of the grid points of the region.
    Mat m_oriented_grid_points;

    // holds the gaussian sigmas for radius quantizations for an incremental
    // application
    Mat m_cube_sigmas;

    // Holds the coordinates (y,x) of the grid points of the region.
    Mat m_grid_points;

    // holds the amount of shift that's required for histogram computation
    double m_orientation_shift_table[360];


private:

    // two possible computational mode
    // ONLY_KEYS -> (mode_1) compute descriptors on demand
    // COMP_FULL -> (mode_2) compute all descriptors from image
    enum { ONLY_KEYS = 0, COMP_FULL = 1 };

    /*
     * DAISY functions
     */

    // initializes the class: computes gradient and structure-points
    inline void initialize();

    // initializes for get_descriptor(double, double, int) mode: pre-computes
    // convolutions of gradient layers in m_smoothed_gradient_layers
    inline void initialize_single_descriptor_mode();

    // set & precompute parameters
    inline void set_parameters();

    // image set image as working
    inline void set_image( InputArray image );

    // releases all the used memory; call this if you want to process
    // multiple images within a loop.
    inline void reset();

    // releases unused memory after descriptor computation is completed.
    inline void release_auxiliary();

    // computes the descriptors for every pixel in the image.
    inline void compute_descriptors();

    // computes scales for every pixel and scales the structure grid so that the
    // resulting descriptors are scale invariant.  you must set
    // m_scale_invariant flag to 1 for the program to call this function
    inline void compute_scales();

    // compute the smoothed gradient layers.
    inline void compute_smoothed_gradient_layers();

    // computes pixel orientations and rotates the structure grid so that
    // resulting descriptors are rotation invariant. If the scales is also
    // detected, then orientations are computed at the computed scales. you must
    // set m_rotation_invariant flag to 1 for the program to call this function
    inline void compute_orientations();

    // computes the histogram at yx; the size of histogram is m_hist_th_q_no
    inline void compute_histogram( float* hcube, int y, int x, float* histogram );

    // reorganizes the cube data so that histograms are sequential in memory.
    inline void compute_histograms();

    // computes the sigma's of layers from descriptor parameters if the user did
    // not sets it. these define the size of the petals of the descriptor.
    inline void compute_cube_sigmas();

    // Computes the locations of the unscaled unrotated points where the
    // histograms are going to be computed according to the given parameters.
    inline void compute_grid_points();

    // Computes the locations of the unscaled rotated points where the
    // histograms are going to be computed according to the given parameters.
    inline void compute_oriented_grid_points();

    // normalizes the descriptor
    inline void normalize_descriptor( float* desc, int nrm_type ) const;

    // applies one of the normalizations (partial,full,sift) to the desciptors.
    inline void normalize_descriptors( int nrm_type = DAISY::NRM_NONE );


    // emulates the way sift is normalized.
    inline void normalize_sift_way( float* desc ) const;

    // normalizes the descriptor histogram by histogram
    inline void normalize_partial( float* desc ) const;

    // normalizes the full descriptor.
    inline void normalize_full( float* desc ) const;

    // normalizes histograms individually
    inline void normalize_histograms();

    inline void update_selected_cubes();

    // Smooth a histogram by using a [1/3 1/3 1/3] kernel.  Assume the histogram
    // is connected in a circular buffer.
    inline void smooth_histogram( Mat hist, int bins );

    // smooths each of the layers by a Gaussian having "sigma" standart
    // deviation.
    inline void smooth_layers( Mat layers, int h, int w, int layer_number, float sigma );

    // returns the descriptor vector for the point (y, x) !!! use this for
    // precomputed operations meaning that you must call compute_descriptors()
    // before calling this function. if you want normalized descriptors, call
    // normalize_descriptors() before calling compute_descriptors()
    inline void get_descriptor( int y, int x, float* &descriptor );

    // does not use interpolation while computing the histogram.
    inline void ni_get_histogram( float* histogram, int y, int x, int shift, float* hcube ) const;

    // returns the interpolated histogram: picks either bi_get_histogram or
    // ti_get_histogram depending on 'shift'
    inline void i_get_histogram( float* histogram, double y, double x, double shift, float* cube ) const;

    // records the histogram that is computed by bilinear interpolation
    // regarding the shift in the spatial coordinates. hcube is the
    // histogram cube for a constant smoothness level.
    inline void bi_get_histogram( float* descriptor, double y, double x, int shift, float* hcube ) const;

    // records the histogram that is computed by trilinear interpolation
    // regarding the shift in layers and spatial coordinates. hcube is the
    // histogram cube for a constant smoothness level.
    inline void ti_get_histogram( float* descriptor, double y, double x, double shift, float* hcube ) const;

    // uses interpolation, for no interpolation call ni_get_descriptor. see also get_descriptor
    inline void i_get_descriptor( double y, double x, int orientation, float* descriptor ) const;

    // does not use interpolation. for w/interpolation, call i_get_descriptor. see also get_descriptor
    inline void ni_get_descriptor( double y, double x, int orientation, float* descriptor ) const;

    // uses interpolation for no interpolation call ni_get_descriptor. see also get_descriptor
    inline bool i_get_descriptor( double y, double x, int orientation, double* H, float* descriptor ) const;

    // does not use interpolation. for w/interpolation, call i_get_descriptor. see also get_descriptor
    inline bool ni_get_descriptor( double y, double x, int orientation, double* H, float* descriptor ) const;

    inline int quantize_radius( float rad ) const;

    inline int filter_size( double sigma );

    // Return a number in the range [-0.5, 0.5] that represents the location of
    // the peak of a parabola passing through the 3 evenly spaced samples.  The
    // center value is assumed to be greater than or equal to the other values
    // if positive, or less than if negative.
    inline float interpolate_peak( float left, float center, float right );


}; // END DAISY_Impl CLASS


// -------------------------------------------------
/* DAISY computation routines */

inline void DAISY_Impl::reset()
{
    m_image.release();

    m_orientation_map.release();
    m_scale_map.release();

    m_dense_descriptors.release();
    m_smoothed_gradient_layers.release();
}

inline void DAISY_Impl::release_auxiliary()
{
    m_orientation_map.release();
    m_scale_map.release();

    m_smoothed_gradient_layers.release();

    m_grid_points.release();
    m_oriented_grid_points.release();
    m_cube_sigmas.release();

    m_image.release();
}

// creates a 1D gaussian filter with N(mean,sigma).
static void gaussian_1d( float* fltr, int fsz, float sigma, float mean )
{
    CV_Assert(fltr != NULL);

    int sz = (fsz - 1) / 2;
    int counter = -1;
    float sum = 0.0f;
    float v = 2 * sigma*sigma;
    for( int x=-sz; x<=sz; x++ )
    {
      counter++;
      fltr[counter] = exp((-((float)x-mean)*((float)x-mean))/v);
      sum += fltr[counter];
    }

    if( sum != 0 )
    for( int x=0; x<fsz; x++ ) fltr[x] /= sum;
}

// computes the gradient of an image
static void gradient( Mat im, int h, int w, Mat dy, Mat dx )
{
    CV_Assert( !dx.empty() );
    CV_Assert( !dy.empty() );

    for( int y=0; y<h; y++ )
    {
      int yw = y*w;
      for( int x=0; x<w; x++ )
      {
        int ind = yw+x;
        // dx
        if( x>0 && x<w-1 ) dx.at<float>(ind) = (im.at<float>(ind+1)-im.at<float>(ind-1)) / 2.0f;
        if( x==0         ) dx.at<float>(ind) =  im.at<float>(ind+1)-im.at<float>(ind  );
        if( x==w-1       ) dx.at<float>(ind) =  im.at<float>(ind  )-im.at<float>(ind-1);

        // dy
        if( y>0 && y<h-1 ) dy.at<float>(ind) = (im.at<float>(ind+w)-im.at<float>(ind-w)) / 2.0f;
        if( y==0         ) dy.at<float>(ind) =  im.at<float>(ind+w)-im.at<float>(ind  );
        if( y==h-1       ) dy.at<float>(ind) =  im.at<float>(ind  )-im.at<float>(ind-w);
      }
    }
}

static Mat layered_gradient( Mat data, int layer_no = 8 )
{
   int data_size = data.rows * data.cols;
   Mat layers( 1, layer_no*data_size, CV_32F, Scalar(0) );

   float kernel[5];
   gaussian_1d(kernel, 5, 0.5f, 0.0f);
   Mat Kernel(1, 5, CV_32F, (float*) kernel);

   Mat cvO;
   // smooth the data matrix
   filter2D( data, cvO, CV_32F, Kernel, Point( -1, -1 ), 0, BORDER_REPLICATE );
   filter2D( cvO, cvO, CV_32F, Kernel.t(), Point( -1, -1 ), 0, BORDER_REPLICATE );

   Mat dx(1, data_size, CV_32F);
   Mat dy(1, data_size, CV_32F);

   gradient(cvO, data.rows, data.cols, dy, dx);
   cvO.release();

#if defined _OPENMP
#pragma omp parallel for
#endif
   for( int l=0; l<layer_no; l++ )
   {
      float angle = (float) (2*l*CV_PI/layer_no);
      float kos = (float) cos( angle );
      float zin = (float) sin( angle );

      float* layer_l = layers.ptr<float>(0) + l*data_size;

      for( int index=0; index<data_size; index++ )
      {
         float value = kos * dx.at<float>(index) + zin * dy.at<float>(index);
         if( value > 0 ) layer_l[index] = value;
         else            layer_l[index] = 0;
      }
   }
   return layers;
}

// data is not destroyed afterwards
static void layered_gradient( Mat data, int layer_no, Mat layers )
{
   CV_Assert( !layers.empty() );

   Mat cvI = data.clone();
   layers.setTo( Scalar(0) );
   int data_size = data.rows * data.cols;

   float kernel[5];
   gaussian_1d(kernel, 5, 0.5f, 0.0f);
   Mat Kernel(1, 5, CV_32F, (float*) kernel);

   filter2D( cvI, cvI, CV_32F, Kernel, Point( -1, -1 ), 0, BORDER_REPLICATE );
   filter2D( cvI, cvI, CV_32F, Kernel.t(), Point( -1, -1 ), 0, BORDER_REPLICATE );

   Mat dx(1, data_size, CV_32F);
   Mat dy(1, data_size, CV_32F);
   gradient( cvI, data.rows, data.cols, dy, dx );

#if defined _OPENMP
#pragma omp parallel for
#endif
   for( int l=0; l<layer_no; l++ )
   {
      float angle = (float) (2*l*CV_PI/layer_no);
      float kos = (float) cos( angle );
      float zin = (float) sin( angle );

      float* layer_l = layers.ptr<float>(0) + l*data_size;

      for( int index=0; index<data_size; index++ )
      {
         float value = kos * dx.at<float>(index) + zin * dy.at<float>(index);
         if( value > 0 ) layer_l[index] = value;
         else            layer_l[index] = 0;
      }
   }
}

// transform a point via the homography
static void point_transform_via_homography( double* H, double x, double y, double &u, double &v )
{
    double kxp = H[0]*x + H[1]*y + H[2];
    double kyp = H[3]*x + H[4]*y + H[5];
    double kp  = H[6]*x + H[7]*y + H[8];
    u = kxp / kp;
    v = kyp / kp;
}

inline void DAISY_Impl::compute_grid_points()
{
    double r_step = m_rad / (double)m_rad_q_no;
    double t_step = 2*CV_PI/ m_th_q_no;

    m_grid_points.release();
    m_grid_points = Mat( m_grid_point_number, 2, CV_64F );

    for( int y=0; y<m_grid_point_number; y++ )
    {
      m_grid_points.at<double>(y,0) = 0;
      m_grid_points.at<double>(y,1) = 0;
    }

    for( int r=0; r<m_rad_q_no; r++ )
    {
      int region = r*m_th_q_no+1;
      for( int t=0; t<m_th_q_no; t++ )
      {
         m_grid_points.at<double>(region+t,0) = (r+1)*r_step * sin( t*t_step );
         m_grid_points.at<double>(region+t,1) = (r+1)*r_step * cos( t*t_step );
      }
    }

    compute_oriented_grid_points();
}

inline void DAISY_Impl::normalize_descriptor( float* desc, int nrm_type = DAISY::NRM_NONE ) const
{
    if( nrm_type == DAISY::NRM_NONE ) nrm_type = m_nrm_type;
    else if( nrm_type == DAISY::NRM_PARTIAL ) normalize_partial(desc);
    else if( nrm_type == DAISY::NRM_FULL    ) normalize_full(desc);
    else if( nrm_type == DAISY::NRM_SIFT    ) normalize_sift_way(desc);
    else
        CV_Error( Error::StsInternal, "No such normalization" );
}

// Computes the descriptor by sampling convoluted orientation maps.
inline void DAISY_Impl::compute_descriptors()
{

    int y_off = m_roi.y;
    int x_off = m_roi.x;
    int y_end = m_roi.y + m_roi.height;
    int x_end = m_roi.x + m_roi.width;

//    if( m_scale_invariant    ) compute_scales();
//    if( m_rotation_invariant ) compute_orientations();

    m_dense_descriptors = Mat( m_roi.width*m_roi.height, m_descriptor_size, CV_32F, Scalar(0) );

    int y, x, index, orientation;
#if defined _OPENMP
#pragma omp parallel for private(y,x,index,orientation)
#endif
    for( y=y_off; y<y_end ; y++ )
    {
      for( x=x_off; x<x_end; x++ )
      {
         index = y*m_image.cols + x;
         orientation = 0;
         if( !m_orientation_map.empty() )
             orientation = (int) m_orientation_map.at<ushort>( x, y );
         if( !( orientation >= 0 && orientation < g_grid_orientation_resolution ) )
             orientation = 0;
         get_unnormalized_descriptor( y, x, orientation, m_dense_descriptors.ptr<float>( index ) );
      }
    }
}

inline void DAISY_Impl::smooth_layers( Mat layers, int h, int w, int layer_number, float sigma )
{

    int i;
    float *layer = NULL;

    int kernel_size = filter_size( sigma );
    std::vector<float> kernel(kernel_size);
    gaussian_1d( &kernel[0], kernel_size, sigma, 0 );

    float* ptr = layers.ptr<float>(0);

#if defined _OPENMP
#pragma omp parallel for private(i, layer)
#endif

    for( i=0; i<layer_number; i++ )
    {
      layer = ptr + i * h*w;

      Mat cvI( h, w, CV_32FC1, (float*) layer );

      Mat Kernel( 1, kernel_size, CV_32FC1, &kernel[0] );
      filter2D( cvI, cvI, CV_32F, Kernel, Point( -1, -1 ), 0, BORDER_REPLICATE );
      filter2D( cvI, cvI, CV_32F, Kernel.t(), Point( -1, -1 ), 0, BORDER_REPLICATE );

    }
}

inline void DAISY_Impl::normalize_partial( float* desc ) const
{
    float norm = 0.0f;
    for( int h=0; h<m_grid_point_number; h++ )
    {
      // l2 norm
      for( int i=0; i<m_hist_th_q_no; i++ )
      {
          norm += sqrt(desc[h*m_hist_th_q_no + i]
                     * desc[h*m_hist_th_q_no + i]);
      }
      if( norm != 0.0 )
      // divide with norm
      for( int i=0; i<m_hist_th_q_no; i++ )
      {
          desc[h*m_hist_th_q_no + i] /= norm;
      }
    }
}

inline void DAISY_Impl::normalize_full( float* desc ) const
{
    // l2 norm
    float norm = 0.0f;
    for( int i=0; i<m_descriptor_size; i++ )
    {
        norm += sqrt(desc[m_descriptor_size + i]
                   * desc[m_descriptor_size + i]);
    }
    if( norm != 0.0 )
    // divide with norm
    for( int i=0; i<m_descriptor_size; i++ )
    {
        desc[m_descriptor_size + i] /= norm;
    }
}

inline void DAISY_Impl::normalize_sift_way( float* desc ) const
{
    int h;
    int iter = 0;
    bool changed = true;
    while( changed && iter < MAX_NORMALIZATION_ITER )
    {
      iter++;
      changed = false;

      float norm = 0.0f;
      for( int i=0; i<m_descriptor_size; i++ )
      {
          norm += sqrt(desc[m_descriptor_size + i]
                     * desc[m_descriptor_size + i]);
      }

      if( norm > 1e-5 )
      // divide with norm
      for( int i=0; i<m_descriptor_size; i++ )
      {
          desc[m_descriptor_size + i] /= norm;
      }

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

inline void DAISY_Impl::normalize_descriptors( int nrm_type )
{
    int d;
    int number_of_descriptors =  m_roi.width * m_roi.height;

#if defined _OPENMP
#pragma omp parallel for private(d)
#endif
    for( d=0; d<number_of_descriptors; d++ )
      normalize_descriptor( m_dense_descriptors.ptr<float>( d ), nrm_type );
}

inline void DAISY_Impl::initialize()
{
    // no image ?
    CV_Assert(m_image.rows != 0);
    CV_Assert(m_image.cols != 0);

    if( m_layer_size == 0 ) {
      m_layer_size = m_image.rows * m_image.cols;
      m_cube_size = m_layer_size * m_hist_th_q_no;
    }

    m_smoothed_gradient_layers = Mat( g_cube_number + 1, m_cube_size, CV_32F);

    layered_gradient( m_image, m_hist_th_q_no, m_smoothed_gradient_layers );

    // assuming a 0.5 image smoothness, we pull this to 1.6 as in sift
    smooth_layers( m_smoothed_gradient_layers, m_image.rows, m_image.cols,
                   m_hist_th_q_no, (float)sqrt(g_sigma_init*g_sigma_init-0.25) );

}

inline void DAISY_Impl::compute_cube_sigmas()
{
    if( m_cube_sigmas.empty() )
    {
      // user didn't set the sigma's;
      // set them from the descriptor parameters
      g_cube_number = m_rad_q_no;

      m_cube_sigmas = Mat(1, g_cube_number, CV_64F);

      double r_step = double(m_rad)/m_rad_q_no;
      for( int r=0; r< m_rad_q_no; r++ )
      {
        m_cube_sigmas.at<double>(r) = (r+1) * r_step/2;
      }
    }
    update_selected_cubes();
}

inline void DAISY_Impl::update_selected_cubes()
{
    for( int r=0; r<m_rad_q_no; r++ )
    {
      double seed_sigma = ((double)r+1)*m_rad/m_rad_q_no/2.0;
       g_selected_cubes[r] = quantize_radius( (float)seed_sigma );
    }
}

inline int DAISY_Impl::quantize_radius( float rad ) const
{
    if( rad <= m_cube_sigmas.at<double>(0) )
        return 0;
    if( rad >= m_cube_sigmas.at<double>(g_cube_number-1) )
        return g_cube_number-1;

    float dist;
    float mindist=FLT_MAX;
    int mini=0;
    for( int c=0; c<g_cube_number; c++ ) {
      dist = (float) fabs( m_cube_sigmas.at<double>(c)-rad );
      if( dist < mindist ) {
         mindist = dist;
         mini=c;
      }
    }
    return mini;
}

inline void DAISY_Impl::compute_histograms()
{
    int r, y, x, ind;
    float* hist=0;

    for( r=0; r<g_cube_number; r++ )
    {

      float* dst = m_smoothed_gradient_layers.ptr<float>(0) +  r   * m_cube_size;
      float* src = m_smoothed_gradient_layers.ptr<float>(0) + (r+1)* m_cube_size;

#if defined _OPENMP
#pragma omp parallel for private(y,x,ind,hist)
#endif
      for( y=0; y<m_image.rows; y++ )
      {
         for( x=0; x<m_image.cols; x++ )
         {
            ind = y*m_image.cols+x;
            hist = dst+ind*m_hist_th_q_no;
            compute_histogram( src, y, x, hist );
         }
      }
    }
}

inline void DAISY_Impl::normalize_histograms()
{
    for( int r=0; r<g_cube_number; r++ )
    {
      float* dst = m_smoothed_gradient_layers.ptr<float>(0) + r*m_cube_size;

#if defined _OPENMP
#pragma omp parallel for
#endif
      for( int y=0; y<m_image.rows; y++ )
      {
          for( int x=0; x<m_image.cols; x++ )
          {
            float* hist = dst + (y*m_image.cols+x) * m_hist_th_q_no;

            float norm = 0.0f;
            for( int i=0; i<m_hist_th_q_no; i++ )
              norm += sqrt( hist[i] * hist[i] );
            if( norm != 0.0 )
            for( int i=0; i<m_hist_th_q_no; i++ )
              hist[i] /= norm;
          }
       }
    }
}

inline void DAISY_Impl::compute_smoothed_gradient_layers()
{

    float* prev_cube = m_smoothed_gradient_layers.ptr<float>(0);
    float* cube = NULL;

    double sigma;
    for( int r=0; r<g_cube_number; r++ )
    {
      cube = m_smoothed_gradient_layers.ptr<float>(0) + (r+1)*m_cube_size;

      // incremental smoothing
      if( r == 0 )
        sigma = m_cube_sigmas.at<double>(0);
      else
        sigma = sqrt( m_cube_sigmas.at<double>(r  ) * m_cube_sigmas.at<double>(r  )
                    - m_cube_sigmas.at<double>(r-1) * m_cube_sigmas.at<double>(r-1) );

      int kernel_size = filter_size( sigma );
      std::vector<float> kernel(kernel_size);
      gaussian_1d(&kernel[0], kernel_size, (float)sigma, 0);

#if defined _OPENMP
#pragma omp parallel for
#endif
      for( int th=0; th<m_hist_th_q_no; th++ )
      {
        Mat cvI( m_image.rows, m_image.cols, CV_32FC1, (float*) prev_cube + th*m_layer_size );
        Mat cvO( m_image.rows, m_image.cols, CV_32FC1, (float*) cube + th*m_layer_size );

        Mat Kernel( 1, kernel_size, CV_32FC1, &kernel[0] );
        filter2D( cvI, cvO, CV_32F, Kernel, Point( -1, -1 ), 0, BORDER_REPLICATE );
        filter2D( cvO, cvO, CV_32F, Kernel.t(), Point( -1, -1 ), 0, BORDER_REPLICATE );
      }
      prev_cube = cube;
    }
    compute_histograms();
}

inline void DAISY_Impl::compute_oriented_grid_points()
{
    m_oriented_grid_points =
        Mat( g_grid_orientation_resolution, m_grid_point_number*2, CV_64F );

    for( int i=0; i<g_grid_orientation_resolution; i++ )
    {
      double angle = -i*2.0*CV_PI/g_grid_orientation_resolution;

      double kos = cos( angle );
      double zin = sin( angle );

      Mat point_list = m_oriented_grid_points.row( i );

      for( int k=0; k<m_grid_point_number; k++ )
      {
         double y = m_grid_points.at<double>(k,0);
         double x = m_grid_points.at<double>(k,1);

         point_list.at<double>(2*k+1) =  x*kos + y*zin; // x
         point_list.at<double>(2*k  ) = -x*zin + y*kos; // y
      }
    }
}

inline void DAISY_Impl::smooth_histogram(Mat hist, int hsz)
{
    int i;
    float prev, temp;

    prev = hist.at<float>(hsz - 1);
    for (i = 0; i < hsz; i++)
    {
      temp = hist.at<float>(i);
      hist.at<float>(i) = (prev + hist.at<float>(i) + hist.at<float>( (i + 1 == hsz) ? 0 : i + 1) ) / 3.0f;
      prev = temp;
    }
}

inline float DAISY_Impl::interpolate_peak(float left, float center, float right)
{
    if( center < 0.0 )
    {
      left = -left;
      center = -center;
      right = -right;
    }
    CV_Assert(center >= left  &&  center >= right);

    float den = (float) (left - 2.0 * center + right);

    if( den == 0 ) return 0;
    else           return (float) (0.5*(left -right)/den);
}

inline int DAISY_Impl::filter_size( double sigma )
{
    int fsz = (int)(5*sigma);

    // kernel size must be odd
    if( fsz%2 == 0 ) fsz++;

    // kernel size cannot be smaller than 3
    if( fsz < 3 ) fsz = 3;

    return fsz;
}

inline void DAISY_Impl::compute_scales()
{
    //###############################################################################
    //# scale detection is work-in-progress! do not use it if you're not Engin Tola #
    //###############################################################################

    int kernel_size = 0;
    float sigma = (float) ( pow( g_sigma_step, g_scale_st)*g_sigma_0 );

    if( kernel_size   == 0 ) kernel_size = (int)(3*sigma);
    if( kernel_size%2 == 0 ) kernel_size++;   // kernel size must be odd
    if( kernel_size    < 3 ) kernel_size = 3; // kernel size cannot be smaller than 3

    std::vector<float> kernel(kernel_size);
    gaussian_1d( &kernel[0], kernel_size, sigma, 0 );
    Mat Kernel( 1, kernel_size, CV_32F, &kernel[0] );

    Mat sim, next_sim;

    // output gaussian image
    filter2D( m_image, sim, CV_32F, Kernel, Point( -1, -1 ), 0, BORDER_REPLICATE );
    filter2D( sim, sim, CV_32F, Kernel.t(), Point( -1, -1 ), 0, BORDER_REPLICATE );

    Mat max_dog( m_image.rows, m_image.cols, CV_32F, Scalar(0) );
    m_scale_map = Mat( m_image.rows, m_image.cols, CV_32F, Scalar(0) );


    int i;
    float sigma_prev;
    float sigma_new;
    float sigma_inc;

    sigma_prev = (float) g_sigma_0;
    for( i=0; i<g_scale_en; i++ )
    {
      sigma_new  = (float) ( pow( g_sigma_step, g_scale_st+i  ) * g_sigma_0 );
      sigma_inc  = sqrt( sigma_new*sigma_new - sigma_prev*sigma_prev );
      sigma_prev = sigma_new;

      kernel_size = filter_size( sigma_inc );
      if( kernel_size   == 0 ) kernel_size = (int)(3 * sigma_inc);
      if( kernel_size%2 == 0 ) kernel_size++;  // kernel size must be odd
      if( kernel_size    < 3 ) kernel_size= 3; // kernel size cannot be smaller than 3

      kernel.resize(kernel_size);
      gaussian_1d( &kernel[0], kernel_size, sigma_inc, 0 );
      Mat NextKernel( 1, kernel_size, CV_32F, &kernel[0] );

      // output gaussian image
      filter2D( sim, next_sim, CV_32F, NextKernel, Point( -1, -1 ), 0, BORDER_REPLICATE );
      filter2D( next_sim, next_sim, CV_32F, NextKernel.t(), Point( -1, -1 ), 0, BORDER_REPLICATE );


#if defined _OPENMP
#pragma omp parallel for
#endif
      for( int r=0; r<m_image.rows; r++ )
      {
        for( int c=0; c<m_image.cols; c++ )
        {
          float dog = (float) fabs( next_sim.at<float>(r,c) - sim.at<float>(r,c) );
          if( dog > max_dog.at<float>(r,c) )
          {
            max_dog.at<float>(r,c) = dog;
            m_scale_map.at<float>(r,c) = (float) i;
          }
        }
      }
      sim.release();
      sim = next_sim;
    }

    kernel_size = filter_size( 10.0f );
    if( kernel_size   == 0 ) kernel_size = (int)(3 * 10.0f);
    if( kernel_size%2 == 0 ) kernel_size++;   // kernel size must be odd
    if( kernel_size    < 3 ) kernel_size = 3; // kernel size cannot be smaller than 3


    kernel.resize(kernel_size);
    gaussian_1d( &kernel[0], kernel_size, 10.0f, 0 );
    Mat FilterKernel( 1, kernel_size, CV_32F, &kernel[0] );

    // output gaussian image
    filter2D( m_scale_map, m_scale_map, CV_32F, FilterKernel, Point( -1, -1 ), 0, BORDER_REPLICATE );
    filter2D( m_scale_map, m_scale_map, CV_32F, FilterKernel.t(), Point( -1, -1 ), 0, BORDER_REPLICATE );

#if defined _OPENMP
#pragma omp parallel for
#endif
      for( int r=0; r<m_image.rows; r++ )
      {
        for( int c=0; c<m_image.cols; c++ )
        {
          m_scale_map.at<float>(r,c) = (float) round( m_scale_map.at<float>(r,c) );
        }
      }
    //save( m_scale_map, m_image.rows, m_image.cols, "scales.dat");
}


inline void DAISY_Impl::compute_orientations()
{
    //#####################################################################################
    //# orientation detection is work-in-progress! do not use it if you're not Engin Tola #
    //#####################################################################################

    CV_Assert( !m_image.empty() );

    int data_size = m_image.cols * m_image.rows;
    Mat rotation_layers = layered_gradient( m_image, m_orientation_resolution );

    m_orientation_map = Mat(m_image.cols, m_image.rows, CV_16U, Scalar(0));

    int ori, max_ind;
    int ind;
    float max_val;

    int next, prev;
    float peak, angle;

    int x, y, kk;

    Mat hist;

    float sigma_inc;
    float sigma_prev = 0.0f;
    float sigma_new;

    for( int scale=0; scale<g_scale_en; scale++ )
    {

      sigma_new  = (float)( pow( g_sigma_step, scale  ) * m_rad/3.0 );
      sigma_inc  = sqrt( sigma_new*sigma_new - sigma_prev*sigma_prev );
      sigma_prev = sigma_new;

      smooth_layers( rotation_layers, m_image.rows, m_image.cols, m_orientation_resolution, sigma_inc );

      for( y=0; y<m_image.rows; y ++ )
      {
         hist = Mat(1, m_orientation_resolution, CV_32F);

         for( x=0; x<m_image.cols; x++ )
         {
            ind = y*m_image.cols+x;

            if( m_scale_invariant && m_scale_map.at<float>(y,x) != scale ) continue;

            for( ori=0; ori<m_orientation_resolution; ori++ )
            {
               hist.at<float>(ori) = rotation_layers.at<float>(ori*data_size+ind);
            }

            for( kk=0; kk<6; kk++ )
               smooth_histogram( hist, m_orientation_resolution );

            max_val = -1;
            max_ind =  0;
            for( ori=0; ori<m_orientation_resolution; ori++ )
            {
               if( hist.at<float>(ori) > max_val )
               {
                  max_val = hist.at<float>(ori);
                  max_ind = ori;
               }
            }

            prev = max_ind-1;
            if( prev < 0 )
               prev += m_orientation_resolution;

            next = max_ind+1;
            if( next >= m_orientation_resolution )
               next -= m_orientation_resolution;

            peak = interpolate_peak(hist.at<float>(prev), hist.at<float>(max_ind), hist.at<float>(next));
            angle = (float)( ((float)max_ind + peak)*360.0/m_orientation_resolution );

            int iangle = int(angle);

            if( iangle <    0 ) iangle += 360;
            if( iangle >= 360 ) iangle -= 360;

            if( !(iangle >= 0.0 && iangle < 360.0) )
            {
               angle = 0;
            }
            m_orientation_map.at<float>(y,x) = (float)iangle;
         }
         hist.release();
      }
    }
    compute_oriented_grid_points();
}

inline void DAISY_Impl::compute_histogram( float* hcube, int y, int x, float* histogram )
{
    if ( ! Point( x, y ).inside(
           Rect( 0, 0, m_image.cols-1, m_image.rows-1 ) )
       ) return;

    float* spatial_shift = hcube + y * m_image.cols + x;
    int data_size =  m_image.cols * m_image.rows;

    for( int h=0; h<m_hist_th_q_no; h++ )
      histogram[h] = *(spatial_shift + h*data_size);
}
inline void DAISY_Impl::i_get_histogram( float* histogram, double y, double x, double shift, float* cube ) const
{
    int ishift=(int)shift;
    double fshift=shift-ishift;
    if     ( fshift < 0.01 ) bi_get_histogram( histogram, y, x, ishift  , cube );
    else if( fshift > 0.99 ) bi_get_histogram( histogram, y, x, ishift+1, cube );
    else                     ti_get_histogram( histogram, y, x,  shift  , cube );
}

inline void DAISY_Impl::bi_get_histogram( float* histogram, double y, double x, int shift, float* hcube ) const
{
    int mnx = int( x );
    int mny = int( y );

    if( mnx >= m_image.cols-2  || mny >= m_image.rows-2 )
    {
      memset(histogram, 0, sizeof(float)*m_hist_th_q_no);
      return;
    }

    int ind =  mny*m_image.cols+mnx;
    // A C --> pixel positions
    // B D
    float* A = hcube+ind*m_hist_th_q_no;
    float* B = A+m_image.cols*m_hist_th_q_no;
    float* C = A+m_hist_th_q_no;
    float* D = A+(m_image.cols+1)*m_hist_th_q_no;

    double alpha = mnx+1-x;
    double beta  = mny+1-y;

    float w0 = (float) (alpha*beta);
    float w1 = (float) (beta-w0); // (1-alpha)*beta;
    float w2 = (float) (alpha-w0); // (1-beta)*alpha;
    float w3 = (float) (1+w0-alpha-beta); // (1-beta)*(1-alpha);

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

inline void DAISY_Impl::ti_get_histogram( float* histogram, double y, double x, double shift, float* hcube ) const
{
    int ishift = int( shift );
    double layer_alpha  = shift - ishift;

    float thist[MAX_CUBE_NO];
    bi_get_histogram( thist, y, x, ishift, hcube );

    for( int h=0; h<m_hist_th_q_no-1; h++ )
      histogram[h] = (float) ((1-layer_alpha)*thist[h]+layer_alpha*thist[h+1]);
    histogram[m_hist_th_q_no-1] = (float) ((1-layer_alpha)*thist[m_hist_th_q_no-1]+layer_alpha*thist[0]);
}

inline void DAISY_Impl::ni_get_histogram( float* histogram, int y, int x, int shift, float* hcube ) const
{
    if ( ! Point( x, y ).inside(
           Rect( 0, 0, m_image.cols-1, m_image.rows-1 ) )
       ) return;

    float* hptr = hcube + (y*m_image.cols+x)*m_hist_th_q_no;

    for( int h=0; h<m_hist_th_q_no; h++ )
    {
      int hi = h+shift;
      if( hi >= m_hist_th_q_no ) hi -= m_hist_th_q_no;
      histogram[h] = hptr[hi];
    }
}

inline void DAISY_Impl::get_descriptor( int y, int x, float* &descriptor )
{
    CV_Assert( !m_dense_descriptors.empty() );
    CV_Assert( y<m_image.rows && x<m_image.cols && y>=0 && x>=0 );
    descriptor = m_dense_descriptors.ptr<float>( y*m_image.cols+x );
}

inline void DAISY_Impl::get_descriptor( double y, double x, int orientation, float* descriptor ) const
{
    get_unnormalized_descriptor(y, x, orientation, descriptor );
    normalize_descriptor(descriptor, m_nrm_type);
}

inline void DAISY_Impl::get_unnormalized_descriptor( double y, double x, int orientation, float* descriptor ) const
{
    if( m_disable_interpolation ) ni_get_descriptor(y,x,orientation,descriptor);
    else                           i_get_descriptor(y,x,orientation,descriptor);
}

inline void DAISY_Impl::i_get_descriptor( double y, double x, int orientation, float* descriptor ) const
{
    // memset( descriptor, 0, sizeof(float)*m_descriptor_size );
    //
    // i'm not changing the descriptor[] values if the gridpoint is outside
    // the image. you should memset the descriptor array to 0 if you don't
    // want to have stupid values there.

    CV_Assert( y >= 0 && y < m_image.rows );
    CV_Assert( x >= 0 && x < m_image.cols );
    CV_Assert( orientation >= 0 && orientation < 360 );
    CV_Assert( !m_smoothed_gradient_layers.empty() );
    CV_Assert( !m_oriented_grid_points.empty() );
    CV_Assert( descriptor != NULL );

    double shift = m_orientation_shift_table[orientation];

    float *ptr = (float *) m_smoothed_gradient_layers.ptr<float>(0);
    i_get_histogram( descriptor, y, x, shift, ptr + g_selected_cubes[0]*m_cube_size );

    int r, rdt, region;
    double yy, xx;
    float* histogram = 0;

    Mat grid = m_oriented_grid_points.row( orientation );

    // petals of the flower
    for( r=0; r<m_rad_q_no; r++ )
    {
      rdt  = r*m_th_q_no+1;
      for( region=rdt; region<rdt+m_th_q_no; region++ )
      {
         yy = y + grid.at<double>(2*region    );
         xx = x + grid.at<double>(2*region + 1);

         if ( ! Point2f( (float)xx, (float)yy ).inside(
                Rect( 0, 0, m_image.cols-1, m_image.rows-1 ) )
            ) continue;

         histogram = descriptor+region*m_hist_th_q_no;
         i_get_histogram( histogram, yy, xx, shift, ptr + g_selected_cubes[r]*m_cube_size );
      }
    }
}

inline void DAISY_Impl::ni_get_descriptor( double y, double x, int orientation, float* descriptor ) const
{
    // memset( descriptor, 0, sizeof(float)*m_descriptor_size );
    //
    // i'm not changing the descriptor[] values if the gridpoint is outside
    // the image. you should memset the descriptor array to 0 if you don't
    // want to have stupid values there.

    CV_Assert( y >= 0 && y < m_image.rows );
    CV_Assert( x >= 0 && x < m_image.cols );
    CV_Assert( orientation >= 0 && orientation < 360 );
    CV_Assert( !m_smoothed_gradient_layers.empty() );
    CV_Assert( !m_oriented_grid_points.empty() );
    CV_Assert( descriptor != NULL );

    double shift = m_orientation_shift_table[orientation];
    int ishift = (int)shift;
    if( shift - ishift > 0.5  ) ishift++;

    int iy = (int)y; if( y - iy > 0.5 ) iy++;
    int ix = (int)x; if( x - ix > 0.5 ) ix++;

    // center
    float *ptr = (float *) m_smoothed_gradient_layers.ptr<float>(0);
    ni_get_histogram( descriptor, iy, ix, ishift, ptr + g_selected_cubes[0]*m_cube_size );

    double yy, xx;
    float* histogram=0;
    // petals of the flower
    int r, rdt, region;
    Mat grid = m_oriented_grid_points.row( orientation );
    for( r=0; r<m_rad_q_no; r++ )
    {
      rdt = r*m_th_q_no+1;
      for( region=rdt; region<rdt+m_th_q_no; region++ )
      {
         yy = y + grid.at<double>(2*region  );
         xx = x + grid.at<double>(2*region+1);
         iy = (int)yy; if( yy - iy > 0.5 ) iy++;
         ix = (int)xx; if( xx - ix > 0.5 ) ix++;

         if ( ! Point2f( (float)xx, (float)yy ).inside(
                Rect( 0, 0, m_image.cols-1, m_image.rows-1 ) )
            ) continue;

         histogram = descriptor+region*m_hist_th_q_no;
         ni_get_histogram( histogram, iy, ix, ishift, ptr + g_selected_cubes[r]*m_cube_size );
      }
    }
}

// Warped get_descriptor's
inline bool DAISY_Impl::get_descriptor( double y, double x, int orientation, double* H, float* descriptor ) const
{
    bool rval = get_unnormalized_descriptor(y,x,orientation, H, descriptor);
    if( rval ) normalize_descriptor(descriptor, m_nrm_type);
    return rval;
}

inline bool DAISY_Impl::get_unnormalized_descriptor( double y, double x, int orientation, double* H, float* descriptor ) const
{
    if( m_disable_interpolation ) return ni_get_descriptor(y,x,orientation,H,descriptor);
    else                          return   i_get_descriptor(y,x,orientation,H,descriptor);
}

inline bool DAISY_Impl::i_get_descriptor( double y, double x, int orientation, double* H, float* descriptor ) const
{
    // memset( descriptor, 0, sizeof(float)*m_descriptor_size );
    //
    // i'm not changing the descriptor[] values if the gridpoint is outside
    // the image. you should memset the descriptor array to 0 if you don't
    // want to have stupid values there.

    CV_Assert( orientation >= 0 && orientation < 360 );
    CV_Assert( !m_smoothed_gradient_layers.empty() );
    CV_Assert( descriptor != NULL );

    int hradius[MAX_CUBE_NO];

    double hy, hx, ry, rx;
    point_transform_via_homography( H, x, y, hx, hy );

    if ( ! Point2f( (float)hx, (float)hy ).inside(
                Rect( 0, 0, m_image.cols-1, m_image.rows-1 ) )
       ) return false;

    point_transform_via_homography( H, x+m_cube_sigmas.at<double>(g_selected_cubes[0]), y, rx, ry);
    double d0 = rx - hx; double d1 = ry - hy;
    double radius = sqrt( d0*d0 + d1*d1 );
    hradius[0] = quantize_radius( (float) radius );

    double shift = m_orientation_shift_table[orientation];
    float *ptr = (float *) m_smoothed_gradient_layers.ptr<float>(0);
    i_get_histogram( descriptor, hy, hx, shift, ptr + hradius[0]*m_cube_size );

    double gy, gx;
    int r, rdt, th, region;
    float* histogram=0;
    for( r=0; r<m_rad_q_no; r++)
    {
      rdt = r*m_th_q_no + 1;
      for( th=0; th<m_th_q_no; th++ )
      {
         region = rdt + th;

         gy = y + m_grid_points.at<double>(region,0);
         gx = x + m_grid_points.at<double>(region,1);

         point_transform_via_homography(H, gx, gy, hx, hy);
         if( th == 0 )
         {
            point_transform_via_homography(H, gx+m_cube_sigmas.at<double>(g_selected_cubes[r]), gy, rx, ry);
            d0 = rx - hx; d1 = ry - hy;
            radius = sqrt( d0*d0 + d1+d1 );
            hradius[r] = quantize_radius( (float) radius );
         }

         if ( ! Point2f( (float)hx, (float)hy ).inside(
                Rect( 0, 0, m_image.cols-1, m_image.rows-1 ) )
            ) continue;

         histogram = descriptor+region*m_hist_th_q_no;
         i_get_histogram( histogram, hy, hx, shift, ptr + hradius[r]*m_cube_size );
      }
    }
    return true;
}

inline bool DAISY_Impl::ni_get_descriptor( double y, double x, int orientation, double* H, float* descriptor ) const
{
    // memset( descriptor, 0, sizeof(float)*m_descriptor_size );
    //
    // i'm not changing the descriptor[] values if the gridpoint is outside
    // the image. you should memset the descriptor array to 0 if you don't
    // want to have stupid values there.

    CV_Assert( orientation >= 0 && orientation < 360 );
    CV_Assert( !m_smoothed_gradient_layers.empty() );
    CV_Assert( descriptor != NULL );

    int hradius[MAX_CUBE_NO];

    double hy, hx, ry, rx;

    point_transform_via_homography(H, x, y, hx, hy );

    if ( ! Point2f( (float)hx, (float)hy ).inside(
                Rect( 0, 0, m_image.cols-1, m_image.rows-1 ) )
       ) return false;

    double shift = m_orientation_shift_table[orientation];
    int  ishift = (int)shift; if( shift - ishift > 0.5  ) ishift++;

    point_transform_via_homography(H, x+m_cube_sigmas.at<double>(g_selected_cubes[0]), y, rx, ry);
    double d0 = rx - hx; double d1 = ry - hy;
    double radius = sqrt( d0*d0 + d1*d1 );
    hradius[0] = quantize_radius( (float) radius );

    int ihx = (int)hx; if( hx - ihx > 0.5 ) ihx++;
    int ihy = (int)hy; if( hy - ihy > 0.5 ) ihy++;

    int r, rdt, th, region;
    double gy, gx;
    float* histogram=0;
    float *ptr = (float *) m_smoothed_gradient_layers.ptr<float>(0);
    ni_get_histogram( descriptor, ihy, ihx, ishift, ptr + hradius[0]*m_cube_size );
    for( r=0; r<m_rad_q_no; r++)
    {
      rdt = r*m_th_q_no + 1;
      for( th=0; th<m_th_q_no; th++ )
      {
         region = rdt + th;

         gy = y + m_grid_points.at<double>(region,0);
         gx = x + m_grid_points.at<double>(region,1);

         point_transform_via_homography(H, gx, gy, hx, hy);
         if( th == 0 )
         {
            point_transform_via_homography(H, gx+m_cube_sigmas.at<double>(g_selected_cubes[r]), gy, rx, ry);
            d0 = rx - hx; d1 = ry - hy;
            radius = sqrt( d0*d0 + d1*d1 );
            hradius[r] = quantize_radius( (float) radius );
         }

         ihx = (int)hx; if( hx - ihx > 0.5 ) ihx++;
         ihy = (int)hy; if( hy - ihy > 0.5 ) ihy++;

         if ( ! Point( ihx, ihy ).inside(
                Rect( 0, 0, m_image.cols-1, m_image.rows-1 ) )
            ) continue;

         histogram = descriptor+region*m_hist_th_q_no;
         ni_get_histogram( histogram, ihy, ihx, ishift, ptr + hradius[r]*m_cube_size );
      }
    }
    return true;
}

inline void DAISY_Impl::initialize_single_descriptor_mode( )
{
    initialize();
    compute_smoothed_gradient_layers();
}

inline void DAISY_Impl::set_parameters( )
{
    m_grid_point_number = m_rad_q_no * m_th_q_no + 1; // +1 is for center pixel
    m_descriptor_size = m_grid_point_number * m_hist_th_q_no;

    for( int i=0; i<360; i++ )
    {
      m_orientation_shift_table[i] = i/360.0 * m_hist_th_q_no;
    }
    m_layer_size = m_image.rows*m_image.cols;
    m_cube_size = m_layer_size*m_hist_th_q_no;

    compute_cube_sigmas();
    compute_grid_points();
}

// set/convert image array for daisy internal routines
// daisy internals use CV_32F image with norm to 1.0f
inline void DAISY_Impl::set_image( InputArray _image )
{
    // release previous image
    // and previous workspace
    reset();
    // fetch new image
    Mat image = _image.getMat();
    // image cannot be empty
    CV_Assert( ! image.empty() );
    // clone image for conversion
    if ( image.depth() != CV_32F ) {

      m_image = image.clone();
      // convert to gray inplace
      if( m_image.channels() > 1 )
          cvtColor( m_image, m_image, COLOR_BGR2GRAY );
      // convert and normalize
      m_image.convertTo( m_image, CV_32F );
      m_image /= 255.0f;
    } else
      // use original user supplied CV_32F image
      // should be a normalized one (cannot check)
      m_image = image;
}


// -------------------------------------------------
/* DAISY interface implementation */

// keypoint scope
void DAISY_Impl::compute( InputArray _image, std::vector<KeyPoint>& keypoints, OutputArray _descriptors )
{
    // do nothing if no image
    if( _image.getMat().empty() )
      return;

    set_image( _image );

    // whole image
    m_roi = Rect( 0, 0, m_image.cols, m_image.rows );

    // get homography
    Mat H = m_h_matrix;

    // convert to double if case
    if ( H.depth() != CV_64F )
        H.convertTo( H, CV_64F );

    set_parameters();

    initialize_single_descriptor_mode();

    // allocate array
    _descriptors.create( (int) keypoints.size(), m_descriptor_size, CV_32F );

    // prepare descriptors
    Mat descriptors = _descriptors.getMat();
    descriptors.setTo( Scalar(0) );

    // iterate over keypoints
    // and fill computed descriptors
    if ( H.empty() )
      for (int k = 0; k < (int) keypoints.size(); k++)
      {
          get_descriptor( keypoints[k].pt.y, keypoints[k].pt.x,
                          m_use_orientation ? (int) keypoints[k].angle : 0,
                          &descriptors.at<float>( k, 0 ) );
      }
    else
      for (int k = 0; k < (int) keypoints.size(); k++)
      {
          get_descriptor( keypoints[k].pt.y, keypoints[k].pt.x,
                          m_use_orientation ? (int) keypoints[k].angle : 0,
                          &H.at<double>( 0 ), &descriptors.at<float>( k, 0 ) );
      }

}

// full scope with roi
void DAISY_Impl::compute( InputArray _image, Rect roi, OutputArray _descriptors )
{
    // do nothing if no image
    if( _image.getMat().empty() )
      return;

    CV_Assert( m_h_matrix.empty() );
    CV_Assert( ! m_use_orientation );

    set_image( _image );

    m_roi = roi;

    set_parameters();
    initialize_single_descriptor_mode();

    // compute full desc
    compute_descriptors();
    normalize_descriptors();

    Mat descriptors = _descriptors.getMat();
    descriptors = m_dense_descriptors;

    release_auxiliary();
}

// full scope
void DAISY_Impl::compute( InputArray _image, OutputArray _descriptors )
{
    // do nothing if no image
    if( _image.getMat().empty() )
      return;

    CV_Assert( m_h_matrix.empty() );
    CV_Assert( ! m_use_orientation );

    set_image( _image );

    // whole image
    m_roi = Rect( 0, 0, m_image.cols, m_image.rows );

    set_parameters();
    initialize_single_descriptor_mode();

    // compute full desc
    compute_descriptors();
    normalize_descriptors();

    Mat descriptors = _descriptors.getMat();
    descriptors = m_dense_descriptors;

    release_auxiliary();
}

// constructor
DAISY_Impl::DAISY_Impl( float _radius, int _q_radius, int _q_theta, int _q_hist,
             int _norm, InputArray _H, bool _interpolation, bool _use_orientation )
           : m_rad(_radius), m_rad_q_no(_q_radius), m_th_q_no(_q_theta), m_hist_th_q_no(_q_hist),
             m_nrm_type(_norm), m_disable_interpolation(_interpolation), m_use_orientation(_use_orientation)
{

    m_image = 0;

    m_descriptor_size = 0;
    m_grid_point_number = 0;

    m_grid_points.release();
    m_dense_descriptors.release();
    m_smoothed_gradient_layers.release();
    m_oriented_grid_points.release();

    m_scale_invariant = false;
    m_rotation_invariant = false;

    m_scale_map.release();
    m_orientation_map.release();
    m_orientation_resolution = 36;

    m_cube_sigmas.release();

    m_cube_size = 0;
    m_layer_size = 0;

    m_descriptor_normalization_threshold = 0.154f; // sift magical number

    m_h_matrix = _H.getMat();
}

// destructor
DAISY_Impl::~DAISY_Impl()
{
    m_scale_map.release();
    m_grid_points.release();
    m_orientation_map.release();
    m_oriented_grid_points.release();
    m_smoothed_gradient_layers.release();
}

Ptr<DAISY> DAISY::create( float radius, int q_radius, int q_theta, int q_hist,
             int norm, InputArray H, bool interpolation, bool use_orientation)
{
    return makePtr<DAISY_Impl>(radius, q_radius, q_theta, q_hist, norm, H, interpolation, use_orientation);
}


} // END NAMESPACE XFEATURES2D
} // END NAMESPACE CV
