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
const double g_sigma_step = std::pow(2,1.0/2);
const int g_scale_st = int( (log(g_sigma_1/g_sigma_0)) / log(g_sigma_step) );
static int g_scale_en = 1;

const double g_sigma_init = 1.6;
const static int g_grid_orientation_resolution = 360;

static const int MAX_CUBE_NO = 64;
static const int MAX_NORMALIZATION_ITER = 5;

int g_selected_cubes[MAX_CUBE_NO]; // m_rad_q_no < MAX_CUBE_NO

void DAISY::compute( InputArrayOfArrays images,
                     std::vector<std::vector<KeyPoint> >& keypoints,
                     OutputArrayOfArrays descriptors )
{
    DescriptorExtractor::compute(images, keypoints, descriptors);
}

/*
 !DAISY implementation
 */
class DAISY_Impl CV_FINAL : public DAISY
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

    virtual ~DAISY_Impl() CV_OVERRIDE;

    /** returns the descriptor length in bytes */
    virtual int descriptorSize() const CV_OVERRIDE {
        // +1 is for center pixel
        return ( (m_rad_q_no * m_th_q_no + 1) * m_hist_th_q_no );
    };

    /** returns the descriptor type */
    virtual int descriptorType() const CV_OVERRIDE { return CV_32F; }

    /** returns the default norm type */
    virtual int defaultNorm() const CV_OVERRIDE { return NORM_L2; }

    /**
     * @param image image to extract descriptors
     * @param keypoints of interest within image
     * @param descriptors resulted descriptors array
     */
    virtual void compute( InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors ) CV_OVERRIDE;

    /** @overload
     * @param image image to extract descriptors
     * @param roi region of interest within image
     * @param descriptors resulted descriptors array
     */
    virtual void compute( InputArray image, Rect roi, OutputArray descriptors ) CV_OVERRIDE;

    /** @overload
     * @param image image to extract descriptors
     * @param descriptors resulted descriptors array
     */
    virtual void compute( InputArray image, OutputArray descriptors ) CV_OVERRIDE;

    /**
     * @param y position y on image
     * @param x position x on image
     * @param orientation orientation on image (0->360)
     * @param descriptor supplied array for descriptor storage
     */
    virtual void GetDescriptor( double y, double x, int orientation, float* descriptor ) const CV_OVERRIDE;

    /**
     * @param y position y on image
     * @param x position x on image
     * @param orientation orientation on image (0->360)
     * @param descriptor supplied array for descriptor storage
     * @param H homography matrix for warped grid
     */
    virtual bool GetDescriptor( double y, double x, int orientation, float* descriptor, double* H ) const CV_OVERRIDE;

    /**
     * @param y position y on image
     * @param x position x on image
     * @param orientation orientation on image (0->360)
     * @param descriptor supplied array for descriptor storage
     */
    virtual void GetUnnormalizedDescriptor( double y, double x, int orientation, float* descriptor ) const CV_OVERRIDE;

    /**
     * @param y position y on image
     * @param x position x on image
     * @param orientation orientation on image (0->360)
     * @param descriptor supplied array for descriptor storage
     * @param H homography matrix for warped grid
     */
    virtual bool GetUnnormalizedDescriptor( double y, double x, int orientation, float* descriptor, double* H ) const CV_OVERRIDE;

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

    // the size of the descriptor vector
    int m_descriptor_size;

    // the number of grid locations
    int m_grid_point_number;

    // number of bins in the histograms while computing orientation
    int m_orientation_resolution;


    /*
     * DAISY switches
     */

    // if set to true, descriptors are scale invariant
    bool m_scale_invariant;

    // if set to true, descriptors are rotation invariant
    bool m_rotation_invariant;

    // if enabled, descriptors are computed with casting non-integer locations
    // to integer positions otherwise we use interpolation.
    bool m_enable_interpolation;

    // switch to enable sample by keypoints orientation
    bool m_use_orientation;

    /*
     * DAISY arrays
     */

    // holds optional H matrix
    Mat m_h_matrix;

    // internal float image.
    Mat m_image;

    // image roi
    Rect m_roi;

    // stores the layered gradients in successively smoothed form :
    // layer[n] = m_gradient_layers * gaussian( sigma_n );
    // n>= 1; layer[0] is the layered_gradient
    std::vector<Mat> m_smoothed_gradient_layers;

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
    inline void compute_descriptors( Mat* m_dense_descriptors );

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

    // applies one of the normalizations (partial,full,sift) to the desciptors.
    inline void normalize_descriptors( Mat* m_dense_descriptors );

    inline void update_selected_cubes();

}; // END DAISY_Impl CLASS


// -------------------------------------------------
/* DAISY computation routines */

inline void DAISY_Impl::reset()
{
    m_image.release();

    m_scale_map.release();
    m_orientation_map.release();

    for (size_t i=0; i<m_smoothed_gradient_layers.size(); i++)
      m_smoothed_gradient_layers[i].release();
    m_smoothed_gradient_layers.clear();
}

inline void DAISY_Impl::release_auxiliary()
{
    reset();

    m_cube_sigmas.release();
    m_grid_points.release();
    m_oriented_grid_points.release();
}

static int filter_size( double sigma, double factor )
{
    int fsz = (int)( factor * sigma );
    // kernel size must be odd
    if( fsz%2 == 0 ) fsz++;
    // kernel size cannot be smaller than 3
    if( fsz < 3 ) fsz = 3;

    return fsz;
}

// transform a point via the homography
static void pt_H( double* H, double x, double y, double &u, double &v )
{
    double kxp = H[0]*x + H[1]*y + H[2];
    double kyp = H[3]*x + H[4]*y + H[5];
    double kp  = H[6]*x + H[7]*y + H[8];
    u = kxp / kp; v = kyp / kp;
}


static float interpolate_peak( float left, float center, float right )
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

static void smooth_histogram( Mat* hist, int hsz )
{
    int i;
    float prev, temp;

    prev = hist->at<float>(hsz - 1);
    for (i = 0; i < hsz; i++)
    {
      temp = hist->at<float>(i);
      hist->at<float>(i) = (prev + hist->at<float>(i) + hist->at<float>( (i + 1 == hsz) ? 0 : i + 1) ) / 3.0f;
      prev = temp;
    }
}

struct LayeredGradientInvoker : ParallelLoopBody
{
    LayeredGradientInvoker( Mat* _layers, Mat& _dy, Mat& _dx )
    {
      dy = _dy;
      dx = _dx;
      layers = _layers;
      layer_no = layers->size[0];
    }

    void operator ()(const cv::Range& range) const CV_OVERRIDE
    {
      for (int l = range.start; l < range.end; ++l)
      {
        double angle = l * 2 * (float)CV_PI / layer_no;
        Mat layer( dx.rows, dx.cols, CV_32F, layers->ptr<float>(l,0,0) );
        addWeighted( dx, cos( angle ), dy, sin( angle ), 0.0f, layer, CV_32F );
        max( layer, 0.0f, layer );
      }
    }

    Mat dy, dx;
    Mat *layers;
    int layer_no;
};

static void layered_gradient( Mat& data, Mat* layers )
{
    Mat cvO, dx, dy;
    int layer_no = layers->size[0];

    GaussianBlur( data, cvO, Size(5, 5), 0.5f, 0.5f, BORDER_REPLICATE );
    Sobel( cvO, dx, CV_32F, 1, 0, 1, 0.5f, 0.0f, BORDER_REPLICATE );
    Sobel( cvO, dy, CV_32F, 0, 1, 1, 0.5f, 0.0f, BORDER_REPLICATE );

    parallel_for_( Range(0, layer_no), LayeredGradientInvoker( layers, dy, dx ) );
}

struct SmoothLayersInvoker : ParallelLoopBody
{
    SmoothLayersInvoker( Mat* _layers, const float _sigma )
    {
      layers = _layers;
      sigma = _sigma;

      h = layers->size[1];
      w = layers->size[2];
      ks = filter_size( sigma, 5.0f );
    }

    void operator ()(const cv::Range& range) const CV_OVERRIDE
    {
      for (int l = range.start; l < range.end; ++l)
      {
        Mat layer( h, w, CV_32FC1, layers->ptr<float>(l,0,0) );
        GaussianBlur( layer, layer, Size(ks, ks), sigma, sigma, BORDER_REPLICATE );
      }
    }

    float sigma;
    int ks, h, w;
    Mat *layers;
};

static void smooth_layers( Mat* layers, float sigma )
{
    int layer_no = layers->size[0];
    parallel_for_( Range(0, layer_no), SmoothLayersInvoker( layers, sigma ) );
}

static int quantize_radius( float rad, const int _rad_q_no, const Mat& _cube_sigmas )
{
    if( rad <= _cube_sigmas.at<double>(0) )
        return 0;
    if( rad >= _cube_sigmas.at<double>(_rad_q_no-1) )
        return _rad_q_no-1;

    int idx_min[2];
    minMaxIdx( abs( _cube_sigmas - rad ), NULL, NULL, idx_min );

    return idx_min[1];
}

static void normalize_partial( float* desc, const int _grid_point_number, const int _hist_th_q_no )
{
    for( int h=0; h<_grid_point_number; h++ )
    {
      // l2 norm
      double sum = 0.0f;
      for( int i=0; i<_hist_th_q_no; i++ )
      {
          sum += desc[h*_hist_th_q_no + i]
               * desc[h*_hist_th_q_no + i];
      }

      float norm = (float)sqrt( sum );

      if( norm != 0.0 )
      // divide with norm
      for( int i=0; i<_hist_th_q_no; i++ )
      {
          desc[h*_hist_th_q_no + i] /= norm;
      }
    }
}

static void normalize_sift_way( float* desc, const int _descriptor_size )
{
    int h;
    int iter = 0;
    bool changed = true;
    while( changed && iter < MAX_NORMALIZATION_ITER )
    {
      iter++;
      changed = false;

      double sum = 0.0f;
      for( int i=0; i<_descriptor_size; i++ )
      {
          sum += desc[i] * desc[i];
      }

      float norm = (float)sqrt( sum );

      if( norm > 1e-5 )
      // divide with norm
      for( int i=0; i<_descriptor_size; i++ )
      {
          desc[i] /= norm;
      }

      for( h=0; h<_descriptor_size; h++ )
      {  // sift magical number
         if( desc[ h ] > 0.154f )
         {
            desc[ h ] = 0.154f;
            changed = true;
         }
      }
    }
}

static void normalize_full( float* desc, const int _descriptor_size )
{
    // l2 norm
    double sum = 0.0f;
    for( int i=0; i<_descriptor_size; i++ )
    {
        sum += desc[i] * desc[i];
    }

    float norm = (float)sqrt( sum );

    if( norm != 0.0 )
    // divide with norm
    for( int i=0; i<_descriptor_size; i++ )
    {
        desc[i] /= norm;
    }
}

static void normalize_descriptor( float* desc, const int nrm_type, const int _grid_point_number,
                                  const int _hist_th_q_no, const int _descriptor_size  )
{
    if( nrm_type == DAISY::NRM_NONE ) return;
    else if( nrm_type == DAISY::NRM_PARTIAL ) normalize_partial(desc,_grid_point_number,_hist_th_q_no);
    else if( nrm_type == DAISY::NRM_FULL    ) normalize_full(desc,_descriptor_size);
    else if( nrm_type == DAISY::NRM_SIFT    ) normalize_sift_way(desc,_descriptor_size);
    else
        CV_Error( Error::StsInternal, "No such normalization" );
}

static void ni_get_histogram( float* histogram, const int y, const int x, const int shift, const Mat* hcube )
{

    if ( ! Point( x, y ).inside(
           Rect( 0, 0, hcube->size[1]-1, hcube->size[0]-1 ) )
       ) return;

    int _hist_th_q_no = hcube->size[2];
    const float* hptr = hcube->ptr<float>(y,x,0);
    for( int h=0; h<_hist_th_q_no; h++ )
    {
      int hi = h+shift;
      if( hi >= _hist_th_q_no ) hi -= _hist_th_q_no;
      histogram[h] = hptr[hi];
    }
}

static void bi_get_histogram( float* histogram, const double y, const double x, const int shift, const Mat* hcube )
{
    int mnx = int( x );
    int mny = int( y );
    int _hist_th_q_no = hcube->size[2];
    if( mnx >= hcube->size[1]-2  || mny >= hcube->size[0]-2 )
    {
      memset(histogram, 0, sizeof(float)*_hist_th_q_no);
      return;
    }

    // A C --> pixel positions
    // B D
    const float* A = hcube->ptr<float>( mny   ,  mnx   , 0);
    const float* B = hcube->ptr<float>((mny+1),  mnx   , 0);
    const float* C = hcube->ptr<float>( mny   , (mnx+1), 0);
    const float* D = hcube->ptr<float>((mny+1), (mnx+1), 0);

    double alpha = mnx+1-x;
    double beta  = mny+1-y;

    float w0 = (float) ( alpha * beta );
    float w1 = (float) ( beta - w0    );         // (1-alpha)*beta;
    float w2 = (float) ( alpha - w0   );         // (1-beta)*alpha;
    float w3 = (float) ( 1 + w0 - alpha - beta); // (1-beta)*(1-alpha);

    int h;

    for( h=0; h<_hist_th_q_no; h++ ) {
      if( h+shift < _hist_th_q_no ) histogram[h] = w0 * A[h+shift];
      else                          histogram[h] = w0 * A[h+shift-_hist_th_q_no];
    }
    for( h=0; h<_hist_th_q_no; h++ ) {
      if( h+shift < _hist_th_q_no ) histogram[h] += w1 * C[h+shift];
      else                          histogram[h] += w1 * C[h+shift-_hist_th_q_no];
    }
    for( h=0; h<_hist_th_q_no; h++ ) {
      if( h+shift < _hist_th_q_no ) histogram[h] += w2 * B[h+shift];
      else                          histogram[h] += w2 * B[h+shift-_hist_th_q_no];
    }
    for( h=0; h<_hist_th_q_no; h++ ) {
      if( h+shift < _hist_th_q_no ) histogram[h] += w3 * D[h+shift];
      else                          histogram[h] += w3 * D[h+shift-_hist_th_q_no];
    }
}

static void ti_get_histogram( float* histogram, const double y, const double x, const double shift, const Mat* hcube )
{
    int ishift = int( shift );
    double layer_alpha  = shift - ishift;

    float thist[MAX_CUBE_NO];
    bi_get_histogram( thist, y, x, ishift, hcube );

    int _hist_th_q_no = hcube->size[2];
    for( int h=0; h<_hist_th_q_no-1; h++ )
      histogram[h] = (float) ((1-layer_alpha)*thist[h]+layer_alpha*thist[h+1]);
    histogram[_hist_th_q_no-1] = (float) ((1-layer_alpha)*thist[_hist_th_q_no-1]+layer_alpha*thist[0]);
}

static void i_get_histogram( float* histogram, const double y, const double x, const double shift, const Mat* hcube )
{
    int ishift = (int)shift;
    double fshift = shift-ishift;
    if     ( fshift < 0.01 ) bi_get_histogram( histogram, y, x, ishift  , hcube );
    else if( fshift > 0.99 ) bi_get_histogram( histogram, y, x, ishift+1, hcube );
    else                     ti_get_histogram( histogram, y, x,  shift  , hcube );
}

static void ni_get_descriptor( const double y, const double x, const int orientation, float* descriptor, const std::vector<Mat>* layers,
                               const Mat* _oriented_grid_points, const double* _orientation_shift_table, const int _th_q_no )
{
    CV_Assert( y >= 0 && y < layers->at(0).size[0] );
    CV_Assert( x >= 0 && x < layers->at(0).size[1] );
    CV_Assert( orientation >= 0 && orientation < 360 );
    CV_Assert( !layers->empty() );
    CV_Assert( !_oriented_grid_points->empty() );
    CV_Assert( descriptor != NULL );

    int _rad_q_no = (int) layers->size();
    int _hist_th_q_no = layers->at(0).size[2];
    double shift = _orientation_shift_table[orientation];
    int ishift = (int)shift;
    if( shift - ishift > 0.5  ) ishift++;

    int iy = (int)y; if( y - iy > 0.5 ) iy++;
    int ix = (int)x; if( x - ix > 0.5 ) ix++;

    // center
    ni_get_histogram( descriptor, iy, ix, ishift, &layers->at(g_selected_cubes[0]) );

    double yy, xx;
    float* histogram=0;
    // petals of the flower
    int r, rdt, region;
    Mat grid = _oriented_grid_points->row( orientation );
    for( r=0; r<_rad_q_no; r++ )
    {
      rdt = r*_th_q_no+1;
      for( region=rdt; region<rdt+_th_q_no; region++ )
      {
         yy = y + grid.at<double>(2*region  );
         xx = x + grid.at<double>(2*region+1);
         iy = (int)yy; if( yy - iy > 0.5 ) iy++;
         ix = (int)xx; if( xx - ix > 0.5 ) ix++;

         if ( ! Point2f( (float)xx, (float)yy ).inside(
                Rect( 0, 0, layers->at(0).size[1]-1, layers->at(0).size[0]-1 ) )
            ) continue;

         histogram = descriptor + region*_hist_th_q_no;
         ni_get_histogram( histogram, iy, ix, ishift, &layers->at(g_selected_cubes[r]) );
      }
    }
}

static void i_get_descriptor( const double y, const double x, const int orientation, float* descriptor, const std::vector<Mat>* layers,
                              const Mat* _oriented_grid_points, const double *_orientation_shift_table, const int _th_q_no )
{
    CV_Assert( y >= 0 && y < layers->at(0).size[0] );
    CV_Assert( x >= 0 && x < layers->at(0).size[1] );
    CV_Assert( orientation >= 0 && orientation < 360 );
    CV_Assert( !layers->empty() );
    CV_Assert( !_oriented_grid_points->empty() );
    CV_Assert( descriptor != NULL );

    int _rad_q_no = (int) layers->size();
    int _hist_th_q_no = layers->at(0).size[2];
    double shift = _orientation_shift_table[orientation];

    i_get_histogram( descriptor, y, x, shift, &layers->at(g_selected_cubes[0]) );

    int r, rdt, region;
    double yy, xx;
    float* histogram = 0;

    Mat grid = _oriented_grid_points->row( orientation );

    // petals of the flower
    for( r=0; r<_rad_q_no; r++ )
    {
      rdt  = r*_th_q_no+1;
      for( region=rdt; region<rdt+_th_q_no; region++ )
      {
         yy = y + grid.at<double>(2*region    );
         xx = x + grid.at<double>(2*region + 1);

         if ( ! Point2f( (float)xx, (float)yy ).inside(
                Rect( 0, 0, layers->at(0).size[1]-1, layers->at(0).size[0]-1 ) )
            ) continue;

         histogram = descriptor + region*_hist_th_q_no;
         i_get_histogram( histogram, yy, xx, shift, &layers->at(r) );
      }
    }
}

static bool ni_get_descriptor_h( const double y, const double x, const int orientation, double* H, float* descriptor, const std::vector<Mat>* layers,
                                 const Mat& _cube_sigmas, const Mat* _grid_points, const double* _orientation_shift_table, const int _th_q_no )
{
    CV_Assert( orientation >= 0 && orientation < 360 );
    CV_Assert( !layers->empty() );
    CV_Assert( descriptor != NULL );

    int hradius[MAX_CUBE_NO];

    double hy, hx, ry, rx;

    pt_H(H, x, y, hx, hy );

    if ( ! Point2f( (float)hx, (float)hy ).inside(
                Rect( 0, 0, layers->at(0).size[1]-1, layers->at(0).size[0]-1 ) )
       ) return false;

    int _rad_q_no = (int) layers->size();
    int _hist_th_q_no = layers->at(0).size[2];
    double shift = _orientation_shift_table[orientation];
    int  ishift = (int)shift; if( shift - ishift > 0.5  ) ishift++;

    pt_H(H, x+_cube_sigmas.at<double>(g_selected_cubes[0]), y, rx, ry);
    double d0 = rx - hx; double d1 = ry - hy;
    double radius = sqrt( d0*d0 + d1*d1 );
    hradius[0] = quantize_radius( (float) radius, _rad_q_no, _cube_sigmas );

    int ihx = (int)hx; if( hx - ihx > 0.5 ) ihx++;
    int ihy = (int)hy; if( hy - ihy > 0.5 ) ihy++;

    int r, rdt, th, region;
    double gy, gx;
    float* histogram=0;
    ni_get_histogram( descriptor, ihy, ihx, ishift, &layers->at(hradius[0]) );
    for( r=0; r<_rad_q_no; r++)
    {
      rdt = r*_th_q_no + 1;
      for( th=0; th<_th_q_no; th++ )
      {
         region = rdt + th;

         gy = y + _grid_points->at<double>(region,0);
         gx = x + _grid_points->at<double>(region,1);

         pt_H(H, gx, gy, hx, hy);
         if( th == 0 )
         {
            pt_H(H, gx+_cube_sigmas.at<double>(g_selected_cubes[r]), gy, rx, ry);
            d0 = rx - hx; d1 = ry - hy;
            radius = sqrt( d0*d0 + d1*d1 );
            hradius[r] = quantize_radius( (float) radius, _rad_q_no, _cube_sigmas );
         }

         ihx = (int)hx; if( hx - ihx > 0.5 ) ihx++;
         ihy = (int)hy; if( hy - ihy > 0.5 ) ihy++;

         if ( ! Point( ihx, ihy ).inside(
                Rect( 0, 0, layers->at(0).size[1]-1, layers->at(0).size[0]-1 ) )
            ) continue;

         histogram = descriptor + region*_hist_th_q_no;
         ni_get_histogram( histogram, ihy, ihx, ishift, &layers->at(hradius[r]) );
      }
    }
    return true;
}

static bool i_get_descriptor_h( const double y, const double x, const int orientation, double* H, float* descriptor, const std::vector<Mat>* layers,
                                const Mat _cube_sigmas, const Mat* _grid_points, const double* _orientation_shift_table, const int _th_q_no )
{
    CV_Assert( orientation >= 0 && orientation < 360 );
    CV_Assert( !layers->empty() );
    CV_Assert( descriptor != NULL );

    int hradius[MAX_CUBE_NO];

    double hy, hx, ry, rx;
    pt_H( H, x, y, hx, hy );

    if ( ! Point2f( (float)hx, (float)hy ).inside(
                Rect( 0, 0, layers->at(0).size[1]-1, layers->at(0).size[0]-1 ) )
       ) return false;

    int _rad_q_no = (int) layers->size();
    int _hist_th_q_no = layers->at(0).size[0];
    pt_H( H, x+_cube_sigmas.at<double>(g_selected_cubes[0]), y, rx, ry);
    double d0 = rx - hx; double d1 = ry - hy;
    double radius = sqrt( d0*d0 + d1*d1 );
    hradius[0] = quantize_radius( (float) radius, _rad_q_no, _cube_sigmas );

    double shift = _orientation_shift_table[orientation];
    i_get_histogram( descriptor, hy, hx, shift, &layers->at(hradius[0]) );

    double gy, gx;
    int r, rdt, th, region;
    float* histogram=0;
    for( r=0; r<_rad_q_no; r++)
    {
      rdt = r*_th_q_no + 1;
      for( th=0; th<_th_q_no; th++ )
      {
         region = rdt + th;

         gy = y + _grid_points->at<double>(region,0);
         gx = x + _grid_points->at<double>(region,1);

         pt_H(H, gx, gy, hx, hy);
         if( th == 0 )
         {
            pt_H(H, gx+_cube_sigmas.at<double>(g_selected_cubes[r]), gy, rx, ry);
            d0 = rx - hx; d1 = ry - hy;
            radius = sqrt( d0*d0 + d1*d1 );
            hradius[r] = quantize_radius( (float) radius, _rad_q_no, _cube_sigmas );
         }

         if ( ! Point2f( (float)hx, (float)hy ).inside(
                Rect( 0, 0, layers->at(0).size[1]-1, layers->at(0).size[0]-1 ) )
            ) continue;

         histogram = descriptor + region*_hist_th_q_no;
         i_get_histogram( histogram, hy, hx, shift, &layers->at(hradius[r]) );
      }
    }
    return true;
}

static void get_unnormalized_descriptor( const double y, const double x, const int orientation, float* descriptor,
            const std::vector<Mat>* m_smoothed_gradient_layers, const Mat* m_oriented_grid_points,
            const double* m_orientation_shift_table, const int m_th_q_no, const bool m_enable_interpolation )
{
    if( m_enable_interpolation )
      i_get_descriptor( y, x, orientation, descriptor, m_smoothed_gradient_layers,
                        m_oriented_grid_points, m_orientation_shift_table, m_th_q_no );
    else
     ni_get_descriptor( y, x, orientation, descriptor, m_smoothed_gradient_layers,
                        m_oriented_grid_points, m_orientation_shift_table, m_th_q_no);
}

static void get_descriptor( const double y, const double x, const int orientation, float* descriptor,
            const std::vector<Mat>* m_smoothed_gradient_layers, const Mat* m_oriented_grid_points,
            const double* m_orientation_shift_table, const int m_th_q_no, const int m_hist_th_q_no,
            const int m_grid_point_number, const int m_descriptor_size, const bool m_enable_interpolation,
            const int m_nrm_type )
{
    get_unnormalized_descriptor( y, x, orientation, descriptor, m_smoothed_gradient_layers,
                                 m_oriented_grid_points, m_orientation_shift_table, m_th_q_no, m_enable_interpolation );
    normalize_descriptor( descriptor, m_nrm_type, m_grid_point_number, m_hist_th_q_no, m_descriptor_size );
}

static bool get_unnormalized_descriptor_h( const double y, const double x, const int orientation, float* descriptor, double* H,
            const std::vector<Mat>* m_smoothed_gradient_layers, const Mat& m_cube_sigmas,
            const Mat* m_grid_points, const double* m_orientation_shift_table, const int m_th_q_no, const bool m_enable_interpolation )

{
    if( m_enable_interpolation )
      return  i_get_descriptor_h( y, x, orientation, H, descriptor, m_smoothed_gradient_layers, m_cube_sigmas,
                                  m_grid_points, m_orientation_shift_table, m_th_q_no );
    else
      return ni_get_descriptor_h( y, x, orientation, H, descriptor, m_smoothed_gradient_layers, m_cube_sigmas,
                                  m_grid_points, m_orientation_shift_table, m_th_q_no );
}

static bool get_descriptor_h( const double y, const double x, const int orientation, float* descriptor, double* H,
            const std::vector<Mat>* m_smoothed_gradient_layers, const Mat& m_cube_sigmas,
            const Mat* m_grid_points, const double* m_orientation_shift_table, const int m_th_q_no,
            const int m_hist_th_q_no, const int m_grid_point_number, const int m_descriptor_size,
            const bool m_enable_interpolation, const int m_nrm_type  )

{
    bool rval =
      get_unnormalized_descriptor_h( y, x, orientation, descriptor, H, m_smoothed_gradient_layers, m_cube_sigmas,
                                   m_grid_points, m_orientation_shift_table, m_th_q_no, m_enable_interpolation );

    if( rval )
      normalize_descriptor( descriptor, m_nrm_type, m_grid_point_number, m_hist_th_q_no, m_descriptor_size );

    return rval;
}

void DAISY_Impl::GetDescriptor( double y, double x, int orientation, float* descriptor ) const
{
    get_descriptor( y, x, orientation, descriptor, &m_smoothed_gradient_layers,
                    &m_oriented_grid_points, m_orientation_shift_table, m_th_q_no,
                    m_hist_th_q_no, m_grid_point_number, m_descriptor_size, m_enable_interpolation,
                    m_nrm_type );
}

bool DAISY_Impl::GetDescriptor( double y, double x, int orientation, float* descriptor, double* H ) const
{
  return
  get_descriptor_h( y, x, orientation, descriptor, H, &m_smoothed_gradient_layers,
                    m_cube_sigmas, &m_grid_points, m_orientation_shift_table, m_th_q_no,
                    m_hist_th_q_no, m_grid_point_number, m_descriptor_size, m_enable_interpolation,
                    m_nrm_type );
}

void DAISY_Impl::GetUnnormalizedDescriptor( double y, double x, int orientation, float* descriptor ) const
{
    get_unnormalized_descriptor( y, x, orientation, descriptor, &m_smoothed_gradient_layers,
                                 &m_oriented_grid_points, m_orientation_shift_table, m_th_q_no,
                                 m_enable_interpolation );
}

bool DAISY_Impl::GetUnnormalizedDescriptor( double y, double x, int orientation, float* descriptor, double* H ) const
{
  return
  get_unnormalized_descriptor_h( y, x, orientation, descriptor, H, &m_smoothed_gradient_layers,
                                 m_cube_sigmas, &m_grid_points, m_orientation_shift_table, m_th_q_no,
                                 m_enable_interpolation );
}

inline void DAISY_Impl::compute_grid_points()
{
    double r_step = m_rad / (double)m_rad_q_no;
    double t_step = 2*CV_PI / m_th_q_no;

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

struct ComputeDescriptorsInvoker : ParallelLoopBody
{
    ComputeDescriptorsInvoker( Mat* _descriptors, Mat* _image, Rect* _roi,
                               std::vector<Mat>* _layers, Mat* _orientation_map,
                               Mat* _oriented_grid_points, double* _orientation_shift_table,
                               int _th_q_no, bool _enable_interpolation )
    {
      x_off = _roi->x;
      x_end = _roi->x + _roi->width;
      image = _image;
      layers = _layers;
      th_q_no = _th_q_no;
      descriptors = _descriptors;
      orientation_map = _orientation_map;
      enable_interpolation = _enable_interpolation;
      oriented_grid_points = _oriented_grid_points;
      orientation_shift_table = _orientation_shift_table;
    }

    void operator ()(const cv::Range& range) const CV_OVERRIDE
    {
      int index, orientation;
      for (int y = range.start; y < range.end; ++y)
      {
        for( int x = x_off; x < x_end; x++ )
        {
          index = y*image->cols + x;
          orientation = 0;
          if( !orientation_map->empty() )
              orientation = (int) orientation_map->at<ushort>( y, x );
          if( !( orientation >= 0 && orientation < g_grid_orientation_resolution ) )
              orientation = 0;
          get_unnormalized_descriptor( y, x, orientation, descriptors->ptr<float>( index ),
                                       layers, oriented_grid_points, orientation_shift_table,
                                       th_q_no, enable_interpolation );
        }
      }
    }

    int th_q_no;
    int x_off, x_end;
    std::vector<Mat>* layers;
    Mat *descriptors;
    Mat *orientation_map;
    bool enable_interpolation;
    double* orientation_shift_table;
    Mat *image, *oriented_grid_points;
};

// Computes the descriptor by sampling convoluted orientation maps.
inline void DAISY_Impl::compute_descriptors( Mat* m_dense_descriptors )
{
    int y_off = m_roi.y;
    int y_end = m_roi.y + m_roi.height;

    if( m_scale_invariant    ) compute_scales();
    if( m_rotation_invariant ) compute_orientations();

    m_dense_descriptors->setTo( Scalar(0) );

    parallel_for_( Range(y_off, y_end),
        ComputeDescriptorsInvoker( m_dense_descriptors, &m_image, &m_roi, &m_smoothed_gradient_layers,
                                   &m_orientation_map, &m_oriented_grid_points, m_orientation_shift_table,
                                   m_th_q_no, m_enable_interpolation )
    );

}

struct NormalizeDescriptorsInvoker : ParallelLoopBody
{
    NormalizeDescriptorsInvoker( Mat* _descriptors, int _nrm_type, int _grid_point_number,
                                 int _hist_th_q_no, int _descriptor_size )
    {
      descriptors = _descriptors;
      nrm_type = _nrm_type;
      grid_point_number = _grid_point_number;
      hist_th_q_no = _hist_th_q_no;
      descriptor_size = _descriptor_size;
    }

    void operator ()(const cv::Range& range) const CV_OVERRIDE
    {
      for (int d = range.start; d < range.end; ++d)
      {
        normalize_descriptor( descriptors->ptr<float>(d), nrm_type,
                              grid_point_number, hist_th_q_no, descriptor_size );
      }
    }

    Mat *descriptors;
    int nrm_type;
    int grid_point_number;
    int hist_th_q_no;
    int descriptor_size;
};

inline void DAISY_Impl::normalize_descriptors( Mat* m_dense_descriptors )
{
    CV_Assert( !m_dense_descriptors->empty() );
    int number_of_descriptors =  m_roi.width * m_roi.height;

    parallel_for_( Range(0, number_of_descriptors),
        NormalizeDescriptorsInvoker( m_dense_descriptors, m_nrm_type, m_grid_point_number, m_hist_th_q_no, m_descriptor_size )
    );
}

inline void DAISY_Impl::initialize()
{
    // no image ?
    CV_Assert(m_image.rows != 0);
    CV_Assert(m_image.cols != 0);

    // (m_rad_q_no + 1) cubes
    // 3 dims tensor (idhist, img_y, img_x);
    m_smoothed_gradient_layers.resize( m_rad_q_no + 1 );

    int dims[3] = { m_hist_th_q_no, m_image.rows, m_image.cols };
    for ( int c=0; c<=m_rad_q_no; c++)
      m_smoothed_gradient_layers[c] = Mat( 3, dims, CV_32F );

    layered_gradient( m_image, &m_smoothed_gradient_layers[0] );

    // assuming a 0.5 image smoothness, we pull this to 1.6 as in sift
    smooth_layers( &m_smoothed_gradient_layers[0], (float)sqrt(g_sigma_init*g_sigma_init-0.25f) );

}

inline void DAISY_Impl::compute_cube_sigmas()
{
    if( m_cube_sigmas.empty() )
    {

      m_cube_sigmas = Mat(1, m_rad_q_no, CV_64F);

      double r_step = (double)m_rad / m_rad_q_no / 2;
      for( int r=0; r<m_rad_q_no; r++ )
      {
        m_cube_sigmas.at<double>(r) = (r+1) * r_step;
      }
    }
    update_selected_cubes();
}

inline void DAISY_Impl::update_selected_cubes()
{
    double scale = m_rad/m_rad_q_no/2.0;
    for( int r=0; r<m_rad_q_no; r++ )
    {
      double seed_sigma = ((double)r+1) * scale;
      g_selected_cubes[r] = quantize_radius( (float)seed_sigma, m_rad_q_no, m_cube_sigmas );
    }
}

struct ComputeHistogramsInvoker : ParallelLoopBody
{
    ComputeHistogramsInvoker( std::vector<Mat>* _layers, int _r )
    {
      r = _r;
      layers = _layers;
      _hist_th_q_no = layers->at(r).size[2];
    }

    void operator ()(const cv::Range& range) const CV_OVERRIDE
    {
      for (int y = range.start; y < range.end; ++y)
      {
        for( int x = 0; x < layers->at(r).size[1]; x++ )
        {
          float* hist = layers->at(r).ptr<float>(y,x,0);
          for( int h = 0; h < _hist_th_q_no; h++ )
          {
            hist[h] = layers->at(r+1).at<float>(h,y,x);
          }
        }
      }
    }

    int r, _hist_th_q_no;
    std::vector<Mat> *layers;
};

inline void DAISY_Impl::compute_histograms()
{
    for( int r=0; r<m_rad_q_no; r++ )
    {
      // remap cubes from Mat(h,y,x) -> Mat(y,x,h)
      // final sampling is speeded up by aligned h dim
      int m_h = m_smoothed_gradient_layers.at(r).size[0];
      int m_y = m_smoothed_gradient_layers.at(r).size[1];
      int m_x = m_smoothed_gradient_layers.at(r).size[2];

      // empty targeted cube
      m_smoothed_gradient_layers.at(r).release();

      // recreate cube space
      int dims[3] = { m_y, m_x, m_h };
      m_smoothed_gradient_layers.at(r) = Mat( 3, dims, CV_32F );

      // copy backward all cubes and realign structure
      parallel_for_( Range(0, m_image.rows), ComputeHistogramsInvoker( &m_smoothed_gradient_layers, r ) );
    }
    // trim unused region from collection of cubes
    m_smoothed_gradient_layers[m_rad_q_no].release();
    m_smoothed_gradient_layers.pop_back();
}

inline void DAISY_Impl::compute_smoothed_gradient_layers()
{
    double sigma;
    for( int r=0; r<m_rad_q_no; r++ )
    {
      // incremental smoothing
      if( r == 0 )
        sigma = m_cube_sigmas.at<double>(0);
      else
        sigma = sqrt( m_cube_sigmas.at<double>(r  ) * m_cube_sigmas.at<double>(r  )
                    - m_cube_sigmas.at<double>(r-1) * m_cube_sigmas.at<double>(r-1) );

      int ks = filter_size( sigma, 5.0f );

      for( int th=0; th<m_hist_th_q_no; th++ )
      {
        Mat cvI( m_image.rows, m_image.cols, CV_32F, m_smoothed_gradient_layers[r  ].ptr<float>(th,0,0) );
        Mat cvO( m_image.rows, m_image.cols, CV_32F, m_smoothed_gradient_layers[r+1].ptr<float>(th,0,0) );
        GaussianBlur( cvI, cvO, Size(ks, ks), sigma, sigma, BORDER_REPLICATE );
      }
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

struct MaxDoGInvoker : ParallelLoopBody
{
    MaxDoGInvoker( Mat* _next_sim, Mat* _sim, Mat* _max_dog, Mat* _scale_map, int _i, int _r )
    {
      i = _i;
      r = _r;
      sim = _sim;
      max_dog = _max_dog;
      next_sim = _next_sim;
      scale_map = _scale_map;
    }

    void operator ()(const cv::Range& range) const CV_OVERRIDE
    {
      for (int c = range.start; c < range.end; ++c)
      {
        float dog = (float) fabs( next_sim->at<float>(r,c) - sim->at<float>(r,c) );
        if( dog > max_dog->at<float>(r,c) )
        {
          max_dog->at<float>(r,c) = dog;
          scale_map->at<float>(r,c) = (float) i;
        }
      }
    }
    int i, r;
    Mat* max_dog;
    Mat* scale_map;
    Mat *sim, *next_sim;
};

struct RoundingInvoker : ParallelLoopBody
{
    RoundingInvoker( Mat* _scale_map, int _r )
    {
      r = _r;
      scale_map = _scale_map;
    }

    void operator ()(const cv::Range& range) const CV_OVERRIDE
    {
      for (int c = range.start; c < range.end; ++c)
      {
        scale_map->at<float>(r,c) = (float) cvRound( scale_map->at<float>(r,c) );
      }
    }
    int r;
    Mat* scale_map;
};

inline void DAISY_Impl::compute_scales()
{
    //###############################################################################
    //# scale detection is work-in-progress! do not use it if you're not Engin Tola #
    //###############################################################################

    Mat sim, next_sim;
    float sigma = (float) ( pow( g_sigma_step, g_scale_st)*g_sigma_0 );

    int ks = filter_size( sigma, 3.0f );
    GaussianBlur( m_image, sim, Size(ks, ks), sigma, sigma, BORDER_REPLICATE );

    Mat max_dog( m_image.rows, m_image.cols, CV_32F, Scalar(0) );
    m_scale_map = Mat( m_image.rows, m_image.cols, CV_32F, Scalar(0) );


    float sigma_prev;
    float sigma_new;
    float sigma_inc;

    sigma_prev = (float) g_sigma_0;
    for( int i=0; i<g_scale_en; i++ )
    {
      sigma_new  = (float) ( pow( g_sigma_step, g_scale_st + i  ) * g_sigma_0 );
      sigma_inc  = sqrt( sigma_new*sigma_new - sigma_prev*sigma_prev );
      sigma_prev = sigma_new;

      ks = filter_size( sigma_inc, 3.0f );

      GaussianBlur( sim, next_sim, Size(ks, ks), sigma_inc, sigma_inc, BORDER_REPLICATE );

      for( int r=0; r<m_image.rows; r++ )
      {
        parallel_for_( Range(0, m_image.cols), MaxDoGInvoker( &next_sim, &sim, &max_dog, &m_scale_map, i, r ) );
      }
      sim.release();
      sim = next_sim;
    }

    ks = filter_size( 10.0f, 3.0f );
    GaussianBlur( m_scale_map, m_scale_map, Size(ks, ks), 10.0f, 10.0f, BORDER_REPLICATE );

    for( int r=0; r<m_image.rows; r++ )
    {
      parallel_for_( Range(0, m_image.cols), RoundingInvoker( &m_scale_map, r ) );
    }
}


inline void DAISY_Impl::compute_orientations()
{
    //#####################################################################################
    //# orientation detection is work-in-progress! do not use it if you're not Engin Tola #
    //#####################################################################################

    CV_Assert( !m_image.empty() );

    int dims[4] = { 1, m_orientation_resolution, m_image.rows, m_image.cols };
    Mat rotation_layers(4, dims, CV_32F);
    layered_gradient( m_image, &rotation_layers );

    m_orientation_map = Mat(m_image.rows, m_image.cols, CV_16U, Scalar(0));

    int ori, max_ind;
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

      sigma_new  = (float)( pow( g_sigma_step, scale  ) * m_rad / 3.0f );
      sigma_inc  = sqrt( sigma_new*sigma_new - sigma_prev*sigma_prev );
      sigma_prev = sigma_new;

      smooth_layers( &rotation_layers, sigma_inc );

      for( y=0; y<m_image.rows; y ++ )
      {
         hist = Mat(1, m_orientation_resolution, CV_32F);

         for( x=0; x<m_image.cols; x++ )
         {
            if( m_scale_invariant && m_scale_map.at<float>(y,x) != scale ) continue;

            for (ori = 0; ori < m_orientation_resolution; ori++)
            {
              hist.at<float>(ori) = rotation_layers.at<float>(ori, y, x);
            }
            for( kk=0; kk<6; kk++ )
               smooth_histogram( &hist, m_orientation_resolution );

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
                          &descriptors.at<float>( k, 0 ), &m_smoothed_gradient_layers,
                          &m_oriented_grid_points, m_orientation_shift_table, m_th_q_no,
                          m_hist_th_q_no, m_grid_point_number, m_descriptor_size, m_enable_interpolation,
                          m_nrm_type );
      }
    else
      for (int k = 0; k < (int) keypoints.size(); k++)
      {
        get_descriptor_h( keypoints[k].pt.y, keypoints[k].pt.x,
                          m_use_orientation ? (int) keypoints[k].angle : 0,
                          &descriptors.at<float>( k, 0 ), &H.at<double>( 0 ), &m_smoothed_gradient_layers,
                          m_cube_sigmas, &m_grid_points, m_orientation_shift_table, m_th_q_no,
                          m_hist_th_q_no, m_grid_point_number, m_descriptor_size, m_enable_interpolation,
                          m_nrm_type );
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

    _descriptors.create( m_roi.width*m_roi.height, m_descriptor_size, CV_32F );

    Mat descriptors = _descriptors.getMat();

    // compute full desc
    compute_descriptors( &descriptors );
    normalize_descriptors( &descriptors );
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

    _descriptors.create( m_roi.width*m_roi.height, m_descriptor_size, CV_32F );

    Mat descriptors = _descriptors.getMat();

    // compute full desc
    compute_descriptors( &descriptors );
    normalize_descriptors( &descriptors );
}

// constructor
DAISY_Impl::DAISY_Impl( float _radius, int _q_radius, int _q_theta, int _q_hist,
             int _norm, InputArray _H, bool _interpolation, bool _use_orientation )
           : m_rad(_radius), m_rad_q_no(_q_radius), m_th_q_no(_q_theta), m_hist_th_q_no(_q_hist),
             m_nrm_type(_norm), m_enable_interpolation(_interpolation), m_use_orientation(_use_orientation)
{

    m_descriptor_size = 0;
    m_grid_point_number = 0;

    m_scale_invariant = false;
    m_rotation_invariant = false;
    m_orientation_resolution = 36;

    m_h_matrix = _H.getMat();
}

// destructor
DAISY_Impl::~DAISY_Impl()
{
    release_auxiliary();
}

Ptr<DAISY> DAISY::create( float radius, int q_radius, int q_theta, int q_hist,
             int norm, InputArray H, bool interpolation, bool use_orientation)
{
    return makePtr<DAISY_Impl>(radius, q_radius, q_theta, q_hist, norm, H, interpolation, use_orientation);
}


} // END NAMESPACE XFEATURES2D
} // END NAMESPACE CV
