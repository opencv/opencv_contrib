/*********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2014, 2015
 *
 * Karen Simonyan <karen at robots dot ox dot ac dot uk>
 * Andrea Vedaldi <vedaldi at robots dot ox dot ac dot uk>
 * Andrew Zisserman <az at robots dot ox dot ac dot uk>
 *
 * Visual Geometry Group
 * Department of Engineering Science, University of Oxford
 *
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
 *   * Neither the name of the copyright holders nor the names of its
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

 "Learning Local Feature Descriptors Using Convex Optimisation",
 Simonyan, K. and Vedaldi, A. and Zisserman, A.,
 IEEE Transactions on Pattern Analysis and Machine Intelligence, 2014

 "Discriminative Learning of Local Image Descriptors",
 Matthew A. Brown, Gang Hua, Simon A. J. Winder,
 IEEE Transactions on Pattern Analysis and Machine Intelligence, 2011

 OpenCV port by: Cristian Balint <cristian dot balint at gmail dot com>

 */

#include "precomp.hpp"



using namespace cv;
using namespace std;


namespace cv
{
namespace xfeatures2d
{

/*
 !VGG implementation
 */
class VGG_Impl CV_FINAL : public VGG
{

public:

    // constructor
    explicit VGG_Impl( int desc = VGG::VGG_80, float isigma = 1.4f,
                       bool img_normalize = true, bool use_scale_orientation = true,
                       float scale_factor = 6.25f, bool dsc_normalize = false );

    // destructor
    virtual ~VGG_Impl() CV_OVERRIDE;

    // returns the descriptor length in bytes
    virtual int descriptorSize() const CV_OVERRIDE { return m_descriptor_size; }

    // returns the descriptor type
    virtual int descriptorType() const CV_OVERRIDE { return CV_32F; }

    // returns the default norm type
    virtual int defaultNorm() const CV_OVERRIDE { return NORM_L2; }

    // compute descriptors given keypoints
    virtual void compute( InputArray image, vector<KeyPoint>& keypoints, OutputArray descriptors ) CV_OVERRIDE;

    // getter / setter
    virtual void setSigma(const float isigma) CV_OVERRIDE { m_isigma = isigma; }
    virtual float getSigma() const CV_OVERRIDE { return m_isigma; }

    virtual void setUseNormalizeImage(const bool img_normalize) CV_OVERRIDE { m_img_normalize = img_normalize; }
    virtual bool getUseNormalizeImage() const CV_OVERRIDE { return m_img_normalize; }

    virtual void setUseScaleOrientation(const bool use_scale_orientation) CV_OVERRIDE { m_use_scale_orientation = use_scale_orientation; }
    virtual bool getUseScaleOrientation() const CV_OVERRIDE { return m_use_scale_orientation; }

    virtual void setScaleFactor(const float scale_factor) CV_OVERRIDE { m_scale_factor = scale_factor; }
    virtual float getScaleFactor() const CV_OVERRIDE { return m_scale_factor; }

    virtual void setUseNormalizeDescriptor(const bool dsc_normalize) CV_OVERRIDE { m_dsc_normalize = dsc_normalize; }
    virtual bool getUseNormalizeDescriptor() const CV_OVERRIDE { return m_dsc_normalize; }

protected:

    /*
     * VGG parameters
     */

    int m_descriptor_size;

    // gauss sigma
    float m_isigma;

    // angle bins
    int m_anglebins;

    // sample window
    float m_scale_factor;

    /*
     * VGG switches
     */

    // normalize image
    bool m_img_normalize;

    // switch to enable sample by keypoints orientation
    bool m_use_scale_orientation;

    // normalize desc
    bool m_dsc_normalize;

    /*
     * VGG arrays
     */

    // image
    Mat m_image;

    // pool regions & proj
    Mat m_PRFilters, m_Proj;

private:

    /*
     * VGG functions
     */

     // initialize parameters
     inline void ini_params( const int PRrows, const int PRcols,
                             const unsigned int PRidx[], const unsigned int PRidxSize, const unsigned int PR[],
                             const int PJrows, const int PJcols,
                             const unsigned int PJidx[], const unsigned int PJidxSize, const unsigned int PJ[] );

}; // END VGG_Impl CLASS

// -------------------------------------------------
/* VGG internal routines */

// sample 64x64 patch from image given keypoint
static inline void get_patch( const KeyPoint kp, Mat& Patch, const Mat& image,
                              const bool use_scale_orientation, const float scale_factor )
{
  // scale & radians
  float scale = kp.size / 64.0f * scale_factor;
  const float angle = (kp.angle == -1)
        ? 0 : ( (kp.angle)*(float)CV_PI ) / 180.f;

  // transforms
  const float tsin = sin(angle) * scale;
  const float tcos = cos(angle) * scale;

  const float half_cols = (float)Patch.cols / 2.0f;
  const float half_rows = (float)Patch.rows / 2.0f;

  // sample form original image
  for ( int x = 0; x < Patch.cols; x++ )
  {
    for ( int y = 0; y < Patch.rows; y++ )
    {
      if ( use_scale_orientation )
      {
        const float xoff = x - half_cols;
        const float yoff = y - half_rows;
        // the rotation shifts & scale
        int img_x = int( (kp.pt.x + 0.5f) + xoff*tcos - yoff*tsin );
        int img_y = int( (kp.pt.y + 0.5f) + xoff*tsin + yoff*tcos );
        // sample only within image
        if ( ( img_x < image.cols ) && ( img_x >= 0 )
          && ( img_y < image.rows ) && ( img_y >= 0 ) )
          Patch.at<float>( y, x ) = image.at<float>( img_y, img_x );
        else
          Patch.at<float>( y, x ) = 0.0f;
      }
      else
      {
        const float xoff = x - half_cols;
        const float yoff = y - half_rows;
        // the samples from image
        int img_x = int( kp.pt.x + 0.5f + xoff );
        int img_y = int( kp.pt.y + 0.5f + yoff );
        // sample only within image
        if ( ( img_x < image.cols ) && ( img_x >= 0 )
          && ( img_y < image.rows ) && ( img_y >= 0 ) )
          Patch.at<float>( y, x ) = image.at<float>( img_y, img_x );
        else
          Patch.at<float>( y, x ) = 0.0f;
      }
    }
  }
}

// get descriptor given 64x64 image patch
static void get_desc( const Mat Patch, Mat& PatchTrans, int anglebins, bool img_normalize )
{
    Mat Ix, Iy;
    // % compute gradient
    float kparam[3] = { -1,  0,  1 };
    Mat Kernel( 1, 3, CV_32F, &kparam );
    filter2D( Patch, Ix, CV_32F, Kernel,     Point( -1, -1 ), 0, BORDER_REPLICATE );
    filter2D( Patch, Iy, CV_32F, Kernel.t(), Point( -1, -1 ), 0, BORDER_REPLICATE );

    Mat GMag, GAngle;

    // % gradient magnitude
    // % GMag = sqrt(Ix .^ 2 + Iy .^ 2);
    magnitude( Ix, Iy, GMag );

    // % gradient orientation: [0; 2 * pi]
    // % GAngle = atan2(Iy, Ix) + pi;
    //phase( Ix, Iy, GAngle, false ); //<- opencv is buggy
    GAngle = Mat( GMag.rows, GMag.cols, CV_32F );
    for ( int i = 0; i < (int)GAngle.total(); i++ )
      GAngle.at<float>(i) = atan2( Iy.at<float>(i), Ix.at<float>(i) ) + (float)CV_PI;

    // % soft-assignment of gradients to the orientation histogram
    float AngleStep = 2.0f * (float) CV_PI / (float) anglebins;
    Mat GAngleRatio = GAngle / AngleStep - 0.5f;

    // % Offset1 = mod(GAngleRatio, 1);
    Mat Offset1( GAngleRatio.rows, GAngleRatio.cols, CV_32F );
    for ( int i = 0; i < (int)GAngleRatio.total(); i++ )
      Offset1.at<float>(i) = GAngleRatio.at<float>(i) - floor( GAngleRatio.at<float>(i) );

    Mat w1 = 1.0f - Offset1.t();
    Mat w2 = Offset1.t();

    Mat Bin1( GAngleRatio.rows, GAngleRatio.cols, CV_8U );
    Mat Bin2( GAngleRatio.rows, GAngleRatio.cols, CV_8U );

    // % Bin1 = ceil(GAngleRatio);
    // % Bin1(Bin1 == 0) = Params.nAngleBins;
    for ( int i = 0; i < (int)GAngleRatio.total(); i++ )
    {
      if ( ceil( GAngleRatio.at<float>(i) - 1.0f) == -1.0f )
        Bin1.at<uchar>(i) = (uchar) anglebins - 1;
      else
        Bin1.at<uchar>(i) = (uchar) ceil( GAngleRatio.at<float>(i) - 1.0f );
    }

    // % Bin2 = Bin1 + 1;
    // % Bin2(Bin2 > Params.nAngleBins) = 1;
    for ( int i = 0; i < (int)GAngleRatio.total(); i++ )
    {
      if ( ( Bin1.at<uchar>(i) + 1 ) > anglebins - 1 )
        Bin2.at<uchar>(i) = 0;
      else
        Bin2.at<uchar>(i) = Bin1.at<uchar>(i) + 1;
    }

    // normalize
    if ( img_normalize )
    {
      // % Quantile = 0.8;
      float q = 0.8f;

      // % T = quantile(GMag(:), Quantile);
      Mat GMagSorted;
      sort( GMag.reshape( 0, 1 ), GMagSorted, SORT_ASCENDING );

      int n = GMagSorted.cols;
      // scipy/stats/mstats_basic.py#L1718 mquantiles()
      // m = alphap + p*(1.-alphap-betap)
      // alphap = 0.5 betap = 0.5 => (m = 0.5)
      // aleph = (n*p + m)
      float aleph = ( n * q + 0.5f );
      int k = cvFloor( aleph );
      if ( k >= n - 1 ) k = n - 1;
      if ( k <= 1 ) k = 1;

      float gamma = aleph - k;
      if ( gamma >= 1.0f ) gamma = 1.0f;
      if ( gamma <= 0.0f ) gamma = 0.0f;
      // quantile out from distribution
      float T = ( 1.0f - gamma ) * GMagSorted.at<float>( k - 1 )
              + gamma * GMagSorted.at<float>( k );

      // avoid NaN
      if ( T != 0.0f ) GMag /= ( T / anglebins );
    }

    Mat Bin1T = Bin1.t();
    Mat Bin2T = Bin2.t();
    Mat GMagT = GMag.t();

    // % feature channels
    PatchTrans = Mat( (int)Patch.total(), anglebins, CV_32F, Scalar::all(0) );

    for ( int i = 0; i < anglebins; i++ )
    {
      for ( int p = 0; p < (int)Patch.total(); p++ )
      {
        if ( Bin1T.at<uchar>(p) == i )
          PatchTrans.at<float>(p,i) = w1.at<float>(p) * GMagT.at<float>(p);
        if ( Bin2T.at<uchar>(p) == i )
          PatchTrans.at<float>(p,i) = w2.at<float>(p) * GMagT.at<float>(p);
      }
    }
}

// -------------------------------------------------
/* VGG interface implementation */

struct ComputeVGGInvoker : ParallelLoopBody
{
    ComputeVGGInvoker( const Mat& _image, Mat* _descriptors,
                        const vector<KeyPoint>& _keypoints,
                        const Mat& _PRFilters, const Mat& _Proj,
                        const int _anglebins, const bool _img_normalize,
                        const bool _use_scale_orientation, const float _scale_factor )
    {
      image = _image;
      keypoints = _keypoints;
      descriptors = _descriptors;

      Proj = _Proj;
      PRFilters = _PRFilters;

      anglebins = _anglebins;
      scale_factor = _scale_factor;
      img_normalize = _img_normalize;
      use_scale_orientation  = _use_scale_orientation;
    }

    void operator ()(const cv::Range& range) const CV_OVERRIDE
    {
      Mat Desc, PatchTrans;
      Mat Patch( 64, 64, CV_32F );
      for (int k = range.start; k < range.end; k++)
      {
        // sample patch from image
        get_patch( keypoints[k], Patch, image, use_scale_orientation, scale_factor );
        // compute transform
        get_desc( Patch, PatchTrans, anglebins, img_normalize );
        // pool features
        Desc = PRFilters * PatchTrans;
        // crop
        min( Desc, 1.0f, Desc );
        // reshape
        Desc = Desc.reshape( 1, (int)Desc.total() );
        // project
        descriptors->row( k ) = Desc.t() * Proj.t();
      }
    }

    Mat image;
    Mat *descriptors;
    vector<KeyPoint> keypoints;

    Mat Proj;
    Mat PRFilters;

    int anglebins;
    float scale_factor;
    bool img_normalize;
    bool use_scale_orientation;
};

// descriptor computation using keypoints
void VGG_Impl::compute( InputArray _image, vector<KeyPoint>& keypoints, OutputArray _descriptors )
{
    // do nothing if no image
    if( _image.getMat().empty() )
      return;

    m_image = _image.getMat().clone();

    // Only 8bit images
    CV_Assert( m_image.depth() == CV_8U );

    // convert to gray inplace
    if( m_image.channels() > 1 )
      cvtColor( m_image, m_image, COLOR_BGR2GRAY );

    //convert
    Mat image;
    m_image.convertTo( image, CV_32F );
    m_image = image;
    image.release();

    // smooth whole image
    GaussianBlur( m_image, m_image, Size( 0, 0 ), m_isigma, m_isigma, BORDER_REPLICATE );

    // allocate array
    _descriptors.create( (int) keypoints.size(), m_descriptor_size, CV_32F );

    // prepare descriptors
    Mat descriptors = _descriptors.getMat();
    descriptors.setTo( Scalar(0) );

    parallel_for_( Range( 0, (int) keypoints.size() ),
        ComputeVGGInvoker( m_image, &descriptors, keypoints, m_PRFilters, m_Proj,
                            m_anglebins, m_img_normalize, m_use_scale_orientation,
                            m_scale_factor )
    );

    // normalize desc
    if ( m_dsc_normalize )
    {
      normalize( descriptors, descriptors, 0.0f, 255.0f, NORM_MINMAX, CV_32F );
      descriptors.convertTo( _descriptors, CV_8U );
    }
}

void VGG_Impl::ini_params( const int PRrows, const int PRcols,
                           const unsigned int PRidx[], const unsigned int PRidxSize,
                           const unsigned int PR[],
                           const int PJrows, const int PJcols,
                           const unsigned int PJidx[], const unsigned int PJidxSize,
                           const unsigned int PJ[] )
{
    int idx;

    // initialize pool-region matrix
    m_PRFilters = Mat::zeros( PRrows, PRcols, CV_32F );
    // initialize projection matrix
    m_Proj = Mat::zeros( PJrows, PJcols, CV_32F );

    idx = 0;
    // fill sparse pool-region matrix
    for ( size_t i = 0; i < PRidxSize; i=i+2 )
    {
      for ( size_t k = 0; k < PRidx[i+1]; k++ )
      {
        // expand floats from hex blobs
        m_PRFilters.at<float>( PRidx[i] + (int)k ) = *(float *)&PR[idx];
        idx++;
      }
    }

    idx = 0;
    // fill sparse projection matrix
    for ( size_t i = 0; i < PJidxSize; i=i+2 )
    {
      for ( size_t k = 0; k < PJidx[i+1]; k++ )
      {
        // expand floats from hex blobs
        m_Proj.at<float>( PJidx[i] + (int)k ) = *(float *)&PJ[idx];
        idx++;
      }
    }
}

// constructor
VGG_Impl::VGG_Impl( int _desc, float _isigma, bool _img_normalize,
                    bool _use_scale_orientation, float _scale_factor, bool _dsc_normalize )
             : m_isigma( _isigma ), m_scale_factor( _scale_factor ),
               m_img_normalize( _img_normalize ),
               m_use_scale_orientation( _use_scale_orientation ),
               m_dsc_normalize( _dsc_normalize )
{
    // constant
    m_anglebins = 8;

    // desc type
    switch ( _desc )
    {
      case VGG::VGG_120:
        {
          #include "vgg_generated_120.i"
          ini_params( PRrows, PRcols, PRidx, sizeof(PRidx)/sizeof(PRidx[0]), PR,
                      PJrows, PJcols, PJidx, sizeof(PJidx)/sizeof(PJidx[0]), PJ );
        }
        break;
      case VGG::VGG_80:
        {
          #include "vgg_generated_80.i"
          ini_params( PRrows, PRcols, PRidx, sizeof(PRidx)/sizeof(PRidx[0]), PR,
                      PJrows, PJcols, PJidx, sizeof(PJidx)/sizeof(PJidx[0]), PJ );
        }
        break;
      case VGG::VGG_64:
        {
          #include "vgg_generated_64.i"
          ini_params( PRrows, PRcols, PRidx, sizeof(PRidx)/sizeof(PRidx[0]), PR,
                      PJrows, PJcols, PJidx, sizeof(PJidx)/sizeof(PJidx[0]), PJ );
        }
        break;
      case VGG::VGG_48:
        {

          #include "vgg_generated_48.i"
          ini_params( PRrows, PRcols, PRidx, sizeof(PRidx)/sizeof(PRidx[0]), PR,
                      PJrows, PJcols, PJidx, sizeof(PJidx)/sizeof(PJidx[0]), PJ );
        }
        break;
      default:
        CV_Error( Error::StsInternal, "Unknown Descriptor Type." );
    }

    // set desc size
    m_descriptor_size = m_Proj.rows;
}

// destructor
VGG_Impl::~VGG_Impl()
{
}

Ptr<VGG> VGG::create( int desc, float isigma, bool img_normalize, bool use_scale_orientation,
                      float scale_factor, bool dsc_normalize )
{
    return makePtr<VGG_Impl>( desc, isigma, img_normalize, use_scale_orientation, scale_factor, dsc_normalize );
}


} // END NAMESPACE XFEATURES2D
} // END NAMESPACE CV
