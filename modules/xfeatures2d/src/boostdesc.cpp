/*********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2013, 2016
 *
 * Tomasz Trzcinski <t dot trzcinski at ii dot pw dot edu dot pl>
 * Mario Christoudias <mariochristoudias at gmail dot com>
 * Vincent Lepetit <lepetit at icg dot tugraz dot at>
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

 "Learning Image Descriptors with Boosting"
 T. Trzcinski, M. Christoudias and V. Lepetit
 IEEE Transactions on Pattern Analysis and Machine Intelligence, 2013

 "Boosting Binary Keypoint Descriptors"
 T. Trzcinski, M. Christoudias, V. Lepetit and P. Fua
 Computer Vision and Pattern Recognition (CVPR), 2013

 Original code: Tomasz Trzcinski <t dot trzcinski at ii dot pw dot edu dot pl>
 OpenCV port by: Cristian Balint <cristian dot balint at gmail dot com>

 */

#include <bitset>
#include "precomp.hpp"



using namespace cv;
using namespace std;


namespace cv
{
namespace xfeatures2d
{

#ifdef OPENCV_XFEATURES2D_HAS_BOOST_DATA

/*
 !BoostDesc implementation
 */
class BoostDesc_Impl CV_FINAL : public BoostDesc
{

public:

    // constructor
    explicit BoostDesc_Impl( int desc = BINBOOST_256,
                             bool use_scale_orientation = true,
                             float scale_factor = 6.25f );

    // destructor
    virtual ~BoostDesc_Impl() CV_OVERRIDE;

    // returns the descriptor length in bytes
    virtual int descriptorSize() const CV_OVERRIDE { return m_descriptor_size; }

    // returns the descriptor type
    virtual int descriptorType() const CV_OVERRIDE { return m_descriptor_type; }

    // returns the default norm type
    virtual int defaultNorm()    const CV_OVERRIDE { return m_descriptor_norm; }

    // compute descriptors given keypoints
    virtual void compute( InputArray image, vector<KeyPoint>& keypoints, OutputArray descriptors ) CV_OVERRIDE;

    // getter / setter
    virtual void setUseScaleOrientation(const bool use_scale_orientation) CV_OVERRIDE { m_use_scale_orientation = use_scale_orientation; }
    virtual bool getUseScaleOrientation() const CV_OVERRIDE { return m_use_scale_orientation; }

    virtual void setScaleFactor(const float scale_factor) CV_OVERRIDE { m_scale_factor = scale_factor; }
    virtual float getScaleFactor() const CV_OVERRIDE { return m_scale_factor; }

protected:

    /*
     * BoostDesc parameters
     */

    // size, type, norm
    int m_descriptor_size;
    int m_descriptor_type;
    int m_descriptor_norm;

    // desc type
    int m_desc_type;

    // gradient
    // assignment types
    enum Assign
    {
      ASSIGN_HARD      = 0,
      ASSIGN_BILINEAR  = 1,
      ASSIGN_SOFT      = 2,
      ASSIGN_HARD_MAGN = 3,
      ASSIGN_SOFT_MAGN = 4
    };

    // dims
    int m_Dims;

    // no. weak
    // learners
    int m_nWLs;

    // gradient type
    int m_grad_atype;

    // patch size
    int m_patch_size;

    // orient quantity
    int m_orient_q;

    // patch scale factor
    float m_scale_factor;

    /*
     * BoostDesc switches
     */

    // switch to enable sample by keypoints orientation
    bool m_use_scale_orientation;


    /*
     * BoostDesc arrays
     */

    // image
    Mat m_image;

    // parameters
    // weak learner
    Mat m_wl_thresh;
    Mat m_wl_orient;
    Mat m_wl_x_min, m_wl_x_max;
    Mat m_wl_y_min, m_wl_y_max;
    Mat m_wl_alpha, m_wl_beta;

private:

    /*
     * BoostDesc functions
     */

     // initialize parameters
     inline void ini_params( const int orientQuant, const int patchSize,
                             const int iGradAssignType,
                             const int nDim, const int nWLs,
                             const unsigned int thresh[], const int orient[],
                             const int x_min[], const int x_max[],
                             const int y_min[], const int y_max[],
                             const unsigned int alpha[],  const unsigned int beta[] );

}; // END BoostDesc_Impl CLASS

// -------------------------------------------------
/* BoostDesc internal routines */

static void computeGradientMaps( const Mat& im,
                                 const int gradAssignType,
                                 const int orientQuant,
                                 vector<Mat>& gradMap )
{
    enum Assign
    {
      ASSIGN_HARD      = 0,
      ASSIGN_BILINEAR  = 1,
      ASSIGN_SOFT      = 2,
      ASSIGN_HARD_MAGN = 3,
      ASSIGN_SOFT_MAGN = 4
    };

    Mat derivx( im.size(), CV_32FC1 );
    Mat derivy( im.size(), CV_32FC1 );

    Sobel( im, derivx, derivx.depth(), 1, 0 );
    Sobel( im, derivy, derivy.depth(), 0, 1 );

    for ( int i = 0; i < orientQuant; i++ )
      gradMap.push_back( Mat::zeros( im.size(), CV_8UC1 ) );

    int index, index2;
    double binCenter, weight;
    double binSize = (2 * CV_PI) / orientQuant;

    // fill in temp matrices with
    // respones to edge detection
    const float* pDerivx = derivx.ptr<float>();
    const float* pDerivy = derivy.ptr<float>();

    for ( int i = 0; i < im.rows; i++ )
    {
      for ( int j = 0; j < im.cols; j++ )
      {
        float gradMagnitude = sqrt( (*pDerivx) * (*pDerivx)
                                  + (*pDerivy) * (*pDerivy) );
        if ( gradMagnitude > 20 )
        {
          double theta = atan2( *pDerivy, *pDerivx );
          theta = ( theta < 0 ) ? theta + 2*CV_PI : theta;
          index = int( theta / binSize );
          index = ( index == orientQuant ) ? 0 : index;

          switch ( gradAssignType )
          {
            case ASSIGN_HARD:
              gradMap[index].at<uchar>(i,j) = 1;
              break;

            case ASSIGN_HARD_MAGN:
              gradMap[index].at<uchar>(i,j) = (uchar) cvRound( gradMagnitude );
              break;

            case ASSIGN_BILINEAR:
              index2 = (int) ceil( theta / binSize );
              index2 = ( index2 == orientQuant ) ? 0 : index2;
              binCenter  = ( index + 0.5f ) * binSize;
              weight = 1 - abs( theta - binCenter ) / binSize;
              gradMap[index ].at<uchar>(i,j) = (uchar) cvRound( 255 * weight );
              gradMap[index2].at<uchar>(i,j) = (uchar) cvRound( 255 * ( 1 - weight ) );
              break;

            case ASSIGN_SOFT:
              for ( int binNum = 0; binNum < orientQuant/2 + 1; binNum++ )
              {
                index2 = ( binNum + index + orientQuant - orientQuant/4 ) % orientQuant;
                binCenter = ( index2 + 0.5f ) * binSize;
                weight = cos( theta - binCenter );
                weight = ( weight < 0 ) ? 0 : weight;
                gradMap[index2].at<uchar>(i,j) = (uchar) cvRound( 255 * weight );
              }
              break;

            case ASSIGN_SOFT_MAGN:
              for ( int binNum = 0; binNum < orientQuant/2 + 1; binNum++ )
              {
                index2 = ( binNum + index + orientQuant - orientQuant/4 ) % orientQuant;
                binCenter = ( index2 + 0.5f ) * binSize;
                weight = cos( theta - binCenter );
                weight = ( weight < 0 ) ? 0 : weight;
                gradMap[index2].at<uchar>(i,j) = (uchar) cvRound( gradMagnitude * weight );
              }
              break;
          } // end switch
        }
        ++pDerivy;
        ++pDerivx;
      }
    }
}

static void computeIntegrals( const vector<Mat>& gradMap,
                              const int orientQuant,
                              vector<Mat>& integralMap )
{
    // init integral images
    int rows = gradMap[0].rows;
    int cols = gradMap[0].cols;

    for ( int i = 0; i < orientQuant+1; i++ )
      integralMap.push_back( Mat::zeros( rows+1, cols+1, CV_8UC1 ) );

    // generate corresponding integral images
    for( int i = 0; i < orientQuant; i++ )
      integral( gradMap[i], integralMap[i] );

    // copy the values from the first quantization bin
    integralMap[0].copyTo( integralMap[orientQuant] );

    int* ptrSum, *ptr;
    for ( int k = 1; k < orientQuant; k++ )
    {
      ptr    = (int*) integralMap[k].ptr<int>();
      ptrSum = (int*) integralMap[orientQuant].ptr<int>();
      for (int i=0; i<(rows+1)*(cols+1); ++i)
      {
        *ptrSum += *ptr;
        ++ptrSum;
        ++ptr;
      }
    }
}

static float computeWLResponse( const int x_min,  const int x_max,
                                const int y_min,  const int y_max,
                                const int orient, const float thresh,
                                const int orientQuant,
                                const vector<Mat>& integralMap )
{
    const int width = integralMap[0].cols;

    const int idx1 = (y_min    ) * width + x_min;
    const int idx2 = (y_min    ) * width + x_max + 1;
    const int idx3 = (y_max + 1) * width + x_min;
    const int idx4 = (y_max + 1) * width + x_max + 1;

    const int* ptr = integralMap[orient].ptr<int>();

    int A, B ,C, D;
    A = ptr[idx1]; B = ptr[idx2];
    C = ptr[idx3]; D = ptr[idx4];

    const float current = float(D + A - B - C);

    ptr = integralMap[orientQuant].ptr<int>();

    A = ptr[idx1]; B = ptr[idx2];
    C = ptr[idx3]; D = ptr[idx4];

    const float total = float(D + A - B - C);

    return total ? ( (current / total) - thresh ) : 0.f;
}

static void rectifyPatch( const Mat& image, const KeyPoint& kp,
                          const int& patchSize, Mat& patch,
                          const bool use_scale_orientation,
                          const float scale_factor )
{
    Mat M;
    if ( use_scale_orientation )
    {
      const float s = scale_factor * (float) kp.size / (float) patchSize;

      const float cosine = (kp.angle>=0) ? cos(kp.angle*(float)CV_PI/180.0f) : 1.f;
      const float sine   = (kp.angle>=0) ? sin(kp.angle*(float)CV_PI/180.0f) : 0.f;

      float M_[] = {
          s*cosine, -s*sine,   (-s*cosine + s*sine  ) * patchSize/2.0f + kp.pt.x,
          s*sine,    s*cosine, (-s*sine   - s*cosine) * patchSize/2.0f + kp.pt.y
      };
      M = Mat( 2, 3, CV_32FC1, M_ ).clone();
    }
    else
    {
        const float s = scale_factor * (float)kp.size / (float)patchSize;
        float M_[] = {
          s,  0.f, -s * patchSize/2.0f + kp.pt.x,
          0.f,  s, -s * patchSize/2.0f + kp.pt.y
      };
      M = Mat( 2, 3, CV_32FC1, M_ ).clone();
    }

    warpAffine( image, patch, M, Size( patchSize, patchSize ),
                WARP_INVERSE_MAP + INTER_CUBIC + WARP_FILL_OUTLIERS );
}

// -------------------------------------------------
/* BoostDesc interface implementation */

struct ComputeBoostDescInvoker : ParallelLoopBody
{
    ComputeBoostDescInvoker( const Mat& _image, Mat* _descriptors,
                        const vector<KeyPoint>& _keypoints,
                        const int _desc_type, const int _grad_atype,
                        const int _orient_q, const int _patch_size,
                        const int _nWLs, const int _Dims,
                        const Mat& _wl_x_min, const Mat& _wl_x_max,
                        const Mat& _wl_y_min, const Mat& _wl_y_max,
                        const Mat& _wl_thresh, const Mat& _wl_orient,
                        const Mat& _wl_alpha, const Mat& _wl_beta,
                        const bool _use_scale_orientation,
                        const float _scale_factor )
    {
      nWLs = _nWLs;
      Dims = _Dims;
      image = _image;
      orient_q = _orient_q;
      desc_type = _desc_type;
      keypoints = _keypoints;
      grad_atype = _grad_atype;
      patch_size = _patch_size;
      descriptors = _descriptors;

      wl_beta = _wl_beta;
      wl_alpha = _wl_alpha;
      wl_x_min = _wl_x_min;
      wl_x_max = _wl_x_max;
      wl_y_min = _wl_y_min;
      wl_y_max = _wl_y_max;
      wl_thresh = _wl_thresh;
      wl_orient = _wl_orient;

      scale_factor = _scale_factor;
      use_scale_orientation  = _use_scale_orientation;
    }

    void operator ()( const cv::Range& range ) const CV_OVERRIDE
    {
      // maps
      vector<Mat> gradMap, integralMap;

      // small binary map
      uchar binLookUp[8];
      for ( unsigned int i = 0; i < 8; i++ )
        binLookUp[i] = (uchar) 1 << i;

      for ( int i = range.start; i < range.end; i++ )
      {

        Mat patch;
        // rectify the patch around a given keypoint
        rectifyPatch( image, keypoints[i], patch_size,
                      patch, use_scale_orientation, scale_factor );

        // compute gradient maps (and integral gradient maps)
        computeGradientMaps( patch, grad_atype, orient_q, gradMap );
        computeIntegrals( gradMap, orient_q, integralMap );

        float WLR;

        /*
         * BGM
         */
        if ( ( desc_type == BGM ) ||
             ( desc_type == BGM_HARD ) ||
             ( desc_type == BGM_BILINEAR )
           )
        {
          uchar* desc = descriptors->ptr<uchar>(i);
          for ( int j = 0; j < nWLs; j++ )
          {
            WLR = computeWLResponse( wl_x_min.at<int>(0,j), wl_x_max.at<int>(0,j),
                                     wl_y_min.at<int>(0,j), wl_y_max.at<int>(0,j),
                                     wl_orient.at<int>(0,j), wl_thresh.at<float>(0,j),
                                     orient_q, integralMap );
            desc[j/8] |=  ( WLR >= 0 ) ? binLookUp[ j % 8 ] : 0;
          }
        } // end BGM

        /*
         * LBGM
         */
        if ( desc_type == LBGM )
        {
          std::bitset<512> wlResponses;

          for ( int j = 0; j < nWLs; j++ )
          {
            WLR = computeWLResponse( wl_x_min.at<int>(0,j), wl_x_max.at<int>(0,j),
                                     wl_y_min.at<int>(0,j), wl_y_max.at<int>(0,j),
                                     wl_orient.at<int>(0,j), wl_thresh.at<float>(0,j),
                                     orient_q, integralMap );
            wlResponses[j] = ( WLR >= 0 ) ? 1 : 0;
        }

          float* desc = descriptors->ptr<float>(i);
          for ( int d = 0; d < Dims; d++ )
          {
            for ( int wl = 0; wl < nWLs; wl++ )
            {
              desc[d] += ( wlResponses[wl] ) ? wl_beta.at<float>(wl,d) : -wl_beta.at<float>(wl,d);
            }
          }
        } // end LBGM

        /*
         * BINBOOST
         */
        if ( ( desc_type == BINBOOST_64  ) ||
             ( desc_type == BINBOOST_128 ) ||
             ( desc_type == BINBOOST_256 )
           )
        {
          float resp;
          for ( int d = 0; d < Dims; d++ )
          {
            resp = 0;
            uchar* desc = descriptors->ptr<uchar>(i);
            for ( int wl = 0; wl < nWLs; wl++ )
            {
              WLR = computeWLResponse( wl_x_min.at<int>(d,wl), wl_x_max.at<int>(d,wl),
                                       wl_y_min.at<int>(d,wl), wl_y_max.at<int>(d,wl),
                                       wl_orient.at<int>(d,wl), wl_thresh.at<float>(d,wl),
                                       orient_q, integralMap );
              resp += ( WLR >= 0 ) ? wl_beta.at<float>(d,wl) : -wl_beta.at<float>(d,wl);
            }
            desc[d/8] |= ( resp >= 0 ) ? binLookUp[d%8] : 0;
          }
        } // end BINBOOST

        // clean-up
        patch.release();
        gradMap.clear();
        integralMap.clear();

      } // end for loop
    } // end operator

    int nWLs;
    int Dims;
    int orient_q;
    int desc_type;
    int patch_size;
    int grad_atype;
    int patch_szie;

    Mat image;
    Mat *descriptors;
    vector<KeyPoint> keypoints;

    Mat wl_x_min, wl_x_max, wl_y_min, wl_y_max;
    Mat wl_thresh, wl_orient, wl_alpha, wl_beta;

    float scale_factor;
    bool use_scale_orientation;

    enum
    {
      BGM = 100, BGM_HARD = 101, BGM_BILINEAR = 102, LBGM = 200,
      BINBOOST_64 = 300, BINBOOST_128 = 301, BINBOOST_256 = 302
    };
};

// descriptor computation using keypoints
void BoostDesc_Impl::compute( InputArray _image, vector<KeyPoint>& keypoints, OutputArray _descriptors )
{
    // do nothing if no image
    if( _image.getMat().empty() )
      return;

    if( keypoints.empty() )
      return;

    m_image = _image.getMat().clone();

    // Only 8bit images
    CV_Assert( m_image.depth() == CV_8U );

    // convert to gray inplace
    if( m_image.channels() > 1 )
      cvtColor( m_image, m_image, COLOR_BGR2GRAY );

    // initialize the variables
    _descriptors.create( (int)keypoints.size(), descriptorSize(), descriptorType() );
    _descriptors.setTo( Scalar::all(0) );

    // descriptor storage
    Mat descriptors = _descriptors.getMat();

    parallel_for_( Range( 0, (int) keypoints.size() ),
        ComputeBoostDescInvoker( m_image, &descriptors, keypoints,
                            m_desc_type, m_grad_atype, m_orient_q,
                            m_patch_size, m_nWLs, m_Dims,
                            m_wl_x_min, m_wl_x_max, m_wl_y_min, m_wl_y_max,
                            m_wl_thresh, m_wl_orient, m_wl_alpha, m_wl_beta,
                            m_use_scale_orientation, m_scale_factor )
    );
}

void BoostDesc_Impl::ini_params( const int orientQuant, const int patchSize,
                                 const int iGradAssignType,
                                 const int nDim, const int nWLs,
                                 const unsigned int thresh[], const int orient[],
                                 const int x_min[], const int x_max[],
                                 const int y_min[], const int y_max[],
                                 const unsigned int alpha[], const unsigned int beta[] )
{
    // desc type, norm, size
    if ( m_desc_type == LBGM )
    {
      m_descriptor_size = nDim;
      m_descriptor_norm = NORM_L2;
      m_descriptor_type = CV_32FC1;
    }
    else
    {
      if ( ( m_desc_type == BGM ) ||
           ( m_desc_type == BGM_HARD ) ||
           ( m_desc_type == BGM_BILINEAR )
         )
        m_descriptor_size = nWLs / 8;
      else
        m_descriptor_size = nDim / 8;

      m_descriptor_type = CV_8UC1;
      m_descriptor_norm = NORM_HAMMING;
    }

    // 2d array dim
    int dim0 = nDim;
    int dim1 = nWLs;

    // override beta dim0 on LBGM
    if ( m_desc_type == LBGM ) dim0 = 1;

    m_Dims = nDim;
    m_nWLs = nWLs;
    m_orient_q = orientQuant;
    m_patch_size = patchSize;
    m_grad_atype = iGradAssignType;

    // cast into opencv Mat type as float
    m_wl_thresh = Mat( dim0, dim1, CV_32F, reinterpret_cast<float *>(const_cast<unsigned int *>(thresh)) );
    m_wl_alpha  = Mat( dim0, dim1, CV_32F, reinterpret_cast<float *>(const_cast<unsigned int *>(alpha )) );

    // cast into opencv Mat type as integer
    m_wl_orient = Mat( dim0, dim1, CV_32S, const_cast<int *>(orient) );
    m_wl_x_min  = Mat( dim0, dim1, CV_32S, const_cast<int *>(x_min ) );
    m_wl_x_max  = Mat( dim0, dim1, CV_32S, const_cast<int *>(x_max ) );
    m_wl_y_min  = Mat( dim0, dim1, CV_32S, const_cast<int *>(y_min ) );
    m_wl_y_max  = Mat( dim0, dim1, CV_32S, const_cast<int *>(y_max ) );

    // no beta
    if ( beta == NULL ) return;

    if ( m_desc_type == LBGM )
      m_wl_beta = Mat( dim1, nDim, CV_32F, reinterpret_cast<float *>(const_cast<unsigned int *>(beta)) );
    else
      m_wl_beta = Mat( dim0, dim1, CV_32F, reinterpret_cast<float *>(const_cast<unsigned int *>(beta)) );
}

// constructor
BoostDesc_Impl::BoostDesc_Impl( int _desc, bool _use_scale_orientation, float _scale_factor )
               : m_desc_type( _desc ), m_scale_factor( _scale_factor ),
                 m_use_scale_orientation( _use_scale_orientation )
{
    // desc type
    switch ( m_desc_type )
    {
      case BGM:
        {
          #include "boostdesc_bgm.i"
          ini_params( orientQuant, patchSize,
                      iGradAssignType,
                      nDim, nWLs, thresh, orient,
                      x_min, x_max, y_min, y_max,
                      alpha, NULL );
        }
        break;
      case BGM_HARD:
        {
          #include "boostdesc_bgm_hd.i"
          ini_params( orientQuant, patchSize,
                      iGradAssignType,
                      nDim, nWLs, thresh, orient,
                      x_min, x_max, y_min, y_max,
                      alpha, NULL );
        }
        break;
      case BGM_BILINEAR:
        {
          #include "boostdesc_bgm_bi.i"
          ini_params( orientQuant, patchSize,
                      iGradAssignType,
                      nDim, nWLs, thresh, orient,
                      x_min, x_max, y_min, y_max,
                      alpha, NULL );
        }
        break;
      case LBGM:
        {
          #include "boostdesc_lbgm.i"
          ini_params( orientQuant, patchSize,
                      iGradAssignType,
                      nDim, nWLs, thresh, orient,
                      x_min, x_max, y_min, y_max,
                      alpha, beta );
        }
        break;
      case BINBOOST_64:
        {
          #include "boostdesc_binboost_064.i"
          ini_params( orientQuant, patchSize,
                      iGradAssignType,
                      nDim, nWLs, thresh, orient,
                      x_min, x_max, y_min, y_max,
                      alpha, beta );
        }
        break;
      case BINBOOST_128:
        {
          #include "boostdesc_binboost_128.i"
          ini_params( orientQuant, patchSize,
                      iGradAssignType,
                      nDim, nWLs, thresh, orient,
                      x_min, x_max, y_min, y_max,
                      alpha, beta );
        }
        break;
      case BINBOOST_256:
        {
          #include "boostdesc_binboost_256.i"
          ini_params( orientQuant, patchSize,
                      iGradAssignType,
                      nDim, nWLs, thresh, orient,
                      x_min, x_max, y_min, y_max,
                      alpha, beta );
        }
        break;
      default:
        CV_Error( Error::StsInternal, "Unknown Descriptor Type." );
    }
}

// destructor
BoostDesc_Impl::~BoostDesc_Impl()
{
}

#endif  // OPENCV_XFEATURES2D_HAS_BOOST_DATA

Ptr<BoostDesc> BoostDesc::create( int desc, bool use_scale_orientation, float scale_factor )
{
#ifdef OPENCV_XFEATURES2D_HAS_BOOST_DATA
    return makePtr<BoostDesc_Impl>( desc, use_scale_orientation, scale_factor );
#else
    CV_UNUSED(desc); CV_UNUSED(use_scale_orientation); CV_UNUSED(scale_factor);
    CV_Error(Error::StsNotImplemented, "The OpenCV xfeatures2d binaries is built without downloaded Boost decriptor features: https://github.com/opencv/opencv_contrib/issues/1301");
#endif
}


} // END NAMESPACE XFEATURES2D
} // END NAMESPACE CV
