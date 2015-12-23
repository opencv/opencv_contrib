/*********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2014, 2015
 * Zhengqin Li <li-zq12 at mails dot tsinghua dot edu dot cn>
 * Jiansheng Chen <jschenthu at mail dot tsinghua dot edu dot cn>
 * Tsinghua University
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

 "Superpixel Segmentation using Linear Spectral Clustering"
 Zhengqin Li, Jiansheng Chen, IEEE Conference on Computer Vision and Pattern
 Recognition (CVPR), Jun. 2015

 OpenCV port by: Cristian Balint <cristian dot balint at gmail dot com>
 */

#include <map>
#include <queue>
#include "precomp.hpp"

using namespace std;


namespace cv {
namespace ximgproc {

class SuperpixelLSCImpl : public SuperpixelLSC
{
public:

    SuperpixelLSCImpl( InputArray image, int region_size, float ratio );

    virtual ~SuperpixelLSCImpl();

    // perform amount of iteration
    virtual void iterate( int num_iterations = 10 );

    // get amount of superpixels
    virtual int getNumberOfSuperpixels() const;

    // get image with labels
    virtual void getLabels( OutputArray labels_out ) const;

    // get mask image with contour
    virtual void getLabelContourMask( OutputArray image, bool thick_line = true ) const;

    // enforce connectivity over labels
    virtual void enforceLabelConnectivity( int min_element_size );


protected:

    // image width
    int m_width;

    // image width
    int m_height;

    // seeds stepx
    int m_stepx;

    // seeds stepy
    int m_stepy;

    // image channels
    int m_nr_channels;

    // region size
    int m_region_size;

    // ratio
    float m_ratio;

private:

    // labels no
    int m_numlabels;

    // color coefficient
    float m_color_coeff;

    // dist coefficient
    float m_dist_coeff;

    // threshold coeff
    int m_threshold_coeff;

    // max value from
    // image channels
    float m_chvec_max;

    // stacked channels
    // of original image
    vector<Mat> m_chvec;

    // seeds on x
    vector<float> m_kseedsx;

    // seeds on y
    vector<float> m_kseedsy;

    // W
    Mat m_W;

    // labels storage
    Mat m_klabels;

    // initialization
    inline void initialize();

    // fetch seeds
    inline void GetChSeeds();

    // precompute vector space
    inline void GetFeatureSpace();

    // LSC
    inline void PerformLSC( const int& num_iterations );

    // pre-enforce connectivity over labels
    inline void PreEnforceLabelConnectivity( int min_element_size );

    // enforce connectivity over labels
    inline void PostEnforceLabelConnectivity( int threshold );

    // re-count superpixles
    inline void countSuperpixels();

};

class Superpixel
{
public:

    int Label, Size;
    vector<int> Neighbor, xLoc, yLoc;

    Superpixel( int L = 0, int S = 0 ) : Label(L), Size(S) { }

    friend bool operator == ( Superpixel& S, int L )
    {
        return S.Label == L;
    }
    friend bool operator == ( int L, Superpixel& S )
    {
        return S.Label == L;
    }
};

CV_EXPORTS Ptr<SuperpixelLSC> createSuperpixelLSC( InputArray image, int region_size, float ratio )
{
    return makePtr<SuperpixelLSCImpl>( image, region_size, ratio );
}

SuperpixelLSCImpl::SuperpixelLSCImpl( InputArray _image, int _region_size, float _ratio )
                   : m_region_size(_region_size), m_ratio(_ratio)
{
    if ( _image.isMat() )
    {
      Mat image = _image.getMat();

      // image should be valid
      CV_Assert( !image.empty() );

      // initialize sizes
      m_width = image.size().width;
      m_height = image.size().height;
      m_nr_channels = image.channels();

      // intialize channels
      split( image, m_chvec );
    }
    else if ( _image.isMatVector() )
    {
      _image.getMatVector( m_chvec );

      // array should be valid
      CV_Assert( !m_chvec.empty() );

      // initialize sizes
      m_width = m_chvec[0].size().width;
      m_height = m_chvec[0].size().height;
      m_nr_channels = (int) m_chvec.size();
    }
    else
      CV_Error( Error::StsInternal, "Invalid InputArray." );

    // init
    initialize();

    // feature space
    GetFeatureSpace();
}

SuperpixelLSCImpl::~SuperpixelLSCImpl()
{
}

int SuperpixelLSCImpl::getNumberOfSuperpixels() const
{
    return m_numlabels;
}

void SuperpixelLSCImpl::initialize()
{
    // basic coeffs
    m_color_coeff = 20.0f;
    m_threshold_coeff = 4;
    m_dist_coeff = m_color_coeff * m_ratio;

    // total amount of superpixels given region size
    m_numlabels = int(float(m_width * m_height)
                /  float(m_region_size * m_region_size));

    // max intensity
    m_chvec_max = 0.0f;
    for( int b = 0; b < m_nr_channels; b++ )
    {
      double chmin, chmax;
      minMaxIdx( m_chvec[b], &chmin, &chmax );
      if ( m_chvec_max < chmax ) m_chvec_max = (float) chmax;
    }

    // intitialize label storage
    m_klabels = Mat( m_height, m_width, CV_32S, Scalar::all(0) );

    // init seeds
    GetChSeeds();
}

void SuperpixelLSCImpl::iterate( int num_iterations )
{
    PerformLSC( num_iterations );
}

void SuperpixelLSCImpl::getLabels(OutputArray labels_out) const
{
    labels_out.assign( m_klabels );
}

void SuperpixelLSCImpl::getLabelContourMask(OutputArray _mask, bool _thick_line) const
{
    // default width
    int line_width = 2;

    if ( !_thick_line ) line_width = 1;

    _mask.create( m_height, m_width, CV_8UC1 );
    Mat mask = _mask.getMat();

    mask.setTo(0);

    const int dx8[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
    const int dy8[8] = {  0, -1, -1, -1, 0, 1, 1,  1 };

    int sz = m_width*m_height;

    vector<bool> istaken(sz, false);

    int mainindex = 0;
    for( int j = 0; j < m_height; j++ )
    {
      for( int k = 0; k < m_width; k++ )
      {
        int np = 0;
        for( int i = 0; i < 8; i++ )
        {
          int x = k + dx8[i];
          int y = j + dy8[i];

          if( (x >= 0 && x < m_width) && (y >= 0 && y < m_height) )
          {
            int index = y*m_width + x;

            if( false == istaken[index] )
            {
              if( m_klabels.at<int>(j,k) != m_klabels.at<int>(y,x) ) np++;
            }
          }
        }
        if( np > line_width )
        {
           mask.at<char>(j,k) = (uchar)255;
           istaken[mainindex] = true;
        }
        mainindex++;
      }
    }
}

/*
 * enforceLabelConnectivity
 *
 *   1. finding an adjacent label for each new component at the start
 *   2. if a certain component is too small, assigning the previously found
 *      adjacent label to this component, and not incrementing the label.
 *
 */
void SuperpixelLSCImpl::enforceLabelConnectivity( int min_element_size )
{
    int threshold = (m_width * m_height)
                  / (m_numlabels * m_threshold_coeff);

    PreEnforceLabelConnectivity( min_element_size );
    PostEnforceLabelConnectivity( threshold );
    countSuperpixels();
}

/*
 * countSuperpixels()
 *
 *   1. count unique superpixels
 *   2. relabel all superpixels
 *
 */
inline void SuperpixelLSCImpl::countSuperpixels()
{
    std::map<int,int> labels;

    int labelNum = 0;
    int prev_label = -1;
    int mark_label = 0;
    for( int x = 0; x < m_width; x++ )
    {
      for( int y = 0; y < m_height; y++ )
      {
        int curr_label = m_klabels.at<int>(y,x);

        // relax, just do relabel
        if ( curr_label == prev_label )
        {
          m_klabels.at<int>(y,x) = mark_label;
          continue;
        }

        // on label change do map lookup
        map<int,int>::iterator it = labels.find( curr_label );

        // if new label seen
        if ( it == labels.end() )
        {
          mark_label = labelNum; labelNum++;
          labels.insert( pair<int,int>( curr_label, mark_label ) );
          m_klabels.at<int>(y,x) = mark_label;
        } else
        {
          mark_label = it->second;
          m_klabels.at<int>(y,x) = mark_label;
        }
        prev_label = curr_label;
      }
    }
    m_numlabels = (int) labels.size();
}

/*
 * PreEnforceLabelConnectivity
 *
 *   1. finding an adjacent label for each new component at the start
 *   2. if a certain component is too small, assigning the previously found
 *      adjacent label to this component, and not incrementing the label.
 *
 */
inline void SuperpixelLSCImpl::PreEnforceLabelConnectivity( int min_element_size )
{
    const int dx8[8] = { -1, -1,  0,  1,  1,  1,  0, -1 };
    const int dy8[8] = {  0, -1, -1, -1,  0,  1,  1,  1 };

    int adj = 0;
    vector<int> xLoc, yLoc;
    cv::Mat mask( m_height, m_width, CV_8U , Scalar::all(0) );

    for( int i = 0; i < m_width; i++ )
    {
      for( int j = 0; j < m_height; j++)
      {
        if( mask.at<uchar>(j,i) == 0 )
        {
          int L = m_klabels.at<int>(j,i);

          for( int k = 0; k < 8; k++ )
          {
            int x = i + dx8[k];
            int y = j + dy8[k];
            if ( x >= 0 && x <= m_width -1
              && y >= 0 && y <= m_height-1)
            {
              if ( mask.at<uchar>(y,x) == 1
                 && m_klabels.at<int>(y,x) != L )
                adj = m_klabels.at<int>(y,x);
              break;
            }
          }

          mask.at<uchar>(j,i) = 1;
          xLoc.insert( xLoc.end(), i );
          yLoc.insert( yLoc.end(), j );

          size_t indexMarker = 0;
          while( indexMarker < xLoc.size() )
          {
            int x = xLoc[indexMarker];
            int y = yLoc[indexMarker];

            indexMarker++;

            int minX = ( x-1 <= 0 ) ? 0 : x-1;
            int minY = ( y-1 <= 0 ) ? 0 : y-1;
            int maxX = ( x+1 >= m_width -1 ) ? m_width -1 : x+1;
            int maxY = ( y+1 >= m_height-1 ) ? m_height-1 : y+1;
            for( int m = minX; m <= maxX; m++ )
            {
              for( int n = minY; n <= maxY; n++ )
              {
                if (   mask.at<uchar>(n,m) == 0
                 && m_klabels.at<int>(n,m) == L )
                {
                  mask.at<uchar>(n,m) = 1;
                  xLoc.insert( xLoc.end(), m );
                  yLoc.insert( yLoc.end(), n );
                }
              }
            }
          }
          if ( indexMarker < (size_t) min_element_size )
          {
            for( size_t k = 0; k < xLoc.size(); k++ )
            {
              int x = xLoc[k];
              int y = yLoc[k];
              m_klabels.at<int>(y,x) = adj;
            }
          }
          xLoc.clear();
          yLoc.clear();
        }
      }
    }
}

/*
 * PostEnforceLabelConnectivity
 *
 */
inline void SuperpixelLSCImpl::PostEnforceLabelConnectivity( int threshold )
{
    float PI2 = float(CV_PI / 2.0f);

    vector<float> centerW;
    queue <int> xLoc, yLoc;
    vector<float> centerX1, centerX2;
    vector<float> centerY1, centerY2;
    vector<int> strayX, strayY, Size;
    vector< vector<float> > centerC1( m_nr_channels );
    vector< vector<float> > centerC2( m_nr_channels );

    cv::Mat mask( m_height, m_width, CV_8U, Scalar::all(0) );

    int L;
    int sLabel = -1;
    for( int i = 0; i < m_width; i++ )
    {
      for( int j = 0; j < m_height; j++ )
      {
        if( mask.at<uchar>(j,i) == 0 )
        {
          sLabel++;
          int count = 1;

          centerW.insert( centerW.end(), 0 );
          for ( int b = 0; b < m_nr_channels; b++ )
          {
            centerC1[b].insert( centerC1[b].end(), 0 );
            centerC2[b].insert( centerC2[b].end(), 0 );
          }
          centerX1.insert( centerX1.end(), 0 );
          centerX2.insert( centerX2.end(), 0 );
          centerY1.insert( centerY1.end(), 0 );
          centerY2.insert( centerY2.end(), 0 );

          strayX.insert( strayX.end(), i );
          strayY.insert( strayY.end(), j );

          float Weight = m_W.at<float>(j,i);

          // accumulate dists
          centerW[sLabel] += Weight;
          for ( int b = 0; b < m_nr_channels; b++ )
          {

            float thetaC = 0.0f;
            switch ( m_chvec[b].depth() )
            {
              case CV_8U:
                thetaC = ( (float) m_chvec[b].at<uchar>(j,i)  / m_chvec_max ) * PI2;
                break;
              case CV_8S:
                thetaC = ( (float) m_chvec[b].at<char>(j,i)   / m_chvec_max ) * PI2;
                break;
              case CV_16U:
                thetaC = ( (float) m_chvec[b].at<ushort>(j,i) / m_chvec_max ) * PI2;
                break;
              case CV_16S:
                thetaC = ( (float) m_chvec[b].at<short>(j,i)  / m_chvec_max ) * PI2;
                break;
              case CV_32S:
                thetaC = ( (float) m_chvec[b].at<int>(j,i)    / m_chvec_max ) * PI2;
                break;
              case CV_32F:
                thetaC = ( (float) m_chvec[b].at<float>(j,i)  / m_chvec_max ) * PI2;
                break;
              case CV_64F:
                thetaC = ( (float) m_chvec[b].at<double>(j,i) / m_chvec_max ) * PI2;
                break;
              default:
                CV_Error( Error::StsInternal, "Invalid matrix depth" );
                break;
            }

            // we do not store pre-computed C1[b], C2[b]
            float C1 = m_color_coeff * cos(thetaC) / m_nr_channels;
            float C2 = m_color_coeff * sin(thetaC) / m_nr_channels;

            centerC1[b][sLabel] += C1; centerC2[b][sLabel] += C2;
          }

          float thetaX = ( (float) i / (float) m_stepx ) * PI2;
          // we do not store pre-computed x1, x2
          float X1 = m_dist_coeff * cos(thetaX);
          float X2 = m_dist_coeff * sin(thetaX);

          float thetaY = ( (float) j / (float) m_stepy ) * PI2;
          // we do not store pre-computed y1, y2
          float Y1 = m_dist_coeff * cos(thetaY);
          float Y2 = m_dist_coeff * sin(thetaY);

          centerX1[sLabel] += X1; centerX2[sLabel] += X2;
          centerY1[sLabel] += Y1; centerY2[sLabel] += Y2;

          L = m_klabels.at<int>(j,i);
          m_klabels.at<int>(j,i) = sLabel;

          mask.at<uchar>(j,i) = 1;

          xLoc.push( i ); yLoc.push( j );
          while( !xLoc.empty() )
          {
            int x = xLoc.front(); xLoc.pop();
            int y = yLoc.front(); yLoc.pop();
            int minX = ( x-1 <=0 ) ? 0 : x-1;
            int minY = ( y-1 <=0 ) ? 0 : y-1;
            int maxX = ( x+1 >= m_width -1 ) ? m_width -1 : x+1;
            int maxY = ( y+1 >= m_height-1 ) ? m_height-1 : y+1;
            for( int m = minX; m <= maxX; m++ )
            {
              for( int n = minY; n <= maxY; n++ )
              {
                if(   mask.at<uchar>(n,m) == 0
                && m_klabels.at<int>(n,m) == L )
                {
                  count++;
                  xLoc.push(m); yLoc.push(n);

                  mask.at<uchar>(n,m) = 1;
                  m_klabels.at<int>(n,m) = sLabel;

                  Weight = m_W.at<float>(n,m);
                  centerW[sLabel] += Weight;
                  for ( int b = 0; b < m_nr_channels; b++ )
                  {
                    float thetaC = 0.0f;
                    switch ( m_chvec[b].depth() )
                    {
                      case CV_8U:
                        thetaC = ( (float) m_chvec[b].at<uchar>(j,i)  / m_chvec_max ) * PI2;
                        break;
                      case CV_8S:
                        thetaC = ( (float) m_chvec[b].at<char>(j,i)   / m_chvec_max ) * PI2;
                        break;
                      case CV_16U:
                        thetaC = ( (float) m_chvec[b].at<ushort>(j,i) / m_chvec_max ) * PI2;
                        break;
                      case CV_16S:
                        thetaC = ( (float) m_chvec[b].at<short>(j,i)  / m_chvec_max ) * PI2;
                        break;
                      case CV_32S:
                        thetaC = ( (float) m_chvec[b].at<int>(j,i)    / m_chvec_max ) * PI2;
                        break;
                      case CV_32F:
                        thetaC = ( (float) m_chvec[b].at<float>(j,i)  / m_chvec_max ) * PI2;
                        break;
                      case CV_64F:
                        thetaC = ( (float) m_chvec[b].at<double>(j,i) / m_chvec_max ) * PI2;
                        break;
                      default:
                        CV_Error( Error::StsInternal, "Invalid matrix depth" );
                        break;
                    }
                    // we do not store pre-computed C1[b], C2[b]
                    float C1 = m_color_coeff * cos(thetaC) / m_nr_channels;
                    float C2 = m_color_coeff * sin(thetaC) / m_nr_channels;

                    centerC1[b][sLabel] += C1; centerC2[b][sLabel] += C2;
                  }

                  thetaX = ( (float) m / (float) m_stepx ) * PI2;
                  // we do not store pre-computed x1, x2
                  X1 = m_dist_coeff * cos(thetaX);
                  X2 = m_dist_coeff * sin(thetaX);

                  thetaY = ( (float) n / (float) m_stepy ) * PI2;
                  // we do not store pre-computed y1, y2
                  Y1 = m_dist_coeff * cos(thetaY);
                  Y2 = m_dist_coeff * sin(thetaY);

                  centerX1[sLabel] += X1; centerX2[sLabel] += X2;
                  centerY1[sLabel] += Y1; centerY2[sLabel] += Y2;

                }
              }
            }
          }
          Size.insert( Size.end(), count );
          for ( int b = 0; b < m_nr_channels; b++ )
          {
            centerC1[b][sLabel] /= centerW[sLabel];
            centerC2[b][sLabel] /= centerW[sLabel];
          }
          centerX1[sLabel] /= centerW[sLabel];
          centerX2[sLabel] /= centerW[sLabel];
          centerY1[sLabel] /= centerW[sLabel];
          centerY2[sLabel] /= centerW[sLabel];
        }
      }
    }
    sLabel++;

    vector<Superpixel> Sarray;
    vector<int>::iterator Pointer;
    for( int i = 0; i < sLabel; i++)
    {
      if( Size[i] < threshold )
      {
        int x = strayX[i];
        int y = strayY[i];

        L = m_klabels.at<int>(y,x);
        mask.at<uchar>(y,x) = 0;

        size_t indexMark = 0;
        Superpixel S( L, Size[i] );

        S.xLoc.insert( S.xLoc.end(),x );
        S.yLoc.insert( S.yLoc.end(),y );
        while( indexMark < S.xLoc.size() )
        {
          x = S.xLoc[indexMark];
          y = S.yLoc[indexMark];

          indexMark++;

          int minX = ( x-1 <= 0 ) ? 0 : x-1;
          int minY = ( y-1 <= 0 ) ? 0 : y-1;
          int maxX = ( x+1 >= m_width -1 ) ? m_width -1 : x+1;
          int maxY = ( y+1 >= m_height-1 ) ? m_height-1 : y+1;
          for( int m = minX; m <= maxX; m++ )
          {
            for( int n = minY; n <= maxY; n++ )
            {
              if(   mask.at<uchar>(n,m) == 1
              && m_klabels.at<int>(n,m) == L )
              {
                mask.at<uchar>(n,m) = 0;

                S.xLoc.insert( S.xLoc.end(), m );
                S.yLoc.insert( S.yLoc.end(), n );
              }
              else if( m_klabels.at<int>(n,m) != L )
              {
                int NewLabel = m_klabels.at<int>(n,m);
                Pointer = find( S.Neighbor.begin(), S.Neighbor.end(), NewLabel );
                if ( Pointer == S.Neighbor.end() )
                {
                  S.Neighbor.insert( S.Neighbor.begin(), NewLabel );
                }
              }
            }
          }
        }
        Sarray.insert(Sarray.end(),S);
      }
    }

    vector<int>::iterator I, I2;
    vector<Superpixel>::iterator S;

    S = Sarray.begin();
    while( S != Sarray.end() )
    {
      int Label1 = (*S).Label;
      int Label2 = -1;

      double MinDist = DBL_MAX;
      for ( I = (*S).Neighbor.begin(); I != (*S).Neighbor.end(); I++ )
      {
        double D = 0.0f;

        for ( int b = 0; b < m_nr_channels; b++ )
        {
          float diffcenterC1 = centerC1[b][Label1]
                             - centerC1[b][*I];
          float diffcenterC2 = centerC2[b][Label1]
                             - centerC2[b][*I];

          D += (diffcenterC1 * diffcenterC1)
             + (diffcenterC2 * diffcenterC2);
        }

        float diffcenterX1 = centerX1[Label1] - centerX1[*I];
        float diffcenterX2 = centerX2[Label1] - centerX2[*I];
        float diffcenterY1 = centerY1[Label1] - centerY1[*I];
        float diffcenterY2 = centerY2[Label1] - centerY2[*I];

        D += (diffcenterX1 * diffcenterX1)
           + (diffcenterX2 * diffcenterX2)
           + (diffcenterY1 * diffcenterY1)
           + (diffcenterY2 * diffcenterY2);

        // if within dist
        if ( D < MinDist )
        {
          MinDist = D;
          Label2 = (*I);
        }
      }

      double W1 = centerW[Label1];
      double W2 = centerW[Label2];

      double W = W1 + W2;

      for ( int b = 0; b < m_nr_channels; b++ )
      {
        centerC1[b][Label2] = float((W2*centerC1[b][Label2] + W1*centerC1[b][Label1]) / W);
        centerC2[b][Label2] = float((W2*centerC2[b][Label2] + W1*centerC2[b][Label1]) / W);
      }
      centerX1[Label2] = float((W2*centerX1[Label2] + W1*centerX1[Label1]) / W);
      centerX2[Label2] = float((W2*centerX2[Label2] + W1*centerX2[Label1]) / W);
      centerY1[Label2] = float((W2*centerY1[Label2] + W1*centerY1[Label1]) / W);
      centerY2[Label2] = float((W2*centerY2[Label2] + W1*centerY2[Label1]) / W);

      centerW[Label2] = (float)W;
      for( size_t i = 0; i < (*S).xLoc.size(); i++ )
      {
        int x = (*S).xLoc[i];
        int y = (*S).yLoc[i];
        m_klabels.at<int>(y,x) = Label2;
      }

      vector<Superpixel>::iterator Stmp;
      Stmp = find( Sarray.begin(), Sarray.end(), Label2 );
      if( Stmp != Sarray.end() )
      {
        Size[Label2] = Size[Label1] + Size[Label2];
        if( Size[Label2] >= threshold )
        {
          Sarray.erase( S );
          Sarray.erase( Stmp );
        }
        else
        {
          (*Stmp).xLoc.insert( (*Stmp).xLoc.end(), (*S).xLoc.begin(), (*S).xLoc.end() );
          (*Stmp).yLoc.insert( (*Stmp).yLoc.end(), (*S).yLoc.begin(), (*S).yLoc.end() );

          (*Stmp).Neighbor.insert( (*Stmp).Neighbor.end(), (*S).Neighbor.begin(), (*S).Neighbor.end() );

          sort( (*Stmp).Neighbor.begin(), (*Stmp).Neighbor.end() );

          I = unique( (*Stmp).Neighbor.begin(), (*Stmp).Neighbor.end() );
          (*Stmp).Neighbor.erase( I, (*Stmp).Neighbor.end() );

          I = find  ( (*Stmp).Neighbor.begin(), (*Stmp).Neighbor.end(), Label1 );
          (*Stmp).Neighbor.erase( I );

          I = find  ( (*Stmp).Neighbor.begin(), (*Stmp).Neighbor.end(), Label2 );
          (*Stmp).Neighbor.erase( I );

          Sarray.erase( S );
        }
      } else Sarray.erase( S );

      for( size_t i = 0; i < Sarray.size(); i++ )
      {
        I  = find( Sarray[i].Neighbor.begin(), Sarray[i].Neighbor.end(), Label1 );
        I2 = find( Sarray[i].Neighbor.begin(), Sarray[i].Neighbor.end(), Label2 );

        if ( I  != Sarray[i].Neighbor.end()
        &&   I2 != Sarray[i].Neighbor.end() )
        {
          Sarray[i].Neighbor.erase( I );
        }
        else
        if ( I  != Sarray[i].Neighbor.end()
         &&  I2 == Sarray[i].Neighbor.end() )
        {
          (*I) = Label2;
        }
      }

      S = Sarray.begin();

    }
}

/*
 * GetChannelsSeeds_ForGivenStepSize
 *
 * The k seed values are
 * taken as uniform spatial
 * pixel samples.
 *
 */
inline void SuperpixelLSCImpl::GetChSeeds()
{
    int ColNum = (int) sqrt( (double) m_numlabels
               * ((double)m_width / (double)m_height) );
    int RowNum = m_numlabels / ColNum;

    m_stepx = m_width / ColNum;
    m_stepy = m_height / RowNum;

    int Col_remain = m_width  - (m_stepx*ColNum);
    int Row_remain = m_height - (m_stepy*RowNum);

    int count = 0;
    int t1 = 1, t2 = 1;
    int centerx, centery;

    for( int x = 0; x < ColNum; x++ )
    {
      t2 = 1;
      centerx = int((x*m_stepx) + (0.5f*m_stepx) + t1);
      if ( centerx >= m_width -1 ) centerx = m_width -1;

      for( int y = 0; y < RowNum; y++ )
      {
        centery = int((y*m_stepy) + (0.5f*m_stepy) + t2);

        if ( t2 < Row_remain ) t2++;
        if ( centery >= m_height-1 ) centery = m_height-1;

        m_kseedsx.push_back( (float)centerx );
        m_kseedsy.push_back( (float)centery );

        count++;
      }
      if ( t1 < Col_remain ) t1++;
    }
    // update amount
    m_numlabels = count;
}

struct FeatureSpaceSigmas
{
    FeatureSpaceSigmas( const vector< Mat >& _chvec, const int _nr_channels,
                        const float _chvec_max, const float _dist_coeff,
                        const float _color_coeff, const int _stepx, const int _stepy )
    {
      chvec = _chvec;
      stepx = _stepx;
      stepy = _stepy;
      chvec_max = _chvec_max;
      dist_coeff = _dist_coeff;
      nr_channels = _nr_channels;
      color_coeff = _color_coeff;

      PI2 = float(CV_PI / 2.0f);
      sigmaX1 = 0; sigmaX2 = 0;
      sigmaY1 = 0; sigmaY2 = 0;
      sigmaC1.resize( nr_channels );
      sigmaC2.resize( nr_channels );
      fill( sigmaC1.begin(), sigmaC1.end(), 0 );
      fill( sigmaC2.begin(), sigmaC2.end(), 0 );
    }

    FeatureSpaceSigmas( const FeatureSpaceSigmas& counter, Split )
    {
      *this = counter;
      sigmaX1 = 0; sigmaX2 = 0;
      sigmaY1 = 0; sigmaY2 = 0;
      sigmaC1.resize( nr_channels );
      sigmaC2.resize( nr_channels );
      fill( sigmaC1.begin(), sigmaC1.end(), 0 );
      fill( sigmaC2.begin(), sigmaC2.end(), 0 );
    }

    void operator()( const BlockedRange& range )
    {
      // previous block state
      double tmp_sigmaX1 = sigmaX1;
      double tmp_sigmaX2 = sigmaX2;
      double tmp_sigmaY1 = sigmaY1;
      double tmp_sigmaY2 = sigmaY2;
      vector<double> tmp_sigmaC1( nr_channels );
      vector<double> tmp_sigmaC2( nr_channels );
      for( int b = 0; b < nr_channels; b++ )
      {
        tmp_sigmaC1[b] = sigmaC1[b];
        tmp_sigmaC2[b] = sigmaC2[b];
      }

      for ( int x = range.begin(); x != range.end(); x++ )
      {
        float thetaX = ( (float) x / (float) stepx ) * PI2;
        // we do not store pre-computed x1, x2
        float x1 = dist_coeff * cos(thetaX);
        float x2 = dist_coeff * sin(thetaX);

        for( int y = 0; y < chvec[0].rows; y++ )
        {
          float thetaY = ( (float) y / (float) stepy ) * PI2;
          // we do not store pre-computed y1, y2
          float y1 = dist_coeff * cos(thetaY);
          float y2 = dist_coeff * sin(thetaY);

          // accumulate distance sigmas
          tmp_sigmaX1 += x1; tmp_sigmaX2 += x2;
          tmp_sigmaY1 += y1; tmp_sigmaY2 += y2;

          for( int b = 0; b < nr_channels; b++ )
          {
            float thetaC = 0.0f;
            switch ( chvec[b].depth() )
            {
              case CV_8U:
                thetaC = ( (float) chvec[b].at<uchar>(y,x)  / chvec_max ) * PI2;
                break;
              case CV_8S:
                thetaC = ( (float) chvec[b].at<char>(y,x)   / chvec_max ) * PI2;
                break;
              case CV_16U:
                thetaC = ( (float) chvec[b].at<ushort>(y,x) / chvec_max ) * PI2;
                break;
              case CV_16S:
                thetaC = ( (float) chvec[b].at<short>(y,x)  / chvec_max ) * PI2;
                break;
              case CV_32S:
                thetaC = ( (float) chvec[b].at<int>(y,x)    / chvec_max ) * PI2;
                break;
              case CV_32F:
                thetaC = ( (float) chvec[b].at<float>(y,x)  / chvec_max ) * PI2;
                break;
              case CV_64F:
                thetaC = ( (float) chvec[b].at<double>(y,x) / chvec_max ) * PI2;
                break;
              default:
                CV_Error( Error::StsInternal, "Invalid matrix depth" );
                break;
            }
            // we do not store pre-computed C1[b], C2[b]
            float C1 = color_coeff * cos(thetaC) / nr_channels;
            float C2 = color_coeff * sin(thetaC) / nr_channels;

            // accumulate sigmas per channels
            tmp_sigmaC1[b] += C1; tmp_sigmaC2[b] += C2;
          }
        }
      }
      sigmaX1 = tmp_sigmaX1; sigmaX2 = tmp_sigmaX2;
      sigmaY1 = tmp_sigmaY1; sigmaY2 = tmp_sigmaY2;
      for( int b = 0; b < nr_channels; b++ )
      {
        sigmaC1[b] = tmp_sigmaC1[b];
        sigmaC2[b] = tmp_sigmaC2[b];
      }
    }

    void join( FeatureSpaceSigmas& fsc )
    {
      sigmaX1 += fsc.sigmaX1; sigmaX2 += fsc.sigmaX2;
      sigmaY1 += fsc.sigmaY1; sigmaY2 += fsc.sigmaY2;
      for( int b = 0; b < nr_channels; b++ )
      {
        sigmaC1[b] += fsc.sigmaC1[b];
        sigmaC2[b] += fsc.sigmaC2[b];
      }
    }

    float PI2;
    int nr_channels;
    int stepx, stepy;

    double sigmaX1, sigmaX2;
    double sigmaY1, sigmaY2;

    float chvec_max;
    float dist_coeff;
    float color_coeff;

    vector<Mat> chvec;
    vector<double> sigmaC1;
    vector<double> sigmaC2;
};

struct FeatureSpaceWeights : ParallelLoopBody
{
    FeatureSpaceWeights( const vector< Mat >& _chvec, Mat* _W,
                         const double _sigmaX1, const double _sigmaX2,
                         const double _sigmaY1, const double _sigmaY2,
                         vector<double>& _sigmaC1, vector<double>& _sigmaC2,
                         const int _nr_channels, const float _chvec_max,
                         const float _dist_coeff, const float _color_coeff,
                         const int _stepx, const int _stepy )
    {
      W = _W;
      chvec = _chvec;
      stepx = _stepx;
      stepy = _stepy;
      chvec_max = _chvec_max;
      dist_coeff = _dist_coeff;
      nr_channels = _nr_channels;
      color_coeff = _color_coeff;

      PI2 = float(CV_PI / 2.0f);

      sigmaX1 = _sigmaX1; sigmaX2 = _sigmaX2;
      sigmaY1 = _sigmaY1; sigmaY2 = _sigmaY2;
      sigmaC1 = _sigmaC1; sigmaC2 = _sigmaC2;
    }

    void operator()( const Range& range ) const
    {
      for( int x = range.start; x < range.end; x++ )
      {
        float thetaX = ( (float) x / (float) stepx ) * PI2;

        for( int y = 0; y < chvec[0].rows; y++ )
        {
          float thetaY = ( (float) y / (float) stepy ) * PI2;

          // accumulate distance channels weighted by sigmas
          W->at<float>(y,x) += float((dist_coeff * cos(thetaX)) * sigmaX1);
          W->at<float>(y,x) += float((dist_coeff * sin(thetaX)) * sigmaX2);
          W->at<float>(y,x) += float((dist_coeff * cos(thetaY)) * sigmaY1);
          W->at<float>(y,x) += float((dist_coeff * sin(thetaY)) * sigmaY2);

          for( int b = 0; b < nr_channels; b++ )
          {
            float thetaC = 0.0f;
            switch ( chvec[b].depth() )
            {
              case CV_8U:
                thetaC = ( (float) chvec[b].at<uchar>(y,x)  / chvec_max ) * PI2;
                break;
              case CV_8S:
                thetaC = ( (float) chvec[b].at<char>(y,x)   / chvec_max ) * PI2;
                break;
              case CV_16U:
                thetaC = ( (float) chvec[b].at<ushort>(y,x) / chvec_max ) * PI2;
                break;
              case CV_16S:
                thetaC = ( (float) chvec[b].at<short>(y,x)  / chvec_max ) * PI2;
                break;
              case CV_32S:
                thetaC = ( (float) chvec[b].at<int>(y,x)    / chvec_max ) * PI2;
                break;
              case CV_32F:
                thetaC = ( (float) chvec[b].at<float>(y,x)  / chvec_max ) * PI2;
                break;
              case CV_64F:
                thetaC = ( (float) chvec[b].at<double>(y,x) / chvec_max ) * PI2;
                break;
              default:
                CV_Error( Error::StsInternal, "Invalid matrix depth" );
                break;
            }
            // accumulate color channels weighted by sigmas
            W->at<float>(y,x) += float((color_coeff * cos(thetaC) / nr_channels) * sigmaC1[b]);
            W->at<float>(y,x) += float((color_coeff * sin(thetaC) / nr_channels) * sigmaC2[b]);
          }
        }
      }
    }

    Mat* W;
    float PI2;
    int nr_channels;
    int stepx, stepy;

    double sigmaX1, sigmaX2;
    double sigmaY1, sigmaY2;

    float chvec_max;
    float dist_coeff;
    float color_coeff;

    vector<Mat> chvec;
    vector<double> sigmaC1;
    vector<double> sigmaC2;
};

/*
 * Compute Feature Space
 *
 *
 */
inline void SuperpixelLSCImpl::GetFeatureSpace()
{
    double sigmaX1 = 0.0f, sigmaX2 = 0.0f;
    double sigmaY1 = 0.0f, sigmaY2 = 0.0f;
    vector<double> sigmaC1( m_nr_channels , 0.0f );
    vector<double> sigmaC2( m_nr_channels , 0.0f );

    // compute feature space accumulation sigmas
    FeatureSpaceSigmas fss( m_chvec, m_nr_channels, m_chvec_max,
                            m_dist_coeff, m_color_coeff, m_stepx, m_stepy );
    parallel_reduce( BlockedRange(0, m_width), fss );

    sigmaX1 = fss.sigmaX1; sigmaX2 = fss.sigmaX2;
    sigmaY1 = fss.sigmaY1; sigmaY2 = fss.sigmaY2;
    for( int b = 0; b < m_nr_channels; b++ )
    {
      sigmaC1[b] = fss.sigmaC1[b];
      sigmaC2[b] = fss.sigmaC2[b];
    }

    // normalize sigmas
    sigmaY1 /= m_width*m_height;
    sigmaY2 /= m_width*m_height;
    sigmaX1 /= m_width*m_height;
    sigmaX2 /= m_width*m_height;
    for( int b = 0; b < m_nr_channels; b++ )
    {
      sigmaC1[b] /= m_width*m_height;
      sigmaC2[b] /= m_width*m_height;
    }

    // compute m_W normalization array
    m_W = Mat( m_height, m_width, CV_32F );
    parallel_for_( Range(0, m_width), FeatureSpaceWeights( m_chvec, &m_W,
                   sigmaX1, sigmaX2, sigmaY1, sigmaY2, sigmaC1, sigmaC2,
                   m_nr_channels, m_chvec_max, m_dist_coeff, m_color_coeff,
                   m_stepx, m_stepy ) );
}

struct FeatureSpaceCenters : ParallelLoopBody
{
    FeatureSpaceCenters( const vector< Mat >& _chvec, const Mat& _W,
                         const vector<float>& _kseedsx, const vector<float>& _kseedsy,
                         vector<float>* _centerX1, vector<float>* _centerX2,
                         vector<float>* _centerY1, vector<float>* _centerY2,
                         vector< vector<float> >* _centerC1, vector< vector<float> >* _centerC2,
                         const int _nr_channels, const float _chvec_max,
                         const float _dist_coeff, const float _color_coeff,
                         const int _stepx, const int _stepy )
    {
      W = _W;
      chvec = _chvec;
      stepx = _stepx;
      stepy = _stepy;
      chvec_max = _chvec_max;
      dist_coeff = _dist_coeff;
      nr_channels = _nr_channels;
      color_coeff = _color_coeff;

      PI2 = float(CV_PI / 2.0f);

      kseedsx = _kseedsx;
      kseedsy = _kseedsy;
      width  = chvec[0].cols;
      height = chvec[0].rows;

      centerX1 = _centerX1; centerX2 = _centerX2;
      centerY1 = _centerY1; centerY2 = _centerY2;
      centerC1 = _centerC1; centerC2 = _centerC2;
    }

    void operator()( const Range& range ) const
    {
      for( int i = range.start; i < range.end; i++ )
      {
        centerX1->at(i) = 0.0f; centerX2->at(i) = 0.0f;
        centerY1->at(i) = 0.0f; centerY2->at(i) = 0.0f;
        for( int b = 0; b < nr_channels; b++ )
        {
          centerC1->at(b)[i] = 0.0f; centerC2->at(b)[i] = 0.0f;
        }

        int X = (int)kseedsx[i]; int Y = (int)kseedsy[i];
        int minX = (X-stepx/4 <= 0) ? 0 : X-stepx/4;
        int minY = (Y-stepy/4 <= 0) ? 0 : Y-stepy/4;
        int maxX = (X+stepx/4 >= width -1) ? width -1 : X+stepx/4;
        int maxY = (Y+stepy/4 >= height-1) ? height-1 : Y+stepy/4;

        int count = 0;
        for( int x = minX; x <= maxX; x++ )
        {
          float thetaX = ( (float) x / (float) stepx ) * PI2;

          float tx1 = dist_coeff * cos(thetaX);
          float tx2 = dist_coeff * sin(thetaX);

          for( int y = minY; y <= maxY; y++ )
          {
            count++;
            float thetaY = ( (float) y / (float) stepy ) * PI2;

            // we do not store pre-computed x1, x2
            float x1 = tx1 / W.at<float>(y,x);
            float x2 = tx2 / W.at<float>(y,x);

            // we do not store pre-computed y1, y2
            float y1 = (dist_coeff * cos(thetaY)) / W.at<float>(y,x);
            float y2 = (dist_coeff * sin(thetaY)) / W.at<float>(y,x);

            centerX1->at(i) += x1; centerX2->at(i) += x2;
            centerY1->at(i) += y1; centerY2->at(i) += y2;

            for( int b = 0; b < nr_channels; b++ )
            {
              float thetaC = 0.0f;
              switch ( chvec[b].depth() )
              {
                case CV_8U:
                  thetaC = ( (float) chvec[b].at<uchar>(y,x)  / chvec_max ) * PI2;
                  break;
                case CV_8S:
                  thetaC = ( (float) chvec[b].at<char>(y,x)   / chvec_max ) * PI2;
                  break;
                case CV_16U:
                  thetaC = ( (float) chvec[b].at<ushort>(y,x) / chvec_max ) * PI2;
                  break;
                case CV_16S:
                  thetaC = ( (float) chvec[b].at<short>(y,x)  / chvec_max ) * PI2;
                  break;
                case CV_32S:
                  thetaC = ( (float) chvec[b].at<int>(y,x)    / chvec_max ) * PI2;
                  break;
                case CV_32F:
                  thetaC = ( (float) chvec[b].at<float>(y,x)  / chvec_max ) * PI2;
                  break;
                case CV_64F:
                  thetaC = ( (float) chvec[b].at<double>(y,x) / chvec_max ) * PI2;
                  break;
                default:
                  CV_Error( Error::StsInternal, "Invalid matrix depth" );
                  break;
              }
              // we do not store pre-computed C1[b], C2[b]
              float C1 = (color_coeff * cos(thetaC) / nr_channels) / W.at<float>(y,x);
              float C2 = (color_coeff * sin(thetaC) / nr_channels) / W.at<float>(y,x);

              centerC1->at(b)[i] += C1; centerC2->at(b)[i] += C2;
            }
          }
        }
        // normalize
        centerX1->at(i) /= count; centerX2->at(i) /= count;
        centerY1->at(i) /= count; centerY2->at(i) /= count;
        for( int b = 0; b < nr_channels; b++ )
        {
          centerC1->at(b)[i] /= count; centerC2->at(b)[i] /= count;
        }
      }
    }

    Mat W;
    float PI2;
    int nr_channels;
    int stepx, stepy;
    int width, height;

    float chvec_max;
    float dist_coeff;
    float color_coeff;

    vector<Mat> chvec;
    vector<float> kseedsx, kseedsy;
    vector< vector<float> >* centerC1;
    vector< vector<float> >* centerC2;
    vector<float> *centerX1, *centerX2;
    vector<float> *centerY1, *centerY2;
};

struct FeatureSpaceKmeans : ParallelLoopBody
{
    FeatureSpaceKmeans( Mat* _klabels, Mat* _dist,
                        const vector< Mat >& _chvec, const Mat& _W,
                        const vector<float>& _kseedsx, const vector<float>& _kseedsy,
                        vector<float>& _centerX1, vector<float>& _centerX2,
                        vector<float>& _centerY1, vector<float>& _centerY2,
                        vector< vector<float> >& _centerC1, vector< vector<float> >& _centerC2,
                        const int _nr_channels, const float _chvec_max,
                        const float _dist_coeff, const float _color_coeff,
                        const int _stepx, const int _stepy )
    {
      W = _W;
      dist = _dist;
      chvec = _chvec;
      stepx = _stepx;
      stepy = _stepy;
      klabels = _klabels;
      chvec_max = _chvec_max;
      dist_coeff = _dist_coeff;
      nr_channels = _nr_channels;
      color_coeff = _color_coeff;

      PI2 = float(CV_PI / 2.0f);

      kseedsx = _kseedsx;
      kseedsy = _kseedsy;
      width  = chvec[0].cols;
      height = chvec[0].rows;

      centerX1 = _centerX1; centerX2 = _centerX2;
      centerY1 = _centerY1; centerY2 = _centerY2;
      centerC1 = _centerC1; centerC2 = _centerC2;
    }

    void operator()( const Range& range ) const
    {
      for( int i = range.start; i < range.end; i++ )
      {
        int X = (int)kseedsx[i]; int Y = (int)kseedsy[i];
        int minX = (X-(stepx) <= 0) ? 0 : X-stepx;
        int minY = (Y-(stepy) <= 0) ? 0 : Y-stepy;
        int maxX = (X+(stepx) >= width -1) ? width -1 : X+stepx;
        int maxY = (Y+(stepy) >= height-1) ? height-1 : Y+stepy;

        for( int x = minX; x <= maxX; x++ )
        {
          float thetaX = ( (float) x / (float) stepx ) * PI2;

          float tx1 = dist_coeff * cos(thetaX);
          float tx2 = dist_coeff * sin(thetaX);

          for( int y = minY; y <= maxY; y++ )
          {
            float thetaY = ( (float) y / (float) stepy ) * PI2;

            // we do not store pre-computed x1, x2
            float x1 = tx1 / W.at<float>(y,x);
            float x2 = tx2 / W.at<float>(y,x);
            // we do not store pre-computed y1, y2
            float y1 = (dist_coeff * cos(thetaY)) / W.at<float>(y,x);
            float y2 = (dist_coeff * sin(thetaY)) / W.at<float>(y,x);

            float diffx1 = x1 - centerX1[i]; float diffx2 = x2 - centerX2[i];
            float diffy1 = y1 - centerY1[i]; float diffy2 = y2 - centerY2[i];

            // compute distance given distance terms
            double D = (diffx1 * diffx1) + (diffx2 * diffx2)
                     + (diffy1 * diffy1) + (diffy2 * diffy2);

            // compute distance given channels terms
            for( int b = 0; b < nr_channels; b++ )
            {
              float thetaC = 0.0f;
              switch ( chvec[b].depth() )
              {
                case CV_8U:
                  thetaC = ( (float) chvec[b].at<uchar>(y,x)  / chvec_max ) * PI2;
                  break;
                case CV_8S:
                  thetaC = ( (float) chvec[b].at<char>(y,x)   / chvec_max ) * PI2;
                  break;
                case CV_16U:
                  thetaC = ( (float) chvec[b].at<ushort>(y,x) / chvec_max ) * PI2;
                  break;
                case CV_16S:
                  thetaC = ( (float) chvec[b].at<short>(y,x)  / chvec_max ) * PI2;
                  break;
                case CV_32S:
                  thetaC = ( (float) chvec[b].at<int>(y,x)    / chvec_max ) * PI2;
                  break;
                case CV_32F:
                  thetaC = ( (float) chvec[b].at<float>(y,x)  / chvec_max ) * PI2;
                  break;
                case CV_64F:
                  thetaC = ( (float) chvec[b].at<double>(y,x) / chvec_max ) * PI2;
                  break;
                default:
                  CV_Error( Error::StsInternal, "Invalid matrix depth" );
                  break;
              }
              // we do not store pre-computed C1[b], C2[b]
              float C1 = (color_coeff * cos(thetaC) / nr_channels) / W.at<float>(y,x);
              float C2 = (color_coeff * sin(thetaC) / nr_channels) / W.at<float>(y,x);

              float diffC1 = C1 - centerC1[b][i]; float diffC2 = C2 - centerC2[b][i];

              D += (diffC1 * diffC1) + (diffC2 * diffC2);
            }

            // assign label if within D
            if ( D < dist->at<float>(y,x) )
            {
              dist->at<float>(y,x) = (float)D;
              klabels->at<int>(y,x) = i;
            }
          }
        }

      }
    }

    Mat W;
    float PI2;
    int nr_channels;
    int stepx, stepy;
    int width, height;

    float chvec_max;
    float dist_coeff;
    float color_coeff;

    Mat* dist;
    Mat* klabels;
    vector<Mat> chvec;
    vector<float> kseedsx, kseedsy;
    vector<float> centerX1, centerX2;
    vector<float> centerY1, centerY2;
    vector< vector<float> > centerC1;
    vector< vector<float> > centerC2;
};

struct FeatureCenterDists
{
    FeatureCenterDists( const vector< Mat >& _chvec, const Mat& _W, const Mat& _klabels,
                        const int _nr_channels, const float _chvec_max, const float _dist_coeff,
                        const float _color_coeff, const int _stepx, const int _stepy, const int _numlabels )
    {
      W = _W;
      chvec = _chvec;
      stepx = _stepx;
      stepy = _stepy;
      klabels = _klabels;
      numlabels = _numlabels;
      chvec_max = _chvec_max;
      dist_coeff = _dist_coeff;
      nr_channels = _nr_channels;
      color_coeff = _color_coeff;

      PI2 = float(CV_PI / 2.0f);

      Wsum.resize(numlabels);
      kseedsx.resize(numlabels);
      kseedsy.resize(numlabels);
      centerX1.resize(numlabels);
      centerX2.resize(numlabels);
      centerY1.resize(numlabels);
      centerY2.resize(numlabels);
      centerC1.resize(nr_channels);
      centerC2.resize(nr_channels);
      clusterSize.resize(numlabels);
      for( int b = 0; b < nr_channels; b++ )
      {
        centerC1[b].resize(numlabels);
        centerC2[b].resize(numlabels);
      }
      // refill with zero all arrays
      fill(centerX1.begin(), centerX1.end(), 0.0f);
      fill(centerX2.begin(), centerX2.end(), 0.0f);
      fill(centerY1.begin(), centerY1.end(), 0.0f);
      fill(centerY2.begin(), centerY2.end(), 0.0f);
      for( int b = 0; b < nr_channels; b++ )
      {
        fill(centerC1[b].begin(), centerC1[b].end(), 0.0f);
        fill(centerC2[b].begin(), centerC2[b].end(), 0.0f);
      }
      fill(Wsum.begin(), Wsum.end(), 0.0f);
      fill(kseedsx.begin(), kseedsx.end(), 0.0f);
      fill(kseedsy.begin(), kseedsy.end(), 0.0f);
      fill(clusterSize.begin(), clusterSize.end(), 0);
    }

    FeatureCenterDists( const FeatureCenterDists& counter, Split )
    {
      *this = counter;
      // refill with zero all arrays
      fill(centerX1.begin(), centerX1.end(), 0.0f);
      fill(centerX2.begin(), centerX2.end(), 0.0f);
      fill(centerY1.begin(), centerY1.end(), 0.0f);
      fill(centerY2.begin(), centerY2.end(), 0.0f);
      for( int b = 0; b < nr_channels; b++ )
      {
        fill(centerC1[b].begin(), centerC1[b].end(), 0.0f);
        fill(centerC2[b].begin(), centerC2[b].end(), 0.0f);
      }
      fill(Wsum.begin(), Wsum.end(), 0.0f);
      fill(kseedsx.begin(), kseedsx.end(), 0.0f);
      fill(kseedsy.begin(), kseedsy.end(), 0.0f);
      fill(clusterSize.begin(), clusterSize.end(), 0);

    }

    void operator()( const BlockedRange& range )
    {
      // previous block state
      vector<float> tmp_Wsum = Wsum;
      vector<float> tmp_kseedsx = kseedsx;
      vector<float> tmp_kseedsy = kseedsy;
      vector<float> tmp_centerX1 = centerX1;
      vector<float> tmp_centerX2 = centerX2;
      vector<float> tmp_centerY1 = centerY1;
      vector<float> tmp_centerY2 = centerY2;
      vector< vector<float> > tmp_centerC1 = centerC1;
      vector< vector<float> > tmp_centerC2 = centerC2;
      vector<int> tmp_clusterSize = clusterSize;

      for ( int x = range.begin(); x != range.end(); x++ )
      {

        float thetaX = ( (float) x / (float) stepx ) * PI2;

        // we do not store pre-computed x1, x2
        float x1 = (dist_coeff * cos(thetaX));
        float x2 = (dist_coeff * sin(thetaX));

        for( int y = 0; y < chvec[0].rows; y++ )
        {
          float thetaY = ( (float) y / (float) stepy ) * PI2;

          // we do not store pre-computed y1, y2
          float y1 = (dist_coeff * cos(thetaY));
          float y2 = (dist_coeff * sin(thetaY));

          int L = klabels.at<int>(y,x);

          tmp_centerX1[L] += x1; tmp_centerX2[L] += x2;
          tmp_centerY1[L] += y1; tmp_centerY2[L] += y2;

          // compute distance given channels terms
          for( int b = 0; b < nr_channels; b++ )
          {
            float thetaC = 0.0f;
            switch ( chvec[b].depth() )
            {
              case CV_8U:
                thetaC = ( (float) chvec[b].at<uchar>(y,x)  / chvec_max ) * PI2;
                break;
              case CV_8S:
                thetaC = ( (float) chvec[b].at<char>(y,x)   / chvec_max ) * PI2;
                break;
              case CV_16U:
                thetaC = ( (float) chvec[b].at<ushort>(y,x) / chvec_max ) * PI2;
                break;
              case CV_16S:
                thetaC = ( (float) chvec[b].at<short>(y,x)  / chvec_max ) * PI2;
                break;
              case CV_32S:
                thetaC = ( (float) chvec[b].at<int>(y,x)    / chvec_max ) * PI2;
                break;
              case CV_32F:
                thetaC = ( (float) chvec[b].at<float>(y,x)  / chvec_max ) * PI2;
                break;
              case CV_64F:
                thetaC = ( (float) chvec[b].at<double>(y,x) / chvec_max ) * PI2;
                break;
              default:
                CV_Error( Error::StsInternal, "Invalid matrix depth" );
                break;
            }

            // we do not store pre-computed C1[b], C2[b]
            float C1 = (color_coeff * cos(thetaC) / nr_channels);
            float C2 = (color_coeff * sin(thetaC) / nr_channels);

            tmp_centerC1[b][L] += C1; tmp_centerC2[b][L] += C2;

          }
          tmp_clusterSize[L]++;
          tmp_Wsum[L] += W.at<float>(y,x);
          tmp_kseedsx[L] += x; tmp_kseedsy[L] += y;
        }
      }

      Wsum = tmp_Wsum;
      kseedsx = tmp_kseedsx;
      kseedsy = tmp_kseedsy;
      clusterSize = tmp_clusterSize;
      centerX1 = tmp_centerX1; centerX2 = tmp_centerX2;
      centerY1 = tmp_centerY1; centerY2 = tmp_centerY2;
      centerC1 = tmp_centerC1; centerC2 = tmp_centerC2;
    }

    void join( FeatureCenterDists& fcd )
    {
      for (int l = 0; l < numlabels; l++)
      {
        Wsum[l] += fcd.Wsum[l];
        kseedsx[l] += fcd.kseedsx[l];
        kseedsy[l] += fcd.kseedsy[l];
        centerX1[l] += fcd.centerX1[l];
        centerX2[l] += fcd.centerX2[l];
        centerY1[l] += fcd.centerY1[l];
        centerY2[l] += fcd.centerY2[l];
        clusterSize[l] += fcd.clusterSize[l];
        for( int b = 0; b < nr_channels; b++ )
        {
            centerC1[b][l] += fcd.centerC1[b][l];
            centerC2[b][l] += fcd.centerC2[b][l];
        }
      }
    }

    Mat W;
    float PI2;
    int numlabels;
    int nr_channels;
    int stepx, stepy;

    float chvec_max;
    float dist_coeff;
    float color_coeff;

    Mat klabels;
    vector<Mat> chvec;

    vector<float> Wsum;
    vector<int> clusterSize;
    vector<float> kseedsx, kseedsy;
    vector<float> centerX1, centerX2;
    vector<float> centerY1, centerY2;
    vector< vector<float> > centerC1, centerC2;

};

struct FeatureNormals : ParallelLoopBody
{
    FeatureNormals( const vector<float>& _Wsum, const vector<int>& _clusterSize,
                    vector<float>* _kseedsx, vector<float>* _kseedsy,
                    vector<float>* _centerX1, vector<float>* _centerX2,
                    vector<float>* _centerY1, vector<float>* _centerY2,
                    vector< vector<float> >* _centerC1, vector< vector<float> >* _centerC2,
                    const int _numlabels, const int _nr_channels )
    {
      Wsum = _Wsum;
      numlabels = _numlabels;
      clusterSize = _clusterSize;
      nr_channels = _nr_channels;

      kseedsx = _kseedsx; kseedsy = _kseedsy;
      centerX1 = _centerX1; centerX2 = _centerX2;
      centerY1 = _centerY1; centerY2 = _centerY2;
      centerC1 = _centerC1; centerC2 = _centerC2;
    }

    void operator()( const Range& range ) const
    {
      for( int i = range.start; i < range.end; i++ )
      {
        if ( Wsum[i] != 0 )
        {
          centerX1->at(i) /= Wsum[i]; centerX2->at(i) /= Wsum[i];
          centerY1->at(i) /= Wsum[i]; centerY2->at(i) /= Wsum[i];
          for( int b = 0; b < nr_channels; b++ )
          {
            centerC1->at(b)[i] /= Wsum[i]; centerC2->at(b)[i] /= Wsum[i];
          }
        }
        if ( clusterSize[i] != 0 )
        {
          kseedsx->at(i) /= clusterSize[i];
          kseedsy->at(i) /= clusterSize[i];
        }
      }
    }

    int numlabels;
    vector<float> Wsum;
    vector<int> clusterSize;
    int nr_channels;

    vector<float> *kseedsx, *kseedsy;
    vector<float> *centerX1, *centerX2;
    vector<float> *centerY1, *centerY2;
    vector< vector<float> > *centerC1;
    vector< vector<float> > *centerC2;
};


/*
 *    PerformSuperpixelLSC
 *
 *    Performs weighted kmeans segmentation
 *    in (4 + 2*m_nr_channels) dimensional space
 *
 */
inline void SuperpixelLSCImpl::PerformLSC( const int&  itrnum )
{
    // allocate initial workspaces
    cv::Mat dist( m_height, m_width, CV_32F );

    vector<float> centerX1( m_numlabels );
    vector<float> centerX2( m_numlabels );
    vector<float> centerY1( m_numlabels );
    vector<float> centerY2( m_numlabels );
    vector< vector<float> > centerC1( m_nr_channels );
    vector< vector<float> > centerC2( m_nr_channels );
    for( int b = 0; b < m_nr_channels; b++ )
    {
      centerC1[b].resize( m_numlabels );
      centerC2[b].resize( m_numlabels );
    }
    vector<float> Wsum( m_numlabels );
    vector<int> clusterSize( m_numlabels );

    // compute weighted distance centers
    parallel_for_( Range(0, m_numlabels), FeatureSpaceCenters(
                   m_chvec, m_W, m_kseedsx, m_kseedsy,
                   &centerX1, &centerX2, &centerY1, &centerY2,
                   &centerC1, &centerC2, m_nr_channels, m_chvec_max,
                   m_dist_coeff, m_color_coeff, m_stepx, m_stepy ) );

    // parallel reduce structure
    FeatureCenterDists fcd( m_chvec, m_W, m_klabels, m_nr_channels, m_chvec_max,
                            m_dist_coeff, m_color_coeff, m_stepx, m_stepy, m_numlabels );

    // K-Means
    for( int itr = 0; itr < itrnum; itr++ )
    {

      dist.setTo( FLT_MAX );

      // k-mean
      parallel_for_( Range(0, m_numlabels), FeatureSpaceKmeans(
                     &m_klabels, &dist, m_chvec, m_W, m_kseedsx, m_kseedsy,
                     centerX1, centerX2, centerY1, centerY2, centerC1, centerC2,
                     m_nr_channels, m_chvec_max, m_dist_coeff, m_color_coeff,
                     m_stepx, m_stepy ) );

      // accumulate center distances
      parallel_reduce( BlockedRange(0, m_width), fcd );

      // featch out the results
      Wsum = fcd.Wsum; clusterSize = fcd.clusterSize;
      m_kseedsx = fcd.kseedsx; m_kseedsy = fcd.kseedsy;
      centerX1 = fcd.centerX1; centerX2 = fcd.centerX2;
      centerY1 = fcd.centerY1; centerY2 = fcd.centerY2;
      centerC1 = fcd.centerC1; centerC2 = fcd.centerC2;


      // normalize accumulated distances
      parallel_for_( Range(0, m_numlabels), FeatureNormals(
                     Wsum, clusterSize, &m_kseedsx, &m_kseedsy,
                     &centerX1, &centerX2, &centerY1, &centerY2,
                     &centerC1, &centerC2, m_numlabels, m_nr_channels ) );
    }
}

} // namespace ximgproc
} // namespace cv
