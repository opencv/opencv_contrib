/*********************************************************************
 * Software License Agreement (BSD License)
 *
 * SLIC, SLICO Copyright (c) 2013
 * Radhakrishna Achanta
 * email : Radhakrishna [dot] Achanta [at] epfl [dot] ch
 * web : http://ivrl.epfl.ch/people/achanta
 *
 * MSLIC Copyright (c) 2016, 2017
 * Yong-Jin Liu
 * email : liuyongjin [at] tsinghua [dot] edu [dot] cn
 * web : http://47.89.51.189/liuyj
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
 "SLIC Superpixels Compared to State-of-the-art Superpixel Methods"
 Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua,
 and Sabine Susstrunk, IEEE TPAMI, Volume 34, Issue 11, Pages 2274-2282,
 November 2012.

 "SLIC Superpixels" Radhakrishna Achanta, Appu Shaji, Kevin Smith,
 Aurelien Lucchi, Pascal Fua, and Sabine Süsstrunk, EPFL Technical
 Report no. 149300, June 2010.

 "Intrinsic Manifold SLIC: A Simple and Efficient Method for Computing
 Content-Sensitive Superpixels"
 Yong-Jin Liu, Cheng-Chi Yu, Min-Jing Yu, Ying He,
 IEEE Transactions on Pattern Analysis and Machine Intelligence,
 March 2017, Issue 99.

 OpenCV port by: Cristian Balint <cristian dot balint at gmail dot com>
 */

#include "precomp.hpp"

using namespace std;


namespace cv {
namespace ximgproc {

class SuperpixelSLICImpl : public SuperpixelSLIC
{
public:

    SuperpixelSLICImpl( InputArray image, int algorithm, int region_size, float ruler );

    virtual ~SuperpixelSLICImpl() CV_OVERRIDE;

    // perform amount of iteration
    virtual void iterate( int num_iterations = 10 ) CV_OVERRIDE;

    // get amount of superpixels
    virtual int getNumberOfSuperpixels() const CV_OVERRIDE;

    // get image with labels
    virtual void getLabels( OutputArray labels_out ) const CV_OVERRIDE;

    // get mask image with contour
    virtual void getLabelContourMask( OutputArray image, bool thick_line = true ) const CV_OVERRIDE;

    // enforce connectivity over labels
    virtual void enforceLabelConnectivity( int min_element_size = 25 ) CV_OVERRIDE;


protected:

    // image width
    int m_width;

    // image width
    int m_height;

    // image channels
    int m_nr_channels;

    // algorithm
    int m_algorithm;

    // region size
    int m_region_size;

    // compactness
    float m_ruler;

    // ratio (MSLIC)
    float m_ratio;

    // split (MSLIC)
    float m_split;

    // current iter
    int m_cur_iter;

    // current iter
    int m_iterations;


private:

    // labels no
    int m_numlabels;

    // stacked channels
    // of original image
    vector<Mat> m_chvec;

    // seeds on x
    vector<float> m_kseedsx;

    // seeds on y
    vector<float> m_kseedsy;

    // labels storage
    Mat m_klabels;

    // seeds storage
    vector< vector<float> > m_kseeds;

    // adaptive k (MSLIC)
    vector<float> m_adaptk;

    // merge threshold (MSLIC)
    float m_merge;

    // initialization
    inline void initialize();

    // detect edges over all channels
    inline void DetectChEdges( Mat& edgemag );

    // random perturb seeds
    inline void PerturbSeeds( const Mat& edgemag );

    // fetch seeds
    inline void GetChSeedsS();

    // fetch seeds
    inline void GetChSeedsK();

    // SLIC
    inline void PerformSLIC( const int& num_iterations );

    // SLICO
    inline void PerformSLICO( const int& num_iterations );

    // MSLIC
    inline void PerformMSLIC( const int& num_iterations );

    // MSLIC
    inline void SuperpixelSplit();

};

CV_EXPORTS Ptr<SuperpixelSLIC> createSuperpixelSLIC( InputArray image, int algorithm, int region_size, float ruler )
{
    return makePtr<SuperpixelSLICImpl>( image, algorithm, region_size, ruler );
}

SuperpixelSLICImpl::SuperpixelSLICImpl( InputArray _image, int _algorithm, int _region_size, float _ruler )
                   : m_algorithm(_algorithm), m_region_size(_region_size), m_ruler(_ruler)
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
}

SuperpixelSLICImpl::~SuperpixelSLICImpl()
{
    m_chvec.clear();
    m_kseeds.clear();
    m_kseedsx.clear();
    m_kseedsy.clear();
    m_klabels.release();
}

int SuperpixelSLICImpl::getNumberOfSuperpixels() const
{
    return m_numlabels;
}

void SuperpixelSLICImpl::initialize()
{
    // total amount of superpixels given its size as input
    m_numlabels = int(float(m_width * m_height)
                /  float(m_region_size * m_region_size));

    // initialize seed storage
    m_kseeds.resize( m_nr_channels );

    // intitialize label storage
    m_klabels = Mat( m_height, m_width, CV_32S, Scalar::all(0) );

    // perturb seeds is not absolutely necessary,
    // one can set this flag to false
    bool perturbseeds = true;

    // storage for edge magnitudes
    Mat edgemag;
    if (perturbseeds)
      DetectChEdges(edgemag);

    if( m_algorithm == SLICO )
      GetChSeedsK();
    else if( ( m_algorithm == SLIC ) ||
             ( m_algorithm == MSLIC ) )
      GetChSeedsS();
    else
      CV_Error( Error::StsInternal, "No such algorithm" );

    // update amount of labels now
    m_numlabels = (int)m_kseeds[0].size();

    // perturb seeds given edges
    if (perturbseeds)
      PerturbSeeds(edgemag);

    if( m_algorithm == MSLIC )
    {
      m_merge = 4.0f;
      m_adaptk.resize( m_numlabels, 1.0f );
    }
}

void SuperpixelSLICImpl::iterate( int num_iterations )
{
    // store total iterations
    m_iterations = num_iterations;

    if( m_algorithm == SLICO )
      PerformSLICO( num_iterations );
    else if( m_algorithm == SLIC )
      PerformSLIC( num_iterations );
    else if( m_algorithm == MSLIC )
      PerformMSLIC( num_iterations );
    else
      CV_Error( Error::StsInternal, "No such algorithm" );

    // re-update amount of labels
    m_numlabels = (int)m_kseeds[0].size();
}

void SuperpixelSLICImpl::getLabels(OutputArray labels_out) const
{
    labels_out.assign( m_klabels );
}

void SuperpixelSLICImpl::getLabelContourMask(OutputArray _mask, bool _thick_line) const
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
 * EnforceLabelConnectivity
 *
 *   1. finding an adjacent label for each new component at the start
 *   2. if a certain component is too small, assigning the previously found
 *      adjacent label to this component, and not incrementing the label.
 *
 */
void SuperpixelSLICImpl::enforceLabelConnectivity( int min_element_size )
{

    if ( min_element_size == 0 ) return;
    CV_Assert( min_element_size >= 0 && min_element_size <= 100 );

    vector<float> adaptk( m_numlabels, 1.0f );

    if( m_algorithm == MSLIC )
    {
      adaptk.clear();
    }

    const int dx4[4] = { -1,  0,  1,  0 };
    const int dy4[4] = {  0, -1,  0,  1 };

    const int sz = m_width * m_height;
    const int supsz = sz / m_numlabels;

    int div = int(100.0f/(float)min_element_size + 0.5f);
    int min_sp_sz = max(3, supsz / div);

    Mat nlabels( m_height, m_width, CV_32S, Scalar(INT_MAX) );

    int label = 0;
    vector<int> xvec(sz);
    vector<int> yvec(sz);

    // MSLIC
    int currentlabel;
    float diffch = 0.0f;
    vector<float> adjch;
    vector<float> curch;
    map<int,int> hashtable;

    if( m_algorithm == MSLIC )
    {
      hashtable[-1] = 0;
      adjch.resize( m_nr_channels, 0 );
      curch.resize( m_nr_channels, 0 );
    }

    //adjacent label
    int adjlabel = 0;

    for( int j = 0; j < m_height; j++ )
    {
        for( int k = 0; k < m_width; k++ )
        {
            if( nlabels.at<int>(j,k) == INT_MAX )
            {
                nlabels.at<int>(j,k) = label;
                //--------------------
                // Start a new segment
                //--------------------
                xvec[0] = k;
                yvec[0] = j;
                currentlabel = m_klabels.at<int>(j,k);
                //-------------------------------------------------------
                // Quickly find an adjacent label for use later if needed
                //-------------------------------------------------------
                for( int n = 0; n < 4; n++ )
                {
                    int x = xvec[0] + dx4[n];
                    int y = yvec[0] + dy4[n];
                    if( (x >= 0 && x < m_width) && (y >= 0 && y < m_height) )
                    {
                        if( nlabels.at<int>(y,x) != INT_MAX )
                        {
                            adjlabel = nlabels.at<int>(y,x);
                            if( m_algorithm == MSLIC )
                            {
                                for( int b = 0; b < m_nr_channels; b++ )
                                {
                                  adjch[b] = m_kseeds[b][m_klabels.at<int>(y,x)];
                                }
                            }
                        }
                    }
                }

                if( m_algorithm == MSLIC )
                {
                    float ssumch = 0.0f;
                    for( int b = 0; b < m_nr_channels; b++ )
                    {
                        curch[b] = m_kseeds[b][m_klabels.at<int>(j,k)];
                        // squared distance
                        float diff = curch[b] - adjch[b];
                        ssumch += diff * diff;
                    }
                    // L2 distance with adj
                    diffch = sqrt( ssumch );
                    adaptk.push_back( m_adaptk[currentlabel] );
                }

                int count(1);
                for( int c = 0; c < count; c++ )
                {
                    for( int n = 0; n < 4; n++ )
                    {
                        int x = xvec[c] + dx4[n];
                        int y = yvec[c] + dy4[n];

                        if( (x >= 0 && x < m_width) && (y >= 0 && y < m_height) )
                        {
                            if( INT_MAX == nlabels.at<int>(y,x) &&
                                m_klabels.at<int>(j,k) == m_klabels.at<int>(y,x) )
                            {
                                xvec[count] = x;
                                yvec[count] = y;
                                nlabels.at<int>(y,x) = label;
                                count++;
                            }
                        }
                    }
                }
                // MSLIC only
                if( m_algorithm == MSLIC )
                {
                  if ( m_cur_iter < m_iterations - 1 )
                  {
                      hashtable[label] = count;
                      //-------------------------------------------------------
                      // If segment size is less then a limit, or is very similar
                      // to it's neighbour assign adjacent label found before,
                      // and decrement label count.
                      //-------------------------------------------------------
                      if( ( count <= min_sp_sz ) ||
                          (
                            ( diffch < m_merge ) &&
                            ( hashtable[adjlabel] + hashtable[(int)adaptk.size()-1]
                           <= 3 * m_region_size * m_region_size )
                          )
                        )
                      {
                          if( ( diffch < m_merge) &&
                              ( hashtable[adjlabel] + hashtable[(int)adaptk.size()-1]
                             <= 3 * m_region_size * m_region_size )
                            )
                          {
                              adaptk[adjlabel] = min( 2.0f, float(adaptk[adjlabel] + adaptk[(int)adaptk.size()-1]) );
                              hashtable[adjlabel] += hashtable[(int)adaptk.size()-1];
                          }

                          for( int c = 0; c < count; c++ )
                          {
                              nlabels.at<int>(yvec[c],xvec[c]) = adjlabel;
                          }

                          label--;
                          adaptk.pop_back();
                      }
                  } else
                  {
                      //-------------------------------------------------------
                      // If segment size is less then a limit, assign an
                      // adjacent label found before, and decrement label count.
                      //-------------------------------------------------------
                      if( count <= min_sp_sz )
                      {
                          for( int c = 0; c < count; c++ )
                          {
                              nlabels.at<int>(yvec[c],xvec[c]) = adjlabel;
                          }
                          label--;
                      }
                  }
                // SLIC or SLICO
                } else
                {
                  //-------------------------------------------------------
                  // If segment size is less then a limit, assign an
                  // adjacent label found before, and decrement label count.
                  //-------------------------------------------------------
                  if( count <= min_sp_sz )
                  {
                      for( int c = 0; c < count; c++ )
                      {
                          nlabels.at<int>(yvec[c],xvec[c]) = adjlabel;
                      }
                      label--;
                  }
                }
                label++;
            }
        }
    }
    // replace old
    m_klabels = nlabels;
    m_numlabels = label;

    m_adaptk.clear();
    m_adaptk = adaptk;
}

/*
 * DetectChEdges
 */
inline void SuperpixelSLICImpl::DetectChEdges( Mat &edgemag )
{
    Mat dx, dy;
    Mat S_dx, S_dy;

    for (int c = 0; c < m_nr_channels; c++)
    {
        // derivate
        Sobel( m_chvec[c], dx, CV_32F, 1, 0, 1, 1.0f, 0.0f, BORDER_DEFAULT );
        Sobel( m_chvec[c], dy, CV_32F, 0, 1, 1, 1.0f, 0.0f, BORDER_DEFAULT );

        // acumulate ^2 derivate
        MatExpr dx2 = dx.mul(dx);
        MatExpr dy2 = dy.mul(dy);
        if (S_dx.empty())
        {
            S_dx = dx2;
            S_dy = dy2;
        }
        else
        {
            S_dx += dx2;
            S_dy += dy2;
        }
    }
    // total magnitude
    edgemag = S_dx + S_dy;
}

/*
 * PerturbSeeds
 */
inline void SuperpixelSLICImpl::PerturbSeeds( const Mat& edgemag )
{
    const int dx8[8] = { -1, -1,  0,  1,  1,  1,  0, -1 };
    const int dy8[8] = {  0, -1, -1, -1,  0,  1,  1,  1 };

    for( int n = 0; n < m_numlabels; n++ )
    {
        int ox = (int)m_kseedsx[n]; //original x
        int oy = (int)m_kseedsy[n]; //original y

        int storex = ox;
        int storey = oy;
        for( int i = 0; i < 8; i++ )
        {
            int nx = ox + dx8[i]; //new x
            int ny = oy + dy8[i]; //new y

            if( nx >= 0 && nx < m_width && ny >= 0 && ny < m_height)
            {
                if( edgemag.at<float>(ny,nx) < edgemag.at<float>(storey,storex) )
                {
                    storex = nx;
                    storey = ny;
                }
            }
        }
        if( storex != ox && storey != oy )
        {
            m_kseedsx[n] = (float)storex;
            m_kseedsy[n] = (float)storey;

            switch ( m_chvec[0].depth() )
            {
              case CV_8U:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b][n] = m_chvec[b].at<uchar>( storey, storex );
                break;

              case CV_8S:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b][n] = m_chvec[b].at<char>( storey, storex );
                break;

              case CV_16U:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b][n] = m_chvec[b].at<ushort>( storey, storex );
                break;

              case CV_16S:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b][n] = m_chvec[b].at<short>( storey, storex );
                break;

              case CV_32S:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b][n] = (float) m_chvec[b].at<int>( storey, storex );
                break;

              case CV_32F:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b][n] = m_chvec[b].at<float>( storey, storex );
                break;

              case CV_64F:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b][n] = (float) m_chvec[b].at<double>( storey, storex );
                break;

              default:
                CV_Error( Error::StsInternal, "Invalid matrix depth" );
                break;
            }
        }
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
inline void SuperpixelSLICImpl::GetChSeedsS()
{
    int n = 0;
    int numseeds = 0;

    int xstrips = int(0.5f + float(m_width) / float(m_region_size) );
    int ystrips = int(0.5f + float(m_height) / float(m_region_size) );

    int xerr = m_width  - m_region_size*xstrips;
    int yerr = m_height - m_region_size*ystrips;

    float xerrperstrip = float(xerr) / float(xstrips);
    float yerrperstrip = float(yerr) / float(ystrips);

    int xoff = m_region_size / 2;
    int yoff = m_region_size / 2;

    numseeds = xstrips*ystrips;

    for ( int b = 0; b < m_nr_channels; b++ )
      m_kseeds[b].resize(numseeds);

    m_kseedsx.resize(numseeds);
    m_kseedsy.resize(numseeds);

    for( int y = 0; y < ystrips; y++ )
    {
        int ye = y * (int)yerrperstrip;
        int Y = y*m_region_size + yoff+ye;
        if( Y > m_height-1 ) continue;
        for( int x = 0; x < xstrips; x++ )
        {
            int xe = x * (int)xerrperstrip;
            int X = x*m_region_size + xoff+xe;
            if( X > m_width-1 ) continue;

            switch ( m_chvec[0].depth() )
            {
              case CV_8U:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b][n] = m_chvec[b].at<uchar>(Y,X);
                break;

              case CV_8S:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b][n] = m_chvec[b].at<char>(Y,X);
                break;

              case CV_16U:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b][n] = m_chvec[b].at<ushort>(Y,X);
                break;

              case CV_16S:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b][n] = m_chvec[b].at<short>(Y,X);
                break;

              case CV_32S:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b][n] = (float) m_chvec[b].at<int>(Y,X);
                break;

              case CV_32F:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b][n] = m_chvec[b].at<float>(Y,X);
                break;

              case CV_64F:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b][n] = (float) m_chvec[b].at<double>(Y,X);
                break;

              default:
                CV_Error( Error::StsInternal, "Invalid matrix depth" );
                break;
            }

            m_kseedsx[n] = (float)X;
            m_kseedsy[n] = (float)Y;

            n++;
        }
    }
}

/*
 * GetChannlesSeeds_ForGivenK
 *
 * The k seed values are
 * taken as uniform spatial
 * pixel samples.
 *
 */
inline void SuperpixelSLICImpl::GetChSeedsK()
{
    int xoff = m_region_size / 2;
    int yoff = m_region_size / 2;
    int n = 0; int r = 0;
    for( int y = 0; y < m_height; y++ )
    {
        int Y = y*m_region_size + yoff;
        if( Y > m_height-1 ) continue;
        for( int x = 0; x < m_width; x++ )
        {
            // hex grid
            int X = x*m_region_size + ( xoff<<( r & 0x1) );
            if( X > m_width-1 ) continue;

            switch ( m_chvec[0].depth() )
            {
              case CV_8U:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b].push_back( m_chvec[b].at<uchar>(Y,X) );
                break;

              case CV_8S:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b].push_back( m_chvec[b].at<char>(Y,X) );
                break;

              case CV_16U:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b].push_back( m_chvec[b].at<ushort>(Y,X) );
                break;

              case CV_16S:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b].push_back( m_chvec[b].at<short>(Y,X) );
                break;

              case CV_32S:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b].push_back( (float) m_chvec[b].at<int>(Y,X) );
                break;

              case CV_32F:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b].push_back( m_chvec[b].at<float>(Y,X) );
                break;

              case CV_64F:
                for( int b = 0; b < m_nr_channels; b++ )
                  m_kseeds[b].push_back( (float) m_chvec[b].at<double>(Y,X) );
                break;

              default:
                CV_Error( Error::StsInternal, "Invalid matrix depth" );
                break;
            }

            m_kseedsx.push_back((float)X);
            m_kseedsy.push_back((float)Y);

            n++;
        }
        r++;
    }
}

struct SeedNormInvoker : ParallelLoopBody
{
    SeedNormInvoker( vector< vector<float> >* _kseeds, vector< vector<float> >* _sigma,
                     vector<int>* _clustersize, vector<float>* _sigmax, vector<float>* _sigmay,
                     vector<float>* _kseedsx, vector<float>* _kseedsy, int _nr_channels )
    {
      sigma = _sigma;
      kseeds = _kseeds;
      sigmax = _sigmax;
      sigmay = _sigmay;
      kseedsx = _kseedsx;
      kseedsy = _kseedsy;
      nr_channels = _nr_channels;
      clustersize = _clustersize;
    }

    void operator ()(const cv::Range& range) const CV_OVERRIDE
    {
      for (int k = range.start; k < range.end; ++k)
      {
            if( clustersize->at(k) <= 0 ) clustersize->at(k) = 1;

            for ( int b = 0; b < nr_channels; b++ )
              kseeds->at(b)[k] = sigma->at(b)[k] / float(clustersize->at(k));

            kseedsx->at(k) = sigmax->at(k) / float(clustersize->at(k));
            kseedsy->at(k) = sigmay->at(k) / float(clustersize->at(k));
      } // end for k
    }
    vector<float>* sigmax;
    vector<float>* sigmay;
    vector<float>* kseedsx;
    vector<float>* kseedsy;
    vector<int>* clustersize;
    vector< vector<float> >* sigma;
    vector< vector<float> >* kseeds;
    int nr_channels;
};

struct SeedsCenters
{
    SeedsCenters( const vector<Mat>& _chvec, const Mat& _klabels,
                  const int _numlabels, const int _nr_channels )
    {
      chvec = _chvec;
      klabels = _klabels;
      numlabels = _numlabels;
      nr_channels = _nr_channels;

      // allocate and init arrays
      sigma.resize(nr_channels);
      for( int b =0 ; b < nr_channels ; b++ )
        sigma[b].assign(numlabels, 0);

      sigmax.assign(numlabels, 0);
      sigmay.assign(numlabels, 0);
      clustersize.assign(numlabels, 0);
    }

    SeedsCenters( const SeedsCenters& counter, Split )
    {
      *this = counter;
      // refill with zero all arrays
      for( int b = 0; b < nr_channels; b++ )
        fill(sigma[b].begin(), sigma[b].end(), 0.0f);

      fill(sigmax.begin(), sigmax.end(), 0.0f);
      fill(sigmay.begin(), sigmay.end(), 0.0f);
      fill(clustersize.begin(), clustersize.end(), 0);
    }

    void operator()( const BlockedRange& range )
    {
      // previous block state
      vector<float> tmp_sigmax = sigmax;
      vector<float> tmp_sigmay = sigmay;
      vector<vector <float> > tmp_sigma = sigma;
      vector<int> tmp_clustersize = clustersize;

      for ( int x = range.begin(); x != range.end(); x++ )
      {
        for( int y = 0; y < chvec[0].rows; y++ )
        {
            int idx = klabels.at<int>(y,x);

            switch ( chvec[0].depth() )
            {
              case CV_8U:
                for( int b = 0; b < nr_channels; b++ )
                  tmp_sigma[b][idx] += chvec[b].at<uchar>(y,x);
                break;

              case CV_8S:
                for( int b = 0; b < nr_channels; b++ )
                  tmp_sigma[b][idx] += chvec[b].at<char>(y,x);
                break;

              case CV_16U:
                for( int b = 0; b < nr_channels; b++ )
                  tmp_sigma[b][idx] += chvec[b].at<ushort>(y,x);
                break;

              case CV_16S:
                for( int b = 0; b < nr_channels; b++ )
                  tmp_sigma[b][idx] += chvec[b].at<short>(y,x);
                break;

              case CV_32S:
                for( int b = 0; b < nr_channels; b++ )
                  tmp_sigma[b][idx] += chvec[b].at<int>(y,x);
                break;

              case CV_32F:
                for( int b = 0; b < nr_channels; b++ )
                  tmp_sigma[b][idx] += chvec[b].at<float>(y,x);
                break;

              case CV_64F:
                for( int b = 0; b < nr_channels; b++ )
                  tmp_sigma[b][idx] += (float) chvec[b].at<double>(y,x);
                break;

              default:
                CV_Error( Error::StsInternal, "Invalid matrix depth" );
                break;
            }

            tmp_sigmax[idx] += x;
            tmp_sigmay[idx] += y;

            tmp_clustersize[idx]++;

        }
      }
      sigma = tmp_sigma;
      sigmax = tmp_sigmax;
      sigmay = tmp_sigmay;
      clustersize = tmp_clustersize;
    }

    void join( SeedsCenters& sc )
    {
      for (int l = 0; l < numlabels; l++)
      {
        sigmax[l] += sc.sigmax[l];
        sigmay[l] += sc.sigmay[l];
        for( int b = 0; b < nr_channels; b++ )
            sigma[b][l] += sc.sigma[b][l];
        clustersize[l] += sc.clustersize[l];
      }
    }

    Mat klabels;
    int numlabels;
    int nr_channels;
    vector<Mat> chvec;
    vector<float> sigmax;
    vector<float> sigmay;
    vector<int> clustersize;
    vector< vector<float> > sigma;
};

struct SLICOGrowInvoker : ParallelLoopBody
{
    SLICOGrowInvoker( vector<Mat>* _chvec, Mat* _distchans, Mat* _distxy, Mat* _distvec,
                      Mat* _klabels, float _kseedsxn, float _kseedsyn, float _xywt,
                      float _maxchansn, vector< vector<float> > *_kseeds,
                      int _x1, int _x2, int _nr_channels, int _n )
    {
      chvec = _chvec;
      distchans = _distchans;
      distxy = _distxy;
      distvec = _distvec;
      kseedsxn = _kseedsxn;
      kseedsyn = _kseedsyn;
      klabels = _klabels;
      maxchansn = _maxchansn;
      kseeds = _kseeds;
      x1 = _x1;
      x2 = _x2;
      n = _n;
      xywt = _xywt;
      nr_channels = _nr_channels;
    }

    void operator ()(const cv::Range& range) const CV_OVERRIDE
    {
      int cols = klabels->cols;
      int rows = klabels->rows;
      for (int y = range.start; y < range.end; ++y)
      {
        for( int x = x1; x < x2; x++ )
        {
          CV_Assert( y < rows && x < cols && y >= 0 && x >= 0 );
          distchans->at<float>(y,x) = 0;

            switch ( chvec->at(0).depth() )
            {
              case CV_8U:
                for( int b = 0; b < nr_channels; b++ )
                {
                  float diff = chvec->at(b).at<uchar>(y,x)
                             - kseeds->at(b)[n];
                  distchans->at<float>(y,x) += diff * diff;
                }
                break;

              case CV_8S:
                for( int b = 0; b < nr_channels; b++ )
                {
                  float diff = chvec->at(b).at<char>(y,x)
                             - kseeds->at(b)[n];
                  distchans->at<float>(y,x) += diff * diff;
                }
                break;

              case CV_16U:
                for( int b = 0; b < nr_channels; b++ )
                {
                  float diff = chvec->at(b).at<ushort>(y,x)
                             - kseeds->at(b)[n];
                  distchans->at<float>(y,x) += diff * diff;
                }
                break;

              case CV_16S:
                for( int b = 0; b < nr_channels; b++ )
                {
                  float diff = chvec->at(b).at<short>(y,x)
                             - kseeds->at(b)[n];
                  distchans->at<float>(y,x) += diff * diff;
                }
                break;

              case CV_32S:
                for( int b = 0; b < nr_channels; b++ )
                {
                  float diff = chvec->at(b).at<int>(y,x)
                             - kseeds->at(b)[n];
                  distchans->at<float>(y,x) += diff * diff;
                }
                break;

              case CV_32F:
                for( int b = 0; b < nr_channels; b++ )
                {
                  float diff = chvec->at(b).at<float>(y,x)
                             - kseeds->at(b)[n];
                  distchans->at<float>(y,x) += diff * diff;
                }
                break;

              case CV_64F:
                for( int b = 0; b < nr_channels; b++ )
                {
                  float diff = float(chvec->at(b).at<double>(y,x)
                             - kseeds->at(b)[n]);
                  distchans->at<float>(y,x) += diff * diff;
                }
                break;

              default:
                CV_Error( Error::StsInternal, "Invalid matrix depth" );
                break;
            }


          float difx = x - kseedsxn;
          float dify = y - kseedsyn;
          distxy->at<float>(y,x) = difx*difx + dify*dify;

          // only varying m, prettier superpixels
          float dist = distchans->at<float>(y,x)
                     / maxchansn + distxy->at<float>(y,x)/xywt;

          if( dist < distvec->at<float>(y,x) )
          {
            distvec->at<float>(y,x) = dist;
            klabels->at<int>(y,x) = n;
          }
        } // end for x
      } // end for y
    }

    Mat* klabels;
    vector< vector<float> > *kseeds;
    float maxchansn, xywt;
    vector<Mat>* chvec;
    Mat *distchans, *distxy, *distvec;
    float kseedsxn, kseedsyn;
    int x1, x2, nr_channels, n;
};

/*
 *
 *    Magic SLIC - no parameters
 *
 *    Performs k mean segmentation. It is fast because it looks locally, not
 * over the entire image.
 * This function picks the maximum value of color distance as compact factor
 * M and maximum pixel distance as grid step size S from each cluster (13 April 2011).
 * So no need to input a constant value of M and S. There are two clear
 * advantages:
 *
 * [1] The algorithm now better handles both textured and non-textured regions
 * [2] There is not need to set any parameters!!!
 *
 * SLICO (or SLIC Zero) dynamically varies only the compactness factor S,
 * not the step size S.
 *
 */
inline void SuperpixelSLICImpl::PerformSLICO( const int&  itrnum )
{
    Mat distxy( m_height, m_width, CV_32F, Scalar::all(FLT_MAX) );
    Mat distvec( m_height, m_width, CV_32F, Scalar::all(FLT_MAX) );
    Mat distchans( m_height, m_width, CV_32F, Scalar::all(FLT_MAX) );

    // this is the variable value of M, just start with 10
    vector<float> maxchans( m_numlabels, FLT_MIN );
    // this is the variable value of M, just start with 10
    vector<float> maxxy( m_numlabels, FLT_MIN );
    // note: this is different from how usual SLIC/LKM works
    const float xywt = float(m_region_size*m_region_size);

    for( int itr = 0; itr < itrnum; itr++ )
    {
        distvec.setTo(FLT_MAX);
        for( int n = 0; n < m_numlabels; n++ )
        {
            int y1 = max(0, (int) m_kseedsy[n] - m_region_size);
            int y2 = min(m_height, (int) m_kseedsy[n] + m_region_size);
            int x1 = max(0, (int) m_kseedsx[n] - m_region_size);
            int x2 = min((int) m_width,(int) m_kseedsx[n] + m_region_size);

            parallel_for_( Range(y1, y2), SLICOGrowInvoker( &m_chvec, &distchans, &distxy, &distvec,
                           &m_klabels, m_kseedsx[n], m_kseedsy[n], xywt, maxchans[n], &m_kseeds,
                           x1, x2, m_nr_channels, n ) );
        }
        //-----------------------------------------------------------------
        // Assign the max color distance for a cluster
        //-----------------------------------------------------------------
        if( itr == 0 )
        {
            maxchans.assign(m_numlabels,FLT_MIN);
            maxxy.assign(m_numlabels,FLT_MIN);
        }

        for( int x = 0; x < m_width; x++ )
        {
          for( int y = 0; y < m_height; y++ )
          {
              int idx = m_klabels.at<int>(y,x);

              if( maxchans[idx] < distchans.at<float>(y,x) )
                  maxchans[idx] = distchans.at<float>(y,x);

              if( maxxy[idx] < distxy.at<float>(y,x) )
                  maxxy[idx] = distxy.at<float>(y,x);
          }
        }
        //-----------------------------------------------------------------
        // Recalculate the centroid and store in the seed values
        //-----------------------------------------------------------------

        // parallel reduce structure
        SeedsCenters sc( m_chvec, m_klabels, m_numlabels, m_nr_channels );

        // accumulate center distances
        parallel_reduce( BlockedRange(0, m_width), sc );

        // normalize centers
        parallel_for_( Range(0, m_numlabels), SeedNormInvoker( &m_kseeds, &sc.sigma,
                       &sc.clustersize, &sc.sigmax, &sc.sigmay, &m_kseedsx, &m_kseedsy, m_nr_channels  ) );

    }
}

struct SLICGrowInvoker : ParallelLoopBody
{
    SLICGrowInvoker( vector<Mat>* _chvec, Mat* _distvec, Mat* _klabels,
                     float _kseedsxn, float _kseedsyn, float _xywt,
                     vector< vector<float> > *_kseeds, int _x1, int _x2,
                     int _nr_channels, int _n )
    {
      chvec = _chvec;
      distvec = _distvec;
      kseedsxn = _kseedsxn;
      kseedsyn = _kseedsyn;
      klabels = _klabels;
      kseeds = _kseeds;
      x1 = _x1;
      x2 = _x2;
      n = _n;
      xywt = _xywt;
      nr_channels = _nr_channels;
    }

    void operator ()(const cv::Range& range) const CV_OVERRIDE
    {
      for (int y = range.start; y < range.end; ++y)
      {
        for( int x = x1; x < x2; x++ )
        {
          float dist = 0;

          switch ( chvec->at(0).depth() )
          {
            case CV_8U:
              for( int b = 0; b < nr_channels; b++ )
              {
                float diff = chvec->at(b).at<uchar>(y,x) - kseeds->at(b)[n];
                dist += diff * diff;
              }
              break;

            case CV_8S:
              for( int b = 0; b < nr_channels; b++ )
              {
                float diff = chvec->at(b).at<char>(y,x) - kseeds->at(b)[n];
                dist += diff * diff;
              }
              break;

            case CV_16U:
              for( int b = 0; b < nr_channels; b++ )
              {
                float diff = chvec->at(b).at<ushort>(y,x) - kseeds->at(b)[n];
                dist += diff * diff;
              }
              break;

            case CV_16S:
              for( int b = 0; b < nr_channels; b++ )
              {
                float diff = chvec->at(b).at<short>(y,x) - kseeds->at(b)[n];
                dist += diff * diff;
              }
              break;

            case CV_32S:
              for( int b = 0; b < nr_channels; b++ )
              {
                float diff = chvec->at(b).at<int>(y,x) - kseeds->at(b)[n];
                dist += diff * diff;
              }
              break;

            case CV_32F:
              for( int b = 0; b < nr_channels; b++ )
              {
                float diff = chvec->at(b).at<float>(y,x) - kseeds->at(b)[n];
                dist += diff * diff;
              }
              break;

            case CV_64F:
              for( int b = 0; b < nr_channels; b++ )
              {
                float diff = float(chvec->at(b).at<double>(y,x) - kseeds->at(b)[n]);
                dist += diff * diff;
              }
              break;

            default:
              CV_Error( Error::StsInternal, "Invalid matrix depth" );
              break;
          }

          float difx = x - kseedsxn;
          float dify = y - kseedsyn;
          float distxy = difx*difx + dify*dify;

          dist += distxy / xywt;

          //this would be more exact but expensive
          //dist = sqrt(dist) + sqrt(distxy/xywt);

          if( dist < distvec->at<float>(y,x) )
          {
            distvec->at<float>(y,x) = dist;
            klabels->at<int>(y,x) = n;
          }
        } //end for x
      } // end for y
    }

    Mat* klabels;
    vector< vector<float> > *kseeds;
    float xywt;
    vector<Mat>* chvec;
    Mat *distvec;
    float kseedsxn, kseedsyn;
    int x1, x2, nr_channels, n;
};

/*
 *    PerformSuperpixelSLIC
 *
 *    Performs k mean segmentation. It is fast because it looks locally, not
 * over the entire image.
 *
 */
inline void SuperpixelSLICImpl::PerformSLIC( const int&  itrnum )
{
    Mat distvec( m_height, m_width, CV_32F );

    const float xywt = (m_region_size/m_ruler)*(m_region_size/m_ruler);

    for( int itr = 0; itr < itrnum; itr++ )
    {
        distvec.setTo(FLT_MAX);
        for( int n = 0; n < m_numlabels; n++ )
        {
            int y1 = max(0, (int) m_kseedsy[n] - m_region_size);
            int y2 = min(m_height, (int) m_kseedsy[n] + m_region_size);
            int x1 = max(0, (int) m_kseedsx[n] - m_region_size);
            int x2 = min((int) m_width,(int) m_kseedsx[n] + m_region_size);

            parallel_for_( Range(y1, y2), SLICGrowInvoker( &m_chvec, &distvec,
                           &m_klabels, m_kseedsx[n], m_kseedsy[n], xywt, &m_kseeds,
                           x1, x2, m_nr_channels, n ) );
        }

        //-----------------------------------------------------------------
        // Recalculate the centroid and store in the seed values
        //-----------------------------------------------------------------
        // instead of reassigning memory on each iteration, just reset.

        // parallel reduce structure
        SeedsCenters sc( m_chvec, m_klabels, m_numlabels, m_nr_channels );

        // accumulate center distances
        parallel_reduce( BlockedRange(0, m_width), sc );

        // normalize centers
        parallel_for_( Range(0, m_numlabels), SeedNormInvoker( &m_kseeds, &sc.sigma,
                       &sc.clustersize, &sc.sigmax, &sc.sigmay, &m_kseedsx, &m_kseedsy, m_nr_channels  ) );

    }
}

/*
 *    PerformSuperpixelMSLIC
 *
 *
 */
inline void SuperpixelSLICImpl::PerformMSLIC( const int&  itrnum )
{
    vector< vector<float> > sigma(m_nr_channels);
    for( int b = 0; b < m_nr_channels; b++ )
      sigma[b].resize(m_numlabels, 0);

    Mat distvec( m_height, m_width, CV_32F );

    const float xywt = (m_region_size/m_ruler)*(m_region_size/m_ruler);

    int offset = m_region_size;

    // from paper
    m_split = 4.0f;
    m_ratio = 5.0f;

    for( int itr = 0; itr < itrnum; itr++ )
    {
        m_cur_iter = itr;

        distvec.setTo(FLT_MAX);
        for( int n = 0; n < m_numlabels; n++ )
        {
            if ( m_adaptk[n] < 1.0f )
                offset = int(m_region_size * m_adaptk[n]);
            else
                offset = int(m_region_size * m_adaptk[n]);

            int y1 = max(0,        (int) m_kseedsy[n] - offset);
            int y2 = min(m_height, (int) m_kseedsy[n] + offset);
            int x1 = max(0,        (int) m_kseedsx[n] - offset);
            int x2 = min(m_width,  (int) m_kseedsx[n] + offset);

            parallel_for_( Range(y1, y2), SLICGrowInvoker( &m_chvec, &distvec,
                           &m_klabels, m_kseedsx[n], m_kseedsy[n], xywt, &m_kseeds,
                           x1, x2, m_nr_channels, n ) );
        }

        //-----------------------------------------------------------------
        // Recalculate the centroid and store in the seed values
        //-----------------------------------------------------------------
        // instead of reassigning memory on each iteration, just reset.

        // parallel reduce structure
        SeedsCenters sc( m_chvec, m_klabels, m_numlabels, m_nr_channels );

        // accumulate center distances
        parallel_reduce( BlockedRange(0, m_width), sc );

        // normalize centers
        parallel_for_( Range(0, m_numlabels), SeedNormInvoker( &m_kseeds, &sc.sigma,
                       &sc.clustersize, &sc.sigmax, &sc.sigmay, &m_kseedsx, &m_kseedsy, m_nr_channels ) );

        // 13% as in original paper
        enforceLabelConnectivity( 13 );
        SuperpixelSplit();
    }
}

inline void SuperpixelSLICImpl::SuperpixelSplit()
{
    Mat klabels = m_klabels.clone();

    // parallel reduce structure
    SeedsCenters msc( m_chvec, m_klabels, m_numlabels, m_nr_channels );

    // accumulate center distances
    parallel_reduce( BlockedRange(0, m_width), msc );

    const float invwt = 1.0f / ( (m_region_size/m_ruler)*(m_region_size/m_ruler) );
    const float sqrt_invwt = sqrt(invwt);

    if ( m_cur_iter < m_iterations - 2 )
    {
        vector<float> avglabs( m_numlabels, 0 );
        for( int y = 0; y < m_height - 1; y++ )
        {
            for( int x = 0; x < m_width - 1; x++ )
            {
                if ( klabels.at<int>( y, x ) == klabels.at<int>( y+1, x ) &&
                     klabels.at<int>( y, x ) == klabels.at<int>( y, x+1 ) )
                {
                    float x1 = 1, y1 = 0;
                    float x2 = 0, y2 = 1;

                    vector<float> ch1(m_nr_channels);
                    vector<float> ch2(m_nr_channels);

                    switch ( m_chvec.at(0).depth() )
                    {
                      case CV_8U:
                        for( int c = 0; c < m_nr_channels; c++ )
                        {
                            ch1[c] = float( m_chvec[c].at<uchar>( y+1, x   )
                                          - m_chvec[c].at<uchar>( y,   x   ) );
                            ch2[c] = float( m_chvec[c].at<uchar>( y,   x+1 )
                                          - m_chvec[c].at<uchar>( y,   x   ) );

                            ch1[c] /= sqrt_invwt;
                            ch2[c] /= sqrt_invwt;
                        }
                        break;

                      case CV_8S:
                        for( int c = 0; c < m_nr_channels; c++ )
                        {
                            ch1[c] = float( m_chvec[c].at<char>( y+1, x   )
                                          - m_chvec[c].at<char>( y,   x   ) );
                            ch2[c] = float( m_chvec[c].at<char>( y,   x+1 )
                                          - m_chvec[c].at<char>( y,   x   ) );

                            ch1[c] /= sqrt_invwt;
                            ch2[c] /= sqrt_invwt;
                        }
                        break;

                      case CV_16U:
                        for( int c = 0; c < m_nr_channels; c++ )
                        {
                            ch1[c] = float( m_chvec[c].at<ushort>( y+1, x   )
                                          - m_chvec[c].at<ushort>( y,   x   ) );
                            ch2[c] = float( m_chvec[c].at<ushort>( y,   x+1 )
                                          - m_chvec[c].at<ushort>( y,   x   ) );

                            ch1[c] /= sqrt_invwt;
                            ch2[c] /= sqrt_invwt;
                        }
                        break;

                      case CV_16S:
                        for( int c = 0; c < m_nr_channels; c++ )
                        {
                            ch1[c] = float( m_chvec[c].at<short>( y+1, x   )
                                          - m_chvec[c].at<short>( y,   x   ) );
                            ch2[c] = float( m_chvec[c].at<short>( y,   x+1 )
                                          - m_chvec[c].at<short>( y,   x   ) );

                            ch1[c] /= sqrt_invwt;
                            ch2[c] /= sqrt_invwt;
                        }
                        break;

                      case CV_32S:
                        for( int c = 0; c < m_nr_channels; c++ )
                        {
                            ch1[c] = float( m_chvec[c].at<int>( y+1, x   )
                                          - m_chvec[c].at<int>( y,   x   ) );
                            ch2[c] = float( m_chvec[c].at<int>( y,   x+1 )
                                          - m_chvec[c].at<int>( y,   x   ) );

                            ch1[c] /= sqrt_invwt;
                            ch2[c] /= sqrt_invwt;
                        }
                        break;

                      case CV_32F:
                        for( int c = 0; c < m_nr_channels; c++ )
                        {
                            ch1[c] = m_chvec[c].at<float>( y+1, x   )
                                   - m_chvec[c].at<float>( y,   x   );
                            ch2[c] = m_chvec[c].at<float>( y,   x+1 )
                                   - m_chvec[c].at<float>( y,   x   );

                            ch1[c] /= sqrt_invwt;
                            ch2[c] /= sqrt_invwt;
                        }
                        break;

                      case CV_64F:
                        for( int c = 0; c < m_nr_channels; c++ )
                        {
                            ch1[c] = float( m_chvec[c].at<double>( y+1, x   )
                                          - m_chvec[c].at<double>( y,   x   ) );
                            ch2[c] = float( m_chvec[c].at<double>( y,   x+1 )
                                          - m_chvec[c].at<double>( y,   x   ) );

                            ch1[c] /= sqrt_invwt;
                            ch2[c] /= sqrt_invwt;
                        }
                        break;

                      default:
                        CV_Error( Error::StsInternal, "Invalid matrix depth" );
                        break;
                    }
                    float ch11sqsum = 0.0f;
                    float ch12sqsum = 0.0f;
                    float ch22sqsum = 0.0f;
                    for( int c = 0; c < m_nr_channels; c++ )
                    {
                       ch11sqsum += ch1[c]*ch1[c];
                       ch12sqsum += ch1[c]*ch2[c];
                       ch22sqsum += ch2[c]*ch2[c];
                    }

                    // adjacent metric for N channels
                    avglabs[ klabels.at<int>(y,x) ]
                              += sqrt(   (x1*x1 + y1*y1 + ch11sqsum) * (x2*x2 + y2*y2 + ch22sqsum)
                                       - (x1*x2 + y1*y2 + ch12sqsum) * (x1*x2 + y1*y2 + ch12sqsum) );
                }
             }
        }
        for ( int i = 0; i < m_numlabels; i++ )
        {
            avglabs[i] /= m_region_size * m_region_size;
        }

        m_kseedsx.clear();
        m_kseedsy.clear();
        m_kseedsx.resize( m_numlabels, 0 );
        m_kseedsy.resize( m_numlabels, 0 );
        for( int c = 0; c < m_nr_channels; c++ )
        {
          m_kseeds[c].clear();
          m_kseeds[c].resize( m_numlabels, 0 );
        }

        for( int k = 0; k < m_numlabels; k++ )
        {
            m_kseedsx[k] = msc.sigmax[k] / msc.clustersize[k];
            m_kseedsy[k] = msc.sigmay[k] / msc.clustersize[k];
            for( int c = 0; c < m_nr_channels; c++ )
                m_kseeds[c][k] = msc.sigma[c][k] / msc.clustersize[k];
        }

        for( int k = 0; k < m_numlabels; k++ )
        {
            int xindex = 0, yindex = 0;
            if ( ( m_adaptk[k] <= 0.5f ) ||
                 ( avglabs[k] < (m_split * m_ratio) ) )
            {
                m_kseedsx[k] = msc.sigmax[k] / msc.clustersize[k];
                m_kseedsy[k] = msc.sigmay[k] / msc.clustersize[k];
                for( int c = 0; c < m_nr_channels; c++ )
                  m_kseeds[c][k] = msc.sigma[c][k] / msc.clustersize[k];

                m_adaptk[k] = sqrt( m_ratio / avglabs[k] );
                m_adaptk[k] = max( 0.5f, m_adaptk[k] );
                m_adaptk[k] = min( 2.0f, m_adaptk[k] );
            }
            // if segment size is too large
            // split it and calculate four new seeds
            else
            {
                xindex = (int)( msc.sigmax[k] / msc.clustersize[k] );
                yindex = (int)( msc.sigmay[k] / msc.clustersize[k] );
                m_adaptk[k] = max( 0.5f, m_adaptk[k] / 2 );

                const float minadaptk = min( 1.0f, m_adaptk[k] ) * m_region_size / 2;

                int x1 = (int)( xindex - minadaptk );
                int x2 = (int)( xindex + minadaptk );
                int x3 = (int)( xindex - minadaptk );
                int x4 = (int)( xindex + minadaptk );

                int y1 = (int)( yindex + minadaptk );
                int y2 = (int)( yindex + minadaptk );
                int y3 = (int)( yindex - minadaptk );
                int y4 = (int)( yindex - minadaptk );

                if ( x1 < 0         ) x1 = 0;
                if ( x2 >= m_width  ) x2 = m_width  - 1;
                if ( x3 < 0         ) x3 = 0;
                if ( x4 >= m_width  ) x4 = m_width  - 1;
                if ( y1 >= m_height ) y1 = m_height - 1;
                if ( y2 >= m_height ) y2 = m_height - 1;
                if ( y3 < 0         ) y3 = 0;
                if ( y4 < 0         ) y4 = 0;

                m_kseedsx[k] = (float)x1;
                m_kseedsy[k] = (float)y1;
                for( int c = 0; c < m_nr_channels; c++ )
                {
                    switch ( m_chvec[c].depth() )
                    {
                      case CV_8U:
                        m_kseeds[c][k] = m_chvec[c].at<uchar>(y1,x1);
                        break;

                      case CV_8S:
                        m_kseeds[c][k] = m_chvec[c].at<char>(y1,x1);
                        break;

                      case CV_16U:
                        m_kseeds[c][k] = m_chvec[c].at<ushort>(y1,x1);
                        break;

                      case CV_16S:
                        m_kseeds[c][k] = m_chvec[c].at<short>(y1,x1);
                        break;

                      case CV_32S:
                        m_kseeds[c][k] = float(m_chvec[c].at<int>(y1,x1));
                        break;

                      case CV_32F:
                        m_kseeds[c][k] = m_chvec[c].at<float>(y1,x1);
                        break;

                      case CV_64F:
                        m_kseeds[c][k] = float(m_chvec[c].at<double>(y1,x1));
                        break;

                      default:
                        CV_Error( Error::StsInternal, "Invalid matrix depth" );
                        break;
                    }
                }

                m_kseedsx.push_back( (float)x2 );
                m_kseedsx.push_back( (float)x3 );
                m_kseedsx.push_back( (float)x4 );
                m_kseedsy.push_back( (float)y2 );
                m_kseedsy.push_back( (float)y3 );
                m_kseedsy.push_back( (float)y4 );

                for( int c = 0; c < m_nr_channels; c++ )
                {
                    switch ( m_chvec[c].depth() )
                    {
                      case CV_8U:
                        m_kseeds[c].push_back( m_chvec[c].at<uchar>(y2,x2) );
                        m_kseeds[c].push_back( m_chvec[c].at<uchar>(y3,x3) );
                        m_kseeds[c].push_back( m_chvec[c].at<uchar>(y4,x4) );
                        break;

                      case CV_8S:
                        m_kseeds[c].push_back( m_chvec[c].at<char>(y2,x2) );
                        m_kseeds[c].push_back( m_chvec[c].at<char>(y3,x3) );
                        m_kseeds[c].push_back( m_chvec[c].at<char>(y4,x4) );
                        break;

                      case CV_16U:
                        m_kseeds[c].push_back( m_chvec[c].at<ushort>(y2,x2) );
                        m_kseeds[c].push_back( m_chvec[c].at<ushort>(y3,x3) );
                        m_kseeds[c].push_back( m_chvec[c].at<ushort>(y4,x4) );
                        break;

                      case CV_16S:
                        m_kseeds[c].push_back( m_chvec[c].at<short>(y2,x2) );
                        m_kseeds[c].push_back( m_chvec[c].at<short>(y3,x3) );
                        m_kseeds[c].push_back( m_chvec[c].at<short>(y4,x4) );
                        break;

                      case CV_32S:
                        m_kseeds[c].push_back( float(m_chvec[c].at<int>(y2,x2)) );
                        m_kseeds[c].push_back( float(m_chvec[c].at<int>(y3,x3)) );
                        m_kseeds[c].push_back( float(m_chvec[c].at<int>(y4,x4)) );
                        break;

                      case CV_32F:
                        m_kseeds[c].push_back( m_chvec[c].at<float>(y2,x2) );
                        m_kseeds[c].push_back( m_chvec[c].at<float>(y3,x3) );
                        m_kseeds[c].push_back( m_chvec[c].at<float>(y4,x4) );
                        break;

                      case CV_64F:
                        m_kseeds[c].push_back( float(m_chvec[c].at<double>(y2,x2)) );
                        m_kseeds[c].push_back( float(m_chvec[c].at<double>(y3,x3)) );
                        m_kseeds[c].push_back( float(m_chvec[c].at<double>(y4,x4)) );
                        break;

                      default:
                        CV_Error( Error::StsInternal, "Invalid matrix depth" );
                        break;
                    }
                }
                m_adaptk.push_back( m_adaptk[k] );
                m_adaptk.push_back( m_adaptk[k] );
                m_adaptk.push_back( m_adaptk[k] );
                msc.clustersize.push_back( 1 );
                msc.clustersize.push_back( 1 );
                msc.clustersize.push_back( 1 );
            }
        }
    }
    else
    {
        m_kseedsx.clear();
        m_kseedsy.clear();
        m_kseedsx.resize( m_numlabels, 0 );
        m_kseedsy.resize( m_numlabels, 0 );
        for( int c = 0; c < m_nr_channels; c++ )
        {
          m_kseeds[c].clear();
          m_kseeds[c].resize( m_numlabels, 0 );
        }

        for( int k = 0; k < m_numlabels; k++ )
        {
            m_kseedsx[k] = msc.sigmax[k] / msc.clustersize[k];
            m_kseedsy[k] = msc.sigmay[k] / msc.clustersize[k];
            for( int c = 0; c < m_nr_channels; c++ )
                m_kseeds[c][k] = msc.sigma[c][k] / msc.clustersize[k];
        }
    }

  m_klabels.release();
  m_klabels = klabels.clone();

  // re-update amount of labels
  m_numlabels = (int)m_kseeds[0].size();

}


} // namespace ximgproc
} // namespace cv
