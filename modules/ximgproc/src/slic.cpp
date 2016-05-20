/*********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2013
 * Radhakrishna Achanta
 * email : Radhakrishna [dot] Achanta [at] epfl [dot] ch
 * web : http://ivrl.epfl.ch/people/achanta
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

    virtual ~SuperpixelSLICImpl();

    // perform amount of iteration
    virtual void iterate( int num_iterations = 10 );

    // get amount of superpixels
    virtual int getNumberOfSuperpixels() const;

    // get image with labels
    virtual void getLabels( OutputArray labels_out ) const;

    // get mask image with contour
    virtual void getLabelContourMask( OutputArray image, bool thick_line = true ) const;

    // enforce connectivity over labels
    virtual void enforceLabelConnectivity( int min_element_size = 25 );


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

    // storage for edge magnitudes
    Mat edgemag = Mat( m_height, m_width, CV_32F, Scalar::all(0) );

    // perturb seeds is not absolutely necessary,
    // one can set this flag to false
    bool perturbseeds = true;

    if ( perturbseeds ) DetectChEdges( edgemag );

    if( m_algorithm == SLICO )
      GetChSeedsK();
    else if( m_algorithm == SLIC )
      GetChSeedsS();
    else
      CV_Error( Error::StsInternal, "No such algorithm" );

    // update amount of labels now
    m_numlabels = (int)m_kseeds[0].size();

    // perturb seeds given edges
    if ( perturbseeds ) PerturbSeeds( edgemag );

}

void SuperpixelSLICImpl::iterate( int num_iterations )
{
    if( m_algorithm == SLICO )
      PerformSLICO( num_iterations );
    else if( m_algorithm == SLIC )
      PerformSLIC( num_iterations );
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
                //-------------------------------------------------------
                // Quickly find an adjacent label for use later if needed
                //-------------------------------------------------------
                for( int n = 0; n < 4; n++ )
                {
                    int x = xvec[0] + dx4[n];
                    int y = yvec[0] + dy4[n];
                    if( (x >= 0 && x < m_width) && (y >= 0 && y < m_height) )
                    {
                        if(nlabels.at<int>(y,x) != INT_MAX)
                          adjlabel = nlabels.at<int>(y,x);
                    }
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
                //-------------------------------------------------------
                // If segment size is less then a limit, assign an
                // adjacent label found before, and decrement label count.
                //-------------------------------------------------------
                if(count <= min_sp_sz)
                {
                    for( int c = 0; c < count; c++ )
                    {
                        nlabels.at<int>(yvec[c],xvec[c]) = adjlabel;
                    }
                    label--;
                }
                label++;
            }
        }
    }
    // replace old
    m_klabels = nlabels;
    m_numlabels = label;
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
        S_dx = S_dx + dx.mul(dx);
        S_dy = S_dy + dy.mul(dy);

    }
    // total magnitude
    edgemag += S_dx + S_dy;
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

    void operator ()(const cv::Range& range) const
    {
      for (int k = range.start; k < range.end; ++k)
      {
            if( clustersize->at(k) <= 0 ) clustersize->at(k) = 1;

            for ( int b = 0; b < nr_channels; b++ )
              kseeds->at(b)[k] = sigma->at(b)[k] / float(clustersize->at(k));;

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

    void ClearArrays( )
    {
      // refill with zero all arrays
      for( int b = 0; b < nr_channels; b++ )
        fill(sigma[b].begin(), sigma[b].end(), 0.0f);

      fill(sigmax.begin(), sigmax.end(), 0.0f);
      fill(sigmay.begin(), sigmay.end(), 0.0f);
      fill(clustersize.begin(), clustersize.end(), 0);
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

    void operator ()(const cv::Range& range) const
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
    float xywt = float(m_region_size*m_region_size);

    // parallel reduce structure
    SeedsCenters sc( m_chvec, m_klabels, m_numlabels, m_nr_channels );

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

        // accumulate center distances
        parallel_reduce( BlockedRange(0, m_width), sc );

        // normalize centers
        parallel_for_( Range(0, m_numlabels), SeedNormInvoker( &m_kseeds, &sc.sigma,
                       &sc.clustersize, &sc.sigmax, &sc.sigmay, &m_kseedsx, &m_kseedsy, m_nr_channels  ) );

        // refill arrays
        sc.ClearArrays();
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

    void operator ()(const cv::Range& range) const
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
    vector< vector<float> > sigma(m_nr_channels);
    for( int b = 0; b < m_nr_channels; b++ )
      sigma[b].resize(m_numlabels, 0);

    vector<float> sigmax(m_numlabels, 0);
    vector<float> sigmay(m_numlabels, 0);
    vector<int> clustersize(m_numlabels, 0);

    Mat distvec( m_height, m_width, CV_32F );

    float xywt = (m_region_size/m_ruler)*(m_region_size/m_ruler);

    // parallel reduce structure
    SeedsCenters sc( m_chvec, m_klabels, m_numlabels, m_nr_channels );

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

        // accumulate center distances
        parallel_reduce( BlockedRange(0, m_width), sc );

        // normalize centers
        parallel_for_( Range(0, m_numlabels), SeedNormInvoker( &m_kseeds, &sigma,
                       &clustersize, &sigmax, &sigmay, &m_kseedsx, &m_kseedsy, m_nr_channels  ) );

        // refill arrays
        sc.ClearArrays();
    }
}

} // namespace ximgproc
} // namespace cv
