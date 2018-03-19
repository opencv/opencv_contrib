/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2014, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "../precomp.hpp"

#include "kyheader.hpp"
#include "CmTimer.hpp"
#include "CmFile.hpp"

namespace cv
{
namespace saliency
{

/**
 * BING Objectness
 */

const char* ObjectnessBING::_clrName[3] =
{ "MAXBGR", "HSV", "I" };

ObjectnessBING::ObjectnessBING()
{
  _base = 2;  // base for window size quantization
  _W = 8;  // feature window size (W, W)
  _NSS = 2;  //non-maximal suppress size NSS
  _logBase = log( _base );
  _minT = cvCeil( log( 10. ) / _logBase );
  _maxT = cvCeil( log( 500. ) / _logBase );
  _numT = _maxT - _minT + 1;
  _Clr = MAXBGR;

  setColorSpace( _Clr );

  className = "BING";
}

ObjectnessBING::~ObjectnessBING()
{

}

void ObjectnessBING::setColorSpace( int clr )
{
  _Clr = clr;
  _modelName = _trainingPath + "/" + std::string( format( "ObjNessB%gW%d%s", _base, _W, _clrName[_Clr] ).c_str() );
  _bbResDir = _resultsDir + "/" + std::string( format( "BBoxesB%gW%d%s/", _base, _W, _clrName[_Clr] ).c_str() );
}

void ObjectnessBING::setTrainingPath( const String& trainingPath )
{
  _trainingPath = trainingPath;
}

void ObjectnessBING::setBBResDir(const String &resultsDir )
{
  _resultsDir = resultsDir;
}

int ObjectnessBING::loadTrainedModel()  // Return -1, 0, or 1 if partial, none, or all loaded
{
  CStr s1 = _modelName + ".wS1", s2 = _modelName + ".wS2", sI = _modelName + ".idx";
  Mat filters1f, reW1f, idx1i, show3u;

  if( !matRead( s1, filters1f ) || !matRead( sI, idx1i ) )
  {
    printf( "Can't load model: %s or %s\r\n", s1.c_str(), sI.c_str() );
    return 0;
  }


  normalize( filters1f, show3u, 1, 255, NORM_MINMAX, CV_8U );
  _tigF.update( filters1f );

  _svmSzIdxs = idx1i;
  CV_Assert( _svmSzIdxs.size() > 1 && filters1f.size() == Size(_W, _W) && filters1f.type() == CV_32F );
  _svmFilter = filters1f;

  if( !matRead( s2, _svmReW1f ) || _svmReW1f.size() != Size( 2, (int) _svmSzIdxs.size() ) )
  {
    _svmReW1f = Mat();
    return -1;
  }
  return 1;
}

void ObjectnessBING::predictBBoxSI( Mat &img3u, ValStructVec<float, Vec4i> &valBoxes, std::vector<int> &sz, int NUM_WIN_PSZ, bool fast )
{
  const int numSz = (int) _svmSzIdxs.size();
  const int imgW = img3u.cols, imgH = img3u.rows;
  valBoxes.reserve( 10000 );
  sz.clear();
  sz.reserve( 10000 );
  for ( int ir = numSz - 1; ir >= 0; ir-- )
  {
    int r = _svmSzIdxs[ir];
    int height = cvRound( pow( _base, r / _numT + _minT ) ), width = cvRound( pow( _base, r % _numT + _minT ) );
    if( height > imgH * _base || width > imgW * _base )
      continue;

    height = min( height, imgH ), width = min( width, imgW );
    Mat im3u, matchCost1f, mag1u;
    resize( img3u, im3u, Size( cvRound( _W * imgW * 1.0 / width ), cvRound( _W * imgH * 1.0 / height ) ), 0, 0, INTER_LINEAR_EXACT );
    gradientMag( im3u, mag1u );

    matchCost1f = _tigF.matchTemplate( mag1u );

    ValStructVec<float, Point> matchCost;
    nonMaxSup( matchCost1f, matchCost, _NSS, NUM_WIN_PSZ, fast );

    // Find true locations and match values
    double ratioX = width / _W, ratioY = height / _W;
    int iMax = min( matchCost.size(), NUM_WIN_PSZ );
    for ( int i = 0; i < iMax; i++ )
    {
      float mVal = matchCost( i );
      Point pnt = matchCost[i];
      Vec4i box( cvRound( pnt.x * ratioX ), cvRound( pnt.y * ratioY ) );
      box[2] = cvRound( min( box[0] + width, imgW ) );
      box[3] = cvRound( min( box[1] + height, imgH ) );
      box[0]++;
      box[1]++;
      valBoxes.pushBack( mVal, box );
      sz.push_back( ir );
    }
  }

}

void ObjectnessBING::predictBBoxSII( ValStructVec<float, Vec4i> &valBoxes, const std::vector<int> &sz )
{
  int numI = valBoxes.size();
  for ( int i = 0; i < numI; i++ )
  {
    const float* svmIIw = _svmReW1f.ptr<float>( sz[i] );
    valBoxes( i ) = valBoxes( i ) * svmIIw[0] + svmIIw[1];
  }
  //valBoxes.sort();
  // Descending order. At the top there are the values with higher
  // values, ie more likely to have objects in the their corresponding rectangles.
  valBoxes.sort( true );
}

// Get potential bounding boxes, each of which is represented by a Vec4i for (minX, minY, maxX, maxY).
// The trained model should be prepared before calling this function: loadTrainedModel() or trainStageI() + trainStageII().
// Use numDet to control the final number of proposed bounding boxes, and number of per size (scale and aspect ratio)
void ObjectnessBING::getObjBndBoxes( Mat &img3u, ValStructVec<float, Vec4i> &valBoxes, int numDetPerSize )
{
  //CV_Assert_(filtersLoaded() , ("SVM filters should be initialized before getting object proposals\n"));
  vecI sz;
  predictBBoxSI( img3u, valBoxes, sz, numDetPerSize, false );
  predictBBoxSII( valBoxes, sz );
  return;
}

void ObjectnessBING::nonMaxSup( Mat &matchCost1f, ValStructVec<float, Point> &matchCost, int NSS, int maxPoint, bool fast )
{
  const int _h = matchCost1f.rows, _w = matchCost1f.cols;
  Mat isMax1u = Mat::ones( _h, _w, CV_8U ), costSmooth1f;
  ValStructVec<float, Point> valPnt;
  matchCost.reserve( _h * _w );
  valPnt.reserve( _h * _w );
  if( fast )
  {
    blur( matchCost1f, costSmooth1f, Size( 3, 3 ) );
    for ( int r = 0; r < _h; r++ )
    {
      const float* d = matchCost1f.ptr<float>( r );
      const float* ds = costSmooth1f.ptr<float>( r );
      for ( int c = 0; c < _w; c++ )
        if( d[c] >= ds[c] )
          valPnt.pushBack( d[c], Point( c, r ) );
    }
  }
  else
  {
    for ( int r = 0; r < _h; r++ )
    {
      const float* d = matchCost1f.ptr<float>( r );
      for ( int c = 0; c < _w; c++ )
        valPnt.pushBack( d[c], Point( c, r ) );
    }
  }

  valPnt.sort();
  for ( int i = 0; i < valPnt.size(); i++ )
  {
    Point &pnt = valPnt[i];
    if( isMax1u.at<BYTE>( pnt ) )
    {
      matchCost.pushBack( valPnt( i ), pnt );
      for ( int dy = -NSS; dy <= NSS; dy++ )
        for ( int dx = -NSS; dx <= NSS; dx++ )
        {
          Point neighbor = pnt + Point( dx, dy );
          if( !CHK_IND( neighbor ) )
            continue;
          isMax1u.at<BYTE>( neighbor ) = false;
        }
    }
    if( matchCost.size() >= maxPoint )
      return;
  }
}

void ObjectnessBING::gradientMag( Mat &imgBGR3u, Mat &mag1u )
{
  switch ( _Clr )
  {
    case MAXBGR:
      gradientRGB( imgBGR3u, mag1u );
      break;
    case G:
      gradientGray( imgBGR3u, mag1u );
      break;
    case HSV:
      gradientHSV( imgBGR3u, mag1u );
      break;
    default:
      printf( "Error: not recognized color space\n" );
  }
}

void ObjectnessBING::gradientRGB( Mat &bgr3u, Mat &mag1u )
{
  const int H = bgr3u.rows, W = bgr3u.cols;
  Mat Ix( H, W, CV_32S ), Iy( H, W, CV_32S );

  // Left/right most column Ix
  for ( int y = 0; y < H; y++ )
  {
    Ix.at<int>( y, 0 ) = bgrMaxDist( bgr3u.at<Vec3b>( y, 1 ), bgr3u.at<Vec3b>( y, 0 ) ) * 2;
    Ix.at<int>( y, W - 1 ) = bgrMaxDist( bgr3u.at<Vec3b>( y, W - 1 ), bgr3u.at<Vec3b>( y, W - 2 ) ) * 2;
  }

  // Top/bottom most column Iy
  for ( int x = 0; x < W; x++ )
  {
    Iy.at<int>( 0, x ) = bgrMaxDist( bgr3u.at<Vec3b>( 1, x ), bgr3u.at<Vec3b>( 0, x ) ) * 2;
    Iy.at<int>( H - 1, x ) = bgrMaxDist( bgr3u.at<Vec3b>( H - 1, x ), bgr3u.at<Vec3b>( H - 2, x ) ) * 2;
  }

  // Find the gradient for inner regions
  for ( int y = 0; y < H; y++ )
  {
    const Vec3b *dataP = bgr3u.ptr<Vec3b>( y );
    for ( int x = 2; x < W; x++ )
      Ix.at<int>( y, x - 1 ) = bgrMaxDist( dataP[x - 2], dataP[x] );  //  bgr3u.at<Vec3b>(y, x+1), bgr3u.at<Vec3b>(y, x-1));
  }
  for ( int y = 1; y < H - 1; y++ )
  {
    const Vec3b *tP = bgr3u.ptr<Vec3b>( y - 1 );
    const Vec3b *bP = bgr3u.ptr<Vec3b>( y + 1 );
    for ( int x = 0; x < W; x++ )
      Iy.at<int>( y, x ) = bgrMaxDist( tP[x], bP[x] );
  }
  gradientXY( Ix, Iy, mag1u );
}

void ObjectnessBING::gradientGray( Mat &bgr3u, Mat &mag1u )
{
  Mat g1u;
  cvtColor( bgr3u, g1u, COLOR_BGR2GRAY );
  const int H = g1u.rows, W = g1u.cols;
  Mat Ix( H, W, CV_32S ), Iy( H, W, CV_32S );

  // Left/right most column Ix
  for ( int y = 0; y < H; y++ )
  {
    Ix.at<int>( y, 0 ) = abs( g1u.at<BYTE>( y, 1 ) - g1u.at<BYTE>( y, 0 ) ) * 2;
    Ix.at<int>( y, W - 1 ) = abs( g1u.at<BYTE>( y, W - 1 ) - g1u.at<BYTE>( y, W - 2 ) ) * 2;
  }

  // Top/bottom most column Iy
  for ( int x = 0; x < W; x++ )
  {
    Iy.at<int>( 0, x ) = abs( g1u.at<BYTE>( 1, x ) - g1u.at<BYTE>( 0, x ) ) * 2;
    Iy.at<int>( H - 1, x ) = abs( g1u.at<BYTE>( H - 1, x ) - g1u.at<BYTE>( H - 2, x ) ) * 2;
  }

  // Find the gradient for inner regions
  for ( int y = 0; y < H; y++ )
    for ( int x = 1; x < W - 1; x++ )
      Ix.at<int>( y, x ) = abs( g1u.at<BYTE>( y, x + 1 ) - g1u.at<BYTE>( y, x - 1 ) );
  for ( int y = 1; y < H - 1; y++ )
    for ( int x = 0; x < W; x++ )
      Iy.at<int>( y, x ) = abs( g1u.at<BYTE>( y + 1, x ) - g1u.at<BYTE>( y - 1, x ) );

  gradientXY( Ix, Iy, mag1u );
}

void ObjectnessBING::gradientHSV( Mat &bgr3u, Mat &mag1u )
{
  Mat hsv3u;
  cvtColor( bgr3u, hsv3u, COLOR_BGR2HSV );
  const int H = hsv3u.rows, W = hsv3u.cols;
  Mat Ix( H, W, CV_32S ), Iy( H, W, CV_32S );

  // Left/right most column Ix
  for ( int y = 0; y < H; y++ )
  {
    Ix.at<int>( y, 0 ) = vecDist3b( hsv3u.at<Vec3b>( y, 1 ), hsv3u.at<Vec3b>( y, 0 ) );
    Ix.at<int>( y, W - 1 ) = vecDist3b( hsv3u.at<Vec3b>( y, W - 1 ), hsv3u.at<Vec3b>( y, W - 2 ) );
  }

  // Top/bottom most column Iy
  for ( int x = 0; x < W; x++ )
  {
    Iy.at<int>( 0, x ) = vecDist3b( hsv3u.at<Vec3b>( 1, x ), hsv3u.at<Vec3b>( 0, x ) );
    Iy.at<int>( H - 1, x ) = vecDist3b( hsv3u.at<Vec3b>( H - 1, x ), hsv3u.at<Vec3b>( H - 2, x ) );
  }

  // Find the gradient for inner regions
  for ( int y = 0; y < H; y++ )
    for ( int x = 1; x < W - 1; x++ )
      Ix.at<int>( y, x ) = vecDist3b( hsv3u.at<Vec3b>( y, x + 1 ), hsv3u.at<Vec3b>( y, x - 1 ) ) / 2;
  for ( int y = 1; y < H - 1; y++ )
    for ( int x = 0; x < W; x++ )
      Iy.at<int>( y, x ) = vecDist3b( hsv3u.at<Vec3b>( y + 1, x ), hsv3u.at<Vec3b>( y - 1, x ) ) / 2;

  gradientXY( Ix, Iy, mag1u );
}

void ObjectnessBING::gradientXY( Mat &x1i, Mat &y1i, Mat &mag1u )
{
  const int H = x1i.rows, W = x1i.cols;
  mag1u.create( H, W, CV_8U );
  for ( int r = 0; r < H; r++ )
  {
    const int *x = x1i.ptr<int>( r ), *y = y1i.ptr<int>( r );
    BYTE* m = mag1u.ptr<BYTE>( r );
    for ( int c = 0; c < W; c++ )
      m[c] = (BYTE) min( x[c] + y[c], 255 );   //((int)sqrt(sqr(x[c]) + sqr(y[c])), 255);
  }
}

void ObjectnessBING::getObjBndBoxesForSingleImage( Mat img, ValStructVec<float, Vec4i> &finalBoxes, int numDetPerSize )
{
  ValStructVec<float, Vec4i> boxes;
  finalBoxes.reserve( 10000 );

  int scales[3] =
  { 1, 3, 5 };
  for ( int clr = MAXBGR; clr <= G; clr++ )
  {
    setColorSpace( clr );
    if (!loadTrainedModel())
      continue;

    CmTimer tm( "Predict" );
    tm.Start();

    getObjBndBoxes( img, boxes, numDetPerSize );
    finalBoxes.append( boxes, scales[clr] );

    tm.Stop();
    printf( "Average time for predicting an image (%s) is %gs\n", _clrName[_Clr], tm.TimeInSeconds() );
  }

  //Write on file the total number and the list of rectangles returned by objectess, one for each row.

  CmFile::MkDir( _bbResDir );
  CStr fName = _bbResDir + "bb";
  std::vector<Vec4i> sortedBB = finalBoxes.getSortedStructVal();
  std::ofstream ofs;
  ofs.open( ( fName + ".txt" ).c_str(), std::ofstream::out );
  std::stringstream dim;
  dim << sortedBB.size();
  ofs << dim.str() << "\n";
  for ( size_t k = 0; k < sortedBB.size(); k++ )
  {
    std::stringstream str;
    str << sortedBB[k][0] << " " << sortedBB[k][1] << " " << sortedBB[k][2] << " " << sortedBB[k][3] << "\n";
    ofs << str.str();
  }
  ofs.close();
}

struct MatchPathSeparator
{
  bool operator()( char ch ) const
  {
    return ch == '/';
  }
};

std::string inline basename( std::string const& pathname )
{
  return std::string( std::find_if( pathname.rbegin(), pathname.rend(), MatchPathSeparator() ).base(), pathname.end() );
}

std::string inline removeExtension( std::string const& filename )
{
  std::string::const_reverse_iterator pivot = std::find( filename.rbegin(), filename.rend(), '.' );
  return pivot == filename.rend() ? filename : std::string( filename.begin(), pivot.base() - 1 );
}

// Read matrix from binary file
bool ObjectnessBING::matRead( const std::string& filename, Mat& _M )
{
  String filenamePlusExt( filename.c_str() );
  filenamePlusExt += ".yml.gz";
  FileStorage fs2( filenamePlusExt, FileStorage::READ );
  if (! fs2.isOpened()) // wrong trainingPath
    return false;

  Mat M;
  fs2[String( removeExtension( basename( filename ) ).c_str() )] >> M;

  M.copyTo( _M );
  return true;
}
std::vector<float> ObjectnessBING::getobjectnessValues()
{
  return objectnessValues;
}

void ObjectnessBING::read()
{

}

void ObjectnessBING::write() const
{

}

bool ObjectnessBING::computeSaliencyImpl( InputArray image, OutputArray objectnessBoundingBox )
{
  ValStructVec<float, Vec4i> finalBoxes;
  getObjBndBoxesForSingleImage( image.getMat(), finalBoxes, 250 );

  // List of rectangles returned by objectess function in descending order.
  // At the top there are the rectangles with higher values, ie more
  // likely to have objects in them.
  std::vector<Vec4i> sortedBB = finalBoxes.getSortedStructVal();
  Mat( sortedBB ).copyTo( objectnessBoundingBox );

  // List of the rectangles' objectness value
  unsigned long int valIdxesSize = (unsigned long int) finalBoxes.getvalIdxes().size();
  objectnessValues.resize( valIdxesSize );
  for ( uint i = 0; i < valIdxesSize; i++ )
    objectnessValues[finalBoxes.getvalIdxes()[i].second] = finalBoxes.getvalIdxes()[i].first;

  return true;
}

template<typename VT, typename ST>
void ObjectnessBING::ValStructVec<VT, ST>::append( const ValStructVec<VT, ST> &newVals, int startV )
{
  int newValsSize = newVals.size();
  for ( int i = 0; i < newValsSize; i++ )
    pushBack( (float) ( ( i + 300 ) * startV ), newVals[i] );
}

template<typename VT, typename ST>
void ObjectnessBING::ValStructVec<VT, ST>::sort( bool descendOrder /* = true */)
{
  if( descendOrder )
    std::sort( valIdxes.begin(), valIdxes.end(), std::greater<std::pair<VT, int> >() );
  else
    std::sort( valIdxes.begin(), valIdxes.end(), std::less<std::pair<VT, int> >() );
}

template<typename VT, typename ST>
const std::vector<ST>& ObjectnessBING::ValStructVec<VT, ST>::getSortedStructVal()
{
  sortedStructVals.resize( sz );
  for ( int i = 0; i < sz; i++ )
    sortedStructVals[i] = structVals[valIdxes[i].second];
  return sortedStructVals;
}

template<typename VT, typename ST>
std::vector<std::pair<VT, int> > ObjectnessBING::ValStructVec<VT, ST>::getvalIdxes()
{
  return valIdxes;
}

template<typename VT, typename ST>
ObjectnessBING::ValStructVec<VT, ST>::ValStructVec()
{
  clear();
}

template<typename VT, typename ST>
int ObjectnessBING::ValStructVec<VT, ST>::size() const
{
  return sz;
}

template<typename VT, typename ST>
void ObjectnessBING::ValStructVec<VT, ST>::clear()
{
  sz = 0;
  structVals.clear();
  valIdxes.clear();
}

template<typename VT, typename ST>
void ObjectnessBING::ValStructVec<VT, ST>::reserve( int resSz )
{
  clear();
  structVals.reserve( resSz );
  valIdxes.reserve( resSz );
}

template<typename VT, typename ST>
void ObjectnessBING::ValStructVec<VT, ST>::pushBack( const VT& val, const ST& structVal )
{
  valIdxes.push_back( std::make_pair( val, sz ) );
  structVals.push_back( structVal );
  sz++;
}

template<typename VT, typename ST>
const VT& ObjectnessBING::ValStructVec<VT, ST>::operator ()( int i ) const
{
  return valIdxes[i].first;
}  // Should be called after sort

template<typename VT, typename ST>
const ST& ObjectnessBING::ValStructVec<VT, ST>::operator []( int i ) const
{
  return structVals[valIdxes[i].second];
}  // Should be called after sort

template<typename VT, typename ST>
VT& ObjectnessBING::ValStructVec<VT, ST>::operator ()( int i )
{
  return valIdxes[i].first;
}  // Should be called after sort

template<typename VT, typename ST>
ST& ObjectnessBING::ValStructVec<VT, ST>::operator []( int i )
{
  return structVals[valIdxes[i].second];
}

} /* namespace saliency */
}/* namespace cv */
