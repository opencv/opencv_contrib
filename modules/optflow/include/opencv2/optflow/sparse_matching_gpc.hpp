/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2016, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

/*
Implementation of the Global Patch Collider algorithm from the following paper:
http://research.microsoft.com/en-us/um/people/pkohli/papers/wfrik_cvpr2016.pdf

@InProceedings{Wang_2016_CVPR,
 author = {Wang, Shenlong and Ryan Fanello, Sean and Rhemann, Christoph and Izadi, Shahram and Kohli, Pushmeet},
 title = {The Global Patch Collider},
 booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 month = {June},
 year = {2016}
}
*/

#ifndef __OPENCV_OPTFLOW_SPARSE_MATCHING_GPC_HPP__
#define __OPENCV_OPTFLOW_SPARSE_MATCHING_GPC_HPP__

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace cv
{
namespace optflow
{

struct CV_EXPORTS_W GPCPatchDescriptor
{
  static const unsigned nFeatures = 18; // number of features in a patch descriptor
  Vec< double, nFeatures > feature;

  GPCPatchDescriptor( const Mat *imgCh, int i, int j );

  static void getAllDescriptorsForImage( const Mat *imgCh, std::vector< GPCPatchDescriptor > &descr, bool allowOpenCL = false );

  static void getCoordinatesFromIndex( size_t index, Size sz, int &x, int &y );
};

typedef std::pair< GPCPatchDescriptor, GPCPatchDescriptor > GPCPatchSample;
typedef std::vector< GPCPatchSample > GPCSamplesVector;

/** @brief Class encapsulating training samples.
 */
class CV_EXPORTS_W GPCTrainingSamples
{
private:
  GPCSamplesVector samples;

public:
  /** @brief This function can be used to extract samples from a pair of images and a ground truth flow.
   * Sizes of all the provided vectors must be equal.
   */
  static Ptr< GPCTrainingSamples > create( const std::vector< String > &imagesFrom, const std::vector< String > &imagesTo,
                                           const std::vector< String > &gt );

  size_t size() const { return samples.size(); }

  operator GPCSamplesVector() const { return samples; }

  operator GPCSamplesVector &() { return samples; }
};

/** @brief Class encapsulating training parameters.
 */
struct GPCTrainingParams
{
  unsigned maxTreeDepth;  // Maximum tree depth to stop partitioning.
  int minNumberOfSamples; // Minimum number of samples in the node to stop partitioning.
  bool printProgress;

  GPCTrainingParams( unsigned _maxTreeDepth = 24, int _minNumberOfSamples = 3, bool _printProgress = true )
      : maxTreeDepth( _maxTreeDepth ), minNumberOfSamples( _minNumberOfSamples ), printProgress( _printProgress )
  {
    CV_Assert( _maxTreeDepth > 0 );
    CV_Assert( _minNumberOfSamples >= 2 );
  }

  GPCTrainingParams( const GPCTrainingParams &params )
      : maxTreeDepth( params.maxTreeDepth ), minNumberOfSamples( params.minNumberOfSamples ), printProgress( params.printProgress ) {}
};

/** @brief Class encapsulating matching parameters.
 */
struct GPCMatchingParams
{
  bool useOpenCL;      // Whether to use OpenCL to speed up the matching.
  int hashTableFactor; // Hash table size multiplier. Change with care! Reducing this will lead to a less number of matches and less memory usage.

  GPCMatchingParams( bool _useOpenCL = false, int _hashTableFactor = 73 ) : useOpenCL( _useOpenCL ), hashTableFactor( _hashTableFactor )
  {
    CV_Assert( _hashTableFactor > 1 );
  }

  GPCMatchingParams( const GPCMatchingParams &params ) : useOpenCL( params.useOpenCL ), hashTableFactor( params.hashTableFactor ) {}
};

class CV_EXPORTS_W GPCTree : public Algorithm
{
public:
  struct Node
  {
    Vec< double, GPCPatchDescriptor::nFeatures > coef; // hyperplane coefficients
    double rhs;
    unsigned left;
    unsigned right;

    bool operator==( const Node &n ) const { return coef == n.coef && rhs == n.rhs && left == n.left && right == n.right; }
  };

private:
  typedef GPCSamplesVector::iterator SIter;

  std::vector< Node > nodes;

  bool trainNode( size_t nodeId, SIter begin, SIter end, unsigned depth, const GPCTrainingParams &params );

public:
  void train( GPCSamplesVector &samples, const GPCTrainingParams params = GPCTrainingParams() );

  void write( FileStorage &fs ) const;

  void read( const FileNode &fn );

  unsigned findLeafForPatch( const GPCPatchDescriptor &descr ) const;

  static Ptr< GPCTree > create() { return makePtr< GPCTree >(); }

  bool operator==( const GPCTree &t ) const { return nodes == t.nodes; }
};

template < int T > class CV_EXPORTS_W GPCForest : public Algorithm
{
private:
  struct Trail
  {
    unsigned leaf[T]; // Inside which leaf of the tree 0..T the patch fell?
    Point2i coord;    // Patch coordinates.

    uint64 getHash( uint64 mod ) const
    {
      uint64 hash = 0;
      for ( int i = 0; i < T; ++i )
        hash = ( hash * 67421 + leaf[i] ) % mod;
      return hash;
    }

    bool operator==( const Trail &trail ) const { return memcmp( leaf, trail.leaf, sizeof( leaf ) ) == 0; }
  };

  GPCTree tree[T];

public:
  /** @brief Train the forest using one sample set for every tree.
   * Please, consider using the next method instead of this one for better quality.
   */
  void train( GPCSamplesVector &samples, const GPCTrainingParams params = GPCTrainingParams() )
  {
    for ( int i = 0; i < T; ++i )
      tree[i].train( samples, params );
  }

  /** @brief Train the forest using individual samples for each tree.
   * It is generally better to use this instead of the first method.
   */
  void train( const std::vector< String > &imagesFrom, const std::vector< String > &imagesTo, const std::vector< String > &gt,
              const GPCTrainingParams params = GPCTrainingParams() )
  {
    for ( int i = 0; i < T; ++i )
    {
      Ptr< GPCTrainingSamples > samples = GPCTrainingSamples::create( imagesFrom, imagesTo, gt ); // Create training set for the tree
      tree[i].train( *samples, params );
    }
  }

  void write( FileStorage &fs ) const
  {
    fs << "ntrees" << T << "trees"
       << "[";
    for ( int i = 0; i < T; ++i )
    {
      fs << "{";
      tree[i].write( fs );
      fs << "}";
    }
    fs << "]";
  }

  void read( const FileNode &fn )
  {
    CV_Assert( T <= (int)fn["ntrees"] );
    FileNodeIterator it = fn["trees"].begin();
    for ( int i = 0; i < T; ++i, ++it )
      tree[i].read( *it );
  }

  /** @brief Find correspondences between two images.
   * @param[in] imgFrom First image in a sequence.
   * @param[in] imgTo Second image in a sequence.
   * @param[out] corr Output vector with pairs of corresponding points.
   * @param[in] params Additional matching parameters for fine-tuning.
   */
  void findCorrespondences( InputArray imgFrom, InputArray imgTo, std::vector< std::pair< Point2i, Point2i > > &corr,
                            const GPCMatchingParams params = GPCMatchingParams() ) const;

  static Ptr< GPCForest > create() { return makePtr< GPCForest >(); }
};

class CV_EXPORTS_W GPCDetails
{
public:
  static void dropOutliers( std::vector< std::pair< Point2i, Point2i > > &corr );
};

template < int T >
void GPCForest< T >::findCorrespondences( InputArray imgFrom, InputArray imgTo, std::vector< std::pair< Point2i, Point2i > > &corr,
                                          const GPCMatchingParams params ) const
{
  CV_Assert( imgFrom.channels() == 3 );
  CV_Assert( imgTo.channels() == 3 );

  Mat from, to;
  imgFrom.getMat().convertTo( from, CV_32FC3 );
  imgTo.getMat().convertTo( to, CV_32FC3 );
  cvtColor( from, from, COLOR_BGR2YCrCb );
  cvtColor( to, to, COLOR_BGR2YCrCb );

  Mat fromCh[3], toCh[3];
  split( from, fromCh );
  split( to, toCh );

  std::vector< GPCPatchDescriptor > descr;
  GPCPatchDescriptor::getAllDescriptorsForImage( fromCh, descr, params.useOpenCL );
  std::vector< std::vector< Trail > > hashTable1( from.size().area() * params.hashTableFactor ),
    hashTable2( from.size().area() * params.hashTableFactor );

  for ( size_t i = 0; i < descr.size(); ++i )
  {
    Trail trail;
    for ( int t = 0; t < T; ++t )
      trail.leaf[t] = tree[t].findLeafForPatch( descr[i] );
    GPCPatchDescriptor::getCoordinatesFromIndex( i, from.size(), trail.coord.x, trail.coord.y );
    hashTable1[trail.getHash( hashTable1.size() )].push_back( trail );
  }

  descr.clear();
  GPCPatchDescriptor::getAllDescriptorsForImage( toCh, descr, params.useOpenCL );

  for ( size_t i = 0; i < descr.size(); ++i )
  {
    Trail trail;
    for ( int t = 0; t < T; ++t )
      trail.leaf[t] = tree[t].findLeafForPatch( descr[i] );
    GPCPatchDescriptor::getCoordinatesFromIndex( i, to.size(), trail.coord.x, trail.coord.y );
    hashTable2[trail.getHash( hashTable2.size() )].push_back( trail );
  }

  for ( size_t i = 0; i < hashTable1.size(); ++i )
    if ( hashTable1[i].size() == 1 && hashTable2[i].size() == 1 && hashTable1[i][0] == hashTable2[i][0] )
      corr.push_back( std::make_pair( hashTable1[i][0].coord, hashTable2[i][0].coord ) );

  GPCDetails::dropOutliers( corr );
}
} // namespace optflow

CV_EXPORTS void write( FileStorage &fs, const String &name, const optflow::GPCTree::Node &node );

CV_EXPORTS void read( const FileNode &fn, optflow::GPCTree::Node &node, optflow::GPCTree::Node );
} // namespace cv

#endif
