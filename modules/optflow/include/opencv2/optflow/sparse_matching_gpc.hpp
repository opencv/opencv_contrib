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

namespace cv
{
namespace optflow
{

struct CV_EXPORTS_W GPCPatchDescriptor
{
  static const unsigned nFeatures = 18; // number of features in a patch descriptor
  Vec< double, nFeatures > feature;

  GPCPatchDescriptor( const Mat *imgCh, int i, int j );
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

  bool trainNode( size_t nodeId, SIter begin, SIter end, unsigned depth );

public:
  void train( GPCSamplesVector &samples );

  void write( FileStorage &fs ) const;

  void read( const FileNode &fn );

  static Ptr< GPCTree > create() { return makePtr< GPCTree >(); }

  bool operator==( const GPCTree &t ) const { return nodes == t.nodes; }
};

template < int T > class CV_EXPORTS_W GPCForest : public Algorithm
{
private:
  GPCTree tree[T];

public:
  /** @brief Train the forest using one sample set for every tree.
   * Please, consider using the next method instead of this one for better quality.
   */
  void train( GPCSamplesVector &samples )
  {
    for ( int i = 0; i < T; ++i )
      tree[i].train( samples );
  }

  /** @brief Train the forest using individual samples for each tree.
   * It is generally better to use this instead of the first method.
   */
  void train( const std::vector< String > &imagesFrom, const std::vector< String > &imagesTo, const std::vector< String > &gt )
  {
    for ( int i = 0; i < T; ++i )
    {
      Ptr< GPCTrainingSamples > samples = GPCTrainingSamples::create( imagesFrom, imagesTo, gt ); // Create training set for the tree
      tree[i].train( *samples );
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
    CV_Assert( T == (int)fn["ntrees"] );
    FileNodeIterator it = fn["trees"].begin();
    for ( int i = 0; i < T; ++i, ++it )
      tree[i].read( *it );
  }

  static Ptr< GPCForest > create() { return makePtr< GPCForest >(); }
};
}

CV_EXPORTS void write( FileStorage &fs, const String &name, const optflow::GPCTree::Node &node );

CV_EXPORTS void read( const FileNode &fn, optflow::GPCTree::Node &node, optflow::GPCTree::Node );
}

#endif
