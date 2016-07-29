/*********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2015
 * Balint Cristian <cristian dot balint at gmail dot com>
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

#ifndef __OPENCV_HDF5_HPP__
#define __OPENCV_HDF5_HPP__

#include <vector>


namespace cv
{
namespace hdf
{
using namespace std;

//! @addtogroup hdf5
//! @{


/** @brief Hierarchical Data Format version 5 interface.

Notice that module is compiled only when hdf5 is correctly installed.

 */
class CV_EXPORTS_W HDF5
{
public:

    CV_WRAP enum
    {
      H5_UNLIMITED = -1, H5_NONE = -1, H5_GETDIMS = 100, H5_GETMAXDIMS = 101, H5_GETCHUNKDIMS = 102,
    };

    virtual ~HDF5() {}

    /** @brief Close and release hdf5 object.
     */
    CV_WRAP virtual void close( ) = 0;

    /** @brief Create a group.
    @param grlabel specify the hdf5 group label.

    Create a hdf5 group.

    @note Groups are useful for better organise multiple datasets. It is possible to create subgroups within any group.
    Existence of a particular group can be checked using hlexists(). In case of subgroups label would be e.g: 'Group1/SubGroup1'
    where SubGroup1 is within the root group Group1.

    - In this example Group1 will have one subgrup labeled SubGroup1:
    @code{.cpp}
      // open / autocreate hdf5 file
      cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( "mytest.h5" );
      // create Group1 if does not exists
      if ( ! h5io->hlexists( "Group1" ) )
        h5io->grcreate( "Group1" );
      else
        printf("Group1 already created, skipping\n" );
      // create SubGroup1 if does not exists
      if ( ! h5io->hlexists( "Group1/SubGroup1" ) )
        h5io->grcreate( "Group1/SubGroup1" );
      else
        printf("SubGroup1 already created, skipping\n" );
      // release
      h5io->close();
    @endcode

    @note When a dataset is created with dscreate() or kpcreate() it can be created right within a group by specifying
    full path within the label, in our example would be: 'Group1/SubGroup1/MyDataSet'. It is not thread safe.
     */
    CV_WRAP virtual void grcreate( String grlabel ) = 0;

    /** @brief Check if label exists or not.
    @param label specify the hdf5 dataset label.

    Returns **true** if dataset exists, and **false** if does not.

    @note Checks if dataset, group or other object type (hdf5 link) exists under the label name. It is thread safe.
     */
    CV_WRAP virtual bool hlexists( String label ) const = 0;

    /* @overload */
    CV_WRAP virtual void dscreate( const int rows, const int cols, const int type,
                 String dslabel ) const = 0;
    /* @overload */
    CV_WRAP virtual void dscreate( const int rows, const int cols, const int type,
                 String dslabel, const int compresslevel ) const = 0;
    /* @overload */
    CV_WRAP virtual void dscreate( const int rows, const int cols, const int type,
                 String dslabel, const int compresslevel, const vector<int>& dims_chunks ) const = 0;
    /** @brief Create and allocate storage for two dimensional single or multi channel dataset.
    @param rows declare amount of rows
    @param cols declare amount of cols
    @param type type to be used
    @param dslabel specify the hdf5 dataset label, any existing dataset with the same label will be overwritten.
    @param compresslevel specify the compression level 0-9 to be used, H5_NONE is default and means no compression.
    @param dims_chunks each array member specify chunking sizes to be used for block i/o,
           by default NULL means none at all.

    @note If the dataset already exists an exception will be thrown.

    - Existence of the dataset can be checked using hlexists(), see in this example:
    @code{.cpp}
      // open / autocreate hdf5 file
      cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( "mytest.h5" );
      // create space for 100x50 CV_64FC2 matrix
      if ( ! h5io->hlexists( "hilbert" ) )
        h5io->dscreate( 100, 50, CV_64FC2, "hilbert" );
      else
        printf("DS already created, skipping\n" );
      // release
      h5io->close();
    @endcode

    @note Activating compression requires internal chunking. Chunking can significantly improve access
    speed booth at read or write time especially for windowed access logic that shifts offset inside dataset.
    If no custom chunking is specified default one will be invoked by the size of **whole** dataset
    as single big chunk of data.

    - See example of level 9 compression using internal default chunking:
    @code{.cpp}
      // open / autocreate hdf5 file
      cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( "mytest.h5" );
      // create level 9 compressed space for CV_64FC2 matrix
      if ( ! h5io->hlexists( "hilbert", 9 ) )
        h5io->dscreate( 100, 50, CV_64FC2, "hilbert", 9 );
      else
        printf("DS already created, skipping\n" );
      // release
      h5io->close();
    @endcode

    @note A value of H5_UNLIMITED for **rows** or **cols** or booth means **unlimited** data on the specified dimension,
    thus is possible to expand anytime such dataset on row, col or booth directions. Presence of H5_UNLIMITED on any
    dimension **require** to define custom chunking. No default chunking will be defined in unlimited scenario since
    default size on that dimension will be zero, and will grow once dataset is written. Writing into dataset that have
    H5_UNLIMITED on some of its dimension requires dsinsert() that allow growth on unlimited dimension instead of dswrite()
    that allows to write only in predefined data space.

    - Example below shows no compression but unlimited dimension on cols using 100x100 internal chunking:
    @code{.cpp}
      // open / autocreate hdf5 file
      cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( "mytest.h5" );
      // create level 9 compressed space for CV_64FC2 matrix
      int chunks[2] = { 100, 100 };
      h5io->dscreate( 100, cv::hdf::HDF5::H5_UNLIMITED, CV_64FC2, "hilbert", cv::hdf::HDF5::H5_NONE, chunks );
      // release
      h5io->close();
    @endcode

    @note It is **not** thread safe, it must be called only once at dataset creation otherwise exception will occur.
    Multiple datasets inside single hdf5 file is allowed.
     */
    CV_WRAP virtual void dscreate( const int rows, const int cols, const int type,
                 String dslabel, const int compresslevel, const int* dims_chunks ) const = 0;

    /* @overload */
    CV_WRAP virtual void dscreate( const int n_dims, const int* sizes, const int type,
                 String dslabel ) const = 0;
    /* @overload */
    CV_WRAP virtual void dscreate( const int n_dims, const int* sizes, const int type,
                 String dslabel, const int compresslevel ) const = 0;
    /* @overload */
    CV_WRAP virtual void dscreate( const vector<int>& sizes, const int type,
                 String dslabel, const int compresslevel = HDF5::H5_NONE,
                 const vector<int>& dims_chunks = vector<int>() ) const = 0;
    /** @brief Create and allocate storage for n-dimensional dataset, single or mutichannel type.
    @param n_dims declare number of dimensions
    @param sizes array containing sizes for each dimensions
    @param type type to be used
    @param dslabel specify the hdf5 dataset label, any existing dataset with the same label will be overwritten.
    @param compresslevel specify the compression level 0-9 to be used, H5_NONE is default and means no compression.
    @param dims_chunks each array member specify chunking sizes to be used for block i/o,
           by default NULL means none at all.
    @note If the dataset already exists an exception will be thrown. Existence of the dataset can be checked
    using hlexists().

    - See example below that creates a 6 dimensional storage space:
    @code{.cpp}
      // open / autocreate hdf5 file
      cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( "mytest.h5" );
      // create space for 6 dimensional CV_64FC2 matrix
      if ( ! h5io->hlexists( "nddata" ) )
        int n_dims = 5;
        int dsdims[n_dims] = { 100, 100, 20, 10, 5, 5 };
        h5io->dscreate( n_dims, sizes, CV_64FC2, "nddata" );
      else
        printf("DS already created, skipping\n" );
      // release
      h5io->close();
    @endcode

    @note Activating compression requires internal chunking. Chunking can significantly improve access
    speed booth at read or write time especially for windowed access logic that shifts offset inside dataset.
    If no custom chunking is specified default one will be invoked by the size of **whole** dataset
    as single big chunk of data.

    - See example of level 0 compression (shallow) using chunking against first
    dimension, thus storage will consists by 100 chunks of data:
    @code{.cpp}
      // open / autocreate hdf5 file
      cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( "mytest.h5" );
      // create space for 6 dimensional CV_64FC2 matrix
      if ( ! h5io->hlexists( "nddata" ) )
        int n_dims = 5;
        int dsdims[n_dims] = { 100, 100, 20, 10, 5, 5 };
        int chunks[n_dims] = {   1, 100, 20, 10, 5, 5 };
        h5io->dscreate( n_dims, dsdims, CV_64FC2, "nddata", 0, chunks );
      else
        printf("DS already created, skipping\n" );
      // release
      h5io->close();
    @endcode

    @note A value of H5_UNLIMITED inside the **sizes** array means **unlimited** data on that dimension, thus is
    possible to expand anytime such dataset on those unlimited directions. Presence of H5_UNLIMITED on any dimension
    **require** to define custom chunking. No default chunking will be defined in unlimited scenario since default size
    on that dimension will be zero, and will grow once dataset is written. Writing into dataset that have H5_UNLIMITED on
    some of its dimension requires dsinsert() instead of dswrite() that allow growth on unlimited dimension instead of
    dswrite() that allows to write only in predefined data space.

    - Example below shows a 3 dimensional dataset using no compression with all unlimited sizes and one unit chunking:
    @code{.cpp}
      // open / autocreate hdf5 file
      cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( "mytest.h5" );
      int n_dims = 3;
      int chunks[n_dims] = { 1, 1, 1 };
      int dsdims[n_dims] = { cv::hdf::HDF5::H5_UNLIMITED, cv::hdf::HDF5::H5_UNLIMITED, cv::hdf::HDF5::H5_UNLIMITED };
      h5io->dscreate( n_dims, dsdims, CV_64FC2, "nddata", cv::hdf::HDF5::H5_NONE, chunks );
      // release
      h5io->close();
    @endcode
     */
    CV_WRAP virtual void dscreate( const int n_dims, const int* sizes, const int type,
                 String dslabel, const int compresslevel, const int* dims_chunks ) const = 0;

    /** @brief Fetch dataset sizes
    @param dslabel specify the hdf5 dataset label to be measured.
    @param dims_flag will fetch dataset dimensions on H5_GETDIMS, and dataset maximum dimensions on H5_GETMAXDIMS.

    Returns vector object containing sizes of dataset on each dimensions.

    @note Resulting vector size will match the amount of dataset dimensions. By default H5_GETDIMS will return
    actual dataset dimensions. Using H5_GETMAXDIM flag will get maximum allowed dimension which normally match
    actual dataset dimension but can hold H5_UNLIMITED value if dataset was prepared in **unlimited** mode on
    some of its dimension. It can be useful to check existing dataset dimensions before overwrite it as whole or subset.
    Trying to write with oversized source data into dataset target will thrown exception. The H5_GETCHUNKDIMS will
    return the dimension of chunk if dataset was created with chunking options otherwise returned vector size
    will be zero.
     */
    CV_WRAP virtual vector<int> dsgetsize( String dslabel, int dims_flag = HDF5::H5_GETDIMS ) const = 0;

    /** @brief Fetch dataset type
    @param dslabel specify the hdf5 dataset label to be checked.

    Returns the stored matrix type. This is an identifier compatible with the CvMat type system,
    like e.g. CV_16SC5 (16-bit signed 5-channel array), and so on.

    @note Result can be parsed with CV_MAT_CN() to obtain amount of channels and CV_MAT_DEPTH() to obtain native cvdata type.
    It is thread safe.
     */
    CV_WRAP virtual int dsgettype( String dslabel ) const = 0;

    /* @overload */
    CV_WRAP virtual void dswrite( InputArray Array, String dslabel ) const = 0;
    /* @overload */
    CV_WRAP virtual void dswrite( InputArray Array, String dslabel,
                 const int* dims_offset ) const = 0;
    /* @overload */
    CV_WRAP virtual void dswrite( InputArray Array, String dslabel,
                 const vector<int>& dims_offset,
                 const vector<int>& dims_counts = vector<int>() ) const = 0;
    /** @brief Write or overwrite a Mat object into specified dataset of hdf5 file.
    @param Array specify Mat data array to be written.
    @param dslabel specify the target hdf5 dataset label.
    @param dims_offset each array member specify the offset location
           over dataset's each dimensions from where InputArray will be (over)written into dataset.
    @param dims_counts each array member specify the amount of data over dataset's
           each dimensions from InputArray that will be written into dataset.

    Writes Mat object into targeted dataset.

    @note If dataset is not created and does not exist it will be created **automatically**. Only Mat is supported and
    it must to be **continuous**. It is thread safe but it is recommended that writes to happen over separate non overlapping
    regions. Multiple datasets can be written inside single hdf5 file.

    - Example below writes a 100x100 CV_64FC2 matrix into a dataset. No dataset precreation required. If routine
    is called multiple times dataset will be just overwritten:
    @code{.cpp}
      // dual channel hilbert matrix
      cv::Mat H(100, 100, CV_64FC2);
      for(int i = 0; i < H.rows; i++)
        for(int j = 0; j < H.cols; j++)
        {
            H.at<cv::Vec2d>(i,j)[0] =  1./(i+j+1);
            H.at<cv::Vec2d>(i,j)[1] = -1./(i+j+1);
            count++;
        }
      // open / autocreate hdf5 file
      cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( "mytest.h5" );
      // write / overwrite dataset
      h5io->dswrite( H, "hilbert" );
      // release
      h5io->close();
    @endcode

    - Example below writes a smaller 50x100 matrix into 100x100 compressed space optimised by two 50x100 chunks.
    Matrix is written twice into first half (0->50) and second half (50->100) of data space using offset.
    @code{.cpp}
      // dual channel hilbert matrix
      cv::Mat H(50, 100, CV_64FC2);
      for(int i = 0; i < H.rows; i++)
        for(int j = 0; j < H.cols; j++)
        {
            H.at<cv::Vec2d>(i,j)[0] =  1./(i+j+1);
            H.at<cv::Vec2d>(i,j)[1] = -1./(i+j+1);
            count++;
        }
      // open / autocreate hdf5 file
      cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( "mytest.h5" );
      // optimise dataset by two chunks
      int chunks[2] = { 50, 100 };
      // create 100x100 CV_64FC2 compressed space
      h5io->dscreate( 100, 100, CV_64FC2, "hilbert", 9, chunks );
      // write into first half
      int offset1[2] = { 0, 0 };
      h5io->dswrite( H, "hilbert", offset1 );
      // write into second half
      int offset2[2] = { 50, 0 };
      h5io->dswrite( H, "hilbert", offset2 );
      // release
      h5io->close();
    @endcode
     */
    CV_WRAP virtual void dswrite( InputArray Array, String dslabel,
                 const int* dims_offset, const int* dims_counts ) const = 0;

    /* @overload */
    CV_WRAP virtual void dsinsert( InputArray Array, String dslabel ) const = 0;
    /* @overload */
    CV_WRAP virtual void dsinsert( InputArray Array,
                 String dslabel, const int* dims_offset ) const = 0;
    /* @overload */
    CV_WRAP virtual void dsinsert( InputArray Array,
                 String dslabel, const vector<int>& dims_offset,
                 const vector<int>& dims_counts = vector<int>() ) const = 0;
    /** @brief Insert or overwrite a Mat object into specified dataset and autoexpand dataset size if **unlimited** property allows.
    @param Array specify Mat data array to be written.
    @param dslabel specify the target hdf5 dataset label.
    @param dims_offset each array member specify the offset location
           over dataset's each dimensions from where InputArray will be (over)written into dataset.
    @param dims_counts each array member specify the amount of data over dataset's
           each dimensions from InputArray that will be written into dataset.

    Writes Mat object into targeted dataset and **autoexpand** dataset dimension if allowed.

    @note Unlike dswrite(), datasets are **not** created **automatically**. Only Mat is supported and it must to be **continuous**.
    If dsinsert() happen over outer regions of dataset dimensions and on that dimension of dataset is in **unlimited** mode then
    dataset is expanded, otherwise exception is thrown. To create datasets with **unlimited** property on specific or more
    dimensions see dscreate() and the optional H5_UNLIMITED flag at creation time. It is not thread safe over same dataset
    but multiple datasets can be merged inside single hdf5 file.

    - Example below creates **unlimited** rows x 100 cols and expand rows 5 times with dsinsert() using single 100x100 CV_64FC2
    over the dataset. Final size will have 5x100 rows and 100 cols, reflecting H matrix five times over row's span. Chunks size is
    100x100 just optimized against the H matrix size having compression disabled. If routine is called multiple times dataset will be
    just overwritten:
    @code{.cpp}
      // dual channel hilbert matrix
      cv::Mat H(50, 100, CV_64FC2);
      for(int i = 0; i < H.rows; i++)
        for(int j = 0; j < H.cols; j++)
        {
            H.at<cv::Vec2d>(i,j)[0] =  1./(i+j+1);
            H.at<cv::Vec2d>(i,j)[1] = -1./(i+j+1);
            count++;
        }
      // open / autocreate hdf5 file
      cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( "mytest.h5" );
      // optimise dataset by chunks
      int chunks[2] = { 100, 100 };
      // create Unlimited x 100 CV_64FC2 space
      h5io->dscreate( cv::hdf::HDF5::H5_UNLIMITED, 100, CV_64FC2, "hilbert", cv::hdf::HDF5::H5_NONE, chunks );
      // write into first half
      int offset[2] = { 0, 0 };
      for ( int t = 0; t < 5; t++ )
      {
        offset[0] += 100 * t;
        h5io->dsinsert( H, "hilbert", offset );
      }
      // release
      h5io->close();
    @endcode
     */
    CV_WRAP virtual void dsinsert( InputArray Array, String dslabel,
                 const int* dims_offset, const int* dims_counts ) const = 0;


    /* @overload */
    CV_WRAP virtual void dsread( OutputArray Array, String dslabel ) const = 0;
    /* @overload */
    CV_WRAP virtual void dsread( OutputArray Array,
                 String dslabel, const int* dims_offset ) const = 0;
    /* @overload */
    CV_WRAP virtual void dsread( OutputArray Array, String dslabel,
                 const vector<int>& dims_offset,
                 const vector<int>& dims_counts = vector<int>() ) const = 0;
    /** @brief Read specific dataset from hdf5 file into Mat object.
    @param Array Mat container where data reads will be returned.
    @param dslabel specify the source hdf5 dataset label.
    @param dims_offset each array member specify the offset location over
           each dimensions from where dataset starts to read into OutputArray.
    @param dims_counts each array member specify the amount over dataset's each
           dimensions of dataset to read into OutputArray.

    Reads out Mat object reflecting the stored dataset.

    @note If hdf5 file does not exist an exception will be thrown. Use hlexists() to check dataset presence.
    It is thread safe.

    - Example below reads a dataset:
    @code{.cpp}
      // open hdf5 file
      cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( "mytest.h5" );
      // blank Mat container
      cv::Mat H;
      // read hibert dataset
      h5io->read( H, "hilbert" );
      // release
      h5io->close();
    @endcode

    - Example below perform read of 3x5 submatrix from second row and third element.
    @code{.cpp}
      // open hdf5 file
      cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( "mytest.h5" );
      // blank Mat container
      cv::Mat H;
      int offset[2] = { 1, 2 };
      int counts[2] = { 3, 5 };
      // read hibert dataset
      h5io->read( H, "hilbert", offset, counts );
      // release
      h5io->close();
    @endcode
     */
    CV_WRAP virtual void dsread( OutputArray Array, String dslabel,
                 const int* dims_offset, const int* dims_counts ) const = 0;

    /** @brief Fetch keypoint dataset size
    @param kplabel specify the hdf5 dataset label to be measured.
    @param dims_flag will fetch dataset dimensions on H5_GETDIMS, and dataset maximum dimensions on H5_GETMAXDIMS.

    Returns size of keypoints dataset.

    @note Resulting size will match the amount of keypoints. By default H5_GETDIMS will return actual dataset dimension.
    Using H5_GETMAXDIM flag will get maximum allowed dimension which normally match actual dataset dimension but can hold
    H5_UNLIMITED value if dataset was prepared in **unlimited** mode. It can be useful to check existing dataset dimension
    before overwrite it as whole or subset. Trying to write with oversized source data into dataset target will thrown
    exception. The H5_GETCHUNKDIMS will return the dimension of chunk if dataset was created with chunking options otherwise
    returned vector size will be zero.
     */
    CV_WRAP virtual int kpgetsize( String kplabel, int dims_flag = HDF5::H5_GETDIMS ) const = 0;

    /** @brief Create and allocate special storage for cv::KeyPoint dataset.
    @param size declare fixed number of KeyPoints
    @param kplabel specify the hdf5 dataset label, any existing dataset with the same label will be overwritten.
    @param compresslevel specify the compression level 0-9 to be used, H5_NONE is default and means no compression.
    @param chunks each array member specify chunking sizes to be used for block i/o,
           H5_NONE is default and means no compression.
    @note If the dataset already exists an exception will be thrown. Existence of the dataset can be checked
    using hlexists().

    - See example below that creates space for 100 keypoints in the dataset:
    @code{.cpp}
      // open hdf5 file
      cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( "mytest.h5" );
      if ( ! h5io->hlexists( "keypoints" ) )
        h5io->kpcreate( 100, "keypoints" );
      else
        printf("DS already created, skipping\n" );
    @endcode

    @note A value of H5_UNLIMITED for **size** means **unlimited** keypoints, thus is possible to expand anytime such
    dataset by adding or inserting. Presence of H5_UNLIMITED **require** to define custom chunking. No default chunking
    will be defined in unlimited scenario since default size on that dimension will be zero, and will grow once dataset
    is written. Writing into dataset that have H5_UNLIMITED on some of its dimension requires kpinsert() that allow
    growth on unlimited dimension instead of kpwrite() that allows to write only in predefined data space.

    - See example below that creates unlimited space for keypoints chunking size of 100 but no compression:
    @code{.cpp}
      // open hdf5 file
      cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( "mytest.h5" );
      if ( ! h5io->hlexists( "keypoints" ) )
        h5io->kpcreate( cv::hdf::HDF5::H5_UNLIMITED, "keypoints", cv::hdf::HDF5::H5_NONE, 100 );
      else
        printf("DS already created, skipping\n" );
    @endcode
     */
    virtual void kpcreate( const int size, String kplabel,
             const int compresslevel = H5_NONE, const int chunks = H5_NONE ) const = 0;

    /** @brief Write or overwrite list of KeyPoint into specified dataset of hdf5 file.
    @param keypoints specify keypoints data list to be written.
    @param kplabel specify the target hdf5 dataset label.
    @param offset specify the offset location on dataset from where keypoints will be (over)written into dataset.
    @param counts specify the amount of keypoints that will be written into dataset.

    Writes vector<KeyPoint> object into targeted dataset.

    @note If dataset is not created and does not exist it will be created **automatically**. It is thread safe but
    it is recommended that writes to happen over separate non overlapping regions. Multiple datasets can be written
    inside single hdf5 file.

    - Example below writes a 100 keypoints into a dataset. No dataset precreation required. If routine is called multiple
    times dataset will be just overwritten:
    @code{.cpp}
      // generate 100 dummy keypoints
      std::vector<cv::KeyPoint> keypoints;
      for(int i = 0; i < 100; i++)
        keypoints.push_back( cv::KeyPoint(i, -i, 1, -1, 0, 0, -1) );
      // open / autocreate hdf5 file
      cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( "mytest.h5" );
      // write / overwrite dataset
      h5io->kpwrite( keypoints, "keypoints" );
      // release
      h5io->close();
    @endcode

    - Example below uses smaller set of 50 keypoints and writes into compressed space of 100 keypoints optimised by 10 chunks.
    Same keypoint set is written three times, first into first half (0->50) and at second half (50->75) then into remaining slots
    (75->99) of data space using offset and count parameters to settle the window for write access.If routine is called multiple times
    dataset will be just overwritten:
    @code{.cpp}
      // generate 50 dummy keypoints
      std::vector<cv::KeyPoint> keypoints;
      for(int i = 0; i < 50; i++)
        keypoints.push_back( cv::KeyPoint(i, -i, 1, -1, 0, 0, -1) );
      // open / autocreate hdf5 file
      cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( "mytest.h5" );
      // create maximum compressed space of size 100 with chunk size 10
      h5io->kpcreate( 100, "keypoints", 9, 10 );
      // write into first half
      h5io->kpwrite( keypoints, "keypoints", 0 );
      // write first 25 keypoints into second half
      h5io->kpwrite( keypoints, "keypoints", 50, 25 );
      // write first 25 keypoints into remained space of second half
      h5io->kpwrite( keypoints, "keypoints", 75, 25 );
      // release
      h5io->close();
    @endcode
     */
    virtual void kpwrite( const vector<KeyPoint> keypoints, String kplabel,
             const int offset = H5_NONE, const int counts = H5_NONE ) const = 0;

    /** @brief Insert or overwrite list of KeyPoint into specified dataset and autoexpand dataset size if **unlimited** property allows.
    @param keypoints specify keypoints data list to be written.
    @param kplabel specify the target hdf5 dataset label.
    @param offset specify the offset location on dataset from where keypoints will be (over)written into dataset.
    @param counts specify the amount of keypoints that will be written into dataset.

    Writes vector<KeyPoint> object into targeted dataset and **autoexpand** dataset dimension if allowed.

    @note Unlike kpwrite(), datasets are **not** created **automatically**. If dsinsert() happen over outer region of dataset
    and dataset has been created in **unlimited** mode then dataset is expanded, otherwise exception is thrown. To create datasets
    with **unlimited** property see kpcreate() and the optional H5_UNLIMITED flag at creation time. It is not thread safe over same
    dataset but multiple datasets can be merged inside single hdf5 file.

    - Example below creates **unlimited** space for keypoints storage, and inserts a list of 10 keypoints ten times into that space.
    Final dataset will have 100 keypoints. Chunks size is 10 just optimized against list of keypoints. If routine is called multiple
    times dataset will be just overwritten:
    @code{.cpp}
      // generate 10 dummy keypoints
      std::vector<cv::KeyPoint> keypoints;
      for(int i = 0; i < 10; i++)
        keypoints.push_back( cv::KeyPoint(i, -i, 1, -1, 0, 0, -1) );
      // open / autocreate hdf5 file
      cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( "mytest.h5" );
      // create unlimited size space with chunk size of 10
      h5io->kpcreate( cv::hdf::HDF5::H5_UNLIMITED, "keypoints", -1, 10 );
      // insert 10 times same 10 keypoints
      for(int i = 0; i < 10; i++)
        h5io->kpinsert( keypoints, "keypoints", i * 10 );
      // release
      h5io->close();
    @endcode
     */
    virtual void kpinsert( const vector<KeyPoint> keypoints, String kplabel,
             const int offset = H5_NONE, const int counts = H5_NONE ) const = 0;

    /** @brief Read specific keypoint dataset from hdf5 file into vector<KeyPoint> object.
    @param keypoints vector<KeyPoint> container where data reads will be returned.
    @param kplabel specify the source hdf5 dataset label.
    @param offset specify the offset location over dataset from where read starts.
    @param counts specify the amount of keypoints from dataset to read.

    Reads out vector<KeyPoint> object reflecting the stored dataset.

    @note If hdf5 file does not exist an exception will be thrown. Use hlexists() to check dataset presence.
    It is thread safe.

    - Example below reads a dataset containing keypoints starting with second entry:
    @code{.cpp}
      // open hdf5 file
      cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( "mytest.h5" );
      // blank KeyPoint container
      std::vector<cv::KeyPoint> keypoints;
      // read keypoints starting second one
      h5io->kpread( keypoints, "keypoints", 1 );
      // release
      h5io->close();
    @endcode

    - Example below perform read of 3 keypoints from second entry.
    @code{.cpp}
      // open hdf5 file
      cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( "mytest.h5" );
      // blank KeyPoint container
      std::vector<cv::KeyPoint> keypoints;
      // read three keypoints starting second one
      h5io->kpread( keypoints, "keypoints", 1, 3 );
      // release
      h5io->close();
    @endcode
     */
    virtual void kpread( vector<KeyPoint>& keypoints, String kplabel,
             const int offset = H5_NONE, const int counts = H5_NONE ) const = 0;

};

  /** @brief Open or create hdf5 file
  @param HDF5Filename specify the HDF5 filename.

  Returns pointer to the hdf5 object class

  @note If hdf5 file does not exist it will be created. Any operations except dscreate() functions on object
  will be thread safe. Multiple datasets can be created inside single hdf5 file, and can be accessed
  from same hdf5 object from multiple instances as long read or write operations are done over
  non-overlapping regions of dataset. Single hdf5 file also can be opened by multiple instances,
  reads and writes can be instantiated at the same time as long non-overlapping regions are involved. Object
  is released using close().

  - Example below open and then release the file.
  @code{.cpp}
    // open / autocreate hdf5 file
    cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( "mytest.h5" );
    // ...
    // release
    h5io->close();
  @endcode

  ![Visualization of 10x10 CV_64FC2 (Hilbert matrix) using HDFView tool](pics/hdfview_demo.gif)

  - Text dump (3x3 Hilbert matrix) of hdf5 dataset using **h5dump** tool:
  @code{.txt}
  $ h5dump test.h5
  HDF5 "test.h5" {
  GROUP "/" {
     DATASET "hilbert" {
        DATATYPE  H5T_ARRAY { [2] H5T_IEEE_F64LE }
        DATASPACE  SIMPLE { ( 3, 3 ) / ( 3, 3 ) }
        DATA {
        (0,0): [ 1, -1 ], [ 0.5, -0.5 ], [ 0.333333, -0.333333 ],
        (1,0): [ 0.5, -0.5 ], [ 0.333333, -0.333333 ], [ 0.25, -0.25 ],
        (2,0): [ 0.333333, -0.333333 ], [ 0.25, -0.25 ], [ 0.2, -0.2 ]
        }
     }
  }
  }
  @endcode
   */
  CV_EXPORTS_W Ptr<HDF5> open( String HDF5Filename );

//! @}

} // end namespace hdf
} // end namespace cv
#endif // _OPENCV_HDF5_HPP_
