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

#include "precomp.hpp"

#include <hdf5.h>

using namespace std;

namespace cv
{
namespace hdf
{

class HDF5Impl : public HDF5
{
public:

    HDF5Impl( String HDF5Filename );

    virtual ~HDF5Impl() { close(); };

    // close and release
    virtual void close( );

    /*
     * h5 generic
     */

    // check if object / link exists
    virtual bool hlexists( String label ) const;

    /*
     * h5 group
     */

    // create a group
    virtual void grcreate( String grlabel );

    /*
     *  cv::Mat
     */

    // get sizes of dataset
    virtual vector<int> dsgetsize( String dslabel, int dims_flag = H5_GETDIMS ) const;

    /* get data type of dataset */
    virtual int dsgettype( String dslabel ) const;

    // overload dscreate() #1
    virtual void dscreate( const int rows, const int cols, const int type, String dslabel ) const;

    // overload dscreate() #2
    virtual void dscreate( const int rows, const int cols, const int type, String dslabel,
             const int compresslevel ) const;

    // overload dscreate() #3
    virtual void dscreate( const int rows, const int cols, const int type, String dslabel,
             const int compresslevel, const vector<int>& dims_chunks ) const;

    /* create two dimensional single or mutichannel dataset */
    virtual void dscreate( const int rows, const int cols, const int type, String dslabel,
             const int compresslevel, const int* dims_chunks ) const;

    // overload dscreate() #1
    virtual void dscreate( const int n_dims, const int* sizes, const int type,
             String dslabel ) const;

    // overload dscreate() #2
    virtual void dscreate( const int n_dims, const int* sizes, const int type,
             String dslabel, const int compresslevel ) const;

    // overload dscreate() #3
    virtual void dscreate( const vector<int>& sizes, const int type, String dslabel,
             const int compresslevel = H5_NONE, const vector<int>& dims_chunks = vector<int>() ) const;

    /* create n-dimensional single or mutichannel dataset */
    virtual void dscreate( const int n_dims, const int* sizes, const int type,
             String dslabel, const int compresslevel, const int* dims_chunks ) const;

    // overload dswrite() #1
    virtual void dswrite( InputArray Array, String dslabel ) const;

    // overload dswrite() #2
    virtual void dswrite( InputArray Array, String dslabel, const int* dims_offset ) const;

    // overload dswrite() #3
    virtual void dswrite( InputArray Array, String dslabel, const vector<int>& dims_offset,
             const vector<int>& dims_counts = vector<int>() ) const;

    /* write into dataset */
    virtual void dswrite( InputArray Array, String dslabel,
             const int* dims_offset, const int* dims_counts ) const;

    // overload dsinsert() #1
    virtual void dsinsert( InputArray Array, String dslabel ) const;

    // overload dsinsert() #2
    virtual void dsinsert( InputArray Array, String dslabel, const int* dims_offset ) const;

    // overload dsinsert() #3
    virtual void dsinsert( InputArray Array, String dslabel,
             const vector<int>& dims_offset, const vector<int>& dims_counts = vector<int>() ) const;

    /* append / merge into dataset */
    virtual void dsinsert( InputArray Array, String dslabel,
             const int* dims_offset = NULL, const int* dims_counts = NULL ) const;

    // overload dsread() #1
    virtual void dsread( OutputArray Array, String dslabel ) const;

    // overload dsread() #2
    virtual void dsread( OutputArray Array, String dslabel, const int* dims_offset ) const;

    // overload dsread() #3
    virtual void dsread( OutputArray Array, String dslabel,
             const vector<int>& dims_offset, const vector<int>& dims_counts = vector<int>() ) const;

    // read from dataset
    virtual void dsread( OutputArray Array, String dslabel,
             const int* dims_offset, const int* dims_counts ) const;

    /*
     *  std::vector<cv::KeyPoint>
     */

    // get size of keypoints dataset
    virtual int kpgetsize( String kplabel, int dims_flag = H5_GETDIMS ) const;

    // create KeyPoint structure
    virtual void kpcreate( const int size, String kplabel,
             const int compresslevel = H5_NONE, const int chunks = H5_NONE ) const;

    // write KeyPoint structures
    virtual void kpwrite( const vector<KeyPoint> keypoints, String kplabel,
             const int offset = H5_NONE, const int counts = H5_NONE ) const;

    // append / merge KeyPoint structures
    virtual void kpinsert( const vector<KeyPoint> keypoints, String kplabel,
             const int offset = H5_NONE, const int counts = H5_NONE ) const;

    // read KeyPoint structure
    virtual void kpread( vector<KeyPoint>& keypoints, String kplabel,
             const int offset = H5_NONE, const int counts = H5_NONE ) const;

private:

    // store filename
    String m_hdf5_filename;

    // hdf5 file handler
    hid_t m_h5_file_id;

    // translate cvType -> h5Type
    inline hid_t GetH5type( int cvType ) const;

    // translate h5Type -> cvType
    inline int GetCVtype( hid_t h5Type ) const;

};

inline hid_t HDF5Impl::GetH5type( int cvType ) const
{
    hid_t h5Type = -1;

    switch ( CV_MAT_DEPTH( cvType ) )
    {
      case CV_64F:
        h5Type = H5T_NATIVE_DOUBLE;
        break;
      case CV_32F:
        h5Type = H5T_NATIVE_FLOAT;
        break;
      case CV_8U:
        h5Type = H5T_NATIVE_UCHAR;
        break;
      case CV_8S:
        h5Type = H5T_NATIVE_CHAR;
        break;
      case CV_16U:
        h5Type = H5T_NATIVE_USHORT;
        break;
      case CV_16S:
        h5Type = H5T_NATIVE_SHORT;
        break;
      case CV_32S:
        h5Type = H5T_NATIVE_INT;
        break;
      default:
        CV_Error( Error::StsInternal, "Unknown cvType." );
    }
    return h5Type;
}

inline int HDF5Impl::GetCVtype( hid_t h5Type ) const
{
    int cvType = -1;

    if      ( H5Tequal( h5Type, H5T_NATIVE_DOUBLE ) )
      cvType = CV_64F;
    else if ( H5Tequal( h5Type, H5T_NATIVE_FLOAT  ) )
      cvType = CV_32F;
    else if ( H5Tequal( h5Type, H5T_NATIVE_UCHAR  ) )
      cvType = CV_8U;
    else if ( H5Tequal( h5Type, H5T_NATIVE_CHAR   ) )
      cvType = CV_8S;
    else if ( H5Tequal( h5Type, H5T_NATIVE_USHORT ) )
      cvType = CV_16U;
    else if ( H5Tequal( h5Type, H5T_NATIVE_SHORT  ) )
      cvType = CV_16S;
    else if ( H5Tequal( h5Type, H5T_NATIVE_INT    ) )
      cvType = CV_32S;
    else
      CV_Error( Error::StsInternal, "Unknown H5Type." );

    return cvType;
}

HDF5Impl::HDF5Impl( String _hdf5_filename )
                  : m_hdf5_filename( _hdf5_filename )
{
    // save old
    // error handler
    void *errdata;
    H5E_auto2_t errfunc;
    hid_t stackid = H5E_DEFAULT;
    H5Eget_auto( stackid, &errfunc, &errdata );

    // turn off error handling
    H5Eset_auto( stackid, NULL, NULL );

    // check HDF5 file presence (err supressed)
    htri_t check = H5Fis_hdf5( m_hdf5_filename.c_str() );

    // restore previous error handler
    H5Eset_auto( stackid, errfunc, errdata );

    if ( check == 1 || check == 0 )
      // open the HDF5 file
      m_h5_file_id = H5Fopen( m_hdf5_filename.c_str(),
                            H5F_ACC_RDWR, H5P_DEFAULT );
    else if ( check == -1 )
      // file does not exist
      m_h5_file_id = H5Fcreate( m_hdf5_filename.c_str(),
                     H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );
    else
      CV_Error( Error::StsInternal, "Unknown file state." );
}

void HDF5Impl::close()
{
    if ( m_h5_file_id != -1 )
      H5Fclose( m_h5_file_id );
    // mark closed
    m_h5_file_id = -1;
}

/*
 * h5 generic
 */

bool HDF5Impl::hlexists( String label ) const
{
    bool exists = false;

    hid_t lid = H5Pcreate( H5P_LINK_ACCESS );
    if ( H5Lexists(m_h5_file_id, label.c_str(), lid) == 1 )
      exists = true;

    H5Pclose(lid);
    return exists;
}

/*
 * h5 group
 */

void HDF5Impl::grcreate( String grlabel )
{
  hid_t gid = H5Gcreate( m_h5_file_id, grlabel.c_str(),
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Gclose( gid );
}

/*
 * cv:Mat
 */

vector<int> HDF5Impl::dsgetsize( String dslabel, int dims_flag ) const
{
    // open dataset
    hid_t dsdata = H5Dopen( m_h5_file_id, dslabel.c_str(), H5P_DEFAULT );

    // get file space
    hid_t fspace = H5Dget_space( dsdata );

    // fetch rank
    int n_dims = H5Sget_simple_extent_ndims( fspace );

    // dims storage
    hsize_t *dims = new hsize_t[n_dims];

    // output storage
    vector<int> SizeVect(0);

    // fetch dims
    if ( dims_flag == H5_GETDIMS ||
         dims_flag == H5_GETMAXDIMS )
    {
      if ( dims_flag == H5_GETDIMS )
        H5Sget_simple_extent_dims( fspace, dims, NULL );
      else
        H5Sget_simple_extent_dims( fspace, NULL, dims );
      SizeVect.resize( n_dims );
    }
    else if ( dims_flag == H5_GETCHUNKDIMS )
    {
      // rank size
      int rank_chunk = -1;
      // fetch chunk size
      hid_t cparms = H5Dget_create_plist( dsdata );
      if ( H5D_CHUNKED == H5Pget_layout ( cparms ) )
      {
         rank_chunk = H5Pget_chunk ( cparms, n_dims, dims );
      }
      if ( rank_chunk > 0 )
        SizeVect.resize( n_dims );
    }
    else
      CV_Error( Error::StsInternal, "Unknown dimension flag." );

    // fill with size data
    for ( size_t d = 0; d < SizeVect.size(); d++ )
      SizeVect[d] = (int) dims[d];

    H5Dclose( dsdata );
    H5Sclose( fspace );

    delete [] dims;

    return SizeVect;
}

int HDF5Impl::dsgettype( String dslabel ) const
{
    hid_t h5type;

    // open dataset
    hid_t dsdata = H5Dopen( m_h5_file_id, dslabel.c_str(), H5P_DEFAULT );

    // get data type
    hid_t dstype = H5Dget_type( dsdata );

    int channs = 1;
    if ( H5Tget_class( dstype ) == H5T_ARRAY )
    {
      // fetch channs
      hsize_t ardims[1];
      H5Tget_array_dims( dstype, ardims );
      channs = (int)ardims[0];
      // fetch depth
      hid_t tsuper = H5Tget_super( dstype );
      h5type = H5Tget_native_type( tsuper, H5T_DIR_ASCEND );
      H5Tclose( tsuper );
    }
    else
      h5type = H5Tget_native_type( dstype, H5T_DIR_DESCEND );

    // convert to CVType
    int cvtype = GetCVtype( h5type );

    H5Tclose( dstype );
    H5Dclose( dsdata );

    return CV_MAKETYPE( cvtype, channs );
}

// overload
void HDF5Impl::dscreate( const int rows, const int cols, const int type,
                         String dslabel ) const
{
    // dataset dims
    int dsizes[2] = { rows, cols };

    // create the two dim array
    dscreate( 2, dsizes, type, dslabel, HDF5::H5_NONE, NULL );
}

// overload
void HDF5Impl::dscreate( const int rows, const int cols, const int type,
                         String dslabel, const int compresslevel ) const
{
    // dataset dims
    int dsizes[2] = { rows, cols };

    // create the two dim array
    dscreate( 2, dsizes, type, dslabel, compresslevel, NULL );
}

// overload
void HDF5Impl::dscreate( const int rows, const int cols, const int type,
                 String dslabel, const int compresslevel,
                 const vector<int>& dims_chunks ) const
{
    CV_Assert( &dims_chunks[0] == NULL || dims_chunks.size() == 2 );
    dscreate( rows, cols, type, dslabel, compresslevel, &dims_chunks[0] );
}

void HDF5Impl::dscreate( const int rows, const int cols, const int type,
                 String dslabel, const int compresslevel, const int* dims_chunks ) const
{
    // dataset dims
    int dsizes[2] = { rows, cols };

    // create the two dim array
    dscreate( 2, dsizes, type, dslabel, compresslevel, dims_chunks );
}

// overload
void HDF5Impl::dscreate( const int n_dims, const int* sizes, const int type,
                 String dslabel ) const
{
    dscreate( n_dims, sizes, type, dslabel, H5_NONE, NULL );
}

// overload
void HDF5Impl::dscreate( const int n_dims, const int* sizes, const int type,
                 String dslabel, const int compresslevel ) const
{
    dscreate( n_dims, sizes, type, dslabel, compresslevel, NULL );
}

// overload
void HDF5Impl::dscreate( const vector<int>& sizes, const int type,
                 String dslabel, const int compresslevel,
                 const vector<int>& dims_chunks ) const
{
    CV_Assert( &dims_chunks[0] == NULL || dims_chunks.size() == sizes.size() );

    const int n_dims = (int) sizes.size();
    dscreate( n_dims, &sizes[0], type, dslabel, compresslevel, &dims_chunks[0] );
}

void HDF5Impl::dscreate( const int n_dims, const int* sizes, const int type,
                 String dslabel, const int compresslevel, const int* dims_chunks ) const
{
    // compress valid H5_NONE, 0-9
    CV_Assert( compresslevel >= H5_NONE && compresslevel <= 9 );

    if ( hlexists( dslabel ) == true )
      CV_Error( Error::StsInternal, "Requested dataset already exists." );

    int channs = CV_MAT_CN( type );

    hsize_t *chunks = new hsize_t[n_dims];
    hsize_t *dsdims = new hsize_t[n_dims];
    hsize_t *maxdim = new hsize_t[n_dims];

    // dimension space
    for ( int d = 0; d < n_dims; d++ )
    {
      CV_Assert( sizes[d] >= H5_UNLIMITED );

      // dataset dimension
      if ( sizes[d] == H5_UNLIMITED )
      {
        CV_Assert( dims_chunks != NULL );

        dsdims[d] = 0;
        maxdim[d] = H5S_UNLIMITED;
      }
      else
      {
        dsdims[d] = sizes[d];
        maxdim[d] = sizes[d];
      }
      // default chunking
      if ( dims_chunks == NULL )
        chunks[d] = sizes[d];
      else
        chunks[d] = dims_chunks[d];
    }

    // create dataset space
    hid_t dspace = H5Screate_simple( n_dims, dsdims, maxdim );

    // create data property
    hid_t dsdcpl = H5Pcreate( H5P_DATASET_CREATE );

    // set properties
    if ( compresslevel >= 0 )
      H5Pset_deflate( dsdcpl, compresslevel );

    if ( dims_chunks != NULL || compresslevel >= 0 )
      H5Pset_chunk( dsdcpl, n_dims, chunks );

    // convert to h5 type
    hid_t dstype = GetH5type( type );

    // expand channs
    if ( channs > 1 )
    {
      hsize_t adims[1] = { (hsize_t)channs };
      dstype = H5Tarray_create( dstype, 1, adims );
    }

    // create data
    H5Dcreate( m_h5_file_id, dslabel.c_str(), dstype,
               dspace, H5P_DEFAULT, dsdcpl, H5P_DEFAULT );

    if ( channs > 1 )
      H5Tclose( dstype );

    delete [] chunks;
    delete [] dsdims;
    delete [] maxdim;

    H5Pclose( dsdcpl );
    H5Sclose( dspace );
}

// overload
void HDF5Impl::dsread( OutputArray Array, String dslabel ) const
{
    dsread( Array, dslabel, NULL, NULL );
}

// overload
void HDF5Impl::dsread( OutputArray Array, String dslabel,
             const int* dims_offset ) const
{
    dsread( Array, dslabel, dims_offset, NULL );
}

// overload
void HDF5Impl::dsread( OutputArray Array, String dslabel,
             const vector<int>& dims_offset,
             const vector<int>& dims_counts ) const
{
    dsread( Array, dslabel, &dims_offset[0], &dims_counts[0] );
}

void HDF5Impl::dsread( OutputArray Array, String dslabel,
             const int* dims_offset, const int* dims_counts ) const
{
    // only Mat support
    CV_Assert( Array.isMat() );

    hid_t h5type;

    // open the HDF5 dataset
    hid_t dsdata = H5Dopen( m_h5_file_id, dslabel.c_str(), H5P_DEFAULT );

    // get data type
    hid_t dstype = H5Dget_type( dsdata );

    int channs = 1;
    if ( H5Tget_class( dstype ) == H5T_ARRAY )
    {
      // fetch channs
      hsize_t ardims[1];
      H5Tget_array_dims( dstype, ardims );
      channs = (int) ardims[0];
      // fetch depth
      hid_t tsuper = H5Tget_super( dstype );
      h5type = H5Tget_native_type( tsuper, H5T_DIR_ASCEND );
      H5Tclose( tsuper );
    } else
      h5type = H5Tget_native_type( dstype, H5T_DIR_ASCEND );

    int dType = GetCVtype( h5type );

    // get file space
    hid_t fspace = H5Dget_space( dsdata );

    // fetch rank
    int n_dims = H5Sget_simple_extent_ndims( fspace );

    // fetch dims
    hsize_t *dsdims = new hsize_t[n_dims];
    H5Sget_simple_extent_dims( fspace, dsdims, NULL );

    // set amount by custom offset
    if ( dims_offset != NULL )
    {
      for ( int d = 0; d < n_dims; d++ )
        dsdims[d] -= dims_offset[d];
    }

    // set custom amount of data
    if ( dims_counts != NULL )
    {
      for ( int d = 0; d < n_dims; d++ )
        dsdims[d] = dims_counts[d];
    }

    // get memory write window
    int *mxdims = new int[n_dims];
    hsize_t *foffset = new hsize_t[n_dims];
    for ( int d = 0; d < n_dims; d++ )
    {
      foffset[d] = 0;
      mxdims[d] = (int) dsdims[d];
    }

    // allocate persistent Mat
    Array.create( n_dims, mxdims, CV_MAKETYPE(dType, channs) );

    // get blank data space
    hid_t dspace = H5Screate_simple( n_dims, dsdims, NULL );

    // get matrix write window
    H5Sselect_hyperslab( dspace, H5S_SELECT_SET,
                         foffset, NULL, dsdims, NULL );

    // set custom offsets
    if ( dims_offset != NULL )
    {
      for ( int d = 0; d < n_dims; d++ )
        foffset[d] = dims_offset[d];
    }

    // get a file read window
    H5Sselect_hyperslab( fspace, H5S_SELECT_SET,
                         foffset, NULL, dsdims, NULL );

    // read from DS
    Mat matrix = Array.getMat();
    H5Dread( dsdata, dstype, dspace, fspace, H5P_DEFAULT, matrix.data );

    delete [] dsdims;
    delete [] mxdims;
    delete [] foffset;

    H5Tclose( dstype );
    H5Sclose( dspace );
    H5Sclose( fspace );
    H5Dclose( dsdata );
}

// overload
void HDF5Impl::dswrite( InputArray Array, String dslabel ) const
{
    dswrite( Array, dslabel, NULL, NULL );
}
// overload
void HDF5Impl::dswrite( InputArray Array, String dslabel,
             const int* dims_offset ) const
{
    dswrite( Array, dslabel, dims_offset, NULL );
}
// overload
void HDF5Impl::dswrite( InputArray Array, String dslabel,
             const vector<int>& dims_offset,
             const vector<int>& dims_counts ) const
{
    dswrite( Array, dslabel, &dims_offset[0], &dims_counts[0] );
}

void HDF5Impl::dswrite( InputArray Array, String dslabel,
             const int* dims_offset, const int* dims_counts ) const
{
    // only Mat support
    CV_Assert( Array.isMat() );

    Mat matrix = Array.getMat();

    // memory array should be compact
    CV_Assert( matrix.isContinuous() );

    int n_dims = matrix.dims;
    int channs = matrix.channels();

    int *dsizes = new int[n_dims];
    hsize_t *dsdims = new hsize_t[n_dims];
    hsize_t *offset = new hsize_t[n_dims];
    // replicate Mat dimensions
    for ( int d = 0; d < n_dims; d++ )
    {
      offset[d] = 0;
      dsizes[d] = matrix.size[d];
      dsdims[d] = matrix.size[d];
    }

    // pre-create dataset if needed
    if ( hlexists( dslabel ) == false )
      dscreate( n_dims, dsizes, matrix.type(), dslabel );

    // set custom amount of data
    if ( dims_counts != NULL )
    {
      for ( int d = 0; d < n_dims; d++ )
        dsdims[d] = dims_counts[d];
    }

    // open dataset
    hid_t dsdata = H5Dopen( m_h5_file_id, dslabel.c_str(), H5P_DEFAULT );

    // create input data space
    hid_t dspace = H5Screate_simple( n_dims, dsdims, NULL );

    // set custom offsets
    if ( dims_offset != NULL )
    {
      for ( int d = 0; d < n_dims; d++ )
        offset[d] = dims_offset[d];
    }

    // create offset write window space
    hid_t fspace = H5Dget_space( dsdata );
    H5Sselect_hyperslab( fspace, H5S_SELECT_SET,
                         offset, NULL, dsdims, NULL );

    // convert type
    hid_t dstype = GetH5type( matrix.type() );

    // expand channs
    if ( matrix.channels() > 1 )
    {
      hsize_t adims[1] = { (hsize_t)channs };
      dstype = H5Tarray_create( dstype, 1, adims );
    }

    // write into dataset
    H5Dwrite( dsdata, dstype, dspace, fspace,
              H5P_DEFAULT, matrix.data );

    if ( matrix.channels() > 1 )
      H5Tclose( dstype );

    delete [] dsizes;
    delete [] dsdims;
    delete [] offset;

    H5Sclose( dspace );
    H5Sclose( fspace );
    H5Dclose( dsdata );
}

// overload
void HDF5Impl::dsinsert( InputArray Array, String dslabel ) const
{
    dsinsert( Array, dslabel, NULL, NULL );
}

// overload
void HDF5Impl::dsinsert( InputArray Array, String dslabel,
             const int* dims_offset ) const
{
    dsinsert( Array, dslabel, dims_offset, NULL );
}

// overload
void HDF5Impl::dsinsert( InputArray Array, String dslabel,
             const vector<int>& dims_offset,
             const vector<int>& dims_counts ) const
{
    dsinsert( Array, dslabel, &dims_offset[0], &dims_counts[0] );
}

void HDF5Impl::dsinsert( InputArray Array, String dslabel,
             const int* dims_offset, const int* dims_counts ) const
{
    // only Mat support
    CV_Assert( Array.isMat() );

    // check dataset exists
    if ( hlexists( dslabel ) == false )
      CV_Error( Error::StsInternal, "Dataset does not exist." );

    Mat matrix = Array.getMat();

    // memory array should be compact
    CV_Assert( matrix.isContinuous() );

    int n_dims = matrix.dims;
    int channs = matrix.channels();

    hsize_t *dsdims = new hsize_t[n_dims];
    hsize_t *offset = new hsize_t[n_dims];
    // replicate Mat dimensions
    for ( int d = 0; d < n_dims; d++ )
    {
      offset[d] = 0;
      dsdims[d] = matrix.size[d];
    }

    // set custom amount of data
    if ( dims_counts != NULL )
    {
      for ( int d = 0; d < n_dims; d++ )
      {
        CV_Assert( dims_counts[d] <= matrix.size[d] );
        dsdims[d] = dims_counts[d];
      }
    }

    // open dataset
    hid_t dsdata = H5Dopen( m_h5_file_id, dslabel.c_str(), H5P_DEFAULT );

    // create input data space
    hid_t dspace = H5Screate_simple( n_dims, dsdims, NULL );

    // set custom offsets
    if ( dims_offset != NULL )
    {
      for ( int d = 0; d < n_dims; d++ )
        offset[d] = dims_offset[d];
    }

    // get actual file space and dims
    hid_t fspace = H5Dget_space( dsdata );
    int f_dims = H5Sget_simple_extent_ndims( fspace );
    hsize_t *fsdims = new hsize_t[f_dims];
    H5Sget_simple_extent_dims( fspace, fsdims, NULL );
    H5Sclose( fspace );

    CV_Assert( f_dims == n_dims );

    // compute new extents
    hsize_t *nwdims = new hsize_t[n_dims];
    for ( int d = 0; d < n_dims; d++ )
    {
      // init
      nwdims[d] = 0;
      // add offset
      if ( dims_offset != NULL )
        nwdims[d] += dims_offset[d];
      // add counts or matrixsize
      if ( dims_counts != NULL )
        nwdims[d] += dims_counts[d];
      else
        nwdims[d] += matrix.size[d];

      // clamp back if smaller
      if ( nwdims[d] < fsdims[d] )
        nwdims[d] = fsdims[d];
    }

    // extend dataset
    H5Dextend( dsdata, nwdims );

    // get the extended data space
    fspace = H5Dget_space( dsdata );

    H5Sselect_hyperslab( fspace, H5S_SELECT_SET,
                         offset, NULL, dsdims, NULL );

    // convert type
    hid_t dstype = GetH5type( matrix.type() );

    // expand channs
    if ( matrix.channels() > 1 )
    {
      hsize_t adims[1] = { (hsize_t)channs };
      dstype = H5Tarray_create( dstype, 1, adims );
    }

    // write into dataset
    H5Dwrite( dsdata, dstype, dspace, fspace,
              H5P_DEFAULT, matrix.data );

    if ( matrix.channels() > 1 )
      H5Tclose( dstype );

    delete [] dsdims;
    delete [] offset;
    delete [] fsdims;
    delete [] nwdims;

    H5Sclose( dspace );
    H5Sclose( fspace );
    H5Dclose( dsdata );
}

/*
 *  std::vector<cv::KeyPoint>
 */

int HDF5Impl::kpgetsize( String kplabel, int dims_flag ) const
{
    vector<int> sizes = dsgetsize( kplabel, dims_flag );

    CV_Assert( sizes.size() == 1 );

    return sizes[0];
}

void HDF5Impl::kpcreate( const int size, String kplabel,
             const int compresslevel, const int chunks ) const
{
    // size valid
    CV_Assert( size >= H5_UNLIMITED );

    // valid chunks
    CV_Assert( chunks == H5_NONE || chunks > 0 );

    // compress valid -1, 0-9
    CV_Assert( compresslevel >= H5_NONE && compresslevel <= 9 );

    if ( hlexists( kplabel ) == true )
      CV_Error( Error::StsInternal, "Requested dataset already exists." );

    hsize_t dchunk[1];
    hsize_t dsdims[1];
    hsize_t maxdim[1];

    // dataset dimension
    if ( size == H5_UNLIMITED )
    {
      dsdims[0] = 0;
      maxdim[0] = H5S_UNLIMITED;
    }
    else
    {
      dsdims[0] = size;
      maxdim[0] = size;
    }

    // default chunking
    if ( chunks == H5_NONE )
      if ( size == H5_UNLIMITED )
        dchunk[0] = 1;
      else
        dchunk[0] = size;
    else
      dchunk[0] = chunks;

    // dataset compound type
    hid_t dstype = H5Tcreate( H5T_COMPOUND, sizeof( KeyPoint ) );
    H5Tinsert( dstype, "xpos",     HOFFSET( KeyPoint, pt.x     ), H5T_NATIVE_FLOAT );
    H5Tinsert( dstype, "ypos",     HOFFSET( KeyPoint, pt.y     ), H5T_NATIVE_FLOAT );
    H5Tinsert( dstype, "size",     HOFFSET( KeyPoint, size     ), H5T_NATIVE_FLOAT );
    H5Tinsert( dstype, "angle",    HOFFSET( KeyPoint, angle    ), H5T_NATIVE_FLOAT );
    H5Tinsert( dstype, "response", HOFFSET( KeyPoint, response ), H5T_NATIVE_FLOAT );
    H5Tinsert( dstype, "octave",   HOFFSET( KeyPoint, octave   ), H5T_NATIVE_INT32 );
    H5Tinsert( dstype, "class_id", HOFFSET( KeyPoint, class_id ), H5T_NATIVE_INT32 );

    // create dataset space
    hid_t dspace = H5Screate_simple( 1, dsdims, maxdim );

    // create data property
    hid_t dsdcpl = H5Pcreate( H5P_DATASET_CREATE );

    // set properties
    if ( compresslevel >= 0 )
      H5Pset_deflate( dsdcpl, compresslevel );

    // if chunking or compression
    if ( dchunk[0] > 0 || compresslevel >= 0 )
      H5Pset_chunk( dsdcpl, 1, dchunk );

    // create data
    H5Dcreate( m_h5_file_id, kplabel.c_str(), dstype,
               dspace, H5P_DEFAULT, dsdcpl, H5P_DEFAULT );

    H5Tclose( dstype );
    H5Pclose( dsdcpl );
    H5Sclose( dspace );
}

void HDF5Impl::kpwrite( const vector<KeyPoint> keypoints, String kplabel,
             const int offset, const int counts ) const
{
    CV_Assert( keypoints.size() > 0 );

    int dskdims[1];
    hsize_t dsddims[1];
    hsize_t doffset[1];

    // replicate vector dimension
    doffset[0] = 0;
    dsddims[0] = keypoints.size();
    dskdims[0] = (int)keypoints.size();

    // pre-create dataset if needed
    if ( hlexists( kplabel ) == false )
      kpcreate( dskdims[0], kplabel );

    // set custom amount of data
    if ( counts != H5_NONE )
      dsddims[0] = counts;

    // open dataset
    hid_t dsdata = H5Dopen( m_h5_file_id, kplabel.c_str(), H5P_DEFAULT );

    // create input data space
    hid_t dspace = H5Screate_simple( 1, dsddims, NULL );

    // set custom offsets
    if ( offset != H5_NONE )
      doffset[0] = offset;

    // create offset write window space
    hid_t fspace = H5Dget_space( dsdata );
    H5Sselect_hyperslab( fspace, H5S_SELECT_SET,
                         doffset, NULL, dsddims, NULL );

    // memory compound type
    hid_t mmtype = H5Tcreate( H5T_COMPOUND, sizeof( KeyPoint ) );
    H5Tinsert( mmtype, "xpos",     HOFFSET( KeyPoint, pt.x     ), H5T_NATIVE_FLOAT );
    H5Tinsert( mmtype, "ypos",     HOFFSET( KeyPoint, pt.y     ), H5T_NATIVE_FLOAT );
    H5Tinsert( mmtype, "size",     HOFFSET( KeyPoint, size     ), H5T_NATIVE_FLOAT );
    H5Tinsert( mmtype, "angle",    HOFFSET( KeyPoint, angle    ), H5T_NATIVE_FLOAT );
    H5Tinsert( mmtype, "response", HOFFSET( KeyPoint, response ), H5T_NATIVE_FLOAT );
    H5Tinsert( mmtype, "octave",   HOFFSET( KeyPoint, octave   ), H5T_NATIVE_INT32 );
    H5Tinsert( mmtype, "class_id", HOFFSET( KeyPoint, class_id ), H5T_NATIVE_INT32 );

    // write into dataset
    H5Dwrite( dsdata, mmtype, dspace, fspace, H5P_DEFAULT, &keypoints[0] );

    H5Tclose( mmtype );
    H5Sclose( dspace );
    H5Sclose( fspace );
    H5Dclose( dsdata );
}

void HDF5Impl::kpinsert( const vector<KeyPoint> keypoints, String kplabel,
             const int offset, const int counts ) const
{
    CV_Assert( keypoints.size() > 0 );

    // check dataset exists
    if ( hlexists( kplabel ) == false )
      CV_Error( Error::StsInternal, "Dataset does not exist." );

    hsize_t dsddims[1];
    hsize_t doffset[1];

    // replicate vector dimension
    doffset[0] = 0;
    dsddims[0] = keypoints.size();

    // set custom amount of data
    if ( counts != H5_NONE )
      dsddims[0] = counts;

    // open dataset
    hid_t dsdata = H5Dopen( m_h5_file_id, kplabel.c_str(), H5P_DEFAULT );

    // create input data space
    hid_t dspace = H5Screate_simple( 1, dsddims, NULL );

    // set custom offsets
    if ( offset != H5_NONE )
      doffset[0] = offset;

    // get actual file space and dims
    hid_t fspace = H5Dget_space( dsdata );
    int f_dims = H5Sget_simple_extent_ndims( fspace );
    hsize_t *fsdims = new hsize_t[f_dims];
    H5Sget_simple_extent_dims( fspace, fsdims, NULL );
    H5Sclose( fspace );

    CV_Assert( f_dims == 1 );

    // compute new extents
    hsize_t nwdims[1] = { 0 };
    // add offset
    if ( offset != H5_NONE )
      nwdims[0] += offset;
    // add counts or matrixsize
    if ( counts != H5_NONE )
      nwdims[0] += counts;
    else
      nwdims[0] += keypoints.size();

    // clamp back if smaller
    if ( nwdims[0] < fsdims[0] )
      nwdims[0] = fsdims[0];

    // extend dataset
    H5Dextend( dsdata, nwdims );

    // get the extended data space
    fspace = H5Dget_space( dsdata );

    H5Sselect_hyperslab( fspace, H5S_SELECT_SET,
                         doffset, NULL, dsddims, NULL );

    // memory compound type
    hid_t mmtype = H5Tcreate( H5T_COMPOUND, sizeof( KeyPoint ) );
    H5Tinsert( mmtype, "xpos",     HOFFSET( KeyPoint, pt.x     ), H5T_NATIVE_FLOAT );
    H5Tinsert( mmtype, "ypos",     HOFFSET( KeyPoint, pt.y     ), H5T_NATIVE_FLOAT );
    H5Tinsert( mmtype, "size",     HOFFSET( KeyPoint, size     ), H5T_NATIVE_FLOAT );
    H5Tinsert( mmtype, "angle",    HOFFSET( KeyPoint, angle    ), H5T_NATIVE_FLOAT );
    H5Tinsert( mmtype, "response", HOFFSET( KeyPoint, response ), H5T_NATIVE_FLOAT );
    H5Tinsert( mmtype, "octave",   HOFFSET( KeyPoint, octave   ), H5T_NATIVE_INT32 );
    H5Tinsert( mmtype, "class_id", HOFFSET( KeyPoint, class_id ), H5T_NATIVE_INT32 );

    // write into dataset
    H5Dwrite( dsdata, mmtype, dspace, fspace, H5P_DEFAULT, &keypoints[0] );

    delete [] fsdims;

    H5Tclose( mmtype );
    H5Sclose( dspace );
    H5Sclose( fspace );
    H5Dclose( dsdata );
}

void HDF5Impl::kpread( vector<KeyPoint>& keypoints, String kplabel,
             const int offset, const int counts ) const
{
    CV_Assert( keypoints.size() == 0 );

    // open the HDF5 dataset
    hid_t dsdata = H5Dopen( m_h5_file_id, kplabel.c_str(), H5P_DEFAULT );

    // get data type
    hid_t dstype = H5Dget_type( dsdata );

    // get file space
    hid_t fspace = H5Dget_space( dsdata );

    // fetch rank
    int n_dims = H5Sget_simple_extent_ndims( fspace );

    CV_Assert( n_dims == 1 );

    // fetch dims
    hsize_t dsddims[1];
    H5Sget_simple_extent_dims( fspace, dsddims, NULL );

    // set amount by custom offset
    if ( offset != H5_NONE )
      dsddims[0] -= offset;

    // set custom amount of data
    if ( counts != H5_NONE )
      dsddims[0] = counts;

    // get memory write window
    hsize_t foffset[1] = { 0 };

    // allocate keypoints vector
    keypoints.resize( dsddims[0] );

    // get blank data space
    hid_t dspace = H5Screate_simple( 1, dsddims, NULL );

    // get matrix write window
    H5Sselect_hyperslab( dspace, H5S_SELECT_SET,
                         foffset, NULL, dsddims, NULL );

    // set custom offsets
    if ( offset != H5_NONE )
      foffset[0] = offset;

    // get a file read window
    H5Sselect_hyperslab( fspace, H5S_SELECT_SET,
                         foffset, NULL, dsddims, NULL );

    // read from DS
    H5Dread( dsdata, dstype, dspace, fspace, H5P_DEFAULT, &keypoints[0] );

    H5Tclose( dstype );
    H5Sclose( dspace );
    H5Sclose( fspace );
    H5Dclose( dsdata );
}

CV_EXPORTS Ptr<HDF5> open( String HDF5Filename )
{
    return makePtr<HDF5Impl>( HDF5Filename );
}

} // end namespace hdf
} // end namespace cv
