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
// Copyright (C) 2014, Itseez Inc, all rights reserved.
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
// In no event shall the Itseez Inc or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifdef HAVE_HDF5
#include "opencv2/datasets/util_hdf5.hpp"
#include "opencv2/datasets/util.hpp"

namespace cv
{
namespace datasets
{

using namespace std;

void writeFileToH5(const string &imagePath, const string &name, hid_t grp_id)
{
    FILE *f = fopen(imagePath.c_str(), "rb");
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = new char[size];
    size_t res = fread(buf, 1, size, f);
    if (size != (long)res) // suppress warning
    {
        res = 0;
    }

    hsize_t dim2[1];
    dim2[0] = size;
    hid_t space_id = H5Screate_simple(1, dim2, NULL);
    hid_t dset_id = H5Dcreate(grp_id, name.c_str(), H5T_STD_U8LE, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_STD_U8LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(dset_id);
    H5Sclose(space_id);

    fclose(f);
    delete[] buf;
}

void write1DToH5(hid_t loc_id, hid_t type_id, const string &name, const void *buf, int num)
{
    hsize_t dim[1];
    dim[0] = num;
    hid_t space_id = H5Screate_simple(1, dim, NULL);
    hid_t dset_id = H5Dcreate(loc_id, name.c_str(), type_id, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(dset_id);
    H5Sclose(space_id);
}

}
}
#endif
