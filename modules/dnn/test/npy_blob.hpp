#ifndef __OPENCV_DNN_TEST_NPY_BLOB_HPP__
#define __OPENCV_DNN_TEST_NPY_BLOB_HPP__
#include "test_precomp.hpp"

#ifdef __GNUC__
#   pragma GCC diagnostic ignored "-Wunused-parameter"
#   pragma GCC diagnostic push
#endif

#include "cnpy.h"

#ifdef __GNUC__
#   pragma GCC diagnostic pop
#endif

inline cv::dnn::Blob blobFromNPY(const cv::String &path)
{
    cnpy::NpyArray npyBlob = cnpy::npy_load(path.c_str());

    cv::dnn::Blob blob;
    blob.fill((int)npyBlob.shape.size(), (int*)&npyBlob.shape[0], CV_32F, npyBlob.data);

    npyBlob.destruct();
    return blob;
}

inline void saveBlobToNPY(cv::dnn::Blob &blob, const cv::String &path)
{
    cv::Vec4i shape = blob.shape4();
    cnpy::npy_save(path.c_str(), blob.ptr<float>(), (unsigned*)&shape[0], 4);
}

#endif
