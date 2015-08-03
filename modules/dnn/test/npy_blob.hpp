#ifndef __OPENCV_DNN_TEST_NPY_BLOB_HPP__
#define __OPENCV_DNN_TEST_NPY_BLOB_HPP__
#include "test_precomp.hpp"
#include "cnpy.h"

inline cv::dnn::Blob blobFromNPY(const cv::String &path)
{
    cnpy::NpyArray npyBlob = cnpy::npy_load(path.c_str());
    cv::dnn::BlobShape shape((int)npyBlob.shape.size(), (int*)&npyBlob.shape[0]);

    cv::dnn::Blob blob;
    blob.fill(shape, CV_32F, npyBlob.data);

    npyBlob.destruct();
    return blob;
}

inline void saveBlobToNPY(cv::dnn::Blob &blob, const cv::String &path)
{
    cv::Vec4i shape = blob.shape4();
    cnpy::npy_save(path.c_str(), blob.ptr<float>(), (unsigned*)&shape[0], 4);
}

#endif
