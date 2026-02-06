// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/**
 * @file test_hdf5.cpp
 * @author Fangjun Kuang <csukuangfj dot at gmail dot com>
 * @date December 2017
 */
#include "opencv2/core/cvdef.h"
#include "opencv2/hdf/hdf5.hpp"
#include "test_precomp.hpp"
#include <iostream>

namespace opencv_test { namespace {

struct HDF5_Test : public testing::Test
{
    virtual void SetUp()
    {
        m_filename = "test.h5";

        // 0 1 2
        // 3 4 5
        m_single_channel.create(2, 3, CV_32F);
        for (size_t i = 0; i < m_single_channel.total(); i++)
        {
            ((float*)m_single_channel.data)[i] = i;
        }

        // 0 1 2 3 4  5
        // 6 7 8 9 10 11
        m_two_channels.create(2, 3, CV_32SC2);
        for (size_t i = 0; i < m_two_channels.total()*m_two_channels.channels(); i++)
        {
            ((int*)m_two_channels.data)[i] = (int)i;
        }
    }

    //! Remove the hdf5 file
    void reset()
    {
        remove(m_filename.c_str());
    }

    String m_filename; //!< filename for testing
    Ptr<hdf::HDF5> m_hdf_io; //!< HDF5 file pointer
    Mat m_single_channel; //!< single channel matrix for test
    Mat m_two_channels; //!< two-channel matrix for test
};

TEST_F(HDF5_Test, create_a_single_group)
{
    reset();

    String group_name = "parent";
    m_hdf_io = hdf::open(m_filename);
    m_hdf_io->grcreate(group_name);

    EXPECT_EQ(m_hdf_io->hlexists(group_name), true);
    EXPECT_EQ(m_hdf_io->hlexists("child"), false);

    // It should fail since it creates a group with an existing name
    EXPECT_ANY_THROW(m_hdf_io->grcreate(group_name));

    m_hdf_io->close();
}


TEST_F(HDF5_Test, create_a_child_group)
{
    reset();

    String parent = "parent";
    String child = parent + "/child";
    m_hdf_io = hdf::open(m_filename);
    m_hdf_io->grcreate(parent);
    m_hdf_io->grcreate(child);

    EXPECT_EQ(m_hdf_io->hlexists(parent), true);
    EXPECT_EQ(m_hdf_io->hlexists(child), true);

    m_hdf_io->close();
}

TEST_F(HDF5_Test, create_dataset)
{
    reset();

    String dataset_single_channel = "/single";
    String dataset_two_channels = "/dual";

    m_hdf_io = hdf::open(m_filename);

    m_hdf_io->dscreate(m_single_channel.rows,
                       m_single_channel.cols,
                       m_single_channel.type(),
                       dataset_single_channel);

    m_hdf_io->dscreate(m_two_channels.rows,
                       m_two_channels.cols,
                       m_two_channels.type(),
                       dataset_two_channels);

    EXPECT_EQ(m_hdf_io->hlexists(dataset_single_channel), true);
    EXPECT_EQ(m_hdf_io->hlexists(dataset_two_channels), true);

    std::vector<int> dims;

    dims = m_hdf_io->dsgetsize(dataset_single_channel, hdf::HDF5::H5_GETDIMS);
    EXPECT_EQ(dims.size(), (size_t)2);
    EXPECT_EQ(dims[0], m_single_channel.rows);
    EXPECT_EQ(dims[1], m_single_channel.cols);

    dims = m_hdf_io->dsgetsize(dataset_two_channels, hdf::HDF5::H5_GETDIMS);
    EXPECT_EQ(dims.size(), (size_t)2);
    EXPECT_EQ(dims[0], m_two_channels.rows);
    EXPECT_EQ(dims[1], m_two_channels.cols);

    int type;
    type = m_hdf_io->dsgettype(dataset_single_channel);
    EXPECT_EQ(type, m_single_channel.type());

    type = m_hdf_io->dsgettype(dataset_two_channels);
    EXPECT_EQ(type, m_two_channels.type());

    m_hdf_io->close();
}


TEST_F(HDF5_Test, write_read_dataset_1)
{
    reset();

    String dataset_single_channel = "/single";
    String dataset_two_channels = "/dual";

    m_hdf_io = hdf::open(m_filename);

    // since the dataset is under the root group, it is created by dswrite() automatically.
    m_hdf_io->dswrite(m_single_channel, dataset_single_channel);
    m_hdf_io->dswrite(m_two_channels, dataset_two_channels);

    EXPECT_EQ(m_hdf_io->hlexists(dataset_single_channel), true);
    EXPECT_EQ(m_hdf_io->hlexists(dataset_two_channels), true);

    // read single channel matrix
    Mat single;
    m_hdf_io->dsread(single, dataset_single_channel);
    EXPECT_EQ(single.type(), m_single_channel.type());
    EXPECT_EQ(single.size(), m_single_channel.size());
    EXPECT_LE(cvtest::norm(single, m_single_channel, NORM_L2), 1e-10);

    // read dual channel matrix
    Mat dual;
    m_hdf_io->dsread(dual, dataset_two_channels);
    EXPECT_EQ(dual.type(), m_two_channels.type());
    EXPECT_EQ(dual.size(), m_two_channels.size());
    EXPECT_LE(cvtest::norm(dual, m_two_channels, NORM_L2), 1e-10);

    m_hdf_io->close();
}

TEST_F(HDF5_Test, write_read_dataset_2)
{
    reset();
    // create the dataset manually if it is not inside
    // the root group

    String parent = "/parent";

    String dataset_single_channel = parent + "/single";
    String dataset_two_channels = parent + "/dual";

    m_hdf_io = hdf::open(m_filename);

    m_hdf_io->grcreate(parent);
    EXPECT_EQ(m_hdf_io->hlexists(parent), true);

    m_hdf_io->dscreate(m_single_channel.rows,
                       m_single_channel.cols,
                       m_single_channel.type(),
                       dataset_single_channel);

    m_hdf_io->dscreate(m_two_channels.rows,
                       m_two_channels.cols,
                       m_two_channels.type(),
                       dataset_two_channels);

    EXPECT_EQ(m_hdf_io->hlexists(dataset_single_channel), true);
    EXPECT_EQ(m_hdf_io->hlexists(dataset_two_channels), true);

    m_hdf_io->dswrite(m_single_channel, dataset_single_channel);
    m_hdf_io->dswrite(m_two_channels, dataset_two_channels);

    EXPECT_EQ(m_hdf_io->hlexists(dataset_single_channel), true);
    EXPECT_EQ(m_hdf_io->hlexists(dataset_two_channels), true);

    // read single channel matrix
    Mat single;
    m_hdf_io->dsread(single, dataset_single_channel);
    EXPECT_EQ(single.type(), m_single_channel.type());
    EXPECT_EQ(single.size(), m_single_channel.size());
    EXPECT_LE(cvtest::norm(single, m_single_channel, NORM_L2), 1e-10);

    // read dual channel matrix
    Mat dual;
    m_hdf_io->dsread(dual, dataset_two_channels);
    EXPECT_EQ(dual.type(), m_two_channels.type());
    EXPECT_EQ(dual.size(), m_two_channels.size());
    EXPECT_LE(cvtest::norm(dual, m_two_channels, NORM_L2), 1e-10);

    m_hdf_io->close();
}

TEST_F(HDF5_Test, test_attribute)
{
    reset();

    String attr_name = "test attribute name";
    int attr_value = 0x12345678;

    m_hdf_io = hdf::open(m_filename);
    EXPECT_EQ(m_hdf_io->atexists(attr_name), false);

    m_hdf_io->atwrite(attr_value, attr_name);
    EXPECT_ANY_THROW(m_hdf_io->atwrite(attr_value, attr_name)); // error! it already exists

    EXPECT_EQ(m_hdf_io->atexists(attr_name), true);

    int expected_attr_value;
    m_hdf_io->atread(&expected_attr_value, attr_name);
    EXPECT_EQ(attr_value, expected_attr_value);

    m_hdf_io->atdelete(attr_name);
    EXPECT_ANY_THROW(m_hdf_io->atdelete(attr_name)); // error! Delete non-existed attribute

    EXPECT_EQ(m_hdf_io->atexists(attr_name), false);

    m_hdf_io->close();
}

TEST_F(HDF5_Test, test_attribute_int)
{
    reset();

    String attr_name = "test int";
    int attr_value = 0x12345678;

    m_hdf_io = hdf::open(m_filename);

    m_hdf_io->atwrite(attr_value, attr_name);

    int expected_attr_value;
    m_hdf_io->atread(&expected_attr_value, attr_name);
    EXPECT_EQ(attr_value, expected_attr_value);

    m_hdf_io->close();
}

TEST_F(HDF5_Test, test_attribute_double)
{
    reset();

    String attr_name = "test double";
    double attr_value = 123.456789;

    m_hdf_io = hdf::open(m_filename);

    m_hdf_io->atwrite(attr_value, attr_name);

    double expected_attr_value;
    m_hdf_io->atread(&expected_attr_value, attr_name);
    EXPECT_NEAR(attr_value, expected_attr_value, 1e-9);

    m_hdf_io->close();
}

TEST_F(HDF5_Test, test_attribute_String)
{
    reset();

    String attr_name = "test-String";
    String attr_value = "----_______----Hello HDF5----_______----\n";

    m_hdf_io = hdf::open(m_filename);

    m_hdf_io->atwrite(attr_value, attr_name);

    String got_attr_value;
    m_hdf_io->atread(&got_attr_value, attr_name);
    EXPECT_EQ(attr_value, got_attr_value);

    m_hdf_io->close();
}

TEST_F(HDF5_Test, test_attribute_String_empty)
{
    reset();

    String attr_name = "test-empty-string";
    String attr_value;

    m_hdf_io = hdf::open(m_filename);

    m_hdf_io->atwrite(attr_value, attr_name);

    String got_attr_value;
    m_hdf_io->atread(&got_attr_value, attr_name);
    EXPECT_EQ(attr_value, got_attr_value);

    m_hdf_io->close();
}

TEST_F(HDF5_Test, test_attribute_InutArray_OutputArray_2d)
{
    reset();

    String attr_name = "test-InputArray-OutputArray-2d";
    cv::Mat attr_value;

    std::vector<int> depth_vec;
    depth_vec.push_back(CV_8U); depth_vec.push_back(CV_8S);
    depth_vec.push_back(CV_16U); depth_vec.push_back(CV_16S);
    depth_vec.push_back(CV_32S); depth_vec.push_back(CV_32F);
    depth_vec.push_back(CV_64F);

    std::vector<int> channel_vec;
    channel_vec.push_back(1); channel_vec.push_back(2);
    channel_vec.push_back(3); channel_vec.push_back(4);
    channel_vec.push_back(5); channel_vec.push_back(6);
    channel_vec.push_back(7); channel_vec.push_back(8);
    channel_vec.push_back(9); channel_vec.push_back(10);

    std::vector<std::vector<int> > dim_vec;
    std::vector<int> dim_2d;
    dim_2d.push_back(2); dim_2d.push_back(3);
    dim_vec.push_back(dim_2d);

    std::vector<int> dim_3d;
    dim_3d.push_back(2);
    dim_3d.push_back(3);
    dim_3d.push_back(4);
    dim_vec.push_back(dim_3d);

    std::vector<int> dim_4d;
    dim_4d.push_back(2); dim_4d.push_back(3);
    dim_4d.push_back(4); dim_4d.push_back(5);
    dim_vec.push_back(dim_4d);

    Mat expected_attr_value;

    m_hdf_io = hdf::open(m_filename);
    for (size_t i = 0; i < depth_vec.size(); i++)
    for (size_t j = 0; j < channel_vec.size(); j++)
    for (size_t k = 0; k < dim_vec.size(); k++)
    {
        if (m_hdf_io->atexists(attr_name))
            m_hdf_io->atdelete(attr_name);

        attr_value.create(dim_vec[k], CV_MAKETYPE(depth_vec[i], channel_vec[j]));
        randu(attr_value, 0, 255);

        m_hdf_io->atwrite(attr_value, attr_name);
        m_hdf_io->atread(expected_attr_value, attr_name);

        double diff = cvtest::norm(attr_value, expected_attr_value, NORM_L2);
        EXPECT_LE(diff, 1e-6);

        EXPECT_EQ(attr_value.size, expected_attr_value.size);
        EXPECT_EQ(attr_value.type(), expected_attr_value.type());
    }

    m_hdf_io->close();
}

#ifdef CV_16F

TEST_F(HDF5_Test, write_read_dataset_CV_16F)
{
    if (!hdf::HDF5_has_f16_support())
    {
        std::cout << "HDF5 float16 support is not available in this build." << std::endl;
        return;
    }
    reset();
    String dataset_half = "/half";
    m_hdf_io = hdf::open(m_filename);
    int rows = 2, cols = 3;
    Mat mat_half(rows, cols, CV_16F);
    for (int i = 0; i < rows * cols; ++i)
        ((cv::hfloat*)mat_half.data)[i] = cv::hfloat(i * 0.5f);
    m_hdf_io->dswrite(mat_half, dataset_half);
    EXPECT_EQ(m_hdf_io->hlexists(dataset_half), true);
    Mat mat_half_read;
    m_hdf_io->dsread(mat_half_read, dataset_half);
    EXPECT_EQ(mat_half_read.type(), CV_16F);
    EXPECT_EQ(mat_half_read.size(), mat_half.size());
    double diff = cvtest::norm(mat_half, mat_half_read, NORM_L2);
    EXPECT_LE(diff, 1e-3);
    m_hdf_io->close();
}

TEST_F(HDF5_Test, write_read_dataset_CV_16FC2)
{
    if (!hdf::HDF5_has_f16_support())
    {
        std::cout << "HDF5 float16 support is not available in this build." << std::endl;
        return;
    }
    reset();
    String dataset_half_array = "/halfc2";
    m_hdf_io = hdf::open(m_filename);
    int rows = 2, cols = 3, channels = 2;
    Mat mat_half(rows, cols, CV_MAKETYPE(CV_16F, channels));
    for (int i = 0; i < rows * cols * channels; ++i)
        ((cv::hfloat*)mat_half.data)[i] = cv::hfloat(i * 0.125f);
    m_hdf_io->dscreate(rows, cols, mat_half.type(), dataset_half_array);
    m_hdf_io->dswrite(mat_half, dataset_half_array);
    Mat mat_half_read;
    m_hdf_io->dsread(mat_half_read, dataset_half_array, NULL, NULL);
    EXPECT_EQ(mat_half_read.type(), mat_half.type());
    EXPECT_EQ(mat_half_read.size(), mat_half.size());
    double diff = cvtest::norm(mat_half, mat_half_read, NORM_L2);
    EXPECT_LE(diff, 1e-3);
    m_hdf_io->close();
}

TEST_F(HDF5_Test, attribute_CV_16FC2_matrix)
{
    if (!hdf::HDF5_has_f16_support())
    {
        std::cout << "HDF5 float16 support is not available in this build." << std::endl;
        return;
    }
    reset();
    String attr_name = "half_array";
    int rows = 2, cols = 3, channels = 2;
    Mat mat_half(rows, cols, CV_MAKETYPE(CV_16F, channels));
    for (int i = 0; i < rows * cols * channels; ++i)
        ((cv::hfloat*)mat_half.data)[i] = cv::hfloat(i * 0.25f);
    m_hdf_io = hdf::open(m_filename);
    m_hdf_io->atwrite(mat_half, attr_name);
    Mat mat_half_read;
    m_hdf_io->atread(mat_half_read, attr_name);
    EXPECT_EQ(mat_half_read.type(), mat_half.type());
    EXPECT_EQ(mat_half_read.size(), mat_half.size());
    double diff = cvtest::norm(mat_half, mat_half_read, NORM_L2);
    EXPECT_LE(diff, 1e-3);
    m_hdf_io->close();
}

TEST_F(HDF5_Test, attribute_CV_16F_matrix)
{
    if (!hdf::HDF5_has_f16_support())
    {
        std::cout << "HDF5 float16 support is not available in this build." << std::endl;
        return;
    }
    reset();
    String attr_name = "half_array";
    int rows = 2, cols = 3, channels = 2;
    Mat mat_half(rows, cols, CV_16F);
    for (int i = 0; i < rows * cols * channels; ++i)
        ((cv::hfloat*)mat_half.data)[i] = cv::hfloat(i * 0.25f);
    m_hdf_io = hdf::open(m_filename);
    m_hdf_io->atwrite(mat_half, attr_name);
    Mat mat_half_read;
    m_hdf_io->atread(mat_half_read, attr_name);
    EXPECT_EQ(mat_half_read.type(), mat_half.type());
    EXPECT_EQ(mat_half_read.size(), mat_half.size());
    double diff = cvtest::norm(mat_half, mat_half_read, NORM_L2);
    EXPECT_LE(diff, 1e-3);
    m_hdf_io->close();
}

TEST_F(HDF5_Test, attribute_CV_16F_scalar)
{
    if (!hdf::HDF5_has_f16_support())
    {
        std::cout << "HDF5 float16 support is not available in this build." << std::endl;
        return;
    }
    reset();
    String attr_name = "half_float";
    cv::hfloat attr_value = cv::hfloat(123.456789f);

    m_hdf_io = hdf::open(m_filename);

    m_hdf_io->atwrite(attr_value, attr_name);

    double expected_attr_value;
    m_hdf_io->atread(&expected_attr_value, attr_name);
    EXPECT_NEAR(attr_value, expected_attr_value, 1e-9);
    m_hdf_io->close();

}

#endif

}} // namespace
