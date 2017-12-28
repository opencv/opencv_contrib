// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/**
 * @file test_hdf5.cpp
 * @author Fangjun Kuang <csukuangfj dot at gmail dot com>
 * @date December 2017
 *
 */
#include<stdio.h> // for remove()

#include "test_precomp.hpp"
#include <vector>

using namespace cv;

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
    EXPECT_NEAR(norm(single-m_single_channel), 0, 1e-10);

    // read dual channel matrix
    Mat dual;
    m_hdf_io->dsread(dual, dataset_two_channels);
    EXPECT_EQ(dual.type(), m_two_channels.type());
    EXPECT_EQ(dual.size(), m_two_channels.size());
    EXPECT_NEAR(norm(dual-m_two_channels), 0, 1e-10);

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
    EXPECT_NEAR(norm(single-m_single_channel), 0, 1e-10);

    // read dual channel matrix
    Mat dual;
    m_hdf_io->dsread(dual, dataset_two_channels);
    EXPECT_EQ(dual.type(), m_two_channels.type());
    EXPECT_EQ(dual.size(), m_two_channels.size());
    EXPECT_NEAR(norm(dual-m_two_channels), 0, 1e-10);

    m_hdf_io->close();
}
