/**
 * @file create_read_write_datasets.cpp
 * @author Fangjun Kuang <csukuangfj dot at gmail dot com>
 * @date December 2017
 *
 * @brief It demonstrates how to create a dataset,  how
 * to write a cv::Mat to the dataset and how to
 * read a cv::Mat from it.
 *
 */

//! [tutorial]
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/hdf.hpp>

using namespace cv;

static void write_root_group_single_channel()
{
    String filename = "root_group_single_channel.h5";
    String dataset_name = "/single"; // Note that it is a child of the root group /

    // prepare data
    Mat data;
    data = (cv::Mat_<float>(2, 3) << 0, 1, 2, 3, 4, 5, 6);

    //! [tutorial_open_file]
    Ptr<hdf::HDF5> h5io = hdf::open(filename);
    //! [tutorial_open_file]

    //! [tutorial_write_root_single_channel]
    // write data to the given dataset
    // the dataset "/single" is created automatically, since it is a child of the root
    h5io->dswrite(data, dataset_name);
    //! [tutorial_write_root_single_channel]

    //! [tutorial_read_dataset]
    Mat expected;
    h5io->dsread(expected, dataset_name);
    //! [tutorial_read_dataset]

    //! [tutorial_check_result]
    double diff = norm(data - expected);
    CV_Assert(abs(diff) < 1e-10);
    //! [tutorial_check_result]

    h5io->close();
}

static void write_single_channel()
{
    String filename = "single_channel.h5";
    String parent_name = "/data";
    String dataset_name = parent_name + "/single";

    // prepare data
    Mat data;
    data = (cv::Mat_<float>(2, 3) << 0, 1, 2, 3, 4, 5);

    Ptr<hdf::HDF5> h5io = hdf::open(filename);

    //! [tutorial_create_dataset]
    // first we need to create the parent group
    if (!h5io->hlexists(parent_name)) h5io->grcreate(parent_name);

    // create the dataset if it not exists
    if (!h5io->hlexists(dataset_name)) h5io->dscreate(data.rows, data.cols, data.type(), dataset_name);
    //! [tutorial_create_dataset]

    // the following is the same with the above function write_root_group_single_channel()

    h5io->dswrite(data, dataset_name);

    Mat expected;
    h5io->dsread(expected, dataset_name);

    double diff = norm(data - expected);
    CV_Assert(abs(diff) < 1e-10);

    h5io->close();
}

/*
 * creating, reading and writing multiple-channel matrices
 * are the same with single channel matrices
 */
static void write_multiple_channels()
{
    String filename = "two_channels.h5";
    String parent_name = "/data";
    String dataset_name = parent_name + "/two_channels";

    // prepare data
    Mat data(2, 3, CV_32SC2);
    for (size_t i = 0; i < data.total()*data.channels(); i++)
        ((int*) data.data)[i] = (int)i;

    Ptr<hdf::HDF5> h5io = hdf::open(filename);

    // first we need to create the parent group
    if (!h5io->hlexists(parent_name)) h5io->grcreate(parent_name);

    // create the dataset if it not exists
    if (!h5io->hlexists(dataset_name)) h5io->dscreate(data.rows, data.cols, data.type(), dataset_name);

    // the following is the same with the above function write_root_group_single_channel()

    h5io->dswrite(data, dataset_name);

    Mat expected;
    h5io->dsread(expected, dataset_name);

    double diff = norm(data - expected);
    CV_Assert(abs(diff) < 1e-10);

    h5io->close();
}

int main()
{
    write_root_group_single_channel();

    write_single_channel();

    write_multiple_channels();

    return 0;
}
//! [tutorial]
