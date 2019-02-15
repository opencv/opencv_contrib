/**
 * @file read_write_attributes.cpp
 * @author Fangjun Kuang <csukuangfj dot at gmail dot com>
 * @date December 2017
 *
 * @brief It demonstrates how to read and write attributes inside the
 * root group.
 *
 * Currently, only the following datatypes can be used as attributes:
 *  - cv::String
 *  - int
 *  - double
 *  - cv::InputArray (n-d continuous multichannel arrays)
 *
 * Although HDF supports associating attributes with both datasets and groups,
 * only support for the root group is implemented by OpenCV at present.
 */

//! [tutorial]
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/hdf.hpp>

using namespace cv;

static void read_write_attributes()
{
    String filename = "attributes.h5";

    //! [tutorial_open_file]
    Ptr<hdf::HDF5> h5io = hdf::open(filename);
    //! [tutorial_open_file]

    //! [tutorial_write_mat]
    String attr_mat_name = "array attribute";
    Mat attr_mat;
    attr_mat = (cv::Mat_<float>(2, 3) << 0, 1, 2, 3, 4, 5, 6);
    if (!h5io->atexists(attr_mat_name))
        h5io->atwrite(attr_mat, attr_mat_name);
    //! [tutorial_write_mat]

    //! [snippets_write_str]
    String attr_str_name = "string attribute";
    String attr_str = "Hello HDF5 from OpenCV!";
    if (!h5io->atexists(attr_str_name))
        h5io->atwrite(attr_str, attr_str_name);
    //! [snippets_write_str]

    String attr_int_name = "int attribute";
    int attr_int = 123456;
    if (!h5io->atexists(attr_int_name))
        h5io->atwrite(attr_int, attr_int_name);

    String attr_double_name = "double attribute";
    double attr_double = 45678.123;
    if (!h5io->atexists(attr_double_name))
        h5io->atwrite(attr_double, attr_double_name);

    // read attributes
    Mat expected_attr_mat;
    int expected_attr_int;
    double expected_attr_double;

    //! [snippets_read_str]
    String expected_attr_str;
    h5io->atread(&expected_attr_str, attr_str_name);
    //! [snippets_read_str]

    //! [tutorial_read_mat]
    h5io->atread(expected_attr_mat, attr_mat_name);
    //! [tutorial_read_mat]
    h5io->atread(&expected_attr_int, attr_int_name);
    h5io->atread(&expected_attr_double, attr_double_name);

    // check results
    CV_Assert(norm(attr_mat - expected_attr_mat) < 1e-10);
    CV_Assert(attr_str.compare(expected_attr_str) == 0);
    CV_Assert(attr_int == expected_attr_int);
    CV_Assert(fabs(attr_double - expected_attr_double) < 1e-10);

    //! [tutorial_close_file]
    h5io->close();
    //! [tutorial_close_file]
}

int main()
{
    read_write_attributes();

    return 0;
}
//! [tutorial]
