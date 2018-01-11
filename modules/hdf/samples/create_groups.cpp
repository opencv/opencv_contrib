/**
 * @file create_groups.cpp
 * @author Fangjun Kuang <csukuangfj dot at gmail dot com>
 * @date December 2017
 *
 * @brief It demonstrates how to create HDF5 groups and subgroups.
 *
 * Basic steps:
 *  1. Use hdf::open to create a HDF5 file
 *  2. Use HDF5::hlexists to check if a group exists or not
 *  3. Use HDF5::grcreate to create a group by specifying its name
 *  4. Use hdf::close to close a HDF5 file after modifying it
 *
 */

//! [tutorial]
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/hdf.hpp>

using namespace cv;

int main()
{
    //! [create_group]

    //! [tutorial_create_file]
    Ptr<hdf::HDF5> h5io = hdf::open("mytest.h5");
    //! [tutorial_create_file]

    //! [tutorial_create_group]
    // "/" means the root group, which is always present
    if (!h5io->hlexists("/Group1"))
       h5io->grcreate("/Group1");
    else
       std::cout << "/Group1 has already been created, skip it.\n";
    //! [tutorial_create_group]

    //! [tutorial_create_subgroup]
    // Note that Group1 has been created above, otherwise exception will occur
    if (!h5io->hlexists("/Group1/SubGroup1"))
       h5io->grcreate("/Group1/SubGroup1");
    else
       std::cout << "/Group1/SubGroup1 has already been created, skip it.\n";
    //! [tutorial_create_subgroup]

    //! [tutorial_close_file]
    h5io->close();
    //! [tutorial_close_file]

    //! [create_group]

    return 0;
}
//! [tutorial]
