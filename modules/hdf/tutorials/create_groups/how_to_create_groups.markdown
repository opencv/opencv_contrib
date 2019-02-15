Creating Groups {#tutorial_hdf_create_groups}
===============================

Goal
----

This tutorial will show you:
 - How to create a HDF5 file?
 - How to create a group?
 - How to check whether a given group exists or not?
 - How to create a subgroup?

Source Code
----

The following code creates two groups: `Group1` and `SubGroup1`, where
`SubGroup1` is a child of `Group1`.

You can download the code from [here][1] or find it in the file
`modules/hdf/samples/create_groups.cpp` of the opencv_contrib source code library.

@snippet samples/create_groups.cpp tutorial

Explanation
----

First, we create a HDF5 file

@snippet samples/create_groups.cpp tutorial_create_file

If the given file does not exist, it will be created. Otherwise, it is open for read and write.

Next, we create the group `Group1`

@snippet samples/create_groups.cpp tutorial_create_group

Note that we have to check whether `/Group1` exists or not using
the function cv::hdf::HDF5::hlexists() before creating it. You can not create
a group with an existing name. Otherwise, an error will occur.

Then, we create the subgroup named `Subgroup1`. In order to
indicate that it is a sub group of `Group1`, we have to
use the group name `/Group1/SubGroup1`:

@snippet samples/create_groups.cpp tutorial_create_subgroup

Note that before creating a subgroup, we have to make sure
that its parent group exists. Otherwise, an error will occur.

In the end, we have to close the file

@snippet samples/create_groups.cpp tutorial_close_file

Result
----

There are many tools that can be used to inspect a given HDF file, such
as HDFView and h5dump. If you are using Ubuntu, you can install
them with the following commands:

@code
sudo apt-get install hdf5-tools hdfview
@endcode

There are also binaries available from the The HDF Group official website <https://support.hdfgroup.org/HDF5/Tutor/tools.html>.

The following figure shows the result visualized with the tool HDFView:

![Figure 1: Results of creating groups and subgroups](pics/create_groups.png)

The output for `h5dump` is:

@code
$ h5dump mytest.h5
HDF5 "mytest.h5" {
GROUP "/" {
   GROUP "Group1" {
      GROUP "SubGroup1" {
      }
   }
}
}
@endcode

[1]: https://github.com/opencv/opencv_contrib/tree/master/modules/hdf/samples/create_groups.cpp
