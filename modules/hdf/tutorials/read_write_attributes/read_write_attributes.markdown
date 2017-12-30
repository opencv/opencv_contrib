Reading and Writing Attributes{#tutorial_hdf_read_write_attributes}
===============================

Goal
----
This tutorial shows you:
 - How to write attributes?
 - How to read attributes?

@note Although attributes can be associated with groups and datasets, only attributes
with the root group are implemented in OpenCV. Supported attribute types are
`int`, `double`, `cv::String` and `cv::InputArray` (only for continuous arrays).

Source Code
----

The following code demonstrates reading and writing attributes
inside the root group with data types `cv::Mat`, `cv::String`, `int`
and `double`.

You can download the code from [here][1] or find it in the file
`modules/hdf/samples/read_write_attributes.cpp` of the opencv_contrib source code library.

@snippet samples/read_write_attributes.cpp tutorial

Explanation
----

The first step is to open the HDF5 file:

@snippet samples/read_write_attributes.cpp tutorial_open_file

Then we use cv::hdf::HDF5::atwrite() to write attributes by specifying its value and name:

@snippet samples/read_write_attributes.cpp tutorial_write_mat

@warning Before writing an attribute, we have to make sure that
the attribute does not exist using cv::hdf::HDF5::atexists().

To read an attribute, we use cv::hdf::HDF5::atread() by specifying the attribute name

@snippet samples/read_write_attributes.cpp tutorial_read_mat

In the end, we have to close the HDF file

@snippet samples/read_write_attributes.cpp tutorial_close_file

Results
----

Figure 1 and Figure 2 give the results visualized using the tool HDFView.

![Figure 1: Attributes of the root group](pics/attributes-file.png)

![Figure 2: Detailed attribute information](pics/attributes-details.png)

[1]: https://github.com/opencv/opencv_contrib/tree/master/modules/hdf/samples/read_write_attributes.cpp
