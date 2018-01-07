Creating, Writing and Reading Datasets {#tutorial_hdf_create_read_write_datasets}
===============================

Goal
----

This tutorial shows you:
 - How to create a dataset?
 - How to write a `cv::Mat` to a dataset?
 - How to read a `cv::Mat` from a dataset?

@note Currently, it supports only reading and writing cv::Mat and the matrix should be continuous
in memory. Supports for other data types have not been implemented yet.

Source Code
----

The following code demonstrates writing a single channel
matrix and a two-channel matrix to datasets and then reading them
back.

You can download the code from [here][1] or find it in the file
`modules/hdf/samples/create_read_write_datasets.cpp` of the opencv_contrib source code library.

@snippet samples/create_read_write_datasets.cpp tutorial

Explanation
----

The first step for creating a dataset is to open the file

@snippet samples/create_read_write_datasets.cpp tutorial_open_file

For the function `write_root_group_single_channel()`, since
the dataset name is `/single`, which is inside the root group, we can use

@snippet samples/create_read_write_datasets.cpp tutorial_write_root_single_channel

to write the data directly to the dataset without the need of creating
it beforehand. Because it is created inside cv::hdf::HDF5::dswrite()
automatically.

@warning This applies only to datasets that reside inside the root group.

Of course, we can create the dataset by ourselves:

@snippet samples/create_read_write_datasets.cpp tutorial_create_dataset

To read data from a dataset, we use

@snippet samples/create_read_write_datasets.cpp tutorial_read_dataset

by specifying the name of the dataset.

We can check that the data read out is exactly the data written before by using

@snippet samples/create_read_write_datasets.cpp tutorial_check_result

Results
----

Figure 1 shows the result visualized using the tool HDFView for the file
`root_group_single_channel`. The results
of matrices for datasets that are not the direct children of the root group
are given in Figure 2 and Figure 3, respectively.

![Figure 1: Result for writing a single channel matrix to a dataset inside the root group](pics/root_group_single_channel.png)

![Figure 2: Result for writing a single channel matrix to a dataset not in the root group](pics/single_channel.png)

![Figure 3: Result for writing a two-channel matrix to a dataset not in the root group](pics/two_channels.png)


[1]: https://github.com/opencv/opencv_contrib/tree/master/modules/hdf/samples/create_read_write_datasets.cpp
