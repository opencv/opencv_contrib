Quasi dense Stereo {#tutorial_qds_quasi_dense_stereo}
==================

Goal
----

In this tutorial you will learn how to

-   Configure a QuasiDenseStero object
-   Compute dense Stereo correspondences.

@include ./samples/dense_disparity.cpp

## Explanation:

The program loads a stereo image pair.

After importing the images.
@snippet ./samples/dense_disparity.cpp load
We need to know the frame size of a single image, in order to create an instance of a `QuasiDesnseStereo` object.

@snippet ./samples/dense_disparity.cpp create

Because we didn't specify the second argument in the constructor, the `QuasiDesnseStereo` object will
load default parameters.

We can then pass the imported stereo images in the process method like this
@snippet ./samples/dense_disparity.cpp process

The process method contains most of the functionality of the class and does two main things.
-   Computes a sparse stereo based in "Good Features to Track" and "pyramidal Lucas-Kanade" flow algorithm
-   Based on those sparse stereo points, densifies the stereo correspondences using Quasi Dense Stereo method.

After the execution of `process()` we can display the disparity Image of the stereo.
@snippet ./samples/dense_disparity.cpp disp


At this point we can also extract all the corresponding points using `getDenseMatches()` method and export them in a file.
@snippet ./samples/dense_disparity.cpp export
