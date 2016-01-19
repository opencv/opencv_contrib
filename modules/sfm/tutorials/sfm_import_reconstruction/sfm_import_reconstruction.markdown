Import Reconstruction {#tutorial_sfm_import_reconstruction}
=====================

Goal
----

In this tutorial you will learn how to import a reconstruction from a given file obtained with Bundler [1]:

-   Load a file containing a set of cameras and 3D points.
-   Show obtained results using Viz.


Code
----

@include sfm/samples/import_reconstruction.cpp

Results
-------

The following picture shows a reconstruction from la *Sagrada Familia* (BCN) using dataset [2].

![](pics/import_sagrada_familia.png)

[1] [http://www.cs.cornell.edu/~snavely/bundler](http://www.cs.cornell.edu/~snavely/bundler)

[2] Penate Sanchez, A. and Moreno-Noguer, F. and Andrade Cetto, J. and Fleuret, F. (2014). LETHA: Learning from High Quality Inputs for 3D Pose Estimation in Low Quality Images. Proceedings of the International Conference on 3D vision (3DV).
[URL](http://www.iri.upc.edu/research/webprojects/pau/datasets/sagfam)
