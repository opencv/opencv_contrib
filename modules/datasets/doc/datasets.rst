*******************************************************
datasets. Framework for working with different datasets
*******************************************************

.. highlight:: cpp

The datasets module includes classes for working with different datasets: load data, evaluate different algorithms on them, contains benchmarks, etc.

It is planned to have:

 * basic: loading code for all datasets to help start work with them.
 * next stage: quick benchmarks for all datasets to show how to solve them using OpenCV and implement evaluation code.
 * finally: implement on OpenCV state-of-the-art algorithms, which solve these tasks.

.. toctree::
    :hidden:

    ar_hmdb
    ar_sports
    fr_adience
    fr_lfw
    gr_chalearn
    gr_skig
    hpe_humaneva
    hpe_parse
    ir_affine
    ir_robot
    is_bsds
    is_weizmann
    msm_epfl
    msm_middlebury
    or_imagenet
    or_mnist
    or_sun
    pd_caltech
    slam_kitti
    slam_tumindoor
    tr_chars
    tr_svt

Action Recognition
------------------

    :doc:`ar_hmdb` [#f1]_

    :doc:`ar_sports`

Face Recognition
----------------

    :doc:`fr_adience`

    :doc:`fr_lfw` [#f1]_

Gesture Recognition
-------------------

    :doc:`gr_chalearn`

    :doc:`gr_skig`

Human Pose Estimation
---------------------

    :doc:`hpe_humaneva`

    :doc:`hpe_parse`

Image Registration
------------------

    :doc:`ir_affine`

    :doc:`ir_robot`

Image Segmentation
------------------

    :doc:`is_bsds`

    :doc:`is_weizmann`

Multiview Stereo Matching
-------------------------

    :doc:`msm_epfl`

    :doc:`msm_middlebury`

Object Recognition
------------------

    :doc:`or_imagenet`

    :doc:`or_mnist` [#f2]_

    :doc:`or_sun`

Pedestrian Detection
--------------------

    :doc:`pd_caltech` [#f2]_

SLAM
----

    :doc:`slam_kitti`

    :doc:`slam_tumindoor`

Text Recognition
----------------

    :doc:`tr_chars`

    :doc:`tr_svt` [#f1]_

*Footnotes*

 .. [#f1] Benchmark implemented
 .. [#f2] Not used in Vision Challenge
