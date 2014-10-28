EPFL Multi-View Stereo
======================
.. ocv:class:: MSM_epfl

Implements loading dataset:

_`"EPFL Multi-View Stereo"`: http://cvlabwww.epfl.ch/~strecha/multiview/denseMVS.html

.. note:: Usage

 1. From link above download dataset files: castle_dense\\castle_dense_large\\castle_entry\\fountain\\herzjesu_dense\\herzjesu_dense_large_bounding\\cameras\\images\\p.tar.gz.

 2. Unpack them in separate folder for each object. For example, for "fountain", in folder fountain/ : fountain_dense_bounding.tar.gz -> bounding/, fountain_dense_cameras.tar.gz -> camera/, fountain_dense_images.tar.gz -> png/, fountain_dense_p.tar.gz -> P/

 3. To load data, for example, for "fountain", run: ./opencv/build/bin/example_datasets_msm_epfl -p=/home/user/path_to_unpacked_folder/fountain/

**References:**

.. [Strecha08] C. Strecha, W. von Hansen, L. Van Gool, P. Fua, U. Thoennessen. On Benchmarking Camera Calibration and Multi-View Stereo for High Resolution Imagery. CVPR, 2008

