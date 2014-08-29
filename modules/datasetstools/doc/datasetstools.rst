*********************************************************
datasetstools. Tools for working with different datasets.
*********************************************************

.. highlight:: cpp

The datasetstools module includes classes for working with different datasets.

First version of this module was implemented for **Fall2014 OpenCV Challenge**.

Action Recognition
------------------

ar_hmdb
=======
.. ocv:class:: ar_hmdb

Implements loading dataset:

_`"HMDB: A Large Human Motion Database"`: http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/

.. note:: Usage

 1. From link above download dataset files: hmdb51_org.rar & test_train_splits.rar.

 2. Unpack them.

 3. To load data run: ./opencv/build/bin/example_datasetstools_ar_hmdb -p=/home/user/path_to_unpacked_folders/

ar_sports
=========
.. ocv:class:: ar_sports

Implements loading dataset:

_`"Sports-1M Dataset"`: http://cs.stanford.edu/people/karpathy/deepvideo/

.. note:: Usage

 1. From link above download dataset files (git clone https://code.google.com/p/sports-1m-dataset/).

 2. To load data run: ./opencv/build/bin/example_datasetstools_ar_sports -p=/home/user/path_to_downloaded_folders/

Face Recognition
----------------

fr_lfw
======
.. ocv:class:: fr_lfw

Implements loading dataset:

_`"Labeled Faces in the Wild-a"`: http://www.openu.ac.il/home/hassner/data/lfwa/

.. note:: Usage

 1. From link above download dataset file: lfwa.tar.gz.

 2. Unpack it.

 3. To load data run: ./opencv/build/bin/example_datasetstools_fr_lfw -p=/home/user/path_to_unpacked_folder/lfw2/

Gesture Recognition
-------------------

gr_chalearn
===========
.. ocv:class:: gr_chalearn

Implements loading dataset:

_`"ChaLearn Looking at People"`: http://gesture.chalearn.org/

.. note:: Usage

 1. Follow instruction from site above, download files for dataset "Track 3: Gesture Recognition": Train1.zip-Train5.zip, Validation1.zip-Validation3.zip (Register on site: www.codalab.org and accept the terms and conditions of competition: https://www.codalab.org/competitions/991#learn_the_details There are three mirrors for downloading dataset files. When I downloaded data only mirror: "Universitat Oberta de Catalunya" works).

 2. Unpack train archives Train1.zip-Train5.zip to one folder (currently loading validation files wasn't implemented)

 3. To load data run: ./opencv/build/bin/example_datasetstools_gr_chalearn -p=/home/user/path_to_unpacked_folder/

gr_skig
=======
.. ocv:class:: gr_skig

Implements loading dataset:

_`"Sheffield Kinect Gesture Dataset"`: http://lshao.staff.shef.ac.uk/data/SheffieldKinectGesture.htm

.. note:: Usage

 1. From link above download dataset files: subject1_dep.7z-subject6_dep.7z, subject1_rgb.7z-subject6_rgb.7z.

 2. Unpack them.

 3. To load data run: ./opencv/build/bin/example_datasetstools_gr_skig -p=/home/user/path_to_unpacked_folders/

Human Pose Estimation
---------------------

hpe_parse
=========
.. ocv:class:: hpe_parse

Implements loading dataset:

_`"PARSE Dataset"`: http://www.ics.uci.edu/~dramanan/papers/parse/

.. note:: Usage

 1. From link above download dataset file: people.zip.

 2. Unpack it.

 3. To load data run: ./opencv/build/bin/example_datasetstools_hpe_parse -p=/home/user/path_to_unpacked_folder/people_all/

Image Registration
------------------

ir_affine
=========
.. ocv:class:: ir_affine

Implements loading dataset:

_`"Affine Covariant Regions Datasets"`: http://www.robots.ox.ac.uk/~vgg/data/data-aff.html

.. note:: Usage

 1. From link above download dataset files: bark\\bikes\\boat\\graf\\leuven\\trees\\ubc\\wall.tar.gz.

 2. Unpack them.

 3. To load data, for example, for "bark", run: ./opencv/build/bin/example_datasetstools_ir_affine -p=/home/user/path_to_unpacked_folder/bark/

ir_robot
========
.. ocv:class:: ir_robot

Implements loading dataset:

_`"Robot Data Set"`: http://roboimagedata.compute.dtu.dk/?page_id=24

.. note:: Usage

 1. From link above download files for dataset "Point Feature Data Set – 2010": SET001_6.tar.gz-SET055_60.tar.gz (there are two data sets: - Full resolution images (1200×1600), ~500 Gb and - Half size image (600×800), ~115 Gb.)
 2. Unpack them to one folder.

 3. To load data run: ./opencv/build/bin/example_datasetstools_ir_robot -p=/home/user/path_to_unpacked_folder/

Image Segmentation
------------------

is_bsds
=======
.. ocv:class:: is_bsds

Implements loading dataset:

_`"The Berkeley Segmentation Dataset and Benchmark"`: https://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/

.. note:: Usage

 1. From link above download dataset files: BSDS300-human.tgz & BSDS300-images.tgz.

 2. Unpack them.

 3. To load data run: ./opencv/build/bin/example_datasetstools_is_bsds -p=/home/user/path_to_unpacked_folder/BSDS300/

is_weizmann
===========
.. ocv:class:: is_weizmann

Implements loading dataset:

_`"Weizmann Segmentation Evaluation Database"`: http://www.wisdom.weizmann.ac.il/~vision/Seg_Evaluation_DB/

.. note:: Usage

 1. From link above download dataset files: Weizmann_Seg_DB_1obj.ZIP & Weizmann_Seg_DB_2obj.ZIP.

 2. Unpack them.

 3. To load data, for example, for 1 object dataset, run: ./opencv/build/bin/example_datasetstools_is_weizmann -p=/home/user/path_to_unpacked_folder/1obj/

Multiview Stereo Matching
-------------------------

msm_epfl
========
.. ocv:class:: msm_epfl

Implements loading dataset:

_`"EPFL Multi-View Stereo"`: http://cvlabwww.epfl.ch/~strecha/multiview/denseMVS.html

.. note:: Usage

 1. From link above download dataset files: castle_dense\\castle_dense_large\\castle_entry\\fountain\\herzjesu_dense\\herzjesu_dense_large_bounding\\cameras\\images\\p.tar.gz.

 2. Unpack them in separate folder for each object. For example, for "fountain", in folder fountain/ : fountain_dense_bounding.tar.gz -> bounding/, fountain_dense_cameras.tar.gz -> camera/, fountain_dense_images.tar.gz -> png/, fountain_dense_p.tar.gz -> P/

 3. To load data, for example, for "fountain", run: ./opencv/build/bin/example_datasetstools_msm_epfl -p=/home/user/path_to_unpacked_folder/fountain/

msm_middlebury
==============
.. ocv:class:: msm_middlebury

Implements loading dataset:

_`"Stereo – Middlebury Computer Vision"`: http://vision.middlebury.edu/mview/

.. note:: Usage

 1. From link above download dataset files: dino\\dinoRing\\dinoSparseRing\\temple\\templeRing\\templeSparseRing.zip

 2. Unpack them.

 3. To load data, for example "temple" dataset, run: ./opencv/build/bin/example_datasetstools_msm_middlebury -p=/home/user/path_to_unpacked_folder/temple/

Object Recognition
------------------

or_imagenet
===========
.. ocv:class:: or_imagenet

Implements loading dataset:

_`"ImageNet"`: http://www.image-net.org/

Currently implemented loading full list with urls. Planned to implement dataset from ILSVRC challenge. 

.. note:: Usage

 1. From link above download dataset file: imagenet_fall11_urls.tgz

 2. Unpack it.

 3. To load data run: ./opencv/build/bin/example_datasetstools_or_imagenet -p=/home/user/path_to_unpacked_file/

or_sun
======
.. ocv:class:: or_sun

Implements loading dataset:

_`"SUN Database"`: http://sun.cs.princeton.edu/

Currently implemented loading "Scene Recognition Benchmark. SUN397". Planned to implement also "Object Detection Benchmark. SUN2012". 

.. note:: Usage

 1. From link above download dataset file: SUN397.tar

 2. Unpack it.

 3. To load data run: ./opencv/build/bin/example_datasetstools_or_sun -p=/home/user/path_to_unpacked_folder/SUN397/

SLAM
----

slam_kitti
==========
.. ocv:class:: slam_kitti

Implements loading dataset:

_`"KITTI Vision Benchmark"`: http://www.cvlibs.net/datasets/kitti/eval_odometry.php

.. note:: Usage

 1. From link above download "Odometry" dataset files: data_odometry_gray\\data_odometry_color\\data_odometry_velodyne\\data_odometry_poses\\data_odometry_calib.zip.

 2. Unpack data_odometry_poses.zip, it creates folder dataset/poses/. After that unpack data_odometry_gray.zip, data_odometry_color.zip, data_odometry_velodyne.zip. Folder dataset/sequences/ will be created with folders 00/..21/. Each of these folders will contain: image_0/, image_1/, image_2/, image_3/, velodyne/ and files calib.txt & times.txt. These two last files will be replaced after unpacking data_odometry_calib.zip at the end.

 3. To load data run: ./opencv/build/bin/example_datasetstools_slam_kitti -p=/home/user/path_to_unpacked_folder/dataset/

slam_tumindoor
==============
.. ocv:class:: slam_tumindoor

Implements loading dataset:

_`"TUMindoor Dataset"`: http://www.navvis.lmt.ei.tum.de/dataset/

.. note:: Usage

 1. From link above download dataset files: dslr\\info\\ladybug\\pointcloud.tar.bz2 for each dataset: 11-11-28 (1st floor)\\11-12-13 (1st floor N1)\\11-12-17a (4th floor)\\11-12-17b (3rd floor)\\11-12-17c (Ground I)\\11-12-18a (Ground II)\\11-12-18b (2nd floor)

 2. Unpack them in separate folder for each dataset. dslr.tar.bz2 -> dslr/, info.tar.bz2 -> info/, ladybug.tar.bz2 -> ladybug/, pointcloud.tar.bz2 -> pointcloud/.

 3. To load each dataset run: ./opencv/build/bin/example_datasetstools_slam_tumindoor -p=/home/user/path_to_unpacked_folders/

Text Recognition
----------------

tr_chars
========
.. ocv:class:: tr_chars

Implements loading dataset:

_`"The Chars74K Dataset"`: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

.. note:: Usage

 1. From link above download dataset files: EnglishFnt\\EnglishHnd\\EnglishImg\\KannadaHnd\\KannadaImg.tgz, ListsTXT.tgz.

 2. Unpack them.

 3. Move *.m files from folder ListsTXT/ to appropriate folder. For example, English/list_English_Img.m for EnglishImg.tgz.

 4. To load data, for example "EnglishImg", run: ./opencv/build/bin/example_datasetstools_tr_chars -p=/home/user/path_to_unpacked_folder/English/

tr_svt
======
.. ocv:class:: tr_svt

Implements loading dataset:

_`"The Street View Text Dataset"`: http://vision.ucsd.edu/~kai/svt/

.. note:: Usage

 1. From link above download dataset file: svt.zip.

 2. Unpack it.

 3. To load data run: ./opencv/build/bin/example_datasetstools_tr_svt -p=/home/user/path_to_unpacked_folder/svt/svt1/

