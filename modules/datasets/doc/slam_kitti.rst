KITTI Vision Benchmark
======================
.. ocv:class:: SLAM_kitti

Implements loading dataset:

_`"KITTI Vision Benchmark"`: http://www.cvlibs.net/datasets/kitti/eval_odometry.php

.. note:: Usage

 1. From link above download "Odometry" dataset files: data_odometry_gray\\data_odometry_color\\data_odometry_velodyne\\data_odometry_poses\\data_odometry_calib.zip.

 2. Unpack data_odometry_poses.zip, it creates folder dataset/poses/. After that unpack data_odometry_gray.zip, data_odometry_color.zip, data_odometry_velodyne.zip. Folder dataset/sequences/ will be created with folders 00/..21/. Each of these folders will contain: image_0/, image_1/, image_2/, image_3/, velodyne/ and files calib.txt & times.txt. These two last files will be replaced after unpacking data_odometry_calib.zip at the end.

 3. To load data run: ./opencv/build/bin/example_datasets_slam_kitti -p=/home/user/path_to_unpacked_folder/dataset/

**References:**

.. [Geiger2012CVPR] Andreas Geiger and Philip Lenz and Raquel Urtasun. Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite. CVPR, 2012

.. [Geiger2013IJRR] Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun. Vision meets Robotics: The KITTI Dataset. IJRR, 2013

.. [Fritsch2013ITSC] Jannik Fritsch and Tobias Kuehnl and Andreas Geiger. A New Performance Measure and Evaluation Benchmark for Road Detection Algorithms. ITSC, 2013

