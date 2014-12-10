Caltech Pedestrian Detection Benchmark
======================================
.. ocv:class:: PD_caltech

Implements loading dataset:

_`"Caltech Pedestrian Detection Benchmark"`: http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/

.. note:: First version of Caltech Pedestrian dataset loading.

 Code to unpack all frames from seq files commented as their number is huge!
 So currently load only meta information without data.

 Also ground truth isn't processed, as need to convert it from mat files first.

.. note:: Usage

 1. From link above download dataset files: set00.tar-set10.tar.

 2. Unpack them to separate folder.

 3. To load data run: ./opencv/build/bin/example_datasets_pd_caltech -p=/home/user/path_to_unpacked_folders/

**References:**

.. [Dollár12] P. Dollár, C. Wojek, B. Schiele and P. Perona. Pedestrian Detection: An Evaluation of the State of the Art. PAMI, 2012.

.. [DollárCVPR09] P. Dollár, C. Wojek, B. Schiele and P. Perona. Pedestrian Detection: A Benchmark. CVPR, 2009

