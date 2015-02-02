ChaLearn Looking at People
==========================
.. ocv:class:: GR_chalearn

Implements loading dataset:

_`"ChaLearn Looking at People"`: http://gesture.chalearn.org/

.. note:: Usage

 1. Follow instruction from site above, download files for dataset "Track 3: Gesture Recognition": Train1.zip-Train5.zip, Validation1.zip-Validation3.zip (Register on site: www.codalab.org and accept the terms and conditions of competition: https://www.codalab.org/competitions/991#learn_the_details There are three mirrors for downloading dataset files. When I downloaded data only mirror: "Universitat Oberta de Catalunya" works).

 2. Unpack train archives Train1.zip-Train5.zip to folder Train/, validation archives Validation1.zip-Validation3.zip to folder Validation/

 3. Unpack all archives in Train/ & Validation/ in the folders with the same names, for example: Sample0001.zip to Sample0001/

 4. To load data run: ./opencv/build/bin/example_datasets_gr_chalearn -p=/home/user/path_to_unpacked_folders/

**References:**

.. [Escalera14] S. Escalera, X. Baró, J. Gonzàlez, M.A. Bautista, M. Madadi, M. Reyes, V. Ponce-López, H.J. Escalante, J. Shotton, I. Guyon, "ChaLearn Looking at People Challenge 2014: Dataset and Results", ECCV Workshops, 2014

