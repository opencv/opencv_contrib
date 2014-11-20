Labeled Faces in the Wild
=========================
.. ocv:class:: FR_lfw

Implements loading dataset:

_`"Labeled Faces in the Wild"`: http://vis-www.cs.umass.edu/lfw/

.. note:: Usage

 1. From link above download any dataset file: lfw.tgz\\lfwa.tar.gz\\lfw-deepfunneled.tgz\\lfw-funneled.tgz and files with pairs: 10 test splits: pairs.txt and developer train split: pairsDevTrain.txt.

 2. Unpack dataset file and place pairs.txt and pairsDevTrain.txt in created folder.

 3. To load data run: ./opencv/build/bin/example_datasets_fr_lfw -p=/home/user/path_to_unpacked_folder/lfw2/

Benchmark
"""""""""

For this dataset was implemented benchmark with accuracy: 0.623833 +- 0.005223 (train split: pairsDevTrain.txt, dataset: lfwa)

To run this benchmark execute:

.. code-block:: bash

 ./opencv/build/bin/example_datasets_fr_lfw_benchmark -p=/home/user/path_to_unpacked_folder/lfw2/

**References:**

.. [Huang07] G.B. Huang, M. Ramesh, T. Berg, and E. Learned-Miller. Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments. 2007

