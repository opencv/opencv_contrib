The Street View Text Dataset
============================
.. ocv:class:: TR_svt

Implements loading dataset:

_`"The Street View Text Dataset"`: http://vision.ucsd.edu/~kai/svt/

.. note:: Usage

 1. From link above download dataset file: svt.zip.

 2. Unpack it.

 3. To load data run: ./opencv/build/bin/example_datasets_tr_svt -p=/home/user/path_to_unpacked_folder/svt/svt1/

Benchmark
"""""""""

For this dataset was implemented benchmark with accuracy (mean f1): 0.217

To run benchmark execute:

.. code-block:: bash

 ./opencv/build/bin/example_datasets_tr_svt_benchmark -p=/home/user/path_to_unpacked_folders/svt/svt1/

**References:**

.. [Wang11] Kai Wang, Boris Babenko and Serge Belongie. End-to-end Scene Text Recognition. ICCV, 2011

.. [Wang10] Kai Wang and Serge Belongie. Word Spotting in the Wild. ECCV, 2010

