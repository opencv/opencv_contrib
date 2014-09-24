HMDB: A Large Human Motion Database
===================================
.. ocv:class:: AR_hmdb

Implements loading dataset:

_`"HMDB: A Large Human Motion Database"`: http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/

.. note:: Usage

 1. From link above download dataset files: hmdb51_org.rar & test_train_splits.rar.

 2. Unpack them.

 3. To load data run: ./opencv/build/bin/example_datasets_ar_hmdb -p=/home/user/path_to_unpacked_folders/

Benchmark
"""""""""

For this dataset was implemented benchmark, which gives accuracy: 0.107407 (using precomputed HOG/HOF "STIP" features from site, averaging for 3 splits)

To run this benchmark execute:

.. code-block:: bash

 ./opencv/build/bin/example_datasets_ar_hmdb_benchmark -p=/home/user/path_to_unpacked_folders/

(precomputed features should be unpacked in the same folder: /home/user/path_to_unpacked_folders/hmdb51_org_stips/)

