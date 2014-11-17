HMDB: A Large Human Motion Database
===================================
.. ocv:class:: AR_hmdb

Implements loading dataset:

_`"HMDB: A Large Human Motion Database"`: http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/

.. note:: Usage

 1. From link above download dataset files: hmdb51_org.rar & test_train_splits.rar.

 2. Unpack them. Unpack all archives from directory: hmdb51_org/ and remove them.

 3. To load data run: ./opencv/build/bin/example_datasets_ar_hmdb -p=/home/user/path_to_unpacked_folders/

Benchmark
"""""""""

For this dataset was implemented benchmark with accuracy: 0.107407 (using precomputed HOG/HOF "STIP" features from site, averaging for 3 splits)

To run this benchmark execute:

.. code-block:: bash

 ./opencv/build/bin/example_datasets_ar_hmdb_benchmark -p=/home/user/path_to_unpacked_folders/

(precomputed features should be unpacked in the same folder: /home/user/path_to_unpacked_folders/hmdb51_org_stips/. Also unpack all archives from directory: hmdb51_org_stips/ and remove them.)

**References:**

.. [Kuehne11] H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre. HMDB: A Large Video Database for Human Motion Recognition. ICCV, 2011

.. [Laptev08] I. Laptev, M. Marszalek, C. Schmid, and B. Rozenfeld. Learning Realistic Human Actions From Movies. CVPR, 2008


