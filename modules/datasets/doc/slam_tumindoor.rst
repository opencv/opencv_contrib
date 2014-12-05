TUMindoor Dataset
=================
.. ocv:class:: SLAM_tumindoor

Implements loading dataset:

_`"TUMindoor Dataset"`: http://www.navvis.lmt.ei.tum.de/dataset/

.. note:: Usage

 1. From link above download dataset files: dslr\\info\\ladybug\\pointcloud.tar.bz2 for each dataset: 11-11-28 (1st floor)\\11-12-13 (1st floor N1)\\11-12-17a (4th floor)\\11-12-17b (3rd floor)\\11-12-17c (Ground I)\\11-12-18a (Ground II)\\11-12-18b (2nd floor)

 2. Unpack them in separate folder for each dataset. dslr.tar.bz2 -> dslr/, info.tar.bz2 -> info/, ladybug.tar.bz2 -> ladybug/, pointcloud.tar.bz2 -> pointcloud/.

 3. To load each dataset run: ./opencv/build/bin/example_datasets_slam_tumindoor -p=/home/user/path_to_unpacked_folders/

**References:**

.. [TUMindoor] R. Huitl and G. Schroth and S. Hilsenbeck and F. Schweiger and E. Steinbach. {TUM}indoor: An Extensive Image and Point Cloud Dataset for Visual Indoor Localization and Mapping. 2012

