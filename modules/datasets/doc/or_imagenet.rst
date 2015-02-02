ImageNet
========
.. ocv:class:: OR_imagenet

Implements loading dataset:

_`"ImageNet"`: http://www.image-net.org/

.. note:: Usage

 1. From link above download dataset files: ILSVRC2010_images_train.tar\\ILSVRC2010_images_test.tar\\ILSVRC2010_images_val.tar & devkit: ILSVRC2010_devkit-1.0.tar.gz (Implemented loading of 2010 dataset as only this dataset has ground truth for test data, but structure for ILSVRC2014 is similar)

 2. Unpack them to: some_folder/train/\\some_folder/test/\\some_folder/val & some_folder/ILSVRC2010_validation_ground_truth.txt\\some_folder/ILSVRC2010_test_ground_truth.txt.

 3. Create file with labels: some_folder/labels.txt, for example, using :ref:`python script <python-script>` below (each file's row format: synset,labelID,description. For example: "n07751451,18,plum").

 4. Unpack all tar files in train.

 5. To load data run: ./opencv/build/bin/example_datasets_or_imagenet -p=/home/user/some_folder/

.. _python-script:

Python script to parse meta.mat:

::

 import scipy.io
 meta_mat = scipy.io.loadmat("devkit-1.0/data/meta.mat")

 labels_dic = dict((m[0][1][0], m[0][0][0][0]-1) for m in meta_mat['synsets']
 label_names_dic = dict((m[0][1][0], m[0][2][0]) for m in meta_mat['synsets']

 for label in labels_dic.keys():
     print "{0},{1},{2}".format(label, labels_dic[label], label_names_dic[label])

**References:**

.. [ILSVRCarxiv14] Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. 2014

