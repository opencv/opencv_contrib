Training the learning-based white balance algorithm {#tutorial_xphoto_training_white_balance}
===================================================

Introduction
------------

Many traditional white balance algorithms are statistics-based, i.e. they rely on the fact that certain assumptions should hold in properly white-balanced images
like the well-known grey-world assumption. However, better results can often be achieved by leveraging large datasets of images with ground-truth
illuminants in a learning-based framework. This tutorial demonstrates how to train a learning-based white balance algorithm and evaluate the quality of the results.


How to train a model
--------------------

-#  Download a dataset for training. In this tutorial we will use the [Gehler-Shi dataset ](http://www.cs.sfu.ca/~colour/data/shi_gehler/). Extract all 568 training images
    in one folder. A file containing ground-truth illuminant values (real_illum_568..mat) is downloaded separately.

-#  We will be using a [Python script ](https://github.com/opencv/opencv_contrib/tree/master/modules/xphoto/samples/learn_color_balance.py) for training.
    Call it with the following parameters:
    @code
        python learn_color_balance.py -i <path to the folder with training images> -g <path to real_illum_568..mat> -r 0,378 --num_trees 30 --max_tree_depth 6 --num_augmented 0
    @endcode
    This should start training a model on the first 378 images (2/3 of the whole dataset). We set the size of the model to be 30 regression tree pairs per feature and limit
    the tree depth to be no more then 6. By default the resulting model will be saved to color_balance_model.yml

-#  Use the trained model by passing its path when constructing an instance of LearningBasedWB:
    @code{.cpp}
    Ptr<xphoto::LearningBasedWB> wb = xphoto::createLearningBasedWB(modelFilename);
    @endcode


How to evaluate a model
----------------------

-#  We will use a [benchmarking script ](https://github.com/opencv/opencv_contrib/tree/master/modules/xphoto/samples/color_balance_benchmark.py) to compare
    the model that we've trained with the classic grey-world algorithm on the remaining 1/3 of the dataset. Call the script with the following parameters:
    @code
        python color_balance_benchmark.py -a grayworld,learning_based:color_balance_model.yml -m <full path to folder containing the model> -i <path to the folder with training images> -g <path to real_illum_568..mat> -r 379,567 -d "img"
    @endcode

-# The objective evaluation results are stored in white_balance_eval_result.html and the resulting white-balanced images are stored in the img folder for a qualitative
   comparison of algorithms. Different algorithms are compared in terms of angular error between the estimated and ground-truth illuminants.