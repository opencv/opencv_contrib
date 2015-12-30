# Fooling Code
This is the code base used to reproduce the "fooling" images in the paper:
[Nguyen A](http://anhnguyen.me), [Yosinski J](http://yosinski.com/), [Clune J](http://jeffclune.com). ["Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images"](http://arxiv.org/abs/1412.1897). In Computer Vision and Pattern Recognition (CVPR '15), IEEE, 2015.

**If you use this software in an academic article, please cite:**

    @inproceedings{nguyen2015deep,
      title={Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images},
      author={Nguyen, Anh and Yosinski, Jason and Clune, Jeff},
      booktitle={Computer Vision and Pattern Recognition (CVPR), 2015 IEEE Conference on},
      year={2015},
      organization={IEEE}
    }

For more information regarding the paper, please visit www.evolvingai.org/fooling

## Requirements
This is an installation process that requires two main software packages (included in this package):

1. Caffe: http://caffe.berkeleyvision.org
  * Our libraries installed to work with Caffe
    * Cuda 6.0
    * Boost 1.52
    * g++ 4.6
  * Use the provided scripts to download the correct version of Caffe for your experiments.
    * `./download_caffe_evolutionary_algorithm.sh` Caffe version for EA experiments
    * `./download_caffe_gradient_ascent.sh` Caffe version for gradient ascent experiments
2. Sferes: https://github.com/jbmouret/sferes2
  * Our libraries installed to work with Sferes
    * OpenCV 2.4.10
    * Boost 1.52
    * g++ 4.9 (a C++ compiler compatible with C++11 standard)
  * Use the provided script `./download_sferes.sh` to download the correct version of Sferes.

Note: These are patched versions of the two frameworks with our additional work necessary to produce the images as in the paper. They are not the same as their master branches.

## Installation
Please see the [Installation_Guide](https://github.com/anguyen8/opencv_contrib/blob/master/modules/dnns_easily_fooled/Installation_Guide.pdf) for more details.

## Usage
* An MNIST experiment (Fig. 4, 5 in the paper) can be run directly on a local machine (4-core) within a reasonable amount of time (around ~5 minutes or less for 200 generations).
* An ImageNet experiment needs to be run on a cluster environment. It took us ~4 days x 128 cores to run 5000 generations and produce 1000 images (Fig. 8 in the paper).
* [How to configure an experiment to test the evolutionary framework quickly](https://github.com/Evolving-AI-Lab/fooling/wiki/How-to-test-the-evolutionary-framework-quickly)
* To reproduce the gradient ascent fooling images (Figures 13, S3, S4, S5, S6, and S7 from the paper), see the [documentation in the caffe/ascent directory](https://github.com/anguyen8/opencv_contrib/tree/master/modules/dnns_easily_fooled/caffe/ascent). You'll need to download the correct Caffe version for this experiment using `./download_caffe_gradient_ascent.sh` script.

## Troubleshooting
1. If Sferes (Waf) can't find your CUDA and Caffe dynamic libraries
> Add obj.libpath to the wscript for exp/images to find libcudart and libcaffe or you can use LD_LIBRARY_PATH (for Linux).

2. Is there a way to monitor the progress of the experiments?
> There is a flag for printing out results (fitness + images) every N generations.
You can adjust the dump_period setting [here](https://github.com/Evolving-AI-Lab/fooling/blob/master/sferes/exp/images/dl_map_elites_images.cpp#L159).

3. Where do I get the pre-trained Caffe models?
> For AlexNet, please download on Caffe's Model Zoo.
> For LeNet, you can grab it [here](https://github.com/anguyen8/opencv_contrib/tree/master/modules/dnns_easily_fooled/model/lenet).

4. How do I run the experiments on my local machine without MPI?
> You can enable MPI or non-MPI mode by commenting/uncommenting a line [here](https://github.com/Evolving-AI-Lab/fooling/blob/master/sferes/exp/images/dl_map_elites_images_mnist.cpp#L190-L191). It can be simple eval::Eval (single-core), eval::Mpi (distributed for clusters).
