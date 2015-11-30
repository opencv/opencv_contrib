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
