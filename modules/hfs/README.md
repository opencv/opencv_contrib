##   OpenCV Hierarchical Feature Selection for Efficient Image Segmentation module

Author and maintainers: Yujun Shi (shiyujun1016@gmail.com), Yun Liu (nk12csly@mail.nankai.edu.cn).

Hierachical Feature Selection (HFS) is a real-time system for image segmentation. It was originally proposed in [1]. Here is the original project website: http://mmcheng.net/zh/hfs/

The algorithm is executed in 3 stages. In the first stage, it obtains an over-segmented image using SLIC(simple linear iterative clustering). In the last 2 stages, it iteratively merges the over-segmented  image with merging method mentioned in EGB(Efficient Graph-based Image Segmentation) and learned SVM parameters.

In our implementation, we wrapped these stages into one single member function of the interface class.

Since this module used cuda in some part of  the implementation, it has to be compiled with cuda support

For more details about the algorithm, please refer to the original paper: [1]

### usage

c++ interface:

```c++
// read a image
Mat img = imread(image_path), res;
int _h = img.rows, _w = img.cols;

// create engine
Ptr<HfsSegment> seg = HfsSegment::create( _h, _w );

// perform segmentation
// now "res" is a matrix of indices
// change the second parameter to "True" to get a rgb image for "res"
res = seg->performSegmentGpu(img, false);
```

python interface:

```python
import cv2
import numpy as np

img = cv2.imread(image_path)

# create engine
engine = cv2.hfs.HfsSegment_create(img.shape[0], img.shape[1])

# perform segmentation
# now "res" is a matrix of indices
# change the second parameter to "True" to get a rgb image for "res"
res = engine.performSegmentGpu(img, False)
```



### Reference

[1]: M. cheng, Y. Liu, Q. Hou, J. Bian, P. Torr, S. Hu, Z. Tu HFS: Hierarchical Feature Selection for Efficient Image Segmentation ECCV, Oct.2016.