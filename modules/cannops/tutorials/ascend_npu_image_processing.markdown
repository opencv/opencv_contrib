Ascend NPU Image Processing {#tutorial_ascend_npu_image_processing}
==========================================================

## Goal

In this guide, you will gain insights into the thread safety of Ascend operators already in use, as well as discover how to effectively employ Ascend operators for image preprocessing and understand their usage limitations.

## Preface

We provide a suite of common matrix operation operators that support the [Ascend NPU](https://www.hiascend.com/en/) within OpenCV. For user convenience, the new 'AscendMat' structure and its associated operators maintain compatibility with the 'Mat' interface in OpenCV. These operators encompass a wide range of frequently used functions, including arithmetic operations, image processing operations, and image color space conversion. All of these operators are implemented utilizing [CANN](https://www.hiascend.com/en/software/cann)(Compute Architecture of Neural Networks). The Ascend operator facilitates accelerated operations on the NPU by making use of CANN. This acceleration effect is particularly noticeable when working with larger images, such as those with dimensions like 2048x2048, 3840x2160, 7680x4320, etc.


## Instructions on Thread Safety

Our stream function is implemented by invoking the CANN operators. In the same stream, tasks are executed sequentially, while across different streams, tasks are executed in parallel. The use of event mechanisms ensures synchronization of tasks between streams, please refer to the [**Stream Management**](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/600alphaX/infacldevg/aclcppdevg/aclcppdevg_000147.html) documentation for details.


## Example for Image Preprocessing

In this section, you will discover how to use Ascend operators for image preprocessing, including functions below:

- Add
- Rotate
- Flip


### code

@add_toggle_cpp
@include opencv_contrib/modules/cannops/samples/image_processing.cpp
@end_toggle

@add_toggle_python
@include opencv_contrib/modules/cannops/samples/image_processing.py
@end_toggle

### Explanation

**Input Image**

@add_toggle_cpp
@snippet opencv_contrib/modules/cannops/samples/image_processing.cpp input_noise
@end_toggle

@add_toggle_python

```python
# Read the input image
img = cv2.imread("/path/to/img")
# Generate gauss noise that will be added into the input image
gaussNoise = np.random.normal(mean=0,sigma=25,(img.shape[0],img.shape[1],img.shape[2])).astype(img.dtype)
```

@end_toggle

**Setup CANN**

@add_toggle_cpp

@snippet opencv_contrib/modules/cannops/samples/image_processing.cpp setup

@end_toggle

@add_toggle_python

@snippet opencv_contrib/modules/cannops/samples/image_processing.py setup

@end_toggle
**Image Preprocessing Example**

@add_toggle_cpp

@snippet opencv_contrib/modules/cannops/samples/image_processing.cpp image-process

@end_toggle

@add_toggle_python

@snippet opencv_contrib/modules/cannops/samples/image_processing.py image-process

@end_toggle

**Tear down CANN**

@add_toggle_cpp
@snippet opencv_contrib/modules/cannops/samples/image_processing.cpp tear-down-cann

@end_toggle

@add_toggle_python

@snippet opencv_contrib/modules/cannops/samples/image_processing.py tear-down-cann

@end_toggle
Results

1. The original RGB input image with dimensions of (480, 640, 3):

   ![puppy](./puppy.jpg)

2. After introducing Gaussian noise, we obtain the following result:

   ![puppy_noisy](./puppy_noisy.jpg)

3. When applying the rotate operation with a rotation code of 0 (90 degrees clockwise), we obtain this result:

   ![puppy_noisy_rotate](./puppy_noisy_rotate.jpg)

4. Upon applying the flip operation with a flip code of 0 (flipping around the x-axis), we achieve the final result:

   ![puppy_processed_normalized](./puppy_processed.jpg)



## Usage Limitations

While Ascend supports most commonly used operators, there are still some limitations that need to be addressed.

- There is no strict limit on the size of the input image used for encoding; however, it depends on the available RAM size of your device.
- Please note that not all data types (dtypes) are supported by every operator. The current dtype limitations are outlined in the following table. We are actively working on addressing these limitations through automatic dtype conversion in an upcoming commit.


| Operator               | Supported Dtype                                              |
| ---------------------- | ------------------------------------------------------------ |
| multiply (with scale)  | float16,float32,int32                                        |
| divide (with scale)    | float16,float,int32,int8,uint8                               |
| bitwise add/or/xor/not | int32,int16,uint16                                           |
| flip                   | float16,float,int64,int32,int16,uint16                       |
| transpose              | float16,float,int64,int32,int16,int8,uint64,uint32,uint16,uint8,bool |
| rotate                 | float16,float,int64,int32,int16,uint16                       |
