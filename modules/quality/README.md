//! @addtogroup quality
//! @{

Quality API, Image Quality Analysis
=======================================

Implementation of various image quality analysis (IQA) algorithms

- **Mean squared error (MSE)**
  https://en.wikipedia.org/wiki/Mean_squared_error

- **Peak signal-to-noise ratio (PSNR)**
  https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

- **Structural similarity (SSIM)**
  https://en.wikipedia.org/wiki/Structural_similarity

- **Gradient Magnitude Similarity Deviation (GMSD)**
  http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm
  In general, the GMSD algorithm should yield the best result for full-reference IQA.

- **Blind/Referenceless Image Spatial Quality Evaluation (BRISQUE)**
  http://live.ece.utexas.edu/research/Quality/nrqa.htm

Interface/Usage
-----------------------------------------
All algorithms can be accessed through the simpler static `compute` methods,
or be accessed by instance created via the static `create` methods.

Instance methods are designed to be more performant when comparing one source
file against multiple comparison files, as the algorithm-specific preprocessing on the
source file need not be repeated with each call.

For performance reaasons, it is recommended, but not required, for users of this module
to convert input images to grayscale images prior to processing.
SSIM and GMSD were originally tested by their respective researchers on grayscale uint8 images,
but this implementation will compute the values for each channel if the user desires to do so.

BRISQUE is a NR-IQA algorithm (No-Reference) which doesn't require a reference image.

Quick Start/Usage
-----------------------------------------
**C++ Implementations**

**For Full Reference IQA Algorithms (MSE, PSNR, SSIM, GMSD)**

```cpp
    #include <opencv2/quality.hpp>
    cv::Mat img1, img2; /* your cv::Mat images to compare */
    cv::Mat quality_map;  /* output quality map (optional) */
    /* compute MSE via static method */
    cv::Scalar result_static = quality::QualityMSE::compute(img1, img2, quality_map);  /* or cv::noArray() if not interested in output quality maps */
    /* alternatively, compute MSE via instance */
    cv::Ptr<quality::QualityBase> ptr = quality::QualityMSE::create(img1);
    cv::Scalar result = ptr->compute( img2 );  /* compute MSE, compare img1 vs img2 */
    ptr->getQualityMap(quality_map);  /* optionally, access output quality maps */
```

**For No Reference IQA Algorithm (BRISQUE)**

```cpp
    #include <opencv2/quality.hpp>
    cv::Mat img = cv::imread("/path/to/my_image.bmp"); // path to the image to evaluate
    cv::String model_path = "path/to/brisque_model_live.yml"; // path to the trained model
    cv::String range_path = "path/to/brisque_range_live.yml"; // path to range file
    /* compute BRISQUE quality score via static method */
    cv::Scalar result_static = quality::QualityBRISQUE::compute(img,
model_path, range_path);
    /* alternatively, compute BRISQUE via instance */
    cv::Ptr<quality::QualityBase> ptr = quality::QualityBRISQUE::create(model_path, range_path);
    cv::Scalar result = ptr->compute(img); /* computes BRISQUE score for img */
```

**Python Implementations**

**For Full Reference IQA Algorithms (MSE, PSNR, SSIM, GSMD)**

```python
    import cv2
    # read images
    img1 = cv2.imread(img1, 1) # specify img1
    img2 = cv2.imread(img2_path, 1) # specify img2_path
    # compute MSE score and quality maps via static method
    result_static, quality_map = cv2.quality.QualityMSE_compute(img1, img2)
    # compute MSE score and quality maps via Instance
    obj = cv2.quality.QualityMSE_create(img1)
    result = obj.compute(img2)
    quality_map = obj.getQualityMap()
```

**For No Reference IQA Algorithm (BRISQUE)**

```python
    import cv2
    # read image
    img = cv2.imread(img_path, 1) # mention img_path
    # compute brisque quality score via static method
    score = cv2.quality.QualityBRISQUE_compute(img, model_path,
range_path) # specify model_path and range_path
    # compute brisque quality score via instance
    # specify model_path and range_path
    obj = cv2.quality.QualityBRISQUE_create(model_path, range_path)
    score = obj.compute(img)
```

Library Design
-----------------------------------------
Each implemented algorithm shall:
- Inherit from `QualityBase`, and properly implement/override `compute`, `empty` and `clear` instance methods, along with a static `compute` method.
- Accept one `cv::Mat` or `cv::UMat` via `InputArray` for computation.  Each input `cv::Mat` or `cv::UMat` may contain one or more channels.  If the algorithm does not support multiple channels, it should be documented and an appropriate assertion should be in place.
- Return a `cv::Scalar` with per-channel computed value
- Compute result via a single, static method named `compute` and via an overridden instance method (see `compute` in `qualitybase.hpp`).
- Perform any setup and/or pre-processing of reference images in the constructor, allowing for efficient computation when comparing the reference image versus multiple comparison image(s).  No-reference algorithms should accept images for evaluation in the `compute` method.
- Optionally compute resulting quality map.  Instance `compute` method should store them in `QualityBase::_qualityMap` as the mat type defined by `QualityBase::_mat_type`, or override `QualityBase::getQualityMap`.  Static `compute` method should return the quality map in an `OutputArray` parameter.
- Document algorithm in this readme and in its respective header.  Documentation should include interpretation for the results of `compute` as well as the format of the output quality map (if supported), along with any other notable usage information.
- Implement tests of static `compute` method and instance methods using single- and multi-channel images and OpenCL enabled and disabled

To Do
-----------------------------------------
- Document the output quality maps for each algorithm
- Investigate precision loss with cv::Filter2D + UMat + CV_32F + OCL for GMSD

//! @}