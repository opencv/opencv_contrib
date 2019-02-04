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


Quick Start/Usage
-----------------------------------------

    #include <opencv2/quality.hpp>

    cv::Mat img1, img2; /* your cv::Mat images */
    std::vector<cv::Mat> quality_maps;  /* output quality map(s) (optional) */

     /* compute MSE via static method */
    cv::Scalar result_static = quality::QualityMSE::compute(img1, img2, quality_maps);  /* or cv::noArray() if not interested in output quality maps */

    /* alternatively, compute MSE via instance */
    cv::Ptr<quality::QualityBase> ptr = quality::QualityMSE::create(img1);
    cv::Scalar result = ptr->compute( img2 );  /* compute MSE, compare img1 vs img2 */
	ptr->getQualityMaps(quality_maps);	/* optionally, access output quality maps */


To Do
-----------------------------------------
- Document the output quality maps for each algorithm
- Implement at least one no-reference IQA algorithm
- Investigate performance difference between Linux/gcc and Win32/msvc (Linux ~15-20X faster on same h/w)
- Investigate precision loss with cv::Filter2D + UMat + CV_32F + OCL for GMSD
- (Fast?) MS-SSIM?
