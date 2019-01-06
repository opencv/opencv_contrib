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


Examples
-----------------------------------------
- TODO


To Do
-----------------------------------------
- Examples
- Brief summary of different IQA algorithms
- Document the output quality maps for each algorithm
- Implement at least one no-reference IQA algorithm
- Investigate performance difference between Linux/gcc and Win32/msvc (Linux ~15-20X faster on same h/w)
- (Fast?) MS-SSIM?